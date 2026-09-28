"""Versioned, fixed-budget, multi-crop attribute-guided development experiments."""
import csv
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from data_tools import ROOT,digest,write_json
from run import seed_all,metrics_from_confusion
from .models import DIMS,load_clip
from .experiment import split_indices
from .preparation import release

VARIANTS=('baseline','region_only','ag_mean','ag_attention','ag_aux')


def prepare_text():
    manifest_path=ROOT/'data/expanded20/manifest.json';attrpath=ROOT/'configs/expanded_attributes.json'
    manifest=json.loads(manifest_path.read_text());attrs=json.loads(attrpath.read_text())
    target=ROOT/'cache/expanded20/text_bank.pt'
    hashes=dict(manifest_sha256=digest(manifest_path),attributes_sha256=digest(attrpath))
    if target.exists():
        bank=torch.load(target,weights_only=True)
        if any(bank[k]!=v for k,v in hashes.items()):raise ValueError('Text cache mismatch')
        return bank
    entries={s['category_id']:s for s in attrs['species']}
    if not attrs.get('species_descriptions_reviewed') or any(c['id'] not in entries for c in manifest['classes']):
        raise ValueError('Source descriptions must cover all selected species')
    clip,_,tokenizer=load_clip();clip=clip.to('cuda').eval()
    def encode(prompts):
        with torch.inference_mode():
            return torch.cat([clip.encode_text(tokenizer(prompts[i:i+32]).to('cuda'),normalize=True).float().cpu()
                              for i in range(0,len(prompts),32)])
    names=[f'a photo of {c["name"]}, {c.get("common_name") or c["name"]}.' for c in manifest['classes']]
    text=encode(attrs['prompts']);class_attributes=[]
    for c in manifest['classes']:
        ids=entries[c['id']]['attribute_ids']
        if not ids or min(ids)<0 or max(ids)>=len(text):raise ValueError('Invalid attribute IDs')
        class_attributes.append(F.normalize(text[ids].mean(0),dim=0))
    bank=dict(**hashes,class_text=encode(names),attribute_text=text,class_attributes=torch.stack(class_attributes),
              class_prompts=names,attribute_prompts=attrs['prompts'])
    torch.save(bank,target);del clip;release();return bank


class ExpandedHead(nn.Module):
    def __init__(self,backbone,bank,variant='baseline'):
        super().__init__();self.variant=variant;self.backbone=backbone
        for key in ('class_text','attribute_text','class_attributes'):self.register_buffer(key,bank[key].float().clone())
        self.projection=nn.Linear(DIMS[backbone],512);self.native=backbone=='clip_vit_b32'
        if self.native:nn.init.zeros_(self.projection.weight);nn.init.zeros_(self.projection.bias)
        self.logit_scale=nn.Parameter(torch.tensor(math.log(14.2857)))
        if variant!='baseline':
            self.token_encoder=nn.Sequential(nn.Linear(1024,128),nn.GELU(),nn.Linear(128,128),nn.LayerNorm(128))
            self.output=nn.Linear(128,512)
            self.gate=nn.Parameter(torch.tensor(.1))
            if variant!='ag_mean':
                self.query=nn.Linear(512,128)
                self.attention=nn.MultiheadAttention(128,4,dropout=.1,batch_first=True)

    def project(self,x):
        return F.normalize(x+self.projection(x) if self.native else self.projection(x),dim=-1)

    def forward(self,g,r):
        global_=self.project(g);embedding=global_
        if self.variant!='baseline':
            regions=self.project(r)
            if self.variant=='region_only':
                text=torch.zeros_like(regions)
            else:
                # Every crop scores the same source-derived vocabulary. Ground-truth
                # labels and species-specific attribute subsets are never inputs here.
                similarities=regions@self.attribute_text.T
                values,indices=similarities.topk(min(4,len(self.attribute_text)),dim=-1)
                text=(F.softmax(values*10,dim=-1).unsqueeze(-1)*self.attribute_text[indices]).sum(-2)
            tokens=self.token_encoder(torch.cat([regions,text],dim=-1))
            if self.variant=='ag_mean':context=tokens.mean(1)
            else:context=self.attention(self.query(global_).unsqueeze(1),tokens,tokens,need_weights=False)[0].squeeze(1)
            # Normalize both terms so the nonzero gate has a controlled feature scale.
            embedding=F.normalize(global_+self.gate.tanh()*F.normalize(self.output(context),dim=-1),dim=-1)
        return self.logit_scale.exp().clamp(max=100)*embedding@self.class_text.T,embedding

    def load_alignment(self,state):
        self.load_state_dict({k:v for k,v in state.items() if k.startswith(('projection.','logit_scale'))},strict=False)


def is_better(metrics,best):
    return best is None or metrics['macro_f1']>best['macro_f1']+1e-12 or (
        abs(metrics['macro_f1']-best['macro_f1'])<=1e-12 and metrics['loss']<best['loss']-1e-10)


@torch.inference_mode()
def evaluate(model,cache,idx):
    model.eval();scores=[];loss=0
    for start in range(0,len(idx),64):
        batch=idx[start:start+64];logits,_=model(cache['global_features'][batch],cache['region_features'][batch])
        loss+=float(F.cross_entropy(logits,cache['labels'][batch],reduction='sum'))
        scores.append(logits.softmax(-1).cpu())
    scores=torch.cat(scores);y=cache['labels'][idx].cpu().numpy();pred=scores.argmax(1).numpy()
    confusion=np.zeros((len(cache['classes']),len(cache['classes'])),dtype=np.int64)
    np.add.at(confusion,(y,pred),1)
    metrics=dict(metrics_from_confusion(confusion),loss=loss/len(idx))
    return metrics,scores,y,confusion


def train(model,cache,train_idx,val_idx,steps,seed,path,aux_weight=.1):
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=steps)
    rng=random.Random(seed);order=[];best=None;history=[];first_gradient=None
    initial={k:p.detach().clone() for k,p in model.named_parameters() if not k.startswith(('projection.','logit_scale'))}
    for step in range(1,steps+1):
        if not order:order=train_idx.copy();rng.shuffle(order)
        batch=order[:32];order=order[32:];model.train()
        logits,embedding=model(cache['global_features'][batch],cache['region_features'][batch])
        labels=cache['labels'][batch];ce=F.cross_entropy(logits,labels);loss=ce
        if model.variant=='ag_aux':
            loss=loss+aux_weight*(1-(embedding*model.class_attributes[labels]).sum(-1)).mean()
        if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        if step==1 and model.variant!='baseline':
            first_gradient=sum(float(p.grad.detach().abs().sum()) for k,p in model.named_parameters()
                               if k.startswith('token_encoder') and p.grad is not None)
            if first_gradient==0:raise RuntimeError('Attribute/crop branch has zero first-step gradient')
        nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();scheduler.step()
        if step%20==0 or step==steps:
            metrics,_,_,_=evaluate(model,cache,val_idx)
            history.append(dict(step=step,train_loss=float(loss.detach()),**metrics))
            if is_better(metrics,best):
                best=metrics;best_step=step;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state)
    path.mkdir(parents=True,exist_ok=True)
    with (path/'history.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=history[0].keys());writer.writeheader();writer.writerows(history)
    diagnostic=dict(first_step_branch_gradient_l1=first_gradient,best_step=best_step)
    if initial:
        diagnostic['branch_parameter_l2_change']=sum(float((p.detach()-initial[k]).square().sum())
            for k,p in model.named_parameters() if k in initial)**.5
        diagnostic['gate']=float(model.gate.detach())
    return state,best,diagnostic


def run_one(name,bank,protocol,shots,seed,device='cuda'):
    output=ROOT/f'runs/expanded20/{name}_shots{shots}_seed{seed}'
    cache_path=ROOT/f'cache/expanded20/{name}_features.pt';cache=torch.load(cache_path,weights_only=True)
    if cache['metadata']['manifest_sha256']!=bank['manifest_sha256']:raise ValueError('Manifest mismatch')
    config=dict(backbone=name,shots=shots,seed=seed,protocol=protocol,
        manifest_sha256=bank['manifest_sha256'],attributes_sha256=bank['attributes_sha256'],cache_sha256=digest(cache_path),
        implementation_sha256=digest(__file__))
    if (output/'config.json').exists() and json.loads((output/'config.json').read_text())!=config:
        raise ValueError('Expanded run differs from stored configuration; use versioned directories.')
    if (output/'validation.json').exists():return
    write_json(output/'config.json',config)
    for key in ('global_features','region_features','labels'):cache[key]=cache[key].to(device)
    train_idx=split_indices(cache,'train',shots,seed);val_idx=split_indices(cache,'val')
    seed_all(seed);alignment=ExpandedHead(name,bank).to(device)
    state,metric,diag=train(alignment,cache,train_idx,val_idx,protocol['alignment_steps'],seed,output/'alignment')
    torch.save(dict(state_dict=state,validation=metric,diagnostic=diag),output/'alignment/best.pt')
    validation={}
    for variant in protocol['variants']:
        seed_all(seed);head=ExpandedHead(name,bank,variant).to(device);head.load_alignment(state)
        trained,metric,diag=train(head,cache,train_idx,val_idx,protocol['comparison_steps'],seed,output/variant,
                                  protocol['aux_weight'])
        torch.save(dict(state_dict=trained,config=config,variant=variant,classes=cache['classes'],
                        validation=metric,diagnostic=diag),output/variant/'best.pt')
        validation[variant]=dict(**metric,**diag)
        print(f'{name} {shots}-shot seed{seed} {variant}: val F1={metric["macro_f1"]:.3f}, step={diag["best_step"]}',flush=True)
    write_json(output/'validation.json',validation)
    del cache,alignment,head;release()


def select_variants(protocol):
    selected=[]
    for name in protocol['backbones']:
        for shots in protocol['shots']:
            tables=[json.loads((ROOT/f'runs/expanded20/{name}_shots{shots}_seed{s}/validation.json').read_text())
                    for s in protocol['seeds']]
            ranked=[]
            for variant in ('ag_mean','ag_attention','ag_aux'):
                ranked.append(dict(variant=variant,macro_f1=float(np.mean([t[variant]['macro_f1'] for t in tables])),
                                   loss=float(np.mean([t[variant]['loss'] for t in tables]))))
            ranked.sort(key=lambda r:(-r['macro_f1'],r['loss'],r['variant']))
            selected.append(dict(backbone=name,shots=shots,selected_ag=ranked[0]['variant'],validation_candidates=ranked))
    path=ROOT/'runs/expanded20/selection_locked.json'
    if path.exists() and json.loads(path.read_text())!=selected:raise ValueError('Cannot change locked test selection')
    write_json(path,selected);return selected


def evaluate_test(protocol):
    # Every model and variant is trained; validation choice is frozen before test labels are evaluated.
    selected=select_variants(protocol);rows=[]
    for item in selected:
        name,shots=item['backbone'],item['shots'];cache=torch.load(ROOT/f'cache/expanded20/{name}_features.pt',weights_only=True)
        idx=split_indices(cache,'test')
        for seed in protocol['seeds']:
            for variant in protocol['variants']:
                folder=ROOT/f'runs/expanded20/{name}_shots{shots}_seed{seed}/{variant}'
                saved=torch.load(folder/'best.pt',weights_only=True,map_location='cpu')
                head=ExpandedHead(name,saved['state_dict'],variant);head.load_state_dict(saved['state_dict'])
                metrics,probabilities,y,confusion=evaluate(head,cache,idx)
                record=dict(backbone=name,shots=shots,seed=seed,variant=variant,images=len(idx),**metrics)
                write_json(folder/'test_metrics.json',record)
                torch.save(dict(probabilities=probabilities,labels=torch.tensor(y),paths=[cache['rows'][i]['path'] for i in idx]),folder/'test_predictions.pt')
                np.savetxt(folder/'test_confusion.csv',confusion,delimiter=',',fmt='%d');rows.append(record)
    write_json(ROOT/'runs/expanded20/test_results.json',rows)
    return rows
