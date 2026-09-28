"""Validation-only, architecture-matched attribute and trainable-tail diagnostics."""
import copy
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from data_tools import ROOT, digest, write_json
from run import seed_all, metrics_from_confusion
from .models import load_backbone
from .expanded import ExpandedHead, is_better
from .experiment import split_indices
from prepare_expanded import five_crops


def derangement(n,seed):
    rng=random.Random(seed);p=list(range(n))
    while True:
        rng.shuffle(p)
        if all(i!=v for i,v in enumerate(p)):return torch.tensor(p)


class ControlledHead(ExpandedHead):
    def __init__(self,name,bank,mode,seed=42,variant=None):
        super().__init__(name,bank,variant or ('baseline' if mode=='baseline' else 'ag_attention'))
        self.mode=mode
        self.register_buffer('permutation',derangement(len(self.attribute_text),seed))

    def retrieve(self,regions):
        values,ids=(regions@self.attribute_text.T).topk(4,dim=-1)
        vectors=self.attribute_text[self.permutation[ids] if self.mode=='permuted' else ids]
        text=(F.softmax(values*10,dim=-1)[...,None]*vectors).sum(-2)
        return torch.zeros_like(text) if self.mode=='zero' else text

    def forward(self,g,r):
        whole=self.project(g);embedding=whole
        if self.variant!='baseline':
            regions=self.project(r);text=self.retrieve(regions)
            tokens=self.token_encoder(torch.cat([regions,text],dim=-1))
            context=tokens.mean(1) if self.variant=='ag_mean' else self.attention(
                self.query(whole)[:,None],tokens,tokens,need_weights=False)[0][:,0]
            embedding=F.normalize(whole+self.gate.tanh()*F.normalize(self.output(context),dim=-1),dim=-1)
        return self.logit_scale.exp().clamp(max=100)*embedding@self.class_text.T,embedding


class Tail(nn.Module):
    def __init__(self,name,encoder):
        super().__init__();self.name=name
        if name=='efficientnet_b0':self.blocks=encoder.features[-2:]
        else:
            assert encoder.attn_pool is None and encoder.pool_type=='tok'
            self.block=encoder.transformer.resblocks[-1]
            self.ln_post=encoder.ln_post
            self.proj=encoder.proj

    def forward(self,x):
        if self.name=='efficientnet_b0':x=self.blocks(x).mean((-2,-1))
        else:x=self.ln_post(self.block(x)[:,0])@self.proj
        return F.normalize(x.float(),dim=-1)


def load_feature_cache(name):
    original=torch.load(ROOT/f'cache/expanded20/{name}_features.pt',weights_only=True)
    ids=[i for i,r in enumerate(original['rows']) if r['split'] in ('train','val')]
    out={k:original[k] for k in ('metadata','classes')}
    out['rows']=[original['rows'][i] for i in ids]
    for k in ('global_features','region_features','labels'):out[k]=original[k][ids]
    assert all(r['split']!='test' for r in out['rows'])
    return out


def prepare_prefix(name):
    target=ROOT/f'cache/diagnostics_v1/{name}_prefix.pt'
    cache=load_feature_cache(name)
    protocol=dict(manifest_sha256=cache['metadata']['manifest_sha256'],dtype='float32',
                  implementation_sha256=digest(__file__),paths=[r['path'] for r in cache['rows']])
    if target.exists():
        old=torch.load(target,weights_only=True)
        if old['metadata']!=protocol:raise ValueError('Prefix cache changed; version before rerun')
        return old
    seed_all(42);encoder,transform,_=load_backbone(name);encoder=encoder.cuda().eval()
    tail=Tail(name,encoder).eval();captured=[]
    boundary=encoder.features[-2] if name=='efficientnet_b0' else encoder.transformer.resblocks[-1]
    hook=boundary.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    manifest=json.loads((ROOT/'data/expanded20/manifest.json').read_text());values=[];full=[];max_error=0.
    try:
        with torch.no_grad():
            for start in range(0,len(cache['rows']),4):
                tensors=[]
                for row in cache['rows'][start:start+4]:
                    with Image.open(Path(manifest['image_root'])/row['path']) as image:image=image.convert('RGB')
                    tensors.extend(transform(v) for v in [image,*five_crops(image)])
                for b in range(0,len(tensors),12):
                    x=torch.stack(tensors[b:b+12]).cuda();captured.clear()
                    reference=F.normalize(encoder(x).float(),dim=-1)
                    prefix=captured[0];reconstructed=tail(prefix)
                    max_error=max(max_error,float((reference-reconstructed).abs().max()))
                    values.append(prefix.cpu());full.append(reference.cpu())
                if start%100==0:print(f'prefix {name}: {start}/{len(cache["rows"])}',flush=True)
    finally:hook.remove()
    if max_error>1e-5:raise ValueError(f'Tail decomposition mismatch: {max_error}')
    prefix=torch.cat(values);features=torch.cat(full).view(len(cache['rows']),6,-1)
    saved=dict(metadata=protocol,rows=cache['rows'],classes=cache['classes'],labels=cache['labels'],
               prefix=prefix.view(len(cache['rows']),6,*prefix.shape[1:]),
               global_features=features[:,0],region_features=features[:,1:],
               decomposition_max_absolute_error=max_error)
    target.parent.mkdir(exist_ok=True,parents=True);torch.save(saved,target)
    del encoder,tail,values,full;torch.cuda.empty_cache()
    return saved


def forward(head,tail,cache,idx,regime):
    if regime=='frozen':
        g=cache['global_features'][idx].cuda();r=cache['region_features'][idx].cuda()
    else:
        x=cache['prefix'][idx].cuda();views=1 if head.variant=='baseline' else 6
        x=x[:,:views];f=tail(x.reshape(-1,*x.shape[2:])).reshape(len(idx),views,-1)
        g=f[:,0];r=f[:,1:] if views>1 else f.new_zeros(len(idx),5,f.shape[-1])
    return head(g,r)


@torch.no_grad()
def evaluate(head,tail,cache,idx,regime):
    head.eval();tail.eval();probs=[];total_loss=0.
    for start in range(0,len(idx),16):
        batch=idx[start:start+16];logits,_=forward(head,tail,cache,batch,regime)
        labels=cache['labels'][batch].cuda();total_loss+=float(F.cross_entropy(logits,labels,reduction='sum'))
        probs.append(logits.softmax(-1).cpu())
    probs=torch.cat(probs);labels=cache['labels'][idx];confusion=np.zeros((len(cache['classes']),)*2,dtype=np.int64)
    np.add.at(confusion,(labels.numpy(),probs.argmax(-1).numpy()),1)
    return dict(metrics_from_confusion(confusion),loss=total_loss/len(idx)),probs


def run_cell(name,cache,tail_initial,bank,protocol,regime,mode,seed,lr):
    root=ROOT/'runs/diagnostics_v1';folder=root/f'{name}_{regime}_{mode}_seed{seed}_lr{lr:g}'
    config=dict(backbone=name,regime=regime,mode=mode,seed=seed,lr=lr,protocol=protocol,
                implementation_sha256=digest(__file__),manifest_sha256=cache['metadata']['manifest_sha256'],
                attributes_sha256=bank['attributes_sha256'])
    if (folder/'config.json').exists() and json.loads((folder/'config.json').read_text())!=config:
        raise ValueError('Diagnostic config differs from saved run')
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    write_json(folder/'config.json',config);seed_all(seed)
    head=ControlledHead(name,bank,mode,seed).cuda()
    alignment=torch.load(ROOT/f'runs/expanded20/{name}_shots10_seed{seed}/alignment/best.pt',weights_only=True)
    head.load_alignment(alignment['state_dict'])
    tail=copy.deepcopy(tail_initial).cuda().eval();tail.requires_grad_(regime=='tail')
    initial={k:p.detach().cpu().clone() for k,p in tail.named_parameters()}
    groups=[dict(params=head.parameters(),lr=lr)]
    if regime=='tail':groups.append(dict(params=tail.parameters(),lr=lr*protocol['tail_lr_ratio']))
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=protocol['updates'])
    ti=split_indices(cache,'train',protocol['shots'],seed);vi=split_indices(cache,'val')
    assert all(cache['rows'][i]['split']=='train' for i in ti)
    assert all(cache['rows'][i]['split']=='val' for i in vi)
    rng=random.Random(seed);order=[];best=None;history=[];tail_gradient=None
    for step in range(1,protocol['updates']+1):
        if not order:order=ti.copy();rng.shuffle(order)
        batch=order[:protocol['batch_size']];order=order[protocol['batch_size']:]
        head.train();tail.eval();optimizer.zero_grad(set_to_none=True)
        logits,_=forward(head,tail,cache,batch,regime);loss=F.cross_entropy(logits,cache['labels'][batch].cuda())
        if not torch.isfinite(loss):raise ValueError('Nonfinite diagnostic loss')
        loss.backward()
        if step==1:
            tail_gradient=sum(float(p.grad.abs().sum()) for p in tail.parameters() if p.grad is not None)
            if regime=='tail' and tail_gradient<=0:raise ValueError('Tail received no gradient')
        nn.utils.clip_grad_norm_(list(head.parameters())+list(tail.parameters()),5)
        optimizer.step();scheduler.step()
        if step in protocol['validation_steps']:
            metric,probs=evaluate(head,tail,cache,vi,regime);history.append(dict(step=step,train_batch_loss=float(loss),**metric))
            if is_better(metric,best):
                best=metric;best_step=step;best_head={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
                best_tail={k:v.detach().cpu().clone() for k,v in tail.state_dict().items()};best_probs=probs
    head.load_state_dict(best_head);tail.load_state_dict(best_tail)
    train_metrics,_=evaluate(head,tail,cache,ti,regime)
    change=sum(float((p.detach().cpu()-initial[k]).square().sum()) for k,p in tail.named_parameters())**.5
    if regime=='frozen' and change!=0:raise ValueError('Frozen tail changed')
    result=dict(backbone=name,regime=regime,mode=mode,seed=seed,lr=lr,best_step=best_step,
                validation=best,training=train_metrics,tail_first_gradient_l1=tail_gradient,
                tail_parameter_l2_change=change,gate=None if mode=='baseline' else float(head.gate.detach()),
                head_parameters=sum(p.numel() for p in head.parameters()),
                tail_trainable_parameters=sum(p.numel() for p in tail.parameters() if p.requires_grad))
    torch.save(dict(head=best_head,tail=best_tail if regime=='tail' else None,config=config),folder/'best.pt')
    torch.save(dict(probabilities=best_probs,paths=[cache['rows'][i]['path'] for i in vi],labels=cache['labels'][vi]),folder/'validation_predictions.pt')
    write_json(folder/'history.json',history);write_json(folder/'result.json',result)
    print(f'{name} {regime} {mode} s{seed} lr{lr}: val={best["top1_accuracy"]:.3f} train={train_metrics["top1_accuracy"]:.3f}',flush=True)
    del head,tail,optimizer;torch.cuda.empty_cache();return result
