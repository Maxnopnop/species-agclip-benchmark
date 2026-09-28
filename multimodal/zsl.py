"""Class-disjoint pilot; all learned weights are reset, only frozen features reused."""
import json,random,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from data_tools import ROOT,digest,write_json
from run import seed_all
from .models import DIMS

OUT=ROOT/'runs/zsl_v1'
VARIANTS=['baseline','region_only','ag_aux','shuffled_attributes']

def prepare_protocol():
    manifest=json.loads((ROOT/'data/expanded20/manifest.json').read_text(encoding='utf-8'))
    rng=random.Random(20260930);seen=[];partners=[]
    for pair in range(10):
        first=pair*2+rng.randrange(2);seen.append(first);partners.append(pair*2+1-(first%2))
    insects=partners[:5];plants=partners[5:];rng.shuffle(insects);rng.shuffle(plants)
    dev=sorted(insects[:2]+plants[:2]);unseen=sorted(insects[2:]+plants[2:]);seen=sorted(seen)
    rows=[]
    for c in seen:
        candidates=sorted([r for r in manifest['splits']['train'] if r['label']==c],key=lambda r:r['path'])
        random.Random(20260930+c).shuffle(candidates)
        rows.extend(dict(r,role='train_seen') for r in candidates[:20])
    for split in ['val','test']:
        for r in manifest['splits'][split]:
            role='dev_unseen' if r['label'] in dev else ('eval_unseen' if r['label'] in unseen else 'eval_seen')
            rows.append(dict(r,role=role))
    protocol=dict(version=1,class_split_seed=20260930,seen_classes=seen,dev_unseen_classes=dev,eval_unseen_classes=unseen,
        backbones=['efficientnet_b0','resnet18','convnext_tiny','vit_b_16','clip_vit_b32'],variants=VARIANTS,seeds=[42,43,44],
        shots=20,learning_rates=[.001,.0003],alignment_steps=200,comparison_steps=200,validation_interval=50,batch_size=32,
        name_alignment_weight=.25,attribute_alignment_weight=.1,
        selection='Checkpoints and LR selected by macro per-class accuracy on four dev-unseen classes, then CE. Final six unseen and ten seen evaluation classes excluded from all selection. No final refit; no GZSL calibration.',
        initialization='Original frozen pretrained feature caches only; initialize projection/fusion afresh. Never load previous all-20-class task checkpoints. One alignment run per model, LR and seed shared by all comparison variants.',
        semantics='Names and source-derived class attributes for all classes are legitimate semantic side information. Classification loss normalizes over ten seen classes only. Auxiliary targets use seen labels only. Shared attribute vocabulary is identical for every image.',
        caveats='Unseen means absent from this adaptation training. Possible ImageNet/CLIP pretraining overlap is not ruled out. Images/classes appeared in earlier project experiments; this is a reused-data exploratory class-disjoint pilot, not an independent final test or standard dataset split. Frozen encoders and fixed crops: AG-inspired, not strict paper reproduction.',
        manifest_sha256=digest(ROOT/'data/expanded20/manifest.json'),attributes_sha256=digest(ROOT/'configs/expanded_attributes.json'))
    path=ROOT/'configs/zsl_v1.json';data=dict(classes=manifest['classes'],rows=rows)
    for target,value in [(path,protocol),(ROOT/'data/zsl_v1/manifest.json',data)]:
        if target.exists() and json.loads(target.read_text(encoding='utf-8'))!=value:raise ValueError('ZSL definition changed: create a new version')
        write_json(target,value)
    assert not set(seen)&set(dev) and not set(seen)&set(unseen) and not set(dev)&set(unseen)
    assert len({r['sha256'] for r in rows})==len(rows)
    write_json(ROOT/'reports/zsl_v1/class_split.json',dict(protocol=protocol,classes=[dict(c,role='seen' if c['label'] in seen else 'dev_unseen' if c['label'] in dev else 'eval_unseen') for c in manifest['classes']],image_counts={role:sum(r['role']==role for r in rows) for role in ['train_seen','dev_unseen','eval_unseen','eval_seen']}))
    return protocol,data

def load_features(name,data,roles):
    cache=torch.load(ROOT/f'cache/expanded20/{name}_features.pt',weights_only=True)
    assert cache['metadata']['frozen'] and cache['metadata']['pretrained']
    assert cache['metadata']['manifest_sha256']==digest(ROOT/'data/expanded20/manifest.json')
    bypath={r['path']:i for i,r in enumerate(cache['rows'])};rows=[r for r in data['rows'] if r['role'] in roles]
    ids=[bypath[r['path']] for r in rows]
    for r,i in zip(rows,ids):assert r['sha256']==cache['rows'][i]['sha256'] and r['label']==int(cache['labels'][i])
    return dict(rows=rows,global_features=cache['global_features'][ids].cuda(),region_features=cache['region_features'][ids].cuda(),labels=cache['labels'][ids].cuda())

class ZSLHead(nn.Module):
    def __init__(self,name,bank,variant,seen_classes,seed):
        super().__init__();self.variant=variant;self.native=name=='clip_vit_b32'
        for key in ['class_text','attribute_text','class_attributes']:self.register_buffer(key,bank[key].clone())
        self.projection=nn.Linear(DIMS[name],512)
        if self.native:nn.init.zeros_(self.projection.weight);nn.init.zeros_(self.projection.bias)
        self.logit_scale=nn.Parameter(torch.tensor(math.log(14.2857)))
        if variant!='baseline':
            self.token_encoder=nn.Sequential(nn.Linear(1024,128),nn.GELU(),nn.Linear(128,128),nn.LayerNorm(128))
            self.query=nn.Linear(512,128);self.attention=nn.MultiheadAttention(128,4,dropout=.1,batch_first=True)
            self.output=nn.Linear(128,512);self.gate=nn.Parameter(torch.tensor(.1))
        gen=torch.Generator().manual_seed(seed+713)
        self.register_buffer('wrong_attribute_order',torch.randperm(len(self.attribute_text),generator=gen))
        # Shuffle targets among seen classes within the same broad taxonomic group only.
        wrong=torch.arange(len(self.class_text));rng=random.Random(seed+714)
        for insect in [True,False]:
            group=[c for c in seen_classes if (c<10)==insect];other=group.copy()
            while True:
                rng.shuffle(other)
                if all(a!=b for a,b in zip(group,other)):break
            wrong[group]=torch.tensor(other)
        self.register_buffer('wrong_class_order',wrong)

    def project(self,x):return F.normalize(x+self.projection(x) if self.native else self.projection(x),dim=-1)

    def forward(self,g,r,intervention=None):
        whole=self.project(g);embedding=whole
        if self.variant!='baseline':
            regions=self.project(r);similarities=regions@self.attribute_text.T
            values,indices=similarities.topk(4,dim=-1)
            if self.variant=='shuffled_attributes' or intervention=='permuted':indices=self.wrong_attribute_order[indices]
            text=(F.softmax(values*10,dim=-1).unsqueeze(-1)*self.attribute_text[indices]).sum(-2)
            if self.variant=='region_only' or intervention=='zero_text':text=torch.zeros_like(text)
            tokens=self.token_encoder(torch.cat([regions,text],dim=-1))
            context=self.attention(self.query(whole).unsqueeze(1),tokens,tokens,need_weights=False)[0].squeeze(1)
            embedding=F.normalize(whole+self.gate.tanh()*F.normalize(self.output(context),dim=-1),dim=-1)
        return self.logit_scale.exp().clamp(max=100)*embedding@self.class_text.T,embedding

def remap_labels(labels,candidates):
    candidates=torch.tensor(candidates,device=labels.device);match=labels[:,None]==candidates[None,:]
    if not bool(match.any(-1).all()):raise ValueError('A label is outside the permitted candidate set')
    return match.long().argmax(-1)

def objective(model,cache,ids,protocol):
    labels=cache['labels'][ids]
    logits,embedding=model(cache['global_features'][ids],cache['region_features'][ids])
    y=remap_labels(labels,protocol['seen_classes'])
    ce=F.cross_entropy(logits[:,protocol['seen_classes']],y)
    name_loss=(1-(embedding*model.class_text[labels]).sum(-1)).mean()
    loss=ce+protocol['name_alignment_weight']*name_loss
    if model.variant in ('ag_aux','shuffled_attributes'):
        target_labels=model.wrong_class_order[labels] if model.variant=='shuffled_attributes' else labels
        loss=loss+protocol['attribute_alignment_weight']*(1-(embedding*model.class_attributes[target_labels]).sum(-1)).mean()
    return loss

@torch.no_grad()
def evaluate(model,cache,ids,candidates,intervention=None):
    model.eval();logits=[]
    for start in range(0,len(ids),64):
        ix=ids[start:start+64];scores,_=model(cache['global_features'][ix],cache['region_features'][ix],intervention)
        logits.append(scores[:,candidates].cpu())
    logits=torch.cat(logits);truth=cache['labels'][ids].cpu();local=remap_labels(truth,candidates)
    pred=torch.tensor(candidates)[logits.argmax(-1)];accs=[float((pred[truth==c]==c).float().mean()) for c in truth.unique().tolist()]
    return dict(accuracy=float((pred==truth).float().mean()),per_class_accuracy=float(np.mean(accs)),loss=float(F.cross_entropy(logits,local))),dict(logits=logits,labels=truth,predictions=pred,candidates=candidates,paths=[cache['rows'][i]['path'] for i in ids])

def better(metric,best):return best is None or (metric['per_class_accuracy'],-metric['loss'])>(best['per_class_accuracy'],-best['loss'])

def train_steps(model,cache,protocol,seed,lr,steps,select):
    ti=[i for i,r in enumerate(cache['rows']) if r['role']=='train_seen'];vi=[i for i,r in enumerate(cache['rows']) if r['role']=='dev_unseen']
    assert {r['role'] for r in cache['rows']}=={'train_seen','dev_unseen'}
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=steps)
    rng=random.Random(seed);history=[];best=None;first_gradient=None
    for step in range(1,steps+1):
        ids=rng.sample(ti,protocol['batch_size']);model.train();optimizer.zero_grad(set_to_none=True)
        loss=objective(model,cache,ids,protocol)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward()
        if step==1 and model.variant!='baseline':
            first_gradient=sum(float(p.grad.abs().sum()) for p in model.token_encoder.parameters() if p.grad is not None)
            if first_gradient<=0:raise ValueError('Dead fusion gradient')
        nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();scheduler.step()
        if select and step%protocol['validation_interval']==0:
            metric,_=evaluate(model,cache,vi,protocol['dev_unseen_classes']);history.append(dict(step=step,training_loss=float(loss.detach()),**metric))
            if better(metric,best):best=metric;best_step=step;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if not select:return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state)
    return state,dict(validation=best,best_step=best_step,first_fusion_gradient_l1=first_gradient,history=history)

def train_matrix():
    protocol,data=prepare_protocol();bank=torch.load(ROOT/'cache/expanded20/text_bank.pt',weights_only=True)
    assert bank['manifest_sha256']==protocol['manifest_sha256'] and bank['attributes_sha256']==protocol['attributes_sha256']
    results=[]
    for name in protocol['backbones']:
        cache=load_features(name,data,['train_seen','dev_unseen'])
        for lr in protocol['learning_rates']:
            for seed in protocol['seeds']:
                common=dict(backbone=name,lr=lr,seed=seed,protocol=protocol,implementation_sha256=digest(__file__),cache_sha256=digest(ROOT/f'cache/expanded20/{name}_features.pt'))
                base=OUT/f'{name}_seed{seed}_lr{lr:g}';config_path=base/'config.json'
                if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8'))!=common:raise ValueError('ZSL run configuration changed')
                write_json(config_path,common);alignment=base/'alignment.pt'
                if not alignment.exists():
                    seed_all(seed);head=ZSLHead(name,bank,'baseline',protocol['seen_classes'],seed).cuda()
                    state=train_steps(head,cache,protocol,seed,lr,protocol['alignment_steps'],False);torch.save(state,alignment)
                initial=torch.load(alignment,weights_only=True)
                for variant in protocol['variants']:
                    folder=base/variant
                    write_json(OUT/'status.json',dict(status='training',backbone=name,seed=seed,lr=lr,variant=variant,completed=len(results)))
                    if (folder/'result.json').exists():result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
                    else:
                        seed_all(seed);head=ZSLHead(name,bank,variant,protocol['seen_classes'],seed).cuda()
                        head.load_state_dict({k:v for k,v in initial.items() if k.startswith(('projection.','logit_scale'))},strict=False)
                        state,result=train_steps(head,cache,protocol,seed,lr,protocol['comparison_steps'],True)
                        result.update(backbone=name,lr=lr,seed=seed,variant=variant)
                        folder.mkdir(parents=True,exist_ok=True);torch.save(dict(state_dict=state,config=common,variant=variant,classes=data['classes']),folder/'best.pt');write_json(folder/'result.json',result)
                        print(f'{name} {variant} seed{seed} lr{lr:g}: dev unseen={result["validation"]["per_class_accuracy"]:.3f}',flush=True)
                    results.append(result)
        del cache;torch.cuda.empty_cache()
    assert len(results)==120
    locked=[]
    for name in protocol['backbones']:
        for variant in protocol['variants']:
            groups=[[r for r in results if r['backbone']==name and r['variant']==variant and r['lr']==lr] for lr in protocol['learning_rates']]
            group=max(groups,key=lambda g:(np.mean([r['validation']['per_class_accuracy'] for r in g]),-np.mean([r['validation']['loss'] for r in g])))
            locked.append(dict(backbone=name,variant=variant,lr=group[0]['lr'],validation_mean=float(np.mean([r['validation']['per_class_accuracy'] for r in group]))))
    lock=dict(implementation_sha256=digest(__file__),protocol_sha256=digest(ROOT/'configs/zsl_v1.json'),selected=locked)
    path=OUT/'selection_locked.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=lock:raise ValueError('Cannot change locked ZSL selection')
    write_json(path,lock);write_json(OUT/'development_results.json',results);write_json(OUT/'status.json',dict(status='selection_locked',completed=len(results)))

def evaluate_locked():
    protocol,data=prepare_protocol();lock=json.loads((OUT/'selection_locked.json').read_text(encoding='utf-8'))
    assert lock['implementation_sha256']==digest(__file__) and lock['protocol_sha256']==digest(ROOT/'configs/zsl_v1.json')
    bank=torch.load(ROOT/'cache/expanded20/text_bank.pt',weights_only=True);results=[]
    for name in protocol['backbones']:
        cache=load_features(name,data,['eval_unseen','eval_seen']);ui=[i for i,r in enumerate(cache['rows']) if r['role']=='eval_unseen'];si=[i for i,r in enumerate(cache['rows']) if r['role']=='eval_seen']
        for selected in [s for s in lock['selected'] if s['backbone']==name]:
            lr=selected['lr'];variant=selected['variant']
            for seed in protocol['seeds']:
                folder=OUT/f'{name}_seed{seed}_lr{lr:g}'/variant;checkpoint=torch.load(folder/'best.pt',weights_only=True)
                head=ZSLHead(name,bank,variant,protocol['seen_classes'],seed).cuda();head.load_state_dict(checkpoint['state_dict'])
                zsl,pred=evaluate(head,cache,ui,protocol['eval_unseen_classes'])
                candidates=sorted(protocol['seen_classes']+protocol['eval_unseen_classes']);u,up=evaluate(head,cache,ui,candidates);s,sp=evaluate(head,cache,si,candidates)
                h=2*u['per_class_accuracy']*s['per_class_accuracy']/max(u['per_class_accuracy']+s['per_class_accuracy'],1e-12)
                interventions={}
                if variant=='ag_aux':
                    for mode in ['zero_text','permuted']:
                        metric,p=evaluate(head,cache,ui,protocol['eval_unseen_classes'],mode)
                        interventions[mode]=dict(**metric,flips=int((p['predictions']!=pred['predictions']).sum()))
                result=dict(backbone=name,variant=variant,seed=seed,lr=lr,zsl=zsl,gzsl_unseen=u['per_class_accuracy'],gzsl_seen=s['per_class_accuracy'],gzsl_h=h,interventions=interventions)
                torch.save(dict(zsl=pred,gzsl_unseen=up,gzsl_seen=sp),folder/'final_predictions.pt');write_json(folder/'final_result.json',result);results.append(result)
        del cache;torch.cuda.empty_cache()
    assert len(results)==60
    write_json(OUT/'final_results.json',results);write_json(OUT/'status.json',dict(status='complete',development_cells=120,final_evaluations=60))
