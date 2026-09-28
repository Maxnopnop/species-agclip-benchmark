"""Explicit visible-attribute learning and region alignment on frozen VLM features."""
import json,math,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from data_tools import ROOT,digest,write_json
from run import seed_all,metrics_from_confusion
from .expanded import is_better


class Adapter(nn.Module):
    def __init__(self,dim):
        super().__init__();self.down=nn.Linear(dim,64);self.up=nn.Linear(64,dim)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
    def forward(self,x):return F.normalize(x+self.up(F.gelu(self.down(x))),dim=-1)


class VisibleHead(nn.Module):
    def __init__(self,cache,variant):
        super().__init__();self.variant=variant;dim=cache['class_text'].shape[-1]
        self.register_buffer('class_text',cache['class_text'].clone());self.register_buffer('attribute_text',cache['attribute_text'].clone())
        self.global_adapter=Adapter(dim);self.patch_adapter=Adapter(dim)
        self.logit_scale=nn.Parameter(torch.tensor(math.log(20.)))
        self.attribute_scale=nn.Parameter(torch.tensor(math.log(10.)))
        self.attribute_bias=nn.Parameter(torch.full((len(self.attribute_text),),-2.))
        self.attribute_classifier=nn.Linear(len(self.attribute_text),len(self.class_text),bias=False)
        nn.init.normal_(self.attribute_classifier.weight,std=.01)

    def forward(self,whole,patches,valid,intervention=None):
        g=self.global_adapter(whole);p=self.patch_adapter(patches)
        similarities=torch.einsum('bpd,ad->bap',p,self.attribute_text)
        attention=F.softmax((similarities*10).masked_fill(~valid[:,None,:],-1e4),dim=-1)
        pooled=(similarities*attention).sum(-1)
        attributes=self.attribute_scale.exp().clamp(max=50)*pooled+self.attribute_bias
        concepts=2*attributes.sigmoid()-1
        if intervention=='zero':concepts=torch.zeros_like(concepts)
        elif intervention=='permuted':concepts=concepts.roll(5,dims=-1)
        whole_logits=self.logit_scale.exp().clamp(max=100)*g@self.class_text.T
        contribution=self.attribute_classifier(concepts)
        logits=whole_logits if self.variant=='baseline' else whole_logits+contribution
        return dict(logits=logits,global_logits=whole_logits,attribute_logits=attributes,
                    attention=attention,attribute_contribution=contribution)


def masked_attribute_loss(logits,targets,pos_weight=None):
    known=targets>=0
    values=F.binary_cross_entropy_with_logits(logits,targets.clamp_min(0),pos_weight=pos_weight,reduction='none')
    return (values*known).sum()/known.sum().clamp_min(1)


def region_loss(attention,targets,evidence,valid):
    mask=evidence&valid[:,None,:];positive=(targets==1)&mask.any(-1)
    distribution=mask.float()/mask.sum(-1,keepdim=True).clamp_min(1)
    kl=(distribution*(distribution.clamp_min(1e-8).log()-attention.clamp_min(1e-8).log())).sum(-1)
    return (kl*positive).sum()/positive.sum().clamp_min(1)


def average_precision(y,s):
    order=np.argsort(-s,kind='stable');y=y[order]
    return float(((np.cumsum(y)/(np.arange(len(y))+1))*y).sum()/y.sum())


def attribute_metrics(targets,scores):
    t=targets.cpu().numpy();s=scores.cpu().numpy();records=[]
    for a in range(t.shape[1]):
        known=t[:,a]>=0;y=t[known,a];p=s[known,a];positives=int((y==1).sum());negatives=int((y==0).sum())
        if not positives or not negatives:continue
        predicted=p>=.5;tp=int(((y==1)&predicted).sum());fp=int(((y==0)&predicted).sum());fn=int(((y==1)&~predicted).sum())
        records.append(dict(id=a,positive=positives,negative=negatives,ap=average_precision(y,p),
                            f1=2*tp/max(1,2*tp+fp+fn),positive_prevalence=positives/len(y)))
    return dict(attribute_map=float(np.mean([r['ap'] for r in records])) if records else None,
                attribute_macro_f1=float(np.mean([r['f1'] for r in records])) if records else None,
                attribute_evaluable_count=len(records),known_attribute_labels=int((t>=0).sum()),per_attribute=records)


def shuffled_indices(cache,train_idx,seed):
    mapping=torch.arange(len(cache['rows']));rng=random.Random(seed)
    for insect in [True,False]:
        group=[i for i in train_idx if 'review_id' in cache['rows'][i] and (cache['rows'][i]['label']<10)==insect]
        other=group.copy()
        while True:
            rng.shuffle(other)
            if all(cache['rows'][a]['label']!=cache['rows'][b]['label'] for a,b in zip(group,other)):break
        mapping[group]=torch.tensor(other)
    return mapping


def batch_forward(model,cache,ids,intervention=None):
    return model(cache['global_features'][ids],cache['patch_features'][ids],cache['valid_patches'][ids],intervention)


@torch.no_grad()
def evaluate(model,cache,ids,intervention=None):
    model.eval();probs=[];attr=[];attentions=[];contribution=[];loss=0.
    for start in range(0,len(ids),32):
        ix=ids[start:start+32];o=batch_forward(model,cache,ix,intervention)
        loss+=float(F.cross_entropy(o['logits'],cache['labels'][ix],reduction='sum'))
        probs.append(o['logits'].softmax(-1).cpu());attr.append(o['attribute_logits'].sigmoid().cpu())
        attentions.append(o['attention'].cpu());contribution.append(o['attribute_contribution'].cpu())
    probs=torch.cat(probs);attr=torch.cat(attr);attention=torch.cat(attentions);contribution=torch.cat(contribution)
    labels=cache['labels'][ids].cpu();targets=cache['attribute_targets'][ids].cpu()
    evidence=cache['evidence_masks'][ids].cpu();valid=cache['valid_patches'][ids].cpu()
    positive=(targets==1)&evidence.any(-1)
    mass=(attention*evidence).sum(-1);peak_inside=evidence.gather(-1,attention.argmax(-1)[...,None]).squeeze(-1)
    area=(evidence&valid[:,None,:]).sum(-1)/valid.sum(-1)[:,None]
    confusion=np.zeros((20,20),dtype=np.int64);np.add.at(confusion,(labels.numpy(),probs.argmax(-1).numpy()),1)
    attrmetric=attribute_metrics(targets,attr)
    metrics=dict(metrics_from_confusion(confusion),loss=loss/len(ids),**{k:v for k,v in attrmetric.items() if k!='per_attribute'},
        positive_grounding_pairs=int(positive.sum()),
        grounding_peak_in_box=float(peak_inside[positive].float().mean()) if positive.any() else None,
        grounding_attention_mass=float(mass[positive].mean()) if positive.any() else None,
        grounding_uniform_area_reference=float(area[positive].mean()) if positive.any() else None,
        attribute_branch_only_accuracy=float((contribution.argmax(-1)==labels).float().mean()) if model.variant!='baseline' else None)
    return metrics,dict(probabilities=probs,attributes=attr,attention=attention,labels=labels,targets=targets,
                        evidence=evidence,paths=[cache['rows'][i]['path'] for i in ids],per_attribute=attrmetric['per_attribute'])


def run_cell(name,cache,variant,seed,lr,protocol):
    folder=ROOT/f'runs/visible_v1/{name}_{variant}_seed{seed}_lr{lr:g}'
    config=dict(backbone=name,variant=variant,seed=seed,lr=lr,protocol=protocol,
        implementation_sha256=digest(__file__),feature_metadata=cache['metadata'])
    if (folder/'config.json').exists() and json.loads((folder/'config.json').read_text(encoding='utf-8'))!=config:
        raise ValueError('Visible run differs from stored config')
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text(encoding='utf-8'))
    write_json(folder/'config.json',config);seed_all(seed);model=VisibleHead(cache,variant).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=protocol['updates'])
    ti=[i for i,r in enumerate(cache['rows']) if r['split']=='train'];vi=[i for i,r in enumerate(cache['rows']) if r['split']=='val']
    annotated=[i for i in ti if bool((cache['attribute_targets'][i]>=0).any())]
    assert len(ti)==len(vi)==200 and all(cache['rows'][i]['split']=='train' for i in annotated)
    mapping=shuffled_indices(cache,ti,seed).cuda() if variant=='shuffled_supervision' else torch.arange(len(cache['rows']),device='cuda')
    targets=cache['attribute_targets'][ti];pos=(targets==1).sum(0);neg=(targets==0).sum(0);pos_weight=(neg/pos.clamp_min(1)).clamp(1,8)
    rng=random.Random(seed);best=None;history=[];first_gradient=None
    supervised=variant in ('attribute_supervision','attribute_region','shuffled_supervision')
    for step in range(1,protocol['updates']+1):
        ids=rng.sample(ti,protocol['class_batch_size'])+rng.sample(annotated,protocol['annotated_extra_batch_size'])
        model.train();optimizer.zero_grad(set_to_none=True);o=batch_forward(model,cache,ids)
        ce=F.cross_entropy(o['logits'],cache['labels'][ids]);target=cache['attribute_targets'][mapping[ids]]
        evidence=cache['evidence_masks'][mapping[ids]];attr=masked_attribute_loss(o['attribute_logits'],target,pos_weight)
        region=region_loss(o['attention'],target,evidence,cache['valid_patches'][ids])
        loss=ce
        if supervised:loss=loss+protocol['attribute_loss_weight']*attr
        if variant in ('attribute_region','shuffled_supervision'):loss=loss+protocol['region_loss_weight']*region
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward()
        if step==1:
            first_gradient=sum(float(p.grad.abs().sum()) for p in model.patch_adapter.parameters() if p.grad is not None)
            if supervised and first_gradient<=0:raise ValueError('Attribute adapter receives no gradient')
        nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();scheduler.step()
        if step in protocol['validation_steps']:
            metrics,pred=evaluate(model,cache,vi);history.append(dict(step=step,train_ce=float(ce.detach()),train_attribute_loss=float(attr.detach()),train_region_loss=float(region.detach()),**metrics))
            if is_better(metrics,best):
                best=metrics;best_step=step;best_pred=pred;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state);train_metrics,_=evaluate(model,cache,ti)
    interventions=[]
    if variant!='baseline':
        ref=best_pred['probabilities']
        for mode in ('zero','permuted'):
            metric,pred=evaluate(model,cache,vi,mode)
            interventions.append(dict(mode=mode,**metric,flipped_predictions=int((ref.argmax(-1)!=pred['probabilities'].argmax(-1)).sum()),
                mean_probability_l1=float((ref-pred['probabilities']).abs().sum(-1).mean())))
    result=dict(backbone=name,variant=variant,seed=seed,lr=lr,best_step=best_step,validation=best,training=train_metrics,
                first_patch_adapter_gradient_l1=first_gradient,interventions=interventions)
    torch.save(dict(state_dict=state,config=config,classes=cache['classes'],attributes=cache['attributes']),folder/'best.pt')
    torch.save(best_pred,folder/'validation_predictions.pt');write_json(folder/'history.json',history);write_json(folder/'result.json',result)
    print(f'{name} {variant} s{seed} lr{lr}: val={best["top1_accuracy"]:.3f} attrAP={best["attribute_map"]:.3f} region={best["grounding_peak_in_box"]:.3f}',flush=True)
    return result
