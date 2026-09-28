"""Positive-only class attributes, matched to the same public text baseline."""
import json
import torch
from torch import nn
from torch.nn import functional as F
from data_tools import ROOT,digest,write_json
from multimodal.cub_data import prepare
from multimodal.cub_rich_data import text_bank,load_data,names
from multimodal.models import load_clip
from multimodal.grounded_experiment import measures,rank

CONFIG=ROOT/'configs/cub_description_attributes_v1.json'
OUT=ROOT/'runs/cub_mapped_attributes_v1'
REPORT=ROOT/'reports/cub_mapped_attributes_v1'


@torch.no_grad()
def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True);p,m=prepare();bank=text_bank()
    cfg=json.loads(CONFIG.read_text());desc_cfg=json.loads((ROOT/'configs/cub_descriptions_v1.json').read_text())
    assert cfg['description_sha256']==digest(ROOT/'configs/cub_descriptions_v1.json')
    assert cfg['mapping_source_sha256']==digest(ROOT/'build_cub_description_attributes.py')
    lookup={names()[c]:i for i,c in enumerate(bank['columns'])}
    profiles=torch.zeros(20,2,len(lookup))
    for s in cfg['species']:
        for j,alternative in enumerate(s['positive_alternatives']):
            for name in alternative:profiles[s['label'],j,lookup[name]]=1
    profiles=profiles/profiles.sum(-1,keepdim=True)
    model,_,tokenizer=load_clip();model=model.cuda().eval();prompts=[d for s in desc_cfg['species'] for d in s['descriptions']];tokens=tokenizer(prompts)
    assert int((tokens!=0).sum(-1).max())<77
    desc=F.normalize(torch.cat([model.encode_text(tokens[i:i+16].cuda(),normalize=True).float().cpu() for i in range(0,len(tokens),16)]).reshape(20,2,-1).mean(1),dim=-1)
    del model;torch.cuda.empty_cache()
    saved=torch.load(ROOT/'runs/cub_rich_v1/probe.pt',weights_only=True);probe=nn.Linear(512,len(lookup));probe.load_state_dict(saved['state']);probe.eval()
    data=load_data('development');train=[i for i,r in enumerate(data['rows']) if r['role']=='train_seen'];train_images=data['native_views'][train,0]
    def probability(image,kind):
        return probe(image).sigmoid() if kind=='trained' else (10*image@(bank['attribute_text']-bank['negative_text']).T).sigmoid()
    normalization={}
    for kind in ['trained','native']:
        q=probability(train_images,kind);mean=q.mean(0);sd=q.std(0).clamp(min=.05)
        z=(q-mean)/sd;score=torch.einsum('na,cpa->ncp',z,profiles).max(-1).values
        offset=score.mean(0);scale=(score-offset).square().mean().sqrt().clamp(min=.01)
        normalization[kind]=(mean,sd,offset,scale)
    def attribute_score(image,kind,constant=False):
        mean,sd,offset,scale=normalization[kind];z=(probability(image,kind)-mean)/sd
        if constant:z=torch.zeros_like(z)
        return (torch.einsum('na,cpa->ncp',z,profiles).max(-1).values-offset)/scale
    variants={'description_only':('trained',False,None),'mapped_trained':('trained',False,None),'mapped_native':('native',False,None),'constant':('trained',True,None)}
    for seed in cfg['permutation_seeds']:variants[f'shuffled_{seed}']=('trained',False,torch.randperm(20,generator=torch.Generator().manual_seed(seed)))
    provenance=dict(source_sha256=digest(__file__),config_sha256=digest(CONFIG),description_sha256=cfg['description_sha256'],probe_sha256=digest(ROOT/'runs/cub_rich_v1/probe.pt'))
    write_json(OUT/'protocol.json',dict(provenance=provenance,config=cfg,scoring='Training-centered/scaled attribute probabilities; mean over source-supported positive attributes, max across two alternative phenotypes; per-class training score centering and shared scale. Unknowns omitted. Equal tuning grid for nontrivial controls.'))
    def components(stage):
        d=load_data(stage);ids=[i for i,r in enumerate(d['rows']) if r['role']!='train_seen'];image=d['native_views'][ids,0]
        attr={}
        for variant,(kind,constant,perm) in variants.items():
            x=attribute_score(image,kind,constant);attr[variant]=x if perm is None else x[:,perm]
        return 20*image@bank['class_text'].T,20*image@desc.T,attr,d['labels'][ids]
    native,described,attrs,labels=components('development');selected=[]
    for variant in variants:
        grid=[]
        for beta in cfg['description_blends']:
            for alpha in ([0.] if variant=='description_only' else cfg['attribute_weights']):
                logits=(1-beta)*native+beta*described+alpha*attrs[variant];metrics,_=measures(logits,labels,p,'development')
                grid.append(dict(beta=beta,alpha=alpha,metrics=metrics))
        chosen=max(grid,key=lambda r:rank(r['metrics']));selected.append(dict(variant=variant,selected=chosen,grid=grid));print('development',variant,chosen['beta'],chosen['alpha'],chosen['metrics']['H'],flush=True)
    write_json(OUT/'selection_locked.json',dict(provenance=provenance,selected=selected))
    native,described,attrs,labels=components('evaluation');final=[]
    for row in selected:
        variant=row['variant'];ch=row['selected'];logits=(1-ch['beta'])*native+ch['beta']*described+ch['alpha']*attrs[variant];metrics,pred=measures(logits,labels,p,'evaluation')
        torch.save(dict(logits=logits,labels=labels,predictions=pred),OUT/f'{variant}_predictions.pt');final.append(dict(variant=variant,beta=ch['beta'],alpha=ch['alpha'],metrics=metrics));print('evaluation',variant,metrics['H'],metrics['ZSL'],flush=True)
    write_json(REPORT/'results.json',dict(provenance=provenance,selection=selected,final=final,mapped_attributes=int((profiles.sum((0,1))>0).sum()),scope='Exploratory reused 20-species split. Source-backed positive-only semantic table, AI-assisted rather than expert validated. No held-out image annotation enters inference or class profiles. This extra mapping is an AG-inspired diagnostic, not paper reproduction.'))


if __name__=='__main__':main()
