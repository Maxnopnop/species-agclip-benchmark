"""Evaluate the already-declared privileged class-profile hypothesis.

NOT a deployable ZSL result: class profiles use held-out-species annotations
from photos disjoint from the benchmark. Prediction still uses image features.
"""
import json
import torch
from torch import nn
from torch.nn import functional as F
from data_tools import ROOT,write_json,digest
from multimodal.cub_data import BASE,lines,prepare
from multimodal.cub_rich_data import text_bank,load_data
from multimodal.grounded_experiment import measures,rank


@torch.no_grad()
def main():
    torch.set_num_threads(4);p,m=prepare();bank=text_bank();columns=bank['columns'];mapping={c['source_id']:c['label'] for c in m['classes']};exclude={r['image_id'] for r in m['rows']}
    image_classes={int(r[0]):int(r[1]) for r in lines(BASE/'image_class_labels.txt')};support={iid:mapping[c] for iid,c in image_classes.items() if c in mapping and iid not in exclude}
    index={col+1:i for i,col in enumerate(columns)};positive=torch.zeros(20,len(columns));known=torch.zeros_like(positive)
    for r in lines(BASE/'attributes/image_attribute_labels.txt'):
        iid,aid,present,certainty=map(int,r[:4])
        if iid in support and aid in index and certainty>=3:positive[support[iid],index[aid]]+=present;known[support[iid],index[aid]]+=1
    assert set(support).isdisjoint(exclude);oracle=(positive+1)/(known+2);class_attrs=F.normalize(oracle-oracle.mean(0),dim=-1)
    saved=torch.load(ROOT/'runs/cub_rich_v1/probe.pt',weights_only=True);model=nn.Linear(512,len(columns));model.load_state_dict(saved['state'])
    variants={'oracle':class_attrs}
    for seed in [42,43,44]:variants[f'shuffled_{seed}']=class_attrs[torch.randperm(20,generator=torch.Generator().manual_seed(seed))]
    def components(stage):
        data=load_data(stage);ids=[i for i,r in enumerate(data['rows']) if r['role']!='train_seen'];image=data['native_views'][ids,0]
        return 20*image@bank['class_text'].T,F.normalize(model(image).sigmoid()-saved['mean_probability'][0],dim=-1),data['labels'][ids]
    native,attrs,labels=components('development');selected=[]
    for variant,descriptor in variants.items():
        grid=[]
        for alpha in [0.,.05,.1,.25,.5,1.]:
            metrics,_=measures(native+alpha*20*attrs@descriptor.T,labels,p,'development');grid.append(dict(alpha=alpha,metrics=metrics))
        chosen=max(grid,key=lambda r:rank(r['metrics']));selected.append(dict(variant=variant,selected=chosen,grid=grid))
    source=dict(source_sha256=digest(__file__),probe_sha256=digest(ROOT/'runs/cub_rich_v1/probe.pt'),scope='PRIVILEGED ORACLE: other photos of held-out species provide class-attribute annotations. Excludes all benchmark photo IDs, but violates strict inductive zero-shot assumptions. Not a deployable AG result or theoretical upper bound.')
    out=ROOT/'runs/cub_oracle_transfer_v1';write_json(out/'selection_locked.json',dict(**source,selected=selected))
    native,attrs,labels=components('evaluation');baseline,_=measures(native,labels,p,'evaluation');final=[]
    for row in selected:
        variant=row['variant'];alpha=row['selected']['alpha'];logits=native+alpha*20*attrs@variants[variant].T;metrics,pred=measures(logits,labels,p,'evaluation')
        torch.save(dict(logits=logits,labels=labels,predictions=pred),out/f'{variant}_predictions.pt');final.append(dict(variant=variant,alpha=alpha,metrics=metrics));print(variant,alpha,metrics['H'],metrics['ZSL'],flush=True)
    write_json(ROOT/'reports/cub_followup_v1/oracle_transfer.json',dict(**source,native=baseline,selection=selected,final=final))


if __name__=='__main__':main()
