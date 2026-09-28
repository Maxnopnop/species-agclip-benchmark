"""Development-only class/attribute compatibility diagnostic with explicit oracle."""
import json
from collections import Counter
import torch
from torch import nn
from torch.nn import functional as F
from data_tools import ROOT,write_json,digest
from multimodal.cub_data import BASE,lines,prepare
from multimodal.cub_rich_data import text_bank,load_data
from multimodal.grounded_experiment import measures,rank


@torch.no_grad()
def main():
    torch.set_num_threads(4);p,m=prepare();bank=text_bank();data=load_data('development');columns=bank['columns'];mapping={c['source_id']:c['label'] for c in m['classes']};exclude={r['image_id'] for r in m['rows']}
    image_classes={int(r[0]):int(r[1]) for r in lines(BASE/'image_class_labels.txt')};support={iid:mapping[c] for iid,c in image_classes.items() if c in mapping and iid not in exclude}
    index={col+1:i for i,col in enumerate(columns)};positive=torch.zeros(20,len(columns));known=torch.zeros_like(positive)
    for r in lines(BASE/'attributes/image_attribute_labels.txt'):
        iid,aid,present,certainty=map(int,r[:4])
        if iid in support and aid in index and certainty>=3:
            positive[support[iid],index[aid]]+=present;known[support[iid],index[aid]]+=1
    assert set(support).isdisjoint(exclude);oracle=(positive+1)/(known+2)
    saved=torch.load(ROOT/'runs/cub_rich_v1/probe.pt',weights_only=True);model=nn.Linear(512,len(columns));model.load_state_dict(saved['state']);ids=[i for i,r in enumerate(data['rows']) if r['role']!='train_seen']
    pred=model(data['native_views'][ids,0]).sigmoid();image_attrs=F.normalize(pred-saved['mean_probability'][0],dim=-1)
    native=20*data['native_views'][ids,0]@bank['class_text'].T;labels=data['labels'][ids];baseline,_=measures(native,labels,p,'development');results=[]
    text_descriptor=(10*bank['class_text']@(bank['attribute_text']-bank['negative_text']).T).sigmoid()
    for name,descriptor in [('frozen_text_only',text_descriptor),('privileged_class_attributes',oracle)]:
        class_attrs=F.normalize(descriptor-descriptor.mean(0),dim=-1);attribute_logits=20*image_attrs@class_attrs.T
        attr_only,_=measures(attribute_logits,labels,p,'development');grid=[]
        for alpha in [0.,.05,.1,.25,.5,1.]:
            metrics,_=measures(native+alpha*attribute_logits,labels,p,'development');grid.append(dict(alpha=alpha,metrics=metrics))
        selected=max(grid,key=lambda r:rank(r['metrics']));results.append(dict(descriptor=name,attribute_only=attr_only,selected=selected,grid=grid));print(name,selected['alpha'],selected['metrics']['H'],attr_only['H'],flush=True)
    report=dict(native=baseline,results=results,support_images_by_class=dict(Counter(support.values())),source_sha256=digest(__file__),scope='Development-only diagnosis. Frozen-text descriptors need no extra image labels. Privileged class profiles use annotations from OTHER photos of all 20 species, disjoint from all 700 benchmark photos, INCLUDING held-out species. This therefore uses unavailable unseen-class image supervision and is NOT a strict zero-shot or deployable AG result. It tests a class/attribute association hypothesis; no final images were evaluated.')
    write_json(ROOT/'reports/cub_followup_v1/class_attribute_diagnosis.json',report)


if __name__=='__main__':main()
