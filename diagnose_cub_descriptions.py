"""Matched public-description baseline and predicted-attribute reranking.

No held-out image annotations enter class profiles. This is an exploratory
AG-inspired compatibility diagnostic, not a reproduction of AG-CLIP training.
"""
import json
import torch
from torch import nn
from torch.nn import functional as F
from data_tools import ROOT, digest, write_json
from multimodal.cub_data import prepare
from multimodal.cub_rich_data import text_bank, load_data
from multimodal.models import load_clip
from multimodal.grounded_experiment import measures, rank

CONFIG=ROOT/'configs/cub_descriptions_v1.json'
OUT=ROOT/'runs/cub_descriptions_v1'
REPORT=ROOT/'reports/cub_descriptions_v1'


@torch.no_grad()
def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    p,m=prepare();cfg=json.loads(CONFIG.read_text());bank=text_bank()
    assert [s['label'] for s in cfg['species']]==list(range(len(m['classes'])))
    prompts=[d for s in cfg['species'] for d in s['descriptions']]
    assert all(len(s['descriptions'])==2 for s in cfg['species'])
    model,_,tokenizer=load_clip();tokens=tokenizer(prompts)
    # Reject text at the context boundary rather than silently truncating.
    lengths=(tokens!=0).sum(-1);assert int(lengths.max())<77
    model=model.cuda().eval()
    emb=torch.cat([model.encode_text(tokens[i:i+16].cuda(),normalize=True).float().cpu() for i in range(0,len(tokens),16)])
    desc=F.normalize(emb.reshape(20,2,-1).mean(1),dim=-1)
    del model;torch.cuda.empty_cache()
    profile=(10*desc@(bank['attribute_text']-bank['negative_text']).T).sigmoid()
    class_attrs=F.normalize(profile-profile.mean(0),dim=-1)
    saved=torch.load(ROOT/'runs/cub_rich_v1/probe.pt',weights_only=True)
    probe=nn.Linear(512,profile.shape[1]);probe.load_state_dict(saved['state']);probe.eval()
    variants={'description_only':None,'attributes':class_attrs,'constant':class_attrs}
    for seed in cfg['permutation_seeds']:
        permutation=torch.randperm(20,generator=torch.Generator().manual_seed(seed))
        variants[f'shuffled_{seed}']=class_attrs[permutation]
    provenance=dict(source_sha256=digest(__file__),config_sha256=digest(CONFIG),probe_sha256=digest(ROOT/'runs/cub_rich_v1/probe.pt'),manifest_sha256=digest(ROOT/'data/cub_attributes_v1/manifest.json'))
    write_json(OUT/'protocol.json',dict(**provenance,config=cfg,max_token_length=int(lengths.max()),note='Descriptions fixed before this experiment. Reuses previously inspected 20-species evaluation. Equal description blend and attribute-weight grids for correct/shuffled/constant controls. Frozen probe uses seen training images only. No privileged class profiles.'))

    def components(stage):
        data=load_data(stage);ids=[i for i,r in enumerate(data['rows']) if r['role']!='train_seen']
        image=data['native_views'][ids,0];labels=data['labels'][ids]
        attrs=F.normalize(probe(image).sigmoid()-saved['mean_probability'][0],dim=-1)
        return 20*image@bank['class_text'].T,20*image@desc.T,attrs,labels

    def scores(native,described,attrs,variant,beta,alpha):
        x=(1-beta)*native+beta*described
        if variant=='constant': attrs=torch.zeros_like(attrs)
        if variants[variant] is not None:x=x+alpha*20*attrs@variants[variant].T
        return x

    native,described,attrs,labels=components('development');selected=[]
    for variant in variants:
        grid=[]
        weights=[0.] if variant=='description_only' else cfg['attribute_weights']
        for beta in cfg['description_blends']:
            for alpha in weights:
                metrics,_=measures(scores(native,described,attrs,variant,beta,alpha),labels,p,'development')
                grid.append(dict(beta=beta,alpha=alpha,metrics=metrics))
        chosen=max(grid,key=lambda r:rank(r['metrics']));selected.append(dict(variant=variant,selected=chosen,grid=grid))
        print('development',variant,chosen['beta'],chosen['alpha'],chosen['metrics']['H'],flush=True)
    write_json(OUT/'selection_locked.json',dict(provenance=provenance,selected=selected))
    # Nothing from evaluation was loaded before this selection file existed.
    native,described,attrs,labels=components('evaluation');final=[]
    native_metrics,native_pred=measures(native,labels,p,'evaluation')
    torch.save(dict(logits=native,labels=labels,predictions=native_pred),OUT/'native_predictions.pt')
    for row in selected:
        variant=row['variant'];ch=row['selected'];logits=scores(native,described,attrs,variant,ch['beta'],ch['alpha'])
        metrics,pred=measures(logits,labels,p,'evaluation');final.append(dict(variant=variant,beta=ch['beta'],alpha=ch['alpha'],metrics=metrics))
        torch.save(dict(logits=logits,labels=labels,predictions=pred),OUT/f'{variant}_predictions.pt')
        print('evaluation',variant,metrics['H'],metrics['ZSL'],flush=True)
    correct=next(r for r in selected if r['variant']=='attributes')['selected'];interventions=[]
    _,original_pred=measures(scores(native,described,attrs,'attributes',correct['beta'],correct['alpha']),labels,p,'evaluation')
    for variant in variants:
        logits=scores(native,described,attrs,variant,correct['beta'],correct['alpha']);metrics,pred=measures(logits,labels,p,'evaluation')
        interventions.append(dict(variant=variant,metrics=metrics,changed_predictions=int((pred!=original_pred).sum())))
    write_json(REPORT/'results.json',dict(provenance=provenance,native=native_metrics,selection=selected,final=final,interventions_at_correct_weights=interventions,scope='Exploratory reused-data, fixed frozen probe and public description paraphrases. Shuffling seeds are controls, not independent training seeds. No held-out image annotations used for descriptors. Not a full AG-CLIP reproduction.'))


if __name__=='__main__':main()
