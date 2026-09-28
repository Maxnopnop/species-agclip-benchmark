"""Attribute transfer follow-up on fixed cub100 features; no fusion changes."""
import json,time
import torch
from torch import nn
from torch.nn import functional as F
from data_tools import ROOT,digest,write_json
from run import seed_all
from multimodal.cub100_data import load_data,text_bank,prepare
from multimodal.cub_token_experiment import attribute_score

OUT=ROOT/'runs/cub_probe_regularization_v1'
REPORT=ROOT/'reports/cub_probe_regularization_v1'
MODES=[('baseline',False,0.),('balanced',True,0.),('anchor_001',False,.001),('anchor_01',False,.01),('balanced_anchor_001',True,.001),('balanced_anchor_01',True,.01)]
STEPS=[0,100,300,600]


def main():
    seed_all(42);OUT.mkdir(parents=True,exist_ok=True);p,m=prepare();bank=text_bank();data=load_data('development')
    train=[i for i,r in enumerate(data['rows']) if r['role']=='train_seen'];val={kind:[i for i,r in enumerate(data['rows']) if r['role']=='dev_'+kind] for kind in ['seen','unseen']}
    assert set(data['labels'][train].tolist())==set(p['seen_classes'])
    x=data['native_views'].cuda();y=data['targets'].cuda();initial=10*(bank['attribute_text']-bank['negative_text']).cuda();a=initial.shape[0]
    known=y[train]>=0;pos=((y[train]==1)&known).sum((0,1));neg=((y[train]==0)&known).sum((0,1));posweight=(neg/pos.clamp(min=1)).sqrt().clamp(min=1,max=5)
    def loss(logits,target,balanced):
        mask=target>=0;loss=F.binary_cross_entropy_with_logits(logits,target.clamp(min=0),reduction='none')
        weight=torch.where(target==1,posweight,1.) if balanced else torch.ones_like(target)
        return (loss*mask*weight).sum()/(mask*weight).sum().clamp(min=1)
    protocol=dict(source_sha256=digest(__file__),parent_probe_sha256=digest(ROOT/'runs/cub100_v1/probe.pt'),manifest_sha256=digest(ROOT/'data/cub100_v1/manifest.json'),modes=MODES,learning_rates=[.01,.001],steps=STEPS,selection='Mean of seen and unseen development global/regional attribute mAP; same criterion for all modes. No species classification score used. Fixed seed, full-batch linear probes; no fusion run in this diagnostic.',scope='Exploratory follow-up after cub100 final results were inspected. Separate locked checkpoints, never overwrite parent probes. Positive weights use only seen training targets. Weight anchoring uses frozen text initialization; biases receive ordinary AdamW decay.')
    write_json(OUT/'protocol.json',protocol);selected=[]
    for name,balanced,anchor in MODES:
        best=None;history=[];start=time.time()
        for lr in [.01,.001]:
            seed_all(42);model=nn.Linear(512,a).cuda()
            with torch.no_grad():model.weight.copy_(initial);model.bias.zero_()
            opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
            for step in range(601):
                if step in STEPS:
                    with torch.no_grad():
                        group={kind:attribute_score(y[ix].cpu(),model(x[ix]).sigmoid().cpu()) for kind,ix in val.items()};score=sum(r['selection_map'] for r in group.values())/2
                    row=dict(mode=name,lr=lr,step=step,score=score,groups=group);history.append(row)
                    if best is None or score>best['score']:
                        best=row;torch.save(dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},selection=row,protocol=protocol),OUT/f'{name}.pt')
                if step==600:break
                logits=model(x[train]);objective=loss(logits[:,0],y[train,0],balanced)+.5*loss(logits[:,1:],y[train,1:],balanced)+anchor*(model.weight-initial).square().sum(-1).mean()
                opt.zero_grad();objective.backward();opt.step()
        write_json(OUT/f'{name}_history.json',history);selected.append(best);print(name,'dev',best['score'],best['lr'],best['step'],'seconds',round(time.time()-start,1),flush=True)
    write_json(OUT/'selection_locked.json',dict(protocol=protocol,selected=selected))
    # Evaluation features and labels enter only after every mode is locked.
    final=load_data('evaluation');results=[]
    for row in selected:
        model=nn.Linear(512,a);model.load_state_dict(torch.load(OUT/f'{row["mode"]}.pt',weights_only=True)['state']);groups={}
        with torch.no_grad():prob=model(final['native_views']).sigmoid()
        for kind in ['seen','unseen']:
            ix=[i for i,r in enumerate(final['rows']) if r['role']=='eval_'+kind];groups[kind]=attribute_score(final['targets'][ix],prob[ix])
        results.append(dict(mode=row['mode'],selection=row,groups=groups));print(row['mode'],'final',groups,flush=True)
    write_json(REPORT/'results.json',dict(protocol=protocol,selected=selected,final=results))


if __name__=='__main__':main()
