"""Class-disjoint CoCa pilot with differentiable cached tails and sealed test."""
import argparse,copy,gc,json,random,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import open_clip
import torch
from torch import nn
from PIL import Image
from data_tools import ROOT,digest,write_json
from run import seed_all
from multimodal.grounded_data import detector_parts,locate
from coca_components import load_full,Tails,Fusion,image_prefix,text_prefix,symmetric_loss,select_text
from coca_data import DATA,OUT,CONFIG,prepare
from coca_download import FOLDER,REVISION,SHA256

CACHE=ROOT/'cache/coca_pilot_v1'
RUN=ROOT/'runs/coca_pilot_v1'

def source():
    paths=[CONFIG,Path(__file__),ROOT/'coca_components.py',ROOT/'coca_data.py',ROOT/'coca_download.py',ROOT/'confidence_fresh.py',ROOT/'multimodal/grounded_data.py',ROOT/'data_tools.py',ROOT/'run.py',DATA/'manifest.json']
    s={p.relative_to(ROOT).as_posix():digest(p) for p in paths}
    s['open_clip_version']=open_clip.__version__;s['weight_sha256']=SHA256;s['weight_revision']=REVISION
    for name in ['transformer.py','coca_model.py','factory.py']:
        s['open_clip/'+name]=digest(Path(open_clip.__file__).parent/name)
    return s

def lock(c):
    s=source();p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())['source']==s,'Locked source changed.'
    else:write_json(p,dict(source=s,config=c,locked_at_utc=datetime.now(timezone.utc).isoformat()))
    return s

def stage_rows(m,stage):return [r for r in m['rows'] if r['role'].startswith('test')==(stage=='test')]

def check_seal(s):
    x=json.loads((OUT/'selection_locked.json').read_text());assert x['source']==s
    for p,h in x['assets'].items():assert digest(ROOT/p)==h,p
    return x

def ground(c,m,s,stage):
    if stage=='test':check_seal(s)
    rows=stage_rows(m,stage);p=CACHE/f'regions_{stage}.json'
    x=json.loads(p.read_text()) if p.exists() else dict(source=s,images={})
    assert x['source']==s
    pending=[r for r in rows if r['path'] not in x['images']]
    if pending:
        processor,model=detector_parts()
        for i,r in enumerate(pending):
            file=Path(m['image_root'])/r['path'];assert digest(file)==r['sha256']
            with Image.open(file) as im:record=locate(im.convert('RGB'),[a['prompt'] for a in m['attributes']],processor,model,c)
            x['images'][r['path']]=record
            if i%20==0:write_json(p,x);print('ground',stage,i+1,'/',len(pending),flush=True)
        write_json(p,x);del model,processor;gc.collect();torch.cuda.empty_cache()
    assert len(x['images'])==len(rows)
    return x['images']

def extract(c,m,s,stage,boxes):
    if stage=='test':check_seal(s)
    target=CACHE/f'prefix_{stage}.pt'
    if target.exists():assert torch.load(target,weights_only=True)['source']==s;return
    model,transform,tokenize=load_full()
    init=CACHE/'initial_tail.pt'
    if not init.exists():
        tail=Tails(model);torch.save(dict(source=s,state_dict=tail.state_dict()),init);del tail
    model=model.cuda();rows=stage_rows(m,stage)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        textfile=CACHE/'text_prefix.pt'
        if not textfile.exists():
            prompts=[f'a photo of a {x["name"]}.' for x in m['classes']]+[a['prompt'] for a in m['attributes']]
            parts=[];masks=[]
            for start in range(0,len(prompts),8):
                t=tokenize(prompts[start:start+8]).cuda();p,mask=text_prefix(model,t);parts.append(p.bfloat16().cpu());masks.append(mask.bfloat16().cpu())
            torch.save(dict(source=s,prefix=torch.cat(parts),mask=torch.cat(masks),prompts=prompts),textfile)
        parts=[];ids=[];scores=[];valid=[]
        for start in range(0,len(rows),2):
            views=[]
            for row in rows[start:start+2]:
                file=Path(m['image_root'])/row['path'];assert digest(file)==row['sha256']
                with Image.open(file) as im:im=im.convert('RGB')
                rec=boxes[row['path']];n=len(rec['scores']);assert n<=2
                views.append(transform(im));views.extend(transform(im.crop(tuple(b))) for b in rec['boxes']);views.extend(transform(im) for _ in range(2-n))
                ids.append(rec['attribute_ids']+[0]*(2-n));scores.append(rec['scores']+[0.]*(2-n));valid.append([True]*n+[False]*(2-n))
            p=image_prefix(model,torch.stack(views).cuda()).bfloat16().cpu();parts.append(p.reshape(-1,3,*p.shape[1:]))
            if start%40==0:print('prefix',stage,start,'/',len(rows),flush=True)
        torch.save(dict(source=s,rows=rows,prefix=torch.cat(parts),ids=torch.tensor(ids),confidence=torch.tensor(scores),valid=torch.tensor(valid),labels=torch.tensor([r['label'] for r in rows]),regions_sha256=digest(CACHE/f'regions_{stage}.json')),target)
    del model,parts;gc.collect();torch.cuda.empty_cache()

def load_stage(stage,s):
    d=torch.load(CACHE/f'prefix_{stage}.pt',weights_only=True);assert d['source']==s
    assert all(r['role'].startswith('test')==(stage=='test') for r in d['rows'])
    assert d['regions_sha256']==digest(CACHE/f'regions_{stage}.json')
    return d

def bank(s):
    b=torch.load(CACHE/'text_prefix.pt',weights_only=True);assert b['source']==s
    return {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in b.items()}

def make_tail():
    model,_,_=load_full();tail=Tails(model);del model;gc.collect();return tail

def encode_text(tail,b,ix):
    p,mask=select_text(b['prefix'],b['mask'],ix)
    return tail.text(p,mask)

def images(tail,fusion,d,ix,attributes):
    raw=d['prefix'][ix]
    if fusion.variant=='baseline':return tail.image(raw[:,0].cuda())
    p=raw.reshape(-1,*raw.shape[2:]).cuda();features=tail.image(p).reshape(-1,3,768)
    attr=attributes[d['ids'][ix].cuda()]
    return fusion(features[:,0],features[:,1:],attr,d['valid'][ix].cuda(),d['confidence'][ix].cuda())

@torch.no_grad()
def evaluate(tail,fusion,d,b,candidates):
    tail.eval();fusion.eval();result=[]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        text=encode_text(tail,b,candidates)
        attributes=encode_text(tail,b,list(range(25,49))) if fusion.variant!='baseline' else None
        for start in range(0,len(d['rows']),8):
            im=images(tail,fusion,d,list(range(start,min(start+8,len(d['rows'])))),attributes)
            result.append((im@text.T).float().cpu())
    return torch.cat(result)

def measures(logits,rows,candidates):
    y=np.array([r['label'] for r in rows]);roles=np.array([r['role'] for r in rows]);pred=np.array(candidates)[logits.argmax(-1).numpy()]
    seen=np.array([r.endswith('_seen') for r in roles]);unseen=np.array([r.endswith('_unseen') for r in roles])
    def acc(mask,p):return float(np.mean([np.mean(p[y==k]==k) for k in np.unique(y[mask])]))
    S=acc(seen,pred);U=acc(unseen,pred);H=2*S*U/(S+U) if S+U else 0.
    uc=[k for k in candidates if k>=10];up=np.array(uc)[logits[:,[candidates.index(k) for k in uc]].argmax(-1).numpy()]
    return dict(seen=S,unseen=U,harmonic=H,zsl=acc(unseen,up))

def subset(d,indices):
    return {k:v[indices] if isinstance(v,torch.Tensor) and len(v)==len(d['rows']) else [v[i] for i in indices] if k=='rows' else v for k,v in d.items()}

def select_key(metric,step):return (metric['harmonic'],metric['zsl'],-step)

def train_all(c,m,s):
    d=load_stage('development',s);b=bank(s);template=make_tail()
    init=torch.load(CACHE/'initial_tail.pt',weights_only=True);assert init['source']==s
    vi=[i for i,r in enumerate(d['rows']) if r['role'].startswith('val')];vd=subset(d,vi)
    byclass={k:[i for i,r in enumerate(d['rows']) if r['role']=='train' and r['label']==k] for k in range(10)}
    for seed in c['seeds']:
        for variant in c['variants']:
            target=RUN/f'{seed}/{variant}.pt';target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert torch.load(target,weights_only=True)['source']==s;continue
            seed_all(seed);tail=copy.deepcopy(template).cuda();tail.load_state_dict(init['state_dict']);fusion=Fusion(variant).cuda()
            optimizer=torch.optim.AdamW([dict(params=tail.parameters(),lr=c['encoder_lr']),dict(params=fusion.parameters(),lr=c['fusion_lr'])],weight_decay=c['weight_decay'])
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,c['steps'])
            rng=random.Random(seed);history=[];best=None;start=time.monotonic();torch.cuda.reset_peak_memory_stats()
            for step in range(c['steps']+1):
                if step%c['validation_interval']==0:
                    logits=evaluate(tail,fusion,vd,b,list(range(15)));metric=measures(logits,vd['rows'],list(range(15)));history.append(dict(step=step,**metric))
                    if best is None or select_key(metric,step)>select_key(best['validation'],best['best_step']):
                        best=dict(validation=metric,best_step=step,tail={k:v.detach().cpu().clone() for k,v in tail.state_dict().items()},fusion={k:v.detach().cpu().clone() for k,v in fusion.state_dict().items()})
                    print('validation',seed,variant,step,metric,flush=True)
                if step==c['steps']:break
                tail.train();fusion.train();ix=[rng.choice(byclass[k]) for k in range(10)]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    text=encode_text(tail,b,list(range(10)))
                    attr=encode_text(tail,b,list(range(25,49))) if variant!='baseline' else None
                    im=images(tail,fusion,d,ix,attr);loss=symmetric_loss(im,text,tail.logit_scale)
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(list(tail.parameters())+list(fusion.parameters()),1.);optimizer.step();scheduler.step()
            torch.save(dict(source=s,seed=seed,variant=variant,history=history,seconds=time.monotonic()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,**best),target)
            del tail,fusion,optimizer,scheduler,best,logits;gc.collect();torch.cuda.empty_cache()
    paths=[CACHE/'initial_tail.pt',CACHE/'text_prefix.pt',CACHE/'prefix_development.pt',CACHE/'regions_development.json']+[RUN/f'{seed}/{v}.pt' for seed in c['seeds'] for v in c['variants']]
    seal=dict(source=s,assets={p.relative_to(ROOT).as_posix():digest(p) for p in paths})
    path=OUT/'selection_locked.json'
    if path.exists():assert json.loads(path.read_text())['assets']==seal['assets']
    else:write_json(path,seal|dict(locked_at_utc=datetime.now(timezone.utc).isoformat()))

def score(c,m,s):
    check_seal(s);path=OUT/'results.json'
    if path.exists():assert json.loads(path.read_text())['source']==s;return
    d=load_stage('test',s);b=bank(s);tail=make_tail().cuda();pred={};rows=[]
    candidates=list(range(10))+list(range(15,25))
    initial=torch.load(CACHE/'initial_tail.pt',weights_only=True);tail.load_state_dict(initial['state_dict'])
    native=evaluate(tail,Fusion('baseline').cuda(),d,b,candidates);pred['native']=native
    native_metrics=measures(native,d['rows'],candidates)
    for seed in c['seeds']:
        for v in c['variants']:
            ck=torch.load(RUN/f'{seed}/{v}.pt',weights_only=True);tail.load_state_dict(ck['tail']);fusion=Fusion(v).cuda();fusion.load_state_dict(ck['fusion'])
            logits=evaluate(tail,fusion,d,b,candidates);metric=measures(logits,d['rows'],candidates)
            pred[f'{seed}/{v}']=logits;rows.append(dict(seed=seed,variant=v,best_step=ck['best_step'],**metric));print('test',seed,v,metric,flush=True)
    torch.save(dict(source=s,predictions=pred,paths=[r['path'] for r in d['rows']],labels=d['labels'],candidates=candidates),RUN/'predictions.pt')
    write_json(path,dict(source=s,scored_at_utc=datetime.now(timezone.utc).isoformat(),native=native_metrics,rows=rows))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','development','train','test','all']);a=parser.parse_args().action
    torch.set_num_threads(4);CACHE.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    c=json.loads(CONFIG.read_text());m=prepare();s=lock(c)
    if a in ('development','all'):boxes=ground(c,m,s,'development');extract(c,m,s,'development',boxes)
    if a in ('train','all'):train_all(c,m,s)
    if a in ('test','all'):check_seal(s);boxes=ground(c,m,s,'test');extract(c,m,s,'test',boxes);score(c,m,s)

if __name__=='__main__':main()
