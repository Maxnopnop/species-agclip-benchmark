"""Temporal FungiTastic pilot; unchanged models and sealed test extraction."""
import argparse,json,random,hashlib,copy
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from scipy.fft import dctn
from data_tools import ROOT,digest,write_json
from run import seed_all
from multimodal.models import load_backbone,load_clip
from multimodal.grounded_data import detector_parts,locate
from multimodal.experiment import split_indices
from confidence_experiment import Head,train,evaluate

CONFIG=ROOT/'configs/fungitastic_confidence_v1.json'
DATA=ROOT/'data/fungitastic_confidence_v1'
CACHE=ROOT/'cache/fungitastic_confidence_v1'
OUT=ROOT/'reports/fungitastic_confidence_v1'
RUN=ROOT/'runs/fungitastic_confidence_v1'
from fungitastic_data import prepare as prepare_manifest

def source(m):
    paths=[CONFIG,Path(__file__),ROOT/'fungitastic_data.py',ROOT/'confidence_fresh.py',ROOT/'confidence_experiment.py',ROOT/'multimodal/models.py',ROOT/'multimodal/grounded_data.py',ROOT/'multimodal/expanded.py',ROOT/'multimodal/experiment.py',ROOT/'run.py',DATA/'manifest.json']
    for p,h in m['metadata_hashes'].items():
        assert digest(ROOT/p)==h
        paths.append(ROOT/p)
    assert digest(ROOT/'cache/confidence_fresh_v1/historical_fingerprints.json')==m['historical_fingerprints_sha256']
    assert digest(ROOT/'data/confidence_fresh_v1/manifest.json')==m['prior_fresh_manifest_sha256']
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}

def lock(c,m):
    s=source(m);path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())['source']==s,'Locked source changed.'
    else:write_json(path,dict(config=c,source=s,locked_at_utc=datetime.now(timezone.utc).isoformat()))
    return s

def check_seal(s):
    saved=json.loads((OUT/'selection_locked.json').read_text());assert saved['source']==s
    for x in saved['checkpoints']:assert digest(RUN/f"{x['backbone']}_{x['seed']}/{x['variant']}.pt")==x['checkpoint_sha256']
    for p,h in saved['development_assets'].items():assert digest(ROOT/p)==h
    return saved

def ground(c,m,s,stage):
    if stage=='test':check_seal(s)
    rows=[r for r in m['rows'] if (r['split']=='test')==(stage=='test')]
    path=CACHE/f'regions_{stage}.json';saved=json.loads(path.read_text()) if path.exists() else dict(source=s,images={})
    assert saved['source']==s;pending=[r for r in rows if r['path'] not in saved['images']]
    if pending:
        processor,model=detector_parts();prompts=[a['prompt'] for a in m['attributes']]
        for j,row in enumerate(pending):
            file=Path(m['image_root'])/row['path'];assert digest(file)==row['sha256']
            with Image.open(file) as im:record=locate(im.convert('RGB'),prompts,processor,model,c)
            saved['images'][row['path']]=record
            if j%20==0:write_json(path,saved);print('ground',stage,j+1,'/',len(pending),flush=True)
        write_json(path,saved);del model,processor;torch.cuda.empty_cache()
    assert len(saved['images'])==len(rows)
    return saved['images']

def text_bank(m,s):
    path=CACHE/'text.pt'
    if path.exists():b=torch.load(path,weights_only=True);assert b['source']==s;return b
    model,_,tokenize=load_clip();model=model.cuda().eval()
    prompts=[f'a photo of a {x["name"]}.' for x in m['classes']]+[a['prompt'] for a in m['attributes']]
    with torch.inference_mode():t=torch.cat([model.encode_text(tokenize(prompts[i:i+16]).cuda(),normalize=True).float().cpu() for i in range(0,len(prompts),16)])
    b=dict(source=s,class_text=t[:10],attribute_text=t[10:]);torch.save(b,path);del model;torch.cuda.empty_cache();return b

def extract(c,m,s,stage,boxes):
    if stage=='test':check_seal(s)
    rows=[r for r in m['rows'] if (r['split']=='test')==(stage=='test')]
    for name in c['backbones']:
        path=CACHE/f'{name}_{stage}.pt';boxhash=digest(CACHE/f'regions_{stage}.json')
        if path.exists():d=torch.load(path,weights_only=True);assert d['source']==s and d['regions_sha256']==boxhash;continue
        model,transform,dim=load_backbone(name);model=model.cuda().eval();parts=[];ids=[];scores=[];valid=[]
        with torch.inference_mode():
            for start in range(0,len(rows),8):
                views=[]
                for row in rows[start:start+8]:
                    file=Path(m['image_root'])/row['path'];assert digest(file)==row['sha256']
                    with Image.open(file) as im:im=im.convert('RGB')
                    rec=boxes[row['path']];n=len(rec['scores']);assert n<=2
                    views.append(transform(im));views.extend(transform(im.crop(tuple(b))) for b in rec['boxes']);views.extend(transform(im) for _ in range(2-n))
                    ids.append(rec['attribute_ids']+[0]*(2-n));scores.append(rec['scores']+[0.]*(2-n));valid.append([True]*n+[False]*(2-n))
                parts.append(F.normalize(model(torch.stack(views).cuda()).float(),dim=-1).cpu().reshape(-1,3,dim))
                if start%80==0:print('features',name,stage,start,flush=True)
        f=torch.cat(parts);torch.save(dict(source=s,regions_sha256=boxhash,rows=rows,classes=m['classes'],g=f[:,0],r=f[:,1:],ids=torch.tensor(ids),confidence=torch.tensor(scores),valid=torch.tensor(valid),labels=torch.tensor([r['label'] for r in rows])),path)
        del model;torch.cuda.empty_cache()

def cache(name,stage):
    d=torch.load(CACHE/f'{name}_{stage}.pt',weights_only=True)
    assert all((r['split']=='test')==(stage=='test') for r in d['rows'])
    return {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in d.items()}

def train_all(c,bank,s):
    for name in c['backbones']:
        d=cache(name,'development');assert len(d['rows'])==300;vi=split_indices(d,'val')
        for seed in c['seeds']:
            ti=split_indices(d,'train',20,seed);assert len(ti)==200
            for variant in c['variants']:
                folder=RUN/f'{name}_{seed}';folder.mkdir(parents=True,exist_ok=True);path=folder/f'{variant}.pt'
                if path.exists():assert torch.load(path,weights_only=True)['source']==s;continue
                initial=variant if variant in ('visual','random_codes') else 'text';ip=folder/f'initial_{initial}.pt'
                if not ip.exists():
                    seed_all(seed);model=Head(name,initial,bank,c).cuda();torch.save(dict(source=s,**train(model,d,ti,vi,c,seed)),ip)
                ck=torch.load(ip,weights_only=True);assert ck['source']==s
                seed_all(seed);model=Head(name,variant,bank,c).cuda();state=ck['state_dict']
                if model.ag:state={k:v for k,v in state.items() if k.startswith(('projection.','logit_scale','class_text'))}
                model.load_state_dict(state,strict=not model.ag)
                result=train(model,d,ti,vi,c,seed)
                torch.save(dict(source=s,backbone=name,seed=seed,variant=variant,parameters=sum(p.numel() for p in model.parameters()),train_paths=[d['rows'][i]['path'] for i in ti],**result),path)
                print('trained',name,seed,variant,'val',result['validation']['top1_accuracy'],flush=True)
        del d;torch.cuda.empty_cache()

def seal(c,s):
    records=[]
    for n in c['backbones']:
        for seed in c['seeds']:
            for v in c['variants']:
                path=RUN/f'{n}_{seed}/{v}.pt';ck=torch.load(path,weights_only=True);assert ck['source']==s
                records.append(dict(backbone=n,seed=seed,variant=v,checkpoint_sha256=digest(path),validation=ck['validation'],step=ck['best_step']))
    assert len(records)==42
    assets=[CACHE/'text.pt',CACHE/'regions_development.json']+[CACHE/f'{n}_development.pt' for n in c['backbones']]
    obj=dict(source=s,checkpoints=records,development_assets={str(p.relative_to(ROOT)):digest(p) for p in assets})
    path=OUT/'selection_locked.json'
    if path.exists():assert json.loads(path.read_text())['checkpoints']==records;check_seal(s)
    else:write_json(path,obj|dict(locked_at_utc=datetime.now(timezone.utc).isoformat()))

def score(c,bank,s):
    check_seal(s);path=OUT/'results.json'
    if path.exists():assert json.loads(path.read_text())['source']==s;return
    rows=[];predictions={}
    for n in c['backbones']:
        d=cache(n,'test');ix=split_indices(d,'test');assert len(ix)==200
        for seed in c['seeds']:
            for v in c['variants']:
                ck=torch.load(RUN/f'{n}_{seed}/{v}.pt',weights_only=True);model=Head(n,v,bank,c).cuda();model.load_state_dict(ck['state_dict'])
                metrics,p,y=evaluate(model,d,ix)
                rows.append(dict(backbone=n,seed=seed,variant=v,training_accuracy=ck['training']['top1_accuracy'],parameters=ck['parameters'],**metrics))
                predictions[f'{n}/{seed}/{v}']=dict(probabilities=p,labels=y,paths=[d['rows'][i]['path'] for i in ix])
        print('final evaluation complete',n,flush=True)
    torch.save(predictions,RUN/'predictions.pt');write_json(path,dict(source=s,scored_at_utc=datetime.now(timezone.utc).isoformat(),rows=rows))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['manifest','development','train','test','all']);a=parser.parse_args().action
    CACHE.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);c=json.loads(CONFIG.read_text());m=prepare_manifest(c);s=lock(c,m)
    if a=='manifest':return
    b=text_bank(m,s)
    if a in ('development','all'):boxes=ground(c,m,s,'development');extract(c,m,s,'development',boxes)
    if a in ('train','all'):train_all(c,b,s);seal(c,s)
    if a in ('test','all'):check_seal(s);boxes=ground(c,m,s,'test');extract(c,m,s,'test',boxes);score(c,b,s)

if __name__=='__main__':main()
