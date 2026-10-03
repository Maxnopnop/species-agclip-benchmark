"""Fresh project-unused CUB species; sealed test feature extraction."""
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

CONFIG=ROOT/'configs/confidence_fresh_v1.json'
DATA=ROOT/'data/confidence_fresh_v1'
CACHE=ROOT/'cache/confidence_fresh_v1'
OUT=ROOT/'reports/confidence_fresh_v1'
RUN=ROOT/'runs/confidence_fresh_v1'
CUB=ROOT/'data/cub/CUB_200_2011'

def fingerprint(path):
    with Image.open(path) as im:
        im=im.convert('RGB');pixel=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
        gray=im.convert('L')
        a=np.asarray(gray.resize((9,8),Image.Resampling.LANCZOS));dbits=(a[:,1:]>a[:,:-1]).flatten()
        spectrum=dctn(np.asarray(gray.resize((32,32),Image.Resampling.LANCZOS),dtype=float),norm='ortho')[:8,:8].flatten()
        pbits=spectrum>np.median(spectrum[1:]);pbits[0]=False
    bits=lambda x:int.from_bytes(np.packbits(x).tobytes(),'big')
    return dict(sha256=digest(path),pixels_sha256=pixel,phash=bits(pbits),dhash=bits(dbits))

def duplicates(a,refs,threshold):
    return any(a['sha256']==b['sha256'] or a['pixels_sha256']==b['pixels_sha256'] or
               (a['phash']^b['phash']).bit_count()<=threshold or (a['dhash']^b['dhash']).bit_count()<=threshold for b in refs)

def index_file(name,convert=str):
    return {int(line.split(' ',1)[0]):convert(line.strip().split(' ',1)[1]) for line in (CUB/name).read_text().splitlines()}

def prepare_manifest(c):
    target=DATA/'manifest.json'
    if target.exists():return json.loads(target.read_text())
    prior={};used=set()
    for path in sorted((ROOT/'data').rglob('*.json')):
        if DATA in path.parents or not ('manifest' in path.name or path.name=='catalog.json'):continue
        obj=json.loads(path.read_text());ids={x['source_id'] for x in obj.get('classes',[]) if 'source_id' in x}
        if ids:used.update(ids);prior[str(path.relative_to(ROOT))]=digest(path)
    assert len(used)==120,'Historical class usage changed; review sampling before proceeding.'
    files=index_file('images.txt');labels=index_file('image_class_labels.txt',int);official=index_file('train_test_split.txt',int);names=index_file('classes.txt')
    historical=[CUB/'images'/files[i] for i in files if labels[i] in used]
    for folder in [ROOT/'data/expanded20/images',ROOT/'data/pilot/images',ROOT/'data/images']:
        historical+=sorted(folder.rglob('*.jpg'))
    historical=sorted(set(historical));refs=[];audit_path=CACHE/'historical_fingerprints.json'
    CACHE.mkdir(parents=True,exist_ok=True)
    if audit_path.exists():
        saved=json.loads(audit_path.read_text());assert saved['paths']==[str(p.relative_to(ROOT)) for p in historical];refs=saved['fingerprints']
    else:
        for j,path in enumerate(historical):
            refs.append(fingerprint(path))
            if j%1000==0:print('historical fingerprint',j,'/',len(historical),flush=True)
        write_json(audit_path,dict(paths=[str(p.relative_to(ROOT)) for p in historical],fingerprints=refs))
    eligible=[i for i in names if i not in used and sum(labels[j]==i and official[j]==1 for j in files)>=30 and sum(labels[j]==i and official[j]==0 for j in files)>=20]
    random.Random(c['sampling_seed']).shuffle(eligible)
    rows=[];classes=[];newrefs=[];rejected=[];skipped=[]
    for cid in eligible:
        candidates={s:[i for i in files if labels[i]==cid and official[i]==s] for s in (1,0)}
        for s in (1,0):random.Random(c['sampling_seed']+cid*2+s).shuffle(candidates[s])
        selected={};localrefs=[]
        for s,needed in ((1,30),(0,20)):
            selected[s]=[]
            for iid in candidates[s]:
                fp=fingerprint(CUB/'images'/files[iid])
                if duplicates(fp,refs+newrefs+localrefs,c['dedup_hamming_threshold']):
                    rejected.append(dict(image_id=iid,source_id=cid,reason='exact/pixel/perceptual threshold'));continue
                localrefs.append(fp);selected[s].append((iid,fp))
                if len(selected[s])==needed:break
        if len(selected[1])<30 or len(selected[0])<20:skipped.append(cid);continue
        label=len(classes);classes.append(dict(label=label,source_id=cid,name=names[cid].split('.',1)[1].replace('_',' ')))
        for s in (1,0):
            for j,(iid,fp) in enumerate(selected[s]):
                role='test' if s==0 else 'train' if j<20 else 'val'
                rows.append(dict(image_id=iid,path=files[iid],label=label,source_id=cid,split=role,official_train=bool(s),**fp))
        newrefs+=localrefs
        if len(classes)==10:break
    assert len(classes)==10 and len(rows)==500,'Insufficient unused duplicate-screened images.'
    attrs=json.loads((ROOT/'data/cub_attributes_v1/manifest.json').read_text())['attributes']
    m=dict(classes=classes,attributes=[dict(source_id=a['source_id'],prompt=a['prompt']) for a in attrs],rows=rows,image_root=str(CUB/'images'),prior_manifests=prior,excluded_species=sorted(used))
    write_json(target,m)
    write_json(OUT/'data_audit.json',dict(historical_classes=len(used),historical_images_fingerprinted=len(refs),selected_classes=classes,images=len(rows),split_counts={s:sum(r['split']==s for r in rows) for s in ['train','val','test']},rejected_candidates=rejected,insufficient_classes=skipped,threshold=c['dedup_hamming_threshold'],exact_and_perceptual_overlap_after_filter=0,prior_manifests=prior,source='Verified local official CUB-200-2011, data/cub/verified.json; no new downloads. Perceptual screening is heuristic.'))
    return m

def source(m):
    paths=[CONFIG,Path(__file__),ROOT/'confidence_experiment.py',ROOT/'multimodal/models.py',ROOT/'multimodal/grounded_data.py',ROOT/'multimodal/expanded.py',ROOT/'multimodal/experiment.py',ROOT/'run.py',ROOT/'data/cub/verified.json',ROOT/'data/cub_attributes_v1/manifest.json',DATA/'manifest.json']
    paths += [ROOT/p for p in m['prior_manifests']]
    for p,h in m['prior_manifests'].items():assert digest(ROOT/p)==h
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
    torch.set_num_threads(4);c=json.loads(CONFIG.read_text());m=prepare_manifest(c);s=lock(c,m)
    if a=='manifest':return
    b=text_bank(m,s)
    if a in ('development','all'):boxes=ground(c,m,s,'development');extract(c,m,s,'development',boxes)
    if a in ('train','all'):train_all(c,b,s);seal(c,s)
    if a in ('test','all'):check_seal(s);boxes=ground(c,m,s,'test');extract(c,m,s,'test',boxes);score(c,b,s)

if __name__=='__main__':main()
