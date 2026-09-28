"""Repeated class partitions with fresh probes and train-only vocabularies."""
import gc,json,random
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from data_tools import ROOT,digest,write_json
from .cub_data import BASE,lines
from .cub_rich_data import PARTS
from . import cub100_data as parent
from . import cub_siglip_data as siglip_parent
from .visible_data import model_parts
from .grounded_data import detector_parts,locate,image_tensors

BASE_VERSION='cub_resplit_v1';DATA=ROOT/'data'/BASE_VERSION;CACHE=ROOT/'cache'/BASE_VERSION
SPLIT=20260929;BACKBONE='clip_b16'


def source():
    return dict(code_sha256=digest(__file__),config_sha256=digest(ROOT/f'configs/{BASE_VERSION}.json'),parent_manifest_sha256=digest(ROOT/'data/cub100_v1/manifest.json'))


def catalog():
    DATA.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True);path=DATA/'catalog.json';meta=source()
    if path.exists():
        m=json.loads(path.read_text());assert m['source']==meta;return m
    _,old=parent.prepare();images={int(r[0]):r[1] for r in lines(BASE/'images.txt')};labels={int(r[0]):int(r[1]) for r in lines(BASE/'image_class_labels.txt')};splits={int(r[0]):int(r[1]) for r in lines(BASE/'train_test_split.txt')};byclass=defaultdict(lambda:defaultdict(list))
    for iid,cl in labels.items():byclass[cl][splits[iid]].append(iid)
    rows=[]
    for cl in old['classes']:
        train=sorted(byclass[cl['source_id']][1]);test=sorted(byclass[cl['source_id']][0]);random.Random(1000+cl['source_id']).shuffle(train);random.Random(2000+cl['source_id']).shuffle(test)
        for official,ids in [('train',train[:15]),('test',test[:10])]:
            for position,iid in enumerate(ids):
                file=BASE/'images'/images[iid]
                with Image.open(file) as im:size=list(im.size)
                rows.append(dict(image_id=iid,path=images[iid],label=cl['label'],official_split=official,position=position,size=size,sha256=digest(file)))
    assert len(rows)==2500 and len({r['sha256'] for r in rows})==2500
    assert {r['image_id'] for r in old['rows']}<=set(r['image_id'] for r in rows)
    index={r['image_id']:i for i,r in enumerate(rows)};parts=defaultdict(dict)
    for r in lines(BASE/'parts/part_locs.txt'):
        if int(r[0]) in index:parts[int(r[0])][int(r[1])]=[float(r[2]),float(r[3]),int(r[4])]
    attributes=[]
    for r in lines(ROOT/'data/cub/attributes.txt'):
        aid,name=int(r[0]),r[1];field=name.split('::')[0][4:];key=next((k for k in sorted(PARTS,key=len,reverse=True) if field==k or field.startswith(k+'_')),None)
        attributes.append(dict(source_id=aid,name=name,part_ids=PARTS[key] if key else []))
    raw=torch.full((2500,312),-1.)
    for r in lines(BASE/'attributes/image_attribute_labels.txt'):
        iid,aid,pos,certainty=map(int,r[:4])
        if iid in index and certainty>=3 and any(parts[iid].get(pid,[0,0,0])[2] for pid in attributes[aid-1]['part_ids']):raw[index[iid],aid-1]=pos
    for row in rows:row['parts']={str(k):v for k,v in parts[row['image_id']].items()}
    torch.save(dict(source=meta,targets=raw),DATA/'raw_targets.pt');m=dict(source=meta,classes=old['classes'],rows=rows,attributes=attributes,image_root=old['image_root']);write_json(path,m);return m


def regions():
    m=catalog();path=CACHE/'regions.json';meta=source()
    if path.exists():saved=json.loads(path.read_text());assert saved['source']==meta
    else:
        saved=dict(source=meta,images={})
        for stage in ['development','evaluation']:saved['images'].update(parent.regions(stage)['images'])
    todo=[r for r in m['rows'] if r['path'] not in saved['images']]
    if todo:
        p,_=parent.prepare();processor,model=detector_parts()
        for i,row in enumerate(todo):
            file=Path(m['image_root'])/row['path'];assert digest(file)==row['sha256']
            with Image.open(file) as im:saved['images'][row['path']]=locate(im.convert('RGB'),p['detector_prompts'],processor,model,p)
            if (i+1)%50==0:write_json(path,saved);print(f'Resplit new grounding: {i+1}/{len(todo)}',flush=True)
        write_json(path,saved);del model,processor;gc.collect();torch.cuda.empty_cache()
    return saved


def feature_catalog():
    m=catalog();path=CACHE/f'{BACKBONE}_features.pt';meta=dict(**source(),backbone=BACKBONE,revision=json.loads((ROOT/'configs/visible_model_revisions.json').read_text())[BACKBONE])
    if path.exists():
        saved=torch.load(path,weights_only=True);assert saved['metadata']==meta;return saved
    region=regions();old={};loader=parent.load_data if BACKBONE=='clip_b16' else siglip_parent.load_data
    for stage in ['development','evaluation']:
        d=loader(stage)
        for i,row in enumerate(d['rows']):old[row['path']]=(d['native_views'][i],d['valid'][i])
    del d;model,_,processor=model_parts(BACKBONE);model=model.cuda().eval()
    if BACKBONE=='clip_b16':
        from open_clip.transform import image_transform
        transform=image_transform(224,is_train=False,mean=processor.image_mean,std=processor.image_std,interpolation='bicubic',resize_mode='shortest')
    raw=torch.load(DATA/'raw_targets.pt',weights_only=True);assert raw['source']==source();features=[];targets=[];valid=[];new=0
    with torch.no_grad():
        for i,row in enumerate(m['rows']):
            record=region['images'][row['path']]
            if row['path'] in old:f,ok=old[row['path']]
            else:
                file=Path(m['image_root'])/row['path'];assert digest(file)==row['sha256']
                if BACKBONE=='clip_b16':pixels,_,_,ok=image_tensors(file,record,transform)
                else:pixels,ok=siglip_parent.photo_views(file,record,processor)
                with torch.autocast('cuda',dtype=torch.bfloat16):f=F.normalize(model.get_image_features(pixel_values=pixels.cuda()).float(),dim=-1).cpu()
                new+=1
                if new%100==0:print(f'Resplit {BACKBONE}: {new}/750 new encodings',flush=True)
            features.append(f);valid.append(ok);vv=[]
            for x1,y1,x2,y2 in [[0,0,*row['size']],*record['boxes']]:
                if BACKBONE=='clip_b16':cx,cy=(x1+x2)/2,(y1+y2)/2;side=min(x2-x1,y2-y1);x1,y1,x2,y2=cx-side/2,cy-side/2,cx+side/2,cy+side/2
                visible={int(pid) for pid,(x,y,shown) in row['parts'].items() if shown and x1<=x<=x2 and y1<=y<=y2};target=raw['targets'][i].clone()
                for a,att in enumerate(m['attributes']):
                    if not visible&set(att['part_ids']):target[a]=-1
                vv.append(target)
            while len(vv)<3:vv.append(torch.full_like(vv[0],-1))
            targets.append(torch.stack(vv))
    result=dict(metadata=meta,native_views=torch.stack(features),valid=torch.stack(valid),targets=torch.stack(targets));torch.save(result,path);del model;gc.collect();torch.cuda.empty_cache();return result


def prepare():
    m=catalog();order=list(range(100));random.Random(SPLIT).shuffle(order);p=json.loads((ROOT/f'configs/{BASE_VERSION}.json').read_text());p.update(seen_classes=sorted(order[:50]),dev_unseen_classes=sorted(order[50:75]),eval_unseen_classes=sorted(order[75:]))
    rows=[];indices=[]
    for i,row in enumerate(m['rows']):
        label=row['label'];role=None
        if label in p['seen_classes']:role='eval_seen' if row['official_split']=='test' else 'train_seen' if row['position']<10 else 'dev_seen'
        elif label in p['dev_unseen_classes'] and row['official_split']=='train' and row['position']<10:role='dev_unseen'
        elif label in p['eval_unseen_classes'] and row['official_split']=='test':role='eval_unseen'
        if role:rows.append(dict(row,role=role));indices.append(i)
    assert len(rows)==1750;features=feature_catalog();train=[i for i,r in zip(indices,rows) if r['role']=='train_seen'];target=features['targets'][train,0];columns=(((target==1).sum(0)>=8)&((target==0).sum(0)>=20)).nonzero().flatten().tolist()
    assert len(columns)>24 and len(train)==500
    manifest=dict(source=source(),split_seed=SPLIT,backbone=BACKBONE,classes=m['classes'],rows=rows,indices=indices,columns=columns,attributes=[m['attributes'][a] for a in columns],image_root=m['image_root'])
    return p,manifest


def load_data(stage):
    p,m=prepare();features=feature_catalog();roles=['train_seen','dev_seen','dev_unseen'] if stage=='development' else ['eval_seen','eval_unseen'];chosen=[i for i,r in enumerate(m['rows']) if r['role'] in roles];index=[m['indices'][i] for i in chosen];rows=[m['rows'][i] for i in chosen]
    assert set(r['label'] for r in rows).isdisjoint(p['eval_unseen_classes'] if stage=='development' else p['dev_unseen_classes'])
    return dict(native_views=features['native_views'][index],valid=features['valid'][index],targets=features['targets'][index][:,:,m['columns']],labels=torch.tensor([r['label'] for r in rows]),rows=rows)


def text_bank():
    p,m=prepare();path=CACHE/f'{BACKBONE}_{SPLIT}_text.pt';meta=dict(**source(),split_seed=SPLIT,backbone=BACKBONE,columns=m['columns'])
    if path.exists():
        saved=torch.load(path,weights_only=True);assert saved['metadata']==meta;return saved
    positives=[parent.prompt(a['name']) for a in m['attributes']];negatives=[s.replace('a bird with','a bird without') for s in positives];texts=[f'a photo of a {c["name"]}.' for c in m['classes']]+positives+negatives
    if BACKBONE=='siglip2_b16':texts=[s.lower() for s in texts]
    model,tokenizer,_=model_parts(BACKBONE);model=model.cuda().eval();result=[]
    with torch.no_grad():
        for i in range(0,len(texts),32):
            t=tokenizer(texts[i:i+32],padding='max_length',max_length=77 if BACKBONE=='clip_b16' else 64,truncation=True,return_tensors='pt').to('cuda');result.append(F.normalize(model.get_text_features(**t).float(),dim=-1).cpu())
    text=torch.cat(result);a=len(positives);saved=dict(metadata=meta,class_text=text[:100],attribute_text=text[100:100+a],negative_text=text[100+a:],positive_prompts=positives,negative_prompts=negatives,columns=m['columns']);torch.save(saved,path);del model;gc.collect();torch.cuda.empty_cache();return saved
