"""Predeclared 100-class follow-up, excluding all 20 previously inspected species."""
import gc,json,random
from collections import defaultdict
from pathlib import Path
import torch
from PIL import Image
from torch.nn import functional as F
from data_tools import ROOT,digest,write_json
from .cub_data import BASE,lines
from .cub_rich_data import PARTS
from .grounded_data import detector_parts,locate,image_tensors
from .visible_data import model_parts

VERSION='cub100_v1';DATA=ROOT/'data'/VERSION;CACHE=ROOT/'cache'/VERSION


def source():return dict(code_sha256=digest(__file__),config_sha256=digest(ROOT/f'configs/{VERSION}.json'))


def prepare():
    DATA.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    p=json.loads((ROOT/f'configs/{VERSION}.json').read_text());path=DATA/'manifest.json'
    if path.exists():
        m=json.loads(path.read_text());assert m['source']==source();return p,m
    verified=json.loads((ROOT/'data/cub/verified.json').read_text());assert verified['official_md5']=='97eceeb196236b17998738112f37df78'
    old=json.loads((ROOT/'data/cub_attributes_v1/manifest.json').read_text());excluded={c['source_id'] for c in old['classes']}
    names={int(r[0]):r[1].split('.',1)[1].replace('_',' ') for r in lines(BASE/'classes.txt')}
    images={int(r[0]):r[1] for r in lines(BASE/'images.txt')};labels={int(r[0]):int(r[1]) for r in lines(BASE/'image_class_labels.txt')};split={int(r[0]):int(r[1]) for r in lines(BASE/'train_test_split.txt')}
    by_class=defaultdict(lambda:defaultdict(list))
    for iid,cl in labels.items():by_class[cl][split[iid]].append(iid)
    eligible=[cl for cl in sorted(names) if cl not in excluded and len(by_class[cl][1])>=15 and len(by_class[cl][0])>=10]
    selected=random.Random(p['class_selection_seed']).sample(eligible,100);rows=[]
    for label,cl in enumerate(selected):
        train=sorted(by_class[cl][1]);test=sorted(by_class[cl][0]);random.Random(1000+cl).shuffle(train);random.Random(2000+cl).shuffle(test)
        assignments=[('train_seen',train[:10]),('dev_seen',train[10:15]),('eval_seen',test[:10])] if label in p['seen_classes'] else [('dev_unseen',train[:10])] if label in p['dev_unseen_classes'] else [('eval_unseen',test[:10])]
        for role,ids in assignments:
            for iid in ids:
                image=BASE/'images'/images[iid]
                with Image.open(image) as im:size=list(im.size)
                rows.append(dict(image_id=iid,path=images[iid],label=label,role=role,size=size,sha256=digest(image)))
    assert len(rows)==1750 and len({r['sha256'] for r in rows})==1750 and set(selected).isdisjoint(excluded)
    index={r['image_id']:i for i,r in enumerate(rows)};parts=defaultdict(dict)
    for r in lines(BASE/'parts/part_locs.txt'):
        if int(r[0]) in index:parts[int(r[0])][int(r[1])]=(float(r[2]),float(r[3]),int(r[4]))
    attribute_names={int(r[0]):r[1] for r in lines(ROOT/'data/cub/attributes.txt')};part_ids={}
    for aid,name in attribute_names.items():
        field=name.split('::')[0][4:];key=next((k for k in sorted(PARTS,key=len,reverse=True) if field==k or field.startswith(k+'_')),None)
        part_ids[aid]=PARTS[key] if key is not None else []
    raw=torch.full((len(rows),312),-1.)
    for r in lines(BASE/'attributes/image_attribute_labels.txt'):
        iid,aid,positive,certainty=map(int,r[:4])
        if iid in index and certainty>=3 and any(parts[iid].get(pid,(0,0,0))[2] for pid in part_ids[aid]):raw[index[iid],aid-1]=positive
    global_targets=raw.clone()
    for row in rows:
        iid=row['image_id'];w,h=row['size'];side=min(w,h);visible={pid for pid,(x,y,ok) in parts[iid].items() if ok and (w-side)/2<=x<=(w+side)/2 and (h-side)/2<=y<=(h+side)/2}
        for aid in attribute_names:
            if not visible & set(part_ids[aid]):global_targets[index[iid],aid-1]=-1
        row['parts']={str(k):list(v) for k,v in parts[iid].items()}
    ti=[i for i,r in enumerate(rows) if r['role']=='train_seen'];support=((global_targets[ti]==1).sum(0)>=8)&((global_targets[ti]==0).sum(0)>=20)
    chosen=(support.nonzero().flatten()+1).tolist();assert len(chosen)>24
    attributes=[dict(source_id=aid,name=attribute_names[aid],part_ids=part_ids[aid]) for aid in chosen]
    torch.save(dict(raw_targets=raw[:,[aid-1 for aid in chosen]],source=source()),DATA/'raw_targets.pt')
    m=dict(source=source(),classes=[dict(label=i,source_id=cl,name=names[cl]) for i,cl in enumerate(selected)],attributes=attributes,rows=rows,image_root=str(BASE/'images'),excluded_prior_source_ids=sorted(excluded))
    write_json(path,m);print(f'Locked 100 new project species, 1750 photos, {len(attributes)} train-supported attributes.',flush=True);return p,m


def prompt(name):
    field,value=name.split('::');field=field[4:];value=value.replace('_',' ').replace('(','').replace(')','')
    if field.endswith('_color'):return f'a bird with {value} {field[:-6].replace("_"," ")} color.'
    if field.endswith('_pattern'):return f'a bird with a {value} pattern on its {field[:-8].replace("_"," ")}.'
    if field.endswith('_shape'):return f'a bird with a {value} {field[:-6].replace("_"," ")} shape.'
    if field=='bill_length':return f'a bird with a bill {value}.'
    if field=='shape':return f'a bird with a {value} body shape.'
    raise ValueError(field)


def text_bank():
    p,m=prepare();path=CACHE/'text.pt';metadata=dict(**source(),manifest_sha256=digest(DATA/'manifest.json'),revision=json.loads((ROOT/'configs/visible_model_revisions.json').read_text())['clip_b16'])
    if path.exists():
        saved=torch.load(path,weights_only=True);assert saved['metadata']==metadata;return saved
    positives=[prompt(a['name']) for a in m['attributes']];negatives=[s.replace('a bird with','a bird without') for s in positives]
    texts=[f'a photo of a {cl["name"]}.' for cl in m['classes']]+positives+negatives
    model,tokenizer,_=model_parts('clip_b16');model=model.cuda().eval()
    with torch.no_grad():
        features=[]
        for start in range(0,len(texts),32):
            batch=tokenizer(texts[start:start+32],padding='max_length',max_length=77,truncation=True,return_tensors='pt').to('cuda');features.append(F.normalize(model.get_text_features(**batch).float(),dim=-1).cpu())
    text=torch.cat(features);a=len(positives);result=dict(metadata=metadata,class_text=text[:100],attribute_text=text[100:100+a],negative_text=text[100+a:],positive_prompts=positives,negative_prompts=negatives)
    torch.save(result,path);del model;gc.collect();torch.cuda.empty_cache();return result


def regions(stage):
    p,m=prepare();roles=['train_seen','dev_seen','dev_unseen'] if stage=='development' else ['eval_seen','eval_unseen'];rows=[r for r in m['rows'] if r['role'] in roles]
    path=CACHE/f'regions_{stage}.json';metadata=dict(**source(),manifest_sha256=digest(DATA/'manifest.json'),grounding_sha256=digest(ROOT/'multimodal/grounded_data.py'))
    saved=json.loads(path.read_text()) if path.exists() else dict(metadata=metadata,images={});assert saved['metadata']==metadata
    todo=[r for r in rows if r['path'] not in saved['images']]
    if todo:
        processor,model=detector_parts()
        for i,row in enumerate(todo):
            image=Path(m['image_root'])/row['path'];assert digest(image)==row['sha256']
            with Image.open(image) as im:record=locate(im.convert('RGB'),p['detector_prompts'],processor,model,p)
            saved['images'][row['path']]=record
            if (i+1)%25==0:write_json(path,saved);print(f'100-class grounding {stage}: {i+1}/{len(todo)} new photos',flush=True)
        write_json(path,saved);del model,processor;gc.collect();torch.cuda.empty_cache()
    return saved


def load_data(stage):
    p,m=prepare();path=CACHE/f'features_{stage}.pt';metadata=dict(**source(),manifest_sha256=digest(DATA/'manifest.json'),revision=json.loads((ROOT/'configs/visible_model_revisions.json').read_text())['clip_b16'])
    if path.exists():
        saved=torch.load(path,weights_only=True);assert saved['metadata']==metadata;return saved
    region=regions(stage);roles=['train_seen','dev_seen','dev_unseen'] if stage=='development' else ['eval_seen','eval_unseen'];rows=[r for r in m['rows'] if r['role'] in roles]
    raw=torch.load(DATA/'raw_targets.pt',weights_only=True);assert raw['source']==source();index={r['image_id']:i for i,r in enumerate(m['rows'])}
    model,_,processor=model_parts('clip_b16');model=model.cuda().eval()
    from open_clip.transform import image_transform
    transform=image_transform(224,is_train=False,mean=processor.image_mean,std=processor.image_std,interpolation='bicubic',resize_mode='shortest')
    features=[];valid=[];targets=[]
    with torch.no_grad():
        for i,row in enumerate(rows):
            record=region['images'][row['path']];images,_,_,ok=image_tensors(Path(m['image_root'])/row['path'],record,transform)
            with torch.autocast('cuda',dtype=torch.bfloat16):f=F.normalize(model.get_image_features(pixel_values=images.cuda()).float(),dim=-1).cpu()
            features.append(f);valid.append(ok);view_targets=[]
            for x1,y1,x2,y2 in [[0,0,*row['size']],*record['boxes']]:
                cx,cy=(x1+x2)/2,(y1+y2)/2;side=min(x2-x1,y2-y1);visible={int(pid) for pid,(x,y,shown) in row['parts'].items() if shown and cx-side/2<=x<=cx+side/2 and cy-side/2<=y<=cy+side/2}
                y=raw['raw_targets'][index[row['image_id']]].clone()
                for a,attribute in enumerate(m['attributes']):
                    if not visible & set(attribute['part_ids']):y[a]=-1
                view_targets.append(y)
            while len(view_targets)<3:view_targets.append(torch.full_like(view_targets[0],-1))
            targets.append(torch.stack(view_targets))
            if i%100==0:print(f'100-class B16 features {stage}: {i}/{len(rows)}',flush=True)
    result=dict(metadata=metadata,rows=rows,native_views=torch.stack(features),valid=torch.stack(valid),targets=torch.stack(targets),labels=torch.tensor([r['label'] for r in rows]))
    torch.save(result,path);del model;gc.collect();torch.cuda.empty_cache();return result
