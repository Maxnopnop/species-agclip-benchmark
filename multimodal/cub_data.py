"""CUB image-level attributes; species-disjoint pilot with no oracle inputs."""
import json,random,re,time
from collections import defaultdict,Counter
from pathlib import Path
import torch
from PIL import Image
from torch.nn import functional as F
from data_tools import ROOT,digest,write_json
from .models import load_clip
from .grounded_data import detector_parts,locate,image_tensors

VERSION='cub_attributes_v1';BASE=ROOT/'data/cub/CUB_200_2011';OUT=ROOT/f'runs/{VERSION}';REPORT=ROOT/f'reports/{VERSION}'
GROUPS={
 'has_wing_color':(['left wing','right wing'],'a bird with {value} wings.'),
 'has_bill_shape':(['beak'],'a bird with a {value} bill.'),
 'has_breast_color':(['breast'],'a bird with a {value} breast.'),
 'has_back_color':(['back'],'a bird with a {value} back.'),
 'has_belly_color':(['belly'],'a bird with a {value} belly.'),
 'has_crown_color':(['crown'],'a bird with a {value} crown.'),
 'has_wing_pattern':(['left wing','right wing'],'a bird with {value} wings.'),
 'has_breast_pattern':(['breast'],'a bird with a {value} breast.')}
DETECT_PROMPTS=['a bird head.','a bird wing.','a bird breast.','a bird tail.','a bird.']

def lines(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():yield line.split()

def fixed(path,value):
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=value:raise ValueError(f'Changed versioned definition: {path}')
    write_json(path,value)

def prepare():
    target=ROOT/f'data/{VERSION}/manifest.json';config=ROOT/f'configs/{VERSION}.json'
    if target.exists():return json.loads(config.read_text(encoding='utf-8')),json.loads(target.read_text(encoding='utf-8'))
    verified=json.loads((ROOT/'data/cub/verified.json').read_text(encoding='utf-8'));assert verified['official_md5']=='97eceeb196236b17998738112f37df78'
    classes={int(r[0]):' '.join(r[1:]).split('.',1)[1].replace('_',' ') for r in lines(BASE/'classes.txt')}
    images={int(r[0]):r[1] for r in lines(BASE/'images.txt')};labels={int(r[0]):int(r[1]) for r in lines(BASE/'image_class_labels.txt')};split={int(r[0]):int(r[1]) for r in lines(BASE/'train_test_split.txt')}
    by_class=defaultdict(lambda:defaultdict(list))
    for i,c in labels.items():by_class[c][split[i]].append(i)
    groups=defaultdict(list)
    for c,name in classes.items():
        if len(by_class[c][1])>=30 and len(by_class[c][0])>=20:groups[name.split()[-1]].append(c)
    groups={g:cs for g,cs in groups.items() if len(cs)>=2};rng=random.Random(20260928)
    chosen_groups=rng.sample(sorted(groups),10);pairs=[rng.sample(sorted(groups[g]),2) for g in chosen_groups]
    selected=[c for pair in pairs for c in pair];mapping={c:i for i,c in enumerate(selected)};seen=list(range(0,20,2));partners=list(range(1,20,2));dev=sorted(rng.sample(partners,4));test=sorted(set(partners)-set(dev))
    rows=[]
    for c in selected:
        label=mapping[c];train=sorted(by_class[c][1]);evaluation=sorted(by_class[c][0]);random.Random(1000+c).shuffle(train);random.Random(2000+c).shuffle(evaluation)
        assignments=[('train_seen',train[:20]),('dev_seen',train[20:30]),('eval_seen',evaluation[:20])] if label in seen else [('dev_unseen',train[:20])] if label in dev else [('eval_unseen',evaluation[:20])]
        for role,ids in assignments:
            for i in ids:
                path=BASE/'images'/images[i]
                with Image.open(path) as im:size=list(im.size)
                rows.append(dict(image_id=i,path=images[i],label=label,role=role,size=size,sha256=digest(path)))
    assert len(rows)==700 and len({r['sha256'] for r in rows})==700
    wanted={r['image_id'] for r in rows};part_names={int(r[0]):' '.join(r[1:]).lower().replace('_',' ') for r in lines(BASE/'parts/parts.txt')};part_ids={v:k for k,v in part_names.items()};parts=defaultdict(dict)
    for r in lines(BASE/'parts/part_locs.txt'):
        i=int(r[0])
        if i in wanted:parts[i][int(r[1])]=[float(r[2]),float(r[3]),int(r[4])]
    attribute_path=BASE/'attributes/attributes.txt'
    if not attribute_path.exists():attribute_path=ROOT/'data/cub/attributes.txt'
    attributes={int(r[0]):r[1] for r in lines(attribute_path)};observations=defaultdict(dict)
    for r in lines(BASE/'attributes/image_attribute_labels.txt'):
        i=int(r[0])
        if i in wanted:observations[i][int(r[1])]=(int(r[2]),int(r[3]))
    def target_for(i,a):
        group=attributes[a].split('::')[0];visibility=any(parts[i].get(part_ids[name],[0,0,0])[2] for name in GROUPS[group][0]);value,certainty=observations[i][a]
        return value if certainty>=3 and visibility else -1
    training=[r for r in rows if r['role']=='train_seen'];chosen=[];support=[]
    for group,(names,template) in GROUPS.items():
        candidates=[]
        for a,name in attributes.items():
            if name.split('::')[0]!=group:continue
            values=[target_for(r['image_id'],a) for r in training];pos=values.count(1);neg=values.count(0)
            if pos>=8 and neg>=20:candidates.append((min(pos,neg),a,pos,neg))
        for _,a,pos,neg in sorted(candidates,key=lambda x:(-x[0],x[1]))[:3]:
            value=attributes[a].split('::')[1].replace('_',' ').replace('(','').replace(')','');prompt=template.format(value=value)
            chosen.append(dict(source_id=a,name=attributes[a],prompt=prompt,negative_prompt=prompt.replace('a bird with','a bird without'),part_ids=[part_ids[n] for n in names]))
            support.append(dict(source_id=a,name=attributes[a],positive=pos,negative=neg))
    assert len(chosen)>=16,f'Insufficient supported attributes: {len(chosen)}'
    for row in rows:
        i=row['image_id'];row['attributes']=[target_for(i,a['source_id']) for a in chosen]
        row['attribute_points']=[[[x,y] for pid in a['part_ids'] for x,y,v in [parts[i].get(pid,[0,0,0])] if v] for a in chosen]
    protocol=dict(version=VERSION,seen_classes=seen,dev_unseen_classes=dev,eval_unseen_classes=test,class_selection_seed=20260928,
        variants=['finetune','region_only','automatic','gold','shuffled','gold_region'],seeds=[42],learning_rates=[1e-5,3e-6],updates=160,validation_steps=[0,40,80,120,160],micro_batch=2,effective_batch=16,trainable_visual_blocks=2,fusion_lr_multiplier=10,
        class_ce_weight=.5,contrastive_weight=.5,preserve_weight=.5,attribute_loss_weight=1.,regional_attribute_weight=.5,
        detector_threshold=.05,detector_nms_iou=.5,min_region_area=.005,max_region_area=.95,max_regions=2,
        attribute_selection='At most three attributes per predeclared group, training-only support >=8 positive and >=20 negative, ranked by min support. Confidence >=3 and corresponding part visible; unknown labels masked. No held-out support used to choose attributes.',
        selection='Development GZSL H; tie by mean U/S then lower CE. Attribute-best checkpoint separately selected by development attribute mAP for diagnosis only. Final images evaluated only after all selections are locked.',
        scope='Fresh project dataset, CUB may overlap with pretrained CLIP/ImageNet. 20 classes, 200 training photos, 180 development photos, 320 final photos. Common-name suffix grouping is a sampling heuristic, not a taxonomy claim. Single-seed mechanistic pilot, not paper reproduction.',
        inference='Image-only; same five fixed anatomical detector queries for all photos. Ground-truth image attributes/parts only construct losses and evaluation targets. Predicted attributes are fused during BOTH training and inference. No ground-truth boxes/crops or class-specific attribute lookup in forward.',
        detector_prompts=DETECT_PROMPTS,source_md5=verified['official_md5'])
    manifest=dict(classes=[dict(label=mapping[c],source_id=c,name=classes[c]) for c in selected],attributes=chosen,image_root=str(BASE/'images'),rows=rows)
    fixed(config,protocol);fixed(target,manifest);write_json(REPORT/'annotation_support.json',dict(attributes=support,groups=chosen_groups,certainty_labels=(BASE/'attributes/certainties.txt').read_text(encoding='utf-8'),counts=dict(Counter(r['role'] for r in rows))))
    return protocol,manifest

def regions(stage):
    p,m=prepare();roles=['train_seen','dev_seen','dev_unseen'] if stage=='development' else ['eval_seen','eval_unseen']
    if stage=='evaluation' and not (OUT/'selection_locked.json').exists():raise ValueError('Lock selection before final image processing')
    path=ROOT/f'cache/{VERSION}/regions_{stage}.json';metadata=dict(protocol_sha256=digest(ROOT/f'configs/{VERSION}.json'),implementation_sha256=digest(__file__))
    saved=json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(metadata=metadata,images={})
    assert saved['metadata']==metadata
    pending=[r for r in m['rows'] if r['role'] in roles and r['path'] not in saved['images']]
    if pending:
        processor,model=detector_parts()
        for i,r in enumerate(pending):
            with Image.open(Path(m['image_root'])/r['path']) as im:record=locate(im.convert('RGB'),p['detector_prompts'],processor,model,p)
            saved['images'][r['path']]=record;write_json(path,saved)
            if i%40==0:print(f'CUB regions {stage} {i+1}/{len(pending)}',flush=True)
        del model,processor;torch.cuda.empty_cache()
    return saved

def text_bank():
    p,m=prepare();path=ROOT/f'cache/{VERSION}/text.pt';metadata=dict(manifest_sha256=digest(ROOT/f'data/{VERSION}/manifest.json'))
    if path.exists():
        result=torch.load(path,weights_only=True);assert result['metadata']==metadata;return result
    model,_,tokenize=load_clip();model=model.cuda().eval();prompts=[f'a photo of a {c["name"]}.' for c in m['classes']]+[a['prompt'] for a in m['attributes']]+[a['negative_prompt'] for a in m['attributes']]
    with torch.no_grad():text=torch.cat([model.encode_text(tokenize(prompts[i:i+32]).cuda(),normalize=True).float().cpu() for i in range(0,len(prompts),32)])
    a=len(m['attributes']);result=dict(metadata=metadata,class_text=text[:20],attribute_text=text[20:20+a],negative_text=text[20+a:]);torch.save(result,path);del model;torch.cuda.empty_cache();return result

def pixels(stage):
    p,m=prepare();r=regions(stage);bank=text_bank();target=ROOT/f'cache/{VERSION}/pixels_{stage}.pt'
    metadata=dict(manifest_sha256=digest(ROOT/f'data/{VERSION}/manifest.json'),regions_sha256=digest(ROOT/f'cache/{VERSION}/regions_{stage}.json'),implementation_sha256=digest(__file__))
    if target.exists():
        result=torch.load(target,weights_only=True);assert result['metadata']==metadata;return result
    roles=['train_seen','dev_seen','dev_unseen'] if stage=='development' else ['eval_seen','eval_unseen'];rows=[row for row in m['rows'] if row['role'] in roles]
    model,transform,_=load_clip();model=model.cuda().eval();xs=[];valid=[];targets=[];teachers=[];pseudo=[];features=[]
    for i,row in enumerate(rows):
        rec=r['images'][row['path']];x,_,_,ok=image_tensors(Path(m['image_root'])/row['path'],rec,transform);t=torch.tensor(row['attributes'],dtype=torch.float);regional=torch.full((2,len(t)),-1.)
        # Parts serve only to decide whether a target is visible in a predicted crop.
        # Account for the CLIP center crop within both original and detector ROI.
        boxes=[[0,0,*row['size']],*rec['boxes']];view_targets=[]
        for box in boxes:
            x1,y1,x2,y2=box;side=min(x2-x1,y2-y1);cx=(x1+x2)/2;cy=(y1+y2)/2
            v=t.clone()
            for a,points in enumerate(row['attribute_points']):
                if not any(cx-side/2<=px<=cx+side/2 and cy-side/2<=py<=cy+side/2 for px,py in points):v[a]=-1
            view_targets.append(v)
        while len(view_targets)<3:view_targets.append(torch.full_like(t,-1))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            f=F.normalize(model.encode_image(x.cuda()).float(),dim=-1).cpu()
        xs.append(x);valid.append(ok);targets.append(torch.stack(view_targets));teachers.append(f[0]);features.append(f)
        pseudo.append((10*f@(bank['attribute_text']-bank['negative_text']).T).sigmoid())
        if i%80==0:print(f'CUB pixels {stage} {i}/{len(rows)}',flush=True)
    result=dict(metadata=metadata,rows=rows,images=torch.stack(xs),valid=torch.stack(valid),targets=torch.stack(targets),teacher=torch.stack(teachers),native_views=torch.stack(features),pseudo=torch.stack(pseudo),labels=torch.tensor([row['label'] for row in rows]))
    torch.save(result,target);del model;torch.cuda.empty_cache();return result
