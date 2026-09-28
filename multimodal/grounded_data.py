"""Versioned raw-image protocol and label-independent OWL-ViT region detection."""
import json,math,time,textwrap
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from torchvision.ops import nms
from data_tools import ROOT,digest,write_json

VERSION='grounded_v1'
DETECTOR_REVISION='cbc355fb364588351c5d51c7f74465e8e7ec6f72'

def prepare():
    source=json.loads((ROOT/'data/expanded20/manifest.json').read_text(encoding='utf-8'))
    old=json.loads((ROOT/'configs/zsl_v1.json').read_text(encoding='utf-8'));seen=old['seen_classes'];dev=old['dev_unseen_classes'];unseen=old['eval_unseen_classes']
    rows=[]
    for split,items in source['splits'].items():
        for row in items:
            c=row['label'];role=None
            if c in seen:role='train_seen' if split=='train' else 'dev_seen' if split=='val' else 'eval_seen'
            elif split in ('val','test'):role='dev_unseen' if c in dev else 'eval_unseen'
            if role:rows.append(dict(row,role=role))
    assert len(rows)==700 and sum(r['role']=='train_seen' for r in rows)==300
    assert len({r['sha256'] for r in rows})==700
    protocol=dict(version=VERSION,seen_classes=seen,dev_unseen_classes=dev,eval_unseen_classes=unseen,
        backbone='OpenAI CLIP ViT-B/32',resolution=224,max_regions=2,detector_repo='google/owlvit-base-patch32',detector_revision=DETECTOR_REVISION,
        detector_threshold=.05,detector_nms_iou=.5,min_region_area=.005,max_region_area=.95,
        variants=['finetune','region_only','ag','shuffled'],seeds=[42],learning_rates=[1e-5,3e-6],fusion_lr_multiplier=10,
        trainable_visual_blocks=2,micro_batch=2,effective_batch=16,updates=120,validation_steps=[0,40,80,120],
        preserve_weight=.5,region_alignment_weight=.1,contrastive_weight=.5,class_ce_weight=.5,
        selection='Maximum development GZSL harmonic mean H on 10 seen + 4 development-unseen classes; tie by mean U/S then lower mixed CE. Select LR separately per variant. Evaluate 6 final-unseen + 10 final-seen classes after locking all selections. Step zero is eligible and explicitly reported.',
        architecture='Two independent CLIP visual towers initialized from original weights, last two transformer blocks plus final norm/projection trainable; frozen text tower. Attribute/region tokens and global token fused using CAF self-attention with a small nonzero residual gate. Region-only has identical modules with zero text. Shuffled uses identical boxes but wrongly paired text and auxiliary targets.',
        loss='Seen-only CE plus multi-positive symmetric image/text contrastive loss over an effective batch of 16; duplicate class names are positives. Preserve original whole-image embedding. AG adds confidence-weighted region/text cosine loss, no ground-truth class attribute selection at inference.',
        precision='CUDA bfloat16 autocast, float32 optimizer states; activation checkpointing; exact two-pass gradient caching across microbatches, dropout disabled and identical input tensors reused.',
        scope='Exploratory reused iNaturalist data, 20 species. Training grows 200 to 300 images; development seen images now separated from final seen images. Not directly comparable to previous protocols. Unseen relative to adaptation only; no pretraining-overlap guarantee. Structural AG reconstruction, not exact CoCa paper reproduction.',
        manifest_sha256=digest(ROOT/'data/expanded20/manifest.json'),attributes_sha256=digest(ROOT/'configs/expanded_attributes.json'))
    manifest=dict(classes=source['classes'],image_root=source['image_root'],rows=rows)
    for path,obj in [(ROOT/f'configs/{VERSION}.json',protocol),(ROOT/f'data/{VERSION}/manifest.json',manifest)]:
        if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=obj:raise ValueError('Grounded definition changed: create a new version')
        write_json(path,obj)
    return protocol,manifest

def detector_parts():
    from transformers import OwlViTProcessor,OwlViTForObjectDetection
    snapshot=ROOT/f'cache/huggingface/hub/models--google--owlvit-base-patch32/snapshots/{DETECTOR_REVISION}'
    processor=OwlViTProcessor.from_pretrained(snapshot,local_files_only=True)
    model=OwlViTForObjectDetection.from_pretrained(snapshot,local_files_only=True).eval().cuda()
    model.requires_grad_(False)
    return processor,model

@torch.no_grad()
def locate(image,prompts,processor,model,protocol):
    inputs=processor(text=[prompts],images=image,return_tensors='pt').to('cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(**inputs)
    result=processor.post_process_object_detection(prediction,threshold=protocol['detector_threshold'],target_sizes=torch.tensor([image.size[::-1]],device='cuda'))[0]
    boxes=result['boxes'].float();scores=result['scores'].float();labels=result['labels']
    boxes[:,[0,2]]=boxes[:,[0,2]].clamp(0,image.width);boxes[:,[1,3]]=boxes[:,[1,3]].clamp(0,image.height)
    area=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])/(image.width*image.height)
    ok=torch.isfinite(boxes).all(-1)&torch.isfinite(scores)&(boxes[:,2]>boxes[:,0]+1)&(boxes[:,3]>boxes[:,1]+1)&(area>=protocol['min_region_area'])&(area<=protocol['max_region_area'])
    boxes,scores,labels=boxes[ok],scores[ok],labels[ok];keep=nms(boxes,scores,protocol['detector_nms_iou'])[:protocol['max_regions']]
    return dict(boxes=boxes[keep].cpu().tolist(),scores=scores[keep].cpu().tolist(),attribute_ids=labels[keep].cpu().tolist(),size=list(image.size))

def ground(stage='development',limit=None):
    p,m=prepare();attrs=json.loads((ROOT/'configs/expanded_attributes.json').read_text(encoding='utf-8'));prompts=attrs['prompts']
    roles=['train_seen','dev_seen','dev_unseen'] if stage=='development' else ['eval_seen','eval_unseen']
    if stage=='evaluation' and not (ROOT/f'runs/{VERSION}/selection_locked.json').exists():raise ValueError('Lock model selection before final-image grounding')
    path=ROOT/f'cache/{VERSION}/regions_{stage}.json'
    metadata=dict(protocol_sha256=digest(ROOT/f'configs/{VERSION}.json'),implementation_sha256=digest(__file__),stage=stage)
    saved=json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(metadata=metadata,images={})
    if saved['metadata']!=metadata:raise ValueError('Grounding cache definition changed')
    rows=[r for r in m['rows'] if r['role'] in roles];pending=[r for r in rows if r['path'] not in saved['images']]
    if limit is not None:pending=pending[:limit]
    if not pending:return saved
    processor,model=detector_parts();start=time.time();torch.cuda.reset_peak_memory_stats()
    for i,row in enumerate(pending):
        with Image.open(Path(m['image_root'])/row['path']) as image:image=image.convert('RGB')
        saved['images'][row['path']]=locate(image,prompts,processor,model,p)
        write_json(path,saved)
        if i%10==0:print(f'OWL {stage}: {len(saved["images"])}/{len(rows)}, elapsed {time.time()-start:.1f}s',flush=True)
    write_json(ROOT/f'reports/{VERSION}/grounding_{stage}.json',dict(images=len(saved['images']),zero_regions=sum(not r['boxes'] for r in saved['images'].values()),mean_regions=sum(len(r['boxes']) for r in saved['images'].values())/len(saved['images']),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,elapsed_this_call_seconds=time.time()-start,scope='Detector proposals are weak supervision, not expert-confirmed attribute regions.'))
    del model,processor;torch.cuda.empty_cache();return saved

def image_tensors(path,record,transform,max_regions=2):
    with Image.open(path) as image:image=image.convert('RGB')
    views=[transform(image)];ids=[];scores=[];valid=[]
    for box,a,score in zip(record['boxes'],record['attribute_ids'],record['scores']):
        views.append(transform(image.crop(tuple(box))));ids.append(a);scores.append(score);valid.append(True)
    while len(ids)<max_regions:
        views.append(torch.zeros_like(views[0]));ids.append(0);scores.append(0.);valid.append(False)
    return torch.stack(views),torch.tensor(ids),torch.tensor(scores),torch.tensor(valid)

def render_audit():
    p,m=prepare();saved=json.loads((ROOT/f'cache/{VERSION}/regions_development.json').read_text(encoding='utf-8'))
    prompts=json.loads((ROOT/'configs/expanded_attributes.json').read_text(encoding='utf-8'))['prompts']
    rows=[]
    for c in p['seen_classes']:rows.extend([r for r in m['rows'] if r['role']=='train_seen' and r['label']==c][:2])
    folder=ROOT/f'work/{VERSION}';folder.mkdir(parents=True,exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    for page in range(math.ceil(len(rows)/4)):
        canvas=Image.new('RGB',(1100,1200),'white');draw=ImageDraw.Draw(canvas)
        for j,row in enumerate(rows[page*4:page*4+4]):
            rec=saved['images'][row['path']]
            with Image.open(Path(m['image_root'])/row['path']) as im:im=im.convert('RGB')
            marked=im.copy();d=ImageDraw.Draw(marked)
            for b,col in zip(rec['boxes'],['red','blue']):d.rectangle(b,outline=col,width=max(2,im.width//150))
            marked.thumbnail((350,240));canvas.paste(marked,(0,j*300+35));draw.text((5,j*300+5),f'Fixed training sample {page*4+j}',font=font,fill='black')
            for k,(b,a,score) in enumerate(zip(rec['boxes'],rec['attribute_ids'],rec['scores'])):
                crop=im.crop(b);crop.thumbnail((350,210));x=370+k*365;canvas.paste(crop,(x,j*300+75))
                draw.multiline_text((x,j*300+3),textwrap.fill(f'{score:.3f}: {prompts[a]}',40),font=font,fill='black')
        canvas.save(folder/f'grounding_audit_{page+1}.jpg',quality=92)
