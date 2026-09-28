"""Pinned VLM features and visible-attribute supervision, with test split excluded."""
import json,random
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from data_tools import ROOT,digest,write_json


def letterbox(image,size=224):
    w,h=image.size;scale=min(size/w,size/h);nw,nh=round(w*scale),round(h*scale)
    left=(size-nw)//2;top=(size-nh)//2
    out=Image.new('RGB',(size,size),(128,128,128));out.paste(image.resize((nw,nh),Image.Resampling.BICUBIC),(left,top))
    return out,(nw/size,nh/size,left/size,top/size)


def patch_box(box,geometry,grid=14):
    sx,sy,dx,dy=geometry;x1,y1,x2,y2=box
    x1,x2=x1*sx+dx,x2*sx+dx;y1,y2=y1*sy+dy,y2*sy+dy
    y,x=torch.meshgrid((torch.arange(grid)+.5)/grid,(torch.arange(grid)+.5)/grid,indexing='ij')
    mask=(x>=x1)&(x<=x2)&(y>=y1)&(y<=y2)
    if not mask.any():mask[torch.argmin((y[:,0]-(y1+y2)/2).abs()),torch.argmin((x[0]-(x1+x2)/2).abs())]=True
    return mask.flatten()


def prepare_manifest():
    source=json.loads((ROOT/'data/expanded20/manifest.json').read_text(encoding='utf-8'));selection=json.loads((ROOT/'configs/visible_review_selection.json').read_text(encoding='utf-8'))
    raw=json.loads((ROOT/'configs/visible_annotations_raw.json').read_text(encoding='utf-8'));review={r['id']:r for r in raw['items']}
    assert len(review)==len(selection)==60
    bypath={r['path']:review[r['id']] for r in selection};rows=[]
    for label in range(20):
        candidates=sorted([r for r in source['splits']['train'] if r['label']==label],key=lambda r:r['path'])
        chosen=[r for r in candidates if r['path'] in bypath]
        assert len(chosen)==2
        remaining=[r for r in candidates if r['path'] not in bypath];random.Random(20260929+label).shuffle(remaining)
        rows.extend(dict(r,split='train') for r in chosen+remaining[:8])
    rows.extend(dict(r,split='val') for r in source['splits']['val'])
    for row in rows:
        record=bypath.get(row['path']);row['attributes']=[-1]*16;row['evidence_boxes']={}
        if record:
            domain=range(7) if record['domain']=='wing' else range(7,16)
            assert set(record['positive']).issubset(domain) and not set(record['positive'])&set(record['unknown'])
            for a in domain:
                if a not in record['unknown']:row['attributes'][a]=int(a in record['positive'])
            row['review_id']=record['id'];row['evidence_box']=record['box']
            for a in record['positive']:
                box=record.get('boxes',{}).get(str(a),record['box']);x1,y1,x2,y2=box
                assert 0<=x1<x2<=1 and 0<=y1<y2<=1
                row['evidence_boxes'][str(a)]=box
    manifest=dict(classes=source['classes'],image_root=source['image_root'],rows=rows,attributes=raw['attributes'],
        review_scope=raw['scope'],annotation_sha256=digest(ROOT/'configs/visible_annotations_raw.json'),
        source_manifest_sha256=digest(ROOT/'data/expanded20/manifest.json'),selection_sha256=digest(ROOT/'configs/visible_review_selection.json'))
    target=ROOT/'data/visible_v1/manifest.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8'))!=manifest:raise ValueError('Visible manifest changed; version the experiment')
    write_json(target,manifest)
    # Training support only; validation support never decides which labels are trained.
    t=torch.tensor([r['attributes'] for r in rows if r['split']=='train'])
    support=[dict(id=a,name=raw['attributes'][a],train_positive=int((t[:,a]==1).sum()),train_negative=int((t[:,a]==0).sum())) for a in range(16)]
    assert all(x['train_positive']>0 and x['train_negative']>0 for x in support)
    write_json(ROOT/'reports/visible_v1/annotation_support.json',support)
    return manifest


def model_parts(name):
    from transformers import AutoModel,AutoTokenizer,AutoImageProcessor
    path=ROOT/f'cache/visible_v1/models/{name}'
    revisions=json.loads((ROOT/'configs/visible_model_revisions.json').read_text(encoding='utf-8'))
    assert (path/revisions[name]['weights']).is_file(),f'{name} weights not downloaded'
    model=AutoModel.from_pretrained(path,local_files_only=True,trust_remote_code=False).eval()
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    processor=AutoImageProcessor.from_pretrained(path,local_files_only=True,use_fast=False)
    model.requires_grad_(False)
    return model,tokenizer,processor


@torch.no_grad()
def encode_pixels(model,pixels,name):
    vision=model.vision_model(pixel_values=pixels)
    if name=='clip_b16':
        whole=model.visual_projection(vision.pooler_output)
        patches=model.visual_projection(model.vision_model.post_layernorm(vision.last_hidden_state[:,1:]))
    else:whole=vision.pooler_output;patches=vision.last_hidden_state
    assert patches.shape[1]==196
    return F.normalize(whole.float(),dim=-1),F.normalize(patches.float(),dim=-1)


def prepare_features(name):
    manifest=prepare_manifest();target=ROOT/f'cache/visible_v1/{name}_features.pt'
    protocol=dict(manifest_sha256=digest(ROOT/'data/visible_v1/manifest.json'),
        revision=json.loads((ROOT/'configs/visible_model_revisions.json').read_text(encoding='utf-8'))[name],
        implementation_sha256=digest(__file__),preprocessing='224 RGB letterbox with gray padding, native mean/std; valid-token masking')
    if target.exists():
        saved=torch.load(target,weights_only=True)
        if saved['metadata']!=protocol:raise ValueError('Visible feature cache mismatch')
        return saved
    model,tokenizer,processor=model_parts(name);model=model.cuda().eval();whole=[];patches=[];valid=[];regions=[]
    prompts=[f'a photo showing {a}.' for a in manifest['attributes']]
    class_prompts=[f'a photo of {c["name"]}, {c.get("common_name") or c["name"]}.' for c in manifest['classes']]
    with torch.no_grad():
        texts=tokenizer(class_prompts+prompts,padding='max_length',max_length=77 if name=='clip_b16' else 64,truncation=True,return_tensors='pt').to('cuda')
        text=F.normalize(model.get_text_features(**texts).float(),dim=-1).cpu()
        for start in range(0,len(manifest['rows']),8):
            images=[]
            for row in manifest['rows'][start:start+8]:
                assert row['split'] in ('train','val')
                with Image.open(Path(manifest['image_root'])/row['path']) as image:image=image.convert('RGB')
                padded,geometry=letterbox(image);images.append(padded);valid.append(patch_box([0,0,1,1],geometry))
                region=torch.zeros(16,196,dtype=torch.bool)
                for key,box in row['evidence_boxes'].items():region[int(key)]=patch_box(box,geometry)
                regions.append(region)
            values=processor(images=images,return_tensors='pt')['pixel_values'].cuda()
            g,p=encode_pixels(model,values,name);whole.append(g.cpu());patches.append(p.cpu())
            if start%80==0:print(f'{name} feature extraction {start}/{len(manifest["rows"])}',flush=True)
    result=dict(metadata=protocol,rows=manifest['rows'],classes=manifest['classes'],attributes=manifest['attributes'],
        global_features=torch.cat(whole),patch_features=torch.cat(patches),valid_patches=torch.stack(valid),
        attribute_targets=torch.tensor([r['attributes'] for r in manifest['rows']],dtype=torch.float32),
        evidence_masks=torch.stack(regions),labels=torch.tensor([r['label'] for r in manifest['rows']]),
        class_text=text[:20],attribute_text=text[20:])
    torch.save(result,target);del model;torch.cuda.empty_cache();return result


def render_boxes():
    manifest=prepare_manifest();root=ROOT/'work/visible_v1';font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    review=sorted([r for r in manifest['rows'] if 'review_id' in r],key=lambda r:r['review_id'])
    for page in range(10):
        canvas=Image.new('RGB',(1200,800),'white');draw=ImageDraw.Draw(canvas)
        for k,row in enumerate(review[page*6:page*6+6]):
            with Image.open(Path(manifest['image_root'])/row['path']) as im:im=im.convert('RGB')
            im.thumbnail((390,325));d=ImageDraw.Draw(im);w,h=im.size
            for box in row['evidence_boxes'].values():d.rectangle([box[0]*w,box[1]*h,box[2]*w,box[3]*h],outline='red',width=2)
            x=(k%3)*400;y=(k//3)*400;canvas.paste(im,(x,y+40));draw.text((x+4,y+3),f'ID {row["review_id"]} pos: '+str([i for i,a in enumerate(row['attributes']) if a==1]),font=font,fill='black')
        canvas.save(root/f'boxes_{page+1:02d}.jpg',quality=95)


if __name__=='__main__':render_boxes()
