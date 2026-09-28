import json
import gc
from pathlib import Path
import torch
from torch.nn import functional as F
from torchvision.ops import nms
from PIL import Image
from data_tools import digest, write_json
from run import load_manifest
from .models import ROOT, load_clip, load_backbone


def load_attributes(path):
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    prompts=obj['prompts']
    if not prompts or len(prompts)!=len(set(prompts)):
        raise ValueError('Attribute prompts must be a nonempty unique list.')
    return obj


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def text_bank(manifest_path, attributes_path, output, device):
    manifest=load_manifest(manifest_path)
    attributes=load_attributes(attributes_path)
    if attributes.get('species_descriptions_reviewed'):
        covered={s['category_id'] for s in attributes.get('species',[])}
        required={c['id'] for c in manifest['classes']}
        if not required.issubset(covered):
            raise ValueError('Source-checked attributes do not cover this manifest. Prepare descriptions for all selected species.')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    clip,_,tokenizer=load_clip()
    clip=clip.to(device).eval()
    names=[c.get('common_name') or c['name'] for c in manifest['classes']]
    class_prompts=[f'a photo of {c["name"]}, {n}.' for c,n in zip(manifest['classes'],names)]
    def encode(prompts):
        encoded=[]
        with torch.inference_mode():
            for start in range(0,len(prompts),32):
                encoded.append(clip.encode_text(tokenizer(prompts[start:start+32]).to(device), normalize=True).float().cpu())
        return torch.cat(encoded)
    bank={'class_text':encode(class_prompts),'attribute_text':encode(attributes['prompts']),
          'class_prompts':class_prompts,'attribute_prompts':attributes['prompts'],
          'manifest_sha256':digest(manifest_path),'attributes_sha256':digest(attributes_path),
          'encoder':'OpenAI CLIP ViT-B/32 via OpenCLIP 3.2.0'}
    torch.save(bank,output/'text_bank.pt')
    del clip
    release()
    return bank


def all_rows(manifest):
    return [dict(r,split=s) for s in ('train','val','test') for r in manifest['splits'][s]]


@torch.inference_mode()
def locate_image(image,prompts,processor,detector,device,topk=3,threshold=.05):
    """Label-independent grounding shared by preparation and single-image use."""
    inputs=processor(text=[prompts],images=image,return_tensors='pt').to(device)
    prediction=detector(**inputs)
    found=processor.post_process_object_detection(prediction,threshold=threshold,
                 target_sizes=torch.tensor([image.size[::-1]],device=device))[0]
    boxes=found['boxes'].float()
    boxes[:,[0,2]]=boxes[:,[0,2]].clamp(0,image.width)
    boxes[:,[1,3]]=boxes[:,[1,3]].clamp(0,image.height)
    valid=(boxes[:,2]-boxes[:,0]>2)&(boxes[:,3]-boxes[:,1]>2)
    boxes,scores,labels=boxes[valid],found['scores'][valid],found['labels'][valid]
    keep=nms(boxes,scores,.5)[:topk]
    return {'boxes':boxes[keep].cpu().tolist(),'scores':scores[keep].cpu().tolist(),
            'attribute_ids':labels[keep].cpu().tolist()}


def ground_regions(manifest_path,attributes_path,output,device,topk=3,threshold=.05):
    from transformers import OwlViTProcessor,OwlViTForObjectDetection
    manifest=load_manifest(manifest_path)
    attrs=load_attributes(attributes_path)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    path=output/'regions.json'
    context={'manifest_sha256':digest(manifest_path),'attributes_sha256':digest(attributes_path),
             'detector':'google/owlvit-base-patch32','topk':topk,'threshold':threshold}
    result=dict(context,images={})
    if path.exists():
        result=json.loads(path.read_text(encoding='utf-8'))
        if any(result.get(k)!=v for k,v in context.items()):
            raise ValueError('Existing region cache has a different protocol. Use a new output directory.')
    rows=all_rows(manifest)
    pending=[r['path'] for r in rows if r['path'] not in result['images']]
    if not pending:
        print('Existing region cache verified.',flush=True)
        return result
    processor=OwlViTProcessor.from_pretrained(context['detector'])
    detector=OwlViTForObjectDetection.from_pretrained(context['detector']).to(device).eval()
    # Use the identical union vocabulary for EVERY image: no true label is read.
    with torch.inference_mode():
        for index,relative in enumerate(pending):
            with Image.open(Path(manifest['image_root'])/relative) as im:
                image=im.convert('RGB')
            result['images'][relative]=locate_image(image,attrs['prompts'],processor,detector,device,topk,threshold)
            if (index+1)%10==0:
                write_json(path,result)
                print(f'OWL-ViT regions: {index+1}/{len(pending)}',flush=True)
    write_json(path,result)
    del detector,processor
    release()
    return result


def extract_features(backbone,manifest_path,attributes_path,output,device,batch_size=16):
    output=Path(output)
    context={'manifest_sha256':digest(manifest_path),'attributes_sha256':digest(attributes_path),
             'backbone':backbone,'pretrained':True,'frozen':True,'augmentation':'none; weight-specific eval transforms'}
    target=output/f'{backbone}_features.pt'
    if target.exists():
        saved=torch.load(target,weights_only=True)
        if saved['metadata']!=context:
            raise ValueError('Feature cache protocol mismatch.')
        print('Verified feature cache:',backbone,flush=True)
        return saved
    manifest=load_manifest(manifest_path)
    regions=json.loads((output/'regions.json').read_text(encoding='utf-8'))
    if regions['manifest_sha256']!=context['manifest_sha256'] or regions['attributes_sha256']!=context['attributes_sha256']:
        raise ValueError('Region cache protocol mismatch.')
    model,transform,dim=load_backbone(backbone)
    model=model.to(device).eval()
    rows=all_rows(manifest)
    global_features=[]
    region_features=[]
    attr_ids=[]
    masks=[]
    topk=regions['topk']
    with torch.inference_mode():
        for start in range(0,len(rows),batch_size):
            batch=rows[start:start+batch_size]
            images,crops=[],[]
            for r in batch:
                with Image.open(Path(manifest['image_root'])/r['path']) as im:
                    im=im.convert('RGB')
                    images.append(transform(im))
                    found=regions['images'][r['path']]
                    count=len(found['boxes'])
                    attr_ids.append(found['attribute_ids']+[0]*(topk-count))
                    masks.append([True]*count+[False]*(topk-count))
                    crops.extend(transform(im.crop(tuple(box))) for box in found['boxes'])
                    crops.extend(transform(im) for _ in range(topk-count))
            def encode(tensors):
                parts=[]
                for b in range(0,len(tensors),batch_size):
                    parts.append(F.normalize(model(torch.stack(tensors[b:b+batch_size]).to(device)).float(),dim=-1).cpu())
                return torch.cat(parts)
            global_features.append(encode(images))
            region_features.append(encode(crops).view(len(batch),topk,dim))
            if start%160==0:
                print(f'{backbone} features: {min(start+batch_size,len(rows))}/{len(rows)}',flush=True)
    saved={'metadata':context,'rows':rows,'classes':manifest['classes'],
           'global':torch.cat(global_features),'regions':torch.cat(region_features),
           'attribute_ids':torch.tensor(attr_ids),'region_mask':torch.tensor(masks),
           'labels':torch.tensor([r['label'] for r in rows]),'pilot':manifest.get('pilot',False)}
    torch.save(saved,target)
    del model
    release()
    return saved
