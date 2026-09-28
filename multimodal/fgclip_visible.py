"""FG-CLIP follow-up using official dense features and unchanged visible protocol."""
import os,json
import torch
from torch.nn import functional as F
from PIL import Image
from pathlib import Path
from data_tools import ROOT,digest
from .visible_data import prepare_manifest,letterbox,patch_box
os.environ.setdefault('HF_HOME',str(ROOT/'cache/huggingface'))
os.environ.setdefault('HF_MODULES_CACHE',str(ROOT/'cache/huggingface/modules'))

def model_parts():
    from transformers import AutoModelForCausalLM,AutoTokenizer,AutoImageProcessor
    path=ROOT/'cache/visible_v1/models/fgclip_b16'
    revision=json.loads((ROOT/'configs/fgclip_revision.json').read_text(encoding='utf-8'))
    for name,sha in revision['code_sha256'].items():
        if digest(path/name)!=sha:raise ValueError('FG-CLIP custom code differs from recorded revision')
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=True).eval()
    model.requires_grad_(False)
    return model,AutoTokenizer.from_pretrained(path,local_files_only=True),AutoImageProcessor.from_pretrained(path,local_files_only=True,use_fast=False)

@torch.no_grad()
def encode_pixels(model,pixels):
    # One vision pass; exactly the official get_image_dense_features formula.
    vision=model.vision_model(pixel_values=pixels,output_hidden_states=True,return_dict=True)
    whole=model.visual_projection(vision.pooler_output)
    patch=model.forward_without_attn(vision.hidden_states[-2])[:,1:]
    patch=model.visual_projection(model.vision_model.post_layernorm(patch))
    return F.normalize(whole.float(),dim=-1),F.normalize(patch.float(),dim=-1)

def prepare_features():
    manifest=prepare_manifest();target=ROOT/'cache/visible_v1/fgclip_b16_features.pt'
    revision=json.loads((ROOT/'configs/fgclip_revision.json').read_text(encoding='utf-8'))
    metadata=dict(manifest_sha256=digest(ROOT/'data/visible_v1/manifest.json'),revision=revision,
        implementation_sha256=digest(__file__),shared_data_sha256=digest(ROOT/'multimodal/visible_data.py'),
        preprocessing='224 RGB letterbox, native mean/std; official FG-CLIP dense features; short text branch, no attention mask as official example')
    if target.exists():
        cache=torch.load(target,weights_only=True)
        if cache['metadata']!=metadata:raise ValueError('FG-CLIP cache changed')
        return cache
    model,tokenizer,processor=model_parts();model=model.cuda();whole=[];patches=[];valid=[];regions=[]
    prompts=[f'a photo of {c["name"]}, {c.get("common_name") or c["name"]}.' for c in manifest['classes']]
    prompts += [f'a photo showing {a}.' for a in manifest['attributes']]
    with torch.no_grad():
        tokens=tokenizer(prompts,padding='max_length',max_length=77,truncation=True,return_tensors='pt')['input_ids'].cuda()
        text=F.normalize(model.get_text_features(tokens,walk_short_pos=True).float(),dim=-1).cpu()
        for start in range(0,len(manifest['rows']),8):
            images=[]
            for row in manifest['rows'][start:start+8]:
                assert row['split'] in ('train','val')
                with Image.open(Path(manifest['image_root'])/row['path']) as im:im=im.convert('RGB')
                padded,geometry=letterbox(im);images.append(padded);valid.append(patch_box([0,0,1,1],geometry))
                region=torch.zeros(16,196,dtype=torch.bool)
                for key,box in row['evidence_boxes'].items():region[int(key)]=patch_box(box,geometry)
                regions.append(region)
            pixels=processor(images=images,return_tensors='pt')['pixel_values'].cuda();g,p=encode_pixels(model,pixels)
            if start==0:
                expected=F.normalize(model.get_image_dense_features(pixels).float(),dim=-1)
                torch.testing.assert_close(p,expected,rtol=1e-5,atol=1e-6)
                torch.testing.assert_close(g,F.normalize(model.get_image_features(pixels).float(),dim=-1),rtol=1e-5,atol=1e-6)
            whole.append(g.cpu());patches.append(p.cpu())
            if start%80==0:print(f'FG-CLIP features {start}/400',flush=True)
    cache=dict(metadata=metadata,rows=manifest['rows'],classes=manifest['classes'],attributes=manifest['attributes'],
        global_features=torch.cat(whole),patch_features=torch.cat(patches),valid_patches=torch.stack(valid),
        attribute_targets=torch.tensor([r['attributes'] for r in manifest['rows']],dtype=torch.float32),
        evidence_masks=torch.stack(regions),labels=torch.tensor([r['label'] for r in manifest['rows']]),class_text=text[:20],attribute_text=text[20:])
    torch.save(cache,target);del model;torch.cuda.empty_cache();return cache
