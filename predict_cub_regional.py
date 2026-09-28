"""Photo-only deployment of frozen-global/trainable-regional models."""
import argparse,gc,json
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from data_tools import ROOT,write_json
from cub_regional_experiment import setup,prepare,restore,OUT,text_bank
from multimodal.models import load_clip
from multimodal.grounded_data import detector_parts,locate,image_tensors


@torch.no_grad()
def predict(checkpoint,image_path,candidates=None):
    c=setup();p,m=prepare();processor,detector=detector_parts()
    with Image.open(image_path) as im:record=locate(im.convert('RGB'),p['detector_prompts'],processor,detector,p)
    del processor,detector;gc.collect();torch.cuda.empty_cache()
    clip,transform,_=load_clip();clip=clip.cuda().eval();images,_,_,valid=image_tensors(image_path,record,transform)
    with torch.autocast('cuda',dtype=torch.bfloat16):native=F.normalize(clip.encode_image(images.cuda()).float(),dim=-1)[0]
    del clip;gc.collect();torch.cuda.empty_cache()
    model,_,saved=restore(checkpoint);n=c['micro_batch'];regions=model.encode_regions(images[None].expand(n,-1,-1,-1,-1).contiguous().cuda())
    o=model.from_regions(regions,native[None].expand(n,-1),valid[None].expand(n,-1).cuda())
    logits=o['logits'][0].cpu();prob=o['probability'][0].cpu();candidates=list(range(len(m['classes']))) if candidates is None else candidates;scores=logits[candidates].softmax(-1)
    result=dict(variant=saved['config']['variant'],seed=saved['config']['seed'],step=saved['step'],predictions=[dict(label=candidates[int(i)],name=m['classes'][candidates[int(i)]]['name'],score=float(scores[i])) for i in scores.argsort(descending=True)[:5]],regions=record['boxes'],note='Photo only; global features from frozen original CLIP, regional crops through trained visual tail. No true attributes/parts/class labels read. Uncalibrated scores.')
    return result,logits,prob


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('image',type=Path);parser.add_argument('--variant',default='tokens');parser.add_argument('--seed',type=int,default=42);parser.add_argument('--output',type=Path);a=parser.parse_args()
    lock=json.loads((OUT/'selection_locked.json').read_text());chosen=next(r for r in lock['selected'] if r['variant']==a.variant and r['seed']==a.seed)
    result,_,_=predict(ROOT/chosen['checkpoint'],a.image)
    if a.output:write_json(a.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
