"""Deploy selected CUB model using only a photo and fixed text/model assets."""
import argparse,gc,json,time
from pathlib import Path
import torch
from PIL import Image
from data_tools import ROOT,write_json
from multimodal.cub_data import prepare,OUT
from multimodal.cub_experiment import setup,restore
from multimodal.grounded_data import detector_parts,locate,image_tensors

@torch.no_grad()
def predict(checkpoint,path,candidates=None):
    setup();p,m=prepare();saved=torch.load(checkpoint,weights_only=True);variant=saved['config']['variant'];start=time.time()
    if variant=='finetune':record=dict(boxes=[],attribute_ids=[],scores=[])
    else:
        processor,detector=detector_parts()
        with Image.open(path) as im:record=locate(im.convert('RGB'),p['detector_prompts'],processor,detector,p)
        del processor,detector;gc.collect();torch.cuda.empty_cache()
    model,transform,saved=restore(checkpoint);model.eval();x,_,_,valid=image_tensors(path,record,transform);n=p['micro_batch']
    with torch.autocast('cuda',dtype=torch.bfloat16):o=model(x[None].expand(n,-1,-1,-1,-1).contiguous().cuda(),valid[None].expand(n,-1).contiguous().cuda())
    logits=(model.logit_scale.exp().clamp(max=100)*o['embedding']@model.class_text.T)[0].cpu();attrs=o['view_logits'][0,0].sigmoid().cpu();candidates=list(range(20)) if candidates is None else candidates;scores=logits[candidates].softmax(-1)
    result=dict(variant=variant,checkpoint_step=saved['step'],candidates=candidates,predictions=[dict(label=candidates[int(i)],name=m['classes'][candidates[int(i)]]['name'],score=float(scores[i])) for i in scores.argsort(descending=True)[:5]],attributes=[dict(name=m['attributes'][int(i)]['name'],score=float(attrs[i])) for i in attrs.argsort(descending=True)[:8]],regions=record['boxes'],seconds=time.time()-start,note='Image-only inference. No image annotations, true parts, class labels or oracle crops are looked up. Scores are not calibrated probabilities.')
    del model;gc.collect();torch.cuda.empty_cache();return result,logits,attrs

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('image',type=Path);parser.add_argument('--checkpoint',type=Path);parser.add_argument('--variant',default='gold',choices=['finetune','region_only','automatic','gold','shuffled','gold_region']);parser.add_argument('--selection',choices=['classification','attribute'],default='classification');parser.add_argument('--output',type=Path);a=parser.parse_args()
    if a.checkpoint is None:
        lock=json.loads((OUT/'selection_locked.json').read_text(encoding='utf-8'));key='selected' if a.selection=='classification' else 'attribute_selected';a.checkpoint=ROOT/next(r['checkpoint'] for r in lock[key] if r['variant']==a.variant)
    result,_,_=predict(a.checkpoint,a.image)
    if a.output:write_json(a.output,result)
    print(json.dumps(result,indent=2))
