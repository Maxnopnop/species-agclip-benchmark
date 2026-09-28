"""Image-only deployment: detect attribute regions, unload detector, classify."""
import argparse,gc,json,time
from pathlib import Path
import torch
from PIL import Image
from data_tools import ROOT,write_json
from multimodal.grounded_data import prepare,detector_parts,locate,image_tensors
from multimodal.grounded_experiment import setup,restore,OUT

@torch.no_grad()
def predict(checkpoint,image_path,candidates=None):
    setup();p,m=prepare();started=time.time()
    prompts=json.loads((ROOT/'configs/expanded_attributes.json').read_text(encoding='utf-8'))['prompts']
    # Inputs are a photo plus fixed, shared vocabulary; no image label is accepted.
    saved=torch.load(checkpoint,weights_only=True);variant=saved['config']['variant']
    if variant=='finetune':record=dict(boxes=[],attribute_ids=[],scores=[])
    else:
        processor,detector=detector_parts()
        with Image.open(image_path) as im:record=locate(im.convert('RGB'),prompts,processor,detector,p)
        del processor,detector;gc.collect();torch.cuda.empty_cache()
    model,transform,saved=restore(checkpoint);model.eval()
    views,ids,confidence,valid=image_tensors(image_path,record,transform)
    # Match the validated BF16 microbatch shape; GPU kernels for B=1 and B=2
    # can otherwise differ numerically. The duplicate is discarded, not voted.
    n=p['micro_batch']
    with torch.autocast('cuda',dtype=torch.bfloat16):
        e=model(views[None].expand(n,-1,-1,-1,-1).contiguous().cuda(),ids[None].expand(n,-1).contiguous().cuda(),valid[None].expand(n,-1).contiguous().cuda())['embedding'][:1]
    logits=(model.logit_scale.exp().clamp(max=100)*e@model.class_text.T)[0].cpu()
    candidates=list(range(len(m['classes']))) if candidates is None else candidates
    probabilities=logits[candidates].softmax(-1);rank=probabilities.argsort(descending=True)
    results=[dict(label=candidates[int(i)],name=m['classes'][candidates[int(i)]]['name'],score=float(probabilities[i])) for i in rank[:5]]
    result=dict(image=str(image_path),variant=variant,checkpoint_step=saved['step'],candidate_labels=candidates,predictions=results,regions=[dict(box=box,attribute=prompts[a],detector_score=s) for box,a,s in zip(record['boxes'],record['attribute_ids'],record['scores'])],seconds=time.time()-started,note='Scores are softmax similarities, not calibrated probabilities. Fixed shared text vocabulary is part of the deployed system; users supply only an image. Detection proposals may be incorrect. Candidate species must match the intended evaluation/task.')
    del model;gc.collect();torch.cuda.empty_cache();return result,logits

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('image',type=Path);parser.add_argument('--checkpoint',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--candidates',choices=['all','gzsl','unseen'],default='all');args=parser.parse_args()
    if args.checkpoint is None:
        lock=json.loads((OUT/'selection_locked.json').read_text(encoding='utf-8'));chosen=next(r for r in lock['selected'] if r['variant']=='ag');args.checkpoint=ROOT/chosen['checkpoint']
    p,_=prepare();candidates=None if args.candidates=='all' else p['eval_unseen_classes'] if args.candidates=='unseen' else p['seen_classes']+p['eval_unseen_classes']
    result,_=predict(args.checkpoint,args.image,candidates)
    if args.output:write_json(args.output,result)
    print(json.dumps(result,ensure_ascii=True,indent=2))
