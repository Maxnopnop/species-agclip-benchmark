"""Photo-only inference for the SigLIP 2 follow-up."""
import argparse,gc,json
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from data_tools import ROOT,write_json
from multimodal import cub_siglip_v2_experiment as exp
from multimodal.cub_siglip_data import photo_views
from multimodal.grounded_data import detector_parts,locate
from multimodal.visible_data import model_parts


@torch.no_grad()
def predict(checkpoint,image_path,candidates=None):
    exp.setup();p,m=exp.prepare();processor,detector=detector_parts()
    with Image.open(image_path) as image:record=locate(image.convert('RGB'),p['detector_prompts'],processor,detector,p)
    del detector,processor;gc.collect();torch.cuda.empty_cache()
    model,_,processor=model_parts('siglip2_b16');model=model.cuda().eval();pixels,valid=photo_views(image_path,record,processor)
    with torch.autocast('cuda',dtype=torch.bfloat16):features=F.normalize(model.get_image_features(pixel_values=pixels.cuda()).float(),dim=-1)
    del model;gc.collect();torch.cuda.empty_cache();fusion,saved=exp.restore(checkpoint);output=fusion(features[None],valid[None].cuda())
    logits=output['logits'][0].cpu();prob=output['probability'][0].cpu();candidates=list(range(100)) if candidates is None else candidates;scores=logits[candidates].softmax(-1)
    result=dict(version=exp.VERSION,variant=saved['config']['variant'],seed=saved['config']['seed'],step=saved['step'],predictions=[dict(label=candidates[int(i)],name=m['classes'][candidates[int(i)]]['name'],score=float(scores[i])) for i in scores.argsort(descending=True)[:5]],attributes=[dict(name=m['attributes'][int(i)]['name'],probability=float(prob[0,i])) for i in prob[0].argsort(descending=True)[:8]],regions=record['boxes'],note='Only image plus fixed local model and text assets. No ground-truth attributes, labels or part locations enter inference. Scores are uncalibrated. Default 100 candidates; final benchmark uses 75.')
    return result,logits,prob


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('image',type=Path);ap.add_argument('--variant',default='tokens');ap.add_argument('--seed',type=int,default=42);ap.add_argument('--output',type=Path);a=ap.parse_args()
    chosen=next(r for r in json.loads((exp.OUT/'selection_locked.json').read_text())['selected'] if r['variant']==a.variant and r['seed']==a.seed)
    result,_,_=predict(ROOT/chosen['checkpoint'],a.image)
    if a.output:write_json(a.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
