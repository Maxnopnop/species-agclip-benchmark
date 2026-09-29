"""Image-only inference for the research branch, optionally beside native SigLIP2."""
import argparse
import gc
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from multimodal.attribute_path_v1 import ROOT, MODEL, PartLocalizer, AttributeReadout


def predict(image_path,seed=42,candidates='development',with_siglip=False):
    torch.set_num_threads(4)
    b=torch.load(ROOT/f'runs/attribute_path_v1/deployment_{seed}.pt',weights_only=True)
    from transformers import AutoModelForCausalLM, AutoImageProcessor
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True).eval().cuda()
    model.requires_grad_(False)
    processor=AutoImageProcessor.from_pretrained(MODEL,local_files_only=True)
    with Image.open(image_path) as im:image=im.convert('RGB')
    batch=processor(images=image,max_num_patches=b['max_patches'],return_tensors='pt').to('cuda')
    localizer=PartLocalizer(b['localizer']['initial'],b['localization_scale']).cuda()
    localizer.load_state_dict(b['localizer'])
    a,dim=b['readout']['weight'].shape
    dummy=dict(positive_text=torch.zeros(a,dim),negative_text=torch.zeros(a,dim),
               profiles=torch.zeros(len(b['classes']),a))
    readout=AttributeReadout(dummy,b['readout']['view_ids'],b['readout_scale']).cuda()
    readout.load_state_dict(b['readout'])
    with torch.no_grad():
        # Match the declared fp16 feature-cache boundary, then compute in fp32.
        patch=F.normalize(model.get_image_dense_feature(**batch).float(),dim=-1).half().float()
        patch=F.normalize(patch,dim=-1)
        whole=F.normalize(model.get_image_features(**batch).float(),dim=-1)
        attention=localizer(patch,batch['pixel_attention_mask'].bool())
        parts=F.normalize(attention@patch,dim=-1)
        scores,q=readout(torch.cat([whole[:,None],parts],1))
        scores=scores[0].cpu();q=q[0].cpu()
        h,w=map(int,batch['spatial_shapes'][0]);peaks=attention[0].argmax(-1).cpu()
    pool=b['development_classes'] if candidates=='development' else list(range(len(b['classes'])))
    def top(values):
        indices=torch.tensor(pool)[values[pool].argsort(descending=True)[:5]]
        return [dict(label=int(i),name=b['classes'][i]['name'],score=float(values[i])) for i in indices]
    winner=top(scores)[0]['label']
    contributions=b['readout_scale']*F.normalize(q,dim=0)*b['readout']['profiles'][winner]
    indices=contributions.abs().argsort(descending=True)[:10]
    result=dict(seed=seed,candidate_set=candidates,candidate_count=len(pool),attribute_branch_top5=top(scores),
        part_peaks_normalized={name:[float((int(i)%w+.5)/w),float((int(i)//w+.5)/h)]
                               for name,i in zip(['head','wing','breast','tail'],peaks)},
        largest_absolute_contributions=[dict(attribute=b['attributes'][i]['name'],
            evidence=float(q[i]),contribution=float(contributions[i])) for i in indices],
        note='Research attribute branch, not the recommended strongest classifier. Scores/contributions are not calibrated probabilities or causal explanations.')
    del model,processor,localizer,readout,batch,patch,whole,attention,parts
    gc.collect();torch.cuda.empty_cache()
    if with_siglip:
        from multimodal.visible_data import model_parts
        native,_,proc=model_parts('siglip2_b16');native=native.cuda().eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            feature=F.normalize(native.get_image_features(**proc(images=image,return_tensors='pt').to('cuda')).float(),dim=-1).cpu()
        result['siglip2_native_top5']=top((20*feature@b['siglip_class_text'].T)[0])
    return result,scores,q


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=[42,43,44],default=42)
    parser.add_argument('--candidates',choices=['development','all100'],default='development')
    parser.add_argument('--with-siglip',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();result,_,_=predict(args.image,args.seed,args.candidates,args.with_siglip)
    text=json.dumps(result,indent=2,ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()
