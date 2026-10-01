"""Image-only inference for the exploratory local-attribute guarded fusion."""
import argparse
import gc
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from local_attribute_experiment import ROOT, RUN, OUT, CONFIG, predict_fused, select_candidate
from attribute_fusion_experiment import standardized, save
from multimodal.attribute_path_v1 import MODEL, AttributeReadout, PartLocalizer


def export_policy():
    c=json.loads(CONFIG.read_text())
    e=torch.load(RUN/'evaluation.pt',weights_only=True)
    attrs=e['attributes']['repaired_local'];native=e['native'];labels=e['labels']
    candidates=[dict(weight=w,penalty=g) for w in c['weights'] for g in c['seen_penalties']]
    predictions=torch.stack([predict_fused(native[None],attrs,t['weight'],t['penalty'],True,c['native_margin_threshold']) for t in candidates])
    index,_=select_candidate(predictions,labels,native.argmax(-1),candidates,True,c['max_unseen_new_error_fraction'])
    result=dict(source=e['source'],variant='repaired_local',**candidates[index],margin_threshold=c['native_margin_threshold'],
                candidate_labels=list(range(75)),scope='All-development selection for future image-only inference; no independent deployment accuracy claim. Fixed 75-candidate protocol only.')
    save(OUT/'inference_policy.json',result)
    return result


def predict(path,seed=42):
    torch.set_num_threads(4)
    policy=json.loads((OUT/'inference_policy.json').read_text())
    ck=torch.load(RUN/f'repaired_local_{seed}.pt',weights_only=True)
    assert ck['source']==policy['source']
    old=torch.load(ROOT/f'runs/attribute_path_v1/deployment_{seed}.pt',weights_only=True)
    assert old['source']==ck['source']['previous']
    from transformers import AutoModelForCausalLM, AutoImageProcessor
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True).eval().cuda()
    proc=AutoImageProcessor.from_pretrained(MODEL,local_files_only=True)
    with Image.open(path) as im:picture=im.convert('RGB')
    batch=proc(images=picture,max_num_patches=old['max_patches'],return_tensors='pt').to('cuda')
    loc=PartLocalizer(old['localizer']['initial'],old['localization_scale']).cuda();loc.load_state_dict(old['localizer'])
    a,dim=ck['state']['weight'].shape
    dummy=dict(positive_text=torch.zeros(a,dim),negative_text=torch.zeros(a,dim),profiles=torch.zeros(100,a))
    head=AttributeReadout(dummy,ck['state']['view_ids'],20.).cuda();head.load_state_dict(ck['state'])
    assert a==144 and bool((head.view_ids>0).all())
    with torch.no_grad():
        patch=F.normalize(model.get_image_dense_feature(**batch).float(),dim=-1).half().float()
        patch=F.normalize(patch,dim=-1)
        attention=loc(patch,batch['pixel_attention_mask'].bool())
        local=F.normalize(attention@patch,dim=-1)
        # No whole-image FG-CLIP2 feature is computed or passed to the readout.
        scores,q=head(torch.cat([torch.zeros_like(local[:,:1]),local],dim=1))
        scores=scores[0,:75].cpu();q=q[0].cpu()
    del model,proc,loc,head,batch,patch,attention,local
    gc.collect();torch.cuda.empty_cache()
    from multimodal.visible_data import model_parts
    native,_,proc=model_parts('siglip2_b16');native=native.cuda().eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        feat=F.normalize(native.get_image_features(**proc(images=picture,return_tensors='pt').to('cuda')).float(),dim=-1).cpu()
        ns=(20*feat@old['siglip_class_text'].T)[0,:75]
    pred=int(predict_fused(ns[None],scores[None],policy['weight'],policy['penalty'],True,policy['margin_threshold'])[0])
    initial=int(ns.argmax())
    top=standardized(ns).topk(2).values;margin=float(top[0]-top[1])
    result=dict(seed=seed,candidate_count=75,native_label=initial,native_class=old['classes'][initial]['name'],
                prediction=pred,predicted_class=old['classes'][pred]['name'],changed=pred!=initial,
                native_standardized_margin=margin,change_allowed=margin<=policy['margin_threshold'],
                policy={k:policy[k] for k in ['weight','penalty','margin_threshold']},
                note=policy['scope'])
    return result,ns,scores,q


def main():
    p=argparse.ArgumentParser();p.add_argument('--export-policy',action='store_true');p.add_argument('--image',type=Path)
    p.add_argument('--seed',type=int,choices=[42,43,44],default=42);p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.export_policy:print(json.dumps(export_policy(),indent=2));return
    if args.image is None:p.error('--image is required unless --export-policy is used')
    result,_,_,_=predict(args.image,args.seed)
    if args.output:save(args.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
