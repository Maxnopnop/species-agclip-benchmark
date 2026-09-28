"""Photo-only inference for the frozen-visual CUB follow-up experiments."""
import argparse,gc,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from data_tools import ROOT,write_json
from multimodal import cub_token_experiment as exp
from multimodal.cub_data import prepare
from multimodal.models import load_clip
from multimodal.grounded_data import detector_parts,locate,image_tensors


def configure(version):
    if version=='cub_bottleneck_v1':
        from cub_bottleneck_experiment import configure as change
        change()
    elif version=='cub_rich_v1':
        from cub_rich_experiment import configure as change
        change()
    elif version=='cub_sparse_v1':
        from cub_sparse_experiment import configure as change
        change()
    elif version=='cub_b16_v1':
        from cub_b16_experiment import configure as change
        change()
    elif version!='cub_tokens_v1': raise ValueError(version)


@torch.no_grad()
def predict(checkpoint,image_path,candidates=None):
    exp.setup();p,m=prepare();start=time.time()
    processor,detector=detector_parts()
    with Image.open(image_path) as image: record=locate(image.convert('RGB'),p['detector_prompts'],processor,detector,p)
    del processor,detector;gc.collect();torch.cuda.empty_cache()
    if exp.VERSION=='cub_b16_v1':
        from multimodal.visible_data import model_parts
        from open_clip.transform import image_transform
        clip,_,processor=model_parts('clip_b16')
        transform=image_transform(224,is_train=False,mean=processor.image_mean,std=processor.image_std,interpolation='bicubic',resize_mode='shortest')
    else:clip,transform,_=load_clip()
    clip=clip.cuda().eval()
    images,_,_,valid=image_tensors(image_path,record,transform)
    # Exactly the original cache extraction: one image's three views per batch.
    with torch.autocast('cuda',dtype=torch.bfloat16):
        encoded=clip.get_image_features(pixel_values=images.cuda()) if exp.VERSION=='cub_b16_v1' else clip.encode_image(images.cuda())
        features=F.normalize(encoded.float(),dim=-1)
    del clip;gc.collect();torch.cuda.empty_cache()
    model,saved=exp.restore(checkpoint);output=model(features[None],valid[None].cuda())
    logits=output['logits'][0].cpu();probability=output['probability'][0].cpu()
    candidates=list(range(20)) if candidates is None else candidates;scores=logits[candidates].softmax(-1)
    bank=exp.text_bank()
    if 'positive_prompts' in bank:attribute_names=bank['positive_prompts']
    else:attribute_names=[a['prompt'] for a in m['attributes']]
    result=dict(version=exp.VERSION,variant=saved['config']['variant'],seed=saved['config']['seed'],step=saved['step'],
                predictions=[dict(label=candidates[int(i)],name=m['classes'][candidates[int(i)]]['name'],score=float(scores[i])) for i in scores.argsort(descending=True)[:5]],
                attributes=[dict(name=attribute_names[int(i)],probability=float(probability[0,i])) for i in probability[0].argsort(descending=True)[:8]],regions=record['boxes'],seconds=time.time()-start,
                note='Image-only input with fixed detector/text assets. No true image attributes or parts read. Scores are uncalibrated; default candidates are all 20 species, unlike the 16-candidate final benchmark.')
    return result,logits,probability


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('image',type=Path);parser.add_argument('--version',default='cub_tokens_v1',choices=['cub_tokens_v1','cub_bottleneck_v1','cub_rich_v1','cub_sparse_v1','cub_b16_v1']);parser.add_argument('--variant',default='tokens');parser.add_argument('--seed',type=int,default=42);parser.add_argument('--output',type=Path);a=parser.parse_args();configure(a.version)
    lock=json.loads((exp.OUT/'selection_locked.json').read_text());chosen=next(r for r in lock['selected'] if r['variant']==a.variant and r['seed']==a.seed)
    result,_,_=predict(ROOT/chosen['checkpoint'],a.image)
    if a.output:write_json(a.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
