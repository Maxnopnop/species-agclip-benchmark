"""Predict from an image using a trained adapter and its fixed text assets."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from data_tools import digest
from multimodal.models import AlignedClassifier,load_backbone
from multimodal.preparation import load_attributes,locate_image,release


@torch.inference_mode()
def predict(checkpoint,image_path,attributes_path,device='cuda'):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    state=saved['state_dict'];config=saved['config']
    variant=saved.get('variant','baseline')
    if 'classes' not in saved:
        raise ValueError('Use a comparison variant checkpoint, not the alignment checkpoint.')
    with Image.open(image_path) as im:
        image=im.convert('RGB')
    found={'boxes':[],'attribute_ids':[],'scores':[]}
    if variant!='baseline':
        if attributes_path is None or digest(attributes_path)!=config['attributes_sha256']:
            raise ValueError('Use the exact attribute JSON used for training.')
        from transformers import OwlViTProcessor,OwlViTForObjectDetection
        attrs=load_attributes(attributes_path)
        processor=OwlViTProcessor.from_pretrained('google/owlvit-base-patch32')
        detector=OwlViTForObjectDetection.from_pretrained('google/owlvit-base-patch32').to(device).eval()
        found=locate_image(image,attrs['prompts'],processor,detector,device)
        del detector,processor
        release()
    backbone,transform,dim=load_backbone(config['backbone'])
    backbone=backbone.to(device).eval()
    global_feature=F.normalize(backbone(transform(image).unsqueeze(0).to(device)).float(),dim=-1)
    regions=torch.zeros(1,3,dim,device=device)
    ids=torch.zeros(1,3,dtype=torch.long,device=device)
    mask=torch.zeros(1,3,dtype=torch.bool,device=device)
    count=len(found['boxes'])
    if count:
        crops=torch.stack([transform(image.crop(tuple(box))) for box in found['boxes']]).to(device)
        regions[0,:count]=F.normalize(backbone(crops).float(),dim=-1)
        ids[0,:count]=torch.tensor(found['attribute_ids'],device=device)
        mask[0,:count]=True
    head=AlignedClassifier(config['backbone'],state['class_text'],state['attribute_text'],variant)
    head.load_state_dict(state);head=head.to(device).eval()
    logits,_=head(global_feature,regions,ids,mask)
    scores,labels=logits.softmax(-1)[0].topk(min(5,len(saved['classes'])))
    return dict(backbone=config['backbone'],variant=variant,pilot=config['pilot'],
                regions=found,predictions=[{'species':saved['classes'][i]['name'],
                'category_id':saved['classes'][i]['id'],'label':i,'score':s}
                for s,i in zip(scores.cpu().tolist(),labels.cpu().tolist())],
                note='Closed-set scores, not calibrated probabilities; fixed text assets are used internally.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--attributes')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--output')
    args=p.parse_args()
    torch.set_num_threads(4)
    result=predict(args.checkpoint,args.image,args.attributes,args.device)
    text=json.dumps(result,indent=2,ensure_ascii=False)
    if args.output:
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text+'\n',encoding='utf-8')
    print(text)


if __name__=='__main__':
    main()
