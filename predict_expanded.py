"""Image-only user input; fixed attribute embeddings are stored in the checkpoint."""
import argparse
import json
import torch
from torch.nn import functional as F
from PIL import Image
from multimodal.models import load_backbone
from multimodal.expanded import ExpandedHead
from prepare_expanded import five_crops


@torch.inference_mode()
def predict(checkpoint,image_path,device='cuda'):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    name=saved['config']['backbone'];variant=saved['variant']
    encoder,transform,_=load_backbone(name);encoder=encoder.to(device).eval()
    with Image.open(image_path) as im:im=im.convert('RGB')
    views=[im] if variant=='baseline' else [im,*five_crops(im)]
    features=F.normalize(encoder(torch.stack([transform(v) for v in views]).to(device)).float(),dim=-1)
    head=ExpandedHead(name,saved['state_dict'],variant);head.load_state_dict(saved['state_dict']);head=head.to(device).eval()
    regions=features[1:].unsqueeze(0) if variant!='baseline' else torch.zeros(1,5,features.shape[-1],device=device)
    logits,_=head(features[:1],regions);scores,ids=logits.softmax(-1)[0].topk(5)
    return dict(backbone=name,variant=variant,predictions=[dict(label=i,species=saved['classes'][i]['name'],score=s)
        for s,i in zip(scores.cpu().tolist(),ids.cpu().tolist())],
        note='20-species closed-set development model. Scores are not calibrated probabilities. Fixed attribute bank is internal.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--image',required=True)
    p.add_argument('--device',default='cuda');args=p.parse_args();torch.set_num_threads(4)
    print(json.dumps(predict(args.checkpoint,args.image,args.device),indent=2))
