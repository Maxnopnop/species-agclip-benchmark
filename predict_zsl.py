"""One image in; prediction among predefined unseen or seen+unseen candidates."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from multimodal.models import load_backbone
from multimodal.zsl import ZSLHead
from prepare_expanded import five_crops

@torch.inference_mode()
def predict(checkpoint,image_path,candidate_set='unseen',device='cuda'):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True);config=saved['config'];p=config['protocol'];name=config['backbone']
    encoder,transform,_=load_backbone(name);encoder=encoder.to(device).eval()
    with Image.open(image_path) as im:im=im.convert('RGB')
    views=[im,*five_crops(im)];inputs=torch.stack([transform(v) for v in views]).to(device)
    padded=inputs.repeat(3,1,1,1)[:16];features=F.normalize(encoder(padded).float(),dim=-1)[:6]
    head=ZSLHead(name,saved['state_dict'],saved['variant'],p['seen_classes'],config['seed']);head.load_state_dict(saved['state_dict']);head=head.to(device).eval()
    logits,_=head(features[:1],features[1:].unsqueeze(0))
    candidates=p['eval_unseen_classes'] if candidate_set=='unseen' else sorted(p['seen_classes']+p['eval_unseen_classes'])
    probs=logits[0,candidates].softmax(-1).cpu();indices=probs.argsort(descending=True)
    result=dict(backbone=name,variant=saved['variant'],candidate_set=candidate_set,candidates=candidates,
        predictions=[dict(label=candidates[i],species=saved['classes'][candidates[i]]['name'],score=float(probs[i])) for i in indices.tolist()],
        note='Unseen relative to this adaptation run, not proven absent from pretraining. Fixed text embeddings are internal. Scores are not calibrated.')
    return result,probs

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--image',required=True);parser.add_argument('--candidate-set',choices=['unseen','all'],default='unseen');parser.add_argument('--output');args=parser.parse_args();torch.set_num_threads(4)
    result,_=predict(args.checkpoint,args.image,args.candidate_set);text=json.dumps(result,indent=2)
    if args.output:Path(args.output).write_text(text,encoding='utf-8')
    else:print(text)
