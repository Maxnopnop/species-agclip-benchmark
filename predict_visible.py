"""Image-only inference: no class label, attribute label, or box is accepted."""
import argparse,json
from pathlib import Path
import torch
from PIL import Image
from data_tools import ROOT
from multimodal.visible_data import letterbox,patch_box,model_parts,encode_pixels
from multimodal.visible_train import VisibleHead

def load_predictor(checkpoint):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True);name=saved['config']['backbone'];state=saved['state_dict']
    head=VisibleHead(dict(class_text=state['class_text'],attribute_text=state['attribute_text']),saved['config']['variant'])
    head.load_state_dict(state);head=head.cuda().eval()
    if name=='fgclip_b16':
        from multimodal.fgclip_visible import model_parts as fg_parts,encode_pixels as fg_encode
        encoder,_,processor=fg_parts();encode=fg_encode
    else:
        encoder,_,processor=model_parts(name);encode=lambda m,p:encode_pixels(m,p,name)
    return saved,head,encoder.cuda().eval(),processor,encode

@torch.no_grad()
def predict_image(predictor,image):
    saved,head,encoder,processor,encode=predictor
    with Image.open(image) as im:padded,geometry=letterbox(im.convert('RGB'))
    # Same batch size as feature extraction to minimize batch-dependent rounding.
    pixels=processor(images=[padded]*8,return_tensors='pt')['pixel_values'].cuda()
    whole,patch=encode(encoder,pixels);valid=patch_box([0,0,1,1],geometry).unsqueeze(0).cuda()
    output=head(whole[:1],patch[:1],valid)
    return {k:v.cpu() for k,v in output.items()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--image',required=True);parser.add_argument('--output');args=parser.parse_args()
    torch.set_num_threads(4);predictor=load_predictor(args.checkpoint);saved=predictor[0];output=predict_image(predictor,args.image)
    probs=output['logits'][0].softmax(-1);order=probs.argsort(descending=True)[:5]
    result=dict(backbone=saved['config']['backbone'],variant=saved['config']['variant'],
        predictions=[dict(class_name=saved['classes'][i]['name'],common_name=saved['classes'][i].get('common_name'),probability=float(probs[i])) for i in order.tolist()],
        visible_attributes=[dict(description=a,score=float(s)) for a,s in zip(saved['attributes'],output['attribute_logits'][0].sigmoid())],
        note='Closed-set 20-species pilot. Attribute scores are uncalibrated; no expert or out-of-distribution reliability claim.')
    text=json.dumps(result,indent=2,ensure_ascii=False)
    if args.output:Path(args.output).write_text(text,encoding='utf-8')
    else:print(text)

if __name__=='__main__':main()
