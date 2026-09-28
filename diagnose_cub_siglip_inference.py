"""Audit SigLIP inference and text formatting using DEVELOPMENT only."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from transformers import SiglipProcessor
from data_tools import ROOT,write_json,digest
from multimodal.visible_data import model_parts
from multimodal.cub_siglip_data import load_data,text_bank,prepare
from multimodal.grounded_experiment import measures


@torch.no_grad()
def main():
    torch.set_num_threads(4);p,m=prepare();data=load_data('development');bank=text_bank();ids=[i for i,r in enumerate(data['rows']) if r['role']!='train_seen'];x=data['native_views'][ids,0];labels=data['labels'][ids]
    model,tokenizer,processor=model_parts('siglip2_b16');model=model.cuda().eval();styles={
        'original':[f'a photo of a {c["name"]}.' for c in m['classes']],
        'lowercase':[f'a photo of a {c["name"].lower()}.' for c in m['classes']],
        'official_template':[f'This is a photo of {c["name"]}.' for c in m['classes']],
        'official_lowercase':[f'this is a photo of {c["name"].lower()}.' for c in m['classes']]}
    rows=[]
    for style,texts in styles.items():
        emb=[]
        for i in range(0,100,32):
            batch=tokenizer(texts[i:i+32],padding='max_length',max_length=64,truncation=True,return_tensors='pt').to('cuda');emb.append(F.normalize(model.get_text_features(**batch).float(),dim=-1).cpu())
        text=torch.cat(emb)
        if style=='original':torch.testing.assert_close(text,bank['class_text'],atol=1e-6,rtol=1e-6)
        metrics,_=measures(20*x@text.T,labels,p,'development');rows.append(dict(style=style,metrics=metrics));print(style,metrics['H'],metrics['ZSL'],flush=True)
    official=SiglipProcessor(image_processor=processor,tokenizer=tokenizer)
    images=[]
    for i in ids[:3]:
        with Image.open(Path(m['image_root'])/data['rows'][i]['path']) as im:images.append(im.convert('RGB'))
    batch=official(images=images,text=styles['original'][:5],padding='max_length',max_length=64,truncation=True,return_tensors='pt').to('cuda');output=model(**batch)
    text=F.normalize(model.get_text_features(input_ids=batch['input_ids'],attention_mask=batch.get('attention_mask')).float(),dim=-1)
    visual=F.normalize(model.get_image_features(pixel_values=batch['pixel_values']).float(),dim=-1)
    manual=model.logit_scale.exp()*visual@text.T+model.logit_bias
    torch.testing.assert_close(manual,output.logits_per_image,rtol=1e-5,atol=1e-5)
    raw=tokenizer(styles['original'][:5],padding='max_length',max_length=64,truncation=True,return_tensors='pt');assert torch.equal(raw['input_ids'],batch['input_ids'].cpu())
    result=dict(source_sha256=digest(__file__),styles=rows,official_forward_max_error=float((manual-output.logits_per_image).abs().max()),cached_BF16_vs_FP32_max_feature_error=float((x[:3]-visual.cpu()).abs().max()),native_logit_scale=float(model.logit_scale.exp()),native_logit_bias=float(model.logit_bias),scope='Development-only audit; no final outcomes selected or changed. Official forward compared with manual normalized feature similarity. Text format variants are diagnostic, not silently substituted into trained experiments.')
    write_json(ROOT/'reports/cub_siglip_v1/inference_audit.json',result)


if __name__=='__main__':main()
