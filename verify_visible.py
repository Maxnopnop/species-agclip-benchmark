"""Check image-only/cached parity and export local attention audit (photos not published)."""
import json
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from data_tools import ROOT,write_json
from predict_visible import load_predictor,predict_image
from multimodal.visible_train import batch_forward
from multimodal.visible_data import letterbox

def main():
    torch.set_num_threads(4);summary=json.loads((ROOT/'reports/visible_v1/selected_summary.json').read_text(encoding='utf-8'));checks=[]
    for row in [r for r in summary if r['variant']=='attribute_region']:
        name=row['backbone'];checkpoint=ROOT/f'runs/visible_v1/{name}_attribute_region_seed42_lr{row["lr"]:g}/best.pt'
        predictor=load_predictor(checkpoint);cache=torch.load(ROOT/f'cache/visible_v1/{name}_features.pt',weights_only=True)
        for k in ('global_features','patch_features','valid_patches'):cache[k]=cache[k].cuda()
        ids=[i for i,r in enumerate(cache['rows']) if r['split']=='val' and any(a==1 for a in r['attributes'])]
        # Fixed first three annotated validation images; not selected for successful outcomes.
        fig,axes=plt.subplots(3,3,figsize=(11,10))
        for position,i in enumerate(ids[:3]):
            sample=cache['rows'][i];path=Path(json.loads((ROOT/'data/visible_v1/manifest.json').read_text(encoding='utf-8'))['image_root'])/sample['path']
            fresh=predict_image(predictor,path)
            with torch.no_grad():cached=batch_forward(predictor[1],cache,[i])
            fresh_probs=fresh['logits'].softmax(-1);cached_probs=cached['logits'].softmax(-1).cpu()
            torch.testing.assert_close(fresh_probs,cached_probs,rtol=1e-4,atol=1e-5)
            torch.testing.assert_close(fresh['attribute_logits'],cached['attribute_logits'].cpu(),rtol=1e-4,atol=1e-4)
            checks.append(dict(backbone=name,review_id=sample['review_id'],max_probability_error=float((fresh_probs-cached_probs).abs().max()),prediction_matches=bool(fresh_probs.argmax()==cached_probs.argmax())))
            with Image.open(path) as im:padded,_=letterbox(im.convert('RGB'))
            a=next(a for a,t in enumerate(sample['attributes']) if t==1)
            axes[position,0].imshow(padded);axes[position,0].set_title(f'Review {sample["review_id"]}; true: {sample["label"]}\npredicted: {int(fresh_probs.argmax())}')
            axes[position,1].imshow(padded);heat=fresh['attention'][0,a].reshape(14,14).numpy()
            axes[position,1].imshow(heat,extent=(0,224,224,0),alpha=.55,cmap='magma');axes[position,1].set_title(f'Attribute {a}: score {float(fresh["attribute_logits"][0,a].sigmoid()):.2f}')
            axes[position,2].imshow(cache['evidence_masks'][i,a].reshape(14,14),vmin=0,vmax=1,cmap='gray');axes[position,2].set_title('Provisional evidence mask\n'+cache['attributes'][a],fontsize=9,wrap=True)
            for ax in axes[position]:ax.axis('off')
        fig.suptitle(f'{name}: fixed first 3 annotated validation images; not proof of semantic grounding')
        fig.tight_layout();folder=ROOT/'work/visible_v1';folder.mkdir(parents=True,exist_ok=True);fig.savefig(folder/f'{name}_attention_audit.png',dpi=140);plt.close(fig)
        del predictor,cache;torch.cuda.empty_cache()
    write_json(ROOT/'reports/visible_v1/inference_verification.json',dict(scope='3 fixed annotated validation images per selected full-AG backbone, seed42; image-only inference vs cached evaluation',checks=checks))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
