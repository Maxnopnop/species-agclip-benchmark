"""Validation-only text interventions, retrieval proxies and local visual audit sheets."""
import json
import random
import textwrap
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from data_tools import ROOT,write_json
from multimodal.diagnostics import ControlledHead,load_feature_cache
from multimodal.expanded import ExpandedHead,evaluate
from multimodal.experiment import split_indices
from prepare_expanded import five_crops


def main():
    torch.set_num_threads(4);out=ROOT/'reports/diagnostics_v1';out.mkdir(exist_ok=True,parents=True)
    bank=torch.load(ROOT/'cache/expanded20/text_bank.pt',weights_only=True)
    attrs=json.loads((ROOT/'configs/expanded_attributes.json').read_text());manifest=json.loads((ROOT/'data/expanded20/manifest.json').read_text())
    entries={e['category_id']:e['attribute_ids'] for e in attrs['species']}
    allowed=torch.zeros(20,len(attrs['prompts']),dtype=torch.bool)
    for c in manifest['classes']:allowed[c['label'],entries[c['id']]]=True
    groups=torch.tensor([0 if c['supercategory']=='Insects' else 1 for c in manifest['classes']])
    attrgroup=torch.empty(len(attrs['prompts']),dtype=torch.long)
    for c in manifest['classes']:attrgroup[entries[c['id']]]=groups[c['label']]
    visual_rows=[];picked=[]
    for label in range(20):
        options=sorted([r for r in manifest['splits']['val'] if r['label']==label],key=lambda r:r['path'])
        picked.append(random.Random(20260928+label).choice(options))
    proxies=[];interventions=[]
    with torch.no_grad():
        for name in ['efficientnet_b0','clip_vit_b32']:
            cache=load_feature_cache(name);vi=split_indices(cache,'val');labels=cache['labels'][vi]
            for seed in [42,43,44]:
                base=ROOT/f'runs/expanded20/{name}_shots10_seed{seed}'
                for variant in ['ag_mean','ag_attention','ag_aux']:
                    saved=torch.load(base/variant/'best.pt',weights_only=True);head=ControlledHead(name,bank,'matched',seed,variant).eval()
                    head.load_state_dict(saved['state_dict'],strict=False);ref=None;refpred=None
                    original_gate=head.gate.detach().clone()
                    for mode in ['matched','permuted','zero','no_branch']:
                        head.mode='matched' if mode=='no_branch' else mode
                        head.gate.copy_(torch.zeros_like(original_gate) if mode=='no_branch' else original_gate)
                        metric,p,y,confusion=evaluate(head,cache,vi)
                        if mode=='matched':ref=p;refpred=p.argmax(-1)
                        interventions.append(dict(backbone=name,seed=seed,variant=variant,mode=mode,**metric,
                            flipped_predictions=int((p.argmax(-1)!=refpred).sum()),
                            mean_probability_l1=float((p-ref).abs().sum(-1).mean())))
                stages=['alignment','ag_attention']+(['native_clip'] if seed==42 and name=='clip_vit_b32' else [])
                for stage in stages:
                    if stage=='native_clip':regions=cache['region_features'][vi]
                    else:
                        saved=torch.load(base/stage/'best.pt',weights_only=True)
                        h=ExpandedHead(name,bank,'baseline' if stage=='alignment' else stage).eval()
                        h.load_state_dict(saved['state_dict']);regions=h.project(cache['region_features'][vi])
                    ids=(regions@bank['attribute_text'].T).topk(4,dim=-1).indices
                    own=allowed[labels[:,None,None],ids]
                    samegroup=attrgroup[ids[:,:,0]]==groups[labels][:,None]
                    proxies.append(dict(backbone=name,seed=seed,stage=stage,crops=1000,
                        own_species_bank_top1_fraction=float(own[:,:,0].float().mean()),
                        own_species_bank_any_top4_fraction=float(own.any(-1).float().mean()),
                        insect_plant_group_top1_fraction=float(samegroup.float().mean()),
                        uniform_random_top1_reference=float(allowed[labels].float().mean()),
                        note='Class-list membership is a weak proxy, not visible-attribute accuracy. Other species may share a trait; listed traits may be invisible.'))
                    if stage=='ag_attention' and seed==42:
                        positions={cache['rows'][i]['path']:j for j,i in enumerate(vi)}
                        for row in picked:
                            attrid=int(ids[positions[row['path']],4,0])
                            visual_rows.append(dict(backbone=name,label=row['label'],species=manifest['classes'][row['label']]['name'],
                                path=row['path'],crop='center_65_percent',top1_attribute_id=attrid,prompt=attrs['prompts'][attrid]))
    write_json(out/'posthoc_interventions.json',interventions);write_json(out/'retrieval_proxies.json',proxies)
    write_json(out/'visual_audit_samples.json',visual_rows)
    local=ROOT/'work/diagnostics_v1';local.mkdir(exist_ok=True,parents=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19);title=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
    for page in range(4):
        canvas=Image.new('RGB',(1300,1250),'white');draw=ImageDraw.Draw(canvas)
        for k,row in enumerate(picked[page*5:page*5+5]):
            y=k*250
            with Image.open(Path(manifest['image_root'])/row['path']) as im:im=im.convert('RGB')
            crop=five_crops(im)[4]
            for x,img in [(0,im),(290,crop)]:
                img=img.copy();img.thumbnail((280,210));canvas.paste(img,(x,y+35))
            draw.text((0,y+5),f'{row["label"]}: full image / center crop',fill='black',font=title)
            draw.text((590,y+8),manifest['classes'][row['label']]['name'],fill='black',font=title)
            for j,name in enumerate(['efficientnet_b0','clip_vit_b32']):
                record=next(r for r in visual_rows if r['backbone']==name and r['path']==row['path'])
                draw.multiline_text((590,y+48+j*88),name+':\n'+'\n'.join(textwrap.wrap(record['prompt'],65)),fill='black',font=font,spacing=3)
        canvas.save(local/f'audit_page_{page+1}.jpg',quality=95)
    print('Saved validation-only interventions, weak retrieval proxies and four local visual audit sheets.')


if __name__=='__main__':main()
