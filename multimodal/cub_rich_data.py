"""Training-supported expanded attributes with the original photo split."""
import json
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from data_tools import ROOT, digest
from .cub_data import prepare, BASE, lines, text_bank as original_bank
from .cub_token_experiment import load_data as original_data
from .models import load_clip

PARTS = {'bill':[2], 'wing':[9,13], 'upperparts':[1,9,13], 'underparts':[3,4],
         'back':[1], 'upper_tail':[14], 'under_tail':[14], 'breast':[4], 'throat':[15],
         'forehead':[6], 'nape':[10], 'belly':[3], 'leg':[8,12], 'crown':[5],
         'eye':[7,11], 'head':[2,5,6,7,10,11,15], 'tail':[14],
         'primary':list(range(1,16)), 'shape':list(range(1,16))}
CACHE = ROOT / 'cache/cub_rich_v1'


def definition():
    return json.loads((ROOT / 'runs/cub_coverage_v1/definition.json').read_text())


def names(): return {int(r[0])-1:r[1] for r in lines(ROOT / 'data/cub/attributes.txt')}


def text_bank():
    CACHE.mkdir(parents=True,exist_ok=True); path=CACHE/'text.pt'
    metadata=dict(source_sha256=digest(__file__),definition_sha256=digest(ROOT/'runs/cub_coverage_v1/definition.json'))
    if path.exists():
        saved=torch.load(path,weights_only=True); assert saved['metadata']==metadata; return saved
    p,m=prepare(); source=original_bank(); columns=definition()['extended']; naming=names(); old={a['source_id']-1:i for i,a in enumerate(m['attributes'])}
    positives=[]; negatives=[]
    for column in columns:
        group,value=naming[column].split('::'); field=group[4:]; value=value.replace('_',' ').replace('(','').replace(')','')
        if field.endswith('_color'): prompt=f'a bird with {value} {field[:-6].replace("_"," ")} color.'
        elif field.endswith('_pattern'): prompt=f'a bird with a {value} pattern on its {field[:-8].replace("_"," ")}.'
        elif field.endswith('_shape'): prompt=f'a bird with a {value} {field[:-6].replace("_"," ")} shape.'
        elif field=='bill_length': prompt=f'a bird with a bill {value}.'
        elif field=='shape': prompt=f'a bird with a {value} body shape.'
        else: raise ValueError(field)
        positives.append(prompt); negatives.append(prompt.replace('a bird with','a bird without'))
    model,_,tokenizer=load_clip(); model=model.cuda().eval(); prompts=positives+negatives
    with torch.no_grad(): text=torch.cat([model.encode_text(tokenizer(prompts[i:i+32]).cuda(),normalize=True).float().cpu() for i in range(0,len(prompts),32)])
    pos,neg=text.chunk(2)
    # Preserve exact wording and vectors for the original 24 attributes.
    for i,column in enumerate(columns):
        if column in old:
            j=old[column]; pos[i]=source['attribute_text'][j]; neg[i]=source['negative_text'][j]
            positives[i]=m['attributes'][j]['prompt']; negatives[i]=m['attributes'][j]['negative_prompt']
    result=dict(metadata=metadata,class_text=source['class_text'],attribute_text=pos,negative_text=neg,columns=columns,positive_prompts=positives,negative_prompts=negatives)
    torch.save(result,path); del model; torch.cuda.empty_cache(); return result


def load_data(stage):
    data=original_data(stage); CACHE.mkdir(parents=True,exist_ok=True); path=CACHE/f'targets_{stage}.pt'
    metadata=dict(source_sha256=digest(__file__),definition_sha256=digest(ROOT/'runs/cub_coverage_v1/definition.json'), regions_sha256=digest(ROOT/f'cache/cub_attributes_v1/regions_{stage}.json'))
    if path.exists():
        saved=torch.load(path,weights_only=True); assert saved['metadata']==metadata
        data['targets']=saved['targets']; return data
    columns=definition()['extended']; naming=names(); index={r['image_id']:i for i,r in enumerate(data['rows'])}; parts=defaultdict(dict); observations={}
    for r in lines(BASE/'parts/part_locs.txt'):
        if int(r[0]) in index: parts[int(r[0])][int(r[1])]=(float(r[2]),float(r[3]),int(r[4]))
    selected={i+1:j for j,i in enumerate(columns)}
    for r in lines(BASE/'attributes/image_attribute_labels.txt'):
        iid,aid,pos,certainty=map(int,r[:4])
        if iid in index and aid in selected and certainty>=3: observations[iid,aid-1]=pos
    regions=json.loads((ROOT/f'cache/cub_attributes_v1/regions_{stage}.json').read_text())['images']
    targets=torch.full((len(index),3,len(columns)),-1.)
    for row in data['rows']:
        iid=row['image_id']; boxes=[[0,0,*row['size']],*regions[row['path']]['boxes']]
        for v,(x1,y1,x2,y2) in enumerate(boxes):
            cx,cy=(x1+x2)/2,(y1+y2)/2; side=min(x2-x1,y2-y1)
            visible={k for k,(x,y,ok) in parts[iid].items() if ok and cx-side/2<=x<=cx+side/2 and cy-side/2<=y<=cy+side/2}
            for a,column in enumerate(columns):
                field=naming[column].split('::')[0][4:]; key=next((k for k in sorted(PARTS,key=len,reverse=True) if field==k or field.startswith(k+'_')),None)
                if key is not None and (iid,column) in observations and visible & set(PARTS[key]): targets[index[iid],v,a]=observations[iid,column]
    _,manifest=prepare(); common=[columns.index(a['source_id']-1) for a in manifest['attributes']]
    torch.testing.assert_close(targets[:,:,common],data['targets'],rtol=0,atol=0)
    torch.save(dict(metadata=metadata,targets=targets),path); data['targets']=targets; return data
