"""Native SigLIP 2 preprocessing on the frozen 100-species photo protocol."""
import gc,json
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from data_tools import ROOT,digest
from .cub100_data import prepare,regions,prompt,DATA,source as parent_source
from .visible_data import model_parts

CACHE=ROOT/'cache/cub_siglip_v1'


def metadata():
    return dict(source_sha256=digest(__file__),config_sha256=digest(ROOT/'configs/cub_siglip_v1.json'),manifest_sha256=digest(DATA/'manifest.json'),parent_code_sha256=digest(ROOT/'multimodal/cub100_data.py'),revision=json.loads((ROOT/'configs/visible_model_revisions.json').read_text())['siglip2_b16'])


def text_bank():
    p,m=prepare();CACHE.mkdir(parents=True,exist_ok=True);path=CACHE/'text.pt';meta=metadata()
    if path.exists():
        saved=torch.load(path,weights_only=True);assert saved['metadata']==meta;return saved
    positives=[prompt(a['name']) for a in m['attributes']];negatives=[s.replace('a bird with','a bird without') for s in positives]
    texts=[f'a photo of a {c["name"]}.' for c in m['classes']]+positives+negatives
    model,tokenizer,_=model_parts('siglip2_b16');model=model.cuda().eval();features=[]
    with torch.no_grad():
        for i in range(0,len(texts),32):
            batch=tokenizer(texts[i:i+32],padding='max_length',max_length=64,truncation=True,return_tensors='pt').to('cuda')
            features.append(F.normalize(model.get_text_features(**batch).float(),dim=-1).cpu())
    text=torch.cat(features);a=len(positives);result=dict(metadata=meta,class_text=text[:100],attribute_text=text[100:100+a],negative_text=text[100+a:],positive_prompts=positives,negative_prompts=negatives)
    torch.save(result,path);del model;gc.collect();torch.cuda.empty_cache();return result


def photo_views(path,record,processor):
    with Image.open(path) as image:
        image=image.convert('RGB');views=[image]+[image.crop(box) for box in record['boxes']]
        valid=torch.zeros(2,dtype=torch.bool);valid[:len(views)-1]=True
        while len(views)<3:views.append(image)
        return processor(images=views,return_tensors='pt')['pixel_values'],valid


def load_data(stage):
    p,m=prepare();CACHE.mkdir(parents=True,exist_ok=True);path=CACHE/f'features_{stage}.pt';meta=metadata()
    if path.exists():
        saved=torch.load(path,weights_only=True);assert saved['metadata']==meta;return saved
    region=regions(stage);roles=['train_seen','dev_seen','dev_unseen'] if stage=='development' else ['eval_seen','eval_unseen'];rows=[r for r in m['rows'] if r['role'] in roles]
    raw=torch.load(DATA/'raw_targets.pt',weights_only=True);assert raw['source']==parent_source();index={r['image_id']:i for i,r in enumerate(m['rows'])}
    model,_,processor=model_parts('siglip2_b16');model=model.cuda().eval();features=[];valid=[];targets=[]
    # SigLIP's native resize sees the entire rectangular crop (no center crop),
    # so visibility masks must be recomputed, never copied from CLIP's cache.
    with torch.no_grad():
        for i,row in enumerate(rows):
            file=Path(m['image_root'])/row['path'];assert digest(file)==row['sha256'];record=region['images'][row['path']];images,ok=photo_views(file,record,processor)
            with torch.autocast('cuda',dtype=torch.bfloat16):f=F.normalize(model.get_image_features(pixel_values=images.cuda()).float(),dim=-1).cpu()
            features.append(f);valid.append(ok);view_targets=[]
            for x1,y1,x2,y2 in [[0,0,*row['size']],*record['boxes']]:
                visible={int(pid) for pid,(x,y,shown) in row['parts'].items() if shown and x1<=x<=x2 and y1<=y<=y2}
                target=raw['raw_targets'][index[row['image_id']]].clone()
                for a,attribute in enumerate(m['attributes']):
                    if not visible & set(attribute['part_ids']):target[a]=-1
                view_targets.append(target)
            while len(view_targets)<3:view_targets.append(torch.full_like(view_targets[0],-1))
            targets.append(torch.stack(view_targets))
            if i%100==0:print(f'SigLIP 2 {stage}: {i}/{len(rows)}',flush=True)
    result=dict(metadata=meta,rows=rows,native_views=torch.stack(features),valid=torch.stack(valid),targets=torch.stack(targets),labels=torch.tensor([r['label'] for r in rows]))
    torch.save(result,path);del model;gc.collect();torch.cuda.empty_cache();return result
