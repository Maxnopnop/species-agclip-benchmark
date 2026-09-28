"""Prepare a fixed, taxonomy-selected 20-species development benchmark."""
import argparse
import json
import random
import tarfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from PIL import Image
import torch
from torch.nn import functional as F
from data_tools import ROOT,write_json,digest,safe_relative,validate_manifest
from run import load_manifest
from multimodal.models import BACKBONES,load_backbone
from multimodal.preparation import all_rows,release

SEED=20260928


def prepare():
    path=ROOT/'data/expanded20/manifest.json'
    if path.exists():
        return load_manifest(path)
    classes=json.loads((ROOT/'configs/expanded_classes.json').read_text())
    meta=json.loads((ROOT/'data/metadata/train_mini.json').read_text())
    records={r['file_name']:r for r in meta['images']}
    wanted={c['id']:c['label'] for c in classes}
    source=ROOT/'data/archives/train_mini.tar.gz'
    if not source.exists(): source=source.with_suffix('.gz.part')
    groups=defaultdict(list);images=path.parent/'images'
    try:
        with tarfile.open(source,'r|gz') as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith('.jpg'):continue
                relative=safe_relative(member.name)
                cid=int(relative.split('/')[1].split('_')[0])
                if cid not in wanted:continue
                blob=tar.extractfile(member).read()
                if len(blob)!=member.size:break
                with Image.open(BytesIO(blob)) as im:im.verify()
                dest=images/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(blob)
                original=records[relative]
                groups[cid].append(dict(path=relative,category_id=cid,label=wanted[cid],image_id=original['id'],
                    license=original.get('license'),rights_holder=original.get('rights_holder'),sha256=digest(dest)))
                if all(len(groups[k])>=50 for k in wanted):break
    except (EOFError,tarfile.ReadError):pass
    if any(len(groups[k])<50 for k in wanted):
        raise RuntimeError('The fixed selected species are not fully downloaded yet.')
    splits={s:[] for s in ('train','val','test')}
    for cid,label in wanted.items():
        rows=sorted(groups[cid],key=lambda r:r['path'])
        random.Random(SEED+cid).shuffle(rows)
        for split,start,end in [('train',0,30),('val',30,40),('test',40,50)]:splits[split].extend(rows[start:end])
    # Exact duplicate content cannot cross splits, even when filenames differ.
    hashes={s:{r['sha256'] for r in rows} for s,rows in splits.items()}
    if any(hashes[a]&hashes[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Duplicate image content crosses a split; review before training.')
    manifest=dict(dataset='iNaturalist 2021 expanded development subset',pilot=False,development=True,
        synthetic=False,seed=SEED,image_root=str(images),classes=classes,splits=splits,licenses=meta.get('licenses',[]),
        selection='20 species: two randomly selected available species in each of 10 families; exclude all four earlier pilot species. Fixed before performance evaluation. 10 insects and 10 plants.',
        protocol='30 candidate training / 10 validation / 10 test per species, all from train_mini; separate from formal 100-species official-val evaluation.',
        selection_config_sha256=digest(ROOT/'configs/expanded_classes.json'),archive_md5_verified=False,
        source='https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz')
    validate_manifest(manifest);write_json(path,manifest)
    print('Prepared 20 species / 1000 verified JPEGs.',flush=True)
    return manifest


def five_crops(image):
    w,h=image.size;cw=max(1,round(w*.65));ch=max(1,round(h*.65))
    return [image.crop((x,y,x+cw,y+ch)) for x,y in
            [(0,0),(w-cw,0),(0,h-ch),(w-cw,h-ch),((w-cw)//2,(h-ch)//2)]]


def features(name,device):
    manifest_path=ROOT/'data/expanded20/manifest.json';manifest=load_manifest(manifest_path)
    target=ROOT/f'cache/expanded20/{name}_features.pt';target.parent.mkdir(parents=True,exist_ok=True)
    protocol=dict(manifest_sha256=digest(manifest_path),backbone=name,
        crop_policy='five fixed 65%-width/height overlapping corner+center windows; no detector and no label-conditioned crop selection',
        pretrained=True,frozen=True)
    if target.exists():
        saved=torch.load(target,weights_only=True)
        if saved['metadata']!=protocol:raise ValueError('Expanded feature cache mismatch')
        return
    model,transform,dim=load_backbone(name);model=model.to(device).eval()
    rows=all_rows(manifest);globals_=[];regions=[]
    with torch.inference_mode():
        for start in range(0,len(rows),8):
            batch=rows[start:start+8];tensors=[]
            for row in batch:
                with Image.open(Path(manifest['image_root'])/row['path']) as im:im=im.convert('RGB')
                tensors.extend(transform(v) for v in [im,*five_crops(im)])
            values=[]
            for b in range(0,len(tensors),16):
                values.append(F.normalize(model(torch.stack(tensors[b:b+16]).to(device)).float(),dim=-1).cpu())
            values=torch.cat(values).view(len(batch),6,dim)
            globals_.append(values[:,0]);regions.append(values[:,1:])
            if start%160==0:print(f'{name}: {min(start+8,len(rows))}/{len(rows)} images',flush=True)
    torch.save(dict(metadata=protocol,rows=rows,classes=manifest['classes'],global_features=torch.cat(globals_),
        region_features=torch.cat(regions),labels=torch.tensor([r['label'] for r in rows])),target)
    del model;release()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',nargs='+',choices=BACKBONES,default=list(BACKBONES))
    p.add_argument('--device',default='cuda');p.add_argument('--data-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);prepare()
    if not args.data_only:
        for name in args.models:features(name,args.device)


if __name__=='__main__':main()
