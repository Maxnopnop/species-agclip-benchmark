"""Extract a bounded, clearly separate pilot from the official archive prefix.

The full 100-species download runs independently. A readable partial gzip archive
is sufficient for a pipeline pilot; whole-archive MD5 is not yet available.
Individual images are decoded and SHA256-hashed. Pilot results are not the main
100-species benchmark and do not use the official validation protocol.
"""
import argparse
import json
import random
import tarfile
from io import BytesIO
from pathlib import Path
from collections import defaultdict
from PIL import Image
from data_tools import ROOT, digest, safe_relative, validate_manifest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--classes', type=int, default=4)
    args = p.parse_args()
    output = ROOT/'data'/'pilot'
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        print('Existing pilot retained:', manifest_path)
        return
    source = ROOT/'data'/'archives'/'train_mini.tar.gz'
    if not source.exists():
        source = source.with_suffix('.gz.part')
    if not source.exists():
        raise FileNotFoundError('Start download_data.cmd before making the pilot.')
    meta = json.loads((ROOT/'data'/'metadata'/'train_mini.json').read_text(encoding='utf-8'))
    categories = {c['id']:c for c in meta['categories']}
    paths = {r['file_name']:r for r in meta['images']}
    groups = defaultdict(list)
    images = output/'images'
    try:
        with tarfile.open(source, 'r|gz') as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith('.jpg'):
                    continue
                path = safe_relative(member.name)
                cid = int(path.split('/')[1].split('_')[0])
                if cid not in groups and len(groups) >= args.classes:
                    break
                blob = tar.extractfile(member).read()
                if len(blob) != member.size:
                    break
                with Image.open(BytesIO(blob)) as im:
                    im.verify()
                dest = images/path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(blob)
                original = paths[path]
                groups[cid].append({'path':path, 'category_id':cid,
                    'image_id':original['id'], 'license':original.get('license'),
                    'rights_holder':original.get('rights_holder'), 'sha256':digest(dest)})
    except (EOFError, tarfile.ReadError):
        pass
    usable = sorted(cid for cid, rows in groups.items() if len(rows)>=15)
    if len(usable) < args.classes:
        raise RuntimeError(f'Only {len(usable)} complete pilot classes available. Let download progress and retry.')
    classes, splits = [], {'train':[], 'val':[], 'test':[]}
    for label, cid in enumerate(usable[:args.classes]):
        classes.append(dict(categories[cid], label=label))
        rows = sorted(groups[cid], key=lambda r:r['path'])
        random.Random(42+cid).shuffle(rows)
        for split, start in [('train',0),('val',5),('test',10)]:
            splits[split].extend(dict(row,label=label) for row in rows[start:start+5])
    manifest = {'dataset':'Official iNaturalist archive prefix PILOT', 'synthetic':False,
                'pilot':True, 'seed':42, 'image_root':str(images), 'classes':classes, 'splits':splits,
                'source':'https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz',
                'protocol':'PILOT ONLY: 5 train / 5 validation / 5 test per class from train_mini. Not the main 100-species protocol.',
                'archive_md5_verified':source.suffix != '.part', 'licenses':meta.get('licenses',[])}
    validate_manifest(manifest)
    write_json(manifest_path,manifest)
    print('Pilot ready:',len(classes),'classes,',sum(map(len,splits.values())),'real images')


if __name__=='__main__':
    main()
