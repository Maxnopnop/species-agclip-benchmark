"""Run the fixed 100-species benchmark after data and attributes are ready."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from data_tools import ROOT
from run import load_manifest
from multimodal.preparation import load_attributes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attributes', required=True, help='Reviewed attribute JSON covering all 100 species')
    p.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    args=p.parse_args()
    manifest_path=ROOT/'data'/'manifest.json'
    manifest=load_manifest(manifest_path)
    attrs=load_attributes(args.attributes)
    required={c['id'] for c in manifest['classes']}
    covered={s['category_id'] for s in attrs.get('species',[])}
    if manifest.get('pilot') or len(required)!=100:
        raise ValueError('Formal matrix requires the fixed 100-species manifest.')
    if not attrs.get('species_descriptions_reviewed') or not required.issubset(covered):
        raise ValueError('Review and source the attribute descriptions for all 100 species first.')
    protocol=json.loads((ROOT/'configs'/'benchmark.json').read_text(encoding='utf-8'))
    shared=['--manifest',str(manifest_path),'--attributes',str(Path(args.attributes).resolve()),
            '--cache',str(ROOT/'cache'/'main_multimodal'),'--output',str(ROOT/'runs'/'main_multimodal'),
            '--device',args.device,'--models',*protocol['backbones']]
    command=[sys.executable,'-u',str(ROOT/'benchmark.py')]
    subprocess.run([*command,'prepare',*shared],check=True,cwd=ROOT)
    for shots in protocol['shots']:
        for seed in protocol['seeds']:
            subprocess.run([*command,'train',*shared,'--shots',str(shots),'--seed',str(seed),
                '--alignment-epochs',str(protocol['alignment_epochs']),
                '--comparison-epochs',str(protocol['comparison_epochs'])],check=True,cwd=ROOT)


if __name__=='__main__':
    main()
