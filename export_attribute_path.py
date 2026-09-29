"""Export inference-only bundles; no training images or annotations are included."""
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent


def main():
    assets=json.loads((ROOT/'configs/attribute_path_v1_assets.json').read_text())
    c=json.loads((ROOT/'configs/attribute_path_v1.json').read_text())
    loc=torch.load(ROOT/'runs/attribute_path_v1/localizers.pt',weights_only=True)
    text=torch.load(ROOT/'cache/cub_siglip_v2/text.pt',weights_only=True)['class_text']
    for seed in c['seeds']:
        readout=torch.load(ROOT/f'runs/attribute_path_v1/learned_part_{seed}.pt',weights_only=True)
        assert readout['source']==loc['source']==assets['source']
        bundle=dict(source=assets['source'],classes=assets['classes'],attributes=assets['attributes'],
            development_classes=assets['seen_classes']+assets['dev_unseen_classes'],
            localizer=loc['states'][str(seed)],readout=readout['state'],siglip_class_text=text,
            localization_scale=c['localization_scale'],readout_scale=c['readout_scale'],
            max_patches=c['max_patches'],seed=seed)
        torch.save(bundle,ROOT/f'runs/attribute_path_v1/deployment_{seed}.pt')
    print('Exported three inference-only bundles.')


if __name__=='__main__':main()
