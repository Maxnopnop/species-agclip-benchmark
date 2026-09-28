"""Download pinned official CLIP B/16 and SigLIP 2 Base checkpoints to E: cache."""
import os
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME',str(ROOT/'cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','120')
from huggingface_hub import HfApi,snapshot_download


def main():
    target=ROOT/'configs/visible_model_revisions.json'
    revisions=json.loads(target.read_text(encoding='utf-8')) if target.exists() else {}
    for name,repo in [('clip_b16','openai/clip-vit-base-patch16'),('siglip2_b16','google/siglip2-base-patch16-224')]:
        info=HfApi().model_info(repo,revision=revisions.get(name,{}).get('revision','main'))
        names={f.rfilename for f in info.siblings}
        weights='model.safetensors' if 'model.safetensors' in names else 'pytorch_model.bin'
        revision=dict(repo=repo,revision=info.sha,weights=weights)
        revisions[name]=revision;target.write_text(json.dumps(revisions,indent=2),encoding='utf-8')
        print(f'Downloading {name} {info.sha}',flush=True)
        snapshot_download(repo,revision=info.sha,allow_patterns=[weights,'config.json','preprocessor_config.json',
            'tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt','spiece.model'],
            max_workers=2,local_dir=ROOT/f'cache/visible_v1/models/{name}')
        print(f'Complete: {name}',flush=True)


if __name__=='__main__':main()
