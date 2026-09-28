"""Pinned FG-CLIP experiment; reviewed custom code is loaded locally only."""
import os,json,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME',str(ROOT/'cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','120')
from huggingface_hub import snapshot_download

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ranged',action='store_true');args=parser.parse_args()
    repo='qihoo360/fg-clip-base';revision='454d76372c2cf5eb48fa0d871fd0534481484d97'
    folder=ROOT/'cache/visible_v1/models/fgclip_b16'
    patterns=['config.json','preprocessor_config.json','tokenizer.json','tokenizer_config.json',
        'special_tokens_map.json','vocab.json','merges.txt','modeling_fgclip.py','modeling_clip.py']
    if not args.ranged:patterns.append('model.safetensors')
    snapshot_download(repo,revision=revision,local_dir=folder,max_workers=2,allow_patterns=patterns)
    if args.ranged:
        import requests,time
        from concurrent.futures import ThreadPoolExecutor
        from huggingface_hub import HfApi
        info=HfApi().model_info(repo,revision=revision,files_metadata=True)
        lfs=next(x.lfs for x in info.siblings if x.rfilename=='model.safetensors')
        parts=ROOT/'work/fgclip_download_parts';parts.mkdir(parents=True,exist_ok=True);chunk=8*1024*1024
        def fetch(start):
            end=min(start+chunk,lfs.size)-1;path=parts/f'{start}.part'
            if path.exists() and path.stat().st_size==end-start+1:return path
            for attempt in range(4):
                try:
                    r=requests.get(f'https://huggingface.co/{repo}/resolve/{revision}/model.safetensors',headers={'Range':f'bytes={start}-{end}'},timeout=(20,60))
                    r.raise_for_status()
                    if r.headers.get('Content-Range')!=f'bytes {start}-{end}/{lfs.size}' or len(r.content)!=end-start+1:raise ValueError('Incorrect response range')
                    path.write_bytes(r.content);print(f'FG-CLIP chunk {start//chunk+1}/{(lfs.size+chunk-1)//chunk}',flush=True);return path
                except Exception:
                    if attempt==3:raise
                    time.sleep(2*(attempt+1))
        with ThreadPoolExecutor(max_workers=4) as pool:ordered=list(pool.map(fetch,range(0,lfs.size,chunk)))
        temporary=folder/'model.safetensors.assembling';sha=hashlib.sha256()
        with temporary.open('wb') as f:
            for path in ordered:
                block=path.read_bytes();sha.update(block);f.write(block)
        if sha.hexdigest()!=lfs.sha256:raise ValueError('Downloaded FG-CLIP weight hash mismatch')
        temporary.replace(folder/'model.safetensors')
    record=dict(repo=repo,revision=revision,weights='model.safetensors',code_sha256={
        name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ['modeling_fgclip.py','modeling_clip.py']})
    path=ROOT/'configs/fgclip_revision.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=record:raise ValueError('Pinned FG-CLIP files changed')
    path.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('FG-CLIP pinned download complete',flush=True)

if __name__=='__main__':main()
