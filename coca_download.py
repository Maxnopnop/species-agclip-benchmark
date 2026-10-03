"""Pinned public CoCa weights; download without executing remote code."""
from pathlib import Path
import time,requests
from data_tools import ROOT,digest,write_json

REVISION='74207cb7fde8eafc9864451ebd332fa8e75b150f'
SHA256='73725652298ad76ed2162caffdae96d8653a05d7a29b6281103e4df81d0ff8ea'
SIZE=2554109637
FOLDER=ROOT/'cache/coca_vitl14'
WEIGHT=FOLDER/'open_clip_pytorch_model.bin'
URL=f'https://huggingface.co/laion/CoCa-ViT-L-14-laion2B-s13B-b90k/resolve/{REVISION}/open_clip_pytorch_model.bin'

def download():
    FOLDER.mkdir(parents=True,exist_ok=True)
    if WEIGHT.exists():assert digest(WEIGHT)==SHA256;return WEIGHT
    part=WEIGHT.with_suffix('.bin.part')
    for attempt in range(5):
        try:
            offset=part.stat().st_size if part.exists() else 0
            if offset==SIZE:break
            r=requests.get(URL,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,90));r.raise_for_status()
            if offset and r.status_code!=206:offset=0
            with part.open('ab' if offset else 'wb') as f:
                last=time.monotonic();total=offset
                for chunk in r.iter_content(4*1024*1024):
                    f.write(chunk);total+=len(chunk)
                    if time.monotonic()-last>15:print('download MiB',round(total/1024**2),'/',round(SIZE/1024**2),flush=True);last=time.monotonic()
            assert part.stat().st_size==SIZE;break
        except Exception as ex:
            print(type(ex).__name__,str(ex)[:180],flush=True)
            if attempt==4:raise
            time.sleep(3)
    assert part.stat().st_size==SIZE and digest(part)==SHA256
    part.replace(WEIGHT)
    write_json(FOLDER/'verified.json',dict(repository='laion/CoCa-ViT-L-14-laion2B-s13B-b90k',revision=REVISION,sha256=SHA256,size=SIZE))
    return WEIGHT

if __name__=='__main__':print(download(),flush=True)
