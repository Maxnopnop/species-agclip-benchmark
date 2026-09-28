"""Download the official CUB archive in bounded ranges; verify MD5 before extraction."""
import concurrent.futures,hashlib,time,tarfile
from pathlib import Path
import requests
from data_tools import ROOT,digest,safe_relative,write_json

URL='https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
MD5='97eceeb196236b17998738112f37df78'
SIZE=1150585339

def main():
    folder=ROOT/'data/cub';folder.mkdir(parents=True,exist_ok=True);target=folder/'CUB_200_2011.tgz';partial=target.with_suffix('.tgz.part')
    if not target.exists():
        offset=partial.stat().st_size if partial.exists() else 0
        if offset>SIZE:raise ValueError('Oversized partial archive')
        step=4*1024*1024
        def chunk(bounds):
            start,end=bounds
            for attempt in range(5):
                try:
                    with requests.get(URL+'&cachebust='+str(time.time_ns()),headers={'Range':f'bytes={start}-{end}'},timeout=(15,60)) as response:
                        response.raise_for_status()
                        if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{SIZE}':raise ValueError('Invalid byte-range response')
                        content=response.content
                        if len(content)!=end-start+1:raise ValueError('Incomplete range')
                        return content
                except (requests.RequestException,ValueError):
                    if attempt==4:raise
                    time.sleep(2**attempt)
        ranges=iter((i,min(SIZE,i+step)-1) for i in range(offset,SIZE,step));started=time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool,partial.open('ab') as output:
            pending=[]
            for _ in range(8):
                bounds=next(ranges,None)
                if bounds:pending.append(pool.submit(chunk,bounds))
            last=0
            while pending:
                data=pending.pop(0).result();output.write(data);offset+=len(data)
                bounds=next(ranges,None)
                if bounds:pending.append(pool.submit(chunk,bounds))
                if time.time()-last>10:print(f'CUB {offset/2**20:.1f}/{SIZE/2**20:.1f} MiB; {time.time()-started:.0f}s',flush=True);last=time.time()
        if digest(partial,'md5')!=MD5:raise ValueError('Official archive MD5 mismatch')
        partial.replace(target)
    if digest(target,'md5')!=MD5:raise ValueError('Archive MD5 mismatch')
    marker=folder/'verified.json'
    if marker.exists():print('CUB already verified and extracted.',flush=True);return
    count=0
    with tarfile.open(target,'r|gz') as archive:
        for member in archive:
            if member.isdir():continue
            if not member.isfile():raise ValueError('Archive contains non-regular member')
            relative=safe_relative(member.name);dest=folder/relative
            if not dest.resolve().is_relative_to(folder.resolve()):raise ValueError('Unsafe extraction path')
            dest.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as source,dest.open('wb') as output:
                while block:=source.read(1024*1024):output.write(block)
            count+=1
    write_json(marker,dict(source=URL,official_md5=MD5,bytes=SIZE,files=count));print(f'Official MD5 verified; extracted {count} files.',flush=True)

if __name__=='__main__':main()
