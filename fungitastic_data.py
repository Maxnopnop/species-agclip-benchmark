"""Small temporal FungiTastic subset from author's public Kaggle mirror."""
import json,random,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import requests
from PIL import Image
from data_tools import ROOT,digest,write_json
from confidence_fresh import fingerprint,duplicates

DATA=ROOT/'data/fungitastic_confidence_v1'
RAW=ROOT/'data/fungitastic'
OUT=ROOT/'reports/fungitastic_confidence_v1'
BASE='https://www.kaggle.com/api/v1/datasets/download/picekl/fungitastic/'
ATTRIBUTES=[f'a mushroom with {a}.' for a in [
 'a red cap','a brown cap','a yellow cap','a white cap','a purple cap','a grey cap',
 'a green cap','a pink cap','a flat cap','a bell-shaped cap','a convex cap',
 'a funnel-shaped cap','a scaly cap','a smooth cap','a striped cap margin',
 'white spots on its cap','white gills','brown gills','pink gills','yellow gills',
 'widely spaced gills','crowded gills','pores under the cap','a white stem',
 'a brown stem','a yellow stem','a thin stem','a thick stem','a ring on its stem',
 'a swollen stem base']]

def fetch(remote,local):
    local=Path(local)
    if local.exists():return local
    local.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(4):
        try:
            response=requests.get(BASE+requests.utils.quote(remote,safe=''),timeout=60)
            response.raise_for_status()
            tmp=local.with_suffix(local.suffix+'.part');tmp.write_bytes(response.content)
            if local.suffix.lower() in ('.jpg','.jpeg'):
                with Image.open(tmp) as im:im.verify()
            else:pd.read_csv(tmp,nrows=1)
            tmp.replace(local);return local
        except Exception:
            if attempt==3:raise
            time.sleep(2**attempt)

def metadata():
    frames={}
    for split,year in [('Val',2022),('Test',2023)]:
        name=f'FungiTastic-Mini-{split}.csv'
        p=fetch(f'metadata/FungiTastic-Mini/{name}',RAW/name)
        d=pd.read_csv(p,low_memory=False)
        assert (pd.to_datetime(d.eventDate).dt.year==year).all()
        assert (d.year==year).all()
        assert d.groupby('observationID').species.nunique().max()==1
        frames[split]=d.sort_values('filename').drop_duplicates('observationID')
    assert not set(frames['Val'].observationID)&set(frames['Test'].observationID)
    return frames

def validate_rows(rows):
    assert len(rows)==500 and len({r['observation_id'] for r in rows})==500
    for r in rows:
        expected=2023 if r['split']=='test' else 2022
        assert r['year']==expected and pd.Timestamp(r['date']).year==expected
    for label in range(10):
        for split,num in [('train',20),('val',10),('test',20)]:
            assert sum(r['label']==label and r['split']==split for r in rows)==num

def prepare(c):
    target=DATA/'manifest.json'
    if target.exists():
        m=json.loads(target.read_text());validate_rows(m['rows']);return m
    frames=metadata();counts={s:d.groupby('species').size() for s,d in frames.items()}
    eligible=sorted(n for n in counts['Val'].index if counts['Val'][n]>=35 and counts['Test'].get(n,0)>=25)
    random.Random(c['sampling_seed']).shuffle(eligible)
    history_path=ROOT/'cache/confidence_fresh_v1/historical_fingerprints.json'
    refs=json.loads(history_path.read_text())['fingerprints']
    recent=json.loads((ROOT/'data/confidence_fresh_v1/manifest.json').read_text())['rows']
    refs=refs+recent;historical_count=len(refs)
    rows=[];classes=[];rejected=[];downloads=0
    for species in eligible:
        selected={};localrefs=[]
        for official,needed in [('Val',30),('Test',20)]:
            candidates=frames[official][frames[official].species==species].to_dict('records')
            random.Random(c['sampling_seed']+int(candidates[0]['category_id'])*2+(official=='Test')).shuffle(candidates)
            selected[official]=[]
            for start in range(0,len(candidates),needed+5):
                batch=candidates[start:start+needed+5]
                def download(row):
                    name=row['filename'];assert Path(name).name==name
                    path=DATA/'images'/official.lower()/name
                    fetch(f'images/FungiTastic-Mini/{official.lower()}/500p/{name}',path)
                    return row,path,fingerprint(path)
                with ThreadPoolExecutor(max_workers=6) as executor:items=list(executor.map(download,batch))
                downloads+=len(items)
                for row,path,fp in items:
                    if duplicates(fp,refs+localrefs,c['dedup_hamming_threshold']):
                        rejected.append(dict(filename=row['filename'],reason='historical/new hash match'));continue
                    selected[official].append((row,path,fp));localrefs.append(fp)
                    if len(selected[official])==needed:break
                if len(selected[official])==needed:break
            if len(selected[official])<needed:break
        if len(selected.get('Val',[]))<30 or len(selected.get('Test',[]))<20:continue
        label=len(classes);classes.append(dict(label=label,name=species,source_id=int(selected['Val'][0][0]['category_id'])))
        for official in ['Val','Test']:
            for i,(r,path,fp) in enumerate(selected[official]):
                split='test' if official=='Test' else 'train' if i<20 else 'val'
                rows.append(dict(label=label,path=path.relative_to(DATA/'images').as_posix(),split=split,
                    observation_id=int(r['observationID']),year=int(r['year']),date=r['eventDate'],**fp))
        refs+=localrefs
        print('selected',label+1,species,'downloaded',downloads,flush=True)
        if len(classes)==10:break
    validate_rows(rows)
    m=dict(classes=classes,attributes=[dict(source_id=i,prompt=p) for i,p in enumerate(ATTRIBUTES)],rows=rows,image_root=str(DATA/'images'),
        metadata_hashes={str(p.relative_to(ROOT)):digest(p) for p in [RAW/'FungiTastic-Mini-Val.csv',RAW/'FungiTastic-Mini-Test.csv']},
        historical_fingerprints_sha256=digest(history_path),prior_fresh_manifest_sha256=digest(ROOT/'data/confidence_fresh_v1/manifest.json'))
    write_json(target,m)
    write_json(OUT/'data_audit.json',dict(classes=classes,images=500,train=200,val=100,test=200,unique_observations=500,
        years={'train':2022,'val':2022,'test':2023},historical_images_screened=historical_count,rejected=rejected,downloaded=downloads,
        eligible_species=len(eligible),metadata_hashes=m['metadata_hashes'],source='https://github.com/BohemianVRA/FungiTastic',mirror='https://www.kaggle.com/datasets/picekl/fungitastic',
        limits='eventDate is observation date,not independently verified shutter date. OWL-ViT pretraining overlap unaudited. No captions,coordinates or observation identity fed into models.'))
    return m

if __name__=='__main__':prepare(json.loads((ROOT/'configs/fungitastic_confidence_v1.json').read_text()))
