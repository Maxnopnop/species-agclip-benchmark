"""Fixed class-disjoint CUB subset, never selected using model scores."""
import json,random
from data_tools import ROOT,digest,write_json
from confidence_fresh import fingerprint,duplicates,index_file,CUB

DATA=ROOT/'data/coca_pilot_v1'
OUT=ROOT/'reports/coca_pilot_v1'
CONFIG=ROOT/'configs/coca_pilot_v1.json'

def validate(m):
    classes=m['classes'];groups={role:{x['label'] for x in classes if x['group']==role} for role in ['seen','dev','test']}
    assert list(map(len,[groups['seen'],groups['dev'],groups['test']]))==[10,5,10]
    assert not groups['seen']&groups['dev'] and not groups['seen']&groups['test'] and not groups['dev']&groups['test']
    rows=m['rows'];assert len(rows)==700 and len({x['image_id'] for x in rows})==700
    for r in rows:
        assert r['label'] in groups['seen' if r['role'] in ('train','val_seen','test_seen') else 'dev' if r['role']=='val_unseen' else 'test']
        assert r['official_train']==(r['role']=='train')
    for role,count in [('train',200),('val_seen',100),('val_unseen',100),('test_seen',100),('test_unseen',200)]:assert sum(r['role']==role for r in rows)==count

def prepare():
    target=DATA/'manifest.json'
    if target.exists():m=json.loads(target.read_text());validate(m);return m
    c=json.loads(CONFIG.read_text())
    old=json.loads((ROOT/'data/confidence_fresh_v1/manifest.json').read_text())
    used=set(old['excluded_species'])|{x['source_id'] for x in old['classes']};assert len(used)==130
    history=ROOT/'cache/confidence_fresh_v1/historical_fingerprints.json'
    refs=json.loads(history.read_text())['fingerprints']+old['rows']
    files=index_file('images.txt');labels=index_file('image_class_labels.txt',int);official=index_file('train_test_split.txt',int);names=index_file('classes.txt')
    eligible=[i for i in names if i not in used];random.Random(c['sampling_seed']).shuffle(eligible)
    rows=[];classes=[];rejected=[]
    for cid in eligible:
        label=len(classes);group='seen' if label<10 else 'dev' if label<15 else 'test'
        selected={};local=[]
        for train,n in ([(1,20),(0,20)] if group=='seen' else [(0,20)]):
            candidates=[i for i in files if labels[i]==cid and official[i]==train]
            random.Random(c['sampling_seed']+cid*2+train).shuffle(candidates);selected[train]=[]
            for iid in candidates:
                fp=fingerprint(CUB/'images'/files[iid])
                if duplicates(fp,refs+local,4):rejected.append(iid);continue
                selected[train].append((iid,fp));local.append(fp)
                if len(selected[train])==n:break
        if any(len(v)<20 for v in selected.values()):continue
        classes.append(dict(label=label,source_id=cid,name=names[cid].split('.',1)[1].replace('_',' '),group=group))
        for train,items in selected.items():
            for j,(iid,fp) in enumerate(items):
                role='train' if train else ('val_seen' if j<10 else 'test_seen') if group=='seen' else 'val_unseen' if group=='dev' else 'test_unseen'
                rows.append(dict(image_id=iid,path=files[iid],label=label,role=role,official_train=bool(train),**fp))
        refs+=local
        if len(classes)==25:break
    attrs=json.loads((ROOT/'data/cub_attributes_v1/manifest.json').read_text())['attributes']
    m=dict(classes=classes,rows=rows,attributes=[dict(prompt=a['prompt'],source_id=a['source_id']) for a in attrs],image_root=str(CUB/'images'),excluded_species=sorted(used),rejected=rejected)
    validate(m);write_json(target,m)
    write_json(OUT/'data_audit.json',dict(images=700,classes=classes,roles={role:sum(x['role']==role for x in rows) for role in sorted({r['role'] for r in rows})},historical_images=8742,rejected=rejected,excluded_species=sorted(used),manifest_sha256=digest(target),limitation='Custom seen/unseen split;not official150/50. Foundation pretraining exposure not excluded.'))
    return m

if __name__=='__main__':print(prepare()['classes'])
