"""Run each partition in a fresh process; lock all before any final scoring."""
import argparse,json,subprocess,sys
from data_tools import ROOT,write_json
from multimodal import cub_resplit_data as data
from multimodal import cub_resplit_experiment as runner


def configure(backbone,split):
    base=json.loads((ROOT/'configs/cub_resplit_v1.json').read_text());assert backbone in base['backbones'] and split in base['class_partition_seeds']
    data.BACKBONE=backbone;data.SPLIT=split;version=f'cub_resplit_{backbone}_{split}'
    runner.VERSION=version;runner.OUT=ROOT/'runs'/version;runner.REPORT=ROOT/'reports'/version;runner.CONFIG=ROOT/'configs'/f'{version}.json'
    config=dict(base,version=version,backbone=backbone,split_seed=split)
    if runner.CONFIG.exists():assert json.loads(runner.CONFIG.read_text())==config
    else:runner.CONFIG.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8',newline='\n')


def protocol():
    p,m=data.prepare();train=[r for r in m['rows'] if r['role']=='train_seen'];dev=[r for r in m['rows'] if r['role'].startswith('dev_')];final=[r for r in m['rows'] if r['role'].startswith('eval_')]
    assert len(train)==500 and len(dev)==500 and len(final)==750
    assert set(p['seen_classes']).isdisjoint(p['dev_unseen_classes']) and set(p['seen_classes']).isdisjoint(p['eval_unseen_classes']) and set(p['dev_unseen_classes']).isdisjoint(p['eval_unseen_classes'])
    assert len({r['sha256'] for r in m['rows']})==1750
    assert all(r['official_split']=='train' for r in train+dev) and all(r['official_split']=='test' for r in final)
    write_json(runner.REPORT/'protocol.json',dict(split_seed=data.SPLIT,backbone=data.BACKBONE,seen=p['seen_classes'],dev_unseen=p['dev_unseen_classes'],final_unseen=p['eval_unseen_classes'],attributes=m['attributes'],images=dict(train=len(train),development=len(dev),final=len(final)),invariants_passed=True,scope=p['scope']))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['train','evaluate','all','matrix']);ap.add_argument('--backbone',default='clip_b16');ap.add_argument('--split',type=int,default=20260929);a=ap.parse_args()
    if a.stage=='matrix':
        c=json.loads((ROOT/'configs/cub_resplit_v1.json').read_text());jobs=[(b,s) for b in c['backbones'] for s in c['class_partition_seeds']]
        for stage in ['train','evaluate']:
            for backbone,split in jobs:
                print(stage,backbone,split,flush=True);subprocess.run([sys.executable,'-X','utf8',__file__,stage,'--backbone',backbone,'--split',str(split)],check=True,cwd=ROOT)
    else:
        configure(a.backbone,a.split)
        if a.stage in ['train','all']:protocol();runner.train_all()
        if a.stage in ['evaluate','all']:runner.final_evaluation()
