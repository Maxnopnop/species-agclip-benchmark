"""Complete the visual -> text-adapted -> attribute-adapted comparison."""
import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from data_tools import ROOT,digest,write_json
from run import seed_all
from multimodal.models import DIMS
from multimodal.expanded import ExpandedHead,train,evaluate,is_better
from multimodal.experiment import split_indices

CONFIG=ROOT/'configs/three_stage_v1.json'
OUT=ROOT/'reports/three_stage_v1'
RUN=ROOT/'runs/three_stage_v1'
OLD=ROOT/'runs/expanded20'


class VisualLinear(nn.Module):
    """No text, attribute, or region input contributes to this classifier."""
    variant='baseline'
    def __init__(self,backbone,classes):
        super().__init__();self.projection=nn.Linear(DIMS[backbone],classes)
    def forward(self,g,r):
        return self.projection(g),g


def random_bank(classes,attributes,seed):
    generator=torch.Generator().manual_seed(seed)
    return dict(class_text=F.normalize(torch.randn(classes,512,generator=generator),dim=-1),
                attribute_text=torch.zeros(attributes,512),class_attributes=torch.zeros(classes,512))


def build(name,variant,bank,c):
    if variant=='visual_linear':return VisualLinear(name,len(bank['class_text']))
    if variant=='random_codes':return ExpandedHead(name,random_bank(len(bank['class_text']),len(bank['attribute_text']),c['random_code_seed']))
    return ExpandedHead(name,bank,variant)


def provenance(c):
    paths=[CONFIG,Path(__file__),ROOT/'multimodal/expanded.py',ROOT/'multimodal/experiment.py',ROOT/'multimodal/models.py',ROOT/'run.py',
           ROOT/'configs/expanded_protocol.json',ROOT/'data/expanded20/manifest.json',ROOT/'configs/expanded_attributes.json',
           ROOT/'cache/expanded20/text_bank.pt',OLD/'selection_locked.json']
    paths += [ROOT/f'cache/expanded20/{name}_features.pt' for name in c['backbones']]
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def lock(c):
    s=provenance(c);path=OUT/'protocol.json'
    if path.exists():
        prior=json.loads(path.read_text());assert prior['source']==s and prior['config']==c,'Use a new version for protocol changes.'
    else:write_json(path,dict(source=s,config=c,locked_at_utc=datetime.now(timezone.utc).isoformat()))
    return s


def cpu_cache(name,bank):
    path=ROOT/f'cache/expanded20/{name}_features.pt'
    cache=torch.load(path,weights_only=True)
    assert cache['metadata']['manifest_sha256']==bank['manifest_sha256']
    assert len(cache['rows'])==1000 and len(cache['classes'])==20
    for split,count in [('train',600),('val',200),('test',200)]:assert len(split_indices(cache,split))==count
    sets=[{cache['rows'][i]['path'] for i in split_indices(cache,split)} for split in ['train','val','test']]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    return cache


def train_all(c,s):
    bank=torch.load(ROOT/'cache/expanded20/text_bank.pt',weights_only=True)
    done=0
    for name in c['backbones']:
        full=cpu_cache(name,bank)
        # Training receives no test images, labels or features.
        ids=[i for i,r in enumerate(full['rows']) if r['split']!='test']
        cache={k:(v[ids].cuda() if k in ['global_features','region_features','labels'] else [v[i] for i in ids] if k=='rows' else v) for k,v in full.items()}
        del full
        for shots in c['shots']:
            for seed in c['seeds']:
                ti=split_indices(cache,'train',shots,seed);vi=split_indices(cache,'val')
                for variant in c['new_variants']:
                    folder=RUN/f'{name}_shots{shots}_seed{seed}'/variant
                    path=folder/'best.pt'
                    if path.exists():
                        ck=torch.load(path,weights_only=True);assert ck['source']==s
                        done+=1;continue
                    seed_all(seed);model=build(name,variant,bank,c).cuda()
                    state,m,d=train(model,cache,ti,vi,c['stage_steps'][0],seed,folder/'stage1')
                    stage1=dict(validation=m,diagnostic=d)
                    seed_all(seed);model=build(name,variant,bank,c).cuda();model.load_state_dict(state)
                    state,m,d=train(model,cache,ti,vi,c['stage_steps'][1],seed,folder/'stage2')
                    training,_,_,_=evaluate(model,cache,ti)
                    record=dict(backbone=name,shots=shots,seed=seed,variant=variant,source=s,validation=m,diagnostic=d,stage1=stage1,
                                training_metrics=training,parameters=sum(p.numel() for p in model.parameters()),
                                train_paths=[cache['rows'][i]['path'] for i in ti])
                    torch.save(record|dict(state_dict=state),path)
                    write_json(folder/'validation.json',record)
                    done+=1;write_json(OUT/'status.json',dict(stage='training',completed=done,total=90))
                    print('trained',done,'/90',name,shots,seed,variant,'val',round(m['top1_accuracy'],3),'train',round(training['top1_accuracy'],3),flush=True)
                    del model
        del cache;torch.cuda.empty_cache()


def seal(c,s):
    new=[]
    for name in c['backbones']:
        for shots in c['shots']:
            for seed in c['seeds']:
                for variant in c['new_variants']:
                    path=RUN/f'{name}_shots{shots}_seed{seed}'/variant/'best.pt'
                    ck=torch.load(path,weights_only=True);assert ck['source']==s
                    new.append(dict(backbone=name,shots=shots,seed=seed,variant=variant,sha256=digest(path),validation=ck['validation'],
                                    stage1_step=ck['stage1']['diagnostic']['best_step'],stage2_step=ck['diagnostic']['best_step']))
    historical=json.loads((OLD/'selection_locked.json').read_text())
    assert len(new)==90 and len(historical)==15
    record=dict(source=s,new_checkpoints=new,historical_ag_choices=historical)
    path=OUT/'selection_locked.json'
    if path.exists():assert json.loads(path.read_text())==record
    else:write_json(path,record)
    return record


def evaluate_all(c,s):
    selection=seal(c,s)
    bank=torch.load(ROOT/'cache/expanded20/text_bank.pt',weights_only=True)
    rows=[];predictions={};checks=[]
    old_protocol=json.loads((ROOT/'configs/expanded_protocol.json').read_text())
    for name in c['backbones']:
        cache=cpu_cache(name,bank);idx=split_indices(cache,'test')
        paths=[cache['rows'][i]['path'] for i in idx]
        for shots in c['shots']:
            for seed in c['seeds']:
                olddir=OLD/f'{name}_shots{shots}_seed{seed}'
                conf=json.loads((olddir/'config.json').read_text())
                assert conf['protocol']==old_protocol
                assert conf['implementation_sha256']==digest(ROOT/'multimodal/expanded.py')
                assert conf['cache_sha256']==s[str(Path(f'cache/expanded20/{name}_features.pt'))]
                assert conf['manifest_sha256']==bank['manifest_sha256'] and conf['attributes_sha256']==bank['attributes_sha256']
                # Prove the new controls selected exactly the historical training images.
                expected=[cache['rows'][i]['path'] for i in split_indices(cache,'train',shots,seed)]
                for variant in c['new_variants']+c['reused_variants']:
                    new=variant in c['new_variants']
                    folder=(RUN/f'{name}_shots{shots}_seed{seed}' if new else olddir)/variant
                    ck=torch.load(folder/'best.pt',weights_only=True,map_location='cpu')
                    if new:
                        assert ck['source']==s and ck['train_paths']==expected
                    model=build(name,variant,bank,c);model.load_state_dict(ck['state_dict'])
                    m,prob,y,cm=evaluate(model,cache,idx)
                    if not new:
                        historical=torch.load(folder/'test_predictions.pt',weights_only=True)
                        assert historical['paths']==paths and np.array_equal(historical['labels'].numpy(),y)
                        torch.testing.assert_close(prob,historical['probabilities'],atol=1e-6,rtol=1e-5)
                        prior=json.loads((folder/'test_metrics.json').read_text())
                        for key in ['top1_accuracy','macro_f1']:assert abs(m[key]-prior[key])<1e-10
                        checks.append(dict(backbone=name,shots=shots,seed=seed,variant=variant,
                                           checkpoint_sha256=digest(folder/'best.pt'),prediction_max_error=float((prob-historical['probabilities']).abs().max())))
                    record=dict(backbone=name,shots=shots,seed=seed,variant=variant,**m,parameters=sum(p.numel() for p in model.parameters()))
                    if new:record['training_accuracy']=ck['training_metrics']['top1_accuracy']
                    rows.append(record)
                    key=f'{name}/{shots}/{seed}/{variant}'
                    predictions[key]=dict(probabilities=prob,labels=torch.tensor(y),paths=paths)
        print('verified and evaluated',name,flush=True)
    assert len(rows)==315 and len(checks)==225
    RUN.mkdir(parents=True,exist_ok=True)
    torch.save(dict(source=s,predictions=predictions),RUN/'predictions.pt')
    write_json(OUT/'results.json',dict(source=s,rows=rows))
    write_json(OUT/'verification.json',dict(source=s,new_cells=90,historical_cells_reproduced=225,
                                          same_training_paths=True,test_rows_excluded_from_training=True,historical_checks=checks))
    write_json(OUT/'status.json',dict(stage='complete',new_cells=90,total_evaluated_cells=315))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['lock','train','evaluate','all'])
    args=parser.parse_args();torch.set_num_threads(4)
    c=json.loads(CONFIG.read_text());s=lock(c)
    if args.action in ['train','all']:train_all(c,s)
    if args.action in ['evaluate','all']:evaluate_all(c,s)


if __name__=='__main__':main()
