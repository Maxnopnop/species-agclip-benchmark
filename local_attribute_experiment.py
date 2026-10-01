"""Train subset-only attribute heads, then cross-fit bounded guarded fusion."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from attribute_fusion_experiment import ROOT, digest, save, standardized, stratified_folds, metrics, changes, summarize
from multimodal.attribute_path_v1 import AttributeReadout, attribute_view_indices, pooled_views, source as old_source

OUT = ROOT/'reports/local_attribute_v1'
RUN = ROOT/'runs/local_attribute_v1'
CONFIG = ROOT/'configs/local_attribute_v1.json'


def source():
    paths = [CONFIG, Path(__file__), ROOT/'attribute_fusion_experiment.py',
             ROOT/'cache/attribute_path_v1/dense.pt', ROOT/'runs/attribute_path_v1/localizers.pt',
             ROOT/'cache/cub_siglip_v1/features_development.pt', ROOT/'cache/cub_siglip_v2/text.pt']
    return dict(files={str(p.relative_to(ROOT)): digest(p) for p in paths}, previous=old_source())


def lock(c):
    path = OUT/'protocol.json'
    s = source()
    if path.exists():
        record = json.loads(path.read_text())
        assert record['source'] == s and record['config'] == c, 'Protocol changed; version new scientific changes.'
    else:
        save(path, dict(locked_at_utc=datetime.now(timezone.utc).isoformat(), source=s, config=c))
    return s


def subset_data(d, keep):
    return {k: d[k][keep] for k in ['positive_text', 'negative_text']} | {'profiles': d['profiles'][:, keep]}


def subset_scores(q, profiles, scale=20.):
    return scale * F.normalize(q, dim=-1) @ F.normalize(profiles, dim=-1).T


def predict_fused(native, attributes, weight, penalty=0., guarded=False, threshold=.5):
    """Only score tensors and fixed candidate membership enter inference."""
    nz = standardized(native)
    base = nz.argmax(-1)
    score = nz + weight * standardized(attributes)
    score = score.clone()
    score[..., :50] -= penalty
    pred = score.argmax(-1)
    if guarded:
        top = nz.topk(2, dim=-1).values
        pred = torch.where(top[..., 0]-top[..., 1] <= threshold, pred, base)
    return pred


def select_candidate(predictions, labels, native_pred, candidates, guarded, max_fraction):
    """Caller passes fitting rows only. No held-out labels or predictions needed."""
    native = metrics(native_pred, labels)
    stats = []
    best = 0
    best_h = -float('inf')
    for index, by_seed in enumerate(predictions):
        ms = [metrics(pred, labels) for pred in by_seed]
        h = float(np.mean([m['H'] for m in ms]))
        u = float(np.mean([m['U'] for m in ms]))
        damage = float(np.mean([changes(pred, native_pred, labels)['unseen']['newly_wrong'] for pred in by_seed]))
        eligible = not guarded or (u >= native['U']-1e-10 and h >= native['H']-1e-10 and damage <= max_fraction*int((labels >= 50).sum())+1e-10)
        stats.append(dict(H=h, U=u, new_unseen_errors=damage, eligible=eligible))
        if eligible and h > best_h+1e-10:
            best, best_h = index, h
    assert np.isfinite(best_h), 'Native fallback must be eligible.'
    return best, stats


def train(c, provenance):
    from multimodal.visible_train import attribute_metrics
    d = torch.load(ROOT/'cache/attribute_path_v1/dense.pt', weights_only=True)
    loc = torch.load(ROOT/'runs/attribute_path_v1/localizers.pt', weights_only=True)
    assert d['source'] == loc['source'] == provenance['previous']
    manifest = json.loads((ROOT/'data/cub100_v1/manifest.json').read_text())
    expected = [r for r in manifest['rows'] if r['role'] in ['train_seen','dev_seen','dev_unseen']]
    assert [(r['image_id'],r['label']) for r in d['rows']] == [(r['image_id'],r['label']) for r in expected]
    ti = torch.tensor([i for i,r in enumerate(d['rows']) if r['role']=='train_seen'])
    vi = torch.tensor([i for i,r in enumerate(d['rows']) if r['role']!='train_seen'])
    assert len(ti)==len(vi)==500 and set(d['labels'][ti].tolist())==set(range(50))
    mapping = attribute_view_indices(manifest)
    local, global_ = mapping > 0, mapping == 0
    assert int(local.sum())==144 and int(global_.sum())==86
    uniform = d['masks'].float()[:,None].expand(-1,4,-1)
    uniform = uniform/uniform.sum(-1,keepdim=True)
    common = dict(native_local=pooled_views(d,loc['maps']['native_part']),
                  uniform_local=pooled_views(d,uniform),
                  whole_local=d['global_features'][:,None].expand(-1,5,-1),
                  whole_global=d['global_features'][:,None].expand(-1,5,-1))
    RUN.mkdir(parents=True,exist_ok=True)
    records=[]
    for seed in c['seeds']:
        repaired = pooled_views(d,loc['learned'][str(seed)])
        features = common | dict(repaired_local=repaired,wrong_local=repaired[:,[0,2,3,4,1]])
        for variant in c['variants']:
            path = RUN/f'{variant}_{seed}.pt'
            if path.exists():
                ck=torch.load(path,weights_only=True);assert ck['source']==provenance
                records.append(ck['record']);continue
            keep = global_ if variant=='whole_global' else local
            view_ids = mapping[keep].clone()
            if variant.startswith('whole_'):view_ids.zero_()
            data = subset_data(d,keep)
            torch.manual_seed(seed)
            generator=torch.Generator().manual_seed(seed)
            model=AttributeReadout(data,view_ids,c['readout_scale']).cuda()
            x=features[variant].cuda()
            # Explicitly remove unused whole-image slot from local arms.
            if not variant.startswith('whole_'): x[:,0]=0
            y=d['attribute_targets'][:,keep].cuda()
            positive=(y[ti]==1).sum(0);negative=(y[ti]==0).sum(0)
            pw=((negative+.5)/(positive+.5)).clamp(1,20)
            opt=torch.optim.AdamW(model.parameters(),lr=c['readout_lr'],weight_decay=1e-4)
            for step in range(c['readout_steps']):
                ix=ti[torch.randint(len(ti),(c['readout_batch'],),generator=generator)]
                logits,q=model(x[ix]);target=y[ix];known=target>=0
                bce=F.binary_cross_entropy_with_logits(q,target.clamp(0,1),pos_weight=pw,reduction='none')
                bce=(bce*known).sum()/known.sum().clamp_min(1)
                ce=F.cross_entropy(logits[:,:50],d['labels'][ix].cuda())
                loss=ce+bce
                opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():
                logits,q=model(x[vi]);logits=logits.cpu();q=q.cpu()
                assert torch.isfinite(logits).all()
                am=attribute_metrics(y[vi].cpu(),q.sigmoid())
                record=dict(seed=seed,variant=variant,attributes=int(keep.sum()),parameters=sum(p.numel() for p in model.parameters()),
                            metrics=metrics(logits[:,:75].argmax(-1),d['labels'][vi]),attribute_map=am['attribute_map'],
                            train_ce=float(ce.detach()),train_bce=float(bce.detach()))
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(dict(source=provenance,state=state,record=record,evidence=q,scores=logits,
                            attribute_indices=keep.nonzero().flatten(),row_ids=[d['rows'][i]['image_id'] for i in vi],labels=d['labels'][vi]),path)
            records.append(record)
            save(OUT/'training_progress.json',dict(completed=len(records),total=18,records=records))
            print('trained',variant,seed,'H',round(record['metrics']['H'],3),flush=True)
            del model,opt,x,y
    save(OUT/'training.json',dict(source=provenance,records=records,training_images=500,development_images=500))


def evaluate(c,provenance):
    sig=torch.load(ROOT/'cache/cub_siglip_v1/features_development.pt',weights_only=True)
    text=torch.load(ROOT/'cache/cub_siglip_v2/text.pt',weights_only=True)['class_text']
    ix=[i for i,r in enumerate(sig['rows']) if r['role']!='train_seen']
    row_ids=[sig['rows'][i]['image_id'] for i in ix]
    labels=sig['labels'][ix]
    native=(20*sig['native_views'][ix,0]@text.T)[:,:75]
    base=native.argmax(-1)
    assert abs(metrics(base,labels)['H']-75.46455026455027)<1e-6
    folds=stratified_folds(labels,c['folds'],c['fold_seed'])
    arms={}; permutation_scores={p:[] for p in c['semantic_permutations']}
    reconstruction=[]
    for variant in c['variants']:
        scores=[]
        for seed in c['seeds']:
            ck=torch.load(RUN/f'{variant}_{seed}.pt',weights_only=True)
            assert ck['source']==provenance and ck['row_ids']==row_ids and torch.equal(ck['labels'],labels)
            recovered=subset_scores(ck['evidence'],ck['state']['profiles'])
            torch.testing.assert_close(recovered,ck['scores'],atol=2e-5,rtol=1e-5)
            assert metrics(ck['scores'][:,:75].argmax(-1),labels)==ck['record']['metrics']
            reconstruction.append(dict(variant=variant,seed=seed,score_max_error=float((recovered-ck['scores']).abs().max())))
            scores.append(ck['scores'][:,:75])
            if variant=='repaired_local':
                for p in c['semantic_permutations']:
                    perm=torch.randperm(144,generator=torch.Generator().manual_seed(p))
                    permutation_scores[p].append(subset_scores(ck['evidence'][:,perm],ck['state']['profiles'])[:,:75])
        arms[variant]=torch.stack(scores)
    arms.update({f'semantic_shuffle_{p}':torch.stack(v) for p,v in permutation_scores.items()})
    arms['native_calibration']=torch.zeros(3,500,75)
    summaries={};detail={};prediction_artifact={};selections={}
    for variant,attributes in arms.items():
        for policy in ['plain','guarded']:
            guarded=policy=='guarded'
            weights=[0.] if variant=='native_calibration' else c['weights']
            penalties=c['seen_penalties'] if guarded else [0.]
            candidates=[dict(weight=w,penalty=g) for w in weights for g in penalties]
            predictions=torch.stack([predict_fused(native[None],attributes,t['weight'],t['penalty'],guarded,c['native_margin_threshold']) for t in candidates])
            assert torch.equal(predictions[0],base[None].expand(3,-1))
            oof=torch.full((3,500),-1,dtype=torch.long)
            selected=[]
            for fold in range(c['folds']):
                fit,held=folds!=fold,folds==fold
                index,stats=select_candidate(predictions[:,:,fit],labels[fit],base[fit],candidates,guarded,c['max_unseen_new_error_fraction'])
                oof[:,held]=predictions[index,:,held]
                selected.append(dict(fold=fold,chosen=candidates[index],fit_candidates=[t|s for t,s in zip(candidates,stats)]))
            runs=[dict(seed=seed,metrics=metrics(oof[j],labels),changes=changes(oof[j],base,labels)) for j,seed in enumerate(c['seeds'])]
            key=f'{variant}/{policy}'
            summaries[key]=summarize(runs);detail[key]=runs;prediction_artifact[key]=oof;selections[key]=selected
            print('fusion',key,'H',round(summaries[key]['H']['mean'],3),'U',round(summaries[key]['U']['mean'],3),flush=True)
    save(OUT/'results.json',dict(source=provenance,baseline=metrics(base,labels),summary=summaries,crossfit=detail,
                               selections=selections,reconstruction=reconstruction,scope=c['scope']))
    torch.save(dict(source=provenance,native=native,labels=labels,row_ids=row_ids,folds=folds,
                    attributes=arms,predictions=prediction_artifact),RUN/'evaluation.pt')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['lock','train','evaluate','all'],default='all')
    args=parser.parse_args();torch.set_num_threads(4)
    c=json.loads(CONFIG.read_text());s=lock(c)
    if args.stage in ['train','all']:train(c,s)
    if args.stage in ['evaluate','all']:evaluate(c,s)


if __name__=='__main__':main()
