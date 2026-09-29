"""Aggregate the fixed-budget runs and perform labelled exploratory diagnostics."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from multimodal.attribute_path_v1 import setup, source, save_json, attribute_view_indices
from multimodal.visible_train import attribute_metrics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports/attribute_path_v1'


def main():
    c,m,p=setup()
    d=torch.load(ROOT/'cache/attribute_path_v1/dense.pt',weights_only=True)
    loc=torch.load(ROOT/'runs/attribute_path_v1/localizers.pt',weights_only=True)
    report=json.loads((OUT/'classification.json').read_text())
    assert d['source']==loc['source']==report['source']==source()
    vi=torch.tensor([i for i,r in enumerate(d['rows']) if r['role']!='train_seen'])
    unseen=torch.tensor([d['rows'][i]['role']=='dev_unseen' for i in vi])
    relevant=attribute_view_indices(m)>0
    patch=F.normalize(d['patches'][vi].float(),dim=-1)
    maps={};diagnostics={}
    for name,state in [('native',dict(query=d['part_text'])),*[(f'learned_{s}',v) for s,v in loc['states'].items()]]:
        score=patch@F.normalize(state['query'],dim=-1).T
        corr=[]
        for i in range(len(vi)):
            values=score[i,d['masks'][vi[i]]][:,[1,3]]
            values-=values.mean(0)
            corr.append(float(F.cosine_similarity(values[:,0],values[:,1],dim=0)))
        peaks=score.masked_fill(~d['masks'][vi,:,None],-1e4).argmax(1)
        diagnostics[name]=dict(mean_raw_wing_tail_correlation=float(np.mean(corr)),
                              same_peak_fraction=float((peaks[:,1]==peaks[:,3]).float().mean()))
    summary={};perclass={}
    for variant in c['variants']:
        rows=[r for r in report['results'] if r['variant']==variant]
        stats={k:dict(mean=float(np.mean([r['classification'][k] for r in rows])),
                     std=float(np.std([r['classification'][k] for r in rows],ddof=1))) for k in ['S','U','H','ZSL']}
        extras=[]
        for r in rows:
            ck=torch.load(ROOT/f"runs/attribute_path_v1/{variant}_{r['seed']}.pt",weights_only=True)
            q=ck['evidence'];target=d['attribute_targets'][vi]
            all_parts=attribute_metrics(target[:,relevant],q[:,relevant].sigmoid())['attribute_map']
            unseen_parts=attribute_metrics(target[unseen][:,relevant],q[unseen][:,relevant].sigmoid())['attribute_map']
            extras.append(dict(seed=r['seed'],part_attribute_map=all_parts,unseen_part_attribute_map=unseen_parts))
        stats['attribute_map']=dict(mean=float(np.mean([r['attribute_map'] for r in rows])),
                                   std=float(np.std([r['attribute_map'] for r in rows],ddof=1)))
        stats['part_attributes']=extras
        summary[variant]=stats
        perclass[variant]=np.array([[r['classification']['per_class'][str(i)] for i in range(75)] for r in rows])
    # Paired resampling of class accuracies; conditional on these trained models.
    # Exploratory, not a confirmatory significance test on independent data.
    rng=np.random.default_rng(20260929)
    si=rng.integers(0,50,(2000,50));ui=rng.integers(50,75,(2000,25))
    def sampled_h(a):
        seen=a[:,si].mean(-1);unseen=a[:,ui].mean(-1)
        return (2*seen*unseen/(seen+unseen).clip(1e-9)).mean(0)
    learned=sampled_h(perclass['learned_part']); intervals={}
    for variant in c['variants']:
        if variant=='learned_part':continue
        delta=learned-sampled_h(perclass[variant])
        intervals[variant]=dict(mean_observed_delta=summary['learned_part']['H']['mean']-summary[variant]['H']['mean'],
                               paired_class_bootstrap_95_percentile=np.quantile(delta,[.025,.975]).tolist(),
                               seed_deltas=(np.array([r['classification']['H'] for r in report['results'] if r['variant']=='learned_part'])-
                                            np.array([r['classification']['H'] for r in report['results'] if r['variant']==variant])).tolist())
    output=dict(source=source(),summary=summary,localization_confusion=diagnostics,paired_exploratory_intervals=intervals,
                part_attribute_count=int(relevant.sum()),native_siglip2=report['siglip2_native'],
                interval_scope='Paired class bootstrap within seen/unseen strata, averaging the three fixed trained seeds; does not include new training/class-split uncertainty. Reused development data, no confirmatory significance claim.')
    save_json(OUT/'analysis.json',output)
    assets=dict(classes=m['classes'],attributes=m['attributes'],seen_classes=p['seen_classes'],dev_unseen_classes=p['dev_unseen_classes'],
                manifest_sha256=source()['data\\cub100_v1\\manifest.json'],source=source())
    save_json(ROOT/'configs/attribute_path_v1_assets.json',assets)
    print(json.dumps(dict(summary={k:{x:round(v[x]['mean'],3) for x in ['H','ZSL','attribute_map']} for k,v in summary.items()},
                    localization=diagnostics,intervals=intervals),indent=2),flush=True)


if __name__=='__main__':main()
