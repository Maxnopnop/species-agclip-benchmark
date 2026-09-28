"""Report every variant and paired, multiplicity-adjusted selected-AG comparisons."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_tools import ROOT,write_json


def paired_statistics(reference,ag,truth,seed=20260928):
    reference=np.asarray(reference);ag=np.asarray(ag);truth=np.asarray(truth)
    good_ref=reference==truth;good_ag=ag==truth
    wins=int((good_ag&~good_ref).sum());losses=int((good_ref&~good_ag).sum());n=wins+losses
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/(2**n)) if n else 1.
    differences=good_ag.astype(float)-good_ref.astype(float)
    rng=np.random.default_rng(seed);boot=np.zeros(5000)
    for label in np.unique(truth):
        values=differences[truth==label]
        boot+=rng.choice(values,size=(5000,len(values)),replace=True).sum(axis=1)/len(truth)
    low,high=np.quantile(boot,[.025,.975])
    return dict(reference_accuracy=float(good_ref.mean()),ag_accuracy=float(good_ag.mean()),
        improvement_percentage_points=float(differences.mean()*100),ag_fixes=wins,ag_breaks=losses,
        mcnemar_exact_two_sided_p=p,paired_stratified_bootstrap_95ci_pp=[float(low*100),float(high*100)])


def holm(records):
    order=sorted(range(len(records)),key=lambda i:records[i]['mcnemar_exact_two_sided_p']);previous=0
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,(len(order)-rank)*records[index]['mcnemar_exact_two_sided_p']))
        records[index]['holm_adjusted_p']=previous
        records[index]['significant_positive_gain']=previous<.05 and records[index]['improvement_percentage_points']>0


def main():
    root=ROOT/'runs/expanded20';out=ROOT/'reports/expanded20';out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'configs/expanded_protocol.json').read_text())
    rows=json.loads((root/'test_results.json').read_text());selected=json.loads((root/'selection_locked.json').read_text())
    expected=len(protocol['backbones'])*len(protocol['shots'])*len(protocol['seeds'])*len(protocol['variants'])
    if len(rows)!=expected:raise ValueError('Incomplete comparison matrix')
    with (out/'all_test_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    groups=defaultdict(list)
    for row in rows:groups[(row['backbone'],row['shots'],row['variant'])].append(row)
    summary=[]
    for (name,shots,variant),values in sorted(groups.items()):
        summary.append(dict(backbone=name,shots=shots,variant=variant,seeds=len(values),
            top1_mean=float(np.mean([v['top1_accuracy'] for v in values])),
            top1_std=float(np.std([v['top1_accuracy'] for v in values],ddof=1)),
            macro_f1_mean=float(np.mean([v['macro_f1'] for v in values]))))
    write_json(out/'summary.json',summary);write_json(out/'selection_locked.json',selected)
    comparisons=[];diagnostics=[]
    for item in selected:
        name,shots,ag=item['backbone'],item['shots'],item['selected_ag']
        ensembles={};identity=None
        for variant in ('baseline','region_only',ag):
            probabilities=[]
            for seed in protocol['seeds']:
                folder=root/f'{name}_shots{shots}_seed{seed}'/variant
                pred=torch.load(folder/'test_predictions.pt',weights_only=True)
                current=(pred['paths'],pred['labels'].tolist())
                if identity is not None and current!=identity:raise ValueError('Paired samples differ')
                identity=current;probabilities.append(pred['probabilities'])
                checkpoint=torch.load(folder/'best.pt',weights_only=True,map_location='cpu')
                diagnostics.append(dict(backbone=name,shots=shots,seed=seed,variant=variant,**checkpoint['diagnostic']))
            ensembles[variant]=torch.stack(probabilities).mean(0).argmax(1).numpy()
        for reference in ('baseline','region_only'):
            comparisons.append(dict(backbone=name,shots=shots,selected_ag=ag,reference=reference,images=len(identity[0]),
                **paired_statistics(ensembles[reference],ensembles[ag],identity[1])))
    holm(comparisons)
    write_json(out/'paired_comparisons.json',comparisons);write_json(out/'training_diagnostics.json',diagnostics)
    lines=['# Expanded 20-species development benchmark','',
        '20 species, 10 family pairs, 1,000 images: 30 candidate training / 10 validation / 10 test per species. '
        'Species were selected by taxonomy and archive availability before observing results; all four earlier pilot species are excluded. '
        'This is not the formal 100-species experiment.','',
        '5 backbones × 3 shot counts × 3 seeds × 5 variants = 225 comparisons. '
        'Frozen encoders; 200 alignment updates and 200 additional updates for every comparison variant. '
        'Validation macro-F1 selects checkpoints, with validation loss breaking ties. '
        'AG variants use fixed overlapping crops and a shared source-derived attribute bank, not OWL-ViT anatomical detections. '
        'These are AG-CLIP-inspired adaptations, not an exact replication.','',
        'AG choice was locked using the mean validation metrics over seeds before any test evaluation. '
        'The table reports per-run test accuracy mean ± sample standard deviation over three seeds. '
        'Paired tests use one prediction per image from a three-seed probability ensemble, not 600 independent observations.','',
        '| Backbone | Shots | Baseline mean ± SD | Selected AG | AG mean ± SD | Mean gain (pp) | Ensemble gain (pp) | Holm p vs baseline |',
        '|---|---:|---:|---|---:|---:|---:|---:|']
    def get(name,shots,variant):return next(r for r in summary if r['backbone']==name and r['shots']==shots and r['variant']==variant)
    for item in selected:
        name,shots,ag=item['backbone'],item['shots'],item['selected_ag'];base=get(name,shots,'baseline');a=get(name,shots,ag)
        c=next(r for r in comparisons if r['backbone']==name and r['shots']==shots and r['reference']=='baseline')
        lines.append(f'| {name} | {shots} | {base["top1_mean"]*100:.1f} ± {base["top1_std"]*100:.1f}% | {ag} | '
                     f'{a["top1_mean"]*100:.1f} ± {a["top1_std"]*100:.1f}% | {(a["top1_mean"]-base["top1_mean"])*100:+.2f} | '
                     f'{c["improvement_percentage_points"]:+.2f} | {c["holm_adjusted_p"]:.4f} |')
    significant=[r for r in comparisons if r['reference']=='baseline' and r['significant_positive_gain']]
    lines.extend(['',f'Positive ensemble gains surviving Holm correction versus baseline: {len(significant)}/{len(selected)} settings.',
        '', 'Correction covers all 30 planned comparisons: 15 settings against baseline and the same 15 against the region-only control. '
        'Bootstrap intervals are unadjusted 95% paired intervals stratified by class; McNemar tests are exact and two-sided. '
        'Statistical conclusions apply to this fixed development set. Shared pretraining, possible near-duplicate observations, '
        'class-level rather than image-level attribute annotations, and availability-based species selection limit generalization.',
        '', 'All five variants, including declines, are preserved in `all_test_results.csv`. '
        'Source URLs and paraphrased traits are in `configs/expanded_attributes.json`. '
        'No test-driven retraining or variant replacement was performed for this report.'])
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=True)
    for ax,shots in zip(axes,protocol['shots']):
        names=protocol['backbones'];x=np.arange(len(names))
        for offset,kind,color in [(-.18,'baseline','#386cb0'),(.18,'selected_ag','#2ca25f')]:
            values=[];errors=[]
            for name in names:
                v=kind if kind=='baseline' else next(r['selected_ag'] for r in selected if r['backbone']==name and r['shots']==shots)
                record=get(name,shots,v);values.append(record['top1_mean']*100);errors.append(record['top1_std']*100)
            ax.bar(x+offset,values,width=.36,yerr=errors,capsize=2,color=color,label=kind)
        ax.set(xticks=x,xticklabels=['EffNet','ResNet','ConvNeXt','ViT','CLIP'],ylim=(0,100),title=f'{shots} images per species')
    axes[0].set_ylabel('Test accuracy (%), mean ± SD across 3 seeds');axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.1),ncol=2)
    fig.suptitle('Expanded development set: AG selected on validation only');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    print('\n'.join(lines))


if __name__=='__main__':main()
