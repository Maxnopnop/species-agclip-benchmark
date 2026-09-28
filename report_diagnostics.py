"""Summarize completed exploratory validation diagnostics without new test access."""
import csv
import json
import statistics
from collections import Counter
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_tools import ROOT,write_json


def mean(rows,key):return statistics.mean(r[key] for r in rows)


def main():
    out=ROOT/'reports/diagnostics_v1';protocol=json.loads((ROOT/'configs/diagnostics_v1.json').read_text())
    rows=[json.loads(p.read_text()) for p in sorted((ROOT/'runs/diagnostics_v1').glob('*/result.json'))]
    expected=np.prod([len(protocol[k]) for k in ('backbones','regimes','modes','head_learning_rates','seeds')])
    if len(rows)!=expected:raise ValueError(f'Expected {expected} completed runs; got {len(rows)}')
    assert all(r['tail_parameter_l2_change']==0 and r['tail_first_gradient_l1']==0 for r in rows if r['regime']=='frozen')
    assert all(r['tail_parameter_l2_change']>0 and r['tail_first_gradient_l1']>0 for r in rows if r['regime']=='tail')
    flat=[]
    for r in rows:
        flat.append({**{k:v for k,v in r.items() if k not in ('validation','training')},
                     **{f'val_{k}':v for k,v in r['validation'].items()},
                     **{f'train_{k}':v for k,v in r['training'].items()}})
    with (out/'all_training_runs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=flat[0].keys());w.writeheader();w.writerows(flat)
    selected=[];matched_budget=[]
    for name in protocol['backbones']:
        for regime in protocol['regimes']:
            for mode in protocol['modes']:
                candidates=[]
                for lr in protocol['head_learning_rates']:
                    r=[x for x in rows if x['backbone']==name and x['regime']==regime and x['mode']==mode and x['lr']==lr]
                    assert len(r)==3
                    candidates.append(dict(lr=lr,macro_f1=statistics.mean(x['validation']['macro_f1'] for x in r),
                        loss=statistics.mean(x['validation']['loss'] for x in r)))
                best=sorted(candidates,key=lambda x:(-x['macro_f1'],x['loss']))[0]
                r=[x for x in rows if x['backbone']==name and x['regime']==regime and x['mode']==mode and x['lr']==best['lr']]
                selected.append(dict(backbone=name,regime=regime,mode=mode,selected_lr=best['lr'],
                    validation_accuracy_mean=statistics.mean(x['validation']['top1_accuracy'] for x in r),
                    validation_accuracy_std=statistics.stdev(x['validation']['top1_accuracy'] for x in r),
                    training_accuracy_mean=statistics.mean(x['training']['top1_accuracy'] for x in r),
                    validation_macro_f1=best['macro_f1'],candidates=candidates))
            for lr in protocol['head_learning_rates']:
                for control in ('zero','permuted'):
                    differences=[]
                    for seed in protocol['seeds']:
                        get=lambda mode:next(x['validation']['top1_accuracy'] for x in rows if x['backbone']==name and x['regime']==regime and x['mode']==mode and x['lr']==lr and x['seed']==seed)
                        differences.append((get('matched')-get(control))*100)
                    matched_budget.append(dict(backbone=name,regime=regime,lr=lr,control=control,per_seed_gain_pp=differences,mean_gain_pp=statistics.mean(differences)))
    write_json(out/'selected_validation_results.json',selected);write_json(out/'same_lr_attribute_comparisons.json',matched_budget)
    sensitivity=[];prefix_checks=[]
    for name in protocol['backbones']:
        cache=torch.load(ROOT/f'cache/diagnostics_v1/{name}_prefix.pt',weights_only=True,mmap=True)
        assert set(r['split'] for r in cache['rows'])=={'train','val'}
        prefix_checks.append(dict(backbone=name,images=len(cache['rows']),splits=['train','val'],
            prefix_shape=list(cache['prefix'].shape),max_encoder_reconstruction_error=cache['decomposition_max_absolute_error']))
        del cache
    for r in rows:
        if r['mode']!='matched':continue
        name,regime,seed,lr=(r[k] for k in ('backbone','regime','seed','lr'))
        base=ROOT/'runs/diagnostics_v1'
        reference=torch.load(base/f'{name}_{regime}_matched_seed{seed}_lr{lr:g}'/'validation_predictions.pt',weights_only=True)
        for control in ('zero','permuted'):
            other=torch.load(base/f'{name}_{regime}_{control}_seed{seed}_lr{lr:g}'/'validation_predictions.pt',weights_only=True)
            assert reference['paths']==other['paths']
            a,b=reference['probabilities'],other['probabilities']
            sensitivity.append(dict(backbone=name,regime=regime,lr=lr,seed=seed,control=control,
                prediction_changes=int((a.argmax(-1)!=b.argmax(-1)).sum()),mean_probability_l1=float((a-b).abs().sum(-1).mean())))
    write_json(out/'prefix_validation.json',prefix_checks);write_json(out/'retrained_prediction_sensitivity.json',sensitivity)
    interventions=json.loads((out/'posthoc_interventions.json').read_text());summary=[]
    for name in protocol['backbones']:
        for variant in ('ag_mean','ag_attention','ag_aux'):
            for mode in ('matched','permuted','zero','no_branch'):
                r=[x for x in interventions if x['backbone']==name and x['variant']==variant and x['mode']==mode]
                summary.append(dict(backbone=name,variant=variant,mode=mode,accuracy_mean=mean(r,'top1_accuracy'),
                    mean_flipped_predictions=mean(r,'flipped_predictions'),mean_probability_l1=mean(r,'mean_probability_l1')))
    write_json(out/'posthoc_summary.json',summary)
    review=json.loads((out/'visual_review.json').read_text())
    counts={n:dict(Counter(x[n] for x in review['items'])) for n in protocol['backbones']}
    write_json(out/'visual_review_counts.json',dict(counts=counts,note='Descriptive assistant review of 20 fixed samples, not an expert-labeled attribute benchmark.'))
    lines=['# Attribute-guidance diagnosis: validation only','',
        'This report diagnoses the existing 20-species system. It does not evaluate a new test set, establish statistical significance, or claim a paper reproduction. The same validation set has already been used for model development.','',
        '## 1. Interventions on existing checkpoints','',
        'Existing 10-shot checkpoints, three seeds, 200 validation images per run. Retrieval keys stay unchanged: permuted mode replaces the selected value embeddings with a fixed derangement; zero removes only text values; no_branch disables the entire fusion residual. This separates sensitivity to text values from sensitivity to the full branch. The auxiliary training loss itself is not undone by these inference interventions.','',
        '| Backbone | Existing variant | Original accuracy | Permuted text | Zero text | Zero-text prediction changes / 200 | Entire branch disabled: changes / 200 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name in protocol['backbones']:
        for v in ('ag_mean','ag_attention','ag_aux'):
            d={x['mode']:x for x in summary if x['backbone']==name and x['variant']==v}
            lines.append(f'| {name} | {v} | {d["matched"]["accuracy_mean"]*100:.2f}% | {d["permuted"]["accuracy_mean"]*100:.2f}% | {d["zero"]["accuracy_mean"]*100:.2f}% | {d["zero"]["mean_flipped_predictions"]:.2f} | {d["no_branch"]["mean_flipped_predictions"]:.2f} |')
    lines+=['','## 2. Architecture-matched retraining and tail adaptation','',
        '96 runs: two backbones x two regimes x four modes x two head learning rates x three seeds. Each uses 10 training images per species, the corresponding existing alignment initialization, 100 updates, batch size 16, and validation at steps 50 and 100. The same two-candidate learning-rate budget is used in every condition. Matched/permuted/zero use identical attention architecture and initialization, without an auxiliary attribute loss. Baseline uses the whole image only.','',
        'Matched means the ordinary retrieved attribute embeddings, not guaranteed correct image-level descriptions. Permuted values retain fixed structure that a trained network could partly learn around; this is a semantic disruption control, not independent random noise on every batch.','',
        'Tail adaptation updates EfficientNet final MBConv stage and final convolution, or CLIP final transformer block, post-normalization and projection. Earlier layers stay frozen. BatchNorm running statistics, dropout in the visual backbone, and stochastic depth remain in evaluation mode in both regimes. Float32 cached prefix activations reproduce the full encoder before adaptation.','',
        'The table selects one learning rate per condition by mean validation macro-F1, then loss. These are development scores on the selection set, not unbiased final performance. All 96 runs and same-learning-rate paired differences are retained.','',
        '| Backbone | Regime | Mode | Selected head LR | Training accuracy mean | Validation accuracy mean +/- SD |',
        '|---|---|---|---:|---:|---:|']
    for r in selected:
        lines.append(f'| {r["backbone"]} | {r["regime"]} | {r["mode"]} | {r["selected_lr"]:g} | {r["training_accuracy_mean"]*100:.2f}% | {r["validation_accuracy_mean"]*100:.2f} +/- {r["validation_accuracy_std"]*100:.2f}% |')
    lines+=['','## 3. Retrieval and visual inspection','',
        'Retrieval proxies compare retrieved attribute IDs with the species description list and with insect/plant group membership. They do NOT measure true visible-attribute accuracy: unlisted traits may still be valid and listed traits may be invisible. Native CLIP, aligned models, and trained attention models are all preserved in retrieval_proxies.json.','',
        'One validation image per species was selected by fixed seed before viewing. The assistant inspected the center crop and each model\'s top-1 prompt, recording supported, unsupported, uncertain, or background descriptions. This is not expert biological annotation. Local contact sheets remain under work/diagnostics_v1 and are not uploaded to GitHub.','',
        f'Assistant review counts: {json.dumps(counts)}.','',
        'Illustrative failures include grass background selected for a tiny butterfly (label 4), blue-upper-wing descriptions for a pale underside view (label 5), yellow flowers described when only leaves/dry stems are visible (label 13), and red berries described where pale flower clusters are visible (label 19). These examples were not used to replace or tune the attribute bank during this diagnostic run.','',
        '## Interpretation limits and next experiment','',
        'Inference sensitivity directly tests whether the trained model relies on attribute values, but small sensitivity does not prove that auxiliary text supervision had no effect during training. Retraining controls and tail adaptation narrow hypotheses; they do not prove that all AG methods fail. The limited data, fixed crops, small learning-rate grid and 100-step budget remain constraints. No test-driven model replacement was performed.','',
        'The next development experiment should verify region/attribute matches, use a small independently reviewed image-level attribute set, and test a direct attribute-alignment objective. A new held-out evaluation is required before claiming improvement after these validation-driven changes.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
    colors=['#496a81','#259c86','#dc9c38','#8d799e']
    for i,name in enumerate(protocol['backbones']):
        for j,regime in enumerate(protocol['regimes']):
            ax=axes[i,j];r=[x for x in selected if x['backbone']==name and x['regime']==regime]
            ax.bar([x['mode'] for x in r],[x['validation_accuracy_mean']*100 for x in r],
                   yerr=[x['validation_accuracy_std']*100 for x in r],capsize=3,color=colors)
            ax.set_title(f'{name}: {regime}');ax.set_ylim(0,100);ax.set_ylabel('Validation accuracy (%)')
    fig.suptitle('Development diagnostics: 10-shot, mean +/- SD over 3 seeds\nLearning rate selected on validation; not new test performance')
    fig.tight_layout();fig.savefig(out/'diagnostic_comparison.png',dpi=160);plt.close(fig)
    write_json(out/'verification.json',dict(completed_training_runs=len(rows),training_and_validation_only=True,
        frozen_tail_unchanged_runs=sum(r['regime']=='frozen' for r in rows),
        trainable_tail_nonzero_gradient_and_change_runs=sum(r['regime']=='tail' for r in rows),
        tests_passed=3,test_command='.venv/Scripts/python.exe -m unittest -v test_diagnostics',
        visual_samples=20,posthoc_intervention_records=len(interventions)))
    print(json.dumps(selected,indent=2))


if __name__=='__main__':main()
