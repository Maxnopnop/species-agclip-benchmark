"""Export comparison tables and figures without copying images or weights."""
import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_tools import ROOT,write_json
from multimodal.experiment import aggregate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',default=str(ROOT/'runs'/'pilot_multimodal'))
    p.add_argument('--output',default=str(ROOT/'reports'/'pilot'))
    p.add_argument('--regions',default=str(ROOT/'cache'/'pilot_multimodal'/'regions.json'))
    args=p.parse_args()
    rows=aggregate(args.runs)
    if not rows:
        raise RuntimeError('No completed model comparisons available.')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Path(args.runs)/'comparison.csv',out/'comparison.csv')
    groups=defaultdict(list)
    for row in rows:
        groups[(row['backbone'],row['shots'],row['variant'])].append(row)
    summary=[]
    for (name,shots,variant),values in sorted(groups.items()):
        record=dict(backbone=name,shots=shots,variant=variant,seeds=len(values),pilot=all(v['pilot'] for v in values))
        for metric in ('top1_accuracy','macro_f1'):
            data=[v[metric] for v in values]
            record[metric+'_mean']=float(np.mean(data))
            record[metric+'_std']=float(np.std(data,ddof=1)) if len(data)>1 else None
        summary.append(record)
    write_json(out/'summary.json',summary)
    note=['# Benchmark results','',
          'PILOT ONLY: 4 species, 60 real images; 5 train / 5 validation / 5 held-out images per species. '
          'This checks the workflow, not the planned 100-species experiment. One seed cannot establish a reliable improvement.'
          if all(r['pilot'] for r in rows) else
          'Formal benchmark: selection uses validation macro-F1. Compare methods within each backbone and shot setting.',
          '', 'Frozen visual and text encoders; supervised projection/adapters and attribute fusion. '
          'AG-CLIP-inspired adaptation, not an exact reproduction of the paper. No claim of unseen-species zero-shot performance.','',
          '| Backbone | Shots | Variant | Seeds | Top-1 mean | Macro-F1 mean |',
          '|---|---:|---|---:|---:|---:|']
    for r in summary:
        note.append(f"| {r['backbone']} | {r['shots']} | {r['variant']} | {r['seeds']} | {r['top1_accuracy_mean']:.3f} | {r['macro_f1_mean']:.3f} |")
    regionpath=Path(args.regions)
    if regionpath.exists():
        regions=json.loads(regionpath.read_text(encoding='utf-8'))['images']
        counts=[len(v['boxes']) for v in regions.values()]
        note.extend(['',f'Region coverage: {sum(c>0 for c in counts)}/{len(counts)} images have at least one detection; '
                     f'mean {np.mean(counts):.2f} regions per image. Missing regions use the global baseline feature.'])
    note.extend(['','Training-time fields exclude frozen feature extraction, OWL-ViT grounding and downloads. '
                 'They are not end-to-end deployment latency.','',
                 'Top-5 accuracy is uninformative for a four-class pilot. Test metrics should not guide further tuning.'])
    (out/'README.md').write_text('\n'.join(note)+'\n',encoding='utf-8')
    for shots in sorted({r['shots'] for r in rows}):
        subset=[r for r in summary if r['shots']==shots]
        names=sorted({r['backbone'] for r in subset})
        fig,ax=plt.subplots(figsize=(10,4.5))
        for v,variant in enumerate(('baseline','average','agclip')):
            values=[next(r['top1_accuracy_mean'] for r in subset if r['backbone']==n and r['variant']==variant) for n in names]
            ax.bar(np.arange(len(names))+(v-1)*.24,values,width=.24,label=variant)
        ax.set(xticks=np.arange(len(names)),xticklabels=names,ylim=(0,1.05),ylabel='Held-out Top-1 accuracy',
               title=f'{"Pilot: " if all(r["pilot"] for r in rows) else ""}{shots} training images per species')
        ax.legend();fig.tight_layout();fig.savefig(out/f'comparison_{shots}shots.png',dpi=160);plt.close(fig)
    print('Report saved:',out)


if __name__=='__main__':
    main()
