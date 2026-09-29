"""Plot aggregate scores only; no source photographs are redistributed."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reports/four_routes_v1'
d = json.loads((OUT / 'fgclip2_results.json').read_text())
names = ['head', 'wing', 'breast', 'tail']
matrix = np.array([[d['extra_diagnostic']['target_part_by_query_hit_rate'][part][q]
                    for q in names] for part in names]) * 100
fig, ax = plt.subplots(figsize=(7, 5.8))
im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=100)
ax.set_xticks(range(4), names)
ax.set_yticks(range(4), [f"{n} (n={d['localization'][n]['n']})" for n in names])
ax.set_xlabel('Text query')
ax.set_ylabel('Ground-truth target part (evaluation only)')
for i in range(4):
    for j in range(4):
        ax.text(j, i, f'{matrix[i,j]:.1f}%', ha='center', va='center',
                color='white' if matrix[i,j] > 55 else 'black', fontsize=12)
ax.set_title('FG-CLIP 2: cross-query pointing diagnostic\n75 development images; fixed 256-patch setting')
fig.colorbar(im, ax=ax, label='Hit rate within 0.1 image diagonal (%)')
fig.text(.5, .015, 'Exploratory check; part sizes and numbers of reference points differ.', ha='center', fontsize=9)
fig.tight_layout(rect=(0,.04,1,1))
fig.savefig(OUT / 'fgclip2_query_pointing.png', dpi=160)
plt.close(fig)
