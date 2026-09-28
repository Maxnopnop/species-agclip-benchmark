# Visible attributes and region alignment

This follow-up asks two separate questions: can a model recognize a small set of visible attributes, and does using those attributes improve species classification? It does not assume that either result follows from the other.

## Data and supervision

The experiment uses 20 existing iNaturalist species: 200 fixed training images (10/species) and 200 validation images. No test image is loaded. The same images are used across optimization seeds. Forty training and twenty validation images were selected deterministically for visual review before reading model predictions. Sixteen visible traits were recorded along with approximate evidence boxes. Labels are provisional assistant annotations, not expert biological ground truth. One reviewed image in each split is entirely unknown; unknown and non-applicable entries never become negatives. Applicable negative labels mean absent in the assessed target region, not absent from the species generally.

Many evidence boxes cover an entire organism or cluster. This is weak region supervision, not precise segmentation. Sparse positive examples and only nineteen validation images with known labels limit attribute conclusions. Review sheets and photographic heatmaps stay local under `work/visible_v1`; source images are not uploaded.

## Models and comparisons

- Official OpenAI CLIP ViT-B/16 and Google SigLIP 2 Base at pinned revisions: three optimization seeds, two learning rates, five variants = 60 cells.
- Conditional follow-up: official FG-CLIP Base, one seed, identical five variants and two learning rates = 10 cells. FG-CLIP already has fine-grained region-text pretraining. Its local custom code is pinned and hash-checked; no arbitrary current remote code is loaded. Its official dense-feature formula is checked numerically against the official API.

All image/text encoders are frozen. Separate global and patch residual adapters, attribute calibration, and a linear attribute-to-class branch are trained. Attributes supervise patch/text similarities via masked BCE; positive evidence boxes supervise patch attention through KL loss. Species cross-entropy is always present. The image is letterboxed to 224 pixels; padding tokens are masked for attention.

Variants: global baseline; same added branch without auxiliary supervision (capacity control); attribute BCE; attribute BCE plus region alignment; training annotations and boxes shuffled across species within insect/plant groups. Hyperparameters and training batches are matched. Shuffling can still encourage generic foreground attention; it is not a perfect negative for localization.

Both baseline and full model must be reported. Comparing only against an underperforming capacity control can overstate AG gains. This is an AG-inspired custom implementation, not a strict reproduction of the original AG-CLIP architecture. Cached CLIP patch projections and native SigLIP patch features are exploratory representations; FG-CLIP uses its dedicated dense-feature path.

## Selection and evaluation

Each cell runs 400 updates and evaluates at steps 100/200/300/400. Checkpoint selection: validation macro-F1 then lower classification CE. Learning-rate selection: the same metrics averaged across the available optimization seeds. No hyperparameter selection uses held-out test data. This repeatedly used validation set remains exploratory, not an independent final test.

Report species accuracy/macro-F1; attribute mAP and threshold-0.5 F1 only where validation has both known positives and negatives; attention peak-in-box/mass versus uniform box area; zeroed/permuted branch interventions; all learning rates and optimization-seed dispersion. The baseline's attribute numbers use untrained attribute projections/calibration and are diagnostics, not a trained attribute classifier. Attribute AP and attention localization must not be presented as species accuracy.

## Run on this machine

```powershell
cd E:\ELEC4240\SpeciesRecognition
.\.venv\Scripts\python.exe download_visible_models.py
.\.venv\Scripts\python.exe prepare_visible_features.py
.\.venv\Scripts\python.exe visible_experiment.py
.\.venv\Scripts\python.exe download_fgclip.py
# If a full-weight HTTP transfer stalls, resumable ranged download verifies the official SHA256:
.\.venv\Scripts\python.exe download_fgclip.py --ranged
.\.venv\Scripts\python.exe fgclip_visible_experiment.py
.\.venv\Scripts\python.exe report_visible.py
.\.venv\Scripts\python.exe verify_visible.py
.\.venv\Scripts\python.exe -m unittest test_visible -q
```

The existing expanded20 manifest/photos are prerequisites. Annotations and selection are versioned in `configs/visible_*.json`; training results/checkpoints live under `runs/visible_v1`; large features and weights are in `cache/visible_v1`. Resume refuses a changed protocol or implementation fingerprint. New scientific changes require a new experiment version instead of overwriting earlier outcomes.

Image-only prediction:

```powershell
.\.venv\Scripts\python.exe predict_visible.py --checkpoint runs/visible_v1/siglip2_b16_attribute_region_seed42_lr0.001/best.pt --image E:\path\photo.jpg --output work/prediction.json
```

No true species, attribute label, or evidence box is accepted by inference. Fixed class/attribute text embeddings remain internal to the saved head, so this is image-only user input, not removal of all text-derived representations at deployment. Classification is restricted to the twenty trained species and scores are not reliability-calibrated.

Results: [report](../reports/visible_v1/README.md), [all cells](../reports/visible_v1/all_runs.csv), [inference verification](../reports/visible_v1/inference_verification.json).
