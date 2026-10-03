# FungiTastic temporal validation

500 images from 500 unique observations across 10 species. 200 training and 100 validation images dated2022;200 test images dated2023.20/10/20 per species,same splits across seeds. Classes selected by counts and fixed random order,never by model scores.

Accuracy is mean +/- sample SD across three seeds. Two frozen backbones,unchanged Head/train,200 updates per stage,seven matched arms. CLIP visual readout retains its language pretraining. Task text adaptation is not CLIP pretraining from scratch,and AG is a lightweight adaptation.

| Method | EfficientNet-B0 (%) | CLIP ViT-B/32 (%) |
|---|---:|---:|
| visual | 64.67 ± 2.93 | 62.50 ± 2.29 |
| random_codes | 73.17 ± 1.04 | 70.00 ± 1.00 |
| text | 72.17 ± 0.76 | 74.17 ± 1.26 |
| region_only | 71.17 ± 1.04 | 74.83 ± 0.76 |
| ag_uniform | 71.67 ± 1.44 | 74.50 ± 1.73 |
| ag_confidence | 71.67 ± 1.44 | 74.83 ± 0.58 |
| ag_shuffled | 71.50 ± 1.32 | 75.00 ± 0.50 |

Visual readout accuracy is only about63%–65%,yet confidence weighting shows no significant positive evidence. EfficientNet weighted and uniform means tie;CLIP weighted mean rises by only0.33pp,below shuffled weights and equal to the region-only control. Text-versus-linear gains cannot be attributed to AG or confidence ordering. Large gains from random prototypes also identify classifier/optimization differences as a confound.

177/200 test images have two regions,so lack of opportunities to weight multiple regions is insufficient as an explanation. Fixed-checkpoint uniform/reversed-weight interventions change no EfficientNet labels and at most3/200 CLIP labels. This suggests limited decision dependence on weight ordering in this implementation,not universal attribute ineffectiveness.

Seed means and probability ensembles are different estimands:CLIP weighting changes the seed mean by+0.33pp but the ensemble by−1.50pp. Selecting only the favorable aggregation would overstate evidence.

## Paired evidence for confidence weighting

| Backbone | Reference | Ensemble delta (pp) | Fixes / breaks | 95% CI (pp) | Holm p |
|---|---|---:|---:|---|---:|
| efficientnet_b0 | ag_uniform | +0.50 | 1/0 | [0.0, 1.5] | 1.0000 |
| efficientnet_b0 | text | +0.50 | 2/1 | [-1.0, 2.0] | 1.0000 |
| efficientnet_b0 | ag_shuffled | +0.00 | 0/0 | [0.0, 0.0] | 1.0000 |
| clip_vit_b32 | ag_uniform | -1.50 | 4/7 | [-4.500000000000001, 1.5] | 1.0000 |
| clip_vit_b32 | text | +1.00 | 6/4 | [-2.0, 4.0] | 1.0000 |
| clip_vit_b32 | ag_shuffled | -2.00 | 0/4 | [-4.0, -0.5] | 0.7500 |

Positive contrasts surviving Holm correction among six predeclared tests: 0.

Tests use200 observation-level predictions from three-seed probability ensembles,not600 independent images. Exact McNemar tests and5000 class-stratified bootstrap resamples;Holm across six contrasts. Intervals are unadjusted and conditional on fixed models. A [0,0] empirical interval does not prove a zero population effect.

## Detection and weight use

| Split | Zero boxes | One box | Two boxes | Mean larger weight |
|---|---:|---:|---:|---:|
| train | 2 | 11 | 187 | 0.679640531539917 |
| val | 3 | 7 | 90 | 0.6819465160369873 |
| test | 8 | 15 | 177 | 0.6737488508224487 |

Thirty fixed visible fungus prompts are shared across all images. No per-image captions,coordinates or test attribute labels enter the model. OWL-ViT uses at most two regions with threshold0.05. Detector confidence is not a calibrated attribute-correctness probability. Fixed-checkpoint uniform/shuffled interventions are diagnostic only,recorded in verification.json.

## Data protection and limits

Historical images screened: 8742. Minimum pHash/dHash distances: 12/10;threshold4. Zero exact/pixel/perceptual-threshold matches after filtering. Observation IDs are disjoint across splits. Photographer identities are unavailable for grouping.

Manifest,code and configuration locked before development feature extraction. All42 validation-selected checkpoints and development assets sealed before test grounding/features.42 predictions replayed for audit. No test-guided changes.

Metadata observation dates2022/2023 postdate the2021 OpenAI CLIP release and ImageNet1K data. This strengthens temporal separation for the old backbones,but observation dates are not independently verified capture dates. OWL-ViT pretraining overlap is unaudited:the entire pipeline cannot be certified image-unseen,and species concepts may already be known.

This is a custom closed-set temporal split,not unseen-class zero-shot recognition or the full official benchmark. Small sample size,short fixed training budget and unverified attribute localization limit generalization. Fresh data cannot guarantee no overfitting. This holdout is now observed for future development. Cross-dataset bird/fungus accuracy differences are not algorithmic gains.

Sources: [official dataset](https://github.com/BohemianVRA/FungiTastic), [author Kaggle mirror](https://www.kaggle.com/datasets/picekl/fungitastic), [CLIP model card](https://github.com/openai/CLIP/blob/main/model-card.md).

```powershell
.\.venv\Scripts\python.exe -m unittest test_fungitastic test_confidence test_statistics -v
.\.venv\Scripts\python.exe fungitastic_experiment.py all
.\.venv\Scripts\python.exe report_fungitastic.py
```

Photos,feature caches and weights remain local on E:. Only code,configurations and aggregated reports are published. Preparation currently uses historical local fingerprints for project-overlap screening;see data_audit.json.
