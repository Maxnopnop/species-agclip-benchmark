# CoCa ViT-L/14: small AG-CLIP-inspired replication

The CAF attribute model changes unseen-class ZSL accuracy by-0.75pp versus the equally trained CoCa baseline. Two-seed exploratory results,not a reproduction of the paper score or a significance claim.

## Implemented pipeline

Public LAION-2B CoCa ViT-L/14 weights,2.55GB verified against publisher LFS SHA-256. Native768-dimensional image/text space. Frozen visual blocks1-23 and text blocks1-11 are cached;last blocks,pooling,normalization,projections and attribute/CAF modules are trained. Global images and regions share the visual encoder.

Cached-tail outputs matched native image/text forward exactly in smoke testing. Visual/text tails,attribute MLP and CAF receive nonzero gradients after two updates. Zero-initialized residual outputs preserve the native global embedding at initialization;zero CAF internal gradient on the first update is expected and becomes nonzero on the next.

## Data and selection

25 project-unused CUB species,700 images:10 seen classes with20 training,10 validation and10 final seen-test images each;5 dev-unseen classes with20 images each;10 final-unseen classes with20 images each. All class groups disjoint. Official training photos used for training,official test photos for remaining roles. Metadata/fixed random sampling,never prediction-based sampling.

Batch10,one image per seen class,with class-name text pairs and symmetric contrastive loss.80 updates per arm,two seeds;encoder LR1e-5,new-module LR1e-4. Select development GZSL H,then dev ZSL,then earlier step;step0 is eligible. Seal every selection before final test grounding/extraction.

Step0 selected: 4/10. Histories,selected steps and parameter-change norms: training_diagnostics.json.

## Held-out results

| Method | Unseen-only ZSL (%) | Seen S (%) | Unseen U (%) | GZSL H (%) | Selected steps |
|---|---:|---:|---:|---:|---|
| Native CoCa,no adaptation | 97.00 | 87.00 | 95.50 | 91.05 | 0 |
| baseline | 97.25 ± 0.35 | 91.00 | 94.50 | 92.72 | [40, 80] |
| attributes | 97.50 ± 0.71 | 91.00 | 95.00 | 92.96 | [40, 60] |
| caf | 96.50 ± 0.71 | 89.50 | 93.75 | 91.53 | [0, 60] |
| confidence | 95.75 ± 1.77 | 92.00 | 90.25 | 90.83 | [20, 0] |
| regions | 97.00 ± 0.00 | 87.00 | 95.50 | 91.05 | [0, 0] |

ZSL chooses among10 final unseen classes;GZSL among10 seen+10 final unseen classes. S/U are per-class means;H is computed per run then averaged. ZSL +/- is sample SD across two seeds,not a confidence interval. Native evaluated once. attributes=no CAF;caf=attributes+CAF;confidence=CAF with detector-score pooling;regions=same CAF but zero attribute-text vectors.

## Differences from the paper

| Item | Paper | This pilot |
|---|---|---|
| Backbone | CoCa ViT-L/14 | Same architecture;public LAION-2B checkpoint,exact paper checkpoint unspecified |
| CUB protocol | 150 seen / 50 unseen | 10 seen / 5 development unseen / 10 final unseen |
| Encoder updates | Image/text encoder fine-tuning | Last visual/text block + pooling/projection only |
| Training | 50 epochs,batch64 | 80 updates,batch10,two seeds |
| Attributes | GPT-4o mining/filtering | Fixed24 shared bird prompts;no test per-image attribute labels |
| Grounding | OWL-ViT top-K | OWL-ViT top2,threshold0.05 |
| Fusion | Attribute aggregation + CAF | Explicit shared encoder,average-before-MLP,custom256dim CAF and zero residual initialization |

Paper Table4 reports CoCa65.4% -> AG without CAF69.0% -> AG with CAF73.3% on CUB50 unseen classes (+7.9pp total). Plant disease70.2% ->78.8% ->84.6% (+14.4pp). [Author paper](https://www.researchgate.net/publication/399822501_AG-CLIP_Attribute-Guided_CLIP_for_Zero-Shot_Fine-Grained_Recognition).

Paper descriptions leave addition/concatenation and shared/separate visual encoders ambiguous;this implementation makes explicit choices. Absolute pilot accuracy with fewer candidate classes is not comparable to73.3% in the paper. Detector-confidence weighting is not the source of the paper-reported gain.

## Limits and reproduction

Project-unused does not rule out CoCa/OWL-ViT pretraining exposure to CUB. Attribute grounding is not manually certified. Small samples,two seeds and short training test feasibility and initial direction only;negative results do not refute the full paper,positive results do not establish exact replication. The test set is now observed for future development.

```powershell
.\.venv\Scripts\python.exe coca_download.py
.\.venv\Scripts\python.exe -m unittest test_coca_pilot -v
.\.venv\Scripts\python.exe coca_experiment.py all
.\.venv\Scripts\python.exe report_coca_pilot.py
```

Preparation depends on local official CUB images and historical manifests/fingerprints. Weights,photos and intermediate prefixes remain local;GitHub contains code/configurations/split identities and reports only.
