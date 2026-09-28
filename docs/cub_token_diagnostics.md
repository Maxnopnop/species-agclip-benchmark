# Frozen-global attribute-token diagnostics

The evolving combined report is [summary_zh.md](../reports/cub_followup_v1/summary_zh.md). Each version is immutable after training starts. Sources/configs are byte-hashed; do not edit an existing trained version to tune it. Create a new version instead. Data, weights and checkpoints stay local; only code/configs/statistical reports are committed.

| Version | Main question | Training cells |
|---|---|---:|
| cub_tokens_v1 | Frozen visuals, separate vs pooled 24-attribute tokens | 30 |
| cub_bottleneck_v1 | Attribute-only residual, no visual shortcut, 640 updates | 24 |
| cub_coverage_v1 | 24 vs 158 training-supported attributes, development probes | 2 probe grids |
| cub_rich_v1 | Same dense fusion with 158 attributes | 30 |
| cub_sparse_v1 | Dense/top-8 crossed with CE/symmetric contrastive loss and controls | 48 |
| cub_b16_v1 | Same 158-attribute dense experiment using CLIP ViT-B/16 | 30 |
| cub_regional_tokens_v1 | Freeze global features, tune only regional visual tail and fusion | 24 |
| cub100_v1 | 100 new project species, 500 train images, 230 attributes, B/16 | 30 |
| cub_siglip_v2 | Corrected lowercase SigLIP 2 on the same cub100 split | 30 |

`cub_siglip_v1` is excluded: its capitalized class prompts exposed a fast Gemma tokenizer normalization defect. See its `INVALIDATED.md` and `inference_audit.json`; the corrected v2 retrains all controls. The same issue qualifies earlier SigLIP results in `visible_v1`; old code/caches remain immutable for audit. CLIP results are unaffected by this specific tokenizer issue.

Additional diagnostics: `diagnose_cub_certainty.py`, `diagnose_cub_regions.py`, `diagnose_cub_oracle_regions.py`, `diagnose_cub_class_attributes.py`, `diagnose_cub_oracle_transfer.py`, `diagnose_cub_descriptions.py`, `diagnose_cub_mapped_attributes.py`, `diagnose_cub_attribute_transfer.py`, and `diagnose_cub_probe_regularization.py`. Class-profile and region oracles explicitly use privileged information and are not deployable results. Public-description and positive-only mapping alternatives have source links in versioned configs; they use no held-out image annotations.

The new-species CLIP report is `reports/cub100_v1/summary_zh.md`; SigLIP's corrected report is `reports/cub_siglip_v2/summary_zh.md`. The original six-version report covers 186 training cells on the reused 20 species only. Do not mix these counts or metrics with the 100-species experiments.

All fusion versions use seeds 42, 43, 44 and two learning rates, selected by development GZSL H, mean S/U and candidate CE in that order. Selections are locked before each version's final evaluation. Three fusion seeds do not cover split/pretraining/probe variability. This is reused-data exploratory research, not an independent confirmation or full AG-CLIP paper reproduction.

The first five versions use frozen CLIP feature caches. The regional version updates the last two visual Transformer blocks, final visual layer norm/projection and fusion; the original global features, class text, temperature and pretrained attribute probe remain frozen. `region_only` in that version still receives regional attribute **supervision**, but has no attribute text/confidence in fusion. `no_attribute_loss` removes that additional supervision too. These are distinct controls.

The native pretrained global classifier remains available. Prediction uses only a photo and fixed local model/text assets; true attributes and part points are used only in training targets and evaluation. Sparse variants keep eight predicted attributes per valid view. Dense pooled controls repeat their mean token to match sequence length and parameter count. Missing regional views are masked, while global tokens remain valid.

## Run from the project root

Use `.venv\Scripts\python.exe -X utf8` for every command below. Existing versioned caches/checkpoints are reused after provenance verification.

```text
verify_cub_tokens.py
cub_token_experiment.py all
verify_cub_bottleneck.py
cub_bottleneck_experiment.py all
diagnose_cub_tokens.py
diagnose_cub_coverage.py
cub_rich_experiment.py all
cub_sparse_experiment.py all
cub_b16_experiment.py all
verify_cub_regional_tokens.py
cub_regional_experiment.py all
calibrate_cub_tokens.py
calibrate_cub_tokens.py --version cub_b16_v1
diagnose_cub_offsets.py
report_cub_followup.py
```

The shared frozen-feature runner retains an old cosmetic progress string saying “30 cells / 15 selections”; actual counts come from the version's configuration and JSON results (e.g. bottleneck 24/12 and sparse 48/24). No model selection depends on that string.

Coverage probes have zero-initialized linear heads. The original 24-dimensional labels are asserted identical to the expanded target extraction on their shared dimensions. Support selection uses training images only. Physical body-size categories are excluded because pixel scale is not physical size. Expanded attributes change both dimensionality and available supervision, not just wording.

`diagnose_cub_tokens.py` tests stronger posthoc gates and training-seen attribute prototypes. True development attributes/visibility are used only in explicitly labeled oracle diagnostics. A naive prototype classifier is not an upper bound. Calibration and class-offset removal use training/development only for their parameters, then evaluate held-out images after saving the choices.

## Photo-only inference and checks

```text
predict_cub_tokens.py E:\path\bird.jpg --version cub_tokens_v1 --variant tokens --seed 42
predict_cub_tokens.py E:\path\bird.jpg --version cub_b16_v1 --variant tokens --seed 42
verify_cub_token_deployment.py --version cub_tokens_v1
verify_cub_token_deployment.py --version cub_bottleneck_v1 --seed 43
verify_cub_token_deployment.py --version cub_rich_v1
verify_cub_token_deployment.py --version cub_b16_v1
predict_cub_tokens.py E:\path\bird.jpg --version cub100_v1 --variant tokens --seed 42
predict_cub_regional.py E:\path\bird.jpg
predict_cub_siglip.py E:\path\bird.jpg --variant tokens --seed 42
verify_cub_token_deployment.py --version cub100_v1
verify_cub_regional_deployment.py
verify_cub_siglip_deployment.py
```

The default prediction candidate set is all 20 species (or 100 in cub100/SigLIP); final benchmark metrics restrict candidates to 10 seen + 6 final-unseen species (or 50 + 25). These are different inference settings. No benchmark ground-truth label chooses inference regions or attribute prompts. Softmax scores are not calibrated probabilities.

The GPU gradient-cache test for regional training compares gradients against a direct micro-batched reference, checks frozen global logits and readout, and rejects held-out training labels. Deployment checks regenerate detector boxes and image features from original photos before comparing logits/probabilities to cached evaluation.

Reference: [AG-CLIP author manuscript](https://www.researchgate.net/publication/399822501_AG-CLIP_Attribute-Guided_CLIP_for_Zero-Shot_Fine-Grained_Recognition), DOI 10.1109/OJCS.2026.3654171. The paper uses attribute mining, region grounding, CAF and contrastive training; its larger backbone/training split are not reproduced here.
