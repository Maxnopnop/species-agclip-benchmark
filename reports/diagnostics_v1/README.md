# Attribute-guidance diagnosis: validation only

This report diagnoses the existing 20-species system. It does not evaluate a new test set, establish statistical significance, or claim a paper reproduction. The same validation set has already been used for model development.

## 1. Interventions on existing checkpoints

Existing 10-shot checkpoints, three seeds, 200 validation images per run. Retrieval keys stay unchanged: permuted mode replaces the selected value embeddings with a fixed derangement; zero removes only text values; no_branch disables the entire fusion residual. This separates sensitivity to text values from sensitivity to the full branch. The auxiliary training loss itself is not undone by these inference interventions.

| Backbone | Existing variant | Original accuracy | Permuted text | Zero text | Zero-text prediction changes / 200 | Entire branch disabled: changes / 200 |
|---|---|---:|---:|---:|---:|---:|
| efficientnet_b0 | ag_mean | 75.67% | 75.67% | 75.67% | 0.00 | 1.33 |
| efficientnet_b0 | ag_attention | 75.67% | 75.67% | 75.67% | 0.00 | 1.00 |
| efficientnet_b0 | ag_aux | 76.67% | 76.67% | 76.67% | 0.00 | 3.33 |
| clip_vit_b32 | ag_mean | 80.67% | 80.83% | 80.17% | 1.00 | 2.33 |
| clip_vit_b32 | ag_attention | 80.33% | 80.17% | 79.83% | 1.00 | 1.33 |
| clip_vit_b32 | ag_aux | 80.67% | 80.33% | 80.33% | 2.33 | 1.67 |

## 2. Architecture-matched retraining and tail adaptation

96 runs: two backbones x two regimes x four modes x two head learning rates x three seeds. Each uses 10 training images per species, the corresponding existing alignment initialization, 100 updates, batch size 16, and validation at steps 50 and 100. The same two-candidate learning-rate budget is used in every condition. Matched/permuted/zero use identical attention architecture and initialization, without an auxiliary attribute loss. Baseline uses the whole image only.

Matched means the ordinary retrieved attribute embeddings, not guaranteed correct image-level descriptions. Permuted values retain fixed structure that a trained network could partly learn around; this is a semantic disruption control, not independent random noise on every batch.

Tail adaptation updates EfficientNet final MBConv stage and final convolution, or CLIP final transformer block, post-normalization and projection. Earlier layers stay frozen. BatchNorm running statistics, dropout in the visual backbone, and stochastic depth remain in evaluation mode in both regimes. Float32 cached prefix activations reproduce the full encoder before adaptation.

The table selects one learning rate per condition by mean validation macro-F1, then loss. These are development scores on the selection set, not unbiased final performance. All 96 runs and same-learning-rate paired differences are retained.

| Backbone | Regime | Mode | Selected head LR | Training accuracy mean | Validation accuracy mean +/- SD |
|---|---|---|---:|---:|---:|
| efficientnet_b0 | frozen | baseline | 0.001 | 100.00% | 76.17 +/- 2.36% |
| efficientnet_b0 | frozen | matched | 0.001 | 100.00% | 76.00 +/- 1.80% |
| efficientnet_b0 | frozen | permuted | 0.001 | 100.00% | 76.00 +/- 1.80% |
| efficientnet_b0 | frozen | zero | 0.001 | 100.00% | 76.00 +/- 1.80% |
| efficientnet_b0 | tail | baseline | 0.001 | 100.00% | 76.17 +/- 2.31% |
| efficientnet_b0 | tail | matched | 0.001 | 100.00% | 76.83 +/- 2.47% |
| efficientnet_b0 | tail | permuted | 0.001 | 100.00% | 76.83 +/- 2.47% |
| efficientnet_b0 | tail | zero | 0.001 | 100.00% | 76.67 +/- 2.31% |
| clip_vit_b32 | frozen | baseline | 0.001 | 100.00% | 80.50 +/- 3.12% |
| clip_vit_b32 | frozen | matched | 0.001 | 100.00% | 80.50 +/- 3.12% |
| clip_vit_b32 | frozen | permuted | 0.001 | 100.00% | 80.50 +/- 3.12% |
| clip_vit_b32 | frozen | zero | 0.001 | 100.00% | 80.50 +/- 3.12% |
| clip_vit_b32 | tail | baseline | 0.0003 | 100.00% | 81.83 +/- 3.40% |
| clip_vit_b32 | tail | matched | 0.0003 | 100.00% | 81.83 +/- 3.40% |
| clip_vit_b32 | tail | permuted | 0.0003 | 100.00% | 81.83 +/- 3.40% |
| clip_vit_b32 | tail | zero | 0.0003 | 100.00% | 81.83 +/- 3.40% |

## 3. Retrieval and visual inspection

Retrieval proxies compare retrieved attribute IDs with the species description list and with insect/plant group membership. They do NOT measure true visible-attribute accuracy: unlisted traits may still be valid and listed traits may be invisible. Native CLIP, aligned models, and trained attention models are all preserved in retrieval_proxies.json.

One validation image per species was selected by fixed seed before viewing. The assistant inspected the center crop and each model's top-1 prompt, recording supported, unsupported, uncertain, or background descriptions. This is not expert biological annotation. Local contact sheets remain under work/diagnostics_v1 and are not uploaded to GitHub.

Assistant review counts: {"efficientnet_b0": {"uncertain": 4, "unsupported": 10, "background": 1, "supported": 5}, "clip_vit_b32": {"supported": 8, "uncertain": 6, "unsupported": 6}}.

Illustrative failures include grass background selected for a tiny butterfly (label 4), blue-upper-wing descriptions for a pale underside view (label 5), yellow flowers described when only leaves/dry stems are visible (label 13), and red berries described where pale flower clusters are visible (label 19). These examples were not used to replace or tune the attribute bank during this diagnostic run.

## Interpretation limits and next experiment

Inference sensitivity directly tests whether the trained model relies on attribute values, but small sensitivity does not prove that auxiliary text supervision had no effect during training. Retraining controls and tail adaptation narrow hypotheses; they do not prove that all AG methods fail. The limited data, fixed crops, small learning-rate grid and 100-step budget remain constraints. No test-driven model replacement was performed.

The next development experiment should verify region/attribute matches, use a small independently reviewed image-level attribute set, and test a direct attribute-alignment objective. A new held-out evaluation is required before claiming improvement after these validation-driven changes.
