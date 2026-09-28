# Visible-attribute and region-alignment follow-up

70 completed training cells. Results below are exploratory **validation**, not held-out test performance.

Fixed 20 species; 200 training images (10/class) and 200 validation images. Sixty images received assistant-reviewed provisional attribute annotations: 40 train and 20 validation. One image in each split is entirely unknown, leaving 39/19 with known labels. Sixteen visual traits; unobserved and non-applicable entries are masked. Some attributes have only one positive training example.

CLIP and SigLIP 2 use three optimization seeds on the same images. FG-CLIP, when present, is a one-seed follow-up. Every variant has two learning rates and four checkpoint evaluations; select checkpoint by validation macro-F1, then CE, and select LR by the same metrics averaged over seeds. This development set has been reused; no significance or generalization claim is made.

| Backbone | Condition | LR | Accuracy mean ± SD (%) | Attribute mAP (%) | Peak inside box (%) |
|---|---|---:|---:|---:|---:|
| clip_b16 | baseline | 0.001 | 81.83 ± 1.26 | 66.57 | 36.11 |
| clip_b16 | capacity_control | 0.0003 | 73.83 ± 1.04 | 48.70 | 43.52 |
| clip_b16 | attribute_supervision | 0.001 | 82.33 ± 2.08 | 71.14 | 25.93 |
| clip_b16 | attribute_region | 0.001 | 81.50 ± 0.50 | 77.14 | 83.33 |
| clip_b16 | shuffled_supervision | 0.001 | 82.00 ± 0.50 | 43.28 | 82.41 |
| siglip2_b16 | baseline | 0.001 | 80.33 ± 1.15 | 44.73 | 36.11 |
| siglip2_b16 | capacity_control | 0.001 | 80.17 ± 0.76 | 47.38 | 45.37 |
| siglip2_b16 | attribute_supervision | 0.001 | 81.67 ± 1.04 | 67.95 | 36.11 |
| siglip2_b16 | attribute_region | 0.001 | 80.67 ± 0.58 | 88.80 | 97.22 |
| siglip2_b16 | shuffled_supervision | 0.001 | 80.50 ± 0.50 | 40.97 | 75.00 |
| fgclip_b16 | baseline | 0.0003 | 76.00 (one seed) | 72.63 | 91.67 |
| fgclip_b16 | capacity_control | 0.001 | 76.00 (one seed) | 48.74 | 75.00 |
| fgclip_b16 | attribute_supervision | 0.0003 | 76.00 (one seed) | 74.01 | 91.67 |
| fgclip_b16 | attribute_region | 0.001 | 76.00 (one seed) | 86.37 | 77.78 |
| fgclip_b16 | shuffled_supervision | 0.001 | 75.50 (one seed) | 53.07 | 69.44 |

## Classification changes and branch interventions

| Backbone | Full AG minus global baseline (pp) | Full AG minus capacity control (pp) | Full AG with attribute branch zeroed (%) | Labels changed by zeroing / 200 | Labels changed by channel permutation / 200 |
|---|---:|---:|---:|---:|---:|
| clip_b16 | -0.33 | +7.67 | 82.50 | 8.33 | 4.33 |
| siglip2_b16 | +0.33 | +0.50 | 80.50 | 3.67 | 2.33 |
| fgclip_b16 | +0.00 | +0.00 | 75.50 | 6.00 | 3.00 |

These post-training interventions measure reliance of a trained model on its branch, not the causal benefit of retraining with attribute supervision. Permuting channels also disrupts the learned linear classifier, so changes alone do not prove semantic use.

## Interpretation and limits

- Attribute prediction and species accuracy are different outcomes. Higher attribute mAP does not establish a species-classification improvement. Compare against the strong global baseline as well as the capacity control; a weak capacity control can exaggerate apparent gains.
- Box localization is coarse. Many boxes enclose an organism or flower cluster, and shuffled annotations can still encourage generic foreground attention. High peak-in-box scores alone do not prove attribute-specific grounding. The CSV includes attention mass and the uniform-area reference.
- Auxiliary labels are assistant-reviewed, not independent expert ground truth. Attribute metrics use only the small reviewed validation subset; only attributes with both positive and negative validation labels enter mAP. Rare-label AP is unstable.
- Frozen encoders with trainable global/patch adapters and a 16-concept additive classification branch. This is an AG-inspired custom experiment, not a strict reproduction of the AG-CLIP paper. No validation labels or boxes are model inputs; at prediction time only an image is supplied, while fixed text embeddings remain inside the checkpoint.
- The new image subset and preprocessing differ from earlier 225-run/96-run experiments. Cross-round scores are not controlled before/after comparisons.
- FG-CLIP already received fine-grained region-text pretraining. It is a stronger alternative backbone, not a model devoid of attribute-related training. Web-pretraining overlap with iNaturalist has not been ruled out, so this is not a contamination-free benchmark.
- All negative results and all learning rates are retained in all_runs.csv. No test images were encoded or evaluated in this follow-up.

## Sources

- [CLIP](https://proceedings.mlr.press/v139/radford21a.html)
- [SigLIP 2 official model](https://huggingface.co/google/siglip2-base-patch16-224)
- [FG-CLIP official model and dense-feature API](https://huggingface.co/qihoo360/fg-clip-base)
- [FG-CLIP paper](https://arxiv.org/abs/2505.05071)
