# Expanded 20-species development benchmark

20 species, 10 family pairs, 1,000 images: 30 candidate training / 10 validation / 10 test per species. Species were selected by taxonomy and archive availability before observing results; all four earlier pilot species are excluded. This is not the formal 100-species experiment.

5 backbones × 3 shot counts × 3 seeds × 5 variants = 225 comparisons. Frozen encoders; 200 alignment updates and 200 additional updates for every comparison variant. Validation macro-F1 selects checkpoints, with validation loss breaking ties. AG variants use fixed overlapping crops and a shared source-derived attribute bank, not OWL-ViT anatomical detections. These are AG-CLIP-inspired adaptations, not an exact replication.

AG choice was locked using the mean validation metrics over seeds before any test evaluation. The table reports per-run test accuracy mean ± sample standard deviation over three seeds. Paired tests use one prediction per image from a three-seed probability ensemble, not 600 independent observations.

| Backbone | Shots | Baseline mean ± SD | Selected AG | AG mean ± SD | Mean gain (pp) | Ensemble gain (pp) | Holm p vs baseline |
|---|---:|---:|---|---:|---:|---:|---:|
| efficientnet_b0 | 5 | 66.8 ± 0.8% | ag_aux | 67.2 ± 0.3% | +0.33 | +2.00 | 1.0000 |
| efficientnet_b0 | 10 | 73.8 ± 1.4% | ag_aux | 74.3 ± 2.8% | +0.50 | +3.00 | 1.0000 |
| efficientnet_b0 | 20 | 81.5 ± 1.8% | ag_aux | 80.8 ± 2.5% | -0.67 | -1.50 | 1.0000 |
| resnet18 | 5 | 59.8 ± 1.5% | ag_aux | 58.3 ± 1.0% | -1.50 | +1.50 | 1.0000 |
| resnet18 | 10 | 67.3 ± 3.8% | ag_aux | 68.2 ± 2.1% | +0.83 | -0.50 | 1.0000 |
| resnet18 | 20 | 75.7 ± 1.3% | ag_mean | 74.5 ± 1.8% | -1.17 | -1.50 | 1.0000 |
| convnext_tiny | 5 | 68.8 ± 2.0% | ag_aux | 68.7 ± 0.8% | -0.17 | +0.50 | 1.0000 |
| convnext_tiny | 10 | 78.5 ± 0.5% | ag_aux | 76.8 ± 1.2% | -1.67 | -5.00 | 0.0586 |
| convnext_tiny | 20 | 83.0 ± 0.5% | ag_aux | 83.2 ± 1.3% | +0.17 | -0.50 | 1.0000 |
| vit_b_16 | 5 | 63.0 ± 7.2% | ag_aux | 63.3 ± 4.3% | +0.33 | -2.50 | 1.0000 |
| vit_b_16 | 10 | 73.2 ± 3.2% | ag_aux | 71.7 ± 3.7% | -1.50 | -0.50 | 1.0000 |
| vit_b_16 | 20 | 79.8 ± 1.3% | ag_aux | 78.8 ± 1.3% | -1.00 | -1.00 | 1.0000 |
| clip_vit_b32 | 5 | 74.5 ± 2.3% | ag_mean | 74.8 ± 2.3% | +0.33 | +0.50 | 1.0000 |
| clip_vit_b32 | 10 | 79.3 ± 1.5% | ag_aux | 80.2 ± 2.0% | +0.83 | +1.00 | 1.0000 |
| clip_vit_b32 | 20 | 83.7 ± 1.6% | ag_mean | 85.2 ± 1.0% | +1.50 | +0.50 | 1.0000 |

Positive ensemble gains surviving Holm correction versus baseline: 0/15 settings.

Correction covers all 30 planned comparisons: 15 settings against baseline and the same 15 against the region-only control. Bootstrap intervals are unadjusted 95% paired intervals stratified by class; McNemar tests are exact and two-sided. Statistical conclusions apply to this fixed development set. Shared pretraining, possible near-duplicate observations, class-level rather than image-level attribute annotations, and availability-based species selection limit generalization.

All five variants, including declines, are preserved in `all_test_results.csv`. Source URLs and paraphrased traits are in `configs/expanded_attributes.json`. No test-driven retraining or variant replacement was performed for this report.
