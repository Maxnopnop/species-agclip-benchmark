# Fresh-data validation of confidence-weighted attribute aggregation

No significant confidence-weighting advantage was established. EfficientNet-B0 improved from 95.50% uniform AG to 96.00% weighted AG in seed-mean accuracy (+0.50 pp); CLIP ViT-B/32 tied at 96.67%. None of the six predeclared paired contrasts had a positive gain surviving Holm correction.

Ten CUB species excluded from all 120 previously used project CUB species; 500 project-unused images, with 200 training, 100 validation and 200 test images. Official source train/test boundaries respected. Exact file hashes, decoded-pixel hashes and pHash/dHash distance <=4 were screened against historical images and earlier accepted fresh images. This is heuristic duplicate protection, not observation-level or foundation-pretraining independence.

Same two backbones, seven arms, three seeds, frozen features and 200+200-step training recipe as confidence_v1. The original Head/train implementation is reused. A fixed previously selected vocabulary of 24 bird attributes replaces the mixed-species vocabulary. All seeds use the same training images. Validation-only checkpoint choices are sealed before any test grounding or feature extraction. No tuning follows final scores.

| Backbone | Method | Accuracy mean (%) | Seed SD (pp) |
|---|---|---:|---:|
| efficientnet_b0 | visual | 94.83 | 0.76 |
| efficientnet_b0 | random_codes | 94.17 | 2.08 |
| efficientnet_b0 | text | 95.67 | 0.58 |
| efficientnet_b0 | region_only | 95.50 | 0.87 |
| efficientnet_b0 | ag_uniform | 95.50 | 0.87 |
| efficientnet_b0 | ag_confidence | 96.00 | 0.50 |
| efficientnet_b0 | ag_shuffled | 95.50 | 0.87 |
| clip_vit_b32 | visual | 92.50 | 0.00 |
| clip_vit_b32 | random_codes | 94.83 | 0.76 |
| clip_vit_b32 | text | 96.33 | 0.29 |
| clip_vit_b32 | region_only | 97.00 | 0.50 |
| clip_vit_b32 | ag_uniform | 96.67 | 0.29 |
| clip_vit_b32 | ag_confidence | 96.67 | 0.29 |
| clip_vit_b32 | ag_shuffled | 96.67 | 0.29 |

Six predeclared paired comparisons use 200 three-seed ensemble predictions: confidence AG versus uniform AG, text, and shuffled confidence for both backbones. Exact McNemar, class-stratified bootstrap intervals, and Holm correction are reported in paired_comparisons.json. Replay and confidence interventions appear in verification.json.

This is closed-set recognition on new project species, not zero-shot evaluation. CLIP retains language pretraining in the visual readout. New data reduces adaptive reuse of the old test set but does not guarantee absence of training overfitting. Bird-only scores are not directly comparable with the prior mixed-species dataset. Once inspected, this test set is no longer untouched for subsequent method development.
