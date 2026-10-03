# Detector-confidence attribute pooling pilot

No additional classification gain was observed: uniform and confidence AG both achieved 92.00% for EfficientNet-B0 and 93.33% for CLIP ViT-B/32. Shuffled confidence tied these results. Fixed-checkpoint weight interventions changed probabilities slightly but no predicted labels in all six backbone/seed settings. This is evidence of low decision sensitivity in this pilot, not proof that attribute weighting is universally ineffective.

Ten classes, 500 existing images: 300 training candidates, 100 validation and 100 historically evaluated test images. Each run uses 20 training images per class. Two frozen backbones, three seeds, seven arms: 42 final models. Each receives 200 initialization steps plus 200 comparison steps; all selections are validation-only and sealed before scoring.

Uniform, confidence-weighted and shuffled-confidence AG heads share identical architecture and initialization. Region/text tokens are pooled, then passed with a global token to CAF. Only pooling weights change. Actual cached OWL-ViT scores are used. Missing detections fall back to the global path. The original CLIP pretraining remains in every CLIP-backbone readout. This is a new ten-class closed-set adaptation, not an exact AG-CLIP replication or the prior twenty-class table.

| Backbone | Method | Accuracy mean (%) | Seed SD (pp) |
|---|---|---:|---:|
| efficientnet_b0 | visual | 88.00 | 1.73 |
| efficientnet_b0 | random_codes | 92.67 | 0.58 |
| efficientnet_b0 | text | 92.67 | 1.53 |
| efficientnet_b0 | region_only | 92.00 | 2.00 |
| efficientnet_b0 | ag_uniform | 92.00 | 2.00 |
| efficientnet_b0 | ag_confidence | 92.00 | 2.00 |
| efficientnet_b0 | ag_shuffled | 92.00 | 2.00 |
| clip_vit_b32 | visual | 90.00 | 1.73 |
| clip_vit_b32 | random_codes | 90.67 | 1.53 |
| clip_vit_b32 | text | 93.00 | 1.73 |
| clip_vit_b32 | region_only | 93.33 | 1.15 |
| clip_vit_b32 | ag_uniform | 93.33 | 1.15 |
| clip_vit_b32 | ag_confidence | 93.33 | 1.15 |
| clip_vit_b32 | ag_shuffled | 93.33 | 1.15 |

Paired tests use one three-seed ensemble prediction per image. Holm correction covers six predeclared contrasts: confidence AG versus uniform AG, text adaptation, and shuffled confidence for both backbones. See paired_comparisons.json for fixes/breaks, exact McNemar tests and conditional class-stratified bootstrap intervals. See verification.json for detector coverage and fixed-checkpoint weight interventions.

Limitations: historically reused small test set, frozen features, short common budget rather than separately optimal tuning, at most two detections, uncalibrated detector confidence, no verified attribute correctness, and one fixed random-prototype bank. No post-result tuning was performed. No significance is not proof of no effect. A degenerate [0,0] bootstrap interval for identical observed predictions does not establish a zero population effect.
