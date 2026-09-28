# Class-disjoint zero-shot pilot: five backbones

120 development cells and 60 locked evaluations completed. This is a reused-data exploratory pilot, not a strict AG-CLIP reproduction or an independent benchmark. Unseen classes are absent from this round of adaptation training; pretraining exposure is not ruled out.

## Protocol

- Original five backbones: EfficientNet-B0, ResNet-18, ConvNeXt-Tiny, ViT-B/16, OpenAI CLIP ViT-B/32. Original frozen pretrained features are reused; all task projections/fusion modules are reinitialized. No earlier all-20-class task checkpoint is loaded.
- Ten seen species (five insects, five plants): 20 training images each = 200. Four other species: 20 images each = 80 development images for checkpoint/LR selection. Six final unseen species: 20 images each = 120. An additional 200 seen-class images are used only for final GZSL evaluation. Class sets are disjoint; image hashes across roles are unique.
- Species names and class attributes for unseen classes are permitted semantic side information. Unseen images and image labels never participate in gradient updates. Training CE uses only seen candidate classes.
- Two learning rates, three optimization seeds, 200 alignment updates plus 200 comparison updates. The initial alignment weights are shared among variants for each model/LR/seed. All variants have class CE plus 0.25 cosine alignment to the seen class name. The AG variant adds 0.1 cosine alignment to the seen class attribute prototype.
- Compare global baseline, region-only capacity control, attribute-guided fusion/auxiliary alignment, and wrongly paired attributes. Every image receives the same text vocabulary; true labels cannot choose inference crops or text. Fixed five overlapping crops are not anatomical detections.
- Development-unseen macro class accuracy selects checkpoint and LR; CE breaks ties. All 20 model/variant LR choices are written to a fingerprinted selection lock before final evaluation. There is no final refit or test-calibrated seen-class bias correction.

## Results

**Critical additional reference:** original CLIP without task adaptation obtains 93.33% ZSL; GZSL U/S/H = 80.00/51.50/62.66%. This reference was checked post hoc with identical prompts/global-image preprocessing and no tuning. In the table, baseline means a seen-class-adapted baseline, not original CLIP. A gain against that baseline can recover adaptation damage without outperforming native CLIP.

| Backbone | Adapted baseline ZSL mean ± SD (%) | Region-only (%) | AG mean ± SD (%) | Shuffled (%) | AG − adapted baseline (pp) | AG GZSL unseen / seen / H (%) |
|---|---:|---:|---:|---:|---:|---:|
| efficientnet_b0 | 67.78 ± 4.11 | 67.78 | 63.89 ± 3.85 | 62.78 | -3.89 | 0.00 / 93.00 / 0.00 |
| resnet18 | 62.78 ± 1.27 | 63.33 | 61.94 ± 3.94 | 61.39 | -0.83 | 0.00 / 88.83 / 0.00 |
| convnext_tiny | 70.28 ± 2.10 | 70.28 | 68.06 ± 1.27 | 67.78 | -2.22 | 0.28 / 95.50 / 0.55 |
| vit_b_16 | 63.89 ± 2.10 | 63.61 | 61.39 ± 1.27 | 60.83 | -2.50 | 0.00 / 91.00 / 0.00 |
| clip_vit_b32 | 81.11 ± 1.73 | 79.17 | 90.56 ± 0.48 | 75.56 | +9.44 | 1.67 / 94.83 / 3.27 |

ZSL uses six unseen candidates (uniform chance 16.67%). GZSL uses sixteen candidates: ten seen plus six unseen. U and S are mean per-class accuracies; H is their harmonic mean, computed per run then averaged. The four development classes are excluded from final candidate sets. Seed SD describes optimization variation on one fixed class/image split, not variability over different unseen species.

## Paired exploratory comparisons

| Backbone | Reference | Ensemble gain (pp) | Paired image bootstrap 95% CI (pp) | Holm-adjusted McNemar p |
|---|---|---:|---:|---:|
| efficientnet_b0 | baseline | +0.83 | [-3.33, +5.00] | 1.0000 |
| efficientnet_b0 | region_only | +0.83 | [-3.33, +5.00] | 1.0000 |
| resnet18 | baseline | +0.00 | [-3.33, +3.33] | 1.0000 |
| resnet18 | region_only | -1.67 | [-5.83, +2.50] | 1.0000 |
| convnext_tiny | baseline | -0.83 | [-4.17, +2.50] | 1.0000 |
| convnext_tiny | region_only | -0.83 | [-3.33, +1.67] | 1.0000 |
| vit_b_16 | baseline | -2.50 | [-7.50, +2.50] | 1.0000 |
| vit_b_16 | region_only | -4.17 | [-8.33, -0.83] | 0.5000 |
| clip_vit_b32 | baseline | +9.17 | [+4.17, +14.17] | 0.0308 |
| clip_vit_b32 | region_only | +10.00 | [+5.00, +15.00] | 0.0183 |

These compare one three-seed ensemble prediction per image. Holm correction covers ten planned comparisons. Confidence intervals are unadjusted, image-level and conditional on these six classes; they do not measure transfer to a wider population of species. Prior reuse of these images and possible pretraining overlap further limit interpretation.

## Limitations and next dataset

This experiment fixes the seen/unseen class protocol while retaining a small frozen-feature AG-inspired method. It does not implement CoCa-L, a separate trainable attribute visual encoder, OWL-ViT grounding, or the exact CAF/training of the AG-CLIP paper. Improved performance would support this adaptation only; no improvement would not refute the original method. Earlier supervised accuracies with twenty candidate classes are not directly comparable to six-way ZSL.

AwA2 offers 50 animal categories, 37,322 images and 85 class-level attributes; use the proposed split designed to avoid ImageNet category overlap. It is broader than birds but not all labels are strict biological species, and not all attributes are visually observable. CUB offers 200 bird species, 11,788 images, 312 image attributes and 15 part locations; its official page warns of possible ImageNet image overlap. Neither automatically guarantees that CLIP pretraining never encountered a class or image.

- [AwA2 official data and proposed splits](https://cvml.ista.ac.at/AwA2/)
- [CUB official data](https://www.vision.caltech.edu/datasets/cub_200_2011/)

All development and selected final cells, class identities, controls and declines are retained. Photos and model weights remain local.
