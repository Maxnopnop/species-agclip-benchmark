# Grounded CLIP starter results

This local CLIP ViT-B/32 implementation trains the last two blocks of independent whole-image and attribute-region visual towers, using OWL-ViT proposals and CAF self-attention. It is an AG-inspired structural reconstruction, not an exact CoCa-paper reproduction.

The reused 20-species iNaturalist subset contains 300 training images, 180 development images and 220 final evaluation images. Four variants, two learning rates and seed 42 yield eight 120-update runs. Development GZSL H selects checkpoints before final evaluation. Update zero is eligible and indicates an untrained selected checkpoint. ZSL uses six unseen candidates; GZSL uses ten seen plus six unseen candidates.

| Model | Selected update | ZSL (6 classes) | GZSL U | GZSL S | GZSL H |
|---|---:|---:|---:|---:|---:|
| Original CLIP | 0 | 93.33 | 80.00 | 50.00 | 61.54 |
| CLIP fine-tuning | 40 | 93.33 | 70.83 | 70.00 | 70.41 |
| Regions, zero text | 40 | 93.33 | 70.00 | 71.00 | 70.50 |
| Attribute-guided | 40 | 93.33 | 70.00 | 71.00 | 70.50 |
| Shuffled attributes | 40 | 93.33 | 70.83 | 71.00 | 70.92 |

AG minus original CLIP H: +8.96 percentage points; paired within-class bootstrap 95% interval [+3.45, +14.94]. AG minus fine-tuning H: +0.08. This interval does not estimate training-seed variability; these are reused exploratory data and pretraining overlap is unknown.

AG minus region-only H: +0.00; AG minus shuffled H: -0.42. The improvement over original CLIP must not be attributed solely to semantic attributes. These controls do not establish an independent classification benefit from the attributes.

Microbatch 2 and exact two-pass gradient caching produce an effective contrastive batch of 16. BF16 and activation checkpointing use a measured maximum allocation of 1187 MiB. Eight runs including development validation took 11.1 minutes, excluding model loading and region detection. Smoke tests prove gradient equivalence, updates to both visual towers and CAF, unchanged frozen early layers, unseen-training rejection and exact missing-region fallback.

The 58 fixed attribute prompts are weak semantic supervision. OWL-ViT found no valid region for 105/480 development images, and visual inspection found background crops and incorrect attributes. Region-only and shuffled controls share the attribute-driven detector; neither removes all semantic guidance. At inference users supply only an image, while the deployed system still uses fixed text banks and the detector. The resulting predictor covers the configured candidate species, not arbitrary wildlife.
