# CUB image-level attribute supervision pilot

Official archive verified; 20 species, 200 seen training images, 180 development images and 320 final evaluation images. Twenty-four attributes were selected using training-only support. Confidence>=3, visible-part and input-center-crop masks exclude uncertain targets. Crowd annotations are not noise-free ground truth. No annotated parts, boxes or true attributes enter model inference.

Classification-selected results:

| Variant | Selected update | ZSL | GZSL H | Attribute mAP | Unseen attribute mAP |
|---|---:|---:|---:|---:|---:|
| Original CLIP | 0 | 92.50 | 75.11 | 41.85 | 42.51 |
| Fine-tuning | 40 | 94.17 | 72.32 | 43.94 | 44.13 |
| Regions, no attributes | 0 | 94.17 | 75.11 | 41.85 | 42.51 |
| Automatic supervision | 0 | 94.17 | 75.64 | 41.85 | 42.51 |
| Human attributes | 0 | 94.17 | 75.64 | 41.85 | 42.51 |
| Shuffled attributes | 0 | 94.17 | 75.64 | 41.85 | 42.51 |
| Human + region supervision | 0 | 94.17 | 75.64 | 41.85 | 42.51 |

Attribute-selected results (a separate locked criterion, not alternative classification model selection):

| Variant | Selected update | ZSL | GZSL H | Attribute mAP | Unseen attribute mAP |
|---|---:|---:|---:|---:|---:|
| Fine-tuning | 80 | 95.83 | 50.53 | 44.13 | 44.49 |
| Regions, no attributes | 80 | 96.67 | 46.73 | 44.00 | 44.37 |
| Automatic supervision | 40 | 95.83 | 54.87 | 43.77 | 44.52 |
| Human attributes | 160 | 97.50 | 45.77 | 50.24 | 47.36 |
| Shuffled attributes | 80 | 96.67 | 47.69 | 43.68 | 43.59 |
| Human + region supervision | 160 | 97.50 | 44.90 | 50.36 | 47.43 |

Frozen-feature linear attribute probe: original 41.86 mAP, supervised probe 58.25. This is an attribute prediction diagnostic, not an AG species classification gain.

Human attributes minus fine-tuning H: +3.32 percentage points, conditional paired bootstrap 95% interval [0.2819005059008941, 6.7391036596520735]. There is one training seed. CUB/ImageNet overlap and unknown CLIP pretraining overlap limit claims about unseen species. Automatic targets are uncalibrated frozen-CLIP positive/negative-prompt scores; shuffled targets preserve training support and marginals. Human+region adds losses for annotated visible parts inside the same predicted detector crops, without oracle input crops.

The classification-selected human model is update 0; update zero means it received no attribute training, so its classification difference is not an attribute-training gain. For the trained attribute-selected models, human supervision improves attribute mAP over shuffled targets by 6.55 points, paired image-bootstrap interval [5.339304130907657, 7.844734925483242]. Unseen-only accuracy improves over original CLIP by 5 points (uncorrected McNemar p=0.03125), but beats region-only/shuffled controls by only one of 120 images (p=1.0). Mixed seen/unseen performance deteriorates. These exploratory contrasts are not multiplicity-corrected.

Gradient caching, annotation isolation, masked losses, parameter updates and eight checkpoint/image-only deployment checks passed (six classification-selected and two trained human-attribute-selected models). This is a CLIP/CAF adaptation, not an exact AG-CLIP/CoCa reproduction. Interpret attribute learning and additional species classification utility separately.
