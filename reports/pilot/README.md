# Benchmark results

PILOT ONLY: 4 species, 60 real images; 5 train / 5 validation / 5 held-out images per species. This checks the workflow, not the planned 100-species experiment. One seed cannot establish a reliable improvement.

Frozen visual and text encoders; supervised projection/adapters and attribute fusion. AG-CLIP-inspired adaptation, not an exact reproduction of the paper. No claim of unseen-species zero-shot performance.

| Backbone | Shots | Variant | Seeds | Top-1 mean | Macro-F1 mean |
|---|---:|---|---:|---:|---:|
| clip_vit_b32 | 5 | agclip | 1 | 0.850 | 0.852 |
| clip_vit_b32 | 5 | average | 1 | 0.850 | 0.852 |
| clip_vit_b32 | 5 | baseline | 1 | 0.850 | 0.852 |
| convnext_tiny | 5 | agclip | 1 | 0.900 | 0.899 |
| convnext_tiny | 5 | average | 1 | 0.900 | 0.899 |
| convnext_tiny | 5 | baseline | 1 | 0.900 | 0.899 |
| efficientnet_b0 | 5 | agclip | 1 | 0.850 | 0.847 |
| efficientnet_b0 | 5 | average | 1 | 0.900 | 0.903 |
| efficientnet_b0 | 5 | baseline | 1 | 0.850 | 0.847 |
| resnet18 | 5 | agclip | 1 | 0.800 | 0.785 |
| resnet18 | 5 | average | 1 | 0.800 | 0.806 |
| resnet18 | 5 | baseline | 1 | 0.800 | 0.785 |
| vit_b_16 | 5 | agclip | 1 | 0.900 | 0.899 |
| vit_b_16 | 5 | average | 1 | 0.900 | 0.899 |
| vit_b_16 | 5 | baseline | 1 | 0.900 | 0.899 |

Region coverage: 21/60 images have at least one detection; mean 0.53 regions per image. Missing regions use the global baseline feature.

See `region_coverage.json` for detection coverage by species and split. Low coverage limits how often the attribute branch can change the global representation.

Training-time fields exclude frozen feature extraction, OWL-ViT grounding and downloads. They are not end-to-end deployment latency.

Top-5 accuracy is uninformative for a four-class pilot. Test metrics should not guide further tuning.
