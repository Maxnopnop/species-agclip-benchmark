# Five-backbone visual / text-adapted / attribute-guided comparison

90 newly trained controls plus 225 individually replayed historical cells. 20 species, 1,000 images; 5/10/20 training images per class; 3 seeds. Closed-set, frozen-feature, exploratory follow-up on previously evaluated test data.

Visual linear heads use no text or attributes. Random-code heads have the exact text-adaptation projection/temperature architecture but random fixed class prototypes. Existing text-adapted and AG heads retain their original validation-selected checkpoints. All controls receive the same 200+200-step training and checkpoint schedule. CLIP ViT-B/32 retains CLIP pretraining under every readout.

| Backbone | Shots | Visual linear (%) | Random codes (%) | Text-adapted (%) | Selected AG (%) |
|---|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 5 | 62.33 | 66.00 | 66.83 | 67.17 |
| EfficientNet-B0 | 10 | 67.17 | 74.00 | 73.83 | 74.33 |
| EfficientNet-B0 | 20 | 73.33 | 80.00 | 81.50 | 80.83 |
| ResNet-18 | 5 | 47.83 | 60.67 | 59.83 | 58.33 |
| ResNet-18 | 10 | 61.67 | 69.00 | 67.33 | 68.17 |
| ResNet-18 | 20 | 63.67 | 74.33 | 75.67 | 74.50 |
| ConvNeXt-Tiny | 5 | 60.83 | 67.50 | 68.83 | 68.67 |
| ConvNeXt-Tiny | 10 | 70.33 | 78.50 | 78.50 | 76.83 |
| ConvNeXt-Tiny | 20 | 71.83 | 81.83 | 83.00 | 83.17 |
| ViT-B/16 | 5 | 56.50 | 60.00 | 63.00 | 63.33 |
| ViT-B/16 | 10 | 63.83 | 72.00 | 73.17 | 71.67 |
| ViT-B/16 | 20 | 67.00 | 78.00 | 79.83 | 78.83 |
| CLIP ViT-B/32* | 5 | 61.67 | 73.33 | 74.50 | 74.83 |
| CLIP ViT-B/32* | 10 | 70.50 | 77.83 | 79.33 | 80.17 |
| CLIP ViT-B/32* | 20 | 75.17 | 83.00 | 83.67 | 85.17 |

Positive paired ensemble gains surviving Holm correction over all 60 contrasts: text vs linear 8/15; text vs random 0/15; AG vs text 0/15; AG vs region-only 0/15.

Random prototypes also outperform the ordinary linear readout in all 15 settings. This prevents attribution of the full text-versus-linear gap to language semantics: parameterization, normalization, temperature, optimization and prototype geometry differ. The random bank uses one fixed seed, so uncertainty from resampling prototypes is not covered. The largest mean AG gain is 1.50 percentage points for CLIP ViT-B/32 at 20 shots (83.67% to 85.17%); it is not significant after correction.

Linear heads have not fully fit the training examples at 20 shots; this alone does not distinguish limited optimization, limited capacity or validation checkpoint selection. No post-result tuning was performed. These are not best-tuned end-to-end visual baselines.

Intervals and tests condition on fixed trained models and a historically reused test set. They do not establish independent generalization, zero-shot capability, or an exact AG-CLIP replication. See the Chinese report, all_results.csv and paired_comparisons.json for full details.
