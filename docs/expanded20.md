# Expanded development protocol, version 2

This experiment increases the previous 60-image pilot to 20 species and 1,000 images. It tests whether attributes help under a harder, taxonomy-selected task. It does not promise a lower baseline or a positive AG result. Baselines receive the same pretrained weights, sampled training images, optimizer, update budget and validation selection rule as their corresponding AG variants.

## Dataset fixed before results

From the 73 complete species available in the downloaded official iNaturalist archive prefix, exclude all four previous pilot species. Ten remaining families contain at least two species. Randomly choose two from each eligible family using seed 20260928. This gives ten insect species and ten plant species. Selection uses taxonomy and availability, not model predictions. Same-family membership does not guarantee visual similarity. The fixed identities are in `configs/expanded_classes.json`.

Each species has 50 images: 30 candidate training, 10 validation and 10 held-out test, with seed 20260928 plus the original category ID. Exact content hashes may not cross splits. Perceptual near-duplicates and observation-level grouping have not been audited. All images come from train_mini; this remains separate from the formal 100-species experiment, whose test images come from official val. The whole image archive is still incomplete, so its final archive MD5 has not yet been verified; extracted JPEGs are decoded and individually hashed.

## Predeclared comparison

Five backbones × 5/10/20 images per species × seeds 42/43/44 × five variants = 225 comparisons. All visual and text encoders remain frozen. Each setting first receives 200 alignment updates. Every comparison variant starts from that same alignment checkpoint and receives 200 additional updates, batch size at most 32, AdamW learning rate 0.001 and weight decay 0.0001, cosine schedule. Validation is evaluated every 20 updates. Highest validation macro-F1 wins; lower validation cross-entropy breaks ties. Test accuracy never selects checkpoints or attribute variants.

| Variant | Visual input | Attribute mechanism |
|---|---|---|
| baseline | Whole image | Class-name text-space alignment only |
| region_only | Whole image + five fixed crops | Same attention architecture as AG, with attribute token inputs set to zero; controls for crops and added capacity |
| ag_mean | Same images/crops | Match each crop to the four closest attribute embeddings, encode visual/text tokens and average |
| ag_attention | Same images/crops | Learned attention from whole-image query to visual/attribute tokens |
| ag_aux | Same as ag_attention | Additional positive class-attribute prototype cosine loss, fixed weight 0.1 |

The crop policy uses four corners and a center window, each spanning 65% of image width and height. Every image has five candidate windows. These are not OWL-ViT detections, verified object parts, or evidence that the desired attribute is present. Candidate attributes are retrieved from the same complete bank for every image; inference never uses a true category label. Attribute supervision uses training labels only. The older OWL-ViT pilot remains unchanged.

The shared bank contains 58 visual descriptions for the 20 species, with source URLs in `configs/expanded_attributes.json`. Descriptions were paraphrased and checked against natural-history references, not labeled on individual photographs by an expert. Appearance can vary by sex, stage, variety and view. Auxiliary supervision therefore supplies positive class-level knowledge, not a claim that every described trait is visible in every image.

Fusion uses a 128-dimensional bottleneck and a nonzero gate initialized at 0.1. Both the global and residual context features are normalized. Training checks that the new branch receives nonzero first-step gradients and records parameter changes and the selected update number. This fixes the earlier zero-gate/first-update selection interaction without selecting a model on test accuracy.

These variants are AG-CLIP-inspired adaptations. They are not the paper's complete CoCa/LLM/OWL-ViT implementation and do not establish generic CLIP or unseen-species zero-shot capabilities.

## Selection and statistical evaluation

For each backbone and shot count, choose among ag_mean, ag_attention and ag_aux by mean validation macro-F1 over all three seeds, breaking ties with mean validation loss. Save `selection_locked.json` before evaluating test labels. Report all five variants even if some perform worse.

Report per-seed metrics and their mean and sample standard deviation. For paired hypothesis tests, average the three seeds' probabilities into one prediction per test image. Compare selected AG against both baseline and region_only using exact two-sided McNemar tests, with Holm correction across all 30 comparisons (15 settings × 2 references). Also report unadjusted 95% paired bootstrap intervals, sampling within classes. Three seeds on the same 200 test images are not 600 independent examples. A claim that attributes specifically help requires evidence beyond the crop-only control.

## Run and inspect

```powershell
.\.venv\Scripts\python.exe prepare_expanded.py
.\.venv\Scripts\python.exe expanded_benchmark.py all
.\.venv\Scripts\python.exe report_expanded.py
.\.venv\Scripts\python.exe -m unittest -v test_expanded test_statistics
```

Alternatively use `expanded_models.cmd`, only when another expanded workflow is not running. Status and logs are in `runs/expanded20/status.json`, `work/expanded_prepare.log` and `work/expanded_train.log`. Checkpoints and training histories stay in `runs/expanded20`. Uploadable tables and figures go to `reports/expanded20`; dataset images and checkpoints remain excluded from Git.

New inference entry point: `predict_expanded.py --checkpoint <variant/best.pt> --image <photo.jpg>`. Users supply only the image, while the frozen class/attribute text embeddings remain inside the saved model.
