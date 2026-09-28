# Class-disjoint zero-shot follow-up

Train on images of seen species. Use a separate set of species for development selection. Evaluate final unseen species only after all model/LR/checkpoint choices are locked. Unseen class names and descriptions are allowed semantic information; images and image labels of final unseen species are excluded from training and selection. If labelled target-class images are used for adaptation, that is few-shot learning instead.

This local pilot reuses existing iNaturalist images, raw frozen pretrained feature caches and source-supported descriptions. Earlier task-trained projections and attribute heads are never reused. Ten family pairs each contribute one seen species; remaining partners are divided deterministically into four development and six final unseen species, balanced between insects and plants. The split and seed are in `configs/zsl_v1.json`; readable identities are in `reports/zsl_v1/class_split.json`.

It is not a new independent dataset: these photos appeared in earlier supervised project experiments. "Unseen" is relative to this adaptation training. ImageNet and CLIP pretraining overlap has not been excluded. This is not the standard split of a published benchmark and results cannot be compared directly to CUB/AwA2 papers.

## Experiment

- Original five backbones: EfficientNet-B0, ResNet-18, ConvNeXt-Tiny, ViT-B/16 and CLIP ViT-B/32. The first four learn a projection into a frozen CLIP text space; CLIP learns a residual adapter. Image/text backbones remain frozen.
- 200 training photos from ten seen species; 80 development photos from four different species; 120 final photos from six unseen species; 200 additional seen-class photos for GZSL. Unique image hashes across roles.
- All classes have fixed semantic prototypes, but training softmax contains only ten seen classes. Shared source-derived attribute vocabulary is the same for every image. The forward function cannot receive true labels. No trainable class-specific output weights prevent recognition of new class prototypes.
- Stage 1: 200 alignment updates. Stage 2: 200 updates for baseline, region-only control, AG fusion plus auxiliary semantic alignment, and shuffled attribute pairing. Stage-1 weights are shared per backbone/seed/LR. Three optimization seeds and two LRs produce 120 development cells.
- The baseline and every comparison use seen-class cross-entropy plus 0.25 cosine alignment to class-name text. AG adds 0.1 cosine alignment to class attributes. The regional branch cross-attends to five fixed image crops with their retrieved attribute text; the wrong-attribute control corrupts both crop/text pairing and seen-class auxiliary targets. It is not a one-factor attribution test.
- Checkpoints and learning rates are selected by development-unseen mean per-class accuracy, then lower CE. All selections are locked before final evaluation. No final refit. Sixty selected seed/variant runs are evaluated on final images.
- ZSL scores only six unseen candidates; GZSL scores ten seen plus six unseen candidates. Report U, S and harmonic mean H without test-set calibration. Development classes are excluded from final candidates. Seed SD does not represent different class splits.

The attribute branch is AG-inspired. Fixed crops, frozen encoders and lightweight fusion differ from the paper's CoCa-L/OWL-ViT/additional attribute visual encoder/CAF design. This pilot isolates a change in task protocol; it does not establish an exact reproduction of the paper.

## Run

```powershell
cd E:\ELEC4240\SpeciesRecognition
.\.venv\Scripts\python.exe -m unittest test_zsl -q
.\.venv\Scripts\python.exe zsl_experiment.py all
.\.venv\Scripts\python.exe diagnose_native_zsl.py
.\.venv\Scripts\python.exe report_zsl.py
.\.venv\Scripts\python.exe verify_zsl.py
```

`train` and `evaluate` stages are available separately; evaluation requires the fingerprinted selection lock. Implementation/protocol changes require a new version directory. Existing expanded20 data and five pretrained feature caches are prerequisites.

The native CLIP reference was added after observing the initial final results. It applies the original global-image/text geometry with no task training or hyperparameter changes. It is reported separately because gains over a seen-class-adapted baseline may merely recover transfer ability lost during adaptation. No final results were used to retrain models or revise selection.

Image-only prediction supports `--candidate-set unseen` for six candidates or `all` for sixteen:

```powershell
.\.venv\Scripts\python.exe predict_zsl.py --checkpoint runs/zsl_v1/MODEL_seed42_lrLR/ag_aux/best.pt --image E:\path\photo.jpg --candidate-set unseen
```

Use the MODEL/LR from the saved selection lock. True image labels, attributes and boxes are not inputs. Fixed text embeddings remain in the head. The scores are not calibrated probabilities.

## Standard datasets for a subsequent benchmark

- [AwA2](https://cvml.ista.ac.at/AwA2/): 50 animal categories, 37,322 images, 85 class-level attributes. Broad animals rather than only birds; not every label is a strict species and habitat/behaviour traits need not be visible in a photograph. The official site recommends the proposed split to avoid ImageNet class overlap.
- [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/): 200 bird species, 11,788 images, 312 image attributes and 15 part locations. Useful for controlled attribute/part grounding; the official site warns about ImageNet/Flickr image overlap.

Neither dataset by itself proves absence from CLIP pretraining. No CUB/AwA2 images were downloaded for this pilot. Photos and weights remain local; code, protocols and numerical reports can be shared.
