# CUB attribute-quality diagnostic

This is a controlled diagnostic on a new project dataset, not a claim that the AG-CLIP paper has been reproduced. It separates two questions: whether visible image-level attributes are learnable, and whether their supervision improves species recognition beyond matched controls.

## Data and isolation

`download_cub.py` downloads the official 1,150,585,339-byte CUB-200-2011 archive from CaltechDATA, verifies MD5 `97eceeb196236b17998738112f37df78`, then safely extracts regular files under `data/cub`. Interrupted range downloads resume in order; unverified archives are not extracted.

The pilot uses 20 species selected before performance evaluation. Ten common-name suffix groups are randomly sampled, with two species in each group; this is a sampling heuristic, not verified taxonomy. One species per pair is seen in training. Four of the remaining species are development-unseen; six are final-unseen. Existing official train/test image flags are respected:

| Role | Classes | Photos |
|---|---:|---:|
| Seen training | 10 | 200 from official train |
| Seen development | same 10 | 100 different official train |
| Unseen development | separate 4 | 80 official train |
| Seen final evaluation | same 10 | 200 official test |
| Unseen final evaluation | separate 6 | 120 official test |

The 700 selected photos have distinct SHA256 hashes. Only seen training photos contribute gradients. Final images are not processed by the detector/model until model-selection records are locked. Image sizes, hashes and annotation records are read during manifest construction; they do not determine hyperparameters or model performance selection.

Attribute groups are declared before reading results: wing color, bill shape, breast color, back color, belly color, crown color, wing pattern and breast pattern. At most three attributes per group are selected using **training-only** support (at least eight positive and twenty negative labels). This yields 24 attributes. Selected attributes cover common signals; they are not guaranteed to capture the most discriminative cues for every species pair.

CUB certainty IDs are 1=not visible, 2=guessing, 3=probably, 4=definitely. Targets require certainty>=3 and a corresponding visible part inside the actual input's approximate center-crop bounds. Unknown attributes are masked, not relabeled negative. Crop geometry uses annotated part centers, not complete part segmentations, so it is an approximation to full visual evidence. CUB is crowdsourced: “gold” names the filtered human-annotation group, not perfect expert truth.

## Matched implementations

All models initialize from original OpenAI CLIP ViT-B/32. The last two visual Transformer blocks, final norm and projection are trainable. Regional variants duplicate the visual tower and use the same two predicted OWL-ViT crops, selected from five fixed queries: bird head, wing, breast, tail and whole bird. True class labels never choose detector queries or boxes.

A shared linear attribute readout initializes from differences between positive and negative CLIP text vectors. Regional predicted attribute probabilities weight the fixed attribute text bank; CAF fuses region and text tokens with the whole image. Both training and inference use **predicted** attributes in fusion. Ground-truth attributes and parts are permitted only as loss targets or evaluation annotations.

| Variant | Attribute training target | Extra regional label loss |
|---|---|---|
| `finetune` | None; whole image only | None |
| `region_only` | None; fusion text zero | None |
| `automatic` | Frozen CLIP positive/negative prompt soft scores | None |
| `gold` | Filtered per-image human attributes | None |
| `shuffled` | Human values permuted across training images | None |
| `gold_region` | Filtered per-image human attributes | Attributes whose visible part lies in the predicted crop |

The first two retain a frozen text-initialized attribute readout for measurement; their attribute mAP is not a separately trained prediction head. They must not be interpreted as supervised attribute models. Whole-image-only variants have no meaningful regional attribute predictor. Automatic targets rely on uncalibrated CLIP negation prompts and are not the previous iNaturalist OWL attribute IDs. The shuffled group preserves each attribute's positive counts and known/unknown mask while breaking image-target correspondence. `gold` versus `gold_region` isolates the extra regional loss while keeping regions and architecture fixed.

The objective is 0.5 CE + 0.5 multi-positive symmetric contrastive loss + 0.5 original-image-feature preservation, plus 1.0 image-level attribute BCE in supervised variants and 0.5 regional BCE for `gold_region`. Repeated class texts count as positives. Unknown targets yield zero loss/gradient. No class-level aggregate attribute signatures are computed from held-out images.

The fixed matrix is six variants × visual LR `1e-5`/`3e-6` × seed 42, with 160 updates per cell. Fusion/readout parameters use ten times the visual LR. Microbatch 2, effective batch 16, BF16, activation checkpointing and exact two-pass gradient caching fit the local RTX 5060 Laptop. Fixed native CLIP transforms avoid stochastic-replay differences. Early frozen weights are checked for no change.

## Model selection and interpretation

Checkpoints at updates 0, 40, 80, 120 and 160 are selected by development GZSL H, then mean U/S, then lower CE. A separate checkpoint and LR are selected by development attribute mAP. Both selections are sealed before final evaluation. Update zero is eligible and explicitly reported as untrained. Attribute-selected checkpoints diagnose attribute learning; they are not extra candidates for choosing the best final classification score.

ZSL evaluates only six unseen candidates. GZSL evaluates ten seen plus six unseen candidates; S and U are mean per-class accuracies and H=2SU/(S+U). Attribute mAP and fixed-threshold macro-F1 use known labels with both positive and negative support. Region metrics score only visible, known attributes in predicted crops. Predictions with zero/permuted fusion text are post-hoc sensitivity probes; they do not erase semantic information already learned by the encoder.

An additional CPU linear probe learns image attributes from frozen original whole-image features. Its LR (`0.01`/`0.001`) and checkpoint (0/100/300/600) are selected by development mAP only. This diagnoses whether the original representation contains learnable attributes; it is not a species classifier and not proof of AG benefit.

Paired within-class bootstrap intervals condition on the chosen species and seed. They do not estimate training-seed variance; exploratory contrasts are not corrected for multiplicity. CUB officially overlaps with ImageNet, and CLIP overlap is unknown. “Unseen” therefore means unseen during this task's adaptation, not necessarily absent from pretraining.

## Local commands and outputs

Double-click `cub_models.cmd` to run or verify the complete workflow. Completed matrix cells are skipped after provenance checks; interrupted cells restart deterministically. Do not launch duplicate GPU workflows. Individual commands:

```powershell
cd E:\ELEC4240\SpeciesRecognition
.\.venv\Scripts\python.exe download_cub.py
.\.venv\Scripts\python.exe verify_cub.py
.\.venv\Scripts\python.exe audit_cub.py
.\.venv\Scripts\python.exe probe_cub.py train
.\.venv\Scripts\python.exe cub_experiment.py all
.\.venv\Scripts\python.exe probe_cub.py evaluate
.\.venv\Scripts\python.exe verify_cub_deployment.py
.\.venv\Scripts\python.exe diagnose_cub.py
.\.venv\Scripts\python.exe compare_cub_attributes.py
.\.venv\Scripts\python.exe report_cub.py
.\.venv\Scripts\python.exe predict_cub.py "E:\path\bird.jpg"
```

Default prediction uses the classification-selected `gold` model and all 20 configured species. `--variant gold_region` selects the additional regional-supervision model. `--selection attribute` uses the separately locked attribute-best checkpoint, useful when the classification criterion selected update zero. The output always states the checkpoint update; do not treat an untrained selected checkpoint as a benefit from attribute training. Scores are not calibrated probabilities. Like the previous runner, a single photo is duplicated to microbatch two and the duplicate output discarded to preserve BF16 kernel-shape consistency.

Artifacts are under `reports/cub_attributes_v1`; local photo audits under `work/cub_attributes_v1`; checkpoints, histories, predictions and selection locks under `runs/cub_attributes_v1`. `diagnose_cub.py` applies text interventions to the trained attribute-selected gold models and measures how similar their attribute-text mixtures are across images; these are post-hoc diagnostics, never used to retune the evaluated models. Original data, audit photos, caches and weights remain local and are excluded from Git. The new version does not overwrite the earlier iNaturalist experiments. Byte-hashed source files and protocol line endings are pinned in `.gitattributes` so checkpoints survive Windows Git checkout.

Sources: [official archive and MD5](https://data.caltech.edu/records/65de6-vp158), [official dataset overview and overlap warning](https://www.vision.caltech.edu/datasets/cub_200_2011/), and the downloaded CUB README, attributes/certainties.txt and parts/parts.txt.
