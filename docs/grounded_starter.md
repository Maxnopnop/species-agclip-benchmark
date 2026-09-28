# Grounded CLIP starter on this PC

This version implements an image-based AG-CLIP adaptation: shared attribute prompts → OWL-ViT boxes → an independent attribute visual tower → attribute/region token projection → CAF self-attention with a global token → similarities to frozen species-name text vectors. Both visual towers train their last two Transformer blocks, final normalization and projection. It replaces the previous frozen-feature experiment while retaining the original CLIP baseline.

It is not an exact reproduction of the CoCa-based paper. Differences include CLIP ViT-B/32 instead of CoCa, 58 existing source-derived descriptors rather than a new LLM attribute-generation stage, top-two weak detector proposals, partial visual fine-tuning, a small residual fusion gate, an original-feature preservation objective and a small reused dataset. No external API key is required.

## Installed environment

- Project: `E:\ELEC4240\SpeciesRecognition`
- Interpreter: `.venv\Scripts\python.exe` (Python 3.12)
- GPU: RTX 5060 Laptop, 8 GB
- PyTorch 2.11.0+cu128; torchvision 0.26.0+cu128
- OpenCLIP 3.2.0; transformers 4.57.6; huggingface-hub 0.36.2
- CLIP: original OpenAI ViT-B/32, existing local hash-verified checkpoint
- Detector: `google/owlvit-base-patch32`, revision `cbc355fb364588351c5d51c7f74465e8e7ec6f72`
- All data, downloaded weights, training checkpoints and caches remain under the E-drive project.

The venv uses existing PyTorch from the machine's base course environment. Keep that base environment in place. The installed snapshot of OWL-ViT is loaded with `local_files_only=True`; inference and training need no network once downloads exist. This runner requires CUDA; it is not a CPU fallback implementation.

## Fixed experiment

`configs/grounded_v1.json` is the versioned protocol. `prepare_grounded.py` derives 700 SHA-verified unique image records from the existing 1,000-image expanded dataset:

| Role | Species | Photos per species | Total |
|---|---:|---:|---:|
| Training, seen | 10 | 30 | 300 |
| Development, seen | same 10 | 10 | 100 |
| Development, unseen | separate 4 | 20 | 80 |
| Final evaluation, seen | same 10 | 10 | 100 |
| Final evaluation, unseen | separate 6 | 20 | 120 |

The earlier experiment trained with 200 images; this version uses 300. There are still 20 species, comprising insects and plants. Images were used during previous exploratory experiments, so this is not a new independent benchmark. “Unseen” refers to this adaptation's training classes, not a guarantee about CLIP pretraining. Candidate class names and the fixed attribute vocabulary are semantic side information. Unseen-class photos never enter the training loss.

Four variants each use two visual learning rates (`1e-5`, `3e-6`) and seed 42. Every run receives 120 optimizer updates. Microbatch 2, effective batch 16, 224-pixel input, maximum two detected crops. Fusion parameters learn at ten times the visual rate. AdamW has weight decay 0.01 and cosine decay to 10% of the starting rate. Inputs are deterministic native CLIP crops without stochastic augmentation for exact gradient replay.

| Variant | Visual towers | Region text | Auxiliary region loss |
|---|---|---|---|
| `finetune` | Whole image | None | None |
| `region_only` | Whole image + independent crops | Zero vectors | None |
| `ag` | Whole image + independent crops | Matched detector attribute | Confidence-weighted cosine |
| `shuffled` | Same as AG | Fixed derangement of attribute identities | Same wrong targets |

All regional variants use identical attribute-driven detector boxes. The region-only control removes the downstream text vectors, not the semantic influence of the detector. The shuffled control jointly changes fusion text and alignment targets, so it does not isolate those two mechanisms separately.

Inference-time zero/permuted-text probes act on fusion inputs only. They do not erase attribute information already learned by the visual encoder through the training alignment loss. Interpret these probes together with the independently trained region-only and shuffled controls.

Loss = 0.5 × seen-class cross-entropy + 0.5 × symmetric multi-positive image/text contrastive loss + 0.5 × original-feature cosine preservation + 0.1 × region/text cosine loss (last term for AG and shuffled only). Duplicate class-name texts are positives rather than false negatives. The preservation target is the original pretrained global image embedding, not an earlier task-trained model.

Two-pass gradient caching first obtains all 16 embeddings without retaining visual graphs, differentiates the batch-level objective, then replays each microbatch and propagates its embedding gradient. This preserves cross-microbatch contrastive comparisons. BF16, activation checkpointing and frozen early blocks limit memory. The detector is unloaded before CLIP training/inference.

Checkpoints at updates 0/40/80/120 are selected using development GZSL harmonic mean H over 10 seen + 4 development-unseen candidates, with mean U/S and then cross-entropy as tie breakers. Learning rate is selected independently for each variant. Step zero is eligible and is explicitly flagged if selected. All choices are written to `runs/grounded_v1/selection_locked.json` before final-image detection and evaluation. No final-set calibration or parameter selection is performed.

ZSL uses six unseen candidates and 120 images. GZSL uses sixteen candidates and 220 images; S and U are mean per-class seen/unseen accuracies, H = 2SU/(S+U). They answer different questions and must not be mixed. The original CLIP reference uses the same image preprocessing, candidate text bank and numerical precision. A shared logit scale of 20 is used initially; it does not change argmax accuracy.

## Run and verify

Double-click `grounded_models.cmd`, or use:

```powershell
cd E:\ELEC4240\SpeciesRecognition
.\.venv\Scripts\python.exe prepare_grounded.py --audit
.\.venv\Scripts\python.exe verify_grounded.py smoke
.\.venv\Scripts\python.exe grounded_experiment.py all
.\.venv\Scripts\python.exe verify_grounded.py deployment
.\.venv\Scripts\python.exe report_grounded.py
```

Completed training cells are skipped only after provenance matches. An interrupted cell is retrained from its fixed seed; optimizer-state mid-cell resume is not implemented. Do not launch duplicate workflows. Changing the protocol/model implementation requires a new experiment version; checkpoint/source hashes intentionally reject mixed definitions. Training input pixels can be cached, but trainable visual features are recomputed from images at every update.

The smoke check verifies gradient-cache equivalence against a retained full graph, nonzero gradients and actual updates in both visual branches and CAF, no early-layer changes, rejection of unseen training labels, duplicate-positive loss, exact no-region fallback, the real effective-batch memory budget and original teacher-feature consistency. Small batch-kernel numerical differences are bounded using coordinate and cosine tolerances.

The deployment check reloads all four selected checkpoints, runs a real photo through localization and prediction, and compares logits and labels with saved evaluation predictions. It does not use the photo's true label to select attributes or boxes. Prediction:

Single-photo inference repeats the image to the validated microbatch size of two and discards the duplicate output. This keeps the BF16 GPU kernel shape consistent with evaluation; it is not test-time voting or augmentation.

```powershell
.\.venv\Scripts\python.exe predict_grounded.py "E:\path\photo.jpg"
.\.venv\Scripts\python.exe predict_grounded.py "E:\path\photo.jpg" --candidates gzsl --output work\prediction.json
```

Default prediction covers all 20 configured species. `gzsl` restricts to the fixed 16 evaluation candidates; `unseen` to the fixed six unseen candidates. Candidate selection defines a task and must not depend on the unknown true label. The printed scores are uncalibrated softmax similarities. Input is only an image, but fixed internal text vectors and the attribute detector remain deployment dependencies.

## Outputs and limitations

- `reports/grounded_v1/smoke.json`: GPU, gradient and memory checks.
- `work/grounded_v1/grounding_audit_*.jpg`: fixed 20-training-photo visual audit, local only.
- `runs/grounded_v1/*/history.json`: training loss and development trajectories.
- `runs/grounded_v1/*/best.pt`: trainable weights and fixed text buffers; frozen weights reconstruct from original CLIP.
- `runs/grounded_v1/selection_locked.json`: sealed selection before final evaluation.
- `reports/grounded_v1/final_results.json`: native and matched control results, including AG inference-time text interventions.
- `reports/grounded_v1/deployment_verification.json`: fresh single-image/checkpoint parity.
- `reports/grounded_v1/comparison.csv`, `comparison.png`, `summary_zh.md`, `summary_en.md`: shareable results.

The 58 prompts are not image-level expert annotations. Visual inspection found correct subjects, coarse entire-insect boxes, background flowers, missing regions and wrong attribute descriptions. OWL-ViT confidence is only a weighting heuristic. Missing regions use the global path exactly. Detection success is not evidence of biological attribute correctness.

Only one seed is used in this starter matrix. Paired within-class bootstrap intervals quantify image resampling uncertainty conditional on these classes and this trained seed; they do not establish robustness across training seeds or datasets. Compare AG with original CLIP, plain fine-tuning and matched region/shuffle controls before attributing improvements to semantic attributes.

Code/configuration/results are tracked in the private GitHub repository. Dataset photos, audit photos, weights and caches are excluded. Rebuilding on another PC requires the authorized source data and matching manifests/weights, not just a Git clone. The old partial iNaturalist archive is not a complete verified download; individual used JPEG hashes are checked.

References: [OpenCLIP](https://github.com/mlfoundations/open_clip), [OWL-ViT model](https://huggingface.co/google/owlvit-base-patch32), [iNaturalist 2021](https://github.com/visipedia/inat_comp/tree/master/2021), [AG-CLIP paper DOI](https://doi.org/10.1109/OJCS.2026.3654171).
