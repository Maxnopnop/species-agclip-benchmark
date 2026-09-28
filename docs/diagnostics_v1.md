# Validation-only diagnosis

Protocol: `configs/diagnostics_v1.json`. Results: `reports/diagnostics_v1/README.md`.

The fixed 96-run matrix uses EfficientNet-B0 and CLIP ViT-B/32, 10-shot, three seeds, two head learning rates, frozen versus trainable final visual blocks, and whole-image baseline / matched attribute values / permuted attribute values / zero attribute values. Attention architecture and initialization match between the last three conditions. No auxiliary attribute loss is used in the retraining comparison. The existing auxiliary-loss models are included in post-training interventions, which do not undo the effects of their training supervision.

Matched denotes automatically retrieved source-derived attributes, not an oracle of visible traits. A fixed permutation acts on selected text values while the retrieval keys stay intact; reordering keys and values together would leave retrieval unchanged. The fixed random mapping could be partially learned around, so it does not prove semantics are never useful.

`multimodal/diagnostics.py` caches full-precision activations before the trainable tail, checks exact reconstruction against the full encoder, and excludes test rows. EfficientNet unfreezes its final MBConv stage and final convolution. CLIP unfreezes its final transformer block, post LayerNorm and projection. BatchNorm running statistics and visual stochastic layers stay in evaluation mode for both conditions. Training/inference behavior is otherwise fixed. A nonzero gradient and parameter change are verified for all 48 tail runs; all 48 frozen tails remain unchanged.

Each condition receives 100 updates and the same two-candidate learning-rate budget. Checkpoints and learning rates are selected on validation macro-F1, breaking ties by loss. Reported diagnostic validation scores are optimistic development measurements on the selection set, not a new independent benchmark. Earlier test results are preserved. No significance claim is made.

## Reproduce on this machine

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_diagnostics
.\.venv\Scripts\python.exe audit_attributes.py
.\.venv\Scripts\python.exe diagnose_ag.py
.\.venv\Scripts\python.exe report_diagnostics.py
```

Existing complete cells are reused only when the saved protocol and implementation fingerprint match. Do not modify the training implementation or protocol and resume into the same results directory; use a new version instead. A tensor-to-scalar warning for logging `train_batch_loss` may appear; this occurs after backpropagation and is not a training failure.

Status: `runs/diagnostics_v1/status.json`. Training log: `work/diagnostics_train.log`. Checkpoints and prefix activations stay under ignored `runs/` and `cache/`. `report_diagnostics.py` checks completeness and frozen/trainable-tail invariants before exporting.

The local visual sheets under `work/diagnostics_v1/audit_page_*.jpg` show one seed-selected validation image per species and its center crop. `reports/diagnostics_v1/visual_review.json` records assistant observations with explicit uncertainty; these are not expert labels. Sheets contain dataset photographs and are intentionally excluded from GitHub. Reproducing the scripted audit generates the sheets and retrieved prompts; the qualitative assessment is a separate human-readable record, not an automated accuracy metric.

New confirmatory performance claims require a fresh held-out evaluation after any development changes motivated by these diagnostics.
