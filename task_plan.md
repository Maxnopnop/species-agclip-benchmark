# Local-attribute isolation and unseen-error protection (2026-10-01)

Goal: isolate the contribution of anatomically pooled attribute evidence and test a bounded, inference-available policy for reducing unseen-class prediction damage.

Finish line: complete matched local-only readout training, negative controls, cross-fitted guarded fusion, verified error accounting, a report, and private-repository delivery. A negative result is valid; do not keep tuning to force a gain.

## Phases
- [x] Inspect prior implementation/results and define scope.
- [x] Lock protocol design and implement/test local-only training and prediction guards.
- [x] Train fixed-budget readouts on existing seen training images; evaluate predeclared arms.
- [x] Audit predictions and report local-specific benefit and unseen damage separately.
- [x] Verify and package authorized code/results for delivery (no images/weights).

Constraints: use existing frozen FG-CLIP2 features and part localizers; preserve prior hashed experiments; 3 seeds, fixed 400 updates; reused development data is exploratory, not final validation. No new model/download or test-set tuning. Freeze small weight/calibration grid before scores are examined.

## Loop boundaries
Implement -> test invariants -> lock -> run -> replay/audit -> deliver. Stop when all declared comparisons are reported; do not enlarge grid after inspecting outcomes. Permission is needed only for a genuinely new blocked action, not routine local experiments already authorized.

## Errors
None. Planning resolver returned empty: no existing plan; using legacy project-root planning files.

Completion: bounded scientific work and verification are complete; final Git commit/push follows packaging. No extra search round is planned. Remaining scientific limitation is independent-data validation, explicitly outside this development-only round.
