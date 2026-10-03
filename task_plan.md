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

# Three-stage five-backbone comparison (2026-10-02)

User goal: compare pure visual classification, supervised image-text adaptation, and attribute-guided adaptation for the original five backbones.

Finish line: matched frozen-feature visual baselines on the exact expanded20 splits/shots/seeds; verify and reuse existing text/AG cells; preserve validation-only choices; report all methods and paired uncertainty, test implementation, publish code/results to the now-public authorized repository. No accuracy-gain requirement.

## Current phases
- [x] Inspect existing code, artifacts, and previous training budget.
- [x] Lock protocol and implement/test two visual controls.
- [x] Complete 90 new training cells and lock choices before evaluation.
- [x] Replay existing 225 cells, evaluate controls, compare three stages.
- [x] Report results/limitations, verify and publish.

Protocol intent: reuse 20 classes/1,000 images; frozen backbones; shots5/10/20; seeds42/43/44; same sampled images, LR1e-3, 200+200 update/checkpoint schedule. Add standard linear visual head and an equal-parameter random-code projection control. CLIP ViT-B/32 visual baseline retains CLIP pretraining and must never be called CLIP-free. Existing test set is reused, so all inference is exploratory. Do not tune to force improvements.

Completion: all 315 cells reported; eight tests passed; plot reviewed; code and bilingual results published to the public repository in 3052dfd. No photos, caches or checkpoints uploaded. Bounded task complete; no additional tuning started.

# Confidence aggregation pilot (2026-10-03)

Goal: matched visual / text / uniform AG / detector-confidence AG comparison, plus random-code, region-only and shuffled-confidence controls. Use ten grounded seen classes, 500 existing images (300 train pool,100 val,100 historically used test);20 shots;EfficientNet-B0 and CLIP ViT-B/32;three seeds. Frozen features;200+200 updates;same initial alignment and identical AG modules. No post-result tuning or gain requirement.

- [x] Inspect available detector outputs; all ten-class images already grounded.
- [x] Implement and test pooling, extract frozen features and lock protocol.
- [x] Complete training, seal validation selections, evaluate all arms.
- [x] Audit and prepare bilingual results with limitations; Git delivery is the final step.

Use root planning files (resolver returned empty). Preserve previous byte-hashed implementations. Comparison is a new closed-set lightweight pilot, not the previous twenty-class scores or an original AG-CLIP replication. No new detector/model downloads required.

Completed42 final heads and prediction replay;five tests passed. Uniform/confidence/shuffled accuracy tied for both backbones. Test100-image coverage:25 zero regions,34 one,41 two. Weight interventions affect probabilities but no class predictions. Bounded experiment complete; no additional model search or post-result tuning.

# Fresh-data confidence validation (2026-10-03)

User goal: replace reused images to reduce adaptive evaluation overfitting. Keep the prior confidence model/training recipe fixed. Select ten previously unused CUB species from the80 not represented in previous project manifests/catalogs;500 fresh pictures with20 train/10 validation/20 test per species. Respect official train/test split; exact/decoded/perceptual duplicate exclusions against historical images and across new splits. Use existing fixed24 bird attributes,not selected on new scores. Two backbones,three seeds,seven arms. No pretrained-overlap guarantee.

- [x] Audit previous usage and lock fresh manifest and protocol before feature/model scoring.
- [x] Ground/extract training and validation only;train42 cells and seal all choices.
- [x] Ground/extract held-out test only after seal;one final evaluation and audit.
- [x] Prepare bilingual results and independence limits;authorized GitHub upload is the final delivery step.

Finish line: completed fixed matrix and verified data separation,including negative outcomes. No post-test tuning. New-dataset shift and ten-class scope must be reported. Initial file probe make_expanded.py was absent; used actual CUB metadata. PowerShell rg glob probe failed; resolved via -g.

Complete:42 trained/scored/replayed models;all prior and cross-split exact/perceptual checks passed. New weighted AG seed-mean gain over uniform is +0.50pp EfficientNet,0.00pp CLIP;no positive Holm-significant contrast. No new tuning after results. Finish bounded task after publication.


# FungiTastic temporal validation (2026-10-03)

User authorized download and testing. Select ten species using metadata counts, one photo per observation;2022 train/validation and2023 test,20/10/20 per class if feasible. Two old fixed backbones,three seeds,seven unchanged arms;new fixed fungus visual vocabulary. No score-based subset selection. Verify dates and grouping, lock before feature scoring, seal before test processing. Qualify pretraining overlap protection as metadata-supported post-release photographs,not unseen concepts or an absolute guarantee.

- [x] Download and audit metadata/images and freeze manifest.
- [x] Implement domain configuration and verify grouping/seal.
- [x] Train matched42 models and evaluate once.
- [x] Audit,bilingual report and authorized GitHub delivery.

Completed42 cells,seven tests,prediction replay maxerror0. Temporal grouping and hash-screening audit passed.600 candidates downloaded,500 used;177/200 test images have two detections. Weighted-minus-uniform seed means0/+0.33pp,ensemble+0.5/-1.5pp;zero positive Holm-significant contrasts. Bilingual report/plot reviewed;Git publication is final delivery step.
