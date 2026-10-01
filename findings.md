# Findings

- Prior fusion: repaired branch H 77.85 vs native SigLIP2 75.46, but U 70.93 vs 72.40; repaired branch introduced 7-8 unseen errors per seed and net lost 3-4 unseen correct images.
- Previous readout jointly trains 230 attributes (144 anatomically mapped, 86 whole-image); post-hoc slicing alone cannot isolate training dependence. New readouts must train solely on each declared subset.
- Fixed localizers and dense feature cache exist. Matching the same 144 attributes on whole-image, uniform, native-part, repaired-part, and wrong-part features controls attribute dimensionality.
- Pure local means no explicit global feature or global-attribute scoring path. Transformer patch features remain contextual; it does not mean isolated image pixels.
- Cross-fitted selection on reused development data cannot establish independent generalization or statistical significance.

2026-10-01 locked experiment completed: all 18 subset-only readouts and 20 fusion/control arms. Pure repaired local readout H ~50, better than native/wrong/uniform/whole-image versions using the same 144 attributes. Cross-fit repaired/plain H77.898 U72.133; repaired/guarded H79.492 U77.733. Native-local/guarded H79.346 and wrong-local/guarded H79.001 are close; local semantic permutations ~75.0-75.5 and native-calibration guard 75.465. Promising guarded local evidence, but repair-specific fusion benefit remains small. Need error-count replay and uncertainty before interpreting.

Audit complete: guarded-local unseen corrected/new errors = 15/2,14/1,15/1 vs plain-local 5/5,3/4,3/4. Seen S falls from plain84.67 to guarded81.33, still above baseline78.80; seen new errors increase to6-8. H gain vs baseline +4.03 conditional bootstrap[+1.69,+6.46]; gain vs native localization +0.15[-0.33,+0.63], vs wrong part +0.49[-0.43,+1.49]. No independent-data significance claim. All folds and all-development inference policy select lambda0.2, seen penalty0.25, fixed margin0.5. Figure reviewed and legible. All20 arms remain in report, including whole-global plain H79.76 (U72.93).
