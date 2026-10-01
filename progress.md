# Progress

2026-10-01: Read planning-with-files and codex-ralph-loop-skill. Repository initially clean. Resolved no existing named plan and initialized root plan. Inspected frozen-cache/readout implementation. Current phase: predeclare new local-only training and guarded-fusion controls. No new scores inspected, no prior experiment altered.

Implemented local_attribute_v1: 18 fresh subset-only heads; same 144 attributes across five visual pooling arms plus 86-global-attribute complement. Three semantic permutations and native-calibration control. Fixed plain/guarded fusion grids; guard uses native margin <=0.5, seen penalties, fit-only U/H constraints, and 1% unseen new-error budget. Three unit tests passed (removed harmless scalar-conversion warning afterward). Next lock hashes before training and evaluation.

Protocol locked before training; all 18 readouts and 20 fusion/control arms finished successfully. Repaired-local guarded cross-fit H79.492, U77.733; pure local plain H77.898 U72.133. No grid changes. Next audit held-out damage, paired comparisons, and score reconstruction.

Verification complete: 9 tests passed (3 local,3 fusion,3 prior path). All20 cross-fit arms/60 seed cells replayed and error counts matched. Real-image pipeline checked on deliberately selected unseen correction image9244: native44 -> fused50, matching cached prediction; native score max delta0.008112, local delta0.0000508. This is a plumbing check, not new accuracy evidence. Exported fixed75-candidate exploratory inference policy. Generated full tables, all predictions, conditional bootstrap, and reviewed plot. Remaining: commit/push code/results; no photos, features, or checkpoints staged.
