# Excluded from scientific backbone/AG comparisons

This run retained capitalized species names. The local fast Gemma tokenizer did not apply the `do_lower_case: true` setting in the downloaded SigLIP 2 tokenizer configuration. Development-only inference audit showed native H increasing from 9.70% to 75.46% with otherwise identical lowercase text. This is a text preprocessing defect, not evidence that SigLIP 2 is weak at species recognition.

All 30 training cells and raw metrics are retained for audit, but must not be used as a valid model comparison or favorable low-accuracy baseline. Corrected text, fresh probe/fusion fits and all matched controls are in `cub_siglip_v2`. Visual features and crop masks are unchanged and can be reused; v2 has independent text/checkpoint provenance. No AG benefit may be attributed to this preprocessing correction.
