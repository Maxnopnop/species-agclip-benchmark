# Frozen-global attribute-token diagnostics

Completed versions: 5; training cells: 162. Pending: cub_regional_tokens_v1.

No stable attribute-specific classification benefit is established. Attribute prediction, backbone replacement and seen/unseen calibration must be distinguished from AG gains. All controls and step-zero selections are retained below.

The study reuses 20 CUB species, 200 training, 180 development and 320 evaluation images. Evaluation had been inspected in earlier experiments; follow-ups are exploratory. Seeds 42/43/44 vary fusion initialization and batches, not the split or pretrained weights/probe. H is the harmonic mean of seen/unseen per-class accuracy; ZSL uses only six unseen candidates. Native results: cub_tokens_v1: H=75.11%, ZSL=92.50%; cub_b16_v1: H=84.61%, ZSL=95.83%.

Independent tokens, an attribute-only residual, 24 versus 158 attributes, confidence-based sparse tokens, symmetric contrastive loss, a B/16 backbone and regional visual fine-tuning are separate diagnostics. Conditional paired bootstrap intervals in paired_contrasts.json do not account for adaptive model search or all training randomness. Human-attribute oracle results in the coverage diagnostic are not deployable results or information-theoretic upper bounds.

| Experiment | Variant | H, mean ± SD | ZSL, mean ± SD | Selected steps |
|---|---|---:|---:|---|
| cub_tokens_v1 | region_only | 73.26 ± 0.00 | 94.17 ± 0.00 | 160/160/160 |
| cub_tokens_v1 | pooled | 73.26 ± 0.00 | 94.17 ± 0.00 | 160/160/160 |
| cub_tokens_v1 | tokens | 73.26 ± 0.00 | 94.17 ± 0.00 | 160/160/160 |
| cub_tokens_v1 | permuted | 73.26 ± 0.00 | 94.17 ± 0.00 | 160/160/160 |
| cub_tokens_v1 | constant | 73.26 ± 0.00 | 94.17 ± 0.00 | 160/160/160 |
| cub_bottleneck_v1 | tokens | 72.13 ± 5.16 | 93.61 ± 1.92 | 0/40/0 |
| cub_bottleneck_v1 | permuted | 73.12 ± 3.91 | 93.33 ± 0.83 | 0/0/40 |
| cub_bottleneck_v1 | identity | 67.43 ± 2.78 | 95.56 ± 1.92 | 40/40/40 |
| cub_bottleneck_v1 | constant | 75.11 ± 0.00 | 92.50 ± 0.00 | 0/0/0 |
| cub_rich_v1 | region_only | 73.26 ± 0.00 | 94.17 ± 0.00 | 160/160/160 |
| cub_rich_v1 | pooled | 73.07 ± 0.33 | 94.17 ± 0.00 | 160/160/160 |
| cub_rich_v1 | tokens | 73.26 ± 0.00 | 94.17 ± 0.00 | 160/160/160 |
| cub_rich_v1 | permuted | 73.26 ± 0.00 | 94.17 ± 0.00 | 160/160/160 |
| cub_rich_v1 | constant | 73.07 ± 0.33 | 94.17 ± 0.00 | 160/160/160 |
| cub_sparse_v1 | tokens | 73.26 ± 0.00 | 94.17 ± 0.00 | 160/160/160 |
| cub_sparse_v1 | dense_contrast | 73.26 ± 0.00 | 94.17 ± 0.00 | 160/160/160 |
| cub_sparse_v1 | sparse_ce | 73.07 ± 0.33 | 94.17 ± 0.00 | 160/160/160 |
| cub_sparse_v1 | dense_ce | 73.26 ± 0.00 | 94.17 ± 0.00 | 160/160/160 |
| cub_sparse_v1 | region_only | 73.26 ± 0.00 | 94.17 ± 0.00 | 160/160/160 |
| cub_sparse_v1 | pooled | 73.26 ± 0.00 | 94.17 ± 0.00 | 160/160/160 |
| cub_sparse_v1 | permuted | 73.45 ± 0.33 | 94.17 ± 0.00 | 160/160/160 |
| cub_sparse_v1 | constant | 73.45 ± 0.33 | 94.17 ± 0.00 | 160/160/160 |
| cub_b16_v1 | region_only | 83.30 ± 0.29 | 96.67 ± 0.00 | 160/160/160 |
| cub_b16_v1 | pooled | 83.16 ± 0.36 | 96.67 ± 0.00 | 160/160/160 |
| cub_b16_v1 | tokens | 83.49 ± 0.08 | 96.67 ± 0.00 | 160/160/80 |
| cub_b16_v1 | permuted | 83.49 ± 0.08 | 96.67 ± 0.00 | 160/160/80 |
| cub_b16_v1 | constant | 83.49 ± 0.08 | 96.67 ± 0.00 | 160/160/80 |
