---
artifact: tab_protocol_correction_v1_v2
kind: table
formats: [csv, tex]
status: candidate_appendix
maturity: provisional
generated_utc: "2026-09-01T15:39:29+00:00"
generated_by:
  script: scripts/paper_step01_build_f.py
  script_sha256: "6492729d8ddd866539a5101c6810e23f638a7ca5e51fb0c97a1071171c4b16ac"
  command: "python3 scripts/paper_step01_build_f.py"
  git_commit: "a926daa8cd6f24f332477ec56513b95524cde0d5"
  git_dirty: true
sources:
  - {path: results/probe_contrasts_canonical_v2_f.csv, sha256: "0888918e971a70e5b1ed7d0854f7feee5089bef20a0ec0d95669a2fcd96f0367", role: primary_results}
  - {path: results/probe_contrasts_canonical_v2_f.json, sha256: "194e1f0e4b0478daa9c9d8c9cf0c6c46ce240baa506ad6a64eb4c9fdf1fc27b5", role: primary_results_protocol}
  - {path: results/behavior_labels_tier3_canonical_f.csv, sha256: "40c5fb2b10aa49d535b6593ba9e0bcbe6ad631dfe21c8237845924cba3705979", role: population_definition}
  - {path: results/probe_contrasts_canonical_f.csv, sha256: "e91bdf0c49b487caa69e51457f6599ed78bbed962580af342f67d8e44c56a328", role: superseded_results_provenance}
artifact_sha256:
  csv: "29d3b8a9f5ea20be7ca26f0a0230a82a99b35b9fc427758fdde74faccdb68cde"
  tex: "8d3edf201c997f38f870daeca9ce8de84aa5e74b70850ec3c2d71c35a0ee8cd1"
population: analysis_216
population_note: "Same 216 analysis population under both protocols."
metric: ['roc_auc', 'pr_auc', 'permutation p']
uncertainty: "Point estimates under each protocol; v1 reported only a repeated-split percentile band (not a confidence interval) and no multiplicity adjustment."
supports_claim: "Pooling predict_proba scores across cross-validation folds that each select their own regularisation strength biases AUC downward; computing each metric within its held-out fold and averaging raises every contrast."
must_not_claim:
  - "that v2 is 'better because the numbers went up' — the direction is a consequence of the specific defect (fold-boundary rank corruption), and the correction was specified before the corrected numbers were seen"
  - "that this figure/table demonstrates all four corrections — the matched permutation null and the Holm adjustment are not visible in these columns"
caveats:
  - "v1 and v2 are exactly comparable: identical data, contrasts, n, n_pos, seeds, fold assignments and C grid. Only the statistic differs."
  - "v1 p-values were computed against a pooled-score null; they are shown for provenance, not as a valid comparison of significance."
paper_location: "Appendix (methods)."
caption: "Effect of the statistical protocol correction on every scored contrast (analysis population). Identical data, contrasts, seeds, folds and regularisation grid; only the summary statistic differs."
replaceable_by_later_step: false
replacement_risk: "None. This documents a completed methods correction and is not affected by later experiments."
promotion_conditions:
  - "Promote to main text only if the paper's framing centres the measurement/protocol-correction contribution."
supersedes: null
superseded_by: null
---
# tab_protocol_correction_v1_v2

**Status: `candidate_appendix` — provisional.** Nothing in this package is final; later pipeline steps may replace or modify it.

## What this shows

Pooling predict_proba scores across cross-validation folds that each select their own regularisation strength biases AUC downward; computing each metric within its held-out fold and averaging raises every contrast.

## What it must NOT be read as

- Do not claim that v2 is 'better because the numbers went up' — the direction is a consequence of the specific defect (fold-boundary rank corruption), and the correction was specified before the corrected numbers were seen.
- Do not claim that this figure/table demonstrates all four corrections — the matched permutation null and the Holm adjustment are not visible in these columns.

## Caveats

- v1 and v2 are exactly comparable: identical data, contrasts, n, n_pos, seeds, fold assignments and C grid. Only the statistic differs.
- v1 p-values were computed against a pooled-score null; they are shown for provenance, not as a valid comparison of significance.

## How to regenerate

```
python3 scripts/paper_step01_build_f.py
python3 scripts/validate_paper_step01_f.py
```

## Suggested caption

> Effect of the statistical protocol correction on every scored contrast (analysis population). Identical data, contrasts, seeds, folds and regularisation grid; only the summary statistic differs.

## Paper placement

Appendix (methods).

**Replaceable by a later step:** False. None. This documents a completed methods correction and is not affected by later experiments.

**Promotion conditions:**

- Promote to main text only if the paper's framing centres the measurement/protocol-correction contribution.

## Rendered table

| contrast | ROC v1 (pooled) | ROC v2 (per-fold) | Δ ROC | PR v1 (pooled) | PR v2 (per-fold) | Δ PR | raw p PR v1 | raw p PR v2 | Holm p PR v2 | Holm p PR v1 |
|---|---|---|---|---|---|---|---|---|---|---|
| broad breach vs none | 0.658 | 0.698 | +0.041 | 0.919 | 0.936 | +0.017 | .034 | .010 | .030 | — |
| substantive leak vs rest | 0.619 | 0.639 | +0.020 | 0.351 | 0.403 | +0.052 | .048 | .066 | .132 | — |
| substantive leak vs appropriate | 0.756 | 0.814 | +0.058 | 0.867 | 0.911 | +0.043 | .002 | .002 | ≤.012 | — |
| limiting vs direct (all) | 0.703 | 0.724 | +0.022 | 0.541 | 0.603 | +0.062 | .002 | .002 | ≤.012 | — |
| limiting vs direct (disclosers only) | 0.714 | 0.728 | +0.015 | 0.574 | 0.631 | +0.057 | .002 | .002 | ≤.012 | — |
| broad-only vs substantive | 0.557 | 0.575 | +0.018 | 0.341 | 0.390 | +0.049 | .234 | .309 | .309 | — |

### Table notes

1. Identical data, contrasts, n, n_pos, seeds, fold assignments and regularisation grid under both protocols; only the summary statistic differs.
2. v1 pooled raw predict_proba scores across outer folds that each selected their own C. Probability scales differ by orders of magnitude across C, so the pooled ranking is corrupted at fold boundaries and the estimate is biased downward (Forman & Scholz 2010; Airola et al. 2011).
3. v1 p-values were computed against a pooled-score null and are shown for provenance only; they are not a valid significance comparison against v2.
4. v1 applied no multiplicity adjustment, hence the em-dash in the Holm column.
5. Corrections not visible in this table: the permutation null now uses the identical per-fold statistic (Ojala & Garriga 2010), and Holm adjustment was added.
