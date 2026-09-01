---
artifact: fig_protocol_correction_v1_v2
kind: figure
formats: [pdf, svg, png]
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
plotdata:
  path: fig_protocol_correction_v1_v2.plotdata.json
  sha256: "82ae649cc33092e21f1733f9fba27b74fd0a11a778d4aff03a96dd6fe7c5fdbe"
artifact_sha256:
  pdf: "32cc622b9c2fc58efe26fabcb2f17245a71b8567a6d767a3aa1fee33006acb11"
  svg: "053d38798d9a40379f5a447cf3b1e494f98646983579bc18c65666f0c39939fb"
  png: "7a1a0d5c4897ceb4064868c60bee6e425c49d2e246e21db46f5fc37827e90445"
population: analysis_216
population_note: "Same 216 analysis population under both protocols."
metric: ['roc_auc', 'pr_auc', 'interval width']
uncertainty: "Panel A shows point estimates under each protocol. Panel B compares interval widths by construction method and is itself the uncertainty result."
supports_claim: "Two of the four statistical corrections, shown directly: fold-wise metric computation raises every contrast, and the previously reported band was split variability rather than sampling uncertainty."
must_not_claim:
  - "that all four corrections are visible here — the matched permutation null and the Holm adjustment are not visualisable and live in the correction table"
  - "that the v1 band is a confidence interval — it is a repeated-split percentile range on the same cases"
  - "that rising numbers validate the new protocol — the correction is justified by the fold-boundary rank-corruption mechanism, not by its direction"
caveats:
  - "v1 and v2 differ only in the summary statistic; data, contrasts, seeds, folds and C grid are identical."
  - "Panel B's Hanley width is an approximation: Hanley assumes a single fixed scoring rule on independent cases, which an average of per-fold AUCs from refit models violates."
paper_location: "Appendix (methods). The most durable Step-1 figure."
caption: "Effect of the statistical protocol correction. (A) Pooling predict_proba scores across cross-validation folds that each select their own regularisation strength (v1) corrupts the pooled ranking and biases AUC downward; computing each metric within its held-out fold and averaging (v2) raises every contrast. Identical data, contrasts, seeds, folds and regularisation grid; only the statistic differs. (B) The interval reported under v1 was the 2.5/97.5 percentile over 20 repeated splits of the same cases — split variability, not sampling uncertainty. Hanley analytic and stratified bootstrap 95% intervals are substantially wider."
replaceable_by_later_step: false
replacement_risk: "None. Later steps adopt this protocol rather than superseding it."
promotion_conditions:
  - "Promote to main text if the paper centres the measurement-validity contribution; otherwise appendix."
supersedes: null
superseded_by: null
---
# fig_protocol_correction_v1_v2

**Status: `candidate_appendix` — provisional.** Nothing in this package is final; later pipeline steps may replace or modify it.

## What this shows

Two of the four statistical corrections, shown directly: fold-wise metric computation raises every contrast, and the previously reported band was split variability rather than sampling uncertainty.

## What it must NOT be read as

- Do not claim that all four corrections are visible here — the matched permutation null and the Holm adjustment are not visualisable and live in the correction table.
- Do not claim that the v1 band is a confidence interval — it is a repeated-split percentile range on the same cases.
- Do not claim that rising numbers validate the new protocol — the correction is justified by the fold-boundary rank-corruption mechanism, not by its direction.

## Caveats

- v1 and v2 differ only in the summary statistic; data, contrasts, seeds, folds and C grid are identical.
- Panel B's Hanley width is an approximation: Hanley assumes a single fixed scoring rule on independent cases, which an average of per-fold AUCs from refit models violates.

## How to regenerate

```
python3 scripts/paper_step01_build_f.py
python3 scripts/validate_paper_step01_f.py
```

## Suggested caption

> Effect of the statistical protocol correction. (A) Pooling predict_proba scores across cross-validation folds that each select their own regularisation strength (v1) corrupts the pooled ranking and biases AUC downward; computing each metric within its held-out fold and averaging (v2) raises every contrast. Identical data, contrasts, seeds, folds and regularisation grid; only the statistic differs. (B) The interval reported under v1 was the 2.5/97.5 percentile over 20 repeated splits of the same cases — split variability, not sampling uncertainty. Hanley analytic and stratified bootstrap 95% intervals are substantially wider.

## Paper placement

Appendix (methods). The most durable Step-1 figure.

**Replaceable by a later step:** False. None. Later steps adopt this protocol rather than superseding it.

**Promotion conditions:**

- Promote to main text if the paper centres the measurement-validity contribution; otherwise appendix.
