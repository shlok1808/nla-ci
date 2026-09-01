---
artifact: tab_interval_methods
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
  csv: "ec6f6a7e4aa95e2c917c50985a148034b5b2df52fca24dcf2a1d0468521b8952"
  tex: "fa28bf5c9533fa52a8835d476c13100018471c6189b464c967604d0ed097d74d"
population: analysis_216
population_note: "Same 216 analysis population; widths are for ROC-AUC."
metric: ['interval width (ROC-AUC)']
uncertainty: "Compares three interval constructions: v1's repeated-split percentile band, Hanley analytic 95%, and stratified bootstrap 95%."
supports_claim: "The repeated-cross-validation percentile band understates sampling uncertainty because repeats share the same cases; it measures split variability only."
must_not_claim:
  - "that the bootstrap interval is a significance test — it is conditional on the fitted models and centred on the observed estimate; testing is by permutation"
  - "that Hanley and the bootstrap should agree exactly — Hanley assumes one fixed scoring rule on independent cases, which an average of per-fold AUCs from refit models violates; it is reported as an approximation"
caveats:
  - "v1's band is retained in this table only to quantify how narrow it was; it must never be presented as a competing confidence interval."
  - "No interval or p-value here covers label uncertainty. All 216 analysis-population labels are single-judge and `provisional_unverified`; the same model family drafted the references and applied the rubric, and the judge prompt asserted the references were human-verified when 216/258 were not (audit §2.5.1). Attenuation is the likely direction, but systematic reference error could manufacture structure and is unquantified."
paper_location: "Appendix (methods)."
caption: "Width of three interval constructions for the same ROC-AUC point estimates. The repeated-split band reported previously is 1.5–2.5x narrower than either sampling-uncertainty interval."
replaceable_by_later_step: false
replacement_risk: "None; documents a completed methods correction."
promotion_conditions:
  - "Appendix only; no promotion path expected."
supersedes: null
superseded_by: null
---
# tab_interval_methods

**Status: `candidate_appendix` — provisional.** Nothing in this package is final; later pipeline steps may replace or modify it.

## What this shows

The repeated-cross-validation percentile band understates sampling uncertainty because repeats share the same cases; it measures split variability only.

## What it must NOT be read as

- Do not claim that the bootstrap interval is a significance test — it is conditional on the fitted models and centred on the observed estimate; testing is by permutation.
- Do not claim that Hanley and the bootstrap should agree exactly — Hanley assumes one fixed scoring rule on independent cases, which an average of per-fold AUCs from refit models violates; it is reported as an approximation.

## Caveats

- v1's band is retained in this table only to quantify how narrow it was; it must never be presented as a competing confidence interval.
- No interval or p-value here covers label uncertainty. All 216 analysis-population labels are single-judge and `provisional_unverified`; the same model family drafted the references and applied the rubric, and the judge prompt asserted the references were human-verified when 216/258 were not (audit §2.5.1). Attenuation is the likely direction, but systematic reference error could manufacture structure and is unquantified.

## How to regenerate

```
python3 scripts/paper_step01_build_f.py
python3 scripts/validate_paper_step01_f.py
```

## Suggested caption

> Width of three interval constructions for the same ROC-AUC point estimates. The repeated-split band reported previously is 1.5–2.5x narrower than either sampling-uncertainty interval.

## Paper placement

Appendix (methods).

**Replaceable by a later step:** False. None; documents a completed methods correction.

**Promotion conditions:**

- Appendix only; no promotion path expected.

## Rendered table

| contrast | v1 split band width | Hanley 95% width | bootstrap 95% width | Hanley / split | bootstrap / split |
|---|---|---|---|---|---|
| broad breach vs none | 0.104 | 0.185 | 0.237 | 1.77x | 2.28x |
| substantive leak vs rest | 0.133 | 0.174 | 0.181 | 1.31x | 1.36x |
| substantive leak vs appropriate | 0.152 | 0.185 | 0.233 | 1.22x | 1.53x |
| limiting vs direct (all) | 0.100 | 0.148 | 0.154 | 1.47x | 1.53x |
| limiting vs direct (disclosers only) | 0.094 | 0.153 | 0.168 | 1.63x | 1.78x |
| broad-only vs substantive | 0.108 | 0.181 | 0.195 | 1.68x | 1.81x |

### Table notes

1. Widths are for ROC-AUC in the 216 analysis population.
2. The v1 band is the 2.5/97.5 percentile over 20 repeated cross-validation splits of the SAME cases. It measures split variability, not sampling uncertainty, and must never be presented as a confidence interval (Nadeau & Bengio 2003; Bates, Hastie & Tibshirani 2023).
3. Hanley & McNeil (1982) assumes a single fixed scoring rule applied to independent cases; an average of per-fold AUCs from models refit on each training split violates that, so it is reported as an approximation alongside the bootstrap.
4. The bootstrap is a stratified prediction-resampling interval (n_boot=1000), conditional on the fitted models; it is not a test against the null.
