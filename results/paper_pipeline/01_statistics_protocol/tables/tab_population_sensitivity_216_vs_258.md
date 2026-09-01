---
artifact: tab_population_sensitivity_216_vs_258
kind: table
formats: [csv, tex]
status: sensitivity
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
artifact_sha256:
  csv: "e37c1d767fc61ed6b58a96e32d0a804a41c6b50f539b137277059bfec2ab2086"
  tex: "4f387224cd7164d62fcccabb55aa9deeb72238df45cdc7febff5cb3a69329831"
population: analysis_216 (primary) and all_258 (superset, sensitivity only)
population_note: "258 ⊃ 216. The 42 additional cases are calibration cases with human-verified references and a markedly different behaviour composition. This is not a replication and the 258 column is never the primary estimate."
metric: ['roc_auc', 'pr_auc', 'Holm-adjusted permutation p (PR)']
uncertainty: "As the primary table, computed independently within each population."
supports_claim: "The direction and ordering of results is stable when the 42 calibration cases are added, with uniformly higher point estimates that are consistent with the calibration set's enrichment and cleaner references."
must_not_claim:
  - "that the 258 results replicate the 216 results — 258 is a strict superset containing every 216 case, so the two are not independent samples"
  - "that the higher 258 estimates are the better estimates — the added cases are enriched for limiting and leaking behaviour and are the only human-verified references, so composition and label quality are both confounded with population"
caveats:
  - "Composition differs sharply: the 42 calibration cases oversample the old `refused` class by design, so limiting and leak rates are far above the analysis population's."
  - "No interval or p-value here covers label uncertainty. All 216 analysis-population labels are single-judge and `provisional_unverified`; the same model family drafted the references and applied the rubric, and the judge prompt asserted the references were human-verified when 216/258 were not (audit §2.5.1). Attenuation is the likely direction, but systematic reference error could manufacture structure and is unquantified."
paper_location: "Appendix (sensitivity analysis)."
caption: "Sensitivity of every contrast to including the 42 calibration cases. The 258 population is a superset of the 216 analysis population, enriched for limiting and disclosing behaviour and differently verified; it is reported as a sensitivity analysis only."
replaceable_by_later_step: true
replacement_risk: "If the calibration references are ever extended to the analysis population (human verification of all 216), this comparison is superseded by a single verified-label analysis."
promotion_conditions:
  - "Never promoted to a primary result. Would move to main text only as an explicit robustness paragraph."
supersedes: null
superseded_by: null
---
# tab_population_sensitivity_216_vs_258

**Status: `sensitivity` — provisional.** Nothing in this package is final; later pipeline steps may replace or modify it.

## What this shows

The direction and ordering of results is stable when the 42 calibration cases are added, with uniformly higher point estimates that are consistent with the calibration set's enrichment and cleaner references.

## What it must NOT be read as

- Do not claim that the 258 results replicate the 216 results — 258 is a strict superset containing every 216 case, so the two are not independent samples.
- Do not claim that the higher 258 estimates are the better estimates — the added cases are enriched for limiting and leaking behaviour and are the only human-verified references, so composition and label quality are both confounded with population.

## Caveats

- Composition differs sharply: the 42 calibration cases oversample the old `refused` class by design, so limiting and leak rates are far above the analysis population's.
- No interval or p-value here covers label uncertainty. All 216 analysis-population labels are single-judge and `provisional_unverified`; the same model family drafted the references and applied the rubric, and the judge prompt asserted the references were human-verified when 216/258 were not (audit §2.5.1). Attenuation is the likely direction, but systematic reference error could manufacture structure and is unquantified.

## How to regenerate

```
python3 scripts/paper_step01_build_f.py
python3 scripts/validate_paper_step01_f.py
```

## Suggested caption

> Sensitivity of every contrast to including the 42 calibration cases. The 258 population is a superset of the 216 analysis population, enriched for limiting and disclosing behaviour and differently verified; it is reported as a sensitivity analysis only.

## Paper placement

Appendix (sensitivity analysis).

**Replaceable by a later step:** True. If the calibration references are ever extended to the analysis population (human verification of all 216), this comparison is superseded by a single verified-label analysis.

**Promotion conditions:**

- Never promoted to a primary result. Would move to main text only as an explicit robustness paragraph.

## Rendered tables

**Population composition (258 ⊃ 216)**

| row | n | % limiting | % substantive leak | % broad breach | reference verification |
|---|---|---|---|---|---|
| analysis 216 (primary) | 216 | 35.2% | 26.4% | 87.0% | provisional_unverified=216 |
| +42 calibration cases | 42 | 66.7% | 61.9% | 85.7% | human_verified=42 |
| all 258 (superset) | 258 | 40.3% | 32.2% | 86.8% | human_verified=42, provisional_unverified=216 |

**Results**

| contrast | n (216) | ROC (216) | PR (216) | Holm p PR (216) | verdict (216) | n (258) | ROC (258) | PR (258) | Holm p PR (258) | verdict (258) |
|---|---|---|---|---|---|---|---|---|---|---|
| broad breach vs none | 216 | 0.698 | 0.936 | .030 | supported | 258 | 0.684 | 0.928 | .018 | supported |
| substantive leak vs rest | 216 | 0.639 | 0.403 | .132 | suggestive | 258 | 0.639 | 0.463 | .036 | supported |
| substantive leak vs appropriate | 81 | 0.814 | 0.911 | ≤.012 | supported | 110 | 0.838 | 0.937 | ≤.012 | supported |
| limiting vs direct (all) | 216 | 0.724 | 0.603 | ≤.012 | supported | 258 | 0.774 | 0.712 | ≤.012 | supported |
| limiting vs direct (disclosers only) | 188 | 0.728 | 0.631 | ≤.012 | supported | 224 | 0.776 | 0.728 | ≤.012 | supported |
| broad-only vs substantive | 188 | 0.575 | 0.390 | .309 | no_evidence | 224 | 0.615 | 0.501 | .036 | supported |

### Table notes

1. 258 is a strict SUPERSET of 216: every analysis case also appears in the 258 population. These are not independent samples and this is not a replication.
2. The 42 additional cases are calibration cases that deliberately oversampled the historical `refused` class, so they are enriched for limiting and disclosing behaviour relative to the analysis population.
3. The 42 calibration cases are also the only cases with human-verified references; all 216 analysis references are provisional and single-judge. Composition and label quality are therefore both confounded with population.
4. The 216 analysis population is primary for every claim. The 258 column is a sensitivity analysis and is never the primary estimate.
