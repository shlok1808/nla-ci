---
artifact: tab_primary_contrasts_216
kind: table
formats: [csv, tex]
status: candidate_main
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
  csv: "bdb415aa3cc684e7f9b0d9e140d621165387a6bcb1fc5db771e68e1ce191d33d"
  tex: "1258568252a52941eea8fd8cc5dbc2bf300ccc0fd4fa3ba283a3378320f70858"
population: analysis_216
population_note: "216 canonical tier-3 analysis cases; contrast-specific subsets range 81–216 and are printed per row. Excludes the 42 calibration cases."
metric: ['pr_auc (primary)', 'roc_auc (supporting)']
uncertainty: "Hanley & McNeil (1982) analytic 95% CI and stratified prediction-resampling bootstrap 95% CI (n_boot=1000). Testing is by label permutation (n_perm=500) through the identical per-fold statistic, Holm-adjusted across the 6 scored contrasts within the population."
supports_claim: "Under corrected statistics, four of six pre-specified behavioural contrasts are decodable above chance from layer-20 activations at the final prompt token; disclosure degree is not."
must_not_claim:
  - "that any contrast is more decodable than any other — no between-contrast test was performed and none is powered; only leak_vs_appropriate vs degree_boundary have non-overlapping marginal intervals, which is not a test, and the samples are dependent"
  - "that substantive_leak is decodable — its ROC interval excludes chance but it fails the primary Holm-adjusted PR test (p=.13) and is reported suggestive"
  - "that PR-AUC above class prevalence indicates signal — the cross-validated PR null exceeds prevalence by ~0.05–0.06 at these n; the permutation null is the correct baseline"
  - "that the model uses this information causally — these are correlational decoding results at a single extraction point (layer 20, final prompt token)"
caveats:
  - "No interval or p-value here covers label uncertainty. All 216 analysis-population labels are single-judge and `provisional_unverified`; the same model family drafted the references and applied the rubric, and the judge prompt asserted the references were human-verified when 216/258 were not (audit §2.5.1). Attenuation is the likely direction, but systematic reference error could manufacture structure and is unquantified."
  - "The six contrasts are not independent samples: leak_vs_appropriate (n=81) and degree_boundary (n=188) share all 57 substantive-leak cases, and limiting_among_disclosers (n=188) is nested inside limiting_vs_direct (n=216). Holm is applied across the six scored contrasts within the population; under this dependence it is conservative-to-unclear rather than exact."
  - "Four of six contrasts select C at the grid floor (1e-7). Below that floor the AUC is flat and equals a standardised difference-of-means probe, so the optimum being outside the grid is moot (audit §2.4) — but it does mean the decodable signal is a class-mean direction, not a high-capacity decision boundary."
  - "The limiting construct is 'discloses and then limits', not deflection or refusal. In the 216 population 72 of 76 limiting cases are `mixed_disclose_then_limit`, 4 are `soft_deflection`, and there are zero `explicit_refusal`. The old deflection construct collapsed (36 → 7 cases) and is unmeasurable here."
  - "leak_vs_appropriate is a subsetted population (n=81): it drops 131 broad_only and 4 refused cases, so its .704 prevalence is an artifact of subsetting, and its 24 negatives are all provisional-labelled."
paper_location: "Main text, results table."
caption: "Linear-probe decodability of six behavioural contrasts from layer-20 residual activations at the final prompt token (analysis population, n=216 scenarios; contrast-specific subsets as listed). PR-AUC is the pre-specified primary metric; its baseline is the empirical permutation-null mean, which exceeds class prevalence under cross-validation. p-values are label-permutation (n_perm=500) through the identical per-fold statistic, Holm-adjusted across the six contrasts."
replaceable_by_later_step: true
replacement_risk: "Step 2 adds a matched text baseline and a privileged Δ = activation − text; the paper's headline quantity may become Δ rather than absolute AUC, in which case this table gains Δ columns or is superseded by a Δ table."
promotion_conditions:
  - "Remains the main results table unless Step 2's Δ supersedes absolute AUC as the headline quantity."
  - "Would be DEMOTED if the Step-2 text baseline shows the privileged increment is near zero for the contrasts presented here, or if the blind label audit (pipeline step 11) changes labels enough to move any verdict."
supersedes: null
superseded_by: null
---
# tab_primary_contrasts_216

**Status: `candidate_main` — provisional.** Nothing in this package is final; later pipeline steps may replace or modify it.

## What this shows

Under corrected statistics, four of six pre-specified behavioural contrasts are decodable above chance from layer-20 activations at the final prompt token; disclosure degree is not.

## What it must NOT be read as

- Do not claim that any contrast is more decodable than any other — no between-contrast test was performed and none is powered; only leak_vs_appropriate vs degree_boundary have non-overlapping marginal intervals, which is not a test, and the samples are dependent.
- Do not claim that substantive_leak is decodable — its ROC interval excludes chance but it fails the primary Holm-adjusted PR test (p=.13) and is reported suggestive.
- Do not claim that PR-AUC above class prevalence indicates signal — the cross-validated PR null exceeds prevalence by ~0.05–0.06 at these n; the permutation null is the correct baseline.
- Do not claim that the model uses this information causally — these are correlational decoding results at a single extraction point (layer 20, final prompt token).

## Caveats

- No interval or p-value here covers label uncertainty. All 216 analysis-population labels are single-judge and `provisional_unverified`; the same model family drafted the references and applied the rubric, and the judge prompt asserted the references were human-verified when 216/258 were not (audit §2.5.1). Attenuation is the likely direction, but systematic reference error could manufacture structure and is unquantified.
- The six contrasts are not independent samples: leak_vs_appropriate (n=81) and degree_boundary (n=188) share all 57 substantive-leak cases, and limiting_among_disclosers (n=188) is nested inside limiting_vs_direct (n=216). Holm is applied across the six scored contrasts within the population; under this dependence it is conservative-to-unclear rather than exact.
- Four of six contrasts select C at the grid floor (1e-7). Below that floor the AUC is flat and equals a standardised difference-of-means probe, so the optimum being outside the grid is moot (audit §2.4) — but it does mean the decodable signal is a class-mean direction, not a high-capacity decision boundary.
- The limiting construct is 'discloses and then limits', not deflection or refusal. In the 216 population 72 of 76 limiting cases are `mixed_disclose_then_limit`, 4 are `soft_deflection`, and there are zero `explicit_refusal`. The old deflection construct collapsed (36 → 7 cases) and is unmeasurable here.
- leak_vs_appropriate is a subsetted population (n=81): it drops 131 broad_only and 4 refused cases, so its .704 prevalence is an artifact of subsetting, and its 24 negatives are all provisional-labelled.

## How to regenerate

```
python3 scripts/paper_step01_build_f.py
python3 scripts/validate_paper_step01_f.py
```

## Suggested caption

> Linear-probe decodability of six behavioural contrasts from layer-20 residual activations at the final prompt token (analysis population, n=216 scenarios; contrast-specific subsets as listed). PR-AUC is the pre-specified primary metric; its baseline is the empirical permutation-null mean, which exceeds class prevalence under cross-validation. p-values are label-permutation (n_perm=500) through the identical per-fold statistic, Holm-adjusted across the six contrasts.

## Paper placement

Main text, results table.

**Replaceable by a later step:** True. Step 2 adds a matched text baseline and a privileged Δ = activation − text; the paper's headline quantity may become Δ rather than absolute AUC, in which case this table gains Δ columns or is superseded by a Δ table.

**Promotion conditions:**

- Remains the main results table unless Step 2's Δ supersedes absolute AUC as the headline quantity.
- Would be DEMOTED if the Step-2 text baseline shows the privileged increment is near zero for the contrasts presented here, or if the blind label audit (pipeline step 11) changes labels enough to move any verdict.

## Rendered table

| contrast | family | n | n_pos | prevalence | ROC-AUC | ROC Hanley 95% CI | ROC boot 95% CI | PR-AUC | PR null mean | PR excess over null | PR boot 95% CI | Holm p (PR) | Holm p (ROC) | modal C | C at grid floor | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| broad breach vs none | disclosure presence | 216 | 188 | 0.870 | 0.698 | [0.606, 0.791] | [0.573, 0.811] | 0.936 | 0.880 | 0.056 | [0.905, 0.965] | .030 | ≤.012 | 0.0001 | no | supported |
| substantive leak vs rest | disclosure presence | 216 | 57 | 0.264 | 0.639 | [0.552, 0.726] | [0.543, 0.724] | 0.403 | 0.324 | 0.079 | [0.346, 0.549] | .132 | ≤.012 | 1e-07 | yes | suggestive |
| substantive leak vs appropriate | disclosure presence | 81 | 57 | 0.704 | 0.814 | [0.721, 0.906] | [0.684, 0.917] | 0.911 | 0.753 | 0.157 | [0.853, 0.967] | ≤.012 | ≤.012 | 1e-07 | yes | supported |
| limiting vs direct (all) | response strategy | 216 | 76 | 0.352 | 0.724 | [0.650, 0.798] | [0.648, 0.801] | 0.603 | 0.402 | 0.202 | [0.538, 0.714] | ≤.012 | ≤.012 | 0.0001 | no | supported |
| limiting vs direct (disclosers only) | response strategy | 188 | 72 | 0.383 | 0.728 | [0.652, 0.805] | [0.642, 0.810] | 0.631 | 0.436 | 0.196 | [0.566, 0.745] | ≤.012 | ≤.012 | 0.0001 | no | supported |
| broad-only vs substantive | disclosure degree | 188 | 57 | 0.303 | 0.575 | [0.485, 0.666] | [0.473, 0.668] | 0.390 | 0.365 | 0.025 | [0.345, 0.518] | .309 | .122 | 1e-07 | yes | no_evidence |

### Table notes

1. PR-AUC is the primary metric, pre-specified in the analysis script docstring (committed before the corrected run); it is not registered in docs/PREREGISTRATION.md. ROC-AUC is supporting evidence and never rescues a PR-failing contrast.
2. Verdict: supported = Holm-adjusted permutation p on PR-AUC <= .05; suggestive = fails that but Holm-adjusted p on ROC-AUC <= .05; no_evidence = neither.
3. p-values are label-permutation with n_perm=500 through the identical per-fold statistic, so the smallest attainable Holm-adjusted value is 6/501 = 0.012; entries at that floor are shown as <=.012.
4. The PR baseline is the empirical permutation-null mean, not class prevalence: cross-validated average precision is upward-biased relative to prevalence by ~0.05-0.06 at these sample sizes.
5. The six contrasts are separate tests on overlapping case sets and are not independent; no between-contrast comparison was performed and none is powered.
6. Intervals reflect sampling uncertainty conditional on the labels only. All 216 analysis labels are single-judge and provisional.
