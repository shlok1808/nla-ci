---
artifact: fig_contrast_effects_dual_metric_216
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
plotdata:
  path: fig_contrast_effects_dual_metric_216.plotdata.json
  sha256: "ac103eddc60d857cf83dc01cb01e445b3e3668b6aeab853de232860888b24876"
artifact_sha256:
  pdf: "a98c75ae3fcbbfca69941de6d001c50cb2d640573f8e0de1866787587426f79a"
  svg: "83835eabdb0c253fbc81530437300f03a9f4a3d808e27319f9398efc9fc9832f"
  png: "0fc02c3d5ae5fac599972d894f66d6a73e4ebbdc78081f9a0b932170f0d87fa5"
population: analysis_216
population_note: "216 canonical tier-3 analysis cases; contrast-specific subsets 81–216, printed in each row label."
metric: ['roc_auc (left panel)', 'pr_auc (right panel, primary)']
uncertainty: "Left: Hanley 95% CI (thick) and stratified bootstrap 95% CI (thin, offset). Right: bootstrap 95% CI, against the empirical permutation-null mean. Marker fill encodes the Holm-adjusted permutation verdict."
supports_claim: "Four contrasts are decodable above chance on the primary metric; the strategy contrasts and the leak-vs-appropriate contrast carry the largest excess over their permutation nulls, while disclosure degree shows none."
must_not_claim:
  - "that any contrast is more decodable than any other — no between-contrast test was performed and none is powered; only leak_vs_appropriate vs degree_boundary have non-overlapping marginal intervals, which is not a test, and the samples are dependent"
  - "that the ROC panel establishes significance — substantive_leak's ROC interval excludes chance while it fails the primary PR test; the PR panel and the marker fill carry the verdict"
  - "that the prevalence tick is the PR chance level — it is not; the open square (permutation null) is"
  - "that the model uses this information causally — these are correlational decoding results at a single extraction point (layer 20, final prompt token)"
caveats:
  - "No interval or p-value here covers label uncertainty. All 216 analysis-population labels are single-judge and `provisional_unverified`; the same model family drafted the references and applied the rubric, and the judge prompt asserted the references were human-verified when 216/258 were not (audit §2.5.1). Attenuation is the likely direction, but systematic reference error could manufacture structure and is unquantified."
  - "The six contrasts are not independent samples: leak_vs_appropriate (n=81) and degree_boundary (n=188) share all 57 substantive-leak cases, and limiting_among_disclosers (n=188) is nested inside limiting_vs_direct (n=216). Holm is applied across the six scored contrasts within the population; under this dependence it is conservative-to-unclear rather than exact."
  - "Four of six contrasts select C at the grid floor (1e-7). Below that floor the AUC is flat and equals a standardised difference-of-means probe, so the optimum being outside the grid is moot (audit §2.4) — but it does mean the decodable signal is a class-mean direction, not a high-capacity decision boundary."
  - "The limiting construct is 'discloses and then limits', not deflection or refusal. In the 216 population 72 of 76 limiting cases are `mixed_disclose_then_limit`, 4 are `soft_deflection`, and there are zero `explicit_refusal`. The old deflection construct collapsed (36 → 7 cases) and is unmeasurable here."
paper_location: "Appendix, companion to the primary results table. Not a main-text candidate while Step 2 is pending."
caption: "Linear-probe decodability of six behavioural contrasts from layer-20 activations at the final prompt token (analysis population, n=216; contrast-specific subsets as labelled). Left: ROC-AUC with Hanley 95% CI (thick) and stratified bootstrap 95% CI (thin, offset below); dotted line marks chance, open ticks the ROC permutation null. Right: PR-AUC (primary metric) with bootstrap 95% CI; the open square is the empirical permutation-null mean — the correct baseline, which exceeds class prevalence (faint tick) by 0.05–0.06 at these sample sizes. Marker fill: filled = survives Holm adjustment on PR-AUC; open with centre dot = survives on ROC only; open = neither. The contrasts are separate tests on overlapping case sets; intervals are marginal and no between-contrast difference was tested."
replaceable_by_later_step: true
replacement_risk: "High. Step 2's activation-minus-text Δ figure presents the same contrasts with a stronger claim (privileged information beyond the visible text) and would likely take this figure's place."
promotion_conditions:
  - "Promote to candidate_main only if Step 2's Δ is null or unusable, leaving absolute decodability as the strongest available result."
  - "Requires the Step-2 text baseline recomputed under the v2 per-fold protocol before any Δ annotation could be added to it."
supersedes: null
superseded_by: null
---
# fig_contrast_effects_dual_metric_216

**Status: `candidate_appendix` — provisional.** Nothing in this package is final; later pipeline steps may replace or modify it.

## What this shows

Four contrasts are decodable above chance on the primary metric; the strategy contrasts and the leak-vs-appropriate contrast carry the largest excess over their permutation nulls, while disclosure degree shows none.

## What it must NOT be read as

- Do not claim that any contrast is more decodable than any other — no between-contrast test was performed and none is powered; only leak_vs_appropriate vs degree_boundary have non-overlapping marginal intervals, which is not a test, and the samples are dependent.
- Do not claim that the ROC panel establishes significance — substantive_leak's ROC interval excludes chance while it fails the primary PR test; the PR panel and the marker fill carry the verdict.
- Do not claim that the prevalence tick is the PR chance level — it is not; the open square (permutation null) is.
- Do not claim that the model uses this information causally — these are correlational decoding results at a single extraction point (layer 20, final prompt token).

## Caveats

- No interval or p-value here covers label uncertainty. All 216 analysis-population labels are single-judge and `provisional_unverified`; the same model family drafted the references and applied the rubric, and the judge prompt asserted the references were human-verified when 216/258 were not (audit §2.5.1). Attenuation is the likely direction, but systematic reference error could manufacture structure and is unquantified.
- The six contrasts are not independent samples: leak_vs_appropriate (n=81) and degree_boundary (n=188) share all 57 substantive-leak cases, and limiting_among_disclosers (n=188) is nested inside limiting_vs_direct (n=216). Holm is applied across the six scored contrasts within the population; under this dependence it is conservative-to-unclear rather than exact.
- Four of six contrasts select C at the grid floor (1e-7). Below that floor the AUC is flat and equals a standardised difference-of-means probe, so the optimum being outside the grid is moot (audit §2.4) — but it does mean the decodable signal is a class-mean direction, not a high-capacity decision boundary.
- The limiting construct is 'discloses and then limits', not deflection or refusal. In the 216 population 72 of 76 limiting cases are `mixed_disclose_then_limit`, 4 are `soft_deflection`, and there are zero `explicit_refusal`. The old deflection construct collapsed (36 → 7 cases) and is unmeasurable here.

## How to regenerate

```
python3 scripts/paper_step01_build_f.py
python3 scripts/validate_paper_step01_f.py
```

## Suggested caption

> Linear-probe decodability of six behavioural contrasts from layer-20 activations at the final prompt token (analysis population, n=216; contrast-specific subsets as labelled). Left: ROC-AUC with Hanley 95% CI (thick) and stratified bootstrap 95% CI (thin, offset below); dotted line marks chance, open ticks the ROC permutation null. Right: PR-AUC (primary metric) with bootstrap 95% CI; the open square is the empirical permutation-null mean — the correct baseline, which exceeds class prevalence (faint tick) by 0.05–0.06 at these sample sizes. Marker fill: filled = survives Holm adjustment on PR-AUC; open with centre dot = survives on ROC only; open = neither. The contrasts are separate tests on overlapping case sets; intervals are marginal and no between-contrast difference was tested.

## Paper placement

Appendix, companion to the primary results table. Not a main-text candidate while Step 2 is pending.

**Replaceable by a later step:** True. High. Step 2's activation-minus-text Δ figure presents the same contrasts with a stronger claim (privileged information beyond the visible text) and would likely take this figure's place.

**Promotion conditions:**

- Promote to candidate_main only if Step 2's Δ is null or unusable, leaving absolute decodability as the strongest available result.
- Requires the Step-2 text baseline recomputed under the v2 per-fold protocol before any Δ annotation could be added to it.
