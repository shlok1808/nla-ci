---
artifact: tab_eval_awareness_control_216
kind: table
formats: [csv, tex]
status: candidate_main
maturity: provisional
generated_utc: "2026-09-01T17:40:25+00:00"
generated_by:
  script: scripts/paper_step02_build_f.py
  script_sha256: "aec7f9b90047c3eb321013838bec307a6ed3d17202a3291afcf7a1b177502dde"
  command: "python3 scripts/paper_step02_build_f.py"
  git_commit: "b6f476e8f0b8503fd630029a39b2f93114efa8d3"
  git_dirty: true
sources:
  - {path: results/text_baselines_canonical_f.csv, sha256: "41d9414e62c31322c69e9bc9356ea089780026338cc166dbb7745af31f4e45be", role: text_baseline_results}
  - {path: results/text_baselines_canonical_f.json, sha256: "166682e234543cbc6a882d06e2ccd58b2bd1fd5578c3752a8d60850149da1676", role: text_baseline_protocol}
  - {path: results/text_baselines_canonical_f_dissociation.csv, sha256: "5a4b3c9e1a9419382beba60cbd2e74785954f7c326d7b57093da2c88ef803e01", role: dissociation_test}
  - {path: results/eval_awareness_canonical_f.csv, sha256: "cd695f9e26cba0fff09de8199be655fd331284ea9264affe10cb9801a3b76ec0", role: eval_awareness_control}
  - {path: results/eval_awareness_canonical_f.json, sha256: "72d0e22c6ee49efe183dbbaa419cc42443a07f06f8fee62aa3f32d389db8c4d8", role: eval_awareness_protocol}
  - {path: results/onset_alignment_f.json, sha256: "4d283e9d094139a42db69af7a4282567f7720b78ee079cf52694bd13d84b06bc", role: step3_alignment_preflight}
  - {path: results/behavior_labels_tier3_canonical_f.csv, sha256: "40c5fb2b10aa49d535b6593ba9e0bcbe6ad631dfe21c8237845924cba3705979", role: population_definition}
artifact_sha256:
  csv: "fdac5adbbf8442aaadaa3d2857f45740913abcbf8134adbe413ad420f6b712d5"
  tex: "a3f99bb075a0b36687c3c5430ab578c5b8ac8fb7df399f3ecf9f5b9df7455f92"
population: analysis_216
population_note: "216 canonical tier-3 analysis cases; contrast-specific subsets 81-216. The 42 calibration cases are excluded."
metric: ["roc_auc", "privileged delta (activation - stronger text baseline)"]
uncertainty: "Paired scenario bootstrap (B=1000) resampling scenarios once per draw and recomputing the full repeat-averaged per-fold statistic for both channels; one-sided p = P(delta* <= 0), Holm-adjusted across the six contrasts."
supports_claim: "The limiting signal is not evaluation-awareness: erasing the eval-awareness direction costs 0.0017 AUC with an interval spanning zero, while the same erasure drives test-vs-deploy decodability from 1.000 to chance."
must_not_claim:
  - "that all caution-like or eval-related structure has been removed — rank-1 erasure removes one direction, and a distributed residual cannot be fully excluded"
  - "that the erasure control validates the limiting result itself — it rules out one alternative explanation"
caveats:
  - "The eval-awareness direction is estimated from framed prompts of the same scenarios; cross-fitting removes the transductive concern but the framing manipulation is our own construct."
  - "All 216 analysis-population labels are single-judge and provisional_unverified; the judge prompt asserted human verification when 216/258 references were not verified. No interval here covers that uncertainty."
paper_location: "Main text or appendix, as the control for the headline result."
caption: "Eval-awareness erasure control with its manipulation check (analysis population)."
replaceable_by_later_step: false
replacement_risk: "None."
promotion_conditions:
  - "Main text if the caution objection is addressed inline."
supersedes: null
superseded_by: null
---
# tab_eval_awareness_control_216

**Status: `candidate_main` — provisional.**

## What this shows

The limiting signal is not evaluation-awareness: erasing the eval-awareness direction costs 0.0017 AUC with an interval spanning zero, while the same erasure drives test-vs-deploy decodability from 1.000 to chance.

## What it must NOT be read as

- Do not claim that all caution-like or eval-related structure has been removed — rank-1 erasure removes one direction, and a distributed residual cannot be fully excluded.
- Do not claim that the erasure control validates the limiting result itself — it rules out one alternative explanation.

## Caveats

- The eval-awareness direction is estimated from framed prompts of the same scenarios; cross-fitting removes the transductive concern but the framing manipulation is our own construct.
- All 216 analysis-population labels are single-judge and provisional_unverified; the judge prompt asserted human verification when 216/258 references were not verified. No interval here covers that uncertainty.

## How to regenerate

```
python3 scripts/text_baselines_canonical_f.py
python3 scripts/eval_awareness_canonical_f.py
python3 scripts/paper_step02_build_f.py
```

## Suggested caption

> Eval-awareness erasure control with its manipulation check (analysis population).

## Paper placement

Main text or appendix, as the control for the headline result.

**Replaceable by a later step:** False. None.

**Promotion conditions:**

- Main text if the caution objection is addressed inline.

## Rendered table

| contrast | n | AUC original | AUC after erasure | drop | drop 95% CI | null drop mean | null drop p95 | isotropic drop | p vs null |
|---|---|---|---|---|---|---|---|---|---|
| limiting vs direct (disclosers only) | 188 | 0.737 | 0.736 | +0.0017 | [-0.0015, +0.0054] | +0.0004 | +0.0028 | +0.0000 | .178 |
| limiting vs direct (all) | 216 | 0.736 | 0.736 | +0.0002 | [-0.0020, +0.0026] | -0.0003 | +0.0014 | +0.0001 | .257 |
| substantive leak vs appropriate | 81 | 0.813 | 0.814 | -0.0007 | [-0.0038, +0.0000] | +0.0010 | +0.0044 | +0.0000 | .851 |
| substantive leak vs rest | 216 | 0.640 | 0.639 | +0.0009 | [-0.0020, +0.0043] | -0.0002 | +0.0019 | +0.0001 | .129 |

### Table notes

1. MANIPULATION CHECK: test-vs-deploy decodability on the framed activations is 1.000 before erasure and 0.500 after. The erasure demonstrably removes the concept, so a surviving limiting probe is a genuine null rather than a failed manipulation.
2. Cross-fitted: v_eval and every null direction are re-derived inside each training fold from framed activations of TRAINING scenarios only. A direction estimated on all data would over-remove on held-out data and bias toward a false negative.
3. The null is 100 side-permuted paired directions from the same estimation pipeline (within each scenario, which 3 of its 6 frames count as 'test' is reassigned at random). Isotropic random directions are reported only to show they are an inadequate null: erasing one costs ~0.0000 AUC, because a random vector in 3584 dimensions is nearly orthogonal to the activation manifold.
4. cos(v_eval, v_limiting) = -0.093, inside the permutation-null band (|cos| 95th percentile 0.152).
5. Rank-1 erasure removes a direction. Eval-awareness collapsing to chance indicates that direction carried essentially all of it here, but a distributed residual cannot be fully excluded.
