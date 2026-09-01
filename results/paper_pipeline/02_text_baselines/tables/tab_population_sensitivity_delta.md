---
artifact: tab_population_sensitivity_delta
kind: table
formats: [csv, tex]
status: sensitivity
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
  csv: "8bd97b5bbe6bf1b1aec06ea66c3ebb7b34710341e09e8c47dc5a81c3a8ce1518"
  tex: "ec52cb1f604c38225da21115a7e54d62fb883bc352e0b1da2101e838d1427169"
population: analysis_216
population_note: "216 canonical tier-3 analysis cases; contrast-specific subsets 81-216. The 42 calibration cases are excluded."
metric: ["roc_auc", "privileged delta (activation - stronger text baseline)"]
uncertainty: "Paired scenario bootstrap (B=1000) resampling scenarios once per draw and recomputing the full repeat-averaged per-fold statistic for both channels; one-sided p = P(delta* <= 0), Holm-adjusted across the six contrasts."
supports_claim: "The privileged increment is stable in direction and ordering when the 42 calibration cases are added."
must_not_claim:
  - "that the 258 results replicate the 216 results — 258 is a strict superset containing every 216 case"
  - "that the larger 258 estimates are the better estimates"
caveats:
  - "The 42 added cases are enriched for limiting and disclosing behaviour and are the only human-verified references."
  - "All 216 analysis-population labels are single-judge and provisional_unverified; the judge prompt asserted human verification when 216/258 references were not verified. No interval here covers that uncertainty."
paper_location: "Appendix (sensitivity)."
caption: "Sensitivity of the privileged increment to population definition."
replaceable_by_later_step: false
replacement_risk: "None."
promotion_conditions:
  - "Never promoted to a primary estimate."
supersedes: null
superseded_by: null
---
# tab_population_sensitivity_delta

**Status: `sensitivity` — provisional.**

## What this shows

The privileged increment is stable in direction and ordering when the 42 calibration cases are added.

## What it must NOT be read as

- Do not claim that the 258 results replicate the 216 results — 258 is a strict superset containing every 216 case.
- Do not claim that the larger 258 estimates are the better estimates.

## Caveats

- The 42 added cases are enriched for limiting and disclosing behaviour and are the only human-verified references.
- All 216 analysis-population labels are single-judge and provisional_unverified; the judge prompt asserted human verification when 216/258 references were not verified. No interval here covers that uncertainty.

## How to regenerate

```
python3 scripts/text_baselines_canonical_f.py
python3 scripts/eval_awareness_canonical_f.py
python3 scripts/paper_step02_build_f.py
```

## Suggested caption

> Sensitivity of the privileged increment to population definition.

## Paper placement

Appendix (sensitivity).

**Replaceable by a later step:** False. None.

**Promotion conditions:**

- Never promoted to a primary estimate.

## Rendered table

| contrast | Δ (216 primary) | Holm p (216) | Δ (258 superset) | Holm p (258) |
|---|---|---|---|---|
| broad breach vs none | +0.033 | .527 | +0.032 | .430 |
| substantive leak vs rest | +0.028 | .527 | +0.016 | .430 |
| substantive leak vs appropriate | +0.057 | .527 | +0.055 | .232 |
| limiting vs direct (all) | +0.125 | .010 | +0.149 | .006 |
| limiting vs direct (disclosers only) | +0.162 | .006 | +0.177 | .006 |
| broad-only vs substantive | -0.007 | .569 | +0.060 | .232 |

### Table notes

1. 258 ⊃ 216. The 42 additional cases are calibration cases, enriched for limiting and disclosing behaviour and the only human-verified references. This is not a replication and the 258 column is never the primary estimate.
2. The direction and ordering are stable across populations; the 258 estimates are larger, consistent with that enrichment.
