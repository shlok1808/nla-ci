---
artifact: tab_dissociation_216
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
  csv: "2b45ea5d816449e5cb7c1ddab453590a6c3377cbef836f5f73e6c959fb472e16"
  tex: "c2a35708b53f4b87ab61814958d9877b99cd528c3b8714ab3c5844f00a23a5e6"
population: analysis_216
population_note: "216 canonical tier-3 analysis cases; contrast-specific subsets 81-216. The 42 calibration cases are excluded."
metric: ["roc_auc", "privileged delta (activation - stronger text baseline)"]
uncertainty: "Paired scenario bootstrap (B=1000) resampling scenarios once per draw and recomputing the full repeat-averaged per-fold statistic for both channels; one-sided p = P(delta* <= 0), Holm-adjusted across the six contrasts."
supports_claim: "The privileged increment is larger for strategy than for disclosure outcome, tested directly rather than inferred from differing significance (+0.130, CI [+0.023, +0.242], p = .010 against substantive_leak)."
must_not_claim:
  - "that every limiting-vs-leak comparison is established — the comparison against leak_vs_appropriate has an interval that grazes zero and is suggestive only"
  - "that the leak contrasts have zero privileged signal — the point estimates are positive but not separable from zero"
caveats:
  - "All 216 analysis-population labels are single-judge and provisional_unverified; the judge prompt asserted human verification when 216/258 references were not verified. No interval here covers that uncertainty."
  - "The four comparisons are a coherent set on overlapping case sets, not independent tests; no multiplicity adjustment is applied across them."
paper_location: "Main text, alongside the primary table."
caption: "Direct test of the strategy-vs-outcome dissociation in privileged increment (analysis population)."
replaceable_by_later_step: false
replacement_risk: "None; a direct test of an already-computed quantity."
promotion_conditions:
  - "Keep in main text while the dissociation is claimed."
supersedes: null
superseded_by: null
---
# tab_dissociation_216

**Status: `candidate_main` — provisional.**

## What this shows

The privileged increment is larger for strategy than for disclosure outcome, tested directly rather than inferred from differing significance (+0.130, CI [+0.023, +0.242], p = .010 against substantive_leak).

## What it must NOT be read as

- Do not claim that every limiting-vs-leak comparison is established — the comparison against leak_vs_appropriate has an interval that grazes zero and is suggestive only.
- Do not claim that the leak contrasts have zero privileged signal — the point estimates are positive but not separable from zero.

## Caveats

- All 216 analysis-population labels are single-judge and provisional_unverified; the judge prompt asserted human verification when 216/258 references were not verified. No interval here covers that uncertainty.
- The four comparisons are a coherent set on overlapping case sets, not independent tests; no multiplicity adjustment is applied across them.

## How to regenerate

```
python3 scripts/text_baselines_canonical_f.py
python3 scripts/eval_awareness_canonical_f.py
python3 scripts/paper_step02_build_f.py
```

## Suggested caption

> Direct test of the strategy-vs-outcome dissociation in privileged increment (analysis population).

## Paper placement

Main text, alongside the primary table.

**Replaceable by a later step:** False. None; a direct test of an already-computed quantity.

**Promotion conditions:**

- Keep in main text while the dissociation is claimed.

## Rendered table

| limiting contrast | leak contrast | Δ limiting | Δ leak | difference | 95% CI | p |
|---|---|---|---|---|---|---|
| limiting vs direct (disclosers only) | substantive leak vs appropriate | +0.162 | +0.057 | +0.104 | [-0.017, 0.220] | .045 |
| limiting vs direct (disclosers only) | substantive leak vs rest | +0.162 | +0.028 | +0.130 | [0.023, 0.242] | .010 |
| limiting vs direct (all) | substantive leak vs appropriate | +0.125 | +0.057 | +0.069 | [-0.055, 0.181] | .144 |
| limiting vs direct (all) | substantive leak vs rest | +0.125 | +0.028 | +0.096 | [-0.010, 0.207] | .042 |

### Table notes

1. A difference between a significant and a non-significant result is not itself significant (Gelman & Stern 2006, The American Statistician 60(4)). This table tests the difference of the two deltas directly.
2. Draws are aligned by index across contrasts (same seed and draw order), so the comparison is paired by construction.
3. The comparison against substantive_leak is supported; the comparison against leak_vs_appropriate has an interval that grazes zero and is reported as suggestive. No multiplicity adjustment is applied across these four comparisons — they are reported as a coherent set, not as independent tests.
