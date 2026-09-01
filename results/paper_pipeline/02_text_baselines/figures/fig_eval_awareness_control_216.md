---
artifact: fig_eval_awareness_control_216
kind: figure
formats: [pdf, svg, png]
status: candidate_appendix
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
plotdata:
  path: fig_eval_awareness_control_216.plotdata.json
  sha256: "73c7bcecfc73a9cce9891bb6d5c8f7f9a627714dfdcd352f924bf58302df976c"
artifact_sha256:
  pdf: "fa8559ae89ab0fde414288cd0582d98320cc72d71459c59c21c6b6d6cdca669a"
  svg: "04e796e1bbeb1d2e6e1631d2b495d4084c851c2c39ab1c7fffe1828eb831279f"
  png: "20726e08a962df51b0d4ff463747fd0e6e95d55e33d5377cae1361b6217872c2"
population: analysis_216
population_note: "216 canonical tier-3 analysis cases; contrast-specific subsets 81-216. The 42 calibration cases are excluded."
metric: ["roc_auc", "privileged delta (activation - stronger text baseline)"]
uncertainty: "Paired scenario bootstrap (B=1000) resampling scenarios once per draw and recomputing the full repeat-averaged per-fold statistic for both channels; one-sided p = P(delta* <= 0), Holm-adjusted across the six contrasts."
supports_claim: "The headline result survives erasure of evaluation-awareness, with a manipulation check demonstrating the erasure worked."
must_not_claim:
  - "that rank-1 erasure removes a concept in full"
  - "that a null drop proves privacy-specificity — it rules out one alternative explanation"
caveats:
  - "Isotropic random directions cost ~0.0000 AUC and are shown only to demonstrate they are an inadequate null."
  - "All 216 analysis-population labels are single-judge and provisional_unverified; the judge prompt asserted human verification when 216/258 references were not verified. No interval here covers that uncertainty."
paper_location: "Appendix, or main text beside the headline result."
caption: "Eval-awareness erasure control. (a) AUC lost by each contrast when the cross-fitted eval-awareness direction is erased, against the ±95th percentile band of 100 permuted-frame null directions. (b) Manipulation check: the same erasure drives test-vs-deploy decodability from 1.000 to chance."
replaceable_by_later_step: false
replacement_risk: "None."
promotion_conditions:
  - "Promote to main text if the caution objection is addressed inline rather than in an appendix."
supersedes: null
superseded_by: null
---
# fig_eval_awareness_control_216

**Status: `candidate_appendix` — provisional.**

## What this shows

The headline result survives erasure of evaluation-awareness, with a manipulation check demonstrating the erasure worked.

## What it must NOT be read as

- Do not claim that rank-1 erasure removes a concept in full.
- Do not claim that a null drop proves privacy-specificity — it rules out one alternative explanation.

## Caveats

- Isotropic random directions cost ~0.0000 AUC and are shown only to demonstrate they are an inadequate null.
- All 216 analysis-population labels are single-judge and provisional_unverified; the judge prompt asserted human verification when 216/258 references were not verified. No interval here covers that uncertainty.

## How to regenerate

```
python3 scripts/text_baselines_canonical_f.py
python3 scripts/eval_awareness_canonical_f.py
python3 scripts/paper_step02_build_f.py
```

## Suggested caption

> Eval-awareness erasure control. (a) AUC lost by each contrast when the cross-fitted eval-awareness direction is erased, against the ±95th percentile band of 100 permuted-frame null directions. (b) Manipulation check: the same erasure drives test-vs-deploy decodability from 1.000 to chance.

## Paper placement

Appendix, or main text beside the headline result.

**Replaceable by a later step:** False. None.

**Promotion conditions:**

- Promote to main text if the caution objection is addressed inline rather than in an appendix.
