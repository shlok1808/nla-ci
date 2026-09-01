---
artifact: tab_privileged_delta_216
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
  csv: "cd7c83e4a249c109817c32f4a3ce28910c758657ac9ae6ee683ff10312534352"
  tex: "6658cc50f5f7b36934da40a31b13605e1c6934485aa38b289194f270e43161e1"
population: analysis_216
population_note: "216 canonical tier-3 analysis cases; contrast-specific subsets 81-216. The 42 calibration cases are excluded."
metric: ["roc_auc", "privileged delta (activation - stronger text baseline)"]
uncertainty: "Paired scenario bootstrap (B=1000) resampling scenarios once per draw and recomputing the full repeat-averaged per-fold statistic for both channels; one-sided p = P(delta* <= 0), Holm-adjusted across the six contrasts."
supports_claim: "Layer-20 activations at the final prompt token predict upcoming disclosure-limiting strategy substantially better than the scenario text does (Δ +0.162 and +0.125, Holm p ≤ .010); the leak contrasts show no such privileged increment."
must_not_claim:
  - "that the activation contains information absent from the prompt — the final-prompt-token activation is a deterministic function of the prompt, so a positive delta means the model's processing makes behaviour-relevant information linearly EXTRACTABLE where text classifiers cannot extract it, not that new information exists"
  - "that no text baseline could close the gap — delta is defined relative to a baseline family (TF-IDF and a frozen MiniLM encoder); a stronger encoder can only shrink it"
  - "that the leak result is null because leak is undecodable — leak_vs_appropriate is decodable at 0.814, but a frozen encoder recovers 0.757 of it from the prompt alone"
  - "that the model uses this representation causally — these are correlational decoding results"
caveats:
  - "All 216 analysis-population labels are single-judge and provisional_unverified; the judge prompt asserted human verification when 216/258 references were not verified. No interval here covers that uncertainty."
  - "Single model, single layer, single extraction point (Qwen2.5-7B, layer 20, final prompt token)."
  - "The limiting construct is 'discloses then limits': 72 of 76 limiting cases are mixed_disclose_then_limit, zero are explicit refusals."
paper_location: "Main text, primary results table."
caption: "Privileged increment of layer-20 activations over matched scenario-text baselines, analysis population (n=216)."
replaceable_by_later_step: true
replacement_risk: "Step 3 adds onset-relative dynamics; this table remains the static result but may be joined by a position-resolved version."
promotion_conditions:
  - "Primary results table unless the blind label audit moves a verdict."
supersedes: null
superseded_by: null
---
# tab_privileged_delta_216

**Status: `candidate_main` — provisional.**

## What this shows

Layer-20 activations at the final prompt token predict upcoming disclosure-limiting strategy substantially better than the scenario text does (Δ +0.162 and +0.125, Holm p ≤ .010); the leak contrasts show no such privileged increment.

## What it must NOT be read as

- Do not claim that the activation contains information absent from the prompt — the final-prompt-token activation is a deterministic function of the prompt, so a positive delta means the model's processing makes behaviour-relevant information linearly EXTRACTABLE where text classifiers cannot extract it, not that new information exists.
- Do not claim that no text baseline could close the gap — delta is defined relative to a baseline family (TF-IDF and a frozen MiniLM encoder); a stronger encoder can only shrink it.
- Do not claim that the leak result is null because leak is undecodable — leak_vs_appropriate is decodable at 0.814, but a frozen encoder recovers 0.757 of it from the prompt alone.
- Do not claim that the model uses this representation causally — these are correlational decoding results.

## Caveats

- All 216 analysis-population labels are single-judge and provisional_unverified; the judge prompt asserted human verification when 216/258 references were not verified. No interval here covers that uncertainty.
- Single model, single layer, single extraction point (Qwen2.5-7B, layer 20, final prompt token).
- The limiting construct is 'discloses then limits': 72 of 76 limiting cases are mixed_disclose_then_limit, zero are explicit refusals.

## How to regenerate

```
python3 scripts/text_baselines_canonical_f.py
python3 scripts/eval_awareness_canonical_f.py
python3 scripts/paper_step02_build_f.py
```

## Suggested caption

> Privileged increment of layer-20 activations over matched scenario-text baselines, analysis population (n=216).

## Paper placement

Main text, primary results table.

**Replaceable by a later step:** True. Step 3 adds onset-relative dynamics; this table remains the static result but may be joined by a position-resolved version.

**Promotion conditions:**

- Primary results table unless the blind label audit moves a verdict.

## Rendered table

| contrast | n | activation AUC | TF-IDF AUC | embedding AUC | stronger text baseline | privileged Δ | Δ 95% CI | Holm p (Δ) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| broad breach vs none | 216 | 0.698 | 0.636 | 0.665 | embed | +0.033 | [-0.042, 0.112] | .527 | no_evidence |
| substantive leak vs rest | 216 | 0.639 | 0.604 | 0.611 | embed | +0.028 | [-0.048, 0.113] | .527 | no_evidence |
| substantive leak vs appropriate | 81 | 0.814 | 0.700 | 0.757 | embed | +0.057 | [-0.035, 0.147] | .527 | no_evidence |
| limiting vs direct (all) | 216 | 0.724 | 0.561 | 0.600 | embed | +0.125 | [0.049, 0.196] | .010 | supported |
| limiting vs direct (disclosers only) | 188 | 0.728 | 0.557 | 0.567 | embed | +0.162 | [0.083, 0.234] | .006 | supported |
| broad-only vs substantive | 188 | 0.575 | 0.579 | 0.582 | embed | -0.007 | [-0.092, 0.072] | .569 | no_evidence |

### Table notes

1. Δ = activation AUC − text AUC against the STRONGER of the two text baselines per contrast (the frozen sentence encoder won on every contrast); using the weaker baseline would inflate Δ.
2. Text channels see the scenario/prompt only. The generated response is never used: it post-dates the probed state and can trivially reveal the label.
3. All three channels share byte-identical cross-validation folds (same seeds), so the bootstrap is exactly paired.
4. Δ inference is a paired scenario bootstrap that resamples scenarios once per draw and recomputes the full repeat-averaged per-fold statistic for both channels; p is one-sided P(Δ* ≤ 0), Holm-adjusted across the six contrasts.
5. A label-permutation null is deliberately not used for Δ: permuting labels nulls both channels, testing 'no signal anywhere' rather than 'no difference'.
6. All 216 analysis-population labels are single-judge and provisional_unverified; the judge prompt asserted human verification when 216/258 references were not verified. No interval here covers that uncertainty.
