---
artifact: fig_privileged_delta_216
kind: figure
formats: [pdf, svg, png]
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
plotdata:
  path: fig_privileged_delta_216.plotdata.json
  sha256: "5660a7e62a84a28124f95c920d39ee4d010b4224885c8d7178e24a1f4bb17ab0"
artifact_sha256:
  pdf: "e1c85193ff5b46caab55b5db63cefe4b952b299a8cb46f15598f952134e41aab"
  svg: "53ee34ea6442b12ddbf019061e9adbdc5a26b70b730748eb82139b0180955eba"
  png: "657f6a3202db5786ad1063d734d563c0131afbc9575ba5222c83819a53cba923"
population: analysis_216
population_note: "216 canonical tier-3 analysis cases; contrast-specific subsets 81-216. The 42 calibration cases are excluded."
metric: ["roc_auc", "privileged delta (activation - stronger text baseline)"]
uncertainty: "Paired scenario bootstrap (B=1000) resampling scenarios once per draw and recomputing the full repeat-averaged per-fold statistic for both channels; one-sided p = P(delta* <= 0), Holm-adjusted across the six contrasts."
supports_claim: "The paper's headline: pre-response activations predict upcoming disclosure strategy beyond what the scenario wording supports, and this is specific to strategy."
must_not_claim:
  - "that the activation contains information absent from the prompt — the final-prompt-token activation is a deterministic function of the prompt, so a positive delta means the model's processing makes behaviour-relevant information linearly EXTRACTABLE where text classifiers cannot extract it, not that new information exists"
  - "that no text baseline could close the gap — delta is defined relative to a baseline family (TF-IDF and a frozen MiniLM encoder); a stronger encoder can only shrink it"
  - "that between-contrast differences other than those in the dissociation table have been tested"
caveats:
  - "All 216 analysis-population labels are single-judge and provisional_unverified; the judge prompt asserted human verification when 216/258 references were not verified. No interval here covers that uncertainty."
  - "Panel (a) shows both text channels; Δ in panel (b) is against the stronger of them, which is the encoder on every contrast."
paper_location: "Main text, Figure 1."
caption: "Layer-20 activations at the final prompt token versus matched scenario-text baselines for six behavioural contrasts (analysis population, n=216). (a) Channel AUCs: filled circle = activation probe, open square = frozen sentence-encoder baseline, grey tick = TF-IDF. (b) Privileged increment Δ = activation − stronger text baseline, with paired scenario-bootstrap 95% CI; filled markers survive Holm correction across the six contrasts."
replaceable_by_later_step: true
replacement_risk: "Step 3's onset-resolved figure may become Figure 1 if the dynamics result is strong; this would then move to Figure 2."
promotion_conditions:
  - "Main figure unless Step 3 supersedes it."
supersedes: null
superseded_by: null
---
# fig_privileged_delta_216

**Status: `candidate_main` — provisional.**

## What this shows

The paper's headline: pre-response activations predict upcoming disclosure strategy beyond what the scenario wording supports, and this is specific to strategy.

## What it must NOT be read as

- Do not claim that the activation contains information absent from the prompt — the final-prompt-token activation is a deterministic function of the prompt, so a positive delta means the model's processing makes behaviour-relevant information linearly EXTRACTABLE where text classifiers cannot extract it, not that new information exists.
- Do not claim that no text baseline could close the gap — delta is defined relative to a baseline family (TF-IDF and a frozen MiniLM encoder); a stronger encoder can only shrink it.
- Do not claim that between-contrast differences other than those in the dissociation table have been tested.

## Caveats

- All 216 analysis-population labels are single-judge and provisional_unverified; the judge prompt asserted human verification when 216/258 references were not verified. No interval here covers that uncertainty.
- Panel (a) shows both text channels; Δ in panel (b) is against the stronger of them, which is the encoder on every contrast.

## How to regenerate

```
python3 scripts/text_baselines_canonical_f.py
python3 scripts/eval_awareness_canonical_f.py
python3 scripts/paper_step02_build_f.py
```

## Suggested caption

> Layer-20 activations at the final prompt token versus matched scenario-text baselines for six behavioural contrasts (analysis population, n=216). (a) Channel AUCs: filled circle = activation probe, open square = frozen sentence-encoder baseline, grey tick = TF-IDF. (b) Privileged increment Δ = activation − stronger text baseline, with paired scenario-bootstrap 95% CI; filled markers survive Holm correction across the six contrasts.

## Paper placement

Main text, Figure 1.

**Replaceable by a later step:** True. Step 3's onset-resolved figure may become Figure 1 if the dynamics result is strong; this would then move to Figure 2.

**Promotion conditions:**

- Main figure unless Step 3 supersedes it.
