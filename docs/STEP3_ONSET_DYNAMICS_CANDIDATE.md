# Step 3 specification — onset-aligned strategy dynamics (v2, frozen)

**Status:** v2 after independent review (2026-09-01). No Step 3 activation result
has been observed. The commit that adds this file is the registration; do not
edit predictions or decision rules afterwards — append a dated amendment.

## Scientific question

Does Qwen2.5-7B-Instruct's residual stream predict whether it will limit a
disclosure **before** the registered cue boundary — the earlier of the
broad-disclosure onset and the earliest strategy-evidence quote — is reached in
its generated response, beyond what the visible text supports?

The primary contrast is `limiting_among_disclosers` in the 216-case analysis
population. Both classes disclose; the positive class also limits or hedges
(all 72 positives are `mixed_disclose_then_limit`), the negative class engages
directly. The 258-case population is a sensitivity analysis, never a replication.

## What the cutoff is — and is not (amendment to v1)

`cutoff = min(broad_disclosure_onset, earliest strategy_evidence start)`. For
mixed cases the judge's `strategy_evidence` quotes the **disclosure sentence as
well as the limiting cue**, so the second term is *not* a limiting-language
onset. Consequences, fixed here:

- the cutoff is never later than the disclosure onset, so the window is
  pre-disclosure for every scenario (conservative for leakage);
- nothing may call it "limiting-language onset"; v1's "within-limiting
  trajectory" secondary is **withdrawn**;
- the anchor is class-dependent (limiting cutoffs are earlier: median 27 vs 31
  tokens; position alone gives ROC 0.609), so every text baseline carries
  prefix position/length and the registered Δ must clear it;
- limiting prefixes carry *tonal* pre-cues ("I'm really disappointed in you")
  that no hedge regex catches. The regex screen plus human review certifies only
  the absence of unrecorded explicit hedge/refusal language; the matched
  visible-prefix baselines are the leakage protection for everything else.

## Primary estimand and decision rule

- Model `Qwen/Qwen2.5-7B-Instruct`, bf16 teacher-forced forward pass over
  prompt + stored response, exact benchmark chat template with
  `add_generation_prompt=True`.
- Layer: reported layer 20 = decoder block index 19 output = HF
  `hidden_states[20]`, identical to Steps 1–2.
- Position semantics: the state stored at response index r is the residual
  state after reading token r, i.e. the state that emits token r+1.
  `offset_-1` emits the onset token; `prompt_final` emits the first response
  token. Window: offsets `[-8, -1]`.
- Population: analysis disclosers whose response covers the whole window
  (`cutoff_tok >= 8`), fixed across offsets: **n = 186 (70 limiting / 116
  direct)**; two limiting cases (cutoff 0 and 2) are excluded and listed.
- Activation statistic: mean over the eight offsets of the repeat-averaged
  per-fold ROC-AUC (StratifiedKFold(5) × 20 repeats, nested inner 3-fold C grid
  scored by average precision, balanced L2 logistic, liblinear; seeds
  20260901+r; folds byte-identical across channels and cells).
- Registered text family (as in Step 2): TF-IDF 1–2-grams on scenario + visible
  prefix, and frozen `all-MiniLM-L6-v2` embeddings of the scenario and of the
  prefix **embedded separately** (the encoder truncates at 256 word pieces and
  scenarios alone reach 252), each with `prefix_tokens`, `cutoff_tokens`,
  `response_tokens` as numeric features. Δ is taken against the stronger of the
  two by window point estimate.
- Inference unit: scenario. Paired stratified scenario bootstrap (B = 1000)
  recomputing the full repeat-averaged estimator on each draw; one-sided
  bootstrap p = P(Δ* ≤ 0) (the Step 2 null). Secondary null: scale-free
  scenario-level channel-swap randomisation (5000 flips) on within-fold ranks.
- **Support:** the paired 95% interval on the mean privileged increment over
  `[-8,-1]` excludes zero **and** the one-sided bootstrap p < .05.

## Registered secondaries (Holm-adjusted as a family)

- **S1 trajectory (activations):** ROC_acts(offset_-1) − ROC_acts(prompt_final),
  same 186 scenarios, same draws.
- **S2 trajectory (privileged Δ):** Δ(offset_-1) − Δ(prompt_final).
- **S3** Δ over the window against the position-only baseline.
- **S4** Δ over the window against an LLM reader shown the identical scenario +
  visible prefix (`onset_prefix_reader_baseline_f.py`) — registered only if the
  scores file exists before any activation result is viewed; otherwise
  exploratory. A strong reader can only shrink Δ; that is the point.

## Controls

1. **Forced-prefix arm (required):** three fixed neutral prefixes with no
   trailing whitespace ("Well,", "Hmm, let me think about how to put this.",
   "Okay, so here's what I'm going to say about that.") on every scenario;
   registered statistic = mean over fixed prefixes of ROC(prefix-final) −
   ROC(prompt_final), two-sided bootstrap p. The in-distribution template
   "Oh, {questioner}," (39% of natural openings, class-balanced) is a secondary.
2. **Step 1/2 cross-check (required gate):** `prompt_final` layer-20 vectors
   must reproduce `results/activations_layer20.npz` per scenario (min cosine
   ≥ 0.98); causal attention makes this exact up to kernel noise.
3. **Cue audit (required gate):** regex screen + per-row human disposition of
   every limiting prefix, hash-bound to the extraction gate.
4. **Position-only** and **logit summaries** (entropy, top-1, top-5 mass) as
   descriptive channels.

## Extraction grid (one forward pass per scenario, all 258)

- Layers {10, 15, 20, 24, 28} = blocks {9, 14, 19, 23, 27}.
- Positions: `prompt_final`, offsets {−32, −16, −8…−1, 0, +1, +2, +4, +8, +16},
  `response_final`; float32 storage (lossless from bf16) with a finiteness
  assertion; next-token logit summaries at every valid cell via final norm +
  unembedding at those positions only.
- Full-sequence dump (exploratory, bf16-exact, ~1.1 GB, gitignored): every
  response position at all five layers, so no position ever needs
  re-extraction.
- Descriptive layer × position grid (`--grid`, 5 repeats, per-cell complete
  case with n reported); no maximum-over-grid claim; promoting any cell requires
  independent validation.

## Interpretation fixed in advance

- **Primary supported, S1/S2 ≈ 0:** the pre-onset signal is the prompt-final
  signal carried forward; Step 3 adds temporal *persistence*, not localisation.
- **Primary supported, S1 > 0 and S2 > 0:** the state sharpens toward the onset
  beyond what the emitted words explain. If the forced-prefix contrast is also
  > 0, continued internal development under matched words is supported (subject
  to forced-prefix distribution shift); if flat, generated wording explains the
  rise.
- **Primary not supported:** Step 2's prompt-final finding stands; Step 3 adds
  no temporal claim.
- The result may be described as not explained by the measured eval-awareness
  direction (Step 2B). It may not be described as privacy-specific, causal, or
  as "information absent from the text".

## Artifact interface

GPU: `results/onset_dynamics_acts_f.npz` + `..._manifest_f.json`;
`results/onset_dynamics_fullseq_f.npz`; `results/onset_dynamics_forced_acts_f.npz`
+ manifest. Local: `results/paper_pipeline/03_onset_dynamics/` (JSON/CSV/OOF
scores). Raw NPZ files are gitignored; manifests, tables and source hashes are
committed. Checkpoints are atomic and reject configuration drift (script, data
and git hashes are part of the checkpoint key).

## Amendment log

- **A1 (2026-09-01, pre-data):** cutoff redefined honestly; within-limiting
  secondary withdrawn; embeddings split; registered null changed to the Step 2
  bootstrap with the swap test secondary; trajectory secondaries S1–S2 and
  baselines S3–S4 added; forced prefixes rewritten; float32 storage; Step 1/2
  cross-check and hash-bound cue review added as gates; n fixed at 186.
