# HANDOFF — read this first

**Purpose:** single A–Z entry point for a fresh Claude Code session (e.g. after an
account switch on the same machine). It narrates the whole project from start to *now*
so you can pick up where the previous session left off without reconstructing state from
scattered logs. Written 2026-06-29.

**Where the real detail lives** (this file is the map, not the territory):
- `CLAUDE.local.md` — project rules, model checkpoints, benchmark table, git rules. Authoritative.
- `docs/logs/session_01..09.md` — chronological session logs. Read the latest before working.
- `docs/LIMITATIONS.md` — L1–L10 methodological caveats. Read before trusting any result.
- `docs/METHODOLOGY_f.md` — §2 run order, §5 per-experiment hypotheses/decision rules.
- `docs/fable_deflection_ideas_audit.md` — audit of two new candidate experiments (ideas 1 & 2).

---

## 1. What this project is (the one-paragraph version)

Wang et al. (2026) proved that Contextual Integrity (CI) norms are *linearly encoded* in
Qwen2.5-7B's residual stream — but their outputs are always scalar scores / directions,
never words. We use **Natural Language Autoencoders (NLAs)** at **layer 20** to *verbalize*
what those CI representations actually say, across all 4 ConfAIde tiers. Three intended
contributions: (1) first sentence-level description of CI-relevant activations; (2) all 4
tiers (Wang used only tier 3); (3) characterizing the privacy-awareness gap in words.
**Hard constraint: no causal claims** — "linearly decodable" ≠ "causally used."

Subject model: `Qwen2.5-7B-Instruct` (bf16). NLA checkpoints: `kitft/nla-qwen2.5-7b-L20-av`
(verbalizer) + `-ar` (reconstructor). Extraction = last-token residual at layer 20.
Infra: Lambda A100 80GB for GPU steps; analysis is local.

---

## 2. Chronological story so far (sessions 1–8 + post-08 work)

**Session 01–03 — setup & benchmark.** Installed deps, parsed all ConfAIde tiers,
ran Qwen inference + GPT-4o-mini judge. The NF4 (4-bit) benchmark is **deprecated** (L4);
the canonical run is **bf16** → `results/benchmark_results_bf16.csv`. Extracted layer-20
last-token activations → `results/activations_layer20.npz`.

**Session 04 — first NLA attempt: negative.** Ran the AV verbalizer per-scenario
(`run_nla.py`) and on the behavioral difference-of-means (`diff_of_means.py`). Result:
descriptions are **structure-focused, not CI-focused** (L6); raw diff vectors are
out-of-distribution for the NLA → injection failure (L7). No CI language came out.

**Session 05 — Fable methodology review (pivotal).** Diagnosed *why* session 04 failed:
the leak-decision direction at the last prompt token is only **AUC 0.68** — real
(p<0.002) but weak (L8), and α=2 interpolation was only a ~5° rotation, so unchanged
descriptions were expected, not a failure. Decision: pivot to **minimal-pairs /
paired counterfactuals** (secret vs public version of each story) to derive a clean
privacy direction `v_privacy` with topic confounds canceled. This `v_privacy` projection
*replaces* "NLA CI signal present/absent" as the 2×2 axis.

**Session 06 — Fable build-out.** Scaffolded the phase-3 experiment suite (the `_f`
scripts) with full designs in docstrings. Built the verbalization-survival triad
(input/acts/desc AUC): leak 0.58/0.68/0.61, refused 0.62/0.92/0.76. Resolved the NLA
injection mechanics (L9): NLA is **direction-only**; `injection_token_id` must be
**149705** — always `grep injection_token_id actor_hf/nla_meta.yaml` before any Lambda
NLA run.

**Session 07 — logit lens: negative (L10).** Reading directions through the unembedding
at layer 20 produces CJK/code junk (8-layer rotation gap). Uninformative. Also: the
**position sweep ran on Lambda** at the end of this session.

**Session 08 — position-sweep analysis (latest log, 2026-06-19).** Full local analysis of
the sweep grid (5 layers × 65 response positions × 3 targets × 270 tier-3 scenarios).
Headline findings:
- **Deflection ("refused") is a distinct, strongly-decodable cluster** — AUC 0.52(L10)→
  **0.89(L20)** at the prompt token, peaks exactly at L20 (justifies our layer choice as
  a feature, not a limitation). refused-vs-appropriate 0.92, leaked-vs-refused 0.88.
- **Leak never crystallizes** — across the *entire* (leak-inflated) transcript, leak
  decodability **never reaches 0.80** (max 0.77). Strongest support yet for "**leakage is
  a default, not a decision**."
- **Drop all "position 42" language** — the argmax peak is ≤1 SE on a broad k≈38–46
  plateau; token 42 is a mid-function-word ~38% through the reply (population smear, not a
  landmark). Candidate L11.
- **leaked-vs-appropriate (0.65) should be the canonical leak contrast**, not
  leaked-vs-not (0.68) — "not-leaked" isn't one cluster; collapsing inflates it.
  Candidate L12.
- Figures saved: `results/position_sweep_{heatmap,trajectory,pos0_depth,threeway}_f.png`.

**Session 09 (2026-06-20/21, build session, no GPU) — `docs/logs/session_09.md`:**
- `minimal_pairs_f.py` rebuilt as a **gated pipeline** (commit 94fcaf1): stages
  `generate → review → extract → validate → project`, with a human checkpoint between
  each (nothing auto-chains). Plus rewrite-quality fixes + a judge/label spot-check
  (`judge_label_spotcheck_f.py` → `results/judge_label_spotcheck_f.csv`).
- `docs/fable_deflection_ideas_audit.md` (commit 33e4b7d): confirmed two genuinely-new
  candidate experiments Fable never proposed — **idea 1** (AV-verbalize the deflection
  direction to test CI-content vs generic-RLHF-caution; one-line add to
  `alpha_sweep_f.py`) and **idea 2** (project deflection onto leaked cases, rank, NLA the
  extremes). Both sit "one edit away" from existing scaffolds.

---

## 3. Current state in one screen

**Benchmark (use bf16 CSV):** tier 3 is the primary contrast set — ~151 leaked /
~119 not-leaked (collapsed). Leak rate 55.9%. Tiers 1/2 are 100% appropriate (they test
sensitivity rating, not disclosure — not useful for the 2×2). Tier 4 exploratory (n=20).

**Pipeline status (per `CLAUDE.local.md` table + session 08):**
- Steps 1–7 (setup → verbalization-survival triad): **done.**
- Logit lens: **done, negative (L10).**
- Position sweep (E1): **run + fully analyzed (session 08).**
- `forced_prefix_f.py` (E2): **scaffolded — this is the next Lambda run.**
- `minimal_pairs_f.py` (E3, the keystone): **gated pipeline built; not yet run.**
- patchscopes / introspection / alpha_sweep / steering (E5/E6/E7/E4): **scaffolded.**

**Two outstanding loose ends from session 08:** (a) dump leaked-vs-refused &
refused-vs-appropriate AUCs at k=42 from `position_sweep_acts_f.npz` (Lambda-only,
~4 lines) to finish the three-way at the plateau; (b) write candidate L11/L12 into
`docs/LIMITATIONS.md`.

---

## 4. The 2×2 (the project's central analysis)

Axis X = leak behavior (leaked vs appropriate — the canonical contrast). Axis Y = privacy
encoding strength = projection of the original activation onto **`v_privacy`** (derived by
minimal-pairs, *without* leak labels → non-circular).
- Leaked + strong encoding → **privacy-awareness gap** (knows but leaks anyway)
- Leaked + weak encoding → **genuine CI blindspot**
- Not-leaked + strong → knows and behaves correctly
- Not-leaked + weak → genuine CI understanding

`v_privacy` does not exist yet — producing it (running `minimal_pairs_f.py` through
`validate`) is the gate to the whole 2×2.

---

## 5. Key limitations to internalize before trusting results (full text in LIMITATIONS.md)

- **L1** confidence column is hardcoded "high" — ignore it.
- **L2** tier-3 leak rate +17pp vs Wang (judge strictness + bf16 verbosity).
- **L3** three mislabeled tier-4 rows (IDs 492, 493, 495).
- **L4** NF4 benchmark deprecated — use bf16.
- **L5** `refused` = CI-preserving *deflection*, not hard refusal. Zero real refusals.
- **L6** per-scenario NLA descriptions are structure-focused, not CI-focused.
- **L7** raw diff vectors are OOD for the NLA (injection failure).
- **L8** leak is only weakly decodable at the extraction point (AUC 0.68, not ~1.0).
- **L9** injection is direction-only; `injection_token_id` must be 149705.
- **L10** logit lens uninformative at L20 (rotation gap → CJK/code junk).
- **L11/L12** (candidates, not yet written): drop "position 42"; use leaked-vs-appropriate.

---

## 6. What to do next (in priority order)

1. **Run `forced_prefix_f.py` (E2) on Lambda** — the text-matched control that isolates
   transcript from decision in the leak "climb." Report (E1 − E2), not E1 alone. This is
   the canonical next step per METHODOLOGY_f §2 and session 08 §10.
2. **Run the minimal-pairs keystone (`minimal_pairs_f.py`)** through its gated stages.
   *Spot-check ≥10 rewrites at GATE 1 before trusting anything* — pairs must differ ONLY
   in privacy flow, not topic/length/wording. This produces `v_privacy` → unlocks the 2×2.
3. Housekeeping: write L11/L12 into LIMITATIONS.md; close the two missing k=42 pairwise
   AUCs; consider the two new audited experiments (ideas 1 & 2).

**Before any Lambda NLA run:** `grep injection_token_id actor_hf/nla_meta.yaml` → must be
149705 (L9). Git rules: no Claude/AI attribution on commits; never push unless asked.

---

*Keep this file current at the end of each session, or let the per-session logs in
`docs/logs/` be the source of truth and update §2–§3 here when state changes materially.*
