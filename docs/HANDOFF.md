# HANDOFF — read this first

**Purpose:** single A–Z entry point for a fresh Claude Code session (e.g. after an
account switch or a model switch on the same machine). It narrates the whole project from
start to *now* so you can pick up where the previous session left off without
reconstructing state from scattered logs.
**Written 2026-06-29; §2/§3/§5/§6 updated 2026-08-25 (session 11).**

**Sitting down to work? Open `docs/RUNBOOK.md`** — the ordered queue (the GPU
session step by step, then the parked NLA-description thread). This file is the
map; the runbook is what to actually type.

**Where the real detail lives** (this file is the map, not the territory):
- `CLAUDE.local.md` — project rules, model checkpoints, benchmark table, git rules. Authoritative.
- `docs/logs/session_01..12.md` — chronological session logs. **Read `session_12.md` first**
  (2026-08-25 pre-flight audit + fixes: L16 parser fork, GATE-2 enforcement, eval-awareness
  prompt rebuild, `verbalize_directions_f.py` supersedes `alpha_sweep_f.py`, prereg amendment A1)
  — it is the most recent state and supersedes older logs where they conflict.
- `docs/LIMITATIONS.md` — **L1–L15** methodological caveats. Read before trusting any result.
- `docs/PREREGISTRATION.md` — dated, locked predictions + decision rules for every
  remaining experiment. Read before running anything; do not edit past entries.
- `docs/REFERENCES.md` — every external citation, grouped by the role it plays in the
  paper. Start here when writing related-work or limitations prose.
- `docs/E3_CLAIM_LANGUAGE.md` — permitted/forbidden claim language for minimal pairs,
  fixed in advance of GATE 2.
- `REPORT.md` (repo root) — independent research audit, 2026-07-02. Verifies every headline
  number and adds five findings that reshaped the paper.
- `scratch/README.md` + `results/audit/*.txt` — the audit's analysis scripts and their
  captured output (re-run and verified 2026-08-25).
- `docs/METHODOLOGY_f.md` — §2 run order, §5 per-experiment hypotheses/decision rules.
  Predates the audit; where it conflicts with `PREREGISTRATION.md`, the latter wins.
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

**Session 10 (2026-06-29, commit `8ede759`, no log written).** GATE-1 review of the
minimal pairs: 4 pairs dropped (289/372/449/475), leaving **233 valid**; L13 written.

**Independent audit (2026-07-02) — `REPORT.md`, `scratch/01–06`.** Left untracked for
eight weeks; adopted into the repo in session 11. Every headline number reproduces. Five
findings that reshaped the paper: (1) nonlinear probes do **not** beat linear on leak —
the signal is genuinely thin, not hidden (kills the planned MLP experiment); (2)
cos(v_leak, v_deflect) = **−0.52** and erasing deflection drops the leak probe
0.684→0.628, so ~⅓ of the leaked-vs-not signal was "absence of deflection"; (3)
full-response text recovers the judge's own leak label at only **0.749** ≈ the activation
ceiling, so ~0.75 may be a label-subtlety ceiling rather than a probe failure; (4) the
sweep covers only the first 64 of a **median-111-token** response; (5) the minimal-pairs
lexical confound is near-total (text AUC **0.956**, 0.824 marker-ablated) and inherent.
Also found a live data bug: the L3 tier-4 label fix never reached the bf16 CSV.

**Session 11 (2026-08-25, commit `ebfc148`, local only) — `docs/logs/session_11.md`.**
Adopted the audit into the record; wrote **L11/L12/L14/L15** and upgraded L13; produced
`docs/PREREGISTRATION.md` and `docs/E3_CLAIM_LANGUAGE.md`; built the eval-awareness
control; and ran a **paired bootstrap on the triad deltas** that changed the paper's
framing (see §3a below).

---

## 3. Current state in one screen

**Benchmark (use bf16 CSV):** tier 3 is the primary contrast set — 151 leaked / 83
appropriate / 36 refused. Leak rate 55.9%. **Canonical leak contrast is leaked-vs-
appropriate (0.65), not leaked-vs-not (0.68)** — see L12. Tiers 1/2 are 100% appropriate
(they test sensitivity rating, not disclosure). Tier 4 exploratory (n=20) and carries a
live label bug (**L14**) — fix or drop before reporting any tier-4 number.

**Pipeline status:**
- Steps 1–7 (setup → verbalization-survival triad): **done.**
- Logit lens: **done, negative (L10).**
- Position sweep (E1): **run + analyzed** (session 08), now qualified by **L15**.
- Independent audit + stats hardening: **done** (`REPORT.md`, `results/audit/`,
  `results/stats_hardening_f.csv`).
- `minimal_pairs_f.py` (E3, keystone): `generate` **done**, GATE-1 review **done**
  (233 valid pairs); `extract` / `validate` / `project` **not yet run.**
- `relative_position_sweep_f.py` (E2b): **written, never run.**
- `forced_prefix_f.py` (E2): **scaffolded, never run.**
- `eval_awareness_f.py` (E-EVAL): `--stage build` **done** (1620 prompts);
  `extract` / `analyze` **not yet run.**
- E-NLA: **`verbalize_directions_f.py`** (new, session 12) supersedes `alpha_sweep_f.py` —
  angle-targeted rotations, v_deflect primary, matched-angle random + off-manifold controls,
  endogenous temp-0 reads. `--dry-run` verified locally (125 calls). Prereg amendment A1.2.
- patchscopes / introspection / steering (E5/E6/E4): **scaffolded**, deprioritized.

### 3a. The result that changed the framing (session 11)

The triad's published CIs are on the *marginal* AUCs and overlap heavily. But the three
probes score the same scenarios, so the correct object is a **paired bootstrap on the
delta** (`scripts/stats_hardening_f.py` → `results/stats_hardening_f.csv`):

| Contrast | acts − input | desc − input | acts − desc |
|---|---|---|---|
| leaked vs not | **+0.103** p=.009 | +0.025 **n.s.** | +0.077 p=.046 |
| leaked vs appropriate | **+0.104** p=.020 | +0.051 **n.s.** | +0.053 **n.s.** |
| refused vs appropriate | **+0.297** p<.001 | **+0.138** p=.042 | **+0.159** p<.001 |

**The leak row cannot carry a "verbalization destroys the signal" claim** — `desc − input`
is indistinguishable from zero, so the description channel never rose above the input
baseline and there is no collapse to demonstrate. **Deflection** is where channel loss is
shown: descriptions retain roughly half the privileged signal and lose the rest. Report
deltas with CIs, never bare AUCs.

**Outstanding loose end from session 08:** dump leaked-vs-refused & refused-vs-appropriate
AUCs at k=42 from `position_sweep_acts_f.npz` (Lambda-only, ~4 lines). Low priority now
that **L11** retires the argmax framing.

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
- **L11** "position 42" retired — argmax of 975 correlated AUCs, ≤1 SE margin.
- **L12** use leaked-vs-appropriate as canonical; leaked-vs-not is inflated by deflection.
- **L13** minimal-pair lexical confound, quantified pre-GPU (text AUC 0.956) — the GATE-2
  confound flag **will fire by design** and does not block `--stage project`.
- **L14** tier-4 label bug live in the bf16 CSV (ID 495 is a confirmed judge hallucination).
- **L15** the position sweep covers only the first 64 of a median-111-token response.
- **L17** the NLA's reconstruction objective does not penalise false claims — description
  evidence is weaker than the reconstruction score implies (verbalizer faithfulness).
- **L16** story-parser fork: 12 tier-3 stories diverge across pipeline stages (9 in the
  valid minimal pairs) — cross-check excludes them; project reports a with/without sensitivity.

---

## 6. What to do next (in priority order)

All local prep is done. The next action is a **single Lambda session running four
experiments back to back** (~2h, ~$5–10). Read `docs/PREREGISTRATION.md` first — each
experiment's prediction and decision rule is locked there.

**Session A — one A100 spin-up, plain HuggingFace transformers:**

1. **`relative_position_sweep_f.py` (E2b, ~40 min)** — samples at relative positions
   {10,25,50,75,90}% of each response plus the final token. Closes **L15**. Prereg §3.
2. **`minimal_pairs_f.py --stage extract` then `--stage validate` (E3, ~30 min)** —
   produces `v_privacy` → unlocks the non-circular 2×2. **Read
   `docs/E3_CLAIM_LANGUAGE.md` §4 at GATE 2 before proceeding to `--stage project`.**
   Prereg §4.
3. **`forced_prefix_f.py` (E2, ~30–60 min)** — text-matched control for the leak climb.
   `scratch/03` predicts flat; run it anyway (TF-IDF is a weak baseline and reviewers
   want the model-grade control). Prereg §2.
4. **`eval_awareness_f.py --stage extract` (E-EVAL, ~20 min)**, then `--stage analyze`
   locally — tests whether the strongest result (deflection, 0.89) is CI content or the
   model detecting that it is being evaluated. Prereg §6.

**Session B — separate, needs SGLang (~1h):**

5. **E-NLA** — point the verbalizer at **`v_deflect`** and **`v_privacy`** via
   **`verbalize_directions_f.py`** (run `--dry-run` first; supersedes `alpha_sweep_f.py`).
   Scheduled after E3 because E3 creates the second direction. Prereg §5 + amendment A1.2.

**Also outstanding:** fix or drop tier 4 (**L14**).

**Before any Lambda NLA run:** `grep injection_token_id actor_hf/nla_meta.yaml` → must be
**149705** (L9), and temperature must be **0** — all prior NLA runs used 1.0 and were
stochastic samples. Git rules: no Claude/AI attribution on commits; never push unless asked.

---

*Keep this file current at the end of each session, or let the per-session logs in
`docs/logs/` be the source of truth and update §2–§3 here when state changes materially.*
