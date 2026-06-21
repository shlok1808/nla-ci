# Audit — did Fable propose verbalizing / projecting the deflection direction?

Written to settle three questions about the Phase-3 plan against what Fable actually
recorded, across two evidence bases: (1) the prose record — session logs 04–08,
`docs/METHODOLOGY_f.md`, `docs/LIMITATIONS.md`; and (2) the scaffolded `_f` code +
`diff_of_means.py`. Companion context: `CLAUDE.local.md`, `docs/logs/session_05.md`
(the Fable methodology review), `docs/logs/session_06.md` (Fable build-out).

The two candidate experiments being checked:

- **Idea 1** — take the `refused − rest` difference-of-means vector and run it through
  the AV verbalizer (`inj_scale=150`, `temperature=0`) to find out whether the strong
  deflection signal is **genuine CI content** ("would disclose X's health status to
  someone not entitled") or **generic RLHF caution** ("sensitive topic, hedge").
- **Idea 2** — project that deflection direction onto the **leaked-class** activations,
  rank leaked scenarios by how much they internally resemble a refusal, then run NLA on
  the high- vs low-scoring leaked cases (the unverbalized-cognition test).

---

## Bottom line

**Fable never proposed idea 1 or idea 2, and never wrote either phrase anywhere.**
Both are genuinely new. The deflection/`refused` direction enters Fable's plan only as
(a) a *row in the survival triad* (TF-IDF on existing descriptions, not the AV) and
(b) a *positive control for the logit lens*. The single script that points the AV at
difference-of-means directions — `alpha_sweep_f.py` — sweeps
`diff_raw / diff_pca / diff_boot / v_privacy` and **pointedly excludes `refusal_dir`**.
Fable built every adjacent piece except these two specific moves.

The code corroborates the prose: NLA was never run on the deflection direction, and no
script ranks leaked cases by deflection-similarity. But two assets sit one edit away.

---

## Part 1 — Evidence from the prose record (session logs / METHODOLOGY / LIMITATIONS)

### Q1 — Did Fable suggest verbalizing the deflection direction (idea 1)?

**No, not via the AV verbalizer.** Three near-misses, none of which is idea 1:

1. **Triad row (closest, but wrong instrument).** Session 06 measured deflection's
   *description-level* survival and read it as
   (`METHODOLOGY_f.md` §3.2 #2, `session_06.md:48-49`):

   > "Deflection partially survives (0.76 ≫ 0.62) — privileged content CAN pass through
   > the channel when the underlying signal is strong."

   That 0.76 is **TF-IDF on the existing per-scenario `run_nla.py` descriptions**, not the
   AV fed a `refused − rest` diff-of-means vector. It is a survival measurement, not a
   verbalization of the deflection direction.

2. **Logit-lens positive control (right vector, wrong channel).** Fable defined exactly
   the idea-1 vector in `logit_lens_f.py:53` —
   `'refusal_dir': acts[labels=='refused'].mean(0) - acts[labels!='refused'].mean(0)` —
   and used it as the E8 positive control (`METHODOLOGY_f.md` §5 E8):

   > "the refusal direction acts as a positive control (expect hedging/deflection tokens)."

   This is a *vocabulary-space* read, not the AV, and it failed (garbled CJK, L10).

3. **Alpha-sweep (the AV verbalizer) — deliberately omits deflection.**
   `alpha_sweep_f.py:16-22` lists the directions it verbalizes: `diff_raw`, `diff_pca`,
   `diff_boot`, `v_privacy`. `refusal_dir` is not among them. In the entire scaffold the
   AV is never run on the deflection direction.

Net: idea 1 is a new combination of two things Fable set up separately. Fable never made
that move.

### Q2 — Did Fable suggest projecting the deflection direction onto leaked cases (idea 2)?

**No.** Every projection Fable proposed uses `v_privacy`, never the deflection direction.
Session 05 step 3(b) (`session_05.md:135-137`):

> "project each *original* tier-3 activation onto `v_privacy` and test whether weak
> privacy-norm encoding predicts leaking — this becomes the non-circular axis for the 2×2."

Same in `minimal_pairs_f.py:27` and `METHODOLOGY_f.md` §5 E3. The projection axis is
always the minimal-pairs privacy direction. Ranking *leaked* scenarios by refusal
resemblance, then NLA-ing the extremes, is nowhere in the plan.

Adjacent fact the pipeline later produced (session 08, post-Fable) that makes idea 2
coherent: leaked vs refused = **0.876** (`session_08.md:117`) — the deflection direction
carries enough structure to rank leaked cases against. But the experiment was never
proposed.

### Q3 — Strong-deflection / weak-leak: did Fable handle it, or flag it as generic caution?

**Partially.** Fable engaged with the pattern but *interpreted* it rather than treating
"is deflection CI-specific or generic RLHF caution?" as a question to resolve.

What Fable said (`session_06.md:50-54`, `METHODOLOGY_f.md` §3.2 #3):

> "at the moment of action the model has largely decided *whether to deflect* (0.92) but
> only weakly *whether its non-deflecting reply will constitute a leak* (0.65). Working
> hypothesis: **leakage is less a decision than a default** — the model fails to engage a
> protective policy rather than choosing disclosure. Direct analogue of the
> harmfulness-vs-refusal dissociation (Zhao et al. 2025, distinct encodings at distinct
> positions)."

**Did it flag deflection as possibly generic caution rather than CI-specific?** Not in
those words — but it *implicitly aligned deflection with the generic-safety family* by
repeatedly tying it to the refusal-direction literature: the harmfulness-vs-refusal
dissociation (Zhao et al.) and the Arditi et al. 2024 refusal-direction borrow notes
(`METHODOLOGY_f.md` §9). The logit-lens control even calls the expected tokens
"hedging/deflection" (`logit_lens_f.py:18-20`) — caution-flavored, not CI-flavored.
So Fable leaned toward "deflection ≈ the model's refusal/hedging mechanism," but **never
scaffolded a test to confirm it's generic vs CI-specific.** That disambiguation — exactly
idea 1's goal — is a gap Fable left open.

The one tool Fable built that partly disambiguates is **Patchscopes** (framed around
leak): `patchscopes_f.py:53-60` asks CI-specific questions of the stored activations
(`secret_present`, `violation_if_shared`, `will_disclose`) plus a `placebo_workplace`,
and the analysis computes the **`refused` AUC** alongside leak (`patchscopes_f.py:175,179`).
If `secret_present` reads `refused` at high AUC while the placebo stays ~0.5, that's
evidence the deflection cluster encodes genuine secret-presence (CI content) — idea 1's
goal by a different method. Caveat: the placebo is a *format/setting* control ("is this a
workplace?"), not a *generic-caution* control, so Patchscopes does not cleanly separate
"CI content" from "generic RLHF hedging" the way verbalizing the deflection direction would.

### Q4 — Fable's recorded next steps, mapped to the candidates

Canonical list: `session_05.md:111-163`, refined into the run order at
`METHODOLOGY_f.md` §2 (E1–E8 designs in §5, unscripted ideas in §6).

| # | Fable's step | Overlaps with… |
|---|---|---|
| E8 | **Logit lens** all directions (incl. `refusal_dir` control) | **Adjacent to idea 1** — same vector, unembedding read not AV. Done, negative (L10). |
| E1 | **Position sweep** (crystallization curve) | Genuinely different. Done (session 08): leak never ≥0.80; deflection flat-high. |
| **E2** | **Forced-prefix control** (text-matched) | **= the forced_prefix control**, exactly. Fable's, scaffolded, next in line. |
| E3 | **Minimal pairs → `v_privacy`** + non-circular 2×2 | Different (keystone). Its projection→leak is the *`v_privacy`* analogue of idea 2, not the deflection one. |
| E4 | **Steering** ±α·v_privacy, re-judge | Different — causal test on `v_privacy`. |
| E5 | **Patchscopes** targeted Yes/No + placebo | **Partial overlap with idea 1's *goal*** (CI-content readout of the deflection state) via a different instrument; computes `refused` AUC. |
| E6 | **Introspection** self-report baseline | Different — three-level dissociation. |
| E7 | **Denoised α-sweep → AV** (diff_raw/pca/boot/v_privacy, temp 0) | **The machinery idea 1 needs** — but it omits `refusal_dir`. Idea 1 = add the deflection direction here. |
| §6 | AR round-trip loss; NLA-on-the-judge; SAE decomposition of v_privacy; attention-from-secret-span; cross-position NLA; tier-2 norm probes | Genuinely different directions not previously raised. |

---

## Part 2 — Evidence from the code files (`_f` scripts + `diff_of_means.py`)

The code confirms NLA was never run on the deflection direction, and no script ranks
leaked cases by deflection-similarity. Two assets sit close to these ideas.

### Idea 1 — AV-verbalize the deflection direction: not in the code

- The only AV-on-a-direction script is `alpha_sweep_f.py`; its swept set is hardcoded to
  `diff_raw / diff_pca / diff_boot` (+ `v_privacy`). `refusal_dir` is absent — adding it
  is a one-line change.
- `diff_of_means.py:52-53` collapses `refused` into the not-leaked baseline
  (`not_leaked_mask = appropriate_mask | refused_mask  # collapsed`), so the deflection
  direction was never isolated for verbalization in the first place.
- Closest precedent: `logit_lens_f.py:18-20` builds `refusal_dir = refused_mean − rest_mean`
  as a positive control, sanity-expecting "deflection/hedging tokens." Session 07 ran it
  (L10) → no CI-specific disclosure language; instead a faint **generic prohibition theme
  in Chinese**, split across directions: `refusal_dir`'s +end carried 拒 (refuse),
  是不可能 (is impossible), 严格 (strict), `asking`; and the not-leaked end of
  `diff_leaked_vs_not` carried 严禁 (forbidden), 绝不 (never), 严格 (strict). Weak,
  cherry-pickable, zero-weighted — but it already leans toward "generic caution" over
  "would disclose X to someone not entitled."
- Idea 1's actual goal (CI-specific vs generic caution): the best existing asset is
  `patchscopes_f.py`. Its questions are CI-specific ("…a secret or shared in confidence?",
  "…a privacy violation to tell the asker everything?", "…about to reveal private
  information?") + a workplace placebo, and it scores every readout against the
  `refused`/deflection label, not just leak. If a CI question's Yes/No readout tracks the
  `refused` label (and the placebo doesn't), that's evidence the deflection state carries
  genuine CI content — the disambiguation idea 1 wants, no AV needed. Already scaffolded;
  Fable never called out this use.

### Idea 2 — project deflection onto leaked cases: not coded, but the template exists

- Not coded anywhere. But `minimal_pairs_f.py:274-297` is the exact template: it projects
  original tier-3 activations onto a non-leak-derived axis (`v_privacy`) and splits by it
  into a non-circular 2×2. Idea 2 = the same machinery with `refusal_dir` as the axis + a
  contrastive AV call on the extremes.
- **Caveat:** `refusal_dir` is behavior-derived, so the non-circularity argument does
  **not** transfer — you'd run idea 2 for the NLA *read* of the extremes, not for a clean
  AUC. (This is the structural difference from the `v_privacy` 2×2, whose whole point is
  that the axis never sees behavior labels.)

### Strong-deflection / weak-leak is a first-class target in code

- `position_sweep_f.py:12` — "does the refusal decision (AUC 0.92 at k=0) stay ahead?"
- `analyze_position_sweep_f.py` TASK 4 (deflection flatness) + TASK 5 (L10-vs-L20 gap).
- `forced_prefix_f.py` is the text-matched control — but aimed at the **leak climb**, not
  at disambiguating deflection. (It does also compute `refused_vs_rest`, line ~179.)

---

## Combined summary — what is genuinely absent

Confirmed missing from Fable's plan, in both prose and code (your additions):

- **Idea 1** — any AV call on the deflection direction. One-line add to `alpha_sweep_f.py`'s
  direction list (`temperature=0`, per L9). The only clean way to answer the
  CI-vs-generic-caution question Fable left open in Q3.
- **Idea 2** — any deflection-ranked contrastive NLA. The only projection scaffolded is
  onto `v_privacy`.
- **MLP / nonlinear probe** on the leak null. Confirmed absent everywhere — a grep of
  `docs/` and `scripts/` for `mlp|nonlinear|random forest|kernel|xgboost` returns zero
  hits. Every probe in the repo is `LogisticRegression` (`probe_diagnostics.py`, the triad,
  the sweep). Ruling out a linear-probe limitation is genuinely new.
- **forced_prefix** — *not* new: that's Fable's E2, already scaffolded and flagged as the
  required control for the leak climb (`session_08.md:213-215`).

Sharpening note for idea 2's premise (post-Fable data): session 08 recommends making
`leaked_vs_approp` (0.65) the canonical leak contrast rather than `leaked_vs_not` (0.68),
because `refused` is a *distinct mode* that inflates the collapsed contrast
(`session_08.md:186-191`). The 0.876 leaked-vs-refused separation is what makes a
deflection-resemblance ranking meaningful.

---

## Faithfulness caveat

Fable never wrote "verbalize the deflection direction" or "project deflection onto leaked
cases" anywhere — so, literally, it suggested neither. What the record shows is that Fable
built every adjacent piece (`refusal_dir` in `logit_lens_f.py`; the AV-sweep machinery in
`alpha_sweep_f.py`; the projection→2×2 template in `minimal_pairs_f.py`; the
`refused`-scored CI questions in `patchscopes_f.py`) except those two specific moves, and
framed the strong-deflection signal by analogy to the generic refusal/safety literature
without scaffolding a CI-vs-caution disambiguation for it.
