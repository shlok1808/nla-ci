# Session 09 — 2026-06-20/21

Build session (no GPU, no Lambda run). Two threads: (1) an **audit** of whether Fable's
phase-3 plan ever proposed verbalizing/projecting the *deflection* direction, and (2)
turning `minimal_pairs_f.py` (E3, the keystone) into a **human-gated pipeline** and
running its first stage locally. Retroactively logged 2026-06-29 from commits `33e4b7d`
(Jun 20) and `94fcaf1` (Jun 21) — session 08 was the last log written at the time.

Commits this session:
- `33e4b7d` — `docs/fable_deflection_ideas_audit.md` (+222).
- `94fcaf1` — `minimal_pairs_f.py` rewrite (gated stages), `judge_label_spotcheck_f.py`,
  generated `data/minimal_pairs_f.csv`, `results/judge_label_spotcheck_f.csv`, `.gitignore`.

---

## 1. Fable deflection-ideas audit (`33e4b7d`)

Question: did Fable ever propose **idea 1** (run the `refused − rest` difference-of-means
direction through the AV verbalizer, to test whether the strong deflection signal is
genuine CI content vs generic RLHF caution) or **idea 2** (project that deflection
direction onto leaked-class activations, rank, then NLA the extremes)?

Checked both evidence bases — the prose record (logs 04–08, METHODOLOGY_f, LIMITATIONS)
and the scaffolded `_f` code + `diff_of_means.py`. **Conclusion: Fable proposed neither;
both are genuinely new.** Fable built every *adjacent* piece but never these two moves:
- `refusal_dir` exists only as the **logit-lens positive control** (`logit_lens_f.py:53`)
  — vocabulary-space, not the AV; it failed (L10).
- The AV-on-a-direction script `alpha_sweep_f.py` sweeps `diff_raw / diff_pca / diff_boot /
  v_privacy` and **pointedly excludes `refusal_dir`** — idea 1 is a one-line add here.
- Every projection Fable scaffolded uses `v_privacy`, never the deflection direction;
  `minimal_pairs_f.py:274-297` is the exact template idea 2 would reuse (with the caveat
  that `refusal_dir` is behavior-derived, so idea 2 buys an NLA *read*, not a clean AUC).

On the strong-deflection/weak-leak pattern (Q3): Fable *interpreted* it ("leakage is a
default, not a decision"; analogized to the Zhao 2025 harmfulness-vs-refusal dissociation
and Arditi 2024 refusal direction) but never scaffolded a CI-vs-generic-caution
disambiguation. The closest existing asset is **Patchscopes** (`patchscopes_f.py`), which
scores CI-specific Yes/No questions against the `refused` label + a workplace placebo —
partial overlap with idea 1's *goal* by a different instrument, but the placebo is a
format control, not a generic-caution control, so it doesn't cleanly separate CI content
from RLHF hedging. Also confirmed absent everywhere: any **MLP/nonlinear probe** on the
leak null (grep of `docs/`+`scripts/` for mlp|nonlinear|forest|kernel|xgboost → 0 hits;
every probe in the repo is LogisticRegression). Full reasoning: `docs/fable_deflection_ideas_audit.md`.

---

## 2. `minimal_pairs_f.py` → gated pipeline (`94fcaf1`)

Rebuilt the keystone (E3) into **five human-gated stages, nothing auto-chains**:
`generate → review → extract → validate → project`, with a checkpoint between each.

- **generate** (local, ~$1, OPENAI_API_KEY) — GPT-4o-mini rewrites each tier-3 story into
  a minimal-pair counterfactual where the SAME info is already public, changing nothing
  else → `data/minimal_pairs_f.csv`. **[GATE 1]**
- **review** — side-by-side word-level diff so a human can confirm pairs differ ONLY in
  the privacy flow (not topic/length/wording).
- **extract** (Lambda A100, ~30 min) — layer-20 last-token acts for BOTH versions →
  `results/minimal_pairs_acts_f.npz`.
- **validate** — builds `v_privacy`; held-out secret-vs-public AUC (expect near-ceiling),
  per-pair direction consistency (one direction or a topic bundle?), length + lexical
  confound checks, and an extraction cross-check that re-extracted "secret" acts match
  `activations_layer20.npz`. Prints PASS/FAIL → `results/v_privacy_f.npz`. **[GATE 2]**
- **project** — projects ORIGINAL tier-3 acts onto `v_privacy` (axis derived WITHOUT leak
  labels → non-circular), restricted to **leaked-vs-appropriate** (drops `refused`, per
  session-08 L12) → `results/minimal_pairs_analysis_f.csv`. This is THE 2×2.

**Rewrite-quality fixes** (12/270 public rewrites had been leaking answers): broadened the
trailing-question strip regex, tightened the prompt to edit the secrecy clause *in place*
(no appends/answers, same length within ~10%), added a word-length post-gate
(`MAX_PUB_EXTRA = 10`), one strict retry on a bad rewrite, and self-healing of only
contaminated/stale pairs on re-run (`--dry-run` / `--retry-invalid`). Validation-gate
thresholds are recommendations only (`LEN_AUC_FLAG = 0.60`) — the human makes the call.

**Generate stage was run.** `data/minimal_pairs_f.csv` = 270 rows
(cols: `scenario_id, story_secret, story_public, what_changed, sim_ratio, dwl, valid`),
**237 valid pairs**, balanced across the 9 ConfAIde info-types.

---

## 3. Judge-label spot-check (`94fcaf1`)

`judge_label_spotcheck_f.py` — local, no GPU/API. Seeded-samples 15 `leaked` + 15
`appropriate` tier-3 rows and renders story + model response + judge reasoning so a human
can confirm the calls are genuine CI violations / non-violations before trusting the 2×2
y-axis (leak label is judge-derived, ~10% noise per METHODOLOGY_f §4).
Ran → `results/judge_label_spotcheck_f.csv` (30 rows, 15/15).

`.gitignore`: now excludes fetched ConfAIde tier files (`data/tier_*.txt`, not vendored).

---

## 4. State after this session

- **E3 generate done + gated** (237/270 valid pairs); next E3 step is **extract on Lambda**,
  then validate (GATE 2), then project.
- Two new audited experiments documented and confirmed novel (ideas 1 & 2), each one edit
  from an existing scaffold; neither run.
- Still outstanding from session 08: write L11 (drop "position 42") + L12 (use
  leaked-vs-appropriate) into `docs/LIMITATIONS.md`; dump the two missing k=42 pairwise
  AUCs from `position_sweep_acts_f.npz` on Lambda.

## 5. Next step

Unchanged from session 08 §10: the canonical next Lambda run is **`forced_prefix_f.py`
(E2)** — text-matched control isolating transcript from decision in the leak climb; report
(E1 − E2). In parallel, E3 is ready to advance through GATE 1 (`--review` the 237 pairs)
into Lambda extraction. Before any Lambda NLA run: `grep injection_token_id
actor_hf/nla_meta.yaml` → must be 149705 (L9).
