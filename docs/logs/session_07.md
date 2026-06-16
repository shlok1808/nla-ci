# Session 07 — 2026-06-16

First Phase-3 experiment executed: the logit lens (METHODOLOGY_f §2 item 1 / run
order step 1). Run on Colab instead of Lambda (no GPU session needed — one matmul
per direction). Clean negative. New limitation L10; notebook tracked in repo.

## 1. What we ran

`notebooks/logit_lens.ipynb` (Colab `1yZj0CA9nWIBEjrk3uX4vjagtdfuftGXH`, copied into
the repo this session). Loads `results/activations_layer20.npz`, restricts to tier 3,
builds five directions, unit-normalizes each, projects through Qwen2.5-7B-Instruct's
final RMSNorm + `lm_head`, prints top±40 promoted/suppressed vocab tokens.

Directions: `diff_leaked_vs_not` (leaked − not-leaked mean), `refusal_dir`
(refused − rest mean), `pc1`/`pc2`/`pc3` (top SVD components of the tier-3 cloud).
**`v_privacy` was not run** — `minimal_pairs_f.py` hasn't produced it yet.

This is the notebook variant of the scaffolded `scripts/logit_lens_f.py` (same method,
minus the `v_privacy` branch). Raw output saved to `results/logit_lens_output.txt`.

## 2. What we got

No privacy-flavored English tokens on any direction. Every top±40 list is CJK
fragments + code/markup tokens (`(nonatomic`, `PROGMEM`, `offsetX`, `);}\n\n`,
`ISOString`) + replacement-char noise (`�`). The three PCs are pure junk — the
dominant directions of the cloud are format/multilingual noise, as expected.

The one nuance (see L10): the garble is not uniform. `refusal_dir`'s +end carries
`'拒'` (refuse), `'是不可能'` (impossible), `'严格'` (strict), `'asking'`; the
not-leaked end of `diff_leaked_vs_not` carries `'严禁'` (forbidden), `'绝不'` (never),
`'严格'` (strict). A faint prohibition/refusal theme surfaces in Chinese on exactly the
directions where it belongs (and not on the PCs). Weak, cherry-pickable, not reportable.

## 3. What it means

- **Expected failure mode, confirmed.** Layer 20 of 28 → 8 transformer blocks of
  rotation remain before the unembedding; raw logit lens is basis-misaligned this far
  back (the tuned lens, Belrose et al. 2023, exists for exactly this). The scaffold
  docstring predicted junk.
- **Positive control garbled, not erased.** `refusal_dir` is our strongest behavioral
  direction (probe ~0.89; deflection AUC 0.92) and was the intended positive control
  (METHODOLOGY_f §5 E8). It produced no clean English hedging tokens — which is the
  useful read: the *lens* failed, not the *signal*. The faint Chinese refusal/prohibition
  tokens on precisely that direction are consistent with garbling-not-erasure.
- Shlok's reading ("CJK + code across all 5, no privacy English anywhere") is accurate.
  The only thing it missed is the faint directional Chinese prohibition cluster above —
  noted honestly, weighted at zero.

## 4. Repo changes

- `results/logit_lens_output.txt` — raw output, with provenance header.
- `notebooks/logit_lens.ipynb` — Colab notebook copied into repo + cross-linked.
- `docs/LIMITATIONS.md` — added **L10** (logit lens uninformative at layer 20).

## 5. Does this change the plan? No.

Logit lens was always the 5-minute "is the cheap read free?" check; the answer is no,
and it closes out cleanly. It does **not** disturb the v_privacy / position-sweep plan.
If a rigorous vocabulary-space read is wanted later it's a tuned lens or Patchscopes
(`patchscopes_f.py`, already scaffolded — routes the direction back through the model
instead of straight to the unembedding), not the raw logit lens.

## 6. Next step

**`scripts/position_sweep_f.py`** is the correct next script — run order step 2
(METHODOLOGY_f §2: `logit_lens` → `position_sweep` → `forced_prefix` → `minimal_pairs`
→ `patchscopes` → `introspection` → `alpha_sweep` → `steering`). It needs the A100
(teacher-forced generation over stored responses, ~1 h), so this is the start of the
Lambda session.

**Before any Lambda NLA run** (not needed for position_sweep, which uses the subject
model only — but required before `minimal_pairs`/`alpha_sweep`/`steering` touch the
NLA): verify `grep injection_token_id actor_hf/nla_meta.yaml` returns **149705** (L9).
The `actor_hf/` checkpoint dir is not present in the local repo — this check happens on
Lambda where the NLA is served. position_sweep and forced_prefix do not invoke the NLA,
so they can run before that verification.

## Pipeline status

| Step | File | Status |
|------|------|--------|
| 1–6 | setup … probe_diagnostics | done (sessions 1–5) |
| 7 | `verbalization_survival_f.py` | done (session 6) |
| — | logit lens (`notebooks/logit_lens.ipynb` / `logit_lens_f.py`) | **done this session — negative, L10** |
| 8 | `position_sweep_f.py` | **next (Lambda)** |
| 9–14 | `forced_prefix_f` / `minimal_pairs_f` / `patchscopes_f` / `introspection_f` / `alpha_sweep_f` / `steering_f` | scaffolded, awaiting Lambda |
