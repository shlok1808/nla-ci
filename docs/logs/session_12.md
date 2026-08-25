# Session 12 — 2026-08-25 (same day as session 11, later)

Local pre-flight audit + hardening session. No GPU, no API spend. An independent
adversarial audit of the five scripts scheduled for the next Lambda session,
verified against the local data (real Qwen tokenization included), followed by
fixes. Findings are labeled F1–F10; full text in the audit plan file (external),
condensed here. Prereg amendment **A1** registers every protocol change
**before** any affected extraction runs.

## Findings (condensed, ranked)

- **F1 (HIGH, fixed)** — `eval_awareness_f.py` built prompts with a third,
  unique story parser and a non-canonical user line ("The conversation turns to
  you." vs "{questioner} turns to you and brings this up."). Verified: 5 stories
  ended in dangling question fragments (281, 315, 380, 410, 440), id 255
  roleplayed Alice instead of Jane, 12/270 stories diverged from the canonical
  parse. v_eval would have been derived from a shifted prompt distribution and
  then erased from `activations_layer20.npz`. **Fixed:** build now imports
  `position_sweep_f.load_tier3` and uses the canonical user line; rebuilt CSV
  verifies 0/270 divergent, 0 questionee/questioner mismatches, id 255 = Jane.
- **F2 (HIGH, handled)** — parser fork: 12 tier-3 stories differ between the
  benchmark-era parser and `minimal_pairs_f.py`'s; 9 are in the 233 valid pairs.
  Written up as **L16**; handled via excluded-from-bar cross-check + per-id
  cosines + with/without-9 sensitivity in `--stage project`. Pairs not
  regenerated.
- **F3 (HIGH, fixed)** — GATE-2 verdict didn't enforce prereg §4 / E3 §4: the
  extraction cross-check was print-only and the (blocking) length flag was
  OR-merged with the (expected, non-blocking) lexical delta. Verdict logic now
  separates them; `CROSSCHECK_COS = 0.98` blocks.
- **F4 (MED-HIGH, fixed)** — `alpha_sweep_f.py`: fixed alphas calibrated to
  ‖diff_raw‖≈4 mis-rotate every other direction (v_deflect ‖·‖=8.48 measured);
  deflection direction missing; no direction-specificity control. **Superseded**
  by `verbalize_directions_f.py` (see below).
- **F5 (MED, fixed)** — `forced_prefix_f.py` lacked the canonical
  `leaked_vs_approp` target its own prereg §2 rule references. Added.
- **F6 (MED, fixed)** — E-EVAL 1-D erasure can't rule out a low-rank
  eval-awareness subspace; rank-3 span erasure added (A1.1: "substantially
  closed" requires both ≥0.80).
- **F7 (MED, mitigated)** — session-blocking env risks: `nla_inference.py`
  absent locally so `client.generate(..., temperature=0)` kwargs were
  unverifiable; injection-token check was manual; eval extract early-exit
  compared row *counts* only. New script asserts the NLAClient signature and
  `injection_token_id == 149705` before connecting; extract early-exit now
  compares key *content* and refuses stale stores.
- **F8 (LOW, fixed where it matters)** — non-atomic `np.savez` checkpoints could
  corrupt resume files on a mid-write kill. Atomic tmp+`os.replace` added to
  E2b, E2, E-EVAL saves.
- **F9 (LOW, noted)** — validate check [4] fits TF-IDF on all texts pre-CV;
  bias is conservative for the privileged-delta claim. Methods footnote.
- **F10 (LOW, fixed)** — E2b now runs E1's k=0-vs-npz sanity check (assert).

**Verified clean:** E2b↔E1↔npz comparability (same k=0 token — final `\n` of
the assistant header — via shared `build_inputs`); index arithmetic against real
tokenized lengths (responses min 65 / median 111 / max 277 → the `<2`-token skip
is dead code, no duplicate fraction columns); resume logic in all five scripts
(no drop/dup paths); probe/CV structure (reproduced 0.6836/0.8898 vs recorded
0.6865/0.8909; no leakage; pair-grouped folds correct).

## New: `verbalize_directions_f.py` (E-NLA, session B)

Angle-targeted (±10/20/30/45/60°, α solved per base/direction — solver
independently reproduces L8's "α≈20 ⇒ 45°" for diff_raw at α=19.9), v_deflect
primary + v_privacy/v_eval conditional, and the controls that make free text
admissible: 5 matched-angle random in-manifold rotations per θ, an off-manifold
raw-direction probe, endogenous temp-0 reads (36 refused + 36 matched
appropriate), divergence-point logging vs each base's θ=0 greedy decode.
`--dry-run` verified locally: **125 calls** (135–139 with v_privacy/v_eval).
Endpoints + falsification pre-registered in A1.2. `alpha_sweep_f.py` kept but
marked superseded.

## New result (free, local): `scratch/07` — what the desc channel actually carries

Grouped-CV TF-IDF probe on the existing tier-3 descriptions (vectorizer inside
folds): refused-vs-appropriate **0.767 ± 0.067** — independently replicates the
triad's 0.76. Top features: ↑refused = *awkward, dilemma, reaction, expecting,
not*; ↑appropriate = *supportive tone, understand where you're coming from,
great point, support*. **Zero privacy vocabulary in the top-20 features on
either side.** The verbalization channel transmits social-tension *tone*, not CI
content — pre-registered consequence (A1.2): the E-NLA lexicon endpoint is
insufficient alone; the text probe is the sensitive readout. Also a prior for
T2: what already passes the channel looks like generic social caution, not CI.
Output: `results/audit/07_desc_deflection_features.txt`.

## State / next actions

Everything above is local. Ready to run:

1. **Session A (A100, ~2h, plain HF), in tmux, gates-first order:**
   E2b `relative_position_sweep_f.py` → E3 `minimal_pairs_f.py --stage extract`
   then `--stage validate` (GATE 2 with the enforced verdict; read
   `E3_CLAIM_LANGUAGE.md` §4) → E-EVAL `--stage build` then `--stage extract`
   (the build MUST be re-run on Lambda — the prompts CSV is gitignored and was
   rebuilt this session) → E2 `forced_prefix_f.py` (predicted flat; run last) →
   `patchscopes_f.py` if ≥30 min remain. scp the npz/CSVs back; run E-EVAL
   `--stage analyze` and the E3 GATE-2 read locally.
2. **Session B (SGLang):** `verbalize_directions_f.py --dry-run` first, then the
   run. `grep injection_token_id actor_hf/nla_meta.yaml` → must be 149705.
3. **Open recommendations for the user (not yet adopted):** second-judge
   robustness pass on the 270 tier-3 responses (~$2 — armors the n=36 deflection
   headline against T3); reframe the paper as a verbalization-channel audit of
   behavioral self-knowledge; pre-commit the "eval-awareness in costume" variant
   in case E-EVAL collapses. Still open from before: HANDOFF §6 refresh, tier-4
   fix-or-drop (L14).
