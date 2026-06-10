# Limitations

Running log of methodological caveats, data quality issues, and known failure modes. Each entry is tagged to the session it was discovered in and cross-linked to the relevant file(s). Update this doc whenever a new issue surfaces — don't bury things in session logs only.

---

## L1 — Confidence field is meaningless

**Status:** Known artifact, workaround in place
**Discovered:** Session 02 (`logs/session_02.md`)
**Appears in:** `results/benchmark_results.csv`, `results/benchmark_results_bf16.csv`

GPT-4o-mini returns `confidence: "high"` for every single judgment regardless of actual uncertainty. The field was designed to let us flag borderline cases for manual review, but it never fires. Do not use the `confidence` column in any analysis.

**Workaround:** Ignore the column. If borderline-case detection is needed later, use judge `reasoning` text or add an explicit uncertainty prompt to the judge system message.

**Reference:** Judge prompt defined in `scripts/benchmark.py` (lines ~139–154). The `confidence` schema field is there but the model ignores it.

---

## L2 — Tier 3 leak rate elevated vs Wang et al.

**Status:** Accepted divergence
**Discovered:** Session 02 (`docs/logs/session_02.md`); updated Session 03 (`docs/logs/session_03.md`)
**Appears in:** `results/benchmark_results.csv` (NF4, 50%), `results/benchmark_results_bf16.csv` (bf16, 55.9%)

| Run | Tier 3 leak rate | Delta vs Wang et al. |
|-----|-----------------|----------------------|
| NF4 | 50.0% | +11.5pp |
| bf16 | 55.9% | +17.4pp |

bf16 leaks *more* than NF4, not less. Likely cause: our judge catches implicit allusions ("Remember what happened with X?") that Wang et al.'s judge passed, and the bf16 model is more verbally fluent (less blunted by quantization), so it produces more of those allusions. The sanity check in `scripts/benchmark.py` (line ~280) flags this as "outside threshold" — expected and not a blocker.

**What it means for analysis:** 55.9% gives a slightly unbalanced contrast set (~151 leaked / ~119 not-leaked) but still usable. Do not back-correct toward Wang et al.'s number. Manual spot-checks confirmed leaked labels are genuine CI violations.

**What it does not mean:** The benchmark is broken or the judge is miscalibrated.

---

## L3 — Three mislabeled Tier 4 rows (IDs 492, 493, 495)

**Status:** Fixed
**Discovered:** Session 02 (`logs/session_02.md`)
**Fixed in:** commit `2ef70fd` (`scripts/` — fix tier 4 judge hallucinations)
**Appears in:** `results/benchmark_results.csv` (fixed), `results/benchmark_results_bf16.csv` (will be correct if bf16 run produces correct labels)

Scenario IDs 492, 493, 495 were labeled `leaked` by GPT-4o-mini despite clean, non-disclosing model responses. Classic judge hallucination — the judge invented a privacy violation. These rows were manually corrected to `appropriate`.

**Tier 4 caveats that remain:**
- Only 20 scenarios total. Small-n stats are fragile.
- Even after fixing these three, ~25% of the tier may be mislabeled in subtle ways.
- Treat tier 4 results as exploratory only. Don't draw conclusions from it.

**Reference:** Tier 4 data is in `data/tier_4.txt`. Judge prompt in `scripts/benchmark.py`.

---

## L4 — Original benchmark ran in 4-bit NF4 quantization

**Status:** Resolved — bf16 re-run complete
**Discovered:** Session 03 (`docs/logs/session_03.md`)
**Affects:** `results/benchmark_results.csv` (NF4, deprecated for NLA pipeline)

The first benchmark run used 4-bit NF4 quantization. NLA checkpoints were trained on non-quantized activations, creating a distribution mismatch. Re-ran in bf16 — output in `results/benchmark_results_bf16.csv`, which is the canonical label source for all downstream NLA pipeline steps.

**Unexpected finding:** bf16 tier 3 leak rate (55.9%) is *higher* than NF4 (50%), not lower. This suggests quantization had a mild safety-dampening effect — the full-precision model is more verbally fluent and produces more implicit allusions that our judge catches. This is consistent with the L2 explanation. Not a blocker; bf16 labels are the correct ones to use regardless.

**Changed in:** `scripts/benchmark.py` (load_model, lines ~80–101; RESULTS_PATH, line ~186).

---

## L5 — `refused` label conflates hard refusals with CI-preserving deflection

**Status:** Known, handled in analysis
**Discovered:** Session 02 (`logs/session_02.md`)
**Appears in:** `results/benchmark_results.csv`, `results/benchmark_results_bf16.csv`

The judge labels a response `refused` when the model declines or deflects — but zero actual hard refusals exist in the dataset. Every `refused` case is a model that stays in character and redirects away from the private detail, which is CI-correct behavior.

**Workaround for analysis:** Collapse `refused` and `appropriate` into a single `not_leaked` category for binary analysis. The 2×2 matrix should use `leaked` vs `not_leaked`, not three-way labels.

**Reference:** Judge prompt in `scripts/benchmark.py` (lines ~139–154) defines the `refused` boolean. Label derivation logic at line ~207 (`leaked` takes priority, then `refused`, then `appropriate`).

---

## L6 — Per-scenario NLA descriptions are structure-focused, not CI-focused

**Status:** Known negative result — pivot to difference-of-means approach
**Discovered:** Session 04 (`docs/logs/session_04.md`)
**Appears in:** `results/nla_descriptions.csv`

All 496 NLA descriptions look structurally identical regardless of label (leaked/appropriate/refused). Every description focuses on:
- The chat/dialogue format ("Structured conversation format with...")
- Character names and surface topic
- Next-token prediction ("Final token 'response\n' opens a direct quote block, immediately requiring...")

No systematic difference between leaked and appropriate groups. The NLA is reading the model's "predict next token" state, not its CI reasoning state. At the last prompt token position, layer 20 is dominated by format/syntax prediction — the CI signal Wang et al. found via linear probes exists geometrically but is swamped by the louder next-token prediction signal in raw per-scenario activations.

**This is a meaningful negative result, not a pipeline failure.** It tells us the CI signal is not the dominant feature at this extraction point.

**Pivot:** Difference-of-means approach — compute `leaked_mean - appropriate_mean` to cancel format noise and isolate the CI direction. See L7 for the next issue encountered.

**Reference:** `results/nla_descriptions.csv`, `scripts/run_nla.py`, session 04 log.

---

## L7 — Raw difference vectors are out-of-distribution for NLA (injection failure)

**Status:** Known, fix in progress
**Discovered:** Session 04 (`docs/logs/session_04.md`)
**Appears in:** `results/diff_means_output.txt`

The raw `leaked_mean - appropriate_mean` difference vector has norm ~4.3, far smaller than natural layer 20 activations. When fed to the NLA verbalizer, it produces Chinese text and math competition gibberish — the documented injection failure mode from nla-inference ("if injection fails, the actor verbalizes something Chinese from free-association").

**Why:** The NLA was trained on real layer 20 activations which have large norms and rich structure. The difference vector has most of its 3584 dimensions near zero (they cancel in subtraction), leaving a sparse, tiny, out-of-distribution vector the NLA has never seen.

**Fix in progress:** Counterfactual interpolation — instead of feeding the raw diff, feed `not_leaked_mean + 2 * diff`. This starts from a natural-looking activation and pushes it along the leaked direction, keeping it in-distribution while amplifying the signal. Implemented in updated `scripts/diff_of_means.py`, awaiting re-run.

**Reference:** `results/diff_means_output.txt`, `scripts/diff_of_means.py`, session 04 log. Related: Francesco Zaffino's background washout approach in [SAE-it Across Models](https://www.lesswrong.com/posts/AtbZQuAn2iY2jCup2/sae-it-across-models-explaining-features-with-foreign-nla).

---

## L8 — Leak behavior is only weakly decodable at the extraction point (probe AUC 0.68, not ~1.0)

**Status:** Known — reframes L6/L7
**Discovered:** Session 05 (Fable methodology review)
**Appears in:** `scripts/probe_diagnostics.py` (run locally on `results/activations_layer20.npz`)

Wang et al.'s near-perfect AUROC is for recovering **CI norm attributes** (information type, recipient, transmission principle) — properties of the input scenario. It is *not* a claim that the **leak/no-leak behavioral outcome** is linearly decodable. Those are different probe targets, and their own privacy awareness gap (correct norm encoding, 38.5% leakage) implies the two must dissociate.

Measured on our tier 3 activations (leaked vs not-leaked, 151/119):

| Quantity | Value |
|---|---|
| 5-fold CV logistic probe AUC | **0.68** (C=1e-3; 0.65–0.68 across C) |
| cosine(leaked_mean, not_leaked_mean) | 0.9990 |
| ‖diff‖ vs mean within-class distance | 4.0 vs 18.5 |
| Permutation test on ‖diff‖ | p < 0.002 (null mean 2.3, max 3.3) — real but weak |
| Held-out projection onto diff direction | AUC 0.67 |
| Control: tier_3 vs tier_1/2 probe | AUC 1.00 |

**Implications:**
- The diff-of-means direction is statistically real (passes permutation test) but carries at most AUC-0.68 worth of signal — roughly a third of its energy is label-sampling noise.
- The NLA "failure" in L6/L7 is therefore largely *expected*: there was never a near-perfect leak signal at this position to surface. The honest claim is "weakly decodable, not verbalizable," not "strongly decodable but unverbalizable."
- Counterfactual interpolation at α=2 rotated the mean vector by only ~5° (‖2·diff‖ ≈ 8 against a mean of norm ~88, with the NLA renormalizing to inj_scale=150 anyway) — identical descriptions were geometrically guaranteed, not evidence of absence.

**Reference:** `scripts/probe_diagnostics.py`, session 05 log.

---

## L9 — Injection mechanics resolved: ID discrepancy was misrecorded; NLA is direction-only (revises L7's mechanism)

**Status:** Resolved — documentation correction + mechanism correction
**Discovered:** Session 06 (read `nla_inference.py` source + shipped `nla_meta.yaml` from HuggingFace)
**Affects:** `docs/logs/session_04.md` (misrecorded fix), L7 (wrong mechanism), all future NLA runs

**1. The injection-token "fix" in session 04 is misrecorded.** The shipped `nla_meta.yaml` on HuggingFace has `injection_token_id: 149705`, and `load_nla_config()` **hard-asserts** `tokenizer.encode('㈎') == [injection_token_id]` at client init — a mismatched ID crashes before any request and cannot silently produce gibberish. Both logged runs printed `id=149705` and completed, so yaml and tokenizer agreed at 149705 at run time. The session-04 claim that "the tokenizer maps ㈎ to 149785" was likely a measurement error — the source explicitly notes `convert_tokens_to_ids('㈎') → None for Qwen; encode('㈎') → [149705]` (byte-level BPE keys on byte strings, not unicode chars). **Action:** if `actor_hf/nla_meta.yaml` on Lambda still carries a sed'd 149785, the next run will crash with "tokenizer drift" — restore 149705 (or re-download the checkpoint). Verify with `grep injection_token_id actor_hf/nla_meta.yaml`.

**2. `injection_scale=150` is an L2-renormalization applied to every injected vector** (`normalize_activation`: `v / (||v|| / 150)`), and injection replaces the embedding row at the marker position. The NLA is therefore **sensitive to direction only — magnitude is always discarded.** This revises L7:
- L7's mechanism ("diff vector norm ~4.3 is too small → OOD") is wrong. Norm cannot be the failure cause; every vector reaches the model at norm 150. The raw diff failed because its *direction* is off the natural activation manifold (≈⅓ of its energy is label-sampling noise per L8, and the signal component is a small perturbation direction never seen in isolation during NLA training) → free-association CJK output, the documented failure mode.
- Counterfactual interpolation can only work by *rotating* the mean direction. α=2 rotated it ~5° (L8) — unchanged descriptions were guaranteed. Meaningful α must rotate substantially: α≈20 for ~45° given ‖diff‖≈4 against means of norm ~88. See `scripts/alpha_sweep_f.py`.

**3. All existing NLA runs sampled at temperature 1.0** (the client default; neither `run_nla.py` nor `diff_of_means.py` passes `temperature`). Descriptions are stochastic samples, not deterministic reads — this adds decode noise to all description-level analyses. Conservative for our negative results (noise can only hurt detectability), but any *contrastive* verbalization (e.g., same vector ± perturbation) must pass `temperature=0`.

**Reference:** `nla_inference.py` (kitft/nla-inference, fetched 2026-06-10): `load_nla_config` asserts, `normalize_activation`, `inject_at_marked_positions`; shipped sidecar at `huggingface.co/kitft/nla-qwen2.5-7b-L20-av/raw/main/nla_meta.yaml`.

---

*Add new entries as they surface. Format: L[N] — title, status, session discovered, files affected, explanation, workaround/resolution.*
