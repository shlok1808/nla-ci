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

*Add new entries as they surface. Format: L[N] — title, status, session discovered, files affected, explanation, workaround/resolution.*
