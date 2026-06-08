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

## L2 — Tier 3 leak rate 11.5pp above Wang et al.

**Status:** Accepted divergence, within expected range
**Discovered:** Session 02 (`logs/session_02.md`)
**Appears in:** `results/benchmark_results.csv` (NF4 run); expect similar in bf16 run

Our tier 3 leak rate is ~50% vs Wang et al.'s 38.5% (+11.5pp). Two likely causes:

1. **Judge strictness on implicit disclosures.** Our judge catches indirect allusions ("Remember what happened with X?") that Wang et al.'s judge may have passed. This inflates our leaked count.
2. **Model version / quantization differences.** Wang et al. may have tested a different checkpoint or precision.

The 10pp sanity check in `scripts/benchmark.py` (line ~280) flags this as "outside threshold." That warning is expected and not a blocker.

**What it means for analysis:** Tier 3 leak/not-leak contrast set is still valid. The 50% rate gives a roughly balanced split (135 leaked / 135 not-leaked) which is actually better for the 2×2 matrix than a 38.5% split would be. Do not back-correct toward Wang et al.'s number.

**What it does not mean:** The benchmark is broken. Manual spot-checks (session 02) confirmed leaked labels are genuine CI violations.

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

**Status:** Mitigated — re-running in bf16
**Discovered:** Session 03 (`logs/session_03.md`)
**Affects:** `results/benchmark_results.csv` (NF4, deprecated for NLA pipeline)

The first benchmark run used 4-bit NF4 quantization (`BitsAndBytesConfig`, `bnb_4bit_quant_type='nf4'`). NLA checkpoints (`kitft/nla-qwen2.5-7b-L20-av`, `kitft/nla-qwen2.5-7b-L20-ar`) were trained on non-quantized activations. Extracting activations in bf16 from a model whose labels came from a 4-bit run creates a distribution mismatch: the labels may not reflect the bf16 model's actual behavior, since quantization can shift safety/alignment behavior.

**Fix:** `scripts/benchmark.py` updated to load in `torch_dtype=torch.bfloat16` (no quantization). Output goes to `results/benchmark_results_bf16.csv`. The NF4 CSV is preserved but should not be used as labels for the NLA pipeline.

**Changed in:** `scripts/benchmark.py` (load_model, lines ~80–101; RESULTS_PATH, line ~186). See also session 03 log.

**Expected outcome:** bf16 tier 3 leak rate should be closer to Wang et al.'s 38.5% than the NF4 50%. If divergence is >15pp, investigate further — could indicate quantization genuinely affected safety behavior, which would itself be a finding worth noting.

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
