# Session 08 — 2026-06-19

Analysis session: the **position sweep** (METHODOLOGY_f §2 step 8 / E1) ran on Lambda
at the end of session 07 and produced `results/position_sweep_aucs_f.csv` (5 layers
{10,15,20,24,28} × 65 response positions × 3 targets × 270 tier-3 scenarios). This
session is the full post-hoc verification, visualization, and reading of that grid. No
GPU needed — everything below is local (the AUC grid is committed; the raw acts
`position_sweep_acts_f.npz` stayed on Lambda, gitignored).

New code this session (both committed):
- `scripts/analyze_position_sweep_f.py` — tasks 1–7: heatmaps, trajectories, peak/
  flatness/depth stats, three-way pairwise, and the npz-reproduction sanity checks.
- `scripts/token42_qualitative_f.py` — task 8: re-tokenizes stored responses with the
  exact sweep call and marks where k=42 lands; response-length distribution.

Figures (all `results/`): `position_sweep_heatmap_f.png`,
`position_sweep_trajectory_f.png`, `position_sweep_pos0_depth_f.png`,
`position_sweep_threeway_f.png`.

---

## 0. Pipeline sanity (do the numbers mean anything?) — yes

- **Sweep ≡ npz at pos 0.** Re-probing `activations_layer20.npz` (tier-3, 270×3584,
  fp32) with the *exact* sweep probe config (StandardScaler + LogReg C=1e-3, 5-fold
  StratifiedKFold rs=0, `cross_val_predict` proba) reproduces the CSV's L20/k=0 AUCs to
  ≤0.003: refused 0.8909→**0.8898**, leaked-vs-not 0.6865→**0.6836**, leaked-vs-approp
  0.6522→**0.6509**. The pos-0 last-prompt-token state equals the extraction point of
  `activations_layer20.npz` by causal masking, and the probe confirms it functionally.
- **The "Sanity vs npz … 0.7500" delta the sweep printed is benign bf16
  nondeterminism.** Layer-20 acts have max|x|=33.75, mean L2-norm 89.6, median|x|=0.66.
  A 0.75 max-abs delta on the largest dim between a prompt-only forward (npz) and a
  prompt+response forward (sweep, 8× longer seq → different reduction order) is normal,
  and it moves the probe by <0.003 AUC. Not a concern; revises nothing.
- **No survivorship/NaN-padding confound.** The CSV's `n` is constant (234/270) at
  *every* position including k=64. Checked with the Qwen tokenizer: **all 270 tier-3
  responses are ≥65 tokens** (min 65). No response is NaN-padded at any swept position,
  so late-position AUCs are on the full set, not a lengthening subset.

---

## 1. The grid (tasks 1–2)

`position_sweep_heatmap_f.png` (3 stacked panels, layer×position, diverging cmap
centered at 0.5 chance) and `position_sweep_trajectory_f.png` (AUC vs position, one
line per layer). Global AUC range across the whole sweep: 0.466 .. 0.895.

Read at a glance: the **refused/deflection** panel is uniformly deep (AUC ~0.80–0.89
everywhere); the two **leak** panels are pale (~0.60–0.77) with a faint vertical band
near k≈40–43 and a clearly weaker top row (layer 10).

---

## 2. The pos-42 "peak" is real as an argmax but **not reportable** (task 3)

leaked_vs_approp argmax per layer:

| layer | argmax k | AUC | pos42 | Δ(42 − mean of nbrs 41,43) |
|---|---|---|---|---|
| 10 | **40** | 0.7351 | 0.6983 | +0.014 |
| 15 | 42 | 0.7686 | 0.7686 | +0.027 |
| 20 | 42 | 0.7716 | 0.7716 | +0.027 |
| 24 | 42 | 0.7710 | 0.7710 | +0.026 |
| 28 | 42 | 0.7478 | 0.7478 | +0.039 |

So the claim "pos 42 peaks at L15/20/24/28, pos 40 at L10" is **literally true**. But the
peak is shallow: the margin over neighbors (0.003–0.039 AUC) is **≤1 SE** (Hanley–McNeil
SE at n=151/83 ≈ 0.030), and it sits on a broad plateau k≈38–46 (window std 0.037–0.05).
**Do not put a token index in the paper** — report "a broad mid-response plateau peaking
around k≈40–43."

---

## 3. Deflection is flat-high with a slight downward drift (task 4)

refused_vs_rest at L20, by bin:

| bin | mean | std | min | max |
|---|---|---|---|---|
| early 0–9 | 0.840 | 0.030 | 0.805 | 0.891 |
| mid 10–39 | 0.837 | 0.026 | 0.792 | 0.886 |
| late 40–64 | 0.805 | 0.031 | 0.743 | 0.866 |
| **all 0–64** | **0.825** | **0.032** | 0.743 (k=50) | 0.891 (**k=0**) |

CV 3.9%, range 0.148. Linear slope **−0.0008/token (−0.053 over the response)** — it
*peaks at the prompt token (k=0)* and slightly decays. Early≈mid (statistically
indistinguishable), late drops ~0.03. "Starts high and stays high" holds; it does not
*strengthen*. Contrast: leaked_vs_not L20 slope is **+0.0008/token** (opposite sign).

---

## 4. Depth at pos 0 — gradual rise, deflection peaks exactly at L20 (task 5)

`position_sweep_pos0_depth_f.png`. AUC at k=0:

| target | L10 | L15 | L20 | L24 | L28 |
|---|---|---|---|---|---|
| refused_vs_rest | 0.524 | 0.744 | **0.891** | 0.870 | 0.815 |
| leaked_vs_not | 0.482 | 0.620 | 0.687 | 0.689 | 0.656 |
| leaked_vs_approp | 0.507 | 0.614 | 0.652 | 0.680 | 0.678 |

All three ~chance at L10 (leak below chance). Deflection rises across L10→L15→L20
(+0.220, +0.147 — gradual two-step, not a single sudden layer) and **peaks at L20**,
then declines. The 0.52→0.89 deflection claim is exact. **Layer 20 (the only released
NLA checkpoint) is near-optimal for the strongest behavioral signal** — feature, not
limitation, now with a curve behind it.

---

## 5. Three-way separation at L20 (task 6)

pos 0 (computed from npz; positive class first):

| pair | AUC | n |
|---|---|---|
| refused vs appropriate | **0.920** | 119 |
| leaked vs refused | **0.876** | 187 |
| leaked vs appropriate | **0.651** | 234 |

Refused/deflection is a **distinct cluster**, well-separated from both other groups;
leaked and appropriate **heavily overlap**. (Reproduces the session-06 triad acts
numbers: 0.92 / 0.65.)

**pos 42 is only partially computable locally:** leaked-vs-appropriate = **0.7716** is in
the CSV, but leaked-vs-refused and refused-vs-appropriate at k=42 need
`position_sweep_acts_f.npz` (Lambda-only). Outstanding — a ~4-line dump on Lambda would
close it.

---

## 6. What is token 42? (task 8) — a population smear, not a landmark

Token k=42 lands **mid-sentence on a mundane function word / word-fragment in all 10
leaked samples** (`' Speaking'`, `' k'`⟨udos⟩, `' the'`, `','`, `' or'`, `' to'`,
`' with'`, `' and'`, `' it'`×2), at a **median 38% of the way through the reply**
(p25–p75: 32–45%). It does **not** consistently sit on the secret-revealing token — the
disclosure (`"I remember how Jane…"`, `"For Pablo, the psychological toll…"`) begins
variably before or after k=42. The AUC bump near k≈42 is an artifact of averaging over
heterogeneous transcripts, not a per-scenario semantic position.

Response lengths (tier-3): leaked min 65 / median 111 / max 266; appropriate 65 / 118 /
277; refused 82 / 108 / 151.

---

## 7. Claims assessment

| Claim | Verdict |
|---|---|
| Deflection 0.52(L10)→0.89(L20) at pos 0 | ✅ exact; gradual two-step rise, peaks at L20 |
| Deflection trajectory flat | ✅ holds; flat-high but slightly *declining* (−0.05), maximal at k=0 |
| Refused forms a distinct cluster | ✅ strong (0.88–0.92 vs both groups) |
| "pos 42 is the peak" (leaked_vs_approp) | ⚠️ technically true, materially misleading — ≤1 SE local max on a k≈38–46 plateau; drop the index |
| Leak "climbs" | ⚠️ weaker than E1 — gradual/modest (~0.65→~0.74), never steep, **never reaches 0.80 anywhere** (global max leaked-vs-not 0.7477, leaked-vs-approp 0.7716; 0 cells ≥0.80). E1's "steep within ~10 tokens" is false (k=10 ≈ 0.63) |
| Sanity delta is a problem | ✅ not a problem — benign nondeterminism |

**Needs more work:** (a) the two missing pos-42 pairwise AUCs (Lambda raw acts); (b) the
leak "climb" is an *uncontrolled upper bound* mixing decision with transcript — the
forced-prefix control (E2, `forced_prefix_f.py`) is required before attributing any of it
to crystallization. Since the uncontrolled climb never crosses 0.85, the controlled
version will be lower.

---

## 8. Recommended figure set for the paper

1. `position_sweep_pos0_depth_f.png` — **hero.** Deflection peaks at L20 (0.89); leak
   plateaus ~0.65–0.69. Carries "deflection decided, leak not" *and* justifies L20.
2. `position_sweep_threeway_f.png` — cleanest statement of "leak-vs-appropriate is the
   hard contrast; deflection is its own cluster."
3. `position_sweep_trajectory_f.png` — leak never crystallizes across the whole response.
4. `position_sweep_heatmap_f.png` — comprehensive supplementary overview.

Lead with 1+2; **drop all "pos 42" language.**

---

## 9. Independent findings (not in the original ask)

1. **Leak never crystallizes — now a hard bound, not a vibe.** Across 5 layers × 65
   positions × the full unfolding (leak-inflated) transcript, leak decodability **never
   reaches 0.80** (max 0.77). By our own E1 rule (>0.85 = crystallizes), the leak
   decision is *never made anywhere*. Strongest support yet for "**leakage is a default,
   not a decision**" — stronger than session 06, because even the transcript-inflated
   ceiling fails the bar.
2. **"Not-leaked" is not one cluster → `leaked_vs_not` is mildly inflated.** Deflection
   is *farther* from appropriate (0.92) than from leaked (0.88) — a distinct mode, not a
   flavor of appropriate. Collapsing refused+appropriate (L5) merges an easy-to-separate
   cluster with the overlapping one, which is why leaked_vs_not (0.68) > leaked_vs_approp
   (0.65). **Recommendation: make `leaked_vs_approp` the canonical leak contrast** in the
   2×2; leaked_vs_not overstates leak decodability. (Candidate **LIMITATIONS L12**.)
3. **Opposite slopes are a mechanistic signature.** Deflection AUC decays over the
   response (−0.0008/tok); leak rises (+0.0008/tok). Reads as deflection = front-loaded
   decision (encoded at the prompt token, not re-represented during execution); leak =
   info that only accrues as the leaky transcript is emitted (byproduct, not plan).
4. **L20 is near-optimal for deflection** — peaks there, declines at L24/L28. Turns the
   "only released checkpoint" limitation into evidence.
5. **Leak signal is depth-invariant in the response region** (k>20: layers 15–28 all
   ~0.65–0.74). Equally-readable-at-every-depth looks lexical/transcript-borne, not a
   deep localized computation → makes the forced-prefix control (E2) load-bearing.
6. **Minor:** deflection has real position-to-position structure (heatmap striping, std
   0.03), likely tracking syntactic boundaries where hedging markers concentrate. One
   sentence, not a headline.

Candidate **LIMITATIONS L11**: pos-42 over-specificity — the argmax is a population smear
(≤1 SE on a plateau; token 42 is a mid-function-word ~38% through), do not report a token
index. (L11/L12 not yet written into `docs/LIMITATIONS.md`.)

---

## 10. Next step

Per METHODOLOGY_f §2 run order, **`forced_prefix_f.py` (E2)** is the correct next Lambda
script — the text-matched control that isolates transcript from decision in the leak
climb. Report (E1 − E2), not E1 alone. Also outstanding: dump leaked-vs-refused and
refused-vs-appropriate at k=42 from `position_sweep_acts_f.npz` on Lambda to finish the
task-6 three-way at the plateau.

## Pipeline status

| Step | File | Status |
|------|------|--------|
| 1–7 | setup … verbalization_survival_f | done (sessions 1–6) |
| — | logit lens | done — negative, L10 (session 7) |
| 8 | `position_sweep_f.py` | **run (Lambda, end of session 7); analyzed this session** |
| — | `analyze_position_sweep_f.py` / `token42_qualitative_f.py` | **done this session** |
| 9 | `forced_prefix_f.py` | **next (Lambda)** |
| 10–15 | minimal_pairs / patchscopes / introspection / alpha_sweep / steering | scaffolded |
