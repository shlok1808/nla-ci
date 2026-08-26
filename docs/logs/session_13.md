# Session 13 — 2026-08-25 (evening, Lambda) + 2026-08-26 (analysis)

**The GPU session.** All four pre-registered experiments ran on one A100-SXM4-40GB
and every one landed on its pre-registered outcome. Total GPU time under an hour;
the estimates in the scripts' docstrings (~30–40 min each) were 20–50× too
conservative — extraction is one forward pass per scenario with no generation, so
233 minimal pairs took **14 seconds**, not 30 minutes.

Raw logs: `results/logs_lambda/`. Figures: `results/figures/`.

---

## 1. E2b — relative-position sweep (closes L15)

```
max leak AUC anywhere = 0.7443      leak cells >= 0.80 : 0/70
deflection max        = 0.9167      deflection >= 0.80 : 10/35
```

**Pre-registered outcome: "still <0.80 everywhere."** L15 retires. The headline
upgrades from *"leak decodability never exceeds 0.77 within the first 64 response
tokens"* to **"never approaches 0.80 anywhere in the response."**

Layer-20 canonical curve:

| k0 | 10% | 25% | 50% | 75% | 90% | last |
|---|---|---|---|---|---|---|
| 0.652 | 0.576 | 0.700 | **0.743** | 0.680 | 0.584 | 0.685 |

Two things worth writing up beyond the null:

- **The peak sits inside the old window.** 50% of a median-111-token response is
  ~token 55. The worry behind L15 was a late crystallization the sweep was
  missing; there isn't one, and decodability *declines* after the midpoint. The
  hole was empty.
- **The final token ticks back up** (0.584 → 0.685). Candidate end-of-sequence
  aggregation. Not investigated; not claimed.

**Extraction verified consistent:** k=0 reproduces session 08 to four decimals
(0.6522 vs 0.652, 0.6865 vs 0.687). The raw max-abs delta printed 0.7500, which
looks alarming but is 2.2% of the largest activation value (33.8) on a box
running torch 2.13 against a reference extracted on an older stack. The AUC match
is the real check and it is exact.

## 2. E3 — minimal pairs → `v_privacy` (the keystone)

Every pre-registered prediction hit:

| check | predicted | observed |
|---|---|---|
| secret-vs-public AUC | ≥0.90 | **0.935 ± 0.011** |
| per-pair cosine | median>0, ≥80% positive | 0.418, **98.7%** |
| length-only AUC | ~0.52, passes | **0.520** |
| privileged delta | ≈0, CONFOUND FLAG fires | **−0.020**, fired |
| extraction cross-check | ~1.0 | **0.9999** (224 same-parser ids) |

`VERDICT: PASS w/ CONFOUND FLAG (expected by design, L13 — does NOT block).`

**The L16 handling worked.** 224 same-parser pairs at cos 0.9999; the 9
parser-divergent ones at 0.9853–0.9945 — visibly lower, correctly excluded from
the bar and reported per-id rather than silently dragging the median.

**The load-bearing claim is now measured:** privacy status **0.935** vs leak
behaviour **0.651**, identical extraction point, probe family, and CV scheme, both
probes seeing the same words. `cos(v_privacy, v_leak) = −0.171` — near-orthogonal.

Note `||v_privacy|| = 3.183`, *smaller* than the behavioural diff's 4.0 (the old
code comment guessed larger). This is exactly why the E-NLA rewrite solves for
rotation angle rather than using fixed alphas — the old grid would have
under-rotated this direction at every step and produced a geometry artifact
indistinguishable from "nothing happens" (the session-04 mistake).

## 3. E2 — forced prefix (text-matched control)

With the text held identical across every scenario:

```
leaked_vs_approp   pos0 0.650 | range 0.588–0.714, mean 0.660
refused_vs_rest    pos0 0.889 | range 0.770–0.927, mean 0.894
prefixes showing a >= +0.05 climb: 0 of 3 (both targets)
```

**Pre-registered H-default.** Leak decodability does not rise when the words are
held constant, so **the mid-response bump in E2b is transcript-borne, not a
decision forming.** The null framing strengthens exactly as registered.

**Honest scope limit:** the prefixes are 3 / 14 / 22 tokens, so this controls the
first ~22 positions. E2b's bump is at ~token 55. Strong evidence about early
positions, *suggestive* about the bump — say so in the paper. A longer neutral
prefix would close it.

## 4. E-EVAL — eval-awareness control (`REPORT.md` T2)

```
cos(v_eval, v_deflect)             = -0.0881
deflection before erasure          = 0.8898
after erasing v_eval (1-D)         = 0.8906
after erasing rank-3 span          = 0.8915   (span = 87.5% of frame-diff variance)
after erasing a random direction   = 0.8898   [control]
per-scenario cosine consistency    = 1.000
```

**Deflection is untouched** — the change is zero to three decimals and identical
to erasing a random direction. It survives both the registered 1-D erasure and
the stronger rank-3 subspace erasure added in session 12.

`frac_pos_cosine = 1.000` means all 270 scenarios shifted the same way under the
framing manipulation, so `v_eval` is a real coherent direction. The manipulation
worked; it simply does not touch deflection.

**Two caveats that stay.** The cosine is near zero to begin with, so the erasure
was never a demanding test — this is partly geometry, not only evidence. And the
manipulation is **explicit**; it bounds the "told it's a test" variety, not
"inferred it from the scenario's style." Prereg §6 already forbids claiming more.

---

## 5. NEW RESULT — the 2×2 is null, and slightly backwards

Not pre-registered as a directional prediction, and the most interesting thing in
the session.

The hypothesis behind the 2×2 was that **weak** privacy encoding predicts
leaking. It does not:

```
AUC(-proj_paired -> leak), leaked vs appropriate : 0.421  95% CI [0.339, 0.503]
AUC(-proj_raw    -> leak)                        : 0.540  95% CI [0.456, 0.623]
excluding the 9 parser-divergent (L16)           : 0.411
leak rate | strong encoding 67.0%   weak encoding 60.4%   diff +6.6pp
chi-square on the 2x2: chi2 = 0.68, p = 0.41
```

Both CIs include chance **and the two projection variants disagree in sign** —
the signature of no relationship. If anything the trend runs the *opposite* way
(more encoding, marginally more leaking), and it is not significant.

**What this means.** The privacy-awareness gap is **not an encoding-strength
gap.** The model does not leak because it encoded privacy faintly on that
scenario; scenarios where it encodes privacy most strongly leak just as often.
Knowing harder does not help. This is consistent with the near-orthogonal
`cos(v_privacy, v_leak) = −0.171`: two independent quantities in the residual
stream.

**Consequence for the paper.** The original four-quadrant framing is dead —
"knows but leaks anyway" vs "genuine blindspot" were meant to be separable
categories, and the split does not track behaviour, so the quadrants are not
meaningful groups. The replacement is simpler and stronger: **near-ceiling
knowledge, weak behavioural signal, and no coupling between them.** Report the
null with its CI and the chi-square; do not present the 2×2 table as though the
cells mean something.

---

## 6. Figures — `scripts/make_figures_f.py` → `results/figures/`

Four figures, every number read from the committed result CSVs (nothing
hardcoded), `.pdf` for LaTeX and `.png` for preview. Palette is
colourblind-safe and validated (blue/orange/violet, worst adjacent CVD ΔE 24.7
protan / 33.6 normal, all ≥3:1 contrast). **Colour follows the entity across all
figures** — leak always blue, deflection always orange, privacy always violet.

| file | shows | suggested caption |
|---|---|---|
| `fig1_asymmetry` | (a) relative-position sweep L20, (b) forced prefix | Deflection is decodable at ~0.89 before the model emits a token and stays there; leaking never approaches 0.80 anywhere in the response, and does not rise when the emitted text is held constant. |
| `fig2_dissociation` | privacy 0.935 / deflection 0.891 / leak 0.652 with 95% CIs | Under an identical extraction point, probe family, and CV scheme, the model's representation of *whether information is private* separates at 0.935 while its representation of *what it will do about it* separates at 0.652. |
| `fig3_null_coupling` | (a) projection densities by behaviour, (b) leak rate by encoding quintile | Privacy-encoding strength does not predict leaking (AUC 0.421, CI includes chance; χ²=0.68, p=0.41). The awareness gap is not an encoding-strength gap. |
| `fig4_layer_trajectory` | deflection/leak AUC by layer at the prompt-final token | Deflection decodability peaks at layer 20, the layer fixed by the available NLA checkpoint — the extraction point is a feature, not a limitation. |

Two figure-integrity decisions worth remembering:

- **Fig 1(b) draws one line per prefix rather than a mean.** The prefixes are 3 /
  14 / 22 tokens, so an average would silently change the underlying set at each
  x — past position 3 only two prefixes exist, past 14 only one. Same
  population-shrinkage trap the relative sweep was built to avoid.
- **Fig 3(a) is density-normalised, not counts.** 128 leaked vs 73 appropriate
  means raw counts would make leaked look larger everywhere purely from class
  imbalance.

---

## 7. Infrastructure notes (cost real time; fixed)

- **`setup_lambda.sh` assumed a torch version that was not there.** The image
  shipped torch 2.7.0+cu128; the script printed that, then `pip install sglang`
  silently upgraded torch to **2.13.0+cu130**, so the hardcoded
  `torchvision==0.26.0/cu130` pin was matched to a torch that no longer existed →
  `operator torchvision::nms does not exist` → `Qwen2ForCausalLM` unimportable.
  Compounded by **two** torchvision installs, the second an apt copy under
  `/usr/lib/python3/dist-packages` that took over when the pip one was removed
  and produced an identical traceback. Fixed in commit `2408aba`: remove
  torchvision entirely (every experiment here is text-only) and remove both
  copies. ~20 minutes lost.
- **matplotlib on the box is built against numpy 1.x** while numpy 2.x is
  installed, so every in-script plot call failed with `_ARRAY_API not found`.
  Harmless — all CSVs wrote correctly and the figures are generated locally.
- **37.5% packet loss** to the instance (the dashboard's "Degraded performance"
  banner was accurate) capped transfer at ~60 KB/s. Pulled the result files with
  `--max-size=30m` and left the two large activation stores behind; they are
  deterministic output from committed code and regenerable in ~2 min of GPU time.

---

## 8. State / next

**Done:** all four GPU experiments, the 2×2, four paper figures.

**Local and free, next:**
1. Blinded reader (`scripts/blinded_reader_f.py`) — the parked NLA-description
   thread, ~$0.15. See `docs/RUNBOOK.md` Part B.
2. Second-judge robustness pass on the 270 tier-3 labels (~$2) — every deflection
   claim rests on 36 positives from one GPT-4o-mini pass, and T3 ("your label is
   the ceiling") is the reviewer attack the audit ranked highest. Still unscheduled.

**Needs a GPU session:** E-NLA. The decision tree resolves to the **simple
branch** — E-EVAL survived and E3 passed, so sweep `v_deflect` (primary) and
`v_privacy` (secondary); no need to add `v_eval` or orthogonalise.
Run `verbalize_directions_f.py --dry-run` first.

**Still open:** tier 4 fix-or-drop (**L14**); `HANDOFF.md` §2/§3 still describe
pre-audit state in places.
