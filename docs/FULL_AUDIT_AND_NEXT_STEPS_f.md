# Full audit and next steps — canonical labels and probes

**Date:** 2026-09-01. **Scope:** independent read-only audit of the canonical tier-3
label set and the canonical probe contrasts, plus a prioritized plan for the paper.
No APIs, judges, model inference, or GPU jobs were run. New analysis code:
`scratch/audit_canonical_consistency_f.py`, `scratch/audit_stats_hardening_f.py`,
`scratch/audit_text_baseline_canonical_f.py` (all seeded, local, read-only;
`results/` untouched).

---

## 1. Verdict: **CONDITIONAL GO**

The data layer is clean: every count, hash, join, and derived label verifies
independently, the probe pipeline has no train/test leakage, and the three headline
contrasts survive multiple-comparison correction. Nothing found requires
withdrawing the canonical-label work.

The conditions, all local-CPU and fixable in hours:

1. **The reported AUC point estimates are biased low by a protocol artifact**
   (§2.1) — fix the pooling before any number goes in the paper. The headline
   `leak_vs_appropriate` is ~0.80, not 0.756, under a corrected protocol.
2. **The `_lo/_hi` columns are not confidence intervals** and must never be
   printed as such (§2.2). Replace with Hanley or scenario-bootstrap CIs.
3. **Only three contrasts are defensible after Holm correction under the
   pre-registered primary metric** (§2.3). `substantive_leak` and `broad_breach`
   must be demoted to suggestive; `degree_boundary` is a null.
4. **The judge prompt told the model its references were "HUMAN-VERIFIED" when
   216/258 were machine-drafted and unverified** (§2.5). Not a numeric error —
   the manifest tracks it honestly — but the paper must disclose it and must not
   describe analysis references as verified.

---

## 2. Audit findings

### 2.0 What verified clean (all checks PASS)

`scratch/audit_canonical_consistency_f.py`, 40+ checks:

- **Arithmetic:** 270 = 258 + 12 excluded; 258 = 42 calibration + 216 analysis. ✅
- **Manifest SHA-256 hashes** match both the canonical CSV and JSON byte-for-byte. ✅
- **JSON ↔ CSV ↔ manifest agreement:** identical scenario-ID sets, identical
  `max_disclosure_level` and `response_strategy` per scenario. ✅
- **Label derivations internally consistent:** `substantive_leak ⇒ broad_breach`;
  `label_substantive` partitions exactly as {leaked=83, broad_only=141,
  appropriate=27, refused=7}; onset chars present iff the corresponding breach. ✅
- **Join integrity:** NPZ scenario IDs unique (496), all 258 canonical IDs present;
  the probe script reads only `scenario_ids` + `activations` and hard-exits on
  mismatch. The stale NPZ `labels` array is never touched — and the crosstab
  against canonical labels shows it disagrees on a large fraction of rows, so
  ignoring it is load-bearing and correct. ✅
- **Every contrast's n / n_pos reconstructs exactly** from the canonical CSV,
  including the refusal skip (4/216 and 7/258 minority < 15). ✅
- **Every ROC-AUC in the summary matches the results file** to 3 decimals. ✅
- **Probe leakage:** none. StandardScaler and C-selection live inside training
  folds (Pipeline + nested GridSearchCV); permutation nulls run through the same
  fitting path; no activation, label, or preprocessing statistic crosses a fold
  boundary. ✅
- **Grid-floor C selections are benign** (§2.4). ✅

### 2.1 Reported AUCs are depressed ~0.04 by pooling scores across folds with heterogeneous C

The protocol pools raw `predict_proba` outputs across outer folds into one AUC.
Nested selection is unstable at these n (chosen C spans 1e-7 to 1.0 *within a
single repeat*), and probability scales differ across C by orders of magnitude, so
the pooled ranking is corrupted at fold boundaries. Reproduced exactly
(`leak_vs_appropriate|216`, seeds `BASE_SEED..+4`):

| protocol | mean AUC |
|---|---|
| as published (nested C, raw-score pooling) | **0.760** (matches the reported 0.756) |
| same folds, fixed C=1e-7 | **0.801** |
| nested C, per-fold rank-normalized before pooling | **0.808** |

One repeat dropped to 0.648 purely from one bad inner selection. This is the
known pooled-`cross_val_predict` AUC pitfall (Forman & Scholz 2010, *SIGKDD
Explorations*; Airola et al. 2011). Consequences:

- Published point estimates are **conservative** — the significant results are, if
  anything, stronger. Nothing is inflated.
- The fix is local and cheap: report the **mean of per-fold AUCs** (or rank-pool
  per fold) instead of pooled raw scores. Expect `leak_vs_appropriate|216` ≈ 0.80,
  `limiting_vs_direct` ≈ 0.72–0.74, `limiting_among_disclosers` ≈ 0.73.
- `modal_C` and `at_grid_edge` are noisy summaries of a wildly varying per-fold
  selection; don't interpret them as "the probe wants C=1e-7."
- The permutation nulls used a *fixed* modal C (homogeneous pooling), so the null
  was clean while the observed statistic was depressed → p-values are valid and
  slightly conservative. Significance claims stand.

### 2.2 The `_lo/_hi` bands understate uncertainty 1.3–2.1× — they are not CIs

The bands are 2.5/97.5 percentiles over 20 repeated-CV runs **on the same 216
scenarios**. Repeats differ only in fold assignment, so the band measures split
noise, not sampling uncertainty. Answer to the audit question "is repeated-CV
uncertainty overstated because repeats are correlated?" — **no, the opposite**:
correlated repeats make the bands far too *narrow* as uncertainty estimates
(Nadeau & Bengio 2003; Bates, Hastie & Tibshirani 2023, *JASA*). Against
Hanley–McNeil 95% intervals:

| contrast (216) | ROC | reported band | Hanley 95% | ratio |
|---|---|---|---|---|
| leak_vs_appropriate | 0.756 | ±0.076 | **±0.107** | 1.4× |
| limiting_vs_direct | 0.703 | ±0.050 | **±0.075** | 1.5× |
| limiting_among_disclosers | 0.714 | ±0.047 | **±0.078** | 1.7× |
| broad_breach | 0.658 | ±0.052 | **±0.099** | 1.9× |
| degree_boundary (258) | 0.587 | ±0.038 | **±0.078** | 2.1× |

**Fix:** report Hanley SEs or a scenario-level bootstrap as the CI; keep the
repeat band only as "split variability," clearly labeled. Any sentence currently
reading the brackets as a 95% CI is an overstated-precision claim.

### 2.3 Multiple comparisons: the defensible set is exactly the three headliners

Holm–Bonferroni over the six scored contrasts in the primary 216 population:

- **Primary metric (PR-AUC, as declared in the script docstring):**
  `leak_vs_appropriate`, `limiting_vs_direct`, `limiting_among_disclosers`
  (all p=.002, the permutation floor of 1/501) **survive**;
  `broad_breach` (p=.034) and `substantive_leak` (p=.048) **fail**;
  `degree_boundary` (p=.23) null.
- Under ROC-AUC the first five survive, but PR-AUC was pre-specified as primary —
  switching metrics post hoc to rescue two contrasts would be outcome shopping.
- `substantive_leak`'s PR lift over prevalence is only +0.087 (0.351 vs 0.264) —
  weak on the primary metric even before correction.

**Defensible confirmatory set: the three headliners.** Report `substantive_leak`
and `broad_breach` as suggestive/exploratory with unadjusted p, and
`degree_boundary` as a null result.

### 2.4 Grid-floor C selections do not undermine anything

Below-floor check (216 population, 5-fold, 3 seeds): AUC is flat from C=1e-9
through 1e-7 (`leak_vs_appropriate` 0.808 at every value) and **exactly equals a
standardized difference-of-means probe** (0.808). As C→0, regularized logistic
converges to a scaled class-centroid difference; the ranking saturates, so the
optimum being "outside the grid" is moot — the C→0 limit is already achieved.
Same pattern for `substantive_leak` (0.648) and `limiting_vs_direct` (0.718).
Side benefit worth one paper line: the decodable signal is essentially the
class-mean difference direction, not a high-capacity decision boundary.

### 2.5 Named misstatements and overstated claims

1. **"HUMAN-VERIFIED REFERENCE" in the judge prompt is false for 216/258 cases.**
   `behavior_annotation_run_f.py:101` force-sets `verified_by_human=True` in
   provisional mode to pass `validate_reference`, and the system prompt asserts
   human verification. The CSV/manifest record `provisional_unverified` honestly.
   Must be disclosed in LIMITATIONS and the paper; consider whether the judge's
   deference to a "verified" reference sheet shaped annotations.
2. **Any "95% CI" reading of the `_lo/_hi` columns** — see §2.2.
3. **"258 sensitivity" numbers must never be read as the better estimate or as a
   replication.** 258 ⊃ 216 (fully overlapping), and the added 42 calibration
   cases are a *biased* sample: 66.7% limiting and 62% leaked vs 35.2% and 26% in
   the analysis population (they oversampled old `refused`), with cleaner
   human-verified references. The higher 258 AUCs (leak 0.796, limiting 0.743)
   are plausibly composition + label-cleanliness artifacts. "Sensitivity
   analysis" is the correct term; keep 216 primary everywhere.
4. **"degree_boundary is marginal in the 258 population"** — p_roc=.034
   unadjusted, in the enriched population, failing every correction. Call it what
   it is: no evidence that disclosure degree is encoded.
5. **The old refusal/deflection AUCs (0.89 vs-rest, 0.92 vs-appropriate) and every
   number derived from the old judge labels** — formally superseded; see claim D
   in §3. The July audit (`REPORT.md`) reproduced them faithfully from the files;
   reproducibility is not validity.
6. Minor: the contrast is named `degree_boundary_broadonly_vs_leaked` in the
   artifacts but `degree_boundary` in summaries — keep one name.

### 2.6 New supporting evidence from this audit (use it)

**Scenario-text baseline** (`scratch/audit_text_baseline_canonical_f.py`; TF-IDF
1–2-grams + logistic, 5-fold, 5 seeds, 216 population):

| contrast | scenario text | activations | privileged Δ |
|---|---|---|---|
| limiting_vs_direct | **0.546** | 0.703 (~0.72 corrected) | **+0.16** |
| limiting_among_disclosers | **0.535** | 0.714 (~0.73 corrected) | **+0.18** |
| leak_vs_appropriate | 0.678 | 0.756 (~0.80 corrected) | +0.08–0.12 |
| substantive_leak | 0.593 | 0.619 | +0.03 |
| degree_boundary | 0.581 | 0.557 | −0.02 |

The limiting contrasts are the paper's cleanest "privileged internal state"
result: the scenario text barely predicts whether the model will hedge/limit,
while the pre-response activation does. For leak_vs_appropriate about half the
signal is already in the scenario. (TF-IDF is a weak text encoder — say so; a
frozen-LLM-embedding baseline would firm this up, ~1h local.)

**No degree confound in the limiting contrast:** among analysis-population
disclosers, limiting rates are flat across disclosure level (38% / 44% / 33% for
existence / explicit / implication) and across substantive-leak status (39% vs
38%). Conditioning on disclosure controls presence, and empirically the degree
composition is balanced too.

---

## 3. The five proposed claims, evaluated

**A. "Limiting behavior is decodable among responses that disclosed,
independently of disclosure content." — SUPPORTED, with two wording fixes.**
Survives Holm on the primary metric (0.714, p=.002), scenario-text baseline near
chance (0.535), degree composition balanced. Fixes: (i) 97 of 104 limiting cases
are `mixed_disclose_then_limit` (6 soft_deflection, 1 explicit_refusal), so state
the construct as *"will disclose then limit" vs "will disclose directly"* — this
is not the old deflection construct; (ii) say "not attributable to disclosure
presence or degree" rather than "independently of disclosure content" — topical
content wasn't controlled (the near-chance text baseline is your best evidence
there; cite it).

**B. "Substantive leak versus appropriate response is the strongest behavioral
contrast." — SOFTEN.** It has the highest point estimate (0.756/0.80), but
Hanley ±0.107 at n=81 overlaps every other significant contrast; no formal
comparison was run and none is powered. Also name the population: 57 clear leaks
vs 24 appropriate — it excludes the 131 `broad_only` and 4 `refused` cases, i.e.
37% of the analysis population, and the negatives (appropriate) are only 24
provisional-labeled cases. Say "highest AUC among contrasts (0.80, 95% CI
±0.11); differences between contrasts are not statistically resolved."

**C. "Activations encode disclosure presence better than disclosure degree." —
PARTIALLY SUPPORTED; state asymmetrically.** The degree null (0.557, p=.17) is
solid as a null. But the presence side is carried by `leak_vs_appropriate`;
pooled `substantive_leak` (0.62) and `broad_breach` (0.66) fail Holm on the
primary metric, and no presence-vs-degree comparison was tested. Honest version:
"clear leaks are decodable against appropriate responses, and upcoming strategy
is decodable, but we find no evidence the *degree* boundary (broad-only vs
substantive) is represented." Exploratory: chosen after inspection.

**D. "The old refusal/deflection AUC ≈0.89 was inflated by contaminated labels
and must be withdrawn." — SUPPORTED, but "inflated" is the wrong mechanism.**
Withdraw it, yes — but the finding is stronger than inflation: the *construct
dissolved*. ~25 of 36 old `refused` cases also disclosed (session 16); under the
evidence-span rubric the class collapsed 36→7 and is unmeasurable (skipped at
4/7 minority). The old 0.89–0.92 measured predictability of a judge artifact.
The defensible successor is `limiting_vs_direct` at ~0.72 — decodable, but far
from 0.9. Frame as a measurement-validity correction with the before/after
numbers; that narrative is itself a contribution.

**E. "Leakage is a default, not a decision," supported by the weak pooled
substantive-leak probe. — REJECT in this form.** The claim was built on the old
asymmetry (deflection 0.89–0.92 vs leak 0.65–0.68). Under canonical labels that
asymmetry has largely dissolved: leak_vs_appropriate ≈0.80 vs limiting ≈0.72–0.73
— the *outcome* contrast is now numerically the strongest. A weak pooled
substantive_leak (0.62, Holm-failing) cannot carry a strong cognitive claim, and
its weakness is partly composition (it pools clear leaks against broad_only
disclosures, which sit near the boundary). What survives: "nothing is
near-ceiling at the pre-response point; the degree boundary is not detectably
represented; strategy is encoded beyond what the scenario predicts." Do not
resurrect "default, not decision" unless the onset experiment shows the leak
signal is transcript-borne while strategy is anticipatory.

**Double-dipping warning.** The contrast *set* was specified in code before
results, but the *interpretations* (which contrast to headline, presence-vs-
degree, "strongest") were chosen after inspection. Mark the three
Holm-surviving contrasts as confirmatory (protocol + null pre-specified in the
script), everything comparative or compositional as exploratory. Freeze the
paper's headline claim in a dated commit *before* running the onset experiment.

---

## 4. The token-position / disclosure-onset experiment

**Assessment: right experiment, second in priority after the local fixes.** It is
the only proposed work that adds a genuinely new axis (dynamics) rather than
hardening existing claims — and the canonical onset chars make it feasible for
the first time (the old sweep couldn't align to disclosure). But it requires GPU
re-extraction, and the paper already stands without it. Run it if the local
must-dos (§5) finish by midday.

**Artifact reality check.** The old `position_sweep_aucs_f.csv`,
`relative_position_aucs_f.csv`, `forced_prefix_aucs_f.csv` were computed on stale
labels; the raw multi-layer NPZs (`position_sweep_acts_f.npz` etc.) were
Lambda-only from the June session and are not local. Unless that instance or a
persistent filesystem survives (check the Lambda dashboard — 10 minutes, not
more), recovery is off. Even if recovered, the old store covers only absolute
positions k≤64 (L15) — it can't serve an onset-aligned design. **Plan for
re-extraction; treat recovery as a bonus.** The old CSVs are appendix history
only; never mix them with canonical numbers.

**Scientific question.** Does the internal state that predicts upcoming behavior
exist *before* the disclosure is emitted (anticipatory / decision-like), or does
it only appear as the transcript makes it readable (post-hoc)?

**Estimand.** For each pre-registered contrast: AUC of a layer-ℓ probe at
position p relative to the canonical broad-disclosure onset token, minus the
matched text-baseline AUC at the same visible prefix — the *privileged increment*
Δ(ℓ, p). Primary cell (pre-register it): mean Δ over the pre-onset window
[−16, −1], layer 20, `limiting_among_disclosers`.

**Alignment: onset-relative primary.** Median onset is ~27% into the response
(~32 tokens); 85% of onsets fall within the first 64 tokens, so a [−32, +16]
onset window is well-populated. Absolute position k is a legacy comparison only;
relative response position (10–90%) as a secondary population-level view.
Suggested grid: offsets {−32, −16, −8, −4, −1, 0, +4, +16} plus prompt-final,
layers {10, 15, 20, 24, 28} — yes, include the layer sweep; it's nearly free in
the same pass and answers whether L20 is the right NLA readout point. That's 45
probe cells per contrast: pre-register the one decision cell, present the rest as
a descriptive heatmap (the L11 lesson — no argmax claims, split-half validate
any cell you want to promote).

**Contrasts under the new rubric.** `limiting_among_disclosers` is the ideal
onset-aligned contrast — *both* classes disclose, so both have real onsets and
alignment is symmetric. For `leak_vs_appropriate`, the negatives have no onset:
assign pseudo-onsets to appropriate cases by sampling from the disclosers'
onset-position distribution (stratified by response length), else position itself
leaks the label. `degree_boundary` at onset (does the state at broad-onset
predict later escalation to substantive?) is a nice exploratory cell.

**Controls (all required).**
1. **Text-prefix baseline at every cell:** TF-IDF (and ideally a frozen-embedding
   encoder) on scenario + visible prefix. The claim lives in Δ, not raw AUC.
2. **Forced-prefix control:** re-run the probe on activations from forcing the
   same prefix text — separates "state computed by reading its own words" from
   state that preceded them. Old forced-prefix CSVs are stale; rerun.
3. **Fixed-population analysis:** at each offset, restrict to scenarios whose
   response covers the whole window (report attrition); per-offset class balance
   must be reported — differential truncation is a silent confound.
4. **Char→token mapping verified:** map canonical `broad_onset_start_char` with
   the Qwen tokenizer's offset mapping against the *exact* prompt template of
   `extract_activations.py` (the L16 parser fork is the known trap here); assert
   round-trip slicing on every scenario before extraction.
5. **Same stats protocol as §2:** per-fold-averaged AUC (not raw-score pooling),
   fixed-C or nested-with-rank-pooling, permutation null through the pipeline,
   Holm over pre-registered cells, scenario as the CV unit.

**Decision rule for "crystallization" (pre-register).** Support: pre-onset
Δ(L20, [−16,−1]) > 0 with permutation p<.05 and rising toward onset, while the
forced-prefix control shows no such rise. Reject: pre-onset Δ ≈ 0 and AUC rises
only post-onset in both activations and text (transcript-borne). Either outcome
is publishable; the reject branch *would* partially resurrect a disciplined
version of "leakage happens rather than being decided" — but only for the leak
contrast, and only if strategy shows the anticipatory pattern.

**Drop "token 42"** — already demoted (L11); the onset-aligned design replaces
it wholesale. NLA readout at the best-justified layer/position: keep gated on the
probe outcome (only if pre-onset Δ is real), and read L17 first — the verbalizer
faithfulness problem does not go away at a better position.

**Cost/benefit.** One A100 session: teacher-forced pass over 258 responses,
5 layers × ~9 positions, fp16 store ≈ 258 × 45 × 3584 × 2B ≈ 80 MB — trivially
storable; extraction + probes ≈ 1.5–2.5 h wall-clock, ~$3–5. Verdict: **worth
it** — it converts the paper from "corrected statics" to "statics + dynamics,"
closes L15 with canonical labels, and the design has two publishable outcomes.
But it is strictly after the local fixes: a beautiful onset curve on top of
miscomputed CIs and an undisclosed reference caveat is a rejected paper.

---

## 5. Prioritized plan

### Must do before submission (all local CPU, ~5–6 h total)

| # | action | cost | benefit / claim served |
|---|---|---|---|
| 1 | **Fix AUC pooling** in `probe_contrasts_canonical_f.py` (per-fold mean or rank-pool; keep everything else); re-run both populations | ~1 h CPU | Every headline number; raises leak_vs_appropriate to ~0.80. Cite Forman & Scholz 2010 |
| 2 | **Replace CI machinery**: Hanley SE or scenario bootstrap per contrast; relabel repeat band as split variability | ~1 h | Honest uncertainty on every claim (§2.2) |
| 3 | **Report Holm-adjusted p** over the 6 contrasts; demote substantive_leak / broad_breach to suggestive; degree_boundary as null | 30 min | Confirmatory/exploratory split (§2.3) |
| 4 | **Promote the scenario-text baseline** to `scripts/` + `results/` (per contrast, both populations) | ~1 h | The privileged-increment evidence for the headline claim (§2.6) |
| 5 | **LIMITATIONS entries**: L18 provisional single-judge references + the "HUMAN-VERIFIED" prompt misstatement; L19 calibration-set enrichment (why 258 ≠ replication); L20 repeat-band ≠ CI; formal withdrawal note for old refusal AUCs | ~1 h | Pre-empts the three likeliest reviewer kills |
| 6 | **Freeze the headline claim + onset-experiment decision rule in a dated commit** before any new number is seen | 15 min | Confirmatory status of everything downstream |

### High-value if time permits (ordered)

| # | action | cost | benefit |
|---|---|---|---|
| 7 | **Eval-awareness control on the limiting contrast**: derive v_eval from local `eval_awareness_acts_f.npz`, report cos(v_eval, v_limiting) + limiting AUC after LEACE-style erasure, canonical labels | ~2 h CPU | Closes "the strategy signal is just eval-awareness/RLHF caution" (old T2) for the *new* headline — currently unvalidated |
| 8 | **Onset/token-position experiment** as specified in §4 | 1 GPU session, ~$5 | The dynamics axis; closes L15 |
| 9 | **Stratified author audit of ~40 analysis-population annotations** (blind: hide labels, sample across strategy × level cells, verify evidence spans justify the label) | ~2 h human | Converts "provisional single-judge" from unbounded caveat to a measured error rate — the cheapest reliability number that materially helps (see below) |
| 10 | **Frozen-embedding text baseline** (e.g., a small sentence encoder, local) to firm up §2.6 against "TF-IDF is a straw baseline" | ~1 h CPU | Hardens the privileged-increment claim |
| 11 | **Response-prefix text-only curve with canonical labels** (local; text side of §4 without GPU) | ~1 h CPU | Free preview of the onset experiment; publishable alone if GPU falls through |

### Exploratory / future work

- Cross-family judge re-annotation of a stratified ~60-case sample (API, ~$2–5)
  — quantifies judge-family dependence; report as agreement, not validation.
- Full second-judge or human annotation of all 216 (the real fix for provisional
  references; not a one-day item).
- Onset-conditioned NLA verbalization at the best pre-onset cell (gated on §4
  outcome and L17).
- degree_boundary with more data or ordinal probing (current null may be power).
- Topic/recipient-relationship stratified probes (canonical references contain
  recipient + transmission-principle fields — unexploited local metadata).

### Do not do

- **Steering / alpha-sweep / transmission-pilot reruns** — expensive, off the
  critical path, and the NLA thread is a documented negative (L17).
- **Tier-4 anything** (L14 stands).
- **Recover-the-Lambda-box archaeology beyond a 10-minute dashboard check.**
- **Metric switching or post-hoc contrast additions** after seeing corrected
  numbers — the prereg discipline is now the paper's own methods contribution.
- **Reporting 258-population numbers as primary anywhere.**

### The provisional-references question, answered directly

216/258 references are machine-drafted, single-judge (same model family drafted
and judged), unverified — and the judge was told otherwise. Consequences: (i)
label noise attenuates AUCs (your numbers are more likely under- than
over-estimates); (ii) *systematic* reference errors could create structure a
probe reads — this is the real risk and it is unquantified. Best value per hour:
the **stratified author audit (#9)** — 40 cases with evidence spans already
quotable makes verification fast, and an empirical error rate ("x% of a blind
stratified sample changed label") is worth more to a reviewer than any amount of
hedging. Cross-family judging is second (cheap, but agreement between two LLMs
under the same rubric is weak validity evidence — impact-judge doc already
establishes this). Full human verification is the eventual fix, not a one-day one.

---

## 6. Recommended headline and framing

**Center the paper on the strategy/limiting result, told inside a
measurement-correction narrative:**

> Before generating a word, the model's activations predict *how* it will handle
> a secret — disclose-then-hedge vs disclose-outright (AUC ~0.72–0.73, scenario
> text ~0.54) — and whether a clear violation will follow (~0.80). Nothing
> approaches the near-ceiling encoding previously reported for "deflection":
> that number measured a judge artifact, and correcting the labels with an
> evidence-span rubric collapsed the class it was built on (36→7) and dissolved
> the leak/deflection asymmetry it implied.

Why this framing:

- **The limiting result is the paper's best property bundle:** Holm-surviving,
  near-chance text baseline (the privileged increment Wang et al. do not have —
  they probe norm attributes in a judgment frame, not upcoming behavior in the
  acting frame), no degree confound, and a genuinely novel construct
  (`mixed_disclose_then_limit` — disclosure-limiting as a *strategy*, distinct
  from refusal directions in the literature).
- **The label-correction arc is a contribution, not a confession.** Old judge →
  demonstrated inconsistencies → evidence-span rubric v1.3 with exact-substring
  verification and deterministic label derivation → headline numbers move,
  one construct dissolves, one new one appears. Interp workshops explicitly
  welcome this; it is also the honest inoculation against "your labels are
  LLM-generated" (which is otherwise the paper's soft underbelly).
- **NLA verbalization cannot remain central.** Its results are negative
  (L6/L7/L17: format-dominated descriptions, confabulated pilot, an instrument
  whose training objective doesn't penalize false claims). Keep it as a
  disciplined secondary section — "what a verbalization channel recovers of
  these signals: little, and here is why that is an instrument property" — with
  Dingeto 2026 and Li et al. (ICML 2026) as context. If the venue narrative
  needs one label, this paper is now a *probing + measurement-validity* paper
  with an NLA cautionary appendix, not an NLA paper.
- **Onset dynamics** (if §4 runs and shows pre-onset signal) slots in as the
  second figure: "the strategy signal exists before the first disclosing token."
  If it shows transcript-borne signal instead, it sharpens the leak claim's
  scope honestly. Either way the paper stands.

**Suggested title shape:** *"Probing what an LLM will do with a secret: upcoming
disclosure strategy is encoded before generation — and what corrected labels
change about the picture."*

---

## Appendix: audit artifacts

| file | contents |
|---|---|
| `scratch/audit_canonical_consistency_f.py` | counts, hashes, JSON↔CSV↔manifest, label derivations, NPZ join, contrast reconstruction, AUC verification — all PASS |
| `scratch/audit_stats_hardening_f.py` | population composition, Hanley-vs-band table, Holm tables, below-grid-floor + diff-of-means check |
| `scratch/audit_text_baseline_canonical_f.py` | scenario-text TF-IDF baseline per canonical contrast |

Key methodology references: Forman & Scholz 2010 (pooled CV-AUC pitfall);
Airola et al. 2011 (CV-AUC estimators); Hanley & McNeil 1982 (AUC SE);
Nadeau & Bengio 2003 (correlated CV repeats); Bates, Hastie & Tibshirani 2023,
*JASA* (what CV estimates and how to get CIs). Positioning references as in
`REPORT.md` §4 (Wang et al. arXiv:2604.00209; Li et al. arXiv:2509.13316;
Dingeto arXiv:2607.20379; refusal-direction line arXiv:2507.11878,
arXiv:2510.06036).
