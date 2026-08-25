# Pre-registration — nla-ci headline contrasts

**Dated 2026-08-25. Locked before the next GPU run.**

Everything below is fixed *before* `forced_prefix_f.py` (E2), the extended
position sweep (E2b), and `minimal_pairs_f.py --stage extract/validate` (E3) are
run. Predictions and decision rules are stated in advance so that no outcome can
be reinterpreted after the fact. Results already in hand are marked **[OBSERVED]**
and are locked as reported, not as predictions.

Git-dated: this file's commit timestamp is the registration time. Do not
retroactively edit predictions — append a dated amendment section instead.

---

## 0. The one-sentence claim the paper is built on

> Under an identical labelling pipeline, extraction point, probe family, and
> cross-validation scheme, one behaviour (CI-preserving **deflection**) is
> ~0.89 decodable from the model's internal state *before it emits a token*,
> while the other (**leaking**) never exceeds 0.77 anywhere we looked.

Everything else is support for, or qualification of, that asymmetry.

---

## 1. Locked observed results **[OBSERVED — not predictions]**

Independently re-derived 2026-07-02 (`REPORT.md` §1) and re-run 2026-08-25
(`results/audit/`). Reported to three decimals with Hanley SEs from
`results/stats_hardening_hanley_f.csv`:

| Contrast (L20, prompt-final token) | AUC | Hanley SE | 95% CI |
|---|---|---|---|
| refused vs rest (t3) | 0.890 | 0.037 | [0.818, 0.961] |
| refused vs appropriate | 0.920 | 0.033 | [0.856, 0.983] |
| leaked vs refused | 0.876 | 0.026 | [0.825, 0.927] |
| leaked vs not (t3) | 0.684 | 0.032 | [0.621, 0.746] |
| **leaked vs appropriate (canonical, L12)** | **0.651** | 0.036 | [0.580, 0.722] |

- Deflection layer trajectory: 0.524 (L10) → 0.744 (L15) → **0.890 (L20)** →
  0.870 (L24) → 0.815 (L28). Peaks at the extraction layer.
- Leak ceiling: **0/650 leak cells reach 0.80** across the sweep; max 0.772.
  Qualified by **L15** — window is the first 64 of a median-111-token response.
- Deflection reaches ≥0.80 in 125/325 cells.

### 1a. Triad paired deltas **[OBSERVED 2026-08-25 — supersedes the raw-AUC reading]**

From `results/stats_hardening_f.csv` (5000 paired bootstrap replicates; the three
probes are evaluated on the same scenarios, so the CI on the difference is the
correct object, not the overlap of marginal CIs):

| Contrast | acts − input | desc − input | acts − desc |
|---|---|---|---|
| leaked vs not | **+0.103** [+0.026, +0.178] p=.009 | +0.025 [−0.065, +0.121] **n.s.** | +0.077 [+0.001, +0.151] p=.046 |
| leaked vs appropriate | **+0.104** [+0.016, +0.195] p=.020 | +0.051 [−0.050, +0.154] **n.s.** | +0.053 [−0.031, +0.138] **n.s.** |
| refused vs appropriate | **+0.297** [+0.197, +0.403] p<.001 | **+0.138** [+0.006, +0.264] p=.042 | **+0.159** [+0.083, +0.246] p<.001 |

**This settles `REPORT.md` T1 and is binding on the paper's framing:**

- The **leak** row cannot carry a "verbalization destroys the signal" claim.
  `desc − input` is **not distinguishable from zero** (p=.59 / p=.33) — the
  description channel never rose above the input baseline in the first place, so
  there is no demonstrated collapse. What *is* established on the leak row is
  that activations carry privileged signal over the input (+0.10, p<.05) which
  the descriptions do not detectably recover.
- The **deflection** row is where channel loss is demonstrated: all three deltas
  are significant, with descriptions retaining roughly half the privileged signal
  (+0.138 of +0.297) and losing the rest (−0.159, p<.001).
- **Binding instruction:** deflection is the demonstration of *partial survival
  through verbalization*; leak is the demonstration that *there was little
  privileged signal to lose*. Report deltas with CIs, never bare AUCs.

---

## 2. E2 — forced-prefix control

**Question.** Is the leak "climb" across response positions internal decision
state, or transcript-reading?

**Prediction (H-default, favoured).** With text held identical across scenarios,
AUC stays near the k=0 baseline (~0.65 canonical / 0.68 collapsed) at every
prefix position, on all three prefixes. Predicted E1 − E2 is **small and flat**.

**Basis.** `scratch/03` free preview: a TF-IDF text baseline climbs 0.58→0.64
over k=0→64 while the best activation AUC moves 0.69→0.74, i.e. the
activation-over-text increment is roughly **constant (+0.08–0.10) at every
position**. There is no position at which activations suddenly know something the
text does not.

**Decision rules:**

- Flat within ±0.03 of baseline across all three prefixes → **"no hidden decision
  forms mid-response."** The null framing strengthens; E1's climb is reported as
  transcript-borne.
- A monotone climb ≥0.05 above baseline on ≥2 of 3 prefixes → **H-decision**:
  leak crystallizes but late. This *replaces* the null framing, and E2b (§3)
  becomes mandatory rather than optional.
- Prefix-specific divergence (one prefix only) → treat as a prefix artifact,
  report all three, claim nothing.

**Committed in advance:** either outcome is publishable and will be reported.
A flat result will **not** be described as a failed experiment.

---

## 3. E2b — extended / relative-position sweep

**Question.** Does the leak ceiling hold across the *whole* response, or only the
first 64 tokens? (Closes **L15**.)

**Prediction.** Leak decodability remains <0.80 at relative positions
{10, 25, 50, 75, 90}% and at the final token.

**Decision rules:**

- Still <0.80 everywhere → headline upgrades to **"never, anywhere in the entire
  response"** and the L15 qualifier is retired.
- Crosses ~0.85 at late positions only → **new positive finding**: leak becomes
  readable only *after* the disclosure is emitted (self-knowledge in hindsight),
  not before it is committed. Reported as such — this is a change of story, not a
  patch to the existing one.
- Interpretive caveat fixed in advance: given `scratch/03` (full-response text
  AUC 0.749 ≈ the activation ceiling), late-position decodability is
  **presumptively transcript-reading** and must be reported with that caveat
  unless E2 independently shows text-controlled climb.

---

## 4. E3 — minimal pairs / `v_privacy`

**Prediction.** Held-out secret-vs-public activation AUC ≥0.90; privileged delta
vs the 0.956 text probe ≈0; **CONFOUND FLAG fires**.

**This is predicted, not feared.** Full reasoning, forbidden/permitted claim
language, the marker-family split protocol, and the GATE-2 checklist are fixed in
`docs/E3_CLAIM_LANGUAGE.md` — that file is part of this registration by
reference.

**The registered claim** is the dissociation, not the level:

> privacy status ≈0.95 vs leak behaviour 0.65, same extraction point, same probe
> family, both probes seeing the same words.

**Decision rules:**

- Activation AUC ≥0.90 + per-pair cosine consistency ≥0.80 → proceed to
  `--stage project`; report the dissociation.
- Per-pair cosine consistency <0.80 → `v_privacy` is a bundle, not a direction.
  **Blocks the 2×2.** Report as a negative methodological result.
- Extraction cross-check failure → pipeline bug; fix before anything else.
- CONFOUND FLAG alone → **does not block**, per §1 of the claim-language file.

---

## 5. E-NLA — verbalizing the deflection direction

**Question.** Is the strongest behavioural direction in the repo describable in
words, and does its description contain CI content or generic caution?

**Prediction.** Genuinely uncertain — registered as three-way, with all three
outcomes committed as reportable in advance:

- (a) Caution/hedging/decline vocabulary → first NL description of a
  behavioural-decision direction; then compare CI-specific vs generic-caution
  vocabulary to address T2.
- (b) CJK/code junk → the direction is off-manifold like the leak diff; reported
  as a channel limitation with L7/L9 mechanics, **not** as absence of content.
- (c) Privacy-specific language → best case; strongest single result in the paper.

**Pre-committed guard against outcome (b) being over-read:** per L9, the NLA is
direction-only and normalizes to `inj_scale=150`, so junk output is a statement
about manifold position, never about magnitude or "how much" is encoded.
Temperature must be 0 (L9 §3) — existing NLA runs used temperature 1.0 and are
stochastic samples.

---

## 6. E-EVAL — eval-awareness confound (`REPORT.md` T2)

**Question.** Is the deflection direction CI content, or the model detecting that
it is being tested?

**Prediction.** Deflection AUC survives erasure of the eval-awareness direction
with a drop of <0.10 (i.e. remains ≥0.80).

**Decision rules:**

- Survives (≥0.80 post-erasure, cos(v_eval, v_deflect) low) → T2 substantially
  closed; state it explicitly in the limitations.
- Collapses (<0.70 post-erasure, or high cosine) → **reportable finding in its own
  right**: "CI-deflection is eval-awareness in costume." This weakens the CI
  framing and will be reported rather than buried.
- Registered in advance because this is the single result most capable of
  undermining the paper's framing, and must not become a file-drawer item.

---

## 7. Contrasts explicitly NOT registered (will not be reported as findings)

- **Per-info-type AUCs.** Between-type SD (0.10) ≈ the n=30 Hanley SE (0.10).
  Descriptive only.
- **Sweep argmax / "position 42."** Post-hoc selected from 975 correlated AUCs
  (**L11**). Retired.
- **Tier-4 numbers.** Live label bug (**L14**, ID 495) plus failed t3→t4 transfer
  (AUC 0.41). Either fixed and re-derived, or dropped entirely.
- **Logit-lens Chinese-token "theme."** Cherry-pickable against random-token
  background (**L10**). Explicitly not evidence.
- **Nonlinear probe on leak.** Already answered and null (`scratch/02`); cited,
  not re-run.

---

## Amendments

*Append dated entries below. Never edit the above after the registration commit.*

(none)
