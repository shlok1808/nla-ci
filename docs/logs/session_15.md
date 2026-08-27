# Session 15 — 2026-08-26 (afternoon): the NLA thread is redesigned

Local session. No GPU, no API spend. Session 14 was the external review of the
*judge* experiment; this log covers what happened after — the NLA thread was
cut down, redesigned around natural activations, and gated behind a pilot.

**Where things stand: the 40-pair pilot is built, audited four times, and ready
to run. Nothing has been generated yet.**

---

## 1. Direction verbalization is CUT

`scripts/verbalize_directions_f.py` will not be run. The reasoning is not that
it was badly built (it was rebuilt carefully in session 12 with angle-targeted
rotations and matched-angle controls) but that **it could not answer anything**:

> Even a perfect result says "the activation verbalizer's prose changes when its
> input is rotated toward a behaviour-associated direction." That is a fact
> about the verbalizer, not about Qwen. Under this project's no-causal-claims
> constraint there is no stronger reading available.

Fixing the geometry fixed an artifact, not the experiment. Keep the script for
the record; do not schedule it.

**What survives from it:** the endogenous temp-0 reads (verbalizing real
`refused` and `appropriate` activations with no injection). That idea is
absorbed into the transmission design below.

## 2. The judge replication is demoted to QA

Not a paper experiment. One second-judge pass on the **original rubric only**
(the roleplay-prompt arm is cut — it changed the frame, the interpersonal
assumptions, and the definition of violation simultaneously, so its number would
have needed a paragraph of caveats and advanced nothing). Report agreement and
label sensitivity in the limitations, not as a result.

## 3. The replacement: an NLA transmission experiment

```
real activation ──AV──> English description ──AR──> reconstructed activation
                              │                            │
                   "can a reader tell which          "does the probe
                    came from the secret one?"        still work?"
```

Run on the 233 minimal pairs — the same story told twice, once confidential and
once common knowledge, everything else fixed.

**Why this beats what it replaced:** it uses *natural* activations, so the
off-manifold problem that has dogged the NLA thread since session 04 (L7/L9)
disappears. And it asks the question the paper is actually about — what a
language channel preserves of a model's internal state — instead of a question
about the verbalizer's quirks.

Every outcome is reportable: privacy survives but leak does not (mirrors the
main finding); nothing survives (the channel discards strong signal);
descriptions read well but reconstruction fails (the words are decoration).

## 4. The finding that forced a pilot

The AV is direction-only — it L2-renormalizes every input, so **only the angle
between two activations can affect its output**. Measured on our own pairs:

```
secret vs its own public version : median  4.45 deg   (66% below 5 deg)
secret vs a DIFFERENT story      : median 16.62 deg
```

**L8 records that a ~5 deg rotation produced identical descriptions in session
04.** That is why `verbalize_directions_f.py` targets 10–60 deg. Our pairs sit
at or below that threshold.

Attempted to settle it from existing data — 270 activations and their 270
descriptions give 36,315 pairs to fit a sensitivity curve. Two results:
description similarity does fall as angle grows (Spearman −0.247, p≈0), so the
AV is not angle-insensitive in general; but **no existing pair is closer than
~8 deg**, so the 3.5–5.5 deg band where two thirds of our pairs live contains
no observations at all. The regime is unexplored and cannot be extrapolated
into.

**Important correction, from the external review:** angle is normalized-input
*distance*, **not** a resolution threshold. The AV is nonlinear and its
sensitivity is anisotropic — a small step along a high-sensitivity semantic
direction can change the output while a larger step along an insensitive one
does not. Session 04's null used a synthetic, partly off-manifold perturbation
along a direction that is ~1/3 label noise; both members of a minimal pair are
natural prompt activations. So session 04 **motivates** the pilot without
predicting its outcome.

## 5. The pilot — `scripts/nla_transmission_f.py`

**40 pairs, stratified 10 per within-pair angle quartile** (1.9–9.5 deg), locked
in `results/nla_transmission_manifest_f.csv` with a SHA of every activation.
Stratified rather than random so a NO-GO cannot be an artifact of sampling only
near-identical inputs, and so the result can be read by quartile — uniform
insensitivity looks different from a threshold response.

**Two verdicts, not one.** P2 (the setup paragraph) is expected to echo the
rewrite's changed words. A whole-description difference driven entirely by P2
would be a **false GO** — the number moves while the hypothesis of interest is
dead. So:

- **FULL branch** — does any human-visible private/public information survive?
- **P3 branch (PRIMARY)** — does it reach the AV's *forecast of the upcoming
  reply*?

FULL success with P3 failure is a distinct, reportable result, not a NO-GO.

**The gate is a blinded 2AFC, not a similarity threshold.** But the
discrimination ceiling is computed first:

```
2AFC ceiling = 1 - f_identical/2
```

A perfect reader still scores chance on identical pairs, so if the ceiling is
below the pre-registered **0.65** floor the pilot stops without asking anyone to
judge anything. Pass mark **26/40** — the first one-sided exact-binomial p<.05.

**Where the paragraph structure fits.** Every tier-3 description has exactly
three paragraphs: format boilerplate, setup quote, and a forecast of the
upcoming reply. That structural observation is the project owner's and it is
what makes "where does it survive" answerable at no extra generation cost —
descriptions are produced once and sliced afterwards. The earlier *quantitative*
claim (P3 alone beats the whole description on leak, 0.635 vs 0.600) was
**retracted**: paired bootstrap gives +0.035, CI [−0.033, +0.103], p=0.30, and
the deflection contrast runs the other way at −0.050. P3 enters here as a
prospectively-locked hypothesis on a *different* target (secret vs public), not
as confirmation of the dead one.

Measured, and relevant: **P3 has the highest overlap with the source scenario**
(0.071 vs P2 0.045 vs P1 0.040). So P3 is both the most likely to show a
difference and the most likely to show it for a boring reason — the AV echoing
the changed clause. The pilot reports the edit-vocabulary fraction as a
diagnostic, and the planned ablation (strip the rewrite's changed words, re-test)
is what separates transmission from echo.

## 6. Four audit rounds — ~12 defects, all in code meant to REFUSE

Every bug lived in a guard, where the happy path works and reading the code does
not reveal the defect:

- resume treated `error` rows as complete, so failures never retried
- a verdict could be computed from a single successful pair
- the pilot verdict was computed over every row in a shared CSV, not the sample
- `--stage verbalize` checked only for `PILOT-COMPLETE`; the advertised GO gate
  did not exist
- export dropped identical pairs and scoring ignored unanswered items, so six
  easy items answered correctly gave p=0.0156 while the 26/40 rule was never
  applied
- manifest checksums used Python's `hash()`, which is randomized per process —
  identical bytes gave different values in consecutive runs
- `injection_scale` was accepted unverified though the whole premise depends on
  it being 150
- `--slice full` on verbalize silently used the P3 authorisation
- a test fixture wrote simulated metrics into real `results/` and they were
  committed looking like a completed pilot

**Consequence:** `tests/test_nla_transmission_f.py`, 45 assertions, most of them
refusals. The rule it enforces: *if a function's job is to stop something, there
is a test proving it stops it.* It also asserts the real `data/` and `results/`
directories are byte-identical after the suite.

The final pass added recomputation of the 2AFC from the frozen packet and key
before the paid run (so editing the summary JSON cannot manufacture
authorisation), a provenance chain through verdict → packet → grade → full run,
and live SGLang identity validation that refuses if the server loaded a
checkpoint other than `./actor_hf`.

## 7. Corrections to the record made today

- **The figure-1 title violates the project's own constraint.** "Deflection is
  *decided* before the model speaks" implies causal commitment. Retitle before
  the paper; the defensible form is about linear predictability.
- **A pattern of ours died under a paired test, for the second time.** The
  pre-response activation appeared to beat the full transcript for deflection
  (0.890 vs 0.873) while losing for leak — an attractive story. Paired
  bootstrap: **+0.009, p=0.78** and **−0.043, p=0.27**. Neither survives. The
  prefix *curve* is still a good supporting figure; the endpoint comparison is
  not a claim.
- **The GPU spec in the docs was wrong** — sessions used a 40GB A100, not 80GB.
  For the round trip an 80GB box helps only by avoiding one server restart; the
  pipeline is inherently sequential (AR inputs do not exist until AV outputs do),
  so generation speed matters more than VRAM.

## 8. Next

1. **Run the pilot** (one GPU spin-up, ~1h including setup).
2. **NO-GO** → write it up scoped: "greedy AV descriptions did not let a blinded
   reader distinguish these natural private/public activation pairs under the
   tested checkpoint and decoding configuration." Close the NLA thread.
3. **PROCEED** → blinded 2AFC (~15 min of human time), then the full 233-pair
   run in the same session if it passes 26/40.
4. Then the AR round trip, then write.

Still open: tier-4 fix-or-drop (**L14**); the judge QA pass; the second-judge
robustness item.
