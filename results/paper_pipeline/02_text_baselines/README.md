# Step 2 — matched text baselines, privileged Δ, and the eval-awareness control

**Everything here is provisional.** Later steps (onset dynamics, blind label audit) may
change the preferred framing or figure. Every status is `candidate_*`.

Rebuild:

```
python3 scripts/text_baselines_canonical_f.py      # 2A
python3 scripts/eval_awareness_canonical_f.py      # 2B
python3 scripts/paper_step02_build_f.py            # package
```

## The question

Step 1 established that layer-20 activations at the final prompt token decode several
behavioural contrasts above chance. That alone is weak: the activation is a deterministic
function of the prompt, so a reviewer can reasonably ask whether the probe is reading the
model's state or just clues already present in the scenario wording. Step 2 answers that
(2A), and then rules out the most likely alternative explanation for what remains (2B).

## 2A — does the activation beat the text?

Three channels per contrast, on byte-identical cross-validation folds under the corrected
Step-1 protocol:

| channel | what it sees |
|---|---|
| `acts` | layer-20 residual activation at the final prompt token |
| `tfidf` | TF-IDF 1–2-grams of the scenario text |
| `embed` | frozen `all-MiniLM-L6-v2` sentence embedding of the scenario text |

The generated response is **never** used — it post-dates the probed state and would
trivially reveal the label. Δ is reported against the **stronger** of the two text
baselines per contrast (the encoder won on all six), because using the weaker one would
inflate the result.

**Result (216 analysis population):**

| contrast | activation | text (encoder) | Δ | 95% CI | Holm p |
|---|---|---|---|---|---|
| limiting vs direct (disclosers only) | 0.728 | 0.567 | **+0.162** | [+0.083, +0.234] | **.006** |
| limiting vs direct (all) | 0.724 | 0.600 | **+0.125** | [+0.049, +0.196] | **.010** |
| substantive leak vs appropriate | 0.814 | 0.757 | +0.057 | [−0.035, +0.147] | .53 |
| broad breach vs none | 0.698 | 0.665 | +0.033 | [−0.042, +0.112] | .53 |
| substantive leak vs rest | 0.639 | 0.611 | +0.028 | [−0.048, +0.113] | .53 |
| broad-only vs substantive | 0.575 | 0.582 | −0.007 | [−0.092, +0.072] | .57 |

The strategy contrasts carry a privileged increment; the disclosure-outcome contrasts do
not. Note what this does to the Step-1 picture: `leak_vs_appropriate` looked like the
strongest result at 0.814, but a frozen sentence encoder recovers 0.757 of it from the
prompt alone. Most of that number is scenario-predictable.

**The dissociation is tested, not eyeballed.** A difference between a significant and a
non-significant result is not itself significant (Gelman & Stern 2006), so the difference
of the two Δs is tested directly: limiting-among-disclosers versus substantive-leak gives
**+0.130, CI [+0.023, +0.242], p = .010**. The comparison against `leak_vs_appropriate`
(+0.104, CI [−0.017, +0.220]) grazes zero and is reported as suggestive only.

## 2B — is it just evaluation-awareness?

The obvious alternative: the probe reads generic caution / "this looks like a test",
not a privacy-handling strategy. `v_eval` is derived from 270 tier-3 scenarios each framed
3× test-like and 3× deploy-like with byte-identical user content, so scenario content
cancels in the paired difference. It never sees a limiting label.

**Result:** erasing it costs the limiting probe **0.0017 AUC** (0.737 → 0.736), CI spanning
zero, p vs null = .18. `cos(v_eval, v_limiting) = −0.093`, inside the permutation-null band.

**The manipulation check is what makes this a real null:** the same cross-fitted erasure
drives test-vs-deploy decodability from **1.000 to 0.500** — perfectly decodable before,
exactly chance after. The erasure demonstrably removed the concept and the limiting probe
did not move.

Three design choices carry this result:

- **Cross-fitting.** `v_eval` and every null direction are re-derived inside each training
  fold from framed activations of *training* scenarios only. A direction estimated on all
  data would over-remove on held-out data and bias toward a false negative.
- **The null is permuted-frame directions, not random vectors.** Within each scenario,
  which 3 of its 6 frames count as "test" is reassigned at random and the direction
  re-derived through the same pipeline — matched norm, matched manifold alignment, no
  content. Isotropic random directions cost ~0.0000 AUC when erased (a random vector in
  3584 dimensions is nearly orthogonal to the data manifold), so `v_eval` would beat that
  null trivially and meaninglessly.
- **A tuned decoder for the manipulation check**, so "1.000 before" is not an artifact of
  arbitrary regularisation.

## What the paper can now claim

1. Pre-response activations predict upcoming disclosure-limiting strategy substantially
   better than the scenario wording does (Δ +0.162, Holm p = .006).
2. This is specific to strategy rather than disclosure outcome (tested, p = .010 against
   substantive leak).
3. It is not evaluation-awareness (erasure control with a passing manipulation check).

## What it must not claim

- **Not "information absent from the prompt."** The activation is a deterministic function
  of the prompt. Δ > 0 means the model's processing makes behaviour-relevant information
  linearly *extractable* where text classifiers cannot extract it. That is the claim.
- **Δ is relative to a baseline family.** TF-IDF and one frozen encoder. A stronger encoder
  can only shrink Δ, never grow it.
- **No causal claim.** These are correlational decoding results at a single extraction point.
- **Rank-1 erasure removes a direction**, not necessarily a concept in full. Eval-awareness
  collapsing to exactly chance suggests that direction carried essentially all of it here,
  but a distributed residual cannot be excluded.
- **Labels remain provisional** — single-judge, `provisional_unverified`, with a judge
  prompt that asserted human verification when 216/258 references were not verified. No
  interval here covers that.

## A statistical correction made during this step

The first 2A run bootstrapped by resampling within a single CV repeat's folds, while the
reported statistic averages 20 repeats. Each draw therefore carried one repeat's full split
noise against a 20-repeat estimator, making intervals too wide. The corrected bootstrap
resamples scenarios once per draw and recomputes the full repeat-averaged per-fold
statistic for both channels — the estimator actually reported.

This was identified **after** seeing the first run, in which the limiting Holm p was .054.
The correction's direction (narrower intervals) follows a priori from averaging reducing
split variance, and point estimates are unchanged. Both inferences are retained in the
results file (`legacy_*` columns) so the change is auditable rather than silent.

A `fast_auc` (Mann-Whitney U with average ranks) replaces `roc_auc_score` in the bootstrap
hot path — 28× faster, verified identical to 1e-12 on 200 cases including heavy ties, with
that assertion running at the start of every execution.

## Step 3 pre-flight (included here)

`results/onset_alignment_f.{csv,json}` — verdict **GO**:

- char→token round-trip verified on all 224 disclosers, zero failures
- primary pre-onset window set empirically to **[−8, −1]**: 97.2% / 100% coverage by class,
  versus 79% / 93% at 16 tokens, which would have introduced class-differential attrition
  inside the primary cell
- positional confound quantified: the min(disclosure, limiting) cutoff position alone
  predicts the class at **AUC 0.609** — the floor the activation probe must clear, and the
  reason the Step-3 text baseline must include prefix position and length

## Contents

| path | what |
|---|---|
| `ARTIFACT_INDEX.md` | one row per artifact with status and placement |
| `PRESENTATION_NOTES.md` | traps to avoid when writing this up |
| `run_metadata.json` | provenance, hashes, both protocols, headline numbers |
| `tables/` | four candidate tables (+ `.tex` + sidecars) |
| `figures/` | the candidate main figure and the control figure |
