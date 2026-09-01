# Presentation notes — Step 2

Traps specific to these numbers, and the design decisions behind the artifacts.

## Claim wording

1. **Never write "the activation contains information the text doesn't."** The
   final-prompt-token activation is a deterministic function of the prompt; no new
   information can exist in it. The correct phrasing is that the model's processing makes
   behaviour-relevant information *linearly extractable* where text classifiers cannot
   extract it. This is the single easiest way to get the paper's headline dismissed, and
   the literature the repo already cites ("linear probes rely on textual evidence",
   arXiv:2509.21344) is exactly about this failure mode.
2. **Δ is baseline-relative.** Reported against TF-IDF and one frozen MiniLM encoder. Any
   stronger text model can only shrink Δ. State the baseline family explicitly; do not
   write "beyond what text can predict" unqualified.
3. **Report Δ against the stronger baseline.** The encoder beat TF-IDF on all six
   contrasts. Quoting the TF-IDF Δ would inflate every number and is indefensible once a
   reviewer runs a better encoder.
4. **The leak null is a finding, not an omission.** `leak_vs_appropriate` decodes at 0.814
   but the encoder recovers 0.757 from the prompt. Report this yourself. If the paper
   claims 0.814 as a privileged result, a reviewer with a sentence encoder will end it.
5. **No causal language.** Correlational decoding at one extraction point.

## Statistics

6. **The dissociation needs its own test.** "Limiting is significant, leak isn't" is not a
   comparison (Gelman & Stern 2006, *Am. Stat.* 60(4)). The tested difference is in
   `tab_dissociation_216`: +0.130 [+0.023, +0.242] p = .010 against substantive leak. The
   comparison against `leak_vs_appropriate` grazes zero — call it suggestive, not
   established.
7. **The four dissociation comparisons are a coherent set on overlapping case sets**, not
   independent tests. No multiplicity adjustment is applied across them; say so.
8. **The bootstrap correction must be disclosed.** It was made after seeing a marginal
   result (Holm p .054 → .006). The fix is correct on its own terms — you bootstrap the
   estimator you report — and its direction was predictable a priori, and point estimates
   did not move. But the sequence is a forking path, so both inferences are in the results
   file and the change belongs in the methods section, not buried.
9. **Bootstrap CIs are not tests.** The Δ p-value is the one-sided bootstrap
   P(Δ* ≤ 0); do not argue significance from interval exclusion elsewhere.
10. **A label-permutation null is wrong for Δ.** Permuting labels nulls both channels,
    testing "no signal anywhere" rather than "no difference between channels".
11. **Labels are provisional throughout** — single-judge, unverified references, and the
    judge prompt falsely asserted verification. Attenuation is the likely direction, but
    systematic reference error is unquantified until the blind audit runs.

## The eval-awareness control

12. **The manipulation check is the load-bearing part.** Without 1.000 → 0.500, a null
    drop would mean "we failed to erase anything," not "the signal isn't caution." Lead
    with the check, then the null.
13. **Isotropic random directions are an inadequate null and are reported only to show
    that.** They cost ~0.0000 AUC when erased because a random vector in 3584 dimensions
    is nearly orthogonal to the activation manifold. The real null is 100 permuted-frame
    directions from the same estimation pipeline.
14. **Cross-fitting matters for the direction of the bias.** A direction fit on all data
    over-removes on held-out data, which would push toward a false "not caution" result —
    i.e. toward the answer we wanted. Say that the control was cross-fitted.
15. **Rank-1 erasure removes a direction.** Eval-awareness collapsing to exactly chance
    indicates that direction carried essentially all of it *in this framing*, but a
    distributed residual cannot be fully excluded, and the framing manipulation is our own
    construct rather than a natural distribution shift.
16. **The control rules out one alternative explanation.** It does not, by itself,
    establish that the signal is privacy-specific — only that it is not the specific
    confound tested.

## Figure decisions

**`fig_privileged_delta_216` claims the main-figure slot** that Step 1 deliberately
reserved. It qualifies where a Step-1 forest plot did not: it shows a *relationship* (two
channels and the gap between them) rather than six independent point estimates, and the
gap is the paper's claim. Panel (a) shows both text channels so the reader can see the
encoder is the stronger baseline; panel (b) shows Δ against that stronger baseline with
its paired CI.

**Colour follows construct family, never significance** (blue = disclosure presence,
orange = strategy, violet = degree), reusing the repo's validated colourblind-safe
palette. Verdict is marker fill only, so the figure survives greyscale. Row order is fixed
a priori by family and never sorted by effect size.

**The dissociation is a table, not a figure.** It is four numbers with intervals, and the
comparison is between quantities the main figure already shows; a second figure would
duplicate identity work for no gain.

**The eval-awareness control keeps its manipulation check in the same figure** (panel b).
Separating them would let a reader take the null drop without seeing the evidence that the
erasure worked — which is the only reason the null is interpretable.

## What Step 3 could change

If the onset experiment shows the strategy signal exists before any hedging or disclosing
token, the onset figure likely becomes Figure 1 and this figure moves to Figure 2 as the
static result. If it shows the signal is transcript-borne, this figure stays as the
headline and Step 3 becomes a scope limitation. Either way `tab_privileged_delta_216`
survives — Step 3 adds a position axis, it does not invalidate the static comparison.
