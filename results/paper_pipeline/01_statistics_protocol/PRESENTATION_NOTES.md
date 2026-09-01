# Presentation notes — read before writing up Step 1

Traps that are easy to fall into with these specific numbers, and the figure decisions taken
in this package. Written for whoever drafts the paper.

## Statistical traps

1. **PR chance is not prevalence.** The cross-validated permutation null exceeds class
   prevalence by ~0.05–0.06 on every contrast. Always report excess over the permutation null.
   The sharpest illustration: `degree_boundary`'s PR bootstrap CI is [.345, .518], entirely
   above its prevalence of .303 — while its permutation p is .31. A prevalence-baseline figure
   would draw the paper's null result as significant.
2. **A bootstrap CI is not a test against the null.** The prediction-resampling bootstrap is
   conditional on the fitted models and centred on the observed estimate. Every "above chance"
   claim cites the permutation p, never CI exclusion.
3. **ROC and PR disagree on `substantive_leak`, and PR is primary.** Both its ROC intervals
   exclude 0.5 and its Holm-adjusted ROC p is at the floor, yet its Holm-adjusted PR p is .132.
   Leading with ROC would report five supported contrasts where the rule says four.
4. **`p = .012` is the resolution floor, not a measurement.** n_perm = 500 → smallest raw p is
   1/501; ×6 for Holm = .01198. Render `≤.012` with an n_perm footnote. Never write "p = .012
   across four contrasts" as if the agreement were informative.
5. **The permutation null is deliberately conservative** — one CV repeat at fixed modal C
   against a 20-repeat observed statistic, so it carries extra split variance. Worth stating:
   the p-values err large, which is the safe direction.
6. **Overlapping CIs ≠ no difference; non-overlapping ≠ difference.** Exactly one of fifteen
   pairs is non-overlapping (`leak_vs_appropriate` vs `degree_boundary`). No between-contrast
   test was run and none is powered. Say "highest point estimate among the contrasts;
   differences between contrasts are not statistically resolved."
7. **The contrasts are dependent samples.** `leak_vs_appropriate` (n=81) and `degree_boundary`
   (n=188) share all 57 substantive-leak cases; `limiting_among_disclosers` (188) is nested in
   `limiting_vs_direct` (216). Holm across them is conservative-to-unclear under dependence —
   state that it was applied within population across the six scored contrasts.
8. **Raw PR values are not comparable across rows** because prevalence differs. `broad_breach`
   PR .936 at prevalence .870 is not the same achievement as `leak_vs_appropriate` PR .911 at
   prevalence .704. Never rank by raw PR; the per-row null is the fix.
9. **`leak_vs_appropriate` is a subsetted population** (n=81): it drops 131 `broad_only` and 4
   `refused` cases, so its .704 prevalence is an artifact of subsetting, and its 24 negatives
   are all provisional-labelled. Always print n alongside it.
10. **No interval covers label noise.** Single-judge, provisional, same model family drafting
    and judging, and the judge prompt falsely asserted human verification. Attenuation is the
    likely direction, but *systematic* reference error could manufacture structure and is
    unquantified.
11. **Four of six contrasts sit at the C grid floor** (1e-7). Below the floor the AUC is flat
    and equals a standardised difference-of-means probe, so being at the edge is benign — but
    it means the decodable signal is a class-mean direction, not a high-capacity boundary.
    Worth one honest sentence rather than a footnote.
12. **The limiting construct is "disclose then limit."** 72 of 76 limiting cases are
    `mixed_disclose_then_limit`; 4 are `soft_deflection`; zero are `explicit_refusal`. Do not
    call it deflection or refusal — that construct collapsed 36 → 7 and is unmeasurable here.
13. **258 is a superset, never a replication.** Enriched composition *and* different label
    verification, both confounded with population. Sensitivity only.
14. **v1's band was never a CI.** If any v1 number appears, label it "split variability."
15. **Hanley's assumptions are violated here.** It assumes one fixed scoring rule on
    independent cases; ours averages per-fold AUCs from models refit per split. Report it as an
    approximation next to the bootstrap, and note they disagree by up to ~0.05.
16. **Step-2 blocker.** The scratch text baseline is pooled-CV and unfiled. Recompute under the
    v2 per-fold protocol with the same folds and seeds, and use a **paired** scenario bootstrap
    for the Δ CI (the channels correlate at r ≈ 0.48, so pairing buys real power). The Δ
    figures quoted in the audit doc are protocol-mismatched in the activation's favour.

## Figure decisions, and why they differ from the initial proposal

**No `candidate_main` figure was produced, by design.** Six point estimates from one
population is a table, not a figure: a figure earns its slot by showing a relationship, and the
table carries columns (n, prevalence, both p-values, grid-edge flag) a reviewer will want. The
main-figure slot is reserved for Step 2's privileged Δ, which is the actual claim.

**The proposed ROC forest plot was merged into a dual-metric figure rather than built alone.**
A ROC-only forest with a chance line shows five of six contrasts clear of chance — visually
contradicting the four-supported verdict, because `substantive_leak` passes on ROC and fails on
the primary PR metric. Panel (b) puts the primary metric in the figure so the geometry and the
verdict agree.

**The standalone PR-vs-prevalence figure was rejected and folded in.** Its baseline was wrong
(trap 1), and two figures showing the same six contrasts in different geometries makes the
reader re-establish identity twice. It is now the right panel, plotted against the permutation
null with prevalence shown as a faint tick so the CV bias is visible rather than hidden.

**The 216-vs-258 figure was rejected outright**, not deprioritised. A figure is an invitation
to compare, and the correct message is "these are not comparable." It ships as a table with the
composition block (limiting %, leak %, verification status) in the same glance as the AUC gap —
a table header can carry a supersetting caveat; a scatter cannot.

**Colour encodes construct family, never significance or verdict**, reusing the repo's
validated colourblind-safe palette. Verdict is encoded by marker fill only, so the figure
survives greyscale printing. Row order is fixed a priori by family and never sorted by effect
size, since sorting manufactures a rank the data does not support.

**Anti-fallacy devices in the dual-metric figure:** no connecting lines between contrasts,
family grouping via whitespace so the eye compares within-family, n in every row label, and an
explicit in-figure note that these are separate tests on overlapping case sets with marginal
intervals and no between-contrast comparison.

**`fig_protocol_correction_v1_v2` is the most durable Step-1 artifact.** It shows a paired
change rather than six independent estimates, later steps adopt the corrected protocol rather
than superseding it, and if Step 1 gets exactly one figure into the paper it should be this
one. Note in the caption that it carries only two of the four corrections — the matched
permutation null and the Holm adjustment are not visualisable and live in the correction table.
