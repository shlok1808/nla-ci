# Step 1 — statistics protocol correction

**Everything here is provisional.** Producing a table or figure in this package does not
commit the paper to using it. Later steps (matched text baselines / privileged Δ,
eval-awareness control, blind label audit, layer × onset experiment) may change the preferred
metric, the preferred figure, or the framing. Every status here is `candidate_*`.

Rebuild and check:

```
python3 scripts/paper_step01_build_f.py
python3 scripts/validate_paper_step01_f.py
```

## What Step 1 tested

Linear probes on layer-20 residual activations at the **final prompt token** — i.e. before the
model has generated a single response token — against six behavioural contrasts derived from
the canonical tier-3 labels. The question is whether the model's internal state at the moment
of acting predicts what it is about to do with private information.

Population: **216 canonical analysis cases** (primary). A 258-case superset exists and is
reported as a sensitivity analysis only.

## The statistical bug that was fixed

The previous script (`scripts/probe_contrasts_canonical_f.py`, "v1") computed one AUC by
pooling raw `predict_proba` scores from all five outer cross-validation folds into a single
ranking. Each fold selects its own regularisation strength `C` through nested CV, and at these
sample sizes that selection is unstable — the chosen `C` ranges from 1e-7 to 1.0 *within a
single repeat*. Probability scales differ by orders of magnitude across `C`, so pooling ranks
timid scores against confident ones and corrupts the ranking at fold boundaries.

The corrected script (`scripts/probe_contrasts_canonical_v2_f.py`, "v2") makes four changes:

1. **Per-fold metric computation.** Every metric is computed inside its own held-out fold and
   averaged. This is the standard fix for the pooled cross-validation AUC pitfall
   (Forman & Scholz 2010, *SIGKDD Explorations* 12(1); Airola et al. 2011, *CSDA* 55(4)).
2. **Permutation null on the identical statistic.** A permutation test is valid only when the
   null distribution is built from the same statistic as the observed value
   (Ojala & Garriga 2010, *JMLR* 11). The null now uses per-fold averaging too. It still runs
   one CV repeat at fixed modal `C` against a 20-repeat observed statistic, so it carries extra
   split variance — conservative, in the safe direction.
3. **Honest intervals.** v1 reported the 2.5/97.5 percentile over 20 repeated splits of the
   *same* cases. That measures split variability, not sampling uncertainty
   (Nadeau & Bengio 2003; Bates, Hastie & Tibshirani 2023, *JASA*), and was 1.5–2.5× too
   narrow. Reported now: Hanley & McNeil analytic 95% and a stratified prediction-resampling
   bootstrap 95%.
4. **Holm adjustment** across the six scored contrasts within each population (Holm 1979).

Data, contrasts, n, positive counts, seeds, fold assignments and the `C` grid are **identical**
between v1 and v2. Only the summary statistic differs.

## What changed numerically

Every contrast rose; the correction was specified before the corrected numbers were seen, and
the direction follows from the defect rather than justifying it. ROC-AUC deltas span
**+0.015 to +0.058** (see `tables/tab_protocol_correction_v1_v2.csv`). The headline
`substantive leak vs appropriate` contrast moved 0.756 → **0.814**.

## Which population is primary

**216 analysis cases.** The 258-case population is a strict *superset* — it adds 42
calibration cases that deliberately oversampled the historical `refused` class, so they are
enriched for limiting (66.7% vs 35.2%) and disclosing (61.9% vs 26.4%) behaviour, and they are
the only cases with human-verified references. Its uniformly higher estimates confound
composition with label quality. It is **not** a replication and is never a primary estimate.

## Verdicts

PR-AUC is the primary metric. It was pre-specified in the analysis script docstring, committed
before the corrected run — it is **not** registered in `docs/PREREGISTRATION.md`, and should
never be described as pre-registered. ROC-AUC is supporting evidence and never rescues a
PR-failing contrast.

| verdict | contrasts |
|---|---|
| **supported** (Holm-adjusted PR p ≤ .05) | substantive leak vs appropriate (ROC .814); limiting vs direct, disclosers only (.728); limiting vs direct, all (.724); broad breach vs none (.698) |
| **suggestive** (fails PR, passes ROC) | substantive leak vs rest (ROC .639, Holm PR p = .132) |
| **no_evidence** | broad-only vs substantive — the disclosure *degree* boundary (ROC .575) |

Full numbers with intervals: `tables/tab_primary_contrasts_216.csv`.

## What remains provisional

- **Labels.** All 216 analysis-population labels are single-judge and
  `provisional_unverified`; the same model family drafted the references and applied the
  rubric, and the judge prompt asserted the references were human-verified when 216/258 were
  not. No interval here covers that uncertainty. The blind stratified audit (pipeline step 11)
  is what would bound it.
- **No text baseline yet.** These are absolute decodability numbers. Whether the activation
  carries information *beyond the visible scenario text* is Step 2 and is not established here.
- **One layer, one position.** Layer 20 at the final prompt token, because that is the only
  publicly released NLA checkpoint layer. No canonical-label evidence exists for other layers;
  the old layer trajectory used superseded labels and is withdrawn.
- **Construct.** The limiting contrast is "discloses and then limits" — 72 of 76 limiting
  cases in this population are `mixed_disclose_then_limit`, with zero explicit refusals. It is
  not the old deflection/refusal construct, which collapsed from 36 cases to 7.

## What later steps could change

- **Step 2 (text baselines / privileged Δ)** is expected to supply the paper's headline
  quantity: activation AUC minus a matched text-baseline AUC. If so,
  `fig_contrast_effects_dual_metric_216` moves permanently to the appendix and the main results
  table gains Δ columns. **The main-figure slot is deliberately reserved for that.**
  *Blocker:* the existing scratch text baseline uses pooled CV scoring — the statistic this
  step abandoned — and writes no result file. It must be recomputed under the v2 per-fold
  protocol, on the same folds and seeds, with a paired scenario bootstrap, before any Δ is
  quoted.
- **Step 3 (eval-awareness control)** could weaken the strategy result if the limiting
  direction turns out to be generic caution rather than privacy-specific.
- **The blind label audit** could move any verdict, most plausibly the two closest to the
  threshold (`broad_breach` at Holm PR .030 and `substantive_leak` at .132).
- **The layer × onset experiment** would add the dynamics axis and could show the pre-response
  signal is anticipatory or merely transcript-borne.

## Contents

| path | what |
|---|---|
| `ARTIFACT_INDEX.md` | one row per artifact: status, claim, placement, promotion conditions |
| `PRESENTATION_NOTES.md` | the statistical traps to avoid when writing this up — read before drafting |
| `run_metadata.json` | provenance: git state, environment, input/output hashes, protocol, verdict rule |
| `SOURCES.sha256` | hashes of every authoritative input |
| `tables/` | four candidate tables (+ `.tex` and sidecar notes) |
| `figures/` | two candidate figures (pdf/svg/png + plot data + sidecar notes) |

Every artifact has a same-name `.md` sidecar stating what it supports, what it must **not** be
claimed to show, its caveats, and what would replace it.

## Related

- Authoritative results: `results/probe_contrasts_canonical_v2_f.{csv,json}`
- Superseded results kept for provenance: `results/probe_contrasts_canonical_f.{csv,json}`
- The audit that motivated this correction: `docs/FULL_AUDIT_AND_NEXT_STEPS_f.md`
- Session-13 figures, renamed rather than deleted because they use superseded judge labels:
  `results/figures_deprecated_oldlabels/`
