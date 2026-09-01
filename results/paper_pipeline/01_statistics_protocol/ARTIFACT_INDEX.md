# Artifact index — Step 1: statistics protocol

Every artifact below is **provisional**. Producing a result here does not commit the paper to using it: later steps (text baselines / privileged Δ, eval-awareness control, blind label audit, layer × onset experiment) may change the preferred metric, figure, or framing.

**Status legend** — `candidate_main`: proposed for main text · `candidate_appendix`: proposed for appendix · `sensitivity`: robustness only, never a primary estimate · `exploratory`: not claim-bearing · `deprecated`: superseded, retained for provenance.

| artifact | kind | status | population | intended claim | paper placement | replaceable by later step | promotion / demotion condition |
|---|---|---|---|---|---|---|---|
| `tab_primary_contrasts_216` | table | `candidate_main` | analysis_216 | Under corrected statistics, four of six pre-specified behavioural contrasts are decodable above chance from la… | Main text, results table. | yes | Remains the main results table unless Step 2's Δ supersedes absolute AUC as the headline quantity. |
| `tab_protocol_correction_v1_v2` | table | `candidate_appendix` | analysis_216 | Pooling predict_proba scores across cross-validation folds that each select their own regularisation strength… | Appendix (methods). | no | Promote to main text only if the paper's framing centres the measurement/protocol-correction contrib… |
| `tab_interval_methods` | table | `candidate_appendix` | analysis_216 | The repeated-cross-validation percentile band understates sampling uncertainty because repeats share the same… | Appendix (methods). | no | Appendix only; no promotion path expected. |
| `tab_population_sensitivity_216_vs_258` | table | `sensitivity` | analysis_216 (primary) and all_258 (superset, sensitivity only) | The direction and ordering of results is stable when the 42 calibration cases are added, with uniformly higher… | Appendix (sensitivity analysis). | yes | Never promoted to a primary result. Would move to main text only as an explicit robustness paragraph… |
| `fig_contrast_effects_dual_metric_216` | figure | `candidate_appendix` | analysis_216 | Four contrasts are decodable above chance on the primary metric; the strategy contrasts and the leak-vs-approp… | Appendix, companion to the primary results table. Not a main-text candidate while Step 2 is pending. | yes | Promote to candidate_main only if Step 2's Δ is null or unusable, leaving absolute decodability as t… |
| `fig_protocol_correction_v1_v2` | figure | `candidate_appendix` | analysis_216 | Two of the four statistical corrections, shown directly: fold-wise metric computation raises every contrast, a… | Appendix (methods). The most durable Step-1 figure. | no | Promote to main text if the paper centres the measurement-validity contribution; otherwise appendix. |

## Reserved

**Main figure slot: reserved.** Step 1 deliberately produces no `candidate_main` figure. Six point estimates from one population are a table, not a figure, and the paper's headline quantity is expected to become the Step-2 privileged Δ (activation − matched text baseline). `fig_contrast_effects_dual_metric_216` is the fallback main candidate if that Δ turns out null or unusable.
