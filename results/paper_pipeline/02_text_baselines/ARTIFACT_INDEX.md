# Artifact index — Step 2: text baselines and eval-awareness control

Every artifact is **provisional**. Later steps (onset dynamics, blind label audit) may change the preferred framing or figure.

**Status legend** — `candidate_main`: proposed for main text · `candidate_appendix`: appendix · `sensitivity`: robustness only · `exploratory`: not claim-bearing · `deprecated`: superseded.

| artifact | kind | status | intended claim | paper placement | replaceable |
|---|---|---|---|---|---|
| `tab_privileged_delta_216` | table | `candidate_main` | Layer-20 activations at the final prompt token predict upcoming disclosure-limiting strategy substantiall… | Main text, primary results table. | yes |
| `tab_dissociation_216` | table | `candidate_main` | The privileged increment is larger for strategy than for disclosure outcome, tested directly rather than… | Main text, alongside the primary table. | no |
| `tab_eval_awareness_control_216` | table | `candidate_main` | The limiting signal is not evaluation-awareness: erasing the eval-awareness direction costs 0.0017 AUC wi… | Main text or appendix, as the control for the headline result. | no |
| `tab_population_sensitivity_delta` | table | `sensitivity` | The privileged increment is stable in direction and ordering when the 42 calibration cases are added. | Appendix (sensitivity). | no |
| `fig_privileged_delta_216` | figure | `candidate_main` | The paper's headline: pre-response activations predict upcoming disclosure strategy beyond what the scena… | Main text, Figure 1. | yes |
| `fig_eval_awareness_control_216` | figure | `candidate_appendix` | The headline result survives erasure of evaluation-awareness, with a manipulation check demonstrating the… | Appendix, or main text beside the headline result. | no |

## Main figure slot

Step 1 reserved the main-figure slot pending this step's Δ. `fig_privileged_delta_216` now claims it: it shows a relationship (two channels and the gap between them) rather than six independent point estimates, and it carries the paper's headline claim.
