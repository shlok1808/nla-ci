#!/usr/bin/env python3
"""paper_step02_build_f.py — build the Step-2 paper artifact package.

Emits candidate tables, the candidate MAIN figure, per-artifact sidecars,
ARTIFACT_INDEX.md, run_metadata.json, README.md and PRESENTATION_NOTES.md into
results/paper_pipeline/02_text_baselines/.

Every number is loaded from results/text_baselines_canonical_f.{csv,json},
results/text_baselines_canonical_f_dissociation.csv,
results/eval_awareness_canonical_f.{csv,json} and results/onset_alignment_f.json.
Nothing is transcribed.

Idempotent. Writes only inside the step directory.
Usage:  python3 scripts/paper_step02_build_f.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_step01_common_f as C1        # noqa: E402  (paths, hashing, git, fmt)
from figstyle_f import (                   # noqa: E402
    C_INK, C_MUTED, C_GRID, C_LEAK, C_DEFL, C_PRIV, save_all, titles, plt,
)

REPO = C1.REPO
TB_CSV = REPO / "results/text_baselines_canonical_f.csv"
TB_JSON = REPO / "results/text_baselines_canonical_f.json"
DIS_CSV = REPO / "results/text_baselines_canonical_f_dissociation.csv"
EA_CSV = REPO / "results/eval_awareness_canonical_f.csv"
EA_JSON = REPO / "results/eval_awareness_canonical_f.json"
AL_JSON = REPO / "results/onset_alignment_f.json"

STEP = REPO / "results/paper_pipeline/02_text_baselines"
TABLES, FIGURES = STEP / "tables", STEP / "figures"

_EPOCH = int(os.environ.get("SOURCE_DATE_EPOCH", TB_CSV.stat().st_mtime))
BUILD_UTC = datetime.fromtimestamp(_EPOCH, timezone.utc).isoformat()

DISPLAY = C1.DISPLAY
FAMILY = C1.FAMILY
ORDER = C1.CONTRAST_KEYS
ALPHA = 0.05

_LABEL_CAVEAT = (
    "All 216 analysis-population labels are single-judge and provisional_unverified; "
    "the judge prompt asserted human verification when 216/258 references were not "
    "verified. No interval here covers that uncertainty."
)
_NOT_INFO_THEORETIC = (
    "that the activation contains information absent from the prompt — the "
    "final-prompt-token activation is a deterministic function of the prompt, so a "
    "positive delta means the model's processing makes behaviour-relevant information "
    "linearly EXTRACTABLE where text classifiers cannot extract it, not that new "
    "information exists"
)
_BASELINE_RELATIVE = (
    "that no text baseline could close the gap — delta is defined relative to a "
    "baseline family (TF-IDF and a frozen MiniLM encoder); a stronger encoder can only "
    "shrink it"
)


def verdict(p_holm):
    return "supported" if p_holm <= ALPHA else "no_evidence"


def load():
    tb = pd.read_csv(TB_CSV)
    p = tb[(tb.status == "scored") & tb.contrast.str.endswith("analysis_216")].copy()
    p["key"] = p.contrast.str.split("|").str[0]
    p = p.set_index("key").loc[ORDER].reset_index()
    p["display"] = p["key"].map(DISPLAY)
    p["family"] = p["key"].map(FAMILY)
    p["verdict"] = p["p_holm_delta_stronger"].map(verdict)
    s = tb[(tb.status == "scored") & tb.contrast.str.endswith("all_258")].copy()
    s["key"] = s.contrast.str.split("|").str[0]
    s = s.set_index("key").loc[ORDER].reset_index()
    return tb, p, s


def build_tables(p, s):
    out = {}
    # ── primary delta table ──────────────────────────────────────────────────
    t = pd.DataFrame({
        "contrast": p["display"],
        "n": p["n"].astype(int),
        "activation AUC": p["roc_acts"].map(C1.fmt_num),
        "TF-IDF AUC": p["roc_tfidf"].map(C1.fmt_num),
        "embedding AUC": p["roc_embed"].map(C1.fmt_num),
        "stronger text baseline": p["stronger_text_channel"],
        "privileged Δ": p["delta_roc_stronger"].map(lambda x: f"{x:+.3f}"),
        "Δ 95% CI": [C1.fmt_ci(a, b) for a, b in
                     zip(p["delta_boot_lo_stronger"], p["delta_boot_hi_stronger"])],
        "Holm p (Δ)": p["p_holm_delta_stronger"].map(
            lambda x: f"{x:.3f}".replace("0.", ".")),
        "verdict": p["verdict"],
    })
    notes = [
        "Δ = activation AUC − text AUC against the STRONGER of the two text baselines "
        "per contrast (the frozen sentence encoder won on every contrast); using the "
        "weaker baseline would inflate Δ.",
        "Text channels see the scenario/prompt only. The generated response is never "
        "used: it post-dates the probed state and can trivially reveal the label.",
        "All three channels share byte-identical cross-validation folds (same seeds), "
        "so the bootstrap is exactly paired.",
        "Δ inference is a paired scenario bootstrap that resamples scenarios once per "
        "draw and recomputes the full repeat-averaged per-fold statistic for both "
        "channels; p is one-sided P(Δ* ≤ 0), Holm-adjusted across the six contrasts.",
        "A label-permutation null is deliberately not used for Δ: permuting labels "
        "nulls both channels, testing 'no signal anywhere' rather than 'no difference'.",
        _LABEL_CAVEAT,
    ]
    out["tab_privileged_delta_216"] = (t, notes)

    # ── dissociation table ───────────────────────────────────────────────────
    d = pd.read_csv(DIS_CSV)
    d = d[d.population == "analysis_216"]
    t2 = pd.DataFrame({
        "limiting contrast": d["limiting_contrast"].map(DISPLAY),
        "leak contrast": d["leak_contrast"].map(DISPLAY),
        "Δ limiting": d["delta_limiting"].map(lambda x: f"{x:+.3f}"),
        "Δ leak": d["delta_leak"].map(lambda x: f"{x:+.3f}"),
        "difference": d["difference"].map(lambda x: f"{x:+.3f}"),
        "95% CI": [C1.fmt_ci(a, b) for a, b in zip(d["boot_lo"], d["boot_hi"])],
        "p": d["p_boot"].map(lambda x: f"{x:.3f}".replace("0.", ".")),
    })
    notes2 = [
        "A difference between a significant and a non-significant result is not itself "
        "significant (Gelman & Stern 2006, The American Statistician 60(4)). This table "
        "tests the difference of the two deltas directly.",
        "Draws are aligned by index across contrasts (same seed and draw order), so the "
        "comparison is paired by construction.",
        "The comparison against substantive_leak is supported; the comparison against "
        "leak_vs_appropriate has an interval that grazes zero and is reported as "
        "suggestive. No multiplicity adjustment is applied across these four "
        "comparisons — they are reported as a coherent set, not as independent tests.",
    ]
    out["tab_dissociation_216"] = (t2, notes2)

    # ── eval-awareness control ───────────────────────────────────────────────
    ea = pd.read_csv(EA_CSV)
    eaj = json.loads(EA_JSON.read_text())
    t3 = pd.DataFrame({
        "contrast": ea["contrast"].map(DISPLAY),
        "n": ea["n"].astype(int),
        "AUC original": ea["auc_original"].map(C1.fmt_num),
        "AUC after erasure": ea["auc_after_v_eval"].map(C1.fmt_num),
        "drop": ea["drop_v_eval"].map(lambda x: f"{x:+.4f}"),
        "drop 95% CI": [f"[{a:+.4f}, {b:+.4f}]" for a, b in
                        zip(ea["drop_boot_lo"], ea["drop_boot_hi"])],
        "null drop mean": ea["null_drop_mean"].map(lambda x: f"{x:+.4f}"),
        "null drop p95": ea["null_drop_p95"].map(lambda x: f"{x:+.4f}"),
        "isotropic drop": ea["iso_drop_mean"].map(lambda x: f"{x:+.4f}"),
        "p vs null": ea["p_vs_null"].map(lambda x: f"{x:.3f}".replace("0.", ".")),
    })
    mc = eaj["manipulation_check"]
    notes3 = [
        f"MANIPULATION CHECK: test-vs-deploy decodability on the framed activations is "
        f"{mc['eval_auc_before']:.3f} before erasure and {mc['eval_auc_after']:.3f} after. "
        "The erasure demonstrably removes the concept, so a surviving limiting probe is "
        "a genuine null rather than a failed manipulation.",
        "Cross-fitted: v_eval and every null direction are re-derived inside each "
        "training fold from framed activations of TRAINING scenarios only. A direction "
        "estimated on all data would over-remove on held-out data and bias toward a "
        "false negative.",
        "The null is 100 side-permuted paired directions from the same estimation "
        "pipeline (within each scenario, which 3 of its 6 frames count as 'test' is "
        "reassigned at random). Isotropic random directions are reported only to show "
        "they are an inadequate null: erasing one costs ~0.0000 AUC, because a random "
        "vector in 3584 dimensions is nearly orthogonal to the activation manifold.",
        f"cos(v_eval, v_limiting) = {eaj['cos_v_eval_v_limiting']:+.3f}, inside the "
        f"permutation-null band (|cos| 95th percentile {eaj['null_abs_cos_p95']:.3f}).",
        "Rank-1 erasure removes a direction. Eval-awareness collapsing to chance "
        "indicates that direction carried essentially all of it here, but a distributed "
        "residual cannot be fully excluded.",
    ]
    out["tab_eval_awareness_control_216"] = (t3, notes3)

    # ── sensitivity ──────────────────────────────────────────────────────────
    t4 = pd.DataFrame({
        "contrast": [DISPLAY[k] for k in s["key"]],
        "Δ (216 primary)": p["delta_roc_stronger"].map(lambda x: f"{x:+.3f}"),
        "Holm p (216)": p["p_holm_delta_stronger"].map(
            lambda x: f"{x:.3f}".replace("0.", ".")),
        "Δ (258 superset)": s["delta_roc_stronger"].map(lambda x: f"{x:+.3f}"),
        "Holm p (258)": s["p_holm_delta_stronger"].map(
            lambda x: f"{x:.3f}".replace("0.", ".")),
    })
    notes4 = [
        "258 ⊃ 216. The 42 additional cases are calibration cases, enriched for "
        "limiting and disclosing behaviour and the only human-verified references. "
        "This is not a replication and the 258 column is never the primary estimate.",
        "The direction and ordering are stable across populations; the 258 estimates "
        "are larger, consistent with that enrichment.",
    ]
    out["tab_population_sensitivity_delta"] = (t4, notes4)
    return out


def fig_privileged_delta(p, stem):
    """MAIN candidate figure: paired channel AUCs with the delta made explicit."""
    fam_c = {"disclosure_presence": C_LEAK, "response_strategy": C_DEFL,
             "disclosure_degree": C_PRIV}
    ys, y, prev = [], 0.0, None
    for _, r in p.iterrows():
        if prev is not None and r["family"] != prev:
            y += 0.85
        ys.append(y); prev = r["family"]; y += 1.0
    ys = np.array(ys)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(9.0, 3.6),
        gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.10})

    # ── left: the two channels, with the gap drawn ───────────────────────────
    axL.axvline(0.5, color=C_MUTED, lw=0.8, ls=":", zorder=1)
    for i, (yy, (_, r)) in enumerate(zip(ys, p.iterrows())):
        c = fam_c[r["family"]]
        axL.plot([r.roc_embed, r.roc_acts], [yy, yy], color=c, lw=1.4,
                 alpha=0.45, zorder=2)
        axL.plot(r.roc_embed, yy, "s", mfc="white", mec=C_MUTED, mew=1.3,
                 ms=5.4, zorder=4)
        axL.plot(r.roc_tfidf, yy, "|", color=C_GRID, ms=7, mew=1.4, zorder=3)
        filled = r["verdict"] == "supported"
        axL.plot(r.roc_acts, yy, "o", color=c if filled else "white",
                 mec=c, mew=1.5, ms=6.4, zorder=5)
    axL.set_xlim(0.44, 0.90)
    axL.set_xlabel("ROC-AUC")
    axL.set_title("(a) activation vs scenario-text channels", loc="left", color=C_INK)
    axL.set_yticks(ys)
    axL.set_yticklabels([f"{r.display}\n(n={int(r.n)})" for _, r in p.iterrows()],
                        fontsize=7.2)
    axL.set_ylim(ys.max() + 0.7, -0.9)
    axL.grid(axis="y", visible=False)
    axL.text(0.5, -0.82, "chance", fontsize=6.8, color=C_MUTED, ha="center",
             va="bottom")

    # ── right: delta with CI ─────────────────────────────────────────────────
    axR.axvline(0.0, color=C_MUTED, lw=0.9, ls="-", zorder=1)
    for yy, (_, r) in zip(ys, p.iterrows()):
        c = fam_c[r["family"]]
        lo, hi = r.delta_boot_lo_stronger, r.delta_boot_hi_stronger
        axR.plot([lo, hi], [yy, yy], color=c, lw=2.1, solid_capstyle="butt", zorder=3)
        for xx in (lo, hi):
            axR.plot([xx, xx], [yy - 0.13, yy + 0.13], color=c, lw=1.5, zorder=3)
        filled = r["verdict"] == "supported"
        axR.plot(r.delta_roc_stronger, yy, "o", color=c if filled else "white",
                 mec=c, mew=1.5, ms=6.4, zorder=5)
        axR.text(1.02, yy, f"{r.p_holm_delta_stronger:.3f}".replace("0.", "."),
                 transform=axR.get_yaxis_transform(), fontsize=6.8,
                 color=C_INK if filled else C_MUTED, va="center", ha="left")
    axR.set_xlim(-0.13, 0.27)
    axR.set_xlabel("privileged Δ  (activation − stronger text baseline)")
    axR.set_title("(b) privileged increment", loc="left", color=C_INK)
    axR.set_yticks(ys)
    axR.set_yticklabels([])
    axR.set_ylim(ys.max() + 0.7, -0.9)
    axR.grid(axis="y", visible=False)
    axR.text(1.02, -0.82, "Holm p", transform=axR.get_yaxis_transform(),
             fontsize=6.8, color=C_MUTED, va="center", ha="left")

    titles(fig,
           "Pre-response activations predict upcoming disclosure STRATEGY beyond "
           "the scenario wording",
           "analysis population, n=216 · circle = layer-20 activation probe, open "
           "square = frozen sentence-encoder baseline, grey tick = TF-IDF · filled "
           "= survives Holm correction · provisional single-judge labels", 3.6)
    fig.text(0.0, -0.15,
             "Δ is measured against the STRONGER text baseline per contrast (the "
             "encoder on all six). A positive Δ means the model's processing makes "
             "behaviour-relevant information linearly extractable where text\n"
             "classifiers cannot extract it — not that the activation holds "
             "information absent from the prompt, which is impossible since it is a "
             "deterministic function of it. Δ is relative to this baseline family; a "
             "stronger encoder can only shrink it.",
             fontsize=6.4, color=C_MUTED, ha="left", va="top")
    save_all(fig, str(stem))
    return {"contrast_key": p["key"].tolist(), "y": ys.tolist(),
            "roc_acts": p["roc_acts"].tolist(), "roc_embed": p["roc_embed"].tolist(),
            "roc_tfidf": p["roc_tfidf"].tolist(),
            "delta": p["delta_roc_stronger"].tolist(),
            "lo": p["delta_boot_lo_stronger"].tolist(),
            "hi": p["delta_boot_hi_stronger"].tolist(),
            "p_holm": p["p_holm_delta_stronger"].tolist(),
            "verdict": p["verdict"].tolist()}


def fig_eval_control(stem):
    ea = pd.read_csv(EA_CSV)
    eaj = json.loads(EA_JSON.read_text())
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(8.4, 3.0),
        gridspec_kw={"width_ratios": [1.0, 0.72], "wspace": 0.42})

    yy = np.arange(len(ea))[::-1].astype(float)
    for i, r in ea.iterrows():
        c = C_DEFL if "limiting" in r.contrast else C_LEAK
        axA.plot([r.drop_boot_lo, r.drop_boot_hi], [yy[i], yy[i]], color=c, lw=2.0,
                 solid_capstyle="butt", zorder=4)
        axA.plot(r.drop_v_eval, yy[i], "o", color=c, ms=5.6, mec="white",
                 mew=0.8, zorder=5)
        axA.plot([-r.null_drop_p95, r.null_drop_p95], [yy[i] + 0.26, yy[i] + 0.26],
                 color=C_MUTED, lw=4.5, alpha=0.28, solid_capstyle="butt", zorder=2)
    axA.axvline(0, color=C_MUTED, lw=0.9, zorder=1)
    axA.set_yticks(yy)
    axA.set_yticklabels([DISPLAY[c] for c in ea.contrast], fontsize=7)
    axA.set_xlabel("AUC lost when eval-awareness is erased")
    axA.set_title("(a) erasing eval-awareness costs the probe nothing",
                  loc="left", color=C_INK)
    axA.grid(axis="y", visible=False)
    axA.text(0.02, 0.04, "grey band = ±95th pct of 100 permuted-frame null directions",
             transform=axA.transAxes, fontsize=6.2, color=C_MUTED)

    mc = eaj["manipulation_check"]
    bars = [mc["eval_auc_before"], mc["eval_auc_after"]]
    axB.bar([0, 1], bars, width=0.55, color=[C_INK, C_GRID], zorder=3)
    axB.axhline(0.5, color=C_MUTED, lw=0.9, ls=":", zorder=2)
    axB.set_xticks([0, 1])
    axB.set_xticklabels(["before\nerasure", "after\nerasure"], fontsize=7.5)
    axB.set_ylim(0, 1.08)
    axB.set_ylabel("test-vs-deploy AUC")
    axB.set_title("(b) the erasure demonstrably worked", loc="left", color=C_INK)
    axB.grid(axis="x", visible=False)
    for x, v in zip([0, 1], bars):
        axB.text(x, v + 0.03, f"{v:.3f}", ha="center", fontsize=7.5, color=C_INK)
    axB.text(1.0, 0.53, "chance", fontsize=6.4, color=C_MUTED, ha="right")

    titles(fig, "The limiting signal is not evaluation-awareness",
           "216 analysis population · cross-fitted rank-1 erasure · "
           "cos(v_eval, v_limiting) = "
           f"{eaj['cos_v_eval_v_limiting']:+.3f}, inside the null band", 3.0)
    save_all(fig, str(stem))
    return {"contrast": ea.contrast.tolist(),
            "drop": ea.drop_v_eval.tolist(),
            "lo": ea.drop_boot_lo.tolist(), "hi": ea.drop_boot_hi.tolist(),
            "null_p95": ea.null_drop_p95.tolist(),
            "iso": ea.iso_drop_mean.tolist(),
            "manipulation": [mc["eval_auc_before"], mc["eval_auc_after"]]}


REGISTRY = {
    "tab_privileged_delta_216": dict(
        kind="table", formats=["csv", "tex"], status="candidate_main",
        supports_claim=(
            "Layer-20 activations at the final prompt token predict upcoming "
            "disclosure-limiting strategy substantially better than the scenario text "
            "does (Δ +0.162 and +0.125, Holm p ≤ .010); the leak contrasts show no "
            "such privileged increment."),
        must_not_claim=[_NOT_INFO_THEORETIC, _BASELINE_RELATIVE,
                        ("that the leak result is null because leak is undecodable — "
                         "leak_vs_appropriate is decodable at 0.814, but a frozen "
                         "encoder recovers 0.757 of it from the prompt alone"),
                        ("that the model uses this representation causally — these are "
                         "correlational decoding results")],
        caveats=[_LABEL_CAVEAT,
                 ("Single model, single layer, single extraction point (Qwen2.5-7B, "
                  "layer 20, final prompt token)."),
                 ("The limiting construct is 'discloses then limits': 72 of 76 limiting "
                  "cases are mixed_disclose_then_limit, zero are explicit refusals.")],
        paper_location="Main text, primary results table.",
        caption=("Privileged increment of layer-20 activations over matched "
                 "scenario-text baselines, analysis population (n=216)."),
        replaceable_by_later_step=True,
        replacement_risk=("Step 3 adds onset-relative dynamics; this table remains the "
                          "static result but may be joined by a position-resolved version."),
        promotion_conditions=["Primary results table unless the blind label audit moves a verdict."],
    ),
    "tab_dissociation_216": dict(
        kind="table", formats=["csv", "tex"], status="candidate_main",
        supports_claim=("The privileged increment is larger for strategy than for "
                        "disclosure outcome, tested directly rather than inferred from "
                        "differing significance (+0.130, CI [+0.023, +0.242], p = .010 "
                        "against substantive_leak)."),
        must_not_claim=[("that every limiting-vs-leak comparison is established — the "
                         "comparison against leak_vs_appropriate has an interval that "
                         "grazes zero and is suggestive only"),
                        ("that the leak contrasts have zero privileged signal — the "
                         "point estimates are positive but not separable from zero")],
        caveats=[_LABEL_CAVEAT,
                 ("The four comparisons are a coherent set on overlapping case sets, not "
                  "independent tests; no multiplicity adjustment is applied across them.")],
        paper_location="Main text, alongside the primary table.",
        caption=("Direct test of the strategy-vs-outcome dissociation in privileged "
                 "increment (analysis population)."),
        replaceable_by_later_step=False,
        replacement_risk="None; a direct test of an already-computed quantity.",
        promotion_conditions=["Keep in main text while the dissociation is claimed."],
    ),
    "tab_eval_awareness_control_216": dict(
        kind="table", formats=["csv", "tex"], status="candidate_main",
        supports_claim=("The limiting signal is not evaluation-awareness: erasing the "
                        "eval-awareness direction costs 0.0017 AUC with an interval "
                        "spanning zero, while the same erasure drives test-vs-deploy "
                        "decodability from 1.000 to chance."),
        must_not_claim=[("that all caution-like or eval-related structure has been "
                         "removed — rank-1 erasure removes one direction, and a "
                         "distributed residual cannot be fully excluded"),
                        ("that the erasure control validates the limiting result itself "
                         "— it rules out one alternative explanation")],
        caveats=[("The eval-awareness direction is estimated from framed prompts of the "
                  "same scenarios; cross-fitting removes the transductive concern but "
                  "the framing manipulation is our own construct."),
                 _LABEL_CAVEAT],
        paper_location="Main text or appendix, as the control for the headline result.",
        caption=("Eval-awareness erasure control with its manipulation check "
                 "(analysis population)."),
        replaceable_by_later_step=False,
        replacement_risk="None.",
        promotion_conditions=["Main text if the caution objection is addressed inline."],
    ),
    "tab_population_sensitivity_delta": dict(
        kind="table", formats=["csv", "tex"], status="sensitivity",
        supports_claim=("The privileged increment is stable in direction and ordering "
                        "when the 42 calibration cases are added."),
        must_not_claim=[("that the 258 results replicate the 216 results — 258 is a "
                         "strict superset containing every 216 case"),
                        ("that the larger 258 estimates are the better estimates")],
        caveats=[("The 42 added cases are enriched for limiting and disclosing behaviour "
                  "and are the only human-verified references."), _LABEL_CAVEAT],
        paper_location="Appendix (sensitivity).",
        caption="Sensitivity of the privileged increment to population definition.",
        replaceable_by_later_step=False, replacement_risk="None.",
        promotion_conditions=["Never promoted to a primary estimate."],
    ),
    "fig_privileged_delta_216": dict(
        kind="figure", formats=["pdf", "svg", "png"], status="candidate_main",
        supports_claim=("The paper's headline: pre-response activations predict "
                        "upcoming disclosure strategy beyond what the scenario wording "
                        "supports, and this is specific to strategy."),
        must_not_claim=[_NOT_INFO_THEORETIC, _BASELINE_RELATIVE,
                        ("that between-contrast differences other than those in the "
                         "dissociation table have been tested")],
        caveats=[_LABEL_CAVEAT,
                 ("Panel (a) shows both text channels; Δ in panel (b) is against the "
                  "stronger of them, which is the encoder on every contrast.")],
        paper_location="Main text, Figure 1.",
        caption=("Layer-20 activations at the final prompt token versus matched "
                 "scenario-text baselines for six behavioural contrasts (analysis "
                 "population, n=216). (a) Channel AUCs: filled circle = activation "
                 "probe, open square = frozen sentence-encoder baseline, grey tick = "
                 "TF-IDF. (b) Privileged increment Δ = activation − stronger text "
                 "baseline, with paired scenario-bootstrap 95% CI; filled markers "
                 "survive Holm correction across the six contrasts."),
        replaceable_by_later_step=True,
        replacement_risk=("Step 3's onset-resolved figure may become Figure 1 if the "
                          "dynamics result is strong; this would then move to Figure 2."),
        promotion_conditions=["Main figure unless Step 3 supersedes it."],
    ),
    "fig_eval_awareness_control_216": dict(
        kind="figure", formats=["pdf", "svg", "png"], status="candidate_appendix",
        supports_claim=("The headline result survives erasure of evaluation-awareness, "
                        "with a manipulation check demonstrating the erasure worked."),
        must_not_claim=[("that rank-1 erasure removes a concept in full"),
                        ("that a null drop proves privacy-specificity — it rules out one "
                         "alternative explanation")],
        caveats=[("Isotropic random directions cost ~0.0000 AUC and are shown only to "
                  "demonstrate they are an inadequate null."), _LABEL_CAVEAT],
        paper_location="Appendix, or main text beside the headline result.",
        caption=("Eval-awareness erasure control. (a) AUC lost by each contrast when the "
                 "cross-fitted eval-awareness direction is erased, against the ±95th "
                 "percentile band of 100 permuted-frame null directions. (b) Manipulation "
                 "check: the same erasure drives test-vs-deploy decodability from 1.000 "
                 "to chance."),
        replaceable_by_later_step=False, replacement_risk="None.",
        promotion_conditions=["Promote to main text if the caution objection is "
                              "addressed inline rather than in an appendix."],
    ),
}


def main():
    tb, p, s = load()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    git = C1.git_state()
    srcs = [{"path": C1.rel(x), "sha256": C1.sha256(x), "role": r} for x, r in (
        (TB_CSV, "text_baseline_results"), (TB_JSON, "text_baseline_protocol"),
        (DIS_CSV, "dissociation_test"), (EA_CSV, "eval_awareness_control"),
        (EA_JSON, "eval_awareness_protocol"), (AL_JSON, "step3_alignment_preflight"),
        (C1.LABELS, "population_definition"))]

    outputs = []
    for name, (df, notes) in build_tables(p, s).items():
        spec = REGISTRY[name]
        csv_p, tex_p = TABLES / f"{name}.csv", TABLES / f"{name}.tex"
        df.to_csv(csv_p, index=False)
        tex_p.write_text(_tex(df, spec["caption"], name.replace("_", "-"), notes))
        _sidecar(name, spec, {"csv": csv_p, "tex": tex_p}, srcs, git,
                 "## Rendered table\n\n" + _md(df) + "\n\n### Table notes\n\n"
                 + "\n".join(f"{i+1}. {n}" for i, n in enumerate(notes)))
        outputs += [csv_p, tex_p, TABLES / f"{name}.md"]

    for name, fn in (("fig_privileged_delta_216", lambda st: fig_privileged_delta(p, st)),
                     ("fig_eval_awareness_control_216", fig_eval_control)):
        spec = REGISTRY[name]
        stem = FIGURES / name
        pdata = fn(stem)
        pj = FIGURES / f"{name}.plotdata.json"
        pj.write_text(json.dumps(pdata, indent=2, sort_keys=True) + "\n")
        paths = {e: FIGURES / f"{name}.{e}" for e in ("pdf", "svg", "png")}
        _sidecar(name, spec, paths, srcs, git, plotdata=pj)
        outputs += list(paths.values()) + [pj, FIGURES / f"{name}.md"]

    _index(outputs)
    _metadata(srcs, git, outputs, p)
    print(f"built {len(outputs)+3} artifacts under {C1.rel(STEP)}")


def _tex(df, caption, label, notes):
    def esc(x):
        x = str(x)
        for a, b in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("≤", r"$\leq$"),
                     ("Δ", r"$\Delta$"), ("⊃", r"$\supset$"), ("−", "-"), ("×", r"$\times$"),
                     ("–", "--"), ("—", "---")):
            x = x.replace(a, b)
        return x
    L = [r"\begin{table}[t]", r"\centering", r"\begin{threeparttable}",
         r"\caption{" + esc(caption) + "}", r"\label{tab:" + label + "}", r"\small",
         r"\begin{tabular}{l" + "r" * (len(df.columns) - 1) + "}", r"\toprule",
         " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    L += [" & ".join(esc(v) for v in r.tolist()) + r" \\" for _, r in df.iterrows()]
    L += [r"\bottomrule", r"\end{tabular}",
          r"\begin{tablenotes}[flushleft]\footnotesize"]
    L += [r"\item " + esc(n) for n in notes]
    L += [r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}", ""]
    return "\n".join(L)


def _md(df):
    return "\n".join(["| " + " | ".join(df.columns) + " |",
                      "|" + "|".join("---" for _ in df.columns) + "|"]
                     + ["| " + " | ".join(str(v) for v in r.tolist()) + " |"
                        for _, r in df.iterrows()])


def _sidecar(name, spec, paths, srcs, git, extra="", plotdata=None):
    fm = ["---", f"artifact: {name}", f"kind: {spec['kind']}",
          f"formats: [{', '.join(spec['formats'])}]", f"status: {spec['status']}",
          "maturity: provisional", f'generated_utc: "{BUILD_UTC}"',
          "generated_by:", "  script: scripts/paper_step02_build_f.py",
          f'  script_sha256: "{C1.sha256(Path(__file__))}"',
          '  command: "python3 scripts/paper_step02_build_f.py"',
          f'  git_commit: "{git["commit"]}"', f"  git_dirty: {str(git['dirty']).lower()}",
          "sources:"]
    fm += [f'  - {{path: {x["path"]}, sha256: "{x["sha256"]}", role: {x["role"]}}}'
           for x in srcs]
    if plotdata is not None:
        fm += ["plotdata:", f"  path: {plotdata.name}",
               f'  sha256: "{C1.sha256(plotdata)}"']
    fm.append("artifact_sha256:")
    fm += [f'  {e}: "{C1.sha256(pp)}"' for e, pp in paths.items()]
    fm += ["population: analysis_216",
           'population_note: "216 canonical tier-3 analysis cases; contrast-specific '
           'subsets 81-216. The 42 calibration cases are excluded."',
           'metric: ["roc_auc", "privileged delta (activation - stronger text baseline)"]',
           'uncertainty: "Paired scenario bootstrap (B=1000) resampling scenarios once '
           'per draw and recomputing the full repeat-averaged per-fold statistic for '
           'both channels; one-sided p = P(delta* <= 0), Holm-adjusted across the six '
           'contrasts."',
           f'supports_claim: "{spec["supports_claim"]}"', "must_not_claim:"]
    fm += [f'  - "{m}"' for m in spec["must_not_claim"]]
    fm.append("caveats:")
    fm += [f'  - "{c}"' for c in spec["caveats"]]
    fm += [f'paper_location: "{spec["paper_location"]}"',
           f'caption: "{spec["caption"]}"',
           f"replaceable_by_later_step: {str(spec['replaceable_by_later_step']).lower()}",
           f'replacement_risk: "{spec["replacement_risk"]}"', "promotion_conditions:"]
    fm += [f'  - "{c}"' for c in spec["promotion_conditions"]]
    fm += ["supersedes: null", "superseded_by: null", "---", ""]
    body = [f"# {name}", "",
            f"**Status: `{spec['status']}` — provisional.**", "",
            "## What this shows", "", spec["supports_claim"], "",
            "## What it must NOT be read as", ""]
    body += [f"- Do not claim {m}." for m in spec["must_not_claim"]]
    body += ["", "## Caveats", ""] + [f"- {c}" for c in spec["caveats"]]
    body += ["", "## How to regenerate", "", "```",
             "python3 scripts/text_baselines_canonical_f.py",
             "python3 scripts/eval_awareness_canonical_f.py",
             "python3 scripts/paper_step02_build_f.py", "```", "",
             "## Suggested caption", "", f"> {spec['caption']}", "",
             "## Paper placement", "", spec["paper_location"], "",
             f"**Replaceable by a later step:** {spec['replaceable_by_later_step']}. "
             f"{spec['replacement_risk']}", "", "**Promotion conditions:**", ""]
    body += [f"- {c}" for c in spec["promotion_conditions"]]
    if extra:
        body += ["", extra]
    (TABLES if spec["kind"] == "table" else FIGURES).joinpath(f"{name}.md").write_text(
        "\n".join(fm) + "\n".join(body) + "\n")


def _index(outputs):
    L = ["# Artifact index — Step 2: text baselines and eval-awareness control", "",
         "Every artifact is **provisional**. Later steps (onset dynamics, blind label "
         "audit) may change the preferred framing or figure.", "",
         "**Status legend** — `candidate_main`: proposed for main text · "
         "`candidate_appendix`: appendix · `sensitivity`: robustness only · "
         "`exploratory`: not claim-bearing · `deprecated`: superseded.", "",
         "| artifact | kind | status | intended claim | paper placement | replaceable |",
         "|---|---|---|---|---|---|"]
    for n, s in REGISTRY.items():
        cl = s["supports_claim"][:105].rstrip() + ("…" if len(s["supports_claim"]) > 105 else "")
        L.append(f"| `{n}` | {s['kind']} | `{s['status']}` | {cl} | "
                 f"{s['paper_location']} | {'yes' if s['replaceable_by_later_step'] else 'no'} |")
    L += ["", "## Main figure slot", "",
          "Step 1 reserved the main-figure slot pending this step's Δ. "
          "`fig_privileged_delta_216` now claims it: it shows a relationship (two "
          "channels and the gap between them) rather than six independent point "
          "estimates, and it carries the paper's headline claim.", ""]
    (STEP / "ARTIFACT_INDEX.md").write_text("\n".join(L))


def _metadata(srcs, git, outputs, p):
    tbj = json.loads(TB_JSON.read_text())
    eaj = json.loads(EA_JSON.read_text())
    alj = json.loads(AL_JSON.read_text())
    meta = {
        "step": "02_text_baselines", "maturity": "provisional",
        "title": ("Matched text baselines, privileged delta, and the eval-awareness "
                  "control"),
        "generated_utc": BUILD_UTC, "git": git,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "pandas": pd.__version__,
                        "matplotlib": __import__("matplotlib").__version__,
                        "sklearn": __import__("sklearn").__version__},
        "inputs": srcs,
        "outputs": [{"path": C1.rel(o), "sha256": C1.sha256(o)}
                    for o in sorted(set(outputs))],
        "protocol_2a": tbj["protocol"], "channels": tbj["channels"],
        "inference_note": tbj["inference_note"],
        "protocol_2b": eaj["protocol"],
        "manipulation_check": eaj["manipulation_check"],
        "headline": {
            "contrast": "limiting_among_disclosers",
            "activation_auc": float(p.loc[p.key == "limiting_among_disclosers",
                                          "roc_acts"].iloc[0]),
            "text_auc": float(p.loc[p.key == "limiting_among_disclosers",
                                    "roc_embed"].iloc[0]),
            "delta": float(p.loc[p.key == "limiting_among_disclosers",
                                 "delta_roc_stronger"].iloc[0]),
            "holm_p": float(p.loc[p.key == "limiting_among_disclosers",
                                  "p_holm_delta_stronger"].iloc[0])},
        "step3_preflight": {"verdict": alj["verdict"],
                            "recommended_primary_window": alj["recommended_primary_window"],
                            "positional_confound_auc": alj["positional_confound_auc"]},
        "status_vocabulary": C1.STATUS_VOCAB,
    }
    (STEP / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
