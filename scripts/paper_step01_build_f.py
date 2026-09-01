#!/usr/bin/env python3
"""paper_step01_build_f.py — build the Step-1 paper artifact package.

Emits candidate tables, candidate figures, per-artifact sidecar notes,
ARTIFACT_INDEX.md, run_metadata.json, SOURCES.sha256, README.md and
PRESENTATION_NOTES.md into results/paper_pipeline/01_statistics_protocol/.

Every number is loaded from results/probe_contrasts_canonical_v2_f.{csv,json},
results/probe_contrasts_canonical_f.csv (v1, provenance) and
results/behavior_labels_tier3_canonical_f.csv. Nothing is transcribed.

Idempotent: running twice produces byte-identical output.
No API, GPU, judge, label, activation or benchmark changes. Writes only inside
the step directory.

Usage:  python3 scripts/paper_step01_build_f.py
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

import paper_step01_common_f as C          # noqa: E402
from figstyle_f import (                    # noqa: E402
    C_INK, C_MUTED, C_GRID, FAMILY_COLOR, hanley_se, save_all, titles, plt,
)

# Deterministic build timestamp: SOURCE_DATE_EPOCH if set, else file mtime of the
# authoritative results (never "now" — that would break idempotence).
_EPOCH = int(os.environ.get("SOURCE_DATE_EPOCH", C.V2_CSV.stat().st_mtime))
BUILD_UTC = datetime.fromtimestamp(_EPOCH, timezone.utc).isoformat()


# ══ helpers ══════════════════════════════════════════════════════════════════

def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _tex_table(df: pd.DataFrame, caption: str, label: str, notes: list[str],
               col_fmt: str | None = None) -> str:
    """booktabs + threeparttable, escaped, no pandas styler dependency."""
    def esc(s):
        s = str(s)
        for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                     ("&", r"\&"), ("#", r"\#"), ("≤", r"$\leq$"), ("⊃", r"$\supset$"),
                     ("–", "--"), ("—", "---"), ("×", r"$\times$")):
            s = s.replace(a, b)
        return s
    cols = list(df.columns)
    fmt = col_fmt or ("l" + "r" * (len(cols) - 1))
    lines = [r"\begin{table}[t]", r"\centering", r"\begin{threeparttable}",
             r"\caption{" + esc(caption) + "}", r"\label{tab:" + label + "}",
             r"\small", r"\begin{tabular}{" + fmt + "}", r"\toprule",
             " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(esc(v) for v in row.tolist()) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        lines.append(r"\begin{tablenotes}[flushleft]\footnotesize")
        for n in notes:
            lines.append(r"\item " + esc(n))
        lines.append(r"\end{tablenotes}")
    lines += [r"\end{threeparttable}", r"\end{table}", ""]
    return "\n".join(lines)


def _md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r.tolist()) + " |"
            for _, r in df.iterrows()]
    return "\n".join([head, sep] + rows)


# ══ tables ═══════════════════════════════════════════════════════════════════

def build_tab_primary(v2: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = C.scored(v2, C.PRIMARY_POP)
    out = pd.DataFrame({
        "contrast": d["display"],
        "family": d["family"].map(C.FAMILY_LABEL),
        "n": d["n"].astype(int),
        "n_pos": d["n_pos"].astype(int),
        "prevalence": d["prevalence"].map(C.fmt_num),
        "ROC-AUC": d["roc_auc"].map(C.fmt_num),
        "ROC Hanley 95% CI": [C.fmt_ci(a, b) for a, b in
                              zip(d["roc_auc_hanley_lo"], d["roc_auc_hanley_hi"])],
        "ROC boot 95% CI": [C.fmt_ci(a, b) for a, b in
                            zip(d["roc_auc_boot_lo"], d["roc_auc_boot_hi"])],
        "PR-AUC": d["pr_auc"].map(C.fmt_num),
        "PR null mean": d["null_pr_auc_mean"].map(C.fmt_num),
        "PR excess over null": d["pr_excess_over_null"].map(C.fmt_num),
        "PR boot 95% CI": [C.fmt_ci(a, b) for a, b in
                           zip(d["pr_auc_boot_lo"], d["pr_auc_boot_hi"])],
        "Holm p (PR)": d["p_holm_pr_auc"].map(C.fmt_p),
        "Holm p (ROC)": d["p_holm_roc_auc"].map(C.fmt_p),
        "modal C": [f"{c:g}" for c in d["modal_C"]],
        "C at grid floor": ["yes" if b else "no" for b in d["at_grid_edge"]],
        "verdict": d["verdict"],
    })
    notes = [
        "PR-AUC is the primary metric, pre-specified in the analysis script docstring "
        "(committed before the corrected run); it is not registered in "
        "docs/PREREGISTRATION.md. ROC-AUC is supporting evidence and never rescues a "
        "PR-failing contrast.",
        "Verdict: supported = Holm-adjusted permutation p on PR-AUC <= .05; suggestive = "
        "fails that but Holm-adjusted p on ROC-AUC <= .05; no_evidence = neither.",
        f"p-values are label-permutation with n_perm={C.N_PERM} through the identical "
        f"per-fold statistic, so the smallest attainable Holm-adjusted value is "
        f"{C.N_CONTRASTS_ADJUSTED}/{C.N_PERM + 1} = {C.P_HOLM_FLOOR:.3f}; entries at that "
        "floor are shown as <=.012.",
        "The PR baseline is the empirical permutation-null mean, not class prevalence: "
        "cross-validated average precision is upward-biased relative to prevalence by "
        "~0.05-0.06 at these sample sizes.",
        "The six contrasts are separate tests on overlapping case sets and are not "
        "independent; no between-contrast comparison was performed and none is powered.",
        "Intervals reflect sampling uncertainty conditional on the labels only. All 216 "
        "analysis labels are single-judge and provisional.",
    ]
    return out, notes


def build_tab_correction(v1: pd.DataFrame, v2: pd.DataFrame):
    a = C.scored(v2, C.PRIMARY_POP)
    b = v1[(v1["status"] == "scored")
           & v1["contrast"].str.endswith("|" + C.PRIMARY_POP)].copy()
    b["key"] = b["contrast"].str.split("|").str[0]
    b = b.set_index("key").loc[C.CONTRAST_KEYS].reset_index()
    out = pd.DataFrame({
        "contrast": a["display"],
        "ROC v1 (pooled)": b["roc_auc"].map(C.fmt_num),
        "ROC v2 (per-fold)": a["roc_auc"].map(C.fmt_num),
        "Δ ROC": (a["roc_auc"].values - b["roc_auc"].values),
        "PR v1 (pooled)": b["pr_auc"].map(C.fmt_num),
        "PR v2 (per-fold)": a["pr_auc"].map(C.fmt_num),
        "Δ PR": (a["pr_auc"].values - b["pr_auc"].values),
        "raw p PR v1": b["p_perm_pr_auc"].map(lambda x: f"{x:.3f}".replace("0.", ".")),
        "raw p PR v2": a["p_perm_pr_auc"].map(lambda x: f"{x:.3f}".replace("0.", ".")),
        "Holm p PR v2": a["p_holm_pr_auc"].map(C.fmt_p),
    })
    out["Δ ROC"] = out["Δ ROC"].map(lambda x: f"{x:+.3f}")
    out["Δ PR"] = out["Δ PR"].map(lambda x: f"{x:+.3f}")
    out.insert(len(out.columns), "Holm p PR v1", "—")
    notes = [
        "Identical data, contrasts, n, n_pos, seeds, fold assignments and regularisation "
        "grid under both protocols; only the summary statistic differs.",
        "v1 pooled raw predict_proba scores across outer folds that each selected their own "
        "C. Probability scales differ by orders of magnitude across C, so the pooled ranking "
        "is corrupted at fold boundaries and the estimate is biased downward "
        "(Forman & Scholz 2010; Airola et al. 2011).",
        "v1 p-values were computed against a pooled-score null and are shown for provenance "
        "only; they are not a valid significance comparison against v2.",
        "v1 applied no multiplicity adjustment, hence the em-dash in the Holm column.",
        "Corrections not visible in this table: the permutation null now uses the identical "
        "per-fold statistic (Ojala & Garriga 2010), and Holm adjustment was added.",
    ]
    return out, notes


def build_tab_intervals(v1: pd.DataFrame, v2: pd.DataFrame):
    a = C.scored(v2, C.PRIMARY_POP)
    b = v1[(v1["status"] == "scored")
           & v1["contrast"].str.endswith("|" + C.PRIMARY_POP)].copy()
    b["key"] = b["contrast"].str.split("|").str[0]
    b = b.set_index("key").loc[C.CONTRAST_KEYS].reset_index()
    split_w = (b["roc_auc_hi"] - b["roc_auc_lo"]).values
    han_w = (a["roc_auc_hanley_hi"] - a["roc_auc_hanley_lo"]).values
    boot_w = (a["roc_auc_boot_hi"] - a["roc_auc_boot_lo"]).values
    out = pd.DataFrame({
        "contrast": a["display"],
        "v1 split band width": [f"{x:.3f}" for x in split_w],
        "Hanley 95% width": [f"{x:.3f}" for x in han_w],
        "bootstrap 95% width": [f"{x:.3f}" for x in boot_w],
        "Hanley / split": [f"{x:.2f}x" for x in han_w / split_w],
        "bootstrap / split": [f"{x:.2f}x" for x in boot_w / split_w],
    })
    notes = [
        "Widths are for ROC-AUC in the 216 analysis population.",
        "The v1 band is the 2.5/97.5 percentile over 20 repeated cross-validation splits of "
        "the SAME cases. It measures split variability, not sampling uncertainty, and must "
        "never be presented as a confidence interval (Nadeau & Bengio 2003; Bates, Hastie & "
        "Tibshirani 2023).",
        "Hanley & McNeil (1982) assumes a single fixed scoring rule applied to independent "
        "cases; an average of per-fold AUCs from models refit on each training split "
        "violates that, so it is reported as an approximation alongside the bootstrap.",
        "The bootstrap is a stratified prediction-resampling interval (n_boot=1000), "
        "conditional on the fitted models; it is not a test against the null.",
    ]
    return out, notes


def build_tab_sensitivity(v2: pd.DataFrame):
    a = C.scored(v2, C.PRIMARY_POP)
    s = C.scored(v2, C.SENSITIVITY_POP)
    comp = C.population_composition()
    rows = []
    for name, key in (("analysis 216 (primary)", "analysis_216"),
                      ("+42 calibration cases", "calibration_42"),
                      ("all 258 (superset)", "all_258")):
        c = comp[key]
        ver = ", ".join(f"{k}={v}" for k, v in sorted(c["reference_verification"].items()))
        rows.append({
            "row": name, "n": c["n"],
            "% limiting": f"{c['pct_limiting']:.1%}",
            "% substantive leak": f"{c['pct_substantive_leak']:.1%}",
            "% broad breach": f"{c['pct_broad_breach']:.1%}",
            "reference verification": ver,
        })
    comp_df = pd.DataFrame(rows)
    res = pd.DataFrame({
        "contrast": a["display"],
        "n (216)": a["n"].astype(int),
        "ROC (216)": a["roc_auc"].map(C.fmt_num),
        "PR (216)": a["pr_auc"].map(C.fmt_num),
        "Holm p PR (216)": a["p_holm_pr_auc"].map(C.fmt_p),
        "verdict (216)": a["verdict"],
        "n (258)": s["n"].astype(int),
        "ROC (258)": s["roc_auc"].map(C.fmt_num),
        "PR (258)": s["pr_auc"].map(C.fmt_num),
        "Holm p PR (258)": s["p_holm_pr_auc"].map(C.fmt_p),
        "verdict (258)": s["verdict"],
    })
    notes = [
        "258 is a strict SUPERSET of 216: every analysis case also appears in the 258 "
        "population. These are not independent samples and this is not a replication.",
        "The 42 additional cases are calibration cases that deliberately oversampled the "
        "historical `refused` class, so they are enriched for limiting and disclosing "
        "behaviour relative to the analysis population.",
        "The 42 calibration cases are also the only cases with human-verified references; "
        "all 216 analysis references are provisional and single-judge. Composition and "
        "label quality are therefore both confounded with population.",
        "The 216 analysis population is primary for every claim. The 258 column is a "
        "sensitivity analysis and is never the primary estimate.",
    ]
    return comp_df, res, notes


# ══ figures ══════════════════════════════════════════════════════════════════

def _rows_with_gaps(d: pd.DataFrame):
    """y positions, top-to-bottom, with a whitespace gap between families."""
    ys, fam_span, y = [], {}, 0.0
    prev = None
    for _, r in d.iterrows():
        if prev is not None and r["family"] != prev:
            y += 0.9
        ys.append(y)
        fam_span.setdefault(r["family"], []).append(y)
        prev = r["family"]
        y += 1.0
    return np.array(ys), fam_span


def fig_dual_metric(v2: pd.DataFrame, stem: Path) -> dict:
    d = C.scored(v2, C.PRIMARY_POP)
    ys, fam_span = _rows_with_gaps(d)
    colors = [FAMILY_COLOR[f] for f in d["family"]]
    marks = [C.marker_code(r.p_holm_pr_auc, r.p_holm_roc_auc) for r in d.itertuples()]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(8.6, 3.6),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.16})

    def draw_point(ax, x, y, color, code, z=5):
        if code == "filled":
            ax.plot(x, y, "o", color=color, ms=6.2, mec="white", mew=0.9, zorder=z)
        elif code == "open_dot":
            ax.plot(x, y, "o", mfc="white", mec=color, mew=1.6, ms=6.6, zorder=z)
            ax.plot(x, y, "o", color=color, ms=2.2, zorder=z + 1)
        else:
            ax.plot(x, y, "o", mfc="white", mec=color, mew=1.6, ms=6.6, zorder=z)

    # ── left: ROC ────────────────────────────────────────────────────────────
    axL.axvline(0.5, color=C_MUTED, lw=0.8, ls=":", zorder=1)
    for i, (y, c, code) in enumerate(zip(ys, colors, marks)):
        r = d.iloc[i]
        axL.plot([r.roc_auc_hanley_lo, r.roc_auc_hanley_hi], [y, y],
                 color=c, lw=2.1, solid_capstyle="butt", zorder=3)
        for xx in (r.roc_auc_hanley_lo, r.roc_auc_hanley_hi):
            axL.plot([xx, xx], [y - 0.13, y + 0.13], color=c, lw=1.5, zorder=3)
        axL.plot([r.roc_auc_boot_lo, r.roc_auc_boot_hi], [y + 0.26, y + 0.26],
                 color=c, lw=0.9, alpha=0.55, zorder=2)
        axL.plot(r.null_roc_auc_mean, y, "|", color=C_MUTED, ms=5, mew=0.9, zorder=4)
        draw_point(axL, r.roc_auc, y, c, code)
    axL.set_xlim(0.44, 0.96)
    axL.set_xlabel("ROC-AUC  (supporting metric)")
    axL.set_title("(a) ROC-AUC", loc="left", color=C_INK)
    axL.text(0.5, -0.72, "chance", fontsize=7, color=C_MUTED,
             ha="center", va="bottom")

    # ── right: PR ────────────────────────────────────────────────────────────
    for i, (y, c, code) in enumerate(zip(ys, colors, marks)):
        r = d.iloc[i]
        axR.plot([r.null_pr_auc_mean, r.pr_auc], [y - 0.26, y - 0.26],
                 color=c, lw=1.1, alpha=0.42, zorder=2)
        axR.plot(r.null_pr_auc_mean, y - 0.26, "s", mfc="white", mec=C_MUTED,
                 mew=1.1, ms=4.6, zorder=4)
        axR.plot(r.prevalence, y - 0.26, "|", color="#a8a6a1", ms=6, mew=1.4, zorder=3)
        axR.plot([r.pr_auc_boot_lo, r.pr_auc_boot_hi], [y, y],
                 color=c, lw=2.1, solid_capstyle="butt", zorder=3)
        for xx in (r.pr_auc_boot_lo, r.pr_auc_boot_hi):
            axR.plot([xx, xx], [y - 0.13, y + 0.13], color=c, lw=1.5, zorder=3)
        draw_point(axR, r.pr_auc, y, c, code)
    axR.set_xlim(0.18, 1.06)
    axR.set_xlabel("PR-AUC  (primary metric)")
    axR.set_title("(b) PR-AUC vs its permutation null", loc="left", color=C_INK)

    # ── shared y formatting ──────────────────────────────────────────────────
    labels = [f"{r.display}\n(n={int(r.n)}, {int(r.n_pos)}+)" for r in d.itertuples()]
    for ax in (axL, axR):
        ax.set_ylim(ys.max() + 0.7, -0.8)
        ax.set_yticks(ys)
        ax.grid(axis="y", visible=False)
    axL.set_yticklabels(labels, fontsize=7.4)
    axR.set_yticklabels([])

    # family brackets, clear of the row labels
    for fam, yy in fam_span.items():
        y0, y1 = min(yy) - 0.34, max(yy) + 0.34
        axL.plot([-0.90, -0.90], [y0, y1], transform=axL.get_yaxis_transform(),
                 color=FAMILY_COLOR[fam], lw=1.6, clip_on=False, zorder=5)
        axL.text(-0.94, float(np.mean(yy)), C.FAMILY_LABEL[fam].replace(" ", "\n"),
                 transform=axL.get_yaxis_transform(), ha="center", va="center",
                 fontsize=6.8, color=FAMILY_COLOR[fam], fontweight="bold",
                 rotation=90, linespacing=0.95, clip_on=False)

    # p-value column at the right edge of panel b
    axR.text(1.10, -0.75, "Holm p\nPR / ROC", transform=axR.get_yaxis_transform(),
             fontsize=6.8, color=C_MUTED, ha="left", va="center", clip_on=False)
    for i, y in enumerate(ys):
        r = d.iloc[i]
        axR.text(1.10, y, f"{C.fmt_p(r.p_holm_pr_auc)} / {C.fmt_p(r.p_holm_roc_auc)}",
                 transform=axR.get_yaxis_transform(), fontsize=6.8,
                 color=C_INK if r.verdict == "supported" else C_MUTED,
                 ha="left", va="center", clip_on=False)

    titles(fig,
           "Decodability of six behavioural contrasts, layer-20 activations "
           "at the final prompt token",
           "analysis population, n=216 scenarios (contrast subsets 81–216)  ·  "
           "marker fill: filled = survives Holm on PR-AUC, open+dot = ROC only, "
           "open = neither  ·  provisional single-judge labels",
           3.5)
    fig.text(0.0, -0.16,
             "Six separate tests on overlapping case sets. Intervals are marginal, not "
             "simultaneous; no between-contrast comparison was performed and none is "
             "powered.\n(a) thick bar = Hanley 95% CI, thin bar above = bootstrap 95% CI, "
             "grey tick = ROC permutation null.  (b) open square = PR permutation null "
             "(the correct baseline); faint tick = class prevalence, which is NOT the PR "
             f"chance level.  n_perm={C.N_PERM}, so ≤{C.P_HOLM_FLOOR:.3f} is the "
             "resolution floor.",
             fontsize=6.4, color=C_MUTED, ha="left", va="top")

    save_all(fig, str(stem))
    return {
        "contrast_key": d["key"].tolist(),
        "display": d["display"].tolist(),
        "family": d["family"].tolist(),
        "y": ys.tolist(),
        "n": d["n"].astype(int).tolist(),
        "n_pos": d["n_pos"].astype(int).tolist(),
        "prevalence": d["prevalence"].tolist(),
        "roc_auc": d["roc_auc"].tolist(),
        "roc_hanley_lo": d["roc_auc_hanley_lo"].tolist(),
        "roc_hanley_hi": d["roc_auc_hanley_hi"].tolist(),
        "roc_boot_lo": d["roc_auc_boot_lo"].tolist(),
        "roc_boot_hi": d["roc_auc_boot_hi"].tolist(),
        "null_roc_auc_mean": d["null_roc_auc_mean"].tolist(),
        "pr_auc": d["pr_auc"].tolist(),
        "pr_boot_lo": d["pr_auc_boot_lo"].tolist(),
        "pr_boot_hi": d["pr_auc_boot_hi"].tolist(),
        "null_pr_auc_mean": d["null_pr_auc_mean"].tolist(),
        "pr_excess_over_null": d["pr_excess_over_null"].tolist(),
        "p_holm_pr_auc": d["p_holm_pr_auc"].tolist(),
        "p_holm_roc_auc": d["p_holm_roc_auc"].tolist(),
        "marker_code": marks,
        "verdict": d["verdict"].tolist(),
    }


def fig_correction(v1: pd.DataFrame, v2: pd.DataFrame, stem: Path) -> dict:
    a = C.scored(v2, C.PRIMARY_POP)
    b = v1[(v1["status"] == "scored")
           & v1["contrast"].str.endswith("|" + C.PRIMARY_POP)].copy()
    b["key"] = b["contrast"].str.split("|").str[0]
    b = b.set_index("key").loc[C.CONTRAST_KEYS].reset_index()

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(9.4, 3.4),
        gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.30})

    # ── A: paired slope ──────────────────────────────────────────────────────
    for i, r in a.iterrows():
        c = FAMILY_COLOR[r["family"]]
        for metric, ls in (("roc_auc", "-"), ("pr_auc", "--")):
            y0, y1 = float(b.iloc[i][metric]), float(r[metric])
            axA.plot([0, 1], [y0, y1], ls, color=c, lw=1.5, alpha=0.9, zorder=3)
            axA.plot([0, 1], [y0, y1], "o", color=c, ms=3.4, mec="white",
                     mew=0.7, zorder=4)
    axA.set_xlim(-0.28, 2.05)
    axA.set_xticks([0, 1])
    axA.set_xticklabels(["v1\npooled scores", "v2\nper-fold mean"], fontsize=8)
    axA.set_ylabel("AUC")
    axA.grid(axis="x", visible=False)
    axA.set_title("(a) every contrast rises", loc="left", color=C_INK)
    # right-edge labels on the ROC lines, greedily dodged so they never overlap
    lab = sorted(((float(r["roc_auc"]), r["display"], FAMILY_COLOR[r["family"]])
                  for _, r in a.iterrows()), reverse=True)
    span = float(max(a["pr_auc"].max(), a["roc_auc"].max())
                 - min(b["pr_auc"].min(), b["roc_auc"].min()))
    gap, placed = 0.043 * span, []
    for yv, text, col in lab:
        y = yv if not placed else min(yv, placed[-1] - gap)
        placed.append(y)
        axA.plot([1.02, 1.10], [yv, y], color=col, lw=0.6, alpha=0.7, zorder=2)
        axA.text(1.13, y, text, fontsize=6.0, color=col, va="center", ha="left")
    d_roc = a["roc_auc"].values - b["roc_auc"].values
    axA.text(-0.24, 0.30, f"ROC Δ  +{d_roc.min():.3f} to +{d_roc.max():.3f}",
             fontsize=7, color=C_INK, va="top", ha="left",
             transform=axA.get_xaxis_transform())
    axA.text(-0.24, 0.24, "solid = ROC-AUC   dashed = PR-AUC",
             fontsize=6.6, color=C_MUTED, va="top", ha="left",
             transform=axA.get_xaxis_transform())

    # ── B: interval widths ───────────────────────────────────────────────────
    split_w = (b["roc_auc_hi"] - b["roc_auc_lo"]).values
    han_w = (a["roc_auc_hanley_hi"] - a["roc_auc_hanley_lo"]).values
    boot_w = (a["roc_auc_boot_hi"] - a["roc_auc_boot_lo"]).values
    yy = np.arange(len(a))[::-1].astype(float)
    h = 0.26
    axB.barh(yy + h, split_w, height=h, color="white", edgecolor=C_MUTED,
             hatch="////", lw=0.8, zorder=3)
    axB.barh(yy, han_w, height=h, color=C_MUTED, alpha=0.55, lw=0, zorder=3)
    axB.barh(yy - h, boot_w, height=h, color=C_INK, alpha=0.72, lw=0, zorder=3)
    axB.set_yticks(yy)
    axB.set_yticklabels(a["display"], fontsize=6.6)
    axB.yaxis.tick_right()
    axB.tick_params(axis="y", length=0, pad=2)
    axB.set_xlabel("width of the reported ROC-AUC interval")
    axB.grid(axis="y", visible=False)
    axB.set_title("(b) the old band was not a confidence interval", loc="left",
                  color=C_INK)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=C_MUTED,
                      hatch="////", lw=0.8),
        plt.Rectangle((0, 0), 1, 1, facecolor=C_MUTED, alpha=0.55, lw=0),
        plt.Rectangle((0, 0), 1, 1, facecolor=C_INK, alpha=0.72, lw=0),
    ]
    axB.legend(handles,
               ["v1 repeated-split band (split variability, NOT a CI)",
                "Hanley 95% CI", "bootstrap 95% CI"],
               loc="lower right", fontsize=6.2, handlelength=1.4,
               handleheight=0.8, borderpad=0.5, labelspacing=0.42)
    axB.set_xlim(0, max(han_w.max(), boot_w.max()) * 1.45)

    titles(fig, "What the statistical protocol correction changed",
           "216 analysis population  ·  identical data, contrasts, seeds, folds and "
           "regularisation grid — only the summary statistic differs", 3.4)
    fig.text(0.0, -0.14,
             "Not shown here (not visualisable, see the correction table): the permutation "
             "null now uses the identical per-fold statistic, and Holm adjustment was added.",
             fontsize=6.4, color=C_MUTED, ha="left", va="top")

    save_all(fig, str(stem))
    return {
        "contrast_key": a["key"].tolist(),
        "display": a["display"].tolist(),
        "family": a["family"].tolist(),
        "roc_v1": b["roc_auc"].tolist(), "roc_v2": a["roc_auc"].tolist(),
        "pr_v1": b["pr_auc"].tolist(), "pr_v2": a["pr_auc"].tolist(),
        "delta_roc": d_roc.tolist(),
        "delta_pr": (a["pr_auc"].values - b["pr_auc"].values).tolist(),
        "split_width": split_w.tolist(),
        "hanley_width": han_w.tolist(),
        "boot_width": boot_w.tolist(),
    }


# ══ sidecars / index / metadata ══════════════════════════════════════════════

def _yaml_list(items, indent="  "):
    out = []
    for it in items:
        s = str(it).replace('"', "'")
        out.append(f'{indent}- "{s}"')
    return "\n".join(out)


def write_sidecar(name: str, spec: dict, artifact_paths: dict[str, Path],
                  sources: list[dict], git: dict, extra_body: str = "",
                  plotdata: Path | None = None) -> None:
    hashes = {ext: C.sha256(p) for ext, p in artifact_paths.items()}
    fm = [
        "---",
        f"artifact: {name}",
        f"kind: {spec['kind']}",
        f"formats: [{', '.join(spec['formats'])}]",
        f"status: {spec['status']}",
        "maturity: provisional",
        f'generated_utc: "{BUILD_UTC}"',
        "generated_by:",
        "  script: scripts/paper_step01_build_f.py",
        f'  script_sha256: "{C.sha256(Path(__file__))}"',
        '  command: "python3 scripts/paper_step01_build_f.py"',
        f'  git_commit: "{git["commit"]}"',
        f"  git_dirty: {str(git['dirty']).lower()}",
        "sources:",
    ]
    for s in sources:
        fm.append(f'  - {{path: {s["path"]}, sha256: "{s["sha256"]}", role: {s["role"]}}}')
    if plotdata is not None:
        fm += ["plotdata:", f"  path: {plotdata.name}",
               f'  sha256: "{C.sha256(plotdata)}"']
    fm.append("artifact_sha256:")
    for ext, h in hashes.items():
        fm.append(f'  {ext}: "{h}"')
    fm += [
        f"population: {spec['population']}",
        f'population_note: "{spec["population_note"]}"',
        f"metric: [{', '.join(repr(m) for m in spec['metric'])}]",
        f'uncertainty: "{spec["uncertainty"]}"',
        f'supports_claim: "{spec["supports_claim"]}"',
        "must_not_claim:",
        _yaml_list(spec["must_not_claim"]),
        "caveats:",
        _yaml_list(spec["caveats"]),
        f'paper_location: "{spec["paper_location"]}"',
        f'caption: "{spec["caption"]}"',
        f"replaceable_by_later_step: {str(spec['replaceable_by_later_step']).lower()}",
        f'replacement_risk: "{spec["replacement_risk"]}"',
        "promotion_conditions:",
        _yaml_list(spec["promotion_conditions"]),
        "supersedes: null",
        "superseded_by: null",
        "---",
        "",
    ]
    body = [
        f"# {name}",
        "",
        f"**Status: `{spec['status']}` — provisional.** Nothing in this package is final; "
        "later pipeline steps may replace or modify it.",
        "",
        "## What this shows",
        "",
        spec["supports_claim"],
        "",
        "## What it must NOT be read as",
        "",
    ]
    body += [f"- Do not claim {m}." for m in spec["must_not_claim"]]
    body += ["", "## Caveats", ""]
    body += [f"- {c}" for c in spec["caveats"]]
    body += [
        "", "## How to regenerate", "",
        "```",
        "python3 scripts/paper_step01_build_f.py",
        "python3 scripts/validate_paper_step01_f.py",
        "```",
        "",
        "## Suggested caption",
        "",
        f"> {spec['caption']}",
        "",
        "## Paper placement",
        "",
        f"{spec['paper_location']}",
        "",
        f"**Replaceable by a later step:** {spec['replaceable_by_later_step']}. "
        f"{spec['replacement_risk']}",
        "",
        "**Promotion conditions:**",
        "",
    ]
    body += [f"- {p}" for p in spec["promotion_conditions"]]
    if extra_body:
        body += ["", extra_body]
    body.append("")
    _write((C.TABLES if spec["kind"] == "table" else C.FIGURES) / f"{name}.md",
           "\n".join(fm) + "\n".join(body))


def main() -> None:
    C.verify_inputs()
    v2, v1 = C.load_v2(), C.load_v1()
    git = C.git_state()
    C.TABLES.mkdir(parents=True, exist_ok=True)
    C.FIGURES.mkdir(parents=True, exist_ok=True)

    src = lambda p, role: {"path": C.rel(p), "sha256": C.sha256(p), "role": role}
    SRC_V2 = [src(C.V2_CSV, "primary_results"), src(C.V2_JSON, "primary_results_protocol"),
              src(C.LABELS, "population_definition")]
    SRC_BOTH = SRC_V2 + [src(C.V1_CSV, "superseded_results_provenance")]

    outputs = []

    # ── tables ───────────────────────────────────────────────────────────────
    specs = {
        "tab_primary_contrasts_216": (build_tab_primary(v2), SRC_V2, None),
        "tab_protocol_correction_v1_v2": (build_tab_correction(v1, v2), SRC_BOTH, None),
        "tab_interval_methods": (build_tab_intervals(v1, v2), SRC_BOTH, None),
    }
    for name, ((df, notes), sources, _) in specs.items():
        spec = C.REGISTRY[name]
        csv_p, tex_p = C.TABLES / f"{name}.csv", C.TABLES / f"{name}.tex"
        df.to_csv(csv_p, index=False)
        _write(tex_p, _tex_table(df, spec["caption"], name.replace("_", "-"), notes))
        write_sidecar(name, spec, {"csv": csv_p, "tex": tex_p}, sources, git,
                      extra_body="## Rendered table\n\n" + _md_table(df) + "\n\n"
                                 "### Table notes\n\n"
                                 + "\n".join(f"{i+1}. {n}" for i, n in enumerate(notes)))
        outputs += [csv_p, tex_p, C.TABLES / f"{name}.md"]

    name = "tab_population_sensitivity_216_vs_258"
    comp_df, res_df, notes = build_tab_sensitivity(v2)
    spec = C.REGISTRY[name]
    csv_p, tex_p = C.TABLES / f"{name}.csv", C.TABLES / f"{name}.tex"
    combined = pd.concat(
        [comp_df.assign(block="population composition"),
         res_df.assign(block="results")], ignore_index=True)
    combined.to_csv(csv_p, index=False)
    _write(tex_p,
           _tex_table(comp_df, "Population composition: 258 is a SUPERSET of 216, not a "
                      "second sample.", name + "-composition", notes[:2])
           + "\n" + _tex_table(res_df, spec["caption"], name, notes[2:]))
    write_sidecar(name, spec, {"csv": csv_p, "tex": tex_p}, SRC_V2, git,
                  extra_body="## Rendered tables\n\n**Population composition (258 ⊃ 216)**\n\n"
                             + _md_table(comp_df) + "\n\n**Results**\n\n" + _md_table(res_df)
                             + "\n\n### Table notes\n\n"
                             + "\n".join(f"{i+1}. {n}" for i, n in enumerate(notes)))
    outputs += [csv_p, tex_p, C.TABLES / f"{name}.md"]

    # ── figures ──────────────────────────────────────────────────────────────
    for name, fn, sources in (
        ("fig_contrast_effects_dual_metric_216",
         lambda stem: fig_dual_metric(v2, stem), SRC_V2),
        ("fig_protocol_correction_v1_v2",
         lambda stem: fig_correction(v1, v2, stem), SRC_BOTH),
    ):
        spec = C.REGISTRY[name]
        stem = C.FIGURES / name
        plotdata = fn(stem)
        pd_path = C.FIGURES / f"{name}.plotdata.json"
        _write(pd_path, json.dumps(plotdata, indent=2, sort_keys=True) + "\n")
        paths = {ext: C.FIGURES / f"{name}.{ext}" for ext in ("pdf", "svg", "png")}
        write_sidecar(name, spec, paths, sources, git, plotdata=pd_path)
        outputs += list(paths.values()) + [pd_path, C.FIGURES / f"{name}.md"]

    # ── index, metadata, sources ─────────────────────────────────────────────
    idx = ["# Artifact index — Step 1: statistics protocol", "",
           "Every artifact below is **provisional**. Producing a result here does not "
           "commit the paper to using it: later steps (text baselines / privileged Δ, "
           "eval-awareness control, blind label audit, layer × onset experiment) may "
           "change the preferred metric, figure, or framing.", "",
           "**Status legend** — `candidate_main`: proposed for main text · "
           "`candidate_appendix`: proposed for appendix · `sensitivity`: robustness only, "
           "never a primary estimate · `exploratory`: not claim-bearing · "
           "`deprecated`: superseded, retained for provenance.", "",
           "| artifact | kind | status | population | intended claim | paper placement | "
           "replaceable by later step | promotion / demotion condition |",
           "|---|---|---|---|---|---|---|---|"]
    for name, spec in C.REGISTRY.items():
        claim = spec["supports_claim"]
        claim = claim[:110].rstrip() + ("…" if len(claim) > 110 else "")
        cond = spec["promotion_conditions"][0]
        cond = cond[:100].rstrip() + ("…" if len(cond) > 100 else "")
        idx.append(f"| `{name}` | {spec['kind']} | `{spec['status']}` | "
                   f"{spec['population']} | {claim} | {spec['paper_location']} | "
                   f"{'yes' if spec['replaceable_by_later_step'] else 'no'} | {cond} |")
    idx += ["", "## Reserved", "",
            "**Main figure slot: reserved.** Step 1 deliberately produces no "
            "`candidate_main` figure. Six point estimates from one population are a table, "
            "not a figure, and the paper's headline quantity is expected to become the "
            "Step-2 privileged Δ (activation − matched text baseline). "
            "`fig_contrast_effects_dual_metric_216` is the fallback main candidate if that "
            "Δ turns out null or unusable.", ""]
    _write(C.STEP_DIR / "ARTIFACT_INDEX.md", "\n".join(idx))
    outputs.append(C.STEP_DIR / "ARTIFACT_INDEX.md")

    proto = C.v2_protocol()
    meta = {
        "step": "01_statistics_protocol",
        "title": "Corrected probe statistics for the canonical tier-3 contrasts",
        "maturity": "provisional",
        "generated_utc": BUILD_UTC,
        "git": git,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__, "pandas": pd.__version__,
            "matplotlib": __import__("matplotlib").__version__,
            "sklearn": __import__("sklearn").__version__,
            "platform": sys.platform,
        },
        "inputs": [src(p, role) for p, role in (
            (C.V2_CSV, "primary_results"), (C.V2_JSON, "primary_results_protocol"),
            (C.V1_CSV, "superseded_results_provenance"),
            (C.V1_JSON, "superseded_results_provenance"),
            (C.LABELS, "population_definition"), (C.MANIFEST, "label_manifest"),
            (C.V2_SCRIPT, "analysis_script"), (C.V1_SCRIPT, "superseded_analysis_script"),
            (C.AUDIT_DOC, "motivating_audit"))],
        "outputs": [{"path": C.rel(p), "sha256": C.sha256(p)} for p in sorted(set(outputs))],
        "protocol": {
            "statistic": "per-fold metric computation, averaged across held-out folds",
            "n_repeats": proto["n_repeats"], "n_perm": proto["n_perm"],
            "n_boot": proto["n_boot"], "base_seed": proto["base_seed"],
            "C_grid": proto["C_grid"], "min_class": proto["min_class"],
            "null": ("label permutation through the identical per-fold statistic, "
                     "fixed modal C, one CV repeat (conservative: the null carries extra "
                     "split variance relative to the 20-repeat observed statistic)"),
            "intervals": ("Hanley & McNeil analytic 95% and stratified "
                          "prediction-resampling bootstrap 95%; the repeated-split "
                          "percentile band is retained as split variability only"),
            "multiplicity": f"Holm across the {C.N_CONTRASTS_ADJUSTED} scored contrasts "
                            "within each population",
        },
        "verdict_rule": {
            "primary_metric": "pr_auc",
            "primary_metric_provenance": (
                "Pre-specified as primary in the analysis script docstring "
                "(scripts/probe_contrasts_canonical_f.py), committed before the corrected "
                "run. NOT registered in docs/PREREGISTRATION.md — do not describe it as "
                "pre-registered."),
            "rule": C.VERDICT_RULE,
            "alpha": C.ALPHA,
            "vocabulary": list(C.VERDICTS),
            "alias_map": C.VERDICT_ALIAS,
            "p_raw_floor": C.P_RAW_FLOOR,
            "p_holm_floor": C.P_HOLM_FLOOR,
        },
        "populations": {
            "primary": C.PRIMARY_POP, "sensitivity": C.SENSITIVITY_POP,
            "note": "258 is a strict superset of 216 and is never a primary estimate.",
            "composition": C.population_composition(),
        },
        "status_vocabulary": C.STATUS_VOCAB,
        "deprecated_alongside": {
            "path": "results/figures_deprecated_oldlabels/",
            "note": ("The four session-13 figures were renamed (not deleted) because every "
                     "behaviour-conditioned number in them uses the superseded judge "
                     "labels. See that directory's README.md."),
        },
    }
    _write(C.STEP_DIR / "run_metadata.json", json.dumps(meta, indent=2) + "\n")
    _write(C.STEP_DIR / "SOURCES.sha256",
           "".join(f"{s['sha256']}  {s['path']}\n" for s in meta["inputs"]))

    print(f"built {len(meta['outputs'])} artifacts under {C.rel(C.STEP_DIR)}")
    for o in meta["outputs"]:
        print(f"  {o['path']}")


if __name__ == "__main__":
    main()
