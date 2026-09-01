#!/usr/bin/env python3
"""Multi-contrast linear probes, canonical tier-3 labels — corrected statistics.

Supersedes probe_contrasts_canonical_f.py (v1). Same data, same contrasts, same
seeds, same C grid, same nested-CV protocol. Four statistical corrections; no
API, GPU, label, activation, or benchmark changes. v1 outputs are retained
untouched for provenance.

  1. PER-FOLD AUC AVERAGING. v1 pooled raw predict_proba outputs across outer
     folds into one AUC. Nested selection is unstable at these n (chosen C spans
     1e-7..1.0 within a repeat) and probability scales differ across C by orders
     of magnitude, so pooled ranking is corrupted at fold boundaries and the
     estimate is biased low (~0.04 on leak_vs_appropriate). Known pitfall:
     Forman & Scholz 2010 (SIGKDD Explorations 12(1)); Airola et al. 2011
     (CSDA 55(4)). All metrics — AUCs and threshold metrics — are now computed
     within each held-out fold and averaged.

  2. PERMUTATION NULL USES THE IDENTICAL STATISTIC. A permutation test is valid
     only when the null distribution is built from the same statistic as the
     observed value (Ojala & Garriga 2010, JMLR 11). Nulls now use per-fold
     averaging too. As in v1, nulls run one CV repeat with the modal C while the
     observed statistic averages 20 repeats — the null therefore carries extra
     split variance, which is conservative.

  3. HONEST INTERVALS. v1's `_lo/_hi` were percentiles over 20 repeats on the
     same scenarios — split noise only, not sampling uncertainty (Nadeau &
     Bengio 2003; Bates, Hastie & Tibshirani 2023, JASA). Reported now:
       roc_auc_hanley_lo/hi   analytic 95% CI (Hanley & McNeil 1982)
       *_boot_lo/hi           95% stratified prediction-resampling bootstrap:
                              resample within each held-out fold preserving
                              class counts, recompute the per-fold mean metric
                              (conditional on fitted models; cycles repeats so
                              split variance is folded in)
       *_split_lo/hi          repeat percentiles, kept ONLY as split
                              variability, never to be reported as a CI
  4. HOLM-ADJUSTED p-VALUES across the scored contrasts within each population
     (Holm 1979), for both PR-AUC (primary metric) and ROC-AUC.

`at_grid_edge` is retained for continuity but is benign: below the grid floor
the AUC is flat and equals a standardized difference-of-means probe (audit
2026-09-01, docs/FULL_AUDIT_AND_NEXT_STEPS_f.md §2.4).

Usage:
  python3 scripts/probe_contrasts_canonical_v2_f.py           # full run
  python3 scripts/probe_contrasts_canonical_v2_f.py --quick   # smoke test ->
                                                              # scratch/, small
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score,
    matthews_corrcoef, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NPZ = Path("results/activations_layer20.npz")
CANONICAL = Path("results/behavior_labels_tier3_canonical_f.csv")

QUICK = "--quick" in sys.argv
if QUICK:
    OUT_CSV = Path("scratch/probe_contrasts_canonical_v2_quick.csv")
    OUT_JSON = Path("scratch/probe_contrasts_canonical_v2_quick.json")
    N_REPEATS, N_PERM, N_BOOT = 3, 50, 200
else:
    OUT_CSV = Path("results/probe_contrasts_canonical_v2_f.csv")
    OUT_JSON = Path("results/probe_contrasts_canonical_v2_f.json")
    N_REPEATS, N_PERM, N_BOOT = 20, 500, 1000

C_GRID = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
N_OUTER, N_INNER = 5, 3
MIN_CLASS = 15
BASE_SEED = 20260901
LIMITING = {"soft_deflection", "explicit_refusal", "mixed_disclose_then_limit"}
METRICS = ("pr_auc", "roc_auc", "balanced_acc", "mcc", "sensitivity", "specificity")


def pipe(C):
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=C, class_weight="balanced",
                                                solver="liblinear", max_iter=5000))])


def fold_predictions(X, y, seed, fixed_C=None):
    """One CV repeat. Returns per-fold (y_te, scores, chosen_C)."""
    cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=seed)
    folds = []
    for tr, te in cv.split(X, y):
        if fixed_C is None:
            g = GridSearchCV(pipe(1.0), {"clf__C": C_GRID}, scoring="average_precision",
                             cv=StratifiedKFold(N_INNER, shuffle=True, random_state=seed))
            g.fit(X[tr], y[tr])
            m, C = g.best_estimator_, g.best_params_["clf__C"]
        else:
            m, C = pipe(fixed_C).fit(X[tr], y[tr]), fixed_C
        folds.append((y[te], m.predict_proba(X[te])[:, 1], C))
    return folds


def fold_metrics(y_te, s_te):
    p = (s_te >= .5).astype(int)
    tp = int(((p == 1) & (y_te == 1)).sum()); fn = int(((p == 0) & (y_te == 1)).sum())
    tn = int(((p == 0) & (y_te == 0)).sum()); fp = int(((p == 1) & (y_te == 0)).sum())
    return {"pr_auc": average_precision_score(y_te, s_te),
            "roc_auc": roc_auc_score(y_te, s_te),
            "balanced_acc": balanced_accuracy_score(y_te, p),
            "mcc": matthews_corrcoef(y_te, p),
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan}


def perfold_mean(folds):
    """The statistic: each metric computed within each held-out fold, averaged."""
    per = pd.DataFrame([fold_metrics(y_te, s_te) for y_te, s_te, _ in folds])
    return per.mean().to_dict()


def hanley_ci(auc, n_pos, n_neg):
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2)
                  + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg))
    return auc - 1.96 * se, auc + 1.96 * se, se


def stratified_boot(all_folds, rng):
    """One bootstrap draw: resample within each fold preserving class counts,
    recompute the per-fold mean ROC/PR. Cycles over stored repeats upstream."""
    rocs, prs = [], []
    for y_te, s_te, _ in all_folds:
        pos = np.flatnonzero(y_te == 1)
        neg = np.flatnonzero(y_te == 0)
        take = np.concatenate([rng.choice(pos, len(pos), replace=True),
                               rng.choice(neg, len(neg), replace=True)])
        rocs.append(roc_auc_score(y_te[take], s_te[take]))
        prs.append(average_precision_score(y_te[take], s_te[take]))
    return float(np.mean(rocs)), float(np.mean(prs))


def run(X, y, tag):
    repeats, allC = [], []
    for r in range(N_REPEATS):
        folds = fold_predictions(X, y, BASE_SEED + r)
        repeats.append(folds)
        allC += [C for _, _, C in folds]
    per_repeat = pd.DataFrame([perfold_mean(f) for f in repeats])
    obs = per_repeat.mean().to_dict()
    modal = max(set(allC), key=allC.count)

    # bootstrap CI (cycles repeats so split variance is included)
    rng = np.random.default_rng(BASE_SEED)
    boots = [stratified_boot(repeats[b % N_REPEATS], rng) for b in range(N_BOOT)]
    b_roc, b_pr = np.array([b[0] for b in boots]), np.array([b[1] for b in boots])

    # permutation null, identical per-fold statistic, fixed modal C (as v1)
    nroc, npr = [], []
    for p in range(N_PERM):
        yp = rng.permutation(y)
        m = perfold_mean(fold_predictions(X, yp, BASE_SEED + 10_000 + p, fixed_C=modal))
        nroc.append(m["roc_auc"]); npr.append(m["pr_auc"])
    nroc, npr = np.array(nroc), np.array(npr)

    han_lo, han_hi, han_se = hanley_ci(obs["roc_auc"], int(y.sum()), int(len(y) - y.sum()))
    out = {"contrast": tag, "status": "scored", "n": int(len(y)), "n_pos": int(y.sum()),
           "prevalence": float(y.mean()), "modal_C": modal,
           "at_grid_edge": bool(modal == min(C_GRID)),
           "p_perm_pr_auc": float((1 + (npr >= obs["pr_auc"]).sum()) / (N_PERM + 1)),
           "p_perm_roc_auc": float((1 + (nroc >= obs["roc_auc"]).sum()) / (N_PERM + 1)),
           "null_pr_auc_mean": float(npr.mean()), "null_roc_auc_mean": float(nroc.mean())}
    for m in METRICS:
        out[m] = float(obs[m])
        out[f"{m}_split_lo"] = float(np.percentile(per_repeat[m], 2.5))
        out[f"{m}_split_hi"] = float(np.percentile(per_repeat[m], 97.5))
    out.update({
        "roc_auc_hanley_se": float(han_se),
        "roc_auc_hanley_lo": float(han_lo), "roc_auc_hanley_hi": float(han_hi),
        "roc_auc_boot_lo": float(np.percentile(b_roc, 2.5)),
        "roc_auc_boot_hi": float(np.percentile(b_roc, 97.5)),
        "pr_auc_boot_lo": float(np.percentile(b_pr, 2.5)),
        "pr_auc_boot_hi": float(np.percentile(b_pr, 97.5)),
        "pr_auc_lift_vs_prevalence": float(obs["pr_auc"] - y.mean()),
    })
    return out


def holm(pvals):
    """Holm 1979 step-down adjusted p-values (monotone, capped at 1)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def contrasts(df):
    lim = df.response_strategy.isin(LIMITING)
    disclosed = df.broad_breach.astype(bool)
    return {
        "substantive_leak":          (pd.Series(True, index=df.index), df.substantive_leak.astype(bool)),
        "broad_breach":              (pd.Series(True, index=df.index), df.broad_breach.astype(bool)),
        "leak_vs_appropriate":       (df.label_substantive.isin(["leaked", "appropriate"]),
                                      df.label_substantive.eq("leaked")),
        "degree_boundary_broadonly_vs_leaked": (df.label_substantive.isin(["broad_only", "leaked"]),
                                      df.label_substantive.eq("leaked")),
        "limiting_vs_direct":        (pd.Series(True, index=df.index), lim),
        "limiting_among_disclosers": (disclosed, lim),
        "refused_vs_appropriate":    (df.label_substantive.isin(["refused", "appropriate"]),
                                      df.label_substantive.eq("refused")),
    }


def main():
    npz = np.load(NPZ, allow_pickle=True)
    canon = pd.read_csv(CANONICAL)
    ids = np.asarray(npz["scenario_ids"], dtype=int)
    missing = set(canon.scenario_id.astype(int)) - set(ids.tolist())
    if missing:
        sys.exit(f"SCENARIO-ID MISMATCH: {sorted(missing)}")
    idx = {s: i for i, s in enumerate(ids)}
    acts = np.asarray(npz["activations"], dtype=np.float64)

    results = []
    for pop, sub in (("analysis_216", canon[canon.population == "analysis"]),
                     ("all_258", canon)):
        sub = sub.reset_index(drop=True)
        rows = np.array([idx[s] for s in sub.scenario_id.astype(int)])
        for name, (mask, pos) in contrasts(sub).items():
            m = mask.to_numpy()
            y = pos.to_numpy()[m].astype(int)
            X = acts[rows[m]]
            tag = f"{name}|{pop}"
            nmin = int(min(y.sum(), len(y) - y.sum()))
            if nmin < MIN_CLASS:
                print(f"== {tag}: SKIPPED, minority class n={nmin} < {MIN_CLASS}", flush=True)
                results.append({"contrast": tag, "status": "underpowered_skipped",
                                "n": int(len(y)), "n_pos": int(y.sum()), "minority_n": nmin})
                continue
            print(f"== {tag}  n={len(y)} pos={y.sum()} ({y.mean():.1%})", flush=True)
            r = run(X, y, tag)
            print(f"   ROC {r['roc_auc']:.3f} hanley[{r['roc_auc_hanley_lo']:.3f},{r['roc_auc_hanley_hi']:.3f}] "
                  f"boot[{r['roc_auc_boot_lo']:.3f},{r['roc_auc_boot_hi']:.3f}] p={r['p_perm_roc_auc']:.4f} | "
                  f"PR {r['pr_auc']:.3f} (prev {r['prevalence']:.3f}) "
                  f"boot[{r['pr_auc_boot_lo']:.3f},{r['pr_auc_boot_hi']:.3f}] p={r['p_perm_pr_auc']:.4f} | "
                  f"C*={r['modal_C']:g}{'  <-- GRID EDGE (benign, see docstring)' if r['at_grid_edge'] else ''}",
                  flush=True)
            results.append(r)

    # Holm adjustment within each population, scored contrasts only
    df = pd.DataFrame(results)
    for metric in ("pr_auc", "roc_auc"):
        df[f"p_holm_{metric}"] = np.nan
    for pop in ("analysis_216", "all_258"):
        scored = df.index[(df.status == "scored") & df.contrast.str.endswith(pop)]
        for metric in ("pr_auc", "roc_auc"):
            df.loc[scored, f"p_holm_{metric}"] = holm(df.loc[scored, f"p_perm_{metric}"].to_numpy())

    df.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps({
        "supersedes": "probe_contrasts_canonical_f.{csv,json}",
        "protocol": "per-fold metric averaging; permutation null on the identical "
                    "statistic (fixed modal C, 1 repeat: conservative); Hanley + "
                    "stratified prediction-resampling bootstrap CIs; Holm within "
                    "population; split band retained as split variability only",
        "C_grid": C_GRID, "n_repeats": N_REPEATS, "n_perm": N_PERM,
        "n_boot": N_BOOT, "min_class": MIN_CLASS, "base_seed": BASE_SEED,
        "quick_mode": QUICK,
        "results": df.to_dict(orient="records")}, indent=2) + "\n")

    print(f"\n{'contrast':52s} {'ROC':>6s} {'hanley 95%':>16s} {'PR':>6s} "
          f"{'p_pr':>7s} {'holm_pr':>8s} {'p_roc':>7s} {'holm_roc':>9s}")
    for _, r in df[df.status == "scored"].iterrows():
        print(f"{r.contrast:52s} {r.roc_auc:6.3f} "
              f"[{r.roc_auc_hanley_lo:.3f},{r.roc_auc_hanley_hi:.3f}] {r.pr_auc:6.3f} "
              f"{r.p_perm_pr_auc:7.4f} {r.p_holm_pr_auc:8.4f} "
              f"{r.p_perm_roc_auc:7.4f} {r.p_holm_roc_auc:9.4f}")
    print(f"\nwrote {OUT_CSV} and {OUT_JSON}")


if __name__ == "__main__":
    main()
