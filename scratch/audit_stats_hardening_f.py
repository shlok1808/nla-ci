#!/usr/bin/env python3
"""Read-only statistical audit of probe_contrasts_canonical_f results.

(a) calibration-vs-analysis composition (is the 258 gain enrichment?)
(b) Hanley-McNeil SE vs reported repeated-CV percentile band half-widths
(c) Holm correction across the 6 primary-population contrasts, PR and ROC
(d) grid-floor sensitivity: AUC below the C floor + diff-of-means baseline

Seeded, local CPU only. Writes nothing outside stdout.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

canon = pd.read_csv("results/behavior_labels_tier3_canonical_f.csv")
probe = pd.read_csv("results/probe_contrasts_canonical_f.csv")
npz = np.load("results/activations_layer20.npz", allow_pickle=True)
ids = np.asarray(npz["scenario_ids"], dtype=int)
idx = {s: i for i, s in enumerate(ids)}
acts = np.asarray(npz["activations"], dtype=np.float64)
LIMITING = {"soft_deflection", "explicit_refusal", "mixed_disclose_then_limit"}

# ---------- (a) composition ----------
print("=== (a) calibration vs analysis composition ===")
for popname, sub in (("analysis(216)", canon[canon.population == "analysis"]),
                     ("calibration(42)", canon[canon.population == "calibration"])):
    lim = sub.response_strategy.isin(LIMITING)
    print(f"{popname}: limiting {lim.sum()}/{len(sub)} ({lim.mean():.1%}); "
          f"label_substantive {sub.label_substantive.value_counts().to_dict()}")

# ---------- (b) Hanley SE vs reported band ----------
print("\n=== (b) Hanley SE vs reported repeated-CV band ===")


def hanley_se(auc, n_pos, n_neg):
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    return np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2)
                    + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg))


for _, r in probe[probe.status == "scored"].iterrows():
    auc, npos, n = r.roc_auc, int(r.n_pos), int(r.n)
    se = hanley_se(auc, npos, n - npos)
    band = (r.roc_auc_hi - r.roc_auc_lo) / 2
    print(f"{r.contrast:48s} ROC {auc:.3f}  Hanley 95% ±{1.96*se:.3f}  "
          f"reported band ±{band:.3f}  ratio {1.96*se/band:.1f}x")

# ---------- (c) Holm correction, primary population ----------
print("\n=== (c) Holm-Bonferroni across 6 scored contrasts, analysis_216 ===")
prim = probe[(probe.status == "scored") & probe.contrast.str.endswith("analysis_216")]
for metric in ("p_perm_pr_auc", "p_perm_roc_auc"):
    rows = prim[["contrast", metric]].sort_values(metric).reset_index(drop=True)
    m = len(rows)
    print(f"-- {metric} (m={m}) --")
    surviving = True
    for k, r in rows.iterrows():
        thresh = 0.05 / (m - k)
        ok = surviving and r[metric] <= thresh
        if not ok:
            surviving = False
        print(f"  {r.contrast:48s} p={r[metric]:.4f}  holm-thresh={thresh:.4f}  "
              f"{'SURVIVES' if ok else 'fails'}")

# ---------- (d) grid-floor sensitivity ----------
print("\n=== (d) below-floor C sensitivity + diff-of-means, analysis_216 ===")
sub = canon[canon.population == "analysis"].reset_index(drop=True)
rows = np.array([idx[s] for s in sub.scenario_id.astype(int)])


def get_xy(name):
    lim = sub.response_strategy.isin(LIMITING)
    if name == "leak_vs_appropriate":
        mask = sub.label_substantive.isin(["leaked", "appropriate"])
        y = sub.label_substantive.eq("leaked")
    elif name == "substantive_leak":
        mask = pd.Series(True, index=sub.index)
        y = sub.substantive_leak.astype(bool)
    elif name == "limiting_vs_direct":
        mask = pd.Series(True, index=sub.index)
        y = lim
    m = mask.to_numpy()
    return acts[rows[m]], y.to_numpy()[m].astype(int)


def cv_auc(X, y, C, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    s = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        p = Pipeline([("sc", StandardScaler()),
                      ("lr", LogisticRegression(C=C, class_weight="balanced",
                                                solver="liblinear", max_iter=5000))])
        p.fit(X[tr], y[tr])
        s[te] = p.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, s)


def dom_auc(X, y, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    s = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Z = (X - mu) / sd
        w = Z[tr][y[tr] == 1].mean(0) - Z[tr][y[tr] == 0].mean(0)
        s[te] = Z[te] @ w
    return roc_auc_score(y, s)


for name in ("leak_vs_appropriate", "substantive_leak", "limiting_vs_direct"):
    X, y = get_xy(name)
    aucs = {f"C={c:g}": np.mean([cv_auc(X, y, c, s) for s in range(3)])
            for c in (1e-9, 1e-8, 1e-7, 1e-5, 1e-3)}
    d = np.mean([dom_auc(X, y, s) for s in range(3)])
    line = "  ".join(f"{k}:{v:.3f}" for k, v in aucs.items())
    print(f"{name:24s} {line}  diff-of-means:{d:.3f}")
