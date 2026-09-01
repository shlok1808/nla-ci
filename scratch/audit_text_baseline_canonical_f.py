#!/usr/bin/env python3
"""Scenario-text TF-IDF baseline for the canonical probe contrasts (216 population).

The activation probe reads the final prompt token; if scenario text alone
predicts the same labels equally well, the activation result is not privileged.
Seeded, local, read-only.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

canon = pd.read_csv("results/behavior_labels_tier3_canonical_f.csv")
bench = pd.read_csv("results/benchmark_results_bf16.csv")[["scenario_id", "scenario"]]
df = canon.merge(bench, on="scenario_id", how="left")
assert df.scenario.notna().all()
sub = df[df.population == "analysis"].reset_index(drop=True)
LIMITING = {"soft_deflection", "explicit_refusal", "mixed_disclose_then_limit"}
lim = sub.response_strategy.isin(LIMITING)

contrasts = {
    "substantive_leak": (pd.Series(True, index=sub.index), sub.substantive_leak.astype(bool)),
    "broad_breach": (pd.Series(True, index=sub.index), sub.broad_breach.astype(bool)),
    "leak_vs_appropriate": (sub.label_substantive.isin(["leaked", "appropriate"]),
                            sub.label_substantive.eq("leaked")),
    "degree_boundary": (sub.label_substantive.isin(["broad_only", "leaked"]),
                        sub.label_substantive.eq("leaked")),
    "limiting_vs_direct": (pd.Series(True, index=sub.index), lim),
    "limiting_among_disclosers": (sub.broad_breach.astype(bool), lim),
}


def text_cv_auc(texts, y, seed):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    s = np.zeros(len(y))
    for tr, te in cv.split(texts, y):
        p = Pipeline([("tf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                             sublinear_tf=True)),
                      ("lr", LogisticRegression(C=1.0, class_weight="balanced",
                                                max_iter=5000))])
        p.fit(texts.iloc[tr], y[tr])
        s[te] = p.predict_proba(texts.iloc[te])[:, 1]
    return roc_auc_score(y, s)


print(f"{'contrast':28s} {'n':>4s} {'pos':>4s}  scenario-text AUC (mean of 5 seeds)")
for name, (mask, pos) in contrasts.items():
    m = mask.to_numpy()
    y = pos.to_numpy()[m].astype(int)
    texts = sub.scenario[m].reset_index(drop=True)
    if min(y.sum(), len(y) - y.sum()) < 15:
        print(f"{name:28s} skipped")
        continue
    aucs = [text_cv_auc(texts, y, s) for s in range(5)]
    print(f"{name:28s} {len(y):4d} {y.sum():4d}  {np.mean(aucs):.3f} "
          f"(range {min(aucs):.3f}-{max(aucs):.3f})")
