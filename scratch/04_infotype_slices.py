"""Audit scratch 04 — per-info-type slices of the leak signal (tier 3).

ConfAIde tier 3 = 9 information types x 30 scenarios, in file order:
block = (scenario_id - 206) // 30.
Checks: leak rate per type; within-type probe AUC (n=30, noisy);
leave-one-type-out probe transfer (the real generalization test);
deflection rate per type.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TYPE_NAMES = ['work-cheat', 'infidelity', 'abortion', 'extremist-past',
              'self-harm', 'mental-health', 'rare-disease', 'appearance',
              'sexual-orient']

d = np.load('results/activations_layer20.npz', allow_pickle=True)
acts, labels, tiers, ids = d['activations'], d['labels'], d['tiers'], d['scenario_ids']
order = np.argsort(ids)
acts, labels, tiers, ids = acts[order], labels[order], tiers[order], ids[order]
t3 = tiers == 'tier_3'
X, ys, ids3 = acts[t3].astype(np.float64), labels[t3], ids[t3]
y = (ys == 'leaked').astype(int)
block = (ids3 - 206) // 30

print('info type        n  leak%  refuse%  within-type AUC  LOTO AUC')
loto_scores = np.zeros(len(y))
for b in range(9):
    m = block == b
    leak_rate = y[m].mean()
    ref_rate = (ys[m] == 'refused').mean()
    # within-type 5-fold CV AUC (n=30 — very noisy, direction only)
    within = np.nan
    if len(np.unique(y[m])) == 2 and min(np.bincount(y[m])) >= 5:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1e-3, max_iter=5000))
        s = cross_val_predict(clf, X[m], y[m],
                              cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              method='predict_proba')[:, 1]
        within = roc_auc_score(y[m], s)
    # leave-one-type-out
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1e-3, max_iter=5000))
    clf.fit(X[~m], y[~m])
    loto_scores[m] = clf.predict_proba(X[m])[:, 1]
    loto = (roc_auc_score(y[m], loto_scores[m])
            if len(np.unique(y[m])) == 2 else np.nan)
    print(f'{TYPE_NAMES[b]:<15} {m.sum():>3}  {leak_rate:.0%}   {ref_rate:>4.0%}'
          f'     {within:.3f}          {loto:.3f}')

print(f'\npooled LOTO AUC = {roc_auc_score(y, loto_scores):.3f} '
      f'(vs within-CV 0.674 — transfer across info types?)')

# same for deflection
y_r = (ys == 'refused').astype(int)
loto_r = np.zeros(len(y_r))
for b in range(9):
    m = block == b
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1e-3, max_iter=5000))
    clf.fit(X[~m], y_r[~m])
    loto_r[m] = clf.predict_proba(X[m])[:, 1]
print(f'deflection pooled LOTO AUC = {roc_auc_score(y_r, loto_r):.3f} '
      f'(vs within-CV 0.883)')
