"""
probe_diagnostics.py — Local sanity checks on layer 20 activations (no GPU needed).

Answers four questions the pipeline currently assumes but never tested:
  1. Is leaked vs not-leaked linearly separable from OUR activations? (CV probe AUC)
  2. How big is the class separation relative to within-class spread?
  3. Is the diff-of-means vector statistically distinguishable from label-shuffle noise?
  4. Does projecting individual activations onto the (cross-validated) diff direction
     separate the classes?

Usage:
    python scripts/probe_diagnostics.py
"""

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)

data = np.load('results/activations_layer20.npz', allow_pickle=True)
acts, labels, tiers = data['activations'], data['labels'], data['tiers']

t3 = tiers == 'tier_3'
X = acts[t3]
y_raw = labels[t3]
y = (y_raw == 'leaked').astype(int)  # leaked=1, not_leaked (appropriate+refused)=0
print(f'Tier 3: n={len(y)}, leaked={y.sum()}, not_leaked={(1 - y).sum()}')
print(f'Activation L2 norms: mean={np.linalg.norm(X, axis=1).mean():.2f} '
      f'std={np.linalg.norm(X, axis=1).std():.2f}')

# ── 1. Cross-validated linear probe ──────────────────────────────────────────
print('\n[1] 5-fold CV logistic probe (leaked vs not-leaked, tier 3)')
for C in [0.001, 0.01, 0.1, 1.0]:
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    scores = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
    print(f'  C={C:<6} AUC={roc_auc_score(y, scores):.3f}')

# ── 2. Class-mean geometry ────────────────────────────────────────────────────
print('\n[2] Class-mean geometry')
mu1, mu0 = X[y == 1].mean(0), X[y == 0].mean(0)
diff = mu1 - mu0
cos = (mu1 @ mu0) / (np.linalg.norm(mu1) * np.linalg.norm(mu0))
within = np.linalg.norm(X - np.where(y[:, None] == 1, mu1, mu0), axis=1).mean()
print(f'  cosine(leaked_mean, not_leaked_mean) = {cos:.6f}')
print(f'  ||diff|| = {np.linalg.norm(diff):.3f}  vs  mean within-class distance = {within:.2f}')
print(f'  diff norm as fraction of mean activation norm: '
      f'{np.linalg.norm(diff) / np.linalg.norm(X, axis=1).mean():.3%}')

# ── 3. Permutation test on diff norm ─────────────────────────────────────────
print('\n[3] Permutation test: is ||diff|| larger than label-shuffle noise?')
null = []
for _ in range(500):
    yp = rng.permutation(y)
    null.append(np.linalg.norm(X[yp == 1].mean(0) - X[yp == 0].mean(0)))
null = np.array(null)
obs = np.linalg.norm(diff)
p = (null >= obs).mean()
print(f'  observed ||diff|| = {obs:.3f}')
print(f'  null: mean={null.mean():.3f} std={null.std():.3f} max={null.max():.3f}')
print(f'  p = {p:.3f}  ({"diff IS real signal" if p < 0.05 else "diff is indistinguishable from noise"})')

# ── 4. Held-out projection onto diff direction ───────────────────────────────
print('\n[4] CV projection onto diff-of-means direction (train-fold diff, test-fold projection)')
cv = StratifiedKFold(5, shuffle=True, random_state=0)
proj = np.zeros(len(y))
for tr, te in cv.split(X, y):
    d = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
    proj[te] = X[te] @ (d / np.linalg.norm(d))
print(f'  projection AUC = {roc_auc_score(y, proj):.3f}')

# ── 5. Tier separability control (should be ~1.0 per the descriptions result) ─
print('\n[5] Control: tier_3 vs tier_1/2 separability from raw activations')
mask12 = np.isin(tiers, ['tier_1', 'tier_2a', 'tier_2b'])
Xc = np.vstack([acts[mask12], X])
yc = np.r_[np.zeros(mask12.sum()), np.ones(len(X))]
clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=5000))
scores = cross_val_predict(clf, Xc, yc, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                           method='predict_proba')[:, 1]
print(f'  AUC = {roc_auc_score(yc, scores):.3f}')
