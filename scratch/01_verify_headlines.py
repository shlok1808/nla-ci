"""Audit scratch 01 — re-derive every headline number from raw results files.

Independent verification (2026-07-02 audit). Reads only:
  results/position_sweep_aucs_f.csv
  results/activations_layer20.npz
  results/benchmark_results_bf16.csv
  results/nla_descriptions.csv

Claims under test:
  C1  deflection probe AUC (refused vs rest) 0.89-0.92, at position 0
  C2  leak probe AUC 0.65-0.77, never > 0.80 anywhere in the 5x65 sweep
  C3  layer trajectory: deflection 0.52 (L10) -> 0.89 (L20) at pos 0
  C4  triad numbers: leak input/acts/desc = 0.58/0.68/0.61, refused 0.62/0.92/0.76
  C5  three-way at pos 0: refused-vs-approp 0.92, leaked-vs-refused 0.876,
      leaked-vs-approp 0.651
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option('display.width', 200)

print('=' * 78)
print('PART A — position_sweep_aucs_f.csv (as-committed numbers)')
print('=' * 78)
sw = pd.read_csv('results/position_sweep_aucs_f.csv')
print(f'rows={len(sw)}, layers={sorted(sw.layer.unique())}, '
      f'pos 0..{sw.pos.max()}, targets={sorted(sw.target.unique())}')
print('n per target:', sw.groupby("target")["n"].unique().to_dict())

# C3: pos-0 depth table
pos0 = sw[sw.pos == 0].pivot(index='target', columns='layer', values='auc')
print('\n[C3] AUC at pos 0 by layer:')
print(pos0.round(4).to_string())

# C2: global maxima per target
print('\n[C2] global max per target across all (layer, pos):')
for tgt, g in sw.groupby('target'):
    top = g.loc[g.auc.idxmax()]
    n80 = (g.auc >= 0.80).sum()
    print(f'  {tgt:<18} max={top.auc:.4f} at L{int(top.layer)}/k{int(top.pos)}; '
          f'cells>=0.80: {n80}/{len(g)}')

# C1: deflection stats
refl = sw[(sw.target == 'refused_vs_rest')]
print(f'\n[C1] refused_vs_rest: L20 pos0 = '
      f'{refl[(refl.layer == 20) & (refl.pos == 0)].auc.iloc[0]:.4f}; '
      f'L20 range over pos: {refl[refl.layer == 20].auc.min():.3f}-'
      f'{refl[refl.layer == 20].auc.max():.3f}')

print()
print('=' * 78)
print('PART B — recompute from activations_layer20.npz (independent code path)')
print('=' * 78)
d = np.load('results/activations_layer20.npz', allow_pickle=True)
acts, labels, tiers, ids = d['activations'], d['labels'], d['tiers'], d['scenario_ids']
order = np.argsort(ids)
acts, labels, tiers, ids = acts[order], labels[order], tiers[order], ids[order]
t3 = tiers == 'tier_3'
X3, y3 = acts[t3], labels[t3]
print(f'tier3 n={t3.sum()}: leaked={np.sum(y3 == "leaked")}, '
      f'appropriate={np.sum(y3 == "appropriate")}, refused={np.sum(y3 == "refused")}')


def cv_auc(X, y, C=1e-3, seed=0, return_scores=False):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    s = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
    return (roc_auc_score(y, s), s) if return_scores else roc_auc_score(y, s)


def hanley_se(auc, n1, n0):
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    return np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc**2)
                    + (n0 - 1) * (q2 - auc**2)) / (n1 * n0))


contrasts = {
    'leaked_vs_not':      (np.ones(len(y3), bool), (y3 == 'leaked').astype(int)),
    'leaked_vs_approp':   (y3 != 'refused', None),
    'refused_vs_rest':    (np.ones(len(y3), bool), (y3 == 'refused').astype(int)),
    'refused_vs_approp':  (y3 != 'leaked', None),
    'leaked_vs_refused':  (y3 != 'appropriate', None),
}
print('\n[C1/C5] pos-0 contrasts, 5-fold CV logistic (C=1e-3), '
      'seed-variance over 10 seeds:')
results_b = {}
for name, (mask, y) in contrasts.items():
    if y is None:
        pos = 'leaked' if name.startswith('leaked') else 'refused'
        y = (y3[mask] == pos).astype(int)
    else:
        y = y[mask]
    aucs = [cv_auc(X3[mask], y, seed=s) for s in range(10)]
    a = float(np.mean(aucs))
    se = hanley_se(a, int(y.sum()), int(len(y) - y.sum()))
    results_b[name] = a
    print(f'  {name:<20} AUC={a:.4f}  seed-range [{min(aucs):.4f},{max(aucs):.4f}]  '
          f'n1={y.sum()}, n0={len(y)-y.sum()}, Hanley SE={se:.3f}')

# permutation test on the leak AUC itself (not just diff norm)
rng = np.random.default_rng(0)
y_leak = (y3 == 'leaked').astype(int)
obs = cv_auc(X3, y_leak)
null = []
for i in range(200):
    null.append(cv_auc(X3, rng.permutation(y_leak), seed=0))
null = np.array(null)
print(f'\n[perm] leaked_vs_not AUC={obs:.4f}; label-shuffled null: '
      f'mean={null.mean():.3f} p95={np.percentile(null, 95):.3f} '
      f'max={null.max():.3f}  p={np.mean(null >= obs):.4f}')

print()
print('=' * 78)
print('PART C — triad re-derivation (input / acts / desc)')
print('=' * 78)
bench = pd.read_csv('results/benchmark_results_bf16.csv').sort_values(
    'scenario_id').reset_index(drop=True)
nla = pd.read_csv('results/nla_descriptions.csv').sort_values(
    'scenario_id').reset_index(drop=True)
assert (bench.scenario_id.values == ids).all()
assert (nla.scenario_id.values == ids).all()
scen = bench.scenario.fillna('').values
desc = nla.description.fillna('').values


def text_auc(texts, y, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    clf = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
                        LogisticRegression(max_iter=2000))
    s = cross_val_predict(clf, texts, y, cv=cv, method='predict_proba')[:, 1]
    return roc_auc_score(y, s)


for name, mask, y in [
        ('leaked vs not (t3)', t3, (labels[t3] == 'leaked').astype(int)),
        ('leaked vs approp (t3)', t3 & (labels != 'refused'), None),
        ('refused vs approp (t3)', t3 & (labels != 'leaked'), None)]:
    if y is None:
        pos = 'leaked' if 'leaked' in name else 'refused'
        y = (labels[mask] == pos).astype(int)
    i_auc = text_auc(scen[mask], y)
    a_auc = cv_auc(acts[mask], y)
    d_auc = text_auc(desc[mask], y)
    print(f'  {name:<24} input={i_auc:.3f}  acts={a_auc:.3f}  desc={d_auc:.3f}  '
          f'privileged={a_auc - i_auc:+.3f}')
