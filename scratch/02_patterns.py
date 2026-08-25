"""Audit scratch 02 — overlooked-pattern hunt on locally available data.

  A. Nonlinear probes on the leak contrast (MLP / RBF-SVM / HistGB / kNN):
     "weak linear AUC — nonlinear encoding or genuinely absent?"
  B. PCA rank analysis: is the leak/deflection signal low-rank or distributed?
  C. Deflection-erasure: does the leak probe survive removing the deflection
     direction (and vice versa)? Are the two signals the same subspace?
  D. Confound checks: probe scores vs scenario/response length.
  E. Topic generalization: leave-one-cluster-out CV for the leak probe.
  F. Tier-4 transfer: train tier-3, test tier-4 (n=20, exploratory).
  G. Class-balance robustness: subsampled balanced leak AUC.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.stats import spearmanr

d = np.load('results/activations_layer20.npz', allow_pickle=True)
acts, labels, tiers, ids = d['activations'], d['labels'], d['tiers'], d['scenario_ids']
order = np.argsort(ids)
acts, labels, tiers, ids = acts[order], labels[order], tiers[order], ids[order]
bench = pd.read_csv('results/benchmark_results_bf16.csv').sort_values(
    'scenario_id').reset_index(drop=True)
t3 = tiers == 'tier_3'
X3, y3s = acts[t3].astype(np.float64), labels[t3]
y_leak = (y3s == 'leaked').astype(int)
y_refu = (y3s == 'refused').astype(int)
m_la = y3s != 'refused'
y_la = (y3s[m_la] == 'leaked').astype(int)


def cv_scores(clf, X, y, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]


def multi_seed_auc(make_clf, X, y, seeds=range(5)):
    return [roc_auc_score(y, cv_scores(make_clf(), X, y, seed=s)) for s in seeds]


print('=' * 78)
print('A. Nonlinear probes — leak contrast (is the weak linear AUC a linearity issue?)')
print('=' * 78)
probes = {
    'logreg C=1e-3 (baseline)': lambda: make_pipeline(
        StandardScaler(), LogisticRegression(C=1e-3, max_iter=5000)),
    'logreg C=1 (weak reg)': lambda: make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
    'MLP 64 (PCA100)': lambda: make_pipeline(
        StandardScaler(), PCA(n_components=100, random_state=0),
        MLPClassifier(hidden_layer_sizes=(64,), alpha=1.0, max_iter=2000,
                      random_state=0)),
    'MLP 256-64 (PCA100)': lambda: make_pipeline(
        StandardScaler(), PCA(n_components=100, random_state=0),
        MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1.0, max_iter=2000,
                      random_state=0)),
    'RBF SVM (PCA100)': lambda: make_pipeline(
        StandardScaler(), PCA(n_components=100, random_state=0),
        SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=0)),
    'HistGB (PCA100)': lambda: make_pipeline(
        StandardScaler(), PCA(n_components=100, random_state=0),
        HistGradientBoostingClassifier(random_state=0)),
    'kNN k=15 (PCA50)': lambda: make_pipeline(
        StandardScaler(), PCA(n_components=50, random_state=0),
        KNeighborsClassifier(n_neighbors=15)),
}
for tgt_name, X, y in [('leaked_vs_not', X3, y_leak),
                       ('leaked_vs_approp', X3[m_la], y_la),
                       ('refused_vs_rest (control)', X3, y_refu)]:
    print(f'\n  target: {tgt_name}  (n={len(y)}, pos={y.sum()})')
    for name, mk in probes.items():
        aucs = multi_seed_auc(mk, X, y, seeds=range(3))
        print(f'    {name:<28} AUC={np.mean(aucs):.3f}  '
              f'[{min(aucs):.3f},{max(aucs):.3f}]')

print()
print('=' * 78)
print('B. PCA rank analysis — how low-rank is each signal?')
print('=' * 78)
for tgt_name, X, y in [('leaked_vs_not', X3, y_leak),
                       ('refused_vs_rest', X3, y_refu)]:
    row = []
    for k in [1, 2, 5, 10, 20, 50, 100, 200]:
        mk = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=0),
                           LogisticRegression(C=1e-2, max_iter=5000))
        row.append((k, roc_auc_score(y, cv_scores(mk, X, y))))
    print(f'  {tgt_name:<18} ' + '  '.join(f'k={k}:{a:.3f}' for k, a in row))

print()
print('=' * 78)
print('C. Direction-erasure cross-tests (rank-1 LEACE-lite: project out CV-fold')
print('   train-split diff-of-means direction, re-probe)')
print('=' * 78)


def erase_and_probe(X, y_probe, y_erase, seed=0):
    """5-fold CV: within each train fold compute the y_erase diff-of-means
    direction, project it out of both splits, train probe on y_probe."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    scores = np.zeros(len(y_probe))
    for tr, te in cv.split(X, y_probe):
        mu1 = X[tr][y_erase[tr] == 1].mean(0)
        mu0 = X[tr][y_erase[tr] == 0].mean(0)
        v = mu1 - mu0
        v /= np.linalg.norm(v)
        P = np.eye(X.shape[1]) - np.outer(v, v)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1e-3, max_iter=5000))
        clf.fit(X[tr] @ P, y_probe[tr])
        scores[te] = clf.predict_proba(X[te] @ P)[:, 1]
    return roc_auc_score(y_probe, scores)


auc_leak_base = roc_auc_score(y_leak, cv_scores(make_pipeline(
    StandardScaler(), LogisticRegression(C=1e-3, max_iter=5000)), X3, y_leak))
auc_refu_base = roc_auc_score(y_refu, cv_scores(make_pipeline(
    StandardScaler(), LogisticRegression(C=1e-3, max_iter=5000)), X3, y_refu))
print(f'  leak probe: base {auc_leak_base:.3f} -> deflection-dir erased '
      f'{erase_and_probe(X3, y_leak, y_refu):.3f}')
print(f'  deflection probe: base {auc_refu_base:.3f} -> leak-dir erased '
      f'{erase_and_probe(X3, y_refu, y_leak):.3f}')
# also raw geometric overlap
v_leak = X3[y_leak == 1].mean(0) - X3[y_leak == 0].mean(0)
v_refu = X3[y_refu == 1].mean(0) - X3[y_refu == 0].mean(0)
cos = v_leak @ v_refu / np.linalg.norm(v_leak) / np.linalg.norm(v_refu)
print(f'  cosine(v_leak_diffmeans, v_deflect_diffmeans) = {cos:.3f}')

print()
print('=' * 78)
print('D. Length confounds — do probe scores just track length?')
print('=' * 78)
scen_len = bench.loc[t3, 'scenario'].fillna('').str.split().str.len().values
resp_len = bench.loc[t3, 'response'].fillna('').str.split().str.len().values
leak_scores = cv_scores(make_pipeline(
    StandardScaler(), LogisticRegression(C=1e-3, max_iter=5000)), X3, y_leak)
refu_scores = cv_scores(make_pipeline(
    StandardScaler(), LogisticRegression(C=1e-3, max_iter=5000)), X3, y_refu)
for nm, v in [('scenario_len', scen_len), ('response_len', resp_len)]:
    print(f'  spearman(leak_probe_score, {nm}) = '
          f'{spearmanr(leak_scores, v).statistic:+.3f}   '
          f'(label vs {nm}: {spearmanr(y_leak, v).statistic:+.3f})')
print(f'  spearman(deflect_probe_score, response_len) = '
      f'{spearmanr(refu_scores, resp_len).statistic:+.3f}   '
      f'(label vs response_len: {spearmanr(y_refu, resp_len).statistic:+.3f})')
print(f'  spearman(leak_probe_score, deflect_probe_score) = '
      f'{spearmanr(leak_scores, refu_scores).statistic:+.3f}')

print()
print('=' * 78)
print('E. Topic generalization — leave-one-cluster-out leak probe')
print('=' * 78)
scen_t3 = bench.loc[t3, 'scenario'].fillna('').values
for k in [5, 9]:
    tf = TfidfVectorizer(max_features=2000, stop_words='english').fit_transform(scen_t3)
    cl = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(tf)
    scores = np.zeros(len(y_leak))
    ok = np.ones(len(y_leak), bool)
    for c in range(k):
        te = cl == c
        tr = ~te
        if len(np.unique(y_leak[te])) < 2:
            ok[te] = False
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1e-3, max_iter=5000))
        clf.fit(X3[tr], y_leak[tr])
        scores[te] = clf.predict_proba(X3[te])[:, 1]
    print(f'  k={k}: LOCO AUC={roc_auc_score(y_leak[ok], scores[ok]):.3f} '
          f'(on {ok.sum()}/{len(y_leak)} rows; cluster sizes '
          f'{np.bincount(cl).tolist()})')

print()
print('=' * 78)
print('F. Tier-4 transfer (exploratory, n=20; ID 495 is a known judge hallucination)')
print('=' * 78)
t4 = tiers == 'tier_4'
X4, y4s, ids4 = acts[t4].astype(np.float64), labels[t4], ids[t4]
y4 = (y4s == 'leaked').astype(int)
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1e-3, max_iter=5000))
clf.fit(X3, y_leak)
s4 = clf.predict_proba(X4)[:, 1]
print(f'  tier3-trained leak probe on tier4: AUC={roc_auc_score(y4, s4):.3f} '
      f'(n=20, {y4.sum()} leaked)')
y4_fix = y4.copy()
y4_fix[ids4 == 495] = 0
print(f'  with ID 495 corrected to appropriate: '
      f'AUC={roc_auc_score(y4_fix, s4):.3f} ({y4_fix.sum()} leaked)')

print()
print('=' * 78)
print('G. Class-balance robustness — subsample leaked to 83 (balanced vs approp)')
print('=' * 78)
rng = np.random.default_rng(0)
Xla, yla = X3[m_la], y_la
b_aucs = []
for _ in range(20):
    idx1 = rng.choice(np.where(yla == 1)[0], size=83, replace=False)
    idx0 = np.where(yla == 0)[0]
    idx = np.r_[idx1, idx0]
    b_aucs.append(roc_auc_score(yla[idx], cv_scores(make_pipeline(
        StandardScaler(), LogisticRegression(C=1e-3, max_iter=5000)),
        Xla[idx], yla[idx])))
print(f'  balanced leaked_vs_approp AUC: mean={np.mean(b_aucs):.3f} '
      f'sd={np.std(b_aucs):.3f} (full-set value 0.651)')
