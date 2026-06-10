"""
verbalization_survival_f.py — "What survives verbalization?" (conceptual figure).

For a set of scenario attributes, compares how decodable each attribute is from
(a) the raw input text (TF-IDF baseline — following the "privileged information"
methodology of arXiv:2509.13316), (b) raw layer 20 activations (linear probe),
and (c) the NLA's natural-language descriptions of those activations (TF-IDF).

Privileged information = activation AUC − input-text AUC: what the model's
internal state knows beyond what the scenario text predicts. Survival = whether
the description AUC tracks the activation AUC or collapses to the input floor.

If the NLA were a faithful channel, description points would lie on the
diagonal vs activations. The hypothesis (from L6/L8): only dominant-variance
attributes survive; weak behavioral signals (leak) collapse to the input floor —
verbalization destroys precisely the privileged part.

Runs entirely locally — needs only results/*.csv and results/*.npz, no GPU.

Usage:
    python scripts/verbalization_survival_f.py

Outputs:
    results/verbalization_survival_f.csv   (per-attribute AUCs + bootstrap CIs)
    results/verbalization_survival_f.png   (scatter: activation AUC vs description AUC)
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────

ACTIVATIONS   = Path('results/activations_layer20.npz')
BENCHMARK_CSV = Path('results/benchmark_results_bf16.csv')
NLA_CSV       = Path('results/nla_descriptions.csv')
OUT_CSV       = Path('results/verbalization_survival_f.csv')
OUT_PNG       = Path('results/verbalization_survival_f.png')

PROBE_C       = 1e-3      # best regularization from probe_diagnostics.py
N_BOOTSTRAP   = 1000
SEED          = 0

# Spontaneous CI-language audit: terms a privacy-aware description would use
CI_TERMS = re.compile(
    r'\b(privacy|private|confidential|secret|disclos\w*|leak\w*|'
    r'betray\w*|reveal\w* (?:a |the )?secret|sensitive information|'
    r'should not (?:be )?(?:shared|told|revealed)|breach)\b', re.IGNORECASE)


# ── Probes ────────────────────────────────────────────────────────────────────

def probe_auc(X, y, kind, seed=SEED):
    """5-fold CV AUC. kind='acts' → standardized logistic; kind='text' → TF-IDF logistic."""
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 5:
        return np.nan
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    if kind == 'acts':
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=PROBE_C, max_iter=5000))
        scores = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
    else:
        # vectorize inside each fold via pipeline to avoid leakage
        clf = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
                            LogisticRegression(max_iter=2000))
        scores = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
    return roc_auc_score(y, scores), scores


def bootstrap_ci(y, scores, n=N_BOOTSTRAP, seed=SEED):
    """Percentile bootstrap CI on AUC over resampled (y, score) pairs."""
    rng = np.random.default_rng(seed)
    aucs = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        aucs.append(roc_auc_score(y[b], scores[b]))
    return np.percentile(aucs, [2.5, 97.5])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    data = np.load(ACTIVATIONS, allow_pickle=True)
    acts, labels, tiers, ids = (data['activations'], data['labels'],
                                data['tiers'], data['scenario_ids'])
    order = np.argsort(ids)
    acts, labels, tiers, ids = acts[order], labels[order], tiers[order], ids[order]

    bench = pd.read_csv(BENCHMARK_CSV).sort_values('scenario_id').reset_index(drop=True)
    nla   = pd.read_csv(NLA_CSV).sort_values('scenario_id').reset_index(drop=True)
    assert (bench['scenario_id'].values == ids).all(), 'benchmark/activations ID mismatch'
    assert (nla['scenario_id'].values == ids).all(), 'nla/activations ID mismatch'
    desc = nla['description'].fillna('').values

    t3 = tiers == 'tier_3'
    t12 = np.isin(tiers, ['tier_1', 'tier_2a', 'tier_2b'])

    # Topic cluster within tier 3: TF-IDF k-means (k=2) on scenario text.
    # Honest label: "dominant topic split", not a semantic category.
    scen_t3 = bench.loc[t3, 'scenario'].fillna('').values
    topic = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit_predict(
        TfidfVectorizer(max_features=2000, stop_words='english').fit_transform(scen_t3))

    resp_len = bench['response'].fillna('').str.split().str.len().values
    scen_len = bench['scenario'].fillna('').str.split().str.len().values

    nl_t3 = t3 & np.isin(labels, ['appropriate', 'refused'])   # not-leaked subset
    scen_text = bench['scenario'].fillna('').values

    # (name, side, mask, y) — y must be binary int over mask
    attributes = [
        ('tier (t3 vs t1/2)',        'input',    t3 | t12,  t3[t3 | t12].astype(int)),
        ('topic cluster (t3)',       'input',    t3,        topic),
        ('scenario length (t3)',     'input',    t3,        (scen_len[t3] > np.median(scen_len[t3])).astype(int)),
        ('leaked vs not (t3)',       'behavior', t3,        (labels[t3] == 'leaked').astype(int)),
        ('leaked vs approp. (t3)',   'behavior', t3 & (labels != 'refused'),
                                                 (labels[t3 & (labels != 'refused')] == 'leaked').astype(int)),
        ('refused vs appropriate',   'behavior', nl_t3,     (labels[nl_t3] == 'refused').astype(int)),
        ('response length (t3)',     'behavior', t3,        (resp_len[t3] > np.median(resp_len[t3])).astype(int)),
    ]

    rows = []
    for name, side, mask, y in attributes:
        y = np.asarray(y)
        a_auc, a_scores = probe_auc(acts[mask], y, 'acts')
        d_auc, d_scores = probe_auc(desc[mask], y, 'text')
        i_auc, i_scores = probe_auc(scen_text[mask], y, 'text')
        a_lo, a_hi = bootstrap_ci(y, a_scores)
        d_lo, d_hi = bootstrap_ci(y, d_scores)
        i_lo, i_hi = bootstrap_ci(y, i_scores)
        rows.append(dict(attribute=name, side=side, n=int(mask.sum()),
                         pos=int(y.sum()),
                         input_auc=i_auc, input_lo=i_lo, input_hi=i_hi,
                         act_auc=a_auc, act_lo=a_lo, act_hi=a_hi,
                         desc_auc=d_auc, desc_lo=d_lo, desc_hi=d_hi,
                         privileged=a_auc - i_auc,
                         survival=(d_auc - 0.5) / max(a_auc - 0.5, 1e-9)))
        print(f'{name:<26} n={mask.sum():<4} input={i_auc:.3f}  acts={a_auc:.3f} '
              f'[{a_lo:.3f},{a_hi:.3f}]  desc={d_auc:.3f} [{d_lo:.3f},{d_hi:.3f}]  '
              f'privileged={a_auc - i_auc:+.3f}')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved {OUT_CSV}')

    # Spontaneous CI-language audit across all 496 descriptions
    hits = pd.Series(desc).str.contains(CI_TERMS).sum()
    print(f'\nSpontaneous CI-language audit: {hits}/496 descriptions contain '
          f'privacy/secrecy/disclosure terms')

    # ── Figure ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6.2))
    colors = {'input': '#1f77b4', 'behavior': '#d62728'}
    for _, r in df.iterrows():
        c = colors[r['side']]
        ax.errorbar(r.act_auc, r.desc_auc,
                    xerr=[[r.act_auc - r.act_lo], [r.act_hi - r.act_auc]],
                    yerr=[[r.desc_auc - r.desc_lo], [r.desc_hi - r.desc_auc]],
                    fmt='o', color=c, markersize=8, capsize=3, lw=1)
        # input-text floor for the same attribute, marked on the y-axis level
        ax.plot(r.act_auc, r.input_auc, marker='_', color=c, markersize=14,
                alpha=0.6, ls='')
        ax.annotate(r.attribute, (r.act_auc, r.desc_auc),
                    textcoords='offset points', xytext=(8, -4), fontsize=8)
    ax.plot([0.45, 1.02], [0.45, 1.02], 'k--', lw=1, alpha=0.5,
            label='faithful channel (y = x)')
    ax.axhline(0.5, color='gray', lw=0.5); ax.axvline(0.5, color='gray', lw=0.5)
    ax.set_xlim(0.45, 1.02); ax.set_ylim(0.45, 1.02)
    ax.set_xlabel('decodable from activations (linear probe AUC)')
    ax.set_ylabel('recoverable from NLA descriptions (TF-IDF AUC)')
    ax.set_title('What survives verbalization?\n'
                 'Qwen2.5-7B layer 20, last prompt token, ConfAIde')
    handles = [plt.Line2D([0], [0], marker='o', color=c, ls='', label=s)
               for s, c in colors.items()]
    handles.append(plt.Line2D([0], [0], color='k', ls='--', label='faithful channel'))
    handles.append(plt.Line2D([0], [0], marker='_', color='gray', ls='',
                              markersize=12, label='input-text floor'))
    ax.legend(handles=handles, loc='lower right', fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=200)
    print(f'Saved {OUT_PNG}')


if __name__ == '__main__':
    main()
