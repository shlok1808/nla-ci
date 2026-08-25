"""
stats_hardening_f.py — Inference-grade error bars for every headline number.

Motivated by REPORT.md T1: the triad's per-AUC bootstrap CIs
(`verbalization_survival_f.csv`) overlap heavily, so "verbalization destroys the
leak signal" (desc 0.606 vs input 0.581) looks unsupported when read off the raw
intervals. But those three probes are evaluated on the SAME scenarios, so their
errors are strongly correlated — the CI on the *difference* is much tighter than
the overlap of the two marginal CIs suggests. The correct object is a PAIRED
bootstrap on the delta, which is what this script produces.

Three tables:

  [A] Triad paired deltas. One resample of scenario indices per replicate, all
      three AUCs recomputed on that same resample, deltas taken within replicate.
      Reports acts−input (privileged signal), desc−input (what survives
      verbalization over the input baseline), and acts−desc (channel loss), each
      with a percentile CI and a two-sided bootstrap p-value.

  [B] Hanley–McNeil standard errors for the headline pos-0 AUCs. Analytic, and a
      function of (AUC, n_pos, n_neg) only — this is the number that says whether
      the n=36 `refused` class can carry the 0.88–0.92 claims.

  [C] Judge-noise attenuation. The labels come from a GPT-4o-mini judge with an
      estimated ~10% error rate (see judge_label_spotcheck_f.csv). Independent
      label noise at rate e attenuates a probe's observed AUC toward 0.5:
          observed ≈ (1−e)·true + e·0.5
      so true ≈ (observed − e/2) / (1 − e). Reported as a sensitivity band, NOT
      as a corrected headline — comparisons between probes are unaffected because
      both are attenuated by the same factor.

Local, CPU-only, ~2 min. Reads only committed artifacts; writes nothing but its
own outputs.

Usage:
    python scripts/stats_hardening_f.py

Outputs:
    results/stats_hardening_f.csv        (table A, tidy)
    results/stats_hardening_hanley_f.csv (tables B + C, tidy)
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

# ── Config (kept in lockstep with verbalization_survival_f.py) ────────────────

ACTIVATIONS   = Path('results/activations_layer20.npz')
BENCHMARK_CSV = Path('results/benchmark_results_bf16.csv')
NLA_CSV       = Path('results/nla_descriptions.csv')
OUT_DELTAS    = Path('results/stats_hardening_f.csv')
OUT_HANLEY    = Path('results/stats_hardening_hanley_f.csv')

PROBE_C     = 1e-3
N_BOOTSTRAP = 5000     # deltas need more replicates than marginals
SEED        = 0
JUDGE_ERR   = 0.10     # estimated label error rate (see L2 / judge spot-check)


# ── Probes (identical config to the triad script) ─────────────────────────────

def probe_scores(X, y, kind, seed=SEED):
    """5-fold CV out-of-fold scores. kind='acts' → standardized logistic; 'text' → TF-IDF."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    if kind == 'acts':
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=PROBE_C, max_iter=5000))
    else:
        clf = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
                            LogisticRegression(max_iter=2000))
    return cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]


def paired_delta_ci(y, score_a, score_b, n=N_BOOTSTRAP, seed=SEED):
    """
    Percentile CI + two-sided bootstrap p for AUC(a) − AUC(b), resampling
    scenarios ONCE per replicate so both AUCs move together (paired).
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    deltas = []
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        yb = y[b]
        if len(np.unique(yb)) < 2:
            continue
        deltas.append(roc_auc_score(yb, score_a[b]) - roc_auc_score(yb, score_b[b]))
    deltas = np.asarray(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # two-sided bootstrap p: how often the delta sits on the other side of 0
    frac = (deltas <= 0).mean() if deltas.mean() > 0 else (deltas >= 0).mean()
    p = min(1.0, 2 * max(frac, 1.0 / len(deltas)))
    return lo, hi, p, deltas.mean()


def hanley_se(auc, n_pos, n_neg):
    """Hanley & McNeil (1982) analytic SE of an AUC."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc)
           + (n_pos - 1) * (q1 - auc ** 2)
           + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def deattenuate(auc, e=JUDGE_ERR):
    """Invert observed ≈ (1−e)·true + e·0.5 for the true AUC under label noise e."""
    return (auc - e / 2) / (1 - e)


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

    desc      = nla['description'].fillna('').values
    scen_text = bench['scenario'].fillna('').values
    t3        = tiers == 'tier_3'
    nl_t3     = t3 & np.isin(labels, ['appropriate', 'refused'])

    # ── [A] Triad paired deltas ───────────────────────────────────────────────
    attributes = [
        ('leaked vs not (t3)',     t3,                             (labels[t3] == 'leaked').astype(int)),
        ('leaked vs approp. (t3)', t3 & (labels != 'refused'),
                                   (labels[t3 & (labels != 'refused')] == 'leaked').astype(int)),
        ('refused vs appropriate', nl_t3,                          (labels[nl_t3] == 'refused').astype(int)),
    ]

    print('=' * 78)
    print('[A] TRIAD PAIRED DELTAS — CIs on the difference, not the marginals')
    print('=' * 78)
    print('    (5000 paired bootstrap replicates; p is two-sided bootstrap)\n')

    rows = []
    for name, mask, y in attributes:
        y = np.asarray(y)
        s_act = probe_scores(acts[mask],      y, 'acts')
        s_des = probe_scores(desc[mask],      y, 'text')
        s_inp = probe_scores(scen_text[mask], y, 'text')
        a, d, i = (roc_auc_score(y, s) for s in (s_act, s_des, s_inp))

        print(f'{name}   n={int(mask.sum())} ({int(y.sum())} pos)')
        print(f'    raw:  input={i:.3f}   acts={a:.3f}   desc={d:.3f}')
        for dname, sa, sb, obs in [
            ('acts − input  (privileged signal)', s_act, s_inp, a - i),
            ('desc − input  (survives verbaliz.)', s_des, s_inp, d - i),
            ('acts − desc   (channel loss)',       s_act, s_des, a - d),
        ]:
            lo, hi, p, mean = paired_delta_ci(y, sa, sb)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
            print(f'    {dname:<36} {obs:+.3f}  [{lo:+.3f}, {hi:+.3f}]  p={p:.3f} {sig}')
            rows.append(dict(attribute=name, delta=dname.split('(')[0].strip(),
                             n=int(mask.sum()), n_pos=int(y.sum()),
                             observed=obs, boot_mean=mean, ci_lo=lo, ci_hi=hi,
                             p_two_sided=p, significant=(p < 0.05)))
        print()

    pd.DataFrame(rows).to_csv(OUT_DELTAS, index=False)

    # ── [B]/[C] Hanley SEs + judge-noise sensitivity ──────────────────────────
    print('=' * 78)
    print('[B] HANLEY SEs  +  [C] JUDGE-NOISE ATTENUATION (e = %.0f%%)' % (JUDGE_ERR * 100))
    print('=' * 78)

    headline = [
        ('deflection (refused vs rest), L20 pos0', nl_t3 | t3, None),
    ]
    # Build the pos-0 contrasts directly so SEs attach to in-script numbers.
    contrasts = [
        ('refused vs rest (t3)',      t3,                          (labels[t3] == 'refused').astype(int)),
        ('refused vs appropriate',    nl_t3,                       (labels[nl_t3] == 'refused').astype(int)),
        ('leaked vs refused',         t3 & (labels != 'appropriate'),
                                      (labels[t3 & (labels != 'appropriate')] == 'leaked').astype(int)),
        ('leaked vs not (t3)',        t3,                          (labels[t3] == 'leaked').astype(int)),
        ('leaked vs approp. (t3)',    t3 & (labels != 'refused'),
                                      (labels[t3 & (labels != 'refused')] == 'leaked').astype(int)),
    ]

    hrows = []
    print(f'{"contrast":<28} {"n+":>4} {"n−":>4} {"AUC":>7} {"HanleySE":>9} '
          f'{"95% CI":>18} {"true@e=10%":>11}')
    print('-' * 84)
    for name, mask, y in contrasts:
        y = np.asarray(y)
        s = probe_scores(acts[mask], y, 'acts')
        auc = roc_auc_score(y, s)
        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        se = hanley_se(auc, n_pos, n_neg)
        lo, hi = auc - 1.96 * se, auc + 1.96 * se
        true = deattenuate(auc)
        print(f'{name:<28} {n_pos:>4} {n_neg:>4} {auc:>7.4f} {se:>9.4f} '
              f'  [{lo:.3f}, {hi:.3f}] {true:>11.3f}')
        hrows.append(dict(contrast=name, n_pos=n_pos, n_neg=n_neg, auc=auc,
                          hanley_se=se, ci_lo=lo, ci_hi=hi,
                          judge_err=JUDGE_ERR, auc_deattenuated=true))

    pd.DataFrame(hrows).to_csv(OUT_HANLEY, index=False)

    print()
    print('Reading [C]: de-attenuation is a SENSITIVITY BAND, not a corrected headline.')
    print('  Both probes in any comparison are attenuated by the same (1−e) factor, so')
    print('  every contrast between them is unaffected. Report observed AUCs; cite the')
    print('  de-attenuated value only as "consistent with a true AUC of ~X".')
    print()
    print(f'wrote {OUT_DELTAS}')
    print(f'wrote {OUT_HANLEY}')


if __name__ == '__main__':
    main()
