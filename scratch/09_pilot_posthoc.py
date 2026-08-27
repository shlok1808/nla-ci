#!/usr/bin/env python3
"""Post-hoc analysis of the NLA transmission pilot.

Reuses the pilot's OWN metric helpers (imported, not reimplemented) so every
number here is directly comparable to the verdict table.

The central question this answers, which the pilot table cannot:
  the descriptions differ -- but is the difference INFORMATIVE or is it NOISE?

Test: compare within-pair similarity (secret_i vs public_i) against cross-pair
similarity (secret_i vs public_j). If a pair's two descriptions are no more
alike than two unrelated scenarios' descriptions, the AV output carries no
scenario-specific information and the differences are chaotic sensitivity,
not transmission. If within >> cross, descriptions track the scenario and the
within-pair difference sits on top of real signal.
"""
import sys, re, json, itertools
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path.home() / 'nla-ci'
sys.path.insert(0, str(REPO / 'scripts'))
import nla_transmission_f as T   # argparse is under __main__, safe to import

PILOT_CSV = REPO / 'results/nla_transmission_pilot_desc_f.csv'
METRICS   = REPO / 'results/nla_transmission_pilot_metrics_f.csv'
MANIFEST  = REPO / 'results/nla_transmission_manifest_f.csv'
PAIRS     = REPO / 'data/minimal_pairs_f.csv'

try:
    from scipy import stats as st
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def hr(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


def main():
    for p in (PILOT_CSV, METRICS, MANIFEST):
        if not p.exists():
            print(f'MISSING: {p}')
            return 1

    df = pd.read_csv(PILOT_CSV)
    M = pd.read_csv(METRICS)
    man = pd.read_csv(MANIFEST)
    pairs = pd.read_csv(PAIRS).set_index('scenario_id')

    base = df[(df.repeat == 0)].copy()
    base = base[[T.text_valid(d, e) for d, e in zip(base.description, base.error)]]
    piv = base.pivot_table(index='scenario_id', columns='variant',
                           values='description', aggfunc='first').dropna()
    ids = sorted(piv.index.astype(int))
    print(f'loaded {len(ids)} complete pairs')

    S = {i: piv.loc[i, 'secret'] for i in ids}
    P = {i: piv.loc[i, 'public'] for i in ids}
    angle = dict(zip(man.scenario_id.astype(int), man.angle_deg))

    # ---------------------------------------------------------------- A
    hr('A. WITHIN-PAIR vs CROSS-PAIR  --  is the difference informative or noise?')
    print('within  = seqsim(secret_i, public_i)      same story, privacy flipped')
    print('cross   = seqsim(secret_i, public_j) i!=j  unrelated stories')
    print('If within ~= cross, the descriptions carry no scenario information.\n')

    for w in ['full', 'P1', 'P2', 'P3']:
        s = {i: T.slice_text(S[i], w) for i in ids}
        p = {i: T.slice_text(P[i], w) for i in ids}
        within = np.array([T.seq_similarity(s[i], p[i]) for i in ids])
        cross_rows = []
        for i in ids:
            vals = [T.seq_similarity(s[i], p[j]) for j in ids if j != i]
            cross_rows.append(np.mean(vals))
        cross = np.array(cross_rows)
        diff = within - cross
        line = (f'{w:>5} | within {within.mean():.3f} +-{within.std():.3f} | '
                f'cross {cross.mean():.3f} +-{cross.std():.3f} | '
                f'delta {diff.mean():+.3f}')
        if HAVE_SCIPY:
            wstat, pval = st.wilcoxon(within, cross)
            n_higher = int((diff > 0).sum())
            line += f' | wilcoxon p={pval:.2e} | within>cross in {n_higher}/{len(ids)}'
        print(line)

    # ---------------------------------------------------------------- B
    hr('B. DOES THE DIFFERENCE TRACK THE ANGLE?')
    print('The AV is direction-only. If the text difference is driven by the')
    print('activation geometry, similarity should FALL as angle grows.')
    print('If uncorrelated, the difference is more likely input-independent noise.\n')
    Mi = M.set_index('scenario_id')
    for w in ['full', 'P1', 'P2', 'P3']:
        col = f'{w}_seqsim'
        if col not in Mi.columns:
            continue
        a = np.array([angle[i] for i in ids])
        v = np.array([Mi.loc[i, col] for i in ids])
        if HAVE_SCIPY:
            rho, pv = st.spearmanr(a, v)
            print(f'{w:>5} | spearman(angle, seqsim) rho={rho:+.3f}  p={pv:.4f}')
        else:
            print(f'{w:>5} | corr={np.corrcoef(a, v)[0,1]:+.3f}')

    # ---------------------------------------------------------------- C
    hr('C. IS THE DIFFERENCE ABOUT PRIVACY?  (lexicon asymmetry)')
    LEX = ['private', 'privacy', 'confidential', 'confidentiality', 'secret',
           'secrecy', 'sensitive', 'disclose', 'disclosure', 'reveal',
           'revealing', 'personal', 'discreet', 'trust', 'trusted', 'share',
           'sharing', 'told', 'hidden', 'conceal', 'protect', 'permission',
           'consent', 'public', 'publicly', 'known', 'openly', 'common']
    rows = []
    for i in ids:
        ts, tp = T.toks(S[i]), T.toks(P[i])
        for wd in LEX:
            cs, cp = ts.count(wd), tp.count(wd)
            if cs or cp:
                rows.append(dict(scenario_id=i, word=wd, secret=cs, public=cp))
    L = pd.DataFrame(rows)
    if len(L):
        agg = (L.groupby('word')[['secret', 'public']].sum()
                 .assign(delta=lambda d: d.secret - d.public)
                 .sort_values('delta'))
        agg['n_pairs'] = L.groupby('word').size()
        print(agg.to_string())
        print(f'\nTOTAL  secret={agg.secret.sum()}  public={agg.public.sum()}  '
              f'delta={agg.secret.sum() - agg.public.sum():+d}')
    else:
        print('no privacy-lexicon words appear in ANY description.')
        print('-> the AV is not describing privacy in explicit vocabulary at all.')

    # ---------------------------------------------------------------- D
    hr('D. DATA-DRIVEN: which words are CONSISTENTLY on one side?')
    print('For each word: in how many of the 40 pairs does it appear in the')
    print('secret description but not the public one (or vice versa)?')
    print('A word with a strong consistent lean is candidate transmitted signal.\n')
    only_s, only_p = {}, {}
    for i in ids:
        a, b = set(T.toks(S[i])), set(T.toks(P[i]))
        for wd in a - b:
            only_s[wd] = only_s.get(wd, 0) + 1
        for wd in b - a:
            only_p[wd] = only_p.get(wd, 0) + 1
    allw = set(only_s) | set(only_p)
    rec = []
    for wd in allw:
        cs, cp = only_s.get(wd, 0), only_p.get(wd, 0)
        n = cs + cp
        if n < 5:
            continue
        d = dict(word=wd, secret_only=cs, public_only=cp, n=n, lean=cs - cp)
        if HAVE_SCIPY:
            d['binom_p'] = st.binomtest(max(cs, cp), n, 0.5).pvalue
        rec.append(d)
    if rec:
        R = pd.DataFrame(rec).sort_values('n', ascending=False)
        print(R.head(30).to_string(index=False))
        if HAVE_SCIPY and 'binom_p' in R.columns:
            sig = R[R.binom_p < 0.05].sort_values('binom_p')
            print(f'\nwords with a significant lean (uncorrected p<0.05): {len(sig)}')
            if len(sig):
                print(sig.head(20).to_string(index=False))
    else:
        print('no word appears in >=5 pairs asymmetrically.')

    # ---------------------------------------------------------------- E
    hr('E. EDIT-VOCAB DISTRIBUTION  (median was 0% -- is it 0 everywhere?)')
    for w in ['full', 'P1', 'P2', 'P3']:
        col = f'{w}_edit_vocab_frac'
        if col in M.columns:
            v = M[col].values
            print(f'{w:>5} | mean {v.mean():.4f} | median {np.median(v):.4f} | '
                  f'max {v.max():.4f} | nonzero in {int((v > 0).sum())}/{len(v)} pairs')

    # ---------------------------------------------------------------- F
    hr('F. STRUCTURE')
    if 'both_3para' in M.columns:
        print(f'pairs where BOTH descriptions parsed into 3 paragraphs: '
              f'{int(M.both_3para.sum())}/{len(M)}')
    npara_s = [len(T.split_paragraphs(S[i])) for i in ids]
    npara_p = [len(T.split_paragraphs(P[i])) for i in ids]
    print(f'paragraph counts secret: {pd.Series(npara_s).value_counts().to_dict()}')
    print(f'paragraph counts public: {pd.Series(npara_p).value_counts().to_dict()}')
    ls = [len(T.toks(S[i])) for i in ids]
    lp = [len(T.toks(P[i])) for i in ids]
    print(f'length (words) secret: median {np.median(ls):.0f}  public: median {np.median(lp):.0f}')

    # ---------------------------------------------------------------- G
    hr('G. QUALITATIVE -- closest-angle pairs (hardest cases)')
    order = sorted(ids, key=lambda i: angle[i])
    for i in order[:3]:
        print(f'\n----- scenario {i}  angle={angle[i]:.2f} deg -----')
        print(f'[what changed in the story] {str(pairs.loc[i, "what_changed"])[:200]}')
        for lbl, txt in (('SECRET', S[i]), ('PUBLIC', P[i])):
            ps = T.split_paragraphs(txt)
            p3 = ps[2] if len(ps) > 2 else '(no P3)'
            print(f'  {lbl} P3: {p3[:400]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
