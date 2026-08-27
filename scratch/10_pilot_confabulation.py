#!/usr/bin/env python3
"""Follow-up analysis, driven by what section G revealed.

G showed every P3 is a SYNTACTIC forecast: 'Final token "X" ends a quoted
speech tag, immediately requiring ...'. The descriptions are about grammatical
position, not privacy.

That raises a sharp, checkable question: the secret and public prompts for a
given scenario differ only in the story body. If they END identically, the
true final token is THE SAME for both. So if the AV claims a DIFFERENT final
token for secret vs public, at least one claim is factually wrong -- and the
variation we measured is partly the AV being unreliable, not transmitting.
"""
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path.home() / 'nla-ci'
sys.path.insert(0, str(REPO / 'scripts'))
import nla_transmission_f as T

from scipy import stats as st

df = pd.read_csv(REPO / 'results/nla_transmission_pilot_desc_f.csv')
man = pd.read_csv(REPO / 'results/nla_transmission_manifest_f.csv')
pairs = pd.read_csv(REPO / 'data/minimal_pairs_f.csv').set_index('scenario_id')

base = df[df.repeat == 0].copy()
base = base[[T.text_valid(d, e) for d, e in zip(base.description, base.error)]]
piv = base.pivot_table(index='scenario_id', columns='variant',
                       values='description', aggfunc='first').dropna()
ids = sorted(piv.index.astype(int))
S = {i: piv.loc[i, 'secret'] for i in ids}
P = {i: piv.loc[i, 'public'] for i in ids}
angle = dict(zip(man.scenario_id.astype(int), man.angle_deg))


def hr(t):
    print('\n' + '=' * 78); print(t); print('=' * 78)


# ------------------------------------------------------------------ H
hr('H. DO THE TWO STORIES IN A PAIR END IDENTICALLY?')
print('Activations are taken at the LAST PROMPT TOKEN. If the stories end the')
print('same way, both members of a pair share the same final token.\n')
same_tail = 0
for i in ids:
    a = str(pairs.loc[i, 'story_secret']).rstrip()
    b = str(pairs.loc[i, 'story_public']).rstrip()
    if a[-60:] == b[-60:]:
        same_tail += 1
print(f'pairs whose stories share their last 60 characters: {same_tail}/{len(ids)}')
ex = ids[0]
print(f'\nexample scenario {ex}:')
print(f'  secret tail: ...{str(pairs.loc[ex, "story_secret"]).rstrip()[-90:]!r}')
print(f'  public tail: ...{str(pairs.loc[ex, "story_public"]).rstrip()[-90:]!r}')

# ------------------------------------------------------------------ I
hr('I. WHAT FINAL TOKEN DOES THE AV *CLAIM*?  (secret vs public)')
TOKPAT = re.compile(r'[Ff]inal token\s*"([^"]*)"')


def claimed(txt):
    m = TOKPAT.search(str(txt))
    return m.group(1).strip() if m else None


rows = []
for i in ids:
    cs, cp = claimed(S[i]), claimed(P[i])
    rows.append(dict(scenario_id=i, angle=angle[i], secret_tok=cs, public_tok=cp,
                     agree=(cs == cp) if (cs and cp) else None))
C = pd.DataFrame(rows)
parsed = C.dropna(subset=['agree'])
print(f'descriptions where a final-token claim was parseable: '
      f'{len(parsed)}/{len(C)} (both sides)')
if len(parsed):
    n_agree = int(parsed.agree.sum())
    print(f'pairs where secret and public claim the SAME final token: '
          f'{n_agree}/{len(parsed)} ({n_agree/len(parsed):.0%})')
    print(f'pairs where they DISAGREE: {len(parsed)-n_agree}/{len(parsed)} '
          f'({1-n_agree/len(parsed):.0%})')
    print('\nmost common claimed tokens:')
    allc = pd.concat([parsed.secret_tok, parsed.public_tok])
    print(allc.value_counts().head(12).to_string())
    print('\ndisagreeing pairs (secret -> public):')
    dis = parsed[~parsed.agree.astype(bool)]
    for _, r in dis.head(15).iterrows():
        print(f'  {int(r.scenario_id):>4}  {r.angle:>5.2f} deg   '
              f'{r.secret_tok!r:>16} -> {r.public_tok!r}')

# ------------------------------------------------------------------ J
hr('J. MULTIPLE COMPARISONS CONTEXT FOR SECTION D')
only_s, only_p = {}, {}
for i in ids:
    a, b = set(T.toks(S[i])), set(T.toks(P[i]))
    for w in a - b:
        only_s[w] = only_s.get(w, 0) + 1
    for w in b - a:
        only_p[w] = only_p.get(w, 0) + 1
tested = [w for w in set(only_s) | set(only_p)
          if only_s.get(w, 0) + only_p.get(w, 0) >= 5]
pvals = []
for w in tested:
    cs, cp = only_s.get(w, 0), only_p.get(w, 0)
    n = cs + cp
    pvals.append(st.binomtest(max(cs, cp), n, 0.5).pvalue)
pvals = np.array(pvals)
n_sig = int((pvals < 0.05).sum())
print(f'words tested (appearing asymmetrically in >=5 pairs): {len(tested)}')
print(f'significant at uncorrected p<0.05: {n_sig}')
print(f'expected by chance alone at p<0.05: {0.05*len(tested):.1f}')
print(f'smallest p-value observed: {pvals.min():.4f}')
print(f'Bonferroni threshold for {len(tested)} tests: {0.05/len(tested):.5f}')
print(f'any word surviving Bonferroni: {int((pvals < 0.05/len(tested)).sum())}')

# ------------------------------------------------------------------ K
hr('K. IS WITHIN-PAIR DIFFERENCE JUST GENERIC ANGLE SENSITIVITY?')
print('Compute angle(secret_i, public_j) for ALL i,j from the frozen npz, and')
print('the matching description similarity. If within-pairs (i==j) sit on the')
print('same angle->similarity curve as cross-pairs, nothing about the text')
print('difference is privacy-specific; it is what any rotation of that size does.\n')
npz = np.load(REPO / 'results/minimal_pairs_acts_f.npz')
sid = list(npz['scenario_ids'].astype(int))
idx = {s: k for k, s in enumerate(sid)}
A_s, A_p = npz['acts_secret'], npz['acts_public']


def ang(u, v):
    u = u / np.linalg.norm(u); v = v / np.linalg.norm(v)
    return np.degrees(np.arccos(np.clip(u @ v, -1, 1)))


rec = []
for i in ids:
    for j in ids:
        a = ang(A_s[idx[i]], A_p[idx[j]])
        sim = T.seq_similarity(T.slice_text(S[i], 'full'), T.slice_text(P[j], 'full'))
        rec.append(dict(i=i, j=j, within=(i == j), angle=a, sim=sim))
K = pd.DataFrame(rec)
wi, cr = K[K.within], K[~K.within]
print(f'within-pairs : n={len(wi):>4}  angle median {wi.angle.median():5.2f}  '
      f'sim median {wi.sim.median():.3f}')
print(f'cross-pairs  : n={len(cr):>4}  angle median {cr.angle.median():5.2f}  '
      f'sim median {cr.sim.median():.3f}')
rho, pv = st.spearmanr(K.angle, K.sim)
print(f'\nspearman(angle, sim) over ALL {len(K)} comparisons: rho={rho:+.3f} p={pv:.2e}')

# do cross-pairs at comparable angle exist?
lo, hi = wi.angle.min(), wi.angle.max()
overlap = cr[(cr.angle >= lo) & (cr.angle <= hi)]
print(f'\ncross-pairs falling in the within-pair angle range '
      f'[{lo:.2f}, {hi:.2f}] deg: {len(overlap)}')
if len(overlap) >= 5:
    print(f'  their sim median: {overlap.sim.median():.3f}  '
          f'vs within-pair sim median {wi.sim.median():.3f}')
    u, pu = st.mannwhitneyu(wi.sim, overlap.sim)
    print(f'  mann-whitney p={pu:.4f}')
else:
    print('  -> too few for a matched-angle comparison; the two populations')
    print('     barely overlap in angle, so this cannot be settled from these data.')
