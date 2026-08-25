"""Audit scratch 05 — quantitative analysis of nla_descriptions.csv.

1. Format-vs-content: fraction of descriptions dominated by structure talk.
2. Clustering: k-means on TF-IDF; cluster themes and label composition
   (chi-square: are clusters associated with leak/refuse labels at all?).
3. Privacy-semantics lexicon counts (narrow + broad) by tier and label.
4. Boilerplate quantification: shared n-gram mass, near-duplicate rate.
"""
import re
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import chi2_contingency

nla = pd.read_csv('results/nla_descriptions.csv').sort_values(
    'scenario_id').reset_index(drop=True)
desc = nla.description.fillna('').values
tier = nla.tier.values
label = nla.label.values
t3 = tier == 'tier_3'

print(f'n={len(desc)}, mean len={np.mean([len(d.split()) for d in desc]):.0f} words')

# ── 1. format vs content term frequencies ────────────────────────────────────
FORMAT_TERMS = re.compile(
    r'\b(structur\w*|format\w*|template|dialogue|conversation\w*|narrative|'
    r'token\w*|quote|header|colon|newline|prompt|respond\w*|response|turn-based|'
    r'speaker|persona|role-?play\w*|instruction\w*|scenario)\b', re.IGNORECASE)
NARROW_CI = re.compile(
    r'\b(privacy|private|confidential\w*|secret\w*|disclos\w*|leak\w*|'
    r'betray\w*|sensitive information|breach)\b', re.IGNORECASE)
BROAD_CI = re.compile(
    r'\b(privacy|private|confidential\w*|secret\w*|disclos\w*|leak\w*|betray\w*|'
    r'sensitive\w*|discret\w*|discreet\w*|conceal\w*|withh\w*|hide|hidden|'
    r'trust\w*|personal (?:information|matter|detail)|reveal\w*|'
    r'not (?:to )?(?:share|tell|mention)|between (?:us|them)|keep\w* (?:it|this|quiet))\b',
    re.IGNORECASE)

fmt_frac = np.mean([bool(FORMAT_TERMS.search(d)) for d in desc])
fmt_per_desc = np.mean([len(FORMAT_TERMS.findall(d)) for d in desc])
print(f'\n[1] format/structure terms: {fmt_frac:.1%} of descriptions contain >=1; '
      f'mean {fmt_per_desc:.1f} format terms per description')

# ── 2. clustering ─────────────────────────────────────────────────────────────
vec = TfidfVectorizer(max_features=3000, stop_words='english', min_df=3)
Xt = vec.fit_transform(desc)
K = 8
km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(Xt)
cl = km.labels_
terms = np.array(vec.get_feature_names_out())
print(f'\n[2] k-means k={K} clusters on TF-IDF of descriptions:')
for c in range(K):
    m = cl == c
    top = terms[np.argsort(km.cluster_centers_[c])[-8:]][::-1]
    tiers_in = pd.Series(tier[m]).value_counts().to_dict()
    labs_in = pd.Series(label[m]).value_counts().to_dict()
    print(f'  c{c} (n={m.sum():>3}): {", ".join(top)}')
    print(f'      tiers={tiers_in} labels={labs_in}')

# cluster-label association within tier 3 only
ct = pd.crosstab(cl[t3], label[t3])
chi2, p, dof, _ = chi2_contingency(ct)
print(f'\n  tier-3 cluster x label chi-square: chi2={chi2:.1f}, dof={dof}, p={p:.4f}')
ct3 = pd.crosstab(cl[t3], label[t3], normalize='columns')
print(ct3.round(2).to_string())

# ── 3. privacy semantics ─────────────────────────────────────────────────────
print('\n[3] privacy/secrecy/disclosure semantics in descriptions:')
for name, rx in [('narrow', NARROW_CI), ('broad', BROAD_CI)]:
    hits = np.array([bool(rx.search(d)) for d in desc])
    print(f'  {name} lexicon: {hits.sum()}/496 overall')
    tab = pd.crosstab(tier[hits], label[hits]) if hits.any() else None
    print(pd.crosstab(pd.Series(tier, name='tier'),
                      pd.Series(hits, name='hit')).to_string())
    t3h = hits[t3]
    print(f'  tier-3 hits by label: '
          f'{pd.crosstab(label[t3], t3h)[True].to_dict() if t3h.any() else 0}')

# what do the tier-3 hits actually say?
rx = BROAD_CI
print('\n  sample tier-3 hits (label | matched | first 160 chars):')
cnt = 0
for i in np.where(t3)[0]:
    m = rx.search(desc[i])
    if m and cnt < 8:
        print(f'   - {label[i]:<11} | {m.group(0):<14} | {desc[i][:160]}')
        cnt += 1

# ── 4. boilerplate ───────────────────────────────────────────────────────────
first_words = pd.Series([' '.join(d.split()[:2]).lower() for d in desc])
print(f'\n[4] top opening bigrams: {first_words.value_counts().head(6).to_dict()}')
# near-duplicate rate via cosine on TF-IDF
from sklearn.metrics.pairwise import cosine_similarity
S = cosine_similarity(Xt)
np.fill_diagonal(S, 0)
print(f'  near-duplicate (cos>0.8) pairs: {int((S > 0.8).sum() / 2)}; '
      f'mean max-similarity per description: {S.max(axis=1).mean():.3f}')
print(f'  mean pairwise cosine (all): {S.mean():.3f}; '
      f'within tier 3: {S[np.ix_(t3, t3)].mean():.3f}')
