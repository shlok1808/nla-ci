"""
08 — Is the signal in the later paragraphs? (user observation, 2026-08-25)

L6 concluded the per-scenario NLA descriptions are "structure-focused, not
CI-focused" and treated that as a negative. But every tier-3 description has
exactly THREE paragraphs with a consistent division of labour:

  P1 (~23 words)  "Structured <format> with <topic>..."   — the boilerplate
  P2 (~40 words)  quotes the prompt pattern, reads the setup
  P3 (~39 words)  "Final token X ... immediately requiring <predicted
                   continuation>" — the NLA's forecast of what the model is
                   about to SAY, with example quoted continuations

If a human reading past P1 can tell what is going on, the signal is not absent,
it is buried under a fixed prefix that the whole-description bag-of-words
analysis averages over. P3 in particular is not description of format at all —
it is a behavioural forecast, and forecasts of an upcoming reply are exactly
what a leak/deflect probe should be able to read.

Test: run the same grouped-CV text probe (scratch/07's) on each paragraph slice
separately. If P3 or P2+P3 beats P1 and beats the whole description, the
"formatting dump" framing is wrong and L6 needs qualifying.

Free, local. Usage:
    python scratch/08_paragraph_slices.py | tee results/audit/08_paragraph_slices.txt
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv('results/nla_descriptions.csv')
t3 = df[df['tier'] == 'tier_3'].copy()
paras = t3['description'].fillna('').map(
    lambda s: [p.strip() for p in s.split('\n\n') if p.strip()])
assert set(paras.map(len)) == {3}, f'expected 3 paragraphs everywhere, got {set(paras.map(len))}'

SLICES = {
    'P1 only (boilerplate)':  paras.map(lambda p: p[0]),
    'P2 only (setup)':        paras.map(lambda p: p[1]),
    'P3 only (forecast)':     paras.map(lambda p: p[2]),
    'P2+P3 (past prefix)':    paras.map(lambda p: ' '.join(p[1:])),
    'whole description':      t3['description'].fillna(''),
}


def probe(texts, y, seed=0):
    """Grouped-CV TF-IDF AUC, vectorizer fit inside each fold (no leakage)."""
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(texts, y):
        vec = TfidfVectorizer(min_df=3, ngram_range=(1, 2))
        Xtr = vec.fit_transform([texts[i] for i in tr])
        Xte = vec.transform([texts[i] for i in te])
        clf = LogisticRegression(C=1.0, max_iter=5000).fit(Xtr, y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


for target, mask in [('refused vs appropriate', (t3['label'] != 'leaked').values),
                     ('refused vs rest',        np.ones(len(t3), bool)),
                     ('leaked vs appropriate',  (t3['label'] != 'refused').values)]:
    sub_y = (t3['label'] == ('refused' if 'refused' in target else 'leaked'))
    y = sub_y[mask].astype(int).values
    print(f'\n=== {target}  (n={mask.sum()}, {y.sum()} positive) ===')
    print(f'{"slice":<26} {"AUC":>6}  {"SD":>5}   {"mean words":>10}')
    for name, series in SLICES.items():
        texts = series[mask].tolist()
        # 10 seeds — n is small (36 positives), single-split AUCs are noisy
        runs = [probe(texts, y, seed=s)[0] for s in range(10)]
        wl = int(np.mean([len(t.split()) for t in texts]))
        print(f'{name:<26} {np.mean(runs):>6.3f}  {np.std(runs):>5.3f}   {wl:>10}')

# What words carry P3, the forecast paragraph?
print('\n=== P3 (forecast) discriminative features, refused vs appropriate ===')
mask = (t3['label'] != 'leaked').values
texts = SLICES['P3 only (forecast)'][mask].tolist()
y = (t3['label'][mask] == 'refused').astype(int).values
vec = TfidfVectorizer(min_df=3, ngram_range=(1, 2))
X = vec.fit_transform(texts)
clf = LogisticRegression(C=1.0, max_iter=5000).fit(X, y)
names = np.array(vec.get_feature_names_out())
order = np.argsort(clf.coef_[0])
print(f'  ↑refused     : {", ".join(names[order[-20:]][::-1])}')
print(f'  ↑appropriate : {", ".join(names[order[:20]])}')

print('\nNOTE: these descriptions are temperature-1.0 samples (L9 §3), so each is')
print('one stochastic draw. Any slice difference here is a LOWER bound — decode')
print('noise can only hurt. The temp-0 endogenous reads in')
print('verbalize_directions_f.py are the clean replacement.')
