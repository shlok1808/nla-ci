"""
07 — What carries the description-channel deflection signal?

The triad says descriptions retain roughly half the privileged deflection signal
(desc refused-vs-rest ~0.76; desc-input +0.138, p=.042). scratch/05 showed the
descriptions contain almost no privacy vocabulary (21/496 narrow-lexicon hits),
so whatever the 0.76 rides on, it is not CI language. This script identifies the
actual features: a pair of grouped-CV TF-IDF probes (refused vs rest, refused vs
appropriate) on the tier-3 descriptions from results/nla_descriptions.csv, with
the top +/- coefficients printed.

Purpose: calibrates the E-NLA scoring endpoints (prereg amendment A1) — if the
deflection trace in descriptions is tonal ("supportive"/"understand" depletion)
rather than lexical, the lexicon endpoint alone is insufficient and the text
probe is the sensitive one. Run locally, free.

Usage: python scratch/07_desc_deflection_features.py
       | tee results/audit/07_desc_deflection_features.txt
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv('results/nla_descriptions.csv')
t3 = df[df['tier'] == 'tier_3'].copy()
print(f'tier-3 descriptions: {len(t3)}  labels: {t3["label"].value_counts().to_dict()}')

for target, mask in [('refused_vs_rest', np.ones(len(t3), bool)),
                     ('refused_vs_approp', (t3['label'] != 'leaked').values)]:
    sub = t3[mask]
    y = (sub['label'] == 'refused').astype(int).values
    texts = sub['description'].fillna('').tolist()

    # vectorizer fit inside each fold — no vocabulary leakage (cf. audit F9)
    aucs, coefs = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(texts, y):
        vec = TfidfVectorizer(min_df=3, ngram_range=(1, 2), stop_words=None)
        Xtr = vec.fit_transform([texts[i] for i in tr])
        Xte = vec.transform([texts[i] for i in te])
        clf = LogisticRegression(C=1.0, max_iter=5000).fit(Xtr, y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    print(f'\n[{target}] grouped-CV TF-IDF AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f} '
          f'(n={len(sub)}, {y.sum()} refused)')

    # full-data fit for feature inspection only (not an AUC claim)
    vec = TfidfVectorizer(min_df=3, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    clf = LogisticRegression(C=1.0, max_iter=5000).fit(X, y)
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    print(f'  ↑refused : {", ".join(names[order[-20:]][::-1])}')
    print(f'  ↑rest    : {", ".join(names[order[:20]])}')

print('\nReading guide: if the ↑refused features are tone/structure words rather '
      'than privacy vocabulary, the E-NLA lexicon endpoint is insufficient alone '
      'and the description-text probe (tertiary endpoint, amendment A1) is the '
      'sensitive readout. NOTE: all descriptions here are temperature-1.0 samples '
      '(L9 §3) — the endogenous temp-0 reads in verbalize_directions_f.py are the '
      'clean replacement.')
