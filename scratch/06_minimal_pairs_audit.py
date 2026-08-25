"""Audit scratch 06 — residual contamination in the 233 valid minimal pairs.

1. Length asymmetries: word / char / Qwen-token deltas; length-only AUC.
2. Lexical confound baseline: pair-grouped 5-fold TF-IDF logistic AUC
   (secret vs public) — replicates stage_validate check [4] locally, pre-GPU.
3. Marker-ablation: strip the obvious secrecy/publicity lexicon and re-run —
   how much lexical signal remains beyond the by-construction markers?
4. Function-word-only probe (style confound).
5. Edit-size stats: words changed per pair, clauses touched.
6. Top discriminative n-grams both directions.
"""
import difflib
import re
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

p = pd.read_csv('data/minimal_pairs_f.csv')
v = p[p.valid].reset_index(drop=True)
sec = v.story_secret.astype(str).values
pub = v.story_public.astype(str).values
n = len(v)
print(f'valid pairs: {n} (of {len(p)})')

# ── 1. length asymmetries ────────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
wl_s = np.array([len(t.split()) for t in sec])
wl_p = np.array([len(t.split()) for t in pub])
cl_s = np.array([len(t) for t in sec])
cl_p = np.array([len(t) for t in pub])
tl_s = np.array([len(tok(t, add_special_tokens=False)['input_ids']) for t in sec])
tl_p = np.array([len(tok(t, add_special_tokens=False)['input_ids']) for t in pub])
for nm, a, b in [('words', wl_s, wl_p), ('chars', cl_s, cl_p),
                 ('qwen tokens', tl_s, tl_p)]:
    dd = a - b
    print(f'  {nm:<12} secret-public delta: mean={dd.mean():+.2f} '
          f'median={np.median(dd):+.0f} sd={dd.std():.1f} '
          f'|d|>10: {(np.abs(dd) > 10).sum()}')


def pair_grouped_auc_feats(F_s, F_p, C=1.0, seed=0):
    """Pair-grouped 5-fold CV AUC on precomputed feature matrices."""
    import scipy.sparse as sp
    stack = sp.vstack if sp.issparse(F_s) else np.vstack
    X = stack([F_s, F_p])
    y = np.r_[np.ones(n, int), np.zeros(n, int)]
    aucs = []
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(np.arange(n)):
        itr = np.r_[tr, tr + n]
        ite = np.r_[te, te + n]
        clf = LogisticRegression(C=C, max_iter=5000).fit(X[itr], y[itr])
        aucs.append(roc_auc_score(y[ite], clf.predict_proba(X[ite])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


a_len, _ = pair_grouped_auc_feats(tl_s.reshape(-1, 1).astype(float),
                                  tl_p.reshape(-1, 1).astype(float))
print(f'  token-length-only AUC: {a_len:.3f} (want ~0.5; gate flags >0.60)')

# ── 2. full TF-IDF lexical baseline (stage_validate [4] replica) ─────────────
def tfidf_auc(sec_txt, pub_txt, label, **vec_kw):
    vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), **vec_kw)
    Xall = vec.fit_transform(list(sec_txt) + list(pub_txt))
    a, sd = pair_grouped_auc_feats(Xall[:n], Xall[n:])
    print(f'  {label:<44} AUC={a:.3f} ± {sd:.3f}')
    return vec, Xall, a


vec, Xall, auc_full = tfidf_auc(sec, pub, 'TF-IDF (full vocab, 1-2gram)')

# top discriminative n-grams (fit on all — descriptive only)
y_all = np.r_[np.ones(n, int), np.zeros(n, int)]
full = LogisticRegression(C=1.0, max_iter=5000).fit(Xall, y_all)
names = np.array(vec.get_feature_names_out())
coef = full.coef_[0]
print(f'    ↑secret: {", ".join(names[np.argsort(coef)[-15:]][::-1])}')
print(f'    ↑public: {", ".join(names[np.argsort(coef)[:15]])}')

# ── 3. marker ablation ───────────────────────────────────────────────────────
MARKERS = re.compile(
    r'\b(only|secret\w*|confid\w*|priv\w*|between (?:us|the two of )?them\w*|'
    r'no one else|nobody else|anyone else|kept?\w* (?:it|this)? ?(?:to (?:her|him|them)sel\w*|quiet|hidden)|'
    r'never (?:told|revealed|shared|mentioned)|discreet\w*|'
    r'everyone|everybody|openly|open(?:ness)?|common knowledge|widely|'
    r'publicly|public|freely|discussed|aware|knew about|well[- ]known|known (?:to|by) all|'
    r'no secret|shared (?:openly|with everyone|freely)|open secret|transparent\w*)\b',
    re.IGNORECASE)
sec_abl = [MARKERS.sub(' ', t) for t in sec]
pub_abl = [MARKERS.sub(' ', t) for t in pub]
_, _, auc_abl = tfidf_auc(sec_abl, pub_abl, 'TF-IDF after marker-lexicon ablation')

# harsher: also remove any word that appears in the pair diff (per-pair edit words)
def diff_words(a, b):
    aw, bw = a.split(), b.split()
    out = set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, aw, bw).get_opcodes():
        if tag != 'equal':
            out |= {w.strip('.,!?;:"\'').lower() for w in aw[i1:i2] + bw[j1:j2]}
    return out


all_edit_words = set()
edit_sizes = []
for s_, p_ in zip(sec, pub):
    dw = diff_words(s_, p_)
    all_edit_words |= dw
    edit_sizes.append(len(dw))
print(f'\n[5] edit sizes: median {np.median(edit_sizes):.0f} distinct words/pair, '
      f'p90={np.percentile(edit_sizes, 90):.0f}; '
      f'union of edit vocab: {len(all_edit_words)} words')
rx_edit = re.compile(r'\b(' + '|'.join(re.escape(w) for w in sorted(all_edit_words)
                                       if w and len(w) > 1) + r')\b', re.IGNORECASE)
sec_h = [rx_edit.sub(' ', t) for t in sec]
pub_h = [rx_edit.sub(' ', t) for t in pub]
_, _, auc_h = tfidf_auc(sec_h, pub_h,
                        'TF-IDF after removing ALL edit-vocab words')

# ── 4. function-word-only probe ──────────────────────────────────────────────
stop = sorted(ENGLISH_STOP_WORDS)
def keep_stop(t):
    return ' '.join(w for w in re.findall(r"[a-z']+", t.lower()) if w in stop)
_, _, auc_fw = tfidf_auc([keep_stop(t) for t in sec], [keep_stop(t) for t in pub],
                         'TF-IDF function-words only (style)')

print(f'\nSummary: full={auc_full:.3f}  marker-ablated={auc_abl:.3f}  '
      f'edit-vocab-removed={auc_h:.3f}  function-words={auc_fw:.3f}  '
      f'length-only={a_len:.3f}')
print('Reading: the activation probe must beat the FULL text AUC for any '
      'privileged-encoding claim (stage_validate [4] delta).')
