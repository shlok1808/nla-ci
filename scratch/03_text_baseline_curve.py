"""Audit scratch 03 — text-only baseline for the position-sweep 'climb'.

At sweep position k the probe sees the residual state after k response tokens.
A pure-text model that sees (scenario + first k response tokens) is the
transcript-information baseline: if its AUC matches the activation AUC at the
same k, the 'climb' is transcript-borne, not internal-state evolution.
This is a free local anticipation of E2 (forced_prefix_f.py).

Also: full-response text AUC (how much leak evidence exists in the whole
response — upper anchor + judge-consistency proxy) and first-64-token vs
full-response comparison (is the sweep's 64-token window even wide enough
to contain the leak evidence?).
"""
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')

bench = pd.read_csv('results/benchmark_results_bf16.csv')
t3 = bench[bench.tier == 'tier_3'].sort_values('scenario_id').reset_index(drop=True)
y = (t3.label == 'leaked').astype(int).values
y_la_mask = (t3.label != 'refused').values
y_la = (t3.label[y_la_mask] == 'leaked').astype(int).values

resp_ids = [tok(str(r), add_special_tokens=False)['input_ids']
            for r in t3.response.fillna('')]
scen = t3.scenario.fillna('').values


def text_auc(texts, yy, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    clf = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
                        LogisticRegression(max_iter=2000))
    s = cross_val_predict(clf, texts, yy, cv=cv, method='predict_proba')[:, 1]
    return roc_auc_score(yy, s)


sw = pd.read_csv('results/position_sweep_aucs_f.csv')

print('k     text(scen+k-prefix)  best-acts-AUC@k(any layer)  acts@k L20')
print('      lvn    lva           lvn    lva                  lvn    lva')
for k in [0, 4, 8, 16, 24, 32, 42, 52, 64]:
    prefixes = np.array([s + ' ||| ' + tok.decode(ri[:k]) for s, ri in
                         zip(scen, resp_ids)])
    a_lvn = text_auc(prefixes, y)
    a_lva = text_auc(prefixes[y_la_mask], y_la)
    g = sw[sw.pos == k]
    b_lvn = g[g.target == 'leaked_vs_not'].auc.max()
    b_lva = g[g.target == 'leaked_vs_approp'].auc.max()
    l20_lvn = g[(g.target == 'leaked_vs_not') & (g.layer == 20)].auc.iloc[0]
    l20_lva = g[(g.target == 'leaked_vs_approp') & (g.layer == 20)].auc.iloc[0]
    print(f'{k:<5} {a_lvn:.3f}  {a_lva:.3f}         {b_lvn:.3f}  {b_lva:.3f}'
          f'                {l20_lvn:.3f}  {l20_lva:.3f}')

full = np.array([s + ' ||| ' + str(r) for s, r in zip(scen, t3.response.fillna(''))])
resp_only = t3.response.fillna('').values
print(f'\nfull response:  scen+resp text AUC lvn={text_auc(full, y):.3f}  '
      f'lva={text_auc(full[y_la_mask], y_la):.3f}')
print(f'response only:  text AUC lvn={text_auc(resp_only, y):.3f}  '
      f'lva={text_auc(resp_only[y_la_mask], y_la):.3f}')
first64 = np.array([tok.decode(ri[:64]) for ri in resp_ids])
print(f'first 64 resp tokens only: text AUC lvn={text_auc(first64, y):.3f}  '
      f'lva={text_auc(first64[y_la_mask], y_la):.3f}')

lens = np.array([len(ri) for ri in resp_ids])
print(f'\nresponse token lengths: min={lens.min()} median={np.median(lens):.0f} '
      f'max={lens.max()}; frac > 64 tokens: {(lens > 64).mean():.2%}')
