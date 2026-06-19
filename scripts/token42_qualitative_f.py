"""token42_qualitative_f.py — what is "token 42"? (task 8)

Re-tokenizes stored tier-3 responses with the EXACT call the sweep uses
(tokenizer(response, add_special_tokens=False)) so that response-token index 41
(0-based) == sweep position k=42. Marks where k=42 lands in each leaked response,
and reports the response-length distribution (to check the constant-n assumption).
"""
import re
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
b = pd.read_csv('results/benchmark_results_bf16.csv')
t3 = b[b.tier == 'tier_3'].copy()
t3['ntok'] = t3['response'].astype(str).apply(lambda s: len(tok(s, add_special_tokens=False)['input_ids']))

print('=' * 78)
print('RESPONSE LENGTH DISTRIBUTION (tier-3, n=%d)' % len(t3))
print('=' * 78)
for lab in ['leaked', 'appropriate', 'refused']:
    n = t3[t3.label == lab]['ntok']
    print(f'  {lab:11s} n={len(n):3d}  min={n.min():3d}  p10={int(n.quantile(.1)):3d}  '
          f'median={int(n.median()):3d}  max={n.max():4d}  | #(<43 tok)={int((n<43).sum())}  #(<65 tok)={int((n<65).sum())}')
alln = t3['ntok']
print(f'  ALL         min={alln.min()} -> so pos k=64 valid for ALL? {(alln>=65).all()}  '
      f'(#responses <65 tok = {int((alln<65).sum())})')
print('  => if any are <65, the CSV constant-n means the sweep did NOT NaN-drop them; check pad logic.')

print('\n' + '=' * 78)
print('WHERE DOES TOKEN k=42 FALL? — 10 leaked tier-3 responses')
print('=' * 78)
leaked = t3[t3.label == 'leaked'].sort_values('scenario_id')
sample = leaked.head(10)
for _, r in sample.iterrows():
    ids = tok(str(r['response']), add_special_tokens=False)['input_ids']
    n = len(ids)
    print('\n' + '-' * 78)
    print(f'scenario_id={r.scenario_id}  response_tokens={n}')
    if n < 42:
        print(f'  [response shorter than 42 tokens — k=42 is NaN-padded for this scenario]')
        print(f'  full response: {r["response"]}')
        continue
    tok42 = tok.decode([ids[41]])
    prefix = tok.decode(ids[:42])      # tokens k=1..42 inclusive
    rest = tok.decode(ids[42:])
    print(f'  token at k=42 = {tok42!r}   (char offset ~{len(prefix)} of {len(str(r["response"]))})')
    marked = prefix + '  ⟦◀k=42⟧  ' + rest
    print('  ' + marked.replace('\n', ' '))

print('\n' + '=' * 78)
print('AGGREGATE: where does k=42 sit relative to response length (all leaked)?')
print('=' * 78)
lk = leaked[leaked.ntok >= 42]
frac = (42 / lk['ntok'])
print(f'  leaked responses with >=42 tokens: {len(lk)}/{len(leaked)}')
print(f'  k=42 as fraction-through-response: median={frac.median():.2f}  '
      f'p25={frac.quantile(.25):.2f}  p75={frac.quantile(.75):.2f}')
print(f'  (i.e. token 42 is typically {frac.median()*100:.0f}% of the way through a leaked reply)')
