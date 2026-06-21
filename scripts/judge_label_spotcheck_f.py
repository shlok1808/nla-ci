"""
judge_label_spotcheck_f.py — eyeball the GPT-4o-mini leak labels (the 2x2 y-axis).

The minimal-pairs projection (minimal_pairs_f.py --stage project) tests whether weak
privacy encoding predicts the *leaked* label. That label is judge-derived and carries
~10% noise (anticipated limitation in METHODOLOGY_f §4). Before trusting the projection,
sanity-check the labels by hand: seeded-sample 15 `leaked` + 15 `appropriate` tier-3
scenarios and render the story, the model's response, and the judge's reasoning so a
human can confirm the calls are genuine CI violations / non-violations.

Local, independent, instant — no GPU, no API key. pandas only.

Usage:
    python scripts/judge_label_spotcheck_f.py            # 15 + 15, seed 0
    python scripts/judge_label_spotcheck_f.py --n 20 --seed 7

Output: results/judge_label_spotcheck_f.csv (+ printed table)
"""

import argparse
import textwrap
import pandas as pd
from pathlib import Path

BENCHMARK_CSV = Path('results/benchmark_results_bf16.csv')
OUT_CSV       = Path('results/judge_label_spotcheck_f.csv')
TIER3_IDS     = range(206, 476)          # tier-3 scenario_id range (inclusive 206..475)


def _wrap(s, width=88, maxlen=900):
    s = str(s).replace('\n', ' ').strip()
    if len(s) > maxlen:
        s = s[:maxlen].rstrip() + ' …'
    return textwrap.fill(s, width=width, subsequent_indent='    ')


def main():
    ap = argparse.ArgumentParser(description='Spot-check tier-3 leak labels by hand.')
    ap.add_argument('--n', type=int, default=15, help='samples per class (leaked / appropriate)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(BENCHMARK_CSV)
    t3 = df[df['scenario_id'].isin(TIER3_IDS)]
    assert len(t3) == 270, f'expected 270 tier-3 rows, got {len(t3)}'

    picks = []
    for label in ['leaked', 'appropriate']:
        grp = t3[t3['label'] == label]
        k = min(args.n, len(grp))
        picks.append(grp.sample(n=k, random_state=args.seed).assign(_label=label))
    sample = pd.concat(picks).sort_values(['label', 'scenario_id'])

    cols = ['scenario_id', 'label', 'scenario', 'response', 'judge_reasoning']
    sample[cols].to_csv(OUT_CSV, index=False)

    counts = t3['label'].value_counts().to_dict()
    print(f'tier-3 label population: {counts}')
    print(f'sampled {args.n}+{args.n} (seed {args.seed}) -> {OUT_CSV}\n')
    for label in ['leaked', 'appropriate']:
        print('#' * 90)
        print(f'#  {label.upper()}  (eyeball: does the judge call look right?)')
        print('#' * 90)
        for _, r in sample[sample['label'] == label].iterrows():
            print(f'\n— id {r["scenario_id"]} —')
            print('  SCENARIO:  ' + _wrap(r['scenario']))
            print('  RESPONSE:  ' + _wrap(r['response']))
            print('  JUDGE:     ' + _wrap(r['judge_reasoning'], maxlen=400))


if __name__ == '__main__':
    main()
