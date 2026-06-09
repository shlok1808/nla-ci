"""
diff_of_means.py — Compute leaked-minus-appropriate difference vector and verbalize it.

Computes the mean activation for leaked scenarios, subtracts the mean for appropriate
scenarios, and feeds the resulting direction vector into the NLA verbalizer.

Rationale: per-scenario NLA descriptions are dominated by format/syntax noise.
The difference vector cancels common noise and leaves only the systematic signal
separating leaked from appropriate — which should be the CI signal if it exists
and is human-readable.

Setup: SGLang must be running with the NLA verbalizer (see run_nla.py docstring).

Usage:
    python scripts/diff_of_means.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from nla_inference import NLAClient


# ── Config ────────────────────────────────────────────────────────────────────

ACTOR_DIR       = Path('./actor_hf')
SGLANG_URL      = 'http://localhost:30000'
ACTIVATIONS     = Path('results/activations_layer20.npz')
OUTPUT_NPZ      = Path('results/diff_vectors.npz')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load activations
    print(f'Loading {ACTIVATIONS}...')
    data = np.load(ACTIVATIONS, allow_pickle=True)
    activations = data['activations']   # (496, 3584)
    labels      = data['labels']
    tiers       = data['tiers']

    # Tier 3 only — primary contrast set
    t3_mask     = tiers == 'tier_3'
    t3_acts     = activations[t3_mask]
    t3_labels   = labels[t3_mask]
    t3_tiers    = tiers[t3_mask]

    leaked_mask     = t3_labels == 'leaked'
    appropriate_mask = t3_labels == 'appropriate'
    refused_mask    = t3_labels == 'refused'
    not_leaked_mask = appropriate_mask | refused_mask   # collapsed

    print(f'\nTier 3 breakdown:')
    print(f'  leaked:      {leaked_mask.sum()}')
    print(f'  appropriate: {appropriate_mask.sum()}')
    print(f'  refused:     {refused_mask.sum()}')
    print(f'  not-leaked:  {not_leaked_mask.sum()}')

    # Compute means
    leaked_mean      = t3_acts[leaked_mask].mean(axis=0)       # (3584,)
    appropriate_mean = t3_acts[appropriate_mask].mean(axis=0)  # (3584,)
    not_leaked_mean  = t3_acts[not_leaked_mask].mean(axis=0)   # (3584,)

    # Difference vectors
    diff_leaked_vs_appropriate = leaked_mean - appropriate_mean
    diff_leaked_vs_not_leaked  = leaked_mean - not_leaked_mean

    print(f'\nDifference vector stats (leaked - appropriate):')
    print(f'  Norm:  {np.linalg.norm(diff_leaked_vs_appropriate):.4f}')
    print(f'  Mean:  {diff_leaked_vs_appropriate.mean():.6f}')
    print(f'  Std:   {diff_leaked_vs_appropriate.std():.6f}')

    print(f'\nDifference vector stats (leaked - not_leaked):')
    print(f'  Norm:  {np.linalg.norm(diff_leaked_vs_not_leaked):.4f}')
    print(f'  Mean:  {diff_leaked_vs_not_leaked.mean():.6f}')
    print(f'  Std:   {diff_leaked_vs_not_leaked.std():.6f}')

    # Verbalize via NLA
    print(f'\nConnecting to SGLang at {SGLANG_URL}...')
    client = NLAClient(str(ACTOR_DIR), sglang_url=SGLANG_URL)
    print('Connected.\n')

    # Counterfactual vectors: start from a natural activation, push along the leaked direction.
    # Raw diff vectors are out-of-distribution (norm ~4, vs natural activations ~100+).
    # Adding the diff back to a mean vector keeps it in-distribution while amplifying signal.
    counterfactual_from_not_leaked = not_leaked_mean + 2 * diff_leaked_vs_not_leaked
    counterfactual_from_appropriate = appropriate_mean + 2 * diff_leaked_vs_appropriate

    np.savez(
        OUTPUT_NPZ,
        leaked_mean=leaked_mean,
        appropriate_mean=appropriate_mean,
        not_leaked_mean=not_leaked_mean,
        diff_leaked_vs_appropriate=diff_leaked_vs_appropriate,
        diff_leaked_vs_not_leaked=diff_leaked_vs_not_leaked,
        counterfactual_from_not_leaked=counterfactual_from_not_leaked,
        counterfactual_from_appropriate=counterfactual_from_appropriate,
    )
    print(f'Vectors saved to {OUTPUT_NPZ}')

    vectors = {
        'leaked_mean':                      leaked_mean,
        'appropriate_mean':                 appropriate_mean,
        'not_leaked_mean':                  not_leaked_mean,
        'counterfactual_from_not_leaked':   counterfactual_from_not_leaked,
        'counterfactual_from_appropriate':  counterfactual_from_appropriate,
    }

    print('=' * 60)
    print('NLA DESCRIPTIONS')
    print('=' * 60)
    for name, vec in vectors.items():
        desc = client.generate(vec)
        print(f'\n[{name}]')
        print(f'  {desc}')

    print('\n' + '=' * 60)
    print('Done.')


if __name__ == '__main__':
    main()
