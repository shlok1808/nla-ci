"""
alpha_sweep_f.py — SUPERSEDED 2026-08-25 by verbalize_directions_f.py (audit F4).
Kept for the record; do not run. The fixed alpha grid below is calibrated to
||diff_raw||~4 and mis-rotates any other direction; the deflection direction is
missing; there is no direction-specificity control. See prereg amendment A1.

Original docstring:

Verbalize counterfactual rotations properly: denoised
directions, meaningful alphas, temperature 0.

Fixes three problems with the session-04 counterfactual attempt (L7/L8/L9):
  1. alpha=2 rotated the mean by ~5 degrees — geometrically invisible to a
     direction-only NLA (inj renormalizes to L2=150). We sweep alpha so the
     rotation reaches ~12/27/45/65 degrees (alpha = 5/10/20/40 at ||diff||~4).
  2. ~1/3 of the raw diff's energy is label-sampling noise. We add two denoised
     variants: (a) projection onto the top-50 PCA subspace of the activation
     cloud (kills off-manifold noise), (b) bootstrap sign-consistency mask
     (zeros dimensions whose diff sign flips across label resamples).
  3. All previous NLA calls sampled at temperature 1.0 (L9). Everything here is
     temperature=0 so vector pairs differing only by alpha are comparable.

Directions swept:
  - diff_raw        leaked_mean - not_leaked_mean (tier 3)
  - diff_pca        diff projected onto top-50 PCs
  - diff_boot       diff masked to sign-consistent dims (>=95% of 200 resamples)
  - v_privacy       minimal-pairs direction, IF results/v_privacy_f.npz exists
                    (run minimal_pairs_f.py first — strongly recommended: it is
                    the only direction here not derived from behavior labels)

For each direction and alpha: verbalize not_leaked_mean + alpha*diff, plus
3 individual not-leaked activations + alpha*diff (richer in-distribution
structure than the mean). CI-term audit on every output.

Honest expectation (do not oversell): the raw/behavioral directions carry only
AUC-0.68 worth of signal; even a perfect verbalizer might have little to say.
v_privacy is the variant with a real shot. If ALL variants stay
format-dominated through 45 degrees and then go to gibberish with no privacy
phase in between, that is the clean negative: no readable window exists.

Setup: SGLang serving the AV (see run_nla.py docstring). ~60 NLA calls, minutes.

Usage:
    tmux new -s alpha
    python scripts/alpha_sweep_f.py

Outputs: results/alpha_sweep_f.csv, results/alpha_sweep_f.txt
"""

import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from nla_inference import NLAClient

# ── Config ────────────────────────────────────────────────────────────────────

ACTOR_DIR    = Path('./actor_hf')
SGLANG_URL   = 'http://localhost:30000'
ACTIVATIONS  = Path('results/activations_layer20.npz')
VPRIV_PATH   = Path('results/v_privacy_f.npz')
OUT_CSV      = Path('results/alpha_sweep_f.csv')
OUT_TXT      = Path('results/alpha_sweep_f.txt')

ALPHAS       = [0, 2, 5, 10, 20, 40]
N_PCS        = 50
N_BOOT       = 200
BOOT_CONSIST = 0.95
N_INDIV      = 3          # individual not-leaked activations to also perturb
INDIV_ALPHAS = [5, 20]
SEED         = 0

CI_TERMS = re.compile(
    r'\b(privacy|private|confidential|secret|disclos\w*|leak\w*|betray\w*|'
    r'sensitive information|should not (?:be )?(?:shared|told|revealed)|breach)\b',
    re.IGNORECASE)


# ── Directions ────────────────────────────────────────────────────────────────

def build_directions(acts, labels):
    rng = np.random.default_rng(SEED)
    leaked = acts[labels == 'leaked']
    notlk = acts[labels != 'leaked']
    diff = leaked.mean(0) - notlk.mean(0)

    # PCA-subspace projection (top N_PCS of the tier-3 cloud)
    X = acts - acts.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt[:N_PCS]                                    # (k, d)
    diff_pca = V.T @ (V @ diff)

    # Bootstrap sign-consistency mask
    signs = np.zeros((N_BOOT, acts.shape[1]), dtype=np.int8)
    y = (labels == 'leaked')
    for b in range(N_BOOT):
        idx = rng.choice(len(acts), len(acts), replace=True)
        d_b = acts[idx][y[idx]].mean(0) - acts[idx][~y[idx]].mean(0)
        signs[b] = np.sign(d_b)
    frac_pos = (signs > 0).mean(0)
    mask = (frac_pos >= BOOT_CONSIST) | (frac_pos <= 1 - BOOT_CONSIST)
    diff_boot = diff * mask
    print(f'bootstrap mask keeps {mask.sum()}/{len(mask)} dims '
          f'({mask.mean():.1%}); ||diff_boot||={np.linalg.norm(diff_boot):.3f} '
          f'vs ||diff||={np.linalg.norm(diff):.3f}')

    dirs = {'diff_raw': diff, 'diff_pca': diff_pca, 'diff_boot': diff_boot}
    if VPRIV_PATH.exists():
        dirs['v_privacy'] = np.load(VPRIV_PATH)['v_privacy']
        print('v_privacy loaded — included in sweep.')
    else:
        print('NOTE: results/v_privacy_f.npz not found. Run minimal_pairs_f.py '
              'first — v_privacy is the variant most likely to verbalize.')
    return dirs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    data = np.load(ACTIVATIONS, allow_pickle=True)
    t3 = data['tiers'] == 'tier_3'
    acts = data['activations'][t3]
    labels = data['labels'][t3]
    notlk_mean = acts[labels != 'leaked'].mean(0)
    base_norm = np.linalg.norm(notlk_mean)

    dirs = build_directions(acts, labels)

    rng = np.random.default_rng(SEED)
    indiv_idx = rng.choice(np.where(labels != 'leaked')[0], N_INDIV, replace=False)

    print(f'\nConnecting to SGLang at {SGLANG_URL}...')
    client = NLAClient(str(ACTOR_DIR), sglang_url=SGLANG_URL)
    print('Connected.\n')

    rows, lines = [], []

    def emit(name, vec, meta):
        desc = client.generate(vec, temperature=0, max_new_tokens=300)
        hit = bool(CI_TERMS.search(desc))
        rows.append(dict(**meta, name=name, l2=float(np.linalg.norm(vec)),
                         ci_terms=hit, description=desc))
        lines.append(f'\n[{name}]  L2={np.linalg.norm(vec):.1f}  '
                     f'CI-terms={"YES" if hit else "no"}\n  {desc}')
        print(lines[-1])

    for dname, d in dirs.items():
        d_norm = np.linalg.norm(d)
        lines.append('\n' + '=' * 70 + f'\nDIRECTION {dname}  ||d||={d_norm:.3f}\n' + '=' * 70)
        print(lines[-1])
        for a in ALPHAS:
            vec = notlk_mean + a * d
            # rotation of the renormalized direction vs the unperturbed mean
            cos = (vec @ notlk_mean) / (np.linalg.norm(vec) * base_norm)
            angle = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
            emit(f'{dname} mean+{a}d ({angle:.0f}deg)', vec,
                 dict(direction=dname, alpha=a, base='not_leaked_mean',
                      rotation_deg=angle))
        for a in INDIV_ALPHAS:
            for j, i in enumerate(indiv_idx):
                emit(f'{dname} indiv{j}+{a}d', acts[i] + a * d,
                     dict(direction=dname, alpha=a, base=f'indiv_{int(i)}',
                          rotation_deg=np.nan))

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    OUT_TXT.write_text('\n'.join(lines))
    print(f'\nSaved {OUT_CSV} and {OUT_TXT}')

    hits = pd.DataFrame(rows)
    print('\nCI-term hit rate by direction:')
    print(hits.groupby('direction')['ci_terms'].mean().round(2).to_string())


if __name__ == '__main__':
    main()
