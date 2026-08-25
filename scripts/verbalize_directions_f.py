"""
verbalize_directions_f.py — E-NLA, redesigned (supersedes alpha_sweep_f.py).

Session-B verbalization of behavioural directions with the controls that make
free-text output admissible as evidence. Three design changes vs alpha_sweep_f.py
(2026-08-25 audit F4; prereg amendment A1):

  1. ANGLE-TARGETED, not alpha-targeted. alpha_sweep's fixed grid [0,2,5,10,20,40]
     was calibrated to ||diff_raw||~4; for v_deflect (||.||~8.5) or v_privacy it
     skips the informative 5-30 degree band. Here alpha is solved per (base,
     direction) to hit target rotations THETAS = +/-{10,20,30,45,60} degrees.
  2. DIRECTION-SPECIFICITY CONTROLS. Without them, "caution vocabulary appeared"
     cannot be distinguished from "any rotation degrades descriptions toward
     caution-flavoured text", and prereg §5's outcome (b) is unfalsifiable:
       - matched-angle random rotations: N_RAND in-manifold random directions
         (differences of random real activation pairs) at every positive theta;
       - off-manifold probe: the raw direction injected alone (expected CJK junk,
         L7/L9) — documents manifold dependence instead of leaving it as an excuse;
       - endogenous reads: the 36 real `refused` activations + 36 matched
         `appropriate` at temp 0 (all previous per-scenario reads were temp-1.0
         stochastic samples, L9 §3).
  3. CONTRASTIVE READOUT. Everything is temperature=0, so for each (base,
     direction, theta) we log the DIVERGENCE POINT — the first word where the
     greedy decode differs from the same base's theta=0 decode — plus the
     divergent continuation. Localizes what the rotation changes.

Directions: v_deflect (refused - rest; primary — the strongest behavioural signal,
AUC 0.89, never verbalized), v_privacy (if results/v_privacy_f.npz exists),
v_eval (if results/eval_awareness_acts_f.npz exists and --include-eval),
diff_raw at theta=45 only (continuity with session 04).

Scoring (fixed in prereg amendment A1 BEFORE this runs): primary = slope over
theta of CI/caution lexicon rates for v_deflect RELATIVE to the matched-angle
random controls; secondary = blinded judge classification (separate script);
tertiary = text probe distinguishing v_deflect-injected from random-injected
descriptions at matched theta. Falsification: v_deflect ~ random controls on all
endpoints at all thetas => "not verbalizable through this channel" is the finding.

Setup: SGLang serving the AV (see run_nla.py docstring). ~140 calls, minutes.
Verify BEFORE launching SGLang:
    grep injection_token_id actor_hf/nla_meta.yaml    # must be 149705 (L9)
    python scripts/verbalize_directions_f.py --dry-run   # call plan + rotation table

Usage:
    tmux new -s verbalize
    python scripts/verbalize_directions_f.py [--include-eval] [--dry-run]

Outputs: results/verbalize_directions_f.csv, results/verbalize_directions_f.txt
"""

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

ACTOR_DIR    = Path('./actor_hf')
SGLANG_URL   = 'http://localhost:30000'
ACTIVATIONS  = Path('results/activations_layer20.npz')
VPRIV_PATH   = Path('results/v_privacy_f.npz')
EVAL_ACTS    = Path('results/eval_awareness_acts_f.npz')
OUT_CSV      = Path('results/verbalize_directions_f.csv')
OUT_TXT      = Path('results/verbalize_directions_f.txt')

THETAS       = [10, 20, 30, 45, 60]     # degrees; applied as +theta and -theta
INDIV_THETAS = [20, 45]                 # +/- on individual bases
N_INDIV      = 3                        # individual appropriate-base activations
N_RAND       = 5                        # matched-angle random directions per theta
N_ENDO       = 36                       # refused count; same n of matched appropriate
MAX_NEW      = 300
SEED         = 0
SAVE_EVERY   = 10
EXPECTED_INJECTION_TOKEN_ID = 149705    # L9 — hard requirement

# Lexicons are FIXED here and mirrored in prereg amendment A1. Do not edit after
# the run. scratch/07 note: description-level deflection signal is tonal, not
# lexical — a lexicon null must not be over-read (the text probe is the
# sensitive endpoint).
CI_TERMS = re.compile(
    r'\b(privacy|private|confidential\w*|confide\w*|secre\w*|disclos\w*|leak\w*|'
    r'betray\w*|sensitive information|not (?:mine|my place) to (?:share|tell)|'
    r'should not (?:be )?(?:shared|told|revealed)|breach|entrust\w*)\b',
    re.IGNORECASE)
CAUTION_TERMS = re.compile(
    r'\b(careful\w*|caution\w*|hedg\w*|tactful\w*|delicate|sensitive topic|'
    r'awkward|uncomfortab\w*|avoid\w*|deflect\w*|redirect\w*|evasive\w*|'
    r'decline\w*|refus\w*|hesitat\w*|reluctan\w*|guarded)\b', re.IGNORECASE)
STRUCTURE_TERMS = re.compile(
    r'\b(format|structured|dialogue|conversation|token|quote\w*|response|'
    r'character|speaker|prompt|chat)\b', re.IGNORECASE)


# ── Geometry ──────────────────────────────────────────────────────────────────

def rotation_deg(base, vec):
    cos = (vec @ base) / (np.linalg.norm(vec) * np.linalg.norm(base))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def alpha_for_angle(base, d, theta_deg, sign=+1):
    """Smallest |alpha| with sign `sign` such that base + alpha*d is rotated
    theta_deg from base. Bisection; returns None if unreachable (theta beyond
    the base->direction asymptote)."""
    lo, hi = 0.0, 1.0
    f = lambda a: rotation_deg(base, base + sign * a * d)
    # grow hi until past target or clearly saturated
    for _ in range(60):
        if f(hi) >= theta_deg:
            break
        hi *= 2.0
        if hi * np.linalg.norm(d) > 1e7:
            return None
    else:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) < theta_deg:
            lo = mid
        else:
            hi = mid
    return sign * hi


# ── Directions & bases ────────────────────────────────────────────────────────

def build_plan(include_eval=False):
    data = np.load(ACTIVATIONS, allow_pickle=True)
    t3 = data['tiers'] == 'tier_3'
    acts = data['activations'][t3].astype(np.float64)
    labels = data['labels'][t3]
    ids = data['scenario_ids'][t3]
    rng = np.random.default_rng(SEED)

    approp = acts[labels == 'appropriate']
    base_mean = approp.mean(0)
    indiv_pos = rng.choice(np.where(labels == 'appropriate')[0], N_INDIV,
                           replace=False)

    dirs = {}
    dirs['v_deflect'] = acts[labels == 'refused'].mean(0) - acts[labels != 'refused'].mean(0)
    dirs['diff_raw'] = acts[labels == 'leaked'].mean(0) - acts[labels != 'leaked'].mean(0)
    if VPRIV_PATH.exists():
        dirs['v_privacy'] = np.load(VPRIV_PATH, allow_pickle=True)['v_privacy'].astype(np.float64)
        print('v_privacy loaded — included.')
    else:
        print('NOTE: v_privacy_f.npz not found (run minimal_pairs_f.py --stage '
              'validate first) — skipping that direction.')
    if include_eval:
        if EVAL_ACTS.exists():
            ex = np.load(EVAL_ACTS, allow_pickle=True)
            keys = pd.Series(ex['keys'])
            side = np.where(keys.str.split('|').str[1].str.startswith('test'),
                            'test', 'deploy')
            ea = ex['acts'].astype(np.float64)
            dirs['v_eval'] = ea[side == 'test'].mean(0) - ea[side == 'deploy'].mean(0)
            print('v_eval loaded — included.')
        else:
            print('WARNING: --include-eval set but eval_awareness_acts_f.npz missing.')

    # matched-angle random in-manifold directions: differences of random real
    # activation pairs (lives in the same difference-vector family as the
    # behavioural directions), unit-normalized
    rand_dirs = []
    for _ in range(N_RAND):
        i, j = rng.choice(len(acts), 2, replace=False)
        r = acts[i] - acts[j]
        rand_dirs.append(r / np.linalg.norm(r))

    # endogenous read set: all refused + matched random appropriate
    ref_idx = np.where(labels == 'refused')[0]
    app_idx = rng.choice(np.where(labels == 'appropriate')[0],
                         min(N_ENDO, (labels == 'appropriate').sum()), replace=False)

    # ── assemble the call plan ───────────────────────────────────────────────
    calls = []   # dicts: name, vec, meta

    def add(name, vec, **meta):
        calls.append(dict(name=name, vec=np.asarray(vec, np.float32), meta=meta))

    # theta=0 baselines (divergence references)
    add('base_mean theta=0', base_mean,
        condition='baseline', base='mean', direction='none', theta=0)
    for j, i in enumerate(indiv_pos):
        add(f'base_indiv{j} theta=0', acts[i],
            condition='baseline', base=f'indiv{j}', direction='none', theta=0)

    swept = [d for d in ('v_deflect', 'v_privacy', 'v_eval') if d in dirs]
    for dname in swept:
        d = dirs[dname]
        thetas = THETAS if dname != 'v_eval' else INDIV_THETAS
        for th in thetas:
            for sign in (+1, -1):
                a = alpha_for_angle(base_mean, d, th, sign)
                if a is None:
                    print(f'  UNREACHABLE: {dname} {sign * th:+d} deg from mean base')
                    continue
                add(f'{dname} mean {sign * th:+d}deg', base_mean + a * d,
                    condition='swept', base='mean', direction=dname,
                    theta=sign * th, alpha=float(a))
        if dname == 'v_deflect':            # individual bases, primary dir only
            for j, i in enumerate(indiv_pos):
                for th in INDIV_THETAS:
                    for sign in (+1, -1):
                        a = alpha_for_angle(acts[i], d, th, sign)
                        if a is None:
                            continue
                        add(f'{dname} indiv{j} {sign * th:+d}deg', acts[i] + a * d,
                            condition='swept', base=f'indiv{j}', direction=dname,
                            theta=sign * th, alpha=float(a))

    # continuity row: diff_raw at 45 deg (session-04 lineage; large-alpha grid
    # killed per audit §5 — the direction is ~1/3 label noise, L8)
    a = alpha_for_angle(base_mean, dirs['diff_raw'], 45, +1)
    if a is not None:
        add('diff_raw mean +45deg', base_mean + a * dirs['diff_raw'],
            condition='swept', base='mean', direction='diff_raw', theta=45,
            alpha=float(a))

    # matched-angle random controls (mean base, positive thetas)
    for k, r in enumerate(rand_dirs):
        for th in THETAS:
            a = alpha_for_angle(base_mean, r, th, +1)
            if a is None:
                continue
            add(f'rand{k} mean +{th}deg', base_mean + a * r,
                condition='random_control', base='mean', direction=f'rand{k}',
                theta=th, alpha=float(a))

    # off-manifold probes: the raw directions alone (expected junk — documents
    # manifold dependence; makes prereg §5 outcome (b) falsifiable)
    for dname in swept:
        add(f'{dname} RAW (off-manifold)', dirs[dname],
            condition='off_manifold', base='none', direction=dname, theta=np.nan)

    # endogenous reads at temp 0
    for i in ref_idx:
        add(f'endo refused id{int(ids[i])}', acts[i],
            condition='endogenous', base=f'id{int(ids[i])}', direction='none',
            theta=np.nan, label='refused', scenario_id=int(ids[i]))
    for i in app_idx:
        add(f'endo approp id{int(ids[i])}', acts[i],
            condition='endogenous', base=f'id{int(ids[i])}', direction='none',
            theta=np.nan, label='appropriate', scenario_id=int(ids[i]))

    return calls, dirs, base_mean


# ── Readout helpers ───────────────────────────────────────────────────────────

def divergence(ref_words, words):
    """Index of first differing word vs the theta=0 reference + a snippet."""
    n = min(len(ref_words), len(words))
    for i in range(n):
        if ref_words[i] != words[i]:
            return i, ' '.join(words[i:i + 15])
    return (n, '') if len(ref_words) != len(words) else (-1, '')


def lexicon_hits(desc):
    return dict(ci_terms=len(CI_TERMS.findall(desc)),
                caution_terms=len(CAUTION_TERMS.findall(desc)),
                structure_terms=len(STRUCTURE_TERMS.findall(desc)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--include-eval', action='store_true',
                    help='also sweep v_eval (after E-EVAL extraction)')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the call plan + rotation table; no SGLang')
    args = ap.parse_args()

    calls, dirs, base_mean = build_plan(include_eval=args.include_eval)
    print(f'\ncall plan: {len(calls)} NLA calls')
    for dname, d in dirs.items():
        print(f'  ||{dname}|| = {np.linalg.norm(d):.3f}')
    print(f'  ||base_mean|| = {np.linalg.norm(base_mean):.3f}')
    plan = pd.DataFrame([dict(name=c['name'], **c['meta']) for c in calls])
    print(plan.groupby('condition').size().to_string())

    if args.dry_run:
        sw = plan[plan['condition'] == 'swept']
        print('\nrotation table (swept):')
        print(sw[['direction', 'base', 'theta', 'alpha']].to_string(index=False))
        print('\n[dry-run] no SGLang calls made.')
        return

    # ── environment gates (F7): fail loudly BEFORE burning session time ─────
    meta_yaml = (ACTOR_DIR / 'nla_meta.yaml').read_text()
    m = re.search(r'injection_token_id:\s*(\d+)', meta_yaml)
    assert m and int(m.group(1)) == EXPECTED_INJECTION_TOKEN_ID, (
        f'injection_token_id != {EXPECTED_INJECTION_TOKEN_ID} in '
        f'{ACTOR_DIR}/nla_meta.yaml (L9) — restore or re-download the checkpoint')

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from nla_inference import NLAClient
    import inspect
    sig = inspect.signature(NLAClient.generate)
    has_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                 for p in sig.parameters.values())
    assert has_kw or {'temperature', 'max_new_tokens'} <= set(sig.parameters), (
        f'NLAClient.generate does not accept temperature/max_new_tokens '
        f'(signature: {sig}) — fix the call before running')

    print(f'\nConnecting to SGLang at {SGLANG_URL}...')
    client = NLAClient(str(ACTOR_DIR), sglang_url=SGLANG_URL)
    print('Connected.\n')

    done = {}
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        done = {r['name']: r for _, r in prev.iterrows()}
        print(f'resuming: {len(done)} calls already recorded')

    rows = [dict(r) for r in done.values()]
    baseline_desc = {r['base']: str(r['description']).split()
                     for r in rows if r.get('condition') == 'baseline'}
    lines = []
    n_new = 0
    for c in calls:
        if c['name'] in done:
            continue
        desc = client.generate(c['vec'], temperature=0, max_new_tokens=MAX_NEW)
        row = dict(name=c['name'], **c['meta'],
                   l2=float(np.linalg.norm(c['vec'])), description=desc,
                   **lexicon_hits(desc))
        if c['meta'].get('condition') == 'baseline':
            baseline_desc[c['meta']['base']] = desc.split()
        ref = baseline_desc.get(c['meta'].get('base'))
        if ref is not None and c['meta'].get('condition') in ('swept', 'random_control'):
            di, snippet = divergence(ref, desc.split())
            row['diverge_word_idx'] = di
            row['diverge_snippet'] = snippet
        rows.append(row)
        lines.append(f"\n[{c['name']}]  CI={row['ci_terms']} caut={row['caution_terms']} "
                     f"struct={row['structure_terms']}\n  {desc}")
        print(lines[-1])
        n_new += 1
        if n_new % SAVE_EVERY == 0:
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    with open(OUT_TXT, 'w') as f:
        f.write('\n'.join(lines))
    print(f'\nSaved {OUT_CSV} ({len(rows)} rows) and {OUT_TXT}')

    df = pd.DataFrame(rows)
    sw = df[df['condition'].isin(['swept', 'random_control'])].copy()
    if len(sw):
        sw['is_control'] = sw['condition'] == 'random_control'
        print('\nmean lexicon hits by (direction-vs-control, theta):')
        print(sw.groupby(['is_control', 'theta'])[
            ['ci_terms', 'caution_terms', 'structure_terms']].mean().round(2).to_string())
    en = df[df['condition'] == 'endogenous']
    if len(en):
        print('\nendogenous reads, mean hits by label:')
        print(en.groupby('label')[['ci_terms', 'caution_terms',
                                   'structure_terms']].mean().round(2).to_string())


if __name__ == '__main__':
    main()
