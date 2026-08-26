"""
nla_transmission_f.py — does privacy information survive the NLA channel?

    real activation  ──AV──>  English description  ──AR──>  reconstructed activation
                                     │                              │
                          "can a reader tell which                "does the probe
                           came from the secret version?"          still work?"

Uses the 233 minimal pairs: the same story told twice, once where the
information is confidential and once where it is common knowledge. Everything
else is held fixed, so the pair difference isolates privacy status.

═══════════════════════════════════════════════════════════════════════════════
STAGE 1 IS A SCIENTIFIC GO/NO-GO, NOT A HEALTH CHECK. READ THIS FIRST.
═══════════════════════════════════════════════════════════════════════════════

The NLA is direction-only: it L2-renormalizes every input (L9), so **only the
angle between two activations matters**. Measured on our own pairs:

    secret vs its own public version : median  4.5 deg   (66% of pairs < 5 deg)
    secret vs a different story      : median 16.6 deg

L8 records that a **~5 deg rotation produced identical descriptions** in session
04 — that is exactly why `verbalize_directions_f.py` targets 10-60 deg. Our
minimal pairs sit *at or below* that threshold.

Worse, this regime is unexplored: across all 36,315 tier-3 scenario pairs, the
minimum angle between any two activations is ~8 deg. No existing description in
this repo was produced from inputs as similar as the ones we are about to feed
in, so the sensitivity cannot be extrapolated from what we have.

**If the AV writes the same paragraph for both versions, every downstream
readout is dead by construction** — a reader cannot discriminate identical text,
and the AR maps identical text to identical activations. One check gates the
whole experiment, so it runs first and costs ~5 minutes.

GO / NO-GO THRESHOLDS, FIXED BEFORE ANY DESCRIPTION IS GENERATED:

    NO-GO    >50% of pairs byte-identical, OR median word-Jaccard >= 0.95
    MARGINAL 20-50% byte-identical, OR median Jaccard 0.85-0.95
    GO       <20% byte-identical AND median Jaccard < 0.85

On NO-GO, stop and report it: "the AV's resolution is coarser than the privacy
manipulation" is an honest, publishable negative that costs one GPU hour instead
of a whole session. Do not retune prompts or thresholds to manufacture a GO.

The pilot also reports how much of each pair's description difference is
accounted for by the **edit vocabulary** — the specific words the rewrite
changed. High overlap is an early warning that any downstream signal is the AV
echoing the prompt rather than reading a privacy representation (the
confabulation failure mode, arXiv:2509.13316). This is diagnostic only; it does
not gate.

Setup (Lambda, in tmux). Do NOT `pip install sglang` during a paid session — it
silently upgraded torch and cost 20 minutes in session 13.
    grep injection_token_id actor_hf/nla_meta.yaml     # must be 149705 (L9)
    python -m sglang.launch_server --model-path ./actor_hf --disable-radix-cache \
        --mem-fraction-static 0.85 --trust-remote-code --port 30000

Usage:
    python scripts/nla_transmission_f.py --stage pilot            # 20 pairs, ~5 min
    python scripts/nla_transmission_f.py --stage pilot --n 40
    python scripts/nla_transmission_f.py --stage verbalize        # all 233, after GO
"""

import os
import re
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Config ────────────────────────────────────────────────────────────────────

ACTOR_DIR   = Path('./actor_hf')
SGLANG_URL  = 'http://localhost:30000'
PAIR_ACTS   = Path('results/minimal_pairs_acts_f.npz')
PAIRS_CSV   = Path('data/minimal_pairs_f.csv')
OUT_CSV     = Path('results/nla_transmission_descriptions_f.csv')
PILOT_TXT   = Path('results/nla_transmission_pilot_f.txt')
MAX_NEW     = 300
SAVE_EVERY  = 10
SEED        = 0
EXPECTED_INJECTION_TOKEN_ID = 149705      # L9 — hard requirement

# Pre-registered go/no-go bounds. Do not edit after the pilot runs.
NOGO_IDENTICAL_FRAC = 0.50
NOGO_JACCARD        = 0.95
GO_IDENTICAL_FRAC   = 0.20
GO_JACCARD          = 0.85


def preflight():
    """Fail loudly before burning session time, not at call 1."""
    meta = (ACTOR_DIR / 'nla_meta.yaml')
    assert meta.exists(), f'{meta} missing — download the AV checkpoint first'
    m = re.search(r'injection_token_id:\s*(\d+)', meta.read_text())
    assert m and int(m.group(1)) == EXPECTED_INJECTION_TOKEN_ID, (
        f'injection_token_id != {EXPECTED_INJECTION_TOKEN_ID} in {meta} (L9) — '
        f'restore it or re-download the checkpoint')
    from nla_inference import NLAClient
    import inspect
    sig = inspect.signature(NLAClient.generate)
    ok = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()) \
        or {'temperature', 'max_new_tokens'} <= set(sig.parameters)
    assert ok, f'NLAClient.generate lacks temperature/max_new_tokens: {sig}'
    env = {}
    try:
        import torch, transformers
        env = dict(torch=torch.__version__, transformers=transformers.__version__)
    except Exception:
        pass
    print(f'preflight OK — injection_token_id={EXPECTED_INJECTION_TOKEN_ID}, '
          f'NLAClient.generate{sig}, {env}')
    return NLAClient


# ── Description handling ──────────────────────────────────────────────────────

def split_paragraphs(text):
    return [p.strip() for p in str(text).split('\n\n') if p.strip()]


def is_valid(text):
    """A row counts as done only if it parsed as a real 3-paragraph description.
    Failures stay pending and retry (the resume bug found in session 14)."""
    return bool(str(text).strip()) and len(split_paragraphs(text)) == 3


WORD = re.compile(r"[a-z']+")


def toks(t):
    return set(WORD.findall(str(t).lower()))


def jaccard(a, b):
    A, B = toks(a), toks(b)
    return len(A & B) / max(len(A | B), 1)


def edit_vocab(secret_story, public_story):
    """Words the rewrite actually changed — the prompt-echo suspects."""
    S, P = toks(secret_story), toks(public_story)
    return (S - P) | (P - S)


def word_diff(a, b, ctx=3):
    import difflib
    aw, bw = str(a).split(), str(b).split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, aw, bw).get_opcodes():
        if tag == 'equal':
            seg = aw[i1:i2]
            out.append(' '.join(seg) if len(seg) <= 2 * ctx
                       else ' '.join(seg[:ctx]) + ' … ' + ' '.join(seg[-ctx:]))
        elif tag == 'delete':
            out.append('[-' + ' '.join(aw[i1:i2]) + '-]')
        elif tag == 'insert':
            out.append('[+' + ' '.join(bw[j1:j2]) + '+]')
        else:
            out.append('[-' + ' '.join(aw[i1:i2]) + '-] [+' + ' '.join(bw[j1:j2]) + '+]')
    return ' '.join(out)


# ── Generation ────────────────────────────────────────────────────────────────

def load_pairs(n=None):
    ex = np.load(PAIR_ACTS, allow_pickle=True)
    ids = ex['scenario_ids']
    S, P = ex['acts_secret'].astype(np.float32), ex['acts_public'].astype(np.float32)
    pairs = pd.read_csv(PAIRS_CSV)
    pairs = pairs[pairs['valid']].set_index('scenario_id') if 'valid' in pairs else pairs.set_index('scenario_id')
    if n is not None and n < len(ids):
        sel = np.random.default_rng(SEED).choice(len(ids), n, replace=False)
        sel.sort()
        ids, S, P = ids[sel], S[sel], P[sel]
    return ids, S, P, pairs


def generate(client, ids, S, P, out_csv):
    done = {}
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        # only VALID rows count as done — errors and malformed rows retry
        for _, r in prev.iterrows():
            if is_valid(r.get('description', '')):
                done[(int(r['scenario_id']), r['variant'])] = dict(r)
    rows = list(done.values())
    todo = [(int(i), v) for i in ids for v in ('secret', 'public')
            if (int(i), v) not in done]
    if done:
        print(f'resuming: {len(done)} valid descriptions already stored')
    print(f'{len(todo)} to generate')

    vec = {('secret', int(i)): s for i, s in zip(ids, S)}
    vec.update({('public', int(i)): p for i, p in zip(ids, P)})

    n_new = 0
    for sid, var in tqdm(todo, desc='AV'):
        try:
            d = client.generate(vec[(var, sid)], temperature=0, max_new_tokens=MAX_NEW)
            err = ''
        except Exception as e:
            d, err = '', str(e)[:200]
        rows.append(dict(scenario_id=sid, variant=var, description=d,
                         n_paragraphs=len(split_paragraphs(d)),
                         valid=is_valid(d), error=err))
        n_new += 1
        if n_new % SAVE_EVERY == 0:
            pd.DataFrame(rows).to_csv(out_csv, index=False)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pd.DataFrame(rows)


# ── Stage: pilot ──────────────────────────────────────────────────────────────

def stage_pilot(n):
    NLAClient = preflight()
    ids, S, P, pairs = load_pairs(n)
    print(f'\npilot on {len(ids)} pairs (seed {SEED}): {[int(i) for i in ids]}\n')

    # report the geometry we are about to test the AV against
    ang = np.degrees(np.arccos(np.clip(
        np.sum(S * P, axis=1) / (np.linalg.norm(S, axis=1) * np.linalg.norm(P, axis=1)), -1, 1)))
    print(f'within-pair angle: median {np.median(ang):.2f} deg, '
          f'min {ang.min():.2f}, max {ang.max():.2f}')
    print(f'(L8: a ~5 deg rotation produced identical descriptions in session 04)\n')

    client = NLAClient(str(ACTOR_DIR), sglang_url=SGLANG_URL)
    df = generate(client, ids, S, P, OUT_CSV)

    piv = df[df.valid].pivot_table(index='scenario_id', columns='variant',
                                   values='description', aggfunc='first').dropna()
    if not len(piv):
        print('\nNO VALID DESCRIPTIONS — check SGLang and the checkpoint.')
        return

    lines, stats = [], []
    for sid, r in piv.iterrows():
        s, p = r['secret'], r['public']
        ev = edit_vocab(pairs.loc[sid, 'story_secret'], pairs.loc[sid, 'story_public']) \
            if sid in pairs.index else set()
        diff_words = toks(s) ^ toks(p)
        stats.append(dict(scenario_id=sid, identical=(s.strip() == p.strip()),
                          jaccard=jaccard(s, p), n_diff_words=len(diff_words),
                          frac_diff_from_edit=len(diff_words & ev) / max(len(diff_words), 1)))
        lines.append('=' * 78)
        lines.append(f'scenario {sid}   jaccard={jaccard(s,p):.3f}   '
                     f'identical={s.strip()==p.strip()}')
        sp, pp = split_paragraphs(s), split_paragraphs(p)
        for k in range(3):
            lines.append(f'\n  --- paragraph {k+1} ---')
            lines.append(f'  {word_diff(sp[k], pp[k])}')
        lines.append('')

    st = pd.DataFrame(stats)
    PILOT_TXT.write_text('\n'.join(lines))
    frac_id, med_j = st.identical.mean(), st.jaccard.median()

    print('\n' + '=' * 78)
    print(f'PILOT VERDICT  (thresholds fixed before the run — see docstring)')
    print('=' * 78)
    print(f'  pairs compared            : {len(st)}')
    print(f'  byte-identical            : {st.identical.sum()}/{len(st)} ({frac_id:.0%})')
    print(f'  word-Jaccard(secret,public): median {med_j:.3f}  '
          f'[{st.jaccard.min():.3f}, {st.jaccard.max():.3f}]')
    print(f'  differing words per pair  : median {st.n_diff_words.median():.0f}')
    print(f'  of those, from edit vocab : median {st.frac_diff_from_edit.median():.0%}  '
          f'[diagnostic only — high = likely prompt echo]')
    print()
    if frac_id > NOGO_IDENTICAL_FRAC or med_j >= NOGO_JACCARD:
        print('  => NO-GO. The AV does not resolve the privacy manipulation.')
        print('     Report as a negative: "the channel\'s resolution is coarser than')
        print('     the manipulation." Do NOT retune to manufacture a GO.')
    elif frac_id < GO_IDENTICAL_FRAC and med_j < GO_JACCARD:
        print('  => GO. Descriptions differ. Proceed to --stage verbalize.')
        print('     Note the edit-vocab figure above: the ablation will need to')
        print('     separate real transmission from prompt echo.')
    else:
        print('  => MARGINAL. Read the side-by-side output before deciding.')
    print(f'\n  side-by-side diffs written to {PILOT_TXT}')
    print('  READ THAT FILE before continuing — the numbers do not capture whether')
    print('  the differences are *about privacy* or incidental.')


def stage_verbalize():
    NLAClient = preflight()
    ids, S, P, _ = load_pairs(None)
    print(f'full run: {len(ids)} pairs = {2*len(ids)} descriptions')
    client = NLAClient(str(ACTOR_DIR), sglang_url=SGLANG_URL)
    df = generate(client, ids, S, P, OUT_CSV)
    ok = df.valid.sum()
    print(f'\nvalid: {ok}/{len(df)}   invalid/error: {(~df.valid).sum()}')
    if (~df.valid).any():
        print('  re-run this stage to retry the failures (they are not counted done)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['pilot', 'verbalize'], required=True)
    ap.add_argument('--n', type=int, default=20, help='pilot pairs (default 20)')
    a = ap.parse_args()
    if a.stage == 'pilot':
        stage_pilot(a.n)
    else:
        stage_verbalize()
