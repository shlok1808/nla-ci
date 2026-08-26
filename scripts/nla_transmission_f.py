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
WHY A PILOT GATES THIS
═══════════════════════════════════════════════════════════════════════════════

The AV is direction-only: it L2-renormalizes every input (L9), so the AV sees
only each activation's unit direction. Measured on our own pairs:

    secret vs its own public version : median  4.45 deg   (66% below 5 deg)
    secret vs a DIFFERENT story      : median 16.62 deg

L8 records that a ~5 deg rotation produced identical descriptions in session 04.
That regime is also unexplored here: across all 36,315 tier-3 scenario pairs the
minimum angle is ~8 deg, so nothing in this repo was ever generated from inputs
this similar.

**Angle is normalized-input distance, NOT a resolution threshold.** The AV is
nonlinear and its sensitivity is anisotropic: a small step along a
high-sensitivity semantic direction can change the output while a larger step
along an insensitive one does not. Session 04's null used a synthetic,
partly off-manifold perturbation along a direction that is ~1/3 label noise;
both members of a minimal pair are natural prompt activations. So session 04
makes the pilot worth running — it does not predict its outcome. There is no
activation-only test that settles this without generating text.

═══════════════════════════════════════════════════════════════════════════════
TWO VERDICTS, NOT ONE
═══════════════════════════════════════════════════════════════════════════════

The forecast paragraph (P3) is the scientifically interesting channel; the setup
paragraph (P2) is expected to echo the rewrite's changed words. **A
whole-description difference driven entirely by P2 would be a false GO** — the
number moves while the hypothesis of interest is dead. So P3 carries its own
verdict and is primary.

    FULL branch : does any human-visible private/public information survive?
    P3   branch : does it reach the AV's forecast of the upcoming reply?

FULL success with P3 failure is a distinct, reportable result, not a NO-GO.

Human time is spent only when it can matter. If most P3 pairs are byte-identical
the discrimination ceiling is already below anything interesting and the pilot
stops without asking anyone to judge anything:

    2AFC ceiling = 1 - f_identical/2

(a perfect reader still scores chance on identical pairs). Pre-registered
minimum interesting effect: **0.65**.

Stages:
    --stage plan         lock a 40-pair manifest, stratified by angle quartile   [local]
    --stage pilot        generate + determinism control + metrics + verdicts     [GPU]
    --stage export-2afc  blinded A/B packet, only if the ceiling allows          [local]
    --stage score-2afc   exact binomial on the returned answers                  [local]
    --stage verbalize    all 233 pairs — refuses to run without a recorded GO    [GPU]

Setup (Lambda, in tmux). Do NOT `pip install sglang` during a paid session — it
silently upgraded torch and cost 20 minutes in session 13.
    grep injection_token_id actor_hf/nla_meta.yaml     # must be 149705 (L9)
    python -m sglang.launch_server --model-path ./actor_hf --disable-radix-cache \
        --mem-fraction-static 0.85 --trust-remote-code --port 30000
"""

import os
import re
import sys
import json
import difflib
import argparse
import platform
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Config ────────────────────────────────────────────────────────────────────

ACTOR_DIR    = Path('./actor_hf')
SGLANG_URL   = 'http://localhost:30000'
PAIR_ACTS    = Path('results/minimal_pairs_acts_f.npz')
PAIRS_CSV    = Path('data/minimal_pairs_f.csv')

MANIFEST     = Path('results/nla_transmission_manifest_f.csv')
PILOT_CSV    = Path('results/nla_transmission_pilot_desc_f.csv')   # pilot only
FULL_CSV     = Path('results/nla_transmission_full_desc_f.csv')    # full run only
VERDICT_JSON = Path('results/nla_transmission_pilot_verdict_f.json')
PILOT_TXT    = Path('results/nla_transmission_pilot_f.txt')
METRICS_CSV  = Path('results/nla_transmission_pilot_metrics_f.csv')
PACKET_CSV   = Path('data/nla_transmission_2afc_packet_f.csv')
KEY_CSV      = Path('results/nla_transmission_2afc_key_f.csv')

PILOT_N      = 40          # 10 per within-pair angle quartile
MAX_NEW      = 300
SAVE_EVERY   = 10
SEED         = 0
N_DETERMINISM = 5          # activations regenerated to prove greedy is greedy
MAX_CONSEC_FAIL = 3        # abort rather than burn the session on a dead server
EXPECTED_INJECTION_TOKEN_ID = 149705                                # L9

# Pre-registered. Do not edit after the pilot runs.
MIN_INTERESTING_2AFC = 0.65    # smallest effect worth continuing for
PILOT_2AFC_PASS      = 26      # /40 — first one-sided exact-binomial p<.05


# ── Text handling ─────────────────────────────────────────────────────────────

PARA_SPLIT = re.compile(r'\n\s*\n')
WORD = re.compile(r"[a-z']+")


def split_paragraphs(text):
    return [p.strip() for p in PARA_SPLIT.split(str(text).strip()) if p.strip()]


def _blank(x):
    """CSV round-trips empty strings as NaN; treat both as absent. Without this,
    every resumed row looks like a failure because str(nan) == 'nan'."""
    return x is None or (isinstance(x, float) and np.isnan(x)) or not str(x).strip()


def text_valid(text, error=''):
    """Generation succeeded and returned something. Errors retry."""
    return _blank(error) and not _blank(text)


def paragraph_valid(text):
    """Additionally parsed into the expected 3 paragraphs. A deterministic
    2- or 4-paragraph output is RECORDED, not retried forever — at temperature 0
    a retry reproduces it exactly."""
    return len(split_paragraphs(text)) == 3


def toks(t):
    return WORD.findall(str(t).lower())


def norm_text(t):
    return re.sub(r'\s+', ' ', str(t).strip())


def seq_similarity(a, b):
    """Token-sequence similarity — keeps order and repetition, unlike Jaccard."""
    return difflib.SequenceMatcher(None, toks(a), toks(b)).ratio()


def jaccard(a, b):
    A, B = set(toks(a)), set(toks(b))
    return len(A & B) / max(len(A | B), 1)


def n_token_edits(a, b):
    sm = difflib.SequenceMatcher(None, toks(a), toks(b))
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes()
               if tag != 'equal')


def edit_vocab(secret_story, public_story):
    S, P = set(toks(secret_story)), set(toks(public_story))
    return (S - P) | (P - S)


def word_diff(a, b, ctx=3):
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


def atomic_write_csv(df, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_csv(tmp, index=False)
    with open(tmp, 'rb') as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ── Inputs ────────────────────────────────────────────────────────────────────

def load_acts():
    ex = np.load(PAIR_ACTS, allow_pickle=True)
    for k in ('scenario_ids', 'acts_secret', 'acts_public'):
        assert k in ex, f'{PAIR_ACTS} missing key {k}'
    ids = ex['scenario_ids'].astype(int)
    S, P = ex['acts_secret'].astype(np.float32), ex['acts_public'].astype(np.float32)
    assert len(ids) == len(S) == len(P), 'npz array lengths disagree'
    assert len(set(ids)) == len(ids), 'duplicate scenario_ids in npz'
    assert S.shape[1] == P.shape[1] == 3584, f'unexpected dim {S.shape}'
    assert np.isfinite(S).all() and np.isfinite(P).all(), 'non-finite activations'
    assert (np.linalg.norm(S, axis=1) > 0).all() and (np.linalg.norm(P, axis=1) > 0).all()
    pairs = pd.read_csv(PAIRS_CSV)
    if 'valid' in pairs:
        pairs = pairs[pairs['valid']]
    pairs = pairs.set_index('scenario_id')
    assert set(ids) <= set(pairs.index), 'npz ids not all present in the valid pairs CSV'
    return ids, S, P, pairs


def pair_angles(S, P):
    cos = np.sum(S * P, axis=1) / (np.linalg.norm(S, axis=1) * np.linalg.norm(P, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1))), cos


# ── Stage: plan ───────────────────────────────────────────────────────────────

def stage_plan(n=PILOT_N):
    """Lock the pilot sample BEFORE any description exists. Stratified by
    within-pair angle quartile so a NO-GO cannot be an artifact of having
    sampled only near-identical inputs, and so the verdict can be read by
    quartile (uniform insensitivity vs a threshold response)."""
    ids, S, P, pairs = load_acts()
    ang, cos = pair_angles(S, P)
    q = pd.qcut(ang, 4, labels=[1, 2, 3, 4])
    rng = np.random.default_rng(SEED)
    sel = []
    per = n // 4
    for qi in [1, 2, 3, 4]:
        pool = np.where(q == qi)[0]
        sel.extend(rng.choice(pool, min(per, len(pool)), replace=False))
    sel = np.sort(np.array(sel))
    man = pd.DataFrame(dict(
        scenario_id=ids[sel], angle_deg=ang[sel].round(4), cosine=cos[sel].round(6),
        angle_quartile=[int(q[i]) for i in sel],
        act_checksum_secret=[int(abs(hash(S[i].tobytes())) % 10**12) for i in sel],
        act_checksum_public=[int(abs(hash(P[i].tobytes())) % 10**12) for i in sel]))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(man, MANIFEST)
    print(f'locked {len(man)} pairs -> {MANIFEST}')
    print(man.groupby('angle_quartile')['angle_deg']
          .agg(['count', 'min', 'median', 'max']).round(2).to_string())
    print(f'\nfull-population angle: median {np.median(ang):.2f} deg, '
          f'range [{ang.min():.2f}, {ang.max():.2f}]')


# ── Generation ────────────────────────────────────────────────────────────────

def preflight():
    meta = ACTOR_DIR / 'nla_meta.yaml'
    assert meta.exists(), f'{meta} missing — download the AV checkpoint'
    txt = meta.read_text()
    m = re.search(r'injection_token_id:\s*(\d+)', txt)
    assert m and int(m.group(1)) == EXPECTED_INJECTION_TOKEN_ID, (
        f'injection_token_id != {EXPECTED_INJECTION_TOKEN_ID} (L9)')
    sc = re.search(r'injection_scale:\s*([\d.]+)', txt)
    from nla_inference import NLAClient
    import inspect
    sig = inspect.signature(NLAClient.generate)
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()) \
        or {'temperature', 'max_new_tokens'} <= set(sig.parameters), \
        f'NLAClient.generate lacks temperature/max_new_tokens: {sig}'
    prov = dict(injection_token_id=EXPECTED_INJECTION_TOKEN_ID,
                injection_scale=(sc.group(1) if sc else 'unknown'),
                python=platform.python_version(), temperature=0, max_new_tokens=MAX_NEW)
    for mod in ('torch', 'transformers', 'sglang'):
        try:
            prov[mod] = __import__(mod).__version__
        except Exception:
            prov[mod] = 'absent'
    print(f'preflight OK: {prov}')
    return NLAClient, prov


def generate(client, work, out_csv, prov, manifest_ids):
    """work: list of (scenario_id, variant, vector, repeat_idx)."""
    done = {}
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        dup = prev.duplicated(['scenario_id', 'variant', 'repeat'], keep=False)
        if dup.any():
            conflicting = prev[dup].groupby(['scenario_id', 'variant', 'repeat'])[
                'description'].nunique().max()
            assert conflicting == 1, ('conflicting duplicate rows in '
                                      f'{out_csv} — delete it and re-run')
            prev = prev.drop_duplicates(['scenario_id', 'variant', 'repeat'], keep='last')
        # ONLY rows belonging to this manifest, and only successful generations
        prev = prev[prev.scenario_id.isin(manifest_ids)]
        for _, r in prev.iterrows():
            if text_valid(r.get('description', ''), r.get('error', '')):
                if str(r.get('prov', '')) and str(r['prov']) != json.dumps(prov, sort_keys=True):
                    raise SystemExit(
                        f'{out_csv} was written under a different environment.\n'
                        f'  stored: {r["prov"]}\n  now:    {json.dumps(prov, sort_keys=True)}\n'
                        f'Delete the file and re-run rather than mixing runs.')
                done[(int(r['scenario_id']), r['variant'], int(r['repeat']))] = dict(r)
    rows = list(done.values())
    todo = [w for w in work if (w[0], w[1], w[3]) not in done]
    if done:
        print(f'resuming: {len(done)} descriptions already valid')
    print(f'{len(todo)} to generate')
    if not todo:
        return pd.DataFrame(rows)

    # one uncaught smoke call: a dead or wrong server should fail now, not after 80 timeouts
    sid0, var0, vec0, rep0 = todo[0]
    smoke = client.generate(vec0, temperature=0, max_new_tokens=MAX_NEW)
    assert str(smoke).strip(), 'smoke generation returned empty text'
    print(f'smoke OK ({len(toks(smoke))} tokens)')
    rows.append(dict(scenario_id=sid0, variant=var0, repeat=rep0, description=smoke,
                     error='', prov=json.dumps(prov, sort_keys=True)))
    todo = todo[1:]

    consec = 0
    for i, (sid, var, vec, rep) in enumerate(tqdm(todo, desc='AV'), 1):
        try:
            d = client.generate(vec, temperature=0, max_new_tokens=MAX_NEW)
            err = ''
            consec = 0 if str(d).strip() else consec + 1
        except Exception as e:
            d, err = '', str(e)[:200]
            consec += 1
        rows.append(dict(scenario_id=sid, variant=var, repeat=rep, description=d,
                         error=err, prov=json.dumps(prov, sort_keys=True)))
        if consec >= MAX_CONSEC_FAIL:
            atomic_write_csv(pd.DataFrame(rows), out_csv)
            raise SystemExit(f'aborting: {consec} consecutive failures — last error: {err!r}')
        if i % SAVE_EVERY == 0:
            atomic_write_csv(pd.DataFrame(rows), out_csv)
    atomic_write_csv(pd.DataFrame(rows), out_csv)
    return pd.DataFrame(rows)


# ── Stage: pilot ──────────────────────────────────────────────────────────────

SLICES = ['full', 'P1', 'P2', 'P3']


def slice_text(desc, which):
    if which == 'full':
        return desc
    ps = split_paragraphs(desc)
    i = int(which[1]) - 1
    return ps[i] if len(ps) > i else ''


def stage_pilot():
    assert MANIFEST.exists(), 'run --stage plan first (it locks the sample)'
    man = pd.read_csv(MANIFEST)
    ids_m = set(man.scenario_id.astype(int))
    NLAClient, prov = preflight()
    ids, S, P, pairs = load_acts()
    idx = {int(s): i for i, s in enumerate(ids)}

    work = []
    for sid in man.scenario_id.astype(int):
        work.append((sid, 'secret', S[idx[sid]], 0))
        work.append((sid, 'public', P[idx[sid]], 0))
    # determinism control: regenerate the first N_DETERMINISM secret activations
    for sid in list(man.scenario_id.astype(int))[:N_DETERMINISM]:
        work.append((sid, 'secret', S[idx[sid]], 1))

    client = NLAClient(str(ACTOR_DIR), sglang_url=SGLANG_URL)
    df = generate(client, work, PILOT_CSV, prov, ids_m)

    # ── INVALID checks ───────────────────────────────────────────────────────
    df['ok'] = [text_valid(d, e) for d, e in zip(df.description, df.error)]
    rep = df[df.repeat == 1]
    nondet = []
    for _, r in rep.iterrows():
        base = df[(df.scenario_id == r.scenario_id) & (df.variant == r.variant)
                  & (df.repeat == 0)]
        if len(base) and norm_text(base.iloc[0].description) != norm_text(r.description):
            nondet.append(int(r.scenario_id))
    base = df[(df.repeat == 0) & df.ok]
    piv = base.pivot_table(index='scenario_id', columns='variant',
                           values='description', aggfunc='first')
    complete = piv.dropna()
    n_expected = len(man)

    verdict = dict(n_expected=n_expected, n_complete=int(len(complete)),
                   nondeterministic=nondet)
    print('\n' + '=' * 78)
    if nondet:
        verdict['status'] = 'INVALID'
        print(f'INVALID — greedy decoding was not deterministic for {len(nondet)} '
              f'activations: {nondet}')
        print('Every similarity metric below would be measuring decoder noise.')
        VERDICT_JSON.write_text(json.dumps(verdict, indent=2)); return
    if len(complete) < n_expected:
        verdict['status'] = 'INVALID'
        print(f'INVALID — {len(complete)}/{n_expected} pairs complete. A verdict '
              f'from a partial subset is not a result; re-run to retry failures.')
        VERDICT_JSON.write_text(json.dumps(verdict, indent=2)); return
    print(f'pilot valid: {len(complete)}/{n_expected} pairs, greedy decoding '
          f'deterministic on {len(rep)} repeats')
    print('=' * 78)

    # ── metrics per slice ────────────────────────────────────────────────────
    rows, lines = [], []
    qmap = dict(zip(man.scenario_id.astype(int), man.angle_quartile))
    amap = dict(zip(man.scenario_id.astype(int), man.angle_deg))
    for sid, r in complete.iterrows():
        s, p = r['secret'], r['public']
        ev = edit_vocab(pairs.loc[sid, 'story_secret'], pairs.loc[sid, 'story_public'])
        rec = dict(scenario_id=int(sid), quartile=qmap[int(sid)], angle=amap[int(sid)],
                   both_3para=paragraph_valid(s) and paragraph_valid(p))
        for w in SLICES:
            a, b = slice_text(s, w), slice_text(p, w)
            diff = set(toks(a)) ^ set(toks(b))
            rec[f'{w}_identical'] = (a == b)
            rec[f'{w}_norm_identical'] = (norm_text(a) == norm_text(b))
            rec[f'{w}_seqsim'] = round(seq_similarity(a, b), 4)
            rec[f'{w}_jaccard'] = round(jaccard(a, b), 4)
            rec[f'{w}_n_edits'] = n_token_edits(a, b)
            rec[f'{w}_edit_vocab_frac'] = round(len(diff & ev) / max(len(diff), 1), 4)
        rows.append(rec)
        lines.append('=' * 78)
        lines.append(f'scenario {sid}  angle={amap[int(sid)]:.2f}deg  q{qmap[int(sid)]}  '
                     f'P3 identical={rec["P3_norm_identical"]}')
        for k, (a, b) in enumerate(zip(split_paragraphs(s), split_paragraphs(p)), 1):
            lines.append(f'\n  --- P{k} ---')
            lines.append(f'  {word_diff(a, b)}')
        lines.append('')
    M = pd.DataFrame(rows)
    PILOT_TXT.write_text('\n'.join(lines))

    print(f'\n{"slice":>6} | {"identical":>10} | {"seq-sim":>8} | {"edits":>6} | '
          f'{"from edit-vocab":>15} | {"2AFC ceiling":>12}')
    print('-' * 78)
    for w in SLICES:
        f_id = M[f'{w}_norm_identical'].mean()
        ceil = 1 - f_id / 2
        print(f'{w:>6} | {f_id:>9.0%} | {M[f"{w}_seqsim"].median():>8.3f} | '
              f'{M[f"{w}_n_edits"].median():>6.0f} | '
              f'{M[f"{w}_edit_vocab_frac"].median():>14.0%} | {ceil:>12.3f}')
        verdict[w] = dict(frac_identical=round(f_id, 4),
                          median_seqsim=round(M[f'{w}_seqsim'].median(), 4),
                          ceiling_2afc=round(ceil, 4))

    print(f'\nby within-pair angle quartile (P3):')
    print(M.groupby('quartile').agg(angle=('angle', 'median'),
                                    P3_identical=('P3_norm_identical', 'mean'),
                                    P3_seqsim=('P3_seqsim', 'median')).round(3).to_string())

    # ── branch verdicts ──────────────────────────────────────────────────────
    print('\n' + '=' * 78)
    for w, label in [('full', 'FULL branch (any survival)'), ('P3', 'P3 branch (PRIMARY)')]:
        ceil = verdict[w]['ceiling_2afc']
        if ceil < MIN_INTERESTING_2AFC:
            st = 'NO-GO'
            why = (f'ceiling {ceil:.3f} < {MIN_INTERESTING_2AFC} — too many identical '
                   f'pairs for a reader to reach an interesting effect')
        else:
            st = 'PROCEED-TO-2AFC'
            why = f'ceiling {ceil:.3f} allows an interesting effect; discrimination untested'
        verdict[w]['status'] = st
        print(f'{label:<28} {st:<18} {why}')
    print('=' * 78)
    print('\nNOTE: these are TEXTUAL-SENSITIVITY diagnostics. Text differing does not')
    print('mean privacy information survived — the difference may be incidental, or')
    print('the AV echoing the rewrite\'s own changed words (see edit-vocab column).')
    print('The scientific gate is the blinded 2AFC: --stage export-2afc')
    print(f'\nside-by-side diffs: {PILOT_TXT}')
    verdict['status'] = 'PILOT-COMPLETE'
    VERDICT_JSON.write_text(json.dumps(verdict, indent=2))
    atomic_write_csv(M, METRICS_CSV)


# ── Stage: export / score the blinded 2AFC ───────────────────────────────────

def stage_export_2afc(which='P3'):
    v = json.loads(VERDICT_JSON.read_text())
    assert v.get('status') == 'PILOT-COMPLETE', 'run --stage pilot first'
    if v[which]['status'] == 'NO-GO':
        print(f'{which} branch is NO-GO (ceiling {v[which]["ceiling_2afc"]:.3f}). '
              f'Not exporting — human time cannot change this.')
        return
    df = pd.read_csv(PILOT_CSV)
    df = df[(df.repeat == 0) & [text_valid(d, e) for d, e in zip(df.description, df.error)]]
    piv = df.pivot_table(index='scenario_id', columns='variant',
                         values='description', aggfunc='first').dropna()
    rng = np.random.default_rng(SEED + 1)
    pk, key = [], []
    for i, (sid, r) in enumerate(piv.iterrows()):
        s, p = slice_text(r['secret'], which), slice_text(r['public'], which)
        if norm_text(s) == norm_text(p):
            continue                      # identical: nothing to judge
        secret_left = bool(rng.integers(2))
        pk.append(dict(item=len(pk) + 1, A=s if secret_left else p,
                       B=p if secret_left else s, your_answer=''))
        key.append(dict(item=len(pk), scenario_id=int(sid),
                        secret_side='A' if secret_left else 'B'))
    PACKET_CSV.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(pd.DataFrame(pk), PACKET_CSV)
    atomic_write_csv(pd.DataFrame(key), KEY_CSV)
    print(f'wrote {len(pk)} blinded items -> {PACKET_CSV}')
    print(f'answer key (do not open) -> {KEY_CSV}')
    print('\nFor each item, one description came from the activation where the')
    print('information was CONFIDENTIAL and one where it was COMMON KNOWLEDGE.')
    print('Fill `your_answer` with A or B. Do not skip items; guess if unsure.')
    print(f'Then: python scripts/nla_transmission_f.py --stage score-2afc')


def stage_score_2afc():
    from scipy.stats import binomtest
    pk = pd.read_csv(PACKET_CSV); key = pd.read_csv(KEY_CSV)
    m = pk.merge(key, on='item')
    ans = m[m.your_answer.astype(str).str.upper().isin(['A', 'B'])]
    if not len(ans):
        print(f'no answers filled in {PACKET_CSV}'); return
    correct = (ans.your_answer.str.upper() == ans.secret_side).sum()
    n = len(ans)
    bt = binomtest(int(correct), n, 0.5, alternative='greater')
    lo, hi = bt.proportion_ci(0.95)
    print(f'blinded 2AFC: {correct}/{n} = {correct/n:.3f}')
    print(f'  95% CI [{lo:.3f}, {hi:.3f}]   one-sided exact binomial p={bt.pvalue:.4f}')
    print(f'  unanswered: {len(m)-n}')
    if bt.pvalue < 0.05 and correct / n >= MIN_INTERESTING_2AFC:
        print(f'\n  => GO. Privacy status is distinguishable from the descriptions '
              f'alone.\n     Proceed to --stage verbalize (all 233 pairs).')
    elif bt.pvalue < 0.05:
        print(f'\n  => SIGNIFICANT BUT SMALL (below the pre-registered {MIN_INTERESTING_2AFC} '
              f'floor).\n     Report the effect; do not scale up on this alone.')
    else:
        print(f'\n  => NO-GO. Scoped result: "greedy AV descriptions did not let a '
              f'blinded\n     reader distinguish these natural private/public activation '
              f'pairs\n     under the tested checkpoint and decoding configuration."')


def stage_verbalize():
    v = json.loads(VERDICT_JSON.read_text()) if VERDICT_JSON.exists() else {}
    assert v.get('status') == 'PILOT-COMPLETE', (
        'refusing to run: no completed pilot verdict. Run --stage pilot first.')
    NLAClient, prov = preflight()
    ids, S, P, _ = load_acts()
    work = [(int(s), 'secret', S[i], 0) for i, s in enumerate(ids)] + \
           [(int(s), 'public', P[i], 0) for i, s in enumerate(ids)]
    client = NLAClient(str(ACTOR_DIR), sglang_url=SGLANG_URL)
    df = generate(client, work, FULL_CSV, prov, set(int(i) for i in ids))
    ok = sum(text_valid(d, e) for d, e in zip(df.description, df.error))
    print(f'\nvalid: {ok}/{len(df)}; re-run to retry any failures')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['plan', 'pilot', 'export-2afc', 'score-2afc', 'verbalize'])
    ap.add_argument('--n', type=int, default=PILOT_N)
    ap.add_argument('--slice', default='P3', choices=SLICES)
    a = ap.parse_args()
    {'plan': lambda: stage_plan(a.n), 'pilot': stage_pilot,
     'export-2afc': lambda: stage_export_2afc(a.slice),
     'score-2afc': stage_score_2afc, 'verbalize': stage_verbalize}[a.stage]()
