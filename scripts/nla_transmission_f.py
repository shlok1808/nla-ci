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
import hashlib
import inspect
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
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
KEY_CSV      = Path('results/nla_transmission_2afc_key_f.csv')   # + _{slice} suffix
GO_JSON      = Path('results/nla_transmission_2afc_result_f.json')  # + _{slice}
AUTHORISING_SLICES = ('P3', 'full')   # P1/P2 are controls; they never authorise

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


def sha(b):
    """Stable across processes. Python's hash() is randomized per interpreter
    (PYTHONHASHSEED) and produced different values for identical bytes in
    consecutive runs — useless as a checksum."""
    return hashlib.sha256(bytes(b)).hexdigest()[:32]


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:32]


def object_digest(obj):
    """Stable digest for JSON-compatible provenance records."""
    payload = json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()
    return sha(payload)


def atomic_write_json(obj, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    with open(tmp, 'rb') as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


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

def stage_plan(n=PILOT_N, force=False):
    """Lock the pilot sample BEFORE any description exists. Stratified by
    within-pair angle quartile so a NO-GO cannot be an artifact of having
    sampled only near-identical inputs, and so the verdict can be read by
    quartile (uniform insensitivity vs a threshold response)."""
    downstream = [p for p in all_artifacts() if p.exists()]
    if downstream:
        raise SystemExit(
            'refusing to re-plan: downstream artifacts already exist —\n  '
            + '\n  '.join(str(p) for p in downstream)
            + '\nRe-planning now would swap the sample under results already '
              'produced from it. Delete them deliberately if that is the intent.')
    if MANIFEST.exists() and not force:
        raise SystemExit(
            f'{MANIFEST} already exists (digest {file_digest(MANIFEST)}).\n'
            f'Re-planning after seeing any output would un-lock the sample. '
            f'Pass --force only if no description has been generated.')
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
        act_sha_secret=[sha(S[i].tobytes()) for i in sel],
        act_sha_public=[sha(P[i].tobytes()) for i in sel]))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(man, MANIFEST)
    print(f'locked {len(man)} pairs -> {MANIFEST}  (digest {file_digest(MANIFEST)})')
    print(man.groupby('angle_quartile')['angle_deg']
          .agg(['count', 'min', 'median', 'max']).round(2).to_string())
    print(f'\nfull-population angle: median {np.median(ang):.2f} deg, '
          f'range [{ang.min():.2f}, {ang.max():.2f}]')


# ── Generation ────────────────────────────────────────────────────────────────

def sglang_model_info():
    """Read the identity of the model the live server actually loaded."""
    errors = []
    for endpoint in ('model_info', 'get_model_info'):
        url = f'{SGLANG_URL.rstrip("/")}/{endpoint}'
        try:
            with urlopen(url, timeout=5) as response:
                info = json.loads(response.read().decode())
            assert isinstance(info, dict), f'{url} did not return a JSON object'
            assert str(info.get('model_path', '')).strip(), (
                f'{url} omitted model_path')
            return info, endpoint
        except HTTPError as e:
            if e.code == 404:
                errors.append(f'{endpoint}: HTTP 404')
                continue
            raise AssertionError(f'cannot read SGLang model identity from {url}: {e}') from e
        except (URLError, TimeoutError, json.JSONDecodeError) as e:
            errors.append(f'{endpoint}: {e}')
    raise AssertionError(
        'cannot verify the live SGLang model identity — ' + '; '.join(errors))


def preflight():
    meta = ACTOR_DIR / 'nla_meta.yaml'
    assert meta.exists(), f'{meta} missing — download the AV checkpoint'
    txt = meta.read_text()
    m = re.search(r'injection_token_id:\s*(\d+)', txt)
    assert m and int(m.group(1)) == EXPECTED_INJECTION_TOKEN_ID, (
        f'injection_token_id != {EXPECTED_INJECTION_TOKEN_ID} (L9)')
    sc = re.search(r'injection_scale:\s*([\d.]+)', txt)
    assert sc and abs(float(sc.group(1)) - 150.0) < 1e-6, (
        f'injection_scale is {sc.group(1) if sc else "absent"}, expected 150 (L9). '
        f'The direction-only premise of this experiment depends on it.')
    from nla_inference import NLAClient
    sig = inspect.signature(NLAClient.generate)
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()) \
        or {'temperature', 'max_new_tokens'} <= set(sig.parameters), \
        f'NLAClient.generate lacks temperature/max_new_tokens: {sig}'
    inf_py = Path(inspect.getfile(NLAClient))     # where it ACTUALLY came from
    server_info, server_endpoint = sglang_model_info()
    server_model = Path(str(server_info['model_path'])).expanduser()
    if not server_model.is_absolute():
        server_model = Path.cwd() / server_model
    server_model = server_model.resolve()
    expected_model = ACTOR_DIR.expanduser().resolve()
    assert server_model == expected_model, (
        f'SGLang model path is {server_model}, expected {expected_model}. '
        f'Restart it with --model-path {ACTOR_DIR}.')
    prov = dict(injection_token_id=EXPECTED_INJECTION_TOKEN_ID,
                injection_scale=float(sc.group(1)),
                actor_dir=str(ACTOR_DIR.resolve()),
                actor_meta_sha=file_digest(meta),
                nla_inference_path=str(inf_py),
                nla_inference_sha=file_digest(inf_py),
                sglang_url=SGLANG_URL,
                server_model_path=str(server_model),
                server_weight_version=server_info.get('weight_version'),
                server_model_info_endpoint=server_endpoint,
                pair_acts_sha=file_digest(PAIR_ACTS),
                pairs_csv_sha=file_digest(PAIRS_CSV),
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
    ids, S, P, pairs = load_acts()
    idx = {int(s): i for i, s in enumerate(ids)}

    # the manifest is the pre-registration; verify it still describes this npz
    assert len(man) == len(ids_m) == PILOT_N, f'manifest is not {PILOT_N} unique ids'
    assert ids_m <= set(int(i) for i in ids), 'manifest holds ids absent from the npz'
    ang_all, cos_all = pair_angles(S, P)
    quartile_all = pd.qcut(ang_all, 4, labels=[1, 2, 3, 4])
    for _, r in man.iterrows():
        i = idx[int(r.scenario_id)]
        assert abs(ang_all[i] - r.angle_deg) < 1e-3, (
            f'manifest angle for {int(r.scenario_id)} does not match the npz — '
            f'{r.angle_deg} vs {ang_all[i]:.4f}. The activations changed under it.')
        assert sha(S[i].tobytes()) == r.act_sha_secret, \
            f'secret activation for {int(r.scenario_id)} does not match its manifest digest'
        assert sha(P[i].tobytes()) == r.act_sha_public, \
            f'public activation for {int(r.scenario_id)} does not match its manifest digest'
        assert abs(cos_all[i] - r.cosine) < 1e-5, (
            f'manifest cosine for {int(r.scenario_id)} does not match the npz')
        assert int(r.angle_quartile) == int(quartile_all[i]), (
            f'manifest quartile for {int(r.scenario_id)} does not match the full '
            f'activation population — {int(r.angle_quartile)} vs '
            f'{int(quartile_all[i])}')
    qc = man.angle_quartile.value_counts().to_dict()
    assert set(qc) == {1, 2, 3, 4} and len(set(qc.values())) == 1, (
        f'manifest quartiles are not balanced: {qc}')
    man_digest = file_digest(MANIFEST)
    print(f'manifest verified against the npz (digest {man_digest})')

    NLAClient, prov = preflight()
    prov['manifest_digest'] = man_digest
    prov_digest = object_digest(prov)

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
                   nondeterministic=nondet, manifest_digest=man_digest,
                   provenance=prov, provenance_digest=prov_digest)
    print('\n' + '=' * 78)
    if nondet:
        verdict['status'] = 'INVALID'
        print(f'INVALID — greedy decoding was not deterministic for {len(nondet)} '
              f'activations: {nondet}')
        print('Every similarity metric below would be measuring decoder noise.')
        atomic_write_json(verdict, VERDICT_JSON); return
    if len(complete) < n_expected:
        verdict['status'] = 'INVALID'
        print(f'INVALID — {len(complete)}/{n_expected} pairs complete. A verdict '
              f'from a partial subset is not a result; re-run to retry failures.')
        atomic_write_json(verdict, VERDICT_JSON); return
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
    # a 2- or 4-paragraph description makes P3 unrecoverable ('' by slicing),
    # which would silently look like an identical pair
    n_bad = int((~M.both_3para).sum())
    if n_bad:
        verdict['P3']['status'] = 'INVALID'
        verdict['P3']['reason'] = f'{n_bad}/{len(M)} pairs lack 3 parsed paragraphs'
        print(f'\n  !! P3 branch INVALID: {n_bad}/{len(M)} pairs did not parse into 3 '
              f'paragraphs, so their P3 is empty and would read as identical.')
    verdict['status'] = 'PILOT-COMPLETE'
    atomic_write_json(verdict, VERDICT_JSON)
    atomic_write_csv(M, METRICS_CSV)


# ── Stage: export / score the blinded 2AFC ───────────────────────────────────

def packet_paths(which):
    return (PACKET_CSV.with_name(f'{PACKET_CSV.stem}_{which}.csv'),
            KEY_CSV.with_name(f'{KEY_CSV.stem}_{which}.csv'))


def go_path(which):
    """One result file per slice, so a full-slice grade cannot overwrite P3."""
    return GO_JSON.with_name(f'{GO_JSON.stem}_{which}.json')


def all_artifacts():
    """Everything a run can generate. Re-planning must refuse if ANY exists —
    listing them by hand missed PILOT_TXT, FULL_CSV and the P1/P2 packets."""
    out = [PILOT_CSV, FULL_CSV, VERDICT_JSON, METRICS_CSV, PILOT_TXT]
    for w in SLICES:
        out += list(packet_paths(w)) + [go_path(w)]
    return out


def stage_export_2afc(which='P3'):
    v = json.loads(VERDICT_JSON.read_text())
    assert v.get('status') == 'PILOT-COMPLETE', 'run --stage pilot first'
    assert v.get('manifest_digest') == file_digest(MANIFEST), (
        'the pilot verdict was produced against a different manifest — '
        're-run --stage pilot before exporting')
    if v[which].get('status') in ('NO-GO', 'INVALID'):
        print(f'{which} branch is {v[which]["status"]} '
              f'(ceiling {v[which]["ceiling_2afc"]:.3f}). Not exporting — human '
              f'judgement cannot change this.')
        return
    man = pd.read_csv(MANIFEST)
    man_digest = file_digest(MANIFEST)
    df = pd.read_csv(PILOT_CSV)
    valid = np.array([text_valid(d, e) for d, e in zip(df.description, df.error)])
    df = df[(df.repeat == 0) & valid]
    assert 'prov' in df, 'pilot descriptions predate provenance binding; re-run pilot'
    prov_values = set(df.prov.astype(str))
    assert len(prov_values) == 1, 'pilot descriptions mix generation environments'
    try:
        pilot_prov = json.loads(next(iter(prov_values)))
    except json.JSONDecodeError as e:
        raise AssertionError('pilot description provenance is not valid JSON') from e
    pilot_prov_digest = object_digest(pilot_prov)
    assert pilot_prov_digest == v.get('provenance_digest'), (
        'pilot verdict provenance does not match the generated descriptions')
    piv = df.pivot_table(index='scenario_id', columns='variant',
                         values='description', aggfunc='first').dropna()
    piv = piv[piv.index.isin(man.scenario_id.astype(int))]
    assert len(piv) == len(man), (
        f'{len(piv)}/{len(man)} manifest pairs available — export the locked '
        f'population or not at all')

    # EVERY locked pair is exported, including identical ones. Dropping them
    # would silently replace the pre-registered n=40 population with an
    # easier subset, and the 26/40 rule would no longer mean what it says.
    rng = np.random.default_rng(SEED + 1)
    order = rng.permutation(len(piv))
    sides = np.array(['A'] * (len(piv) // 2) + ['B'] * (len(piv) - len(piv) // 2))
    rng.shuffle(sides)                        # exact A/B balance
    pk, key = [], []
    for item, (pos, side) in enumerate(zip(order, sides), 1):
        sid = piv.index[pos]
        s_txt = slice_text(piv.iloc[pos]['secret'], which)
        p_txt = slice_text(piv.iloc[pos]['public'], which)
        pk.append(dict(item=item, A=s_txt if side == 'A' else p_txt,
                       B=p_txt if side == 'A' else s_txt, your_answer='',
                       slice=which, manifest_digest=man_digest,
                       provenance_digest=pilot_prov_digest))
        key.append(dict(item=item, scenario_id=int(sid), secret_side=side,
                        identical=(norm_text(s_txt) == norm_text(p_txt)),
                        slice=which, manifest_digest=man_digest,
                        provenance_digest=pilot_prov_digest))
    pcsv, kcsv = packet_paths(which)
    pcsv.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(pd.DataFrame(pk), pcsv)
    atomic_write_csv(pd.DataFrame(key), kcsv)
    n_id = sum(k['identical'] for k in key)
    print(f'wrote {len(pk)} blinded items ({which}) -> {pcsv}')
    print(f'  {n_id} of them are identical pairs — kept deliberately; guessing on '
          f'those is what the {MIN_INTERESTING_2AFC} ceiling already accounts for')
    print(f'  A/B balance: {sum(s=="A" for s in sides)}/{sum(s=="B" for s in sides)}')
    print(f'answer key (do not open) -> {kcsv}')
    print('\nOne description came from the activation where the information was')
    print('CONFIDENTIAL, one where it was COMMON KNOWLEDGE. Put A or B in')
    print(f'`your_answer` for EVERY item — guess when unsure; a skipped item is')
    print(f'not neutral, it silently shrinks the population.')
    print(f'Then: python scripts/nla_transmission_f.py --stage score-2afc --slice {which}')


def compute_2afc_result(which):
    """Recompute the complete grade from the frozen packet and key.

    Both scoring and paid-run authorisation call this same function, so an
    edited summary JSON can never become a second source of truth.
    """
    from scipy.stats import binomtest
    pcsv, kcsv = packet_paths(which)
    pk, key = pd.read_csv(pcsv), pd.read_csv(kcsv)
    n_locked = len(key)

    # integrity: the graded population must BE the locked population. Matching
    # packet/key files are not enough — a truncated PAIR would grade 26/30 as GO.
    man = pd.read_csv(MANIFEST)
    man_ids, man_digest = set(man.scenario_id.astype(int)), file_digest(MANIFEST)
    assert len(pk) == n_locked, f'packet has {len(pk)} items, key has {n_locked}'
    assert pk.item.is_unique and key.item.is_unique, 'duplicate item numbers'
    assert set(pk.item) == set(key.item), 'packet and key items disagree'
    assert n_locked == len(man_ids) == PILOT_N, (
        f'key holds {n_locked} items but the manifest locks {len(man_ids)}')
    assert key.scenario_id.is_unique, 'duplicate scenario_id in the key'
    assert set(key.scenario_id.astype(int)) == man_ids, (
        'the graded scenarios are not the locked manifest scenarios — '
        f'{len(set(key.scenario_id.astype(int)) ^ man_ids)} differ')
    for frame, nm in ((pk, 'packet'), (key, 'key')):
        assert 'manifest_digest' in frame, f'{nm} predates digest binding; re-export'
        assert set(frame.manifest_digest) == {man_digest}, (
            f'{nm} was exported against a different manifest — re-export')
        assert set(frame['slice']) == {which}, (
            f'{nm} is slice {set(frame["slice"])}, scoring {which}')
        assert 'provenance_digest' in frame, (
            f'{nm} predates generation-provenance binding; re-export')
        assert frame.provenance_digest.nunique() == 1, (
            f'{nm} mixes generation provenance')
    provenance_digest = str(pk.provenance_digest.iloc[0])
    assert set(key.provenance_digest.astype(str)) == {provenance_digest}, (
        'packet and key provenance disagree')
    nA = int((key.secret_side == 'A').sum())
    assert abs(nA - n_locked / 2) <= 1, f'A/B balance is {nA}/{n_locked - nA}'
    m = pk.merge(key, on='item')
    ansd = m.your_answer.astype(str).str.strip().str.upper()
    bad = ~ansd.isin(['A', 'B'])
    if bad.any():
        raise SystemExit(
            f'{int(bad.sum())}/{n_locked} items are unanswered or invalid '
            f'(items {list(m.item[bad])[:10]}).\nEvery locked item must be graded — '
            f'dropping items replaces the pre-registered population with an easier '
            f'subset and invalidates the {PILOT_2AFC_PASS}/{n_locked} rule.')

    correct = int((ansd == m.secret_side).sum())
    n = n_locked
    bt = binomtest(correct, n, 0.5, alternative='greater')
    lo, hi = bt.proportion_ci(0.95)
    n_id = int(m.identical.sum())
    go = (correct >= PILOT_2AFC_PASS) and (correct / n >= MIN_INTERESTING_2AFC)
    res = dict(slice=which, n=n, correct=correct, accuracy=round(correct / n, 4),
               p_value=round(bt.pvalue, 6), ci=[round(lo, 4), round(hi, 4)],
               n_identical=n_id, pass_threshold=PILOT_2AFC_PASS,
               min_interesting=MIN_INTERESTING_2AFC,
               manifest_digest=man_digest, provenance_digest=provenance_digest,
               packet_digest=file_digest(pcsv), key_digest=file_digest(kcsv),
               status='GO' if go else 'NO-GO')
    return res, bt, m


def stage_score_2afc(which='P3', regrade=False, reason=''):
    gp = go_path(which)
    if gp.exists() and not regrade:
        prev = json.loads(gp.read_text())
        raise SystemExit(
            f'{gp} already records {prev.get("status")} '
            f'({prev.get("correct")}/{prev.get("n")}, p={prev.get("p_value")}).\n'
            f'Re-scoring the same answers cannot change them; re-scoring DIFFERENT '
            f'answers after seeing a result is exactly what the pre-registration '
            f'forbids. Pass --regrade only to correct a transcription error.')
    res, bt, m = compute_2afc_result(which)
    correct, n, n_id = res['correct'], res['n'], res['n_identical']
    lo, hi = res['ci']
    print(f'blinded 2AFC ({which}): {correct}/{n} = {correct/n:.3f}')
    print(f'  95% CI [{lo:.3f}, {hi:.3f}]   one-sided exact binomial p={bt.pvalue:.4f}')
    print(f'  identical pairs included: {n_id} (ceiling {1 - n_id/(2*n):.3f})')
    print(f'  pre-registered pass: {PILOT_2AFC_PASS}/{n}')

    go = res['status'] == 'GO'
    # append-only: a correction never erases the grade it replaces
    if gp.exists():
        old = json.loads(gp.read_text())
        hist = old.pop('history', [])
        hist.append(dict(old, superseded=True))
        res['history'] = hist
        res['regrade_reason'] = reason or '(none given)'
        print(f'  NOTE: superseding a previous {old.get("status")} at '
              f'{old.get("correct")}/{old.get("n")} — both are kept in {gp}')
    atomic_write_json(res, gp)
    print()
    if go:
        print(f'  => GO (recorded in {gp}). --stage verbalize is now unlocked.')
    else:
        print(f'  => NO-GO. Scoped result: "greedy AV descriptions did not let a')
        print(f'     blinded reader distinguish these natural private/public')
        print(f'     activation pairs under the tested checkpoint and decoding')
        print(f'     configuration." Recorded in {gp}; verbalize stays locked.')


def stage_verbalize(which='P3'):
    """The paid run. Every field of the authorisation is checked, because
    'status == GO' alone would let a P1 control, a shrunken population, or an
    edited answer sheet unlock 466 generations."""
    gp = go_path(which)
    if which not in AUTHORISING_SLICES:
        raise SystemExit(
            f'{which} is a control slice; only {AUTHORISING_SLICES} can authorise '
            f'scaling up.')
    if not gp.exists():
        raise SystemExit(
            f'refusing to run: no recorded 2AFC result at {gp}. A completed pilot '
            f'is not authorisation — the scientific gate is --stage score-2afc.')
    g = json.loads(gp.read_text())
    checks = [
        (g.get('status') == 'GO', f'recorded status is {g.get("status")}'),
        (g.get('slice') == which, f'result is for slice {g.get("slice")}, not {which}'),
        (g.get('n') == PILOT_N, f'graded n={g.get("n")}, expected {PILOT_N}'),
        (g.get('correct', -1) >= PILOT_2AFC_PASS,
         f'{g.get("correct")}/{g.get("n")} is below the {PILOT_2AFC_PASS} threshold'),
        (g.get('manifest_digest') == file_digest(MANIFEST),
         'the manifest changed since the 2AFC was scored'),
    ]
    pcsv, kcsv = packet_paths(which)
    for f, fld, nm in ((pcsv, 'packet_digest', 'answer sheet'), (kcsv, 'key_digest', 'key')):
        checks.append((f.exists() and g.get(fld) == file_digest(f),
                       f'the {nm} changed after scoring ({f.name})'))
    bad = [why for ok, why in checks if not ok]
    if bad:
        raise SystemExit('refusing to run:\n  ' + '\n  '.join(bad)
                         + '\nDo not re-grade to obtain a GO.')

    recomputed, _, _ = compute_2afc_result(which)
    mismatch = [
        f'stored grade {field}={g.get(field)!r} does not match recomputed {value!r}'
        for field, value in recomputed.items() if g.get(field) != value
    ]
    if mismatch:
        raise SystemExit('refusing to run:\n  ' + '\n  '.join(mismatch)
                         + '\nThe frozen packet/key, not the summary JSON, are authoritative.')

    NLAClient, prov = preflight()
    prov['manifest_digest'] = file_digest(MANIFEST)
    current_prov_digest = object_digest(prov)
    if current_prov_digest != g.get('provenance_digest'):
        raise SystemExit(
            'refusing to run: current generation provenance does not match the pilot.\n'
            f'  pilot:   {g.get("provenance_digest")}\n'
            f'  current: {current_prov_digest}\n'
            'Use the same checkpoint, server model, code, inputs and package versions.')
    print(f'authorised by {gp}: {which} {g["correct"]}/{g["n"]} (p={g["p_value"]})')
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
    ap.add_argument('--reason', default='',
                    help='score-2afc: why a regrade was necessary (kept in history)')
    ap.add_argument('--regrade', action='store_true',
                    help='score-2afc: overwrite a recorded result (transcription fixes only)')
    ap.add_argument('--force', action='store_true',
                    help='plan: overwrite an existing locked manifest')
    a = ap.parse_args()
    {'plan': lambda: stage_plan(a.n, a.force), 'pilot': stage_pilot,
     'export-2afc': lambda: stage_export_2afc(a.slice),
     'score-2afc': lambda: stage_score_2afc(a.slice, a.regrade, a.reason),
     'verbalize': lambda: stage_verbalize(a.slice)}[a.stage]()
