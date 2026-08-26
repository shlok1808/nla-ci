"""
Regression tests for scripts/nla_transmission_f.py.

Every bug this file guards against was shipped at least once. The common thread:
they were all in code whose job is to REFUSE something — a resume filter, a
completeness check, an authorisation gate — and the happy path worked fine, so
reading the code did not reveal them.

**Rule this file enforces: if a function's job is to stop something, there is a
test here that proves it stops it.**

Also asserts that running the suite leaves the real `data/` and `results/`
directories byte-identical, because a fixture once wrote simulated metrics into
`results/` and it was committed as if it were a real pilot output.

Run:  python tests/test_nla_transmission_f.py
"""

import os
import sys
import json
import shutil
import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'scripts'))
os.chdir(REPO)
import nla_transmission_f as nt   # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")


def refuses(fn, *a, **k):
    """True when fn refuses via SystemExit or AssertionError."""
    try:
        fn(*a, **k)
        return False, 'did NOT refuse'
    except (SystemExit, AssertionError) as e:
        return True, str(e).split('\n')[0][:70]


def dir_digest(p):
    p = Path(p)
    if not p.exists():
        return 'absent'
    h = hashlib.sha256()
    for f in sorted(p.rglob('*')):
        if f.is_file():
            h.update(f.name.encode())
            h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()[:16]


def main():
    before = {d: dir_digest(REPO / d) for d in ('data', 'results')}
    tmp = Path(tempfile.mkdtemp(prefix='nlatx-'))
    print(f'sandbox: {tmp}\n')

    # ── 1. checksum stability across processes ──────────────────────────────
    print('checksums')
    a = subprocess.run([sys.executable, '-c',
                        f"import sys; sys.path.insert(0,{str(REPO/'scripts')!r}); "
                        "import nla_transmission_f as n; print(n.sha(b'abc'))"],
                       capture_output=True, text=True).stdout.strip()
    b = subprocess.run([sys.executable, '-c',
                        f"import sys; sys.path.insert(0,{str(REPO/'scripts')!r}); "
                        "import nla_transmission_f as n; print(n.sha(b'abc'))"],
                       capture_output=True, text=True).stdout.strip()
    check('sha() is stable across separate interpreters', a == b and bool(a),
          f'{a[:16]} vs {b[:16]}')
    check('sha() is not Python hash()', a != str(abs(hash(b'abc')) % 10**12))

    # ── fixture: a fake manifest + activations ──────────────────────────────
    rng = np.random.default_rng(0)
    n_pairs, D = 8, 3584
    ids = np.arange(300, 300 + n_pairs)
    S = rng.normal(size=(n_pairs, D)).astype(np.float32)
    P = S + rng.normal(scale=0.05, size=(n_pairs, D)).astype(np.float32)
    (tmp / 'results').mkdir(parents=True); (tmp / 'data').mkdir()
    ang, cos = nt.pair_angles(S, P)
    man = pd.DataFrame(dict(scenario_id=ids, angle_deg=ang.round(4),
                            cosine=cos.round(6), angle_quartile=1,
                            act_sha_secret=[nt.sha(x.tobytes()) for x in S],
                            act_sha_public=[nt.sha(x.tobytes()) for x in P]))
    nt.MANIFEST = tmp / 'results/man.csv'
    nt.GO_JSON = tmp / 'results/go.json'
    GP = lambda w='P3': nt.go_path(w)
    nt.VERDICT_JSON = tmp / 'results/verdict.json'
    nt.PILOT_CSV = tmp / 'results/pilot.csv'
    nt.PACKET_CSV = tmp / 'data/packet.csv'
    nt.KEY_CSV = tmp / 'results/key.csv'
    nt.atomic_write_csv(man, nt.MANIFEST)

    # ── 2. the GPU authorisation gate ───────────────────────────────────────
    print('\nverbalize authorisation gate')
    ok, why = refuses(nt.stage_verbalize)
    check('refuses with no recorded 2AFC result', ok, why)

    nt.atomic_write_json(dict(status='NO-GO', correct=12, n=40, p_value=0.9,
                              slice='P3', manifest_digest=nt.file_digest(nt.MANIFEST)),
                         GP())
    ok, why = refuses(nt.stage_verbalize)
    check('refuses on a recorded NO-GO', ok, why)

    nt.atomic_write_json(dict(status='GO', correct=30, n=40, p_value=0.001,
                              slice='P3', manifest_digest='stale-digest-000'),
                         GP())
    ok, why = refuses(nt.stage_verbalize)
    check('refuses when the manifest changed after scoring', ok, why)

    # a PILOT-COMPLETE verdict alone must NOT authorise the run
    nt.atomic_write_json(dict(status='PILOT-COMPLETE'), nt.VERDICT_JSON)
    GP().unlink()
    ok, why = refuses(nt.stage_verbalize)
    check('PILOT-COMPLETE alone does not authorise', ok, why)

    for sl in ('P1', 'P2'):
        ok, why = refuses(nt.stage_verbalize, sl)
        check(f'{sl} (a control slice) can never authorise', ok, why)

    # ── 3. 2AFC population integrity ────────────────────────────────────────
    print('\n2AFC scoring gate')
    nt.PILOT_N = n_pairs                       # locked population for these fixtures
    dg = nt.file_digest(nt.MANIFEST)

    def fixtures(n=n_pairs, digest=None, sl='P3', sids=None):
        sids = list(ids[:n]) if sids is None else sids
        sides = ['A', 'B'] * (n // 2) + (['A'] if n % 2 else [])
        k = pd.DataFrame(dict(item=range(1, n + 1), scenario_id=sids,
                              secret_side=sides, identical=[False] * n,
                              slice=sl, manifest_digest=digest or dg))
        p_ = pd.DataFrame(dict(item=range(1, n + 1), A=['x'] * n, B=['y'] * n,
                               your_answer=sides, slice=sl,
                               manifest_digest=digest or dg))
        return p_, k

    pc, kc = nt.packet_paths('P3')

    def put(p_, k):
        nt.atomic_write_csv(p_, pc); nt.atomic_write_csv(k, kc)
        if GP().exists():
            GP().unlink()

    p_, k = fixtures(); blank = p_.copy(); blank.loc[2:, 'your_answer'] = ''
    put(blank, k)
    ok, why = refuses(nt.stage_score_2afc, 'P3')
    check('refuses when items are left unanswered', ok, why)

    p_, k = fixtures(); junk = p_.copy(); junk.loc[1, 'your_answer'] = 'maybe'
    put(junk, k)
    ok, why = refuses(nt.stage_score_2afc, 'P3')
    check('refuses on an invalid answer token', ok, why)

    # the bug Codex flagged: a MATCHED truncated pair would grade the wrong n
    p_, k = fixtures(n=n_pairs - 2, sids=list(ids[:n_pairs - 2]))
    put(p_, k)
    ok, why = refuses(nt.stage_score_2afc, 'P3')
    check('refuses a matched truncated packet+key (not the locked population)', ok, why)

    p_, k = fixtures(sids=list(ids[:-1]) + [99999])
    put(p_, k)
    ok, why = refuses(nt.stage_score_2afc, 'P3')
    check('refuses when a graded scenario is not in the manifest', ok, why)

    p_, k = fixtures(digest='stale-digest-0000')
    put(p_, k)
    ok, why = refuses(nt.stage_score_2afc, 'P3')
    check('refuses a packet exported against a different manifest', ok, why)

    p_, k = fixtures(sl='full')
    put(p_, k)
    ok, why = refuses(nt.stage_score_2afc, 'P3')
    check('refuses when the packet slice does not match', ok, why)

    # a clean, fully-answered sheet grades against the locked n
    p_, k = fixtures(); put(p_, k)
    nt.stage_score_2afc('P3')
    g = json.loads(GP().read_text())
    check('grades against the locked population', g['n'] == n_pairs,
          f"n={g['n']} correct={g['correct']}")

    ok, why = refuses(nt.stage_score_2afc, 'P3')
    check('refuses to re-grade without --regrade', ok, why)

    # the pre-registered boundary, scaled to this fixture population
    nt.PILOT_2AFC_PASS = 6
    for c, want in ((5, 'NO-GO'), (6, 'GO')):
        p_, k = fixtures()
        ans = list(k.secret_side)
        for i in range(c, n_pairs):
            ans[i] = 'B' if ans[i] == 'A' else 'A'
        p_['your_answer'] = ans
        put(p_, k)
        nt.stage_score_2afc('P3')
        got = json.loads(GP().read_text())
        check(f'{c}/{n_pairs} -> {want}',
              got['status'] == want and got['correct'] == c,
              f"got {got['status']} at {got['correct']}/{n_pairs}")

    # ── 3b. authorisation completeness ──────────────────────────────────────
    print('\nauthorisation completeness')
    p_, k = fixtures(); put(p_, k)
    nt.stage_score_2afc('P3')
    g = json.loads(GP().read_text())
    check('scoring records packet+key digests', 'packet_digest' in g and 'key_digest' in g)

    tampered = p_.copy(); tampered.loc[0, 'your_answer'] = 'B'
    nt.atomic_write_csv(tampered, pc)
    ok, why = refuses(nt.stage_verbalize, 'P3')
    check('refuses when the answer sheet changed after scoring', ok, why)
    nt.atomic_write_csv(p_, pc)

    g2 = dict(g, n=g['n'] - 1)
    nt.atomic_write_json(g2, GP())
    ok, why = refuses(nt.stage_verbalize, 'P3')
    check('refuses when the graded n is not the locked population', ok, why)

    g3 = dict(g, correct=nt.PILOT_2AFC_PASS - 1, status='GO')
    nt.atomic_write_json(g3, GP())
    ok, why = refuses(nt.stage_verbalize, 'P3')
    check('refuses a hand-edited GO below the threshold', ok, why)

    nt.atomic_write_json(g, GP())
    nt.stage_score_2afc('P3', regrade=True, reason='transcription fix')
    g4 = json.loads(GP().read_text())
    check('a regrade preserves the superseded result', len(g4.get('history', [])) == 1,
          f"history={len(g4.get('history', []))}")
    check('a regrade records its reason', bool(g4.get('regrade_reason')))

    # ── 4. manifest locking ─────────────────────────────────────────────────
    print('\nmanifest locking')
    ok, why = refuses(nt.stage_plan, 8, True)     # even with --force
    check('plan refuses --force once downstream artifacts exist', ok, why)
    for f in (GP(), nt.VERDICT_JSON, *nt.packet_paths('P3'), *nt.all_artifacts()):
        if f.exists():
            f.unlink()
    ok, why = refuses(nt.stage_plan, 8, False)
    check('plan refuses to overwrite a locked manifest without --force', ok, why)

    # ── 5. CSV NaN round-trip (shipped bug) ─────────────────────────────────
    print('\nCSV round-trip')
    rt = tmp / 'results/rt.csv'
    nt.atomic_write_csv(pd.DataFrame([dict(description='hello', error='')]), rt)
    row = pd.read_csv(rt).iloc[0]
    check('empty error survives as valid, not as the string "nan"',
          nt.text_valid(row.description, row.error))

    # ── 6. paragraph slicing ────────────────────────────────────────────────
    print('\nparagraph handling')
    check('splits on blank lines with stray whitespace',
          len(nt.split_paragraphs('a\n\nb\n   \nc')) == 3)
    check('P3 of a 2-paragraph description is empty, not a crash',
          nt.slice_text('a\n\nb', 'P3') == '')
    check('paragraph_valid rejects a 2-paragraph description',
          not nt.paragraph_valid('a\n\nb'))

    # ── 6b. stage_pilot end-to-end against a fake client ────────────────────
    print('\nstage_pilot (where the contamination bug lived)')
    nt.METRICS_CSV = tmp / 'results/metrics.csv'
    nt.PILOT_TXT = tmp / 'results/pilot.txt'
    nt.PAIR_ACTS = tmp / 'results/acts.npz'
    np.savez(nt.PAIR_ACTS, scenario_ids=ids, acts_secret=S, acts_public=P)
    pairs_csv = tmp / 'data/pairs.csv'
    pd.DataFrame(dict(scenario_id=ids, story_secret=['only Ann knew'] * n_pairs,
                      story_public=['everyone knew'] * n_pairs,
                      valid=True)).to_csv(pairs_csv, index=False)
    nt.PAIRS_CSV = pairs_csv
    nt.N_DETERMINISM = 2

    class FakeAV:
        def __init__(self, *a, **k): pass
        def generate(self, v, **k):
            tag = 'guarded' if float(v[0]) > 0 else 'open'
            return f'Structured format.\n\nSetup {tag}.\n\nFinal token {tag}.'
    nt.preflight = lambda: (FakeAV, dict(fake=1))
    for f in (nt.PILOT_CSV, nt.VERDICT_JSON):
        if f.exists():
            f.unlink()
    nt.stage_pilot()
    v = json.loads(nt.VERDICT_JSON.read_text())
    check('stage_pilot runs and records a verdict', v.get('status') == 'PILOT-COMPLETE',
          str(v.get('status')))
    check('stage_pilot wrote metrics to the redirected path', nt.METRICS_CSV.exists())

    # a manifest whose digests no longer match the activations must be caught
    S2 = S.copy(); S2[0] += 1.0
    np.savez(nt.PAIR_ACTS, scenario_ids=ids, acts_secret=S2, acts_public=P)
    nt.PILOT_CSV.unlink()
    ok, why = refuses(nt.stage_pilot)
    check('stage_pilot refuses when activations changed under the manifest', ok, why)

    # ── 7. no test wrote into the real repo ─────────────────────────────────
    print('\nisolation')
    after = {d: dir_digest(REPO / d) for d in ('data', 'results')}
    for d in ('data', 'results'):
        check(f'real {d}/ is byte-identical after the suite',
              before[d] == after[d], f'{before[d]} -> {after[d]}')

    shutil.rmtree(tmp, ignore_errors=True)
    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        print('FAILED: ' + ', '.join(FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
