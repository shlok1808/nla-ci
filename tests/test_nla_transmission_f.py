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
                         nt.GO_JSON)
    ok, why = refuses(nt.stage_verbalize)
    check('refuses on a recorded NO-GO', ok, why)

    nt.atomic_write_json(dict(status='GO', correct=30, n=40, p_value=0.001,
                              slice='P3', manifest_digest='stale-digest-000'),
                         nt.GO_JSON)
    ok, why = refuses(nt.stage_verbalize)
    check('refuses when the manifest changed after scoring', ok, why)

    # a PILOT-COMPLETE verdict alone must NOT authorise the run
    nt.atomic_write_json(dict(status='PILOT-COMPLETE'), nt.VERDICT_JSON)
    nt.GO_JSON.unlink()
    ok, why = refuses(nt.stage_verbalize)
    check('PILOT-COMPLETE alone does not authorise', ok, why)

    # ── 3. 2AFC population integrity ────────────────────────────────────────
    print('\n2AFC scoring gate')
    key = pd.DataFrame(dict(item=range(1, 41), scenario_id=range(1, 41),
                            secret_side=['A', 'B'] * 20, identical=[False] * 40))
    pk = pd.DataFrame(dict(item=range(1, 41), A=['x'] * 40, B=['y'] * 40,
                           your_answer=['A', 'B'] * 20))
    pc, kc = nt.packet_paths('P3')
    nt.atomic_write_csv(key, kc)

    blank = pk.copy(); blank.loc[6:, 'your_answer'] = ''      # only 6 answered
    nt.atomic_write_csv(blank, pc)
    ok, why = refuses(nt.stage_score_2afc, 'P3')
    check('refuses when items are left unanswered', ok, why)

    junk = pk.copy(); junk.loc[3, 'your_answer'] = 'maybe'
    nt.atomic_write_csv(junk, pc)
    ok, why = refuses(nt.stage_score_2afc, 'P3')
    check('refuses on an invalid answer token', ok, why)

    short = pk.iloc[:20].copy()
    nt.atomic_write_csv(short, pc)
    ok, why = refuses(nt.stage_score_2afc, 'P3')
    check('refuses when the packet is smaller than the locked population', ok, why)

    # a full, all-correct sheet must be graded against the LOCKED n and rule
    full = pk.copy(); full['your_answer'] = key.secret_side
    nt.atomic_write_csv(full, pc)
    nt.stage_score_2afc('P3')
    g = json.loads(nt.GO_JSON.read_text())
    check('grades against the locked n=40, not the answered subset', g['n'] == 40,
          f"n={g['n']} correct={g['correct']}")
    check('40/40 correct records GO', g['status'] == 'GO')

    # 26/40 is the pre-registered boundary — 25 must fail, 26 must pass
    for c, want in ((25, 'NO-GO'), (26, 'GO')):
        ans = list(key.secret_side)
        for i in range(c, 40):
            ans[i] = 'B' if ans[i] == 'A' else 'A'
        sheet = pk.copy(); sheet['your_answer'] = ans
        nt.atomic_write_csv(sheet, pc)
        nt.stage_score_2afc('P3')
        got = json.loads(nt.GO_JSON.read_text())
        check(f'{c}/40 -> {want}', got['status'] == want and got['correct'] == c,
              f"got {got['status']} at {got['correct']}/40")

    # ── 4. manifest locking ─────────────────────────────────────────────────
    print('\nmanifest locking')
    ok, why = refuses(nt.stage_plan, 8, False)
    check('plan refuses to overwrite a locked manifest', ok, why)

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
