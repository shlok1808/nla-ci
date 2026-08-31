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
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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

    cli = subprocess.run(
        [sys.executable, str(REPO / 'scripts/nla_transmission_f.py'),
         '--stage', 'verbalize', '--slice', 'full'],
        capture_output=True, text=True)
    cli_output = cli.stdout + cli.stderr
    check('CLI passes --slice full to the verbalize gate',
          cli.returncode != 0 and 'result_f_full.json' in cli_output,
          cli_output.split('\n')[0][:90])

    # ── fixture: a fake manifest + activations ──────────────────────────────
    rng = np.random.default_rng(0)
    n_pairs, D = 8, 3584
    ids = np.arange(300, 300 + n_pairs)
    S = rng.normal(size=(n_pairs, D)).astype(np.float32)
    P = S + rng.normal(scale=0.05, size=(n_pairs, D)).astype(np.float32)
    (tmp / 'results').mkdir(parents=True); (tmp / 'data').mkdir()
    ang, cos = nt.pair_angles(S, P)
    angle_order = np.argsort(ang)
    quartiles = np.empty(n_pairs, dtype=int)
    for qi, members in enumerate(np.array_split(angle_order, 4), 1):
        quartiles[members] = qi
    man = pd.DataFrame(dict(scenario_id=ids, angle_deg=ang.round(4),
                            cosine=cos.round(6),
                            angle_quartile=quartiles,
                            act_sha_secret=[nt.sha(x.tobytes()) for x in S],
                            act_sha_public=[nt.sha(x.tobytes()) for x in P]))
    nt.MANIFEST = tmp / 'results/man.csv'
    nt.GO_JSON = tmp / 'results/go.json'
    GP = lambda w='P3': nt.go_path(w)
    nt.VERDICT_JSON = tmp / 'results/verdict.json'
    nt.PILOT_CSV = tmp / 'results/pilot.csv'
    nt.FULL_CSV = tmp / 'results/full.csv'
    nt.PACKET_CSV = tmp / 'data/packet.csv'
    nt.KEY_CSV = tmp / 'results/key.csv'
    # METRICS_CSV and PILOT_TXT must be redirected HERE, with the rest — not
    # later at their point of use. all_artifacts() reads these globals at call
    # time and the manifest-locking test unlinks everything it returns; any
    # path still pointing at the real repo gets DELETED. That deletion silently
    # destroyed two committed pilot outputs (2026-08-27). The isolation check
    # at the end of the suite catches it, but only after the files are gone.
    nt.METRICS_CSV = tmp / 'results/metrics.csv'
    nt.PILOT_TXT = tmp / 'results/pilot.txt'
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
    fixture_prov = 'fixture-provenance-digest'

    def fixtures(n=n_pairs, digest=None, sl='P3', sids=None):
        sids = list(ids[:n]) if sids is None else sids
        sides = ['A', 'B'] * (n // 2) + (['A'] if n % 2 else [])
        k = pd.DataFrame(dict(item=range(1, n + 1), scenario_id=sids,
                              secret_side=sides, identical=[False] * n,
                              slice=sl, manifest_digest=digest or dg,
                              provenance_digest=fixture_prov))
        p_ = pd.DataFrame(dict(item=range(1, n + 1), A=['x'] * n, B=['y'] * n,
                               your_answer=sides, slice=sl,
                               manifest_digest=digest or dg,
                               provenance_digest=fixture_prov))
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
    check('scoring carries the pilot provenance digest',
          g.get('provenance_digest') == fixture_prov,
          str(g.get('provenance_digest')))

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
    # METRICS_CSV / PILOT_TXT are redirected far above, with the other globals
    nt.PAIR_ACTS = tmp / 'results/acts.npz'
    np.savez(nt.PAIR_ACTS, scenario_ids=ids, acts_secret=S, acts_public=P)
    pairs_csv = tmp / 'data/pairs.csv'
    pd.DataFrame(dict(scenario_id=ids, story_secret=['only Ann knew'] * n_pairs,
                      story_public=['everyone knew'] * n_pairs,
                      valid=True)).to_csv(pairs_csv, index=False)
    nt.PAIRS_CSV = pairs_csv
    nt.N_DETERMINISM = 2
    real_preflight = nt.preflight

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
    check('stage_pilot records a canonical provenance digest',
          bool(v.get('provenance_digest')), str(v.get('provenance_digest')))

    # The provenance must survive verdict -> packet/key -> grade, and the paid
    # stage must independently recompute both the grade and current environment.
    v['P3']['status'] = 'PROCEED-TO-2AFC'
    nt.atomic_write_json(v, nt.VERDICT_JSON)
    nt.stage_export_2afc('P3')
    pc, kc = nt.packet_paths('P3')
    exported, exported_key = pd.read_csv(pc), pd.read_csv(kc)
    pilot_prov = v.get('provenance_digest')
    check('2AFC export carries pilot provenance',
          pilot_prov and set(exported.provenance_digest) == {pilot_prov}
          and set(exported_key.provenance_digest) == {pilot_prov})
    exported['your_answer'] = exported_key.secret_side
    nt.atomic_write_csv(exported, pc)
    if GP().exists():
        GP().unlink()
    nt.stage_score_2afc('P3')
    graded = json.loads(GP().read_text())
    check('2AFC grade carries pilot provenance',
          graded.get('provenance_digest') == pilot_prov,
          str(graded.get('provenance_digest')))

    forged_packet = exported.copy()
    for i in range(5, n_pairs):
        side = exported_key.loc[i, 'secret_side']
        forged_packet.loc[i, 'your_answer'] = 'B' if side == 'A' else 'A'
    nt.atomic_write_csv(forged_packet, pc)
    forged_grade = dict(graded, correct=nt.PILOT_2AFC_PASS, status='GO',
                        packet_digest=nt.file_digest(pc))
    nt.atomic_write_json(forged_grade, GP())
    if nt.FULL_CSV.exists():
        nt.FULL_CSV.unlink()
    try:
        nt.stage_verbalize('P3')
        forged_msg = 'did not refuse'
    except (SystemExit, AssertionError) as e:
        forged_msg = str(e)
    check('verbalize recomputes the frozen 2AFC grade',
          'recomputed' in forged_msg.lower(), forged_msg.split('\n')[0][:90])

    nt.atomic_write_csv(exported, pc)
    nt.stage_score_2afc('P3', regrade=True, reason='restore test fixture')
    if nt.FULL_CSV.exists():
        nt.FULL_CSV.unlink()
    nt.preflight = lambda: (FakeAV, dict(fake=2))
    try:
        nt.stage_verbalize('P3')
        provenance_msg = 'did not refuse'
    except (SystemExit, AssertionError) as e:
        provenance_msg = str(e)
    check('verbalize refuses a different generation environment',
          'provenance' in provenance_msg.lower(), provenance_msg.split('\n')[0][:90])
    nt.preflight = lambda: (FakeAV, dict(fake=1))

    # ── 6c. live-server identity is a fail-closed system boundary ─────────────
    print('\nSGLang model identity')
    actor = tmp / 'actor_hf'
    actor.mkdir()
    (actor / 'nla_meta.yaml').write_text(
        'injection_token_id: 149705\ninjection_scale: 150\n')

    class BoundaryClient:
        def __init__(self, *a, **k): pass
        def generate(self, vector, temperature=0, max_new_tokens=300): return 'ok'

    fake_module = types.ModuleType('nla_inference')
    fake_module.__file__ = __file__
    fake_module.NLAClient = BoundaryClient
    old_module = sys.modules.get('nla_inference')
    sys.modules['nla_inference'] = fake_module

    class ModelInfoHandler(BaseHTTPRequestHandler):
        model_path = str(tmp / 'wrong_checkpoint')
        def do_GET(self):
            if self.path not in ('/model_info', '/get_model_info'):
                self.send_response(404); self.end_headers(); return
            body = json.dumps(dict(model_path=self.model_path,
                                   weight_version='test-revision')).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers(); self.wfile.write(body)
        def log_message(self, *a): pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), ModelInfoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    old_actor, old_url = nt.ACTOR_DIR, nt.SGLANG_URL
    nt.ACTOR_DIR = actor
    nt.SGLANG_URL = f'http://127.0.0.1:{server.server_port}'
    ok, why = refuses(real_preflight)
    check('preflight refuses a server loaded with the wrong checkpoint',
          ok and 'model path' in why.lower(), why)

    ModelInfoHandler.model_path = str(actor.resolve())
    _, live_prov = real_preflight()
    check('preflight records live model and input identities',
          live_prov.get('server_model_path') == str(actor.resolve())
          and live_prov.get('server_weight_version') == 'test-revision'
          and live_prov.get('pair_acts_sha') == nt.file_digest(nt.PAIR_ACTS)
          and live_prov.get('pairs_csv_sha') == nt.file_digest(nt.PAIRS_CSV))
    server.shutdown(); server.server_close(); thread.join(timeout=2)
    nt.ACTOR_DIR, nt.SGLANG_URL = old_actor, old_url
    if old_module is None:
        del sys.modules['nla_inference']
    else:
        sys.modules['nla_inference'] = old_module

    # Balanced counts are not enough: the labels must be the actual quartiles
    # of the full activation population.
    good_man = pd.read_csv(nt.MANIFEST)
    bad_man = good_man.copy()
    bad_man['angle_quartile'] = bad_man.angle_quartile.map({1: 2, 2: 1, 3: 4, 4: 3})
    nt.atomic_write_csv(bad_man, nt.MANIFEST)
    ok, why = refuses(nt.stage_pilot)
    check('stage_pilot refuses balanced but incorrect quartile labels',
          ok and 'quartile' in why.lower(), why)
    nt.atomic_write_csv(good_man, nt.MANIFEST)

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
