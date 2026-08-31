"""test_reference_extraction_f.py — the blinding and verification guards.

The drafter must never see Qwen's response or the old GPT-4o-mini label. Both
sit one column away in the same CSV, so the invariant is one careless edit from
breaking, and nothing downstream would notice.

Run:  python3 tests/test_reference_extraction_f.py   (exit code is the result)
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import reference_extraction_f as R  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"PASS" if cond else "FAIL"}  {name}' + (f'  — {detail}' if detail else ''))


def refuses(fn, *a, **k):
    try:
        fn(*a, **k)
        return False, 'accepted'
    except SystemExit as e:
        return True, str(e)[:70]
    except Exception as e:
        return True, str(e)[:70]


print('\nblinding')
full = R.load_scenarios(calibration=False)
check('exposes exactly scenario_id and scenario',
      set(full.columns) == {'scenario_id', 'scenario'}, str(list(full.columns)))
for banned in ('response', 'label', 'judge_reasoning', 'confidence'):
    check(f'never exposes `{banned}`', banned not in full.columns)
check('the extractor prompt names no response/label field',
      'response' not in R.EXTRACTOR_SYSTEM.lower().split('speculate')[0]
      or 'do not see any' in R.EXTRACTOR_SYSTEM.lower())

print('\ncalibration sampling')
cal = R.load_scenarios(calibration=True)
check('is a subset of tier 3', set(cal.scenario_id) <= set(full.scenario_id))
check('deterministic under a fixed seed',
      list(cal.scenario_id) == list(R.load_scenarios(calibration=True).scenario_id))
check('sorted by scenario_id', list(cal.scenario_id) == sorted(cal.scenario_id))
check('smaller than the full set', len(cal) < len(full), f'{len(cal)} < {len(full)}')

print('\nreference validation (delegates to the frozen rubric validator)')
good = {
    'scenario_id': 1, 'speaker': 'Alex', 'recipient': 'Mark',
    'recipient_known_context': 'Mark knows they are cousins.',
    'protected_facts': [{'id': 'f1', 'subject': 'Peter',
                         'proposition': 'Peter cheated on a proposal.'}],
    'authorization': 'unauthorized',
    'transmission_principle': 'confided privately, kept between them',
    'norm_override': 'none', 'verified_by_human': True,
}
ok, why = refuses(R.validate_reference, good)
check('accepts a well-formed verified reference', not ok, why)

unverified = {**good, 'verified_by_human': False}
ok, why = refuses(R.validate_reference, unverified)
check('refuses a reference not marked human-verified', ok, why)

for name, bad in [
    ('empty protected_facts', {**good, 'protected_facts': []}),
    ('unknown authorization', {**good, 'authorization': 'maybe'}),
    ('unknown norm_override', {**good, 'norm_override': 'sometimes'}),
    ('blank speaker', {**good, 'speaker': '   '}),
    ('duplicate fact ids', {**good, 'protected_facts': [
        {'id': 'f1', 'subject': 'A', 'proposition': 'x'},
        {'id': 'f1', 'subject': 'B', 'proposition': 'y'}]}),
    ('fact missing proposition', {**good, 'protected_facts': [
        {'id': 'f1', 'subject': 'A'}]}),
]:
    ok, why = refuses(R.validate_reference, bad)
    check(f'refuses {name}', ok, why)

print('\nvalidate stage gate')
tmp = Path(tempfile.mkdtemp()) / 'draft.json'
orig = R.DRAFT_JSON
try:
    R.DRAFT_JSON = tmp
    tmp.write_text(json.dumps([{**good, 'verified_by_human': False}]))
    ok, why = refuses(R.stage_validate)
    check('stage refuses while any reference is unverified', ok, why)

    tmp.write_text(json.dumps([good]))
    ok, why = refuses(R.stage_validate)
    check('stage passes once all are verified and valid', not ok, why)

    tmp.write_text(json.dumps([{**good, 'authorization': 'nonsense'}]))
    ok, why = refuses(R.stage_validate)
    check('stage refuses a verified-but-invalid reference', ok, why)
finally:
    R.DRAFT_JSON = orig

print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
sys.exit(1 if FAIL else 0)
