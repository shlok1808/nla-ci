"""test_judge_replication_f.py — the temperature-fallback guard.

Modern reasoning models removed sampling parameters (Claude Opus 5 / Opus 4.8 /
4.7 / Fable 5 / Sonnet 5 return 400 on temperature; the GPT-5.6 tiers are
undocumented). judge_call() exists to survive that WITHOUT silently absorbing
unrelated failures into 270 'error' rows.

Every test here is about what the guard REFUSES to do, per the rule established
in tests/test_nla_transmission_f.py: if a function's job is to stop something,
there is a test proving it stops it.

Run:  python tests/test_judge_replication_f.py   (exit code is the result)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import judge_replication_f as J

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  {"PASS" if cond else "FAIL"}  {name}' + (f'  — {detail}' if detail else ''))


class FakeCompletions:
    def __init__(self, outer):
        self.outer = outer

    def create(self, **kw):
        self.outer.calls.append(kw)
        if 'temperature' in kw and self.outer.reject_temp:
            raise RuntimeError(self.outer.error_text)
        if self.outer.always_raise:
            raise RuntimeError(self.outer.error_text)
        return 'RESPONSE'


class FakeChat:
    def __init__(self, outer):
        self.completions = FakeCompletions(outer)


class FakeClient:
    """Minimal stand-in for the OpenAI client; records every call's kwargs."""

    def __init__(self, reject_temp=False, always_raise=False,
                 error_text='400 Unsupported parameter: temperature'):
        self.calls = []
        self.reject_temp = reject_temp
        self.always_raise = always_raise
        self.error_text = error_text
        self.chat = FakeChat(self)


def call(client):
    return J.judge_call(client, 'test-model', 'SYSTEM', 'USER')


def reset():
    """The flag is module-level and persists across a run by design."""
    J._SUPPORTS_TEMPERATURE = True


print('\ntemperature accepted')
reset()
c = FakeClient(reject_temp=False)
resp, temp0 = call(c)
check('sends temperature=0 when the model accepts it', c.calls[0].get('temperature') == 0)
check('reports temp0=True', temp0 is True)
check('does not retry', len(c.calls) == 1, f'{len(c.calls)} call(s)')
check('preserves the original config exactly', resp == 'RESPONSE')

print('\ntemperature rejected')
reset()
c = FakeClient(reject_temp=True)
resp, temp0 = call(c)
check('retries without temperature', len(c.calls) == 2, f'{len(c.calls)} call(s)')
check('first attempt carried temperature', 'temperature' in c.calls[0])
check('retry omits temperature', 'temperature' not in c.calls[1])
check('reports temp0=False so the CSV records the deviation', temp0 is False)
check('still returns a usable response', resp == 'RESPONSE')
check('retry keeps json_object response_format',
      c.calls[1].get('response_format') == {'type': 'json_object'})
check('retry keeps the judge prompt intact',
      c.calls[1]['messages'][0]['content'] == 'SYSTEM')

print('\nflag is remembered (no repeated failed calls across 270 scenarios)')
reset()
c = FakeClient(reject_temp=True)
call(c)
n_after_first = len(c.calls)
call(c)
call(c)
check('later calls skip the temperature attempt entirely',
      len(c.calls) == n_after_first + 2, f'{len(c.calls)} total call(s)')
check('no later call carries temperature',
      not any('temperature' in k for k in c.calls[n_after_first:]))

print('\nrefuses to swallow unrelated failures')
for label, text in [('auth', '401 Incorrect API key provided'),
                    ('rate limit', '429 Rate limit reached'),
                    ('bad model id', '404 The model `nope` does not exist')]:
    reset()
    c = FakeClient(always_raise=True, error_text=text)
    try:
        call(c)
        check(f'{label} error propagates', False, 'it was swallowed')
    except RuntimeError as e:
        check(f'{label} error propagates', text in str(e))
    check(f'{label} does not disable temperature for the rest of the run',
          J._SUPPORTS_TEMPERATURE is True)

print('\nprompt integrity — the arm must differ from the original only as declared')
check('JUDGE_ORIGINAL is byte-identical to benchmark.py',
      J.JUDGE_ORIGINAL == __import__('benchmark').JUDGE_SYSTEM)

print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
sys.exit(1 if FAIL else 0)
