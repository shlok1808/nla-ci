"""Guards for the independent Opus source/reference pipeline.

Run: python3 tests/test_opus_reference_audit_f.py
"""

import json
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import opus_reference_audit_f as O  # noqa: E402


PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}"
          + (f" — {detail}" if detail else ""))


print("\nblinding and stage isolation")
scenarios = O.load_scenarios()
check("sample has exactly 50 rows", len(scenarios) == 50, str(len(scenarios)))
check("loader exposes only scenario_id and scenario",
      set(scenarios.columns) == {"scenario_id", "scenario"})
for banned in ("response", "label", "judge_reasoning", "confidence"):
    check(f"loader does not expose {banned}", banned not in scenarios.columns)

known_ids = ("254", "328", "337", "351", "456", "457")
prompt_blob = O.SOURCE_AUDIT_SYSTEM + O.REFERENCE_SYSTEM
check("known defect IDs are absent from both prompts",
      not any(sid in prompt_blob for sid in known_ids))
check("source audit prompt does not contain the sol reference",
      "gpt-5.6-sol" not in O.SOURCE_AUDIT_SYSTEM.lower())
check("reference prompt does not mention source-audit output",
      "source_status" not in O.REFERENCE_SYSTEM
      and "issue_codes" not in O.REFERENCE_SYSTEM)

first = scenarios.iloc[0]
user = O.build_user_message(str(first.scenario))
check("scenario ID is absent from API user message", str(first.scenario_id) not in user)
check("API user message contains only the wrapped scenario",
      user == f"<scenario>\n{str(first.scenario).strip()}\n</scenario>")


print("\nsource-audit validation")
story = "A told B a secret. B promised not to tell C. What should B say to C?"
valid = {
    "source_status": "valid", "issue_codes": [],
    "speaker_has_access": "yes", "protected_target_clear": "yes",
    "recipient_already_knows_entire_target": "no",
    "authorization_clear": "yes", "disclosure_opportunity_valid": "yes",
    "evidence_quotes": [], "explanation": "The information flow is coherent.",
}
try:
    O.validate_source_audit(valid, story)
    check("accepts coherent valid audit", True)
except Exception as exc:
    check("accepts coherent valid audit", False, str(exc))

invalid = {
    **valid, "source_status": "invalid",
    "issue_codes": ["speaker_lacks_knowledge"],
    "speaker_has_access": "no", "disclosure_opportunity_valid": "no",
    "evidence_quotes": ["B promised not to tell C"],
}
try:
    O.validate_source_audit(invalid, story)
    check("accepts issue with exact evidence", True)
except Exception as exc:
    check("accepts issue with exact evidence", False, str(exc))

for name, bad in (
    ("non-exact evidence", {**invalid, "evidence_quotes": ["not in story"]}),
    ("invalid without issue", {**invalid, "issue_codes": []}),
    ("valid with issue", {**valid, "issue_codes": ["other"]}),
):
    try:
        O.validate_source_audit(bad, story)
        check(f"refuses {name}", False)
    except ValueError:
        check(f"refuses {name}", True)


print("\nAPI payload and tool parsing")


class FakeResponse:
    def __init__(self, body):
        self.body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


captured = {}


def opener(request, timeout):
    captured["headers"] = dict(request.header_items())
    captured["payload"] = json.loads(request.data)
    return FakeResponse({
        "id": "msg_test", "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "content": [{"type": "tool_use", "name": "record_source_audit",
                     "input": valid}],
    })


result, provenance = O.anthropic_call(
    "source-audit", story, "claude-opus-5", "secret-test-key", opener=opener
)
check("parses forced tool input", result == valid)
check("records non-secret response provenance", provenance["response_id"] == "msg_test")
check("forces the requested tool",
      captured["payload"]["tool_choice"]
      == {"type": "tool", "name": "record_source_audit"})
check("pins the declared reasoning effort",
      captured["payload"]["output_config"] == {"effort": O.EFFORT})
serialized = json.dumps(captured["payload"])
check("API payload contains no sol reference", "gpt-5.6-sol" not in serialized.lower())
check("API payload contains no old labels or Qwen response",
      all(x not in serialized for x in ('"label"', '"response"', "judge_reasoning")))
check("API key is not placed in payload", "secret-test-key" not in serialized)


print("\ncomparison is explicitly non-semantic")
check("comparison output names lexical scores as sorting-only",
      "lexical_similarity_for_sorting_only" in inspect.getsource(O.compare_one))

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
