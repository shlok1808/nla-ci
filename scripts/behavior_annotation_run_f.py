#!/usr/bin/env python3
"""Run the canonical GPT-5.6-sol behavior judge on the 42-case calibration set.

The API sees only a verified semantic reference and Qwen's response. It never
sees the source scenario, historical GPT-4o-mini label/reasoning, reference
provenance, or another judge's output. Structured output is validated locally;
invalid rows are saved as errors and never converted to an outcome label.

Usage:
    export OPENAI_API_KEY=sk-...
    python3 scripts/behavior_annotation_run_f.py --dry-run
    python3 scripts/behavior_annotation_run_f.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from behavior_annotation_rubric_f import (  # noqa: E402
    ANNOTATION_JSON_SCHEMA,
    JUDGE_REFERENCE_FIELDS,
    RUBRIC_VERSION,
    build_judge_messages,
    prompt_hash,
    validate_annotation,
    validate_reference,
)


REFERENCES = Path("results/references_verified_f.json")
BENCHMARK = Path("results/benchmark_results_bf16.csv")
OUTPUT = Path("results/behavior_annotations_sol_calibration_f.json")
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_EFFORT = "high"
DEFAULT_MAX_COMPLETION_TOKENS = 8000
SAVE_EVERY = 1
RUNNER_VERSION = "behavior-judge-runner-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def runner_source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def semantic_reference(reference: dict[str, Any]) -> dict[str, Any]:
    """Return the only reference fields allowed to reach an annotation model."""
    validate_reference(reference)
    return {field: reference[field] for field in JUDGE_REFERENCE_FIELDS}


def load_calibration() -> list[dict[str, Any]]:
    references = json.loads(REFERENCES.read_text())
    by_id: dict[int, dict[str, Any]] = {}
    for reference in references:
        validate_reference(reference)
        scenario_id = int(reference["scenario_id"])
        if scenario_id in by_id:
            raise ValueError(f"duplicate verified reference for scenario {scenario_id}")
        by_id[scenario_id] = reference
    if len(by_id) != 42:
        raise ValueError(f"expected 42 verified calibration references, got {len(by_id)}")

    # Enforce blinding at load time: old labels, old reasoning, and the source
    # scenario never enter this runner's dataframe.
    benchmark = pd.read_csv(BENCHMARK, usecols=["scenario_id", "response"])
    selected = benchmark.loc[
        benchmark.scenario_id.astype(int).isin(by_id), ["scenario_id", "response"]
    ].copy()
    if len(selected) != len(by_id) or selected.scenario_id.nunique() != len(by_id):
        raise ValueError("verified references do not map one-to-one to benchmark responses")
    if selected.response.isna().any() or not selected.response.astype(str).str.strip().all():
        raise ValueError("calibration contains a missing or blank Qwen response")

    return [
        {
            "scenario_id": int(row.scenario_id),
            "reference": by_id[int(row.scenario_id)],
            "response": str(row.response),
        }
        for row in selected.sort_values("scenario_id").itertuples(index=False)
    ]


def config_hash(model: str, effort: str, max_completion_tokens: int) -> str:
    config = {
        "runner_version": RUNNER_VERSION,
        "runner_source_sha256": runner_source_hash(),
        "rubric_version": RUBRIC_VERSION,
        "rubric_prompt_sha256": prompt_hash(),
        "annotation_schema": ANNOTATION_JSON_SCHEMA,
        "model": model,
        "reasoning_effort": effort,
        "max_completion_tokens": max_completion_tokens,
    }
    return sha256_text(canonical_json(config))


def make_client():
    from openai import OpenAI

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("set OPENAI_API_KEY")
    return OpenAI(api_key=key, max_retries=5, timeout=180.0)


def call_judge(
    client: Any,
    model: str,
    effort: str,
    max_completion_tokens: int,
    reference: dict[str, Any],
    response: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    messages = build_judge_messages(reference, response)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        reasoning_effort=effort,
        max_completion_tokens=max_completion_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "contextual_privacy_annotation",
                "strict": True,
                "schema": ANNOTATION_JSON_SCHEMA,
            },
        },
        store=False,
    )
    content = completion.choices[0].message.content
    if not content:
        refusal = getattr(completion.choices[0].message, "refusal", None)
        raise ValueError(f"judge returned no annotation content; refusal={refusal!r}")
    raw = json.loads(content)
    normalized = validate_annotation(reference, response, raw)
    usage = getattr(completion, "usage", None)
    provenance = {
        "response_id": getattr(completion, "id", None),
        "created": getattr(completion, "created", None),
        "system_fingerprint": getattr(completion, "system_fingerprint", None),
        "finish_reason": getattr(completion.choices[0], "finish_reason", None),
        "usage": usage.model_dump() if usage is not None else None,
    }
    return raw, normalized, provenance


def save_rows(path: Path, rows: dict[int, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps([rows[key] for key in sorted(rows)], indent=2) + "\n")
    temporary.replace(path)


def load_previous(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text())
    rows: dict[int, dict[str, Any]] = {}
    for row in loaded:
        scenario_id = int(row["scenario_id"])
        if scenario_id in rows:
            raise ValueError(f"duplicate saved annotation for scenario {scenario_id}")
        rows[scenario_id] = row
    return rows


def run(
    model: str,
    effort: str,
    max_completion_tokens: int,
    output: Path,
    limit: int | None,
    dry_run: bool,
) -> None:
    items = load_calibration()
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        items = items[:limit]

    run_hash = config_hash(model, effort, max_completion_tokens)
    if dry_run:
        first = items[0]
        visible_reference = semantic_reference(first["reference"])
        message_text = build_judge_messages(first["reference"], first["response"])[1]["content"]
        forbidden = (
            "drafted_by", "verification_note", "draft_error", "verified_by_human",
            "SCENARIO:", "judge_reasoning",
        )
        leaked = [value for value in forbidden if value in message_text]
        if leaked:
            raise AssertionError(f"dry-run prompt leaked forbidden fields: {leaked}")
        print(f"model: {model}")
        print(f"reasoning effort: {effort}")
        print(f"rubric: {RUBRIC_VERSION}")
        print(f"prompt sha256: {prompt_hash()}")
        print(f"run config sha256: {run_hash}")
        print(f"cases: {len(items)}")
        print(f"semantic reference fields: {list(visible_reference)}")
        print("blinding check: PASS (no scenario, old label/reasoning, or provenance)")
        print("dry-run only: no API calls and no output written")
        return

    previous = load_previous(output)
    for row in previous.values():
        if row.get("run_config_sha256") != run_hash:
            raise ValueError(
                f"{output} contains a different run configuration; choose a new --output "
                "instead of mixing arms"
            )

    client = make_client()
    saved = dict(previous)
    pending = []
    for item in items:
        scenario_id = item["scenario_id"]
        reference_sha = sha256_text(canonical_json(semantic_reference(item["reference"])))
        response_sha = sha256_text(item["response"])
        old = saved.get(scenario_id)
        if (
            old
            and old.get("status") == "ok"
            and old.get("reference_sha256") == reference_sha
            and old.get("response_sha256") == response_sha
        ):
            continue
        pending.append((item, reference_sha, response_sha))

    print(f"{len(pending)} to judge | {len(items) - len(pending)} already complete | model={model}")
    for index, (item, reference_sha, response_sha) in enumerate(
        tqdm(pending, desc="behavior judge"), 1
    ):
        scenario_id = item["scenario_id"]
        base = {
            "scenario_id": scenario_id,
            "status": "error",
            "runner_version": RUNNER_VERSION,
            "runner_source_sha256": runner_source_hash(),
            "model": model,
            "reasoning_effort": effort,
            "max_completion_tokens": max_completion_tokens,
            "rubric_version": RUBRIC_VERSION,
            "rubric_prompt_sha256": prompt_hash(),
            "run_config_sha256": run_hash,
            "reference_sha256": reference_sha,
            "response_sha256": response_sha,
            "annotated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            raw_annotation, annotation, provenance = call_judge(
                client, model, effort, max_completion_tokens,
                item["reference"], item["response"],
            )
            saved[scenario_id] = {
                **base,
                "status": "ok",
                "raw_annotation": raw_annotation,
                "annotation": annotation,
                "api_provenance": provenance,
                "error": None,
            }
        except Exception as exc:
            saved[scenario_id] = {
                **base,
                "raw_annotation": None,
                "annotation": None,
                "api_provenance": None,
                "error": f"{type(exc).__name__}: {exc}"[:1000],
            }
        if index % SAVE_EVERY == 0:
            save_rows(output, saved)
    save_rows(output, saved)

    selected_ids = {item["scenario_id"] for item in items}
    selected = [row for sid, row in saved.items() if sid in selected_ids]
    errors = [row for row in selected if row.get("status") != "ok"]
    reviews = [
        row for row in selected
        if row.get("status") == "ok"
        and row["annotation"].get("assessment_status") == "human_review"
    ]
    print(f"wrote {output}: {len(selected) - len(errors)} ok, {len(errors)} errors, "
          f"{len(reviews)} human_review")
    if errors:
        print("error scenario IDs:", [row["scenario_id"] for row in errors])
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default=DEFAULT_EFFORT,
                        choices=["low", "medium", "high", "xhigh", "max", "ultra"])
    parser.add_argument("--max-completion-tokens", type=int,
                        default=DEFAULT_MAX_COMPLETION_TOKENS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int, help="debug only: first N cases")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.model, args.effort, args.max_completion_tokens,
        args.output, args.limit, args.dry_run)


if __name__ == "__main__":
    main()
