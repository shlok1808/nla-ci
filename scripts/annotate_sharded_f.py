#!/usr/bin/env python3
"""Run the behavior judge in parallel shards, then merge into one batch file.

The runner is serial (~30 s/case), so 258 cases take ~2 h. Scenario IDs are
independent -- the judge sees one reference and one response per call and never
looks across cases -- so the work shards cleanly. Each shard filters with
--scenario-id and writes its own output file, which removes any write race;
the merge then verifies the shards agree.

Safety properties this preserves:
  * identical prompt, model, effort and rubric in every shard, so the merged
    batch is indistinguishable from a serial run except for wall-clock;
  * every shard carries the same run_config_sha256 -- the merge asserts it;
  * per-shard resume, so an interrupted run re-does only its own gaps;
  * the merge asserts full coverage and no duplicate scenario IDs.

Usage:
    export OPENAI_API_KEY=...  NLA_MODEL_TAG=llama31_8b
    python scripts/annotate_sharded_f.py \
        --references results/references_verified_f.json \
        --output results/behavior_annotations_llama31_8b_calib_f.json \
        --shards 6
    python scripts/annotate_sharded_f.py \
        --references results/references_tier3_corrected_adjudicated_f.json \
        --output results/behavior_annotations_llama31_8b_analysis_f.json \
        --shards 8 --allow-unverified --expected-count 216

Start conservatively (4-6). Concurrent reasoning-model calls can hit rate
limits; the client retries with backoff, so the failure mode is slowdown rather
than data loss, but more shards is not always faster.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RUNNER = Path(__file__).with_name("behavior_annotation_run_f.py")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shards", type=int, default=6)
    p.add_argument("--expected-count", type=int, default=42)
    p.add_argument("--allow-unverified", action="store_true")
    p.add_argument("--model", default=None)
    p.add_argument("--effort", default=None)
    p.add_argument("--keep-shards", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("set OPENAI_API_KEY")
    ids = sorted(int(x["scenario_id"]) for x in json.loads(args.references.read_text()))
    n = max(1, min(args.shards, len(ids)))
    # round-robin so every shard gets a mix of short and long responses
    shards = [ids[i::n] for i in range(n)]
    shard_dir = args.output.parent / (args.output.stem + "_shards")
    shard_dir.mkdir(parents=True, exist_ok=True)

    procs, paths = [], []
    started = time.time()
    for i, chunk in enumerate(shards):
        out = shard_dir / f"shard{i:02d}.json"
        paths.append(out)
        cmd = [sys.executable, str(RUNNER), "--references", str(args.references),
               "--output", str(out), "--expected-count", str(args.expected_count)]
        if args.allow_unverified:
            cmd.append("--allow-unverified")
        if args.model:
            cmd += ["--model", args.model]
        if args.effort:
            cmd += ["--effort", args.effort]
        for sid in chunk:
            cmd += ["--scenario-id", str(sid)]
        log = shard_dir / f"shard{i:02d}.log"
        procs.append((i, subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT), log, len(chunk)))
        print(f"shard {i}: {len(chunk)} cases -> {out.name}", flush=True)

    print(f"\n{n} shards running; tail a log with:  tail -f {shard_dir}/shard00.log\n", flush=True)
    failed = []
    for i, proc, log, k in procs:
        rc = proc.wait()
        status = "ok" if rc == 0 else f"EXIT {rc}"
        print(f"shard {i} ({k} cases): {status}  [{time.time()-started:.0f}s]", flush=True)
        if rc != 0:
            failed.append((i, log))
    if failed:
        print("\nFAILED SHARDS -- inspect logs, then re-run this command (shards resume):")
        for i, log in failed:
            print(f"  shard {i}: {log}")
        raise SystemExit(1)

    merged, configs = {}, set()
    for out in paths:
        for row in json.loads(out.read_text()):
            sid = int(row["scenario_id"])
            if sid in merged:
                raise SystemExit(f"duplicate scenario {sid} across shards")
            merged[sid] = row
            configs.add(row.get("run_config_sha256"))
    if len(configs) != 1:
        raise SystemExit(f"shards used different run configurations: {configs}")
    missing = sorted(set(ids) - set(merged))
    if missing:
        raise SystemExit(f"merged batch is missing {len(missing)} scenarios: {missing[:10]}")

    combined = [merged[s] for s in sorted(merged)]
    args.output.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n")
    ok = sum(1 for x in combined if x.get("status") == "ok")
    hr = sum(1 for x in combined if (x.get("annotation") or {}).get("assessment_status") == "human_review")
    print(f"\nwrote {args.output}: {len(combined)} cases, {ok} ok, {len(combined)-ok} errors, "
          f"{hr} human_review  [{time.time()-started:.0f}s total]")
    print(f"run_config_sha256: {configs.pop()}")
    if not args.keep_shards:
        print(f"shard files kept at {shard_dir} (delete once the merge is verified)")


if __name__ == "__main__":
    main()
