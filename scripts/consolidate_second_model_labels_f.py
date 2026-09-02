#!/usr/bin/env python3
"""Consolidate a second subject model's v1.3 judge annotations into the
canonical label table the Step 3 pipeline consumes.

Why a separate script. consolidate_canonical_tier3_labels_f.py replays the
Qwen annotation history: a v1.2 calibration batch, v1.3 failure replacements,
and manual adjudications for specific scenario IDs. None of that applies to a
new model, whose labels come from two clean v1.3 batches (42 cases judged
against human-verified references, 216 against adjudicated-but-unverified
ones). This script does only what is generic:

  * concatenate the two batches and check every record is a completed
    v1.3 annotation for a tier-3 scenario;
  * exclude rows the judge could not assess (status != ok, or
    assessment_status != assessable) -- the judge-determined analogue of the
    12 Qwen exclusions, recorded with reasons, never hand-adjusted;
  * tag population (calibration = the verified-reference IDs);
  * write the canonical JSON, the 14-column CSV in the exact Qwen schema, and
    a manifest with counts, hashes, and the exclusion list.

No annotation content is modified. There is no manual adjudication path.

Usage:
    NLA_MODEL_TAG=llama31_8b python scripts/consolidate_second_model_labels_f.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_registry_f as _registry  # noqa: E402

RUBRIC_VERSION = "ci-disclosure-v1.3.0"
TIER3_IDS = set(range(206, 476))
FIELDS = ["scenario_id", "population", "assessment_status", "max_disclosure_level", "response_strategy",
          "broad_breach", "substantive_leak", "label_broad", "label_substantive",
          "broad_onset_start_char", "substantive_onset_start_char", "source_rubric_version",
          "reference_verification", "canonicalization_status"]


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    spec = _registry.get(os.environ.get("NLA_MODEL_TAG"))
    if spec.tag == _registry.DEFAULT_TAG:
        raise SystemExit("Qwen's canonical labels are built by consolidate_canonical_tier3_labels_f.py; "
                         "set NLA_MODEL_TAG to a second model.")
    P = _registry.paths(spec)
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", type=Path, default=Path(f"results/behavior_annotations_{spec.tag}_calib_f.json"))
    ap.add_argument("--analysis", type=Path, default=Path(f"results/behavior_annotations_{spec.tag}_analysis_f.json"))
    ap.add_argument("--verified-references", type=Path, default=Path("results/references_verified_f.json"))
    args = ap.parse_args()

    calib = json.loads(args.calibration.read_text())
    analysis = json.loads(args.analysis.read_text())
    verified_ids = {int(x["scenario_id"]) for x in json.loads(args.verified_references.read_text())}
    if {int(x["scenario_id"]) for x in calib} != verified_ids:
        raise ValueError("calibration batch IDs do not equal the human-verified reference IDs")
    if {int(x["scenario_id"]) for x in analysis} & verified_ids:
        raise ValueError("analysis batch overlaps the calibration IDs")

    combined, excluded = [], []
    for row in sorted(calib + analysis, key=lambda x: int(x["scenario_id"])):
        sid = int(row["scenario_id"])
        if sid not in TIER3_IDS:
            raise ValueError(f"scenario {sid} is not a tier-3 ID")
        a = row.get("annotation") or {}
        if row.get("status") != "ok" or not a:
            excluded.append({"scenario_id": sid, "reason": f"status={row.get('status')!r}", "error": row.get("error")})
            continue
        if a.get("rubric_version") != RUBRIC_VERSION:
            raise ValueError(f"scenario {sid}: rubric {a.get('rubric_version')!r} != {RUBRIC_VERSION}")
        if a.get("assessment_status") != "assessable":
            excluded.append({"scenario_id": sid, "reason": f"assessment_status={a.get('assessment_status')!r}",
                             "review_reason": a.get("review_reason")})
            continue
        row = dict(row)
        row["canonicalization"] = {"status": "none", "note": "second-model batch; no adjudication"}
        combined.append(row)

    ids = [int(x["scenario_id"]) for x in combined]
    ex_ids = {e["scenario_id"] for e in excluded}
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate scenario IDs after consolidation")
    # The reference set itself already excludes some tier-3 scenarios: no
    # reference was ever written for them, so no model can be judged on them.
    # Those are inherited exclusions, not this model's, and are recorded as
    # such. The judged universe is the reference set, not all of tier 3.
    referenced = {int(x["scenario_id"]) for x in json.loads(args.calibration.read_text())} | \
                 {int(x["scenario_id"]) for x in json.loads(args.analysis.read_text())}
    not_referenced = sorted(TIER3_IDS - referenced)
    if set(ids) | ex_ids != referenced:
        missing = sorted(referenced - set(ids) - ex_ids)
        raise ValueError(f"canonical plus excluded IDs do not reconstruct the reference set; missing {missing[:10]}")

    P["annotations"].write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n")
    with P["canonical"].open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=FIELDS)
        w.writeheader()
        for row in combined:
            a, sid = row["annotation"], int(row["scenario_id"])
            w.writerow({
                "scenario_id": sid, "population": "calibration" if sid in verified_ids else "analysis",
                "assessment_status": a["assessment_status"], "max_disclosure_level": a["max_disclosure_level"],
                "response_strategy": a["response_strategy"], "broad_breach": a["broad_breach"],
                "substantive_leak": a["substantive_leak"], "label_broad": a["label_broad"],
                "label_substantive": a["label_substantive"], "broad_onset_start_char": a["broad_onset_start_char"],
                "substantive_onset_start_char": a["substantive_onset_start_char"],
                "source_rubric_version": row.get("rubric_version"),
                "reference_verification": row.get("reference_verification", "provisional_unverified"),
                "canonicalization_status": "none",
            })
    n_cal = sum(1 for s in ids if s in verified_ids)
    manifest_path = P["canonical"].with_name(P["canonical"].stem.replace("_f", "") + "_manifest_f.json")
    manifest = {
        "model_tag": spec.tag, "model_id": spec.model_id,
        "tier3_total": len(TIER3_IDS), "canonical_count": len(ids),
        "calibration_count": n_cal, "analysis_count": len(ids) - n_cal,
        "excluded_count": len(excluded), "excluded": excluded,
        "not_referenced_count": len(not_referenced), "not_referenced": not_referenced,
        "not_referenced_note": ("Tier-3 scenarios with no reference sheet in either batch. "
                                "Inherited from the reference construction, identical for every "
                                "subject model; not an exclusion this model's judge made."),
        "rubric_version": RUBRIC_VERSION,
        "calibration_source": str(args.calibration), "analysis_source": str(args.analysis),
        "canonical_json_sha256": sha256_file(P["annotations"]),
        "canonical_csv_sha256": sha256_file(P["canonical"]),
        "methodology_note": ("Calibration references are human-verified; analysis references are "
                             "machine-drafted and adjudicated but unverified (same limitation as Qwen). "
                             "Exclusions are judge-determined; no manual adjudication was applied."),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("model_tag", "canonical_count", "calibration_count",
                                                "analysis_count", "excluded_count", "not_referenced_count")}, indent=2))
    print("excluded:", [(e["scenario_id"], e["reason"]) for e in excluded])
    print(f"wrote {P['annotations']}, {P['canonical']}, {manifest_path}")


if __name__ == "__main__":
    main()
