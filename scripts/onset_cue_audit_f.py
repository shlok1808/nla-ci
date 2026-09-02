#!/usr/bin/env python3
"""Pre-GPU cue audit for the Step 3 primary window, plus the reviewer record.

What the primary window sees is the response text *before* the registered
cutoff (the earlier of the broad-disclosure onset and the first
`strategy_evidence` quote). Two things must be true of that prefix for the
window to be "pre-onset":

  1. it contains no unrecorded explicit hedge/refusal cue — screened here by a
     deliberately broad regex (false positives are cheap), then confirmed by a
     human who reads every limiting prefix; and
  2. any *tonal* pre-cue it still carries ("I'm really disappointed in you")
     is absorbed by the matched visible-prefix text baselines, which is why
     those baselines — not this screen — are the leakage protection in the
     registered analysis.

Important: for `mixed_disclose_then_limit` cases the stored strategy evidence
quotes the disclosure sentence as well as the limiting cue, so the stored
"limiting onset" is the earliest strategy-evidence quote, not a clean
limiting-language onset. The min() cutoff is therefore conservative (never
later than the disclosure onset), but nothing downstream may describe it as
"limiting-language onset".

Usage:
    python scripts/onset_cue_audit_f.py                 # screen + reviewer sheet
    python scripts/onset_cue_audit_f.py --record-review \
        --reviewed-sheet results/onset_cue_audit_sheet_reviewed_f.csv \
        --reviewer "Name" [--notes "..."]

The reviewer copies `results/onset_cue_audit_sheet_f.csv` to
`..._sheet_reviewed_f.csv`, fills `reviewer_disposition` for EVERY row with
`ok` (no earlier limiting cue in the prefix), `earlier_cue` (an unrecorded
limiting cue precedes the cutoff), or `unsure`, and runs --record-review. The
verdict is GO only when every row is `ok`; the review record binds the sheet
and the candidate CSV by sha256, and the GPU extraction gate checks both.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import pandas as pd

from onset_dynamics_common_f import (
    ALIGNMENT,
    LIMITING,
    PRIMARY_OFFSETS,
    REVIEW_EXCLUDED_IDS,
    SPEC,
    load_step3_rows,
    sha256_file,
)
import model_registry_f as _registry

_P = _registry.paths(SPEC)
_sfx = SPEC.suffix
ANNOTATIONS = _P["annotations"]
OUT = Path("results/" + _sfx("onset_cue_audit_candidates", "csv"))
SHEET = _P["cue_sheet"]
SUMMARY = Path("results/" + _sfx("onset_cue_audit", "json"))
REVIEW = _P["cue_review"]
DISPOSITIONS = {"ok", "earlier_cue", "unsure"}

# Deliberately broad. False positives are cheaper than a leaked primary window.
CUE = re.compile(
    r"\b(?:"
    r"(?:I\s+)?(?:can(?:not|'t)|won't|should(?:n't|\s+not)|would(?:n't|\s+not)|"
    r"(?:am|'m)\s+not\s+(?:going|able)\s+to|don't\s+(?:want|think\s+I\s+should|feel)\s+(?:to\s+)?)\s*"
    r"(?:really\s+)?(?:share|say|tell|disclose|reveal|discuss|talk\s+about|go\s+into|"
    r"get\s+into|provide|comment|speak|answer|repeat|spread|betray|divulge|go\s+there)"
    r"|(?:not|isn't|is\s+not)\s+(?:my\s+place|really\s+my\s+place|appropriate|comfortable|"
    r"my\s+(?:story|business|secret)|(?:the|a)\s+(?:right|good)\s+(?:time|place))"
    r"|(?:respect|protect|honou?r)(?:ing)?\s+(?:their|his|her|your|the)?\s*(?:privacy|confidence|trust|wishes)"
    r"|(?:private|personal|confidential|sensitive|delicate)\s+(?:matter|information|details?|stuff|thing|issue|topic)"
    r"|without\s+(?:sharing|revealing|disclosing|going\s+into|betraying)"
    r"|(?:prefer|rather)\s+not(?:\s+to)?"
    r"|(?:hesitant|reluctant|uncomfortable)\s+(?:to|about|sharing|talking)"
    r"|keep\s+(?:it|this|that|things|this\s+one)\s+(?:between|to\s+myself|private|confidential|in\s+check|quiet|under\s+wraps)"
    r"|between\s+(?:us|me\s+and|him\s+and|her\s+and|them\s+and)"
    r"|(?:told|shared|confided)\s+(?:me|this|that|it)?\s*in\s+confidence"
    r"|in\s+confidence"
    r"|none\s+of\s+(?:your|their|our|my)\s+business"
    r"|not\s+sure\s+(?:I|if\s+I|that\s+I)\s+should"
    r"|wish\s+you\s+hadn't"
    r"|(?:let's|let\s+us)\s+not"
    r"|(?:leave|let's\s+leave)\s+it\s+at\s+that"
    r"|(?:his|her|their)\s+(?:story|place|business|decision|call)\s+to\s+(?:tell|share|make)"
    r"|off\s+limits"
    r"|(?:can't|cannot|won't)\s+really"
    r"|change\s+the\s+subject"
    r"|(?:drop|move\s+on\s+from)\s+(?:it|this|that)"
    r")\b",
    re.IGNORECASE,
)


def _cutoff_char(r) -> int | None:
    vals = [v for v in (r.get("disclosure_onset_char"), r.get("limiting_onset_char")) if pd.notna(v)]
    return int(min(vals)) if vals else None


def screen() -> None:
    raw = json.loads(ANNOTATIONS.read_text())
    ann = {int(x["scenario_id"]): x["annotation"] for x in raw}
    rows = load_step3_rows()
    align = pd.read_csv(ALIGNMENT).set_index("scenario_id")
    candidates, sheet = [], []
    limiting_rows = 0
    for row in rows.itertuples(index=False):
        sid = int(row.scenario_id)
        a = ann[sid]
        strategy = a.get("response_strategy")
        if strategy not in LIMITING:
            continue
        limiting_rows += 1
        evidence = sorted(a.get("strategy_evidence") or [], key=lambda q: int(q["start_char"]))
        stored = int(evidence[0]["start_char"]) if evidence else None
        al = align.loc[sid]
        cutoff_char = _cutoff_char(al)
        response = str(row.response)
        prefix = response[:cutoff_char] if cutoff_char is not None else ""
        hits = [m for m in CUE.finditer(prefix)]
        disclosure_quote, disclosure_quote_start = "", 1 << 30
        for ev in a.get("disclosure_events") or []:
            for q in ev.get("attribution_evidence") or []:
                start = int(q.get("start_char", 1 << 30))
                if start < disclosure_quote_start:
                    disclosure_quote, disclosure_quote_start = q.get("quote", ""), start
        if stored is None:
            candidates.append(
                {
                    "scenario_id": sid, "reason": "missing_strategy_evidence",
                    "candidate_start_char": None, "stored_start_char": None,
                    "candidate_text": "", "prefix_context": response[:240],
                }
            )
        for m in hits:
            candidates.append(
                {
                    "scenario_id": sid, "reason": "possible_earlier_limiting_cue",
                    "candidate_start_char": m.start(), "stored_start_char": cutoff_char,
                    "candidate_text": m.group(0),
                    "prefix_context": response[max(0, m.start() - 80): min(len(response), (cutoff_char or 0) + 80)],
                }
            )
        cutoff_tok = None if pd.isna(al.cutoff_tok) else int(al.cutoff_tok)
        sheet.append(
            {
                "scenario_id": sid,
                "population": row.population,
                "response_strategy": strategy,
                "in_primary_population": bool(row.population == "analysis" and bool(row.broad_breach)),
                "cutoff_char": cutoff_char,
                "cutoff_tok": cutoff_tok,
                "covers_primary_window": bool(cutoff_tok is not None and cutoff_tok + min(PRIMARY_OFFSETS) >= 0),
                "cutoff_source": (
                    "strategy_evidence" if pd.notna(al.limiting_onset_char)
                    and (pd.isna(al.disclosure_onset_char) or al.limiting_onset_char < al.disclosure_onset_char)
                    else "disclosure_onset"
                ),
                "prefix_before_cutoff": prefix,
                "first_strategy_quote": evidence[0]["quote"] if evidence else "",
                "first_disclosure_quote": disclosure_quote,
                "regex_hits": " | ".join(m.group(0) for m in hits),
                "reviewer_disposition": "",
                "reviewer_note": "",
            }
        )
    pd.DataFrame(
        candidates,
        columns=["scenario_id", "reason", "candidate_start_char", "stored_start_char",
                 "candidate_text", "prefix_context"],
    ).to_csv(OUT, index=False)
    pd.DataFrame(sheet).to_csv(SHEET, index=False)
    summary = {
        "status": "REVIEW_REQUIRED",
        "n_limiting_rows": limiting_rows,
        "n_candidates": len(candidates),
        "n_candidate_scenarios": len({x["scenario_id"] for x in candidates}),
        "n_sheet_rows": len(sheet),
        "candidates": str(OUT),
        "candidates_sha256": sha256_file(OUT),
        "sheet": str(SHEET),
        "sheet_sha256": sha256_file(SHEET),
        "interpretation": (
            "Regex candidates require semantic review, and a clean regex screen does "
            "not prove earliestness. Every row of the sheet must receive a reviewer "
            "disposition (--record-review) before GPU extraction. Tonal pre-cues that "
            "are not hedges are expected and are handled by the matched text baselines, "
            "not by this screen."
        ),
        "cutoff_definition": (
            "min(broad_disclosure_onset_char, earliest strategy_evidence start_char); for "
            "mixed cases the strategy evidence includes the disclosure sentence, so this "
            "is NOT a limiting-language onset"
        ),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def record_review(reviewed_sheet: Path, reviewer: str, notes: str) -> None:
    if not SHEET.exists() or not OUT.exists():
        raise SystemExit("run the screen first (no sheet/candidates on disk)")
    generated = pd.read_csv(SHEET)
    reviewed = pd.read_csv(reviewed_sheet)
    key = ["scenario_id", "cutoff_char", "cutoff_tok", "prefix_before_cutoff"]
    g = generated[key].fillna("").astype(str).sort_values("scenario_id").reset_index(drop=True)
    r = reviewed[key].fillna("").astype(str).sort_values("scenario_id").reset_index(drop=True)
    if not g.equals(r):
        raise SystemExit("reviewed sheet does not match the current generated sheet; re-run the screen and re-review")
    disp = reviewed.reviewer_disposition.fillna("").astype(str).str.strip().str.lower()
    bad = reviewed.loc[~disp.isin(DISPOSITIONS), "scenario_id"].tolist()
    if bad:
        raise SystemExit(f"every row needs a disposition in {sorted(DISPOSITIONS)}; missing/invalid for {bad[:10]}...")
    counts = disp.value_counts().to_dict()
    earlier_ids = frozenset(int(x) for x in reviewed.loc[disp.eq("earlier_cue"), "scenario_id"])
    if disp.eq("unsure").any():
        verdict = "NO-GO"
    elif not earlier_ids:
        verdict = "GO"
    elif earlier_ids == REVIEW_EXCLUDED_IDS:
        verdict = "GO_WITH_EXCLUSIONS"
    else:
        verdict = "NO-GO"
    record = {
        "verdict": verdict,
        "reviewer": reviewer,
        "reviewed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_rows": int(len(reviewed)),
        "dispositions": {k: int(v) for k, v in counts.items()},
        "excluded_scenario_ids": sorted(earlier_ids),
        "candidates_sha256": sha256_file(OUT),
        "sheet_sha256": sha256_file(SHEET),
        "sheet_reviewed_path": str(reviewed_sheet),
        "sheet_reviewed_sha256": sha256_file(reviewed_sheet),
        "notes": notes,
        "rule": (
            "GO requires every row 'ok'. GO_WITH_EXCLUSIONS is permitted only "
            "for the frozen reviewer-confirmed exclusion set; raw extraction retains "
            "all scenarios and analysis excludes those IDs rather than hand-placing "
            "new onset boundaries. Any other earlier_cue or unsure is NO-GO."
        ),
    }
    REVIEW.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--record-review", action="store_true")
    p.add_argument("--reviewed-sheet", type=Path, default=Path("results/onset_cue_audit_sheet_reviewed_f.csv"))
    p.add_argument("--reviewer", default="")
    p.add_argument("--notes", default="")
    args = p.parse_args()
    if args.record_review:
        if not args.reviewer:
            raise SystemExit("--reviewer is required to record a review")
        record_review(args.reviewed_sheet, args.reviewer, args.notes)
    else:
        screen()


if __name__ == "__main__":
    main()
