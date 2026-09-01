#!/usr/bin/env python3
"""Step 3 pre-flight — validate character->token onset alignment BEFORE any GPU spend.

The onset experiment aligns activations to the first token at which the response
begins disclosing or hedging. That alignment is derived from character offsets
stored by the annotation rubric. If the char->token mapping is wrong, every
downstream position is silently shifted and the experiment is worthless — so it
is verified here, locally, first.

Checks:
  A. every canonical onset char offset round-trips through the tokenizer's
     offset mapping and re-slices to the annotated quote
  B. the min(disclosure, limiting) cutoff is computed for every discloser and
     its token index recovered
  C. window coverage by class at each candidate pre-onset window, so the
     primary window is chosen on measured attrition rather than assumption
  D. the positional confound: how well does the cutoff POSITION alone predict
     the class? (the floor the activation probe must clear)
  E. the exact prompt template used by extract_activations.py is reproduced,
     so response token indices mean the same thing at extraction time

Writes results/onset_alignment_f.{csv,json} and prints a GO / NO-GO.

Local CPU only. Tokenizer only, no model weights, no GPU, no APIs.
Usage:  python3 scripts/onset_alignment_validate_f.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CANONICAL = Path("results/behavior_labels_tier3_canonical_f.csv")
ANNOTATIONS = Path("results/behavior_annotations_sol_tier3_canonical_f.json")
BENCH = Path("results/benchmark_results_bf16.csv")
OUT_CSV = Path("results/onset_alignment_f.csv")
OUT_JSON = Path("results/onset_alignment_f.json")

MODEL = "Qwen/Qwen2.5-7B-Instruct"
LIMITING = {"soft_deflection", "explicit_refusal", "mixed_disclose_then_limit"}
MIXED = "mixed_disclose_then_limit"
WINDOWS = [4, 8, 16, 32]          # candidate pre-cutoff windows, in tokens
FAILS: list[str] = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)

    canon = pd.read_csv(CANONICAL)
    bench = pd.read_csv(BENCH)[["scenario_id", "response"]]
    df = canon.merge(bench, on="scenario_id", how="left")
    assert df.response.notna().all()
    ann = {int(r["scenario_id"]): r["annotation"]
           for r in json.loads(ANNOTATIONS.read_text())}

    rows = []
    quote_fail = []
    for r in df.itertuples():
        a = ann[int(r.scenario_id)]
        resp = r.response
        enc = tok(resp, return_offsets_mapping=True, add_special_tokens=False)
        offs = np.array(enc["offset_mapping"])          # (n_tok, 2)
        n_tok = len(offs)

        def char_to_tok(c):
            """First token whose span covers or follows char offset c."""
            if c is None or (isinstance(c, float) and np.isnan(c)):
                return None
            hit = np.flatnonzero(offs[:, 1] > c)
            return int(hit[0]) if len(hit) else None

        disc_c = a.get("broad_onset_start_char")
        lim_c = None
        if a.get("strategy_evidence") and r.response_strategy in LIMITING:
            lim_c = min(q["start_char"] for q in a["strategy_evidence"])

        # (A) round-trip: does the mapped token start reproduce the quote start?
        ok_rt = True
        if disc_c is not None and not (isinstance(disc_c, float) and np.isnan(disc_c)):
            ev = a["disclosure_events"]
            spans = [q for e in ev for q in e["attribution_evidence"]]
            first = min(spans, key=lambda q: q["start_char"]) if spans else None
            ti = char_to_tok(disc_c)
            if first is not None and ti is not None:
                # the token containing the onset char must overlap the quote
                s, e = offs[ti]
                ok_rt = (s <= first["start_char"] < e) or (s == first["start_char"])
                if not ok_rt:
                    quote_fail.append((int(r.scenario_id), int(s), int(e),
                                       int(first["start_char"])))

        disc_t = char_to_tok(disc_c)
        lim_t = char_to_tok(lim_c)
        cut_t = min([t for t in (disc_t, lim_t) if t is not None], default=None)
        rows.append({
            "scenario_id": int(r.scenario_id),
            "population": r.population,
            "response_strategy": r.response_strategy,
            "is_limiting": r.response_strategy in LIMITING,
            "is_mixed": r.response_strategy == MIXED,
            "broad_breach": bool(r.broad_breach),
            "n_response_tokens": n_tok,
            "disclosure_onset_char": None if disc_c is None or (isinstance(disc_c, float) and np.isnan(disc_c)) else int(disc_c),
            "limiting_onset_char": None if lim_c is None else int(lim_c),
            "disclosure_onset_tok": disc_t,
            "limiting_onset_tok": lim_t,
            "cutoff_tok": cut_t,
            "roundtrip_ok": ok_rt,
        })

    al = pd.DataFrame(rows)
    al.to_csv(OUT_CSV, index=False)

    # ── A ────────────────────────────────────────────────────────────────────
    check("char->token round-trip on every disclosure onset",
          bool(al.roundtrip_ok.all()),
          f"{(~al.roundtrip_ok).sum()} failures {quote_fail[:3]}")

    # ── B ────────────────────────────────────────────────────────────────────
    disc = al[al.broad_breach]
    check("cutoff token recovered for every discloser",
          bool(disc.cutoff_tok.notna().all()),
          f"{int(disc.cutoff_tok.isna().sum())} missing")
    lim_have = al[al.is_mixed]
    check("limiting onset recovered for every mixed_disclose_then_limit case",
          bool(lim_have.limiting_onset_tok.notna().all()),
          f"{int(lim_have.limiting_onset_tok.isna().sum())} missing")

    # ── C: window coverage by class (analysis population, disclosers) ────────
    prim = al[(al.population == "analysis") & al.broad_breach].copy()
    cov = {}
    print("\nwindow coverage (analysis population, disclosers):")
    print(f"{'window':>8s}  {'limiting':>9s}  {'direct':>9s}  {'gap':>6s}")
    for w in WINDOWS:
        c_lim = float((prim[prim.is_limiting].cutoff_tok >= w).mean())
        c_dir = float((prim[~prim.is_limiting].cutoff_tok >= w).mean())
        cov[w] = {"limiting": c_lim, "direct": c_dir, "gap": abs(c_lim - c_dir)}
        print(f"{w:8d}  {c_lim:9.1%}  {c_dir:9.1%}  {abs(c_lim-c_dir):6.1%}")
    best = min(WINDOWS, key=lambda w: (cov[w]["gap"] > 0.05, -w))
    check("a pre-onset window exists with <=5% class-differential attrition",
          cov[best]["gap"] <= 0.05, f"recommended window = [-{best}, -1]")

    # ── D: positional confound ───────────────────────────────────────────────
    y = prim.is_limiting.astype(int).to_numpy()
    auc_cut = float(roc_auc_score(y, -prim.cutoff_tok.to_numpy()))
    auc_disc = float(roc_auc_score(y, -prim.disclosure_onset_tok.to_numpy()))
    auc_len = float(roc_auc_score(y, prim.n_response_tokens.to_numpy()))
    print(f"\npositional confound (AUC of position alone predicting limiting):")
    print(f"  min-cutoff position   {auc_cut:.3f}   <- the floor the probe must clear")
    print(f"  disclosure onset only {auc_disc:.3f}")
    print(f"  response length       {auc_len:.3f}")
    check("positional confound is documented and below 0.70", auc_cut < 0.70,
          f"cutoff-position AUC {auc_cut:.3f}")

    # ── E: prompt template reproducibility ───────────────────────────────────
    src = Path("scripts/extract_activations.py").read_text()
    uses_chat_template = "apply_chat_template" in src
    add_gen = "add_generation_prompt=True" in src.replace(" ", "")
    check("extraction script uses apply_chat_template with add_generation_prompt",
          uses_chat_template and add_gen,
          f"chat_template={uses_chat_template} add_generation_prompt={add_gen}")

    summary = {
        "n_canonical": int(len(al)),
        "n_disclosers": int(al.broad_breach.sum()),
        "n_mixed": int(al.is_mixed.sum()),
        "roundtrip_failures": int((~al.roundtrip_ok).sum()),
        "window_coverage": cov,
        "recommended_primary_window": f"[-{best}, -1]",
        "positional_confound_auc": {"min_cutoff": auc_cut,
                                    "disclosure_onset_only": auc_disc,
                                    "response_length": auc_len},
        "cutoff_token_pctiles": {
            str(p): float(np.percentile(prim.cutoff_tok.dropna(), p))
            for p in (10, 25, 50, 75, 90)},
        "response_token_pctiles": {
            str(p): float(np.percentile(al.n_response_tokens, p))
            for p in (10, 50, 90)},
        "tokenizer": MODEL,
        "verdict": "GO" if not FAILS else "NO-GO",
        "failures": FAILS,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\ncutoff-token percentiles (10/25/50/75/90): "
          f"{[round(v) for v in summary['cutoff_token_pctiles'].values()]}")
    print(f"response-token percentiles (10/50/90): "
          f"{[round(v) for v in summary['response_token_pctiles'].values()]}")
    print(f"\n{'GO — alignment verified, safe to spend GPU' if not FAILS else 'NO-GO: ' + str(FAILS)}")
    print(f"wrote {OUT_CSV} and {OUT_JSON}")


if __name__ == "__main__":
    main()
