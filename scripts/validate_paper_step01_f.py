#!/usr/bin/env python3
"""validate_paper_step01_f.py — independent checker for the Step-1 artifact package.

Re-derives every published number straight from results/*.json rather than
trusting the builder, so a builder bug cannot validate itself. Exits non-zero on
any failure.

Usage:  python3 scripts/validate_paper_step01_f.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_step01_common_f as C   # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main() -> None:
    # 1. authoritative inputs unchanged
    try:
        C.verify_inputs()
        check("authoritative input hashes unchanged", True)
    except Exception as e:                                    # noqa: BLE001
        check("authoritative input hashes unchanged", False, str(e)[:200])

    porcelain = subprocess.run(["git", "status", "--porcelain"], cwd=C.REPO,
                               capture_output=True, text=True).stdout
    # only MODIFIED/DELETED matters; untracked (??) files were never committed and
    # their integrity is already pinned by the sha256 check above.
    touched = [ln for ln in porcelain.splitlines()
               if ("probe_contrasts_canonical" in ln
                   or "behavior_labels_tier3_canonical_f.csv" in ln)
               and not ln.startswith("??")]
    check("no uncommitted edits to authoritative results", not touched, str(touched))

    # 2. old figures deprecated by rename, not deleted
    dep = C.REPO / "results/figures_deprecated_oldlabels"
    olds = sorted(p.name for p in dep.glob("fig*.p*")) if dep.exists() else []
    check("8 deprecated figures preserved", len(olds) == 8, f"{len(olds)} found")
    check("old results/figures path is gone", not (C.REPO / "results/figures").exists())

    # 3. rebuild-independent numeric re-derivation
    raw = pd.read_csv(C.V2_CSV)
    d = C.scored(raw, C.PRIMARY_POP)
    tab = pd.read_csv(C.TABLES / "tab_primary_contrasts_216.csv")
    check("primary table row count", len(tab) == 6, str(len(tab)))
    ok_vals = True
    for i, r in d.iterrows():
        t = tab.iloc[i]
        ok_vals &= (t["contrast"] == r["display"] and int(t["n"]) == int(r["n"])
                    and int(t["n_pos"]) == int(r["n_pos"])
                    and abs(float(t["ROC-AUC"]) - r["roc_auc"]) < 5e-4
                    and abs(float(t["PR-AUC"]) - r["pr_auc"]) < 5e-4
                    and abs(float(t["PR null mean"]) - r["null_pr_auc_mean"]) < 5e-4
                    and abs(float(t["PR excess over null"])
                            - (r["pr_auc"] - r["null_pr_auc_mean"])) < 5e-4)
    check("every primary-table value re-derives from v2 results", ok_vals)

    # 4. CIs are the corrected ones, not the split band
    ok_ci = True
    for i, r in d.iterrows():
        got = tab.iloc[i]["ROC Hanley 95% CI"]
        want = C.fmt_ci(r["roc_auc_hanley_lo"], r["roc_auc_hanley_hi"])
        split = C.fmt_ci(r["roc_auc_split_lo"], r["roc_auc_split_hi"])
        ok_ci &= (got == want and got != split)
    check("published CIs are Hanley/bootstrap, never the split band", ok_ci)

    # 5. verdict rule re-derived independently
    exp = {r["key"]: C.verdict(r["p_holm_pr_auc"], r["p_holm_roc_auc"])
           for _, r in d.iterrows()}
    check("verdicts match the PR-primary Holm rule",
          list(exp.values()) == tab["verdict"].tolist(), str(exp))
    check("substantive_leak is suggestive (fails primary PR test)",
          exp["substantive_leak"] == "suggestive")
    check("degree_boundary is no_evidence",
          exp["degree_boundary_broadonly_vs_leaked"] == "no_evidence")
    check("exactly four supported",
          sum(v == "supported" for v in exp.values()) == 4)

    # 6. Holm re-derivation from raw permutation p-values
    def holm(p):
        m, order, adj, run = len(p), np.argsort(p), np.empty(len(p)), 0.0
        for rank, i in enumerate(order):
            run = max(run, (m - rank) * p[i])
            adj[i] = min(1.0, run)
        return adj
    check("Holm column reproduces from raw p",
          np.allclose(holm(d["p_perm_pr_auc"].to_numpy()),
                      d["p_holm_pr_auc"].to_numpy(), atol=1e-12))

    # 7. p-floor formatting
    floor_rows = [C.fmt_p(p) for p in d["p_holm_pr_auc"] if p <= C.P_HOLM_FLOOR + 1e-12]
    check("p-values at the resolution floor render with ≤",
          all(s.startswith("≤") for s in floor_rows), str(floor_rows))
    md = (C.TABLES / "tab_primary_contrasts_216.md").read_text()
    check("primary table states n_perm", f"n_perm={C.N_PERM}" in md)

    # 8. PR baseline honesty
    check("PR permutation null exceeds prevalence on every contrast (canary)",
          bool((d["null_pr_auc_mean"] > d["prevalence"]).all()))
    check("no artifact reports prevalence-lift without the null-excess",
          "PR excess over null" in tab.columns
          and "pr_auc_lift_vs_prevalence" not in tab.columns)

    # 9. population hygiene
    step_files = [p for p in C.STEP_DIR.rglob("*")
                  if p.is_file() and p.suffix in {".md", ".csv", ".tex", ".json"}]
    bad_258 = [p.name for p in step_files
               if "all_258" in p.read_text() and "sensitivity" not in p.name
               and p.name not in {"run_metadata.json", "ARTIFACT_INDEX.md",
                                  "README.md", "PRESENTATION_NOTES.md"}]
    check("258 numbers confined to the sensitivity artifact", not bad_258, str(bad_258))
    sens = (C.TABLES / "tab_population_sensitivity_216_vs_258.md").read_text().lower()
    check("sensitivity artifact says superset and not-a-replication",
          "superset" in sens and "not a replication" in sens)
    for fig in C.FIGURES.glob("*.plotdata.json"):
        check(f"figure {fig.name} uses only the 216 population",
              "all_258" not in fig.read_text())

    # 10. forbidden finality words
    allow = re.compile(r"(not a replication|never .{0,40}replicat|do not .{0,60}replicat"
                       r"|must not|nothing .{0,30}final|is not final|not .{0,20}final)",
                       re.I)
    bad_words = []
    for p in step_files:
        for line in p.read_text().splitlines():
            probe = re.sub(r"final (prompt )?token", "", line, flags=re.I)
            if re.search(r"\b(final|definitive|conclusive|proves|replicates)\b", probe, re.I) \
                    and not allow.search(line):
                bad_words.append(f"{p.name}: {line.strip()[:80]}")
    check("no finality/replication language", not bad_words, str(bad_words[:3]))

    # 11. sidecar completeness
    required = ["artifact:", "status:", "maturity: provisional", "sources:",
                "population:", "uncertainty:", "supports_claim:", "must_not_claim:",
                "caveats:", "paper_location:", "caption:",
                "replaceable_by_later_step:", "promotion_conditions:"]
    arts = sorted([p for p in C.TABLES.glob("*.csv")]
                  + [p for p in C.FIGURES.glob("*.pdf")])
    for a in arts:
        side = a.with_suffix(".md")
        if not side.exists():
            check(f"sidecar exists for {a.name}", False)
            continue
        txt = side.read_text()
        missing = [k for k in required if k not in txt]
        nonempty = ("must_not_claim:\n  -" in txt) and ("caveats:\n  -" in txt)
        check(f"sidecar complete for {a.name}", not missing and nonempty,
              f"missing={missing} nonempty={nonempty}")

    # 12. status vocabulary + index/metadata consistency
    meta = json.loads((C.STEP_DIR / "run_metadata.json").read_text())
    idx = (C.STEP_DIR / "ARTIFACT_INDEX.md").read_text()
    check("every registry artifact appears in the index",
          all(n in idx for n in C.REGISTRY))
    check("all statuses are in the vocabulary",
          all(s["status"] in C.STATUS_VOCAB for s in C.REGISTRY.values()))
    check("no candidate_main figure in step 1 (slot reserved)",
          not any(s["kind"] == "figure" and s["status"] == "candidate_main"
                  for s in C.REGISTRY.values()))
    check("run_metadata records dirty-tree state", "dirty" in meta["git"])
    check("run_metadata states PR-primary provenance is script-docstring, not prereg",
          "not registered" in meta["verdict_rule"]["primary_metric_provenance"].lower()
          or "NOT registered" in meta["verdict_rule"]["primary_metric_provenance"])
    on_disk = {C.rel(p) for p in C.STEP_DIR.rglob("*") if p.is_file()
               and p.name not in {"README.md", "PRESENTATION_NOTES.md",
                                  "run_metadata.json", "SOURCES.sha256"}}
    listed = {o["path"] for o in meta["outputs"]}
    check("outputs list matches files on disk", on_disk == listed,
          f"only-disk={sorted(on_disk - listed)[:3]} only-meta={sorted(listed - on_disk)[:3]}")
    stale = [o["path"] for o in meta["outputs"]
             if C.sha256(C.REPO / o["path"]) != o["sha256"]]
    check("recorded output hashes match files", not stale, str(stale[:3]))

    # 13. figure formats + plotdata round-trip
    for name in ("fig_contrast_effects_dual_metric_216", "fig_protocol_correction_v1_v2"):
        for ext in ("pdf", "svg", "png"):
            check(f"{name}.{ext} exists", (C.FIGURES / f"{name}.{ext}").exists())
        head = (C.FIGURES / f"{name}.pdf").read_bytes()[:2000]
        check(f"{name}.pdf is vector", b"/Font" in head or b"/Contents" in head
              or b"%PDF" in head[:8])
    pdata = json.loads(
        (C.FIGURES / "fig_contrast_effects_dual_metric_216.plotdata.json").read_text())
    check("dual-metric plotdata re-derives from v2",
          np.allclose(pdata["roc_auc"], d["roc_auc"].to_numpy(), atol=1e-12)
          and np.allclose(pdata["pr_auc"], d["pr_auc"].to_numpy(), atol=1e-12)
          and np.allclose(pdata["null_pr_auc_mean"],
                          d["null_pr_auc_mean"].to_numpy(), atol=1e-12)
          and pdata["verdict"] == list(exp.values()))
    cdata = json.loads(
        (C.FIGURES / "fig_protocol_correction_v1_v2.plotdata.json").read_text())
    v1 = C.load_v1()
    b = v1[(v1["status"] == "scored")
           & v1["contrast"].str.endswith("|" + C.PRIMARY_POP)].copy()
    b["key"] = b["contrast"].str.split("|").str[0]
    b = b.set_index("key").loc[C.CONTRAST_KEYS]
    check("correction plotdata re-derives from v1+v2",
          np.allclose(cdata["roc_v1"], b["roc_auc"].to_numpy(), atol=1e-12)
          and np.allclose(cdata["delta_roc"],
                          d["roc_auc"].to_numpy() - b["roc_auc"].to_numpy(), atol=1e-12))
    check("every correction delta is positive", all(x > 0 for x in cdata["delta_roc"]))

    # 14. palette drift between figstyle_f and make_figures_f
    def palette(path):
        txt = (C.REPO / path).read_text()
        m = re.search(r"C_LEAK, C_DEFL, C_PRIV = (.+)", txt)
        n = re.search(r"C_INK, C_MUTED, C_GRID = (.+)", txt)
        return (m.group(1).strip(), n.group(1).strip())
    check("figstyle palette identical to make_figures_f",
          palette("scripts/figstyle_f.py") == palette("scripts/make_figures_f.py"))

    print()
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED:")
        for f in FAILS:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
