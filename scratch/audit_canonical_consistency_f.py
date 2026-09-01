#!/usr/bin/env python3
"""Read-only audit: canonical labels vs JSON vs manifest vs probe results.

Verifies counts, hashes, internal label consistency, join integrity, and
reconstructs every probe contrast's n / n_pos from the canonical CSV.
Writes nothing outside stdout.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results")
csv_p = R / "behavior_labels_tier3_canonical_f.csv"
json_p = R / "behavior_annotations_sol_tier3_canonical_f.json"
man_p = R / "behavior_labels_tier3_canonical_manifest_f.json"
probe_p = R / "probe_contrasts_canonical_f.csv"
npz_p = R / "activations_layer20.npz"

fails = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        fails.append(name)
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


man = json.loads(man_p.read_text())
df = pd.read_csv(csv_p)
raw = json.loads(json_p.read_text())

# --- 1. manifest arithmetic and hashes ---
check("manifest arithmetic 270=258+12",
      man["tier3_total"] == man["canonical_count"] + man["excluded_count"] == 270)
check("manifest split 258=42+216",
      man["canonical_count"] == man["calibration_count"] + man["analysis_count"] == 258)
check("csv sha256 matches manifest", sha(csv_p) == man["canonical_csv_sha256"])
check("json sha256 matches manifest", sha(json_p) == man["canonical_json_sha256"])

# --- 2. CSV counts ---
check("csv rows == 258", len(df) == 258, f"got {len(df)}")
check("csv scenario_id unique", df.scenario_id.is_unique)
pop = df.population.value_counts().to_dict()
check("csv populations 216/42",
      pop.get("analysis") == 216 and pop.get("calibration") == 42, str(pop))
excl = set(man["excluded_scenario_ids"])
check("excluded ids absent from csv", not (set(df.scenario_id) & excl))
check("excluded count == 12", len(excl) == 12)

# --- 3. JSON <-> CSV agreement ---
jrecs = raw if isinstance(raw, list) else raw.get("annotations", raw.get("records", []))
print(f"    json top-level type: {type(raw).__name__}; records: {len(jrecs)}")
jids = {int(r["scenario_id"]) for r in jrecs}
check("json ids == csv ids", jids == set(df.scenario_id.astype(int)),
      f"json-only={sorted(jids - set(df.scenario_id))[:5]} csv-only={sorted(set(df.scenario_id) - jids)[:5]}")
jmap = {int(r["scenario_id"]): r for r in jrecs}
mismatch = []
for _, row in df.iterrows():
    j = jmap[int(row.scenario_id)]
    for col in ("max_disclosure_level", "response_strategy"):
        jv = j.get(col, j.get("annotation", {}).get(col) if isinstance(j.get("annotation"), dict) else None)
        if jv is not None and str(jv) != str(row[col]):
            mismatch.append((row.scenario_id, col, jv, row[col]))
check("json/csv field agreement (disclosure level, strategy)", not mismatch, str(mismatch[:5]))

# --- 4. internal label-derivation consistency ---
# substantive_leak implies broad_breach
check("substantive => broad", not ((df.substantive_leak) & (~df.broad_breach)).any())
# label_broad consistency
lb = np.where(df.broad_breach, "leaked", None)
check("label_broad: broad_breach True -> leaked",
      (df.loc[df.broad_breach, "label_broad"] == "leaked").all())
print("    label_broad values:", df.label_broad.value_counts().to_dict())
print("    label_substantive values:", df.label_substantive.value_counts().to_dict())
print("    response_strategy values:", df.response_strategy.value_counts().to_dict())
print("    max_disclosure_level values:", df.max_disclosure_level.value_counts().to_dict())
# label_substantive: leaked iff substantive_leak; broad_only iff broad and not substantive
check("label_substantive leaked == substantive_leak",
      set(df.loc[df.label_substantive == "leaked"].index) == set(df.loc[df.substantive_leak].index))
check("label_substantive broad_only == broad&~substantive",
      set(df.loc[df.label_substantive == "broad_only"].index)
      == set(df.loc[df.broad_breach & ~df.substantive_leak].index))
nonb = df.loc[~df.broad_breach, "label_substantive"]
check("non-breach rows labeled refused/appropriate",
      nonb.isin(["refused", "appropriate"]).all(), str(nonb.value_counts().to_dict()))
# onset chars present iff breach
check("broad onset present iff broad_breach",
      (df.broad_onset_start_char.notna() == df.broad_breach).all())
check("substantive onset present iff substantive_leak",
      (df.substantive_onset_start_char.notna() == df.substantive_leak).all())
# assessment status
print("    assessment_status:", df.assessment_status.value_counts().to_dict())
print("    reference_verification by population:",
      df.groupby("population").reference_verification.value_counts().to_dict())

# --- 5. NPZ join integrity ---
npz = np.load(npz_p, allow_pickle=True)
ids = np.asarray(npz["scenario_ids"], dtype=int)
check("npz ids unique", len(ids) == len(set(ids.tolist())))
check("all canonical ids present in npz",
      set(df.scenario_id.astype(int)) <= set(ids.tolist()))
stale = np.asarray(npz["labels"])
# how discrepant are stale npz labels vs canonical? (informational)
idx = {s: i for i, s in enumerate(ids)}
stale_for_canon = pd.Series([str(stale[idx[s]]) for s in df.scenario_id.astype(int)], index=df.index)
tab = pd.crosstab(stale_for_canon, df.label_substantive)
print("    stale NPZ label vs canonical label_substantive crosstab:")
print(tab.to_string().replace("\n", "\n    "))

# --- 6. reconstruct probe contrast n / n_pos ---
LIMITING = {"soft_deflection", "explicit_refusal", "mixed_disclose_then_limit"}
probe = pd.read_csv(probe_p)


def recon(sub):
    lim = sub.response_strategy.isin(LIMITING)
    disc = sub.broad_breach.astype(bool)
    return {
        "substantive_leak": (len(sub), int(sub.substantive_leak.sum())),
        "broad_breach": (len(sub), int(sub.broad_breach.sum())),
        "leak_vs_appropriate": (int(sub.label_substantive.isin(["leaked", "appropriate"]).sum()),
                                int(sub.label_substantive.eq("leaked").sum())),
        "degree_boundary_broadonly_vs_leaked": (int(sub.label_substantive.isin(["broad_only", "leaked"]).sum()),
                                                int(sub.label_substantive.eq("leaked").sum())),
        "limiting_vs_direct": (len(sub), int(lim.sum())),
        "limiting_among_disclosers": (int(disc.sum()), int((lim & disc).sum())),
        "refused_vs_appropriate": (int(sub.label_substantive.isin(["refused", "appropriate"]).sum()),
                                   int(sub.label_substantive.eq("refused").sum())),
    }


for popname, sub in (("analysis_216", df[df.population == "analysis"]), ("all_258", df)):
    exp = recon(sub)
    for name, (n, npos) in exp.items():
        row = probe[probe.contrast == f"{name}|{popname}"]
        if row.empty:
            check(f"probe row exists {name}|{popname}", False)
            continue
        rn, rp = int(row.n.iloc[0]), int(row.n_pos.iloc[0])
        check(f"n/n_pos {name}|{popname}", (rn, rp) == (n, npos),
              f"probe=({rn},{rp}) recon=({n},{npos})")

# --- 7. reported headline AUCs vs file ---
expected = {
    ("leak_vs_appropriate", "analysis_216"): 0.756, ("leak_vs_appropriate", "all_258"): 0.796,
    ("limiting_vs_direct", "analysis_216"): 0.703, ("limiting_vs_direct", "all_258"): 0.743,
    ("limiting_among_disclosers", "analysis_216"): 0.714, ("limiting_among_disclosers", "all_258"): 0.741,
    ("broad_breach", "analysis_216"): 0.658, ("broad_breach", "all_258"): 0.656,
    ("substantive_leak", "analysis_216"): 0.619, ("substantive_leak", "all_258"): 0.615,
    ("degree_boundary_broadonly_vs_leaked", "analysis_216"): 0.557,
    ("degree_boundary_broadonly_vs_leaked", "all_258"): 0.587,
}
for (name, popname), want in expected.items():
    row = probe[probe.contrast == f"{name}|{popname}"]
    got = float(row.roc_auc.iloc[0])
    check(f"ROC {name}|{popname} ≈ {want}", abs(got - want) < 0.0005, f"file={got:.4f}")

print()
print("FAILURES:" if fails else "ALL CHECKS PASSED", fails or "")
