#!/usr/bin/env python3
"""Step 2B — is the limiting signal privacy strategy, or generic eval-awareness?

Step 2A established that the canonical limiting contrasts carry a privileged
increment over text baselines. The obvious alternative explanation: the probe
reads generic evaluation-awareness / caution rather than a privacy-handling
strategy. This script tests that by erasing the eval-awareness direction and
re-probing.

v_eval comes from results/eval_awareness_acts_f.npz — 270 tier-3 scenarios, each
framed 3x test-like and 3x deploy-like with byte-identical user content, so
scenario content cancels in the paired within-scenario difference. It never sees
a limiting label.

CROSS-FITTED THROUGHOUT. v_eval is re-derived inside every training fold from
the framed activations of TRAINING scenarios only, then applied to the held-out
fold. A direction estimated on all data (including test scenarios) would
over-remove on test and bias the result toward "erasure worked", i.e. toward a
false negative for the caution hypothesis.

Arms, per contrast, identical protocol (fixed C, StratifiedKFold(5) x N_REPEATS,
per-fold mean ROC — relative drops are the estimand, so all arms share one
protocol):

  none         original activations
  v_eval       fold-wise rank-1 projection of the eval-awareness direction removed
  null x N     SAME estimation pipeline on side-permuted frames: within each
               scenario, randomly reassign which 3 of its 6 frames count as
               "test". Matched norm, matched manifold alignment, no content.
               This is the null distribution for the drop. Isotropic random
               directions are NOT an adequate null here -- a random vector in
               3584-d is nearly orthogonal to the activation manifold and costs
               almost nothing to erase, so v_eval would "win" trivially.
  iso x N      isotropic random unit directions, reported only to demonstrate
               that weak null and quantify how much easier it is to beat.

Primary test: p = (1 + #{null drops >= v_eval drop}) / (N_NULL + 1) on
limiting_among_disclosers|analysis_216, plus a paired scenario-bootstrap CI on
the drop itself. Specificity: same arms on limiting_vs_direct,
leak_vs_appropriate, substantive_leak.

Manipulation check (also cross-fitted, and with a tuned decoder): test-vs-deploy
decodability on the framed activations before and after erasure, scenario-grouped
so frames of one scenario never straddle a fold, with C selected by nested CV so
the "before" number is not understated by arbitrary regularisation. If
eval-awareness remains decodable after erasure, the rank-1 eraser removed a
direction rather than the concept, and a null result is bounded accordingly
(outcome: inconclusive, not "not caution").

Local CPU only. No APIs, GPU, judges, or label/activation changes.
Usage:  python3 scripts/eval_awareness_canonical_f.py [--quick]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NPZ = Path("results/activations_layer20.npz")
CANONICAL = Path("results/behavior_labels_tier3_canonical_f.csv")
EVAL_NPZ = Path("results/eval_awareness_acts_f.npz")

QUICK = "--quick" in sys.argv
if QUICK:
    OUT_CSV = Path("scratch/eval_awareness_canonical_quick.csv")
    OUT_JSON = Path("scratch/eval_awareness_canonical_quick.json")
    N_REPEATS, N_NULL, N_ISO, N_BOOT = 2, 15, 8, 200
else:
    OUT_CSV = Path("results/eval_awareness_canonical_f.csv")
    OUT_JSON = Path("results/eval_awareness_canonical_f.json")
    N_REPEATS, N_NULL, N_ISO, N_BOOT = 5, 100, 50, 1000

FIXED_C = 1e-4          # modal C for the limiting contrasts in the v2 run
C_GRID = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
N_OUTER = 5
BASE_SEED = 20260901
LIMITING = {"soft_deflection", "explicit_refusal", "mixed_disclose_then_limit"}
PRIMARY = "limiting_among_disclosers"
FRAMES_PER_SCENARIO = 6


def pipe(C=FIXED_C):
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=C, class_weight="balanced",
                                                solver="liblinear", max_iter=5000))])


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def erase(X, u):
    """Remove the rank-1 component along unit vector u."""
    return X - np.outer(X @ u, u)


def paired_direction(acts, sides, scen, keep=None):
    """Mean over scenarios of (mean test frames - mean deploy frames).

    `keep` restricts the estimate to a subset of scenario ids (training folds).
    """
    diffs = []
    for s in np.unique(scen):
        if keep is not None and s not in keep:
            continue
        m = scen == s
        t, d = acts[m & (sides == 1)], acts[m & (sides == 0)]
        if len(t) and len(d):
            diffs.append(t.mean(0) - d.mean(0))
    return unit(np.mean(diffs, axis=0))


def cv_scores(X, y, scen_ids, eacts, esides, escen, side_variants,
              iso_dirs=None):
    """One pass of repeated CV returning, per arm, the per-fold-mean ROC and the
    out-of-fold score vector (for bootstrapping).

    Arms: 'none', 'v_eval', 'null_k' for each side-variant, 'iso_k'.
    Directions are re-derived inside each training fold from the framed
    activations of TRAINING scenarios only.
    """
    arms = ["none", "v_eval"] + [f"null_{k}" for k in range(len(side_variants))]
    if iso_dirs is not None:
        arms += [f"iso_{k}" for k in range(len(iso_dirs))]
    oof = {a: np.zeros((N_REPEATS, len(y))) for a in arms}

    for r in range(N_REPEATS):
        cv = StratifiedKFold(N_OUTER, shuffle=True, random_state=BASE_SEED + r)
        for tr, te in cv.split(X, y):
            train_scen = set(scen_ids[tr].tolist())
            # fold-wise direction estimates (training scenarios only)
            dirs = {"none": None,
                    "v_eval": paired_direction(eacts, esides, escen, keep=train_scen)}
            for k, sv in enumerate(side_variants):
                dirs[f"null_{k}"] = paired_direction(eacts, sv, escen, keep=train_scen)
            if iso_dirs is not None:
                for k, d in enumerate(iso_dirs):
                    dirs[f"iso_{k}"] = d
            for arm, u in dirs.items():
                Xa = X if u is None else erase(X, u)
                m = pipe().fit(Xa[tr], y[tr])
                oof[arm][r, te] = m.predict_proba(Xa[te])[:, 1]
    return oof, arms


def perfold_from_oof(oof_arm, y, seeds_folds):
    """Per-fold-mean ROC averaged over repeats, from stored OOF scores."""
    vals = []
    for r, folds in enumerate(seeds_folds):
        fa = []
        for te in folds:
            yy = y[te]
            if yy.min() != yy.max():
                fa.append(roc_auc_score(yy, oof_arm[r, te]))
        if fa:
            vals.append(np.mean(fa))
    return float(np.mean(vals))


def fold_index(y):
    out = []
    for r in range(N_REPEATS):
        cv = StratifiedKFold(N_OUTER, shuffle=True, random_state=BASE_SEED + r)
        out.append([te for _, te in cv.split(np.zeros((len(y), 1)), y)])
    return out


def boot_drop(oof_a, oof_b, y, seeds_folds, rng, n_boot):
    """Paired scenario-bootstrap of (AUC_a - AUC_b), resampling scenarios once
    per draw and recomputing the repeat-averaged per-fold statistic for both."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    fold_of = np.zeros((len(seeds_folds), len(y)), dtype=int)
    for r, folds in enumerate(seeds_folds):
        for k, te in enumerate(folds):
            fold_of[r, te] = k
    out = []
    for _ in range(n_boot):
        draw = np.concatenate([rng.choice(pos, len(pos), replace=True),
                               rng.choice(neg, len(neg), replace=True)])
        stats = []
        for src in (oof_a, oof_b):
            reps = []
            for r in range(len(seeds_folds)):
                fa = []
                for k in range(N_OUTER):
                    sel = draw[fold_of[r, draw] == k]
                    yy = y[sel]
                    if len(yy) and yy.min() != yy.max():
                        fa.append(roc_auc_score(yy, src[r, sel]))
                if fa:
                    reps.append(np.mean(fa))
            stats.append(np.mean(reps) if reps else np.nan)
        out.append(stats[0] - stats[1])
    return np.array(out)


def main():
    npz = np.load(NPZ, allow_pickle=True)
    canon = pd.read_csv(CANONICAL)
    ids = np.asarray(npz["scenario_ids"], dtype=int)
    idx = {s: i for i, s in enumerate(ids)}
    acts = np.asarray(npz["activations"], dtype=np.float64)

    ez = np.load(EVAL_NPZ, allow_pickle=True)
    ekeys = [str(k) for k in ez["keys"]]
    eacts = np.asarray(ez["acts"], dtype=np.float64)
    escen = np.array([int(k.split("|")[0]) for k in ekeys])
    esides = np.array([1 if k.split("|")[1].startswith("test") else 0 for k in ekeys])

    # ── assertions: exact coverage and balanced frames ───────────────────────
    canon_ids = set(canon.scenario_id.astype(int).tolist())
    analysis_ids = set(canon.loc[canon.population == "analysis", "scenario_id"]
                       .astype(int).tolist())
    counts = pd.Series(escen).value_counts()
    assert (counts == FRAMES_PER_SCENARIO).all(), \
        f"unbalanced frame counts: {counts[counts != FRAMES_PER_SCENARIO].to_dict()}"
    for s in np.unique(escen):
        m = escen == s
        assert esides[m].sum() == 3 and (~esides[m].astype(bool)).sum() == 3, \
            f"scenario {s} is not 3 test + 3 deploy"
    missing = analysis_ids - set(escen.tolist())
    assert not missing, f"analysis scenarios absent from eval acts: {sorted(missing)[:10]}"
    print(f"eval acts: {len(eacts)} rows, {len(np.unique(escen))} scenarios, "
          f"3+3 frames each; covers all {len(analysis_ids)} analysis scenarios "
          f"({len(canon_ids & set(escen.tolist()))}/{len(canon_ids)} canonical)",
          flush=True)

    rng = np.random.default_rng(BASE_SEED)
    side_variants = []
    for _ in range(N_NULL):
        perm = esides.copy()
        for s in np.unique(escen):
            m = np.flatnonzero(escen == s)
            perm[m] = rng.permutation(np.array([1, 1, 1, 0, 0, 0]))
        side_variants.append(perm)
    iso_dirs = [unit(rng.standard_normal(acts.shape[1])) for _ in range(N_ISO)]

    sub = canon[canon.population == "analysis"].reset_index(drop=True)
    rows = np.array([idx[s] for s in sub.scenario_id.astype(int)])
    scen_ids_all = sub.scenario_id.astype(int).to_numpy()
    Xall = acts[rows]
    lim = sub.response_strategy.isin(LIMITING).to_numpy()
    disc = sub.broad_breach.to_numpy().astype(bool)

    # geometry diagnostic (global directions, descriptive only)
    v_eval_global = paired_direction(eacts, esides, escen)
    v_lim = unit(Xall[disc & lim].mean(0) - Xall[disc & ~lim].mean(0))
    cos_eval_lim = float(v_eval_global @ v_lim)
    null_cos = [float(paired_direction(eacts, sv, escen) @ v_lim)
                for sv in side_variants[:min(30, N_NULL)]]
    print(f"cos(v_eval, v_limiting) = {cos_eval_lim:+.3f}  "
          f"(null |cos| 95th pct {np.percentile(np.abs(null_cos), 95):.3f})", flush=True)

    contrasts = {
        "limiting_among_disclosers": (disc, lim),
        "limiting_vs_direct": (np.ones(len(sub), bool), lim),
        "leak_vs_appropriate": (sub.label_substantive.isin(["leaked", "appropriate"]).to_numpy(),
                                sub.label_substantive.eq("leaked").to_numpy()),
        "substantive_leak": (np.ones(len(sub), bool), sub.substantive_leak.to_numpy()),
    }

    results = []
    for name, (mask, pos) in contrasts.items():
        y = pos[mask].astype(int)
        X = Xall[mask]
        scen_ids = scen_ids_all[mask]
        folds = fold_index(y)
        oof, arms = cv_scores(X, y, scen_ids, eacts, esides, escen,
                              side_variants, iso_dirs)
        auc = {a: perfold_from_oof(oof[a], y, folds) for a in arms}
        auc0 = auc["none"]
        drop_ev = auc0 - auc["v_eval"]
        null_drops = np.array([auc0 - auc[f"null_{k}"] for k in range(N_NULL)])
        iso_drops = np.array([auc0 - auc[f"iso_{k}"] for k in range(N_ISO)])
        p_null = float((1 + (null_drops >= drop_ev).sum()) / (N_NULL + 1))

        bs = boot_drop(oof["none"], oof["v_eval"], y, folds,
                       np.random.default_rng(BASE_SEED), N_BOOT)
        row = {"contrast": name, "primary": name == PRIMARY,
               "n": int(len(y)), "n_pos": int(y.sum()),
               "auc_original": auc0, "auc_after_v_eval": auc["v_eval"],
               "drop_v_eval": drop_ev,
               "drop_boot_lo": float(np.nanpercentile(bs, 2.5)),
               "drop_boot_hi": float(np.nanpercentile(bs, 97.5)),
               "null_drop_mean": float(null_drops.mean()),
               "null_drop_p95": float(np.percentile(null_drops, 95)),
               "iso_drop_mean": float(iso_drops.mean()),
               "p_vs_null": p_null}
        print(f"== {name:28s} auc {auc0:.3f} -> {auc['v_eval']:.3f}  "
              f"drop {drop_ev:+.4f} [{row['drop_boot_lo']:+.4f},{row['drop_boot_hi']:+.4f}] | "
              f"null mean {null_drops.mean():+.4f} p95 {np.percentile(null_drops,95):+.4f} | "
              f"iso mean {iso_drops.mean():+.4f} | p_vs_null={p_null:.3f}", flush=True)
        results.append(row)

    # ── manipulation check: cross-fitted, tuned decoder ──────────────────────
    gkf = GroupKFold(N_OUTER)
    before, after = [], []
    for tr, te in gkf.split(eacts, esides, groups=escen):
        g = GridSearchCV(pipe(), {"clf__C": C_GRID}, scoring="roc_auc",
                         cv=GroupKFold(3).split(eacts[tr], esides[tr], escen[tr]))
        g.fit(eacts[tr], esides[tr])
        before.append(roc_auc_score(esides[te],
                                    g.best_estimator_.predict_proba(eacts[te])[:, 1]))
        # direction from TRAINING scenarios only, then erase both sides
        u = paired_direction(eacts[tr], esides[tr], escen[tr])
        Etr, Ete = erase(eacts[tr], u), erase(eacts[te], u)
        g2 = GridSearchCV(pipe(), {"clf__C": C_GRID}, scoring="roc_auc",
                          cv=GroupKFold(3).split(Etr, esides[tr], escen[tr]))
        g2.fit(Etr, esides[tr])
        after.append(roc_auc_score(esides[te],
                                   g2.best_estimator_.predict_proba(Ete)[:, 1]))
    ev_before, ev_after = float(np.mean(before)), float(np.mean(after))
    erased_ok = ev_after < 0.60
    print(f"\nmanipulation check (cross-fitted, tuned): test-vs-deploy "
          f"{ev_before:.3f} -> {ev_after:.3f} after fold-wise rank-1 erasure "
          f"[{'eval-awareness substantially removed' if erased_ok else 'STILL DECODABLE — a null result here is INCONCLUSIVE'}]",
          flush=True)

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps({
        "purpose": "eval-awareness erasure control for the canonical limiting result",
        "protocol": {
            "fixed_C": FIXED_C, "n_repeats": N_REPEATS, "n_null": N_NULL,
            "n_iso": N_ISO, "n_boot": N_BOOT, "base_seed": BASE_SEED,
            "statistic": "per-fold mean ROC, identical across arms",
            "cross_fitting": "v_eval and every null direction are re-derived inside "
                             "each training fold from framed activations of TRAINING "
                             "scenarios only, then applied to the held-out fold",
            "null": "side-permuted paired directions from the same estimation "
                    "pipeline (primary); isotropic random unit directions "
                    "(secondary, deliberately weak)",
            "drop_ci": "paired scenario bootstrap of the repeat-averaged statistic"},
        "v_eval_source": str(EVAL_NPZ),
        "frames_per_scenario": FRAMES_PER_SCENARIO,
        "cos_v_eval_v_limiting": cos_eval_lim,
        "null_abs_cos_p95": float(np.percentile(np.abs(null_cos), 95)),
        "manipulation_check": {
            "eval_auc_before": ev_before, "eval_auc_after": ev_after,
            "cross_fitted": True, "decoder_C_tuned": True,
            "erasure_effective": bool(erased_ok),
            "note": "Rank-1 erasure of the fold-wise mean-difference direction. "
                    "Residual decodability means eval-awareness is distributed "
                    "beyond one direction, so a surviving limiting probe bounds "
                    "rather than refutes the caution hypothesis."},
        "interpretation_guide": {
            "limiting_survives": "limiting AUC ~unchanged and drop within the null "
                                 "band -> not explained by eval-awareness",
            "limiting_collapses": "drop exceeds the null band -> the limiting signal "
                                  "may be substantially generic caution",
            "manipulation_failed": "eval-awareness still decodable after erasure -> "
                                   "inconclusive, not a negative result"},
        "quick_mode": QUICK,
        "results": df.to_dict(orient="records")}, indent=2) + "\n")
    print(f"\nwrote {OUT_CSV} and {OUT_JSON}")


if __name__ == "__main__":
    main()
