#!/usr/bin/env python3
"""Analyze the Step 3 forced-prefix control on the local CPU.

The same fixed prefixes are appended to every scenario. If limiting-strategy
decodability rises while the generated words are held constant, that rise
cannot be attributed to class-correlated response wording.

Registered control statistic: mean over the three FIXED prefixes of
ROC(prefix-final) − ROC(prompt_final), paired stratified scenario bootstrap
(two-sided p, since "flat" and "rise" are both pre-specified outcomes). The
`natural` template prefix is a secondary. All draws are shared by scenario
across prefixes and repeats.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from onset_dynamics_common_f import limiting_labels, load_step3_rows, primary_mask, sha256_file
from onset_dynamics_stats_f import (
    BASE_SEED,
    MIN_CLASS,
    N_OUTER,
    PairedFoldAUC,
    dense_pipe,
    fit_oof,
    outer_splits,
    stratified_draws,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, default=Path("results/onset_dynamics_forced_acts_f.npz"))
    p.add_argument("--manifest", type=Path, default=Path("results/onset_dynamics_forced_manifest_f.json"))
    p.add_argument("--out-dir", type=Path, default=Path("results/paper_pipeline/03_onset_dynamics"))
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-jobs", type=int, default=4)
    return p.parse_args()


def contrast(after, before, fold_ids, y, n_boot, seed=BASE_SEED):
    """mean_fold_auc(after) − mean_fold_auc(before), shared stratified draws."""
    ea, eb = PairedFoldAUC(after, fold_ids, y), PairedFoldAUC(before, fold_ids, y)
    ones = np.ones((1, len(y)))
    obs = float(ea.weighted(ones).mean() - eb.weighted(ones).mean())
    W = stratified_draws(y, n_boot, np.random.default_rng(seed))
    boots = ea.weighted(W).mean(axis=1) - eb.weighted(W).mean(axis=1)
    p_le = float((1 + np.sum(boots <= 0)) / (n_boot + 1))
    p_ge = float((1 + np.sum(boots >= 0)) / (n_boot + 1))
    return {
        "roc_before": float(eb.weighted(ones).mean()), "roc_after": float(ea.weighted(ones).mean()),
        "delta": obs, "boot_lo": float(np.nanpercentile(boots, 2.5)), "boot_hi": float(np.nanpercentile(boots, 97.5)),
        "p_boot_two_sided": float(min(1.0, 2 * min(p_le, p_ge))), "p_boot_one_sided_rise": p_le, "n_boot": int(n_boot),
    }


def main():
    args = parse_args()
    repeats, n_boot = (3, 200) if args.quick else (20, 1000)
    z = np.load(args.bundle, allow_pickle=False)
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("output_sha256") != sha256_file(args.bundle):
        raise ValueError("forced-prefix bundle hash does not match manifest")
    required = {"scenario_ids", "activations", "valid", "prefix_names", "prefix_token_lengths", "prefix_is_template"}
    if required - set(z.files):
        raise ValueError(f"forced-prefix bundle missing {sorted(required - set(z.files))}")

    rows = load_step3_rows()
    ids = z["scenario_ids"].astype(int)
    if len(ids) != 258 or len(set(ids)) != 258:
        raise ValueError("production forced-prefix analysis requires 258 unique IDs")
    lookup = {sid: i for i, sid in enumerate(ids)}
    primary = rows.loc[primary_mask(rows)].reset_index(drop=True)
    y = limiting_labels(primary)
    if min(np.bincount(y)) < MIN_CLASS:
        raise ValueError("underpowered")
    source = np.asarray([lookup[int(s)] for s in primary.scenario_id])
    names = z["prefix_names"].astype(str)
    is_template = z["prefix_is_template"].astype(bool)
    lengths = z["prefix_token_lengths"][source]
    for j, name in enumerate(names):
        if not is_template[j] and not np.all(lengths[:, j] == lengths[0, j]):
            raise ValueError(f"fixed prefix {name} token length varies across scenarios")

    fold_ids = np.full((repeats, len(y)), -1, dtype=np.int8)
    splits_by_rep = []
    for r in range(repeats):
        splits = outer_splits(y, BASE_SEED + r)
        for k, (_, te) in enumerate(splits):
            fold_ids[r, te] = k
        splits_by_rep.append(splits)

    trajectory, before, after = [], {}, {}
    for j, name in enumerate(names):
        if is_template[j]:
            # per-scenario length: probe prompt_final and the prefix-final position only
            positions = {"prompt_final": np.zeros(len(y), int), "prefix_final": lengths[:, j].astype(int)}
        else:
            n_pos = int(lengths[0, j]) + 1
            positions = {f"k{k}": np.full(len(y), k) for k in range(n_pos)}
        scores = {}
        for pname, pos in positions.items():
            if not z["valid"][source, j][np.arange(len(y)), pos].all():
                raise ValueError(f"invalid forced-prefix cell for {name}/{pname}")
            X = z["activations"][source, j][np.arange(len(y)), pos].astype(np.float64)
            sc = np.full((repeats, len(y)), np.nan)
            for r in range(repeats):
                sc[r], _ = fit_oof(X, y, splits_by_rep[r], BASE_SEED + r, dense_pipe, n_jobs=args.n_jobs)
            scores[pname] = sc
            obs = PairedFoldAUC(sc[None], fold_ids, y).weighted(np.ones((1, len(y))))[0, 0]
            trajectory.append({"prefix": name, "template": bool(is_template[j]), "position": pname, "roc": float(obs)})
            print(f"{name}/{pname}: roc {obs:.3f}", flush=True)
        keys = list(positions)
        before[name], after[name] = scores[keys[0]], scores[keys[-1]]

    fixed = [n for n, t in zip(names, is_template) if not t]
    registered = contrast(np.stack([after[n] for n in fixed]), np.stack([before[n] for n in fixed]), fold_ids, y, n_boot)
    per_prefix = {n: contrast(after[n][None], before[n][None], fold_ids, y, n_boot) for n in names}

    result = {
        "status": "complete",
        "analysis": "Step3 forced-prefix transcript control",
        "n": int(len(y)), "n_limiting": int(y.sum()), "n_direct": int((1 - y).sum()),
        "fixed_prefixes": fixed, "template_prefixes": [n for n in names if n not in fixed],
        "registered_fixed_prefix_contrast": registered,
        "per_prefix_contrast": per_prefix,
        "trajectory": trajectory,
        "n_repeats": repeats, "outer_folds": N_OUTER,
        "bundle_sha256": sha256_file(args.bundle),
        "interpretation_guardrail": (
            "A rise under fixed words weakens transcript-wording explanations but remains subject to "
            "distribution shift from forced prefixes; a flat curve means generated wording explains any "
            "natural-response rise."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = "quick" if args.quick else "primary"
    pd.DataFrame(trajectory).to_csv(args.out_dir / f"forced_prefix_{stem}_trajectory.csv", index=False)
    (args.out_dir / f"forced_prefix_{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"registered": registered, "per_prefix": per_prefix}, indent=2))


if __name__ == "__main__":
    main()
