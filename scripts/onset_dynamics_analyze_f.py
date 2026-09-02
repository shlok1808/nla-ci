#!/usr/bin/env python3
"""Local Step 3 analysis for an onset-aligned activation bundle.

Registered primary (frozen in docs/STEP3_ONSET_DYNAMICS_CANDIDATE.md): layer 20,
mean over the eight offsets [-8,-1], `limiting_among_disclosers`, 216-case
analysis population restricted to scenarios whose response covers the whole
window (complete case, fixed across offsets). Activation, TF-IDF, frozen
embedding, position-only and (optional) LLM-reader channels use byte-identical
outer folds. A scenario is resampled or channel-swapped once and carried across
every cell, repeat and channel.

Registered secondaries: the trajectory contrasts (offset_-1 minus
prompt_final, for activations and for the privileged Δ), Δ against the
position-only baseline, and Δ against the LLM reader when reader scores exist.
Everything else (`--grid`, per-cell tables, logit channels) is descriptive.

Usage:
    python scripts/onset_dynamics_analyze_f.py [--quick] [--reader-scores CSV]
    python scripts/onset_dynamics_analyze_f.py --grid      # descriptive heatmap
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import model_registry_f as _registry
from onset_dynamics_common_f import (
    SPEC,
    BLOCK_INDICES,
    HAS_STEP2_CROSSCHECK,
    MODEL_ID,
    POSITION_NAMES,
    PRIMARY_BLOCK,
    PRIMARY_CELLS,
    PRIMARY_REPORTED_LAYER,
    REPORTED_LAYERS,
    STEP2_ACTS,
    build_tokenized_example,
    limiting_labels,
    load_step3_rows,
    primary_mask,
    sha256_file,
    visible_prefix,
)

_P = _registry.paths(SPEC)
from onset_dynamics_stats_f import (
    BASE_SEED,
    MIN_CLASS,
    N_INNER,
    N_OUTER,
    NUMERIC_COLUMNS,
    PairedFoldAUC,
    dense_pipe,
    fit_oof,
    holm,
    outer_splits,
    paired_inference,
    stratified_draws,
    tfidf_pipe,
)

DEFAULT_BUNDLE = _P["acts"]
DEFAULT_MANIFEST = _P["manifest"]
OUT_DIR = _P["out_dir"]
EMBED_MODEL = "all-MiniLM-L6-v2"
TEXT_FAMILY = ("tfidf", "embed")          # registered baseline family (as in Step 2)
CROSSCHECK_MIN_COS = 0.98
GRID_REPEATS = 5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--reader-scores", type=Path, default=Path("results/onset_reader_scores_f.csv"),
                   help="optional LLM-reader baseline scores (scenario_id, position, p_limiting)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--grid", action="store_true", help="descriptive layer x position heatmap instead of the primary")
    p.add_argument("--n-jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    p.add_argument("--allow-source-drift", action="store_true")
    p.add_argument("--allow-crosscheck-fail", action="store_true", help="development only")
    return p.parse_args()


# ── validation ──────────────────────────────────────────────────────────────

def validate_bundle(bundle, manifest, bundle_path: Path, allow_source_drift: bool):
    required = {
        "scenario_ids", "activations", "absolute_indices", "response_indices", "valid",
        "prompt_lengths", "response_lengths", "block_indices", "reported_layers", "position_names",
        "next_token_entropy", "next_token_top1", "next_token_top5_mass",
    }
    missing = required - set(bundle.files)
    if missing:
        raise ValueError(f"bundle missing arrays: {sorted(missing)}")
    if list(bundle["block_indices"]) != list(BLOCK_INDICES):
        raise ValueError("bundle block indices do not match the registered grid")
    if list(bundle["reported_layers"]) != list(REPORTED_LAYERS):
        raise ValueError("bundle reported layers do not match the registered grid")
    if list(bundle["position_names"].astype(str)) != list(POSITION_NAMES):
        raise ValueError("bundle position names do not match the registered grid")
    if manifest.get("output_sha256") and manifest["output_sha256"] != sha256_file(bundle_path):
        raise ValueError("bundle hash does not match its manifest")
    if not allow_source_drift:
        for name, expected in manifest.get("source_hashes", {}).items():
            path = Path(name)
            if not path.exists() or sha256_file(path) != expected:
                raise ValueError(f"source drift since extraction: {name}")


def crosscheck_step2(bundle, allow_fail: bool) -> dict:
    """prompt_final at the primary layer must reproduce the Step 1/2 store.

    Only the registered Qwen run has such a store; for any other subject model
    there is nothing to compare against and the check is skipped explicitly
    rather than silently passing."""
    if not HAS_STEP2_CROSSCHECK:
        return {"status": "not_applicable", "reason": "no Step 1/2 store for this model"}
    if not STEP2_ACTS.exists():
        return {"status": "skipped", "reason": "store not found"}
    z = np.load(STEP2_ACTS, allow_pickle=True)
    old = {int(s): z["activations"][i] for i, s in enumerate(z["scenario_ids"])}
    layer = list(bundle["reported_layers"]).index(PRIMARY_REPORTED_LAYER)
    pf = list(bundle["position_names"].astype(str)).index("prompt_final")
    cos = []
    for i, sid in enumerate(bundle["scenario_ids"].astype(int)):
        if sid in old:
            a = bundle["activations"][i, layer, pf].astype(np.float64)
            b = old[sid].astype(np.float64)
            cos.append(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    cos = np.asarray(cos)
    out = {"n_compared": int(cos.size), "cos_min": float(cos.min()), "cos_median": float(np.median(cos))}
    if cos.min() < CROSSCHECK_MIN_COS:
        msg = f"Step 2 cross-check failed: min cosine {cos.min():.4f} < {CROSSCHECK_MIN_COS}"
        if not allow_fail:
            raise ValueError(msg)
        out["status"] = "FAILED_OVERRIDDEN"
        print("WARNING:", msg)
    else:
        out["status"] = "pass"
    return out


# ── features ────────────────────────────────────────────────────────────────

class Features:
    """Everything a text/position/logit channel may see at a cell. Response
    text is only ever used as the visible prefix ending at that cell."""

    def __init__(self, rows, tokenizer, encoder, bundle, source_rows):
        self.rows = rows
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.bundle = bundle
        self.source_rows = source_rows
        self.response_ids = {
            int(r.scenario_id): build_tokenized_example(r, tokenizer)[3]
            for r in rows.itertuples(index=False)
        }
        self.scenario_emb = np.asarray(
            encoder.encode(rows.scenario.tolist(), batch_size=32, show_progress_bar=False), dtype=np.float64
        )
        self._prefix_emb_cache: dict[str, np.ndarray] = {}

    def at(self, position_index: int):
        b, src = self.bundle, self.source_rows
        r_index = b["response_indices"][src, position_index]
        prefixes = [
            visible_prefix(self.tokenizer, self.response_ids[int(sid)], int(ri))
            for sid, ri in zip(self.rows.scenario_id, r_index)
        ]
        numeric = pd.DataFrame(
            {
                "prefix_tokens": np.maximum(r_index + 1, 0).astype(float),
                "cutoff_tokens": self.rows.cutoff_tok.astype(float).to_numpy(),
                "response_tokens": b["response_lengths"][src].astype(float),
            }
        )
        text_frame = numeric.copy()
        text_frame.insert(0, "text", [f"{s}\n\nVISIBLE RESPONSE PREFIX:\n{p}" for s, p in zip(self.rows.scenario, prefixes)])
        todo = [p for p in set(prefixes) if p not in self._prefix_emb_cache]
        if todo:
            emb = self.encoder.encode(todo, batch_size=64, show_progress_bar=False)
            self._prefix_emb_cache.update({p: np.asarray(e, dtype=np.float64) for p, e in zip(todo, emb)})
        prefix_emb = np.stack([self._prefix_emb_cache[p] for p in prefixes])
        # scenario and prefix are embedded SEPARATELY: MiniLM truncates at 256
        # word pieces and scenarios alone run to 252, so a concatenated string
        # would silently drop the prefix for the longest scenarios.
        embed_X = np.column_stack([self.scenario_emb, prefix_emb, numeric.to_numpy(dtype=np.float64)])
        logits_X = np.column_stack(
            [b[k][src, position_index] for k in ("next_token_entropy", "next_token_top1", "next_token_top5_mass")]
        ).astype(np.float64)
        return {
            "tfidf": text_frame,
            "embed": embed_X,
            "position": numeric.to_numpy(dtype=np.float64),
            "logits": logits_X,
            "prefixes": prefixes,
        }


CHANNEL_FACTORIES = {
    "acts": dense_pipe, "tfidf": tfidf_pipe, "embed": dense_pipe, "position": dense_pipe, "logits": dense_pipe,
}


def fit_cell(X_by_channel, y, n_repeats, fold_ids, n_jobs, selected_C):
    """OOF scores for every channel at one cell; fills fold_ids on first use."""
    out = {ch: np.full((n_repeats, len(y)), np.nan) for ch in X_by_channel}
    for r in range(n_repeats):
        seed = BASE_SEED + r
        splits = outer_splits(y, seed)
        for k, (_, te) in enumerate(splits):
            if (fold_ids[r, te] >= 0).any() and not (fold_ids[r, te] == k).all():
                raise RuntimeError("fold assignment drifted between cells")
            fold_ids[r, te] = k
        for ch, X in X_by_channel.items():
            scores, cs = fit_oof(X, y, splits, seed, CHANNEL_FACTORIES[ch], n_jobs=n_jobs)
            out[ch][r] = scores
            selected_C.setdefault(ch, []).extend(cs)
    return out


def load_reader_scores(path: Path, scenario_ids, cells) -> np.ndarray | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    need = {"scenario_id", "position", "p_limiting"}
    if need - set(df.columns):
        raise ValueError(f"reader scores must have columns {sorted(need)}")
    table = df.set_index(["scenario_id", "position"]).p_limiting
    out = np.full((len(cells), len(scenario_ids)), np.nan)
    for ci, cell in enumerate(cells):
        for si, sid in enumerate(scenario_ids):
            key = (int(sid), cell)
            if key in table.index:
                out[ci, si] = float(table.loc[key])
    if np.isnan(out).any():
        raise ValueError("reader scores do not cover every (scenario, cell) of the primary population")
    return out


def cell_stats(scores, fold_ids, y, n_boot, seed=BASE_SEED):
    """Per-cell repeat-averaged ROC with a scenario bootstrap interval."""
    eng = PairedFoldAUC(scores, fold_ids, y)
    obs = eng.weighted(np.ones((1, len(y))))[0]
    W = stratified_draws(y, n_boot, np.random.default_rng(seed))
    boots = eng.weighted(W)
    return obs, np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)


def linear_contrast(terms, fold_ids, y, n_boot, seed=BASE_SEED):
    """Σ coef · mean_fold_auc(scores) with a shared-draw stratified bootstrap.

    terms: list of (scores (n_cells, n_repeats, n), coef). Used for the
    trajectory contrasts. p_boot_one_sided = P(stat* <= 0); two-sided is
    2·min(P(<=0), P(>=0)).
    """
    engines = [(PairedFoldAUC(s, fold_ids, y), c) for s, c in terms]
    ones = np.ones((1, len(y)))
    observed = float(sum(c * e.weighted(ones).mean() for e, c in engines))
    W = stratified_draws(y, n_boot, np.random.default_rng(seed))
    boots = sum(c * e.weighted(W).mean(axis=1) for e, c in engines)
    p_le = float((1 + np.sum(boots <= 0)) / (n_boot + 1))
    p_ge = float((1 + np.sum(boots >= 0)) / (n_boot + 1))
    return {
        "estimate": observed,
        "boot_lo": float(np.nanpercentile(boots, 2.5)),
        "boot_hi": float(np.nanpercentile(boots, 97.5)),
        "p_boot_one_sided": p_le,
        "p_boot_two_sided": float(min(1.0, 2 * min(p_le, p_ge))),
        "n_boot": int(n_boot),
    }


def _strip(d):
    return {k: v for k, v in d.items() if k != "boot_draws"}


# ── primary ─────────────────────────────────────────────────────────────────

def run_primary(args, bundle, manifest, rows, n_repeats, n_boot, n_perm):
    names = list(bundle["position_names"].astype(str))
    cells = list(PRIMARY_CELLS)
    cell_idx = [names.index(c) for c in cells]
    offset_cells = np.arange(1, len(cells))          # everything but prompt_final
    layer_idx = list(bundle["reported_layers"]).index(PRIMARY_REPORTED_LAYER)
    bidx = {int(s): i for i, s in enumerate(bundle["scenario_ids"].astype(int))}

    mask = primary_mask(rows)
    prim = rows.loc[mask].reset_index(drop=True)
    y_all = limiting_labels(prim)
    src = np.asarray([bidx[int(s)] for s in prim.scenario_id])
    complete = bundle["valid"][src][:, cell_idx].all(axis=1)
    dropped = prim.loc[~complete, ["scenario_id", "response_strategy", "cutoff_tok"]]
    prim = prim.loc[complete].reset_index(drop=True)
    y = y_all[complete]
    src = src[complete]
    if min(np.bincount(y)) < MIN_CLASS:
        raise ValueError("primary complete-case population is underpowered")
    print(f"primary fixed population: n={len(y)}, limiting={int(y.sum())}, direct={int((1-y).sum())}; "
          f"dropped (window not covered): {dropped.scenario_id.tolist()}", flush=True)

    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(manifest.get("model_id", MODEL_ID),
                                              revision=manifest.get("revision_requested", "main"))
    feats = Features(prim, tokenizer, SentenceTransformer(EMBED_MODEL), bundle, src)
    reader = load_reader_scores(args.reader_scores, prim.scenario_id.to_numpy(), cells)

    channels = ["acts", "tfidf", "embed", "position", "logits"]
    S = {ch: np.full((len(cells), n_repeats, len(y)), np.nan) for ch in channels}
    fold_ids = np.full((n_repeats, len(y)), -1, dtype=np.int8)
    selected_C: dict[str, list] = {}
    prefix_char_len = np.zeros((len(cells), len(y)))
    for ci, (cell, pi) in enumerate(zip(cells, cell_idx)):
        F = feats.at(pi)
        prefix_char_len[ci] = [len(p) for p in F["prefixes"]]
        X = {
            "acts": bundle["activations"][src, layer_idx, pi].astype(np.float64),
            "tfidf": F["tfidf"], "embed": F["embed"], "position": F["position"], "logits": F["logits"],
        }
        if not np.isfinite(X["acts"]).all() or not np.isfinite(X["logits"]).all():
            raise ValueError(f"non-finite features at {cell}")
        out = fit_cell(X, y, n_repeats, fold_ids, args.n_jobs, selected_C)
        for ch in channels:
            S[ch][ci] = out[ch]
        print(f"fit {cell}", flush=True)
    if (fold_ids < 0).any() or any(np.isnan(v).any() for v in S.values()):
        raise RuntimeError("incomplete OOF state")
    if reader is not None:
        S["reader"] = np.repeat(reader[:, None, :], n_repeats, axis=1)

    # observed channel ROC over the eight offset cells and at prompt_final
    def mean_roc(ch, sel):
        return float(PairedFoldAUC(S[ch][sel], fold_ids, y).weighted(np.ones((1, len(y)))).mean())

    window_roc = {ch: mean_roc(ch, offset_cells) for ch in S}
    pf_roc = {ch: mean_roc(ch, [0]) for ch in S}
    stronger = max(TEXT_FAMILY, key=lambda ch: window_roc[ch])

    primary = paired_inference(S["acts"], S[stronger], fold_ids, y, n_boot, n_perm, cells=offset_cells)
    vs = {ch: _strip(paired_inference(S["acts"], S[ch], fold_ids, y, n_boot, n_perm, cells=offset_cells))
          for ch in S if ch != "acts"}

    last = [len(cells) - 1]                                    # offset_-1
    secondaries = {
        "S1_trajectory_acts": linear_contrast([(S["acts"][last], 1.0), (S["acts"][[0]], -1.0)], fold_ids, y, n_boot),
        "S2_trajectory_delta": linear_contrast(
            [(S["acts"][last], 1.0), (S[stronger][last], -1.0), (S["acts"][[0]], -1.0), (S[stronger][[0]], 1.0)],
            fold_ids, y, n_boot),
        "S3_delta_vs_position_only": vs["position"],
    }
    if reader is not None:
        secondaries["S4_delta_vs_reader"] = vs["reader"]
    pvals = [v["p_boot_one_sided"] for v in secondaries.values()]
    for key, adj in zip(secondaries, holm(pvals)):
        secondaries[key]["p_holm_one_sided"] = float(adj)

    # per-cell descriptive table with per-cell bootstrap intervals
    per_cell = []
    obs_by_ch, lo_by_ch, hi_by_ch = {}, {}, {}
    for ch in S:
        obs_by_ch[ch], lo_by_ch[ch], hi_by_ch[ch] = cell_stats(S[ch], fold_ids, y, n_boot)
    eng_delta = PairedFoldAUC(S["acts"], fold_ids, y, S[stronger])
    W = stratified_draws(y, n_boot, np.random.default_rng(BASE_SEED))
    delta_boots = eng_delta.weighted(W, "a") - eng_delta.weighted(W, "b")
    for ci, cell in enumerate(cells):
        row = {"position": cell, "prefix_chars_mean": float(prefix_char_len[ci].mean())}
        for ch in S:
            row[f"roc_{ch}"] = float(obs_by_ch[ch][ci])
            row[f"roc_{ch}_lo"] = float(lo_by_ch[ch][ci])
            row[f"roc_{ch}_hi"] = float(hi_by_ch[ch][ci])
        row["delta_vs_stronger"] = row["roc_acts"] - row[f"roc_{stronger}"]
        row["delta_vs_stronger_lo"] = float(np.nanpercentile(delta_boots[:, ci], 2.5))
        row["delta_vs_stronger_hi"] = float(np.nanpercentile(delta_boots[:, ci], 97.5))
        per_cell.append(row)

    supported = primary["delta_boot_lo"] > 0 and primary["p_boot_one_sided"] < 0.05
    result = {
        "status": "complete",
        "analysis": "Step3 primary onset dynamics",
        "population": "analysis_216_disclosers_complete_case_window",
        "contrast": "limiting_among_disclosers",
        "reported_layer": PRIMARY_REPORTED_LAYER,
        "block_index": PRIMARY_BLOCK,
        "primary_cells": cells[1:],
        "n": int(len(y)), "n_limiting": int(y.sum()), "n_direct": int((1 - y).sum()),
        "dropped_scenarios": dropped.to_dict(orient="records"),
        "registered_text_family": list(TEXT_FAMILY),
        "stronger_text_channel": stronger,
        "window_roc": window_roc,
        "prompt_final_roc": pf_roc,
        "primary": {**_strip(primary), "decision_rule": "paired 95% bootstrap interval on mean Δ over [-8,-1] excludes 0 AND one-sided bootstrap p < .05",
                    "supported": bool(supported)},
        "delta_vs_each_channel": vs,
        "registered_secondaries": secondaries,
        "selected_C": {ch: {str(c): int(v.count(c)) for c in sorted(set(v))} for ch, v in selected_C.items()},
        "n_repeats": n_repeats, "outer_folds": N_OUTER, "inner_folds": N_INNER,
        "embed_model": EMBED_MODEL,
        "reader_scores": str(args.reader_scores) if reader is not None else None,
        "bundle_sha256": sha256_file(args.bundle),
        "step2_crosscheck": manifest.get("_crosscheck"),
        "per_cell": per_cell,
        "claim_guardrail": (
            "Predictive linear decodability beyond matched visible-prefix baselines in the registered "
            "family; relative to that family (a stronger reader can only shrink Δ); not proof of a causal "
            "decision variable; not exclusive privacy specificity; the cutoff is the earliest "
            "strategy-evidence quote, not a limiting-language onset."
        ),
    }
    stem = "quick" if args.quick else "primary"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_cell).to_csv(args.out_dir / f"onset_dynamics_{stem}_per_cell.csv", index=False)
    (args.out_dir / f"onset_dynamics_{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        args.out_dir / f"onset_dynamics_{stem}_oof.npz",
        scenario_ids=prim.scenario_id.to_numpy(dtype=np.int32), y=y.astype(np.int8),
        cells=np.asarray(cells), fold_ids=fold_ids,
        **{f"scores_{ch}": S[ch] for ch in S},
    )
    print(json.dumps({"n": len(y), "stronger": stronger, "window_roc": window_roc,
                      "primary": _strip(primary), "secondaries": secondaries}, indent=2))


# ── descriptive grid ────────────────────────────────────────────────────────

def run_grid(args, bundle, manifest, rows, n_boot):
    names = list(bundle["position_names"].astype(str))
    bidx = {int(s): i for i, s in enumerate(bundle["scenario_ids"].astype(int))}
    mask = primary_mask(rows)
    prim_all = rows.loc[mask].reset_index(drop=True)
    y_all = limiting_labels(prim_all)
    src_all = np.asarray([bidx[int(s)] for s in prim_all.scenario_id])

    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(manifest.get("model_id", MODEL_ID),
                                              revision=manifest.get("revision_requested", "main"))
    encoder = SentenceTransformer(EMBED_MODEL)
    out_rows = []
    for pi, cell in enumerate(names):
        valid = bundle["valid"][src_all, pi]
        prim = prim_all.loc[valid].reset_index(drop=True)
        y, src = y_all[valid], src_all[valid]
        if len(y) == 0 or min(np.bincount(y, minlength=2)) < MIN_CLASS:
            out_rows.append({"position": cell, "status": "underpowered_skipped", "n": int(len(y)),
                             "n_limiting": int(y.sum())})
            continue
        feats = Features(prim, tokenizer, encoder, bundle, src)
        F = feats.at(pi)
        X = {"tfidf": F["tfidf"], "embed": F["embed"], "position": F["position"], "logits": F["logits"]}
        for li, layer in enumerate(REPORTED_LAYERS):
            X[f"acts_L{layer}"] = bundle["activations"][src, li, pi].astype(np.float64)
            CHANNEL_FACTORIES.setdefault(f"acts_L{layer}", dense_pipe)
        fold_ids = np.full((GRID_REPEATS, len(y)), -1, dtype=np.int8)
        S = fit_cell(X, y, GRID_REPEATS, fold_ids, args.n_jobs, {})
        stronger = max(TEXT_FAMILY, key=lambda ch: PairedFoldAUC(S[ch][None], fold_ids, y).weighted(np.ones((1, len(y))))[0, 0])
        row = {"position": cell, "status": "scored", "n": int(len(y)), "n_limiting": int(y.sum()),
               "n_direct": int((1 - y).sum()), "stronger_text_channel": stronger}
        for ch, sc in S.items():
            obs, lo, hi = cell_stats(sc[None], fold_ids, y, n_boot)
            row[f"roc_{ch}"] = float(obs[0]); row[f"roc_{ch}_lo"] = float(lo[0]); row[f"roc_{ch}_hi"] = float(hi[0])
        for layer in REPORTED_LAYERS:
            row[f"delta_L{layer}_vs_stronger"] = row[f"roc_acts_L{layer}"] - row[f"roc_{stronger}"]
        out_rows.append(row)
        print(f"grid {cell}: n={len(y)} " + " ".join(f"L{l}={row[f'roc_acts_L{l}']:.3f}" for l in REPORTED_LAYERS)
              + f" text={row[f'roc_{stronger}']:.3f}", flush=True)
    stem = "quick" if args.quick else "primary"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out_rows)
    df.to_csv(args.out_dir / f"onset_dynamics_{stem}_grid.csv", index=False)
    (args.out_dir / f"onset_dynamics_{stem}_grid.json").write_text(json.dumps({
        "analysis": "Step3 descriptive layer x position grid (exploratory; no maximum-over-grid claims)",
        "repeats": GRID_REPEATS, "n_boot": n_boot, "population_rule": "analysis disclosers valid at the cell (varies by cell)",
        "bundle_sha256": sha256_file(args.bundle), "rows": out_rows,
    }, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    n_repeats, n_boot, n_perm = (3, 200, 200) if args.quick else (20, 1000, 5000)
    bundle = np.load(args.bundle, allow_pickle=False)
    manifest = json.loads(args.manifest.read_text())
    validate_bundle(bundle, manifest, args.bundle, args.allow_source_drift)
    manifest["_crosscheck"] = crosscheck_step2(bundle, args.allow_crosscheck_fail)
    print("step2 cross-check:", manifest["_crosscheck"], flush=True)

    rows = load_step3_rows()
    ids = bundle["scenario_ids"].astype(int)
    if len(ids) != 258 or len(set(ids)) != 258:
        raise ValueError("production analysis requires exactly 258 unique extracted IDs")
    missing = set(rows.scenario_id.astype(int)) - set(ids)
    if missing:
        raise ValueError(f"bundle is missing canonical IDs: {sorted(missing)}")

    if args.grid:
        run_grid(args, bundle, manifest, rows, n_boot=200 if args.quick else 1000)
    else:
        run_primary(args, bundle, manifest, rows, n_repeats, n_boot, n_perm)


if __name__ == "__main__":
    main()
