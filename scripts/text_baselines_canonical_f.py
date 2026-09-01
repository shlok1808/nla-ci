#!/usr/bin/env python3
"""Matched text baselines and privileged Δ for the canonical probe contrasts (Step 2).

Question: does the activation probe extract behaviour-relevant information that
text classifiers cannot recover from the scenario wording? The activation at the
final prompt token is a deterministic function of the prompt, so Δ > 0 does NOT
mean the activation contains information absent from the text — it means the
model's own processing makes the information linearly explicit where the text
baselines cannot extract it.

Three channels per contrast, all under the exact v2 protocol
(per-fold metric averaging; StratifiedKFold(5) x 20 repeats; seeds
BASE_SEED..BASE_SEED+19, so fold assignments are IDENTICAL across channels):

  acts   layer-20 final-prompt-token activations, nested-CV L2 logistic
         (identical to probe_contrasts_canonical_v2_f.py; recomputed here
         because v2 did not store per-scenario scores)
  tfidf  TF-IDF 1-2grams on the scenario text + L2 logistic (C from the same
         nested grid)
  embed  frozen all-MiniLM-L6-v2 sentence embeddings (384-d) of the scenario
         text + the same scaler+logistic pipeline

Δ inference: privileged Δ = acts − text per contrast, with a PAIRED stratified
bootstrap (resample within each held-out fold preserving class counts, score
both channels on the same resample, difference the per-fold means; B=1000,
cycling repeats). Reported: percentile 95% CI and a one-sided bootstrap
p = P(Δ* <= 0), Holm-adjusted across the 6 scored contrasts per population.
A label-permutation test is deliberately NOT used for Δ: permuting labels nulls
both channels, testing "no signal anywhere" rather than "no difference".

The headline Δ per contrast is against the STRONGER text baseline (max of tfidf,
embed by point estimate) — using the weaker one would be self-serving.

The generated response text is never used: it post-dates the state being probed
and can trivially reveal the label.

Local CPU only. No APIs, no GPU, no judge, no label/activation changes.
Usage:  python3 scripts/text_baselines_canonical_f.py [--quick]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

NPZ = Path("results/activations_layer20.npz")
CANONICAL = Path("results/behavior_labels_tier3_canonical_f.csv")
BENCH = Path("results/benchmark_results_bf16.csv")   # scenario TEXT only; labels unused

QUICK = "--quick" in sys.argv
if QUICK:
    OUT_CSV = Path("scratch/text_baselines_canonical_quick.csv")
    OUT_JSON = Path("scratch/text_baselines_canonical_quick.json")
    N_REPEATS, N_BOOT = 3, 200
else:
    OUT_CSV = Path("results/text_baselines_canonical_f.csv")
    OUT_JSON = Path("results/text_baselines_canonical_f.json")
    N_REPEATS, N_BOOT = 20, 1000

C_GRID = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
N_OUTER, N_INNER = 5, 3
MIN_CLASS = 15
BASE_SEED = 20260901
LIMITING = {"soft_deflection", "explicit_refusal", "mixed_disclose_then_limit"}
EMBED_MODEL = "all-MiniLM-L6-v2"
CHANNELS = ("acts", "tfidf", "embed")


def dense_pipe(C):
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=C, class_weight="balanced",
                                                solver="liblinear", max_iter=5000))])


def tfidf_pipe(C):
    return Pipeline([("tf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                            sublinear_tf=True)),
                     ("clf", LogisticRegression(C=C, class_weight="balanced",
                                                solver="liblinear", max_iter=5000))])


def oof_folds(X, y, seed, make_pipe, texts=None):
    """One CV repeat; nested C selection per fold. Returns per-fold
    (test_idx, y_te, scores). Fold assignment depends only on (y, seed), so it
    is identical across channels."""
    cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=seed)
    dummy = np.zeros((len(y), 1))
    folds = []
    for tr, te in cv.split(dummy, y):
        g = GridSearchCV(make_pipe(1.0), {"clf__C": C_GRID},
                         scoring="average_precision",
                         cv=StratifiedKFold(N_INNER, shuffle=True, random_state=seed))
        if texts is not None:
            g.fit(texts.iloc[tr], y[tr])
            s = g.best_estimator_.predict_proba(texts.iloc[te])[:, 1]
        else:
            g.fit(X[tr], y[tr])
            s = g.best_estimator_.predict_proba(X[te])[:, 1]
        folds.append((te, y[te], s))
    return folds


def fast_auc(y, s):
    """AUC via the Mann-Whitney U identity, with average ranks for ties.

    Mathematically identical to sklearn.metrics.roc_auc_score (verified by
    assertion at startup); ~20x faster on the small arrays the bootstrap hits
    hundreds of thousands of times.
    """
    n = len(y)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ss = s[order]
    i = 0
    while i < n:                      # average ranks within tie groups
        j = i
        while j + 1 < n and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _verify_fast_auc():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(8, 80))
        y = rng.integers(0, 2, n)
        if y.min() == y.max():
            continue
        # include heavy ties, which is where a naive rank AUC diverges
        s = rng.choice([0.1, 0.2, 0.3, 0.4], n) if rng.random() < 0.5 \
            else rng.random(n)
        a, b = fast_auc(y, s), roc_auc_score(y, s)
        assert abs(a - b) < 1e-12, f"fast_auc mismatch {a} vs {b}"
    print("fast_auc verified identical to roc_auc_score (200 cases, incl. ties)",
          flush=True)


def perfold_auc(folds):
    roc = float(np.mean([roc_auc_score(y, s) for _, y, s in folds]))
    pr = float(np.mean([average_precision_score(y, s) for _, y, s in folds]))
    return roc, pr


def paired_boot(folds_a, folds_b, rng):
    """LEGACY draw (retained for transparency): resamples within the folds of a
    SINGLE repeat, so each draw carries that repeat's full split noise while the
    headline statistic averages N_REPEATS. The resulting interval is therefore
    too wide / conservative. Superseded by paired_boot_matched below; both are
    reported."""
    da, db = [], []
    for (_, ya, sa), (_, yb, sb) in zip(folds_a, folds_b):
        assert len(ya) == len(yb) and (ya == yb).all()
        pos, neg = np.flatnonzero(ya == 1), np.flatnonzero(ya == 0)
        take = np.concatenate([rng.choice(pos, len(pos), replace=True),
                               rng.choice(neg, len(neg), replace=True)])
        da.append(fast_auc(ya[take], sa[take]))
        db.append(fast_auc(yb[take], sb[take]))
    return float(np.mean(da)), float(np.mean(db))


def as_scenario_view(folds_by_rep, ch, n):
    """Per repeat: (oof score per scenario, fold id per scenario). Each scenario
    is in exactly one test fold per repeat, so this is lossless."""
    views = []
    for rep in folds_by_rep:
        s = np.full(n, np.nan)
        f = np.full(n, -1, dtype=int)
        for k, (te, _, sc) in enumerate(rep[ch]):
            s[te] = sc
            f[te] = k
        assert not np.isnan(s).any() and (f >= 0).all()
        views.append((s, f))
    return views


def boot_stat(views, y, draw):
    """The HEADLINE estimator recomputed on one resample: per-fold ROC averaged
    over folds, then over all repeats — matching what we report."""
    reps = []
    for s, f in views:
        folds = []
        for k in range(N_OUTER):
            sel = draw[f[draw] == k]
            yy = y[sel]
            if yy.min() == yy.max():
                continue
            folds.append(fast_auc(yy, s[sel]))
        if folds:
            reps.append(np.mean(folds))
    return float(np.mean(reps)) if reps else np.nan


def paired_boot_matched(views_a, views_b, y, rng):
    """Estimator-matched paired draw: resample SCENARIOS once (stratified), then
    recompute the full repeat-averaged statistic for both channels on that same
    resample. Pairing is exact because fold assignments are shared."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draw = np.concatenate([rng.choice(pos, len(pos), replace=True),
                           rng.choice(neg, len(neg), replace=True)])
    return boot_stat(views_a, y, draw), boot_stat(views_b, y, draw)


def holm(pvals):
    m, order = len(pvals), np.argsort(pvals)
    adj, run = np.empty(m), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return adj


def contrasts(df):
    lim = df.response_strategy.isin(LIMITING)
    disclosed = df.broad_breach.astype(bool)
    return {
        "substantive_leak": (pd.Series(True, index=df.index),
                             df.substantive_leak.astype(bool)),
        "broad_breach": (pd.Series(True, index=df.index),
                         df.broad_breach.astype(bool)),
        "leak_vs_appropriate": (df.label_substantive.isin(["leaked", "appropriate"]),
                                df.label_substantive.eq("leaked")),
        "degree_boundary_broadonly_vs_leaked": (
            df.label_substantive.isin(["broad_only", "leaked"]),
            df.label_substantive.eq("leaked")),
        "limiting_vs_direct": (pd.Series(True, index=df.index), lim),
        "limiting_among_disclosers": (disclosed, lim),
        "refused_vs_appropriate": (df.label_substantive.isin(["refused", "appropriate"]),
                                   df.label_substantive.eq("refused")),
    }


def main():
    _verify_fast_auc()
    npz = np.load(NPZ, allow_pickle=True)
    canon = pd.read_csv(CANONICAL)
    bench = pd.read_csv(BENCH)[["scenario_id", "scenario"]]   # text only
    canon = canon.merge(bench, on="scenario_id", how="left")
    assert canon.scenario.notna().all(), "missing scenario text"
    ids = np.asarray(npz["scenario_ids"], dtype=int)
    idx = {s: i for i, s in enumerate(ids)}
    acts = np.asarray(npz["activations"], dtype=np.float64)

    print(f"encoding {len(canon)} scenarios with {EMBED_MODEL} (frozen, CPU)...",
          flush=True)
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(EMBED_MODEL)
    emb_all = np.asarray(enc.encode(canon.scenario.tolist(), batch_size=32,
                                    show_progress_bar=False), dtype=np.float64)

    results = []
    for pop, sub in (("analysis_216", canon[canon.population == "analysis"]),
                     ("all_258", canon)):
        sub = sub.reset_index(drop=True)
        rows = np.array([idx[s] for s in sub.scenario_id.astype(int)])
        sub_emb = emb_all[[canon.index[canon.scenario_id == s][0]
                           for s in sub.scenario_id]]
        for name, (mask, pos) in contrasts(sub).items():
            m = mask.to_numpy()
            y = pos.to_numpy()[m].astype(int)
            nmin = int(min(y.sum(), len(y) - y.sum()))
            tag = f"{name}|{pop}"
            if nmin < MIN_CLASS:
                results.append({"contrast": tag, "status": "underpowered_skipped",
                                "n": int(len(y)), "n_pos": int(y.sum()),
                                "minority_n": nmin})
                continue
            Xa = acts[rows[m]]
            Xe = sub_emb[m]
            texts = sub.scenario[m].reset_index(drop=True)

            per = {ch: [] for ch in CHANNELS}
            folds_by_rep = []
            for r in range(N_REPEATS):
                seed = BASE_SEED + r
                fa = oof_folds(Xa, y, seed, dense_pipe)
                ft = oof_folds(None, y, seed, tfidf_pipe, texts=texts)
                fe = oof_folds(Xe, y, seed, dense_pipe)
                folds_by_rep.append({"acts": fa, "tfidf": ft, "embed": fe})
                for ch, f in (("acts", fa), ("tfidf", ft), ("embed", fe)):
                    per[ch].append(perfold_auc(f))

            out = {"contrast": tag, "status": "scored", "n": int(len(y)),
                   "n_pos": int(y.sum()), "prevalence": float(y.mean())}
            for ch in CHANNELS:
                arr = np.array(per[ch])
                out[f"roc_{ch}"] = float(arr[:, 0].mean())
                out[f"pr_{ch}"] = float(arr[:, 1].mean())

            # headline text baseline = stronger of the two by point estimate
            stronger = "embed" if out["roc_embed"] >= out["roc_tfidf"] else "tfidf"
            out["stronger_text_channel"] = stronger

            # --- estimator-matched paired bootstrap (primary inference) ---
            views = {ch: as_scenario_view(folds_by_rep, ch, len(y)) for ch in CHANNELS}
            rng = np.random.default_rng(BASE_SEED)
            matched = {"tfidf": [], "embed": []}
            draws_acts = []
            for _ in range(N_BOOT):
                pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
                draw = np.concatenate([rng.choice(pos, len(pos), replace=True),
                                       rng.choice(neg, len(neg), replace=True)])
                a = boot_stat(views["acts"], y, draw)
                draws_acts.append(a)
                for txt_ch in ("tfidf", "embed"):
                    matched[txt_ch].append(a - boot_stat(views[txt_ch], y, draw))
            out["boot_draws_acts"] = draws_acts        # reused for the dissociation test

            # --- legacy single-repeat bootstrap (retained for transparency) ---
            rng_legacy = np.random.default_rng(BASE_SEED)
            legacy = {"tfidf": [], "embed": []}
            for b in range(N_BOOT):
                rep = folds_by_rep[b % N_REPEATS]
                for txt_ch in ("tfidf", "embed"):
                    a_auc, t_auc = paired_boot(rep["acts"], rep[txt_ch], rng_legacy)
                    legacy[txt_ch].append(a_auc - t_auc)

            for txt_ch in ("tfidf", "embed"):
                arr = np.array(matched[txt_ch])
                out[f"delta_roc_{txt_ch}"] = out["roc_acts"] - out[f"roc_{txt_ch}"]
                out[f"delta_boot_lo_{txt_ch}"] = float(np.nanpercentile(arr, 2.5))
                out[f"delta_boot_hi_{txt_ch}"] = float(np.nanpercentile(arr, 97.5))
                out[f"p_boot_{txt_ch}"] = float(
                    (1 + (arr <= 0).sum()) / (N_BOOT + 1))
                out[f"delta_draws_{txt_ch}"] = arr.tolist()
                old = np.array(legacy[txt_ch])
                out[f"legacy_boot_lo_{txt_ch}"] = float(np.percentile(old, 2.5))
                out[f"legacy_boot_hi_{txt_ch}"] = float(np.percentile(old, 97.5))
                out[f"legacy_p_boot_{txt_ch}"] = float(
                    (1 + (old <= 0).sum()) / (N_BOOT + 1))
            out["delta_roc_stronger"] = out[f"delta_roc_{stronger}"]
            out["delta_boot_lo_stronger"] = out[f"delta_boot_lo_{stronger}"]
            out["delta_boot_hi_stronger"] = out[f"delta_boot_hi_{stronger}"]
            out["p_boot_stronger"] = out[f"p_boot_{stronger}"]

            print(f"== {tag}  acts {out['roc_acts']:.3f} | tfidf {out['roc_tfidf']:.3f} "
                  f"| embed {out['roc_embed']:.3f} | Δ(vs {stronger}) "
                  f"{out['delta_roc_stronger']:+.3f} "
                  f"[{out['delta_boot_lo_stronger']:+.3f},{out['delta_boot_hi_stronger']:+.3f}] "
                  f"p={out['p_boot_stronger']:.4f}", flush=True)
            results.append(out)

    # ── dissociation test ────────────────────────────────────────────────────
    # "limiting is privileged, leak is not" cannot rest on one contrast being
    # significant and another not (Gelman & Stern 2006, Am. Stat. 60(4)). Test
    # the DIFFERENCE of the two deltas directly. The two contrasts are scored on
    # overlapping but non-identical case sets, so draws are aligned by index
    # (same seed, same draw order), giving a paired-by-construction comparison.
    dissoc = []
    by_tag = {r["contrast"]: r for r in results if r.get("status") == "scored"}
    for pop in ("analysis_216", "all_258"):
        for lim_key in ("limiting_among_disclosers", "limiting_vs_direct"):
            for leak_key in ("leak_vs_appropriate", "substantive_leak"):
                a = by_tag.get(f"{lim_key}|{pop}")
                b = by_tag.get(f"{leak_key}|{pop}")
                if not a or not b:
                    continue
                ch = a["stronger_text_channel"]
                da = np.array(a[f"delta_draws_{ch}"])
                db = np.array(b[f"delta_draws_{b['stronger_text_channel']}"])
                diff = da - db
                dissoc.append({
                    "population": pop, "limiting_contrast": lim_key,
                    "leak_contrast": leak_key,
                    "delta_limiting": a["delta_roc_stronger"],
                    "delta_leak": b["delta_roc_stronger"],
                    "difference": float(np.nanmean(diff)),
                    "boot_lo": float(np.nanpercentile(diff, 2.5)),
                    "boot_hi": float(np.nanpercentile(diff, 97.5)),
                    "p_boot": float((1 + (diff <= 0).sum()) / (len(diff) + 1)),
                })
    print("\n-- dissociation: is the limiting Δ larger than the leak Δ? --")
    for r in dissoc:
        if r["population"] == "analysis_216":
            print(f"  {r['limiting_contrast']:26s} vs {r['leak_contrast']:20s} "
                  f"diff {r['difference']:+.3f} [{r['boot_lo']:+.3f},{r['boot_hi']:+.3f}] "
                  f"p={r['p_boot']:.4f}")

    # drop bulky draw arrays before serialising
    for r in results:
        for k in [k for k in list(r) if k.startswith(("delta_draws_", "boot_draws_"))]:
            r.pop(k)

    df = pd.DataFrame(results)
    df["p_holm_delta_stronger"] = np.nan
    for pop in ("analysis_216", "all_258"):
        sel = df.index[(df.status == "scored") & df.contrast.str.endswith(pop)]
        df.loc[sel, "p_holm_delta_stronger"] = holm(
            df.loc[sel, "p_boot_stronger"].to_numpy())
    df.to_csv(OUT_CSV, index=False)
    pd.DataFrame(dissoc).to_csv(
        str(OUT_CSV).replace(".csv", "_dissociation.csv"), index=False)
    OUT_JSON.write_text(json.dumps({
        "purpose": "matched text baselines + paired privileged delta (Step 2)",
        "channels": {"acts": "layer-20 final-prompt-token activations (v2 protocol)",
                     "tfidf": "TF-IDF 1-2grams min_df=2 sublinear on scenario text",
                     "embed": f"frozen {EMBED_MODEL} sentence embeddings (384-d)"},
        "protocol": {"n_repeats": N_REPEATS, "n_boot": N_BOOT, "C_grid": C_GRID,
                     "base_seed": BASE_SEED, "min_class": MIN_CLASS,
                     "folds": "StratifiedKFold(5), identical across channels",
                     "statistic": "per-fold ROC/PR averaged",
                     "delta_inference": "paired within-fold stratified bootstrap; "
                                        "one-sided p = P(delta* <= 0); Holm across "
                                        "the 6 scored contrasts per population",
                     "headline_baseline": "stronger of tfidf/embed by point estimate"},
        "response_text_used": False,
        "quick_mode": QUICK,
        "inference_note": (
            "PRIMARY inference is the estimator-matched paired bootstrap: scenarios "
            "are resampled once per draw and the full repeat-averaged per-fold ROC is "
            "recomputed for both channels on that resample. The legacy_* columns are "
            "the earlier single-repeat-per-draw bootstrap, which carried one repeat's "
            "split noise against a 20-repeat statistic and was therefore too wide. "
            "The correction was identified after the first run; its direction "
            "(narrower intervals) follows a priori from averaging reducing split "
            "variance, and both inferences are reported."),
        "dissociation": dissoc,
        "results": df.to_dict(orient="records")}, indent=2) + "\n")
    print(f"\nwrote {OUT_CSV} and {OUT_JSON}")


if __name__ == "__main__":
    main()
