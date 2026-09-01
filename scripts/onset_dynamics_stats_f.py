#!/usr/bin/env python3
"""Step 3 inference machinery: nested-CV probes and exact, vectorised paired
inference on repeat-averaged per-fold ROC-AUC.

Protocol continuity with Steps 1-2 (`probe_contrasts_canonical_v2_f.py`,
`text_baselines_canonical_f.py`): StratifiedKFold(5) x repeats with seeds
BASE_SEED..BASE_SEED+R-1 (fold assignment depends only on (y, seed), so it is
byte-identical across channels and cells), inner StratifiedKFold(3) grid over
C_GRID scored by average precision, class_weight="balanced", liblinear, the
reported statistic is the per-fold ROC-AUC averaged over folds and repeats
(Forman & Scholz 2010), and the primary interval is a paired stratified
scenario bootstrap that recomputes the full repeat-averaged estimator on every
draw (Step 2 §3.5 correction). The one-sided bootstrap p = P(Δ* <= 0) is the
registered null, exactly as in Step 2.

Everything is exact. The bootstrap and the scenario-level channel-swap
randomisation are evaluated through precomputed pairwise comparison matrices
(the Mann-Whitney identity: AUC = mean over positive/negative pairs of
[s+ > s-] + 0.5[s+ = s-]), so a resample is a weighted bilinear form and
thousands of draws cost seconds instead of millions of `roc_auc_score` calls.
`tests/test_onset_dynamics_f.py` asserts equality with the naive loops.

Channel-swap null: swapping raw `predict_proba` between two differently
calibrated classifiers mixes score scales, so scores are first converted to
average ranks within each (cell, repeat, fold) test set. ROC-AUC is invariant
to that transform, so observed statistics are unchanged; only the null becomes
scale-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
N_OUTER, N_INNER = 5, 3
BASE_SEED = 20260901
MIN_CLASS = 15
NUMERIC_COLUMNS = ["prefix_tokens", "cutoff_tokens", "response_tokens"]


# ── probes ──────────────────────────────────────────────────────────────────

def _logreg(C):
    return LogisticRegression(
        C=C, class_weight="balanced", solver="liblinear", max_iter=5000
    )


def dense_pipe(C=1.0):
    return Pipeline([("scale", StandardScaler()), ("clf", _logreg(C))])


def tfidf_pipe(C=1.0):
    """TF-IDF on `text` plus standardised numeric position features."""
    features = ColumnTransformer(
        [
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "text"),
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
        ]
    )
    return Pipeline([("features", features), ("clf", _logreg(C))])


def outer_splits(y, seed):
    cv = StratifiedKFold(N_OUTER, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros((len(y), 1)), y))


def fit_oof(X, y, splits, seed, make_pipe, n_jobs=1):
    """Out-of-fold scores under nested C selection. Returns (scores, chosen_C).

    `n_jobs` only parallelises the inner grid; results are identical for any
    value (each candidate fit is deterministic given the data and the split).
    """
    scores = np.full(len(y), np.nan)
    chosen = []
    is_frame = isinstance(X, pd.DataFrame)
    for tr, te in splits:
        inner = StratifiedKFold(N_INNER, shuffle=True, random_state=seed)
        grid = GridSearchCV(
            make_pipe(), {"clf__C": C_GRID}, scoring="average_precision",
            cv=inner, n_jobs=n_jobs,
        )
        grid.fit(X.iloc[tr] if is_frame else X[tr], y[tr])
        test = X.iloc[te] if is_frame else X[te]
        scores[te] = grid.best_estimator_.predict_proba(test)[:, 1]
        chosen.append(float(grid.best_params_["clf__C"]))
    if np.isnan(scores).any():
        raise RuntimeError("OOF prediction did not cover every scenario")
    return scores, chosen


# ── naive reference statistic (used by tests and for the observed value) ────

def naive_mean_fold_auc(scores, fold_ids, y, draw=None):
    """Mean over (cell, repeat, fold) of ROC-AUC on one scenario draw.

    scores: (n_cells, n_repeats, n). Folds with one class are skipped, matching
    the vectorised implementation.
    """
    n_cells, n_repeats, n = scores.shape
    draw = np.arange(n) if draw is None else np.asarray(draw, dtype=int)
    values = []
    for c in range(n_cells):
        for r in range(n_repeats):
            for k in range(N_OUTER):
                sel = draw[fold_ids[r, draw] == k]
                yy = y[sel]
                if len(sel) and yy.min() != yy.max():
                    values.append(roc_auc_score(yy, scores[c, r, sel]))
    return float(np.mean(values)) if values else np.nan


# ── exact vectorised machinery ─────────────────────────────────────────────

def _pair_matrix(a, b):
    """[a_i > b_j] + 0.5 [a_i == b_j], a: (cells, m_pos), b: (cells, m_neg)."""
    diff = a[:, :, None] - b[:, None, :]
    return (diff > 0).astype(np.float64) + 0.5 * (diff == 0)


class PairedFoldAUC:
    """Repeat-averaged per-fold ROC-AUC for one or two channels, evaluated on
    arbitrary scenario weightings (bootstrap draws) or channel swaps.

    scores_a / scores_b: (n_cells, n_repeats, n) OOF scores; fold_ids:
    (n_repeats, n); y: (n,) in {0,1}. Every statistic is an exact re-evaluation
    of the reported estimator on the resampled/swapped data.
    """

    def __init__(self, scores_a, fold_ids, y, scores_b=None):
        self.a = np.asarray(scores_a, dtype=np.float64)
        self.b = None if scores_b is None else np.asarray(scores_b, dtype=np.float64)
        self.fold_ids = np.asarray(fold_ids)
        self.y = np.asarray(y).astype(int)
        self.n_cells, self.n_repeats, self.n = self.a.shape
        if self.b is not None and self.b.shape != self.a.shape:
            raise ValueError("channel score arrays must have identical shapes")
        self.folds = []  # (r, k, pos_idx, neg_idx, Haa, Hbb, Hab, Hba)
        for r in range(self.n_repeats):
            for k in range(N_OUTER):
                members = np.flatnonzero(self.fold_ids[r] == k)
                pos = members[self.y[members] == 1]
                neg = members[self.y[members] == 0]
                if len(pos) == 0 or len(neg) == 0:
                    continue
                ra = self._ranked(self.a[:, r, members])
                Haa = _pair_matrix(ra[:, self.y[members] == 1], ra[:, self.y[members] == 0])
                if self.b is None:
                    self.folds.append((r, k, pos, neg, Haa, None, None, None))
                    continue
                rb = self._ranked(self.b[:, r, members])
                pm, nm = self.y[members] == 1, self.y[members] == 0
                self.folds.append((
                    r, k, pos, neg, Haa,
                    _pair_matrix(rb[:, pm], rb[:, nm]),
                    _pair_matrix(ra[:, pm], rb[:, nm]),
                    _pair_matrix(rb[:, pm], ra[:, nm]),
                ))

    @staticmethod
    def _ranked(s):
        """Average ranks within each cell's fold test set (ties preserved)."""
        return rankdata(s, axis=1, method="average")

    def weighted(self, weights, which="a"):
        """Mean over (repeat, fold) of weighted AUC per cell for each weight
        row. weights: (B, n) non-negative scenario multiplicities. Returns
        (B, n_cells). Folds where a class has zero total weight are skipped
        for that draw (nan-mean), matching `naive_mean_fold_auc`.
        """
        W = np.asarray(weights, dtype=np.float64)
        if W.ndim == 1:
            W = W[None, :]
        out = np.zeros((W.shape[0], self.n_cells))
        count = np.zeros(W.shape[0])
        for r, k, pos, neg, Haa, Hbb, _, _ in self.folds:
            H = Haa if which == "a" else Hbb
            wp, wn = W[:, pos], W[:, neg]
            den = wp.sum(1) * wn.sum(1)
            ok = den > 0
            if not ok.any():
                continue
            num = np.einsum("bi,cij,bj->bc", wp[ok], H, wn[ok])
            out[ok] += num / den[ok, None]
            count[ok] += 1
        with np.errstate(invalid="ignore"):
            return out / count[:, None]

    def swapped_delta(self, flips):
        """Channel-swap randomisation: for each row of `flips` (B, n) bool,
        scenario i takes channel b's score in the swapped 'a' channel where
        flips[i] is True (and vice versa). Returns (B, n_cells) with
        mean_fold_auc(a') - mean_fold_auc(b'). Exact on rank-normalised scores.
        """
        if self.b is None:
            raise ValueError("two channels are required for a swap null")
        F = np.asarray(flips, dtype=np.float64)
        if F.ndim == 1:
            F = F[None, :]
        out = np.zeros((F.shape[0], self.n_cells))
        count = 0
        for r, k, pos, neg, Haa, Hbb, Hab, Hba in self.folds:
            fp, fn = F[:, pos], F[:, neg]
            gp, gn = 1.0 - fp, 1.0 - fn
            den = float(len(pos) * len(neg))
            # a' = a where not flipped, b where flipped
            num_a = (
                np.einsum("bi,cij,bj->bc", gp, Haa, gn)
                + np.einsum("bi,cij,bj->bc", gp, Hab, fn)
                + np.einsum("bi,cij,bj->bc", fp, Hba, gn)
                + np.einsum("bi,cij,bj->bc", fp, Hbb, fn)
            )
            num_b = (
                np.einsum("bi,cij,bj->bc", fp, Haa, fn)
                + np.einsum("bi,cij,bj->bc", fp, Hab, gn)
                + np.einsum("bi,cij,bj->bc", gp, Hba, fn)
                + np.einsum("bi,cij,bj->bc", gp, Hbb, gn)
            )
            out += (num_a - num_b) / den
            count += 1
        return out / count


def stratified_draws(y, n_boot, rng):
    """Scenario multiplicity matrix (n_boot, n) for stratified bootstrap draws:
    positives resampled among positives, negatives among negatives."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    W = np.zeros((n_boot, len(y)), dtype=np.float64)
    for b in range(n_boot):
        draw = np.concatenate(
            [rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        )
        W[b] = np.bincount(draw, minlength=len(y))
    return W


def paired_inference(scores_a, scores_b, fold_ids, y, n_boot, n_perm, seed=BASE_SEED,
                     cells=None):
    """Paired comparison of two channels on the mean over `cells` (default all)
    of the repeat-averaged per-fold ROC-AUC.

    Returns the observed AUCs and Δ, the paired stratified bootstrap 95%
    percentile interval and one-sided bootstrap p = P(Δ* <= 0) (registered
    null, as in Step 2), and the scale-free scenario-level channel-swap
    randomisation p (secondary null). Draws/flips are per scenario and shared
    across all cells and repeats.
    """
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    cells = np.arange(scores_a.shape[0]) if cells is None else np.asarray(cells)
    eng = PairedFoldAUC(scores_a[cells], fold_ids, y, scores_b[cells])
    ones = np.ones((1, len(y)))
    obs_a = float(eng.weighted(ones, "a").mean())
    obs_b = float(eng.weighted(ones, "b").mean())
    observed = obs_a - obs_b

    rng = np.random.default_rng(seed)
    W = stratified_draws(y, n_boot, rng)
    boots = (eng.weighted(W, "a") - eng.weighted(W, "b")).mean(axis=1)
    flips = rng.random((n_perm, len(y))) < 0.5
    null = eng.swapped_delta(flips).mean(axis=1)
    return {
        "roc_a": obs_a,
        "roc_b": obs_b,
        "delta_roc": observed,
        "delta_boot_lo": float(np.nanpercentile(boots, 2.5)),
        "delta_boot_hi": float(np.nanpercentile(boots, 97.5)),
        "p_boot_one_sided": float((1 + np.sum(boots <= 0)) / (n_boot + 1)),
        "p_swap_randomization": float((1 + np.sum(null >= observed)) / (n_perm + 1)),
        "n_boot": int(n_boot),
        "n_swap_randomization": int(n_perm),
        "boot_draws": boots,
    }


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    m, order = len(p), np.argsort(p)
    adj, run = np.empty(m), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj
