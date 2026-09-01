"""paper_step01_common_f.py — shared loaders, verdict rule, and artifact registry
for the Step-1 (statistics protocol) paper artifact package.

Everything that both the builder and the human-readable docs need lives here, so
sidecars, ARTIFACT_INDEX.md and run_metadata.json are all GENERATED from one
registry rather than hand-written (6 artifacts x ~20 fields would drift by the
second edit).

Read-only with respect to results/: this module loads
results/probe_contrasts_canonical_v2_f.{csv,json} (authoritative),
results/probe_contrasts_canonical_f.csv (v1, provenance) and
results/behavior_labels_tier3_canonical_f.csv (population composition).
It never writes outside results/paper_pipeline/01_statistics_protocol/.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO = Path(__file__).resolve().parent.parent
V2_CSV = REPO / "results/probe_contrasts_canonical_v2_f.csv"
V2_JSON = REPO / "results/probe_contrasts_canonical_v2_f.json"
V1_CSV = REPO / "results/probe_contrasts_canonical_f.csv"
V1_JSON = REPO / "results/probe_contrasts_canonical_f.json"
LABELS = REPO / "results/behavior_labels_tier3_canonical_f.csv"
MANIFEST = REPO / "results/behavior_labels_tier3_canonical_manifest_f.json"
V2_SCRIPT = REPO / "scripts/probe_contrasts_canonical_v2_f.py"
V1_SCRIPT = REPO / "scripts/probe_contrasts_canonical_f.py"
AUDIT_DOC = REPO / "docs/FULL_AUDIT_AND_NEXT_STEPS_f.md"

STEP_DIR = REPO / "results/paper_pipeline/01_statistics_protocol"
TABLES = STEP_DIR / "tables"
FIGURES = STEP_DIR / "figures"

# Hashes of the authoritative inputs at the time this package was designed.
# A mismatch means an "authoritative" file changed under us -> refuse to build.
EXPECTED_SHA256 = {
    "results/probe_contrasts_canonical_v2_f.csv":
        "0888918e971a70e5b1ed7d0854f7feee5089bef20a0ec0d95669a2fcd96f0367",
    "results/probe_contrasts_canonical_v2_f.json":
        "194e1f0e4b0478daa9c9d8c9cf0c6c46ce240baa506ad6a64eb4c9fdf1fc27b5",
    "results/probe_contrasts_canonical_f.csv":
        "e91bdf0c49b487caa69e51457f6599ed78bbed962580af342f67d8e44c56a328",
    "results/probe_contrasts_canonical_f.json":
        "aa3de486c64b09c7a037bb8ecc07b9aede97946dbca5fc9ea3010d0b07803c9b",
    "results/behavior_labels_tier3_canonical_f.csv":
        "40c5fb2b10aa49d535b6593ba9e0bcbe6ad631dfe21c8237845924cba3705979",
}

# ── Protocol constants (mirrored from the v2 run, asserted against its JSON) ───

N_PERM = 500
N_BOOT = 1000
N_REPEATS = 20
BASE_SEED = 20260901
N_CONTRASTS_ADJUSTED = 6           # scored contrasts per population
P_RAW_FLOOR = 1.0 / (N_PERM + 1)   # 0.001996007984031936
P_HOLM_FLOOR = N_CONTRASTS_ADJUSTED * P_RAW_FLOOR   # 0.011976047904191616
ALPHA = 0.05

PRIMARY_POP = "analysis_216"
SENSITIVITY_POP = "all_258"

# ── Contrast presentation ─────────────────────────────────────────────────────

FAMILY_ORDER = ["disclosure_presence", "response_strategy", "disclosure_degree"]
FAMILY_LABEL = {
    "disclosure_presence": "disclosure presence",
    "response_strategy": "response strategy",
    "disclosure_degree": "disclosure degree",
}

# Row order is fixed a priori by family (never sorted by effect size); within a
# family, larger n first.
CONTRASTS = [
    # key, display name, family
    ("broad_breach", "broad breach vs none", "disclosure_presence"),
    ("substantive_leak", "substantive leak vs rest", "disclosure_presence"),
    ("leak_vs_appropriate", "substantive leak vs appropriate", "disclosure_presence"),
    ("limiting_vs_direct", "limiting vs direct (all)", "response_strategy"),
    ("limiting_among_disclosers", "limiting vs direct (disclosers only)", "response_strategy"),
    ("degree_boundary_broadonly_vs_leaked", "broad-only vs substantive", "disclosure_degree"),
]
CONTRAST_KEYS = [c[0] for c in CONTRASTS]
DISPLAY = {k: d for k, d, _ in CONTRASTS}
FAMILY = {k: f for k, _, f in CONTRASTS}

# ── Verdict vocabulary ────────────────────────────────────────────────────────
# PR-AUC is the primary metric (pre-specified in the v2 analysis script docstring,
# committed before the corrected run — NOT registered in docs/PREREGISTRATION.md).
# ROC is supporting evidence and never rescues a PR-failing contrast.

VERDICTS = ("supported", "suggestive", "no_evidence")
VERDICT_ALIAS = {"supported": "confirmed"}   # historical wording, kept traceable
VERDICT_RULE = (
    "supported: Holm-adjusted permutation p on PR-AUC (primary metric) <= 0.05. "
    "suggestive: fails that test but Holm-adjusted permutation p on ROC-AUC <= 0.05. "
    "no_evidence: neither. ROC is supporting evidence only and never rescues a "
    "PR-failing contrast into 'supported'."
)


def verdict(p_holm_pr: float, p_holm_roc: float) -> str:
    if p_holm_pr <= ALPHA:
        return "supported"
    if p_holm_roc <= ALPHA:
        return "suggestive"
    return "no_evidence"


def marker_code(p_holm_pr: float, p_holm_roc: float) -> str:
    """Greyscale-safe verdict encoding used by the figures (fill only)."""
    return {"supported": "filled",
            "suggestive": "open_dot",
            "no_evidence": "open"}[verdict(p_holm_pr, p_holm_roc)]


# ── Formatting ────────────────────────────────────────────────────────────────

def fmt_p(p: float) -> str:
    """Render a permutation p-value, honouring the resolution floor.

    With n_perm=500 the smallest attainable raw p is 1/501; after Holm x6 the
    smallest attainable adjusted p is 6/501. Printing '0.0120' implies precision
    the design cannot deliver, so anything at (or below) the floor renders '<=.012'.
    """
    if p <= P_HOLM_FLOOR + 1e-12:
        return f"≤{P_HOLM_FLOOR:.3f}".replace("0.", ".")
    return f"{p:.3f}".replace("0.", ".")


def fmt_ci(lo: float, hi: float, nd: int = 3) -> str:
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


def fmt_num(x: float, nd: int = 3) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


# ── Hashing / provenance ──────────────────────────────────────────────────────

def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(REPO))


def git_state() -> dict:
    def run(*args):
        return subprocess.run(args, cwd=REPO, capture_output=True,
                              text=True).stdout.strip()
    dirty = run("git", "status", "--porcelain")
    dirty_paths = sorted(line[3:] for line in dirty.splitlines() if line.strip())
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
        "dirty_note": ("Working tree had uncommitted changes when this package was "
                       "built; the commit hash alone does not reproduce it. Inputs "
                       "are pinned by sha256 below."),
    }


def verify_inputs() -> dict:
    """Fail closed if any authoritative input changed. Returns {relpath: sha}."""
    seen = {}
    for relpath, expected in EXPECTED_SHA256.items():
        p = REPO / relpath
        if not p.exists():
            raise FileNotFoundError(f"authoritative input missing: {relpath}")
        got = sha256(p)
        if got != expected:
            raise ValueError(
                f"AUTHORITATIVE INPUT CHANGED: {relpath}\n  expected {expected}\n"
                f"  got      {got}\nRefusing to build. Investigate before proceeding.")
        seen[relpath] = got
    return seen


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_v2() -> pd.DataFrame:
    df = pd.read_csv(V2_CSV)
    meta = json.loads(V2_JSON.read_text())
    assert meta["n_perm"] == N_PERM, "n_perm drift vs v2 JSON"
    assert meta["n_boot"] == N_BOOT, "n_boot drift vs v2 JSON"
    assert meta["n_repeats"] == N_REPEATS, "n_repeats drift vs v2 JSON"
    assert meta["base_seed"] == BASE_SEED, "base_seed drift vs v2 JSON"
    assert not meta.get("quick_mode", False), "v2 results are from a --quick run"
    return df


def load_v1() -> pd.DataFrame:
    return pd.read_csv(V1_CSV)


def v2_protocol() -> dict:
    return json.loads(V2_JSON.read_text())


def scored(df: pd.DataFrame, pop: str) -> pd.DataFrame:
    """Scored rows for one population, in fixed family order."""
    sub = df[(df["status"] == "scored")
             & (df["contrast"].str.endswith("|" + pop))].copy()
    sub["key"] = sub["contrast"].str.split("|").str[0]
    sub = sub.set_index("key").loc[CONTRAST_KEYS].reset_index()
    sub["display"] = sub["key"].map(DISPLAY)
    sub["family"] = sub["key"].map(FAMILY)
    sub["verdict"] = [verdict(r.p_holm_pr_auc, r.p_holm_roc_auc)
                      for r in sub.itertuples()]
    sub["pr_excess_over_null"] = sub["pr_auc"] - sub["null_pr_auc_mean"]
    return sub


def population_composition() -> dict:
    """Composition of the 216 analysis population vs the 42 calibration cases.

    Used by the sensitivity table so the enrichment sits in the same glance as
    the AUC gap. Derived from the canonical labels CSV, never transcribed.
    """
    lab = pd.read_csv(LABELS)
    limiting = {"soft_deflection", "explicit_refusal", "mixed_disclose_then_limit"}
    out = {}
    for name, sub in (("analysis_216", lab[lab["population"] == "analysis"]),
                      ("calibration_42", lab[lab["population"] == "calibration"]),
                      ("all_258", lab)):
        vc = sub["reference_verification"].value_counts().to_dict()
        out[name] = {
            "n": int(len(sub)),
            "pct_limiting": float(sub["response_strategy"].isin(limiting).mean()),
            "pct_substantive_leak": float(sub["substantive_leak"].mean()),
            "pct_broad_breach": float(sub["broad_breach"].mean()),
            "reference_verification": {k: int(v) for k, v in vc.items()},
        }
    return out


# ── Registry: single source of truth for artifact metadata ────────────────────

_LABEL_CAVEAT = (
    "No interval or p-value here covers label uncertainty. All 216 analysis-population "
    "labels are single-judge and `provisional_unverified`; the same model family drafted "
    "the references and applied the rubric, and the judge prompt asserted the references "
    "were human-verified when 216/258 were not (audit §2.5.1). Attenuation is the "
    "likely direction, but systematic reference error could manufacture structure and is "
    "unquantified."
)
_DEPENDENCE_CAVEAT = (
    "The six contrasts are not independent samples: leak_vs_appropriate (n=81) and "
    "degree_boundary (n=188) share all 57 substantive-leak cases, and "
    "limiting_among_disclosers (n=188) is nested inside limiting_vs_direct (n=216). "
    "Holm is applied across the six scored contrasts within the population; under this "
    "dependence it is conservative-to-unclear rather than exact."
)
_GRID_EDGE_CAVEAT = (
    "Four of six contrasts select C at the grid floor (1e-7). Below that floor the AUC is "
    "flat and equals a standardised difference-of-means probe, so the optimum being outside "
    "the grid is moot (audit §2.4) — but it does mean the decodable signal is a "
    "class-mean direction, not a high-capacity decision boundary."
)
_CONSTRUCT_CAVEAT = (
    "The limiting construct is 'discloses and then limits', not deflection or refusal. In "
    "the 216 population 72 of 76 limiting cases are `mixed_disclose_then_limit`, 4 are "
    "`soft_deflection`, and there are zero `explicit_refusal`. The old deflection construct "
    "collapsed (36 → 7 cases) and is unmeasurable here."
)
_NO_BETWEEN_TEST = (
    "that any contrast is more decodable than any other — no between-contrast test was "
    "performed and none is powered; only leak_vs_appropriate vs degree_boundary have "
    "non-overlapping marginal intervals, which is not a test, and the samples are dependent"
)
_NO_CAUSAL = (
    "that the model uses this information causally — these are correlational decoding "
    "results at a single extraction point (layer 20, final prompt token)"
)

REGISTRY = {
    # ── tables ────────────────────────────────────────────────────────────────
    "tab_primary_contrasts_216": dict(
        kind="table", formats=["csv", "tex"], status="candidate_main",
        population=PRIMARY_POP,
        population_note=(
            "216 canonical tier-3 analysis cases; contrast-specific subsets range 81–216 "
            "and are printed per row. Excludes the 42 calibration cases."),
        metric=["pr_auc (primary)", "roc_auc (supporting)"],
        uncertainty=(
            "Hanley & McNeil (1982) analytic 95% CI and stratified prediction-resampling "
            "bootstrap 95% CI (n_boot=1000). Testing is by label permutation (n_perm=500) "
            "through the identical per-fold statistic, Holm-adjusted across the 6 scored "
            "contrasts within the population."),
        supports_claim=(
            "Under corrected statistics, four of six pre-specified behavioural contrasts are "
            "decodable above chance from layer-20 activations at the final prompt token; "
            "disclosure degree is not."),
        must_not_claim=[
            _NO_BETWEEN_TEST,
            ("that substantive_leak is decodable — its ROC interval excludes chance but "
             "it fails the primary Holm-adjusted PR test (p=.13) and is reported suggestive"),
            ("that PR-AUC above class prevalence indicates signal — the cross-validated "
             "PR null exceeds prevalence by ~0.05–0.06 at these n; the permutation null "
             "is the correct baseline"),
            _NO_CAUSAL,
        ],
        caveats=[_LABEL_CAVEAT, _DEPENDENCE_CAVEAT, _GRID_EDGE_CAVEAT, _CONSTRUCT_CAVEAT,
                 ("leak_vs_appropriate is a subsetted population (n=81): it drops 131 "
                  "broad_only and 4 refused cases, so its .704 prevalence is an artifact of "
                  "subsetting, and its 24 negatives are all provisional-labelled.")],
        paper_location="Main text, results table.",
        caption=(
            "Linear-probe decodability of six behavioural contrasts from layer-20 residual "
            "activations at the final prompt token (analysis population, n=216 scenarios; "
            "contrast-specific subsets as listed). PR-AUC is the pre-specified primary "
            "metric; its baseline is the empirical permutation-null mean, which exceeds "
            "class prevalence under cross-validation. p-values are label-permutation "
            "(n_perm=500) through the identical per-fold statistic, Holm-adjusted across "
            "the six contrasts."),
        replaceable_by_later_step=True,
        replacement_risk=(
            "Step 2 adds a matched text baseline and a privileged Δ = activation − "
            "text; the paper's headline quantity may become Δ rather than absolute AUC, "
            "in which case this table gains Δ columns or is superseded by a Δ table."),
        promotion_conditions=[
            "Remains the main results table unless Step 2's Δ supersedes absolute AUC as "
            "the headline quantity.",
            "Would be DEMOTED if the Step-2 text baseline shows the privileged increment is "
            "near zero for the contrasts presented here, or if the blind label audit "
            "(pipeline step 11) changes labels enough to move any verdict.",
        ],
    ),
    "tab_protocol_correction_v1_v2": dict(
        kind="table", formats=["csv", "tex"], status="candidate_appendix",
        population=PRIMARY_POP,
        population_note="Same 216 analysis population under both protocols.",
        metric=["roc_auc", "pr_auc", "permutation p"],
        uncertainty=(
            "Point estimates under each protocol; v1 reported only a repeated-split "
            "percentile band (not a confidence interval) and no multiplicity adjustment."),
        supports_claim=(
            "Pooling predict_proba scores across cross-validation folds that each select "
            "their own regularisation strength biases AUC downward; computing each metric "
            "within its held-out fold and averaging raises every contrast."),
        must_not_claim=[
            ("that v2 is 'better because the numbers went up' — the direction is a "
             "consequence of the specific defect (fold-boundary rank corruption), and the "
             "correction was specified before the corrected numbers were seen"),
            ("that this figure/table demonstrates all four corrections — the matched "
             "permutation null and the Holm adjustment are not visible in these columns"),
        ],
        caveats=[
            ("v1 and v2 are exactly comparable: identical data, contrasts, n, n_pos, seeds, "
             "fold assignments and C grid. Only the statistic differs."),
            ("v1 p-values were computed against a pooled-score null; they are shown for "
             "provenance, not as a valid comparison of significance."),
        ],
        paper_location="Appendix (methods).",
        caption=(
            "Effect of the statistical protocol correction on every scored contrast "
            "(analysis population). Identical data, contrasts, seeds, folds and "
            "regularisation grid; only the summary statistic differs."),
        replaceable_by_later_step=False,
        replacement_risk=(
            "None. This documents a completed methods correction and is not affected by "
            "later experiments."),
        promotion_conditions=[
            "Promote to main text only if the paper's framing centres the measurement/"
            "protocol-correction contribution.",
        ],
    ),
    "tab_interval_methods": dict(
        kind="table", formats=["csv", "tex"], status="candidate_appendix",
        population=PRIMARY_POP,
        population_note="Same 216 analysis population; widths are for ROC-AUC.",
        metric=["interval width (ROC-AUC)"],
        uncertainty=(
            "Compares three interval constructions: v1's repeated-split percentile band, "
            "Hanley analytic 95%, and stratified bootstrap 95%."),
        supports_claim=(
            "The repeated-cross-validation percentile band understates sampling uncertainty "
            "because repeats share the same cases; it measures split variability only."),
        must_not_claim=[
            ("that the bootstrap interval is a significance test — it is conditional on "
             "the fitted models and centred on the observed estimate; testing is by "
             "permutation"),
            ("that Hanley and the bootstrap should agree exactly — Hanley assumes one "
             "fixed scoring rule on independent cases, which an average of per-fold AUCs "
             "from refit models violates; it is reported as an approximation"),
        ],
        caveats=[
            ("v1's band is retained in this table only to quantify how narrow it was; it "
             "must never be presented as a competing confidence interval."),
            _LABEL_CAVEAT,
        ],
        paper_location="Appendix (methods).",
        caption=(
            "Width of three interval constructions for the same ROC-AUC point estimates. "
            "The repeated-split band reported previously is 1.5–2.5x narrower than "
            "either sampling-uncertainty interval."),
        replaceable_by_later_step=False,
        replacement_risk="None; documents a completed methods correction.",
        promotion_conditions=["Appendix only; no promotion path expected."],
    ),
    "tab_population_sensitivity_216_vs_258": dict(
        kind="table", formats=["csv", "tex"], status="sensitivity",
        population="analysis_216 (primary) and all_258 (superset, sensitivity only)",
        population_note=(
            "258 ⊃ 216. The 42 additional cases are calibration cases with "
            "human-verified references and a markedly different behaviour composition. "
            "This is not a replication and the 258 column is never the primary estimate."),
        metric=["roc_auc", "pr_auc", "Holm-adjusted permutation p (PR)"],
        uncertainty="As the primary table, computed independently within each population.",
        supports_claim=(
            "The direction and ordering of results is stable when the 42 calibration cases "
            "are added, with uniformly higher point estimates that are consistent with the "
            "calibration set's enrichment and cleaner references."),
        must_not_claim=[
            ("that the 258 results replicate the 216 results — 258 is a strict superset "
             "containing every 216 case, so the two are not independent samples"),
            ("that the higher 258 estimates are the better estimates — the added cases "
             "are enriched for limiting and leaking behaviour and are the only "
             "human-verified references, so composition and label quality are both "
             "confounded with population"),
        ],
        caveats=[
            ("Composition differs sharply: the 42 calibration cases oversample the old "
             "`refused` class by design, so limiting and leak rates are far above the "
             "analysis population's."),
            _LABEL_CAVEAT,
        ],
        paper_location="Appendix (sensitivity analysis).",
        caption=(
            "Sensitivity of every contrast to including the 42 calibration cases. The 258 "
            "population is a superset of the 216 analysis population, enriched for limiting "
            "and disclosing behaviour and differently verified; it is reported as a "
            "sensitivity analysis only."),
        replaceable_by_later_step=True,
        replacement_risk=(
            "If the calibration references are ever extended to the analysis population "
            "(human verification of all 216), this comparison is superseded by a single "
            "verified-label analysis."),
        promotion_conditions=[
            "Never promoted to a primary result. Would move to main text only as an "
            "explicit robustness paragraph.",
        ],
    ),
    # ── figures ───────────────────────────────────────────────────────────────
    "fig_contrast_effects_dual_metric_216": dict(
        kind="figure", formats=["pdf", "svg", "png"], status="candidate_appendix",
        population=PRIMARY_POP,
        population_note=(
            "216 canonical tier-3 analysis cases; contrast-specific subsets 81–216, "
            "printed in each row label."),
        metric=["roc_auc (left panel)", "pr_auc (right panel, primary)"],
        uncertainty=(
            "Left: Hanley 95% CI (thick) and stratified bootstrap 95% CI (thin, offset). "
            "Right: bootstrap 95% CI, against the empirical permutation-null mean. "
            "Marker fill encodes the Holm-adjusted permutation verdict."),
        supports_claim=(
            "Four contrasts are decodable above chance on the primary metric; the strategy "
            "contrasts and the leak-vs-appropriate contrast carry the largest excess over "
            "their permutation nulls, while disclosure degree shows none."),
        must_not_claim=[
            _NO_BETWEEN_TEST,
            ("that the ROC panel establishes significance — substantive_leak's ROC "
             "interval excludes chance while it fails the primary PR test; the PR panel and "
             "the marker fill carry the verdict"),
            ("that the prevalence tick is the PR chance level — it is not; the open "
             "square (permutation null) is"),
            _NO_CAUSAL,
        ],
        caveats=[_LABEL_CAVEAT, _DEPENDENCE_CAVEAT, _GRID_EDGE_CAVEAT, _CONSTRUCT_CAVEAT],
        paper_location=(
            "Appendix, companion to the primary results table. Not a main-text candidate "
            "while Step 2 is pending."),
        caption=(
            "Linear-probe decodability of six behavioural contrasts from layer-20 "
            "activations at the final prompt token (analysis population, n=216; "
            "contrast-specific subsets as labelled). Left: ROC-AUC with Hanley 95% CI "
            "(thick) and stratified bootstrap 95% CI (thin, offset below); dotted line "
            "marks chance, open ticks the ROC permutation null. Right: PR-AUC (primary "
            "metric) with bootstrap 95% CI; the open square is the empirical "
            "permutation-null mean — the correct baseline, which exceeds class "
            "prevalence (faint tick) by 0.05–0.06 at these sample sizes. Marker fill: "
            "filled = survives Holm adjustment on PR-AUC; open with centre dot = survives "
            "on ROC only; open = neither. The contrasts are separate tests on overlapping "
            "case sets; intervals are marginal and no between-contrast difference was "
            "tested."),
        replaceable_by_later_step=True,
        replacement_risk=(
            "High. Step 2's activation-minus-text Δ figure presents the same contrasts "
            "with a stronger claim (privileged information beyond the visible text) and "
            "would likely take this figure's place."),
        promotion_conditions=[
            "Promote to candidate_main only if Step 2's Δ is null or unusable, leaving "
            "absolute decodability as the strongest available result.",
            "Requires the Step-2 text baseline recomputed under the v2 per-fold protocol "
            "before any Δ annotation could be added to it.",
        ],
    ),
    "fig_protocol_correction_v1_v2": dict(
        kind="figure", formats=["pdf", "svg", "png"], status="candidate_appendix",
        population=PRIMARY_POP,
        population_note="Same 216 analysis population under both protocols.",
        metric=["roc_auc", "pr_auc", "interval width"],
        uncertainty=(
            "Panel A shows point estimates under each protocol. Panel B compares interval "
            "widths by construction method and is itself the uncertainty result."),
        supports_claim=(
            "Two of the four statistical corrections, shown directly: fold-wise metric "
            "computation raises every contrast, and the previously reported band was split "
            "variability rather than sampling uncertainty."),
        must_not_claim=[
            ("that all four corrections are visible here — the matched permutation null "
             "and the Holm adjustment are not visualisable and live in the correction table"),
            ("that the v1 band is a confidence interval — it is a repeated-split "
             "percentile range on the same cases"),
            ("that rising numbers validate the new protocol — the correction is "
             "justified by the fold-boundary rank-corruption mechanism, not by its direction"),
        ],
        caveats=[
            ("v1 and v2 differ only in the summary statistic; data, contrasts, seeds, folds "
             "and C grid are identical."),
            ("Panel B's Hanley width is an approximation: Hanley assumes a single fixed "
             "scoring rule on independent cases, which an average of per-fold AUCs from "
             "refit models violates."),
        ],
        paper_location="Appendix (methods). The most durable Step-1 figure.",
        caption=(
            "Effect of the statistical protocol correction. (A) Pooling predict_proba "
            "scores across cross-validation folds that each select their own regularisation "
            "strength (v1) corrupts the pooled ranking and biases AUC downward; computing "
            "each metric within its held-out fold and averaging (v2) raises every contrast. "
            "Identical data, contrasts, seeds, folds and regularisation grid; only the "
            "statistic differs. (B) The interval reported under v1 was the 2.5/97.5 "
            "percentile over 20 repeated splits of the same cases — split variability, "
            "not sampling uncertainty. Hanley analytic and stratified bootstrap 95% "
            "intervals are substantially wider."),
        replaceable_by_later_step=False,
        replacement_risk=(
            "None. Later steps adopt this protocol rather than superseding it."),
        promotion_conditions=[
            "Promote to main text if the paper centres the measurement-validity "
            "contribution; otherwise appendix.",
        ],
    ),
}

STATUS_VOCAB = ["candidate_main", "candidate_appendix", "exploratory",
                "sensitivity", "deprecated"]
