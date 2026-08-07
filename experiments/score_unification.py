"""Harvest + score the unification campaign (paper2) under the paper's smearing contract.

Reads the per-task npz trees written by ``src/unification.py`` (the executor is
the source of truth for the schema: row_id, t, y_fit, yhat, rv_raw, baseline,
valid_mask + scalar suffstats + json meta), scores every (root, arm) pair, and
emits:

  * one CSV row per (root, arm)                    (``--out``)
  * ``<tex-dir>/campaign_numbers.tex``             (scalar macros)
  * ``<tex-dir>/table_buckets.tex``                (tabular body, buckets.tex format)
  * ``<tex-dir>/table_lasso_ridge.tex``            (tabular body, dense_weak.tex format)
  * ``<tex-dir>/table_blocks.tex``                 (tabular body: ladder + diagnostics)

Scoring contract (writeup/sections/smearing.tex — the causal calibrated
second-moment stack; author directive 2026-08-06, replaces the in-window
scalar smear entirely): one evaluation window == one CHUNK, chunks processed
in strictly ascending index order (they tile the panel in row order).  Per
window k, with ``y_raw = sqrt(rv_raw / B)`` the unwinsorized sqrt-scale
target (nonneg-guarded):

  * MEAN CALIBRATION (causal): ``m_t = a + b*yhat_t`` with (a, b) the
    Mincer-Zarnowitz OLS of y_raw on [1, yhat] over window k-1's valid bars;
    a window whose predecessor is missing from the harvest (mid-panel gap —
    zero on a complete harvest) falls back to the identity (0, 1), flagged.
  * SCALAR SECOND MOMENT (causal; FINAL contract, author decision
    2026-08-06): ``sigma2_k = mean(e^2)`` over window k-1's valid bars, with
    e^2 = (y_raw - m)^2 and m as applied THERE — one scalar per window,
    previous-window estimated.  Positivity is trivial (a mean of squares)
    and the estimator cannot extrapolate.  Two bar-conditional
    parameterizations (affine with zero floor; log-linear with Duan
    retransformation) were implemented and REJECTED on measurement — the
    documented negative result is smearing.tex's "conditional-variance
    record".  Mid-panel harvest gap (no predecessor present): in-window
    scalar mean, flagged (``var_fallback_windows``).
  * RAW FORECAST: ``f_t = (m_t^2 + sigma2_t) * B_t``; raw target = the
    PERSISTED unwinsorized ``rv_raw`` (never y_fit^2 * B).  Per-bar QLIKE
    ``r - log r - 1`` with ``r = rv_raw / f``, excluding bars where either
    member of the pair is non-positive — the exact metrics.py exclusion rule,
    via ``src/evaluation/diebold_mariano.qlike_per_bar`` (REUSED).  Pooled
    arm QLIKE = mean over all included bars.

CALIBRATION BURN-IN (author decision 2026-08-07): the FIRST PRESENT
evaluation window of each (root, arm) — chunk 0 for full-coverage arms,
chunk 11 for the legality-89 arms — only estimates the maps for its
successor (its in-sample MZ fit supplies the m its successor's variance
residuals are measured against) and is NOT scored: none of its bars enters
pooled QLIKE, DM joins, MZ diagnostics, or the sensitivity conventions, so
all three conventions score identical row sets.  Scoring begins at the
second present window; every scored forecast is computable at its own bar
with no exceptions.  The mid-panel fallbacks remain only for genuine
harvest gaps and are expected 0/0 on complete harvests (a warning prints
if they fire).

Ladder-increment inference (the product / transmission marginal-value
statistics): paired per-bar DM between ADJACENT ladder rungs — (blk3, blk2)
and (blk4, blk3) under each convention (user / doc / tuned) — from the same
joined per-bar contract losses the a0 comparisons use (join on row_id,
dm_test on the loss differential).  Emitted as
results/unification_increments.csv (sibling of --out) and macros
\\unifIncr<UpperRungCamel>{DM,DQ} (e.g. \\unifIncrBlkFourUserDM for
blk4_user vs blk3_user), pending until BOTH arms are complete.

Smear-sensitivity layer (author directive, exactly three conventions): pooled
QLIKE is also computed under (1) NONE — ``f = yhat^2 * B``, no correction —
and (2) DUAN — the traditional in-window scalar smear from FIT-scale
residuals, ``f = (yhat^2 + mean((y_fit - yhat)^2)) * B`` — on the same arrays
and exclusion rule.  These are comparison columns only (CSV qlike_none /
qlike_duan, table_smear_sensitivity.tex, Kendall-tau macros
\\unifSmearTauNone / \\unifSmearTauDuan); the CONTRACT convention above is
the sole basis for the headline macros, DM tests, and tables.

Same-environment comparisons: each arm vs ITS OWN root's ``a0_ols_har``, joined
per-bar on row_id.  DM t reuses ``src/evaluation/diebold_mariano.dm_test``
(Newey-West HAC, automatic lag floor(4*(T/100)^(2/9)), Harvey-Leybourne-Newbold
small-sample correction — the repo's established DM utility).  OOS R^2 is on
the sqrt FIT scale: 1 - SSE_arm/SSE_a0 from per-bar (y_fit - yhat)^2 on the
joined rows.  MZ alpha/beta reuse ``src/evaluation/metrics.mz_regression``
(rv_raw on the raw forecast, positive pairs).

Single-cluster campaign (author directive 2026-08-06): the cross-cluster a0
float-parity canary was retired with the Hoffman2 leg; ``--roots`` still
accepts several trees and each is scored independently.

Partial-harvest robust: runs cleanly at ANY completion level (zero files emits
all-pending macros and pending tables), always exits 0, one summary line per
(root, arm).  Float64 throughout; numpy only for the core scoring.

CLI:
    python experiments/score_unification.py \
        --roots results/unification_carc results/unification_h2 \
        --out results/unification_scores.csv --tex-dir writeup/generated
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np

# ── repo-root bootstrap (script may be invoked from any cwd) ──────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.evaluation.diebold_mariano import dm_test, qlike_per_bar  # noqa: E402
from src.evaluation.metrics import mz_regression  # noqa: E402

EXPECTED_CHUNKS = 100
A0 = "a0_ols_har"
# Chunks that are illegal BY DESIGN for the 24000-bar-window product /
# transmission arms (selection/frame block + training window -> first legal
# OOS row 48000). v3 geometry (panel 300,317 -> ~2,763-bar chunks): row
# 48000 falls in chunk 9, so chunks 0-8 are excluded and these arms are
# complete at 91.
_LEGAL_MISSING: dict[str, set[int]] = {
    arm: set(range(9))
    for arm in (
        "blk3_user",
        "blk4_user",
        "c4_product_alone_user",
        "d3_transmission_alone_user",
        # tuned twins run the same 24000-bar window; if their harvest covers
        # only rows 48000+ the same legality rule applies (complete at 89),
        # and a full 100-chunk harvest is simply never "missing" these.
        "blk3_tuned",
        "blk4_tuned",
        "c4_product_alone_tuned",
        "d3_transmission_alone_tuned",
        # transmission-revival arms: same 24000-bar window + frozen frame ->
        # same first legal OOS row 48000
        "d3_transmission_alone_trail",
        "blk4_trail",
        # ablation triple (same legality)
        "blk4_trailG",
        "blk4_trailGhat",
        "blk4_trail_tuned",
    )
}
_CHUNK_RE = re.compile(r"^chunk_(\d+)\.npz$")

# ── arm registry facts (mirrors src/unification.py ARMS; static so the scorer
#    never needs the panel, the executor's heavy deps, or the cluster env) ─────
#    camel: LaTeX macro stem (\unif<camel>QLIKE / \unif<camel>DM). Names for
#    b1/b2/blk* match the pre-existing placeholder writeup/generated/
#    campaign_numbers.tex exactly (BOneRidge, BTwoLasso, BlkTwoUser, ...).

_BUCKET_TEX = {
    "a_bucket_all_features": r"all\_features (joint)",
    "a_bucket_moments": "moments",
    "a_bucket_liquidity": "liquidity",
    "a_bucket_implied_vol": r"implied\_vol",
    "a_bucket_market_vw": r"market\_vw",
    "a_bucket_market_ew": r"market\_ew",
    "a_bucket_vol_demand": r"vol\_demand",
    "a_bucket_sentiment": "sentiment",
}

# (arm, camel) in canonical registry order.
_REGISTRY: list[tuple[str, str]] = [
    (A0, "Azero"),
    ("a_bucket_moments", "Moments"),
    ("a_bucket_liquidity", "Liquidity"),
    ("a_bucket_market_ew", "MarketEw"),
    ("a_bucket_market_vw", "MarketVw"),
    ("a_bucket_sentiment", "Sentiment"),
    ("a_bucket_implied_vol", "ImpliedVol"),
    ("a_bucket_vol_demand", "VolDemand"),
    ("a_bucket_all_features", "AllFeatures"),
    ("b1_ridge", "BOneRidge"),
    ("b2_lasso", "BTwoLasso"),
    # Causally-tuned penalty controls for the head-to-head (battery protocol:
    # periodic causal forward-split re-selection; window 24000, full design).
    ("b1_ridge_tuned", "BOneRidgeTuned"),
    ("b2_lasso_tuned", "BTwoLassoTuned"),
    # v2 addition: causally tuned elastic net (battery reclasticnet grid,
    # window 24000, full design). tab:lasso_vs_ridge is macro-wired in
    # dense_weak.tex, so the macro trio below is the complete wiring.
    ("b3_enet_tuned", "BThreeEnetTuned"),
    # Merged-§4/§5 bucket grid: per-bucket causally tuned ridge / free-l1 enet
    # (8 designs = 7 single buckets + the joint all_features, as with A1..A8).
    ("br_tuned_moments", "BrTunedMoments"),
    ("br_tuned_liquidity", "BrTunedLiquidity"),
    ("br_tuned_market_ew", "BrTunedMarketEw"),
    ("br_tuned_market_vw", "BrTunedMarketVw"),
    ("br_tuned_sentiment", "BrTunedSentiment"),
    ("br_tuned_implied_vol", "BrTunedImpliedVol"),
    ("br_tuned_vol_demand", "BrTunedVolDemand"),
    ("br_tuned_all_features", "BrTunedAllFeatures"),
    ("be_tuned_moments", "BeTunedMoments"),
    ("be_tuned_liquidity", "BeTunedLiquidity"),
    ("be_tuned_market_ew", "BeTunedMarketEw"),
    ("be_tuned_market_vw", "BeTunedMarketVw"),
    ("be_tuned_sentiment", "BeTunedSentiment"),
    ("be_tuned_implied_vol", "BeTunedImpliedVol"),
    ("be_tuned_vol_demand", "BeTunedVolDemand"),
    ("be_tuned_all_features", "BeTunedAllFeatures"),
    # Penalty-jiggle appendix arms: b1_ridge at fixed alternative alphas
    # (LaTeX-safe camels — macro names cannot carry bare digits).
    ("b1_ridge_a0p1", "BOneRidgeApOne"),
    ("b1_ridge_a0p3", "BOneRidgeApThree"),
    ("b1_ridge_a3", "BOneRidgeAthree"),
    ("b1_ridge_a10", "BOneRidgeAten"),
    ("blk2_user", "BlkTwoUser"),
    ("blk3_user", "BlkThreeUser"),
    ("blk4_user", "BlkFourUser"),
    ("blk2_doc", "BlkTwoDoc"),
    ("blk3_doc", "BlkThreeDoc"),
    ("blk4_doc", "BlkFourDoc"),
    # Causally-tuned block-ladder arms (per-block causal penalty selection).
    ("blk2_tuned", "BlkTwoTuned"),
    ("blk3_tuned", "BlkThreeTuned"),
    ("blk4_tuned", "BlkFourTuned"),
    # Transmission-revival: trailing-standardized transmission (causal
    # replacement for the full-sample-standardization look-ahead).
    ("blk4_trail", "BlkFourTrail"),
    # Ablation triple: factor levels only / lead-lag flow only / tuned contender.
    ("blk4_trailG", "BlkFourTrailG"),
    ("blk4_trailGhat", "BlkFourTrailGhat"),
    ("blk4_trail_tuned", "BlkFourTrailTuned"),
    ("c4_product_alone_user", "ProductAloneUser"),
    ("c4_product_alone_doc", "ProductAloneDoc"),
    ("c4_product_alone_tuned", "ProductAloneTuned"),
    ("d3_transmission_alone_user", "TransAloneUser"),
    ("d3_transmission_alone_doc", "TransAloneDoc"),
    ("d3_transmission_alone_tuned", "TransAloneTuned"),
    ("d3_transmission_alone_trail", "TransAloneTrail"),
    # Tree bank synthetics (assembled scorer-side from the 20 tree_expert_*
    # banks; the expert arms themselves are CSV-only — digits are illegal in
    # LaTeX macro stems, and the paper quotes the composites, not the experts).
    ("tree_tuned", "TreeTuned"),
    ("tree_hedge", "TreeHedge"),
]
_CAMEL = dict(_REGISTRY)
_ORDER = {arm: i for i, (arm, _) in enumerate(_REGISTRY)}

# Reader-facing descriptive names (appendix master table; used in ALL
# generated fragments with arm-name rows — underscores never reader-facing).
_ARM_TEX: dict[str, str] = {
    A0: "OLS on HAR + calendar (benchmark)",
    "a_bucket_moments": "HAR + moments bucket (OLS)",
    "a_bucket_liquidity": "HAR + liquidity bucket (OLS)",
    "a_bucket_market_ew": "HAR + market EW bucket (OLS)",
    "a_bucket_market_vw": "HAR + market VW bucket (OLS)",
    "a_bucket_sentiment": "HAR + sentiment bucket (OLS)",
    "a_bucket_implied_vol": "HAR + implied vol bucket (OLS)",
    "a_bucket_vol_demand": "HAR + vol demand bucket (OLS)",
    "a_bucket_all_features": "HAR + all features bucket (OLS)",
    "b1_ridge": r"Ridge, fixed $\alpha=1$",
    "b2_lasso": r"Lasso, fixed $\alpha=10^{-4}$",
    "b1_ridge_tuned": "Ridge, causally tuned",
    "b2_lasso_tuned": "Lasso, causally tuned",
    "b3_enet_tuned": "Elastic net, causally tuned",
    "br_tuned_moments": "HAR + moments bucket (ridge, causally tuned)",
    "br_tuned_liquidity": "HAR + liquidity bucket (ridge, causally tuned)",
    "br_tuned_market_ew": "HAR + market EW bucket (ridge, causally tuned)",
    "br_tuned_market_vw": "HAR + market VW bucket (ridge, causally tuned)",
    "br_tuned_sentiment": "HAR + sentiment bucket (ridge, causally tuned)",
    "br_tuned_implied_vol": "HAR + implied vol bucket (ridge, causally tuned)",
    "br_tuned_vol_demand": "HAR + vol demand bucket (ridge, causally tuned)",
    "br_tuned_all_features": "HAR + all features bucket (ridge, causally tuned)",
    "be_tuned_moments": "HAR + moments bucket (elastic net, causally tuned, free mixing)",
    "be_tuned_liquidity": "HAR + liquidity bucket (elastic net, causally tuned, free mixing)",
    "be_tuned_market_ew": "HAR + market EW bucket (elastic net, causally tuned, free mixing)",
    "be_tuned_market_vw": "HAR + market VW bucket (elastic net, causally tuned, free mixing)",
    "be_tuned_sentiment": "HAR + sentiment bucket (elastic net, causally tuned, free mixing)",
    "be_tuned_implied_vol": "HAR + implied vol bucket (elastic net, causally tuned, free mixing)",
    "be_tuned_vol_demand": "HAR + vol demand bucket (elastic net, causally tuned, free mixing)",
    "be_tuned_all_features": "HAR + all features bucket (elastic net, causally tuned, free mixing)",
    "tree_tuned": "Gradient-boosted trees (causally tuned expert bank)",
    "tree_hedge": "Gradient-boosted trees (hedged expert bank)",
    "b1_ridge_a0p1": r"Ridge, fixed $\alpha=0.1$",
    "b1_ridge_a0p3": r"Ridge, fixed $\alpha=0.3$",
    "b1_ridge_a3": r"Ridge, fixed $\alpha=3$",
    "b1_ridge_a10": r"Ridge, fixed $\alpha=10$",
    "blk2_user": "Two-block ridge (stated penalties)",
    "blk3_user": "Three-block ridge (stated penalties)",
    "blk4_user": "Four-block ridge (stated penalties)",
    "blk2_doc": "Two-block ridge (documented penalties)",
    "blk3_doc": "Three-block ridge (documented penalties)",
    "blk4_doc": "Four-block ridge (documented penalties)",
    "blk2_tuned": "Two-block ridge (per-block causal tuning)",
    "blk3_tuned": "Three-block ridge (per-block causal tuning)",
    "blk4_tuned": "Four-block ridge (per-block causal tuning)",
    "blk4_trail": "Four-block ridge (trailing-standardized transmission)",
    "blk4_trailG": "Four-block ridge (trailing, factor levels only)",
    "blk4_trailGhat": "Four-block ridge (trailing, lead-lag flow only)",
    "blk4_trail_tuned": "Four-block ridge (trailing transmission, per-block causal tuning)",
    "c4_product_alone_user": "HAR + product block only (stated)",
    "c4_product_alone_doc": "HAR + product block only (documented)",
    "c4_product_alone_tuned": "HAR + product block only (tuned)",
    "d3_transmission_alone_user": "HAR + transmission block only (stated)",
    "d3_transmission_alone_doc": "HAR + transmission block only (documented)",
    "d3_transmission_alone_tuned": "HAR + transmission block only (tuned)",
    "d3_transmission_alone_trail": "HAR + transmission block only (trailing)",
}

# blocks table: (arm, window bars, per-block alpha string) — 6 ladder + 4 diag,
# windows/alphas per src/unification.py (USER_ALPHAS/DOC_ALPHAS, DOC 250d=12000).
_BLOCKS_TABLE: list[tuple[str, int, str]] = [
    ("blk2_user", 24000, "1/100"),
    ("blk3_user", 24000, "1/100/1000"),
    ("blk4_user", 24000, "1/100/1000/1000"),
    ("blk2_doc", 12000, "1/3e3"),
    ("blk3_doc", 12000, "1/3e3/3e4"),
    ("blk4_doc", 12000, "1/3e3/3e4/3e3"),
    ("blk2_tuned", 24000, "tuned (per-block causal)"),
    ("blk3_tuned", 24000, "tuned (per-block causal)"),
    ("blk4_tuned", 24000, "tuned (per-block causal)"),
    ("blk4_trail", 24000, "1/100/1000/1000 (trailing-std transmission)"),
    ("blk4_trailG", 24000, "1/100/1000/1000 (trailing, G only)"),
    ("blk4_trailGhat", 24000, "1/100/1000/1000 (trailing, Ghat only)"),
    ("blk4_trail_tuned", 24000, "tuned (per-block causal; trailing transmission)"),
    ("c4_product_alone_user", 24000, "1/1000"),
    ("c4_product_alone_doc", 12000, "1/3e4"),
    ("c4_product_alone_tuned", 24000, "tuned (per-block causal)"),
    ("d3_transmission_alone_user", 24000, "1/1000"),
    ("d3_transmission_alone_doc", 12000, "1/3e3"),
    ("d3_transmission_alone_tuned", 24000, "tuned (per-block causal)"),
    ("d3_transmission_alone_trail", 24000, "1/1000 (trailing-std transmission)"),
]

# Adjacent-rung increment pairs (upper rung first): the paper's
# product-increment (blk3 vs blk2) and transmission-increment (blk4 vs blk3)
# statistics, per ladder convention. Macros are keyed by the UPPER rung.
_INCREMENT_PAIRS: list[tuple[str, str]] = [
    ("blk3_user", "blk2_user"),
    ("blk4_user", "blk3_user"),
    ("blk3_doc", "blk2_doc"),
    ("blk4_doc", "blk3_doc"),
    ("blk3_tuned", "blk2_tuned"),
    ("blk4_tuned", "blk3_tuned"),
    # the revival's verdict statistic: trailing-transmission increment over the
    # same untouched 3-block lower rung -> \unifIncrBlkFourTrail{DM,DQ}
    ("blk4_trail", "blk3_user"),
    # ablation triple verdicts: each component's increment over its matching
    # lower rung -> \unifIncrBlkFourTrailG{DM,DQ}, \unifIncrBlkFourTrailGhat
    # {DM,DQ}, \unifIncrBlkFourTrailTuned{DM,DQ}
    ("blk4_trailG", "blk3_user"),
    ("blk4_trailGhat", "blk3_user"),
    ("blk4_trail_tuned", "blk3_tuned"),
    # bucket-grid within-design family comparisons (free-l1 enet vs tuned
    # ridge on the SAME design) -> \unifIncrBeTuned<Bucket>{DM,DQ}
    ("be_tuned_moments", "br_tuned_moments"),
    ("be_tuned_liquidity", "br_tuned_liquidity"),
    ("be_tuned_market_ew", "br_tuned_market_ew"),
    ("be_tuned_market_vw", "br_tuned_market_vw"),
    ("be_tuned_sentiment", "br_tuned_sentiment"),
    ("be_tuned_implied_vol", "br_tuned_implied_vol"),
    ("be_tuned_vol_demand", "br_tuned_vol_demand"),
    ("be_tuned_all_features", "br_tuned_all_features"),
    # tree-bank verdicts (all tree_* joins are 2003+ by construction: the
    # synthetic arms are assembled on the dev-prefix-excluded row set)
    ("tree_tuned", "blk3_tuned"),
    ("tree_tuned", "blk4_trail_tuned"),
    ("tree_hedge", "tree_tuned"),
]


# ── per-(root, arm) container ─────────────────────────────────────────────────


@dataclass
class ArmResult:
    root: str  # root label (basename of the root dir)
    arm: str
    n_chunks: int = 0
    missing: list[int] = field(default_factory=list)
    extra: list[int] = field(default_factory=list)
    incomplete: bool = True
    n_rows: int = 0
    n_valid: int = 0
    n_invalid: int = 0
    n_qlike: int = 0  # bars entering the pooled QLIKE mean
    n_dupes: int = 0
    n_gaps: int = 0
    contiguous_in_chunk: bool = True
    qlike: float | None = None
    sigma2_mean: float | None = None  # mean per-bar sigma2_t over valid bars
    calib_a_mean: float | None = None  # mean fitted MZ intercept over windows
    calib_b_mean: float | None = None  # mean fitted MZ slope over windows
    calib_fallback_windows: int = 0  # windows scored with the identity (0, 1)
    var_fallback_windows: int = 0  # windows without the 2-param variance fit
    f_min: float | None = None  # min finite raw forecast (positivity check)
    qlike_none: float | None = None  # sensitivity: no correction
    qlike_duan: float | None = None  # sensitivity: in-window fit-scale Duan
    masked_col_events: int = 0
    dropped_col_events: int = 0
    dm_t: float | None = None
    oos_r2: float | None = None
    mz_alpha: float | None = None
    mz_beta: float | None = None
    mz_r2: float | None = None
    n_asym: int | None = None
    warnings: list[str] = field(default_factory=list)
    # accumulators for the causal calibration stack
    _ab: list[tuple[float, float]] = field(default_factory=list)
    _s2_sum: float = 0.0
    _s2_n: int = 0
    # per-bar arrays (sorted by row_id, deduped)
    row_id: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    t: np.ndarray = field(default_factory=lambda: np.empty(0, dtype="datetime64[ns]"))
    loss: np.ndarray = field(default_factory=lambda: np.empty(0))
    loss_none: np.ndarray = field(default_factory=lambda: np.empty(0))
    loss_duan: np.ndarray = field(default_factory=lambda: np.empty(0))
    err2: np.ndarray = field(default_factory=lambda: np.empty(0))
    f_raw: np.ndarray = field(default_factory=lambda: np.empty(0))
    rv_raw: np.ndarray = field(default_factory=lambda: np.empty(0))
    yhat: np.ndarray = field(default_factory=lambda: np.empty(0))


def _load_chunk(path: str) -> dict:
    """Load one chunk npz (raw arrays only; scoring is sequential per arm)."""
    with np.load(path, allow_pickle=False) as z:
        out = {
            "row_id": np.asarray(z["row_id"], dtype=np.int64),
            "t": np.asarray(z["t"], dtype="datetime64[ns]"),
            "y_fit": np.asarray(z["y_fit"], dtype=np.float64),
            "yhat": np.asarray(z["yhat"], dtype=np.float64),
            "rv_raw": np.asarray(z["rv_raw"], dtype=np.float64),
            "baseline": np.asarray(z["baseline"], dtype=np.float64),
            "valid": np.asarray(z["valid_mask"], dtype=bool),
        }
        masked = int(z["ols_masked_cols"]) if "ols_masked_cols" in z.files else -1
        dropped = int(z["ols_dropped_cols"]) if "ols_dropped_cols" in z.files else 0
        if masked < 0 and "meta" in z.files:
            meta = json.loads(str(z["meta"]))
            mc = meta.get("ols_masked_cols", {})
            masked = len(mc) if isinstance(mc, dict) else int(mc)
    out["masked"] = max(masked, 0)
    out["dropped"] = dropped
    return out


def _ols2(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    """OLS of y on [1, x]; None when degenerate (rank < 2 or non-finite coef)."""
    design = np.column_stack([np.ones_like(x), x])
    coef, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < 2 or not np.all(np.isfinite(coef)):
        return None
    return float(coef[0]), float(coef[1])


def _y_raw_of(rv_raw: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Unwinsorized sqrt-scale target sqrt(rv_raw / B); NaN where undefined
    (nonneg guard: baseline must be positive, rv_raw nonnegative)."""
    y_raw = np.full(len(rv_raw), np.nan)
    ok = np.isfinite(rv_raw) & np.isfinite(baseline) & (baseline > 0) & (rv_raw >= 0)
    y_raw[ok] = np.sqrt(rv_raw[ok] / baseline[ok])
    return y_raw


def _burnin_state(c: dict) -> dict:
    """CALIBRATION BURN-IN (author decision 2026-08-07): the first present
    window of an arm estimates the maps for its successor and is NOT scored.

    Its chain state carries its own in-sample MZ mean m — the mean the
    successor's variance residuals are measured against. If the in-sample
    fit is degenerate the identity stands in; the successor's own mean fit
    is then degenerate too and flags itself."""
    yhat, valid = c["yhat"], c["valid"]
    y_raw = _y_raw_of(c["rv_raw"], c["baseline"])
    a, b = 0.0, 1.0
    s = valid & np.isfinite(y_raw) & np.isfinite(yhat)
    if s.sum() >= 3:
        fit = _ols2(yhat[s], y_raw[s])
        if fit is not None:
            a, b = fit
    return {"yhat": yhat, "y_raw": y_raw, "m": a + b * yhat, "valid": valid}


def _score_chunk_causal(c: dict, prev: dict | None, res: ArmResult) -> dict:
    """Score one evaluation window under the calibrated second-moment stack
    (smearing.tex): the mean map and the scalar variance are both fit on the
    PREVIOUS window and applied here — causal, hence deployable."""
    yhat, rv_raw, baseline, valid = c["yhat"], c["rv_raw"], c["baseline"], c["valid"]
    n = len(yhat)
    # 1. unwinsorized sqrt-scale target
    y_raw = _y_raw_of(rv_raw, baseline)

    def _sel(p: dict) -> np.ndarray:
        return p["valid"] & np.isfinite(p["y_raw"]) & np.isfinite(p["yhat"])

    # 2. causal mean calibration: MZ OLS of y_raw on [1, yhat], previous window
    a, b = 0.0, 1.0
    fitted = False
    if prev is not None:
        s = _sel(prev)
        if s.sum() >= 3:
            fit = _ols2(prev["yhat"][s], prev["y_raw"][s])
            if fit is not None:
                a, b = fit
                fitted = True
    if fitted:
        res._ab.append((a, b))
    else:
        # identity (0, 1) applied — mid-panel harvest gap only (the first
        # present window never reaches here: it is the calibration burn-in)
        res.calib_fallback_windows += 1
    m = a + b * yhat

    # 3. causal SCALAR second moment (FINAL contract, author decision
    # 2026-08-06): sigma2_k = mean of window k-1's e^2 = (y_raw - m)^2, m as
    # applied there — one scalar per window, previous-window estimated.
    # Positivity is trivial (mean of squares); no extrapolation. Two
    # bar-conditional parameterizations were implemented and rejected on
    # measurement (affine zero-floor: forecast collapse on quiet bars;
    # log-linear + Duan retransformation: exponential extrapolation to ~1e44
    # on extreme bars) — smearing.tex, "the conditional-variance record".
    sigma2: np.ndarray | None = None
    var_fitted = False
    if prev is not None:
        s = _sel(prev) & np.isfinite(prev["m"])
        if s.sum() >= 3:
            e2 = (prev["y_raw"][s] - prev["m"][s]) ** 2
            sigma2 = np.full(n, float(np.mean(e2)))
            var_fitted = True
    if sigma2 is None:  # mid-panel harvest gap: in-window scalar (never on
        # a complete harvest — the first present window is the burn-in)
        s0 = valid & np.isfinite(y_raw)
        s_val = float(np.mean((y_raw[s0] - m[s0]) ** 2)) if s0.any() else np.nan
        sigma2 = np.full(n, s_val)
    if not var_fitted:
        res.var_fallback_windows += 1

    # 4. raw forecast + per-bar QLIKE (metrics.py exclusion rule, REUSED)
    f = np.full(n, np.nan)
    okf = valid & np.isfinite(m) & np.isfinite(sigma2) & np.isfinite(baseline)
    f[okf] = (m[okf] ** 2 + sigma2[okf]) * baseline[okf]
    loss = qlike_per_bar(f, rv_raw)
    loss[~valid] = np.nan
    err2 = np.full(n, np.nan)
    y_fit = c["y_fit"]
    err2[valid] = (y_fit[valid] - yhat[valid]) ** 2
    sv = valid & np.isfinite(sigma2)
    res._s2_sum += float(np.sum(sigma2[sv]))
    res._s2_n += int(sv.sum())

    # 5. sensitivity conventions (comparison columns only): NONE = naive
    # squared back-transform; DUAN = traditional in-window scalar smear from
    # FIT-scale residuals (the textbook analytic form as first implemented
    # here). Same arrays, same exclusion rule; the contract stays the headline.
    f_none = np.full(n, np.nan)
    f_duan = np.full(n, np.nan)
    okb = valid & np.isfinite(baseline)
    f_none[okb] = yhat[okb] ** 2 * baseline[okb]
    if valid.any():
        s2_fit = float(np.mean((y_fit[valid] - yhat[valid]) ** 2))
        f_duan[okb] = (yhat[okb] ** 2 + s2_fit) * baseline[okb]
    loss_none = qlike_per_bar(f_none, rv_raw)
    loss_none[~valid] = np.nan
    loss_duan = qlike_per_bar(f_duan, rv_raw)
    loss_duan[~valid] = np.nan
    return {
        "row_id": c["row_id"],
        "loss": loss,
        "loss_none": loss_none,
        "loss_duan": loss_duan,
        "err2": err2,
        "f": f,
        "rv_raw": rv_raw,
        "yhat": yhat,
        "valid": valid,
        # chain state consumed by the NEXT window's calibration fits
        "state": {"yhat": yhat, "y_raw": y_raw, "m": m, "valid": valid},
    }


def _harvest_arm(root_path: str, root_label: str, arm: str) -> ArmResult:
    res = ArmResult(root=root_label, arm=arm)
    arm_dir = os.path.join(root_path, arm)
    found: dict[int, str] = {}
    for fn in sorted(os.listdir(arm_dir)):
        m = _CHUNK_RE.match(fn)
        if m:
            found[int(m.group(1))] = os.path.join(arm_dir, fn)
    res.n_chunks = len(found)
    res.missing = sorted(set(range(EXPECTED_CHUNKS)) - set(found))
    res.extra = sorted(i for i in found if i >= EXPECTED_CHUNKS)
    # Structurally excluded chunks: the 24000-bar-window product/transmission
    # arms refuse rows inside the selection/frame block plus the training
    # window (first legal OOS row 48000 -> chunks 0-10 illegal BY DESIGN, per
    # src/unification.py's legality check). Those arms are complete at 89.
    legal_missing = _LEGAL_MISSING.get(res.arm, set())
    res.incomplete = bool(set(res.missing) - legal_missing)
    if not found:
        return res

    chunks = []
    idxs = sorted(found)
    # The causal maps chain window-to-window: strictly ascending order required.
    assert all(j > i for i, j in zip(idxs, idxs[1:])), "chunk order not ascending"
    prev_state: dict | None = None
    prev_idx: int | None = None
    for idx in idxs:
        try:
            raw = _load_chunk(found[idx])
        except Exception as err:  # a corrupt chunk must not sink the harvest
            res.warnings.append(f"chunk_{idx}: unreadable ({err}); skipped")
            res.incomplete = True
            if idx not in res.missing:
                res.missing.append(idx)
            prev_state, prev_idx = None, None  # calibration chain broken
            continue
        if len(raw["row_id"]) > 1 and not np.all(np.diff(raw["row_id"]) == 1):
            res.contiguous_in_chunk = False
            res.warnings.append(f"chunk_{idx}: row_id not contiguous within chunk")
        res.masked_col_events += raw["masked"]
        res.dropped_col_events += raw["dropped"]
        if prev_idx is None and not chunks:
            # CALIBRATION BURN-IN: the first present window estimates the
            # maps for its successor and is NOT scored — no loss rows,
            # excluded from pooled QLIKE, DM joins, MZ, and the sensitivity
            # conventions alike.
            prev_state, prev_idx = _burnin_state(raw), idx
            continue
        # predecessor must be the immediately preceding chunk index; a gap in
        # the harvest breaks the chain and the mid-panel fallbacks fire
        # (flagged; expected never on a complete harvest).
        prev = prev_state if prev_idx == idx - 1 else None
        c = _score_chunk_causal(raw, prev, res)
        prev_state, prev_idx = c.pop("state"), idx
        c["t"] = raw["t"]  # timestamps ride along (tree dev-prefix filter)
        chunks.append(c)
    if res.calib_fallback_windows or res.var_fallback_windows:
        res.warnings.append(
            "mid-panel degenerate-predecessor fallback fired (calib="
            f"{res.calib_fallback_windows}, var={res.var_fallback_windows}) "
            "— expected 0/0 on a complete harvest"
        )
    if not chunks:
        return res

    row_id = np.concatenate([c["row_id"] for c in chunks])
    order = np.argsort(row_id, kind="stable")
    row_id = row_id[order]
    dupe = np.zeros(len(row_id), dtype=bool)
    dupe[1:] = row_id[1:] == row_id[:-1]
    res.n_dupes = int(dupe.sum())
    if res.n_dupes:
        res.warnings.append(
            f"{res.n_dupes} overlapping row_id(s) across chunks; first occurrence kept"
        )
    keep = order[~dupe]
    res.row_id = row_id[~dupe]
    res.n_gaps = int((np.diff(res.row_id) > 1).sum()) if len(res.row_id) > 1 else 0

    for name in ("loss", "loss_none", "loss_duan", "err2", "f", "rv_raw", "yhat", "t"):
        arr = np.concatenate([c[name] for c in chunks])[keep]
        setattr(res, {"f": "f_raw"}.get(name, name), arr)
    valid = np.concatenate([c["valid"] for c in chunks])[keep]

    res.n_rows = len(res.row_id)
    res.n_valid = int(valid.sum())
    res.n_invalid = res.n_rows - res.n_valid
    res.n_qlike = int(np.isfinite(res.loss).sum())
    if res.n_qlike:
        res.qlike = float(np.nanmean(res.loss))
    if np.isfinite(res.loss_none).any():
        res.qlike_none = float(np.nanmean(res.loss_none))
    if np.isfinite(res.loss_duan).any():
        res.qlike_duan = float(np.nanmean(res.loss_duan))
    ffin = res.f_raw[np.isfinite(res.f_raw)]
    if ffin.size:
        res.f_min = float(ffin.min())
    if res._s2_n:
        res.sigma2_mean = res._s2_sum / res._s2_n
    if res._ab:  # fitted (non-fallback) windows only; fallbacks counted apart
        res.calib_a_mean = float(np.mean([ab[0] for ab in res._ab]))
        res.calib_b_mean = float(np.mean([ab[1] for ab in res._ab]))

    # Mincer-Zarnowitz: rv_raw on the raw forecast, positive finite pairs.
    m = np.isfinite(res.f_raw) & np.isfinite(res.rv_raw)
    m &= (res.f_raw > 0) & (res.rv_raw > 0)
    if m.sum() >= 3:
        mz = mz_regression(res.rv_raw[m], res.f_raw[m])
        res.mz_alpha, res.mz_beta = mz["alpha"], mz["beta"]
        res.mz_r2 = mz["r2"]
    return res


# ── tree expert bank -> synthetic causal-selection arms ───────────────────────
# (author directive 2026-08-07). tree_tuned: at each retune boundary (every
# TREE_TUNE_EVERY joined bars, TREE_EMBARGO-bar gap, TREE_TAIL-bar tail —
# identical 250/25/125 protocol as the linear tuners) the expert with the
# lowest tail mean per-bar loss fills the NEXT block from its PERSISTED
# forecasts. tree_hedge: exponential weights on the same trailing tail losses,
# eta = sqrt(8 ln K / L) with L = TREE_TAIL — the standard exponentially
# weighted average forecaster rate (Cesa-Bianchi & Lugosi 2006, "Prediction,
# Learning, and Games", Thm 2.2), no free constant. The first boundary before
# any tail exists is a BURN-IN SKIP (consistent with the calibration burn-in).
# ALL tree_* rows are restricted to t >= TREE_DEV_END (2003+): the menu was
# tuned on the pre-2003 dev prefix, so that span is excluded from every tree
# comparison — implemented as a timestamp row filter at assembly, which also
# restricts every downstream join (vs a0, increment pairs) to 2003+.

_TREE_EXPERTS = tuple(f"tree_expert_{k:02d}" for k in range(20))
TREE_TUNE_EVERY = 250
TREE_EMBARGO = 25
TREE_TAIL = 125
TREE_DEV_END = np.datetime64("2003-01-01")
_TREE_MENU_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "tree_menu.json"
)


def _tree_family_of() -> dict[str, str]:
    """Map tree_expert_<k> -> family tag from the frozen menu (mixed-family
    expansion 2026-08-07); '?' per expert when the menu is not deployed
    alongside the scorer (harvest still runs; the exhibit columns degrade)."""
    try:
        with open(_TREE_MENU_PATH, encoding="utf-8") as fh:
            menu = json.load(fh)
        return {
            f"tree_expert_{k:02d}": str(e.get("family", "?"))
            for k, e in enumerate(menu)
        }
    except (OSError, ValueError):
        return {}


def _qlike_bar(f: np.ndarray, rv: np.ndarray) -> np.ndarray:
    """Per-bar QLIKE on raw forecasts (the repo convention: r - ln r - 1)."""
    out = np.full(len(f), np.nan)
    ok = np.isfinite(f) & np.isfinite(rv) & (f > 0) & (rv > 0)
    r = rv[ok] / f[ok]
    out[ok] = r - np.log(r) - 1.0
    return out


def _assemble_tree_arms(
    by_root: dict[str, list[ArmResult]], sel_rows: list[dict]
) -> list[ArmResult]:
    """Build tree_tuned / tree_hedge per root from the harvested expert banks
    (see the block comment above for the protocol). Appends the selection
    trajectory to ``sel_rows`` for the CSV sidecar."""
    out: list[ArmResult] = []
    for root_label, entries in by_root.items():
        experts = sorted(
            (r for r in entries if r.arm in _TREE_EXPERTS and r.n_rows),
            key=lambda r: r.arm,
        )
        if len(experts) < 2:
            continue
        k_n = len(experts)
        common = experts[0].row_id
        for e in experts[1:]:
            common = np.intersect1d(common, e.row_id, assume_unique=True)
        if len(common) == 0:
            continue
        f_mat = np.empty((k_n, len(common)))
        rv = t_arr = None
        for k, e in enumerate(experts):
            _, _, ib = np.intersect1d(
                common, e.row_id, assume_unique=True, return_indices=True
            )
            f_mat[k] = e.f_raw[ib]
            if rv is None:
                rv, t_arr = e.rv_raw[ib], e.t[ib]
        dev = t_arr >= TREE_DEV_END  # dev-prefix exclusion (2003+ only)
        common, f_mat, rv, t_arr = common[dev], f_mat[:, dev], rv[dev], t_arr[dev]
        n = len(common)
        if n == 0:
            continue
        loss_mat = np.vstack([_qlike_bar(f_mat[k], rv) for k in range(k_n)])
        eta = float(np.sqrt(8.0 * np.log(k_n) / TREE_TAIL))
        family_of = _tree_family_of()
        sel_family = np.empty(n, dtype=object)
        sel_family[:] = ""
        f_sel = np.full(n, np.nan)
        f_hed = np.full(n, np.nan)
        for b in range(0, n, TREE_TUNE_EVERY):
            tail_hi = b - TREE_EMBARGO
            tail_lo = tail_hi - TREE_TAIL
            if tail_lo < 0:
                continue  # burn-in skip: no tail exists before the first boundary
            tail_loss = loss_mat[:, tail_lo:tail_hi]
            mean_tail = np.nanmean(tail_loss, axis=1)
            if not np.isfinite(mean_tail).any():
                continue
            sel = int(np.nanargmin(mean_tail))
            blk = slice(b, min(b + TREE_TUNE_EVERY, n))
            f_sel[blk] = f_mat[sel][blk]
            fam = family_of.get(experts[sel].arm, "?")
            sel_family[blk] = fam
            w = np.exp(-eta * np.nan_to_num(np.nansum(tail_loss, axis=1)))
            w = w / w.sum()
            f_hed[blk] = w @ f_mat[:, blk]
            sel_rows.append(
                {
                    "root": root_label,
                    "boundary_row": int(common[b]),
                    "boundary_year": int(str(t_arr[b].astype("datetime64[Y]"))),
                    "selected_expert": sel,
                    "selected_arm": experts[sel].arm,
                    "family": fam,
                    "eta": eta,
                }
            )
        for arm_name, f in (("tree_tuned", f_sel), ("tree_hedge", f_hed)):
            filled = np.isfinite(f)
            if not filled.any():
                continue
            res = ArmResult(root=root_label, arm=arm_name)
            res.n_chunks = min(e.n_chunks for e in experts)
            res.incomplete = any(e.incomplete for e in experts)
            res.row_id = common[filled]
            res.t = t_arr[filled]
            res.f_raw = np.asarray(f[filled])
            res.rv_raw = rv[filled]
            res.loss = _qlike_bar(res.f_raw, res.rv_raw)
            nf = int(filled.sum())
            res.loss_none = np.full(nf, np.nan)
            res.loss_duan = np.full(nf, np.nan)
            res.err2 = np.full(nf, np.nan)  # fit-space err undefined for composites
            res.yhat = np.full(nf, np.nan)
            res.n_rows = nf
            res.n_valid = nf
            res.n_qlike = int(np.isfinite(res.loss).sum())
            if res.n_qlike:
                res.qlike = float(np.nanmean(res.loss))
            ffin = res.f_raw[np.isfinite(res.f_raw)]
            if ffin.size:
                res.f_min = float(ffin.min())
            m = (res.f_raw > 0) & (res.rv_raw > 0)
            if m.sum() >= 3:
                mz = mz_regression(res.rv_raw[m], res.f_raw[m])
                res.mz_alpha, res.mz_beta = mz["alpha"], mz["beta"]
                res.mz_r2 = mz["r2"]
            res.warnings.append(
                f"synthetic: causal composite of {k_n} expert banks; rows "
                "restricted to t >= 2003-01-01 (dev-prefix exclusion)"
            )
            if arm_name == "tree_tuned":
                # revealed family preference (paper exhibit): fraction of
                # eval bars served by each family
                fams = sel_family[filled]
                tot = len(fams)
                frac = {
                    f: round(int((fams == f).sum()) / tot, 4)
                    for f in sorted({str(x) for x in fams if x})
                }
                res.warnings.append(f"family fractions of eval bars: {frac}")
            out.append(res)
    return out


def _compare_vs_a0(res: ArmResult, a0: ArmResult) -> None:
    """Same-environment comparison: join per-bar on row_id vs the root's own a0."""
    common, ia, ib = np.intersect1d(
        res.row_id, a0.row_id, assume_unique=True, return_indices=True
    )
    res.n_asym = (len(res.row_id) - len(common)) + (len(a0.row_id) - len(common))
    if res.n_asym:
        res.warnings.append(
            f"row sets vs {A0} not identical: {len(res.row_id) - len(common)} "
            f"arm-only + {len(a0.row_id) - len(common)} a0-only rows; "
            "compared on the intersection"
        )
    if len(common) == 0:
        return
    dm = dm_test(res.loss[ia], a0.loss[ib], h=1)  # repo DM utility, reused exactly
    if np.isfinite(dm.get("dm", float("nan"))):
        res.dm_t = float(dm["dm"])
    both = np.isfinite(res.err2[ia]) & np.isfinite(a0.err2[ib])
    sse_a0 = float(np.sum(a0.err2[ib][both]))
    if both.any() and sse_a0 > 0:
        res.oos_r2 = 1.0 - float(np.sum(res.err2[ia][both])) / sse_a0


# ── formatting ────────────────────────────────────────────────────────────────


def _fmt(x: float | None, spec: str) -> str:
    return "" if x is None else format(x, spec)


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


_PENDING_CELL = r"\pending{awaiting}"


def _pick_primary(entries: list[ArmResult]) -> ArmResult | None:
    """Best (root, arm) entry for macros/tables: complete first, then most bars."""
    if not entries:
        return None
    return sorted(entries, key=lambda r: (r.incomplete, -r.n_valid))[0]


def _macro_lines(
    by_arm: dict[str, list[ArmResult]],
    tau_none: float | None,
    tau_duan: float | None,
    incs: list[dict],
) -> list[str]:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    lines = [
        "% GENERATED FILE -- do not edit by hand.",
        f"% Written by experiments/score_unification.py at {stamp}.",
        "% Scoring: causal calibrated second-moment stack (smearing.tex) --",
        "% prev-window MZ mean + conditional-variance maps; rv_raw evaluation",
        "% target; DM vs the same root's a0 (src/evaluation/diebold_mariano).",
        r"% \unifAzeroRsq is a0's Mincer-Zarnowitz R^2 (rv_raw on raw forecast).",
        "% Incomplete arms render as red pending boxes until harvest completes.",
    ]

    def pend(arm: str) -> str:
        return r"\pending{%s awaiting harvest}" % _tex_escape(arm)

    def emit(name: str, value: str) -> None:
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")

    a0 = _pick_primary(by_arm.get(A0, []))
    a0_done = a0 is not None and not a0.incomplete
    emit(
        "unifAzeroQLIKE",
        _fmt(a0.qlike, ".5f") if a0_done and a0.qlike is not None else pend(A0),
    )
    emit(
        "unifAzeroRsq",
        _fmt(a0.mz_r2, ".3f") if a0_done and a0.mz_r2 is not None else pend(A0),
    )
    emit(
        "unifAzeroMZbeta",
        _fmt(a0.mz_beta, ".3f") if a0_done and a0.mz_beta is not None else pend(A0),
    )
    # Benchmark row under the two alternative smear conventions (appendix
    # master table); \unifAzeroQLIKE above stays the contract value.
    emit(
        "unifAzeroQLIKENone",
        _fmt(a0.qlike_none, ".5f")
        if a0_done and a0.qlike_none is not None
        else pend(A0),
    )
    emit(
        "unifAzeroQLIKEDuan",
        _fmt(a0.qlike_duan, ".5f")
        if a0_done and a0.qlike_duan is not None
        else pend(A0),
    )

    a0q = a0.qlike if a0_done else None
    for arm, camel in _REGISTRY:
        if arm == A0:
            continue
        r = _pick_primary(by_arm.get(arm, []))
        done = r is not None and not r.incomplete
        emit(
            f"unif{camel}QLIKE",
            _fmt(r.qlike, ".5f") if done and r.qlike is not None else pend(arm),
        )
        emit(
            f"unif{camel}DM",
            _fmt(r.dm_t, "+.1f") if done and r.dm_t is not None else pend(arm),
        )
        # Delta vs a0 (signed, 5 dec) — wired directly into section tables.
        delta_ok = done and r.qlike is not None and a0q is not None
        emit(
            f"unif{camel}Delta",
            _fmt(r.qlike - a0q, "+.5f") if delta_ok else pend(arm),
        )
    # Smear-sensitivity rank stability: Kendall tau of each alternative
    # convention's arm ranking against the contract's, over completed arms.
    pend_tau = r"\pending{smear sensitivity awaiting harvest}"
    emit(
        "unifSmearTauNone",
        _fmt(tau_none, ".3f") if tau_none is not None else pend_tau,
    )
    emit(
        "unifSmearTauDuan",
        _fmt(tau_duan, ".3f") if tau_duan is not None else pend_tau,
    )
    # Ladder-increment statistics (product / transmission marginal value):
    # paired per-bar DM between adjacent rungs, keyed by the upper rung;
    # pending until BOTH arms of the pair are complete.
    for rec in incs:
        camel = _CAMEL[rec["hi"]]
        pend_i = r"\pending{%s vs %s awaiting harvest}" % (
            _tex_escape(rec["hi"]),
            _tex_escape(rec["lo"]),
        )
        ok = rec["complete"]
        emit(
            f"unifIncr{camel}DM",
            _fmt(rec["dm_t"], "+.1f") if ok and rec["dm_t"] is not None else pend_i,
        )
        emit(
            f"unifIncr{camel}DQ",
            _fmt(rec["dqlike"], "+.5f") if ok and rec["dqlike"] is not None else pend_i,
        )
    return lines


def _buckets_table(by_arm: dict[str, list[ArmResult]]) -> list[str]:
    a0 = _pick_primary(by_arm.get(A0, []))
    a0q = a0.qlike if a0 is not None and not a0.incomplete else None

    def row(arm: str) -> str:
        name = _BUCKET_TEX[arm]
        r = _pick_primary(by_arm.get(arm, []))
        if r is None or r.incomplete or r.qlike is None:
            return (
                f"{name} & {_PENDING_CELL} & {_PENDING_CELL} & "
                f"{_PENDING_CELL} & {_PENDING_CELL} \\\\"
            )
        q = f"{r.qlike:.5f}"
        if arm == "a_bucket_all_features":
            q = rf"\textbf{{{q}}}"
        # a0 IS the OLS class's no-exogenous baseline, so Delta_own == Delta.
        d = rf"${r.qlike - a0q:+.5f}$" if a0q is not None else _PENDING_CELL
        t = rf"${r.dm_t:+.1f}$" if r.dm_t is not None else _PENDING_CELL
        return f"{name} & {q} & {d} & {d} & {t} \\\\"

    def key(arm: str) -> tuple:
        r = _pick_primary(by_arm.get(arm, []))
        if r is None or r.incomplete or r.qlike is None:
            return (1, _ORDER[arm])
        return (0, r.qlike)

    singles = sorted((a for a in _BUCKET_TEX if a != "a_bucket_all_features"), key=key)
    a0_cell = f"{a0q:.5f}" if a0q is not None else _PENDING_CELL
    return [
        r"\toprule",
        r"Bucket & QLIKE & $\Delta$ vs.\ incumbent & $\Delta_{\text{own}}$ & DM $t$ \\",
        r"\midrule",
        row("a_bucket_all_features"),
        r"\midrule",
        *[row(a) for a in singles],
        r"\midrule",
        rf"(no exogenous $=$ incumbent \texttt{{a0}}) & {a0_cell} & --- & --- & --- \\",
        r"\bottomrule",
    ]


def _lasso_ridge_table(by_arm: dict[str, list[ArmResult]]) -> list[str]:
    a0 = _pick_primary(by_arm.get(A0, []))
    a0q = a0.qlike if a0 is not None and not a0.incomplete else None

    def cell(r: ArmResult | None, bold: bool) -> str:
        if r is None or r.incomplete or r.qlike is None:
            return _PENDING_CELL
        q = f"{r.qlike:.5f}"
        if bold:
            q = rf"\textbf{{{q}}}"
        return f"{q} (${r.dm_t:+.1f}$)" if r.dm_t is not None else q

    def pair_row(label: str, ridge_arm: str, lasso_arm: str) -> str:
        ridge = _pick_primary(by_arm.get(ridge_arm, []))
        lasso = _pick_primary(by_arm.get(lasso_arm, []))
        rq = ridge.qlike if ridge is not None and not ridge.incomplete else None
        lq = lasso.qlike if lasso is not None and not lasso.incomplete else None
        both = rq is not None and lq is not None
        return (
            label
            + " & "
            + cell(ridge, both and rq <= lq)
            + " & "
            + cell(lasso, both and lq < rq)
            + r" \\"
        )

    a0_cell = f"{a0q:.5f}" if a0q is not None else _PENDING_CELL
    return [
        r"\toprule",
        r"Bucket & Ridge & Lasso \\",
        r"\midrule",
        pair_row(r"\texttt{all\_features} (fixed penalty)", "b1_ridge", "b2_lasso"),
        pair_row(
            r"\texttt{all\_features} (causally tuned)",
            "b1_ridge_tuned",
            "b2_lasso_tuned",
        ),
        r"\midrule",
        r"Incumbent: OLS on \texttt{baseline} & "
        rf"\multicolumn{{2}}{{c}}{{{a0_cell}}} \\",
        r"\bottomrule",
    ]


def _blocks_table(by_arm: dict[str, list[ArmResult]]) -> list[str]:
    def row(arm: str, window: int, alphas: str) -> str:
        w = f"${window:,}$".replace(",", "{,}")
        r = _pick_primary(by_arm.get(arm, []))
        if r is None or r.incomplete or r.qlike is None:
            q = t = _PENDING_CELL
        else:
            q = f"{r.qlike:.5f}"
            t = rf"${r.dm_t:+.1f}$" if r.dm_t is not None else _PENDING_CELL
        name = _ARM_TEX.get(arm, rf"\texttt{{{_tex_escape(arm)}}}")
        return rf"{name} & {w} & \texttt{{{alphas}}} & {q} & {t} \\"

    ladder = _BLOCKS_TABLE[:9]
    diags = _BLOCKS_TABLE[9:]
    return [
        r"\toprule",
        r"Arm & Window (bars) & $\alpha$ per block & QLIKE & DM $t$ vs.\ \texttt{a0} \\",
        r"\midrule",
        *[row(*x) for x in ladder],
        r"\midrule",
        *[row(*x) for x in diags],
        r"\bottomrule",
    ]


def _increments(by_arm: dict[str, list[ArmResult]]) -> list[dict]:
    """Paired per-bar DM between adjacent ladder rungs (upper vs lower), on
    the joined contract losses — the product/transmission increment tests.

    Same-environment rule: both rungs must come from the same root; the
    repo DM utility (dm_test) is reused on the per-bar loss differential.
    """
    out: list[dict] = []
    for hi_arm, lo_arm in _INCREMENT_PAIRS:
        hi = _pick_primary(by_arm.get(hi_arm, []))
        lo = _pick_primary(by_arm.get(lo_arm, []))
        rec: dict = {
            "hi": hi_arm,
            "lo": lo_arm,
            "root": "",
            "n_common": 0,
            "dqlike": None,
            "dm_t": None,
            "complete": False,
        }
        if hi is not None and lo is not None and hi.n_rows and lo.n_rows:
            if hi.root != lo.root:  # same-environment rule: never cross roots
                rec["root"] = f"MIXED({hi.root},{lo.root})"
            else:
                rec["root"] = hi.root
                _, ia, ib = np.intersect1d(
                    hi.row_id, lo.row_id, assume_unique=True, return_indices=True
                )
                rec["n_common"] = int(len(ia))
                if len(ia):
                    dm = dm_test(hi.loss[ia], lo.loss[ib], h=1)  # repo utility
                    if np.isfinite(dm.get("dm", float("nan"))):
                        rec["dm_t"] = float(dm["dm"])
                    if np.isfinite(dm.get("mean_diff", float("nan"))):
                        rec["dqlike"] = float(dm["mean_diff"])
                rec["complete"] = (not hi.incomplete) and (not lo.incomplete)
        out.append(rec)
    return out


def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float | None:
    """Kendall rank correlation (tau-a) via pairwise sign concordance.

    Numpy only (no scipy dependency); tied pairs contribute zero, matching
    tau-a. None when fewer than two observations.
    """
    n = len(x)
    if n < 2:
        return None
    i, j = np.triu_indices(n, 1)
    s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
    return float(np.sum(s) / len(s))


def _smear_sensitivity(
    by_arm: dict[str, list[ArmResult]],
) -> tuple[list[str], float | None, float | None]:
    """Three-convention sensitivity: table body (one row per COMPLETED arm in
    registry order; QLIKE under none / Duan / contract with within-convention
    ranks) + Kendall taus of each alternative ranking vs the contract's."""
    rows: list[tuple[str, float, float, float]] = []
    for arm, _ in _REGISTRY:
        r = _pick_primary(by_arm.get(arm, []))
        if r is None or r.incomplete:
            continue
        if r.qlike is None or r.qlike_none is None or r.qlike_duan is None:
            continue
        rows.append((arm, r.qlike_none, r.qlike_duan, r.qlike))
    # Self-contained tabular: \input-ing bare booktabs rows inside a tabular
    # breaks (trailing \par -> "Misplaced \noalign"), so the fragment carries
    # its own environment and the section \input's it directly.
    lines = [
        r"\begin{tabular}{lrrr}%",
        r"\toprule",
        r"Arm & QLIKE (none) & QLIKE (Duan) & QLIKE (contract) \\",
        r"\midrule",
    ]
    tau_none: float | None = None
    tau_duan: float | None = None
    if rows:
        qn = np.array([r[1] for r in rows])
        qd = np.array([r[2] for r in rows])
        qc = np.array([r[3] for r in rows])

        def _rank(v: np.ndarray) -> np.ndarray:
            rk = np.empty(len(v), dtype=np.int64)
            rk[np.argsort(v, kind="stable")] = np.arange(1, len(v) + 1)
            return rk

        rn, rd, rc = _rank(qn), _rank(qd), _rank(qc)
        for k, (arm, _n, _d, _c) in enumerate(rows):
            name = _ARM_TEX.get(arm, rf"\texttt{{{_tex_escape(arm)}}}")
            lines.append(
                rf"{name} & {_n:.5f} ({rn[k]}) & "
                rf"{_d:.5f} ({rd[k]}) & {_c:.5f} ({rc[k]}) \\"
            )
        tau_none = _kendall_tau(qc, qn)
        tau_duan = _kendall_tau(qc, qd)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    return lines, tau_none, tau_duan


# ── driver ────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="result roots, one per cluster (walked for <arm>/chunk_*.npz)",
    )
    ap.add_argument("--out", required=True, help="scores CSV path")
    ap.add_argument(
        "--tex-dir",
        default=os.path.join("writeup", "generated"),
        help="dir for campaign_numbers.tex + table bodies",
    )
    args = ap.parse_args(argv)

    # 1. DISCOVER + 2/3. VALIDATE + SCORE per (root, arm).
    labels: dict[str, str] = {}
    for root in args.roots:
        lab = os.path.basename(os.path.normpath(root))
        labels[root] = lab if lab not in labels.values() else root
    results: list[ArmResult] = []
    for root in args.roots:
        if not os.path.isdir(root):
            print(f"[{labels[root]}] root missing on disk: {root} (0 arms)")
            continue
        for arm in sorted(os.listdir(root)):
            arm_dir = os.path.join(root, arm)
            if not os.path.isdir(arm_dir):
                continue
            if not any(_CHUNK_RE.match(f) for f in os.listdir(arm_dir)):
                continue
            if arm not in _CAMEL:
                print(
                    f"[{labels[root]}] NOTE: unknown arm dir '{arm}' "
                    "(scored, CSV only, no macro/table slot)"
                )
            results.append(_harvest_arm(root, labels[root], arm))

    # 4. COMPARISONS: within each root, vs that root's own a0.
    by_root: dict[str, list[ArmResult]] = {}
    for r in results:
        by_root.setdefault(r.root, []).append(r)

    # 4a. TREE SYNTHETICS: assemble tree_tuned / tree_hedge causally from the
    # harvested expert banks (2003+ rows only), then let the ordinary
    # comparison/table machinery treat them like any other arm. Selection
    # trajectory lands in tree_selection.csv next to --out.
    tree_sel_rows: list[dict] = []
    for r in _assemble_tree_arms(by_root, tree_sel_rows):
        results.append(r)
        by_root.setdefault(r.root, []).append(r)
    if tree_sel_rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        sel_csv = os.path.join(
            os.path.dirname(os.path.abspath(args.out)), "tree_selection.csv"
        )
        with open(sel_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "root",
                    "boundary_row",
                    "boundary_year",
                    "selected_expert",
                    "selected_arm",
                    "family",
                    "eta",
                ]
            )
            for rec in tree_sel_rows:
                w.writerow(
                    [
                        rec["root"],
                        rec["boundary_row"],
                        rec["boundary_year"],
                        rec["selected_expert"],
                        rec["selected_arm"],
                        rec["family"],
                        f"{rec['eta']:.6g}",
                    ]
                )
        print(
            f"tree selection trajectory -> {sel_csv} ({len(tree_sel_rows)} boundaries)"
        )
        # per-era family preference (paper exhibit): boundary counts per
        # (root, year, family) + fraction within the year
        fam_csv = os.path.join(
            os.path.dirname(os.path.abspath(args.out)), "tree_family_summary.csv"
        )
        agg: dict[tuple[str, int], dict[str, int]] = {}
        for rec in tree_sel_rows:
            key = (rec["root"], rec["boundary_year"])
            agg.setdefault(key, {})
            agg[key][rec["family"]] = agg[key].get(rec["family"], 0) + 1
        with open(fam_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["root", "year", "family", "n_boundaries", "frac_of_year"])
            for (root_l, year), fams in sorted(agg.items()):
                tot = sum(fams.values())
                for fam, cnt in sorted(fams.items()):
                    w.writerow([root_l, year, fam, cnt, f"{cnt / tot:.4f}"])
        print(f"tree family preference by era -> {fam_csv}")

    for root_label, entries in by_root.items():
        a0 = next((r for r in entries if r.arm == A0), None)
        if a0 is None or a0.n_rows == 0:
            print(f"[{root_label}] no {A0} in this root -> DM/OOS-R2 unavailable here")
            continue
        for r in entries:
            if r.n_rows:
                _compare_vs_a0(r, a0)

    # (Cross-cluster canary removed by author directive 2026-08-06: Hoffman2
    # was abandoned and the campaign is single-cluster (CARC only), so there
    # is no second environment to float-parity against. Intentional — do not
    # restore without a second cluster in the campaign.)

    # 6a. CSV.
    results.sort(
        key=lambda r: (
            args.roots.index(next(k for k, v in labels.items() if v == r.root))
            if r.root in labels.values()
            else 99,
            _ORDER.get(r.arm, 98),
            r.arm,
        )
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "arm",
                "root",
                "n_bars",
                "n_chunks_present",
                "incomplete",
                "qlike",
                "qlike_none",
                "qlike_duan",
                "dm_t_vs_a0",
                "oos_r2_vs_a0",
                "mz_alpha",
                "mz_beta",
                "sigma2_mean",
                "masked_col_events",
                "calib_a_mean",
                "calib_b_mean",
                "calib_fallback_windows",
                "var_fallback_windows",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.arm,
                    r.root,
                    r.n_valid,
                    r.n_chunks,
                    r.incomplete,
                    _fmt(r.qlike, ".10g"),
                    _fmt(r.qlike_none, ".10g"),
                    _fmt(r.qlike_duan, ".10g"),
                    _fmt(r.dm_t, ".6g"),
                    _fmt(r.oos_r2, ".10g"),
                    _fmt(r.mz_alpha, ".10g"),
                    _fmt(r.mz_beta, ".10g"),
                    _fmt(r.sigma2_mean, ".10g"),
                    r.masked_col_events,
                    _fmt(r.calib_a_mean, ".6g"),
                    _fmt(r.calib_b_mean, ".6g"),
                    r.calib_fallback_windows,
                    r.var_fallback_windows,
                ]
            )

    # 6b/c. LaTeX macros + table bodies (pending-robust at any completion level).
    by_arm: dict[str, list[ArmResult]] = {}
    for r in results:
        by_arm.setdefault(r.arm, []).append(r)
    incs = _increments(by_arm)
    inc_csv = os.path.join(
        os.path.dirname(os.path.abspath(args.out)), "unification_increments.csv"
    )
    with open(inc_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "root", "n_common", "complete", "dqlike", "dm_t"])
        for rec in incs:
            w.writerow(
                [
                    f"{rec['hi']}-vs-{rec['lo']}",
                    rec["root"],
                    rec["n_common"],
                    rec["complete"],
                    _fmt(rec["dqlike"], ".10g"),
                    _fmt(rec["dm_t"], ".6g"),
                ]
            )
    os.makedirs(args.tex_dir, exist_ok=True)
    sens_lines, tau_none, tau_duan = _smear_sensitivity(by_arm)
    outputs = {
        "campaign_numbers.tex": _macro_lines(by_arm, tau_none, tau_duan, incs),
        "table_buckets.tex": _buckets_table(by_arm),
        "table_lasso_ridge.tex": _lasso_ridge_table(by_arm),
        "table_blocks.tex": _blocks_table(by_arm),
        "table_smear_sensitivity.tex": sens_lines,
    }
    for name, lines in outputs.items():
        with open(os.path.join(args.tex_dir, name), "w", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")

    # 7. Summary: one line per (root, arm); missing arms listed as pending.
    print(f"\n== unification harvest summary ({len(results)} (root, arm) pairs) ==")
    seen = {(r.root, r.arm) for r in results}
    for r in results:
        miss = ""
        if r.missing:
            head = ",".join(map(str, r.missing[:8]))
            miss = f" missing=[{head}{',...' if len(r.missing) > 8 else ''}]"
        extras = f" extra_chunks={r.extra}" if r.extra else ""
        flags = "".join(
            [
                "" if r.contiguous_in_chunk else " NONCONTIG",
                f" dupes={r.n_dupes}" if r.n_dupes else "",
                f" gaps={r.n_gaps}" if r.n_gaps and not r.missing else "",
                f" invalid={r.n_invalid}" if r.n_invalid else "",
                f" asym={r.n_asym}" if r.n_asym else "",
                f" fb={r.calib_fallback_windows}/{r.var_fallback_windows}"
                if r.n_chunks
                else "",
            ]
        )
        print(
            f"[{r.root}] {r.arm:28s} chunks {r.n_chunks:3d}/{EXPECTED_CHUNKS}"
            f" n_valid={r.n_valid:7d} qlike={_fmt(r.qlike, '.5f') or '------'}"
            f" dm={_fmt(r.dm_t, '+.1f') or '----'}"
            f" fmin={_fmt(r.f_min, '.2e') or '----'}"
            f"{' INCOMPLETE' if r.incomplete else ''}{miss}{extras}{flags}"
        )
        for wmsg in r.warnings:
            print(f"    ! {wmsg}")
    for arm, _ in _REGISTRY:
        roots_missing = [
            lab for lab in dict.fromkeys(labels.values()) if (lab, arm) not in seen
        ]
        if len(roots_missing) == len(set(labels.values())):
            print(f"[--] {arm:28s} no chunks in any root -> pending")
    for rec in incs:
        state = "" if rec["complete"] else " PENDING"
        print(
            f"[incr] {rec['hi']:12s} vs {rec['lo']:12s} n={rec['n_common']:7d}"
            f" dq={_fmt(rec['dqlike'], '+.5f') or '--------'}"
            f" dm={_fmt(rec['dm_t'], '+.1f') or '----'}{state}"
        )
    print(
        f"\nwrote {args.out} + {inc_csv} + "
        f"{', '.join(os.path.join(args.tex_dir, n) for n in outputs)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
