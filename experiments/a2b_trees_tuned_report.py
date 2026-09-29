"""A2b report: the causally TUNED per-bar trees against the per-bar linear arms and
the untuned trees at the 15:30-16:00 bar, plus the hyperparameter grid edges.

Checklist item A2b.  Reads the merged tuned arms (``specs/causal_tune_trees_tuned.py``
-> ``experiments/reduce_trees_tuned_chunks.py``; results_<bar>.csv = the
MSE-selected configuration, the arm of record; results_qsel_<bar>.csv = the
QLIKE-selected one), the untuned tree arms (``specs/causal_tune_trees.py``) and the
per-bar linear arms (ridge / lasso / elastic net; OLS on the HAR + calendar bucket).

ONE SCORER.  Every forecast is rebuilt with the research scorer's causal clock
back-transform (``score_linear_subsection_causal.causal_forecasts``: variance =
(f^2 + s) x B, s = mean squared adjusted-scale error of the arm's own previous 250
sessions at that bar, lagged one session) -- the 16:00-bar recalibration, not the
rv_iv notebook's 13-bar one.  The 15:30 sign(s) trade is the research scorer's
(``score_linear_subsection.trade_1530``: buy the straddle when the 16:00-bar
forecast exceeds the implied variance, sell otherwise; mid = q x R, crossed = pay the
ask to buy / hit the bid to sell), rebuilt here as a daily series so that paired
intervals can be drawn; its Sharpe is gated against ``trade_1530`` itself.

SAME ROWS.  Per bucket, every forecast is scored on the stamps where ALL of them
exist: "deck" = those stamps on the deck's trade days (the 866 sessions of
results/atm_straddle_0dte_1530/daily_blk2.parquet), "common" = all of them
(2018-2024, after the back-transform's warm-up).  The target (true_raw) must agree
across every forecast to 1e-9 relative on those stamps (gate).

EXTREME VALUES.  The plain back-transform f^2 x B (no second-moment term) is
reported beside the scored one: an adjusted-scale forecast f near zero squares to a
variance near zero, and QLIKE (y / f - log(y / f) - 1) explodes on that row (the
stored all-features ridge has f = 0.005 at 16:00 on one day).  The scored
back-transform adds the causal term s >= 0, which bounds the variance forecast
below by s x B, so a near-zero f cannot blow the loss.  The table lists, per
forecast, the smallest f, the rows with f <= 0 (f^2 would lose the sign), the
plain / as-scored / clock QLIKE and the largest single-row clock QLIKE; nothing is
clipped, floored or dropped.

INTERVALS.  Paired differences (row minus reference, same stamps): circular
day-block bootstrap (block 21 sessions, 2000 draws; ``score_linear_subsection``'s
constants) of the mean loss difference, and a Diebold-Mariano t; paired Sharpe
differences: the same bootstrap indices applied to both daily P&L series.

Tables (``--out``, default <root>/a2b):
  qlike_1600.csv            per bucket x forecast: n, QLIKE clock / as scored / plain,
                            row median and max, smallest f, rows f <= 0 -- deck and common
  qlike_1600_paired.csv     tuned vs ridge / lasso / untuned twin; qsel vs mse; untuned
                            vs ridge / lasso; lasso / enet vs ridge
  trade_1600.csv            15:30 sign(s) trade per bucket x forecast (deck days): % buy,
                            Sharpe mid / crossed, sign agreement with the ridge
  trade_1600_paired.csv     paired Sharpe differences (mid, crossed) with intervals
  grid_edges_model_axis.csv per (model, rule, axis) over every bucket, bar and tuning
                            date: share of picks at the low / high grid edge, whether that
                            edge is a natural parameter bound, the chance rate (share of
                            the 32 fixed candidates carrying that value), widened flag and
                            still_at_edge = picks at a non-natural edge above chance
  grid_edges_bucket_axis.csv  the same per bucket
  grid_edges_by_date.csv    per (model, rule, axis, tuning index): share of arms (bucket x
                            bar) whose pick sits at each edge, and the median tuning date
  grid_edges_long.csv       one row per (arm, tuning point, rule, axis): the chosen value
                            and its edge
  tuned_vs_untuned_bars.csv per (model, bucket, rule) over the 13 bars: mean / median %
                            QLIKE difference (clock, common stamps) and bars whose interval
                            excludes zero, from the scorer's trees_tuned_vs_untuned.csv
  vs_linear_bars.csv        the same for tuned (both rules) and untuned vs ridge / lasso
                            (trees_vs_linear.csv, trees_qsel_vs_linear.csv of both scorers)
  qsel_vs_mse.csv           per (model, bucket): tuning points where the two rules pick the
                            same configuration, bars with identical paths, mean % difference

Usage:  python experiments/a2b_trees_tuned_report.py
            [--root results/linear_subsection_trees_tuned]
            [--scored results/linear_subsection_trees_tuned/rescore_local]
            [--untuned-root results/linear_subsection_trees]
            [--untuned-scored results/linear_subsection_trees/rescore_local]
            [--linear-root DIR ...] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402
import build_subsection_tree_yhat as bst  # noqa: E402
import score_linear_subsection as base  # noqa: E402
import score_linear_subsection_causal as slc  # noqa: E402
import score_trees_subsection as sts  # noqa: E402
from src.evaluation.diebold_mariano import dm_test  # noqa: E402

SEG, HHMM = "bar1600", "16:00"  # the 15:30-16:00 bar; the forecast is issued at 15:30
TW = 2000
MODELS = ("lgbm", "xgb", "rf")
BUCKETS = ("all_features", "live_feasible", "baseline")
LIN_LABEL = {
    "ridge": "ridge",
    "reclasso": "lasso",
    "reclasticnet": "enet",
    "ols": "ols",
}
REFS = ("ridge", "lasso")  # the linear references of the paired tables
GATE_REL = sts.GATE_REL  # 1e-9: same design, same target
# edges the 2026-09-29 widening moved (specs/causal_tune_trees_tuned.py: FRACTION_STEPS
# gained 0.25, L2_ROWS gained 1000, RF_MAX_FEATURES gained 1/27)
WIDENED = {
    ("lgbm", "feature_fraction"): "lo",
    ("lgbm", "lambda_l2"): "hi",
    ("xgb", "subsample"): "lo",
    ("xgb", "colsample_bytree"): "lo",
    ("xgb", "reg_lambda"): "hi",
    ("rf", "max_features"): "lo",
}
ROUNDS_CAP = "rounds_cap"  # pseudo-axis of the boosted models: the round cap bound
UNBOUNDED = "None"  # the spec's spelling of an unbounded RF max_depth


# ---------------------------------------------------------------- helpers
def num(v) -> float:
    return np.inf if str(v).strip() == UNBOUNDED else float(v)


def ci_dm(d: pd.Series, la: pd.Series, lb: pd.Series) -> tuple[float, float, float]:
    lo, hi = base.day_block_ci(d)
    return lo, hi, float(dm_test(la.to_numpy(), lb.to_numpy())["dm"])


def sharpe(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x.mean(axis=axis) / x.std(axis=axis, ddof=1) * base.ANN


def boot_idx(n: int) -> np.ndarray:
    rng = np.random.default_rng([base.BOOT_SEED, n])
    return asl.circular_block_bootstrap_idx(rng, n, base.BOOT_BLOCK, base.BOOT_B)


def trade_series(pred: pd.Series, deck: pd.DataFrame) -> pd.DataFrame:
    """base.trade_1530's daily legs: q (+1 buy / -1 sell), mid and crossed returns."""
    f = pred[pred.index.strftime("%H:%M") == HHMM].copy()
    f.index = f.index.normalize()
    f = f.reindex(deck.index)
    ok = f.notna().to_numpy()
    dk = deck[ok]
    q = np.where(f.to_numpy(float)[ok] > dk["iv_var"].to_numpy(float), 1.0, -1.0)
    ask = (dk["ask_c"] + dk["ask_p"]).to_numpy(float)
    bid = (dk["bid_c"] + dk["bid_p"]).to_numpy(float)
    ex, r = dk["exit"].to_numpy(float), dk["R"].to_numpy(float)
    return pd.DataFrame(
        {
            "q": q,
            "mid": q * r,
            "crossed": q * np.where(q > 0, ex / ask - 1.0, ex / bid - 1.0),
        },
        index=dk.index,
    )


# ---------------------------------------------------------------- 16:00 forecasts
def forecasts_1600(
    bucket: str, root: Path, untuned_root: Path, lin_roots: list[Path]
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for m in MODELS:
        p = bst.tree_arm_path(root, bucket, SEG, m, TW)
        if p is None:
            print(f"note: no tuned {m} arm for {bucket}/{SEG}")
        else:
            r = sts.load_forecasts(p)
            if r is not None:
                out[f"tuned {m}"] = r
            q = p.with_name(f"results_qsel_{SEG}.csv")
            rq = sts.load_forecasts(q) if q.is_file() else None
            if rq is not None:
                out[f"tuned {m} qsel"] = rq
        u = bst.tree_arm_path(untuned_root, bucket, SEG, m, TW)
        ru = sts.load_forecasts(u) if u is not None else None
        if ru is not None:
            out[f"untuned {m}"] = ru
    for est, (path, _src) in sorted(
        sts.linear_paths(lin_roots, bucket, SEG, TW).items()
    ):
        r = sts.load_forecasts(path)
        if r is not None:
            out[LIN_LABEL.get(est, est)] = r
    return out


def qlike_tables(
    bucket: str,
    fc: dict[str, pd.DataFrame],
    deck_days: pd.DatetimeIndex,
    gates: list[dict],
) -> tuple[list[dict], list[dict], pd.DatetimeIndex]:
    stamps = None
    for r in fc.values():
        idx = r.dropna(subset=["pred_clock"]).index
        idx = idx[idx.strftime("%H:%M") == HHMM]
        stamps = idx if stamps is None else stamps.intersection(idx)
    assert stamps is not None
    ref_y = next(iter(fc.values()))["true_raw"].reindex(stamps)
    for lab, r in fc.items():
        g = sts.rel_gap(r["true_raw"].reindex(stamps), ref_y)
        gates.append(
            {
                "bucket": bucket,
                "forecast": lab,
                "check": "true_raw vs first forecast",
                "n": len(stamps),
                "max_rel": g,
                "ok": bool(g < GATE_REL),
            }
        )
        if not g < GATE_REL:
            print(f"GATE FAIL {bucket} {lab}: true_raw differs by {g:.2e}")
    samples = {"deck": stamps[stamps.normalize().isin(deck_days)], "common": stamps}
    rows, paired = [], []
    loss: dict[tuple[str, str], pd.Series] = {}
    for smp, st in samples.items():
        for lab, r in fc.items():
            s = r.reindex(st)
            y = s["true_raw"]
            qc = slc.qlike(y, s["pred_clock"])
            loss[(smp, lab)] = qc
            plain = s["pred_adj"] ** 2 * s["baseline"]
            rows.append(
                {
                    "bucket": bucket,
                    "sample": smp,
                    "forecast": lab,
                    "n": len(s),
                    "first": str(st.min().date()),
                    "last": str(st.max().date()),
                    "QLIKE_clock": float(qc.mean()),
                    "QLIKE_as_scored": float(slc.qlike(y, s["pred_raw"]).mean()),
                    "QLIKE_plain_f2B": float(slc.qlike(y, plain).mean()),
                    "row_QLIKE_clock_median": float(qc.median()),
                    "row_QLIKE_clock_max": float(qc.max()),
                    "row_QLIKE_plain_max": float(slc.qlike(y, plain).max()),
                    "min_pred_adj": float(s["pred_adj"].min()),
                    "rows_pred_adj_le_0": int((s["pred_adj"] <= 0).sum()),
                    "mse_adj": float(s["e2"].mean()),
                }
            )
        pairs = []
        for m in MODELS:
            for rule in (f"tuned {m}", f"tuned {m} qsel", f"untuned {m}"):
                pairs += [(rule, ref) for ref in REFS]
            pairs += [
                (f"tuned {m}", f"untuned {m}"),
                (f"tuned {m} qsel", f"untuned {m}"),
                (f"tuned {m} qsel", f"tuned {m}"),
            ]
        pairs += [("lasso", "ridge"), ("enet", "ridge"), ("ols", "ridge")]
        for a, b in pairs:
            if (smp, a) not in loss or (smp, b) not in loss:
                continue
            la, lb = loss[(smp, a)], loss[(smp, b)]
            d = la - lb
            lo, hi, dm = ci_dm(d, la, lb)
            paired.append(
                {
                    "bucket": bucket,
                    "sample": smp,
                    "forecast": a,
                    "reference": b,
                    "n": len(d),
                    "QLIKE_forecast": float(la.mean()),
                    "QLIKE_reference": float(lb.mean()),
                    "diff": float(d.mean()),
                    "pct": float(100 * d.mean() / lb.mean()),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "pct_ci_lo": float(100 * lo / lb.mean()),
                    "pct_ci_hi": float(100 * hi / lb.mean()),
                    "dm_stat": dm,
                    "better": bool(hi < 0),
                    "worse": bool(lo > 0),
                }
            )
    return rows, paired, samples["deck"]


def trade_tables(
    bucket: str,
    fc: dict[str, pd.DataFrame],
    deck: pd.DataFrame,
    stamps: pd.DatetimeIndex,
    gates: list[dict],
) -> tuple[list[dict], list[dict]]:
    series = {
        lab: trade_series(r["pred_clock"].reindex(stamps), deck)
        for lab, r in fc.items()
    }
    days = None
    for s in series.values():
        days = s.index if days is None else days.intersection(s.index)
    assert days is not None
    rows, paired = [], []
    ridge = series.get("ridge")
    for lab, s in series.items():
        s = s.reindex(days)
        ref = base.trade_1530(fc[lab]["pred_clock"].reindex(stamps))
        g = max(
            abs(ref["Sharpe_mid"] - float(sharpe(s["mid"].to_numpy()))),
            abs(ref["Sharpe_crossed"] - float(sharpe(s["crossed"].to_numpy()))),
        )
        gates.append(
            {
                "bucket": bucket,
                "forecast": lab,
                "check": "trade series Sharpe vs base.trade_1530",
                "n": len(s),
                "max_rel": g,
                "ok": bool(g < 1e-12 and ref["deck_days"] == len(s)),
            }
        )
        rows.append(
            {
                "bucket": bucket,
                "forecast": lab,
                "days": len(s),
                "pct_buy": float(100 * (s["q"] > 0).mean()),
                "Sharpe_mid": float(sharpe(s["mid"].to_numpy())),
                "Sharpe_crossed": float(sharpe(s["crossed"].to_numpy())),
                "mean_mid": float(s["mid"].mean()),
                "sign_agrees_with_ridge": float(
                    (s["q"] == ridge.reindex(days)["q"]).mean()
                )
                if ridge is not None
                else np.nan,
            }
        )
    idx = boot_idx(len(days))
    pairs = []
    for m in MODELS:
        for rule in (f"tuned {m}", f"tuned {m} qsel", f"untuned {m}"):
            pairs += [(rule, ref) for ref in REFS]
        pairs += [(f"tuned {m}", f"untuned {m}"), (f"tuned {m} qsel", f"tuned {m}")]
    pairs += [("lasso", "ridge"), ("enet", "ridge"), ("ols", "ridge")]
    for a, b in pairs:
        if a not in series or b not in series:
            continue
        sa, sb = series[a].reindex(days), series[b].reindex(days)
        row = {
            "bucket": bucket,
            "forecast": a,
            "reference": b,
            "days": len(days),
            "sign_agreement": float((sa["q"] == sb["q"]).mean()),
        }
        for leg in ("mid", "crossed"):
            xa, xb = sa[leg].to_numpy(float), sb[leg].to_numpy(float)
            dist = sharpe(xa[idx], axis=1) - sharpe(xb[idx], axis=1)
            lo, hi = np.percentile(dist, [2.5, 97.5])
            row |= {
                f"Sharpe_{leg}_forecast": float(sharpe(xa)),
                f"Sharpe_{leg}_reference": float(sharpe(xb)),
                f"diff_{leg}": float(sharpe(xa) - sharpe(xb)),
                f"ci_lo_{leg}": float(lo),
                f"ci_hi_{leg}": float(hi),
            }
        paired.append(row)
    return rows, paired


# ---------------------------------------------------------------- grid edges
def restore_unbounded(
    arm: bst.TreeArm, tr: pd.DataFrame, gates: list[dict]
) -> pd.DataFrame:
    """Put the spec's UNBOUNDED spelling back into a merged RF trace.

    The chunk reducer reads each chunk's tune_trace with pandas' default NA values,
    which include the string "None" -- the spec's spelling of an unbounded RF
    max_depth -- and writes it back as an empty cell.  The chunk-level traces keep
    "None"; they are concatenated in chunk order and compared row by row (every
    empty merged cell must be "None" there, every other cell the same number), and
    only then is the empty cell restored."""
    ax = "max_depth"
    if arm.model != "rf" or ax not in tr:
        return tr
    empty = tr[ax].isna() | (tr[ax].astype(str).str.strip() == "")
    if not empty.any():
        return tr
    top = arm.csv.parents[3]  # <bucket>/<seg>/<model>/tw<TW>
    chunk_files = sorted(
        top.glob(
            f"chunks/c*/{bst.SPEC_DIR}/{arm.model}/{arm.bucket}/tune_trace_{arm.seg}.csv"
        ),
        key=lambda p: int(p.parts[-5][1:]),
    )
    keys = {"bucket": arm.bucket, "forecast": arm.model, "segment": arm.seg}
    detail = ""
    if not chunk_files:
        detail = "no chunk-level traces to restore from"
    else:
        ch = pd.concat(
            [
                pd.read_csv(p, keep_default_na=False, na_values=[""])
                for p in chunk_files
            ],
            ignore_index=True,
        )
        if (
            len(ch) != len(tr)
            or not (ch["rule"].to_numpy() == tr["rule"].to_numpy()).all()
        ):
            detail = f"chunk traces have {len(ch)} rows, merged {len(tr)}"
        else:
            cv = ch[ax].astype(str).str.strip()
            bad_empty = int((empty & (cv != UNBOUNDED)).sum())
            mv = pd.to_numeric(tr[ax], errors="coerce")
            bad_num = int(
                (
                    ~empty
                    & ~np.isclose(
                        mv, pd.to_numeric(cv, errors="coerce"), rtol=GATE_REL, atol=0
                    )
                ).sum()
            )
            if bad_empty or bad_num:
                detail = f"{bad_empty} empty cells not 'None' in the chunks, {bad_num} numbers differ"
    ok = not detail
    gates.append(
        keys
        | {
            "check": "RF max_depth 'None' restored from chunk traces",
            "n": int(empty.sum()),
            "max_rel": 0.0,
            "ok": ok,
            "detail": detail,
        }
    )
    if not ok:
        print(f"GATE FAIL {keys}: {detail}")
        return tr
    tr = tr.copy()
    tr[ax] = tr[ax].astype(object).where(~empty, UNBOUNDED)
    return tr


def edge_long(root: Path, gates: list[dict]) -> pd.DataFrame:
    parts = []
    for arm in bst.discover_tree_arms(root, TW):
        tp = arm.csv.with_name(f"tune_trace_{arm.seg}.csv")
        gp = arm.csv.with_name(f"grid_{arm.seg}.json")
        if not tp.is_file() or not gp.is_file():
            print(f"note: no trace / grid for {arm.key}")
            continue
        tr = restore_unbounded(
            arm, pd.read_csv(tp, keep_default_na=False, na_values=[""]), gates
        )
        grid = json.loads(gp.read_text(encoding="utf-8"))
        axes, nat, cands = grid["axes"], grid["natural_bounds"], grid["candidates"]
        for ax, vals in axes.items():
            v = [num(x) for x in vals]
            cand_v = np.array([num("None" if c[ax] is None else c[ax]) for c in cands])
            chance_lo = float(np.isclose(cand_v, v[0], rtol=GATE_REL, atol=0).mean())
            chance_hi = float(np.isclose(cand_v, v[-1], rtol=GATE_REL, atol=0).mean())
            chosen = tr[ax].map(num).to_numpy(float)
            edge = np.where(
                np.isclose(chosen, v[0], rtol=GATE_REL, atol=0),
                "lo",
                np.where(
                    np.isclose(chosen, v[-1], rtol=GATE_REL, atol=0)
                    | (np.isinf(chosen) & np.isinf(v[-1])),
                    "hi",
                    "",
                ),
            )
            got = tr[f"{ax}_edge"].fillna("").astype(str).str.strip().to_numpy()
            assert (got == edge).all(), (
                f"{arm.key} {ax}: edge labels disagree with the grid file"
            )
            parts.append(
                pd.DataFrame(
                    {
                        "model": arm.model,
                        "bucket": arm.bucket,
                        "segment": arm.seg,
                        "tune_idx": tr["tune_idx"].to_numpy(),
                        "forecast_date": tr["forecast_date"].to_numpy(),
                        "rule": tr["rule"].to_numpy(),
                        "axis": ax,
                        "value": tr[ax].astype(str).to_numpy(),
                        "grid": " ".join(str(x) for x in vals),
                        "edge": edge,
                        "lo_natural": bool(nat[ax]["lo"]),
                        "hi_natural": bool(nat[ax]["hi"]),
                        "chance_lo": chance_lo,
                        "chance_hi": chance_hi,
                    }
                )
            )
        if arm.model in ("lgbm", "xgb"):
            hit = (
                tr["rounds_cap_hit"]
                .astype(str)
                .str.lower()
                .isin(("true", "1"))
                .to_numpy()
            )
            parts.append(
                pd.DataFrame(
                    {
                        "model": arm.model,
                        "bucket": arm.bucket,
                        "segment": arm.seg,
                        "tune_idx": tr["tune_idx"].to_numpy(),
                        "forecast_date": tr["forecast_date"].to_numpy(),
                        "rule": tr["rule"].to_numpy(),
                        "axis": ROUNDS_CAP,
                        "value": tr["rounds"].astype(str).to_numpy(),
                        "grid": f"early-stopped, cap {grid.get('rounds_max')}",
                        "edge": np.where(hit, "hi", ""),
                        "lo_natural": True,
                        "hi_natural": False,
                        "chance_lo": np.nan,
                        "chance_hi": np.nan,
                    }
                )
            )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def edge_summary(long: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in long.groupby(by, sort=True):
        k = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        lo_nat, hi_nat = bool(g["lo_natural"].iloc[0]), bool(g["hi_natural"].iloc[0])
        s_lo, s_hi = (
            float((g["edge"] == "lo").mean()),
            float((g["edge"] == "hi").mean()),
        )
        c_lo, c_hi = float(g["chance_lo"].mean()), float(g["chance_hi"].mean())
        wid = WIDENED.get((str(k.get("model")), str(k.get("axis"))), "")
        # a non-natural edge picked more often than the candidates carry it: the grid
        # still binds there (the optimum may lie beyond it)
        still = []
        if not lo_nat and s_lo > 0 and not (s_lo <= c_lo):
            still.append("lo")
        if not hi_nat and s_hi > 0 and not (s_hi <= c_hi):
            still.append("hi")
        rows.append(
            k
            | {
                "n_picks": len(g),
                "arms": g[["bucket", "segment"]].drop_duplicates().shape[0],
                "grid": g["grid"].iloc[0],
                "share_lo": s_lo,
                "lo_natural": lo_nat,
                "chance_lo": c_lo,
                "share_hi": s_hi,
                "hi_natural": hi_nat,
                "chance_hi": c_hi,
                "lo_over_chance": s_lo / c_lo if c_lo > 0 else np.nan,
                "hi_over_chance": s_hi / c_hi if c_hi > 0 else np.nan,
                "share_at_non_natural_edge": float(
                    (
                        ((g["edge"] == "lo") & ~g["lo_natural"])
                        | ((g["edge"] == "hi") & ~g["hi_natural"])
                    ).mean()
                ),
                "widened_edge": wid,
                "still_at_edge": "/".join(still),
                "widened_edge_still_binding": bool(wid and wid in still),
            }
        )
    return pd.DataFrame(rows)


def by_date(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (m, rule, ax, ti), g in long.groupby(
        ["model", "rule", "axis", "tune_idx"], sort=True
    ):
        d = pd.to_datetime(g["forecast_date"])
        rows.append(
            {
                "model": m,
                "rule": rule,
                "axis": ax,
                "tune_idx": ti,
                "median_date": str(d.median().date()),
                "first_date": str(d.min().date()),
                "last_date": str(d.max().date()),
                "arms": len(g),
                "share_lo": float((g["edge"] == "lo").mean()),
                "share_hi": float((g["edge"] == "hi").mean()),
                "lo_natural": bool(g["lo_natural"].iloc[0]),
                "hi_natural": bool(g["hi_natural"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- native importance (for the feature-importance work)
def importance_summary(imp_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (bucket, model, bar, feature): the mean over refits of each refit's importance
    SHARE (importance / its refit's total: LightGBM gain and XGBoost total_gain are on
    arbitrary scales, RF impurity already sums to one), the share of refits using the
    feature, and its rank; plus the same summed by feature family (score_trees_subsection's
    rules)."""
    feats, fams = [], []
    for p in sorted(imp_dir.glob("importance_refits_*_*.parquet")):
        tab = pd.read_parquet(p)
        bucket, model = p.stem.removeprefix("importance_refits_").rsplit("_", 1)
        names = [
            c
            for c in tab.columns
            if c not in ("segment", "refit_idx", "refit_row", "date")
        ]
        fam = pd.DataFrame(
            [sts.feature_family(n) for n in names],
            columns=["family", "source"],
            index=names,
        )
        for seg, g in tab.groupby("segment", sort=True):
            x = g[names].to_numpy(np.float64)
            tot = x.sum(axis=1, keepdims=True)
            share = np.divide(x, tot, out=np.zeros_like(x), where=tot > 0).mean(axis=0)
            t = pd.DataFrame(
                {
                    "bucket": bucket,
                    "model": model,
                    "segment": seg,
                    "feature": names,
                    "family": fam["family"].to_numpy(),
                    "mean_share": share,
                    "used_in_refits": (x > 0).mean(axis=0),
                    "refits": len(g),
                }
            )
            t["rank"] = t["mean_share"].rank(ascending=False, method="min").astype(int)
            feats.append(t)
            f = (
                t.groupby("family")
                .agg(n_features=("feature", "size"), share=("mean_share", "sum"))
                .reset_index()
            )
            fams.append(
                f.assign(bucket=bucket, model=model, segment=seg, refits=len(g))
            )
    if not feats:
        return pd.DataFrame(), pd.DataFrame()
    fam_tab = pd.concat(fams, ignore_index=True)[
        ["bucket", "model", "segment", "family", "n_features", "share", "refits"]
    ]
    return pd.concat(feats, ignore_index=True), fam_tab


# ---------------------------------------------------------------- all-bar summaries
def bar_summary(
    tab: pd.DataFrame, by: list[str], ref_col: str | None = None
) -> pd.DataFrame:
    t = tab[(tab["back_transform"] == "clock") & (tab["sample"] == "common")]
    if ref_col is not None:
        t = t[t["estimator"].isin(["ridge", "reclasso"])]
    blown = t["blown"].astype(bool)
    g = t[~blown].groupby(by)
    out = g.agg(
        bars=("segment", "nunique"),
        mean_pct=("pct", "mean"),
        median_pct=("pct", "median"),
        better=("improves", "sum"),
        worse=("worse", "sum"),
    )
    out["blown_rows_left_out"] = (
        t[blown].groupby(by).size().reindex(out.index).fillna(0).astype(int)
    )
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/linear_subsection_trees_tuned")
    ap.add_argument(
        "--scored",
        default=None,
        help="the local scorer's out dir (default <root>/rescore_local)",
    )
    ap.add_argument("--untuned-root", default="results/linear_subsection_trees")
    ap.add_argument(
        "--untuned-scored", default="results/linear_subsection_trees/rescore_local"
    )
    ap.add_argument("--linear-root", action="append", default=None)
    ap.add_argument("--out", default=None, help="default <root>/a2b")
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 500)
    root = bst.resolve(a.root)
    scored = bst.resolve(a.scored) if a.scored else root / "rescore_local"
    untuned_root, untuned_scored = (
        bst.resolve(a.untuned_root),
        bst.resolve(a.untuned_scored),
    )
    lin_roots = sts.split_roots(a.linear_root)
    out = bst.resolve(a.out) if a.out else root / "a2b"
    out.mkdir(parents=True, exist_ok=True)
    deck = pd.read_parquet(base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).normalize()
    deck_days = pd.DatetimeIndex(deck.index)
    print(
        f"tuned root {root}\nuntuned root {untuned_root}\nlinear roots {[str(r) for r in lin_roots]}\ndeck days {len(deck_days)}"
    )

    gates: list[dict] = []
    q_rows, q_pair, t_rows, t_pair = [], [], [], []
    for bucket in BUCKETS:
        fc = forecasts_1600(bucket, root, untuned_root, lin_roots)
        print(f"\n{bucket}: {len(fc)} forecasts at {HHMM}: {', '.join(fc)}")
        if len(fc) < 2:
            continue
        r, p, deck_stamps = qlike_tables(bucket, fc, deck_days, gates)
        q_rows += r
        q_pair += p
        tr, tp = trade_tables(bucket, fc, deck, deck_stamps, gates)
        t_rows += tr
        t_pair += tp
    tabs = {
        "qlike_1600.csv": pd.DataFrame(q_rows),
        "qlike_1600_paired.csv": pd.DataFrame(q_pair),
        "trade_1600.csv": pd.DataFrame(t_rows),
        "trade_1600_paired.csv": pd.DataFrame(t_pair),
    }

    long = edge_long(root, gates)
    if not long.empty:
        tabs["grid_edges_long.csv"] = long
        tabs["grid_edges_model_axis.csv"] = edge_summary(
            long, ["model", "rule", "axis"]
        )
        tabs["grid_edges_bucket_axis.csv"] = edge_summary(
            long, ["model", "bucket", "rule", "axis"]
        )
        tabs["grid_edges_by_date.csv"] = by_date(long)
        same = (
            long[long["axis"] != ROUNDS_CAP]
            .pivot_table(
                index=["model", "bucket", "segment", "tune_idx", "axis"],
                columns="rule",
                values="value",
                aggfunc="first",
            )
            .reset_index()
        )
        same["equal"] = same["mse"] == same["qlike"]
        cfg_same = (
            same.groupby(["model", "bucket", "segment", "tune_idx"])["equal"]
            .all()
            .reset_index()
        )
        qs = cfg_same.groupby(["model", "bucket"]).agg(
            tuning_points=("equal", "size"), same_config_share=("equal", "mean")
        )
        qvm = scored / "trees_qsel_vs_mse.csv"
        if qvm.is_file():
            v = pd.read_csv(qvm)
            v = v[(v["back_transform"] == "clock") & (v["sample"] == "common")]
            qs = qs.join(
                v.groupby(["model", "bucket"]).agg(
                    bars=("segment", "nunique"),
                    bars_identical_path=("diff", lambda s: int((s == 0).sum())),
                    mean_pct_qsel_vs_mse=("pct", "mean"),
                    qsel_better=("improves", "sum"),
                    qsel_worse=("worse", "sum"),
                )
            )
        tabs["qsel_vs_mse.csv"] = qs.reset_index()

    vu = scored / "trees_tuned_vs_untuned.csv"
    if vu.is_file():
        tabs["tuned_vs_untuned_bars.csv"] = bar_summary(
            pd.read_csv(vu), ["model", "bucket", "rule"]
        )
    vl_parts = []
    for path, tag in (
        (scored / "trees_vs_linear.csv", "tuned mse"),
        (scored / "trees_qsel_vs_linear.csv", "tuned qsel"),
        (untuned_scored / "trees_vs_linear.csv", "untuned"),
    ):
        if path.is_file():
            t = pd.read_csv(path)
            t["model"] = t["model"].str.removesuffix("_qsel")
            s = bar_summary(t, ["model", "bucket", "estimator"], ref_col="estimator")
            s.insert(0, "forecast", tag)
            vl_parts.append(s)
        else:
            print(f"note: {path} missing")
    if vl_parts:
        tabs["vs_linear_bars.csv"] = pd.concat(vl_parts, ignore_index=True)
    imp_dir = root / "importance"
    if imp_dir.is_dir():
        feat, fam = importance_summary(imp_dir)
        if not feat.empty:
            feat.to_parquet(imp_dir / "importance_share_by_arm.parquet", index=False)
            fam.to_csv(imp_dir / "importance_family_share_by_arm.csv", index=False)
            print(
                f"wrote {imp_dir / 'importance_share_by_arm.parquet'} ({len(feat)} rows) and importance_family_share_by_arm.csv ({len(fam)} rows)"
            )
            f16 = fam[fam["segment"] == SEG].sort_values(
                ["bucket", "model", "share"], ascending=[True, True, False]
            )
            tabs["importance_family_1600.csv"] = f16
    tabs["gates.csv"] = pd.DataFrame(gates)
    for name, t in tabs.items():
        if t.empty:
            print(f"no rows for {name}")
            continue
        t.to_csv(out / name, index=False)
        print(f"wrote {out / name} ({len(t)} rows)")

    # ------------------------------------------------------------ print
    print("\n==== 16:00 QLIKE (clock back-transform), same rows per bucket")
    q = tabs["qlike_1600.csv"]
    print(q.round(4).to_string(index=False))
    print(
        "\n==== paired QLIKE differences (forecast - reference; % of reference; day-block 95 % interval)"
    )
    p = tabs["qlike_1600_paired.csv"]
    print(
        p[
            [
                "bucket",
                "sample",
                "forecast",
                "reference",
                "n",
                "QLIKE_forecast",
                "QLIKE_reference",
                "pct",
                "pct_ci_lo",
                "pct_ci_hi",
                "dm_stat",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print(
        "\n==== 15:30 sign(s) trade, deck days (research scorer, 16:00-bar recalibration)"
    )
    print(tabs["trade_1600.csv"].round(3).to_string(index=False))
    print(
        "\n==== paired Sharpe differences (forecast - reference; block-bootstrap 95 % interval)"
    )
    tp = tabs["trade_1600_paired.csv"]
    print(
        tp[
            [
                "bucket",
                "forecast",
                "reference",
                "days",
                "sign_agreement",
                "Sharpe_mid_forecast",
                "Sharpe_mid_reference",
                "diff_mid",
                "ci_lo_mid",
                "ci_hi_mid",
                "diff_crossed",
                "ci_lo_crossed",
                "ci_hi_crossed",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    for name in (
        "grid_edges_model_axis.csv",
        "qsel_vs_mse.csv",
        "tuned_vs_untuned_bars.csv",
        "vs_linear_bars.csv",
    ):
        if name in tabs:
            print(f"\n==== {name}")
            t = tabs[name]
            if name == "grid_edges_model_axis.csv":
                t = t.drop(columns=["grid"])
            print(t.round(3).to_string(index=False))
    g = tabs["gates.csv"]
    bad = g[~g["ok"]] if not g.empty else g
    print(f"\ngates: {len(g)} checked, {len(bad)} failed")
    if len(bad):
        print(bad.to_string(index=False))
        sys.exit(f"{len(bad)} gate(s) failed")


if __name__ == "__main__":
    main()
