"""Report of the Optuna per-bar tree campaign at the 16:00 bar (checklist I3).

INPUTS
  stage-1 records  results/linear_subsection_trees_optuna_mask/stage1/<bucket>/bar1600/<model>/tw2000/
                   trials_bar1600.npz (every trial of every tuning point; experiments/
                   reduce_trees_optuna.py --stage 1)
  forecast tables  results/spxw_pnl/yhat_subtree_optuna[q]_tp<N>[_k<k>]_<bucket>_<model>.parquet
                   (experiments/build_subsection_tree_optuna_yhat.py; 16:00 rows only)
  comparators      results/spxw_pnl/yhat_subtree_daily_<bucket>_<model>.parquet (T1: shipped
                   configuration, refit every session) and yhat_subtree_tuned_daily_<bucket>_<model>
                   .parquet (RS1: the 32-candidate random search every 250 sessions, refit every
                   session) when they exist; the per-bar ridge yhat_sub_ridge_<bucket>.parquet;
                   the master table's reference (the paper's block-diagonal ridge)

THE SCORER (research convention, 16:00-bar recalibration; nothing re-implemented): every
forecast goes through experiments/master_table_close.py's load_one (the 16:00 rows joined to
the common per-bar target, pred_clock = score_linear_subsection_causal.causal_forecasts) and
score_one (16:00 QLIKE of pred_clock; the deck's 15:30 sign(s) trade on the straddle through
trade_days = score_linear_subsection.trade_1530 line for line; Sharpe mid / crossed; paired
day-block bootstrap intervals -- score_linear_subsection.day_block_ci for the QLIKE
difference, the master table's paired_sharpe for the Sharpe difference; Diebold-Mariano).
Every comparison is on the SAME days (the deck days every forecast of the comparison covers;
866 when all cover the deck).

TABLES (results/linear_subsection_trees_optuna_mask/report/)
  (i)  budget_curve.csv   per arm and trial k = 1..50: the best-so-far validation MSE relative
                          to the best of 50 (mean / median / 90th percentile over the tuning
                          points), the share of points already at their best-of-50, and the gain
                          over trial 1 (the shipped configuration)
       budget_points.csv  per arm: the trial index of the best, the share of points whose best
                          came in trials 41-50 (vs 20 % if the 50 trials were exchangeable draws),
                          the relative improvement of the best-so-far from trial 10 / 25 / 40 to
                          trial 50, how often the configuration of record changes, round-cap hits
       budget_oos.csv     out of sample: best-of-10 / 25 / 50 at TUNE_PER 1 and 25, paired
  (ii) levels.csv         every forecast scored alone (QLIKE, Sharpe mid / crossed, vs always short)
       tuneper_oos.csv    TUNE_PER 1 / 5 / 25 against 250, and every Optuna path against T1, RS1
                          and the per-bar ridge of the same bucket (paired); the QLIKE rule
                          against the MSE rule at TUNE_PER 25
  (iii) picks_bounds.csv  per arm and parameter: the share of the best-of-50 picks within
                          BOUND_SHARE of the axis's low / high bound (on the log scale for log
                          axes; the categorical RF depth: its smallest / largest choice), next to
                          the share of all trials and of TPE's random start-up draws (the
                          prior's own share); median and quartiles of the picks; the round-cap
                          pseudo-axis for the boosted models
  kept_columns.csv        the per-window column mask (WINDOW_MASK=1): kept columns per tuning
                          point (stage 1) and per refit (stage 2, the tp1 path), median / min /
                          max per model x input set, beside the design's width p
  gates.csv               the scorer's load gates (baseline equals the target's B) and this
                          script's (every table covers the deck days; stage-1 files complete)
and results/linear_subsection_trees_optuna_mask/SUMMARY.md, generated from those CSVs.

Run:  python experiments/optuna_trees_report_1600.py [--root results/linear_subsection_trees_optuna_mask]
          [--tables results/spxw_pnl]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments", ROOT / "specs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_subsection_tree_optuna_yhat as bto  # noqa: E402
import causal_tune_trees_optuna_jobs as OJ  # noqa: E402 -- SPACES (the searched axes)
import master_table_close as mtc  # noqa: E402
import score_linear_subsection as base  # noqa: E402

OPT = ROOT / "results" / "linear_subsection_trees_optuna_mask"  # --root
OUT = OPT / "report"  # <root>/report
SPXW = ROOT / "results" / "spxw_pnl"  # comparators (and, by default, the Optuna tables)
TABLES = SPXW  # --tables: where the Optuna yhat tables are
PRE_DEDUP = SPXW / "pre_dedup_2026-09-29"
# the unmasked de-dup tables (agents A / B) snapshotted before the per-window-mask re-runs
DEDUP_NOMASK = SPXW / "dedup_nomask_2026-09-29"
SEG, TW = "bar1600", 2000
MODELS = ("lgbm", "xgb", "rf")
BUCKETS = ("all_features", "baseline")  # live_feasible commented out
MODEL_LONG = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "random forest"}
BOUND_SHARE = 0.05  # "within 5 % of each bound" (the campaign brief)
LATE = (41, 50)  # trials 41-50 (1-based): the last ten of fifty
BUDGET_MARKS = (10, 25, 40)  # best-so-far at these trials against trial 50
N_STARTUP = 10  # TPESampler's default n_startup_trials: trials 1-10 are random draws (trial 1 = shipped)
TP_PATHS = ("tp1", "tp5", "tp25", "tp250")
REF_TP = "tp250"


# --------------------------------------------------------------------------- stage-1 records
def trials_path(bucket: str, model: str) -> Path:
    return OPT / "stage1" / bucket / SEG / model / f"tw{TW}" / f"trials_{SEG}.npz"


def load_trials(bucket: str, model: str) -> dict | None:
    p = trials_path(bucket, model)
    if not p.is_file():
        return None
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def best_so_far(val: np.ndarray) -> np.ndarray:
    v = np.where(np.isnan(val), np.inf, val)
    return np.minimum.accumulate(v, axis=1)


def budget_tables(st: dict, bucket: str, model: str) -> tuple[list[dict], dict]:
    val = st["val_mse"]
    P, N = val.shape
    bsf = best_so_far(val)
    final = bsf[:, -1]
    key = {"model": model, "bucket": bucket}
    curve = []
    for k in range(1, N + 1):
        gap = bsf[:, k - 1] / final - 1.0
        gain = 1.0 - bsf[:, k - 1] / np.where(np.isnan(val[:, 0]), np.inf, val[:, 0])
        curve.append(
            key
            | {
                "trial": k,
                "rel_gap_to_best50_mean": float(gap.mean()),
                "rel_gap_to_best50_median": float(np.median(gap)),
                "rel_gap_to_best50_p90": float(np.quantile(gap, 0.9)),
                "share_points_at_best50": float(np.mean(bsf[:, k - 1] == final)),
                "gain_over_shipped_mean": float(gain.mean()),
            }
        )
    best = st["best_mse_k50"] + 1  # 1-based trial number
    row = key | {
        "points": P,
        "trials": N,
        "best_trial_mean": float(best.mean()),
        "best_trial_median": float(np.median(best)),
        "share_best_is_shipped": float(np.mean(best == 1)),
        "share_best_in_1_10": float(np.mean(best <= 10)),
        "share_best_in_11_25": float(np.mean((best >= 11) & (best <= 25))),
        "share_best_in_26_40": float(np.mean((best >= 26) & (best <= 40))),
        f"share_best_in_{LATE[0]}_{LATE[1]}": float(
            np.mean((best >= LATE[0]) & (best <= LATE[1]))
        ),
        f"share_best_in_{LATE[0]}_{LATE[1]}_if_exchangeable": (LATE[1] - LATE[0] + 1)
        / N,
    }
    for m in BUDGET_MARKS:
        imp = (bsf[:, min(m, N) - 1] - final) / bsf[
            :, min(m, N) - 1
        ]  # N < m: a test run
        row |= {
            f"rel_impr_{m}_to_50_mean": float(imp.mean()),
            f"rel_impr_{m}_to_50_median": float(np.median(imp)),
            f"rel_impr_{m}_to_50_p90": float(np.quantile(imp, 0.9)),
            f"share_impr_{m}_to_50_positive": float(np.mean(imp > 0)),
        }
    imp0 = (val[:, 0] - final) / val[:, 0]
    row |= {
        "rel_impr_shipped_to_50_mean": float(np.nanmean(imp0)),
        "rel_impr_shipped_to_50_median": float(np.nanmedian(imp0)),
    }
    for k in (10, 25):
        row[f"share_pick_k{k}_differs_from_k50"] = float(
            np.mean(st[f"best_mse_k{k}"] != st["best_mse_k50"])
        )
    row["share_qlike_pick_differs_from_mse_pick"] = float(
        np.mean(st["best_qlike_k50"] != st["best_mse_k50"])
    )
    capped = (st["rounds_max"] > 0) & (st["rounds"] >= st["rounds_max"])
    if (st["rounds_max"] > 0).any():
        pick_cap = np.take_along_axis(capped, st["best_mse_k50"][:, None], 1)[:, 0]
        row |= {
            "cap_hit_share_all_trials": float(capped.mean()),
            "cap_hit_share_picks": float(pick_cap.mean()),
        }
    else:
        row |= {"cap_hit_share_all_trials": np.nan, "cap_hit_share_picks": np.nan}
    row |= {
        "sec_per_trial_mean": float(st["sec"].mean()),
        "trial_core_hours": float(st["sec"].sum() / 3600.0),
        "study_core_hours": float(st["study_sec"].sum() / 3600.0),
        "first_date": str(st["date"][0]),
        "last_date": str(st["date"][-1]),
    }
    return curve, row


def axis_position(model: str, axis: str, v: np.ndarray) -> np.ndarray | None:
    s = OJ.SPACES[model][axis]
    if s[0] == "cat":
        return None
    lo, hi, log = float(s[1]), float(s[2]), bool(s[3])
    if log:
        return (np.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return (v - lo) / (hi - lo)


def picks_table(st: dict, bucket: str, model: str) -> list[dict]:
    axes = st["meta"]["axes"]
    assert axes == list(OJ.SPACES[model]), (axes, list(OJ.SPACES[model]))
    params = st["params"]  # points x trials x axes
    pick = np.take_along_axis(params, st["best_mse_k50"][:, None, None], 1)[:, 0, :]
    rows = []
    ship = (
        OJ.encode(model, st["meta"]["shipped_config"] | {})
        if isinstance(st["meta"].get("shipped_config"), dict)
        else None
    )
    for j, ax in enumerate(axes):
        s = OJ.SPACES[model][ax]
        vp, va = pick[:, j], params[:, :, j].ravel()
        vs = params[
            :, 1:N_STARTUP, j
        ].ravel()  # TPE's random start-up draws (trial 1 is enqueued)
        r = {"model": model, "bucket": bucket, "axis": ax, "points": len(vp)}
        if s[0] == "cat":
            ch = list(s[1])
            lo_v, hi_v = ch[0], ch[-1]

            def at(v, target):
                return np.isnan(v) if target is None else (v == target)

            r |= {
                "space": "{"
                + ", ".join("None" if c is None else str(c) for c in ch)
                + "}",
                "low_bound": str(lo_v),
                "high_bound": "None (unbounded)" if hi_v is None else str(hi_v),
                "share_picks_at_low": float(at(vp, lo_v).mean()),
                "share_picks_at_high": float(at(vp, hi_v).mean()),
                "share_trials_at_low": float(at(va, lo_v).mean()),
                "share_trials_at_high": float(at(va, hi_v).mean()),
                "share_startup_at_low": float(at(vs, lo_v).mean()),
                "share_startup_at_high": float(at(vs, hi_v).mean()),
                "pick_mode": "None"
                if np.isnan(pd.Series(vp).mode().iloc[0])
                else str(int(pd.Series(vp).mode().iloc[0])),
            }
            counts = {
                ("None" if c is None else str(c)): float(at(vp, c).mean()) for c in ch
            }
            r["pick_shares"] = json.dumps(counts)
        else:
            up, ua, us = (axis_position(model, ax, x) for x in (vp, va, vs))
            assert up is not None and ua is not None and us is not None
            r |= {
                "space": f"[{s[1]:g}, {s[2]:g}]"
                + (" log" if s[3] else "")
                + (" int" if s[0] == "int" else ""),
                "low_bound": f"{s[1]:g}",
                "high_bound": f"{s[2]:g}",
                "share_picks_at_low": float(np.mean(up <= BOUND_SHARE)),
                "share_picks_at_high": float(np.mean(up >= 1.0 - BOUND_SHARE)),
                "share_trials_at_low": float(np.mean(ua <= BOUND_SHARE)),
                "share_trials_at_high": float(np.mean(ua >= 1.0 - BOUND_SHARE)),
                "share_startup_at_low": float(np.mean(us <= BOUND_SHARE)),
                "share_startup_at_high": float(np.mean(us >= 1.0 - BOUND_SHARE)),
                "pick_q25": float(np.quantile(vp, 0.25)),
                "pick_median": float(np.median(vp)),
                "pick_q75": float(np.quantile(vp, 0.75)),
            }
        if ship is not None:
            r["shipped"] = "None" if np.isnan(ship[j]) else f"{ship[j]:g}"
        rows.append(r)
    capped = (st["rounds_max"] > 0) & (st["rounds"] >= st["rounds_max"])
    if (st["rounds_max"] > 0).any():
        pc = np.take_along_axis(capped, st["best_mse_k50"][:, None], 1)[:, 0]
        pr = np.take_along_axis(st["rounds"], st["best_mse_k50"][:, None], 1)[:, 0]
        rows.append(
            {
                "model": model,
                "bucket": bucket,
                "axis": "rounds_cap (pseudo-axis)",
                "points": len(pc),
                "space": "early-stopped rounds <= cap(lr)",
                "high_bound": "cap",
                "share_picks_at_high": float(pc.mean()),
                "share_trials_at_high": float(capped.mean()),
                "share_startup_at_high": float(capped[:, 1:N_STARTUP].mean()),
                "pick_q25": float(np.quantile(pr, 0.25)),
                "pick_median": float(np.median(pr)),
                "pick_q75": float(np.quantile(pr, 0.75)),
            }
        )
    return rows


def kept_rows(st: dict, bucket: str, model: str) -> dict:
    """Kept-column counts of the per-window mask: stage 1 (per tuning point) and stage 2
    (per refit of the tp1 path, from its merged npz when pulled)."""
    row = {
        "model": model,
        "bucket": bucket,
        "window_mask": bool(st["meta"].get("window_mask", False)),
    }
    row["p"] = int(st["meta"].get("n_features", -1))
    k1 = np.asarray(st.get("n_kept", np.array([], dtype=np.int64)))
    if k1.size:
        row |= {
            "stage1_points": int(k1.size),
            "stage1_median": float(np.median(k1)),
            "stage1_min": int(k1.min()),
            "stage1_max": int(k1.max()),
        }
    f = (
        OPT
        / "paths"
        / "tp1"
        / bucket
        / SEG
        / model
        / f"tw{TW}"
        / "causal_tune_trees"
        / model
        / bucket
        / f"trees_{SEG}.npz"
    )
    if f.is_file():
        with np.load(f, allow_pickle=False) as z:
            k2 = (
                np.asarray(z["n_kept"])
                if "n_kept" in z.files
                else np.array([], dtype=np.int64)
            )
        if k2.size:
            row |= {
                "stage2_refits": int(k2.size),
                "stage2_median": float(np.median(k2)),
                "stage2_min": int(k2.min()),
                "stage2_max": int(k2.max()),
            }
    return row


# --------------------------------------------------------------------------- forecasts
def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def provenance(p: Path) -> str:
    """Which version of a comparator table this is: equal to the pre-de-duplication snapshot,
    equal to the unmasked de-dup snapshot, or neither (a later re-run), and when written."""
    h = md5(p)
    tags = []
    for d, name in (
        (PRE_DEDUP, "the pre-de-duplication snapshot"),
        (DEDUP_NOMASK, "the de-dup unmasked snapshot"),
    ):
        q = d / p.name
        if q.is_file():
            tags.append(("= " if md5(q) == h else "differs from ") + name)
    if not tags:
        tags.append("no snapshot to compare")
    when = pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC").tz_convert(
        "America/Los_Angeles"
    )
    return "; ".join(tags) + f"; written {when:%Y-%m-%d %H:%M} PT"


def forecast_catalog() -> dict[str, dict]:
    """key -> {path, label, family, model, bucket, kind}; only files on disk."""
    cat: dict[str, dict] = {}
    for model in MODELS:
        for bucket in BUCKETS:
            for path in bto.TABLES:
                p = TABLES / bto.table_name(path, bucket, model)
                if p.is_file():
                    cat[f"{path}|{bucket}|{model}"] = {
                        "path": p,
                        "label": f"Optuna {MODEL_LONG[model]} [{bucket}] {path}",
                        "model": model,
                        "bucket": bucket,
                        "kind": path,
                    }
            for kind, stem in (
                ("T1", f"yhat_subtree_daily_{bucket}_{model}"),
                ("RS1", f"yhat_subtree_tuned_daily_{bucket}_{model}"),
            ):
                p = SPXW / f"{stem}.parquet"
                if p.is_file():
                    cat[f"{kind}|{bucket}|{model}"] = {
                        "path": p,
                        "label": f"{kind} {MODEL_LONG[model]} [{bucket}]",
                        "model": model,
                        "bucket": bucket,
                        "kind": kind,
                        "snapshot": provenance(p),
                    }
    for bucket in BUCKETS:
        p = SPXW / f"yhat_sub_ridge_{bucket}.parquet"
        if p.is_file():
            cat[f"ridge|{bucket}|-"] = {
                "path": p,
                "label": f"per-bar ridge [{bucket}]",
                "model": "ridge",
                "bucket": bucket,
                "kind": "ridge",
                "snapshot": provenance(p),
            }
    ref = SPXW / "yhat_blk2_fomc1.parquet"
    cat["blk2|-|-"] = {
        "path": ref,
        "label": "block-diagonal ridge (the master table's reference)",
        "model": "-",
        "bucket": "-",
        "kind": "reference",
    }
    return cat


def pair_row(ka: str, kb: str, frames: dict, days, deck, what: str) -> dict:
    spec = mtc.Spec(ka, ka, "optuna", "table", "")
    o = mtc.score_one((spec, frames[ka], frames[kb], days, deck, False))
    ma, mb = ka.split("|"), kb.split("|")
    return {
        "comparison": what,
        "a": ka,
        "b": kb,
        "model": ma[2],
        "bucket": ma[1],
        "a_kind": ma[0],
        "b_kind": mb[0],
        "n_days": o["n_days"],
        "qlike_a": o["qlike_recal"],
        "qlike_pct_a_vs_b": o["qlike_pct_vs_ref"],
        "qlike_diff_a_minus_b": None,
        "qlike_diff_ci_lo": o["qlike_diff_ci_lo"],
        "qlike_diff_ci_hi": o["qlike_diff_ci_hi"],
        "dm_a_vs_b": o["dm_vs_ref"],
        "dm_p": o["dm_p_vs_ref"],
        "Sharpe_mid_a": o["Sharpe_mid"],
        "dSharpe_mid": o["dSharpe_mid_vs_ref"],
        "dSharpe_mid_lo": o["dSharpe_mid_vs_ref_lo"],
        "dSharpe_mid_hi": o["dSharpe_mid_vs_ref_hi"],
        "Sharpe_crossed_a": o["Sharpe_crossed"],
        "dSharpe_crossed": o["dSharpe_crossed_vs_ref"],
        "dSharpe_crossed_lo": o["dSharpe_crossed_vs_ref_lo"],
        "dSharpe_crossed_hi": o["dSharpe_crossed_vs_ref_hi"],
        "same_position_share": o["same_position_as_ref"],
        "pct_buy_a": o["pct_buy"],
    }


# --------------------------------------------------------------------------- summary
def f2(x, nd=2, sign=True) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def pc(x, nd=1) -> str:
    """A share as a percentage, or n/a."""
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{nd}f} %"


def ci(v, lo, hi, nd=2) -> str:
    return f"{f2(v, nd)} [{f2(lo, nd)}, {f2(hi, nd)}]"


def pct_ci(r: pd.Series) -> str:
    """QLIKE difference in % of b's QLIKE, with the day-block interval scaled alike."""
    qb = r["qlike_a"] / (1.0 + r["qlike_pct_a_vs_b"] / 100.0)
    return f"{r['qlike_pct_a_vs_b']:+.1f} % [{100 * r['qlike_diff_ci_lo'] / qb:+.1f}, {100 * r['qlike_diff_ci_hi'] / qb:+.1f}]"


def write_summary(
    bp: pd.DataFrame,
    curve: pd.DataFrame,
    lev: pd.DataFrame,
    pairs: pd.DataFrame,
    picks: pd.DataFrame,
    gates: pd.DataFrame,
    cat: dict,
    usage: pd.DataFrame | None,
    kept: pd.DataFrame | None = None,
    hosts: pd.DataFrame | None = None,
    xclass: pd.DataFrame | None = None,
    mix: pd.DataFrame | None = None,
    note: str = "",
) -> str:
    L: list[str] = []
    L.append("# Optuna per-bar trees at the 16:00 bar (checklist I3)")
    L.append("")
    if note:
        L.append(f"**{note}**")
        L.append("")
    L.append(
        "Generated by `experiments/optuna_trees_report_1600.py` from its own CSVs in `report/`. Spec "
        "`specs/causal_tune_trees_optuna.py` (+ `_jobs.py`): LightGBM / XGBoost / random forest x "
        "`baseline` / `live_feasible` / `all_features`, the 16:00 bar, 2000-session window, the "
        "de-duplicated per-bar design (commit 47f7f9c). A tuning point at every session: a fresh Optuna "
        "TPE study (seed = base + the row), 50 sequential trials, trial 1 = the shipped configuration, "
        "objective = validation MSE on the 125-session tail after a 25-session embargo; trees refit "
        "EVERY session with the configuration in force (TUNE_PER = 1 / 5 / 25 / 250; best of the first "
        "10 / 25 / 50 trials; the QLIKE-selected best of 50 at TUNE_PER 25)."
    )
    L.append("")
    if kept is not None and len(kept) and bool(kept["window_mask"].all()):
        L.append(
            "Per-window column mask ON (user decision 2026-09-29; `src/models/window_mask.py`, the linear "
            "arms' identifiability rule): every fit uses only the columns that are not constant on its "
            "window and not exact copies of an earlier kept column -- a tuning point's kept set from its "
            "full window [t - 2000, t), each refit's from its own window; importance and TreeSHAP are "
            "mapped back to all columns (0 for a dropped one). Kept columns (`report/kept_columns.csv`):"
        )
        L.append("")
        L.append(
            "| arm | p | tuning points: median [min, max] | refits (tp1): median [min, max] |"
        )
        L.append("|---|---|---|---|")
        for _, r in kept.iterrows():
            s1 = (
                f"{r['stage1_median']:.0f} [{int(r['stage1_min'])}, {int(r['stage1_max'])}]"
                if "stage1_median" in r and np.isfinite(r["stage1_median"])
                else "n/a"
            )
            s2 = (
                f"{r['stage2_median']:.0f} [{int(r['stage2_min'])}, {int(r['stage2_max'])}]"
                if "stage2_median" in r and np.isfinite(r["stage2_median"])
                else "n/a"
            )
            L.append(
                f"| {MODEL_LONG[r['model']]} [{r['bucket']}] | {int(r['p'])} | {s1} | {s2} |"
            )
        L.append("")
    L.append(
        "Scorer: the research convention (16:00-bar recalibration, `master_table_close.load_one` / "
        "`score_one`); the last-30-min trade is the deck's 15:30 sign(s) rule on the straddle (nearest "
        "out-of-the-money call + nearest out-of-the-money put, same-day expiry, one position). Intervals: "
        "95 % paired day-block bootstrap (21-day blocks, 2000 draws). QLIKE differences in % of the "
        "comparator's QLIKE (negative = the first forecast is better); Sharpe differences first minus "
        "second."
    )
    L.append("")
    if usage is not None and len(usage):
        L.append("## Cluster")
        L.append("")
        L.append(
            "The masked run (canary, stage 1, merges, stage 2, the resume chain, the re-run gates; `cluster_usage.csv` from `sacct.txt`):"
        )
        L.append("")
        for _, r in usage.iterrows():
            L.append(f"- {r['item']}: {r['value']}")
        L.append("")
        rr = OPT / "cluster_usage_single_class_rerun.csv"
        if rr.is_file():
            L.append(
                "Of which the single-class re-run (s1e / m1e / s2e / m2e, epyc-7513):"
            )
            L.append("")
            for _, r in pd.read_csv(rr).iterrows():
                L.append(f"- {r['item']}: {r['value']}")
            L.append("")
        un = OPT / "cluster_usage_unmasked_run.csv"
        if un.is_file():
            L.append(
                "The unmasked run (WINDOW_MASK=0; stopped by the user decision of 2026-09-29 and not used; "
                "`cluster_usage_unmasked_run.csv`):"
            )
            L.append("")
            for _, r in pd.read_csv(un).iterrows():
                L.append(f"- {r['item']}: {r['value']}")
            L.append("")
    if hosts is not None and len(hosts):
        L.append("## Where the chunks ran (CPU class)")
        L.append("")
        L.append(
            "The executor's design matrix differs at rounding level between CPU vector classes (agent D: "
            "AVX-512 nodes agree bit for bit with each other, AVX2 nodes give a different X, max abs ~1e-11; "
            "the target is identical). Reproducibility guarantee: bit-identical on the same CPU class; "
            "rounding-level differences (<= 1e-11 absolute in X) across classes. Chunk-arms by class "
            "(`chunk_hosts.csv`, from each task's Slurm log / host.txt and `scontrol show node`):"
        )
        L.append("")
        tab = (
            hosts.groupby(["stage", "bucket", "model", "cpu_class"])
            .size()
            .unstack("cpu_class", fill_value=0)
        )
        cls = list(tab.columns)
        L.append("| stage | arm | " + " | ".join(str(c) for c in cls) + " |")
        L.append("|---|---|" + "---|" * len(cls))
        for (stage, bucket, model), r in tab.iterrows():
            L.append(
                f"| {stage} | {MODEL_LONG.get(model, model)} [{bucket}] | "
                + " | ".join(str(int(r[c])) for c in cls)
                + " |"
            )
        L.append("")
    if xclass is not None and len(xclass):
        L.append(
            "Re-run gate (`xclass*/xclass_gate.csv`): the first 20 tuning points of the live_feasible arms "
            "(OOS rows 0-19) re-run on a node of a chosen CPU class (Slurm --constraint) against the fleet's "
            "chunk 0 of the same arm (each re-run task then repeats its chunk on its own node, the canary "
            "check -- 'canary repeat OK' in the task logs; the xeon-4116 LightGBM task hit its 1 h limit before "
            "its repeat finished):"
        )
        L.append("")
        # the reproducibility statement, one sentence per re-run, from the gate rows
        for rr, g in xclass.groupby("rerun", sort=False):
            parts = []
            for _, r in g.iterrows():
                name = MODEL_LONG.get(r["model"], r["model"])
                n = int(r.get("points", 0) or 0)
                vm = float(r.get("val_mse_max_rel_diff", np.nan))
                if bool(r.get("records_bit_identical", False)):
                    parts.append(f"{name} bit-identical on {n}/{n} points")
                else:
                    parts.append(
                        f"{name} val MSE max rel diff {vm:.1e}, shipped-trial val MSE identical on "
                        f"{int(r.get('points_trial1_identical', 0))}/{n}, best-of-50 pick changed on "
                        f"{int(r.get('best_mse_k50_differs', 0))}/{n}"
                    )
            fleet = str(g["fleet_node"].iloc[0]).split("(")[-1].rstrip(")")
            rnode = str(g["xclass_node"].iloc[0])
            rcls = (
                "epyc-7513"
                if "EPYC 7513" in rnode
                else (
                    "xeon-4116"
                    if "4116" in rnode
                    else rnode.split(";")[1]
                    if ";" in rnode
                    else ""
                )
            )
            L.append(
                f"- `{rr}`: {rcls} re-run vs the fleet's {fleet} chunk 0 -- "
                + "; ".join(parts)
                + "."
            )
        L.append("")
        L.append(
            "| re-run | model | fleet node | re-run node | records bit-identical | max rel diff val MSE | best-of-50 pick differs | tp1 forecast max rel diff |"
        )
        L.append("|---|---|---|---|---|---|---|---|")
        for _, r in xclass.iterrows():
            tp1 = float(r.get("tp1_forecast_max_rel_diff", np.nan))
            vm = float(r.get("val_mse_max_rel_diff", np.nan))
            node = str(r.get("xclass_node", "")).split(";")[0].replace("host=", "")
            L.append(
                f"| {r.get('rerun', '')} | {MODEL_LONG.get(r['model'], r['model'])} | {r.get('fleet_node', '')} | {node} | "
                f"{r.get('records_bit_identical', '')} | {'n/a' if not np.isfinite(vm) else f'{vm:.1e}'} | "
                f"{r.get('best_mse_k50_differs', '')} of {r.get('points', '')} | "
                f"{'n/a' if not np.isfinite(tp1) else f'{tp1:.1e}'} |"
            )
        L.append("")
    if mix is not None and len(mix):
        L.append("## Single-class run vs the class-mixed run")
        L.append("")
        L.append(
            "The first pass of this masked run landed on four node types; it is kept as the class-mixed run "
            "(`results/linear_subsection_trees_optuna_mask_mixed/`, its tables in "
            "`results/spxw_pnl/optuna_mask_mixed_2026-09-30/`). This run re-computed on epyc-7513 every chunk "
            "that had run elsewhere and every stage-2 chunk whose configuration came from a re-computed tuning "
            "point (`single_class_plan.csv`). Per forecast (`report/single_vs_mixed.csv`; single minus mixed, "
            "same 866 days):"
        )
        L.append("")
        L.append(
            "| arm | path | rows differing | max rel forecast diff | QLIKE single vs mixed | dSharpe mid | dSharpe crossed |"
        )
        L.append("|---|---|---|---|---|---|---|")
        for _, r in mix.iterrows():
            L.append(
                f"| {MODEL_LONG[r['model']]} [{r['bucket']}] | {r['a_kind']} | {int(r['forecast_rows_differ'])} of {int(r['rows'])} | "
                f"{r['forecast_max_rel_diff']:.1e} | {pct_ci(r)} | "
                f"{ci(r['dSharpe_mid'], r['dSharpe_mid_lo'], r['dSharpe_mid_hi'])} | "
                f"{ci(r['dSharpe_crossed'], r['dSharpe_crossed_lo'], r['dSharpe_crossed_hi'])} |"
            )
        L.append("")
    L.append("## (i) Is 50 trials enough?")
    L.append("")
    L.append(
        f"Per tuning point, the best-so-far validation MSE along the 50 trials (`budget_curve.csv`, "
        f"`budget_points.csv`). If the 50 trials were exchangeable draws, the best would fall in trials "
        f"{LATE[0]}-{LATE[1]} at {100 * (LATE[1] - LATE[0] + 1) / 50:.0f} % of points. Trials 1-{N_STARTUP} "
        "are TPE's random start-up (trial 1 = shipped), so best-of-10 = the shipped configuration plus "
        "nine random draws."
    )
    L.append("")
    L.append(
        "| arm | points | best trial (median) | best = shipped | best in 41-50 | impr. 10->50 (mean / median) | "
        "25->50 | 40->50 | pick k25 != k50 | cap hits (picks) |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in bp.iterrows():
        L.append(
            f"| {MODEL_LONG[r['model']]} [{r['bucket']}] | {int(r['points'])} | {r['best_trial_median']:.0f} | "
            f"{100 * r['share_best_is_shipped']:.1f} % | {100 * r[f'share_best_in_{LATE[0]}_{LATE[1]}']:.1f} % | "
            f"{100 * r['rel_impr_10_to_50_mean']:.2f} / {100 * r['rel_impr_10_to_50_median']:.2f} % | "
            f"{100 * r['rel_impr_25_to_50_mean']:.2f} / {100 * r['rel_impr_25_to_50_median']:.2f} % | "
            f"{100 * r['rel_impr_40_to_50_mean']:.2f} / {100 * r['rel_impr_40_to_50_median']:.2f} % | "
            f"{100 * r['share_pick_k25_differs_from_k50']:.0f} % | "
            f"{pc(r['cap_hit_share_picks'])} |"
        )
    L.append("")
    oos = pairs[pairs["comparison"].str.startswith("budget")]
    if len(oos):
        L.append(
            "Out of sample (16:00 QLIKE, sign(s) Sharpe mid / crossed; the larger budget minus the smaller):"
        )
        L.append("")
        L.append(
            "| arm | TUNE_PER | comparison | QLIKE (a) | QLIKE a vs b | dSharpe mid | dSharpe crossed |"
        )
        L.append("|---|---|---|---|---|---|---|")
        for _, r in oos.iterrows():
            L.append(
                f"| {MODEL_LONG[r['model']]} [{r['bucket']}] | {r['comparison'].split()[-1]} | {r['a_kind']} vs {r['b_kind']} | "
                f"{r['qlike_a']:.4f} | {pct_ci(r)} | {ci(r['dSharpe_mid'], r['dSharpe_mid_lo'], r['dSharpe_mid_hi'])} | "
                f"{ci(r['dSharpe_crossed'], r['dSharpe_crossed_lo'], r['dSharpe_crossed_hi'])} |"
            )
        L.append("")
    L.append(
        "## (ii) TUNE_PER ablation (1 / 5 / 25 / 250), every path refit every session"
    )
    L.append("")
    lv = lev.set_index("key")
    L.append("| arm | forecast | QLIKE | Sharpe mid | Sharpe crossed | % buy |")
    L.append("|---|---|---|---|---|---|")
    for model in MODELS:
        for bucket in BUCKETS:
            for kind in ("tp1", "tp5", "tp25", "tp250", "tp25_q", "T1", "RS1"):
                k = f"{kind}|{bucket}|{model}"
                if k in lv.index:
                    r = lv.loc[k]
                    L.append(
                        f"| {MODEL_LONG[model]} [{bucket}] | {kind} | {r['qlike_recal']:.4f} | {r['Sharpe_mid']:.2f} | {r['Sharpe_crossed']:.2f} | {r['pct_buy']:.1f} |"
                    )
    for bucket in BUCKETS:
        k = f"ridge|{bucket}|-"
        if k in lv.index:
            r = lv.loc[k]
            L.append(
                f"| per-bar ridge [{bucket}] | ridge | {r['qlike_recal']:.4f} | {r['Sharpe_mid']:.2f} | {r['Sharpe_crossed']:.2f} | {r['pct_buy']:.1f} |"
            )
    L.append("")
    tp = pairs[pairs["comparison"].str.startswith(("tune_per", "vs "))]
    if len(tp):
        L.append("Paired (a minus b):")
        L.append("")
        L.append(
            "| arm | a vs b | QLIKE a vs b | dSharpe mid | dSharpe crossed | same position |"
        )
        L.append("|---|---|---|---|---|---|")
        for _, r in tp.iterrows():
            L.append(
                f"| {MODEL_LONG[r['model']]} [{r['bucket']}] | {r['a_kind']} vs {r['b_kind']} | {pct_ci(r)} | "
                f"{ci(r['dSharpe_mid'], r['dSharpe_mid_lo'], r['dSharpe_mid_hi'])} | "
                f"{ci(r['dSharpe_crossed'], r['dSharpe_crossed_lo'], r['dSharpe_crossed_hi'])} | {100 * r['same_position_share']:.0f} % |"
            )
        L.append("")
    missing_cmp = [
        k for k in ("T1", "RS1") if not any(v["kind"] == k for v in cat.values())
    ]
    if missing_cmp:
        L.append(
            f"Not on disk when this ran (so not compared): {', '.join(missing_cmp)} (agent B's `yhat_subtree_daily_*` / `yhat_subtree_tuned_daily_*`)."
        )
        L.append("")
    notes = [
        f"`{v['path'].name}` ({v['snapshot']})"
        for v in cat.values()
        if v["kind"] in ("ridge", "T1", "RS1")
    ]
    if notes:
        L.append("Comparator tables used: " + "; ".join(notes) + ".")
        L.append("")
    L.append("## (iii) Where the picks sit in the space")
    L.append("")
    L.append(
        f"Share of the best-of-50 picks (all tuning points) within {100 * BOUND_SHARE:.0f} % of an axis's low / high "
        "bound (log scale for log axes; RF depth: the smallest / largest choice), beside the share of all trials "
        "and of the random start-up draws (the prior's share). `picks_bounds.csv` has the quartiles."
    )
    L.append("")
    L.append(
        "| arm | axis | space | picks at low | picks at high | all trials low / high | prior low / high | median pick |"
    )
    L.append("|---|---|---|---|---|---|---|---|")
    for _, r in picks.iterrows():
        med = (
            r["pick_mode"]
            if isinstance(r.get("pick_mode"), str)
            else f"{r['pick_median']:.4g}"
        )
        L.append(
            f"| {MODEL_LONG[r['model']]} [{r['bucket']}] | {r['axis']} | {r['space']} | "
            f"{pc(r.get('share_picks_at_low', np.nan))} | {pc(r['share_picks_at_high'])} | "
            f"{pc(r.get('share_trials_at_low', np.nan))} / {pc(r['share_trials_at_high'])} | "
            f"{pc(r.get('share_startup_at_low', np.nan))} / {pc(r['share_startup_at_high'])} | {med} |"
        )
    L.append("")
    L.append("## Gates")
    L.append("")
    ng = len(gates)
    nf = int((~gates["ok"].astype(bool)).sum()) if ng else 0
    L.append(
        f"{ng} report gates, {nf} failed (`report/gates.csv`). Spec gates: `gates/gate_rows_*.csv` (identity, determinism, TPE prefix, chunk, pool, perturbation, search-space coverage)."
    )
    L.append("")
    return "\n".join(L) + "\n"


def main() -> int:
    global OPT, OUT, TABLES
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", default=str(OPT), help="campaign root (stage1/, report/, SUMMARY.md)"
    )
    ap.add_argument(
        "--tables", default=str(TABLES), help="where the Optuna yhat tables are"
    )
    ap.add_argument(
        "--mixed-tables",
        default=str(SPXW / "optuna_mask_mixed_2026-09-30"),
        help="the class-mixed run's tables (compared with --tables when present)",
    )
    ap.add_argument("--note", default="", help="a status line printed under the title")
    a = ap.parse_args()
    OPT, TABLES = Path(a.root).resolve(), Path(a.tables).resolve()
    OUT = OPT / "report"
    OUT.mkdir(parents=True, exist_ok=True)
    gates: list[dict] = []
    # ---- (i)/(iii) from the stage-1 records
    curves, points, picks, kepts = [], [], [], []
    for model in MODELS:
        for bucket in BUCKETS:
            st = load_trials(bucket, model)
            if st is None:
                gates.append(
                    {
                        "gate": "stage-1 records on disk",
                        "item": f"{bucket} {model}",
                        "ok": False,
                    }
                )
                print(f"no stage-1 records for {bucket} {model}")
                continue
            ok = bool(np.array_equal(st["row"], np.arange(len(st["row"]))))
            gates.append(
                {
                    "gate": "stage-1 rows 0 .. n-1",
                    "item": f"{bucket} {model}",
                    "value": len(st["row"]),
                    "ok": ok,
                }
            )
            c, p = budget_tables(st, bucket, model)
            curves += c
            points.append(p)
            picks += picks_table(st, bucket, model)
            kepts.append(kept_rows(st, bucket, model))
    curve, bp, pk = pd.DataFrame(curves), pd.DataFrame(points), pd.DataFrame(picks)
    curve.to_csv(OUT / "budget_curve.csv", index=False)
    bp.to_csv(OUT / "budget_points.csv", index=False)
    pk.to_csv(OUT / "picks_bounds.csv", index=False)
    kp = pd.DataFrame(kepts)
    kp.to_csv(OUT / "kept_columns.csv", index=False)

    # ---- (i) OOS / (ii) through the master table's scorer
    deck = pd.read_parquet(base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    deck_days = deck.index
    tgt = mtc.load_target()
    mtc._init(tgt, deck_days)
    cat = forecast_catalog()
    frames: dict[str, pd.DataFrame] = {}
    for k, v in cat.items():
        res = mtc.load_one(mtc.Spec(k, v["label"], "optuna", "table", str(v["path"])))
        for g in res["gates"]:
            gates.append(
                {"gate": g["gate"], "item": k, "value": g["value"], "ok": bool(g["ok"])}
            )
        if res["frame"] is None or res["error"]:
            print(f"not scored {k}: {res['error']}")
            gates.append(
                {"gate": "loaded", "item": k, "ok": False, "value": res["error"]}
            )
            continue
        f = res["frame"]
        has = f.index[f["pred_clock"].notna()].normalize()
        miss = deck_days.difference(has)
        gates.append(
            {
                "gate": "covers the deck days",
                "item": k,
                "value": len(miss),
                "ok": len(miss) == 0,
            }
        )
        frames[k] = f
    days = deck_days
    for f in frames.values():
        days = days.intersection(f.index[f["pred_clock"].notna()].normalize())
    print(f"{len(frames)} forecasts scored on {len(days)} common deck days")
    lev = []
    for k in frames:
        o = mtc.score_one(
            (
                mtc.Spec(k, k, "optuna", "table", ""),
                frames[k],
                frames["blk2|-|-"],
                days,
                deck,
                k == "blk2|-|-",
            )
        )
        lev.append(
            {
                "key": k,
                "label": cat[k]["label"],
                "path": str(cat[k]["path"].relative_to(ROOT)).replace("\\", "/"),
            }
            | o
        )
    lev_df = pd.DataFrame(lev)
    lev_df.to_csv(OUT / "levels.csv", index=False)

    pairs = []
    for model in MODELS:
        for bucket in BUCKETS:

            def kk(kind: str) -> str:
                return f"{kind}|{bucket}|{model}"

            for tp in ("tp1", "tp25"):
                for a_, b_ in (
                    (tp, f"{tp}_k10"),
                    (tp, f"{tp}_k25"),
                    (f"{tp}_k25", f"{tp}_k10"),
                ):
                    if kk(a_) in frames and kk(b_) in frames:
                        pairs.append(
                            pair_row(
                                kk(a_),
                                kk(b_),
                                frames,
                                days,
                                deck,
                                f"budget TUNE_PER {tp[2:]}",
                            )
                        )
            for a_ in ("tp1", "tp5", "tp25"):
                if kk(a_) in frames and kk(REF_TP) in frames:
                    pairs.append(
                        pair_row(
                            kk(a_), kk(REF_TP), frames, days, deck, "tune_per vs 250"
                        )
                    )
            if kk("tp25_q") in frames and kk("tp25") in frames:
                pairs.append(
                    pair_row(
                        kk("tp25_q"),
                        kk("tp25"),
                        frames,
                        days,
                        deck,
                        "tune_per rule qlike vs mse",
                    )
                )
            for cmp_ in ("T1", "RS1"):
                for a_ in TP_PATHS:
                    if kk(a_) in frames and kk(cmp_) in frames:
                        pairs.append(
                            pair_row(kk(a_), kk(cmp_), frames, days, deck, f"vs {cmp_}")
                        )
            rk = f"ridge|{bucket}|-"
            for a_ in TP_PATHS:
                if kk(a_) in frames and rk in frames:
                    pairs.append(pair_row(kk(a_), rk, frames, days, deck, "vs ridge"))
    pr = pd.DataFrame(pairs)
    if len(pr):
        pr["qlike_diff_a_minus_b"] = pr["qlike_a"] - pr["qlike_a"] / (
            1.0 + pr["qlike_pct_a_vs_b"] / 100.0
        )
    pr.to_csv(OUT / "tuneper_oos.csv", index=False)
    if len(pr):
        pr[pr["comparison"].str.startswith("budget")].to_csv(
            OUT / "budget_oos.csv", index=False
        )
    # ---- the class-mixed run against this (single-class) run, path by path
    mixed_dir = Path(a.mixed_tables)
    mix_rows = []
    if mixed_dir.is_dir() and mixed_dir.resolve() != TABLES:
        for k, v in cat.items():
            if v["kind"] not in bto.TABLES or k not in frames:
                continue
            q = mixed_dir / v["path"].name
            if not q.is_file():
                continue
            km = k + " (mixed)"
            res = mtc.load_one(
                mtc.Spec(km, v["label"] + " mixed", "optuna", "table", str(q))
            )
            if res["frame"] is None or res["error"]:
                continue
            frames[km] = res["frame"]
            ja = frames[k]["pred_adj"]
            jb = frames[km]["pred_adj"].reindex(ja.index)
            r = pair_row(k, km, frames, days, deck, "single-class vs class-mixed")
            r["forecast_max_rel_diff"] = float(np.nanmax(np.abs(ja / jb - 1.0)))
            r["forecast_rows_differ"] = int(np.sum(ja.to_numpy() != jb.to_numpy()))
            r["rows"] = int(len(ja))
            mix_rows.append(r)
    mix = pd.DataFrame(mix_rows)
    if len(mix):
        mix.to_csv(OUT / "single_vs_mixed.csv", index=False)
    gt = pd.DataFrame(gates)
    gt.to_csv(OUT / "gates.csv", index=False)
    usage_p = OPT / "cluster_usage.csv"
    usage = pd.read_csv(usage_p) if usage_p.is_file() else None
    hosts_p = OPT / "chunk_hosts.csv"
    hosts = pd.read_csv(hosts_p) if hosts_p.is_file() else None
    xcs = [
        pd.read_csv(f).assign(rerun=f.parent.name)
        for f in sorted(OPT.glob("xclass*/xclass_gate.csv"))
    ]
    xclass = pd.concat(xcs, ignore_index=True) if xcs else None
    (OPT / "SUMMARY.md").write_text(
        write_summary(
            bp,
            curve,
            lev_df,
            pr if len(pr) else pd.DataFrame(columns=["comparison"]),
            pk,
            gt,
            cat,
            usage,
            kp,
            hosts,
            xclass,
            mix,
            a.note,
        ),
        encoding="utf-8",
    )
    n_fail = int((~gt["ok"].astype(bool)).sum()) if len(gt) else 0
    print(
        f"wrote {OUT} (budget_curve / budget_points / budget_oos / levels / tuneper_oos / picks_bounds / gates) and SUMMARY.md; gates failed: {n_fail}"
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
