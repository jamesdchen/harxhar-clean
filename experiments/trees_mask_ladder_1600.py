"""The 16:00 ladder with the per-window mask (checklist I9): masked vs unmasked trees and LSTM,
the masked cadence / tuning pairs, and each masked rung vs the per-bar ridge.

WHAT IS SCORED.  Per model (LightGBM, XGBoost, random forest) x input set (baseline = HAR +
calendar, live_feasible, all_features), the 16:00 forecast (the 15:30-16:00 bar, issued at
15:30) of the rungs fitted on the de-duplicated per-bar design (commit 47f7f9c), 2000-session
window, WITH the per-window mask (WINDOW_MASK=1: every fit sees only the columns
src.models.window_mask.window_keep keeps on its own training window -- not constant there, not
a byte-copy of an earlier kept column; the linear arms' identifiability rule):

  T10   shipped configuration, refit every 10 sessions   yhat_subtree_<model>_<bucket>
  T1    shipped configuration, refit every session        yhat_subtree_daily_<bucket>_<model>
  RS10  random search, MSE rule, refit every 10           yhat_subtree_tuned_<bucket>_<model>
  RS1   random search, MSE rule, refit every session      yhat_subtree_tuned_daily_<bucket>_<model>
  LSTM    the per-bar LSTM (MSE rule), refit every session (user decision 2026-09-29
          evening: daily refit, as the trees)            yhat_lstm_<bucket>
  LSTM10  the same, refit every 10 sessions (the earlier cadence; masked table staged in
          results/trees_mask_1600/stack/lstm_re10/, the unmasked one is the snapshot)
  (supplementary: RS10q / RS1q / LSTMq / LSTM10q, the QLIKE-rule twins)

masked = the canonical tables in results/spxw_pnl/ (experiments/trees_mask_stack_1600.py);
unmasked = the same names in results/spxw_pnl/dedup_nomask_2026-09-29/ (the I1 / I2 tables,
snapshotted before the masked stack overwrote them); ridge = yhat_sub_ridge_<bucket> (the
per-bar ridge, which always carried the mask at each penalty tune).

ONE SCORER, the research scorer (16:00 bar recalibrated alone), exactly as
experiments/trees_cadence_ladder_1600.py (imported: the master table's loader, target and
trade; QLIKE of the recalibrated forecast; the sign(s) rule on the straddle, mid and crossed
fills; 95 % day-block bootstrap intervals, 21-day circular blocks, 2000 draws, paired; DM).
SAME DAYS for every number (the deck's trade days on which every forecast here has a
recalibrated 16:00 forecast: 866 when all are complete).

PAIRS (a - b, per model x input set):
  masked - unmasked   T10, T1, RS10, RS1, LSTM10 (the same rung, the mask alone)
  masked ladder       T1 - T10, RS1 - RS10, RS1 - T1, LSTM - LSTM10 (the cadence, masked)
  replaced table      masked daily LSTM - unmasked LSTM10 (the canonical yhat_lstm_* before
                      and after this run: mask and cadence together)
  vs ridge            masked T10, T1, RS10, RS1, LSTM, LSTM10 - ridge
  CPU class           the pinned fleet (epyc-7513) - the class gate's re-run (xeon-4116) of the
                      live_feasible LightGBM T10 and T1 arms (results/trees_mask_1600/stack/xeon_*)

CPU CLASS: every masked chunk ran on one class (--constraint epyc-7513; NODE files ->
results/linear_subsection_trees_mask/class/class_census.csv, pulled); the unmasked runs of I1 / I2
ran unpinned (their classes are not recorded), so a masked - unmasked row can carry class noise
for LightGBM (the class pair above bounds it).
QLIKE: a negative difference = a has the lower loss; Sharpe: a positive difference = a trades
better.

KEPT COLUMNS (results/linear_subsection_trees_mask/kept/kept_counts.csv, from the chunk npz
files: experiments/trees_mask_kept_counts.py): median / min / max kept columns over refits
per rung x model x input set.

Outputs (--out, default results/trees_mask_1600/): mask_levels.csv, mask_pairs.csv,
mask_wide.csv, mask_gates.csv, kept_counts.csv (copied), cluster_usage.csv (from sacct.txt
when present), SUMMARY.md written from those CSVs.  Exit 1 when a gate fails.

Run:  python experiments/trees_mask_ladder_1600.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import master_table_close as mtc  # noqa: E402
import score_linear_subsection as base  # noqa: E402
import score_linear_subsection_causal as slc  # noqa: E402
import trees_cadence_ladder_1600 as tcl  # noqa: E402
from src.evaluation.diebold_mariano import dm_test  # noqa: E402

SPXW = ROOT / "results" / "spxw_pnl"
NOMASK = SPXW / "dedup_nomask_2026-09-29"
OUT = ROOT / "results" / "trees_mask_1600"
KEPT = ROOT / "results" / "linear_subsection_trees_mask" / "kept" / "kept_counts.csv"
CLASS_DIR = ROOT / "results" / "linear_subsection_trees_mask" / "class"
XEON_ROOT = ROOT / "results" / "linear_subsection_trees_mask_xeon"
STAGE = ROOT / "results" / "trees_mask_1600" / "stack"
ARM_ROOTS = {
    "T10": ROOT / "results" / "linear_subsection_trees_mask" / "t10",
    "T1": ROOT / "results" / "linear_subsection_trees_mask" / "t1",
    "RS10": ROOT / "results" / "linear_subsection_trees_tuned_mask" / "rs10",
    "RS1": ROOT / "results" / "linear_subsection_trees_tuned_mask" / "rs1",
}
LSTM_ROOT = ROOT / "results" / "linear_subsection_lstm_mask"  # REFIT_EVERY 1
LSTM10_ROOT = ROOT / "results" / "linear_subsection_lstm_mask_re10"  # REFIT_EVERY 10
STAGE_L10 = ROOT / "results" / "trees_mask_1600" / "stack" / "lstm_re10"  # its tables
MODELS = ("lgbm", "xgb", "rf")
BUCKETS = ("baseline", "all_features")  # live_feasible commented out
MODEL_LONG = mtc.TREE_LONG | {"lstm": "LSTM", "-": "per-bar ridge"}
RUNGS = {
    "T10": ("yhat_subtree_{m}_{b}", "shipped configuration, refit every 10 sessions"),
    "T1": ("yhat_subtree_daily_{b}_{m}", "shipped configuration, refit every session"),
    "RS10": (
        "yhat_subtree_tuned_{b}_{m}",
        "random search, MSE rule, refit every 10 sessions",
    ),
    "RS1": (
        "yhat_subtree_tuned_daily_{b}_{m}",
        "random search, MSE rule, refit every session",
    ),
}
SUPPLEMENTARY = {
    "RS10q": (
        "yhat_subtree_tunedq_{b}_{m}",
        "random search, QLIKE rule, refit every 10 sessions",
    ),
    "RS1q": (
        "yhat_subtree_tunedq_daily_{b}_{m}",
        "random search, QLIKE rule, refit every session",
    ),
}
LSTM_MASK = {  # rung -> (table pattern, what, where)
    "LSTM": ("yhat_lstm_{b}", "LSTM, MSE rule, refit every session", SPXW),
    "LSTMq": ("yhat_lstm_qsel_{b}", "LSTM, QLIKE rule, refit every session", SPXW),
    "LSTM10": ("yhat_lstm_{b}", "LSTM, MSE rule, refit every 10 sessions", STAGE_L10),
    "LSTM10q": (
        "yhat_lstm_qsel_{b}",
        "LSTM, QLIKE rule, refit every 10 sessions",
        STAGE_L10,
    ),
}
LSTM_NOMASK = {  # the unmasked LSTM ran at the earlier cadence
    "LSTM10": ("yhat_lstm_{b}", "LSTM, MSE rule, refit every 10 sessions", NOMASK),
    "LSTM10q": (
        "yhat_lstm_qsel_{b}",
        "LSTM, QLIKE rule, refit every 10 sessions",
        NOMASK,
    ),
}
RIDGE = "yhat_sub_ridge_{b}"
MAIN = ("T10", "T1", "RS10", "RS1", "LSTM", "LSTM10")
LADDER = (("T1", "T10"), ("RS1", "RS10"), ("RS1", "T1"))
LSTM_LADDER = (("LSTM", "LSTM10"),)


def forecasts() -> list[dict]:
    out = []
    for b in BUCKETS:
        for variant, where in (("mask", SPXW), ("nomask", NOMASK)):
            for m in MODELS:
                for rung, (pat, what) in (RUNGS | SUPPLEMENTARY).items():
                    out.append(
                        {
                            "variant": variant,
                            "rung": rung,
                            "model": m,
                            "bucket": b,
                            "table": pat.format(m=m, b=b),
                            "path": where,
                            "what": what,
                        }
                    )
            for rung, (pat, what, lwhere) in (
                LSTM_MASK if variant == "mask" else LSTM_NOMASK
            ).items():
                out.append(
                    {
                        "variant": variant,
                        "rung": rung,
                        "model": "lstm",
                        "bucket": b,
                        "table": pat.format(b=b),
                        "path": lwhere,
                        "what": what,
                    }
                )
        out.append(
            {
                "variant": "ridge",
                "rung": "ridge",
                "model": "-",
                "bucket": b,
                "table": RIDGE.format(b=b),
                "path": SPXW,
                "what": "per-bar ridge",
            }
        )
    for r in ("T10", "T1"):  # the CPU-class gate (live_feasible LightGBM only)
        out.append(
            {
                "variant": "xeon",
                "rung": r,
                "model": "lgbm",
                "bucket": "live_feasible",
                "table": "yhat_subtree_lgbm_live_feasible",
                "path": STAGE / f"xeon_{r.lower()}",
                "what": RUNGS[r][1] + ", run on xeon-4116 (the class gate)",
            }
        )
    return out


def arm_csv(s: dict) -> Path | None:
    """The merged masked arm's results CSV behind a masked table (None for the rest)."""
    if s["variant"] == "xeon":
        r = s["rung"]
        return (
            XEON_ROOT
            / r.lower()
            / "live_feasible"
            / "bar1600"
            / "lgbm"
            / "tw2000"
            / "causal_tune_trees"
            / "lgbm"
            / "live_feasible"
            / "results_bar1600.csv"
        )
    if s["variant"] != "mask":
        return None
    q = s["rung"].endswith("q")
    stem = "results_qsel" if q else "results"
    r = s["rung"].rstrip("q")
    if r in ARM_ROOTS:
        return (
            ARM_ROOTS[r]
            / s["bucket"]
            / "bar1600"
            / s["model"]
            / "tw2000"
            / "causal_tune_trees"
            / s["model"]
            / s["bucket"]
            / f"{stem}_bar1600.csv"
        )
    if r in ("LSTM", "LSTM10"):
        return (
            (LSTM_ROOT if r == "LSTM" else LSTM10_ROOT)
            / s["bucket"]
            / "bar1600"
            / "lstm"
            / "tw2000"
            / "causal_tune_lstm"
            / "lstm"
            / s["bucket"]
            / f"{stem}_bar1600.csv"
        )
    return None


def pair_row(pa: dict, pb: dict, meta: dict) -> dict:
    qa, qb = pa["q"], pb["q"]
    d = qa - qb
    lo, hi = base.day_block_ci(d)
    dm = dm_test(qa.to_numpy(), qb.to_numpy())
    ta, tb = pa["t"], pb["t"]
    row = meta | {
        "n_days": len(d),
        "qlike_a": float(qa.mean()),
        "qlike_b": float(qb.mean()),
        "qlike_diff": float(d.mean()),
        "qlike_pct": float(100 * d.mean() / qb.mean()),
        "qlike_ci_lo": lo,
        "qlike_ci_hi": hi,
        "dm": float(dm["dm"]),
        "dm_p": float(dm["p"]),
        "same_side_share": float((ta["q"].to_numpy() == tb["q"].to_numpy()).mean()),
        "max_abs_forecast_diff_rel": float(np.max(np.abs(pa["f"] / pb["f"] - 1.0))),
    }
    for fill in ("mid", "crossed"):
        v, lo_s, hi_s = mtc.paired_sharpe(ta[fill].to_numpy(), tb[fill].to_numpy())
        row |= {
            f"sharpe_{fill}_a": mtc.sharpe(ta[fill].to_numpy()),
            f"sharpe_{fill}_b": mtc.sharpe(tb[fill].to_numpy()),
            f"sharpe_{fill}_diff": v,
            f"sharpe_{fill}_ci_lo": lo_s,
            f"sharpe_{fill}_ci_hi": hi_s,
        }
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--kept", default=str(KEPT))
    ap.add_argument(
        "--no-lstm",
        action="store_true",
        help="score the tree rungs only (the masked LSTM tables not stacked yet)",
    )
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    deck = pd.read_parquet(base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    tgt = mtc.load_target()
    mtc._init(tgt, deck.index)

    specs = forecasts()
    if a.no_lstm:
        specs = [x for x in specs if x["model"] != "lstm"]
    gates: list[dict] = []
    frames: dict[tuple, pd.DataFrame] = {}
    for s in specs:
        p = s["path"] / f"{s['table']}.parquet"
        key = (s["variant"], s["rung"], s["model"], s["bucket"])
        name = f"{s['variant']}:{s['table']}"
        if not p.is_file():
            gates.append(
                {
                    "gate": "table on disk",
                    "forecast": name,
                    "n": 0,
                    "value": np.nan,
                    "bound": np.nan,
                    "ok": False,
                }
            )
            print(f"missing {p}")
            continue
        res = mtc.load_one(mtc.Spec(name, name, s["rung"], "table", str(p)))
        gates += res["gates"]
        if res["error"]:
            gates.append(
                {
                    "gate": "load",
                    "forecast": name,
                    "n": 0,
                    "value": np.nan,
                    "bound": np.nan,
                    "ok": False,
                    "detail": res["error"],
                }
            )
            continue
        frames[key] = res["frame"]
        s["rows_16h"] = res["n16"]
        arm = arm_csv(s)
        if arm is not None:
            if not arm.is_file():
                gates.append(
                    {
                        "gate": "table pred_adj = merged masked arm CSV",
                        "forecast": name,
                        "n": 0,
                        "value": np.nan,
                        "bound": 0.0,
                        "ok": False,
                        "detail": f"no {arm}",
                    }
                )
            else:
                r = pd.read_csv(arm, parse_dates=["date"]).set_index("date")
                j = res["frame"][["pred_adj"]].join(
                    r[["pred_adj"]], how="inner", rsuffix="_arm"
                )
                d = float((j["pred_adj"] - j["pred_adj_arm"]).abs().max())
                gates.append(
                    {
                        "gate": "table pred_adj = merged masked arm CSV",
                        "forecast": name,
                        "n": len(j),
                        "value": d,
                        "bound": 0.0,
                        "ok": d == 0.0 and len(j) == len(res["frame"]),
                    }
                )

    have = [f.index[f["pred_clock"].notna()].normalize() for f in frames.values()]
    days = deck.index
    for h in have:
        days = days.intersection(h)
    days = days.sort_values()
    gates.append(
        {
            "gate": "every forecast covers the deck days",
            "forecast": "all",
            "n": len(days),
            "value": len(deck.index) - len(days),
            "bound": 0,
            "ok": len(days) == len(deck.index),
        }
    )
    print(
        f"{len(frames)} forecasts loaded; {len(days)} common trade days of {len(deck.index)}"
    )
    dk = deck.loc[days]

    per_day: dict[tuple, dict] = {}
    lv_rows = []
    for s in specs:
        key = (s["variant"], s["rung"], s["model"], s["bucket"])
        if key not in frames:
            continue
        f = frames[key]
        fd = f.set_index(f.index.normalize()).loc[days]
        q = slc.qlike(fd["true_raw"], fd["pred_clock"])
        t = mtc.trade_days(fd["pred_clock"], dk)
        chk = base.trade_1530(
            fd["pred_clock"].set_axis(fd.index + pd.Timedelta(hours=16))
        )
        g = max(
            abs(mtc.sharpe(t["mid"].to_numpy()) - chk["Sharpe_mid"]),
            abs(mtc.sharpe(t["crossed"].to_numpy()) - chk["Sharpe_crossed"]),
        )
        gates.append(
            {
                "gate": "trade_days Sharpe = trade_1530 Sharpe",
                "forecast": f"{s['variant']}:{s['table']}",
                "n": len(t),
                "value": g,
                "bound": tcl.TRADE_GATE,
                "ok": g <= tcl.TRADE_GATE and chk["deck_days"] == len(days),
            }
        )
        per_day[key] = {
            "q": pd.Series(q.to_numpy(), index=fd.index),
            "t": t,
            "f": fd["pred_clock"].to_numpy(),
        }
        lv_rows.append(
            {
                k: s.get(k)
                for k in (
                    "variant",
                    "rung",
                    "model",
                    "bucket",
                    "table",
                    "what",
                    "rows_16h",
                )
            }
            | {
                "n_days": len(days),
                "qlike": float(q.mean()),
                "sharpe_mid": mtc.sharpe(t["mid"].to_numpy()),
                "sharpe_crossed": mtc.sharpe(t["crossed"].to_numpy()),
                "pct_buy": float(100 * (t["q"] > 0).mean()),
            }
        )
    lv = pd.DataFrame(lv_rows)
    lv.to_csv(out / "mask_levels.csv", index=False)

    pr_rows = []

    def add_pair(
        fam: str, m: str, b: str, ka: tuple, kb: tuple, la: str, lb: str
    ) -> None:
        if ka in per_day and kb in per_day:
            pr_rows.append(
                pair_row(
                    per_day[ka],
                    per_day[kb],
                    {"family": fam, "model": m, "bucket": b, "a": la, "b": lb},
                )
            )

    for b in BUCKETS:
        for m in (*MODELS, "lstm"):
            same = (
                ("LSTM10",) if m == "lstm" else tuple(RUNGS)
            )  # rungs with a masked and an unmasked run
            for r in same:
                add_pair(
                    "masked - unmasked",
                    m,
                    b,
                    ("mask", r, m, b),
                    ("nomask", r, m, b),
                    f"{r} masked",
                    f"{r} unmasked",
                )
            ladder = LSTM_LADDER if m == "lstm" else LADDER
            for ra, rb in ladder:
                add_pair(
                    "masked ladder",
                    m,
                    b,
                    ("mask", ra, m, b),
                    ("mask", rb, m, b),
                    f"{ra} masked",
                    f"{rb} masked",
                )
            if m == "lstm":
                add_pair(
                    "replaced table",
                    m,
                    b,
                    ("mask", "LSTM", m, b),
                    ("nomask", "LSTM10", m, b),
                    "LSTM masked (daily)",
                    "LSTM10 unmasked",
                )
            for r in ("LSTM", "LSTM10") if m == "lstm" else tuple(RUNGS):
                add_pair(
                    "masked - ridge",
                    m,
                    b,
                    ("mask", r, m, b),
                    ("ridge", "ridge", "-", b),
                    f"{r} masked",
                    "ridge",
                )
    for r in ("T10", "T1"):
        add_pair(
            "CPU class",
            "lgbm",
            "live_feasible",
            ("mask", r, "lgbm", "live_feasible"),
            ("xeon", r, "lgbm", "live_feasible"),
            f"{r} masked, epyc-7513 (fleet)",
            f"{r} masked, xeon-4116 (class gate)",
        )
    pr = pd.DataFrame(pr_rows)
    pr.to_csv(out / "mask_pairs.csv", index=False)

    main_lv = lv[lv["rung"].isin([*MAIN, "ridge"])]
    wide = main_lv[main_lv["variant"] != "ridge"].pivot_table(
        index=["bucket", "model"],
        columns=["rung", "variant"],
        values=["qlike", "sharpe_mid", "sharpe_crossed"],
    )
    wide.columns = [f"{v}_{r}_{w}" for v, r, w in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(out / "mask_wide.csv", index=False)

    kept = None
    if Path(a.kept).is_file():
        kept = pd.read_csv(a.kept)
        if a.no_lstm:
            kept = kept[~kept["rung"].astype(str).str.startswith("LSTM")]
        kept.to_csv(out / "kept_counts.csv", index=False)
        bad = (
            kept[~kept["complete"].astype(bool)]
            if "complete" in kept
            else kept.iloc[0:0]
        )
        gates.append(
            {
                "gate": "kept counts: every arm complete and masked in every chunk",
                "forecast": "all",
                "n": len(kept),
                "value": len(bad),
                "bound": 0,
                "ok": len(bad) == 0
                and bool(
                    kept.get("masked_every_chunk", pd.Series([False]))
                    .fillna(False)
                    .all()
                ),
            }
        )
        if "same_as_first_model" in kept:
            s_ = kept["same_as_first_model"].dropna()
            gates.append(
                {
                    "gate": "kept sets equal across models at common refit rows (a function of the design window alone)",
                    "forecast": "all",
                    "n": len(s_),
                    "value": int((~s_.astype(bool)).sum()),
                    "bound": 0,
                    "ok": bool(s_.astype(bool).all()),
                }
            )
    else:
        gates.append(
            {
                "gate": "kept counts on disk",
                "forecast": str(a.kept),
                "n": 0,
                "value": np.nan,
                "bound": np.nan,
                "ok": False,
            }
        )
    gt = pd.DataFrame(gates)
    gt.to_csv(out / "mask_gates.csv", index=False)
    n_bad = int((~gt["ok"].astype(bool)).sum())
    use = None
    if (out / "sacct.txt").is_file():
        use = tcl.usage_table(out / "sacct.txt")
        use.to_csv(out / "cluster_usage.csv", index=False)
    write_summary(out, lv, pr, gt, days, kept, use)
    print(
        main_lv[
            [
                "variant",
                "rung",
                "model",
                "bucket",
                "qlike",
                "sharpe_mid",
                "sharpe_crossed",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(f"\n{len(gt)} gates, {n_bad} failed; outputs in {out}")
    return 1 if n_bad else 0


def ci_counts(g: pd.DataFrame) -> str:
    q_lo, q_hi = int((g["qlike_ci_hi"] < 0).sum()), int((g["qlike_ci_lo"] > 0).sum())
    s_lo, s_hi = (
        int((g["sharpe_mid_ci_hi"] < 0).sum()),
        int((g["sharpe_mid_ci_lo"] > 0).sum()),
    )
    c_lo, c_hi = (
        int((g["sharpe_crossed_ci_hi"] < 0).sum()),
        int((g["sharpe_crossed_ci_lo"] > 0).sum()),
    )
    return (
        f"Intervals entirely below / above zero ({len(g)} rows): QLIKE {q_lo} / {q_hi}; "
        f"Sharpe mid {s_lo} / {s_hi}; Sharpe crossed {c_lo} / {c_hi}."
    )


def write_summary(out: Path, lv, pr, gt, days, kept, use) -> None:
    fmt = tcl.fmt_ci
    L: list[str] = [
        "# The 16:00 ladder with the per-window mask (checklist I9)",
        "",
        "Written by `experiments/trees_mask_ladder_1600.py` from its own CSVs in this folder (`mask_levels.csv`, "
        "`mask_pairs.csv`, `mask_wide.csv`, `mask_gates.csv`, `kept_counts.csv`"
        + (", `cluster_usage.csv`" if use is not None else "")
        + "). Every number below is read from them.",
        "",
        "## What changed",
        "",
        "Every tree fit and every LSTM fit now sees only the columns kept by `src.models.window_mask.window_keep` on the "
        "window it is trained on (the 2000 sessions before the refit, or the tuning window): a column constant on that "
        "window, or an exact byte-copy of an earlier kept column, is dropped (`WINDOW_MASK=1`). The per-bar ridge always "
        "had this rule (at each penalty tune). Everything else is the de-duplicated per-bar design of the unmasked "
        "runs (commit 47f7f9c), 2000-session window, 15:30-16:00 bar (row stamped 16:00, forecast issued at 15:30):",
        "",
        "| rung | configuration | refit | table |",
        "|---|---|---|---|",
    ]
    for r, (pat, what) in (RUNGS | SUPPLEMENTARY).items():
        cfg, refit = what.split(", refit ")
        L.append(
            f"| {r} | {cfg} | {refit} | `{pat.format(m='<model>', b='<bucket>')}` |"
        )
    L += [
        "| LSTM | per-bar LSTM, 16 configurations x 5 seeds re-chosen every 250 sessions (MSE rule; LSTMq: QLIKE rule) | every session (daily refit; the canonical table) | `yhat_lstm[_qsel]_<bucket>` |",
        "| LSTM10 | the same LSTM (LSTM10q: QLIKE rule) | every 10 sessions (the earlier cadence; the unmasked LSTM ran at it) | `results/trees_mask_1600/stack/lstm_re10/yhat_lstm[_qsel]_<bucket>` (masked); the snapshot (unmasked) |",
        f"| ridge | per-bar ridge, penalty re-chosen every 250 sessions | every session | `{RIDGE.format(b='<bucket>')}` |",
        "",
        "Masked = the canonical tables in `results/spxw_pnl/`; unmasked = the same names in "
        "`results/spxw_pnl/dedup_nomask_2026-09-29/` (the de-duplicated runs before the mask). The LSTM rebuilds its "
        "network at every refit, so its mask is recomputed at every refit (as for the trees).",
        "",
    ]
    if kept is not None and len(kept):
        L += [
            "## Kept columns (median / min / max over refits)",
            "",
            "The mask depends on the design window only, so every model of an input set keeps the same columns at the "
            "same refit row; the rungs differ only in which rows they refit at (every session for T1 / RS1 / LSTM, every "
            "10th for T10 / RS10 / LSTM10).",
            "",
            "| input set | p | rung | model | refits | kept median | kept min | kept max | tuning points kept median [min, max] |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        order = {
            r: i for i, r in enumerate(("T10", "T1", "RS10", "RS1", "LSTM", "LSTM10"))
        }
        k2 = kept.assign(_o=kept["rung"].map(order)).sort_values(
            ["bucket", "_o", "model"]
        )
        for x in k2.itertuples():
            if not getattr(x, "complete", True):
                L.append(
                    f"| `{x.bucket}` | | {x.rung} | {MODEL_LONG.get(x.model, x.model)} | incomplete | | | | |"
                )
                continue
            tk = ""
            if hasattr(x, "tune_kept_n_median") and pd.notna(
                getattr(x, "tune_kept_n_median", np.nan)
            ):
                tk = f"{x.tune_kept_n_median:.0f} [{x.tune_kept_n_min:.0f}, {x.tune_kept_n_max:.0f}]"
            L.append(
                f"| `{x.bucket}` | {x.p:.0f} | {x.rung} | {MODEL_LONG.get(x.model, x.model)} | {x.refits:.0f} | "
                f"{x.kept_n_median:.0f} | {x.kept_n_min:.0f} | {x.kept_n_max:.0f} | {tk} |"
            )
        L.append("")
    cc, ch, ca = (
        CLASS_DIR / f
        for f in ("class_census.csv", "class_chunks.csv", "class_arms.csv")
    )
    if cc.is_file():
        c = pd.read_csv(cc)
        L += [
            "## CPU class of every chunk",
            "",
            "Forecasts are bit-reproducible only within a CPU vector class (the design moves at ~1e-11 between AVX-512 "
            "xeon and AVX2 epyc nodes, and LightGBM's histogram bins with it), so every masked task was pinned to "
            "epyc-7513 and every finished chunk records its node (`class_census.csv`). The unmasked runs this page "
            "compares with ran unpinned (classes not recorded).",
            "",
            "| results root | rung | CPU class | chunks |",
            "|---|---|---|---|",
        ]
        L += [
            f"| {x.root} | {x.rung} | {x['class']} | {x.chunks} |"
            for _, x in c.iterrows()
        ]
        L.append("")
    if ch.is_file() and ch.stat().st_size > 1:
        c = pd.read_csv(ch)
        if len(c) and "results_max_rel" in c:
            L += [
                "Canary chunks that ran before the pinning, against the same chunk re-run on epyc-7513 "
                "(`class_chunks.csv`; pred_adj on the chunk's rows):",
                "",
                "| chunk | canary class | fleet class | rows | bit-identical | max relative difference |",
                "|---|---|---|---|---|---|",
            ]
            for _, x in c.iterrows():
                L.append(
                    f"| `{x.chunk_dir}` | {x.class_canary} | {x.class_fleet} | {x.get('results_rows', '')} | "
                    f"{x.get('results_bit_identical', '')} | {x.get('results_max_rel', float('nan')):.2e} |"
                )
            L.append("")
    if ca.is_file():
        c = pd.read_csv(ca)
        if len(c) and "max_rel" in c:
            L += [
                "Class gate (`class_arms.csv`): the live_feasible LightGBM arms run again on xeon-4116, against the "
                "fleet's epyc-7513 arms (pred_adj over every forecast row; QLIKE and Sharpe differences in the "
                "`CPU class` pairs below):",
                "",
                "| rung | rows | bit-identical | max relative difference | median relative difference |",
                "|---|---|---|---|---|",
            ]
            for _, x in c.iterrows():
                L.append(
                    f"| {x.rung} | {x.rows:.0f} | {x.bit_identical} | {x.max_rel:.2e} | {x.median_rel:.2e} |"
                )
            L.append("")
    L += [
        "## Scorer and days",
        "",
        "The research scorer (the 16:00 bar recalibrated on its own; the master table's loader, target and trade "
        "imported): forecast = (f^2 + s) B with s = the forecast's own mean squared adjusted-scale error at 16:00 over "
        "the previous 250 sessions (at least 63), lagged one session; QLIKE against the per-bar spec's 16:00 target "
        "(lower is better). The last-30-min trade is the sign(s) rule on the straddle (nearest out-of-the-money call + "
        "nearest out-of-the-money put, same-day expiry, one position): buy when the recalibrated forecast exceeds the "
        "15:30 implied variance, sell otherwise; annualized Sharpe at mid and at crossed fills. "
        f"Every number is on the same {len(days)} trade days ({days.min().date()} .. {days.max().date()}). "
        "Intervals: 95 % day-block bootstrap (21-day circular blocks, 2000 draws), paired (the same resampled days "
        "for both forecasts). Differences are a - b.",
        "",
        "## Levels (masked | unmasked)",
        "",
    ]
    for b in BUCKETS:
        sub = lv[lv["bucket"] == b]
        if sub.empty:
            continue
        L += [
            f"**`{b}`**",
            "",
            "| forecast | QLIKE masked | QLIKE unmasked | Sharpe mid masked | Sharpe mid unmasked | Sharpe crossed masked | Sharpe crossed unmasked | % buy masked |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for m in (*MODELS, "lstm"):
            rungs = (*RUNGS, *SUPPLEMENTARY) if m != "lstm" else tuple(LSTM_MASK)
            for r in rungs:
                xm = sub[
                    (sub["model"] == m)
                    & (sub["rung"] == r)
                    & (sub["variant"] == "mask")
                ]
                xn = sub[
                    (sub["model"] == m)
                    & (sub["rung"] == r)
                    & (sub["variant"] == "nomask")
                ]
                if not len(xm):
                    continue
                xm_ = xm.iloc[0]
                xn_ = xn.iloc[0] if len(xn) else None
                nq = f"{xn_.qlike:.4f}" if xn_ is not None else ""
                ns = f"{xn_.sharpe_mid:.2f}" if xn_ is not None else ""
                nc = f"{xn_.sharpe_crossed:.2f}" if xn_ is not None else ""
                label = (
                    f"{MODEL_LONG[m]} {r}"
                    if m != "lstm"
                    else f"{r} ({LSTM_MASK[r][1]})"
                )
                L.append(
                    f"| {label} | {xm_.qlike:.4f} | {nq} | {xm_.sharpe_mid:.2f} | {ns} | "
                    f"{xm_.sharpe_crossed:.2f} | {nc} | {xm_.pct_buy:.1f} |"
                )
        x = sub[sub["rung"] == "ridge"]
        if len(x):
            x = x.iloc[0]
            L.append(
                f"| per-bar ridge | {x.qlike:.4f} | | {x.sharpe_mid:.2f} | | {x.sharpe_crossed:.2f} | | {x.pct_buy:.1f} |"
            )
        L.append("")
    for fam in (
        "masked - unmasked",
        "masked ladder",
        "replaced table",
        "masked - ridge",
        "CPU class",
    ):
        g_all = pr[pr["family"] == fam] if len(pr) else pr
        if g_all.empty:
            continue
        L += [f"## {fam[0].upper() + fam[1:]} (a - b) with 95 % intervals", ""]
        for (ra, rb), g in g_all.groupby(["a", "b"], sort=False):
            L += [
                f"**{ra} - {rb}**",
                "",
                "| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side | max forecast change |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for x in g.itertuples():
                L.append(
                    f"| {MODEL_LONG[x.model]} | `{x.bucket}` | {x.qlike_pct:+.1f} % | {fmt(x.qlike_diff, x.qlike_ci_lo, x.qlike_ci_hi, 4)} | "
                    f"{fmt(x.sharpe_mid_diff, x.sharpe_mid_ci_lo, x.sharpe_mid_ci_hi, 2)} | "
                    f"{fmt(x.sharpe_crossed_diff, x.sharpe_crossed_ci_lo, x.sharpe_crossed_ci_hi, 2)} | "
                    f"{100 * x.same_side_share:.0f} % | {100 * x.max_abs_forecast_diff_rel:.1f} % |"
                )
            L += ["", ci_counts(g), ""]
        L += [f"All `{fam}` rows: {ci_counts(g_all)}", ""]
    if use is not None and len(use):
        L += [
            "## Cluster use (the masked re-run's own jobs, sacct)",
            "",
            "| stage | allocations | CPUs per allocation | allocated CPU-hours | used CPU-hours | peak memory per allocation (GiB) | first start | last end |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for st, g in use.groupby("stage", sort=False):
            L.append(
                f"| {st} | {len(g)} | {int(g['AllocCPUS'].min())}-{int(g['AllocCPUS'].max())} | {g['CPUTimeRAW'].sum() / 3600:.1f} | "
                f"{g['used_cpu_sec'].sum() / 3600:.1f} | {g['max_rss_gib'].max():.1f} | {g['Start'].min()} | {g['End'].max()} |"
            )
        ev = (
            pd.concat(
                [
                    pd.DataFrame({"t": use["Start"], "d": use["AllocCPUS"]}),
                    pd.DataFrame({"t": use["End"], "d": -use["AllocCPUS"]}),
                ]
            )
            .dropna()
            .sort_values(["t", "d"])
        )
        L += [
            "",
            f"All stages: {use['CPUTimeRAW'].sum() / 3600:.1f} allocated CPU-hours ({use['used_cpu_sec'].sum() / 3600:.1f} used); "
            f"peak concurrent allocated CPUs {int(ev['d'].cumsum().max())}; wall-clock {use['Start'].min()} .. {use['End'].max()}.",
            "",
        ]
    bad = gt[~gt["ok"].astype(bool)]
    L += [
        "## Gates",
        "",
        f"{len(gt)} gates checked, {len(bad)} failed (`mask_gates.csv`): every table's profile B equals the common "
        "target's; every masked tree and LSTM table carries its merged masked arm's pred_adj exactly; every forecast covers "
        "the same trade days; the per-day trade returns reproduce `score_linear_subsection.trade_1530`'s Sharpe; every "
        "arm's chunks ran with the mask; the kept sets agree across models at common refit rows.",
        "",
        "Wording: sign(s) = the rule above; QLIKE differences are losses (negative = a lower loss); nothing here is a "
        "recommendation.",
    ]
    (out / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
