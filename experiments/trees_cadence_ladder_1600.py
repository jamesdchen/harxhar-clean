"""The 16:00 tree cadence ladder (checklist I2): refit every 10 sessions vs every session,
shipped configuration vs random search, beside the per-bar ridge.

WHAT IS SCORED.  Per model (LightGBM, XGBoost, random forest) x input set (baseline =
HAR + calendar, live_feasible, all_features), the 16:00 forecast (the 15:30-16:00 bar,
issued at 15:30) of four rungs fitted on the de-duplicated per-bar design (commit
47f7f9c) with a 2000-session window:

  T10   shipped configuration, refit every 10 sessions   yhat_subtree_<model>_<bucket>
  T1    shipped configuration, refit every session        yhat_subtree_daily_<bucket>_<model>
  RS10  random search (32 fixed candidates, chosen every 250 sessions on a causal
        validation tail by validation MSE), refit every 10  yhat_subtree_tuned_<bucket>_<model>
  RS1   the same, refit every session                     yhat_subtree_tuned_daily_<bucket>_<model>
  ridge the per-bar ridge of the same input set           yhat_sub_ridge_<bucket>

(the QLIKE-rule twins of RS10 / RS1, yhat_subtree_tunedq[_daily]_*, are scored as
supplementary levels).  The ridge table is labelled "dedup" when it differs from the
pre-change snapshot results/spxw_pnl/pre_dedup_2026-09-29/ (rebuilt on the new design)
and "pre-dedup" when it is byte-identical to it.

ONE SCORER, the research scorer (16:00 bar recalibrated alone), imported, not re-written:
every table is loaded by experiments/master_table_close.py's load_one (its common 16:00
target TARGET_ARM, the per-bar spec's target; the recalibration
score_linear_subsection_causal.causal_forecasts: forecast = (f^2 + s) B with s the
forecast's own mean squared adjusted-scale error at 16:00 over the previous 250 sessions,
>= 63, lagged one session); QLIKE = score_linear_subsection_causal.qlike against that
target; the last-30-min trade = the sign(s) rule on the straddle (buy when the
recalibrated forecast exceeds the 15:30 implied variance, sell otherwise), mid and
crossed fills, per-day returns by master_table_close.trade_days (=
score_linear_subsection.trade_1530 line for line; gated here against trade_1530's
Sharpe); intervals: day-block bootstrap (score_linear_subsection.day_block_ci: 21-day
circular blocks, 2000 draws) for QLIKE differences, the same resampled days for Sharpe
differences (master_table_close.paired_sharpe); DM = src.evaluation.diebold_mariano.
SAME DAYS for every number: the deck's trade days on which every forecast of the ladder
has a recalibrated 16:00 forecast (866 when all are complete).

PAIRS (a - b, per model x input set): T1 - T10, RS10 - T10, RS1 - RS10, RS1 - T1, and
each of T10, T1, RS10, RS1 - ridge.  QLIKE: lower is better (a negative difference: a has
the lower loss); Sharpe: a positive difference: a trades better.

Outputs (--out, default results/trees_cadence_1600/): ladder_levels.csv, ladder_pairs.csv,
ladder_wide.csv, ladder_gates.csv, cluster_usage.csv (from sacct.txt when present: the
campaign's own jobs, `sacct -X -P -n -o JobID,JobName,AllocCPUS,ElapsedRaw,Start,End,
State,CPUTimeRAW,TotalCPU`), and SUMMARY.md written from those CSVs.  Exit 1 when a gate
fails.

Run:  python experiments/trees_cadence_ladder_1600.py
"""

from __future__ import annotations

import argparse
import hashlib
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
from src.evaluation.diebold_mariano import dm_test  # noqa: E402

SPXW = ROOT / "results" / "spxw_pnl"
SNAPSHOT = SPXW / "pre_dedup_2026-09-29"
OUT = ROOT / "results" / "trees_cadence_1600"
ARM_ROOTS = {
    "T10": ROOT / "results" / "linear_subsection_trees_dedup" / "t10",
    "T1": ROOT / "results" / "linear_subsection_trees_dedup" / "t1",
    "RS10": ROOT / "results" / "linear_subsection_trees_tuned_dedup" / "rs10",
    "RS1": ROOT / "results" / "linear_subsection_trees_tuned_dedup" / "rs1",
}
MODELS = ("lgbm", "xgb", "rf")
BUCKETS = ("baseline", "live_feasible", "all_features")
MODEL_LONG = mtc.TREE_LONG
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
RIDGE = "yhat_sub_ridge_{b}"
PAIRS = (
    ("T1", "T10"),
    ("RS10", "T10"),
    ("RS1", "RS10"),
    ("RS1", "T1"),
    ("T10", "ridge"),
    ("T1", "ridge"),
    ("RS10", "ridge"),
    ("RS1", "ridge"),
)
TRADE_GATE = 1e-12  # trade_days vs trade_1530: the same arithmetic


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def ridge_design(tables: Path, b: str) -> str:
    cur, snap = (
        tables / f"{RIDGE.format(b=b)}.parquet",
        SNAPSHOT / f"{RIDGE.format(b=b)}.parquet",
    )
    if not snap.is_file():
        return "unknown (no snapshot)"
    return "pre-dedup" if md5(cur) == md5(snap) else "dedup"


def fmt_ci(v: float, lo: float, hi: float, nd: int) -> str:
    return f"{v:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def usage_table(sacct: Path) -> pd.DataFrame:
    """cluster_usage.csv rows from a sacct -X -P dump (one line per allocation)."""
    cols = [
        "JobID",
        "JobName",
        "AllocCPUS",
        "ElapsedRaw",
        "Start",
        "End",
        "State",
        "CPUTimeRAW",
        "TotalCPU",
    ]
    d = pd.read_csv(sacct, sep="|", header=None, names=cols, dtype=str)
    d = d[d["JobID"].str.strip().ne("")]
    d["AllocCPUS"] = d["AllocCPUS"].astype(int)
    d["CPUTimeRAW"] = d["CPUTimeRAW"].astype(float)
    d["Start"] = pd.to_datetime(d["Start"], errors="coerce")
    d["End"] = pd.to_datetime(d["End"], errors="coerce")
    d["stage"] = d["JobName"].str.replace(r"^tc_", "", regex=True)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument(
        "--tables",
        default=str(SPXW),
        help="where the yhat tables are read (default results/spxw_pnl)",
    )
    a = ap.parse_args()
    out = Path(a.out)
    tables = Path(a.tables)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    deck = pd.read_parquet(base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    tgt = mtc.load_target()
    mtc._init(tgt, deck.index)

    # ---- the forecasts
    specs: list[dict] = []
    for b in BUCKETS:
        for m in MODELS:
            for rung, (pat, what) in (RUNGS | SUPPLEMENTARY).items():
                specs.append(
                    {
                        "rung": rung,
                        "model": m,
                        "bucket": b,
                        "table": pat.format(m=m, b=b),
                        "what": what,
                    }
                )
        specs.append(
            {
                "rung": "ridge",
                "model": "-",
                "bucket": b,
                "table": RIDGE.format(b=b),
                "what": f"per-bar ridge ({ridge_design(tables, b)} design)",
            }
        )
    gates: list[dict] = []
    frames: dict[tuple, pd.DataFrame] = {}
    for s in specs:
        p = tables / f"{s['table']}.parquet"
        key = (s["rung"], s["model"], s["bucket"])
        if not p.is_file():
            gates.append(
                {
                    "gate": "table on disk",
                    "forecast": s["table"],
                    "n": 0,
                    "value": np.nan,
                    "bound": np.nan,
                    "ok": s["rung"] in SUPPLEMENTARY,
                }
            )
            print(f"missing {p.name}")
            continue
        res = mtc.load_one(mtc.Spec(s["table"], s["table"], s["rung"], "table", str(p)))
        gates += res["gates"]
        if res["error"]:
            gates.append(
                {
                    "gate": "load",
                    "forecast": s["table"],
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
        # the stacked table carries the merged arm's pred_adj (tree rungs, when the arm CSV is local)
        root = ARM_ROOTS.get(s["rung"].rstrip("q"))
        if root is not None and s["model"] != "-":
            stem = "results_qsel" if s["rung"].endswith("q") else "results"
            arm = (
                root
                / s["bucket"]
                / "bar1600"
                / s["model"]
                / "tw2000"
                / "causal_tune_trees"
                / s["model"]
                / s["bucket"]
                / f"{stem}_bar1600.csv"
            )
            if arm.is_file():
                r = pd.read_csv(arm, parse_dates=["date"]).set_index("date")
                j = res["frame"][["pred_adj"]].join(
                    r[["pred_adj"]], how="inner", rsuffix="_arm"
                )
                d = float((j["pred_adj"] - j["pred_adj_arm"]).abs().max())
                gates.append(
                    {
                        "gate": "table pred_adj = merged arm CSV",
                        "forecast": s["table"],
                        "n": len(j),
                        "value": d,
                        "bound": 0.0,
                        "ok": d == 0.0 and len(j) == len(res["frame"]),
                    }
                )

    # ---- the same days for every forecast
    have = [f.index[f["pred_clock"].notna()].normalize() for f in frames.values()]
    days = deck.index
    for h in have:
        days = days.intersection(h)
    days = days.sort_values()
    for s in specs:
        key = (s["rung"], s["model"], s["bucket"])
        if key in frames:
            f = frames[key]
            s["deck_days_covered"] = int(
                deck.index.isin(f.index[f["pred_clock"].notna()].normalize()).sum()
            )
    gates.append(
        {
            "gate": "every ladder forecast covers the deck days",
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
        key = (s["rung"], s["model"], s["bucket"])
        if key not in frames:
            continue
        f = frames[key]
        fd = f.set_index(f.index.normalize()).loc[days]
        q = slc.qlike(fd["true_raw"], fd["pred_clock"])
        qu = slc.qlike(fd["rv_unclipped"], fd["pred_clock"])
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
                "forecast": s["table"],
                "n": len(t),
                "value": g,
                "bound": TRADE_GATE,
                "ok": g <= TRADE_GATE and chk["deck_days"] == len(days),
            }
        )
        per_day[key] = {"q": pd.Series(q.to_numpy(), index=fd.index), "t": t}
        lv_rows.append(
            {
                k: s.get(k)
                for k in (
                    "rung",
                    "model",
                    "bucket",
                    "table",
                    "what",
                    "rows_16h",
                    "deck_days_covered",
                )
            }
            | {
                "n_days": len(days),
                "qlike": float(q.mean()),
                "qlike_unclipped_rv": float(qu.mean()),
                "sharpe_mid": mtc.sharpe(t["mid"].to_numpy()),
                "sharpe_crossed": mtc.sharpe(t["crossed"].to_numpy()),
                "pct_buy": float(100 * (t["q"] > 0).mean()),
            }
        )
    lv = pd.DataFrame(lv_rows)
    lv.to_csv(out / "ladder_levels.csv", index=False)

    pr_rows = []
    for b in BUCKETS:
        for m in MODELS:
            for ra, rb in PAIRS:
                ka, kb = (ra, m, b), (rb, "-" if rb == "ridge" else m, b)
                if ka not in per_day or kb not in per_day:
                    continue
                qa, qb = per_day[ka]["q"], per_day[kb]["q"]
                d = qa - qb
                lo, hi = base.day_block_ci(d)
                dm = dm_test(qa.to_numpy(), qb.to_numpy())
                ta, tb = per_day[ka]["t"], per_day[kb]["t"]
                row = {
                    "model": m,
                    "bucket": b,
                    "a": ra,
                    "b": rb,
                    "n_days": len(d),
                    "qlike_a": float(qa.mean()),
                    "qlike_b": float(qb.mean()),
                    "qlike_diff": float(d.mean()),
                    "qlike_pct": float(100 * d.mean() / qb.mean()),
                    "qlike_ci_lo": lo,
                    "qlike_ci_hi": hi,
                    "dm": float(dm["dm"]),
                    "dm_p": float(dm["p"]),
                    "same_side_share": float(
                        (ta["q"].to_numpy() == tb["q"].to_numpy()).mean()
                    ),
                }
                for fill in ("mid", "crossed"):
                    v, lo_s, hi_s = mtc.paired_sharpe(
                        ta[fill].to_numpy(), tb[fill].to_numpy()
                    )
                    row |= {
                        f"sharpe_{fill}_a": mtc.sharpe(ta[fill].to_numpy()),
                        f"sharpe_{fill}_b": mtc.sharpe(tb[fill].to_numpy()),
                        f"sharpe_{fill}_diff": v,
                        f"sharpe_{fill}_ci_lo": lo_s,
                        f"sharpe_{fill}_ci_hi": hi_s,
                    }
                pr_rows.append(row)
    pr = pd.DataFrame(pr_rows)
    pr.to_csv(out / "ladder_pairs.csv", index=False)

    main_lv = lv[lv["rung"].isin(list(RUNGS))]
    wide = main_lv.pivot_table(
        index=["bucket", "model"],
        columns="rung",
        values=["qlike", "sharpe_mid", "sharpe_crossed"],
    )
    wide.columns = [f"{v}_{r}" for v, r in wide.columns]
    rg = lv[lv["rung"] == "ridge"].set_index("bucket")
    for col in ("qlike", "sharpe_mid", "sharpe_crossed"):
        wide[f"{col}_ridge"] = [
            rg.loc[b, col] if b in rg.index else np.nan for b, _m in wide.index
        ]
    wide = wide.reset_index()
    wide.to_csv(out / "ladder_wide.csv", index=False)
    gt = pd.DataFrame(gates)
    gt.to_csv(out / "ladder_gates.csv", index=False)
    n_bad = int((~gt["ok"].astype(bool)).sum())

    use = None
    if (out / "sacct.txt").is_file():
        use = usage_table(out / "sacct.txt")
        use.to_csv(out / "cluster_usage.csv", index=False)
    write_summary(out, lv, pr, gt, days, use)
    print(
        main_lv[["rung", "model", "bucket", "qlike", "sharpe_mid", "sharpe_crossed"]]
        .round(4)
        .to_string(index=False)
    )
    print(f"\n{len(gt)} gates, {n_bad} failed; outputs in {out}")
    return 1 if n_bad else 0


def write_summary(
    out: Path,
    lv: pd.DataFrame,
    pr: pd.DataFrame,
    gt: pd.DataFrame,
    days: pd.DatetimeIndex,
    use: pd.DataFrame | None,
) -> None:
    L: list[str] = []
    rg = lv[lv["rung"] == "ridge"]
    L += [
        "# The 16:00 tree cadence ladder (checklist I2)",
        "",
        "Written by `experiments/trees_cadence_ladder_1600.py` from its own CSVs in this folder "
        "(`ladder_levels.csv`, `ladder_pairs.csv`, `ladder_wide.csv`, `ladder_gates.csv`"
        + (", `cluster_usage.csv`" if use is not None else "")
        + "). Every number below is read from them.",
        "",
        "## What is scored",
        "",
        "Per model (LightGBM, XGBoost, random forest) and input set (`baseline` = HAR ladder `har_ma_*` + calendar; "
        "`live_feasible`; `all_features`), the forecast of the 15:30-16:00 bar's realized variance (row stamped 16:00, "
        "issued at 15:30), fitted on the per-bar design without the session-edge duplicates (commit 47f7f9c) with a "
        "2000-session window:",
        "",
        "| rung | configuration | refit | table |",
        "|---|---|---|---|",
    ]
    for r, (pat, what) in (RUNGS | SUPPLEMENTARY).items():
        cfg, refit = what.split(", refit ")
        L.append(
            f"| {r} | {cfg} | {refit} | `{pat.format(m='<model>', b='<bucket>')}` |"
        )
    L.append(
        f"| ridge | per-bar ridge, penalty re-chosen every 250 sessions | every session | `{RIDGE.format(b='<bucket>')}` |"
    )
    L += [
        "",
        "Random search = 32 fixed candidates of each model's grid, re-chosen every 250 sessions on the 125-session "
        "validation tail of the current window (25-session embargo), by validation MSE (RS10, RS1) or validation QLIKE "
        "(RS10q, RS1q; supplementary). RS1 ran without TreeSHAP (the tuned workers record native importance at every refit); "
        "T1 records native importance and TreeSHAP at every 10th refit, T10 at every refit.",
        "",
        "Per-bar ridge design used here: "
        + ", ".join(
            f"`{r.bucket}` {r.what.split('(')[1].rstrip(')')}" for r in rg.itertuples()
        )
        + ".",
        "",
        "## Scorer and days",
        "",
        "The research scorer (the 16:00 bar recalibrated on its own; the master table's loader, target and trade "
        "imported): forecast = (f^2 + s) B with s = the forecast's own mean squared adjusted-scale error at 16:00 over "
        "the previous 250 sessions (at least 63), lagged one session; QLIKE against the per-bar spec's 16:00 target "
        "(lower is better). The last-30-min trade is the sign(s) rule on the straddle (nearest out-of-the-money call + "
        "nearest out-of-the-money put, same-day expiry, one position): buy when the recalibrated forecast exceeds the "
        "15:30 implied variance, sell otherwise; annualized Sharpe at mid and at crossed fills. "
        f"Every number is on the same {len(days)} trade days ({days.min().date()} .. {days.max().date()}). "
        "Intervals: 95 % day-block bootstrap (21-day circular blocks, 2000 draws), paired (the same resampled days for "
        "both forecasts). Differences are a - b.",
        "",
        "## Levels",
        "",
    ]
    for b in BUCKETS:
        sub = lv[(lv["bucket"] == b)]
        if sub.empty:
            continue
        L += [
            f"**`{b}`**",
            "",
            "| forecast | QLIKE | Sharpe mid | Sharpe crossed | % buy |",
            "|---|---|---|---|---|",
        ]
        for m in MODELS:
            for r in (*RUNGS, *SUPPLEMENTARY):
                x = sub[(sub["model"] == m) & (sub["rung"] == r)]
                if len(x):
                    x = x.iloc[0]
                    L.append(
                        f"| {MODEL_LONG[m]} {r} | {x.qlike:.4f} | {x.sharpe_mid:.2f} | {x.sharpe_crossed:.2f} | {x.pct_buy:.1f} |"
                    )
        x = sub[sub["rung"] == "ridge"]
        if len(x):
            x = x.iloc[0]
            L.append(
                f"| per-bar ridge ({x.what.split('(')[1].rstrip(')')}) | {x.qlike:.4f} | {x.sharpe_mid:.2f} | {x.sharpe_crossed:.2f} | {x.pct_buy:.1f} |"
            )
        L.append("")
    L += ["## Paired differences (a - b) with 95 % intervals", ""]
    for (ra, rb), g in pr.groupby(["a", "b"], sort=False):
        L += [
            f"**{ra} - {rb}**",
            "",
            "| model | input set | QLIKE diff (%) | QLIKE diff [interval] | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side |",
            "|---|---|---|---|---|---|---|",
        ]
        for x in g.itertuples():
            L.append(
                f"| {MODEL_LONG[x.model]} | `{x.bucket}` | {x.qlike_pct:+.1f} % | {fmt_ci(x.qlike_diff, x.qlike_ci_lo, x.qlike_ci_hi, 4)} | "
                f"{fmt_ci(x.sharpe_mid_diff, x.sharpe_mid_ci_lo, x.sharpe_mid_ci_hi, 2)} | "
                f"{fmt_ci(x.sharpe_crossed_diff, x.sharpe_crossed_ci_lo, x.sharpe_crossed_ci_hi, 2)} | {100 * x.same_side_share:.0f} % |"
            )
        q_lo, q_hi = (
            int((g["qlike_ci_hi"] < 0).sum()),
            int((g["qlike_ci_lo"] > 0).sum()),
        )
        s_lo, s_hi = (
            int((g["sharpe_mid_ci_hi"] < 0).sum()),
            int((g["sharpe_mid_ci_lo"] > 0).sum()),
        )
        c_lo, c_hi = (
            int((g["sharpe_crossed_ci_hi"] < 0).sum()),
            int((g["sharpe_crossed_ci_lo"] > 0).sum()),
        )
        L += [
            "",
            f"Intervals entirely below / above zero ({len(g)} rows): QLIKE {q_lo} / {q_hi}; Sharpe mid {s_lo} / {s_hi}; "
            f"Sharpe crossed {c_lo} / {c_hi}.",
            "",
        ]
    if use is not None and len(use):
        L += [
            "## Cluster use (the campaign's own jobs, sacct)",
            "",
            "| stage | allocations | CPUs per allocation | allocated CPU-hours | first start | last end |",
            "|---|---|---|---|---|---|",
        ]
        for st, g in use.groupby("stage", sort=False):
            L.append(
                f"| {st} | {len(g)} | {int(g['AllocCPUS'].min())}-{int(g['AllocCPUS'].max())} | {g['CPUTimeRAW'].sum() / 3600:.1f} | "
                f"{g['Start'].min()} | {g['End'].max()} |"
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
            f"All stages: {use['CPUTimeRAW'].sum() / 3600:.1f} allocated CPU-hours; peak concurrent allocated CPUs "
            f"{int(ev['d'].cumsum().max())}; wall-clock {use['Start'].min()} .. {use['End'].max()}.",
            "",
        ]
    bad = gt[~gt["ok"].astype(bool)]
    L += [
        "## Gates",
        "",
        f"{len(gt)} gates checked, {len(bad)} failed (`ladder_gates.csv`): every table's profile B equals the "
        "common target's; every stacked tree table carries its merged arm's pred_adj exactly; every forecast covers the "
        "same trade days; the per-day trade returns reproduce `score_linear_subsection.trade_1530`'s Sharpe.",
        "",
    ]
    L += [
        "Wording: sign(s) = the rule above; QLIKE differences are losses (negative = a lower loss); nothing here is "
        "a recommendation."
    ]
    (out / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
