"""Plain-forecast scores of the dev-tuned random forest beside the default random forest, the 2016-19
pre-tuned random forest, LightGBM and XGBoost (user decision 2026-10-06).

Per input set (baseline, all_features, lean_vol_vix) and refit cadence (T1: every session, T10: every 10):
  default RF   the stored canonical table (sklearn defaults: yhat_subtree_daily_<b>_rf / yhat_subtree_rf_<b>)
  dev RF       this run (experiments/trees_rfdev_stack_1600.py: yhat_subtree_daily_<b>_rfdev / yhat_subtree_rfdev_<b>)
  pre-tuned RF the frozen_all arm of results/trees_pretune_2026-10-06/arms.csv (pre-tune on 2016-2019 sessions;
               its numbers as that study reports them, same scorer -- its own scorer gate against the master table)
  LightGBM, XGBoost  the stored canonical tables of the same cadence
Scorer: the master table's plain scorer, imported from experiments/master_table_close.py (not edited): target of
results/linear_subsection_dedup, load_one (16:00 rows, plain f^2 B, gates) and score_one, on every trade day of the
deck (866 days, 2020-01-03 .. 2024-04-30), no recalibration.  QLIKE against the per-bar 16:00 target; the
last-30-min trade q = sign(plain - iv_var) on the straddle (nearest out-of-the-money call + nearest out-of-the-money
put, same-day expiry, one position), annualized Sharpe and Sortino (writeup/master_table_gui/rows.py _sortino: mean
over the root-mean-square of min(r, 0), x sqrt(252)) at mid and crossed fills.  Pairs (dev RF - comparison):
DM statistic / p of the per-day QLIKE difference (score_one with the comparison as reference; negative = dev RF
more accurate), Newey-West t of the per-day mid and crossed P&L differences.
GATES: the stored rows' levels equal results/close_master_table/master_table.csv where the key is there (1e-12);
the stored random forest's Sortino equals the pre-tune study's stored_T1 / stored_T10 Sortino (1e-9); load_one's gates.

Writes (--out, default results/trees_rfdev_1600/): levels.csv, pairs.csv, gates.csv, SUMMARY.md (from the CSVs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (
    ROOT,
    ROOT / "notebooks",
    ROOT / "experiments",
    ROOT / "writeup" / "master_table_gui",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402
import master_table_close as mtc  # noqa: E402
import rows as gui_rows  # noqa: E402  (writeup/master_table_gui/rows.py: the Sortino the GUI shows)

ARM_ROOT = "linear_subsection_dedup"
MASTER = ROOT / "results" / "close_master_table" / "master_table.csv"
PRETUNE = ROOT / "results" / "trees_pretune_2026-10-06" / "arms.csv"
BUCKETS = ("baseline", "all_features", "lean_vol_vix")
RUNGS = {"T1": 1, "T10": 10}
KEYS = {  # (rung, forecast) -> table key pattern
    ("T1", "dev RF"): "subtree_daily_{b}_rfdev",
    ("T1", "default RF"): "subtree_daily_{b}_rf",
    ("T1", "LightGBM"): "subtree_daily_{b}_lgbm",
    ("T1", "XGBoost"): "subtree_daily_{b}_xgb",
    ("T10", "dev RF"): "subtree_rfdev_{b}",
    ("T10", "default RF"): "subtree_rf_{b}",
    ("T10", "LightGBM"): "subtree_lgbm_{b}",
    ("T10", "XGBoost"): "subtree_xgb_{b}",
}
COMPARE = ("default RF", "LightGBM", "XGBoost")
REPRO_TOL = 1e-12
SORTINO_TOL = 1e-9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "trees_rfdev_1600"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mtc.set_arm_root(ARM_ROOT)
    deck = pd.read_parquet(mtc.base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    tgt = mtc.load_target()
    mtc._init(tgt, deck.index)
    paper = mtc._paper_tags()
    ref = mtc.load_one(
        mtc.describe_table(
            next(mtc.SPXW / n for n, t in paper.items() if t == mtc.REFERENCE), paper
        )
    )["frame"]
    days = deck.index
    dk = deck.loc[days]
    master = pd.read_csv(MASTER).set_index("key")
    pre = pd.read_csv(PRETUNE)
    pre = pre[pre["model"] == "rf"]
    gates: list[dict] = []
    levels: list[dict] = []
    pairs: list[dict] = []

    def trade(f: pd.DataFrame) -> pd.DataFrame:
        return mtc.trade_days(f.set_index(f.index.normalize()).loc[days, "plain"], dk)

    for b in BUCKETS:
        for rung, cad in RUNGS.items():
            frames: dict[str, Any] = {}
            for name in ("dev RF", *COMPARE):
                key = KEYS[(rung, name)].format(b=b)
                p = mtc.SPXW / f"yhat_{key}.parquet"
                if not p.is_file():
                    gates.append(
                        {
                            "gate": "table on disk",
                            "forecast": key,
                            "ok": False,
                            "value": np.nan,
                        }
                    )
                    continue
                spec = mtc.Spec(key, key, "per-bar tree", "table", str(p))
                res = mtc.load_one(spec)
                gates.extend(g | {"forecast": key} for g in res["gates"])
                if res["error"]:
                    gates.append(
                        {
                            "gate": f"load ({res['error']})",
                            "forecast": key,
                            "ok": False,
                            "value": np.nan,
                        }
                    )
                    continue
                f = res["frame"]
                miss = days.difference(f.index[f["plain"].notna()].normalize())
                if len(miss):
                    gates.append(
                        {
                            "gate": "covers every trade day",
                            "forecast": key,
                            "ok": False,
                            "value": float(len(miss)),
                        }
                    )
                    continue
                frames[name] = (spec, f)
                lv = mtc.score_one((spec, f, ref, days, deck, False))
                t = trade(f)
                row = {
                    "bucket": b,
                    "rung": rung,
                    "forecast": name,
                    "key": key,
                    "qlike": float(lv["qlike_raw"]),
                    "sharpe_mid": float(lv["Sharpe_mid"]),
                    "sharpe_crossed": float(lv["Sharpe_crossed"]),
                    "sortino_mid": gui_rows._sortino(t["mid"].to_numpy()),
                    "sortino_crossed": gui_rows._sortino(t["crossed"].to_numpy()),
                    "pct_buy": float(lv["pct_buy"]),
                }
                levels.append(row)
                if key in master.index:
                    dev = max(
                        abs(float(lv[c]) - float(master.loc[key, c]))
                        for c in (
                            "qlike_raw",
                            "Sharpe_mid",
                            "Sharpe_crossed",
                            "pct_buy",
                        )
                    )
                    gates.append(
                        {
                            "gate": "reproduction of master_table.csv (max abs diff)",
                            "forecast": key,
                            "ok": dev <= REPRO_TOL,
                            "value": dev,
                        }
                    )
                if name == "default RF":
                    st = pre[
                        (pre["bucket"] == b)
                        & (pre["cadence"] == cad)
                        & (pre["arm"] == f"stored_T{cad}")
                    ]
                    if len(st):
                        dev = max(
                            abs(row["sortino_mid"] - float(st["sortino_mid"].iloc[0])),
                            abs(
                                row["sortino_crossed"]
                                - float(st["sortino_crossed"].iloc[0])
                            ),
                        )
                        gates.append(
                            {
                                "gate": "Sortino == the pre-tune study's stored row",
                                "forecast": key,
                                "ok": dev <= SORTINO_TOL,
                                "value": dev,
                            }
                        )
            fr = pre[
                (pre["bucket"] == b)
                & (pre["cadence"] == cad)
                & (pre["arm"] == "frozen_all")
            ]
            if len(fr):
                r0 = fr.iloc[0]
                levels.append(
                    {
                        "bucket": b,
                        "rung": rung,
                        "forecast": "pre-tuned RF (2016-19)",
                        "key": r0["series_id"],
                        "qlike": float(r0["qlike"]),
                        "sharpe_mid": float(r0["sharpe_mid"]),
                        "sharpe_crossed": float(r0["sharpe_crossed"]),
                        "sortino_mid": float(r0["sortino_mid"]),
                        "sortino_crossed": float(r0["sortino_crossed"]),
                        "pct_buy": float(r0["pct_buy"]),
                    }
                )
            if "dev RF" not in frames:
                continue
            spec, f = frames["dev RF"]
            td = trade(f)
            for name in COMPARE:
                if name not in frames:
                    continue
                r = mtc.score_one((spec, f, frames[name][1], days, deck, False))
                tc = trade(frames[name][1])
                tx, _ = asl.newey_west_t(
                    td["crossed"].to_numpy() - tc["crossed"].to_numpy()
                )
                pairs.append(
                    {
                        "bucket": b,
                        "rung": rung,
                        "vs": name,
                        "qlike_pct": float(r["qlike_pct_vs_ref"]),
                        "dm": float(r["dm_vs_ref"]),
                        "dm_p": float(r["dm_p_vs_ref"]),
                        "hac_t_mid": float(r["t_hac_mid_vs_ref"]),
                        "hac_t_crossed": float(tx),
                        "dsharpe_mid": float(r["dSharpe_mid_vs_ref"]),
                        "same_position": float(r["same_position_as_ref"]),
                    }
                )
    lv_df, pr_df, gt = pd.DataFrame(levels), pd.DataFrame(pairs), pd.DataFrame(gates)
    lv_df.to_csv(out / "levels.csv", index=False)
    pr_df.to_csv(out / "pairs.csv", index=False)
    gt.to_csv(out / "gates.csv", index=False)
    n_bad = int((~gt["ok"].astype(bool)).sum()) if len(gt) else 0
    write_summary(
        out, lv_df, pr_df, gt, len(days), str(days.min().date()), str(days.max().date())
    )
    print(
        lv_df.pivot_table(
            index=["bucket", "rung"], columns="forecast", values="qlike", sort=False
        )
        .round(4)
        .to_string()
    )
    print(f"gates: {len(gt) - n_bad} ok, {n_bad} failed")
    return 1 if n_bad else 0


def write_summary(
    out: Path,
    lv: pd.DataFrame,
    pr: pd.DataFrame,
    gt: pd.DataFrame,
    n: int,
    d0: str,
    d1: str,
) -> None:
    if pr.empty:  # no dev-RF table scored yet
        pr = pd.DataFrame(
            columns=["bucket", "rung", "vs", "dm", "dm_p", "hac_t_mid", "hac_t_crossed"]
        )

    def f(x: Any, k: int) -> str:
        return "" if x is None or pd.isna(x) else f"{float(x):.{k}f}"

    lines = [
        "# Dev-tuned random forest at 16:00: plain-forecast scores",
        "",
        "Written by `experiments/trees_rfdev_score_1600.py` from `levels.csv`, `pairs.csv` and `gates.csv` in this folder.",
        "",
        f"The master table's plain scorer (f^2 B, no recalibration), {n} trade days {d0} .. {d1}; Sortino as the master-table GUI. "
        "dev RF = the random forest with the 1998-2002 dev-span configuration (results/rf_devmenu_2026-10-06/rf_config.json); "
        "default RF = sklearn defaults (the canonical tables); pre-tuned RF = the 2016-19 pre-tune study's frozen_all arm "
        "(its own numbers). DM / HAC t: dev RF minus the named forecast (DM negative = dev RF more accurate; t positive = more P&L).",
        "",
        "| input set | refit | forecast | QLIKE | Sharpe mid / crossed | Sortino mid / crossed | DM vs it (p) | HAC t mid / crossed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in lv.iterrows():
        p = pr[
            (pr.bucket == r["bucket"])
            & (pr.rung == r["rung"])
            & (pr.vs == r["forecast"])
        ]
        dm = f"{f(p['dm'].iloc[0], 2)} ({f(p['dm_p'].iloc[0], 3)})" if len(p) else ""
        ht = (
            f"{f(p['hac_t_mid'].iloc[0], 2)} / {f(p['hac_t_crossed'].iloc[0], 2)}"
            if len(p)
            else ""
        )
        lines.append(
            f"| {r['bucket']} | {r['rung']} | {r['forecast']} | {f(r['qlike'], 4)} | {f(r['sharpe_mid'], 2)} / {f(r['sharpe_crossed'], 2)} | "
            f"{f(r['sortino_mid'], 2)} / {f(r['sortino_crossed'], 2)} | {dm} | {ht} |"
        )
    lines += [
        "",
        "## Gates",
        "",
        f"- {int(gt['ok'].astype(bool).sum())} of {len(gt)} pass (reproduction of master_table.csv rows, Sortino vs the pre-tune study, load_one's).",
    ]
    (out / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    sys.exit(main())
