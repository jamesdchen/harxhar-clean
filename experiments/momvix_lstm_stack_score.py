"""Stack and score the moments_vix LSTMs (per-bar and intraday-sequence), 16:00 bar, PLAIN forecast.

2026-10-06, user request: every model also on the input set moments_vix (src.data.loading.SUBGROUPS:
the HAR + calendar backbone + the 7 ES return moments + vix), beside baseline and all_features.  The two
LSTMs ran in the configuration of their canonical stored runs (cluster/slurm/submit_momvix_lstm.sh):

  per-bar   results/linear_subsection_lstm_mask_momvix/      (specs/causal_tune_lstm.py, WINDOW_MASK=1,
            REFIT_EVERY=1 = the canonical yhat_lstm_* run results/linear_subsection_lstm_mask)
  intraday  results/linear_subsection_lstm_intraday_momvix/  (specs/causal_tune_lstm_intraday.py =
            results/linear_subsection_lstm_intraday)

STACK (--stack; only once results/close_master_table/PLAIN_DONE exists): the canonical builders' own
functions -- build_subsection_lstm_yhat.lstm_arm_path / build_subsection_lstm_intraday_yhat.arm_path,
build_subsection_yhat.read_arm and with_production_rv (B agrees with the production table to GATE_REL,
the production rv_raw is carried) -- for the one bucket, both selection rules:
  results/spxw_pnl/yhat_lstm_moments_vix.parquet, yhat_lstm_qsel_moments_vix.parquet,
  results/spxw_pnl/yhat_lstm_intraday_moments_vix.parquet, yhat_lstm_intraday_qsel_moments_vix.parquet
Without --stack the same tables are written to the score directory only (scratch copies to score from).
Gate: each table's yhat equals its arm CSV's pred_adj (the builders' parse) on every row, its stamps are
the arm's.

SCORE: experiments/master_table_close.py's own functions, imported, never re-implemented (load_target,
load_one: plain = f^2 B on the common 16:00 target; score_one: 16:00 QLIKE of the plain forecast, the
15:30 sign(s) trade with q = sign(plain - iv_var) on the straddle, Sharpe mid / crossed, vs the fixed
reference blk2) on the 866 trade days (asserted).  No recalibration.  The same rows are recomputed for
the stored baseline / all_features / live_feasible LSTM tables; when PLAIN_DONE exists they are gated
against the plain results/close_master_table/master_table.csv (every compared column within 1e-12
relative), and the moments_vix rows are written beside the master table's baseline / all_features rows.

Outputs: results/lstm_momvix_score/ (scores.csv: every scored row; side_by_side.csv: the report table;
gates.csv; SUMMARY.md generated from them).
Run:  python experiments/momvix_lstm_stack_score.py [--stack]
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

import build_subsection_lstm_intraday_yhat as bli  # noqa: E402
import build_subsection_lstm_yhat as bly  # noqa: E402
import build_subsection_yhat as bsy  # noqa: E402
import master_table_close as mt  # noqa: E402

BUCKET = "moments_vix"
ROOTS = {
    "lstm": ROOT / "results" / "linear_subsection_lstm_mask_momvix",
    "lstm_intraday": ROOT / "results" / "linear_subsection_lstm_intraday_momvix",
}
SPX = bsy.OUT  # results/spxw_pnl
PLAIN_DONE = ROOT / "results" / "close_master_table" / "PLAIN_DONE"
MASTER = ROOT / "results" / "close_master_table" / "master_table.csv"
OUT = ROOT / "results" / "lstm_momvix_score"
ARM_ROOT = (
    mt.DEDUP_ARM_ROOT
)  # the 16:00 campaign's arm root (its target = the default root's, gated there)
N_TRADE_DAYS = 866
COMPARE_BUCKETS = ("baseline", "all_features", "live_feasible")
GATED_COLS = ("qlike_raw", "Sharpe_mid", "Sharpe_crossed", "qlike_pct_vs_ref", "n_buy")
REPORT_COLS = (
    "qlike_raw",
    "qlike_pct_vs_ref",
    "dm_vs_ref",
    "Sharpe_mid",
    "Sharpe_crossed",
    "dSharpe_mid_vs_ref",
    "dSharpe_crossed_vs_ref",
    "dSharpe_mid_vs_short",
    "pct_buy",
    "hit_rate_mid",
    "n_extreme_days",
    "calib_mean_realized_over_forecast",
)
LABEL = {"lstm": "per-bar LSTM", "lstm_intraday": "intraday-sequence LSTM"}


def arm_csv(model: str, infix: str) -> Path | None:
    if model == "lstm":
        return bly.lstm_arm_path(ROOTS[model], BUCKET, bly.TW, infix)
    return bli.arm_path(ROOTS[model], BUCKET, infix)


def build(out: Path, gates: list[dict]) -> dict[str, Path]:
    """The canonical builders' tables for the one bucket -> {key: path}."""
    out.mkdir(parents=True, exist_ok=True)
    got: dict[str, Path] = {}
    for model in ROOTS:
        for infix in ("", "qsel_"):
            key = f"{model}_{infix}{BUCKET}"
            p = arm_csv(model, infix)
            if p is None:
                print(f"skip {key}: no merged arm under {ROOTS[model]}")
                continue
            tab = bsy.read_arm(p).sort_values("t").reset_index(drop=True)
            assert not tab["t"].duplicated().any(), key
            tab = bsy.with_production_rv(tab, f"yhat_{key}.parquet")
            arm = pd.read_csv(p, parse_dates=["date"])  # the builders' parse
            arm = arm[
                (arm["true_adj"] > 0)
                & (arm["true_raw"] > 0)
                & np.isfinite(arm["pred_adj"])
            ]
            t_arm = (
                pd.DatetimeIndex(arm["date"])
                .tz_localize("America/New_York")
                .tz_convert("UTC")
            )
            same_t = bool(
                (pd.DatetimeIndex(tab["t"]).as_unit("ns") == t_arm.as_unit("ns")).all()
            )
            same_y = bool(
                np.array_equal(tab["yhat"].to_numpy(), arm["pred_adj"].to_numpy(float))
            )
            gates.append(
                {
                    "gate": "table yhat = arm pred_adj (bit), stamps = arm stamps",
                    "forecast": key,
                    "n": len(tab),
                    "ok": same_t and same_y and len(tab) == len(arm),
                }
            )
            path = out / f"yhat_{key}.parquet"
            tab.to_parquet(path, index=False)
            et = tab["t"].dt.tz_convert("America/New_York")
            print(
                f"wrote {path.relative_to(ROOT)}: {len(tab):,} rows ({et.min().date()} .. {et.max().date()})"
            )
            got[key] = path
    return got


def score(tables: dict[str, Path], gates: list[dict]) -> pd.DataFrame:
    mt.set_arm_root(ARM_ROOT)
    deck = pd.read_parquet(mt.base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    tgt = mt.load_target()
    mt._init(tgt, deck.index)
    paper = mt._paper_tags()
    ref_spec = mt.describe_table(SPX / "yhat_blk2_fomc1.parquet", paper)
    assert ref_spec.key == mt.REFERENCE, ref_spec.key
    specs = [ref_spec]
    for key, path in tables.items():
        model = "lstm_intraday" if key.startswith("lstm_intraday") else "lstm"
        rule = ", QLIKE-selected" if "_qsel_" in key else ""
        bucket = key.rsplit("_qsel_", 1)[-1].removeprefix(f"{model}_")
        specs.append(
            mt.Spec(
                key,
                f"{LABEL[model]}{rule} [{bucket}]",
                LABEL[model],
                "table",
                str(path),
            )
        )
    frames, bad = {}, []
    for s in specs:
        res = mt.load_one(s)
        for g in res["gates"]:
            gates.append(
                {
                    "gate": g["gate"],
                    "forecast": s.key,
                    "n": g["n"],
                    "ok": bool(g["ok"]),
                    "value": g["value"],
                }
            )
        if res["error"]:
            bad.append(f"{s.key}: {res['error']}")
            continue
        frames[s.key] = res["frame"]
    assert not bad, bad
    days = deck.index
    for f in frames.values():
        days = days.intersection(f.index[f["plain"].notna()].normalize())
    gates.append(
        {
            "gate": "common trade days = 866",
            "forecast": "(all)",
            "n": len(days),
            "ok": len(days) == N_TRADE_DAYS,
        }
    )
    assert len(days) == N_TRADE_DAYS, len(days)
    ref = frames[mt.REFERENCE]
    rows = []
    for s in specs:
        r = mt.score_one((s, frames[s.key], ref, days, deck, s.key == mt.REFERENCE))
        rows.append({"key": s.key, "label": s.label} | r)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stack",
        action="store_true",
        help="write the canonical tables to results/spxw_pnl",
    )
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gates: list[dict] = []
    plain = PLAIN_DONE.is_file()
    if a.stack and not plain:
        sys.exit(
            f"{PLAIN_DONE} does not exist: the canonical tables wait for the plain master table"
        )
    mine = build(SPX if a.stack else OUT / "tables", gates)
    stored = {
        f"lstm{m}_{q}{b}": SPX / f"yhat_lstm{m}_{q}{b}.parquet"
        for m in ("", "_intraday")
        for q in ("", "qsel_")
        for b in COMPARE_BUCKETS
    }
    stored = {k: p for k, p in stored.items() if p.is_file()}
    tab = score(mine | stored, gates)
    if plain:  # my recomputation of the stored LSTM rows = the plain master table's, column by column
        m = pd.read_csv(MASTER).set_index("key")
        for k in stored:
            if k not in m.index:
                gates.append(
                    {
                        "gate": "stored LSTM row in the plain master table",
                        "forecast": k,
                        "n": 1,
                        "ok": False,
                    }
                )
                continue
            mine_row = tab.set_index("key").loc[k]
            worst = max(
                abs(float(mine_row[c]) - float(m.loc[k, c]))
                / max(1.0, abs(float(m.loc[k, c])))
                for c in GATED_COLS
            )
            gates.append(
                {
                    "gate": f"recomputed row = plain master_table.csv ({', '.join(GATED_COLS)}; rel)",
                    "forecast": k,
                    "n": 1,
                    "value": worst,
                    "ok": worst <= 1e-12,
                }
            )
    tab.to_csv(OUT / "scores.csv", index=False)
    g = pd.DataFrame(gates)
    g.to_csv(OUT / "gates.csv", index=False)
    side = []
    for model in ROOTS:
        for q in ("", "qsel_"):
            for b in (BUCKET, "baseline", "all_features"):
                k = f"{model}_{q}{b}"
                if k in set(tab["key"]):
                    r = tab.set_index("key").loc[k]
                    side.append(
                        {
                            "model": LABEL[model] + (" (QLIKE-selected)" if q else ""),
                            "bucket": b,
                            "key": k,
                        }
                        | {c: r[c] for c in REPORT_COLS}
                    )
    side_df = pd.DataFrame(side)
    side_df.to_csv(OUT / "side_by_side.csv", index=False)
    pd.set_option("display.width", 250)
    print(side_df.round(4).to_string(index=False))
    n_bad = int((~g["ok"].astype(bool)).sum())
    print(
        f"gates: {len(g)} checked, {n_bad} failed; plain master table {'present' if plain else 'NOT YET (PLAIN_DONE missing): rows recomputed only'}"
    )
    write_summary(side_df, g, plain, a.stack)
    if n_bad:
        print(g[~g["ok"].astype(bool)].to_string())
        sys.exit(1)


def write_summary(
    side: pd.DataFrame, g: pd.DataFrame, plain: bool, stacked: bool
) -> None:
    def fmt(v: float, nd: int = 4) -> str:
        return "" if pd.isna(v) else f"{v:.{nd}f}"

    lines = [
        "# moments_vix LSTMs at 16:00: plain forecast, 866 trade days",
        "",
        "Generated by `experiments/momvix_lstm_stack_score.py` from `scores.csv`, `side_by_side.csv`, `gates.csv` in this folder.",
        "",
        "Scorer: experiments/master_table_close.py's own functions (plain forecast f^2 B, no recalibration; 16:00 QLIKE "
        "against the per-bar target; sign(s) with q = sign(plain - iv_var) on the straddle, Sharpe mid / crossed; "
        "reference blk2).  Point estimates (the master table's intervals are disabled since b761b28).",
        f"Tables {'stacked into results/spxw_pnl/' if stacked else 'NOT stacked yet (scored from tables/ here)'}; "
        f"the stored baseline / all_features rows {'gated against the plain master_table.csv' if plain else 'recomputed only (plain master table not done)'}.",
        "",
        "| model | bucket | QLIKE | vs ref % | DM | Sharpe mid | Sharpe crossed | dSharpe mid vs ref | % buy |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in side.iterrows():
        lines.append(
            f"| {r['model']} | {r['bucket']} | {fmt(r['qlike_raw'])} | {fmt(r['qlike_pct_vs_ref'], 1)} | {fmt(r['dm_vs_ref'], 2)} | "
            f"{fmt(r['Sharpe_mid'], 2)} | {fmt(r['Sharpe_crossed'], 2)} | {fmt(r['dSharpe_mid_vs_ref'], 2)} | {fmt(r['pct_buy'], 1)} |"
        )
    lines += [
        "",
        f"Gates: {len(g)} checked, {int((~g['ok'].astype(bool)).sum())} failed (gates.csv).",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
