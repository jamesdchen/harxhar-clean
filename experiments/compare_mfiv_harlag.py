"""Fair comparison of the CARC campaign of 2026-09-23 against the incumbent per-bar arms.

Two questions, one script:

  A. the chain's own implied variance as a regressor -- buckets mfiv_only,
     live_feasible_mfiv, live_feasible_plus_mfiv (tw 500; the chain starts
     2020-01) against the incumbent live_feasible tw 500 arms, on IDENTICAL rows
     restricted to 2022-01-03 .. 2024-04-30 (the first 500-day window that holds
     only real MFIV rows ends late 2021);
  B. the HAR ladder of the per-bar regression -- base-2, base-3 and the same-
     clock ladder (lag_scope=intra) against the production base-5 ladder, same
     bucket (live_feasible), same bar, same estimator, same window, on the rows
     both arms forecast.

Every forecast is put through the same causal back-transform as the campaign
scorer (``causal_forecasts`` of score_linear_subsection_causal.py: per-clock
smear, 250-day window, 63-day warm-up) and the trade is the deck's 15:30
sign(s) (``trade_1530`` of score_linear_subsection.py).  QLIKE is paired on the
common rows; the interval on the daily QLIKE difference is the scorer's
circular block bootstrap.

Inputs: results/linear_subsection/arms_carc/live_feasible (incumbent, flat
layout est/twN/results_<bar>.csv) and results/linear_subsection/mfiv_carc
(pulled from CARC, the spec's nested layout).  Outputs beside the pulled arms:
compare_mfiv.csv, compare_harlag.csv.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "experiments", ROOT / "notebooks"):
    sys.path.insert(0, str(p))

from score_linear_subsection import day_block_ci, trade_1530  # noqa: E402
from score_linear_subsection_causal import causal_forecasts, qlike  # noqa: E402

INCUMBENT = ROOT / "results" / "linear_subsection" / "arms_carc" / "live_feasible"
PULLED = ROOT / "results" / "linear_subsection" / "mfiv_carc"
OUT = PULLED
BARS = [
    f"bar{h:02d}{m:02d}" for h in range(10, 17) for m in (0, 30) if (h, m) <= (16, 0)
]
EST = {"ridge": "ridge", "reclasticnet": "enet", "reclasso": "lasso"}
MFIV_BUCKETS = ("mfiv_only", "live_feasible_mfiv", "live_feasible_plus_mfiv")
MFIV_START, MFIV_END = "2022-01-03", "2024-04-30"
HAR_VARIANTS = ("base2", "base3", "intra")
HAR_BARS = ("bar1430", "bar1600")


def _prep(f: Path) -> pd.DataFrame | None:
    """The causal scorer's load_adj on a path, then the causal back-transform."""
    if not f.exists():
        return None
    r = pd.read_csv(f, parse_dates=["date"]).set_index("date").sort_index()
    r = r[(r["true_adj"] > 0) & (r["true_raw"] > 0)]
    r["baseline"] = r["true_raw"] / r["true_adj"] ** 2
    r["e2"] = (r["true_adj"] - r["pred_adj"]) ** 2
    r["hhmm"] = r.index.strftime("%H:%M")
    r["day"] = r.index.normalize()
    return causal_forecasts(r)


def incumbent(est: str, tw: int, seg: str) -> pd.DataFrame | None:
    return _prep(INCUMBENT / est / f"tw{tw}" / f"results_{seg}.csv")


def pulled(root: str, bucket: str, seg: str, est: str, tw: int) -> pd.DataFrame | None:
    d = (
        PULLED
        / root
        / bucket
        / seg
        / est
        / f"tw{tw}"
        / "causal_tune_linear"
        / est
        / bucket
    )
    return _prep(d / f"results_{seg}.csv")


def paired_qlike(
    mine: pd.DataFrame, ref: pd.DataFrame, lo: str | None, hi: str | None
) -> dict:
    j = (
        mine[["true_raw", "pred_clock"]]
        .join(ref[["pred_clock"]], how="inner", rsuffix="_ref")
        .dropna()
    )
    if lo:
        j = j[j.index >= lo]
    if hi:
        j = j[j.index <= f"{hi} 23:59"]
    if j.empty:
        return {"rows": 0}
    q_m = qlike(j["true_raw"], j["pred_clock"])
    q_r = qlike(j["true_raw"], j["pred_clock_ref"])
    d = q_m - q_r
    ci_lo, ci_hi = day_block_ci(d)
    return {
        "rows": int(len(j)),
        "first": str(j.index.min().date()),
        "qlike": float(q_m.mean()),
        "qlike_ref": float(q_r.mean()),
        "pct_vs_ref": float(100 * (q_m.mean() / q_r.mean() - 1)),
        "d_ci_lo": float(ci_lo),
        "d_ci_hi": float(ci_hi),
    }


def trade(mine: pd.DataFrame, lo: str | None) -> dict:
    f = mine["pred_clock"]
    if lo:
        f = f[f.index >= lo]
    return trade_1530(f)


def part_a() -> pd.DataFrame:
    rows = []
    for est, label in EST.items():
        root = (
            "linear_subsection_lassofix" if est == "reclasso" else "linear_subsection"
        )
        ref = {b: incumbent(est, 500, b) for b in BARS + ["rth"]}
        ref16 = ref["bar1600"]
        assert ref16 is not None, f"incumbent {est} tw500 bar1600 missing"
        t_ref = trade(ref16, MFIV_START)
        rows.append(
            {
                "bucket": "live_feasible (incumbent)",
                "estimator": label,
                "arm": "bar1600",
                "qlike_pct_vs_incumbent_16": 0.0,
                "d_ci": "",
                "qlike_pct_mean_13_bars": 0.0,
                "bars_better": "",
                **{f"trade_{k}": v for k, v in t_ref.items()},
            }
        )
        for b in MFIV_BUCKETS:
            per_bar = []
            for seg in BARS:
                m = pulled(root, b, seg, est, 500)
                if m is None or ref[seg] is None:
                    continue
                per_bar.append(
                    paired_qlike(m, ref[seg], MFIV_START, MFIV_END)["pct_vs_ref"]
                )
            m16 = pulled(root, b, "bar1600", est, 500)
            assert m16 is not None, f"{b} {est} bar1600 missing"
            pq = paired_qlike(m16, ref16, MFIV_START, MFIV_END)
            t = trade(m16, MFIV_START)
            rows.append(
                {
                    "bucket": b,
                    "estimator": label,
                    "arm": "bar1600",
                    "qlike_pct_vs_incumbent_16": pq["pct_vs_ref"],
                    "d_ci": f"[{pq['d_ci_lo']:+.4f}, {pq['d_ci_hi']:+.4f}]",
                    "qlike_pct_mean_13_bars": float(np.mean(per_bar)),
                    "bars_better": f"{int(np.sum(np.array(per_bar) < 0))}/{len(per_bar)}",
                    **{f"trade_{k}": v for k, v in t.items()},
                }
            )
            # the whole-session arm, scored at the close only
            mr = pulled(root, b, "rth", est, 500)
            if mr is not None and ref["rth"] is not None:
                m16r = mr[mr["hhmm"] == "16:00"]
                r16r = ref["rth"][ref["rth"]["hhmm"] == "16:00"]
                pqr = paired_qlike(m16r, r16r, MFIV_START, MFIV_END)
                tr = trade(m16r, MFIV_START)
                rows.append(
                    {
                        "bucket": b,
                        "estimator": label,
                        "arm": "rth @16:00",
                        "qlike_pct_vs_incumbent_16": pqr["pct_vs_ref"],
                        "d_ci": f"[{pqr['d_ci_lo']:+.4f}, {pqr['d_ci_hi']:+.4f}]",
                        "qlike_pct_mean_13_bars": np.nan,
                        "bars_better": "",
                        **{f"trade_{k}": v for k, v in tr.items()},
                    }
                )
    return pd.DataFrame(rows)


def part_b() -> pd.DataFrame:
    rows = []
    for est, label in EST.items():
        for tw in (500, 2000):
            for seg in HAR_BARS:
                ref = incumbent(est, tw, seg)
                if ref is None:
                    continue
                for v in HAR_VARIANTS:
                    m = pulled(
                        "linear_subsection_har/" + v, "live_feasible", seg, est, tw
                    )
                    if m is None:
                        continue
                    pq_all = paired_qlike(m, ref, None, None)
                    pq_deck = paired_qlike(m, ref, "2020-01-03", "2024-04-30")
                    row = {
                        "ladder": v,
                        "estimator": label,
                        "train_win": tw,
                        "bar": seg,
                        "rows": pq_all["rows"],
                        "first": pq_all["first"],
                        "qlike_pct_vs_base5": pq_all["pct_vs_ref"],
                        "d_ci": f"[{pq_all['d_ci_lo']:+.4f}, {pq_all['d_ci_hi']:+.4f}]",
                        "qlike_pct_vs_base5_deck": pq_deck["pct_vs_ref"],
                    }
                    if seg == "bar1600":
                        t = trade(m, None)
                        t_ref = trade(ref, None)
                        row.update(
                            {
                                "Sharpe_mid": t["Sharpe_mid"],
                                "Sharpe_crossed": t["Sharpe_crossed"],
                                "pct_buy": t["pct_buy"],
                                "base5_Sharpe_mid": t_ref["Sharpe_mid"],
                                "base5_Sharpe_crossed": t_ref["Sharpe_crossed"],
                                "base5_pct_buy": t_ref["pct_buy"],
                            }
                        )
                    rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 250)
    a = part_a()
    a.to_csv(OUT / "compare_mfiv.csv", index=False)
    print(
        f"A  MFIV buckets vs the incumbent live_feasible, tw 500, identical rows {MFIV_START} .. {MFIV_END}; "
        f"trade = 15:30 sign(s) on the deck days in that window"
    )
    cols = [
        "bucket",
        "estimator",
        "arm",
        "qlike_pct_vs_incumbent_16",
        "d_ci",
        "qlike_pct_mean_13_bars",
        "bars_better",
        "trade_deck_days",
        "trade_pct_buy",
        "trade_Sharpe_mid",
        "trade_Sharpe_crossed",
        "trade_ridge_Sharpe_mid_same_days",
    ]
    print(a[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    b = part_b()
    b.to_csv(OUT / "compare_harlag.csv", index=False)
    print(
        "\nB  HAR ladder vs the production base-5, live_feasible, same bar / estimator / window, common rows "
        "(deck column: 2020-01 .. 2024-04); trade rows for the 16:00 bar on all deck days"
    )
    print(b.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
