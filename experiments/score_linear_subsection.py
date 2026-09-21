"""Score subsection arms of the production linear spec against the pooled arm.

Reads what experiments/run_linear_subsection.py (or the cluster run) wrote
under results/linear_subsection/<bucket>/<segment>/<estimator>/tw<days>/ and
compares every subsection arm with the pooled arm (segment ``none``) on the
SAME bars, matched by bar-end stamp: QLIKE on the raw variance level, the
paired difference with a Diebold-Mariano t (src.evaluation.diebold_mariano)
and a circular day-block bootstrap interval, over the whole common sample and
over the deck period.  For the 15:30-16:00 bar it also scores the deck's
sign(s) straddle trade with each arm's forecast in place of the ridge's.

Usage:  python experiments/score_linear_subsection.py [--bucket baseline]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402
from src.evaluation.diebold_mariano import dm_test  # noqa: E402

BASE = ROOT / "results" / "linear_subsection"
DECK = ROOT / "results" / "atm_straddle_0dte_1530" / "daily_blk2.parquet"
DECK_START, DECK_END = "2020-01-03", "2024-04-30"
BOOT_B, BOOT_BLOCK, BOOT_SEED = 2000, 21, 0
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))


def load_arm(bucket: str, seg: str, est: str, tw: int) -> pd.DataFrame | None:
    d = BASE / bucket / seg / est / f"tw{tw}" / "causal_tune_linear" / est / bucket
    f = d / ("results.csv" if seg == "none" else f"results_{seg}.csv")
    if not f.exists():
        return None
    r = pd.read_csv(f, parse_dates=["date"])
    r["qlike"] = (
        r["true_raw"] / r["pred_raw"] - np.log(r["true_raw"] / r["pred_raw"]) - 1.0
    )
    return r.set_index("date")[["true_raw", "pred_raw", "qlike"]]


def day_block_ci(d: pd.Series) -> tuple[float, float]:
    daily = d.groupby(d.index.normalize()).mean().to_numpy(float)
    idx = asl.circular_block_bootstrap_idx(
        np.random.default_rng([BOOT_SEED, daily.size]), daily.size, BOOT_BLOCK, BOOT_B
    )
    lo, hi = np.percentile(daily[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def trade_1530(pred: pd.Series) -> dict[str, float]:
    """The deck's 15:30 sign(s) trade with ``pred`` (bar ending 16:00) as the forecast."""
    deck = pd.read_parquet(DECK).sort_index()
    days = pd.DatetimeIndex(pd.to_datetime(deck.index))
    f = pred[pred.index.strftime("%H:%M") == "16:00"]
    f.index = f.index.normalize()
    f = f.reindex(days)
    ok = f.notna().to_numpy()
    q = np.where(f.to_numpy(float) > deck["iv_var"].to_numpy(float), 1.0, -1.0)[ok]
    dk = deck[ok]
    ask = (dk["ask_c"] + dk["ask_p"]).to_numpy(float)
    bid = (dk["bid_c"] + dk["bid_p"]).to_numpy(float)
    ex, r = dk["exit"].to_numpy(float), dk["R"].to_numpy(float)
    mid = q * r
    crossed = q * np.where(q > 0, ex / ask - 1.0, ex / bid - 1.0)
    ridge = np.where(dk["signal"].to_numpy(float) > 0, 1.0, -1.0) * r
    return {
        "deck_days": int(ok.sum()),
        "pct_buy": float(100 * (q > 0).mean()),
        "Sharpe_mid": float(mid.mean() / mid.std(ddof=1) * ANN),
        "Sharpe_crossed": float(crossed.mean() / crossed.std(ddof=1) * ANN),
        "ridge_Sharpe_mid_same_days": float(ridge.mean() / ridge.std(ddof=1) * ANN),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="baseline")
    a = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    root = BASE / a.bucket
    arms = sorted(
        (p.parts[-4], p.parts[-3], int(p.parts[-2][2:]))
        for p in root.glob("*/*/tw*/causal_tune_linear")
    )
    rows, trades = [], []
    for seg, est, tw in arms:
        mine = load_arm(a.bucket, seg, est, tw)
        if mine is None:
            continue
        if (mine.index.strftime("%H:%M") == "16:00").any():
            trades.append(
                {"segment": seg, "estimator": est, "train_win": tw}
                | trade_1530(mine["pred_raw"])
            )
        if seg == "none":
            continue
        for ref_tw in sorted({tw, 500}):
            ref = load_arm(a.bucket, "none", est, ref_tw)
            if ref is None:
                continue
            j = mine.join(ref, how="inner", rsuffix="_pool")
            assert np.allclose(j["true_raw"], j["true_raw_pool"], rtol=1e-9), (seg, tw)
            hhmm = j.index.strftime("%H:%M")
            deck = (j.index >= DECK_START) & (j.index <= DECK_END + " 23:59")
            for label, m in [("all bars", np.ones(len(j), bool))] + [
                (h, hhmm == h) for h in sorted(set(hhmm))
            ]:
                for smp, mm in (("common", m), ("deck period", m & deck)):
                    jj = j[mm]
                    if len(jj) < 100:
                        continue
                    d = jj["qlike"] - jj["qlike_pool"]
                    dm = dm_test(jj["qlike"].to_numpy(), jj["qlike_pool"].to_numpy())
                    lo, hi = day_block_ci(d)
                    rows.append(
                        {
                            "segment": seg,
                            "estimator": est,
                            "train_win": tw,
                            "pooled_train_win": ref_tw,
                            "bar_end": label,
                            "sample": smp,
                            "n": len(jj),
                            "first": str(jj.index.min().date()),
                            "QLIKE_subsection": float(jj["qlike"].mean()),
                            "QLIKE_pooled": float(jj["qlike_pool"].mean()),
                            "diff": float(d.mean()),
                            "pct": float(100 * d.mean() / jj["qlike_pool"].mean()),
                            "ci_lo": lo,
                            "ci_hi": hi,
                            "dm_stat": float(dm["dm"]),
                            "improves": bool(hi < 0.0),
                        }
                    )
    tab = pd.DataFrame(rows)
    out = root / "subsection_vs_pooled.csv"
    tab.to_csv(out, index=False)
    print(tab.round(5).to_string(index=False))
    print(f"\nwrote {out}")
    if trades:
        tt = pd.DataFrame(trades)
        tt.to_csv(root / "trade_1530.csv", index=False)
        print(
            "\n15:30 sign(s) trade with each arm's forecast (deck ridge: 1.338 mid / 0.870 crossed)"
        )
        print(tt.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
