"""Score the block ridge with per-clock HAR deltas against the pooled block ridge.

Reads the yhat tables ``experiments/dump_unif_yhat.py`` wrote for the pooled arm
``blk2_user`` and the three ``blk2_clockhar_a*`` arms (same panel, same code,
same cluster run) and passes each through the deck's own back-transform,
``atm_straddle_lib.load_yhat_panel_mz`` (the 250-session recalibration, fitted
on regular-hours rows), so the numbers are the traded forecast's.

  GATE   the rerun pooled arm equals ``results/spxw_pnl/yhat_blk2.parquet`` (the
         incumbent-panel block ridge on disk): the panel cache and the code are
         the ones that produced the published table.
  A      QLIKE by bar-end clock, each clock arm against pooled on identical
         rows, whole sample and the deck period, with a circular day-block
         bootstrap interval of the paired daily difference.
  B      the 15:30 sign(s) straddle trade on the deck days with each arm's
         forecast of the 15:30-16:00 bar.

Usage:  python experiments/score_blockridge_clock.py --root results/unification_clock
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

POOLED = "blk2_user"
CLOCK_ARMS = ("blk2_clockhar_a1", "blk2_clockhar_a100", "blk2_clockhar_a1000")
PUBLISHED = ROOT / "results" / "spxw_pnl" / "yhat_blk2.parquet"
DECK = ROOT / "results" / "atm_straddle_0dte_1530" / "daily_blk2.parquet"
DECK_START, DECK_END = "2020-01-03", "2024-04-30"
BOOT_B, BOOT_BLOCK, BOOT_SEED = 2000, 21, 0
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
GATE_TOL = 1e-9


def qlike(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    r = y / f
    return r - np.log(r) - 1.0


def day_block_ci(d: pd.Series) -> tuple[float, float]:
    daily = d.groupby(d.index).mean().to_numpy(float)
    idx = asl.circular_block_bootstrap_idx(
        np.random.default_rng([BOOT_SEED, daily.size]), daily.size, BOOT_BLOCK, BOOT_B
    )
    lo, hi = np.percentile(daily[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def trade_1530(panel: pd.DataFrame) -> dict[str, float]:
    deck = pd.read_parquet(DECK).sort_index()
    days = pd.DatetimeIndex(pd.to_datetime(deck.index))
    last = panel[panel["mins"] == 16 * 60]
    f = pd.Series(
        last["rv_hat"].to_numpy(float), index=pd.DatetimeIndex(last["date"])
    ).reindex(days)
    ok = f.notna().to_numpy()
    dk = deck[ok]
    q = np.where(f.to_numpy(float)[ok] > dk["iv_var"].to_numpy(float), 1.0, -1.0)
    ask = (dk["ask_c"] + dk["ask_p"]).to_numpy(float)
    bid = (dk["bid_c"] + dk["bid_p"]).to_numpy(float)
    ex, r = dk["exit"].to_numpy(float), dk["R"].to_numpy(float)
    mid = q * r
    crossed = q * np.where(q > 0, ex / ask - 1.0, ex / bid - 1.0)
    return {
        "deck_days": int(ok.sum()),
        "pct_buy": float(100 * (q > 0).mean()),
        "Sharpe_mid": float(mid.mean() / mid.std(ddof=1) * ANN),
        "Sharpe_crossed": float(crossed.mean() / crossed.std(ddof=1) * ANN),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/unification_clock")
    a = ap.parse_args()
    root = ROOT / a.root
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)

    raw = pd.read_parquet(root / f"yhat_{POOLED}.parquet").sort_values("t")
    pub = pd.read_parquet(PUBLISHED).sort_values("t")
    j = raw.merge(pub, on="t", suffixes=("", "_pub"))
    dev = float(np.nanmax(np.abs(j["yhat"] - j["yhat_pub"])))
    print(
        f"GATE  rerun pooled blk2_user vs published yhat_blk2.parquet: {len(j)} of "
        f"{len(pub)} rows matched, max |yhat diff| {dev:.2e}"
    )
    assert len(j) == len(pub) and dev < GATE_TOL, (len(j), len(pub), dev)

    panels = {
        arm: asl.load_yhat_panel_mz(root / f"yhat_{arm}.parquet")
        for arm in (POOLED, *CLOCK_ARMS)
    }
    base = panels[POOLED]
    keep = base["in_fit"].to_numpy(bool) & np.isfinite(base["rv_hat"].to_numpy(float))
    for arm in CLOCK_ARMS:
        p = panels[arm]
        assert (p["t"].to_numpy() == base["t"].to_numpy()).all(), arm
        keep &= np.isfinite(p["rv_hat"].to_numpy(float))
    y = base["rv_raw"].to_numpy(float)
    day = pd.DatetimeIndex(base["date"])
    mins = base["mins"].to_numpy(int)
    deck = np.asarray((day >= DECK_START) & (day <= DECK_END))
    l_pool = qlike(y, base["rv_hat"].to_numpy(float))

    rows = []
    clocks = sorted(set(mins[keep]))
    for arm in CLOCK_ARMS:
        l_arm = qlike(y, panels[arm]["rv_hat"].to_numpy(float))
        for label, m in [("all regular-hours bars", keep)] + [
            (f"{c // 60:02d}:{c % 60:02d}", keep & (mins == c)) for c in clocks
        ]:
            for smp, mm in (("whole", m), ("deck period", m & deck)):
                d = pd.Series(l_arm[mm] - l_pool[mm], index=day[mm])
                lo, hi = day_block_ci(d)
                rows.append(
                    {
                        "arm": arm,
                        "bar_end": label,
                        "sample": smp,
                        "n": int(mm.sum()),
                        "QLIKE_clock": float(l_arm[mm].mean()),
                        "QLIKE_pooled": float(l_pool[mm].mean()),
                        "diff": float(d.mean()),
                        "pct": float(100 * d.mean() / l_pool[mm].mean()),
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "improves": bool(hi < 0.0),
                        "worse": bool(lo > 0.0),
                    }
                )
    tab = pd.DataFrame(rows)
    tab.to_csv(root / "clock_vs_pooled.csv", index=False)
    print(tab.round(5).to_string(index=False))

    trades = pd.DataFrame([{"arm": arm} | trade_1530(p) for arm, p in panels.items()])
    trades.to_csv(root / "trade_1530.csv", index=False)
    print("\n15:30 sign(s) trade, mid / crossed Sharpe")
    print(trades.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
