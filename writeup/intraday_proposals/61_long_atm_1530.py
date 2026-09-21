"""61 - going net LONG the at-the-money straddle at 15:30 on the signal's buy days.

Study 60 cancelled the insurance book's remaining gamma on buy days (bought g
at-the-money straddles, g = gamma of the held legs over gamma of the 15:30
straddle) and found it no better than holding.  The question here is the next
one: on a buy day, go past neutral and be LONG the straddle the signal says is
cheap, on top of the short book, which is held to settlement as always.

Everything is study 60's gated tape (imported, gates re-run): per entry clock
10:00 .. 15:00, deck days (the signal ends 2024-04-30), index points per
contract of the short book, 15:30 straddles bought at the quoted ask and sold at
the quoted bid, held unhedged to cash settlement.

Written before running:

  P0   hold (the book)
  P2   buy g straddles on buy days (study 60: neutral)
  P7   buy (g + 1) straddles on buy days: net long one straddle
  P8   P7, and on sell days sell (1 - g) straddles: net short one on sell days
  P9   buy g straddles plus as many straddles as the short's own entry premium
       buys at the 15:30 midpoint: net long the same premium the book collected
       (the cheap-straddle days get more contracts, the way the 15:30 trade is
       sized per unit of premium)
  P10  buy the held legs back and buy one straddle (the quick check's variant)
  P7r  control: P7 on SELL days instead
  P11  P7 with the month-end override (no short that day, one straddle bought)

  H4: going net long ADDS to the book if P7 or P9 beats P0 with a paired
  block-bootstrap interval above zero at 11:00 or 13:30 (four headline cells;
  about 0.1 pass by chance).  Also reported: each policy against P2, so the
  long leg is judged apart from the neutralising leg.

Run:  python writeup/intraday_proposals/61_long_atm_1530.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "notebooks", ROOT / "writeup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "61"
DECK = ROOT / "results" / "atm_straddle_0dte_1530"
HEADLINE = ("11:00", "13:30")
B, BLOCK, SEED = 2000, 21, 0


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def policies(x: pd.DataFrame) -> dict[str, np.ndarray]:
    hold, flat, g = x["hold"].to_numpy(), x["flat"].to_numpy(), x["g"].to_numpy()
    la, sb = x["atm_long_ask"].to_numpy(), x["atm_short_bid"].to_numpy()
    buy, me = x["buy"].to_numpy(bool), x["month_end"].to_numpy(bool)
    n_prem = x["entry_pts"].to_numpy() / x["atm_mid"].to_numpy()
    sell_add = np.where(g <= 1.0, (1.0 - g) * sb, (g - 1.0) * la)
    p = {
        "P0 hold": hold,
        "P2 buy g (neutral)": np.where(buy, hold + g * la, hold),
        "P7 buy g + 1 (net long one)": np.where(buy, hold + (g + 1.0) * la, hold),
        "P8 P7 + sell 1 - g on sell days": np.where(
            buy, hold + (g + 1.0) * la, hold + sell_add
        ),
        "P9 buy g + the book's premium (net long its premium)": np.where(
            buy, hold + (g + n_prem) * la, hold
        ),
        "P10 buy back the held legs + one straddle": np.where(buy, flat + la, hold),
        "P7r control: P7 on SELL days": np.where(~buy, hold + (g + 1.0) * la, hold),
    }
    p["P11 P7 + month-end override"] = np.where(
        me, la, p["P7 buy g + 1 (net long one)"]
    )
    return p


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    p43 = _load(HERE / "43_causal_entry_over_time.py", "p43_causal_entry")
    p60 = _load(HERE / "60_moneyness_1530_atm.py", "p60_moneyness")
    long, _ = p60.build(p43)
    flags = pd.read_csv(
        DECK / "proposals" / "54" / "c_daily.csv", index_col=0, parse_dates=True
    )
    long["month_end"] = (
        pd.DatetimeIndex(long["date"])
        .map(flags["month_end"].astype(bool))
        .fillna(False)
        .astype(bool)
    )
    deck = pd.read_parquet(DECK / "daily_blk2.parquet").sort_index()
    sig = pd.Series(
        deck["signal"].to_numpy(float) > 0, index=pd.DatetimeIndex(deck.index)
    )
    d = long[long["date"].isin(sig.index)].copy()
    d["buy"] = pd.DatetimeIndex(d["date"]).map(sig).astype(bool)

    rows = []
    for clock in sorted(d["clock"].unique()):
        x = d[d["clock"] == clock].dropna(
            subset=[
                "hold",
                "flat",
                "g",
                "atm_long_ask",
                "atm_short_bid",
                "atm_mid",
                "entry_pts",
            ]
        )
        pol = policies(x)
        n = len(x)
        idx = asl.circular_block_bootstrap_idx(
            np.random.default_rng([SEED, n]), n, BLOCK, B
        )
        s = {k: p60.sharpe_rows(v[idx]) for k, v in pol.items()}
        buy = x["buy"].to_numpy(bool)
        for name, v in pol.items():
            dh = s[name] - s["P0 hold"]
            d2 = s[name] - s["P2 buy g (neutral)"]
            rows.append(
                {
                    "entry": clock,
                    "policy": name,
                    "days": n,
                    "mean_pts": float(v.mean()),
                    "sd": float(v.std(ddof=1)),
                    "Sharpe": p60.sharpe(v),
                    "vs_hold": p60.sharpe(v) - p60.sharpe(pol["P0 hold"]),
                    "vs_hold_lo": float(np.percentile(dh, 2.5)),
                    "vs_hold_hi": float(np.percentile(dh, 97.5)),
                    "vs_P2": p60.sharpe(v) - p60.sharpe(pol["P2 buy g (neutral)"]),
                    "vs_P2_lo": float(np.percentile(d2, 2.5)),
                    "vs_P2_hi": float(np.percentile(d2, 97.5)),
                    "worst": float(v.min()),
                    "buy_days_mean": float(v[buy].mean()),
                }
            )
        if clock in HEADLINE:
            la = x["atm_long_ask"].to_numpy()
            n_prem = (x["entry_pts"] / x["atm_mid"]).to_numpy()
            print(
                f"{clock}: on the {int(buy.sum())} buy days one 15:30 straddle bought at the ask earns "
                f"{la[buy].mean():+.3f} points per contract (t {la[buy].mean() / la[buy].std(ddof=1) * np.sqrt(buy.sum()):+.2f}); "
                f"the premium-sized long is {np.median(n_prem[buy]):.1f} straddles at the median "
                f"and earns {(n_prem * la)[buy].mean():+.3f} points per short contract"
            )
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "b_long_policies_by_clock.csv", index=False)
    print(
        "\nper contract of the short book, deck days; Sharpe differences with paired block-bootstrap intervals"
    )
    print(tab[tab["entry"].isin(HEADLINE)].round(3).to_string(index=False))
    print("\nSharpe by entry clock")
    print(
        tab.pivot(index="entry", columns="policy", values="Sharpe").round(2).to_string()
    )
    cells = [
        (c, p)
        for c in HEADLINE
        for p in (
            "P7 buy g + 1 (net long one)",
            "P9 buy g + the book's premium (net long its premium)",
        )
        if tab[(tab["entry"] == c) & (tab["policy"] == p)]["vs_hold_lo"].iloc[0] > 0.0
    ]
    print(
        f"\nH4  headline cells beating hold with an interval above zero: {cells or 'none'} -> "
        f"{'SUPPORTED' if cells else 'NOT SUPPORTED'}"
    )


if __name__ == "__main__":
    main()
