"""Study 81 -- the SPX line's displayed size at 15:30 on month-ends (Databento OPRA cbbo-1m).

Study 79 moved the month-end long to SPX on spread (4.1 % vs XSP's 18.2 %)
but the chain has no sizes, so whether the SPX touch holds the order was
open.  From ``pull_databento_xsp.py --root SPXW --month-ends`` (42 month-ends
2023-03-31 .. 2026-08-31, data/archive/spxw_opra/): per month-end, the
nearest-OTM SPXW pair (call at/above the 15:30 spot, put at/below) at the
BBO standing at 15:30:00, its ask / bid, relative spread and the touch size
at the ask (the smaller of the two legs), against the straddles the 15 %
month-end budget buys at that ask; the same for XSP from study 79's daily
table on the same days.

The 15:30 spot comes from put-call parity on the SPXW 15:30 book (study 79's
``parity_spot``, gated there against the chain at median 5e-5); the pair is
then the listed strikes around it.

    python writeup/intraday_proposals/81_spx_month_end_depth.py [--capital 70000 5528]
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location(
    "study79", REPO / "writeup" / "intraday_proposals" / "79_xsp_close_book.py"
)
assert _spec is not None and _spec.loader is not None
s79 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s79)

OPRA_DIR = REPO / "data" / "archive" / "spxw_opra"
XSP_DAILY = s79.OUT / "xsp_close_book_daily.csv"
OUT = REPO / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "81"
MONTH_END_FRACTION = 0.15
MULT = 100.0


def one_day(day: pd.Timestamp, q: pd.DataFrame) -> dict[str, float]:
    b = s79.book_at(q, day, s79.DECISION)
    spot = s79.parity_spot(b)
    strikes = np.sort(b.index.get_level_values(0).unique().to_numpy(dtype=float))
    kc, kp = s79.nearest_otm(strikes, spot)
    bc, ac, bsc, asc = s79.quote(b, kc, "C")
    bp, ap, bsp, asp = s79.quote(b, kp, "P")
    bid, ask = bc + bp, ac + ap
    mid = (bid + ask) / 2
    return {
        "spot": spot,
        "kc": kc,
        "kp": kp,
        "bid": bid,
        "ask": ask,
        "rel_spread": (ask - bid) / mid if mid > 0 else float("nan"),
        "ask_size": min(asc, asp),
        "ask_size_c": asc,
        "ask_size_p": asp,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, nargs="+", default=[70000.0, 5528.0])
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(str(OPRA_DIR / "cbbo1m_*.parquet"))):
        day = pd.Timestamp(Path(f).stem.split("_")[1])
        r = one_day(day, s79.load_quotes(Path(f)))
        r["day"] = day
        rows.append(r)
    x = pd.DataFrame(rows).set_index("day").sort_index()
    xsp = pd.read_csv(XSP_DAILY, index_col=0, parse_dates=True)
    x = x.join(
        xsp[["ask", "rel_spread", "ask_size", "spot_xsp"]].add_prefix("xsp_"),
        how="left",
    )
    lines = [
        f"SPXW month-ends with a 15:30 book: {len(x)} ({x.index.min().date()} .. {x.index.max().date()}); "
        f"parity spot vs XSP spot x 10: median |rel err| "
        f"{np.nanmedian(np.abs(x.spot / (x.xsp_spot_xsp * 10) - 1)):.1e}",
        f"SPX pair at 15:30: ask median {x.ask.median():.2f}, rel spread median {x.rel_spread.median():.1%} "
        f"(p90 {x.rel_spread.quantile(0.9):.1%}); touch at the ask median {x.ask_size.median():.0f} "
        f"(p10 {x.ask_size.quantile(0.1):.0f}, min {x.ask_size.min():.0f}); XSP same days: rel spread median "
        f"{x.xsp_rel_spread.median():.1%}, touch median {x.xsp_ask_size.median():.0f}",
    ]
    for cap in a.capital:
        n_spx = np.floor(cap * MONTH_END_FRACTION / (x.ask * MULT))
        n_xsp = np.floor(cap * MONTH_END_FRACTION / (x.xsp_ask * MULT))
        x[f"n_spx_{cap:.0f}"] = n_spx
        x[f"n_xsp_{cap:.0f}"] = n_xsp
        ok = x.ask_size.notna()
        lines.append(
            f"capital ${cap:,.0f} (15 % = ${cap * MONTH_END_FRACTION:,.0f}): SPX N median {n_spx.median():.0f} "
            f"(0 on {int((n_spx == 0).sum())} days); fits the SPX touch on {int((n_spx[ok] <= x.ask_size[ok]).sum())} "
            f"of {int(ok.sum())} (N > 0: {int(((n_spx <= x.ask_size) & (n_spx > 0)).sum())}); "
            f"XSP N median {n_xsp.median():.0f}, fits the XSP touch on "
            f"{int((n_xsp <= x.xsp_ask_size).sum())} of {int(x.xsp_ask_size.notna().sum())}"
        )
    x.to_csv(OUT / "spx_month_end_depth.csv", float_format="%.6g")
    txt = "\n".join(lines)
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
