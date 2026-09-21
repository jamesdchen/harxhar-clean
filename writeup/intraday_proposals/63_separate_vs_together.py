"""63 - is doing the insurance trade and the 15:30 trade separately the same as doing them together?

Two trades, one contract each, index points per contract, deck days (the 15:30
signal ends 2024-04-30):

  insurance  sell the nearest-OTM straddle at the entry clock (11:00 or 13:30)
             at the quoted bid, hedge its delta every 30 minutes, hold it to
             cash settlement (proposal 43's tape via proposal 52's book)
  15:30      the 15:30 nearest-OTM straddle on sign(s): bought at the quoted ask
             on buy days, sold at the quoted bid on sell days, held unhedged to
             cash settlement

SEPARATELY  each trade in its own account; the total is the sum.
TOGETHER    one account: at 15:30 the account's option positions are netted
            strike by strike (on days the two straddles share a strike the
            15:30 purchase closes part of the short, a 15:30 sale adds to it),
            one futures hedge, one settlement.

  A  P&L: the together account is rebuilt from its own cash flows (premium in,
     premium out, the hedge) and the netted settlement value, and compared with
     the separate sum day by day.  Positions are linear, so the two must agree
     to rounding; the check makes sure nothing in the books (a strike, a sign,
     a fill) differs between the two ways of writing it.
  B  what each trade does alone and what the sum does: mean, sd, Sharpe, worst
     day, drawdown, correlation.
  C  what does differ: the capital at risk at 15:30.  The index is moved at
     15:30 across a grid of settlements within the live package's stress jump
     (live.ibkr.sizing.STRESS_JUMP) and each position is marked from its 15:30
     midpoint.  Separately, each account must hold its own worst loss; together,
     the account holds the worst loss of the netted position, which is smaller
     when one trade gains where the other loses (the 15:30 long against the
     insurance short on buy days).

GATES  proposal 52's book equals proposal 43's at 15:30 (its own gate); the
       15:30 straddle's midpoint return equals the deck's R; the insurance means
       reproduce study 61 (+1.527 at 11:00, +0.798 at 13:30), and so does the
       15:30 purchase on buy days (+0.037 points at the ask).

Run:  python writeup/intraday_proposals/63_separate_vs_together.py
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
from live.ibkr.sizing import STRESS_JUMP  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = ROOT / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "63"
DECK = ROOT / "results" / "atm_straddle_0dte_1530"
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
CLOSE = "15:30"
ENTRIES = ("11:00", "13:30")
GRID_POINTS = 201  # settlement grid across [-STRESS_JUMP, +STRESS_JUMP], zero included
GATE_BOOK = {"11:00": 1.527, "13:30": 0.798}
GATE_LONG_BUY_DAYS = 0.037
GATE_ROUND_TOL = 5e-4  # the gated means were printed to three decimals
GATE_R_TOL = 1e-6  # the chain stores quotes as float32
IDENTITY_TOL = 1e-9


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sharpe(x: np.ndarray) -> float:
    return float(np.mean(x) / np.std(x, ddof=1) * ANN)


def maxdd(x: np.ndarray) -> float:
    path = np.cumsum(x)
    return float((path - np.maximum(np.maximum.accumulate(path), 0.0)).min())


def payoff(s: np.ndarray, kc: np.ndarray, kp: np.ndarray) -> np.ndarray:
    return np.maximum(s - kc, 0.0) + np.maximum(kp - s, 0.0)


def main() -> None:  # noqa: PLR0915
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    p52 = _load(HERE / "52_flatten_1500_vrp.py", "p52_flatten")
    p43 = p52.p43
    stamp, _ = p43.session_stamps()
    sessions = pd.DatetimeIndex(stamp.index).difference(p43.half_sessions(stamp))
    ch = p43.build_chain(stamp, sessions)
    quotes = p52.held_leg_quotes(stamp, ch)
    idx = ch["dates"]
    k = p43.CLOCKS.index(CLOSE)
    s15, v15, sc = ch["S"][:, k], ch["tot"][:, k], ch["S_close"]
    kc15, kp15 = ch["K_c"][:, k], ch["K_p"][:, k]
    mid15, bid15 = ch["entry"][:, k], ch["bid"][:, k]
    ask15 = 2.0 * mid15 - bid15
    pay15 = payoff(sc, kc15, kp15)

    deck = pd.read_parquet(DECK / "daily_blk2.parquet").sort_index()
    di = pd.DatetimeIndex(deck.index)
    pos = idx.get_indexer(di)
    assert (pos >= 0).all()
    dev = float(
        np.nanmax(np.abs(pay15[pos] / mid15[pos] - 1.0 - deck["R"].to_numpy(float)))
    )
    assert dev < GATE_R_TOL, dev
    print(
        f"GATE  15:30 straddle midpoint return vs the deck's R on {len(di)} days: {dev:.1e}"
    )
    q = np.where(deck["signal"].to_numpy(float) > 0, 1.0, -1.0)

    grid = np.linspace(-STRESS_JUMP, STRESS_JUMP, GRID_POINTS)
    rows_b, rows_c, daily = [], [], []
    for e in ENTRIES:
        j = p43.CLOCKS.index(e)
        b = p52.book(ch, quotes, e, CLOSE)
        kc, kp = ch["K_c"][:, j], ch["K_p"][:, j]
        bid_e = ch["bid"][:, j]
        settle_e = payoff(sc, kc, kp)
        hedge_hold = (
            b["hold_pts"].to_numpy() + settle_e - bid_e
        )  # hold = bid - settle + hedge
        held_mid15 = quotes[CLOSE][f"{e}_mid"]
        delta15 = p43.pkg_delta_vec(v15, s15, kc, kp)

        f = pd.DataFrame(
            {
                "insurance": b["hold_pts"].to_numpy(),
                "t1530": np.full(len(idx), np.nan),
                "hedge_hold": hedge_hold,
                "bid_e": bid_e,
                "kc": kc,
                "kp": kp,
                "kc15": kc15,
                "kp15": kp15,
                "held_mid15": held_mid15,
                "delta15": delta15,
            },
            index=idx,
        ).iloc[pos]
        f["q"] = q
        f["t1530"] = np.where(q > 0, pay15[pos] - ask15[pos], bid15[pos] - pay15[pos])
        f = f.dropna(subset=["insurance", "t1530", "held_mid15", "delta15"])
        m = pos[np.isin(di, f.index)]
        qq = f["q"].to_numpy()

        # gates against study 61
        got = float(f["insurance"].mean())
        assert abs(got - GATE_BOOK[e]) < GATE_ROUND_TOL, (e, got)
        got_l = float((pay15[m] - ask15[m])[qq > 0].mean())
        assert abs(got_l - GATE_LONG_BUY_DAYS) < GATE_ROUND_TOL, got_l

        # A: the together account from its own cash flows and netted settlement
        n = len(f)
        sc_m = sc[m]
        kc_e, kp_e = f["kc"].to_numpy(), f["kp"].to_numpy()
        kc_t, kp_t = f["kc15"].to_numpy(), f["kp15"].to_numpy()

        def call(strike: np.ndarray) -> np.ndarray:
            return np.maximum(sc_m - strike, 0.0)

        def put(strike: np.ndarray) -> np.ndarray:
            return np.maximum(strike - sc_m, 0.0)

        # the account's call book: one line of quantity (q - 1) where the two
        # straddles share the call strike, two lines where they do not; the same
        # for the puts
        netted_value = np.where(
            kc_e == kc_t, (qq - 1.0) * call(kc_e), -call(kc_e) + qq * call(kc_t)
        ) + np.where(kp_e == kp_t, (qq - 1.0) * put(kp_e), -put(kp_e) + qq * put(kp_t))
        cash = f["bid_e"].to_numpy() + np.where(qq > 0, -ask15[m], bid15[m])
        together = cash + netted_value + f["hedge_hold"].to_numpy()
        separate = f["insurance"].to_numpy() + f["t1530"].to_numpy()
        gap = float(np.max(np.abs(together - separate)))
        assert gap < IDENTITY_TOL, gap
        share_both = float(((f["kc"] == f["kc15"]) & (f["kp"] == f["kp15"])).mean())
        share_one = float(((f["kc"] == f["kc15"]) ^ (f["kp"] == f["kp15"])).mean())
        print(
            f"A  {e}: together minus separately, largest daily gap {gap:.1e} points over {n} days; "
            f"the two straddles share both strikes on {share_both:.0%} of days and one on {share_one:.0%}"
        )

        # B: each alone and the sum
        ins, t = f["insurance"].to_numpy(), f["t1530"].to_numpy()
        for name, x in (
            ("insurance alone", ins),
            ("15:30 alone", t),
            ("both (sum = together)", ins + t),
        ):
            rows_b.append(
                {
                    "entry": e,
                    "book": name,
                    "days": n,
                    "mean_pts": float(x.mean()),
                    "sd": float(x.std(ddof=1)),
                    "Sharpe": sharpe(x),
                    "worst": float(x.min()),
                    "max_drawdown": maxdd(x),
                }
            )
        rows_b.append(
            {
                "entry": e,
                "book": "correlation",
                "mean_pts": float(np.corrcoef(ins, t)[0, 1]),
            }
        )
        daily.append(
            pd.DataFrame(
                {
                    "entry": e,
                    "insurance": ins,
                    "t1530": t,
                    "together": together,
                    "q": qq,
                },
                index=f.index,
            )
        )

        # C: capital at risk at 15:30, marked from the 15:30 midpoints
        s1 = s15[m][:, None] * (1.0 + grid[None, :])
        ins_pnl = -(
            payoff(s1, f["kc"].to_numpy()[:, None], f["kp"].to_numpy()[:, None])
            - f["held_mid15"].to_numpy()[:, None]
        ) + f["delta15"].to_numpy()[:, None] * (s1 - s15[m][:, None])
        t_pnl = qq[:, None] * (
            payoff(s1, kc15[m][:, None], kp15[m][:, None]) - mid15[m][:, None]
        )
        worst_sep = (-ins_pnl).max(axis=1) + (-t_pnl).max(axis=1)
        worst_tog = (-(ins_pnl + t_pnl)).max(axis=1)
        for side, mk in (
            ("all days", np.ones(n, bool)),
            ("buy days (15:30 long)", qq > 0),
            ("sell days (15:30 short)", qq < 0),
        ):
            rows_c.append(
                {
                    "entry": e,
                    "days": side,
                    "n": int(mk.sum()),
                    "capital_separately_pts": float(worst_sep[mk].mean()),
                    "capital_together_pts": float(worst_tog[mk].mean()),
                    "together_over_separately": float(
                        worst_tog[mk].mean() / worst_sep[mk].mean()
                    ),
                }
            )

    tb = pd.DataFrame(rows_b)
    tb.to_csv(OUT / "b_separately_and_together.csv", index=False)
    print("\nB  one contract each, index points per contract, deck days")
    print(tb.round(3).to_string(index=False))
    tc = pd.DataFrame(rows_c)
    tc.to_csv(OUT / "c_capital_at_1530.csv", index=False)
    print(
        f"\nC  worst loss from 15:30 marks over settlements within +-{STRESS_JUMP:.0%}, index points "
        f"(what each account must hold)"
    )
    print(tc.round(3).to_string(index=False))
    pd.concat(daily).to_csv(OUT / "a_daily.csv")


if __name__ == "__main__":
    main()
