"""60 - moneyness at 15:30: what the 15:30 signal says about the insurance book.

Background.  The 15:30 signal s = rv_hat - slice compares a forecast of the
last half hour's variance with the price of a FRESH at-the-money straddle.  The
insurance book (sell the nearest-OTM straddle at an earlier entry clock, hedge
the package delta every 30 minutes, hold to cash settlement) holds, at 15:30,
the legs it sold hours earlier.  A quick check found that buying those legs back
on the signal's buy days costs the 11:00 book 0.30 index points a day (Sharpe
3.35 -> 2.76) and does nothing for the 13:30 book, and that the signal splits
the 13:30 book's last half hour (+0.04 on buy days, +0.74 on sell days) but not
the 11:00 book's (+0.74, +0.87).  The explanation offered: by 15:30 the 11:00
strikes are usually far from the index, so the book holds little of the gamma
the signal is about, while the 13:30 strikes are still near the money.  Every
earlier buy-back experiment (studies 37, 41, 42, 47, 52) acted on the HELD legs
whatever their moneyness, and none bought the straddle that is at the money at
15:30.

The measure of moneyness, from 15:30 information only: the Black-76 gamma of the
held legs divided by the gamma of the 15:30 nearest-OTM straddle, both at the
15:30 index level and the total volatility implied by that straddle's own
midpoint,

    g = Gamma(held legs) / Gamma(15:30 straddle).

g is the number of 15:30 straddles whose last-half-hour gamma the book still
carries; g is near one when the held strikes are at the money and near zero
when both are far from it.  (One volatility for every strike: the smile is
ignored, which matters only for strikes far from the index, where g is small
whatever volatility is used.)

PART A - the explanation, on the deck days (the signal exists to 2024-04-30),
pooling the eleven entry clocks 10:00 .. 15:00 (one row per session and clock).
L = the held short's last half hour = holding to settlement minus buying the
held legs back at their 15:30 quoted ask, index points per contract.

  A1  L on the signal's buy and sell days by tercile of g (terciles of the
      pooled rows), with a day-block bootstrap of the sell-minus-buy gap and of
      the gap's difference between the top and bottom terciles.
  A2  regression of L on the signal, g and their product with clock fixed
      effects and a separate signal effect for every clock, standard errors
      clustered by session: the product's coefficient is the moneyness effect
      WITHIN a clock, so it cannot be the entry clock in disguise.

  H1 (written before running): the explanation is SUPPORTED if the product's
  coefficient in A2 is negative with t < -2 AND the A1 top-minus-bottom
  difference of the gap has a bootstrap interval above zero.

PART B - the at-the-money straddle as the instrument.  Per entry clock, deck
days, index points per contract of the short book, straddles bought at the
15:30 quoted ask and sold at the quoted bid, held unhedged to settlement:

  P0  hold always (the book)
  P1  buy the held legs back on buy days (the earlier experiments)
  P2  buy days: keep the short and BUY g at-the-money straddles (cancel exactly
      the gamma the signal prices); sell days: hold
  P3  buy days: keep the short and buy one at-the-money straddle (the quick check)
  P4  P2 on buy days; sell days: sell (1 - g) at-the-money straddles, so the
      book always carries one straddle's worth of short gamma on sell days and
      none on buy days (a negative amount is a purchase at the ask)
  P2r the reverse of P2 (buy the g straddles on SELL days): the control
  P5  the month-end override on the book: on the last session of a month no
      short, one at-the-money straddle bought at 15:30 (studies 54, 55)
  P6  P4 with the month-end override

  H2 (written before running): the at-the-money straddle is the better
  instrument if P2 beats P1 at 11:00 (paired interval of the Sharpe difference
  above zero) and P2 is not significantly worse than P1 at 13:30.
  H3: the gamma-matched overlay ADDS to the book if P2 or P4 beats P0 with an
  interval above zero at 11:00 or 13:30 (four headline cells; about 0.1 pass
  by chance on the bootstrap leg).

GATES  proposal 43's gates on the tape; the 15:30 straddle's midpoint return
       equals the deck's R; the gamma formula equals a central difference of the
       package delta; the quick check reproduces (13:30: hold +0.798, flat on
       buy days +0.781, plus one straddle +0.813; 11:00: +1.527, +1.231, +1.542).

Run:  python writeup/intraday_proposals/60_moneyness_1530_atm.py
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
OUT = ROOT / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "60"
DECK = ROOT / "results" / "atm_straddle_0dte_1530"
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))
CLOSE = "15:30"
HEADLINE = ("11:00", "13:30")
B, BLOCK, SEED = 2000, 21, 0
T_BAR = 2.0
GATE_R_TOL = 1e-6  # the chain stores quotes as float32
GATE_QUICK = {
    "13:30": {"P0": 0.798, "P1": 0.781, "P3": 0.813},
    "11:00": {"P0": 1.527, "P1": 1.231, "P3": 1.542},
}
GATE_QUICK_TOL = 5e-4  # the quick check was printed to three decimals
FD_REL_STEP = 1e-5  # central-difference step for the gamma gate, relative to spot
GATE_FD_TOL = 1e-4  # relative agreement of the gamma formula with the difference


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sharpe(x: np.ndarray) -> float:
    return float(np.mean(x) / np.std(x, ddof=1) * ANN)


def sharpe_rows(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) / x.std(axis=1, ddof=1) * ANN


def pkg_gamma(
    v: np.ndarray, f: np.ndarray, kc: np.ndarray, kp: np.ndarray
) -> np.ndarray:
    """Black-76 gamma of a call at kc plus a put at kp (forward = spot, r = 0)."""
    v, f = np.asarray(v, float), np.asarray(f, float)
    kc, kp = np.asarray(kc, float), np.asarray(kp, float)
    ok = (
        np.isfinite(v)
        & (v > 0)
        & np.isfinite(f)
        & (f > 0)
        & np.isfinite(kc)
        & (kc > 0)
        & np.isfinite(kp)
        & (kp > 0)
    )
    vv, ff = np.where(ok, v, 1.0), np.where(ok, f, 1.0)
    kcc, kpp = np.where(ok, kc, 1.0), np.where(ok, kp, 1.0)
    d1c = (np.log(ff / kcc) + 0.5 * vv * vv) / vv
    d1p = (np.log(ff / kpp) + 0.5 * vv * vv) / vv
    pdf = (np.exp(-0.5 * d1c * d1c) + np.exp(-0.5 * d1p * d1p)) / np.sqrt(2.0 * np.pi)
    return np.where(ok, pdf / (ff * vv), np.nan)


def ols_cluster(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """OLS with standard errors clustered by ``groups`` (the usual small-sample factor)."""
    n, p = x.shape
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    u = y - x @ beta
    codes, uniq = pd.factorize(groups)
    g = len(uniq)
    s = np.zeros((g, p))
    np.add.at(s, codes, x * u[:, None])
    factor = g / (g - 1) * (n - 1) / (n - p)
    v = xtx_inv @ (s.T @ s) @ xtx_inv * factor
    return beta, np.sqrt(np.diag(v))


def day_counts(n_days: int) -> np.ndarray:
    """(B, n_days) multiplicities of a circular day-block bootstrap."""
    idx = asl.circular_block_bootstrap_idx(
        np.random.default_rng([SEED, n_days]), n_days, BLOCK, B
    )
    flat = (idx + n_days * np.arange(B)[:, None]).ravel()
    return np.bincount(flat, minlength=B * n_days).reshape(B, n_days).astype(float)


def build(p43: ModuleType) -> tuple[pd.DataFrame, pd.DataFrame]:
    stamp, _ = p43.session_stamps()
    sessions = pd.DatetimeIndex(stamp.index).difference(p43.half_sessions(stamp))
    ch = p43.build_chain(stamp, sessions)
    # proposal 43's per-clock tapes, built in this process: its clock_books hands
    # the worker a function of a module loaded by path, which a spawned worker
    # cannot import; each clock is one numpy pass, so nothing is lost.
    arrays = {
        k: ch[k] for k in ("S", "K_c", "K_p", "entry", "bid", "tot", "ask_c", "ask_p")
    }
    arrays["S_close"] = ch["S_close"]
    books = {
        clock: {
            k: pd.Series(v, index=ch["dates"])
            for k, v in p43._clock_book((j, arrays)).items()
        }
        for j, clock in enumerate(p43.ENTRIES)
    }
    p43.gate_zero(books, ch)
    p43.gate_one(books)
    idx = ch["dates"]
    k = p43.CLOCKS.index(CLOSE)
    s15, v15 = ch["S"][:, k], ch["tot"][:, k]
    kc15, kp15 = ch["K_c"][:, k], ch["K_p"][:, k]
    mid15, bid15 = ch["entry"][:, k], ch["bid"][:, k]
    ask15 = 2.0 * mid15 - bid15
    sc = ch["S_close"]
    settle15 = np.maximum(sc - kc15, 0.0) + np.maximum(kp15 - sc, 0.0)
    ok15 = np.isfinite(mid15) & (mid15 > 0) & (bid15 > 0) & np.isfinite(settle15)
    atm = pd.DataFrame(
        {
            "atm_long_ask": np.where(ok15, settle15 - ask15, np.nan),
            "atm_short_bid": np.where(ok15, bid15 - settle15, np.nan),
            "atm_R_mid": np.where(ok15, settle15 / mid15 - 1.0, np.nan),
            "atm_mid": np.where(ok15, mid15, np.nan),
        },
        index=idx,
    )
    deck = pd.read_parquet(DECK / "daily_blk2.parquet")["R"]
    dev = float((atm["atm_R_mid"].reindex(deck.index) - deck).abs().max())
    assert dev < GATE_R_TOL, dev
    print(
        f"GATE  15:30 straddle midpoint return vs the deck's R on {len(deck)} days: {dev:.1e}"
    )

    g_atm = pkg_gamma(v15, s15, kc15, kp15)
    # gamma gate: the formula against a central difference of the package delta
    h = s15 * FD_REL_STEP
    fd = (
        p43.pkg_delta_vec(v15, s15 + h, kc15, kp15)
        - p43.pkg_delta_vec(v15, s15 - h, kc15, kp15)
    ) / (2.0 * h)
    both = np.isfinite(g_atm) & np.isfinite(fd) & (g_atm > 0)
    rel = float(np.max(np.abs(fd[both] / g_atm[both] - 1.0)))
    assert rel < GATE_FD_TOL, rel
    print(
        f"GATE  gamma formula vs a central difference of the delta: max relative gap {rel:.1e}"
    )

    rows = []
    for j, clock in enumerate(p43.ENTRIES):
        bk = books[clock]
        entry = bk["entry"].to_numpy(float)
        hold = bk["hold"].to_numpy(float) * entry
        flat = bk["flatten"].to_numpy(float) * entry
        kc, kp = ch["K_c"][:, j], ch["K_p"][:, j]
        g_held = pkg_gamma(v15, s15, kc, kp)
        centre = np.sqrt(kc * kp)
        rows.append(
            pd.DataFrame(
                {
                    "date": idx,
                    "clock": clock,
                    "hold": hold,
                    "flat": flat,
                    "L": hold - flat,
                    "g": g_held / g_atm,
                    "moves_from_centre": np.log(s15 / centre) / v15,
                    "entry_pts": entry,
                }
            )
        )
    long = pd.concat(rows, ignore_index=True)
    long = long.merge(atm, left_on="date", right_index=True, how="left")
    return long, atm


def part_a(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    d = d.dropna(subset=["L", "g"]).copy()
    days = pd.DatetimeIndex(sorted(d["date"].unique()))
    code = days.get_indexer(pd.DatetimeIndex(d["date"]))
    cnt = day_counts(len(days))
    cuts = np.quantile(d["g"].to_numpy(), [1 / 3, 2 / 3])
    d["tercile"] = np.digitize(d["g"].to_numpy(), cuts)
    L = d["L"].to_numpy()
    buy = d["buy"].to_numpy(bool)

    def boot_mean(mask: np.ndarray) -> np.ndarray:
        s = np.bincount(code[mask], weights=L[mask], minlength=len(days))
        n = np.bincount(code[mask], minlength=len(days)).astype(float)
        return (cnt @ s) / (cnt @ n)

    rows, gaps = [], {}
    for t, name in enumerate(
        ("low g (far from the money)", "middle", "high g (near the money)")
    ):
        mt = d["tercile"].to_numpy() == t
        mb, ms = mt & buy, mt & ~buy
        gap_b = boot_mean(ms) - boot_mean(mb)
        gaps[t] = gap_b
        lo, hi = np.percentile(gap_b, [2.5, 97.5])
        rows.append(
            {
                "tercile": name,
                "g_range": f"{d.loc[mt, 'g'].min():.2f}..{d.loc[mt, 'g'].max():.2f}",
                "g_median": float(d.loc[mt, "g"].median()),
                "rows": int(mt.sum()),
                "L_buy_days": float(L[mb].mean()),
                "L_sell_days": float(L[ms].mean()),
                "gap_sell_minus_buy": float(L[ms].mean() - L[mb].mean()),
                "gap_ci_lo": float(lo),
                "gap_ci_hi": float(hi),
            }
        )
    did = gaps[2] - gaps[0]
    did_lo, did_hi = np.percentile(did, [2.5, 97.5])
    a1 = pd.DataFrame(rows)
    obs_did = float(a1["gap_sell_minus_buy"].iloc[2] - a1["gap_sell_minus_buy"].iloc[0])
    print(
        "\nA1  the held short's last half hour (index points per contract), by moneyness tercile"
    )
    print(a1.round(3).to_string(index=False))
    print(
        f"    top minus bottom tercile, difference of the gap: {obs_did:+.3f} [{did_lo:+.3f}, {did_hi:+.3f}]"
    )

    clocks = sorted(d["clock"].unique())
    b = buy.astype(float)
    g = d["g"].to_numpy()
    cols = {"const": np.ones(len(d)), "buy": b, "g": g, "buy_x_g": b * g}
    x1 = np.column_stack(list(cols.values()))
    beta1, se1 = ols_cluster(x1, L, code)
    cols2: dict[str, np.ndarray] = {"const": np.ones(len(d))}
    for c in clocks[1:]:
        cols2[f"clock {c}"] = (d["clock"].to_numpy() == c).astype(float)
    for c in clocks:
        cols2[f"buy x clock {c}"] = b * (d["clock"].to_numpy() == c)
    cols2["g"] = g
    cols2["buy_x_g"] = b * g
    x2 = np.column_stack(list(cols2.values()))
    beta2, se2 = ols_cluster(x2, L, code)
    reg = pd.DataFrame(
        [
            {
                "spec": "pooled",
                "term": k,
                "coef": beta1[i],
                "se": se1[i],
                "t": beta1[i] / se1[i],
            }
            for i, k in enumerate(cols)
        ]
        + [
            {
                "spec": "clock fixed effects + signal by clock",
                "term": k,
                "coef": beta2[i],
                "se": se2[i],
                "t": beta2[i] / se2[i],
            }
            for i, k in enumerate(cols2)
            if k in ("g", "buy_x_g")
        ]
    )
    print(
        "\nA2  L on the signal, g and their product; session-clustered standard errors"
    )
    print(reg.round(3).to_string(index=False))
    t_int = float(beta2[-1] / se2[-1])
    h1 = bool(t_int < -T_BAR and did_lo > 0.0)
    print(
        f"\nH1  within-clock product t {t_int:+.2f}, top-minus-bottom interval "
        f"[{did_lo:+.3f}, {did_hi:+.3f}] -> {'SUPPORTED' if h1 else 'NOT SUPPORTED'}"
    )
    by_clock = (
        d.groupby("clock")
        .agg(
            g_median=("g", "median"),
            g_q25=("g", lambda v: float(np.quantile(v, 0.25))),
            g_q75=("g", lambda v: float(np.quantile(v, 0.75))),
            moves_median=("moves_from_centre", lambda v: float(np.median(np.abs(v)))),
            L_buy=("L", lambda v: float(v[d.loc[v.index, "buy"]].mean())),
            L_sell=("L", lambda v: float(v[~d.loc[v.index, "buy"]].mean())),
        )
        .reset_index()
    )
    print(
        "\n    by entry clock: moneyness at 15:30 and the held short's last half hour"
    )
    print(by_clock.round(3).to_string(index=False))
    a1.to_csv(OUT / "a1_terciles.csv", index=False)
    reg.to_csv(OUT / "a2_regression.csv", index=False)
    by_clock.to_csv(OUT / "a_by_clock.csv", index=False)
    return a1, reg, h1


def policies(x: pd.DataFrame) -> dict[str, np.ndarray]:
    hold, flat, g = x["hold"].to_numpy(), x["flat"].to_numpy(), x["g"].to_numpy()
    la, sb = x["atm_long_ask"].to_numpy(), x["atm_short_bid"].to_numpy()
    buy, me = x["buy"].to_numpy(bool), x["month_end"].to_numpy(bool)
    # selling (1 - g) straddles; a negative amount is a purchase of (g - 1) at the ask
    sell_add = np.where(g <= 1.0, (1.0 - g) * sb, (g - 1.0) * la)
    p = {
        "P0 hold": hold,
        "P1 buy back the held legs on buy days": np.where(buy, flat, hold),
        "P2 buy g at-the-money straddles on buy days": np.where(
            buy, hold + g * la, hold
        ),
        "P3 buy one at-the-money straddle on buy days": np.where(buy, hold + la, hold),
        "P4 P2 + sell (1 - g) on sell days": np.where(
            buy, hold + g * la, hold + sell_add
        ),
        "P2r control: buy g straddles on SELL days": np.where(
            ~buy, hold + g * la, hold
        ),
    }
    p["P5 month-end override on the book"] = np.where(me, la, hold)
    p["P6 P4 + month-end override"] = np.where(
        me, la, p["P4 P2 + sell (1 - g) on sell days"]
    )
    return p


def part_b(d: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, bool]]:
    rows = []
    diffs: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    for clock in sorted(d["clock"].unique()):
        x = d[d["clock"] == clock].dropna(
            subset=["hold", "flat", "g", "atm_long_ask", "atm_short_bid"]
        )
        pol = policies(x)
        n = len(x)
        idx = asl.circular_block_bootstrap_idx(
            np.random.default_rng([SEED, n]), n, BLOCK, B
        )
        base = pol["P0 hold"]
        s_base = sharpe_rows(base[idx])
        s_p1 = sharpe_rows(pol["P1 buy back the held legs on buy days"][idx])
        if clock in GATE_QUICK:
            for key, name in (
                ("P0", "P0 hold"),
                ("P1", "P1 buy back the held legs on buy days"),
                ("P3", "P3 buy one at-the-money straddle on buy days"),
            ):
                got = float(pol[name].mean())
                assert abs(got - GATE_QUICK[clock][key]) < GATE_QUICK_TOL, (
                    clock,
                    key,
                    got,
                )
        for name, v in pol.items():
            sv = sharpe_rows(v[idx])
            dv = sv - s_base
            dp1 = sv - s_p1
            lo, hi = np.percentile(dv, [2.5, 97.5])
            lo1, hi1 = np.percentile(dp1, [2.5, 97.5])
            rows.append(
                {
                    "entry": clock,
                    "policy": name,
                    "days": n,
                    "mean_pts": float(v.mean()),
                    "sd": float(v.std(ddof=1)),
                    "Sharpe": sharpe(v),
                    "vs_hold": sharpe(v) - sharpe(base),
                    "vs_hold_lo": float(lo),
                    "vs_hold_hi": float(hi),
                    "vs_P1": sharpe(v)
                    - sharpe(pol["P1 buy back the held legs on buy days"]),
                    "vs_P1_lo": float(lo1),
                    "vs_P1_hi": float(hi1),
                    "worst": float(v.min()),
                }
            )
            diffs[(clock, name, "hold")] = (
                sharpe(v) - sharpe(base),
                float(lo),
                float(hi),
            )
            diffs[(clock, name, "P1")] = (
                sharpe(v) - sharpe(pol["P1 buy back the held legs on buy days"]),
                float(lo1),
                float(hi1),
            )
    print("GATE  the quick check reproduces at 11:00 and 13:30")
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "b_policies_by_clock.csv", index=False)
    show = tab[tab["entry"].isin(HEADLINE)]
    print(
        "\nB  per contract, deck days; Sharpe differences with paired block-bootstrap intervals"
    )
    print(show.round(3).to_string(index=False))
    wide = tab.pivot(index="entry", columns="policy", values="Sharpe")
    print("\n   Sharpe by entry clock")
    print(wide.round(2).to_string())
    p2 = "P2 buy g at-the-money straddles on buy days"
    p4 = "P4 P2 + sell (1 - g) on sell days"
    h2 = bool(
        diffs[("11:00", p2, "P1")][1] > 0.0 and diffs[("13:30", p2, "P1")][2] > 0.0
    )
    h3_cells = [
        (c, p) for c in HEADLINE for p in (p2, p4) if diffs[(c, p, "hold")][1] > 0.0
    ]
    h3 = bool(h3_cells)
    print(
        f"\nH2  P2 vs P1: 11:00 {diffs[('11:00', p2, 'P1')][0]:+.2f} "
        f"[{diffs[('11:00', p2, 'P1')][1]:+.2f}, {diffs[('11:00', p2, 'P1')][2]:+.2f}], "
        f"13:30 {diffs[('13:30', p2, 'P1')][0]:+.2f} "
        f"[{diffs[('13:30', p2, 'P1')][1]:+.2f}, {diffs[('13:30', p2, 'P1')][2]:+.2f}] "
        f"-> {'SUPPORTED' if h2 else 'NOT SUPPORTED'}"
    )
    print(
        f"H3  headline cells beating hold with an interval above zero: {h3_cells or 'none'} "
        f"-> {'SUPPORTED' if h3 else 'NOT SUPPORTED'}"
    )
    return tab, {"H2": h2, "H3": h3}


def part_b_all_sessions(long: pd.DataFrame) -> pd.DataFrame:
    """Signal-free policies on every chain session (the holdout included)."""
    rows = []
    for clock in HEADLINE:
        x = long[long["clock"] == clock].dropna(subset=["hold", "atm_long_ask"])
        me = x["month_end"].to_numpy(bool)
        hold, la = x["hold"].to_numpy(), x["atm_long_ask"].to_numpy()
        for per, m in (
            ("deck period", (x["date"] <= "2024-04-30").to_numpy()),
            ("HOLDOUT", (x["date"] >= "2024-05-01").to_numpy()),
            ("all", np.ones(len(x), bool)),
        ):
            for name, v in (
                ("P0 hold", hold),
                ("P5 month-end override", np.where(me, la, hold)),
            ):
                rows.append(
                    {
                        "entry": clock,
                        "sample": per,
                        "policy": name,
                        "days": int(m.sum()),
                        "mean_pts": float(v[m].mean()),
                        "Sharpe": sharpe(v[m]),
                        "month_end_days_mean": float(v[m & me].mean()),
                    }
                )
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "b_month_end_all_sessions.csv", index=False)
    print("\n   the month-end override on every session (no signal needed)")
    print(tab.round(3).to_string(index=False))
    return tab


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    p43 = _load(HERE / "43_causal_entry_over_time.py", "p43_causal_entry")
    long, _ = build(p43)
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
    print(
        f"deck rows {len(d)} ({d['date'].nunique()} sessions x {d['clock'].nunique()} entry clocks); "
        f"buy-signal sessions {int(sig.sum())}"
    )
    _, _, h1 = part_a(d)
    _, h = part_b(d)
    part_b_all_sessions(long)
    print(
        f"\nSUMMARY  H1 (moneyness explains the signal's reach) {h1}; "
        f"H2 (at-the-money straddle beats buying back the held legs) {h['H2']}; "
        f"H3 (gamma-matched overlay beats the book) {h['H3']}"
    )


if __name__ == "__main__":
    main()
