"""34 - which volatility the hedge delta of the 11:00 straddle is computed with.

The book of record is proposal 32's: every session sells the nearest-OTM SPX
0DTE straddle at 11:00 ET, delta-hedges the package on the vendor spot every
30 minutes, and buys the straddle back at 15:30 at the QUOTED ask with the
entry taken at the quoted bid (fully crossed).  The "exit mid" variant buys
back at the quoted midpoint against an entry at the midpoint, and the "hold"
variant carries the package to cash settlement with the hedge running to the
official close.  Returns are one unit a day in units of the 11:00 midpoint
entry premium.

At every stamp t the book sets its hedge from the Black-76 package delta
``package_delta(total_vol_t, S_t, Kc, Kp)`` and holds it to the next stamp.
The book's ``total_vol_t`` is the IMPLIED one: the stamp's hourly implied
volatility (re-inverted by the live engine where the vendor's solver
censored it) times ``sqrt(hours to the 16:00 close)``.  This proposal asks
one question and nothing else: does a DIFFERENT volatility in that delta --
the repo's own remaining-window variance forecast instead of the implied
slice -- hedge the package better?

The forecast enters in the units section 5b of the intraday notebook already
uses.  At a stamp the panel carries ``rv_hat``, the forecast of the realized
variance of the bar ``[t, t + 30m]``, and ``w_slice``, the expanding
(63-session, one-day-lagged) mean of that clock's share of the session's
remaining-to-close realized variance.  ``rv_hat / w_slice`` is therefore a
forecast of the REMAINING-WINDOW variance, in exactly the units of the
implied ``iv_hourly^2 h_rem`` the book hedges with; its square root is a
total volatility that drops straight into ``package_delta``.

  V0  implied      the book: the re-inverted implied total volatility.
  V1  forecast     sqrt(rv_hat / w_slice), one variant per forecast
                   (the block-diagonal ridge, the a0 baseline, XGBoost and
                   LightGBM).
  V2  blend        sqrt(lambda V1^2 + (1 - lambda) V0^2), lambda in
                   BLEND_GRID, on the headline forecast.
  V3  max          max(V0, V1), headline forecast.
  V4  min          min(V0, V1), headline forecast.
  V5  calibrated   V1 with its VARIANCE multiplied by the expanding,
                   one-day-lagged, 63-session mean of the realized-over-
                   forecast remaining-window variance ratio at that clock,
                   so a biased forecast is corrected causally.
  V6  realized     NON-CAUSAL reference only: the total volatility of the
                   variance the session actually goes on to realize between
                   the stamp and the close.  It is the ceiling of what a
                   perfect variance forecast could do to the hedge; it is
                   never a candidate.

Every candidate is causal at the stamp.  A stamp whose forecast is missing,
or whose vendor implied volatility sits on the solver's bracket node, falls
back to V0 and is counted.  Entry premium, buy-back and hedge timing are
unchanged throughout: only the number inside ``package_delta`` differs.

The gate is the repo's standard and nothing new is computed until it passes:
proposal 32's published book numbers, the hold book's two published Sharpes,
proposal 32's trend/whipsaw tercile cut points, and -- the identity this
proposal turns on -- the hedge leg rebuilt here with the IMPLIED total
volatility reproduces ``live.ibkr.parity.replay_day``'s own hedge, at the
15:30 buy-back and at the settlement, to 1e-12.

Run:  python writeup/intraday_proposals/34_hedge_vol.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live.ibkr.parity import research_modules  # noqa: E402
from live.ibkr.pricing import package_delta  # noqa: E402


def _load_module(path: Path, name: str) -> Any:
    """The repo's read-only import: a script is imported by path, never edited."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HERE = Path(__file__).resolve().parent
p30 = _load_module(HERE / "30_skip_day_rule.py", "p34_p30")
p32 = _load_module(HERE / "32_loss_anatomy.py", "p34_p32")
asl = p30.asl

HOLD = p32.HOLD
OUT = HOLD / "proposals" / "34"

ENTRY = p32.ENTRY
CLOSE = p32.CLOSE
SESSION: tuple[str, ...] = p32.SESSION
H_REM: tuple[float, ...] = p32.H_REM
ERA0 = p32.ERA0
ANN = p32.ANN

#: The forecasts whose remaining-window variance is tried in the delta.
FORECAST_TAGS: tuple[str, ...] = ("blk2", "a0", "xgb", "lgbm")
#: The forecast the blends, the max/min and the calibration are built on.
HEADLINE_TAG = "blk2"
#: V2: the weight the forecast variance carries in the blended variance.
BLEND_GRID: tuple[float, ...] = (0.25, 0.50, 0.75)
#: V5: the expanding calibration's warm-up, the repo's 63-session standard
#: (the same ``min_periods`` the remaining-share profile itself uses).
CALIB_WARMUP = p30.WARMUP_SESSIONS

#: The books this proposal scores.  The first is the primary.
BOOKS: tuple[str, ...] = ("exit_crossed", "exit_mid", "hold")
PRIMARY_BOOK = "exit_crossed"
SAMPLES: tuple[str, ...] = ("whole", "daily_era")
#: The sample the verdict is read on.
VERDICT_SAMPLE = "daily_era"

#: The deck's circular block bootstrap: 2000 draws, 21-day blocks, seed 0.
BOOT_B = 2000
BOOT_BLOCK = 21
BOOT_SEED = 0
#: The percentile interval the verdict reads.
CI_PCT: tuple[float, float] = (2.5, 97.5)

#: Published reference numbers, reproduced before anything new is computed.
GATE_N_DAYS = p32.GATE_N_DAYS
GATE_CROSSED_WHOLE = p32.GATE_CROSSED_WHOLE
GATE_CROSSED_ERA = p32.GATE_CROSSED_ERA
GATE_TOL = p32.GATE_TOL
#: The hold-to-settlement book, midpoint and crossed (proposal 30's
#: REFERENCE_BOOKS, which are the standalone deck's own numbers).
GATE_HOLD_MID = 4.200674
GATE_HOLD_CROSSED = 3.582391
#: Proposal 32's full-sample trend/whipsaw tercile cut points, as published.
GATE_TREND_LO = 0.2076
GATE_TREND_HI = 0.4782
#: The cut points are published to four decimals, so they are gated there.
TREND_TOL = 5e-5
#: The hedge identity against the live engine is a float path, not a bar.
EXACT_TOL = 1e-12


# ------------------------------------------------------------------ show ----
def show(df: pd.DataFrame, title: str, cols: list[str] | None = None) -> None:
    """Print a table the way the other proposals print theirs."""
    use = df if cols is None else df[cols]
    print(f"\n{title}")
    with pd.option_context("display.width", 250, "display.max_columns", 80):
        print(use.to_string(index=False))


def write(df: pd.DataFrame, name: str, title: str) -> None:
    """Persist a table under the proposal's own directory."""
    df.to_csv(OUT / name, index=False)
    print(f"\nwrote {name}: {len(df)} rows, {len(df.columns)} columns  [{title}]")


# ------------------------------------------------------------- the grids ----
def clock_grid(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    """A forecast panel column on the (session, trade clock) grid.

    Exactly the transform ``30_skip_day_rule.panel_profile`` applies to
    ``rv_raw``: the panel's stamps are bar-end labelled, so the bar stamped
    ``t`` is the trade clock ``t - 30 minutes``, and only the fit mask's rows
    are kept.  Using the same transform for ``rv_hat`` as for ``rv_raw``
    keeps the forecast, the realized tape and the remaining-share profile on
    one grid; the gate below asserts that its 11:00 column reproduces
    proposal 30's own 11:00 signal for every forecast, exactly.
    """
    pf = panel.reset_index()
    pf = pf[pf["in_fit"].to_numpy(dtype=bool)].copy()
    clk = pd.to_datetime(pf["t"], utc=True).dt.tz_convert(
        "America/New_York"
    ) - pd.Timedelta(minutes=30)
    pf["pdate"] = clk.dt.normalize().dt.tz_localize(None)
    pf["phhmm"] = clk.dt.strftime("%H:%M")
    return pf.pivot_table(
        index="pdate", columns="phhmm", values=col, aggfunc="mean"
    ).sort_index()


def variance_grids(
    pkg: pd.DataFrame, panel: pd.DataFrame, dates: pd.DatetimeIndex
) -> dict[str, Any]:
    """The remaining-window variances every variant is built from.

    Returns, all on the book's days x the book's ten stamps:

    ``fc[tag]``   the forecast of the remaining-window variance,
                  ``rv_hat / w_slice``;
    ``realized``  the variance the session goes on to realize from the stamp
                  to the close (the sum of ``rv_raw`` over the remaining
                  trade clocks), NON-CAUSAL;
    ``scale``     the causal calibration of the headline forecast: the
                  expanding, one-day-lagged, 63-session mean of
                  realized / forecast at that clock, fitted on the whole
                  panel's session history;
    ``censored``  True where the stamp's vendor implied volatility sat on the
                  solver's bracket node.
    """
    clocks = sorted(pkg["hhmm"].unique())
    cols = list(SESSION)
    paths = asl.yhat_paths(ROOT)
    panels = {
        tag: asl.load_yhat_panel(paths[tag]).set_index("t") for tag in FORECAST_TAGS
    }
    prof, w_full = p30.panel_profile(panels[HEADLINE_TAG], clocks)

    # the realized remaining-window variance, on the panel's own sessions
    rem_full = pd.DataFrame(index=prof.index, columns=clocks, dtype=float)
    for i, c in enumerate(clocks):
        rem_full[c] = prof[list(clocks[i:])].sum(axis=1, min_count=len(clocks) - i)

    rvhat_full = {
        tag: clock_grid(pan, "rv_hat").reindex(columns=clocks)
        for tag, pan in panels.items()
    }
    fc_full = {tag: g / w_full[clocks] for tag, g in rvhat_full.items()}
    ratio = rem_full / fc_full[HEADLINE_TAG]
    scale_full = ratio.expanding(min_periods=CALIB_WARMUP).mean().shift(1)

    fc = {tag: g.reindex(index=dates, columns=cols) for tag, g in fc_full.items()}
    realized = rem_full.reindex(index=dates, columns=cols)
    scale = scale_full.reindex(index=dates, columns=cols)

    # ---- the gate: the 11:00 column is proposal 30's own 11:00 signal ------
    ref = p30.signals_1100(pkg, dates)
    work = pkg.copy()
    work["date"] = pd.to_datetime(work["date"])
    iv_raw = work.pivot_table(
        index="date", columns="hhmm", values="iv_hourly", aggfunc="first"
    ).reindex(index=dates, columns=cols)
    w_book = w_full.reindex(index=dates, columns=cols).to_numpy(float)
    slice_book = iv_raw.to_numpy(float) ** 2 * np.asarray(H_REM)[None, :] * w_book
    print(f"\nGATE - the all-stamp forecast grid against proposal 30's {ENTRY} signal")
    for tag in FORECAST_TAGS:
        rvh = rvhat_full[tag].reindex(index=dates, columns=cols).to_numpy(float)
        got = rvh[:, 0] - slice_book[:, 0]
        want = ref[f"s_{tag}"].to_numpy(float)
        ok = np.isfinite(got) & np.isfinite(want)
        assert (np.isfinite(got) == np.isfinite(want)).all(), tag
        assert np.array_equal(got[ok], want[ok]), tag
        print(
            f"  {tag:<6s} rv_hat - iv_hourly^2 h_rem w_slice at {ENTRY} reproduces "
            f"s_{tag} on all {int(ok.sum())} days, exactly  OK"
        )

    _, rule_mod = research_modules(ROOT)
    flag = rule_mod.on_vendor_node(
        panel["impl_volatility_c"]
    ) | rule_mod.on_vendor_node(panel["impl_volatility_p"])
    censored = (
        panel.assign(_cen=np.asarray(flag, dtype=float))
        .pivot_table(index="date", columns="hhmm", values="_cen", aggfunc="max")
        .reindex(index=dates, columns=cols)
        .fillna(1.0)
        .to_numpy(float)
        > 0
    )
    print(
        f"censored implied volatility: {int(censored.sum())} of {censored.size} "
        f"stamp-cells on the book's days (those stamps hedge with V0)"
    )
    print(
        f"calibration scale of {HEADLINE_TAG}: fitted on {int(scale_full.notna().any(axis=1).sum())} "
        f"panel sessions, min_periods {CALIB_WARMUP}, lagged one session; on the "
        f"book's days the median variance scale is "
        f"{float(np.nanmedian(scale.to_numpy(float))):.6f} (volatility scale "
        f"{float(np.sqrt(np.nanmedian(scale.to_numpy(float)))):.6f})"
    )
    for j, c in enumerate(SESSION):
        col = scale.to_numpy(float)[:, j]
        print(
            f"  {c}  median scale {np.nanmedian(col):.6f}   "
            f"stamps without a scale {int((~np.isfinite(col)).sum())}"
        )
    return {
        "fc": fc,
        "realized": realized,
        "scale": scale,
        "censored": censored,
        "w": w_book,
        "slice": slice_book,
    }


# ---------------------------------------------------------- the variants ----
def total_vol_variants(
    tape: dict[str, Any], grids: dict[str, Any]
) -> list[dict[str, Any]]:
    """Every total-volatility grid the hedge delta is tried with.

    ``var0`` is the book's own implied remaining-window variance; a variant's
    variance falls back to it wherever the variant's own input is missing or
    the stamp's implied volatility was censored, so every variant hedges the
    same days and differs from the book only where it has something to say.
    """
    var0 = (tape["sig"] * np.sqrt(np.asarray(H_REM)[None, :])) ** 2
    cen = grids["censored"]

    def usable(x: np.ndarray) -> np.ndarray:
        return np.isfinite(x) & (x > 0.0) & ~cen

    out: list[dict[str, Any]] = [
        {
            "variant": "V0_implied",
            "family": "V0 implied",
            "description": "the book: the re-inverted implied total volatility",
            "variance": var0,
            "n_fallback": 0,
            "causal": True,
        }
    ]
    head = grids["fc"][HEADLINE_TAG].to_numpy(float)
    ok_head = usable(head)
    for tag in FORECAST_TAGS:
        v = grids["fc"][tag].to_numpy(float)
        ok = usable(v)
        out.append(
            {
                "variant": f"V1_forecast_{tag}",
                "family": "V1 forecast",
                "description": f"sqrt(rv_hat / w_slice) of {tag}",
                "variance": np.where(ok, v, var0),
                "n_fallback": int((~ok).sum()),
                "causal": True,
            }
        )
    for lam in BLEND_GRID:
        out.append(
            {
                "variant": f"V2_blend_{lam:.2f}",
                "family": "V2 blend",
                "description": (
                    f"sqrt({lam:.2f} V1^2 + {1.0 - lam:.2f} V0^2), {HEADLINE_TAG}"
                ),
                "variance": np.where(
                    ok_head,
                    lam * np.where(ok_head, head, var0) + (1.0 - lam) * var0,
                    var0,
                ),
                "n_fallback": int((~ok_head).sum()),
                "causal": True,
            }
        )
    out.append(
        {
            "variant": "V3_max",
            "family": "V3 max",
            "description": f"max(V0, V1), {HEADLINE_TAG}",
            "variance": np.where(
                ok_head, np.maximum(np.where(ok_head, head, var0), var0), var0
            ),
            "n_fallback": int((~ok_head).sum()),
            "causal": True,
        }
    )
    out.append(
        {
            "variant": "V4_min",
            "family": "V4 min",
            "description": f"min(V0, V1), {HEADLINE_TAG}",
            "variance": np.where(
                ok_head, np.minimum(np.where(ok_head, head, var0), var0), var0
            ),
            "n_fallback": int((~ok_head).sum()),
            "causal": True,
        }
    )
    sc = grids["scale"].to_numpy(float)
    ok_cal = ok_head & np.isfinite(sc) & (sc > 0.0)
    out.append(
        {
            "variant": "V5_calibrated",
            "family": "V5 calibrated",
            "description": (
                f"V1 variance times its expanding lagged {CALIB_WARMUP}-session "
                f"realized-over-forecast mean, {HEADLINE_TAG}"
            ),
            "variance": np.where(
                ok_cal, np.where(ok_cal, sc, 1.0) * np.where(ok_cal, head, var0), var0
            ),
            "n_fallback": int((~ok_cal).sum()),
            "causal": True,
        }
    )
    rz = grids["realized"].to_numpy(float)
    ok_rz = usable(rz)
    out.append(
        {
            "variant": "V6_realized",
            "family": "V6 realized (non-causal)",
            "description": (
                "the variance actually realized from the stamp to the close "
                "(reference ceiling, NOT a candidate)"
            ),
            "variance": np.where(ok_rz, rz, var0),
            "n_fallback": int((~ok_rz).sum()),
            "causal": False,
        }
    )
    for row in out:
        row["total_vol"] = np.sqrt(row["variance"])
    return out


# ------------------------------------------------------------- the hedge ----
def traded_matrix(pos: np.ndarray) -> np.ndarray:
    """|delta traded| at each stamp: the opening trade, the rebalances, the flatten.

    ``pos[:, j]`` is the hedge held out of stamp ``j``; the extra last column
    is the flatten, which happens at the stamp after the last position.  This
    is proposal 32's Part C construction, one column wider.
    """
    n = pos.shape[0]
    prev = np.concatenate([np.zeros((n, 1)), pos[:, :-1]], axis=1)
    return np.abs(
        np.concatenate([pos, np.zeros((n, 1))], axis=1)
        - np.concatenate([prev, pos[:, -1:]], axis=1)
    )


def hedge_legs(total_vol: np.ndarray, tape: dict[str, Any]) -> dict[str, np.ndarray]:
    """The hedge P&L, the turnover and the hedge charge of one volatility grid.

    The delta at stamp ``j`` is the engine's own ``package_delta`` and is
    held to stamp ``j + 1``.  The exit book's hedge stops at the 15:30
    buy-back; the hold book carries the 15:30 delta into the official close,
    which is exactly the step ``replay_day`` takes when it is told to hold.
    The charge is the repo's own ``HEDGE_COST_BP`` basis points of
    ``S x |delta traded|``; it is REPORTED, never taken out of a book (the
    published books are gross of it).
    """
    spot = tape["spot"]
    kc = tape["Kc"]
    kp = tape["Kp"]
    n, m = total_vol.shape
    dlt = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dlt[i, j] = package_delta(total_vol[i, j], spot[i, j], kc[i], kp[i])
    d_spot = spot[:, 1:] - spot[:, :-1]
    hedge_exit = (dlt[:, :-1] * d_spot).sum(axis=1)
    hedge_hold = hedge_exit + dlt[:, -1] * (tape["S_close"] - spot[:, -1])
    bp = p32.HEDGE_COST_BP * 1e-4
    tr_exit = traded_matrix(dlt[:, :-1])
    tr_hold = traded_matrix(dlt)
    px_hold = np.concatenate([spot, tape["S_close"][:, None]], axis=1)
    return {
        "delta": dlt,
        "hedge_exit": hedge_exit,
        "hedge_hold": hedge_hold,
        "turnover_exit": tr_exit.sum(axis=1),
        "turnover_hold": tr_hold.sum(axis=1),
        "cost_exit": (tr_exit * spot).sum(axis=1) * bp,
        "cost_hold": (tr_hold * px_hold).sum(axis=1) * bp,
    }


def book_returns(
    tape: dict[str, Any], mid: np.ndarray, ask: np.ndarray, legs: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """The three books' daily returns under one hedge."""
    em = tape["entry_mid"]
    eb = tape["entry_bid"]
    return {
        "exit_crossed": (-(ask[:, -1] - eb) + legs["hedge_exit"]) / em,
        "exit_mid": (-(mid[:, -1] - em) + legs["hedge_exit"]) / em,
        "hold": (-(tape["settle"] - em) + legs["hedge_hold"]) / em,
    }


def book_turnover(legs: dict[str, np.ndarray], book: str) -> np.ndarray:
    """The book's own hedge turnover (the hold book flattens at the close)."""
    return legs["turnover_hold"] if book == "hold" else legs["turnover_exit"]


def book_cost(legs: dict[str, np.ndarray], book: str) -> np.ndarray:
    """The book's own hedge charge in index points, at the repo's basis points."""
    return legs["cost_hold"] if book == "hold" else legs["cost_exit"]


# -------------------------------------------------------------- the gate ----
def gate(
    tape: dict[str, Any],
    mid: np.ndarray,
    ask: np.ndarray,
    dates: pd.DatetimeIndex,
    day: pd.DataFrame,
    legs0: dict[str, np.ndarray],
) -> tuple[np.ndarray, float, float]:
    """Reproduce every published number, and the engine's own hedge, first."""
    p32.gate(tape, mid, ask, dates)
    em = tape["entry_mid"]
    eb = tape["entry_bid"]
    r_hold_mid = pd.Series(
        (-(tape["settle"] - em) + tape["hedge_hold"]) / em, index=dates
    )
    r_hold_x = pd.Series(
        (-(tape["settle"] - eb) + tape["hedge_hold"]) / em, index=dates
    )
    print("\nGATE - the hold-to-settlement book")
    for label, have, want in (
        ("hold Sharpe_ann mid", p32.sharpe_ann(r_hold_mid), GATE_HOLD_MID),
        ("hold Sharpe_ann crossed", p32.sharpe_ann(r_hold_x), GATE_HOLD_CROSSED),
    ):
        assert abs(have - want) < GATE_TOL, (label, have, want)
        print(f"  {label:<26s} {have:>16.12f}  reference {want:.6f}  OK")

    print("\nGATE - the hedge leg rebuilt with the implied total volatility")
    d_exit = float(np.max(np.abs(legs0["hedge_exit"] - tape["hedge_ref"])))
    d_hold = float(np.max(np.abs(legs0["hedge_hold"] - tape["hedge_hold"])))
    assert d_exit < EXACT_TOL, d_exit
    assert d_hold < EXACT_TOL, d_hold
    print(
        f"  sum delta dS to the {CLOSE} buy-back matches replay_day to {d_exit:.3e}  OK"
    )
    print(f"  the same sum carried into the settlement matches to {d_hold:.3e}  OK")
    for book, ref in (
        ("exit_crossed", GATE_CROSSED_WHOLE),
        ("exit_mid", float("nan")),
        ("hold", GATE_HOLD_MID),
    ):
        r = pd.Series(book_returns(tape, mid, ask, legs0)[book], index=dates)
        have = p32.sharpe_ann(r)
        if np.isfinite(ref):
            assert abs(have - ref) < GATE_TOL, (book, have, ref)
            print(f"  book {book:<13s} Sharpe_ann whole {have:>16.12f}  OK")
        else:
            print(
                f"  book {book:<13s} Sharpe_ann whole {have:>16.12f}  "
                f"(the quoted-midpoint exit, context, not a published number)"
            )
    era = np.asarray(dates >= ERA0, dtype=bool)
    r_era = pd.Series(
        book_returns(tape, mid, ask, legs0)[PRIMARY_BOOK][era], index=dates[era]
    )
    assert abs(p32.sharpe_ann(r_era) - GATE_CROSSED_ERA) < GATE_TOL
    print(
        f"  book {PRIMARY_BOOK} Sharpe_ann era {p32.sharpe_ann(r_era):.12f}  "
        f"reference {GATE_CROSSED_ERA:.12f}  OK"
    )

    q1, q2 = day["trend_ratio"].quantile([1 / 3, 2 / 3]).to_numpy()
    assert abs(float(q1) - GATE_TREND_LO) < TREND_TOL, q1
    assert abs(float(q2) - GATE_TREND_HI) < TREND_TOL, q2
    lab = np.full(len(dates), "2_mixed", dtype=object)
    lab[day["trend_ratio"].to_numpy(float) <= q1] = "1_choppy"
    lab[day["trend_ratio"].to_numpy(float) > q2] = "3_trend"
    print(
        "\nGATE - proposal 32's shape terciles of |sum of bar returns| / sum |bar returns|"
    )
    print(
        f"  cut points {float(q1):.6f} / {float(q2):.6f}  reference "
        f"{GATE_TREND_LO:.4f} / {GATE_TREND_HI:.4f}  OK; "
        f"{int((lab == '1_choppy').sum())} choppy, {int((lab == '2_mixed').sum())} "
        f"mixed, {int((lab == '3_trend').sum())} trend days"
    )
    print("GATE PASSED")
    return lab, float(q1), float(q2)


# ------------------------------------------------------------ statistics ----
def stat_row(x: np.ndarray, dates: pd.DatetimeIndex) -> dict[str, Any]:
    """n / mean / sd / Sharpe_ann / t / MaxDD / worst day of a daily series."""
    ok = np.isfinite(x)
    v = x[ok]
    d = dates[ok]
    n = int(v.size)
    mean = float(v.mean()) if n else float("nan")
    sd = float(v.std(ddof=1)) if n >= 2 else float("nan")
    live = bool(np.isfinite(sd)) and sd > 0.0
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "Sharpe_ann": mean / sd * ANN if live else float("nan"),
        "t": float(np.sqrt(n)) * mean / sd if live else float("nan"),
        "MaxDD": p32.maxdd(v),
        "worst_day": float(v.min()) if n else float("nan"),
        "worst_date": str(pd.Timestamp(d[int(np.argmin(v))]).date()) if n else "",
    }


_BOOT_IDX: dict[int, np.ndarray] = {}


def boot_idx(n: int) -> np.ndarray:
    """The deck's circular block bootstrap index, drawn once per sample size."""
    if n not in _BOOT_IDX:
        _BOOT_IDX[n] = asl.circular_block_bootstrap_idx(
            np.random.default_rng(BOOT_SEED), n, BOOT_BLOCK, BOOT_B
        )
    return _BOOT_IDX[n]


def sharpe_diff_ci(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Circular block bootstrap of Sharpe(x) - Sharpe(y) on the common days.

    The two series are resampled on the SAME drawn index, so the interval is
    of the paired difference, not of two independent Sharpes.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    a = x[ok]
    b = y[ok]
    n = int(a.size)
    hat = (a.mean() / a.std(ddof=1) - b.mean() / b.std(ddof=1)) * ANN
    idx = boot_idx(n)
    aa = a[idx]
    bb = b[idx]
    ds = (
        aa.mean(axis=1) / aa.std(axis=1, ddof=1)
        - bb.mean(axis=1) / bb.std(axis=1, ddof=1)
    ) * ANN
    lo, hi = np.percentile(ds, list(CI_PCT))
    return {
        "n": n,
        "dSharpe": float(hat),
        "boot_lo": float(lo),
        "boot_hi": float(hi),
        "pct_draws_positive": float(100.0 * (ds > 0).mean()),
        "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def paired_row(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """The paired daily difference x - y: mean, plain t, HAC t, days improved."""
    ok = np.isfinite(x) & np.isfinite(y)
    d = x[ok] - y[ok]
    n = int(d.size)
    sd = float(d.std(ddof=1)) if n >= 2 else float("nan")
    hac_t, lag = asl.newey_west_t(pd.Series(d))
    return {
        "n": n,
        "mean_diff": float(d.mean()) if n else float("nan"),
        "sd_diff": sd,
        "t_diff": float(np.sqrt(n)) * float(d.mean()) / sd
        if (n and np.isfinite(sd) and sd > 0.0)
        else float("nan"),
        "hac_t_diff": float(hac_t),
        "hac_lag": int(lag),
        "frac_days_improved": float((d > 0).mean()) if n else float("nan"),
        "frac_days_unchanged": float((d == 0).mean()) if n else float("nan"),
    }


# ------------------------------------------------------ forecast quality ----
def qlike(y: np.ndarray, f: np.ndarray) -> tuple[float, float, int, float]:
    """Patton's QLIKE ``y/f - log(y/f) - 1`` and the log-ratio bias ``log(f/y)``."""
    m = np.isfinite(y) & np.isfinite(f) & (y > 0.0) & (f > 0.0)
    if int(m.sum()) < 2:
        return float("nan"), float("nan"), int(m.sum()), float("nan")
    r = y[m] / f[m]
    return (
        float(np.mean(r - np.log(r) - 1.0)),
        float(np.mean(np.log(f[m]) - np.log(y[m]))),
        int(m.sum()),
        float(f[m].mean() / y[m].mean()),
    )


def forecast_quality(
    grids: dict[str, Any], tape: dict[str, Any], dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """QLIKE and log-ratio bias of every remaining-window forecast, per clock.

    The target is the variance the session actually realizes from the stamp
    to the close.  The implied row is the SAME object the book hedges with,
    ``iv_hourly_used^2 h_rem``, so the reader sees on one scale whether a
    forecast is closer to the realized remaining window than the implied
    slice the book already uses.
    """
    y = grids["realized"].to_numpy(float)
    var0 = (tape["sig"] * np.sqrt(np.asarray(H_REM)[None, :])) ** 2
    era = np.asarray(dates >= ERA0, dtype=bool)
    rows: list[dict[str, Any]] = []
    sources: list[tuple[str, np.ndarray]] = [
        (tag, grids["fc"][tag].to_numpy(float)) for tag in FORECAST_TAGS
    ]
    sources.append(("implied", var0))
    for name, f in sources:
        for sample, m in (
            ("whole", np.ones(len(dates), dtype=bool)),
            ("daily_era", era),
        ):
            for j, clock in enumerate(SESSION):
                q, bias, n, ratio = qlike(y[m, j], f[m, j])
                rows.append(
                    {
                        "forecast": name,
                        "sample": sample,
                        "clock": clock,
                        "h_rem": H_REM[j],
                        "n": n,
                        "QLIKE": q,
                        "bias_log_f_over_realized": bias,
                        "mean_f_over_mean_realized": ratio,
                        "median_forecast": float(np.nanmedian(f[m, j])),
                        "median_realized": float(np.nanmedian(y[m, j])),
                    }
                )
            q, bias, n, ratio = qlike(y[m].ravel(), f[m].ravel())
            rows.append(
                {
                    "forecast": name,
                    "sample": sample,
                    "clock": "pooled",
                    "h_rem": float("nan"),
                    "n": n,
                    "QLIKE": q,
                    "bias_log_f_over_realized": bias,
                    "mean_f_over_mean_realized": ratio,
                    "median_forecast": float(np.nanmedian(f[m])),
                    "median_realized": float(np.nanmedian(y[m])),
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main ----
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel, ent, dates = p32.load_panel()
    assert len(dates) == GATE_N_DAYS, (len(dates), GATE_N_DAYS)
    tape = p32.build_tape(panel, ent, dates)
    _, mid, ask = p32.quote_grids(ent, dates)
    assert np.max(np.abs(mid[:, 0] - tape["entry_mid"])) < EXACT_TOL, "11:00 mid"
    _, day = p32.attribute(tape, mid, ask, dates)

    var0 = (tape["sig"] * np.sqrt(np.asarray(H_REM)[None, :])) ** 2
    legs0 = hedge_legs(np.sqrt(var0), tape)
    lab, q1, q2 = gate(tape, mid, ask, dates, day, legs0)

    pkg = pd.read_parquet(sorted((HOLD / "cache").glob("trade_*.parquet"))[-1])
    grids = variance_grids(pkg, panel, dates)
    variants = total_vol_variants(tape, grids)
    causal = [v for v in variants if v["causal"] and v["variant"] != "V0_implied"]
    print(
        f"\n{len(variants)} total-volatility variants: V0 (the book), "
        f"{len(causal)} causal candidates and 1 non-causal reference; "
        f"{len(FORECAST_TAGS)} forecasts, blend grid "
        f"{', '.join(f'{x:.2f}' for x in BLEND_GRID)}, headline forecast "
        f"{HEADLINE_TAG}"
    )

    era = np.asarray(dates >= ERA0, dtype=bool)
    masks: dict[str, np.ndarray] = {
        "whole": np.ones(len(dates), dtype=bool),
        "daily_era": era,
    }
    legs = {v["variant"]: hedge_legs(v["total_vol"], tape) for v in variants}
    rets = {
        v["variant"]: book_returns(tape, mid, ask, legs[v["variant"]]) for v in variants
    }
    base = rets["V0_implied"]

    # -------------------------------------------------------- variants.csv ---
    vrows: list[dict[str, Any]] = []
    for v in variants:
        name = v["variant"]
        for book in BOOKS:
            turn = book_turnover(legs[name], book)
            cost = book_cost(legs[name], book) / tape["entry_mid"]
            r = rets[name][book]
            d = r - base[book]
            for sample, m in masks.items():
                row: dict[str, Any] = {
                    "variant": name,
                    "family": v["family"],
                    "causal": v["causal"],
                    "description": v["description"],
                    "book": book,
                    "sample": sample,
                    "n_fallback_stamps": v["n_fallback"],
                }
                row.update(stat_row(r[m], dates[m]))
                row["sd_diff_vs_V0"] = (
                    float(np.std(d[m], ddof=1)) if int(m.sum()) >= 2 else float("nan")
                )
                row["mean_abs_diff_vs_V0"] = float(np.mean(np.abs(d[m])))
                row["mean_turnover"] = float(np.mean(turn[m]))
                row["mean_hedge_cost_prem"] = float(np.mean(cost[m]))
                for cls, tag in (("1_choppy", "choppy"), ("3_trend", "trend")):
                    sel = m & (lab == cls)
                    sub = r[sel]
                    row[f"mean_{tag}"] = float(np.mean(sub))
                    row[f"Sharpe_{tag}"] = p32.sharpe_ann(pd.Series(sub))
                    row[f"n_{tag}"] = int(sel.sum())
                vrows.append(row)
    variants_df = pd.DataFrame(vrows)
    write(variants_df, "variants.csv", "the twelve hedge volatilities, scored")
    vcols = [
        "variant",
        "n",
        "mean",
        "sd",
        "Sharpe_ann",
        "t",
        "MaxDD",
        "worst_day",
        "worst_date",
        "sd_diff_vs_V0",
        "mean_abs_diff_vs_V0",
        "mean_turnover",
        "mean_hedge_cost_prem",
        "n_fallback_stamps",
        "mean_choppy",
        "Sharpe_choppy",
        "mean_trend",
        "Sharpe_trend",
    ]
    for book in BOOKS:
        for sample in SAMPLES:
            sub = variants_df[
                (variants_df["book"] == book) & (variants_df["sample"] == sample)
            ]
            show(sub, f"variants.csv  |  book {book}  |  sample {sample}", vcols)
    show(
        variants_df[
            ["variant", "family", "causal", "description", "n_fallback_stamps"]
        ].drop_duplicates("variant"),
        "variants.csv  |  what each variant puts inside package_delta",
    )
    print(
        f"\nmean_hedge_cost_prem is the repo's {p32.HEDGE_COST_BP} bp charge on "
        f"S x |delta traded|, in units of the entry premium.  It is REPORTED, not "
        f"taken out of any book: every Sharpe, mean and difference in these tables "
        f"is gross of it, exactly as the published books are."
    )

    # ----------------------------------------------------------- pairs.csv ---
    prows: list[dict[str, Any]] = []
    for v in variants:
        if v["variant"] == "V0_implied":
            continue
        for book in BOOKS:
            for sample, m in masks.items():
                row = {
                    "variant": v["variant"],
                    "family": v["family"],
                    "causal": v["causal"],
                    "book": book,
                    "sample": sample,
                }
                row.update(paired_row(rets[v["variant"]][book][m], base[book][m]))
                prows.append(row)
    pairs_df = pd.DataFrame(prows)
    write(pairs_df, "pairs.csv", "the paired daily difference against V0")
    pcols = [
        "variant",
        "n",
        "mean_diff",
        "sd_diff",
        "t_diff",
        "hac_t_diff",
        "hac_lag",
        "frac_days_improved",
        "frac_days_unchanged",
    ]
    for book in BOOKS:
        for sample in SAMPLES:
            sub = pairs_df[(pairs_df["book"] == book) & (pairs_df["sample"] == sample)]
            show(sub, f"pairs.csv  |  book {book}  |  sample {sample}", pcols)

    # ------------------------------------------------------- bootstrap.csv ---
    brows: list[dict[str, Any]] = []
    for v in variants:
        if v["variant"] == "V0_implied":
            continue
        for book in BOOKS:
            for sample, m in masks.items():
                row = {
                    "variant": v["variant"],
                    "family": v["family"],
                    "causal": v["causal"],
                    "book": book,
                    "sample": sample,
                    "B": BOOT_B,
                    "block": BOOT_BLOCK,
                    "seed": BOOT_SEED,
                    "ci_pct_lo": CI_PCT[0],
                    "ci_pct_hi": CI_PCT[1],
                }
                row.update(sharpe_diff_ci(rets[v["variant"]][book][m], base[book][m]))
                is_verdict_row = book == PRIMARY_BOOK and sample == VERDICT_SAMPLE
                if not v["causal"]:
                    row["verdict"] = (
                        "reference only (non-causal)" if is_verdict_row else ""
                    )
                elif is_verdict_row:
                    row["verdict"] = (
                        "improves"
                        if row["ci_excludes_zero"] and row["dSharpe"] > 0
                        else "no evidence"
                    )
                else:
                    row["verdict"] = ""
                brows.append(row)
    boot_df = pd.DataFrame(brows)
    write(boot_df, "bootstrap.csv", "the bootstrap interval of the Sharpe difference")
    bcols = [
        "variant",
        "n",
        "dSharpe",
        "boot_lo",
        "boot_hi",
        "pct_draws_positive",
        "ci_excludes_zero",
        "verdict",
    ]
    for book in BOOKS:
        for sample in SAMPLES:
            sub = boot_df[(boot_df["book"] == book) & (boot_df["sample"] == sample)]
            show(
                sub,
                f"bootstrap.csv  |  book {book}  |  sample {sample}  |  "
                f"{BOOT_B} circular block draws, block {BOOT_BLOCK}, seed {BOOT_SEED}",
                bcols,
            )

    # ------------------------------------------------ forecast_quality.csv ---
    fq = forecast_quality(grids, tape, dates)
    write(
        fq,
        "forecast_quality.csv",
        "QLIKE and log-ratio bias against the realized remaining window",
    )
    for sample in SAMPLES:
        sub = fq[fq["sample"] == sample]
        show(sub, f"forecast_quality.csv  |  sample {sample}")
        print(f"\nQLIKE by clock, sample {sample} (lower is better)")
        with pd.option_context("display.width", 250, "display.max_columns", 80):
            print(
                sub.pivot(index="clock", columns="forecast", values="QLIKE").to_string(
                    float_format=lambda x: f"{x:.5f}"
                )
            )
        print(f"\nlog-ratio bias log(forecast / realized) by clock, sample {sample}")
        with pd.option_context("display.width", 250, "display.max_columns", 80):
            print(
                sub.pivot(
                    index="clock", columns="forecast", values="bias_log_f_over_realized"
                ).to_string(float_format=lambda x: f"{x:+.5f}")
            )

    # ------------------------------------------------------------ verdicts ---
    verdicts = boot_df[
        (boot_df["book"] == PRIMARY_BOOK) & (boot_df["sample"] == VERDICT_SAMPLE)
    ]
    show(
        verdicts,
        f"VERDICT  |  primary book {PRIMARY_BOOK}, sample {VERDICT_SAMPLE}: "
        f"'improves' only where the {int(CI_PCT[1] - CI_PCT[0])}% bootstrap interval "
        f"of the Sharpe difference excludes zero",
        ["variant", "family", "causal", "dSharpe", "boot_lo", "boot_hi", "verdict"],
    )
    n_improve = int((verdicts["verdict"] == "improves").sum())
    print(
        f"\n{len(causal)} causal variants were tried against V0 "
        f"({len(variants)} total-volatility grids in all, of which one is the book "
        f"and one is the non-causal realized reference); {n_improve} of them improve "
        f"the primary book in the era by the stated bar, {len(causal) - n_improve} "
        f"return no evidence."
    )
    print(
        f"\nshape terciles used for the choppy / trend columns: "
        f"{q1:.6f} / {q2:.6f} of |sum of bar returns| / sum |bar returns|"
    )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
