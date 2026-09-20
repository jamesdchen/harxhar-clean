"""39 - the hedge-density bound: what hedging more often would remove, and cost.

The book of record is proposal 32's: every session sells the nearest-OTM SPX
0DTE straddle at 11:00 ET, delta-hedges the package on the vendor spot every
30 minutes, and buys the straddle back at 15:30 at the QUOTED ask with the
entry taken at the quoted bid (fully crossed).  Returns are one unit a day in
units of the 11:00 midpoint entry premium.  The secondary book holds the
straddle through cash settlement at 16:00 and hedges the 15:30-16:00 stretch
as well (``parity.replay_day`` with ``exit_clock=None``).

The question is how much of the gamma bill the 30-minute cadence pays is a
HEDGING ERROR -- the part a denser hedge would not have paid -- and what the
denser hedge would cost in turnover.  There is no minute tape of the hedge
instrument in this repo, so the bound is built from the two minute-level SUMS
the 30-minute index panel already carries (``data/core_stats.parquet``):
``sumret2``, the sum of squared one-minute log returns in the bar (the
paper's realized variance, ``writeup/sections/data_prep.tex`` line 67), and
``sumabsret``, the sum of their absolute values.

Part A  the density bound.  Over bar ``[t, t + 30]`` the short's exact
        30-minute P&L carries the tape's own convexity term, which is the
        full re-priced value change at a frozen volatility net of the hedge.
        The MINUTE-HEDGED book replaces the SECOND-ORDER part of that term,
        ``0.5 G_t (dS)^2``, by ``0.5 G_t S_t^2 sumret2_t`` -- the same gamma,
        held at its start-of-bar value, applied to the bar's realized
        variance instead of to its endpoint move.  The substitution is
        therefore, per bar and exactly as implemented,

            P&L_minute = P&L_30 + 0.5 G_t (dS_t)^2 - 0.5 G_t S_t^2 sumret2_t

        so the two agree in expectation whenever E[(dS)^2] = S^2 E[sumret2],
        which is the martingale statement the table checks directly.  Every
        higher-order piece of the tape's convexity term, the theta term, the
        volatility-path term and both quote-versus-mark basis terms are
        carried over UNCHANGED.  APPROXIMATIONS, all of them stated again at
        the point of use: the gamma is frozen at the stamp (Part C measures
        the drift), the delta is re-set at minute closes at no slippage, the
        hedge is on the index rather than on ES (no basis), and the minute
        returns come from the index panel while ``dS`` comes from the book's
        own vendor spot tape.

        Cadences between the two anchors cannot be reconstructed from sums.
        They are INTERPOLATED under the stated model that the hedging-error
        variance falls like 1/N in the number of sub-intervals N = 30 / k:

            return_k = return_30 + (return_1 - return_30) (1 - 1/N)

        and, because the total variation of a Brownian path sampled N times
        grows like sqrt(N) while the table has only the two anchors, the
        turnover is interpolated log-linearly between them,

            turnover_k = turnover_30 (turnover_1 / turnover_30)^(log N/log 30)

        which is exact at both anchors.  Both are labelled INTERPOLATION in
        every table; only k = 30 and k = 1 are built.

Part B  the forecast-conditional density.  Proposal 35's causal combined
        next-bar forecaster gives ``r_t = f_comb_t / slice_t``; the rules
        hedge at one minute only where ``r_t`` is large and at 30 minutes
        elsewhere, and the controls do the reverse.  The same is run on the
        ridge's own ratio and on the implied slice's level alone.

Part C  what the minute data cannot be replaced by: the within-bar gamma and
        delta drift the frozen-gamma approximation ignores, the ES basis, the
        minute fill prices, and the exact dataset that would settle it.

Gates, all before anything new is computed: proposal 32's published book
numbers (n = 865, crossed-quoted Sharpe 2.2525467 whole / 2.5131322 era), the
per-bar hedge leg against ``parity.replay_day``'s own replays to 1e-12 for
both books, the 30-minute turnover against proposal 32's own construction to
1e-12, and proposal 35's pooled QLIKE of the combined forecaster (0.150514
whole / 0.138981 era).

Run:  python writeup/intraday_proposals/39_hedge_density.py
"""

from __future__ import annotations

import importlib.util
import sys
from math import log, pi, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(path: Path, name: str) -> Any:
    """The repo's read-only import: a script is imported by path, never edited."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HERE = Path(__file__).resolve().parent
p35 = _load_module(HERE / "35_combined_forecast_toggle.py", "p39_p35")
p30 = p35.p30
p32 = p35.p32
asl = p35.asl

from live.ibkr.pricing import package_delta  # noqa: E402

HOLD = p32.HOLD
OUT = HOLD / "proposals" / "39"
PANEL = ROOT / "data" / "core_stats.parquet"

SESSION: tuple[str, ...] = p32.SESSION
H_REM: tuple[float, ...] = p32.H_REM
ERA0 = p32.ERA0
ANN = p32.ANN
WARMUP = p30.WARMUP_SESSIONS

#: The bar-END stamps of the ten bars this proposal prices: the nine bars the
#: 11:00 book lives on (11:00-11:30 .. 15:00-15:30) and the settlement stretch
#: 15:30-16:00 the hold book adds.  The panel is bar-END labelled, so the bar
#: [t, t + 30] is the panel row stamped t + 30.
BAR_START: tuple[str, ...] = (*SESSION[:-1], "15:30")
BAR_END: tuple[str, ...] = (*SESSION[1:], "16:00")
#: Bars of the primary (exit 15:30) book and of the hold book.
N_BAR_EXIT = len(SESSION) - 1
N_BAR_HOLD = N_BAR_EXIT + 1

#: The hedge cadences the ladder reports, in minutes.  30 and 1 are built;
#: everything between them is the stated interpolation.
CADENCE_MINUTES: tuple[int, ...] = (30, 15, 10, 5, 3, 2, 1)
#: Turnover charges, basis points of S x |delta traded|.  0.5 bp is the repo's
#: figure (``32_loss_anatomy.HEDGE_COST_BP``); 0.2 bp is passive ES at scale
#: and 1.0 bp is MES.
COST_BP_GRID: tuple[float, ...] = (0.2, 0.5, 1.0)
#: The cost level the headline net numbers and the bootstrap are read at.
HEADLINE_BP = p32.HEDGE_COST_BP

#: The deck's circular block bootstrap: 2000 draws, 21-day blocks, seed 0.
BOOT_B = p35.BOOT_B
BOOT_BLOCK = p35.BOOT_BLOCK
BOOT_SEED = p35.BOOT_SEED
CI_PCT: tuple[float, float] = p35.CI_PCT

#: Part B's ratio cut points, f_comb_t / slice_t.
Q_GRID: tuple[float, ...] = (0.8, 1.0, 1.25, 1.5)
#: The expanding, lagged tercile rules' quantiles.
TERCILE_HI = 2.0 / 3.0
TERCILE_LO = 1.0 / 3.0

#: Proposal 35's published pooled QLIKE of the combined forecaster.
GATE_QLIKE_WHOLE = 0.150514
GATE_QLIKE_ERA = 0.138981
GATE_QLIKE_TOL = 5e-7
#: Tape identities are float paths.
EXACT_TOL = 1e-12

#: Part C: the 1-minute ES dataset that would settle the bound.
ES_VENDOR = "Databento GLBX.MDP3 ES front-month ohlcv-1m"
ES_FROM = pd.Timestamp("2020-01-02")
ES_TO = pd.Timestamp("2024-04-30")
#: RTH minutes from 11:00 to 16:00 ET, the stretch the book is exposed over.
ES_MINUTES_PER_SESSION = 300


# ------------------------------------------------------------------ show ----
def show(df: pd.DataFrame, title: str, cols: list[str] | None = None) -> None:
    """Print a table the way the other proposals print theirs."""
    use = df if cols is None else df[cols]
    print(f"\n{title}")
    with pd.option_context("display.width", 250, "display.max_columns", 80):
        print(use.to_string(index=False))


def write(
    df: pd.DataFrame, name: str, title: str, cols: list[str] | None = None
) -> None:
    """Persist a table under the proposal's own directory and print it."""
    df.to_csv(OUT / name, index=False)
    show(df, f"{title}   [{name}]", cols)


# ------------------------------------------------------------- the greeks ---
def package_gamma(
    total_vol: np.ndarray, f: np.ndarray, kc: np.ndarray, kp: np.ndarray
) -> np.ndarray:
    """Black-76 package gamma, r = 0 and F = S, in index units per straddle.

    ``live.ibkr.pricing.package_delta`` is ``N(d1c) + N(d1p) - 1`` with
    ``d1 = (log(F / K) + s^2 / 2) / s``, so its derivative in the underlying
    is ``[phi(d1c) + phi(d1p)] / (S s)`` with ``s`` the TOTAL volatility of
    the stamp.  Zero wherever the engine's delta is zero (non-positive or
    non-finite total volatility, non-positive underlying).
    """
    ok = np.isfinite(total_vol) & (total_vol > 0.0) & np.isfinite(f) & (f > 0.0)
    s = np.where(ok, total_vol, 1.0)
    ff = np.where(ok, f, 1.0)
    d1c = (np.log(ff / kc) + 0.5 * s * s) / s
    d1p = (np.log(ff / kp) + 0.5 * s * s) / s
    phi = np.exp(-0.5 * d1c * d1c) + np.exp(-0.5 * d1p * d1p)
    return np.where(ok, phi / (sqrt(2.0 * pi) * ff * s), 0.0)


def stamp_greeks(tape: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Package delta and gamma at each of the ten 30-minute stamps.

    The delta is the live engine's own, stamp by stamp, so the row sums of
    ``delta_j (S_{j+1} - S_j)`` are the hedge legs ``replay_day`` accumulates.
    """
    spot = tape["spot"]
    tot = tape["sig"] * np.sqrt(np.asarray(H_REM)[None, :])
    kc = tape["Kc"]
    kp = tape["Kp"]
    n, m = spot.shape
    delta = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            delta[i, j] = package_delta(tot[i, j], spot[i, j], kc[i], kp[i])
    gamma = package_gamma(tot, spot, kc[:, None], kp[:, None])
    return delta, gamma


# -------------------------------------------------------- the minute sums ---
def minute_sums(dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """The 30-minute index panel's minute-level sums on the book's ten bars.

    ``data/core_stats.parquet`` is index-free and bar-END labelled in naive
    ET, so the bar ``[t, t + 30]`` is the row stamped ``t + 30``; the stamps
    are parsed WITHOUT ``utc=True`` (the panel clock is naive ET).
    """
    raw = pd.read_parquet(
        PANEL, columns=["endbartime", "sumret", "sumabsret", "sumret2", "numobs"]
    )
    stamp = pd.to_datetime(raw["endbartime"])
    raw = raw.assign(pdate=stamp.dt.normalize(), phhmm=stamp.dt.strftime("%H:%M"))
    out: dict[str, np.ndarray] = {}
    for col in ("sumret", "sumabsret", "sumret2", "numobs"):
        grid = raw.pivot_table(
            index="pdate", columns="phhmm", values=col, aggfunc="first"
        ).reindex(index=dates, columns=list(BAR_END))
        out[col] = grid.to_numpy(float)
    miss = int((~np.isfinite(out["sumret2"])).sum())
    assert miss == 0, f"{miss} of the book's bars carry no panel row"
    short = int((out["numobs"] < 30.0).sum())
    print(
        f"\nminute sums: {out['sumret2'].shape[0]} sessions x "
        f"{out['sumret2'].shape[1]} bars ({BAR_END[0]}..{BAR_END[-1]} bar-end "
        f"stamps, naive ET) joined from {PANEL.name}; every bar carries a "
        f"panel row; {short} of them carry fewer than 30 one-minute "
        f"observations"
    )
    return out


# ------------------------------------------------------------- the books ----
def exit_spec(
    g: dict[str, Any], delta: np.ndarray, gamma: np.ndarray, sums: dict[str, np.ndarray]
) -> dict[str, Any]:
    """The primary book: entry 11:00 quoted bid, buy-back 15:30 quoted ask."""
    spot = g["spot"]
    em = g["em"]
    r_base = (-(g["ask"][:, -1] - g["eb"]) + g["tape"]["hedge_ref"]) / em
    return {
        "book": "exit_1530_crossed",
        "spot": spot,
        "gamma": gamma,
        "delta": delta,
        "dS": spot[:, 1:] - spot[:, :-1],
        "sumret2": sums["sumret2"][:, :N_BAR_EXIT],
        "sumabsret": sums["sumabsret"][:, :N_BAR_EXIT],
        "r_base": r_base,
        "em": em,
        # stamps 11:30..15:00 are rebalance stamps; 15:30 is the flatten
        "n_rebalance_bars": N_BAR_EXIT - 1,
        "delta_end_dense": delta[:, -1],
        "delta_end_sparse": delta[:, -2],
        "spot_flat": spot[:, -1],
        "hedge_ref": g["tape"]["hedge_ref"],
    }


def hold_spec(
    g: dict[str, Any], delta: np.ndarray, gamma: np.ndarray, sums: dict[str, np.ndarray]
) -> dict[str, Any]:
    """The secondary book: the straddle held through the 16:00 cash settlement.

    ``replay_day(exit_clock=None)`` hedges the 15:30-16:00 stretch as well,
    with ``dS = S_close - S_1530``, and flattens the futures leg at the
    settlement.  APPROXIMATION: the minute path's terminal delta at the
    settlement is not observable from the panel's sums, so BOTH cadences are
    charged the same flatten, the tape's own 15:30 delta at ``S_close``; the
    term therefore cancels from the cadence difference.
    """
    spot = g["spot"]
    em = g["em"]
    s_close = g["tape"]["S_close"]
    d_spot = np.concatenate(
        [spot[:, 1:] - spot[:, :-1], (s_close - spot[:, -1])[:, None]], axis=1
    )
    r_base = (-(g["tape"]["settle"] - g["eb"]) + g["tape"]["hedge_hold"]) / em
    return {
        "book": "hold_settlement_crossed",
        "spot": spot,
        "gamma": gamma,
        "delta": delta,
        "dS": d_spot,
        "sumret2": sums["sumret2"],
        "sumabsret": sums["sumabsret"],
        "r_base": r_base,
        "em": em,
        # stamps 11:30..15:30 are rebalance stamps; the settlement flattens
        "n_rebalance_bars": N_BAR_HOLD - 1,
        "delta_end_dense": delta[:, -1],
        "delta_end_sparse": delta[:, -1],
        "spot_flat": s_close,
        "hedge_ref": g["tape"]["hedge_hold"],
    }


def cadence_series(dense: np.ndarray, sp: dict[str, Any]) -> dict[str, np.ndarray]:
    """Day return, hedge turnover and traded notional under a per-bar cadence.

    ``dense[i, j]`` True means bar j of day i is hedged EVERY MINUTE (the
    bound) rather than once, at the bar's far end, on the 30-minute tape.

    The return substitutes the second-order gamma bill bar by bar, exactly as
    the module docstring states.  The turnover is the entry hedge, then, for
    each bar, either the minute path's total variation ``G_t S_t sumabsret_t``
    (APPROXIMATION: ``|d delta| = G |dS|`` per minute with the gamma frozen at
    the stamp) or the single 30-minute rebalance ``|delta_{j+1} - delta_j|``
    at the bar's far stamp, and finally the flatten.  Each minute trade is
    charged at the bar's STARTING spot (APPROXIMATION: the within-bar spot
    variation is second order in the cost); each 30-minute rebalance is
    charged at its own stamp's spot, which is proposal 32's own convention.
    """
    gam = sp["gamma"]
    spot = sp["spot"]
    dlt = sp["delta"]
    d_s = sp["dS"]
    nbar = d_s.shape[1]
    extra = 0.5 * gam[:, :nbar] * (d_s**2 - spot[:, :nbar] ** 2 * sp["sumret2"])
    ret = sp["r_base"] + np.where(dense, extra, 0.0).sum(axis=1) / sp["em"]
    turn = np.abs(dlt[:, 0])
    notional = np.abs(dlt[:, 0]) * spot[:, 0]
    zero = np.zeros(dlt.shape[0])
    for j in range(nbar):
        minute = gam[:, j] * spot[:, j] * sp["sumabsret"][:, j]
        if j < sp["n_rebalance_bars"]:
            reb = np.abs(dlt[:, j + 1] - dlt[:, j])
            reb_spot = spot[:, j + 1]
        else:
            reb = zero
            reb_spot = spot[:, j]
        turn = turn + np.where(dense[:, j], minute, reb)
        notional = notional + np.where(dense[:, j], minute * spot[:, j], reb * reb_spot)
    pos = np.where(dense[:, -1], sp["delta_end_dense"], sp["delta_end_sparse"])
    return {
        "ret": ret,
        "turnover": turn + np.abs(pos),
        "notional": notional + np.abs(pos) * sp["spot_flat"],
    }


# -------------------------------------------------------------- the gate ----
def gate_tape(
    g: dict[str, Any], delta: np.ndarray, specs: list[dict[str, Any]]
) -> None:
    """The published book numbers, the hedge legs and the 30-minute turnover."""
    p32.gate(g["tape"], g["mid"], g["ask"], g["dates"])
    for sp in specs:
        nbar = sp["dS"].shape[1]
        hedged = (delta[:, :nbar] * sp["dS"]).sum(axis=1)
        d = float(np.max(np.abs(hedged - sp["hedge_ref"])))
        assert d < EXACT_TOL, (sp["book"], d)
        print(
            f"  {sp['book']:<24s} hedge leg = sum delta_j dS_j over {nbar} bars, "
            f"max |difference| against replay_day {d:.3e}  OK"
        )
    # proposal 32's own 30-minute turnover construction (part_c, band 0)
    sp = specs[0]
    npos = delta[:, :N_BAR_EXIT]
    prev = np.zeros((npos.shape[0], N_BAR_EXIT + 1))
    prev[:, 1:] = npos
    traded = np.abs(np.concatenate([npos, np.zeros((npos.shape[0], 1))], 1) - prev)
    flat = np.zeros(sp["dS"].shape, dtype=bool)
    base = cadence_series(flat, sp)
    ref_n = (traded * g["spot"]).sum(axis=1)
    d_t = float(np.max(np.abs(base["turnover"] - traded.sum(axis=1))))
    # the notional is in index points (order 1e4 a day), so it is compared
    # RELATIVELY: the same float path, summed in a different order
    d_n = float(np.max(np.abs(base["notional"] - ref_n) / np.abs(ref_n)))
    assert d_t < EXACT_TOL and d_n < EXACT_TOL, (d_t, d_n)
    print(
        f"  30-minute turnover reproduces proposal 32's own construction to "
        f"{d_t:.3e} delta units, and the traded notional to {d_n:.3e} relative"
        f"  OK"
    )
    d_r = float(np.max(np.abs(base["ret"] - sp["r_base"])))
    assert d_r < EXACT_TOL, d_r
    print(f"  the all-sparse cadence IS the book of record to {d_r:.3e}  OK")


def gate_forecast(fc: dict[str, pd.DataFrame], y: pd.DataFrame) -> str:
    """Proposal 35's pooled QLIKE of the combined forecaster, reproduced."""
    yv = y.to_numpy(float)
    dates = pd.DatetimeIndex(y.index)
    common = np.isfinite(yv) & (yv > 0.0)
    for name in (*p35.BASELINES, *p35.COMBINATIONS):
        v = fc[name].to_numpy(float)
        common &= np.isfinite(v) & (v > 0.0)
    rank = []
    for cand in p35.COMBINATIONS:
        fv = np.where(common, fc[cand].to_numpy(float), np.nan)
        q, _ = p35.qlike_mean(np.where(common, yv, np.nan), fv)
        rank.append((q, cand))
    best = sorted(rank)[0][1]
    era = np.asarray(dates >= ERA0, dtype=bool)
    fb = fc[best].to_numpy(float)
    q_w, n_w = p35.qlike_mean(yv, fb)
    q_e, n_e = p35.qlike_mean(yv[era], fb[era])
    print("\nGATE - proposal 35's combined next-bar forecaster")
    for label, have, want, n in (
        ("pooled QLIKE whole", q_w, GATE_QLIKE_WHOLE, n_w),
        ("pooled QLIKE daily era", q_e, GATE_QLIKE_ERA, n_e),
    ):
        assert abs(have - want) < GATE_QLIKE_TOL, (label, have, want)
        print(
            f"  {best}  {label:<24s} {have:.9f}  reference {want:.6f}  n_bars {n}  OK"
        )
    print("GATE PASSED")
    return best


# ------------------------------------------------------------- statistics ---
def boot_sharpe_diff(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Circular block bootstrap of Sharpe(a) - Sharpe(b) on paired day series.

    The two series are resampled TOGETHER (the same day indices), so the draw
    keeps each day's pair intact and the statistic is the difference the table
    reports.
    """
    n = int(a.size)
    idx = asl.circular_block_bootstrap_idx(
        np.random.default_rng(BOOT_SEED), n, BOOT_BLOCK, BOOT_B
    )

    def sharpe_rows(v: np.ndarray) -> np.ndarray:
        d = v[idx]
        sd = d.std(axis=1, ddof=1)
        return np.where(sd > 0.0, d.mean(axis=1) / np.where(sd > 0.0, sd, 1.0), np.nan)

    draws = sharpe_rows(a) - sharpe_rows(b)
    draws = draws[np.isfinite(draws)] * ANN
    lo, hi = (float(v) for v in np.percentile(draws, list(CI_PCT)))
    return {
        "n_days": n,
        "Sharpe_dense": p32.sharpe_ann(a),
        "Sharpe_30": p32.sharpe_ann(b),
        "diff": p32.sharpe_ann(a) - p32.sharpe_ann(b),
        "ci_lo": lo,
        "ci_hi": hi,
        "excludes_zero": bool((lo > 0.0) or (hi < 0.0)),
        "B": BOOT_B,
        "block": BOOT_BLOCK,
        "seed": BOOT_SEED,
    }


def cost_columns(
    row: dict[str, Any], ret: np.ndarray, notional: np.ndarray, em: np.ndarray
) -> None:
    """Add the net mean / Sharpe / charged cost at each of the three bp levels."""
    for bp in COST_BP_GRID:
        cost = bp * 1e-4 * notional / em
        tag = f"{bp:g}bp"
        row[f"cost_prem_{tag}"] = float(cost.mean())
        row[f"mean_net_{tag}"] = float((ret - cost).mean())
        row[f"Sharpe_net_{tag}"] = p32.sharpe_ann(ret - cost)


def ladder_row(
    book: str,
    sample: str,
    cadence: int,
    source: str,
    ret: np.ndarray,
    turn: np.ndarray,
    notional: np.ndarray,
    em: np.ndarray,
    dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    """One (book, sample, cadence) line of the density ladder."""
    s = pd.Series(ret, index=dates)
    row: dict[str, Any] = {
        "book": book,
        "sample": sample,
        "cadence_min": cadence,
        "N_sub": 30.0 / cadence,
        "source": source,
        "n": int(s.size),
        "mean_gross": float(s.mean()),
        "sd_gross": float(s.std(ddof=1)),
        "Sharpe_gross": p32.sharpe_ann(s),
        "turnover": float(turn.mean()),
        "notional_pts": float(notional.mean()),
    }
    cost_columns(row, ret, notional, em)
    row["MaxDD_gross"] = p32.maxdd(s)
    row["worst_day_gross"] = float(s.min())
    row["worst_date_gross"] = str(s.idxmin().date())
    net = s - HEADLINE_BP * 1e-4 * notional / em
    row["MaxDD_net_headline"] = p32.maxdd(net)
    row["worst_day_net_headline"] = float(net.min())
    return row


# ------------------------------------------------------ Part A: the ladder --
def part_a(
    specs: list[dict[str, Any]], dates: pd.DatetimeIndex
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The cadence ladder, the expectation check, and the bootstrap verdict."""
    era = np.asarray(dates >= ERA0, dtype=bool)
    samples: tuple[tuple[str, np.ndarray], ...] = (
        ("whole", np.ones(len(dates), dtype=bool)),
        ("daily_era", era),
    )
    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    boots: list[dict[str, Any]] = []
    for sp in specs:
        nbar = sp["dS"].shape[1]
        flat = np.zeros((len(dates), nbar), dtype=bool)
        a30 = cadence_series(flat, sp)
        a01 = cadence_series(~flat, sp)
        em = sp["em"]

        # ---- the expectation check: 0.5 G dS^2 against 0.5 G S^2 sumret2 ----
        endpoint = 0.5 * sp["gamma"][:, :nbar] * sp["dS"] ** 2
        path = 0.5 * sp["gamma"][:, :nbar] * sp["spot"][:, :nbar] ** 2 * sp["sumret2"]
        for smp, m in samples:
            e = endpoint[m]
            p = path[m]
            checks.append(
                {
                    "book": sp["book"],
                    "sample": smp,
                    "scope": "pooled over bars",
                    "n_bars": int(e.size),
                    "mean_endpoint_pts": float(e.mean()),
                    "mean_path_pts": float(p.mean()),
                    "ratio_path_over_endpoint": float(p.mean() / e.mean()),
                    "mean_removed_pts": float((e - p).mean()),
                    "mean_removed_prem": float(((e - p).sum(axis=1) / em[m]).mean()),
                }
            )
            if sp is specs[0]:
                for j in range(nbar):
                    checks.append(
                        {
                            "book": sp["book"],
                            "sample": smp,
                            "scope": f"{BAR_START[j]}-{BAR_END[j]}",
                            "n_bars": int(e[:, j].size),
                            "mean_endpoint_pts": float(e[:, j].mean()),
                            "mean_path_pts": float(p[:, j].mean()),
                            "ratio_path_over_endpoint": float(
                                p[:, j].mean() / e[:, j].mean()
                            ),
                            "mean_removed_pts": float((e[:, j] - p[:, j]).mean()),
                            "mean_removed_prem": float(
                                ((e[:, j] - p[:, j]) / em[m]).mean()
                            ),
                        }
                    )

        # ---- the ladder ------------------------------------------------------
        for smp, m in samples:
            r30, r01 = a30["ret"][m], a01["ret"][m]
            t30, t01 = a30["turnover"][m], a01["turnover"][m]
            n30, n01 = a30["notional"][m], a01["notional"][m]
            assert (t30 > 0.0).all() and (t01 > 0.0).all(), "a day trades nothing"
            for k in CADENCE_MINUTES:
                nn = 30.0 / k
                if k == 30:
                    ret, turn, notional, src = r30, t30, n30, "built (the tape)"
                elif k == 1:
                    ret, turn, notional, src = r01, t01, n01, "built (the bound)"
                else:
                    ret = r30 + (r01 - r30) * (1.0 - 1.0 / nn)
                    pw = log(nn) / log(30.0)
                    turn = t30 * (t01 / t30) ** pw
                    notional = n30 * (n01 / n30) ** pw
                    src = "INTERPOLATION (1/N variance, log-linear turnover)"
                rows.append(
                    ladder_row(
                        sp["book"], smp, k, src, ret, turn, notional, em[m], dates[m]
                    )
                )
            # the interpolation formula evaluated at its own far anchor
            ret_f = r30 + (r01 - r30) * (1.0 - 1.0 / 30.0)
            rows.append(
                ladder_row(
                    sp["book"],
                    smp,
                    1,
                    "INTERPOLATION formula at N = 30 (diagnostic)",
                    ret_f,
                    t01,
                    n01,
                    em[m],
                    dates[m],
                )
            )

            # ---- the bootstrap ----------------------------------------------
            cost30 = HEADLINE_BP * 1e-4 * n30 / em[m]
            cost01 = HEADLINE_BP * 1e-4 * n01 / em[m]
            for leg, a, b in (
                ("gross", r01, r30),
                (f"net {HEADLINE_BP:g} bp", r01 - cost01, r30 - cost30),
            ):
                boots.append(
                    {
                        "book": sp["book"],
                        "sample": smp,
                        "leg": leg,
                        "comparison": "1 minute - 30 minute",
                        **boot_sharpe_diff(a, b),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(checks), pd.DataFrame(boots)


# -------------------------------------------------- Part B: the conditional --
def expanding_tercile(
    vals: np.ndarray, dates: pd.DatetimeIndex, p: float
) -> np.ndarray:
    """Per-clock expanding, lagged quantile of a (session, clock) grid.

    ``30_skip_day_rule.expanding_lagged_quantile``, column by column: the cut
    used on day d at clock c is a function of days before d at that clock
    only, after the repo's 63-session warm-up.
    """
    out = np.full(vals.shape, np.nan)
    for j in range(vals.shape[1]):
        s = pd.Series(vals[:, j], index=dates)
        out[:, j] = p30.expanding_lagged_quantile(s, p, WARMUP).to_numpy(float)
    return out


def build_dense_rules(
    r_comb: np.ndarray, r_ridge: np.ndarray, slc: np.ndarray, dates: pd.DatetimeIndex
) -> list[dict[str, Any]]:
    """The per-bar density rules and their controls.

    Every rule is a mask over the nine bars of the primary book.  A bar whose
    signal (or whose expanding cut) is not available -- the 63-session warm-up
    and any cell the forecast panel does not carry -- FALLS BACK to the
    unconditional rule, the 30-minute cadence, and is counted as such.
    """
    rules: list[dict[str, Any]] = []
    for tag, r in (("comb", r_comb), ("ridge", r_ridge)):
        fin = np.isfinite(r)
        for q in Q_GRID:
            rules.append(
                {
                    "family": f"{tag}_ratio_hot",
                    "param": f"q={q:g}",
                    "name": f"dense when {tag} f_t / slice_t > {q:g}",
                    "dense": fin & (r > q),
                    "n_no_signal": int((~fin).sum()),
                }
            )
            rules.append(
                {
                    "family": f"{tag}_ratio_calm_control",
                    "param": f"q={q:g}",
                    "name": f"CONTROL dense when {tag} f_t / slice_t < {q:g}",
                    "dense": fin & (r < q),
                    "n_no_signal": int((~fin).sum()),
                }
            )
        hi = expanding_tercile(r, dates, TERCILE_HI)
        lo = expanding_tercile(r, dates, TERCILE_LO)
        rules.append(
            {
                "family": f"{tag}_tercile_hot",
                "param": "top tercile by clock",
                "name": (
                    f"dense when {tag} f_t / slice_t is above its expanding, "
                    f"lagged top-tercile cut at that clock"
                ),
                "dense": fin & np.isfinite(hi) & (r > hi),
                "n_no_signal": int((~(fin & np.isfinite(hi))).sum()),
            }
        )
        rules.append(
            {
                "family": f"{tag}_tercile_calm_control",
                "param": "bottom tercile by clock",
                "name": (
                    f"CONTROL dense when {tag} f_t / slice_t is below its "
                    f"expanding, lagged bottom-tercile cut at that clock"
                ),
                "dense": fin & np.isfinite(lo) & (r < lo),
                "n_no_signal": int((~(fin & np.isfinite(lo))).sum()),
            }
        )
    fin_s = np.isfinite(slc)
    hi_s = expanding_tercile(slc, dates, TERCILE_HI)
    lo_s = expanding_tercile(slc, dates, TERCILE_LO)
    rules.append(
        {
            "family": "slice_level_hot",
            "param": "top tercile by clock",
            "name": (
                "dense when the implied slice's LEVEL is above its expanding, "
                "lagged top-tercile cut at that clock (no forecast at all)"
            ),
            "dense": fin_s & np.isfinite(hi_s) & (slc > hi_s),
            "n_no_signal": int((~(fin_s & np.isfinite(hi_s))).sum()),
        }
    )
    rules.append(
        {
            "family": "slice_level_calm_control",
            "param": "bottom tercile by clock",
            "name": (
                "CONTROL dense when the implied slice's LEVEL is below its "
                "expanding, lagged bottom-tercile cut at that clock"
            ),
            "dense": fin_s & np.isfinite(lo_s) & (slc < lo_s),
            "n_no_signal": int((~(fin_s & np.isfinite(lo_s))).sum()),
        }
    )
    return rules


def part_b(
    sp: dict[str, Any], rules: list[dict[str, Any]], dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Score every density rule against the two anchors, on both samples."""
    era = np.asarray(dates >= ERA0, dtype=bool)
    samples: tuple[tuple[str, np.ndarray], ...] = (
        ("whole", np.ones(len(dates), dtype=bool)),
        ("daily_era", era),
    )
    nbar = sp["dS"].shape[1]
    flat = np.zeros((len(dates), nbar), dtype=bool)
    anchors = {"30": cadence_series(flat, sp), "1": cadence_series(~flat, sp)}
    every: list[dict[str, Any]] = [
        {
            "family": "reference",
            "param": "every bar at 30 minutes",
            "name": "the book of record",
            "dense": flat,
            "n_no_signal": 0,
        },
        {
            "family": "reference",
            "param": "every bar at 1 minute",
            "name": "the density bound",
            "dense": ~flat,
            "n_no_signal": 0,
        },
        *rules,
    ]
    rows: list[dict[str, Any]] = []
    for rule in every:
        out = cadence_series(rule["dense"], sp)
        for smp, m in samples:
            em = sp["em"][m]
            ret = out["ret"][m]
            turn = out["turnover"][m]
            s30 = p32.sharpe_ann(anchors["30"]["ret"][m])
            s01 = p32.sharpe_ann(anchors["1"]["ret"][m])
            t30 = float(anchors["30"]["turnover"][m].mean())
            t01 = float(anchors["1"]["turnover"][m].mean())
            row: dict[str, Any] = {
                "sample": smp,
                "family": rule["family"],
                "param": rule["param"],
                "rule": rule["name"],
                "n_days": int(m.sum()),
                "n_bars": int(rule["dense"][m].size),
                "frac_dense": float(rule["dense"][m].mean()),
                "n_no_signal": int(rule["n_no_signal"]),
                "mean_gross": float(ret.mean()),
                "Sharpe_gross": p32.sharpe_ann(ret),
                "turnover": float(turn.mean()),
                "notional_pts": float(out["notional"][m].mean()),
            }
            cost_columns(row, ret, out["notional"][m], em)
            # the raw differences the two shares below divide: when the bound
            # itself does not gain, the share's denominator is near zero and
            # the share is not readable on its own
            row["Sharpe_gain_vs_book"] = row["Sharpe_gross"] - s30
            row["bound_Sharpe_gain"] = s01 - s30
            row["turnover_vs_book"] = row["turnover"] - t30
            row["bound_turnover"] = t01 - t30
            row["share_of_bound_Sharpe_gain"] = (
                (row["Sharpe_gross"] - s30) / (s01 - s30)
                if abs(s01 - s30) > 0.0
                else float("nan")
            )
            row["share_of_bound_turnover"] = (
                (row["turnover"] - t30) / (t01 - t30)
                if abs(t01 - t30) > 0.0
                else float("nan")
            )
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------- Part C: what minutes would add --
def part_c(
    sp: dict[str, Any], delta: np.ndarray, gamma: np.ndarray, dates: pd.DatetimeIndex
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The within-bar gamma and delta drift, and the dataset that settles it."""
    era = np.asarray(dates >= ERA0, dtype=bool)
    rows: list[dict[str, Any]] = []
    for smp, m in (("whole", np.ones(len(dates), dtype=bool)), ("daily_era", era)):
        for j in range(N_BAR_EXIT):
            g0 = gamma[m, j]
            g1 = gamma[m, j + 1]
            ok = g0 > 0.0
            ratio = g1[ok] / g0[ok]
            rows.append(
                {
                    "sample": smp,
                    "bar": f"{BAR_START[j]}-{BAR_END[j]}",
                    "n": int(ok.sum()),
                    "gamma_start_mean": float(g0[ok].mean()),
                    "gamma_end_mean": float(g1[ok].mean()),
                    "gamma_ratio_mean": float(ratio.mean()),
                    "gamma_ratio_median": float(np.median(ratio)),
                    "gamma_ratio_p90": float(np.percentile(ratio, 90.0)),
                    "delta_drift_mean_abs": float(
                        np.abs(delta[m, j + 1] - delta[m, j]).mean()
                    ),
                    "delta_start_mean_abs": float(np.abs(delta[m, j]).mean()),
                }
            )
    drift = pd.DataFrame(rows)

    raw = pd.read_parquet(PANEL, columns=["endbartime", "numobs"])
    stamp = pd.to_datetime(raw["endbartime"])
    keep = (stamp.dt.normalize().between(ES_FROM, ES_TO)) & stamp.dt.strftime(
        "%H:%M"
    ).isin(list(BAR_END))
    sub = raw.loc[keep].assign(pdate=stamp.loc[keep].dt.normalize())
    per_day = sub.groupby("pdate")["numobs"]
    full = per_day.size() == len(BAR_END)
    sessions = int(full.sum())
    info = {
        "vendor": ES_VENDOR,
        "from": str(ES_FROM.date()),
        "to": str(ES_TO.date()),
        "sessions": sessions,
        "minutes_per_session": ES_MINUTES_PER_SESSION,
        "bars_nominal": sessions * ES_MINUTES_PER_SESSION,
        # the index panel's own minute count on the SAME sessions and stretch:
        # below the nominal count because a half session's bars are short
        "panel_minutes_1100_1600": int(per_day.sum()[full].sum()),
    }
    return drift, info


# ------------------------------------------------------------------ main ----
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------ gates first ----
    g = p35.book_grids()
    dates = g["dates"]
    delta, gamma = stamp_greeks(g["tape"])
    d_gam = float(np.max(np.abs(gamma - g["gamma"])))
    assert d_gam < 1e-15, d_gam
    sums = minute_sums(dates)
    sp_exit = exit_spec(g, delta, gamma, sums)
    sp_hold = hold_spec(g, delta, gamma, sums)
    gate_tape(g, delta, [sp_exit, sp_hold])
    print(
        f"  the package gamma written out in this script matches proposal 35's "
        f"to {d_gam:.3e} on all {gamma.size} stamp-cells  OK"
    )

    work, clocks, _prof = p35.forecast_frame()
    p35.gate_qlike(work)
    fc, _weights, y = p35.combination_grids(work, clocks)
    best = gate_forecast(fc, y)

    # ------------------------------------------------------------ Part A ----
    ladder, checks, boots = part_a([sp_exit, sp_hold], dates)
    ladder_cols = [
        "book",
        "sample",
        "cadence_min",
        "N_sub",
        "source",
        "n",
        "mean_gross",
        "sd_gross",
        "Sharpe_gross",
        "turnover",
        "notional_pts",
        *[f"Sharpe_net_{bp:g}bp" for bp in COST_BP_GRID],
        *[f"cost_prem_{bp:g}bp" for bp in COST_BP_GRID],
        "MaxDD_gross",
        "worst_day_gross",
        "worst_date_gross",
    ]
    write(
        checks,
        "a_expectation_check.csv",
        "Part A  the substitution's expectation check: 0.5 G (dS)^2 against "
        "0.5 G S^2 sumret2, index points a bar",
    )
    ladder.to_csv(OUT / "a_cadence_ladder.csv", index=False)
    for book in (sp_exit["book"], sp_hold["book"]):
        for smp in ("whole", "daily_era"):
            sub = ladder[(ladder["book"] == book) & (ladder["sample"] == smp)]
            show(
                sub,
                f"Part A  the cadence ladder  |  book {book}  |  sample {smp}"
                f"   [a_cadence_ladder.csv]",
                ladder_cols,
            )
    show(
        ladder[ladder["source"].str.startswith("built")],
        "Part A  the two BUILT anchors: drawdown and worst day",
        [
            "book",
            "sample",
            "cadence_min",
            "MaxDD_gross",
            "worst_day_gross",
            "worst_date_gross",
            "MaxDD_net_headline",
            "worst_day_net_headline",
        ],
    )
    write(
        boots,
        "a_bootstrap.csv",
        f"Part A  Sharpe(1 minute) - Sharpe(30 minute), {BOOT_B} circular block "
        f"draws, block {BOOT_BLOCK}, seed {BOOT_SEED}, "
        f"{CI_PCT[0]}-{CI_PCT[1]} percentile interval",
    )
    hit = boots[(boots["book"] == sp_exit["book"]) & (boots["sample"] == "daily_era")]
    for _, r in hit.iterrows():
        verdict = "EXCLUDES zero" if r["excludes_zero"] else "DOES NOT EXCLUDE zero"
        print(
            f"\nPart A verdict ({sp_exit['book']}, daily era, {r['leg']}): the "
            f"1-minute bound's Sharpe is {r['Sharpe_dense']:.3f} against the "
            f"30-minute book's {r['Sharpe_30']:.3f}, a difference of "
            f"{r['diff']:+.3f} whose interval [{r['ci_lo']:+.3f}, "
            f"{r['ci_hi']:+.3f}] {verdict}."
        )

    # ------------------------------------------------------------ Part B ----
    cols = list(SESSION)
    comb = fc[best].reindex(index=dates, columns=cols).to_numpy(float)
    bars = slice(0, N_BAR_EXIT)
    slc = g["slice"][:, bars]
    with np.errstate(invalid="ignore", divide="ignore"):
        r_comb = np.where(
            slc > 0.0, comb[:, bars] / np.where(slc > 0.0, slc, 1.0), np.nan
        )
        r_ridge = np.where(
            slc > 0.0, g["rv_hat"][:, bars] / np.where(slc > 0.0, slc, 1.0), np.nan
        )
    print(
        f"\nPart B signals on the {N_BAR_EXIT} bars of the primary book: the "
        f"combination {best} carries {int(np.isfinite(r_comb).sum())} of "
        f"{r_comb.size} ratio cells, the ridge {int(np.isfinite(r_ridge).sum())}, "
        f"the implied slice {int(np.isfinite(slc).sum())}; every missing cell "
        f"(the {WARMUP}-session warm-up of the expanding weight, and any bar the "
        f"forecast panel does not carry) falls back to the 30-minute cadence and "
        f"is counted in n_no_signal"
    )
    rules = build_dense_rules(r_comb, r_ridge, slc, dates)
    b_rules = part_b(sp_exit, rules, dates)
    b_cols = [
        "sample",
        "family",
        "param",
        "n_days",
        "frac_dense",
        "n_no_signal",
        "mean_gross",
        "Sharpe_gross",
        "turnover",
        *[f"Sharpe_net_{bp:g}bp" for bp in COST_BP_GRID],
        "Sharpe_gain_vs_book",
        "bound_Sharpe_gain",
        "share_of_bound_Sharpe_gain",
        "turnover_vs_book",
        "bound_turnover",
        "share_of_bound_turnover",
    ]
    b_rules.to_csv(OUT / "b_density_rules.csv", index=False)
    for smp in ("whole", "daily_era"):
        show(
            b_rules[b_rules["sample"] == smp],
            f"Part B  forecast-conditional hedge density  |  book "
            f"{sp_exit['book']}  |  sample {smp}   [b_density_rules.csv]",
            b_cols,
        )

    # ------------------------------------------------------------ Part C ----
    drift, info = part_c(sp_exit, delta, gamma, dates)
    write(
        drift,
        "c_within_bar_drift.csv",
        "Part C  the within-bar gamma and delta drift the frozen-gamma "
        "approximation ignores",
    )
    print(
        "\nPart C  what the approximation cannot see, in numbers.  The gamma is "
        f"frozen at the stamp: the mean ratio of the package gamma at the bar's "
        f"end to its start is printed above bar by bar, and it is the last "
        f"bars that move (see the 14:30-15:00 and 15:00-15:30 rows).  The delta "
        f"is re-set at minute closes at no slippage, and the mean absolute "
        f"30-minute delta drift per bar is the delta_drift_mean_abs column.  "
        f"Three things have no number here at all: the ES basis (the hedge is "
        f"priced on the vendor INDEX, not on the futures the book would trade), "
        f"the minute fill prices (every within-bar trade is charged at the bar's "
        f"starting spot), and the within-bar path itself (the panel carries only "
        f"its sums).  The dataset that would settle all four: {info['vendor']}, "
        f"{info['from']} to {info['to']}, RTH 11:00-16:00 ET -- "
        f"{info['sessions']} sessions x {info['minutes_per_session']} minutes = "
        f"{info['bars_nominal']} one-minute bars (the index panel's own minute "
        f"count over the same window and stretch is "
        f"{info['panel_minutes_1100_1600']})."
    )
    pd.DataFrame([info]).to_csv(OUT / "c_dataset.csv", index=False)
    print(f"\nwrote c_dataset.csv: {info}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
