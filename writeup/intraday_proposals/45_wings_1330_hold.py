"""45 - the 13:30 iron butterfly, held to cash settlement.

Forecast-free, and one instrument only.  Proposal 43 built the always-short
delta-hedged tape for every 30-minute entry clock on all 1279 scored sessions
2020-01-03..2025-12-31; its 13:30 HOLD book is the best per-contract clock it
found (unseen 413 sessions +1.173 index points a day, HAC t 3.210).  That book
is naked: the two short legs carry an unbounded loss.  This script buys the
wings.

The structure, per session and per width w in {0.5, 1.0, 1.5, 2.0, 3.0}% of the
13:30 spot:

  short   the 13:30 nearest-OTM straddle, SOLD AT THE QUOTED BID (proposal 43's
          own entry fill), delta-hedged on the index every 30 minutes from 13:30
          through the 16:00 cash settlement, exactly as the naked book;
  long    a call at K_c + w and a put at K_p - w, each rounded OUTWARD to the
          nearest listed strike at the 13:30 stamp (proposal 29's Q1
          construction), bought at the quoted ASK (basis "crossed") or at the
          midpoint (basis "mid").  The wings are NOT hedged: the delta of the
          book is the straddle's alone, which is what makes the winged book's
          hedge P&L and hedge cost identical to the naked book's.

Everything settles into the 16:00 cash print: the short straddle pays its
intrinsic, the long wings collect theirs, and the settlement itself costs
nothing to receive.  A day whose wing strike is unlisted, or whose wing quote is
the vendor's bid == ask == 0 no-quote sentinel, is UNTRADEABLE at that width and
is dropped from that width's book (counted, never silently filled).

The four OPTION legs are capped: the worst the butterfly can do is the width
minus the net credit.  The FUTURES hedge is not capped -- it is a linear
position in the index and its loss is unbounded -- so every table separates the
hedge contribution from the four-leg contribution.

Gates, in order, all asserted before a single new number is computed:
  GATE 0  proposal 43's 11:00 clock reproduces proposal 41's chain build day by
          day, and its deck in-sample mid-inverted Sharpes 2.2530773 (whole) /
          2.5133605 (era from 2022-05-16)   [p43.gate_zero]
  GATE 1  that 11:00 book on the unseen 413 sessions: mean +0.025014 premium
          units, HAC t 1.243, per-contract mean -0.0205 index points
          [p43.gate_one]
  GATE 2  the 13:30 HOLD book on the unseen 413 sessions: +1.173 index points a
          day per contract, HAC t 3.210; the 13:30 FLATTEN book +0.641, t 2.168
  GATE 3  the naked per-contract tape rebuilt from its components equals
          proposal 43's own hold series times the entry premium, to 1e-12
  GATE 4  the realised four-leg loss never exceeds the stated cap, on every
          tradeable day of every width and both price bases

Verdict gate (a COST BAR, not a significance test), on the unseen sample, per
contract: a width is "worth it" only if BOTH
  (a) the hard loss cap on the four option legs is below 5 units of straddle
      premium (median over that width's tradeable days), AND
  (b) the circular-block bootstrap CI of the PAIRED Sharpe difference against
      the naked book on the same days has a lower bound above -0.5, i.e. the
      insurance costs at most half a Sharpe with 95% confidence.
5 widths x 2 price bases = 10 cells.  Both legs are reported for every cell.

Chain reads: proposal 43's builder makes one pyarrow-filtered read of
data/spxw_chain.parquet covering all 12 stamps of all 1279 sessions (it is
imported, not edited, so that read stays where it is); this script adds one
further pyarrow-filtered read for the 13:30 strike LADDER, which the builder
does not return.  Two reads, both single-shot and stamp-filtered.

Run:  python writeup/intraday_proposals/45_wings_1330_hold.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT))
import atm_straddle_lib as asl  # noqa: E402

HOLD = ROOT / "results" / "atm_straddle_intraday_holdclose"
OUT = HOLD / "proposals" / "45"
CHAIN = ROOT / "data" / "spxw_chain.parquet"
P43_PATH = ROOT / "writeup" / "intraday_proposals" / "43_causal_entry_over_time.py"
#: proposal 29 Q1, the 11:00 wing reference this study is compared with.
REF29 = HOLD / "proposals" / "29" / "q1_wings.csv"

ENTRY = "13:30"
REF_CLOCK = "11:00"
CLOSE = "15:30"

#: wing distance, as a fraction of the 13:30 spot; the only width grid tried.
WIDTH_FRACS: tuple[float, ...] = (0.005, 0.010, 0.015, 0.020, 0.030)
#: the printed label of each width, and the map back to its fraction.
LABEL: dict[float, str] = {wf: f"{wf * 100:.1f}% of spot" for wf in WIDTH_FRACS}
FRAC: dict[str, float] = {v: k for k, v in LABEL.items()}
NAKED = "naked straddle (reference)"
#: the two price bases.  The straddle is sold at the quoted bid in BOTH; the
#: basis names the WING fill, so the difference between the two columns is the
#: wing's half-spread and nothing else, and the naked reference is the same
#: series in both columns.
BASES: tuple[str, ...] = ("crossed", "mid")
SAMPLES: tuple[str, ...] = ("deck in-sample", "unseen", "all")

DECK_START = "2020-01-03"
DECK_END = "2024-04-30"
OOS_START = "2024-05-01"
OOS_END = "2025-12-31"

SPX_MULT = 100.0  # dollars per index point, one SPX option contract
HEDGE_COST_BP = 0.5e-4  # 0.5 bp of S on every index unit traded
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))

BOOT_B = 2000
BOOT_BLOCK = 21
BOOT_SEED = 0
CI_LO_PCT = 2.5
CI_HI_PCT = 97.5

# The verdict gate.
CAP_BAR_UNITS = 5.0
SHARPE_CI_FLOOR = -0.5

# Stress grid (proposal 28's construction, moved to the 13:30 entry).
JUMPS: tuple[float, ...] = (0.01, -0.01, 0.02, -0.02, 0.03, -0.03, 0.05, -0.05)
#: the mid-day jump is placed inside the 14:00 -> 14:30 bar, so every stamp
#: from 14:30 on and the settlement print carry it.
MIDDAY_BAR_END = "14:30"
PLACEMENTS: tuple[str, ...] = ("last bar", "mid-day 14:00->14:30")

# GATE targets (proposal 43's own prints).
GATE_OOS_N = 413
GATE_1330_HOLD_PTS = 1.173
GATE_1330_HOLD_T = 3.210
GATE_1330_FLAT_PTS = 0.641
GATE_1330_FLAT_T = 2.168
GATE_PTS_TOL = 5e-4
GATE_T_TOL = 5e-4
GATE_REBUILD_TOL = 1e-12
GATE_CAP_TOL = 1e-9


# --------------------------------------------------------------- bootstrap --
def _boot_cell(job: tuple[str, np.ndarray, np.ndarray]) -> dict[str, Any]:
    """One paired bootstrap cell: CI of the mean difference and of d Sharpe.

    One circular-block index array (B draws, block length 21, seed 0) is drawn
    once and applied to BOTH legs, so every draw is a paired resample of the
    same days.  Run in a worker process: the 30 cells are independent.
    """
    key, w, nk = job
    n = int(w.size)
    out: dict[str, Any] = {"key": key, "n": n}
    if n < 2:
        for f in (
            "ci_lo_mean",
            "ci_hi_mean",
            "ci_lo_dsharpe",
            "ci_hi_dsharpe",
            "d_sharpe",
        ):
            out[f] = float("nan")
        return out
    rng = np.random.default_rng(BOOT_SEED)
    idx = asl.circular_block_bootstrap_idx(rng, n, BOOT_BLOCK, BOOT_B)
    big_w = w[idx]
    big_n = nk[idx]
    dm = (big_w - big_n).mean(axis=1)
    sd_w = big_w.std(axis=1, ddof=1)
    sd_n = big_n.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sh_w = np.where(sd_w > 0, big_w.mean(axis=1) / sd_w * ANN, np.nan)
        sh_n = np.where(sd_n > 0, big_n.mean(axis=1) / sd_n * ANN, np.nan)
    ds = sh_w - sh_n
    ds = ds[np.isfinite(ds)]
    out["ci_lo_mean"] = float(np.percentile(dm, CI_LO_PCT))
    out["ci_hi_mean"] = float(np.percentile(dm, CI_HI_PCT))
    out["ci_lo_dsharpe"] = float(np.percentile(ds, CI_LO_PCT)) if ds.size else np.nan
    out["ci_hi_dsharpe"] = float(np.percentile(ds, CI_HI_PCT)) if ds.size else np.nan
    out["n_boot_finite_dsharpe"] = int(ds.size)
    return out


# ------------------------------------------------------------------ module --
def load_p43() -> ModuleType:
    """Proposal 43, imported by path (never edited, never re-implemented)."""
    spec = importlib.util.spec_from_file_location("p43_for_45", P43_PATH)
    assert spec is not None and spec.loader is not None, P43_PATH
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------- the ladders --
def entry_chain(stamps: list[pd.Timestamp], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Every 0DTE quote at the 13:30 stamp, keyed (date, cp, strike).

    One pyarrow-filtered read.  bid == ask == 0 is the vendor's no-quote
    sentinel and is FLAGGED (``live``), never dropped, so the untradeable days
    can be counted.  The 16:00 row is never touched: 13:30 is a live stamp and
    the settlement print enters only through the GSPC close.
    """
    t0 = time.time()
    tbl = pq.read_table(
        CHAIN,
        columns=[
            "expiration",
            "timestamp",
            "strike",
            "cp",
            "bid",
            "ask",
            "hours_to_expiration",
        ],
        filters=[("timestamp", "in", stamps)],
    )
    ch = tbl.to_pandas()
    key = pd.DataFrame(
        {
            "timestamp": pd.Series(stamps).astype(ch["timestamp"].dtype),
            "date": pd.DatetimeIndex(dates),
        }
    )
    ch = ch.merge(key, on="timestamp", how="inner")
    ch = ch[ch["expiration"].astype("datetime64[ns]") == ch["date"]]  # 0DTE only
    n_frozen = int((ch["hours_to_expiration"].astype(float) <= 0.0).sum())
    ch["strike"] = ch["strike"].astype(float)
    ch["cp"] = ch["cp"].astype(str)
    ch["bid"] = ch["bid"].astype(float)
    ch["ask"] = ch["ask"].astype(float)
    ch["mid"] = asl.quote_mid(ch["bid"], ch["ask"]).to_numpy()
    ch["live"] = ~((ch["bid"] == 0.0) & (ch["ask"] == 0.0))
    ch = ch.drop_duplicates(["date", "cp", "strike"])
    out = ch.set_index(["date", "cp", "strike"])[
        ["bid", "ask", "mid", "live"]
    ].sort_index()
    print(
        f"chain read once for the {ENTRY} ladder: {len(stamps)} stamps, "
        f"{len(out):,} 0DTE quotes over "
        f"{out.index.get_level_values(0).nunique()} sessions, rows with "
        f"hours_to_expiration <= 0 at {ENTRY}: {n_frozen} "
        f"({time.time() - t0:.1f} s)"
    )
    return out


def ladder_matrix(q: pd.DataFrame, cp: str, dates: pd.DatetimeIndex) -> np.ndarray:
    """(sessions x max strikes) ascending strike ladder, padded with +inf.

    Built with one scatter, no python loop over sessions.  The padding is +inf
    so it is counted by neither ``< target`` nor ``<= target``.
    """
    sub = q.xs(cp, level="cp").reset_index()[["date", "strike"]]
    sub = sub.sort_values(["date", "strike"], kind="stable")
    col = sub.groupby("date").cumcount().to_numpy()
    row = dates.get_indexer(pd.DatetimeIndex(sub["date"]))
    keep = row >= 0
    mat = np.full((len(dates), int(col.max()) + 1), np.inf)
    mat[row[keep], col[keep]] = sub["strike"].to_numpy(float)[keep]
    return mat


def outward_up(mat: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Nearest listed strike at or ABOVE target, per session; NaN if none."""
    k = (mat < target[:, None]).sum(axis=1)
    k = np.minimum(k, mat.shape[1] - 1)
    out = mat[np.arange(mat.shape[0]), k]
    out = np.where(np.isfinite(target), out, np.inf)
    return np.where(np.isfinite(out), out, np.nan)


def outward_dn(mat: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Nearest listed strike at or BELOW target, per session; NaN if none."""
    k = (mat <= target[:, None]).sum(axis=1) - 1
    ok = k >= 0
    out = mat[np.arange(mat.shape[0]), np.maximum(k, 0)]
    out = np.where(ok & np.isfinite(target), out, np.inf)
    return np.where(np.isfinite(out), out, np.nan)


def leg_field(
    q: pd.DataFrame, dates: pd.DatetimeIndex, cp: str, strikes: np.ndarray, field: str
) -> np.ndarray:
    """One quote field at (date, cp, strike), aligned to the session index."""
    key = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(dates), np.full(len(dates), cp), np.asarray(strikes, float)]
    )
    return q[field].reindex(key).to_numpy(float)


def wing_legs(
    q: pd.DataFrame,
    dates: pd.DatetimeIndex,
    mat_c: np.ndarray,
    mat_p: np.ndarray,
    k_c: np.ndarray,
    k_p: np.ndarray,
    s_entry: np.ndarray,
    s_close: np.ndarray,
    wf: float,
) -> dict[str, np.ndarray]:
    """The two long wings at wf of the 13:30 spot, and their 13:30 quotes."""
    kw_c = outward_up(mat_c, k_c + wf * s_entry)
    kw_p = outward_dn(mat_p, k_p - wf * s_entry)
    ask = leg_field(q, dates, "C", kw_c, "ask") + leg_field(q, dates, "P", kw_p, "ask")
    mid = leg_field(q, dates, "C", kw_c, "mid") + leg_field(q, dates, "P", kw_p, "mid")
    live = (leg_field(q, dates, "C", kw_c, "live") == 1.0) & (
        leg_field(q, dates, "P", kw_p, "live") == 1.0
    )
    no_listed = ~(np.isfinite(kw_c) & np.isfinite(kw_p))
    w_up = kw_c - k_c
    w_dn = k_p - kw_p
    return {
        "K_wing_c": kw_c,
        "K_wing_p": kw_p,
        "no_listed": no_listed,
        "cost_crossed": ask,
        "cost_mid": mid,
        "live": live & ~no_listed & np.isfinite(ask) & np.isfinite(mid),
        "intrinsic": np.maximum(s_close - kw_c, 0.0) + np.maximum(kw_p - s_close, 0.0),
        "w_up": w_up,
        "w_dn": w_dn,
        "width": np.maximum(w_up, w_dn),
        "capped": (s_close >= kw_c) | (s_close <= kw_p),
    }


# ------------------------------------------------------------- statistics ---
def both_units(
    p43: ModuleType, pts: pd.Series, units: pd.Series, prefix: str = ""
) -> dict[str, Any]:
    a = dict(p43.stats_row(pts.dropna()))
    b = dict(p43.stats_row(units.dropna()))
    row: dict[str, Any] = {f"{prefix}n": a["n"]}
    for k in ("mean", "sd", "Sharpe_ann", "t_hac", "t_lag", "MaxDD", "worst"):
        row[f"{prefix}{k}_pts"] = a[k]
        row[f"{prefix}{k}_units"] = b[k]
    row[f"{prefix}worst_date"] = a["worst_date"]
    return row


# ------------------------------------------------------------------ stress --
def day_hedge_settle(
    p43: ModuleType,
    s_row: np.ndarray,
    tot_row: np.ndarray,
    j: int,
    k_c: float,
    k_p: float,
    s_settle: float,
) -> tuple[float, float, np.ndarray]:
    """Hedge P&L and straddle settlement on one day's (possibly jumped) path.

    Proposal 28's reprice, on proposal 43's mid-inverted tape: the total
    volatility at each stamp is the one bisected out of that stamp's observed
    straddle midpoint and is held FIXED; only the spot path and the settlement
    print move.  The hedge holds +delta index units per short straddle over
    each 30-minute step from the entry clock through the settlement print.
    """
    tot = tot_row.copy()
    tot[:j] = np.nan
    dlt = p43.pkg_delta_vec(tot, s_row, k_c, k_p)
    dlt[:j] = 0.0
    nxt = np.full_like(s_row, np.nan)
    nxt[:-1] = s_row[1:]
    nxt[-1] = s_settle
    d_s = np.where(np.isfinite(s_row) & np.isfinite(nxt), nxt - s_row, 0.0)
    d_s[:j] = 0.0
    hedge = float((dlt * d_s).sum())
    settle = max(s_settle - k_c, 0.0) + max(k_p - s_settle, 0.0)
    return hedge, settle, dlt


def jumped_path(
    clocks: tuple[str, ...],
    s_row: np.ndarray,
    s_settle: float,
    x: float,
    placement: str,
) -> tuple[np.ndarray, float]:
    """The day's spot path and settlement with one jump x placed in one bar."""
    if placement == "last bar":
        # The 15:30 hedge is frozen: only the settlement print moves.
        return s_row.copy(), float(s_row[-1]) * (1.0 + x)
    if placement == "mid-day 14:00->14:30":
        k = clocks.index(MIDDAY_BAR_END)
        f = np.ones_like(s_row)
        f[k:] = 1.0 + x
        return s_row * f, s_settle * (1.0 + x)
    raise ValueError(placement)


# -------------------------------------------------------------------- main --
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 90)
    pd.set_option("display.max_rows", 500)
    t_start = time.time()
    p43 = load_p43()

    # ------------------------------------------------- proposal 43's tape --
    stamp, _px = p43.session_stamps()
    half = p43.half_sessions(stamp)
    sessions = pd.DatetimeIndex(stamp.index).difference(half)
    print(
        f"chain sessions {len(stamp)}; half sessions dropped "
        f"(hours_to_expiration <= 0 at {CLOSE}, {len(half)}): "
        + ", ".join(str(d.date()) for d in half)
        + f"; sessions scored {len(sessions)}"
    )
    ch = p43.build_chain(stamp, sessions)
    idx = pd.DatetimeIndex(ch["dates"])
    clocks: tuple[str, ...] = tuple(p43.CLOCKS)
    j_entry = list(p43.ENTRIES).index(ENTRY)
    arrays = {
        k: ch[k] for k in ("S", "K_c", "K_p", "entry", "bid", "tot", "ask_c", "ask_p")
    }
    arrays["S_close"] = ch["S_close"]
    books = {
        c: {
            k: pd.Series(v, index=idx)
            for k, v in p43._clock_book((list(p43.ENTRIES).index(c), arrays)).items()
        }
        for c in (REF_CLOCK, ENTRY)
    }

    # ------------------------------------------------------------- gates ---
    p43.gate_zero(books, ch)
    p43.gate_one(books)

    b = books[ENTRY]
    entry_mid = b["entry"].to_numpy(float)
    bid = b["bid"].to_numpy(float)
    hedge_hold = b["hedge_hold_pts"].to_numpy(float)
    settle_str = b["settle_pts"].to_numpy(float)
    k_c = b["K_c"].to_numpy(float)
    k_p = b["K_p"].to_numpy(float)
    naked_pts_arr = -(settle_str - bid) + hedge_hold
    naked_pts_arr = np.where(
        np.isfinite(b["hold"].to_numpy(float)), naked_pts_arr, np.nan
    )
    naked_pts = pd.Series(naked_pts_arr, index=idx)
    naked_units = b["hold"]
    oos = np.asarray(idx >= pd.Timestamp(OOS_START))
    for name, series in (("hold", b["hold"]), ("flatten", b["flatten"])):
        v = (series * b["entry"])[oos].dropna()
        t, lag = asl.newey_west_t(v)
        tgt_m = GATE_1330_HOLD_PTS if name == "hold" else GATE_1330_FLAT_PTS
        tgt_t = GATE_1330_HOLD_T if name == "hold" else GATE_1330_FLAT_T
        assert int(v.size) == GATE_OOS_N, int(v.size)
        assert abs(float(v.mean()) - tgt_m) < GATE_PTS_TOL, float(v.mean())
        assert abs(t - tgt_t) < GATE_T_TOL, t
        print(
            f"GATE 2  the {ENTRY} {name} book on the unseen sessions: n "
            f"{int(v.size)} (target {GATE_OOS_N}), per-contract mean "
            f"{float(v.mean()):+.6f} index points (target {tgt_m}), HAC t "
            f"{t:+.4f} (lag {lag}, target {tgt_t})"
        )
    dev = float(
        np.nanmax(np.abs(naked_pts_arr - (naked_units * b["entry"]).to_numpy()))
    )
    assert dev < GATE_REBUILD_TOL, dev
    print(
        f"GATE 3  the naked per-contract tape rebuilt from -(settlement - bid) "
        f"+ hedge equals proposal 43's hold series x the entry premium: max abs "
        f"difference {dev:.1e} (tolerance {GATE_REBUILD_TOL:.0e})"
    )

    # ----------------------------------------------------- the 13:30 ladder --
    stamps = [pd.Timestamp(stamp.loc[d, ENTRY]) for d in idx]
    q = entry_chain(stamps, idx)
    mat_c = ladder_matrix(q, "C", idx)
    mat_p = ladder_matrix(q, "P", idx)
    s_entry = ch["S"][:, j_entry].copy()
    s_close = ch["S_close"].copy()
    print(
        f"strike ladder at {ENTRY}: calls {mat_c.shape[1]} deep at the widest "
        f"session, puts {mat_p.shape[1]}; median listed calls per session "
        f"{float(np.median(np.isfinite(mat_c).sum(axis=1))):.0f}, puts "
        f"{float(np.median(np.isfinite(mat_p).sum(axis=1))):.0f}"
    )

    base_ok = np.isfinite(naked_pts_arr)
    masks = {
        "deck in-sample": np.asarray(idx <= pd.Timestamp(DECK_END)),
        "unseen": np.asarray(idx >= pd.Timestamp(OOS_START)),
        "all": np.ones(len(idx), bool),
    }
    print(
        "sessions: "
        + ", ".join(
            f"{s} {int(m.sum())} (naked book priced on {int((m & base_ok).sum())})"
            for s, m in masks.items()
        )
    )

    # ------------------------------------------------------------- wings ----
    legs: dict[float, dict[str, np.ndarray]] = {}
    trad_rows: list[dict[str, Any]] = []
    for wf in WIDTH_FRACS:
        g = wing_legs(q, idx, mat_c, mat_p, k_c, k_p, s_entry, s_close, wf)
        legs[wf] = g
        ok = base_ok & g["live"]
        trad_rows.append(
            {
                "width": LABEL[wf],
                "w_frac": wf,
                "n_naked_scored": int(base_ok.sum()),
                "n_no_listed_strike": int((base_ok & g["no_listed"]).sum()),
                "n_dead_wing_quote": int(
                    (base_ok & ~g["no_listed"] & ~g["live"]).sum()
                ),
                "n_untradeable": int((base_ok & ~g["live"]).sum()),
                "n_tradeable": int(ok.sum()),
                "med_w_up_pts": float(np.nanmedian(g["w_up"][ok])),
                "med_w_dn_pts": float(np.nanmedian(g["w_dn"][ok])),
                "med_width_pts": float(np.nanmedian(g["width"][ok])),
                "med_width_pct_spot": float(
                    np.nanmedian(100.0 * g["width"][ok] / s_entry[ok])
                ),
                "frac_days_cap_bound": float(np.mean(g["capped"][ok])),
            }
        )
    trad = pd.DataFrame(trad_rows)
    print(
        f"\n--- A. wing tradeability at {ENTRY} ({len(WIDTH_FRACS)} widths; a day "
        "with an unlisted strike or a bid == ask == 0 wing quote is untradeable "
        "at that width)"
    )
    print(trad.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # ---------------------------------------------------------- the books ---
    daily: dict[str, pd.Series] = {
        "naked|pts": naked_pts,
        "naked|units": naked_units,
        "entry_mid_pts": b["entry"],
        "bid_pts": b["bid"],
        "hedge_hold_pts": b["hedge_hold_pts"],
        "cost_hold_pts": b["cost_hold_pts"],
    }
    book_rows: list[dict[str, Any]] = []
    cap_dev = 0.0
    for sname, m in masks.items():
        row = {
            "sample": sname,
            "structure": NAKED,
            "w_frac": np.nan,
            "basis": "straddle at the bid",
            "n_untradeable": 0,
            "wing_cost_pct_prem_med": np.nan,
            "net_credit_units_med": np.nan,
            "cap_units_med": np.inf,
            "cap_dollars_med": np.inf,
            "frac_days_cap_bound": np.nan,
        }
        row.update(both_units(p43, naked_pts[m & base_ok], naked_units[m & base_ok]))
        row.update({"n_naked_matched": int((m & base_ok).sum())})
        for f in ("mean_pts", "Sharpe_ann_pts", "mean_units", "Sharpe_ann_units"):
            row[f"naked_{f}"] = row[f]
        book_rows.append(row)

    for wf in WIDTH_FRACS:
        g = legs[wf]
        label = LABEL[wf]
        ok = base_ok & g["live"]
        four_leg_loss = settle_str - g["intrinsic"]
        cap_dev = max(
            cap_dev,
            float(np.nanmax((four_leg_loss - g["width"])[ok])) if ok.any() else 0.0,
        )
        for basis in BASES:
            cost = g[f"cost_{basis}"]
            net_credit = bid - cost
            cap_pts = g["width"] - net_credit
            pts = np.where(ok, naked_pts_arr + (g["intrinsic"] - cost), np.nan)
            s_pts = pd.Series(pts, index=idx)
            s_units = s_pts / b["entry"]
            daily[f"{label}|{basis}|pts"] = s_pts
            daily[f"{label}|{basis}|units"] = s_units
            for sname, m in masks.items():
                sel = m & ok
                row = {
                    "sample": sname,
                    "structure": label,
                    "w_frac": wf,
                    "basis": basis,
                    "n_untradeable": int((m & base_ok & ~g["live"]).sum()),
                    "wing_cost_pct_prem_med": float(
                        np.nanmedian(100.0 * cost[sel] / entry_mid[sel])
                    )
                    if sel.any()
                    else np.nan,
                    "net_credit_units_med": float(
                        np.nanmedian(net_credit[sel] / entry_mid[sel])
                    )
                    if sel.any()
                    else np.nan,
                    "cap_units_med": float(np.nanmedian(cap_pts[sel] / entry_mid[sel]))
                    if sel.any()
                    else np.nan,
                    "cap_dollars_med": float(np.nanmedian(cap_pts[sel] * SPX_MULT))
                    if sel.any()
                    else np.nan,
                    "frac_days_cap_bound": float(np.mean(g["capped"][sel]))
                    if sel.any()
                    else np.nan,
                }
                row.update(both_units(p43, s_pts[sel], s_units[sel]))
                nk_p = naked_pts[sel]
                nk_u = naked_units[sel]
                row["n_naked_matched"] = int(nk_p.notna().sum())
                row["naked_mean_pts"] = float(nk_p.mean())
                row["naked_Sharpe_ann_pts"] = p43.sharpe(nk_p)
                row["naked_mean_units"] = float(nk_u.mean())
                row["naked_Sharpe_ann_units"] = p43.sharpe(nk_u)
                book_rows.append(row)
    assert cap_dev <= GATE_CAP_TOL, cap_dev
    print(
        f"\nGATE 4  the realised four-leg loss (short straddle intrinsic minus "
        f"wing intrinsic) never exceeds the stated width on any tradeable day of "
        f"any width: worst excess {cap_dev:+.1e} index points (tolerance "
        f"{GATE_CAP_TOL:.0e})"
    )
    book = pd.DataFrame(book_rows)
    show = [
        "sample",
        "structure",
        "basis",
        "n",
        "n_untradeable",
        "wing_cost_pct_prem_med",
        "net_credit_units_med",
        "mean_pts",
        "Sharpe_ann_pts",
        "t_hac_pts",
        "mean_units",
        "Sharpe_ann_units",
        "t_hac_units",
        "MaxDD_units",
        "MaxDD_pts",
        "worst_pts",
        "worst_date",
        "cap_units_med",
        "cap_dollars_med",
        "frac_days_cap_bound",
    ]
    for sname in SAMPLES:
        print(f"\n--- B. the {ENTRY} hold book, sample = {sname}")
        print(
            book.loc[book["sample"] == sname, show].to_string(
                index=False, float_format=lambda v: f"{v:,.4f}"
            )
        )

    # -------------------------------------------------- paired differences --
    jobs: list[tuple[str, np.ndarray, np.ndarray]] = []
    meta: dict[str, dict[str, Any]] = {}
    for wf in WIDTH_FRACS:
        g = legs[wf]
        label = LABEL[wf]
        ok = base_ok & g["live"]
        for basis in BASES:
            w_pts = daily[f"{label}|{basis}|pts"].to_numpy(float)
            for sname, m in masks.items():
                sel = m & ok & np.isfinite(w_pts)
                key = f"{label}|{basis}|{sname}"
                jobs.append((key, w_pts[sel], naked_pts_arr[sel]))
                meta[key] = {
                    "structure": label,
                    "w_frac": wf,
                    "basis": basis,
                    "sample": sname,
                }
    t0 = time.time()
    n_workers = min(len(jobs), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        boots = list(pool.map(_boot_cell, jobs))
    print(
        f"\n{len(jobs)} paired bootstrap cells ({len(WIDTH_FRACS)} widths x "
        f"{len(BASES)} bases x {len(SAMPLES)} samples), B={BOOT_B}, block "
        f"{BOOT_BLOCK}, seed {BOOT_SEED}, on {n_workers} worker processes: "
        f"{time.time() - t0:.1f} s"
    )
    pair_rows: list[dict[str, Any]] = []
    for job, bt in zip(jobs, boots, strict=True):
        key, w_pts, n_pts = job
        d = pd.Series(w_pts - n_pts)
        t, lag = asl.newey_west_t(d)
        row = dict(meta[key])
        row.update(
            {
                "n": int(w_pts.size),
                "mean_diff_pts": float(d.mean()) if d.size else np.nan,
                "t_hac": t,
                "t_lag": lag,
                "ci_lo_mean_pts": bt["ci_lo_mean"],
                "ci_hi_mean_pts": bt["ci_hi_mean"],
                "Sharpe_winged": p43.sharpe(pd.Series(w_pts)),
                "Sharpe_naked": p43.sharpe(pd.Series(n_pts)),
                "d_Sharpe": p43.sharpe(pd.Series(w_pts)) - p43.sharpe(pd.Series(n_pts)),
                "ci_lo_dSharpe": bt["ci_lo_dsharpe"],
                "ci_hi_dSharpe": bt["ci_hi_dsharpe"],
            }
        )
        pair_rows.append(row)
    pair = pd.DataFrame(pair_rows)
    print(
        f"\n--- C. paired daily difference, winged minus naked, per contract "
        f"(index points), on the width's own tradeable days; HAC t and a "
        f"circular-block bootstrap CI (B={BOOT_B}, block {BOOT_BLOCK}, seed "
        f"{BOOT_SEED})"
    )
    for sname in SAMPLES:
        print(f"\n  sample = {sname}")
        print(
            pair.loc[pair["sample"] == sname]
            .drop(columns=["sample"])
            .to_string(index=False, float_format=lambda v: f"{v:,.4f}")
        )

    # ------------------------------------------------------- hedge cost -----
    cost_pts = b["cost_hold_pts"]
    cost_rows: list[dict[str, Any]] = []
    for sname, m in masks.items():
        for structure, sel, s_pts in [(NAKED, base_ok, naked_pts)] + [
            (
                LABEL[wf],
                base_ok & legs[wf]["live"],
                daily[f"{wf * 100:.1f}% of spot|crossed|pts"],
            )
            for wf in WIDTH_FRACS
        ]:
            k = m & sel
            gp = s_pts[k]
            cp_ = cost_pts[k]
            net = gp - cp_
            cost_rows.append(
                {
                    "sample": sname,
                    "structure": structure,
                    "basis": "crossed",
                    "n": int(gp.notna().sum()),
                    "mean_hedge_cost_pts": float(cp_.mean()),
                    "mean_hedge_cost_units": float((cp_ / b["entry"][k]).mean()),
                    "mean_reported_pts": float(gp.mean()),
                    "mean_charged_pts": float(net.mean()),
                    "Sharpe_reported_pts": p43.sharpe(gp),
                    "Sharpe_charged_pts": p43.sharpe(net),
                    "mean_reported_units": float((gp / b["entry"][k]).mean()),
                    "mean_charged_units": float((net / b["entry"][k]).mean()),
                    "Sharpe_reported_units": p43.sharpe(gp / b["entry"][k]),
                    "Sharpe_charged_units": p43.sharpe(net / b["entry"][k]),
                }
            )
    cost_tab = pd.DataFrame(cost_rows)
    print(
        f"\n--- D. hedge cost at {HEDGE_COST_BP * 1e4:.1f} bp of S on every index "
        "unit traded (the entry hedge, every 30-minute rebalance and the unwind "
        "into the settlement print): reported, then charged.  The wings are "
        "unhedged, so this line is the SAME for the naked book and every width; "
        "it differs across rows only through the day sets."
    )
    print(cost_tab.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))

    # ----------------------------------------------------------- stress -----
    ratio = pd.Series(np.where(base_ok, entry_mid / s_entry, np.nan), index=idx)
    srt = ratio.dropna().sort_values()
    med_day = pd.Timestamp(srt.index[len(srt) // 2])
    i_med = int(idx.get_loc(med_day))
    print(
        f"\nrepresentative day: the order statistic at index {len(srt) // 2} of "
        f"the {len(srt)} sorted {ENTRY} premium/spot ratios -- {med_day.date()}, "
        f"premium {entry_mid[i_med]:.4f} index points on spot "
        f"{s_entry[i_med]:.2f} ({100 * srt.iloc[len(srt) // 2]:.4f}% of spot); "
        f"5th percentile {100 * ratio.quantile(0.05):.4f}%, 95th "
        f"{100 * ratio.quantile(0.95):.4f}%"
    )
    s_row0 = ch["S"][i_med].copy()
    tot_row0 = ch["tot"][i_med].copy()
    kc0 = float(k_c[i_med])
    kp0 = float(k_p[i_med])
    bid0 = float(bid[i_med])
    entry0 = float(entry_mid[i_med])
    settle0 = float(s_close[i_med])
    stress_rows: list[dict[str, Any]] = []
    for placement in ("base (no jump)",) + PLACEMENTS:
        grid = (0.0,) if placement == "base (no jump)" else JUMPS
        for x in grid:
            if placement == "base (no jump)":
                s_row, st = s_row0.copy(), settle0
            else:
                s_row, st = jumped_path(clocks, s_row0, settle0, x, placement)
            hedge, settle_s, _dlt = day_hedge_settle(
                p43, s_row, tot_row0, j_entry, kc0, kp0, st
            )
            for structure in (NAKED,) + tuple(LABEL[wf] for wf in WIDTH_FRACS):
                if structure.startswith("naked"):
                    intr = 0.0
                    w_cost = 0.0
                    live = True
                    kwc = kwp = np.nan
                    cap = np.inf
                else:
                    wf = FRAC[structure]
                    g = legs[wf]
                    kwc = float(g["K_wing_c"][i_med])
                    kwp = float(g["K_wing_p"][i_med])
                    live = bool(g["live"][i_med])
                    w_cost = float(g["cost_crossed"][i_med])
                    intr = max(st - kwc, 0.0) + max(kwp - st, 0.0)
                    cap = float(g["width"][i_med] - (bid0 - w_cost))
                four_leg = -(settle_s - bid0) + (intr - w_cost)
                pnl = four_leg + hedge
                stress_rows.append(
                    {
                        "placement": placement,
                        "jump_pct": 100.0 * x,
                        "structure": structure,
                        "basis": "crossed",
                        "tradeable": live,
                        "K_wing_c": kwc,
                        "K_wing_p": kwp,
                        "S_settle_stressed": st,
                        "straddle_settle_pts": settle_s,
                        "wing_intrinsic_pts": intr,
                        "wing_cost_pts": w_cost,
                        "four_leg_pts": four_leg,
                        "hedge_pts": hedge,
                        "pnl_pts": pnl,
                        "pnl_units": pnl / entry0,
                        "pnl_dollars": pnl * SPX_MULT,
                        "four_leg_units": four_leg / entry0,
                        "four_leg_dollars": four_leg * SPX_MULT,
                        "hedge_units": hedge / entry0,
                        "hedge_dollars": hedge * SPX_MULT,
                        "cap_units": cap / entry0,
                        "cap_dollars": cap * SPX_MULT,
                    }
                )
    stress = pd.DataFrame(stress_rows)
    n_cells = int(len(PLACEMENTS) * len(JUMPS) * (1 + len(WIDTH_FRACS)))
    print(
        f"\n--- E. crash stress on {med_day.date()}, hold book, crossed basis: "
        f"{len(PLACEMENTS)} placements x {len(JUMPS)} jumps x "
        f"{1 + len(WIDTH_FRACS)} structures = {n_cells} cells, plus the "
        f"{1 + len(WIDTH_FRACS)} unjumped base rows.  The four option legs are "
        "capped at width minus net credit; the FUTURES HEDGE IS UNBOUNDED and "
        "its contribution is the hedge_* columns."
    )
    print(stress.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    # ------------------------------------------------------------ verdict ---
    gate = pair.loc[pair["sample"] == "unseen"].merge(
        book.loc[book["sample"] == "unseen", ["structure", "basis", "cap_units_med"]],
        on=["structure", "basis"],
        how="left",
    )
    gate = gate.loc[gate["structure"] != NAKED].copy()
    gate["cap_below_5_units"] = gate["cap_units_med"] < CAP_BAR_UNITS
    gate["dSharpe_ci_lo_above_-0.5"] = gate["ci_lo_dSharpe"] > SHARPE_CI_FLOOR
    gate["worth_it"] = gate["cap_below_5_units"] & gate["dSharpe_ci_lo_above_-0.5"]
    print(
        f"\n--- F. verdict gate, unseen sample, per contract: "
        f"{len(WIDTH_FRACS)} widths x {len(BASES)} price bases = "
        f"{len(WIDTH_FRACS) * len(BASES)} cells.  A width is worth it only if "
        f"(a) the median loss cap on the four option legs is below "
        f"{CAP_BAR_UNITS:.0f} units of straddle premium AND (b) the bootstrap CI "
        f"of the paired Sharpe difference against the naked book has a lower "
        f"bound above {SHARPE_CI_FLOOR}.  This is a COST BAR, not a significance "
        "test: leg (b) asks whether the insurance costs at most half a Sharpe "
        "with 95% confidence, and passing it is not evidence of an edge."
    )
    print(
        gate[
            [
                "structure",
                "basis",
                "n",
                "cap_units_med",
                "cap_below_5_units",
                "mean_diff_pts",
                "t_hac",
                "ci_lo_mean_pts",
                "ci_hi_mean_pts",
                "d_Sharpe",
                "ci_lo_dSharpe",
                "ci_hi_dSharpe",
                "dSharpe_ci_lo_above_-0.5",
                "worth_it",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:,.4f}")
    )
    for _, r in gate.iterrows():
        print(
            f"  {r['structure']:16s} {r['basis']:8s} cap "
            f"{r['cap_units_med']:.4f} u "
            + ("< 5 PASS" if r["cap_below_5_units"] else ">= 5 FAIL")
            + f"; d Sharpe {r['d_Sharpe']:+.4f} CI ["
            f"{r['ci_lo_dSharpe']:+.4f}, {r['ci_hi_dSharpe']:+.4f}] "
            + (
                "lower bound > -0.5 PASS"
                if r["dSharpe_ci_lo_above_-0.5"]
                else "lower bound <= -0.5 FAIL"
            )
            + "  -> "
            + ("WORTH IT" if r["worth_it"] else "not worth it")
        )
    n_pass = int(gate["worth_it"].sum())
    print(
        f"GATE F  {n_pass} of {len(gate)} cells are worth it "
        f"({len(WIDTH_FRACS)} widths x {len(BASES)} price bases tried)"
    )

    # -------------------------------------------------------- paragraph -----
    un = book.loc[
        (book["sample"] == "unseen") & (book["basis"] == "crossed")
    ].set_index("structure")
    nak_un = book.loc[(book["sample"] == "unseen") & (book["structure"] == NAKED)].iloc[
        0
    ]
    p_un = pair.loc[
        (pair["sample"] == "unseen") & (pair["basis"] == "crossed")
    ].set_index("structure")
    capped = [
        LABEL[wf]
        for wf in WIDTH_FRACS
        if float(un.loc[LABEL[wf], "cap_units_med"]) < CAP_BAR_UNITS
    ]
    print("\n--- G. one paragraph, numbers only")
    if capped:
        s0 = min(capped, key=lambda s: -float(p_un.loc[s, "mean_diff_pts"]))
        print(
            f"  On the {ENTRY} hold book, unseen sample, crossed, "
            f"{len(capped)} of the {len(WIDTH_FRACS)} widths cap the four "
            f"option legs under {CAP_BAR_UNITS:.0f} units of straddle premium: "
            + "; ".join(
                f"{s} median cap {float(un.loc[s, 'cap_units_med']):.4f} units "
                f"(${float(un.loc[s, 'cap_dollars_med']):,.0f} per contract), "
                f"costing {-float(p_un.loc[s, 'mean_diff_pts']):.4f} index "
                f"points a day per contract "
                f"(${-float(p_un.loc[s, 'mean_diff_pts']) * SPX_MULT:,.2f})"
                for s in capped
            )
            + f".  The cheapest of them per day is {s0}: wings "
            f"{float(un.loc[s0, 'wing_cost_pct_prem_med']):.2f}% of the straddle "
            f"premium at the ask (median), net credit "
            f"{float(un.loc[s0, 'net_credit_units_med']):.4f} units, "
            f"{-float(p_un.loc[s0, 'mean_diff_pts']):.4f} index points a day per "
            f"contract (${-float(p_un.loc[s0, 'mean_diff_pts']) * SPX_MULT:,.2f}) "
            f"over {int(p_un.loc[s0, 'n'])} sessions, HAC t "
            f"{float(p_un.loc[s0, 't_hac']):+.3f}, bootstrap CI of the paired "
            f"difference [{float(p_un.loc[s0, 'ci_lo_mean_pts']):+.4f}, "
            f"{float(p_un.loc[s0, 'ci_hi_mean_pts']):+.4f}] index points, which "
            f"takes the per-contract mean from {float(nak_un['mean_pts']):+.4f} "
            f"to {float(un.loc[s0, 'mean_pts']):+.4f} index points and the "
            f"per-contract Sharpe from {float(nak_un['Sharpe_ann_pts']):.4f} to "
            f"{float(un.loc[s0, 'Sharpe_ann_pts']):.4f} (d Sharpe "
            f"{float(p_un.loc[s0, 'd_Sharpe']):+.4f}, CI "
            f"[{float(p_un.loc[s0, 'ci_lo_dSharpe']):+.4f}, "
            f"{float(p_un.loc[s0, 'ci_hi_dSharpe']):+.4f}])."
        )
    else:
        print(
            f"  No width on the grid caps the four option legs under "
            f"{CAP_BAR_UNITS:.0f} units of premium on the {ENTRY} hold book; the "
            "smallest median cap is "
            f"{float(un['cap_units_med'].min()):.4f} units."
        )
    print(
        f"  The {ENTRY} book's medians on the whole grid, unseen sample, "
        "crossed: "
        + ", ".join(
            f"{wf * 100:.1f}% cap "
            f"{float(un.loc[LABEL[wf], 'cap_units_med']):.2f} u / "
            f"wings {float(un.loc[LABEL[wf], 'wing_cost_pct_prem_med']):.2f}% "
            f"of premium / Sharpe "
            f"{float(un.loc[LABEL[wf], 'Sharpe_ann_pts']):.2f}"
            for wf in WIDTH_FRACS
        )
        + f", naked {float(nak_un['Sharpe_ann_pts']):.2f}."
    )
    if REF29.exists():
        r29 = pd.read_csv(REF29)
        r29 = r29.loc[
            (r29["sample"] == "era")
            & (r29["basis"] == "crossed")
            & (r29["exit_mode"] == "hold")
        ].set_index("width")
        print(
            "  Proposal 29's 11:00 hold book on the same width grid (daily-0DTE "
            f"era, crossed, n {int(r29.loc[LABEL[WIDTH_FRACS[-1]], 'n'])}, read "
            f"from {REF29}): "
            + ", ".join(
                f"{wf * 100:.1f}% cap "
                f"{float(r29.loc[LABEL[wf], 'cap_units']):.2f} u / wings "
                f"{float(r29.loc[LABEL[wf], 'wing_cost_pct_prem']):.2f}% of "
                f"premium / Sharpe "
                f"{float(r29.loc[LABEL[wf], 'Sharpe_ann']):.2f}"
                for wf in WIDTH_FRACS
            )
            + f", naked {float(r29.loc['naked straddle', 'Sharpe_ann']):.2f}."
        )

    # ------------------------------------------------------------- write ----
    trad.to_csv(OUT / "a_wing_tradeability.csv", index=False)
    book.to_csv(OUT / "b_book_by_width.csv", index=False)
    pair.to_csv(OUT / "c_paired_vs_naked.csv", index=False)
    cost_tab.to_csv(OUT / "d_hedge_cost.csv", index=False)
    stress.to_csv(OUT / "e_stress.csv", index=False)
    gate.to_csv(OUT / "f_gate.csv", index=False)
    pd.DataFrame(daily).to_csv(OUT / "g_daily.csv")
    print(
        f"\nwrote {OUT}: a_wing_tradeability.csv, b_book_by_width.csv, "
        "c_paired_vs_naked.csv, d_hedge_cost.csv, e_stress.csv, f_gate.csv, "
        "g_daily.csv"
    )
    print(f"total runtime {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
