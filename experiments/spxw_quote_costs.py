"""Quote-cost surface for the 30-min 0DTE SPXW leg strategy.

The research spec (Aug 2026) proposes trading near-ATM SPXW calls and puts
separately, delta-hedged, once per 30-min bar, entering on the sign of a
price-form VRP signal s = ln(IV_ttc / sqrt(RVhat_ttc)) with a dead zone
theta.  Before any edge is sized we need the COST SURFACE that strategy has
to clear.  No trade sizes exist on disk, so this is a QUOTE-COST study --
what one round trip costs at the posted top of book -- not a capacity
study: depth, queue position and market impact are not measurable from this
file and are out of scope.

Measured per (entry hour ET, |vendor delta| bucket, side C/P, era):

  1. relative half-spread (ask-bid)/2/mid  -- median, p25, p75, p90
  2. absolute half-spread (ask-bid)/2      -- median, in index points
  3. crossing cost in VOL points: BS implied vol at the ask minus BS implied
     vol at the mid, reported in absolute vol and as a fraction of the mid
     IV.  This is the unit the spec's signal lives in: crossing once moves
     the quoted vol by that fraction, so a signal must clear roughly twice
     it to pay for a round trip.
  4. break-even signal floor theta_min = ln(IV_ask / IV_bid): the round-trip
     crossing cost expressed directly as a log-vol ratio, the same unit as
     s.  |s| below theta_min cannot be profitable at quoted spreads no
     matter how good the forecast -- it is a dead zone imposed by the
     market, not a tuning knob.
  5. quote availability: fraction of (day, 30-min stamp) snapshots carrying
     at least one both-sides-live contract in the bucket, and the median
     contract count per snapshot per bucket (absent buckets counted as 0).
  6. a compact hour x era table of the median relative half-spread for the
     pooled 0.30-0.70 delta band, both sides pooled.

Conventions follow experiments/spxw_delta_hedged_legs.py: r = 0, tau = hours
to 16:00 ET on the expiration day / (252 * 6.5), DST-aware through
America/New_York, and the terminal 16:00 stamp (tau = 0) is dropped.  The
vendor impl_volatility column is ignored (unknown unit); every IV here is
solved from the quote with scipy.optimize.brentq.  Vendor delta is used only
to bucket moneyness.

Two properties of this file drive filtering decisions and are re-derived at
run time rather than assumed:

  * `early_close` is True on EVERY row, so it carries no half-session
    information and is not used.  Half sessions are found instead by
    disagreement between the 16:00 clock and the vendor's own
    hours_to_expiration (6.5 at 09:30 on a full day, 3.5 on a 13:00 close);
    those days are dropped whole, because on them the 16:00 tau is wrong and
    the post-13:00 stamps carry stale, already-expired quotes.
  * at the 09:30 stamp the vendor supplies NEITHER underlying_price NOR
    delta on any row, so the opening bar cannot be bucketed by moneyness,
    IV-solved, or delta-hedged from this file.  Roughly 18% of its quotes
    are two-sided, but they are unusable; hour 9 survives in the
    availability table as a row of zeros, which is a data gap and not a
    market fact.  This file cannot cost the 09:30 entry.

No subsampling: the |delta| in [0.10, 0.90] live-quote band is small enough
(a few hundred thousand rows) that every expiration day is solved.
IV_DAY_STRIDE is the knob and is recorded in quote_costs_notes.csv.

Outputs results/spxw_pnl/quote_costs_{by_cell,breakeven,availability,
timeofday,notes}.csv.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "spxw_pnl")
HOURS_PER_YEAR = 252.0 * 6.5
DAILY_0DTE = pd.Timestamp("2022-05-16")
BUCKET_CUTS = (0.30, 0.50, 0.70)
BUCKET_LO, BUCKET_HI = 0.10, 0.90
BUCKET_LABELS = ("[0.10,0.30)", "[0.30,0.50)", "[0.50,0.70)", "[0.70,0.90]")
MID_BAND = BUCKET_LABELS[1:3]  # the pooled 0.30-0.70 band used for table 6
IV_DAY_STRIDE = 1  # 1 = every expiration day; >1 subsamples the IV solves
IV_LO, IV_HI = 1e-4, 5.0
ERAS = ("all", "daily_0dte")
QUANTS = (0.25, 0.50, 0.75, 0.90)


def _ncdf(x: float) -> float:
    """Standard normal CDF (erfc form; much faster than scipy inside a solver)."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def bs_price(S: float, K: float, T: float, sig: float, call: bool) -> float:
    if T <= 0.0 or sig <= 0.0:
        return max(S - K, 0.0) if call else max(K - S, 0.0)
    v = sig * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * v * v) / v
    d2 = d1 - v
    if call:
        return S * _ncdf(d1) - K * _ncdf(d2)
    return K * _ncdf(-d2) - S * _ncdf(-d1)


def bs_iv(price: float, S: float, K: float, T: float, call: bool) -> float:
    """BS implied vol of a quote; nan when the quote sits outside no-arb bounds."""
    if not (math.isfinite(price) and math.isfinite(S) and price > 0.0 and T > 0.0):
        return float("nan")
    intr = max(S - K, 0.0) if call else max(K - S, 0.0)
    upper = S if call else K
    if price <= intr + 1e-9 or price >= upper:
        return float("nan")
    try:
        return float(
            brentq(
                lambda s: bs_price(S, K, T, s, call) - price, IV_LO, IV_HI, xtol=1e-6
            )
        )
    except (ValueError, RuntimeError):
        return float("nan")


def _bucket_index(adelta: np.ndarray) -> np.ndarray:
    """0..3 for the four |delta| buckets, -1 outside [0.10, 0.90]."""
    idx = np.searchsorted(np.asarray(BUCKET_CUTS), adelta, side="right").astype(int)
    idx[~((adelta >= BUCKET_LO) & (adelta <= BUCKET_HI))] = -1
    return idx


def _load() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Return (live in-band quotes with tau, snapshot universe, drop counts)."""
    ch = pd.read_parquet(
        os.path.join(ROOT, "data", "spxw_chain.parquet"),
        columns=[
            "expiration",
            "timestamp",
            "strike",
            "cp",
            "bid",
            "ask",
            "mid",
            "underlying_price",
            "delta",
            "hours_to_expiration",
        ],
    )
    ch["expiration"] = pd.to_datetime(ch["expiration"]).dt.normalize()
    ts = pd.to_datetime(ch["timestamp"], utc=True)
    ch["hour_et"] = ts.dt.tz_convert("America/New_York").dt.hour.to_numpy()

    # 16:00 ET on the expiration day, DST-aware, expressed in UTC.
    settle = (
        (ch["expiration"].dt.tz_localize("America/New_York") + pd.Timedelta(hours=16))
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )
    hrs = (settle.to_numpy() - ts.dt.tz_localize(None).to_numpy()) / np.timedelta64(
        1, "h"
    )
    ch["tau"] = hrs.astype(float) / HOURS_PER_YEAR

    # Half sessions: the vendor clock disagrees with the 16:00 close on every
    # row of the day, so this is a whole-day drop, not a per-row one.
    n_days_all = int(ch["expiration"].nunique())
    agrees = (
        np.abs(hrs.astype(float) - ch["hours_to_expiration"].to_numpy(float)) < 1e-3
    )
    full_days = ch.loc[agrees, "expiration"].unique()
    ch = ch[ch["expiration"].isin(full_days)]
    n_days_full = int(ch["expiration"].nunique())

    ch = ch[ch["tau"] > 0.0]  # drop the 16:00 settle stamp: nothing to trade
    drops = {"days_all": n_days_all, "days_full_session": n_days_full}

    # Snapshot universe = every (expiration day, 30-min stamp) the chain shows,
    # whether or not a given bucket has a live contract in it.
    snaps = (
        ch[["expiration", "timestamp", "hour_et"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    quoted = (ch["bid"] > 0) & (ch["mid"] > 0) & (ch["ask"] >= ch["bid"])
    usable = np.isfinite(ch["underlying_price"]) & np.isfinite(ch["delta"])
    drops["quotes_two_sided"] = int(quoted.sum())
    drops["quotes_two_sided_no_spot_or_delta"] = int((quoted & ~usable).sum())
    live = ch[quoted & usable].copy()
    bidx = _bucket_index(live["delta"].abs().to_numpy(float))
    live = live[bidx >= 0].copy()
    live["bucket"] = np.asarray(BUCKET_LABELS, dtype=object)[bidx[bidx >= 0]]

    half = (live["ask"].to_numpy(float) - live["bid"].to_numpy(float)) / 2.0
    live["hs_abs"] = half
    live["hs_rel"] = half / live["mid"].to_numpy(float)
    return live.reset_index(drop=True), snaps, drops


def _solve_ivs(live: pd.DataFrame) -> pd.DataFrame:
    """Add iv_bid / iv_mid / iv_ask on a strided-day subsample of `live`."""
    days = np.sort(live["expiration"].unique())[::IV_DAY_STRIDE]
    sub = live[live["expiration"].isin(days)].copy()
    S = sub["underlying_price"].to_numpy(float)
    K = sub["strike"].to_numpy(float)
    T = sub["tau"].to_numpy(float)
    call = (sub["cp"] == "C").to_numpy()
    t0 = time.time()
    for name in ("bid", "mid", "ask"):
        p = sub[name].to_numpy(float)
        sub[f"iv_{name}"] = np.fromiter(
            (bs_iv(p[i], S[i], K[i], T[i], bool(call[i])) for i in range(len(sub))),
            dtype=float,
            count=len(sub),
        )
        print(f"  iv_{name} solved ({time.time() - t0:.0f}s)", flush=True)
    ok = (
        np.isfinite(sub["iv_bid"].to_numpy(float))
        & np.isfinite(sub["iv_mid"].to_numpy(float))
        & np.isfinite(sub["iv_ask"].to_numpy(float))
    )
    sub["iv_ok"] = ok
    sub["vol_cost_abs"] = np.where(
        ok, sub["iv_ask"].to_numpy(float) - sub["iv_mid"].to_numpy(float), np.nan
    )
    sub["vol_cost_frac"] = sub["vol_cost_abs"] / sub["iv_mid"]
    sub["theta_min"] = np.where(
        ok,
        np.log(sub["iv_ask"].to_numpy(float) / sub["iv_bid"].to_numpy(float)),
        np.nan,
    )
    return sub


def _era_mask(df: pd.DataFrame, era: str) -> np.ndarray:
    if era == "all":
        return np.ones(len(df), bool)
    return (df["expiration"] >= DAILY_0DTE).to_numpy(bool)


def _by_cell(live: pd.DataFrame, iv: pd.DataFrame) -> pd.DataFrame:
    """Tables 1-3: spread and vol-cost distribution per (hour, bucket, side, era)."""
    rows = []
    for era in ERAS:
        gl = live[_era_mask(live, era)].groupby(["hour_et", "bucket", "cp"], sort=True)
        gv = iv[_era_mask(iv, era)].groupby(["hour_et", "bucket", "cp"], sort=True)
        stats = gl["hs_rel"].quantile(list(QUANTS)).unstack()
        abs_med, mid_med, n = gl["hs_abs"].median(), gl["mid"].median(), gl.size()
        ivm, vca = gv["iv_mid"].median(), gv["vol_cost_abs"].median()
        vcf, thm = gv["vol_cost_frac"].median(), gv["theta_min"].median()
        n_iv, ok_rate = gv["iv_ok"].sum(), gv["iv_ok"].mean()
        for key in stats.index:
            hour, bucket, cp = key
            rows.append(
                {
                    "era": era,
                    "hour_et": int(hour),
                    "bucket": bucket,
                    "cp": cp,
                    "n_quotes": int(n.loc[key]),
                    "mid_med": float(mid_med.loc[key]),
                    "hs_rel_p25": float(stats.loc[key, 0.25]),
                    "hs_rel_med": float(stats.loc[key, 0.50]),
                    "hs_rel_p75": float(stats.loc[key, 0.75]),
                    "hs_rel_p90": float(stats.loc[key, 0.90]),
                    "hs_abs_med": float(abs_med.loc[key]),
                    "n_iv_solved": int(n_iv.get(key, 0)),
                    "iv_solve_rate": float(ok_rate.get(key, np.nan)),
                    "iv_mid_med": float(ivm.get(key, np.nan)),
                    "vol_cost_abs_med": float(vca.get(key, np.nan)),
                    "vol_cost_frac_med": float(vcf.get(key, np.nan)),
                    "theta_min_med": float(thm.get(key, np.nan)),
                }
            )
    return pd.DataFrame(rows)


def _breakeven(iv: pd.DataFrame) -> pd.DataFrame:
    """Table 4: the dead-zone floor theta_min per (hour, bucket, era), sides pooled."""
    rows = []
    for era in ERAS:
        vo = iv[_era_mask(iv, era) & iv["iv_ok"].to_numpy(bool)]
        g = vo.groupby(["hour_et", "bucket"], sort=True)
        thm = g["theta_min"].quantile(list(QUANTS)).unstack()
        vcf, vca, ivm, n = (
            g["vol_cost_frac"].median(),
            g["vol_cost_abs"].median(),
            g["iv_mid"].median(),
            g.size(),
        )
        for key in thm.index:
            hour, bucket = key
            rows.append(
                {
                    "era": era,
                    "hour_et": int(hour),
                    "bucket": bucket,
                    "n_iv": int(n.loc[key]),
                    "iv_mid_med": float(ivm.loc[key]),
                    "vol_cost_abs_med": float(vca.loc[key]),
                    "oneway_frac_med": float(vcf.loc[key]),
                    "theta_min_p25": float(thm.loc[key, 0.25]),
                    "theta_min_med": float(thm.loc[key, 0.50]),
                    "theta_min_p75": float(thm.loc[key, 0.75]),
                    "theta_min_p90": float(thm.loc[key, 0.90]),
                }
            )
    return pd.DataFrame(rows)


def _availability(live: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """Table 5: can the strategy even find a quote in the bucket it wants?"""
    cnt = (
        live.groupby(["expiration", "timestamp", "hour_et", "bucket"], sort=False)
        .size()
        .rename("n")
        .reset_index()
    )
    rows = []
    for era in ERAS:
        sn = snaps[_era_mask(snaps, era)]
        cn = cnt[_era_mask(cnt, era)]
        denom = sn.groupby("hour_et").size()
        for bucket in BUCKET_LABELS:
            cb = cn[cn["bucket"] == bucket]
            live_snaps = cb.groupby("hour_et").size()
            for hour in denom.index:
                d = int(denom.loc[hour])
                have = int(live_snaps.get(hour, 0))
                counts = cb.loc[cb["hour_et"] == hour, "n"].to_numpy(float)
                padded = np.concatenate([counts, np.zeros(max(d - have, 0))])
                rows.append(
                    {
                        "era": era,
                        "hour_et": int(hour),
                        "bucket": bucket,
                        "n_snapshots": d,
                        "frac_snapshots_live": have / d if d else float("nan"),
                        "n_contracts_med": float(np.median(padded)),
                        "n_contracts_p25": float(np.percentile(padded, 25)),
                        "n_contracts_p75": float(np.percentile(padded, 75)),
                    }
                )
    return pd.DataFrame(rows)


def _timeofday(live: pd.DataFrame) -> pd.DataFrame:
    """Table 6: hour x era median relative half-spread, 0.30-0.70 band, sides pooled."""
    band = live[live["bucket"].isin(MID_BAND)]
    parts = []
    for era in ERAS:
        g = band[_era_mask(band, era)].groupby("hour_et")
        parts.append(
            pd.DataFrame(
                {
                    f"n_{era}": g.size(),
                    f"hs_rel_med_{era}": g["hs_rel"].median(),
                    f"hs_abs_med_{era}": g["hs_abs"].median(),
                    f"mid_med_{era}": g["mid"].median(),
                }
            )
        )
    wide = pd.concat(parts, axis=1)
    wide.index.name = "hour_et"
    return wide.reset_index()


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    live, snaps, drops = _load()
    print(
        f"in-band live quotes: {len(live)} over {live['expiration'].nunique()} days, "
        f"{len(snaps)} snapshots ({time.time() - t0:.0f}s); "
        f"half sessions dropped: {drops['days_all'] - drops['days_full_session']}",
        flush=True,
    )
    iv = _solve_ivs(live)
    print(f"iv rows: {len(iv)}, solved {iv['iv_ok'].mean():.4f}", flush=True)

    by_cell = _by_cell(live, iv)
    breakeven = _breakeven(iv)
    avail = _availability(live, snaps)
    tod = _timeofday(live)

    notes = pd.DataFrame(
        [
            ("data", "data/spxw_chain.parquet"),
            ("filter", "bid>0 & mid>0 & ask>=bid & finite underlying & finite delta"),
            ("stamps", "30-min RTH; the terminal 16:00 ET stamp (tau=0) is dropped"),
            ("tau", "hours to 16:00 ET on the expiration day / (252*6.5), DST-aware"),
            ("rate", "r = 0"),
            ("iv", "solved from the quote with scipy brentq; vendor IV column ignored"),
            ("moneyness", "vendor delta column, |delta| buckets"),
            ("early_close", "column is True on every row: uninformative, not used"),
            (
                "half_sessions",
                "days whose vendor hours_to_expiration disagrees with the 16:00 "
                f"clock are dropped whole: {drops['days_all']} days on file, "
                f"{drops['days_full_session']} kept",
            ),
            (
                "hour_9_gap",
                "the 09:30 stamp carries neither underlying_price nor delta on any "
                "row, so the opening bar cannot be bucketed, IV-solved or hedged "
                "from this file; it shows as 0 availability (a data gap, not a "
                "market fact)",
            ),
            (
                "two_sided_quotes",
                f"{drops['quotes_two_sided']} two-sided rows, of which "
                f"{drops['quotes_two_sided_no_spot_or_delta']} carry no "
                "underlying_price/delta and are unusable",
            ),
            ("iv_day_stride", str(IV_DAY_STRIDE)),
            (
                "iv_subsampling",
                "none: every expiration day solved"
                if IV_DAY_STRIDE == 1
                else f"every {IV_DAY_STRIDE}th expiration day solved",
            ),
            ("iv_rows", str(len(iv))),
            ("iv_solve_rate", f"{float(iv['iv_ok'].mean()):.4f}"),
            ("live_quotes", str(len(live))),
            ("snapshots", str(len(snaps))),
            ("days", str(int(live["expiration"].nunique()))),
            ("era_daily_0dte", "expiration >= 2022-05-16"),
            ("theta_min", "ln(IV_ask/IV_bid): round-trip crossing cost in log-vol"),
            ("oneway_frac", "(IV_ask-IV_mid)/IV_mid: one-way crossing, fraction of IV"),
            ("scope", "quote cost only; depth/impact/capacity not measurable here"),
            ("runtime_s", f"{time.time() - t0:.0f}"),
        ],
        columns=["key", "value"],
    )

    by_cell.to_csv(os.path.join(OUT, "quote_costs_by_cell.csv"), index=False)
    breakeven.to_csv(os.path.join(OUT, "quote_costs_breakeven.csv"), index=False)
    avail.to_csv(os.path.join(OUT, "quote_costs_availability.csv"), index=False)
    tod.to_csv(os.path.join(OUT, "quote_costs_timeofday.csv"), index=False)
    notes.to_csv(os.path.join(OUT, "quote_costs_notes.csv"), index=False)

    pd.set_option("display.width", 240)
    fmt4 = "{:.4f}".format
    fmt3 = "{:.3f}".format
    print("\n=== table 4: break-even theta_min = ln(IV_ask/IV_bid), median ===")
    print(
        breakeven.pivot_table(
            index="hour_et", columns=["era", "bucket"], values="theta_min_med"
        ).to_string(float_format=fmt4)
    )
    print("\n=== table 6: median relative half-spread, 0.30-0.70 band ===")
    print(
        tod[["hour_et"] + [f"hs_rel_med_{e}" for e in ERAS]].to_string(
            index=False, float_format=fmt4
        )
    )
    print("\n=== table 5: frac of snapshots with a live contract in the bucket ===")
    print(
        avail.pivot_table(
            index="hour_et", columns=["era", "bucket"], values="frac_snapshots_live"
        ).to_string(float_format=fmt3)
    )
    print(f"\nwrote 5 CSVs to {OUT} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
