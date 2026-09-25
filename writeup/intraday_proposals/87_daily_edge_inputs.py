"""Study 87 -- which post-2024 input change, if any, explains the live card's loss?

QUESTION (the operator's).  The daily card -- buy the SPXW 0DTE nearest-OTM
strangle at 15:30 iff its ask <= P* = ``live.ibkr.pricing.package_price``
(sqrt(rv_hat), S, Kc, Kp), hold to the 16:00 settlement, month-ends excluded --
made +0.0455/day at the SPXW 15:30 ask on the research deck's 814
non-month-end days 2020-01 .. 2024-04 (``daily_sub_live_ridge``: the per-bar
ridge of specs/causal_tune_linear.py, bucket live_feasible, vendor panel).
Study 85 found the LIVE card's pipeline (bucket free_vix_only, extended panel
= vendor rows + live/close_signal/state/panel_free.parquet) at -0.0129/day on
SPX 2024-05 .. 2025-12.  The post-2024 inputs differ from research in three
ways: (a) VIX / VVIX / VIX3M from 2024-02-13 are Yahoo HOURLY prints
(``cboe_gap_yahoo``: exact print every other stamp, the half hours carry it,
every other stamp carries a close -- so EVERY row is "observed"); (b) ES
30-minute moments after 2024-04-30 are built from Databento 1-minute bars
(level shifts vs the vendor, and no 17:30 / 18:00 / Friday-evening rows);
(c) the bucket.  Which of them, if any, kills the edge -- or is it the period?

DESIGN.  Every test runs on the RESEARCH period, where the full-quality inputs
exist, through study 80's in-process engine (``run_horizon`` at h = 1: the 13
bar arms of the spec in one ``run_executor`` call, argument for argument; only
the bucket and the data directory change), the deck's MZ loader
(``forecast.recalibrate``) and the SPXW 15:30 book of study 79
(``spxw_book``: chain stamps TRUE UTC -> ET, ask > 0, nearest-OTM on the
listed strikes).  One change per variant:

  deck      live_feasible, vendor panel as is (the reproduction gate)
  T1        free_vix_only, vendor panel (the live card's feature set)
  T2        live_feasible, vendor VIX / VVIX / VIX3M degraded to the Yahoo form
            from DEG_START on: the vendor's own print at each hourly bar END
            (VIX :00 stamps 04:00 .. 17:00, VVIX / VIX3M :30 stamps 10:30 ..
            16:30, read off cboe_gap_yahoo's tags) fed as hourly bars into
            ``ingest.cboe_rows_from_yahoo`` itself (same stagger, same-day
            carry, previous-session daily close from the Yahoo daily files
            before the first print and on days without bars).  DEG_START is
            the live analog: as many sessions before the first deck day as
            2024-02-13 is before 2024-05-01, so the ridge's window mixes the
            two forms as the live one does.  Gate G2: the construction on the
            vendor overlap 2023-12-07 .. 2024-02-12 against the same function
            fed the REAL Yahoo hourly snapshot.
  T2m       T2 masked to the vendor's own observation pattern (cells outside
            the stamps the vendor prints -- VIX 04:00-17:00, VVIX 10:00-16:00,
            VIX3M 10:00-16:30 -- and on non-Cboe days set NaN): the staleness
            alone, without the carries that make every row "observed".
  T2full    T2 over the whole history (a consistently hourly model).
  T1T2      free_vix_only + T2 (the live model with the live VIX form).
  T3        live_feasible, vendor ES moments from the first deck day on
            multiplied by the MEASURED Databento/vendor median ratios (by
            column; RTH 10:00-16:00, evening 19:00-20:00, other) on the
            2024-03-31 .. 04-30 overlap, and the vendor-only rows (the
            (weekday, stamp) pairs the vendor prints and Databento does not on
            the overlap) dropped.  A CRUDE sensitivity: the level shift is
            the median, the per-stamp noise is not modelled.
  ALL       free_vix_only + T2 + T3: the post-2024 live input quality on the
            research years.
Post-2024, on the live pipeline itself (extended panel as
``forecast.write_ext_data`` builds it from panel_free at HEAD, placeholders and
stamps from 2026-09-24 dropped, as study 85):
  post_live        free_vix_only on the extended panel (gate G3: == study 85's
                   15:30 card rv_hat and its SPX P&L)
  post_fix_mask    the Yahoo cells from 2024-02-13 masked to the vendor pattern
  post_fix_consist the vendor's VIX history converted to the Yahoo form (T2full's
                   cells up to 2024-02-12), the live cells as they are

SCORING.  Research: the 866 deck days, headline on the 814 non-month-end ones.
P* on the book's pair and spot, buy iff ask <= P*, R = payoff / ask - 1 at the
deck's S_close, per-day P&L = R on a buy day, 0 otherwise.  Per variant:
QLIKE (vs the VENDOR 16:00-bar realized variance, the deck's rv_raw), buys,
per-day P&L and its iid t, mean R on buy days and t, decision agreement with
the deck, the paired circular 20-session block bootstrap (10,000 draws) of the
per-day P&L difference vs the deck.  Post: study 85's SPX 15:30 books (395
non-month-end, non-early-close sessions 2024-05 .. 2025-12), the same numbers
vs post_live.  Period (T4): the unconditional long (buy every day at the ask)
and the card rule by calendar half-year 2020 .. 2025 on SPX.

INHERITED CAVEAT (as studies 80 and 85): the pipeline's full-sample medians
(impute fill, signed-diurnal floor) move with any input change, so every
variant also moves them -- as the live card's own inputs do.

OUTPUTS (results/atm_straddle_intraday_holdclose/proposals/87/):
  arms/<variant>/h1/causal_tune_linear/results_bar*.csv  raw arm outputs
  gate.json, vix_form_gate.csv, vix_form_diagnostics.csv, es_shift_factors.csv,
  forecasts.parquet, per_day_pnl.parquet/.csv, summary.csv, bootstrap.csv,
  regime_halfyear.csv, period_bootstrap.csv, summary.txt, run.log (stdout)

REPRODUCE (measured 2026-09-24: 12 engine runs of 74-160 s each, run as two
--engine-only invocations of 6 with one worker each in parallel, 740 s / 804 s,
worker peak ~2.2 GB; then the scoring pass with --reuse-arms, 553 s):
    C:/Users/james/miniconda3/envs/285J/python.exe writeup/intraday_proposals/87_daily_edge_inputs.py
    (--reuse-arms skips engine runs whose 13 CSVs exist; --workers 2; --only a,b;
     --engine-only stops after the arms; --inputs-only after the data roots)
G3 note: post_live's rv_hat is 1.3e-8 relative from study 85's (study 85 ran a
frozen commit, 85c2192; the MZ library and forecast.py moved since) -- above
the pre-set 1e-9 bar, inside the deck gate's 1e-7; all 395 decisions and the
P&L are identical.  gate.json carries both flags.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PROPOSALS = REPO / "results" / "atm_straddle_intraday_holdclose" / "proposals"
OUT = PROPOSALS / "87"
DATA = REPO / "data"
DECK = REPO / "results" / "atm_straddle_0dte_1530" / "daily_sub_live_ridge.parquet"
RESEARCH_TABLE = REPO / "results" / "spxw_pnl" / "yhat_sub_ridge_live_feasible.parquet"
STATE = REPO / "live" / "close_signal" / "state"
CBOE_GAP = STATE / "cboe_gap_yahoo.parquet"
PANEL_FREE = STATE / "panel_free.parquet"
FOMC_CSV = STATE / "fomc_statement_dates.csv"
FREE_FEED = DATA / "free_feed"
S85 = PROPOSALS / "85"
VENDOR_FILES = (
    "core_stats.parquet",
    "vix_and_voldemand.parquet",
    "releases.parquet",
    "time_categories.parquet",
)
CBOE = ("vix", "vvix", "vix3m")
ES_COLS = (
    "sumret",
    "sumabsret",
    "sumret2",
    "sumret3",
    "sumret4",
    "sumpret2",
    "sumbipow",
    "sumautocov",
    "sumvolume",
    "numobs",
)
INTENSITY = ("sumabsret", "sumret2", "sumret4", "sumpret2", "sumbipow", "sumvolume")
LIVE_YAHOO_START = pd.Timestamp("2024-02-13")  # ingest.GAP_START
LIVE_TEST_START = pd.Timestamp("2024-05-01")  # study 85's FIRST_DAY
OVERLAP_CBOE = (pd.Timestamp("2023-12-07"), pd.Timestamp("2024-02-12"))
OVERLAP_ES = (pd.Timestamp("2024-03-31"), pd.Timestamp("2024-04-30 23:30"))
PANEL_END = pd.Timestamp("2026-09-24")  # study 85: today's live rows dropped
HOUR = pd.Timedelta(hours=1)
#: the vendor stamps whose print the day's last Yahoo hourly bar closes at (see synthetic_hourly)
LAST_BAR_CLOSE_AT = {"vix": ("17:00", "16:30"), "vvix": ("16:00",), "vix3m": ("16:00",)}
BLOCK = 20
N_BOOT = 10_000
SEED = 87
RV_HAT_REL_TOL = 1e-7  # study 80's gate tolerances
YHAT_REL_TOL = 1e-6
POST_REL_TOL = 1e-9

#: variant -> (bucket, root kind, cboe modification, es modification, period)
VARIANTS: dict[str, tuple[str, str, str | None, str | None, str]] = {
    "deck": ("live_feasible", "vendor", None, None, "research"),
    "T1": ("free_vix_only", "vendor", None, None, "research"),
    "T2": ("live_feasible", "vendor", "yahoo_live_analog", None, "research"),
    "T2m": ("live_feasible", "vendor", "yahoo_live_analog_masked", None, "research"),
    "T1T2": ("free_vix_only", "vendor", "yahoo_live_analog", None, "research"),
    "T3": ("live_feasible", "vendor", None, "databento_shift", "research"),
    "ALL": (
        "free_vix_only",
        "vendor",
        "yahoo_live_analog",
        "databento_shift",
        "research",
    ),
    "T2full": ("live_feasible", "vendor", "yahoo_full_history", None, "research"),
    "T1T2m": ("free_vix_only", "vendor", "yahoo_live_analog_masked", None, "research"),
    "post_live": ("free_vix_only", "ext", None, None, "post"),
    "post_fix_mask": ("free_vix_only", "ext", "mask_post", None, "post"),
    "post_fix_consist": (
        "free_vix_only",
        "ext",
        "yahoo_full_history_pre",
        None,
        "post",
    ),
}
RUN_ORDER = (
    "deck",
    "T1",
    "T2",
    "T2m",
    "T1T2",
    "post_live",
    "post_fix_mask",
    "T3",
    "ALL",
    "T2full",
    "post_fix_consist",
    "T1T2m",
)

_LOG: list[str] = []


def log(msg: str = "") -> None:
    _LOG.append(msg)
    print(msg, flush=True)


def _import(rel: str, name: str):
    p = REPO / rel
    spec = importlib.util.spec_from_file_location(name, p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _p80():
    return _import("writeup/intraday_proposals/80_earliest_decision_clock.py", "p80")


def _p79():
    return _import("writeup/intraday_proposals/79_xsp_close_book.py", "p79")


# --------------------------------------------------------------------------- engine
def run_variant(name: str, bucket: str, data_dir: str, out_dir: str) -> dict[str, str]:
    """Study 80's run_horizon at h = 1 with the bucket swapped (its module global)."""
    t0 = time.time()
    p80 = _p80()
    p80.BUCKET = bucket
    csvs = p80.run_horizon(1, data_dir, out_dir)
    print(f"[{name}] {bucket} on {data_dir}: {time.time() - t0:.0f}s", flush=True)
    return csvs


def arm_csvs(name: str) -> dict[str, Path]:
    from live.close_signal.forecast import BARS

    base = OUT / "arms" / name / "h1" / "causal_tune_linear"
    return {b: base / f"results_{b}.csv" for b in BARS}


# --------------------------------------------------------------------------- VIX form
def _read_daily() -> dict[str, pd.Series]:
    from live.close_signal.ingest import read_yahoo_daily

    return {c: read_yahoo_daily(FREE_FEED / f"{c}_1d_yahoo.parquet") for c in CBOE}


def _read_hourly() -> dict[str, pd.DataFrame]:
    from live.close_signal.ingest import read_yahoo_hourly

    return {c: read_yahoo_hourly(FREE_FEED / f"{c}_1h_yahoo.parquet") for c in CBOE}


def hourly_end_slots() -> dict[str, list[str]]:
    """The stamps a Yahoo hourly bar ENDS on, per column, read off cboe_gap_yahoo's tags."""
    g = pd.read_parquet(CBOE_GAP)
    t = pd.to_datetime(g["endbartime"])
    tod = t.dt.strftime("%H:%M")
    out = {}
    for c in CBOE:
        is_h = g[f"{c}_source"] == "hourly"
        n_days = t[is_h].dt.normalize().nunique()
        frac = is_h.groupby(tod).sum() / n_days
        out[c] = sorted(frac.index[frac >= 0.5])
    return out


def vendor_slots(vend: pd.DataFrame) -> dict[str, list[str]]:
    """The stamps the vendor prints each column on (>= half of 2021-2023 weekdays)."""
    t = vend["endbartime"]
    w = vend[(t >= "2021-01-01") & (t < "2024-01-01") & (t.dt.dayofweek < 5)]
    tod = w["endbartime"].dt.strftime("%H:%M")
    fr = w[list(CBOE)].notna().groupby(tod).mean()
    return {c: sorted(fr.index[fr[c] >= 0.5]) for c in CBOE}


def synthetic_hourly(
    vend: pd.DataFrame,
    daily: dict[str, pd.Series],
    slots: dict[str, list[str]],
    lo: pd.Timestamp,
    hi: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Hourly bars (index = bar START, close) an hourly feed would have given, from the vendor.

    The vendor's value at T is the print standing just before T and a Yahoo
    hourly bar ending at T closes at exactly that print (the ingest's measured
    convention), so the bar ending at T closes at the vendor's T value.  The
    day's LAST bar is the exception, measured on the overlap against the real
    snapshot: the VVIX / VIX3M 15:30-16:30 bar closes at the vendor's 16:00
    print (45 / 45 days); the VIX 16:00-17:00 bar at the vendor's 17:00 / 16:30
    print (the close; 51 %, the other days within one tick).  Where none of
    those exists, the day's daily close.
    """
    vi = vend.set_index("endbartime")
    out = {}
    for c in CBOE:
        d = daily[c]
        days = d.index[(d.index >= lo.normalize()) & (d.index <= hi.normalize())]
        offs = [pd.Timedelta(hours=int(s[:2]), minutes=int(s[3:])) for s in slots[c]]
        ends = pd.DatetimeIndex(
            np.concatenate([(days + o).to_numpy() for o in offs])
        ).sort_values()
        vals = vi[c].reindex(ends).to_numpy(float)
        last = ends.strftime("%H:%M") == slots[c][-1]
        for src in LAST_BAR_CLOSE_AT[c]:
            at = (
                vi[c]
                .reindex(ends.normalize() + pd.Timedelta(src + ":00"))
                .to_numpy(float)
            )
            first = src == LAST_BAR_CLOSE_AT[c][0]
            vals = np.where(last & (first | ~np.isfinite(vals)), at, vals)
        close_same_day = d.reindex(ends.normalize()).to_numpy(float)
        vals = np.where(last & ~np.isfinite(vals), close_same_day, vals)
        ok = np.isfinite(vals)
        out[c] = pd.DataFrame({"close": vals[ok]}, index=ends[ok] - HOUR)
    return out


def yahoo_form(
    hourly: dict[str, pd.DataFrame],
    daily: dict[str, pd.Series],
    lo: pd.Timestamp,
    hi: pd.Timestamp,
) -> pd.DataFrame:
    """``ingest.cboe_rows_from_yahoo`` itself: every stamp of every calendar day in [lo, hi]."""
    from live.close_signal.ingest import cboe_rows_from_yahoo

    return cboe_rows_from_yahoo(hourly, daily, lo, hi)


def mask_to_vendor_pattern(
    frame: pd.DataFrame,
    sel: np.ndarray,
    vslots: dict[str, list[str]],
    daily: dict[str, pd.Series],
) -> pd.DataFrame:
    """NaN the CBOE cells (rows ``sel``) outside the vendor's slots or on non-Cboe days."""
    f = frame.copy()
    t = pd.DatetimeIndex(f["endbartime"])
    tod = t.strftime("%H:%M")
    for c in CBOE:
        on_day = t.normalize().isin(daily[c].index)
        keep = np.isin(tod, vslots[c]) & on_day
        f.loc[sel & ~keep, c] = np.nan
    return f


def apply_form(
    vix_file: pd.DataFrame, form: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp
) -> tuple[pd.DataFrame, dict]:
    """The vix file with the CBOE cells of [lo, hi] replaced by the form's (NaN stays NaN)."""
    v = vix_file.copy()
    t = pd.to_datetime(v["endbartime"])
    v["endbartime"] = t
    sel = ((t >= lo) & (t <= hi)).to_numpy()
    f = form.set_index("endbartime").reindex(t[sel])
    info = {}
    for c in CBOE:
        new = f[c].to_numpy(float)
        old = v.loc[sel, c].to_numpy(float)
        info[f"{c}_vendor_finite_form_nan"] = int(
            (np.isfinite(old) & ~np.isfinite(new)).sum()
        )
        # a vendor print the form cannot reach (no Yahoo daily yet) is kept as is
        new = np.where(~np.isfinite(new) & np.isfinite(old), old, new)
        v.loc[sel, c] = new
        info[f"{c}_cells_changed"] = int(
            (
                ~np.isclose(
                    np.nan_to_num(old, nan=-1),
                    np.nan_to_num(new, nan=-1),
                    rtol=0,
                    atol=1e-12,
                )
            ).sum()
        )
    return v, info


# --------------------------------------------------------------------------- ES shift
def es_shift_factors(core: pd.DataFrame, pf: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Databento / vendor median log ratios on the overlap, by column and stamp class."""
    es = pf[pf["source"] == "databento_es"].copy()
    es["endbartime"] = pd.to_datetime(es["endbartime"])
    v = core[
        (core["endbartime"] >= OVERLAP_ES[0]) & (core["endbartime"] <= OVERLAP_ES[1])
    ]
    e = es[(es["endbartime"] >= OVERLAP_ES[0]) & (es["endbartime"] <= OVERLAP_ES[1])]
    j = v.set_index("endbartime")[list(ES_COLS)].join(
        e.set_index("endbartime")[list(ES_COLS)], rsuffix="_db", how="inner"
    )
    cls = stamp_class(pd.DatetimeIndex(j.index))
    rows = []
    for col in INTENSITY:
        for k in ("rth", "eve", "other"):
            m = cls == k
            a, b = j.loc[m, col].to_numpy(float), j.loc[m, col + "_db"].to_numpy(float)
            ok = (a > 0) & (b > 0)
            lr = np.log(b[ok] / a[ok])
            rows.append(
                {
                    "column": col,
                    "stamp_class": k,
                    "stamps": int(ok.sum()),
                    "median_log_ratio": float(np.median(lr)),
                    "factor": float(np.exp(np.median(lr))),
                    "mad_log_ratio": float(np.median(np.abs(lr - np.median(lr)))),
                    "frac_identical": float(np.mean(np.abs(lr) < 1e-12)),
                    "sum_ratio": float(b[ok].sum() / a[ok].sum()),
                }
            )
    # vendor-only rows: (weekday, stamp) pairs the vendor prints and Databento does not
    vs = set(v.loc[v["sumret2"].notna(), "endbartime"])
    ds = set(e.loc[e["sumret2"].notna(), "endbartime"])
    only = pd.DatetimeIndex(sorted(vs - ds))
    pairs = pd.Series(
        1, index=pd.MultiIndex.from_arrays([only.dayofweek, only.strftime("%H:%M")])
    )
    cnt = pairs.groupby(level=[0, 1]).sum()
    drop = [(int(a), str(b)) for (a, b), n in cnt.items() if n >= 2]
    return pd.DataFrame(rows), drop


def stamp_class(t: pd.DatetimeIndex) -> np.ndarray:
    mod = t.hour * 60 + t.minute
    return np.where(
        (mod >= 600) & (mod <= 960),
        "rth",
        np.where((mod >= 1140) & (mod <= 1200), "eve", "other"),
    )


def apply_es(
    core: pd.DataFrame, fac: pd.DataFrame, drop: list, start: pd.Timestamp
) -> tuple[pd.DataFrame, dict]:
    c = core.copy()
    t = pd.DatetimeIndex(c["endbartime"])
    sel = t >= start
    cls = stamp_class(t)
    for _, r in fac.iterrows():
        m = sel & (cls == r["stamp_class"])
        c.loc[m, r["column"]] = c.loc[m, r["column"]] * r["factor"]
    key = pd.MultiIndex.from_arrays([t.dayofweek, t.strftime("%H:%M")])
    hit = sel & key.isin(drop) & c["sumret2"].notna().to_numpy()
    c.loc[hit, list(ES_COLS)] = np.nan
    return c, {
        "rows_dropped": int(hit.sum()),
        "drop_pairs": [f"{a}:{b}" for a, b in drop],
    }


# --------------------------------------------------------------------------- scoring
def qlike(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    r = y / f
    return r - np.log(r) - 1.0


def tstat(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def score(
    rv_hat: pd.Series,
    S: pd.Series,
    kc: pd.Series,
    kp: pd.Series,
    ask: pd.Series,
    payoff: pd.Series,
) -> pd.DataFrame:
    from live.ibkr.pricing import package_price

    ps = np.array(
        [
            package_price(np.sqrt(v), s, a, b)
            if np.isfinite(v) and np.isfinite(s)
            else np.nan
            for v, s, a, b in zip(rv_hat, S, kc, kp)
        ]
    )
    buy = ask.to_numpy(float) <= ps
    R = payoff.to_numpy(float) / ask.to_numpy(float) - 1.0
    return pd.DataFrame(
        {
            "rv_hat": rv_hat.to_numpy(float),
            "P_star": ps,
            "buy": buy,
            "R": np.where(buy, R, np.nan),
            "pnl": np.where(buy, R, 0.0),
            "R_uncond": R,
        },
        index=rv_hat.index,
    )


def boot_diff(a: np.ndarray, b: np.ndarray, idx: np.ndarray) -> dict:
    d = a - b
    bm = d[idx].mean(1)
    return {
        "d_pnl": float(d.mean()),
        "d_ci_lo": float(np.quantile(bm, 0.025)),
        "d_ci_hi": float(np.quantile(bm, 0.975)),
        "p_boot_gt0": float((bm > 0).mean()),
    }


def summarize(
    name: str,
    period: str,
    s: pd.DataFrame,
    ref: pd.DataFrame,
    y: np.ndarray,
    idx: np.ndarray,
) -> dict:
    pnl = s["pnl"].to_numpy(float)
    Rb = s.loc[s["buy"], "R"].to_numpy(float)
    q = qlike(y, s["rv_hat"].to_numpy(float))
    out = {
        "variant": name,
        "bucket": VARIANTS[name][0],
        "period": period,
        "days": int(len(s)),
        "qlike": float(np.nanmean(q)),
        "buys": int(s["buy"].sum()),
        "pnl_per_day": float(pnl.mean()),
        "t_pnl": tstat(pnl),
        "mean_R_buy": float(Rb.mean()) if len(Rb) else float("nan"),
        "t_R_buy": tstat(Rb),
        "hit_buy": float((Rb > 0).mean()) if len(Rb) else float("nan"),
        "agree_vs_ref": float((s["buy"].to_numpy() == ref["buy"].to_numpy()).mean()),
        # selection: R at the ask on the days the rule skips (what buying them would have made)
        "mean_R_skip": float(s.loc[~s["buy"], "R_uncond"].mean()),
        "buy_minus_skip_R": float(Rb.mean() - s.loc[~s["buy"], "R_uncond"].mean())
        if len(Rb)
        else float("nan"),
        "sd_pnl": float(pnl.std(ddof=1)),
        # iid days for t = 2 if the true per-day mean were this sample's
        "days_for_t2": float((2.0 * pnl.std(ddof=1) / pnl.mean()) ** 2)
        if pnl.mean() > 0
        else float("nan"),
        "median_rv_hat_over_ref": float(np.median(s["rv_hat"] / ref["rv_hat"])),
        "corr_log_rv_hat_ref": float(
            np.corrcoef(np.log(s["rv_hat"]), np.log(ref["rv_hat"]))[0, 1]
        ),
    }
    out.update(boot_diff(pnl, ref["pnl"].to_numpy(float), idx))
    return out


def half(d: pd.Timestamp) -> str:
    return f"{d.year}H{1 if d.month <= 6 else 2}"


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--reuse-arms", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--work", default="")
    ap.add_argument("--inputs-only", action="store_true")
    ap.add_argument("--engine-only", action="store_true")
    a = ap.parse_args(argv)
    T0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    work = Path(a.work) if a.work else Path(tempfile.mkdtemp(prefix="p87_"))
    work.mkdir(parents=True, exist_ok=True)
    names = [n for n in RUN_ORDER if not a.only or n in a.only.split(",")]
    if "deck" not in names and not a.engine_only:
        names = ["deck", *names]
    if (
        any(VARIANTS[n][4] == "post" for n in names)
        and "post_live" not in names
        and not a.engine_only
    ):
        names.append("post_live")
    gate: dict = {}

    from live.close_signal import forecast as F
    from live.close_signal.state import StateStore
    from live.ibkr.calendar_guard import is_last_session_of_month

    deck = pd.read_parquet(DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).normalize()
    first_deck = deck.index.min()

    # ---- 1. inputs
    vend_core = pd.read_parquet(DATA / "core_stats.parquet")
    vend_core["endbartime"] = pd.to_datetime(vend_core["endbartime"])
    vend_vix = pd.read_parquet(DATA / "vix_and_voldemand.parquet")
    vend_vix["endbartime"] = pd.to_datetime(vend_vix["endbartime"])
    cboe_end = vend_vix.loc[vend_vix["vix"].notna(), "endbartime"].max()
    sessions = pd.DatetimeIndex(
        sorted(
            vend_core.loc[
                vend_core["sumret2"].notna()
                & (vend_core["endbartime"].dt.strftime("%H:%M") == "16:00"),
                "endbartime",
            ]
            .dt.normalize()
            .unique()
        )
    )
    # the live analog: sessions between the Yahoo switch and the first test day
    pf_all = pd.read_parquet(PANEL_FREE)
    pf_all["endbartime"] = pd.to_datetime(pf_all["endbartime"])
    ext_sessions = pd.DatetimeIndex(
        sorted(
            set(sessions)
            | set(
                pf_all.loc[
                    pf_all["sumret2"].notna()
                    & (pf_all["endbartime"].dt.strftime("%H:%M") == "16:00"),
                    "endbartime",
                ].dt.normalize()
            )
        )
    )
    n_lead = int(
        ((ext_sessions >= LIVE_YAHOO_START) & (ext_sessions < LIVE_TEST_START)).sum()
    )
    deg_start = sessions[int(sessions.searchsorted(first_deck)) - n_lead]
    log(
        f"vendor Cboe last print {cboe_end}; live: {n_lead} sessions from the Yahoo switch "
        f"{LIVE_YAHOO_START.date()} to the first test day {LIVE_TEST_START.date()} -> "
        f"research analog DEG_START {deg_start.date()} ({n_lead} sessions before {first_deck.date()})"
    )
    gate["deg_start"] = str(deg_start.date())
    gate["n_lead_sessions"] = n_lead

    daily = _read_daily()
    hslots = hourly_end_slots()
    vslots = vendor_slots(vend_vix)
    log(f"hourly bar-end stamps (cboe_gap_yahoo tags): {json.dumps(hslots)}")
    log(
        f"vendor print stamps (2021-2023 weekdays): { {c: (v[0], v[-1], len(v)) for c, v in vslots.items()} }"
    )

    # G2: the synthetic construction vs the REAL Yahoo snapshot on the vendor overlap
    lo, hi = OVERLAP_CBOE
    real = yahoo_form(
        _read_hourly(), daily, lo, hi + pd.Timedelta(hours=23, minutes=30)
    )
    syn = yahoo_form(
        synthetic_hourly(
            vend_vix, daily, hslots, lo, hi + pd.Timedelta(hours=23, minutes=30)
        ),
        daily,
        lo,
        hi + pd.Timedelta(hours=23, minutes=30),
    )
    es_stamps = set(vend_core.loc[vend_core["sumret2"].notna(), "endbartime"])
    rj = real.set_index("endbartime")
    sj = syn.set_index("endbartime").reindex(rj.index)
    on_rows = rj.index.isin(list(es_stamps))
    g2 = []
    for c in CBOE:
        for scope, m in (
            ("all_rows", on_rows),
            (
                "rth_10_16",
                on_rows
                & np.isin(
                    rj.index.strftime("%H:%M"),
                    [
                        f"{h:02d}:{mm:02d}"
                        for h in range(10, 17)
                        for mm in (0, 30)
                        if (h, mm) <= (16, 0)
                    ],
                ),
            ),
        ):
            x, y = sj.loc[m, c].to_numpy(float), rj.loc[m, c].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            le = np.abs(np.log(x[ok] / y[ok]))
            g2.append(
                {
                    "column": c,
                    "scope": scope,
                    "rows": int(m.sum()),
                    "both_finite": int(ok.sum()),
                    "nan_mismatch": int((np.isfinite(x) != np.isfinite(y)).sum()),
                    "frac_equal_1e-4": float(np.mean(le < 1e-4)),
                    "log_err_p95": float(np.quantile(le, 0.95)),
                    "source_tag_agree": float(
                        (sj.loc[m, f"{c}_source"] == rj.loc[m, f"{c}_source"]).mean()
                    ),
                }
            )
    g2 = pd.DataFrame(g2)
    g2.to_csv(OUT / "vix_form_gate.csv", index=False)
    log(
        "G2 synthetic-from-vendor Yahoo form vs the real Yahoo snapshot, overlap "
        f"{lo.date()} .. {hi.date()} (panel rows):\n" + g2.to_string(index=False)
    )
    gate["G2_min_frac_equal_rth"] = float(
        g2.loc[g2.scope == "rth_10_16", "frac_equal_1e-4"].min()
    )

    # the forms on the vendor span
    need_forms = {VARIANTS[n][2] for n in names if VARIANTS[n][2]}
    forms: dict[str, pd.DataFrame] = {}
    hi_v = cboe_end
    if need_forms & {"yahoo_live_analog", "yahoo_live_analog_masked"}:
        f = yahoo_form(
            synthetic_hourly(vend_vix, daily, hslots, deg_start, hi_v),
            daily,
            deg_start,
            hi_v,
        )
        forms["live_analog"] = f
    if need_forms & {"yahoo_full_history", "yahoo_full_history_pre"}:
        lo_full = vend_vix["endbartime"].min().normalize()
        forms["full"] = yahoo_form(
            synthetic_hourly(vend_vix, daily, hslots, lo_full, hi_v),
            daily,
            lo_full,
            hi_v,
        )

    # es factors
    fac, drop = es_shift_factors(vend_core, pf_all)
    fac.to_csv(OUT / "es_shift_factors.csv", index=False)
    log(
        "ES Databento / vendor on the overlap (median log ratio -> factor applied in T3/ALL):\n"
        + fac.to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )
    log(f"vendor-only (weekday, stamp) rows on the overlap, dropped in T3/ALL: {drop}")

    # the extended panel (post variants)
    ext_data = None
    if any(VARIANTS[n][1] == "ext" for n in names):
        st = work / "state"
        st.mkdir(exist_ok=True)
        pf = pf_all[
            (~pf_all["placeholder"].astype(bool)) & (pf_all["endbartime"] < PANEL_END)
        ]
        pf.to_parquet(st / "panel_free.parquet", index=False)
        ext_root = work / "ext_root"
        F.write_ext_data(REPO, StateStore(st), ext_root, FOMC_CSV)
        ext_data = ext_root / "data"
        info = json.loads((ext_data / "extension.json").read_text())
        log(f"extended panel: panel_free {len(pf_all)} rows -> {len(pf)}; {info}")
        gate["ext_info"] = info

    # ---- 2. variant data directories
    diag_rows = []
    roots: dict[str, Path] = {}
    for n in names:
        bucket, kind, cmod, emod, period = VARIANTS[n]
        root = work / "roots" / n
        d = root / "data"
        d.mkdir(parents=True, exist_ok=True)
        src_dir = DATA if kind == "vendor" else ext_data
        assert src_dir is not None
        for f in VENDOR_FILES:
            shutil.copy2(src_dir / f, d / f)
        vix = pd.read_parquet(d / "vix_and_voldemand.parquet")
        vix["endbartime"] = pd.to_datetime(vix["endbartime"])
        info = {"variant": n}
        if cmod in ("yahoo_live_analog", "yahoo_live_analog_masked"):
            vix, inf = apply_form(vix, forms["live_analog"], deg_start, hi_v)
            info.update(inf)
            if cmod.endswith("masked"):
                t = vix["endbartime"]
                vix = mask_to_vendor_pattern(
                    vix, ((t >= deg_start) & (t <= hi_v)).to_numpy(), vslots, daily
                )
        elif cmod == "yahoo_full_history":
            vix, inf = apply_form(
                vix, forms["full"], forms["full"]["endbartime"].min(), hi_v
            )
            info.update(inf)
        elif cmod == "yahoo_full_history_pre":
            vix, inf = apply_form(
                vix, forms["full"], forms["full"]["endbartime"].min(), hi_v
            )
            info.update(inf)
        elif cmod == "mask_post":
            t = vix["endbartime"]
            vix = mask_to_vendor_pattern(
                vix, (t >= LIVE_YAHOO_START).to_numpy(), vslots, daily
            )
        if cmod:
            vix.to_parquet(d / "vix_and_voldemand.parquet", index=False)
        if emod == "databento_shift":
            core = pd.read_parquet(d / "core_stats.parquet")
            core["endbartime"] = pd.to_datetime(core["endbartime"])
            core, inf = apply_es(core, fac, drop, first_deck)
            info.update(inf)
            core.to_parquet(d / "core_stats.parquet", index=False)
        # VIX-form diagnostics on the scoring window's RTH / all rows with ES prints
        core_t = pd.read_parquet(
            d / "core_stats.parquet", columns=["endbartime", "sumret2"]
        )
        rows_t = pd.DatetimeIndex(
            pd.to_datetime(core_t.loc[core_t["sumret2"].notna(), "endbartime"])
        )
        vi = vix.set_index("endbartime").reindex(rows_t)
        w = (rows_t >= (first_deck if period == "research" else LIVE_TEST_START)) & (
            rows_t <= (cboe_end if period == "research" else PANEL_END)
        )
        rth = w & np.isin(
            rows_t.strftime("%H:%M"),
            [
                f"{h:02d}:{mm:02d}"
                for h in range(10, 17)
                for mm in (0, 30)
                if (h, mm) <= (16, 0)
            ],
        )
        for c in CBOE:
            x = vi[c]
            per_day = x[rth].groupby(rows_t[rth].normalize()).nunique()
            s1530 = x[w & (rows_t.strftime("%H:%M") == "15:30")]
            s1500 = x.reindex(s1530.index - pd.Timedelta(minutes=30)).to_numpy()
            info[f"{c}_obs_frac_all_rows"] = float(x[w].notna().mean())
            info[f"{c}_distinct_per_13_rth"] = (
                float(per_day.mean()) if len(per_day) else float("nan")
            )
            info[f"{c}_1530_equals_1500"] = (
                float(np.mean(s1530.to_numpy() == s1500))
                if len(s1530)
                else float("nan")
            )
        diag_rows.append(info)
        roots[n] = root
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(OUT / "vix_form_diagnostics.csv", index=False)
    cols_show = ["variant"] + [
        c for c in diag.columns if c.startswith("vix_") or c in ("rows_dropped",)
    ]
    log(
        "VIX-family form on each variant's scoring window (rows with ES prints):\n"
        + diag[cols_show].to_string(index=False, float_format=lambda v: f"{v:.3f}")
    )
    if a.inputs_only:
        (OUT / "gate.json").write_text(json.dumps(gate, indent=1, default=str))
        log(f"--inputs-only: stopped before the engine; work dir {work}")
        return 0

    # ---- 3. engine runs (the input frames are released first: the workers hold ~1.5 GB each)
    del forms, real, syn, rj, sj, vend_core, vend_vix, pf_all
    import gc

    gc.collect()
    todo = [
        n
        for n in names
        if not (a.reuse_arms and all(p.exists() for p in arm_csvs(n).values()))
    ]
    for ev in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[ev] = "1"
    t1 = time.time()
    if todo:
        log(f"engine: {len(todo)} runs, {a.workers} at a time: {todo}")
        with ProcessPoolExecutor(max_workers=max(1, min(a.workers, len(todo)))) as ex:
            futs = {
                n: ex.submit(
                    run_variant,
                    n,
                    VARIANTS[n][0],
                    str(roots[n] / "data"),
                    str(OUT / "arms" / n),
                )
                for n in todo
            }
            for n, fu in futs.items():
                fu.result()
                log(f"  {n} done at {time.time() - t1:.0f}s")
    gate["engine_wall_s"] = round(time.time() - t1, 1)
    if a.engine_only:
        log(
            f"--engine-only: {todo} in {gate['engine_wall_s']}s; rerun with --reuse-arms to score"
        )
        return 0

    # ---- 4. tables + the deck's MZ map
    s85 = pd.read_parquet(S85 / "forecasts.parquet")
    s85 = s85[
        (s85["iset"] == "idealized")
        & (s85["w"] == "15:30")
        & (s85["scheme"] == "profile")
    ]
    s85 = s85.set_index(
        pd.DatetimeIndex(pd.to_datetime(s85["date"])).normalize()
    ).sort_index()
    post_days = s85.index
    fc: dict[str, pd.DataFrame] = {}
    tabs: dict[str, pd.DataFrame] = {}
    for n in names:
        per = VARIANTS[n][4]
        csvs = {b: p for b, p in arm_csvs(n).items()}
        tab = F.assemble_yhat_table(csvs, roots[n])
        tabs[n] = tab
        need = deck.index if per == "research" else post_days
        px = F.recalibrate(tab, need, REPO, work / "mz" / n, tag=f"p87_{n}")
        fc[n] = px.reindex(need)

    # ---- 5. gates
    ref = pd.read_parquet(RESEARCH_TABLE).sort_values("t").reset_index(drop=True)
    j = ref.merge(
        tabs["deck"], on="t", how="outer", suffixes=("_ref", ""), indicator=True
    )
    both = j[j["_merge"] == "both"]
    rel = (np.abs(fc["deck"]["rv_hat"] - deck["rv_hat"]) / deck["rv_hat"]).to_numpy(
        float
    )
    g1 = {
        "table_rows_unmatched": int((j["_merge"] != "both").sum()),
        "table_max_rel_yhat": float(
            (np.abs(both["yhat"] - both["yhat_ref"]) / np.abs(both["yhat_ref"])).max()
        ),
        "table_max_abs_baseline": float(
            np.abs(both["baseline"] - both["baseline_ref"]).max()
        ),
        "table_max_abs_rv_raw": float(
            np.abs(both["rv_raw"] - both["rv_raw_ref"]).max()
        ),
        "deck_days_matched": int(np.isfinite(rel).sum()),
        "deck_max_rel_rv_hat": float(np.nanmax(rel)),
    }
    g1["passed"] = bool(
        g1["table_rows_unmatched"] == 0
        and g1["table_max_rel_yhat"] <= YHAT_REL_TOL
        and g1["table_max_abs_baseline"] == 0.0
        and g1["table_max_abs_rv_raw"] == 0.0
        and g1["deck_days_matched"] == len(deck)
        and g1["deck_max_rel_rv_hat"] <= RV_HAT_REL_TOL
    )
    gate["G1_deck"] = g1
    log("G1 deck variant vs the research table and the deck: " + json.dumps(g1))
    if not g1["passed"]:
        (OUT / "gate.json").write_text(json.dumps(gate, indent=1, default=str))
        log("G1 FAILED -- no variant number is reported")
        return 2

    # ---- 6. research scoring
    p79 = _p79()
    book = p79.spxw_book(deck.index)
    book.index = pd.DatetimeIndex(book.index).normalize()
    book = book.reindex(deck.index)
    gate["book_days"] = int(book["spx_ask"].notna().sum())
    gate["book_pair_equals_deck"] = int(
        ((book["spx_kc"] == deck["K_c"]) & (book["spx_kp"] == deck["K_p"])).sum()
    )
    payoff = np.maximum(deck["S_close"] - book["spx_kc"], 0.0) + np.maximum(
        book["spx_kp"] - deck["S_close"], 0.0
    )
    me = pd.Series(
        [is_last_session_of_month(d.date()) for d in deck.index], index=deck.index
    )
    nme = deck.index[(~me) & book["spx_ask"].notna()]
    y_res = fc["deck"]["rv_raw"].reindex(nme).to_numpy(float)  # the vendor 16:00 bar
    sc: dict[str, pd.DataFrame] = {}
    for n in names:
        if VARIANTS[n][4] != "research":
            continue
        s = score(
            fc[n]["rv_hat"].reindex(deck.index),
            book["spx_S"],
            book["spx_kc"],
            book["spx_kp"],
            book["spx_ask"],
            payoff,
        )
        sc[n] = s
    rng = np.random.default_rng(SEED)
    p80 = _p80()
    idx_res = p80.block_boot_idx(len(nme), BLOCK, N_BOOT, rng)
    rows = []
    for n, s in sc.items():
        rows.append(
            summarize(
                n,
                "2020-01..2024-04 non-month-end",
                s.loc[nme],
                sc["deck"].loc[nme],
                y_res,
                idx_res,
            )
        )
    # the study-80 cross-check: the deck variant's card == study 80's h = 1
    s80 = pd.read_csv(PROPOSALS / "80" / "summary_by_horizon.csv")
    s80 = s80[(s80["h"].astype(str) == "1") & (s80["scope"] == "non_month_end")].iloc[0]
    gate["deck_vs_study80"] = {
        "buys": int(sc["deck"].loc[nme, "buy"].sum()),
        "study80_buys": int(s80["traded_ask"]),
        "pnl_per_day": float(sc["deck"].loc[nme, "pnl"].mean()),
        "study80_pnl_per_day": float(s80["mean_pnl_per_day_ask"]),
    }

    # ---- 7. post scoring (study 85's SPX books)
    bk = pd.read_parquet(S85 / "books.parquet")
    bk = bk[bk["venue"] == "SPX"].copy()
    bk.index = pd.DatetimeIndex(pd.to_datetime(bk["date"])).normalize()
    bk = bk.sort_index()
    sp: dict[str, pd.DataFrame] = {}
    y_post = None
    if "post_live" in fc:
        relp = (
            np.abs(fc["post_live"]["rv_hat"] - s85["rv_hat"]) / s85["rv_hat"]
        ).to_numpy(float)
        g3 = {
            "days": int(np.isfinite(relp).sum()),
            "max_rel_rv_hat_vs_study85": float(np.nanmax(relp)),
        }
        for n in names:
            if VARIANTS[n][4] != "post":
                continue
            sp[n] = score(
                fc[n]["rv_hat"].reindex(bk.index),
                bk["S_1530"],
                bk["kc_1530"],
                bk["kp_1530"],
                bk["ask_1530"],
                bk["payoff_1530"],
            )
        pd85 = pd.read_parquet(S85 / "per_day_pnl.parquet")
        pd85 = pd85[
            (pd85.venue == "SPX")
            & (pd85.iset == "idealized")
            & (pd85.w == "15:30")
            & (pd85.variant == "i")
            & (pd85.scheme == "profile")
        ]
        pd85.index = pd.DatetimeIndex(pd.to_datetime(pd85["date"])).normalize()
        pd85 = pd85.reindex(bk.index)
        g3["spx_days"] = int(len(bk))
        g3["buys"] = int(sp["post_live"]["buy"].sum())
        g3["study85_buys"] = int(pd85["buy"].sum())
        g3["buy_agree_study85"] = int(
            (sp["post_live"]["buy"].to_numpy() == pd85["buy"].to_numpy(bool)).sum()
        )
        g3["pnl_per_day"] = float(sp["post_live"]["pnl"].mean())
        g3["study85_pnl_per_day"] = float(pd85["pnl"].mean())
        g3["passed"] = bool(
            g3["max_rel_rv_hat_vs_study85"] <= POST_REL_TOL
            and g3["buy_agree_study85"] == len(bk)
        )
        # the pre-set 1e-9 bar (study 85's in-run substitution tolerance) failed at 1.3e-8 on the
        # first run (study 85 froze commit 85c2192; atm_straddle_lib / forecast.py moved since);
        # reported beside the deck gate's own 1e-7 bar, not replaced by it
        g3["passed_at_deck_tol_1e-7"] = bool(
            g3["max_rel_rv_hat_vs_study85"] <= RV_HAT_REL_TOL
            and g3["buy_agree_study85"] == len(bk)
        )
        gate["G3_post_live"] = g3
        log("G3 post_live vs study 85's 15:30 card: " + json.dumps(g3))
        y_post = fc["post_live"]["rv_raw"].reindex(bk.index).to_numpy(float)
        idx_post = p80.block_boot_idx(
            len(bk), BLOCK, N_BOOT, np.random.default_rng(SEED + 1)
        )
        for n, s in sp.items():
            rows.append(
                summarize(
                    n,
                    "2024-05..2025-12 SPX non-month-end",
                    s,
                    sp["post_live"],
                    y_post,
                    idx_post,
                )
            )
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "summary.csv", index=False)

    # per-day table
    pdl = []
    for n, s in sc.items():
        x = s.copy()
        x["variant"], x["month_end"] = n, me.reindex(x.index).to_numpy()
        x["rv_raw_vendor"] = fc["deck"]["rv_raw"].reindex(x.index).to_numpy()
        x["S"], x["Kc"], x["Kp"], x["ask"] = (
            book["spx_S"],
            book["spx_kc"],
            book["spx_kp"],
            book["spx_ask"],
        )
        pdl.append(x)
    for n, s in sp.items():
        x = s.copy()
        x["variant"], x["month_end"] = n, False
        x["rv_raw_vendor"] = fc["post_live"]["rv_raw"].reindex(x.index).to_numpy()
        x["S"], x["Kc"], x["Kp"], x["ask"] = (
            bk["S_1530"],
            bk["kc_1530"],
            bk["kp_1530"],
            bk["ask_1530"],
        )
        pdl.append(x)
    pdl = pd.concat(pdl)
    pdl.index.name = "date"
    pdl.reset_index().to_parquet(OUT / "per_day_pnl.parquet", index=False)
    pdl.reset_index().to_csv(OUT / "per_day_pnl.csv", index=False)
    fcat = pd.concat([f.assign(variant=n) for n, f in fc.items()])
    fcat.index.name = "date"
    fcat.reset_index().to_parquet(OUT / "forecasts.parquet", index=False)

    # ---- 8. T4 period table: unconditional long and the card by half-year on SPX
    reg = []
    res_days = pd.DataFrame(index=nme)
    res_days["R_uncond"] = sc["deck"].loc[nme, "R_uncond"]
    res_days["ask"], res_days["S"] = book.loc[nme, "spx_ask"], book.loc[nme, "spx_S"]
    res_days["payoff"], res_days["move"] = (
        payoff.loc[nme],
        (deck.loc[nme, "S_close"] - book.loc[nme, "spx_S"]).abs(),
    )
    res_days["rv"] = fc["deck"]["rv_raw"].reindex(nme)
    cards_res = {n: sc[n].loc[nme, "pnl"] for n in ("deck", "T1", "ALL") if n in sc}
    post_df = pd.DataFrame(index=bk.index)
    post_df["R_uncond"] = bk["payoff_1530"] / bk["ask_1530"] - 1.0
    post_df["ask"], post_df["S"], post_df["payoff"] = (
        bk["ask_1530"],
        bk["S_1530"],
        bk["payoff_1530"],
    )
    post_df["move"] = (bk["S_close"] - bk["S_1530"]).abs()
    post_df["rv"] = (
        fc["post_live"]["rv_raw"].reindex(bk.index) if "post_live" in fc else np.nan
    )
    cards_post = {n: sp[n]["pnl"] for n in sp}
    for label, dfp, cards in (
        ("research deck days", res_days, cards_res),
        ("live SPX books", post_df, cards_post),
    ):
        hk = pd.Series([half(d) for d in dfp.index], index=dfp.index)
        for h in sorted(hk.unique()):
            m = (hk == h).to_numpy()
            x = dfp[m]
            r = {
                "half": h,
                "source": label,
                "days": int(m.sum()),
                "uncond_mean_R": float(x["R_uncond"].mean()),
                "uncond_t": tstat(x["R_uncond"].to_numpy(float)),
                "payoff_over_ask": float(x["payoff"].sum() / x["ask"].sum()),
                "ask_over_S_bp": float((x["ask"] / x["S"]).mean() * 1e4),
                "abs_move_over_S_bp": float((x["move"] / x["S"]).mean() * 1e4),
                "rv16_vol_bp": float(np.sqrt(x["rv"]).mean() * 1e4),
            }
            for n, pnl in cards.items():
                r[f"card_{n}_buys"] = int(
                    (sc[n] if n in sc else sp[n]).loc[x.index, "buy"].sum()
                )
                r[f"card_{n}_pnl_per_day"] = float(pnl[m].mean())
                r[f"card_{n}_t"] = tstat(pnl[m].to_numpy(float))
            reg.append(r)
    reg = pd.DataFrame(reg)
    reg.to_csv(OUT / "regime_halfyear.csv", index=False)
    # whole-period unconditional + the period difference (independent block bootstraps)
    pb = []
    rngp = np.random.default_rng(SEED + 2)
    ib_r = p80.block_boot_idx(len(nme), BLOCK, N_BOOT, rngp)
    ib_p = p80.block_boot_idx(len(bk), BLOCK, N_BOOT, rngp)
    for lab, ra, pa in (
        (
            "uncond long",
            res_days["R_uncond"].to_numpy(float),
            post_df["R_uncond"].to_numpy(float),
        ),
        (
            "card deck(research) vs post_live",
            sc["deck"].loc[nme, "pnl"].to_numpy(float),
            sp["post_live"]["pnl"].to_numpy(float) if sp else None,
        ),
        (
            "card T1(research) vs post_live",
            sc["T1"].loc[nme, "pnl"].to_numpy(float) if "T1" in sc else None,
            sp["post_live"]["pnl"].to_numpy(float) if sp else None,
        ),
        (
            "card ALL(research) vs post_live",
            sc["ALL"].loc[nme, "pnl"].to_numpy(float) if "ALL" in sc else None,
            sp["post_live"]["pnl"].to_numpy(float) if sp else None,
        ),
    ):
        if ra is None or pa is None:
            continue
        br, bp_ = ra[ib_r].mean(1), pa[ib_p].mean(1)
        pb.append(
            {
                "comparison": lab,
                "research_mean": float(ra.mean()),
                "research_ci_lo": float(np.quantile(br, 0.025)),
                "research_ci_hi": float(np.quantile(br, 0.975)),
                "post_mean": float(pa.mean()),
                "post_ci_lo": float(np.quantile(bp_, 0.025)),
                "post_ci_hi": float(np.quantile(bp_, 0.975)),
                "post_minus_research": float(pa.mean() - ra.mean()),
                "diff_ci_lo": float(np.quantile(bp_ - br, 0.025)),
                "diff_ci_hi": float(np.quantile(bp_ - br, 0.975)),
                "p_diff_lt0": float((bp_ - br < 0).mean()),
            }
        )
    pb = pd.DataFrame(pb)
    pb.to_csv(OUT / "period_bootstrap.csv", index=False)

    # ---- 9. report
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    ff = lambda v: f"{v:.4f}"  # noqa: E731
    log(
        f"\nBOOK (research): {gate['book_days']} of {len(deck)} deck days, pair == deck on {gate['book_pair_equals_deck']}; "
        f"non-month-end scored {len(nme)}"
    )
    log("deck variant vs study 80 h=1: " + json.dumps(gate["deck_vs_study80"]))
    show = [
        "variant",
        "bucket",
        "period",
        "days",
        "qlike",
        "buys",
        "pnl_per_day",
        "t_pnl",
        "mean_R_buy",
        "t_R_buy",
        "agree_vs_ref",
        "median_rv_hat_over_ref",
        "mean_R_skip",
        "buy_minus_skip_R",
        "days_for_t2",
        "d_pnl",
        "d_ci_lo",
        "d_ci_hi",
        "p_boot_gt0",
    ]
    log(
        "\nVARIANTS (research: ref = deck; post: ref = post_live; per-day P&L at the SPXW 15:30 ask):"
    )
    log(summ[show].to_string(index=False, float_format=ff))
    log(
        "\nT4 PERIOD: unconditional long strangle at the 15:30 ask and the card, by half-year (SPX, non-month-end):"
    )
    log(reg.to_string(index=False, float_format=ff))
    log(
        "\nPERIOD DIFFERENCE (independent 20-session circular block bootstraps, 10,000 draws):"
    )
    log(pb.to_string(index=False, float_format=ff))
    gate["total_wall_s"] = round(time.time() - T0, 1)
    (OUT / "gate.json").write_text(json.dumps(gate, indent=1, default=str))
    (OUT / "summary.txt").write_text("\n".join(_LOG) + "\n", encoding="utf-8")
    log(f"\nwall clock {gate['total_wall_s']}s; work dir {work}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
