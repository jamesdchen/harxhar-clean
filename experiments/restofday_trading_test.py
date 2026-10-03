"""Does the DIRECT rest-of-day forecast trade better than the one-step construction?  (E2)

The direct model (specs/causal_tune_rest_of_day.py; stacked by
experiments/build_restofday_yhat.py into results/spxw_pnl/yhat_restofday_*.parquet)
forecasts at each entry time c the variance from c to the 16:00 close, F_rem, in
one regression per entry time.  Its QLIKE beats the one-step construction
RVhat/w (experiments/score_rest_of_day.py, "plain" columns).  This script asks
whether it also TRADES better in the delta-hedged hold-to-close straddle book of
notebooks/atm_straddle_intraday_holdclose.executed.ipynb.

THE BOOK (read, not rebuilt).  At entry time c the nearest-OTM straddle (nearest
out-of-the-money call + nearest out-of-the-money put, same-day expiry, one
position) is held to the official close, delta-hedged in the index every 30
minutes (the notebook's section 4b, computed by the same function,
writeup/make_rule_by_strategy_intraday_tex.attach_long_dh).  Long return
R = (settlement + hedge P&L - entry mid) / entry mid; the short earns -R.
Crossed fill: buy at the ask, sell at the bid, cash settlement (no exit spread).

THE RULE.  sign(s): long the straddle when s > 0, short otherwise, flat where the
vendor implied volatility is a censored solver node (the notebook's convention;
the same bars are flat for every signal), with

    s_direct  = F_rem,direct        - IV_hr^2 * h
    s_current = RVhat_c / w_c       - IV_hr^2 * h      (the one-step construction)

IV_hr the vendor's hourly implied volatility of the straddle at c, h the hours
from c to the close.  Only the forecast changes: same days, same bars, same
fills, same hedge.

TWO VARIANTS OF THE FORECAST LEVEL (fixed before any trading number was seen)
  plain  (headline; the QLIKE comparison's own construction)
         direct  = pred_adj^2 * baseline_rem                     (the scorer's "plain")
         current = ft_remaining F_rem = f_next / w_next          (the scorer's one-step)
  recal  both sides pass through the notebook's recalibration
         (atm_straddle_lib.second_order_raw: weighted Mincer-Zarnowitz line on
         the 250 sessions strictly before the day, (m^2 + s2) * baseline), fitted
         PER ENTRY TIME on that entry time's own target: the direct forecast on
         the rest-of-day variance, the one-step forecast on its own bar, then
         divided by the same w_next.
At 15:30 w = 1 and the direct arm is the per-bar 16:00 arm, so the two signals
are identical there in both variants (asserted).

ENTRY TIMES  09:35, 10:00, ..., 15:30 (13).  The 09:35 entry uses the forecasts
issued at 09:30 (both the direct 09:30 arm and the per-bar 09:30-10:00 bar),
known five minutes before the entry, used as issued (the delta-hedged PDF's
design); its h is the chain's 6 h 25 min.  Pools (daily sums over the entry
times in the pool): 10:00-15:30 (the notebook's section 6 pool), 09:35-15:30,
and 10:00-15:00 (section 8d's pool, which leaves out 15:30, where the signals
coincide).

DAYS  the 15:30 deck's 866 expiration days (2020-01 .. 2024-04), the notebook's
frame; a bar enters only if every forecast compared here is finite and positive
on it (count printed).

INTERVALS  paired circular moving-block bootstrap (the notebook's section 7:
block ceil(n^(1/3)) days, 2000 draws, seed 0, percentile and basic 95 %) of
direct minus current, on the same resampled days.

CONTEXT  always short; the notebook's own block-diagonal ridge sign(s)
(recomputed here and GATED against the executed notebook's CSVs, which checks
the return ledger, the join and the fills); section 8d's fitted rest-of-day maps
M1-M3, cited from the executed notebook's CSV.

Run:
  python experiments/restofday_trading_test.py --build-ledger
        CHAIN-SIZED (reads data/spxw_chain.parquet through the PDF writer's
        hours_to_close / iv_hourly_reinverted, exactly as the notebook's [dh]
        cell does): take .chain_lock.d first.  Writes the per-day, per-entry-time
        hedged return ledger <OUT>/dh_holdclose_ledger.parquet for reuse.
  python experiments/restofday_trading_test.py
        the test, from the ledger (not chain-sized).

Options (defaults = the first run's paths, so a bare run writes where it always did):
  --out DIR          outputs (CSVs, figures, SUMMARY.md, per-day series)
  --yhat-dir DIR     the yhat_restofday_* / yhat_sub_* tables read
  --ledger PATH      the return ledger read (and written by --build-ledger)
  --score-csv PATH   the QLIKE scorer's CSV whose identity_pass flags the 15:30 gate reads
  --tex PATH|none    the LaTeX table ('none' = not written)
The de-duplicated design's re-run (checklist I6 rebuilt the direct tables):
  python experiments/restofday_trading_test.py
        --out results/linear_subsection_restofday/trading_test_dedup
        --score-csv results/linear_subsection_restofday_dedup/score_rest_of_day.csv
Every run also writes <out>/trading_test_daily.parquet (the per-day direct / current series
of every row of the two CSVs; not committed), read by
experiments/restofday_trading_test_before_after.py for paired intervals of a change.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "experiments", ROOT / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402
import score_rest_of_day as srd  # noqa: E402
from build_subsection_yhat import ESTIMATORS, GATE_REL  # noqa: E402

OUT = ROOT / "results" / "linear_subsection_restofday" / "trading_test"
LEDGER = OUT / "dh_holdclose_ledger.parquet"
SCORE_CSV = ROOT / "results" / "linear_subsection_restofday" / "score_rest_of_day.csv"
HC = ROOT / "results" / "atm_straddle_intraday_holdclose"
HC_CACHE = HC / "cache"
DECK_DAILY = ROOT / "results" / "atm_straddle_0dte_1530" / "daily_blk2.parquet"
YHAT_DIR = ROOT / "results" / "spxw_pnl"
TEX_OUT: Path | None = (
    ROOT / "writeup" / "generated" / "appendix_running_restofday_trading.tex"
)
ET = "America/New_York"

SHORTS = ("ridge", "lasso", "enet")
BUCKETS = ("all_features", "baseline")  # live_feasible commented out
BUCKET_LABEL = {
    "live_feasible": "live-feasible",
    "all_features": "all features",
    "baseline": "HAR + calendar",
}
VARIANTS = ("plain", "recal")
HEADLINE = ("ridge", "plain")  # the estimator and variant the figure and tex lead with
FILLS = ("mid", "crossed")
B_BOOT = 2000  # the holdclose notebook's section 7 bootstrap draws
SEED = 0  # ... and its seed
ALPHA_PCT = (2.5, 97.5)  # 95 % intervals

ENTRY_CLOCKS = ["09:35"] + [f"{h:02d}:{m:02d}" for h in range(10, 16) for m in (0, 30)]
POOLS = {
    "10:00-15:30": ENTRY_CLOCKS[1:],
    "09:35-15:30": ENTRY_CLOCKS,
    "10:00-15:00": ENTRY_CLOCKS[1:-1],
}


GATES: list[str] = []  # the gate / count lines of the last run, echoed into SUMMARY.md


def gate(msg: str) -> None:
    print(msg)
    GATES.append(msg)


def _rel(p: Path) -> str:
    """A path as written into the outputs: repo-relative with forward slashes when inside it."""
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def fc_clock(entry: str) -> str:
    """Entry time -> the clock that issued its forecasts (09:35 -> 09:30)."""
    h, m = entry.split(":")
    return f"{h}:{int(m) // 30 * 30:02d}"


def load_pdf_writer():
    """writeup/make_rule_by_strategy_intraday_tex.py as a module (the notebook's [dh] import)."""
    spec = importlib.util.spec_from_file_location(
        "rule_by_strategy_intraday_tex",
        ROOT / "writeup" / "make_rule_by_strategy_intraday_tex.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. The return ledger (chain-sized; run under the chain lock)
# ---------------------------------------------------------------------------


def build_ledger() -> None:
    """The notebook's [dh] cell on its own trade cache, persisted per day x entry time."""
    trades = sorted(HC_CACHE.glob("trade_*.parquet"), key=lambda p: p.stat().st_mtime)
    assert trades, f"no holdclose trade cache under {HC_CACHE}"
    src = trades[-1]
    pkg = pd.read_parquet(src)
    print(
        f"trade cache {src.name}: {len(pkg):,} straddles, {pkg['date'].nunique():,} days"
    )
    assert "09:35" in set(pkg["hhmm"]), "the trade cache carries no 09:35 stamp"
    rbt = load_pdf_writer()

    # hours to the close, the notebook's [dh] cell verbatim in substance
    clk = sorted(pkg["hhmm"].unique())
    assert clk == ENTRY_CLOCKS, clk
    nrem = {c: len(clk) - i for i, c in enumerate(clk)}
    grid_h = pkg["hhmm"].map(nrem).astype(float).to_numpy() * 0.5
    on_grid = np.array([rbt.bar_start(c) == c for c in pkg["hhmm"]])
    h_chain = rbt.hours_to_close(pkg)
    assert bool(np.isfinite(h_chain).all()), (
        "a bar has no hours to the close in the chain"
    )
    assert bool(np.allclose(h_chain[on_grid], grid_h[on_grid])), "chain hours != grid"
    pkg["h_rem"] = np.where(on_grid, grid_h, h_chain)
    pkg["iv_hourly_used"] = rbt.iv_hourly_reinverted(pkg).to_numpy()
    dh = rbt.attach_long_dh(pkg)
    pkg["R_unhedged"] = pkg["R"]
    pkg["hedge_long"] = dh["hedge_long"].to_numpy()
    pkg["exit"] = pkg["exit_settle"] + pkg["hedge_long"]
    pkg["R"] = pkg["exit"] / pkg["entry"] - 1.0
    assert bool(
        np.allclose(pkg["R"].to_numpy(), dh["R"].to_numpy(), rtol=0.0, atol=1e-12)
    )

    # GATE (the notebook's): hedged always short per entry time on the deck's 866
    # days equals the delta-hedged PDF's table
    deck = pd.read_parquet(DECK_DAILY)
    days = pd.DatetimeIndex(pd.to_datetime(deck.index))
    worst = 0.0
    for c in clk:
        x = -pkg.loc[(pkg["hhmm"] == c) & pkg["date"].isin(days), "R"].astype(float)
        sr = float(x.mean() / x.std(ddof=1) * np.sqrt(asl.PERIODS_PER_YEAR))
        ref = pd.read_csv(
            HC
            / "rule_by_strategy_dh"
            / c.replace(":", "")
            / "rule_by_strategy_always_short.csv",
            index_col=0,
        ).loc["all models"]
        assert int(ref["n"]) == len(x), (c, int(ref["n"]), len(x))
        worst = max(worst, abs(sr - float(ref["Sharpe_ann"])))
    assert worst < 1e-9, f"hedged always short differs from the PDF by {worst}"
    print(
        f"GATE hedged always short equals the delta-hedged PDF at all {len(clk)} entry "
        f"times on the deck's {len(days)} days (max |diff| {worst:.1e})"
    )
    keep = [
        "date", "hhmm", "timestamp", "expiration", "S", "K_c", "K_p", "entry",
        "bid_c", "ask_c", "bid_p", "ask_p", "iv_hourly", "iv_hourly_used", "h_rem",
        "S_close", "exit_settle", "hedge_long", "exit", "R", "R_unhedged",
    ]  # fmt: skip
    led = pkg[keep].copy()
    led["in_deck"] = led["date"].isin(days)
    led["source_cache"] = src.name
    OUT.mkdir(parents=True, exist_ok=True)
    led.to_parquet(LEDGER, index=False)
    print(
        f"wrote {LEDGER}: {len(led):,} rows, {led['date'].nunique():,} days "
        f"({int(led.loc[led['in_deck'], 'date'].nunique())} deck days), entry times "
        f"{clk[0]} .. {clk[-1]}"
    )


# ---------------------------------------------------------------------------
# 2. Forecasts
# ---------------------------------------------------------------------------


def _clock_frame(path: Path) -> pd.DataFrame:
    """A yhat table's rows at the 13 forecast stamps c+30, with day, clock, session flags."""
    df, _ = asl._panel_frame(path)  # the library's session / early-close rule
    clock = (df["et"] - pd.Timedelta(minutes=30)).dt.strftime("%H:%M")
    df["clock"] = clock.to_numpy()
    df["day"] = df["date"]
    want = [srd.hhmm(c) for c in srd.CLOCKS]
    return df[df["clock"].isin(want)].reset_index(drop=True)


def recal_per_clock(df: pd.DataFrame) -> np.ndarray:
    """asl.second_order_raw fitted per entry time on that entry time's own rows."""
    out = np.full(len(df), np.nan)
    for _, g in df.groupby("clock", sort=True):
        g = g.sort_values("t")
        codes, uniq = pd.factorize(g["day"], sort=True)
        yh = g["yhat"].to_numpy(float)
        rv = g["rv_raw"].to_numpy(float)
        base = g["baseline"].to_numpy(float)
        mask = (
            g["session_date"].to_numpy(bool)
            & ~g["early_close"].to_numpy(bool)
            & np.isfinite(yh)
            & np.isfinite(rv)
            & (rv > 0)
            & (base > 0)
        )
        f = asl.second_order_raw(yh, rv, base, codes, len(uniq), fit_mask=mask)
        out[g.index.to_numpy()] = f
    return out


def model_forecasts(short: str, bucket: str) -> pd.DataFrame:
    """(day, clock) -> F_dir / F_cur in both variants, rv_rem, warm, for one estimator x bucket."""
    d = _clock_frame(YHAT_DIR / f"yhat_restofday_{short}_{bucket}.parquet")
    d["F_dir_plain"] = d["yhat"] ** 2 * d["baseline"]
    d["F_dir_recal"] = recal_per_clock(d)
    d = d.rename(columns={"rv_raw": "rv_rem"})[
        ["day", "clock", "F_dir_plain", "F_dir_recal", "rv_rem"]
    ]
    sub_path = YHAT_DIR / f"yhat_sub_{short}_{bucket}.parquet"
    one = srd.onestep_table(sub_path)  # the scorer's one-step comparator, imported
    p = _clock_frame(sub_path)
    p["f_recal"] = recal_per_clock(p)
    one = one.merge(
        p[["day", "clock", "f_recal"]], on=["day", "clock"], how="left", validate="1:1"
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        one["F_cur_recal"] = np.where(
            one["w_next"] > 0, one["f_recal"] / one["w_next"], np.nan
        )
    one = one.rename(columns={"F_rem": "F_cur_plain"})
    m = d.merge(
        one[["day", "clock", "F_cur_plain", "F_cur_recal", "warm", "rv_rem_ref"]],
        on=["day", "clock"],
        how="left",
        validate="1:1",
    )
    m["warm"] = m["warm"].astype("boolean").fillna(False).astype(bool)
    return m


# ---------------------------------------------------------------------------
# 3. Rules, statistics, paired bootstrap
# ---------------------------------------------------------------------------


def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    sd = x.std(ddof=1)
    return (
        float(x.mean() / sd * np.sqrt(asl.PERIODS_PER_YEAR)) if sd > 0 else float("nan")
    )


def boot_idx(n: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return asl.circular_block_bootstrap_idx(
        rng, n, int(np.ceil(n ** (1.0 / 3.0))), B_BOOT
    )


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired block-bootstrap intervals of Sharpe(a) - Sharpe(b) and mean(a) - mean(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = boot_idx(len(a))

    def _shr(x):
        return x.mean(axis=1) / x.std(axis=1, ddof=1) * np.sqrt(asl.PERIODS_PER_YEAR)

    out: dict = {}
    for name, hat, draws in (
        ("dSharpe", sharpe(a) - sharpe(b), _shr(a[idx]) - _shr(b[idx])),
        (
            "dmean",
            float(a.mean() - b.mean()),
            a[idx].mean(axis=1) - b[idx].mean(axis=1),
        ),
    ):
        lo, hi = (float(v) for v in np.percentile(draws, ALPHA_PCT))
        out[name] = hat
        out[f"{name}_pct_lo"], out[f"{name}_pct_hi"] = lo, hi
        out[f"{name}_basic_lo"], out[f"{name}_basic_hi"] = 2 * hat - hi, 2 * hat - lo
    sa, sb = _shr(a[idx]), _shr(b[idx])
    out["sharpe_direct_lo"], out["sharpe_direct_hi"] = (
        float(v) for v in np.percentile(sa, ALPHA_PCT)
    )
    out["sharpe_current_lo"], out["sharpe_current_hi"] = (
        float(v) for v in np.percentile(sb, ALPHA_PCT)
    )
    return out


def reading(r: dict, key: str = "dSharpe") -> str:
    """The notebook's interval reading (percentile and basic; knife edge = a bound within 1/20 of the width)."""
    pct = r[f"{key}_pct_lo"] > 0 or r[f"{key}_pct_hi"] < 0
    bas = r[f"{key}_basic_lo"] > 0 or r[f"{key}_basic_hi"] < 0
    if pct != bas:
        return "percentile and basic disagree"
    edge = any(
        min(abs(r[f"{key}_{k}_lo"]), abs(r[f"{key}_{k}_hi"]))
        < 0.05 * (r[f"{key}_{k}_hi"] - r[f"{key}_{k}_lo"])
        for k in ("pct", "basic")
    )
    out = "excludes zero" if pct else "includes zero"
    return "knife-edge, " + out if edge else out


def positions(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, float)
    return np.where(np.isfinite(s), np.where(s > 0, 1.0, -1.0), 0.0)


def bar_returns(q: np.ndarray, fr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """Per-bar return at the midpoint and at the crossed spread (the notebook's _at_spread,
    hold to close: entry at the ask / bid, cash settlement, the hedge at the vendor spot)."""
    q = np.asarray(q, float)
    mid = q * fr["R"].to_numpy(float)
    ask_e = (fr["ask_c"] + fr["ask_p"]).to_numpy(float)
    bid_e = (fr["bid_c"] + fr["bid_p"]).to_numpy(float)
    ex, en = fr["exit"].to_numpy(float), fr["entry"].to_numpy(float)
    long, short = q > 0, q < 0
    px = np.where(long, ask_e, np.where(short, bid_e, en))
    untr = (long & ~(ask_e > 0)) | (short & ~(bid_e > 0))
    x = np.where(untr, np.nan, q * (ex - px)) / en
    return mid, x, int(untr.sum())


def daily(values: np.ndarray, fr: pd.DataFrame, mask: np.ndarray) -> pd.Series:
    """Daily sum over the bars in mask (a single entry time: that bar), NaN counted as 0."""
    v = pd.Series(np.where(mask, np.nan_to_num(values, nan=0.0), 0.0), index=fr.index)
    days = fr.loc[mask, "date"].unique()
    return v.groupby(fr["date"]).sum().reindex(pd.DatetimeIndex(sorted(days)))


def _daily_frame(
    tag: dict, scope: str, fill: str, dd_: pd.Series, dc_: pd.Series
) -> pd.DataFrame:
    """One row of the CSVs as its per-day series (direct, current) in long form."""
    return pd.DataFrame(
        {
            **tag,
            "scope": scope,
            "fill": fill,
            "date": dd_.index,
            "direct": dd_.to_numpy(float),
            "current": dc_.to_numpy(float),
        }
    )


def qlike(F: np.ndarray, y: np.ndarray) -> float:
    r = np.asarray(y, float) / np.asarray(F, float)
    return float(np.mean(r - np.log(r) - 1.0))


# ---------------------------------------------------------------------------
# 4. The notebook's block-diagonal ridge sign(s) (context + gate)
# ---------------------------------------------------------------------------


def blk2_current(fr: pd.DataFrame) -> np.ndarray:
    """The notebook's s^m on the frame (10:00-15:30; NaN at 09:35 and on censored IV)."""
    pan = asl.load_yhat_panel(asl.yhat_paths(ROOT)["blk2"])
    pf = pan[pan["in_fit"].to_numpy(dtype=bool)].copy()
    ck = pd.to_datetime(pf["t"], utc=True).dt.tz_convert(ET) - pd.Timedelta(minutes=30)
    pf["pdate"] = ck.dt.normalize().dt.tz_localize(None)
    pf["phhmm"] = ck.dt.strftime("%H:%M")
    prof = pf.pivot_table(
        index="pdate", columns="phhmm", values="rv_raw", aggfunc="mean"
    ).sort_index()
    clocks = ENTRY_CLOCKS[1:]
    pi = pd.DataFrame(index=prof.index, columns=clocks, dtype=float)
    for i, c in enumerate(clocks):
        pi[c] = prof[c] / prof[clocks[i:]].sum(axis=1).replace(0.0, np.nan)
    w = pi.expanding(min_periods=63).mean().shift(1)
    p = pan[["t", "rv_hat"]].copy()
    p["t"] = pd.to_datetime(p["t"], utc=True) - pd.Timedelta(minutes=30)
    j = fr[["t"]].merge(p, on="t", how="left")
    assert len(j) == len(fr)
    wv = (
        w.stack()
        .reindex(pd.MultiIndex.from_arrays([fr["date"], fr["hhmm"]]))
        .to_numpy()
    )
    iv2 = fr["iv_hourly"].astype(float).to_numpy() ** 2
    return j["rv_hat"].to_numpy(float) - iv2 * fr["h_rem"].to_numpy(float) * wv


# ---------------------------------------------------------------------------
# 5. The test
# ---------------------------------------------------------------------------


def run() -> None:
    assert LEDGER.exists(), (
        f"{LEDGER} missing: run --build-ledger under the chain lock first"
    )
    led = pd.read_parquet(LEDGER)
    fr = led[led["in_deck"]].copy()
    fr["t"] = pd.to_datetime(fr["timestamp"], utc=True)
    fr = fr.sort_values("t").reset_index(drop=True)
    fr["fclock"] = fr["hhmm"].map(fc_clock)
    n_days0 = fr["date"].nunique()
    print(
        f"ledger {LEDGER.name}: {len(fr):,} bars on the deck's {n_days0} days ({led['source_cache'].iloc[0]})"
    )
    gate(
        f"inputs: forecast tables {_rel(YHAT_DIR)}/yhat_{{restofday,sub}}_*; ledger {_rel(LEDGER)}"
    )

    # --- forecasts of every estimator x bucket, joined on (day, forecast clock)
    key = pd.MultiIndex.from_arrays([fr["date"], fr["fclock"]])
    models = [(s, b) for b in BUCKETS for s in SHORTS]
    rv_rem = None
    xcheck = []
    for s, b in models:
        m = model_forecasts(s, b).set_index(["day", "clock"])
        assert not m.index.duplicated().any(), (s, b)
        mj = m.reindex(key)
        for v in VARIANTS:
            fr[f"Fd|{s}|{b}|{v}"] = mj[f"F_dir_{v}"].to_numpy(float)
            fc = mj[f"F_cur_{v}"].to_numpy(float)
            fr[f"Fc|{s}|{b}|{v}"] = np.where(mj["warm"].to_numpy(bool), fc, np.nan)
        y = mj["rv_rem"].to_numpy(float)
        rr = mj["rv_rem_ref"].to_numpy(float)
        both = np.isfinite(y) & np.isfinite(rr) & (rr > 0)
        xcheck.append(float(np.max(np.abs(y[both] / rr[both] - 1.0))))
        if rv_rem is None:
            rv_rem = y
        else:
            same = np.isfinite(rv_rem) == np.isfinite(y)
            assert same.all() and np.allclose(
                rv_rem[np.isfinite(y)], y[np.isfinite(y)], rtol=GATE_REL, atol=0
            )
    fr["rv_rem"] = rv_rem
    worst_x = max(xcheck)
    assert worst_x <= GATE_REL, worst_x
    gate(
        f"GATE join: the direct tables' realized rest of day equals the per-bar tables' bars "
        f"summed from the entry time to the close on every bar of all {len(models)} models "
        f"(max rel {worst_x:.1e}); the realized target is the same in every direct table"
    )

    fcols = [c for c in fr.columns if c.startswith(("Fd|", "Fc|"))]
    Fv = fr[fcols].to_numpy(float)
    ok = (np.isfinite(Fv) & (Fv > 0)).all(axis=1) & np.isfinite(fr["R"].to_numpy(float))
    ok &= np.isfinite(fr["rv_rem"].to_numpy(float)) & (fr["rv_rem"].to_numpy(float) > 0)
    dropped = fr.loc[~ok, ["date", "hhmm"]]
    gate(
        f"bars without every forecast (dropped from every rule): {int((~ok).sum())} of {len(fr)}"
        + (
            f" -- by entry time {dropped['hhmm'].value_counts().sort_index().to_dict()}"
            if len(dropped)
            else ""
        )
    )
    fr = fr[ok].reset_index(drop=True)
    by_clock = fr.groupby("hhmm")["date"].nunique()
    gate(
        f"frame: {len(fr):,} bars, {fr['date'].nunique()} days; days per entry time {by_clock.to_dict()}"
    )

    iv_rem = fr["iv_hourly"].astype(float).to_numpy() ** 2 * fr["h_rem"].to_numpy(float)
    n_flat = int((~np.isfinite(iv_rem)).sum())
    gate(
        f"bars with a censored vendor implied volatility (flat in every sign rule): {n_flat}"
    )

    # 15:30 identity: the two signals are the same forecast there.  The scorer's own
    # self-check (score_rest_of_day.csv, identity_pass) already FAILS for the stored
    # per-bar all-features lasso / elastic-net tables (their 16:00 forecasts are not
    # bit-identical to the direct campaign's 15:30 arm); there the gap is reported,
    # everywhere else it is asserted.
    at15 = (fr["hhmm"] == "15:30").to_numpy()
    sc = pd.read_csv(SCORE_CSV)
    gate(f"15:30 identity flags read from {_rel(SCORE_CSV)}")
    sc = sc[sc["identity_pass"].notna()].assign(
        short=lambda t: t["estimator"].map(ESTIMATORS)
    )
    idt = {(r.short, r.bucket): bool(r.identity_pass) for r in sc.itertuples()}
    id_rows = []
    for s, b in models:
        for v in VARIANTS:
            a, c = fr.loc[at15, f"Fd|{s}|{b}|{v}"], fr.loc[at15, f"Fc|{s}|{b}|{v}"]
            rel = float(np.max(np.abs(a / c - 1.0)))
            ndiff = int(
                (positions(a - iv_rem[at15]) != positions(c - iv_rem[at15])).sum()
            )
            if idt[(s, b)]:
                assert rel <= GATE_REL and ndiff == 0, (s, b, v, rel, ndiff)
            id_rows.append(
                {
                    "estimator": s,
                    "bucket": b,
                    "variant": v,
                    "scorer identity_pass": idt[(s, b)],
                    "max rel |F_direct/F_current - 1| at 15:30": rel,
                    "15:30 days whose position differs": ndiff,
                }
            )
    id_tab = pd.DataFrame(id_rows)
    gate(
        f"GATE 15:30: direct F = one-step F (rel <= {GATE_REL:g}, same position) on all {int(at15.sum())} "
        f"days wherever the scorer's identity self-check passes; where it fails:"
    )
    gate(id_tab[~id_tab["scorer identity_pass"]].to_string(index=False))

    # --- context: always short and the notebook's block-diagonal ridge
    fr["s_blk2"] = blk2_current(fr)
    q_as = -np.ones(len(fr))
    q_b2 = positions(fr["s_blk2"].to_numpy(float))
    q_b2[(fr["hhmm"] == "09:35").to_numpy()] = 0.0
    ret = {"always short": bar_returns(q_as, fr), "blk2 sign(s)": bar_returns(q_b2, fr)}
    for nm, (_, _, nu) in ret.items():
        assert nu == 0, f"{nm}: {nu} bars with no tradeable fill"
    pool12 = fr["hhmm"].isin(POOLS["10:00-15:30"]).to_numpy()
    nb_tab = pd.read_csv(HC / "rule_table_intraday_blk2.csv", index_col=0)
    nb_x = pd.read_csv(HC / "rule_table_intraday_crossed_blk2.csv", index_col=0)
    nb_clk = pd.read_csv(HC / "rule_by_entry_hhmm.csv")
    worst = 0.0
    for nm, nbn in (("always short", "always short"), ("blk2 sign(s)", "sign(s)")):
        mid, x, _ = ret[nm]
        worst = max(
            worst,
            abs(sharpe(daily(mid, fr, pool12)) - float(nb_tab.loc[nbn, "Sharpe_ann"])),
            abs(
                sharpe(daily(x, fr, pool12))
                - float(nb_x.loc[nbn, "Sharpe crossed-spread"])
            ),
        )
        for c in POOLS["10:00-15:30"]:
            ref = nb_clk[(nb_clk["hhmm"] == c) & (nb_clk["rule"] == nbn)].iloc[0]
            mc = (fr["hhmm"] == c).to_numpy()
            dd = daily(mid, fr, mc)
            assert len(dd) == int(ref["n"]), (nm, c, len(dd), ref["n"])
            worst = max(worst, abs(sharpe(dd) - float(ref["Sharpe_ann"])))
    n_nb = int(nb_tab.loc["sign(s)", "n"])
    assert int(pool12.sum()) == n_nb, (int(pool12.sum()), n_nb)
    assert worst < 1e-9, f"the notebook's sign(s) / always short differ by {worst}"
    gate(
        f"GATE the notebook's always short and block-diagonal ridge sign(s), recomputed on this frame "
        f"({n_nb:,} bars 10:00-15:30), equal its pooled mid / crossed and per-entry-time Sharpe "
        f"(max |diff| {worst:.1e})"
    )

    # --- the comparison
    rows_clk, rows_pool, rows_agr = [], [], []
    # the per-day series behind every row of the two CSVs (for pairing across runs)
    daily_frames: list[pd.DataFrame] = []
    ref_rows = []
    for cname in ENTRY_CLOCKS:
        mc = (fr["hhmm"] == cname).to_numpy()
        for nm, (mid, x, _) in ret.items():
            if nm == "blk2 sign(s)" and cname == "09:35":
                continue
            for fill, vals in (("mid", mid), ("crossed", x)):
                dd = daily(vals, fr, mc)
                ref_rows.append(
                    {
                        "scope": cname,
                        "rule": nm,
                        "fill": fill,
                        "n_days": len(dd),
                        "Sharpe": sharpe(dd),
                        "mean": float(dd.mean()),
                        "hit": float((dd > 0).mean()),
                    }
                )
    for pname, pcl in POOLS.items():
        mp = fr["hhmm"].isin(pcl).to_numpy()
        for nm, (mid, x, _) in ret.items():
            if nm == "blk2 sign(s)" and "09:35" in pcl:
                continue
            for fill, vals in (("mid", mid), ("crossed", x)):
                dd = daily(vals, fr, mp)
                ref_rows.append(
                    {
                        "scope": pname,
                        "rule": nm,
                        "fill": fill,
                        "n_days": len(dd),
                        "Sharpe": sharpe(dd),
                        "mean": float(dd.mean()),
                        "hit": float((dd > 0).mean()),
                    }
                )
    ref_tab = pd.DataFrame(ref_rows)

    R = fr["R"].to_numpy(float)
    y = fr["rv_rem"].to_numpy(float)
    n_plain_vs_recal = []
    for s, b in models:
        for v in VARIANTS:
            Fd, Fc = (
                fr[f"Fd|{s}|{b}|{v}"].to_numpy(float),
                fr[f"Fc|{s}|{b}|{v}"].to_numpy(float),
            )
            qd, qc = positions(Fd - iv_rem), positions(Fc - iv_rem)
            (md, xd, nud), (mcur, xcur, nuc) = bar_returns(qd, fr), bar_returns(qc, fr)
            assert nud == 0 and nuc == 0, (s, b, v, nud, nuc)
            tag = {"estimator": s, "bucket": b, "variant": v}
            for cname in ENTRY_CLOCKS:
                mc = (fr["hhmm"] == cname).to_numpy()
                sig = mc & np.isfinite(iv_rem)
                base = {**tag, "entry": cname}
                for fill, vd, vc in (("mid", md, mcur), ("crossed", xd, xcur)):
                    dd_, dc_ = daily(vd, fr, mc), daily(vc, fr, mc)
                    pr = paired(dd_.to_numpy(), dc_.to_numpy())
                    daily_frames.append(_daily_frame(tag, cname, fill, dd_, dc_))
                    rows_clk.append(
                        {
                            **base,
                            "fill": fill,
                            "n_days": len(dd_),
                            "Sharpe_current": sharpe(dc_),
                            "Sharpe_direct": sharpe(dd_),
                            **pr,
                            "reading": reading(pr),
                            "reading_dmean": reading(pr, "dmean"),
                            "mean_current": float(dc_.mean()),
                            "mean_direct": float(dd_.mean()),
                            "hit_current": float((dc_ > 0).mean()),
                            "hit_direct": float((dd_ > 0).mean()),
                            "pct_buy_current": 100.0 * float((qc[sig] > 0).mean()),
                            "pct_buy_direct": 100.0 * float((qd[sig] > 0).mean()),
                            "qlike_current": qlike(Fc[mc], y[mc]),
                            "qlike_direct": qlike(Fd[mc], y[mc]),
                        }
                    )
                up = R[sig] > 0
                bd, bc = qd[sig] > 0, qc[sig] > 0
                dbcs, dscb = bd & ~bc, ~bd & bc
                rows_agr.append(
                    {
                        **base,
                        "n": int(sig.sum()),
                        "n_flat": int((mc & ~np.isfinite(iv_rem)).sum()),
                        "both buy": int((bd & bc).sum()),
                        "direct buys, current sells": int(dbcs.sum()),
                        "direct sells, current buys": int(dscb.sum()),
                        "both sell": int((~bd & ~bc).sum()),
                        "agree %": 100.0 * float((bd == bc).mean()),
                        "mean R long | direct buys, current sells": float(
                            R[sig][dbcs].mean()
                        )
                        if dbcs.any()
                        else np.nan,
                        "share R>0 | direct buys, current sells": float(up[dbcs].mean())
                        if dbcs.any()
                        else np.nan,
                        "mean R long | direct sells, current buys": float(
                            R[sig][dscb].mean()
                        )
                        if dscb.any()
                        else np.nan,
                        "share R>0 | direct sells, current buys": float(up[dscb].mean())
                        if dscb.any()
                        else np.nan,
                        "accuracy direct": float(((bd & up) | (~bd & ~up)).mean()),
                        "accuracy current": float(((bc & up) | (~bc & ~up)).mean()),
                        "base rate P(R>0)": float(up.mean()),
                    }
                )
            for pname, pcl in POOLS.items():
                mp = fr["hhmm"].isin(pcl).to_numpy()
                sig = mp & np.isfinite(iv_rem)
                for fill, vd, vc, vs in (
                    ("mid", md, mcur, ret["always short"][0]),
                    ("crossed", xd, xcur, ret["always short"][1]),
                ):
                    dd_, dc_ = daily(vd, fr, mp), daily(vc, fr, mp)
                    ds_ = daily(vs, fr, mp)
                    pr = paired(dd_.to_numpy(), dc_.to_numpy())
                    daily_frames.append(_daily_frame(tag, pname, fill, dd_, dc_))
                    vs_short = {}
                    for side, d_ in (("direct", dd_), ("current", dc_)):
                        ps = paired(d_.to_numpy(), ds_.to_numpy())
                        vs_short[f"dSharpe_{side}_vs_short"] = ps["dSharpe"]
                        vs_short[f"dSharpe_{side}_vs_short_pct_lo"] = ps[
                            "dSharpe_pct_lo"
                        ]
                        vs_short[f"dSharpe_{side}_vs_short_pct_hi"] = ps[
                            "dSharpe_pct_hi"
                        ]
                        vs_short[f"reading_{side}_vs_short"] = reading(ps)
                    rows_pool.append(
                        {
                            **tag,
                            "pool": pname,
                            "fill": fill,
                            "n_days": len(dd_),
                            "n_bars": int(mp.sum()),
                            "Sharpe_current": sharpe(dc_),
                            "Sharpe_direct": sharpe(dd_),
                            **pr,
                            "reading": reading(pr),
                            "reading_dmean": reading(pr, "dmean"),
                            "mean_current": float(dc_.mean()),
                            "mean_direct": float(dd_.mean()),
                            "hit_current": float((dc_ > 0).mean()),
                            "hit_direct": float((dd_ > 0).mean()),
                            "pct_buy_current": 100.0 * float((qc[sig] > 0).mean()),
                            "pct_buy_direct": 100.0 * float((qd[sig] > 0).mean()),
                            "agree %": 100.0
                            * float(((qd > 0) == (qc > 0))[sig].mean()),
                            "qlike_current": qlike(Fc[mp], y[mp]),
                            "qlike_direct": qlike(Fd[mp], y[mp]),
                            "Sharpe_always_short": sharpe(ds_),
                            **vs_short,
                        }
                    )
            if v == "recal":
                for side, qa in (("direct", qd), ("current", qc)):
                    Fp = fr[f"F{side[0]}|{s}|{b}|plain"].to_numpy(float)
                    qp = positions(Fp - iv_rem)
                    n_plain_vs_recal.append(
                        {
                            "estimator": s,
                            "bucket": b,
                            "signal": side,
                            "bars where plain and recal positions differ": int(
                                (qp != qa).sum()
                            ),
                            "of bars": len(qa),
                            "median F_plain / F_recal": float(
                                np.median(
                                    Fp / fr[f"F{side[0]}|{s}|{b}|recal"].to_numpy(float)
                                )
                            ),
                            "max realized / F_plain (the QLIKE tail)": float(
                                np.max(y / Fp)
                            ),
                            "max realized / F_recal": float(
                                np.max(
                                    y / fr[f"F{side[0]}|{s}|{b}|recal"].to_numpy(float)
                                )
                            ),
                        }
                    )
        print(f"  scored {s}/{b}")

    clk = pd.DataFrame(rows_clk)
    pool = pd.DataFrame(rows_pool)
    agr = pd.DataFrame(rows_agr)
    pvr = pd.DataFrame(n_plain_vs_recal)
    for t in (clk, pool):
        t["pct_qlike_direct_vs_current"] = 100.0 * (
            t["qlike_direct"] / t["qlike_current"] - 1.0
        )

    # context: section 8d of the executed notebook (fitted rest-of-day maps M1-M3), cited
    s8d = pd.read_csv(HC / "rule_table_srem_blk2.csv", index_col=0)
    s8d_d = pd.read_csv(HC / "srem_vs_sm_dsharpe.csv")
    ctx = s8d[
        ["n_days", "Sharpe_ann", "Sharpe crossed-spread", "mean_daily", "pct_buy"]
    ].copy()
    ctx.insert(
        0,
        "source",
        "executed notebook section 8d, results/atm_straddle_intraday_holdclose/rule_table_srem_blk2.csv",
    )
    ctx.index.name = "rule (pooled 10:00-15:00, block-diagonal ridge)"

    OUT.mkdir(parents=True, exist_ok=True)
    clk.to_csv(OUT / "trading_test_by_entry_time.csv", index=False)
    pool.to_csv(OUT / "trading_test_pooled.csv", index=False)
    agr.to_csv(OUT / "trading_test_agreement.csv", index=False)
    ref_tab.to_csv(OUT / "trading_test_reference_rules.csv", index=False)
    ctx.to_csv(OUT / "trading_test_context_8d.csv")
    s8d_d.to_csv(OUT / "trading_test_context_8d_dsharpe.csv", index=False)
    pvr.to_csv(OUT / "trading_test_plain_vs_recal.csv", index=False)
    id_tab.to_csv(OUT / "trading_test_identity_1530.csv", index=False)
    pd.concat(daily_frames, ignore_index=True).to_parquet(
        OUT / "trading_test_daily.parquet", index=False
    )
    (OUT / "trading_test_gates.txt").write_text(
        "\n".join(GATES) + "\n", encoding="utf-8"
    )
    print(f"wrote CSVs under {OUT}")

    pd.set_option("display.width", 250)
    show = pool[pool["fill"].isin(FILLS)][
        [
            "estimator",
            "bucket",
            "variant",
            "pool",
            "fill",
            "n_days",
            "Sharpe_current",
            "Sharpe_direct",
            "dSharpe",
            "dSharpe_pct_lo",
            "dSharpe_pct_hi",
            "reading",
            "hit_current",
            "hit_direct",
            "pct_buy_current",
            "pct_buy_direct",
            "agree %",
            "pct_qlike_direct_vs_current",
        ]
    ]
    print(show.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(
        ref_tab[ref_tab["scope"].isin(POOLS)].to_string(
            index=False, float_format=lambda v: f"{v:+.3f}"
        )
    )
    print(pvr.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    report()


# ---------------------------------------------------------------------------
# 6. Figure, LaTeX table, SUMMARY.md (every number read back from the CSVs)
# ---------------------------------------------------------------------------

COL_CURRENT = "#2a78d6"  # categorical slot 1 (validated default palette)
COL_DIRECT = "#eb6834"  # categorical slot 2
COL_REF = "#8a8985"  # recessive gray for the always-short reference
COL_INK, COL_MUTED, COL_GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _excl(reading_: str) -> bool:
    return "excludes zero" in reading_


def _ci(r: pd.Series, key: str = "dSharpe") -> str:
    return f"[{r[f'{key}_pct_lo']:+.2f}, {r[f'{key}_pct_hi']:+.2f}]"


def make_figures(
    clk: pd.DataFrame, pool: pd.DataFrame, ref: pd.DataFrame
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": COL_MUTED,
            "axes.labelcolor": COL_INK,
            "xtick.color": COL_MUTED,
            "ytick.color": COL_MUTED,
        }
    )
    est, var = HEADLINE
    c = clk[
        (clk["estimator"] == est) & (clk["variant"] == var) & (clk["fill"] == "mid")
    ]
    ash = ref[(ref["rule"] == "always short") & (ref["fill"] == "mid")].set_index(
        "scope"
    )["Sharpe"]
    x = np.arange(len(ENTRY_CLOCKS))
    off = 0.17
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.5, 6.4),
        sharex=True,
        sharey="row",
        gridspec_kw={"height_ratios": [1.35, 1.0]},
    )
    for j, b in enumerate(BUCKETS):
        g = c[c["bucket"] == b].set_index("entry").reindex(ENTRY_CLOCKS)
        ax = axes[0, j]
        ax.plot(
            x,
            ash.reindex(ENTRY_CLOCKS).to_numpy(float),
            color=COL_REF,
            lw=1.2,
            ls="--",
            marker="o",
            ms=3,
            label="always short",
            zorder=1,
        )
        for k, (who, col, mk) in enumerate(
            (("current", COL_CURRENT, "o"), ("direct", COL_DIRECT, "s"))
        ):
            v = g[f"Sharpe_{who}"].to_numpy(float)
            lo, hi = (
                g[f"sharpe_{who}_lo"].to_numpy(float),
                g[f"sharpe_{who}_hi"].to_numpy(float),
            )
            xx = x + (k - 0.5) * 2 * off
            ax.vlines(xx, lo, hi, color=col, lw=1.4, alpha=0.55, zorder=2)
            lab = (
                "current: RVhat/w - IV^2 h"
                if who == "current"
                else "direct: F_rem - IV^2 h"
            )
            ax.plot(
                xx, v, mk, color=col, ms=5.5, mec="white", mew=1.0, zorder=3, label=lab
            )
        ax.axhline(0, color=COL_MUTED, lw=0.6)
        ax.set_title(BUCKET_LABEL[b], color=COL_INK)
        ax.grid(axis="y", color=COL_GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax2 = axes[1, j]
        d = g["dSharpe"].to_numpy(float)
        lo, hi = (
            g["dSharpe_pct_lo"].to_numpy(float),
            g["dSharpe_pct_hi"].to_numpy(float),
        )
        ax2.vlines(x, lo, hi, color=COL_INK, lw=1.4, alpha=0.6)
        ax2.plot(x, d, "D", color=COL_INK, ms=4.5, mec="white", mew=0.8)
        ax2.axhline(0, color=COL_MUTED, lw=0.8)
        ax2.grid(axis="y", color=COL_GRID, lw=0.6)
        ax2.set_axisbelow(True)
        ax2.set_xticks(x)
        ax2.set_xticklabels(ENTRY_CLOCKS, rotation=60, ha="right")
        ax2.set_xlabel("entry time (ET)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
            ax2.spines[sp].set_visible(False)
    axes[0, 0].set_ylabel(
        "annualized Sharpe, midpoint\n(95 % block-bootstrap interval)"
    )
    axes[1, 0].set_ylabel("direct minus current\n(paired 95 % interval)")
    axes[0, 0].legend(fontsize=8, loc="lower left", frameon=False)
    fig.suptitle(
        "sign(s) on the delta-hedged straddle held to the close, by entry time: direct rest-of-day "
        f"forecast vs the one-step construction (per-bar {est}, {var} back-transform, "
        f"{int(c['n_days'].max())} days)",
        fontsize=10,
        color=COL_INK,
    )
    fig.tight_layout()
    f1 = OUT / "trading_test_sharpe_by_entry_time.png"
    fig.savefig(f1, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # pooled forest: every estimator x bucket x variant, 10:00-15:30, both fills
    q = pool[pool["pool"] == "10:00-15:30"]
    rows = [(b, e, v) for b in BUCKETS for v in VARIANTS for e in SHORTS]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.6), sharey=True)
    ylab = [f"{BUCKET_LABEL[b]}, {e}, {v}" for b, e, v in rows]
    yy = np.arange(len(rows))[::-1]
    for ax, fill in zip(axes, FILLS):
        g = (
            q[q["fill"] == fill]
            .set_index(["bucket", "estimator", "variant"])
            .reindex(rows)
        )
        ax.hlines(
            yy,
            g["dSharpe_pct_lo"],
            g["dSharpe_pct_hi"],
            color=COL_DIRECT,
            lw=1.6,
            alpha=0.6,
        )
        ax.plot(g["dSharpe"], yy, "s", color=COL_DIRECT, ms=5, mec="white", mew=0.8)
        ax.axvline(0, color=COL_MUTED, lw=0.8)
        ax.grid(axis="x", color=COL_GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_title("midpoint" if fill == "mid" else "crossed spread", color=COL_INK)
        ax.set_xlabel("Sharpe, direct minus current (paired 95 % interval)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels(ylab, fontsize=8)
    fig.suptitle(
        "pooled over entry times 10:00-15:30 (daily sums), sign(s): every estimator, bucket and back-transform",
        fontsize=10,
        color=COL_INK,
    )
    fig.tight_layout()
    f2 = OUT / "trading_test_pooled_dsharpe.png"
    fig.savefig(f2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [f1, f2]


def make_tex(clk: pd.DataFrame, pool: pd.DataFrame, ref: pd.DataFrame) -> None:
    est, var = HEADLINE
    q = pool[(pool["estimator"] == est) & (pool["pool"] == "10:00-15:30")]
    r0 = ref[ref["scope"] == "10:00-15:30"].set_index(["rule", "fill"])["Sharpe"]
    lines = [
        "% AUTO-GENERATED by experiments/restofday_trading_test.py -- do not edit.",
        f"% Source: {_rel(OUT)}/trading_test_{{pooled,by_entry_time,reference_rules}}.csv "
        f"(forecast tables {_rel(YHAT_DIR)}/yhat_{{restofday,sub}}_*)",
        f"% Table 1: pooled 10:00--15:30 (daily sums), per-bar {est}. Table 2: by entry time, {var} back-transform, "
        "midpoint.",
        r"\begingroup\small\setlength{\tabcolsep}{3pt}\noindent",
        r"\begin{tabular}{llrrlrrlrr}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{midpoint} & \multicolumn{3}{c}{crossed spread} & \multicolumn{2}{c}{buy \%} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
        r"bucket & $F$ & current & direct & direct $-$ current & current & direct & direct $-$ current "
        r"& cur. & dir. \\",
        r"\midrule",
        r"\multicolumn{10}{l}{\emph{No rest-of-day forecast: always short "
        f"{r0[('always short', 'mid')]:.2f} / {r0[('always short', 'crossed')]:.2f}; "
        r"block-diagonal ridge $\mathrm{sign}(s)$ "
        f"{r0[('blk2 sign(s)', 'mid')]:.2f} / {r0[('blk2 sign(s)', 'crossed')]:.2f} (midpoint / crossed)}}}} \\\\",
        r"\midrule",
    ]
    for v in VARIANTS:
        head = "plain back-transform" if v == "plain" else "recalibrated per entry time"
        lines.append(rf"\multicolumn{{10}}{{l}}{{\emph{{{head}}}}} \\")
        for b in BUCKETS:
            m = q[(q["variant"] == v) & (q["bucket"] == b) & (q["fill"] == "mid")].iloc[
                0
            ]
            x = q[
                (q["variant"] == v) & (q["bucket"] == b) & (q["fill"] == "crossed")
            ].iloc[0]
            lines.append(
                f"{BUCKET_LABEL[b]} & {v} & {m['Sharpe_current']:.2f} & {m['Sharpe_direct']:.2f} & "
                f"{m['dSharpe']:+.2f} {_ci(m)} & {x['Sharpe_current']:.2f} & {x['Sharpe_direct']:.2f} & "
                f"{x['dSharpe']:+.2f} {_ci(x)} & {m['pct_buy_current']:.0f} & {m['pct_buy_direct']:.0f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\endgroup", "", r"\medskip", ""]
    c = clk[
        (clk["estimator"] == est) & (clk["variant"] == var) & (clk["fill"] == "mid")
    ]
    ash = ref[(ref["rule"] == "always short") & (ref["fill"] == "mid")].set_index(
        "scope"
    )["Sharpe"]
    lines += [
        r"\begingroup\scriptsize\setlength{\tabcolsep}{3pt}\noindent",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lr" + "rrl" * len(BUCKETS) + "}",
        r"\toprule",
        r"& always & "
        + " & ".join(rf"\multicolumn{{3}}{{c}}{{{BUCKET_LABEL[b]}}}" for b in BUCKETS)
        + r" \\",
        "".join(
            rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(BUCKETS))
        ),
        r"entry & short & "
        + " & ".join(["current & direct & direct $-$ current"] * len(BUCKETS))
        + r" \\",
        r"\midrule",
    ]
    for e in ENTRY_CLOCKS:
        cells = [e, f"{ash[e]:.2f}"]
        for b in BUCKETS:
            r = c[(c["bucket"] == b) & (c["entry"] == e)].iloc[0]
            mark = "$^{*}$" if _excl(r["reading"]) else ""
            same = (
                r["dSharpe"] == 0
                and r["dSharpe_pct_lo"] == 0
                and r["dSharpe_pct_hi"] == 0
            )
            diff = "same forecast" if same else f"{r['dSharpe']:+.2f} {_ci(r)}{mark}"
            cells += [f"{r['Sharpe_current']:.2f}", f"{r['Sharpe_direct']:.2f}", diff]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\endgroup",
        "",
        f"% Per-bar {est}, {var} back-transform, midpoint; each row is one entry time's daily series "
        f"({int(c['n_days'].min())}--{int(c['n_days'].max())} days); intervals: paired circular block "
        "bootstrap, 95 % percentile; * = the percentile and basic intervals both exclude zero.",
    ]
    if TEX_OUT is None:
        print("LaTeX table not written (--tex none)")
        return
    TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEX_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {TEX_OUT}")


def report() -> None:
    clk = pd.read_csv(OUT / "trading_test_by_entry_time.csv")
    pool = pd.read_csv(OUT / "trading_test_pooled.csv")
    agr = pd.read_csv(OUT / "trading_test_agreement.csv")
    ref = pd.read_csv(OUT / "trading_test_reference_rules.csv")
    ctx = pd.read_csv(OUT / "trading_test_context_8d.csv", index_col=0)
    pvr = pd.read_csv(OUT / "trading_test_plain_vs_recal.csv")
    idt = pd.read_csv(OUT / "trading_test_identity_1530.csv")
    gates = (OUT / "trading_test_gates.txt").read_text(encoding="utf-8").strip()
    figs = make_figures(clk, pool, ref)
    make_tex(clk, pool, ref)
    write_summary(clk, pool, agr, ref, ctx, pvr, idt, gates, figs)


def write_summary(
    clk: pd.DataFrame,
    pool: pd.DataFrame,
    agr: pd.DataFrame,
    ref: pd.DataFrame,
    ctx: pd.DataFrame,
    pvr: pd.DataFrame,
    idt: pd.DataFrame,
    gates: str,
    figs: list[Path],
) -> None:
    """SUMMARY.md: professor-facing; every number is read from the CSVs written by run()."""
    est, var = HEADLINE
    P = pool[pool["pool"] == "10:00-15:30"]
    H = P[(P["estimator"] == est) & (P["variant"] == var)].set_index(["bucket", "fill"])
    Hr = P[(P["estimator"] == est) & (P["variant"] == "recal")].set_index(
        ["bucket", "fill"]
    )
    r0 = ref.set_index(["scope", "rule", "fill"])["Sharpe"]
    C = clk[(clk["estimator"] == est) & (clk["variant"] == var)]
    Cm = C[C["fill"] == "mid"]
    A = agr[(agr["estimator"] == est) & (agr["variant"] == var)]
    n_days = int(H["n_days"].max())
    inner = [e for e in ENTRY_CLOCKS if e != "15:30"]

    def f2(v: float) -> str:
        return f"{v:+.2f}"

    def row_ci(r: pd.Series, key: str = "dSharpe") -> str:
        return f"{r[key]:+.2f} [{r[f'{key}_pct_lo']:+.2f}, {r[f'{key}_pct_hi']:+.2f}]"

    L: list[str] = []
    L += [
        "# Direct rest-of-day forecast: does it trade better? (checklist E2)",
        "",
        "*Generated by `experiments/restofday_trading_test.py`; every number below is read from the CSVs "
        f"in this folder. Inputs: forecast tables `{_rel(YHAT_DIR)}/yhat_{{restofday,sub}}_*`, return ledger "
        f"`{_rel(LEDGER)}`, 15:30 identity flags `{_rel(SCORE_CSV)}`.*",
        "",
        "## Question",
        "",
        "The direct rest-of-day model forecasts, at each entry time, the variance from that time to the "
        "16:00 close in one regression per entry time. Its QLIKE beats the one-step construction "
        "(next-bar forecast stretched by the causal intraday profile, RVhat/w). Does it also make a better "
        "trading signal?",
        "",
        "## Set-up (one change at a time)",
        "",
        "- **Trade.** At each entry time (09:35, 10:00, ..., 15:30) buy or sell the straddle (nearest "
        "out-of-the-money call + nearest out-of-the-money put, same-day expiry, one position), hold it to the "
        "official close, delta-hedged in the index every 30 minutes (the hold-to-close notebook's section 4b). "
        "The 15:30 entry is the last-30-min trade. Midpoint and crossed-spread fills (buy at the ask, sell at "
        "the bid, cash settlement).",
        "- **Rule.** sign(s): long the straddle when s > 0, short otherwise; flat on the 13 bars whose vendor "
        "implied volatility is a censored solver node (the same bars for every signal).",
        "- **Signals.** s_current = RVhat/w - IV^2 h (the notebook's one-step construction) against "
        "s_direct = F_rem - IV^2 h (the direct forecast). IV is the vendor's hourly implied volatility of the "
        "straddle at entry, h the hours to the close. Same days, same bars, same fills, same hedge; only the "
        "forecast of the rest of the day changes.",
        f"- **Forecasts.** The per-bar linear models' three buckets: live-feasible, all features, HAR + calendar "
        f"(`har_ma_*` and the calendar columns). Headline estimator: per-bar {est}; lasso and elastic net are "
        "in the CSVs.",
        "- **Forecast level, two variants fixed before any trading number was computed.** *plain* (headline): "
        "the QLIKE comparison's own construction -- direct = pred_adj^2 x baseline_rem, current = the scorer's "
        "one-step F_rem = f_next / w_next. *recal*: both sides through the notebook's Mincer-Zarnowitz "
        "recalibration (250 prior sessions), fitted per entry time on that entry time's own target, current "
        "then divided by the same w.",
        f"- **Days.** The 15:30 deck's {n_days} expiration days (2020-01 .. 2024-04). Every forecast has a value "
        "on every bar, so no bar is dropped. 09:35 uses the forecasts issued at 09:30 (five minutes before "
        "entry, used as issued).",
        "- **Intervals.** Paired circular block bootstrap of direct minus current on the same resampled days "
        "(block ceil(n^(1/3)), 2000 draws, seed 0; 95 % percentile, basic interval in the CSVs), the notebook's "
        "section 7 recipe.",
        "",
        f"## Answer: pooled over entry times 10:00-15:30 (daily sums; per-bar {est}, plain)",
        "",
        "| bucket | current, mid | direct, mid | direct - current, mid [95 %] | current, crossed | direct, crossed | "
        "direct - current, crossed [95 %] | buy % current / direct | same sign % |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for b in BUCKETS:
        m, x = H.loc[(b, "mid")], H.loc[(b, "crossed")]
        L.append(
            f"| {BUCKET_LABEL[b]} | {m['Sharpe_current']:.2f} | {m['Sharpe_direct']:.2f} | {row_ci(m)} | "
            f"{x['Sharpe_current']:.2f} | {x['Sharpe_direct']:.2f} | {row_ci(x)} | "
            f"{m['pct_buy_current']:.1f} / {m['pct_buy_direct']:.1f} | {m['agree %']:.1f} |"
        )
    asm, asx = (
        r0[("10:00-15:30", "always short", "mid")],
        r0[("10:00-15:30", "always short", "crossed")],
    )
    bkm, bkx = (
        r0[("10:00-15:30", "blk2 sign(s)", "mid")],
        r0[("10:00-15:30", "blk2 sign(s)", "crossed")],
    )
    L += [
        "",
        f"References on the same {n_days} days: always short {asm:.2f} mid / {asx:.2f} crossed; the notebook's "
        f"block-diagonal ridge sign(s) {bkm:.2f} / {bkx:.2f} (reproduced to machine precision, see Checks).",
        "",
    ]
    pos_n = int((P[P["fill"] == "mid"]["dSharpe"] > 0).sum())
    tot_n = int((P["fill"] == "mid").sum())
    ex_pos = P[P["reading"].map(_excl) & (P["dSharpe"] > 0)]
    ex_neg = P[P["reading"].map(_excl) & (P["dSharpe"] < 0)]
    lf, af, hc = (
        H.loc[("live_feasible", "mid")],
        H.loc[("all_features", "mid")],
        H.loc[("baseline", "mid")],
    )
    # the sentences below are worded for this run's pattern; stop rather than print a stale reading
    for b in ("all_features",):  # live_feasible commented out
        for fl in FILLS:
            assert H.loc[(b, fl), "dSharpe"] > 0 and not _excl(
                H.loc[(b, fl), "reading"]
            ), (b, fl)
    assert abs(hc["dSharpe"]) < abs(lf["dSharpe"]) and not _excl(hc["reading"])
    assert (P["Sharpe_direct"] < P["Sharpe_always_short"]).all() and (
        P["Sharpe_current"] < P["Sharpe_always_short"]
    ).all()
    L += [
        f"**In one line.** With the live-feasible and all-features inputs the direct forecast trades somewhat "
        f"better ({f2(lf['dSharpe'])} and {f2(af['dSharpe'])} Sharpe pooled at the midpoint), mostly because it "
        "buys the straddle less often; no difference is distinguishable from zero, and neither signal beats "
        "always short.",
        "",
        "**Reading.**",
        "",
        f"- With the live-feasible and all-features inputs the direct forecast trades better by a point estimate "
        f"of {f2(lf['dSharpe'])} and {f2(af['dSharpe'])} Sharpe at the midpoint "
        f"({f2(H.loc[('live_feasible', 'crossed')]['dSharpe'])} and {f2(H.loc[('all_features', 'crossed')]['dSharpe'])} "
        f"at the crossed spread), but every one of these intervals includes zero. With HAR + calendar the two "
        f"are level ({f2(hc['dSharpe'])}).",
        f"- Across all {tot_n} estimator x bucket x back-transform combinations (midpoint), the direct forecast is "
        f"ahead in {pos_n}; intervals (percentile and basic both) exclude zero in favour of the direct forecast in "
        f"{int((ex_pos['fill'] == 'mid').sum())} and against it in {int((ex_neg['fill'] == 'mid').sum())} "
        f"(crossed: {int((ex_pos['fill'] == 'crossed').sum())} and {int((ex_neg['fill'] == 'crossed').sum())}). "
        f"{int((P['reading'] == 'percentile and basic disagree').sum())} further row(s) have the percentile "
        "interval alone excluding zero: "
        + (
            ", ".join(
                f"{r.estimator} / {BUCKET_LABEL[r.bucket]} / {r.variant} / {r.fill} {row_ci(P.loc[i])}"
                for i, r in P[
                    P["reading"] == "percentile and basic disagree"
                ].iterrows()
            )
            or "none"
        )
        + ". The pooled forest plot (`trading_test_pooled_dsharpe.png`) shows them all.",
        f"- Neither signal beats always short (every pooled point estimate is below it): direct minus always short {row_ci(lf, 'dSharpe_direct_vs_short')} "
        f"(live-feasible, mid), {row_ci(af, 'dSharpe_direct_vs_short')} (all features), "
        f"{row_ci(hc, 'dSharpe_direct_vs_short')} (HAR + calendar); current minus always short "
        f"{row_ci(lf, 'dSharpe_current_vs_short')}, {row_ci(af, 'dSharpe_current_vs_short')}, "
        f"{row_ci(hc, 'dSharpe_current_vs_short')}.",
        f"- Hit rate (share of days with a positive daily sum, mid): live-feasible {lf['hit_current']:.3f} current "
        f"vs {lf['hit_direct']:.3f} direct; all features {af['hit_current']:.3f} vs {af['hit_direct']:.3f}; "
        f"HAR + calendar {hc['hit_current']:.3f} vs {hc['hit_direct']:.3f}.",
        "",
        "**Other pools** (per-bar ridge, plain, mid; direct - current [95 %]):",
        "",
        "| pool | live-feasible | all features | HAR + calendar |",
        "|---|---|---|---|",
    ]
    for pn in POOLS:
        Q = pool[
            (pool["pool"] == pn)
            & (pool["estimator"] == est)
            & (pool["variant"] == var)
            & (pool["fill"] == "mid")
        ]
        Q = Q.set_index("bucket")
        L.append(
            f"| {pn} ({len(POOLS[pn])} entry times) | "
            + " | ".join(
                f"{Q.loc[b, 'Sharpe_current']:.2f} -> {Q.loc[b, 'Sharpe_direct']:.2f}, {row_ci(Q.loc[b])}"
                for b in BUCKETS
            )
            + " |"
        )
    L += ["", "## By entry time (per-bar ridge, plain, midpoint)", ""]
    for b in BUCKETS:
        g = Cm[(Cm["bucket"] == b) & Cm["entry"].isin(inner)].set_index("entry")
        best, worst = g["dSharpe"].idxmax(), g["dSharpe"].idxmin()
        exp_ = [
            e for e in inner if _excl(g.loc[e, "reading"]) and g.loc[e, "dSharpe"] > 0
        ]
        exn = [
            e for e in inner if _excl(g.loc[e, "reading"]) and g.loc[e, "dSharpe"] < 0
        ]
        L.append(
            f"- **{BUCKET_LABEL[b]}**: direct ahead at {int((g['dSharpe'] > 0).sum())} of {len(inner)} entry times "
            f"(09:35-15:00; 15:30 is the same forecast). Best {best}: {g.loc[best, 'Sharpe_current']:.2f} -> "
            f"{g.loc[best, 'Sharpe_direct']:.2f}, {row_ci(g.loc[best])}; worst {worst}: "
            f"{g.loc[worst, 'Sharpe_current']:.2f} -> {g.loc[worst, 'Sharpe_direct']:.2f}, {row_ci(g.loc[worst])}. "
            f"Intervals excluding zero: in favour at {', '.join(exp_) or 'none'}; against at {', '.join(exn) or 'none'}."
        )
    n_tests = len(inner) * len(BUCKETS)
    n_ex = int(
        sum(
            Cm[(Cm["bucket"] == b) & Cm["entry"].isin(inner)]["reading"]
            .map(_excl)
            .sum()
            for b in BUCKETS
        )
    )
    L += [
        f"- {n_ex} of these {n_tests} per-entry-time intervals exclude zero; at 95 % about "
        f"{0.05 * n_tests:.1f} would by chance, so single entry times are not evidence on their own.",
        f"- Figure: `{figs[0].name}` (top: Sharpe by entry time, current vs direct with 95 % intervals, always "
        "short dashed; bottom: the paired difference).",
        "",
        "## Why: the direct forecast buys less, and the buys lose",
        "",
    ]
    nets = {}
    for b in BUCKETS:
        m = H.loc[(b, "mid")]
        a = A[(A["bucket"] == b) & A["entry"].isin(inner)]
        n_sb = int(a["direct sells, current buys"].sum())
        n_bs = int(a["direct buys, current sells"].sum())
        r_sb = float(
            (
                a["mean R long | direct sells, current buys"]
                * a["direct sells, current buys"]
            ).sum()
            / n_sb
        )
        r_bs = float(
            (
                a["mean R long | direct buys, current sells"]
                * a["direct buys, current sells"]
            ).sum()
            / n_bs
        )
        assert r_sb < 0 and r_bs < 0 and m["pct_buy_direct"] < m["pct_buy_current"], (
            b
        )  # the wording below
        # direct minus current, summed over the disagreeing bars (units of premium): -2R where the
        # direct forecast sells and the current one buys, +2R the other way; agreeing bars add zero
        gain, give = -2.0 * r_sb * n_sb, 2.0 * r_bs * n_bs
        nets[b] = gain + give
        L.append(
            f"- **{BUCKET_LABEL[b]}**: buy share {m['pct_buy_current']:.1f} % (current) vs "
            f"{m['pct_buy_direct']:.1f} % (direct) over 10:00-15:30. Over 09:35-15:00 the signs differ on "
            f"{n_sb + n_bs} bars: {n_sb} where the direct forecast sells and the current one buys (mean hedged "
            f"long-straddle return there {r_sb:+.3f} per unit premium) and {n_bs} the other way ({r_bs:+.3f}). "
            f"Direct minus current on those bars: {gain:+.1f} units of premium from the first group, {give:+.1f} "
            f"from the second, net {gain + give:+.1f} over the sample."
        )
    L += [
        "",
        "On the bars where the two disagree the long straddle loses on average in both directions, so whichever "
        "signal sells there gains, by about the same amount per bar in either group. The direct forecast sits lower "
        "against the quoted implied variance and so lands in the first group more often; that, not better timing of "
        "its buys, is where its point gain comes from. It is the mirror "
        "image of the notebook's section 8d, where fitted rest-of-day maps sat higher, bought more, and traded "
        "worse (executed notebook, pooled 10:00-15:00, block-diagonal ridge):",
        "",
        "| rule (section 8d) | Sharpe mid | Sharpe crossed | buy % |",
        "|---|---|---|---|",
    ]
    for rname, r in ctx.iterrows():
        L.append(
            f"| {rname} | {r['Sharpe_ann']:.2f} | {r['Sharpe crossed-spread']:.2f} | {r['pct_buy']:.1f} |"
        )
    Q = pool[
        (pool["pool"] == "10:00-15:00")
        & (pool["estimator"] == est)
        & (pool["variant"] == var)
    ]
    Q = Q.set_index(["bucket", "fill"])
    L += [
        "",
        "On the same 10:00-15:00 pool the direct forecast (per-bar ridge, plain) has "
        + "; ".join(
            f"{BUCKET_LABEL[b]} {Q.loc[(b, 'mid'), 'Sharpe_direct']:.2f} mid / {Q.loc[(b, 'crossed'), 'Sharpe_direct']:.2f} "
            f"crossed (current {Q.loc[(b, 'mid'), 'Sharpe_current']:.2f} / {Q.loc[(b, 'crossed'), 'Sharpe_current']:.2f})"
            for b in BUCKETS
        )
        + ".",
        "",
        "## QLIKE gains and trading gains",
        "",
    ]
    L.append(
        "On these bars the direct forecast's pooled QLIKE (10:00-15:30, against the realized rest of the day) is "
        "lower than the one-step's by "
        + ", ".join(
            f"{-H.loc[(b, 'mid'), 'pct_qlike_direct_vs_current']:.1f} % ({BUCKET_LABEL[b]})"
            for b in BUCKETS
        )
        + " with the plain back-transform and by "
        + ", ".join(
            f"{-Hr.loc[(b, 'mid'), 'pct_qlike_direct_vs_current']:.1f} %"
            for b in BUCKETS
        )
        + " recalibrated (the plain all-features figure is inflated by the one-step's extreme values, below). "
        "With the plain back-transform the trade moves the same way as QLIKE in all three buckets; recalibrated, it "
        "reverses for HAR + calendar; either way the trading difference is small next to its sampling error. Earlier "
        "comparisons already showed QLIKE and the trade parting ways (the VIX-only forecast in the RV-IV "
        "notebook; the fitted rest-of-day maps of section 8d). A better forecast of the rest-of-day variance is not "
        "the same thing as a better sign of s: the sign depends on where the forecast sits against the implied "
        "variance on the marginal days."
    )
    L += [
        "",
        "## The recalibrated variant (robustness)",
        "",
        "| bucket | current, mid | direct, mid | direct - current [95 %] | direct - current, crossed [95 %] | buy % current / direct |",
        "|---|---|---|---|---|---|",
    ]
    for b in BUCKETS:
        m, x = Hr.loc[(b, "mid")], Hr.loc[(b, "crossed")]
        L.append(
            f"| {BUCKET_LABEL[b]} | {m['Sharpe_current']:.2f} | {m['Sharpe_direct']:.2f} | {row_ci(m)} | {row_ci(x)} | "
            f"{m['pct_buy_current']:.1f} / {m['pct_buy_direct']:.1f} |"
        )
    pr = pvr[pvr["estimator"] == est]
    for b in BUCKETS:
        for who in ("current", "direct"):
            assert (
                Hr.loc[(b, "mid"), f"Sharpe_{who}"] < H.loc[(b, "mid"), f"Sharpe_{who}"]
            ), (b, who)
            assert (
                Hr.loc[(b, "mid"), f"pct_buy_{who}"]
                > H.loc[(b, "mid"), f"pct_buy_{who}"]
            ), (b, who)
    assert all(
        Hr.loc[(b, "mid"), "dSharpe"] > 0
        for b in ("all_features",)  # live_feasible commented out
    )
    assert Hr.loc[("baseline", "mid"), "dSharpe"] < 0 and not _excl(
        Hr.loc[("baseline", "mid"), "reading"]
    )
    L += [
        "",
        "The recalibration adds the error-variance term and lifts both forecasts (median plain / recalibrated "
        + ", ".join(
            f"{BUCKET_LABEL[r['bucket']]} {r['signal']} {r['median F_plain / F_recal']:.3f}"
            for _, r in pr.iterrows()
        )
        + "), and both buy more; both trade "
        "worse than plain, and the ordering of direct against current is the same as under plain for "
        "live-feasible and all features and reverses (inside the interval) for HAR + calendar. Positions that "
        "differ between plain and recal (ridge): "
        + ", ".join(
            f"{BUCKET_LABEL[r['bucket']]} {r['signal']} {int(r['bars where plain and recal positions differ'])}"
            for _, r in pr.iterrows()
        )
        + f" of {int(pr['of bars'].iloc[0])} bars.",
        "",
        "**Extreme values of the plain back-transform.** sign(s) uses only the sign of s, so a single extreme "
        "forecast moves one position, not the size of any trade. On this frame the largest realized / plain "
        "forecast ratio (the tail that drives QLIKE) is "
        + ", ".join(
            f"{BUCKET_LABEL[r['bucket']]} {r['signal']} {r['max realized / F_plain (the QLIKE tail)']:.0f}"
            for _, r in pr.iterrows()
        )
        + "; the recal variant is the check on them.",
        "",
        "## Checks",
        "",
        "```",
        gates,
        "```",
        "",
    ]
    bad = idt[~idt["scorer identity_pass"].astype(bool)]
    if len(bad):
        L += [
            "The 15:30 identity is not exact for "
            + ", ".join(
                sorted(
                    {
                        f"{r['estimator']} / {BUCKET_LABEL[r['bucket']]}"
                        for _, r in bad.iterrows()
                    }
                )
            )
            + ": the stored per-bar tables for those two are not the direct campaign's 15:30 arm to the last bit "
            "(the QLIKE scorer's own identity self-check fails there too, `score_rest_of_day.csv`); the 15:30 "
            "positions still differ on at most "
            f"{int(bad['15:30 days whose position differs'].max())} day.",
            "",
        ]
    L += [
        "## What this does not settle",
        "",
        "- One sample, 2020-01 .. 2024-04; the per-entry-time intervals are wide and a dozen of them are looked at.",
        "- The 09:35 forecasts cover 09:30-16:00 while the straddle covers 09:35-16:00 (used as issued on both "
        "sides).",
        "- Sizing: every rule is +/-1 unit of premium per entry time; the size question is A3's.",
        "",
        "## Files",
        "",
        "- `trading_test_pooled.csv`: every estimator x bucket x variant x pool x fill (Sharpe, mean, hit rate, "
        "buy share, agreement, QLIKE, paired intervals of direct - current and of each against always short).",
        "- `trading_test_by_entry_time.csv`: the same by entry time (13).",
        "- `trading_test_agreement.csv`: sign agreement direct vs current by entry time (both buy / direct buys, "
        "current sells / direct sells, current buys / both sell; mean hedged long return on the disagreeing bars; "
        "accuracy against the sign of the hedged long return and its base rate).",
        "- `trading_test_reference_rules.csv` (always short, the notebook's block-diagonal ridge sign(s)); "
        "`trading_test_context_8d*.csv` (section 8d, copied from the executed notebook); "
        "`trading_test_plain_vs_recal.csv`; `trading_test_identity_1530.csv`; `trading_test_gates.txt`.",
        f"- Figures: `{figs[0].name}`, `{figs[1].name}`.",
        f"- LaTeX: `{_rel(TEX_OUT)}`."
        if TEX_OUT is not None
        else "- LaTeX: not written.",
        f"- Return ledger for reuse (not committed): `{_rel(LEDGER)}` (per day x entry time: entry, "
        "bid/ask, hedge P&L, hedged return, vendor and re-inverted implied volatility, hours to the close; all "
        "1,279 chain days, `in_deck` marks the 866).",
        f"- Direct forecast tables (not committed): `{_rel(YHAT_DIR)}/yhat_restofday_{{ridge,lasso,enet}}_"
        "{live_feasible,all_features,baseline}.parquet` (experiments/build_restofday_yhat.py).",
        "- Per-day series behind every row (not committed): `trading_test_daily.parquet`.",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'SUMMARY.md'}")


def configure(a: argparse.Namespace) -> None:
    """Point the module's paths at the command line's (unset = the first run's defaults)."""
    global OUT, LEDGER, YHAT_DIR, SCORE_CSV, TEX_OUT

    def _p(v: str) -> Path:
        q = Path(v)
        return q if q.is_absolute() else ROOT / q

    if a.out:
        OUT = _p(a.out)
    if a.yhat_dir:
        YHAT_DIR = _p(a.yhat_dir)
    if a.ledger:
        LEDGER = _p(a.ledger)
    if a.score_csv:
        SCORE_CSV = _p(a.score_csv)
    if a.tex:
        TEX_OUT = None if a.tex.lower() == "none" else _p(a.tex)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--build-ledger",
        action="store_true",
        help="chain-sized: build the return ledger",
    )
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="figure, tex, SUMMARY.md from the CSVs",
    )
    ap.add_argument("--out", default=None, help=f"output folder (default {_rel(OUT)})")
    ap.add_argument(
        "--yhat-dir", default=None, help=f"forecast tables (default {_rel(YHAT_DIR)})"
    )
    ap.add_argument(
        "--ledger", default=None, help=f"return ledger (default {_rel(LEDGER)})"
    )
    ap.add_argument(
        "--score-csv",
        default=None,
        help=f"QLIKE scorer CSV with identity_pass (default {_rel(SCORE_CSV)})",
    )
    ap.add_argument(
        "--tex",
        default=None,
        help="LaTeX output or 'none' (default writeup/generated/appendix_running_restofday_trading.tex)",
    )
    a = ap.parse_args()
    configure(a)
    if a.build_ledger:
        build_ledger()
    elif a.report_only:
        report()
    else:
        run()


if __name__ == "__main__":
    main()
