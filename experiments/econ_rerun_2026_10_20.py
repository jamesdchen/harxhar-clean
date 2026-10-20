"""Economics re-score of the pooled linear reruns of 2026-10-19 (2026-10-20).

Read-only on every input; writes only results/econ_rerun_2026-10-20/.

FORECASTS (the like-for-like design of the new Table 1: base-5 HAR, FOMC in, E6 and E9 column cuts, data fixes on,
frozen pre-sample penalties, 2,000-day window)
  Panel A, 48 intervals: results/pooled_linear_rerun_2026-10-19_48/forecasts/<model>_<set>.parquet, column plain
    (= pred_adj^2 x B) on the row whose naive-ET interval end is 16:00, i.e. the 15:30 forecast of the closing half hour.
  Panel B, 13 intervals: results/pooled_linear_rerun_2026-10-19_13/forecasts.parquet, the same column on the 16:00 row.
  Models: OLS (HAR benchmark), ridge, lasso, elastic net on the HAR benchmark and on all features (7 a panel).

QUOTE BOOK  results/atm_straddle_0dte_1530/daily_a0.parquet (= make_rehaul_economics.py's gui._QUOTE_BOOK; identical
  in R, exit, bids, asks and iv_var to the master table's deck daily_blk2.parquet): the 866 option days.

TRADE (make_rehaul_economics.py load / add_pooled_trees; experiments/master_table_close.py trade_days)
  q = +1 if plain > iv_var else -1; mid r = q R; bid/ask: a buy pays the ask, a sale gets the bid, spread at entry
  only (r = q (exit / ask - 1) for q > 0, q (exit / bid - 1) for q < 0).  Always short: q = -1.

STATISTICS (the paper of record, 07_economics.tex and make_rehaul_economics.py)
  SR       asl.sharpe_ann_by_year (N_y-weighted average of the calendar-year ratios) on days t > 252 (253-866, 614);
  mean, hit rate, buys and every HAC t on all 866 days: asl.newey_west_t (Bartlett, lag floor(1.5 n^(1/3)) = 14);
  dSR      difference of the two printed SR; its t is the HAC t of the daily return difference (a test of the mean
           difference, not of the Sharpe difference), mid and bid/ask;
  Sharpe test  Ledoit and Wolf (2008), as experiments/sharpe_tests_pooled.py (its functions imported, unchanged):
           HAC delta-method p (p_hac) and the studentized circular block bootstrap p (b = 5 primary, b = 10, and the
           calibrated block b_cal), M = 4999, on the daily (fixed-factor) Sharpe difference, on 614 and on 866 days;
           Lo (2002) SR = 0 for every series.  Seed 20261017 + 20.
  QLIKE    mean daily QLIKE of the 16:00 forecast against the actual RV of the forecast file, on the 866 option days
           (context only; Table 1 scores the 1,082-day span).

GATES  always short reproduces \\rheShortSh (0.77), \\rheShortShX (0.42), \\rheShortT (0.43), \\rheShortHit (61.8) and the
  master table's always_short row (Sharpe_mid, Sharpe_crossed, mean_mid, hit_rate_mid on 866 days, fixed factor), and
  sharpe_tests_pooled.csv's Lo row of the always-short series (614 days); every forecast covers the 866 days.

Run:  python experiments/econ_rerun_2026_10_20.py [--no-calib]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import atm_straddle_lib as asl  # noqa: E402
import sharpe_tests_pooled as st  # noqa: E402

OUT = ROOT / "results" / "econ_rerun_2026-10-20"
QUOTES = ROOT / "results" / "atm_straddle_0dte_1530" / "daily_a0.parquet"
RUN48 = ROOT / "results" / "pooled_linear_rerun_2026-10-19_48" / "forecasts"
RUN13 = ROOT / "results" / "pooled_linear_rerun_2026-10-19_13" / "forecasts.parquet"
MT = ROOT / "results" / "close_master_table" / "master_table.csv"
ST_CSV = ROOT / "results" / "sharpe_tests_pooled_2026-10-17" / "sharpe_tests_pooled.csv"
WARMUP, N_OPT, HAC_LAG = 252, 866, 14
SEED = st.SEED + 20
MODELS = (("ols", "baseline"), ("ridge", "baseline"), ("lasso", "baseline"), ("enet", "baseline"),
          ("ridge", "all_features"), ("lasso", "all_features"), ("enet", "all_features"))
PANELS = ("48", "13")
SHORT = "always_short"
# the printed always-short numbers (writeup/generated/rehaul_econ_numbers.tex, read 2026-10-20)
PRINTED_SHORT = {"sh_mid": 0.77, "sh_x": 0.42, "t_mean": 0.43, "hit": 61.8}


def key(panel: str, model: str, pset: str) -> str:
    return f"p{panel}_{model}_{pset}"


# ============================================================================= data
def load_book() -> pd.DataFrame:
    q = pd.read_parquet(QUOTES, columns=["R", "exit", "bid_c", "bid_p", "ask_c", "ask_p", "iv_var"])
    q.index = pd.DatetimeIndex(pd.to_datetime(q.index)).normalize().as_unit("ns")
    assert len(q) == N_OPT and q.index.is_monotonic_increasing and not q.index.duplicated().any()
    return pd.DataFrame({"R": q["R"].astype(float), "ex": q["exit"].astype(float),
                         "bid": (q["bid_c"] + q["bid_p"]).astype(float),
                         "ask": (q["ask_c"] + q["ask_p"]).astype(float), "iv_var": q["iv_var"].astype(float)},
                        index=q.index)


def at16(df: pd.DataFrame, rv_col: str) -> pd.DataFrame:
    t = pd.to_datetime(df["date"])
    m = ((t.dt.hour == 16) & (t.dt.minute == 0)).to_numpy()
    g = pd.DataFrame({"plain": df["plain"].to_numpy(float)[m], "rv": df[rv_col].to_numpy(float)[m]},
                     index=pd.DatetimeIndex(t[m]).normalize().as_unit("ns"))
    assert not g.index.duplicated().any()
    return g


def load_forecasts() -> dict[str, pd.DataFrame]:
    out = {}
    for model, pset in MODELS:
        out[key("48", model, pset)] = at16(pd.read_parquet(RUN48 / f"{model}_{pset}.parquet", columns=["date", "plain", "rv"]), "rv")
    f13 = pd.read_parquet(RUN13, columns=["model", "predictor_set", "date", "plain", "rv_actual"])
    for model, pset in MODELS:
        g = f13[(f13["model"] == model) & (f13["predictor_set"] == pset)]
        assert len(g), (model, pset)
        out[key("13", model, pset)] = at16(g, "rv_actual")
    return out


def crossed(q, ex, bid, ask) -> np.ndarray:
    """writeup/master_table_gui/rows.py _crossed_return (as writeup/make_rehaul_results.py crossed)."""
    return np.where(q > 0, q * (ex / ask - 1.0), np.where(q < 0, q * (ex / bid - 1.0), 0.0))


def trade(book: pd.DataFrame, fc: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    days = book.index
    R, ex, bid, ask, iv = (book[c].to_numpy(float) for c in ("R", "ex", "bid", "ask", "iv_var"))
    series = {}
    for k, g in fc.items():
        g = g.reindex(days)
        p, rv = g["plain"].to_numpy(float), g["rv"].to_numpy(float)
        assert np.isfinite(p).all() and (p > 0).all(), (k, int((~np.isfinite(p)).sum()))
        assert np.isfinite(rv).all() and (rv > 0).all(), k
        q = np.where(p > iv, 1.0, -1.0)
        ratio = rv / p
        series[k] = pd.DataFrame({"plain": p, "rv": rv, "iv_var": iv, "q": q, "mid": q * R,
                                  "x": crossed(q, ex, bid, ask), "qlike": ratio - np.log(ratio) - 1.0}, index=days)
    q = -np.ones(len(days))
    series[SHORT] = pd.DataFrame({"iv_var": iv, "q": q, "mid": -R, "x": crossed(q, ex, bid, ask)}, index=days)
    return series


# ============================================================================= statistics
def sharpe(x, days) -> float:
    """The paper's Sharpe ratio: year-by-year annualization on days t > WARMUP."""
    x = np.asarray(x, float)
    assert len(x) == N_OPT
    return float(asl.sharpe_ann_by_year(x[WARMUP:], dates=days[WARMUP:])["sharpe"])


def nw_t(x) -> float:
    t, lag = asl.newey_west_t(np.asarray(x, float))
    assert lag == HAC_LAG, lag
    return float(t)


def levels(series, days) -> pd.DataFrame:
    rows = []
    for k, g in series.items():
        mid, x = g["mid"].to_numpy(float), g["x"].to_numpy(float)
        panel, model, pset = ("-", SHORT, "-") if k == SHORT else (k[1:3], k.split("_", 2)[1], k.split("_", 2)[2])
        r = {"key": k, "panel": panel, "model": model, "predictor_set": pset,
             "sh_mid": sharpe(mid, days), "sh_x": sharpe(x, days),
             "sh_mid_252_614": float(mid[WARMUP:].mean() / mid[WARMUP:].std(ddof=1) * math.sqrt(252)),
             "sh_mid_252_866": float(mid.mean() / mid.std(ddof=1) * math.sqrt(252)),
             "sh_x_252_866": float(x.mean() / x.std(ddof=1) * math.sqrt(252)),
             "mean_mid": float(mid.mean()), "t_mean_mid": nw_t(mid), "mean_x": float(x.mean()), "t_mean_x": nw_t(x),
             "hit_pct": 100 * float((mid > 0).mean()), "buys_pct": 100 * float((g["q"] > 0).mean()),
             "qlike_866": float(g["qlike"].mean()) if "qlike" in g else np.nan}
        for sample, sl in (("614", slice(WARMUP, None)), ("866", slice(None))):
            lo = st.lo_test(mid[sl])
            r[f"lo_z_{sample}"], r[f"lo_p_{sample}"] = lo["stat"], lo["p_hac"]
        rows.append(r)
    return pd.DataFrame(rows)


def pair_list() -> list[tuple[str, str, str]]:
    """(comparison, a, b): the differential is a minus b."""
    out = []
    for p in PANELS:
        for m, s in MODELS:
            out.append(("vs_short", key(p, m, s), SHORT))
        for m, s in MODELS[1:]:
            out.append(("vs_ols", key(p, m, s), key(p, "ols", "baseline")))
        for m in ("ridge", "lasso", "enet"):
            out.append(("all_vs_har", key(p, m, "all_features"), key(p, m, "baseline")))
    for m, s in MODELS:
        out.append(("48_vs_13", key("48", m, s), key("13", m, s)))
    return out


def pairs(series, days, calib: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    rows, cal = [], []
    for comp, a, b in pair_list():
        ga, gb = series[a], series[b]
        same = float((ga["q"].to_numpy() == gb["q"].to_numpy()).mean())
        r = {"comparison": comp, "a": a, "b": b, "same_position_pct": 100 * same}
        for side, col in (("mid", "mid"), ("x", "x")):
            ra, rb = ga[col].to_numpy(float), gb[col].to_numpy(float)
            r[f"dsh_{side}"] = sharpe(ra, days) - sharpe(rb, days)
            d = ra - rb
            r[f"dmean_{side}"] = float(d.mean())
            r[f"t_dmean_{side}"] = nw_t(d) if np.any(d != 0) else np.nan
        if "qlike" in ga and "qlike" in gb:
            r["dqlike_pct_866"] = 100 * (ga["qlike"].mean() / gb["qlike"].mean() - 1.0)
        ra_all, rb_all = ga["mid"].to_numpy(float), gb["mid"].to_numpy(float)
        for sample, sl in (("614", slice(WARMUP, None)), ("866", slice(None))):
            ri, rn = ra_all[sl], rb_all[sl]
            if np.array_equal(ri, rn):
                for c in ("lw_dsr_ann", "lw_z", "lw_p_hac", "lw_p_boot", "lw_p_boot_b10", "lw_b_cal", "lw_p_boot_bcal"):
                    r[f"{c}_{sample}"] = np.nan
                continue
            L = st.nw_lag(len(ri))
            dd, se = st.lw_hac(ri, rn, L)
            p5 = st.lw_boot_p(ri, rn, st.B_PRIMARY, st.M_BOOT, rng, dd, se)
            p10 = st.lw_boot_p(ri, rn, st.B_SENS, st.M_BOOT, rng, dd, se)
            b_cal, p_cal = np.nan, np.nan
            if calib and sample == "614":
                b_cal, cov = st.lw_calibrate(ri, rn, L, rng)
                p_cal = {5: p5, 10: p10}.get(b_cal) or st.lw_boot_p(ri, rn, b_cal, st.M_BOOT, rng, dd, se)
                cal.append({"a": a, "b": b, "b_cal": b_cal, **{f"cover_b{bb}": v for bb, v in cov.items()}})
            r.update({f"lw_dsr_ann_{sample}": dd * st.ANN, f"lw_z_{sample}": dd / se, f"lw_p_hac_{sample}": st.p2(dd / se),
                      f"lw_p_boot_{sample}": p5, f"lw_p_boot_b10_{sample}": p10, f"lw_b_cal_{sample}": b_cal,
                      f"lw_p_boot_bcal_{sample}": p_cal, f"lw_lag_{sample}": L})
        rows.append(r)
        print(comp, a, b, "done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(cal)


# ============================================================================= gates
def gates(lv: pd.DataFrame) -> list[str]:
    out = []
    s = lv.set_index("key").loc[SHORT]
    for c, want in PRINTED_SHORT.items():
        got = {"hit": s["hit_pct"], "t_mean": s["t_mean_mid"]}.get(c, s.get(c))
        dec = 1 if c == "hit" else 2
        assert round(float(got), dec) == want, (c, got, want)
        out.append(f"always short {c}: {got:.6f} rounds to the printed {want}")
    mt = pd.read_csv(MT).set_index("key").loc[SHORT]
    for got, col in ((s["sh_mid_252_866"], "Sharpe_mid"), (s["sh_x_252_866"], "Sharpe_crossed"),
                     (s["mean_mid"], "mean_mid"), (s["hit_pct"] / 100, "hit_rate_mid")):
        assert abs(float(got) - float(mt[col])) < 1e-8, (col, got, mt[col])
        out.append(f"always short {col} = master_table.csv ({float(mt[col]):.8f})")
    if ST_CSV.is_file():
        stc = pd.read_csv(ST_CSV)
        want = float(stc[(stc["sample"] == "oos614") & (stc["key"] == SHORT) & (stc["test"] == "Lo2002 SR=0")]["SR_ann"].iloc[0])
        assert abs(float(s["sh_mid_252_614"]) - want) < 1e-5, (s["sh_mid_252_614"], want)
        out.append(f"always short fixed-factor SR on 614 days = sharpe_tests_pooled.csv ({want:.6f})")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-calib", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    book = load_book()
    series = trade(book, load_forecasts())
    days = book.index
    lv = levels(series, days)
    gl = gates(lv)
    pr, cal = pairs(series, days, calib=not a.no_calib)
    lv.to_csv(OUT / "levels.csv", index=False, float_format="%.8g")
    pr.to_csv(OUT / "pairs.csv", index=False, float_format="%.8g")
    if len(cal):
        cal.to_csv(OUT / "block_calibration.csv", index=False)
    daily = pd.concat({k: g[["q", "mid", "x"]] for k, g in series.items()}, names=["key", "day"]).reset_index()
    daily.to_parquet(OUT / "daily.parquet", index=False)
    (OUT / "gates.txt").write_text("\n".join(gl) + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    print("\n".join(gl))
    print(lv[["key", "sh_mid", "sh_x", "t_mean_mid", "hit_pct", "buys_pct", "qlike_866"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
