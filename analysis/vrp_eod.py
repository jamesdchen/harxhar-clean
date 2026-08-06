"""Tier-1 session VRP pricing — the consumer of writeup/om_0dte_atopen_export_spec.md.

Prices the §29.1 objects against real OptionMetrics quotes. Tier 1 = the close-decision trade:
at 16:00 of day D the dte=1 ATM straddle (expiry = next trading day) prices the close-to-close
window [16:00 D, 16:00 D+1]; the model side is the SAME window's embargoed forecast (the
16:30-labeled row of `straddle_eod.npz` — its features are known at 16:00 D and its label
integrates exactly to the next close). The overnight decomposition (straddle_overnight.npz +
the model-free baseline share) attributes the edge between the overnight and session pieces —
the session piece is what an at-open 0DTE straddle (Tier 2, purchase-gated) would trade.

Data expected (gitignored land, not in the repo):
  data/om_friction/chain_109820_dte0_4.parquet   (or the prior chain_109820_dte0_10.parquet)
  data/optionm_spx_spot.parquet                  (date, close — v3 conventions, spot = close/10)

Stages:
  ``gates``  — the five known-answer gates from the spec. Run FIRST; a failed gate means an
               export/join bug, and no research readout is trustworthy past it.
  ``vrp``    — the readout: VRP ratio, gross/net edge per cost tier, era split (2022-11+ first),
               day-of-week split, overnight/session attribution.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_manifestation import TW  # noqa: E402
from analysis.alpha_panel import load_panel  # noqa: E402
from analysis.synthesis import _p  # noqa: E402
from src.evaluation.metrics import apply_duan_smearing  # noqa: E402

CHAIN_CANDIDATES = [
    "data/om_friction/chain_109820_dte0_4.parquet",
    "data/om_friction/chain_109820_dte0_10.parquet",
]
SPOT = "data/optionm_spx_spot.parquet"
ERA_0DTE = "2022-11-01"
SQ2PI = float(np.sqrt(2.0 / np.pi))


def _load_chain() -> tuple[pd.DataFrame, str]:
    for f in CHAIN_CANDIDATES:
        if os.path.exists(f):
            cols = ["date", "exdate", "cp_flag", "strike_price", "best_bid", "best_offer",
                    "impl_volatility", "vega"]
            fr = pd.read_parquet(f)
            extra = [c for c in ("volume", "open_interest", "ss_flag", "am_settlement")
                     if c in fr.columns]
            fr = fr[cols + extra]
            if "ss_flag" in fr.columns:
                fr = fr[fr.ss_flag == 0]
            if "am_settlement" in fr.columns:
                fr = fr[fr.am_settlement == 0]
            return fr, f
    raise SystemExit(
        "No chain parquet found. Land the export per writeup/om_0dte_atopen_export_spec.md at\n"
        "  data/om_friction/chain_109820_dte0_4.parquet  (schema of the prior dte0_10 land)\n"
        "  data/optionm_spx_spot.parquet\n"
        "then re-run. (In a remote session, upload and `mkdir -p data/om_friction && mv ...`.)")


def _atm_straddles() -> pd.DataFrame:
    """One row per (date, exdate): ATM straddle iv, mid, HSV — v3 conventions + the spec's
    Brenner-Subrahmanyam fallback for OM's null IVs at ultra-short dte."""
    fr, src = _load_chain()
    print(f"chain: {src} ({len(fr)} rows)")
    sp = pd.read_parquet(SPOT)
    fr = fr.merge(sp[["date", "close"]], on="date", how="inner")
    fr["spot"] = fr.close / 10.0  # secid 109820 = SPY; strikes on SPX scale (v3 convention)
    fr = fr[(fr.best_offer > 0) & (fr.best_offer >= fr.best_bid)]
    fr["dist"] = (fr.strike_price / 1000.0 - fr.spot).abs()
    best = fr.loc[fr.groupby(["date", "exdate"])["dist"].idxmin(),
                  ["date", "exdate", "strike_price"]]
    atm = fr.merge(best, on=["date", "exdate", "strike_price"])
    g = (atm.groupby(["date", "exdate"])
         .agg(iv_om=("impl_volatility", "mean"), ask=("best_offer", "sum"),
              bid=("best_bid", "sum"), vega=("vega", "sum"), ncp=("cp_flag", "nunique"),
              dist=("dist", "first"), spot=("spot", "first"))
         .reset_index())
    g = g[(g.ncp == 2) & (g.dist / g.spot < 0.02) & (g.bid > 0)]
    # trading-day gap from the OM calendar itself (dates present in the chain)
    tdays = np.sort(pd.to_datetime(sp.date.unique()).values)
    di = np.searchsorted(tdays, pd.to_datetime(g.date).values)
    ei = np.searchsorted(tdays, pd.to_datetime(g.exdate).values)
    g["ntd"] = ei - di
    g["tau"] = g.ntd / 252.0
    g["mid"] = (g.ask + g.bid) / 2.0
    # BS ATM approximation: straddle_mid = spot * iv * sqrt(2 tau / pi)
    g["iv_bs"] = g.mid / (g.spot * SQ2PI * np.sqrt(g.tau.clip(lower=1e-9)))
    g["iv"] = g.iv_om.fillna(g.iv_bs)
    vega_fallback = 2.0 * g.spot * 0.3989 * np.sqrt(g.tau.clip(lower=1e-9))
    g["hsv"] = (g.ask - g.bid) / 2.0 / g.vega.fillna(vega_fallback).replace(0.0, np.nan)
    g["ivvar"] = g.iv**2 * g.tau  # variance in squared-return units over the window
    return g


def _model_side() -> pd.DataFrame:
    """Per-day model forecasts in raw variance units: close-to-close (16:30 row of the EOD
    walk), plus overnight and open-session pieces when their caches exist."""
    p = load_panel()
    ts = pd.Series(pd.to_datetime(p.t))
    z = np.load(_p("straddle_eod.npz"))
    yh, Bh, hb = z["y_eod"], z["B_eod"], z["h_bars"]
    out = {}
    for arm in ("one-stage_679", "backbone"):
        f = np.full(len(ts), np.nan)
        f[TW:] = z[arm]  # walk outputs are aligned to rows [TW, n)
        m = np.isfinite(f) & np.isfinite(yh)
        praw = np.full(len(ts), np.nan)
        praw[m] = apply_duan_smearing(f[m], yh[m], Bh[m])[0]
        out[arm] = praw
    rows = (ts.dt.hour == 16) & (ts.dt.minute == 30)
    d = pd.DataFrame({
        "date": ts[rows].dt.normalize().to_numpy(),
        "F_cc": out["one-stage_679"][rows], "F_cc_backbone": out["backbone"][rows],
        "R_cc": (yh**2 * Bh)[rows],
    })
    if os.path.exists(_p("straddle_overnight.npz")):
        zo = np.load(_p("straddle_overnight.npz"))
        fo = np.full(len(ts), np.nan)
        fo[TW:] = zo["one-stage_679"]  # walk alignment: rows [TW, n)
        mo = np.isfinite(fo) & np.isfinite(zo["y_on"])
        praw = np.full(len(ts), np.nan)
        praw[mo] = apply_duan_smearing(fo[mo], zo["y_on"][mo], zo["B_on"][mo])[0]
        d["F_on"] = praw[rows]
        d["R_on"] = (zo["y_on"] ** 2 * zo["B_on"])[rows]
    # model-free session share: trailing-year fraction of baseline in RTH bars
    hr = ts.dt.hour.to_numpy()
    sess = (hr >= 10) & (hr <= 16)
    day = ts.dt.normalize()
    bs = pd.DataFrame({"day": day, "b": p.baseline, "s": np.where(sess, p.baseline, 0.0)})
    dd = bs.groupby("day").sum()
    share = (dd.s.rolling(252, min_periods=60).sum() / dd.b.rolling(252, min_periods=60).sum())
    d = d.merge(share.rename("sess_share").reset_index().rename(columns={"day": "date"}),
                on="date", how="left")
    return d


def _join() -> pd.DataFrame:
    g = _atm_straddles()
    tier1 = g[g.ntd == 1].sort_values("dist").groupby("date").first().reset_index()
    m = _model_side()
    tier1["date"] = pd.to_datetime(tier1.date).dt.normalize()
    j = m.merge(tier1[["date", "iv", "ivvar", "hsv", "ntd", "spot"]], on="date", how="inner")
    j = j[np.isfinite(j.F_cc) & np.isfinite(j.ivvar) & (j.R_cc > 0)]
    return j


def stage_gates() -> None:
    j = _join()
    n = len(j)
    print(f"\nmatched close-decision days: {n}")
    ratio = j.ivvar.mean() / j.R_cc.mean()
    print(f"GATE 1  VRP ratio (implied/realized, dte=1): {ratio:.3f}  "
          f"[band 1.05-1.60] {'PASS' if 1.05 <= ratio <= 1.60 else 'FAIL'}")
    j["era"] = np.where(j.date >= ERA_0DTE, "0dte", np.where(j.date >= "2016-01-01",
                                                             "mwf", "early"))
    tot = j.groupby("era").size()
    days = _model_side()
    days["era"] = np.where(days.date >= ERA_0DTE, "0dte",
                           np.where(days.date >= "2016-01-01", "mwf", "early"))
    denom = days.groupby("era").size()
    print("GATE 2  match rate per era (expect ~1.00 / ~0.60 / thin):")
    for e in ("0dte", "mwf", "early"):
        print(f"        {e:6s} {tot.get(e, 0) / max(denom.get(e, 1), 1):.2f} "
              f"({tot.get(e, 0)}/{denom.get(e, 0)})")
    med = float(np.nanmedian(j.hsv)) * 100
    print(f"GATE 3  median half-spread {med:.2f} vol pts  "
          f"[expect >= 0.15, < 0.75] {'PASS' if 0.15 <= med < 0.75 else 'CHECK'}")
    g = _atm_straddles()
    z0 = g[g.ntd == 0]
    if len(z0):
        print(f"GATE 4  dte=0 straddle mid at close: median {z0.mid.median():.3f} "
              f"(PM-settled -> should be near intrinsic/zero; large values = settlement bug)")
    else:
        print("GATE 4  no dte=0 rows in export (fine if export starts at dte>=1)")
    dup = j.date.duplicated().sum()
    print(f"GATE 5  join integrity: {dup} duplicate dates {'PASS' if dup == 0 else 'FAIL'}")


def stage_vrp() -> None:
    j = _join()
    j["era"] = np.where(j.date >= ERA_0DTE, "0dte",
                        np.where(j.date >= "2016-01-01", "mwf", "early"))
    # edge in vol points on the close-to-close window (v3 units): sigma in decimal vol
    j["sig_f"] = np.sqrt(j.F_cc.clip(lower=0) * 252.0)   # R/F in squared-return units, 1 td
    j["sig_i"] = np.sqrt(j.ivvar * 252.0)
    j["sig_r"] = np.sqrt(j.R_cc * 252.0)
    j["edge"] = j.sig_f - j.sig_i
    j["pnl_gross"] = np.sign(j.edge) * (j.sig_r - j.sig_i)   # variance-swap proxy, vol units
    print(f"days {len(j)}   VRP ratio {j.ivvar.mean() / j.R_cc.mean():.3f}   "
          f"mean(sig_i - sig_r) {100 * (j.sig_i - j.sig_r).mean():+.2f} vol pts")
    if "F_on" in j.columns and j.F_on.notna().any():
        att = j.dropna(subset=["F_on"])
        imp_sess_model = att.ivvar - att.F_on
        imp_sess_share = att.ivvar * att.sess_share
        print("\novernight decomposition of implied (session piece, share of total):")
        print(f"  model overnight subtraction: {np.mean(imp_sess_model / att.ivvar):.2f}  "
              f"baseline share: {np.mean(imp_sess_share / att.ivvar):.2f}  "
              f"corr of the two session-implieds: "
              f"{np.corrcoef(imp_sess_model, imp_sess_share)[0, 1]:+.3f}")
    else:
        print("\n(no straddle_overnight.npz yet — overnight attribution skipped)")
    print("\nreadout per era / cost tier (trade when |edge| > tier cost; vol-pt P&L/day):")
    for era in ("0dte", "mwf", "early", "ALL"):
        s = j if era == "ALL" else j[j.era == era]
        if len(s) < 30:
            print(f"  {era:5s} n={len(s)} (thin, skipped)")
            continue
        for tier, cost in (("mid", 0.0), ("base", 0.5 * 1.0), ("full", 1.0)):
            c = cost * s.hsv
            tr = s[np.abs(s.edge) > c]
            if len(tr) < 20:
                print(f"  {era:5s} {tier:4s} traded {len(tr)} (thin)")
                continue
            pnl = np.sign(tr.edge) * (tr.sig_r - tr.sig_i) - c[tr.index]
            t = pnl.mean() / (pnl.std() / np.sqrt(len(pnl)))
            print(f"  {era:5s} {tier:4s} traded {len(tr):4d}/{len(s):4d}  "
                  f"mean {100 * pnl.mean():+.3f} volpt  t {t:+.2f}  "
                  f"hit {(pnl > 0).mean():.2f}")
    dow = j.assign(dow=pd.to_datetime(j.date).dt.dayofweek).groupby("dow").apply(
        lambda s: pd.Series({"n": len(s),
                             "vrp": 100 * (s.sig_i - s.sig_r).mean(),
                             "edge_t": (np.sign(s.edge) * (s.sig_r - s.sig_i)).mean()
                             / ((np.sign(s.edge) * (s.sig_r - s.sig_i)).std()
                                / np.sqrt(len(s)))}), include_groups=False)
    print("\nday-of-week (expiry-day effects; 0=Mon):")
    print(dow.to_string())
    out = "results/alpha_manifestation/vrp_tier1.csv"
    j.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gates", "vrp"], required=True)
    a = ap.parse_args()
    {"gates": stage_gates, "vrp": stage_vrp}[a.stage]()
