"""Strategy variations of the 15:30 last-30-min trade (checklist F1).

The variations are written down in results/strategy_variations/VARIATIONS.md:
strangles (one strike wide, +-0.5 %, +-1 %, +-2 %), iron butterflies and iron
condors, credit put / call verticals, call / put butterflies at the money, 1x2
ratio spreads, and the straddle with wings on its selling days, all same-day
expiry, entered at the 15:30 quote and cash-settled at the official close.

Two steps.

  --extract   CHAIN-SIZED (hold the chain lock): one pass over
              data/spxw_chain.parquet that keeps the 15:30 and 16:00 ET same-day
              expiry quotes (bid / ask / mid, strike, type) within BAND_FRAC of
              the 15:30 index level and writes
              results/strategy_variations/legs_1530_1600.parquet.
  (default)   scores every variation from that small file on the straddle
              trade's 866 days: sign(s) with the per-bar ridge on the
              live-feasible set (research scorer, 16:00-bar recalibration) and
              with the block-diagonal ridge (the paper's forecast, deck scorer),
              plus always short; fills mid and crossed; two return frames;
              paired block-bootstrap intervals of each variation minus the
              straddle; the bid-ask cost line; a figure; the LaTeX table.

Usage:
    python experiments/strategy_variations_1530.py --extract
    python experiments/strategy_variations_1530.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "experiments", ROOT / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402

CHAIN = ROOT / "data" / "spxw_chain.parquet"
DECK_DIR = ROOT / "results" / "atm_straddle_0dte_1530"
OUT = ROOT / "results" / "strategy_variations"
LEGS = OUT / "legs_1530_1600.parquet"
TEX = ROOT / "writeup" / "generated" / "strategy_variations.tex"
RESEARCH_ARM = (
    ROOT
    / "results"
    / "linear_subsection"
    / "arms_carc"
    / "live_feasible"
    / "ridge"
    / "tw2000"
    / "results_bar1600.csv"
)
NY = "America/New_York"

# Strikes kept around the 15:30 index level. The farthest leg any variation
# needs is the 2 % strangle, or the one-strike strangle plus the 50-point wing
# (50 points is 2.3 % of the index at its 2020 low); 5 % holds both with room.
# Strikes inside the band are complete, so a leg the band cut off would show up
# as an unpriced day in the coverage count, and the scorer prints the farthest
# chosen leg's distance from the index.
BAND_FRAC = 0.05
STRANGLE_OTM = (0.005, 0.01, 0.02)  # the moneyness strangles named in the task
# Wing widths in index points: 25 and 50 are the earlier condor lab's widths;
# 10 is the narrowest width of the earlier ladder.
WIDTHS = (10.0, 25.0, 50.0)
# Paired bootstrap: the rule tables' convention (block 21 trading days, 2000 draws, seed 0).
PAIR_BLOCK, PAIR_B, PAIR_SEED = 21, 2000, 0
TAIL_DAYS = int(round(asl.PERIODS_PER_YEAR / 12))  # one trading month: 21 days
# The gates compare Sharpe ratios rebuilt from the extracted float32 quotes with the
# published ones; summation order differs, so equality is to 1e-6, not to the bit.
GATE_TOL = 1e-6
ANN = float(np.sqrt(asl.PERIODS_PER_YEAR))

RULE_RESEARCH = "sign(s), per-bar ridge (live-feasible), research scorer"
RULE_DECK = "sign(s), block-diagonal ridge (paper), deck scorer"
RULE_SHORT = "always short"
RULES = (RULE_RESEARCH, RULE_DECK, RULE_SHORT)
FILLS = ("mid", "crossed")
FRAMES = ("per straddle premium", "own notional")


# ----------------------------------------------------------------------------
# step 1: extraction (chain-sized)
# ----------------------------------------------------------------------------
def extract() -> None:
    cols = [
        "expiration",
        "timestamp",
        "strike",
        "cp",
        "bid",
        "ask",
        "mid",
        "underlying_price",
        "impl_volatility",
        "hours_to_expiration",
    ]
    st = os.stat(CHAIN)
    print(f"source {CHAIN.name}: {st.st_size} bytes, mtime_ns {st.st_mtime_ns}")
    ts = pd.to_datetime(
        pd.read_parquet(CHAIN, columns=["timestamp"])["timestamp"], utc=True
    )
    uts = pd.DatetimeIndex(ts.unique())
    del ts
    uet = uts.tz_convert(NY)
    keep = uts[
        ((uet.hour == 15) & (uet.minute == 30)) | ((uet.hour == 16) & (uet.minute == 0))
    ]
    print(f"unique stamps {len(uts)}; 15:30/16:00 ET stamps kept {len(keep)}")
    ch = pd.read_parquet(CHAIN, columns=cols, filters=[("timestamp", "in", list(keep))])
    print(f"rows at the two stamps {len(ch):,}")
    ch["timestamp"] = pd.to_datetime(ch["timestamp"], utc=True)
    et = ch["timestamp"].dt.tz_convert(NY)
    ch["hhmm"] = np.where(et.dt.hour == 15, "15:30", "16:00")
    ch["day"] = et.dt.tz_localize(None).dt.normalize()
    exp = pd.to_datetime(ch["expiration"])
    if getattr(exp.dt, "tz", None) is not None:
        exp = exp.dt.tz_convert(NY).dt.tz_localize(None)
    ch = ch[exp.dt.normalize() == ch["day"]].copy()
    print(f"same-day-expiry rows {len(ch):,} on {ch['day'].nunique()} days")
    ch["cp"] = ch["cp"].astype(str).str.upper().str[0]
    half = asl.early_close_days(ch.assign(et=et.loc[ch.index]))
    print(f"half sessions flagged (kept in the file, never scored): {len(half)}")
    s30 = (
        ch[ch["hhmm"] == "15:30"]
        .dropna(subset=["underlying_price"])
        .groupby("day")["underlying_price"]
        .first()
    )
    ch["S_1530"] = ch["day"].map(s30).astype(float)
    band = (ch["strike"].astype(float) - ch["S_1530"]).abs() <= BAND_FRAC * ch["S_1530"]
    ch = ch[band].copy()
    ch["half_session"] = ch["day"].isin(half)
    out = ch[
        [
            "day",
            "hhmm",
            "timestamp",
            "strike",
            "cp",
            "bid",
            "ask",
            "mid",
            "underlying_price",
            "impl_volatility",
            "hours_to_expiration",
            "S_1530",
            "half_session",
        ]
    ].sort_values(["day", "hhmm", "cp", "strike"])
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LEGS, index=False)
    print(
        f"wrote {LEGS} rows {len(out):,} days {out['day'].nunique()} "
        f"({LEGS.stat().st_size / 1e6:.2f} MB); strikes within {BAND_FRAC:.0%} of the 15:30 index"
    )


# ----------------------------------------------------------------------------
# step 2: scoring
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Variation:
    name: str  # row label
    family: str
    width: str  # "w10", "0.5%", "" ...
    kind: str  # "premium", "defined", "ratio", "combo"


def build_variations() -> list[Variation]:
    v = [Variation("straddle", "straddle", "", "premium")]
    v.append(Variation("strangle, one strike wide", "strangle", "1 strike", "premium"))
    for x in STRANGLE_OTM:
        v.append(
            Variation(
                f"strangle {100 * x:g}% OTM", "strangle", f"{100 * x:g}%", "premium"
            )
        )
    for fam, kind in (
        ("iron butterfly", "defined"),
        ("iron condor", "defined"),
        ("put vertical", "defined"),
        ("call vertical", "defined"),
        ("call butterfly", "defined"),
        ("put butterfly", "defined"),
        ("put ratio 1x2", "ratio"),
        ("call ratio 1x2", "ratio"),
        ("straddle + wing hedge", "combo"),
    ):
        for w in WIDTHS:
            v.append(Variation(f"{fam}, w{int(w)}", fam, f"w{int(w)}", kind))
    return v


class DayBook:
    """Live 15:30 quotes of one day, by type, sorted by strike."""

    def __init__(self, q: pd.DataFrame) -> None:
        self.k: dict[str, np.ndarray] = {}
        self.px: dict[str, np.ndarray] = {}
        for cp in ("C", "P"):
            g = q[q["cp"] == cp].sort_values("strike")
            self.k[cp] = g["strike"].to_numpy(float)
            self.px[cp] = g[["mid", "bid", "ask"]].to_numpy(float)

    def at_least(self, cp: str, x: float) -> float:
        k = self.k[cp]
        i = int(np.searchsorted(k, x - 1e-9, side="left"))
        return float(k[i]) if i < len(k) else np.nan

    def at_most(self, cp: str, x: float) -> float:
        k = self.k[cp]
        i = int(np.searchsorted(k, x + 1e-9, side="right")) - 1
        return float(k[i]) if i >= 0 else np.nan

    def exact(self, cp: str, x: float) -> float:
        k = self.k[cp]
        i = int(np.searchsorted(k, x - 1e-9, side="left"))
        return float(k[i]) if i < len(k) and abs(k[i] - x) < 1e-6 else np.nan

    def quote(self, cp: str, K: float) -> np.ndarray:
        k = self.k[cp]
        i = int(np.searchsorted(k, K - 1e-9, side="left"))
        assert i < len(k) and abs(k[i] - K) < 1e-6, (cp, K)
        return self.px[cp][i]


Leg = tuple[str, float, float]  # (cp, strike, quantity in the buy-side form L)


def legs_for(
    v: Variation, b: DayBook, S: float, K_c: float, K_p: float
) -> tuple[list[Leg], list[Leg]] | None:
    """(L_plus, L_minus): the structure held long on buying days and the one sold on selling days."""
    straddle: list[Leg] = [("C", K_c, 1.0), ("P", K_p, 1.0)]
    w = float(v.width[1:]) if v.width.startswith("w") else np.nan
    if v.family == "straddle":
        L = straddle
    elif v.name == "strangle, one strike wide":
        L = [
            ("C", b.at_least("C", K_c + 1e-6), 1.0),
            ("P", b.at_most("P", K_p - 1e-6), 1.0),
        ]
    elif v.family == "strangle":
        x = float(v.width.rstrip("%")) / 100.0
        L = [
            ("C", b.at_least("C", S * (1 + x)), 1.0),
            ("P", b.at_most("P", S * (1 - x)), 1.0),
        ]
    elif v.family in ("iron butterfly", "straddle + wing hedge"):
        L = straddle + [
            ("C", b.at_least("C", K_c + w), -1.0),
            ("P", b.at_most("P", K_p - w), -1.0),
        ]
    elif v.family == "iron condor":
        kc1, kp1 = b.at_least("C", K_c + 1e-6), b.at_most("P", K_p - 1e-6)
        L = [
            ("C", kc1, 1.0),
            ("P", kp1, 1.0),
            ("C", b.at_least("C", kc1 + w), -1.0),
            ("P", b.at_most("P", kp1 - w), -1.0),
        ]
    elif v.family == "put vertical":
        L = [("P", K_p, 1.0), ("P", b.at_most("P", K_p - w), -1.0)]
    elif v.family == "call vertical":
        L = [("C", K_c, 1.0), ("C", b.at_least("C", K_c + w), -1.0)]
    elif v.family in ("call butterfly", "put butterfly"):
        cp = v.family[0].upper()
        k0 = K_c if (K_c - S) <= (S - K_p) else K_p
        L = [
            (cp, b.exact(cp, k0 - w), -1.0),
            (cp, b.exact(cp, k0), 2.0),
            (cp, b.exact(cp, k0 + w), -1.0),
        ]
    elif v.family == "put ratio 1x2":
        L = [("P", K_p, -1.0), ("P", b.at_most("P", K_p - w), 2.0)]
    elif v.family == "call ratio 1x2":
        L = [("C", K_c, -1.0), ("C", b.at_least("C", K_c + w), 2.0)]
    else:  # pragma: no cover
        raise ValueError(v)
    if any(not np.isfinite(k) for _, k, _ in L):
        return None
    if v.family == "straddle + wing hedge":
        return straddle, L
    return L, L


def price_structure(L: list[Leg], b: DayBook, S_close: float) -> dict[str, Any]:
    mid = bid_side = ask_side = pay = 0.0
    zero_bid_sold_long = zero_bid_sold_short = 0
    for cp, k, n in L:
        m, bd, ak = b.quote(cp, k)
        mid += n * m
        # buying L: its + legs at the ask, its - legs at the bid
        ask_side += n * (ak if n > 0 else bd)
        # selling L: its + legs at the bid, its - legs at the ask
        bid_side += n * (bd if n > 0 else ak)
        pay += n * (max(S_close - k, 0.0) if cp == "C" else max(k - S_close, 0.0))
        if n < 0 and not bd > 0:
            zero_bid_sold_long += 1  # a leg sold when L is bought has no bid
        if n > 0 and not bd > 0:
            zero_bid_sold_short += 1  # a leg sold when L is sold has no bid
    gaps = []
    for cp in ("C", "P"):
        ks = sorted({k for c, k, _ in L if c == cp})
        if len(ks) >= 2:
            gaps.append(ks[-1] - ks[0] if len(ks) == 2 else max(np.diff(ks)))
    return {
        "mid": mid,
        "cost_buy": ask_side,
        "proceeds_sell": bid_side,
        "payoff": pay,
        "gap": max(gaps) if gaps else np.nan,
        "zb_buy": zero_bid_sold_long,
        "zb_sell": zero_bid_sold_short,
        "legs": ";".join(
            f"{'+' if n > 0 else '-'}{abs(n):g}{cp}{k:g}" for cp, k, n in L
        ),
    }


def research_pred_1600() -> pd.Series:
    """The research scorer's forecast of the 15:30-16:00 bar (causal per-clock term), 16:00-stamped."""
    from compare_mfiv_harlag import _prep

    r = _prep(RESEARCH_ARM)
    assert r is not None, RESEARCH_ARM
    f = r["pred_clock"]
    return f[f.index.strftime("%H:%M") == "16:00"]


def load_research_forecast(days: pd.DatetimeIndex) -> pd.Series:
    f = research_pred_1600().copy()
    f.index = f.index.normalize()
    return f.reindex(days)


def sharpe(x: np.ndarray) -> float:
    sd = float(np.std(x, ddof=1))
    return float(np.mean(x) / sd * ANN) if sd > 0 else np.nan


def max_dd(x: np.ndarray) -> float:
    cum = np.cumsum(x)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    return float((cum - peak).min())


_IDX: dict[int, np.ndarray] = {}


def boot_idx(n: int) -> np.ndarray:
    if n not in _IDX:
        _IDX[n] = asl.circular_block_bootstrap_idx(
            np.random.default_rng(PAIR_SEED), n, PAIR_BLOCK, PAIR_B
        )
    return _IDX[n]


def _draw_sharpe(d: np.ndarray) -> np.ndarray:
    # a draw whose days all carry the same return has no Sharpe ratio: NaN, left out of the percentiles
    sd = d.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, d.mean(axis=1) / sd * ANN, np.nan)


def boot_sharpe(x: np.ndarray) -> np.ndarray:
    return _draw_sharpe(x[boot_idx(len(x))])


def stats_row(x: np.ndarray, q: np.ndarray) -> dict[str, float]:
    n = len(x)
    tot = float(x.sum())
    top = float(np.sort(x)[::-1][:TAIL_DAYS].sum())
    bs = boot_sharpe(x)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {
        "n": n,
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)),
        "Sharpe": sharpe(x),
        "Sharpe_lo": float(lo),
        "Sharpe_hi": float(hi),
        "t": float(x.mean() / x.std(ddof=1) * np.sqrt(n)),
        "hit_rate": float((x > 0).mean()),
        "max_drawdown": max_dd(x),
        "worst_day": float(x.min()),
        "best_day": float(x.max()),
        "top21_share": top / tot if tot != 0 else np.nan,
        # what the other days earn once the best month of days is taken out
        "mean_ex_top21": float(np.sort(x)[::-1][TAIL_DAYS:].mean()),
        "pct_buy": float(100 * (q > 0).mean()),
    }


def paired_row(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    idx = boot_idx(len(a))
    da, db = a[idx], b[idx]
    ds = _draw_sharpe(da) - _draw_sharpe(db)
    dm = (da - db).mean(axis=1)
    t_hac, _ = asl.newey_west_t(a - b)
    s_lo, s_hi = np.nanpercentile(ds, [2.5, 97.5])
    m_lo, m_hi = np.nanpercentile(dm, [2.5, 97.5])
    return {
        "dSharpe": sharpe(a) - sharpe(b),
        "dSharpe_lo": float(s_lo),
        "dSharpe_hi": float(s_hi),
        "dmean": float((a - b).mean()),
        "dmean_lo": float(m_lo),
        "dmean_hi": float(m_hi),
        "t_hac_dmean": t_hac,
    }


def score() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    legs = pd.read_parquet(LEGS)
    print(f"legs file {LEGS.name}: {len(legs):,} rows, {legs['day'].nunique()} days")
    deck = pd.read_parquet(DECK_DIR / "daily_blk2.parquet").sort_index()
    days = pd.DatetimeIndex(pd.to_datetime(deck.index)).normalize()
    deck.index = days
    print(
        f"straddle trade's scored days: {len(days)} ({days.min().date()} .. {days.max().date()})"
    )

    q30 = legs[(legs["hhmm"] == "15:30") & np.isfinite(legs["mid"]) & (legs["mid"] > 0)]
    q30 = q30[q30["day"].isin(days)]
    q16 = legs[legs["hhmm"] == "16:00"]
    tape16 = (
        q16.dropna(subset=["underlying_price"])
        .groupby("day")["underlying_price"]
        .first()
    )
    # gate: the 16:00 tape and the straddle trade's copy of it agree
    gap16 = (tape16.reindex(days) - deck["S_1600_tape"]).abs()
    print(
        f"gate: 16:00 tape vs the straddle trade's S_1600_tape, max |diff| {float(gap16.max()):.3g} on {int(gap16.notna().sum())} days"
    )
    assert float(gap16.max()) < 1e-6
    # vendor mid vs (bid + ask) / 2 on the live legs
    qm = asl.quote_mid(q30["bid"], q30["ask"]).to_numpy()
    print(
        f"vendor mid vs (bid+ask)/2 on live 15:30 legs: max |diff| {float(np.nanmax(np.abs(qm - q30['mid'].to_numpy()))):.3g}"
    )

    books = {d: DayBook(g) for d, g in q30.groupby("day")}
    assert set(days) <= set(books), "a scored day has no live 15:30 quotes"

    # ---- positions -------------------------------------------------------
    f_res = load_research_forecast(days)
    assert f_res.notna().all(), "research forecast missing on a scored day"
    iv = deck["iv_var"].to_numpy(float)
    pos = {
        RULE_RESEARCH: np.where(f_res.to_numpy(float) > iv, 1.0, -1.0),
        RULE_DECK: deck["pos"].to_numpy(float),
        RULE_SHORT: -np.ones(len(days)),
    }
    ratio = f_res.to_numpy(float) / iv
    print(
        "per-bar ridge forecast: only its sign against the implied variance enters sign(s), so no "
        f"value is clipped; days with forecast > 10x the implied variance: {int((ratio > 10).sum())} "
        f"of {len(days)} (max ratio {float(ratio.max()):.1f}); buying days {int((pos[RULE_RESEARCH] > 0).sum())}"
    )

    # ---- structures, day by day -------------------------------------------
    variations = build_variations()
    recs = []
    for d in days:
        b = books[d]
        r = deck.loc[d]
        S, K_c, K_p, S_close = (
            float(r["S"]),
            float(r["K_c"]),
            float(r["K_p"]),
            float(r["S_close"]),
        )
        # gate: the straddle's legs from the extracted file are the trade's legs
        assert b.at_least("C", S) == K_c and b.at_most("P", S) == K_p, d
        for v in variations:
            lp = legs_for(v, b, S, K_c, K_p)
            if lp is None:
                continue
            Lp, Lm = lp
            pp = price_structure(Lp, b, S_close)
            pm = pp if Lm is Lp else price_structure(Lm, b, S_close)
            recs.append(
                {
                    "day": d,
                    "variation": v.name,
                    "family": v.family,
                    "width": v.width,
                    "kind": v.kind,
                    "legs_plus": pp["legs"],
                    "legs_minus": pm["legs"],
                    "mid_plus": pp["mid"],
                    "cost_buy_plus": pp["cost_buy"],
                    "payoff_plus": pp["payoff"],
                    "gap_plus": pp["gap"],
                    "zb_buy_plus": pp["zb_buy"],
                    "mid_minus": pm["mid"],
                    "proceeds_sell_minus": pm["proceeds_sell"],
                    "payoff_minus": pm["payoff"],
                    "gap_minus": pm["gap"],
                    "zb_sell_minus": pm["zb_sell"],
                    "S": S,
                    "S_close": S_close,
                    "S_1600_tape": float(r["S_1600_tape"]),
                    "strad_mid": float(r["entry"]),
                    "strad_ask": float(r["ask_c"] + r["ask_p"]),
                    "strad_bid": float(r["bid_c"] + r["bid_p"]),
                }
            )
    daily = pd.DataFrame(recs)
    daily["day"] = pd.to_datetime(daily["day"])
    # gate: the straddle's structure reproduces the trade's entry, settlement and return
    st = daily[daily["variation"] == "straddle"].set_index("day").reindex(days)
    assert np.allclose(st["mid_plus"], deck["entry"]) and np.allclose(
        st["payoff_plus"], deck["exit"]
    )
    print(
        "gate: straddle entry and settlement reproduce the straddle trade on all",
        len(st),
        "days",
    )
    far = max(
        abs(float(re.split("[CP]", t)[1]) / s0 - 1.0)
        for lp, lm, s0 in zip(daily["legs_plus"], daily["legs_minus"], daily["S"])
        for t in (lp + ";" + lm).split(";")
    )
    print(
        f"farthest chosen leg from the 15:30 index: {100 * far:.2f}% (extraction band {100 * BAND_FRAC:.0f}%)"
    )
    cov = daily.groupby("variation", sort=False).size()
    print(
        "days priced per variation (of",
        len(days),
        "):",
        cov[cov < len(days)].to_dict() or "all priced on every day",
    )

    # ---- per-day returns -----------------------------------------------------
    rows, prs, rets = [], [], {}
    pos_s = {k: pd.Series(v, index=days) for k, v in pos.items()}
    for v in variations:
        dv = daily[daily["variation"] == v.name].set_index("day").sort_index()
        dd = dv.index
        for rule in RULES:
            q = pos_s[rule].reindex(dd).to_numpy(float)
            buy = q > 0
            pnl_mid = np.where(
                buy,
                dv["payoff_plus"] - dv["mid_plus"],
                dv["mid_minus"] - dv["payoff_minus"],
            )
            pnl_x = np.where(
                buy,
                dv["payoff_plus"] - dv["cost_buy_plus"],
                dv["proceeds_sell_minus"] - dv["payoff_minus"],
            )
            den_mid = dv["strad_mid"].to_numpy(float)
            den_x = np.where(buy, dv["strad_ask"], dv["strad_bid"])
            own_mid_den = np.where(buy, dv["mid_plus"], dv["mid_minus"])
            own_x_den = np.where(buy, dv["cost_buy_plus"], dv["proceeds_sell_minus"])
            gap = np.where(buy, dv["gap_plus"], dv["gap_minus"])
            for fill, pnl, den, own_den in (
                ("mid", pnl_mid, den_mid, own_mid_den),
                ("crossed", pnl_x, den_x, own_x_den),
            ):
                r_common = pnl / den
                if v.kind == "premium":
                    r_own = np.where(
                        own_den > 0,
                        pnl / np.where(own_den > 0, own_den, np.nan),
                        np.nan,
                    )
                elif v.kind == "defined":
                    r_own = pnl / gap
                    if fill == "mid" and rule != RULE_SHORT:
                        n_floor = int((r_own < -1.0 - 1e-9).sum())
                        if n_floor:
                            print(
                                f"  {v.name}, {rule}: {n_floor} day(s) below the -1 capital-at-risk floor at mid"
                            )
                else:
                    r_own = np.full(len(pnl), np.nan)
                for frame, rr in (
                    ("per straddle premium", r_common),
                    ("own notional", r_own),
                ):
                    s = pd.Series(rr, index=dd)
                    rets[(v.name, rule, fill, frame)] = s
                    ok = np.isfinite(rr)
                    if ok.sum() < 2:
                        continue
                    rows.append(
                        {
                            "variation": v.name,
                            "family": v.family,
                            "width": v.width,
                            "rule": rule,
                            "fill": fill,
                            "frame": frame,
                            **stats_row(rr[ok], q[ok]),
                        }
                    )
    summ = pd.DataFrame(rows)

    # gates: the straddle rows are the published numbers of the rule tables and the research scorer
    def sh(rule, fill, var="straddle", frame="per straddle premium"):
        m = (
            (summ["variation"] == var)
            & (summ["rule"] == rule)
            & (summ["fill"] == fill)
            & (summ["frame"] == frame)
        )
        return float(summ.loc[m, "Sharpe"].iloc[0])

    from score_linear_subsection import trade_1530

    ref = trade_1530(research_pred_1600())
    print(
        f"gate: straddle sign(s) Sharpe, research scorer mid {sh(RULE_RESEARCH, 'mid'):.4f} (scorer {ref['Sharpe_mid']:.4f}), "
        f"crossed {sh(RULE_RESEARCH, 'crossed'):.4f} (scorer {ref['Sharpe_crossed']:.4f}); deck scorer mid "
        f"{sh(RULE_DECK, 'mid'):.4f} crossed {sh(RULE_DECK, 'crossed'):.4f}; always short mid {sh(RULE_SHORT, 'mid'):.4f} "
        f"crossed {sh(RULE_SHORT, 'crossed'):.4f}"
    )
    assert abs(sh(RULE_RESEARCH, "mid") - ref["Sharpe_mid"]) < GATE_TOL
    assert abs(sh(RULE_RESEARCH, "crossed") - ref["Sharpe_crossed"]) < GATE_TOL
    xr = asl.crossed_premium_return(
        deck["pos"],
        deck["exit"],
        deck["bid_c"] + deck["bid_p"],
        deck["ask_c"] + deck["ask_p"],
    )
    assert abs(sh(RULE_DECK, "crossed") - sharpe(xr.to_numpy())) < GATE_TOL
    assert (
        abs(sh(RULE_DECK, "mid") - sharpe((deck["pos"] * deck["R"]).to_numpy()))
        < GATE_TOL
    )

    # ---- paired: variation minus straddle, same rule, fill, frame, days -------
    for (vn, rule, fill, frame), s in rets.items():
        if vn == "straddle":
            continue
        base = rets[("straddle", rule, fill, frame)]
        j = pd.concat([s, base], axis=1, keys=["v", "b"]).dropna()
        if len(j) < 2:
            continue
        v0 = next(v for v in variations if v.name == vn)
        prs.append(
            {
                "variation": vn,
                "family": v0.family,
                "width": v0.width,
                "rule": rule,
                "fill": fill,
                "frame": frame,
                "n": len(j),
                "Sharpe_variation": sharpe(j["v"].to_numpy()),
                "Sharpe_straddle_same_days": sharpe(j["b"].to_numpy()),
                **paired_row(j["v"].to_numpy(), j["b"].to_numpy()),
            }
        )
    paired = pd.DataFrame(prs)

    # ---- the cost line -----------------------------------------------------------
    cost_rows = []
    for v in variations:
        dv = daily[daily["variation"] == v.name].set_index("day").sort_index()
        prem_p = dv["mid_plus"].abs()
        prem_m = dv["mid_minus"].abs()
        buy_cost = (
            dv["cost_buy_plus"] - dv["mid_plus"]
        )  # points paid over mid to buy L_plus
        sell_cost = (
            dv["mid_minus"] - dv["proceeds_sell_minus"]
        )  # points given up to sell L_minus
        rec = {
            "variation": v.name,
            "family": v.family,
            "width": v.width,
            "days_priced": len(dv),
            "median_mid_premium_bought_pts": float(dv["mid_plus"].median()),
            "median_mid_premium_sold_pts": float(dv["mid_minus"].median()),
            "median_cost_buy_pts": float(buy_cost.median()),
            "median_cost_sell_pts": float(sell_cost.median()),
            "median_cost_buy_share_of_own_premium": float(
                (buy_cost / prem_p).replace([np.inf], np.nan).median()
            ),
            "median_cost_sell_share_of_own_premium": float(
                (sell_cost / prem_m).replace([np.inf], np.nan).median()
            ),
            "median_cost_buy_share_of_straddle_premium": float(
                (buy_cost / dv["strad_mid"]).median()
            ),
            "median_cost_sell_share_of_straddle_premium": float(
                (sell_cost / dv["strad_mid"]).median()
            ),
            "days_a_sold_leg_has_no_bid_when_bought": int(
                (dv["zb_buy_plus"] > 0).sum()
            ),
            "days_a_sold_leg_has_no_bid_when_sold": int(
                (dv["zb_sell_minus"] > 0).sum()
            ),
            "days_L_mid_price_nonpositive": int((dv["mid_minus"] <= 0).sum()),
        }
        for rule in RULES:
            m = (
                (summ["variation"] == v.name)
                & (summ["rule"] == rule)
                & (summ["frame"] == "per straddle premium")
            )
            a = summ[m].set_index("fill")
            if {"mid", "crossed"} <= set(a.index):
                tag = {
                    RULE_RESEARCH: "research",
                    RULE_DECK: "deck",
                    RULE_SHORT: "short",
                }[rule]
                rec[f"mean_erosion_{tag}"] = float(
                    a.loc["mid", "mean"] - a.loc["crossed", "mean"]
                )
                rec[f"cost_over_mid_mean_{tag}"] = (
                    float(
                        (a.loc["mid", "mean"] - a.loc["crossed", "mean"])
                        / a.loc["mid", "mean"]
                    )
                    if a.loc["mid", "mean"] != 0
                    else np.nan
                )
                rec[f"Sharpe_erosion_{tag}"] = float(
                    a.loc["mid", "Sharpe"] - a.loc["crossed", "Sharpe"]
                )
        cost_rows.append(rec)
    cost = pd.DataFrame(cost_rows)

    # the tape check: days on which the official close and the 16:00 tape settle a leg on different sides
    flips = []
    for v in variations:
        dv = daily[daily["variation"] == v.name]
        n_flip = 0
        n_out = 0
        for _, r in dv.iterrows():
            ks = [
                float(re.split("[CP]", t)[1])
                for t in (r["legs_plus"] + ";" + r["legs_minus"]).split(";")
            ]
            if np.isfinite(r["S_1600_tape"]) and any(
                (r["S_close"] > k) != (r["S_1600_tape"] > k) for k in ks
            ):
                n_flip += 1
            # the close lands beyond the structure's outermost strike: a wing's cap binds
            n_out += int(r["S_close"] > max(ks) or r["S_close"] < min(ks))
        flips.append(
            {
                "variation": v.name,
                "days_close_and_tape_disagree_on_a_leg": n_flip,
                "share_close_beyond_outermost_strike": n_out / max(len(dv), 1),
                "days": len(dv),
            }
        )
    flips_df = pd.DataFrame(flips)
    cost = cost.merge(flips_df, on="variation", how="left")

    # ---- write -------------------------------------------------------------------
    catalog = (
        daily.groupby(["variation", "family", "width", "kind"], sort=False)
        .agg(
            days=("day", "size"),
            example_legs_plus=("legs_plus", "last"),
            example_legs_minus=("legs_minus", "last"),
        )
        .reset_index()
    )
    catalog.to_csv(OUT / "variation_catalog.csv", index=False)
    summ.to_csv(OUT / "summary_all.csv", index=False)
    paired.to_csv(OUT / "paired_vs_straddle.csv", index=False)
    cost.to_csv(OUT / "cost_line.csv", index=False)
    daily.to_parquet(OUT / "variations_daily.parquet", index=False)
    rets_df = pd.DataFrame(
        {"|".join(k): v for k, v in rets.items() if k[3] == "per straddle premium"}
    )
    rets_df.index.name = "day"
    rets_df.to_parquet(OUT / "returns_daily_per_straddle_premium.parquet")
    head = summ[summ["frame"] == "per straddle premium"].merge(
        paired[paired["frame"] == "per straddle premium"][
            [
                "variation",
                "rule",
                "fill",
                "dSharpe",
                "dSharpe_lo",
                "dSharpe_hi",
                "dmean",
                "dmean_lo",
                "dmean_hi",
                "t_hac_dmean",
            ]
        ],
        on=["variation", "rule", "fill"],
        how="left",
    )
    head.to_csv(OUT / "headline_per_straddle_premium.csv", index=False)
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.csv"))))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    show = [
        "variation",
        "fill",
        "n",
        "mean",
        "Sharpe",
        "Sharpe_lo",
        "Sharpe_hi",
        "hit_rate",
        "max_drawdown",
        "worst_day",
        "top21_share",
        "mean_ex_top21",
        "dSharpe",
        "dSharpe_lo",
        "dSharpe_hi",
        "dmean",
        "dmean_lo",
        "dmean_hi",
    ]
    for rule in RULES:
        print(f"\n=== {rule}; per straddle premium ===")
        print(head[head["rule"] == rule][show].round(3).to_string(index=False))
    print(
        "\n=== own notional frame (premium / capital at risk), sign(s) per-bar ridge ==="
    )
    own = summ[(summ["frame"] == "own notional") & (summ["rule"] == RULE_RESEARCH)]
    print(
        own[
            [
                "variation",
                "fill",
                "n",
                "mean",
                "Sharpe",
                "Sharpe_lo",
                "Sharpe_hi",
                "hit_rate",
                "worst_day",
                "top21_share",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print("\n=== cost line ===")
    print(cost.round(3).to_string(index=False))

    make_figure(head)
    make_tex(head, cost)


# the figure's common Sharpe axis; intervals running past it end in an arrow at the edge
FIG_XLIM = (-4.0, 4.0)


def make_figure(head: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"mid": "#2a78d6", "crossed": "#eb6834"}
    ink, grid = "#52514e", "#e4e3df"
    order = list(dict.fromkeys(head["variation"]))
    fam = head.drop_duplicates("variation").set_index("variation")["family"]
    y = np.arange(len(order))[::-1].astype(float)
    ypos = dict(zip(order, y))
    lo_x, hi_x = FIG_XLIM
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 0.27 * len(order) + 2.0), sharey=True, sharex=True
    )
    for ax, rule in zip(axes, RULES):
        h = head[head["rule"] == rule]
        for k, fill in enumerate(FILLS):
            g = h[h["fill"] == fill]
            yy = np.array([ypos[v] for v in g["variation"]]) + (
                0.17 if k == 0 else -0.17
            )
            lo = g["Sharpe_lo"].to_numpy(float)
            hi = g["Sharpe_hi"].to_numpy(float)
            ax.hlines(
                yy,
                np.clip(lo, lo_x, hi_x),
                np.clip(hi, lo_x, hi_x),
                color=colors[fill],
                lw=1.4,
            )
            for yv, a, b in zip(yy, lo, hi):
                if b > hi_x:
                    ax.plot(hi_x, yv, ">", ms=5, color=colors[fill], clip_on=False)
                if a < lo_x:
                    ax.plot(lo_x, yv, "<", ms=5, color=colors[fill], clip_on=False)
            ax.plot(
                np.clip(g["Sharpe"].to_numpy(float), lo_x, hi_x),
                yy,
                "o",
                ms=5,
                color=colors[fill],
                mfc=colors[fill] if fill == "mid" else "white",
                mew=1.5,
                label=f"{fill} fill, 95% interval",
            )
            st = g[g["variation"] == "straddle"]
            ax.axvline(float(st["Sharpe"].iloc[0]), color=colors[fill], lw=0.8, ls="--")
        ax.axvline(0.0, color=ink, lw=0.8)
        for a, b in zip(order[:-1], order[1:]):
            if fam[a] != fam[b]:
                ax.axhline((ypos[a] + ypos[b]) / 2, color=grid, lw=0.8)
        ax.set_xlim(lo_x, hi_x)
        title = rule.replace(
            ", research scorer", "\n(research scorer: 16:00-bar recalibration)"
        )
        title = title.replace(", deck scorer", "\n(deck scorer, as in the paper)")
        if rule == RULE_SHORT:
            title += "\n(no forecast)"
        ax.set_title(title, fontsize=9, color="#0b0b0b")
        ax.grid(axis="x", color=grid, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("annualized Sharpe ratio, return per straddle premium", color=ink)
        ax.tick_params(colors=ink)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order, fontsize=8)
    axes[0].set_ylim(y.min() - 0.8, y.max() + 0.8)
    axes[0].legend(fontsize=8, loc="lower left", frameon=False)
    fig.suptitle(
        "Strategy variations of the 15:30 last-30-min trade, same 866 days: Sharpe ratio by variation "
        "and width, mid vs crossed fills\n(dashed: the straddle under the same rule and fill; arrows: "
        f"an interval that runs past {hi_x:g} or below {lo_x:g})",
        fontsize=10,
    )
    fig.tight_layout()
    p = OUT / "sharpe_by_variation.png"
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", p)


def _f(x: float, nd: int = 2) -> str:
    if not np.isfinite(x):
        return ""
    return f"{x:.{nd}f}" if x >= 0 else f"$-${abs(x):.{nd}f}"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}\\%"


def _tex_label(vn: str) -> str:
    return vn.replace("%", r"\%").replace("1x2", r"$1\times2$")


# The written-down variations (VARIATIONS.md), one row each: the buy-side form L,
# the structure held when s < 0 (-L), the own notional, and the purpose.
TEX_DEFS = [
    (
        "straddle (baseline)",
        r"$+C(K_c)+P(K_p)$",
        "short straddle",
        "premium",
        "convexity both ways at the money",
    ),
    (
        "strangle, one strike wide",
        r"$+C(K_c^{+})+P(K_p^{-})$",
        "short strangle",
        "premium",
        "all time value; cheaper, less gamma at the money",
    ),
    (
        r"strangle $x$ OTM, $x\in\{0.5,1,2\}\%$",
        r"$+C(\ge S(1+x))+P(\le S(1-x))$",
        "short strangle",
        "premium",
        "tail-only exposure; sold, a small premium for the tails",
    ),
    (
        r"iron butterfly, $w$",
        r"$+C(K_c)+P(K_p)-C(\ge K_c+w)-P(\le K_p-w)$",
        "short straddle + long wings (credit)",
        "larger wing gap",
        "tail cap on the selling side",
    ),
    (
        r"iron condor, $w$",
        r"body: the strangle one strike wide; wings $w$ beyond it",
        "short strangle + long wings (credit)",
        "larger wing gap",
        "cheaper body, tail cap",
    ),
    (
        r"put vertical, $w$",
        r"$+P(K_p)-P(\le K_p-w)$",
        "credit put vertical",
        "wing gap",
        "premium from the put side alone, capped",
    ),
    (
        r"call vertical, $w$",
        r"$+C(K_c)-C(\ge K_c+w)$",
        "credit call vertical",
        "wing gap",
        "premium from the call side alone, capped",
    ),
    (
        r"call (put) butterfly, $w$",
        r"short butterfly $-C(K_0-w)+2C(K_0)-C(K_0+w)$; $K_0$ the straddle strike nearer $S$",
        r"long butterfly (debit; pays on a pin at $K_0$)",
        r"$w$",
        "defined-risk selling with a pin payoff; one wing in the money",
    ),
    (
        r"put (call) ratio $1\times2$, $w$",
        r"backspread $-P(K_p)+2P(\le K_p-w)$",
        r"front ratio $+P(K_p)-2P(\le K_p-w)$, one leg uncovered",
        "none",
        "tail convexity when bought, tail premium when sold",
    ),
    (
        r"straddle + wing hedge, $w$",
        "the straddle",
        "the short iron butterfly",
        "none (two units)",
        "cap the selling side's tail, keep the buying side's convexity",
    ),
]


def _cost_share(cst: pd.DataFrame, v: str, of: str) -> float:
    return float(
        np.mean(
            [
                cst.loc[v, f"median_cost_buy_share_of_{of}_premium"],
                cst.loc[v, f"median_cost_sell_share_of_{of}_premium"],
            ]
        )
    )


def _results_table(
    head: pd.DataFrame, cost: pd.DataFrame, rule: str, caption: str, label: str
) -> list[str]:
    h = head[head["rule"] == rule]
    out = [
        r"\begingroup\footnotesize\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{lrrlrrlrr}",
        r"\caption{" + caption + r"}\label{" + label + r"}\\",
        r"\toprule",
        r"& \multicolumn{3}{c}{mid fill} & \multicolumn{3}{c}{crossed fill} & worst & cost \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"variation & Sharpe & $\Delta$Sharpe & [95\% interval] & Sharpe & $\Delta$Sharpe & [95\% interval] & day & share \\",
        r"\midrule",
        r"\endhead",
    ]
    cst = cost.set_index("variation")
    fam_prev = None
    for vn in dict.fromkeys(h["variation"]):
        m = h[(h["variation"] == vn) & (h["fill"] == "mid")].iloc[0]
        x = h[(h["variation"] == vn) & (h["fill"] == "crossed")].iloc[0]
        if fam_prev is not None and m["family"] != fam_prev:
            out.append(r"\addlinespace[2pt]")
        fam_prev = m["family"]
        share = _cost_share(cst, vn, "own")
        if vn == "straddle":
            dm = dx = "---"
            im = ix = ""
        else:
            dm, dx = _f(m["dSharpe"]), _f(x["dSharpe"])
            im = f"[{_f(m['dSharpe_lo'])}, {_f(m['dSharpe_hi'])}]"
            ix = f"[{_f(x['dSharpe_lo'])}, {_f(x['dSharpe_hi'])}]"
        out.append(
            f"{_tex_label(vn)} & {_f(m['Sharpe'])} & {dm} & {im} & {_f(x['Sharpe'])} & {dx} & {ix} "
            f"& {_f(m['worst_day'])} & {_f(share)} \\\\"
        )
    out += [r"\bottomrule", r"\end{longtable}", r"\endgroup", ""]
    return out


def _best_note(hr: pd.DataFrame, best_mid: str, best_x: str) -> str:
    """The paired difference of a best row that is not the straddle, with its interval."""
    notes = []
    for vn, fill in ((best_mid, "mid"), (best_x, "crossed")):
        if vn != "straddle":
            r = hr.loc[(vn, fill)]
            notes.append(
                f" The {fill}-fill lead of the {_tex_label(vn)} row over the straddle is "
                f"{_f(float(r['dSharpe']))} [{_f(float(r['dSharpe_lo']))}, {_f(float(r['dSharpe_hi']))}]."
            )
    return "".join(notes)


def make_tex(head: pd.DataFrame, cost: pd.DataFrame) -> None:
    hr = head[head["rule"] == RULE_RESEARCH].set_index(["variation", "fill"])
    wing = [
        v
        for v in dict.fromkeys(head["variation"])
        if v.startswith(("iron butterfly", "straddle + wing"))
    ]
    below = {
        f: sum(float(hr.loc[(v, f), "dSharpe_hi"]) < 0 for v in wing) for f in FILLS
    }
    cst = cost.set_index("variation")
    ifly = [_cost_share(cst, f"iron butterfly, w{int(w)}", "own") for w in WIDTHS]
    icon = [_cost_share(cst, f"iron condor, w{int(w)}", "own") for w in WIDTHS]
    bfly = [
        _cost_share(cst, f"{t} butterfly, w{int(w)}", "straddle")
        for t in ("call", "put")
        for w in WIDTHS
    ]
    sh_mid = hr.xs("mid", level="fill")["Sharpe"]
    sh_x = hr.xs("crossed", level="fill")["Sharpe"]
    best_mid, best_x = str(sh_mid.idxmax()), str(sh_x.idxmax())
    lines = [
        "% AUTO-GENERATED by experiments/strategy_variations_1530.py -- do not edit.",
        "% Sources: results/strategy_variations/VARIATIONS.md (definitions), headline_per_straddle_premium.csv,",
        "% cost_line.csv. Every number below is read from those files by the generator.",
        r"\subsection{Strategy variations of the last-30-min trade}",
        r"\label{sec:app_strategy_variations}",
        "",
        r"The straddle --- the nearest out-of-the-money call and the nearest out-of-the-money put,",
        r"same-day expiry, one position --- is one choice of instrument for the $\mathrm{sign}(s)$",
        r"rule. Table~\ref{tab:app_strategy_variations_defs} writes down the standard alternatives",
        r"and the way the rule maps onto each: a variation is written as the structure $L$ bought",
        r"when $s>0$, and it is sold (the position is $-L$) when $s<0$; always short sells it every",
        r"day. The straddle with a wing hedge is the one exception: it holds the plain straddle on",
        r"buying days and the short iron butterfly on selling days. All structures are same-day",
        r"expiry SPXW options entered at the 15:30 quotes and cash-settled at the official close,",
        r"on the same 866 days as the straddle. Strikes are the live 15:30 strikes ($K_c$ the",
        r"smallest call strike at or above the index $S$, $K_p$ the largest put strike at or below",
        r"it); a wing is the nearest live strike at least $w\in\{10,25,50\}$ index points beyond the",
        r"body strike. Fills are the midpoint of every leg, or crossed (every leg bought at the ask",
        r"and sold at the bid). The return is the day's profit and loss in index points divided by",
        r"that day's straddle premium at the same fill, so every row shares the straddle row's",
        r"denominator and a paired difference isolates the structure; the straddle row is the",
        r"straddle trade's return. Intervals are 95\% circular block bootstrap intervals (block 21",
        r"days, 2{,}000 draws), paired against the straddle on the same days, rule and fill.",
        "",
        r"\begingroup\footnotesize\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{p{2.7cm}p{4.3cm}p{3.4cm}p{1.6cm}p{3.4cm}}",
        r"\caption{The strategy variations. $C(K)$, $P(K)$: a call, a put at strike $K$; $+$ bought,",
        r"$-$ sold in the buy-side form $L$. $K_c^{+}$, $K_p^{-}$: the next live strike beyond $K_c$,",
        r"$K_p$.}\label{tab:app_strategy_variations_defs}\\",
        r"\toprule",
        r"variation & bought when $s>0$ ($L$) & sold when $s<0$ ($-L$) & own notional & purpose \\",
        r"\midrule",
        r"\endhead",
    ]
    lines += [" & ".join(r) + r" \\" for r in TEX_DEFS]
    lines += [r"\bottomrule", r"\end{longtable}", r"\endgroup", ""]
    lines += [
        r"\paragraph{What the variations show.} Under $\mathrm{sign}(s)$ with the per-bar ridge",
        r"(Table~\ref{tab:app_strategy_variations_res}), the highest Sharpe ratio at the midpoint is",
        f"the {_tex_label(best_mid)} row ({_f(float(sh_mid[best_mid]))}; the straddle",
        f"{_f(float(sh_mid['straddle']))}), and at crossed fills the {_tex_label(best_x)} row",
        f"({_f(float(sh_x[best_x]))}).{_best_note(hr, best_mid, best_x)} Wings cost the rule at crossed fills: {below['crossed']}",
        f"of the {len(wing)} iron-butterfly and wing-hedge rows lie below the straddle with an interval",
        f"that excludes zero, against {below['mid']} of {len(wing)} at the midpoint, where the cost shrinks",
        r"as the wing moves out. The median entry cost (crossed minus mid) is",
        f"{_pct(_cost_share(cst, 'straddle', 'own'))} of the straddle's own premium, "
        f"{_pct(ifly[0])} (w10) falling to {_pct(ifly[-1])} (w50) of the iron",
        f"butterfly's, {_pct(icon[0])} falling to {_pct(icon[-1])} of the iron condor's, and the single-type",
        f"butterflies, with one wing in the money, cost {_pct(min(bfly))}--{_pct(max(bfly))} of the straddle's",
        f"premium against {_pct(_cost_share(cst, 'straddle', 'straddle'))} for the straddle itself.",
        "",
    ]
    lines += _results_table(
        head,
        cost,
        RULE_RESEARCH,
        r"Strategy variations under $\mathrm{sign}(s)$ with the per-bar ridge on the live-feasible set"
        r" (16:00-bar recalibration), 866 days (861 for the butterflies at $w=25,50$). Sharpe ratios"
        r" annualized by $\sqrt{252}$ per trade day, return per straddle premium; $\Delta$Sharpe:"
        r" the variation minus the straddle, paired. Worst day: mid fill, in straddle premiums."
        r" Cost share: the median crossed-minus-mid entry cost over the structure's own absolute"
        r" midpoint premium, averaged over the buying and the selling side.",
        "tab:app_strategy_variations_res",
    )
    lines += _results_table(
        head,
        cost,
        RULE_DECK,
        r"As Table~\ref{tab:app_strategy_variations_res}, with $\mathrm{sign}(s)$ on the block-diagonal"
        r" ridge (the paper's forecast, with the paper's recalibration over the session bars).",
        "tab:app_strategy_variations_res_blk",
    )
    lines += _results_table(
        head,
        cost,
        RULE_SHORT,
        r"As Table~\ref{tab:app_strategy_variations_res}, always short (no forecast): every structure"
        r" sold every day.",
        "tab:app_strategy_variations_res_short",
    )
    TEX.parent.mkdir(parents=True, exist_ok=True)
    TEX.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", TEX)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--extract",
        action="store_true",
        help="chain-sized extraction (hold the chain lock)",
    )
    a = ap.parse_args()
    if a.extract:
        extract()
    else:
        score()


if __name__ == "__main__":
    main()
