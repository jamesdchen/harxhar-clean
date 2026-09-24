"""Study 84 -- how long to leave the card's limit order at P* working before cancelling.

QUESTION (the operator's).  The daily card says: at ~15:31 place a limit BUY
for the 0DTE nearest-OTM strangle (call at/above the 15:30 spot, put at/below)
at P* = ``live.ibkr.pricing.package_price(sqrt(rv_hat), S, Kc, Kp)``, the
break-even for a 15:30 entry held to the 16:00 cash settlement.  If the ask is
already <= P* it fills at once at the ask; otherwise the order rests at P*.
How long should it be left working?  A late fill buys the same strangle with
less time left, and the ask comes down to P* when the market is quiet (adverse
selection), so a late fill is probably a worse trade.

DATA.  XSP 0DTE cbbo-1m NBBO (``data/archive/xsp_opra/cbbo1m_<day>.parquet``,
ts_recv = the sample stamp, UTC -> America/New_York), loaded with study 79's
own ``load_quotes`` / ``quote``.  Per day, the 15:30 pair, spot and settlement
payoff (SPX close / 10) come from study 79's ``xsp_close_book_daily.csv``;
rv_hat is the deck's (``daily_sub_live_ridge``, identical to study 80's h = 1
forecasts), so the sample is the 273 deck days in the XSP span
(2023-03-28 .. 2024-04-30).  No honest h = 1 rv_hat exists on disk after
2024-04-30 (the vendor-panel arms end there; producing one from the live
store would mean running the live forecast code, which is out of scope here).
Month-end days are bought unconditionally at the ask by the card (not a limit
at P*) and are EXCLUDED: 259 days.

FILL MODEL (conservative, the model of record).  Placement t0 in {15:30,
15:31, 15:32} with the 15:30 strikes and P* (the card's).  If the package ask
(call ask + put ask) at the t0 sample is <= P*: filled at t0 at that ask.
Otherwise the order rests at P* and fills at the first later minute sample
m in (t0, T] whose package ask <= P*, at P*.  Cancel T = t0, t0+1, .., 15:59.
R = payoff / fill - 1 (the leg is budget-sized, so R is the return on the
budget); a day without a fill contributes 0.

SENSITIVITIES (labelled, not of record).
  * TOUCH-AT-ASK: as the model of record, but a resting fill is priced at the
    touching minute's ask (<= P*) instead of P* -- a complex order legging into
    the leg market at its prices.
  * MID: the resting order fills at the first m in (t0, T] whose package
    midpoint <= P* (at P*); the immediate set is unchanged.
  * SPX-WIDTH PROXY: the XSP mid re-spread to the SPX line's width -- the ask
    is replaced by mid * (1 + s/2), s = that day's SPXW 15:30 relative spread
    from the chain (study 79's ``spx_rel_spread``), both for the immediate test
    (fill at that proxy ask) and the resting test (fill at P*).  A proxy for
    an SPX book, NOT an SPX book: the XSP strikes and XSP mid path are kept.
  * SPX (real): if SPXW cbbo-1m files for NON-month-end deck days are on disk
    (``data/archive/spxw_opra/``), the conservative model is run on them with
    the SPXW pair / P* / payoff of study 79.  The 42 files there at the time of
    writing are month-ends only.

GATES.  (1) P* recomputed here equals study 79's ``p_star_xsp``; (2) the
package ask at the 15:30 sample rebuilt from the minute series equals study
79's 15:30 ``ask`` on every day, and the 15:30 pass count (ask <= P*) equals
study 79's ``card_long`` count.

    python writeup/intraday_proposals/84_limit_order_window.py
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from live.ibkr.pricing import package_price  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "s79_xsp_close_book",
    REPO / "writeup" / "intraday_proposals" / "79_xsp_close_book.py",
)
assert _spec is not None and _spec.loader is not None
s79 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s79)

XSP_DIR = REPO / "data" / "archive" / "xsp_opra"
SPXW_DIR = REPO / "data" / "archive" / "spxw_opra"
DAILY79 = (
    REPO
    / "results"
    / "atm_straddle_intraday_holdclose"
    / "proposals"
    / "79"
    / "xsp_close_book_daily.csv"
)
DECK = REPO / "results" / "atm_straddle_0dte_1530" / "daily_sub_live_ridge.parquet"
OUT = REPO / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "84"

MINUTES = [f"15:{m:02d}" for m in range(30, 60)]  # 15:30 .. 15:59, index 0 .. 29
T0S = {"15:30": 0, "15:31": 1, "15:32": 2}
BUCKETS = [
    (0, 0, "0 (immediate)"),
    (1, 2, "1-2"),
    (3, 5, "3-5"),
    (6, 10, "6-10"),
    (11, 29, "11-29"),
]


def minute_book(path: str, day: str, kc: float, kp: float) -> dict[str, np.ndarray]:
    """Package bid / ask on the 15:30 .. 15:59 minute samples for one pair (study 79's quote())."""
    q = s79.load_quotes(Path(path))
    d = pd.Timestamp(day)
    q = q[
        ((q["strike"] == kc) & (q["cp"] == "C"))
        | ((q["strike"] == kp) & (q["cp"] == "P"))
    ]
    bid = np.full(len(MINUTES), np.nan)
    ask = np.full(len(MINUTES), np.nan)
    for i, hm in enumerate(MINUTES):
        b = s79.book_at(q, d, f"{hm}:00")
        bc, ac, _, _ = s79.quote(b, kc, "C")
        bp, ap, _, _ = s79.quote(b, kp, "P")
        bid[i], ask[i] = bc + bp, ac + ap
    return {"bid": bid, "ask": ask}


def _job(args: tuple[str, str, float, float]) -> tuple[str, dict[str, np.ndarray]]:
    path, day, kc, kp = args
    return day, minute_book(path, day, kc, kp)


def fills(
    ask_t: np.ndarray,
    cond: np.ndarray,
    pstar: float,
    payoff: float,
    i0: int,
    at_touch: bool = False,
) -> tuple[float, float, float]:
    """(delay in minutes or NaN, fill price, R): ask at t0 <= P* -> at that ask; else the first m > t0 with
    cond, at P* (or, ``at_touch``, at that minute's ask, which is <= P* there)."""
    if np.isfinite(ask_t[i0]) and ask_t[i0] <= pstar:
        px = float(ask_t[i0])
        return 0.0, px, payoff / px - 1.0
    later = np.flatnonzero(cond[i0 + 1 :])
    if len(later) == 0:
        return np.nan, np.nan, np.nan
    m = int(later[0]) + 1
    px = float(ask_t[i0 + m]) if at_touch else pstar
    return float(m), px, payoff / px - 1.0


def tstat(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or np.std(x, ddof=1) == 0:
        return float("nan")
    return float(np.mean(x) / (np.std(x, ddof=1) / np.sqrt(len(x))))


def cancel_table(ev: pd.DataFrame, model: str) -> pd.DataFrame:
    """Per (t0, T): the per-day return on budget (0 on no fill) and the fill statistics."""
    rows = []
    n = ev["day"].nunique()
    for t0, i0 in T0S.items():
        e = ev[(ev.model == model) & (ev.t0 == t0)].set_index("day")
        base = np.where(e.delay == 0, e.R, 0.0)
        for j in range(i0, len(MINUTES)):
            k = j - i0
            f = e.delay.notna() & (e.delay <= k)
            x = np.where(f, e.R, 0.0)
            rf = e.R[f]
            rows.append(
                {
                    "model": model,
                    "t0": t0,
                    "cancel": MINUTES[j],
                    "min_after_t0": k,
                    "days": n,
                    "filled": int(f.sum()),
                    "immediate": int((e.delay == 0).sum()),
                    "late": int((f & (e.delay > 0)).sum()),
                    "mean_per_day": float(x.mean()),
                    "t": tstat(x),
                    "mean_R_filled": float(rf.mean()) if len(rf) else np.nan,
                    "hit_filled": float((rf > 0).mean()) if len(rf) else np.nan,
                    "gain_vs_immediate": float((x - base).mean()),
                    "t_gain": tstat(x - base),
                }
            )
    return pd.DataFrame(rows)


def delay_curve(ev: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = []
    for t0 in T0S:
        e = ev[(ev.model == model) & (ev.t0 == t0) & ev.delay.notna()]
        for lo, hi, lab in BUCKETS:
            g = e[(e.delay >= lo) & (e.delay <= hi)]
            rows.append(
                {
                    "model": model,
                    "t0": t0,
                    "delay_min": lab,
                    "n": len(g),
                    "mean_R": float(g.R.mean()) if len(g) else np.nan,
                    "median_R": float(g.R.median()) if len(g) else np.nan,
                    "t": tstat(g.R.to_numpy()),
                    "hit": float((g.R > 0).mean()) if len(g) else np.nan,
                    "sum_R": float(g.R.sum()),
                    "fill_over_mid1530": float((g.fill / g.mid1530).mean())
                    if len(g)
                    else np.nan,
                }
            )
        g = e[e.delay > 0]
        rows.append(
            {
                "model": model,
                "t0": t0,
                "delay_min": "all late (>=1)",
                "n": len(g),
                "mean_R": float(g.R.mean()) if len(g) else np.nan,
                "median_R": float(g.R.median()) if len(g) else np.nan,
                "t": tstat(g.R.to_numpy()),
                "hit": float((g.R > 0).mean()) if len(g) else np.nan,
                "sum_R": float(g.R.sum()),
                "fill_over_mid1530": float((g.fill / g.mid1530).mean())
                if len(g)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def events(
    days: pd.DataFrame, books: dict[str, dict[str, np.ndarray]], models: dict[str, str]
) -> pd.DataFrame:
    """One row per (day, model, t0): the fill delay, price and R."""
    rows = []
    for day, r in days.iterrows():
        bk = books[day.strftime("%Y-%m-%d")]
        bid, ask = bk["bid"], bk["ask"]
        mid = (bid + ask) / 2
        with np.errstate(invalid="ignore"):
            series = {
                "conservative": (ask, ask <= r.pstar),
                "touch_at_ask": (ask, ask <= r.pstar),
                "mid": (ask, mid <= r.pstar),
            }
            if "spx_width_proxy" in models:
                pa = mid * (1 + r.spx_rel_spread / 2)
                series["spx_width_proxy"] = (pa, pa <= r.pstar)
        for model in models:
            a_t, cond = series[model]
            for t0, i0 in T0S.items():
                dly, px, R = fills(
                    a_t,
                    np.nan_to_num(cond, nan=0).astype(bool),
                    r.pstar,
                    r.payoff,
                    i0,
                    at_touch=model == "touch_at_ask",
                )
                rows.append(
                    {
                        "day": day,
                        "model": model,
                        "t0": t0,
                        "delay": dly,
                        "fill": px,
                        "R": R,
                        "mid1530": mid[0],
                        "ask_t0": a_t[i0],
                        "pstar": r.pstar,
                        "payoff": r.payoff,
                    }
                )
    return pd.DataFrame(rows)


def fmt_cancel(tab: pd.DataFrame, t0: str, every: list[int]) -> list[str]:
    g = tab[(tab.t0 == t0) & tab.min_after_t0.isin(every)]
    out = [
        "  cancel  +min  filled (imm+late)  per-day   t     R|filled  hit   gain_vs_imm (t)"
    ]
    for r in g.itertuples():
        out.append(
            f"  {r.cancel}  {r.min_after_t0:>3}   {r.filled:>3} ({r.immediate}+{r.late:<3})    "
            f"{r.mean_per_day:+.4f}  {r.t:5.2f}  {r.mean_R_filled:+.3f}   {r.hit_filled:.0%}  "
            f"{r.gain_vs_immediate:+.4f} ({r.t_gain:+.2f})"
        )
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    x = pd.read_csv(DAILY79, index_col=0, parse_dates=True)
    deck = pd.read_parquet(DECK)
    deck.index = pd.DatetimeIndex(deck.index)
    d = x[x.in_deck & x.rv_hat.notna()].copy()
    # rv_hat straight from the deck (the csv's %.6g formatting is not the source)
    d["rv_hat"] = deck.loc[d.index, "rv_hat"].astype(float)
    d["pstar"] = [
        package_price(np.sqrt(r.rv_hat), r.spot_xsp, r.kc, r.kp) for r in d.itertuples()
    ]
    lines: list[str] = []
    g1 = np.nanmax(np.abs(d.pstar / d.p_star_xsp - 1))
    lines.append(
        f"GATE 1 P* vs study 79 p_star_xsp on {len(d)} deck days: max |rel diff| {g1:.1e}"
    )
    months_end = int(d.month_end.sum())
    ev_days = d[~d.month_end.astype(bool)].copy()
    lines.append(
        f"sample: {len(d)} deck days with an XSP book ({d.index.min().date()} .. {d.index.max().date()}); "
        f"{months_end} month-ends excluded (bought unconditionally at the ask) -> {len(ev_days)} days"
    )

    jobs = [
        (
            str(XSP_DIR / f"cbbo1m_{day:%Y-%m-%d}.parquet"),
            f"{day:%Y-%m-%d}",
            float(r.kc),
            float(r.kp),
        )
        for day, r in d.iterrows()
    ]
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
        books = dict(ex.map(_job, jobs, chunksize=8))
    a1530 = np.array([books[f"{day:%Y-%m-%d}"]["ask"][0] for day in d.index])
    g2 = np.nanmax(np.abs(a1530 / d["ask"].to_numpy() - 1))
    pass1530 = int(np.sum(a1530 <= d.pstar.to_numpy()))
    lines.append(
        f"GATE 2 15:30 package ask from the minute series vs study 79 on {len(d)} days: max |rel diff| {g2:.1e}; "
        f"15:30 passes {pass1530} vs study 79 card_long {int(d.card_long.sum())}"
    )
    miss = {
        k: int(np.isnan(v["ask"]).sum())
        for k, v in books.items()
        if np.isnan(v["ask"]).any()
    }
    lines.append(
        f"minute samples with no two-sided quote on the pair (no fill possible that minute): "
        f"{sum(miss.values())} of {len(d) * len(MINUTES)} on {len(miss)} days"
    )
    if g1 > 1e-4 or g2 > 1e-4 or pass1530 != int(d.card_long.sum()):
        print("\n".join(lines))
        print("GATE FAILED -- stopping")
        return 1

    models = {
        "conservative": "ask <= P*, resting fill at P*",
        "touch_at_ask": "ask <= P*, resting fill at that minute's ask (sensitivity)",
        "mid": "mid <= P*, resting fill at P* (sensitivity)",
        "spx_width_proxy": "XSP mid re-spread to the SPXW width, fill at P* (proxy)",
    }
    ev = events(ev_days, books, models)
    ev.to_csv(OUT / "fill_events.csv", index=False, float_format="%.6g")
    tab = pd.concat([cancel_table(ev, m) for m in models], ignore_index=True)
    tab.to_csv(OUT / "cancel_table.csv", index=False, float_format="%.6g")
    cur = pd.concat([delay_curve(ev, m) for m in models], ignore_index=True)
    cur.to_csv(OUT / "adverse_selection_curve.csv", index=False, float_format="%.6g")

    show = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 27, 28, 29]
    for model, lab in models.items():
        lines.append("")
        lines.append(f"=== model: {model} ({lab}) ===")
        t = tab[tab.model == model]
        for t0 in T0S:
            tt = t[t.t0 == t0]
            best = tt.loc[tt.mean_per_day.idxmax()]
            lines.append(
                f"t0 {t0}: best cancel {best.cancel} (+{best.min_after_t0} min) per-day {best.mean_per_day:+.4f} "
                f"(t {best.t:.2f}); cancel at once {tt.iloc[0].mean_per_day:+.4f} (t {tt.iloc[0].t:.2f}); "
                f"to 15:59 {tt.iloc[-1].mean_per_day:+.4f} (t {tt.iloc[-1].t:.2f})"
            )
            if model == "conservative" or t0 == "15:31":
                lines.extend(fmt_cancel(t, t0, show))
        c = cur[cur.model == model]
        lines.append(
            f"adverse-selection curve (fills by delay after t0, cancel 15:59), {model}:"
        )
        for r in c.itertuples():
            lines.append(
                f"  t0 {r.t0} delay {r.delay_min:<14} n {r.n:>3}  mean R {r.mean_R:+.3f}  median {r.median_R:+.3f}  "
                f"t {r.t:5.2f}  hit {r.hit:.0%}  sum R {r.sum_R:+.2f}  fill/mid1530 {r.fill_over_mid1530:.3f}"
            )

    # split halves of the sample (model of record, t0 15:31, cancel 15:59)
    e = ev[(ev.model == "conservative") & (ev.t0 == "15:31") & ev.delay.notna()]
    cut = ev_days.index[len(ev_days) // 2]
    lines.append("")
    lines.append(
        f"split halves (conservative, t0 15:31, cancel 15:59; second half from {cut.date()}):"
    )
    for lab, h in (("first", e[e.day < cut]), ("second", e[e.day >= cut])):
        im, lt = h[h.delay == 0].R, h[h.delay > 0].R
        lines.append(
            f"  {lab} half: immediate n {len(im)} mean R {im.mean():+.3f} (t {tstat(im.to_numpy()):.2f}); "
            f"late n {len(lt)} mean R {lt.mean():+.3f} (t {tstat(lt.to_numpy()):.2f}), "
            f"expired worthless {(lt <= -0.999).mean():.0%} vs immediate {(im <= -0.999).mean():.0%}"
        )

    # the venue gap on the same days: the 15:30 ask test on XSP vs the SPXW chain
    lines.append("")
    both = ev_days[ev_days.spx_ask.notna()]
    lines.append(
        f"venue gap on {len(both)} non-month-end days: rel spread at 15:30 median XSP {both.rel_spread.median():.1%} vs "
        f"SPXW {both.spx_rel_spread.median():.1%}; 15:30 ask <= P* on XSP {int(both.card_long.sum())} days vs "
        f"SPXW {int(both.card_long_spx.sum())} days (the chain has 30-minute stamps only, so no SPX resting-order path)"
    )

    # SPX (real): SPXW 1-minute files for non-month-end deck days, if any are on disk.
    # A pull may be writing into the directory: files touched in the last 2 minutes are skipped.
    now = time.time()
    spx_files = {
        Path(f).stem.split("_")[1]: f
        for f in glob.glob(str(SPXW_DIR / "cbbo1m_*.parquet"))
        if now - os.path.getmtime(f) > 120
    }
    on_disk = np.array(
        [f"{day:%Y-%m-%d}" in spx_files for day in ev_days.index], dtype=bool
    )
    sd = ev_days[on_disk & ev_days.spx_kc.notna().to_numpy()].copy()
    lines.append(
        f"SPX section: {len(spx_files)} settled SPXW cbbo-1m files on disk; "
        f"non-month-end deck days among them: {len(sd)}"
        + (f" ({sd.index.min().date()} .. {sd.index.max().date()})" if len(sd) else "")
    )
    if len(sd):
        # study 79's SPXW pair (nearest-OTM on the chain's listed strikes at 15:30), P* on the SPX spot
        sd["kc"], sd["kp"], sd["payoff"] = sd.spx_kc, sd.spx_kp, sd.spx_payoff
        sd["pstar"] = [
            package_price(np.sqrt(r.rv_hat), r.spx_S, r.kc, r.kp)
            for r in sd.itertuples()
        ]
        sjobs = [
            (spx_files[f"{day:%Y-%m-%d}"], f"{day:%Y-%m-%d}", float(r.kc), float(r.kp))
            for day, r in sd.iterrows()
        ]
        with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
            sbooks = dict(ex.map(_job, sjobs, chunksize=8))
        oa = np.array([sbooks[f"{day:%Y-%m-%d}"]["ask"][0] for day in sd.index])
        rel = np.abs(oa / sd.spx_ask.to_numpy() - 1)
        smiss = sum(int(np.isnan(v["ask"]).sum()) for v in sbooks.values())
        lines.append(
            f"SPX GATE SPXW OPRA 15:30 package ask vs the chain's 15:30 ask (study 79) on "
            f"{int(np.isfinite(rel).sum())} days: median |rel diff| {np.nanmedian(rel):.1%}, "
            f"p90 {np.nanquantile(rel, 0.9):.1%}; 15:30 ask <= P* OPRA {int(np.sum(oa <= sd.pstar.to_numpy()))} "
            f"vs chain {int(sd.card_long_spx.sum())}; minute samples without a two-sided quote "
            f"{smiss} of {len(sd) * len(MINUTES)}"
        )
        smodels = {
            "conservative": "ask <= P* (SPXW)",
            "mid": "mid <= P* (SPXW, sensitivity)",
        }
        sev = events(sd, sbooks, smodels)
        sev.to_csv(OUT / "spx_fill_events.csv", index=False, float_format="%.6g")
        st = pd.concat([cancel_table(sev, m) for m in smodels], ignore_index=True)
        st.to_csv(OUT / "spx_cancel_table.csv", index=False, float_format="%.6g")
        sc = pd.concat([delay_curve(sev, m) for m in smodels], ignore_index=True)
        sc.to_csv(
            OUT / "spx_adverse_selection_curve.csv", index=False, float_format="%.6g"
        )
        for model, lab in smodels.items():
            lines.append(
                f"=== SPX (SPXW cbbo-1m, {len(sd)} days), model {model} ({lab}) ==="
            )
            t = st[st.model == model]
            for t0 in T0S:
                tt = t[t.t0 == t0]
                best = tt.loc[tt.mean_per_day.idxmax()]
                lines.append(
                    f"t0 {t0}: best cancel {best.cancel} (+{best.min_after_t0} min) per-day "
                    f"{best.mean_per_day:+.4f} (t {best.t:.2f}); cancel at once {tt.iloc[0].mean_per_day:+.4f} "
                    f"(t {tt.iloc[0].t:.2f}); to 15:59 {tt.iloc[-1].mean_per_day:+.4f} (t {tt.iloc[-1].t:.2f})"
                )
                if t0 in ("15:30", "15:31"):
                    lines.extend(fmt_cancel(t, t0, show))
            for r in sc[sc.model == model].itertuples():
                lines.append(
                    f"  t0 {r.t0} delay {r.delay_min:<14} n {r.n:>3}  mean R {r.mean_R:+.3f}  "
                    f"median {r.median_R:+.3f}  t {r.t:5.2f}  hit {r.hit:.0%}  sum R {r.sum_R:+.2f}  "
                    f"fill/mid1530 {r.fill_over_mid1530:.3f}"
                )
    else:
        lines.append(
            "  none: the SPX answer needs SPXW 1-minute quotes on the non-month-end days "
            "(live/close_signal/pull_databento_xsp.py --root SPXW --pull; files land in data/archive/spxw_opra/)"
        )

    txt = "\n".join(lines)
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
