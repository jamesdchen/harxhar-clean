"""Study 90 -- a walk-forward regime filter on the daily card.

The card buys the SPXW 0DTE nearest-OTM strangle at 15:30 iff ask <= P*
(rule F0).  After 2024 it buys on half the days and loses; in 2025 Q2-Q3 it
made money.  Idea: trade only in conditions that have been paying.  To stay
honest the condition is never picked by looking at the test years: each
candidate splits days into buckets by something known at 15:30, and on day d
the card's buy is taken only if F0's buys IN THAT BUCKET made money over the
previous N sessions (N = 250; fewer than MIN_BUYS trailing buys in the bucket
-> trade as F0).  Bucket edges that depend on the level (VIX, the day's
realized move) are terciles of the trailing N sessions, not of the sample.

Candidates (each known at 15:30):
  C1 VIX at 15:30 (vendor 30-min prints to 2024-02-12, the store after)
  C2 the day's realized variance 10:00-15:30 (sum of the card table's bar
     rv_raw) over its trailing 22-session mean
  C3 FOMC statement day (live/close_signal/state/fomc_statement_dates.csv)
  C4 the card's own form: mean R of F0's last 20 buys > 0 (momentum)
  C5 the price's cushion: log(iv_var / trailing-22 mean realized) -- how
     rich the 15:30 implied is against the recent realized

Data: study 88's per-day table (one model, the card's own, through its
causal MZ map; SPX 15:30 ask from the chain / OPRA; R on every day).  Month-
ends excluded.  Compared on the same days as study 88 (from 2021-09-20,
when 0DTE SPXW trades every day), split at 2024-05-01.

    python writeup/intraday_proposals/90_regime_filter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
P88 = REPO / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "88"
OUT = REPO / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "90"
YHAT = REPO / "live" / "close_signal" / ".scratch" / "yhat_close_signal.parquet"
VENDOR_VIX = REPO / "data" / "vix_and_voldemand.parquet"
STORE = REPO / "live" / "close_signal" / "state" / "panel_free.parquet"
FOMC = REPO / "live" / "close_signal" / "state" / "fomc_statement_dates.csv"
ET = "America/New_York"
N = 250
MIN_BUYS = 15
START = pd.Timestamp("2021-09-20")
SPLIT = pd.Timestamp("2024-05-01")
BLOCK = 20
N_BOOT = 4000


def vix_1530() -> pd.Series:
    def at(df: pd.DataFrame) -> pd.Series:
        t = pd.to_datetime(df["endbartime"])
        m = t.dt.strftime("%H:%M") == "15:30"
        return pd.Series(
            df.loc[m, "vix"].to_numpy(), index=t[m].dt.normalize().to_numpy()
        )

    v = at(pd.read_parquet(VENDOR_VIX, columns=["endbartime", "vix"])).dropna()
    s = at(pd.read_parquet(STORE, columns=["endbartime", "vix"])).dropna()
    out = pd.concat([v, s])
    return out[~out.index.duplicated(keep="first")].sort_index()


def morning_rv() -> pd.Series:
    """Sum of the card table's bar rv_raw 10:00 .. 15:30 (bar-end stamps) per session."""
    y = pd.read_parquet(YHAT, columns=["t", "rv_raw"])
    t = pd.to_datetime(y["t"])
    if t.dt.tz is not None:  # the table's t is true UTC (read_arm localizes ET first)
        t = t.dt.tz_convert(ET).dt.tz_localize(None)
    hm = t.dt.strftime("%H:%M")
    m = (hm >= "10:00") & (hm <= "15:30")
    return y.loc[m, "rv_raw"].groupby(t[m].dt.normalize()).sum()


def trailing_tercile(x: pd.Series, n: int) -> pd.Series:
    """Tercile (0/1/2) of x_d against the previous n values (no look-ahead)."""
    out = pd.Series(np.nan, index=x.index)
    vals = x.to_numpy()
    for i in range(len(x)):
        past = vals[max(0, i - n) : i]
        past = past[np.isfinite(past)]
        if len(past) < 60 or not np.isfinite(vals[i]):
            continue
        lo, hi = np.quantile(past, [1 / 3, 2 / 3])
        out.iloc[i] = 0 if vals[i] <= lo else (1 if vals[i] <= hi else 2)
    return out


def walk_forward(d: pd.DataFrame, bucket: pd.Series) -> pd.Series:
    """Trade F0's buy on day i iff F0's buys in bucket b_i made money over the previous N sessions."""
    buy0 = d["buy0"].to_numpy()
    R = d["R"].to_numpy()
    b = bucket.to_numpy()
    take = np.zeros(len(d), dtype=bool)
    for i in range(len(d)):
        if not buy0[i]:
            continue
        lo = max(0, i - N)
        if not np.isfinite(b[i]):
            take[i] = True
            continue
        m = buy0[lo:i] & (b[lo:i] == b[i])
        if m.sum() < MIN_BUYS:
            take[i] = True
            continue
        take[i] = float(np.nanmean(R[lo:i][m])) > 0.0
    return pd.Series(take, index=d.index)


def block_boot_diff(
    a: np.ndarray, b: np.ndarray, seed: int = 0
) -> tuple[float, float, float]:
    d = a - b
    n = len(d)
    rng = np.random.default_rng(seed)
    k = max(1, n // BLOCK)
    starts = rng.integers(0, max(1, n - BLOCK), size=(N_BOOT, k))
    boot = np.array(
        [d[(s[:, None] + np.arange(BLOCK)).ravel() % n].mean() for s in starts]
    )
    return (
        float(d.mean()),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(P88 / "days.parquet")
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").set_index("date")
    d = d[~d["month_end"].astype(bool)]
    d = d[np.isfinite(d["R"]) & np.isfinite(d["P_star"]) & (d["ask"] > 0)]
    d["buy0"] = d["ask"] <= d["P_star"]
    vix = vix_1530()
    mrv = morning_rv()
    d["vix"] = vix.reindex(d.index)
    d["mrv_ratio"] = mrv.reindex(d.index) / mrv.rolling(
        22, min_periods=15
    ).mean().shift(1).reindex(d.index)
    rv16 = (
        d["rv_raw"].shift(1).rolling(22, min_periods=15).mean()
    )  # yesterday back: known at 15:30
    d["cushion"] = np.log(d["iv_var"] / rv16)
    # one YYYY-MM-DD per line after '#' comment lines (the file's own format)
    fomc = {
        pd.Timestamp(ln.strip()[:10])
        for ln in FOMC.read_text(encoding="utf-8").splitlines()
        if ln.strip()[:1].isdigit()
    }
    buckets = {
        "C1 VIX tercile": trailing_tercile(d["vix"], N),
        "C2 morning realized tercile": trailing_tercile(np.log(d["mrv_ratio"]), N),
        "C3 FOMC day": pd.Series(
            [1.0 if t in fomc else 0.0 for t in d.index], index=d.index
        ),
        "C5 cushion tercile": trailing_tercile(d["cushion"], N),
    }
    rules = {"F0 card": d["buy0"]}
    for name, b in buckets.items():
        rules[name] = walk_forward(d, b)
    # C4: the card's own form -- mean R of F0's last 20 buys > 0
    buy0, R = d["buy0"].to_numpy(), d["R"].to_numpy()
    form = np.zeros(len(d), dtype=bool)
    for i in range(len(d)):
        past = R[:i][buy0[:i]][-20:]
        form[i] = bool(buy0[i]) and (len(past) < 20 or float(np.mean(past)) > 0.0)
    rules["C4 own form (last 20 buys)"] = pd.Series(form, index=d.index)

    ev = d[d.index >= START]
    lines = [
        f"days {len(ev)} ({ev.index.min().date()} .. {ev.index.max().date()}), month-ends excluded; "
        f"research part {int((ev.index < SPLIT).sum())}, post {int((ev.index >= SPLIT).sum())}; N={N}, MIN_BUYS={MIN_BUYS}",
        f"FOMC days in span: {int(buckets['C3 FOMC day'].reindex(ev.index).sum())}",
        "rule                          | research 2021-09..2024-04: share, per-day (t) | post 2024-05..2026-09: share, per-day (t), diff vs F0 [95% CI]",
    ]
    rows = []
    for name, take in rules.items():
        t_ = take.reindex(ev.index).fillna(False).astype(bool)
        pnl = np.where(t_, ev["R"], 0.0)
        out: dict[str, object] = {"rule": name}
        seg_txt = []
        for seg, m in (("research", ev.index < SPLIT), ("post", ev.index >= SPLIT)):
            p = pnl[m]
            p0 = np.where(ev["buy0"][m], ev["R"][m], 0.0)
            tt = p.mean() / (p.std(ddof=1) / np.sqrt(len(p)))
            diff, lo, hi = block_boot_diff(p, p0)
            out |= {
                f"{seg}_share": float(t_[m].mean()),
                f"{seg}_pnl": float(p.mean()),
                f"{seg}_t": float(tt),
                f"{seg}_diff": diff,
                f"{seg}_lo": lo,
                f"{seg}_hi": hi,
            }
            seg_txt.append(
                f"{t_[m].mean():4.0%}, {p.mean():+.4f} ({tt:+.2f})"
                + ("" if name == "F0 card" else f", {diff:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            )
        rows.append(out)
        lines.append(f"{name:30s}| {seg_txt[0]:44s}| {seg_txt[1]}")
    by_year = []
    for name, take in rules.items():
        t_ = take.reindex(ev.index).fillna(False).astype(bool)
        pnl = pd.Series(np.where(t_, ev["R"], 0.0), index=ev.index)
        for y, g in pnl.groupby(pnl.index.year):
            by_year.append(
                {
                    "rule": name,
                    "year": y,
                    "days": len(g),
                    "buys": int(t_[g.index].sum()),
                    "pnl_day": g.mean(),
                }
            )
    by = pd.DataFrame(by_year).pivot(index="rule", columns="year", values="pnl_day")
    lines += ["", "per-day P&L by year:", by.round(4).to_string()]
    lines += [
        "",
        "caveat: five candidate filters, none chosen after seeing the post years; each is walk-forward, "
        "but the list itself was picked with knowledge that 2024 lost and 2025 Q2-Q3 made money",
    ]
    txt = "\n".join(lines)
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False, float_format="%.6g")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
