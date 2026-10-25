"""Write writeup/generated/rehaul_live_numbers.tex: when the late-starting series are live under the late-start rule.

Section 4, step 7 (src/backtest/executor.py LATE_START_WARMUP, UNOBSERVED_NEUTRAL): a series whose first print comes
after the first day of the sample has its standardized k-interval value average set to 0 until every row of the
estimation window (W = 39,000 regular-hours bars, 3,000 days of 13 bars) is live, a row being live when its k-bar
average holds no bar before the day of the first print; and set to 0 whenever the k-bar average holds no observed bar
(a stopped feed). Liveness is monotone, so the value average of horizon k is live on the bars from
first-print-day start + k + W to the last observed bar + k. This script counts, per late series and horizon, the share
of the scored regular-hours bars (2014-01-02 .. 2024-04-30, the every-bar span of Table 1) and of the 16:00 bars of
the 614 post-warmup option days that are live; macros give the range over the horizons k (Lo = the longest horizon,
the least live; Hi = k = 1). Availability averages are not affected by the rule.

    python writeup/make_rehaul_live_windows.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "writeup" / "generated" / "rehaul_live_numbers.tex"
W = 39_000  # rows of the estimation window: 3,000 days x 13 regular-hours bars
BARS = 13
HORIZONS = (1, 5, 25, 125, 625, 3125)  # base-5 HAR horizons of the final design
SPAN = ("2014-01-02", "2024-04-30")  # every-bar span of Table 1
N_PW = 614  # post-warmup option days (Section 7), asserted below
WARMUP = 252

# series: macro stem, data file, column(s); the four volatility-demand series share their dates, as do the
# StockTwits sentiment and tag count (checked below)
SERIES: list[tuple[str, str, list[str]]] = [
    ("Vix", "vix_and_voldemand.parquet", ["vix"]),
    ("VixThree", "vix_and_voldemand.parquet", ["vix3m"]),
    ("Vvix", "vix_and_voldemand.parquet", ["vvix"]),
    ("StAtt", "spy_and_sentiment.parquet", ["stocktwits_attention"]),
    ("StSent", "spy_and_sentiment.parquet", ["stocktwits_sentiment", "stocktwits_sentcount"]),
    (
        "Vd",
        "vix_and_voldemand.parquet",
        [
            "voldemand_spx_open_and_close",
            "voldemand_spx_open_only",
            "voldemand_all_open_and_close",
            "voldemand_all_open_only",
        ],
    ),
]
# the VIX family's feed stops 2024-02-12 in the panel and is completed to 2024-04-30 from hourly closes
# (src/data/loading.py, Section 3), so it is observed to the end of the sample
OBSERVED_TO_END = {"Vix", "VixThree", "Vvix"}


def trading_days() -> pd.DatetimeIndex:
    """The days with regular-hours constituent bars (the sample's trading calendar)."""
    t = pd.to_datetime(pd.read_parquet(REPO / "data" / "ewstock_stats.parquet", columns=["endbartime"])["endbartime"])
    mins = t.dt.hour * 60 + t.dt.minute
    return pd.DatetimeIndex(sorted(t[(mins > 9 * 60 + 30) & (mins <= 16 * 60)].dt.normalize().unique()))


def option_days() -> pd.DatetimeIndex:
    """The 614 post-warmup option days (the 866 option days of the master table, after the first 252)."""
    d = pd.read_parquet(REPO / "results" / "close_master_table" / "master_table_daily.parquet", columns=["day"])
    days = pd.DatetimeIndex(pd.to_datetime(d["day"]).dt.normalize().drop_duplicates().sort_values())
    assert len(days) == 866, len(days)
    pw = days[WARMUP:]
    assert len(pw) == N_PW, len(pw)
    return pw


def main() -> None:
    days = trading_days()
    pos = {d: i for i, d in enumerate(days)}
    span_days = days[(days >= pd.Timestamp(SPAN[0])) & (days <= pd.Timestamp(SPAN[1]))]
    # every regular-hours bar of the span, as bar indices on the 13-bar frame
    span_bars = (np.array([pos[d] for d in span_days])[:, None] * BARS + np.arange(BARS)[None, :]).ravel()
    pw_bars = np.array([pos[d] for d in option_days()]) * BARS + (BARS - 1)  # the 16:00 bar of each option day
    macros: list[tuple[str, str, str]] = []
    for stem, fname, cols in SERIES:
        d = pd.read_parquet(REPO / "data" / fname, columns=["endbartime", *cols])
        t = pd.to_datetime(d["endbartime"])
        firsts = {c: t[d[c].notna()].min().normalize() for c in cols}
        lasts = {c: t[d[c].notna()].max().normalize() for c in cols}
        assert len(set(firsts.values())) == 1 and len(set(lasts.values())) == 1, (stem, firsts, lasts)
        first, last = next(iter(firsts.values())), next(iter(lasts.values()))
        start_bar = pos[days[days.searchsorted(first)]] * BARS  # first row of the first-print day
        last_obs_bar = (len(days) if stem in OBSERVED_TO_END else pos[days[days.searchsorted(last)]] + 1) * BARS - 1
        ev, op = [], []
        for k in HORIZONS:
            lo_bar, hi_bar = start_bar + k + W, last_obs_bar + k
            ev.append(100.0 * float(np.mean((span_bars >= lo_bar) & (span_bars <= hi_bar))))
            op.append(100.0 * float(np.mean((pw_bars >= lo_bar) & (pw_bars <= hi_bar))))
        first_live = days[min(len(days) - 1, (start_bar + 1 + W) // BARS)]
        macros += [
            (f"rhlv{stem}EvLo", f"{min(ev):.1f}", f"% live, scored bars {SPAN[0]}..{SPAN[1]}, longest horizon"),
            (f"rhlv{stem}EvHi", f"{max(ev):.1f}", "% live, scored bars, k = 1"),
            (f"rhlv{stem}OptLo", f"{min(op):.1f}", "% live, 16:00 bars of the 614 post-warmup option days, longest k"),
            (f"rhlv{stem}OptHi", f"{max(op):.1f}", "% live, 16:00 bars of the 614 post-warmup option days, k = 1"),
            (f"rhlv{stem}First", f"{first.year}", f"first print {first.date()}"),
            (f"rhlv{stem}LiveYear", f"{first_live.year}", f"value average with k = 1 live from {first_live.date()}"),
        ]
        print(
            f"{stem:9s} first {first.date()} last {last.date()} k=1 live {first_live.date()} "
            f"scored {min(ev):5.1f}-{max(ev):5.1f}%  option days {min(op):5.1f}-{max(op):5.1f}%"
        )
    lines = [
        "% AUTO-GENERATED by writeup/make_rehaul_live_windows.py -- do not edit; re-run the script.",
        "% \\rhlv... macros: the share of scored bars on which each late-starting series' value averages are live under",
        "% the late-start rule of Section 4 (step 7); Lo = longest horizon k, Hi = k = 1.",
    ]
    lines += [f"\\providecommand{{\\{n}}}{{{v}}}  % {c}" for n, v, c in macros]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.name}: {len(macros)} macros")


if __name__ == "__main__":
    main()
