"""Write writeup/generated/rehaul_live_numbers.tex: when the late-starting series are live under the late-start rule.

Section 4, step 7 (src/backtest/executor.py LATE_START_WARMUP, UNOBSERVED_NEUTRAL): a series whose first print comes
after the first day of the sample has its standardized k-interval value average set to 0 until every row of the
estimation window (W = 39,000 regular-hours bars, 3,000 days of 13 bars) is live, a row being live when its k-bar
average holds no bar before the day of the first print; and set to 0 whenever the k-bar average holds no observed bar
(a stopped feed).

2026-10-25: the liveness is now computed by the executor's own code on the executor's own frames, so it cannot
diverge from the runs.  The trailing averages of the final 13-bar design are computed on the FULL bar grid (lag scope
"global", overnight intervals included), so k counts full-grid bars, while W counts rows of the 13-bar sample:
  * the full frame: load_raw_data("data", allow_missing=True), the observed flags <c>__obs recorded before any fill
    (after apply_overnight_fills, a no-op today, as the runner passes overnight_fill=True), then the rows with an RV
    print (executor.run_executor);
  * the ages <c>__age on that frame (executor._series_age_columns) and the availability averages
    <c>_avail_ma_<k>, the trailing mean of the observed flag, as har.generate_har_features computes every average
    (rolling(k, min_periods=1).mean().shift(1));
  * the 13-bar sample: the HAR burn-in dropped on the bar grid (BURNIN_IN_BAR_SPACE), segment rth13
    (segmentation.slice_to_segment), the early-close sessions dropped (DROP_EARLY_CLOSE_SESSIONS); W from
    segmentation.compute_segment_train_window (3,000 days x the median 13 rows a day);
  * executor.late_start_warmup and executor.unobserved_neutral applied to a matrix of ones: a value is live where it
    is still 1.
Was (2026-10-24): k and the first print were placed on an idealized 13-bar day frame, which overstated the delay of
the long horizons (k = 3,125 bars is about 65 days on the full grid, 240 days on 13 bars).

This script counts, per late series and horizon, the share of the scored regular-hours bars (the 33,527 bars of
Table 1, 2014-01-02 .. 2024-04-30, selected as writeup/make_rehaul_oos_rerun.py does) and of the 16:00 bars of the 614
post-warmup option days that are live; macros give the range over the horizons k (Lo = the least live horizon,
Hi = the most live).  Availability averages are not affected by the rule.

    python writeup/make_rehaul_live_windows.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.backtest import executor as ex  # noqa: E402
from src.backtest.segmentation import compute_segment_train_window, slice_to_segment  # noqa: E402
from src.data.loading import NYSE_EARLY_CLOSE_DATES, apply_overnight_fills, load_raw_data  # noqa: E402

OUT = REPO / "writeup" / "generated" / "rehaul_live_numbers.tex"
TRAIN_DAYS = 3_000  # estimation window of the final design, in days
W = 39_000  # rows of the estimation window: 3,000 days x 13 regular-hours bars (asserted below)
HORIZONS = (1, 5, 25, 125, 625, 3125)  # base-5 HAR horizons of the final design
SPAN = ("2014-01-02", "2024-04-30")  # every-bar span of Table 1
N_DAYS, N_BARS = 2_579, 33_527  # its evaluation days and scored bars (\rhorNIntB), asserted below
N_PW = 614  # post-warmup option days (Section 7), asserted below
WARMUP = 252

# series: macro stem, column(s); the four volatility-demand series share their dates, as do the StockTwits sentiment
# and tag count (checked below: every column of a stem must give the same liveness)
SERIES: list[tuple[str, list[str]]] = [
    ("Vix", ["vix"]),
    ("VixThree", ["vix3m"]),
    ("Vvix", ["vvix"]),
    ("StAtt", ["stocktwits_attention"]),
    ("StSent", ["stocktwits_sentiment", "stocktwits_sentcount"]),
    (
        "Vd",
        [
            "voldemand_spx_open_and_close",
            "voldemand_spx_open_only",
            "voldemand_all_open_and_close",
            "voldemand_all_open_only",
        ],
    ),
]


def option_days() -> pd.DatetimeIndex:
    """The 614 post-warmup option days (the 866 option days of the master table, after the first 252)."""
    d = pd.read_parquet(REPO / "results" / "close_master_table" / "master_table_daily.parquet", columns=["day"])
    days = pd.DatetimeIndex(pd.to_datetime(d["day"]).dt.normalize().drop_duplicates().sort_values())
    assert len(days) == 866, len(days)
    pw = days[WARMUP:]
    assert len(pw) == N_PW, len(pw)
    return pw


def sample_frame(cols: list[str]) -> pd.DataFrame:
    """The 13-bar sample of the final runs, with the ages and availability averages of ``cols``, as the executor
    builds it (see the module docstring)."""
    df = load_raw_data(str(REPO / "data"), allow_missing=True)
    apply_overnight_fills(df, cols)
    for c in cols:
        df[f"{c}__obs"] = df[c].notna().to_numpy(copy=True)
    df = df[["t", "RV", *[f"{c}__obs" for c in cols]]].dropna(subset=["RV"]).reset_index(drop=True)
    df = ex._series_age_columns(df, [f"adj_{c}" for c in cols])
    for c in cols:
        flag = df[f"{c}__obs"].astype("float64")
        for k in HORIZONS:
            df[f"{c}_avail_ma_{k}"] = flag.rolling(window=k, min_periods=1).mean().shift(1)
    assert ex.BURNIN_IN_BAR_SPACE and ex.DROP_EARLY_CLOSE_SESSIONS
    df = df.iloc[max(HORIZONS) :].reset_index(drop=True)
    seg = slice_to_segment(df, "rth13")
    keep = ~pd.to_datetime(seg["t"]).dt.normalize().isin(pd.to_datetime(list(NYSE_EARLY_CLOSE_DATES)))
    return seg.loc[keep.to_numpy()].reset_index(drop=True)


def _range(xs: list[float]) -> str:
    """'lo--hi' at one decimal, or the single value when both ends print the same."""
    lo, hi = f"{min(xs):.1f}", f"{max(xs):.1f}"
    return lo if lo == hi else f"{lo}--{hi}"


def main() -> None:
    assert ex.LATE_START_WARMUP and ex.UNOBSERVED_NEUTRAL
    cols = [c for _, cs in SERIES for c in cs]
    seg = sample_frame(cols)
    assert compute_segment_train_window(seg["t"], TRAIN_DAYS) == W
    t = pd.to_datetime(seg["t"])
    day = t.dt.normalize()
    # the scored bars of Table 1 (writeup/make_rehaul_oos_rerun.py evaluation_days, scored_set): the days of the span
    # with a positive closing-half-hour print, early closes out, and on them the bars with a positive print
    c = pd.read_parquet(REPO / "data" / "core_stats.parquet", columns=["endbartime", "sumret2"])
    rv = c.set_index(pd.to_datetime(c["endbartime"]))["sumret2"].sort_index()
    close = rv[(rv.index.hour == 16) & (rv.index.minute == 0)]
    eval_days = pd.DatetimeIndex(close.index.normalize())[(close.to_numpy() > 0) & np.isfinite(close.to_numpy())]
    eval_days = eval_days[(eval_days >= pd.Timestamp(SPAN[0])) & (eval_days <= pd.Timestamp(SPAN[1]))]
    eval_days = eval_days[~eval_days.isin(pd.to_datetime(list(NYSE_EARLY_CLOSE_DATES)))]
    assert len(eval_days) == N_DAYS, len(eval_days)
    rv_t = rv.reindex(pd.DatetimeIndex(t)).to_numpy()
    in_span = (day.isin(eval_days).to_numpy()) & (rv_t > 0) & np.isfinite(rv_t)
    assert in_span.sum() == N_BARS, in_span.sum()
    at_1600 = ((t.dt.hour == 16) & (t.dt.minute == 0) & day.isin(option_days())).to_numpy()
    assert at_1600.sum() == N_PW, at_1600.sum()
    print(f"13-bar sample: {len(seg)} rows, {in_span.sum()} scored bars in {SPAN[0]}..{SPAN[1]}, W = {W}")
    macros: list[tuple[str, str, str]] = []
    for stem, cs in SERIES:
        firsts = {c: day[seg[f"{c}__obs"].to_numpy(dtype=bool)].min() for c in cs}
        assert all(f"{c}__age" in seg.columns for c in cs), stem  # every series starts after the frame's first day
        first = pd.Timestamp(next(iter(firsts.values())))
        ev, op = [], []
        first_live = None
        for k in HORIZONS:
            names = [f"adj_{c}_ma_{k}" for c in cs]
            X = np.ones((len(seg), len(names)))
            X = ex.late_start_warmup(X, seg, names, W)
            X = ex.unobserved_neutral(X, seg, names)
            live = X != 0.0
            assert (live == live[:, :1]).all(), (stem, k)  # the columns of a stem share their liveness
            live = live[:, 0]
            ev.append(100.0 * float(live[in_span].mean()))
            op.append(100.0 * float(live[at_1600].mean()))
            if k == 1:
                first_live = day.iloc[int(np.argmax(live))] if live.any() else None
        assert first_live is not None, stem
        macros += [
            (f"rhlv{stem}EvLo", f"{min(ev):.1f}", f"% live, scored bars {SPAN[0]}..{SPAN[1]}, least live horizon"),
            (f"rhlv{stem}EvHi", f"{max(ev):.1f}", "% live, scored bars, most live horizon"),
            (f"rhlv{stem}OptLo", f"{min(op):.1f}", "% live, 16:00 bars of the 614 post-warmup option days, least"),
            (f"rhlv{stem}OptHi", f"{max(op):.1f}", "% live, 16:00 bars of the 614 post-warmup option days, most"),
            # 2026-10-25: the range as printed, a single number when the horizons agree (attention: 100.0 on the option days)
            (f"rhlv{stem}EvRange", _range(ev), "% live, scored bars, least--most live horizon"),
            (f"rhlv{stem}OptRange", _range(op), "% live, 16:00 bars of the option days, least--most live horizon"),
            (f"rhlv{stem}First", f"{first.year}", f"first print {first.date()}"),
            (f"rhlv{stem}LiveYear", f"{first_live.year}", f"value average with k = 1 live from {first_live.date()}"),
        ]
        print(
            f"{stem:9s} first {first.date()} k=1 live {first_live.date()} "
            f"scored {' '.join(f'{v:5.1f}' for v in ev)}  option days {' '.join(f'{v:5.1f}' for v in op)}"
        )
    lines = [
        "% AUTO-GENERATED by writeup/make_rehaul_live_windows.py -- do not edit; re-run the script.",
        "% \\rhlv... macros: the share of scored bars on which each late-starting series' value averages are live under",
        "% the late-start rule of Section 4 (step 7); Lo = the least live horizon k, Hi = the most live.",
    ]
    lines += [f"\\providecommand{{\\{n}}}{{{v}}}  % {c}" for n, v, c in macros]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.name}: {len(macros)} macros")


if __name__ == "__main__":
    main()
