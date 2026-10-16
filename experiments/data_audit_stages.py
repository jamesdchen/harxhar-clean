"""Analysis stages of experiments/data_audit_2026_10_15.py (the data problems register, 2026-10-15).

Each stage reads data / the designs read-only and writes CSVs (and figures) to
results/data_audit_2026-10-15/.  Run through the main script:
    python experiments/data_audit_2026_10_15.py <stage>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "experiments", REPO / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import data_audit_2026_10_15 as A  # noqa: E402

OUT, WORK = A.OUT, A.WORK
TEST0, TEST1 = pd.Timestamp("2020-01-03"), pd.Timestamp("2024-04-30 23:59")
FC0 = pd.Timestamp("2018-06-25")  # the per-bar design's first forecast row
DECK = REPO / "results" / "atm_straddle_0dte_1530" / "daily_blk2.parquet"


def say(msg: str) -> None:
    A.say(msg)


def raw_panel() -> pd.DataFrame:
    """load_raw_data as the record built it (invalid prints NOT nulled), cached."""
    f = WORK / "panel_old.pkl"
    if f.is_file():
        return pd.read_pickle(f)
    import src.data.loading as LD

    LD.NULL_INVALID_PRINTS = False
    P = LD.load_raw_data(str(A.DATA), allow_missing=True)
    P.to_pickle(f)
    return P


def trade_days() -> pd.DatetimeIndex:
    d = pd.read_parquet(DECK)
    return pd.DatetimeIndex(pd.to_datetime(d.index)).normalize()


def hhmm(t: pd.Series) -> pd.Series:
    return pd.to_datetime(t).dt.strftime("%H:%M")


def bars_1530_rows(P: pd.DataFrame, days: pd.DatetimeIndex) -> np.ndarray:
    """Row positions of the 15:30 bar of each day (the bar the 16:00 row's ma_1 reads)."""
    t = pd.to_datetime(P["t"])
    m = (hhmm(t) == "15:30").to_numpy()
    s = pd.Series(np.flatnonzero(m), index=t[m].dt.normalize().to_numpy())
    return s.reindex(days).to_numpy()


# ============================================================================ item 2: feeds
def feeds() -> None:
    import src.data.loading as LD

    P = raw_panel()
    t = pd.to_datetime(P["t"]).reset_index(drop=True)
    days = trade_days()
    r1530 = bars_1530_rows(P, days)
    assert np.isfinite(r1530).all()
    r1530 = r1530.astype(int)
    t16 = t[(hhmm(t) == "16:00").to_numpy()]
    s16 = t16[(t16 >= TEST0) & (t16 <= TEST1)]
    src = {}
    for f in sorted(A.DATA.glob("*.parquet")):
        for c in pd.read_parquet(f).columns:
            src.setdefault("RV" if c == "sumret2" else c, f.stem)
    rows, gaps = [], []
    n = len(P)
    pos = np.arange(n)
    for c in ["RV"] + list(LD.ALL_FEATURES):
        obs = P[c].notna().to_numpy()
        idx = np.flatnonzero(obs)
        first, last = t[idx[0]], t[idx[-1]]
        lastpos = pd.Series(np.where(obs, pos, np.nan)).ffill().to_numpy()
        age = pos - lastpos  # bars since the last observed print (0 = observed)
        age_days = (
            t.to_numpy() - t.to_numpy()[np.nan_to_num(lastpos, nan=0).astype(int)]
        ) / np.timedelta64(1, "D")
        a_tr = age[r1530]
        ad_tr = age_days[r1530]
        dead = days > last.normalize()
        o = obs[idx[0] : idx[-1] + 1]
        d = np.diff(np.r_[1, o.astype(int), 1])
        st, en = np.flatnonzero(d == -1), np.flatnonzero(d == 1)
        L = en - st
        for s_, e_, l_ in zip(st, en, L):
            if l_ > 48:
                gaps.append(
                    dict(
                        series=c,
                        source=src.get(c, "derived"),
                        gap_start=str(t[idx[0] + s_])[:16],
                        gap_end=str(t[idx[0] + e_ - 1])[:16],
                        bars=int(l_),
                        in_test_window=bool(t[idx[0] + s_] >= TEST0),
                    )
                )
        # sessions of the test window whose 16:00 row reads an unobserved 15:30 bar
        rows.append(
            dict(
                series=c,
                source=src.get(
                    c,
                    "derived ("
                    + ("releases" if c.startswith("fomc") else "stock files")
                    + ")",
                ),
                first_print=str(first)[:16],
                last_print=str(last)[:16],
                days_short_of_panel_end=round(
                    (A.PANEL_END - last) / pd.Timedelta("1D"), 1
                ),
                coverage_rv_grid=round(float(obs.mean()), 4),
                trade_days_after_last_print=int(dead.sum()),
                sessions_1600_after_last_print_test_window=int(
                    (s16.dt.normalize() > last.normalize()).sum()
                ),
                trade_days_1530_unobserved=int((a_tr > 0).sum()),
                trade_days_1530_unobserved_live_feed=int(((a_tr > 0) & ~dead).sum()),
                trade_days_age_gt_26_bars=int((a_tr > 26).sum()),
                max_age_days_on_trade_days=round(float(np.nanmax(ad_tr)), 1),
                gaps_gt48_bars=int((L > 48).sum()),
                gaps_gt48_bars_test_window=0,
                max_gap_bars=int(L.max()) if len(L) else 0,
            )
        )
    R = pd.DataFrame(rows)
    G = pd.DataFrame(gaps)
    if len(G):
        cnt = G[G["in_test_window"]].groupby("series").size()
        R["gaps_gt48_bars_test_window"] = R["series"].map(cnt).fillna(0).astype(int)
    R.to_csv(OUT / "feeds.csv", index=False)
    # one row per gap and source family (the stock files' 24 series share their gaps)
    if len(G):
        G["family"] = np.where(
            G["source"].str.contains("stock"),
            "stock files",
            np.where(G["series"].str.startswith("voldemand"), "voldemand", G["series"]),
        )
        G.drop_duplicates(["family", "gap_start", "gap_end"]).drop(
            columns=["series"]
        ).to_csv(OUT / "feed_gaps.csv", index=False)
    # whole NYSE sessions missing from a live feed (a trade-calendar day: a weekday whose ES tape
    # prints the 12:00 and 16:00 bars and is not an early close)
    import atm_straddle_lib as asl

    day = t.dt.normalize()
    has = lambda m: set(day[m.to_numpy() if hasattr(m, "to_numpy") else m])  # noqa: E731
    es_days = has((hhmm(t) == "16:00") & P["RV"].notna()) & has(
        (hhmm(t) == "12:00") & P["RV"].notna()
    )
    ec = set(pd.to_datetime(list(asl.EARLY_CLOSE_DATES)))
    rth = (
        (t.dt.hour * 60 + t.dt.minute >= 600) & (t.dt.hour * 60 + t.dt.minute <= 960)
    ).to_numpy()
    miss = []
    for fam, col, start in (
        ("stock files", "sumret2_vwstock", "1998-01-05"),
        ("vix", "vix", "2006-05-22"),
        ("vvix", "vvix", "2012-03-26"),
        ("vix3m", "vix3m", "2009-08-12"),
        ("voldemand", "voldemand_spx_open_only", "2011-01-03"),
        ("stocktwits", "stocktwits_attention", "2009-07-01"),
    ):
        last = t[P[col].notna().to_numpy()].max().normalize()
        live = has(P[col].notna() & rth)
        for d_ in sorted(es_days - ec):
            if pd.Timestamp(start) <= d_ <= last and d_ not in live:
                miss.append(
                    dict(
                        family=fam,
                        day=str(d_.date()),
                        is_trade_day=bool(d_ in set(days)),
                        in_per_bar_window=bool(d_ >= pd.Timestamp("2010-07-12")),
                    )
                )
    M = pd.DataFrame(miss)
    M.to_csv(OUT / "feed_missing_sessions.csv", index=False)
    # the release calendar (the FOMC dead tail is a stale file, not a stopped feed)
    rel = pd.read_parquet(A.DATA / "releases.parquet")
    rt = pd.to_datetime(rel["endbartime"])
    cal = [
        dict(
            release=c,
            events=int((rel[c] > 0).sum()),
            first=str(rt[rel[c] > 0].min())[:16],
            last=str(rt[rel[c] > 0].max())[:16],
        )
        for c in rel.columns
        if c.endswith("release")
    ]
    cal.append(
        dict(
            release="fomc release: scheduled statements missing in the panel span",
            events=len(A.FOMC_EXTRA),
            first=A.FOMC_EXTRA[0],
            last=A.FOMC_EXTRA[-1],
        )
    )
    pd.DataFrame(cal).to_csv(OUT / "release_calendar.csv", index=False)
    # the Yahoo hourly snapshot vs the panel's own prints where both exist (splice check)
    v = pd.read_parquet(A.DATA / "vix_and_voldemand.parquet")
    v["t"] = pd.to_datetime(v["endbartime"])
    yo = []
    for col in ("vix", "vix3m", "vvix"):
        ys = A.yahoo_bar_ends(col)
        s = v.set_index("t")[col].dropna()
        j = s.to_frame("panel").join(ys.rename("yahoo"), how="inner")
        rel_ = np.abs(j["yahoo"] / j["panel"] - 1)
        last = s.index.max()
        tail = ys[(ys.index > last) & (ys.index <= A.PANEL_END)]
        yo.append(
            dict(
                series=col,
                overlap_stamps=len(j),
                overlap_first=str(j.index.min())[:16],
                exact_share=float((np.abs(j["yahoo"] - j["panel"]) < 0.006).mean()),
                median_rel_diff=float(rel_.median()),
                p99_rel_diff=float(rel_.quantile(0.99)),
                tail_hourly_stamps=len(tail),
                tail_first=str(tail.index.min())[:16],
                tail_last=str(tail.index.max())[:16],
            )
        )
    pd.DataFrame(yo).to_csv(OUT / "vix_yahoo_overlap.csv", index=False)
    # how the design of record encodes the dead tails at 16:00 (avail / active / adj ladders)
    stale_design()
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(R.to_string())
        print(
            M.groupby("family").agg(
                n=("day", "size"),
                trade_days=("is_trade_day", "sum"),
                per_bar=("in_per_bar_window", "sum"),
            )
        )
        print(pd.DataFrame(yo).to_string())


def stale_design() -> None:
    """Design of record, 16:00 rows: for each dead-tail series, every column's tail values against
    its range over the 2,000 training sessions before the tail (extrapolation) and window_keep."""
    from src.models.window_mask import window_keep

    d = pd.read_parquet(A.RECORD_DESIGN)
    dt = pd.to_datetime(d["date"])
    names = [c for c in d.columns if c not in A.DESIGN_META]
    X = d[names].to_numpy(np.float64)
    days = set(trade_days())
    tails = {
        "vix": "2024-02-12 17:00",
        "vvix": "2024-02-12 16:00",
        "vix3m": "2024-02-12 16:30",
        "voldemand_spx_open_and_close": "2023-08-31 19:00",
        "voldemand_spx_open_only": "2023-08-31 19:00",
        "voldemand_all_open_and_close": "2023-08-31 19:00",
        "voldemand_all_open_only": "2023-08-31 19:00",
        "fomc_release": "2023-11-01 14:30",
        "fomc_day": "2023-11-01 14:30",
        "fomc_until_inv": "2023-11-01 14:30",
        "fomc_since_inv": "2023-11-01 14:30",
    }
    rows = []
    for s, last in tails.items():
        tail = np.asarray(dt > pd.Timestamp(last))
        i0 = int(np.flatnonzero(tail)[0])
        win = slice(max(0, i0 - 2000), i0)
        keep = set(window_keep(X[win]).tolist())
        trade = np.asarray([x.normalize() in days for x in dt])
        for j, c in enumerate(names):
            if not (
                c == f"adj_{s}"
                or c.startswith(f"adj_{s}_ma_")
                or c.startswith(f"{s}_avail_ma_")
                or c.startswith(f"{s}_active_ma_")
            ):
                continue
            w = X[win, j]
            tv = X[tail & trade, j]
            lo, hi = float(w.min()), float(w.max())
            iqr = float(np.subtract(*np.percentile(w, [75, 25]))) or 1.0
            out = (tv < lo - 1e-12) | (tv > hi + 1e-12)
            rows.append(
                dict(
                    series=s,
                    column=c,
                    tail_trade_days=int(len(tv)),
                    kept_in_last_window=bool(j in keep),
                    train_min=lo,
                    train_max=hi,
                    train_median=float(np.median(w)),
                    tail_mean=float(tv.mean()),
                    tail_min=float(tv.min()),
                    tail_max=float(tv.max()),
                    tail_days_outside_train_range=int(out.sum()),
                    tail_mean_shift_in_train_iqr=float(
                        (tv.mean() - np.median(w)) / iqr
                    ),
                )
            )
    pd.DataFrame(rows).to_csv(OUT / "stale_tail_design.csv", index=False)


# ============================================================================ item 3: activity flags
def active() -> None:
    import src.backtest.executor as EX
    import src.data.loading as LD

    P = raw_panel()
    t = pd.to_datetime(P["t"]).reset_index(drop=True)
    days = trade_days()
    r1530 = bars_1530_rows(P, days).astype(int)
    rows = []
    for c in LD.ALL_FEATURES:
        x = P[c]
        obs = x.notna().to_numpy()
        ff = x.ffill().to_numpy(np.float64)
        infl, mode = EX.mode_inflated(ff)
        fin = ff[~np.isnan(ff)]
        share_ff = float(np.mean(fin == mode)) if fin.size else np.nan
        xo = x.dropna().to_numpy()
        mode_o = EX.modal_value(xo)
        share_o = float(np.mean(xo == mode_o)) if xo.size else np.nan
        r = dict(
            series=c,
            mode_inflated=bool(infl),
            mode=float(mode),
            mode_share_ffilled=share_ff,
            mode_share_observed=share_o,
            mode_observed=float(mode_o),
            inflated_if_observed_only=bool(share_o >= EX.ZERO_INFLATED_FRAC),
        )
        if infl:
            act_ff = (np.nan_to_num(ff, nan=mode) != mode).astype(float)
            act_obs = act_ff[obs]
            neutral = float(np.round(act_obs.mean()))
            act_fix = np.where(obs, act_ff, neutral)
            act_zero = np.where(obs, act_ff, 0.0)
            dif = act_ff != act_fix
            difz = act_ff != act_zero
            r.update(
                observed_share=float(obs.mean()),
                active_share_observed=float(act_obs.mean()),
                neutral_mode_of_indicator=neutral,
                rows_unobserved=int((~obs).sum()),
                rows_carried_active=int(((~obs) & (act_ff == 1)).sum()),
                rows_changed_mode_rule=int(dif.sum()),
                rows_changed_zero_rule=int(difz.sum()),
                rows_changed_mode_rule_from_2018_06=int(
                    dif[np.asarray(t >= FC0)].sum()
                ),
                rows_changed_mode_rule_test_window=int(
                    dif[np.asarray((t >= TEST0) & (t <= TEST1))].sum()
                ),
                trade_days_1530_changed_mode_rule=int(dif[r1530].sum()),
                trade_days_1530_changed_zero_rule=int(difz[r1530].sum()),
            )
        rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "active_flags.csv", index=False)
    with pd.option_context("display.width", 250):
        print(R[R["mode_inflated"] | (R["mode_share_observed"] > 0.1)].to_string())


# ============================================================================ item 5: look-ahead
def lookahead() -> None:  # noqa: C901 -- one table per element
    import src.backtest.executor as EX
    import src.data.loading as LD
    import src.features.transforms.target as TG

    P = raw_panel()
    LD.NULL_INVALID_PRINTS = False
    TG.SIGN_BY_NAME = False
    df = P.dropna(subset=["RV"]).reset_index(drop=True)
    t = pd.to_datetime(df["t"])
    cols = list(LD.ALL_FEATURES)
    obs = {c: df[c].notna().to_numpy() for c in cols}
    F = df[cols].ffill()
    F["time_of_day"] = df["time_of_day"]
    days = trade_days()
    r1530 = bars_1530_rows(df, days).astype(int)
    test = np.asarray((t >= TEST0) & (t <= TEST1))
    # (i) the neutral median: full sample vs causal (expanding, observed rows strictly before)
    med_rows, floor_rows, route_rows, mode_rows = [], [], [], []
    for c in cols:
        adj, base = TG.robust_transform(F, c, use_transform=True, use_diurnal=True)
        a_obs = adj.where(pd.Series(obs[c], index=adj.index))
        full = float(a_obs.median())
        cm = A.causal_fill(a_obs).to_numpy()
        iqr = float(np.subtract(*np.nanpercentile(a_obs, [75, 25]))) or 1.0
        uno = ~obs[c]
        med_rows.append(
            dict(
                series=c,
                full_sample_median=full,
                causal_median_at_2018_06_25=float(cm[np.searchsorted(t, FC0)]),
                causal_median_at_2020_01_03=float(cm[np.searchsorted(t, TEST0)]),
                causal_median_at_end=float(cm[-1]),
                max_abs_gap_test_window_in_iqr=float(
                    np.max(np.abs(cm[test & uno] - full)) / iqr
                )
                if (test & uno).any()
                else 0.0,
                mean_abs_gap_test_window_in_iqr=float(
                    np.mean(np.abs(cm[test & uno] - full)) / iqr
                )
                if (test & uno).any()
                else 0.0,
                unobserved_rows_test_window=int((test & uno).sum()),
                trade_days_1530_unobserved=int(uno[r1530].sum()),
            )
        )
        # (ii) the signed-sd floor and (iii) the asinh routing (signed, diurnally adjusted)
        if TG.series_is_signed(c, F[c]) and not TG.is_diurnal_excluded(c):
            g = pd.DataFrame({"val": F[c], "slot": F["time_of_day"]}).groupby("slot")[
                "val"
            ]
            sd = g.transform(
                lambda s: (
                    s.rolling(TG.DIURNAL_WINDOW, min_periods=TG.DIURNAL_MIN_PERIODS)
                    .std()
                    .shift(1)
                )
            )
            typical = float(sd.replace(0, np.nan).abs().median())
            floor_full = TG.DIURNAL_STD_FLOOR_FRAC * typical
            causal_typ = (
                sd.replace(0, np.nan).abs().expanding(min_periods=50).median().shift(1)
            )
            floor_c = (
                (TG.DIURNAL_STD_FLOOR_FRAC * causal_typ).fillna(floor_full).to_numpy()
            )
            band = TG._band_to_slot_level(
                sd, F["time_of_day"], TG.DIURNAL_SLOT_BAND_FRAC
            )
            b_full = band.clip(lower=floor_full).fillna(floor_full).to_numpy()
            b_c = np.maximum(band.to_numpy(), floor_c)
            b_c = np.where(np.isnan(band.to_numpy()), floor_c, b_c)
            ch = ~np.isclose(b_full, b_c, rtol=1e-12, atol=0)
            pinned_full = float(np.isclose(b_full, b_full.min(), rtol=1e-9).mean())
            floor_rows.append(
                dict(
                    series=c,
                    floor_full_sample=floor_full,
                    floor_causal_at_2018_06_25=float(floor_c[np.searchsorted(t, FC0)]),
                    floor_causal_at_2020_01_03=float(
                        floor_c[np.searchsorted(t, TEST0)]
                    ),
                    rows_floor_binds_full=int(
                        (band.fillna(-np.inf).to_numpy() < floor_full).sum()
                    ),
                    rows_divisor_changed=int(ch.sum()),
                    rows_divisor_changed_test_window=int((ch & test).sum()),
                    trade_days_1530_divisor_changed=int(ch[r1530].sum()),
                    has_name_stabilizer=TG.has_name_stabilizer(c),
                )
            )
            if not TG.has_name_stabilizer(c):
                # the routing share on the full sample (the shipped rule) vs on each causal prefix
                for cut in (FC0, TEST0, pd.Timestamp("2022-01-03"), None):
                    m = (
                        np.asarray(t < cut)
                        if cut is not None
                        else np.ones(len(t), bool)
                    )
                    bb = b_full[m]
                    share = (
                        float(np.isclose(bb, bb.min(), rtol=1e-9).mean())
                        if bb.size
                        else np.nan
                    )
                    route_rows.append(
                        dict(
                            series=c,
                            data_up_to=str(cut.date())
                            if cut is not None
                            else "full sample (shipped)",
                            pinned_share=share,
                            routed_to_asinh=bool(share >= TG.DIURNAL_PINNED_MAX),
                            shipped_routing=bool(pinned_full >= TG.DIURNAL_PINNED_MAX),
                        )
                    )
        # (iv) the mode-inflated class and its mode (on the forward-filled raw column, full sample)
        vals = F[c].to_numpy(np.float64)
        infl, mode = EX.mode_inflated(vals)
        for cut in (FC0, TEST0, None):
            m = np.asarray(t < cut) if cut is not None else np.ones(len(t), bool)
            i2, m2 = EX.mode_inflated(vals[m])
            fin = vals[m][~np.isnan(vals[m])]
            if infl or i2:
                mode_rows.append(
                    dict(
                        series=c,
                        data_up_to=str(cut.date())
                        if cut is not None
                        else "full sample (shipped)",
                        mode_inflated=bool(i2),
                        mode=float(m2),
                        share=float(np.mean(fin == m2)) if fin.size else np.nan,
                        same_as_shipped=bool(i2 == infl and (m2 == mode or not i2)),
                    )
                )
    pd.DataFrame(med_rows).to_csv(OUT / "lookahead_neutral_median.csv", index=False)
    pd.DataFrame(floor_rows).to_csv(OUT / "lookahead_signed_sd_floor.csv", index=False)
    pd.DataFrame(route_rows).to_csv(OUT / "lookahead_asinh_routing.csv", index=False)
    pd.DataFrame(mode_rows).to_csv(OUT / "lookahead_mode_inflated.csv", index=False)
    lookahead_guard()


def lookahead_guard() -> None:
    import src.backtest.executor as EX

    # (v) the X-column mode test inside _build_scale_guards and the first scaling window
    z = A.load_npz(A.OLD_LOCAL / "design_bar1600_all_features.npz")
    names = [str(x) for x in z["names"]]
    W = int(z["W"])
    dates = pd.to_datetime(z["date"])
    xr = []
    g = sorted(WORK.glob("guards_abcd_erode_3469x*.npz"))
    if g:  # the unscaled matrix the guard saw
        gz = np.load(g[0])
        Xu = gz["X"]
        for j, c in enumerate(names):
            if not c.startswith("adj_"):
                continue
            i_full, m_full = EX.mode_inflated(Xu[:, j])
            i_pre, m_pre = EX.mode_inflated(Xu[:W, j])
            if i_full or i_pre:
                xr.append(
                    dict(
                        column=c,
                        inflated_full=bool(i_full),
                        mode_full=float(m_full),
                        inflated_first_window=bool(i_pre),
                        mode_first_window=float(m_pre),
                        differs=bool(i_full != i_pre or (i_full and m_full != m_pre)),
                    )
                )
    pd.DataFrame(xr).to_csv(OUT / "lookahead_guard_mode.csv", index=False)
    first = dict(
        rows_scaled_by_first_window=W,
        first_window=f"{dates[0].date()} .. {dates[W - 1].date()}",
        first_forecast=str(dates[W].date()),
        test_rows_scaled_by_first_window=0,
        forecasts_whose_training_window_holds_such_rows=int(min(len(dates) - W, W)),
        latest_data_used_by_first_window_stats=str(dates[W - 1].date()),
    )
    pd.DataFrame([first]).to_csv(OUT / "lookahead_first_window.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        for f in ("lookahead_guard_mode", "lookahead_first_window"):
            print(f)
            print(pd.read_csv(OUT / f"{f}.csv").to_string())


# ============================================================================ item 6: erosion units
def erosion() -> None:
    gs = sorted(WORK.glob("guards_abcd_erode_*.npz"))
    rows = []
    for g in gs:
        z = np.load(g)
        names = [str(x) for x in z["names"]]
        dates = pd.to_datetime(z["date"])
        seg = "bar1600" if dates.strftime("%H:%M").nunique() == 1 else "last30"
        W = 2000 if seg == "bar1600" else 4000
        fr = np.zeros(len(dates), bool)
        fr[W:] = True
        test = np.asarray(dates >= TEST0)
        rs, rb = z["ref_sessions"], z["ref_bars"]
        for j, c in enumerate(names):
            if not c.startswith("adj_"):
                continue
            a, b = rs[:, j], rb[:, j]
            if not (a.any() or b.any()):
                continue
            rows.append(
                dict(
                    segment=seg,
                    column=c,
                    test_rows_warmup_floor_sessions=int(((a == 1.0) & test).sum()),
                    test_rows_warmup_floor_bars=int(((b == 1.0) & test).sum()),
                    median_ref_test_sessions=float(np.median(a[test])),
                    median_ref_test_bars=float(np.median(b[test])),
                    test_rows_ref_changed=int(
                        (~np.isclose(a, b, rtol=1e-12) & test).sum()
                    ),
                )
            )
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "erosion_ref_iqr.csv", index=False)
    if len(R):
        print(
            R.groupby("segment")[
                [
                    "test_rows_warmup_floor_sessions",
                    "test_rows_warmup_floor_bars",
                    "test_rows_ref_changed",
                ]
            ].agg(["sum", lambda x: int((x > 0).sum())])
        )


# ============================================================================ item 8: scoring target
def scoring() -> None:
    from src.evaluation.diebold_mariano import dm_test

    m = pd.read_csv(REPO / "results" / "close_master_table" / "master_table.csv")
    a = m[m["table"] == "A"].copy()
    a = a[a["key"] != "always_short"]
    # 2026-10-15 (#10 adopted): the master table's qlike_raw is now actual RV and the old trimmed
    # value is qlike_raw_trimmed (qlike_raw_unclipped is gone); this stage measured the change
    # BEFORE #10 and its CSVs are the record of that measurement
    trim, unclip = (
        ("qlike_raw_trimmed", "qlike_raw")
        if "qlike_raw_trimmed" in a.columns
        else ("qlike_raw", "qlike_raw_unclipped")
    )
    a["rank_clipped"] = a[trim].rank()
    a["rank_unclipped"] = a[unclip].rank()
    rho = float(a[[trim, unclip]].corr(method="spearman").iloc[0, 1])
    days = trade_days()
    daily = pd.read_parquet(
        REPO / "results" / "close_master_table" / "master_table_daily.parquet",
        columns=["key", "day", "plain", "target", "rv_unclipped"],
    )
    daily["day"] = pd.to_datetime(daily["day"]).dt.normalize()
    ref = daily[daily["key"] == "blk2"].set_index("day").reindex(days)
    rv_u = ref["rv_unclipped"].to_numpy()
    clip = ref["target"] / ref["rv_unclipped"]

    def ql(y, f):
        r = y / f
        return r - np.log(r) - 1

    q_ref_c = ql(ref["target"].to_numpy(), ref["plain"].to_numpy())
    q_ref_u = ql(rv_u, ref["plain"].to_numpy())
    rows = []
    for k, r in a.iterrows():
        dk = daily[daily["key"] == r["key"]].set_index("day").reindex(days)
        if dk["plain"].isna().any():
            continue
        qc = ql(dk["target"].to_numpy(), dk["plain"].to_numpy())
        qu = ql(rv_u, dk["plain"].to_numpy())
        dmc = dm_test(qc, q_ref_c)["dm"] if r["key"] != "blk2" else np.nan
        dmu = dm_test(qu, q_ref_u)["dm"] if r["key"] != "blk2" else np.nan
        rows.append(
            dict(
                key=r["key"],
                label=r["label"],
                family=r["family"],
                is_headline=r.get("is_headline"),
                qlike_clipped=float(qc.mean()),
                qlike_unclipped=float(qu.mean()),
                rank_clipped=r["rank_clipped"],
                rank_unclipped=r["rank_unclipped"],
                dm_vs_blk2_clipped=dmc,
                dm_vs_blk2_unclipped=dmu,
                sign_vs_blk2_flips=bool(np.sign(dmc) != np.sign(dmu))
                if np.isfinite(dmc)
                else False,
                significance_flips=bool((abs(dmc) > 1.96) != (abs(dmu) > 1.96))
                if np.isfinite(dmc)
                else False,
            )
        )
    R = pd.DataFrame(rows)
    R["rank_clipped"] = R["qlike_clipped"].rank()
    R["rank_unclipped"] = R["qlike_unclipped"].rank()
    R["rank_shift"] = R["rank_unclipped"] - R["rank_clipped"]
    R.to_csv(OUT / "scoring_target.csv", index=False)
    top = R.nsmallest(20, "qlike_clipped")
    summ = dict(
        forecasts=len(R),
        spearman_qlike=float(
            R[["qlike_clipped", "qlike_unclipped"]].corr("spearman").iloc[0, 1]
        ),
        master_spearman_table_A=rho,
        trade_days_target_clipped=int((np.abs(clip - 1) > 1e-12).sum()),
        trade_days_clipped_down=int((clip < 1 - 1e-12).sum()),
        trade_days_clipped_up=int((clip > 1 + 1e-12).sum()),
        median_clip_ratio_when_clipped=float(clip[np.abs(clip - 1) > 1e-12].median()),
        top20_same_set=int(
            len(set(top["key"]) & set(R.nsmallest(20, "qlike_unclipped")["key"]))
        ),
        max_abs_rank_shift_top20=float(top["rank_shift"].abs().max()),
        dm_sign_flips=int(R["sign_vs_blk2_flips"].sum()),
        dm_significance_flips=int(R["significance_flips"].sum()),
        mean_qlike_clipped=float(R["qlike_clipped"].mean()),
        mean_qlike_unclipped=float(R["qlike_unclipped"].mean()),
    )
    pd.DataFrame([summ]).to_csv(OUT / "scoring_target_summary.csv", index=False)
    print(json.dumps(summ, indent=1, default=str))
    print(
        top[
            [
                "key",
                "qlike_clipped",
                "qlike_unclipped",
                "rank_clipped",
                "rank_unclipped",
                "dm_vs_blk2_clipped",
                "dm_vs_blk2_unclipped",
            ]
        ].to_string()
    )


# ============================================================================ item 13: grid rules
def grid() -> None:  # noqa: C901 -- one table of facts
    import atm_straddle_lib as asl

    c = pd.read_parquet(A.DATA / "core_stats.parquet")
    c["t"] = pd.to_datetime(c["endbartime"])
    c = c[c["sumret2"].notna()].reset_index(drop=True)
    t = c["t"]
    dow, hm, yr = t.dt.dayofweek, hhmm(t), t.dt.year
    tod = t.dt.time
    facts = []

    def fact(name, n, detail=""):
        facts.append(dict(check=name, n=int(n), detail=detail))

    cut = (
        ((dow == 4) & (tod > pd.Timestamp("20:00").time()))
        | (dow == 5)
        | ((dow == 6) & (tod < pd.Timestamp("18:30").time()))
    )
    fact(
        "RV prints removed by the weekend cut",
        cut.sum(),
        "; ".join(str(x)[:16] for x in t[cut]),
    )
    fri = (dow == 4) & (tod > pd.Timestamp("17:00").time()) & ~cut
    fact(
        "RV bars kept on Friday after the 17:00 ET CME close (bar ends 17:30..20:00)",
        fri.sum(),
        f"{int(fri[yr >= 2010].sum())} from 2010; median volume {c.loc[fri, 'sumvolume'].median():.0f} vs "
        f"{c.loc[(dow < 4) & (hm == '19:00'), 'sumvolume'].median():.0f} at Mon-Thu 19:00; median numobs "
        f"{c.loc[fri, 'numobs'].median():.0f}",
    )
    halt = (dow < 4) & hm.isin(["17:30", "18:00"])
    fact(
        "RV bars kept in the daily 17:00-18:00 ET maintenance window (bar ends 17:30, 18:00, Mon-Thu)",
        halt.sum(),
        f"median volume 17:30 {c.loc[halt & (hm == '17:30'), 'sumvolume'].median():.0f}, "
        f"18:00 {c.loc[halt & (hm == '18:00'), 'sumvolume'].median():.0f}; median numobs "
        f"{c.loc[halt, 'numobs'].median():.0f}",
    )
    fact(
        "RV == 0 bars kept",
        (c["sumret2"] == 0).sum(),
        f"at 16:00: {int(((c['sumret2'] == 0) & (hm == '16:00')).sum())}",
    )
    # NYSE sessions: a stock-file or VIX regular-hours print
    v = pd.read_parquet(
        A.DATA / "vwstock_stats.parquet", columns=["endbartime", "sumret2_vwstock"]
    )
    v["t"] = pd.to_datetime(v["endbartime"])
    vm = v["sumret2_vwstock"].notna() & (
        v["t"].dt.hour * 60 + v["t"].dt.minute
    ).between(600, 960)
    nyse = set(v.loc[vm, "t"].dt.normalize())
    vx = pd.read_parquet(
        A.DATA / "vix_and_voldemand.parquet", columns=["endbartime", "vix"]
    )
    vx["t"] = pd.to_datetime(vx["endbartime"])
    nyse |= set(
        vx.loc[
            vx["vix"].notna() & (vx["t"].dt.hour.between(10, 15)), "t"
        ].dt.normalize()
    )
    s16 = t[hm == "16:00"].dt.normalize()
    s16set = set(s16)
    ec = set(pd.to_datetime(list(asl.EARLY_CLOSE_DATES)))
    hol = sorted(s16set - nyse)
    fact(
        "16:00 RV rows on days with no NYSE session (no stock-file or VIX RTH print)",
        len(hol),
        "; ".join(str(d.date()) for d in hol[:40]),
    )
    fact(
        "... of which inside the per-bar design span (2010-07-12 .. 2024-04-30)",
        sum(d >= pd.Timestamp("2010-07-12") for d in hol),
        "; ".join(str(d.date()) for d in hol if d >= pd.Timestamp("2010-07-12")),
    )
    ecr = sorted(s16set & ec)
    fact(
        "16:00 RV rows on NYSE early-close days (post-13:00 futures bars; target = post-close variance)",
        len(ecr),
        "; ".join(str(d.date()) for d in ecr),
    )
    fact(
        "... of which inside the per-bar design span (training rows of the per-bar models)",
        sum(d >= pd.Timestamp("2010-07-12") for d in ecr),
        "; ".join(str(d.date()) for d in ecr if d >= pd.Timestamp("2010-07-12")),
    )
    fact(
        "... of which inside the scored window 2020-01-03 .. 2024-04-30 (dropped from the 866 trade days)",
        sum(TEST0 <= d <= TEST1 for d in ecr),
        "; ".join(str(d.date()) for d in ecr if TEST0 <= d <= TEST1),
    )
    nyse_days = sorted(
        d
        for d in nyse
        if d.dayofweek < 5 and d not in ec and d >= pd.Timestamp("1998-02-01")
    )
    missing16 = [d for d in nyse_days if d not in s16set]
    fact(
        "NYSE full sessions without an ES 16:00 bar (the session is absent from the per-bar design)",
        len(missing16),
        "; ".join(str(d.date()) for d in missing16[:60]),
    )
    tt = set(t)
    gaps = [
        d + pd.Timedelta(minutes=30 * k)
        for d in nyse_days
        for k in range(20, 33)
        if d + pd.Timedelta(minutes=30 * k) not in tt
    ]
    G = pd.Series(gaps, dtype="datetime64[ns]")
    late = G[G >= pd.Timestamp("2010-07-12")]
    fact(
        "RTH bars (10:00..16:00) missing from the ES tape on NYSE full sessions",
        len(G),
        f"on {G.dt.normalize().nunique() if len(G) else 0} days; from 2010-07-12: {len(late)}: "
        + "; ".join(str(x)[:16] for x in late[:20]),
    )
    yy = (
        pd.DataFrame({"year": yr, "fri_after_close": fri, "maint_window": halt})
        .groupby("year")
        .sum()
    )
    yy.to_csv(OUT / "grid_closed_window_bars_by_year.csv")
    F = pd.DataFrame(facts)
    F.to_csv(OUT / "grid_audit.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 250):
        print(F.to_string())


# ============================================================================ item 12: time alignment
def _dst(ts) -> np.ndarray:
    loc = pd.to_datetime(pd.Series(ts)).dt.tz_localize(
        "America/New_York", ambiguous=True, nonexistent="shift_forward"
    )
    return np.array([bool(x.dst()) for x in loc])


def _transition_week(days) -> np.ndarray:
    """The 7 days after a US DST change (second Sunday of March, first Sunday of November; 2007+)."""
    d = pd.to_datetime(pd.Series(days)).dt.normalize()
    out = np.zeros(len(d), bool)
    for y in range(2007, 2026):
        mar = pd.Timestamp(f"{y}-03-01")
        sun_mar = mar + pd.Timedelta(days=(6 - mar.dayofweek) % 7 + 7)
        nov = pd.Timestamp(f"{y}-11-01")
        sun_nov = nov + pd.Timedelta(days=(6 - nov.dayofweek) % 7)
        for s in (sun_mar, sun_nov):
            out |= ((d > s) & (d <= s + pd.Timedelta(days=7))).to_numpy()
    return out


def align() -> None:  # noqa: C901
    import atm_straddle_lib as asl

    rows = []

    def row(check, n, value, detail=""):
        rows.append(dict(check=check, n=n, value=value, detail=detail))

    deck = pd.read_parquet(DECK)
    days = pd.DatetimeIndex(pd.to_datetime(deck.index)).normalize()
    ts = pd.to_datetime(deck["timestamp_c"])
    et = ts.dt.tz_convert("America/New_York")
    utc_h = ts.dt.tz_convert("UTC").dt.strftime("%H:%M")
    dst = np.array([bool(x.dst()) for x in et])
    exp_utc = np.where(dst, "19:30", "20:30")
    row(
        "trade days whose 15:30 ET entry stamp has the UTC clock of its DST state (19:30 EDT / 20:30 EST)",
        len(deck),
        int((utc_h.to_numpy() == exp_utc).sum()),
    )
    row(
        "trade days with the vendor's hours_to_expiration = 0.5 at the entry stamp",
        len(deck),
        int((deck["hours_to_expiration_c"] == 0.5).sum()),
    )
    ec = set(pd.to_datetime(list(asl.EARLY_CLOSE_DATES)))
    row(
        "trade days on an NYSE early-close date",
        len(deck),
        int(sum(d in ec for d in days)),
    )
    ch = pd.read_parquet(
        REPO / "data" / "spxw_chain.parquet",
        columns=["expiration", "timestamp", "early_close", "hours_to_expiration"],
    )
    ch["exp"] = pd.to_datetime(ch["expiration"]).dt.normalize()
    if getattr(ch["exp"].dt, "tz", None) is not None:
        ch["exp"] = ch["exp"].dt.tz_localize(None)
    ch["et"] = pd.to_datetime(ch["timestamp"]).dt.tz_convert("America/New_York")
    e = ch.groupby("exp").agg(early=("early_close", "max"), last_et=("et", "max"))
    e = e[(e.index >= TEST0) & (e.index <= TEST1)]
    flag = set(e.index[e["early"].astype(bool)])
    row("expirations 2020-01-03..2024-04-30 in the chain", len(e), len(e))
    cal = {d for d in ec if TEST0 <= d <= TEST1}
    row(
        "chain early_close flags vs the NYSE calendar (2020-2024): symmetric difference",
        len(flag),
        int(len(flag ^ (cal & set(e.index)))),
        "flagged: " + "; ".join(str(d.date()) for d in sorted(flag)),
    )
    v = pd.read_parquet(
        A.DATA / "vwstock_stats.parquet",
        columns=["endbartime", "sumret2_vwstock", "sumret3_vwstock"],
    )
    v["t"] = pd.to_datetime(v["endbartime"])
    nyse = set(v.loc[v["sumret2_vwstock"].notna(), "t"].dt.normalize())
    non = [d for d in e.index if d not in nyse]
    row(
        "expirations on a day with no stock-file print (non-NYSE day)",
        len(e),
        len(non),
        "; ".join(str(d.date()) for d in non),
    )
    hte = ch.loc[ch["exp"].isin(days), ["et", "hours_to_expiration", "exp"]].copy()
    hte["implied_close"] = hte["et"].dt.tz_localize(None) + pd.to_timedelta(
        hte["hours_to_expiration"], unit="h"
    )
    bad = hte[hte["implied_close"].dt.round("5min").dt.strftime("%H:%M") != "16:00"]
    row(
        "chain rows on trade days whose stamp + hours_to_expiration is not 16:00 ET",
        len(hte),
        len(bad),
        "; ".join(sorted(set(str(x.date()) for x in bad["exp"]))[:20]),
    )
    # ES vs spot: 30-minute returns at lags (bar-end labels)
    c = pd.read_parquet(
        A.DATA / "core_stats.parquet", columns=["endbartime", "sumret", "sumret2"]
    )
    c["t"] = pd.to_datetime(c["endbartime"])
    es = c.set_index("t")["sumret"].dropna()
    sp = pd.read_parquet(REPO / "data" / "spxw_spot.parquet")
    sp["et"] = (
        pd.to_datetime(sp["timestamp"])
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )
    sp = (
        sp[sp["et"].dt.minute.isin([0, 30])]
        .drop_duplicates("et")
        .set_index("et")["spot"]
        .sort_index()
    )
    r = np.log(sp).diff()
    gap = (sp.index.to_series().diff() == pd.Timedelta("30min")).to_numpy()
    r = r[gap]
    lag_rows = []
    tw = _transition_week(r.index)
    dstm = _dst(r.index)
    for k in (-2, -1, 0, 1, 2):
        e_k = es.reindex(r.index + pd.Timedelta(minutes=30 * k)).to_numpy()
        ok = np.isfinite(e_k) & np.isfinite(r.to_numpy())
        for lab, m in (
            ("all", ok),
            ("EDT", ok & dstm),
            ("EST", ok & ~dstm),
            ("DST transition weeks", ok & tw),
        ):
            lag_rows.append(
                dict(
                    pair="SPX spot (chain) vs ES sumret",
                    lag_bars=k,
                    subset=lab,
                    n=int(m.sum()),
                    corr=float(np.corrcoef(r.to_numpy()[m], e_k[m])[0, 1])
                    if m.sum() > 10
                    else np.nan,
                )
            )
    vv = v.set_index("t")["sumret3_vwstock"].dropna()
    mins = vv.index.hour * 60 + vv.index.minute
    vv = vv[(mins >= 600) & (mins <= 960)]
    dst_v = _dst(vv.index)
    tw_v = _transition_week(vv.index)
    x = np.cbrt(vv.to_numpy())
    for k in (-2, -1, 0, 1, 2):
        e_k = es.reindex(vv.index + pd.Timedelta(minutes=30 * k)).to_numpy()
        ok = np.isfinite(e_k)
        for lab, m in (
            ("all", ok),
            ("EDT", ok & dst_v),
            ("EST", ok & ~dst_v),
            ("DST transition weeks", ok & tw_v),
        ):
            lag_rows.append(
                dict(
                    pair="VW stock cube-root sumret3 vs ES sumret",
                    lag_bars=k,
                    subset=lab,
                    n=int(m.sum()),
                    corr=float(np.corrcoef(x[m], e_k[m])[0, 1]),
                )
            )
    L = pd.DataFrame(lag_rows)
    L.to_csv(OUT / "align_lags.csv", index=False)
    # each DST-transition week separately: where does the spot / ES correlation peak?
    wk = []
    rr = r.to_frame("r")
    for k in (-2, 0, 2):
        rr[f"es{k}"] = es.reindex(r.index + pd.Timedelta(minutes=30 * k)).to_numpy()
    rr["tw"] = tw
    rr["week"] = rr.index.to_period("W").astype(str)
    for w_, g in rr[rr["tw"]].groupby("week"):
        cs = {k: g[["r", f"es{k}"]].dropna().corr().iloc[0, 1] for k in (-2, 0, 2)}
        wk.append(
            dict(
                week=w_,
                n=len(g),
                corr_lag0=cs[0],
                corr_lag_m2=cs[-2],
                corr_lag_p2=cs[2],
                best_lag=max(cs, key=lambda k: -np.inf if np.isnan(cs[k]) else cs[k]),
            )
        )
    WK = pd.DataFrame(wk)
    WK.to_csv(OUT / "align_dst_weeks.csv", index=False)
    row(
        "DST-transition weeks 2020-2025 whose spot/ES correlation peaks at lag 0",
        len(WK),
        int((WK["best_lag"] == 0).sum()),
    )
    vw_wk = []
    vf = pd.DataFrame({"x": x}, index=vv.index)
    for k in (-2, 0, 2):
        vf[f"es{k}"] = es.reindex(vv.index + pd.Timedelta(minutes=30 * k)).to_numpy()
    vf["tw"] = tw_v
    vf["week"] = vf.index.to_period("W").astype(str)
    for w_, g in vf[vf["tw"]].groupby("week"):
        cs = {k: g[["x", f"es{k}"]].dropna().corr().iloc[0, 1] for k in (-2, 0, 2)}
        vw_wk.append(
            dict(
                week=w_,
                best_lag=max(cs, key=lambda k: -np.inf if np.isnan(cs[k]) else cs[k]),
                corr_lag0=cs[0],
            )
        )
    VW = pd.DataFrame(vw_wk)
    VW.to_csv(OUT / "align_dst_weeks_stockfile.csv", index=False)
    row(
        "DST-transition weeks 2007-2024 whose stock-file/ES correlation peaks at lag 0",
        len(VW),
        int((VW["best_lag"] == 0).sum()),
        "not at lag 0: " + "; ".join(VW.loc[VW["best_lag"] != 0, "week"].tolist()[:20]),
    )
    # the 15:30 / 16:00 join on the trade days
    mv = np.log(deck["S_close"] / deck["S"]).to_numpy()
    for k, lab in ((0, "16:00 bar (15:30-16:00)"), (-1, "15:30 bar"), (1, "16:30 bar")):
        e_k = es.reindex(
            days + pd.Timedelta(hours=16) + pd.Timedelta(minutes=30 * k)
        ).to_numpy()
        ok = np.isfinite(e_k)
        row(
            f"corr(log S_close / S at 15:30, ES sumret of the {lab}) on trade days",
            int(ok.sum()),
            round(float(np.corrcoef(mv[ok], e_k[ok])[0, 1]), 4),
        )
    tw_d = _transition_week(days)
    e0 = es.reindex(days + pd.Timedelta(hours=16)).to_numpy()
    row(
        "... the 16:00 bar, DST-transition-week trade days only",
        int(tw_d.sum()),
        round(float(np.corrcoef(mv[tw_d], e0[tw_d])[0, 1]), 4),
    )
    rv16 = c.set_index("t")["sumret2"].reindex(days + pd.Timedelta(hours=16)).to_numpy()
    row(
        "median of ES RV(16:00 bar) / (log S_close/S)^2 on trade days",
        len(days),
        round(float(np.nanmedian(rv16 / np.maximum(mv**2, 1e-12))), 3),
    )
    have = lambda s, hm_: set(  # noqa: E731
        pd.to_datetime(s.index[s.index.strftime("%H:%M") == hm_]).normalize()
    )  # noqa: E731
    es16 = have(c.set_index("t")["sumret2"].dropna(), "16:00")
    vs = v.set_index("t")["sumret2_vwstock"].dropna()
    v16, v1530 = have(vs, "16:00"), have(vs, "15:30")
    row(
        "trade days without an ES 16:00 bar",
        len(days),
        int(sum(d not in es16 for d in days)),
    )
    row(
        "trade days without a stock-file 15:30 or 16:00 bar",
        len(days),
        int(sum((d not in v16) or (d not in v1530) for d in days)),
    )
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "align_checks.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 160):
        print(R.to_string())
        print(
            L.pivot_table(index=["pair", "subset"], columns="lag_bars", values="corr")
            .round(3)
            .to_string()
        )
        print(WK.to_string())


# ============================================================================ item 11: bad ticks
KNOWN_EVENTS = {  # dates (ET) of documented market events, for the classification of candidates
    "2001-09-17": "9/11 reopening",
    "2008-09-29": "TARP vote",
    "2008-10-10": "GFC",
    "2008-10-13": "GFC",
    "2008-10-24": "GFC",
    "2010-05-06": "flash crash",
    "2011-08-08": "US downgrade",
    "2013-04-23": "AP hack",
    "2014-10-15": "Treasury flash rally",
    "2015-08-24": "ETF crash open",
    "2016-06-24": "Brexit",
    "2016-11-09": "US election",
    "2018-02-05": "Volmageddon",
    "2018-02-06": "Volmageddon",
    "2018-02-08": "Volmageddon",
    "2020-03-09": "COVID",
    "2020-03-12": "COVID",
    "2020-03-16": "COVID",
    "2020-03-18": "COVID",
    "2020-03-20": "COVID quad witch",
    "2022-11-10": "CPI",
    "2024-01-31": "FOMC",
}
TICK_SERIES = {
    # name: (file stem, transform, sibling series at the same bar)
    "RV": ("core_stats", "log", ("sumret2_vwstock", "sumret2_ewstock")),
    "sumabsret": ("core_stats", "log", ("sumabsret_vwstock",)),
    "sumvolume": ("core_stats", "log", ("turnover_vwstock",)),
    "sumret2_vwstock": ("vwstock_stats", "log", ("sumret2_ewstock", "RV")),
    "sumret2_ewstock": ("ewstock_stats", "log", ("sumret2_vwstock", "RV")),
    "sumret4_vwstock": ("vwstock_stats", "log", ("sumret4_ewstock",)),
    "sumret4_ewstock": ("ewstock_stats", "log", ("sumret4_vwstock",)),
    "sumabsret_vwstock": ("vwstock_stats", "log", ("sumabsret_ewstock", "sumabsret")),
    "turnover_vwstock": ("vwstock_stats", "log", ("turnover_ewstock",)),
    "turnover_ewstock": ("ewstock_stats", "log", ("turnover_vwstock",)),
    "spread_vwstock": ("vwstock_stats", "log", ("spread_ewstock",)),
    "spread_ewstock": ("ewstock_stats", "log", ("spread_vwstock",)),
    "effspread_vwstock": ("vwstock_stats", "log", ("effspread_ewstock",)),
    "effspread_ewstock": ("ewstock_stats", "log", ("effspread_vwstock",)),
    "vix": ("vix_and_voldemand", "log", ("vix3m",)),
    "vix3m": ("vix_and_voldemand", "log", ("vix",)),
    "vvix": ("vix_and_voldemand", "log", ("vix",)),
    "stocktwits_attention": ("spy_and_sentiment", "log", ()),
}


def _tick_frame() -> pd.DataFrame:
    frames = {}
    for f in (
        "core_stats",
        "vwstock_stats",
        "ewstock_stats",
        "vix_and_voldemand",
        "spy_and_sentiment",
    ):
        d = pd.read_parquet(A.DATA / f"{f}.parquet")
        d["t"] = pd.to_datetime(d["endbartime"])
        frames[f] = d.drop(columns=["endbartime"]).set_index("t")
    X = pd.concat(frames.values(), axis=1)
    X = X.rename(columns={"sumret2": "RV"})
    return X[X.index >= pd.Timestamp("1998-01-05")]


def _slot_z(v: pd.Series, slot, win: int = 250, minp: int = 60) -> pd.Series:
    """Robust z of v against the same clock slot's trailing win observations (strictly before)."""
    g = v.groupby(slot)
    med = g.transform(lambda q: q.rolling(win, min_periods=minp).median().shift(1))
    dev = (v - med).abs()
    mad = dev.groupby(slot).transform(
        lambda q: q.rolling(win, min_periods=minp).median().shift(1)
    )
    return (v - med) / (1.4826 * mad.where(mad > 0))


def ticks(top: int = 15) -> None:  # noqa: C901
    """Isolated extreme prints, by the signature each family's errors leave:
    ES        a 1-minute print out and back inside the bar: RV extreme for its slot while the bar's
              first-order autocovariance carries most of it with a negative sign (autocov / RV < -0.35)
    stocks    the cross-section's RV (EW, VW) extreme RELATIVE TO ES's RV for the slot while ES is
              ordinary (one stock's bad print moves a weighted average of per-stock RVs; ES does not move)
    VIX       a level jump reverted at the next bar (log level, both legs > 6 robust sd)
    spreads   quoted / effective spread extreme for its slot and isolated from both neighbours, the
              other weighting ordinary"""
    X = _tick_frame()
    slot = X.index.strftime("%H:%M")
    mins = X.index.hour * 60 + X.index.minute
    rth = (mins >= 600) & (mins <= 960)
    rows = []

    def add(fam, s, t, z, extra, cls):
        prev = X[s].loc[:t].dropna().iloc[-2:-1]
        nxt = X[s].loc[t:].dropna().iloc[1:2]
        ev = KNOWN_EVENTS.get(str(t.date()), "")
        rows.append(
            dict(
                family=fam,
                series=s,
                t=str(t)[:16],
                value=float(X.at[t, s]),
                prev_value=float(prev.iloc[0]) if len(prev) else np.nan,
                next_value=float(nxt.iloc[0]) if len(nxt) else np.nan,
                score=float(z),
                evidence=extra,
                known_event=ev,
                classification=f"market event ({ev})" if ev else cls,
                in_per_bar_span=bool(t >= pd.Timestamp("2010-07-12")),
                in_test_window=bool(TEST0 <= t <= TEST1),
            )
        )

    # ---- ES
    lrv = np.log(X["RV"].where(X["RV"] > 0))
    z_rv = _slot_z(lrv, slot)
    ac = X["sumautocov"] / X["RV"]
    q_es = np.log(X["RV"] / X["sumret2_vwstock"])
    z_q_es = _slot_z(q_es.where(rth), slot)
    cand = (z_rv > 5) & (ac < -0.35)
    for t in z_rv[cand].sort_values(ascending=False).index:
        zs = float(z_q_es.get(t, np.nan))
        cls = (
            "likely data error (print out and back inside the bar)"
            if (ac[t] < -0.4 and (not np.isfinite(zs) or zs > 3))
            else "unclear"
        )
        add(
            "ES",
            "RV",
            t,
            z_rv[t],
            f"autocov/RV {ac[t]:.2f}; RV vs stocks z {zs:.1f}",
            cls,
        )
    # ---- stock files: RV relative to ES's
    for w in ("vwstock", "ewstock"):
        q = np.log(X[f"sumret2_{w}"].where(X[f"sumret2_{w}"] > 0) / X["RV"])
        zq = _slot_z(q, slot)
        other = "ewstock" if w == "vwstock" else "vwstock"
        qo = _slot_z(
            np.log(X[f"sumret2_{other}"].where(X[f"sumret2_{other}"] > 0) / X["RV"]),
            slot,
        )
        cand = (zq > 6) & (z_rv < 3)
        X[f"_flag_{w}"] = cand.fillna(False)
        for t in zq[cand].sort_values(ascending=False).index:
            cls = "likely data error (cross-section spikes, ES ordinary)"
            add(
                f"stock files ({w})",
                f"sumret2_{w}",
                t,
                zq[t],
                f"ES RV slot z {z_rv[t]:.1f}; {other} relative z {qo.get(t, np.nan):.1f}",
                cls,
            )
    # ---- VIX family: jump and revert
    for s in ("vix", "vix3m", "vvix"):
        v = np.log(X[s]).dropna()
        d1, d2 = v.diff(), -v.diff(-1)
        sd = 1.4826 * (v.diff() - v.diff().median()).abs().median()
        rev = (
            (np.sign(d1) != np.sign(d2))
            & (np.minimum(d1.abs(), d2.abs()) > 6 * sd)
            & ((v.shift(-1) - v.shift(1)).abs() < 0.25 * np.minimum(d1.abs(), d2.abs()))
        )
        for t in v[rev].index:
            add(
                "VIX family",
                s,
                t,
                float(np.minimum(abs(d1[t]), abs(d2[t])) / sd),
                f"jump {d1[t]:+.3f}, revert {-d2[t]:+.3f} (log)",
                "likely data error (level jumps and reverts)",
            )
    # ---- spreads
    for s, o in (
        ("spread_vwstock", "spread_ewstock"),
        ("spread_ewstock", "spread_vwstock"),
        ("effspread_vwstock", "effspread_ewstock"),
        ("effspread_ewstock", "effspread_vwstock"),
    ):
        lv = np.log(X[s].where(X[s] > 0))
        zs, zo = _slot_z(lv, slot), _slot_z(np.log(X[o].where(X[o] > 0)), slot)
        o_ = lv.dropna()
        iso = (
            pd.concat(
                [
                    o_ - np.maximum(o_.shift(1), o_.shift(-1)),
                    np.minimum(o_.shift(1), o_.shift(-1)) - o_,
                ],
                axis=1,
            )
            .max(axis=1)
            .reindex(X.index)
        )
        cand = (zs.abs() > 8) & (iso > np.log(3)) & (zo.abs() < 3)
        for t in zs[cand].abs().sort_values(ascending=False).index:
            add(
                "spreads",
                s,
                t,
                zs[t],
                f"{o} slot z {zo[t]:.1f}; isolation x{np.exp(iso[t]):.1f}",
                "likely data error (one weighting spikes alone)",
            )
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "bad_tick_candidates.csv", index=False)
    yr = pd.to_datetime(R["t"]).dt.year
    by = R.assign(year=yr).pivot_table(
        index="year", columns="family", values="t", aggfunc="count", fill_value=0
    )
    by.to_csv(OUT / "bad_tick_by_year.csv")
    T = R.sort_values("score", ascending=False).groupby("family").head(top)
    T.to_csv(OUT / "bad_tick_top.csv", index=False)
    k = R[R["t"] == "2018-11-12 11:00"]
    print(
        "known example (2018-11-12 11:00):",
        k[["family", "series", "score", "evidence", "classification"]].to_string(),
    )
    print(by.to_string())
    with pd.option_context("display.width", 260, "display.max_colwidth", 70):
        print(
            T[
                [
                    "family",
                    "series",
                    "t",
                    "value",
                    "prev_value",
                    "next_value",
                    "score",
                    "evidence",
                    "classification",
                ]
            ].to_string()
        )
    _tick_figure(X, T.groupby("family").head(3).head(9))
    # test-window 16:00 targets touched by an ES candidate (the target itself)
    es16 = R[
        (R["family"] == "ES") & (R["t"].str.endswith("16:00")) & R["in_test_window"]
    ]
    print("ES candidates on 16:00 bars in the test window:", len(es16))


def _tick_figure(X: pd.DataFrame, T: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(T)
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for ax, (_, r) in zip(axes.ravel(), T.iterrows()):
        t = pd.Timestamp(r["t"])
        s = r["series"]
        w = X[s].loc[t - pd.Timedelta(hours=8) : t + pd.Timedelta(hours=8)].dropna()
        ax.plot(w.index, w.to_numpy(), marker=".", lw=1, color="#1f5fa8", label=s)
        sibs = {
            "RV": "sumret2_vwstock",
            "sumret2_vwstock": "RV",
            "sumret2_ewstock": "RV",
            "vix": "vix3m",
            "vix3m": "vix",
            "vvix": "vix",
            "spread_vwstock": "spread_ewstock",
            "spread_ewstock": "spread_vwstock",
            "effspread_vwstock": "effspread_ewstock",
            "effspread_ewstock": "effspread_vwstock",
        }
        for k_ in [sibs.get(s)] if sibs.get(s) else []:
            ws = X[k_].loc[w.index.min() : w.index.max()].dropna()
            if len(ws):
                ax2 = ax.twinx()
                ax2.plot(
                    ws.index, ws.to_numpy(), lw=1, color="#c0504d", alpha=0.7, label=k_
                )
                ax2.set_yscale("log")
                ax2.tick_params(labelsize=6)
        ax.axvline(t, color="k", lw=0.5, ls=":")
        ax.set_yscale("log")
        ax.set_title(f"{s} {r['t']}\n{r['classification'][:40]}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(
        "Top bad-tick candidates (blue: series, red: first sibling; log scale; +-8 h)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "bad_tick_top.png", dpi=130)


# ============================================================================ item 9: pooled duplicates
POOLED_CACHE = (
    REPO.parent / "harxhar-clean" / "results" / "prep_cache_all_features_b2.npz"
)
POOLED_W = 24_000
BURNIN = 3125


def _pooled():
    import zipfile

    sys.path.insert(0, str(REPO / "experiments" / "puffer_forecast"))
    import prep_pooled as PP

    X = PP.member_memmap(POOLED_CACHE, "X.npy")
    with zipfile.ZipFile(POOLED_CACHE) as zf:
        with zf.open("names.npy") as fh:
            names = [str(x) for x in np.lib.format.read_array(fh, allow_pickle=True)]
        with zf.open("y.npy") as fh:
            y = np.lib.format.read_array(fh)
        with zf.open("baselines.npy") as fh:
            B = np.lib.format.read_array(fh)
    P = raw_panel()
    t = pd.to_datetime(P["t"]).to_numpy()[BURNIN:][: len(y)]
    assert len(t) == len(y) == X.shape[0]
    return X, names, y, B, pd.DatetimeIndex(t)


def _kind(name: str) -> str:
    if "_avail_ma_" in name:
        return "avail ladder"
    if "_active_ma_" in name:
        return "active ladder"
    if name.startswith("adj_"):
        return "value ladder"
    return "backbone"


def _groups(Xw: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    import hashlib

    Xw = np.ascontiguousarray(np.asarray(Xw, dtype=np.float64).T)
    bits = Xw.view(np.uint64)
    const = (bits == bits[:, :1]).all(axis=1)
    seen: dict[bytes, list[int]] = {}
    for j in np.flatnonzero(~const):
        seen.setdefault(hashlib.sha1(Xw[j].tobytes()).digest(), []).append(int(j))
    return const, [g for g in seen.values() if len(g) > 1]


def dups(n_windows: int = 12) -> None:
    X, names, y, B, t = _pooled()
    n = X.shape[0]
    ends = list(np.linspace(POOLED_W, n, n_windows).astype(int))
    for d in (
        "2020-01-03",
        "2021-01-04",
        "2022-01-03",
        "2023-01-03",
        "2024-01-02",
        "2024-04-30",
    ):
        ends.append(int(np.searchsorted(t, pd.Timestamp(d) + pd.Timedelta(hours=16))))
    rows, grp_rows = [], []
    for e in sorted(set(ends)):
        e = min(e, n)
        const, groups = _groups(X[e - POOLED_W : e])
        sizes = [len(g) for g in groups]
        ex = [g for g in groups if all(_kind(names[j]) != "backbone" for j in g)]
        rows.append(
            dict(
                window_end=str(t[e - 1])[:16],
                constant_cols=int(const.sum()),
                dup_groups=len(groups),
                exog_dup_groups=len(ex),
                max_copies=max(sizes) if sizes else 0,
                redundant_cols=int(sum(s - 1 for s in sizes)),
                groups_by_kind=json.dumps(
                    pd.Series([_kind(names[g[0]]) for g in groups])
                    .value_counts()
                    .to_dict()
                ),
            )
        )
        for g in groups:
            grp_rows.append(
                dict(
                    window_end=str(t[e - 1])[:16],
                    copies=len(g),
                    kind=_kind(names[g[0]]),
                    members=";".join(names[j] for j in g[:30]),
                )
            )
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "pooled_dup_windows.csv", index=False)
    G = pd.DataFrame(grp_rows)
    G.to_csv(OUT / "pooled_dup_groups.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 140):
        print(R.to_string())
        last = G[G["window_end"] == G["window_end"].iloc[-1]].sort_values(
            "copies", ascending=False
        )
        print(last.head(40).to_string())


def blk2dedup(every: int = 1) -> None:  # noqa: C901
    """blk2 (backbone x10 under alpha 100 = backbone 1 / exog 100; window 24,000; unpenalized intercept)
    replicated by a rolling Gram on the 16:00 rows of the 866 trade days, with and without the
    window's byte-identical exogenous copies (first kept, as window_keep)."""
    import src.unification as U

    X, names, y, B, t = _pooled()
    n, p = X.shape
    bb = np.asarray(U._backbone_cols(names))
    ex = np.asarray(U._exog_all_cols(names))
    cols = np.r_[bb, ex]
    scale = np.r_[
        np.full(len(bb), np.sqrt(U.USER_ALPHAS["exog"] / U.USER_ALPHAS["backbone"])),
        np.ones(len(ex)),
    ]
    alpha = U.USER_ALPHAS["exog"]
    days = trade_days()
    rows_t = np.searchsorted(t, days + pd.Timedelta(hours=16))
    assert (t[rows_t] == days + pd.Timedelta(hours=16)).all()
    rows_t = rows_t[::every]
    days = days[::every]
    m = len(cols)
    # block statistics for the window identity test (exact: identical columns -> identical sums)
    Bk = 1000
    rng = np.random.default_rng(0)
    w1, w2 = rng.standard_normal(n), rng.standard_normal(n)
    nb = (n + Bk - 1) // Bk
    S1 = np.zeros((nb, m))
    S2 = np.zeros((nb, m))
    MX = np.full((nb, m), -np.inf)
    MN = np.full((nb, m), np.inf)
    for b in range(nb):
        blk = np.asarray(X[b * Bk : (b + 1) * Bk][:, cols], dtype=np.float64)
        S1[b] = (blk * w1[b * Bk : b * Bk + len(blk), None]).sum(axis=0)
        S2[b] = (blk * w2[b * Bk : b * Bk + len(blk), None]).sum(axis=0)
        MX[b], MN[b] = blk.max(axis=0), blk.min(axis=0)

    def window_stats(a, e):
        b0, b1 = -(-a // Bk), e // Bk  # full blocks [b0, b1)
        parts = []
        if b0 * Bk > a:
            parts.append((a, min(b0 * Bk, e)))
        if b1 >= b0 and b1 * Bk < e:
            parts.append((max(b1 * Bk, a), e))
        s1 = S1[b0:b1].sum(axis=0) if b1 > b0 else np.zeros(m)
        s2 = S2[b0:b1].sum(axis=0) if b1 > b0 else np.zeros(m)
        mx = MX[b0:b1].max(axis=0) if b1 > b0 else np.full(m, -np.inf)
        mn = MN[b0:b1].min(axis=0) if b1 > b0 else np.full(m, np.inf)
        for r0, r1 in parts:
            blk = np.asarray(X[r0:r1][:, cols], dtype=np.float64)
            s1 = s1 + (blk * w1[r0:r1, None]).sum(axis=0)
            s2 = s2 + (blk * w2[r0:r1, None]).sum(axis=0)
            mx, mn = np.maximum(mx, blk.max(axis=0)), np.minimum(mn, blk.min(axis=0))
        return s1, s2, mx > mn

    def design(r0, r1):
        return np.asarray(X[r0:r1][:, cols], dtype=np.float64) * scale

    G = S = Fy = None
    sy = 0.0
    prev = None
    out = []
    for k, i in enumerate(rows_t):
        a, e = i - POOLED_W, i
        if G is None or k % 50 == 0:
            F = design(a, e)
            G, S, Fy, sy = F.T @ F, F.sum(axis=0), F.T @ y[a:e], float(y[a:e].sum())
        else:
            pa, pe = prev
            Fa, Fr = design(pe, e), design(pa, a)
            G += Fa.T @ Fa - Fr.T @ Fr
            S += Fa.sum(axis=0) - Fr.sum(axis=0)
            Fy += Fa.T @ y[pe:e] - Fr.T @ y[pa:a]
            sy += float(y[pe:e].sum() - y[pa:a].sum())
        prev = (a, e)
        xt = np.asarray(X[i][cols], dtype=np.float64) * scale
        s1, s2, live = window_stats(a, e)
        keep = np.ones(m, bool)
        seen = {}
        nd = 0
        for j in range(m):
            if j < len(bb) or not live[j]:
                continue
            key = (s1[j], s2[j])
            if key in seen:
                keep[j] = False
                nd += 1
            else:
                seen[key] = j
        res = {}
        for lab, kp in (("full", np.ones(m, bool)), ("dedup", keep)):
            idx = np.flatnonzero(kp)
            Aug = np.empty((len(idx) + 1, len(idx) + 1))
            Aug[0, 0] = POOLED_W
            Aug[0, 1:] = Aug[1:, 0] = S[idx]
            Aug[1:, 1:] = G[np.ix_(idx, idx)] + alpha * np.eye(len(idx))
            rhs = np.r_[sy, Fy[idx]]
            sol = np.linalg.solve(Aug, rhs)
            res[lab] = float(sol[0] + xt[idx] @ sol[1:])
        out.append(
            dict(
                day=days[k],
                row=int(i),
                yhat_full=res["full"],
                yhat_dedup=res["dedup"],
                dropped_copies=nd,
                baseline=float(B[i]),
                y=float(y[i]),
            )
        )
    O = pd.DataFrame(out)  # noqa: E741
    pub = pd.read_parquet(REPO / "results" / "spxw_pnl" / "yhat_blk2.parquet")
    pt = pd.to_datetime(pub["t"]).dt.tz_convert("America/New_York").dt.tz_localize(None)
    ps = pd.Series(pub["yhat"].to_numpy(float), index=pd.DatetimeIndex(pt))
    O["yhat_published"] = ps.reindex(O["day"] + pd.Timedelta(hours=16)).to_numpy()
    O.to_parquet(WORK / "blk2_dedup.parquet", index=False)
    gate = float(
        np.nanmax(
            np.abs(O["yhat_full"] - O["yhat_published"]) / np.abs(O["yhat_published"])
        )
    )
    S0 = A.scorer_for(None)
    res = []
    if every == 1:
        dd = {}
        for lab in ("published", "full", "dedup"):
            plain = O[f"yhat_{lab}"].to_numpy() ** 2 * O["baseline"].to_numpy()
            dd[lab] = S0.daily_from_plain(plain)
            st = S0.stats(dd[lab])
            res.append(
                dict(
                    forecast=f"blk2 ({lab})",
                    qlike=st["qlike"],
                    sharpe_mid=st["sharpe_mid"],
                    sharpe_crossed=st["sharpe_crossed"],
                )
            )
        c = S0.contrast(dd["dedup"], dd["full"])
        res.append(
            dict(
                forecast="dedup vs full",
                qlike=c["dq_pct"],
                sharpe_mid=c["dm"],
                sharpe_crossed=c["n_diff_position"],
                note="qlike = % change; sharpe_mid column = DM; sharpe_crossed column = positions changed",
            )
        )
    res.append(
        dict(
            forecast="gate: replicated full vs published yhat_blk2 (max rel diff)",
            qlike=gate,
        )
    )
    res.append(
        dict(
            forecast="exog copies dropped per window (median, max)",
            qlike=float(O["dropped_copies"].median()),
            sharpe_mid=float(O["dropped_copies"].max()),
        )
    )
    res.append(
        dict(
            forecast="forecast change dedup vs full (median, max abs rel)",
            qlike=float(np.median(np.abs(O["yhat_dedup"] / O["yhat_full"] - 1))),
            sharpe_mid=float(np.max(np.abs(O["yhat_dedup"] / O["yhat_full"] - 1))),
        )
    )
    Rr = pd.DataFrame(res)
    Rr.to_csv(OUT / "pooled_blk2_dedup.csv", index=False)
    print(Rr.to_string())


# ============================================================================ item 10: burn-in order
def burnin_n() -> None:
    old = A.load_npz(A.OLD_LOCAL / "design_bar1600_all_features.npz")
    rows = [
        dict(
            design="record (burn-in after the slice)",
            rows=len(old["y"]),
            first=str(old["date"][0])[:10],
            W=int(old["W"]),
            first_forecast=str(old["date"][int(old["W"])])[:10],
        )
    ]
    f = A.design_dir("abcd_burnin") / "design_bar1600_all_features.npz"
    if f.is_file():
        z = A.load_npz(f)
        rows.append(
            dict(
                design="burn-in in bar space (before the slice)",
                rows=len(z["y"]),
                first=str(z["date"][0])[:10],
                W=int(z["W"]),
                first_forecast=str(z["date"][int(z["W"])])[:10],
            )
        )
    q = A.load_npz(A.OLD_LOCAL / "design_last30_all_features.npz")
    d30 = pd.to_datetime(q["date"])
    rows.append(
        dict(
            design="record last-hour design (the pre-tune's source)",
            rows=len(q["y"]),
            first=str(d30[0])[:10],
            W=int(q["W"]),
            first_forecast=str(d30[int(q["W"])])[:10],
        )
    )
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "burnin.csv", index=False)
    print(R.to_string())


# ============================================================================ impact on the affected days
SUBSETS = {  # the trade days a variant can touch directly (the rest move only through refits)
    "vix tail (2024-02-13 .. 04-30)": ("2024-02-13", "2024-04-30"),
    "FOMC tail (2023-11-02 .. 2024-04-30)": ("2023-11-02", "2024-04-30"),
    "voldemand tail (2023-09-01 .. 2024-04-30)": ("2023-09-01", "2024-04-30"),
    "before 2023-09-01": ("2020-01-03", "2023-08-31"),
}


def subsets() -> None:
    """QLIKE and mid P&L of every scored variant against the record on the affected trade days."""
    import atm_straddle_lib as asl

    F = pd.read_parquet(WORK / "linear_forecasts.parquet")
    S0 = A.scorer_for(None)
    days = S0.days
    ec = set(pd.to_datetime(list(asl.EARLY_CLOSE_DATES)))
    # the B-contaminated days: an early close among the previous 20 16:00 sessions
    tg = pd.read_parquet(A.RECORD / "targets.parquet")
    d16 = pd.DatetimeIndex(pd.to_datetime(tg["date"])).normalize()
    e = pd.Series(d16.isin(ec).astype(int), index=d16)
    aff = (e.rolling(20, min_periods=1).sum().shift(1).fillna(0) > 0) & (e == 0)
    bdays = set(aff.index[aff.to_numpy()])
    masks = {
        k: np.asarray((days >= pd.Timestamp(a)) & (days <= pd.Timestamp(b)))
        for k, (a, b) in SUBSETS.items()
    }
    masks["B-contaminated (94)"] = np.asarray([d in bdays for d in days])
    rows = []
    rec = pd.read_parquet(A.OLD_STUDY / "forecasts.parquet")  # noqa: F841
    for (v, s), g in F.groupby(["variant", "series"]):
        est, b = s.split("_", 1)
        b = b.rsplit("_", 1)[0]
        S = A.scorer_for(v)
        fd = pd.DatetimeIndex(pd.to_datetime(g["date"]))
        o = A.record_pred(b, est, fd)
        dn, do = S.daily(g["pred_adj"].to_numpy()), S0.daily(o)
        for k, m in masks.items():
            if not m.any():
                continue
            rows.append(
                dict(
                    variant=v,
                    series=s,
                    subset=k,
                    days=int(m.sum()),
                    qlike_old=float(do["ql"][m].mean()),
                    qlike_new=float(dn["ql"][m].mean()),
                    mid_pnl_old=float(do["mid"][m].sum()),
                    mid_pnl_new=float(dn["mid"][m].sum()),
                    positions_changed=int((dn["q"][m] != do["q"][m]).sum()),
                )
            )
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "linear_subsets.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 400):
        print(R.round(4).to_string())


# ============================================================================ marginal contrasts
def marginals(*pairs: str) -> None:
    """Each fix against the variant it is added to (pairs 'variant:base'; base 'record' = the stored
    forecasts): QLIKE, DM, Sharpe mid / crossed and positions changed on the 866 trade days."""
    F = pd.read_parquet(WORK / "linear_forecasts.parquet")
    rows = []
    for pr in pairs:
        v, b = pr.split(":")
        Sv = A.scorer_for(v)
        Sb = A.scorer_for(None if b == "record" else b)
        for s, g in F[F["variant"] == v].groupby("series"):
            est, bk = s.split("_", 1)
            bk = bk.rsplit("_", 1)[0]
            fd = pd.DatetimeIndex(pd.to_datetime(g["date"]))
            if b == "record":
                pb = A.record_pred(bk, est, fd)
            else:
                gb = F[(F["variant"] == b) & (F["series"] == s)]
                if (
                    gb.empty
                ):  # the base variant has no such bucket (baseline): the record's
                    pb = A.record_pred(bk, est, fd)
                    Sb_ = A.scorer_for(None)
                else:
                    pb = gb["pred_adj"].to_numpy()
                    Sb_ = Sb
            Sb_ = (
                Sb
                if b != "record"
                and not F[(F["variant"] == b) & (F["series"] == s)].empty
                else A.scorer_for(None)
            )
            dv, db = Sv.daily(g["pred_adj"].to_numpy()), Sb_.daily(pb)
            sv, sb = Sv.stats(dv), Sb_.stats(db)
            c = Sv.contrast(dv, db)
            rows.append(
                dict(
                    fix=v,
                    against=b,
                    series=s,
                    qlike_base=sb["qlike"],
                    qlike=sv["qlike"],
                    dq_pct=c["dq_pct"],
                    dm=c["dm"],
                    dm_p=c["dm_p"],
                    sharpe_mid_base=sb["sharpe_mid"],
                    sharpe_mid=sv["sharpe_mid"],
                    d_sharpe_mid=sv["sharpe_mid"] - sb["sharpe_mid"],
                    sharpe_x_base=sb["sharpe_crossed"],
                    sharpe_x=sv["sharpe_crossed"],
                    d_sharpe_x=sv["sharpe_crossed"] - sb["sharpe_crossed"],
                    positions_changed=c["n_diff_position"],
                )
            )
    R = pd.DataFrame(rows)
    f = OUT / "linear_marginals.csv"
    if f.is_file():
        old = pd.read_csv(f)
        keys = set(zip(R["fix"], R["against"]))
        old = old[[(a, b) not in keys for a, b in zip(old["fix"], old["against"])]]
        R = pd.concat([old, R], ignore_index=True)
    R.to_csv(f, index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        print(
            R[
                [
                    (a, b) in {tuple(p.split(":")) for p in pairs}
                    for a, b in zip(R["fix"], R["against"])
                ]
            ]
            .drop(columns=["qlike_base", "sharpe_mid_base", "sharpe_x_base"])
            .round(4)
            .to_string()
        )


# ============================================================================ src gates: the pooled panel
POOLED_GATE_SERIES = [
    "voldemand_spx_open_only",
    "numobs",
    "sumret3_ewstock",
    "vix",
    "spread_vwstock",
    "stocktwits_sentiment",
    "ofi_vwstock",
    "sumvolume",
]


def _pooled_cols_src(series: list[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """The pooled panel's columns of ``series`` built by run_geometry_local.prepare_full's own
    steps (base-2 ladder, 3,125-bar burn-in, 24,000-bar scaling, guards, masked scaling) under the
    src flags currently set; each exog column's chain is independent of the others."""
    import os as _os

    import src.backtest.executor as EX
    from src.features.extractors.har import resolve_har_lags
    from src.features.transforms.scaling import rolling_robust_scale

    _os.environ["UNIF_SCALE_PROCS"] = "4"  # laptop rule: at most 4 workers
    df, adj = EX.load_and_transform(
        str(A.data_dir()),
        series,
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=False,
        overnight_fill=True,
        impute_indicate=True,
        diurnal_mode="divide",
    )
    rungs, x = [], 1.0
    while round(x) <= resolve_har_lags()[-1]:
        rungs.append(int(round(x)))
        x *= 2.0
    df, names = EX._build_har_and_calendar(
        df, adj, add_calendar=True, har_lags=sorted(set(rungs))
    )
    df = df.iloc[resolve_har_lags()[-1] :].reset_index(drop=True)
    keep = [
        n
        for n in names
        if any(n.startswith(f"adj_{s}_ma_") or n.startswith(f"{s}_") for s in series)
    ]
    X = df[keep].values.astype(np.float64)
    ref_iqr, fixed = EX._build_scale_guards(X, df, keep)
    Xs = rolling_robust_scale(X, POOLED_W, ref_iqr=ref_iqr, fixed_cols=fixed)
    Xs = EX.apply_masked_scaling(Xs, X, df, keep, POOLED_W)
    return Xs, keep, df["t"].to_numpy()


def pooled_gate() -> None:
    import zipfile

    sys.path.insert(0, str(REPO / "experiments" / "puffer_forecast"))
    import prep_pooled as PP

    Xc = PP.member_memmap(POOLED_CACHE, "X.npy")
    with zipfile.ZipFile(POOLED_CACHE) as zf, zf.open("names.npy") as fh:
        cn = [str(x) for x in np.lib.format.read_array(fh, allow_pickle=True)]
    rows, res = [], {}
    for tag, on in (("srcoff", set()), ("v2", set(A.SRC_SETS["v2"]))):
        A.set_src_flags(on)
        X, names, t = _pooled_cols_src(POOLED_GATE_SERIES)
        res[tag] = (X, names, pd.DatetimeIndex(t))
        say(f"pooled {tag}: {X.shape}")
    Xo, names, t = res["srcoff"]
    idx = [cn.index(n) for n in names if n in cn]
    common = [n for n in names if n in cn]
    assert len(Xc) == len(Xo), (len(Xc), len(Xo))
    Cc = np.asarray(Xc[:, idx])
    Oo = Xo[:, [names.index(n) for n in common]]
    eq = np.array_equal(Cc, Oo, equal_nan=True)
    rows.append(
        dict(
            gate=f"a: pooled panel, every flag off: {len(common)} columns of {len(POOLED_GATE_SERIES)} series "
            "== prep_cache_all_features_b2.npz, bit for bit",
            ok=bool(eq),
            value=float(np.nanmax(np.abs(Cc - Oo))),
        )
    )
    Xn, nn, tn = res["v2"]
    test = np.asarray(tn >= TEST0)
    diff = []
    for j, c in enumerate(nn):
        if c in names:
            d = np.abs(Xn[:, j] - Xo[:, names.index(c)])
            diff.append(
                dict(
                    column=c,
                    rows_changed=int((d > 0).sum()),
                    rows_changed_test_window=int((d[test] > 0).sum()),
                    max_abs=float(np.nanmax(d)),
                )
            )
        else:
            diff.append(
                dict(
                    column=c,
                    rows_changed=-1,
                    rows_changed_test_window=-1,
                    max_abs=np.nan,
                    note="new column",
                )
            )
    for c in names:
        if c not in nn:
            diff.append(
                dict(
                    column=c,
                    rows_changed=-1,
                    rows_changed_test_window=-1,
                    max_abs=np.nan,
                    note="dropped column",
                )
            )
    D = pd.DataFrame(diff)
    D.to_csv(OUT / "pooled_v2_diff.csv", index=False)
    rows.append(
        dict(
            gate=f"pooled panel, all 8 fixes on: the same series build ({len(nn)} columns; "
            f"{int((D['rows_changed'] > 0).sum())} changed, {int((D.get('note') == 'new column').sum())} new, "
            f"{int((D.get('note') == 'dropped column').sum())} dropped)",
            ok=True,
            value=np.nan,
        )
    )
    G = pd.DataFrame(rows)
    f = OUT / "src_gates.csv"
    old = pd.read_csv(f) if f.is_file() else pd.DataFrame()
    if len(old):
        old = old[~old["gate"].str.contains("pooled panel")]
    pd.concat([old, G], ignore_index=True).to_csv(f, index=False)
    print(G.to_string())


def pooled_gate_cols() -> None:
    """Per-column version of the flags-off pooled gate (which columns differ from the cache)."""
    import zipfile

    sys.path.insert(0, str(REPO / "experiments" / "puffer_forecast"))
    import prep_pooled as PP

    A.set_src_flags(set())
    X, names, t = _pooled_cols_src(POOLED_GATE_SERIES)
    np.savez(WORK / "pooled_srcoff_cols.npz", X=X, names=np.array(names))
    Xc = PP.member_memmap(POOLED_CACHE, "X.npy")
    with zipfile.ZipFile(POOLED_CACHE) as zf, zf.open("names.npy") as fh:
        cn = [str(x) for x in np.lib.format.read_array(fh, allow_pickle=True)]
    rows = []
    for j, c in enumerate(names):
        col = np.asarray(Xc[:, cn.index(c)])
        d = np.abs(col - X[:, j])
        rows.append(
            dict(
                column=c,
                equal=bool(np.array_equal(col, X[:, j], equal_nan=True)),
                max_abs=float(np.nanmax(d)),
                rows_diff=int((d > 0).sum()),
                first_diff=str(t[np.flatnonzero(d > 0)[0]])[:16]
                if (d > 0).any()
                else "",
            )
        )
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "pooled_gate_columns.csv", index=False)
    print(R[~R["equal"]].to_string())
    print(
        R.assign(
            series=R["column"]
            .str.replace(r"^adj_", "", regex=True)
            .str.replace(r"_(avail|active)?_?ma_\d+$", "", regex=True)
        )
        .groupby("series")["equal"]
        .agg(["sum", "size"])
    )


def pooled_v2() -> None:
    """The 8 gate series' pooled columns under all adopted fixes vs the flags-off build (saved by
    pooled_gate_cols): the pooled path runs every fix; what changes and where."""
    A.set_src_flags(set(A.SRC_SETS["v2"]))
    Xn, nn, tn = _pooled_cols_src(POOLED_GATE_SERIES)
    z = np.load(WORK / "pooled_srcoff_cols.npz")
    Xo, names = z["X"], [str(v) for v in z["names"]]
    t = pd.DatetimeIndex(tn)
    test = np.asarray(t >= TEST0)
    rows = []
    for j, c in enumerate(nn):
        if c in names:
            d = np.abs(Xn[:, j] - Xo[:, names.index(c)])
            rows.append(
                dict(
                    column=c,
                    status="changed" if (d > 0).any() else "same",
                    rows_changed=int((d > 0).sum()),
                    rows_changed_test_window=int((d[test] > 0).sum()),
                    mean_abs_test=float(np.nanmean(d[test])),
                )
            )
        else:
            rows.append(dict(column=c, status="new"))
    rows += [dict(column=c, status="dropped") for c in names if c not in nn]
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "pooled_v2_diff.csv", index=False)
    print(D["status"].value_counts().to_string())
    D["series"] = (
        D["column"]
        .str.replace(r"^adj_", "", regex=True)
        .str.replace(r"_(avail|active)?_?ma_\d+$", "", regex=True)
    )
    print(D.groupby(["series", "status"]).size().unstack(fill_value=0).to_string())
