"""Regime conditioning and exit timing for the 0DTE delta-hedged single-leg book.

Consumes the ledger written by ``experiments/spxw_delta_hedged_legs.py``
(``results/spxw_pnl/dh_legs_ledger.parquet``: one row per near-ATM contract
entry at a 30-min bar, held to 16:00 ET settlement).  Two questions:

PART 1/2 -- does the edge concentrate in an observable STATE?  State features
are attached per (expiration day, ET entry hour) from the 30-min panel
(``data/vix_and_voldemand.parquet``, ``data/releases.parquet``,
``data/time_categories.parquet``) plus two derived quantities: the overnight
opening gap and the VIX term slope vix/vix3m.  Performance is then tabulated by
causal VIX quintile, term-slope tercile, day of week, release/FOMC day, |gap|
tercile and era, at theta in {0.05, 0.10}, for both the blk2 and a0 remaining-
variance signals, at mids and crossed.

CLOCK.  The ledger's ``t`` is TRUE UTC (it comes from the option chain); the
panel's ``endbartime`` is NAIVE America/New_York wall clock, BAR-END labelled.
The panel row labelled ``HH:MM`` therefore carries information over the half
hour ENDING at HH:MM ET, i.e. exactly the information set available to a trader
quoting at the ledger stamp whose ET wall clock reads HH:MM.  We join ledger ET
hour H to the panel row labelled H.  The convention is verified in code
(``_clock_check``): the chain's realized 10:00->11:00 ET underlying log return
is reproduced by summing the panel's ``sumret`` over the rows labelled 10:30 and
11:00 (end-labelled) and NOT by the rows labelled 10:00 and 10:30
(start-labelled).  The panel is never parsed with utc=True.

Panel data ends 2024-04-30, which is also the last ledger expiration, so the
regime analysis is restricted to <= 2024-04-30 and spans the full ledger.
Individual state series stop earlier than the file does -- VIX/VVIX/VIX3M at
2024-02-12, the release flags at their last nonzero day, voldemand at
2023-08-31 -- so each conditional table reports its own surviving day count and
the release buckets are blanked (not read as "no release") past the last
populated day.  Coverage is written to dh_regime_coverage.csv.

PART 3 -- does an earlier exit beat holding to settlement?  Mark-to-market paths
are rebuilt per contract from ``data/spxw_chain.parquet`` (same expiration /
strike / cp at later 30-min stamps) together with a per-bar decomposition of the
ledger's delta hedge, recomputed with the identical convention (BS delta at the
ENTRY implied vol, rebalanced on the chain's underlying path, on the bar-of-day
ET grid).  Summing every per-bar hedge increment and marking the option at
intrinsic must return the ledger's ``long_dh_mid`` -- that is asserted.  Exit
rules evaluated at theta = 0.10 (blk2): hold-to-settle, exit on the first later
bar where the re-measured signal crosses back through zero, and fixed horizons
of 1, 2 and 4 bars.  Crossed variants pay the option spread on the extra
round-trip (exit at bid when long, at ask when short) and unwind the hedge at
the exit bar.

Outputs results/spxw_pnl/dh_regime_*.csv and results/spxw_pnl/dh_exit_*.csv.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from spxw_delta_hedged_legs import bs_delta_v, bs_iv_v  # noqa: E402

OUT = os.path.join(ROOT, "results", "spxw_pnl")
DATA = os.path.join(ROOT, "data")

ANN = float(np.sqrt(252.0))
UNDERLYING_COST_BP = 0.5
HOURS_PER_YEAR = 252.0 * 6.5
DAILY_0DTE = pd.Timestamp("2022-05-16")
DATA_END = pd.Timestamp("2024-04-30")
VIX_BURN = 63
THETAS = (0.05, 0.10)
EXIT_THETA = 0.10
EXIT_MODEL = "b2"


# --------------------------------------------------------------------------
# small stats helpers
# --------------------------------------------------------------------------
def _sh(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3 or float(x.std()) == 0.0:
        return float("nan")
    return float(x.mean() / x.std())


def daily_stats(
    pnl: np.ndarray, traded: np.ndarray, days: np.ndarray
) -> dict[str, Any]:
    """Daily aggregation: mean pnl over the day's traded contracts, then Sharpe."""
    p = np.asarray(pnl, float)
    m = np.asarray(traded, bool) & np.isfinite(p)
    if int(m.sum()) == 0:
        return {
            "n": 0,
            "n_days": 0,
            "sh": float("nan"),
            "hit": float("nan"),
            "mean": float("nan"),
        }
    d = pd.Series(p[m], index=np.asarray(days)[m]).groupby(level=0).mean()
    return {
        "n": int(m.sum()),
        "n_days": int(d.size),
        "sh": _sh(d.to_numpy(float)) * ANN,
        "hit": float((p[m] > 0).mean()),
        "mean": float(p[m].mean()),
    }


def _positions(sig: np.ndarray, th: float) -> np.ndarray:
    """Short if sig > theta, long if sig < -theta, flat otherwise."""
    return np.where(sig > th, -1.0, np.where(sig < -th, 1.0, 0.0))


# --------------------------------------------------------------------------
# PART 1 -- state features
# --------------------------------------------------------------------------
def _panel(name: str, cols: list[str]) -> pd.DataFrame:
    """Load a 30-min panel file; endbartime is NAIVE ET, bar-END labelled."""
    df = pd.read_parquet(os.path.join(DATA, name), columns=["endbartime", *cols])
    et = pd.to_datetime(df["endbartime"])  # naive ET on purpose -- never utc=True
    df["day"] = et.dt.normalize()
    df["hhmm"] = et.dt.strftime("%H:%M")
    return df.drop(columns=["endbartime"])


def _clock_check(led: pd.DataFrame) -> pd.DataFrame:
    """Bar-END vs bar-START labelling test on the naive-ET panel."""
    u = led.groupby(["expiration", "bod"])["underlying_price"].first().unstack()
    cs = _panel("core_stats.parquet", ["sumret"])
    piv = cs.pivot_table(index="day", columns="hhmm", values="sumret")
    days = u.index.intersection(piv.index)
    u, piv = u.loc[days], piv.loc[days]
    realized = np.log(u["11:00"] / u["10:00"])
    rows = []
    for lab, v in (
        ("bar_end_labelled", piv["10:30"] + piv["11:00"]),
        ("bar_start_labelled", piv["10:00"] + piv["10:30"]),
    ):
        m = realized.notna() & v.notna()
        rows.append(
            {
                "test": "panel_sumret_vs_chain_underlying_return_1000_to_1100",
                "hypothesis": lab,
                "n_days": int(m.sum()),
                "corr": float(np.corrcoef(realized[m], v[m])[0, 1]),
                "rmse": float(np.sqrt(((realized[m] - v[m]) ** 2).mean())),
            }
        )
    # second check: the VIX row labelled 10:00 must line up with the ledger's
    # 10:00 ET near-ATM implied vol on the SAME day, better than the prior day's
    vixp = _panel("vix_and_voldemand.parquet", ["vix"])
    vix10 = vixp[vixp["hhmm"] == "10:00"].set_index("day")["vix"]
    iv10 = led[led["hhmm"] == "10:00"].groupby("expiration")["iv"].median()
    for lab, series in (
        ("same_day_row_labelled_1000", vix10),
        ("previous_day_row_labelled_1000", vix10.shift(1)),
    ):
        j = pd.concat(
            [iv10.rename("iv"), series.rename("vix")], axis=1, join="inner"
        ).dropna()
        rows.append(
            {
                "test": "vix_at_1000_vs_ledger_atm_iv_at_1000ET",
                "hypothesis": lab,
                "n_days": int(len(j)),
                "corr": float(j.corr().iloc[0, 1]),
                "rmse": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_state(led: pd.DataFrame) -> pd.DataFrame:
    """One row per (expiration day, ET entry hour) with the regime state."""
    keys = (
        led[["expiration", "hhmm"]]
        .drop_duplicates()
        .rename(columns={"expiration": "day"})
    )

    vix = _panel(
        "vix_and_voldemand.parquet",
        [
            "vix",
            "vvix",
            "vix3m",
            "voldemand_spx_open_and_close",
            "voldemand_all_open_and_close",
        ],
    )
    st = keys.merge(vix, on=["day", "hhmm"], how="left")
    st["slope"] = st["vix"] / st["vix3m"]

    rel = _panel(
        "releases.parquet",
        [
            "cpi release",
            "employment release",
            "ppi release",
            "gdp release",
            "trade release",
            "retail release",
            "fomc release",
        ],
    )
    relday = rel.groupby("day").max(numeric_only=True)
    relday["any_release"] = (relday.max(axis=1) > 0).astype(float)
    relday["fomc"] = (relday["fomc release"] > 0).astype(float)
    # the flag columns run to the end of the panel but stop being POPULATED
    # after the last nonzero flag; past that a 0 means "no data", not "no
    # release", so blank the buckets rather than mislabel quiet days
    rel_end = relday.index[relday["any_release"] > 0].max()
    relday.loc[relday.index > rel_end, ["any_release", "fomc"]] = np.nan
    st = st.merge(
        relday[["any_release", "fomc"]], left_on="day", right_index=True, how="left"
    )

    tc = _panel("time_categories.parquet", ["DOW"])
    dow = tc.groupby("day")["DOW"].first()
    st = st.merge(dow.rename("dow"), left_on="day", right_index=True, how="left")

    # opening gap: only when the previous ledger expiration is the prior weekday
    first = led.sort_values("t").groupby("expiration").first()
    s_t = led.groupby("expiration")["S_T"].first()
    d = pd.DataFrame(
        {
            "S_open": first["underlying_price"].astype(float),
            "S_T": s_t.astype(float),
        }
    ).sort_index()
    prev_day = pd.Series(d.index, index=d.index).shift(1)
    prev_close = d["S_T"].shift(1)
    dates = d.index.to_numpy("datetime64[D]")
    prior_wd = np.full(dates.shape, np.datetime64("NaT", "D"), dtype="datetime64[D]")
    if dates.size > 1:
        prior_wd[1:] = np.busday_offset(dates[1:], -1, roll="backward")
    ok = prev_day.to_numpy("datetime64[D]") == prior_wd
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.where(
            ok, np.log(d["S_open"].to_numpy() / prev_close.to_numpy()), np.nan
        )
    st = st.merge(
        pd.Series(gap, index=d.index, name="gap"),
        left_on="day",
        right_index=True,
        how="left",
    )
    st["abs_gap"] = st["gap"].abs()

    # causal VIX quintile: expanding quantiles WITHIN entry hour, shifted one day
    st = st.sort_values(["hhmm", "day"]).reset_index(drop=True)
    q = np.full(len(st), np.nan)
    for _, gh in st.groupby("hhmm"):
        v = gh["vix"].astype(float)
        cuts = [
            v.expanding(min_periods=VIX_BURN).quantile(p).shift(1)
            for p in (0.2, 0.4, 0.6, 0.8)
        ]
        rank = np.zeros(len(gh))
        for c in cuts:
            rank = rank + (v.to_numpy() > c.to_numpy()).astype(float)
        q[gh.index.to_numpy()] = np.where(
            np.isfinite(cuts[0].to_numpy()), rank + 1.0, np.nan
        )
    st["vix_q"] = q
    st["vix_z"] = (st["vix"] - st["vix"].mean()) / st["vix"].std()
    st["era"] = np.where(st["day"] >= DAILY_0DTE, "b_daily_0dte", "a_pre_daily")
    return st.rename(columns={"day": "expiration"})


def _labelled(missing: pd.Series, labels: np.ndarray) -> pd.Series:
    """Object-dtype label column, blanked wherever the state feature is missing."""
    return pd.Series(labels, index=missing.index, dtype=object).mask(missing)


def _tercile(x: pd.Series) -> pd.Series:
    """Full-sample terciles (descriptive conditioning, not a trading rule)."""
    try:
        return pd.qcut(x, 3, labels=["T1_low", "T2_mid", "T3_high"])
    except ValueError:
        return pd.Series(pd.NA, index=x.index, dtype="object")


# --------------------------------------------------------------------------
# PART 2 -- regime tables
# --------------------------------------------------------------------------
def regime_rows(led: pd.DataFrame, dim: str, bucket: pd.Series) -> pd.DataFrame:
    lm = led["long_dh_mid"].to_numpy(float)
    lx = led["long_dh_x"].to_numpy(float)
    sx = led["short_dh_x"].to_numpy(float)
    days = led["expiration"].to_numpy()
    b = bucket.to_numpy(dtype=object)
    labels: list[Any] = [x for x in pd.unique(bucket.dropna())]
    try:
        labels = sorted(labels)
    except TypeError:
        pass
    rows: list[dict[str, Any]] = []
    for th in THETAS:
        for tag in ("b2", "a0"):
            sig = led[f"sig_{tag}"].to_numpy(float)
            pos = _positions(sig, th)
            pnl_mid = pos * lm
            pnl_x = np.where(pos > 0, lx, np.where(pos < 0, sx, np.nan))
            traded = pos != 0
            for lab in [*labels, "ALL"]:
                sel = np.ones(len(led), bool) if lab == "ALL" else (b == lab)
                m = sel & traded
                a = daily_stats(pnl_mid, m, days)
                c = daily_stats(pnl_x, m, days)
                al = daily_stats(-lm, sel, days)  # always-short control
                rows.append(
                    {
                        "dim": dim,
                        "bucket": str(lab),
                        "theta": th,
                        "model": tag,
                        "n_rows": int(sel.sum()),
                        "n_traded": a["n"],
                        "n_days": a["n_days"],
                        "frac_traded": float(traded[sel].mean())
                        if sel.any()
                        else float("nan"),
                        "frac_long": float((pos[m] > 0).mean())
                        if m.any()
                        else float("nan"),
                        "sh_mid": a["sh"],
                        "sh_crossed": c["sh"],
                        "hit_mid": a["hit"],
                        "mean_mid": a["mean"],
                        "mean_crossed": c["mean"],
                        "sh_always_short": al["sh"],
                        "med_abs_sig": float(np.nanmedian(np.abs(sig[sel])))
                        if sel.any()
                        else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def descriptive_ols(led: pd.DataFrame) -> pd.DataFrame:
    """pnl ~ 1 + sig + sig*vix_z.  DESCRIPTIVE association, not causal."""
    y = led["long_dh_mid"].to_numpy(float)
    sig = led["sig_b2"].to_numpy(float)
    z = led["vix_z"].to_numpy(float)
    X = np.column_stack([np.ones_like(sig), sig, sig * z])
    m = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y, days = X[m], y[m], led["expiration"].to_numpy()[m]
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    e = y - X @ beta
    n, k = X.shape
    s2 = float(e @ e) / (n - k)
    se_ols = np.sqrt(np.diag(xtx_inv) * s2)
    codes = pd.factorize(days)[0]
    ng = int(codes.max()) + 1
    meat = np.zeros((k, k))
    for g in range(ng):
        idx = codes == g
        sg = X[idx].T @ e[idx]
        meat += np.outer(sg, sg)
    scale = (ng / (ng - 1.0)) * ((n - 1.0) / (n - k))
    vcl = xtx_inv @ meat @ xtx_inv * scale
    se_cl = np.sqrt(np.diag(vcl))
    return pd.DataFrame(
        {
            "term": ["const", "sig", "sig_x_vix_z"],
            "coef": beta,
            "se_ols": se_ols,
            "t_ols": beta / se_ols,
            "se_day_clustered": se_cl,
            "t_day_clustered": beta / se_cl,
            "n_obs": n,
            "n_days": ng,
            "note": "DESCRIPTIVE association, not causal",
        }
    )


# --------------------------------------------------------------------------
# PART 3 -- mark-to-market paths and exit rules
# --------------------------------------------------------------------------
def build_paths(led: pd.DataFrame) -> dict[str, Any]:
    """Bar-of-day underlying grid, per-bar hedge increments, quotes, re-measured sig."""
    ch = pd.read_parquet(
        os.path.join(DATA, "spxw_chain.parquet"),
        columns=[
            "expiration",
            "timestamp",
            "strike",
            "cp",
            "bid",
            "ask",
            "mid",
            "underlying_price",
        ],
    )
    ch = ch[(ch["bid"] > 0) & (ch["mid"] > 0) & np.isfinite(ch["underlying_price"])]
    ch = ch.rename(columns={"timestamp": "t"})
    ch["expiration"] = pd.to_datetime(ch["expiration"])
    ch["bod"] = (
        pd.to_datetime(ch["t"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%H:%M")
    )

    path = ch.groupby(["expiration", "bod"])["underlying_price"].first().reset_index()
    stamps = sorted(path["bod"].unique())
    day_grid = path.pivot(
        index="expiration", columns="bod", values="underlying_price"
    ).reindex(columns=stamps)

    # per-contract quote matrices (contract = expiration x strike x cp)
    key = pd.MultiIndex.from_arrays([led["expiration"], led["strike"], led["cp"]])
    cids, uniq = pd.factorize(key)
    led = led.assign(cid=cids)
    ucon = pd.DataFrame(
        {
            "expiration": uniq.get_level_values(0),
            "strike": uniq.get_level_values(1),
            "cp": uniq.get_level_values(2),
        }
    )
    ucon["cid"] = np.arange(len(ucon))
    q = ch.merge(ucon, on=["expiration", "strike", "cp"], how="inner")
    nc, m = len(ucon), len(stamps)
    col_of = pd.Index(stamps)
    qcol = col_of.get_indexer(q["bod"].to_numpy())
    qmid = np.full((nc, m), np.nan)
    qbid = np.full((nc, m), np.nan)
    qask = np.full((nc, m), np.nan)
    qrow = q["cid"].to_numpy()
    qmid[qrow, qcol] = q["mid"].to_numpy(float)
    qbid[qrow, qcol] = q["bid"].to_numpy(float)
    qask[qrow, qcol] = q["ask"].to_numpy(float)

    bod_hours = np.array([int(b[:2]) + int(b[3:]) / 60.0 for b in stamps])
    tau_row = np.maximum(16.0 - bod_hours, 0.0) / HOURS_PER_YEAR

    # ---- per-bar hedge increments, ledger convention exactly ----
    grid = day_grid.loc[led["expiration"].to_numpy()].to_numpy(float)
    t_idx = col_of.get_indexer(led["bod"].to_numpy())
    n = len(led)
    colv = np.arange(m)[None, :]
    active = colv >= t_idx[:, None]
    grid = np.where(active, grid, np.nan)
    nxt = np.full_like(grid, np.nan)
    nxt[:, :-1] = grid[:, 1:]
    last_active = np.where(active, colv, -1).max(axis=1)
    s_t_col = led["S_T"].to_numpy(float)
    for j in range(m):
        take = last_active == j
        nxt[take, j] = s_t_col[take]
    for j in range(m - 2, -1, -1):
        gap = active[:, j] & np.isnan(nxt[:, j])
        nxt[gap, j] = nxt[gap, j + 1]
    tau_grid = np.broadcast_to(tau_row[None, :], grid.shape)
    karr = led["strike"].to_numpy(float)
    carr = (led["cp"] == "C").to_numpy()
    varr = led["iv"].to_numpy(float)
    sg_grid = np.where(np.isnan(grid), 1.0, grid)
    dl = bs_delta_v(sg_grid, karr[:, None], tau_grid, varr[:, None], carr[:, None])
    dl = np.where(active & ~np.isnan(grid), dl, 0.0)
    d_s = np.where(active & ~np.isnan(grid) & ~np.isnan(nxt), nxt - grid, 0.0)
    cum_h = np.cumsum(-dl * d_s, axis=1)
    prev = np.zeros_like(dl)
    prev[:, 1:] = dl[:, :-1]
    turn = np.abs(dl - prev) * np.where(np.isnan(grid), 0.0, grid)
    cum_c = np.cumsum(turn * UNDERLYING_COST_BP * 1e-4, axis=1)

    # ---- signal re-measured at every bar for every contract ----
    ugrid = day_grid.reindex(ucon["expiration"].to_numpy()).to_numpy(float)
    b2 = (
        led.groupby(["expiration", "bod"])["b2_rem_cal"]
        .first()
        .unstack()
        .reindex(columns=stamps)
    )
    b2g = b2.reindex(ucon["expiration"].to_numpy()).to_numpy(float)
    ivg = bs_iv_v(
        qmid,
        ugrid,
        ucon["strike"].to_numpy(float)[:, None],
        np.broadcast_to(tau_row[None, :], qmid.shape),
        (ucon["cp"] == "C").to_numpy()[:, None],
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        mvolg = np.sqrt(np.maximum(b2g, 1e-18) / np.maximum(tau_row[None, :], 1e-9))
        sigg = np.log(ivg / mvolg)
    sigg = np.where(np.isfinite(b2g) & np.isfinite(ivg), sigg, np.nan)

    return {
        "led": led,
        "stamps": stamps,
        "m": m,
        "n": n,
        "t_idx": t_idx,
        "grid": grid,
        "active": active,
        "cum_h": cum_h,
        "cum_c": cum_c,
        "dl": dl,
        "qmid": qmid,
        "qbid": qbid,
        "qask": qask,
        "sigg": sigg,
        "last_active": last_active,
        "ucon": ucon,
    }


def exit_pnl(p: dict[str, Any], e_idx: np.ndarray) -> dict[str, np.ndarray]:
    """PnL of the delta-hedged leg exiting at bar e_idx (-1 = hold to settlement)."""
    led = p["led"]
    n, m = p["n"], p["m"]
    cid = led["cid"].to_numpy()
    r = np.arange(n)
    mid0 = led["mid"].to_numpy(float)
    ask0 = led["ask"].to_numpy(float)
    bid0 = led["bid"].to_numpy(float)
    intr = led["intr"].to_numpy(float)
    hold = e_idx < 0
    e = np.where(hold, m - 1, e_idx)
    ep = np.maximum(e - 1, 0)
    hedge = np.where(hold, p["cum_h"][r, m - 1], p["cum_h"][r, ep])
    cost = np.where(hold, p["cum_c"][r, m - 1], p["cum_c"][r, ep])
    unwind = np.abs(p["dl"][r, ep]) * p["grid"][r, e] * UNDERLYING_COST_BP * 1e-4
    cost = cost + np.where(hold | ~np.isfinite(unwind), 0.0, unwind)
    v_mid = np.where(hold, intr, p["qmid"][cid, e])
    v_bid = np.where(hold, intr, p["qbid"][cid, e])
    v_ask = np.where(hold, intr, p["qask"][cid, e])
    return {
        "long_mid": (v_mid - mid0 + hedge) / mid0,
        "long_x": (v_bid - ask0 + hedge - cost) / mid0,
        "short_x": (bid0 - v_ask - hedge - cost) / mid0,
        "e": e,
        "hold": hold,
    }


def rule_exit_index(p: dict[str, Any], pos: np.ndarray, rule: str) -> np.ndarray:
    """-1 = hold to settlement; otherwise the bar-of-day column index of the exit."""
    n, m = p["n"], p["m"]
    t_idx, la = p["t_idx"], p["last_active"]
    cid = p["led"]["cid"].to_numpy()
    rows = np.arange(n)
    quotable = np.isfinite(p["qmid"][cid, :]) & np.isfinite(p["grid"])
    if rule.startswith("k"):
        k = int(rule[1:])
        e = t_idx + k
        for _ in range(m):
            live = (e < m) & (e <= la)
            bad = live & ~quotable[rows, np.minimum(e, m - 1)]
            if not bad.any():
                break
            e = np.where(bad, e + 1, e)
        ok = (e < m) & (e <= la) & quotable[rows, np.minimum(e, m - 1)]
        return np.where(ok, e, -1)
    if rule == "sigcross":
        sg = p["sigg"][cid, :]
        col = np.arange(m)[None, :]
        later = (col > t_idx[:, None]) & (col <= la[:, None]) & quotable
        # a short (pos<0) entered at sig>theta exits when sig falls back to <=0
        cross = np.where(pos[:, None] < 0, sg <= 0.0, sg >= 0.0)
        cand = later & np.isfinite(sg) & cross
        return np.where(cand.any(axis=1), cand.argmax(axis=1), -1)
    raise ValueError(f"unknown rule {rule}")


# --------------------------------------------------------------------------
def main() -> None:
    led = pd.read_parquet(os.path.join(OUT, "dh_legs_ledger.parquet"))
    led["expiration"] = pd.to_datetime(led["expiration"])
    led = led.sort_values(["expiration", "t", "strike", "cp"]).reset_index(drop=True)
    print(f"ledger rows {len(led)}  days {led['expiration'].nunique()}", flush=True)

    # ---------------- baseline reproduction assertion ----------------
    lm = led["long_dh_mid"].to_numpy(float)
    lx = led["long_dh_x"].to_numpy(float)
    sx = led["short_dh_x"].to_numpy(float)
    days = led["expiration"].to_numpy()
    pos0 = _positions(led["sig_b2"].to_numpy(float), 0.10)
    tr0 = pos0 != 0
    base_mid = daily_stats(pos0 * lm, tr0, days)
    base_x = daily_stats(
        np.where(pos0 > 0, lx, np.where(pos0 < 0, sx, np.nan)), tr0, days
    )
    base_as = daily_stats(-lm, np.ones(len(led), bool), days)
    got = {
        "sh_daily_mid": base_mid["sh"],
        "sh_daily_crossed": base_x["sh"],
        "frac_traded": float(tr0.mean()),
        "hit_mid": base_mid["hit"],
        "always_short_sh": base_as["sh"],
    }
    want = {
        "sh_daily_mid": 6.78,
        "sh_daily_crossed": 3.62,
        "frac_traded": 0.58,
        "hit_mid": 0.697,
        "always_short_sh": 3.81,
    }
    chk = pd.DataFrame(
        [
            {
                "stat": k,
                "expected": want[k],
                "got": got[k],
                "abs_diff": abs(got[k] - want[k]),
                "within_0p05": bool(abs(got[k] - want[k]) < 0.05),
            }
            for k in want
        ]
    )
    print(chk.to_string(index=False), flush=True)
    chk.to_csv(os.path.join(OUT, "dh_regime_baseline_check.csv"), index=False)
    assert bool(chk["within_0p05"].all()), "baseline reproduction failed"

    clock = _clock_check(led)
    print(clock.to_string(index=False), flush=True)
    clock.to_csv(os.path.join(OUT, "dh_regime_clockcheck.csv"), index=False)

    # ---------------- PART 1: state ----------------
    st = build_state(led)
    led = led.merge(st, on=["expiration", "hhmm"], how="left")
    led = led[led["expiration"] <= DATA_END].reset_index(drop=True)
    cov = pd.DataFrame(
        [
            {
                "ledger_days": int(led["expiration"].nunique()),
                "ledger_rows": int(len(led)),
                "days_with_vix": int(
                    led.loc[led["vix"].notna(), "expiration"].nunique()
                ),
                "days_with_causal_vix_quintile": int(
                    led.loc[led["vix_q"].notna(), "expiration"].nunique()
                ),
                "days_with_gap": int(
                    led.loc[led["gap"].notna(), "expiration"].nunique()
                ),
                "days_with_release_flags": int(
                    led.loc[led["any_release"].notna(), "expiration"].nunique()
                ),
                "release_days": int(
                    led.loc[led["any_release"] > 0, "expiration"].nunique()
                ),
                "fomc_days": int(led.loc[led["fomc"] > 0, "expiration"].nunique()),
                "days_with_voldemand": int(
                    led.loc[
                        led["voldemand_spx_open_and_close"].notna(), "expiration"
                    ].nunique()
                ),
                "analysis_window_end": str(DATA_END.date()),
                "last_day_with_vix": str(
                    pd.Timestamp(led.loc[led["vix"].notna(), "expiration"].max()).date()
                ),
                "last_day_with_release_flags": str(
                    pd.Timestamp(
                        led.loc[led["any_release"].notna(), "expiration"].max()
                    ).date()
                ),
                "mean_vix_at_1000": float(
                    led.loc[led["hhmm"] == "10:00", "vix"].mean()
                ),
            }
        ]
    )
    print(cov.to_string(index=False), flush=True)
    cov.to_csv(os.path.join(OUT, "dh_regime_coverage.csv"), index=False)

    led["slope_terc"] = _tercile(led["slope"]).astype(object)
    led["gap_terc"] = _tercile(led["abs_gap"]).astype(object)
    led["vvix_terc"] = _tercile(led["vvix"]).astype(object)
    led["vd_terc"] = _tercile(led["voldemand_spx_open_and_close"]).astype(object)
    led["backwardation"] = _labelled(
        led["slope"].isna(),
        np.where(led["slope"].to_numpy(float) > 1.0, "backwardation", "contango"),
    )
    led["rel_bucket"] = _labelled(
        led["any_release"].isna(),
        np.where(led["any_release"].to_numpy(float) > 0, "release", "no_release"),
    )
    led["fomc_bucket"] = _labelled(
        led["fomc"].isna(),
        np.where(led["fomc"].to_numpy(float) > 0, "fomc", "no_fomc"),
    )
    dow_map = {0: "0_Mon", 1: "1_Tue", 2: "2_Wed", 3: "3_Thu", 4: "4_Fri"}
    led["dow_lab"] = led["dow"].map(dow_map)
    led["vix_q_lab"] = led["vix_q"].map(
        lambda v: f"Q{int(v)}" if np.isfinite(v) else None
    )

    # ---------------- PART 2: regime tables ----------------
    tables = {
        "vixq": ("vix_quintile_causal", led["vix_q_lab"]),
        "slope": ("vix_term_slope_tercile", led["slope_terc"]),
        "backwardation": ("contango_vs_backwardation", led["backwardation"]),
        "dow": ("day_of_week", led["dow_lab"]),
        "release": ("release_day", led["rel_bucket"]),
        "fomc": ("fomc_day", led["fomc_bucket"]),
        "gap": ("abs_open_gap_tercile", led["gap_terc"]),
        "era": ("era", led["era"]),
        "vvix": ("vvix_tercile", led["vvix_terc"]),
        "voldemand": ("voldemand_tercile", led["vd_terc"]),
        "hour": ("entry_hour", led["hhmm"]),
    }
    for tag, (dim, bucket) in tables.items():
        tb = regime_rows(led, dim, pd.Series(bucket, index=led.index))
        tb.to_csv(os.path.join(OUT, f"dh_regime_{tag}.csv"), index=False)

    sigmag = (
        led.groupby("vix_q_lab")
        .agg(
            n=("sig_b2", "size"),
            mean_vix=("vix", "mean"),
            med_sig_b2=("sig_b2", "median"),
            med_abs_sig_b2=("sig_b2", lambda s: float(np.nanmedian(np.abs(s)))),
            med_sig_a0=("sig_a0", "median"),
            med_abs_sig_a0=("sig_a0", lambda s: float(np.nanmedian(np.abs(s)))),
            med_iv=("iv", "median"),
            med_mvol_b2=("mvol_b2", "median"),
        )
        .reset_index()
    )
    for th in THETAS:
        fr = (
            led.assign(tr=np.abs(led["sig_b2"].to_numpy(float)) > th)
            .groupby("vix_q_lab")["tr"]
            .mean()
        )
        sigmag[f"frac_traded_b2_th{th}"] = fr.reindex(sigmag["vix_q_lab"]).to_numpy()
    sigmag.to_csv(os.path.join(OUT, "dh_regime_sigmag.csv"), index=False)
    print(sigmag.to_string(index=False), flush=True)

    ols = descriptive_ols(led)
    ols.to_csv(os.path.join(OUT, "dh_regime_ols.csv"), index=False)
    print(ols.to_string(index=False), flush=True)

    # ---------------- PART 3: exits ----------------
    p = build_paths(led)
    led = p["led"]
    n, m = p["n"], p["m"]

    hold = exit_pnl(p, np.full(n, -1))
    d_mid = hold["long_mid"] - led["long_dh_mid"].to_numpy(float)
    d_x = hold["long_x"] - led["long_dh_x"].to_numpy(float)
    mtm_chk = pd.DataFrame(
        [
            {
                "check": "hold_to_settle_long_dh_mid",
                "n": int(np.isfinite(d_mid).sum()),
                "max_abs_diff": float(np.nanmax(np.abs(d_mid))),
                "frac_within_1e-6": float((np.abs(d_mid) < 1e-6).mean()),
                "frac_within_1e-9": float((np.abs(d_mid) < 1e-9).mean()),
            },
            {
                "check": "hold_to_settle_long_dh_x",
                "n": int(np.isfinite(d_x).sum()),
                "max_abs_diff": float(np.nanmax(np.abs(d_x))),
                "frac_within_1e-6": float((np.abs(d_x) < 1e-6).mean()),
                "frac_within_1e-9": float((np.abs(d_x) < 1e-9).mean()),
            },
        ]
    )
    print(mtm_chk.to_string(index=False), flush=True)
    mtm_chk.to_csv(os.path.join(OUT, "dh_exit_mtm_check.csv"), index=False)
    assert float(np.nanmax(np.abs(d_mid))) < 1e-6, "MTM hold-to-settle mismatch"

    pos = _positions(led[f"sig_{EXIT_MODEL}"].to_numpy(float), EXIT_THETA)
    traded = pos != 0
    dayv = led["expiration"].to_numpy()
    hh = led["hhmm"].to_numpy()
    hours = sorted(pd.unique(hh))

    rows: list[dict[str, Any]] = []
    for rule in ("hold", "sigcross", "k1", "k2", "k4"):
        e_idx = np.full(n, -1) if rule == "hold" else rule_exit_index(p, pos, rule)
        pe = exit_pnl(p, e_idx)
        pmid = pos * pe["long_mid"]
        px = np.where(pos > 0, pe["long_x"], np.where(pos < 0, pe["short_x"], np.nan))
        early = e_idx >= 0
        bars_held = np.where(e_idx >= 0, e_idx - p["t_idx"], m - 1 - p["t_idx"])
        for grp, sel in [("ALL", np.ones(n, bool))] + [(h, hh == h) for h in hours]:
            mm = sel & traded
            a = daily_stats(pmid, mm, dayv)
            c = daily_stats(px, mm, dayv)
            rows.append(
                {
                    "rule": rule,
                    "entry_hour": grp,
                    "theta": EXIT_THETA,
                    "model": EXIT_MODEL,
                    "n_traded": a["n"],
                    "n_days": a["n_days"],
                    "frac_exit_early": float(early[mm].mean())
                    if mm.any()
                    else float("nan"),
                    "mean_bars_held": float(np.nanmean(bars_held[mm]))
                    if mm.any()
                    else float("nan"),
                    "sh_mid": a["sh"],
                    "hit_mid": a["hit"],
                    "mean_mid": a["mean"],
                    "sh_crossed": c["sh"],
                    "hit_crossed": c["hit"],
                    "mean_crossed": c["mean"],
                }
            )
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "dh_exit_rules.csv"), index=False)

    # realized fraction of settlement pnl after k bars
    final = pos * hold["long_mid"]
    frac_rows: list[dict[str, Any]] = []
    for k in range(1, m):
        e = p["t_idx"] + k
        ok = (e < m) & (e <= p["last_active"])
        pk = exit_pnl(p, np.where(ok, np.minimum(e, m - 1), -1))
        cum = pos * pk["long_mid"]
        valid = traded & ok & np.isfinite(cum) & np.isfinite(final)
        for grp, sel in [("ALL", np.ones(n, bool))] + [(h, hh == h) for h in hours]:
            mm = sel & valid
            if int(mm.sum()) < 20:
                continue
            frac_rows.append(
                {
                    "entry_hour": grp,
                    "k_bars": k,
                    "n": int(mm.sum()),
                    "mean_cum_pnl": float(cum[mm].mean()),
                    "mean_final_pnl": float(final[mm].mean()),
                    "realized_fraction": float(cum[mm].mean() / final[mm].mean()),
                }
            )
    pd.DataFrame(frac_rows).to_csv(
        os.path.join(OUT, "dh_exit_realized_fraction.csv"), index=False
    )

    pd.set_option("display.width", 220)
    er = pd.DataFrame(rows)
    print(er[er["entry_hour"] == "ALL"].to_string(index=False), flush=True)
    fr2 = pd.DataFrame(frac_rows)
    print(fr2[fr2["entry_hour"] == "ALL"].to_string(index=False), flush=True)
    print(f"wrote {OUT}/dh_regime_*.csv and dh_exit_*.csv", flush=True)


if __name__ == "__main__":
    main()
