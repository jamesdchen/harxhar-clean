"""Rest-of-day variance PATH forecasting for 0DTE: does a path model beat the
one-step sum?

At every 30-min RTH bar the 0DTE book needs the whole remaining per-bar variance
path [rv_{k+1}, ..., rv_T] conditional on F_{t_k} -- the weather-nowcast object --
not just the next bar.  The incumbent answer is "sum the per-bar one-step
forecasts".  This script tests the cheapest honest alternatives against it.

Convention.  Bars are the 12 half-hour RTH slots 10:00..15:30 ET.  Entry bar k
means: the day's bars 0..k are realized and observed, the model-free implied
variance mark C_k of bar k is observed, and the target is the remaining path
bars k+1..11 (so k runs 0..10 -- with j > k strictly there are 11 entry points,
not 12).

Models (all causal: expanding fits over strictly earlier days, min 63 days, the
fit applied one day forward, and a causal multiplicative rescale applied on
exp() of every raw prediction):

  M0    sum of the per-bar blk2 forecasts pb over the remaining bars (baseline)
  M1    M0's raw total x causal expanding mean intraday shape of rv by slot
  M2    direct multivariate: per (entry bar k, remaining bar j) expanding OLS of
        log rv_j on [1, log cum rv through k, log pb_rem, log C_k,
        log rv_j(previous day)], pinv when ill-conditioned
  M3    conditional analog ensemble: the 50 nearest past days on the causally
        standardized partial state (log rv path 10:00..k, log C_k, log pb_rem);
        remaining path = neighbour mean path rescaled by the ratio of today's
        so-far cumulative rv to the neighbours'
  M3k   M3 with Gaussian kernel weights exp(-d^2/h^2), h = median neighbour
        distance
  M4    M0's raw total x M3's neighbour path shape

M1 and M4 differ from M0 only in how the total is distributed across the
remaining bars, so their calibrated SUM forecast is identical to M0's by
construction; they are shape tests.  Only M2/M3/M3k can move the sum metric.

Metrics: QLIKE on the remaining SUM (the strategy-relevant object) and QLIKE per
remaining bar averaged (path shape), with Newey-West Diebold-Mariano t-stats
against M0 on a common scored sample.

Outputs: results/spxw_pnl/restofday_{sum,path,bybar,preds}.csv
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "spxw_pnl")
SRC = os.path.join(OUT, "everybar_mtm_trades.parquet")

N_SLOT = 12
MIN_FIT = 63
MIN_SCALE = 63
N_NEIGH = 50
ERA0 = pd.Timestamp("2022-05-16")
MODELS = ("M0", "M1", "M2", "M3", "M3k", "M4")
FLOOR = 1e-18


def _slot_labels() -> list[str]:
    base = pd.Timestamp("2000-01-01 10:00")
    return [
        (base + pd.Timedelta(minutes=30 * s)).strftime("%H:%M") for s in range(N_SLOT)
    ]


def _panel() -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Day x slot panels of realized rv, blk2 forecast pb, and the MFIV mark C."""
    d = pd.read_parquet(SRC)
    et = pd.to_datetime(d["et"])
    d = d.assign(slot=(et.dt.hour * 2 + et.dt.minute // 30 - 20).astype(int))
    ok = np.isfinite(d["rv"]) & np.isfinite(d["pb"]) & np.isfinite(d["C"])
    ok &= (d["rv"] > 0) & (d["pb"] > 0) & (d["C"] > 0)
    ok &= d["slot"].between(0, N_SLOT - 1)
    d = d.loc[ok].copy()
    d = d.drop_duplicates(subset=["day", "slot"]).sort_values(["day", "slot"])
    full = d.groupby("day")["slot"].size() == N_SLOT
    d = d[d["day"].isin(full.index[full.to_numpy()])]
    days = pd.DatetimeIndex(sorted(d["day"].unique()))
    piv = {
        c: d.pivot(index="day", columns="slot", values=c).reindex(days)
        for c in ("rv", "pb", "C")
    }
    print(
        f"panel: {len(days)} complete {N_SLOT}-bar days  {days[0].date()}..{days[-1].date()}",
        flush=True,
    )
    return (
        days,
        piv["rv"].to_numpy(float),
        piv["pb"].to_numpy(float),
        piv["C"].to_numpy(float),
    )


def _ql(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    r = np.maximum(y, FLOOR) / np.maximum(f, FLOOR)
    return r - np.log(r) - 1.0


def _rescale(raw: np.ndarray, act: np.ndarray) -> np.ndarray:
    """Causal multiplicative rescale: raw_t x (sum past actual / sum past raw).

    Past means strictly earlier days on which this model emitted a usable raw
    prediction, so the correction never sees the day it corrects.
    """
    raw2 = np.atleast_2d(raw.T).T if raw.ndim > 1 else raw.reshape(-1, 1)
    act2 = np.broadcast_to(act.reshape(raw2.shape[0], -1), raw2.shape)
    v = np.isfinite(raw2) & (raw2 > 0) & np.isfinite(act2) & (act2 > 0)
    zr = np.where(v, raw2, 0.0)
    za = np.where(v, act2, 0.0)
    z = np.zeros((1, raw2.shape[1]))
    pr = np.vstack([z, np.cumsum(zr, axis=0)[:-1]])
    pa = np.vstack([z, np.cumsum(za, axis=0)[:-1]])
    pn = np.vstack([z, np.cumsum(v, axis=0)[:-1]])
    good = (pn >= MIN_SCALE) & (pr > 0) & np.isfinite(raw2) & (raw2 > 0)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        sc = raw2 * (pa / np.where(pr > 0, pr, 1.0))
    out = np.where(good & np.isfinite(sc) & (sc > 0), sc, np.nan)
    return out if raw.ndim > 1 else out[:, 0]


def _solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batched normal-equation solve, minimum-norm pinv when ill-conditioned."""
    try:
        return np.linalg.solve(a, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return np.einsum("kij,kj->ki", np.linalg.pinv(a), b)


def _m2_raw(
    k: int, rv: np.ndarray, pb: np.ndarray, cc: np.ndarray, cum: np.ndarray
) -> np.ndarray:
    n_day, _ = rv.shape
    m = N_SLOT - k - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        base = np.column_stack(
            [
                np.ones(n_day),
                np.log(cum[:, k]),
                np.log(pb[:, k + 1 :].sum(axis=1)),
                np.log(cc[:, k]),
            ]
        )
        prev = np.full((n_day, m), np.nan)
        prev[1:] = np.log(rv[:-1, k + 1 :])
        y = np.log(rv[:, k + 1 :])
    raw = np.full((n_day, m), np.nan)
    xtx = np.zeros((m, 5, 5))
    xty = np.zeros((m, 5))
    cnt = 0
    for t in range(n_day):
        xt = np.concatenate(
            [np.repeat(base[t][None, :], m, axis=0), prev[t][:, None]], axis=1
        )
        fin = bool(np.isfinite(xt).all())
        if cnt >= MIN_FIT and fin:
            beta = _solve(xtx, xty)
            with np.errstate(over="ignore"):
                raw[t] = np.exp(np.einsum("ij,ij->i", xt, beta))
        if fin and bool(np.isfinite(y[t]).all()):
            xtx += xt[:, :, None] * xt[:, None, :]
            xty += xt * y[t][:, None]
            cnt += 1
    raw[~np.isfinite(raw)] = np.nan
    return raw


def _m3_raw(
    k: int, rv: np.ndarray, pb: np.ndarray, cc: np.ndarray, cum: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Analog ensemble over past days; returns (equal-weight, kernel-weight)."""
    n_day, _ = rv.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        state = np.column_stack(
            [
                np.log(rv[:, : k + 1]),
                np.log(cc[:, k]),
                np.log(pb[:, k + 1 :].sum(axis=1)),
            ]
        )
    rem = rv[:, k + 1 :]
    c0 = cum[:, k]
    good = np.isfinite(state).all(axis=1) & np.isfinite(rem).all(axis=1) & (c0 > 0)
    flat = np.full((n_day, N_SLOT - k - 1), np.nan)
    kern = np.full_like(flat, np.nan)
    for t in range(n_day):
        idx = np.flatnonzero(good[:t])
        if idx.size < MIN_FIT or not good[t]:
            continue
        p = state[idx]
        mu = p.mean(axis=0)
        sd = p.std(axis=0)
        sd = np.where(sd > 0.0, sd, 1.0)
        d2 = (((p - mu) / sd - (state[t] - mu) / sd) ** 2).sum(axis=1)
        nn = min(N_NEIGH, idx.size)
        sel = np.argpartition(d2, nn - 1)[:nn]
        pick = idx[sel]
        dd = d2[sel]
        h2 = float(np.median(dd))
        wk = np.exp(-dd / h2) if h2 > 0.0 else np.ones(nn)
        for w, dest in ((np.ones(nn), flat), (wk, kern)):
            s = float(w.sum())
            if not np.isfinite(s) or s <= 0.0:
                continue
            wn = w / s
            cn = float(wn @ c0[pick])
            if cn > 0.0:
                dest[t] = (wn @ rem[pick]) * (c0[t] / cn)
    return flat, kern


def _shape_norm(x: np.ndarray) -> np.ndarray:
    s = x.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = x / np.where(np.isfinite(s) & (s > 0), s, np.nan)
    return w


def _dm(loss: np.ndarray, base: np.ndarray) -> tuple[float, float]:
    """Newey-West Diebold-Mariano t on the loss differential (negative = better)."""
    d = np.asarray(loss - base, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 8 or float(np.max(np.abs(d))) == 0.0:
        return 0.0, 1.0
    dm = d - d.mean()
    lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    var = float(dm @ dm) / n
    for lg in range(1, min(lag, n - 1) + 1):
        g = float(dm[lg:] @ dm[:-lg]) / n
        var += 2.0 * (1.0 - lg / (lag + 1.0)) * g
    if not np.isfinite(var) or var <= 0.0:
        return float("nan"), float("nan")
    t = float(d.mean() / np.sqrt(var / n))
    return t, float(2.0 * stats.norm.sf(abs(t)))


def main() -> None:
    days, rv, pb, cc = _panel()
    n_day = rv.shape[0]
    cum = np.cumsum(rv, axis=1)
    lab = _slot_labels()
    day_naive = pd.DatetimeIndex(days).tz_localize(None)
    eras = {
        "all": np.ones(n_day, bool),
        "daily_0dte": np.asarray(day_naive >= ERA0, bool),
    }

    # causal expanding mean intraday shape of rv by slot (days strictly earlier)
    seas = np.full_like(rv, np.nan)
    seas[1:] = np.cumsum(rv, axis=0)[:-1] / np.arange(1, n_day)[:, None]

    rows_sum: list[dict] = []
    rows_path: list[dict] = []
    rows_bar: list[dict] = []
    rows_pred: list[dict] = []

    for k in range(N_SLOT - 1):
        m = N_SLOT - k - 1
        y_path = rv[:, k + 1 :]
        y_sum = y_path.sum(axis=1)
        pb_rem = pb[:, k + 1 :]
        tot = pb_rem.sum(axis=1)
        warm = np.arange(n_day) >= MIN_FIT

        m3, m3k = _m3_raw(k, rv, pb, cc, cum)
        raw: dict[str, np.ndarray] = {
            "M0": pb_rem.copy(),
            "M1": tot[:, None] * _shape_norm(seas[:, k + 1 :]),
            "M2": _m2_raw(k, rv, pb, cc, cum),
            "M3": m3,
            "M3k": m3k,
            "M4": tot[:, None] * _shape_norm(m3),
        }
        for nm in raw:
            raw[nm] = np.where(warm[:, None], raw[nm], np.nan)

        # M1/M4 redistribute M0's total across the remaining bars, so their raw
        # sum IS M0's raw sum -- take it literally rather than re-summing a
        # normalised shape, which would differ only by float noise.
        raw_sum = {nm: r.sum(axis=1) for nm, r in raw.items()}
        m0_sum = np.where(warm, tot, np.nan)
        for nm in ("M0", "M1", "M4"):
            raw_sum[nm] = np.where(np.isfinite(raw[nm]).all(axis=1), m0_sum, np.nan)

        pred_sum = {nm: _rescale(r, y_sum) for nm, r in raw_sum.items()}
        pred_path = {nm: _rescale(r, y_path) for nm, r in raw.items()}

        ok_sum = np.ones(n_day, bool)
        ok_path = np.ones(n_day, bool)
        for nm in MODELS:
            ok_sum &= np.isfinite(pred_sum[nm]) & (pred_sum[nm] > 0)
            ok_path &= np.isfinite(pred_path[nm]).all(axis=1) & (pred_path[nm] > 0).all(
                axis=1
            )
        ok_sum &= np.isfinite(y_sum) & (y_sum > 0)
        ok_path &= ok_sum & np.isfinite(y_path).all(axis=1) & (y_path > 0).all(axis=1)

        for era, emask in eras.items():
            ms = ok_sum & emask
            mp = ok_path & emask
            if int(ms.sum()) == 0:
                continue
            lbase = _ql(y_sum[ms], pred_sum["M0"][ms])
            for nm in MODELS:
                lm = _ql(y_sum[ms], pred_sum[nm][ms])
                t, p = _dm(lm, lbase)
                rows_sum.append(
                    {
                        "era": era,
                        "k": k,
                        "entry_et": lab[k],
                        "n_rem_bars": m,
                        "model": nm,
                        "n_days": int(ms.sum()),
                        "qlike_sum": float(lm.mean()),
                        "dm_t_vs_M0": t,
                        "dm_p_vs_M0": p,
                    }
                )
                lp = _ql(y_path[mp], pred_path[nm][mp])
                rows_path.append(
                    {
                        "era": era,
                        "k": k,
                        "entry_et": lab[k],
                        "model": nm,
                        "n_days": int(mp.sum()),
                        "n_obs": int(mp.sum()) * m,
                        "qlike_path": float(lp.mean()) if lp.size else float("nan"),
                    }
                )
                for jj in range(m):
                    rows_bar.append(
                        {
                            "era": era,
                            "k": k,
                            "entry_et": lab[k],
                            "target_et": lab[k + 1 + jj],
                            "model": nm,
                            "n_days": int(mp.sum()),
                            "qlike": float(lp[:, jj].mean())
                            if lp.size
                            else float("nan"),
                        }
                    )

        for i in np.flatnonzero(ok_sum):
            r = {
                "day": day_naive[i].date().isoformat(),
                "k": k,
                "entry_et": lab[k],
                "rv_rem": float(y_sum[i]),
            }
            r.update({f"pred_{nm}": float(pred_sum[nm][i]) for nm in MODELS})
            rows_pred.append(r)
        print(
            f"entry {lab[k]} (k={k}, {m} remaining bars): scored {int(ok_sum.sum())} days",
            flush=True,
        )

    ds = pd.DataFrame(rows_sum)
    dp = pd.DataFrame(rows_path)
    db = pd.DataFrame(rows_bar)
    dd = pd.DataFrame(rows_pred)
    os.makedirs(OUT, exist_ok=True)
    ds.to_csv(os.path.join(OUT, "restofday_sum.csv"), index=False)
    dp.to_csv(os.path.join(OUT, "restofday_path.csv"), index=False)
    db.to_csv(os.path.join(OUT, "restofday_bybar.csv"), index=False)
    dd.to_csv(os.path.join(OUT, "restofday_preds.csv"), index=False)

    for era in eras:
        print(
            f"\n=== QLIKE on the remaining SUM  [era={era}]  (DM t vs M0, neg = better) ===",
            flush=True,
        )
        head = "entry  n    " + "".join(f"{nm:>18s}" for nm in MODELS)
        print(head, flush=True)
        sub = ds[ds["era"] == era]
        for k, g in sub.groupby("k", sort=True):
            g = g.set_index("model")
            cells = ""
            for nm in MODELS:
                q = g.loc[nm, "qlike_sum"]
                t = g.loc[nm, "dm_t_vs_M0"]
                cells += (
                    f"  {q:7.4f}({t:+6.2f})" if nm != "M0" else f"  {q:7.4f}(  base)"
                )
            print(
                f"{g['entry_et'].iloc[0]}  {int(g['n_days'].iloc[0]):4d}{cells}",
                flush=True,
            )
        w = sub.pivot_table(index="model", values="qlike_sum", aggfunc="mean")
        print(
            "mean over entry bars: "
            + "  ".join(f"{nm}={float(w.loc[nm, 'qlike_sum']):.4f}" for nm in MODELS),
            flush=True,
        )

        print(
            f"\n=== QLIKE per remaining bar (path shape)  [era={era}] ===", flush=True
        )
        print("entry  n    " + "".join(f"{nm:>10s}" for nm in MODELS), flush=True)
        subp = dp[dp["era"] == era]
        for k, g in subp.groupby("k", sort=True):
            g = g.set_index("model")
            cells = "".join(f"  {float(g.loc[nm, 'qlike_path']):8.4f}" for nm in MODELS)
            print(
                f"{g['entry_et'].iloc[0]}  {int(g['n_days'].iloc[0]):4d}{cells}",
                flush=True,
            )
        wp = subp.pivot_table(index="model", values="qlike_path", aggfunc="mean")
        print(
            "mean over entry bars: "
            + "  ".join(f"{nm}={float(wp.loc[nm, 'qlike_path']):.4f}" for nm in MODELS),
            flush=True,
        )

    print(
        "\n=== pooled DM vs M0 on the SUM (entry bars stacked; DM on day means) ===",
        flush=True,
    )
    print(
        "(one day contributes 11 correlated entry bars, so the DM series is the",
        flush=True,
    )
    print(
        " per-day mean loss differential -- clustering the dependence away)", flush=True
    )
    rows_pool: list[dict] = []
    for era in eras:
        sub = dd.copy()
        keep = pd.DatetimeIndex(sub["day"]) >= (
            ERA0 if era == "daily_0dte" else pd.Timestamp("1900-01-01")
        )
        sub = sub[np.asarray(keep, bool)]
        y = sub["rv_rem"].to_numpy(float)
        loss = pd.DataFrame(
            {nm: _ql(y, sub[f"pred_{nm}"].to_numpy(float)) for nm in MODELS}
        )
        loss["day"] = sub["day"].to_numpy()
        per_day = loss.groupby("day", sort=True).mean()
        for nm in MODELS:
            t, p = _dm(per_day[nm].to_numpy(float), per_day["M0"].to_numpy(float))
            rows_pool.append(
                {
                    "era": era,
                    "k": -1,
                    "entry_et": "POOLED",
                    "n_rem_bars": -1,
                    "model": nm,
                    "n_days": int(per_day.shape[0]),
                    "qlike_sum": float(loss[nm].mean()),
                    "dm_t_vs_M0": t,
                    "dm_p_vs_M0": p,
                }
            )
            print(
                f"  era={era:11s} {nm:4s} n_obs={len(sub):6d} n_days={per_day.shape[0]:4d} "
                f"QLIKE={loss[nm].mean():.4f} DM_t={t:+6.2f} p={p:.3f}",
                flush=True,
            )
    ds = pd.concat([ds, pd.DataFrame(rows_pool)], ignore_index=True)
    ds.to_csv(os.path.join(OUT, "restofday_sum.csv"), index=False)
    print(
        f"\nwrote {os.path.join(OUT, 'restofday_sum.csv')} and 3 companions", flush=True
    )


if __name__ == "__main__":
    main()
