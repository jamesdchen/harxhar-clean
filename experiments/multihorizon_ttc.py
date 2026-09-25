"""Multi-horizon (time-till-close) forecasts from the paper's per-bar models.

The paper's models issue one forecast per 30-minute bar. At any bar t of
a session a natural multi-horizon object is the variance remaining to
the 16:00 close, RV_{t,T} = sum of the per-bar realized variances from
t+1 to the last RTH bar. This script asks how well each per-bar model
extends to that horizon, using ONLY information available at t
(strictly F_t-measurable), on the paper's own panel -- every RTH session
in the yhat exports, no option data.

Constructions (per entry hour h, expanding fits over strictly earlier
days, minimum 63, applied one day forward, then the causal QLIKE-optimal
multiplicative rescale c_h = expanding mean of realized/forecast):

  M1  one-step mapped:  log RV_{t,T} ~ a_h + b_h log vhat_t
        vhat_t = the model's forecast for bar t+1, issued at t.
  M2  + realized so far: log RV_{t,T} ~ a_h + b_h log vhat_t + d_h log RV_{open,t}
        RV_{open,t} = realized variance from the open through bar t
        (known at t; panel-native).
  M3  + yesterday's same-horizon realized: adds log RV_{t,T}(d-1).
  M0  naive: expanding mean of log RV_{t,T} for hour h (no model) --
        the horizon-specific unconditional benchmark.

Each of M1-M3 is run for both per-bar models (a0 = OLS-HAR incumbent,
blk2 = two-block ridge). Scoring: per-day QLIKE on remaining variance;
DM t on paired daily loss differences (iid s.e.): blk2 vs a0 within
each construction, and each construction vs M0. Also the pooled-with-
hour-dummies alternative for M2 (one fit, hour dummies) vs per-hour.

Outputs results/multihorizon/ttc_{byhour,pooled,coefs}.csv.

Opt-in (the default run's outputs are unchanged by these flags):

  --save-forecasts   also write the per-day, per-hour forecasts behind the
                     QLIKE tables (M0; a0 and blk2 M1, M2, M3, M2pooled)
                     and the realized target y to ttc<tag>_forecasts.parquet.
                     experiments/multihorizon_frem_check.py re-scores that
                     file and asserts it reproduces ttc<tag>_byhour.csv
                     byte for byte.
  --through-close    end the grid at the stamp-16:00 row. The yhat panels
                     are bar-END labelled (stamp tau = the bar [tau-30, tau]),
                     so the default grid 10:00..15:30 stops at the bar
                     ending 15:30 and RV_{t,T} omits the last RTH bar
                     (15:30-16:00); with this flag the target runs to the
                     16:00 close and 15:30 becomes an entry hour (one bar
                     left). Requires --tag.
  --blk2-file NAME   the blk2 yhat export (default yhat_blk2.parquet, the
                     run without FOMC-calendar columns);
                     yhat_blk2_fomc1.parquet is the FOMC panel of record.
                     A non-default file requires --tag.
  --tag TAG          suffix for every output file (ttc<TAG>_*.csv), so a
                     variant never overwrites the default outputs.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YH = os.path.join(ROOT, "results", "spxw_pnl")
OUT = os.path.join(ROOT, "results", "multihorizon")
BURN = 63
GRID = [f"{h:02d}:{m:02d}" for h in range(10, 16) for m in (0, 30)]  # 10:00..15:30
ENTRY = GRID[:-1]  # last bar has no remaining horizon
BLK2_FILE = "yhat_blk2.parquet"
TAG = ""
SAVE_FORECASTS = False
FORECAST_KEYS = (
    "M0",
    "a0_M1",
    "a0_M2",
    "a0_M2pooled",
    "a0_M3",
    "blk2_M1",
    "blk2_M2",
    "blk2_M2pooled",
    "blk2_M3",
)


def _ql(f: np.ndarray, y: np.ndarray) -> np.ndarray:
    f = np.maximum(f, 1e-18)
    y = np.maximum(y, 1e-18)
    return y / f - np.log(y / f) - 1.0


def _dm(d: np.ndarray) -> float:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size < 3 or float(d.std(ddof=1)) == 0.0:
        return float("nan")
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))


def _load_wide() -> dict[str, pd.DataFrame]:
    a0 = pd.read_parquet(os.path.join(YH, "yhat_a0.parquet"))
    b2 = pd.read_parquet(os.path.join(YH, BLK2_FILE))
    m = a0.merge(b2, on="t", suffixes=("_a", "_b"))
    m["pa"] = m["yhat_a"].to_numpy(float) ** 2 * m["baseline_a"].to_numpy(float)
    m["pb"] = m["yhat_b"].to_numpy(float) ** 2 * m["baseline_b"].to_numpy(float)
    m["rv"] = m["rv_raw_a"].to_numpy(float)
    et = m["t"].dt.tz_convert("America/New_York")
    m["day"] = et.dt.normalize().dt.tz_localize(None)
    m["hm"] = et.dt.strftime("%H:%M")
    m = m[m["hm"].isin(GRID)]
    wide: dict[str, pd.DataFrame] = {}
    for col in ("rv", "pa", "pb"):
        w = m.pivot_table(index="day", columns="hm", values=col, aggfunc="first")
        wide[col] = w.reindex(columns=GRID)
    keep = np.ones(len(wide["rv"]), bool)
    for col in wide:
        keep &= wide[col].notna().all(1).to_numpy() & (wide[col] > 0).all(1).to_numpy()
    for col in wide:
        wide[col] = wide[col].loc[keep]
    return wide


def _expanding_ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    n = len(y)
    Z = np.column_stack([np.ones(n), X])
    out = np.full(n, np.nan)
    for j in range(BURN, n):
        beta, *_ = np.linalg.lstsq(Z[:j], y[:j], rcond=None)
        out[j] = Z[j] @ beta
    return out


def _causal_scale(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    r = pd.Series(y / np.maximum(f, 1e-18))
    return r.expanding(min_periods=BURN).mean().shift(1).to_numpy(float) * f


ERA_SPLIT = "2016-01-01"


def byhour_table(L: dict[str, np.ndarray], days: np.ndarray) -> pd.DataFrame:
    """Per-hour QLIKE and DM rows from per-day losses L[key] (n_days x n_entry)."""
    rows_h: list[dict] = []

    def summarize(mask_days: np.ndarray, era: str) -> None:
        for i, h in enumerate(ENTRY):
            base0 = L["M0"][mask_days, i]
            for key in sorted(L):
                q = L[key][mask_days, i]
                rec = {
                    "era": era,
                    "hour": h,
                    "forecast": key,
                    "n_days": int(np.isfinite(q).sum()),
                    "qlike": float(np.nanmean(q)),
                    "dm_vs_naive": _dm(q - base0),
                }
                if key.startswith("blk2_"):
                    a_key = "a0_" + key[len("blk2_") :]
                    rec["dm_blk2_vs_a0"] = _dm(q - L[a_key][mask_days, i])
                if key.endswith("_M2pooled"):
                    rec["dm_pooled_vs_perhour"] = _dm(
                        q - L[key.replace("_M2pooled", "_M2")][mask_days, i]
                    )
                rows_h.append(rec)

    summarize(np.ones(len(days), bool), "all")
    summarize(days >= np.datetime64(ERA_SPLIT), "post2016")
    return pd.DataFrame(rows_h)


def configure(
    through_close: bool = False,
    blk2_file: str = "yhat_blk2.parquet",
    tag: str = "",
    save_forecasts: bool = False,
) -> None:
    """Set the grid, the blk2 input and the output names (defaults = the committed run)."""
    global GRID, ENTRY, BLK2_FILE, TAG, SAVE_FORECASTS
    if (through_close or blk2_file != "yhat_blk2.parquet") and not tag:
        raise SystemExit(
            "--through-close / --blk2-file need --tag: a variant never overwrites the default outputs"
        )
    GRID = [f"{h:02d}:{m:02d}" for h in range(10, 16) for m in (0, 30)]
    if through_close:
        GRID = GRID + ["16:00"]  # the bar ending at the close (bar-end labels)
    ENTRY = GRID[:-1]
    BLK2_FILE = blk2_file
    TAG = tag
    SAVE_FORECASTS = save_forecasts


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    wide = _load_wide()
    days = wide["rv"].index.to_numpy()
    rv = wide["rv"].to_numpy(float)
    pa = wide["pa"].to_numpy(float)
    pb = wide["pb"].to_numpy(float)
    n_days, m = rv.shape
    print(f"panel: {n_days} complete RTH days, {m} bars/day", flush=True)

    rem = rv[:, ::-1].cumsum(1)[:, ::-1]  # inclusive reverse cumsum
    sofar = rv.cumsum(1)
    coefs: list[dict] = []
    # collect per-day losses for pooling: dict[(model, construction)] -> (n_days x n_entry) matrix
    L: dict[str, np.ndarray] = {}
    # the forecasts behind L and the realized target (persisted only with --save-forecasts)
    F: dict[str, np.ndarray] = {}
    Y = np.full((n_days, len(ENTRY)), np.nan)

    def put(key: str, i: int, q: np.ndarray) -> None:
        if key not in L:
            L[key] = np.full((n_days, len(ENTRY)), np.nan)
        L[key][:, i] = q

    def keep(key: str, i: int, f: np.ndarray) -> None:
        if key not in F:
            F[key] = np.full((n_days, len(ENTRY)), np.nan)
        F[key][:, i] = f

    for i, h in enumerate(ENTRY):
        y = rem[:, i + 1]  # variance from bar i+1 through close
        Y[:, i] = y
        ly = np.log(y)
        lso = np.log(np.maximum(sofar[:, i], 1e-18))
        lprev = np.r_[np.nan, ly[:-1]]  # yesterday's realized same-horizon
        # M0 naive: expanding mean of log y (shifted), then causal scale
        f0 = np.exp(
            pd.Series(ly).expanding(min_periods=BURN).mean().shift(1).to_numpy(float)
        )
        f0 = _causal_scale(y, f0)
        put("M0", i, _ql(f0, y))
        keep("M0", i, f0)
        for tag, per in (("a0", pa), ("blk2", pb)):
            lv = np.log(
                np.maximum(per[:, i + 1], 1e-18)
            )  # one-step forecast issued at bar i for bar i+1
            f1 = _causal_scale(y, np.exp(_expanding_ols(ly, lv.reshape(-1, 1))))
            f2 = _causal_scale(
                y, np.exp(_expanding_ols(ly, np.column_stack([lv, lso])))
            )
            ok3 = np.isfinite(lprev)
            f3 = np.full(n_days, np.nan)
            f3[ok3] = _causal_scale(
                y[ok3],
                np.exp(
                    _expanding_ols(
                        ly[ok3], np.column_stack([lv[ok3], lso[ok3], lprev[ok3]])
                    )
                ),
            )
            put(f"{tag}_M1", i, _ql(f1, y))
            put(f"{tag}_M2", i, _ql(f2, y))
            put(f"{tag}_M3", i, _ql(f3, y))
            keep(f"{tag}_M1", i, f1)
            keep(f"{tag}_M2", i, f2)
            keep(f"{tag}_M3", i, f3)
            # descriptive full-sample coefficients for M2
            Z = np.column_stack([np.ones(n_days), lv, lso])
            beta, *_ = np.linalg.lstsq(Z, ly, rcond=None)
            coefs.append(
                {
                    "hour": h,
                    "model": tag,
                    "const": beta[0],
                    "b_onestep": beta[1],
                    "d_sofar": beta[2],
                }
            )

        # pooled-with-hour-dummies M2 handled after loop (needs all hours)

    # pooled M2: one expanding fit over all (day, hour) rows with hour dummies, per model
    for tag, per in (("a0", pa), ("blk2", pb)):
        recs = []
        for i, h in enumerate(ENTRY):
            recs.append(
                pd.DataFrame(
                    {
                        "day": days,
                        "hidx": i,
                        "ly": np.log(rem[:, i + 1]),
                        "lv": np.log(np.maximum(per[:, i + 1], 1e-18)),
                        "lso": np.log(np.maximum(sofar[:, i], 1e-18)),
                        "y": rem[:, i + 1],
                    }
                )
            )
        P = pd.concat(recs).sort_values(["day", "hidx"]).reset_index(drop=True)
        D = np.eye(len(ENTRY))[P["hidx"].to_numpy()][:, 1:]  # hour dummies (drop first)
        X = np.column_stack([P["lv"].to_numpy(float), P["lso"].to_numpy(float), D])
        # expanding by DAY, not by row: refit when the day changes
        yv = P["ly"].to_numpy(float)
        Z = np.column_stack([np.ones(len(P)), X])
        f = np.full(len(P), np.nan)
        day_arr = P["day"].to_numpy()
        uniq = np.unique(day_arr)
        day_pos = np.searchsorted(uniq, day_arr)
        beta = None
        last_fit_day = -1
        for r_ in range(len(P)):
            dpos = day_pos[r_]
            if dpos < BURN:
                continue
            if dpos != last_fit_day:
                cut = np.searchsorted(day_pos, dpos)  # first row of today
                beta, *_ = np.linalg.lstsq(Z[:cut], yv[:cut], rcond=None)
                last_fit_day = dpos
            f[r_] = Z[r_] @ beta
        P["f"] = np.exp(f)
        # causal scale per hour
        q = np.full(len(P), np.nan)
        for i in range(len(ENTRY)):
            sel = (P["hidx"] == i).to_numpy()
            fs = _causal_scale(
                P.loc[sel, "y"].to_numpy(float), P.loc[sel, "f"].to_numpy(float)
            )
            q[sel] = _ql(fs, P.loc[sel, "y"].to_numpy(float))
            # one row per (day, hour), sorted by day: this hour's rows are in `days` order
            assert (P.loc[sel, "day"].to_numpy() == days).all()
            keep(f"{tag}_M2pooled", i, fs)
        for i in range(len(ENTRY)):
            sel = (P["hidx"] == i).to_numpy()
            put(f"{tag}_M2pooled", i, q[sel])

    # ---- tables ----
    assert set(F) == set(L) == set(FORECAST_KEYS)
    for key in L:  # the forecasts kept are exactly the ones scored
        assert np.array_equal(L[key], _ql(F[key], Y), equal_nan=True), key
    all_mask = np.ones(n_days, bool)
    # era split at the panel's midpoint by date is arbitrary; use 2016-01-01 as a pre-registered split
    split = days >= np.datetime64(ERA_SPLIT)
    byh = byhour_table(L, days)
    byh.to_csv(os.path.join(OUT, f"ttc{TAG}_byhour.csv"), index=False)
    if SAVE_FORECASTS:
        fc = pd.DataFrame(
            {
                "day": np.repeat(days, len(ENTRY)),
                "hour": np.tile(np.asarray(ENTRY), n_days),
                "y": Y.reshape(-1),
            }
        )
        for key in FORECAST_KEYS:
            fc[key] = F[key].reshape(-1)
        fc.to_parquet(os.path.join(OUT, f"ttc{TAG}_forecasts.parquet"), index=False)
        print(
            f"wrote ttc{TAG}_forecasts.parquet: {len(fc)} rows "
            f"({n_days} days x {len(ENTRY)} entry hours {ENTRY[0]}..{ENTRY[-1]}; "
            f"target through the bar ending {GRID[-1]}; blk2 = {BLK2_FILE})",
            flush=True,
        )

    # pooled over hours: per-day mean loss across entry hours, then DM
    pooled: list[dict] = []
    for era, mask in (("all", all_mask), ("post2016", split)):
        base0 = np.nanmean(L["M0"][mask], axis=1)
        for key in sorted(L):
            q = np.nanmean(L[key][mask], axis=1)
            rec = {
                "era": era,
                "forecast": key,
                "n_days": int(np.isfinite(q).sum()),
                "qlike": float(np.nanmean(q)),
                "dm_vs_naive": _dm(q - base0),
            }
            if key.startswith("blk2_"):
                rec["dm_blk2_vs_a0"] = _dm(
                    q - np.nanmean(L["a0_" + key[len("blk2_") :]][mask], axis=1)
                )
            if key.endswith("_M2pooled"):
                rec["dm_pooled_vs_perhour"] = _dm(
                    q - np.nanmean(L[key.replace("_M2pooled", "_M2")][mask], axis=1)
                )
            pooled.append(rec)
    pl = pd.DataFrame(pooled)
    pl.to_csv(os.path.join(OUT, f"ttc{TAG}_pooled.csv"), index=False)
    pd.DataFrame(coefs).to_csv(os.path.join(OUT, f"ttc{TAG}_coefs.csv"), index=False)

    pd.set_option("display.width", 220)
    print("\n=== pooled over entry hours ===", flush=True)
    print(pl.round(4).to_string(index=False), flush=True)
    print("\n=== by hour, era all, M1 & M2 (blk2 vs a0) ===", flush=True)
    sub = byh[
        (byh.era == "all")
        & byh.forecast.isin(["M0", "a0_M1", "blk2_M1", "a0_M2", "blk2_M2"])
    ]
    print(sub.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="multi-horizon (time-till-close) forecasts"
    )
    ap.add_argument("--save-forecasts", action="store_true")
    ap.add_argument("--through-close", action="store_true")
    ap.add_argument("--blk2-file", default="yhat_blk2.parquet")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    configure(a.through_close, a.blk2_file, a.tag, a.save_forecasts)
    main()
