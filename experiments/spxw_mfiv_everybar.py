"""Every-bar 0DTE log-strip mark-to-market.

At each RTH stamp t, C_t = integrated MFIV to the 16:00 settle.
Hold N units one bar:

  listed MTM:     N * (RV_t + C_{t+1} - C_t)
  paper (target): N * (RV_t - a0_t)
  hybrid (doc):   N * (RV_t + C_{t+1} - a0_t)

N = (blk2 - a0) / a0^2, 1% winsor. Last bar of the day has C_{t+1}=0.

Alignment: yhat-panel stamps are bar-END labelled (the row at stamp
tau carries the realized variance of [tau-30, tau] and forecasts
issued at tau-30), so the panel stamps are shifted back one bar
before the join. RV_t, a0_t and the N sizing at decision stamp t
then all refer to the bar [t, t+30] actually held -- the same window
as the strip legs C_t -> C_{t+1} -- with forecasts issued at t.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from experiments.spxw_mfiv_toclose import (  # noqa: E402
    CHAIN,
    OUT,
    PARTS,
    _mfiv_one,
    _ql,
    _sh,
    _stamp_bounds,
)

ANN_BAR = float(np.sqrt(252.0 * 48.0))
ANN_DAY = float(np.sqrt(252.0))
RTH_ENTRY = (10, 11, 12, 13, 14, 15)


def shard(part: int, parts: int) -> None:
    lo, hi = _stamp_bounds(parts)[part]
    print(f"part {part}/{parts}  {lo} -> {hi}", flush=True)
    cols = ["timestamp", "expiration", "strike", "cp", "mid", "underlying_price"]
    df = pd.read_parquet(
        CHAIN,
        columns=cols,
        filters=[("timestamp", ">=", lo), ("timestamp", "<", hi)],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["expiration"] = pd.to_datetime(df["expiration"])
    print(f"  loaded={len(df):,} stamps={df['timestamp'].nunique():,}", flush=True)
    rows = []
    for t0, snap in df.groupby("timestamp", sort=True):
        exp = pd.Timestamp(snap["expiration"].iloc[0])
        meta = _mfiv_one(snap, pd.Timestamp(t0), exp)
        if meta is None:
            continue
        rows.append(
            {
                "t": pd.Timestamp(t0),
                "expiration": exp,
                "C": float(meta["mfiv_int"]),
                "n_otm": int(meta["n_otm"]),
                "F": float(meta["F"]),
            }
        )
    out = pd.DataFrame(rows)
    os.makedirs(PARTS, exist_ok=True)
    path = os.path.join(PARTS, f"mfiv_bar_part{part}.parquet")
    out.to_parquet(path, index=False)
    print(f"  wrote {path} n={len(out)}", flush=True)


def _row(name: str, x: np.ndarray, daily: bool = False) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    sh = _sh(x)
    ann = ANN_DAY if daily else ANN_BAR
    return {
        "book": name,
        "n": int(x.size),
        "mean": float(x.mean()) if x.size else float("nan"),
        "sharpe_ann": sh * ann if np.isfinite(sh) else float("nan"),
        "hit": float((x > 0).mean()) if x.size else float("nan"),
        "unit": "day" if daily else "bar",
    }


def _print(r: dict) -> None:
    print(
        f"{r['book']:44s} n={r['n']:6d}  ann={r['sharpe_ann']:+.2f}  "
        f"hit={r['hit']:.1%}  ({r['unit']})",
        flush=True,
    )


def reduce(parts: int) -> None:
    files = [os.path.join(PARTS, f"mfiv_bar_part{i}.parquet") for i in range(parts)]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"missing shards: {missing}")
    C = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    C["t"] = pd.to_datetime(C["t"], utc=True).astype("datetime64[ns, UTC]")
    C["expiration"] = pd.to_datetime(C["expiration"])
    C = C.sort_values("t").drop_duplicates("t")
    print(f"MFIV stamps={len(C):,}  med n_otm={C['n_otm'].median():.0f}", flush=True)

    a0 = pd.read_parquet(os.path.join(OUT, "yhat_a0.parquet"))
    b2 = pd.read_parquet(os.path.join(OUT, "yhat_blk2.parquet"))
    a0["t"] = pd.to_datetime(a0["t"], utc=True).astype("datetime64[ns, UTC]")
    b2["t"] = pd.to_datetime(b2["t"], utc=True).astype("datetime64[ns, UTC]")
    yh = a0.merge(b2, on="t", suffixes=("_a", "_b"))
    yh["pa"] = yh["yhat_a"].to_numpy(float) ** 2 * yh["baseline_a"].to_numpy(float)
    yh["pb"] = yh["yhat_b"].to_numpy(float) ** 2 * yh["baseline_b"].to_numpy(float)
    yh["rv"] = yh["rv_raw_a"].to_numpy(float)
    yh = yh[["t", "pa", "pb", "rv"]].sort_values("t")
    # Bar-END-labelled panel: shift stamps back one bar so the decision
    # row at t joins the forecasts issued AT t and the realized variance
    # of the held bar [t, t+30] (previously this attached the stale
    # forecast and the accrual of the bar that had just ended).
    yh["t"] = yh["t"] - pd.Timedelta(minutes=30)

    j = pd.merge_asof(
        C.sort_values("t"),
        yh,
        on="t",
        direction="backward",
        # < one bar: match only the fresh row; a missing row yields NaN
        # (entry dropped) instead of silently falling back one bar.
        tolerance=pd.Timedelta(minutes=29),
    )
    j["et"] = j["t"].dt.tz_convert("America/New_York")
    j["hod"] = j["et"].dt.hour
    j["day"] = j["et"].dt.normalize()
    j["year"] = j["et"].dt.year
    j = j.sort_values("t")
    j["C_next"] = j.groupby("day")["C"].shift(-1)
    last = j.groupby("day")["t"].transform("max") == j["t"]
    j.loc[last, "C_next"] = 0.0

    entry = (
        j["hod"].isin(RTH_ENTRY)
        & j["pa"].notna()
        & j["C"].notna()
        & j["C_next"].notna()
    )
    j = j.loc[entry].copy()
    pa = j["pa"].to_numpy(float)
    pb = j["pb"].to_numpy(float)
    rv = j["rv"].to_numpy(float)
    c0 = j["C"].to_numpy(float)
    c1 = j["C_next"].to_numpy(float)
    ok = np.isfinite(pa) & np.isfinite(pb) & np.isfinite(rv) & (pa > 0) & (pb > 0)
    pa, pb, rv, c0, c1 = pa[ok], pb[ok], rv[ok], c0[ok], c1[ok]
    j = j.loc[ok].copy()

    gap = pb - pa
    N = gap / np.maximum(pa, 1e-18) ** 2
    lo, hi = np.nanquantile(N, [0.01, 0.99])
    N = np.clip(N, lo, hi)
    paper = N * (rv - pa)
    mtm = N * (rv + c1 - c0)
    hybrid = N * (rv + c1 - pa)
    dq = _ql(pa, rv) - _ql(pb, rv)

    et = j["et"]
    last_bar = (j.groupby("day")["t"].transform("max") == j["t"]).to_numpy()
    daily = et >= "2022-05-02"
    drop20 = et.dt.year != 2020

    rows = []
    specs = [
        ("paper N*(RV-a0) all RTH", paper, False),
        ("listed MTM N*(RV+C1-C) all RTH", mtm, False),
        ("hybrid N*(RV+C1-a0) all RTH", hybrid, False),
        ("QLIKE increment all RTH", dq, False),
        ("paper last bar only", paper[last_bar], True),
        ("listed MTM last bar (=expire)", mtm[last_bar], True),
        ("paper drop 2020", paper[drop20.to_numpy()], False),
        ("listed MTM drop 2020", mtm[drop20.to_numpy()], False),
        ("paper daily-0DTE", paper[daily.to_numpy()], False),
        ("listed MTM daily-0DTE", mtm[daily.to_numpy()], False),
        ("paper weekly-0DTE", paper[(~daily).to_numpy()], False),
        ("listed MTM weekly-0DTE", mtm[(~daily).to_numpy()], False),
    ]
    print(
        f"entries={len(j):,}  last-bar={int(last_bar.sum())}  N clip=[{lo:.3e},{hi:.3e}]",
        flush=True,
    )
    print(
        f"med C={np.median(c0):.3e}  med a0={np.median(pa):.3e}  med RV={np.median(rv):.3e}",
        flush=True,
    )
    for name, x, daily_u in specs:
        r = _row(name, x, daily_u)
        rows.append(r)
        _print(r)

    out = pd.DataFrame(rows)
    csv = os.path.join(OUT, "everybar_mtm.csv")
    out.to_csv(csv, index=False)
    j.assign(N=N, paper=paper, mtm=mtm, hybrid=hybrid, dq=dq).to_parquet(
        os.path.join(OUT, "everybar_mtm_trades.parquet"), index=False
    )
    print(f"wrote {csv}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=None)
    ap.add_argument("--parts", type=int, default=4)
    ap.add_argument("--reduce", action="store_true")
    a = ap.parse_args()
    if a.reduce:
        reduce(a.parts)
    elif a.part is not None:
        shard(a.part, a.parts)
    else:
        raise SystemExit("pass --part K --parts N  or  --reduce")


if __name__ == "__main__":
    main()
