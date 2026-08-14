"""0DTE MFIV log-strip at 10:00 ET, settled at 16:00.

Shard the chain by timestamp range (--part/--parts). Reduce joins the
dumped a0/blk2 remaining-path book and scores paper vs strip vs VRP.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "spxw_pnl")
PARTS = os.path.join(OUT, "parts")
CHAIN = os.path.join(ROOT, "data", "spxw_chain.parquet")
ANN_DAY = float(np.sqrt(252.0))
RTH = (10, 11, 12, 13, 14, 15)


def _sh(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3 or float(x.std()) == 0.0:
        return float("nan")
    return float(x.mean() / x.std())


def _ql(f: np.ndarray, y: np.ndarray) -> np.ndarray:
    f = np.maximum(f, 1e-18)
    y = np.maximum(y, 1e-18)
    return y / f - np.log(y / f) - 1.0


def _stamp_bounds(parts: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    st = pd.read_parquet(os.path.join(OUT, "zdte_stamps.parquet"))
    t = pd.to_datetime(st["t"], utc=True).sort_values().to_numpy()
    cuts = np.linspace(0, len(t), parts + 1, dtype=int)
    bounds = []
    for i in range(parts):
        lo = pd.Timestamp(t[cuts[i]])
        hi = (
            pd.Timestamp(t[-1] + np.timedelta64(1, "s"))
            if i == parts - 1
            else pd.Timestamp(t[cuts[i + 1]])
        )
        bounds.append((lo, hi))
    return bounds


def _mfiv_one(snap: pd.DataFrame, t0: pd.Timestamp, exp: pd.Timestamp) -> dict | None:
    live = snap[np.isfinite(snap["mid"]) & (snap["mid"].to_numpy(float) > 0)].copy()
    if len(live) < 8:
        return None
    live["cp"] = live["cp"].astype(str).str.upper().str[0]
    spot = live["underlying_price"].dropna()
    S0 = float(spot.iloc[-1]) if len(spot) else float("nan")
    both = live.pivot_table(index="strike", columns="cp", values="mid", aggfunc="last")
    if "C" not in both.columns or "P" not in both.columns:
        return None
    pc = both.dropna(subset=["C", "P"])
    if pc.empty:
        return None
    F_imp = pc.index.to_numpy(float) + pc["C"].to_numpy(float) - pc["P"].to_numpy(float)
    F = float(np.median(F_imp)) if F_imp.size else S0
    if not np.isfinite(F) or F <= 0:
        return None
    puts = live[live["cp"] == "P"][["strike", "mid"]].drop_duplicates("strike")
    calls = live[live["cp"] == "C"][["strike", "mid"]].drop_duplicates("strike")
    otm_p = puts[puts["strike"] < F]
    otm_c = calls[calls["strike"] > F]
    k0_cands = np.unique(
        np.concatenate(
            [puts["strike"].to_numpy(float), calls["strike"].to_numpy(float)]
        )
    )
    k0_cands = k0_cands[k0_cands <= F]
    if k0_cands.size == 0:
        return None
    K0 = float(k0_cands.max())
    q_rows = []
    for _, r in otm_p.iterrows():
        q_rows.append((float(r["strike"]), float(r["mid"]), "P"))
    for _, r in otm_c.iterrows():
        q_rows.append((float(r["strike"]), float(r["mid"]), "C"))
    # K0: average of C and P if both exist
    c0 = calls.loc[np.isclose(calls["strike"], K0), "mid"]
    p0 = puts.loc[np.isclose(puts["strike"], K0), "mid"]
    if len(c0) and len(p0):
        q_rows.append((K0, 0.5 * (float(c0.iloc[-1]) + float(p0.iloc[-1])), "K0"))
    elif len(c0):
        q_rows.append((K0, float(c0.iloc[-1]), "K0"))
    elif len(p0):
        q_rows.append((K0, float(p0.iloc[-1]), "K0"))
    if len(q_rows) < 8:
        return None
    q = (
        pd.DataFrame(q_rows, columns=["K", "Q", "kind"])
        .drop_duplicates("K")
        .sort_values("K")
    )
    K = q["K"].to_numpy(float)
    Q = q["Q"].to_numpy(float)
    dK = np.empty_like(K)
    if K.size == 1:
        return None
    dK[0] = K[1] - K[0]
    dK[-1] = K[-1] - K[-2]
    if K.size > 2:
        dK[1:-1] = 0.5 * (K[2:] - K[:-2])
    contrib = float(np.sum((dK / np.maximum(K, 1e-8) ** 2) * Q))
    close = pd.Timestamp(exp)
    if close.tzinfo is None:
        close = close.tz_localize("America/New_York")
    close = (close.normalize() + pd.Timedelta(hours=16)).tz_convert("UTC")
    T = (close - pd.Timestamp(t0)).total_seconds() / (365.25 * 24 * 3600)
    if T <= 0:
        return None
    fwd = (F / K0 - 1.0) ** 2
    mfiv_int = 2.0 * contrib - fwd
    return {
        "t0": pd.Timestamp(t0),
        "expiration": pd.Timestamp(exp),
        "F": F,
        "S0": S0,
        "K0": K0,
        "T": float(T),
        "n_otm": int(K.size),
        "mfiv_int": float(mfiv_int),
        "strip_cost": float(2.0 * contrib),
        "K": K,
        "dK": dK,
        "kind": q["kind"].to_numpy(),
    }


def _strip_T(K: np.ndarray, dK: np.ndarray, kind: np.ndarray, S: float) -> float:
    """Vanilla-strip mark at expiry: 2 Σ (ΔK/K²) intrinsic. No VIX forward term."""
    intr = np.where(kind == "C", np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    intr = np.where(kind == "K0", np.abs(S - K), intr)
    return 2.0 * float(np.sum((dK / np.maximum(K, 1e-8) ** 2) * intr))


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
    et = df["timestamp"].dt.tz_convert("America/New_York")
    df = df.assign(hour=et.dt.hour, minute=et.dt.minute)
    entry = df[(df["hour"] == 10) & (df["minute"] == 0)]
    close = df[(df["hour"] == 16) & (df["minute"] == 0)]
    print(
        f"  loaded={len(df):,}  10:00 rows={len(entry):,}  16:00 rows={len(close):,}",
        flush=True,
    )
    sT = (
        close.dropna(subset=["underlying_price"])
        .groupby(close["expiration"])["underlying_price"]
        .last()
    )
    rows = []
    for t0, snap in entry.groupby("timestamp", sort=True):
        exp = pd.Timestamp(snap["expiration"].iloc[0])
        meta = _mfiv_one(snap, pd.Timestamp(t0), exp)
        if meta is None:
            continue
        ST = float(sT.get(exp, np.nan))
        if not np.isfinite(ST):
            u = snap["underlying_price"].dropna()
            ST = float(u.iloc[-1]) if len(u) else float("nan")
        if not np.isfinite(ST):
            continue
        strip_t = _strip_T(meta.pop("K"), meta.pop("dK"), meta.pop("kind"), ST)
        meta["S_T"] = ST
        meta["strip_T"] = float(strip_t)
        meta["strip_pnl"] = float(strip_t - meta["strip_cost"])
        rows.append(meta)
    out = pd.DataFrame(rows)
    os.makedirs(PARTS, exist_ok=True)
    path = os.path.join(PARTS, f"mfiv_part{part}.parquet")
    out.to_parquet(path, index=False)
    print(f"  wrote {path} n={len(out)}", flush=True)


def _entries() -> pd.DataFrame:
    a0 = pd.read_parquet(os.path.join(OUT, "yhat_a0.parquet"))
    b2 = pd.read_parquet(os.path.join(OUT, "yhat_blk2.parquet"))
    a0["t"] = pd.to_datetime(a0["t"], utc=True).astype("datetime64[ns, UTC]")
    b2["t"] = pd.to_datetime(b2["t"], utc=True).astype("datetime64[ns, UTC]")
    m = a0.merge(b2, on="t", suffixes=("_a", "_b"))
    m["pa"] = m["yhat_a"].to_numpy(float) ** 2 * m["baseline_a"].to_numpy(float)
    m["pb"] = m["yhat_b"].to_numpy(float) ** 2 * m["baseline_b"].to_numpy(float)
    m["rv"] = m["rv_raw_a"].to_numpy(float)
    ok = np.isfinite(m["rv"]) & np.isfinite(m["pa"]) & np.isfinite(m["pb"])
    ok &= (m["rv"] > 0) & (m["pa"] > 0) & (m["pb"] > 0)
    m = m.loc[ok].copy()
    m["et"] = m["t"].dt.tz_convert("America/New_York")
    m["day"] = m["et"].dt.normalize()
    m["hod"] = m["et"].dt.hour
    m["dow"] = m["et"].dt.dayofweek
    m["absrel"] = np.abs(np.log(m["pb"] / m["pa"]))
    d10 = float(m["absrel"].quantile(0.9))
    rth = m[m["hod"].isin(RTH) & (m["dow"] < 5) & (m["et"] >= "2020-01-01")].copy()
    gr = rth.groupby("day", sort=False)
    rth["rv_rem"] = gr["rv"].transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1])
    rth["pa_rem"] = gr["pa"].transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1])
    rth["pb_rem"] = gr["pb"].transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1])
    first = rth[rth["hod"] == 10].groupby("day", sort=False).head(1)
    first["d10"] = d10
    return first


def _report(name: str, x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    sh = _sh(x)
    row = {
        "book": name,
        "n": int(x.size),
        "mean": float(x.mean()) if x.size else float("nan"),
        "sharpe": sh,
        "sharpe_ann": sh * ANN_DAY if np.isfinite(sh) else float("nan"),
        "hit": float((x > 0).mean()) if x.size else float("nan"),
    }
    print(
        f"{name:42s} n={row['n']:5d}  mean={row['mean']:+.4e}  "
        f"ann={row['sharpe_ann']:+.2f}  hit={row['hit']:.1%}",
        flush=True,
    )
    return row


def reduce(parts: int) -> None:
    files = [os.path.join(PARTS, f"mfiv_part{i}.parquet") for i in range(parts)]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"missing shards: {missing}")
    strip = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    strip["t0"] = pd.to_datetime(strip["t0"], utc=True).astype("datetime64[ns, UTC]")
    print(
        f"strip rows={len(strip):,}  med n_otm={strip['n_otm'].median():.0f}",
        flush=True,
    )
    ent = _entries()
    j = pd.merge_asof(
        strip.sort_values("t0"),
        ent.sort_values("t")[
            ["t", "pa", "pb", "rv", "pa_rem", "pb_rem", "rv_rem", "absrel", "d10"]
        ],
        left_on="t0",
        right_on="t",
        direction="backward",
        tolerance=pd.Timedelta("40min"),
    )
    j = j.dropna(subset=["pa_rem", "mfiv_int", "rv_rem", "strip_pnl"])
    d10 = float(j["d10"].iloc[0])
    keep = j["absrel"] <= d10
    f = j.loc[keep].copy()
    gap = f["pb_rem"] - f["pa_rem"]
    size_u = gap.to_numpy(float)
    hess = size_u / np.maximum(f["pa_rem"].to_numpy(float), 1e-18) ** 2
    lo, hi = np.nanquantile(hess, [0.01, 0.99])
    hess = np.clip(hess, lo, hi)
    rv = f["rv_rem"].to_numpy(float)
    a0 = f["pa_rem"].to_numpy(float)
    iv = f["mfiv_int"].to_numpy(float)
    sp = f["strip_pnl"].to_numpy(float)
    print(f"joined n={len(j)}  D10 keep={int(keep.sum())}  thr={d10:.4f}", flush=True)
    print(
        f"med rv_rem={np.median(rv):.3e}  med a0={np.median(a0):.3e}  "
        f"med mfiv={np.median(iv):.3e}  med strip_pnl={np.median(sp):.3e}",
        flush=True,
    )
    rows = [
        _report("paper unit (RV-a0)*gap", size_u * (rv - a0)),
        _report("paper hess (RV-a0)*1/f2", hess * (rv - a0)),
        _report(
            "paper QLIKE remaining", _ql(a0, rv) - _ql(f["pb_rem"].to_numpy(float), rv)
        ),
        _report("VRP unit (RV-MFIV)*gap", size_u * (rv - iv)),
        _report("VRP hess (RV-MFIV)*1/f2", hess * (rv - iv)),
        _report("strip unit (strip_T-MFIV)*gap", size_u * sp),
        _report("strip hess (strip_T-MFIV)*1/f2", hess * sp),
        _report("strip+shift unit ~paper", size_u * (sp + iv - a0)),
        _report("strip+shift hess ~paper", hess * (sp + iv - a0)),
        _report("unsigned (RV-a0)", rv - a0),
        _report("unsigned (RV-MFIV)", rv - iv),
        _report("unsigned strip_pnl", sp),
    ]
    out = pd.DataFrame(rows)
    path = os.path.join(OUT, "mfiv_toclose.csv")
    out.to_csv(path, index=False)
    jpath = os.path.join(OUT, "mfiv_toclose_trades.parquet")
    f.to_parquet(jpath, index=False)
    print(f"wrote {path}", flush=True)
    print(f"wrote {jpath}", flush=True)


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
        raise SystemExit("pass --part K --parts N  or  --reduce --parts N")


if __name__ == "__main__":
    main()
