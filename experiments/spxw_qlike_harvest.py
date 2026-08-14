"""Harvest the blk2-vs-a0 QLIKE increment on the dumped series.

The weekly ATM sign book does not load on this increment. This script trades
the scoring rule itself, with the filters that actually hold the edge:

  * one-bar RV, struck at a0 (not mid-IV)
  * size = 1 (unit QLIKE) or Hessian 1/f^2, never |gap|
  * clock: 10:00-16:00 ET (skip open + overnight)
  * drop top decile of |log(blk2/a0)| (largest revisions lose)

Also overlays the same filter on the ATM weekly tape (same contracts, fewer bars).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "spxw_pnl")
PARTS = os.path.join(OUT, "parts")
ANN = float(np.sqrt(252.0 * 48.0))
RTH_HOURS = (10, 11, 12, 13, 14, 15)


def _ql(f: np.ndarray, y: np.ndarray) -> np.ndarray:
    f = np.maximum(f, 1e-18)
    y = np.maximum(y, 1e-18)
    return y / f - np.log(y / f) - 1.0


def _sh(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3 or float(x.std()) == 0.0:
        return float("nan")
    return float(x.mean() / x.std())


def _load_panel(a0_path: str, b2_path: str) -> pd.DataFrame:
    a0 = pd.read_parquet(a0_path)
    b2 = pd.read_parquet(b2_path)
    a0["t"] = pd.to_datetime(a0["t"], utc=True).astype("datetime64[ns, UTC]")
    b2["t"] = pd.to_datetime(b2["t"], utc=True).astype("datetime64[ns, UTC]")
    m = a0.merge(b2, on="t", suffixes=("_a", "_b"))
    m["pa"] = m["yhat_a"].to_numpy(float) ** 2 * m["baseline_a"].to_numpy(float)
    m["pb"] = m["yhat_b"].to_numpy(float) ** 2 * m["baseline_b"].to_numpy(float)
    rv = m["rv_raw_a"].to_numpy(float)
    pa = m["pa"].to_numpy(float)
    pb = m["pb"].to_numpy(float)
    ok = (
        np.isfinite(rv)
        & np.isfinite(pa)
        & np.isfinite(pb)
        & (rv > 0)
        & (pa > 0)
        & (pb > 0)
    )
    m = m.loc[ok].copy()
    m["rv"] = m["rv_raw_a"].to_numpy(float)
    m["dq"] = _ql(m["pa"].to_numpy(float), m["rv"].to_numpy(float)) - _ql(
        m["pb"].to_numpy(float), m["rv"].to_numpy(float)
    )
    m["gap"] = m["pb"] - m["pa"]
    m["absrel"] = np.abs(np.log(m["pb"] / m["pa"]))
    m["et"] = m["t"].dt.tz_convert("America/New_York")
    m["hod"] = m["et"].dt.hour
    m["rth_post_open"] = m["hod"].isin(RTH_HOURS)
    return m


def _row(name: str, x: np.ndarray, extra: dict | None = None) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    sh = _sh(x)
    out = {
        "book": name,
        "n": int(x.size),
        "mean": float(x.mean()) if x.size else float("nan"),
        "sharpe_bar": sh,
        "sharpe_ann": sh * ANN if np.isfinite(sh) else float("nan"),
        "hit": float((x > 0).mean()) if x.size else float("nan"),
    }
    if extra:
        out.update(extra)
    return out


def _hess_pay(m: pd.DataFrame, clip: float = 0.01) -> np.ndarray:
    """QLIKE-Hessian size (gap / f^2) * (rv - a0), winsorized."""
    pa = m["pa"].to_numpy(float)
    gap = m["gap"].to_numpy(float)
    rv = m["rv"].to_numpy(float)
    size = gap / np.maximum(pa, 1e-18) ** 2
    lo, hi = np.nanquantile(size, [clip, 1.0 - clip])
    size = np.clip(size, lo, hi)
    return size * (rv - pa)


def harvest(m: pd.DataFrame) -> pd.DataFrame:
    d10 = float(m["absrel"].quantile(0.9))
    m = m.copy()
    m["keep_mod"] = m["absrel"] <= d10
    filt = m["rth_post_open"] & m["keep_mod"]
    am = m["hod"].isin([10, 11]) & m["keep_mod"]
    rows = [
        _row("qlike full panel", m["dq"], {"filter": "none"}),
        _row("qlike RTH 10-16", m.loc[m["rth_post_open"], "dq"], {"filter": "clock"}),
        _row("qlike RTH 10-16 drop D10", m.loc[filt, "dq"], {"filter": "clock+D10"}),
        _row("qlike AM 10-12 drop D10", m.loc[am, "dq"], {"filter": "AM+D10"}),
        _row(
            "mse-varswap full (gap*(rv-mid))",
            m["gap"] * (m["rv"] - 0.5 * (m["pa"] + m["pb"])),
            {"filter": "none"},
        ),
        _row(
            "mse-varswap RTH 10-16 drop D10",
            (
                m.loc[filt, "gap"]
                * (m.loc[filt, "rv"] - 0.5 * (m.loc[filt, "pa"] + m.loc[filt, "pb"]))
            ),
            {"filter": "clock+D10"},
        ),
        _row("hess-varswap full (winsor 1%)", _hess_pay(m), {"filter": "none"}),
        _row(
            "hess-varswap RTH 10-16 drop D10",
            _hess_pay(m.loc[filt]),
            {"filter": "clock+D10"},
        ),
        _row(
            "hess-varswap AM 10-12 drop D10", _hess_pay(m.loc[am]), {"filter": "AM+D10"}
        ),
    ]
    return pd.DataFrame(rows), d10, filt


def _spxw_stamps() -> pd.DatetimeIndex:
    path = os.path.join(PARTS, "h1_sweep.parquet")
    tr = pd.read_parquet(path, columns=["t0"])
    t0 = pd.to_datetime(tr["t0"], utc=True).astype("datetime64[ns, UTC]")
    return pd.DatetimeIndex(np.sort(t0.drop_duplicates()))


def harvest_spxw(m: pd.DataFrame, d10: float) -> pd.DataFrame:
    t0 = _spxw_stamps()
    sub = pd.merge_asof(
        pd.DataFrame({"t": t0}),
        m.sort_values("t"),
        on="t",
        direction="backward",
    )
    sub["keep_mod"] = sub["absrel"] <= d10
    filt = sub["rth_post_open"] & sub["keep_mod"]
    rows = [
        _row("qlike SPXW stamps", sub["dq"], {"filter": "SPXW"}),
        _row(
            "qlike SPXW skip open",
            sub.loc[sub["rth_post_open"], "dq"],
            {"filter": "SPXW+clock"},
        ),
        _row(
            "qlike SPXW skip open drop D10",
            sub.loc[filt, "dq"],
            {"filter": "SPXW+clock+D10"},
        ),
        _row(
            "hess-varswap SPXW skip open drop D10",
            _hess_pay(sub.loc[filt]),
            {"filter": "SPXW+clock+D10"},
        ),
    ]
    return pd.DataFrame(rows), sub, filt


def overlay_weeklies(m: pd.DataFrame, d10: float) -> pd.DataFrame:
    """ATM weekly increment Sharpe on all bars vs filter-passing bars."""
    yh = m.sort_values("t")[["t", "pa", "pb", "absrel", "rth_post_open", "dq", "gap"]]
    files = sorted(
        glob.glob(os.path.join(PARTS, "h*_sweep.parquet")),
        key=lambda p: int(os.path.basename(p).split("_")[0][1:]),
    )
    rows = []
    for path in files:
        if os.path.basename(path) in {"h16_all_all.parquet", "h4_all_all.parquet"}:
            continue
        tr = pd.read_parquet(path)
        tr["t0"] = pd.to_datetime(tr["t0"], utc=True).astype("datetime64[ns, UTC]")
        h = int(tr["h"].iloc[0])
        j = pd.merge_asof(
            tr.sort_values("t0"), yh, left_on="t0", right_on="t", direction="backward"
        )
        d = j["d_long"].to_numpy(float)
        gap = j["gap"].to_numpy(float)
        z = gap / (np.nanstd(gap) + 1e-18)
        keep = j["rth_post_open"].to_numpy(bool) & (j["absrel"].to_numpy(float) <= d10)
        rows.append(
            {
                "h": h,
                "n_all": int(np.isfinite(d).sum()),
                "incr_all": _sh(z * d),
                "n_filt": int(keep.sum()),
                "incr_filt": _sh(z[keep] * d[keep]),
                "qlike_all": _sh(j["dq"].to_numpy(float)),
                "qlike_filt": _sh(j.loc[keep, "dq"].to_numpy(float)),
            }
        )
    return pd.DataFrame(rows)


def _plot(books: pd.DataFrame, weekly: pd.DataFrame, out: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), dpi=150)
    q = books[books["book"].str.startswith("qlike")].copy()
    ax = axes[0]
    colors = [
        "#888888" if "full" in b or b.endswith("stamps") else "#1f77b4"
        for b in q["book"]
    ]
    ax.barh(q["book"], q["sharpe_ann"], color=colors)
    ax.axvline(0.0, color="k", lw=0.7, alpha=0.5)
    ax.set_xlabel("ann. Sharpe of ΔQLIKE")
    ax.set_title("QLIKE harvest")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.plot(
        weekly["h"],
        weekly["incr_all"],
        "o--",
        color="#888888",
        lw=1.5,
        ms=4,
        label="weekly incr, all bars",
    )
    ax.plot(
        weekly["h"],
        weekly["incr_filt"],
        "o-",
        color="#d62728",
        lw=2.0,
        ms=4,
        label="weekly incr, filter",
    )
    ax.axhline(0.0, color="k", lw=0.7, alpha=0.45)
    ax.set_xticks(list(weekly["h"]))
    ax.set_xlabel("horizon h")
    ax.set_ylabel("Sharpe / trade")
    ax.set_title("ATM weekly overlay")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default=os.path.join(OUT, "yhat_a0.parquet"))
    ap.add_argument("--blk2", default=os.path.join(OUT, "yhat_blk2.parquet"))
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    m = _load_panel(a.a0, a.blk2)
    books, d10, _ = harvest(m)
    spxw_books, _, _ = harvest_spxw(m, d10)
    weekly = overlay_weeklies(m, d10)
    all_books = pd.concat([books, spxw_books], ignore_index=True)

    print(f"D10 |log(blk2/a0)| threshold = {d10:.4f}", flush=True)
    print(
        "book                              n     mean        sh/bar   ann    hit",
        flush=True,
    )
    for _, r in all_books.iterrows():
        print(
            f"{r['book']:32s} {r['n']:6.0f}  {r['mean']:+.4e}  {r['sharpe_bar']:+.4f}  "
            f"{r['sharpe_ann']:+.2f}  {r['hit']:.1%}",
            flush=True,
        )
    print("\nh  n_all incr_all n_filt incr_filt qlike_all qlike_filt", flush=True)
    for _, r in weekly.iterrows():
        print(
            f"{int(r['h']):2d} {int(r['n_all']):5d} {r['incr_all']:+.3f}  "
            f"{int(r['n_filt']):5d} {r['incr_filt']:+.3f}  "
            f"{r['qlike_all']:+.3f} {r['qlike_filt']:+.3f}",
            flush=True,
        )

    csv_b = os.path.join(a.out, "qlike_harvest.csv")
    csv_w = os.path.join(a.out, "qlike_harvest_weekly.csv")
    png = os.path.join(a.out, "qlike_harvest.png")
    all_books.to_csv(csv_b, index=False)
    weekly.to_csv(csv_w, index=False)
    _plot(all_books, weekly, png)
    print(f"wrote {csv_b}", flush=True)
    print(f"wrote {csv_w}", flush=True)
    print(f"wrote {png}", flush=True)


if __name__ == "__main__":
    main()
