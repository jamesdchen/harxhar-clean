"""Two sleeves, two strikes. Never mix.

A: always short 10:00 0DTE variance at the smile (VRP).
B: remaining RV vs a0, sized (blk2-a0), drop top |log| decile.
Combo is risk-parity (each sleeve unit-vol) so B is not a rounding error.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "spxw_pnl")
ANN = float(np.sqrt(252.0))


def _sh(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3 or float(x.std()) == 0.0:
        return float("nan")
    return float(x.mean() / x.std())


def _row(name: str, x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    sh = _sh(x)
    return {
        "book": name,
        "n": int(x.size),
        "mean": float(x.mean()) if x.size else float("nan"),
        "vol": float(x.std()) if x.size else float("nan"),
        "sharpe": sh,
        "sharpe_ann": sh * ANN if np.isfinite(sh) else float("nan"),
        "hit": float((x > 0).mean()) if x.size else float("nan"),
    }


def _print(r: dict) -> None:
    print(
        f"{r['book']:42s} n={r['n']:4d}  ann={r['sharpe_ann']:+.2f}  "
        f"hit={r['hit']:.1%}  mean={r['mean']:+.3e}",
        flush=True,
    )


def main() -> None:
    tr = pd.read_parquet(os.path.join(OUT, "mfiv_toclose_trades.parquet"))
    tr["t0"] = pd.to_datetime(tr["t0"], utc=True)
    tr = tr.sort_values("t0").reset_index(drop=True)
    rv = tr["rv_rem"].to_numpy(float)
    a0 = tr["pa_rem"].to_numpy(float)
    b2 = tr["pb_rem"].to_numpy(float)
    iv = tr["mfiv_int"].to_numpy(float)
    strip_long = tr["strip_pnl"].to_numpy(float)
    gap = b2 - a0
    absrel = np.abs(np.log(np.maximum(b2, 1e-18) / np.maximum(a0, 1e-18)))
    d10 = float(np.nanquantile(absrel, 0.9))
    keep_b = absrel <= d10

    A = iv - rv
    A_strip = -strip_long
    B = np.where(keep_b, gap * (rv - a0), 0.0)

    sA = float(np.nanstd(A))
    sB = float(np.nanstd(B[keep_b]))
    rp = A / (sA + 1e-18) + np.where(keep_b, (gap * (rv - a0)) / (sB + 1e-18), 0.0)
    rp_strip = A_strip / (float(np.nanstd(A_strip)) + 1e-18) + np.where(
        keep_b, (gap * (rv - a0)) / (sB + 1e-18), 0.0
    )
    dollar = A + B
    corr = float(np.corrcoef(A[keep_b], (gap * (rv - a0))[keep_b])[0, 1])

    rows = [
        _row("A  short smile (MFIV-RV)", A),
        _row("A  short strip", A_strip),
        _row("B  incr vs a0, drop D10", (gap * (rv - a0))[keep_b]),
        _row("A+B dollar (VRP swallows B)", dollar),
        _row("A+B risk-parity", rp),
        _row("A_strip+B risk-parity", rp_strip),
    ]
    print(
        f"n={len(tr)}  D10={d10:.4f}  B days={int(keep_b.sum())}  corr(A,B)={corr:+.3f}",
        flush=True,
    )
    print(
        f"vol A={sA:.3e}  vol B={sB:.3e}  volA/volB={sA / (sB + 1e-30):.1f}", flush=True
    )
    for r in rows:
        _print(r)

    out = pd.DataFrame(rows)
    csv = os.path.join(OUT, "two_sleeve.csv")
    out.to_csv(csv, index=False)
    book = tr[["t0"]].copy()
    book["A_smile"] = A
    book["A_strip"] = A_strip
    book["B"] = B
    book["B_on"] = keep_b
    book["rp"] = rp
    book["rp_strip"] = rp_strip
    pq = os.path.join(OUT, "two_sleeve_daily.parquet")
    book.to_parquet(pq, index=False)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=150)
    ax = axes[0]
    labs = [r["book"] for r in rows]
    vals = [r["sharpe_ann"] for r in rows]
    cols = ["#d62728", "#ff7f0e", "#2ca02c", "#888888", "#1f77b4", "#9467bd"]
    ax.barh(labs[::-1], vals[::-1], color=cols[::-1])
    ax.axvline(0.0, color="k", lw=0.7, alpha=0.5)
    ax.set_xlabel("ann. Sharpe")
    ax.set_title("Two strikes, two sleeves")
    ax = axes[1]
    t = np.arange(len(book))
    ax.plot(
        t,
        book["A_smile"].cumsum() / (sA + 1e-18),
        color="#d62728",
        lw=1.2,
        label="A short smile (unit vol)",
    )
    b_cs = np.cumsum(np.where(keep_b, gap * (rv - a0), 0.0)) / (sB + 1e-18)
    ax.plot(t, b_cs, color="#2ca02c", lw=1.2, label="B incr vs a0 (unit vol)")
    ax.plot(t, np.cumsum(rp), color="#1f77b4", lw=1.6, label="risk-parity")
    ax.set_xlabel("0DTE day")
    ax.set_ylabel("cumulative unit-vol PnL")
    ax.set_title("B is small in dollars, real in Sharpe")
    ax.legend(frameon=False, loc="upper left")
    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    png = os.path.join(OUT, "two_sleeve.png")
    fig.savefig(png)
    print(f"wrote {csv}", flush=True)
    print(f"wrote {pq}", flush=True)
    print(f"wrote {png}", flush=True)


if __name__ == "__main__":
    main()
