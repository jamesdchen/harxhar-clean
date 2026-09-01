"""Paper comparison: a0 vs blk2 on the 10:00-16:00 0DTE variance claim.

Same days, same realized path, same smile. Only the forecast changes.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "spxw_pnl")
ANN = float(np.sqrt(252.0))


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


def _row(name: str, x: np.ndarray, extra: dict | None = None) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    sh = _sh(x)
    out = {
        "book": name,
        "n": int(x.size),
        "mean": float(x.mean()) if x.size else float("nan"),
        "sharpe_ann": sh * ANN if np.isfinite(sh) else float("nan"),
        "hit": float((x > 0).mean()) if x.size else float("nan"),
        "frac_long": float("nan"),
    }
    if extra:
        out.update(extra)
    return out


def _print(r: dict) -> None:
    fl = r.get("frac_long", float("nan"))
    fls = f"  long={fl:.1%}" if np.isfinite(fl) else ""
    print(
        f"{r['book']:48s} n={r['n']:4d}  ann={r['sharpe_ann']:+.2f}  "
        f"hit={r['hit']:.1%}{fls}",
        flush=True,
    )


def main() -> None:
    tr = pd.read_parquet(os.path.join(OUT, "mfiv_toclose_trades.parquet"))
    rv = tr["rv_rem"].to_numpy(float)
    # At-entry (F_t-measurable) remaining forecasts are the decision
    # inputs. The p*_rem sums condition on bars inside the window and are
    # kept only as labelled ex-post record rows ("pRem ..." below).
    a0 = tr["F_a0_rem"].to_numpy(float)
    b2 = tr["F_b2_rem"].to_numpy(float)
    pa = tr["pa_rem"].to_numpy(float)
    pb = tr["pb_rem"].to_numpy(float)
    iv = tr["mfiv_int"].to_numpy(float)
    ok = np.isfinite(rv) & np.isfinite(a0) & np.isfinite(b2) & np.isfinite(iv)
    rv, a0, b2, iv = rv[ok], a0[ok], b2[ok], iv[ok]
    pa, pb = pa[ok], pb[ok]

    ql_a = _ql(a0, rv)
    ql_b = _ql(b2, rv)
    rows = []

    # Forecast quality on this exact claim
    rows.append(_row("QLIKE remaining a0 (loss)", ql_a))
    rows.append(_row("QLIKE remaining blk2 (loss)", ql_b))
    rows.append(_row("QLIKE increment (a0-blk2)", ql_a - ql_b))
    # Ex-post record: the within-window p*_rem sums (not decision-quotable)
    rows.append(_row("pRem QLIKE remaining a0 (ex-post record)", _ql(pa, rv)))
    rows.append(_row("pRem QLIKE remaining blk2 (ex-post record)", _ql(pb, rv)))
    rows.append(
        _row("pRem QLIKE increment (ex-post record)", _ql(pa, rv) - _ql(pb, rv))
    )

    # Same strategy: trade remaining variance vs the smile, forecast = f
    for name, f in (("a0", a0), ("blk2", b2)):
        sgn = np.sign(f - iv)
        rows.append(
            _row(
                f"sign(f-MFIV)*(RV-MFIV)  {name}",
                sgn * (rv - iv),
                {"frac_long": float((sgn > 0).mean())},
            )
        )
        rows.append(
            _row(
                f"(f-MFIV)*(RV-MFIV)  {name}",
                (f - iv) * (rv - iv),
            )
        )

    # Always-short smile (no model)
    rows.append(_row("always short smile (MFIV-RV)", iv - rv, {"frac_long": 0.0}))

    # Model as strike (calibration PnL, not a cross-model vote)
    rows.append(_row("RV-a0  (a0 as strike)", rv - a0))
    rows.append(_row("RV-blk2 (blk2 as strike)", rv - b2))

    # Increment: the only book that isolates blk2 vs a0
    gap = b2 - a0
    absrel = np.abs(np.log(np.maximum(b2, 1e-18) / np.maximum(a0, 1e-18)))
    keep = absrel <= np.nanquantile(absrel, 0.9)
    rows.append(_row("(blk2-a0)*(RV-a0)  all days", gap * (rv - a0)))
    rows.append(_row("(blk2-a0)*(RV-a0)  drop D10", (gap * (rv - a0))[keep]))

    print(
        f"n={int(ok.sum())}  P(a0<MFIV)={(a0 < iv).mean():.1%}  P(blk2<MFIV)={(b2 < iv).mean():.1%}",
        flush=True,
    )
    print(
        f"mean QLIKE a0={ql_a.mean():.5f}  blk2={ql_b.mean():.5f}  dQL={ql_a.mean() - ql_b.mean():+.5f}",
        flush=True,
    )
    for r in rows:
        _print(r)

    out = pd.DataFrame(rows)
    path = os.path.join(OUT, "a0_vs_blk2_strategy.csv")
    out.to_csv(path, index=False)

    weekly_path = os.path.join(OUT, "complete_table.csv")
    if os.path.exists(weekly_path):
        w = pd.read_csv(weekly_path)
        print("\nweekly ATM mid-IV sign (already scored)", flush=True)
        print(
            w[["h", "n", "long_sh", "a0_sh", "b2_sh"]].to_string(index=False),
            flush=True,
        )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), dpi=150)
    ax = axes[0]
    labs = ["a0", "blk2"]
    ax.bar(labs, [ql_a.mean(), ql_b.mean()], color=["#1f77b4", "#d62728"], width=0.55)
    ax.set_ylabel("QLIKE  (lower is better)")
    ax.set_title("10:00→16:00 remaining variance")
    for i, v in enumerate([ql_a.mean(), ql_b.mean()]):
        ax.text(i, v, f"  {v:.4f}", ha="center", va="bottom")

    ax = axes[1]
    names = [
        "always short smile",
        "a0 sign vs smile",
        "blk2 sign vs smile",
        "a0 size vs smile",
        "blk2 size vs smile",
        "incr (blk2-a0) drop D10",
    ]
    vals = [
        _sh(iv - rv) * ANN,
        _sh(np.sign(a0 - iv) * (rv - iv)) * ANN,
        _sh(np.sign(b2 - iv) * (rv - iv)) * ANN,
        _sh((a0 - iv) * (rv - iv)) * ANN,
        _sh((b2 - iv) * (rv - iv)) * ANN,
        _sh((gap * (rv - a0))[keep]) * ANN,
    ]
    cols = ["#888888", "#1f77b4", "#d62728", "#5fa8d3", "#e07a5f", "#2ca02c"]
    ax.barh(names[::-1], vals[::-1], color=cols[::-1])
    ax.axvline(0.0, color="k", lw=0.7, alpha=0.5)
    ax.set_xlabel("ann. Sharpe")
    ax.set_title("Same claim, a0 vs blk2 as the forecast")
    for a in axes:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    png = os.path.join(OUT, "a0_vs_blk2_strategy.png")
    fig.savefig(png)
    print(f"wrote {path}", flush=True)
    print(f"wrote {png}", flush=True)


if __name__ == "__main__":
    main()
