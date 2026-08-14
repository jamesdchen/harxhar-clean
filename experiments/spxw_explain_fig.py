"""One-panel explainer: three prices for the same 6-hour variance."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "spxw_pnl"
)


def main() -> None:
    tr = pd.read_parquet(os.path.join(OUT, "mfiv_toclose_trades.parquet"))
    rv = tr["rv_rem"].to_numpy(float)
    a0 = tr["pa_rem"].to_numpy(float)
    iv = tr["mfiv_int"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=150)
    # one typical day near the median RV
    i = int(np.argmin(np.abs(rv - np.median(rv))))
    names = [
        "What happened\n(realized variance)",
        "HAR's price\n(a0)",
        "Options market's price\n(the smile)",
    ]
    vals = [rv[i], a0[i], iv[i]]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    bars = ax.bar(names, vals, color=colors, width=0.62)
    ax.set_ylabel("variance from 10:00 to 16:00  (one typical day)")
    ax.set_title("You are buying the same thing at three different prices")
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v,
            f"  {v:.2e}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    # annotation
    ax.annotate(
        "HAR is about right.\nOptions charge ~6x.\nThat extra is the VRP,\nnot the forecast increment.",
        xy=(2, iv[i]),
        xytext=(1.35, iv[i] * 0.55),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="#d62728"),
        color="#333333",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(OUT, "explain_one_day.png")
    fig.savefig(path)
    print(path)
    print(
        f"example day t0={tr['t0'].iloc[i]}  RV={rv[i]:.3e} a0={a0[i]:.3e} MFIV={iv[i]:.3e}  ratio={iv[i] / rv[i]:.1f}"
    )


if __name__ == "__main__":
    main()
