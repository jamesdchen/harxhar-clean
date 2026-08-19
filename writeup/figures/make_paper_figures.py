"""Professor-facing paper figures: diurnal RV, unpenalized buckets, block ladder.

Diurnal profile is computed from data/core_stats.parquet (mean sumret2 by
end-of-bar clock slot). Bucket ΔQLIKE and ladder QLIKE/DM are the harvested
values below; they are not recomputed.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DPI = 180

# Okabe–Ito + ink. One accent; otherwise black/gray.
INK = "#1a1a1a"
MUTED = "#6e6e6e"
GRAY_BAR = "#b8b8b8"
ACCENT = "#0072b2"
RTH_FILL = "#e8e8e8"
ZERO = "#444444"

# Unpenalized (minimum-norm) bucket ΔQLIKE vs OLS–HAR. Positive = worse.
BUCKETS = [
    ("all_features", "All exogenous", 0.00419, True),
    ("liquidity", "Liquidity", 0.00414, False),
    ("moments", "Moments", 0.00242, False),
    ("market_ew", "Market EW", 0.00167, False),
    ("market_vw", "Market VW", 0.00122, False),
    ("implied_vol", "Implied vol", 0.00104, False),
    ("vol_demand", "Vol. demand", 0.00101, False),
    ("sentiment", "Sentiment", 0.00043, False),
]

# Stated-convention block ladder. Do not add documented-convention or tree arms.
LADDER = [
    ("OLS–HAR\n(benchmark)", 0.22519, None),
    ("Two-block ridge\n(1, 100)", 0.22350, -3.1),
    ("Three-block ridge\n(1, 100, 1000)", 0.21486, -4.2),
]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def fig_diurnal() -> Path:
    src = ROOT / "data" / "core_stats.parquet"
    df = pd.read_parquet(src, columns=["endbartime", "sumret2"])
    df = df.dropna(subset=["sumret2"])
    t = pd.to_datetime(df["endbartime"])
    hour = t.dt.hour + t.dt.minute / 60.0
    mean_rv = df.groupby(hour, sort=True)["sumret2"].mean()
    x = mean_rv.index.to_numpy(dtype=float)
    y = mean_rv.to_numpy(dtype=float) * 1e6  # ×10^{-6}

    fig, ax = plt.subplots(figsize=(6.4, 3.15))
    ymax = float(np.nanmax(y)) * 1.08
    ax.axvspan(9.5, 16.0, color=RTH_FILL, lw=0, zorder=0)
    ax.text(
        12.75,
        ymax * 0.93,
        "RTH",
        ha="center",
        va="top",
        color=MUTED,
        fontsize=8,
        zorder=1,
    )
    ax.plot(x, y, color=INK, lw=1.5, solid_capstyle="round", zorder=2)
    ax.set_xlim(-0.4, 23.9)
    ax.set_ylim(0.0, ymax)
    ax.set_xticks([0, 4, 8, 12, 16, 20])
    ax.set_xticklabels(["00:00", "04:00", "08:00", "12:00", "16:00", "20:00"])
    ax.set_xlabel("Time of day (ET, bar end)")
    ax.set_ylabel(r"Mean realized variance ($\times 10^{-6}$)")
    fig.tight_layout()
    path = OUT / "fig_diurnal.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_buckets() -> Path:
    rows = sorted(BUCKETS, key=lambda r: r[2], reverse=True)
    # barh: first row at bottom; reverse so largest Δ is at the top.
    plot = list(reversed(rows))
    labels = [r[1] for r in plot]
    deltas = [r[2] for r in plot]
    colors = [ACCENT if r[3] else GRAY_BAR for r in plot]
    y = np.arange(len(plot))

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    ax.barh(y, deltas, color=colors, height=0.68, edgecolor="none", zorder=2)
    ax.axvline(0.0, color=ZERO, lw=0.9, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$\Delta$ QLIKE vs. OLS–HAR")
    xmax = max(deltas) * 1.18
    ax.set_xlim(0.0, xmax)
    for yi, d in zip(y, deltas):
        ax.text(d + 0.00006, yi, f"+{d:.5f}", va="center", ha="left", fontsize=8, color=INK)
    fig.tight_layout()
    path = OUT / "fig_buckets.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_ladder() -> Path:
    labels = [r[0] for r in LADDER]
    qlikes = [r[1] for r in LADDER]
    dms = [r[2] for r in LADDER]
    colors = [GRAY_BAR, GRAY_BAR, ACCENT]
    x = np.arange(len(LADDER))

    y0, y1 = 0.210, 0.230
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.bar(x, qlikes, color=colors, width=0.62, edgecolor="none", zorder=2)
    ax.set_ylim(y0, y1)
    ax.set_xticks(x)
    ax.set_yticks([0.210, 0.215, 0.220, 0.225, 0.230])
    ax.set_xticklabels(labels)
    ax.set_ylabel("QLIKE (axis cropped below 0.21)")
    for i, (q, dm) in enumerate(zip(qlikes, dms)):
        if dm is None:
            txt = f"{q:.5f}"
        else:
            txt = f"{q:.5f}\nDM −{abs(dm):.1f}"
        ax.text(i, q + 0.00045, txt, ha="center", va="bottom", fontsize=8, color=INK)
    fig.tight_layout()
    path = OUT / "fig_ladder.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (fig_diurnal, fig_buckets, fig_ladder):
        path = fn()
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
