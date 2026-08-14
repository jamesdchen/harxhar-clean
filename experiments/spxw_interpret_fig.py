"""One figure: why a0-struck RV works and the smile does not."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "spxw_pnl")

# Okabe–Ito
INK = "#1a1a1a"
MUTED = "#6e6e6e"
C_RV = "#0072b2"
C_A0 = "#009e73"
C_MFIV = "#d55e00"
C_STRIP = "#e69f00"
C_QLIKE = "#0072b2"
C_UNIT = "#56b4e9"
C_GRAY = "#7f7f7f"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
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


def _letter(ax, s: str) -> None:
    ax.text(
        0.0,
        1.03,
        s,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def main() -> None:
    _style()
    tr = pd.read_parquet(os.path.join(OUT, "mfiv_toclose_trades.parquet"))
    books = pd.read_csv(os.path.join(OUT, "mfiv_toclose.csv"))
    tr["t0"] = pd.to_datetime(tr["t0"], utc=True)
    ts = tr.sort_values("t0").copy()
    t_et = ts["t0"].dt.tz_convert("America/New_York")

    med_rv = float(tr["rv_rem"].median())
    med_a0 = float(tr["pa_rem"].median())
    med_mf = float(tr["mfiv_int"].median())
    smile_x = med_mf / med_rv
    ratio_a = (tr["rv_rem"] / tr["pa_rem"]).to_numpy(float)
    ratio_i = (tr["rv_rem"] / tr["mfiv_int"]).to_numpy(float)
    med_ra = float(np.nanmedian(ratio_a))
    med_ri = float(np.nanmedian(ratio_i))
    n = int(len(tr))

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), dpi=160)

    # --- A: Sharpe by strike -------------------------------------------------
    ax = axes[0, 0]
    want = [
        ("paper QLIKE remaining", "QLIKE remaining  (RV vs a0)", C_QLIKE),
        ("paper unit (RV-a0)*gap", "unit increment, a0 strike", C_A0),
        ("unsigned (RV-a0)", "always-long RV vs a0", C_GRAY),
        ("VRP unit (RV-MFIV)*gap", "unit increment, MFIV strike", C_MFIV),
        ("strip unit (strip_T-MFIV)*gap", "log-strip on smile", C_STRIP),
        ("unsigned (RV-MFIV)", "always-long RV vs smile", "#a34a1a"),
    ]
    labs, vals, cols = [], [], []
    for name, lab, c in want:
        row = books.loc[books["book"] == name]
        if row.empty:
            continue
        labs.append(lab)
        vals.append(float(row["sharpe_ann"].iloc[0]))
        cols.append(c)
    y = np.arange(len(labs))
    ax.barh(y, vals, color=cols, height=0.72, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labs)
    ax.axvline(0.0, color=INK, lw=0.7, alpha=0.55)
    xmax = max(vals) if vals else 1.0
    xmin = min(vals) if vals else -1.0
    pad = 0.08 * (xmax - xmin + 1e-9)
    ax.set_xlim(xmin - 2.4 * pad, xmax + 1.8 * pad)
    for yi, v in zip(y, vals):
        ha = "left" if v >= 0 else "right"
        ax.text(
            v + (0.12 if v >= 0 else -0.12),
            yi,
            f"{v:+.2f}",
            va="center",
            ha=ha,
            fontsize=8,
            color=INK,
        )
    ax.set_xlabel("ann. Sharpe  (10:00→16:00, drop D10)")
    ax.set_title("PnL lives at the a0 strike")
    _letter(ax, "A")

    # --- B: levels, log, dates ----------------------------------------------
    ax = axes[0, 1]
    roll = 21
    ax.plot(t_et, ts["rv_rem"], lw=0.45, color=C_RV, alpha=0.28, zorder=1)
    ax.plot(t_et, ts["pa_rem"], lw=0.45, color=C_A0, alpha=0.28, zorder=1)
    ax.plot(t_et, ts["mfiv_int"], lw=0.45, color=C_MFIV, alpha=0.28, zorder=1)
    ax.plot(
        t_et,
        ts["rv_rem"].rolling(roll, min_periods=8).median(),
        lw=1.55,
        color=C_RV,
        label="RV to close",
        zorder=3,
    )
    ax.plot(
        t_et,
        ts["pa_rem"].rolling(roll, min_periods=8).median(),
        lw=1.55,
        color=C_A0,
        label="a0 remaining",
        zorder=3,
    )
    ax.plot(
        t_et,
        ts["mfiv_int"].rolling(roll, min_periods=8).median(),
        lw=1.55,
        color=C_MFIV,
        label="MFIV (smile)",
        zorder=3,
    )
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("0DTE day")
    ax.set_ylabel("integrated variance, 10:00–16:00")
    ax.set_title(f"Smile is {smile_x:.1f}× realized  (a0 tracks RV)")
    ax.legend(frameon=False, loc="upper left", ncol=1)
    ax.text(
        0.98,
        0.04,
        f"median RV   {med_rv:.1e}\nmedian a0   {med_a0:.1e}\nmedian MFIV {med_mf:.1e}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        family="monospace",
        color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.5},
    )
    _letter(ax, "B")

    # --- C: scatter ----------------------------------------------------------
    ax = axes[1, 0]
    ax.scatter(
        tr["pa_rem"],
        tr["rv_rem"],
        s=10,
        alpha=0.38,
        c=C_A0,
        linewidths=0,
        label="RV vs a0",
        zorder=3,
    )
    ax.scatter(
        tr["mfiv_int"],
        tr["rv_rem"],
        s=10,
        alpha=0.32,
        c=C_MFIV,
        linewidths=0,
        label="RV vs MFIV",
        zorder=2,
    )
    lo = float(
        np.nanmin([tr["pa_rem"].min(), tr["rv_rem"].min(), tr["mfiv_int"].min()])
    )
    hi = float(
        np.nanmax([tr["pa_rem"].max(), tr["rv_rem"].max(), tr["mfiv_int"].max()])
    )
    ax.plot([lo, hi], [lo, hi], color=INK, ls="--", lw=0.85, alpha=0.65, zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo * 0.7, hi * 1.4)
    ax.set_ylim(lo * 0.7, hi * 1.4)
    ax.set_xlabel("forecast / implied  (10:00)")
    ax.set_ylabel("realized remaining  (→16:00)")
    ax.set_title("a0 sits on the diagonal; MFIV sits to the right")
    ax.legend(frameon=False, loc="upper left")
    _letter(ax, "C")

    # --- D: ratios -----------------------------------------------------------
    ax = axes[1, 1]
    bins = np.linspace(0.0, 3.0, 41)
    ax.hist(
        np.clip(ratio_a, 0, 3),
        bins=bins,
        alpha=0.72,
        color=C_A0,
        label="RV / a0",
        zorder=2,
    )
    ax.hist(
        np.clip(ratio_i, 0, 3),
        bins=bins,
        alpha=0.62,
        color=C_MFIV,
        label="RV / MFIV",
        zorder=3,
    )
    ax.axvline(1.0, color=INK, lw=0.85, alpha=0.65)
    ax.axvline(med_ra, color=C_A0, lw=1.15, ls="--", alpha=0.9)
    ax.axvline(med_ri, color=C_MFIV, lw=1.15, ls="--", alpha=0.9)
    ymax = ax.get_ylim()[1]
    ax.text(med_ra + 0.04, 0.92 * ymax, f"med {med_ra:.2f}", color=C_A0, fontsize=8)
    ax.text(med_ri + 0.04, 0.78 * ymax, f"med {med_ri:.2f}", color=C_MFIV, fontsize=8)
    ax.set_xlabel("realized / price  (clipped at 3)")
    ax.set_ylabel("days")
    ax.set_title("Buying a0 is roughly fair; buying the smile is not")
    ax.legend(frameon=False, loc="upper right")
    _letter(ax, "D")

    qlike = float(
        books.loc[books["book"] == "paper QLIKE remaining", "sharpe_ann"].iloc[0]
    )
    umfiv = float(
        books.loc[books["book"] == "unsigned (RV-MFIV)", "sharpe_ann"].iloc[0]
    )
    fig.suptitle(
        "0DTE 10:00–16:00 variance  —  increment lives at the a0 (HAR) strike, not on the smile",
        fontsize=12.5,
        fontweight="bold",
        y=1.015,
    )
    fig.text(
        0.5,
        -0.012,
        f"n={n} days, 2020–2024, drop D10.  "
        f"Median integrated var: RV {med_rv:.1e},  a0 {med_a0:.1e},  "
        f"MFIV {med_mf:.1e}  ({smile_x:.1f}× realized).  "
        f"QLIKE remaining Sharpe {qlike:+.2f};  unsigned RV–MFIV {umfiv:+.2f}.",
        ha="center",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout()
    path = os.path.join(OUT, "interpret_strike.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path, flush=True)
    print(
        f"med RV={med_rv:.3e}  med a0={med_a0:.3e}  med MFIV={med_mf:.3e}  "
        f"med MFIV/RV={smile_x:.2f}  QLIKE={qlike:+.2f}  unsigned MFIV={umfiv:+.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
