"""Figures for the spectral-kNN diagnostics (reads results/spectral_graph_diagnostics/).

Four figures, one per finding:
  1. model ladder            — ridge dominates every neighborhood method
  2. spectral energy profile — the representation, not the eigenmaps, carries signal
  3. spike failure profile   — both model families fail in the top decile
  4. alt-data increments     — implied vol is the one live exog channel

Outputs PNGs to writeup/figures/.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = "results/spectral_graph_diagnostics"
OUT = "writeup/figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 10,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": False,
})


def style_ax(ax, grid_axis="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- figure 1
def fig_model_ladder():
    df = pd.read_csv(f"{RES}/knn_vs_linear.csv")
    df = pd.concat(
        [pd.DataFrame([{"model": "constant (window mean)", "mse": 0.28531}]), df]
    ).sort_values("mse")

    def klass(m):
        if m.startswith("ridge"):
            return "linear"
        if m.startswith("constant"):
            return "baseline"
        return "neighborhood"

    colors = {"linear": ORANGE, "neighborhood": BLUE, "baseline": MUTED}
    labels = {
        "ridge_multiscale": "ridge · multiscale (5 coords)",
        "ridge_har": "ridge · HAR features",
        "loclin_multiscale_k25": "local-linear · multiscale k=25",
        "loclin_har_k100": "local-linear · HAR k=100",
        "knn_multiscale_k5": "kNN · multiscale k=5",
        "knn_multiscale_k25": "kNN · multiscale k=25",
        "knn_prognostic_k25": "kNN · prognostic score k=25",
        "knn_har_betaweighted_k25": "kNN · beta-weighted HAR k=25",
        "knn_har_k25": "kNN · HAR features k=25",
        "knn_prognostic_k100": "kNN · prognostic score k=100",
        "knn_multiscale_k100": "kNN · multiscale k=100",
        "loclin_multiscale_k100": "local-linear · multiscale k=100",
    }
    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=150)
    ypos = np.arange(len(df))[::-1]
    kl = df["model"].map(klass)
    ax.barh(ypos, df["mse"], height=0.62,
            color=[colors[k] for k in kl], edgecolor=SURFACE, linewidth=1.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels.get(m, m) for m in df["model"]], color=INK2, fontsize=9.5)
    for yp, v in zip(ypos, df["mse"]):
        ax.text(v + 0.004, yp, f"{v:.4f}", va="center", color=INK2, fontsize=8.6)
    style_ax(ax)
    ax.set_xlim(0, 0.325)
    ax.set_xlabel("test MSE (transformed target) — lower is better")
    ax.set_title("Same window, same targets: linear beats every neighborhood method",
                 loc="left", fontsize=12.5, color=INK, pad=14)
    ax.text(0, 1.015, "window 2013-11 → 2015-09, chronological 80/20 split",
            transform=ax.transAxes, color=INK2, fontsize=9.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k in
               ("linear", "neighborhood", "baseline")]
    ax.legend(handles, ["linear (ridge)", "neighborhood (kNN / local-linear)",
                        "constant baseline"],
              loc="center right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/diag_model_ladder.png", bbox_inches="tight")


# ---------------------------------------------------------------- figure 2
def fig_energy_profiles():
    df = pd.read_csv(f"{RES}/energy_profiles.csv")
    win = df["window"].iloc[0]
    sub = df[(df.window == win) & (df.views == "raw") & (df.weighting == "binary")]
    series = [("multiscale", "multiscale (5 level coords)", BLUE),
              ("rawW48", "raw delay W=48", ORANGE),
              ("har", "HAR feature rows", AQUA),
              ("raw960", "raw delay W=960", YELLOW)]
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=150)
    g0 = sub[sub.representation == "raw960"].sort_values("d")
    ax.fill_between(g0.d, 100 * (g0.null_mean - 2 * g0.null_std),
                    100 * (g0.null_mean + 2 * g0.null_std),
                    color=GRID, alpha=0.9, label=None)
    ax.text(50, 3.0, "permutation null ±2σ", color=MUTED, fontsize=9)
    nudge = {"rawW48": 5, "har": -6}  # de-collide the two ~52% end labels
    for rep, label, color in series:
        g = sub[sub.representation == rep].sort_values("d")
        ax.plot(g.d, 100 * g.captured, color=color, linewidth=2)
        ax.annotate(label, (g.d.iloc[-1], 100 * g.captured.iloc[-1]),
                    xytext=(6, nudge.get(rep, 0)), textcoords="offset points",
                    va="center", color=color, fontsize=9.5)
    style_ax(ax, grid_axis="y")
    ax.set_xlim(1, 92)
    ax.set_xticks([1, 8, 16, 32, 48, 64])
    ax.set_xlabel("d — Laplacian eigenvectors kept")
    ax.set_ylabel("% of forecast-target variance captured")
    ax.set_title("The graph's smooth part holds signal only if the representation does",
                 loc="left", fontsize=12.5, color=INK, pad=14)
    ax.text(0, 1.015, f"raw-target views, binary k=10 graph, window {win}",
            transform=ax.transAxes, color=INK2, fontsize=9.5)
    ax.axvline(8, color=BASELINE, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(8.7, 20, "pipeline d=8", color=MUTED, fontsize=8.8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/diag_energy_profiles.png", bbox_inches="tight")


# ---------------------------------------------------------------- figure 3
def fig_spike_failure():
    df = pd.read_csv(f"{RES}/knn_vs_linear_deciles.csv")
    x = np.arange(10)
    w = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.4), dpi=150)
    for ax, (c1, c2), ylab, title in (
        (ax1, ("mse_knn_ms", "mse_ridge_har"), "test MSE",
         "Error concentrates in the top decile…"),
        (ax2, ("bias_knn", "bias_ridge"), "bias (pred − true)",
         "…where both models underpredict"),
    ):
        ax.bar(x - w / 2, df[c1], width=w, color=BLUE, edgecolor=SURFACE,
               linewidth=1.2, label="kNN · multiscale")
        ax.bar(x + w / 2, df[c2], width=w, color=ORANGE, edgecolor=SURFACE,
               linewidth=1.2, label="ridge · HAR")
        style_ax(ax, grid_axis="y")
        ax.set_xticks(x)
        ax.set_xlabel("true-target decile (9 = highest vol)")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=8)
        ax.axhline(0, color=BASELINE, linewidth=1)
    ax1.text(9 - w / 2, df.mse_knn_ms[9] + 0.012, f"{df.mse_knn_ms[9]:.2f}",
             ha="center", color=INK2, fontsize=8.4)
    ax1.text(9 + w / 2 + 0.06, df.mse_ridge_har[9] + 0.012,
             f"{df.mse_ridge_har[9]:.2f}", ha="center", color=INK2, fontsize=8.4)
    ax1.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("The shared failure mode: vol spikes (window 2013-11 → 2015-09)",
                 x=0.005, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"{OUT}/diag_spike_failure.png", bbox_inches="tight")


# ---------------------------------------------------------------- figure 4
def fig_altdata():
    order = ["vix_level", "vix_struct", "sent", "breadth", "jump", "flow",
             "voldemand", "ALL"]
    names = {"vix_level": "VIX level", "vix_struct": "VIX structure",
             "sent": "sentiment", "breadth": "EW/VW breadth", "jump": "jump fraction",
             "flow": "flow imbalance", "voldemand": "vol demand", "ALL": "ALL blocks"}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), dpi=150, sharey=True)
    for ax, fn, title in ((axes[0], "altdata_2020-07-01.csv",
                           "COVID test tail (Feb–Jun 2020)"),
                          (axes[1], "altdata_2019-07-01.csv",
                           "calm test tail (Feb–Jun 2019)")):
        df = pd.read_csv(f"{RES}/{fn}")
        base = df.iloc[0]
        df = df.set_index(df["block"].str.lstrip("+"))
        d_all = [100 * (df.loc[b, "mse"] / base["mse"] - 1) for b in order]
        d_top = [100 * (df.loc[b, "mse_top"] / base["mse_top"] - 1) for b in order]
        x = np.arange(len(order))
        w = 0.38
        ax.bar(x - w / 2, d_all, width=w, color=BLUE, edgecolor=SURFACE,
               linewidth=1.2, label="Δ overall MSE")
        ax.bar(x + w / 2, d_top, width=w, color=ORANGE, edgecolor=SURFACE,
               linewidth=1.2, label="Δ top-decile (spike) MSE")
        style_ax(ax, grid_axis="y")
        ax.set_xticks(x)
        ax.set_xticklabels([names[b] for b in order], fontsize=8.4, color=INK2,
                           rotation=28, ha="right")
        ax.axhline(0, color=BASELINE, linewidth=1)
        ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=8)
    axes[0].set_ylabel("Δ% vs own-history base — negative is better")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Alt data added to the own-history base: implied vol is the live channel",
                 x=0.005, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{OUT}/diag_altdata_increments.png", bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_model_ladder()
    fig_energy_profiles()
    fig_spike_failure()
    fig_altdata()
    print(f"wrote 4 figures -> {OUT}/")
