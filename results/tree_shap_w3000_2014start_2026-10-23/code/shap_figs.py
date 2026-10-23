"""Figures of the tree SHAP study (16:9 pages, parquet variable names, a boxed "Measure:" line each; the style of the
linear FI pages, make_fi_pages_codes.py) and one combined PDF.  Reads the tables written by shap_reduce.py.

    python shap_figs.py  -> figures/*.png, tree_shap_figures.pdf
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

import shap_common as C  # noqa: E402

RES = C.RES
FIG = RES / "figures"
FIG.mkdir(exist_ok=True)
FOOTER = (
    "Pooled trees, full set (564 columns), W = 3,000 days, 126 monthly refits 2014–2024 (the walk's refit at "
    "each origin); TreeSHAP (path-dependent) on the 33,970 forecast rows, adjusted scale"
)
INK, INK2, MUTED, SHADE, RULE, GRID = (
    "#1f1f1d",
    "#4a4a46",
    "#7a7a74",
    "#f2f1ed",
    "#bdbcb6",
    "#e4e3de",
)
BOXFC = "#f7f6f2"
NS_C, ST_C = "#eb6834", "#2a78d6"
EC = {"lgbm": "#2a78d6", "xgb": "#1baf7a", "rf_skx": "#6250d6"}
LC = "#7a7a74"  # linear reference
BLUE_RAMP = [
    "#f4f8fd",
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]
SEQ = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
MINUS = "−"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8a8a85",
        "axes.labelcolor": "#2b2b29",
        "xtick.color": INK2,
        "ytick.color": INK2,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)
EN = C.EST_NAME
MODELS = "LightGBM / XGBoost / random forest"
PAGES: list[str] = []
F_SHARE = (
    r"$s_A=\Sigma_i\,|\Sigma_{j\in A}\,\phi_{ij}|\;/\;\Sigma_B\,\Sigma_i\,|\Sigma_{j\in B}\,\phi_{ij}|$"
    "    (i = forecast row, n = 33,970; j = column; A, B = the 48 variables; $\\phi$ = TreeSHAP value under the "
    "row's origin refit)"
)


def frame(fig, title: str, subtitle: str | None) -> None:
    fig.text(
        0.04,
        0.955,
        title,
        fontsize=28,
        fontweight="bold",
        color=INK,
        ha="left",
        va="top",
    )
    if subtitle:
        fig.text(
            0.04,
            0.892,
            subtitle,
            fontsize=15,
            color=MUTED,
            ha="left",
            va="top",
            linespacing=1.35,
        )
    fig.text(0.04, 0.020, FOOTER, fontsize=11, color=MUTED, ha="left", va="bottom")


def measure_box(
    fig, top: float, words: str, formula: str, left=0.04, right=0.96, fs=13.5
) -> float:
    """The boxed "Measure:" line; the font shrinks (to 10.5 at least) until the text fits the box."""
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    while True:
        lab = fig.text(
            left + 0.012,
            top - 0.014,
            "Measure:",
            fontsize=fs,
            fontweight="bold",
            color=INK,
            ha="left",
            va="top",
            zorder=2,
        )
        bb = lab.get_window_extent(r).transformed(inv)
        tw = fig.text(
            bb.x1 + 0.007,
            top - 0.014,
            words,
            fontsize=fs,
            color=INK,
            ha="left",
            va="top",
            zorder=2,
            linespacing=1.35,
        )
        bw = tw.get_window_extent(r).transformed(inv)
        bottom, x1 = bw.y0 - 0.014, bw.x1
        tf = None
        if formula:
            tf = fig.text(
                bb.x1 + 0.007,
                bw.y0 - 0.012,
                formula,
                fontsize=fs * 0.95,
                color=INK2,
                ha="left",
                va="top",
                zorder=2,
            )
            bf = tf.get_window_extent(r).transformed(inv)
            bottom, x1 = bf.y0 - 0.014, max(x1, bf.x1)
        if x1 < right - 0.006 or fs <= 10.5:
            break
        for t in (lab, tw, tf):
            if t is not None:
                t.remove()
        fs -= 0.5
    assert x1 < right - 0.004, ("measure box overflow", x1, words[:40])
    fig.add_artist(
        FancyBboxPatch(
            (left, bottom),
            right - left,
            top - bottom,
            boxstyle="round,pad=0,rounding_size=0.008",
            fc=BOXFC,
            ec=INK2,
            lw=1.3,
            transform=fig.transFigure,
            zorder=1,
        )
    )
    return bottom


def check_text(fig, name: str) -> None:
    r = fig.canvas.get_renderer()
    for t in fig.texts:
        bb = t.get_window_extent(r).transformed(fig.transFigure.inverted())
        assert bb.x1 <= 0.99 and bb.y0 >= 0.0, (
            name,
            "text clipped",
            t.get_text()[:50],
            bb,
        )


def save(fig, name: str) -> None:
    check_text(fig, name)
    fig.savefig(FIG / name, dpi=150, facecolor="white")
    plt.close(fig)
    PAGES.append(name)
    print("wrote", FIG / name)


def p(v: float, d: int = 1) -> str:
    return f"{100 * v:.{d}f}".replace("-", MINUS)


# ============================================================================ data
m1 = pd.read_csv(RES / "m1_variables.csv")
bk = pd.read_csv(RES / "m1_buckets.csv")
cls = pd.read_csv(RES / "m1_classes.csv")
by_year = pd.read_csv(RES / "m1_by_year.csv")
m2 = pd.read_csv(RES / "m2_rank.csv")
md = pd.read_csv(RES / "m2_nonstandard_D.csv")
m3 = pd.read_csv(RES / "m3_shapes.csv")
m3b = pd.read_csv(RES / "m3_bins.csv")
m4 = pd.read_csv(RES / "m4_subsets.csv")
con = pd.read_csv(RES / "m4_contrasts.csv")
m5s = pd.read_csv(RES / "m5_summary.csv")
m5b = pd.read_csv(RES / "m5_buckets.csv")
m5p = pd.read_csv(RES / "m5_pairs_nonstandard.csv")
ests = [e for e in C.EST if e in set(m1["est"])]
lin = pd.read_csv(RES / "m2_linear_shares.csv")

# ============================================================================ P1 M1 buckets
fig = plt.figure(figsize=(16, 9), dpi=150)
nsx = cls[cls["cls"] == "NON-STANDARD"].set_index("est")
frame(
    fig,
    "M1  Where the trees put their attribution",
    f"{MODELS}: the 36 NON-STANDARD variables hold "
    + " / ".join(f"{p(nsx.at[e, 'share'])}" for e in ests)
    + "% of the total; the HAR terms "
    + " / ".join(
        f"{p(bk.set_index(['est', 'bucket']).at[(e, 'har_ma'), 'share'])}" for e in ests
    )
    + "%.",
)
yb = measure_box(
    fig,
    0.845,
    "share of a bucket = sum of its variables' shares; a variable's share = its mean |SHAP| "
    "(the SHAP values of all its columns summed\nper row, absolute value, averaged over the forecast rows) "
    "over the same summed over all 48 variables.  [ ] = 95% block-bootstrap interval over origins.",
    F_SHARE,
)
cols = [("Bucket", 0.0, 0.30, "left"), ("Class", 0.30, 0.13, "left")] + [
    (f"{EN[e]}\n(% of total)", 0.43 + 0.19 * i, 0.17, "right")
    for i, e in enumerate(ests)
]
NSB = {"sentiment", "vol_demand", "fomc", "market_ew", "market_vw"}
rows = []
bki = bk.set_index(["est", "bucket"])
for b in C.BUCKETS:
    cl = (
        "mixed"
        if b in ("liquidity", "implied_vol")
        else ("NON-STD" if b in NSB else "STANDARD")
    )
    rows.append(
        [C.BUCKET_NAME[b], cl]
        + [
            f"{p(bki.at[(e, b), 'share'])}  [{p(bki.at[(e, b), 'lo'])}, "
            f"{p(bki.at[(e, b), 'hi'])}]"
            for e in ests
        ]
    )
clsi = cls.set_index(["est", "cls"])
for name, lab in (
    ("STANDARD", "STANDARD (12 variables)"),
    ("NON-STANDARD", "NON-STANDARD (36 variables)"),
):
    rows.append(
        [lab, ""]
        + [
            f"{p(clsi.at[(e, name), 'share'])}  [{p(clsi.at[(e, name), 'lo'])}, "
            f"{p(clsi.at[(e, name), 'hi'])}]"
            for e in ests
        ]
    )
L, R_ = 0.04, 0.96
Wd = R_ - L
y = yb - 0.02
for hdr, x0, w, al in cols:
    xx = L + Wd * (x0 + (w if al == "right" else 0))
    fig.text(
        xx,
        y - 0.01,
        hdr,
        fontsize=13,
        color=INK2,
        fontweight="bold",
        ha=al,
        va="top",
        linespacing=1.2,
    )
y -= 0.075
fig.add_artist(plt.Line2D([L, R_], [y, y], color=INK2, lw=1.4))
rh = 0.043
for i, row in enumerate(rows):
    if i == len(C.BUCKETS):
        fig.add_artist(plt.Line2D([L, R_], [y, y], color=INK2, lw=1.0))
    if i % 2 == 0:
        fig.add_artist(
            plt.Rectangle((L, y - rh), Wd, rh, fc=SHADE, ec="none", zorder=0)
        )
    for (hdr, x0, w, al), v in zip(cols, row):
        xx = L + Wd * (x0 + (w if al == "right" else 0))
        bold = i >= len(C.BUCKETS)
        col = NS_C if (v == "NON-STD" or (bold and "NON" in row[0])) else INK
        fig.text(
            xx,
            y - rh / 2,
            v,
            fontsize=14,
            color=col,
            ha=al,
            va="center",
            fontweight="bold" if bold else "normal",
        )
    y -= rh
fig.add_artist(plt.Line2D([L, R_], [y, y], color=RULE, lw=1.1))
fig.text(
    0.04,
    y - 0.015,
    "Liquidity: E-mini sumvolume / numobs (STANDARD) and the 12 constituent liquidity variables "
    "(NON-STANDARD); implied volatility: vix (STANDARD), vvix and vix3m (NON-STANDARD).",
    fontsize=11.5,
    color=MUTED,
    ha="left",
    va="top",
)
save(fig, "p1_m1_buckets.png")

# ============================================================================ P2 M1 top variables
fig = plt.figure(figsize=(16, 9), dpi=150)
frame(fig, "M1  The variables the trees use most", None)
top15 = {
    e: m1[m1["est"] == e].sort_values("share", ascending=False).head(15) for e in ests
}
all_used = all((t["K_used"] == 126).all() for t in top15.values())
yb = measure_box(
    fig,
    0.885,
    "share of total mean |SHAP| per variable (all its columns summed per row), the 15 largest of each model; "
    "blue = STANDARD,\norange = NON-STANDARD.  "
    + (
        "Every refit's trees split on each of these variables (126 of 126)."
        if all_used
        else "In brackets: refits (of 126) whose trees split on the variable, where fewer than 126."
    )
    + "  HAR-terms bar cut at the axis end.",
    "",
)
for i, e in enumerate(ests):
    ax = fig.add_axes((0.165 + 0.305 * i, 0.11, 0.13, yb - 0.17))
    t = top15[e].iloc[::-1]
    colr = [NS_C if c == "NON-STANDARD" else ST_C for c in t["cls"]]
    xmax = 100 * t.loc[t["variable"] != "har_ma", "share"].max() * 1.35
    ax.barh(
        np.arange(len(t)), np.minimum(100 * t["share"], xmax), color=colr, height=0.7
    )
    ax.set_yticks(np.arange(len(t)))
    ax.set_yticklabels(
        [v if u == 126 else f"{v} [{u}]" for v, u in zip(t["variable"], t["K_used"])],
        fontsize=10.5,
        family="DejaVu Sans Mono",
    )
    for yi, v in enumerate(100 * t["share"]):
        ax.text(min(v, xmax), yi, f" {v:.1f}", va="center", fontsize=10, color=INK)
    ax.set_xlabel("% of total mean |SHAP|", fontsize=11)
    ax.set_title(EN[e], fontsize=15, color=INK, loc="left", fontweight="bold")
    ax.set_xlim(0, xmax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
save(fig, "p2_m1_top_variables.png")

# ============================================================================ P3 M1 by year
fig = plt.figure(figsize=(16, 9), dpi=150)
frame(fig, "M1  Non-standard share by year", None)
yb = measure_box(
    fig,
    0.885,
    "the M1 shares computed on the forecast rows of each calendar year (2024: January–"
    "April).  Left: NON-STANDARD share of the total;\nright: HAR-terms share.  Each row is explained by "
    "the refit of its own origin (the origin's 21 sessions).",
    "",
)
yrs = sorted(by_year["year"].unique())
for j, (what, lab) in enumerate(
    (
        ("NON-STANDARD", "NON-STANDARD share (% of total)"),
        ("har_ma", "HAR-terms share (% of total)"),
    )
):
    ax = fig.add_axes((0.07 + 0.47 * j, 0.10, 0.40, yb - 0.17))
    for e in ests:
        t = by_year[by_year["est"] == e]
        v = (
            (
                t[t["cls"] == what]
                if what == "NON-STANDARD"
                else t[t["variable"] == what]
            )
            .groupby("year")["share"]
            .sum()
        )
        ax.plot(yrs, 100 * v.reindex(yrs), "-o", color=EC[e], lw=2.2, ms=5, label=EN[e])
    ax.set_ylabel(lab)
    ax.set_xticks(yrs)
    ax.set_xticklabels([str(y) for y in yrs], fontsize=11)
    ax.grid(color=GRID, lw=0.8)
    ax.set_ylim(bottom=0)
    if j == 0:
        ax.legend(frameon=False, fontsize=12)
save(fig, "p3_m1_by_year.png")

# ============================================================================ P4 M2 vs ridge
fig = plt.figure(figsize=(16, 9), dpi=150)
r46 = m2[(m2["scope"] == "46 exogenous") & (m2["linear"] == "ridge")].set_index("est")
frame(
    fig,
    "M2  Trees against the linear ridge",
    f"{MODELS} vs ridge, 46 exogenous variables: Spearman "
    + " / ".join(f"{r46.at[e, 'spearman']:+.2f}" for e in ests).replace("-", MINUS)
    + "; top-10 overlap "
    + " / ".join(f"{r46.at[e, 'top10_overlap']}" for e in ests)
    + " of 10.",
)
yb = measure_box(
    fig,
    0.845,
    "x: tree share of the exogenous total (M1 numerator over the 46 exogenous variables); "
    "y: ridge s.d.-weighted share $\\Sigma_k\\Sigma_{j\\in A}|\\hat\\beta_{kj}|\\,sd_{kj}$ / the same over "
    "the\n46 exogenous variables (the W = 3,000 ridge walk at the same 126 origins).  Orange = NON-STANDARD;"
    " dashed = equal shares.",
    "",
)
lsh = lin[lin["linear"] == "ridge"].set_index("variable")["share_exog"]
for i, e in enumerate(ests):
    ax = fig.add_axes((0.065 + 0.32 * i, 0.30, 0.25, yb - 0.37))
    t = m1[(m1["est"] == e) & m1["variable"].isin(C.EXOG)].set_index("variable")
    x = 100 * t["share_exog"]
    yv = 100 * lsh.reindex(t.index)
    c = [NS_C if cl == "NON-STANDARD" else ST_C for cl in t["cls"]]
    ax.scatter(x.clip(lower=1e-3), yv, c=c, s=26, zorder=3)
    lo, hi = 1e-2, max(x.max(), yv.max()) * 1.6
    ax.plot([lo, hi], [lo, hi], "--", color=MUTED, lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(max(lo, x[x > 0].min() * 0.6) if (x > 0).any() else lo, hi)
    ax.set_ylim(max(lo, yv.min() * 0.6), hi)
    labs = sorted(
        set(x.sort_values().index[-4:]) | set(yv.sort_values().index[-3:]),
        key=lambda v: -yv[v],
    )
    for n_, v in enumerate(labs):
        ax.annotate(
            v,
            (max(x[v], 1e-3), yv[v]),
            fontsize=8.5,
            color=INK2,
            xytext=(4, 9 - 9 * (n_ % 3)),
            textcoords="offset points",
            family="DejaVu Sans Mono",
            arrowprops=dict(arrowstyle="-", color=RULE, lw=0.6),
        )
    ax.set_xlabel(f"{EN[e]}: % of exogenous mean|SHAP|", fontsize=12)
    if i == 0:
        ax.set_ylabel("ridge: % of exogenous $\\Sigma|\\hat\\beta|$·s.d.", fontsize=12)
    ax.grid(color=GRID, lw=0.6, which="major")
    ax.set_title(
        f"ρ = {r46.at[e, 'spearman']:+.2f}".replace("-", MINUS),
        fontsize=13,
        loc="left",
        color=INK,
    )
y = 0.205
fig.text(
    0.04,
    y,
    "NON-STANDARD share of the exogenous total, tree vs ridge (D = tree − ridge, paired block "
    "bootstrap 95%):",
    fontsize=13.5,
    color=INK,
    fontweight="bold",
    ha="left",
    va="top",
)
dd = md[
    (md["linear"] == "ridge") & (md["denominator"] == "of exogenous total")
].set_index("est")
for i, e in enumerate(ests):
    r = dd.loc[e]
    fig.text(
        0.04 + 0.31 * i,
        y - 0.05,
        f"{EN[e]}: tree {p(r.tree)}%, ridge {p(r.linear_share)}%\nD = {p(r.D)} "
        f"[{p(r.lo)}, {p(r.hi)}] pp  → {r.verdict}",
        fontsize=13,
        color=INK,
        ha="left",
        va="top",
        linespacing=1.4,
    )
save(fig, "p4_m2_vs_ridge.png")

# ============================================================================ P5 M3 shapes (one page per model)
for e in ests:
    t = m3[m3["est"] == e].sort_values("share", ascending=False).reset_index(drop=True)
    if not len(t):
        continue
    n = len(t)
    ncol = 4 if n > 6 else 3
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(16, 9), dpi=150)
    frame(fig, f"M3  Response shapes: {EN[e]}", None)
    yb = measure_box(
        fig,
        0.885,
        "mean SHAP of the variable (all columns summed) by decile of its dominant value column "
        "(the model input: trailing-window robust-scaled value), as % of the\nmean forecast; band = 10"
        "–90% of the row SHAP.  Label = the RULE's shape class (FLAT: range < half the mean |SHAP|).",
        "",
    )
    gw, gh = 0.92 / ncol, (yb - 0.10) / nrow
    for i, r in t.iterrows():
        rr, cc = divmod(i, ncol)
        ax = fig.add_axes(
            (0.06 + cc * gw, yb - 0.035 - (rr + 1) * gh + 0.055, gw - 0.05, gh - 0.085)
        )
        b = m3b[(m3b["est"] == e) & (m3b["variable"] == r["variable"])].sort_values(
            "bin"
        )
        mf = r["mean_forecast"]
        ax.fill_between(
            b["bin"],
            100 * b["p10_shap_var"] / mf,
            100 * b["p90_shap_var"] / mf,
            color=NS_C,
            alpha=0.15,
            lw=0,
        )
        ax.plot(b["bin"], 100 * b["mean_shap_var"] / mf, "-o", color=NS_C, lw=2, ms=4)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(b["bin"])
        ax.set_xticklabels([f"{v:.2g}" for v in b["x_mid"]], fontsize=7.5, rotation=45)
        ax.tick_params(axis="y", labelsize=9)
        lab = r["shape"] + (
            f" + {r['asymmetry']}"
            if isinstance(r["asymmetry"], str) and r["asymmetry"]
            else ""
        )
        ax.set_title(
            f"{r['variable']}  {p(r['share'], 2)}%"
            + (" (<0.5%)" if r["selection"] != ">= 0.5%" else ""),
            fontsize=10.5,
            loc="left",
            color=INK,
            family="DejaVu Sans Mono",
        )
        ax.text(
            0.02,
            0.96,
            lab,
            transform=ax.transAxes,
            fontsize=9,
            color=INK,
            va="top",
            fontweight="bold",
        )
        ax.text(
            0.98,
            0.04,
            r["dominant_column"],
            transform=ax.transAxes,
            fontsize=7.5,
            color=MUTED,
            ha="right",
            va="bottom",
            family="DejaVu Sans Mono",
        )
    save(fig, f"p5_m3_shapes_{e}.png")

# ============================================================================ P6 M4 conditional
fig = plt.figure(figsize=(16, 9), dpi=150)
frame(fig, "M4  When do the trees use the non-standard variables?", None)
yb = measure_box(
    fig,
    0.885,
    "NON-STANDARD share of total mean |SHAP| (M1) on subsets of forecast rows: by VIX tercile "
    "at the origin (prior-day close, terciles of the 126 origins),\nby bar of the day, and on scheduled "
    "FOMC statement days vs other days.  Bars/lines: point estimate; whiskers: 95% block-bootstrap.",
    "",
)
axs = [
    fig.add_axes((0.06, 0.30, 0.26, yb - 0.38)),
    fig.add_axes((0.38, 0.30, 0.34, yb - 0.38)),
    fig.add_axes((0.78, 0.30, 0.18, yb - 0.38)),
]
wd = 0.8 / len(ests)
for i, e in enumerate(ests):
    t = m4[(m4["est"] == e) & (m4["kind"] == "vix")]
    xpos = np.arange(len(t)) + (i - (len(ests) - 1) / 2) * wd
    axs[0].bar(xpos, 100 * t["ns_share"], width=wd, color=EC[e], label=EN[e])
    axs[0].errorbar(
        xpos,
        100 * t["ns_share"],
        yerr=[100 * (t["ns_share"] - t["ns_lo"]), 100 * (t["ns_hi"] - t["ns_share"])],
        fmt="none",
        ecolor=INK2,
        lw=1,
    )
    t = m4[(m4["est"] == e) & (m4["kind"] == "bar")]
    axs[1].plot(np.arange(len(t)), 100 * t["ns_share"], "-o", color=EC[e], lw=2, ms=4)
    axs[1].fill_between(
        np.arange(len(t)),
        100 * t["ns_lo"],
        100 * t["ns_hi"],
        color=EC[e],
        alpha=0.12,
        lw=0,
    )
    t = m4[(m4["est"] == e) & (m4["kind"] == "fomc")]
    xpos = np.arange(len(t)) + (i - (len(ests) - 1) / 2) * wd
    axs[2].bar(xpos, 100 * t["ns_share"], width=wd, color=EC[e])
    axs[2].errorbar(
        xpos,
        100 * t["ns_share"],
        yerr=[100 * (t["ns_share"] - t["ns_lo"]), 100 * (t["ns_hi"] - t["ns_share"])],
        fmt="none",
        ecolor=INK2,
        lw=1,
    )
axs[0].set_xticks(range(3))
axs[0].set_xticklabels(["low VIX", "mid", "high VIX"])
bars = m4[(m4["est"] == ests[0]) & (m4["kind"] == "bar")]["subset"].tolist()
axs[1].set_xticks(range(len(bars)))
axs[1].set_xticklabels(bars, rotation=45, fontsize=10)
axs[2].set_xticks(range(2))
axs[2].set_xticklabels(["FOMC day", "other day"])
for ax, ttl in zip(
    axs,
    (
        "by VIX tercile at the origin",
        "by bar of the day (bar end, ET)",
        "FOMC statement days",
    ),
):
    ax.set_title(ttl, fontsize=13, loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
axs[0].set_ylabel("NON-STANDARD share (% of total)")
axs[1].legend(
    [plt.Line2D([], [], color=EC[e], lw=2.2) for e in ests],
    [EN[e] for e in ests],
    frameon=False,
    fontsize=11,
    loc="upper left",
)
axs[0].set_ylim(0, 1.15 * axs[0].get_ylim()[1])
y = 0.205
for i, e in enumerate(ests):
    c = con[(con["est"] == e) & (con["scope"] == "of total")].set_index("contrast")
    cf = con[(con["est"] == e) & (con["scope"] == "bucket fomc of total")].set_index(
        "contrast"
    )
    txt = (
        f"{EN[e]}: VIX high − low {p(c.at['VIX high - low', 'diff'])} pp ({c.at['VIX high - low', 'verdict']}); "
        f"FOMC day − other {p(c.at['FOMC day - other day', 'diff'])} pp "
        f"({c.at['FOMC day - other day', 'verdict']}; FOMC-calendar bucket {p(cf.at['FOMC day - other day', 'diff'])} pp); "
        f"16:00 − 10:00 {p(c.at['16:00 - 10:00', 'diff'])} pp ({c.at['16:00 - 10:00', 'verdict']})"
    )
    fig.text(0.04, y - 0.045 * i, txt, fontsize=11.5, color=INK, ha="left", va="top")
save(fig, "p6_m4_conditional.png")

# ============================================================================ P7 M5 interactions
fig = plt.figure(figsize=(16, 9), dpi=150)
e5 = [e for e in ests if e in set(m5s["est"])]
s5 = m5s.set_index("est")
frame(
    fig,
    "M5  Interactions",
    " / ".join(EN[e] for e in e5)
    + ": interactions carry "
    + " / ".join(p(s5.at[e, "inter_share_variable"]) for e in e5)
    + "% of all attribution (variable level);\nthrough interactions: of a NON-STANDARD variable's attribution "
    + " / ".join(p(s5.at[e, "inter_frac_nonstandard"]) for e in e5)
    + "%, of a STANDARD one's "
    + " / ".join(p(s5.at[e, "inter_frac_standard"]) for e in e5)
    + "%.",
)
yb = measure_box(
    fig,
    0.80,
    "SHAP interaction values $\\Phi_{ijl}$ on 1,008 fixed rows (8 a neighbourhood), summed to "
    "variables: $G_{iAB}=\\Sigma_{j\\in A}\\Sigma_{l\\in B}\\Phi_{ijl}$.  Cell (P, Q) of the bucket map: "
    "mean |2$\\Sigma_{A\\in P,B\\in Q}G_{iAB}$|\n(P ≠ Q) or mean |$\\Sigma_{A,B\\in P}G_{iAB}$| "
    "(diagonal: main effects + within-bucket interactions), as % of the sum over the upper triangle.",
    "",
)
short = {
    "har_ma": "HAR",
    "calendar": "calendar",
    "moments": "moments",
    "liquidity": "liquidity",
    "market_ew": "mkt EW",
    "market_vw": "mkt VW",
    "sentiment": "StockTwits",
    "implied_vol": "impl. vol",
    "vol_demand": "vol demand",
    "fomc": "FOMC",
}
for i, e in enumerate(e5):
    ax = fig.add_axes((0.065 + 0.31 * i, 0.10, 0.27, yb - 0.16))
    t = m5b[m5b["est"] == e]
    nb = len(C.BUCKETS)
    Mx = np.full((nb, nb), np.nan)
    for r in t.itertuples():
        a_, b_ = C.BUCKETS.index(r.bucket_a), C.BUCKETS.index(r.bucket_b)
        Mx[b_, a_] = 100 * r.share
    ax.imshow(
        np.log10(np.where(np.isnan(Mx), np.nan, np.maximum(Mx, 1e-3))),
        cmap=SEQ,
        vmin=-2,
        vmax=2,
    )
    for a_ in range(nb):
        for b_ in range(nb):
            if not np.isnan(Mx[b_, a_]):
                v = Mx[b_, a_]
                ax.text(
                    a_,
                    b_,
                    f"{v:.1f}" if v >= 0.1 else ("<.1" if v > 0 else "0"),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if v > 10 else INK,
                )
    ax.set_xticks(range(nb))
    ax.set_yticks(range(nb))
    ax.set_xticklabels(
        [short[b] for b in C.BUCKETS], rotation=60, fontsize=9, ha="right"
    )
    ax.set_yticklabels([short[b] for b in C.BUCKETS], fontsize=9)
    ax.set_title(EN[e], fontsize=14, loc="left", color=INK, fontweight="bold")
    ax.spines[:].set_visible(False)
x0 = 0.085 + 0.31 * len(e5)
fig.text(
    x0,
    yb - 0.02,
    "Top 10 pairs with a NON-STANDARD\nvariable (% of all pair cells)",
    fontsize=12.5,
    color=INK,
    fontweight="bold",
    ha="left",
    va="top",
)
yy = yb - 0.10
for e in e5:
    fig.text(
        x0,
        yy,
        EN[e],
        fontsize=11.5,
        color=EC[e],
        fontweight="bold",
        ha="left",
        va="top",
    )
    yy -= 0.028
    for r in m5p[m5p["est"] == e].head(10).itertuples():
        fig.text(
            x0,
            yy,
            f"{r.var_a} × {r.var_b}  {p(r.share, 2)}",
            fontsize=9,
            color=INK,
            ha="left",
            va="top",
            family="DejaVu Sans Mono",
        )
        yy -= 0.0235
    yy -= 0.01
save(fig, "p7_m5_interactions.png")

# ============================================================================ combined PDF
from PIL import Image  # noqa: E402

imgs = [Image.open(FIG / f).convert("RGB") for f in PAGES]
pdf = RES / "tree_shap_figures.pdf"
imgs[0].save(pdf, save_all=True, append_images=imgs[1:], resolution=150.0)
print("wrote", pdf, len(imgs), "pages")
