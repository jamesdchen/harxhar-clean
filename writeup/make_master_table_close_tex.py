"""Render the closing-strategy master table (results/close_master_table/*.csv) as LaTeX and a standalone PDF.

Reads the CSVs experiments/master_table_close.py wrote (never retypes a number) and writes

  writeup/generated/table_master_close.tex   three longtables (needs booktabs + longtable at the
                                             include site): forecast accuracy at the 16:00 bar,
                                             the last-30-min sign(s) trade, and deck VRP sizing
  writeup/master_table_close.tex             standalone landscape document: the reading notes,
                                             the rank-correlation and vs-headline tables, the three
                                             longtables and the QLIKE-vs-Sharpe figure
  results/close_master_table/qlike_vs_sharpe.png
  writeup/master_table_close.pdf             (pdflatex, two passes; the PDF and aux are not committed)

2026-10-06 (user decisions): recalibration removed -- every number is the PLAIN forecast f^2 B
(QLIKE column qlike_raw, the trade q = sign(plain - iv_var)); intervals commented out -- every
table prints point estimates only (the bootstrap is disabled since b761b28, so an interval would
equal its point); the DM statistic and the HAC t are statistics, not intervals, and stay.

Run:  python writeup/make_master_table_close_tex.py [--per-bar-only]
      (master_table_close.py calls it at the end, both ways)
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "close_master_table"
GEN = ROOT / "writeup" / "generated"
WRITEUP = ROOT / "writeup"
FIG = SRC / "qlike_vs_sharpe.png"
STEM = "master_table_close"
# --per-bar-only (2026-09-30, user request): the same document restricted to the forecasts FITTED
# on the 16:00 bar -- the per-bar linear arms (incl. the VIX-only, implied-vol, HAR-ladder and
# chain-period bucket studies), every per-bar tree rung and both LSTMs -- leaving out the paper's
# pooled 48-bar forecasts, their pooled twins and the rest-of-day check rows (exact copies of
# their per-bar twins). Always short stays as the benchmark row. Separate outputs.
PER_BAR_EXCLUDED = ("paper", "pooled twin", "direct rest-of-day at 15:30 (check)")
PER_BAR_STEM = "master_table_close_16h"
PER_BAR_FIG = SRC / "qlike_vs_sharpe_16h.png"
CHECK = "direct rest-of-day at 15:30 (check)"

# figure groups (three categorical slots: the reference palette's first three, validated all-pairs)
GROUPS = {
    "48-bar forecasts (paper, pooled twins)": (
        "#2a78d6",
        "o",
        ("paper", "pooled twin"),
    ),
    "per-bar linear (incl. VIX-only, bucket studies)": (
        "#eb6834",
        "s",
        (
            "per-bar linear",
            "VIX-only family",
            "implied-vol representations",
            "HAR-ladder variants",
        ),
    ),
    # 2026-10-13 (user): LSTM parked for now (double-standardisation bug):
    # "per-bar nonlinear (trees, LSTM)": (
    #     "#1baf7a",
    #     "^",
    #     (
    #         "per-bar tree",
    #         "LSTM",
    #     ),  # every tree rung (T10, T1, RS10, RS1, Optuna) and both LSTMs: by prefix
    # ),
    "per-bar nonlinear (trees)": (
        "#1baf7a",
        "^",
        ("per-bar tree",),  # every tree rung (T10, T1, RS10, RS1, Optuna): by prefix
    ),
}
PREFIX_GROUPS = {
    # 2026-10-13 (user): LSTM parked -- was "per-bar nonlinear (trees, LSTM)"
    "per-bar nonlinear (trees)"
}  # families matched by prefix, not by name


def is_lstm_text(s: str) -> bool:
    """2026-10-13 (user): LSTM parked for now (double-standardisation bug) -- a row / line about an LSTM."""
    return "lstm" in str(s).lower()


def in_group(fam: pd.Series, name: str, fams: tuple[str, ...]) -> pd.Series:
    if name in PREFIX_GROUPS:
        return fam.str.startswith(fams)
    return fam.isin(fams)


INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def tex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


# 2026-10-06 (user): "remove adj_ in the names of any variable being presented" -- every design
# column name this maker prints drops its leading adj_ (adj_vix_ma_* -> vix_ma_*; har_ma_* unchanged);
# the CSVs, keys and code identifiers keep their names.
ADJ_PREFIX = re.compile(r"(?<![A-Za-z0-9_])adj_(?=[A-Za-z])")


def display_name(name: str) -> str:
    """A design column name as presented: the leading adj_ dropped."""
    return ADJ_PREFIX.sub("", str(name))


def tex_label(label: str) -> str:
    """Bucket names in brackets as design names (typewriter), the rest escaped."""
    parts = re.split(r"(\[[^\]]+\])", display_name(label))
    out = []
    for p in parts:
        if p.startswith("[") and p.endswith("]"):
            out.append(r"[\texttt{" + tex_escape(p[1:-1]) + "}]")
        else:
            out.append(tex_escape(p).replace("1e-4", r"$10^{-4}$"))
    return "".join(out)


def f(v, fmt: str) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    s = fmt.format(v)
    return s.replace("-", "$-$") if "$" not in s else s


def pt(v, nd: int = 2) -> str:
    """A signed point estimate (2026-10-06: intervals commented out, user)."""
    return f(v, "{:+.%df}" % nd)


def ci(v, lo, hi, nd: int = 2) -> str:
    """2026-10-06: no longer called (intervals commented out, user); kept for the parked lines."""
    if not all(np.isfinite([v, lo, hi])):
        return f(v, "{:+.%df}" % nd) if np.isfinite(v) else ""
    return (
        f(v, "{:+.%df}" % nd)
        + r"\,["
        + f(lo, "{:+.%df}" % nd)
        + ", "
        + f(hi, "{:+.%df}" % nd)
        + "]"
    )


def row_name(r: pd.Series, dup_label: dict[str, str]) -> str:
    name = tex_label(r["label"])
    if r["is_reference"]:
        name += r"$^{\mathrm{R}}$"
    if r["is_headline"]:
        name = r"\textbf{" + name + r"}$^{\mathrm{H}}$"
    if isinstance(r.get("duplicate_of"), str) and r["duplicate_of"]:
        name += r"$^{=}$"
    return name


def panels(tab: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    out = []
    for t in ("A", "B"):
        g = tab[tab["table"] == t]
        if not len(g):
            continue
        if t == "A":
            head = f"Table A: the same {int(g['n_days'].max())} trade days, {g['first_day'].min()} to {g['last_day'].max()}"
        else:
            spans = "; ".join(
                f"{n} trade days from {d}"
                for (d, n), _ in g.groupby(["first_day", "n_days"])
            )
            head = f"Table B: scored on the forecast's own days, paired with the reference on them ({spans})"
        out.append((head, g))
    return out


def longtable(
    tab: pd.DataFrame,
    cols: list[tuple[str, str]],
    cells,
    caption: str,
    label: str,
    colspec: str,
) -> list[str]:
    ncol = len(cols) + 1
    head = "Forecast & " + " & ".join(h for h, _ in cols) + r" \\"
    L = [
        r"\begingroup\scriptsize\setlength{\tabcolsep}{2.5pt}",
        r"\begin{longtable}{" + colspec + "}",
        r"\caption{" + caption + r"}\label{" + label + r"}\\",
        r"\toprule",
        head,
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{%d}{l}{\emph{(continued)}} \\" % ncol,
        r"\toprule",
        head,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    dup_label = dict(zip(tab["key"], tab["label"]))
    for title, g in panels(tab):
        L.append(r"\midrule")
        L.append(r"\multicolumn{%d}{l}{\textbf{%s}} \\" % (ncol, tex_escape(title)))
        for fam, gg in g.groupby("family", sort=False):
            L.append(
                r"\multicolumn{%d}{l}{\quad\emph{%s}} \\" % (ncol, tex_escape(fam))
            )
            for _, r in gg.iterrows():
                L.append(row_name(r, dup_label) + " & " + " & ".join(cells(r)) + r" \\")
    L += [r"\end{longtable}", r"\endgroup", ""]
    return L


def accuracy_table(tab: pd.DataFrame) -> list[str]:
    t = tab[tab["key"] != "always_short"]
    # 2026-10-06: recalibration removed, user decision -- the "QLIKE recal." and "recal. vs RV"
    # columns are gone; intervals commented out (user) -- the "95% interval, daily diff." column too:
    # ("QLIKE recal.", ""), (r"recal.\ vs RV", ""), (r"95\% interval, daily diff.", ""),
    cols = [
        ("QLIKE", ""),
        # (r"QLIKE vs RV", ""),  # 2026-10-15 (user): parked -- QLIKE itself is against the actual RV now (data audit #10)
        (r"QLIKE, trimmed", ""),
        (r"\% vs R", ""),
        ("DM", ""),
        (r"$\overline{RV}/\overline{F}$", ""),
        (r"extreme (all sess.)", ""),
    ]

    def cells(r: pd.Series) -> list[str]:
        # iv = "[" + f(r["qlike_diff_ci_lo"], "{:+.4f}") + ", " + f(r["qlike_diff_ci_hi"], "{:+.4f}") + "]"
        return [
            f(r["qlike_raw"], "{:.4f}"),
            # f(r["qlike_raw_unclipped"], "{:.4f}"),  # 2026-10-15 (user): parked -- now qlike_raw (data audit #10)
            f(r["qlike_raw_trimmed"], "{:.4f}"),
            "" if r["is_reference"] else f(r["qlike_pct_vs_ref"], "{:+.1f}"),
            "" if r["is_reference"] else f(r["dm_vs_ref"], "{:+.2f}"),
            f(r["calib_mean_realized_over_forecast"], "{:.3f}"),
            f(r["n_extreme_all_sessions"], "{:.0f}"),
        ]

    lag = int(t["dm_hac_lag"].dropna().iloc[0]) if t["dm_hac_lag"].notna().any() else 0
    cap = (
        r"Forecast accuracy at the 16:00 bar (15:30--16:00). Every forecast is the plain back-transform $f^2B$ ($f$ the forecast on "
        # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10):
        # r"the adjusted scale $\sqrt{RV/B}$, $B$ the diurnal baseline). QLIKE: against the per-bar spec's 16:00 target; QLIKE vs RV: the "
        # r"same forecast against the unclipped realized variance. \% vs R: the mean daily loss difference against the reference R; "
        r"the adjusted scale $\sqrt{RV/B}$, $B$ the diurnal baseline). QLIKE: against the actual (unwinsorized) realized variance; "
        r"QLIKE, trimmed: the same forecast against the winsorized training target (diagnostic). \% vs R: the mean daily loss "
        r"difference against the reference R; "
        r"DM: Diebold--Mariano statistic on that difference, Newey--West variance (Bartlett, lag %d) with the Harvey--Leybourne--Newbold "
        r"correction; $\overline{RV}/\overline{F}$: mean realized variance over mean forecast; extreme: plain forecasts below PLAIN\_FLOOR on every "
        r"16:00 session of the arm span. Point estimates. R = reference, H = recommended headline, $=$ identical to another row."
    ).replace("lag %d", f"lag {lag}")
    return longtable(
        t, cols, cells, cap, "tab:master_close_accuracy", "l" + "r" * len(cols)
    )


def trade_table(tab: pd.DataFrame) -> list[str]:
    cols = [
        ("Sharpe", ""),
        (r"Sharpe$^{\times}$", ""),
        ("mean", ""),
        (r"mean$^{\times}$", ""),
        (r"hit \%", ""),
        ("buys", ""),
        (r"$\Delta$Sharpe vs R", ""),
        (r"$\Delta$Sharpe$^{\times}$ vs R", ""),
        (r"$t$ vs R", ""),
        (r"$\Delta$Sharpe vs short", ""),
        (r"$\Delta$Sharpe$^{\times}$ vs short", ""),
    ]

    def cells(r: pd.Series) -> list[str]:
        short = r["key"] == "always_short"
        # 2026-10-06: intervals commented out (user) -- point estimates only; was
        # ci(r["dSharpe_mid_vs_ref"], r["dSharpe_mid_vs_ref_lo"], r["dSharpe_mid_vs_ref_hi"]) etc.
        return [
            f(r["Sharpe_mid"], "{:.2f}"),
            f(r["Sharpe_crossed"], "{:.2f}"),
            f(r["mean_mid"], "{:+.3f}"),
            f(r["mean_crossed"], "{:+.3f}"),
            f(100 * r["hit_rate_mid"], "{:.1f}"),
            f(r["n_buy"], "{:.0f}"),
            "" if r["is_reference"] else pt(r["dSharpe_mid_vs_ref"]),
            "" if r["is_reference"] else pt(r["dSharpe_crossed_vs_ref"]),
            "" if r["is_reference"] else pt(r["t_hac_mid_vs_ref"]),
            "" if short else pt(r["dSharpe_mid_vs_short"]),
            "" if short else pt(r["dSharpe_crossed_vs_short"]),
        ]

    cap = (
        r"The last-30-min trade: $\mathrm{sign}(s)$ on the straddle (nearest out-of-the-money call and put, same-day expiry, one position), "
        r"entered at 15:30, held to the close; buy when the plain 16:00 forecast $f^2B$ exceeds the 15:30 implied variance, sell otherwise. "
        r"Returns per unit of premium; Sharpe annualized by $\sqrt{252}$ per trade day; $^{\times}$: the entry pays the spread (buy at the ask, "
        r"sell at the bid); hit: share of days with a positive midpoint return; buys: days the rule buys. $\Delta$Sharpe: the row minus the "
        r"reference R (or minus always short) on the same days, point estimates; $t$ vs R: Newey--West HAC $t$ statistic of the mean daily "
        r"midpoint-return difference against R."
    )
    t = pd.concat(
        [tab[tab["key"] == "always_short"], tab[tab["key"] != "always_short"]]
    )
    return longtable(
        t,
        cols,
        cells,
        cap,
        "tab:master_close_trade",
        r">{\raggedright\arraybackslash}p{0.30\linewidth}" + "r" * len(cols),
    )


# --------------------------------------------------------------------------- figure
def figure(tab: pd.DataFrame) -> str:
    A = tab[
        (tab["table"] == "A")
        & (tab["key"] != "always_short")
        & (tab["family"] != CHECK)
        & (tab["duplicate_of"].fillna("") == "")
    ]
    ql = A[
        "qlike_raw"
    ]  # 2026-10-06: the plain forecast's QLIKE (recalibration removed, user decision)
    # the x-range covers the bulk; forecasts beyond the far edge are named in the caption, not dropped silently
    hi_edge = float(
        np.percentile(ql, 75) + 3.0 * (np.percentile(ql, 75) - np.percentile(ql, 25))
    )
    off = A[ql > hi_edge]
    on = A[ql <= hi_edge]
    fig, ax = plt.subplots(figsize=(8.6, 4.9), dpi=200)
    for name, (col, mk, fams) in GROUPS.items():
        g = on[in_group(on["family"], name, fams)]
        if len(g):
            ax.scatter(
                g["qlike_raw"],
                g["Sharpe_mid"],
                s=34,
                c=col,
                marker=mk,
                edgecolors="white",
                linewidths=1.0,
                label=f"{name} ({len(g)})",
                zorder=3,
            )
    short = tab.loc[tab["key"] == "always_short", "Sharpe_mid"]
    if len(short):
        ax.axhline(float(short.iloc[0]), color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.text(
            on["qlike_raw"].max(),
            float(short.iloc[0]) + 0.03,
            f"always short {float(short.iloc[0]):.2f}",
            color=INK2,
            fontsize=8,
            ha="right",
            va="bottom",
        )
    for flag, txt in (("is_reference", "R: "), ("is_headline", "H: ")):
        r = on[on[flag]]
        if len(r):
            r = r.iloc[0]
            ax.scatter(
                [r["qlike_raw"]],
                [r["Sharpe_mid"]],
                s=150,
                facecolors="none",
                edgecolors=INK,
                linewidths=1.2,
                zorder=4,
            )
            ax.annotate(
                txt + r["label"],
                (r["qlike_raw"], r["Sharpe_mid"]),
                xytext=(10, -14 if flag == "is_reference" else 8),
                textcoords="offset points",
                fontsize=8,
                color=INK,
            )
    ax.set_xlabel(
        "QLIKE at the 16:00 bar, plain forecast f²B (lower is better)",
        color=INK,
        fontsize=9,
    )
    ax.set_ylabel("sign(s) Sharpe, midpoint fill", color=INK, fontsize=9)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG)
    plt.close(fig)
    if len(off):
        names = "; ".join(
            f"{r['label']} (QLIKE {r['qlike_raw']:.3f}, Sharpe {r['Sharpe_mid']:.2f})"
            for _, r in off.iterrows()
        )
        return (
            f"{len(off)} forecasts lie beyond the plotted QLIKE range (upper quartile + 3 interquartile ranges = {hi_edge:.3f}) and are not drawn: "
            + names
            + "."
        )
    return ""


# --------------------------------------------------------------------------- document
PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[landscape,margin=0.5in]{geometry}
\usepackage{array}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\sloppy
\begin{document}
\setlength{\LTcapwidth}{\textwidth}
"""


def md_to_tex(s: str) -> str:
    """The SUMMARY's inline markup (**bold**, `code`) to LaTeX, the rest escaped."""
    out = []
    for part in re.split(r"(`[^`]+`|\*\*[^*]+\*\*)", s):
        if part.startswith("`") and part.endswith("`"):
            out.append(r"\texttt{" + tex_escape(display_name(part[1:-1])) + "}")
        elif part.startswith("**") and part.endswith("**"):
            out.append(r"\textbf{" + md_to_tex(part[2:-2]) + "}")
        else:
            out.append(tex_escape(part))
    t = "".join(out)
    for a, b in (
        ("²", r"\textsuperscript{2}"),
        ("≥", r"\ensuremath{\geq}"),
        ("≤", r"\ensuremath{\leq}"),
        ("–", "--"),
        ("—", "---"),
        ("·", r"\textperiodcentered{}"),
        ("Δ", r"\ensuremath{\Delta}"),
        ("…", r"\ldots{}"),
        ("×", r"\ensuremath{\times}"),
        ("−", r"\ensuremath{-}"),
        (
            "/",
            r"/\allowbreak{}",
        ),  # paths and a/b may break after the slash (narrow table cells)
    ):
        t = t.replace(a, b)
    return t


# SUMMARY tables: a column whose longest cell exceeds WRAP_CHARS characters becomes a ragged-right
# paragraph column when the table would not fit the landscape text width (layout estimates, in
# scriptsize characters; 2026-10-06: the tables were natural-width l/r columns and overflowed).
CHARS_PER_LINE = 200  # scriptsize characters across the landscape text width
WRAP_CHARS = 28  # longest cell above this: the column may wrap
MIN_WRAP_CHARS = 14  # the narrowest a wrapped column is made
CELL_PAD = 3  # characters of column separation per column


def md_colspec(rows: list[list[str]]) -> str:
    """Column spec of a SUMMARY markdown table: l + r natural columns, wide text columns wrapped."""
    n = max(len(r) for r in rows)
    w = [max((len(r[i]) for r in rows if i < len(r)), default=0) for i in range(n)]
    wide = [i for i in range(n) if w[i] > WRAP_CHARS]
    if not wide or sum(w) + CELL_PAD * n <= CHARS_PER_LINE:
        return "l" + "r" * (n - 1)
    narrow = sum(w[i] for i in range(n) if i not in wide) + CELL_PAD * n
    left = float(max(CHARS_PER_LINE - narrow, MIN_WRAP_CHARS * len(wide)))
    alloc: dict[int, float] = {}
    rest = {i: float(w[i]) for i in wide}
    while (
        rest
    ):  # water-filling: a column narrower than an equal share keeps its natural width
        share = left / len(rest)
        small = {i: v for i, v in rest.items() if v <= share}
        if not small:
            alloc.update({i: share for i in rest})
            break
        for i, v in small.items():
            alloc[i] = v
            left -= v
            del rest[i]
    spec = ""
    for i in range(n):
        if i in alloc:
            frac = alloc[i] / CHARS_PER_LINE
            spec += r">{\raggedright\arraybackslash}p{%.3f\linewidth}" % frac
        else:
            spec += "l" if i == 0 else "r"
    return spec


def summary_blocks() -> list[str]:
    """SUMMARY.md sections (a)-(c), converted (it is written from the same CSVs)."""
    md = (SRC / "SUMMARY.md").read_text(encoding="utf-8").splitlines()
    L: list[str] = []
    in_table = False
    table_rows: list[list[str]] = []

    def flush() -> None:
        nonlocal table_rows
        if not table_rows:
            return
        L.append(r"\begingroup\scriptsize\setlength{\tabcolsep}{2.5pt}")
        L.append(r"\begin{longtable}{" + md_colspec(table_rows) + "}")
        L.append(r"\toprule")
        L.append(" & ".join(md_to_tex(c) for c in table_rows[0]) + r" \\")
        L.append(r"\midrule")
        for rr in table_rows[1:]:
            L.append(" & ".join(md_to_tex(c) for c in rr) + r" \\")
        L.append(r"\bottomrule")
        L.append(r"\end{longtable}")
        L.append(r"\endgroup")
        table_rows = []

    for line in md[1:]:
        s = line.strip()
        if is_lstm_text(
            s
        ):  # 2026-10-13 (user): LSTM parked -- SUMMARY.md rows / lines about an LSTM dropped
            continue
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(re.fullmatch(r":?-+:?", c) for c in cells):
                continue
            table_rows.append(cells)
            in_table = True
            continue
        if in_table:
            flush()
            in_table = False
        if s.startswith("## "):
            L.append(r"\subsection*{" + md_to_tex(s[3:]) + "}")
        elif s.startswith("- "):
            L.append(r"$\bullet$ " + md_to_tex(s[2:]) + r"\par")
        elif s:
            L.append(md_to_tex(s) + r"\par")
    flush()
    return L


_VRP_TAG = {
    "a0": "a0",
    "blk2": "blk2",
    "blk2_inc": "blk2_inc",
    "lgbm": "lgbm",
    "xgb": "xgb",
    "lasso_t": "lasso_t",
    "lasso_f": "lasso_f",
    "enet": "enet",
    "sub_ridge_baseline": "sub_base",
    "sub_ridge_all_features": "sub_ridge",
    "sub_lasso_all_features": "sub_lasso",
    "sub_enet_all_features": "sub_enet",
}


def vrp_sizing_table(tab: pd.DataFrame) -> list[str]:
    """Deck sizing on the 865 days from the second day, one row per forecast that has it."""
    src = ROOT / "results" / "atm_straddle_0dte_1530" / "vrp_sized_rules.csv"
    if not src.is_file():
        return []
    v = pd.read_csv(src)
    v = v[v["block"] == "from the second day"]
    have = set(tab["key"])
    if not any(k in have for k in _VRP_TAG):
        return []
    days = int(v["days"].iloc[0])
    # 2026-10-06 (user decision, the rv_iv notebook's re-run on the plain forecast): the robust
    # clip replaces the proportional size and the centred rank is removed -- parked:
    # ("(a) proportional s/rms", "proportional"), ("(c) centred rank $2F(s)-1$", "centred")
    rules = [
        ("(d) robust clip s/med|s|", "robust clip"),
        ("(b) rank of |s|, sign kept", "rank"),
    ]
    assert set(v["rule"]) == {"sign(s)", *(r for r, _ in rules)}, sorted(set(v["rule"]))
    head = (
        r"forecast & sign(s) & "
        + " & ".join(rf"\multicolumn{{2}}{{c}}{{{name}}}" for _rule, name in rules)
        + r" \\"
        + "\n"
        + r" & Sharpe & "
        + " & ".join(r"Sharpe & $\Delta$ vs sign(s)" for _ in rules)
        + r" \\"
    )
    shown = tab[tab["key"].isin(_VRP_TAG) & (tab["table"] == "A")]
    if not len(shown):
        shown = tab[tab["key"].isin(_VRP_TAG)]
    rows = []
    for key in [k for k in _VRP_TAG if k in set(shown["key"])]:
        r = shown[shown["key"] == key].iloc[0]
        g = v[v["tag"] == _VRP_TAG[key]].set_index("rule")
        cells = [row_name(r, {}), f(g.loc["sign(s)", "Sharpe mid"], "{:.2f}")]
        for rule, _name in rules:
            x = g.loc[rule]
            cells.append(f(x["Sharpe mid"], "{:.2f}"))
            # 2026-10-06: intervals commented out (user) -- was ci(x["dSharpe mid"], x["lo mid"], x["hi mid"])
            cells.append(pt(x["dSharpe mid"]))
        rows.append(" & ".join(cells) + r" \\")
    cap = (
        f"VRP sizing on the deck's scorer, {days} days from the second deck day, midpoint fill. "
        r"The size on day $t$ uses every earlier deck day back to the first: "
        r"robust clip $q_t = \mathrm{clip}(s_t/\tilde m_t, -1, 1)$ with $\tilde m_t$ the median "
        r"of $|s|$ on the earlier days, and the rank of $|s_t|$, sign kept. "
        r"$\Delta$ is the rule's Sharpe minus sign(s) on the same days (point estimate). "
        r"These Sharpes are the deck's own scorer, not the plain-forecast Sharpes of the trade table."
    )
    return _vrp_longtable(
        head, rows, cap, "lr" + "rl" * len(rules), ncol=2 + 2 * len(rules)
    )


def _vrp_longtable(
    head: str, rows: list[str], caption: str, colspec: str, ncol: int = 8
) -> list[str]:
    return [
        r"\begingroup\scriptsize\setlength{\tabcolsep}{2.5pt}",
        r"\begin{longtable}{" + colspec + "}",
        r"\caption{" + caption + r"}\label{tab:master_vrp_sizing}\\",
        r"\toprule",
        head,
        r"\midrule",
        r"\endfirsthead",
        rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{(continued)}}}} \\",
        r"\toprule",
        head,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
        *rows,
        r"\end{longtable}",
        r"\endgroup",
        "",
    ]


def main() -> None:
    global STEM, FIG
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--per-bar-only",
        action="store_true",
        help="only the forecasts fitted on the 16:00 bar (no pooled 48-bar forecasts, no check rows)",
    )
    a = ap.parse_args()
    tab = pd.read_csv(SRC / "master_table.csv", keep_default_na=True)
    tab["duplicate_of"] = tab["duplicate_of"].fillna("")
    live = tab["key"].astype(str).str.contains("live_feasible") | tab["label"].astype(
        str
    ).str.contains(r"live[-_]feasible", case=False, regex=True)
    tab = tab.loc[~live].copy()
    # 2026-10-13 (user): LSTM parked for now (double-standardisation bug): no LSTM row in either document
    lstm = tab["key"].astype(str).str.startswith("lstm") | tab["family"].astype(
        str
    ).str.contains("LSTM")
    tab = tab.loc[~lstm].copy()
    tab["label"] = tab["label"].astype(str).str.replace(", same spec", "", regex=False)
    gen_name = "table_master_close.tex"
    n_all = len(tab)
    if a.per_bar_only:
        STEM, FIG = PER_BAR_STEM, PER_BAR_FIG
        gen_name = "table_master_close_16h.tex"
        tab = tab[~tab["family"].isin(PER_BAR_EXCLUDED)].reset_index(drop=True)
        # a duplicate whose original is excluded is its own row again
        tab.loc[~tab["duplicate_of"].isin(tab["key"]), "duplicate_of"] = ""
    for c in ("is_reference", "is_headline"):
        tab[c] = tab[c].astype(bool)
    body_acc = accuracy_table(tab)
    body_trade = trade_table(tab)
    body_vrp = vrp_sizing_table(tab)
    GEN.mkdir(parents=True, exist_ok=True)
    gen = GEN / gen_name
    head = [
        "% AUTO-GENERATED by writeup/make_master_table_close_tex.py — do not edit.",
        "% Source: results/close_master_table/master_table.csv (experiments/master_table_close.py; plain forecast f^2 B, point estimates).",
        "% Needs \\usepackage{booktabs,longtable,amsmath} at the include site.",
        "",
    ]
    gen.write_text(
        "\n".join(head + body_acc + body_trade + body_vrp),
        encoding="utf-8",
        newline="\n",
    )
    print("wrote", gen)

    off_note = figure(tab)
    doc = [PREAMBLE]
    if a.per_bar_only:
        doc.append(
            r"\section*{Master table for the closing strategy: the forecasts fitted on the 16:00 bar, one scorer, the same days}"
        )
        n_excl = n_all - len(tab)
        doc.append(r"\subsection*{What this version contains}")
        doc.append(
            f"{len(tab) - int((tab['key'] == 'always_short').sum())} forecasts, every one fitted on the 16:00 bar alone: the per-bar linear arms "
            "(ridge, lasso, elastic net on the HAR + calendar and the all-features inputs; the VIX-only, implied-volatility, "
            "HAR-ladder and chain-period bucket studies), every per-bar tree rung (the shipped configuration refit every 10 sessions and "
            "every session; the random search at both cadences; the Optuna-tuned paths by tuning cadence and trial budget; MSE- and "
            # 2026-10-13 (user): LSTM parked for now (double-standardisation bug):
            # "QLIKE-selected twins) and both LSTMs (the per-bar sequence of sessions, refit every session, and the intraday sequence of bars). "
            "QLIKE-selected twins). "
            f"Left out ({n_excl} rows of the full table): the paper's pooled 48-bar forecasts, their pooled twins and the rest-of-day check "
            "rows (exact copies of their per-bar twins). Always short is kept as the benchmark row. Every number is the full table's "
            r"(experiments/master\_table\_close.py; every forecast is the plain back-transform $f^2B$, no recalibration; 866 trade days; "
            "point estimates, with the Diebold--Mariano statistic and the HAC $t$ where shown); the columns against the reference still compare with the paper's block-diagonal ridge, "
            "which is not itself listed here. The per-bar design has no session-edge duplicates, every fit keeps only the columns "
            # 2026-10-13 (user): LSTM parked -- was "every tree and LSTM chunk"
            "identifiable on its own training window, and every tree chunk was computed on one processor class."
            + r"\par"
        )
    else:
        doc.append(
            r"\section*{Master table for the closing strategy: every 16:00-bar forecast, one scorer, the same days}"
        )
        doc += summary_blocks()
    doc.append(r"\clearpage")
    doc.append(r"\input{generated/" + gen_name + "}")
    doc.append(r"\clearpage")
    doc.append(r"\begin{figure}[h]\centering")
    doc.append(
        r"\includegraphics[width=0.85\linewidth]{../results/close_master_table/"
        + FIG.name
        + "}"
    )
    doc.append(
        r"\caption{16:00 QLIKE of the plain forecast $f^2B$ against the sign(s) Sharpe ratio (midpoint fill), table A (check rows and identical forecasts "
        r"left out); the dashed line is always short. R = reference, H = recommended headline. "
        + tex_escape(off_note)
        + "}"
    )
    doc.append(r"\end{figure}")
    doc.append(r"\end{document}")
    tex = WRITEUP / f"{STEM}.tex"
    tex.write_text("\n".join(doc) + "\n", encoding="utf-8", newline="\n")
    print("wrote", tex)
    ok = True
    for _ in range(2):  # longtable needs a second pass for its column widths
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
            cwd=WRITEUP,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        ok = ok and r.returncode == 0
    if ok:
        print("wrote", WRITEUP / f"{STEM}.pdf")
    else:
        print(r.stdout[-3000:])
        raise SystemExit("pdflatex failed")


if __name__ == "__main__":
    main()
