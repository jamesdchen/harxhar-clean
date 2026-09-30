"""Render the closing-strategy master table (results/close_master_table/*.csv) as LaTeX and a standalone PDF.

Reads the CSVs experiments/master_table_close.py wrote (never retypes a number) and writes

  writeup/generated/table_master_close.tex   two longtables (needs booktabs + longtable at the
                                             include site): forecast accuracy at the 16:00 bar,
                                             and the last-30-min sign(s) trade
  writeup/master_table_close.tex             standalone landscape document: the reading notes,
                                             the rank-correlation and vs-headline tables, the two
                                             longtables and the QLIKE-vs-Sharpe figure
  results/close_master_table/qlike_vs_sharpe.png
  writeup/master_table_close.pdf             (pdflatex, two passes; the PDF and aux are not committed)

Run:  python writeup/make_master_table_close_tex.py   (master_table_close.py calls it at the end)
"""

from __future__ import annotations

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
    "per-bar nonlinear (trees, LSTM)": (
        "#1baf7a",
        "^",
        (
            "per-bar tree",
            "LSTM",
        ),  # every tree rung (T10, T1, RS10, RS1, Optuna) and both LSTMs: by prefix
    ),
}
PREFIX_GROUPS = {
    "per-bar nonlinear (trees, LSTM)"
}  # families matched by prefix, not by name


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


def tex_label(label: str) -> str:
    """Bucket names in brackets as design names (typewriter), the rest escaped."""
    parts = re.split(r"(\[[^\]]+\])", str(label))
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


def ci(v, lo, hi, nd: int = 2) -> str:
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
    cols = [
        ("QLIKE raw", ""),
        ("QLIKE recal.", ""),
        (r"recal.\ vs RV", ""),
        (r"\% vs R", ""),
        (r"95\% interval, daily diff.", ""),
        ("DM", ""),
        (r"$\overline{RV}/\overline{F}$", ""),
        (r"extreme (all sess.)", ""),
    ]

    def cells(r: pd.Series) -> list[str]:
        iv = (
            ""
            if r["is_reference"]
            else "["
            + f(r["qlike_diff_ci_lo"], "{:+.4f}")
            + ", "
            + f(r["qlike_diff_ci_hi"], "{:+.4f}")
            + "]"
        )
        return [
            f(r["qlike_raw"], "{:.4f}"),
            f(r["qlike_recal"], "{:.4f}"),
            f(r["qlike_recal_unclipped"], "{:.4f}"),
            "" if r["is_reference"] else f(r["qlike_pct_vs_ref"], "{:+.1f}"),
            iv,
            "" if r["is_reference"] else f(r["dm_vs_ref"], "{:+.2f}"),
            f(r["calib_mean_realized_over_forecast"], "{:.3f}"),
            f(r["n_extreme_all_sessions"], "{:.0f}"),
        ]

    lag = int(t["dm_hac_lag"].dropna().iloc[0]) if t["dm_hac_lag"].notna().any() else 0
    cap = (
        r"Forecast accuracy at the 16:00 bar (15:30--16:00), research scorer. QLIKE raw: the plain back-transform $f^2B$; "
        r"QLIKE recal.: $(f^2+s)B$, $s$ the forecast's own mean squared adjusted-scale error at 16:00 over the previous 250 sessions, "
        r"lagged one session; both against the per-bar spec's 16:00 target; recal.\ vs RV: the same forecast against the unclipped realized "
        r"variance. \% vs R and the interval (circular day blocks of 21, 2{,}000 draws) on the daily loss difference against the reference R; "
        r"DM: Diebold--Mariano statistic, Newey--West variance (Bartlett, lag %d) with the Harvey--Leybourne--Newbold correction; "
        r"$\overline{RV}/\overline{F}$: mean target over mean recalibrated forecast; extreme: plain forecasts below PLAIN\_FLOOR on every 16:00 "
        r"session of the arm span (none falls on a trade day). R = reference, H = recommended headline, $=$ identical to another row."
    ).replace("lag %d", f"lag {lag}")
    return longtable(
        t, cols, cells, cap, "tab:master_close_accuracy", "l" + "r" * 4 + "c" + "r" * 3
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
        (r"$\Delta$Sharpe vs short", ""),
        (r"$\Delta$Sharpe$^{\times}$ vs short", ""),
    ]

    def cells(r: pd.Series) -> list[str]:
        short = r["key"] == "always_short"
        return [
            f(r["Sharpe_mid"], "{:.2f}"),
            f(r["Sharpe_crossed"], "{:.2f}"),
            f(r["mean_mid"], "{:+.3f}"),
            f(r["mean_crossed"], "{:+.3f}"),
            f(100 * r["hit_rate_mid"], "{:.1f}"),
            f(r["n_buy"], "{:.0f}"),
            ""
            if r["is_reference"]
            else ci(
                r["dSharpe_mid_vs_ref"],
                r["dSharpe_mid_vs_ref_lo"],
                r["dSharpe_mid_vs_ref_hi"],
            ),
            ""
            if r["is_reference"]
            else ci(
                r["dSharpe_crossed_vs_ref"],
                r["dSharpe_crossed_vs_ref_lo"],
                r["dSharpe_crossed_vs_ref_hi"],
            ),
            ""
            if short
            else ci(
                r["dSharpe_mid_vs_short"],
                r["dSharpe_mid_vs_short_lo"],
                r["dSharpe_mid_vs_short_hi"],
            ),
            ""
            if short
            else ci(
                r["dSharpe_crossed_vs_short"],
                r["dSharpe_crossed_vs_short_lo"],
                r["dSharpe_crossed_vs_short_hi"],
            ),
        ]

    cap = (
        r"The last-30-min trade: $\mathrm{sign}(s)$ on the straddle (nearest out-of-the-money call and put, same-day expiry, one position), "
        r"entered at 15:30, held to the close; buy when the recalibrated 16:00 forecast exceeds the 15:30 implied variance, sell otherwise. "
        r"Returns per unit of premium; Sharpe annualized by $\sqrt{252}$ per trade day; $^{\times}$: the entry pays the spread (buy at the ask, "
        r"sell at the bid); hit: share of days with a positive midpoint return; buys: days the rule buys. $\Delta$Sharpe: the row minus the "
        r"reference R (or minus always short) on the same days, with the 95\% percentile interval of a paired circular block bootstrap "
        r"(blocks of 21 days, 2{,}000 draws, one set of resampled days for every row)."
    )
    t = pd.concat(
        [tab[tab["key"] == "always_short"], tab[tab["key"] != "always_short"]]
    )
    return longtable(
        t, cols, cells, cap, "tab:master_close_trade", "l" + "r" * 6 + "c" * 4
    )


# --------------------------------------------------------------------------- figure
def figure(tab: pd.DataFrame) -> str:
    A = tab[
        (tab["table"] == "A")
        & (tab["key"] != "always_short")
        & (tab["family"] != CHECK)
        & (tab["duplicate_of"].fillna("") == "")
    ]
    ql = A["qlike_recal"]
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
                g["qlike_recal"],
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
            on["qlike_recal"].max(),
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
                [r["qlike_recal"]],
                [r["Sharpe_mid"]],
                s=150,
                facecolors="none",
                edgecolors=INK,
                linewidths=1.2,
                zorder=4,
            )
            ax.annotate(
                txt + r["label"],
                (r["qlike_recal"], r["Sharpe_mid"]),
                xytext=(10, -14 if flag == "is_reference" else 8),
                textcoords="offset points",
                fontsize=8,
                color=INK,
            )
    ax.set_xlabel(
        "QLIKE at the 16:00 bar, recalibrated (lower is better)", color=INK, fontsize=9
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
            f"{r['label']} (QLIKE {r['qlike_recal']:.3f}, Sharpe {r['Sharpe_mid']:.2f})"
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
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\begin{document}
\setlength{\LTcapwidth}{\textwidth}
"""


def md_to_tex(s: str) -> str:
    """The SUMMARY's inline markup (**bold**, `code`) to LaTeX, the rest escaped."""
    out = []
    for part in re.split(r"(`[^`]+`|\*\*[^*]+\*\*)", s):
        if part.startswith("`") and part.endswith("`"):
            out.append(r"\texttt{" + tex_escape(part[1:-1]) + "}")
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
    ):
        t = t.replace(a, b)
    return t


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
        n = len(table_rows[0])
        L.append(r"\begingroup\scriptsize\setlength{\tabcolsep}{2.5pt}")
        L.append(r"\begin{longtable}{" + "l" + "r" * (n - 1) + "}")
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


def main() -> None:
    tab = pd.read_csv(SRC / "master_table.csv", keep_default_na=True)
    tab["duplicate_of"] = tab["duplicate_of"].fillna("")
    for c in ("is_reference", "is_headline"):
        tab[c] = tab[c].astype(bool)
    body_acc = accuracy_table(tab)
    body_trade = trade_table(tab)
    GEN.mkdir(parents=True, exist_ok=True)
    gen = GEN / "table_master_close.tex"
    head = [
        "% AUTO-GENERATED by writeup/make_master_table_close_tex.py — do not edit.",
        "% Source: results/close_master_table/master_table.csv (experiments/master_table_close.py; research scorer).",
        "% Needs \\usepackage{booktabs,longtable,amsmath} at the include site.",
        "",
    ]
    gen.write_text(
        "\n".join(head + body_acc + body_trade), encoding="utf-8", newline="\n"
    )
    print("wrote", gen)

    off_note = figure(tab)
    doc = [PREAMBLE]
    doc.append(
        r"\section*{Master table for the closing strategy: every 16:00-bar forecast, one scorer, the same days}"
    )
    doc += summary_blocks()
    doc.append(r"\clearpage")
    doc.append(r"\input{generated/table_master_close.tex}")
    doc.append(r"\clearpage")
    doc.append(r"\begin{figure}[h]\centering")
    doc.append(
        r"\includegraphics[width=0.85\linewidth]{../results/close_master_table/qlike_vs_sharpe.png}"
    )
    doc.append(
        r"\caption{Recalibrated 16:00 QLIKE against the sign(s) Sharpe ratio (midpoint fill), table A (check rows and identical forecasts "
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
