"""Standalone PDF: sizing the 15:30 last-30-min straddle trade by the forecast gap (w_t = f(VRP)).

Reads, never retyping a number:
  results/atm_straddle_0dte_1530/vrp_sized_rules.csv   proportional / rank sizing (rv_iv notebook section 10)
  results/close_kelly/kelly_rules.csv                  the Kelly-type family (experiments/close_kelly_sizing.py)
  results/close_kelly/kelly_bins.csv                   the conditional payoff by gap bin (last refit)
  results/close_kelly/kelly_wealth.png, kelly_f_vs_s.png
Writes writeup/generated/vrp_sizing_*.tex (tables), results/close_kelly/vrp_sizing_*.png (figures),
writeup/vrp_sizing.tex and, via pdflatex, writeup/vrp_sizing.pdf (the PDF and aux are not committed).

Both sources use the deck's scorer (the 13-bar recalibration of the rv_iv notebook), on the same 614 days
after the 252-day warm-up (2021-08-11 .. 2024-04-30), midpoint and crossed fills; the proportional / rank
file also carries the 865-day block from the second day.  Nothing here is comparable with the master
table's research-scorer numbers, and the document says so.

Run:  python writeup/make_vrp_sizing_tex.py
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
VRP = ROOT / "results" / "atm_straddle_0dte_1530" / "vrp_sized_rules.csv"
KELLY = ROOT / "results" / "close_kelly"
GEN = ROOT / "writeup" / "generated"
WRITEUP = ROOT / "writeup"
STEM = "vrp_sizing"
PRIMARY_BLOCK = "from the second day"
PRIMARY_WINDOW, PRIMARY_CAP, PRIMARY_KELLY = "expanding", "history ruin bound", "half"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#8d5bd6",
    "#c9a227",
    "#d64f8a",
    "#3f8f9f",
    "#7a7a7a",
]


def clean(s: str) -> str:
    """Rule names carry one non-ASCII glyph (the multiplication sign) that the CSV mangled."""
    s = re.sub(r"[^\x20-\x7e]+", "x", str(s))
    return s.replace("(unit, x10)", "(unit, x10)")


def tex(s: str) -> str:
    s = clean(s)
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("|", r"$|$"),
    ):
        s = s.replace(a, b)
    return s


def f2(v: float, plus: bool = False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:+.2f}" if plus else f"{v:.2f}"


def ci(d: float, lo: float, hi: float) -> str:
    if np.isnan(d):
        return "--"
    return f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def longtable(
    colspec: str, header: str, rows: list[str], caption: str, label: str
) -> list[str]:
    return [
        r"\begingroup\scriptsize\setlength{\tabcolsep}{2.5pt}",
        r"\begin{longtable}{" + colspec + "}",
        r"\caption{" + caption + r"}\label{" + label + r"}\\",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{(continued)}\\",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
        *rows,
        r"\end{longtable}",
        r"\endgroup",
        "",
    ]


# ------------------------------------------------------------------ section 1: proportional / rank
def prop_tables(v: pd.DataFrame) -> tuple[list[str], dict]:
    rules = [r for r in v["rule"].unique() if r != "sign(s)"]
    forecasts = list(v["forecast"].unique())
    out: list[str] = []
    facts: dict = {}
    for block in (PRIMARY_BLOCK,):
        b = v[v["block"] == block]
        ndays = int(b["days"].iloc[0])
        n_above = {r: 0 for r in rules}
        n_below = {r: 0 for r in rules}
        for fill in ("mid", "crossed"):
            rows = []
            for fc in forecasts:
                g = b[b["forecast"] == fc].set_index("rule")
                base = g.loc["sign(s)"]
                cells = [tex(fc), f2(base[f"Sharpe {fill}"])]
                for r in rules:
                    x = g.loc[r]
                    cells += [
                        f2(x[f"Sharpe {fill}"]),
                        ci(x[f"dSharpe {fill}"], x[f"lo {fill}"], x[f"hi {fill}"]),
                    ]
                    if fill == "mid":
                        if x["lo mid"] > 0:
                            n_above[r] += 1
                        if x["hi mid"] < 0:
                            n_below[r] += 1
                rows.append(" & ".join(cells) + r" \\")
            header = (
                r"forecast & sign(s) & "
                + " & ".join(r"\multicolumn{2}{c}{" + tex(r) + "}" for r in rules)
                + r" \\"
                + "\n"
                + r" & Sharpe & "
                + " & ".join(r"Sharpe & $\Delta$ vs sign(s) [95\%]" for _ in rules)
            )
            cap = (
                f"Sizing by the size of the forecast gap, {ndays} days ({block}), {fill} fill; the deck's scorer. Every rule keeps the "
                r"sign of $s$ and changes only the size: (a) proportional, $q_t = s_t / \mathrm{rms}(s)$ with the rms from past days; "
                r"(b) the rank of $|s_t|$ among past days, sign kept; (c) the centred rank $2F(s_t)-1$. Returns per unit of premium; "
                r"$\Delta$ = the rule's Sharpe minus sign(s)'s on the same days, with the 95\% interval of a paired circular block "
                r"bootstrap (block 21, 2{,}000 draws)."
            )
            out += longtable(
                "l" + "r" + "rl" * len(rules),
                header,
                rows,
                cap,
                f"tab:vrp_{re.sub(r'\W+', '_', block)}_{fill}",
            )
        facts[block] = {
            "days": ndays,
            "n_forecasts": len(forecasts),
            "above": n_above,
            "below": n_below,
        }
    return out, facts


def prop_figure(v: pd.DataFrame) -> Path:
    b = v[(v["block"] == PRIMARY_BLOCK) & (v["rule"] != "sign(s)")]
    rules = list(b["rule"].unique())
    forecasts = list(b["forecast"].unique())
    fig, axes = plt.subplots(
        1, len(rules), figsize=(4.0 * len(rules), 5.2), dpi=180, sharey=True
    )
    for ax, r in zip(np.atleast_1d(axes), rules):
        g = b[b["rule"] == r].set_index("forecast").reindex(forecasts)
        y = np.arange(len(forecasts))
        ax.axvline(0, color=INK2, lw=0.8)
        ax.hlines(y, g["lo mid"], g["hi mid"], color=COLORS[0], lw=1.6)
        ax.plot(g["dSharpe mid"], y, "o", color=COLORS[0], ms=5, label="mid")
        ax.hlines(y + 0.28, g["lo crossed"], g["hi crossed"], color=COLORS[1], lw=1.2)
        ax.plot(
            g["dSharpe crossed"], y + 0.28, "s", color=COLORS[1], ms=4, label="crossed"
        )
        ax.set_title(clean(r), fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels([clean(x) for x in forecasts], fontsize=7)
        ax.grid(axis="x", color=GRID)
        ax.set_xlabel("Sharpe minus sign(s), 95% paired interval", fontsize=8)
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="lower left")
    ax.invert_yaxis()
    fig.tight_layout()
    p = KELLY / "vrp_sizing_prop_rank.png"
    fig.savefig(p)
    return p


# ------------------------------------------------------------------ section 2: Kelly family
def kelly_tables(k: pd.DataFrame) -> tuple[list[str], dict]:
    prim = k[(k["window"] == PRIMARY_WINDOW) & (k["cap"] == PRIMARY_CAP)]
    forecasts = list(prim["forecast"].unique())
    rules = [r for r in prim["rule"].unique() if r not in ("always short", "sign(s)")]
    out: list[str] = []
    facts: dict = {}
    for fill in ("mid", "crossed"):
        rows = []
        for fc in forecasts:
            g = prim[(prim["forecast"] == fc) & (prim["fill"] == fill)]
            base = g[(g["rule"] == "sign(s)")].iloc[0]
            rows.append(
                r"\multicolumn{7}{l}{\quad\emph{"
                + tex(fc)
                + r"}: sign(s), one unit of premium, Sharpe "
                + f2(base["Sharpe"])
                + r", growth at 3\% of wealth "
                + f2(base["g_ann"])
                + r"} \\"
            )
            for r in rules:
                for kf in ("full", "half", "quarter", "unit (growth at 3%)"):
                    x = g[(g["rule"] == r) & (g["kelly"] == kf)]
                    if not len(x):
                        continue
                    x = x.iloc[0]
                    rows.append(
                        " & ".join(
                            [
                                tex(r)
                                + (
                                    " (" + kf.split(" ")[0] + ")"
                                    if kf != "unit (growth at 3%)"
                                    else ""
                                ),
                                f2(x["Sharpe"]),
                                ci(
                                    x["dSharpe vs sign(s)"],
                                    x["lo vs sign(s)"],
                                    x["hi vs sign(s)"],
                                ),
                                f2(x["g_ann"]),
                                f2(x["maxDD_frac"]),
                                f2(x["mean |w|"]),
                                f"{100 * x['share opposite sign(s)']:.0f}",
                            ]
                        )
                        + r" \\"
                    )
        header = r"rule (Kelly fraction) & Sharpe & $\Delta$Sharpe vs sign(s) [95\%] & growth $g$ & max DD & mean $|w|$ & opposite side \%"
        ndays = int(prim["days"].iloc[0])
        cap = (
            f"Kelly-type sizing, {ndays} days after the 252-day warm-up, {fill} fill; the deck's scorer; expanding estimation window, "
            r"stake capped at the history ruin bound $1/|\min_{u<t} R'_u|$. Rules (every quantity at $t$ from days before $t$): "
            r"(a) Kelly $f=\hat\mu/\hat\sigma^2$ with the mean and variance of the trade's return conditional on $|s|$ terciles $\times$ sign, or on a "
            r"regression of the return on $(s, |s|)$ with or without a sign intercept; (b) Kelly by the sign of $s$ alone (log-optimal, and "
            r"mean--variance); (c) proportional to $s$ at the sign(s) Kelly stake; (d) always short at its Kelly stake; and sign(s) itself levered "
            r"at its Kelly stake. Fractions full / half / quarter. $g$ = annualized mean of $\log(1 + w_t R'_t)$; max DD = the largest fall of "
            r"wealth from its running peak; mean $|w|$ = the average stake as a share of wealth; opposite side = share of days the rule takes the "
            r"side opposite to sign(s). $\Delta$ = the rule's Sharpe minus sign(s)'s with the paired 95\% interval (block 21, 2{,}000 draws)."
        )
        out += longtable("lrlrrrr", header, rows, cap, f"tab:kelly_{fill}")
        sub = prim[
            (prim["fill"] == fill)
            & (~prim["rule"].isin(["always short", "sign(s)"]))
            & (prim["kelly"] != "unit (growth at 3%)")
        ]
        facts[fill] = {
            "rows": len(sub),
            "above": int((sub["lo vs sign(s)"] > 0).sum()),
            "below": int((sub["hi vs sign(s)"] < 0).sum()),
        }
    facts["days"] = int(prim["days"].iloc[0])
    facts["n_forecasts"] = len(forecasts)
    return out, facts


def kelly_figure(k: pd.DataFrame) -> Path:
    prim = k[
        (k["window"] == PRIMARY_WINDOW)
        & (k["cap"] == PRIMARY_CAP)
        & (k["fill"] == "mid")
        & (k["kelly"] == PRIMARY_KELLY)
    ]
    rules = [r for r in prim["rule"].unique() if r not in ("always short", "sign(s)")]
    forecasts = list(prim["forecast"].unique())
    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=180)
    y0 = np.arange(len(forecasts))
    off = np.linspace(-0.32, 0.32, len(rules))
    ax.axvline(0, color=INK2, lw=0.8)
    for j, r in enumerate(rules):
        g = prim[prim["rule"] == r].set_index("forecast").reindex(forecasts)
        ax.hlines(
            y0 + off[j],
            g["lo vs sign(s)"],
            g["hi vs sign(s)"],
            color=COLORS[j % len(COLORS)],
            lw=1.2,
            alpha=0.9,
        )
        ax.plot(
            g["dSharpe vs sign(s)"],
            y0 + off[j],
            "o",
            color=COLORS[j % len(COLORS)],
            ms=3.5,
            label=clean(r),
        )
    ax.set_yticks(y0)
    ax.set_yticklabels([clean(x) for x in forecasts], fontsize=7)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID)
    ax.set_xlabel(
        "half-Kelly rule's Sharpe minus sign(s), midpoint fill; 95% paired interval",
        fontsize=8,
    )
    ax.legend(fontsize=6.5, loc="lower left", ncol=2)
    fig.tight_layout()
    p = KELLY / "vrp_sizing_kelly_delta.png"
    fig.savefig(p)
    return p


def bins_table(path: Path) -> list[str]:
    if not path.exists():
        return []
    b = _drop_live_feasible(pd.read_csv(path))
    b = b[(b["fill"] == "mid") & b["cell"].str.startswith("|s| bin")]
    b["bin"] = b["cell"].str.extract(r"bin (\d)").astype(int)
    b["side_k"] = np.where(b["side"].str.startswith("s > 0"), "buy", "sell")
    rows = []
    n_buy_top_weakest = 0
    n_sell_mid_best = 0
    forecasts = list(b["forecast"].unique())
    for fc in forecasts:
        g = b[b["forecast"] == fc]
        buy = g[g["side_k"] == "buy"].set_index("bin")["mean r_long"].reindex([1, 2, 3])
        sell = (
            g[g["side_k"] == "sell"].set_index("bin")["mean r_short"].reindex([1, 2, 3])
        )
        nb = g[g["side_k"] == "buy"].set_index("bin")["n"].reindex([1, 2, 3])
        ns = g[g["side_k"] == "sell"].set_index("bin")["n"].reindex([1, 2, 3])
        if buy.idxmin() == 3:
            n_buy_top_weakest += 1
        if sell.idxmax() == 2:
            n_sell_mid_best += 1
        rows.append(
            " & ".join(
                [tex(fc)]
                + [f"{buy[i]:+.3f} ({int(nb[i])})" for i in (1, 2, 3)]
                + [f"{sell[i]:+.3f} ({int(ns[i])})" for i in (1, 2, 3)]
            )
            + r" \\"
        )
    head = (
        r"forecast & \multicolumn{3}{c}{buy days ($s>0$): mean return of the long straddle} & "
        r"\multicolumn{3}{c}{sell days ($s\le 0$): mean return of the short straddle} \\"
        + "\n"
        r" & $|s|$ tercile 1 & tercile 2 & tercile 3 & tercile 1 & tercile 2 & tercile 3"
    )
    cap = (
        r"The conditional payoff the Kelly rules estimate, at the last refit (2024-04-30, midpoint fill; results/close\_kelly/kelly\_bins.csv): "
        r"the mean return of the position sign(s) takes, by tercile of $|s|$ among the days on that side (day counts in parentheses). "
        f"On the buy side the largest-gap tercile is the weakest of the three for {n_buy_top_weakest} of {len(forecasts)} forecasts; on the sell "
        f"side the middle tercile pays the short most for {n_sell_mid_best} of {len(forecasts)}."
    )
    return longtable("l" + "r" * 6, head, rows, cap, "tab:kelly_bins")


PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[landscape,margin=0.5in]{geometry}
\usepackage{array}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{float}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\begin{document}
"""


def _drop_live_feasible(df: pd.DataFrame) -> pd.DataFrame:
    # live-feasible forecasts commented out of this document
    if "forecast" not in df.columns:
        return df
    keep = ~df["forecast"].astype(str).str.contains(
        r"live[-_]feasible", case=False, regex=True
    )
    return df.loc[keep].copy()


def main() -> None:
    v = _drop_live_feasible(pd.read_csv(VRP))
    k = _drop_live_feasible(pd.read_csv(KELLY / "kelly_rules.csv"))
    k["rule"] = k["rule"].map(clean)
    v["rule"] = v["rule"].map(clean)
    GEN.mkdir(parents=True, exist_ok=True)
    t1, f1 = prop_tables(v)
    t2, f2_ = kelly_tables(k)
    t3 = bins_table(KELLY / "kelly_bins.csv")
    fig1 = prop_figure(v)
    fig2 = kelly_figure(k)
    (GEN / "vrp_sizing_tables.tex").write_text(
        "\n".join(
            ["% AUTO-GENERATED by writeup/make_vrp_sizing_tex.py -- do not edit."]
            + t1
            + t2
            + t3
        ),
        encoding="utf-8",
        newline="\n",
    )
    pb = f1[PRIMARY_BLOCK]
    kd = f2_
    doc = [PREAMBLE]
    doc.append(
        r"\section*{Sizing the last-30-min straddle trade by the forecast gap: $w_t = f(\mathrm{VRP}_t)$}"
    )
    doc.append(
        r"\textbf{What is here.} The 15:30 rule trades the straddle (nearest out-of-the-money call and put, same-day expiry, one position) "
        r"on the sign of $s_t$ = the 16:00-bar variance forecast minus the implied variance at 15:30. This document records every rule that "
        r"lets the \emph{size} of the position depend on $s_t$ (or on the past distribution of the trade's return given $s_t$), against the "
        r"flat sign(s) rule on the same days. Two studies, one scorer: the rv\_iv deck's 13-bar recalibration. "
        r"The proportional and rank rules are scored on the "
        + str(pb["days"])
        + r" days from the second deck day (the first day has no prior gap to scale by). "
        r"Each size uses every earlier deck day, back to the first. The Kelly family is a separate sample, the "
        + str(kd["days"])
        + r" days after a 252-day warm-up. "
        + str(pb["n_forecasts"])
        + r" forecasts, midpoint and crossed fills, paired circular block-bootstrap intervals "
        r"(block 21, 2{,}000 draws). These Sharpes are the deck's and are not comparable with the master table's research-scorer numbers. "
        r"Every rule is causal: a stake at $t$ uses days before $t$ only (gated by multiplying every future return by 50 and checking that no "
        r"stake changes). Returns are per unit of premium; the Kelly rules' wealth paths compound.\par"
    )
    doc.append(
        r"\textbf{Read-offs, recorded from the tables.} Proportional and rank sizing ("
        + PRIMARY_BLOCK
        + "): "
        + "; ".join(
            f"{tex(r)}: {pb['above'][r]} of {pb['n_forecasts']} intervals above sign(s), {pb['below'][r]} below"
            for r in pb["above"]
        )
        + r". Kelly family (expanding window, history-ruin-bound cap, every rule $\times$ full/half/quarter): midpoint "
        + f"{kd['mid']['above']} of {kd['mid']['rows']} rows with an interval above sign(s), {kd['mid']['below']} below; crossed "
        + f"{kd['crossed']['above']} of {kd['crossed']['rows']} above, {kd['crossed']['below']} below."
        + r"\par"
    )
    doc.append(r"\clearpage")
    doc.append(r"\input{generated/vrp_sizing_tables.tex}")
    doc.append(r"\clearpage")
    for p, cap in (
        (
            fig1,
            "Proportional and rank sizing: each rule's Sharpe minus sign(s)'s, per forecast, midpoint and crossed fills, with the paired 95\\% interval ("
            + PRIMARY_BLOCK
            + ").",
        ),
        (
            fig2,
            "The Kelly family at half Kelly, midpoint fill: each rule's Sharpe minus sign(s)'s, per forecast, with the paired 95\\% interval.",
        ),
        (
            KELLY / "kelly_wealth.png",
            "Wealth paths (from experiments/close\\_kelly\\_sizing.py): sign(s) at a fixed 3\\% of wealth, sign(s) Kelly-levered and always short.",
        ),
        (
            KELLY / "kelly_f_vs_s.png",
            "The Kelly fraction against the forecast gap $s_t$ (from experiments/close\\_kelly\\_sizing.py).",
        ),
    ):
        rel = Path("..") / p.relative_to(ROOT)
        doc.append(r"\begin{figure}[H]\centering")
        doc.append(r"\includegraphics[width=0.9\linewidth]{" + rel.as_posix() + "}")
        doc.append(r"\caption{" + cap + "}")
        doc.append(r"\end{figure}")
    doc.append(r"\end{document}")
    texp = WRITEUP / f"{STEM}.tex"
    texp.write_text("\n".join(doc) + "\n", encoding="utf-8", newline="\n")
    print("wrote", texp)
    ok = True
    for _ in range(2):
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", texp.name],
            cwd=WRITEUP,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        ok = ok and r.returncode == 0
    if not ok:
        print(r.stdout[-3000:])
        raise SystemExit("pdflatex failed")
    print("wrote", WRITEUP / f"{STEM}.pdf")


if __name__ == "__main__":
    main()
