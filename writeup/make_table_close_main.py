"""The paper's close-option results table and the numbers its prose quotes (D1, 2026-09-29).

Reads the closing-strategy master table (experiments/master_table_close.py; ONE scorer, the
16:00-bar recalibration (f^2 + e) B of the research scorers) and writes

  writeup/generated/table_close_main.tex     Table 7 of the paper: selected master-table rows
  writeup/generated/close_main_numbers.tex   \\cm... macros for every number in Section 5.4's prose
  writeup/generated/table_close_ladder.tex   Table 8: one change at a time (paired steps)
  writeup/generated/close_main_ladder.csv    the executed pairs behind Table 8

Nothing is typed by hand: every cell and every macro is read from the CSVs named below, and
every qualitative claim the prose makes about them is asserted here, so a re-run after the
master table changes (e.g. when the LSTM rows land) fails loudly if a claim flips.

Sources
  results/close_master_table/master_table.csv          one row per forecast (table A: 866 days)
  results/spxw_pnl/MANIFEST.md                         design panel of each of the paper's tables
  results/close_pnl_decomp/research_scorer/*.csv       the P&L decomposition under the same scorer
  results/close_master_table/master_table_daily.parquet  per-day series (paired steps)
  results/linear_subsection_trees_tuned/a2b/*.csv      causally tuned per-bar trees (A2b)

Usage:  python writeup/make_table_close_main.py
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "results" / "close_master_table" / "master_table.csv"
VS_HEADLINE = ROOT / "results" / "close_master_table" / "master_table_vs_headline.csv"
DAILY = ROOT / "results" / "close_master_table" / "master_table_daily.parquet"
MANIFEST = ROOT / "results" / "spxw_pnl" / "MANIFEST.md"
# the P&L decomposition re-run under the same 16:00-bar recalibration (checklist B1)
PNL_RS = ROOT / "results" / "close_pnl_decomp" / "research_scorer"
CSV_TOL = 5e-6  # the P&L decomposition writes six significant digits; agreement to that precision
GEN = ROOT / "writeup" / "generated"
TABLE_OUT = GEN / "table_close_main.tex"
NUMBERS_OUT = GEN / "close_main_numbers.tex"
LADDER_OUT = GEN / "table_close_ladder.tex"
A2B = ROOT / "results" / "linear_subsection_trees_tuned" / "a2b"  # tuned trees (A2b)
LADDER_CSV = GEN / "close_main_ladder.csv"  # the executed pairs behind LADDER_OUT
GATE_TOL = 1e-9  # a recomputed pair must equal the master table's stored pair

REFERENCE = "blk2"  # the paper's forecast of record (master table's fixed reference)
HEADLINE = (
    "sub_ridge_live_feasible"  # the section's headline forecast (master table's H)
)
SHORT = "always_short"
N_DAYS = 866  # the trade frame every row of table A shares (asserted, not assumed)
# the paper's columns fitted on the earlier design panel (dagger); checked against MANIFEST.md
DAGGER = ("lgbm", "xgb", "lasso_t", "enet")
ON_RECORD_PANEL = ("blk2", "lasso_f")

# (panel heading, [(key, label, inputs)]) in display order; inputs = design name of the input set
PAPER = [
    ("a0", "baseline (HAR + calendar OLS)", ""),
    ("blk2", "block-diagonal ridge", ""),
    ("blk2_inc", "block-diagonal ridge, without the FOMC columns", ""),
    ("lgbm", "LightGBM", ""),
    ("xgb", "XGBoost", ""),
    ("lasso_t", "lasso (causally tuned)", ""),
    ("lasso_f", r"lasso ($\alpha=10^{-4}$)", ""),
    ("enet", "elastic net (causally tuned)", ""),
]
POOLED = [
    ("pool_ridge_baseline", "ridge", "baseline"),
    ("pool_ridge_all_features", "ridge", "all_features"),
    ("pool_ridge_live_feasible", "ridge", "live_feasible"),
]
PERBAR_LINEAR = [
    ("sub_ols_baseline", "OLS", "baseline"),
    ("sub_ridge_baseline", "ridge", "baseline"),
    ("sub_lasso_baseline", "lasso", "baseline"),
    ("sub_enet_baseline", "elastic net", "baseline"),
    ("sub_ridge_all_features", "ridge", "all_features"),
    ("sub_lasso_all_features", "lasso", "all_features"),
    ("sub_enet_all_features", "elastic net", "all_features"),
    ("sub_ridge_live_feasible", "ridge", "live_feasible"),
    ("sub_lasso_live_feasible", "lasso", "live_feasible"),
    ("sub_enet_live_feasible", "elastic net", "live_feasible"),
]
PERBAR_TREES = [
    ("subtree_lgbm_live_feasible", "LightGBM", "live_feasible"),
    ("subtree_xgb_live_feasible", "XGBoost", "live_feasible"),
    ("subtree_rf_live_feasible", "random forest", "live_feasible"),
    ("subtree_lgbm_all_features", "LightGBM", "all_features"),
    ("subtree_xgb_all_features", "XGBoost", "all_features"),
    ("subtree_rf_all_features", "random forest", "all_features"),
]
PANELS = [
    ("A. The paper's forecasts: one regression for all 48 bars of the day", PAPER),
    (
        "B. Pooled twins: the per-bar specification fitted on all 48 bars",
        POOLED,
    ),
    ("C. Per-bar linear: one regression for the 16:00 bar alone", PERBAR_LINEAR),
    ("D. Per-bar trees, untuned: the same rows, inputs and window", PERBAR_TREES),
]
CHECK_FAMILY = "direct rest-of-day at 15:30 (check)"  # the master table's check rows


# --------------------------------------------------------------------------- formatting
def _neg(s: str) -> str:
    return s.replace("-", r"\ensuremath{-}")


def f_(x: float, nd: int = 2) -> str:
    """Plain number, text-safe minus."""
    return _neg(f"{x:.{nd}f}")


def s_(x: float, nd: int = 2) -> str:
    """Signed number, text-safe minus and explicit plus."""
    return _neg(f"{x:+.{nd}f}")


def ci_(v: float, lo: float, hi: float, nd: int = 2) -> str:
    return f"{s_(v, nd)} [{s_(lo, nd)}, {s_(hi, nd)}]"


def tt(name: str) -> str:
    return r"\texttt{" + name.replace("_", r"\_") + "}" if name else ""


# --------------------------------------------------------------------------- inputs
def load() -> pd.DataFrame:
    d = pd.read_csv(MASTER).set_index("key")
    assert d.index.is_unique, "master table keys must be unique"
    return d


def check_manifest() -> None:
    """The dagger set is the MANIFEST's incumbent-panel tables (a0 is panel-invariant)."""
    text = MANIFEST.read_text(encoding="utf-8")
    panel = {}
    for m in re.finditer(r"^\| `(\w+)` \|.*\| `\w+` \| ([^|]+) \|", text, flags=re.M):
        panel[m.group(1)] = m.group(2).strip()
    for k in DAGGER:
        assert panel.get(k, "").startswith("incumbent"), (k, panel.get(k))
    for k in ON_RECORD_PANEL:
        assert panel.get(k, "").startswith("FOMC"), (k, panel.get(k))


# --------------------------------------------------------------------------- the table
def row_cells(d: pd.DataFrame, key: str, label: str, inputs: str) -> str:
    r = d.loc[key]
    mark = ""
    if key == REFERENCE:
        mark = r"$^{\mathrm{R}}$"
    if key in DAGGER:
        mark = r"$^{\dagger}$"
    if key == HEADLINE:
        mark = r"$^{\mathrm{H}}$"
        label = r"\textbf{" + label + "}"
    is_ref = key == REFERENCE
    cells = [
        label + mark,
        tt(inputs),
        f"{r['qlike_recal']:.4f}",
        "" if is_ref else s_(r["qlike_pct_vs_ref"], 1),
        "" if is_ref else s_(r["dm_vs_ref"], 2),
        f_(r["Sharpe_mid"]),
        f_(r["Sharpe_crossed"]),
        ""
        if is_ref
        else ci_(
            r["dSharpe_mid_vs_ref"],
            r["dSharpe_mid_vs_ref_lo"],
            r["dSharpe_mid_vs_ref_hi"],
        ),
        ci_(
            r["dSharpe_mid_vs_short"],
            r["dSharpe_mid_vs_short_lo"],
            r["dSharpe_mid_vs_short_hi"],
        ),
        f"{r['pct_buy']:.1f}",
    ]
    return " & ".join(cells) + r" \\"


def short_row(d: pd.DataFrame) -> str:
    r = d.loc[SHORT]
    cells = [
        "always short (no forecast)",
        "",
        "",
        "",
        "",
        f_(r["Sharpe_mid"]),
        f_(r["Sharpe_crossed"]),
        ci_(
            r["dSharpe_mid_vs_ref"],
            r["dSharpe_mid_vs_ref_lo"],
            r["dSharpe_mid_vs_ref_hi"],
        ),
        "",
        f"{r['pct_buy']:.1f}",
    ]
    return " & ".join(cells) + r" \\"


def write_table(d: pd.DataFrame) -> None:
    ncol = 10
    lines = [
        "% AUTO-GENERATED by writeup/make_table_close_main.py -- do not edit.",
        "% Source: results/close_master_table/master_table.csv (experiments/master_table_close.py;",
        "% the 16:00-bar recalibration). Selected rows of table A, all on the same 866 trade days.",
        r"\begingroup\small\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{llrrrrrccr}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{vs R} & \multicolumn{2}{c}{Sharpe$_{\mathrm{ann}}$}"
        r" & \multicolumn{2}{c}{$\Delta$Sharpe, mid [95\%]} & \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"forecast & inputs & QLIKE & \% & DM & mid & crossed & vs R & vs always short"
        r" & buys, \% \\",
        r"\midrule",
        short_row(d),
    ]
    for head, rows in PANELS:
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{{head}}}}} \\")
        for key, label, inputs in rows:
            lines.append(row_cells(d, key, label, inputs))
    lines += [r"\bottomrule", r"\end{tabular}", r"\endgroup", ""]
    TABLE_OUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- the prose numbers
def macros(d: pd.DataFrame) -> dict[str, str]:
    m: dict[str, str] = {}
    keys = [k for _, rows in PANELS for k, _, _ in rows] + [SHORT]
    assert (d.loc[keys, "n_days"] == N_DAYS).all(), (
        "every tabled row must be on the 866 days"
    )
    assert bool(d.loc[REFERENCE, "is_reference"]), (
        "blk2 must be the master table's reference"
    )
    m["cmNdays"] = str(N_DAYS)
    m["cmNrows"] = str(len(keys) - 1)  # forecasts in the table (always short excluded)

    s = d.loc[SHORT]
    m["cmShortMean"] = f_(s["mean_mid"], 3)
    m["cmShortShMid"] = f_(s["Sharpe_mid"])
    m["cmShortShX"] = f_(s["Sharpe_crossed"])
    # always short reads no forecast: its own interval is the P&L decomposition's (same 866
    # days, same bootstrap constants), gated against the master table's point value
    hp = pd.read_csv(PNL_RS / "hit_payoff.csv")
    for fill, col in (("mid", "Sharpe_mid"), ("crossed", "Sharpe_crossed")):
        h = hp[
            (hp["series"] == "always short")
            & (hp["fill"] == fill)
            & (hp["cell"] == "all days")
        ]
        assert len(h) == 1 and abs(h["Sharpe_ann"].iloc[0] - s[col]) < CSV_TOL, (
            fill,
            h,
        )
        m["cmShortShCi" + ("Mid" if fill == "mid" else "X")] = (
            f"[{f_(h['Sharpe_ann_lo'].iloc[0])}, {f_(h['Sharpe_ann_hi'].iloc[0])}]"
        )
    assert "\\ensuremath{-}" in m["cmShortShCiMid"], (
        "claim: always short's interval includes zero"
    )

    paper = d.loc[[k for k, _, _ in PAPER]]
    m["cmPaperShMin"] = f_(paper["Sharpe_mid"].min())
    m["cmPaperShMax"] = f_(paper["Sharpe_mid"].max())
    m["cmPaperXMin"] = f_(paper["Sharpe_crossed"].min())
    m["cmPaperXMax"] = f_(paper["Sharpe_crossed"].max())
    m["cmPaperVsShortMin"] = s_(paper["dSharpe_mid_vs_short"].min())
    m["cmPaperVsShortMax"] = s_(paper["dSharpe_mid_vs_short"].max())
    above = paper[paper["dSharpe_mid_vs_short_lo"] > 0]
    m["cmPaperVsShortAboveN"] = {0: "none", 1: "one", 2: "two", 3: "three"}.get(
        len(above), str(len(above))
    )
    labels = dict((k, lab) for k, lab, _ in PAPER)
    m["cmPaperVsShortAboveWho"] = ", ".join(labels[k] for k in above.index) or "none"
    if len(above) == 1:
        a = above.iloc[0]
        m["cmPaperVsShortAboveCi"] = ci_(
            a["dSharpe_mid_vs_short"],
            a["dSharpe_mid_vs_short_lo"],
            a["dSharpe_mid_vs_short_hi"],
        )
    others = paper.drop(index=REFERENCE)
    m["cmPaperSameMin"] = f"{100 * others['same_position_as_ref'].min():.0f}"
    m["cmPaperSameMax"] = f"{100 * others['same_position_as_ref'].max():.0f}"

    r = d.loc[REFERENCE]
    m["cmRefQ"] = f"{r['qlike_recal']:.4f}"
    m["cmRefShMid"] = f_(r["Sharpe_mid"])
    m["cmRefShX"] = f_(r["Sharpe_crossed"])
    m["cmRefBuy"] = f"{r['pct_buy']:.0f}"
    m["cmRefVsShort"] = ci_(
        r["dSharpe_mid_vs_short"],
        r["dSharpe_mid_vs_short_lo"],
        r["dSharpe_mid_vs_short_hi"],
    )
    assert r["dSharpe_mid_vs_short_lo"] < 0 < r["dSharpe_mid_vs_short_hi"], (
        "claim: includes zero"
    )

    f = d.loc["blk2_inc"]
    m["cmFomcSh"] = f_(f["Sharpe_mid"])
    m["cmFomcD"] = ci_(
        f["dSharpe_mid_vs_ref"], f["dSharpe_mid_vs_ref_lo"], f["dSharpe_mid_vs_ref_hi"]
    )
    assert f["dSharpe_mid_vs_ref_lo"] < 0 < f["dSharpe_mid_vs_ref_hi"], (
        "claim: FOMC gap unresolved"
    )

    best = paper["Sharpe_mid"].idxmax()
    assert best == "lasso_f", (
        "prose names the fixed lasso as the paper's highest Sharpe"
    )
    b = d.loc[best]
    m["cmLassoFShMid"] = f_(b["Sharpe_mid"])
    m["cmLassoFVsRef"] = ci_(
        b["dSharpe_mid_vs_ref"], b["dSharpe_mid_vs_ref_lo"], b["dSharpe_mid_vs_ref_hi"]
    )
    assert b["dSharpe_mid_vs_ref_lo"] > 0, (
        "claim: the fixed lasso's interval vs R lies above zero"
    )
    m["cmLassoFSame"] = f"{100 * b['same_position_as_ref']:.0f}"

    for tag, rows in (("Pool", POOLED), ("Sub", PERBAR_LINEAR), ("Tree", PERBAR_TREES)):
        x = d.loc[[k for k, _, _ in rows], "Sharpe_mid"]
        m[f"cm{tag}ShMin"] = f_(x.min())
        m[f"cm{tag}ShMax"] = f_(x.max())

    # every tabled forecast against R: how many intervals lie wholly above / below zero
    tab = d.loc[[k for k in keys if k not in (REFERENCE, SHORT)]]
    m["cmTabVsRefN"] = str(len(tab))
    m["cmTabVsRefAbove"] = str(int((tab["dSharpe_mid_vs_ref_lo"] > 0).sum()))
    m["cmTabVsRefBelow"] = str(int((tab["dSharpe_mid_vs_ref_hi"] < 0).sum()))
    # the same count over every forecast of the master table's table A (check rows excluded)
    a_all = d[(d["table"] == "A") & (d["family"] != CHECK_FAMILY) & (d.index != SHORT)]
    a_oth = a_all.drop(index=REFERENCE)
    m["cmMasterN"] = str(len(a_all))
    m["cmMasterVsRefN"] = str(len(a_oth))
    m["cmMasterVsRefAbove"] = str(int((a_oth["dSharpe_mid_vs_ref_lo"] > 0).sum()))

    m.update(headline_macros(d, a_all))

    # the crossed spread: how much each tabled forecast gives up
    fc = d.loc[[k for k in keys if k != SHORT]]
    loss = fc["Sharpe_mid"] - fc["Sharpe_crossed"]
    m["cmXLossMin"] = f_(loss.min())
    m["cmXLossMax"] = f_(loss.max())
    return m


def headline_macros(d: pd.DataFrame, a_all: pd.DataFrame) -> dict[str, str]:
    """The headline forecast (change 4 of D1): its numbers and its nearest alternatives."""
    m: dict[str, str] = {}
    assert bool(d.loc[HEADLINE, "is_headline"]), (
        "the master table must name the same headline"
    )
    h = d.loc[HEADLINE]
    m["cmHQ"] = f"{h['qlike_recal']:.4f}"
    m["cmHQPct"] = s_(h["qlike_pct_vs_ref"], 1)
    m["cmHQCi"] = f"[{s_(h['qlike_diff_ci_lo'], 4)}, {s_(h['qlike_diff_ci_hi'], 4)}]"
    m["cmHDM"] = s_(h["dm_vs_ref"], 2)
    m["cmHDMp"] = f"{h['dm_p_vs_ref']:.3f}"
    assert (
        h["dm_p_vs_ref"] > 0.05 and h["qlike_diff_ci_lo"] < 0 < h["qlike_diff_ci_hi"]
    ), "claim: the accuracy gain over R is not significant"
    m["cmHShMid"] = f_(h["Sharpe_mid"])
    m["cmHShX"] = f_(h["Sharpe_crossed"])
    m["cmHMean"] = f_(h["mean_mid"], 3)
    m["cmHHit"] = f"{100 * h['hit_rate_mid']:.1f}"
    m["cmHBuy"] = f"{h['pct_buy']:.1f}"
    for fill, tag in (("mid", "Mid"), ("crossed", "X")):
        for vs, vtag in (("ref", "Ref"), ("short", "Short")):
            c = f"dSharpe_{fill}_vs_{vs}"
            m[f"cmHVs{vtag}{tag}"] = ci_(h[c], h[c + "_lo"], h[c + "_hi"])
            assert h[c + "_lo"] > 0, (
                f"claim: the headline's {fill} interval vs {vs} is above zero"
            )
    # rank among the master table's table-A forecasts (check rows excluded)
    m["cmHRankSh"] = str(int(a_all["Sharpe_mid"].rank(ascending=False)[HEADLINE]))
    m["cmHRankQ"] = str(int(a_all["qlike_recal"].rank(ascending=True)[HEADLINE]))
    top = a_all["Sharpe_mid"].idxmax()
    assert top != HEADLINE, "prose says the headline is not the table's maximum Sharpe"
    m["cmTopShLabel"] = a_all.loc[top, "label"].split(" [")[0]
    m["cmTopShInputs"] = tt(a_all.loc[top, "label"].split("[")[1].rstrip("]"))
    m["cmTopShMid"] = f_(a_all.loc[top, "Sharpe_mid"])
    # paired against the headline itself (master_table_vs_headline.csv)
    v = pd.read_csv(VS_HEADLINE).set_index("key")
    for key, tag in (
        ("sub_lasso_live_feasible", "Lasso"),
        ("sub_enet_live_feasible", "Enet"),
        ("sub_ridge_all_features", "RidgeAll"),
        (top, "Top"),
    ):
        r = v.loc[key]
        m[f"cmVsH{tag}"] = ci_(
            r["dSharpe_mid_vs_headline"],
            r["dSharpe_mid_vs_headline_lo"],
            r["dSharpe_mid_vs_headline_hi"],
        )
        assert r["dSharpe_mid_vs_headline_lo"] < 0 < r["dSharpe_mid_vs_headline_hi"], (
            f"claim: {key} is indistinguishable from the headline on the trade"
        )
        m[f"cmVsH{tag}Same"] = f"{100 * r['same_position_as_headline']:.0f}"
        m[f"cmVsH{tag}QPct"] = s_(r["qlike_pct_vs_headline"], 1)
        m[f"cmVsH{tag}DM"] = s_(r["dm_vs_headline"], 2)
    m["cmVsHLassoSh"] = f_(d.loc["sub_lasso_live_feasible", "Sharpe_mid"])
    oth = v[v.index.isin(a_all.index) & (v.index != HEADLINE)]
    assert len(oth) == len(a_all) - 1, (len(oth), len(a_all))
    m["cmVsHN"] = str(len(oth))
    m["cmVsHAbove"] = str(int((oth["dSharpe_mid_vs_headline_lo"] > 0).sum()))
    m["cmVsHBelow"] = str(int((oth["dSharpe_mid_vs_headline_hi"] < 0).sum()))
    assert m["cmVsHAbove"] == "0", (
        "claim: no forecast trades above the headline, interval > 0"
    )
    return m


# --------------------------------------------------------------------------- one change at a time
# (block, from key, to key, what changes). Every row is "to minus from" on the same 866 days.
LADDER = [
    (
        "From the paper's forecast to the per-bar specification",
        REFERENCE,
        "pool_ridge_all_features",
        r"block-diagonal ridge $\to$ pooled ridge, \texttt{all\_features}",
    ),
    (
        "Fit the 16:00 bar alone: pooled twin $\\to$ per-bar ridge",
        "pool_ridge_baseline",
        "sub_ridge_baseline",
        r"\texttt{baseline}",
    ),
    (
        "Fit the 16:00 bar alone: pooled twin $\\to$ per-bar ridge",
        "pool_ridge_all_features",
        "sub_ridge_all_features",
        r"\texttt{all\_features}",
    ),
    (
        "Fit the 16:00 bar alone: pooled twin $\\to$ per-bar ridge",
        "pool_ridge_live_feasible",
        HEADLINE,
        r"\texttt{live\_feasible}",
    ),
    (
        "Inputs of the per-bar ridge: other set $\\to$ \\texttt{live\\_feasible}",
        "sub_ridge_all_features",
        HEADLINE,
        r"\texttt{all\_features} $\to$ \texttt{live\_feasible}",
    ),
    (
        "Inputs of the per-bar ridge: other set $\\to$ \\texttt{live\\_feasible}",
        "sub_ridge_baseline",
        HEADLINE,
        r"\texttt{baseline} $\to$ \texttt{live\_feasible}",
    ),
    (
        "Estimator on \\texttt{live\\_feasible}: headline ridge $\\to$ alternative",
        HEADLINE,
        "sub_lasso_live_feasible",
        "lasso",
    ),
    (
        "Estimator on \\texttt{live\\_feasible}: headline ridge $\\to$ alternative",
        HEADLINE,
        "sub_enet_live_feasible",
        "elastic net",
    ),
    (
        "Estimator on \\texttt{live\\_feasible}: headline ridge $\\to$ alternative",
        HEADLINE,
        "subtree_lgbm_live_feasible",
        "LightGBM, untuned",
    ),
    (
        "Estimator on \\texttt{live\\_feasible}: headline ridge $\\to$ alternative",
        HEADLINE,
        "subtree_xgb_live_feasible",
        "XGBoost, untuned",
    ),
    (
        "Estimator on \\texttt{live\\_feasible}: headline ridge $\\to$ alternative",
        HEADLINE,
        "subtree_rf_live_feasible",
        "random forest, untuned",
    ),
    (
        "Estimator on \\texttt{live\\_feasible}: headline ridge $\\to$ alternative",
        HEADLINE,
        "subtree_tuned_live_feasible_lgbm",
        "LightGBM, causally tuned",
    ),
    (
        "Estimator on \\texttt{live\\_feasible}: headline ridge $\\to$ alternative",
        HEADLINE,
        "subtree_tuned_live_feasible_xgb",
        "XGBoost, causally tuned",
    ),
    (
        "Estimator on \\texttt{live\\_feasible}: headline ridge $\\to$ alternative",
        HEADLINE,
        "subtree_tuned_live_feasible_rf",
        "random forest, causally tuned",
    ),
]


def ladder() -> pd.DataFrame:
    """Paired step differences from the master table's own daily series and functions.

    The daily frame (master_table_daily.parquet) and the paired statistics (paired_sharpe,
    day_block_ci, dm_test) are the master table's; a step whose pair the master table
    already stores (vs the reference, vs the headline) is gated against it.
    """
    import sys

    for sub in ("experiments", "notebooks", ""):
        pth = str(ROOT / sub) if sub else str(ROOT)
        if pth not in sys.path:
            sys.path.insert(0, pth)
    import master_table_close as mtc  # the master table's paired Sharpe (same resampled days)
    import score_linear_subsection as base  # its day-block interval
    from src.evaluation.diebold_mariano import dm_test

    daily = pd.read_parquet(DAILY)
    frames = {k: g.set_index("day").sort_index() for k, g in daily.groupby("key")}
    master = pd.read_csv(MASTER).set_index("key")
    vs_h = pd.read_csv(VS_HEADLINE).set_index("key")
    rows = []
    for block, frm, to, what in LADDER:
        a, b = frames[to], frames[frm]
        assert len(a) == len(b) == N_DAYS and (a.index == b.index).all(), (frm, to)
        dq = a["qlike_recal"] - b["qlike_recal"]
        base_q = float(b["qlike_recal"].mean())
        lo, hi = base.day_block_ci(dq)
        dm = dm_test(a["qlike_recal"].to_numpy(), b["qlike_recal"].to_numpy())
        r = {
            "block": block,
            "from": frm,
            "to": to,
            "what": what,
            "qlike_pct": 100.0 * float(dq.mean()) / base_q,
            "qlike_pct_lo": 100.0 * lo / base_q,
            "qlike_pct_hi": 100.0 * hi / base_q,
            "dm": float(dm["dm"]),
            "dm_p": float(dm["p"]),
            "same_position": float((a["q"] == b["q"]).mean()),
        }
        for fill in ("mid", "crossed"):
            dh, slo, shi = mtc.paired_sharpe(
                a[f"ret_{fill}"].to_numpy(), b[f"ret_{fill}"].to_numpy()
            )
            r |= {
                f"dSharpe_{fill}": dh,
                f"dSharpe_{fill}_lo": slo,
                f"dSharpe_{fill}_hi": shi,
            }
        # gates: reproduce the master table's stored pairs where it has them
        if frm == REFERENCE:
            s = master.loc[to]
            stored = (
                s["dSharpe_mid_vs_ref"],
                s["dSharpe_mid_vs_ref_lo"],
                s["dm_vs_ref"],
            )
        elif frm == HEADLINE:
            s = vs_h.loc[to]
            stored = (
                s["dSharpe_mid_vs_headline"],
                s["dSharpe_mid_vs_headline_lo"],
                s["dm_vs_headline"],
            )
        elif to == HEADLINE:
            s = vs_h.loc[
                frm
            ]  # stored the other way round: negate the point and swap bounds
            stored = (
                -s["dSharpe_mid_vs_headline"],
                -s["dSharpe_mid_vs_headline_hi"],
                -s["dm_vs_headline"],
            )
        else:
            stored = None
        if stored is not None:
            mine = (r["dSharpe_mid"], r["dSharpe_mid_lo"], r["dm"])
            assert all(abs(x - y) < GATE_TOL for x, y in zip(mine, stored)), (
                frm,
                to,
                mine,
                stored,
            )
            r["gate"] = "reproduces the master table"
        else:
            r["gate"] = "new pair (no stored comparator)"
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(LADDER_CSV, index=False)
    return out


def write_ladder_table(lad: pd.DataFrame, d: pd.DataFrame) -> None:
    lines = [
        "% AUTO-GENERATED by writeup/make_table_close_main.py -- do not edit.",
        "% Source: writeup/generated/close_main_ladder.csv, computed from",
        "% results/close_master_table/master_table_daily.parquet with the master table's paired",
        "% statistics (experiments/master_table_close.py paired_sharpe, day_block_ci, dm_test).",
        r"\begingroup\small\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lcrccr}",
        r"\toprule",
        r"step (to minus from) & QLIKE, \% [95\%] & DM & $\Delta$Sharpe, mid [95\%]"
        r" & $\Delta$Sharpe, crossed [95\%] & same side, \% \\",
        r"\midrule",
    ]
    block = None
    for _, r in lad.iterrows():
        if r["block"] != block:
            if block is not None:
                lines.append(r"\addlinespace")
            block = r["block"]
            lines.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{block}}}}} \\")
        cells = [
            r"\quad " + r["what"],
            ci_(r["qlike_pct"], r["qlike_pct_lo"], r["qlike_pct_hi"], 1),
            s_(r["dm"], 2),
            ci_(r["dSharpe_mid"], r["dSharpe_mid_lo"], r["dSharpe_mid_hi"]),
            ci_(r["dSharpe_crossed"], r["dSharpe_crossed_lo"], r["dSharpe_crossed_hi"]),
            f"{100 * r['same_position']:.0f}",
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\endgroup", ""]
    LADDER_OUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def ladder_macros(lad: pd.DataFrame) -> dict[str, str]:
    m: dict[str, str] = {}
    key = {(r["from"], r["to"]): r for _, r in lad.iterrows()}

    def one(tag: str, frm: str, to: str) -> None:
        r = key[(frm, to)]
        m[f"cmLd{tag}"] = ci_(
            r["dSharpe_mid"], r["dSharpe_mid_lo"], r["dSharpe_mid_hi"]
        )
        m[f"cmLd{tag}Q"] = ci_(r["qlike_pct"], r["qlike_pct_lo"], r["qlike_pct_hi"], 1)
        m[f"cmLd{tag}DM"] = s_(r["dm"], 2)

    one("Spec", REFERENCE, "pool_ridge_all_features")
    one("PerBarBase", "pool_ridge_baseline", "sub_ridge_baseline")
    one("PerBarAll", "pool_ridge_all_features", "sub_ridge_all_features")
    one("PerBarLive", "pool_ridge_live_feasible", HEADLINE)
    one("InAll", "sub_ridge_all_features", HEADLINE)
    one("InBase", "sub_ridge_baseline", HEADLINE)
    est = lad[(lad["from"] == HEADLINE) & ~lad["to"].str.contains("tuned")]
    assert est.loc[est["dSharpe_mid"].idxmax(), "to"] == "sub_lasso_live_feasible", (
        "prose: lasso"
    )
    assert est.loc[est["dSharpe_mid"].idxmin(), "to"] == "sub_enet_live_feasible", (
        "prose: enet"
    )
    assert (est["dSharpe_mid"] < 0).all(), "claim: changing the estimator costs"
    m["cmLdEstMin"] = s_(est["dSharpe_mid"].min())
    m["cmLdEstMax"] = s_(est["dSharpe_mid"].max())
    m["cmLdN"] = str(len(lad))
    res = lad[(lad["dSharpe_mid_lo"] > 0) | (lad["dSharpe_mid_hi"] < 0)]
    m["cmLdResolved"] = str(len(res))
    assert list(res["to"]) == ["subtree_tuned_live_feasible_lgbm"], (
        "claim: the only resolved step is the tuned LightGBM, below the headline"
    )
    assert (res["dSharpe_mid_hi"] < 0).all()
    path = [
        (REFERENCE, "pool_ridge_all_features"),
        ("pool_ridge_all_features", "sub_ridge_all_features"),
        ("sub_ridge_all_features", HEADLINE),
    ]
    total = sum(key[p]["dSharpe_mid"] for p in path)
    master = pd.read_csv(MASTER).set_index("key")
    assert abs(total - master.loc[HEADLINE, "dSharpe_mid_vs_ref"]) < GATE_TOL, (
        "claim: the three path steps add up to the headline's lead over R"
    )
    top = lad.loc[lad["dSharpe_mid"].idxmax()]
    assert top["block"].startswith("Fit the 16:00 bar alone"), (
        "claim: the largest step is per-bar"
    )
    pb = key[("pool_ridge_baseline", "sub_ridge_baseline")]
    assert pb["qlike_pct_hi"] < 0, (
        "claim: per-bar beats pooled on QLIKE on HAR + calendar"
    )
    assert pb["dSharpe_mid_lo"] < 0 < pb["dSharpe_mid_hi"], (
        "claim: ... but not on the trade"
    )
    return m


# --------------------------------------------------------------------------- tuned trees (A2b)
def tuned_macros(d: pd.DataFrame) -> dict[str, str]:
    """Causally tuned per-bar trees at 16:00 (results/linear_subsection_trees_tuned/a2b/).

    a2b scores the 866 deck days ('deck' sample) with the same 16:00-bar recalibration; its
    trade Sharpe ratios are gated against the master table's rows for the same tables.
    """
    m: dict[str, str] = {}
    q = pd.read_csv(A2B / "qlike_1600_paired.csv")
    q = q[q["sample"] == "deck"]
    t = pd.read_csv(A2B / "trade_1600_paired.csv")
    models = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "random forest"}
    buckets = ("baseline", "all_features", "live_feasible")
    # gate: a2b's tuned-tree Sharpe ratios = the master table's (same tables, same scorer)
    for b in buckets:
        for mod in models:
            row = t[
                (t["bucket"] == b)
                & (t["forecast"] == f"tuned {mod}")
                & (t["reference"] == "ridge")
            ]
            key = f"subtree_tuned_{b}_{mod}"
            assert (
                len(row) == 1
                and abs(row["Sharpe_mid_forecast"].iloc[0] - d.loc[key, "Sharpe_mid"])
                < GATE_TOL
            ), key
    # accuracy against the per-bar ridge on the same inputs: both selection rules, 3 x 3
    vs_r = q[q["forecast"].str.startswith("tuned") & (q["reference"] == "ridge")]
    assert len(vs_r) == 18, len(vs_r)
    m["cmTtVsRidgeN"] = str(len(vs_r))
    m["cmTtVsRidgeBetter"] = str(int(vs_r["better"].sum()))
    m["cmTtVsRidgeWorse"] = str(int(vs_r["worse"].sum()))
    assert m["cmTtVsRidgeBetter"] == "0", (
        "claim: no tuned tree is more accurate than the ridge"
    )
    af = vs_r[
        (vs_r["bucket"] == "all_features") & (vs_r["forecast"] == "tuned lgbm")
    ].iloc[0]
    m["cmTtLgbmAllQ"] = ci_(af["pct"], af["pct_ci_lo"], af["pct_ci_hi"], 1)
    # tuned against untuned (the MSE rule = the arm of record; the QLIKE rule recorded beside it)
    for rule, tag in (("", "Mse"), (" qsel", "Qsel")):
        vs_u = q[
            q["forecast"].isin([f"tuned {mod}{rule}" for mod in models])
            & q["reference"].str.startswith("untuned")
        ]
        vs_u = vs_u[
            vs_u["forecast"].str.split().str[1] == vs_u["reference"].str.split().str[1]
        ]
        assert len(vs_u) == 9, (rule, len(vs_u))
        m[f"cmTtVsUntuned{tag}Better"] = str(int(vs_u["better"].sum()))
        m[f"cmTtVsUntuned{tag}Worse"] = str(int(vs_u["worse"].sum()))
        assert vs_u["better"].sum() == 0, (
            "claim: tuning never makes a tree more accurate"
        )
    lf = q[
        (q["bucket"] == "live_feasible")
        & (q["forecast"] == "tuned lgbm")
        & (q["reference"] == "untuned lgbm")
    ].iloc[0]
    m["cmTtLgbmLiveVsUntuned"] = ci_(lf["pct"], lf["pct_ci_lo"], lf["pct_ci_hi"], 1)
    # the trade, against the per-bar ridge on the same inputs
    for b, tag in (("live_feasible", "Live"), ("all_features", "All")):
        r = t[
            (t["bucket"] == b)
            & (t["forecast"] == "tuned lgbm")
            & (t["reference"] == "ridge")
        ].iloc[0]
        m[f"cmTtLgbm{tag}Sh"] = f_(r["Sharpe_mid_forecast"])
        m[f"cmTtLgbm{tag}RidgeSh"] = f_(r["Sharpe_mid_reference"])
        m[f"cmTtLgbm{tag}VsRidge"] = ci_(r["diff_mid"], r["ci_lo_mid"], r["ci_hi_mid"])
    rl = t[
        (t["bucket"] == "live_feasible")
        & (t["forecast"] == "tuned lgbm")
        & (t["reference"] == "ridge")
    ].iloc[0]
    assert rl["ci_hi_mid"] < 0, (
        "claim: tuned LightGBM on live_feasible trades below the ridge"
    )
    ra = t[
        (t["bucket"] == "all_features")
        & (t["forecast"] == "tuned lgbm")
        & (t["reference"] == "ridge")
    ].iloc[0]
    assert ra["ci_lo_mid"] < 0 < ra["ci_hi_mid"], (
        "claim: tuned LightGBM on all_features unresolved"
    )
    # the best tuned tree on the trade, against the headline (master_table_vs_headline.csv)
    tuned = d[d["family"] == "per-bar tree (tuned)"]
    best = tuned["Sharpe_mid"].idxmax()
    assert best == "subtree_tuned_all_features_lgbm", best
    v = pd.read_csv(VS_HEADLINE).set_index("key").loc[best]
    m["cmTtBestSh"] = f_(d.loc[best, "Sharpe_mid"])
    m["cmTtBestVsH"] = ci_(
        v["dSharpe_mid_vs_headline"],
        v["dSharpe_mid_vs_headline_lo"],
        v["dSharpe_mid_vs_headline_hi"],
    )
    m["cmTtN"] = str(len(tuned))
    return m


def write_numbers(m: dict[str, str]) -> None:
    lines = [
        "% AUTO-GENERATED by writeup/make_table_close_main.py -- do not edit.",
        "% Numbers quoted in the prose of writeup/sections/results_close_option.tex; each is read",
        "% from results/close_master_table/master_table.csv (16:00-bar recalibration).",
    ]
    for k, v in m.items():
        assert re.fullmatch(r"cm[A-Za-z]+", k), k
        assert "nan" not in v.lower() and not any(
            isinstance(x, float) and math.isnan(x) for x in [v]
        ), (k, v)
        lines.append(rf"\newcommand{{\{k}}}{{{v}}}")
    NUMBERS_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    check_manifest()
    d = load()
    write_table(d)
    m = macros(d)
    lad = ladder()
    write_ladder_table(lad, d)
    m.update(ladder_macros(lad))
    m.update(tuned_macros(d))
    write_numbers(m)
    print(
        f"wrote {TABLE_OUT.relative_to(ROOT)} and {NUMBERS_OUT.relative_to(ROOT)} ({len(m)} macros)"
    )


if __name__ == "__main__":
    main()
