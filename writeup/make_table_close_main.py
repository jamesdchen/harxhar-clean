"""The paper's close-option results table and the numbers its prose quotes (D1, 2026-09-29).

Reads the closing-strategy master table (experiments/master_table_close.py; ONE scorer) and
writes

  writeup/generated/table_close_main.tex     Table 7 of the paper: selected master-table rows
  writeup/generated/close_main_numbers.tex   \\cm... macros for every number in Section 5.4's prose
  writeup/generated/table_close_ladder.tex   Table 8: one change at a time (paired steps)
  writeup/generated/close_main_ladder.csv    the executed pairs behind Table 8
  writeup/generated/table_close_pnl.tex      Appendix D: where the headline's P&L comes from

2026-10-06: recalibration removed / intervals removed, user decision.  The scorer is the PLAIN
back-transform f^2 B for every forecast (QLIKE column qlike_raw; the sign(s) position
q = sign(plain - iv_var)); every number is a point estimate.  Statistics, not intervals: the
Diebold-Mariano statistic of the daily QLIKE difference (DM) and the Newey-West HAC t of the
daily mid-return difference (HAC t).  A count of forecasts above / below a comparator counts
point estimates; its ...Sig twin counts those significant at the two-sided SIG level (DM for
QLIKE, HAC t for Sharpe).  The 2026-09-29 scorer (the research scorers' 16:00-bar
recalibration (f^2 + s) B) and its 95 % paired day-block bootstrap intervals are parked in
the comments; the macros that printed only an interval are parked (listed, commented, at the
end of close_main_numbers.tex).

2026-10-06, second round (user decisions): the paper has NO headline model and live-feasible
is NOT part of the paper.  Table 7 carries no headline marker and no live-feasible row (panel D
reads all_features); Table 8 is a set of model-set steps (no path to a headline); every count,
median and range is over the MODEL SET (model_set(): the master table's table A without the
check rows, always short and every live-feasible forecast, camp.is_live); the headline-only
macros are parked, except the P&L decomposition's (\\cmPnl..., Appendix D), which its appendix
still reads.  LSTMs are parked as well (user, 2026-10-06: "the LSTM numbers are also not to be
trusted"): no LSTM forecast (per-bar lstm_*, intraday lstm_intraday_*, both selection rules) is
in the model set, Table 7, Table 8 or any count; the LSTM-only macros are parked.
2026-10-06, third round (user): "make the lean set the third input set in the paper, but it has
to be a follow up to the feature importance study".  The model set is camp.paper_model_set (one
filter for every maker): the paper's forecasts + every per-bar linear, pooled linear and tree
forecast on baseline, all_features and lean_vol_vix.  Table 7 and Table 8 carry the lean rows /
steps; the tree counts read the lean pairs that make_appendix_campaign_16h_tex.py builds.
2026-10-08 parked (lean set; replaced by an MCS): the user parked the lean set everywhere in the paper (a Model Confidence Set over the
features replaces it); the model set is baseline + all_features again (camp.PAPER_SETS), and the
lean rows / steps / macros below are commented out, not deleted.
2026-10-06, fourth round (user): "use the pre-tuned version for all linear models" -- every
per-bar ridge / lasso / elastic net of the paper is arm (f) of results/linear_pretune_2026-10-06/
(master-table key subpt_<est>_<bucket>); the one-penalty sub_<est>_* rows are out of the model
set (camp.in_paper_set), Table 7 panel C, Table 8 and the trees' ridge comparator read subpt_*;
the per-bar OLS baseline stays.  pretune_macros: the facts the methods text quotes.
2026-10-06 (user decision): the POOLED linear rows are pre-tuned as well (poolpt_<est>_<bucket>,
arm f of results/pool_linear_pretune_2026-10-06/, experiments/stack_pool_pretune_yhat.py); the
stored one-penalty pool_* rows stay in the master table but are out of the paper set; poolpt_macros
report the pooled f arm against the causal arm (a) honestly.
2026-10-23 (user, E32, option A): every closing-interval tree of the paper refits at every trading
session.  The model set (camp.paper_model_set) leaves out the two ten-day families (T10, RS10 /
RS10q; camp.TEN_DAY_FAMILIES) and keeps their every-session twins (T1, RS1 / RS1q) and the Optuna
rungs, which were in it already; Table 7 panel D parks its two ten-day rows and Table 8's
estimator steps read the every-session twins.  e32_macros: the forecasts left out and the twin
evidence (results/e32_cadence_gate/twins.csv, picks.csv; experiments/e32_cadence_gate.py).  The
campaign's own steps (the per-window mask on T10 / RS10, the refit cadence T10 -> T1) stay: they
are the record of that ablation, not model-set forecasts.

Nothing is typed by hand: every cell and every macro is read from the CSVs named below.  The
qualitative statements the prose makes about them were asserted here; while the text is
re-checked against the plain point estimates (its TODO-PLAIN lines) they are CHECKED instead
(claim(): recorded, printed and listed at the end of close_main_numbers.tex as holds /
FLIPPED), so a flipped statement is reported rather than stopping the build.  Every gate on the
data (same days, reproduced pairs, provenance) is still asserted.

Sources
  results/close_master_table/master_table.csv          one row per forecast (table A: 866 days)
  results/spxw_pnl/MANIFEST.md                         design panel of each of the paper's tables
  results/close_pnl_decomp/research_scorer/*.csv       the P&L decomposition, plain forecast
  results/close_master_table/master_table_daily.parquet  per-day series (paired steps)
  results/strategy_variations/variations_daily.parquet  the strategy variations' daily legs (F1),
                                                       re-scored here on the plain headline
  The 16:00 campaign (writeup/CAMPAIGN_16H_2026-09-29.md; added 2026-09-30, I4b):
  results/close_master_table/{before_after,master_table_provenance}.csv  design de-dup, provenance
  results/linear_subsection_dedup/gates.csv            per-bar design column counts (agent A)
  results/trees_mask_1600/{mask_pairs,kept_counts}.csv masked trees / LSTM ladder (agent H)
  results/linear_subsection_trees_mask/class/*.csv     CPU class of every chunk (agent H)
  results/linear_subsection_trees_optuna_mask/{report,xclass,xclass_epyc}/, chunk_hosts.csv
                                                       Optuna trees (agent C)
  results/linear_subsection_lstm_intraday/score/lstmi_pairs.csv  intraday LSTM (agent I)
  The campaign's pair CSVs (H, C, I) were scored with the recalibration; their pairs are
  re-scored on the plain forecast from their own forecast tables by
  writeup/make_appendix_campaign_16h_tex.py plain_tables() (imported), which writes them, beside
  the stored value, to writeup/generated/appendix_campaign_16h_plain_pairs.csv.
  (The causally tuned trees of the first pass, results/linear_subsection_trees_tuned/a2b/, were
  read here until 2026-09-30; the master table's random-search rows are now the campaign's
  masked re-run, so the trees paragraph reads the campaign's CSVs instead.)

Usage:  python writeup/make_table_close_main.py
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "writeup", ROOT / "experiments", ROOT / "notebooks", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import make_appendix_campaign_16h_tex as camp  # noqa: E402  (the campaign's plain re-score)

MASTER = ROOT / "results" / "close_master_table" / "master_table.csv"
VS_HEADLINE = ROOT / "results" / "close_master_table" / "master_table_vs_headline.csv"
DAILY = ROOT / "results" / "close_master_table" / "master_table_daily.parquet"
MANIFEST = ROOT / "results" / "spxw_pnl" / "MANIFEST.md"
# the P&L decomposition re-run on the plain forecast (2026-10-06; checklist B1)
PNL_RS = ROOT / "results" / "close_pnl_decomp" / "research_scorer"
CSV_TOL = 5e-6  # the P&L decomposition writes six significant digits; agreement to that precision
GEN = ROOT / "writeup" / "generated"
TABLE_OUT = GEN / "table_close_main.tex"
NUMBERS_OUT = GEN / "close_main_numbers.tex"
LADDER_OUT = GEN / "table_close_ladder.tex"
SV = ROOT / "results" / "strategy_variations"  # strategy variations (F1)
# the 16:00 campaign (writeup/CAMPAIGN_16H_2026-09-29.md)
RES = ROOT / "results"
BEFORE_AFTER = RES / "close_master_table" / "before_after.csv"  # pre-campaign vs now
PROVENANCE = RES / "close_master_table" / "master_table_provenance.csv"
A_GATES = (
    RES / "linear_subsection_dedup" / "gates.csv"
)  # agent A: per-bar design counts
H_PAIRS = RES / "trees_mask_1600" / "mask_pairs.csv"  # agent H: masked ladder, pairs
H_KEPT = RES / "trees_mask_1600" / "kept_counts.csv"  # agent H: kept columns per refit
H_CLASS = (
    RES / "linear_subsection_trees_mask" / "class"
)  # agent H: CPU class census / gate
C_ROOT = RES / "linear_subsection_trees_optuna_mask"  # agent C: Optuna, single class
C_OOS = C_ROOT / "report" / "tuneper_oos.csv"
C_BUDGET = C_ROOT / "report" / "budget_points.csv"
I_PAIRS = (
    RES / "linear_subsection_lstm_intraday" / "score" / "lstmi_pairs.csv"
)  # agent I
# the campaign's rungs are shown for one tree and one input set: LightGBM (the paper's first
# tree column) on live_feasible (the headline's inputs); every other model / input set is
# counted in the prose from the same CSVs
CAMPAIGN_MODEL = "lgbm"
CAMPAIGN_BUCKET = "all_features"  # 2026-10-06: was live_feasible
CAMPAIGN_CLASS = (
    "epyc-7513"  # the one CPU class every canonical tree / LSTM chunk ran on
)
N_DUP = 12  # session-edge columns dropped from the per-bar design (campaign decision 1)
FILL_TOL = 1e-6  # leg-by-leg vs summed-quote crossed fills: equal to far below the 2 decimals shown
PNL_OUT = GEN / "table_close_pnl.tex"  # Appendix D: where the P&L comes from
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
    ("poolpt_ridge_baseline", "ridge", "baseline"),
    ("poolpt_ridge_all_features", "ridge", "all_features"),
    # ("poolpt_ridge_lean_vol_vix", "ridge", "lean_vol_vix"),  # 2026-10-08 parked (lean set; replaced by an MCS)
    # ("poolpt_ridge_live_feasible", "ridge", "live_feasible"),  # 2026-10-06: live-feasible removed
]
PERBAR_LINEAR = [  # 2026-10-06: ridge / lasso / elastic net pre-tuned (subpt_*; was sub_*)
    ("sub_ols_baseline", "OLS", "baseline"),
    ("subpt_ridge_baseline", "ridge", "baseline"),
    ("subpt_lasso_baseline", "lasso", "baseline"),
    ("subpt_enet_baseline", "elastic net", "baseline"),
    ("subpt_ridge_all_features", "ridge", "all_features"),
    ("subpt_lasso_all_features", "lasso", "all_features"),
    ("subpt_enet_all_features", "elastic net", "all_features"),
    # 2026-10-06 (third round): the lean set
    # 2026-10-08 parked (lean set; replaced by an MCS):
    # ("subpt_ridge_lean_vol_vix", "ridge", "lean_vol_vix"),
    # ("subpt_lasso_lean_vol_vix", "lasso", "lean_vol_vix"),
    # ("subpt_enet_lean_vol_vix", "elastic net", "lean_vol_vix"),
    # 2026-10-06: live-feasible removed from the paper, user decision:
    # ("subpt_ridge_live_feasible", "ridge", "live_feasible"),
    # ("subpt_lasso_live_feasible", "lasso", "live_feasible"),
    # ("subpt_enet_live_feasible", "elastic net", "live_feasible"),
]
# Panel D (I4b, 2026-09-30): ONE row per tree / LSTM family of the 16:00 campaign (the
# canonical masked, single-CPU-class runs), all LightGBM / LSTM on one input set, so that
# consecutive rows differ in one thing. The family of each key is asserted against the
# master table, and its provenance (per-window mask, one CPU class) against
# master_table_provenance.csv.  2026-10-06: on all_features (was live_feasible, removed from
# the paper by user decision).
# 2026-10-23 (user, E32): parked -- the two ten-day rows (T10, RS10), out of the model set (camp.TEN_DAY_FAMILIES);
# the rows below are the every-session rungs. Were the first and third rows:
#     ("subtree_lgbm_all_features", "LightGBM, shipped settings, refit every 10", "all_features"),
#     ("subtree_tuned_all_features_lgbm", "LightGBM, random search, refit every 10", "all_features"),
PERBAR_NONLINEAR = [
    (
        "subtree_daily_all_features_lgbm",
        "LightGBM, shipped settings, refit daily",
        "all_features",
    ),
    (
        "subtree_tuned_daily_all_features_lgbm",
        "LightGBM, random search, refit daily",
        "all_features",
    ),
    (
        "subtree_optuna_tp1_all_features_lgbm",
        "LightGBM, Optuna tuned daily, refit daily",
        "all_features",
    ),
    # 2026-10-06: LSTMs parked, user decision:
    # ("lstm_all_features", "LSTM over sessions, refit daily", "all_features"),
    # ("lstm_intraday_all_features", "LSTM over half-hour bars, refit daily", "all_features"),
]
NONLINEAR_FAMILY = {  # key -> the master table's family (one row per family)
    # "subtree_lgbm_all_features": "per-bar tree (untuned)",  # 2026-10-23 (user, E32): parked (ten-day row)
    "subtree_daily_all_features_lgbm": "per-bar tree (untuned, daily refit)",
    # "subtree_tuned_all_features_lgbm": "per-bar tree (tuned)",  # 2026-10-23 (user, E32): parked (ten-day row)
    "subtree_tuned_daily_all_features_lgbm": "per-bar tree (random search, daily)",
    "subtree_optuna_tp1_all_features_lgbm": "per-bar tree (Optuna, TUNE_PER=1, best-of-50)",
    # "lstm_all_features": "LSTM", "lstm_intraday_all_features": "LSTM (intraday sequence)",
    # 2026-10-06: LSTMs parked, user decision
}
PANELS = [
    ("A. The paper's forecasts: one regression for all 48 bars of the day", PAPER),
    (
        "B. Pooled twins: the per-bar specification fitted on all 48 bars",
        POOLED,
    ),
    ("C. Per-bar linear: one regression for the 16:00 bar alone", PERBAR_LINEAR),
    (
        "D. Per-bar trees: one row per rung, the same rows, target and window",  # 2026-10-06: LSTMs parked
        PERBAR_NONLINEAR,
    ),
]
CHECK_FAMILY = "direct rest-of-day at 15:30 (check)"  # the master table's check rows


SIG = (
    camp.SIG
)  # two-sided level of the DM / HAC t significance counts (statistics, not intervals)
RECON_TOL = 1e-12  # strategy variations: rebuilt daily returns = the stored ones (same arithmetic)
# a "tail trade": the 10 best days of the difference to always short carry more than this share
# of it (the prose's reading of \cmPnlDiffExTen against \cmPnlDiff)
TAIL_SHARE = 0.5

CLAIMS: list[tuple[bool, str]] = []  # statements of the text, checked (2026-10-06)
PARKED: list[str] = []  # macros no longer written (2026-10-06), with the reason


def claim(ok: bool, text: str) -> None:
    """A statement the text makes about these numbers: recorded and printed, not asserted.

    2026-10-06: these were assertions on the recalibrated numbers and their intervals; the
    text is being re-checked against the plain point estimates (its TODO-PLAIN lines), so a
    flipped statement is listed (stdout, end of close_main_numbers.tex), not fatal.
    """
    CLAIMS.append((bool(ok), text))


def park(name: str, why: str) -> None:
    PARKED.append(f"{name}: {why}")


def _mtc() -> Any:
    """experiments/master_table_close.py (its sharpe, hac_p, trade_days; the plain scorer)."""
    camp._engine()  # sys.path, the common target and the deck (shared with the re-score)
    import master_table_close as mtc

    return mtc


def hac_p(t: float) -> float:
    return _mtc().hac_p(t)


# --------------------------------------------------------------------------- formatting
def _neg(s: str) -> str:
    return s.replace("-", r"\ensuremath{-}")


def f_(x: float, nd: int = 2) -> str:
    """Plain number, text-safe minus."""
    return _neg(f"{x:.{nd}f}")


def s_(x: float, nd: int = 2) -> str:
    """Signed number, text-safe minus and explicit plus."""
    return _neg(f"{x:+.{nd}f}")


# 2026-10-06: intervals removed, user decision -- no "[lo, hi]" is printed any more:
# def ci_(v: float, lo: float, hi: float, nd: int = 2) -> str:
#     return f"{s_(v, nd)} [{s_(lo, nd)}, {s_(hi, nd)}]"


NUMWORD = (
    "none",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
)  # cmCadN's prose word


def word(n: int) -> str:
    """A small count as a word, for prose ("none", "one", ...); larger counts stay digits."""
    return ("none", "one", "two", "three", "four", "five")[n] if 0 <= n <= 5 else str(n)


def tt(name: str) -> str:
    # presentation rule (user, 2026-10-06): no "adj_" in a presented variable name
    name = re.sub(r"\badj_", "", name)
    return r"\texttt{" + name.replace("_", r"\_") + "}" if name else ""


# --------------------------------------------------------------------------- inputs
def load() -> pd.DataFrame:
    d = pd.read_csv(MASTER).set_index("key")
    assert d.index.is_unique, "master table keys must be unique"
    assert "qlike_raw" in d and "qlike_recal" not in d, (
        "the plain master table (2026-10-06)"
    )
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
    # 2026-10-06: no headline model (user decision) -- was the H marker and bold label:
    # if key == HEADLINE: mark = r"$^{\mathrm{H}}$"; label = r"\textbf{" + label + "}"
    is_ref = key == REFERENCE
    cells = [
        label + mark,
        tt(inputs),
        f"{r['qlike_raw']:.4f}",  # 2026-10-06: plain forecast (was qlike_recal)
        "" if is_ref else s_(r["qlike_pct_vs_ref"], 1),
        "" if is_ref else s_(r["dm_vs_ref"], 2),
        f_(r["Sharpe_mid"]),
        f_(r["Sharpe_crossed"]),
        "" if is_ref else s_(r["dSharpe_mid_vs_ref"]),  # point estimates (2026-10-06)
        s_(r["dSharpe_mid_vs_short"]),
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
        s_(r["dSharpe_mid_vs_ref"]),
        "",
        f"{r['pct_buy']:.1f}",
    ]
    return " & ".join(cells) + r" \\"


def write_table(d: pd.DataFrame) -> None:
    ncol = 10
    lines = [
        "% AUTO-GENERATED by writeup/make_table_close_main.py -- do not edit.",
        "% Source: results/close_master_table/master_table.csv (experiments/master_table_close.py;",
        "% the plain forecast f^2 B, point estimates, DM statistic -- 2026-10-06: recalibration and",
        "% intervals removed, user decision). Selected rows of table A, all on the same 866 trade days.",
        r"\begingroup\small\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{vs R} & \multicolumn{2}{c}{Sharpe$_{\mathrm{ann}}$}"
        r" & \multicolumn{2}{c}{$\Delta$Sharpe, mid} & \\",
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
def _counts(m: dict[str, str], stem: str, x: pd.Series, t: pd.Series, fmt=str) -> None:
    """\\<stem>Above / Below: point estimates above / below zero; ...Sig: of those, the ones
    whose statistic t (HAC t or DM) is significant at SIG.  2026-10-06: were counts of
    intervals wholly above / below zero."""
    s = t.map(hac_p) < SIG
    m[f"{stem}Above"] = fmt(int((x > 0).sum()))
    m[f"{stem}Below"] = fmt(int((x < 0).sum()))
    m[f"{stem}AboveSig"] = fmt(int(((x > 0) & s).sum()))
    m[f"{stem}BelowSig"] = fmt(int(((x < 0) & s).sum()))


def model_set(d: pd.DataFrame) -> pd.DataFrame:
    """The paper's model set: the master table's table A without the check rows, always short
    and the live-feasible forecasts (2026-10-06, user decision: live-feasible is not part of the
    paper; camp.is_live: every input set derived from it)."""
    # 2026-10-06 (third round): camp.paper_model_set, the one filter of every maker (was: table A
    # minus check rows, always short, live-feasible and LSTMs)
    a = camp.paper_model_set(d)
    assert SHORT not in a.index and not (a["family"] == CHECK_FAMILY).any()
    assert (
        not a.index.map(camp.is_live).any()
        and not a["family"].str.contains("LSTM").any()
    )
    return a


def macros(d: pd.DataFrame) -> dict[str, str]:
    m: dict[str, str] = {}
    keys = [k for _, rows in PANELS for k, _, _ in rows] + [SHORT]
    assert not any(camp.is_live(k) for k in keys), "no live-feasible row in Table 7"
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
    # always short reads no forecast: the P&L decomposition's always-short row (same 866 days)
    # must carry the master table's Sharpe ratios (gate kept)
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
        # 2026-10-06: intervals removed, user decision -- was its bootstrap interval:
        # m["cmShortShCi" + ...] = f"[{f_(h['Sharpe_ann_lo'].iloc[0])}, {f_(h['Sharpe_ann_hi'].iloc[0])}]"
        park(
            "cmShortShCi" + ("Mid" if fill == "mid" else "X"),
            "interval of always short's Sharpe",
        )
    m["cmShortT"] = s_(s["t_hac_mid_vs_ref"])  # always short minus R, HAC t (mid)

    paper = d.loc[[k for k, _, _ in PAPER]]
    labels = dict((k, lab) for k, lab, _ in PAPER)
    m["cmPaperShMin"] = f_(paper["Sharpe_mid"].min())
    m["cmPaperShMax"] = f_(paper["Sharpe_mid"].max())
    m["cmPaperXMin"] = f_(paper["Sharpe_crossed"].min())
    m["cmPaperXMax"] = f_(paper["Sharpe_crossed"].max())
    m["cmPaperVsShortMin"] = s_(paper["dSharpe_mid_vs_short"].min())
    m["cmPaperVsShortMax"] = s_(paper["dSharpe_mid_vs_short"].max())
    claim(
        (paper["dSharpe_mid_vs_short"] > 0).all(),
        "5.4: sign(s) lifts every one of the paper's eight forecasts above always short (mid)",
    )
    # was: the paper's forecasts whose interval vs always short lay above zero (N, Who, Ci)
    above = paper["dSharpe_mid_vs_short"] > 0
    sig = paper["t_hac_mid_vs_short"].map(hac_p) < SIG
    m["cmPaperVsShortAboveN"] = word(int(above.sum()))
    m["cmPaperVsShortAboveSigN"] = word(int((above & sig).sum()))
    m["cmPaperVsShortAboveSigWho"] = (
        ", ".join(labels[k] for k in paper.index[above & sig]) or "none"
    )
    park(
        "cmPaperVsShortAboveWho",
        "the paper's forecasts whose interval vs always short lay above zero",
    )
    park("cmPaperVsShortAboveCi", "that forecast's interval")
    others = paper.drop(index=REFERENCE)
    m["cmPaperSameMin"] = f"{100 * others['same_position_as_ref'].min():.0f}"
    # the share of days the rule sells (implied variance above the forecast), quoted in 4.5
    m["cmSellPaperMin"] = f"{100 - paper['pct_buy'].max():.0f}"
    m["cmSellPaperMax"] = f"{100 - paper['pct_buy'].min():.0f}"
    m["cmPaperSameMax"] = f"{100 * others['same_position_as_ref'].max():.0f}"
    # m["cmSellHead"] = ...  # 2026-10-06: no headline model (user decision)
    park("cmSellHead", "the former headline's sell share (no headline model)")

    r = d.loc[REFERENCE]
    m["cmRefQ"] = f"{r['qlike_raw']:.4f}"
    m["cmRefShMid"] = f_(r["Sharpe_mid"])
    m["cmRefShX"] = f_(r["Sharpe_crossed"])
    m["cmRefBuy"] = f"{r['pct_buy']:.0f}"
    m["cmRefVsShort"] = s_(r["dSharpe_mid_vs_short"])  # point (2026-10-06)
    m["cmRefVsShortT"] = s_(r["t_hac_mid_vs_short"])
    claim(
        r["dSharpe_mid_vs_short"] > 0,
        "5.4: R trades above always short (cmRefVsShort > 0)",
    )

    f = d.loc["blk2_inc"]
    m["cmFomcSh"] = f_(f["Sharpe_mid"])
    m["cmFomcD"] = s_(f["dSharpe_mid_vs_ref"])  # point (2026-10-06)
    m["cmFomcDT"] = s_(f["t_hac_mid_vs_ref"])
    claim(
        hac_p(f["t_hac_mid_vs_ref"]) >= SIG,
        "5.4: the FOMC columns' Sharpe gap is not significant (HAC t)",
    )

    best = paper["Sharpe_mid"].idxmax()
    b = d.loc[
        "lasso_f"
    ]  # the fixed lasso (kept for the macros the text may still name)
    m["cmLassoFShMid"] = f_(b["Sharpe_mid"])
    m["cmLassoFVsRef"] = s_(b["dSharpe_mid_vs_ref"])  # point (2026-10-06)
    m["cmLassoFVsRefT"] = s_(b["t_hac_mid_vs_ref"])
    m["cmLassoFSame"] = f"{100 * b['same_position_as_ref']:.0f}"
    # the paper column furthest from R on the trade, and the paper's highest Sharpe
    far = others["dSharpe_mid_vs_ref"].abs().idxmax()
    m["cmPaperFarLabel"] = labels[far]
    m["cmPaperFarVsRef"] = s_(d.loc[far, "dSharpe_mid_vs_ref"])
    m["cmPaperFarVsRefT"] = s_(d.loc[far, "t_hac_mid_vs_ref"])
    m["cmPaperBestLabel"] = labels[best]
    m["cmPaperBestSh"] = f_(paper.loc[best, "Sharpe_mid"])

    # panel D: one row per tree / LSTM family, each the canonical masked single-class run
    prov = pd.read_csv(PROVENANCE).set_index("family")
    fams = [NONLINEAR_FAMILY[k] for k, _, _ in PERBAR_NONLINEAR]
    assert len(set(fams)) == len(fams), "one row per family"
    for k, _, inputs in PERBAR_NONLINEAR:
        fam = NONLINEAR_FAMILY[k]
        assert d.loc[k, "family"] == fam and inputs == CAMPAIGN_BUCKET, (
            k,
            d.loc[k, "family"],
        )
        assert "per-window mask" in prov.loc[fam, "window_mask"], (fam, "mask")
        assert CAMPAIGN_CLASS in prov.loc[fam, "cpu_class"], (fam, "one CPU class")
    ms = model_set(d)
    nl_all = ms[ms["family"].str.contains("tree")]  # the model set has no LSTM
    # every tree / LSTM family of the model set has its row, except the Optuna ablations (other
    # TUNE_PER, best of the first 10 / 25 trials), which Table 8 and the prose read instead
    rest = set(nl_all["family"]) - set(fams)
    assert set(fams) <= set(nl_all["family"]), (
        "a panel-D family is missing from the model set"
    )
    assert all(f.startswith("per-bar tree (Optuna, ") for f in rest), rest
    m["cmNlFamN"] = word(len(fams))
    m["cmNlMasterN"] = str(len(nl_all))  # tree + LSTM forecasts in the model set
    for tag, rows in (
        ("Pool", POOLED),
        ("Sub", PERBAR_LINEAR),
        ("Nl", PERBAR_NONLINEAR),
    ):
        x = d.loc[[k for k, _, _ in rows], "Sharpe_mid"]
        m[f"cm{tag}ShMin"] = f_(x.min())
        m[f"cm{tag}ShMax"] = f_(x.max())

    # every tabled forecast against R: point estimates above / below (and the significant ones)
    tab = d.loc[[k for k in keys if k not in (REFERENCE, SHORT)]]
    m["cmTabVsRefN"] = str(len(tab))
    _counts(m, "cmTabVsRef", tab["dSharpe_mid_vs_ref"], tab["t_hac_mid_vs_ref"])
    m["cmTabVsRefBelow"] = word(
        int((tab["dSharpe_mid_vs_ref"] < 0).sum())
    )  # prose: a word
    m.update(master_macros(d, ms))

    # the crossed spread: how much each tabled forecast gives up
    fc = d.loc[[k for k in keys if k != SHORT]]
    loss = fc["Sharpe_mid"] - fc["Sharpe_crossed"]
    m["cmXLossMin"] = f_(loss.min())
    m["cmXLossMax"] = f_(loss.max())
    return m


def master_macros(d: pd.DataFrame, ms: pd.DataFrame) -> dict[str, str]:
    """The model set as a whole (2026-10-06, user decisions: no headline model, no live-feasible).

    Counts against R and always short (point estimates, ...Sig: HAC t at SIG), medians, the
    sell share, the QLIKE-Sharpe rank correlation and the top Sharpe ratio.  Replaces
    headline_macros (the headline's numbers and the counts against it, parked below).
    """
    m: dict[str, str] = {}
    oth = ms.drop(index=REFERENCE)
    m["cmMasterN"] = str(len(ms))
    iset = ms.index.map(lambda k: camp.input_set(k) or "paper")
    # (cmMasterNLean / MedQLean / MedShLean / MedShXLean: lean_unused_macros, outside the set)
    for b, tag in (("paper", "Paper"), ("baseline", "Base"), ("all_features", "All")):
        x = ms[iset == b]
        m[f"cmMasterN{tag}"] = str(len(x))
        if b != "paper":
            m[f"cmMasterMedQ{tag}"] = f"{x['qlike_raw'].median():.4f}"
            m[f"cmMasterMedSh{tag}"] = f_(x["Sharpe_mid"].median())
            m[f"cmMasterMedShX{tag}"] = f_(x["Sharpe_crossed"].median())
    m["cmMasterSetsN"] = NUMWORD[len(set(iset) - {"paper"})]
    m["cmMasterVsRefN"] = str(len(oth))
    _counts(m, "cmMasterVsRef", oth["dSharpe_mid_vs_ref"], oth["t_hac_mid_vs_ref"])
    _counts(m, "cmMasterVsShort", ms["dSharpe_mid_vs_short"], ms["t_hac_mid_vs_short"])
    m["cmMasterVsShortN"] = str(len(ms))
    q = oth["dm_p_vs_ref"] < SIG
    m["cmMasterQVsRefBelow"] = str(int((oth["qlike_pct_vs_ref"] < 0).sum()))
    m["cmMasterQVsRefAbove"] = str(int((oth["qlike_pct_vs_ref"] > 0).sum()))
    m["cmMasterQVsRefBelowSig"] = str(int(((oth["qlike_pct_vs_ref"] < 0) & q).sum()))
    m["cmMasterQVsRefAboveSig"] = str(int(((oth["qlike_pct_vs_ref"] > 0) & q).sum()))
    m["cmMasterMedQ"] = f"{ms['qlike_raw'].median():.4f}"
    m["cmMasterMedSh"] = f_(ms["Sharpe_mid"].median())
    m["cmMasterMedShX"] = f_(ms["Sharpe_crossed"].median())
    m["cmMasterShMin"] = f_(ms["Sharpe_mid"].min())
    m["cmMasterShMax"] = f_(ms["Sharpe_mid"].max())
    m["cmMasterXMin"] = f_(ms["Sharpe_crossed"].min())
    m["cmMasterXMax"] = f_(ms["Sharpe_crossed"].max())
    m["cmMasterVsShortMed"] = s_(ms["dSharpe_mid_vs_short"].median())
    # Spearman rank correlation of QLIKE with the Sharpe ratio across the model set (a loss:
    # negative = the more accurate forecasts trade better)
    m["cmMasterRankCorr"] = s_(
        ms["qlike_raw"].corr(ms["Sharpe_mid"], method="spearman")
    )
    m["cmMasterRankCorrX"] = s_(
        ms["qlike_raw"].corr(ms["Sharpe_crossed"], method="spearman")
    )
    # the share of days the rule sells (implied variance above the forecast), methods 4.5
    sell = 100 - ms["pct_buy"]
    m["cmSellMasterMin"] = f"{sell.min():.0f}"
    m["cmSellMasterMax"] = f"{sell.max():.0f}"
    m["cmSellMasterMed"] = f"{sell.median():.0f}"
    claim(
        sell.min() > 50,
        "4.5: every forecast of the model set leans short (sells on > half the days)",
    )
    top = ms["Sharpe_mid"].idxmax()
    m["cmTopShLabel"] = ms.loc[top, "label"].split(" [")[0].replace("_", r"\_")
    m["cmTopShInputs"] = tt(ms.loc[top, "label"].split("[")[1].split("]")[0])
    m["cmTopShMid"] = f_(ms.loc[top, "Sharpe_mid"])
    claim(
        hac_p(ms.loc[top, "t_hac_mid_vs_ref"]) >= SIG,
        f"5.4: the model set's top Sharpe ({top}) is not significantly above R (HAC t)",
    )
    # 2026-10-06: no headline model (user decision) -- the headline's numbers, ranks and the
    # counts against it are parked (headline_macros; master_table_vs_headline.csv)
    for nm in (
        "cmHQ",
        "cmHQPct",
        "cmHQCi",
        "cmHDM",
        "cmHDMp",
        "cmHShMid",
        "cmHShX",
        "cmHMean",
        "cmHHit",
        "cmHBuy",
        "cmHVsRefMid",
        "cmHVsRefX",
        "cmHVsShortMid",
        "cmHVsShortX",
        "cmHVsRefT",
        "cmHVsRefP",
        "cmHVsShortT",
        "cmHVsShortP",
        "cmHRankSh",
        "cmHRankQ",
        "cmVsHN",
        "cmVsHAbove",
        "cmVsHBelow",
        "cmVsHAboveSig",
        "cmVsHBelowSig",
        "cmVsHLassoSh",
    ):
        park(nm, "the former headline (no headline model)")
    for tag in ("Lasso", "Enet", "RidgeAll", "Top"):
        for suf in ("", "T", "Same", "QPct", "DM"):
            park(
                f"cmVsH{tag}{suf}",
                "paired against the former headline (no headline model)",
            )
    return m


# --------------------------------------------------------------------------- one change at a time
# (block, from key, to key, what changes). Every row is "to minus from" on the same 866 days.
# (block, from key, to key, what changes). Every row is "to minus from" on the same 866 days.
# 2026-10-06 (user decisions: no headline model, no live-feasible): model-set steps -- the path
# from the paper's forecast to the per-bar specification, fitting the 16:00 bar alone on
# baseline and all_features, the input set, and the estimator on all_features.  The steps to
# the former headline (pooled -> per-bar on live_feasible, all_features / baseline ->
# live_feasible, its estimator alternatives) are parked.
LADDER = [
    (
        "From the paper's forecast to the per-bar specification",
        REFERENCE,
        "poolpt_ridge_all_features",
        r"block-diagonal ridge $\to$ pooled ridge, \texttt{all\_features}",
    ),
    (
        "Fit the 16:00 bar alone: pooled twin $\\to$ per-bar ridge",
        "poolpt_ridge_baseline",
        "subpt_ridge_baseline",
        r"\texttt{baseline}",
    ),
    (
        "Fit the 16:00 bar alone: pooled twin $\\to$ per-bar ridge",
        "poolpt_ridge_all_features",
        "subpt_ridge_all_features",
        r"\texttt{all\_features}",
    ),
    # 2026-10-08 parked (lean set; replaced by an MCS): the lean steps --
    # pooled -> per-bar ridge on lean_vol_vix; baseline -> lean_vol_vix; lean_vol_vix -> all_features
    (
        "Inputs of the per-bar ridge",
        "subpt_ridge_baseline",
        "subpt_ridge_all_features",
        r"\texttt{baseline} $\to$ \texttt{all\_features}",
    ),
    (
        "Estimator on \\texttt{all\\_features}: per-bar ridge $\\to$ alternative",
        "subpt_ridge_all_features",
        "subpt_lasso_all_features",
        "lasso",
    ),
    (
        "Estimator on \\texttt{all\\_features}: per-bar ridge $\\to$ alternative",
        "subpt_ridge_all_features",
        "subpt_enet_all_features",
        "elastic net",
    ),
    # 2026-10-23 (user, E32): parked -- the trees' estimator steps on the ten-day rungs (T10 subtree_<est>_all_features,
    # RS10 subtree_tuned_all_features_<est>), out of the model set (camp.TEN_DAY_FAMILIES); the steps below read their
    # every-session twins (T1, RS1). Were, each from "subpt_ridge_all_features" in the block "Estimator on
    # \\texttt{all\\_features}: per-bar ridge $\\to$ alternative":
    #     ("subtree_lgbm_all_features", "LightGBM, shipped settings, refit every 10"),
    #     ("subtree_xgb_all_features", "XGBoost, shipped settings, refit every 10"),
    #     ("subtree_rf_all_features", "random forest, shipped settings, refit every 10"),
    #     ("subtree_tuned_all_features_lgbm", "LightGBM, random search, refit every 10"),
    #     ("subtree_tuned_all_features_xgb", "XGBoost, random search, refit every 10"),
    #     ("subtree_tuned_all_features_rf", "random forest, random search, refit every 10"),
    # 2026-10-13 parked (user; review items 76-79: the forest on trial 255 of the 400-trial search, behind the rf
    # keys of the master table -- experiments/master_table_close.py PERBAR_SWITCH):
    # (
    #     "Estimator on \\texttt{all\\_features}: per-bar ridge $\\to$ alternative",
    #     "subpt_ridge_all_features",
    #     "subtree_rfdev_all_features",  # 2026-10-06: the dev-tuned forest (was subtree_rf_*)
    #     "random forest, dev-tuned settings, refit every 10",
    # ),
    *[
        (
            "Estimator on \\texttt{all\\_features}: per-bar ridge $\\to$ alternative",
            "subpt_ridge_all_features",
            f"subtree_{rung}all_features_{est}",
            f"{name}, {how}, refit every session",
        )
        for rung, how in (
            ("daily_", "shipped settings"),
            ("tuned_daily_", "random search"),
        )
        for est, name in (
            ("lgbm", "LightGBM"),
            ("xgb", "XGBoost"),
            ("rf", "random forest"),
        )
    ],
]
EST_FROM = (
    "subpt_ridge_all_features"  # 2026-10-06: pre-tuned  # the estimator steps' "from"
)

# The 16:00 campaign's rungs as paired steps (I4b, 2026-09-30), LightGBM / LSTM on
# live_feasible. kind "daily": both forecasts are master-table rows, so the pair is recomputed
# from the daily frame with the master table's statistics and gated against the campaign CSV
# that first reported it; kind "h": one side is not a master-table row (the unmasked tables,
# the LSTM refit every 10), so the pair is read from agent H's mask_pairs.csv; kind
# "before_after": the design de-dup, read from the master table's before/after (same days).
_M = CAMPAIGN_MODEL
_B = CAMPAIGN_BUCKET
B_DESIGN = f"Campaign, design: the {N_DUP} session-edge duplicates dropped"
B_MASK = "Campaign, per-window mask: masked minus unmasked"
B_CAD = r"Campaign, refit cadence: every 10 sessions $\to$ every session"
B_TUNE = "Campaign, tuning (LightGBM, refit daily)"
B_SEQ = r"Campaign, LSTM sequence: sessions $\to$ half-hour bars"
CAMPAIGN: list[dict[str, Any]] = [
    dict(
        block=B_DESIGN,
        kind="before_after",
        frm=f"sub_ridge_{CAMPAIGN_BUCKET}@pre",  # 2026-10-06: was the headline
        to=f"sub_ridge_{CAMPAIGN_BUCKET}",
        what="per-bar ridge, one penalty (the campaign's)",
    ),
    dict(
        block=B_MASK,
        kind="h",
        family="masked - unmasked",
        model=_M,
        frm="T10 unmasked",
        to="T10 masked",
        what="LightGBM, shipped settings, refit every 10",
    ),
    dict(
        block=B_MASK,
        kind="h",
        family="masked - unmasked",
        model=_M,
        frm="T1 unmasked",
        to="T1 masked",
        what="LightGBM, shipped settings, refit daily",
    ),
    dict(
        block=B_MASK,
        kind="h",
        family="masked - unmasked",
        model=_M,
        frm="RS10 unmasked",
        to="RS10 masked",
        what="LightGBM, random search, refit every 10",
    ),
    dict(
        block=B_MASK,
        kind="h",
        family="masked - unmasked",
        model=_M,
        frm="RS1 unmasked",
        to="RS1 masked",
        what="LightGBM, random search, refit daily",
    ),
    # 2026-10-06: LSTMs parked, user decision: the LSTM rungs of the campaign (H: LSTM10 masked - unmasked,
    # LSTM masked - LSTM10 masked; the sessions -> half-hour bars sequence step, B_SEQ)
    dict(
        block=B_CAD,
        kind="daily",
        frm=f"subtree_{_M}_{_B}",
        to=f"subtree_daily_{_B}_{_M}",
        what="LightGBM, shipped settings",
        gate=("h", "masked ladder", _M, "T1 masked", "T10 masked"),
    ),
    dict(
        block=B_TUNE,
        kind="daily",
        frm=f"subtree_daily_{_B}_{_M}",
        to=f"subtree_tuned_daily_{_B}_{_M}",
        what=r"shipped settings $\to$ random search",
        gate=("h", "masked ladder", _M, "RS1 masked", "T1 masked"),
    ),
    dict(
        block=B_TUNE,
        kind="daily",
        frm=f"subtree_daily_{_B}_{_M}",
        to=f"subtree_optuna_tp1_{_B}_{_M}",
        what=r"shipped settings $\to$ Optuna, tuned daily",
        gate=("c", "vs T1", _M, "tp1", "T1"),
    ),
    dict(
        block=B_TUNE,
        kind="daily",
        frm=f"subtree_optuna_tp250_{_B}_{_M}",
        to=f"subtree_optuna_tp1_{_B}_{_M}",
        what=r"Optuna: tuned every 250 sessions $\to$ daily",
        gate=("c", "tune_per vs 250", _M, "tp1", "tp250"),
    ),
]
# H's masked rung labels -> the master-table rows they are (their levels are gated)
H_LEVEL_KEY = {
    "T10 masked": f"subtree_{_M}_{_B}",
    "T1 masked": f"subtree_daily_{_B}_{_M}",
    "RS10 masked": f"subtree_tuned_{_B}_{_M}",
    "RS1 masked": f"subtree_tuned_daily_{_B}_{_M}",
    "LSTM masked": f"lstm_{_B}",
}


def _plain() -> dict[str, pd.DataFrame]:
    """The campaign's pairs (H, C, I) re-scored on the plain forecast (2026-10-06)."""
    return camp.plain_tables(("H", "C", "I"))


def _h_row(h: pd.DataFrame, family: str, model: str, a: str, b: str) -> pd.Series:
    x = h[
        (h["family"] == family)
        & (h["model"] == model)
        & (h["bucket"] == _B)
        & (h["a"] == a)
        & (h["b"] == b)
    ]
    assert len(x) == 1, (family, model, a, b, len(x))
    return x.iloc[0]


def _gate_row(g: tuple) -> pd.Series:
    """The plain re-score of a campaign pair (its CSV's identifiers), one row.

    2026-10-06: was the stored (recalibrated) CSV row, whose (QLIKE diff, dm, dSharpe mid, lo,
    hi) the ladder reproduced; the gate now reproduces the plain re-score of the same pair, and
    the stored recalibrated value is reported beside it (sharpe_mid_diff_recal).
    """
    src, fam, model, a, b = g
    p = _plain()
    if src == "h":
        return _h_row(p["H"], fam, model, a, b)
    if src == "c":
        c = p["C"]
        x = c[
            (c["source"] == "C tuneper_oos")
            & (c["comparison"] == fam)
            & (c["model"] == model)
            & (c["bucket"] == _B)
            & (c["a_kind"] == a)
            & (c["b_kind"] == b)
        ]
        assert len(x) == 1, (g, len(x))
        return x.iloc[0]
    i = p["I"]
    x = i[(i["bucket"] == _B) & (i["a"] == a) & (i["b"] == b)]
    assert len(x) == 1, (g, len(x))
    return x.iloc[0]


def campaign_csv_rows(master: pd.DataFrame) -> list[dict]:
    """The campaign steps whose one side is not a master-table row (re-scored / read)."""
    h = _plain()["H"]
    rows = []
    for c in CAMPAIGN:
        if c["kind"] == "h":
            r = _h_row(h, c["family"], c["model"], c["to"], c["frm"])
            assert int(r["n_days"]) == N_DAYS, c
            for lab, sh in (("a", r["sharpe_mid_a"]), ("b", r["sharpe_mid_b"])):
                k = H_LEVEL_KEY.get(r[lab])
                if (
                    k is not None
                ):  # a masked rung that is a master-table row: same level
                    assert abs(sh - master.loc[k, "Sharpe_mid"]) < GATE_TOL, (k, sh)
            rows.append(
                {
                    "block": c["block"],
                    "from": "H:" + c["frm"],
                    "to": "H:" + c["to"],
                    "what": c["what"],
                    "qlike_pct": r["qlike_pct"],
                    "dm": r["dm"],
                    "dm_p": r["dm_p"],
                    "same_position": r["same_side_share"],
                    "dSharpe_mid": r["sharpe_mid_diff"],
                    "dSharpe_crossed": r["sharpe_crossed_diff"],
                    "t_hac_mid": r["t_hac"],
                    "t_p": r["t_p"],
                    "dSharpe_mid_recal": r["sharpe_mid_diff_recal"],
                    "gate": "plain re-score of results/trees_mask_1600/mask_pairs.csv "
                    "(writeup/generated/appendix_campaign_16h_plain_pairs.csv; levels gated)",
                }
            )
        elif c["kind"] == "before_after":
            ba = pd.read_csv(BEFORE_AFTER).set_index("key").loc[c["to"]]
            assert ba["status"] == "both" and int(ba["n_days_paired"]) == N_DAYS, c
            t = ba["t_hac_mid_after_vs_before"]
            rows.append(
                {
                    "block": c["block"],
                    "from": c["frm"],
                    "to": c["to"],
                    "what": c["what"],
                    "qlike_pct": ba["qlike_pct_change"],
                    "dm": float("nan"),
                    "dm_p": float("nan"),  # not stored (see the gate)
                    "same_position": 1.0
                    - ba["positions_changed"] / ba["n_days_paired"],
                    "dSharpe_mid": ba["dSharpe_mid"],
                    "dSharpe_crossed": ba["dSharpe_crossed"],
                    "t_hac_mid": t,
                    "t_p": hac_p(t),
                    "dSharpe_mid_recal": float("nan"),
                    "gate": "read from results/close_master_table/before_after.csv (plain)",
                }
            )
    return rows


def ladder() -> pd.DataFrame:
    """Paired step differences from the master table's own daily series and functions.

    The daily frame (master_table_daily.parquet: the plain forecast's QLIKE, position and
    returns) and the statistics (the master table's sharpe, dm_test, the HAC t newey_west_t) are
    the master table's; a step whose pair the master table already stores (vs the reference,
    vs the headline) is gated against it, and a campaign step against the plain re-score of
    the campaign CSV that reported it.  2026-10-06: point estimates only (was: day_block_ci
    for the QLIKE difference and paired_sharpe's bootstrap interval for the Sharpe one).
    """
    if not DAILY.exists():
        # the per-day frame (6 MB) is not committed; fall back to the committed executed
        # pairs, which the last run with the frame present wrote and gated
        print(
            f"note: {DAILY.relative_to(ROOT)} absent -- Table 8 from {LADDER_CSV.name}"
        )
        return pd.read_csv(LADDER_CSV)
    mtc = _mtc()
    import atm_straddle_lib as asl  # the master table's newey_west_t
    from src.evaluation.diebold_mariano import dm_test

    daily = pd.read_parquet(DAILY)
    assert "qlike_raw" in daily and "qlike_recal" not in daily, "the plain daily frame"
    frames = {k: g.set_index("day").sort_index() for k, g in daily.groupby("key")}
    master = pd.read_csv(MASTER).set_index("key")
    vs_h = pd.read_csv(VS_HEADLINE).set_index("key")
    rows = []
    steps: list[tuple[str, str, str, str, Any]] = [
        (block, frm, to, what, None) for block, frm, to, what in LADDER
    ]
    csv_rows = campaign_csv_rows(master)
    for c in CAMPAIGN:
        if c["kind"] == "daily":
            steps.append((c["block"], c["frm"], c["to"], c["what"], c["gate"]))
    for block, frm, to, what, cgate in steps:
        a, b = frames[to], frames[frm]
        assert len(a) == len(b) == N_DAYS and (a.index == b.index).all(), (frm, to)
        dq = a["qlike_raw"] - b["qlike_raw"]
        base_q = float(b["qlike_raw"].mean())
        dm = dm_test(a["qlike_raw"].to_numpy(), b["qlike_raw"].to_numpy())
        t_hac = asl.newey_west_t(a["ret_mid"].to_numpy() - b["ret_mid"].to_numpy())[0]
        r = {
            "block": block,
            "from": frm,
            "to": to,
            "what": what,
            "qlike_pct": 100.0 * float(dq.mean()) / base_q,
            "dm": float(dm["dm"]),
            "dm_p": float(dm["p"]),
            "same_position": float((a["q"] == b["q"]).mean()),
            "t_hac_mid": t_hac,
            "t_p": hac_p(t_hac),
        }
        # lo, hi = base.day_block_ci(dq)  # 2026-10-06: intervals removed, user decision
        for fill in ("mid", "crossed"):
            r[f"dSharpe_{fill}"] = mtc.sharpe(a[f"ret_{fill}"].to_numpy()) - mtc.sharpe(
                b[f"ret_{fill}"].to_numpy()
            )
            # dh, slo, shi = mtc.paired_sharpe(...)  # 2026-10-06: the interval is parked
        if cgate is not None:
            # a campaign step: the plain re-score of its CSV's pair (the same loader and
            # scorer, from the forecast tables) reports the same pair on the same days
            g = _gate_row(cgate)
            mine_c = (float(dq.mean()), r["dm"], r["dSharpe_mid"], r["t_hac_mid"])
            theirs = (g["qlike_diff"], g["dm"], g["sharpe_mid_diff"], g["t_hac"])
            assert all(abs(x - y) < GATE_TOL for x, y in zip(mine_c, theirs)), (
                cgate,
                mine_c,
                theirs,
            )
            r["dSharpe_mid_recal"] = g["sharpe_mid_diff_recal"]
            r["gate"] = (
                f"reproduces the plain re-score of the campaign CSV ({cgate[0]}: "
                f"{cgate[1] or 'pairs'})"
            )
            rows.append(r)
            continue
        # gates: reproduce the master table's stored pairs where it has them
        if frm == REFERENCE:
            s = master.loc[to]
            stored = (s["dSharpe_mid_vs_ref"], s["t_hac_mid_vs_ref"], s["dm_vs_ref"])
        elif frm == HEADLINE:
            s = vs_h.loc[to]
            stored = (
                s["dSharpe_mid_vs_headline"],
                s["t_hac_mid_vs_headline"],
                s["dm_vs_headline"],
            )
        elif to == HEADLINE:
            s = vs_h.loc[frm]  # stored the other way round: negate
            stored = (
                -s["dSharpe_mid_vs_headline"],
                -s["t_hac_mid_vs_headline"],
                -s["dm_vs_headline"],
            )
        else:
            stored = None
        # 2026-10-06: the stored lower bound (dSharpe_mid_vs_ref_lo / _vs_headline_lo / _hi)
        # was the second member of the gate; it is replaced by the HAC t
        if stored is not None:
            mine = (r["dSharpe_mid"], r["t_hac_mid"], r["dm"])
            assert all(abs(x - y) < GATE_TOL for x, y in zip(mine, stored)), (
                frm,
                to,
                mine,
                stored,
            )
            r["gate"] = "reproduces the master table"
        else:
            r["gate"] = "new pair (no stored comparator)"
        r["dSharpe_mid_recal"] = float("nan")
        rows.append(r)
    # campaign steps in CAMPAIGN order, after the steps from the paper's forecast
    by_pair = {(r["from"], r["to"]): r for r in rows + csv_rows}
    ordered = rows[: len(LADDER)]
    for c in CAMPAIGN:
        pre = "H:" if c["kind"] == "h" else ""
        ordered.append(by_pair[(pre + c["frm"], pre + c["to"])])
    out = pd.DataFrame(ordered)
    assert len(out) == len(LADDER) + len(CAMPAIGN)
    # the scorer change on the campaign steps, reported (recalibrated -> plain dSharpe mid)
    for _, r in out[out["dSharpe_mid_recal"].notna()].iterrows():
        print(
            f"  ladder {r['from']} -> {r['to']}: dSharpe mid recalibrated "
            f"{r['dSharpe_mid_recal']:+.2f} -> plain {r['dSharpe_mid']:+.2f}"
        )
    out.to_csv(LADDER_CSV, index=False, lineterminator="\n")
    return out


def write_ladder_table(lad: pd.DataFrame, d: pd.DataFrame) -> None:
    lines = [
        "% AUTO-GENERATED by writeup/make_table_close_main.py -- do not edit.",
        "% Source: writeup/generated/close_main_ladder.csv, computed from",
        "% results/close_master_table/master_table_daily.parquet (the plain forecast f^2 B) with the",
        "% master table's statistics (experiments/master_table_close.py sharpe, dm_test, newey_west_t);",
        "% campaign rows: recomputed the same way and gated against the plain re-score of the campaign",
        "% CSV, or read from it (writeup/generated/appendix_campaign_16h_plain_pairs.csv) /",
        "% results/close_master_table/before_after.csv (column gate of close_main_ladder.csv).",
        "% Point estimates; DM: Diebold-Mariano statistic of the daily QLIKE difference (---: not",
        "% stored for that pair); HAC t: Newey-West t of the daily mid-return difference.",
        "% 2026-10-06: recalibration and intervals removed, user decision.",
        r"\begingroup\small\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"step (to minus from) & QLIKE, \% & DM & $\Delta$Sharpe, mid & HAC $t$"
        r" & $\Delta$Sharpe, crossed & same side, \% \\",
        r"\midrule",
    ]
    block = None
    for _, r in lad.iterrows():
        if r["block"] != block:
            if block is not None:
                lines.append(r"\addlinespace")
            block = r["block"]
            lines.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{block}}}}} \\")
        cells = [
            r"\quad " + r["what"],
            s_(r["qlike_pct"], 1),
            "---" if pd.isna(r["dm"]) else s_(r["dm"], 2),
            s_(r["dSharpe_mid"]),
            "---" if pd.isna(r["t_hac_mid"]) else s_(r["t_hac_mid"], 2),
            s_(r["dSharpe_crossed"]),
            f"{100 * r['same_position']:.0f}",
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\endgroup", ""]
    LADDER_OUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def ladder_macros(lad: pd.DataFrame) -> dict[str, str]:
    m: dict[str, str] = {}
    key = {(r["from"], r["to"]): r for _, r in lad.iterrows()}

    def one(tag: str, frm: str, to: str) -> None:
        if (
            frm,
            to,
        ) not in key:  # a step whose forecast has not landed yet (pending_rows)
            for suf in ("", "Q", "T", "DM"):
                park(
                    f"cmLd{tag}{suf}",
                    f"pending: {frm} or {to} not in the master table yet",
                )
            return
        r = key[(frm, to)]
        m[f"cmLd{tag}"] = s_(r["dSharpe_mid"])  # point (2026-10-06; was ci_)
        m[f"cmLd{tag}Q"] = s_(r["qlike_pct"], 1)
        if not pd.isna(
            r["t_hac_mid"]
        ):  # undefined when the daily difference is all zero
            m[f"cmLd{tag}T"] = s_(
                r["t_hac_mid"]
            )  # HAC t of the daily mid-return difference
        if not pd.isna(r["dm"]):  # the before/after pair stores no DM statistic
            m[f"cmLd{tag}DM"] = s_(r["dm"], 2)

    one("Spec", REFERENCE, "poolpt_ridge_all_features")
    one("PerBarBase", "poolpt_ridge_baseline", "subpt_ridge_baseline")
    one("PerBarAll", "poolpt_ridge_all_features", "subpt_ridge_all_features")
    one("In", "subpt_ridge_baseline", "subpt_ridge_all_features")  # new (2026-10-06)
    # 2026-10-06 (third round): the lean set's steps
    # one("PerBarLean", ...); one("InBaseLean", ...); one("InLeanAll", ...)  # 2026-10-08 parked (lean set; replaced by an MCS)
    # (their macros stay defined, outside Table 8, by lean_unused_macros -- the text still names them)
    # 2026-10-06 parked (no headline model, no live-feasible; user decisions): the steps to the
    # former headline -- pooled -> per-bar on live_feasible, all_features / baseline -> live_feasible
    # one("PerBarLive", "poolpt_ridge_live_feasible", HEADLINE); one("InAll", ...); one("InBase", ...)
    for tg in ("PerBarLive", "InAll", "InBase"):
        for suf in ("", "Q", "T", "DM"):
            park(
                f"cmLd{tg}{suf}",
                "a step to the former headline (no headline, no live-feasible)",
            )
    est = lad[(lad["from"] == EST_FROM) & ~lad["to"].str.contains("tuned")]
    hi_, lo_ = (
        est.loc[est["dSharpe_mid"].idxmax(), "to"],
        est.loc[est["dSharpe_mid"].idxmin(), "to"],
    )
    m["cmLdEstMin"] = s_(est["dSharpe_mid"].min())
    m["cmLdEstMax"] = s_(est["dSharpe_mid"].max())
    m["cmLdEstMinLabel"] = tt(str(lo_))
    m["cmLdEstMaxLabel"] = tt(str(hi_))
    # 2026-10-13: the two ends in words (the step's own label up to its first comma, e.g. "random forest"), so the
    # prose names them by macro (the minimum moved from LightGBM to the forest when the trees took their top entries)
    who = {to: lab.split(",")[0] for _, frm, to, lab in LADDER if frm == EST_FROM}
    m["cmLdEstMinWho"] = who[lo_]
    m["cmLdEstMaxWho"] = who[hi_]
    m["cmLdN"] = str(len(lad))
    m["cmLdNPath"] = str(len(LADDER))
    m["cmLdNCampaign"] = (
        word(len(CAMPAIGN)) if len(CAMPAIGN) <= 5 else str(len(CAMPAIGN))
    )
    # 2026-10-06: was the steps whose Sharpe interval excluded zero; now the steps whose HAC t
    # of the daily mid-return difference is significant at SIG (and cmLdQResolved: DM)
    res = lad[lad["t_p"] < SIG]
    m["cmLdResolved"] = word(len(res))
    m["cmLdQResolved"] = word(int((lad["dm_p"] < SIG).sum()))
    claim(
        len(res) == 0,
        f"5.4: no step of Table 8 moves the Sharpe ratio significantly (HAC t; {len(res)} do)",
    )
    # the campaign's steps (the last block of Table 8)
    cmp_tags = [
        "Dedup",
        "MaskTten",
        "MaskTone",
        "MaskRsten",
        "MaskRsone",
        "CadLgbm",
        "TuneRs",
        "TuneOp",
        "TuneOpPer",
    ]
    for tg in ("MaskLstm", "CadLstm", "Seq"):  # 2026-10-06: LSTMs parked, user decision
        for suf in ("", "Q", "T", "DM"):
            park(f"cmLdC{tg}{suf}", "an LSTM step of the campaign (LSTMs parked)")
    assert len(cmp_tags) == len(CAMPAIGN)
    for tag, c in zip(cmp_tags, CAMPAIGN):
        pre = "H:" if c["kind"] == "h" else ""
        one("C" + tag, pre + c["frm"], pre + c["to"])
    cad = key[(f"subtree_{_M}_{_B}", f"subtree_daily_{_B}_{_M}")]
    claim(
        cad["qlike_pct"] < 0,
        "5.4: daily refits make LightGBM more accurate (cmLdCCadLgbmQ < 0)",
    )
    claim(cad["t_p"] >= SIG, "5.4: ... without moving its trade significantly (HAC t)")
    for tg, pr in (
        ("TuneRs", (f"subtree_daily_{_B}_{_M}", f"subtree_tuned_daily_{_B}_{_M}")),
        ("TuneOp", (f"subtree_daily_{_B}_{_M}", f"subtree_optuna_tp1_{_B}_{_M}")),
    ):
        claim(
            key[pr]["qlike_pct"] > 0, f"5.4: tuning does not help ({tg}: QLIKE % > 0)"
        )
    # the design de-dup on the per-bar ridge: its QLIKE moves at floating-point rounding and no
    # position or Sharpe ratio moves (the point estimate)
    dd = lad.iloc[len(LADDER)]
    assert dd["to"] == f"sub_ridge_{_B}" and dd["from"] == f"sub_ridge_{_B}@pre"
    claim(
        abs(dd["qlike_pct"]) < GATE_TOL,
        "5.4: the de-dup moves the per-bar ridge's QLIKE only at rounding",
    )
    claim(
        dd["dSharpe_mid"] == 0 and dd["dSharpe_crossed"] == 0,
        "5.4: the de-dup leaves the trade",
    )
    claim(dd["same_position"] == 1, "5.4: no position of the per-bar ridge changed")
    # 2026-10-06: the identity "the three path steps add up to the headline's lead over R" and the
    # claims on the path to the headline are parked (no headline model)
    claim(
        abs(key[(REFERENCE, "poolpt_ridge_all_features")]["dSharpe_mid"])
        < abs(
            key[("poolpt_ridge_all_features", "subpt_ridge_all_features")][
                "dSharpe_mid"
            ]
        ),
        "5.4: the specification step moves little on the trade (cmLdSpec)",
    )
    pb = key[("poolpt_ridge_baseline", "subpt_ridge_baseline")]
    claim(
        pb["qlike_pct"] < 0,
        "5.4: per-bar beats pooled on QLIKE on HAR + calendar (cmLdPerBarBaseQ < 0)",
    )
    claim(pb["dm_p"] < SIG, "5.4: ... significantly (DM)")
    claim(pb["t_p"] >= SIG, "5.4: ... but not on the trade (HAC t not significant)")
    return m


# --------------------------------------------------------------------------- the 16:00 campaign
def _below_above(x: pd.DataFrame, col: str, p: str | None = None) -> tuple[int, int]:
    """Point estimates below / above zero; with p (a p-value column), the significant ones.

    2026-10-06: intervals removed, user decision -- was: intervals wholly below / above zero.
    """
    s = x[p] < SIG if p else pd.Series(True, index=x.index)
    return int(((x[col] < 0) & s).sum()), int(((x[col] > 0) & s).sum())


def campaign_macros(d: pd.DataFrame) -> dict[str, str]:
    """What the 16:00 campaign changed, and what the trees / LSTMs show on the new tables.

    I4b 2026-09-30. Replaces tuned_macros (the first-pass causally tuned trees, A2b): the
    master table's random-search rows are now the campaign's masked, single-CPU-class re-run.
    Every count is read from the campaign's CSVs (agents A, H, C, I) -- 2026-10-06: their pairs
    re-scored on the plain forecast (camp.plain_tables) -- and every count is of point
    estimates, with a ...Sig twin (DM for QLIKE, HAC t for Sharpe, at SIG).  The statements
    the prose makes about them are checked (claim).
    """
    m: dict[str, str] = {}
    # 2026-10-06: no live-feasible (user decision; b761b28) -- every count over these input sets
    buckets = tuple(camp.BUCKETS)
    btag = {"baseline": "Base", "live_feasible": "Live", "all_features": "All"}
    nb = len(buckets)
    n_tree, n_rung, n_tp = (
        len(camp.TREES),
        4,
        4,
    )  # trees; H's T10/T1/RS10/RS1; C's TUNE_PER 1/5/25/250
    for nm in (
        "cmPNewLive",
        "cmPOldLive",
        "cmKeptLive",
        "cmKeptLiveMin",
        "cmKeptLiveMax",
        "cmLstmCadLive",
    ):
        park(nm, "live_feasible input set (live-feasible removed from the paper)")

    # ---- design: the session-edge duplicates dropped (agent A's gates)
    g = pd.read_csv(A_GATES)
    for b in buckets:
        key = f"sub_ridge_{b}"
        new = g[
            g["gate"].str.startswith("(a) design: no session-edge column")
            & (g["forecast"] == key)
        ]
        drop = g[
            g["gate"].str.startswith("(a) old design minus new")
            & (g["forecast"] == key)
        ]
        assert len(new) == 1 and len(drop) == 1, b
        assert new["ok"].all() and drop["ok"].all(), b
        assert int(drop["value"].iloc[0]) == N_DUP, (b, "12 columns dropped")
        m[f"cmPNew{btag[b]}"] = str(int(new["value"].iloc[0]))
        m[f"cmPOld{btag[b]}"] = str(int(new["value"].iloc[0]) + N_DUP)
    m["cmNDup"] = str(N_DUP)  # = every arm's gate value above
    ba = pd.read_csv(BEFORE_AFTER)
    assert "qlike_raw_before" in ba, "the plain before / after (2026-10-06)"
    lin = ba[
        (ba["status"] == "both")
        & (ba["table_after"] == "A")  # the 866 days (table B's buckets start in 2022)
        & ba["what_changed"].str.startswith("design de-dup (the mask had already")
        & ~ba["key"].map(camp.is_live)  # 2026-10-06: no live-feasible
    ].copy()
    assert len(lin) > 0 and (lin["n_days_paired"] == N_DAYS).all()
    m["cmDedupN"] = str(len(lin))
    m["cmDedupQMax"] = f"{lin['qlike_pct_change'].abs().max():.2f}"
    m["cmDedupPosMax"] = word(int(lin["positions_changed"].max()))
    lin["t_p"] = lin["t_hac_mid_after_vs_before"].map(hac_p)
    n = sum(_below_above(lin, "dSharpe_mid", "t_p"))
    claim(
        n == 0,
        "5.4: no linear forecast's trade moved significantly with the de-dup (HAC t)",
    )
    # (2026-10-06: the claim "the headline keeps every position" is parked: no headline model)

    # ---- the per-window mask: kept columns (agent H), identical across rungs and models
    k = pd.read_csv(H_KEPT)
    for b in buckets:
        kb = k[k["bucket"] == b]
        assert kb["kept_n_median"].nunique() == 1 and kb["kept_n_min"].nunique() == 1, b
        assert kb["kept_n_max"].nunique() == 1, b
        assert int(kb["p"].iloc[0]) == int(m[f"cmPNew{btag[b]}"]), (b, "same design")
        m[f"cmKept{btag[b]}"] = f"{kb['kept_n_median'].iloc[0]:.0f}"
        m[f"cmKept{btag[b]}Min"] = str(int(kb["kept_n_min"].iloc[0]))
        m[f"cmKept{btag[b]}Max"] = str(int(kb["kept_n_max"].iloc[0]))
    p = _plain()
    h = camp.nonlive(p["H"])  # 2026-10-06: no live-feasible
    ht = h[h["model"] != "lstm"]
    mu = ht[ht["family"] == "masked - unmasked"]
    # assert len(mu) == n_tree * n_rung * mu["bucket"].nunique() and (mu["n_days"] == N_DAYS).all()
    # 2026-10-13: XGBoost / the forest on their top entries were re-run unmasked at T10 / T1 only (RS10 / RS1: none)
    n_mask = (n_rung + 2 * 2) * mu[
        "bucket"
    ].nunique()  # LightGBM 4 rungs, XGBoost and the forest 2 each
    assert len(mu) == n_mask and (mu["n_days"] == N_DAYS).all(), (len(mu), n_mask)
    m["cmMaskN"] = str(len(mu))
    m["cmMaskQMed"] = f"{mu['qlike_pct'].abs().median():.1f}"
    qb, qa = _below_above(mu, "qlike_diff")
    sb, sa = _below_above(mu, "sharpe_mid_diff")
    qbs, qas = _below_above(mu, "qlike_diff", "dm_p")
    sbs, sas = _below_above(mu, "sharpe_mid_diff", "t_p")
    m["cmMaskQAbove"] = word(qa)
    m["cmMaskShBelow"] = word(sb)
    m["cmMaskQAboveSig"] = word(qas)
    m["cmMaskShBelowSig"] = word(sbs)
    claim(
        qbs == 0 and sas == 0,
        "5.4: the mask made no tree significantly better (DM / HAC t)",
    )

    # ---- trees against the per-bar ridge on the same inputs (H: 4 rungs, C: 4 TUNE_PER)
    c = camp.nonlive(p["C"])
    c = c[c["source"] == "C tuneper_oos"]
    hr = ht[ht["family"] == "masked - ridge"]
    cr = c[c["comparison"] == "vs ridge"]
    assert len(hr) == n_tree * n_rung * hr["bucket"].nunique()
    assert len(cr) == n_tree * n_tp * cr["bucket"].nunique()
    tr = pd.concat([hr, cr])
    qb, qa = _below_above(tr, "qlike_diff")
    sb, sa = _below_above(tr, "sharpe_mid_diff")
    m["cmTrVsRidgeN"] = str(len(tr))
    m["cmTrSetsN"] = NUMWORD[tr["bucket"].nunique()]  # input sets of the tree counts
    m["cmMaskSetsN"] = NUMWORD[mu["bucket"].nunique()]  # (no unmasked lean run)
    m["cmTrVsRidgeQAbove"] = str(qa)
    m["cmTrVsRidgeShBelow"] = word(sb)
    m["cmTrVsRidgeQAboveSig"] = str(_below_above(tr, "qlike_diff", "dm_p")[1])
    m["cmTrVsRidgeShBelowSig"] = word(_below_above(tr, "sharpe_mid_diff", "t_p")[0])
    claim(
        qb == 0,
        f"5.4: no tree more accurate than the ridge ({qb} QLIKE differences below 0)",
    )
    claim(
        sa == 0,
        f"5.4: no tree trades above the ridge ({sa} Sharpe differences above 0)",
    )

    # ---- refit cadence: T1 - T10 (H's masked ladder)
    cad = ht[
        (ht["family"] == "masked ladder")
        & (ht["a"] == "T1 masked")
        & (ht["b"] == "T10 masked")
    ]
    assert len(cad) == n_tree * cad["bucket"].nunique()
    claim((cad["qlike_diff"] < 0).all(), "5.4: daily refits lower every tree's QLIKE")
    m["cmCadN"] = NUMWORD[len(cad)]  # was "nine" (three input sets)
    m["cmCadQBelow"] = word(_below_above(cad, "qlike_diff")[0])
    m["cmCadQBelowSig"] = word(_below_above(cad, "qlike_diff", "dm_p")[0])
    up = cad[(cad["model"] == "xgb") & (cad["bucket"] == "all_features")].iloc[0]
    claim(
        cad["sharpe_mid_diff"].abs().idxmax() == up.name,
        "5.4: daily refits move XGBoost on all_features' trade the most",
    )
    m["cmCadShUp"] = s_(up["sharpe_mid_diff"])  # point (2026-10-06; was ci_)
    m["cmCadShUpT"] = s_(up["t_hac"])
    claim(up["t_p"] < SIG, "5.4 (was): that move is resolved (HAC t significant)")

    # ---- tuning: random search (H) and Optuna (C) against the shipped settings, refit daily
    rs = ht[
        (ht["family"] == "masked ladder")
        & (ht["a"] == "RS1 masked")
        & (ht["b"] == "T1 masked")
    ]
    assert len(rs) == n_tree * rs["bucket"].nunique()
    qb, qa = _below_above(rs, "qlike_diff")
    claim(
        qb == 0,
        f"5.4: random search is never more accurate than the shipped settings ({qb} below)",
    )
    m["cmRsQAbove"] = word(qa)
    m["cmRsQAboveSig"] = word(_below_above(rs, "qlike_diff", "dm_p")[1])
    m["cmRsN"] = str(len(rs))
    op = c[c["comparison"] == "vs T1"]
    assert len(op) == n_tree * n_tp * op["bucket"].nunique()
    qb, qa = _below_above(op, "qlike_diff")
    claim(
        qb == 0,
        f"5.4: Optuna is never more accurate than the shipped settings ({qb} below)",
    )
    m["cmOpN"] = str(len(op))
    m["cmOpQAbove"] = str(qa)
    m["cmOpQAboveSig"] = str(_below_above(op, "qlike_diff", "dm_p")[1])
    n_ex = sum(_below_above(rs, "sharpe_mid_diff", "t_p")) + sum(
        _below_above(op, "sharpe_mid_diff", "t_p")
    )
    claim(
        n_ex == 0,
        f"5.4: no tuned tree's trade differs significantly from the shipped one ({n_ex})",
    )
    tp = c[c["comparison"] == "tune_per vs 250"]
    assert len(tp) == n_tree * (n_tp - 1) * tp["bucket"].nunique()
    qb, qa = _below_above(tp, "qlike_diff")
    claim(qa == 0, f"5.4: tuning more often is never less accurate ({qa} above)")
    claim(
        sum(_below_above(tp, "sharpe_mid_diff", "t_p")) == 0,
        "5.4: ... and never significantly different on the trade (HAC t)",
    )
    m["cmTpN"] = str(len(tp))
    m["cmTpQBelow"] = word(qb)
    m["cmTpQBelowSig"] = word(_below_above(tp, "qlike_diff", "dm_p")[0])
    b50 = c[
        c["comparison"].str.startswith("budget") & c["a_kind"].isin(["tp1", "tp25"])
    ]
    assert (
        len(b50) == n_tree * n_tp * b50["bucket"].nunique()
    )  # (tp1, tp25) x (vs 10, vs 25)
    qb, qa = _below_above(b50, "qlike_diff")
    claim(
        qb == 0,
        f"5.4: the best of 50 trials is never more accurate than of 10 / 25 ({qb} below)",
    )
    m["cmBudN"] = str(len(b50))
    m["cmBudQAbove"] = word(qa)
    m["cmBudQAboveSig"] = word(_below_above(b50, "qlike_diff", "dm_p")[1])
    m["cmBudQBelow"] = word(
        qb
    )  # new: "best of 50 more accurate" (the TODO-PLAIN asks for it)
    # bp = camp.nonlive(pd.read_csv(C_BUDGET))  # validation MSE of the trials: no scorer in it
    # assert len(bp) == n_tree * nb and bp["share_best_in_41_50_if_exchangeable"].nunique() == 1
    # 2026-10-13 (user: the per-bar trees on their top entries): LightGBM's rows (agent C) + the forest's from the
    # rf400 stage-1 records; XGBoost's xgbtop records are not local, so it is left out (D.19 does the same)
    bp = camp.budget_points_paper()
    # assert len(bp) == 2 * nb and bp["share_best_in_41_50_if_exchangeable"].nunique() == 1  # 2026-10-13
    # 2026-10-14: XGBoost's rows from the xgbtop stage-1 records (pulled) -- every tree on its top entry again
    assert (
        len(bp) == n_tree * nb
        and bp["share_best_in_41_50_if_exchangeable"].nunique() == 1
    )
    assert set(bp["model"]) == set(camp.TREES), sorted(set(bp["model"]))
    exch = bp["share_best_in_41_50_if_exchangeable"].iloc[0]
    m["cmBudLastMin"] = f"{100 * bp['share_best_in_41_50'].min():.0f}"
    m["cmBudLastMax"] = f"{100 * bp['share_best_in_41_50'].max():.0f}"
    m["cmBudLastExch"] = f"{100 * exch:.0f}"
    assert bp["share_best_in_41_50"].min() > exch, (
        "claim: late trials win more than chance"  # validation scores: not scorer-dependent
    )

    # ---- LSTMs: daily refits (H), against the ridge (I), intraday vs per-bar (I)
    # 2026-10-06: LSTMs parked, user decision: cmLstmCad{Base,All}, cmLstmCadBelow(Sig), cmLstmVsRidgeMin / Max,
    # cmIntraVsRidgeMin / Max, cmIntraAllQ / DM (H's LSTM - LSTM10, I's pairs; the code is in
    # the 2026-10-06 plain version of this script and in make_appendix_campaign_16h_tex.py's re-score)
    for nm in (
        "cmLstmCadBase",
        "cmLstmCadAll",
        "cmLstmCadBelow",
        "cmLstmCadBelowSig",
        "cmLstmVsRidgeMin",
        "cmLstmVsRidgeMax",
        "cmIntraVsRidgeMin",
        "cmIntraVsRidgeMax",
        "cmIntraAllQ",
        "cmIntraAllDM",
    ):
        park(nm, "LSTM (LSTMs parked)")

    # ---- the best tree forecast of the model set on the trade (no LSTM in the model set)
    ms = model_set(d)
    nl = ms[ms["family"].str.contains("tree")]
    best = nl["Sharpe_mid"].idxmax()
    m["cmNlBestSh"] = f_(d.loc[best, "Sharpe_mid"])
    m["cmNlBestLabel"] = d.loc[best, "label"].split(" [")[0].replace("_", r"\_")
    m["cmNlBestInputs"] = tt(d.loc[best, "label"].split("[")[1].split("]")[0])
    m["cmNlBestVsRef"] = s_(d.loc[best, "dSharpe_mid_vs_ref"])
    m["cmNlBestVsRefT"] = s_(d.loc[best, "t_hac_mid_vs_ref"])
    _counts(m, "cmNlVsRef", nl["dSharpe_mid_vs_ref"], nl["t_hac_mid_vs_ref"])
    # 2026-10-06: no headline model -- the best tree / LSTM against the headline and the count of
    # trees / LSTMs above it are parked (master_table_vs_headline.csv)
    for nm in ("cmNlBestVsH", "cmNlBestVsHT", "cmNlVsHAbove", "cmNlVsHAboveSig"):
        park(nm, "against the former headline (no headline model)")

    # ---- CPU class: every canonical chunk on one class; bit-identical within it (H, C)
    cen = pd.read_csv(H_CLASS / "class_census.csv")
    fleet = cen[
        (cen["root"] != "xeon")
        & ~cen["root"].str.startswith("lstm")  # 2026-10-06: LSTMs parked, user decision
    ]  # the xeon rows are H's class gate, not the fleet
    assert (fleet["class"] == CAMPAIGN_CLASS).all()
    hosts = pd.read_csv(C_ROOT / "chunk_hosts.csv")
    assert (hosts["cpu_class"] == CAMPAIGN_CLASS).all()
    m["cmClassChunks"] = str(int(fleet["chunks"].sum()) + len(hosts))
    same = pd.read_csv(C_ROOT / "xclass_epyc" / "xclass_gate.csv").set_index("model")
    cross = pd.read_csv(C_ROOT / "xclass" / "xclass_gate.csv").set_index("model")
    assert same["records_bit_identical"].all() and same["points"].nunique() == 1
    assert (same["points_trial1_identical"] == same["points"]).all()
    m["cmClassPts"] = str(int(same["points"].iloc[0]))
    assert (cross["points"] == same["points"]).all()
    m["cmClassLgbmPick"] = str(int(cross.loc["lgbm", "best_mse_k50_differs"]))
    assert int(cross.loc["lgbm", "best_mse_k50_differs"]) > 0
    assert int(cross.loc["xgb", "best_mse_k50_differs"]) == 0
    assert int(cross.loc["rf", "best_mse_k50_differs"]) == 0
    arms = pd.read_csv(H_CLASS / "class_arms.csv").set_index("rung")
    assert not arms["bit_identical"].any()
    m["cmClassMaxRelMin"] = f"{100 * arms['max_rel'].min():.1f}"
    m["cmClassMaxRelMax"] = f"{100 * arms['max_rel'].max():.1f}"
    return m


# --------------------------------------------------------------------------- where the P&L comes from
HEAD_SERIES = "sign(s): per-bar ridge (live-feasible)"  # the headline in the B1 CSVs
SHORT_SERIES = "always short"
PNL_CELLS = [  # (csv, cell label in the CSV, label in the table)
    ("calendar", "month-end (last session)", "month-end, last session"),
    ("calendar", "month-end T-1", "month-end, the session before"),
    ("calendar", "month-end T+1", "month-end, the session after"),
    (
        "calendar",
        "not within one session of a month-end",
        "more than one session from a month-end",
    ),
    ("calendar", "FOMC statement day", "FOMC statement day"),
    ("regime", "VIX tercile: low", "previous VIX close: low tercile"),
    ("regime", "VIX tercile: mid", "previous VIX close: middle tercile"),
    ("regime", "VIX tercile: high", "previous VIX close: high tercile"),
    ("calendar", "year 2020", "2020"),
    ("calendar", "year 2021", "2021"),
    ("calendar", "year 2022", "2022"),
    ("calendar", "year 2023", "2023"),
    ("calendar", "year 2024", "2024 (to 30 April)"),
]


def _pnl_frames() -> dict[str, pd.DataFrame]:
    return {
        "calendar": pd.read_csv(PNL_RS / "calendar_cells.csv"),
        "regime": pd.read_csv(PNL_RS / "regime_cells.csv"),
        "diff": pd.read_csv(PNL_RS / "diff_attribution.csv"),
        "tail": pd.read_csv(PNL_RS / "tail_concentration.csv"),
        "hit": pd.read_csv(PNL_RS / "hit_payoff.csv"),
    }


def _cell(df: pd.DataFrame, series: str, cell: str, fill: str = "mid") -> pd.Series:
    x = df[(df["series"] == series) & (df["fill"] == fill) & (df["cell"] == cell)]
    assert len(x) == 1, (series, cell, fill, len(x))
    return x.iloc[0]


def _diff(df: pd.DataFrame, cell: str, fill: str = "mid") -> pd.Series:
    x = df[
        (df["sign_s_series"] == HEAD_SERIES)
        & (df["minus"] == SHORT_SERIES)
        & (df["fill"] == fill)
        & (df["cell"] == cell)
    ]
    assert len(x) == 1, (cell, fill, len(x))
    return x.iloc[0]


def _sum(r: pd.Series, nd: int = 1) -> str:
    """A cell's summed P&L, point estimate (2026-10-06; was _sum_ci with its interval)."""
    return s_(r["sum"], nd)


def pnl_macros(d: pd.DataFrame) -> dict[str, str]:
    """The headline's P&L decomposition on the plain forecast (B1, research_scorer/; 2026-10-06)."""
    f = _pnl_frames()
    m: dict[str, str] = {}
    # gate: the decomposition's headline series is Table 7's headline row (plain forecast)
    h = d.loc[HEADLINE]
    for fill, col in (("mid", "Sharpe_mid"), ("crossed", "Sharpe_crossed")):
        a = _cell(f["hit"], HEAD_SERIES, "all days", fill)
        assert abs(a["Sharpe_ann"] - h[col]) < CSV_TOL and int(a["n_buy"]) == int(
            h["n_buy"]
        ), fill
    t = f["tail"]
    tm = t[(t["series"] == HEAD_SERIES) & (t["fill"] == "mid")].set_index("k")
    tx = t[(t["series"] == HEAD_SERIES) & (t["fill"] == "crossed")].set_index("k")
    m["cmPnlTotal"] = f_(tm.loc[20, "total"], 1)
    m["cmPnlTotalX"] = f_(tx.loc[20, "total"], 1)
    m["cmPnlTopTenShare"] = f"{tm.loc[10, 'top_k_share_pct']:.0f}"
    m["cmPnlTopTwentyShare"] = f"{tm.loc[20, 'top_k_share_pct']:.0f}"
    m["cmPnlTopTwentyShareX"] = f"{tx.loc[20, 'top_k_share_pct']:.0f}"
    claim(
        int(tm.loc[20, "top_k_n_buy"]) == 20 and int(tm.loc[20, "bottom_k_n_buy"]) == 0,
        "5.4 / D: the 20 best days are all buys and the 20 worst all sells",
    )
    m["cmPnlExTwenty"] = s_(tm.loc[20, "total_ex_top_k"], 1)
    m["cmPnlExTwentySh"] = f_(tm.loc[20, "Sharpe_ex_top_k"])
    m["cmPnlMktTwenty"] = str(int(tm.loc[20, "mkt_top_k_bought"]))
    m["cmPnlMktTwentyExp"] = f_(tm.loc[20, "mkt_top_k_bought_expected"], 1)
    m["cmPnlMktTwentyP"] = f"{tm.loc[20, 'mkt_top_k_bought_p_hypergeom']:.3f}"
    claim(
        tm.loc[20, "mkt_top_k_bought"] > tm.loc[20, "mkt_top_k_bought_expected"],
        "D: the rule owns more of the straddle's largest payoffs than its buy rate gives",
    )
    hall = _cell(f["hit"], HEAD_SERIES, "all days")
    m["cmPnlDaysAll"] = str(int(hall["days_for_100pct_of_pnl"]))
    m["cmPnlDaysAllPct"] = f_(hall["pct_days_for_100pct_of_pnl"], 1)
    me = _cell(f["calendar"], HEAD_SERIES, "month-end (last session)")
    me_s = _cell(f["calendar"], SHORT_SERIES, "month-end (last session)")
    far = _cell(f["calendar"], HEAD_SERIES, "not within one session of a month-end")
    m["cmPnlMEn"] = str(int(me["n"]))
    m["cmPnlMEbuys"] = str(int(me["n_buy"]))
    m["cmPnlME"] = _sum(me)
    m["cmPnlMEShort"] = _sum(me_s)
    m["cmPnlNotMEn"] = str(int(far["n"]))
    m["cmPnlNotME"] = _sum(far)
    # 2026-10-06: was "me's interval includes 0 and far's lies above 0" (intervals removed)
    claim(
        far["sum"] > 0 and far["sum"] > me["sum"],
        "5.4 / D: month-ends are not the source (cmPnlNotME > cmPnlME, > 0)",
    )
    claim(
        me_s["sum"] < 0,
        "5.4 / D: the long straddle pays on month-ends on average: always short loses there "
        "(cmPnlMEShort < 0)",
    )
    dall = _diff(f["diff"], "all days")
    dx = _diff(f["diff"], "all days", "crossed")
    buy = _diff(f["diff"], "buy days")
    m["cmPnlDiffBuyShare"] = f"{buy['share_of_total_pct']:.0f}"
    assert abs(buy["sum"] - dall["sum"]) < CSV_TOL, (
        "all of the difference is on buy days"  # an identity: on sell days both are short
    )
    m["cmPnlDiff"] = _sum(dall)
    m["cmPnlDiffX"] = _sum(dx)
    claim(dall["sum"] > 0, "5.4 / 2: the difference to always short is positive (mid)")
    ten = _diff(f["diff"], "top 10 days of the difference")
    m["cmPnlDiffExTen"] = s_(ten["total_ex_cell"], 1)  # point (2026-10-06; was ci_)
    claim(
        ten["total_ex_cell"] < TAIL_SHARE * dall["sum"],
        "5.4 / D: a tail trade -- without its 10 best days the difference keeps less than "
        f"{TAIL_SHARE:.0%} of itself (cmPnlDiffExTen against cmPnlDiff)",
    )
    hv = _diff(f["diff"], "VIX tercile: high")
    m["cmPnlDiffHighVix"] = _sum(hv)
    m["cmPnlHighVixN"] = str(int(hv["n"]))
    return m


def write_pnl_table() -> None:
    f = _pnl_frames()
    lines = [
        "% AUTO-GENERATED by writeup/make_table_close_main.py -- do not edit.",
        "% Source: results/close_pnl_decomp/research_scorer/{calendar_cells,regime_cells,",
        "% diff_attribution,tail_concentration}.csv (experiments/close_pnl_decomposition.py, the",
        "% plain forecast f^2 B; midpoint fills; point estimates -- 2026-10-06: recalibration and",
        "% intervals removed, user decision).",
        r"\begingroup\small\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"cell & days & buys & $\mathrm{sign}(s)$, headline & always short & difference \\",
        r"\midrule",
    ]
    head = _cell(f["hit"], HEAD_SERIES, "all days")
    short = _cell(f["hit"], SHORT_SERIES, "all days")
    dall = _diff(f["diff"], "all days")
    lines.append(
        " & ".join(
            [
                "all days",
                str(int(head["n"])),
                str(int(head["n_buy"])),
                _sum(head),
                _sum(short),
                _sum(dall),
            ]
        )
        + r" \\"
    )
    lines.append(r"\addlinespace")
    for src, cell, label in PNL_CELLS:
        a = _cell(f[src], HEAD_SERIES, cell)
        b = _cell(f[src], SHORT_SERIES, cell)
        c = _diff(f["diff"], cell)
        assert int(a["n"]) == int(b["n"]) == int(c["n"]), cell
        lines.append(
            " & ".join(
                [
                    label,
                    str(int(a["n"])),
                    str(int(a["n_buy"])),
                    _sum(a),
                    _sum(b),
                    _sum(c),
                ]
            )
            + r" \\"
        )
        if cell in ("FOMC statement day", "VIX tercile: high"):
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\endgroup", ""]
    PNL_OUT.write_text("\n".join(lines), encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- strategy variations (F1)
SV_RULE = "sign(s), per-bar ridge (live-feasible), research scorer"  # the headline in F1's CSVs
SV_FRAME = "per straddle premium"
WING_FAMILIES = ("iron butterfly", "straddle + wing hedge")  # the straddle plus wings


def _sv_returns(dv: pd.DataFrame, q: np.ndarray) -> dict[str, np.ndarray]:
    """One variation's daily return per straddle premium under the positions q (+1 buy).

    experiments/strategy_variations_1530.py score(), its lines for the frame "per straddle
    premium": buy L+ at mid / at the ask side, sell L- at mid / at the bid side, settled at the
    close, over the straddle's mid / ask / bid premium.
    """
    buy = q > 0
    pnl_mid = np.where(
        buy, dv["payoff_plus"] - dv["mid_plus"], dv["mid_minus"] - dv["payoff_minus"]
    )
    pnl_x = np.where(
        buy,
        dv["payoff_plus"] - dv["cost_buy_plus"],
        dv["proceeds_sell_minus"] - dv["payoff_minus"],
    )
    return {
        "mid": pnl_mid / dv["strad_mid"].to_numpy(float),
        "crossed": pnl_x / np.where(buy, dv["strad_ask"], dv["strad_bid"]),
    }


def sv_rescore(d: pd.DataFrame) -> pd.DataFrame:
    """Every strategy variation minus the straddle under the PLAIN headline (2026-10-06).

    F1 (experiments/strategy_variations_1530.py) scored the variations with the research
    scorer's recalibrated headline; its daily legs (variations_daily.parquet: entry mid / ask
    / bid sides and settlement of each structure, no option chain needed) are re-scored here
    with the headline's plain positions q = sign(plain - iv_var) from the master table's daily
    frame.  Gates: rebuilt with the recalibrated positions (the daily frame's diagnostic
    pred_clock) and with always short, the returns equal F1's stored ones; the straddle under
    the plain positions has Table 7's headline Sharpe ratios.
    """
    import atm_straddle_lib as asl
    import strategy_variations_1530 as svm  # its sharpe (the annualization of F1)

    daily = pd.read_parquet(DAILY)
    hd = daily[daily["key"] == HEADLINE].set_index("day").sort_index()
    assert len(hd) == N_DAYS
    q_plain = pd.Series(np.where(hd["plain"] > hd["iv_var"], 1.0, -1.0), index=hd.index)
    assert (q_plain.to_numpy() == hd["q"].to_numpy()).all(), (
        "the daily frame's position"
    )
    # q_recal = np.where(pred_clock > iv_var, 1, -1): the gate before F1 was re-run on plain (parked)
    q_short = pd.Series(-1.0, index=hd.index)
    vd = pd.read_parquet(SV / "variations_daily.parquet")
    vd["day"] = pd.to_datetime(vd["day"])
    stored = pd.read_parquet(SV / "returns_daily_per_straddle_premium.parquet")
    fam = vd.groupby("variation", sort=False)["family"].first()
    rets: dict[tuple[str, str], pd.Series] = {}
    worst = 0.0
    for v in fam.index:
        dv = vd[vd["variation"] == v].set_index("day").sort_index()
        # F1 was re-run on the plain forecast (2026-10-06, experiments/strategy_variations_1530.py):
        # its stored research-rule returns are the plain positions (were q_recal before)
        for rule, q in ((SV_RULE, q_plain), ("always short", q_short)):
            r = _sv_returns(dv, q.reindex(dv.index).to_numpy(float))
            for fill in ("mid", "crossed"):
                col = f"{v}|{rule}|{fill}|{SV_FRAME}"
                s = stored[col].reindex(dv.index).to_numpy(float)
                ok = np.isfinite(s) & np.isfinite(r[fill])
                assert (np.isfinite(s) == np.isfinite(r[fill])).all(), col
                worst = max(
                    worst, float(np.max(np.abs(s[ok] - r[fill][ok]), initial=0.0))
                )
        r = _sv_returns(dv, q_plain.reindex(dv.index).to_numpy(float))
        for fill in ("mid", "crossed"):
            rets[(v, fill)] = pd.Series(r[fill], index=dv.index)
    assert worst <= RECON_TOL, ("rebuilt F1 returns differ from the stored ones", worst)
    h = d.loc[HEADLINE]
    st_mid = svm.sharpe(rets[("straddle", "mid")].to_numpy())
    st_x = svm.sharpe(rets[("straddle", "crossed")].to_numpy())
    assert abs(st_mid - h["Sharpe_mid"]) < GATE_TOL, (
        "straddle row = Table 7 (mid)",
        st_mid,
    )
    # F1 prices the crossed fill leg by leg, the master table from the summed quotes: the two
    # agree to rounding, so this gate uses FILL_TOL rather than GATE_TOL
    assert abs(st_x - h["Sharpe_crossed"]) < FILL_TOL, ("(crossed)", st_x)
    rows = []
    for v in fam.index:
        if v == "straddle":
            continue
        for fill in ("mid", "crossed"):
            j = pd.concat(
                [rets[(v, fill)], rets[("straddle", fill)]], axis=1, keys=["v", "b"]
            ).dropna()
            a_, b_ = j["v"].to_numpy(), j["b"].to_numpy()
            t = asl.newey_west_t(a_ - b_)[0]
            rows.append(
                {
                    "variation": v,
                    "family": fam[v],
                    "fill": fill,
                    "n": len(j),
                    "Sharpe_variation": svm.sharpe(a_),
                    "Sharpe_straddle_same_days": svm.sharpe(b_),
                    "dSharpe": svm.sharpe(a_) - svm.sharpe(b_),
                    "t_hac_dmean": t,
                    "t_p": hac_p(t),
                }
            )
    p = pd.DataFrame(rows)
    old = pd.read_csv(SV / "paired_vs_straddle.csv")
    old = old[(old["rule"] == SV_RULE) & (old["frame"] == SV_FRAME)].set_index(
        ["variation", "fill"]
    )
    x = p.set_index(["variation", "fill"]).join(old[["dSharpe"]], rsuffix="_recal")
    print(
        f"strategy variations re-scored on the plain headline: {len(p)} pairs; dSharpe sign "
        f"flipped by the scorer change in {int((np.sign(x['dSharpe']) != np.sign(x['dSharpe_recal'])).sum())}"
    )
    return p


def sv_macros(d: pd.DataFrame) -> dict[str, str]:
    """Strategy variations under the headline forecast, plain (2026-10-06; see sv_rescore)."""
    m: dict[str, str] = {}
    p = sv_rescore(d)
    m["cmSvN"] = str(p["variation"].nunique())
    above = p["dSharpe"] > 0
    m["cmSvAbove"] = word(
        int(above.sum())
    )  # point estimates, both fills (was: intervals)
    m["cmSvAboveSig"] = word(int((above & (p["t_p"] < SIG)).sum()))
    claim(
        int(above.sum()) == 0,
        "5.4: no variation beats the straddle (point, both fills)",
    )
    mid = p[p["fill"] == "mid"].set_index("variation")
    best = mid["dSharpe"].idxmax()
    claim(
        best == "iron butterfly, w50",
        f"5.4: the iron butterfly with the widest wings is closest at mid ({best})",
    )
    b = mid.loc[best]
    m["cmSvBestMidSh"] = f_(b["Sharpe_variation"])
    m["cmSvBestMid"] = s_(b["dSharpe"])  # point (2026-10-06; was ci_)
    m["cmSvBestMidT"] = s_(b["t_hac_dmean"])
    crossed = p[p["fill"] == "crossed"]
    claim(
        (crossed["dSharpe"] < 0).all(), "5.4: at crossed fills the straddle ranks first"
    )
    for fill, tag in (("mid", "Mid"), ("crossed", "X")):
        w = p[(p["fill"] == fill) & p["family"].isin(WING_FAMILIES)]
        m["cmSvWingN"] = str(len(w))
        m[f"cmSvWing{tag}"] = str(
            int((w["dSharpe"] < 0).sum())
        )  # point (was: intervals)
        m[f"cmSvWing{tag}Sig"] = str(int(((w["dSharpe"] < 0) & (w["t_p"] < SIG)).sum()))
    claim(
        m["cmSvWingX"] == m["cmSvWingN"],
        "5.4: wings cost at every width at crossed fills",
    )
    # the bid-ask cost line does not depend on the forecast (F1's cost_line.csv)
    cost = pd.read_csv(SV / "cost_line.csv").set_index("variation")
    share = (
        cost["median_cost_buy_share_of_own_premium"]
        + cost["median_cost_sell_share_of_own_premium"]
    ) / 2.0
    m["cmSvCostStraddle"] = f_(100 * share.loc["straddle"], 1)
    ibf = share[cost["family"] == "iron butterfly"]
    m["cmSvCostIbfMin"] = f_(100 * ibf.min(), 1)
    m["cmSvCostIbfMax"] = f_(100 * ibf.max(), 1)
    return m


# --------------------------------------------------------------------------- the model set's tail
TAIL_K = (
    20  # the best / worst days of the P&L decomposition's tail reading (its k = 20)
)
TAIL_TEN = (
    10  # the best days of the difference to always short (its "top 10 days" cell)
)


def _tail_row(g: pd.DataFrame, short: np.ndarray) -> dict[str, float]:
    """One forecast's tail: the P&L decomposition's definitions (tail_concentration.csv k = 20,
    diff_attribution.csv "top 10 days of the difference"), midpoint fill, premium units."""
    x, q = g["ret_mid"].to_numpy(), g["q"].to_numpy()
    tot = float(x.sum())
    o = np.argsort(-x, kind="stable")
    diff = x - short
    dd = np.sort(diff)[::-1]
    return {
        "total": tot,
        "top_share": 100.0 * float(x[o[:TAIL_K]].sum()) / tot if tot > 0 else np.nan,
        "top_all_buys": bool((q[o[:TAIL_K]] > 0).all()),
        "bottom_all_sells": bool((q[o[-TAIL_K:]] < 0).all()),
        "diff": float(diff.sum()),
        "diff_ex_ten": float(diff.sum() - dd[:TAIL_TEN].sum()),
    }


def tail_macros(d: pd.DataFrame, ms: pd.DataFrame) -> dict[str, str]:
    """How concentrated the trade's P&L is, over the model set (2026-10-06; the headline's tail
    reading of the P&L decomposition, applied to every forecast).  Gate: the same functions on the
    former headline reproduce the P&L decomposition's numbers."""
    m: dict[str, str] = {}
    deck = camp._engine()["deck"]
    daily = pd.read_parquet(DAILY)
    short = -deck["R"].to_numpy(float)

    def row(k: str) -> dict[str, float]:
        g = daily[daily["key"] == k].set_index("day").sort_index()
        assert len(g) == N_DAYS and (g.index == deck.index).all(), k
        return _tail_row(g, short)

    # gate: the former headline's tail = the P&L decomposition's (six significant digits)
    f = _pnl_frames()
    t = f["tail"]
    tm = t[(t["series"] == HEAD_SERIES) & (t["fill"] == "mid")].set_index("k")
    ten = _diff(f["diff"], "top 10 days of the difference")
    dall = _diff(f["diff"], "all days")
    h = row(HEADLINE)
    for mine, theirs in (
        (h["total"], tm.loc[TAIL_K, "total"]),
        (h["top_share"], tm.loc[TAIL_K, "top_k_share_pct"]),
        (h["diff"], dall["sum"]),
        (h["diff_ex_ten"], ten["total_ex_cell"]),
    ):
        assert abs(mine - theirs) <= CSV_TOL * max(1.0, abs(theirs)), (mine, theirs)
    tl = pd.DataFrame({k: row(k) for k in ms.index}).T
    pos = tl[tl["total"] > 0]
    m["cmMasterPosTotalN"] = str(len(pos))
    m["cmMasterTopTwentyShareMed"] = f"{pos['top_share'].median():.0f}"
    m["cmMasterTopTwentyShareMin"] = f"{pos['top_share'].min():.0f}"
    m["cmMasterTopTwentyShareMax"] = f"{pos['top_share'].max():.0f}"
    m["cmMasterTopTwentyBuysN"] = str(int(tl["top_all_buys"].sum()))
    m["cmMasterBottomTwentySellsN"] = str(int(tl["bottom_all_sells"].sum()))
    m["cmMasterDiffMed"] = s_(tl["diff"].median(), 1)
    m["cmMasterDiffPosN"] = str(int((tl["diff"] > 0).sum()))
    m["cmMasterDiffExTenMed"] = s_(tl["diff_ex_ten"].median(), 1)
    dp = tl[tl["diff"] > 0]
    m["cmMasterDiffExTenHalfN"] = str(
        int((dp["diff_ex_ten"] < TAIL_SHARE * dp["diff"]).sum())
    )
    m["cmMasterDiffExTenNegN"] = str(int((tl["diff_ex_ten"] < 0).sum()))
    claim(
        int(tl["top_all_buys"].sum()) == len(tl),
        "5.4: for every forecast the 20 best days are buys (the tail-trade reading)",
    )
    return m


# --------------------------------------------------------------------------- strategy variations, model set
def sv_set_macros(d: pd.DataFrame, ms: pd.DataFrame) -> dict[str, str]:
    """Every strategy variation minus the straddle under every forecast of the model set.

    2026-10-06 (no headline model): F1's daily legs (variations_daily.parquet) re-scored with each
    forecast's plain positions q = sign(plain - iv_var) from the master table's daily frame; per
    straddle premium, mid and crossed fills (F1's arithmetic, _sv_returns); point estimates and
    the HAC t of the daily mid-return difference.  No option chain is read.  Gates: the straddle
    under each forecast's positions has that forecast's master-table Sharpe ratios; F1's stored
    returns are rebuilt exactly (sv_rescore, run before this).
    """
    import atm_straddle_lib as asl
    import strategy_variations_1530 as svm

    m: dict[str, str] = {}
    vd = pd.read_parquet(SV / "variations_daily.parquet")
    vd["day"] = pd.to_datetime(vd["day"])
    fam = vd.groupby("variation", sort=False)["family"].first()
    daily = pd.read_parquet(DAILY)
    Q = daily[daily["key"].isin(ms.index)].pivot(index="day", columns="key", values="q")
    Q = Q[list(ms.index)].sort_index()
    assert Q.shape == (N_DAYS, len(ms)) and Q.notna().all().all()

    def rets(v: str) -> tuple[pd.Index, dict[str, np.ndarray]]:
        dv = vd[vd["variation"] == v].set_index("day").sort_index()
        qv = Q.reindex(dv.index).to_numpy(float)
        buy = qv > 0
        col = {
            c: dv[c].to_numpy(float)[:, None]
            for c in dv.columns
            if dv[c].dtype.kind == "f"
        }
        mid = np.where(
            buy,
            col["payoff_plus"] - col["mid_plus"],
            col["mid_minus"] - col["payoff_minus"],
        )
        xx = np.where(
            buy,
            col["payoff_plus"] - col["cost_buy_plus"],
            col["proceeds_sell_minus"] - col["payoff_minus"],
        )
        return dv.index, {
            "mid": mid / col["strad_mid"],
            "crossed": xx / np.where(buy, col["strad_ask"], col["strad_bid"]),
        }

    def sharpe_cols(x: np.ndarray) -> np.ndarray:
        sd = x.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(sd > 0, x.mean(axis=0) / sd * svm.ANN, np.nan)

    sdays, st = rets("straddle")
    assert (sdays == Q.index).all()
    # gate: the straddle under each forecast's positions is that forecast's master-table row
    gm = np.abs(sharpe_cols(st["mid"]) - ms["Sharpe_mid"].to_numpy()).max()
    gx = np.abs(sharpe_cols(st["crossed"]) - ms["Sharpe_crossed"].to_numpy()).max()
    assert gm < GATE_TOL and gx < FILL_TOL, (gm, gx)
    rows = []
    for v in fam.index:
        if v == "straddle":
            continue
        dd, r = rets(v)
        on = Q.index.get_indexer(dd)
        for fill in ("mid", "crossed"):
            a = r[fill]
            b = st[fill][on]
            ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
            a, b = a[ok], b[ok]
            dsh = sharpe_cols(a) - sharpe_cols(b)
            tt_ = [asl.newey_west_t(a[:, j] - b[:, j])[0] for j in range(a.shape[1])]
            for j, k in enumerate(ms.index):
                rows.append(
                    {
                        "variation": v,
                        "family": fam[v],
                        "fill": fill,
                        "key": k,
                        "dSharpe": dsh[j],
                        "t": tt_[j],
                        "t_p": hac_p(tt_[j]),
                    }
                )
    p = pd.DataFrame(rows)
    nv, nf = p["variation"].nunique(), len(ms)
    m["cmSvN"] = str(nv)
    m["cmSvSetN"] = str(nf)
    m["cmSvSetPairsN"] = str(nv * nf)
    for fill, tag in (("mid", "Mid"), ("crossed", "X")):
        x = p[p["fill"] == fill]
        up = x["dSharpe"] > 0
        m[f"cmSvSetAbove{tag}"] = str(int(up.sum()))
        m[f"cmSvSetAbove{tag}Sig"] = str(int((up & (x["t_p"] < SIG)).sum()))
        first = x.groupby("key")["dSharpe"].max() < 0
        m[f"cmSvSetStraddleFirst{tag}"] = str(int(first.sum()))
        w = x[x["family"].isin(WING_FAMILIES)]
        m["cmSvSetWingN"] = str(len(w))
        m[f"cmSvSetWing{tag}Below"] = str(int((w["dSharpe"] < 0).sum()))
    xm = p[p["fill"] == "mid"]
    med = xm.groupby("variation")["dSharpe"].median()
    best = med.idxmax()
    m["cmSvSetBestMidLabel"] = best.replace("%", r"\%").replace("_", r"\_")
    m["cmSvSetBestMidMed"] = s_(med[best])
    bb = xm[xm["variation"] == best]
    m["cmSvSetBestMidAboveN"] = str(int((bb["dSharpe"] > 0).sum()))
    m["cmSvSetBestMidAboveSig"] = str(
        int(((bb["dSharpe"] > 0) & (bb["t_p"] < SIG)).sum())
    )
    claim(
        m["cmSvSetStraddleFirstX"] == str(nf),
        "5.4: at crossed fills the straddle ranks first for every forecast of the model set",
    )
    claim(
        m["cmSvSetWingXBelow"] == m["cmSvSetWingN"],
        "5.4: wings cost at every width at crossed fills, for every forecast",
    )
    # the bid-ask cost line does not depend on the forecast (F1's cost_line.csv)
    cost = pd.read_csv(SV / "cost_line.csv").set_index("variation")
    share = (
        cost["median_cost_buy_share_of_own_premium"]
        + cost["median_cost_sell_share_of_own_premium"]
    ) / 2.0
    m["cmSvCostStraddle"] = f_(100 * share.loc["straddle"], 1)
    ibf = share[cost["family"] == "iron butterfly"]
    m["cmSvCostIbfMin"] = f_(100 * ibf.min(), 1)
    m["cmSvCostIbfMax"] = f_(100 * ibf.max(), 1)
    # 2026-10-06: no headline model (user decision) -- the former headline's variations are parked
    # (sv_rescore still runs, as the gate on the rebuilt returns)
    for nm in (
        "cmSvAbove",
        "cmSvAboveSig",
        "cmSvBestMid",
        "cmSvBestMidSh",
        "cmSvBestMidT",
        "cmSvWingN",
        "cmSvWingMid",
        "cmSvWingX",
        "cmSvWingMidSig",
        "cmSvWingXSig",
    ):
        park(nm, "strategy variations under the former headline (no headline model)")
    return m


# --------------------------------------------------------------------------- two robustness bounds
BOUNDS = (
    RES / "close_bounds_plain"
)  # experiments/close_bounds_plain.py (2026-10-06, plain, model set)
TEXT_B_BOUND = (
    0.1  # the bound the text stated for the B contamination ("under a tenth")
)
TEXT_CENS_BOUND = 0.06  # the bound the text stated for dropping the censored-IV days


def bounds_macros(ms: pd.DataFrame) -> dict[str, str]:
    """The B-contamination audit and the censored-IV bound of Section 4.5, over the model set.

    results/close_bounds_plain/{b_contamination,censored_iv}.csv: one row per model-set
    forecast, the plain forecast's sign(s) Sharpe ratio on all days and with the early-close B
    repaired / with the censored-IV days dropped (experiments/close_bounds_plain.py).  Gates:
    the same forecasts as the model set, and the all-days Sharpe ratios are the master table's.
    """
    m: dict[str, str] = {}
    paper = [k for k, _, _ in PAPER]
    for f, tag in (("b_contamination.csv", "BContam"), ("censored_iv.csv", "CensIv")):
        b = pd.read_csv(BOUNDS / f).set_index("key")
        assert set(ms.index) <= set(b.index), (f, "the model set")
        b = b.loc[
            ms.index
        ]  # 2026-10-06: LSTMs parked, user decision (the bounds' CSVs also hold the LSTMs)
        assert (
            b["Sharpe_mid"] - ms.loc[b.index, "Sharpe_mid"]
        ).abs().max() < GATE_TOL, f
        for sub, s in (("", b), ("Paper", b.loc[paper])):
            m[f"cm{tag}{sub}MaxDSh"] = f_(s["dSharpe_mid"].abs().max())
            m[f"cm{tag}{sub}MaxDShX"] = f_(s["dSharpe_crossed"].abs().max())
            m[f"cm{tag}{sub}MedDSh"] = f_(s["dSharpe_mid"].abs().median())
        m[f"cm{tag}Days"] = str(int(b["n_days_affected"].iloc[0]))
        assert b["n_days_affected"].nunique() == 1
    b = pd.read_csv(BOUNDS / "b_contamination.csv").set_index("key").loc[ms.index]
    m["cmBContamMaxFlips"] = str(int(b["positions_flipped"].max()))
    m["cmBContamMedFlips"] = f"{b['positions_flipped'].median():.0f}"
    m["cmBContamPaperMaxFlips"] = str(int(b.loc[paper, "positions_flipped"].max()))
    m["cmBContamMaxRel"] = f"{100 * b['max_rel_forecast_change'].max():.0f}"
    claim(
        b["dSharpe_mid"].abs().max() < TEXT_B_BOUND,
        f"4.2: the B contamination moves no Sharpe ratio by {TEXT_B_BOUND} (model set)",
    )
    c = pd.read_csv(BOUNDS / "censored_iv.csv").set_index("key").loc[ms.index]
    claim(
        c["dSharpe_mid"].abs().max() <= TEXT_CENS_BOUND,
        f"4.5: dropping the censored days moves no Sharpe ratio by more than {TEXT_CENS_BOUND}",
    )
    return m


# --------------------------------------------------------------------------- the pre-tuned linear models
PRETUNE_FACTS = (
    RES / "linear_pretune_2026-10-06" / "paper_facts.csv"
)  # experiments/stack_pretune_yhat.py
PT_EST = {"ridge": "Ridge", "lasso": "Lasso", "enet": "Enet"}
PT_SET = {
    "baseline": "Base",
    "all_features": "All",
}  # "lean_vol_vix": "Lean" -- 2026-10-08 parked (lean set; replaced by an MCS)


def _date(s: str) -> str:
    d = pd.Timestamp(s)
    return f"{d.day}~{d:%B}~{d.year}"


def _pow10(x: float) -> str:
    k = math.log10(float(x))
    assert abs(k - round(k)) < 1e-9, x  # every grid point is a power of ten
    return rf"\ensuremath{{10^{{{int(round(k))}}}}}"


def pretune_macros() -> dict[str, str]:
    """How the paper's per-bar linear models are tuned (2026-10-06, user decision: "use the
    pre-tuned version for all linear models"): arm (f) of results/linear_pretune_2026-10-06/,
    read from its paper_facts.csv (written from the study's own CSVs and constants)."""
    f = pd.read_csv(PRETUNE_FACTS).set_index("item")["value"]
    m: dict[str, str] = {}
    m["cmPtPreSessions"] = str(int(f["pre-2020 16:00 sessions in the design"]))
    m["cmPtPreStart"] = _date(f["first pre-2020 16:00 session"])
    m["cmPtValFirst"] = _date(f["pre-tune validation: first session"])
    m["cmPtValLast"] = _date(f["pre-tune validation: last session"])
    m["cmPtFolds"] = word(int(f["pre-tune folds"]))
    m["cmPtFoldLen"] = str(int(f["pre-tune fold length (sessions)"]))
    m["cmPtFitWin"] = f"{int(f['fit window (sessions)']):,}".replace(",", "{,}")
    m["cmPtEmbargo"] = str(int(f["embargo (sessions)"]))
    r = float(f["backbone penalty ratio (HAR + calendar / exogenous)"])
    m["cmPtRatio"] = f"1/{int(round(1 / r))}"
    m["cmPtBlock"] = str(int(f["retune: block length (sessions)"]))
    dates = [x.strip() for x in f["retune dates (block starts)"].split(";")]
    m["cmPtRetuneN"] = word(len(dates))
    m["cmPtRetuneFirst"] = _date(dates[0])
    m["cmPtRetuneLast"] = _date(dates[-1])
    fired_all, of_all = (int(x) for x in f["retunes fired, total"].split(" of "))
    # 2026-10-08 parked (lean set; replaced by an MCS): the totals count the paper's arms (PT_SET), not every arm of the study
    fired, of = 0, 0
    for est in PT_EST:
        for b in ("baseline", "all_features", "lean_vol_vix"):
            if b not in PT_SET:
                for nm in (f"cmPtFired{PT_EST[est]}Lean", f"cmPtPen{PT_EST[est]}Lean"):
                    park(nm, "2026-10-08 parked (lean set; replaced by an MCS)")
    n_arms = 0
    for est, et in PT_EST.items():
        for b, bt in PT_SET.items():
            fi, oo = (int(x) for x in f[f"{est} {b}: retunes fired"].split(" of "))
            assert oo == len(dates), (est, b)
            fired += fi
            of += oo
            m[f"cmPtFired{et}{bt}"] = word(fi)
            m[f"cmPtPen{et}{bt}"] = _pow10(f[f"{est} {b}: pre-tuned penalty"])
            n_arms += 1
    assert n_arms * len(dates) == of, "the total counts every paper arm's retunes"
    assert fired <= fired_all and of <= of_all
    m["cmPtFiredTotal"] = str(fired)
    m["cmPtFiredOf"] = str(of)
    claim(fired < of, "4.x: the margin rule lets few retunes fire")
    m.update(causal_rechoice_macros())
    return m


PRETUNE_SWITCHES = RES / "linear_pretune_2026-10-06" / "causal_switch_log.csv"
PRETUNE_SUMMARY = RES / "linear_pretune_2026-10-06" / "SUMMARY.md"


def causal_rechoice_macros() -> dict[str, str]:
    """Why the penalty is pre-tuned: the evidence of the causal re-choice (arms a and b of the
    pre-tune study, causal_switch_log.csv).  Re-choices are the block starts after the first
    choice; a switch is a re-choice that changed the penalty; the margin is the tree study's
    rule (HAC t of the paired squared-error difference); an edge pick sits at an end of the
    grid; flipped = trade-day positions that changed in the switched blocks, of their trade days.
    The study's own arms span all its input sets (\\cmPtCausal...); the ...Paper twins restrict
    them to the paper's three (2026-10-06)."""
    c = pd.read_csv(PRETUNE_SWITCHES)
    m: dict[str, str] = {}
    for tag, x in (("", c), ("Paper", c[c["bucket"].isin(camp.PAPER_SETS)])):
        re_ = x[x["block"] > 0]
        sw = re_[re_["switched"].astype(bool)]
        m[f"cmPtCausalSwitches{tag}"] = str(len(sw))
        m[f"cmPtCausalChoices{tag}"] = str(len(re_))
        m[f"cmPtCausalMarginPass{tag}"] = str(
            int(sw["passes_margin"].astype(bool).sum())
        )
        m[f"cmPtCausalEdgePicks{tag}"] = str(int(x["at_edge"].astype(bool).sum()))
        m[f"cmPtCausalPicks{tag}"] = str(len(x))
        assert sw["block_trade_days"].notna().all()
        flip = 100.0 * sw["block_position_changes"].sum() / sw["block_trade_days"].sum()
        m[f"cmPtSwitchFlipPct{tag}"] = f"{flip:.0f}"
    # gate: the study's SUMMARY states the same all-arm counts
    s = PRETUNE_SUMMARY.read_text(encoding="utf-8")
    want = (
        f"{m['cmPtCausalSwitches']} switches in {m['cmPtCausalChoices']} re-choices; "
        f"{m['cmPtCausalEdgePicks']} of {m['cmPtCausalPicks']} choices at a grid edge; "
        f"{m['cmPtCausalMarginPass']} of {m['cmPtCausalSwitches']} switches pass the margin rule"
    )
    assert want in s, ("the pre-tune SUMMARY states other counts", want)
    return m


# the \cmPnl... macros of D.14 (pnl_macros), parked 2026-10-06
PNL_MACROS = (
    "cmPnlTotal",
    "cmPnlTotalX",
    "cmPnlTopTenShare",
    "cmPnlTopTwentyShare",
    "cmPnlTopTwentyShareX",
    "cmPnlExTwenty",
    "cmPnlExTwentySh",
    "cmPnlMktTwenty",
    "cmPnlMktTwentyExp",
    "cmPnlMktTwentyP",
    "cmPnlDaysAll",
    "cmPnlDaysAllPct",
    "cmPnlMEn",
    "cmPnlMEbuys",
    "cmPnlME",
    "cmPnlMEShort",
    "cmPnlNotMEn",
    "cmPnlNotME",
    "cmPnlDiffBuyShare",
    "cmPnlDiff",
    "cmPnlDiffX",
    "cmPnlDiffExTen",
    "cmPnlDiffHighVix",
    "cmPnlHighVixN",
)
MODEL_SET_OUT = (
    RES / "close_master_table" / "paper_model_set.csv"
)  # read by the D.16-D.18 re-base


def write_model_set(ms: pd.DataFrame) -> None:
    """The exact paper model set (camp.paper_model_set) for the other makers (2026-10-06)."""
    out = pd.DataFrame(
        {
            "key": ms.index,
            "family": ms["family"].to_numpy(),
            "input_set": [camp.input_set(k) or "paper" for k in ms.index],
        }
    )
    assert out["key"].is_unique and len(out) == len(ms)
    out.to_csv(MODEL_SET_OUT, index=False, lineterminator="\n")
    print(f"wrote {MODEL_SET_OUT.relative_to(ROOT)} ({len(out)} forecasts)")


# 2026-10-23 (user, E32, option A): the ten-day tree rungs leave the model set for their every-session twins; the twin
# evidence is the tracked output of experiments/e32_cadence_gate.py (twins: every ten-day table against its twin;
# picks: the random-search tuning traces at both cadences)
E32_GATE = RES / "e32_cadence_gate"


def _int_tex(n: int) -> str:
    return f"{int(n):,}".replace(",", "{,}")


def e32_macros(d: pd.DataFrame, ms: pd.DataFrame) -> dict[str, str]:
    """The forecasts the every-session refit takes out of the model set, and the evidence that each has its twin in it.

    A forecast left out is a model-set forecast of a ten-day family (camp.TEN_DAY_FAMILIES); its gate row
    (results/e32_cadence_gate/twins.csv, the table the master table reads) must be ok -- same stamps, baseline and
    realized variance, the forecast bit for bit on every ten-day refit row -- and its twin must be in the model set.
    The random-search rungs' tuning traces (picks.csv) must agree in every column but the tuning time.
    """
    m: dict[str, str] = {}
    pre = camp.paper_model_set(d, keep_ten_day=True)
    drop = [k for k in pre.index if k not in ms.index]
    assert set(ms.index) <= set(pre.index)
    assert set(pre.loc[drop, "family"]) == set(camp.TEN_DAY_FAMILIES), sorted(
        set(pre.loc[drop, "family"])
    )
    assert not ms["family"].isin(camp.TEN_DAY_FAMILIES).any()
    tw = pd.read_csv(E32_GATE / "twins.csv")
    tw = tw[tw["read_by_master"].astype(bool)].set_index("key10")
    assert tw.index.is_unique
    missing = [k for k in drop if k not in tw.index]
    assert not missing, ("a forecast left out has no gate row", missing)
    x = tw.loc[drop]
    assert x["present"].astype(bool).all() and x["ok"].astype(bool).all()
    assert (x["refit_rows_bit_equal"] == x["refit_rows"]).all()
    assert x["key1"].isin(ms.index).all(), "a twin is not in the model set"
    for c in ("rows", "refit_rows", "other_rows"):
        assert x[c].nunique() == 1, c
    tuned = x[x["rung"].str.startswith("RS10")]
    pk = pd.read_csv(E32_GATE / "picks.csv").set_index(["token", "bucket"])
    tr = pk.loc[sorted(set(zip(tuned["token"], tuned["bucket"])))]
    assert tr["ok"].astype(bool).all() and (tr["columns_differing"] == "tune_sec").all()
    assert tr["tuning_points"].nunique() == 1
    m["cmTenDayN"] = str(len(drop))
    m["cmMasterNWithTenDay"] = str(len(pre))
    m["cmTenDayUntunedN"] = str(int((x["rung"] == "T10").sum()))
    m["cmTenDayTunedN"] = str(len(tuned))
    m["cmTenDayRows"] = _int_tex(x["rows"].iloc[0])
    m["cmTenDayRefitRows"] = str(int(x["refit_rows"].iloc[0]))
    m["cmTenDayOtherRows"] = _int_tex(x["other_rows"].iloc[0])
    m["cmTenDayTraces"] = str(len(tr))
    m["cmTenDayTunePoints"] = str(int(tr["tuning_points"].iloc[0]))
    print(
        f"E32: {len(drop)} ten-day forecasts out of the model set ({len(pre)} -> {len(ms)}); twins bit-equal on "
        f"{m['cmTenDayRefitRows']} of {x['rows'].iloc[0]} rows; {len(tr)} tuning traces identical"
    )
    return m


# --------------------------------------------------------------------------- the pre-tuned pooled models
POOL_STUDY = (
    RES / "pool_linear_pretune_2026-10-06"
)  # experiments/pool_linear_pretune.py (Hoffman2)
POOL_CLUSTER = (
    "h2"  # the run of record (Hoffman2); the CARC rerun is a reproduction check
)


def poolpt_macros(d: pd.DataFrame) -> dict[str, str]:
    """The pooled pre-tuned arm (f) against the causal one-penalty arm (a), reported honestly.

    2026-10-06 (user decision: one protocol for every linear model, decided before the pooled
    results).  Arm (a) is the stored pooled table (pool_*; the study reproduces it, gate in
    experiments/stack_pool_pretune_yhat.py), arm (f) the paper's poolpt_*: per est x input set,
    the QLIKE change in % (DM), the Sharpe change (HAC t), from the master table's daily frame.
    The causal re-choice evidence: the study's causal_switch_log.csv / retune_log.csv (Hoffman2).
    """
    import atm_straddle_lib as asl
    from src.evaluation.diebold_mariano import dm_test

    mtc = _mtc()
    daily = pd.read_parquet(DAILY)
    fr = {k: g.set_index("day").sort_index() for k, g in daily.groupby("key")}
    rows = []
    for est in ("ridge", "lasso", "enet"):
        for b in camp.PAPER_SETS:
            fk, ak = f"poolpt_{est}_{b}", f"pool_{est}_{b}"
            if fk not in fr or ak not in fr:
                continue
            f, a = fr[fk], fr[ak]
            assert len(f) == len(a) == N_DAYS and (f.index == a.index).all(), (fk, ak)
            dq = f["qlike_raw"] - a["qlike_raw"]
            t_ = asl.newey_west_t(f["ret_mid"].to_numpy() - a["ret_mid"].to_numpy())[0]
            rows.append(
                {
                    "bucket": b,
                    "q_pct": 100.0 * dq.mean() / a["qlike_raw"].mean(),
                    "dm": float(
                        dm_test(f["qlike_raw"].to_numpy(), a["qlike_raw"].to_numpy())[
                            "dm"
                        ]
                    ),
                    "dsh": mtc.sharpe(f["ret_mid"].to_numpy())
                    - mtc.sharpe(a["ret_mid"].to_numpy()),
                    "t": t_,
                }
            )
    x = pd.DataFrame(rows)
    m: dict[str, str] = {}
    m["cmPoolPtN"] = str(len(x))
    m["cmPoolPtQPctMean"] = s_(x["q_pct"].mean(), 1)
    m["cmPoolPtQWorse"] = str(int((x["q_pct"] > 0).sum()))
    m["cmPoolPtDMMax"] = f_(x["dm"].abs().max())
    m["cmPoolPtTMax"] = f_(x["t"].abs().max())
    crit = 1.96  # the two-sided 5 % normal critical value, as hac_p uses (reported, not tested)
    m["cmPoolPtSigN"] = str(int(((x["dm"].abs() > crit) | (x["t"].abs() > crit)).sum()))
    al = x[x["bucket"] == "all_features"]["dsh"]
    ot = x[x["bucket"] != "all_features"]["dsh"]
    m["cmPoolPtShAllMin"], m["cmPoolPtShAllMax"] = s_(al.min()), s_(al.max())
    m["cmPoolPtShOtherMin"], m["cmPoolPtShOtherMax"] = (
        s_(ot.max()),
        s_(ot.min()),
    )  # closest .. furthest
    c = pd.read_csv(POOL_STUDY / "causal_switch_log.csv")
    c = c[c["cluster"] == POOL_CLUSTER]
    g = c.groupby(["bucket", "est", "arm"]).agg(
        blocks=("block", "size"),
        sw=("switch", lambda s: int(s.astype(bool).sum())),
        mp=("passes_margin", lambda s: int(s.astype(bool).sum())),
    )
    assert g["blocks"].nunique() == 1
    m["cmPoolPtBlocks"] = str(int(g["blocks"].iloc[0]))
    m["cmPoolPtCausalSwMin"], m["cmPoolPtCausalSwMax"] = (
        str(int(g["sw"].min())),
        str(int(g["sw"].max())),
    )
    m["cmPoolPtCausalMarginMin"], m["cmPoolPtCausalMarginMax"] = (
        str(int(g["mp"].min())),
        str(int(g["mp"].max())),
    )
    r = pd.read_csv(POOL_STUDY / "retune_log.csv")
    r = r[r["cluster"] == POOL_CLUSTER]
    gr = r.groupby(["bucket", "est"]).agg(
        n=("block", "size"), sw=("switched", lambda s: int(s.astype(bool).sum()))
    )
    assert gr["n"].nunique() == 1
    m["cmPoolPtRetunes"] = str(int(gr["n"].iloc[0]))
    m["cmPoolPtRetuneSwMin"], m["cmPoolPtRetuneSwMax"] = (
        str(int(gr["sw"].min())),
        str(int(gr["sw"].max())),
    )
    claim(
        x["q_pct"].mean() < 0,
        "methods: pre-tuning improves the pooled models' QLIKE (mean change < 0)",
    )
    return m


# --------------------------------------------------------------------------- the trees' settings
# RF_CONFIG = RES / "rf_devmenu_2026-10-06" / "rf_config.json"  # experiments/tree_menu_dev_rf.py
# 2026-10-13 (user; review items 77-79): the forest's settings are the 400-trial freeze (trial 255)
RF_CONFIG = (
    RES / "rf_devmenu_2026-10-06" / "budget400" / "rf_config.json"
)  # experiments/tree_menu_dev_rf.py
RF_SUMMARY = RES / "rf_devmenu_2026-10-06" / "SUMMARY.md"
POOLED_WIN, PER_BAR_WIN = (
    24000,
    2000,
)  # the menu's pooled window, the per-bar window (bars)
LGBM_POOLED_MIN_CHILD = (
    98  # lgbm_opt_00's min_child_samples (specs/causal_tune_trees.py LGBM_PARAMS)
)


def tree_prov_macros() -> dict[str, str]:
    """Where the per-bar trees' settings come from (2026-10-06, user decision: the forest is
    dev-tuned like LightGBM / XGBoost): the forest's dev-span pick and the XGBoost provenance
    fact, read from results/rf_devmenu_2026-10-06/ (rf_config.json, SUMMARY.md)."""
    import json

    c = json.loads(RF_CONFIG.read_text(encoding="utf-8"))
    s = RF_SUMMARY.read_text(encoding="utf-8")
    m: dict[str, str] = {}
    m["cmTreeProvRfName"] = tt(c["name"])
    m["cmTreeProvRfTrials"] = str(int(c["trials"]["complete_considered"]))
    m["cmTreeProvRfBudget"] = str(int(c["trials"]["nominal_budget"]))
    m["cmTreeProvRfDevMse"] = f"{c['dev_screen_mse']:.6f}"
    cmp_ = c["compare_dev_screen_mse"]
    m["cmTreeProvLgbmDevMse"] = f"{cmp_['lgbm_opt_00']:.6f}"
    m["cmTreeProvXgbDevMse"] = f"{cmp_['xgb_opt_06']:.6f}"
    m["cmTreeProvXgbTopDevMse"] = f"{cmp_['xgb_opt_00']:.6f}"
    pr = c["params"]
    m["cmTreeProvRfTrees"] = str(int(pr["n_estimators"]))
    m["cmTreeProvRfDepth"] = str(int(pr["max_depth"]))
    m["cmTreeProvRfFeatures"] = f"{pr['max_features']:.2f}"
    m["cmTreeProvRfSamples"] = f"{pr['max_samples']:.2f}"
    m["cmTreeProvRfLeafDev"] = str(int(pr["min_samples_leaf"]))
    leaf = int(c["min_samples_leaf_scaled_to_2000_rows"])
    assert leaf == max(1, round(pr["min_samples_leaf"] * PER_BAR_WIN / POOLED_WIN)), (
        leaf
    )
    m["cmTreeProvRfLeaf"] = str(leaf)
    spec = (ROOT / "specs" / "causal_tune_trees.py").read_text(encoding="utf-8")
    mc = re.search(r'"min_child_samples": (\d+)', spec)
    assert mc and int(mc.group(1)) == LGBM_POOLED_MIN_CHILD, (
        "lgbm_opt_00's min_child_samples"
    )
    m["cmTreeProvLgbmLeafDev"] = str(LGBM_POOLED_MIN_CHILD)
    m["cmTreeProvLgbmLeaf"] = str(
        max(1, round(LGBM_POOLED_MIN_CHILD * PER_BAR_WIN / POOLED_WIN))
    )
    # 2026-10-13 parked (user; review item 76: XGBoost takes its top entry, xgb_opt_00 -- the look-ahead disclosure
    # goes): the XGBoost provenance macros below are still written (nothing live reads them); the new macro
    # cmTreeProvXgbTopName names the entry in use
    m["cmTreeProvXgbTopName"] = tt("xgb_opt_00")
    xcfg = json.loads(
        (RES / "trees_xgbtop_1600" / "xgb_config.json").read_text(encoding="utf-8")
    )
    assert xcfg["config"]["name"] == "xgb_opt_00", "the xgbtop run is not on xgb_opt_00"
    # the XGBoost provenance fact (SUMMARY.md, its own sentences; parsed, asserted present)
    rk = re.search(
        r"`xgb_opt_06` rank among the xgb Optuna entries \(1 = lowest MSE\): dev (\d)/(\d), "
        r"2003\+ (\d)/(\d), 2020-24 (\d)/(\d)",
        s,
    )
    ql = re.search(r"paper-scorer QLIKE ([\d.]+), rank (\d)/(\d)", s)
    dt = re.search(r"hand-picked by (\d{4}-\d{2}-\d{2})", s)
    assert rk and ql and dt, (
        "the XGBoost provenance sentences of the rf_devmenu SUMMARY"
    )
    m["cmTreeProvXgbName"] = tt("xgb_opt_06")
    m["cmTreeProvXgbDevRank"] = rk.group(1)
    m["cmTreeProvXgbArms"] = rk.group(2)
    m["cmTreeProvXgbPostRank"] = rk.group(3)  # 2003+ MSE
    m["cmTreeProvXgbRecentRank"] = rk.group(5)  # 2020-24 MSE
    m["cmTreeProvXgbQRank"] = ql.group(2)  # the paper scorer's QLIKE
    m["cmTreeProvXgbPickDate"] = _date(dt.group(1))
    return m


# --------------------------------------------------------------------------- the parked lean set
LEAN_STEPS = (  # the three lean steps Table 8 carried until the lean set was parked
    ("PerBarLean", "poolpt_ridge_lean_vol_vix", "subpt_ridge_lean_vol_vix"),
    ("InBaseLean", "subpt_ridge_baseline", "subpt_ridge_lean_vol_vix"),
    ("InLeanAll", "subpt_ridge_lean_vol_vix", "subpt_ridge_all_features"),
)


def lean_unused_macros(d: pd.DataFrame) -> dict[str, str]:
    """2026-10-08 parked (lean set; replaced by an MCS).

    The lean set is in no table, step, count or model set of the paper any more.  The few lean
    macros the text still names (until its parking pass) are kept DEFINED here, from the master
    table's lean rows -- the forecasts the paper set held on lean_vol_vix before (pre-tuned
    per-bar / pooled linear, every tree rung, the dev-tuned forest's T1 / T10) -- so that nothing
    is undefined.  Same statistics as Table 8 and master_macros.
    """
    import atm_straddle_lib as asl
    from src.evaluation.diebold_mariano import dm_test

    mtc = _mtc()
    m: dict[str, str] = {}
    a = d[(d["table"] == "A") & d.index.str.contains("_" + camp.LEAN)]
    fam = a["family"].astype(str)
    keep = (
        (fam.isin(camp.PAPER_LINEAR_FAMILIES) | fam.str.startswith("per-bar tree"))
        & ~a.index.str.startswith(camp.ONE_PENALTY)
        & ~a.index.str.startswith("pool_")
        & ~(
            a.index.str.startswith("subtree_rf_")
            | (a.index.str.startswith("subtree_daily_") & a.index.str.endswith("_rf"))
        )
    )
    x = a[keep]
    m["cmMasterNLean"] = str(len(x))
    m["cmMasterMedQLean"] = f"{x['qlike_raw'].median():.4f}"
    m["cmMasterMedShLean"] = f_(x["Sharpe_mid"].median())
    m["cmMasterMedShXLean"] = f_(x["Sharpe_crossed"].median())
    daily = pd.read_parquet(DAILY)
    fr = {k: g.set_index("day").sort_index() for k, g in daily.groupby("key")}
    for tag, frm, to in LEAN_STEPS:
        aa, bb = fr[to], fr[frm]
        assert len(aa) == len(bb) == N_DAYS and (aa.index == bb.index).all(), (frm, to)
        dq = aa["qlike_raw"] - bb["qlike_raw"]
        m[f"cmLd{tag}"] = s_(
            mtc.sharpe(aa["ret_mid"].to_numpy()) - mtc.sharpe(bb["ret_mid"].to_numpy())
        )
        m[f"cmLd{tag}Q"] = s_(
            100.0 * float(dq.mean()) / float(bb["qlike_raw"].mean()), 1
        )
        m[f"cmLd{tag}T"] = s_(
            asl.newey_west_t(aa["ret_mid"].to_numpy() - bb["ret_mid"].to_numpy())[0]
        )
        m[f"cmLd{tag}DM"] = s_(
            float(
                dm_test(aa["qlike_raw"].to_numpy(), bb["qlike_raw"].to_numpy())["dm"]
            ),
            2,
        )
    return m


def write_numbers(m: dict[str, str]) -> None:
    lines = [
        "% AUTO-GENERATED by writeup/make_table_close_main.py -- do not edit.",
        "% Numbers quoted in the prose of writeup/sections/results_close_option.tex; each is read",
        "% from results/close_master_table/master_table.csv and the sources in the script.",
        "% 2026-10-06: recalibration removed / intervals removed, user decision -- the plain forecast",
        "% f^2 B, point estimates; ...T = HAC t of the daily mid-return difference, ...DM = Diebold-",
        "% Mariano statistic; a ...Above / ...Below count counts point estimates, its ...Sig twin those",
        f"% significant at {100 * SIG:.0f} % (DM for QLIKE, HAC t for Sharpe).",
    ]
    for k, v in m.items():
        assert re.fullmatch(r"cm[A-Za-z]+", k), k
        assert "nan" not in v.lower() and not any(
            isinstance(x, float) and math.isnan(x) for x in [v]
        ), (k, v)
        assert "??" not in v and "[" not in v, (k, v)  # no stub, no interval
        assert not re.search(r"(?<!\\)_", v), (k, v)  # text-safe
        lines.append(rf"\newcommand{{\{k}}}{{{v}}}")
    lines += [
        "%",
        "% 2026-10-06 parked (recalibration removed / intervals removed, user decision):",
    ]
    lines += [f"%   \\{p}" for p in PARKED]
    lines += [
        "%",
        "% statements of the text checked on these numbers (2026-10-06; not asserted):",
    ]
    lines += [f"%   {'holds  ' if ok else 'FLIPPED'} {t}" for ok, t in CLAIMS]
    NUMBERS_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def pending_rows(d: pd.DataFrame) -> list[str]:
    """Rows / steps of Tables 7 and 8 whose forecast is not in the master table yet.

    2026-10-06 (speed-up, coordinator): the pooled lean tables (yhat_pool_*_lean_vol_vix) run
    on the cluster and land later; until then their Table 7 row and Table 8 step are left out
    (and reported), and the increment re-run fills them.  Every other row is required.
    """
    pend = [k for k, _, _ in POOLED if k not in d.index]
    assert all(k.endswith(camp.LEAN) for k in pend), ("a non-lean row is missing", pend)
    POOLED[:] = [r for r in POOLED if r[0] in d.index]  # PANELS holds this list
    LADDER[:] = [s for s in LADDER if s[1] in d.index and s[2] in d.index]
    if pend:
        print(f"PENDING (not in the master table yet): {pend}")
    return pend


def main() -> None:
    check_manifest()
    d = load()
    pend = pending_rows(d)
    write_table(d)
    m = macros(d)
    ms = model_set(d)
    lad = ladder()
    write_ladder_table(lad, d)
    m.update(ladder_macros(lad))
    m.update(campaign_macros(d))
    # 2026-10-06 (user decision): D.14 (where the former headline's P&L comes from) is parked --
    # its \cmPnl... macros and generated/table_close_pnl.tex are no longer written:
    # m.update(pnl_macros(d))
    for nm in PNL_MACROS:
        park(
            nm, "D.14, the former headline's P&L decomposition (parked, user decision)"
        )
    sv_rescore(d)  # the gate on F1's rebuilt returns (its macros are parked)
    m.update(sv_set_macros(d, ms))
    m.update(tail_macros(d, ms))
    m.update(bounds_macros(ms))
    m.update(pretune_macros())  # 2026-10-06: the pre-tuned per-bar linear models
    m.update(poolpt_macros(d))  # 2026-10-06: the pre-tuned pooled linear models
    m.update(tree_prov_macros())  # 2026-10-06: the trees' settings (dev-tuned forest)
    m.update(
        e32_macros(d, ms)
    )  # 2026-10-23 (user, E32): every closing-interval tree refit every session
    m.update(
        lean_unused_macros(d)
    )  # 2026-10-08 parked (lean set; replaced by an MCS): defined, unused
    # write_pnl_table()  # 2026-10-06: D.14 parked (user decision); a comment-only stub instead
    PNL_OUT.write_text(
        "% AUTO-GENERATED by writeup/make_table_close_main.py -- do not edit.\n"
        "% 2026-10-06: D.14 (the former headline's P&L decomposition) is parked, user decision.\n",
        encoding="utf-8",
        newline="\n",
    )
    write_model_set(ms)
    write_numbers(m)
    print(
        f"wrote {TABLE_OUT.relative_to(ROOT)} and {NUMBERS_OUT.relative_to(ROOT)} ({len(m)} macros)"
    )
    print(f"parked {len(PARKED)} macros; pending rows: {pend or 'none'}")
    flipped = [t for ok, t in CLAIMS if not ok]
    print(f"{len(CLAIMS)} statements of the text checked, {len(flipped)} FLIPPED:")
    for t in flipped:
        print("  FLIPPED: " + t)


if __name__ == "__main__":
    main()
