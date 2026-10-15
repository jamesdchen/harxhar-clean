"""Tables and figures for the rehauled paper's economic section (writeup/rehaul/07_economics.tex).

Outline: writeup/PAPER_REHAUL_OUTLINE_2026-10-15.md, section 7 (E1-E4 and the figures).

DATA
  The paper model set (results/close_master_table/paper_model_set.csv, 112 forecasts) and its per-day frame
  results/close_master_table/master_table_daily.parquet (q, ret_mid, ret_crossed, plain, iv_var, qlike_raw on 866
  trade days); master_table.csv gates every recomputed number.  The straddle book (R, exit, bids, asks) is the GUI's
  quote book (results/atm_straddle_0dte_1530/daily_a0.parquet).  The per-bar MinGRU (ridge-initialised shortcut,
  QLIKE-tuned; results/puffer_forecast_2026-10-12/qlike_study/score_rinitq/forecasts.csv) is read and scored by the
  GUI's own loader (master_table_gui.rows._nn_daily), so its rows are the GUI's rows.

TRADE (experiments/master_table_close.py:1168-1187, 1293-1312)
  q = sign(s), s = plain - iv_var, ties short; mid r = q R; crossed: a buy pays the ask, a sale gets the bid, spread
  paid at entry only.  Always short: q = -1.  Sharpe = mean / sd(ddof 1) x sqrt(252), per trade day.  Every t is
  atm_straddle_lib.newey_west_t (Bartlett, lag floor(1.5 n^(1/3)) = 14 on 866 days); dSharpe's t is the HAC t of the
  daily return difference (it tests the mean difference, not the Sharpe difference).  DM (E3 only):
  src.evaluation.diebold_mariano.dm_test on the daily QLIKE difference, the master table's setting (automatic lag).

SIZING (writeup/master_table_gui/rows.py:88-134 _position_sizes, :1140-1180 _score_sizes)
  The GUI's functions are called directly: sizes use only earlier days; scored on days 253-866 (614).

OUTPUTS
  writeup/generated/rehaul_econ_e1_control.tex     Table E1  (tab:rh_econ_control)
  writeup/generated/rehaul_econ_e2_allfeat.tex     Table E2  (tab:rh_econ_allfeat)
  writeup/generated/rehaul_econ_e3_pooled.tex      Table E3  (tab:rh_econ_pooled)
  writeup/generated/rehaul_econ_e4_sizing.tex      Table E4  (tab:rh_econ_sizing)
  writeup/generated/rehaul_econ_fig_cumpnl.tex     float for figures/rehaul_econ_cumpnl.pdf  (fig:rh_econ_cumpnl)
  writeup/generated/rehaul_econ_fig_sizing.tex     float for figures/rehaul_econ_sizing.pdf  (fig:rh_econ_sizing)
  writeup/generated/rehaul_econ_fig_byyear.tex     float for figures/rehaul_econ_byyear.pdf  (fig:rh_econ_byyear)
  writeup/generated/rehaul_econ_numbers.tex        \\rhe... macros for the section's prose (not \\input by the floats)

TABLE LEGIBILITY (2026-10-15)
  Tables follow writeup/NOTATION_GUIDE_2026-10-15.md section 8 and TABLE_FIXES_FROM_FLOATS_2026-10-15.md: short
  captions, Notes of about five sentences, stars for the 10/5/1 % levels on the estimate with its HAC t (or DM) in
  parentheses underneath, at most 7 columns, Panel A/B, the guide's vocabulary.  The numbers the notes and the
  cumulative-return caption cite are \\rhe... macros, so every table and the figure float \\input
  rehaul_econ_numbers.tex when \\rheNdays is not yet defined.  The sizing and by-year figures are parked by the floats
  file (guide section 7); their code here is unchanged.

Run:  python writeup/make_rehaul_economics.py [--preview DIR]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WRITEUP = ROOT / "writeup"
for _p in (ROOT, ROOT / "notebooks", WRITEUP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from master_table_gui import rows as gui  # noqa: E402

from src.evaluation.diebold_mariano import dm_test  # noqa: E402

RES = ROOT / "results" / "close_master_table"
GEN = WRITEUP / "generated"
FIG = WRITEUP / "figures"
ANN = np.sqrt(252.0)
SIG = 1.959963984540054  # two-sided 5 %
WARMUP = gui._SIZING_WARMUP  # 252
HAC_LAG = 14

# ----------------------------------------------------------------------------- forecasts of record
SHORT = "always_short"
OLS = "sub_ols_baseline"
NN_ALL, NN_BASE = "mingru_rinitq_all_features", "mingru_rinitq_baseline"
PERBAR_ALL = (
    "subpt_enet_all_features",
    "subpt_lasso_all_features",
    "subpt_ridge_all_features",
)
POOLED_ALL = (
    "poolpt_enet_all_features",
    "poolpt_lasso_all_features",
    "poolpt_ridge_all_features",
)
TREE_EST = (("lgbm", "LightGBM"), ("xgb", "XGBoost"), ("rf", "Random forest"))
EST_NAME = {"enet": "Elastic net", "lasso": "Lasso", "ridge": "Ridge"}

# ----------------------------------------------------------------------------- palette (one hue per family)
# Validated with the dataviz skill's validate_palette.js on a white surface, all pairs (lines cross): lightness band,
# chroma floor, CVD (worst deutan dE 8.2) and normal-vision (worst dE 19.8) floors and 3:1 contrast all pass.  Always
# short is the neutral ink.  Every family also has its own marker, so identity is never colour alone.
FAM_COLOR = {
    "linear": "#2a78d6",
    "tree": "#d55e00",
    "nn": "#009e73",
    "pooled": "#b9479b",
    "short": "#222222",
}
FAM_MARKER = {"linear": "o", "tree": "s", "nn": "^", "pooled": "D", "short": "x"}
# 2026-10-15 (table legibility): parked -- code terms (per-bar, 48 bars); the guide's model classes below
# FAM_LABEL = {
#     "linear": "Per-bar linear",
#     "tree": "Per-bar trees",
#     "nn": "Per-bar MinGRU",
#     "pooled": "Pooled (48 bars)",
#     "short": "Always short",
# }
FAM_LABEL = {
    "linear": "Closing-interval linear",
    "tree": "Closing-interval trees",
    "nn": "Closing-interval MinGRU",
    "pooled": "Pooled intraday",
    "short": "Short straddle",  # 2026-10-15 (consolidation): was "Always short" (guide section 6b)
}
TOP_DAYS = (
    10  # the tail check: each forecast's 10 days of largest gain over always short
)
GRID = "#d9d9d9"
INK2 = "#52514e"

plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.edgecolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.labelcolor": "#111111",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)
FIG_W = 6.5  # \linewidth of paper_rehaul.tex (letter, 1in margins), so the PDF is placed at 100 %


# ============================================================================= data
def sharpe(x) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * ANN)


def nw_t(x) -> float:
    t, lag = asl.newey_west_t(np.asarray(x, float))
    if len(x) == 866:
        assert lag == HAC_LAG, lag
    return float(t)


def family_of(key: str, mt_family: str | None) -> str:
    if key.startswith("mingru_"):
        return "nn"
    if key == SHORT:
        return "short"
    if mt_family == "per-bar linear":
        return "linear"
    if (mt_family or "").startswith("per-bar tree"):
        return "tree"
    if mt_family in (
        "pooled twin",
        "paper",
    ):  # the paper's nine forecasts are pooled 48-bar fits too
        return "pooled"
    raise ValueError(f"no figure family for {key} ({mt_family})")


def load():
    mt = pd.read_csv(RES / "master_table.csv").set_index("key")
    ps = pd.read_csv(RES / "paper_model_set.csv")
    model_set = list(ps["key"])
    assert len(model_set) == 112, len(model_set)
    daily = pd.read_parquet(RES / "master_table_daily.parquet")
    daily = daily[daily["key"].isin(model_set)].copy()
    daily["day"] = pd.to_datetime(daily["day"]).dt.normalize()

    quotes = pd.read_parquet(
        gui._QUOTE_BOOK, columns=["R", "exit", "bid_c", "bid_p", "ask_c", "ask_p"]
    )
    quotes.index = pd.to_datetime(quotes.index).normalize().astype("datetime64[ns]")
    book = pd.DataFrame(
        {
            "R": quotes["R"].astype(float),
            "ex": quotes["exit"].astype(float),
            "bid": (quotes["bid_c"] + quotes["bid_p"]).astype(float),
            "ask": (quotes["ask_c"] + quotes["ask_p"]).astype(float),
        }
    )

    series: dict[str, pd.DataFrame] = {}
    for key, g in daily.groupby("key", sort=False):
        g = g.set_index("day").sort_index().join(book, how="inner")
        assert len(g) == 866, (key, len(g))
        assert np.allclose(g["q"] * g["R"], g["ret_mid"], rtol=0, atol=1e-10), key
        cx = gui._crossed_return(
            g["q"].to_numpy(),
            g["ex"].to_numpy(),
            g["bid"].to_numpy(),
            g["ask"].to_numpy(),
        )
        assert np.allclose(cx, g["ret_crossed"], rtol=0, atol=1e-10), key
        series[key] = pd.DataFrame(
            {
                "q": g["q"],
                "mid": g["ret_mid"],
                "x": g["ret_crossed"],
                "s": g["plain"] - g["iv_var"],
                "qlike": g["qlike_raw"],
                "R": g["R"],
                "ex": g["ex"],
                "bid": g["bid"],
                "ask": g["ask"],
            }
        )
    days = next(iter(series.values())).index
    for key, g in series.items():
        assert g.index.equals(days), key

    # the MinGRU rows, exactly as the GUI builds them (rows._nn_daily: plain = f^2 B, the quote book's straddle)
    nn = gui._nn_daily()
    assert nn is not None, "the GUI's MinGRU loader found no forecasts"
    for key in (NN_BASE, NN_ALL):
        g = nn[nn["key"] == key].copy()
        g["day"] = pd.to_datetime(g["day"]).dt.normalize().astype("datetime64[ns]")
        g = g.set_index("day").sort_index().join(book, how="inner")
        assert g.index.equals(days), key
        # ratio = (g["target"] / g["plain"]).to_numpy(float)  # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10)
        ratio = (g["rv_unclipped"] / g["plain"]).to_numpy(float)
        series[key] = pd.DataFrame(
            {
                "q": g["q"],
                "mid": g["ret_mid"],
                "x": g["ret_crossed"],
                "s": g["plain"] - g["iv_var"],
                "qlike": ratio - np.log(ratio) - 1.0,
                "R": g["R"],
                "ex": g["ex"],
                "bid": g["bid"],
                "ask": g["ask"],
            }
        )

    b = series[OLS]
    series[SHORT] = pd.DataFrame(
        {
            "q": -1.0,
            "mid": -b["R"],
            "x": 1.0 - b["ex"] / b["bid"],
            "R": b["R"],
            "ex": b["ex"],
            "bid": b["bid"],
            "ask": b["ask"],
        },
        index=days,
    )
    fam = {k: family_of(k, mt["family"].get(k)) for k in series}
    return mt, model_set, series, fam


# ============================================================================= statistics
def trade_stats(series, key) -> dict:
    g, sh = series[key], series[SHORT]
    mid, x = g["mid"].to_numpy(float), g["x"].to_numpy(float)
    out = {
        "mean": float(mid.mean()),
        "t_mean": nw_t(mid),
        "sh_mid": sharpe(mid),
        "sh_x": sharpe(x),
        "hit": float((mid > 0).mean()),
        "buys": float((g["q"] > 0).mean()),
    }
    if key != SHORT:
        d = mid - sh["mid"].to_numpy(float)
        out["dsh"] = out["sh_mid"] - sharpe(sh["mid"])
        out["t_dsh"] = nw_t(d)
    return out


def gate_against_master(mt, series, keys):
    """Every recomputed E1 statistic must equal the master table's stored value."""
    for key in keys:
        if key not in mt.index:
            continue
        st, row = trade_stats(series, key), mt.loc[key]
        pairs = [
            (st["sh_mid"], row["Sharpe_mid"]),
            (st["sh_x"], row["Sharpe_crossed"]),
            (st["mean"], row["mean_mid"]),
            (st["hit"], row["hit_rate_mid"]),
            (100 * st["buys"], row["pct_buy"]),
        ]
        if key != SHORT:
            pairs += [
                (st["dsh"], row["dSharpe_mid_vs_short"]),
                (st["t_dsh"], row["t_hac_mid_vs_short"]),
            ]
        for got, want in pairs:
            assert abs(got - float(want)) < 1e-8, (key, got, want)


def sizing_stats(series, key) -> dict:
    """The GUI's own sizing scorer (rows._score_sizes) on this forecast's gap, scored on days 253-866."""
    g = series[key]
    return gui._score_sizes(
        g["s"].to_numpy(float),
        g["R"].to_numpy(float),
        g["ex"].to_numpy(float),
        g["bid"].to_numpy(float),
        g["ask"].to_numpy(float),
    )


def pair_stats(series, base, new) -> dict:
    a, b = series[new], series[base]
    return {
        "sh_b": sharpe(b["mid"]),
        "sh_a": sharpe(a["mid"]),
        "d_mid": sharpe(a["mid"]) - sharpe(b["mid"]),
        "t_mid": nw_t(a["mid"].to_numpy() - b["mid"].to_numpy()),
        "d_x": sharpe(a["x"]) - sharpe(b["x"]),
        "t_x": nw_t(a["x"].to_numpy() - b["x"].to_numpy()),
    }


def tree_est(key: str) -> str:
    """lgbm / xgb / rf: the token sits at the end (subtree_tuned_all_features_lgbm) or after subtree_ (subtree_xgb_baseline)."""
    hits = [e for e, _ in TREE_EST if key.endswith(f"_{e}") or f"_{e}_" in key]
    assert len(hits) == 1, (key, hits)
    return hits[0]


def best_tree(series, fam, model_set, est) -> str:
    cands = [
        k
        for k in model_set
        if fam[k] == "tree" and "_all_features" in k and tree_est(k) == est
    ]
    return max(cands, key=lambda k: sharpe(series[k]["mid"]))


# 2026-10-15 (table legibility): parked -- code terms (sessions, refit, MSE-selected); replaced by tree_spec below
# def tree_rung(key: str) -> str:
#     """Plain-words configuration of a per-bar tree key (the master table's family plus the key's selection)."""
#     sel = "QLIKE-selected" if ("tunedq" in key or "optunaq" in key) else "MSE-selected"
#     if key.startswith("subtree_daily_"):
#         return "default settings, refit daily"
#     if key.startswith("subtree_tuned_daily_") or key.startswith(
#         "subtree_tunedq_daily_"
#     ):
#         return f"random search, refit daily, {sel}"
#     if key.startswith("subtree_tuned") and "optuna" not in key:
#         return f"random search, refit every 10 sessions, {sel}"
#     if key.startswith("subtree_optuna"):
#         tp = key.split("_tp")[1].split("_")[0]
#         k = (
#             key.split("_k")[1].split("_")[0]
#             if "_k" in key.split("_tp")[1][:8]
#             else "50"
#         )
#         every = "every session" if tp == "1" else f"every {tp} sessions"
#         return f"Optuna, retuned {every}, best of {k} trials, {sel}"
#     if key.startswith("subtree_"):
#         return "default settings, refit every 10 sessions"
#     raise ValueError(key)


def tree_spec(key: str) -> str:
    """Plain-words specification of a closing-interval tree key, in the guide's terms (trading days, re-estimated,
    hyperparameters selected on squared-error loss or QLIKE)."""
    loss = "QLIKE" if ("tunedq" in key or "optunaq" in key) else "squared-error loss"
    if key.startswith("subtree_daily_"):
        return "fixed hyperparameters, re-estimated daily"
    if key.startswith("subtree_tuned_daily_") or key.startswith(
        "subtree_tunedq_daily_"
    ):
        return f"random search, re-estimated daily, selected on {loss}"
    if key.startswith("subtree_tuned") and "optuna" not in key:
        return f"random search, re-estimated every 10 trading days, selected on {loss}"
    if key.startswith("subtree_optuna"):
        tp = key.split("_tp")[1].split("_")[0]
        k = (
            key.split("_k")[1].split("_")[0]
            if "_k" in key.split("_tp")[1][:8]
            else "50"
        )
        every = "daily" if tp == "1" else f"every {tp} trading days"
        return f"Optuna search ({k} trials), re-tuned {every}, selected on {loss}"
    if key.startswith("subtree_"):
        return "fixed hyperparameters, re-estimated every 10 trading days"
    raise ValueError(key)


# ============================================================================= LaTeX helpers
def num(x: float, dp: int = 2, plus: bool = False, bold: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    v = round(float(x), dp)
    if v == 0:
        v = 0.0  # no negative zero
    body = f"{abs(v):.{dp}f}"
    sign = "-" if v < 0 else ("+" if plus and v > 0 else "")
    s = f"${sign}{body}$"
    return f"{{\\bfseries\\boldmath {s}}}" if bold else s


def pct(x: float, dp: int = 1, bold: bool = False) -> str:
    return num(100 * x, dp, bold=bold)


def sig(t: float) -> bool:
    return bool(np.isfinite(t) and abs(t) > SIG)


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")
    print("wrote", path.relative_to(ROOT))


HEADER = "% AUTO-GENERATED by writeup/make_rehaul_economics.py -- do not edit; re-run the script.\n"

# 2026-10-15 (table legibility): parked -- notes and panel helper of the old tables; replaced by the finance-style helpers below
# NOTE_FILLS = (
#     r"Mid: the straddle is bought or sold at the 15:30 midpoint; crossed: a buy pays the ask and a sale receives "
#     r"the bid, the spread paid at entry only (settlement is cash at intrinsic)."
# )
# NOTE_SHARPE = (
#     r"Sharpe ratio $=\sqrt{252}\,\bar r/\mathrm{sd}(r)$, annualised per trade day (the trade runs about 200 days a "
#     r"year)."
# )
# NOTE_HAC = (
#     r"Every $t$ is Newey--West (Bartlett kernel, lag $\lfloor 1.5\,n^{1/3}\rfloor = 14$ at $n = 866$); the $t$ of a "
#     r"$\Delta$Sharpe is that of the mean daily return difference, so it tests the mean difference, not the Sharpe "
#     r"difference. Bold: $|t| > 1.96$ (5\,\%, two-sided), on the estimate and its $t$."
# )
#
#
# def panel(title: str, ncol: int, first: bool = False) -> str:
#     return (
#         "" if first else "\\addlinespace[3pt]\n"
#     ) + f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\\n"


# ----------------------------------------------------------------------------- 2026-10-15 (table legibility)
# Finance-journal table style (writeup/NOTATION_GUIDE_2026-10-15.md section 8, TABLE_FIXES_FROM_FLOATS_2026-10-15.md):
# short caption; Notes of about five sentences below; stars for the 10/5/1 % levels on the estimate, its test statistic
# in parentheses underneath; at most about 7 columns; Panel A/B; the guide's vocabulary; numbers in the notes are the
# \rhe... macros of generated/rehaul_econ_numbers.tex, which every table \inputs if they are not yet defined.
LEVELS = (0.10, 0.05, 0.01)
STAR_NOTE = (
    r"$^{*}$, $^{**}$ and $^{***}$ denote significance at the 10\%, 5\% and 1\% levels."
)
NUMBERS_GUARD = (
    "\\ifdefined\\rheNdays\\else\\input{generated/rehaul_econ_numbers}\\fi\n"
)
# 2026-10-15 (consolidation): the Newey-West lag is q_NW (q_t is the position; NOTATION_GUIDE section 6b).
HAC_NOTE = (
    r"HAC $t$-statistic on the daily return differential (Newey--West, $q_{\mathrm{NW}} = \rheHACLag$; "
    r"Section~\ref{sec:rh_methods_tests})"
)
# 2026-10-15 (consolidation): one name for the q_t = -1 benchmark (guide section 6b): "the short-straddle benchmark" in
# prose and notes, "Short straddle" as the row and legend label.
SHORT_LABEL = "Short straddle"


def p_two(t: float) -> float:
    """Two-sided normal p-value of a t or DM statistic (as dm_test and the paper's HAC t use)."""
    return math.erfc(abs(t) / math.sqrt(2.0)) if np.isfinite(t) else float("nan")


def n_stars(t: float) -> int:
    p = p_two(t)
    return int(sum(p < a for a in LEVELS)) if np.isfinite(p) else 0


def _fmt(x: float, dp: int) -> str:
    v = round(float(x), dp)
    if v == 0:
        v = 0.0  # no negative zero
    return f"{'-' if v < 0 else ''}{abs(v):.{dp}f}"


def cell(x: float, dp: int = 2, stars: int = 0, starred: bool = True) -> str:
    """An estimate in math mode. In a starred column every cell carries a three-star superscript, the unused stars as
    a phantom, so the decimals line up in an r column."""
    if x is None or not np.isfinite(x):
        return "--$^{\\phantom{***}}$" if starred else "--"
    body = _fmt(x, dp)
    if not starred:
        return f"${body}$"
    pad = "*" * (3 - stars)
    sup = "*" * stars + (f"\\phantom{{{pad}}}" if pad else "")
    return f"${body}^{{{sup}}}$"


def tstat(t: float, dp: int = 2) -> str:
    """A test statistic in parentheses under its estimate, padded like a starred cell."""
    if t is None or not np.isfinite(t):
        return ""
    return f"$({_fmt(t, dp)})^{{\\phantom{{***}}}}$"


def count(k: int, n: int) -> str:
    """Number of pairs significant at the 5 % level, in brackets under a median (padded like a starred cell)."""
    return f"$[{k}/{n}]^{{\\phantom{{***}}}}$"


def rows2(label: str, top: list[str], under: list[str]) -> str:
    """One estimate row, then the row of test statistics under it (skipped when there is none)."""
    out = f"{label} & " + " & ".join(top) + " \\\\\n"
    if any(under):
        out += " & " + " & ".join(under) + " \\\\[2pt]\n"
    return out


def panel_head(title: str, ncol: int, first: bool = False) -> str:
    return (
        ("" if first else "\\addlinespace[4pt]\n")
        + f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\\n"
        + "\\addlinespace[2pt]\n"
    )


def sub_head(title: str, ncol: int) -> str:
    return f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{{title}}} \\\\\n"


def float_head(label: str, what: str) -> str:
    return HEADER + f"% {label} (outline 7.4): {what}\n" + NUMBERS_GUARD


# ============================================================================= E1
# 2026-10-15 (table legibility): parked -- 9-column E1 with bold significance; replaced by table_e1 below (stars, t under the estimate, panels)
# def table_e1(series, fam, model_set, trees) -> tuple[str, dict]:
#     st = {
#         k: trade_stats(series, k)
#         for k in [SHORT, OLS, *PERBAR_ALL, *trees.values(), NN_ALL, *POOLED_ALL]
#     }
#
#     def row(label: str, key: str) -> str:
#         s = st[key]
#         tm = sig(s["t_mean"])
#         cells = [
#             pct(s["mean"], 1, bold=tm),
#             num(s["t_mean"], bold=tm),
#             num(s["sh_mid"]),
#             num(s["sh_x"]),
#             pct(s["hit"]),
#             pct(s["buys"]),
#         ]
#         if key == SHORT:
#             cells += ["--", "--"]
#         else:
#             td = sig(s["t_dsh"])
#             cells += [num(s["dsh"], plus=True, bold=td), num(s["t_dsh"], bold=td)]
#         return f"{label} & " + " & ".join(cells) + " \\\\\n"
#
#     n_trees = {
#         e: sum(
#             1
#             for k in model_set
#             if fam[k] == "tree" and "_all_features" in k and tree_est(k) == e
#         )
#         for e, _ in TREE_EST
#     }
#     body = row(r"Always short ($q_t = -1$)", SHORT)
#     body += panel("Per-bar linear (16:00 bar)", 9)
#     body += row("OLS, HAR + calendar", OLS)
#     for k in PERBAR_ALL:
#         body += row(f"{EST_NAME[k.split('_')[1]]}, all features", k)
#     body += panel("Per-bar trees, all features", 9)
#     for est, name in TREE_EST:
#         body += row(f"{name}\\tnote{{a}}", trees[est])
#     body += panel("Per-bar neural net", 9)
#     body += row(r"MinGRU, all features\tnote{b}", NN_ALL)
#     body += panel("Pooled linear (all 48 bars)", 9)
#     for k in POOLED_ALL:
#         body += row(f"{EST_NAME[k.split('_')[1]]}, all features", k)
#
#     assert len(set(n_trees.values())) == 1, n_trees
#     tree_note = "; ".join(
#         f"{name.lower() if e == 'rf' else name}: {tree_rung(trees[e])}"
#         for e, name in TREE_EST
#     )
#     tex = (
#         HEADER
#         + rf"""% Table E1 (outline 7.4): always short and the key forecasts under sign(s), 866 trade days.
# \begin{{table}}[tbp]
# \centering
# \small
# \setlength{{\tabcolsep}}{{4.5pt}}
# \begin{{threeparttable}}
# \caption{{The closing trade under $q_t = \sign(s_t)$: always short and the key forecasts, 866 trade days.}}
# \label{{tab:rh_econ_control}}
# \begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}
# \toprule
#  & \multicolumn{{2}}{{c}}{{Mid return}} & \multicolumn{{2}}{{c}}{{Sharpe ratio}} & & & \multicolumn{{2}}{{c}}{{vs always short}} \\
# \cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}\cmidrule(l){{8-9}}
# Forecast & Mean (\%) & $t$ & Mid & Crossed & Hit (\%) & Buys (\%) & $\Delta$Sharpe & $t$ \\
# \midrule
# {body}\bottomrule
# \end{{tabular}}
# \begin{{tablenotes}}[flushleft]
# \footnotesize
# \item \textit{{Notes:}} One 0DTE SPXW straddle (a one-strike strangle when spot is between strikes) is bought
# ($q_t = +1$) or sold ($q_t = -1$) at 15:30 ET and held to the close; $r_t = q_t R_t$, with $R_t$ the return on the
# premium; 3 January 2020 to 30 April 2024. {NOTE_FILLS} Mean: the mean mid return, in \% of the premium. {NOTE_SHARPE}
# Hit: share of days with $r_t > 0$ at mid; Buys: share of days with $q_t = +1$. $\Delta$Sharpe: mid Sharpe ratio minus
# always short's ({num(st[SHORT]["sh_mid"])}). {NOTE_HAC}
# \item[a] Of each estimator's {n_trees["lgbm"]} configurations on all features in the model set, the one with the
# highest mid Sharpe ratio, so selected on the evaluation sample. {tree_note}.
# \item[b] Ridge-initialised linear shortcut plus a MinGRU branch, trained and selected on QLIKE; the mean of 5 seeds.
# \end{{tablenotes}}
# \end{{threeparttable}}
# \end{{table}}
# """
#     )
#     return tex, st


def tree_specs(fam, model_set) -> int:
    """Specifications per tree estimator on the full predictor set in the model set (equal across estimators)."""
    n_trees = {
        e: sum(
            1
            for k in model_set
            if fam[k] == "tree" and "_all_features" in k and tree_est(k) == e
        )
        for e, _ in TREE_EST
    }
    assert len(set(n_trees.values())) == 1, n_trees
    return n_trees["lgbm"]


def table_e1(series, fam, model_set, trees) -> tuple[str, dict]:
    st = {
        k: trade_stats(series, k)
        for k in [SHORT, OLS, *PERBAR_ALL, *trees.values(), NN_ALL, *POOLED_ALL]
    }

    def row(label: str, key: str) -> str:
        s = st[key]
        top = [
            cell(100 * s["mean"], 1, n_stars(s["t_mean"])),
            cell(s["sh_mid"], starred=False),
            cell(s["sh_x"], starred=False),
            cell(100 * s["hit"], 1, starred=False),
            cell(100 * s["buys"], 1, starred=False),
        ]
        under = [tstat(s["t_mean"]), "", "", "", ""]
        if key == SHORT:
            top.append(cell(float("nan")))
            under.append("")
        else:
            top.append(cell(s["dsh"], 2, n_stars(s["t_dsh"])))
            under.append(tstat(s["t_dsh"]))
        return rows2(label, top, under)

    tree_specs(
        fam, model_set
    )  # asserts the estimators have equally many specifications
    body = row(SHORT_LABEL, SHORT)
    body += panel_head("Panel A: Closing-interval models", 7)
    body += row("OLS, HAR benchmark", OLS)
    for k in PERBAR_ALL:
        body += row(EST_NAME[k.split("_")[1]], k)
    for e, name in TREE_EST:
        body += row(f"{name}\\tnote{{a}}", trees[e])
    body += row(r"MinGRU\tnote{b}", NN_ALL)
    body += panel_head("Panel B: Pooled intraday models", 7)
    for k in POOLED_ALL:
        body += row(EST_NAME[k.split("_")[1]], k)

    tree_note = "; ".join(
        f"{name if e != 'rf' else name.lower()}: {tree_spec(trees[e])}"
        for e, name in TREE_EST
    )
    tex = (
        float_head("Table E1", "the sign strategy and the short-straddle benchmark")
        + rf"""\begin{{table}}[tbp]
\centering
\small
\begin{{threeparttable}}
\caption{{Returns of the sign strategy}}
\label{{tab:rh_econ_control}}
\begin{{tabular}}{{@{{}}lrrrrrr@{{}}}}
\toprule
 & Mean & \multicolumn{{2}}{{c}}{{$\SR$}} & Hit & Long & \\
\cmidrule(lr){{3-4}}
 & return (\%) & Mid-quotes & Bid--ask & rate (\%) & days (\%) & $\Delta\SR$ \\
\midrule
{body}\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item \textit{{Notes:}} Daily returns of the straddle position over the out-of-sample period, \rheSampleStart{{}} to
\rheSampleEnd{{}} ($T_{{\mathrm{{oos}}}} = \rheNdays$ trading days). The sign strategy holds
$q_t = \sign(\hat\sigma^2_t - \mathrm{{IV}}^2_t)$ straddles; the short-straddle benchmark (Short straddle) holds
$q_t \equiv -1$; every model except the
OLS uses the full predictor set. Mean return is at mid-quotes, in percent of the premium; $\SR$ is the annualized
Sharpe ratio at mid-quotes and at quoted bid/ask prices; hit rate and long days are the shares of days with
$r_t > 0$ at mid-quotes and with $q_t = +1$. $\Delta\SR$ is $\SR$ at mid-quotes minus that of the short-straddle
benchmark (\rheShortSh); in parentheses, the HAC $t$-statistic of the mean return and, under $\Delta\SR$, the {HAC_NOTE}.
{STAR_NOTE}
\item[a] The specification with the highest $\SR$ at mid-quotes among the estimator's \rheTreeSpecs{{}} on the full
predictor set, so selected ex post on the out-of-sample period. {tree_note[0].upper() + tree_note[1:]}.
\item[b] MinGRU network (ridge-initialized, trained and selected on QLIKE; Section~\ref{{sec:rh_methods_models}}).
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""
    )
    return tex, st


# ============================================================================= E2
# 2026-10-15 (table legibility): parked -- E2 with counts in the t columns and code terms; replaced by table_e2 below
# def table_e2(series, fam, model_set) -> tuple[str, dict]:
#     pairs = [
#         (k, k.replace("_baseline", "_all_features"))
#         for k in model_set
#         if "_baseline" in k
#     ]
#     pairs = [(b, a) for b, a in pairs if a in series]
#     ps = {b: pair_stats(series, b, a) for b, a in pairs}
#     nn = pair_stats(series, NN_BASE, NN_ALL)
#
#     def row(label: str, s: dict) -> str:
#         tm, tx = sig(s["t_mid"]), sig(s["t_x"])
#         return (
#             f"{label} & {num(s['sh_b'])} & {num(s['sh_a'])} & {num(s['d_mid'], plus=True, bold=tm)} & "
#             f"{num(s['t_mid'], bold=tm)} & {num(s['d_x'], plus=True, bold=tx)} & {num(s['t_x'], bold=tx)} \\\\\n"
#         )
#
#     def summary(label: str, keys: list[str]) -> str:
#         v = pd.DataFrame([ps[k] for k in keys])
#         n = len(v)
#         return (
#             f"{label} ({n}) & {num(v.sh_b.median())} & {num(v.sh_a.median())} & {num(v.d_mid.median(), plus=True)} & "
#             f"{int((v.t_mid > SIG).sum())}/{n} & {num(v.d_x.median(), plus=True)} & {int((v.t_x > SIG).sum())}/{n} \\\\\n"
#         )
#
#     body = panel("Per-bar linear (16:00 bar)", 7, first=True)
#     for est in ("enet", "lasso", "ridge"):
#         body += row(EST_NAME[est], ps[f"subpt_{est}_baseline"])
#     body += panel("Per-bar neural net", 7)
#     body += row("MinGRU", nn)
#     body += panel("Pooled linear (all 48 bars)", 7)
#     for est in ("enet", "lasso", "ridge"):
#         body += row(EST_NAME[est], ps[f"poolpt_{est}_baseline"])
#     body += panel(
#         r"Per-bar trees: medians over configurations; the $t$ columns count $t > 1.96$",
#         7,
#     )
#     tree_keys = [b for b in ps if fam[b] == "tree"]
#     for est, name in TREE_EST:
#         body += summary(name, [b for b in tree_keys if tree_est(b) == est])
#     body += summary("All trees", tree_keys)
#     body += "\\midrule\n"
#     body += summary("Model set, all pairs", list(ps))
#
#     for est in (
#         "enet",
#         "lasso",
#         "ridge",
#     ):  # the note says the HAR + calendar per-bar fits trade as the OLS
#         assert (series[f"subpt_{est}_baseline"]["q"] == series[OLS]["q"]).all(), est
#     v = pd.DataFrame(ps.values())
#     info = {
#         "n": len(v),
#         "up": int((v.d_mid > 0).sum()),
#         "sig": int((v.t_mid > SIG).sum()),
#         "sig_neg": int((v.t_mid < -SIG).sum()),
#         "up_x": int((v.d_x > 0).sum()),
#         "med": float(v.d_mid.median()),
#         "med_x": float(v.d_x.median()),
#         "n_tree": len(tree_keys),
#         "sig_keys": [b for b in ps if ps[b]["t_mid"] > SIG],
#         "pairs": ps,
#         "nn": nn,
#     }
#     tex = (
#         HEADER
#         + rf"""% Table E2 (outline 7.4): Sharpe change from HAR + calendar to all features, within each family.
# \begin{{table}}[tbp]
# \centering
# \small
# \setlength{{\tabcolsep}}{{5pt}}
# \begin{{threeparttable}}
# \caption{{Adding the exogenous inputs: the Sharpe ratio from HAR + calendar to all features, within each family
# ($q_t = \sign(s_t)$, 866 trade days).}}
# \label{{tab:rh_econ_allfeat}}
# \begin{{tabular}}{{@{{}}lrrrrrr@{{}}}}
# \toprule
#  & \multicolumn{{2}}{{c}}{{Sharpe ratio, mid}} & \multicolumn{{4}}{{c}}{{All features minus HAR + calendar}} \\
# \cmidrule(lr){{2-3}}\cmidrule(l){{4-7}}
#  & HAR + cal. & All & $\Delta$Sharpe mid & $t$ & $\Delta$Sharpe crossed & $t$ \\
# \midrule
# {body}\bottomrule
# \end{{tabular}}
# \begin{{tablenotes}}[flushleft]
# \footnotesize
# \item \textit{{Notes:}} Each pair is one model fitted on HAR + calendar (\texttt{{baseline}}) and on all features,
# with nothing else changed; the change is all features minus HAR + calendar on the same days. The per-bar linear
# models are pre-tuned; on HAR + calendar all three take the OLS's position
# on every day, so they share its Sharpe ratio. The tree rows summarise
# {len(tree_keys)} pairs (every tuning configuration of the model set, {len(tree_keys) // 3} per estimator): the Sharpe
# and $\Delta$Sharpe columns are medians, and each $t$ column gives the number of pairs with $t > 1.96$. The model-set
# row covers the {info["n"]} pairs of the 112-forecast model set (the MinGRU is not in it): $\Delta$Sharpe mid
# $> 0$ in {info["up"]} of {info["n"]}, none significantly negative. {NOTE_FILLS} {NOTE_SHARPE} {NOTE_HAC}
# \end{{tablenotes}}
# \end{{threeparttable}}
# \end{{table}}
# """
#     )
#     assert info["sig_neg"] == 0, info["sig_neg"]
#     return tex, info


def table_e2(series, fam, model_set) -> tuple[str, dict]:
    pairs = [
        (k, k.replace("_baseline", "_all_features"))
        for k in model_set
        if "_baseline" in k
    ]
    pairs = [(b, a) for b, a in pairs if a in series]
    ps = {b: pair_stats(series, b, a) for b, a in pairs}
    nn = pair_stats(series, NN_BASE, NN_ALL)

    def row(label: str, s: dict) -> str:
        top = [
            cell(s["sh_b"], starred=False),
            cell(s["sh_a"], starred=False),
            cell(s["d_mid"], 2, n_stars(s["t_mid"])),
            cell(s["d_x"], 2, n_stars(s["t_x"])),
        ]
        return rows2(label, top, ["", "", tstat(s["t_mid"]), tstat(s["t_x"])])

    def summary(label: str, keys: list[str]) -> str:
        v = pd.DataFrame([ps[k] for k in keys])
        n = len(v)
        top = [
            cell(v.sh_b.median(), starred=False),
            cell(v.sh_a.median(), starred=False),
            cell(v.d_mid.median()),
            cell(v.d_x.median()),
        ]
        under = [
            "",
            "",
            count(int((v.t_mid > SIG).sum()), n),
            count(int((v.t_x > SIG).sum()), n),
        ]
        return rows2(label, top, under)

    tree_keys = [b for b in ps if fam[b] == "tree"]
    n_each = {e: sum(tree_est(b) == e for b in tree_keys) for e, _ in TREE_EST}
    assert len(set(n_each.values())) == 1, (
        n_each
    )  # the notes give one count per estimator
    body = panel_head("Panel A: Closing-interval models", 5, first=True)
    for e in ("enet", "lasso", "ridge"):
        body += row(EST_NAME[e], ps[f"subpt_{e}_baseline"])
    body += row(r"MinGRU\tnote{a}", nn)
    body += panel_head("Panel B: Pooled intraday models", 5)
    for e in ("enet", "lasso", "ridge"):
        body += row(EST_NAME[e], ps[f"poolpt_{e}_baseline"])
    body += panel_head("Panel C: Medians across pairs", 5)
    for e, name in TREE_EST:
        body += summary(
            f"{name} (\\rheTreePairsEach)", [b for b in tree_keys if tree_est(b) == e]
        )
    body += summary(r"All trees (\rheTreePairs)", tree_keys)
    body += summary(r"Model set (\rhePairs)", list(ps))

    for e in (
        "enet",
        "lasso",
        "ridge",
    ):  # the note says the HAR-benchmark closing-interval fits trade as the OLS
        assert (series[f"subpt_{e}_baseline"]["q"] == series[OLS]["q"]).all(), e
    v = pd.DataFrame(ps.values())
    info = {
        "n": len(v),
        "up": int((v.d_mid > 0).sum()),
        "sig": int((v.t_mid > SIG).sum()),
        "sig_neg": int((v.t_mid < -SIG).sum()),
        "up_x": int((v.d_x > 0).sum()),
        "med": float(v.d_mid.median()),
        "med_x": float(v.d_x.median()),
        "n_tree": len(tree_keys),
        "n_tree_each": n_each["lgbm"],
        "sig_keys": [b for b in ps if ps[b]["t_mid"] > SIG],
        "pairs": ps,
        "nn": nn,
    }
    tex = (
        float_head("Table E2", "the full predictor set against the HAR benchmark")
        + rf"""\begin{{table}}[tbp]
\centering
\small
\begin{{threeparttable}}
\caption{{Sharpe ratios: HAR benchmark versus full predictor set}}
\label{{tab:rh_econ_allfeat}}
\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\toprule
 & \multicolumn{{2}}{{c}}{{$\SR$ at mid-quotes}} & \multicolumn{{2}}{{c}}{{$\Delta\SR$}} \\
\cmidrule(lr){{2-3}}\cmidrule(l){{4-5}}
 & HAR benchmark & Full set & Mid-quotes & Bid--ask \\
\midrule
{body}\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item \textit{{Notes:}} Each pair is one model estimated on the HAR benchmark and on the full predictor set, with
everything else unchanged, over the out-of-sample period ($T_{{\mathrm{{oos}}}} = \rheNdays$ trading days). $\SR$ is the
annualized Sharpe ratio of the sign strategy; $\Delta\SR$ is the full-set minus the HAR-benchmark $\SR$, at
mid-quotes and at quoted bid/ask prices, with the {HAC_NOTE} in parentheses. On the HAR benchmark the three
closing-interval linear models hold the same position as the closing-interval OLS on every day, so they share its
$\SR$. Panel C gives medians across the closing-interval tree pairs (every hyperparameter specification in the model
set) and across all pairs of the \rheModelSetN{{}} forecasts of the model set, with the number of pairs whose
differential is significantly positive at the 5\% level in brackets (none is significantly negative at
that level). {STAR_NOTE}
\item[a] MinGRU network (ridge-initialized, trained and selected on QLIKE); not in the model set.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""
    )
    assert info["sig_neg"] == 0, info["sig_neg"]
    assert (
        int((v.t_x < -SIG).sum()) == 0
    )  # the note's "none is significantly negative", at bid/ask prices too
    return tex, info


# ============================================================================= E3
# 2026-10-15 (table legibility): parked -- 9-column E3; replaced by table_e3 below (t and DM under the estimates)
# def table_e3(series) -> tuple[str, list]:
#     rows_ = []
#     body = ""
#     for set_, set_label in (
#         ("all_features", "All features"),
#         ("baseline", "HAR + calendar"),
#     ):
#         body += panel(set_label, 9, first=set_ == "all_features")
#         for est in ("enet", "lasso", "ridge"):
#             p, b = series[f"poolpt_{est}_{set_}"], series[f"subpt_{est}_{set_}"]
#             s = pair_stats(series, f"poolpt_{est}_{set_}", f"subpt_{est}_{set_}")
#             dq = 100 * (b["qlike"].mean() / p["qlike"].mean() - 1)
#             dm = dm_test(b["qlike"].to_numpy(float), p["qlike"].to_numpy(float))["dm"]
#             tm, tx, td = sig(s["t_mid"]), sig(s["t_x"]), sig(dm)
#             body += (
#                 f"{EST_NAME[est]} & {num(s['sh_b'])} & {num(s['sh_a'])} & {num(s['d_mid'], plus=True, bold=tm)} & "
#                 f"{num(s['t_mid'], bold=tm)} & {num(s['d_x'], plus=True, bold=tx)} & {num(s['t_x'], bold=tx)} & "
#                 f"{num(dq, 1, plus=True, bold=td)} & {num(dm, bold=td)} \\\\\n"
#             )
#             rows_.append(dict(est=est, set=set_, **s, dq=dq, dm=dm))
#     tex = (
#         HEADER
#         + rf"""% Table E3 (outline 7.4): pooled (48 bars) vs per-bar (16:00 bar) linear fits, the trade and QLIKE.
# \begin{{table}}[tbp]
# \centering
# \small
# \setlength{{\tabcolsep}}{{4.5pt}}
# \begin{{threeparttable}}
# \caption{{Per-bar against pooled linear fits: the trade and accuracy ($q_t = \sign(s_t)$, 866 trade days).}}
# \label{{tab:rh_econ_pooled}}
# \begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}
# \toprule
#  & \multicolumn{{2}}{{c}}{{Sharpe ratio, mid}} & \multicolumn{{6}}{{c}}{{Per-bar minus pooled}} \\
# \cmidrule(lr){{2-3}}\cmidrule(l){{4-9}}
#  & Pooled & Per-bar & $\Delta$Sharpe mid & $t$ & $\Delta$Sharpe crossed & $t$ & $\Delta$QLIKE (\%) & DM \\
# \midrule
# {body}\bottomrule
# \end{{tabular}}
# \begin{{tablenotes}}[flushleft]
# \footnotesize
# \item \textit{{Notes:}} Pooled: one model fitted on all 48 half-hour bars of the day, read at the 16:00 bar; per-bar:
# fitted on the 16:00 bar alone; both pre-tuned, on the same inputs and days. $\Delta$QLIKE: per-bar mean QLIKE over
# pooled, minus 1, in \%; DM: Diebold--Mariano statistic of the daily QLIKE difference (Harvey--Leybourne--Newbold
# corrected, automatic Newey--West lag), negative when the per-bar fit is more accurate. {NOTE_FILLS} {NOTE_SHARPE}
# {NOTE_HAC} Bold DM: $|\mathrm{{DM}}| > 1.96$.
# \end{{tablenotes}}
# \end{{threeparttable}}
# \end{{table}}
# """
#     )
#     return tex, rows_


def table_e3(series) -> tuple[str, list]:
    rows_ = []
    body = ""
    for set_, set_label in (
        ("all_features", "Panel A: Full predictor set"),
        ("baseline", "Panel B: HAR benchmark"),
    ):
        body += panel_head(set_label, 6, first=set_ == "all_features")
        for e in ("enet", "lasso", "ridge"):
            p, b = series[f"poolpt_{e}_{set_}"], series[f"subpt_{e}_{set_}"]
            # 2026-10-15 (consolidation): parked -- closing-interval minus pooled intraday; the paper's one convention
            # is now pooled intraday minus closing-interval, as in Table T2 (tab:rh_axis2) and Results.
            # s = pair_stats(series, f"poolpt_{e}_{set_}", f"subpt_{e}_{set_}")
            # dq = 100 * (b["qlike"].mean() / p["qlike"].mean() - 1)
            # dmr = dm_test(b["qlike"].to_numpy(float), p["qlike"].to_numpy(float))
            s = pair_stats(series, f"subpt_{e}_{set_}", f"poolpt_{e}_{set_}")
            dq = 100 * (p["qlike"].mean() / b["qlike"].mean() - 1)
            dmr = dm_test(p["qlike"].to_numpy(float), b["qlike"].to_numpy(float))
            dm = dmr["dm"]
            top = [
                cell(
                    s["sh_a"], starred=False
                ),  # pooled intraday (the "new" leg of the pair)
                cell(s["sh_b"], starred=False),  # closing interval
                cell(s["d_mid"], 2, n_stars(s["t_mid"])),
                cell(s["d_x"], 2, n_stars(s["t_x"])),
                cell(dq, 1, int(sum(dmr["p"] < a for a in LEVELS))),
            ]
            under = ["", "", tstat(s["t_mid"]), tstat(s["t_x"]), tstat(dm)]
            body += rows2(EST_NAME[e], top, under)
            rows_.append(
                dict(est=e, set=set_, **s, dq=dq, dm=dm, dm_lag=dmr["hac_lag"])
            )
    assert len({r["dm_lag"] for r in rows_}) == 1, rows_  # one lag in the notes
    tex = (
        float_head("Table E3", "closing-interval against pooled intraday linear models")
        + rf"""\begin{{table}}[tbp]
\centering
\small
\begin{{threeparttable}}
\caption{{Closing-interval versus pooled intraday models}}
\label{{tab:rh_econ_pooled}}
\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\toprule
 & \multicolumn{{2}}{{c}}{{$\SR$ at mid-quotes}} & \multicolumn{{3}}{{c}}{{Pooled intraday minus closing-interval}} \\
\cmidrule(lr){{2-3}}\cmidrule(l){{4-6}}
 & Pooled & Closing & $\Delta\SR$, & $\Delta\SR$, & \\
 & intraday & interval & mid-quotes & bid--ask & $\Delta\QLIKE$ (\%) \\
\midrule
{body}\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item \textit{{Notes:}} Each row compares the closing-interval model with the pooled intraday model of the same
estimator and predictor set, both with hyperparameters selected on a pre-sample, over the out-of-sample period
($T_{{\mathrm{{oos}}}} = \rheNdays$ trading days); the pooled intraday model is evaluated on the closing interval. $\SR$
is the annualized Sharpe ratio of the sign strategy; $\Delta\SR$ is the pooled intraday minus the closing-interval
$\SR$, at mid-quotes and at quoted bid/ask prices, with the {HAC_NOTE} in parentheses. $\Delta\QLIKE$ is the
pooled intraday model's mean QLIKE relative to the closing-interval model's, in percent, with the Diebold--Mariano
statistic (Harvey--Leybourne--Newbold correction, Newey--West $q_{{\mathrm{{NW}}}} = \rheDMLag$) in parentheses; positive values favor
the closing-interval model, as in Table~\ref{{tab:rh_axis2}}. {STAR_NOTE}
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""
    )
    return tex, rows_


# ============================================================================= E4
RULES = (
    ("sign", r"$\sign(s)$"),
    ("clip", "Clip"),
    ("lfss", "Long/scaled"),
    ("rank", r"Rank $|s|$"),
)


# 2026-10-15 (table legibility): parked -- 9-column E4 (mid and crossed side by side); replaced by table_e4 below (Panel A mid, Panel B bid-ask)
# def table_e4(series, model_set, trees) -> tuple[str, dict]:
#     sz = {k: sizing_stats(series, k) for k in model_set}
#     for k in [OLS, *PERBAR_ALL, *trees.values(), NN_ALL, *POOLED_ALL]:
#         sz.setdefault(k, sizing_stats(series, k))
#     short_mid = sharpe(series[SHORT]["mid"].to_numpy()[WARMUP:])
#     short_x = sharpe(series[SHORT]["x"].to_numpy()[WARMUP:])
#     n_days = len(series[SHORT]) - WARMUP
#
#     def row(label: str, key: str) -> str:
#         s = sz[key]
#         mids = [num(s[r]["sharpe_mid"]) for r, _ in RULES]
#         xs = [num(s[r]["sharpe_crossed"]) for r, _ in RULES]
#         return f"{label} & " + " & ".join(mids + xs) + " \\\\\n"
#
#     body = (
#         rf"Always short ($q_t = -1$) & \multicolumn{{4}}{{c}}{{{num(short_mid)} under every rule}} & "
#         rf"\multicolumn{{4}}{{c}}{{{num(short_x)} under every rule}} \\" + "\n"
#     )
#     body += panel("Per-bar linear (16:00 bar)", 9)
#     body += row("OLS, HAR + calendar", OLS)
#     for k in PERBAR_ALL:
#         body += row(f"{EST_NAME[k.split('_')[1]]}, all features", k)
#     body += panel(
#         "Per-bar trees, all features (as in Table~\\ref{tab:rh_econ_control})", 9
#     )
#     for est, name in TREE_EST:
#         body += row(name, trees[est])
#     body += panel("Per-bar neural net", 9)
#     body += row("MinGRU, all features", NN_ALL)
#     body += panel("Pooled linear (all 48 bars)", 9)
#     for k in POOLED_ALL:
#         body += row(f"{EST_NAME[k.split('_')[1]]}, all features", k)
#
#     ms = {r: [sz[k][r] for k in model_set] for r, _ in RULES}
#     med_mid = {r: float(np.median([v["sharpe_mid"] for v in ms[r]])) for r, _ in RULES}
#     med_x = {
#         r: float(np.median([v["sharpe_crossed"] for v in ms[r]])) for r, _ in RULES
#     }
#     beat_mid = {
#         r: int(sum(v["d_vs_sign"] > 0 for v in ms[r])) for r, _ in RULES if r != "sign"
#     }
#     beat_x = {
#         r: int(sum(v["d_vs_sign_x"] > 0 for v in ms[r]))
#         for r, _ in RULES
#         if r != "sign"
#     }
#     med_gain = {
#         r: float(np.median([v["d_vs_sign"] for v in ms[r]]))
#         for r, _ in RULES
#         if r != "sign"
#     }
#     n = len(model_set)
#     body += "\\midrule\n"
#     body += (
#         f"Model set ({n}): median & "
#         + " & ".join(
#             [num(med_mid[r]) for r, _ in RULES] + [num(med_x[r]) for r, _ in RULES]
#         )
#         + " \\\\\n"
#     )
#     body += (
#         "Model set: above $\\sign(s)$ & -- & "
#         + " & ".join(f"{beat_mid[r]}" for r, _ in RULES if r != "sign")
#         + " & -- & "
#         + " & ".join(f"{beat_x[r]}" for r, _ in RULES if r != "sign")
#         + " \\\\\n"
#     )
#     info = dict(
#         med_mid=med_mid,
#         med_x=med_x,
#         beat_mid=beat_mid,
#         beat_x=beat_x,
#         med_gain=med_gain,
#         n=n,
#         short_mid=short_mid,
#         short_x=short_x,
#         n_days=n_days,
#         sz=sz,
#     )
#     head = " & ".join(lab for _, lab in RULES)
#     tex = (
#         HEADER
#         + rf"""% Table E4 (outline 7.4): sizing rules, scored on days 253-866 exactly as the GUI (rows._score_sizes).
# \begin{{table}}[tbp]
# \centering
# \small
# \setlength{{\tabcolsep}}{{4pt}}
# \begin{{threeparttable}}
# \caption{{Sizing rules: Sharpe ratio on days {WARMUP + 1}--{WARMUP + n_days} ({n_days} days).}}
# \label{{tab:rh_econ_sizing}}
# \begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}
# \toprule
#  & \multicolumn{{4}}{{c}}{{Mid}} & \multicolumn{{4}}{{c}}{{Crossed}} \\
# \cmidrule(lr){{2-5}}\cmidrule(l){{6-9}}
# Forecast & {head} & {head} \\
# \midrule
# {body}\bottomrule
# \end{{tabular}}
# \begin{{tablenotes}}[flushleft]
# \footnotesize
# \item \textit{{Notes:}} Position $q_t$ in straddles, from the gap $s_t = \hat\sigma^2_t - (IV^{{30}}_t)^2$ and the
# earlier days only: $\sign(s)$, $q_t = \pm 1$; Clip (robust clip), $q_t = \clip(s_t/m_t, -1, 1)$ with $m_t$ the median
# $|s_u|$ over $u < t$; Long/scaled (full long, scaled short), $q_t = 1$ if $s_t > 0$, else $\max(s_t/m_t, -1)$; Rank
# $|s|$, $q_t = \sign(s_t)$ times the share of earlier days with $|s_u| \le |s_t|$. The first {WARMUP} days only build the
# scale, so every rule, $\sign(s)$ included, is scored on the last {n_days} of the 866 days. Crossed: the spread is paid
# on $|q_t|$ straddles. Model-set rows: the median over the {n} forecasts of the model set (the MinGRU is not in it), and
# the number of them whose Sharpe ratio is above their own $\sign(s)$ Sharpe ratio. Point estimates; no test.
# {NOTE_SHARPE}
# \end{{tablenotes}}
# \end{{threeparttable}}
# \end{{table}}
# """
#     )
#     return tex, info


def table_e4(series, model_set, trees) -> tuple[str, dict]:
    sz = {k: sizing_stats(series, k) for k in model_set}
    for k in [OLS, *PERBAR_ALL, *trees.values(), NN_ALL, *POOLED_ALL]:
        sz.setdefault(k, sizing_stats(series, k))
    short_mid = sharpe(series[SHORT]["mid"].to_numpy()[WARMUP:])
    short_x = sharpe(series[SHORT]["x"].to_numpy()[WARMUP:])
    n_days = len(series[SHORT]) - WARMUP

    ms = {r: [sz[k][r] for k in model_set] for r, _ in RULES}
    med_mid = {r: float(np.median([v["sharpe_mid"] for v in ms[r]])) for r, _ in RULES}
    med_x = {
        r: float(np.median([v["sharpe_crossed"] for v in ms[r]])) for r, _ in RULES
    }
    beat_mid = {
        r: int(sum(v["d_vs_sign"] > 0 for v in ms[r])) for r, _ in RULES if r != "sign"
    }
    beat_x = {
        r: int(sum(v["d_vs_sign_x"] > 0 for v in ms[r]))
        for r, _ in RULES
        if r != "sign"
    }
    med_gain = {
        r: float(np.median([v["d_vs_sign"] for v in ms[r]]))
        for r, _ in RULES
        if r != "sign"
    }
    n = len(model_set)

    def block(field: str, short_v: float, med: dict, beat: dict) -> str:
        def row(label: str, key: str) -> str:
            cells = [cell(sz[key][r][field], starred=False) for r, _ in RULES]
            return f"\\quad {label} & " + " & ".join(cells) + " \\\\\n"

        out = (
            rf"{SHORT_LABEL} & \multicolumn{{4}}{{c}}{{{cell(short_v, starred=False)} under every rule}} \\"
            + "\n"
        )
        out += sub_head("Closing-interval models", 5)
        out += row("OLS, HAR benchmark", OLS)
        for k in PERBAR_ALL:
            out += row(EST_NAME[k.split("_")[1]], k)
        for e, name in TREE_EST:
            out += row(name, trees[e])
        out += row("MinGRU", NN_ALL)
        out += sub_head("Pooled intraday models", 5)
        for k in POOLED_ALL:
            out += row(EST_NAME[k.split("_")[1]], k)
        out += "\\addlinespace[2pt]\n"
        out += (
            r"Model set, median & "
            + " & ".join(cell(med[r], starred=False) for r, _ in RULES)
            + " \\\\\n"
        )
        out += (
            r"Model set, number above sign & -- & "
            + " & ".join(f"{beat[r]}" for r, _ in RULES if r != "sign")
            + " \\\\\n"
        )
        return out

    body = panel_head("Panel A: $\\SR$ at mid-quotes", 5, first=True)
    body += block("sharpe_mid", short_mid, med_mid, beat_mid)
    body += panel_head("Panel B: $\\SR$ at quoted bid/ask prices", 5)
    body += block("sharpe_crossed", short_x, med_x, beat_x)

    info = dict(
        med_mid=med_mid,
        med_x=med_x,
        beat_mid=beat_mid,
        beat_x=beat_x,
        med_gain=med_gain,
        n=n,
        short_mid=short_mid,
        short_x=short_x,
        n_days=n_days,
        sz=sz,
    )
    tex = (
        float_head(
            "Table E4",
            "position-sizing rules, scored exactly as the GUI (rows._score_sizes)",
        )
        + rf"""\begin{{table}}[tbp]
\centering
\small
\begin{{threeparttable}}
\caption{{Sharpe ratios by position-sizing rule}}
\label{{tab:rh_econ_sizing}}
\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\toprule
 & & Robust & Full long, & \\
 & Sign & clip & scaled short & Rank \\
\midrule
{body}\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item \textit{{Notes:}} Annualized Sharpe ratios of the straddle position $q_t$ under four sizing rules
(Section~\ref{{sec:rh_econ_strategy}}), each a function of $\hat\sigma^2_t - \mathrm{{IV}}^2_t$ and of its values on
earlier days $t' < t$ only; the sign rule is the sign strategy, and the short-straddle benchmark (Short straddle)
holds $q_t \equiv -1$ under every rule. The first \rheSzWarmup{{}} trading days of the out-of-sample period only estimate the
scale of $|\hat\sigma^2_{{t'}} - \mathrm{{IV}}^2_{{t'}}|$, so every rule is evaluated on the remaining \rheSzDays{{}} of
the \rheNdays{{}} trading days; at quoted bid/ask prices the bid--ask spread is paid on $|q_t|$ straddles. Every model except the OLS uses the full predictor set; the trees are
the specifications of Table~\ref{{tab:rh_econ_control}}. The last two rows of each panel give the median across the
\rheModelSetN{{}} forecasts of the model set (the MinGRU is not in it) and the number of them whose $\SR$ is above that of
their own sign strategy. Point estimates; no test is reported.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""
    )
    return tex, info


# ============================================================================= figures
PREVIEW_DIR: Path | None = (
    None  # --preview DIR: also write PNG previews there (never into writeup/figures)
)


def _preview(fig, stem: str) -> None:
    if PREVIEW_DIR is not None:
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREVIEW_DIR / f"{stem}.png", dpi=200)


def legend_handle(fam: str, marker: bool = True):
    from matplotlib.lines import Line2D

    short = fam == "short"
    return Line2D(
        [],
        [],
        color=FAM_COLOR[fam],
        lw=1.1,
        ls="--" if short else "-",
        marker=FAM_MARKER[fam] if marker else None,
        ms=4.5 if short else 4,
        mfc=FAM_COLOR[fam],
        mec=FAM_COLOR[fam] if short else "white",
        mew=1.0 if short else 0.5,
    )


def _style(ax, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, lw=0.4, alpha=0.8)
    ax.set_axisbelow(True)


def _save(fig, stem: str) -> None:
    w, h = fig.get_size_inches()
    assert abs(w - FIG_W) < 1e-9, (
        stem,
        w,
    )  # placed at \linewidth = 100 %, so 9 pt stays 9 pt
    fig.savefig(FIG / f"{stem}.pdf")
    _preview(fig, stem)
    plt.close(fig)
    print(f"wrote writeup/figures/{stem}.pdf ({w:.2f} x {h:.2f} in)")


def fig_cumpnl(series, key4: dict) -> dict:
    """Top: cumulative mid return for always short and one forecast per family. Bottom: cumulative gain over always
    short, each forecast's 10 best days of gain marked."""
    from matplotlib.lines import Line2D

    days = series[SHORT].index
    short = series[SHORT]["mid"]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(FIG_W, 4.4), sharex=True, layout="constrained"
    )
    for ax in (ax1, ax2):
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
        _style(ax)
    ax1.plot(days, short.cumsum(), color=FAM_COLOR["short"], lw=1.1, ls="--", zorder=3)
    info = {}
    for fam_, (key, _label) in key4.items():
        g = series[key]
        c = FAM_COLOR[fam_]
        ax1.plot(
            days,
            g["mid"].cumsum(),
            color=c,
            lw=1.1,
            zorder=4,
            marker=FAM_MARKER[fam_],
            markevery=(40, 120),
            ms=3.5,
            mec="white",
            mew=0.4,
        )
        edge = g["mid"] - short
        ce = edge.cumsum()
        ax2.plot(days, ce, color=c, lw=1.1, zorder=4)
        top = edge.nlargest(TOP_DAYS).index
        assert (g.loc[top, "q"] > 0).all(), key  # a gain over always short needs a buy
        ax2.plot(
            top,
            ce.loc[top],
            ls="none",
            marker=FAM_MARKER[fam_],
            ms=4.5,
            mfc=c,
            mec="white",
            mew=0.5,
            zorder=5,
        )
        info[key] = {
            "edge": float(edge.sum()),
            "edge_ex10": float(edge.sum() - edge.nlargest(TOP_DAYS).sum()),
            "final": float(g["mid"].sum()),
            "minR_top10": float(g.loc[top, "R"].min()),
        }
    info[SHORT] = {"final": float(short.sum())}
    # ax1.set_ylabel("Cumulative return\n(premia, mid)")  # 2026-10-15 (table legibility): parked -- guide terms
    # ax2.set_ylabel("Cumulative gain over\nalways short (premia)")  # 2026-10-15 (table legibility): parked -- ditto
    ax1.set_ylabel("Cumulative return\n(straddle premia, mid-quotes)")
    # 2026-10-15 (consolidation): parked -- one name for the q_t = -1 benchmark (guide section 6b)
    # ax2.set_ylabel("Cumulative gain over always\nshort (straddle premia)")
    ax2.set_ylabel("Cumulative gain over the short\nstraddle (straddle premia)")
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.set_xlim(days[0] - pd.Timedelta(days=10), days[-1] + pd.Timedelta(days=10))
    handles = [legend_handle(f, marker=f != "short") for f in key4] + [
        legend_handle("short", marker=False)
    ]
    labels = [lab for _, (_, lab) in key4.items()] + [SHORT_LABEL]
    # 2026-10-15 (table legibility): one row of five no longer fits the longer model-class labels; three columns,
    # read row by row (matplotlib fills legends by column, so interleave)
    order_ = [0, 3, 1, 4, 2]
    fig.legend(
        [handles[i] for i in order_],
        [labels[i] for i in order_],
        loc="outside upper center",
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
    )
    ax2.legend(
        [
            Line2D(
                [],
                [],
                ls="none",
                marker="o",
                ms=4.5,
                mfc="#888888",
                mec="white",
                mew=0.5,
            )
        ],
        # ["each forecast's 10 best days of gain"],  # 2026-10-15 (table legibility): parked -- wording
        [f"each model's {TOP_DAYS} days of largest gain"],
        loc="upper left",
        frameon=False,
        handletextpad=0.3,
    )
    fig.align_ylabels((ax1, ax2))
    _save(fig, "rehaul_econ_cumpnl")
    return info


def fig_sizing(
    series, fam, model_set, sz: dict, short_mid: float, short_x: float, med_mid, med_x
) -> None:
    from matplotlib.lines import Line2D

    rule_names = [
        "$\\mathrm{sign}(s)$",
        "Robust clip",
        "Full long,\nscaled short",
        "Rank of $|s|$",
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_W, 2.9), sharey=True, layout="constrained"
    )
    rng = np.random.default_rng(20261015)
    jit = {k: rng.uniform(-0.26, 0.26) for k in model_set}
    order = [
        "tree",
        "pooled",
        "linear",
    ]  # trees (90) first, so the 22 linear and pooled forecasts stay visible
    for ax, field, shv, med in (
        (axes[0], "sharpe_mid", short_mid, med_mid),
        (axes[1], "sharpe_crossed", short_x, med_x),
    ):
        _style(ax, "x")
        for i, (r, _) in enumerate(RULES):
            y0 = len(RULES) - 1 - i
            for f in order:
                ks = [k for k in model_set if fam[k] == f]
                ax.scatter(
                    [sz[k][r][field] for k in ks],
                    [y0 + jit[k] for k in ks],
                    s=12,
                    color=FAM_COLOR[f],
                    marker=FAM_MARKER[f],
                    edgecolors="white",
                    linewidths=0.35,
                    alpha=0.9,
                    zorder=3,
                )
            v = sz[NN_ALL][r][field]
            ax.scatter(
                [v],
                [y0],
                s=30,
                color=FAM_COLOR["nn"],
                marker=FAM_MARKER["nn"],
                edgecolors="white",
                linewidths=0.6,
                zorder=6,
            )
            ax.plot(
                [med[r], med[r]],
                [y0 - 0.34, y0 + 0.34],
                color="#111111",
                lw=1.8,
                zorder=5,
                solid_capstyle="butt",
            )
            ax.text(
                med[r],
                y0 + 0.37,
                f"{med[r]:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#111111",
                zorder=7,
                bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.9),
            )
        ax.axvline(shv, color=FAM_COLOR["short"], lw=0.9, ls="--", zorder=2)
        ax.set_ylim(-0.5, len(RULES) - 0.2)
    axes[0].set_yticks(range(len(RULES)))
    axes[0].set_yticklabels(rule_names[::-1])
    axes[0].tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", length=0)
    axes[0].set_xlabel("Sharpe ratio, mid")
    axes[1].set_xlabel("Sharpe ratio, crossed")
    handles = [
        Line2D(
            [],
            [],
            ls="none",
            marker=FAM_MARKER[f],
            ms=4.5,
            mfc=FAM_COLOR[f],
            mec="white",
            mew=0.35,
        )
        for f in ("linear", "tree", "pooled")
    ] + [
        Line2D(
            [],
            [],
            ls="none",
            marker=FAM_MARKER["nn"],
            ms=5.5,
            mfc=FAM_COLOR["nn"],
            mec="white",
            mew=0.6,
        ),
        Line2D([], [], color="#111111", lw=1.8),
        Line2D([], [], color=FAM_COLOR["short"], lw=0.9, ls="--"),
    ]
    labels = [
        FAM_LABEL["linear"],
        FAM_LABEL["tree"],
        FAM_LABEL["pooled"],
        "Per-bar MinGRU (not in the set)",
        "Model-set median",
        SHORT_LABEL,
    ]
    # three columns, read row by row: matplotlib fills legends by column, so interleave
    order_ = [0, 3, 1, 4, 2, 5]
    fig.legend(
        [handles[i] for i in order_],
        [labels[i] for i in order_],
        loc="outside upper center",
        ncol=3,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.6,
    )
    _save(fig, "rehaul_econ_sizing")


def fig_byyear(series, model_set, key4: dict) -> dict:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    days = series[SHORT].index
    years = sorted(set(days.year))
    yr = days.year

    def by_year(x: pd.Series) -> list[float]:
        return [sharpe(x[yr == y]) for y in years]

    set_by = np.array([by_year(series[k]["mid"]) for k in model_set])
    q25, q50, q75 = np.percentile(set_by, [25, 50, 75], axis=0)
    fig, ax = plt.subplots(figsize=(FIG_W, 2.9), layout="constrained")
    xs = np.arange(len(years))
    ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax.fill_between(xs, q25, q75, color="#e8e8e8", lw=0, zorder=1)
    ax.plot(xs, q50, color="#9a9a9a", lw=0.9, zorder=1.5)
    out = {
        "years": years,
        "set_med": q50.tolist(),
        "set_q25": q25.tolist(),
        "set_q75": q75.tolist(),
    }
    fams = list(key4)
    offs = np.linspace(-0.16, 0.16, len(fams) + 1)
    sh_short = by_year(series[SHORT]["mid"])
    out[SHORT] = sh_short
    ax.plot(
        xs + offs[-1],
        sh_short,
        color=FAM_COLOR["short"],
        lw=1.0,
        ls="--",
        marker=FAM_MARKER["short"],
        ms=4.5,
        mew=1.0,
        zorder=3,
    )
    for j, f in enumerate(fams):
        key = key4[f][0]
        v = by_year(series[key]["mid"])
        out[key] = v
        ax.plot(
            xs + offs[j],
            v,
            color=FAM_COLOR[f],
            lw=1.0,
            marker=FAM_MARKER[f],
            ms=4.5,
            mec="white",
            mew=0.5,
            zorder=4,
        )
    n_by = [int((yr == y).sum()) for y in years]
    labs = [f"{y}\n({n} days)" for y, n in zip(years, n_by)]
    labs[-1] = f"{years[-1]}, Jan.–Apr.\n({n_by[-1]} days)"
    ax.set_xticks(xs)
    ax.set_xticklabels(labs)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, len(years) - 0.5)
    ax.set_ylabel("Sharpe ratio, mid, within year")
    _style(ax)
    handles = (
        [legend_handle(f) for f in fams]
        + [legend_handle("short")]
        + [Patch(color="#e8e8e8", lw=0), Line2D([], [], color="#9a9a9a", lw=0.9)]
    )
    labels = [key4[f][1] for f in fams] + [
        SHORT_LABEL,
        "Model set: interquartile range",
        "Model set: median",
    ]
    # four columns, read row by row (matplotlib fills by column): row 1 the forecasts, row 2 the references
    order_ = [0, 4, 1, 5, 2, 6, 3]
    fig.legend(
        [handles[i] for i in order_],
        [labels[i] for i in order_],
        loc="outside upper center",
        ncol=4,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
    )
    out["n_by"] = n_by
    _save(fig, "rehaul_econ_byyear")
    return out


# ============================================================================= main
def main() -> None:
    mt, model_set, series, fam = load()
    gate_against_master(mt, series, [SHORT, *model_set])

    trees = {est: best_tree(series, fam, model_set, est) for est, _ in TREE_EST}
    e1, st = table_e1(series, fam, model_set, trees)
    e2, e2i = table_e2(series, fam, model_set)
    e3, e3r = table_e3(series)
    e4, e4i = table_e4(series, model_set, trees)

    # ---- the asserts the outline fixes (rounded as printed)
    r2 = lambda x: round(x + 0.0, 2)  # noqa: E731
    assert (r2(st[SHORT]["sh_mid"]), r2(st[SHORT]["sh_x"])) == (0.20, -0.27), st[SHORT]
    assert (
        r2(st["subpt_enet_all_features"]["sh_mid"]),
        r2(st["subpt_enet_all_features"]["sh_x"]),
    ) == (1.92, 1.43)
    assert (r2(st[NN_ALL]["sh_mid"]), r2(st[NN_ALL]["sh_x"])) == (1.53, 1.05), st[
        NN_ALL
    ]
    nb = trade_stats(series, NN_BASE)
    assert (r2(nb["sh_mid"]), r2(nb["sh_x"])) == (1.04, 0.57), nb
    meds = tuple(r2(e4i["med_mid"][r]) for r, _ in RULES)
    assert meds == (0.92, 1.34, 1.12, 1.34), meds
    # outline cross-checks
    assert (
        e2i["n"] == 51
        and e2i["up"] == 50
        and e2i["sig"] == 4
        and r2(e2i["med"]) == 0.54
    ), e2i
    assert e4i["beat_mid"]["clip"] == 98, e4i["beat_mid"]
    assert r2(e4i["short_mid"]) == 0.34, e4i["short_mid"]
    sig_vs_short = sum(trade_stats(series, k)["t_dsh"] > SIG for k in model_set)
    assert sig_vs_short == 20, sig_vs_short

    # one forecast per family for the figures
    # 2026-10-15 (table legibility): parked -- code terms (per-bar); the guide's model classes below
    # key4 = {
    #     "linear": ("subpt_enet_all_features", "Per-bar elastic net"),
    #     "tree": (trees["lgbm"], "Per-bar LightGBM"),
    #     "nn": (NN_ALL, "Per-bar MinGRU"),
    #     "pooled": ("poolpt_enet_all_features", "Pooled elastic net"),
    # }
    key4 = {
        "linear": ("subpt_enet_all_features", "Closing-interval elastic net"),
        "tree": (trees["lgbm"], "Closing-interval LightGBM"),
        "nn": (NN_ALL, "Closing-interval MinGRU"),
        "pooled": ("poolpt_enet_all_features", "Pooled intraday elastic net"),
    }
    if sharpe(series[trees["xgb"]]["mid"]) > sharpe(series[trees["lgbm"]]["mid"]):
        raise ValueError("the best tree is not the LightGBM: relabel key4['tree']")
    cum = fig_cumpnl(series, key4)
    fig_sizing(
        series,
        fam,
        model_set,
        e4i["sz"],
        e4i["short_mid"],
        e4i["short_x"],
        e4i["med_mid"],
        e4i["med_x"],
    )
    yrs = fig_byyear(series, model_set, key4)

    # tail: the gain over always short without each forecast's 10 best days, over the model set
    neg_ex10 = 0
    for k in model_set:
        edge = series[k]["mid"] - series[SHORT]["mid"]
        neg_ex10 += int(edge.sum() - edge.nlargest(TOP_DAYS).sum() < 0)

    # ---- figure floats
    # 2026-10-15 (table legibility): parked -- the caption's numbers are now \rheGain... macros (see below)
    # def ex10(f):
    #     return num(cum[key4[f][0]]["edge_ex10"], 1, plus=True)
    #
    # def tot(f):
    #     return num(cum[key4[f][0]]["edge"], 1, plus=True)

    fam_order = ("linear", "tree", "nn", "pooled")
    min_r = min(cum[key4[f][0]]["minR_top10"] for f in fam_order)
    # 2026-10-15 (table legibility): parked -- code terms (closing trade, sign(s), at mid, per-bar, family) and typed
    # numbers; the caption below uses the guide's terms and the \rhe... macros
    # lab = {
    #     "linear": "per-bar elastic net",
    #     "tree": "per-bar LightGBM",
    #     "nn": "per-bar MinGRU",
    #     "pooled": "pooled elastic net",
    # }
    # totals = ", ".join(f"{lab[f]} {tot(f)}" for f in fam_order)
    # ex10s = ", ".join(f"{ex10(f)}" for f in fam_order)
    # fig1 = (
    #     HEADER
    #     + rf"""\begin{{figure}}[tbp]
    # \centering
    # \includegraphics[width=\linewidth]{{figures/rehaul_econ_cumpnl.pdf}}
    # \caption{{The closing trade over time, $q_t = \sign(s_t)$ at mid, for one forecast per family (all features) and
    # always short. Top: cumulative return $\sum_{{u \le t}} r_u$, in units of the straddle premium. Bottom: cumulative gain
    # over always short, $\sum_{{u \le t}} \bigl(r_u - r^{{\mathrm{{short}}}}_u\bigr)$; the markers are each forecast's 10
    # days of largest gain, each a day on which it bought and $R_t \ge {min_r:.2f}$. The gain over the 866 days is
    # {totals}; without each forecast's 10 best days it is {ex10s}. For {neg_ex10} of the 112 forecasts of the model set the
    # gain is negative without their 10 best days.}}
    # \label{{fig:rh_econ_cumpnl}}
    # \end{{figure}}
    # """
    # )
    fig1 = (
        HEADER
        + NUMBERS_GUARD
        + r"""\begin{figure}[tbp]
\centering
\includegraphics[width=\linewidth]{figures/rehaul_econ_cumpnl.pdf}
\caption{Cumulative returns of the sign strategy. The top panel plots the cumulative return at mid-quotes,
$\sum_{t' \le t} r_{t'}$, in units of the straddle premium, for the short-straddle benchmark ($q_t \equiv -1$) and for
one model of each class on the full predictor set: the closing-interval elastic net, LightGBM (the specification of
Table~\ref{tab:rh_econ_control}) and MinGRU, and the pooled intraday elastic net. The bottom panel plots the
cumulative gain over the short-straddle benchmark,
$\sum_{t' \le t} \bigl(r_{t'} + R_{t'}\bigr)$; the markers are each model's \rheTopDays{} days of largest gain, all
of them days with $q_t = +1$ and $R_t \ge \rheTopRMin$. Over the \rheNdays{} trading days of the out-of-sample period
the gain is \rheGainEnet, \rheGainLgbm, \rheGainNN{} and \rheGainPoolEnet{} premia, in the order above, and without
each model's \rheTopDays{} best days it is \rheGainExEnet, \rheGainExLgbm, \rheGainExNN{} and \rheGainExPoolEnet.
Without their \rheTopDays{} best days, \rheTailNeg{} of the \rheModelSetN{} forecasts of the model set have a negative
gain.}
\label{fig:rh_econ_cumpnl}
\end{figure}
"""
    )
    mm = e4i["med_mid"]
    fig2 = (
        HEADER
        + rf"""\begin{{figure}}[tbp]
\centering
\includegraphics[width=\linewidth]{{figures/rehaul_econ_sizing.pdf}}
\caption{{Sharpe ratio by sizing rule over the {e4i["n"]} forecasts of the model set, mid (left) and crossed (right),
on days {WARMUP + 1}--{WARMUP + e4i["n_days"]}. One marker per forecast, coloured by family and jittered vertically;
the bar and its label mark the model-set median (mid: {", ".join(f"{mm[r]:.2f}" for r, _ in RULES)} for
the sign strategy, robust clip, full long / scaled short and rank of the absolute gap). The closing-interval MinGRU on
the full predictor set (triangle) is
drawn but not in the median. Dashed: the short-straddle benchmark ({num(e4i["short_mid"])} mid, {num(e4i["short_x"])} crossed). The
robust clip is above the sign strategy for {e4i["beat_mid"]["clip"]} of the {e4i["n"]} forecasts at mid (median gain
{num(e4i["med_gain"]["clip"], plus=True)}). Rules and scoring as in Table~\ref{{tab:rh_econ_sizing}}.}}
\label{{fig:rh_econ_sizing}}
\end{{figure}}
"""
    )
    ys = yrs[SHORT]
    fig3 = (
        HEADER
        + rf"""\begin{{figure}}[tbp]
\centering
\includegraphics[width=\linewidth]{{figures/rehaul_econ_byyear.pdf}}
\caption{{Sharpe ratio by calendar year at mid, $q_t = \sign(\hat\sigma^2_t - \mathrm{{IV}}^2_t)$, for the forecasts of
Figure~\ref{{fig:rh_econ_cumpnl}} and the short-straddle benchmark (2020 to 2024: {", ".join(num(v) for v in ys)}); 2024 runs from
January to April. Each year's ratio is annualised per trade day, as in the tables. Grey: the interquartile range
and median of the 112 forecasts of the model set (median {", ".join(num(v) for v in yrs["set_med"])}).}}
\label{{fig:rh_econ_byyear}}
\end{{figure}}
"""
    )

    # ---- prose macros (\rhe...), for the section text; not \input by the floats
    def mac(name: str, val: str, note: str) -> str:
        return f"\\providecommand{{\\{name}}}{{}}\\renewcommand{{\\{name}}}{{{val}}}  % {note}\n"

    def m2(x: float, plus: bool = False) -> str:
        v = round(float(x), 2) + 0.0
        s = f"{abs(v):.2f}"
        return (
            r"\ensuremath{-}"
            if v < 0
            else (r"\ensuremath{+}" if plus and v > 0 else "")
        ) + s

    nums = (
        HEADER
        + "% \\rhe... macros for writeup/rehaul/07_economics.tex; values as in the rehaul_econ_* floats.\n"
    )
    nums += mac(
        "rheShortSh", m2(st[SHORT]["sh_mid"]), "always short, Sharpe mid, 866 days"
    )
    nums += mac("rheShortShX", m2(st[SHORT]["sh_x"]), "always short, Sharpe crossed")
    nums += mac(
        "rheShortT",
        m2(st[SHORT]["t_mean"]),
        "always short, Newey-West t of the mean mid return",
    )
    nums += mac(
        "rheShortHit", f"{100 * st[SHORT]['hit']:.1f}", "always short, hit rate %"
    )
    nums += mac(
        "rheEnetSh",
        m2(st["subpt_enet_all_features"]["sh_mid"]),
        "per-bar enet all features, Sharpe mid",
    )
    nums += mac("rheEnetShX", m2(st["subpt_enet_all_features"]["sh_x"]), "... crossed")
    nums += mac(
        "rheEnetT",
        m2(st["subpt_enet_all_features"]["t_dsh"]),
        "... HAC t vs always short",
    )
    nums += mac("rheNNSh", m2(st[NN_ALL]["sh_mid"]), "MinGRU all features, Sharpe mid")
    nums += mac("rheNNShX", m2(st[NN_ALL]["sh_x"]), "... crossed")
    nums += mac("rheNNBaseSh", m2(nb["sh_mid"]), "MinGRU HAR + calendar, Sharpe mid")
    nums += mac("rheNNBaseShX", m2(nb["sh_x"]), "... crossed")
    nums += mac(
        "rheSigVsShort",
        str(sig_vs_short),
        "model-set forecasts with HAC t > 1.96 vs always short",
    )
    nums += mac(
        "rhePairs", str(e2i["n"]), "HAR + cal. -> all features pairs in the model set"
    )
    nums += mac("rhePairsUp", str(e2i["up"]), "... Sharpe mid higher with all features")
    nums += mac("rhePairsSig", str(e2i["sig"]), "... HAC t > 1.96")
    nums += mac("rhePairsMed", m2(e2i["med"], plus=True), "... median dSharpe mid")
    for r, nm in (
        ("sign", "Sign"),
        ("clip", "Clip"),
        ("lfss", "Lfss"),
        ("rank", "Rank"),
    ):
        nums += mac(
            f"rheSz{nm}Med",
            m2(e4i["med_mid"][r]),
            f"sizing {r}: model-set median Sharpe mid, 614 days",
        )
    nums += mac(
        "rheSzShort",
        m2(e4i["short_mid"]),
        "always short, Sharpe mid on the 614 sizing days",
    )
    nums += mac(
        "rheSzClipBeats",
        str(e4i["beat_mid"]["clip"]),
        "clip above sign(s), of 112 (mid)",
    )
    nums += mac(
        "rheSzClipGain",
        m2(e4i["med_gain"]["clip"], plus=True),
        "median clip - sign(s) Sharpe mid",
    )
    nums += mac(
        "rheTailNeg",
        str(neg_ex10),
        "model-set forecasts whose gain over short is < 0 without 10 best days",
    )
    # 2026-10-15 (consolidation): numbers 07_economics.tex typed (lasso 1.88; elastic net +0.55, lasso +0.50; the
    # short straddle's losing years), now macros; values as printed in Tables E1 and E2.
    lasso_all = "subpt_lasso_all_features"
    nums += mac(
        "rheLassoSh",
        m2(st[lasso_all]["sh_mid"]),
        "per-bar lasso all features, Sharpe mid",
    )
    nums += mac("rheLassoShX", m2(st[lasso_all]["sh_x"]), "... crossed")
    nums += mac(
        "rheAfEnetDSh",
        m2(e2i["pairs"]["subpt_enet_baseline"]["d_mid"], plus=True),
        "per-bar enet: dSharpe mid, all features minus HAR + calendar (E2)",
    )
    nums += mac(
        "rheAfLassoDSh",
        m2(e2i["pairs"]["subpt_lasso_baseline"]["d_mid"], plus=True),
        "per-bar lasso: dSharpe mid, all features minus HAR + calendar (E2)",
    )
    short_years = [float(v) for v in yrs[SHORT]]
    nums += mac(
        "rheShortLossYears",
        str(sum(v < 0 for v in short_years)),
        "calendar years in which always short has a negative Sharpe mid (negative mean return)",
    )
    nums += mac(
        "rheNYears", str(len(short_years)), "calendar years in the out-of-sample period"
    )

    # 2026-10-15 (table legibility): the numbers the table notes and the figure caption used to type, as macros
    def mdp(x: float, dp: int, plus: bool = False) -> str:
        v = round(float(x), dp) + 0.0
        sgn = (
            r"\ensuremath{-}"
            if v < 0
            else (r"\ensuremath{+}" if plus and v > 0 else "")
        )
        return sgn + f"{abs(v):.{dp}f}"

    days = series[SHORT].index
    assert len(days) == 866 and len({r["dm_lag"] for r in e3r}) == 1
    nums += mac(
        "rheNdays", str(len(days)), "T_oos: trading days of the out-of-sample period"
    )
    nums += mac(
        "rheSampleStart", f"{days[0].day} {days[0]:%B %Y}", "first out-of-sample day"
    )
    nums += mac(
        "rheSampleEnd", f"{days[-1].day} {days[-1]:%B %Y}", "last out-of-sample day"
    )
    nums += mac(
        "rheHACLag",
        str(HAC_LAG),
        "Newey-West lag q of every HAC t (floor(1.5 T^(1/3)))",
    )
    nums += mac(
        "rheDMLag",
        str(e3r[0]["dm_lag"]),
        "Newey-West lag q of the DM statistics in Table E3",
    )
    nums += mac("rheModelSetN", str(len(model_set)), "forecasts in the model set")
    nums += mac(
        "rheTreeSpecs",
        str(tree_specs(fam, model_set)),
        "tree specifications per estimator on the full predictor set (E1 note a)",
    )
    nums += mac(
        "rheTreePairs", str(e2i["n_tree"]), "HAR -> full-set tree pairs (E2 Panel C)"
    )
    nums += mac("rheTreePairsEach", str(e2i["n_tree_each"]), "... per estimator")
    nums += mac("rheSzWarmup", str(WARMUP), "sizing: days that only build the scale")
    nums += mac("rheSzDays", str(e4i["n_days"]), "sizing: days scored")
    nums += mac(
        "rheTopDays", str(TOP_DAYS), "the tail check: days of largest gain dropped"
    )
    nums += mac(
        "rheTopRMin",
        f"{min_r:.2f}",
        "smallest straddle return R_t among the 4 models' top days",
    )
    for f, nm in (
        ("linear", "Enet"),
        ("tree", "Lgbm"),
        ("nn", "NN"),
        ("pooled", "PoolEnet"),
    ):
        c = cum[key4[f][0]]
        nums += mac(
            f"rheGain{nm}",
            mdp(c["edge"], 1, plus=True),
            f"{key4[f][1]}: cumulative gain over always short, premia",
        )
        nums += mac(
            f"rheGainEx{nm}",
            mdp(c["edge_ex10"], 1, plus=True),
            "... without its 10 best days",
        )

    write(GEN / "rehaul_econ_e1_control.tex", e1)
    write(GEN / "rehaul_econ_e2_allfeat.tex", e2)
    write(GEN / "rehaul_econ_e3_pooled.tex", e3)
    write(GEN / "rehaul_econ_e4_sizing.tex", e4)
    write(GEN / "rehaul_econ_fig_cumpnl.tex", fig1)
    write(GEN / "rehaul_econ_fig_sizing.tex", fig2)
    write(GEN / "rehaul_econ_fig_byyear.tex", fig3)
    write(GEN / "rehaul_econ_numbers.tex", nums)

    # ---- console summary
    print("\nE1:")
    for k, s in st.items():
        print(
            f"  {k:45s} mean {100 * s['mean']:5.2f}% t {s['t_mean']:5.2f} Sh {s['sh_mid']:5.2f}/{s['sh_x']:5.2f} "
            f"hit {100 * s['hit']:4.1f} buys {100 * s['buys']:4.1f} dSh {s.get('dsh', np.nan):+.2f} "
            f"t {s.get('t_dsh', np.nan):.2f}"
        )
    print("trees:", trees)
    print(
        "E2: n",
        e2i["n"],
        "up",
        e2i["up"],
        "sig",
        e2i["sig"],
        "med",
        round(e2i["med"], 3),
        "med_x",
        round(e2i["med_x"], 3),
        "sig keys",
        e2i["sig_keys"],
    )
    for est in ("enet", "lasso", "ridge"):
        for pre in ("subpt", "poolpt"):
            p = e2i["pairs"][f"{pre}_{est}_baseline"]
            print(
                f"  {pre} {est}: {p['sh_b']:.2f} -> {p['sh_a']:.2f} d {p['d_mid']:+.2f} (t {p['t_mid']:.2f}) "
                f"x {p['d_x']:+.2f} (t {p['t_x']:.2f})"
            )
    p = e2i["nn"]
    print(
        f"  MinGRU: {p['sh_b']:.2f} -> {p['sh_a']:.2f} d {p['d_mid']:+.2f} (t {p['t_mid']:.2f})"
    )
    print("E3:")
    for r in e3r:
        print(
            f"  {r['est']:5s} {r['set']:12s} pooled {r['sh_a']:.2f} perbar {r['sh_b']:.2f} d {r['d_mid']:+.2f} "
            f"(t {r['t_mid']:.2f}) x {r['d_x']:+.2f} (t {r['t_x']:.2f}) dQ {r['dq']:+.1f}% DM {r['dm']:.2f}"
        )
    print(
        "E4 medians mid",
        {r: round(v, 3) for r, v in e4i["med_mid"].items()},
        "crossed",
        {r: round(v, 3) for r, v in e4i["med_x"].items()},
    )
    print(
        "   beat sign mid",
        e4i["beat_mid"],
        "crossed",
        e4i["beat_x"],
        "short",
        round(e4i["short_mid"], 3),
        round(e4i["short_x"], 3),
    )
    print(
        "cum:",
        {k: {a: round(b, 2) for a, b in v.items()} for k, v in cum.items()},
        "neg_ex10",
        neg_ex10,
    )
    print(
        "by year:",
        {
            k: [round(x, 2) for x in v] if isinstance(v, list) else v
            for k, v in yrs.items()
        },
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--preview",
        type=Path,
        default=None,
        help="directory for PNG previews of the figures",
    )
    PREVIEW_DIR = ap.parse_args().preview
    main()
