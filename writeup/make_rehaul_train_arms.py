"""Robustness appendix: the pre-registered three-way comparison of training samples (tab:rh_train_arms), 2026-10-26.

The paper's main design estimates every pooled model (ridge, lasso, elastic net, LightGBM, XGBoost, random forest; on
the HAR benchmark set and on the full predictor set) on the 13 regular-hours bars of each day (RTH13, bar ends 10:00 ..
16:00 ET), with the predictors (HAR components and trailing averages) computed on the full 48-interval grid, overnight
included, on a rolling window of 3,000 trading days (39,000 bars).  A comparison fixed in a written rule before any of
its jobs (RULE.txt on each branch below) also trained the same models on

  ALL48     all 48 half-hour intervals of the futures' day (overnight, weekend and holiday intervals and the NYSE
            early-close sessions included), the same 3,000-day window (144,000 rows);
  CLOSE     the closing half hour only (the 16:00 rows of the main design), 3,000 closes;

plus, for the linear models only, the secondary arms CLOSE_NAT (the closing-bar design built natively on the closing
rows), ALL48@16 and RTH13@16 (penalties re-chosen on the 16:00 validation rows; walked only where the choice changes),
and, by an addendum written before its jobs, the equal-row arms ALL48_N3k, RTH13_N3k (the last 3,000 rows) and
ALL48_N39k (the last 39,000 rows) of the sibling's sample (CLOSE is the N = 3,000 close arm, RTH13 the N = 39,000 RTH
arm).  RTH13 is the main design: its forecasts are those of Table 1, reused (each branch's gate).  A follow-up sweep,
with its own rule, re-ran the CLOSE arm at windows W = 1,000 .. 3,500 closes (Panel F).

SOURCES  read with `git show <commit>:<path>` from the pushed result branches (nothing is checked out or merged; the
         commits are pinned in SRC, the branch tips at writing):
           linear  insurance-selling-2026-09-19-e34-train3way          results/linear_train3way_2014start_w3000/
                   scores_all/{gap,dm,qlike_16,qlike_bar}.csv, scores_all/f16.parquet, econ/econ_d6.csv,
                   econ/econ_tables.txt
           trees   insurance-selling-2026-09-19-e34-train3way-trees    results/trees_train3way_2014start_w3000/
                   scores/{gap,dm,qlike_16,qlike_bar}.csv, scores/f16.parquet, econ/econ_d6.csv (final merge 3501ab7,
                   the ALL48 forest full set including the 159 Hoffman2 pieces of -train3way-trees-h2), and the
                   interim scores_w1/-w3 (checked equal to the final scores where both exist)
         2026-10-28 (D19): f16.parquet, each arm's 16:00 forecast (plain) on the 2,579 days, from which the sign
         strategy's returns are rebuilt here; econ_d6.csv (the branches' D6 ratios) is read only for the gate that the
         rebuilt returns reproduce it.  Was also: econ/{econ_diff,econ_gap}.csv (the D6 differences and their
         bootstrap p), no longer read.
         the sign strategy's inputs (local, read only; make_rehaul_robust_window.trade_frame): the 866 option days,
           iv_var and rv_unclipped of the closing-interval OLS row (results/close_master_table/master_table_daily.parquet,
           sub_ols_baseline) joined with the quote book results/atm_straddle_0dte_1530/daily_a0.parquet
           window  insurance-selling-2026-09-19-e34-closeonly-wsweep   results/closeonly_wsweep_2014start/
                   {linear,trees}/gap.csv (CLOSE at W = 1,000 .. 3,500; W = 1,000: penalized linear models on 1,001
                   closes, the OLS on 1,000)
         and, for the gates, the main design's own score files and the printed paper numbers:
           results/window_sweep_2014start_2026-10-22/part2/{qlike,gap,econ}.csv (W = 3,000; linear)
           results/pooled_trees_final_2014start_w3000/scores/, results/pooled_rf_skx_2014start_w3000/scores/
             {metrics,dm,econ}.csv (trees)
           writeup/generated/rehaul_results_oos.tex (Table 1 cells), rehaul_oos_rerun_numbers.tex (\\rhorDm...B),
             rehaul_robust_w3500_numbers.tex (\\rhrb...Main: 16:00 and every-bar QLIKE, full vs HAR, Sharpe ratios),
             rehaul_close_accuracy_numbers.tex (\\rhca...CloseSpan), rehaul_econ_main_numbers.tex (\\rhem...); the
             last two are not yet in the repository, so their gates run when the files are present (a note otherwise)
SCORING  as on the branches (their score scripts; nothing is recomputed here): the 2,579 trading days 2014-01-02 ..
         2024-04-30 (early closes dropped); 16:00 = the closing half hour (2,579 forecasts), every bar = the 13 RTH
         bars (33,527 with RV > 0); QLIKE averaged over them; Diebold-Mariano (HLN, Newey-West lag 8) on the daily loss
         (16:00 loss; the daily mean over the 13 bars for every bar).  Full vs HAR: the full set against the HAR
         benchmark of the same model and arm (negative = the full set is more accurate).  Across arms: the first-named
         arm against the second, full set (negative = the first is more accurate).
SHARPE   2026-10-28 (author decision D19, as Section 7): recomputed here (the branches stored D6 only).  The sign
         strategy of make_rehaul_economics.add_pooled_linear / add_pooled_trees (copied in sign_returns): on the 866
         option days, q = +1 if the arm's 16:00 plain exceeds the OLS row's iv_var, else -1; mid return q R, bid/ask
         return master_table_gui.rows._crossed_return.  The Sharpe ratio is the paper's pooled ratio
         (hac_inference.paper_sharpe: sqrt(q) mean / sd, q = T / sum_y c_y = 225.5) on the 614 post-warmup option days,
         at mid-quotes (and bid/ask, macros only); every difference (full - HAR of an arm; one arm - another, full set)
         is tested by Ledoit-Wolf (2008) (hac_inference.paper_sharpe_test), the paired block bootstrap of the pooled
         difference a check (writeup/generated/sharpe_tests_d19_train_arms.csv, p_boot).  Was: the paper's D6 (the
         equal-year mean of the SR_y) read from the branches' econ files, the differences by the paired block bootstrap
         of hac_inference.d6_diff_test.
GATES    nothing is written unless all pass:
         (0) 2026-10-28 (D19): the rebuilt returns reproduce every D6 ratio of the branches' econ_d6.csv (mid and
             bid/ask, every arm, model and set) to 1e-12 (relative), and q is the printed \\rheAnnDays;
         (1) MAIN DESIGN: every RTH13 number of the branches (16:00 and every-bar QLIKE of both sets, full-vs-HAR % /
             DM / p, D6 Sharpe ratios at mid and bid/ask) equals the main design's score files to 1e-12 (relative), and
             (D19) the rebuilt RTH13 pooled ratios and full-vs-HAR Ledoit-Wolf p equal those files' sr614 (scaled by
             sqrt(q / 252)) and lw_p columns (Was: the trees' full-vs-HAR bootstrap p);
         (2) PRINTED: the same numbers reproduce, at their printed precision, Table 1's cells and \\rhorDm...B, the
             \\rhrb...Main macros of Table tab:rh_robust_w3500 (integrated in the appendix), the \\rhca...CloseSpan and
             the \\rhem... macros;
         (3) KEY NUMBERS reported to the author (KEY16, ALL48 vs RTH13 on every bar favours RTH13 in all six classes);
         (4) CONSISTENCY across the branches: the window sweep's W = 3,000 equals the CLOSE arm (linear and trees, to
             1e-12), the trees' interim scores equal the final ones, the linear @16 arms equal the primary arms where
             the penalty choice is unchanged, every arm is scored on 2,579 days (33,527 bars).
CLAIMS   the statements of writeup/drafts/2026-10-26_train_arms.tex that name classes, directions or counts are
         asserted in claims(); a rerun whose data no longer bear them out stops before writing.
WRITES   writeup/generated/rehaul_train_arms_numbers.tex   \\rhta... macros (\\providecommand, a comment per macro)
         writeup/generated/rehaul_train_arms_table.tex     Table tab:rh_train_arms (Panels A-C) and its continuation
                                                           (Panels D-F, \\ContinuedFloat, label tab:rh_train_arms_cont)
         writeup/generated/sharpe_tests_d19_train_arms.csv  2026-10-28 (D19): every Sharpe-ratio difference tested,
                                                           with its Ledoit-Wolf p and the bootstrap check p
         --gates-csv PATH writes the gate log (default: printed only).

Run:  python writeup/make_rehaul_train_arms.py      (from anywhere; git must reach the pinned commits)
"""

from __future__ import annotations

import argparse
import io
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent  # writeup/
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import hac_inference as hi  # noqa: E402  (2026-10-28, D19: the paper's Sharpe ratio and its Ledoit-Wolf test)

GEN = ROOT / "generated"
OUT_NUM = GEN / "rehaul_train_arms_numbers.tex"
OUT_TAB = GEN / "rehaul_train_arms_table.tex"
OUT_TESTS = GEN / "sharpe_tests_d19_train_arms.csv"  # 2026-10-28 (D19): the tested differences, with the check p
HEAD = "% AUTO-GENERATED by writeup/make_rehaul_train_arms.py -- do not edit; re-run the script."

# pushed result branches: (branch, pinned commit, result directory)
SRC = {
    "lin": (
        "insurance-selling-2026-09-19-e34-train3way",
        "0106ccb704b16713ccdf691f1511b86ade47a067",
        "results/linear_train3way_2014start_w3000",
    ),
    "tree": (
        "insurance-selling-2026-09-19-e34-train3way-trees",
        "3501ab7859259930c3c7fd71655709f6d26b3abc",
        "results/trees_train3way_2014start_w3000",
    ),
    "ws": (
        "insurance-selling-2026-09-19-e34-closeonly-wsweep",
        "f4abb6902b6e1dd30c80ac2221959c2c61a7692d",
        "results/closeonly_wsweep_2014start",
    ),
}
LIN_MAIN = REPO / "results" / "window_sweep_2014start_2026-10-22" / "part2"
TREE_MAIN = REPO / "results" / "pooled_trees_final_2014start_w3000" / "scores"
RF_MAIN = REPO / "results" / "pooled_rf_skx_2014start_w3000" / "scores"
W_MAIN = 3000
# 2026-10-28 (D19): the sign strategy's inputs (make_rehaul_robust_window.MT_DAILY / QUOTE_BOOK / OLS_CLOSE)
MT_DAILY = REPO / "results" / "close_master_table" / "master_table_daily.parquet"
QUOTE_BOOK = REPO / "results" / "atm_straddle_0dte_1530" / "daily_a0.parquet"  # master_table_gui.rows._QUOTE_BOOK
OLS_CLOSE = "sub_ols_baseline"  # make_rehaul_economics.OLS: the option days, iv_var and the realized variance
N_OPT = 866  # option days (make_rehaul_economics.N_OPT)
WARMUP = 252  # make_rehaul_economics.WARMUP (master_table_gui.rows._SIZING_WARMUP)
N_PW = N_OPT - WARMUP  # 614 post-warmup days, the sample of every Sharpe ratio

N_DAYS = 2579  # evaluation days, 2014-01-02 .. 2024-04-30 (early closes dropped)
N_BARS = 33527  # scored regular-hours bars
HAC_LAG = 8
N_INTERVALS = 48
N_BARS_DAY = 13
SESSIONS_YEAR = 252
SIG = 0.05

LIN = ("ridge", "lasso", "enet")
TREE = ("lgbm", "xgb", "rf")
EST = LIN + TREE
SETS = ("baseline", "all_features")
NAME = {
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "Elastic net",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "Random forest",
}
MTOK = {"ols": "Ols", "ridge": "Ridge", "lasso": "Lasso", "enet": "Enet", "lgbm": "Lgbm", "xgb": "Xgb", "rf": "Rf"}
STOK = {"baseline": "Har", "all_features": "Full"}
# arms: macro token, table label, family that has it
ATOK = {
    "RTH13": "Rth",
    "ALL48": "All",
    "CLOSE": "Cl",
    "CLOSE_NAT": "ClNat",
    "ALL48@16": "AllCt",
    "RTH13@16": "RthCt",
    "ALL48_N3k": "AllThreeK",
    "RTH13_N3k": "RthThreeK",
    "ALL48_N39k": "AllThirtyNineK",
}
ALAB = {
    "RTH13": "RTH13 (main)",
    "ALL48": "ALL48",
    "CLOSE": "CLOSE",
    "CLOSE_NAT": "CLOSE-nat",
    "ALL48@16": "ALL48@16",
    "RTH13@16": "RTH13@16",
    "ALL48_N3k": "ALL48-3k",
    "RTH13_N3k": "RTH13-3k",
    "ALL48_N39k": "ALL48-39k",
}
ADESC = {  # macro comments
    "RTH13": "RTH13 (main design: the 13 regular-hours bars, 3,000 days)",
    "ALL48": "ALL48 (all 48 intervals, 3,000 days)",
    "CLOSE": "CLOSE (the closing half hour only, 3,000 closes)",
    "CLOSE_NAT": "CLOSE_NAT (closing rows, design built natively on them; linear only)",
    "ALL48@16": "ALL48@16 (ALL48, penalties chosen on the 16:00 validation rows; linear only)",
    "RTH13@16": "RTH13@16 (RTH13, penalties chosen on the 16:00 validation rows; linear only)",
    "ALL48_N3k": "ALL48_N3k (ALL48, the last 3,000 rows)",
    "RTH13_N3k": "RTH13_N3k (RTH13, the last 3,000 rows)",
    "ALL48_N39k": "ALL48_N39k (ALL48, the last 39,000 rows)",
}
PRIMARY = ("RTH13", "ALL48", "CLOSE")
SECONDARY = ("CLOSE_NAT", "ALL48@16", "RTH13@16")  # linear only
EQUAL_ROWS = ("ALL48_N3k", "RTH13_N3k", "ALL48_N39k")
TGT = {"16:00": "close", "every bar": "bar"}
TTOK = {"close": "Close", "bar": "Bar"}
TLAB = {"close": "16:00", "bar": "all"}
TDESC = {"close": "16:00 bar", "bar": "every bar"}
WS_W = (1000, 1500, 2000, 2500, 3000, 3500)
WTOK = {1000: "One", 1500: "OneHalf", 2000: "Two", 2500: "TwoHalf", 3000: "Three", 3500: "ThreeHalf"}
WORD = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}

# Panel C and D comparisons (first arm vs second, full set), in table order
PAIRS_C = [
    ("ALL48", "RTH13", "close"),
    ("ALL48", "RTH13", "bar"),
    ("CLOSE", "RTH13", "close"),
    ("CLOSE", "ALL48", "close"),
]
# the linear rule's secondary pairs ("secondary arms vs RTH13 too"; every bar: ALL48@16, RTH13@16 vs RTH13)
PAIRS_C_SEC = [
    ("CLOSE_NAT", "RTH13", "close"),
    ("ALL48@16", "RTH13", "close"),
    ("RTH13@16", "RTH13", "close"),
    ("ALL48@16", "RTH13", "bar"),
    ("RTH13@16", "RTH13", "bar"),
]
PAIRS_D = [
    ("ALL48_N3k", "RTH13_N3k", "close"),
    ("ALL48_N3k", "RTH13_N3k", "bar"),
    ("CLOSE", "RTH13_N3k", "close"),
    ("CLOSE", "ALL48_N3k", "close"),
    ("ALL48_N39k", "RTH13", "close"),
    ("ALL48_N39k", "RTH13", "bar"),
    ("ALL48_N3k", "ALL48", "close"),
    ("ALL48_N3k", "ALL48", "bar"),
    ("RTH13_N3k", "RTH13", "close"),
    ("RTH13_N3k", "RTH13", "bar"),
    ("ALL48_N39k", "ALL48", "close"),
    ("ALL48_N39k", "ALL48", "bar"),
]
SH_ARMS = PRIMARY + SECONDARY + EQUAL_ROWS
SH_PAIRS = [("ALL48", "RTH13"), ("CLOSE", "RTH13")]

# numbers reported to the author before this table (16:00, full vs HAR: % change, DM p), checked at that precision
KEY16 = {
    ("ALL48", "lgbm"): (-6.1, "0.000"),
    ("ALL48", "xgb"): (-6.5, "0.000"),
    ("ALL48", "rf"): (-5.6, "0.000"),
    ("RTH13", "lgbm"): (-2.7, "0.071"),
    ("RTH13", "xgb"): (-3.4, "0.030"),
    ("RTH13", "rf"): (-1.4, "0.357"),
    ("CLOSE", "lgbm"): (-7.0, "0.000"),
    ("CLOSE", "xgb"): (-8.5, "0.000"),
    ("CLOSE", "rf"): (-3.7, "0.000"),
    ("RTH13", "ridge"): (0.8, None),
    ("RTH13", "lasso"): (3.3, None),
    ("RTH13", "enet"): (-0.5, None),
    ("ALL48", "ridge"): (-2.2, None),
    ("ALL48", "lasso"): (-2.6, None),
    ("ALL48", "enet"): (-1.8, None),
    ("CLOSE", "ridge"): (-6.1, "0.000"),
    ("CLOSE", "lasso"): (-5.4, "0.000"),
    ("CLOSE", "enet"): (-6.5, "0.000"),
}


# --------------------------------------------------------------------------- reading


def git_text(src: str, rel: str) -> str:
    branch, sha, base = SRC[src]
    r = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{sha}:{base}/{rel}"], capture_output=True, text=True, encoding="utf-8"
    )
    if r.returncode != 0:
        raise SystemExit(f"git show {sha[:7]}:{base}/{rel} failed ({branch}; git fetch origin {branch}): {r.stderr}")
    return r.stdout


def git_csv(src: str, rel: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(git_text(src, rel)))


def git_parquet(src: str, rel: str) -> pd.DataFrame:
    """2026-10-28 (D19): a parquet file of a pinned commit (git show, binary)."""
    branch, sha, base = SRC[src]
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{sha}:{base}/{rel}"], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"git show {sha[:7]}:{base}/{rel} failed ({branch}; git fetch origin {branch}): {r.stderr!r}")
    return pd.read_parquet(io.BytesIO(r.stdout))


# --------------------------------------------------------------------------- the sign strategy (2026-10-28, D19)


def crossed_return(q, ex, bid, ask) -> np.ndarray:
    """writeup/master_table_gui/rows.py _crossed_return: a buy pays the ask, a sale gets the bid (entry only)."""
    return np.where(q > 0, q * (ex / ask - 1.0), np.where(q < 0, q * (ex / bid - 1.0), 0.0))


def trade_frame() -> pd.DataFrame:
    """make_rehaul_robust_window.trade_frame (copied): the 866 option days of the closing-interval OLS row
    (make_rehaul_economics.load, its OLS row and book): iv_var, rv_unclipped, and the quote book's straddle (R, exit,
    bid, ask); gate: the row's mid return is q R."""
    d = pd.read_parquet(MT_DAILY, columns=["key", "day", "q", "ret_mid", "iv_var", "rv_unclipped"])
    d = d[d["key"] == OLS_CLOSE].copy()
    d["day"] = pd.to_datetime(d["day"]).dt.normalize()
    qb = pd.read_parquet(QUOTE_BOOK, columns=["R", "exit", "bid_c", "bid_p", "ask_c", "ask_p"])
    qb.index = pd.to_datetime(qb.index).normalize().astype("datetime64[ns]")
    book = pd.DataFrame(
        {
            "R": qb["R"].astype(float),
            "ex": qb["exit"].astype(float),
            "bid": (qb["bid_c"] + qb["bid_p"]).astype(float),
            "ask": (qb["ask_c"] + qb["ask_p"]).astype(float),
        }
    )
    g = d.set_index("day").sort_index()
    g.index = g.index.astype("datetime64[ns]")
    g = g.join(book, how="inner")
    assert len(g) == N_OPT, len(g)
    assert np.allclose(g["q"] * g["R"], g["ret_mid"], rtol=0, atol=1e-10), "quote-book R"
    assert np.isfinite(g[["iv_var", "rv_unclipped"]].to_numpy()).all()
    return g


def sign_returns(g: pd.DataFrame, tf: pd.DataFrame) -> dict[str, np.ndarray]:
    """experiments/{linear,trees}_train3way_econ.py (the branches' economics; make_rehaul_economics.add_pooled_linear /
    add_pooled_trees): an arm's 16:00 forecasts (f16.parquet rows: day, plain, rv_actual) on the 866 option days,
    q = +1 if plain > iv_var else -1; the mid return q R and the bid/ask return on the 614 post-warmup days.  Gates:
    a finite positive forecast every option day; the row's realized variance is the OLS row's rv_unclipped."""
    a = g.set_index(pd.DatetimeIndex(pd.to_datetime(g["day"])).astype("datetime64[ns]")).sort_index()
    assert not a.index.duplicated().any()
    a = a.reindex(tf.index)
    p = a["plain"].to_numpy(float)
    assert np.isfinite(p).all() and (p > 0).all(), int((~np.isfinite(p)).sum())
    assert np.allclose(a["rv_actual"].to_numpy(float), tf["rv_unclipped"].to_numpy(float), rtol=1e-12, atol=0.0)
    q = np.where(p > tf["iv_var"].to_numpy(float), 1.0, -1.0)
    mid = q * tf["R"].to_numpy(float)
    ba = crossed_return(q, tf["ex"].to_numpy(float), tf["bid"].to_numpy(float), tf["ask"].to_numpy(float))
    return {"mid": mid[WARMUP:], "ba": ba[WARMUP:]}


def sharpe_frames(S: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """2026-10-28 (D19): the Sharpe ratios and tests of Panel E, from each arm's rebuilt returns (sign_returns).
    S: the branches' econ_d6.csv rows (arm, model, set, d6_mid, d6_ba).  Returns
      S      + sr_mid, sr_ba (the paper's pooled ratio, hac_inference.paper_sharpe) and d6_mid_re, d6_ba_re (D6 of the
               rebuilt returns, for gate (0));
      SG     full minus HAR of an arm, mid (arm, model, dsr_mid, p_mid): the tree ensembles in every arm, the linear
               models in RTH13 (as before: the comparison the main design's files carry);
      SD     one arm minus another, full set, mid (a, b, model, set, sr_a, sr_b, dsr_mid, p_mid): SH_PAIRS;
      TESTS  every tested difference with its Ledoit-Wolf p, the bootstrap check p and the D6 difference;
      q      the annualization T / sum_y c_y of the 614 post-warmup days.
    Every p is hac_inference.paper_sharpe_test's (Ledoit-Wolf 2008, block LW_BLOCK, LW_BOOT resamples, LW_SEED)."""
    tf = trade_frame()
    pw = pd.DatetimeIndex(tf.index[WARMUP:])
    assert len(pw) == N_PW, len(pw)
    q = float(hi.paper_q(pw)["q"])
    RET: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for fam, sub in (("lin", "scores_all"), ("tree", "scores")):
        f16 = git_parquet(fam, f"{sub}/f16.parquet")
        f16 = f16.assign(model=tidy_model(f16["model"].astype(str)))
        for (v, m, s), g in f16.groupby(["variant", "model", "predictor_set"], sort=False):
            k = (str(v), str(m), str(s))
            assert k not in RET, k
            RET[k] = sign_returns(g, tf)
    keys = set(zip(S.arm, S.model, S.set))
    assert keys == set(RET), (sorted(keys ^ set(RET)))[:10]
    sr: dict[str, list[float]] = {"sr_mid": [], "sr_ba": [], "d6_mid_re": [], "d6_ba_re": []}
    for k in zip(S.arm, S.model, S.set):
        for side in ("mid", "ba"):
            r = hi.paper_sharpe(RET[k][side], dates=pw)
            assert math.isclose(r["q"], q, rel_tol=1e-15), (k, r["q"], q)
            sr[f"sr_{side}"].append(float(r["sharpe"]))
            sr[f"d6_{side}_re"].append(float(r["sharpe_d6"]))
    S = S.assign(**sr)
    tests: list[dict] = []

    def test(ka: tuple[str, str, str], kb: tuple[str, str, str], what: str) -> dict:
        a, b = RET[ka]["mid"], RET[kb]["mid"]
        t = hi.paper_sharpe_test(a, b, pw)  # Ledoit-Wolf p; p_boot the paired block bootstrap check
        d6 = hi.d6_sharpe(a, pw)["sharpe"] - hi.d6_sharpe(b, pw)["sharpe"]
        tests.append(
            dict(
                comparison=what,
                arm_a=ka[0],
                model_a=ka[1],
                set_a=ka[2],
                arm_b=kb[0],
                model_b=kb[1],
                set_b=kb[2],
                col="mid",
                sr_a=t["sr_a"],
                sr_b=t["sr_b"],
                d_sr=t["delta"],
                p_lw=t["p"],
                p_boot=t["p_boot"],
                d_sr_d6=d6,
            )
        )
        return t

    sg = []
    for v, m, s in RET:
        if s != "all_features" or not (m in TREE or (m in LIN and v == "RTH13")):
            continue
        t = test((v, m, "all_features"), (v, m, "baseline"), "full vs HAR")
        sg.append(dict(arm=v, model=m, dsr_mid=t["delta"], p_mid=t["p"]))
    sd = []
    for a, b in SH_PAIRS:
        for e in EST:
            t = test((a, e, "all_features"), (b, e, "all_features"), f"{a} vs {b}, full set")
            sd.append(
                dict(
                    a=a,
                    b=b,
                    model=e,
                    set="all_features",
                    sr_a=t["sr_a"],
                    sr_b=t["sr_b"],
                    dsr_mid=t["delta"],
                    p_mid=t["p"],
                )
            )
    return S, pd.DataFrame(sg), pd.DataFrame(sd), pd.DataFrame(tests), q


def branch_tips() -> list[str]:
    """A note for each branch whose tip is no longer the pinned commit (the numbers stay those of the pinned one)."""
    out = []
    for branch, sha, _ in SRC.values():
        r = subprocess.run(["git", "-C", str(REPO), "rev-parse", f"origin/{branch}"], capture_output=True, text=True)
        tip = r.stdout.strip()
        if r.returncode == 0 and tip != sha:
            out.append(f"NOTE: origin/{branch} is at {tip[:7]}, the numbers are read from the pinned {sha[:7]}")
    return out


def tidy_model(m: pd.Series) -> pd.Series:
    return m.replace({"rf_skx": "rf"})


def load() -> dict[str, pd.DataFrame]:
    """Every branch table this script uses, in tidy form (model rf_skx -> rf; target 16:00 -> close, every bar ->
    bar)."""
    D: dict[str, pd.DataFrame] = {}
    gaps, dms, q16, qbar, d6 = [], [], [], [], []
    for fam, sub in (("lin", "scores_all"), ("tree", "scores")):
        g = git_csv(fam, f"{sub}/gap.csv")
        g["family"] = fam
        gaps.append(g)
        d = git_csv(fam, f"{sub}/dm.csv")
        d["family"] = fam
        dms.append(d)
        a = git_csv(fam, f"{sub}/qlike_16.csv")
        a["family"] = fam
        q16.append(a)
        b = git_csv(fam, f"{sub}/qlike_bar.csv")
        b["family"] = fam
        qbar.append(b)
        e = git_csv(fam, "econ/econ_d6.csv")
        e["family"] = fam
        d6.append(e)
        # 2026-10-28 (D19): the D6 differences (econ_diff.csv) and the trees' full-vs-HAR D6 tests (econ_gap.csv) are
        # no longer read; every Sharpe ratio and test is recomputed from the 16:00 forecasts (sharpe_frames). Was:
        # f = git_csv(fam, "econ/econ_diff.csv"); f["family"] = fam; ddiff.append(f)
    G = pd.concat(gaps, ignore_index=True)
    G = G.assign(model=tidy_model(G.model), target=G.target.map(TGT)).rename(
        columns={"variant": "arm", "dm_qlike": "dm", "dm_qlike_p": "p"}
    )
    X = pd.concat(dms, ignore_index=True)
    X = X.assign(model=tidy_model(X.model), target=X.target.map(TGT)).rename(
        columns={"dm_qlike": "dm", "dm_qlike_p": "p"}
    )
    Q16 = pd.concat(q16, ignore_index=True)
    Q16 = Q16.assign(model=tidy_model(Q16.model)).rename(columns={"variant": "arm"})
    QB = pd.concat(qbar, ignore_index=True)
    QB = QB.assign(model=tidy_model(QB.model)).rename(columns={"variant": "arm"})
    S = pd.concat(d6, ignore_index=True)
    S = S.assign(model=tidy_model(S.model)).rename(columns={"variant": "arm"})
    # 2026-10-28 (D19): the paper's pooled ratio and the Ledoit-Wolf tests, from the rebuilt returns. Was:
    # SD = pd.concat(ddiff, ignore_index=True); SD = SD.assign(model=tidy_model(SD.model))
    # SG = git_csv("tree", "econ/econ_gap.csv")
    # SG = SG.assign(model=tidy_model(SG.model)).rename(columns={"variant": "arm"})
    S, SG, SD, TESTS, q = sharpe_frames(S)
    D.update(G=G, X=X, Q16=Q16, QB=QB, S=S, SD=SD, SG=SG, TESTS=TESTS)
    D["paper_q"] = pd.DataFrame({"q": [q]})
    # the window sweep of the CLOSE arm
    ws = []
    for fam, sub in (("lin", "linear"), ("tree", "trees")):
        w = git_csv("ws", f"{sub}/gap.csv")
        w["family"] = fam
        ws.append(w)
    WS = pd.concat(ws, ignore_index=True)
    D["WS"] = WS.assign(model=tidy_model(WS.model)).rename(columns={"dm_qlike": "dm", "dm_qlike_p": "p"})
    # the trees' interim scores (consistency gate)
    for w in ("scores_w1", "scores_w2", "scores_w3"):
        a = git_csv("tree", f"{w}/qlike_16.csv")
        b = git_csv("tree", f"{w}/qlike_bar.csv")
        D[f"T16_{w}"] = a.assign(model=tidy_model(a.model)).rename(columns={"variant": "arm"})
        D[f"TB_{w}"] = b.assign(model=tidy_model(b.model)).rename(columns={"variant": "arm"})
    # the option days of the branches' (D6) Sharpe ratio, from the econ tables' header line (2026-10-28, D19: gated
    # equal to the 614 post-warmup days of the rebuilt returns)
    for fam in ("lin", "tree"):
        m = re.search(r"D6 Sharpe ratio, (\d+) post-warmup option days", git_text(fam, "econ/econ_tables.txt"))
        assert m, fam
        D[f"opt_days_{fam}"] = pd.DataFrame({"n": [int(m.group(1))]})
    # the main design's own score files
    lq = pd.read_csv(LIN_MAIN / "qlike.csv")
    lg = pd.read_csv(LIN_MAIN / "gap.csv")
    le = pd.read_csv(LIN_MAIN / "econ.csv")
    D["main_lin_q"] = lq[lq.W == W_MAIN]
    D["main_lin_gap"] = lg[lg.W == W_MAIN]
    D["main_lin_econ"] = le[le.W == W_MAIN]
    tm, td, te = [], [], []
    for d in (TREE_MAIN, RF_MAIN):
        tm.append(pd.read_csv(d / "metrics.csv"))
        td.append(pd.read_csv(d / "dm.csv"))
        te.append(pd.read_csv(d / "econ.csv"))
    M = pd.concat(tm, ignore_index=True)
    D["main_tree_metrics"] = M[(M.W == W_MAIN) & (M.family == "tree")].assign(model=lambda x: tidy_model(x.model))
    lin_rows = M[(M.W == W_MAIN) & (M.family == "linear")]
    assert lin_rows.drop_duplicates().groupby(["model", "set"]).size().max() == 1  # both files carry the same rows
    D["main_lin_metrics"] = lin_rows.drop_duplicates(["model", "set"])
    T = pd.concat(td, ignore_index=True)
    D["main_tree_dm"] = T[
        (T.W == W_MAIN) & (T.comparison == "all features vs HAR benchmark") & (T.loss == "qlike")
    ].assign(model=lambda x: tidy_model(x.model), target=lambda x: x.bars.map(TGT))
    E = pd.concat(te, ignore_index=True)
    D["main_tree_econ"] = E[E.W == W_MAIN].assign(model=lambda x: tidy_model(x.model))
    return D


# --------------------------------------------------------------------------- formatting


def _fmt(x: float, dp: int) -> str:
    v = round(float(x), dp)
    if v == 0:
        v = 0.0
    return f"{'-' if v < 0 else ''}{abs(v):.{dp}f}"


def pct_dp(x: float) -> int:
    """Decimals of a printed % change: 1, or 2 when one decimal would print a nonzero change as 0.0."""
    return 2 if (x != 0 and abs(x) < 0.05) else 1


def mnum(x: float, dp: int) -> str:
    """A number for running text: \\ensuremath{-} for the minus sign."""
    s = _fmt(x, dp)
    return s.replace("-", "\\ensuremath{-}", 1) if s.startswith("-") else s


def mpct(x: float) -> str:
    return mnum(x, pct_dp(x))


def mpval(p: float) -> str:
    """The whole p statement (make_rehaul_results.mpval)."""
    if p < 0.001:
        return "\\ensuremath{p < 0.001}"
    s = f"{p:.2f}" if p >= 0.10 else f"{p:.3f}"
    if s in ("0.10", "0.100", "0.050", "0.010"):
        s = f"{p:.4f}"
    return f"\\ensuremath{{p = {s}}}"


def tnum(x: float | None, dp: int) -> str:
    """A table number in math mode ('--' when absent)."""
    if x is None or not np.isfinite(x):
        return "--"
    return f"${_fmt(x, dp)}$"


def tp(p: float | None) -> str:
    """A table p-value in parentheses, three decimals ('<0.001' below)."""
    if p is None or not np.isfinite(p):
        return "(--)"
    return "(\\ensuremath{<}0.001)" if p < 0.001 else f"({p:.3f})"


def tcell(x: float | None, p: float | None, dp: int | None = None) -> str:
    """'x (p)', the p in \\scriptsize, or '--' when the cell does not exist."""
    if x is None or not np.isfinite(x):
        return "--"
    return f"{tnum(x, pct_dp(x) if dp is None else dp)} {{\\scriptsize {tp(p)}}}"


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def thousands_txt(n: int) -> str:
    return f"{n:,}"


def num_txt(x: float) -> str:
    """62.5, 812.5 or an integer with a thousands separator."""
    return thousands(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:,.1f}".replace(",", "{,}")


# --------------------------------------------------------------------------- lookups


def gap(D: dict, arm: str, model: str, target: str) -> pd.Series | None:
    G = D["G"]
    r = G[(G.arm == arm) & (G.model == model) & (G.target == target)]
    assert len(r) <= 1, (arm, model, target)
    return None if r.empty else r.iloc[0]


def xarm(D: dict, a: str, b: str, model: str, target: str, set_: str = "all_features") -> pd.Series | None:
    X = D["X"]
    r = X[(X.a == a) & (X.b == b) & (X.model == model) & (X.target == target) & (X.set == set_)]
    assert len(r) <= 1, (a, b, model, target)
    return None if r.empty else r.iloc[0]


def sharpe(D: dict, arm: str, model: str, set_: str) -> pd.Series | None:
    S = D["S"]
    r = S[(S.arm == arm) & (S.model == model) & (S.set == set_)]
    assert len(r) <= 1
    return None if r.empty else r.iloc[0]


# 2026-10-28 (D19): parked -- the D6 bootstrap p (the trees' econ_gap.csv; the linear RTH13 p from part2/econ.csv).
# Was:
#     if model in TREE:
#         SG = D["SG"]
#         r = SG[(SG.arm == arm) & (SG.model == model)]
#         return None if r.empty else float(r.iloc[0].p_mid)
#     if arm == "RTH13":
#         e = D["main_lin_econ"]
#         return float(e[e.model == model].iloc[0].d6_boot_p_mid)
#     return None
def sharpe_p(D: dict, arm: str, model: str) -> float | None:
    """Ledoit-Wolf p of the full-set minus HAR Sharpe ratio (mid; D19, sharpe_frames): the tree ensembles in every arm,
    the linear models only in RTH13 (the comparison the main design's files carry; gated against them)."""
    SG = D["SG"]
    r = SG[(SG.arm == arm) & (SG.model == model)]
    assert len(r) <= 1
    return None if r.empty else float(r.iloc[0].p_mid)


def _lt(p: float | None, level: float) -> bool:
    """p < level for an existing p (an absent p is not significant)."""
    return p is not None and p < level


def sharpe_diff(D: dict, a: str, b: str, model: str, set_: str = "all_features") -> pd.Series | None:
    SD = D["SD"]
    r = SD[(SD.a == a) & (SD.b == b) & (SD.model == model) & (SD.set == set_)]
    assert len(r) <= 1
    return None if r.empty else r.iloc[0]


def wsweep(D: dict, w: int, model: str) -> pd.Series:
    WS = D["WS"]
    r = WS[(WS.W == w) & (WS.model == model)]
    assert len(r) == 1, (w, model)
    return r.iloc[0]


# --------------------------------------------------------------------------- gates

GATES: list[dict] = []


def gate(name: str, item: str, got: object, want: object, ok: bool | None = None) -> None:
    ok = (got == want) if ok is None else ok
    GATES.append(dict(gate=name, item=item, got=str(got), want=str(want), ok=bool(ok)))


def close(a: float, b: float, rtol: float = 1e-12) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=rtol, atol=1e-15))


def read_macros(path: Path) -> dict[str, str]:
    """name -> value of every \\providecommand / \\renewcommand / \\newcommand in a numbers file (balanced braces)."""
    txt = path.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for m in re.finditer(r"\\(?:renew|provide|new)command\{\\([A-Za-z]+)\}", txt):
        i = m.end()
        if txt[i] != "{":
            continue
        depth, j = 0, i
        while True:
            c = txt[j]
            depth += c == "{"
            depth -= c == "}"
            j += 1
            if depth == 0:
                break
        val = txt[i + 1 : j - 1]
        if val or m.group(1) not in out:
            out[m.group(1)] = val
    return out


def table_rows(path: Path) -> dict[str, list[str]]:
    """First cell -> the other cells of every body row of a generated table."""
    out = {}
    body = path.read_text(encoding="utf-8").split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]
    for line in body.splitlines():
        if " & " in line and not line.lstrip().startswith("\\multicolumn"):
            cells = [c.strip() for c in line.split("\\\\")[0].split(" & ")]
            out[cells[0]] = cells[1:]
    return out


def opt_macros(name: str) -> dict[str, str] | None:
    """The macros of a generated numbers file, or None (with a note) when the file is absent."""
    path = GEN / name
    if not path.is_file():
        print(f"NOTE: writeup/generated/{name} absent: its gates are skipped")
        return None
    return read_macros(path)


def printed(x: float, s: str) -> bool:
    """Does the printed string s (a macro value or a table cell) show x at its own precision?"""
    t = s
    for a, b in (("\\ensuremath{-}", "-"), ("$-$", "-"), ("\\ensuremath", ""), ("{,}", ""), ("$", ""), ("p = ", "")):
        t = t.replace(a, b)
    t = t.replace("{", "").replace("}", "").replace("(", "").replace(")", "").strip()
    if t.startswith("p <") or t.startswith("<"):
        return float(x) < float(t.split("<")[1])
    if "^" in t:
        t = t.split("^")[0]
    v = float(t)
    d = len(t.split(".")[1]) if "." in t else 0
    return abs(float(x) - v) <= 0.5 * 10.0**-d + 1e-12


def _fomcfix() -> bool:
    """2026-10-26: the paper's numbers come from the FOMC-patched final runs (final_run.py's REHAUL_FOMCFIX switch).
    Since the D13 regeneration the patched runs are final_run's default (REHAUL_FOMCFIX=0 opts out), so the printed
    Table 1 is the patched one unless REHAUL_FOMCFIX is explicitly off. Was: on only with REHAUL_FOMCFIX=1."""
    import os

    return os.environ.get("REHAUL_FOMCFIX", "").strip().lower() not in {"0", "false", "no", "off"}


def run_gates(D: dict) -> None:
    """GATES (module docstring): the RTH13 arm is the main design, reproduces the printed numbers and the key numbers;
    the branches agree with one another.  Stops on any failure."""
    G, Q16, QB = D["G"], D["Q16"], D["QB"]
    # --- (0) 2026-10-28 (D19): the rebuilt returns reproduce the branches' D6 ratios; q is the printed one
    S = D["S"]
    for r in S.itertuples():
        for side in ("mid", "ba"):
            got, want = getattr(r, f"d6_{side}_re"), getattr(r, f"d6_{side}")
            gate("rebuilt returns = branch econ_d6", f"{r.arm} {r.model} {r.set} {side}", got, want, close(got, want))
    q = float(D["paper_q"].q[0])
    mann = read_macros(GEN / "rehaul_econ_numbers.tex")["rheAnnDays"]
    gate("q = \\rheAnnDays", "614 post-warmup days", f"{q:.1f}", mann)
    sc252 = math.sqrt(q / 252.0)  # the main design's files carry the pooled ratio scaled by sqrt(252) (sr614_*)
    # --- (1) RTH13 = the main design's own score files
    lq, lgp, le = D["main_lin_q"], D["main_lin_gap"], D["main_lin_econ"]
    lm = D["main_lin_metrics"]  # the linear rows of the trees' metrics.csv (the same w3000 forecasts, OLS included)
    for e in ("ols",) + LIN:
        for s in SETS if e != "ols" else ("baseline",):
            r16 = Q16[(Q16.arm == "RTH13") & (Q16.model == e) & (Q16.set == s)].iloc[0]
            rb = QB[(QB.arm == "RTH13") & (QB.model == e) & (QB.set == s)].iloc[0]
            m2 = lm[(lm.model == e) & (lm.set == s)].iloc[0]
            gate(
                "main lin qlike16 (metrics.csv)", f"{e} {s}", r16.qlike, m2.qlike_1600, close(r16.qlike, m2.qlike_1600)
            )
            gate("main lin qlike_bar (metrics.csv)", f"{e} {s}", rb.qlike, m2.qlike, close(rb.qlike, m2.qlike))
            gate("main lin n (metrics.csv)", f"{e} {s}", (r16.n_days, rb.n_bars), (m2.n_days16, m2.n_bars))
            if e == "ols":
                continue
            m = lq[(lq.model == e) & (lq.set == s)].iloc[0]
            gate("main lin qlike16", f"{e} {s}", r16.qlike, m.qlike16, close(r16.qlike, m.qlike16))
            gate("main lin qlike_bar", f"{e} {s}", rb.qlike, m.qlike_bar, close(rb.qlike, m.qlike_bar))
            gate("main lin n", f"{e} {s}", (r16.n_days, rb.n_bars), (m.n_days16, m.n_bars))
    for e in LIN:
        m = lgp[lgp.model == e].iloc[0]
        for t, suf in (("close", "16"), ("bar", "bar")):
            r = gap(D, "RTH13", e, t)
            assert r is not None
            for c, mc in (("pct", f"gap_pct_{suf}"), ("dm", f"gap_dm_{suf}"), ("p", f"gap_p_{suf}")):
                gate("main lin gap", f"{e} {t} {c}", r[c], m[mc], close(r[c], m[mc]))
        me = le[le.model == e].iloc[0]
        for s, sc in (("baseline", "har"), ("all_features", "full")):
            r = sharpe(D, "RTH13", e, s)
            assert r is not None
            gate("main lin D6 mid", f"{e} {s}", r.d6_mid, me[f"d6_mid_{sc}"], close(r.d6_mid, me[f"d6_mid_{sc}"]))
            gate("main lin D6 ba", f"{e} {s}", r.d6_ba, me[f"d6_ba_{sc}"], close(r.d6_ba, me[f"d6_ba_{sc}"]))
            for side, col in (("mid", "sr_mid"), ("ba", "sr_ba")):  # 2026-10-28 (D19)
                want = me[f"sr614_{side}_{sc}"] * sc252
                gate(f"main lin SR {side}", f"{e} {s}", r[col], want, close(r[col], want, 1e-10))
        p = sharpe_p(D, "RTH13", e)  # 2026-10-28 (D19): the Ledoit-Wolf p of full minus HAR
        gate("main lin LW p", e, p, me.lw_p_mid, p is not None and close(p, me.lw_p_mid))
    tm, td, te = D["main_tree_metrics"], D["main_tree_dm"], D["main_tree_econ"]
    for e in TREE:
        for s in SETS:
            m = tm[(tm.model == e) & (tm.set == s)].iloc[0]
            r16 = Q16[(Q16.arm == "RTH13") & (Q16.model == e) & (Q16.set == s)].iloc[0]
            rb = QB[(QB.arm == "RTH13") & (QB.model == e) & (QB.set == s)].iloc[0]
            gate("main tree qlike16", f"{e} {s}", r16.qlike, m.qlike_1600, close(r16.qlike, m.qlike_1600))
            gate("main tree qlike_bar", f"{e} {s}", rb.qlike, m.qlike, close(rb.qlike, m.qlike))
            gate("main tree n", f"{e} {s}", (r16.n_days, rb.n_bars), (m.n_days16, m.n_bars))
        for t in ("close", "bar"):
            r = gap(D, "RTH13", e, t)
            assert r is not None
            m = td[(td.model == e) & (td.target == t)].iloc[0]
            gate("main tree dm", f"{e} {t}", r.dm, m.dm, close(r.dm, m.dm))
            gate("main tree p", f"{e} {t}", r.p, m.p, close(r.p, m.p))
            pct = 100.0 * (m.mean_a / m.mean_b - 1.0)
            gate("main tree pct", f"{e} {t}", r.pct, pct, close(r.pct, pct, 1e-10))
        me = te[te.model == e].iloc[0]
        for s, sc in (("baseline", "har"), ("all_features", "full")):
            r = sharpe(D, "RTH13", e, s)
            assert r is not None
            gate("main tree D6 mid", f"{e} {s}", r.d6_mid, me[f"d6_mid_{sc}"], close(r.d6_mid, me[f"d6_mid_{sc}"]))
            gate("main tree D6 ba", f"{e} {s}", r.d6_ba, me[f"d6_ba_{sc}"], close(r.d6_ba, me[f"d6_ba_{sc}"]))
            for side, col in (("mid", "sr_mid"), ("ba", "sr_ba")):  # 2026-10-28 (D19)
                want = me[f"sr614_{side}_{sc}"] * sc252
                gate(f"main tree SR {side}", f"{e} {s}", r[col], want, close(r[col], want, 1e-10))
        # 2026-10-28 (D19): the Ledoit-Wolf p. Was: gate("main tree D6 p", e, p, me.d6_boot_p_mid, ...)
        p = sharpe_p(D, "RTH13", e)
        gate("main tree LW p", e, p, me.lw_p_mid, p is not None and close(p, me.lw_p_mid))

    # --- (2) RTH13 = the printed numbers
    # 2026-10-26 (integration, APPENDIX): the training arms run on the design before the FOMC patch (D13). Once the
    # paper's numbers come from the patched runs (final_run.py REHAUL_FOMCFIX=1, with REHAUL_FINAL_RUN=1), Table 1 and
    # the printed macros are the patched ones and these gates cannot hold: they are skipped (labelled), and gate (1),
    # RTH13 = the pre-patch template's own score files, stands.
    if _fomcfix():
        print("NOTE: REHAUL_FOMCFIX on: the printed-number gates (2) are skipped -- the arms are pre-patch")
    else:
        t1 = table_rows(GEN / "rehaul_results_oos.tex")
        mor = read_macros(GEN / "rehaul_oos_rerun_numbers.tex")
        mrb = read_macros(GEN / "rehaul_robust_w3500_numbers.tex")
        # generated by the 2026-10-25 generators, not yet in the repository: their gates run when the files are present
        mca = opt_macros("rehaul_close_accuracy_numbers.tex")
        mem = opt_macros("rehaul_econ_main_numbers.tex")
        ols = QB[(QB.arm == "RTH13") & (QB.model == "ols")].iloc[0]
        gate("T1 QLIKE OLS", "ols", f"{ols.qlike:.4f}", t1["OLS"][1])
        for e in EST:
            rb = {s: QB[(QB.arm == "RTH13") & (QB.model == e) & (QB.set == s)].iloc[0] for s in SETS}
            r16 = {s: Q16[(Q16.arm == "RTH13") & (Q16.model == e) & (Q16.set == s)].iloc[0] for s in SETS}
            g16, gb = gap(D, "RTH13", e, "close"), gap(D, "RTH13", e, "bar")
            assert g16 is not None and gb is not None
            row = t1[NAME[e]]
            gate("T1 QLIKE HAR", e, f"{rb['baseline'].qlike:.4f}", row[1])
            gate("T1 QLIKE Full", e, f"{rb['all_features'].qlike:.4f}", row[4])
            dm_s, p_s = row[6].split(" (", 1)
            gate("T1 DM", e, gb.dm, dm_s, printed(gb.dm, dm_s))
            gate("T1 DM p", e, gb.p, p_s, printed(gb.p, p_s))
            gate("rhorDm..B", e, gb.dm, mor[f"rhorDm{MTOK[e]}B"], printed(gb.dm, mor[f"rhorDm{MTOK[e]}B"]))
            k = MTOK[e]
            for s in SETS:
                for t, rr in (("Bar", rb[s]), ("Close", r16[s])):
                    nm = f"rhrbQ{k}{STOK[s]}{t}Main"
                    gate("rhrbQ", f"{e} {s} {t}", rr.qlike, mrb[nm], printed(rr.qlike, mrb[nm]))
                nm = f"rhcaQ{k}{STOK[s]}CloseSpan"
                if mca is not None:
                    gate("rhcaQ", f"{e} {s}", r16[s].qlike, mca[nm], printed(r16[s].qlike, mca[nm]))
                sh = sharpe(D, "RTH13", e, s)
                assert sh is not None
                # 2026-10-28 (D19): the printed ratios are the pooled ones. Was: d6_mid, d6_ba (both loops)
                for col, tok in (("sr_mid", "Mid"), ("sr_ba", "Ba")):
                    nm = f"rhrbSR{k}{STOK[s]}{tok}Main"
                    gate("rhrbSR", f"{e} {s} {tok}", sh[col], mrb[nm], printed(sh[col], mrb[nm]))
                for col, tok in (("sr_mid", ""), ("sr_ba", "X")):
                    nm = f"rhemSh{tok}{k}{STOK[s]}"
                    if mem is not None:
                        gate("rhemSh", f"{e} {s} {tok}", sh[col], mem[nm], printed(sh[col], mem[nm]))
            for t, g in (("Bar", gb), ("Close", g16)):
                for c, tok in (("pct", "PctF"), ("dm", "DMF"), ("p", "PF")):
                    nm = f"rhrb{tok}{k}{t}Main"
                    gate("rhrb full vs HAR", f"{e} {t} {c}", g[c], mrb[nm], printed(g[c], mrb[nm]))
            for c, tok in (("pct", "Pct"), ("dm", "Dm"), ("p", "P")):
                nm = f"rhca{tok}{k}CloseSpan"
                if mca is not None:
                    gate("rhca full vs HAR", f"{e} {c}", g16[c], mca[nm], printed(g16[c], mca[nm]))
            p = sharpe_p(D, "RTH13", e)
            assert p is not None
            if mem is not None:
                nm = f"rhemPDSh{k}FullVsHar"
                gate("rhemPDSh..FullVsHar", e, p, mem[nm], printed(p, mem[nm]))
            gate("rhrbPSRF..MidMain", e, p, mrb[f"rhrbPSRF{k}MidMain"], printed(p, mrb[f"rhrbPSRF{k}MidMain"]))

    # --- (3) the key numbers reported to the author
    for (arm, e), (pct, pstr) in KEY16.items():
        r = gap(D, arm, e, "close")
        assert r is not None
        gate("key 16:00 pct", f"{arm} {e}", round(float(r.pct), 1), pct, round(float(r.pct), 1) == pct)
        if pstr is not None:
            gate("key 16:00 p", f"{arm} {e}", f"{r.p:.3f}", pstr)
        else:
            gate("key 16:00 ns", f"{arm} {e}", r.p, "> 0.10", bool(r.p > 0.10))
    for e in EST:
        r = xarm(D, "ALL48", "RTH13", e, "bar")
        assert r is not None
        gate("key every bar ALL48 vs RTH13 favours RTH13", e, r.pct, "> 0", bool(r.pct > 0))

    # --- (4) consistency across the branches
    for fam in ("lin", "tree"):
        for e in LIN if fam == "lin" else TREE:
            w = wsweep(D, W_MAIN, e)
            r = gap(D, "CLOSE", e, "close")
            assert r is not None
            for c in ("qlike_har", "qlike_full", "pct", "dm", "p"):
                gate("window sweep W=3000 = CLOSE", f"{e} {c}", w[c], r[c], close(w[c], r[c]))
    for w in ("scores_w1", "scores_w2", "scores_w3"):
        for kind, fin in (("T16", Q16), ("TB", QB)):
            a = D[f"{kind}_{w}"].merge(fin, on=["arm", "model", "set"], suffixes=("", "_f"))
            gate("trees interim = final", f"{w} {kind} ({len(a)} rows)", float((a.qlike - a.qlike_f).abs().max()), 0.0)
    # the linear @16 arms walked only the models whose penalty changed: the others equal the primary arm
    same16 = {
        "RTH13@16": [("ols", "baseline"), ("ridge", "baseline"), ("ridge", "all_features"), ("enet", "all_features")],
        "ALL48@16": [("ols", "baseline"), ("ridge", "baseline"), ("enet", "all_features")],
    }
    for arm, cells in same16.items():
        prim = arm.split("@")[0]
        for e, s in cells:
            for T in (Q16, QB):
                a = T[(T.arm == arm) & (T.model == e) & (T.set == s)].iloc[0].qlike
                b = T[(T.arm == prim) & (T.model == e) & (T.set == s)].iloc[0].qlike
                gate("@16 unchanged = primary", f"{arm} {e} {s}", a, b)
    gate("days", "16:00", sorted(set(Q16.n_days)), [N_DAYS])
    gate("days", "every bar", sorted(set(QB.n_days)), [N_DAYS])
    gate("bars", "every bar", sorted(set(QB.n_bars)), [N_BARS])
    gate("DM T / lag", "gap", sorted(set(zip(G.dm_T, G.dm_hac_lag))), [(N_DAYS, HAC_LAG)])
    gate("DM T / lag", "cross-arm", sorted(set(zip(D["X"].dm_T, D["X"].dm_hac_lag))), [(N_DAYS, HAC_LAG)])
    gate("DM T / lag", "window sweep", sorted(set(zip(D["WS"].dm_T, D["WS"].dm_hac_lag))), [(N_DAYS, HAC_LAG)])
    gate("option days", "lin = tree", int(D["opt_days_lin"].n[0]), int(D["opt_days_tree"].n[0]))
    gate("option days", "lin = rebuilt returns (D19)", int(D["opt_days_lin"].n[0]), N_PW)
    gate("window sweep W", "both", sorted(set(D["WS"].W)), list(WS_W))
    # every Panel A-D cell of a primary or equal-row arm exists for every class
    for arm in PRIMARY + EQUAL_ROWS:
        for e in EST:
            gate("cell exists", f"{arm} {e} 16:00", gap(D, arm, e, "close") is not None, True)
            if arm != "CLOSE":
                gate("cell exists", f"{arm} {e} every bar", gap(D, arm, e, "bar") is not None, True)
    for a, b, t in PAIRS_C + PAIRS_D:
        for e in EST:
            gate("pair exists", f"{a} vs {b} {t} {e}", xarm(D, a, b, e, t) is not None, True)
    for a, b, t in PAIRS_C_SEC:  # linear only: the trees have no secondary arm
        for e in EST:
            gate("secondary pair", f"{a} vs {b} {t} {e}", xarm(D, a, b, e, t) is not None, e in LIN)

    g = pd.DataFrame(GATES)
    bad = g[~g.ok]
    print(f"gates: {len(g)} checks, {len(bad)} failed")
    for name, grp in g.groupby("gate", sort=False):
        print(f"  {name}: {int(grp.ok.sum())}/{len(grp)}")
    if len(bad):
        print(bad.to_string())
        raise SystemExit("GATE FAILED: nothing written")


# --------------------------------------------------------------------------- claims


def claims(D: dict) -> None:
    """The statements of writeup/drafts/2026-10-26_train_arms.tex that name classes, directions or counts."""

    def gp(arm: str, t: str) -> pd.DataFrame:
        G = D["G"]
        return G[(G.arm == arm) & (G.target == t)].set_index("model")

    def xp(a: str, b: str, t: str) -> pd.DataFrame:
        X = D["X"]
        return X[(X.a == a) & (X.b == b) & (X.target == t) & (X.set == "all_features")].set_index("model")

    def sig(df: pd.DataFrame, level: float = SIG) -> set[str]:
        return {e for e in df.index if df.loc[e, "p"] < level}

    # Panel A: the closing bar, full vs HAR
    a16, r16, c16 = gp("ALL48", "close"), gp("RTH13", "close"), gp("CLOSE", "close")
    assert (a16.loc[list(TREE), "pct"] < 0).all() and (a16.loc[list(TREE), "p"] < 0.001).all()
    assert (a16.loc[list(LIN), "pct"] < 0).all() and (a16.loc[list(LIN), "p"] > 0.10).all()
    assert (c16.loc[list(EST), "pct"] < 0).all() and (c16.loc[list(EST), "p"] < 0.001).all()
    assert sig(r16) == {"xgb"} and sig(r16, 0.10) == {"xgb", "lgbm"}
    # a larger 16:00 gain than the main design's in every class, under ALL48 and under CLOSE
    for df in (a16, c16):
        assert (df.loc[list(EST), "pct"] < r16.loc[list(EST), "pct"]).all()
    for arm in ("ALL48@16", "RTH13@16"):
        assert (gp(arm, "close").loc[list(LIN), "p"] > 0.10).all(), arm
    cn = gp("CLOSE_NAT", "close")
    assert (cn.loc[list(LIN), "pct"] < 0).all() and (cn.loc[list(LIN), "p"] < 0.001).all()
    # the HAR benchmark trained on the close alone: higher closing-bar QLIKE for the trees, lower for the linear models
    for e in TREE:
        assert c16.loc[e, "qlike_har"] > r16.loc[e, "qlike_har"], e
    for e in LIN:
        assert c16.loc[e, "qlike_har"] < r16.loc[e, "qlike_har"], e
    # Panel B: every bar, full vs HAR: significant in every arm and class; larger under RTH13 than ALL48 in all six
    rb, ab = gp("RTH13", "bar"), gp("ALL48", "bar")
    for df in (rb, ab, gp("ALL48@16", "bar"), gp("RTH13@16", "bar")):
        assert (df.pct < 0).all() and (df.p < 0.001).all()
    assert (rb.loc[list(EST), "pct"] < ab.loc[list(EST), "pct"]).all()
    # Panel C: ALL48 vs RTH13 (full set)
    x16, xb = xp("ALL48", "RTH13", "close"), xp("ALL48", "RTH13", "bar")
    assert (x16.pct < 0).all() and sig(x16) == set(EST) - {"enet"}, sig(x16)
    assert (xb.pct > 0).all() and (xb.p < 0.01).all()
    rc = xp("RTH13@16", "RTH13", "close")  # the 16:00-tuned penalties: only the lasso's full-set forecast moves
    assert rc.loc["lasso", "pct"] < 0 and rc.loc["lasso", "p"] < 0.001
    assert (rc.loc[["ridge", "enet"], "pct"] == 0).all()
    g16 = gp("RTH13@16", "close")
    assert g16.loc["lasso", "qlike_har"] != r16.loc["lasso", "qlike_har"]
    xc = xp("CLOSE", "RTH13", "close")
    assert (xc.loc[list(LIN), "pct"] < 0).all() and (xc.loc[list(LIN), "p"] < 0.001).all()
    assert (xc.loc[list(TREE), "p"] > 0.10).all()
    # Panel D: equal rows
    l16, lb = xp("ALL48_N39k", "RTH13", "close"), xp("ALL48_N39k", "RTH13", "bar")
    assert sig(l16) == {"lasso", "rf"} and (l16.loc[["lasso", "rf"], "pct"] < 0).all(), sig(l16)
    assert (l16.loc[list(EST), "pct"] > x16.loc[list(EST), "pct"]).all()  # ALL48_N39k's 16:00 edge below ALL48's
    assert (lb.pct > 0).all() and (lb.p < 0.001).all()
    v16 = xp("ALL48_N39k", "ALL48", "close")
    assert (v16.pct > 0).all() and sig(v16) == set(EST) - {"enet"}, sig(v16)
    s16, sb = xp("ALL48_N3k", "RTH13_N3k", "close"), xp("ALL48_N3k", "RTH13_N3k", "bar")
    assert (s16.pct < 0).all() and sig(s16) == set(EST) - {"xgb"}, sig(s16)
    assert (sb.pct > 0).all() and (sb.p < 0.001).all()
    cs = xp("CLOSE", "RTH13_N3k", "close")
    assert (cs.pct < 0).all() and (cs.p < 0.01).all()
    for t in ("close", "bar"):
        rs = xp("RTH13_N3k", "RTH13", t)
        assert (rs.pct > 0).all() and (rs.p < 0.01).all(), t
    # Panel E (2026-10-28, D19: the pooled ratio, Ledoit-Wolf p; the claims below were D6 / bootstrap until then)
    sd_p = {(a, e): sharpe_diff(D, a, b, e).p_mid for a, b in SH_PAIRS for e in EST}  # type: ignore[union-attr]
    nsig = {arm: sum(_lt(sharpe_p(D, arm, e), SIG) for e in TREE) for arm in PRIMARY}
    print(f"claims, Panel E (D19): cross-arm p {sd_p}; tree full-vs-HAR p < 0.05 by arm {nsig}")
    # no full-set Sharpe ratio difference ALL48 - RTH13 or CLOSE - RTH13 significant at 5 %
    for (a, e), p in sd_p.items():
        assert p >= SIG, (a, e, p)
    # the trees' full-vs-HAR Sharpe ratio difference: significant at 5 % in 2 / 0 / 1 classes under RTH13 / ALL48 / CLOSE
    # (2026-10-28, D19, Ledoit-Wolf; was 1 / 2 / 2 under D6 and the bootstrap), and each significant one a gain (the
    # text: "the full-set strategy's gain over the HAR benchmark's is significant")
    assert nsig == {"RTH13": 2, "ALL48": 0, "CLOSE": 1}, nsig
    SG = D["SG"]
    assert (SG[SG.p_mid < SIG].dsr_mid > 0).all(), SG[SG.p_mid < SIG]
    assert all(not _lt(sharpe_p(D, "RTH13", e), SIG) for e in LIN)
    for e in TREE:
        sc, sr = sharpe(D, "CLOSE", e, "baseline"), sharpe(D, "RTH13", e, "baseline")
        # 2026-10-28 (D19): was sc.d6_mid < sr.d6_mid
        assert sc is not None and sr is not None and sc.sr_mid < sr.sr_mid, e
    # Panel F: the window sweep: lower QLIKE with the full set at every W and class; significant at 5 % from W = 2,000
    WS = D["WS"]
    assert (WS.pct < 0).all()
    assert (WS[WS.W >= 2000].p < SIG).all()
    assert (WS[WS.W < 2000].p >= SIG).any()
    # no sample with every-bar forecasts has a lower every-bar QLIKE of the full set than RTH13, in any class
    QB = D["QB"]
    qf = QB[QB.set == "all_features"].pivot_table(index="arm", columns="model", values="qlike")
    for e in EST:
        assert qf.loc["RTH13", e] <= qf[e].dropna().min() + 1e-15, (e, qf[e].idxmin())
    # the main design's 16:00 full-set QLIKE is above ALL48's in every class (Panel C) -- the trade-off the draft names
    assert (xp("ALL48", "RTH13", "close").pct < 0).all()
    print("claims: all hold")


# --------------------------------------------------------------------------- macros


def write_macros(D: dict) -> set[str]:
    rows: list[tuple[str, str, str]] = []

    def add(name: str, val: str, com: str) -> None:
        assert re.fullmatch(r"rhta[A-Za-z]+", name), name
        rows.append((name, val, com))

    def rng(prefix: str, xs: list[float], what: str) -> None:
        add(f"{prefix}Min", mpct(min(xs)), f"{what}: lowest of the range, %")
        add(f"{prefix}Max", mpct(max(xs)), f"{what}: highest of the range, %")

    G, X = D["G"], D["X"]
    n_opt = int(D["opt_days_lin"].n[0])
    # --- constants
    add("rhtaNdays", thousands(N_DAYS), "trading days scored, 2 January 2014 .. 30 April 2024 (early closes dropped)")
    add("rhtaNbars", thousands(N_BARS), "regular-hours bars scored (every bar; RV > 0)")
    add("rhtaStart", "2 January 2014", "first evaluation day")
    add("rhtaEnd", "30 April 2024", "last evaluation day")
    add("rhtaHacLag", str(HAC_LAG), "Newey-West lag of the daily Diebold-Mariano tests")
    add("rhtaNIntervals", str(N_INTERVALS), "half-hour intervals of the futures' day (ALL48)")
    add("rhtaNBarsDay", str(N_BARS_DAY), "regular-hours bars of a day, bar ends 10:00 .. 16:00 ET (RTH13)")
    add("rhtaWinDays", thousands(W_MAIN), "rolling window of the equal-calendar arms, trading days")
    add("rhtaRowsAll", thousands(W_MAIN * N_INTERVALS), "training rows of ALL48 (3,000 days x 48 intervals)")
    add("rhtaRowsRth", thousands(W_MAIN * N_BARS_DAY), "training rows of RTH13 (3,000 days x 13 bars)")
    add("rhtaRowsCl", thousands(W_MAIN), "training rows of CLOSE (3,000 closes)")
    add("rhtaRowsSmall", thousands(3000), "training rows of the small equal-row arms (ALL48_N3k, RTH13_N3k; CLOSE)")
    add("rhtaRowsLarge", thousands(39000), "training rows of the large equal-row arm ALL48_N39k (= RTH13's)")
    add("rhtaSpanAllSmall", num_txt(3000 / N_INTERVALS), "days of 48 intervals spanned by 3,000 rows (about)")
    add("rhtaSpanRthSmall", num_txt(round(3000 / N_BARS_DAY)), "sessions of 13 bars spanned by 3,000 rows (about)")
    add("rhtaSpanClSmall", num_txt(round(3000 / SESSIONS_YEAR)), "years of closes spanned by 3,000 closes (about)")
    add("rhtaSpanAllLarge", num_txt(39000 / N_INTERVALS), "days of 48 intervals spanned by 39,000 rows (about)")
    add(
        "rhtaValYear", "2013", "validation year of the pre-sample fold on which every arm's linear penalties are chosen"
    )
    # 2026-10-28 (D19): was "post-warmup option days of the D6 Sharpe ratio (Section 7)"
    add("rhtaOptDays", thousands(n_opt), "post-warmup option days of the Sharpe ratio (Section 7; D19 pooled ratio)")
    add("rhtaNClasses", str(len(EST)), "model classes (ridge, lasso, elastic net, LightGBM, XGBoost, random forest)")
    add("rhtaWsMin", thousands(min(WS_W)), "smallest window of the close-only sweep, closes")
    add("rhtaWsMax", thousands(max(WS_W)), "largest window of the close-only sweep, closes")

    # --- per cell: full vs HAR
    for arm, at in ATOK.items():
        for t in ("close", "bar"):
            for e in EST:
                r = gap(D, arm, e, t)
                if r is None:
                    continue
                stem = f"{MTOK[e]}{at}{TTOK[t]}"
                where = f"{NAME[e]}, {ADESC[arm]}, {TDESC[t]}"
                add(f"rhtaPct{stem}", mpct(r.pct), f"QLIKE % change, full set vs HAR benchmark: {where}")
                add(f"rhtaDm{stem}", mnum(r.dm, 2), "... its Diebold-Mariano statistic (negative: full more accurate)")
                add(f"rhtaP{stem}", mpval(r.p), "... its p statement")
                add(f"rhtaQHar{stem}", f"{r.qlike_har:.4f}", "... mean QLIKE, HAR benchmark")
                add(f"rhtaQFull{stem}", f"{r.qlike_full:.4f}", "... mean QLIKE, full set")
    # --- per cell: across arms (full set)
    for a, b, t in PAIRS_C + PAIRS_C_SEC + PAIRS_D:
        for e in EST:
            r = xarm(D, a, b, e, t)
            if r is None:  # a secondary pair of a tree class (none run)
                continue
            stem = f"{MTOK[e]}{ATOK[a]}Vs{ATOK[b]}{TTOK[t]}"
            add(
                f"rhtaXPct{stem}",
                mpct(r.pct),
                f"QLIKE % change, {a} vs {b} (full set; negative: {a} more accurate), {NAME[e]}, {TDESC[t]}",
            )
            add(f"rhtaXDm{stem}", mnum(r.dm, 2), "... its Diebold-Mariano statistic")
            add(f"rhtaXP{stem}", mpval(r.p), "... its p statement")
    # --- Sharpe ratios (2026-10-28, D19: the paper's pooled ratio, Ledoit-Wolf p; was D6 and the bootstrap p, read
    # from the branches: r.d6_mid, r.d6_ba, r.dd6_mid)
    for arm in SH_ARMS:
        for e in EST:
            for s in SETS:
                r = sharpe(D, arm, e, s)
                if r is None:
                    continue
                stem = f"{MTOK[e]}{STOK[s]}{ATOK[arm]}"
                add(f"rhtaSh{stem}", mnum(r.sr_mid, 2), f"Sharpe ratio (D19), mid-quotes: {NAME[e]}, {s}, {ADESC[arm]}")
                add(f"rhtaShX{stem}", mnum(r.sr_ba, 2), "... at quoted bid/ask prices")
            p = sharpe_p(D, arm, e)
            if p is not None:
                add(
                    f"rhtaPSh{MTOK[e]}{ATOK[arm]}",
                    mpval(p),
                    f"Ledoit-Wolf p, Sharpe ratio full minus HAR (mid): {NAME[e]}, {arm}",
                )
    for a, b in SH_PAIRS:
        for e in EST:
            r = sharpe_diff(D, a, b, e)
            assert r is not None
            stem = f"{MTOK[e]}{ATOK[a]}Vs{ATOK[b]}"
            add(f"rhtaDSh{stem}", mnum(r.dsr_mid, 2), f"Sharpe ratio (D19, mid), {a} minus {b}, full set, {NAME[e]}")
            add(f"rhtaPDSh{stem}", mpval(r.p_mid), "... its Ledoit-Wolf p")
    # --- the close-only window sweep
    for w in WS_W:
        for e in EST:
            r = wsweep(D, w, e)
            stem = f"{MTOK[e]}W{WTOK[w]}"
            add(
                f"rhtaWsPct{stem}",
                mpct(r.pct),
                f"CLOSE at W = {w:,} closes, 16:00, QLIKE % change full vs HAR, {NAME[e]}",
            )
            add(f"rhtaWsP{stem}", mpval(r.p), "... its p statement")

    # --- summaries for the draft (claims() asserts the directions and counts they summarize)
    def pcts(df: pd.DataFrame, models: tuple[str, ...]) -> list[float]:
        return [float(df.loc[e, "pct"]) for e in models]

    def gp(arm: str, t: str) -> pd.DataFrame:
        return G[(G.arm == arm) & (G.target == t)].set_index("model")

    def xp(a: str, b: str, t: str) -> pd.DataFrame:
        return X[(X.a == a) & (X.b == b) & (X.target == t) & (X.set == "all_features")].set_index("model")

    for arm in ("ALL48", "CLOSE", "RTH13", "CLOSE_NAT"):
        df = gp(arm, "close")
        at = ATOK[arm]
        rng(f"rhtaLinPct{at}Close", pcts(df, LIN), f"16:00, full vs HAR, {arm}, penalized linear models")
        if arm != "CLOSE_NAT":
            rng(f"rhtaTreePct{at}Close", pcts(df, TREE), f"16:00, full vs HAR, {arm}, tree ensembles")
            rng(f"rhtaPct{at}Close", pcts(df, EST), f"16:00, full vs HAR, {arm}, all six classes")
    add(
        "rhtaLinAllClosePMin", mpval(gp("ALL48", "close").loc[list(LIN), "p"].min()), "16:00, ALL48, linear: smallest p"
    )
    for arm in ("ALL48@16", "RTH13@16"):
        add(
            f"rhtaLin{ATOK[arm]}ClosePMin",
            mpval(gp(arm, "close").loc[list(LIN), "p"].min()),
            f"16:00, {arm}, linear: smallest p",
        )
    for arm in ("RTH13", "ALL48"):
        df = gp(arm, "bar")
        rng(f"rhtaLinPct{ATOK[arm]}Bar", pcts(df, LIN), f"every bar, full vs HAR, {arm}, penalized linear models")
        rng(f"rhtaTreePct{ATOK[arm]}Bar", pcts(df, TREE), f"every bar, full vs HAR, {arm}, tree ensembles")
    x = xp("ALL48", "RTH13", "close")
    rng("rhtaAllVsRthClosePct", pcts(x, EST), "16:00, ALL48 vs RTH13, full set, all six classes")
    add("rhtaNAllVsRthCloseSig", WORD[int((x.p < SIG).sum())], "16:00, ALL48 vs RTH13: classes with p < 0.05 (lower)")
    x = xp("ALL48", "RTH13", "bar")
    rng(
        "rhtaAllVsRthBarPct", pcts(x, EST), "every bar, ALL48 vs RTH13, full set, all six classes (RTH13 more accurate)"
    )
    add("rhtaAllVsRthBarPMax", mpval(x.p.max()), "every bar, ALL48 vs RTH13: largest p of the six")
    x = xp("CLOSE", "RTH13", "close")
    rng("rhtaLinClVsRthClosePct", pcts(x, LIN), "16:00, CLOSE vs RTH13, full set, penalized linear models")
    rng("rhtaTreeClVsRthClosePct", pcts(x, TREE), "16:00, CLOSE vs RTH13, full set, tree ensembles")
    add("rhtaTreeClVsRthClosePMin", mpval(x.loc[list(TREE), "p"].min()), "... tree ensembles: smallest p")
    add("rhtaTreeClVsRthClosePMax", mpval(x.loc[list(TREE), "p"].max()), "... tree ensembles: largest p")
    x = xp("ALL48_N39k", "RTH13", "close")
    rng("rhtaAllThirtyNineKVsRthClosePct", pcts(x, EST), "16:00, ALL48_N39k vs RTH13, full set, all six classes")
    add("rhtaNAllThirtyNineKVsRthCloseSig", WORD[int((x.p < SIG).sum())], "... classes with p < 0.05")
    x = xp("ALL48_N39k", "RTH13", "bar")
    rng("rhtaAllThirtyNineKVsRthBarPct", pcts(x, EST), "every bar, ALL48_N39k vs RTH13, full set, all six classes")
    x = xp("ALL48_N39k", "ALL48", "close")
    rng(
        "rhtaAllThirtyNineKVsAllClosePct", pcts(x, EST), "16:00, ALL48_N39k vs ALL48 (39,000 vs 144,000 rows), full set"
    )
    add("rhtaNAllThirtyNineKVsAllCloseSig", WORD[int((x.p < SIG).sum())], "... classes with p < 0.05 (higher loss)")
    x = xp("ALL48_N3k", "RTH13_N3k", "close")
    rng("rhtaAllThreeKVsRthThreeKClosePct", pcts(x, EST), "16:00, ALL48_N3k vs RTH13_N3k, full set, all six classes")
    add("rhtaNAllThreeKVsRthThreeKCloseSig", WORD[int((x.p < SIG).sum())], "... classes with p < 0.05 (lower)")
    x = xp("ALL48_N3k", "RTH13_N3k", "bar")
    rng("rhtaAllThreeKVsRthThreeKBarPct", pcts(x, EST), "every bar, ALL48_N3k vs RTH13_N3k, full set, all six classes")
    x = xp("CLOSE", "RTH13_N3k", "close")
    rng("rhtaClVsRthThreeKClosePct", pcts(x, EST), "16:00, CLOSE vs RTH13_N3k (3,000 rows each), full set")
    add("rhtaClVsRthThreeKClosePMax", mpval(x.p.max()), "... largest p of the six")
    x = xp("RTH13_N3k", "RTH13", "close")
    rng("rhtaRthThreeKVsRthClosePct", pcts(x, EST), "16:00, RTH13_N3k vs RTH13 (3,000 vs 39,000 rows), full set")
    x = xp("RTH13_N3k", "RTH13", "bar")
    rng("rhtaRthThreeKVsRthBarPct", pcts(x, EST), "every bar, RTH13_N3k vs RTH13 (3,000 vs 39,000 rows), full set")
    # Sharpe ratios (2026-10-28, D19: sr_mid; was d6_mid)
    for arm in PRIMARY:
        v = [float(sharpe(D, arm, e, "all_features").sr_mid) for e in EST]  # type: ignore[union-attr]
        add(
            f"rhtaShFull{ATOK[arm]}Min", mnum(min(v), 2), f"Sharpe ratio (D19, mid), full set, {arm}: lowest of the six"
        )
        add(f"rhtaShFull{ATOK[arm]}Max", mnum(max(v), 2), "... highest of the six")
        n = sum(_lt(sharpe_p(D, arm, e), SIG) for e in TREE)
        add(
            f"rhtaNTreeShSig{ATOK[arm]}",
            WORD[n],
            f"tree classes whose full-vs-HAR Sharpe ratio difference has Ledoit-Wolf p < 0.05, {arm}",
        )
    v = [float(sharpe(D, "CLOSE", e, "baseline").sr_mid) for e in TREE]  # type: ignore[union-attr]
    add("rhtaShTreeHarClMin", mnum(min(v), 2), "Sharpe ratio (D19, mid), HAR benchmark, CLOSE, tree ensembles: lowest")
    add("rhtaShTreeHarClMax", mnum(max(v), 2), "... highest")
    ps = [float(sharpe_diff(D, a, b, e).p_mid) for a, b in SH_PAIRS for e in EST]  # type: ignore[union-attr]
    add(
        "rhtaNDShTests", str(len(ps)), "Sharpe ratio tests ALL48 - RTH13 and CLOSE - RTH13, full set (six classes each)"
    )
    add("rhtaNDShSig", WORD[sum(p < SIG for p in ps)], "... of them with Ledoit-Wolf p < 0.05")
    add("rhtaDShPMin", mpval(min(ps)), "... the smallest Ledoit-Wolf p")
    # window sweep
    WS = D["WS"]
    add("rhtaWsNCells", str(len(WS)), "window x class cells of the close-only sweep (16:00, full vs HAR)")
    add("rhtaWsNLower", str(int((WS.pct < 0).sum())), "... with the lower QLIKE on the full set")
    add("rhtaWsNSig", str(int((WS.p < SIG).sum())), "... with p < 0.05")
    add(
        "rhtaWsSigFrom",
        thousands(int(min(w for w in WS_W if (WS[WS.W >= w].p < SIG).all()))),
        "smallest W from which every cell has p < 0.05",
    )
    rng("rhtaWsLinPct", WS[WS.model.isin(LIN)].pct.tolist(), "close-only sweep, 16:00, full vs HAR, linear, all W")
    rng("rhtaWsTreePct", WS[WS.model.isin(TREE)].pct.tolist(), "close-only sweep, 16:00, full vs HAR, trees, all W")

    names = [r[0] for r in rows]
    dup = {n for n in names if names.count(n) > 1}
    assert not dup, dup
    lines = [
        HEAD,
        "% \\rhta... macros: the pre-registered comparison of training samples (tab:rh_train_arms; draft",
        "% writeup/drafts/2026-10-26_train_arms.tex).  Names: <Stat><Model><Arm><Bar>; Arm Rth (RTH13, the main design),",
        "% All (ALL48), Cl (CLOSE), ClNat (CLOSE_NAT), AllCt / RthCt (penalties chosen at 16:00), AllThreeK / RthThreeK /",
        "% AllThirtyNineK (the last 3,000 / 39,000 rows); Bar Close (16:00) or Bar (every bar); X... = first arm vs second",
        "% (full set); Sh... = Sharpe ratio (D19: the paper's pooled ratio; p by Ledoit-Wolf); Ws... = the close-only",
        "% window sweep.",
        "% Sources (git show, pinned): "
        + "; ".join(f"{SRC[k][0]} {SRC[k][1][:7]}" for k in ("lin", "tree", "ws"))
        + ".",
    ]
    for mname, mval, mcom in rows:
        lines.append(f"\\providecommand{{\\{mname}}}{{}}\\renewcommand{{\\{mname}}}{{{mval}}}  % {mcom}")
    OUT_NUM.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return set(names)


# --------------------------------------------------------------------------- table

NCOL1 = 1 + len(EST)  # Panels A-C: a label column and the six classes
NCOL2 = 2 + len(EST)  # Panels D-F: two label columns (first / second sample; sample / Full or HAR; window)


def header(nlab: int) -> list[str]:
    lo, hi = nlab + 1, nlab + len(LIN)
    return [
        " & " * nlab + "\\multicolumn{3}{c}{Penalized linear} & \\multicolumn{3}{c}{Tree ensembles} \\\\",
        f"\\cmidrule(lr){{{lo}-{hi}}}\\cmidrule(l){{{hi + 1}-{hi + len(TREE)}}}",
        " & " * nlab + " & ".join(NAME[e] for e in EST) + " \\\\",
    ]


def panel(title: str, ncol: int) -> str:
    return f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\"


def block(title: str, ncol: int) -> str:
    return f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{{title}}} \\\\"


def gap_cells(D: dict, arm: str, t: str) -> str:
    cells = []
    for e in EST:
        r = gap(D, arm, e, t)
        cells.append("--" if r is None else tcell(r.pct, r.p))
    return " & ".join(cells)


def pair_cells(D: dict, a: str, b: str, t: str) -> str:
    cells = []
    for e in EST:
        r = xarm(D, a, b, e, t)
        cells.append("--" if r is None else tcell(r.pct, r.p))
    return " & ".join(cells)


def tabular_open(nlab: int) -> str:
    return f"\\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}{'l' * nlab}{'r' * len(EST)}@{{}}}}"


def short(arm: str) -> str:
    return ALAB[arm].replace(" (main)", "")


def write_table(D: dict) -> None:
    src = "; ".join(f"{SRC[k][0]} {SRC[k][1][:7]}" for k in ("lin", "tree", "ws"))
    L = [
        HEAD,
        "% tab:rh_train_arms (appendix): the pre-registered three-way comparison of training samples (RTH13 = the main",
        "% design, ALL48, CLOSE; the linear secondary arms; the equal-row arms) and the close-only window sweep, scored on",
        "% the 2,579 days 2014-01-02 .. 2024-04-30; DM (HLN, Newey-West lag 8) on the daily loss; Sharpe ratio of the sign",
        "% strategy on the 614 post-warmup option days (D19: the paper's pooled ratio, Ledoit-Wolf p; was D6, bootstrap p).",
        "% Two floats: Panels A-C, and D-F (\\ContinuedFloat).",
        f"% Sources (git show, pinned): {src}.",
        "\\ifdefined\\rhtaNdays\\else\\input{generated/rehaul_train_arms_numbers}\\fi",
        "\\begin{table}[p]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{1.5pt}",
        "\\begin{threeparttable}",
        "\\caption{Robustness: the training sample}",
        "\\label{tab:rh_train_arms}",
        tabular_open(1),
        "\\toprule",
        *header(1),
        "\\midrule",
        panel("Panel A: full predictor set vs HAR benchmark, 16:00 bar", NCOL1),
    ]
    for i, arm in enumerate(PRIMARY + SECONDARY + EQUAL_ROWS):
        if i in (len(PRIMARY), len(PRIMARY + SECONDARY)):
            L.append("\\addlinespace[2pt]")
        L.append(f"\\quad {ALAB[arm]} & {gap_cells(D, arm, 'close')} \\\\")
    L += ["\\addlinespace[4pt]", panel("Panel B: full predictor set vs HAR benchmark, every bar", NCOL1)]
    for i, arm in enumerate(("RTH13", "ALL48", "ALL48@16", "RTH13@16") + EQUAL_ROWS):
        if i in (2, 4):
            L.append("\\addlinespace[2pt]")
        L.append(f"\\quad {ALAB[arm]} & {gap_cells(D, arm, 'bar')} \\\\")
    L += ["\\addlinespace[4pt]", panel("Panel C: across training samples, full predictor set", NCOL1)]
    for t in ("close", "bar"):
        L.append(block(TDESC[t].capitalize(), NCOL1))
        for a, b, tt in PAIRS_C:
            if tt == t:
                L.append(f"\\quad {short(a)} vs {short(b)} & {pair_cells(D, a, b, t)} \\\\")
        for a, b, tt in PAIRS_C_SEC:
            if tt == t:
                L.append(f"\\quad {short(a)} vs {short(b)} & {pair_cells(D, a, b, t)} \\\\")
    L += [
        "\\bottomrule",
        "\\end{tabular*}",
        "\\begin{tablenotes}[flushleft]",
        "\\footnotesize",
        # 2026-10-28 (author: prune "pooled"; "the pooled regular-hours bars" of Panel F's note is kept: against CLOSE).
        # Was: "\\item \\textit{Notes:} The pooled models of Table~\\ref{tab:rh_oos} trained on other samples, ..."
        "\\item \\textit{Notes:} The models of Table~\\ref{tab:rh_oos} trained on other samples, everything else "
        "as in the main design: predictors on the full grid of \\rhtaNIntervals{} half-hour intervals, overnight "
        "included; a refit at every training row; the tree ensembles' frozen configurations; the linear penalties "
        "chosen in each sample by the main design's rule on that sample's pre-sample rows (validation year "
        "\\rhtaValYear). RTH13, the main design: the \\rhtaNBarsDay{} regular-hours bars of each day (bar ends 10:00 "
        "to 16:00~ET), a rolling window of \\rhtaWinDays{} trading days (\\rhtaRowsRth{} rows). ALL48: all "
        "\\rhtaNIntervals{} intervals of the futures' day, overnight, weekend and holiday intervals and early-close "
        "sessions included (with the overnight indicator), \\rhtaWinDays{} days (\\rhtaRowsAll{} rows). CLOSE: the "
        "16:00 bar only (the 15:30 forecast of the closing half hour), \\rhtaRowsCl{} closes. Penalized linear models "
        "only: CLOSE-nat builds the closing-bar design on the closing rows (its own scaling); ALL48@16 and RTH13@16 "
        "choose the penalties on the 16:00 validation rows (the primary sample's forecast where the choice is "
        "unchanged). Equal training rows: ALL48-3k and RTH13-3k train on the last \\rhtaRowsSmall{} rows of their "
        "sample (about \\rhtaSpanAllSmall{} and \\rhtaSpanRthSmall{} days; CLOSE's \\rhtaRowsCl{} closes span about "
        "\\rhtaSpanClSmall{} years), ALL48-39k on the last \\rhtaRowsLarge{} (about \\rhtaSpanAllLarge{} days; RTH13 "
        "has as many rows). Every sample is scored on the same \\rhtaNdays{} trading days, \\rhtaStart{} to "
        "\\rhtaEnd: the 16:00 bar and every regular-hours bar (\\rhtaNbars{} bars). Entries: the percentage change in "
        "mean QLIKE of the full predictor set relative to the HAR benchmark of the same model and sample (Panels A "
        "and B), or of the first-named sample relative to the second, full predictor set (Panel C); negative: the "
        "full set, or the first sample, has the lower loss. In parentheses, the $p$-value of the Diebold--Mariano "
        "test (Harvey--Leybourne--Newbold correction, Newey--West lag \\rhtaHacLag) on the daily 16:00 loss or the "
        "daily mean over the \\rhtaNBarsDay{} bars. The RTH13 rows are the forecasts of Tables~\\ref{tab:rh_oos} "
        "(every bar) and~\\ref{tab:rh_robust_w3500} (16:00, $W = \\rhtaWinDays$). --: no forecast; the tree "
        "ensembles have no CLOSE-nat, ALL48@16 or RTH13@16 arm (their configurations are not re-tuned).",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table}",
        "",
        # ---- the continuation: equal rows, Sharpe ratios, the close-only window sweep
        "\\begin{table}[p]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{1.5pt}",
        "\\begin{threeparttable}",
        "\\ContinuedFloat  % caption package: the same table number (inside threeparttable, before \\caption)",
        "\\caption{Robustness: the training sample (continued)}",
        "\\label{tab:rh_train_arms_cont}",
        tabular_open(2),
        "\\toprule",
        *header(2),
        "\\midrule",
        panel("Panel D: equal training rows, first sample vs second, full predictor set", NCOL2),
    ]
    for t in ("close", "bar"):
        L.append(block(TDESC[t].capitalize(), NCOL2))
        for a, b, tt in PAIRS_D:
            if tt == t:
                L.append(f"{short(a)} & {short(b)} & {pair_cells(D, a, b, t)} \\\\")
    L += ["\\addlinespace[4pt]", panel("Panel E: Sharpe ratio of the sign strategy, mid-quotes", NCOL2)]
    for arm in SH_ARMS:  # 2026-10-28 (D19): sr_mid, dsr_mid and the Ledoit-Wolf p; was d6_mid, dd6_mid, bootstrap p
        full, har = [], []
        for e in EST:
            f, h = sharpe(D, arm, e, "all_features"), sharpe(D, arm, e, "baseline")
            full.append("--" if f is None else f"{tnum(f.sr_mid, 2)} {{\\scriptsize {tp(sharpe_p(D, arm, e))}}}")
            har.append("--" if h is None else tnum(h.sr_mid, 2))
        L.append(f"{short(arm)} & Full & " + " & ".join(full) + " \\\\")
        L.append(" & HAR & " + " & ".join(har) + " \\\\")
    L.append("\\addlinespace[2pt]")
    for a, b in SH_PAIRS:
        cells = []
        for e in EST:
            r = sharpe_diff(D, a, b, e)
            cells.append("--" if r is None else f"{tnum(r.dsr_mid, 2)} {{\\scriptsize {tp(r.p_mid)}}}")
        L.append(f"{short(a)} & $-$ {short(b)} & " + " & ".join(cells) + " \\\\")
    L += [
        "\\addlinespace[4pt]",
        panel("Panel F: CLOSE on windows of $W$ closes, full predictor set vs HAR benchmark, 16:00 bar", NCOL2),
    ]
    for w in WS_W:
        cells = [tcell(wsweep(D, w, e).pct, wsweep(D, w, e).p) for e in EST]
        L.append(f"$W = {thousands(w)}$ & & " + " & ".join(cells) + " \\\\")
    L += [
        "\\bottomrule",
        "\\end{tabular*}",
        "\\begin{tablenotes}[flushleft]",
        "\\footnotesize",
        "\\item \\textit{Notes:} Samples, entries and $p$-values as in Panels A--C. Panel D: the equal-row samples "
        "against one another and against their equal-calendar siblings. Panel E: the Sharpe ratio $\\SR$ of the sign "
        "strategy of Section~\\ref{sec:rh_econ_measures} on the \\rhtaOptDays{} post-warmup option days, at "
        # 2026-10-28 (D19): the p-values are Ledoit-Wolf's. Was: "... in parentheses after Full, the bootstrap
        # $p$-value of Full minus HAR (paired block bootstrap, Appendix~\\ref{sec:rh_methods_tests}), computed for the
        # linear models in RTH13 only (--). Rows ALL48 $-$ RTH13 and CLOSE $-$ RTH13: the difference in $\\SR$
        # between the full-set strategies of two samples, with its bootstrap $p$-value."
        "mid-quotes, for the forecast of the full predictor set (Full) and of the HAR benchmark (HAR); in parentheses "
        "after Full, the $p$-value of the \\citet{LedoitWolf2008} test of Full against HAR "
        "(Appendix~\\ref{sec:rh_methods_tests}), computed for the linear models in RTH13 only (--). Rows ALL48 $-$ "
        "RTH13 and CLOSE $-$ RTH13: the difference in $\\SR$ between the full-set strategies of two samples, with the "
        "$p$-value of the same test. Panel F: CLOSE on rolling windows of $W$ closes, everything else unchanged (the linear "
        "penalties re-chosen at each $W$ by the same rule; at $W = 1{,}000$ the penalized models use 1{,}001 closes; "
        "the tree ensembles keep the configurations chosen on the pooled regular-hours bars); $W = \\rhtaWinDays$ is "
        "the CLOSE row of Panel~A.",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table}",
    ]
    OUT_TAB.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- checks of the outputs


def check_macro_names(names: set[str]) -> None:
    """No \\rhta name is defined in another generated file, and none of ours collides with an existing macro."""
    for f in sorted(GEN.glob("*.tex")):
        if f.name in {OUT_NUM.name, OUT_TAB.name}:
            continue
        txt = f.read_text(encoding="utf-8", errors="replace")
        assert "command{\\rhta" not in txt, f
        for m in re.finditer(r"command\{\\([A-Za-z]+)\}", txt):
            assert m.group(1) not in names, (f, m.group(1))


def used_macros_defined(names: set[str]) -> None:
    """Every \\rhta macro of the table (and of the draft, when present) is defined."""
    files = [OUT_TAB] + [p for p in [ROOT / "drafts" / "2026-10-26_train_arms.tex"] if p.is_file()]
    for f in files:
        for m in re.finditer(r"\\(rhta[A-Za-z]+)", f.read_text(encoding="utf-8")):
            assert m.group(1) in names, (f.name, m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--gates-csv", type=Path, default=None, help="write the gate log here")
    args = ap.parse_args()
    for n in branch_tips():
        print(n)
    D = load()
    run_gates(D)
    if args.gates_csv:
        pd.DataFrame(GATES).to_csv(args.gates_csv, index=False)
    claims(D)
    # 2026-10-28 (D19): every tested Sharpe-ratio difference, Ledoit-Wolf p and the bootstrap check p
    D["TESTS"].to_csv(OUT_TESTS, index=False, float_format="%.6f")
    names = write_macros(D)
    check_macro_names(names)
    write_table(D)
    used_macros_defined(names)
    print(f"wrote {OUT_NUM.relative_to(REPO)} ({len(names)} macros), {OUT_TAB.relative_to(REPO)}")


if __name__ == "__main__":
    main()
