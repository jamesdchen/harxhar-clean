"""Multiple-comparison procedures for the pooled intraday forecasts (2026-10-25, note E14, Appendix I).

The paper names the model confidence set (MCS) of Hansen, Lunde and Nason (2011), the test for superior predictive
ability (SPA) of Hansen (2005) and the StepM procedure of Romano and Wolf (2005), and reported none of them (note E14).
This script runs them on the final pooled rerun (REHAUL_FINAL_RUN=1) and writes macros, two table floats and CSVs.

FORECASTS
  The 12 pooled forecasts of the final rerun (ridge, lasso, elastic net, LightGBM, XGBoost, random forest x {HAR
  benchmark, full predictor set}; writeup/pooled_linear_13.py, writeup/pooled_trees_13.py through the loaders of
  writeup/make_rehaul_oos_rerun.py) and the closing-interval HAR benchmark (master-table key sub_ols_baseline, read by
  experiments/rolling_eval_2026_10_15.scored).  The pooled ridge, lasso and elastic net on the HAR benchmark have
  numerically zero frozen penalties and reproduce the pooled OLS (Table 1; their losses differ from the OLS's by about
  1e-8 relative), so the MCS treats them as ONE model, represented by the pooled OLS forecast ("Linear, HAR"): 10
  distinct pooled forecasts, 11 with the closing-interval benchmark.  The SPA / StepM families pair each class's full
  set with its own HAR forecast (as Table 1's DM column), so the three linear HAR forecasts enter there as they are.

SAMPLES (accuracy)
  bar   every regular-hours bar (13 a day) on Table 1's 2,579 trading days 2014-01-02 .. 2024-04-30; the bar losses are
        averaged to one loss a day (as Table 1's DM tests), so the bootstrap resamples days;
  all   the 16:00 bar (the forecast of the closing half hour made at 15:30) on the same 2,579 days;
  oos   the 16:00 bar on T_oos, the 1,082 days 2020-01-03 .. 2024-04-30 of Section 6 (rolling.common_span);
  ch    the 16:00 bar on the days on which the closing-interval HAR benchmark exists (its first forecast is 25 June 2018:
        a 2,000-session window on a design that starts in July 2010), a supplement: the benchmark has no forecast on
        2014-01-02 .. 2018-06-22, so it is not in the 2014-2024 sets.
  Losses: QLIKE r - ln r - 1 with r = RV / sigma2 (main) and the squared error (RV - sigma2)^2 (secondary).

MCS (Hansen, Lunde and Nason 2011, Econometrica 79(2), 453-497)
  T_max: t_i = dbar_i. / sqrt(var*(dbar_i.)), dbar_i. = model i's mean loss minus the mean over the set, T_max = max_i
  t_i, eliminate argmax_i t_i;  T_R: t_ij = dbar_ij / sqrt(var*(dbar_ij)), T_R = max_ij |t_ij|, eliminate
  argmax_i max_j t_ij.  Variances and the null distributions from the bootstrap (statistics recentred at the sample
  values), the same resamples at every step.  Elimination runs to the last model, so every model has an MCS p-value:
  the running maximum of the step p-values up to its elimination (1 for the last model).  The (1 - a) MCS is the set of
  models with p >= a: the 90 % set (p >= 0.10) and the 75 % set (p >= 0.25).
  Bootstrap: the circular moving-block bootstrap of src/evaluation/model_confidence_set.py (resample=True; its
  RandomState block starts), computed as resampling counts (B_ACC replicates, blocks of BLOCK days, seed SEED_ACC);
  gate: the counts reproduce the module's replicate means, and model_confidence_set() stopped at a = 0.10 returns the
  set, elimination order and p-values of this implementation (T_max).  Block-length sensitivity at BLOCK_ALT.

SPA (Hansen 2005, JBES 23(4), 365-380) and StepM (Romano and Wolf 2005, Econometrica 73(4), 1237-1282)
  On d_k (positive: alternative k better than its benchmark).  SPA: experiments/close_kfull_testlib.spa_rc (studentized
  by the bootstrap standard deviation; p-values consistent / lower / upper), the p-value of H0: no alternative beats
  its benchmark.  StepM: the studentized one-sided step-down procedure; adjusted p-values by the step-down algorithm
  (adjusted p of the j-th largest t = max over i <= j of P*(max over the alternatives ranked i and below of
  Z*_k >= t_(i)), Z*_k = (theta*_k - theta_k) / sd*_k); gate: the rejections at 5 % and 10 % equal those of
  experiments/feature_mcs_spa.stepm, except where the adjusted p sits on the level (np.quantile interpolates the
  critical value; within 2 / B of the level, reported in stepm_on_level.csv).  Single-test p: P*(Z*_k >= t_k), the
  same bootstrap without adjustment.  Two-sided versions (on |t_k| and |Z*_k|) are written beside the one-sided ones,
  for comparison with the paper's two-sided Diebold-Mariano and fixed-b tests.
  Accuracy families (QLIKE and MSE, each on bar / all / oos):  af  the full set of each class against that class's HAR
  forecast (6);  bm  the full set of each class against the closing-interval HAR benchmark (6; oos and ch);  bmall
  every distinct pooled forecast against the closing-interval HAR benchmark (10; oos and ch), the multiple-testing
  counterpart of Section 6's pairwise tests against it.
  Strategy families (sign strategy of Section 7 on the 614 post-warmup days, mid-quotes and bid/ask):  short  each
  full-set strategy against the unconditional short straddle (6);  twin  each full-set strategy against the HAR-
  benchmark strategy of its class (6).  Statistics: the mean daily return differential, and the difference of the
  paper's Sharpe ratios (D6, hac_inference.d6_sharpe), bootstrapped as hac_inference.d6_diff_test does (paired
  circular block bootstrap, hac_inference._boot_index: block hi.LW_BLOCK, hi.LW_BOOT resamples, seed hi.D6_SEED, each
  day keeps its year and A_y), studentized by the bootstrap standard deviation of each difference.  Gate: the two-
  sided D6 p of each twin pair equals hac_inference.d6_diff_test (the p_bt of the paper's T1).

GATES (the run stops if one fails)
  Table 1's every-bar QLIKE (results/oos_final_2014start/table1_w3000/metrics.csv, 1e-12; and the printed 4-decimal
  cells of generated/rehaul_results_oos.tex) and its DM statistics (dm.csv); the 16:00 QLIKE of the 12 pooled forecasts
  and of the closing-interval benchmark on T_oos (\\rhrQ<M>Pool<X>, \\rhrOlsQ); the sign-strategy Sharpe ratios of the
  12 pooled forecasts at mid-quotes and bid/ask (\\rhrSh<M>Pool<X>, \\rhrShX<M>Pool<X>, among them \\rhrShEnetPoolFull
  = 1.94) and of the short straddle (\\rheShortSh, \\rheShortShX); the bootstrap and algorithm gates above.

WRITES (new files)
  writeup/generated/rehaul_mcs_numbers.tex   \\rhmcs... macros (macro-only)
  writeup/generated/rehaul_mcs_table.tex     tab:rh_mcs (MCS p-values), tab:rh_mcs_spa (SPA / StepM)
  results/rehaul_mcs_2026-10-25/             mcs.csv, mcs_steps.csv, spa.csv, stepm.csv, stepm_on_level.csv,
                                             gates.csv (every block length; the .tex files use BLOCK)

Run:  REHAUL_FINAL_RUN=1 python writeup/make_rehaul_mcs.py
"""

from __future__ import annotations

import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent  # writeup/
REPO = ROOT.parent
for _p in (REPO, REPO / "notebooks", REPO / "experiments", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import final_run as fr  # noqa: E402

if not fr.FINAL:
    raise SystemExit("set REHAUL_FINAL_RUN=1: the MCS runs on the final pooled rerun")
import hac_inference as hi  # noqa: E402
import make_rehaul_oos_rerun as oos  # noqa: E402  (Table 1's loaders and scoring)
import pooled_linear_13 as pl13  # noqa: E402
import pooled_trees_13 as pt13  # noqa: E402
import rolling_eval_2026_10_15 as rolling  # noqa: E402
from close_kfull_testlib import spa_rc  # noqa: E402  (Hansen 2005 SPA, White 2000 RC)
from feature_mcs_spa import stepm as stepm_reject  # noqa: E402  (Romano-Wolf StepM, rejections at a level)

from src.evaluation import model_confidence_set as MCSM  # noqa: E402
from src.evaluation.diebold_mariano import dm_test  # noqa: E402

GEN = fr.GEN  # where the outputs go (REHAUL_GEN_DIR redirects a test run)
PRINTED = ROOT / "generated"  # the printed files the gates read
OUT_NUM = GEN / "rehaul_mcs_numbers.tex"
OUT_TAB = GEN / "rehaul_mcs_table.tex"
RES = fr.res_dir(REPO / "results" / "rehaul_mcs_2026-10-25")
T1_CSV = REPO / "results" / "oos_final_2014start" / f"table1_w{fr.WINDOW_DAYS}"
MASTER_DAILY = REPO / "results" / "close_master_table" / "master_table_daily.parquet"
QUOTES = REPO / "results" / "atm_straddle_0dte_1530" / "daily_a0.parquet"
HEAD = "% AUTO-GENERATED by writeup/make_rehaul_mcs.py -- do not edit; re-run the script."

# bootstrap settings
B_ACC = 4999  # replicates, accuracy (MCS, SPA, StepM)
BLOCK = 10  # block length in trading days (= hac_inference.LW_BLOCK, the paper's Sharpe-ratio bootstraps)
BLOCK_ALT = (5, 20)  # sensitivity of the MCS memberships
SEED_ACC = 20261025  # seed of the accuracy bootstrap
B_ECON, BLOCK_ECON, SEED_ECON = hi.LW_BOOT, hi.LW_BLOCK, hi.D6_SEED  # the paper's D6 bootstrap (4,999; 10; 20261022)
LEVELS_MCS = (0.10, 0.25)  # 90 % and 75 % sets
FWER = (0.05, 0.10)
ALPHA = 0.05
assert BLOCK == hi.LW_BLOCK and B_ACC == hi.LW_BOOT

N_OPT, WARMUP, N_PW = 866, 252, 614
OLS_KEY = "sub_ols_baseline"

EST = ("ridge", "lasso", "enet", "lgbm", "xgb", "rf")
LIN = ("ridge", "lasso", "enet")
EST_NAME = {
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "Elastic net",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "Random forest",
}
EST_TAG = {"ridge": "Ridge", "lasso": "Lasso", "enet": "Enet", "lgbm": "Lgbm", "xgb": "Xgb", "rf": "Rf"}
SET_TAG = {"baseline": "Har", "all_features": "Full"}
SET_NAME = {"baseline": "HAR", "all_features": "Full"}
# the MCS models: the three linear HAR forecasts as one (the pooled OLS), then the full linear, then the trees
MCS_MODELS: list[tuple[str, str]] = [("ols", "baseline")] + [(e, "all_features") for e in LIN]
MCS_MODELS += [(e, s) for e in ("lgbm", "xgb", "rf") for s in ("baseline", "all_features")]
BENCH = ("bench", "bench")  # the closing-interval HAR benchmark


def mtag(k: tuple[str, str]) -> str:
    if k == ("ols", "baseline"):
        return "LinHar"
    if k == BENCH:
        return "Bench"
    return EST_TAG[k[0]] + SET_TAG[k[1]]


def mname(k: tuple[str, str]) -> str:
    if k == ("ols", "baseline"):
        return "Linear, HAR"
    if k == BENCH:
        return "Closing-interval HAR benchmark"
    return f"{EST_NAME[k[0]]}, {SET_NAME[k[1]]}"


def mname_text(k: tuple[str, str]) -> str:
    """The model in running text."""
    if k == ("ols", "baseline"):
        return "the linear models on the HAR benchmark"
    if k == BENCH:
        return "the closing-interval HAR benchmark"
    what = "the full predictor set" if k[1] == "all_features" else "the HAR benchmark"
    nm = {
        "ridge": "ridge regression",
        "lasso": "the lasso",
        "enet": "the elastic net",
        "rf": "the random forest",
    }.get(k[0], EST_NAME[k[0]])
    return f"{nm} on {what}"


GATES: list[dict] = []
BOUNDARY: list[dict] = []  # StepM decisions on the level (run_family)


def gate(name: str, item: str, value: object, ok: bool) -> None:
    # 2026-10-26 (integration plan D2/D3/D9): the draft's claims on T_close (16:00, the 1,082 days with the
    # closing-interval HAR benchmark, now the rebuilt CI-HAR) belong to the closing-interval appendix section; they are
    # reported, not asserted (a changed one prints CHANGED and is listed in gates.csv with ok = False)
    reported = name == "draft claim" and item.startswith("16:00 T_oos")
    # 2026-10-26 (D13): on the patched design the elastic net's 16:00 full-vs-HAR increase on T_close is significant
    # too (p 0.036; old: ridge and lasso only), the same change close_accuracy reports; reported, not asserted
    reported = reported or (name == "Section 6" and "exactly for ridge and lasso" in item)
    # 2026-10-26 (D13): the draft claims below were written on the pre-patch design and fail on the patched one; they
    # are REPORTED, not asserted, until the text is rewritten (old -> new, patched design):
    #   every bar 90 % MCS: {LightGBM Full} -> {LightGBM Full, XGBoost Full} (T_max, T_R, blocks 5/10/20)
    #   16:00 2014-24, T_max 90 % MCS: every forecast -> all but the lasso Full
    #   16:00 2014-24, 75 % MCS: {XGBoost Full} -> {LightGBM Full, XGBoost Full}
    #   16:00 2014-24, one-sided StepM: XGBoost alone at blocks 5/10/20 -> at blocks 5 and 10 only (none at 20)
    #   vs short, mean: five rejected (all but XGBoost) -> three (lasso, elastic net, random forest)
    #   vs HAR twin, mean: LightGBM and forest (one-sided), forest (two-sided) -> forest only (one-sided); forest
    #   two-sided at blocks 10/20 only
    reported = reported or (
        name == "draft claim"
        and fr.FOMCFIX
        and item
        in {
            "every bar, QLIKE: the 90 % MCS is LightGBM Full alone (T_max, T_R; blocks 5, 10, 20)",
            "16:00 2014-2024, QLIKE: every forecast is in the 90 % MCS under T_max",
            "16:00 2014-2024, QLIKE: the 75 % MCS is XGBoost Full alone (T_max, T_R)",
            "16:00 2014-2024: StepM rejects XGBoost alone (one-sided; blocks 5, 10, 20)",
            "vs short, mean: StepM rejects five, all but XGBoost, which sits on the level (mid and bid/ask)",
            "vs HAR twin, mean: StepM rejects LightGBM and the random forest (one-sided), the forest alone (two-sided)",
        }
    )
    GATES.append({"gate": name, "item": item, "value": str(value), "ok": bool(ok)})
    print(f"  [{'ok' if ok else ('CHANGED (reported)' if reported else 'FAIL')}] {name}: {item} ({value})")
    assert ok or reported, (name, item, value)


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------------- printed values


def macros_of(path: Path) -> dict[str, str]:
    """name -> value of the one-line macro definitions of a generated numbers file (both forms)."""
    out: dict[str, str] = {}
    renew = re.compile(r"\\renewcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*(?:%.*)?$")
    provide = re.compile(r"^\\providecommand\{\\([A-Za-z]+)\}\{(.*)\}\s*(?:%.*)?$")
    for line in path.read_text(encoding="utf-8").splitlines():
        m = (renew if "\\renewcommand" in line else provide).search(line.split("  % ")[0] if "  % " in line else line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def num_of(s: str) -> float:
    return float(s.replace("\\ensuremath{-}", "-").replace("\\ensuremath{+}", "").replace("$-$", "-"))


# --------------------------------------------------------------------------------------------- data: accuracy


def accuracy_data() -> dict:
    """Daily loss matrices of every sample, gated against Table 1 and Section 6."""
    cfg = oos.SPANS["2014"]
    rv = oos.core()
    days = oos.evaluation_days(rv, cfg["first"], cfg["last"], cfg["n_days"])
    fc = oos.forecasts("13", cfg["r13"])
    idx = oos.scored_set(fc, rv, days)
    ft = oos.tree_forecasts(cfg)
    gate(
        "scored bars",
        "the trees forecast every bar the linear rows score",
        len(idx),
        oos.scored_set({**fc, **ft}, rv, days).equals(idx),
    )
    fa = {**fc, **ft}
    met = pd.read_csv(T1_CSV / "metrics.csv")
    met = met[met["panel"].astype(str) == "13"].set_index(["model", "set"])
    dmc = pd.read_csv(T1_CSV / "dm.csv")
    dmc = dmc[(dmc["panel"].astype(str) == "13") & (dmc["comparison"] == "all features vs HAR benchmark")]
    printed = (PRINTED / "rehaul_results_oos.tex").read_text(encoding="utf-8")
    dayidx = pd.DatetimeIndex(idx.normalize())
    udays = pd.DatetimeIndex(dayidx.unique()).sort_values()
    assert len(udays) == cfg["n_days"] == 2579, len(udays)
    at16 = (idx.hour == 16) & (idx.minute == 0)
    assert at16.sum() == len(udays) and pd.DatetimeIndex(idx[at16].normalize()).equals(udays)
    bar_q, bar_m, c16_q, c16_m = {}, {}, {}, {}
    q_all: dict[tuple[str, str], float] = {}
    for k, f in fa.items():
        q, se = oos.loss(f, idx, rv)
        q_all[k] = float(q.mean())
        ok = np.isclose(q.mean(), met.loc[k, "qlike"], rtol=1e-12, atol=0) and np.isclose(
            se.mean(), met.loc[k, "mse"], rtol=1e-12, atol=0
        )
        gate("Table 1", f"every-bar QLIKE and MSE of {k} = metrics.csv (rtol 1e-12)", f"QLIKE {q.mean():.6f}", ok)
        bar_q[k] = oos.daily(q, idx)
        bar_m[k] = oos.daily(se, idx)
        c16_q[k] = q[at16]
        c16_m[k] = se[at16]
    # the printed cells: QLIKE to 4 decimals in each row of rehaul_results_oos.tex
    rows = {
        "OLS": "ols",
        "Ridge": "ridge",
        "Lasso": "lasso",
        "Elastic net": "enet",
        "LightGBM": "lgbm",
        "XGBoost": "xgb",
        "Random forest": "rf",
    }
    for lab, e in rows.items():
        line = next(ln for ln in printed.splitlines() if ln.startswith(lab + " &"))
        cells = [c.strip() for c in line.rstrip("\\ ").split("&")]
        got = [cells[2]] + ([cells[5]] if e != "ols" else [])
        mine = [f"{q_all[(e, s)]:.4f}" for s in ("baseline", "all_features") if (e, s) in q_all]
        gate("Table 1 printed", f"{lab}: every-bar QLIKE (HAR, Full)", f"printed {got}, recomputed {mine}", got == mine)
    for _, r in dmc.iterrows():
        if r["loss"] != "qlike":
            continue
        e = r["model"]
        x = dm_test(bar_q[(e, "all_features")], bar_q[(e, "baseline")])
        gate(
            "Table 1 DM",
            f"{e}: DM of the daily QLIKE, Full - HAR",
            f"{x['dm']:.4f} vs {r['dm']:.4f}",
            np.isclose(x["dm"], r["dm"], rtol=1e-10, atol=0) and int(x["T"]) == 2579,
        )
    # the 16:00 bar on T_oos: the common rolling span of Section 6, and the closing-interval benchmark
    cs = rolling.common_span()
    hist = rolling.history()
    in_oos = udays >= cs[0]
    gate(
        "T_oos",
        "rolling.common_span() = Table 1 days from 2020-01-03",
        f"{len(cs)} days",
        pd.DatetimeIndex(udays[in_oos]).equals(cs) and len(cs) == 1082,
    )
    rv16 = hist.reindex(udays).to_numpy(float)
    for k, f in fa.items():
        y = f["rv"].reindex(idx[at16]).to_numpy(float)
        assert np.allclose(y, rv16, rtol=1e-12, atol=0), k
    sc = rolling.scored(OLS_KEY)
    gb = sc.reindex(cs)
    assert gb["plain"].notna().all() and np.allclose(
        gb["rv"].to_numpy(float), hist.reindex(cs).to_numpy(float), rtol=0, atol=0
    )
    pb = gb["plain"].to_numpy(float)
    yb = hist.reindex(cs).to_numpy(float)
    rb = yb / pb
    bq_oos, bm_oos = rb - np.log(rb) - 1.0, (yb - pb) ** 2
    rm = macros_of(PRINTED / "rehaul_results_numbers.tex")
    gate(
        "Section 6",
        "QLIKE of the closing-interval HAR benchmark on T_oos = \\rhrOlsQ",
        f"{bq_oos.mean():.4f} vs {rm['rhrOlsQ']}",
        f"{bq_oos.mean():.4f}" == rm["rhrOlsQ"],
    )
    for e in EST:
        for s in ("baseline", "all_features"):
            mname_ = f"rhrQ{EST_TAG[e]}Pool{SET_TAG[s]}"
            v = float(c16_q[(e, s)][in_oos].mean())
            ok = f"{v:.4f}" == rm[mname_] and np.isclose(v, met.loc[(e, s), "qlike_1600"], rtol=1e-12, atol=0)
            gate("Section 6", f"16:00 QLIKE on T_oos, {e} {s} = \\{mname_}", f"{v:.4f} vs {rm[mname_]}", ok)
        x = dm_test(c16_q[(e, "all_features")][in_oos], c16_q[(e, "baseline")][in_oos])
        want = rm[f"rhrAf{EST_TAG[e]}PoolDm"]
        gate(
            "Section 6",
            f"16:00 DM on T_oos, {e} Full - HAR = \\rhrAf{EST_TAG[e]}PoolDm",
            f"{num_of(want):.2f} vs {x['dm']:.4f}",
            f"{x['dm']:.2f}" == f"{num_of(want):.2f}",
        )
        # the draft's "significant in the pairwise test": the increases of ridge regression and the lasso only
        gate(
            "Section 6",
            f"16:00 DM on T_oos, {e} Full - HAR: p < 0.05 exactly for ridge and lasso (both increases)",
            f"p {x['p']:.4f}",
            (x["p"] < ALPHA) == (e in ("ridge", "lasso")) and (e not in ("ridge", "lasso") or x["dm"] > 0),
        )
        # against the closing-interval benchmark: \rhrVsBm<M>Pool<X>Dm, and the draft's "each significant in the
        # pairwise test" (every HAR forecast, the linear full-set forecasts and the random forest's: higher loss)
        for s in ("baseline", "all_features"):
            xb = dm_test(c16_q[(e, s)][in_oos], bq_oos)
            nm_ = f"rhrVsBm{EST_TAG[e]}Pool{SET_TAG[s]}Dm"
            sig = s == "baseline" or e in LIN + ("rf",)
            gate(
                "Section 6",
                f"16:00 DM on T_oos vs the benchmark, {e} {s} = \\{nm_}; significant (higher loss) iff {sig}",
                f"{xb['dm']:.4f} vs {rm[nm_]}, p {xb['p']:.4f}",
                f"{xb['dm']:.2f}" == f"{num_of(rm[nm_]):.2f}"
                and (xb["p"] < ALPHA) == sig
                and (not sig or xb["dm"] > 0),
            )
    # the closing-interval benchmark's own span (supplement)
    ch_days = pd.DatetimeIndex(udays.intersection(sc.index)).sort_values()
    gch = sc.reindex(ch_days)
    assert np.allclose(gch["rv"].to_numpy(float), hist.reindex(ch_days).to_numpy(float), rtol=0, atol=0)
    pch, ych = gch["plain"].to_numpy(float), gch["rv"].to_numpy(float)
    in_ch = udays.isin(ch_days)
    bq_ch, bm_ch = ych / pch - np.log(ych / pch) - 1.0, (ych - pch) ** 2
    lin_gap = max(float(np.max(np.abs(bar_q[(e, "baseline")] / bar_q[("ols", "baseline")] - 1.0))) for e in LIN)
    return dict(
        days=udays,
        oos_days=cs,
        ch_days=ch_days,
        in_oos=in_oos,
        in_ch=in_ch,
        bar_q=bar_q,
        bar_m=bar_m,
        c16_q=c16_q,
        c16_m=c16_m,
        bench_q_oos=bq_oos,
        bench_m_oos=bm_oos,
        bench_q_ch=bq_ch,
        bench_m_ch=bm_ch,
        lin_gap=lin_gap,
        n_bars=len(idx),
    )


# --------------------------------------------------------------------------------------------- data: trades


def crossed(q: np.ndarray, ex: np.ndarray, bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    """Bid/ask return (make_rehaul_results.crossed): a buy pays the ask, a sale gets the bid, spread at entry only."""
    return np.where(q > 0, q * (ex / ask - 1.0), np.where(q < 0, q * (ex / bid - 1.0), 0.0))


def trade_data() -> dict:
    """The sign strategies of the 12 pooled forecasts and the short straddle on the 866 option days (the construction
    of make_rehaul_results.frame / make_rehaul_economics.add_pooled_linear), gated against the printed Sharpe ratios."""
    d = pd.read_parquet(MASTER_DAILY, columns=["key", "day", "iv_var", "rv_unclipped", "ret_mid", "q"])
    d = d[d["key"] == OLS_KEY].copy()
    d["day"] = pd.to_datetime(d["day"]).dt.normalize()
    ref = d.set_index("day").sort_index()
    quotes = pd.read_parquet(QUOTES, columns=["R", "exit", "bid_c", "bid_p", "ask_c", "ask_p"])
    quotes.index = pd.to_datetime(quotes.index).normalize()
    j = ref.join(quotes, how="inner")
    assert len(j) == N_OPT, len(j)
    assert np.allclose(j["ret_mid"], j["q"] * j["R"], rtol=1e-12, atol=1e-15), "quote-book R"
    days = pd.DatetimeIndex(j.index).as_unit("ns")
    iv, rvu = j["iv_var"].to_numpy(float), j["rv_unclipped"].to_numpy(float)
    R, ex = j["R"].to_numpy(float), j["exit"].to_numpy(float)
    bid = (j["bid_c"] + j["bid_p"]).to_numpy(float)
    ask = (j["ask_c"] + j["ask_p"]).to_numpy(float)
    mid: dict[tuple[str, str], np.ndarray] = {}
    bax: dict[tuple[str, str], np.ndarray] = {}
    for e in EST:
        for s in ("baseline", "all_features"):
            a = (pl13 if e in LIN else pt13).at16(e, s)
            a.index = a.index.astype(days.dtype)
            a = a.reindex(days)
            p = a["plain"].to_numpy(float)
            assert np.isfinite(p).all() and (p > 0).all(), (e, s)
            assert np.allclose(a["rv"].to_numpy(float), rvu, rtol=1e-12, atol=0), (e, s)
            q = np.where(p > iv, 1.0, -1.0)
            mid[(e, s)] = q * R
            bax[(e, s)] = crossed(q, ex, bid, ask)
    short_mid, short_ba = -R, 1.0 - ex / bid
    pw = days[WARMUP:]
    assert len(pw) == N_PW

    def d6(x: np.ndarray) -> float:
        return float(hi.d6_sharpe(x[WARMUP:], dates=pw)["sharpe"])

    rm = macros_of(PRINTED / "rehaul_results_numbers.tex")
    em = macros_of(PRINTED / "rehaul_econ_numbers.tex")
    for e in EST:
        for s in ("baseline", "all_features"):
            for col, x in (("Sh", mid[(e, s)]), ("ShX", bax[(e, s)])):
                name = f"rhr{col}{EST_TAG[e]}Pool{SET_TAG[s]}"
                v = d6(x)
                gate(
                    "Section 7",
                    f"D6 Sharpe ratio \\{name}",
                    f"{v:.4f} vs {rm[name]}",
                    f"{v:.2f}" == f"{num_of(rm[name]):.2f}",
                )
    for name, x in (("rheShortSh", short_mid), ("rheShortShX", short_ba)):
        v = d6(x)
        gate(
            "Section 7", f"D6 Sharpe ratio \\{name}", f"{v:.4f} vs {em[name]}", f"{v:.2f}" == f"{num_of(em[name]):.2f}"
        )
    same_lin = all(np.array_equal(mid[(e, "baseline")], mid[("ridge", "baseline")]) for e in LIN) and all(
        np.array_equal(bax[(e, "baseline")], bax[("ridge", "baseline")]) for e in LIN
    )
    return dict(days=days, pw=pw, mid=mid, bax=bax, short_mid=short_mid, short_ba=short_ba, same_lin=same_lin)


# --------------------------------------------------------------------------------------------- bootstrap


def cb_means(L: np.ndarray, B: int, block: int, seed: int, chunk: int = 500) -> np.ndarray:
    """(B, m) replicate column means of L (T, m) under the circular moving-block bootstrap of
    src/evaluation/model_confidence_set._boot_col_means(resample=True): the same RandomState(seed) block starts (all
    drawn at once), computed through resampling counts, chunk replicates at a time."""
    T, m = L.shape
    rng = np.random.RandomState(seed)
    nblk = int(np.ceil(T / block))
    starts = rng.randint(0, T, size=(B, nblk))
    offs = np.arange(block)
    out = np.empty((B, m))
    for s0 in range(0, B, chunk):
        st = starts[s0 : s0 + chunk]
        c = len(st)
        idx = ((st[:, :, None] + offs).reshape(c, -1) % T)[:, :T]
        flat = (idx + (np.arange(c) * T)[:, None]).ravel()
        W = np.bincount(flat, minlength=c * T).reshape(c, T).astype(np.float64)
        out[s0 : s0 + c] = W @ L / T
    return out


def mcs(lbar: np.ndarray, lstar: np.ndarray, stat: str) -> dict:
    """Model confidence set p-values of every model (Hansen, Lunde and Nason 2011), elimination to the last model.
    lbar (m,) sample mean losses, lstar (B, m) bootstrap replicate means (the same resamples at every step).
    stat "max": T_max with t_i. (elimination argmax_i t_i.); "R": T_R = max |t_ij| (elimination argmax_i max_j t_ij).
    Returns {p (m,), order (eliminated first .. survivor last), step_p, step_stat}."""
    m = len(lbar)
    active = list(range(m))
    pv = np.empty(m)
    order: list[int] = []
    step_p: list[float] = []
    step_stat: list[float] = []
    run = 0.0
    while len(active) > 1:
        a = np.asarray(active)
        if stat == "max":
            dbar = lbar[a] - lbar[a].mean()
            ds = lstar[:, a] - lstar[:, a].mean(axis=1, keepdims=True)
            var = np.maximum(((ds - dbar) ** 2).mean(axis=0), 1e-300)
            t = dbar / np.sqrt(var)
            t0 = float(t.max())
            ts = ((ds - dbar) / np.sqrt(var)).max(axis=1)
            worst = int(np.argmax(t))
        elif stat == "R":
            k = len(a)
            dbar = lbar[a][:, None] - lbar[a][None, :]
            ds = lstar[:, a][:, :, None] - lstar[:, a][:, None, :]
            var = ((ds - dbar) ** 2).mean(axis=0)
            off = ~np.eye(k, dtype=bool)
            var = np.where(off, np.maximum(var, 1e-300), 1.0)
            t = np.where(off, dbar / np.sqrt(var), 0.0)
            t0 = float(np.abs(t).max())
            ts = (np.abs(ds - dbar) / np.sqrt(var) * off).max(axis=(1, 2))
            worst = int(np.argmax(np.where(off, t, -np.inf).max(axis=1)))
        else:
            raise ValueError(stat)
        p = float(np.mean(ts >= t0))
        run = max(run, p)
        i = active[worst]
        pv[i] = run
        order.append(i)
        step_p.append(p)
        step_stat.append(t0)
        del active[worst]
    pv[active[0]] = 1.0
    order.append(active[0])
    return {"p": pv, "order": order, "step_p": step_p, "step_stat": step_stat}


def spa_theta(theta: np.ndarray, theta_star: np.ndarray, n: int) -> dict:
    """spa_rc (experiments/close_kfull_testlib.py) for a statistic theta (m,) with bootstrap replicates theta_star (B, m)
    in place of the sample means of d: omega_k^2 = n x the bootstrap variance of theta_k; consistent / lower / upper
    p-values.  Equals spa_rc when theta is the mean of d (gated)."""
    om = np.sqrt(n * ((theta_star - theta) ** 2).mean(axis=0))
    t_spa = max(0.0, float(np.max(np.sqrt(n) * theta / om)))
    A = np.sqrt(om**2 / n * 2.0 * np.log(np.log(n)))
    g = {"lower": np.maximum(theta, 0.0), "consistent": theta * (theta >= -A), "upper": theta}
    out: dict = dict(stat_spa=t_spa, best=int(np.argmax(np.sqrt(n) * theta / om)))
    for k, gk in g.items():
        ts = np.maximum(0.0, np.max(np.sqrt(n) * (theta_star - gk) / om, axis=1))
        out[f"p_{k}"] = float(np.mean(ts > t_spa))
    return out


def stepm_adjusted(theta: np.ndarray, theta_star: np.ndarray) -> dict:
    """Romano-Wolf StepM, studentized, one-sided (H_k: theta_k <= 0): step-down adjusted p-values and the single-test
    bootstrap p-values.  t_k = theta_k / sd*_k, Z*_k = (theta*_k - theta_k) / sd*_k."""
    sd = np.sqrt(((theta_star - theta) ** 2).mean(axis=0))
    t = theta / sd
    Z = (theta_star - theta) / sd

    def stepdown(stat: np.ndarray, null: np.ndarray) -> np.ndarray:
        order = np.argsort(-stat, kind="stable")
        adj = np.empty(len(stat))
        run = 0.0
        for j, k in enumerate(order):
            p = float(np.mean(null[:, order[j:]].max(axis=1) >= stat[k]))
            run = max(run, p)
            adj[k] = run
        return adj

    # one-sided (H_k: theta_k <= 0) and two-sided (H_k: theta_k = 0, on |t| and |Z*|) step-down adjusted p-values
    adj, adj2 = stepdown(t, Z), stepdown(np.abs(t), np.abs(Z))
    raw, raw2 = (Z >= t).mean(axis=0), (np.abs(Z) >= np.abs(t)).mean(axis=0)
    return {"t": t, "adj": adj, "raw": raw, "adj2": adj2, "raw2": raw2, "sd": sd}


# --------------------------------------------------------------------------------------------- formatting


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def ptxt(p: float) -> str:
    """A p-value in text: 3 decimals below 0.10, else 2; <0.001; 1 as 1.00."""
    if p < 0.001:
        return "\\ensuremath{<}0.001"
    s = f"{p:.2f}" if p >= 0.10 else f"{p:.3f}"
    if s in ("0.10", "0.25"):  # keep the 90 % / 75 % cuts visible
        s = f"{p:.3f}"
    return s


def sci(x: float) -> str:
    """x as 'm \\times 10^{e}' with a one-digit mantissa (math mode)."""
    e = int(math.floor(math.log10(abs(x))))
    m = round(x / 10**e)
    if m == 10:
        m, e = 1, e + 1
    return f"{m} \\times 10^{{{e}}}"


def words(n: int) -> str:
    w = {
        0: "none",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
    }
    return w[n]


def join_names(names: list[str]) -> str:
    if not names:
        return "none"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + (", and " if len(names) > 2 else " and ") + names[-1]


def date_text(t: pd.Timestamp) -> str:
    return f"{t.day} {t.strftime('%B')} {t.year}"


# --------------------------------------------------------------------------------------------- main


def main() -> None:  # noqa: C901
    t_start = time.time()
    say(fr.tag())
    say("gates and accuracy data")
    acc = accuracy_data()
    say("trade data")
    trd = trade_data()
    RES.mkdir(parents=True, exist_ok=True)
    M: dict[str, tuple[str, str]] = {}  # macro name -> (value, comment)

    def mac(name: str, value: str, comment: str) -> None:
        assert re.fullmatch(r"rhmcs[A-Za-z]+", name), name
        assert name not in M, name
        M[name] = (value, comment)

    # ------------------------------------------------------------------ the samples' loss matrices (days x models)
    in_oos, in_ch = acc["in_oos"], acc["in_ch"]
    samples = {
        # tag: (label, loss source per model, mask of days, has the benchmark)
        "Bar": ("every bar, 2014--2024", "bar", None, False),
        "All": ("16:00 bar, 2014--2024", "c16", None, False),
        "Oos": ("16:00 bar, T_oos", "c16", in_oos, True),
        "Ch": ("16:00 bar, the benchmark's span", "c16", in_ch, True),
    }

    def loss_mat(tag: str, loss: str, keys: list[tuple[str, str]]) -> np.ndarray:
        _, src, mask, _ = samples[tag]
        cols = []
        for k in keys:
            if k == BENCH:
                cols.append(acc[f"bench_{'q' if loss == 'Q' else 'm'}_{'oos' if tag == 'Oos' else 'ch'}"])
                continue
            v = acc[f"{src}_{'q' if loss == 'Q' else 'm'}"][k]
            cols.append(v if mask is None else v[mask])
        return np.column_stack(cols)

    # ------------------------------------------------------------------ bootstrap gates (once, on the Bar QLIKE)
    say("bootstrap gates")
    Lg = loss_mat("Bar", "Q", MCS_MODELS)
    mine = cb_means(Lg, B_ACC, BLOCK, SEED_ACC)
    mod = MCSM._boot_col_means(Lg, B_ACC, BLOCK, SEED_ACC, resample=True)
    gate(
        "bootstrap",
        "the counts reproduce model_confidence_set._boot_col_means(resample=True)",
        f"max rel {np.max(np.abs(mine / mod - 1)):.1e}",
        np.allclose(mine, mod, rtol=1e-12, atol=0),
    )
    gate(
        "bootstrap", "replicate means vary", f"min sd {mine.std(axis=0).min():.2e}", bool((mine.std(axis=0) > 0).all())
    )
    for tag in ("Bar", "Oos"):
        keys = MCS_MODELS + ([BENCH] if samples[tag][3] else [])
        L = loss_mat(tag, "Q", keys)
        ref = MCSM.model_confidence_set(
            {mtag(k): L[:, i] for i, k in enumerate(keys)},
            alpha=0.10,
            B=B_ACC,
            block=BLOCK,
            seed=SEED_ACC,
            resample=True,
        )
        r = mcs(L.mean(axis=0), cb_means(L, B_ACC, BLOCK, SEED_ACC), "max")
        el = [mtag(keys[i]) for i in r["order"][: len(ref["eliminated"])]]
        same_p = all(
            np.isclose(ref["pvals"][mtag(keys[i])], r["p"][i], rtol=1e-12, atol=0)
            for i in r["order"][: len(ref["eliminated"])]
        )
        in90 = sorted(mtag(keys[i]) for i in range(len(keys)) if r["p"][i] >= 0.10)
        gate(
            "MCS algorithm",
            f"{tag} QLIKE: model_confidence_set (T_max, a = 0.10) = this implementation",
            f"eliminated {ref['eliminated']}; set {sorted(ref['mcs'])}",
            el == ref["eliminated"] and same_p and in90 == sorted(ref["mcs"]),
        )

    # ------------------------------------------------------------------ MCS
    say("MCS")
    mcs_rows, step_rows = [], []
    mcs_res: dict[tuple[str, str, str, int], dict] = {}
    mcs_keys: dict[str, list[tuple[str, str]]] = {}
    for tag, (label, _, _, has_b) in samples.items():
        keys = MCS_MODELS + ([BENCH] if has_b else [])
        mcs_keys[tag] = keys
        for loss in ("Q", "M"):
            L = loss_mat(tag, loss, keys)
            scale = 1e10 if loss == "M" else 1.0  # squared errors in units of 1e-10 (numerics only; p-values unchanged)
            L = L * scale
            for blk in (BLOCK,) + BLOCK_ALT:
                ls = cb_means(L, B_ACC, blk, SEED_ACC)
                for stat in ("max", "R"):
                    r = mcs(L.mean(axis=0), ls, stat)
                    mcs_res[(tag, loss, stat, blk)] = r
                    for i, k in enumerate(keys):
                        mcs_rows.append(
                            dict(
                                sample=tag,
                                label=label,
                                loss="qlike" if loss == "Q" else "mse",
                                stat=stat,
                                block=blk,
                                B=B_ACC,
                                seed=SEED_ACC,
                                T=len(L),
                                model=mtag(k),
                                mean_loss=float(L[:, i].mean() / scale),
                                p_mcs=float(r["p"][i]),
                                in90=bool(r["p"][i] >= 0.10),
                                in75=bool(r["p"][i] >= 0.25),
                            )
                        )
                    for j, (i, p, s) in enumerate(zip(r["order"], r["step_p"], r["step_stat"])):
                        step_rows.append(
                            dict(
                                sample=tag,
                                loss=loss,
                                stat=stat,
                                block=blk,
                                step=j + 1,
                                eliminated=mtag(keys[i]),
                                stat_value=s,
                                p_step=p,
                            )
                        )
            say(f"  MCS {tag} {loss}: done")
    pd.DataFrame(mcs_rows).to_csv(RES / "mcs.csv", index=False)
    pd.DataFrame(step_rows).to_csv(RES / "mcs_steps.csv", index=False)

    # ------------------------------------------------------------------ SPA / StepM, accuracy
    say("SPA and StepM, accuracy")
    spa_rows, stp_rows = [], []
    fam_res: dict[tuple[str, int], dict] = {}
    blocks = (BLOCK,) + BLOCK_ALT

    def run_family(
        fid: str,
        blk: int,
        d: np.ndarray | None,
        theta: np.ndarray,
        theta_star: np.ndarray,
        meta: dict,
        keys: list[tuple[str, str]] | None = None,
    ) -> None:
        """SPA and StepM of one family at one block length; gated against the repository's implementations (mean
        statistics, main block length).  keys: the alternatives (default: the six full-set forecasts)."""
        keys = keys if keys is not None else [(e, "all_features") for e in EST]
        alts = [mtag(k) for k in keys]
        n = int(meta["T"])
        sp = spa_theta(theta, theta_star, n)
        st = stepm_adjusted(theta, theta_star)
        if d is not None and blk == meta["main_block"]:
            ref = spa_rc(d, theta_star)
            gate(
                "SPA",
                f"{fid}: spa_theta = close_kfull_testlib.spa_rc",
                f"p_c {sp['p_consistent']:.4f}",
                all(
                    np.isclose(sp[f"p_{q}"], ref[f"p_{q}"], rtol=1e-12, atol=0)
                    for q in ("consistent", "lower", "upper")
                ),
            )
            for a_ in FWER:
                rej, _ = stepm_reject(d, theta_star, a_)
                mine_rej = {int(i) for i in np.flatnonzero(st["adj"] < a_)}
                diff = sorted(mine_rej ^ set(rej))
                # the two rules differ only where the adjusted p sits on the level (np.quantile interpolates the
                # critical value; the p-value counts exceedances): allowed within 2 / B of the level, and reported
                edge = all(abs(st["adj"][i] - a_) <= 2.0 / len(theta_star) for i in diff)
                note = ""
                if diff:
                    note = f" (on the level: {[alts[i] for i in diff]} at p = {[round(float(st['adj'][i]), 4) for i in diff]})"
                    BOUNDARY.append(
                        dict(
                            family=fid,
                            level=a_,
                            alternatives=[alts[i] for i in diff],
                            p=[float(st["adj"][i]) for i in diff],
                        )
                    )
                gate(
                    "StepM",
                    f"{fid}: adjusted p < {a_} = feature_mcs_spa.stepm rejections{note}",
                    f"{[alts[i] for i in sorted(rej)]}",
                    edge,
                )
        fam_res[(fid, blk)] = dict(spa=sp, stepm=st, theta=theta, keys=keys, **meta)
        m_ = {k: v for k, v in meta.items() if k != "main_block"}
        spa_rows.append(
            dict(
                family=fid,
                **m_,
                block=blk,
                n_alt=len(alts),
                stat_spa=sp["stat_spa"],
                p_consistent=sp["p_consistent"],
                p_lower=sp["p_lower"],
                p_upper=sp["p_upper"],
                best=alts[sp["best"]],
            )
        )
        for i, alt in enumerate(alts):
            stp_rows.append(
                dict(
                    family=fid,
                    **m_,
                    block=blk,
                    alternative=alt,
                    theta=float(theta[i]),
                    t=float(st["t"][i]),
                    p_single=float(st["raw"][i]),
                    p_stepm=float(st["adj"][i]),
                    reject5=bool(st["adj"][i] < 0.05),
                    reject10=bool(st["adj"][i] < 0.10),
                    p_single_two=float(st["raw2"][i]),
                    p_stepm_two=float(st["adj2"][i]),
                )
            )

    for tag in ("Bar", "All", "Oos", "Ch"):
        for loss in ("Q", "M"):
            scale = 1e10 if loss == "M" else 1.0
            fams = []
            if tag != "Ch":  # af: Full of each class against that class's own HAR forecast
                fams.append(("Af", "own HAR", loss_mat(tag, loss, [(e, "baseline") for e in EST]) * scale))
            if samples[tag][3]:  # bm: Full of each class against the closing-interval HAR benchmark
                fams.append(("Bm", "closing-interval HAR", loss_mat(tag, loss, [BENCH] * len(EST)) * scale))
            La = loss_mat(tag, loss, [(e, "all_features") for e in EST]) * scale
            for ftag, versus, Lb in fams:
                d = Lb - La
                for blk in blocks:
                    ds = cb_means(d, B_ACC, blk, SEED_ACC)
                    meta = dict(
                        kind="accuracy",
                        sample=tag,
                        loss=loss,
                        versus=versus,
                        stat="mean loss differential",
                        T=len(d),
                        B=B_ACC,
                        seed=SEED_ACC,
                        main_block=BLOCK,
                    )
                    run_family(f"{tag}{loss}{ftag}", blk, d, d.mean(axis=0), ds, meta)
            # all: every distinct pooled forecast (the three linear HAR forecasts as one) against the closing-interval
            # HAR benchmark, the multiple-testing counterpart of Section 6's pairwise tests against it
            if samples[tag][3]:
                Lb = loss_mat(tag, loss, [BENCH] * len(MCS_MODELS)) * scale
                d = Lb - loss_mat(tag, loss, MCS_MODELS) * scale
                for blk in blocks:
                    ds = cb_means(d, B_ACC, blk, SEED_ACC)
                    meta = dict(
                        kind="accuracy",
                        sample=tag,
                        loss=loss,
                        versus="closing-interval HAR (every pooled forecast)",
                        stat="mean loss differential",
                        T=len(d),
                        B=B_ACC,
                        seed=SEED_ACC,
                        main_block=BLOCK,
                    )
                    run_family(f"{tag}{loss}BmAll", blk, d, d.mean(axis=0), ds, meta, list(MCS_MODELS))

    # ------------------------------------------------------------------ SPA / StepM, strategies
    say("SPA and StepM, strategies")
    pw = trd["pw"]
    T = N_PW
    yy = hi.d6_years(pw)
    ys = sorted(yy["A"])
    code = np.searchsorted(np.array(ys), yy["years"])
    sqrt_a = np.sqrt(np.array([yy["A"][y] for y in ys], dtype=float))
    one = np.arange(T)[None, :]

    def d6_and_boot(x: np.ndarray, blk: int) -> tuple[float, np.ndarray]:
        """D6 of x and its replicates on hac_inference.d6_diff_test's resamples (block blk)."""
        idx_b = hi._boot_index(T, blk, B_ECON, SEED_ECON)
        sr = float(hi._d6_stat(x[one], code[one], sqrt_a)[0])
        star = np.empty(B_ECON)
        for s0 in range(0, B_ECON, 1000):
            ix = idx_b[s0 : s0 + 1000]
            star[s0 : s0 + len(ix)] = hi._d6_stat(x[ix], code[ix], sqrt_a)
        return sr, star

    def mean_boot(x: np.ndarray, blk: int) -> np.ndarray:
        idx_b = hi._boot_index(T, blk, B_ECON, SEED_ECON)
        out = np.empty(B_ECON)
        for s0 in range(0, B_ECON, 1000):
            ix = idx_b[s0 : s0 + 1000]
            out[s0 : s0 + len(ix)] = x[ix].mean(axis=1)
        return out

    sr_cache: dict[tuple[str, int], tuple[float, np.ndarray]] = {}

    def sr_of(name: str, x: np.ndarray, blk: int) -> tuple[float, np.ndarray]:
        if (name, blk) not in sr_cache:
            sr_cache[(name, blk)] = d6_and_boot(x, blk)
        return sr_cache[(name, blk)]

    econ_twin_p: dict[str, float] = {}
    for exe, src, short in (("Mid", trd["mid"], trd["short_mid"]), ("Ba", trd["bax"], trd["short_ba"])):
        for vs in ("Short", "Twin"):
            A = np.column_stack([src[(e, "all_features")][WARMUP:] for e in EST])
            if vs == "Short":
                Bm = np.column_stack([short[WARMUP:]] * len(EST))
                bnames = ["short"] * len(EST)
            else:
                Bm = np.column_stack([src[(e, "baseline")][WARMUP:] for e in EST])
                bnames = [f"{e}_baseline" for e in EST]
            d = A - Bm
            for blk in blocks:
                meta = dict(
                    kind="strategy",
                    sample="pw614",
                    loss=exe,
                    versus=vs,
                    T=T,
                    B=B_ECON,
                    seed=SEED_ECON,
                    main_block=BLOCK_ECON,
                )
                ds = np.column_stack([mean_boot(d[:, i], blk) for i in range(len(EST))])
                run_family(f"{exe}{vs}Mean", blk, d, d.mean(axis=0), ds, dict(meta, stat="mean return differential"))
                th, th_s = np.empty(len(EST)), np.empty((B_ECON, len(EST)))
                for i, e in enumerate(EST):
                    sa, sa_s = sr_of(f"{exe}_{e}_all", A[:, i], blk)
                    sb, sb_s = sr_of(f"{exe}_{bnames[i]}", Bm[:, i], blk)
                    th[i], th_s[:, i] = sa - sb, sa_s - sb_s
                    if vs == "Twin" and exe == "Mid" and blk == BLOCK_ECON:
                        two = float(np.mean(np.abs(th_s[:, i] - th[i]) >= abs(th[i])))
                        ref = hi.d6_diff_test(A[:, i], Bm[:, i], pw)
                        gate(
                            "D6 bootstrap",
                            f"{e}: two-sided p of the Full - HAR D6 difference = hac_inference.d6_diff_test",
                            f"{two:.4f} vs {ref['p']:.4f}",
                            two == ref["p"] and bool(np.isclose(th[i], ref["delta"], rtol=1e-12)),
                        )
                        econ_twin_p[e] = two
                run_family(f"{exe}{vs}Sr", blk, None, th, th_s, dict(meta, stat="D6 Sharpe ratio difference"))
            say(f"  {exe} vs {vs}: done")
    raw2 = fam_res[("MidTwinSr", BLOCK_ECON)]["stepm"]["raw2"]
    gate(
        "StepM",
        "MidTwinSr: two-sided single-test p = the paper's D6 bootstrap p (studentizing one difference cancels)",
        f"{np.round(raw2, 4).tolist()}",
        all(abs(raw2[i] - econ_twin_p[e]) <= 1.0 / B_ECON for i, e in enumerate(EST)),
    )
    pd.DataFrame(spa_rows).to_csv(RES / "spa.csv", index=False)
    pd.DataFrame(stp_rows).to_csv(RES / "stepm.csv", index=False)
    pd.DataFrame(BOUNDARY).to_csv(RES / "stepm_on_level.csv", index=False)

    # ------------------------------------------------------------------ macros
    say("macros")
    days, oos_days, ch_days = acc["days"], acc["oos_days"], acc["ch_days"]
    mac("rhmcsB", thousands(B_ACC), "bootstrap replications, every MCS / SPA / StepM (accuracy and strategies)")
    mac("rhmcsBlock", str(BLOCK), "block length in trading days (circular moving-block bootstrap), main")
    mac("rhmcsBlockLo", str(BLOCK_ALT[0]), "block length, sensitivity (shorter)")
    mac("rhmcsBlockHi", str(BLOCK_ALT[1]), "block length, sensitivity (longer)")
    mac("rhmcsSeed", str(SEED_ACC), "random seed of the accuracy bootstrap (MCS, SPA, StepM on losses)")
    mac("rhmcsSeedEcon", str(SEED_ECON), "random seed of the strategy bootstrap (= the paper's D6 bootstrap seed)")
    mac(
        "rhmcsNModels",
        str(len(MCS_MODELS)),
        "distinct pooled forecasts in the MCS (the three linear HAR forecasts as one)",
    )
    mac("rhmcsNModelsWord", words(len(MCS_MODELS)), "rhmcsNModels as a word")
    mac("rhmcsNModelsBench", str(len(MCS_MODELS) + 1), "... with the closing-interval HAR benchmark")
    mac("rhmcsNModelsBenchWord", words(len(MCS_MODELS) + 1), "rhmcsNModelsBench as a word")
    mac("rhmcsNPool", str(2 * len(EST)), "pooled forecasts (6 classes x HAR, Full)")
    mac(
        "rhmcsNAlt",
        str(len(EST)),
        "alternatives in every SPA / StepM family (the six full-set forecasts or strategies)",
    )
    mac("rhmcsNAltWord", words(len(EST)), "rhmcsNAlt as a word")
    mac(
        "rhmcsLinGap",
        sci(acc["lin_gap"]),
        "largest relative gap between the daily every-bar QLIKE of a linear HAR forecast and the pooled OLS (math mode)",
    )
    mac("rhmcsDaysAll", thousands(len(days)), "trading days 2014-2024 (Table 1's span)")
    mac("rhmcsDaysOos", thousands(len(oos_days)), "trading days of T_oos")
    mac(
        "rhmcsDaysCh", thousands(len(ch_days)), "trading days of the closing-interval benchmark's span within 2014-2024"
    )
    mac("rhmcsChStart", date_text(ch_days[0]), "first day of the closing-interval HAR benchmark's forecasts")
    mac("rhmcsNBars", thousands(acc["n_bars"]), "bars of the every-bar sample (13 a day)")
    mac("rhmcsNPw", str(N_PW), "post-warmup option days of the strategy tests")
    mac(
        "rhmcsLinSameTrades",
        "yes" if trd["same_lin"] else "no",
        "the three linear HAR sign strategies have identical daily returns (mid and bid/ask)",
    )

    stat_tag = {"max": "Tmax", "R": "Trange"}
    loss_tag = {"Q": "Qlike", "M": "Mse"}
    for tag, (label, _, _, _) in samples.items():
        keys = mcs_keys[tag]
        for loss in ("Q", "M"):
            for stat in ("max", "R"):
                r = mcs_res[(tag, loss, stat, BLOCK)]
                pre = f"rhmcs{tag}{loss_tag[loss]}{stat_tag[stat]}"
                for i, k in enumerate(keys):
                    mac(
                        f"{pre}P{mtag(k)}",
                        ptxt(float(r["p"][i])),
                        f"MCS p-value, {label}, {loss_tag[loss]}, {stat_tag[stat]}: {mname(k)}",
                    )
                for lev, lt in ((0.10, "Ninety"), (0.25, "SevenFive")):
                    mem = [i for i in r["order"][::-1] if r["p"][i] >= lev]
                    mac(
                        f"{pre}N{lt}",
                        str(len(mem)),
                        f"models in the {int(round(100 * (1 - lev)))} % MCS (of {len(keys)})",
                    )
                    mac(
                        f"{pre}Set{lt}",
                        join_names([mname_text(keys[i]) for i in mem]),
                        f"the {int(round(100 * (1 - lev)))} % MCS, best (last eliminated) first",
                    )
                    # stability across block lengths
                    same = all(
                        sorted(i for i in range(len(keys)) if mcs_res[(tag, loss, stat, b)]["p"][i] >= lev)
                        == sorted(mem)
                        for b in BLOCK_ALT
                    )
                    mac(
                        f"{pre}Stable{lt}",
                        "yes" if same else "no",
                        f"the {int(round(100 * (1 - lev)))} % MCS is the same at block lengths {BLOCK_ALT}",
                    )
                    out_ = [i for i in range(len(keys)) if r["p"][i] < lev]
                    mac(
                        f"{pre}PMaxOut{lt}",
                        ptxt(max(float(r["p"][i]) for i in out_)) if out_ else "--",
                        f"largest MCS p-value outside the {int(round(100 * (1 - lev)))} % set",
                    )
                    for b, bt in zip(BLOCK_ALT, ("Lo", "Hi")):
                        rb = mcs_res[(tag, loss, stat, b)]
                        memb = [i for i in rb["order"][::-1] if rb["p"][i] >= lev]
                        mac(f"{pre}N{lt}{bt}", str(len(memb)), f"... at block length {b}")
                        mac(
                            f"{pre}Set{lt}{bt}",
                            join_names([mname_text(keys[i]) for i in memb]),
                            f"... at block length {b}",
                        )
                mac(f"{pre}Best", mname_text(keys[r["order"][-1]]), "the last model left (MCS p-value 1)")

    for (fid, blk), fr_ in fam_res.items():
        if blk != BLOCK:
            continue
        pre = f"rhmcs{fid}"
        sp, st = fr_["spa"], fr_["stepm"]
        mac(f"{pre}SpaC", ptxt(sp["p_consistent"]), f"SPA p-value (consistent), family {fid}")
        mac(f"{pre}SpaL", ptxt(sp["p_lower"]), "... lower")
        mac(f"{pre}SpaU", ptxt(sp["p_upper"]), "... upper")
        nrej = int((st["adj"] < ALPHA).sum())
        nraw = int((st["raw"] < ALPHA).sum())
        mac(f"{pre}NRej", str(nrej), f"StepM rejections at FWER 5 % (of {len(fr_['keys'])})")
        mac(f"{pre}NRejWord", words(nrej), "... as a word")
        mac(f"{pre}NRaw", str(nraw), "single-test bootstrap p < 0.05 (no adjustment)")
        mac(f"{pre}NRawWord", words(nraw), "... as a word")
        mac(
            f"{pre}NPos",
            str(int((fr_["theta"] > 0).sum())),
            "alternatives with a positive estimate (better than the benchmark)",
        )
        mac(f"{pre}PMin", ptxt(float(st["adj"].min())), "smallest StepM-adjusted p-value")
        mac(f"{pre}PMax", ptxt(float(st["adj"].max())), "largest StepM-adjusted p-value")
        mac(f"{pre}PTwoMin", ptxt(float(st["adj2"].min())), "smallest two-sided StepM-adjusted p-value")
        mac(f"{pre}PTwoMax", ptxt(float(st["adj2"].max())), "largest two-sided StepM-adjusted p-value")
        mac(f"{pre}RawMin", ptxt(float(st["raw"].min())), "smallest single-test bootstrap p-value")
        mac(
            f"{pre}Rej",
            join_names([mname_text(k) for i, k in enumerate(fr_["keys"]) if st["adj"][i] < ALPHA]),
            "the alternatives StepM rejects at 5 % (none: none)",
        )
        nrej2 = int((st["adj2"] < ALPHA).sum())
        mac(f"{pre}NRejTwo", str(nrej2), "StepM rejections at FWER 5 %, two-sided")
        mac(f"{pre}NRejTwoWord", words(nrej2), "... as a word")
        mac(f"{pre}NRawTwo", str(int((st["raw2"] < ALPHA).sum())), "single-test bootstrap p < 0.05, two-sided")
        mac(
            f"{pre}RejTwo",
            join_names([mname_text(k) for i, k in enumerate(fr_["keys"]) if st["adj2"][i] < ALPHA]),
            "the alternatives the two-sided StepM rejects at 5 % (none: none)",
        )
        default = fr_["keys"] == [(e, "all_features") for e in EST]
        for i, k in enumerate(fr_["keys"]):
            tg, nm = (EST_TAG[k[0]], EST_NAME[k[0]]) if default else (mtag(k), mname(k))
            mac(f"{pre}P{tg}", ptxt(float(st["adj"][i])), f"StepM-adjusted p-value (one-sided), {nm}")
            mac(f"{pre}Raw{tg}", ptxt(float(st["raw"][i])), f"single-test bootstrap p-value (one-sided), {nm}")
            mac(f"{pre}PTwo{tg}", ptxt(float(st["adj2"][i])), f"StepM-adjusted p-value (two-sided), {nm}")
            mac(f"{pre}RawTwo{tg}", ptxt(float(st["raw2"][i])), f"single-test bootstrap p-value (two-sided), {nm}")
        for b, bt in zip(BLOCK_ALT, ("Lo", "Hi")):
            fb = fam_res[(fid, b)]
            mac(
                f"{pre}NRej{bt}",
                str(int((fb["stepm"]["adj"] < ALPHA).sum())),
                f"StepM rejections at 5 %, block length {b}",
            )
            mac(f"{pre}SpaC{bt}", ptxt(fb["spa"]["p_consistent"]), f"SPA p-value (consistent), block length {b}")
            mac(f"{pre}NRejTwo{bt}", str(int((fb["stepm"]["adj2"] < ALPHA).sum())), f"... two-sided, block length {b}")
            mac(f"{pre}PMin{bt}", ptxt(float(fb["stepm"]["adj"].min())), f"smallest adjusted p, block length {b}")
            mac(f"{pre}PTwoMin{bt}", ptxt(float(fb["stepm"]["adj2"].min())), f"... two-sided, block length {b}")
    for e in EST:
        mac(
            f"rhmcsTwinTwoSidedP{EST_TAG[e]}",
            ptxt(econ_twin_p[e]),
            f"two-sided D6 bootstrap p of the Full - HAR Sharpe difference, {EST_NAME[e]} (= the paper's T1 p_bt)",
        )

    draft_claims(mcs_res, mcs_keys, fam_res)

    # collision check against every other generated file
    other: set[str] = set()
    for p in PRINTED.glob("*.tex"):
        if p.name in (OUT_NUM.name, OUT_TAB.name):
            continue
        other |= set(
            re.findall(
                r"\\(?:re)?(?:provide|new)command\*?\{?\\([A-Za-z]+)", p.read_text(encoding="utf-8", errors="replace")
            )
        )
    clash = sorted(set(M) & other)
    gate(
        "macros",
        "no \\rhmcs macro is defined in another generated file",
        f"{len(M)} macros; clashes {clash}",
        not clash,
    )

    lines = [
        HEAD,
        "% \\rhmcs... macros: model confidence set (Hansen, Lunde and Nason 2011), SPA (Hansen 2005) and StepM (Romano and",
        "% Wolf 2005) on the 12 pooled forecasts of the final rerun (the three linear HAR forecasts as one) and the",
        "% closing-interval HAR benchmark. Samples: Bar (every bar, 2014-2024, daily-averaged losses), All (16:00, 2014-2024),",
        "% Oos (16:00, T_oos), Ch (16:00, the benchmark's span). Losses: Qlike, Mse. MCS statistics: Tmax, Trange (T_R).",
        "% SPA / StepM families: <Sample><Q|M>Af (Full vs own HAR), <Sample><Q|M>Bm (Full vs the closing-interval HAR),",
        "% <Mid|Ba><Short|Twin><Mean|Sr> (sign strategies on the 614 post-warmup days, vs the short straddle or the HAR twin,",
        "% mean return or D6 Sharpe ratio). p-values of StepM and SPA are one-sided (H_k: no better than the benchmark).",
    ]
    for name, (val, com) in M.items():
        lines.append(f"\\providecommand{{\\{name}}}{{}}\\renewcommand{{\\{name}}}{{{val}}}  % {com}")
    OUT_NUM.write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(f"wrote {fr.rel(OUT_NUM)} ({len(M)} macros)")

    # ------------------------------------------------------------------ tables
    OUT_TAB.write_text(table_mcs(mcs_res, mcs_keys, acc) + "\n" + table_spa(fam_res), encoding="utf-8")
    say(f"wrote {fr.rel(OUT_TAB)}")
    pd.DataFrame(GATES).to_csv(RES / "gates.csv", index=False)
    say(f"wrote {fr.rel(RES)} (mcs.csv, mcs_steps.csv, spa.csv, stepm.csv, gates.csv); {time.time() - t_start:.0f} s")


def draft_claims(mcs_res: dict, mcs_keys: dict, fam_res: dict) -> None:
    """DRAFT CLAIMS: the words of writeup/drafts/2026-10-25_mcs.tex that no macro carries ("only", "every", "none",
    "all", which forecasts drop out, which tests survive).  A rerun whose data no longer bear them out stops here."""

    def mset(tag: str, loss: str, stat: str, lev: float, blk: int = BLOCK) -> set[str]:
        r = mcs_res[(tag, loss, stat, blk)]
        return {mtag(k) for i, k in enumerate(mcs_keys[tag]) if r["p"][i] >= lev}

    def rej(fid: str, two: bool = False, blk: int = BLOCK) -> set[str]:
        f = fam_res[(fid, blk)]
        p = f["stepm"]["adj2" if two else "adj"]
        return {mtag(k) for i, k in enumerate(f["keys"]) if p[i] < ALPHA}

    full = {mtag((e, "all_features")) for e in EST}
    lin_full = {mtag((e, "all_features")) for e in LIN}
    allm = {mtag(k) for k in MCS_MODELS}
    c = "draft claim"
    # Section 6, every bar: only LightGBM on the full set is in the 90 % MCS, both statistics, every block length
    gate(
        c,
        "every bar, QLIKE: the 90 % MCS is LightGBM Full alone (T_max, T_R; blocks 5, 10, 20)",
        "",
        all(mset("Bar", "Q", s, 0.10, b) == {"LgbmFull"} for s in ("max", "R") for b in (BLOCK,) + BLOCK_ALT),
    )
    gate(
        c,
        "every bar, squared error: every forecast is in the 75 % MCS (T_max, T_R)",
        "",
        all(mset("Bar", "M", s, 0.25) == allm for s in ("max", "R")),
    )
    gate(
        c,
        "16:00 2014-2024, QLIKE: every forecast is in the 90 % MCS under T_max",
        "",
        mset("All", "Q", "max", 0.10) == allm,
    )
    gate(
        c,
        "16:00 2014-2024, QLIKE: the 75 % MCS is XGBoost Full alone (T_max, T_R)",
        "",
        all(mset("All", "Q", s, 0.25) == {"XgbFull"} for s in ("max", "R")),
    )
    gate(
        c,
        "16:00 T_oos, QLIKE: the closing-interval benchmark is the last model left (T_max, T_R)",
        "",
        all(mcs_keys["Oos"][mcs_res[("Oos", "Q", s, BLOCK)]["order"][-1]] == BENCH for s in ("max", "R")),
    )
    gate(
        c,
        "16:00 T_oos, QLIKE: no linear forecast (HAR or Full) is in the 90 % MCS (T_max, T_R)",
        "",
        all(not (mset("Oos", "Q", s, 0.10) & (lin_full | {"LinHar"})) for s in ("max", "R")),
    )
    gate(
        c,
        "16:00 T_oos, QLIKE: the 90 % MCS holds both boosted trees on both predictor sets (T_max, T_R)",
        "",
        all({"LgbmHar", "LgbmFull", "XgbHar", "XgbFull", "Bench"} <= mset("Oos", "Q", s, 0.10) for s in ("max", "R")),
    )
    gate(
        c,
        "squared error: every forecast is in the 75 % MCS at 16:00 (2014-2024 and T_oos; T_max, T_R)",
        "",
        all(mset(t, "M", s, 0.25) == {mtag(k) for k in mcs_keys[t]} for t in ("All", "Oos") for s in ("max", "R")),
    )
    # Section 6, SPA / StepM
    gate(
        c,
        "every bar: StepM rejects all six (one- and two-sided; blocks 5, 10, 20)",
        "",
        all(rej("BarQAf", tw, b) == full for tw in (False, True) for b in (BLOCK,) + BLOCK_ALT),
    )
    gate(
        c,
        "16:00 2014-2024: StepM rejects XGBoost alone (one-sided; blocks 5, 10, 20)",
        "",
        all(rej("AllQAf", False, b) == {"XgbFull"} for b in (BLOCK,) + BLOCK_ALT),
    )
    gate(c, "16:00 2014-2024: no two-sided StepM rejection", "", rej("AllQAf", True) == set())
    gate(
        c,
        "16:00 T_oos: no one-sided StepM rejection (Full vs own HAR, Full vs benchmark)",
        "",
        rej("OosQAf") == set() and rej("OosQBm") == set() and rej("OosQBmAll") == set(),
    )
    gate(
        c,
        "16:00 T_oos: the two-sided StepM rejects the lasso alone (Full worse than its HAR)",
        "",
        rej("OosQAf", True) == {"LassoFull"} and fam_res[("OosQAf", BLOCK)]["theta"][EST.index("lasso")] < 0,
    )
    want = {"LinHar", "RidgeFull", "LassoFull", "EnetFull", "RfHar"}
    gate(
        c,
        "16:00 T_oos, every pooled forecast vs the benchmark: two-sided StepM rejects these five (all worse)",
        "",
        rej("OosQBmAll", True) == want
        and all(fam_res[("OosQBmAll", BLOCK)]["theta"][[mtag(k) for k in MCS_MODELS].index(m)] < 0 for m in want),
    )
    gate(
        c,
        "squared error: no StepM rejection in any accuracy family (one-sided)",
        "",
        all(rej(f) == set() for f in ("BarMAf", "AllMAf", "OosMAf", "OosMBm", "ChMBm")),
    )
    # Section 7
    gate(
        c,
        "vs short, mean: StepM rejects five, all but XGBoost, which sits on the level (mid and bid/ask)",
        "",
        all(rej(f"{x}ShortMean") == full - {"XgbFull"} for x in ("Mid", "Ba"))
        and all(
            abs(fam_res[(f"{x}ShortMean", BLOCK)]["stepm"]["adj"][EST.index("xgb")] - ALPHA) <= 2.0 / B_ECON
            for x in ("Mid", "Ba")
        ),
    )
    gate(
        c,
        "vs short, mean, two-sided: StepM rejects the elastic net and the random forest (mid and bid/ask)",
        "",
        all(rej(f"{x}ShortMean", True) == {"EnetFull", "RfFull"} for x in ("Mid", "Ba")),
    )
    gate(
        c,
        "vs short, Sharpe ratio: StepM rejects all six (one- and two-sided, mid and bid/ask, blocks 5, 10, 20)",
        "",
        all(
            rej(f"{x}ShortSr", tw, b) == full
            for x in ("Mid", "Ba")
            for tw in (False, True)
            for b in (BLOCK,) + BLOCK_ALT
        ),
    )
    gate(
        c,
        "vs HAR twin, mean: StepM rejects LightGBM and the random forest (one-sided), the forest alone (two-sided)",
        "",
        all(
            rej(f"{x}TwinMean") == {"LgbmFull", "RfFull"} and rej(f"{x}TwinMean", True) == {"RfFull"}
            for x in ("Mid", "Ba")
        ),
    )
    gate(
        c,
        "vs HAR twin, Sharpe ratio: StepM rejects the random forest alone (one-sided at blocks 5, 10, 20; two-sided "
        "at block 10; mid and bid/ask)",
        "",
        all(rej(f"{x}TwinSr", False, b) == {"RfFull"} for x in ("Mid", "Ba") for b in (BLOCK,) + BLOCK_ALT)
        and all(rej(f"{x}TwinSr", True) == {"RfFull"} for x in ("Mid", "Ba")),
    )
    gate(
        c,
        "vs HAR twin: the smallest two-sided adjusted p is the random forest's at every block length (mean, Sharpe)",
        "",
        all(
            int(np.argmin(fam_res[(f"{x}Twin{s}", b)]["stepm"]["adj2"])) == EST.index("rf")
            for x in ("Mid", "Ba")
            for s in ("Mean", "Sr")
            for b in (BLOCK,) + BLOCK_ALT
        ),
    )
    raw2 = fam_res[("MidTwinSr", BLOCK)]["stepm"]["raw2"]
    gate(
        c,
        "vs HAR twin, Sharpe ratio: LightGBM has the second smallest single-test two-sided p (after the forest), and "
        "the forest's is the only one below 5 %",
        "",
        [EST[i] for i in np.argsort(raw2, kind="stable")[:2]] == ["rf", "lgbm"]
        and [EST[i] for i in np.flatnonzero(raw2 < ALPHA)] == ["rf"],
    )
    gate(
        c,
        "every bar, QLIKE: XGBoost Full is the last forecast eliminated before LightGBM Full (T_max, T_R)",
        "",
        all(
            [mtag(mcs_keys["Bar"][i]) for i in mcs_res[("Bar", "Q", s, BLOCK)]["order"][-2:]] == ["XgbFull", "LgbmFull"]
            for s in ("max", "R")
        ),
    )
    gate(
        c,
        "16:00 T_oos, QLIKE: the 90 % set under T_max is the one under T_R plus the random forest Full",
        "",
        mset("Oos", "Q", "max", 0.10) == mset("Oos", "Q", "R", 0.10) | {"RfFull"}
        and "RfFull" not in mset("Oos", "Q", "R", 0.10),
    )
    gate(
        c,
        "16:00 T_oos, QLIKE: the 75 % set is the benchmark and XGBoost Full (T_max, T_R), the benchmark alone with "
        "blocks of 20 days",
        "",
        all(mset("Oos", "Q", s, 0.25) == {"Bench", "XgbFull"} for s in ("max", "R"))
        and all(mset("Oos", "Q", s, 0.25, BLOCK_ALT[1]) == {"Bench"} for s in ("max", "R")),
    )
    gate(
        c,
        "16:00 on the benchmark's own span, QLIKE: the closing-interval benchmark is the last model left (T_max, T_R)",
        "",
        all(mcs_keys["Ch"][mcs_res[("Ch", "Q", s, BLOCK)]["order"][-1]] == BENCH for s in ("max", "R")),
    )
    gate(
        c,
        "every SPA / StepM estimate vs the short straddle and the HAR twin is positive",
        "",
        all(
            (fam_res[(f"{x}{v}{s}", BLOCK)]["theta"] > 0).all()
            for x in ("Mid", "Ba")
            for v in ("Short", "Twin")
            for s in ("Mean", "Sr")
        ),
    )


def ptab(p: float) -> str:
    """A p-value in a table cell: three decimals, <0.001."""
    return "\\ensuremath{<}0.001" if p < 0.001 else f"{p:.3f}"


def pcell(p: float) -> str:
    """A p-value cell of the MCS table: bold in the 75 % set, italic in the 90 % set only."""
    s = ptab(p)
    if p >= 0.25:
        return f"\\textbf{{{s}}}"
    if p >= 0.10:
        return f"\\textit{{{s}}}"
    return s


def table_mcs(res: dict, keys: dict, acc: dict) -> str:
    cols = [("Bar", "Every bar"), ("All", "16:00 bar"), ("Oos", "16:00 bar")]
    yrs = f"{acc['days'][0].year}--{acc['days'][-1].year}"
    out = [
        HEAD,
        "% tab:rh_mcs: MCS p-values (T_max and T_R) of the pooled intraday forecasts and the closing-interval HAR",
        "% benchmark; circular block bootstrap, daily losses (every-bar losses averaged to the day).",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{threeparttable}",
        "\\caption{Model confidence sets of the pooled intraday forecasts}",
        # 2026-10-26 (integration, RESULTS/APPENDIX): the 1,082-day span is T_close (T_oos is Table 1's span)
        "\\label{tab:rh_mcs}",
        "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lrrrrrr@{}}",
        "\\toprule",
        f" & \\multicolumn{{2}}{{c}}{{Every bar, {yrs}}} & \\multicolumn{{2}}{{c}}{{16:00 bar, {yrs}}}"
        " & \\multicolumn{2}{c}{16:00 bar, $\\mathcal T_{\\mathrm{close}}$} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "Forecast & $T_{\\max}$ & $T_R$ & $T_{\\max}$ & $T_R$ & $T_{\\max}$ & $T_R$ \\\\",
        "\\midrule",
    ]
    allk = keys["Oos"]
    for loss, title in (("Q", "Panel A: QLIKE"), ("M", "Panel B: Squared error")):
        out.append(
            ("" if loss == "Q" else "\\addlinespace[4pt]\n")
            + f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\"
        )
        out.append("\\addlinespace[2pt]")
        for k in allk:
            lab = mname(k) + ("$^{a}$" if k == ("ols", "baseline") else "") + ("$^{b}$" if k == BENCH else "")
            cells = []
            for tag, _ in cols:
                for stat in ("max", "R"):
                    if k not in keys[tag]:
                        cells.append("--")
                        continue
                    i = keys[tag].index(k)
                    cells.append(pcell(float(res[(tag, loss, stat, BLOCK)]["p"][i])))
            out.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    out += [
        "\\bottomrule",
        "\\end{tabular*}",
        "\\begin{tablenotes}[flushleft]",
        "\\footnotesize",
        "\\item \\textit{Notes:} MCS $p$-values \\citep{HansenLundeNason2011} of the pooled intraday forecasts of "
        "Table~\\ref{tab:rh_oos} and, on $\\mathcal T_{\\mathrm{close}}$, of the closing-interval HAR benchmark. A "
        "forecast is in the 90\\% (75\\%) model confidence set when its $p$-value is at least 0.10 (0.25); "
        "\\textbf{bold}: in the 75\\% set, \\textit{italic}: in the 90\\% set only. $T_{\\max}$ and $T_R$: the two "
        "statistics of the elimination rule. Losses are daily (on every bar, the mean loss of the day's "
        "\\rhorNBars{} bars), so the bootstrap resamples trading days: the \\rhmcsDaysAll{} days from "
        "\\rhorSpanStart{} to \\rhorSpanEnd{} in the first four columns and the \\rhmcsDaysOos{} days of "
        "$\\mathcal T_{\\mathrm{close}}$ in the last two. Circular moving-block bootstrap, blocks of \\rhmcsBlock{} "
        "trading days, \\rhmcsB{} replications, seed \\rhmcsSeed; the same resamples at every elimination step. "
        "$^{a}$The pooled ridge, lasso and elastic net on the HAR benchmark reproduce the pooled OLS (their "
        "penalties are numerically zero), so they enter as one model, the OLS forecast. "
        "$^{b}$The closing-interval HAR benchmark has forecasts from \\rhmcsChStart{} only, so it is not in the "
        f"{yrs} sets.",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table}",
        "",
    ]
    return "\n".join(out)


def table_spa(fam: dict) -> str:
    est_cols = " & ".join(EST_NAME[e] for e in EST)
    out = [
        "% tab:rh_mcs_spa: SPA p-values and StepM-adjusted p-values (single-test bootstrap p in parentheses).",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{threeparttable}",
        "\\caption{Superior predictive ability and StepM tests of the full predictor set}",
        "\\label{tab:rh_mcs_spa}",
        "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lrrrrrrr@{}}",
        "\\toprule",
        " & SPA & \\multicolumn{6}{c}{StepM, by model class} \\\\",
        "\\cmidrule(lr){3-8}",
        f"Comparison & $p$ & {est_cols} \\\\",
        "\\midrule",
    ]

    def rows(fid: str, label: str) -> list[str]:
        f = fam[(fid, BLOCK)]
        st = f["stepm"]
        top = [ptab(f["spa"]["p_consistent"])] + [
            (f"\\textbf{{{ptab(float(p))}}}" if p < ALPHA else ptab(float(p))) for p in st["adj"]
        ]
        under = [""] + [f"({ptab(float(p))})" for p in st["raw"]]
        return [f"{label} & " + " & ".join(top) + " \\\\", " & " + " & ".join(under) + " \\\\"]

    def head(title: str, first: bool = False) -> list[str]:
        return ([] if first else ["\\addlinespace[4pt]"]) + [
            f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\",
            "\\addlinespace[2pt]",
        ]

    out += head("Panel A: QLIKE, full predictor set against the HAR benchmark of the same class", True)
    out += rows("BarQAf", "Every bar, 2014--2024")
    out += rows("AllQAf", "16:00 bar, 2014--2024")
    out += rows("OosQAf", "16:00 bar, $\\mathcal T_{\\mathrm{close}}$")
    out += head("Panel B: QLIKE, full predictor set against the closing-interval HAR benchmark")
    out += rows("OosQBm", "16:00 bar, $\\mathcal T_{\\mathrm{close}}$")
    out += head("Panel C: sign strategies against the unconditional short straddle")
    out += rows("MidShortMean", "Mean return, mid-quotes")
    out += rows("BaShortMean", "Mean return, bid/ask")
    out += rows("MidShortSr", "Sharpe ratio, mid-quotes")
    out += rows("BaShortSr", "Sharpe ratio, bid/ask")
    out += head("Panel D: sign strategies against the HAR-benchmark strategy of the same class")
    out += rows("MidTwinMean", "Mean return, mid-quotes")
    out += rows("BaTwinMean", "Mean return, bid/ask")
    out += rows("MidTwinSr", "Sharpe ratio, mid-quotes")
    out += rows("BaTwinSr", "Sharpe ratio, bid/ask")
    out += [
        "\\bottomrule",
        "\\end{tabular*}",
        "\\begin{tablenotes}[flushleft]",
        "\\footnotesize",
        "\\item \\textit{Notes:} Each row is one family of \\rhmcsNAlt{} one-sided hypotheses $H_k$: the full-set "
        "forecast (strategy) of class $k$ is no better than its comparison. SPA: the consistent $p$-value of the "
        "test of superior predictive ability of \\citet{Hansen2005}, $H_0$: no full-set forecast in the row beats "
        "its comparison. StepM: $p$-values adjusted for the \\rhmcsNAlt{} comparisons of the row by the studentized "
        "step-down procedure of \\citet{RomanoWolf2005}, which controls the familywise error rate; \\textbf{bold}: "
        "below 0.05. In parentheses: the one-sided bootstrap $p$-value of the comparison alone, without the "
        "adjustment. Panels A and B: daily QLIKE loss differentials (every bar: the mean over the day's "
        "\\rhorNBars{} bars), circular moving-block bootstrap with blocks of \\rhmcsBlock{} trading days, "
        "\\rhmcsB{} replications, seed \\rhmcsSeed. Panels C and D: daily returns of the sign strategy "
        "(Section~\\ref{sec:rh_econ_strategy}) on the \\rhmcsNPw{} post-warmup days, the mean daily return "
        "differential and the difference of the Sharpe ratios of Section~\\ref{sec:rh_econ_measures}, with the "
        "paired circular block bootstrap of Appendix~\\ref{sec:rh_methods_tests} (blocks of \\rhmcsBlock{} days, "
        "\\rhmcsB{} replications, seed \\rhmcsSeedEcon), each statistic studentized by its bootstrap standard "
        "deviation. The single-test $p$-values in parentheses are those of this bootstrap, one-sided; they are not "
        "the two-sided Diebold--Mariano and fixed-$b$ $p$-values of Sections~\\ref{sec:rh_results} "
        "and~\\ref{sec:rh_econ}.",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table}",
        "",
    ]
    return "\n".join(out)


if __name__ == "__main__":
    main()
