"""Appendix robustness table: the final pooled models on a rolling window of 3,500 trading days against 3,000.

The final design (writeup/final_run.py) fits every pooled intraday model on a rolling window of W = 3,000 trading days
(the main window, Table 1 and Sections 6-7) and, as a robustness run, on W = 3,500 with everything else unchanged
(first evaluation day 2 January 2014, the same frozen penalties and tree configurations, refit at every bar).  This
script compares the two windows arm by arm (model x predictor set) and writes one appendix table and its macros.

RUNS     read directly (the paths of final_run.py's defaults, one per window W):
           linear  results/window_sweep_2014start_2026-10-22/w<W>/forecasts.parquet   OLS, ridge, lasso, elastic net
           trees   results/pooled_trees_final_2014start_w<W>/forecasts.parquet         LightGBM (lgbm), XGBoost (xgb)
           forest  results/pooled_rf_skx_2014start_w<W>/forecasts.parquet              random forest (final_run.RF_KEY)
         A run without forecasts.parquet is PENDING at that window (final_run's PENDING convention): its cells and
         macros print \\rhPending{} (final_run.PENDING_TEX), each table row whose cells are all pending ends with
         final_run.ROW_MARK, every comparison that needs it (3,500 vs 3,000; full vs HAR at that window) is pending,
         and the counts run over the finished arms only (with ...All twins).  Rerunning the script after the run lands
         fills the cells in; nothing else changes.  The pooled MinGRU is parked (NN content) and is not read.
SCORING  as writeup/make_rehaul_oos_rerun.py scores Table 1 (its rules copied below with a source note, so a concurrent
         edit of that generator cannot break this one): every bar of the 2,579 evaluation days 2014-01-02 ..
         2024-04-30 (a positive closing-half-hour print, NYSE early closes dropped), the bars with a positive RV and a
         finite positive forecast of every linear arm of the window; asserted: every tree / forest arm forecasts each
         of those bars (so the scored set is Table 1's) and the two windows score the same bars.  QLIKE = mean of
         r - ln r - 1, r = RV / sigma2, sigma2 = the files' plain (pred_adj^2 B, asserted), RV the core_stats print
         (asserted equal to the files' rv_actual).  16:00: the same rule on the 16:00 bar (the 15:30 forecast of the
         closing half hour), one per evaluation day.
TESTS    Diebold-Mariano, src.evaluation.diebold_mariano.dm_test (HLN correction, Newey-West lag max(h - 1, automatic)
         = 8 on 2,579 days), on the daily average loss for every bar and on the daily 16:00 loss: the 3,500-day arm
         against the 3,000-day arm of the same model and set (a = 3,500, b = 3,000: positive = the longer window has
         the higher loss), and the full set against the HAR benchmark of the same model at each window (negative =
         the full set is more accurate), as Table 1.
ECONOMY  the sign strategy of Section 7, as make_rehaul_economics.add_pooled_trees / add_pooled_linear build the pooled
         rows (copied, source noted): on the 866 option days of the closing-interval OLS row (master_table_daily,
         sub_ols_baseline, joined with the GUI's quote book), q = +1 if the arm's 16:00 plain exceeds that row's
         iv_var, else -1; mid return q R; bid/ask return master_table_gui.rows._crossed_return.  The Sharpe ratio is
         the paper's D6 (hac_inference.d6_sharpe) on the 614 post-warmup days t > 252, and every Sharpe-ratio
         difference (full - HAR at a window; 3,500 - 3,000 of an arm) is tested by hac_inference.d6_diff_test (paired
         circular block bootstrap, block 10, 4,999 resamples, seed D6_SEED), the p the paper prints (stars, [p]).
         The pooled OLS is Table 1's reference row, not one of the paper's 12 pooled forecasts: accuracy macros only.
SELF-CHECK  every W = 3,000 (and W = 3,500) number above that a run's own score files carry is compared with them and
         must agree (results/pooled_trees_final_2014start_w<W>/scores/{metrics,dm,cross_window_dm,econ}.csv,
         results/pooled_rf_skx_2014start_w<W>/scores/, the linear runs' scores.csv / scores_all_bars.csv on their own
         2020-2024 days, results/window_sweep_2014start_2026-10-22/part2/econ.csv); the comparison is written to
         selfcheck.csv and a mismatch stops the run.

WRITES   writeup/generated/rehaul_robust_w3500_table.tex     the appendix table (tab:rh_robust_w3500)
         writeup/generated/rehaul_robust_w3500_numbers.tex   \\rhrb... macros (the table \\inputs it when undefined)
         results/rehaul_robust_window_final/                 metrics.csv, dm_window.csv, dm_full_vs_har.csv,
                                                             sharpe.csv, selfcheck.csv
         (final_run.GEN and final_run.res_dir: REHAUL_GEN_DIR / REHAUL_RES_DIR redirect a test run.)

Run:  python writeup/make_rehaul_robust_window.py     (from the repository root; independent of REHAUL_FINAL_RUN)
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WRITEUP = ROOT / "writeup"
for _p in (ROOT, ROOT / "notebooks", WRITEUP):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


def _import(name: str):
    """Import a module another session may be editing; retry a transient failure (a half-written file)."""
    for k in range(5):
        try:
            return importlib.import_module(name)
        except Exception:  # noqa: BLE001 -- retried, then raised
            if k == 4:
                raise
            time.sleep(3.0)
    raise AssertionError("unreachable")


fr = _import(
    "final_run"
)  # the final design's constants and output switches (read only)
import atm_straddle_lib as asl  # noqa: E402  (EARLY_CLOSE_DATES, the day rule of Table 1)
import hac_inference as hi  # noqa: E402  (d6_sharpe, d6_diff_test: the paper's Sharpe ratio and its test)

from src.evaluation.diebold_mariano import dm_test  # noqa: E402

RESULTS = ROOT / "results"
WINDOWS = (3000, 3500)  # main, robustness
W_MAIN, W_LONG = WINDOWS
WTAG = {W_MAIN: "Main", W_LONG: "Long"}
RF_KEY = getattr(fr, "RF_KEY", "rf_skx")
PENDING_TEX = getattr(fr, "PENDING_TEX", r"\rhPending{}")
PENDING_DEF = getattr(
    fr, "PENDING_DEF", r"\providecommand{\rhPending}{\textit{pending}}"
)
ROW_MARK = getattr(fr, "ROW_MARK", "% rhPending row")


def rel(p: Path) -> str:
    """final_run.rel with forward slashes (log lines, comments, CSV cells)."""
    return Path(fr.rel(p)).as_posix()


def linear_run(W: int) -> Path:
    return RESULTS / "window_sweep_2014start_2026-10-22" / f"w{W}"


def trees_run(W: int) -> Path:
    return RESULTS / f"pooled_trees_final_2014start_w{W}"


def rf_run(W: int) -> Path:
    return RESULTS / f"pooled_rf_skx_2014start_w{W}"


# final_run's own paths for its window are these (unless an environment variable points a loader elsewhere)
for _env, _got, _want in (
    ("REHAUL_LINEAR_RUN", fr.LINEAR_RUN, linear_run(fr.WINDOW_DAYS)),
    ("REHAUL_TREES_RUN", fr.TREES_RUN, trees_run(fr.WINDOW_DAYS)),
    ("REHAUL_RF_RUN", fr.RF_RUN, rf_run(fr.WINDOW_DAYS)),
):
    if not os.environ.get(_env, "").strip():
        assert Path(_got).resolve() == _want.resolve(), (_env, _got, _want)

CORE = ROOT / "data" / "core_stats.parquet"
MT_DAILY = RESULTS / "close_master_table" / "master_table_daily.parquet"
QUOTE_BOOK = (
    RESULTS / "atm_straddle_0dte_1530" / "daily_a0.parquet"
)  # master_table_gui.rows._QUOTE_BOOK
PART2_ECON = RESULTS / "window_sweep_2014start_2026-10-22" / "part2" / "econ.csv"
OLS_CLOSE = "sub_ols_baseline"  # make_rehaul_economics.OLS: the option days, iv_var and the realized variance
FIRST, LAST, N_DAYS = fr.FIRST_DAY, fr.LAST_DAY, 2579
CK = (
    pd.Timestamp("2020-01-03"),
    pd.Timestamp("2024-04-30"),
    1082,
)  # the linear runs' own scores.csv days
N_OPT, WARMUP, N_PW = (
    866,
    252,
    614,
)  # make_rehaul_economics N_OPT, WARMUP (gui._SIZING_WARMUP), N_PW
EARLY = pd.DatetimeIndex(pd.to_datetime(list(asl.EARLY_CLOSE_DATES)))
LEVELS = (0.10, 0.05, 0.01)
SIG = 0.05  # the level of the counts

LIN_EST = ("ridge", "lasso", "enet")
TREE_EST = ("lgbm", "xgb", "rf")
MODELS = (
    LIN_EST + TREE_EST
)  # the six pooled classes the paper counts (final_run.POOL_LINEAR + POOL_TREES)
SETS = ("baseline", "all_features")
ARMS = [("ols", "baseline")] + [(m, s) for m in MODELS for s in SETS]
NAME = {
    "ols": "OLS",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "Elastic net",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "Random forest",
}
MTAG = {
    "ols": "Ols",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "Enet",
    "lgbm": "Lgbm",
    "xgb": "Xgb",
    "rf": "Rf",
}
STAG = {"baseline": "Har", "all_features": "Full"}
SET_WORD = {"baseline": "HAR benchmark", "all_features": "full predictor set"}
BARS = (("bar", "Bar", "every bar"), ("close", "Close", "16:00 bar"))
SIDES = (("mid", "Mid", "mid-quotes"), ("x", "Ba", "bid/ask prices"))
HEAD = "% AUTO-GENERATED by writeup/make_rehaul_robust_window.py -- do not edit; re-run the script."


# --------------------------------------------------------------------------- data (rules of make_rehaul_oos_rerun.py)


def core() -> pd.Series:
    """make_rehaul_oos_rerun.core: the bar's realized variance (sumret2) by naive-ET bar end."""
    c = pd.read_parquet(CORE, columns=["endbartime", "sumret2"])
    c["endbartime"] = pd.to_datetime(c["endbartime"])
    assert c["endbartime"].dt.tz is None, "core_stats stamps are naive ET"
    assert not c["endbartime"].duplicated().any()
    return c.set_index("endbartime")["sumret2"].sort_index()


def evaluation_days(rv: pd.Series) -> pd.DatetimeIndex:
    """make_rehaul_oos_rerun.evaluation_days: a positive closing-half-hour print, early closes dropped."""
    s = rv[(rv.index.hour == 16) & (rv.index.minute == 0)]
    d = pd.DatetimeIndex(s.index.normalize())
    keep = (s.to_numpy() > 0) & np.isfinite(s.to_numpy()) & ~d.isin(EARLY)
    days = d[keep]
    days = days[(days >= FIRST) & (days <= LAST)]
    assert len(days) == N_DAYS and days[0] == FIRST and days[-1] == LAST, (
        len(days),
        days[0],
        days[-1],
    )
    return days


def scored_set(fc: dict, rv: pd.Series, days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """make_rehaul_oos_rerun.scored_set: on an evaluation day, a positive RV and a finite positive forecast of every
    arm given; each file's truth is the core_stats print."""
    idx = None
    for f in fc.values():
        ok = f.index[np.isfinite(f["plain"].to_numpy()) & (f["plain"].to_numpy() > 0)]
        idx = ok if idx is None else idx.intersection(ok)
    assert idx is not None
    idx = idx[pd.DatetimeIndex(idx.normalize()).isin(days)]
    t = rv.reindex(idx)
    idx = idx[(t.to_numpy() > 0) & np.isfinite(t.to_numpy())]
    for k, f in fc.items():
        assert np.array_equal(
            f["rv"].reindex(idx).to_numpy(), rv.reindex(idx).to_numpy()
        ), k
    return idx.sort_values()


def qlike(f: pd.DataFrame, idx: pd.DatetimeIndex, rv: pd.Series) -> np.ndarray:
    """make_rehaul_oos_rerun.loss (its QLIKE half): r - ln r - 1, r = RV / plain, on the bars idx."""
    p, y = f["plain"].reindex(idx).to_numpy(float), rv.reindex(idx).to_numpy(float)
    r = y / p
    return r - np.log(r) - 1.0


def daily(x: np.ndarray, idx: pd.DatetimeIndex) -> np.ndarray:
    """make_rehaul_oos_rerun.daily: the daily average of a per-bar loss."""
    return (
        pd.Series(x, index=pd.DatetimeIndex(idx.normalize()))
        .groupby(level=0)
        .mean()
        .to_numpy()
    )


def _frame(
    path: Path, keys: set[tuple[str, str]], n_rows: int, rename: dict[str, str]
) -> dict:
    """(model, set) -> frame by bar end with plain and rv, from one forecasts.parquet; gates of pooled_*_13._frame:
    the arms, rows an arm, plain = pred_adj^2 B, 13 bars a day ending at 16:00."""
    f = pd.read_parquet(
        path,
        columns=[
            "model",
            "predictor_set",
            "date",
            "pred_adj",
            "B",
            "plain",
            "rv_actual",
        ],
    )
    f["model"] = f["model"].replace(rename)
    n = f.groupby(["model", "predictor_set"]).size()
    assert set(n.index) == keys, (path, sorted(n.index))
    assert (n == n_rows).all(), (path, n.to_dict())
    v = f[["pred_adj", "B", "plain", "rv_actual"]].to_numpy(float)
    assert np.isfinite(v).all() and (f["plain"] > 0).all() and (f["B"] > 0).all(), path
    rel = float(np.max(np.abs(f["plain"] / (f["pred_adj"] ** 2 * f["B"]) - 1.0)))
    assert rel <= 1e-12, (path, rel)
    t = pd.to_datetime(f["date"])
    clock = t.dt.strftime("%H:%M")
    assert (
        clock.nunique() == 13 and clock.min() == "10:00" and clock.max() == "16:00"
    ), path
    assert t.dt.normalize().max() == LAST, (path, t.max())
    out = {}
    for (m, s), g in f.groupby(["model", "predictor_set"]):
        x = (
            g.set_index(pd.to_datetime(g["date"]))
            .rename(columns={"rv_actual": "rv"})[["plain", "rv"]]
            .sort_index()
        )
        assert not x.index.duplicated().any(), (path, m, s)
        out[(m, s)] = x
    return out


def load_window(W: int) -> tuple[dict, list[str]]:
    """Every finished arm of window W: (model, set) -> frame; and the pending estimators (no forecasts yet)."""
    lr = linear_run(W)
    ri = json.loads((lr / "run_info.json").read_text())
    fr.guard_standin(ri, lr)
    assert int(ri["window_days"]) == W and pd.Timestamp(ri["first_day"]) == FIRST, (
        lr,
        ri["window_days"],
    )
    fc = _frame(
        lr / "forecasts.parquet",
        {("ols", "baseline")} | {(m, s) for m in LIN_EST for s in SETS},
        fr.N_LINEAR_ROWS,
        {},
    )
    assert min(x.index.min() for x in fc.values()) == fr.LINEAR_WALK_START, lr
    pending = []
    for est_keys, run, rename in (
        (("lgbm", "xgb"), trees_run(W), {}),
        (("rf",), rf_run(W), {RF_KEY: "rf"}),
    ):
        if not (run / "forecasts.parquet").is_file():
            pending += list(est_keys)
            continue
        meta = json.loads((run / "forecasts.json").read_text())
        fr.guard_standin(meta, run)
        assert (
            meta["refit"],
            int(meta["window_days"]),
            pd.Timestamp(meta["span"][0]),
        ) == ("bar", W, FIRST), (
            run,
            meta["refit"],
            meta["window_days"],
            meta["span"],
        )
        ft = _frame(
            run / "forecasts.parquet",
            {(e, s) for e in est_keys for s in SETS},
            fr.N_BARS,
            rename,
        )
        for k, x in ft.items():
            assert x.index.min().normalize() == FIRST, (run, k, x.index.min())
        fc.update(ft)
    return fc, pending


def penalties(W: int) -> dict:
    p = json.loads((linear_run(W) / "frozen_penalties.json").read_text())
    return {m: p[m] for m in ("ols",) + LIN_EST}


# --------------------------------------------------------------------------- the trade (make_rehaul_economics rules)


def crossed_return(q, ex, bid, ask) -> np.ndarray:
    """writeup/master_table_gui/rows.py _crossed_return: a buy pays the ask, a sale gets the bid (entry only)."""
    return np.where(
        q > 0, q * (ex / ask - 1.0), np.where(q < 0, q * (ex / bid - 1.0), 0.0)
    )


def trade_frame() -> pd.DataFrame:
    """The 866 option days of the closing-interval OLS row (make_rehaul_economics.load, its OLS row and book): iv_var,
    rv_unclipped, and the quote book's straddle (R, exit, bid, ask); gate: the row's mid return is q R."""
    d = pd.read_parquet(
        MT_DAILY, columns=["key", "day", "q", "ret_mid", "iv_var", "rv_unclipped"]
    )
    d = d[d["key"] == OLS_CLOSE].copy()
    d["day"] = pd.to_datetime(d["day"]).dt.normalize()
    qb = pd.read_parquet(
        QUOTE_BOOK, columns=["R", "exit", "bid_c", "bid_p", "ask_c", "ask_p"]
    )
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
    assert np.allclose(g["q"] * g["R"], g["ret_mid"], rtol=0, atol=1e-10), (
        "quote-book R"
    )
    assert np.isfinite(g[["iv_var", "rv_unclipped"]].to_numpy()).all()
    return g


def at16(f: pd.DataFrame) -> pd.DataFrame:
    """pooled_trees_13.at16 / pooled_linear_13.at16: the 16:00 rows of an arm, by trading day."""
    g = f[(f.index.hour == 16) & (f.index.minute == 0)]
    out = g.set_index(pd.DatetimeIndex(g.index.normalize()).astype("datetime64[ns]"))
    assert not out.index.duplicated().any()
    return out


def trade(f: pd.DataFrame, tf: pd.DataFrame) -> dict[str, np.ndarray]:
    """make_rehaul_economics.add_pooled_trees: q = sign(plain - iv_var), ties short; mid and bid/ask returns. Gates of
    tree_16: a finite positive forecast on every option day, and the arm's 16:00 RV is the OLS row's rv_unclipped."""
    a = at16(f).reindex(tf.index)
    p = a["plain"].to_numpy(float)
    assert np.isfinite(p).all() and (p > 0).all(), int((~np.isfinite(p)).sum())
    assert np.allclose(
        a["rv"].to_numpy(float),
        tf["rv_unclipped"].to_numpy(float),
        rtol=1e-12,
        atol=0.0,
    )
    q = np.where(p > tf["iv_var"].to_numpy(float), 1.0, -1.0)
    return {
        "mid": q * tf["R"].to_numpy(float),
        "x": crossed_return(
            q, tf["ex"].to_numpy(), tf["bid"].to_numpy(), tf["ask"].to_numpy()
        ),
    }


# --------------------------------------------------------------------------- formatting


def _fmt(x: float, dp: int) -> str:
    v = round(float(x), dp)
    if v == 0:
        v = 0.0  # no negative zero
    return f"{'-' if v < 0 else ''}{abs(v):.{dp}f}"


def n_stars(p: float) -> int:
    return int(sum(p < a for a in LEVELS)) if np.isfinite(p) else 0


def sup(k: int) -> str:
    """Three star slots, the unused ones as a phantom (make_rehaul_economics.cell), so r columns line up."""
    pad = "*" * (3 - k)
    return "*" * k + (f"\\phantom{{{pad}}}" if pad else "")


def tcell(x: float | None, dp: int, p: float | None = None, paren: bool = False) -> str:
    """A table cell in math mode: a level (no stars), or a statistic / difference with the stars of p."""
    if x is None:
        return PENDING_TEX
    body = _fmt(x, dp)
    if paren:
        body = f"({body})"
    return f"${body}^{{{sup(n_stars(p) if p is not None else 0)}}}$"


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def mnum(x: float | None, dp: int) -> str:
    """A macro value: text-safe minus, no plus sign (make_rehaul_results.mnum); pending when x is None."""
    if x is None:
        return PENDING_TEX
    v = round(float(x), dp)
    if v == 0:
        v = 0.0
    return ("\\ensuremath{-}" if v < 0 else "") + f"{abs(v):.{dp}f}"


def mpval(p: float | None) -> str:
    """make_rehaul_results.mpval: the whole p statement ('$p = 0.18$' / '$p < 0.001$'); pending when p is None."""
    if p is None:
        return PENDING_TEX
    if p < 0.001:
        return "\\ensuremath{p < 0.001}"
    s = f"{p:.2f}" if p >= 0.10 else f"{p:.3f}"
    if s in ("0.10", "0.100", "0.050", "0.010"):
        s = f"{p:.4f}"
    return f"\\ensuremath{{p = {s}}}"


# --------------------------------------------------------------------------- main


def main() -> None:
    t0 = time.time()
    print(fr.tag())
    gen = fr.GEN
    res = fr.res_dir(RESULTS / "rehaul_robust_window_final")
    gen.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)

    # one design apart from the window: the same frozen penalties and tree configurations
    assert penalties(W_MAIN) == penalties(W_LONG), (
        "the two windows' linear penalties differ"
    )
    cfg = {
        W: json.loads((trees_run(W) / "run.json").read_text()).get("configs")
        for W in WINDOWS
    }
    assert cfg[W_MAIN] == cfg[W_LONG], "the two windows' tree configurations differ"

    rv = core()
    days = evaluation_days(rv)
    tf = trade_frame()
    pw = pd.DatetimeIndex(tf.index[WARMUP:])
    assert len(pw) == N_PW
    print(
        f"{len(days):,} evaluation days {days[0].date()} .. {days[-1].date()}; {len(tf)} option days, {len(pw)} "
        f"post-warmup ({time.time() - t0:.0f} s)"
    )

    FC: dict[int, dict] = {}
    PEND: dict[int, list[str]] = {}
    IDX: dict[int, pd.DatetimeIndex] = {}
    for W in WINDOWS:
        fc, pend = load_window(W)
        FC[W], PEND[W] = fc, pend
        lin = {k: v for k, v in fc.items() if k[0] in ("ols",) + LIN_EST}
        idx = scored_set(lin, rv, days)
        idx_all = scored_set(
            fc, rv, days
        )  # also asserts each tree's / the forest's RV is the core_stats print
        miss = idx.difference(idx_all)
        assert idx_all.equals(idx), (
            f"w{W}: an arm misses {len(miss)} bars the linear rows score, e.g. {list(miss[:5])}"
        )
        IDX[W] = idx
        print(
            f"W={W:,}: {len(fc)} arms, pending {pend or 'none'}; {len(idx):,} bars scored ({time.time() - t0:.0f} s)"
        )
    assert IDX[W_MAIN].equals(IDX[W_LONG]), "the two windows score different bars"
    idx = IDX[W_MAIN]
    on16 = idx[(idx.hour == 16) & (idx.minute == 0)]
    assert len(on16) == N_DAYS, len(on16)

    def have(W: int, k: tuple[str, str]) -> bool:
        return k in FC[W]

    # ------------------------------------------------------------------ losses, Sharpe ratios
    L: dict = {}  # (W, m, s) -> {"bar": per-bar QLIKE, "close": 16:00 QLIKE}
    R: dict = {}  # (W, m, s) -> {"mid": 614 returns, "x": 614 returns}
    for W in WINDOWS:
        for k, f in FC[W].items():
            L[(W, *k)] = {"bar": qlike(f, idx, rv), "close": qlike(f, on16, rv)}
            if k[0] != "ols":
                tr = trade(f, tf)
                R[(W, *k)] = {side: tr[side][WARMUP:] for side in ("mid", "x")}

    def q_mean(W, m, s, bars) -> float | None:
        x = L.get((W, m, s))
        return None if x is None else float(x[bars].mean())

    sr_cache: dict = {}

    def sr(W, m, s, side) -> float | None:
        x = R.get((W, m, s))
        if x is None:
            return None
        key = (W, m, s, side)
        if key not in sr_cache:
            sr_cache[key] = float(hi.d6_sharpe(x[side], dates=pw)["sharpe"])
        return sr_cache[key]

    def dm(la: np.ndarray, lb: np.ndarray, bars: str) -> dict:
        r = (
            dm_test(daily(la, idx), daily(lb, idx))
            if bars == "bar"
            else dm_test(la, lb)
        )
        assert r["T"] == N_DAYS, r
        return r

    def boot(a: np.ndarray, b: np.ndarray) -> dict:
        r = hi.d6_diff_test(a, b, pw)
        return r

    # rows of the CSVs and the dictionaries the table / macros read
    met_rows, dmw_rows, dmf_rows, sh_rows = [], [], [], []
    for W in WINDOWS:
        for m, s in ARMS:
            if not have(W, (m, s)):
                continue
            met_rows.append(
                {
                    "W": W,
                    "model": m,
                    "set": s,
                    "n_bars": len(idx),
                    "n_days": N_DAYS,
                    "qlike": q_mean(W, m, s, "bar"),
                    "qlike_1600": q_mean(W, m, s, "close"),
                    "sr_mid": sr(W, m, s, "mid") if m != "ols" else np.nan,
                    "sr_ba": sr(W, m, s, "x") if m != "ols" else np.nan,
                }
            )
    DMW: dict = {}  # (m, s, bars) -> dm result (a = 3,500, b = 3,000)
    for m, s in ARMS:
        if not (have(W_MAIN, (m, s)) and have(W_LONG, (m, s))):
            continue
        for bars, _, _ in BARS:
            r = dm(L[(W_LONG, m, s)][bars], L[(W_MAIN, m, s)][bars], bars)
            DMW[(m, s, bars)] = r
            qa, qb = q_mean(W_LONG, m, s, bars), q_mean(W_MAIN, m, s, bars)
            assert qa is not None and qb is not None
            dmw_rows.append(
                {
                    "model": m,
                    "set": s,
                    "bars": bars,
                    "W_a": W_LONG,
                    "W_b": W_MAIN,
                    "qlike_a": qa,
                    "qlike_b": qb,
                    "pct": 100.0 * (qa / qb - 1.0),
                    "dm": r["dm"],
                    "p": r["p"],
                    "T": r["T"],
                    "hac_lag": r["hac_lag"],
                }
            )
    DMF: dict = {}  # (W, m, bars) -> dm result (a = full, b = HAR)
    for W in WINDOWS:
        for m in MODELS:
            if not (have(W, (m, "all_features")) and have(W, (m, "baseline"))):
                continue
            for bars, _, _ in BARS:
                r = dm(
                    L[(W, m, "all_features")][bars], L[(W, m, "baseline")][bars], bars
                )
                DMF[(W, m, bars)] = r
                qa, qb = (
                    q_mean(W, m, "all_features", bars),
                    q_mean(W, m, "baseline", bars),
                )
                assert qa is not None and qb is not None
                dmf_rows.append(
                    {
                        "W": W,
                        "model": m,
                        "bars": bars,
                        "qlike_full": qa,
                        "qlike_har": qb,
                        "pct": 100.0 * (qa / qb - 1.0),
                        "dm": r["dm"],
                        "p": r["p"],
                        "T": r["T"],
                        "hac_lag": r["hac_lag"],
                    }
                )
    SRF: dict = {}  # (W, m, side) -> bootstrap result (full - HAR)
    SRW: dict = {}  # (m, s, side) -> bootstrap result (3,500 - 3,000)
    for side, _, _ in SIDES:
        for W in WINDOWS:
            for m in MODELS:
                if (W, m, "all_features") in R and (W, m, "baseline") in R:
                    r = boot(
                        R[(W, m, "all_features")][side], R[(W, m, "baseline")][side]
                    )
                    a, b = sr(W, m, "all_features", side), sr(W, m, "baseline", side)
                    assert (
                        a is not None
                        and b is not None
                        and abs(r["delta"] - (a - b)) < 1e-10
                    )
                    SRF[(W, m, side)] = r
                    sh_rows.append(
                        {
                            "test": "full - HAR",
                            "W": W,
                            "model": m,
                            "set": "",
                            "side": side,
                            "sr_a": a,
                            "sr_b": b,
                            "delta": r["delta"],
                            "p": r["p"],
                            "T": r["T"],
                        }
                    )
        for m, s in ARMS:
            if m == "ols" or not ((W_MAIN, m, s) in R and (W_LONG, m, s) in R):
                continue
            r = boot(R[(W_LONG, m, s)][side], R[(W_MAIN, m, s)][side])
            a, b = sr(W_LONG, m, s, side), sr(W_MAIN, m, s, side)
            assert a is not None and b is not None and abs(r["delta"] - (a - b)) < 1e-10
            SRW[(m, s, side)] = r
            sh_rows.append(
                {
                    "test": "3500 - 3000",
                    "W": np.nan,
                    "model": m,
                    "set": s,
                    "side": side,
                    "sr_a": a,
                    "sr_b": b,
                    "delta": r["delta"],
                    "p": r["p"],
                    "T": r["T"],
                }
            )
    lags = {r["hac_lag"] for r in list(DMW.values()) + list(DMF.values())}
    assert len(lags) == 1, lags
    dm_lag = lags.pop()
    print(f"tests done ({time.time() - t0:.0f} s)")

    # ------------------------------------------------------------------ self-check against the runs' score files
    chk: list[dict] = []

    def cmp(
        src: str,
        W,
        m,
        s,
        what: str,
        mine,
        theirs,
        rtol: float = 1e-9,
        atol: float = 1e-12,
    ) -> None:
        ok = bool(np.isclose(float(mine), float(theirs), rtol=rtol, atol=atol))
        chk.append(
            {
                "source": src,
                "W": W,
                "model": m,
                "set": s,
                "quantity": what,
                "mine": float(mine),
                "theirs": float(theirs),
                "abs_diff": abs(float(mine) - float(theirs)),
                "ok": ok,
            }
        )

    def mkey(x: str) -> str:
        return "rf" if x == RF_KEY else x

    for W in WINDOWS:
        for run in (trees_run(W), rf_run(W)):
            sd = run / "scores"
            if not (sd / "metrics.csv").is_file():
                continue
            src = rel(sd)
            M = pd.read_csv(sd / "metrics.csv")
            for _, r in M.iterrows():
                m, s = mkey(r["model"]), r["set"]
                if not have(W, (m, s)):
                    continue
                cmp(
                    src + "/metrics.csv", W, m, s, "n_bars", len(idx), r["n_bars"], 0, 0
                )
                cmp(
                    src + "/metrics.csv",
                    W,
                    m,
                    s,
                    "n_days16",
                    len(on16),
                    r["n_days16"],
                    0,
                    0,
                )
                cmp(
                    src + "/metrics.csv",
                    W,
                    m,
                    s,
                    "qlike every bar",
                    q_mean(W, m, s, "bar"),
                    r["qlike"],
                )
                cmp(
                    src + "/metrics.csv",
                    W,
                    m,
                    s,
                    "qlike 16:00",
                    q_mean(W, m, s, "close"),
                    r["qlike_1600"],
                )
            D = pd.read_csv(sd / "dm.csv")
            D = D[
                (D["comparison"] == "all features vs HAR benchmark")
                & (D["loss"] == "qlike")
            ]
            for _, r in D.iterrows():
                m = mkey(r["model"])
                bars = "bar" if r["bars"] == "every bar" else "close"
                if (W, m, bars) not in DMF:
                    continue
                cmp(
                    src + "/dm.csv",
                    W,
                    m,
                    "full vs HAR",
                    f"DM {r['bars']}",
                    DMF[(W, m, bars)]["dm"],
                    r["dm"],
                )
                cmp(
                    src + "/dm.csv",
                    W,
                    m,
                    "full vs HAR",
                    f"DM p {r['bars']}",
                    DMF[(W, m, bars)]["p"],
                    r["p"],
                )
            if (sd / "cross_window_dm.csv").is_file():
                X = pd.read_csv(sd / "cross_window_dm.csv")
                for _, r in X.iterrows():
                    m, s = mkey(r["model"]), r["set"]
                    bars = "bar" if r["bars"] == "every bar" else "close"
                    if (m, s, bars) not in DMW or (int(r["W_a"]), int(r["W_b"])) != (
                        W_LONG,
                        W_MAIN,
                    ):
                        continue
                    cmp(
                        src + "/cross_window_dm.csv",
                        "3500 vs 3000",
                        m,
                        s,
                        f"DM {r['bars']}",
                        DMW[(m, s, bars)]["dm"],
                        r["dm"],
                    )
                    cmp(
                        src + "/cross_window_dm.csv",
                        "3500 vs 3000",
                        m,
                        s,
                        f"DM p {r['bars']}",
                        DMW[(m, s, bars)]["p"],
                        r["p"],
                    )
            E = pd.read_csv(sd / "econ.csv")
            for _, r in E.iterrows():
                econ_check(cmp, src + "/econ.csv", W, mkey(r["model"]), r, sr, SRF)
    if PART2_ECON.is_file():
        E = pd.read_csv(PART2_ECON)
        for _, r in E[E["W"].isin(WINDOWS)].iterrows():
            econ_check(cmp, rel(PART2_ECON), int(r["W"]), r["model"], r, sr, SRF)
    for W in WINDOWS:  # the linear runs' own 2020-2024 scores (their scores.csv and scores_all_bars.csv)
        lr = linear_run(W)
        ck16 = on16[(on16.normalize() >= CK[0]) & (on16.normalize() <= CK[1])]
        ckb = idx[(idx.normalize() >= CK[0]) & (idx.normalize() <= CK[1])]
        assert len(ck16) == CK[2]
        sc = pd.read_csv(lr / "scores.csv")
        for _, r in sc.iterrows():
            k = (r["model"], r["predictor_set"])
            cmp(
                rel(lr / "scores.csv"),
                W,
                *k,
                "qlike 16:00, 2020-2024",
                qlike(FC[W][k], ck16, rv).mean(),
                r["qlike"],
                rtol=1e-10,
                atol=0,
            )
        sab = pd.read_csv(lr / "scores_all_bars.csv")
        for _, r in sab.iterrows():
            k = (r["model"], r["predictor_set"])
            cmp(
                rel(lr / "scores_all_bars.csv"),
                W,
                *k,
                "n bars, 2020-2024",
                len(ckb),
                r["n_bars"],
                0,
                0,
            )
            cmp(
                rel(lr / "scores_all_bars.csv"),
                W,
                *k,
                "qlike every bar, 2020-2024",
                qlike(FC[W][k], ckb, rv).mean(),
                r["qlike"],
                rtol=1e-10,
                atol=0,
            )
    CHK = pd.DataFrame(chk)
    CHK.to_csv(res / "selfcheck.csv", index=False)
    bad = CHK[~CHK["ok"]]
    print(
        f"self-check: {len(CHK)} comparisons, {int(CHK['ok'].sum())} agree; max |diff| "
        f"{CHK['abs_diff'].max():.3g}"
    )
    if len(bad):
        print(bad.to_string(index=False))
        raise SystemExit(
            "self-check failed: the robustness numbers do not reproduce the runs' score files"
        )

    # ------------------------------------------------------------------ CSVs
    pd.DataFrame(met_rows).to_csv(res / "metrics.csv", index=False)
    pd.DataFrame(dmw_rows).to_csv(res / "dm_window.csv", index=False)
    pd.DataFrame(dmf_rows).to_csv(res / "dm_full_vs_har.csv", index=False)
    pd.DataFrame(sh_rows).to_csv(res / "sharpe.csv", index=False)

    # ------------------------------------------------------------------ macros
    macros: list[tuple[str, str, str]] = []

    def mac(name: str, value: str, note: str) -> None:
        assert name.isalpha() and name.startswith("rhrb"), name
        assert name not in {x[0] for x in macros}, name
        macros.append((name, value, note))

    rf_long_pending = "rf" in PEND[W_LONG]
    mac("rhrbWinMain", thousands(W_MAIN), "main rolling window, trading days")
    mac("rhrbWinLong", thousands(W_LONG), "robustness rolling window, trading days")
    mac(
        "rhrbWinMainBars",
        thousands(13 * W_MAIN),
        "main rolling window, 13-bar observations",
    )
    mac(
        "rhrbWinLongBars",
        thousands(13 * W_LONG),
        "robustness rolling window, 13-bar observations",
    )
    mac("rhrbNdays", thousands(N_DAYS), "evaluation days (every bar and 16:00)")
    mac(
        "rhrbNbars",
        thousands(len(idx)),
        "scored bars, the same at both windows (asserted)",
    )
    mac("rhrbSpanStart", f"{FIRST.day} {FIRST:%B %Y}", "first evaluation day")
    mac("rhrbSpanEnd", f"{LAST.day} {LAST:%B %Y}", "last evaluation day")
    mac("rhrbDMLag", str(dm_lag), "Newey-West lag of every DM test")
    mac("rhrbNopt", str(N_OPT), "option days of the sign strategy")
    mac("rhrbNpw", str(N_PW), "post-warmup option days, the Sharpe-ratio sample")
    mac(
        "rhrbBootN",
        thousands(hi.LW_BOOT),
        "bootstrap resamples of the Sharpe-ratio test",
    )
    mac("rhrbBootBlock", str(hi.LW_BLOCK), "block length of the Sharpe-ratio bootstrap")
    mac(
        "rhrbPendingNote",
        f"The random forest on the \\rhrbWinLong-day window is {PENDING_TEX}."
        if rf_long_pending
        else "",
        "table-note sentence on the pending forest (empty when nothing is pending)",
    )
    for m, s in ARMS:
        for W in WINDOWS:
            for bars, btag, bword in BARS:
                mac(
                    f"rhrbQ{MTAG[m]}{STAG[s]}{btag}{WTAG[W]}",
                    mnum(q_mean(W, m, s, bars), 4),
                    f"QLIKE, {bword}, {NAME[m]}, {SET_WORD[s]}, W = {W:,}",
                )
            if m != "ols":
                for side, stag, sword in SIDES:
                    mac(
                        f"rhrbSR{MTAG[m]}{STAG[s]}{stag}{WTAG[W]}",
                        mnum(sr(W, m, s, side), 2),
                        f"Sharpe ratio (D6), sign strategy at {sword}, {NAME[m]}, {SET_WORD[s]}, W = {W:,}",
                    )
        for bars, btag, bword in BARS:
            rt = DMW.get((m, s, bars))
            qa, qb = q_mean(W_LONG, m, s, bars), q_mean(W_MAIN, m, s, bars)
            mac(
                f"rhrbPct{MTAG[m]}{STAG[s]}{btag}",
                mnum(
                    None
                    if rt is None or qa is None or qb is None
                    else 100.0 * (qa / qb - 1.0),
                    1,
                ),
                f"QLIKE change, %, W = 3,500 vs 3,000, {bword}, {NAME[m]}, {SET_WORD[s]}",
            )
            mac(
                f"rhrbDM{MTAG[m]}{STAG[s]}{btag}",
                mnum(None if rt is None else rt["dm"], 2),
                f"DM, W = 3,500 vs 3,000 (positive: 3,500 has the higher loss), {bword}, {NAME[m]}, {SET_WORD[s]}",
            )
            mac(
                f"rhrbP{MTAG[m]}{STAG[s]}{btag}",
                mpval(None if rt is None else rt["p"]),
                "... its p statement",
            )
        if m != "ols":
            for side, stag, sword in SIDES:
                rt = SRW.get((m, s, side))
                mac(
                    f"rhrbDSR{MTAG[m]}{STAG[s]}{stag}",
                    mnum(None if rt is None else rt["delta"], 2),
                    f"Sharpe ratio change, W = 3,500 minus 3,000, {sword}, {NAME[m]}, {SET_WORD[s]}",
                )
                mac(
                    f"rhrbPSR{MTAG[m]}{STAG[s]}{stag}",
                    mpval(None if rt is None else rt["p"]),
                    "... its p statement (D6 bootstrap)",
                )
    for m in MODELS:
        for W in WINDOWS:
            for bars, btag, bword in BARS:
                rt = DMF.get((W, m, bars))
                qa, qb = (
                    q_mean(W, m, "all_features", bars),
                    q_mean(W, m, "baseline", bars),
                )
                mac(
                    f"rhrbPctF{MTAG[m]}{btag}{WTAG[W]}",
                    mnum(
                        None
                        if rt is None or qa is None or qb is None
                        else 100.0 * (qa / qb - 1.0),
                        1,
                    ),
                    f"QLIKE, full set vs HAR benchmark, %, {bword}, {NAME[m]}, W = {W:,}",
                )
                mac(
                    f"rhrbDMF{MTAG[m]}{btag}{WTAG[W]}",
                    mnum(None if rt is None else rt["dm"], 2),
                    f"DM, full set vs HAR benchmark (negative: full more accurate), {bword}, {NAME[m]}, W = {W:,}",
                )
                mac(
                    f"rhrbPF{MTAG[m]}{btag}{WTAG[W]}",
                    mpval(None if rt is None else rt["p"]),
                    "... its p statement",
                )
            for side, stag, sword in SIDES:
                rt = SRF.get((W, m, side))
                mac(
                    f"rhrbDSRF{MTAG[m]}{stag}{WTAG[W]}",
                    mnum(None if rt is None else rt["delta"], 2),
                    f"Sharpe ratio, full set minus HAR benchmark, {sword}, {NAME[m]}, W = {W:,}",
                )
                mac(
                    f"rhrbPSRF{MTAG[m]}{stag}{WTAG[W]}",
                    mpval(None if rt is None else rt["p"]),
                    "... its p statement (D6 bootstrap)",
                )
    # counts over the finished arms / classes (the paper's 12 pooled forecasts; the OLS reference row left out)
    arms12 = [(m, s) for m in MODELS for s in SETS]
    both = [(m, s) for m, s in arms12 if (m, s, "bar") in DMW]
    mac(
        "rhrbNArms",
        str(len(both)),
        "pooled forecasts compared at both windows (finished; OLS reference row out)",
    )
    mac(
        "rhrbNArmsAll",
        str(len(arms12)),
        "pooled forecasts in all (finished or pending)",
    )
    mac(
        "rhrbNArmsPending",
        str(len(arms12) - len(both)),
        "pooled forecasts pending at a window",
    )
    for bars, btag, bword in BARS:
        mac(
            f"rhrbNSig{btag}",
            str(sum(DMW[(m, s, bars)]["p"] < SIG for m, s in both)),
            f"of rhrbNArms: QLIKE differs between the windows at 5 % (DM), {bword}",
        )
        mac(
            f"rhrbNLower{btag}",
            str(sum(DMW[(m, s, bars)]["dm"] < 0 for m, s in both)),
            f"of rhrbNArms: lower QLIKE at 3,500 (DM < 0), {bword}",
        )
        mac(
            f"rhrbNHigherSig{btag}",
            str(
                sum(
                    DMW[(m, s, bars)]["dm"] > 0 and DMW[(m, s, bars)]["p"] < SIG
                    for m, s in both
                )
            ),
            f"of rhrbNArms: significantly higher QLIKE at 3,500 (DM > 0, 5 %), {bword}",
        )
        mac(
            f"rhrbNLowerSig{btag}",
            str(
                sum(
                    DMW[(m, s, bars)]["dm"] < 0 and DMW[(m, s, bars)]["p"] < SIG
                    for m, s in both
                )
            ),
            f"of rhrbNArms: significantly lower QLIKE at 3,500 (DM < 0, 5 %), {bword}",
        )
        pcts = [
            abs(x["pct"]) for x in dmw_rows if x["bars"] == bars and x["model"] != "ols"
        ]
        mac(
            f"rhrbMaxAbsPct{btag}",
            mnum(max(pcts), 1),
            f"largest absolute QLIKE change between the windows, %, {bword} (finished arms)",
        )
    for W in WINDOWS:
        done = [m for m in MODELS if (W, m, "bar") in DMF]
        mac(
            f"rhrbNModels{WTAG[W]}",
            str(len(done)),
            f"model classes with both predictor sets finished, W = {W:,}",
        )
        for bars, btag, bword in BARS:
            mac(
                f"rhrbNFullLowerSig{btag}{WTAG[W]}",
                str(
                    sum(
                        DMF[(W, m, bars)]["dm"] < 0 and DMF[(W, m, bars)]["p"] < SIG
                        for m in done
                    )
                ),
                f"of rhrbNModels{WTAG[W]}: full set significantly more accurate than HAR (DM < 0, 5 %), {bword}",
            )
        for side, stag, sword in SIDES:
            mac(
                f"rhrbNDSRFPos{stag}{WTAG[W]}",
                str(sum(SRF[(W, m, side)]["delta"] > 0 for m in done)),
                f"of rhrbNModels{WTAG[W]}: full-set Sharpe ratio above HAR, {sword}",
            )
            mac(
                f"rhrbNDSRFSig{stag}{WTAG[W]}",
                str(
                    sum(
                        SRF[(W, m, side)]["delta"] > 0 and SRF[(W, m, side)]["p"] < SIG
                        for m in done
                    )
                ),
                f"of rhrbNModels{WTAG[W]}: ... and bootstrap p < 0.05, {sword}",
            )
    mac("rhrbNModelsAll", str(len(MODELS)), "model classes in all")
    for side, stag, sword in SIDES:
        n_sr = [k for k in SRW if k[2] == side]
        mac(
            f"rhrbNSRSig{stag}",
            str(sum(SRW[k]["p"] < SIG for k in n_sr)),
            f"of rhrbNArms: Sharpe ratio differs between the windows (bootstrap p < 0.05), {sword}",
        )
        mac(
            f"rhrbMaxAbsDSR{stag}",
            mnum(max(abs(SRW[k]["delta"]) for k in n_sr), 2),
            f"largest absolute Sharpe-ratio change between the windows, {sword} (finished arms)",
        )
        mac(
            f"rhrbMinPSR{stag}",
            mpval(min(SRW[k]["p"] for k in n_sr)),
            f"smallest bootstrap p of a Sharpe-ratio change between the windows, {sword}",
        )

    num_lines = [
        HEAD,
        "% \\rhrb... macros of the appendix robustness table tab:rh_robust_w3500 (rehaul_robust_w3500_table.tex): the final",
        f"% pooled models on a rolling window of {W_LONG:,} against {W_MAIN:,} trading days; every bar scored on the "
        f"{N_DAYS:,} days {FIRST.date()} .. {LAST.date()}",
        f"% ({len(idx):,} bars), Sharpe ratios (D6) on the {N_PW} post-warmup option days. Names: Q (QLIKE), SR (Sharpe "
        "ratio), Pct / DM / P",
        "% (3,500 vs 3,000: % change, DM, p statement), DSR / PSR (Sharpe change 3,500 - 3,000, bootstrap p), PctF / DMF "
        "/ PF / DSRF / PSRF",
        "% (full set vs HAR benchmark at a window); Model Ols Ridge Lasso Enet Lgbm Xgb Rf; Set Har Full; Bar (every "
        "bar) / Close (16:00);",
        "% Mid / Ba (bid/ask); Main (W = 3,000) / Long (W = 3,500). Pending: "
        + (", ".join(f"{e} at W = {W:,}" for W in WINDOWS for e in PEND[W]) or "none")
        + ".",
        PENDING_DEF,
    ]
    for name, value, note in macros:
        num_lines.append(
            f"\\providecommand{{\\{name}}}{{}}\\renewcommand{{\\{name}}}{{{value}}}  % {note}"
        )
    (gen / "rehaul_robust_w3500_numbers.tex").write_text(
        "\n".join(num_lines) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"wrote {rel(gen / 'rehaul_robust_w3500_numbers.tex')} ({len(macros)} macros)"
    )

    # ------------------------------------------------------------------ the table
    def line(label: str, wlab: str, cells: list[str]) -> str:
        out = f"{label} & {wlab} & " + " & ".join(cells) + " \\\\"
        if cells and all(c == PENDING_TEX or c == "" for c in cells):
            out += f"  {ROW_MARK}"
        return out

    body: list[str] = []
    # Panel A: QLIKE
    body += [
        "\\multicolumn{8}{@{}l}{\\textit{Panel A: QLIKE}} \\\\",
    ]
    for m in MODELS:
        for W in WINDOWS:
            cells = []
            for bars, _, _ in BARS:
                rt = DMF.get((W, m, bars))
                cells += [
                    tcell(q_mean(W, m, "baseline", bars), 4),
                    tcell(q_mean(W, m, "all_features", bars), 4),
                    tcell(
                        None if rt is None else rt["dm"],
                        2,
                        None if rt is None else rt["p"],
                    ),
                ]
            body.append(
                line(NAME[m] if W == W_MAIN else "", f"\\rhrbWin{WTAG[W]}", cells)
            )
        cells = []
        for bars, _, _ in BARS:
            for s in SETS:
                rt = DMW.get((m, s, bars))
                cells.append(
                    tcell(
                        None if rt is None else rt["dm"],
                        2,
                        None if rt is None else rt["p"],
                        paren=True,
                    )
                )
            cells.append("")
        body.append(line("", "Change", cells))
    # Panel B: Sharpe ratio
    body += [
        "\\addlinespace[4pt]",
        "\\multicolumn{8}{@{}l}{\\textit{Panel B: Sharpe ratio of the sign strategy}} \\\\",
        " & & \\multicolumn{3}{c}{Mid-quotes} & \\multicolumn{3}{c}{Bid/ask prices} \\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(l){6-8}",
        " & & HAR & Full & $\\Delta\\SR$ & HAR & Full & $\\Delta\\SR$ \\\\",
        "\\addlinespace[2pt]",
    ]
    for m in MODELS:
        for W in WINDOWS:
            cells = []
            for side, _, _ in SIDES:
                rt = SRF.get((W, m, side))
                cells += [
                    tcell(sr(W, m, "baseline", side), 2),
                    tcell(sr(W, m, "all_features", side), 2),
                    tcell(
                        None if rt is None else rt["delta"],
                        2,
                        None if rt is None else rt["p"],
                    ),
                ]
            body.append(
                line(NAME[m] if W == W_MAIN else "", f"\\rhrbWin{WTAG[W]}", cells)
            )
        cells = []
        for side, _, _ in SIDES:
            for s in SETS:
                rt = SRW.get((m, s, side))
                cells.append(
                    tcell(
                        None if rt is None else rt["delta"],
                        2,
                        None if rt is None else rt["p"],
                    )
                )
            cells.append("")
        body.append(line("", "Change", cells))
    # no extra space between the models' blocks: the float must fit one page (\textheight at 11pt, 1in margins)
    out_body: list[str] = list(body)

    tex = f"""{HEAD}
% tab:rh_robust_w3500 (appendix): the final pooled models on a rolling window of {W_LONG:,} trading days against
% {W_MAIN:,} (same first day, frozen hyperparameters and refit at every bar); QLIKE on every bar and on the 16:00 bar of
% the {N_DAYS:,} evaluation days, DM on the daily loss (HLN, Newey-West lag {dm_lag}); Sharpe ratio (D6) of the sign
% strategy on the {N_PW} post-warmup option days, differences tested by the paired block bootstrap (hac_inference).
% Runs: {rel(linear_run(W_MAIN))}, {rel(trees_run(W_MAIN))}, {rel(rf_run(W_MAIN))} and their w{W_LONG} twins.
% Pending: {", ".join(f"{e} at W = {W:,}" for W in WINDOWS for e in PEND[W]) or "none"}.
{PENDING_DEF}
\\ifdefined\\rhrbNdays\\else\\input{{generated/rehaul_robust_w3500_numbers}}\\fi
\\begin{{table}}[tbp]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{threeparttable}}
\\caption{{Robustness: a rolling window of \\rhrbWinLong{{}} trading days}}
\\label{{tab:rh_robust_w3500}}
\\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}llrrrrrr@{{}}}}
\\toprule
 & & \\multicolumn{{3}}{{c}}{{Every bar}} & \\multicolumn{{3}}{{c}}{{16:00 bar}} \\\\
\\cmidrule(lr){{3-5}}\\cmidrule(l){{6-8}}
Model & Window & HAR & Full & $\\DM$ & HAR & Full & $\\DM$ \\\\
\\midrule
{chr(10).join(out_body)}
\\bottomrule
\\end{{tabular*}}
\\begin{{tablenotes}}[flushleft]
\\footnotesize
\\item \\textit{{Notes:}} The models of Table~\\ref{{tab:rh_oos}} on a rolling window of \\rhrbWinLong{{}} instead of
\\rhrbWinMain{{}} trading days, all else equal; HAR: HAR benchmark; Full: full predictor set. Panel A: mean QLIKE on
every bar (\\rhrbNbars{{}} bars) and on the 16:00 bar of the $T_{{\\mathrm{{oos}}}} = \\rhrbNdays$ trading days,
\\rhrbSpanStart{{}} to \\rhrbSpanEnd{{}}. $\\DM$: Diebold--Mariano statistic (Harvey--Leybourne--Newbold,
$q_{{\\mathrm{{NW}}}} = \\rhrbDMLag$) of the daily loss, full minus HAR (negative: full more accurate); Change, in
parentheses: \\rhrbWinLong-day minus \\rhrbWinMain-day loss (positive: the longer window is less accurate). Panel B:
Sharpe ratio of the sign strategy (Section~\\ref{{sec:rh_econ_measures}}) on the \\rhrbNpw{{}} post-warmup option days;
$\\Delta\\SR$: full minus HAR; Change: \\rhrbWinLong-day minus \\rhrbWinMain-day ratio; stars from the paired
block-bootstrap test of equal Sharpe ratios (Section~\\ref{{sec:rh_methods_tests}}). $^{{*}}$, $^{{**}}$, $^{{***}}$:
significance at the 10\\%, 5\\% and 1\\% levels. \\rhrbPendingNote
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}
"""
    (gen / "rehaul_robust_w3500_table.tex").write_text(
        tex, encoding="utf-8", newline="\n"
    )
    print(
        f"wrote {rel(gen / 'rehaul_robust_w3500_table.tex')}; CSVs in {rel(res)} ({time.time() - t0:.0f} s)"
    )


def econ_check(cmp, src: str, W: int, m: str, r: pd.Series, sr, SRF: dict) -> None:
    """The D6 columns of a run's econ.csv (score_trees_final.py / score14.py) against this script's."""
    for side, lab in (("mid", "mid"), ("x", "ba")):
        a, b = sr(W, m, "baseline", side), sr(W, m, "all_features", side)
        if a is None or b is None or (W, m, side) not in SRF:
            continue
        cmp(src, W, m, "baseline", f"D6 {lab}", a, r[f"d6_{lab}_har"])
        cmp(src, W, m, "all_features", f"D6 {lab}", b, r[f"d6_{lab}_full"])
        cmp(
            src,
            W,
            m,
            "full - HAR",
            f"dD6 {lab}",
            SRF[(W, m, side)]["delta"],
            r[f"dd6_{lab}"],
        )
        cmp(
            src,
            W,
            m,
            "full - HAR",
            f"dD6 bootstrap p {lab}",
            SRF[(W, m, side)]["p"],
            r[f"d6_boot_p_{lab}"],
            rtol=0,
            atol=1e-12,
        )


if __name__ == "__main__":
    main()
