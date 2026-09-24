"""Study 80 -- the earliest decision clock: forecast the 15:30-16:00 bar from 15:00, 14:30, 14:00.

QUESTION (the operator's).  The card forecasts the variance of the SPX
15:30-16:00 bar with data through the 15:30 stamp (h = 1) and at 15:30 buys the
SPXW 0DTE nearest-OTM straddle iff its ASK <= P* = Black-76 package price at
total vol sqrt(rv_hat) (``live.ibkr.pricing.package_price``), held to the 16:00
cash settlement, long only.  The card lands ~15:34, too tight.  Keep the 15:30
purchase; make the forecast from data through 15:00 (h = 2), 14:30 (h = 3),
14:00 (h = 4).  What is the earliest h whose card rule is still profitable?

METHOD -- a DIRECT h-step forecast with the deck's own engine.  The deck's
rv_hat (``daily_sub_live_ridge``) is the per-bar ridge arm of
``specs/causal_tune_linear.py`` (ESTIMATOR ridge, EXOG_BUCKET live_feasible,
LAG_SCOPE global, TRAIN_WIN 2000 sessions, SEGMENT bar1000 .. bar1600, run
through ``src.backtest.executor.run_executor``; rolling rank-1 refit every
session, alpha re-tuned every 250 solves on a trailing forward split), the 13
arms stacked into a yhat table and mapped by the deck's loader
``atm_straddle_lib.load_yhat_1530_mz_cached`` (the causal second-order MZ map:
per-session a, b, s2 fitted by weighted LS on the 10:30..16:00 stamps of the
250 sessions strictly before the day; rv_hat = (m^2 + s2) * baseline).  That
engine is run here UNCHANGED except for ONE substitution: every data-derived
regressor of the design is a HAR rolling mean built by
``src.features.extractors.har.generate_har_features`` as ``.shift(1)`` of the
panel; for horizon h it is built as ``.shift(h)`` (the function's own source,
with that one token replaced; asserted to occur exactly once).  The panel is
row-per-bar with bar-END naive-ET stamps, so the 16:00 row's regressors then
end at the 16:00 - 30*(h-1) min stamp: 15:30 / 15:00 / 14:30 / 14:00 for
h = 1..4.  Nothing else in the design is data-derived at the row: calendar and
expiry columns are functions of the timestamp and the session calendar; the
HAR x open/close gates multiply the (shifted) HAR columns; the target's
diurnal baseline is a shift(1) per-slot rolling statistic of PRIOR sessions.
The ridge at a session trains on the prior 2000 sessions' same-bar rows, whose
targets closed on earlier days; the MZ map uses strictly prior sessions.  So
every forecast is F-measurable at its decision stamp.  The same substitution is
applied to all 13 bars (each bar forecast h bars ahead) and the SAME MZ map is
fitted per horizon on that horizon's 13-bar table -- the calibration approach
is identical at every h.  h = 1 is the substitution with h = 1, i.e. the deck.

Engine plumbing: the spec's module-level code runs every arm plus an OLS
incumbent on load, so its estimator (``RollingTunedLinear``,
``fit_predict_lin_tuned``, ``_batch_theta`` and the constants they read) is
lifted by AST (imports + defs + the tuning constants only) and handed to
``run_executor`` with the spec's own argument list; the 13 bars run inside ONE
executor call per horizon (``segment="all"`` over the 13 bar segments -- the
global HAR build is shared, each segment is sliced, prescaled and backtested
exactly as a single-segment run).  The data root holds only the four vendor
files, as ``live.close_signal.forecast.vendor_root`` does.  Horizons run as
parallel processes (ProcessPoolExecutor).

GATES.
  1. h = 1 must BE the research: the assembled table equals
     ``results/spxw_pnl/yhat_sub_ridge_live_feasible.parquet`` on every stamp
     (baseline and rv_raw exactly, yhat to 1e-6 relative), and rv_hat equals the
     deck's on all 866 days (1e-7 relative, every sign(s) agreeing).  If not,
     the script stops before any h >= 2 number.
  2. LEAK TEST (perturbation): on a mid-sample deck session the realized
     variance (sumret2) of the bars after the decision stamp -- and of the
     16:00 target bar itself -- is multiplied by 50 and the bar1600 arm re-run
     at that h; its 16:00 forecast that day must be bit-identical to the
     unperturbed run (and every earlier row too).  Positive control: at h = 1
     the 15:30 bar IS information, and perturbing it must move the forecast.
     The perturbation is on the target channel only: every exog column is
     lagged by the same ``generate_har_features`` call, so the channel is the
     representative one; perturbing an exog column would also move the
     full-sample medians inherited from the pipeline (caveat below) and so
     could not be bit-exact.
  3. The SPXW book: the 15:30 ET rows of ``data/spxw_chain.parquet`` (timestamp
     is TRUE UTC -> America/New_York), expiration == the day, ask > 0, the
     nearest-OTM pair on the listed strikes around the median 15:30 underlying
     (study 79's ``spxw_book``, imported); its strikes are compared with the
     deck's K_c / K_p.

INHERITED CAVEAT (not fixed here, identical at every h, because h = 1 must BE
the deck): ``src.backtest.executor.load_and_transform`` fills unobserved exog
rows with the FULL-SAMPLE median of the transformed column
(``adj.fillna(adj.median())``, the 2026-09-05 audit's "full-sample median
fill"), and ``src.features.transforms.target`` has two full-sample medians
used as floors (signed-feature diurnal std floor; asinh scale).  One scalar per
column, a known pipeline defect of the deck itself.

SCORING (866 deck days, 2020-01-03 .. 2024-04-30).  Card rule at horizon h:
buy iff ask_1530 <= P*(rv_hat_h) = package_price(sqrt(rv_hat_h), S, Kc, Kp) on
the book's pair; payoff at settlement max(S_close - Kc, 0) + max(Kp - S_close,
0) with the deck's S_close; R = payoff / ask - 1.  Also at the mid for
comparison with the deck: long iff rv_hat_h > iv_var (the deck's sign(s) long
leg), R = the deck's R (its SPXW mid entry).  Per horizon: days traded, mean R,
iid t, hit rate, sum R, QLIKE of rv_hat_h vs the realized 16:00-bar variance
(the panel's sumret2 at the 16:00 stamp).  Month-ends (last session of month,
``live.ibkr.calendar_guard``) are reported separately: the live card buys them
UNCONDITIONALLY, so that leg does not depend on the forecast clock at all.
The h - 1 difference: a circular block bootstrap (20-session blocks, 10,000
draws) of the mean per-day P&L at the ask, P&L = R on a traded day and 0 on a
no-trade day, over all days with a 15:30 book (the SAME days for both h).

OUTPUTS (results/atm_straddle_intraday_holdclose/proposals/80/):
  arms/h{h}/causal_tune_linear/results_bar*.csv   the raw arm outputs
  yhat_h{h}.parquet        the 13-bar yhat table per horizon (t UTC, yhat, baseline, rv_raw)
  forecasts.parquet        per (date, h): yhat, baseline, m, s2, rv_hat, rv_raw, deck rv_hat
  per_day_pnl.parquet/.csv per (date, h): book, P*, trade flags, R at ask / mid, P&L
  summary_by_horizon.csv   the table (scope all / non_month_end / month_end)
  bootstrap_diff.csv       h - 1 per-day P&L difference CIs
  forecast_diagnostics.csv per h: QLIKE, level bias, correlations
  leak_test.csv            the perturbation test
  gate.json, run_log.txt   the gate numbers and the wall clock (run.log: stdout)

REPRODUCE (295 s measured 2026-09-24; 4 horizon arms + 5 leak arms, 4 processes, ~1.3 GB each):
    C:/Users/james/miniconda3/envs/285J/python.exe writeup/intraday_proposals/80_earliest_decision_clock.py
    (--reuse-arms skips the arm runs when their CSVs exist; --no-leak-test;
     --horizons 1,2,3,4; --workers 4)
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "results" / "atm_straddle_intraday_holdclose" / "proposals" / "80"
SPEC = REPO / "specs" / "causal_tune_linear.py"
DECK = REPO / "results" / "atm_straddle_0dte_1530" / "daily_sub_live_ridge.parquet"
RESEARCH_TABLE = REPO / "results" / "spxw_pnl" / "yhat_sub_ridge_live_feasible.parquet"
VENDOR_FILES = (
    "core_stats.parquet",
    "vix_and_voldemand.parquet",
    "releases.parquet",
    "time_categories.parquet",
)
BARS = tuple(
    f"bar{h:02d}{m:02d}" for h in range(10, 17) for m in (0, 30) if (h, m) <= (16, 0)
)
BUCKET = "live_feasible"  # the deck's research bucket (DECK_BUCKET in forecast.py)
TRAIN_WIN = 2000
DECISION_STAMP = {1: "15:30", 2: "15:00", 3: "14:30", 4: "14:00"}
BLOCK = 20
N_BOOT = 10_000
SEED = 80
# gate tolerances: forecast.assert_reproduces_deck's (one decade above the
# measured cross-machine float-path deviations)
YHAT_REL_TOL = 1e-6
RV_HAT_REL_TOL = 1e-7


# --------------------------------------------------------------------------- engine
def _spec_estimator() -> dict:
    """The spec's estimator lifted by AST: imports, defs and the tuning constants only."""
    tree = ast.parse(SPEC.read_text(encoding="utf-8"))
    keep_names = {
        "HORIZON",
        "REFIT_FREQUENCY",
        "TUNE_PER",
        "VAL_TAIL",
        "EMBARGO",
        "SEED",
        "ESTIMATOR_GRIDS",
    }
    body: list[ast.stmt] = []
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            body.append(n)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            tg = n.targets if isinstance(n, ast.Assign) else [n.target]
            if all(isinstance(t, ast.Name) and t.id in keep_names for t in tg):
                body.append(n)
    ns: dict = {"__name__": "causal_tune_linear_lifted", "__file__": str(SPEC)}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SPEC), "exec"), ns)
    assert ns["HORIZON"] == 1 and ns["REFIT_FREQUENCY"] == 1
    assert (ns["TUNE_PER"], ns["VAL_TAIL"], ns["EMBARGO"]) == (250, 125, 25)
    return ns


def _har_shifted(h: int):
    """generate_har_features with its one ``.shift(1)`` replaced by ``.shift(h)``."""
    from src.features.extractors import har

    src = inspect.getsource(har.generate_har_features)
    token = ".mean().shift(1)"
    assert src.count(token) == 1, (
        "generate_har_features changed: re-derive the substitution"
    )
    src = src.replace(token, ".mean().shift(_H_SHIFT)")
    ns = dict(vars(har))
    ns["_H_SHIFT"] = int(h)
    exec(compile(src, har.__file__, "exec"), ns)
    return ns["generate_har_features"]


def run_horizon(
    h: int, data_dir: str, out_dir: str, bars: tuple[str, ...] = BARS
) -> dict[str, str]:
    """The bar arms at horizon h in one executor call; bar -> results CSV."""
    os.chdir(REPO)
    import src.backtest.executor as E
    from src.backtest.segmentation import BAR_SEGMENTS
    from src.data.loading import get_bucket

    t0 = time.time()
    ns = _spec_estimator()
    ns["RollingTunedLinear"].grid = ns["ESTIMATOR_GRIDS"]["ridge"]
    E.generate_har_features = _har_shifted(h)
    E.SEGMENT_DEFINITIONS = {b: BAR_SEGMENTS[b] for b in bars}
    out = Path(out_dir) / f"h{h}" / "causal_tune_linear" / "results.csv"
    # the spec's run_executor call, argument for argument (TRAIN_WIN, the bucket
    # and the segment as the research arm's HPC_KW_* env set them)
    E.run_executor(
        method_name="lin_tuned_ridge",
        fit_predict=ns["fit_predict_lin_tuned"],
        hyperparams={"_refit_frequency": ns["REFIT_FREQUENCY"]},
        data_path=data_dir,
        output_file=str(out),
        horizon=ns["HORIZON"],
        train_window=TRAIN_WIN,
        start=0,
        end=-1,
        halo=0,
        exog_cols=get_bucket(BUCKET),
        segment="all",
        lag_scope="global",
        har_lags=None,
        add_calendar=True,
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=False,
        overnight_fill=True,
        impute_indicate=True,
        diurnal_mode="divide",
        prescale=True,
        seed=ns["SEED"],
    )
    csvs = {b: str(out.with_name(f"results_{b}.csv")) for b in bars}
    missing = [b for b, p in csvs.items() if not Path(p).exists()]
    if missing:
        raise RuntimeError(f"h={h}: no arm output for {missing}")
    print(f"[h={h}] {len(bars)} arms done in {time.time() - t0:.0f}s", flush=True)
    return csvs


#: Perturbation cases (h, stamps of the test day whose realized variance is
#: multiplied by LEAK_FACTOR, the 16:00 forecast must be invariant?).  The
#: 16:00 row is the target itself (the placeholder claim); the rows between the
#: decision stamp and 16:00 are what an h-step forecast must not see.  The
#: second h = 1 case is the POSITIVE control: the 15:30 bar is in h = 1's
#: information set, so the test must see it move.
LEAK_CASES: tuple[tuple[int, tuple[str, ...], bool], ...] = (
    (1, ("16:00",), True),
    (1, ("15:30", "16:00"), False),
    (2, ("15:30", "16:00"), True),
    (3, ("15:00", "15:30", "16:00"), True),
    (4, ("14:30", "15:00", "15:30", "16:00"), True),
)
LEAK_FACTOR = 50.0


def _leak_case(
    k: int, h: int, stamps: tuple[str, ...], day: str, vendor_data: str, work: str
) -> str:
    """One perturbed bar1600 arm: the day's sumret2 x LEAK_FACTOR at ``stamps``."""
    d = Path(work) / f"leak{k}" / "data"
    d.mkdir(parents=True, exist_ok=True)
    for f in VENDOR_FILES:
        if f != "core_stats.parquet":
            shutil.copy2(Path(vendor_data) / f, d / f)
    core = pd.read_parquet(Path(vendor_data) / "core_stats.parquet")
    t = pd.to_datetime(core["endbartime"])
    hit = t.isin([pd.Timestamp(f"{day} {s}") for s in stamps])
    assert int(hit.sum()) == len(stamps), (day, stamps, int(hit.sum()))
    core.loc[hit, "sumret2"] = core.loc[hit, "sumret2"] * LEAK_FACTOR
    core.to_parquet(d / "core_stats.parquet", index=False)
    return run_horizon(h, str(d), str(Path(work) / f"leak{k}" / "arms"), ("bar1600",))[
        "bar1600"
    ]


# --------------------------------------------------------------------------- helpers
def _import_79():
    p = REPO / "writeup" / "intraday_proposals" / "79_xsp_close_book.py"
    spec = importlib.util.spec_from_file_location("p79_xsp_close_book", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def qlike(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    r = y / f
    return r - np.log(r) - 1.0


def stats(r: pd.Series) -> dict[str, float]:
    r = r.dropna()
    n = len(r)
    if n == 0:
        return {"n": 0, "mean_R": np.nan, "t": np.nan, "hit": np.nan, "sum_R": 0.0}
    sd = r.std(ddof=1) if n > 1 else np.nan
    return {
        "n": n,
        "mean_R": float(r.mean()),
        "t": float(r.mean() / (sd / np.sqrt(n))) if n > 1 and sd > 0 else np.nan,
        "hit": float((r > 0).mean()),
        "sum_R": float(r.sum()),
    }


def block_boot_idx(n: int, block: int, n_boot: int, rng) -> np.ndarray:
    """Circular block bootstrap index matrix (n_boot, n)."""
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_boot, nb * block)[:, :n]


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="1,2,3,4")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reuse-arms", action="store_true")
    ap.add_argument("--no-leak-test", action="store_true")
    a = ap.parse_args(argv)
    hs = [int(x) for x in a.horizons.split(",")]
    assert 1 in hs, "h=1 is the gate"
    T0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    arms_dir = OUT / "arms"
    work = Path(tempfile.mkdtemp(prefix="p80_"))
    root = work / "vendor_root"
    (root / "data").mkdir(parents=True)
    for f in VENDOR_FILES:
        shutil.copy2(REPO / "data" / f, root / "data" / f)
    log: dict[str, object] = {"horizons": hs}

    # ---- 1. the arms, one process per horizon
    def csvs_for(h: int) -> dict[str, str]:
        base = arms_dir / f"h{h}" / "causal_tune_linear"
        return {b: str(base / f"results_{b}.csv") for b in BARS}

    todo = [
        h
        for h in hs
        if not (a.reuse_arms and all(Path(p).exists() for p in csvs_for(h).values()))
    ]
    deck = pd.read_parquet(DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).normalize()
    leak_day = str(deck.index[len(deck) // 2].date())  # a mid-sample deck session
    t1 = time.time()
    for env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[env_var] = "1"
    leak_futs = {}
    n_jobs = len(todo) + (0 if a.no_leak_test else len(LEAK_CASES))
    if n_jobs:
        with ProcessPoolExecutor(max_workers=max(1, min(a.workers, n_jobs))) as ex:
            futs = {
                h: ex.submit(run_horizon, h, str(root / "data"), str(arms_dir))
                for h in todo
            }
            if not a.no_leak_test:
                for k, (h, stamps, _) in enumerate(LEAK_CASES):
                    leak_futs[k] = ex.submit(
                        _leak_case,
                        k,
                        h,
                        stamps,
                        leak_day,
                        str(root / "data"),
                        str(work),
                    )
            for h, fu in futs.items():
                fu.result()
            leak_csv = {k: fu.result() for k, fu in leak_futs.items()}
    log["arms_wall_s"] = round(time.time() - t1, 1)
    log["arms_run_for"] = todo
    print(
        f"arms: {log['arms_wall_s']}s for horizons {todo} (+ {len(leak_futs)} leak-test arms)",
        flush=True,
    )

    from live.close_signal import forecast as F

    # ---- 1b. LEAK TEST: perturb the test day's realized variance after the decision stamp
    if leak_futs:
        cut = pd.Timestamp(f"{leak_day} 16:00")
        rows_lt = []
        for k, (h, stamps, expect_same) in enumerate(LEAK_CASES):
            ref_ = pd.read_csv(csvs_for(h)["bar1600"], parse_dates=["date"]).set_index(
                "date"
            )
            per = pd.read_csv(leak_csv[k], parse_dates=["date"]).set_index("date")
            ref_, per = ref_[ref_.index <= cut], per[per.index <= cut]
            assert ref_.index.equals(per.index) and ref_.index[-1] == cut
            dev = np.abs(per["pred_adj"] - ref_["pred_adj"]) / np.abs(ref_["pred_adj"])
            rows_lt.append(
                {
                    "case": k,
                    "h": h,
                    "decision_stamp": DECISION_STAMP[h],
                    "day": leak_day,
                    "perturbed_stamps": " ".join(stamps),
                    "factor": LEAK_FACTOR,
                    "expect_invariant": expect_same,
                    "rel_dev_16h00_forecast": float(dev.iloc[-1]),
                    "max_rel_dev_earlier_rows": float(dev.iloc[:-1].max()),
                    "pass": bool(
                        (dev.iloc[-1] == 0.0) if expect_same else (dev.iloc[-1] > 1e-6)
                    )
                    and bool(dev.iloc[:-1].max() == 0.0),
                }
            )
        lt = pd.DataFrame(rows_lt)
        lt.to_csv(OUT / "leak_test.csv", index=False)
        print(
            "LEAK TEST (bar1600 arm, sumret2 x50 on the listed stamps of the test day):",
            flush=True,
        )
        print(lt.to_string(index=False), flush=True)
        log["leak_test_pass"] = bool(lt["pass"].all())
        if not lt["pass"].all():
            print("LEAK TEST FAILED -- no h >= 2 number is reported", flush=True)
            return 3

    # ---- 2. tables + the deck's MZ map per horizon
    fc = []
    tabs = {}
    for h in hs:
        csvs = {b: Path(p) for b, p in csvs_for(h).items()}
        tab = F.assemble_yhat_table(csvs, root)
        tab.to_parquet(OUT / f"yhat_h{h}.parquet", index=False)
        tabs[h] = tab
        px = F.recalibrate(tab, deck.index, REPO, work, tag=f"p80_h{h}")
        px = px.reindex(deck.index)
        px["h"] = h
        px["decision_stamp"] = DECISION_STAMP[h]
        fc.append(px)
    fc = pd.concat(fc)
    fc.index.name = "date"
    fc["deck_rv_hat"] = deck["rv_hat"].reindex(fc.index).to_numpy()
    fc.reset_index().to_parquet(OUT / "forecasts.parquet", index=False)

    # ---- 3. GATE: h=1 is the research table and the deck
    ref = pd.read_parquet(RESEARCH_TABLE).sort_values("t").reset_index(drop=True)
    j = ref.merge(tabs[1], on="t", how="outer", suffixes=("_ref", ""), indicator=True)
    both = j[j["_merge"] == "both"]
    f1 = fc[fc.h == 1]
    rel = (np.abs(f1["rv_hat"] - deck["rv_hat"]) / deck["rv_hat"]).to_numpy(float)
    y = f1["rv_raw"].to_numpy(float)
    gate = {
        "table_rows": int(len(tabs[1])),
        "table_rows_unmatched": int((j["_merge"] != "both").sum()),
        "table_max_rel_yhat": float(
            (np.abs(both["yhat"] - both["yhat_ref"]) / np.abs(both["yhat_ref"])).max()
        ),
        "table_max_abs_baseline": float(
            np.abs(both["baseline"] - both["baseline_ref"]).max()
        ),
        "table_max_abs_rv_raw": float(
            np.abs(both["rv_raw"] - both["rv_raw_ref"]).max()
        ),
        "deck_days": int(len(deck)),
        "deck_days_matched": int(np.isfinite(rel).sum()),
        "deck_max_rel_rv_hat": float(np.nanmax(rel)),
        "deck_corr_rv_hat": float(np.corrcoef(f1["rv_hat"], deck["rv_hat"])[0, 1]),
        "deck_signs_agree": int(
            (((f1["rv_hat"] - deck["iv_var"]) > 0) == (deck["signal"] > 0)).sum()
        ),
        "qlike_h1": float(qlike(y, f1["rv_hat"].to_numpy(float)).mean()),
        "qlike_deck": float(qlike(y, deck["rv_hat"].to_numpy(float)).mean()),
    }
    gate["passed"] = bool(
        gate["table_rows_unmatched"] == 0
        and gate["table_max_rel_yhat"] <= YHAT_REL_TOL
        and gate["table_max_abs_baseline"] == 0.0
        and gate["table_max_abs_rv_raw"] == 0.0
        and gate["deck_days_matched"] == len(deck)
        and gate["deck_max_rel_rv_hat"] <= RV_HAT_REL_TOL
        and gate["deck_signs_agree"] == len(deck)
    )
    (OUT / "gate.json").write_text(json.dumps(gate, indent=1))
    print("GATE h=1 vs research table + deck:", json.dumps(gate, indent=1), flush=True)
    if not gate["passed"]:
        print("GATE FAILED -- no h >= 2 number is reported", flush=True)
        return 2

    # ---- 3b. forecast diagnostics per horizon on the 866 deck days
    w = fc.pivot_table(index=fc.index, columns="h", values="rv_hat")
    yv = fc.loc[fc.h == 1, "rv_raw"].reindex(w.index)
    diag = pd.DataFrame(
        [
            {
                "h": h,
                "decision_stamp": DECISION_STAMP[h],
                "qlike": float(qlike(yv.to_numpy(), w[h].to_numpy()).mean()),
                "mean_rv_hat_over_mean_rv": float(w[h].mean() / yv.mean()),
                "median_rv_hat_over_rv": float(np.median(w[h] / yv)),
                "corr_log_rv_hat_log_rv": float(
                    np.corrcoef(np.log(w[h]), np.log(yv))[0, 1]
                ),
                "corr_log_with_h1": float(
                    np.corrcoef(np.log(w[h]), np.log(w[1]))[0, 1]
                ),
                "frac_rv_hat_gt_iv_var": float(
                    (w[h] > deck["iv_var"].reindex(w.index)).mean()
                ),
            }
            for h in hs
        ]
    )
    diag.to_csv(OUT / "forecast_diagnostics.csv", index=False)
    print("\nFORECAST DIAGNOSTICS (16:00 bar, 866 deck days):", flush=True)
    print(diag.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)

    # ---- 4. the SPXW 15:30 book (study 79's spxw_book) and the card rule per horizon
    from live.ibkr.calendar_guard import is_last_session_of_month
    from live.ibkr.pricing import package_price

    p79 = _import_79()
    book = p79.spxw_book(deck.index)
    book.index = pd.DatetimeIndex(book.index).normalize()
    book = book.reindex(deck.index)
    has_book = book["spx_ask"].notna()
    strikes_match = int(
        ((book["spx_kc"] == deck["K_c"]) & (book["spx_kp"] == deck["K_p"]))[
            has_book
        ].sum()
    )
    month_end = pd.Series(
        [is_last_session_of_month(d.date()) for d in deck.index], index=deck.index
    )
    payoff = np.maximum(deck["S_close"] - book["spx_kc"], 0.0) + np.maximum(
        book["spx_kp"] - deck["S_close"], 0.0
    )
    R_ask_all = payoff / book["spx_ask"] - 1.0
    rows = []
    for h in hs:
        rv_h = fc.loc[fc.h == h, "rv_hat"].reindex(deck.index)
        pstar = pd.Series(
            [
                package_price(np.sqrt(v), s, kc, kp)
                if np.isfinite(v) and np.isfinite(s)
                else np.nan
                for v, s, kc, kp in zip(
                    rv_h, book["spx_S"], book["spx_kc"], book["spx_kp"]
                )
            ],
            index=deck.index,
        )
        trade = has_book & (book["spx_ask"] <= pstar)
        long_mid = rv_h > deck["iv_var"]
        rows.append(
            pd.DataFrame(
                {
                    "date": deck.index,
                    "h": h,
                    "decision_stamp": DECISION_STAMP[h],
                    "month_end": month_end.to_numpy(),
                    "rv_hat": rv_h.to_numpy(),
                    "rv_raw": fc.loc[fc.h == h, "rv_raw"]
                    .reindex(deck.index)
                    .to_numpy(),
                    "has_book": has_book.to_numpy(),
                    "S_1530": book["spx_S"].to_numpy(),
                    "Kc": book["spx_kc"].to_numpy(),
                    "Kp": book["spx_kp"].to_numpy(),
                    "bid": book["spx_bid"].to_numpy(),
                    "ask": book["spx_ask"].to_numpy(),
                    "P_star": pstar.to_numpy(),
                    "S_close": deck["S_close"].to_numpy(),
                    "payoff": payoff.to_numpy(),
                    "trade_ask": trade.to_numpy(),
                    "R_ask": np.where(trade, R_ask_all, np.nan),
                    "pnl_ask": np.where(
                        has_book, np.where(trade, R_ask_all, 0.0), np.nan
                    ),
                    "iv_var": deck["iv_var"].to_numpy(),
                    "entry_mid": deck["entry"].to_numpy(),
                    "trade_mid": long_mid.to_numpy(),
                    "R_mid": np.where(long_mid, deck["R"], np.nan),
                    "R_ask_unconditional": R_ask_all.to_numpy(),
                }
            )
        )
    pdl = pd.concat(rows, ignore_index=True)
    pdl.to_parquet(OUT / "per_day_pnl.parquet", index=False)
    pdl.to_csv(OUT / "per_day_pnl.csv", index=False)

    summ = []
    for h in hs:
        d = pdl[pdl.h == h]
        q = qlike(d["rv_raw"].to_numpy(float), d["rv_hat"].to_numpy(float))
        for scope, m in (
            ("all_days", np.ones(len(d), bool)),
            ("non_month_end", ~d["month_end"].to_numpy()),
            ("month_end_rule", d["month_end"].to_numpy()),
        ):
            dd = d[m]
            sa = stats(dd.loc[dd.trade_ask, "R_ask"])
            sm = stats(dd.loc[dd.trade_mid, "R_mid"])
            summ.append(
                {
                    "h": h,
                    "decision_stamp": DECISION_STAMP[h],
                    "scope": scope,
                    "days": int(len(dd)),
                    "days_with_book": int(dd.has_book.sum()),
                    "traded_ask": sa["n"],
                    "mean_R_ask": sa["mean_R"],
                    "t_ask": sa["t"],
                    "hit_ask": sa["hit"],
                    "sum_R_ask": sa["sum_R"],
                    "mean_pnl_per_day_ask": float(dd["pnl_ask"].mean()),
                    "traded_mid": sm["n"],
                    "mean_R_mid": sm["mean_R"],
                    "t_mid": sm["t"],
                    "hit_mid": sm["hit"],
                    "sum_R_mid": sm["sum_R"],
                    "qlike": float(np.nanmean(q[m])),
                }
            )
    summ = pd.DataFrame(summ)
    # the month-end leg as the card runs it: bought unconditionally at the ask (h-independent)
    me = pdl[(pdl.h == hs[0]) & pdl.month_end & pdl.has_book]
    sme = stats(me["R_ask_unconditional"])
    me_row = {
        "h": "any",
        "decision_stamp": "n/a",
        "scope": "month_end_unconditional",
        "days": int(len(pdl[(pdl.h == hs[0]) & pdl.month_end])),
        "days_with_book": int(len(me)),
        "traded_ask": sme["n"],
        "mean_R_ask": sme["mean_R"],
        "t_ask": sme["t"],
        "hit_ask": sme["hit"],
        "sum_R_ask": sme["sum_R"],
    }
    summ = pd.concat([summ, pd.DataFrame([me_row])], ignore_index=True)
    summ.to_csv(OUT / "summary_by_horizon.csv", index=False)

    # ---- 5. block bootstrap of h - 1 on per-day P&L (0 on no-trade days), same days
    base = pdl[pdl.h == 1].set_index("date")
    boots = []
    for scope in ("all_days", "non_month_end"):
        keep = base.has_book & (~base.month_end if scope == "non_month_end" else True)
        days_s = base.index[keep]  # sorted by date: the blocks are runs of sessions
        rng = np.random.default_rng(SEED)
        idx = block_boot_idx(len(days_s), BLOCK, N_BOOT, rng)
        p1 = base.loc[days_s, "pnl_ask"].to_numpy(float)
        t1 = base.loc[days_s, "trade_ask"].to_numpy(bool)
        for h in [x for x in hs if x != 1]:
            other = pdl[pdl.h == h].set_index("date")
            ph = other.loc[days_s, "pnl_ask"].to_numpy(float)
            th = other.loc[days_s, "trade_ask"].to_numpy(bool)
            dlt = ph - p1
            bm = dlt[idx].mean(1)
            # secondary: traded-day mean R, each horizon on its own traded days,
            # the same resampled sessions for both
            with np.errstate(invalid="ignore", divide="ignore"):
                mr = ph[idx].sum(1) / th[idx].sum(1) - p1[idx].sum(1) / t1[idx].sum(1)
            boots.append(
                {
                    "h_minus_1": f"{h}-1",
                    "scope": scope,
                    "days": int(len(days_s)),
                    "stat": "mean per-day P&L at ask (R on a traded day, 0 on a no-trade day)",
                    "point": float(dlt.mean()),
                    "ci_lo_95": float(np.quantile(bm, 0.025)),
                    "ci_hi_95": float(np.quantile(bm, 0.975)),
                    "p_boot_gt0": float((bm > 0).mean()),
                    "traded_day_meanR_point": float(
                        ph.sum() / th.sum() - p1.sum() / t1.sum()
                    ),
                    "traded_day_meanR_ci_lo_95": float(np.nanquantile(mr, 0.025)),
                    "traded_day_meanR_ci_hi_95": float(np.nanquantile(mr, 0.975)),
                    "block": BLOCK,
                    "n_boot": N_BOOT,
                }
            )
    boots = pd.DataFrame(boots)
    boots.to_csv(OUT / "bootstrap_diff.csv", index=False)

    # ---- 6. report
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print(
        f"\nBOOK: {int(has_book.sum())} of {len(deck)} deck days have a 15:30 SPXW book; "
        f"pair == deck (K_c, K_p) on {strikes_match}; month-ends {int(month_end.sum())} "
        f"({int((month_end & has_book).sum())} with a book)",
        flush=True,
    )
    cols = [
        "h",
        "decision_stamp",
        "scope",
        "traded_ask",
        "mean_R_ask",
        "t_ask",
        "hit_ask",
        "sum_R_ask",
        "mean_pnl_per_day_ask",
        "traded_mid",
        "mean_R_mid",
        "t_mid",
        "qlike",
    ]
    print(
        summ[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True
    )
    print("\nBLOCK BOOTSTRAP (20-session circular blocks, 10,000 draws):", flush=True)
    print(boots.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    rng_q = {
        int(h): float(summ[(summ.h == h) & (summ.scope == "all_days")]["qlike"].iloc[0])
        for h in hs
    }
    log.update(
        {
            "book_days": int(has_book.sum()),
            "book_pair_equals_deck": strikes_match,
            "month_end_days": int(month_end.sum()),
            "qlike_by_h": rng_q,
            "total_wall_s": round(time.time() - T0, 1),
        }
    )
    (OUT / "run_log.txt").write_text(json.dumps(log, indent=1))
    print("\nwall clock:", json.dumps(log, indent=1), flush=True)
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
