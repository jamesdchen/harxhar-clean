"""causal_tune_rest_of_day -- DIRECT rest-of-day variance, one regression per entry clock.

PURPOSE: the multi-horizon counterpart of the per-bar linear campaign
(specs/causal_tune_linear.py with SEGMENT = one bar, LAG_SCOPE = global).
For an entry clock c the target is that day's realized variance from c to the
16:00 close,

    RV_rem(d, c) = sum of the panel's per-bar RV (sumret2) over the bars
                   stamped c+30, c+60, ..., 16:00 of day d,

placed on the row stamped c+30 -- the row on which the per-bar arm ``bar<c+30>``
issues its forecast of the next bar at c.  At c = 15:30 the target is exactly
the 16:00 bar's RV, so the 15:30 direct arm IS the per-bar ``bar1600`` arm (the
identity gate).  Entry clocks 09:30 (serves the 09:35 entry), 10:00, ...,
15:30: THIRTEEN clocks (the segments bar1000..bar1600).

REUSE (nothing re-derived):
  * machinery -- RollingTunedLinear, _batch_theta, fit_predict_lin_tuned,
    ESTIMATOR_GRIDS and the constants HORIZON / REFIT_FREQUENCY / TUNE_PER /
    VAL_TAIL / EMBARGO / SEED are EXECUTED FROM THE SOURCE of
    specs/causal_tune_linear.py (the named top-level statements are pulled out
    of its AST and run here, so the per-bar arms' code is the only copy);
  * data prep -- src.backtest.executor.run_executor with the per-bar arms'
    exact invocation (impute_indicate, dropna_with_exog=False, calendar,
    diurnal target, winsor 240, prescale, global lags, production HAR ladder,
    segment = the one bar stamped c+30);
  * target transform -- src.features.transforms.target.robust_transform(...,
    "RV_rem", is_target=True, use_diurnal=True, winsor_window=240), the call
    load_and_transform makes for "RV", on the SAME frame (see TARGET below).

TARGET TRANSFORM (the one design choice): robust_transform is called on the full
panel frame in which the RV column is replaced, on the clock's own rows
(stamped c+30) only, by RV_rem; every other row keeps its per-bar RV.  So
  - the diurnal baseline of the clock's rows is the per-slot causal rolling
    mean of RV_rem at that clock (DIURNAL_WINDOW prior sessions, shift 1);
  - sqrt (the column name contains "rv");
  - the causal rolling winsorization runs over the same pooled 240-bar window
    as the per-bar target (1/99 bounds, shift 1).
At c = 15:30 the frame IS the per-bar frame, so target and baseline are
bit-identical to adj_RV / baseline.  Rows whose rest of the day is incomplete
(half sessions, missing bars) are dropped from the frame before the call (no
NaN enters the rolling windows) and carry no target; their count is printed.

INJECTION: two thin wrappers installed on src.backtest.executor (module
attributes, restored by nothing -- this process runs one spec):
  load_and_transform  -> the original, then adds RV_rem / adj_RV_rem /
                         baseline_rem / rem_nbars on the returned frame (HAR
                         features are built afterwards from the untouched
                         adj_RV, so every feature is the per-bar arm's);
  _backtest_and_save  -> drops the clock's incomplete rows, swaps the target
                         (adj_RV <- adj_RV_rem, baseline <- baseline_rem),
                         asserts one row per session in strictly increasing
                         day order (so the trailing window X[t-W:t] of
                         MultiStageBacktest holds only sessions before the
                         forecast session, each of whose rest-of-day closed at
                         16:00 of an earlier day), then calls the original.
An optional raw-RV perturbation hook (leak gate only) wraps load_raw_data.

ENV (HPC_KW_* or bare): EXOG_BUCKET, ESTIMATOR, CLOCK (entry clock HHMM, e.g.
1200; empty = all 13), TRAIN_WIN (sessions, default 2000), START / END / HALO
(run_executor's chunk seam; one-bar arms run whole-series).  Output under
$HPC_RESULT_DIR/causal_tune_rest_of_day/<est>/<bucket>/:
  results_bar<HHMM of c+30>.csv   the executor's results table (date = stamp
                                  c+30, naive ET; pred_adj = the forecast)
  targets_bar<...>.csv            every clock row: RV_rem (raw, unclipped),
                                  rem_nbars, complete flag, adj_RV_rem,
                                  baseline_rem
"""

from __future__ import annotations

import ast
import math
import os
import sys
from pathlib import Path

_START = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_ROOT = _START
while not (_ROOT / "src").is_dir() and _ROOT != _ROOT.parent:
    _ROOT = _ROOT.parent
if (_ROOT / "src").is_dir():
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    os.chdir(_ROOT)

import numpy as np
import pandas as pd

import src.backtest.executor as executor
from src.backtest.executor import run_executor
from src.backtest.multi_stage import MultiStageBacktest
from src.backtest.segmentation import SEGMENT_DEFINITIONS
from src.data.loading import get_bucket
from src.features.transforms.residualizer import IdentityResidualizer
from src.features.transforms.target import robust_transform
from src.models.reclasso_har import enet_coef, enet_online, forward_window_split

LINEAR_SPEC = _ROOT / "specs" / "causal_tune_linear.py"
# The top-level statements of the per-bar spec this spec executes verbatim.
_MACHINERY = (
    "HORIZON",
    "TRAIN_WIN",
    "REFIT_FREQUENCY",
    "TUNE_PER",
    "VAL_TAIL",
    "EMBARGO",
    "SEED",
    "ESTIMATOR_GRIDS",
    "_batch_theta",
    "RollingTunedLinear",
    "fit_predict_lin_tuned",
)


def _load_linear_machinery() -> dict:
    """Execute the named top-level statements of specs/causal_tune_linear.py.

    Only assignments to / definitions of the names in _MACHINERY are taken (the
    first assignment of each constant: TRAIN_WIN is re-read from the env below
    exactly as the per-bar spec re-reads it); the spec's env reading, its arm
    loop and its incumbent run are not executed."""
    tree = ast.parse(LINEAR_SPEC.read_text(encoding="utf-8"))
    body, seen = [], set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _MACHINERY:
            body.append(node)
            seen.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {t.id for t in tgts if isinstance(t, ast.Name)}
            hit = (names & set(_MACHINERY)) - seen
            if hit:
                body.append(node)
                seen |= hit
    missing = set(_MACHINERY) - seen
    assert not missing, f"causal_tune_linear.py no longer defines {sorted(missing)}"
    ns = dict(
        np=np, pd=pd, enet_coef=enet_coef, enet_online=enet_online,
        forward_window_split=forward_window_split,
        MultiStageBacktest=MultiStageBacktest,
        IdentityResidualizer=IdentityResidualizer,
        __name__="causal_tune_linear_machinery",
    )
    exec(compile(ast.Module(body=body, type_ignores=[]), str(LINEAR_SPEC), "exec"), ns)
    return ns


M = _load_linear_machinery()
RollingTunedLinear = M["RollingTunedLinear"]
fit_predict_lin_tuned = M["fit_predict_lin_tuned"]
ESTIMATOR_GRIDS = M["ESTIMATOR_GRIDS"]
HORIZON, REFIT_FREQUENCY, SEED = M["HORIZON"], M["REFIT_FREQUENCY"], M["SEED"]

DATA_PATH = "data"
CLOSE_MIN = 16 * 60  # the 16:00 bar (15:30-16:00) is the last bar of the session
BAR_MIN = 30  # panel bar length in minutes (bar-END labelled stamps)
# Entry clocks 09:30 .. 15:30: the first remaining bar of each is a per-bar
# segment bar1000 .. bar1600.
CLOCKS = [f"{m // 60:02d}{m % 60:02d}" for m in range(9 * 60 + 30, CLOSE_MIN, BAR_MIN)]
TARGET_KW = dict(is_target=True, use_diurnal=True, winsor_window=240)  # the per-bar
# arms' target call: load_and_transform(target_use_diurnal=True, target_winsor_window=240)


def _env(name: str, default: str) -> str:
    return os.environ.get(f"HPC_KW_{name}", os.environ.get(name, default))


def clock_minutes(clock: str) -> int:
    return int(clock[:2]) * 60 + int(clock[2:])


def segment_of(clock: str) -> str:
    """The per-bar segment whose row carries clock c's forecast: the bar ending c+30."""
    m = clock_minutes(clock) + BAR_MIN
    seg = f"bar{m // 60:02d}{m % 60:02d}"
    assert seg in SEGMENT_DEFINITIONS, seg
    return seg


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------


def rest_of_day(df: pd.DataFrame, clock: str) -> pd.DataFrame:
    """RV_rem on the clock's rows (stamped c+30) of the panel frame ``df``.

    Returns a frame indexed like those rows: RV_rem (exactly rounded sum,
    math.fsum, of the per-bar RV over the bars stamped c+30 .. 16:00 of the
    same day; NaN if any of them is absent), rem_nbars (bars present),
    complete (all present)."""
    first = clock_minutes(clock) + BAR_MIN
    need = (CLOSE_MIN - first) // BAR_MIN + 1
    t = df["t"]
    mins = t.dt.hour * 60 + t.dt.minute
    day = t.dt.normalize()
    win = (mins >= first) & (mins <= CLOSE_MIN)
    g = df.loc[win, "RV"].groupby(day[win])
    s = g.agg(lambda v: math.fsum(v.to_numpy(float)))
    n = g.size()
    rows = df.index[mins == first]
    out = pd.DataFrame(index=rows)
    out["t"] = t.loc[rows]
    d = day.loc[rows]
    out["rem_nbars"] = n.reindex(d).to_numpy()
    out["complete"] = out["rem_nbars"].to_numpy() == need
    out["RV_rem"] = np.where(out["complete"], s.reindex(d).to_numpy(), np.nan)
    return out


def rest_target_columns(df: pd.DataFrame, clock: str) -> pd.DataFrame:
    """RV_rem plus its transform, by the per-bar target's own call, on ``df``'s index.

    The frame handed to robust_transform is the panel with RV replaced by RV_rem
    on the clock's rows; the clock's incomplete rows are left out of it."""
    rem = rest_of_day(df, clock)
    col = df["RV"].astype(float).copy()
    col.loc[rem.index] = rem["RV_rem"].to_numpy()
    drop = rem.index[~rem["complete"].to_numpy()]
    frame = pd.DataFrame({"RV_rem": col, "time_of_day": df["time_of_day"]}).drop(index=drop)
    adj, base = robust_transform(frame, "RV_rem", **TARGET_KW)
    out = pd.DataFrame(index=df.index, columns=["RV_rem", "adj_RV_rem", "baseline_rem",
                                                "rem_nbars"], dtype=float)
    ok = rem.index[rem["complete"].to_numpy()]
    out.loc[rem.index, "RV_rem"] = rem["RV_rem"].to_numpy()
    out.loc[rem.index, "rem_nbars"] = rem["rem_nbars"].to_numpy()
    out.loc[ok, "adj_RV_rem"] = adj.loc[ok].to_numpy()
    out.loc[ok, "baseline_rem"] = base.loc[ok].to_numpy()
    return out


# ---------------------------------------------------------------------------
# Injection into the production scaffold
# ---------------------------------------------------------------------------

_ORIG_LOAD_AND_TRANSFORM = executor.load_and_transform
_ORIG_BACKTEST_AND_SAVE = executor._backtest_and_save
_ORIG_LOAD_RAW = executor.load_raw_data

STATE: dict = {
    "clock": None,       # the active entry clock (HHMM)
    "perturb": None,     # callable(df_raw) -> df_raw, leak gate only
    "targets": None,     # the last rest-target table (evidence / sidecar)
    "dropped": None,     # incomplete rows dropped from the last segment
    "cache": None,       # {} to cache load_and_transform in-process (local gates)
}


def _load_raw_hook(*a, **kw):
    df = _ORIG_LOAD_RAW(*a, **kw)
    if STATE["perturb"] is not None:
        df = STATE["perturb"](df)
    return df


def _load_and_transform_rest(data_path, exog_cols, **kw):
    cache = STATE["cache"]
    key = (data_path, tuple(exog_cols), tuple(sorted(kw.items())))
    if cache is not None and STATE["perturb"] is None and key in cache:
        df, cols = cache[key]
        df, cols = df.copy(), list(cols)
    else:
        df, cols = _ORIG_LOAD_AND_TRANSFORM(data_path, exog_cols, **kw)
        if cache is not None and STATE["perturb"] is None:
            cache[key] = (df.copy(), list(cols))
    tgt = rest_target_columns(df, STATE["clock"])
    for c in tgt.columns:
        df[c] = tgt[c].to_numpy()
    STATE["targets"] = df.loc[df["RV_rem"].notna() | df["rem_nbars"].notna(),
                              ["t", "RV", "RV_rem", "rem_nbars", "adj_RV", "adj_RV_rem",
                               "baseline", "baseline_rem"]].copy()
    return df, cols


def _backtest_and_save_rest(df, feature_names, fit_predict, hyperparams, train_win_periods,
                            horizon, start, end, halo, output_file, prescale=False,
                            diurnal_mode="divide"):
    keep = df["adj_RV_rem"].notna().to_numpy()
    STATE["dropped"] = int((~keep).sum())
    job = df.loc[keep].reset_index(drop=True).copy()
    job["adj_RV"] = job["adj_RV_rem"].to_numpy(float)
    job["baseline"] = job["baseline_rem"].to_numpy(float)
    day = job["t"].dt.normalize().to_numpy()
    # one row per session, sessions strictly increasing: the trailing window
    # X[t-W:t] then holds only sessions before the forecast session, whose
    # rest-of-day closed at 16:00 of an earlier day, before this day's entry.
    assert (np.diff(day) > np.timedelta64(0, "ns")).all(), "a session repeats or goes back"
    print(f"[rest-of-day] clock {STATE['clock']}: {len(job)} rows with a complete rest of day, "
          f"{STATE['dropped']} incomplete rows dropped from the segment")
    return _ORIG_BACKTEST_AND_SAVE(job, feature_names, fit_predict, hyperparams,
                                   train_win_periods, horizon, start, end, halo, output_file,
                                   prescale, diurnal_mode)


executor.load_and_transform = _load_and_transform_rest
executor._backtest_and_save = _backtest_and_save_rest
executor.load_raw_data = _load_raw_hook


# ---------------------------------------------------------------------------
# One arm
# ---------------------------------------------------------------------------


def arm_dir(results_root: str, estimator: str, bucket: str) -> str:
    return os.path.join(results_root, "causal_tune_rest_of_day", estimator, bucket)


def run_arm(clock: str, bucket: str, estimator: str, *, results_root: str,
            train_win: int, start: int = 0, end: int = -1, halo: int = 0) -> pd.DataFrame:
    """Run one clock x bucket x estimator arm; return its results table."""
    STATE["clock"] = clock
    seg = segment_of(clock)
    RollingTunedLinear.grid = ESTIMATOR_GRIDS[estimator]
    d = arm_dir(results_root, estimator, bucket)
    out_csv = os.path.join(d, "results.csv")
    run_executor(
        method_name=f"rest_of_day_{estimator}",
        fit_predict=fit_predict_lin_tuned,
        hyperparams={"_refit_frequency": REFIT_FREQUENCY},
        data_path=DATA_PATH,
        output_file=out_csv,
        horizon=HORIZON,
        train_window=train_win,
        start=start,
        end=end,
        halo=halo,
        exog_cols=get_bucket(bucket),
        segment=seg,
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
        seed=SEED,
    )
    res = pd.read_csv(os.path.join(d, f"results_{seg}.csv"), parse_dates=["date"])
    tg = STATE["targets"]
    tg = tg.assign(complete=tg["adj_RV_rem"].notna())
    tg.to_csv(os.path.join(d, f"targets_{seg}.csv"), index=False)
    alphas = [a for _, a, _ in RollingTunedLinear.trace]
    print(f"{estimator}/{bucket}/{clock} ({seg}): {len(res)} OOS rows "
          f"{res['date'].min().date()} .. {res['date'].max().date()}; target rows "
          f"{int(tg['complete'].sum())} complete, {int((~tg['complete']).sum())} incomplete; "
          f"{len(alphas)} tunings, alpha path min={min(alphas):.2e} max={max(alphas):.2e}; "
          f"masked cols per tune min={min(RollingTunedLinear.mask_trace)} "
          f"max={max(RollingTunedLinear.mask_trace)}; between-tune mask additions "
          f"{len(RollingTunedLinear.reseed_trace)}")
    return res


def main() -> None:
    bucket = _env("EXOG_BUCKET", "live_feasible")
    est = _env("ESTIMATOR", "")
    clk = _env("CLOCK", "")
    train_win = int(_env("TRAIN_WIN", "2000"))
    start, end, halo = int(_env("START", "0")), int(_env("END", "-1")), int(_env("HALO", "0"))
    results_root = os.environ.get("HPC_RESULT_DIR", "results")
    ests = [est] if est else list(ESTIMATOR_GRIDS)
    clocks = [c for c in clk.replace(";", ",").split(",") if c] if clk else CLOCKS
    for c in clocks:
        assert c in CLOCKS, f"clock {c} not in {CLOCKS}"
    for e in ests:
        for c in clocks:
            run_arm(c, bucket, e, results_root=results_root, train_win=train_win,
                    start=start, end=end, halo=halo)


if __name__ == "__main__":
    main()
