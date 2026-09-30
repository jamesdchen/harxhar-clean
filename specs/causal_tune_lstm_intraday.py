"""causal_tune_lstm_intraday -- an LSTM over the day's half-hour bars for the 15:30-16:00 bar.

PURPOSE (user decision 2026-09-29, late evening; campaign plan
writeup/CAMPAIGN_16H_2026-09-29.md, agent I): the per-bar LSTM
(specs/causal_tune_lstm.py) reads a sequence of the last L SESSIONS' 16:00-row
vectors of the per-bar design, i.e. of the HAR ladders.  This spec changes ONLY
the sequence and its inputs: the sequence is the last N HALF-HOUR BARS of the
near-24-hour panel ending with the 15:30 bar, one step per bar, with bar-level
inputs instead of the ladders, so the network has to learn the aggregation the
ladders impose.  Everything else is the per-bar LSTM's.

ONE ROW PER SESSION, THE PER-BAR ARMS' TARGET.  The rows, stamps and target are
those of the per-bar linear / tree / LSTM arms at bar1600: the executor's
load_and_transform (target: diurnal -> sqrt -> rolling winsorisation, window 240;
impute_indicate=True, dropna_with_exog=False, overnight fills), the segment slice
of the global-lag frame, the 3125-row HAR burn-in drop, horizon 1, the 2000-row
window, the executor's own results table (results_<seg>.csv, its Duan pred_raw).
The spec calls the executor's own pieces (load_and_transform ->
_build_har_and_calendar -> slice_to_segment -> _backtest_and_save) with the
linear arms' keyword arguments (executor_kwargs); _backtest_and_save is handed
the per-bar BASELINE design (har_ma_* + calendar), which carries its prescale /
feature-health path unchanged and is never read by the network.  Gate
(experiments/gate_lstm_intraday.py, "identity"): sha256 of the target and stamps
this spec's model receives = those of the per-bar linear spec's own invocation.

THE STEP MATRIX P (one row per panel bar, prepare()): the executor's transformed
frame, bar level, never the HAR ladders --
  adj_RV                       the bar's adjusted target value (the same column the
                               target and the har_ma_* ladder are built from)
  adj_<x>                      the adjusted value of each exogenous series x of the
                               input set (the executor's robust_transform: diurnal ->
                               semantic transform -> rolling winsorisation, window
                               240; unobserved bars = the column median)
  <x>_avail, <x>_active        the availability indicator of every x and the
                               activity indicator of every mode-inflated x
  DOW_0..DOW_4, hour, is_overnight, is_open, is_close, is_opex, is_opex_week,
  is_quad_witch, is_rebalance_close, is_month_end, is_quarter_end, days_to_opex
                               the executor's calendar block (the columns the per-bar
                               designs carry), at the bar's own stamp
  tod_sin, tod_cos             the half-hour clock of the bar end: sin / cos of
                               2 pi slot / SLOTS_PER_DAY, slot = minutes / 30
Input sets: baseline = adj_RV + calendar + clock; live_feasible / all_features add
the adj_<x>, <x>_avail, <x>_active columns of that bucket (src.data.loading).
PRESCALE AND WINSORISATION: the winsorisation is inside the executor's columns
(adj_RV and every adj_<x> are rolling-winsorised by robust_transform; the impute
and indicators are the executor's), reused, not re-derived.  The executor's
prescale is a rolling robust scaling of the per-bar HAR DESIGN (each ladder column
over the previous 2000 per-bar rows, with the scale guards of imputed exogenous
ladders); the ladders are not inputs here, so it does not apply to P.  The network
standardises each step feature with the mean / sd of its training rows (below), as
the per-bar LSTM does after the prescale.  Nothing else is clipped or floored.

THE SEQUENCE (specs/causal_tune_lstm_intraday_jobs.py): for the session whose 16:00
bar sits at panel row e, the bars e - N .. e - 1 -- the bar ending 15:30 and the
N - 1 bars before it -- everything observable when the forecast is issued at 15:30
and nothing from the 15:30-16:00 bar.  The panel has gaps (holidays, early closes,
short early years: 32-48 bars a day), so "the last N bars" are the last N panel
rows; the clock and calendar inputs tell the network where each step lies.

SEQUENCE LENGTH N (a tuning axis, SEQ_LENS = (13, 48, 96)):
  13  on a full day the bars ending 09:30 .. 15:30: the regular session through
      15:30 plus the bar into the open -- the har_ma_1 / har_ma_5 horizon;
  48  one full day of the 48-slot grid: from the previous 16:00 bar (the previous
      session's target bar) through the overnight to 15:30 -- the har_ma_25 horizon;
  96  two days.  The ladder's slow rungs (har_ma_125 .. har_ma_3125, 2.6 to 65 days)
      are beyond every sequence: here only the window standardisation and the two-day
      path carry the level.  Longer sequences cost in proportion (an epoch at N = 96
      costs several times one at 13), which is why the grid stops at two days.
N is tuned (it enters the grid) rather than run as separate arms: separate arms
would triple the refits, the dominant cost.  Embargo: a validation sequence must
not read the fit block's last target bar; the number of 16:00 bars inside any
sequence of the longest N is at most MAX_SESSIONS_IN_SEQ (asserted <= EMBARGO over
the whole series), and at every tuning point the first bar of every validation
sequence is asserted to lie after the last fit row's 16:00 bar.

GRID (the per-bar LSTM's, read from its spec): seq_len N in SEQ_LENS x hidden in
HIDDEN x dropout in DROPOUTS x Adam learning rate in LEARNING_RATES = 24
configurations, all scored at every tuning point; BATCH, GRAD_CLIP read from the
per-bar spec; PATIENCE, MAX_EPOCHS, N_SEEDS its campaign values (20, 200, 5).

TUNING (the per-bar arms' scheme): every TUNE_PER = 250 rows of the one-bar series
(counted from the whole series' first forecast) the window [t - W, t) is split by
forward_window_split(t, W, VAL_TAIL = 125, EMBARGO = 25); every configuration x seed
is trained on the fit block with early stopping on the validation tail; the
configuration whose seed-averaged validation forecast has the lowest MSE on the
transformed target is held until the next tuning point (the rule of record); the
QLIKE-argmin configuration is walked forward too (QSEL = 1) with its own refits when
it differs, written as results_qsel_<seg>.csv.  TUNE_PER, VAL_TAIL, EMBARGO, SEED,
HORIZON, DATA_PATH are read from specs/causal_tune_linear.py.

REFITS: EVERY SESSION (REFIT_EVERY = 1): the configuration in force is refitted on
the sessions [t - W, t) -- each seed for its own early-stopped epoch count from the
tuning point -- and forecasts session t from its own sequence.  The forecast is the
mean of the N_SEEDS networks.  Standardisation: mean / sd per step feature over the
COVERED BARS of the training rows (the union of the bars their sequences read, each
bar once; the fit block at a tuning point, the window at a refit); the target is
standardised over the training rows' targets.

WINDOW MASK (env axis WINDOW_MASK, default 1 here -- this spec has no run before the
decision to reproduce; 0 for the mask gate): src.models.window_mask.window_keep on the
per-step columns over the training rows' sequences, flattened steps x rows (computed on
their covered bars: the same set of distinct bars, hence the same kept columns --
gated).  Cadence per fit, as the per-bar LSTM (rebuilt from its seed at every refit):
a tuning point computes one mask per N on its window [t - W, t)'s sequences at that N
(every candidate of that N uses it); every refit computes the mask of its own window
at the chosen N.  Recorded: kept_n per refit (both rules) and per (tuning point, N).

CAUSALITY: a tuning point writes for the workers only the window's targets and the
bars its sequences read (at the longest N), i.e. nothing after the last window row's
15:30 bar; a refit reads the window's targets and the bars up to its forecast row's
15:30 bar.  Gates: target x50 from a row on -> every earlier forecast and tuning
record bit-identical; every input of every bar from a session's 16:00 bar on x50
-> that session's 15:30 forecast (and every earlier one) bit-identical.

PARALLELISM: the (candidate x seed) fits of a tuning point, and the (row x seed x
rule) refits, run in a pool of N_WORKERS = SLURM_CPUS_PER_TASK spawned processes;
every fit is single-threaded and deterministic, so every number is independent of
the pool size.  TIME CHUNKS: run_executor's START / END / HALO seam (HALO = W; START
= W + the chunk's first whole-series OOS row, START = 0 with HALO = 0 for the whole
series).  A chunk may start anywhere: its first rows use the tuning point in force
(the last multiple of TUNE_PER at or before its start), recomputed from the
whole-series arrays -- or read from TUNE_CACHE, where STAGE=tune runs wrote it
(two-stage campaign: stage 1 = one task per tuning point, stage 2 = refit chunks);
a cached record carries a fingerprint (grid, settings, sha256 of the window's
targets and bars) that must match or the run stops.  A chunk owns (writes to its
tune tables) the tuning points inside its rows; experiments/reduce_lstm_intraday_chunks.py
merges the chunks of an arm.  Gate: chunks (incl. a mid-period start, with and
without the cache) = the unchunked walk, bit for bit.

PERSISTED (under $HPC_RESULT_DIR/causal_tune_lstm_intraday/lstm_intraday/<bucket>/):
  results_<seg>.csv            the executor's results table (MSE rule = the arm of record)
  results_qsel_<seg>.csv       the QLIKE-rule path, same columns
  lstm_intraday_<seg>.npz      stamps, pred_adj, per-seed forecasts, the owned tuning
                               records, kept_n / kept per refit, step names, meta
  tune_trace_<seg>.csv         one row per (owned tuning point, rule): chosen config,
                               epochs, losses, kept columns per N
  tune_candidates_<seg>.csv    one row per (owned tuning point, candidate)
  refit_trace_<seg>.csv        one row per forecast: the configuration in force (both
                               rules), epochs per seed, kept_n, seconds
  grid_<seg>.json              the grid and every training constant
All row indices are whole-series OOS rows (row 0 = the first forecast of the series).

ENV AXES (HPC_KW_<name> or <name>): EXOG_BUCKET (live_feasible | all_features |
baseline), SEGMENT (bar1600 only), TRAIN_WIN (2000), LAG_SCOPE (global), START / END
/ HALO, QSEL (0 | 1, default 1), WINDOW_MASK (0 | 1, default 1), STAGE (all | tune),
TUNE_CACHE (directory; empty = none), REQUIRE_CACHE (1 = a needed tuning point missing
from the cache is an error); for the gates only: N_SEEDS, MAX_EPOCHS, PATIENCE,
DATA_PATH.
"""
# ruff: noqa: E402 -- the path bootstrap must run before the repo imports (as the other specs)

# %%
import sys
from pathlib import Path

import os

_START = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_ROOT = _START
while not (_ROOT / "src").is_dir() and _ROOT != _ROOT.parent:
    _ROOT = _ROOT.parent
if (_ROOT / "src").is_dir():
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    os.chdir(_ROOT)

# %%
import ast
import gc
import hashlib
import itertools
import json
import multiprocessing
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.models.reclasso_har import forward_window_split
from src.models.window_mask import window_keep

if str(_ROOT / "specs") not in sys.path:
    sys.path.insert(0, str(_ROOT / "specs"))
import causal_tune_lstm_intraday_jobs as J  # noqa: E402 -- the worker-side jobs

LINEAR_SPEC = _ROOT / "specs" / "causal_tune_linear.py"
LSTM_SPEC = _ROOT / "specs" / "causal_tune_lstm.py"


def load_constants(path: Path, names: tuple[str, ...]) -> dict:
    """The first top-level literal assignment of each name in the file at path."""
    out: dict = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in names and tgt.id not in out:
                    out[tgt.id] = ast.literal_eval(node.value)
    missing = set(names) - set(out)
    assert not missing, f"{path.name} no longer assigns {sorted(missing)}"
    return out


def env_defaults(path: Path, names: tuple[str, ...]) -> dict:
    """For NAME = int(_env("NAME", "<default>")) in the file at path: {NAME: int(default)}."""
    out: dict = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if name not in names or name in out:
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Call) and getattr(sub.func, "id", "") == "_env":
                out[name] = int(ast.literal_eval(sub.args[1]))
                break
    missing = set(names) - set(out)
    assert not missing, f"{path.name}: no _env default for {sorted(missing)}"
    return out


_LIN = load_constants(
    LINEAR_SPEC, ("TUNE_PER", "VAL_TAIL", "EMBARGO", "SEED", "HORIZON", "DATA_PATH")
)
TUNE_PER: int = _LIN["TUNE_PER"]  # 250 rows between tunings (the per-bar arms' cadence)
VAL_TAIL: int = _LIN["VAL_TAIL"]  # 125-row validation tail inside the window
EMBARGO: int = _LIN["EMBARGO"]  # 25-row gap between fit block and validation tail
SEED: int = _LIN["SEED"]
HORIZON: int = _LIN["HORIZON"]
assert HORIZON == 1, (
    f"one-step forecasts only (the sequence ends at the bar before the target): {HORIZON}"
)
_NN = load_constants(
    LSTM_SPEC, ("HIDDEN", "DROPOUTS", "LEARNING_RATES", "BATCH", "GRAD_CLIP")
)
_NN_ENV = env_defaults(LSTM_SPEC, ("PATIENCE", "MAX_EPOCHS", "N_SEEDS"))
REFIT_EVERY = (
    1  # every session (user decision 2026-09-29): the per-bar linear arms' cadence
)


def _env(name: str, default: str) -> str:
    return os.environ.get(f"HPC_KW_{name}", os.environ.get(name, default))


DATA_PATH: str = _env("DATA_PATH", _LIN["DATA_PATH"])
MODEL = "lstm_intraday"
SPEC_DIR = "causal_tune_lstm_intraday"
EXOG_BUCKET = _env("EXOG_BUCKET", "live_feasible")
SEGMENT = _env("SEGMENT", "bar1600")
if SEGMENT != "bar1600":
    raise SystemExit(
        f"the intraday LSTM forecasts the 15:30-16:00 bar only: SEGMENT {SEGMENT!r}"
    )
LAG_SCOPE = _env("LAG_SCOPE", "global")
TRAIN_WIN = int(_env("TRAIN_WIN", "2000"))
START = int(_env("START", "0"))
END = int(_env("END", "-1"))
HALO = int(_env("HALO", "0"))
QSEL = _env("QSEL", "1") == "1"
_WM = _env("WINDOW_MASK", "1")
if _WM not in ("0", "1"):
    raise SystemExit(f"WINDOW_MASK must be 0 or 1, got {_WM!r}")
WINDOW_MASK = _WM == "1"
STAGE = _env("STAGE", "all")
if STAGE not in ("all", "tune"):
    raise SystemExit(f"STAGE must be all or tune, got {STAGE!r}")
TUNE_CACHE = _env("TUNE_CACHE", "")
REQUIRE_CACHE = _env("REQUIRE_CACHE", "0") == "1"
N_WORKERS: int = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

# ---------------------------------------------------------------- grid (named choices)
SEQ_LENS = (13, 48, 96)  # bars; see the module note
SLOTS_PER_DAY = 48  # the panel's half-hour grid (the clock encoding's period)
HIDDEN: tuple = _NN["HIDDEN"]  # (16, 64)
DROPOUTS: tuple = _NN["DROPOUTS"]  # (0.0, 0.2)
LEARNING_RATES: tuple = _NN["LEARNING_RATES"]  # (1e-3, 1e-2)
BATCH: int = _NN["BATCH"]  # 128
GRAD_CLIP: float = _NN["GRAD_CLIP"]  # 1.0
PATIENCE = int(_env("PATIENCE", str(_NN_ENV["PATIENCE"])))  # 20
MAX_EPOCHS = int(_env("MAX_EPOCHS", str(_NN_ENV["MAX_EPOCHS"])))  # 200
N_SEEDS = int(_env("N_SEEDS", str(_NN_ENV["N_SEEDS"])))  # 5
SEEDS = tuple(SEED + j for j in range(N_SEEDS))
N_MAX = max(SEQ_LENS)

AXES: dict[str, list] = {
    "seq_len": list(SEQ_LENS),
    "hidden": list(HIDDEN),
    "dropout": list(DROPOUTS),
    "lr": list(LEARNING_RATES),
}
CANDIDATES: list[dict] = [
    dict(zip(AXES, combo)) for combo in itertools.product(*AXES.values())
]
for _a, _vals in AXES.items():
    assert list(_vals) == sorted(_vals), f"grid axis {_a} is not ascending: {_vals}"
print(
    f"lstm_intraday: bucket={EXOG_BUCKET} segment={SEGMENT} tw={TRAIN_WIN} grid {len(CANDIDATES)} "
    f"configs x {N_SEEDS} seeds, N in {SEQ_LENS}, tune every {TUNE_PER} (val {VAL_TAIL}, embargo "
    f"{EMBARGO}), refit every {REFIT_EVERY}, epochs <= {MAX_EPOCHS} (patience {PATIENCE}), batch "
    f"{BATCH}, clip {GRAD_CLIP}, workers {N_WORKERS}, qsel={QSEL}, window_mask={int(WINDOW_MASK)}, "
    f"stage={STAGE}, cache={TUNE_CACHE or '-'}, slice=({START},{END},{HALO}), torch {J.torch.__version__}",
    flush=True,
)


def grid_doc() -> dict:
    return {
        "model": MODEL,
        "axes": AXES,
        "candidates": CANDIDATES,
        "seeds": list(SEEDS),
        "batch": BATCH,
        "grad_clip": GRAD_CLIP,
        "patience": PATIENCE,
        "max_epochs": MAX_EPOCHS,
        "tune_per": TUNE_PER,
        "val_tail": VAL_TAIL,
        "embargo": EMBARGO,
        "refit_every": REFIT_EVERY,
        "slots_per_day": SLOTS_PER_DAY,
        "torch": J.torch.__version__,
    }


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def executor_kwargs(fit_predict, output_file: str) -> dict:
    """The run_executor keyword arguments of the per-bar arms, argument for argument
    (the identity gate compares them with the linear spec's own)."""
    from src.data.loading import get_bucket

    return dict(
        method_name=f"lstm_intraday_tuned_{EXOG_BUCKET}",
        fit_predict=fit_predict,
        hyperparams={},
        data_path=DATA_PATH,
        output_file=output_file,
        horizon=HORIZON,
        train_window=TRAIN_WIN,
        start=START,
        end=END,
        halo=HALO,
        exog_cols=get_bucket(EXOG_BUCKET),
        segment=SEGMENT,
        lag_scope=LAG_SCOPE,
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


# %% ---------------------------------------------------------------- the data
def prepare(kw: dict | None = None) -> dict:
    """The executor's transformed frame -> the step matrix P and the per-bar series.

    Returns {P (bars x steps, float64), step_names, pos (panel row of every series row's
    16:00 bar, after the burn-in drop), y (the series' targets = P[pos, 0]), stamps,
    seg_df + design + train_win (for _backtest_and_save), kw}."""
    from src.backtest.executor import _build_har_and_calendar, load_and_transform
    from src.backtest.segmentation import (
        SEGMENT_DEFINITIONS,
        compute_segment_train_window,
        slice_to_segment,
    )
    from src.features.extractors.har import resolve_har_lags

    kw = kw or executor_kwargs(None, "")
    assert kw["lag_scope"] == "global" and kw["segment"] == SEGMENT, kw
    a = time.time()
    df, adj_exog = load_and_transform(  # run_executor's call, argument for argument
        kw["data_path"],
        kw["exog_cols"],
        target_use_diurnal=kw["target_use_diurnal"],
        target_winsor_window=kw["target_winsor_window"],
        dropna_with_exog=kw["dropna_with_exog"],
        overnight_fill=kw["overnight_fill"],
        impute_indicate=kw["impute_indicate"],
        diurnal_mode=kw["diurnal_mode"],
    )
    # _iter_TOD_segment's global-lag build, with the target's ladder only (the exogenous
    # ladders are not inputs here; har_ma_* and the calendar block are the same columns)
    df, design = _build_har_and_calendar(
        df, [], kw["add_calendar"], kw["har_lags"], session_edge=False
    )
    cal = [c for c in design if not c.startswith("har_ma_")]
    minutes = (df["t"].dt.hour * 60 + df["t"].dt.minute).to_numpy()
    ang = 2.0 * np.pi * (minutes / 30.0) / SLOTS_PER_DAY
    df["tod_sin"], df["tod_cos"] = np.sin(ang), np.cos(ang)
    step_names = ["adj_RV", *adj_exog, *cal, "tod_sin", "tod_cos"]
    assert len(set(step_names)) == len(step_names), step_names
    P = np.ascontiguousarray(df[step_names].to_numpy(dtype=np.float64))
    lo_m, hi_m = SEGMENT_DEFINITIONS[SEGMENT]
    seg_pos = np.flatnonzero((minutes >= lo_m) & (minutes <= hi_m))
    seg_df = slice_to_segment(df, SEGMENT)
    assert np.array_equal(seg_df["t"].to_numpy(), df["t"].to_numpy()[seg_pos]), (
        "segment rows"
    )
    stamps_all = df["t"].to_numpy()
    del df
    gc.collect()
    max_lag = resolve_har_lags()[
        -1
    ]  # _backtest_and_save's burn-in drop (arm-invariant)
    pos = seg_pos[max_lag:]
    train_win = compute_segment_train_window(seg_df["t"], TRAIN_WIN)
    assert train_win == TRAIN_WIN, (train_win, TRAIN_WIN)  # one row per session
    assert pos[0] - N_MAX >= 0
    used = np.zeros(len(P), dtype=bool)  # every bar a sequence of the series can read
    for e in pos:
        used[e - N_MAX : e] = True
    assert np.isfinite(P[used]).all() and np.isfinite(P[pos, 0]).all(), (
        "non-finite step input"
    )
    y = P[pos, 0].copy()
    # sessions inside a sequence of the longest N: the 16:00 bars it reads + its own
    inside = np.searchsorted(pos, pos, side="left") - np.searchsorted(
        pos, pos - N_MAX, side="left"
    )
    print(
        f"prepare: {len(P):,} bars x {len(step_names)} step features ({len(adj_exog)} exogenous "
        f"columns, {len(cal)} calendar, 2 clock), {len(pos)} series rows after the {max_lag}-row "
        f"burn-in, {time.time() - a:.0f}s; 16:00 bars inside an N={N_MAX} sequence <= {int(inside.max())}",
        flush=True,
    )
    return dict(
        P=P,
        step_names=step_names,
        pos=pos,
        y=y,
        stamps=pd.Series(stamps_all[pos]),
        bar_stamps=stamps_all,
        seg_df=seg_df,
        design=design,
        train_win=train_win,
        kw=kw,
        max_sessions_in_seq=int(inside.max()),
    )


def sessions_in_sequence(pos: np.ndarray, N: int) -> np.ndarray:
    """Per series row: the number of earlier 16:00 bars its N-bar sequence reads."""
    return np.searchsorted(pos, pos, side="left") - np.searchsorted(
        pos, pos - N, side="left"
    )


# %% ---------------------------------------------------------------- tuning
def _fingerprint(
    bucket: str, W: int, R: int, y_w: np.ndarray, P_w: np.ndarray, ends_w
) -> dict:
    return {
        "grid": {k: v for k, v in grid_doc().items() if k != "torch"},
        "torch": J.torch.__version__,
        "bucket": bucket,
        "segment": SEGMENT,
        "W": W,
        "R": R,
        "window_mask": int(WINDOW_MASK),
        "sha_y": sha(y_w),
        "sha_P": sha(P_w),
        "sha_ends": sha(np.asarray(ends_w, dtype=np.int64)),
    }


REC_ARRAYS = ("val_mse", "val_qlike", "seed_val_mse", "epochs", "fit_sec", "split")


def cache_path(R: int) -> Path:
    return Path(TUNE_CACHE) / EXOG_BUCKET / f"tune_R{R:05d}.npz"


def save_rec(rec: dict, fp: dict) -> None:
    p = cache_path(rec["R"])
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name("tmp_" + p.name)
    np.savez(
        tmp,
        **{k: np.asarray(rec[k]) for k in REC_ARRAYS},  # type: ignore[arg-type]
        keep_mask=np.asarray(rec["keep_mask"], dtype=np.uint8),
        scalars=json.dumps(
            {k: rec[k] for k in ("R", "pick_mse", "pick_qlike", "tune_sec")}
        ),
        fingerprint=json.dumps(fp, sort_keys=True),
    )
    os.replace(tmp, p)


def load_rec(R: int, fp: dict) -> dict | None:
    p = cache_path(R)
    if not TUNE_CACHE or not p.is_file():
        return None
    with np.load(p, allow_pickle=False) as z:
        got = json.loads(str(z["fingerprint"]))
        if got != json.loads(json.dumps(fp, sort_keys=True)):
            diff = sorted(k for k in set(got) | set(fp) if got.get(k) != fp.get(k))
            raise SystemExit(
                f"tuning cache {p} was written for other settings/data: {diff}"
            )
        rec = {k: z[k] for k in REC_ARRAYS} | json.loads(str(z["scalars"]))
        rec["keep_mask"] = z["keep_mask"].astype(bool)
    rec["split"] = tuple(int(v) for v in rec["split"])
    rec["from_cache"] = True
    return rec


def tune(P, pos, y, W: int, R: int, run, tmp: str) -> dict:
    """Score every candidate x seed on the forward split of the window before the forecast
    row of whole-series OOS row R (series row t = W + R).

    Only the window's targets y[R : W + R] and the bars its sequences read at the longest
    N (panel rows pos[R] - N_MAX .. pos[W + R - 1] - 1: nothing after the last window
    row's 15:30 bar) are written for the workers."""
    t = W + R
    assert R >= 0 and t < len(pos), (R, t, len(pos))
    fit_lo, fit_hi, val_lo, val_hi = forward_window_split(t, W, VAL_TAIL, EMBARGO)
    assert fit_lo == t - W and val_hi == t, (fit_lo, val_hi, t, W)
    assert fit_hi + EMBARGO == val_lo and val_hi - val_lo == VAL_TAIL, (fit_hi, val_lo)
    assert fit_lo < fit_hi < val_lo < val_hi, (fit_lo, fit_hi, val_lo, val_hi)
    # embargo: no validation sequence (any N) reads the fit block's last 16:00 bar or earlier
    assert pos[val_lo] - N_MAX > pos[fit_hi - 1], (pos[val_lo], N_MAX, pos[fit_hi - 1])
    base, top = int(pos[R]) - N_MAX, int(pos[t - 1])
    P_w = P[base:top]
    ends_w = (pos[R:t] - base).astype(np.int64)
    y_w = np.asarray(y[R:t], dtype=np.float64)
    fp = _fingerprint(EXOG_BUCKET, W, R, y_w, P_w, ends_w)
    rec = load_rec(R, fp)
    if rec is not None:
        print(f"  tune @row {R}: read from {cache_path(R)}", flush=True)
        TRACE.append(rec)
        return rec
    if REQUIRE_CACHE:
        raise SystemExit(
            f"REQUIRE_CACHE=1 and no tuning record for row {R} in {TUNE_CACHE}"
        )
    xp, yp = os.path.join(tmp, f"win{R}_P.npy"), os.path.join(tmp, f"win{R}_y.npy")
    np.save(xp, P_w)
    np.save(yp, y_w)
    p = P_w.shape[1]
    keeps: dict[int, np.ndarray | None] = {}
    keep_mask = np.ones((len(SEQ_LENS), p), dtype=bool)
    for k, N in enumerate(SEQ_LENS):  # the window's mask at each N (all W window rows)
        if WINDOW_MASK:
            kN = window_keep(P_w[J.covered(ends_w, N, len(P_w))])
            keeps[N] = kN
            keep_mask[k] = np.isin(np.arange(p), kN)
        else:
            keeps[N] = None
    lo = t - W
    jobs = [
        dict(
            X=xp,
            y=yp,
            ends=ends_w,
            fit=(fit_lo - lo, fit_hi - lo),
            val=(val_lo - lo, val_hi - lo),
            keep=keeps[int(cfg["seq_len"])],
            cfg=cfg,
            seed=seed,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            batch=BATCH,
            grad_clip=GRAD_CLIP,
        )
        for cfg in CANDIDATES
        for seed in SEEDS
    ]
    a0 = time.time()
    got = list(run(J.fit_candidate, jobs))
    yv = np.asarray(y[val_lo:val_hi], dtype=np.float64)
    nc, ns = len(CANDIDATES), len(SEEDS)
    vp = np.array([g["val_pred"] for g in got]).reshape(nc, ns, VAL_TAIL)
    epochs = np.array([g["epochs"] for g in got], dtype=np.int64).reshape(nc, ns)
    secs = np.array([g["sec"] for g in got]).reshape(nc, ns)
    ens = vp.mean(axis=1)
    mse, qlk = np.full(nc, np.inf), np.full(nc, np.inf)
    for g in range(nc):
        if np.isfinite(ens[g]).all():
            mse[g], qlk[g] = J.val_losses(ens[g], yv)
    rec = dict(
        R=R,
        split=(
            fit_lo - W,
            fit_hi - W,
            val_lo - W,
            val_hi - W,
        ),  # whole-series OOS-row coordinates
        val_mse=mse,
        val_qlike=qlk,
        seed_val_mse=((vp - yv) ** 2).mean(axis=2),
        epochs=epochs,
        fit_sec=secs,
        pick_mse=int(np.argmin(mse)),  # ties -> the lower index
        pick_qlike=int(np.argmin(qlk)),
        tune_sec=time.time() - a0,
        keep_mask=keep_mask,
        from_cache=False,
    )
    assert np.isfinite(mse[rec["pick_mse"]]), (
        f"no candidate finished with a finite loss at row {R}"
    )
    if TUNE_CACHE:
        save_rec(rec, fp)
    TRACE.append(rec)
    return rec


# %% ---------------------------------------------------------------- the walk
TRACE: list[dict] = []
SIDE: dict = {}


def make_run(n_workers: int):
    if (
        n_workers > 1
    ):  # spawn: never fork a process that already holds torch / BLAS threads
        pool = ProcessPoolExecutor(
            n_workers, mp_context=multiprocessing.get_context("spawn")
        )
        return pool, pool.map
    return None, map


def walk(P, pos, y, W: int, i_lo: int, i_hi: int, n_workers: int | None = None) -> dict:
    """Forecast whole-series OOS rows [i_lo, i_hi) (OOS row i = series row W + i).

    P, pos, y are WHOLE-SERIES arrays (bars x steps, the series rows' 16:00-bar panel rows,
    the series targets), so a chunk that starts inside a tuning period recomputes (or reads)
    the tuning point in force exactly as the whole-series walk does."""
    assert 0 <= i_lo < i_hi <= len(pos) - W, (i_lo, i_hi, len(pos), W)
    TRACE.clear()
    R0 = (i_lo // TUNE_PER) * TUNE_PER
    tune_rows = [R0, *range(R0 + TUNE_PER, i_hi, TUNE_PER)]
    n = i_hi - i_lo
    ns = len(SEEDS)
    tmp = tempfile.mkdtemp(prefix="lstmi_", dir=os.environ.get("TMPDIR") or None)
    pool, run = make_run(N_WORKERS if n_workers is None else n_workers)
    t0 = time.time()
    try:
        recs = {R: tune(P, pos, y, W, R, run, tmp) for R in tune_rows}
        for R in tune_rows:
            r = recs[R]
            gm, gq = r["pick_mse"], r["pick_qlike"]
            print(
                f"  tune @row {R}: {'cached' if r['from_cache'] else '%.0fs' % r['tune_sec']}; mse pick "
                f"{gm} {CANDIDATES[gm]} epochs {r['epochs'][gm].tolist()} val_mse {r['val_mse'][gm]:.5f}; "
                f"qlike pick {gq} {CANDIDATES[gq]}; kept per N "
                f"{dict(zip(SEQ_LENS, r['keep_mask'].sum(axis=1).tolist()))}",
                flush=True,
            )
        # the chunk's arrays for the refits: series rows [i_lo, W + i_hi) and the bars they read
        base, top = int(pos[i_lo]) - N_MAX, int(pos[W + i_hi - 1])
        xp, ep, yp = (
            os.path.join(tmp, f)
            for f in ("chunk_P.npy", "chunk_ends.npy", "chunk_y.npy")
        )
        np.save(xp, P[base:top])
        np.save(ep, (pos[i_lo : W + i_hi] - base).astype(np.int64))
        np.save(
            yp, np.asarray(y[i_lo : W + i_hi - 1], dtype=np.float64)
        )  # no forecast row's own target
        jobs, keys = [], []
        for i in range(i_lo, i_hi):
            R = (i // TUNE_PER) * TUNE_PER
            r = recs[R]
            rules = [("mse", r["pick_mse"])]
            if QSEL and r["pick_qlike"] != r["pick_mse"]:
                rules.append(("qlike", r["pick_qlike"]))
            for rule, g in rules:
                for j, seed in enumerate(SEEDS):
                    jobs.append(
                        dict(
                            X=xp,
                            ends=ep,
                            y=yp,
                            t=W + i - i_lo,  # chunk row of the forecast
                            W=W,
                            k=1,
                            cfg=CANDIDATES[g],
                            seed=seed,
                            epochs=int(r["epochs"][g][j]),
                            mask=WINDOW_MASK,
                            batch=BATCH,
                            grad_clip=GRAD_CLIP,
                        )
                    )
                    keys.append((i, rule, j))
        got = list(run(J.refit_block, jobs))
    finally:
        if pool is not None:
            pool.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)
    p = P.shape[1]
    out_seeds = {rule: np.full((n, ns), np.nan) for rule in ("mse", "qlike")}
    kept = {rule: np.zeros((n, p), dtype=np.uint8) for rule in ("mse", "qlike")}
    kept_n = {rule: np.full(n, -1, dtype=np.int64) for rule in ("mse", "qlike")}
    sec = {rule: np.zeros(n) for rule in ("mse", "qlike")}
    for (i, rule, j), g in zip(keys, got):
        out_seeds[rule][i - i_lo, j] = g["preds"][0]
        sec[rule][i - i_lo] += g["fit_sec"]
        kk = np.arange(p) if g["keep"] is None else g["keep"]
        row = np.zeros(p, dtype=np.uint8)
        row[kk] = 1
        if j == 0:
            kept[rule][i - i_lo], kept_n[rule][i - i_lo] = row, len(kk)
        else:  # one window, one mask for every seed
            assert np.array_equal(kept[rule][i - i_lo], row), (
                f"seeds disagree on the mask at row {i}"
            )
    q_own = kept_n["qlike"] >= 0  # rows whose QLIKE pick differs (refitted separately)
    for rule in ("qlike",):
        out_seeds[rule][~q_own] = out_seeds["mse"][~q_own]
        kept[rule][~q_own], kept_n[rule][~q_own] = (
            kept["mse"][~q_own],
            kept_n["mse"][~q_own],
        )
    preds, preds_q = out_seeds["mse"].mean(axis=1), out_seeds["qlike"].mean(axis=1)
    assert np.isfinite(preds).all() and np.isfinite(preds_q).all(), (
        "non-finite forecast"
    )
    in_force = np.array(
        [(i // TUNE_PER) * TUNE_PER for i in range(i_lo, i_hi)], dtype=np.int64
    )
    return dict(
        preds=preds,
        preds_q=preds_q,
        preds_seeds=out_seeds["mse"],
        preds_q_seeds=out_seeds["qlike"],
        kept=kept["mse"],
        kept_n=kept_n["mse"],
        kept_q=kept["qlike"],
        kept_n_q=kept_n["qlike"],
        q_refit=q_own,
        fit_sec=sec["mse"],
        qsel_sec=sec["qlike"],
        in_force=in_force,
        recs=recs,
        i_lo=i_lo,
        i_hi=i_hi,
        wall_sec=time.time() - t0,
    )


# %% ---------------------------------------------------------------- tables
def trace_tables(
    out: dict, dates: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(tune_trace, tune_candidates, refit_trace); dates = the emitted rows' stamps.
    A chunk owns the tuning points inside its rows (the reducer concatenates owners)."""
    keys = {
        "model": MODEL,
        "bucket": EXOG_BUCKET,
        "segment": SEGMENT,
        "train_win": TRAIN_WIN,
    }
    trace_rows, cand_rows, refit_rows = [], [], []
    for R, rec in out["recs"].items():
        if not out["i_lo"] <= R < out["i_hi"]:
            continue  # in force here, owned by the chunk that holds row R
        fit_lo, fit_hi, val_lo, val_hi = rec["split"]
        base = (
            keys
            | {
                "tune_row": R,
                "forecast_date": str(dates.iloc[R - out["i_lo"]]),
                "from_cache": bool(rec["from_cache"]),
            }
            | {
                f"kept_n_N{N}": int(rec["keep_mask"][k].sum())
                for k, N in enumerate(SEQ_LENS)
            }
        )
        for g, cfg in enumerate(CANDIDATES):
            ep = rec["epochs"][g]
            cand_rows.append(
                base
                | {"cand_idx": g}
                | cfg
                | {
                    "epochs": ";".join(str(int(e)) for e in ep),
                    "epochs_mean": float(ep.mean()),
                    "epochs_cap_hits": int((ep >= MAX_EPOCHS).sum()),
                    "val_mse": float(rec["val_mse"][g]),
                    "val_qlike": float(rec["val_qlike"][g]),
                    "seed_val_mse_min": float(rec["seed_val_mse"][g].min()),
                    "seed_val_mse_max": float(rec["seed_val_mse"][g].max()),
                    "fit_sec": float(rec["fit_sec"][g].sum()),
                    "chosen_mse": g == int(rec["pick_mse"]),
                    "chosen_qlike": g == int(rec["pick_qlike"]),
                }
            )
        for rule in ("mse", "qlike"):
            g = int(rec[f"pick_{rule}"])
            ep = rec["epochs"][g]
            row = base | {
                "rule": rule,
                "fit_lo": fit_lo,
                "fit_hi": fit_hi,
                "val_lo": val_lo,
                "val_hi": val_hi,
                "cand_idx": g,
                "epochs": ";".join(str(int(e)) for e in ep),
                "epochs_mean": float(ep.mean()),
                "epochs_cap_hits": int((ep >= MAX_EPOCHS).sum()),
                "val_mse": float(rec["val_mse"][g]),
                "val_qlike": float(rec["val_qlike"][g]),
                "n_candidates": len(CANDIDATES),
                "tune_sec": float(rec["tune_sec"]),
            }
            for a, v in CANDIDATES[g].items():
                vals = AXES[a]
                row |= {
                    a: v,
                    f"{a}_edge": "lo"
                    if v == vals[0]
                    else ("hi" if v == vals[-1] else ""),
                }
            trace_rows.append(row)
    for k, i in enumerate(range(out["i_lo"], out["i_hi"])):
        R = int(out["in_force"][k])
        rec = out["recs"][R]
        row = keys | {"oos_row": i, "date": str(dates.iloc[k]), "tune_row": R}
        for rule, sfx in (("mse", ""), ("qlike", "_q")):
            g = int(rec[f"pick_{rule}"])
            row |= {f"cand_idx{sfx}": g} | {
                f"{a}{sfx}": v for a, v in CANDIDATES[g].items()
            }
            row |= {f"epochs{sfx}": ";".join(str(int(e)) for e in rec["epochs"][g])}
        row |= {
            "kept_n": int(out["kept_n"][k]),
            "kept_n_q": int(out["kept_n_q"][k]),
            "q_refit_separately": bool(out["q_refit"][k]),
            "fit_sec": float(out["fit_sec"][k]),
            "qsel_sec": float(out["qsel_sec"][k]),
            "seed_sd": float(out["preds_seeds"][k].std()),
        }
        refit_rows.append(row)
    return pd.DataFrame(trace_rows), pd.DataFrame(cand_rows), pd.DataFrame(refit_rows)


RESULTS_ROOT = os.path.join(os.environ.get("HPC_RESULT_DIR", "results"), SPEC_DIR)
OUT_DIR = os.path.join(RESULTS_ROOT, MODEL, EXOG_BUCKET)


def chunk_rows(n_series: int, W: int) -> tuple[int, int]:
    """(i_lo, i_hi): the whole-series OOS rows run_executor's START / END / HALO emit."""
    if START == 0:
        assert HALO == 0, f"START 0 with HALO {HALO}"
    else:
        assert HALO == W, (
            f"chunk halo must be the training window: HALO {HALO} != W {W}"
        )
    load_start = max(0, START - HALO)
    actual_end = n_series if END < 0 else END
    return load_start, actual_end - W


# %% ---- RUN ----
if __name__ == "__main__":
    from src.backtest.executor import _backtest_and_save

    t_run = time.time()
    C = prepare()
    W = C["train_win"]
    i_lo, i_hi = chunk_rows(len(C["pos"]), W)
    assert C["max_sessions_in_seq"] <= EMBARGO, (C["max_sessions_in_seq"], EMBARGO)

    if STAGE == "tune":  # stage 1: the tuning points inside [i_lo, i_hi) -> TUNE_CACHE
        assert TUNE_CACHE, "STAGE=tune needs TUNE_CACHE"
        rows = [R for R in range(i_lo, i_hi) if R % TUNE_PER == 0]
        assert rows, f"no tuning row in [{i_lo}, {i_hi})"
        tmp = tempfile.mkdtemp(
            prefix="lstmi_tune_", dir=os.environ.get("TMPDIR") or None
        )
        pool, run = make_run(N_WORKERS)
        try:
            for R in rows:
                rec = tune(C["P"], C["pos"], C["y"], W, R, run, tmp)
                print(
                    f"stage tune: row {R} {'(was cached)' if rec['from_cache'] else '%.0fs' % rec['tune_sec']} "
                    f"mse pick {CANDIDATES[int(rec['pick_mse'])]} qlike pick {CANDIDATES[int(rec['pick_qlike'])]} "
                    f"-> {cache_path(R)}",
                    flush=True,
                )
        finally:
            if pool is not None:
                pool.shutdown()
            shutil.rmtree(tmp, ignore_errors=True)
        sys.exit(0)

    def fit_predict_intraday(X_chunk, y_chunk, train_win_periods, hyperparams):
        """The executor's model slot: the chunk's rows are series rows [i_lo, W + i_hi);
        the network reads the whole-series step matrix, never X_chunk (the per-bar
        baseline design)."""
        assert int(train_win_periods) == W, (train_win_periods, W)
        assert len(X_chunk) == W + i_hi - i_lo, (len(X_chunk), W, i_lo, i_hi)
        # the executor's target is the step matrix's adj_RV at the 16:00 bars, bit for bit
        assert np.array_equal(
            np.asarray(y_chunk, dtype=np.float64), C["y"][i_lo : W + i_hi]
        ), "target"
        out = walk(C["P"], C["pos"], C["y"], W, i_lo, i_hi)
        SIDE.update(out)
        return out["preds"]

    out_csv = os.path.join(OUT_DIR, "results.csv")
    out_read = os.path.join(OUT_DIR, f"results_{SEGMENT}.csv")
    _backtest_and_save(  # run_executor's per-segment call (output file results_<seg>.csv)
        C["seg_df"],
        C["design"],
        fit_predict_intraday,
        {},
        W,
        HORIZON,
        START,
        END,
        HALO,
        out_read,
        C["kw"]["prescale"],
        C["kw"]["diurnal_mode"],
    )
    res = pd.read_csv(out_read, parse_dates=["date"])
    stamps = res["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    assert (
        stamps.to_numpy()
        == pd.Series(C["stamps"])
        .dt.strftime("%Y-%m-%d %H:%M:%S")
        .to_numpy()[W + i_lo : W + i_hi]
    ).all()
    ok = (res["true_adj"] > 0) & (res["true_raw"] > 0)
    baseline = np.where(ok, res["true_raw"] / res["true_adj"].where(ok) ** 2, np.nan)
    if QSEL:
        pq = SIDE["preds_q"]
        smear = float(np.mean((res["true_adj"].to_numpy(float) - pq) ** 2))
        qsel = res[["date", "horizon", "true_adj"]].copy()
        qsel["pred_adj"] = pq
        qsel["true_raw"] = res["true_raw"]
        qsel["pred_raw"] = (pq**2 + smear) * baseline
        qsel.to_csv(os.path.join(OUT_DIR, f"results_qsel_{SEGMENT}.csv"), index=False)
    trace, cands, refits = trace_tables(SIDE, stamps)
    trace.to_csv(os.path.join(OUT_DIR, f"tune_trace_{SEGMENT}.csv"), index=False)
    cands.to_csv(os.path.join(OUT_DIR, f"tune_candidates_{SEGMENT}.csv"), index=False)
    refits.to_csv(os.path.join(OUT_DIR, f"refit_trace_{SEGMENT}.csv"), index=False)
    with open(
        os.path.join(OUT_DIR, f"grid_{SEGMENT}.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(grid_doc(), fh, indent=1)
    versions = {}
    for mod in ("numpy", "pandas", "torch"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            versions[mod] = "absent"
    owned = [SIDE["recs"][R] for R in sorted(SIDE["recs"]) if i_lo <= R < i_hi]
    np.savez_compressed(
        os.path.join(OUT_DIR, f"lstm_intraday_{SEGMENT}.npz"),
        date=stamps.to_numpy().astype("U19"),
        pred_adj=res["pred_adj"].to_numpy(float),
        true_adj=res["true_adj"].to_numpy(float),
        true_raw=res["true_raw"].to_numpy(float),
        baseline=baseline,
        e2_adj=((res["true_adj"] - res["pred_adj"]) ** 2).to_numpy(float),
        pred_adj_seeds=SIDE["preds_seeds"],
        pred_adj_qsel=SIDE["preds_q"],
        pred_adj_qsel_seeds=SIDE["preds_q_seeds"],
        fit_sec=SIDE["fit_sec"],
        qsel_sec=SIDE["qsel_sec"],
        kept_n=SIDE["kept_n"],
        kept=SIDE["kept"],
        kept_n_qsel=SIDE["kept_n_q"],
        kept_qsel=SIDE["kept_q"],
        qsel_refit=SIDE["q_refit"],
        in_force=SIDE["in_force"],
        step_names=np.array(C["step_names"]),
        oos_offset=i_lo,
        tune_row=np.array([r["R"] for r in owned], dtype=np.int64),
        tune_split=np.array([r["split"] for r in owned], dtype=np.int64).reshape(
            len(owned), 4
        ),
        tune_sec=np.array([r["tune_sec"] for r in owned]),
        tune_kept=np.array([r["keep_mask"] for r in owned], dtype=np.uint8).reshape(
            len(owned), len(SEQ_LENS), len(C["step_names"])
        ),
        cand_val_mse=np.array([r["val_mse"] for r in owned]).reshape(
            len(owned), len(CANDIDATES)
        ),
        cand_val_qlike=np.array([r["val_qlike"] for r in owned]).reshape(
            len(owned), len(CANDIDATES)
        ),
        cand_seed_val_mse=np.array([r["seed_val_mse"] for r in owned]).reshape(
            len(owned), len(CANDIDATES), len(SEEDS)
        ),
        cand_epochs=np.array([r["epochs"] for r in owned]).reshape(
            len(owned), len(CANDIDATES), len(SEEDS)
        ),
        cand_fit_sec=np.array([r["fit_sec"] for r in owned]).reshape(
            len(owned), len(CANDIDATES), len(SEEDS)
        ),
        chosen_mse=np.array([r["pick_mse"] for r in owned], dtype=np.int64),
        chosen_qlike=np.array([r["pick_qlike"] for r in owned], dtype=np.int64),
        meta=json.dumps(
            {
                "model": MODEL,
                "bucket": EXOG_BUCKET,
                "segment": SEGMENT,
                "train_win_days": TRAIN_WIN,
                "threads": J.MODEL_THREADS,
                "workers": N_WORKERS,
                "oos_offset": i_lo,
                "slice": [START, END, HALO],
                "n_step_features": len(C["step_names"]),
                "max_sessions_in_seq": C["max_sessions_in_seq"],
                "versions": versions,
                "walk_sec": SIDE["wall_sec"],
                "run_sec": time.time() - t_run,
                "grid": grid_doc(),
                "qsel": QSEL,
                "window_mask": int(WINDOW_MASK),
                "tune_cache": TUNE_CACHE,
            }
        ),
    )
    e2 = (res["true_adj"] - res["pred_adj"]) ** 2
    print(
        f"wrote {out_read} ({len(res)} OOS rows {i_lo}..{i_hi - 1}, {res['date'].min()} .. {res['date'].max()}); "
        f"{len(owned)} owned tunings; {len(SIDE['fit_sec'])} refits ({SIDE['fit_sec'].mean():.1f} process-s each, "
        f"all seeds), {int(SIDE['q_refit'].sum())} separate qsel refits; kept {SIDE['kept_n'].min()}..{SIDE['kept_n'].max()} "
        f"of {len(C['step_names'])}; walk {SIDE['wall_sec']:.0f}s, run {time.time() - t_run:.0f}s; fit-space MSE {e2.mean():.5f}",
        flush=True,
    )
