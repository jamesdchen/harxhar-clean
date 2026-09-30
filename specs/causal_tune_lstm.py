"""causal_tune_lstm -- a small LSTM for the 15:30-16:00 bar, beside the per-bar linear and tree arms.

PURPOSE: a neural-network forecast in the comparison of 15:30 forecasts (the
professor's request, 2026-09-29), built with exactly the data, target, window
and tuning discipline of the per-bar linear arms (specs/causal_tune_linear.py)
and the causally tuned per-bar trees (specs/causal_tune_trees_tuned.py), so its
forecasts stack and score with the same tooling.

DESIGN (identical to the linear and tree arms, not re-derived): the same
src.backtest.executor.run_executor call -- load_and_transform (diurnal -> sqrt ->
winsorize target, winsor 240), HAR ladder (production powers of 5) + calendar,
impute_indicate=True with dropna_with_exog=False, prescale=True, SEGMENT = one
regular-hours bar with global lags, TRAIN_WIN = 2000 rows (one row per session),
horizon 1 -- and the executor's own results table results_<seg>.csv.  The target
is the executor's adjusted target (the square root of the diurnally adjusted
variance, winsorized), the scale every per-bar arm forecasts on.
experiments/gate_lstm.py asserts that the design matrix, target, stamps and
column names this spec hands the model equal (sha256) those of the linear
spec's own run_executor invocation for the same bucket and bar.

THE MODEL (specs/causal_tune_lstm_jobs.py): one LSTM layer over the last L rows
of the design (row t = the information set of the forecast issued for row t, so
the sequence is the last L sessions' feature vectors at this bar), final hidden
state -> dropout -> one linear output; inputs and target standardised with the
training rows' mean / sd; Adam on MSE, minibatches of BATCH, gradient norm
clipped at GRAD_CLIP.  The forecast is the mean of N_SEEDS networks that differ
only in their seed (initial weights, dropout masks, minibatch order).

GRID: seq_len L in SEQ_LENS x hidden in HIDDEN x dropout in DROPOUTS x Adam
learning rate in LEARNING_RATES = 16 configurations, all scored at every tuning
point (no subset).

TUNING (the tuned trees' scheme, copied): every TUNE_PER = 250 rows of the
one-bar series the CURRENT training window [t - W, t) is split by
src.models.reclasso_har.forward_window_split(t, W, VAL_TAIL = 125, EMBARGO = 25):
fit block [t - W, t - 150), 25-row embargo, validation tail [t - 125, t).  Every
configuration x seed is trained on the fit block with early stopping on the
validation tail (validation MSE, patience PATIENCE epochs, cap MAX_EPOCHS); the
configuration whose seed-averaged validation forecast has the lowest MSE on the
transformed target is held until the next tuning point (the linear and tree
arms' criterion).  Between tuning points the chosen configuration is refitted
every REFIT_EVERY = 10 rows on the full window [t - W, t), each seed for its own
early-stopped epoch count, and forecasts its block of rows.  TUNE_PER, VAL_TAIL,
EMBARGO, SEED, HORIZON and DATA_PATH are READ from specs/causal_tune_linear.py,
REFIT_EVERY's default from specs/causal_tune_trees.py (ast literals), so the arms
share one copy.  REFIT CADENCE (env axis REFIT_EVERY, 2026-09-29 evening, user
decision: the LSTM refits every session, as the trees and the linear arms do):
default = that literal (10), so every run before the decision reproduces; the
campaign sets REFIT_EVERY=1.  TUNE_PER % REFIT_EVERY == 0 (asserted).  max(SEQ_LENS) - 1 <= EMBARGO (asserted): a validation sequence shares
no row with the fit block.

SECOND RULE, RECORDED: every candidate's validation QLIKE (the tuned trees'
val_losses: the executor's Duan back-transform on the validation tail) is kept
beside its MSE and the QLIKE-argmin configuration is walked forward too (QSEL=1,
default), with its own refits when it differs from the MSE choice; written as
results_qsel_<seg>.csv.

CAUSALITY: the tuning step writes only the window X[t - W : t], y[t - W : t]
for the workers and asserts that fit block, embargo and validation tail tile it
exactly; a refit reads the window's targets y[t - W : t] and the design rows up
to its block's last forecast row (each forecast uses its own last L rows).

PARALLELISM: the (candidate x seed) fits of a tuning point, and then the (block x
seed) refits of its period, run in a pool of N_WORKERS = SLURM_CPUS_PER_TASK
spawned processes; every fit is single-threaded and deterministic, so every
number is independent of the pool size.  TIME CHUNKS: run_executor's START /
END / HALO seam with HALO = W and START - W a multiple of TUNE_PER (asserted),
so a chunk's tuning and refit rows are the whole series' and its first tuning
window is exactly its halo; the seeds do not depend on the chunk.
experiments/reduce_lstm_chunks.py merges the chunks of an arm.

PERSISTED (under $HPC_RESULT_DIR/causal_tune_lstm/lstm/<bucket>/, the tree
layout with this spec's directory):
  results_<seg>.csv          the executor's results table (MSE rule = the arm of record)
  results_qsel_<seg>.csv     the QLIKE-rule path, same columns
  lstm_<seg>.npz             stamps, pred_adj, per-seed forecasts, the tuning
                             record (every candidate's validation MSE / QLIKE /
                             epochs per seed / seconds), refit rows and seconds
  tune_trace_<seg>.csv       one row per (tuning point, rule): chosen config, epochs, losses
  tune_candidates_<seg>.csv  one row per (tuning point, candidate)
  grid_<seg>.json            the grid and every training constant

WINDOW MASK (env axis WINDOW_MASK, default 0 = every run before 2026-09-29
evening, reproduced bit for bit; user decision 2026-09-29, option 3: every
per-bar model class sees the same effective design, the linear arms'
identifiability rule, src.models.window_mask.window_keep: drop the columns
constant on the training window and the exact byte-copies of an earlier kept
column).  CADENCE: PER REFIT.  The network is rebuilt from its seed at every refit
(causal_tune_lstm_jobs.train constructs a new Net for every fit; nothing is
warm-started), so its input width can follow each refit's own window [t - W, t)
at no cost to the design; the per-tuning-period cadence of the linear arms would
be the choice only for a model carried from one refit to the next.  A tuning
point computes the mask of ITS window [t - W, t) and writes only those columns
for the workers (fit block, validation tail and every candidate x seed); each
refit (every seed, and the QLIKE rule's twin) computes the mask of its own window.
The epoch counts a refit uses were early-stopped at the tuning point, on the
tuning window's kept columns.  Persisted in addition (WINDOW_MASK=1 only): kept_n
/ kept (uint8, refits x p) at every refit, tune_kept_n / tune_kept at every tuning
point.

ENV AXES (HPC_KW_<name> or <name>): EXOG_BUCKET (live_feasible | all_features |
baseline | ...), SEGMENT (default bar1600), TRAIN_WIN (default 2000), LAG_SCOPE
(default global), HAR_LAGS / HAR_BASE, START / END / HALO, QSEL (0 | 1, default
1), WINDOW_MASK (0 | 1, default 0), REFIT_EVERY (default: the trees spec's
literal, 10); for the local gates only: N_SEEDS,
MAX_EPOCHS, PATIENCE (defaults = the campaign's values below) and DATA_PATH
(default: the linear spec's).
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
import itertools
import json
import multiprocessing
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.features.extractors.har import resolve_har_lags
from src.models.reclasso_har import forward_window_split
from src.models.window_mask import window_keep

if str(_ROOT / "specs") not in sys.path:
    sys.path.insert(0, str(_ROOT / "specs"))
import causal_tune_lstm_jobs as J  # noqa: E402 -- the worker-side jobs

LINEAR_SPEC = _ROOT / "specs" / "causal_tune_linear.py"
TREE_SPEC = _ROOT / "specs" / "causal_tune_trees.py"


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


_LIN = load_constants(LINEAR_SPEC, ("TUNE_PER", "VAL_TAIL", "EMBARGO", "SEED", "HORIZON", "DATA_PATH"))
TUNE_PER: int = _LIN["TUNE_PER"]  # 250 rows between tunings (the linear and tree arms' cadence)
VAL_TAIL: int = _LIN["VAL_TAIL"]  # 125-row validation tail inside the window
EMBARGO: int = _LIN["EMBARGO"]  # 25-row gap between fit block and validation tail
SEED: int = _LIN["SEED"]
HORIZON: int = _LIN["HORIZON"]
REFIT_EVERY: int = load_constants(TREE_SPEC, ("REFIT_EVERY",))["REFIT_EVERY"]  # 10 sessions


def _env(name: str, default: str) -> str:
    return os.environ.get(f"HPC_KW_{name}", os.environ.get(name, default))


# 2026-09-29 evening (user decision): the refit cadence is an env axis; its default is the
# trees spec's literal above (the runs before the decision), the campaign sets 1 (every session).
REFIT_EVERY = int(_env("REFIT_EVERY", str(REFIT_EVERY)))
if REFIT_EVERY < 1:
    raise SystemExit(f"REFIT_EVERY must be >= 1, got {REFIT_EVERY}")


# The loader reads EVERY parquet of the data directory and then skips the ones
# without a bar key (the option-chain exports); a local gate may point DATA_PATH
# at a directory holding the bar-keyed files only, so it never reads the chain.
# Unset (the campaign): the linear spec's own "data".
DATA_PATH: str = _env("DATA_PATH", _LIN["DATA_PATH"])
MODEL = "lstm"
EXOG_BUCKET = _env("EXOG_BUCKET", "live_feasible")
SEGMENT = _env("SEGMENT", "bar1600")
if not SEGMENT.startswith("bar"):
    raise SystemExit(f"per-bar LSTM only: SEGMENT must be a one-bar segment, got {SEGMENT!r}")
LAG_SCOPE = _env("LAG_SCOPE", "global")
TRAIN_WIN = int(_env("TRAIN_WIN", "2000"))
START = int(_env("START", "0"))
END = int(_env("END", "-1"))
HALO = int(_env("HALO", "0"))
QSEL = _env("QSEL", "1") == "1"
_WINDOW_MASK_ENV = _env("WINDOW_MASK", "0")
if _WINDOW_MASK_ENV not in ("0", "1"):
    raise SystemExit(f"WINDOW_MASK must be 0 or 1, got {_WINDOW_MASK_ENV!r}")
WINDOW_MASK = _WINDOW_MASK_ENV == "1"  # per-refit window mask (module note)
_HAR_LAGS_ENV = _env("HAR_LAGS", "")
_HAR_BASE_ENV = _env("HAR_BASE", "")
if _HAR_LAGS_ENV:
    HAR_LAGS: list[int] | None = sorted(
        {int(v) for v in _HAR_LAGS_ENV.replace(";", ",").split(",") if v.strip()}
    )
elif _HAR_BASE_ENV:
    HAR_LAGS = resolve_har_lags(base=int(_HAR_BASE_ENV))
else:
    HAR_LAGS = None
# pool size = the task's cores; every fit inside a worker is single-threaded
# (J.MODEL_THREADS), so results do not depend on it
N_WORKERS: int = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

# ---------------------------------------------------------------- grid (named choices)
# Sequence length in sessions: one trading week and one trading month of the
# bar's history.  The design's HAR ladder already carries 1..3125-bar averages,
# so the recurrence only has to add short-run dynamics; the month is also the
# longest sequence that fits inside the 25-row embargo (asserted below).
SEQ_LENS = (5, 20)
# Hidden units: 16 is below the live-feasible design's own width, 64 is the
# common small default; the fit block holds ~1850 sequences, so larger nets
# would have more weights than training rows even at the live-feasible width.
HIDDEN = (16, 64)
# Dropout on the final hidden state: none, and the common light setting.
DROPOUTS = (0.0, 0.2)
# Adam step size: the library default and ten times it.
LEARNING_RATES = (1e-3, 1e-2)
# Minibatch: ~15 Adam steps per epoch on the fit block (~1850 sequences).
BATCH = 128
# Gradient-norm clip: the usual guard for recurrent nets, which matters for the
# 1e-2 step size (an exploding step would otherwise end a fit in NaN).
GRAD_CLIP = 1.0
# Early stopping on the validation tail: patience PATIENCE epochs (~300 Adam
# steps), cap MAX_EPOCHS = 10 x patience; cap hits are recorded (a budget edge).
PATIENCE = int(_env("PATIENCE", "20"))
MAX_EPOCHS = int(_env("MAX_EPOCHS", "200"))
# Seed ensemble: the forecast averages N_SEEDS networks that differ only in the
# seed, so a comparison with the linear and tree arms is not one draw of the
# initial weights.  Seeds SEED .. SEED + N_SEEDS - 1 at every fit (chunk-invariant).
N_SEEDS = int(_env("N_SEEDS", "5"))
SEEDS = tuple(SEED + j for j in range(N_SEEDS))

AXES: dict[str, list[float]] = {
    "seq_len": list(SEQ_LENS),
    "hidden": list(HIDDEN),
    "dropout": list(DROPOUTS),
    "lr": list(LEARNING_RATES),
}
CANDIDATES: list[dict] = [dict(zip(AXES, combo)) for combo in itertools.product(*AXES.values())]
assert max(SEQ_LENS) - 1 <= EMBARGO, (SEQ_LENS, EMBARGO)
assert TUNE_PER % REFIT_EVERY == 0, (TUNE_PER, REFIT_EVERY)  # tunings fall on refit rows
for _a, _vals in AXES.items():
    assert list(_vals) == sorted(_vals), f"grid axis {_a} is not ascending: {_vals}"
print(
    f"lstm: bucket={EXOG_BUCKET} segment={SEGMENT} lag_scope={LAG_SCOPE} tw={TRAIN_WIN} "
    f"grid {len(CANDIDATES)} configs x {N_SEEDS} seeds, tune every {TUNE_PER} rows (val tail "
    f"{VAL_TAIL}, embargo {EMBARGO}), refit every {REFIT_EVERY}, epochs <= {MAX_EPOCHS} "
    f"(patience {PATIENCE}), batch {BATCH}, clip {GRAD_CLIP}, workers {N_WORKERS}, qsel={QSEL}, "
    f"slice=({START},{END},{HALO}), torch {J.torch.__version__}, window_mask={int(WINDOW_MASK)}"
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
        "torch": J.torch.__version__,
    }


# %%
TRACE: list[dict] = []  # one entry per tuning point (evidence; reset per walk)
SIDE: dict = {}


def chunk_offset(W: int) -> int:
    """Whole-series OOS index of this run's first forecast row (the tuned trees' rule:
    a chunk replays exactly the training window as halo and starts on a tuning row)."""
    if START == 0:
        assert HALO == 0, f"START 0 with HALO {HALO}"
        return 0
    off = START - W
    assert HALO == W, f"chunk halo must be the training window: HALO {HALO} != W {W}"
    assert off >= 0 and off % TUNE_PER == 0, f"chunk start {START}: OOS offset {off} is not a tuning row"
    return off


def _train_kw() -> dict:
    return {"batch": BATCH, "grad_clip": GRAD_CLIP}


def tune(X, y, t: int, W: int, i: int, off: int, tmp: str, run) -> dict:
    """Score every candidate x seed on the forward split of the window before row t.

    Only the window X[t - W : t], y[t - W : t] -- rows strictly before the
    forecast row t -- is written for the workers; the split is asserted to tile
    it exactly (fit | embargo | validation tail)."""
    assert t - W >= 0 and t <= len(X), (t, W, len(X))
    fit_lo, fit_hi, val_lo, val_hi = forward_window_split(t, W, VAL_TAIL, EMBARGO)
    assert fit_lo == t - W and val_hi == t, (fit_lo, val_hi, t, W)
    assert fit_hi + EMBARGO == val_lo and val_hi - val_lo == VAL_TAIL, (fit_hi, val_lo)
    assert fit_lo < fit_hi < val_lo < val_hi, (fit_lo, fit_hi, val_lo, val_hi)
    xp, yp = os.path.join(tmp, f"win{i}_X.npy"), os.path.join(tmp, f"win{i}_y.npy")
    Xwin = X[t - W : t]
    keep = window_keep(Xwin) if WINDOW_MASK else None  # the window's mask (WINDOW_MASK)
    # the workers see the window (its kept columns under WINDOW_MASK) and nothing else
    np.save(xp, Xwin if keep is None else Xwin[:, keep])
    np.save(yp, y[t - W : t])
    lo = t - W
    jobs = [
        dict(
            X=xp,
            y=yp,
            fit=(fit_lo - lo, fit_hi - lo),
            val=(val_lo - lo, val_hi - lo),
            cfg=cfg,
            seed=seed,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            **_train_kw(),
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
    ens = vp.mean(axis=1)  # seed-averaged validation forecast per candidate
    mse, qlk = np.full(nc, np.inf), np.full(nc, np.inf)
    for g in range(nc):
        if np.isfinite(ens[g]).all():
            mse[g], qlk[g] = J.val_losses(ens[g], yv)
    seed_mse = ((vp - yv) ** 2).mean(axis=2)
    rec = dict(
        i=i,
        t=t,
        split=tuple(int(v + off) for v in (fit_lo, fit_hi, val_lo, val_hi)),  # whole-series rows
        val_mse=mse,
        val_qlike=qlk,
        seed_val_mse=seed_mse,
        epochs=epochs,
        fit_sec=secs,
        pick_mse=int(np.argmin(mse)),  # ties -> the lower index
        pick_qlike=int(np.argmin(qlk)),
        tune_sec=time.time() - a0,
        keep=keep,
    )
    assert np.isfinite(mse[rec["pick_mse"]]), f"no candidate finished with a finite loss at row {t}"
    TRACE.append(rec)
    return rec


def refit_period(xp, yp, W, blocks, n_test, cfg, epochs, run) -> tuple[list[dict], np.ndarray]:
    """The (block x seed) refits of one configuration; [(block out)], per-seed seconds."""
    jobs = [
        dict(
            X=xp,
            y=yp,
            t=W + i,
            W=W,
            k=min(REFIT_EVERY, n_test - i),
            cfg=cfg,
            seed=seed,
            epochs=int(epochs[j]),
            mask=WINDOW_MASK,
            **_train_kw(),
        )
        for i in blocks
        for j, seed in enumerate(SEEDS)
    ]
    got = list(run(J.refit_block, jobs))
    ns = len(SEEDS)
    out = []
    for b in range(len(blocks)):
        per_seed = np.array([got[b * ns + j]["preds"] for j in range(ns)])  # (seeds, k)
        keeps = [got[b * ns + j]["keep"] for j in range(ns)]  # one window: one mask
        assert all(
            (kk is None and keeps[0] is None) or np.array_equal(kk, keeps[0]) for kk in keeps
        ), f"seeds of block {blocks[b]} disagree on the window mask"
        out.append({"seeds": per_seed, "preds": per_seed.mean(axis=0), "keep": keeps[0]})
    secs = np.array([g["fit_sec"] for g in got]).reshape(len(blocks), ns)
    return out, secs


def fit_predict_lstm(X_chunk, y_chunk, train_win_periods, hyperparams):
    """Walk-forward on the one-bar series: tune every TUNE_PER rows (counted from
    the whole series' first forecast) on the current window, refit every
    REFIT_EVERY rows on the trailing W rows with the configuration in force."""
    X = np.ascontiguousarray(X_chunk, dtype=np.float64)
    y = np.ascontiguousarray(y_chunk, dtype=np.float64)
    assert np.isfinite(X).all() and np.isfinite(y).all(), "non-finite design or target"
    W = int(train_win_periods)
    n_test = len(X) - W
    p = X.shape[1]
    off = chunk_offset(W)
    preds = np.empty(n_test)
    preds_q = np.empty(n_test)
    preds_seeds = np.empty((n_test, len(SEEDS)))
    refit_row, fit_sec, qsel_sec = [], [], []
    kept_n, kept = [], []  # WINDOW_MASK: the columns of every refit's fit
    TRACE.clear()
    tmp = tempfile.mkdtemp(prefix="lstm_", dir=os.environ.get("TMPDIR") or None)
    xp, yp = os.path.join(tmp, "chunk_X.npy"), os.path.join(tmp, "chunk_y.npy")
    np.save(xp, X)
    np.save(yp, y)
    pool = None
    if N_WORKERS > 1:  # spawn: never fork a process that already holds torch / BLAS threads
        pool = ProcessPoolExecutor(N_WORKERS, mp_context=multiprocessing.get_context("spawn"))
    run = pool.map if pool is not None else map
    t0 = time.time()
    try:
        for i0 in range(0, n_test, TUNE_PER):  # off % TUNE_PER == 0: whole-series tuning rows
            rec = tune(X, y, W + i0, W, i0, off, tmp, run)
            gm, gq = rec["pick_mse"], rec["pick_qlike"]
            print(
                f"  tune @row {off + i0}: {len(CANDIDATES)}x{len(SEEDS)} fits in {rec['tune_sec']:.0f}s; "
                f"mse pick {gm} {CANDIDATES[gm]} epochs {rec['epochs'][gm].tolist()} "
                f"val_mse {rec['val_mse'][gm]:.5f}; qlike pick {gq} {CANDIDATES[gq]}",
                flush=True,
            )
            blocks = list(range(i0, min(i0 + TUNE_PER, n_test), REFIT_EVERY))
            outs, secs = refit_period(xp, yp, W, blocks, n_test, CANDIDATES[gm], rec["epochs"][gm], run)
            for i, out, s in zip(blocks, outs, secs):
                k = len(out["preds"])
                preds[i : i + k] = out["preds"]
                preds_seeds[i : i + k] = out["seeds"].T
                refit_row.append(i)
                fit_sec.append(float(s.sum()))
                if WINDOW_MASK:
                    kept_n.append(len(out["keep"]))
                    row = np.zeros(p, dtype=np.uint8)
                    row[out["keep"]] = 1
                    kept.append(row)
            if QSEL and gq != gm:
                outs_q, secs_q = refit_period(
                    xp, yp, W, blocks, n_test, CANDIDATES[gq], rec["epochs"][gq], run
                )
                for i, out in zip(blocks, outs_q):
                    preds_q[i : i + len(out["preds"])] = out["preds"]
                qsel_sec.extend(float(s.sum()) for s in secs_q)
            else:
                for i, out in zip(blocks, outs):
                    preds_q[i : i + len(out["preds"])] = out["preds"]
            print(f"  refits to row {off + blocks[-1]}: elapsed {time.time() - t0:.0f}s", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)
    assert np.isfinite(preds).all() and np.isfinite(preds_q).all(), "non-finite forecast"
    SIDE.update(
        feature_names=np.array(
            [str(c) for c in hyperparams.get("_feature_names", [f"f{j}" for j in range(p)])]
        ),
        refit_row=np.array(refit_row, dtype=np.int64),
        oos_offset=off,
        fit_sec=np.array(fit_sec),
        qsel_sec=np.array(qsel_sec),
        pred_q=preds_q,
        pred_seeds=preds_seeds,
        kept_n=np.array(kept_n, dtype=np.int64),
        kept=np.array(kept, dtype=np.uint8).reshape(len(kept), p),
        n_features=p,
        train_rows=W,
        wall_sec=time.time() - t0,
        trace=list(TRACE),
    )
    if WINDOW_MASK and kept_n:
        print(
            f"  window mask: kept {min(kept_n)}..{max(kept_n)} of {p} columns "
            f"(median {int(np.median(kept_n))}) over {len(kept_n)} refits; tuning points "
            f"{[len(r['keep']) for r in TRACE]}",
            flush=True,
        )
    return preds


def trace_tables(dates: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(tune_trace, tune_candidates) from SIDE['trace']; dates = the OOS stamps."""
    keys = {"model": MODEL, "bucket": EXOG_BUCKET, "segment": SEGMENT, "train_win": TRAIN_WIN}
    trace_rows, cand_rows = [], []
    for j, rec in enumerate(SIDE["trace"]):
        fit_lo, fit_hi, val_lo, val_hi = rec["split"]
        base = keys | {
            "tune_idx": j,
            "tune_row": rec["i"],  # row of this run's results table (+ chunk_offset = whole series)
            "chunk_offset": SIDE["oos_offset"],
            "forecast_date": str(dates.iloc[rec["i"]]),
        }
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
                    "chosen_mse": g == rec["pick_mse"],
                    "chosen_qlike": g == rec["pick_qlike"],
                }
            )
        for rule in ("mse", "qlike"):
            g = rec[f"pick_{rule}"]
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
                row |= {a: v, f"{a}_edge": "lo" if v == vals[0] else ("hi" if v == vals[-1] else "")}
            trace_rows.append(row)
    return pd.DataFrame(trace_rows), pd.DataFrame(cand_rows)


def executor_kwargs(fit_predict, output_file: str) -> dict:
    """The run_executor invocation: the linear / tree arms' call, argument for argument."""
    from src.data.loading import get_bucket

    return dict(
        method_name=f"lstm_tuned_{EXOG_BUCKET}",
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
        har_lags=HAR_LAGS,
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


RESULTS_ROOT = os.path.join(os.environ.get("HPC_RESULT_DIR", "results"), "causal_tune_lstm")
OUT_DIR = os.path.join(RESULTS_ROOT, MODEL, EXOG_BUCKET)

# %% ---- RUN ----
if __name__ == "__main__":
    from src.backtest.executor import run_executor

    out_csv = os.path.join(OUT_DIR, "results.csv")
    out_read = os.path.join(OUT_DIR, f"results_{SEGMENT}.csv")
    t_run = time.time()
    run_executor(**executor_kwargs(fit_predict_lstm, out_csv))
    res = pd.read_csv(out_read, parse_dates=["date"])
    n_oos = len(res)
    assert len(SIDE["refit_row"]) == -(-n_oos // REFIT_EVERY), (len(SIDE["refit_row"]), n_oos)
    assert len(SIDE["trace"]) == -(-n_oos // TUNE_PER), (len(SIDE["trace"]), n_oos)
    ok = (res["true_adj"] > 0) & (res["true_raw"] > 0)
    baseline = np.where(ok, res["true_raw"] / res["true_adj"].where(ok) ** 2, np.nan)

    if QSEL:  # the QLIKE-rule path, in the executor's own table format (its Duan pred_raw)
        pq = SIDE["pred_q"]
        smear = float(np.mean((res["true_adj"].to_numpy(float) - pq) ** 2))
        qsel = res[["date", "horizon", "true_adj"]].copy()
        qsel["pred_adj"] = pq
        qsel["true_raw"] = res["true_raw"]
        qsel["pred_raw"] = (pq**2 + smear) * baseline
        qsel.to_csv(os.path.join(OUT_DIR, f"results_qsel_{SEGMENT}.csv"), index=False)

    stamps = res["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    trace, cands = trace_tables(stamps)
    trace.to_csv(os.path.join(OUT_DIR, f"tune_trace_{SEGMENT}.csv"), index=False)
    cands.to_csv(os.path.join(OUT_DIR, f"tune_candidates_{SEGMENT}.csv"), index=False)
    with open(os.path.join(OUT_DIR, f"grid_{SEGMENT}.json"), "w", encoding="utf-8") as fh:
        json.dump(grid_doc(), fh, indent=1)

    versions = {}
    for mod in ("numpy", "pandas", "torch"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            versions[mod] = "absent"
    tr = SIDE["trace"]
    npz_path = os.path.join(OUT_DIR, f"lstm_{SEGMENT}.npz")
    np.savez_compressed(
        npz_path,
        date=stamps.to_numpy().astype("U19"),
        pred_adj=res["pred_adj"].to_numpy(float),
        true_adj=res["true_adj"].to_numpy(float),
        true_raw=res["true_raw"].to_numpy(float),
        baseline=baseline,
        e2_adj=((res["true_adj"] - res["pred_adj"]) ** 2).to_numpy(float),
        pred_adj_seeds=SIDE["pred_seeds"],
        pred_adj_qsel=SIDE["pred_q"] if QSEL else np.zeros(0),
        refit_row=SIDE["refit_row"],
        fit_sec=SIDE["fit_sec"],
        qsel_sec=SIDE["qsel_sec"],
        feature_names=SIDE["feature_names"],
        tune_row=np.array([r["i"] for r in tr], dtype=np.int64),  # + oos_offset = whole series
        oos_offset=SIDE["oos_offset"],
        tune_split=np.array([r["split"] for r in tr], dtype=np.int64),
        tune_sec=np.array([r["tune_sec"] for r in tr]),
        cand_val_mse=np.array([r["val_mse"] for r in tr]),
        cand_val_qlike=np.array([r["val_qlike"] for r in tr]),
        cand_seed_val_mse=np.array([r["seed_val_mse"] for r in tr]),
        cand_epochs=np.array([r["epochs"] for r in tr]),
        cand_fit_sec=np.array([r["fit_sec"] for r in tr]),
        chosen_mse=np.array([r["pick_mse"] for r in tr], dtype=np.int64),
        chosen_qlike=np.array([r["pick_qlike"] for r in tr], dtype=np.int64),
        # WINDOW_MASK=1 only (an unmasked run writes exactly the keys it always wrote)
        **(
            {
                "kept_n": SIDE["kept_n"],
                "kept": SIDE["kept"],
                "tune_kept_n": np.array([len(r["keep"]) for r in tr], dtype=np.int64),
                "tune_kept": np.array(
                    [np.isin(np.arange(SIDE["n_features"]), r["keep"]) for r in tr], dtype=np.uint8
                ).reshape(len(tr), SIDE["n_features"]),
            }
            if WINDOW_MASK
            else {}
        ),
        meta=json.dumps(
            {
                "model": MODEL,
                "bucket": EXOG_BUCKET,
                "segment": SEGMENT,
                "lag_scope": LAG_SCOPE,
                "train_win_days": TRAIN_WIN,
                "train_rows": SIDE["train_rows"],
                "threads": J.MODEL_THREADS,
                "workers": N_WORKERS,
                "oos_offset": SIDE["oos_offset"],
                "har_lags": HAR_LAGS,
                "slice": [START, END, HALO],
                "n_features": SIDE["n_features"],
                "versions": versions,
                "walk_sec": SIDE["wall_sec"],
                "run_sec": time.time() - t_run,
                "grid": grid_doc(),
                "qsel": QSEL,
                "window_mask": int(WINDOW_MASK),
            }
        ),
    )
    e2 = (res["true_adj"] - res["pred_adj"]) ** 2
    picks = [CANDIDATES[r["pick_mse"]] for r in tr]
    print(
        f"wrote {out_read} ({n_oos} OOS rows, {res['date'].min()} .. {res['date'].max()}) and {npz_path}; "
        f"{len(tr)} tunings ({np.mean([r['tune_sec'] for r in tr]):.0f}s each), "
        f"{len(SIDE['refit_row'])} refits ({SIDE['fit_sec'].mean():.1f} process-s each, all seeds), "
        f"{len(SIDE['qsel_sec'])} extra qsel refits, walk {SIDE['wall_sec']:.0f}s, "
        f"run {time.time() - t_run:.0f}s; picks {picks}; fit-space MSE {e2.mean():.5f}, p={SIDE['n_features']}"
    )
