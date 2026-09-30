"""Worker-side jobs of specs/causal_tune_lstm_intraday.py (importable by pool workers).

Two jobs, both pure functions of their arguments plus the arrays they name on
disk, so a process pool (spawn) can run them and the result does not depend on
which process ran them or how many there are:

  fit_candidate  one grid candidate x one seed on the fit block of a tuning
                 window: Adam on MSE with early stopping on the validation tail;
                 returns the epoch count at the best validation MSE and the
                 validation-tail forecasts at that epoch
  refit_block    one configuration x one seed refitted on the full trailing
                 window (the sessions [t - W, t)) for a fixed epoch count, and its
                 forecasts of the rows t .. t + k - 1

THE NETWORK, THE TRAINING LOOP AND THE STANDARDISATION ARE THE PER-BAR LSTM'S,
IMPORTED, NOT COPIED (specs/causal_tune_lstm_jobs.py: Net = one LSTM layer ->
final hidden state -> dropout -> linear head; train = Adam on MSE of the
standardised target, minibatches, gradient-norm clip, early stopping on the
validation MSE; standardise = column mean / sd over the training rows, sd 1 for a
constant column; val_losses = the tuned trees' validation MSE and QLIKE; every fit
single-threaded and deterministic).  What differs is only the input sequence:

  THE SEQUENCE of a session (a row of the one-bar series) is the N half-hour bars
  of the near-24-hour panel immediately BEFORE the row's own 16:00 bar -- panel
  rows e - N .. e - 1 where e is the panel position of the 16:00 bar -- i.e. the
  bars ending 15:30, 15:00, ... : everything observable when the forecast is issued
  at 15:30 and nothing from the 15:30-16:00 bar.  One step per bar; the step's
  features are the columns of the panel step matrix P the spec builds (the bar's
  adjusted target, the adjusted exogenous values, their availability / activity
  indicators, the calendar columns and the half-hour clock).

  THE TRAINING ROWS of a fit are the sessions of its window (the fit block at a
  tuning point, the whole window at a refit); a training row's sequence may reach
  bars before the window's first 16:00 bar -- they are observed history, as the
  per-bar design's HAR ladder reaches 3125 bars back.  The COVERED BARS of a set of
  rows are the union of the bars their sequences read (each bar once).
  Standardisation (mean / sd per step feature) and the window mask are computed on
  the covered bars of the training rows: for the mask this is exactly
  window_keep over the training rows' sequences flattened steps x rows (a column is
  constant, or a byte-copy of another, over the flattened rows iff it is over the
  distinct bars they contain; the gates check it), and for the moments it is the
  per-bar LSTM's convention of counting each design row once.

EVERY FIT IS SINGLE-THREADED (the per-bar jobs module sets torch to one thread and
deterministic algorithms when it is imported), so every number is independent of
the pool size.

WINDOW MASK (job key "keep" at a tuning point: the columns the spec computed on the
tuning window for the candidate's sequence length; job key "mask" at a refit: True =
compute keep = window_keep over the covered bars of the refit window's sequences
here).  Without either every number is the unmasked one.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "specs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

# the per-bar LSTM's worker module: importing it fixes torch to MODEL_THREADS = 1 thread and
# deterministic algorithms in this process
from causal_tune_lstm_jobs import (  # noqa: E402,F401 -- MODEL_THREADS / val_losses re-exported
    MODEL_THREADS,
    arr,
    sequences,
    standardise,
    train,
    val_losses,
)
from src.models.window_mask import window_keep  # noqa: E402


def covered(ends: np.ndarray, N: int, n: int) -> np.ndarray:
    """Sorted indices of the bars (of an array of n bars) read by the sequences that end
    just before each of ``ends``: bars e - N .. e - 1 for every e (each bar once)."""
    ends = np.asarray(ends, dtype=np.int64)
    assert ends.min() - N >= 0 and ends.max() <= n, (ends.min(), N, ends.max(), n)
    d = np.zeros(n + 1, dtype=np.int64)
    np.add.at(d, ends - N, 1)
    np.add.at(d, ends, -1)
    return np.flatnonzero(np.cumsum(d[:-1]) > 0)


def seqs(Z: np.ndarray, ends: np.ndarray, N: int) -> np.ndarray:
    """(len(ends), N, p) float32: bars e - N .. e - 1 of Z for every e (the per-bar
    sequences() with its end row = the bar before the 16:00 bar)."""
    return sequences(Z, np.asarray(ends, dtype=np.int64) - 1, N)


def fit_candidate(job: dict) -> dict:
    """job: X = path of the tuning WINDOW's bars (the bars the window's sequences read
    at the longest N, nothing after the last window row's 15:30 bar), ends = the panel
    positions of the W window rows' 16:00 bars relative to that array, y = the window's
    targets, fit / val = row ranges of the window, keep (None or column indices),
    cfg (seq_len, hidden, dropout, lr), seed, max_epochs, patience, batch, grad_clip.
    Returns the epoch count at the best validation MSE and the validation-tail
    forecasts (target units) there."""
    a = time.time()
    P = np.array(arr(job["X"]))
    ends, yw = np.asarray(job["ends"], dtype=np.int64), np.array(arr(job["y"]))
    (f0, f1), (v0, v1) = job["fit"], job["val"]
    N = int(job["cfg"]["seq_len"])
    if job.get("keep") is not None:
        P = P[:, np.asarray(job["keep"], dtype=np.int64)]
    fe, ve = ends[f0:f1], ends[v0:v1]
    mu, sd = standardise(P, covered(fe, N, len(P)))  # type: ignore[arg-type]  # X[rows], rows = the fit rows' covered bars
    Z = (P - mu) / sd
    yf = yw[f0:f1]
    ymu, ysd = float(yf.mean()), float(yf.std())
    net, ep, pv = train(
        seqs(Z, fe, N),
        (yf - ymu) / ysd,
        job["cfg"],
        int(job["seed"]),
        int(job["max_epochs"]),
        int(job["batch"]),
        float(job["grad_clip"]),
        Xv=seqs(Z, ve, N),
        yv_std=(yw[v0:v1] - ymu) / ysd,
        patience=int(job["patience"]),
    )
    val_pred = pv * ysd + ymu if pv is not None else np.full(len(ve), np.nan)
    return {"epochs": int(ep), "val_pred": val_pred, "sec": time.time() - a}


def refit_block(job: dict) -> dict:
    """job: X / ends / y = paths of the chunk's bar array, the chunk rows' 16:00-bar
    positions in it, and the chunk rows' targets; t = the block's first row (chunk
    row index), W, k = rows in the block, cfg, seed, epochs, batch, grad_clip, mask
    (True = the window mask).  The configuration is fitted on the sessions
    [t - W, t) for exactly ``epochs`` epochs and forecasts rows t .. t + k - 1, each
    from its own sequence (the N bars before its 16:00 bar)."""
    a = time.time()
    t, W, k = int(job["t"]), int(job["W"]), int(job["k"])
    N = int(job["cfg"]["seq_len"])
    e = np.asarray(arr(job["ends"])[t - W : t + k], dtype=np.int64)
    lo, hi = (
        int(e[0]) - N,
        int(e[-1]),
    )  # first bar of the first window sequence .. the last
    P = np.array(
        arr(job["X"])[lo:hi]
    )  # forecast row's 15:30 bar; nothing from its 16:00 bar on
    yw = np.array(arr(job["y"])[t - W : t])  # targets of the window only
    e = e - lo
    we, fe = e[:W], e[W:]
    cov = covered(we, N, len(P))  # the window rows' covered bars (the training rows)
    keep = window_keep(P[cov]) if job.get("mask") else None
    if keep is not None:
        P = P[:, keep]
    mu, sd = standardise(P, cov)  # type: ignore[arg-type]  # X[rows] with an index array
    Z = (P - mu) / sd
    ymu, ysd = float(yw.mean()), float(yw.std())
    net, _, _ = train(
        seqs(Z, we, N),
        (yw - ymu) / ysd,
        job["cfg"],
        int(job["seed"]),
        int(job["epochs"]),
        int(job["batch"]),
        float(job["grad_clip"]),
    )
    net.eval()
    with torch.no_grad():
        p = net(torch.from_numpy(seqs(Z, fe, N))).numpy()
    return {
        "preds": p.astype(np.float64) * ysd + ymu,
        "fit_sec": time.time() - a,
        "keep": keep,
        "kept_n": P.shape[1],
    }
