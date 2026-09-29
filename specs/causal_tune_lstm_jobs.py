"""Worker-side jobs of specs/causal_tune_lstm.py (importable by pool workers).

Two jobs, both pure functions of their arguments plus the arrays they name on
disk, so a process pool (spawn) can run them and the result does not depend on
which process ran them or how many there are:

  fit_candidate  one grid candidate x one seed on the fit block of a tuning
                 window: Adam on MSE with early stopping on the validation tail;
                 returns the epoch count at the best validation MSE and the
                 validation-tail forecasts at that epoch
  refit_block    one configuration x one seed refitted on the full trailing
                 window for a fixed epoch count, and its forecasts of a block of
                 rows

THE NETWORK: one LSTM layer over the last L rows of the one-bar design (one row
per session; row t is the information set of the forecast issued for row t), the
final hidden state -> dropout -> one linear output.  Inputs are standardised
with the mean / sd of the TRAINING rows (the fit block at a tuning point, the
full window at a refit); a column constant on those rows is centred to zero (sd
taken as 1).  The target is standardised the same way and the output mapped
back.  A training sequence lies entirely inside the rows it is trained on (its
end row r and its L - 1 predecessors), so nothing outside the window enters.

EVERY FIT IS SINGLE-THREADED (MODEL_THREADS = 1) and deterministic
(torch.use_deterministic_algorithms; the seed fixes the initial weights, the
dropout masks and the minibatch order), so parallelism is across processes only
and every number is independent of the core count.

The validation losses (MSE on the transformed target, QLIKE of the executor's
Duan back-transform) are the tuned trees' own function
(specs/causal_tune_trees_tuned_jobs.val_losses), imported, not copied.
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

from causal_tune_trees_tuned_jobs import val_losses  # noqa: E402,F401 -- re-exported for the spec

MODEL_THREADS = 1  # see the module note: results independent of the core count
torch.set_num_threads(MODEL_THREADS)
try:
    torch.set_num_interop_threads(MODEL_THREADS)
except RuntimeError:  # already set in this process (a second import path)
    pass
torch.use_deterministic_algorithms(True)

_ARR: dict[str, np.ndarray] = {}


def arr(path: str) -> np.ndarray:
    """A float64 array saved by the spec, memory-mapped once per process."""
    if path not in _ARR:
        _ARR[path] = np.load(path, mmap_mode="r")
    return _ARR[path]


class Net(torch.nn.Module):
    """LSTM(p -> hidden, one layer) -> last hidden state -> dropout -> linear(hidden -> 1)."""

    def __init__(self, p: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.lstm = torch.nn.LSTM(p, hidden, num_layers=1, batch_first=True)
        self.drop = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.head(self.drop(h[-1])).squeeze(-1)


def standardise(X: np.ndarray, rows: slice) -> tuple[np.ndarray, np.ndarray]:
    """Column mean and sd over X[rows]; a column constant there gets sd 1 (centred to zero)."""
    mu = X[rows].mean(axis=0)
    sd = X[rows].std(axis=0)
    return mu, np.where(sd > 0.0, sd, 1.0)


def sequences(Z: np.ndarray, ends: np.ndarray, L: int) -> np.ndarray:
    """(len(ends), L, p) float32: rows end - L + 1 .. end of Z for every end row."""
    idx = ends[:, None] + np.arange(-L + 1, 1)[None, :]
    assert idx.min() >= 0 and idx.max() < len(Z), (idx.min(), idx.max(), len(Z))
    return np.ascontiguousarray(Z[idx], dtype=np.float32)


def train(
    Xs: np.ndarray,
    ys: np.ndarray,
    cfg: dict,
    seed: int,
    epochs: int,
    batch: int,
    grad_clip: float,
    Xv: np.ndarray | None = None,
    yv_std: np.ndarray | None = None,
    patience: int | None = None,
) -> tuple[Net, int, np.ndarray | None]:
    """Adam on MSE (standardised target).  With a validation set: up to ``epochs``
    epochs, stopping ``patience`` epochs after the best validation MSE; returns
    (net, best epoch count, validation predictions at that epoch).  Without one:
    exactly ``epochs`` epochs; returns (net, epochs, None)."""
    torch.manual_seed(seed)
    net = Net(Xs.shape[2], int(cfg["hidden"]), float(cfg["dropout"]))
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg["lr"]))
    gen = torch.Generator().manual_seed(seed)
    xt, yt = torch.from_numpy(Xs), torch.from_numpy(ys.astype(np.float32))
    xv = torch.from_numpy(Xv) if Xv is not None else None
    n = len(xt)
    best_mse, best_ep, best_pv = np.inf, 0, None
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch):
            b = perm[i : i + batch]
            opt.zero_grad()
            loss = torch.mean((net(xt[b]) - yt[b]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            opt.step()
        if xv is None:
            continue
        net.eval()
        with torch.no_grad():
            pv = net(xv).numpy().astype(np.float64)
        mse = float(np.mean((pv - yv_std) ** 2))
        if mse < best_mse:  # a non-finite loss never improves: it only runs down the patience
            best_mse, best_ep, best_pv = mse, ep, pv
        elif patience is not None and ep - best_ep >= patience:
            break
    if xv is None:
        return net, epochs, None
    return net, best_ep, best_pv


def fit_candidate(job: dict) -> dict:
    """job: X / y = paths of the tuning WINDOW's arrays (the W rows before the
    forecast row, nothing else), fit / val = row ranges inside it, cfg, seed,
    max_epochs, patience, batch, grad_clip.  Returns the epoch count at the best
    validation MSE and the validation-tail forecasts (target units) there."""
    a = time.time()
    Xw, yw = np.array(arr(job["X"])), np.array(arr(job["y"]))
    (f0, f1), (v0, v1) = job["fit"], job["val"]
    L = int(job["cfg"]["seq_len"])
    mu, sd = standardise(Xw, slice(f0, f1))
    Z = (Xw - mu) / sd
    tr_ends = np.arange(f0 + L - 1, f1)  # sequences entirely inside the fit block
    va_ends = np.arange(v0, v1)
    ymu, ysd = float(yw[tr_ends].mean()), float(yw[tr_ends].std())
    net, ep, pv = train(
        sequences(Z, tr_ends, L),
        (yw[tr_ends] - ymu) / ysd,
        job["cfg"],
        int(job["seed"]),
        int(job["max_epochs"]),
        int(job["batch"]),
        float(job["grad_clip"]),
        Xv=sequences(Z, va_ends, L),
        yv_std=(yw[va_ends] - ymu) / ysd,
        patience=int(job["patience"]),
    )
    val_pred = pv * ysd + ymu if pv is not None else np.full(len(va_ends), np.nan)
    return {"epochs": int(ep), "val_pred": val_pred, "sec": time.time() - a}


def refit_block(job: dict) -> dict:
    """job: X / y = paths of the chunk's arrays, t = the block's first row, W, k =
    rows in the block, cfg, seed, epochs, batch, grad_clip.  The configuration is
    fitted on the window [t - W, t) for exactly ``epochs`` epochs and forecasts
    rows t .. t + k - 1 (each from its own last L rows)."""
    a = time.time()
    X, y = arr(job["X"]), arr(job["y"])
    t, W, k = int(job["t"]), int(job["W"]), int(job["k"])
    L = int(job["cfg"]["seq_len"])
    Xc = np.array(X[t - W : t + k])  # the window and the block; row t - W -> 0
    yc = np.array(y[t - W : t])  # targets of the window only
    mu, sd = standardise(Xc, slice(0, W))
    Z = (Xc - mu) / sd
    tr_ends = np.arange(L - 1, W)
    ymu, ysd = float(yc[tr_ends].mean()), float(yc[tr_ends].std())
    net, _, _ = train(
        sequences(Z, tr_ends, L),
        (yc[tr_ends] - ymu) / ysd,
        job["cfg"],
        int(job["seed"]),
        int(job["epochs"]),
        int(job["batch"]),
        float(job["grad_clip"]),
    )
    net.eval()
    with torch.no_grad():
        p = net(torch.from_numpy(sequences(Z, np.arange(W, W + k), L))).numpy()
    return {"preds": p.astype(np.float64) * ysd + ymu, "fit_sec": time.time() - a}
