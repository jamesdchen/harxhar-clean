"""I5 cross-cluster check: the design, the target and the per-window KEPT-COLUMN SET of an Optuna chunk.

Runs specs/causal_tune_trees_optuna.py as __main__ (runpy) with its own env axes (HPC_KW_MODEL,
HPC_KW_EXOG_BUCKET, HPC_KW_SEGMENT, HPC_KW_TRAIN_WIN, HPC_KW_START / END / HALO, ...), with
src.backtest.executor.run_executor wrapped so that the spec's OWN executor call (every keyword
argument as the spec passes it) runs with a capturing fit_predict in place of the stage's one.
The capture records, for the chunk's design X (rows x p) and target y handed to fit_predict:
  sha256 of X's and y's float64 bytes, their shape, and for every forecast row t = W + i of the
  chunk the kept set src.models.window_mask.window_keep(X[t - W : t]) (the window a tuning point
  and a refit at row t are both fitted on): its size and the sha256 of its int64 indices.
Nothing is fitted; the spec stops after the executor returns.  Written to --out (CSV, one row per
forecast row; the chunk-level hashes repeated on every row); --save-x also writes X and y (.npz), so
two machines' designs can be compared element by element.

Run on each cluster with the cluster's runtime and the same env axes, then compare the CSVs
(experiments/optuna_h2_crosscheck.py).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import runpy
import socket
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "specs" / "causal_tune_trees_optuna.py"


class _Captured(Exception):
    """Raised after the spec's executor call returns: the rest of the spec's run is skipped."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cluster", required=True)
    ap.add_argument(
        "--save-x",
        default="",
        help="also save the chunk's X and y (float64) to this .npz",
    )
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from src.backtest import executor as ex
    from src.models.window_mask import window_keep

    rows: list[dict] = []
    orig = ex.run_executor

    def capture(X_chunk, y_chunk, train_win_periods, hyperparams):
        X = np.ascontiguousarray(X_chunk, dtype=np.float64)
        y = np.ascontiguousarray(y_chunk, dtype=np.float64)
        W = int(train_win_periods)
        if a.save_x:
            Path(a.save_x).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(a.save_x, X=X, y=y, host=socket.gethostname())
        xs = hashlib.sha256(X.tobytes()).hexdigest()
        ys = hashlib.sha256(y.tobytes()).hexdigest()
        start = int(os.environ["HPC_KW_START"])
        off = start - W if start else 0
        for i in range(len(X) - W):
            keep = window_keep(X[i : W + i])
            rows.append(
                {
                    "row": off + i,
                    "n_kept": len(keep),
                    "kept_sha": hashlib.sha256(
                        np.asarray(keep, np.int64).tobytes()
                    ).hexdigest(),
                    "window_sha": hashlib.sha256(X[i : W + i].tobytes()).hexdigest(),
                    "X_sha": xs,
                    "y_sha": ys,
                    "n_rows": X.shape[0],
                    "p": X.shape[1],
                }
            )
        return np.full(len(X) - W, np.nan)

    def wrapped(**kw):
        kw = dict(kw)
        kw["fit_predict"] = capture
        kw["output_file"] = os.path.join(tempfile.mkdtemp(prefix="probe_"), "probe.csv")
        orig(**kw)
        raise _Captured

    ex.run_executor = wrapped  # type: ignore[assignment]
    try:
        runpy.run_path(str(SPEC), run_name="__main__")
    except _Captured:
        pass
    assert rows, "the spec's executor call was not captured"
    tab = pd.DataFrame(rows)
    tab.insert(0, "cluster", a.cluster)
    tab["host"] = socket.gethostname()
    tab["machine"] = platform.processor() or platform.machine()
    tab["model"] = os.environ.get("HPC_KW_MODEL")
    tab["bucket"] = os.environ.get("HPC_KW_EXOG_BUCKET")
    tab["window_mask"] = os.environ.get("HPC_KW_WINDOW_MASK", "0")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(a.out, index=False)
    print(
        f"probe {a.cluster}: {len(tab)} rows, p={tab['p'].iloc[0]}, X {tab['X_sha'].iloc[0][:12]}, "
        f"y {tab['y_sha'].iloc[0][:12]}, n_kept {tab['n_kept'].min()}..{tab['n_kept'].max()} -> {a.out}"
    )


if __name__ == "__main__":
    main()
