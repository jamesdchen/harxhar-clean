"""PufferLib-5.0 native MinGRU forecaster (2026-10-12, Phase 1): inputs and job files for the native trainer.

User decisions (2026-10-12): replace the per-bar LSTM by PufferLib's own network, trained through
PufferLib's native C/CUDA backend (5.0, no PyTorch), purely SUPERVISED on the forecasting loss; fit space
= the adjusted target y (plain variance forecast = f^2 x B); the per-bar 16:00 designs of record
(experiments/feature_mcs_runner.load_inputs(): the all_features 16:00-bar arrays and the candidate
designs, of which Phase 1 uses `baseline` = HAR + calendar and `all_features`).

Writes (results/puffer_forecast_2026-10-12/_work/):
  data/<design>/X.f32, y.f32, meta.txt   the design's 16:00 rows (float32, row-major) and target y
  data/toy/...                           a synthetic toy problem (gradient / trajectory checks)
  dates.csv                              row -> 16:00 stamp of the design rows
  jobs_phase1.txt                        the Phase-1 window: tuning (fit block + early stopping on the
                                         125-session validation tail), refits on the whole window with
                                         each tune job's best epoch count, the 10-session forecasts,
                                         QLIKE / Huber variants, determinism duplicates, warm starts
  jobs_scaling.txt                       the same tune jobs run one at a time (concurrency scaling)
  jobs_toy.txt, jobs_grad.txt            the reference checks (experiments/puffer_forecast/reference.py)
  losscheck_<loss>.bin                   loss-kernel inputs (random and extreme forecasts / targets)

    python experiments/puffer_forecast/prep.py
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "experiments", REPO / "specs"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = REPO / "results" / "puffer_forecast_2026-10-12"
WORK = OUT / "_work"
DESIGNS = ("baseline", "all_features")
# the window of record for Phase 1: the 2,000 16:00 sessions ending 2019-12-31 (rows W0 .. W1 - 1),
# forecasting the next FC_N sessions (2020-01-02 .. 2020-01-15)
WINDOW_END = "2019-12-31 16:00:00"
W = 2000
FC_N = 10
SEEDS = tuple(20261012 + k for k in range(5))  # 5 seeds, as the LSTM spec
# the LSTM spec's grid where it carries over (seq len 5 / 20, hidden 16 / 64; no dropout in the
# native network), the learning rate on Muon's scale (upstream default 0.015)
GRID = [
    dict(L=L, H=H, lr=lr)
    for L in (5, 20)
    for H in (16, 64)
    for lr in (0.003, 0.01, 0.03)
]
COMMON = dict(
    layers=1,
    batch=128,
    max_epochs=200,
    patience=20,
    val_tail=125,
    embargo=25,
    momentum=0.95,
    max_grad_norm=1.0,
    anneal=1,
    min_lr_ratio=0.0,
    fc_n=FC_N,
)


def kv(**k) -> str:
    return " ".join(f"{a}={b}" for a, b in k.items())


def write_design(name: str, X: np.ndarray, y: np.ndarray) -> None:
    d = WORK / "data" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "X.f32").write_bytes(np.ascontiguousarray(X, dtype="<f4").tobytes())
    (d / "y.f32").write_bytes(np.ascontiguousarray(y, dtype="<f4").tobytes())
    (d / "meta.txt").write_bytes(f"{X.shape[0]} {X.shape[1]}\n".encode())


def toy() -> None:
    """y_t = 0.6 + |0.8 x_{t,0} + 0.5 x_{t-1,1} - 0.3 x_{t,2}| + noise, x an AR(1) panel: positive
    targets with a lagged dependence the recurrence must carry."""
    rng = np.random.default_rng(7)
    n, p = 400, 6
    X = np.zeros((n, p))
    for t in range(1, n):
        X[t] = 0.7 * X[t - 1] + rng.normal(size=p)
    y = (
        0.6
        + np.abs(0.8 * X[:, 0] + 0.5 * np.roll(X[:, 1], 1) - 0.3 * X[:, 2])
        + 0.1 * rng.normal(size=n) ** 2
    )
    write_design("toy", X, y)


def losscheck_inputs() -> None:
    """(o, y) pairs: random, plus extreme forecasts / targets (tiny, huge, negative f, f near 0)."""
    rng = np.random.default_rng(11)
    mu, sd = 0.95, 0.44
    o_l = list(rng.normal(size=200))
    y_l = list(np.exp(rng.normal(-0.1, 0.5, size=200)))
    f_ext = [1e-6, 1e-4, 1e-3, 0.01, 0.1, 5.0, 50.0, 500.0, -0.01, -1.0, -50.0]
    y_ext = [1e-4, 0.01, 0.16, 1.0, 7.25, 100.0]
    for f in f_ext:
        for yy in y_ext:
            o_l.append((f - mu) / sd)
            y_l.append(yy)
    o = np.asarray(o_l, np.float64)
    y = np.asarray(y_l, np.float64)
    # central-difference triplets for the kernel's own finite-difference check: (o, o + h, o - h)
    h = 1e-2 * np.maximum(1.0, np.abs(o))
    o = np.concatenate([o, o + h, o - h]).astype(np.float32)
    y = np.concatenate([y, y, y]).astype(np.float32)
    for kind, name in ((0, "mse"), (1, "huber"), (2, "qlike")):
        hdr = struct.pack("<ii5f", len(o), kind, mu, sd, 0.6, 1e-3, 25.0)
        (WORK / f"losscheck_{name}.bin").write_bytes(hdr + o.tobytes() + y.tobytes())


def main() -> None:
    import feature_mcs_runner as FR

    inp = FR.load_inputs()
    names = inp["names"]
    dates = pd.DatetimeIndex(pd.to_datetime(inp["dates"]))
    WORK.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"row": np.arange(len(dates)), "date": dates.strftime("%Y-%m-%d %H:%M:%S")}
    ).to_csv(WORK / "dates.csv", index=False)
    shapes = {}
    for m in DESIGNS:
        cols = inp["models"][m]["cols"]
        idx = [names.index(c) for c in cols]
        assert idx == sorted(idx)
        X = inp["bar_X"][:, idx]
        y = inp["bar_y"]
        assert np.isfinite(X).all() and np.isfinite(y).all() and (y > 0).all()
        write_design(m, X, y)
        shapes[m] = X.shape
    toy()
    losscheck_inputs()
    w1 = int(np.flatnonzero(dates == pd.Timestamp(WINDOW_END))[0]) + 1
    w0 = w1 - W
    assert w0 >= 0 and w1 + FC_N <= len(dates)
    D = "data"  # data dirs relative to the run directory on the cluster

    lines, scal = [], []
    for m in DESIGNS:
        for g in GRID:
            for s in SEEDS:
                tid = f"tune_{m}_L{g['L']}_H{g['H']}_lr{g['lr']}_s{s}"
                rid = tid.replace("tune_", "refit_", 1)
                base = dict(
                    data=f"{D}/{m}",
                    w0=w0,
                    w1=w1,
                    L=g["L"],
                    H=g["H"],
                    lr=g["lr"],
                    seed=s,
                    **COMMON,
                )
                lines.append(kv(id=tid, mode="tune", loss="mse", **base))
                lines.append(
                    kv(
                        id=rid,
                        mode="refit",
                        loss="mse",
                        epochs_from=tid,
                        save_params=1,
                        **base,
                    )
                )
                if s == SEEDS[0]:
                    scal.append(kv(id=tid, mode="tune", loss="mse", **base))
    # loss variants on one configuration (QLIKE with the capped gradient; Huber), both designs
    for m in DESIGNS:
        for loss in ("qlike", "huber"):
            for s in SEEDS:
                tid = f"tune_{m}_L20_H16_lr0.01_s{s}_{loss}"
                base = dict(
                    data=f"{D}/{m}", w0=w0, w1=w1, L=20, H=16, lr=0.01, seed=s, **COMMON
                )
                lines.append(kv(id=tid, mode="tune", loss=loss, **base))
                lines.append(
                    kv(
                        id=tid.replace("tune_", "refit_", 1),
                        mode="refit",
                        loss=loss,
                        epochs_from=tid,
                        **base,
                    )
                )
    # determinism: exact duplicates of two tune jobs (same wave) -- compared bit for bit with the originals
    for m in DESIGNS:
        s = SEEDS[0]
        base = dict(
            data=f"{D}/{m}", w0=w0, w1=w1, L=20, H=64, lr=0.01, seed=s, **COMMON
        )
        lines.append(
            kv(id=f"dup_tune_{m}_L20_H64_lr0.01_s{s}", mode="tune", loss="mse", **base)
        )
    # warm starts: the window shifted by 10 sessions, cold (full schedule) vs warm (from the window-of-record
    # refit, 10 / 25 epochs), seed 1, one configuration per design
    for m in DESIGNS:
        s = SEEDS[0]
        src = f"refit_{m}_L20_H16_lr0.01_s{s}"
        base = dict(
            data=f"{D}/{m}",
            w0=w0 + FC_N,
            w1=w1 + FC_N,
            L=20,
            H=16,
            lr=0.01,
            seed=s,
            **COMMON,
        )
        lines.append(
            kv(
                id=f"shift10_cold_{m}",
                mode="refit",
                loss="mse",
                phase=2,
                epochs_from=f"tune_{m}_L20_H16_lr0.01_s{s}",
                **base,
            )
        )
        for e in (10, 25):
            wb = dict(base, lr=0.003, anneal=0)
            lines.append(
                kv(
                    id=f"shift10_warm{e}_{m}",
                    mode="refit",
                    loss="mse",
                    phase=2,
                    epochs=e,
                    init_from=f"@{src}",
                    **wb,
                )
            )
    (WORK / "jobs_phase1.txt").write_bytes(("\n".join(lines) + "\n").encode())
    (WORK / "jobs_scaling.txt").write_bytes(("\n".join(scal) + "\n").encode())

    # reference checks: toy trajectory (fixed epochs, every loss), gradient dumps (toy + real designs)
    toyc = dict(
        data=f"{D}/toy",
        w0=0,
        w1=390,
        fc_n=10,
        layers=2,
        batch=32,
        max_epochs=40,
        patience=40,
        val_tail=60,
        embargo=25,
        momentum=0.95,
        max_grad_norm=1.0,
        anneal=1,
        min_lr_ratio=0.0,
    )
    tl = []
    for loss in ("mse", "qlike", "huber"):
        tl.append(
            kv(
                id=f"toy_{loss}",
                mode="refit",
                loss=loss,
                epochs=12,
                L=6,
                H=8,
                lr=0.01,
                seed=5,
                **toyc,
            )
        )
    (WORK / "jobs_toy.txt").write_bytes(("\n".join(tl) + "\n").encode())
    gl = [
        kv(
            id=f"grad_toy_{loss}",
            mode="refit",
            loss=loss,
            epochs=1,
            L=6,
            H=8,
            lr=0.01,
            seed=5,
            **toyc,
        )
        for loss in ("mse", "qlike", "huber")
    ]
    for m in DESIGNS:
        gl.append(
            kv(
                id=f"grad_{m}",
                mode="tune",
                loss="mse",
                data=f"{D}/{m}",
                w0=w0,
                w1=w1,
                L=5,
                H=16,
                lr=0.01,
                seed=SEEDS[0],
                **{k: v for k, v in COMMON.items()},
            )
        )
    (WORK / "jobs_grad.txt").write_bytes(("\n".join(gl) + "\n").encode())
    meta = dict(
        window_end=WINDOW_END,
        w0=w0,
        w1=w1,
        W=W,
        fc_n=FC_N,
        fc_dates=list(dates[w1 : w1 + FC_N].strftime("%Y-%m-%d")),
        shapes={k: list(v) for k, v in shapes.items()},
        grid=GRID,
        seeds=list(SEEDS),
        common=COMMON,
        n_jobs_phase1=len(lines),
    )
    (WORK / "prep.json").write_bytes(json.dumps(meta, indent=1).encode())
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
