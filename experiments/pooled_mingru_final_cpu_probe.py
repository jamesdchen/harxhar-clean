"""CPU timing proxy for the pooled MinGRU (2026-10-23, coordinator: "measure whether CPU training gets us there faster").

There is no CPU path that implements the trainer of this run (native_g9 forecast_train.cu is CUDA-only; its initial
weights are cuRAND draws).  experiments/puffer_forecast/torch_train.py is the earlier PyTorch port of the same network
stack (encoder, MinGRU with highway gates, 2-column decoder, Muon with the upstream Newton-Schulz coefficients, cosine
lr, splitmix64 minibatches), without the later pieces (linear shortcut / ridge init, softplus link, QLIKE early
stopping, wd_enc / wd_lin, dec_zero, window mask).  Its per-epoch compute is the same order as the native model's, so
this times it on the CPU as a cost proxy: one training epoch over N = W - L + 1 sequences (batch 128, last-step loss,
backward, Muon) plus the validation / forecast forward, at W = 39,000 rows (W = 3,000 days), L 20, H 64, P 294 (the
full set's 293 kept columns + the constant) and P 13 (HAR), at 8 / 16 / 32 threads.  Random inputs (the cost does
not depend on the values).  No forecast is compared: the port is not the run's trainer.

    python pooled_mingru_final_cpu_probe.py <dir with torch_train.py> [threads ...]   (prints a table, writes cpu_probe.csv)
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
import torch_train as tt  # noqa: E402

THREADS = [int(t) for t in sys.argv[2:]] or [8, 16, 32]
W, L, H, B, N_VAL = 39_000, 20, 64, 128, 3_250


def epoch_seconds(p: int, threads: int, epochs: int = 2) -> tuple[float, float]:
    torch.set_num_threads(threads)
    g = torch.Generator().manual_seed(0)
    z = torch.randn(W, p, generator=g)
    y = torch.randn(W, generator=g)
    net = tt.MinGRU(p, H, 1)
    tt.init_weights(net, 20261012)
    params = list(net.parameters())
    moms = [torch.zeros_like(q) for q in params]
    n = W - L + 1
    s = math.ceil(n / B)
    ends_all = torch.arange(L - 1, W)
    times = []
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        t0 = time.perf_counter()
        for k in range(s):
            idx = ends_all[perm[k * B : (k + 1) * B]]
            x = tt.sequences(z, idx, L)
            o = net(x)[:, -1, 0]
            loss = ((o - y[idx]) ** 2).mean()
            for q in params:
                q.grad = None
            loss.backward()
            tt.muon_step(params, moms, 0.003, 0.95, 1.0)
        with torch.no_grad():
            ev = torch.arange(W - N_VAL, W)
            net(tt.sequences(z, ev, L))[:, -1, 0]
        times.append(time.perf_counter() - t0)
    return times[-1], float(np.mean(times))


def main() -> int:
    rows = []
    for p, name in ((294, "all_features"), (13, "baseline")):
        for th in THREADS:
            last, mean = epoch_seconds(p, th)
            rows.append((name, p, th, last, mean))
            print(
                f"{name:12s} P {p:3d} threads {th:2d}: {last:7.2f} s / epoch (mean of 2: {mean:.2f})",
                flush=True,
            )
    import csv

    with open("cpu_probe.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "P", "threads", "s_per_epoch_last", "s_per_epoch_mean2"])
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
