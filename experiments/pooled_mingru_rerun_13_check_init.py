"""Check (2026-10-22): the ridge init of experiments/pooled_mingru_rerun_13.py reproduces the original pooled study's
shipped inits (results/puffer_forecast_2026-10-12/_work/pooled_lin_init, CARC ~/puffer_forecast/data/pooled_lin_init)
on the original pooled panel (~/puffer_forecast/data/pooled_walk_<set>): the refit init at block 0's first origin
(r24000, L 20 and 48, every wd_lin) and block 0's tune inits, both sets.  Run on CARC (cluster/pooled_mingru13_cpu_carc
.sbatch MODE=check CHECK=experiments/pooled_mingru_rerun_13_check_init.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pooled_mingru_rerun_13 as M  # noqa: E402

OLD = Path.home() / "puffer_forecast" / "data"
W, VAL, EMB, O = 24_000, 2_400, 48, 24_000  # noqa: E741  (O: the origin, as pooled_rinit.py)
worst = 0.0
for s in ("baseline", "all_features"):
    d = OLD / f"pooled_walk_{s}"
    n, p = map(int, (d / "meta.txt").read_text().split())
    X32 = np.memmap(d / "X.f32", "<f4", "r", shape=(n, p))
    X64 = np.memmap(d / "X.f64", "<f8", "r", shape=(n, p))
    y = np.fromfile(d / "y.f32", "<f4")
    pf = np.fromfile(d / "enc_pf.f64", "<f8")
    keep = M.window_keep64(X64, O - W, O)
    pfk = np.append(pf[keep], 0.0)
    for L in (20, 48):
        for kind, r1 in (("refit", O), ("tune", O - VAL - EMB)):
            A, b, mu, sd, nn = M.gram(X32, y, keep, O - W + L - 1, r1)
            for wl in (0.03, 0.3, 3.0):
                v = M.solve(A, b, pfk, wl / sd**2).astype("<f4")
                f = (
                    OLD
                    / "pooled_lin_init"
                    / s
                    / (
                        f"r{O}_L{L}_wl{wl:g}.f32"
                        if kind == "refit"
                        else f"tune_b0_L{L}_wl{wl:g}.f32"
                    )
                )
                ref = np.fromfile(f, "<f4")
                assert len(ref) == len(v), (f, len(ref), len(v))
                rel = float(np.max(np.abs(v.astype(float) - ref)) / np.max(np.abs(ref)))
                worst = max(worst, rel)
                print(
                    f"{s} {kind} L{L} wl{wl:g}: P {len(v)}, max |new - shipped| / max |shipped| = {rel:.2e}, "
                    f"bit-identical {np.array_equal(v, ref)}"
                )
print(f"worst relative difference {worst:.2e}")
assert worst < 1e-5, worst
print("INIT_REPRO_OK")
