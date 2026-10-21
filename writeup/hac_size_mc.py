"""Monte Carlo size check of writeup/hac_inference.py on synthetic returns, T = 614 (2026-10-21).

No market data: every series is simulated here, so this runs anywhere (it ran in the cloud session that wrote the
helper; the author need not re-run it).

MEAN TEST (H0: E x_t = 0), REPS_MEAN replications for each design:
  iid_t5       x_t i.i.d. Student t(5)
  iid_skew     x_t i.i.d. centered lognormal, exp(0.8 Z) - E exp(0.8 Z) (right skew, heavy right tail)
  ar03_t5      x_t = 0.3 x_{t-1} + e_t, e_t i.i.d. t(5)
  ar05_t5      x_t = 0.5 x_{t-1} + e_t, e_t i.i.d. t(5)
  Tests: the LLSW / fixed-b test (bandwidth 33, fixed-b critical value), and for comparison the Newey-West t with
  normal critical values at lag 5 (Newey-West 1994 rule) and lag 12 (floor(1.5 T^(1/3))).

SHARPE TEST (H0: mu_a / sigma_a = mu_b / sigma_b), REPS_LW replications, LW_BOOT_MC resamples each:
  a_t = 0.05 + u_t, b_t = 2 (0.05 + v_t) (equal daily Sharpe ratio 0.05, different scales), corr(u, v) = 0.8, with
  u, v from the innovations of the mean designs (iid_t5, iid_skew, ar03_t5); blocks 5, 10, 20. One extra run at the
  helper's defaults (block 10, 4,999 resamples).

Writes results/hac_size_mc_2026-10-21/size.csv and SUMMARY.md.
Run:  python writeup/hac_size_mc.py [--quick]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "writeup"))
import hac_inference as h  # noqa: E402

T = 614
OUT = ROOT / "results" / "hac_size_mc_2026-10-21"
SEED = 614


def innov(kind: str, rng, n: int, m: int = 1) -> np.ndarray:
    if kind == "t5":
        return rng.standard_t(5, size=(n, m))
    if kind == "skew":
        return np.exp(0.8 * rng.standard_normal((n, m))) - np.exp(0.32)
    raise ValueError(kind)


def series(design: str, rng, m: int = 1, corr: float | None = None) -> np.ndarray:
    """(T, m) draws of a mean-zero design; with corr, m = 2 columns correlated through a common factor."""
    kind = "skew" if design == "iid_skew" else "t5"
    rho = {"iid_t5": 0.0, "iid_skew": 0.0, "ar03_t5": 0.3, "ar05_t5": 0.5}[design]
    burn = 200
    if corr is None:
        e = innov(kind, rng, T + burn, m)
    else:
        c = innov(kind, rng, T + burn, 1)
        i = innov(kind, rng, T + burn, m)
        e = np.sqrt(corr) * c + np.sqrt(1 - corr) * i
    if rho:
        x = np.empty_like(e)
        x[0] = e[0]
        for t in range(1, len(e)):
            x[t] = rho * x[t - 1] + e[t]
        e = x
    return e[burn:]


def mean_size(design: str, reps: int, rng) -> dict:
    S, b = h.llsw_bandwidth(T), h.llsw_bandwidth(T) / T
    cv = h.fixedb_cv(b)
    rej = np.zeros(3)
    for _ in range(reps):
        x = series(design, rng)[:, 0]
        rej += [
            abs(h.nw_t_bandwidth(x, S)) > cv,
            h.normal_p(h.nw_t_lag(x, h.nw94_lag(T))) < 0.05,
            h.normal_p(h.nw_t_lag(x, h.old_lag(T))) < 0.05,
        ]
    r = rej / reps
    return {"llsw_fixedb": r[0], "nw_lag5_normal": r[1], "nw_lag12_normal": r[2]}


def lw_size(design: str, reps: int, block: int, n_boot: int, rng) -> float:
    rej = 0
    for k in range(reps):
        uv = series(design, rng, m=2, corr=0.8)
        uv = uv / uv.std(axis=0)  # unit scale, mean zero in population (sample sd only rescales both columns)
        a = 0.05 + uv[:, 0]
        b = 2.0 * (0.05 + uv[:, 1])
        rej += h.lw_sharpe_test(a, b, block=block, n_boot=n_boot, seed=SEED + k)["p"] < 0.05
    return rej / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    q = ap.parse_args().quick
    reps_mean, reps_lw, boot_mc, reps_lw_full = (500, 100, 199, 50) if q else (10_000, 1_000, 999, 500)
    rng = np.random.default_rng(SEED)
    rows = []
    t0 = time.time()
    for d in ("iid_t5", "iid_skew", "ar03_t5", "ar05_t5"):
        for test, r in mean_size(d, reps_mean, rng).items():
            rows.append({"hypothesis": "mean = 0", "design": d, "test": test, "block": "", "resamples": "",
                         "reps": reps_mean, "rejection_rate": r})
            print(rows[-1], f"{time.time() - t0:.0f}s", flush=True)
    for d in ("iid_t5", "iid_skew", "ar03_t5"):
        for blk in (5, 10, 20):
            r = lw_size(d, reps_lw, blk, boot_mc, rng)
            rows.append({"hypothesis": "equal Sharpe ratios", "design": d, "test": "ledoit_wolf", "block": blk,
                         "resamples": boot_mc, "reps": reps_lw, "rejection_rate": r})
            print(rows[-1], f"{time.time() - t0:.0f}s", flush=True)
        r = lw_size(d, reps_lw_full, h.LW_BLOCK, h.LW_BOOT, rng)
        rows.append({"hypothesis": "equal Sharpe ratios", "design": d, "test": "ledoit_wolf", "block": h.LW_BLOCK,
                     "resamples": h.LW_BOOT, "reps": reps_lw_full, "rejection_rate": r})
        print(rows[-1], f"{time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df["mc_se"] = np.sqrt(0.05 * 0.95 / df["reps"])
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_quick" if q else ""
    df.to_csv(OUT / f"size{tag}.csv", index=False, float_format="%.4f")
    S = h.llsw_bandwidth(T)
    lines = [
        "# Size of the strategy-return tests on synthetic data (writeup/hac_size_mc.py, 2026-10-21)",
        "",
        f"T = {T}; LLSW bandwidth S = {S}, b = {S / T:.4f}; simulated fixed-b two-sided 5 % critical value "
        f"{h.fixedb_cv(S / T):.3f} (Kiefer-Vogelsang cubic {h.kv_cubic_cv975(S / T):.3f}). Nominal level 5 %; "
        "Monte Carlo standard error in column mc_se. Designs and tests: the module docstring.",
        "",
        df.to_markdown(index=False, floatfmt=".4f") if _has_tabulate() else df.to_string(index=False),
        "",
    ]
    (OUT / f"SUMMARY{tag}.md").write_text("\n".join(lines))
    print(f"wrote {OUT}", f"{time.time() - t0:.0f}s")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
    except ImportError:
        return False
    return True


if __name__ == "__main__":
    main()
