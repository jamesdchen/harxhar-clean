"""Unit tests of writeup/hac_inference.py (2026-10-21). Synthetic data only.

Run:  python -m pytest writeup/tests/test_hac_inference.py   (or: python writeup/tests/test_hac_inference.py)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "writeup", ROOT / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import atm_straddle_lib as asl  # noqa: E402
import hac_inference as h  # noqa: E402

T = 614


def test_bandwidth_rules():
    assert h.llsw_bandwidth(T) == 33  # ceil(1.3 * 24.78) = ceil(32.21)
    assert h.llsw_bandwidth(200) == 19  # LLSW (2018), text below rule (3)
    assert h.llsw_bandwidth(100) == 13
    assert h.nw94_lag(T) == 5
    assert h.old_lag(T) == 12 == asl.newey_west_lag(T)
    assert h.old_lag(866) == 14 == asl.newey_west_lag(866)


def test_nw_t_matches_library():
    rng = np.random.default_rng(0)
    x = 0.1 + rng.standard_normal(T)
    for L in (0, 5, 12, 32):
        t_lib, lag = asl.newey_west_t(x, lag=L)
        assert lag == L
        assert math.isclose(h.nw_t_lag(x, L), t_lib, rel_tol=1e-12)
        assert math.isclose(h.nw_t_bandwidth(x, L + 1), t_lib, rel_tol=1e-12)


def test_lrv_divides_by_T():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(50)
    e = x - x.mean()
    S = 4
    want = sum((1 - abs(j) / S) * float(e[abs(j):] @ e[: len(e) - abs(j)]) / len(e) for j in range(-S + 1, S))
    assert math.isclose(h.bartlett_lrv(x, S), want, rel_tol=1e-12)


def test_fixedb_cv_against_kv_cubic_and_normal():
    b = 33 / T
    a = h.simulate_fixedb_abs_t(b, grid=1000, draws=60_000, seed=3)
    cv = float(np.quantile(a, 0.95))
    assert abs(cv - h.kv_cubic_cv975(b)) < 0.04, (cv, h.kv_cubic_cv975(b))
    assert 2.05 < cv < 2.2
    a0 = h.simulate_fixedb_abs_t(0.004, grid=1000, draws=60_000, seed=4)  # S = 4 points: close to N(0,1)
    assert abs(float(np.quantile(a0, 0.95)) - 1.96) < 0.04
    a5 = h.simulate_fixedb_abs_t(0.5, grid=1000, draws=60_000, seed=5)
    assert abs(float(np.quantile(a5, 0.95)) - h.kv_cubic_cv975(0.5)) < 0.08


def test_fixedb_p_monotone_and_calibrated():
    b = 33 / T
    cv = h.fixedb_cv(b)
    assert abs(h.fixedb_p(cv, b) - 0.05) < 1e-3
    ps = [h.fixedb_p(t, b) for t in (0.0, 1.0, 1.96, 2.5, 4.0)]
    assert ps[0] == 1.0 and all(p1 > p2 for p1, p2 in zip(ps, ps[1:]))
    assert h.fixedb_p(-2.5, b) == h.fixedb_p(2.5, b)
    assert h.fixedb_p(1.96, b) > h.normal_p(1.96)  # fixed-b tails are heavier


def test_llsw_test_fields():
    rng = np.random.default_rng(6)
    r = h.llsw_test(0.2 + rng.standard_normal(T))
    assert r["S"] == 33 and r["T"] == T and math.isclose(r["b"], 33 / T)
    assert math.isclose(r["t"], h.nw_t_bandwidth(0.2 + np.random.default_rng(6).standard_normal(T), 33))
    assert r["t"] > r["cv"] and r["p"] < 0.05


def test_lw_point_estimate_and_symmetry():
    rng = np.random.default_rng(7)
    z = rng.standard_normal((T, 2))
    a = 0.08 + z[:, 0]
    b = 0.02 + 2.0 * (0.8 * z[:, 0] + 0.6 * z[:, 1])
    r = h.lw_sharpe_test(a, b, n_boot=499)
    assert math.isclose(r["delta"], a.mean() / a.std() - b.mean() / b.std(), rel_tol=1e-10)
    r2 = h.lw_sharpe_test(b, a, n_boot=499)
    assert math.isclose(r2["delta"], -r["delta"]) and r2["p"] == r["p"]
    assert 1 / 500 <= r["p"] <= 1.0
    # scale invariance: a positive rescaling of either series changes nothing
    r3 = h.lw_sharpe_test(3 * a, 0.5 * b, n_boot=499)
    assert math.isclose(r3["delta"], r["delta"], rel_tol=1e-10) and math.isclose(r3["d"], r["d"], rel_tol=1e-8)


def test_lw_se_matches_iid_delta_method():
    # block = 1: Psi is the sample covariance of (r_a, r_b, r_a^2, r_b^2); compare with a direct delta method
    rng = np.random.default_rng(8)
    a, b = 0.05 + rng.standard_normal(T), 0.1 + rng.standard_normal(T)
    r = h.lw_sharpe_test(a, b, block=1, n_boot=99)
    y = np.column_stack([a, b, a * a, b * b])
    psi = np.cov(y.T, ddof=0)
    m = y.mean(axis=0)
    eps = 1e-6
    f = lambda u: u[0] / math.sqrt(u[2] - u[0] ** 2) - u[1] / math.sqrt(u[3] - u[1] ** 2)  # noqa: E731
    g = np.array([(f(m + eps * np.eye(4)[i]) - f(m - eps * np.eye(4)[i])) / (2 * eps) for i in range(4)])
    assert math.isclose(r["se"], math.sqrt(g @ psi @ g / T), rel_tol=1e-6)


def test_lw_detects_large_gap_and_seed_fixed():
    rng = np.random.default_rng(9)
    z = rng.standard_normal((T, 2))
    a = 0.25 + z[:, 0]
    b = 0.0 + 0.9 * z[:, 0] + 0.4359 * z[:, 1]
    r = h.lw_sharpe_test(a, b, n_boot=999)
    assert r["p"] < 0.01
    assert h.lw_sharpe_test(a, b, n_boot=999)["p"] == r["p"]  # fixed seed


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
