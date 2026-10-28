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
    want = sum((1 - abs(j) / S) * float(e[abs(j) :] @ e[: len(e) - abs(j)]) / len(e) for j in range(-S + 1, S))
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


# ----------------------------------------------------------------------------- 2026-10-22 (user, D6)
def _d6_calendar():
    """A calendar like the paper's post-warmup days: 2021-08-11 .. 2024-04-30, Mon/Wed/Fri until 13 May 2022, every
    weekday after."""
    import pandas as pd

    d = pd.bdate_range("2021-08-11", "2024-04-30")
    return d[(d >= pd.Timestamp("2022-05-16")) | d.dayofweek.isin([0, 2, 4])]


def _ref_stat(x, yrs, A):
    """The coordinator's scratch script sharpe_equalyears.py, stat(kind="D6s"), verbatim but for A passed in."""
    srs = []
    for y in sorted(set(yrs.tolist())):
        v = x[yrs == y]
        if len(v) < 2 or not v.std(ddof=1) > 0:
            continue
        srs.append(v.mean() / v.std(ddof=1) * np.sqrt(A[y]))
    return float(np.mean(srs))


def _ref_boot_p(a, b, years, A, block=10, n=1999, seed=20261022):
    """sharpe_equalyears.py boot_p(kind="D6s"), verbatim but for years / A passed in."""
    rng = np.random.default_rng(seed)
    T = len(a)
    d0 = _ref_stat(a, years, A) - _ref_stat(b, years, A)
    nb = int(np.ceil(T / block))
    out = np.empty(n)
    for i in range(n):
        st = rng.integers(0, T, nb)
        idx = ((st[:, None] + np.arange(block)[None, :]) % T).ravel()[:T]
        out[i] = _ref_stat(a[idx], years[idx], A) - _ref_stat(b[idx], years[idx], A)
    return float(np.mean(np.abs(out - d0) >= abs(d0)))


def test_d6_years_coverage():
    d = _d6_calendar()
    yy = h.d6_years(d)
    assert sorted(yy["A"]) == [2021, 2022, 2023, 2024]
    assert math.isclose(yy["coverage"][2021], 143 / 365) and math.isclose(yy["coverage"][2024], 121 / 366)
    assert yy["coverage"][2022] == 1.0 == yy["coverage"][2023]
    for y in yy["A"]:
        assert math.isclose(yy["A"][y], yy["N"][y] / yy["coverage"][y])
    # span widens the sample: a year inside it is fully covered
    import pandas as pd

    yy2 = h.d6_years(d[d.year == 2022], span=(pd.Timestamp("2021-08-11"), pd.Timestamp("2024-04-30")))
    assert yy2["coverage"] == {2022: 1.0}


def test_d6_sharpe_matches_scratch_and_definition():
    d = _d6_calendar()
    rng = np.random.default_rng(10)
    x = 0.05 + rng.standard_normal(len(d))
    yy = h.d6_years(d)
    r = h.d6_sharpe(x, d)
    assert math.isclose(r["sharpe"], _ref_stat(x, yy["years"], yy["A"]), rel_tol=1e-12)
    # each year: mean / sd * sqrt(A_y); the years combined with equal weights
    srs = [x[yy["years"] == y].mean() / x[yy["years"] == y].std(ddof=1) * math.sqrt(yy["A"][y]) for y in yy["A"]]
    assert math.isclose(r["sharpe"], float(np.mean(srs)), rel_tol=1e-12)
    # a sample spanning the whole calendar year: A_y = N_y, so the year's term is the per-year ratio of new-notes 15
    # (the sample's own first and last days, 2 Jan and 29 Dec, cover 362/365 of it)
    import pandas as pd

    full = pd.bdate_range("2023-01-01", "2023-12-31")
    z = 0.1 + rng.standard_normal(len(full))
    whole = (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"))
    d1 = asl.sharpe_ann_by_year(z, dates=full)["sharpe"]
    assert math.isclose(h.d6_sharpe(z, full, span=whole)["sharpe"], d1, rel_tol=1e-12)
    assert math.isclose(h.d6_sharpe(z, full)["sharpe"], d1 * math.sqrt(365 / 362), rel_tol=1e-12)


def test_d6_diff_test_matches_scratch_boot_p():
    d = _d6_calendar()
    rng = np.random.default_rng(11)
    z = rng.standard_normal((len(d), 2))
    a = 0.08 + z[:, 0]
    b = 0.02 + 0.8 * z[:, 0] + 0.6 * z[:, 1]
    yy = h.d6_years(d)
    r = h.d6_diff_test(a, b, d, n_boot=199, seed=20261022)
    assert math.isclose(
        r["delta"],
        _ref_stat(a, yy["years"], yy["A"]) - _ref_stat(b, yy["years"], yy["A"]),
        rel_tol=1e-12,
    )
    assert r["p"] == _ref_boot_p(a, b, yy["years"], yy["A"], n=199)
    # antisymmetry and scale invariance; a fixed seed
    r2 = h.d6_diff_test(b, a, d, n_boot=199, seed=20261022)
    assert math.isclose(r2["delta"], -r["delta"]) and r2["p"] == r["p"]
    r3 = h.d6_diff_test(3 * a, 0.5 * b, d, n_boot=199, seed=20261022)
    assert math.isclose(r3["delta"], r["delta"], rel_tol=1e-10) and r3["p"] == r["p"]
    assert h.d6_diff_test(a, b, d, n_boot=199, seed=20261022)["p"] == r["p"]


def test_d6_diff_test_detects_large_gap():
    d = _d6_calendar()
    rng = np.random.default_rng(12)
    z = rng.standard_normal((len(d), 2))
    a = 0.3 + z[:, 0]
    b = 0.0 + 0.9 * z[:, 0] + 0.4359 * z[:, 1]
    assert h.d6_diff_test(a, b, d, n_boot=999)["p"] < 0.01


# 2026-10-26 (author decision D19): the paper's pooled Sharpe ratio and its tests
def test_paper_sharpe_definition():
    d = _d6_calendar()
    rng = np.random.default_rng(13)
    x = 0.05 + rng.standard_normal(len(d))
    yy = h.d6_years(d)
    pq = h.paper_q(d)
    assert math.isclose(pq["sum_c"], sum(yy["coverage"].values()), rel_tol=1e-14)
    assert math.isclose(pq["q"], len(d) / pq["sum_c"], rel_tol=1e-14)
    r = h.paper_sharpe(x, d)
    assert math.isclose(r["sharpe"], math.sqrt(pq["q"]) * x.mean() / x.std(ddof=1), rel_tol=1e-12)
    assert math.isclose(r["sharpe_d6"], h.d6_sharpe(x, d)["sharpe"], rel_tol=1e-14)
    # one full calendar year spanned: q = N_y, the classical sqrt(N) annualization
    import pandas as pd

    full = pd.bdate_range("2023-01-01", "2023-12-31")
    whole = (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"))
    assert math.isclose(h.paper_q(full, span=whole)["q"], len(full), rel_tol=1e-14)


def test_paper_sharpe_test_is_ledoit_wolf_and_boot_check():
    d = _d6_calendar()
    rng = np.random.default_rng(14)
    z = rng.standard_normal((len(d), 2))
    a = 0.08 + z[:, 0]
    b = 0.02 + 0.8 * z[:, 0] + 0.6 * z[:, 1]
    r = h.paper_sharpe_test(a, b, d, n_boot=199)
    lw = h.lw_sharpe_test(a, b, n_boot=199)
    assert r["p"] == r["p_lw"] == lw["p"]
    assert math.isclose(r["delta"], h.paper_sharpe(a, d)["sharpe"] - h.paper_sharpe(b, d)["sharpe"], rel_tol=1e-12)
    bt = h.pooled_diff_boot(a, b, d, n_boot=199)
    assert r["p_boot"] == bt["p"] and math.isclose(bt["delta"], r["delta"], rel_tol=1e-12)
    # the bootstrap check uses d6_diff_test's resamples: antisymmetric, scale invariant
    bt2 = h.pooled_diff_boot(b, a, d, n_boot=199)
    assert math.isclose(bt2["delta"], -bt["delta"]) and bt2["p"] == bt["p"]
    bt3 = h.pooled_diff_boot(3 * a, 0.5 * b, d, n_boot=199)
    assert math.isclose(bt3["delta"], bt["delta"], rel_tol=1e-10) and bt3["p"] == bt["p"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
