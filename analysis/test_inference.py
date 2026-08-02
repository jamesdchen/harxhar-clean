"""Validation for analysis/inference.py.

Checks the routines against scipy/statsmodels where an independent
implementation exists, and against analytic properties where it does not.
Run: python3 analysis/test_inference.py
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import (  # noqa: E402
    qlike_loss, newey_west_lrv, dm_test, model_confidence_set,
    holm, benjamini_hochberg, stationary_bootstrap_indices, _student_t_sf,
)

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAIL.append(name)


print("\n[1] QLIKE loss")
y = np.array([1.0, 2.0, 0.5])
check("minimised at yhat=y", np.allclose(qlike_loss(y, y), 0.0))
check("positive off-optimum", np.all(qlike_loss(y, y * 1.3) > 0))
# asymmetry: under-prediction must cost more than the mirrored over-prediction
under = qlike_loss(np.array([1.0]), np.array([0.5]))[0]
over = qlike_loss(np.array([1.0]), np.array([2.0]))[0]
check("penalises under-prediction harder", under > over, f"{under:.4f} > {over:.4f}")
try:
    qlike_loss(np.array([1.0]), np.array([0.0]))
    check("rejects non-positive forecast", False)
except ValueError:
    check("rejects non-positive forecast", True)

print("\n[2] Student-t tail vs scipy")
from scipy import stats  # noqa: E402
ok = True
for t in [0.1, 1.0, 2.5, 7.3, 19.9]:
    for df in [10, 100, 2188]:
        mine = _student_t_sf(t, df)
        ref = 2 * stats.t.sf(t, df)
        if not np.isclose(mine, ref, rtol=1e-8, atol=1e-300):
            ok = False
            print(f"      t={t} df={df}: mine={mine:.6e} scipy={ref:.6e}")
check("two-sided p matches scipy to 1e-8", ok)

print("\n[3] Newey-West long-run variance")
rng = np.random.default_rng(0)
x = rng.standard_normal(5000)
check("iid: LRV ~ variance at lag 0", np.isclose(newey_west_lrv(x, lag=0), x.var(), rtol=1e-10))
# AR(1) with phi: true LRV = sigma_e^2/(1-phi)^2; check we move toward it
phi = 0.6
e = rng.standard_normal(200000)
ar = np.zeros_like(e)
for i in range(1, len(e)):
    ar[i] = phi * ar[i - 1] + e[i]
lrv = newey_west_lrv(ar, lag=40)
true_lrv = 1.0 / (1 - phi) ** 2
check("AR(1) LRV within 10% of truth", abs(lrv / true_lrv - 1) < 0.10,
      f"{lrv:.3f} vs {true_lrv:.3f}")
check("LRV > naive var under positive autocorrelation", lrv > ar.var())

print("\n[4] Newey-West vs statsmodels")
try:
    import statsmodels.api as sm
    d = ar[:3000]
    X = np.ones((d.size, 1))
    lag = 12
    res = sm.OLS(d, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False})
    sm_se = float(np.sqrt(res.cov_params()[0, 0]))
    mine_se = float(np.sqrt(newey_west_lrv(d, lag=lag) / d.size))
    check("HAC SE of the mean matches statsmodels", np.isclose(mine_se, sm_se, rtol=1e-6),
          f"{mine_se:.6e} vs {sm_se:.6e}")
except ImportError:
    print("      statsmodels unavailable - skipped")

print("\n[5] Diebold-Mariano")
n = 4000
la = rng.standard_normal(n) + 1.0
check("identical series -> t=0, p=1", dm_test(la, la.copy())["t"] == 0.0 or
      np.isnan(dm_test(la, la.copy())["t"]) or abs(dm_test(la, la.copy())["t"]) < 1e-9)
lb = la + 0.05 + 0.01 * rng.standard_normal(n)
r_ab = dm_test(la, lb)
r_ba = dm_test(lb, la)
check("A better than B gives negative t", r_ab["t"] < 0, f"t={r_ab['t']:.2f}")
check("antisymmetric in argument order", np.isclose(r_ab["t"], -r_ba["t"], rtol=1e-12))
check("p-values agree both directions", np.isclose(r_ab["p"], r_ba["p"], rtol=1e-12))
# size: under the null of equal accuracy, rejection rate should be near nominal
rej = 0
TRIALS = 400
for s in range(TRIALS):
    g = np.random.default_rng(1000 + s)
    a = g.standard_normal(500)
    b = g.standard_normal(500)
    if dm_test(a, b)["p"] < 0.05:
        rej += 1
rate = rej / TRIALS
check("empirical size near 5% under the null", 0.02 < rate < 0.09, f"{rate:.3f}")

print("\n[6] Stationary bootstrap")
idx = stationary_bootstrap_indices(1000, 500, 10.0, np.random.default_rng(3))
check("index matrix shape", idx.shape == (500, 1000))
check("indices in range", idx.min() >= 0 and idx.max() < 1000)
mu = np.arange(1000).mean()
boot_mu = np.arange(1000)[idx].mean(axis=1)
check("bootstrap mean unbiased for the sample mean", abs(boot_mu.mean() - mu) < 15,
      f"{boot_mu.mean():.1f} vs {mu:.1f}")

print("\n[7] Model confidence set")
g = np.random.default_rng(7)
n = 2000
# Three genuinely tied models and one clearly worse
base = g.standard_normal((n, 1))
L = np.hstack([base + 0.02 * g.standard_normal((n, 1)) for _ in range(3)]
              + [base + 1.0 + 0.02 * g.standard_normal((n, 1))])
res = model_confidence_set(L, ["a", "b", "c", "bad"], alpha=0.10, B=1500, seed=1)
check("clearly-worse model eliminated", "bad" not in res["surviving"], str(res["surviving"]))
check("tied models retained", set(res["surviving"]) >= {"a", "b", "c"})
check("every model has an MCS p-value", len(res["pvalues"]) == 4)
check("eliminated model's p-value below alpha", res["pvalues"]["bad"] < 0.10)
# All-tied case: nothing should be eliminated
Lt = np.hstack([base + 0.02 * g.standard_normal((n, 1)) for _ in range(4)])
res2 = model_confidence_set(Lt, list("wxyz"), alpha=0.10, B=1500, seed=2)
check("all-tied set survives intact", len(res2["surviving"]) == 4, str(res2["surviving"]))
res3 = model_confidence_set(L, ["a", "b", "c", "bad"], alpha=0.10, B=1500, seed=1,
                            statistic="TR")
check("TR statistic also eliminates the worse model", "bad" not in res3["surviving"])

print("\n[8] Multiplicity")
p = {"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.5}
h = holm(p, alpha=0.05)
check("Holm adjusted p >= raw p", all(h[k][0] >= p[k] for k in p))
check("Holm monotone in rank", h["a"][0] <= h["b"][0] <= h["c"][0] <= h["d"][0])
check("Holm smallest p: 4*0.001", np.isclose(h["a"][0], 0.004))
bh = benjamini_hochberg(p, alpha=0.05)
check("BH adjusted p >= raw p", all(bh[k][0] >= p[k] for k in p))
check("BH monotone in rank", bh["a"][0] <= bh["b"][0] <= bh["c"][0] <= bh["d"][0])
check("BH no stricter than Holm", all(bh[k][0] <= h[k][0] + 1e-12 for k in p))
try:
    from statsmodels.stats.multitest import multipletests
    keys = list(p)
    raw = [p[k] for k in keys]
    _, hs, _, _ = multipletests(raw, alpha=0.05, method="holm")
    _, bhs, _, _ = multipletests(raw, alpha=0.05, method="fdr_bh")
    check("Holm matches statsmodels", all(np.isclose(h[k][0], hs[i]) for i, k in enumerate(keys)))
    check("BH matches statsmodels", all(np.isclose(bh[k][0], bhs[i]) for i, k in enumerate(keys)))
except ImportError:
    print("      statsmodels unavailable - skipped")

print("\n" + "=" * 60)
if FAIL:
    print(f"FAILED ({len(FAIL)}): {FAIL}")
    sys.exit(1)
print("all checks passed")
