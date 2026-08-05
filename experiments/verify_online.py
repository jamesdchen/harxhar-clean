"""Verify enet_online (Garrigues online homotopy) reproduces the batch HAR-unpenalized enet bar-for-bar
over a sliding window. Seed once from batch; then online add+remove per bar; compare theta/pred to a
fresh batch solve each bar. No fallback -- if this matches to ~1e-7 with no drift, it's exact."""

import time

import json
import numpy as np

from reclasso_har import GramState, enet_coef, enet_online  # noqa: F401

B = "results/covid_imp_rank/all_buckets"
HAR = {"har_ma_1", "har_ma_5", "har_ma_25", "har_ma_125", "har_ma_625", "har_ma_3125"}
feats = json.load(open(f"{B}/meta.json"))["feats"]
har = np.array([f in HAR for f in feats])
exog = ~har
X = np.load(f"{B}/X_imp.npy")
y = np.load(f"{B}/y.npy").astype(np.float64)

tw = 12000
s0 = 170000
nbars = 60
CONFIGS = [(1e-3, 1.0), (3e-3, 1.0), (1e-3, 0.2), (1e-3, 0.5)]  # pure lasso first, then dense

# augmented layout: col 0 = intercept(ones), cols 1.. = X
m = X.shape[1] + 1
har_aug = np.concatenate([[False], har])
exog_aug = np.concatenate([[False], exog])
locked = np.zeros(m, bool)
locked[0] = True
locked[har_aug] = True


def aug_rows(lo, hi):
    Xr = X[lo:hi].astype(np.float64)
    return np.concatenate([np.ones((hi - lo, 1)), Xr], axis=1)


def batch_ref(lo, hi, a, l1):
    """Batch HAR-unpenalized enet via FWL (pinv HAR) + verified enet_coef on the exog residual."""
    Xr = aug_rows(lo, hi)
    yr = y[lo:hi]
    Hh = Xr[:, locked]
    Ee = Xr[:, exog_aug]
    Hp = np.linalg.pinv(Hh)
    bHy = Hp @ yr
    bHE = Hp @ Ee
    Eres = Ee - Hh @ bHE
    yres = yr - Hh @ bHy
    bE = enet_coef(Eres.T @ Eres, Eres.T @ yres, hi - lo, a, l1)
    bH = bHy - bHE @ bE
    theta = np.zeros(m)
    theta[locked] = bH
    theta[exog_aug] = bE
    return theta


def run_config(a, l1):
    mu = tw * a * l1
    lam2 = tw * a * (1.0 - l1)
    mu_vec = np.zeros(m)
    mu_vec[exog_aug] = mu
    lo, hi = s0, s0 + tw
    Xa = aug_rows(lo, hi)
    Gr = Xa.T @ Xa
    ei = np.where(exog_aug)[0]
    Gr[ei, ei] += lam2
    c = Xa.T @ y[lo:hi]
    theta = batch_ref(lo, hi, a, l1)
    A = list(np.where((np.abs(theta) > 1e-9) | locked)[0])
    s = [0.0 if locked[j] else float(np.sign(theta[j])) for j in A]
    max_th = max_pred = t_on = 0.0
    ev = 0
    for k in range(1, nbars + 1):
        add_i, rem_i = s0 + (k - 1) + tw, s0 + (k - 1)
        ua = np.concatenate([[1.0], X[add_i].astype(np.float64)])
        ur = np.concatenate([[1.0], X[rem_i].astype(np.float64)])
        t0 = time.time()
        theta, A, s = enet_online(Gr, c, mu_vec, ua, float(y[add_i]), +1.0, theta, A, s, locked)
        theta, A, s = enet_online(Gr, c, mu_vec, ur, float(y[rem_i]), -1.0, theta, A, s, locked)
        t_on += time.time() - t0
        ev += len(A)
        lo, hi = s0 + k, s0 + k + tw
        ref = batch_ref(lo, hi, a, l1)
        max_th = max(max_th, float(np.max(np.abs(theta - ref))))
        xt = np.concatenate([[1.0], X[hi].astype(np.float64)])
        max_pred = max(max_pred, abs(float(xt @ theta) - float(xt @ ref)))
    tag = "pure lasso" if l1 == 1.0 else ("dense" if l1 <= 0.2 else "enet")
    print(
        f"a={a:g} l1={l1:g} [{tag}] active~{ev // nbars:3d}  max|dtheta|={max_th:.2e}  "
        f"max|dpred|={max_pred:.2e}  {'PASS' if max_th < 1e-6 else '*** FAIL ***'}  "
        f"({1000 * t_on / nbars:.1f} ms/bar)",
        flush=True,
    )


for a, l1 in CONFIGS:
    run_config(a, l1)
