"""Check (b), ridge, in detail: where the largest per-coefficient relative difference between the walk's weighted fit
and the explicit standardized fit sits, and how both compare with a refined solution of the same system (two steps of
iterative refinement in the standardized space).

usage: python check_b_ridge_detail.py <frozen_penalties.json of the flag-on select> <out json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

WT = Path(r"C:\Users\james\AppData\Local\Temp\std_wt")
DZ = Path(
    r"C:\Users\james\AppData\Local\Temp\claude\C--Users-james-CC-Allowed-harxhar-0dte-professor"
    r"\bd4224e0-12aa-4614-b6a6-3fe50535ade1\scratchpad\unobs\d_on\_work\design_all_features"
)
sys.path.insert(0, str(WT / "experiments"))
sys.path.insert(0, str(WT))
import pool_linear_pretune as P  # noqa: E402

fr = json.loads(Path(sys.argv[1]).read_text())
P.SAMPLE, P.TW_DAYS, P.PERIODS_PER_DAY, P.W, P.HAR_BASE = "rth13", 1000, 13, 13000, "5"
P.EX_FLAGS = dict(
    DROP_ACTIVITY_MA=True,
    CALENDAR_THROUGH_OVERNIGHT=True,
    DROP_OVERNIGHT_INDICATOR=True,
    LEAN_GLOBAL_LAGS=True,
    LATE_START_WARMUP=True,
)
ns = P.spec_section("ridge", "all_features", 0, -1, 0)
meta = json.loads((DZ / "design.json").read_text())
names = meta["names"] + ["intercept"]
bb = P.backbone_mask(meta["names"])
X = np.load(DZ / "X.npy", mmap_mode="r")
y = np.load(DZ / "y.npy")
W, o = 13000, 44250 + 250 * 20
p = X.shape[1]
locked = np.zeros(p + 1, dtype=bool)
locked[-1] = True
pf = np.ones(p + 1)
pf[:-1][bb] = 0.01
a = float(fr["ridge"]["all_features"])
Base = ns["RollingTunedLinear"]
Base.grid = [("ridge", a, 0.0)]
Base.trace, Base.mask_trace, Base.reseed_trace = [], [], []
Cls = P.make_class(ns, 0.01, P.ConstSched(a), o - W, [])
Cls.bb = bb
Cls.std_seeds, Cls.std_ridge, Cls.std_blockend = [], [], []
rg = Cls()
rg.init_window(np.asarray(X[o - W : o]), y[o - W : o])
th = rg._th.copy()
Xa, yr = rg._masked_window()
s = np.std(Xa, axis=0)
out = ~locked & (s < P.STD_MIN)
sc = np.where(out | locked, 1.0, s)
k = np.flatnonzero(~out)
Z = (Xa / sc)[:, k]
A = Z.T @ Z
A[np.diag_indices_from(A)] += np.where(locked[k], 0.0, a * pf[k])
b = Z.T @ yr
g = np.linalg.solve(A, b)
gr = g.copy()
for _ in range(2):  # iterative refinement (residual in float64)
    gr = gr + np.linalg.solve(A, b - A @ gr)
beta_e = np.zeros(p + 1)
beta_e[k] = g / sc[k]
beta_r = np.zeros(p + 1)
beta_r[k] = gr / sc[k]
cond = float(np.linalg.cond(A))
Xo = Xa[:, k]
M = Xo.T @ Xo
M[np.diag_indices_from(M)] += np.where(locked[k], 0.0, a * (pf[k] * sc[k] ** 2))
condM = float(np.linalg.cond(M))


def stats(x, ref):
    d = np.abs(x - ref)
    sup = np.abs(ref) > 0
    rel = d[sup] / np.abs(ref[sup])
    j = int(np.flatnonzero(sup)[np.argmax(rel)])
    return dict(
        normwise=float(d.max() / np.abs(ref).max()),
        worst_coef_rel=float(rel.max()),
        worst_coef=names[j],
        worst_coef_abs_over_max=float(abs(ref[j]) / np.abs(ref).max()),
        n_coef_rel_gt_1e8=int((rel > 1e-8).sum()),
        gamma_space_normwise=float(np.abs((x - ref) * sc).max() / np.abs(ref * sc).max()),
    )


rep = dict(
    origin=o,
    alpha=a,
    cond_standardized_system=cond,
    cond_original_units_system=condM,
    walk_vs_explicit=stats(th, beta_e),
    walk_vs_refined=stats(th, beta_r),
    explicit_vs_refined=stats(beta_e, beta_r),
)
print(json.dumps(rep, indent=1))
Path(sys.argv[2]).write_text(json.dumps(rep, indent=1))
