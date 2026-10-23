"""The warm path's active-set system at a 2016 origin, flag off vs on (full-set lasso / enet, the frozen penalties): the
active-set size, how many active columns have a small window s.d., and the condition number of the matrix the online
homotopy solves (the uncentred Gram of the active columns and the intercept, + the L2 term).

usage: python diag_cond.py <flag-on frozen_penalties.json> <flag-off frozen_penalties.json> <out json>
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

fon = json.loads(Path(sys.argv[1]).read_text())
foff = json.loads(Path(sys.argv[2]).read_text())
P.SAMPLE, P.TW_DAYS, P.PERIODS_PER_DAY, P.W, P.HAR_BASE = "rth13", 1000, 13, 13000, "5"
P.EX_FLAGS = dict(
    DROP_ACTIVITY_MA=True,
    CALENDAR_THROUGH_OVERNIGHT=True,
    DROP_OVERNIGHT_INDICATOR=True,
    LEAN_GLOBAL_LAGS=True,
    LATE_START_WARMUP=True,
)
ns = P.spec_section("reclasso", "all_features", 0, -1, 0)
meta = json.loads((DZ / "design.json").read_text())
bb = P.backbone_mask(meta["names"])
X = np.load(DZ / "X.npy", mmap_mode="r")
y = np.load(DZ / "y.npy")
W = 13000
p = X.shape[1]
locked = np.zeros(p + 1, dtype=bool)
locked[-1] = True
pf = np.ones(p + 1)
pf[:-1][bb] = 0.01
rep: dict = {}
for o in (44250 + 250 * 60, 44250 + 250 * 120):
    Xa = np.hstack([np.asarray(X[o - W : o]), np.ones((W, 1))])
    probe = ns["RollingTunedLinear"]()
    probe._locked = locked
    probe._recompute_mask(Xa)
    Xa[:, probe._maskout] = 0.0
    yr = y[o - W : o]
    s = np.std(Xa, axis=0)
    for m, l1 in (("lasso", 1.0), ("enet", 0.5)):
        for tag, fr, std in (("off", foff, False), ("on", fon, True)):
            a = float(fr[m]["all_features"])
            if std:
                th = P.fit_std(Xa, yr, locked, m, a, l1, pf, P.evid_ref(meta["names"], bb))[0]
                sw = np.where(locked, 1.0, np.where(s < P.STD_MIN, 1.0, s))
            else:
                th = P.batch_theta_pf(ns, Xa, yr, locked, a, l1, pf)
                sw = np.ones(p + 1)
            A = np.flatnonzero((th != 0) | locked)
            lam2 = W * a * (1.0 - l1)
            ei = ~locked[A]
            Mc = None
            for cen in (False, True):  # the uncentred state (before addendum 2's centring) and the centred one
                XA = Xa[:, A] - (np.where(locked[A], 0.0, Xa[:, A].mean(axis=0)) if cen else 0.0)
                M = XA.T @ XA
                M[np.flatnonzero(ei), np.flatnonzero(ei)] += lam2 * (pf[A] * sw[A] * sw[A])[ei]
                if cen:
                    Mc = M
                else:
                    Mu = M
            sa = s[A][ei]
            low = [(meta["names"][j], round(float(s[j]), 5), int(Xa.shape[0] - P.col_mode(Xa[:, [j]])[1][0])) for j in A[:-1] if s[j] < 0.05]
            rep[f"{o}/{m}/{tag}"] = dict(low_s_active=low,
                alpha=a,
                n_active=int(len(A) - 1),
                n_active_s_lt_0p05=int((sa < 0.05).sum()),
                n_active_s_lt_0p01=int((sa < 0.01).sum()),
                min_active_s=float(sa.min()),
                cond_active_system_uncentred=float(np.linalg.cond(Mu)),
                cond_active_system_centred=float(np.linalg.cond(Mc)),
            )
            print(o, m, tag, rep[f"{o}/{m}/{tag}"], flush=True)
Path(sys.argv[3]).write_text(json.dumps(rep, indent=1))
