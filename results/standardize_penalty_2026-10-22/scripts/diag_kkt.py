"""Why the standardized lasso walk fails the KKT check often: a short flag-on full-set lasso walk from a 2016 origin,
recording at every KKT failure the violation (relative to max(max|c|, 1)), the violating column, its s at the seed,
its weight, whether it is active, and |theta_j|.

usage: python diag_kkt.py <frozen_penalties.json of the flag-on select> <out json> [origin] [solves] [std 0|1]
"""

from __future__ import annotations

import json
import sys
import time
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
OUTJ = Path(sys.argv[2])
o = int(sys.argv[3]) if len(sys.argv) > 3 else 44250 + 250 * 60
NS = int(sys.argv[4]) if len(sys.argv) > 4 else 250
STD = (sys.argv[5] == "1") if len(sys.argv) > 5 else True
P.SAMPLE, P.TW_DAYS, P.PERIODS_PER_DAY, P.W, P.HAR_BASE = "rth13", 1000, 13, 13000, "5"
P.EX_FLAGS = dict(
    DROP_ACTIVITY_MA=True,
    CALENDAR_THROUGH_OVERNIGHT=True,
    DROP_OVERNIGHT_INDICATOR=True,
    LEAN_GLOBAL_LAGS=True,
    LATE_START_WARMUP=True,
)
ns = P.spec_section("reclasso", "all_features", 0, -1, 0)
import src.models.reclasso_har as RH  # noqa: E402

P.STANDARDIZE_PENALTY = STD
meta = json.loads((DZ / "design.json").read_text())
names = meta["names"] + ["intercept"]
bb = P.backbone_mask(meta["names"])
X = np.load(DZ / "X.npy", mmap_mode="r")
y = np.load(DZ / "y.npy")
W = 13000
a = float(fr["lasso"]["all_features"])
Base = ns["RollingTunedLinear"]
Base.grid = [("lasso", a, 1.0)]
Base.trace, Base.mask_trace, Base.reseed_trace = [], [], []
Cls = P.make_class(ns, 0.01, P.ConstSched(a), o - W, [])
Cls.bb = bb
Cls.kkt_rows, Cls.kkt_reseeds, Cls.kkt_sec = [], [], [0.0, 0.0]
Cls.std_seeds, Cls.std_ridge, Cls.std_blockend = [], [], []
fails: list = []


class Diag(Cls):
    def _kkt_after(self, warm):
        if warm and RH.WALK_KKT_RESEED:
            ok, v, sc = RH.walk_kkt(self._Gr, self._c, self._mu_vec, self._th)
            if not ok:
                r = self._c - self._Gr @ self._th
                on = self._th != 0.0
                viol = np.where(on, np.abs(r - self._mu_vec * np.sign(self._th)), np.abs(r) - self._mu_vec)
                j = int(np.argmax(viol))
                sd = getattr(self, "_sd_seed", np.full(len(r), np.nan))
                fails.append(
                    dict(
                        solve=int(self._n_solve - 1),
                        rel=float(v / sc),
                        col=names[j],
                        active=bool(on[j]),
                        theta=float(self._th[j]),
                        s_seed=float(sd[j]) if np.isfinite(sd[j]) else None,
                        mu_j=float(self._mu_vec[j]),
                        n_active=int(on.sum()),
                        n_viol=int((viol > RH.KKT_TOL_REL * sc).sum()),
                    )
                )
        super()._kkt_after(warm)


bt = ns["MultiStageBacktest"](residualizer=ns["IdentityResidualizer"](), regressor_factory=Diag, refit_frequency=1)
t0 = time.time()
bt.run(np.asarray(X[o - W : o + NS]), y[o - W : o + NS], W, desc="d")
rep = dict(
    origin=o,
    solves=NS,
    standardize=STD,
    alpha=a,
    sec=time.time() - t0,
    kkt_reseeds=len(Cls.kkt_reseeds),
    kkt_sec=Cls.kkt_sec,
    fails=fails,
    rel_quantiles=[float(v) for v in np.quantile([f["rel"] for f in fails], [0.0, 0.5, 0.9, 1.0])] if fails else [],
)
OUTJ.write_text(json.dumps(rep, indent=1))
print(json.dumps({k: v for k, v in rep.items() if k != "fails"}, indent=1))
for f in fails[:40]:
    print(f)
