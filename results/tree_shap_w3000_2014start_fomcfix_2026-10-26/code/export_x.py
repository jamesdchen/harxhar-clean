"""The design values of the 33,970 forecast rows (design rows 50,777 .. 84,746), float32, for the M3 dependence bins
-> WORK/x_forecast_rows.npz (x, r0, md5 of the design X.npy's MD5SUMS line)."""

import numpy as np

import shap_common as C

o = C.origins()
r0, r1 = int(o["r"].iloc[0]), int(o["r_end"].iloc[-1])
X = np.load(C.DESIGN / "_work" / "design_all_features" / "X.npy", mmap_mode="r")
x = np.ascontiguousarray(X[r0:r1], dtype=np.float64)
assert x.shape == (33970, 564) and np.isfinite(x).all()
C.WORK.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    C.WORK / "x_forecast_rows.tmp.npz",
    x=x.astype(np.float32),
    r0=r0,
    r1=r1,
    exact_float32=np.array_equal(x.astype(np.float32).astype(np.float64), x),
)
(C.WORK / "x_forecast_rows.tmp.npz").replace(C.WORK / "x_forecast_rows.npz")
C.say(f"x_forecast_rows: {x.shape}, rows {r0}..{r1 - 1}")
