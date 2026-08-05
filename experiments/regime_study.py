"""Intraday-regime test battery on the OOS residual (y - enet), with a temporal
train/test split so the R2s are honest. (1) control out the leftover U-shape main
effect (hour dummies); (2) do CLOCK-regime interactions (predictor x hour-bucket)
add OOS structure beyond that?; (3) STATE-regime (RV vs diurnal norm) vs clock."""

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)
import json

import numpy as np
from sklearn.linear_model import Ridge

import resid_amortized as ra

cid = "ebm_all_buckets_tw1000_enet_rf480_slim"
c = ra.load_cache(cid)
Xs, y = c["Xs"], c["y"]
tw = int(c["cell"]["train_win"])
ridge_oos = np.asarray(c["ridge_oos"], float)
coefs = c["coefs"]
feats = json.load(open(f"results/covid_imp_rank/{c['cell']['bucket']}/meta.json"))["feats"]
hidx = feats.index("hour")
h1idx = feats.index("har_ma_1")

n_oos = len(ridge_oos)
sl = slice(tw, tw + n_oos)
hour = Xs[sl, hidx]
resid = y[sl] - ridge_oos
N = len(resid)
H = np.unique(hour)

# top-10 linearly-important predictors (mean |enet coef| across windows), standardized
mabs = np.abs(coefs).mean(0)
topk = np.argsort(mabs)[::-1][:10]
print("top predictors:", [feats[j] for j in topk], flush=True)
P = Xs[sl][:, topk].astype(float)
P = (P - P.mean(0)) / (P.std(0) + 1e-9)

# main-effect control: full hour dummies (removes leftover U)
Hd = (hour[:, None] == H[None, :]).astype(float)
# clock regime: 6 hour-buckets
hb = np.searchsorted(np.quantile(hour, [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]), hour)
Dclock = np.eye(6)[np.clip(hb, 0, 5)]
# state regime: har_ma_1 relative to its diurnal-typical level -> terciles
h1 = Xs[sl][:, h1idx].astype(float)
dm = {hh: h1[hour == hh].mean() for hh in H}
h1norm = h1 - np.array([dm[hh] for hh in hour])
st = np.searchsorted(np.quantile(h1norm, [1 / 3, 2 / 3]), h1norm)
Dstate = np.eye(3)[np.clip(st, 0, 2)]


def inter(D):
    return (P[:, :, None] * D[:, None, :]).reshape(N, -1)


cut = int(0.6 * N)


def r2_oos(F):
    m = Ridge(alpha=1.0).fit(F[:cut], resid[:cut])
    pr = m.predict(F[cut:])
    rr = resid[cut:]
    return 1.0 - np.sum((rr - pr) ** 2) / np.sum((rr - rr.mean()) ** 2)


print(f"\nresidual std={resid.std():.4f}  N={N}  test-half OOS R2 on residual:", flush=True)
print(f"  hour dummies (leftover U main effect)   : {r2_oos(Hd):+.5f}", flush=True)
print(f"  hour dummies + CLOCK-regime interactions: {r2_oos(np.hstack([Hd, inter(Dclock)])):+.5f}", flush=True)
print(f"  hour dummies + STATE-regime interactions: {r2_oos(np.hstack([Hd, inter(Dstate)])):+.5f}", flush=True)
print(f"  CLOCK interactions alone                : {r2_oos(inter(Dclock)):+.5f}", flush=True)
print(f"  STATE interactions alone                : {r2_oos(inter(Dstate)):+.5f}", flush=True)

# regime-conditional slopes: how does each top predictor's residual-slope vary across clock regimes?
print("\nper-CLOCK-regime residual-slope (corr) for top predictors (variation => regime structure):", flush=True)
for k in range(min(6, len(topk))):
    slopes = []
    for b in range(6):
        mb = Dclock[:, b] > 0
        x = P[mb, k]
        r = resid[mb]
        slopes.append(float(np.corrcoef(x, r)[0, 1]) if mb.sum() > 10 else np.nan)
    sp = np.array(slopes)
    print(f"  {feats[topk[k]][:34]:34s} corr-by-regime: [{', '.join(f'{s:+.3f}' for s in sp)}]  spread={np.nanmax(sp)-np.nanmin(sp):.3f}", flush=True)
