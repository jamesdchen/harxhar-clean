"""How many of the 6 HAR windows flip sign at clock bucket 4 (late-day regime)?"""

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)
import json

import numpy as np

import resid_amortized as ra

cid = "ebm_all_buckets_tw1000_enet_rf480_slim"
c = ra.load_cache(cid)
Xs, y = c["Xs"], c["y"]
tw = int(c["cell"]["train_win"])
ridge_oos = np.asarray(c["ridge_oos"], float)
feats = json.load(open(f"results/covid_imp_rank/{c['cell']['bucket']}/meta.json"))["feats"]
hidx = feats.index("hour")
n_oos = len(ridge_oos)
sl = slice(tw, tw + n_oos)
hour = Xs[sl, hidx]
resid = y[sl] - ridge_oos
hb = np.searchsorted(np.quantile(hour, [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6]), hour)

HARS = ["har_ma_1", "har_ma_5", "har_ma_25", "har_ma_125", "har_ma_625", "har_ma_3125"]
print("HAR window  | corr(feature, residual) by clock bucket 0..5            | bucket4  others  flip?", flush=True)
nflip = 0
for name in HARS:
    if name not in feats:
        print(f"  {name}: NOT in matrix")
        continue
    x = Xs[sl, feats.index(name)].astype(float)
    sp = []
    for b in range(6):
        m = hb == b
        sp.append(float(np.corrcoef(x[m], resid[m])[0, 1]) if m.sum() > 10 else np.nan)
    sp = np.array(sp)
    others = float(np.nanmean([sp[k] for k in range(6) if k != 4]))
    flip = (np.sign(sp[4]) != np.sign(others)) and abs(sp[4]) > 0.01 and abs(others) > 0.005
    nflip += int(flip)
    print(f"  {name:11s} [{', '.join(f'{s:+.3f}' for s in sp)}]   {sp[4]:+.3f}  {others:+.3f}  {'FLIP' if flip else '-'}", flush=True)
print(f"\n=> {nflip}/6 HAR windows flip sign at bucket 4 (late-day regime)", flush=True)
