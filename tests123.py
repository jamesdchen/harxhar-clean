"""Mechanism tests for the late-day HAR reversal:
 1) does it scale with voldemand (hedging-demand proxy)? -> gamma/hedging channel
 2) is it concentrated in the LAST slot (auction/MOC) or gradual (gamma)?
 3) is it stronger at month/quarter-end (index rebalancing)?"""
import json

import numpy as np

import resid_amortized as ra

cid = "ebm_all_buckets_tw1000_enet_rf480_slim"
c = ra.load_cache(cid)
Xs, y = c["Xs"], c["y"]
tw = int(c["cell"]["train_win"])
ridge_oos = np.asarray(c["ridge_oos"], float)
feats = json.load(open(f"results/covid_imp_rank/{c['cell']['bucket']}/meta.json"))["feats"]
sl = slice(tw, tw + len(ridge_oos))
hour = Xs[sl, feats.index("hour")]
resid = y[sl] - ridge_oos
H = np.unique(hour)
har5 = Xs[sl, feats.index("har_ma_5")].astype(float)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 30 else np.nan


vd_feats = [f for f in feats if "voldemand" in f]
cal_feats = [f for f in feats if any(k in f.lower() for k in
             ["month", "dom", "eom", "quarter", "_eofm", "monthend", "turn", "day_of", "calendar"])]
print(f"voldemand feats ({len(vd_feats)}):", vd_feats[:6], flush=True)
print("calendar/date-ish feats:", cal_feats[:20], flush=True)

# TEST 2: fine hour profile
print("\n=== TEST 2: corr(har_ma_5, residual) by hour slot ===", flush=True)
prof = [(h, corr(har5[hour == h], resid[hour == h]), int((hour == h).sum())) for h in H]
for h, co, nn in prof:
    flag = " <== NEG" if co < -0.02 else ""
    print(f"  hour {h:6.3f}: {co:+.4f} (n={nn}){flag}", flush=True)
neg_hours = [h for h, co, _ in prof if co < -0.02]
print("flip hours (corr<-0.02):", [f"{h:.2f}" for h in neg_hours], flush=True)
last = H[-1]
print(f"last slot ({last:.2f}) corr={dict((h, co) for h, co, _ in prof)[last]:+.4f}", flush=True)

# TEST 1: voldemand conditioning
print("\n=== TEST 1: does the late-day reversal scale with voldemand? ===", flush=True)
vd_close = next((f for f in vd_feats if "open_and_close" in f and f.endswith("_ma_5")), vd_feats[0] if vd_feats else None)
print("voldemand feature:", vd_close, flush=True)
if vd_close:
    vd = Xs[sl, feats.index(vd_close)].astype(float)
    late = np.isin(hour, neg_hours)
    for region, mask in [("LATE (flip)", late), ("REST", ~late)]:
        med = np.median(vd[mask])
        hi, lo = mask & (vd > med), mask & (vd <= med)
        print(f"  {region:11s}: high-voldemand corr={corr(har5[hi], resid[hi]):+.4f}  "
              f"low-voldemand corr={corr(har5[lo], resid[lo]):+.4f}", flush=True)

# TEST 3: month/quarter-end
print("\n=== TEST 3: rebalance days (month/quarter-end) ===", flush=True)
me_feat = next((f for f in cal_feats if any(k in f.lower() for k in ["eom", "month_end", "monthend", "quarter", "turn"])), None)
if me_feat:
    print("using:", me_feat, flush=True)
    me = Xs[sl, feats.index(me_feat)].astype(float)
    late = np.isin(hour, neg_hours)
    thr = np.quantile(me, 0.8)
    for region, mask in [("LATE (flip)", late), ("REST", ~late)]:
        rb, nr = mask & (me > thr), mask & (me <= thr)
        print(f"  {region:11s}: rebalance-day corr={corr(har5[rb], resid[rb]):+.4f}  "
              f"normal-day corr={corr(har5[nr], resid[nr]):+.4f}", flush=True)
else:
    print("No month-end/quarter-end feature in the slim matrix.", flush=True)
    print("(all_buckets exog = moments+sentiment+impliedvol+voldemand + HAR + DOW; no day-of-month.)", flush=True)
    print("Test 3 needs the calendar date index, which isn't in the cache — would require re-deriving dates.", flush=True)
