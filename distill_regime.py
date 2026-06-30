"""Distillation test: can explicit HAR×intraday-regime features in the LINEAR enet
capture what the residualized EBM learned? Compare enet base vs enet+regime to the
EBM best (0.12414). Two encodings: full (HAR×6 buckets, L1-pruned) and minimal
(HAR × single late-day indicator = the bucket-4 sign-flip)."""
import json
import time

import numpy as np

import resid_amortized as ra

cid = "ebm_all_buckets_tw1000_enet_rf480_slim"
c = ra.load_cache(cid)
Xs, y, base = c["Xs"], c["y"], c["base"]
tw = int(c["cell"]["train_win"])
refit = int(c["cell"]["refit"])
N, p = Xs.shape
feats = json.load(open(f"results/covid_imp_rank/{c['cell']['bucket']}/meta.json"))["feats"]
hour = Xs[:, feats.index("hour")].astype(float)
qs = np.quantile(hour, [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])
hb = np.clip(np.searchsorted(qs, hour), 0, 5)
B6 = np.eye(6)[hb]                                   # N x 6 bucket dummies
lateday = (hb == 4).astype(float)[:, None]          # N x 1 late-day indicator

HARS = ["har_ma_1", "har_ma_5", "har_ma_25", "har_ma_125", "har_ma_625", "har_ma_3125"]
HAR = Xs[:, [feats.index(h) for h in HARS]].astype(float)   # N x 6


def build(D):
    M = (HAR[:, :, None] * D[:, None, :]).reshape(N, -1)
    return np.ascontiguousarray(M[:, M.std(0) > 1e-9])


INT_full = build(B6)         # HAR x 6 buckets
INT_min = build(lateday)     # HAR x late-day indicator
aug_full = np.ascontiguousarray(np.hstack([Xs, INT_full]))
aug_min = np.ascontiguousarray(np.hstack([Xs, INT_min]))


def run(M, label):
    t0 = time.time()
    starts, coefs, intercepts = ra._cadence_enet(M, y, tw, refit)
    oos = ra._cadence_ridge_oos(M, tw, starts, coefs, intercepts)
    q = ra._qlike(oos, y, base, tw)
    nz = (np.abs(coefs[:, p:]) > 0).sum(1).mean() if M.shape[1] > p else 0.0
    print(f"{label:34s} qlike={q:.5f}  ({time.time() - t0:.0f}s)  nonzero_regime_coefs/refit={nz:.1f}", flush=True)
    return q


print("=== distill intraday regime into the linear base ===", flush=True)
print("(enet base=0.12516 ; residualized-EBM best=0.12414)\n", flush=True)
qb = run(Xs, "enet base")
qm = run(aug_min, f"enet + HAR x late-day ({INT_min.shape[1]})")
qf = run(aug_full, f"enet + HAR x 6 regimes ({INT_full.shape[1]})")
print(f"\nbase={qb:.5f}  +lateday={qm:.5f}  +6regimes={qf:.5f}", flush=True)
print(f"best regime - base = {min(qm, qf) - qb:+.6f}", flush=True)
print(f"gap to EBM (0.12414): linear-regime best is {min(qm, qf) - 0.12414:+.6f} away", flush=True)
