
"""Learned-smear probe: can anything beat the EWMA second moment OOS?
Walk-forward, YEARLY refits, causal, level-pinned like variance_sidecar.
Models for z_t = log e_t^2:
  ewma      baseline: zhat = log s2_ewma(t)
  ridge     standardized [log lev, lev, log s2, z_{t-1..10}]
  lgbm      same features (flexible tabular stand-in)
  lstm      1-layer LSTM h=32 on the trailing 50-bar z-residual sequence,
            features per step: [z - log s2] only (pure residual-sequence signal)
Scoring: v = exp(zhat)*pin, pin per year = mean e^2 / mean exp(zhat) on TRAIN;
forecast (m^2 + v) B; pooled per-bar QLIKE on scored bars (skip year 1 = burn-in).
"""
import numpy as np
import time
e = np.load("/u/scratch/j/jamesdc1/harxhar-clean/results/a0_e.npy"); m = np.load("/u/scratch/j/jamesdc1/harxhar-clean/results/a0_m.npy"); s2 = np.load("/u/scratch/j/jamesdc1/harxhar-clean/results/a0_s2ewma.npy")
# need rv pairs per bar: rebuild quickly = (m^2 + v)*B vs rv_raw; B and rv_raw
# were not saved; reconstruct: y_raw = e + m, so rv = y_raw^2 * B. B cancels
# in QLIKE if we evaluate in sqrt space?? No — QLIKE needs raw pairs. But
# (m^2+v)B vs (e+m)^2 B: B>0 cancels in the RATIO rv/f = (e+m)^2/(m^2+v)!
# QLIKE = ratio - log ratio - 1, so B drops out entirely. Score in B-free form.
y = e + m
def qlike_v(v):
    f = m**2 + v
    r = (y**2) / f
    return float(np.mean(r - np.log(r) - 1.0))
e2 = e**2
floor = np.min(e2[e2>0])
z = np.log(np.maximum(e2, floor))
lev = m**2
logs2 = np.log(s2)
N = len(z)
BARS_PER_YEAR = N // 24
bounds = [(i*BARS_PER_YEAR, min((i+1)*BARS_PER_YEAR, N)) for i in range(24)]
bounds[-1] = (bounds[-1][0], N)

def feats(i0, i1):
    cols = [np.log(np.maximum(lev[i0:i1],1e-12)), lev[i0:i1], logs2[i0:i1]]
    for L in range(1, 11):
        zl = np.full(i1-i0, np.nan); zl[L:] = z[i0:i1-L]
        cols.append(zl)
    return np.column_stack(cols)

ZH = {"ewma": logs2.copy(), "ridge": np.full(N,np.nan), "lgbm": np.full(N,np.nan), "lstm": np.full(N,np.nan)}
t0=time.time()
import torch
import torch.nn as nn
torch.manual_seed(0)
for k in range(1, 24):
    tr0, tr1 = 0, bounds[k-1][1]      # expanding train: everything before year k
    te0, te1 = bounds[k]
    pin_den = None
    # ridge
    X = feats(tr0, tr1); ok = np.all(np.isfinite(X), axis=1)
    mu, sd = X[ok].mean(0), X[ok].std(0)+1e-12
    Xs = (X[ok]-mu)/sd
    lam_r = 1e2
    beta = np.linalg.solve(Xs.T@Xs + lam_r*np.eye(Xs.shape[1]), Xs.T@z[tr0:tr1][ok])
    Xte = feats(te0, te1); okt = np.all(np.isfinite(Xte), axis=1)
    zh = np.full(te1-te0, np.nan)
    zh[okt] = ((Xte[okt]-mu)/sd)@beta
    ZH["ridge"][te0:te1] = zh
    # lgbm
    import lightgbm as lgb
    g = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=200,
                          min_child_samples=100, num_threads=4, random_state=0, verbosity=-1)
    g.fit(X[ok], z[tr0:tr1][ok])
    pred_l = np.full(te1-te0, np.nan)
    if okt.any():
        pred_l[okt] = g.predict(Xte[okt])
    ZH["lgbm"][te0:te1] = pred_l
    # lstm on residual sequence r_t = z_t - log s2_t
    r = z - logs2
    SEQ = 50
    def seqs(a, b):
        xs, ys = [], []
        for t in range(max(a, SEQ), b):
            xs.append(r[t-SEQ:t]); ys.append(r[t])
        return np.asarray(xs, dtype=np.float32)[...,None], np.asarray(ys, dtype=np.float32)
    Xs_, Ys_ = seqs(tr0, tr1)
    net = nn.LSTM(1, 32, batch_first=True)
    head = nn.Linear(32, 1)
    opt = torch.optim.Adam(list(net.parameters())+list(head.parameters()), lr=1e-3)
    Xt = torch.from_numpy(Xs_); Yt = torch.from_numpy(Ys_)
    n_ep = 3
    for ep in range(n_ep):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 512):
            idx = perm[i:i+512]
            o, _ = net(Xt[idx])
            loss = ((head(o[:, -1]).squeeze(-1) - Yt[idx])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Xte_s, _ = seqs(te0, te1)
        o, _ = net(torch.from_numpy(Xte_s))
        pred = head(o[:, -1]).squeeze(-1).numpy()
    zh2 = np.full(te1-te0, np.nan)
    off = max(te0, SEQ) - te0
    zh2[off:] = logs2[max(te0,SEQ):te1] + pred
    ZH["lstm"][te0:te1] = zh2
    print(f"year {k+1}/24 done ({time.time()-t0:.0f}s)", flush=True)

np.save("/u/scratch/j/jamesdc1/harxhar-clean/results/lstm_probe_ZH.npy", np.column_stack([ZH[k] for k in ("ewma","ridge","lgbm","lstm")]))
print("\n=== pooled QLIKE (level-pinned, scored bars) ===")
for name, zh in ZH.items():
    ok = np.isfinite(zh)
    mo, yo, eo = m[ok], y[ok], e2[ok]
    v = np.exp(zh[ok])
    pin = eo.mean() / v.mean()
    f = mo**2 + v*pin
    r = (yo**2)/f
    print(f"  {name:6s} {float(np.mean(r-np.log(r)-1.0)):.5f}   (n={ok.sum()})")
