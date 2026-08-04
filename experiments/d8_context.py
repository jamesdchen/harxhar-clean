"""Context-collapse analysis: can d8's 36% higher-order be decomposed into context-routed PAIRWISE pieces?

For each candidate context variable c, bin d8's learned function ghat(X) by c and fit a per-bin GA2M shadow.
If conditioning on c collapses the higher-order (per-bin pairwise R2 >> global pairwise R2 = 0.64), then
routing on c turns the 3+-way structure into exploitable pairwise -> a shallow gate on c + pairwise experts
could capture it (legible + context-aware). A RANDOM-bin placebo controls for the extra per-bin flexibility:
the real signal is per-bin R2(c) - per-bin R2(random). All c ~ random => higher-order is diffuse (needs depth).

Run on Hoffman2 hpc-pi. Env: D8_BLOCK (0.75), D8_SUB (12000), D8_INTER (8), D8_QBINS (4)."""

from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)

import json
import os
import sys

import numpy as np

import resid_amortized as ra
from interpret.glassbox import ExplainableBoostingRegressor as EBR

CID = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim"
)
BLOCK_FRAC = float(os.environ.get("D8_BLOCK", "0.75"))
SUB = int(os.environ.get("D8_SUB", "12000"))
N_INTER = int(os.environ.get("D8_INTER", "8"))
Q = int(os.environ.get("D8_QBINS", "4"))

c = ra.load_cache(CID)
tw, feats, starts = c["cell"]["train_win"], c["feats"], c["starts"]
gcfg = json.loads(os.environ.get("GLOBAL_CFG", "{}"))
i = max(0, min(len(starts) - 1, int(len(starts) * BLOCK_FRAC)))
t_r = int(starts[i])
cols = c["masks"][i]
colnames = [f for f, m in zip(feats, cols) if m]
Xtr = c["Xs"][t_r - tw : t_r][:, cols].astype(np.float64)
r1 = (
    c["y"][t_r - tw : t_r]
    - (c["Xs"][t_r - tw : t_r] @ c["coefs"][i] + c["intercepts"][i])
).astype(np.float64)

g = ra._tree_factory("xgb", gcfg)()
g.fit(Xtr, r1)
ghat = g.predict(Xtr).astype(np.float64)

rng = np.random.default_rng(0)
sidx = (
    rng.choice(len(Xtr), SUB, replace=False) if len(Xtr) > SUB else np.arange(len(Xtr))
)
X, gh = Xtr[sidx], ghat[sidx]
name2idx = {n: k for k, n in enumerate(colnames)}
print(f"block i={i} n={len(sidx)} cols={X.shape[1]} Q={Q} inter={N_INTER}", flush=True)


def r2(yhat, y):
    return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


# global GA2M reference (no context) — the dissection's 0.64
glob = EBR(interactions=N_INTER, n_jobs=4, random_state=42).fit(X, gh)
R2_GLOBAL = r2(glob.predict(X), gh)
print(f"GLOBAL GA2M R2={R2_GLOBAL:.4f}  higher_order={1 - R2_GLOBAL:.4f}", flush=True)


def perbin_r2(cvals):
    edges = np.quantile(cvals, np.linspace(0, 1, Q + 1)[1:-1])
    b = np.digitize(cvals, edges)
    pred = np.empty_like(gh)
    for k in range(Q):
        m = b == k
        if m.sum() < 300:
            pred[m] = gh[m].mean() if m.any() else 0.0
            continue
        pred[m] = (
            EBR(interactions=N_INTER, n_jobs=4, random_state=42)
            .fit(X[m], gh[m])
            .predict(X[m])
        )
    return r2(pred, gh)


# placebo: random context (same Q bins, no structure) — controls for per-bin flexibility
R2_PLACEBO = perbin_r2(rng.standard_normal(len(gh)))
print(f"PLACEBO (random bins) per-bin R2={R2_PLACEBO:.4f}", flush=True)

candidates = [
    "hour",
    "har_ma_125",
    "har_ma_1",
    "har_ma_5",
    "adj_stocktwits_sentiment_ma_1",
    "adj_stocktwits_attention_ma_1",
    "adj_sumret_ma_1",
    "adj_vix_ma_125",
]
print(
    "CONTEXT-COLLAPSE  (signal = per-bin R2 - placebo; >0 => c decomposes higher-order into routable pairwise)",
    flush=True,
)
rows = []
for cname in candidates:
    if cname not in name2idx:
        continue
    rb = perbin_r2(X[:, name2idx[cname]])
    rows.append((cname, rb, rb - R2_PLACEBO, 1 - rb))
for cname, rb, sig, ho in sorted(rows, key=lambda z: -z[2]):
    print(
        f"  CTX {cname:32} perbin_R2={rb:.4f}  vs_placebo={sig:+.4f}  higher_order_left={ho:.4f}",
        flush=True,
    )
print("D8_CONTEXT_DONE", flush=True)
