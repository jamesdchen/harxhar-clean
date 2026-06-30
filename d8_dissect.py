"""Dissect the d8 global grey box (depth-8 XGB) into a functional-ANOVA decomposition via EBM distillation.

The cascade's bulk residual work is done by d8, the one stage never made interpretable. Distill its learned
function ĝ(X) into a GAM shadow (interactions=0) and a GA2M shadow (pairwise) on a representative refit
window. Reports:
  R2_additive            -> fraction of ĝ that is 1-D shapes (NOT routing-rich; a GAM suffices)
  R2_pairwise - additive -> interaction structure (ROUTING-RICH; which feature PAIRS form the regime)
  1 - R2_pairwise        -> higher-order remainder (the only part a true router could add over a GA2M)
plus the top main effects and interaction pairs (the readable dissection) and d8's own gain importances.
Also fits the same shadows to r1 directly (what's LEARNABLE) as a reference. Diagnostic, single-window.

Run on Hoffman2 hpc-pi. Env: D8_BLOCK (block index frac, default 0.75), D8_INTER (max interactions, 16)."""

from __future__ import annotations

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
N_INTER = int(os.environ.get("D8_INTER", "16"))
SUB = int(
    os.environ.get("D8_SUB", "12000")
)  # EBM-shadow row subsample (full window is too slow + unneeded)

c = ra.load_cache(CID)
tw = c["cell"]["train_win"]
feats = c["feats"]
starts = c["starts"]
gcfg = json.loads(os.environ.get("GLOBAL_CFG", "{}"))
if feats is None:
    raise SystemExit("need aligned feats (cell feats.json) for readable names")

i = max(0, min(len(starts) - 1, int(len(starts) * BLOCK_FRAC)))
t_r = int(starts[i])
cols = c["masks"][i]
colnames = [f for f, m in zip(feats, cols) if m]
Xtr = c["Xs"][t_r - tw : t_r][:, cols].astype(np.float64)
r1 = (
    c["y"][t_r - tw : t_r]
    - (c["Xs"][t_r - tw : t_r] @ c["coefs"][i] + c["intercepts"][i])
).astype(np.float64)
print(
    f"block i={i}/{len(starts)} t_r={t_r} tw={tw} n_cols={Xtr.shape[1]} "
    f"var(r1)={r1.var():.3e} gcfg_depth={gcfg.get('max_depth')}",
    flush=True,
)


def r2(yhat, y):
    return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


# the actual d8 global stage on this block
g = ra._tree_factory("xgb", gcfg)()
g.fit(Xtr, r1)
ghat = g.predict(Xtr).astype(np.float64)
print(
    f"d8 fit: var(ghat)={ghat.var():.3e}  in-sample R2(ghat,r1)={r2(ghat, r1):.4f}",
    flush=True,
)

# subsample rows for the EBM shadows (full window is slow and unneeded to capture d8's function)
rng = np.random.default_rng(0)
sidx = (
    rng.choice(len(Xtr), SUB, replace=False) if len(Xtr) > SUB else np.arange(len(Xtr))
)
Xsub = Xtr[sidx]
print(f"EBM shadows on {len(sidx)} subsampled rows, interactions={N_INTER}", flush=True)


def distill(target, label):
    tsub = target[sidx]
    gam = EBR(interactions=0, feature_names=colnames, n_jobs=4, random_state=42)
    gam.fit(Xsub, tsub)
    ga2m = EBR(interactions=N_INTER, feature_names=colnames, n_jobs=4, random_state=42)
    ga2m.fit(Xsub, tsub)
    ra_add = r2(gam.predict(Xsub), tsub)
    ra_pair = r2(ga2m.predict(Xsub), tsub)
    print(
        f"D8DISSECT[{label}] R2_additive={ra_add:.4f} R2_pairwise={ra_pair:.4f} "
        f"pairwise_gain={ra_pair - ra_add:.4f} higher_order={1 - ra_pair:.4f}",
        flush=True,
    )
    return ga2m


# (1) distill what d8 LEARNED; (2) fit r1 directly = what's LEARNABLE (reference)
ga2m = distill(ghat, "ghat=what_d8_learned")
distill(r1, "r1=what_is_learnable")

# readable dissection: top main effects + interaction pairs from the ĝ shadow
imps = np.asarray(ga2m.term_importances())
names = list(ga2m.term_names_)
order = np.argsort(imps)[::-1]
mains = [(names[j], imps[j]) for j in order if " & " not in names[j]][:10]
inters = [(names[j], imps[j]) for j in order if " & " in names[j]][:10]
print("TOP MAIN EFFECTS (not routing-rich):", flush=True)
for nm, im in mains:
    print(f"   {im:8.4e}  {nm}", flush=True)
print("TOP INTERACTION PAIRS (routing-rich = where a regime/gate lives):", flush=True)
for nm, im in inters:
    print(f"   {im:8.4e}  {nm}", flush=True)

# d8's own gain importances (cross-check)
try:
    fi = np.asarray(g.feature_importances_)
    top = np.argsort(fi)[::-1][:10]
    print("d8 XGB gain importances (top 10):", flush=True)
    for j in top:
        print(f"   {fi[j]:8.4e}  {colnames[j]}", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"(xgb importances unavailable: {e})", flush=True)
print("D8_DISSECT_DONE", flush=True)
