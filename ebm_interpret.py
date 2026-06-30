"""Fit the best-config EBM on a representative window's enet-residual (resid_subset)
and extract the learned representation: importance ranking (vs enet linear coef),
per-feature shape functions with cross-bag stability bands, and interaction pairs."""
import json
import time

import numpy as np
import pandas as pd

import resid_amortized as ra

cid = "ebm_all_buckets_tw1000_enet_rf480_slim"
cfg = {"learning_rate": 0.04815472381591801, "max_leaves": 4, "interactions": 5,
       "max_bins": 680, "min_samples_leaf": 5, "max_rounds": 100, "outer_bags": 4}
c = ra.load_cache(cid)
Xs, y = c["Xs"], c["y"]
tw = int(c["cell"]["train_win"])
starts, coefs, intercepts, masks = c["starts"], c["coefs"], c["intercepts"], c["masks"]
feats = json.load(open(f"results/covid_imp_rank/{c['cell']['bucket']}/meta.json"))["feats"]

i = len(starts) // 2                          # representative mid-backtest window
t_r = int(starts[i])
Xtr = Xs[t_r - tw:t_r]
r_train = y[t_r - tw:t_r] - (Xtr @ coefs[i] + intercepts[i])   # enet residual (what the tree fits)
cols = masks[i]
surv_idx = np.where(cols)[0]
sub_names = [feats[j] for j in surv_idx]
print(f"window i={i}/{len(starts)} t_r={t_r} survivors={len(sub_names)} resid_std={r_train.std():.5f}", flush=True)

df = pd.DataFrame(Xtr[:, cols], columns=sub_names)
ebm = ra._tree_factory("ebm", cfg)()
t0 = time.time()
ebm.fit(df, r_train)
print(f"EBM fit {time.time() - t0:.0f}s; terms={len(ebm.term_names_)}", flush=True)

imps = np.asarray(ebm.term_importances())
tnames = list(ebm.term_names_)
tfeat = list(ebm.term_features_)
order = np.argsort(imps)[::-1]

# enet |coef| rank among survivors (linear importance) for the same window
surv_enet = np.abs(coefs[i])[surv_idx]
enet_rank = {sub_names[k]: int((surv_enet > surv_enet[k]).sum()) + 1 for k in range(len(surv_idx))}

print("\n=== TOP 15 TERMS by EBM importance ===", flush=True)
print(f"{'term':38s} {'ebm_imp':>9s} {'kind':>5s} {'enet|c|rank':>11s}", flush=True)
for o in order[:15]:
    kind = "PAIR" if len(tfeat[o]) == 2 else "main"
    er = enet_rank.get(tnames[o], "-") if kind == "main" else f"/{len(surv_idx)}"
    print(f"{tnames[o][:38]:38s} {imps[o]:9.5f} {kind:>5s} {str(er):>11s}", flush=True)

# shape functions + cross-bag bands for top main terms
glob = ebm.explain_global()
save = {}
main_terms = [o for o in order if len(tfeat[o]) == 1][:8]
print("\n=== shape-function stability (top main terms; SNR=amp/cross-bag band) ===", flush=True)
print(f"{'feature':38s} {'amp':>8s} {'band':>8s} {'SNR':>6s}", flush=True)
for o in main_terms:
    d = glob.data(o)
    sc = np.asarray(d["scores"], dtype=float)
    up = np.asarray(d.get("upper_bounds", sc), dtype=float)
    lo = np.asarray(d.get("lower_bounds", sc), dtype=float)
    amp = float(np.abs(sc).max())
    band = float(np.mean(up - lo) / 2)
    print(f"{tnames[o][:38]:38s} {amp:8.5f} {band:8.5f} {amp / (band + 1e-12):6.2f}", flush=True)
    save[f"{tnames[o]}__s"] = sc
    save[f"{tnames[o]}__u"] = up
    save[f"{tnames[o]}__l"] = lo

print("\n=== interaction PAIRS (the event/joint structure) ===", flush=True)
for o in order:
    if len(tfeat[o]) == 2:
        print(f"  {tnames[o]:50s} imp={imps[o]:.5f}", flush=True)

np.savez("results/resid_ab/ebm_repr.npz", **save)
print("\nSAVED results/resid_ab/ebm_repr.npz", flush=True)
