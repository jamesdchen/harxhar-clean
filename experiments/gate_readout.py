"""Read the learned SoftTreeGate hyperplane from the regime-MoE on one cadence block.

The deliverable behind the MoE even if QLIKE doesn't move (discovery_process_methodology §5):
does the learned partition RECOVER `hour` (validating the hand-found close regime), or find a
partition the clock smeared? The cascade refits per block and persists nothing, so this re-fits
ONE block's regime MoE exactly as `preds_chunk`'s resid_regime arm does, then dumps the gate's
decision-node weights (standardized-feature space; |w| ranks the splitting features).

Run cluster-side on a compute node (loads the 758MB Xs cache; CPU torch):
  READOUT_DEPTH=1 READOUT_BLOCK=300 $PY gate_readout.py <CID>
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)

import json
import os
import sys

import numpy as np

CID = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim"
)
DEPTH = int(os.environ.get("READOUT_DEPTH", "1"))  # 1 = single hyperplane (clearest regime axis)
os.environ.setdefault(
    "GLOBAL_CFG",
    json.dumps(
        {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "subsample": 0.7,
            "colsample_bytree": 0.5,
            "min_child_weight": 50.0,
            "reg_lambda": 2.0,
            "n_jobs": 4,
            "max_depth": 8,
        }
    ),
)

import resid_amortized as ra  # noqa: E402
from src.models.regime_moe import RegimeMoE  # noqa: E402

c = ra.load_cache(CID)
tw = c["cell"]["train_win"]
feats = c["feats"]
hr = c["Xs"][:, feats.index("hour")]
starts = c["starts"]
gcfg = json.loads(os.environ["GLOBAL_CFG"])
mk_g = ra._tree_factory("xgb", gcfg)

# default to a late block (ample, settled h16-19 history)
i = int(os.environ.get("READOUT_BLOCK", str(len(starts) * 3 // 4)))
t_r = int(starts[i])
cols = c["masks"][i]
fm = c.get("force_mask")
if fm is not None:
    cols = cols | fm
colnames = [feats[j] for j in np.where(cols)[0]]

Xtr = c["Xs"][t_r - tw : t_r]
r1 = c["y"][t_r - tw : t_r] - (Xtr @ c["coefs"][i] + c["intercepts"][i])
g = mk_g()
g.fit(Xtr[:, cols], r1)
r2 = r1 - g.predict(Xtr[:, cols]).ravel()
m_tr = ra._close_mask(hr[t_r - tw : t_r])
print(
    f"block i={i} t_r={t_r} survivors={len(colnames)} "
    f"h{ra.CLOSE_LO}-{ra.CLOSE_HI}_train_rows={int(m_tr.sum())} depth={DEPTH}",
    flush=True,
)

e = RegimeMoE(depth=DEPTH, epochs=150, hidden_dim=16, aux_weight=0.5, seed=42)
e.fit(Xtr[:, cols][m_tr], r2[m_tr])
# bag-aggregate the gate hyperplanes: each bag's gate covers its feature SUBSET (feature
# subsampling), so accumulate per global survivor column and average over the bags that used it.
num_nodes = e.models_[0][0].gate.num_nodes
Wsum = np.zeros((num_nodes, len(colnames)))
cnt = np.zeros(len(colnames))
for m, fidx in e.models_:
    fi = fidx.cpu().numpy()
    Wsum[:, fi] += m.gate.decision_nodes.weight.detach().cpu().numpy()
    cnt[fi] += 1
W = Wsum / np.maximum(cnt, 1)  # [num_nodes, p_global]

for node in range(W.shape[0]):
    order = np.argsort(-np.abs(W[node]))[:12]
    print(f"--- gate node {node} (P(left)=sigmoid(w·x+b)) top |w| ---", flush=True)
    for j in order:
        print(f"  {colnames[j]:34s} w={W[node, j]:+.3f}", flush=True)

if "hour" in colnames:
    hj = colnames.index("hour")
    ranks = [int(np.where(np.argsort(-np.abs(W[node])) == hj)[0][0]) for node in range(W.shape[0])]
    print(f"hour_rank_per_node (0=top splitter): {ranks}", flush=True)
else:
    print("hour NOT in survivor set this block (gate keys off other features)", flush=True)
print("GATE_READOUT_DONE", flush=True)
