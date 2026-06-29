"""Sequence-vs-state head-to-head on the linbest residual — the rigorous order-content closer (ROBUST).

The level-3 / 3-channel (tau,cumret,cumrv) log-signature null only ruled out LOW-ORDER order content in
THOSE channels. Untested corner: does the ORDER of the FULL multivariate bar HISTORY carry signal the
current state row doesn't? Isolated by a SHUFFLE PLACEBO -- the SAME GRU on true order vs per-sample
time-permuted windows. ROBUSTNESS: repeat over a WALK-FORWARD of folds x multiple SEEDS, and report the
order_delta = qlike(seq_true) - qlike(seq_shuffle) DISTRIBUTION vs the seed-noise scale. order_delta
robustly < 0 (mean clears the noise) => genuine order content; straddles 0 => noise, state framing earned.

Arms (close-window h16-19 of r = y - linbest_base): state (current bar) | seq_shuffle (set) | seq_true.
Run on Hoffman2 hpc-pi (CPU torch). Envs: SEQ_L SEQ_H SEQ_EPOCHS SEQ_FOLDS SEQ_SEEDS.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

CID = sys.argv[1] if len(sys.argv) > 1 else "xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim"
L = int(os.environ.get("SEQ_L", "24"))
HID = int(os.environ.get("SEQ_H", "16"))
EPOCHS = int(os.environ.get("SEQ_EPOCHS", "80"))
FOLDS = int(os.environ.get("SEQ_FOLDS", "4"))
SEEDS = int(os.environ.get("SEQ_SEEDS", "3"))

import resid_amortized as ra  # noqa: E402
from src.evaluation.metrics import apply_duan_smearing  # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
c = ra.load_cache(CID)
tw = c["cell"]["train_win"]
feats = c["feats"]
Xs = np.asarray(c["Xs"], dtype=np.float32)
n = len(Xs)
y = np.asarray(c["y"], dtype=np.float64)
ridge_oos = np.asarray(c["ridge_oos"], dtype=np.float64)
base = np.asarray(c["base"], dtype=np.float64)
hour = Xs[:, feats.index("hour")]
y_oos, base_oos = y[tw:], base[tw:]
r_oos = (y_oos - ridge_oos).astype(np.float32)
abs_t = np.arange(tw, n)
close = ra._close_mask(hour[tw:]) & (abs_t >= L - 1)
ci = np.where(close)[0]
t_close = abs_t[ci]
keep = [f for f in feats if f.startswith("har_ma_") or "cumrv" in f.lower() or f == "hour"]
fidx = np.array([feats.index(f) for f in keep])
F = len(fidx)
N = len(ci)
W = np.empty((N, L, F), dtype=np.float32)
for j, t in enumerate(t_close):
    W[j] = Xs[t - L + 1 : t + 1, fidx]
rr = r_oos[ci].astype(np.float32)
rc, yc, bc = ridge_oos[ci], y_oos[ci], base_oos[ci]  # for QLIKE: pred_adj = ridge + resid_pred
print(f"close usable rows={N} L={L} F={F} hidden={HID} folds={FOLDS} seeds={SEEDS}", flush=True)


class SeqModel(nn.Module):
    def __init__(self, f, h, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(f, h, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        _, hN = self.gru(x)
        return self.head(self.drop(hN[-1]))


def _perm_each(x):  # per-sample random L-permutation -> destroys order, keeps the multiset
    b, ll, _ = x.shape
    idx = torch.argsort(torch.rand(b, ll, device=x.device), dim=1)
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(2)))


def qlike(pred_resid, te):
    pr, trr = apply_duan_smearing(rc[te] + pred_resid, yc[te], bc[te])
    m = (trr > 0) & (pr > 0)
    rt = trr[m] / pr[m]
    return float(np.mean(rt - np.log(rt) - 1.0))


def train_eval(tr, te, arm, seed):
    torch.manual_seed(seed)
    mu = W[tr].reshape(-1, F).mean(0)
    sd = np.where(W[tr].reshape(-1, F).std(0) > 0, W[tr].reshape(-1, F).std(0), 1.0)
    Wn = ((W - mu) / sd).astype(np.float32)
    rmu, rsd = float(rr[tr].mean()), float(rr[tr].std() or 1.0)
    Wt = torch.as_tensor(Wn, device=dev)
    yt = torch.as_tensor(((rr - rmu) / rsd).astype(np.float32), device=dev).unsqueeze(1)
    Xtr = Wt[tr][:, -1:, :] if arm == "state" else Wt[tr]
    Xte = Wt[te][:, -1:, :] if arm == "state" else Wt[te]
    n_v = max(64, int(0.15 * len(tr)))
    fi, vi = slice(0, len(tr) - n_v), slice(len(tr) - n_v, len(tr))
    m = SeqModel(F, HID).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    mse = nn.MSELoss()
    best, bs, wait = 1e9, None, 0
    for _ in range(EPOCHS):
        m.train()
        opt.zero_grad(set_to_none=True)
        xb = _perm_each(Xtr[fi]) if arm == "seq_shuffle" else Xtr[fi]
        mse(m(xb), yt[tr][fi]).backward()
        opt.step()
        m.eval()
        with torch.no_grad():
            xv = _perm_each(Xtr[vi]) if arm == "seq_shuffle" else Xtr[vi]
            v = mse(m(xv), yt[tr][vi]).item()
        if v < best:
            best, wait, bs = v, 0, {k: vv.detach().clone() for k, vv in m.state_dict().items()}
        else:
            wait += 1
            if wait >= 12:
                break
    if bs:
        m.load_state_dict(bs)
    m.eval()
    with torch.no_grad():
        xe = _perm_each(Xte) if arm == "seq_shuffle" else Xte
        pred = m(xe).squeeze(1).cpu().numpy() * rsd + rmu
    return qlike(pred, te)


# walk-forward: the last 40% split into FOLDS sequential test segments; expanding train on all prior rows.
edges = (N * np.linspace(0.6, 1.0, FOLDS + 1)).astype(int)
deltas = []
for f in range(FOLDS):
    te = np.arange(edges[f], edges[f + 1])
    tr = np.arange(0, edges[f])
    q_base = qlike(np.zeros(len(te)), te)
    res = {}
    for arm in ("state", "seq_shuffle", "seq_true"):
        qs = [train_eval(tr, te, arm, s) for s in range(SEEDS)]
        res[arm] = (float(np.mean(qs)), float(np.std(qs)))
    od = res["seq_true"][0] - res["seq_shuffle"][0]
    deltas.append(od)
    print(
        f"fold {f} n_te={len(te)} base={q_base:.5f} | state={res['state'][0]:.5f} "
        f"shuffle={res['seq_shuffle'][0]:.5f} true={res['seq_true'][0]:.5f} "
        f"| order_delta={od:+.5f} (seed_sd true={res['seq_true'][1]:.5f} shuf={res['seq_shuffle'][1]:.5f})",
        flush=True,
    )
d = np.array(deltas)
print(
    f"ORDER_DELTA over {FOLDS} folds: mean={d.mean():+.5f} sd={d.std():.5f} "
    f"min={d.min():+.5f} max={d.max():+.5f}  ({(d < 0).sum()}/{FOLDS} folds favor true order)",
    flush=True,
)
print(
    "VERDICT: robustly < 0 and |mean| > seed_sd => genuine order content; straddles 0 => noise (state framing earned)",
    flush=True,
)
print("SEQ_VS_STATE_DONE", flush=True)
