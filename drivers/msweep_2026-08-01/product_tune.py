# ruff: noqa: E402, E741, E731  (archival experiment drivers, verified by execution)
"""Product-model tuning at intraday H (honest labels).

Arms:
  inc / inc_vix / inc_shape / inc_vix_shape       — organ arms on the small base
  full 2-BLOCK ridge grid: alpha_har x alpha_exog — light shrinkage on the
      HAR+calendar block (the incumbent nested inside), heavy on wide exog;
      {0,100} x {1e4,1e5} + uniform 1e4 reference
  fullbest_vix_shape — best 2-block setting + VIX + announcement organ
      (VIX/ANN/SHP ride in the LIGHT block; in-run best-pick, disclosed)
  W12000 — window-length probe at the best 2-block setting

env: H, TAG
"""

import os
import time
import numpy as np
import pandas as pd

H = int(os.environ["H"])
TAG = os.environ.get("TAG", f"pt_h{H}")
W = 24000
N0, N1 = 24048, 72048
D = "results/b2_mmap"

from src.data.loading import load_raw_data
from src.evaluation.metrics import apply_duan_smearing, forecast_metrics
from src.evaluation.diebold_mariano import dm_test, qlike_per_bar
from block_ridge import BlockRidge

X = np.load(f"{D}/X.npy", mmap_mode="r")
b = np.load(f"{D}/b.npy")
names = [str(s) for s in np.load(f"{D}/names.npy", allow_pickle=True)]
n = X.shape[0]
raw = (
    load_raw_data("data", allow_missing=True)
    .dropna(subset=["RV"])
    .reset_index(drop=True)
)
RVraw = raw["RV"].to_numpy()[3125:][:n]
raw["t"] = pd.to_datetime(raw["t"])

vx = pd.read_parquet("data/vix_and_voldemand.parquet")
vx["endbartime"] = pd.to_datetime(vx["endbartime"])
mv = raw[["t"]].merge(vx, left_on="t", right_on="endbartime", how="left")
VIXB = mv[["vix", "vvix", "vix3m"]].ffill().bfill().to_numpy()[3125:][:n]
VIX = np.column_stack([np.log(VIXB), np.log(VIXB[:, 0] / VIXB[:, 2])])

rel = pd.read_parquet("data/releases.parquet")
rel["endbartime"] = pd.to_datetime(rel["endbartime"])
mr = raw[["t"]].merge(rel, left_on="t", right_on="endbartime", how="left").fillna(0.0)
flags = mr[[c for c in rel.columns if c != "endbartime"]].to_numpy()[3125:][:n]
types = [c for c in rel.columns if c != "endbartime"]
del raw, mr, mv
ar = np.arange(n)
ANN = np.zeros((n, 4 * len(types)))
for i in range(len(types)):
    f = flags[:, i]
    idx = np.flatnonzero(f)
    last = np.full(n, -float(n))
    nxt = np.full(n, float(n))
    pos = np.searchsorted(idx, ar, side="right")
    hp = pos > 0
    last[hp] = idx[np.clip(pos - 1, 0, None)][hp]
    hn = pos < len(idx)
    nxt[hn] = idx[np.clip(pos, None, len(idx) - 1)][hn]
    cum = np.concatenate([[0], np.cumsum(f)])
    cnt = np.zeros(n)
    cnt[: n - H] = cum[H + 1 : n + 1] - cum[1 : n - H + 1]
    ANN[:, 4 * i] = np.log1p(ar - last)
    ANN[:, 4 * i + 1] = np.log1p(nxt - ar)
    ANN[:, 4 * i + 2] = cnt
    ANN[:, 4 * i + 3] = np.log1p(np.minimum(nxt - ar, H + 1.0))


def fwd(arr, HH):
    c = np.concatenate([[0.0], np.cumsum(arr)])
    out = np.full(len(arr), np.nan)
    out[: len(arr) - HH + 1] = c[HH:] - c[:-HH]
    return out


RVf, Bf = fwd(RVraw, H), fwd(b, H)
yH = np.sqrt(RVf / Bf)
yH = np.where(np.isnan(yH), 0.0, yH)

inc_cols = [
    j
    for j, nm in enumerate(names)
    if nm.startswith("har_ma_")
    or nm
    in (
        "DOW_0",
        "DOW_1",
        "DOW_2",
        "DOW_3",
        "DOW_4",
        "hour",
        "is_overnight",
        "is_open",
        "is_close",
    )
]
Xi = np.ascontiguousarray(X[:N1, inc_cols])
exog_mask = np.ones(X.shape[1], bool)
exog_mask[inc_cols] = False


def run(F, tag, pens, Wt=W):
    """pens: list of (label, scalar-or-vector penalty). One gram, k solves."""
    t0 = time.time()
    P = {lb: np.empty(N1 - N0) for lb, _ in pens}
    br = BlockRidge(1.0)
    prev = None
    for t in range(N0, N1, 48):
        lo, hi = t - H - Wt, t - H
        if prev is None or br.needs_rebuild():
            br.build(F[lo:hi], yH[lo:hi])
        else:
            br.slide(
                F[prev[1] : hi], yH[prev[1] : hi], F[prev[0] : lo], yH[prev[0] : lo]
            )
        prev = (lo, hi)
        blk = np.asarray(F[t : t + 48])
        for lb, pen in pens:
            beta = br.coef(pen)
            P[lb][t - N0 : t - N0 + 48] = blk @ beta[1:] + beta[0]
    print(f"[{TAG}] {tag} done ({time.time() - t0:.0f}s)", flush=True)
    return P


def score(P, ref=None):
    pr, tr = apply_duan_smearing(P, yH[N0:N1], Bf[N0:N1])
    mm = forecast_metrics(pr, tr)
    out = f"qlike={mm['qlike']:.6f} mz={mm['mz_beta']:.3f}"
    if ref is not None:
        r = dm_test(qlike_per_bar(pr, tr), qlike_per_bar(ref[0], ref[1]), h=H)
        out += f" vs inc diff={r['mean_diff']:+.5f} p={r['p']:.4f}"
    return out, (pr, tr), mm["qlike"]


# small-base organ arms
Pi = run(Xi, "inc", [("inc", 1.0)])["inc"]
s, refit, _ = score(Pi)
print(f"[{TAG}] inc          {s}", flush=True)
for tag, cols in (
    ("inc_vix", [VIX]),
    ("inc_shape", [ANN]),
    ("inc_vix_shape", [VIX, ANN]),
):
    F = np.hstack([Xi] + [c[:N1] for c in cols])
    Pm = run(F, tag, [(tag, 1.0)])[tag]
    s, _, _ = score(Pm, refit)
    print(f"[{TAG}] {tag:<13} {s}", flush=True)

# full-stack 2-block grid (one gram, 5 solves)
p_ex = X.shape[1]
pens = [("u1e4", 1e4)]
for ah in (0.0, 100.0):
    for ae in (1e4, 1e5):
        v = np.where(exog_mask, ae, ah)
        pens.append((f"h{ah:g}_e{ae:g}", v))
Pf = run(X, "full2block", pens)
best_lb, best_q, best_pen = None, 1e9, None
for lb, pen in pens:
    s, _, q = score(Pf[lb], refit)
    print(f"[{TAG}] full {lb:<10} {s}", flush=True)
    if q < best_q:
        best_lb, best_q, best_pen = lb, q, pen
print(f"[{TAG}] best 2-block: {best_lb}", flush=True)


# best full + organ, with and without VIX (extras ride in the light block)
def light_pen(nx):
    if np.isscalar(best_pen):
        return np.concatenate([np.full(p_ex, best_pen), np.full(nx, best_pen)])
    lp = np.min(best_pen[~exog_mask]) if (~exog_mask).any() else 100.0
    return np.concatenate([best_pen, np.full(nx, lp)])


Fs_ = np.hstack([np.asarray(X[:N1]), ANN[:N1]])
Ps_ = run(Fs_, "fullbest_shape", [("fs", light_pen(ANN.shape[1]))])["fs"]
s, _, _ = score(Ps_, refit)
print(f"[{TAG}] full_best+shape      {s}", flush=True)
del Fs_
Fx = np.hstack([np.asarray(X[:N1]), VIX[:N1], ANN[:N1]])
Pb = run(Fx, "fullbest_vix_shape", [("fb", light_pen(VIX.shape[1] + ANN.shape[1]))])[
    "fb"
]
s, _, _ = score(Pb, refit)
print(f"[{TAG}] full_best+vix+shape  {s}", flush=True)
del Fx

# window probe at best setting
Pw = run(X, "W12000", [("w12", best_pen)], Wt=12000)["w12"]
s, _, _ = score(Pw, refit)
print(f"[{TAG}] full W=12000 {s}", flush=True)
print(f"[{TAG}] DONE", flush=True)
