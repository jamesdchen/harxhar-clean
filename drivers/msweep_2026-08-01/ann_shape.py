# ruff: noqa: E402, E741, E731  (archival experiment drivers, verified by execution)
"""Announcement SHAPE organ at intraday-H, extended eval for event power.

Eval [24000, 72000) (~4 years vs the pilot's 45 days). Arms (nested):
  full            — the b2 stack (control)
  +v1             — since/until log1p features (the pilot's ANN)
  +shape          — v1 + window-aware features: per type, count of releases
                    inside (t, t+H] and bars-until-first-release-in-window
                    (both ex-ante: release schedules are published).
Pilot training convention retained (matched arms share it), HAC DM at h=H.

env: H (8|16), TAG
"""

import os
import time
import numpy as np
import pandas as pd

H = int(os.environ["H"])
TAG = os.environ.get("TAG", f"ann_h{H}")
W = 24000
N0, N1 = (
    24048,
    72048,
)  # +1 day so the honest-lag window [N0-H-W, N0-H) stays in-range for H<=48
D = "results/b2_mmap"

from src.data.loading import load_raw_data
from src.evaluation.metrics import apply_duan_smearing, forecast_metrics
from src.evaluation.diebold_mariano import dm_test, qlike_per_bar

X = np.load(f"{D}/X.npy", mmap_mode="r")
b = np.load(f"{D}/b.npy")
n = X.shape[0]
raw = (
    load_raw_data("data", allow_missing=True)
    .dropna(subset=["RV"])
    .reset_index(drop=True)
)
RVraw = raw["RV"].to_numpy()[3125:][:n]
rel = pd.read_parquet("data/releases.parquet")
rel["endbartime"] = pd.to_datetime(rel["endbartime"])
raw["t"] = pd.to_datetime(raw["t"])
m = raw[["t"]].merge(rel, left_on="t", right_on="endbartime", how="left").fillna(0.0)
flags = m[[c for c in rel.columns if c != "endbartime"]].to_numpy()[3125:][:n]
types = [c for c in rel.columns if c != "endbartime"]
del raw, m

# v1: since/until log1p (the pilot's features)
ANN = np.zeros((n, 2 * len(types)))
# shape: count in (t, t+H] + bars-until-first-in-window (H+1 when none)
SHP = np.zeros((n, 2 * len(types)))
ar = np.arange(n)
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
    ANN[:, 2 * i] = np.log1p(ar - last)
    ANN[:, 2 * i + 1] = np.log1p(nxt - ar)
    cum = np.concatenate([[0], np.cumsum(f)])
    cnt = np.zeros(n)
    cnt[: n - H] = cum[H + 1 : n + 1] - cum[1 : n - H + 1]  # releases in (t, t+H]
    SHP[:, 2 * i] = cnt
    until_w = np.minimum(nxt - ar, H + 1.0)  # first release in window, capped
    SHP[:, 2 * i + 1] = np.log1p(until_w)


def fwd(arr, HH):
    c = np.concatenate([[0.0], np.cumsum(arr)])
    out = np.full(len(arr), np.nan)
    out[: len(arr) - HH + 1] = c[HH:] - c[:-HH]
    return out


RVf, Bf = fwd(RVraw, H), fwd(b, H)
yH = np.sqrt(RVf / Bf)
okr = ~np.isnan(yH)
yH = np.where(okr, yH, 0.0)
assert okr[N0 - W - H : N1].all()

from block_ridge import BlockRidge


def run(F, tag):
    # HONEST convention: training window [t-H-W, t-H) — every label's
    # forward RV window is realized at prediction time t.
    t0 = time.time()
    P = np.empty(N1 - N0)
    br = BlockRidge(1.0)
    prev = None
    for t in range(N0, N1, 48):
        lo, hi = t - H - W, t - H
        if prev is None or br.needs_rebuild():
            br.build(F[lo:hi], yH[lo:hi])
        else:
            br.slide(
                F[prev[1] : hi], yH[prev[1] : hi], F[prev[0] : lo], yH[prev[0] : lo]
            )
        prev = (lo, hi)
        beta = br.coef()
        P[t - N0 : t - N0 + 48] = np.asarray(F[t : t + 48]) @ beta[1:] + beta[0]
        if (t - N0) % 9600 == 9552:
            print(
                f"  [{TAG}] {tag} {t - N0 + 48}/{N1 - N0} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    pr, tr = apply_duan_smearing(P, yH[N0:N1], Bf[N0:N1])
    mm = forecast_metrics(pr, tr)
    np.savez(f"results/geo_preds/{TAG}_{tag}.npz", pred_raw=pr, true_raw=tr)
    print(
        f"[{TAG}] {tag:<10} qlike={mm['qlike']:.6f} mz={mm['mz_beta']:.3f} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return pr, tr


names_l = [str(s) for s in np.load(f"{D}/names.npy", allow_pickle=True)]
inc_cols = [
    j
    for j, nm in enumerate(names_l)
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

if os.environ.get("ALPHA_SWEEP"):
    ALPHAS = [
        float(a) for a in os.environ.get("ALPHAS", "1,10,100,1000,10000").split(",")
    ]

    def run_multi(F, tag):
        t0 = time.time()
        P = {a: np.empty(N1 - N0) for a in ALPHAS}
        br = BlockRidge(1.0)
        prev = None
        for t in range(N0, N1, 48):
            lo, hi = t - H - W, t - H
            if prev is None or br.needs_rebuild():
                br.build(F[lo:hi], yH[lo:hi])
            else:
                br.slide(
                    F[prev[1] : hi], yH[prev[1] : hi], F[prev[0] : lo], yH[prev[0] : lo]
                )
            prev = (lo, hi)
            blk = np.asarray(F[t : t + 48])
            for a in ALPHAS:
                beta = br.coef(a)
                P[a][t - N0 : t - N0 + 48] = blk @ beta[1:] + beta[0]
        print(f"[{TAG}] {tag} done ({time.time() - t0:.0f}s)", flush=True)
        return P

    names_l = [str(s) for s in np.load(f"{D}/names.npy", allow_pickle=True)]
    inc_cols = [
        j
        for j, nm in enumerate(names_l)
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
    pi, tr = run(Xi, "inc")
    Pfull = run_multi(X, "full_ladder")
    for a in ALPHAS:
        pr, _ = apply_duan_smearing(Pfull[a], yH[N0:N1], Bf[N0:N1])
        mm = forecast_metrics(pr, tr)
        r = dm_test(qlike_per_bar(pr, tr), qlike_per_bar(pi, tr), h=H)
        print(
            f"[{TAG}] full a={a:<7g} qlike={mm['qlike']:.6f} mz={mm['mz_beta']:.3f} "
            f"vs inc diff={r['mean_diff']:+.5f} p={r['p']:.4f}",
            flush=True,
        )
    print(f"[{TAG}] DONE", flush=True)
elif os.environ.get("INC_SHAPE"):
    pi, tr = run(Xi, "inc")
    Fis = np.hstack([Xi, ANN[:N1], SHP[:N1]])
    ps, _ = run(Fis, "inc_shape")
    r = dm_test(qlike_per_bar(ps, tr), qlike_per_bar(pi, tr), h=H)
    print(f"[{TAG}] inc_shape vs inc diff={r['mean_diff']:+.5f} p={r['p']:.4f}")
    print(f"[{TAG}] DONE", flush=True)
else:
    pi, tr = run(Xi, "inc")
    pf, _ = run(X, "full")
    Fs = np.hstack([X[:N1], ANN[:N1], SHP[:N1]])
    ps, _ = run(Fs, "shape")
    del Fs
    r1 = dm_test(qlike_per_bar(pf, tr), qlike_per_bar(pi, tr), h=H)
    r2 = dm_test(qlike_per_bar(ps, tr), qlike_per_bar(pf, tr), h=H)
    r3 = dm_test(qlike_per_bar(ps, tr), qlike_per_bar(pi, tr), h=H)
    print(f"[{TAG}] full vs inc   diff={r1['mean_diff']:+.5f} p={r1['p']:.4f}")
    print(f"[{TAG}] shape vs full diff={r2['mean_diff']:+.5f} p={r2['p']:.4f}")
    print(f"[{TAG}] shape vs inc  diff={r3['mean_diff']:+.5f} p={r3['p']:.4f}")
    print(f"[{TAG}] DONE", flush=True)
