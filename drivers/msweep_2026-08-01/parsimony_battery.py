"""Parsimony-champion lever battery on hybrid_dp (0.165145 @ 257p, chunk).

Arms (each a one-knob variant of the hybrid_dp run; DM vs the saved
hybrid257_dp per-bar preds, h=1, nested/matched):
  a03 / a30   — residualizer ridge alpha 0.3 / 3.0 (closes the alpha=1
                unswept exposure for the parsimony champion)
  k3 / k8     — shape-dictionary size K (private kernels may substitute
                for shapes: does K shrink?)
  privmix     — replace ALL fitted betas (target + per-channel scalar
                optimizations) with fixed-beta mixtures: target 5 betas
                (0.3,0.6,1.0,1.5,2.2), channels 2 betas (0.6,1.5); ridge
                learns the weights (freshness-by-reweighting law applied
                to the private organs; kills minimize_scalar entirely)
"""

import time
import numpy as np
import re
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import minimize_scalar
from src.evaluation.metrics import apply_duan_smearing, forecast_metrics
from src.evaluation.diebold_mariano import dm_test, qlike_per_bar
from src.features.transforms.residualizer import RollingRidgeResidualizer

import os

D = os.environ.get("PB_MMAP", "results/b2_mmap")
START = int(os.environ.get("PB_START", "24000"))
END = int(os.environ.get("PB_END", "26189"))
TRAIN_WIN = 24000
X = np.asarray(np.load(f"{D}/X.npy", mmap_mode="r")[: END + 300])
y = np.load(f"{D}/y.npy")
b = np.load(f"{D}/b.npy")
names = [str(s) for s in np.load(f"{D}/names.npy", allow_pickle=True)]
n0, n1, U = START, END, 256

chan = {}
for j, nm in enumerate(names):
    m = re.match(r"^adj_(.+)_ma_(\d+)$", nm)
    if m:
        chan.setdefault(m.group(1), []).append((int(m.group(2)), j))
for c in chan:
    chan[c].sort()
labels = list(chan)
ladder_cols = [j for c in chan for _, j in chan[c]]
other_cols = [j for j in range(len(names)) if j not in set(ladder_cols)]
L = np.stack([X[:, [j for _, j in chan[c]]] for c in chan], axis=0)
O = X[:, other_cols]
cal_cols = [
    names.index(nm)
    for nm in [
        "DOW_0",
        "DOW_1",
        "DOW_2",
        "DOW_3",
        "DOW_4",
        "hour",
        "is_overnight",
        "is_open",
        "is_close",
    ]
]
bar1 = {c: X[:, chan[c][0][1]] for c in labels}


def ridge_fit(A, yy):
    G = A.T @ A + np.eye(A.shape[1])
    G[0, 0] -= 1.0
    return np.linalg.solve(G, A.T @ yy)


def kfeat_series(s, beta):
    u = np.arange(1, U + 1, dtype=np.float64)
    w = u ** (-beta)
    w /= w.sum()
    K = np.zeros(n1 + 8)
    sw = sliding_window_view(s[: n1 + 8], U)
    K[U:] = sw[:-1] @ w[::-1]
    return K


def prof_series(s, t):
    if s[t - TRAIN_WIN : t].std() <= 0:
        return 1.0

    def loss(be):
        K = kfeat_series(s, be)
        A = np.stack([np.ones(TRAIN_WIN), K[t - TRAIN_WIN : t]], axis=1)
        cf, *_ = np.linalg.lstsq(A, y[t - TRAIN_WIN : t], rcond=None)
        r = y[t - TRAIN_WIN : t] - A @ cf
        return float(r @ r)

    return minimize_scalar(loss, bounds=(0.05, 3.0), method="bounded").x


import os as _os

if _os.environ.get("PB_WIDE"):
    MIX_T = [0.12, 0.3, 0.6, 1.0, 1.5, 2.2, 3.5]
    MIX_C = [0.3, 0.6, 1.5, 3.0]
else:
    MIX_T = [0.3, 0.6, 1.0, 1.5, 2.2]
    MIX_C = [0.6, 1.5]

# enet-identified dead families (unanimous across segments, logs/enet_zeros.log)
DROP = {
    "stocktwits_attention",
    "stocktwits_sentiment",
    "stocktwits_sentcount",
    "vvix",
    "vix3m",
    "voldemand_spx_open_and_close",
    "voldemand_spx_open_only",
    "voldemand_all_open_and_close",
    "voldemand_all_open_only",
}

# top raw cells by PCovR-correction loading (logs/pcovr_loadings.log) —
# the measured unspanned directions, added back as named dictionary columns
RAWCELLS = [
    "adj_sumautocov_ma_128",
    "adj_sumvolume_ma_8",
    "adj_numobs_ma_16",
    "adj_spread_vwstock_ma_8",
    "adj_sumautocov_ma_4",
    "adj_sumvolume_ma_4",
    "adj_sumautocov_ma_1",
    "adj_sumpret2_ewstock_ma_64",
    "adj_sumautocov_ma_2",
    "adj_sumret4_ma_64",
    "adj_sumret4_ma_1",
    "adj_sumret3_ewstock_ma_32",
    "adj_sumautocov_ma_64",
    "adj_sumret_ma_2",
    "adj_sumpret2_vwstock_ma_128",
    "adj_sumret4_ma_32",
    "adj_sumabsret_ma_4",
    "adj_sumbipow_vwstock_ma_32",
    "adj_spread_ewstock_ma_16",
    "adj_sumautocov_ma_16",
    "adj_sumret_ma_64",
    "adj_sumabsret_ma_2",
    "adj_spread_vwstock_ma_2",
    "adj_spread_vwstock_ma_64",
]
if os.environ.get("PB_CELLS"):
    _keep = os.environ["PB_CELLS"].split(",")
    RAWCELLS = [c for c in RAWCELLS if any(s in c for s in _keep)]


def run(
    tag,
    alpha=1.0,
    K=5,
    mixture=False,
    cadence=240,
    eff=10.0,
    pool_mode="kbar",
    products=False,
    prune=None,
    texture=0.0,
):
    # prune: None | "amp" (drop DROP channels from amplitude features only)
    #             | "full" (drop from the shape probe too — 32-channel model)
    keep = [i for i, c in enumerate(labels) if prune is None or c not in DROP]
    keep_lab = [labels[i] for i in keep]
    if prune == "full":
        p_ladder = [j for c in keep_lab for _, j in chan[c]]
    else:
        p_ladder = ladder_cols
    P = np.empty(n1 - n0)
    Kbar = None
    Pbar = None
    lam = 1.0 - 1.0 / eff
    t0 = time.time()
    if mixture:
        KM_T = np.stack([kfeat_series(y, be) for be in MIX_T], axis=1)
        KM_C = np.stack(
            [kfeat_series(bar1[c], be) for c in labels for be in MIX_C], axis=1
        )
    for seg in range(0, n1 - n0, cadence):
        t = n0 + seg
        full = slice(t - TRAIN_WIN, t)
        A0 = np.hstack([np.ones((TRAIN_WIN, 1)), O[full], X[full][:, p_ladder]])
        cff = ridge_fit(A0, y[full])
        Km = cff[1 + O.shape[1] :].reshape(-1, 12)
        if pool_mode == "proj":
            # subspace stabilization: fresh SVD, EWMA the PROJECTOR (gauge-
            # free — ridge only sees the span), top-K evecs of the pool
            _, sv, Vt = np.linalg.svd(Km, full_matrices=False)
            P5 = Vt[:K].T @ Vt[:K]
            Pbar = P5 if Pbar is None else lam * Pbar + (1.0 - lam) * P5
            ew, EV = np.linalg.eigh(Pbar)
            S = EV[:, ::-1][:, :K].T
        else:
            Kbar = Km if Kbar is None else lam * Kbar + (1.0 - lam) * Km
            _, sv, Vt = np.linalg.svd(Kbar, full_matrices=False)
            sgn = np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(axis=1)])
            S = (Vt * sgn[:, None])[:K]
        lo, hi = t - TRAIN_WIN, min(t + cadence, n1)
        Lsub = L[:, lo:hi] if prune is None else L[np.ix_(keep, np.arange(lo, hi))]
        if prune == "full":
            Sc = S
        elif prune == "amp":
            Sc = S  # shapes from the full 41-channel probe, applied to kept channels
        else:
            Sc = S
        C = np.einsum("cnr,kr->nck", Lsub, Sc).reshape(hi - lo, -1)
        if mixture:
            F = np.hstack([KM_T[lo:hi], KM_C[lo:hi], C, X[lo:hi][:, cal_cols]])
            if products:
                # dense-weak interaction bake (the b2_q pattern on this
                # dictionary): target-core pairs, core squares, and
                # target x channel-slow products — shrunk, never selected
                T5, C82 = KM_T[lo:hi], KM_C[lo:hi]
                iu = np.triu_indices(5)
                blocks = {
                    "pairs": T5[:, iu[0]] * T5[:, iu[1]],
                    "sq": C82 * C82,
                    "cross": np.repeat(T5[:, 2:3], 82, axis=1) * C82,
                }
                use = products if isinstance(products, (list, tuple)) else list(blocks)
                F = np.hstack([F] + [blocks[k] for k in use])
        elif getattr(run, "_tshape", False):
            har_idx = [
                oi
                for oi, j in enumerate(other_cols)
                if re.match(r"^har_ma_\d+$", names[j])
            ]
            kt = cff[1:][har_idx]  # target self-kernel estimate this refit
            if not hasattr(run, "_khist"):
                run._khist = []
            run._khist.append(kt)
            KH = np.array(run._khist)
            wts = (1.0 - 1.0 / 10.0) ** np.arange(len(KH))[::-1]
            _, _, Vt_y = np.linalg.svd(KH * wts[:, None], full_matrices=False)
            Sy = Vt_y[: min(5, len(Vt_y))]
            har_cols = [other_cols[oi] for oi in har_idx]
            TB = X[lo:hi][:, har_cols] @ Sy.T
            cores2 = [kfeat_series(bar1[c], prof_series(bar1[c], t)) for c in keep_lab]
            F = np.hstack(
                [TB] + [Kc[lo:hi, None] for Kc in cores2] + [C, X[lo:hi][:, cal_cols]]
            )
        else:
            cores = [kfeat_series(y, prof_series(y, t))]
            cores += [kfeat_series(bar1[c], prof_series(bar1[c], t)) for c in keep_lab]
            F = np.hstack(
                [Kc[lo:hi, None] for Kc in cores] + [C, X[lo:hi][:, cal_cols]]
            )
            if products:  # channel-core curvature (the prod_sq winner)
                F = np.hstack(
                    [F] + [np.column_stack([Kc[lo:hi] ** 2 for Kc in cores[1:]])]
                )
            cmod = _os.environ.get("PB_CMOD")
            if cmod:
                # clock-conditional shape amplitudes: modulate the C block by
                # daily-phase harmonics (smooth) or session gates (crude)
                hr = X[lo:hi][:, names.index("hour")]
                if cmod == "fourier":
                    ph_ = 2 * np.pi * hr / 24.0
                    mods = [np.sin(ph_), np.cos(ph_), np.sin(2 * ph_), np.cos(2 * ph_)]
                else:
                    mods = [
                        X[lo:hi][:, names.index(g)]
                        for g in ("is_open", "is_close", "is_overnight")
                    ]
                s_ = float(_os.environ.get("PB_CMOD_S", "1.0"))
                F = np.hstack([F] + [C * m_[:, None] * s_ for m_ in mods])
            if texture > 0:
                # dense-weak rung texture overlay: ALL live-channel ladder
                # cells, heavily shrunk via column scaling (penalty ~ a/s^2)
                # — shrink-over-everything, never select
                tex_cols = [j for c in keep_lab for _, j in chan[c]]
                F = np.hstack([F, X[lo:hi][:, tex_cols] * texture])
            if prune == "rawc":
                RC = X[lo:hi][:, [names.index(nm) for nm in RAWCELLS]]
                F = np.hstack([F, RC])
                if _os.environ.get("PB_GATES"):
                    g1 = X[lo:hi][:, names.index("is_close")][:, None]
                    g2 = X[lo:hi][:, names.index("is_overnight")][:, None]
                    ac = RC[:, [i for i, nm in enumerate(RAWCELLS) if "autocov" in nm]]
                    F = np.hstack([F, ac * g1, ac * g2])
        if _os.environ.get("PB_PANEL"):
            alphas = [0.1, 1.0, 10.0]
            rrs = []
            for a_ in alphas:
                r_ = RollingRidgeResidualizer(a_, fast_inverse=True)
                r_.fit(F[:TRAIN_WIN], y[lo : lo + TRAIN_WIN])
                rrs.append(r_)
            if seg == 0:
                run.pnl_p, run.pnl_y = [], []
            for i in range(hi - t):
                tt = TRAIN_WIN + i
                preds = [r_.predict(F[tt : tt + 1])[0] for r_ in rrs]
                if len(run.pnl_y) >= 1000:
                    if _os.environ.get("PB_PANEL") == "hedge":
                        # parameter-free simplex weights: w_k ~ 1/trailing-MSE_k
                        Pm = np.array(run.pnl_p[-1000:])
                        yv = np.array(run.pnl_y[-1000:])
                        mse = ((Pm - yv[:, None]) ** 2).mean(axis=0)
                        wts = 1.0 / mse
                        wts /= wts.sum()
                        P[seg + i] = float(wts @ np.array(preds))
                    else:
                        A2 = np.column_stack(
                            [np.ones(1000)]
                            + [
                                np.array([p[k] for p in run.pnl_p[-1000:]])
                                for k in range(3)
                            ]
                        )
                        wf, *_ = np.linalg.lstsq(
                            A2, np.array(run.pnl_y[-1000:]), rcond=None
                        )
                        P[seg + i] = wf[0] + sum(wf[1 + k] * preds[k] for k in range(3))
                else:
                    P[seg + i] = preds[1]
                run.pnl_p.append(preds)
                run.pnl_y.append(float(y[t + i]))
                if i < hi - t - 1:
                    for r_ in rrs:
                        r_.roll(F[tt], y[lo + tt], F[i], y[lo + i])
        else:
            rr = RollingRidgeResidualizer(alpha, fast_inverse=True)
            rr.fit(F[:TRAIN_WIN], y[lo : lo + TRAIN_WIN])
            for i in range(hi - t):
                tt = TRAIN_WIN + i
                P[seg + i] = rr.predict(F[tt : tt + 1])[0]
                if i < hi - t - 1:
                    rr.roll(F[tt], y[lo + tt], F[i], y[lo + i])
    pr, tr = apply_duan_smearing(P, y[n0:n1], b[n0:n1])
    mm = forecast_metrics(pr, tr)
    np.savez(f"results/geo_preds/pb_{tag}.npz", pred_raw=pr, true_raw=tr)
    if START == 24000:
        z = np.load("results/geo_preds/hybrid257_dp.npz")
        r = dm_test(
            qlike_per_bar(pr, tr), qlike_per_bar(z["pred_raw"], z["true_raw"]), h=1
        )
        print(
            f"[pb] {tag:<10} qlike={mm['qlike']:.6f} mz={mm['mz_beta']:.3f} p={F.shape[1] + 1} "
            f"| vs hybrid_dp diff={r['mean_diff']:+.5f} p={r['p']:.4f} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    else:
        print(
            f"[pb] {tag:<10} qlike={mm['qlike']:.6f} mz={mm['mz_beta']:.3f} p={F.shape[1] + 1} "
            f"| tile [{START},{END}) ({time.time() - t0:.0f}s)",
            flush=True,
        )


import os

arms = os.environ.get("PB_ARMS", "a03:0.3,a30:3.0,k3:K3,k8:K8,privmix:mix")
for spec in arms.split(","):
    tag, val = spec.split(":")
    if val == "comp":
        run(tag, prune="rawc", products=True, alpha=0.1)
    elif val == "mix":
        run(tag, mixture=True)
    elif val == "prod":
        run(tag, mixture=True, products=True)
    elif val.startswith("prod_"):
        run(tag, mixture=True, products=[val[5:]])
    elif val.startswith("K"):
        run(tag, K=int(val[1:]))
    elif val.startswith("c"):
        cad, ef = val[1:].split("e")
        run(tag, cadence=int(cad), eff=float(ef))
    elif val == "prsq":
        run(tag, prune="amp", products=True)
    elif val == "rawc":
        run(tag, prune="rawc")
    elif val == "rawc03":
        run(tag, prune="rawc", alpha=0.3)
    elif val == "rawc01":
        run(tag, prune="rawc", alpha=0.1)
    elif val == "pramp":
        run(tag, prune="amp")
    elif val == "tshape":
        run._tshape = True
        run(tag, prune="amp")
        run._tshape = False
    elif val == "a3x":
        run(tag, prune="amp", alpha=3.0)
    elif val == "a10x":
        run(tag, prune="amp", alpha=10.0)
    elif val == "pramp01":
        run(tag, prune="amp", alpha=0.1)
    elif val.startswith("tex"):
        run(tag, prune="amp", texture=float(val[3:]))
    elif val == "prfull":
        run(tag, prune="full")
    elif val.startswith("p"):
        run(tag, eff=float(val[1:]), pool_mode="proj")
    else:
        run(tag, alpha=float(val))
print("[pb] DONE", flush=True)
