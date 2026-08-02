# ruff: noqa: E402, E741, E731  (experiment driver, verified by execution)
"""One-step penalized functional RRR vs two-step probe->SVD for the 5-shape operator.

The 5-shape construction is two-step: ridge probe on [1|O|ladder] -> reshape
ladder coefs to Km (41 x nr) -> SVD -> top-K lag shapes. Statistically that is
the SVD-truncation estimator of a reduced-rank regression (RRR). This driver
asks three questions the smooth-beta frame raises:

STATIC (windows ending at each tile start):
  1. Does one-step rank-5 RRR (alternating least squares on the same gram)
     find a different subspace than SVD-of-the-probe? (principal angles, SSE)
  2. Does a roughness penalty on the shapes (P-spline D2 along the lag axis,
     the "penalized functional" part) change the answer — and does the lag-1
     innovations spike survive it? Penalty variants: full D2 vs spike-exempt
     (rung 1 left out of the difference penalty, so the contrast structure
     is not taxed).
  3. RANK CEILING: on the base-sqrt2 panel the ladder has ~22 rungs, raising
     the operator's rank bound from 12 to ~22 — does the singular spectrum
     still knee at ~5? (the rank-stability check)

WALK-FORWARD (both tiles, cadence-240 shape refresh, EWMA pooling, identical
amplitude model [1 | C(41xK) | self-ladder | calendar] ridge alpha=1 per-bar):
  arms differ ONLY in how shapes are estimated:
    ctrl   two-step, Kbar EWMA pool -> SVD (the champion recipe)
    proj   two-step, projector EWMA pool (gauge-free control)
    rrr0   one-step RRR, lambda=0, warm-started ALS, projector pool
    rrrS   one-step penalized fRRR, spike-exempt D2 at LAM_REL
    rrrF   one-step penalized fRRR, full D2 at LAM_REL
    sq2    two-step on the base-sqrt2 panel (rank ceiling 22, K=5) — "can we
           raise the possible rank without sacrificing QLIKE?"

env: PREP_B2, PREP_SQ2, TILES ("1,2"), CADENCE (240), EFF (10 pool memory),
     K (5), LAM_REL (1.0; penalty scaled by tr(G_ZZ)/60 so it is unitless),
     STATIC_ONLY, ARMS
Outputs: results/pspline_race/op_{shapes,preds}_*.npz + printed tables.
"""

import os
import re
import time

import numpy as np

from src.evaluation.diebold_mariano import dm_test, qlike_per_bar
from src.evaluation.metrics import apply_duan_smearing, forecast_metrics

PREP_B2 = os.environ.get("PREP_B2", "results/prep_cache_all_features_b2_r32000.npz")
PREP_SQ2 = os.environ.get(
    "PREP_SQ2", "results/prep_cache_all_features_b1.41421_r32000.npz")
CADENCE = int(os.environ.get("CADENCE", "240"))
EFF = float(os.environ.get("EFF", "10"))
K = int(os.environ.get("K", "5"))
LAM_REL = float(os.environ.get("LAM_REL", "1.0"))
TILES = [int(v) for v in os.environ.get("TILES", "1,2").split(",")]
W = 24000
OUT = "results/pspline_race"
os.makedirs(OUT, exist_ok=True)
TILE_BOUNDS = {1: (24000, 26189), 2: (26189, 28378)}


def load_panel(path):
    z = np.load(path, allow_pickle=True)
    X, y, b = z["X"], z["y"], z["baselines"]
    names = [str(s) for s in z["names"]]
    chan = {}
    for j, nm in enumerate(names):
        m = re.match(r"^adj_(.+)_ma_(\d+)$", nm)
        if m:
            chan.setdefault(m.group(1), []).append((int(m.group(2)), j))
    for c in chan:
        chan[c].sort()
    labels = list(chan)
    nr = len(chan[labels[0]])
    assert all(len(chan[c]) == nr for c in labels), "unequal ladders"
    ladder_cols = [j for c in labels for _, j in chan[c]]
    other_cols = [j for j in range(len(names)) if j not in set(ladder_cols)]
    har_rungs = sorted(
        int(m.group(1)) for nm in names if (m := re.match(r"^har_ma_(\d+)$", nm)))
    cal_cols = [names.index(c) for c in
                ["DOW_0", "DOW_1", "DOW_2", "DOW_3", "DOW_4",
                 "hour", "is_overnight", "is_open", "is_close"]]
    har_cols = [names.index(f"har_ma_{r}") for r in har_rungs]
    return dict(X=np.ascontiguousarray(X), y=np.asarray(y, float),
                b=np.asarray(b, float), names=names, labels=labels, nr=nr,
                ladder=ladder_cols, other=other_cols, cal=cal_cols,
                har=har_cols)


def probe_gram(P, t):
    """Gram + moment vector of [1 | O | ladder] on window [t-W, t)."""
    A = np.hstack([np.ones((W, 1)), P["X"][t - W : t][:, P["other"]],
                   P["X"][t - W : t][:, P["ladder"]]])
    G = A.T @ A
    c = A.T @ P["y"][t - W : t]
    yy = float(P["y"][t - W : t] @ P["y"][t - W : t])
    return G, c, yy


def ridge_solve(G, c, pen_diag_free0=1.0):
    Gp = G + pen_diag_free0 * np.eye(len(G))
    Gp[0, 0] -= pen_diag_free0
    return np.linalg.solve(Gp, c)


def probe_km(P, G, c):
    nO = len(P["other"])
    cf = ridge_solve(G, c)
    Km = cf[1 + nO :].reshape(len(P["labels"]), P["nr"])
    return Km


def svd_shapes(Km, k):
    _, sv, Vt = np.linalg.svd(Km, full_matrices=False)
    sgn = np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(axis=1)])
    return sv, (Vt * sgn[:, None])[:k]


def d2mat(nr, spike_exempt=False):
    D = np.diff(np.eye(nr), n=2, axis=0)
    return D[1:] if spike_exempt else D


def als_rrr(P, G, c, yy, k, lam_rel=0.0, spike_exempt=False, V0=None,
            iters=8, tol=1e-9):
    """Rank-k (penalized functional) RRR on the probe gram, by ALS.

    Model: y ~ mu + O g + sum_ck B[c,k] sum_r V[k,r] L[c,r]; roughness
    penalty lam * sum_k ||D2 V_k||^2 (V-step only; lam scaled by
    tr(G_ZZ)/60 to be unitless). Returns V (k x nr, orthonormal rows),
    B (nc x k), sse trace.
    """
    nO, nc, nr = len(P["other"]), len(P["labels"]), P["nr"]
    G4 = G[1 + nO :, 1 + nO :].reshape(nc, nr, nc, nr)
    G_O_L = G[: 1 + nO, 1 + nO :].reshape(1 + nO, nc, nr)
    G_OO = G[: 1 + nO, : 1 + nO]
    c_O, c_L = c[: 1 + nO], c[1 + nO :].reshape(nc, nr)
    if V0 is None:
        _, V = svd_shapes(probe_km(P, G, c), k)
    else:
        V = V0.copy()
    D = d2mat(nr, spike_exempt)
    DTD = D.T @ D
    sses, B = [], None
    for it in range(iters):
        # B-step: features [1|O|C], C_(c,k) = sum_r V[k,r] L[c,r]
        G_CC = np.einsum("kr,crds,ls->ckdl", V, G4, V,
                         optimize=True).reshape(nc * k, nc * k)
        G_OC = np.einsum("ocr,kr->ock", G_O_L, V,
                         optimize=True).reshape(1 + nO, nc * k)
        c_C = (c_L @ V.T).reshape(nc * k)
        Gb = np.block([[G_OO, G_OC], [G_OC.T, G_CC]])
        cb = np.concatenate([c_O, c_C])
        cf = ridge_solve(Gb, cb)
        B = cf[1 + nO :].reshape(nc, k)
        # V-step: features [1|O|Z], Z_(k,r) = sum_c B[c,k] L[c,r]
        G_ZZ = np.einsum("ck,crds,dl->krls", B, G4, B,
                         optimize=True).reshape(k * nr, k * nr)
        G_OZ = np.einsum("ocr,ck->okr", G_O_L, B,
                         optimize=True).reshape(1 + nO, k * nr)
        c_Z = (B.T @ c_L).reshape(k * nr)
        lam = lam_rel * np.trace(G_ZZ) / (k * nr)
        pen = np.zeros((1 + nO + k * nr, 1 + nO + k * nr))
        pen[1 : 1 + nO, 1 : 1 + nO] = np.eye(nO)
        Pv = np.kron(np.eye(k), lam * DTD) + 1e-8 * np.trace(G_ZZ) / (
            k * nr) * np.eye(k * nr)
        pen[1 + nO :, 1 + nO :] = Pv
        Gv = np.block([[G_OO, G_OZ], [G_OZ.T, G_ZZ]])
        cv = np.concatenate([c_O, c_Z])
        cf = np.linalg.solve(Gv + pen, cv)
        Vfl = cf[1 + nO :].reshape(k, nr)
        sse = yy - 2 * cf @ cv + cf @ (Gv @ cf)
        sses.append(float(sse))
        # re-orthonormalize rows (gauge fix; B absorbs the mixing next B-step)
        Q, _ = np.linalg.qr(Vfl.T)
        Vn = Q.T
        if it > 0 and abs(sses[-2] - sses[-1]) < tol * abs(sses[-1]):
            V = Vn
            break
        V = Vn
    return V, B, sses


def angles(V1, V2):
    """Principal angles (deg) between row spaces."""
    Q1, _ = np.linalg.qr(V1.T)
    Q2, _ = np.linalg.qr(V2.T)
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1, 1)))


def sse_of_shapes(P, G, c, yy, V):
    """In-window SSE of the [1|O|C(V)] ridge — fit quality of a fixed frame."""
    nO, nc = len(P["other"]), len(P["labels"])
    k = len(V)
    G4 = G[1 + nO :, 1 + nO :].reshape(nc, P["nr"], nc, P["nr"])
    G_O_L = G[: 1 + nO, 1 + nO :].reshape(1 + nO, nc, P["nr"])
    G_CC = np.einsum("kr,crds,ls->ckdl", V, G4, V,
                     optimize=True).reshape(nc * k, nc * k)
    G_OC = np.einsum("ocr,kr->ock", G_O_L, V,
                     optimize=True).reshape(1 + nO, nc * k)
    c_C = (c[1 + nO :].reshape(nc, P["nr"]) @ V.T).reshape(nc * k)
    Gb = np.block([[G[: 1 + nO, : 1 + nO], G_OC], [G_OC.T, G_CC]])
    cb = np.concatenate([c[: 1 + nO], c_C])
    cf = ridge_solve(Gb, cb)
    return float(yy - 2 * cf @ cb + cf @ (Gb @ cf))


# ── load panels ──────────────────────────────────────────────────────────
PB = load_panel(PREP_B2)
PS = load_panel(PREP_SQ2) if os.path.exists(PREP_SQ2) else None
if PS is not None:
    assert np.allclose(PB["y"], PS["y"]), "panels disagree on y"
print(f"b2 panel: {PB['X'].shape}, {len(PB['labels'])} channels x "
      f"{PB['nr']} rungs; sqrt2 panel: "
      f"{None if PS is None else (PS['X'].shape, PS['nr'])}", flush=True)

# ── STATIC ───────────────────────────────────────────────────────────────
for tile in TILES:
    t = TILE_BOUNDS[tile][0]
    print(f"\n=== STATIC window [{t - W},{t}) (tile-{tile} start) ===",
          flush=True)
    t0 = time.time()
    G, c, yy = probe_gram(PB, t)
    Km = probe_km(PB, G, c)
    sv, V2 = svd_shapes(Km, K)
    tot = float((np.linalg.svd(Km, compute_uv=False) ** 2).sum())
    print(f"two-step sv={np.round(sv[:8], 3)}  "
          f"top-{K} energy={float((sv[:K]**2).sum())/tot:.3f}  "
          f"sse={sse_of_shapes(PB, G, c, yy, V2):.4f}", flush=True)
    res = {"sv_b2": sv, "V_2step": V2, "Km": Km}
    for lam, sx, tag in [(0.0, False, "rrr0"), (LAM_REL, True, "rrrS"),
                         (LAM_REL, False, "rrrF")]:
        V, B, sses = als_rrr(PB, G, c, yy, K, lam, sx, V0=V2)
        Vr, Br, sr = als_rrr(PB, G, c, yy, K, lam, sx,
                             V0=np.linalg.qr(
                                 np.cos(np.outer(np.arange(1, K + 1),
                                                 np.arange(PB["nr"])))
                                 .T)[0].T)
        ang = angles(V2, V)
        angr = angles(V, Vr)
        rough2 = float((d2mat(PB["nr"]) @ V2.T ** 1).ravel() @ (
            d2mat(PB["nr"]) @ V2.T).ravel())
        roughV = float((d2mat(PB["nr"]) @ V.T).ravel() @ (
            d2mat(PB["nr"]) @ V.T).ravel())
        print(f"{tag:<5} lam_rel={lam:<4} sse={sses[-1]:.4f} "
              f"(iters={len(sses)}, init-robust max-angle={angr.max():.1f}d) "
              f"| angles vs 2step (deg)={np.round(ang, 1)} "
              f"| rough(V)={roughV:.3f} vs 2step {rough2:.3f}", flush=True)
        res[f"V_{tag}"] = V
        res[f"B_{tag}"] = B
    if PS is not None:
        Gs, cs, yys = probe_gram(PS, t)
        Kms = probe_km(PS, Gs, cs)
        svs = np.linalg.svd(Kms, compute_uv=False)
        tots = float((svs ** 2).sum())
        cum = np.cumsum(svs ** 2) / tots
        print(f"sqrt2 panel (rank ceiling {PS['nr']}): sv={np.round(svs[:10], 3)}"
              f"  cum-energy first 8: {np.round(cum[:8], 3)}", flush=True)
        res["sv_sq2"] = svs
        res["Km_sq2"] = Kms
    np.savez(f"{OUT}/op_shapes_t{tile}.npz", **res)
    print(f"  static tile {tile} in {time.time() - t0:.0f}s", flush=True)

if os.environ.get("STATIC_ONLY"):
    raise SystemExit(0)


# ── WALK-FORWARD ─────────────────────────────────────────────────────────
def amp_features(P, V, lo, hi):
    Lz = P["X"][lo:hi][:, P["ladder"]].reshape(hi - lo, len(P["labels"]),
                                               P["nr"])
    C = np.einsum("ncr,kr->nck", Lz, V, optimize=True).reshape(
        hi - lo, -1)
    return np.hstack([np.ones((hi - lo, 1)), C,
                      P["X"][lo:hi][:, P["har"]], P["X"][lo:hi][:, P["cal"]]])


def amp_predict(P, V, t, hi):
    """Per-bar sliding ridge (alpha=1, free intercept) on the amplitude model."""
    A = amp_features(P, V, t - W, hi)
    G = A[:W].T @ A[:W]
    c = A[:W].T @ P["y"][t - W : t]
    out = np.empty(hi - t)
    for i in range(hi - t):
        cf = ridge_solve(G, c)
        out[i] = A[W + i] @ cf
        if i < hi - t - 1:
            an, ao = A[W + i], A[i]
            G += np.outer(an, an) - np.outer(ao, ao)
            c += an * P["y"][t + i] - ao * P["y"][t - W + i]
    return out


ALL_ARMS = ["ctrl", "proj", "rrr0", "rrrS", "rrrF"] + (
    ["sq2"] if PS is not None else [])
ARMS = [a for a in os.environ.get("ARMS", ",".join(ALL_ARMS)).split(",") if a]
lamb = 1.0 - 1.0 / EFF

for tile in TILES:
    n0, n1 = TILE_BOUNDS[tile]
    print(f"\n=== WALK-FORWARD tile {tile} [{n0},{n1}) arms={ARMS} ===",
          flush=True)
    t0 = time.time()
    state = {a: {} for a in ARMS}
    preds = {a: np.empty(n1 - n0) for a in ARMS}
    for t in range(n0, n1, CADENCE):
        hi = min(t + CADENCE, n1)
        G, c, yy = probe_gram(PB, t)
        Km = probe_km(PB, G, c)
        if PS is not None and "sq2" in ARMS:
            Gs, cs, yys = probe_gram(PS, t)
            Kms = probe_km(PS, Gs, cs)
        for a in ARMS:
            st = state[a]
            if a == "ctrl":
                st["Kbar"] = Km if "Kbar" not in st else \
                    lamb * st["Kbar"] + (1 - lamb) * Km
                _, Vuse = svd_shapes(st["Kbar"], K)
            elif a == "sq2":
                st["Kbar"] = Kms if "Kbar" not in st else \
                    lamb * st["Kbar"] + (1 - lamb) * Kms
                _, Vuse = svd_shapes(st["Kbar"], K)
            else:
                if a == "proj":
                    _, Vf = svd_shapes(Km, K)
                else:
                    lam, sx = {"rrr0": (0.0, False), "rrrS": (LAM_REL, True),
                               "rrrF": (LAM_REL, False)}[a]
                    Vf, _, _ = als_rrr(PB, G, c, yy, K, lam, sx,
                                       V0=st.get("Vprev"), iters=4)
                    st["Vprev"] = Vf
                P5 = Vf.T @ Vf
                st["Pbar"] = P5 if "Pbar" not in st else \
                    lamb * st["Pbar"] + (1 - lamb) * P5
                ew, EV = np.linalg.eigh(st["Pbar"])
                Vuse = EV[:, ::-1][:, :K].T
            Ppanel = PS if a == "sq2" else PB
            preds[a][t - n0 : hi - n0] = amp_predict(Ppanel, Vuse, t, hi)
        if (t - n0) // CADENCE % 3 == 0:
            print(f"  seg t={t} done ({time.time() - t0:.0f}s)", flush=True)
    losses = {}
    for a in ARMS:
        pr, tr = apply_duan_smearing(preds[a], PB["y"][n0:n1], PB["b"][n0:n1])
        mm = forecast_metrics(pr, tr)
        losses[a] = qlike_per_bar(pr, tr)
        sfx = "" if K == 5 else f"K{K}"
        np.savez(f"{OUT}/op_preds_{a}{sfx}_t{n0}.npz", pred_raw=pr,
                 true_raw=tr)
        line = f"  {a:<5} qlike={mm['qlike']:.6f} mz={mm['mz_beta']:.3f}"
        if a != "ctrl" and "ctrl" in losses:
            r = dm_test(losses[a], losses["ctrl"], h=1)
            line += (f"  vs ctrl: diff={r['mean_diff']:+.6f} "
                     f"dm={r['dm']:+.2f} p={r['p']:.4f}")
        print(line, flush=True)
    print(f"  tile {tile} walk-forward in {time.time() - t0:.0f}s", flush=True)
