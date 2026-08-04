"""EVERY-BAR (refit=1) window x penalty ablation cell -- SLURM array task, self-contained.

Literally refits every bar via the verified online machinery (no cadence, no fallback):
  * lasso / enet -> reclasso_har.enet_online (Garrigues online homotopy, warm add+remove per bar);
  * ridge        -> Sherman-Morrison rank-1 update of the ridged inverse (exact);
HAR + intercept are unpenalized (locked-active / no ridge) -> matches _cadence_enet_har_unpen but at
refit=1. Fixed eval [EVAL_START:n] (same rows every window). Periodic exact REFRESH controls float drift.

Reads SLURM_ARRAY_TASK_ID -> CELLS[id] = (window, bucket). Writes results/winablate_r1/cell_<id>.json.
Env EVAL_MAX caps eval rows (local smoke test). Verified bar-for-bar vs batch in verify_online.py.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from reclasso_har import enet_coef, enet_online

B = "results/covid_imp_rank/all_buckets"
HAR = {"har_ma_1", "har_ma_5", "har_ma_25", "har_ma_125", "har_ma_625", "har_ma_3125"}
OUTDIR = "results/winablate_full"  # battery + per-bar loss (DM/MCS-ready); QLIKE-only run was winablate_r1
WINDOWS = [48_000, 72_000, 96_000, 120_000, 144_000, 168_000]
EVAL_START = max(WINDOWS)
BUCKETS = [
    "har_only",
    "moments",
    "liquidity",
    "market_ew",
    "market_vw",
    "vol_demand",
    "sentiment",
    "implied_vol",
    "all_buckets",
]
CELLS = [(w, b) for w in WINDOWS for b in BUCKETS]
RIDGE_A = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
LASSO_A = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
ENET_A = [1e-3, 3e-3, 1e-2]
ENET_L1 = [0.2, 0.4, 0.6, 0.8]
REFRESH = 2000  # exact re-anchor cadence (drift control, not a per-bar fallback)


def fam(f):
    if f in HAR:
        return "HAR"
    fl = f.lower()
    if "stocktwits" in fl:
        return "sentiment"
    if "voldemand" in fl:
        return "vol_demand"
    if "vvix" in fl or "vix" in fl:
        return "implied_vol"
    if "ewstock" in fl:
        return (
            "liquidity" if any(x in fl for x in ("turnover", "spread")) else "market_ew"
        )
    if "vwstock" in fl:
        return (
            "liquidity" if any(x in fl for x in ("turnover", "spread")) else "market_vw"
        )
    if any(x in fl for x in ("turnover", "spread", "volume", "numobs")):
        return "liquidity"
    if "sum" in fl or "ret" in fl or "bipow" in fl or "pret" in fl or "autocov" in fl:
        return "moments"
    return "structural"


def qlike(preds, yy, bb):
    f = np.asarray(preds)
    smear = np.mean((yy - f) ** 2)
    pr, tr = (f**2 + smear) * bb, (yy**2) * bb
    mk = (tr > 0) & (pr > 0)
    r = tr[mk] / pr[mk]
    return float(np.mean(r - np.log(r) - 1))


def batch_theta(Xa, yr, locked, exog_aug, a, l1):
    """Batch HAR-unpenalized enet on an augmented window (FWL pinv + enet_coef). Cold seed / refresh."""
    Hh, Ee = Xa[:, locked], Xa[:, exog_aug]
    Hp = np.linalg.pinv(Hh)
    bHy, bHE = Hp @ yr, Hp @ Ee
    Eres, yres = Ee - Hh @ bHE, yr - Hh @ bHy
    bE = enet_coef(Eres.T @ Eres, Eres.T @ yres, len(yr), a, l1)
    th = np.zeros(Xa.shape[1])
    th[locked] = bHy - bHE @ bE
    th[exog_aug] = bE
    return th


def run_lasso_enet(Xaug, y, tw, locked, exog_aug, a, l1, n, eval_lo, eval_hi):
    mu = tw * a * l1
    lam2 = tw * a * (1.0 - l1)
    mu_vec = np.zeros(Xaug.shape[1])
    mu_vec[exog_aug] = mu
    ei = np.where(exog_aug)[0]
    preds = np.empty(eval_hi - eval_lo)
    Gr = c = th = None
    A = s = None
    for pos, t in enumerate(range(eval_lo, eval_hi)):
        lo = t - tw
        if pos == 0 or (t - eval_lo) % REFRESH == 0:  # cold seed / exact re-anchor
            Xa = Xaug[lo:t]
            Gr = Xa.T @ Xa
            Gr[ei, ei] += lam2
            c = Xa.T @ y[lo:t]
            th = batch_theta(Xa, y[lo:t], locked, exog_aug, a, l1)
            A = list(np.where((np.abs(th) > 1e-9) | locked)[0])
            s = [0.0 if locked[j] else float(np.sign(th[j])) for j in A]
        else:  # slide one bar: add row t-1, remove row t-1-tw
            ua, ur = Xaug[t - 1], Xaug[t - 1 - tw]
            th, A, s = enet_online(
                Gr, c, mu_vec, ua, float(y[t - 1]), +1.0, th, A, s, locked
            )
            th, A, s = enet_online(
                Gr, c, mu_vec, ur, float(y[t - 1 - tw]), -1.0, th, A, s, locked
            )
        preds[pos] = Xaug[t] @ th
    return preds


def run_ridge(Xaug, y, tw, exog_aug, a, n, eval_lo, eval_hi):
    lam2 = a  # sklearn Ridge(alpha=a): (XtX + a I) theta = Xty  (no 1/n normalization, unlike enet)
    ei = np.where(exog_aug)[0]
    D = np.zeros(Xaug.shape[1])
    D[ei] = lam2
    preds = np.empty(eval_hi - eval_lo)
    K = c = None
    for pos, t in enumerate(range(eval_lo, eval_hi)):
        lo = t - tw
        if pos == 0 or (t - eval_lo) % REFRESH == 0:
            Xa = Xaug[lo:t]
            G = Xa.T @ Xa
            G[np.diag_indices_from(G)] += D
            K = np.linalg.inv(G)
            c = Xa.T @ y[lo:t]
        else:
            u, v = Xaug[t - 1], Xaug[t - 1 - tw]
            Ku = K @ u
            K = K - np.outer(Ku, Ku) / (1.0 + u @ Ku)
            c = c + u * float(y[t - 1])
            Kv = K @ v
            K = K + np.outer(Kv, Kv) / (1.0 - v @ Kv)
            c = c - v * float(y[t - 1 - tw])
        preds[pos] = Xaug[t] @ (K @ c)
    return preds


def main():
    if "SGE_TASK_ID" in os.environ:  # SGE arrays are 1-based
        tid = int(os.environ["SGE_TASK_ID"]) - 1
    else:
        tid = int(os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("CELL_ID", "0")))
    eval_max = int(os.environ.get("EVAL_MAX", "0"))
    w, bk = CELLS[tid]
    os.makedirs(OUTDIR, exist_ok=True)
    out_path = f"{OUTDIR}/cell_{tid:02d}.json"
    t0 = time.time()
    feats = json.load(open(f"{B}/meta.json"))["feats"]
    fams = [fam(f) for f in feats]
    X = np.ascontiguousarray(np.load(f"{B}/X_imp.npy"))
    y = np.load(f"{B}/y.npy").astype(np.float64)
    base = np.load(f"{B}/base.npy").astype(np.float64)
    n = len(X)
    eval_hi = min(n, EVAL_START + eval_max) if eval_max else n
    yv, bv = y[EVAL_START:eval_hi], base[EVAL_START:eval_hi]
    print(
        f"cell {tid}: w={w}({w // 48}d) bucket={bk} eval[{EVAL_START}:{eval_hi}]",
        flush=True,
    )

    if bk == "har_only":
        cols = [i for i, f in enumerate(feats) if f in HAR]
    elif bk == "all_buckets":
        cols = list(range(len(feats)))
    else:
        cols = [i for i in range(len(feats)) if fams[i] in ("HAR", bk)]
    ones = np.ones((n, 1))
    Xaug = np.ascontiguousarray(np.concatenate([ones, X[:, cols]], axis=1))
    del X  # free the 1 GB source; everything downstream uses Xaug
    fnames = ["__intercept__"] + [feats[i] for i in cols]
    locked = np.array([fn == "__intercept__" or fn in HAR for fn in fnames])
    exog_aug = ~locked
    rec = {
        "id": tid,
        "window": w,
        "days": w // 48,
        "bucket": bk,
        "n_exog": int(exog_aug.sum()),
    }

    def battery(preds):
        """Full metric battery (raw-variance, Duan-smeared) + per-bar QLIKE loss (for DM)."""
        f = np.asarray(preds)
        smear = np.mean((yv - f) ** 2)
        pr, tr = (f**2 + smear) * bv, (yv**2) * bv
        err = tr - pr
        msk = (pr > 0) & (tr > 0)
        r = tr[msk] / pr[msk]
        lossbar = np.full(len(tr), np.nan)
        lossbar[msk] = r - np.log(r) - 1.0
        ssb = float(np.sum((tr - tr.mean()) ** 2))
        mm = {
            "qlike": float(np.nanmean(lossbar)),
            "mse": float(np.mean(err**2)),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "hmse": float(np.mean((r - 1.0) ** 2)),
            "hmae": float(np.mean(np.abs(r - 1.0))),
            "oos_r2": (1.0 - float(np.sum(err**2)) / ssb) if ssb > 0 else float("nan"),
            "n": int(len(tr)),
        }
        return mm, lossbar

    best, per_bar = {}, {}
    if bk == "har_only":  # no exog -> HAR-only OLS; lasso/ridge/enet identical
        mm, lb = battery(run_ridge(Xaug, y, w, exog_aug, 0.0, n, EVAL_START, eval_hi))
        rec["har_only"] = mm["qlike"]
        rec["har_only_metrics"] = mm
        per_bar["har_only"] = lb
        print(f"  HAR_only={mm['qlike']:.5f} oos_r2={mm['oos_r2']:.4f}", flush=True)
    else:

        def consider(pen, preds, cfg):
            mm, lb = battery(preds)
            if pen not in best or mm["qlike"] < best[pen][0]["qlike"]:
                best[pen] = (mm, cfg, lb)

        for a in LASSO_A:
            consider(
                "lasso",
                run_lasso_enet(
                    Xaug, y, w, locked, exog_aug, a, 1.0, n, EVAL_START, eval_hi
                ),
                [a, 1.0],
            )
        for a in RIDGE_A:
            consider(
                "ridge",
                run_ridge(Xaug, y, w, exog_aug, a, n, EVAL_START, eval_hi),
                [a, None],
            )
        for a in ENET_A:
            for l1 in ENET_L1:
                consider(
                    "enet",
                    run_lasso_enet(
                        Xaug, y, w, locked, exog_aug, a, l1, n, EVAL_START, eval_hi
                    ),
                    [a, l1],
                )
        for pen, (mm, cfg, lb) in best.items():
            rec[pen], rec[f"{pen}_cfg"], rec[f"{pen}_metrics"] = mm["qlike"], cfg, mm
            per_bar[pen] = lb
        print(
            f"  lasso={rec['lasso']:.5f} ridge={rec['ridge']:.5f} enet={rec['enet']:.5f} "
            f"(enet oos_r2={rec['enet_metrics']['oos_r2']:.4f})",
            flush=True,
        )

    np.savez(
        f"{OUTDIR}/cell_{tid:02d}_lossbar.npz", **per_bar
    )  # per-bar QLIKE loss -> DM offline

    rec["secs"] = round(time.time() - t0, 1)
    json.dump(rec, open(out_path, "w"), indent=2)
    print(f"wrote {out_path} ({rec['secs']}s)", flush=True)


if __name__ == "__main__":
    main()
