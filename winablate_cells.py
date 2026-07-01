"""One (window, bucket) cell of the FULL penalty ablation -- SLURM array task, self-contained.

MATCHES THE DEPLOYED BASE: refit=480 bars (10 trading days), the value from harunpen_prep.sbatch
(`prep ebm all_buckets 1000 1 480 slim enetreg2_harunpen`) -- NOT a coarse cadence. Warm-started
coordinate descent chains the fine refits (the deployed enet base's method; there is no rank-1 for L1,
which is why the pipeline uses this fine cadence rather than literal every-bar).

Efficiency: the HAR-out FWL projection (E_res, y_res) is penalty-INDEPENDENT, so it is computed ONCE
per refit block and reused across the whole alpha/l1 grid (otherwise refit=480 -> ~60x the fits).
Only blocks covering the FIXED eval [EVAL_START:n] are fit (their coefs are all the eval needs).

Reads SLURM_ARRAY_TASK_ID -> cell = CELLS[id] = (window, bucket). Writes results/winablate_cells/
cell_<id>.json. Self-contained: numpy + sklearn only. Run from repo root.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge

B = "results/covid_imp_rank/all_buckets"
HAR = {"har_ma_1", "har_ma_5", "har_ma_25", "har_ma_125", "har_ma_625", "har_ma_3125"}
OUTDIR = "results/winablate_r480"

WINDOWS = [48_000, 72_000, 96_000, 120_000, 144_000, 168_000]  # 1000/1500/2000/2500/3000/3500 d
EVAL_START = max(WINDOWS)  # 168000
BUCKETS = [
    "har_only", "moments", "liquidity", "market_ew", "market_vw",
    "vol_demand", "sentiment", "implied_vol", "all_buckets",
]
CELLS = [(w, b) for w in WINDOWS for b in BUCKETS]  # 54

REFIT = 480  # deployed cadence: 10 trading days (harunpen_prep.sbatch)
ITER, TOL = 2000, 1e-4  # deployed enet solver settings (_cadence_enet_har_unpen defaults)
RIDGE_A = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
LASSO_A = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
ENET_A = [1e-3, 3e-3, 1e-2]
ENET_L1 = [0.2, 0.4, 0.6, 0.8]


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
        return "liquidity" if any(x in fl for x in ("turnover", "spread")) else "market_ew"
    if "vwstock" in fl:
        return "liquidity" if any(x in fl for x in ("turnover", "spread")) else "market_vw"
    if any(x in fl for x in ("turnover", "spread", "volume", "numobs")):
        return "liquidity"
    if "sum" in fl or "ret" in fl or "bipow" in fl or "pret" in fl or "autocov" in fl:
        return "moments"
    return "structural"


def qlike(preds, y, base):
    f, yt, b = preds.astype(np.float64), y.astype(np.float64), base.astype(np.float64)
    smear = np.mean((yt - f) ** 2)
    pr, tr = (f**2 + smear) * b, (yt**2) * b
    m = (tr > 0) & (pr > 0)
    r = tr[m] / pr[m]
    return float(np.mean(r - np.log(r) - 1))


def eval_blocks(tw, n):
    """Cadence starts (tw + k*REFIT) whose block [s, s_next) intersects the fixed eval [EVAL_START:n)."""
    alls = list(range(tw, n, REFIT))
    out = []
    for i, s in enumerate(alls):
        s_end = alls[i + 1] if i + 1 < len(alls) else n
        if s_end > EVAL_START:
            out.append((s, s_end))
    return out


def har_only_qlike(X, y, base, tw, har_idx):
    """Cadence OLS on the 6 HAR mains (penalty irrelevant), eval-block matvec."""
    n = len(X)
    Xh = np.ascontiguousarray(X[:, har_idx])
    oos = np.full(n - EVAL_START, np.nan)
    for s, s_end in eval_blocks(tw, n):
        Xw, yw = Xh[s - tw : s], y[s - tw : s]
        xb, yb = Xw.mean(0), float(yw.mean())
        b, *_ = np.linalg.lstsq(Xw - xb, yw - yb, rcond=None)
        lo, hi = max(s, EVAL_START), s_end
        oos[lo - EVAL_START : hi - EVAL_START] = Xh[lo:hi] @ b + (yb - xb @ b)
    assert not np.isnan(oos).any()
    return qlike(oos, y[EVAL_START:], base[EVAL_START:])


def sweep_penalties(X, y, base, tw, hm):
    """FWL-amortized: one HAR-out projection per block, reused across the whole penalty grid.
    Warm-started estimators chain across blocks (per config). Returns {penalty: (best_q, best_cfg)}."""
    n, p = X.shape
    hm = np.asarray(hm, dtype=bool)
    em = ~hm
    # config list: (key, penalty, estimator-or-None-for-ridge, alpha, l1)
    cfgs = []
    ests = {}
    for a in RIDGE_A:
        cfgs.append((f"ridge_{a:g}", "ridge", a, None))
    for a in LASSO_A:
        k = f"lasso_{a:g}"
        cfgs.append((k, "lasso", a, 1.0))
        ests[k] = ElasticNet(alpha=a, l1_ratio=1.0, fit_intercept=False, warm_start=True, max_iter=ITER, tol=TOL)
    for a in ENET_A:
        for l1 in ENET_L1:
            k = f"enet_{a:g}_{l1:g}"
            cfgs.append((k, "enet", a, l1))
            ests[k] = ElasticNet(alpha=a, l1_ratio=l1, fit_intercept=False, warm_start=True, max_iter=ITER, tol=TOL)
    oos = {k: np.full(n - EVAL_START, np.nan) for k, *_ in cfgs}
    blocks = eval_blocks(tw, n)
    for s, s_end in blocks:
        Xw, yw = X[s - tw : s], y[s - tw : s]
        xbar, ybar = Xw.mean(0), float(yw.mean())
        Xc, yc = Xw - xbar, yw - ybar
        H = np.ascontiguousarray(Xc[:, hm])
        E = np.ascontiguousarray(Xc[:, em])
        Hpinv = np.linalg.pinv(H)
        bH_y, bH_E = Hpinv @ yc, Hpinv @ E
        E_res, y_res = E - H @ bH_E, yc - H @ bH_y  # penalty-independent -> computed ONCE per block
        lo, hi = max(s, EVAL_START), s_end
        Xblk = X[lo:hi]
        for k, pen, a, l1 in cfgs:
            if pen == "ridge":
                bE = Ridge(alpha=a, fit_intercept=False).fit(E_res, y_res).coef_
            else:
                ests[k].fit(E_res, y_res)
                bE = ests[k].coef_
            beta = np.zeros(p)
            beta[hm] = bH_y - bH_E @ bE
            beta[em] = bE
            oos[k][lo - EVAL_START : hi - EVAL_START] = Xblk @ beta + (ybar - xbar @ beta)
    res = {}
    yv, bv = y[EVAL_START:], base[EVAL_START:]
    for pen in ("lasso", "ridge", "enet"):
        bq, bc = np.inf, None
        for k, cpen, a, l1 in cfgs:
            if cpen != pen:
                continue
            q = qlike(oos[k], yv, bv)
            if q < bq:
                bq, bc = q, [a, l1]
        res[pen] = (bq, bc)
    return res


def main():
    tid = int(os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("CELL_ID", "0")))
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
    print(f"cell {tid}: window={w} ({w//48}d) bucket={bk}  refit={REFIT}  eval[{EVAL_START}:{n}]", flush=True)

    rec = {"id": tid, "window": w, "days": w // 48, "bucket": bk, "refit": REFIT}
    if bk == "har_only":
        har_idx = [i for i, f in enumerate(feats) if f in HAR]
        rec["n_exog"] = 0
        rec["har_only"] = har_only_qlike(X, y, base, w, har_idx)
        print(f"  HAR_only={rec['har_only']:.5f}", flush=True)
    else:
        cols = list(range(len(feats))) if bk == "all_buckets" else [i for i in range(len(feats)) if fams[i] in ("HAR", bk)]
        Xsub = np.ascontiguousarray(X[:, cols])
        hm = [feats[cc] in HAR for cc in cols]
        rec["n_exog"] = len(cols) - sum(hm)
        res = sweep_penalties(Xsub, y, base, w, hm)
        for pen in ("lasso", "ridge", "enet"):
            rec[pen], rec[f"{pen}_cfg"] = res[pen][0], res[pen][1]
            print(f"  {pen:5s} best {res[pen][0]:.5f} cfg={res[pen][1]}", flush=True)

    rec["secs"] = round(time.time() - t0, 1)
    json.dump(rec, open(out_path, "w"), indent=2)
    print(f"wrote {out_path} ({rec['secs']}s)", flush=True)


if __name__ == "__main__":
    main()
