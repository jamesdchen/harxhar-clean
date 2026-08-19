"""Stitch + score the direct multi-horizon shards (results/direct_mh).

Per model {a0, blk2}: concatenate shard_{i}ofN.npz in row order, then for
each horizon k = 1..K score under the paper's contract exactly as
probe_multih.py does (window-tiled MZ mean calibration on the sqrt scale,
EWMA(0.6) second moment seeded per window, raw-scale QLIKE with the
metrics.py exclusion rule; first chunk is calibration burn-in). Then:

  * per-horizon table: n, a0 QLIKE, blk2 QLIKE, dQ, DM (paired, blk2 vs a0)
  * the cumulative-sum path: at each origin the sum of the K per-bar
    raw-scale forecasts vs the realized sum -- the direct route to the
    remaining-variance total, scored by QLIKE + DM
  * both on the whole eval span and on the RTH-only origins

Writes results/direct_mh/dense_{byhorizon,path}.csv and prints them.
"""

from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import score_unification as su  # noqa: E402

RES = os.path.join(ROOT, "results", "direct_mh")
CHUNK = 2763
LAM = 0.6


def _load(model: str) -> dict[str, np.ndarray]:
    files = glob.glob(os.path.join(RES, model, "shard_*of*.npz"))
    if not files:
        raise SystemExit(f"no shards for {model}")
    ns = {int(re.search(r"of(\d+)\.npz$", f).group(1)) for f in files}
    if len(ns) != 1:
        raise SystemExit(f"{model}: mixed tilings {ns} -- clear stale shards")
    N = ns.pop()
    files = sorted(files, key=lambda f: int(re.search(r"shard_(\d+)of", f).group(1)))
    if len(files) != N:
        print(f"WARNING {model}: {len(files)}/{N} shards present -- partial stitch", flush=True)
    parts = [np.load(f) for f in files]
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("rows", "yhat", "y_true", "rv_raw_k", "B_k")}
    out["t"] = np.concatenate([p["t"] for p in parts])
    assert np.all(np.diff(out["rows"]) == 1), "rows not contiguous"
    return out


def _contract_losses(yhat: np.ndarray, y_sqrt: np.ndarray, base: np.ndarray) -> np.ndarray:
    """probe_multih._contract_losses, verbatim semantics."""
    n = len(yhat)
    loss = np.full(n, np.nan)
    bounds = [(i * CHUNK, min((i + 1) * CHUNK, n)) for i in range((n + CHUNK - 1) // CHUNK)]
    for k in range(1, len(bounds)):
        a0_, b0 = bounds[k - 1]
        a1, b1 = bounds[k]
        s = np.isfinite(yhat[a0_:b0]) & np.isfinite(y_sqrt[a0_:b0])
        a, b = 0.0, 1.0
        if s.sum() >= 3:
            f = su._ols2(yhat[a0_:b0][s], y_sqrt[a0_:b0][s])
            if f is not None:
                a, b = f
        m_prev = a + b * yhat[a0_:b0]
        e_prev = y_sqrt[a0_:b0] - m_prev
        seed = float(np.nanmean(e_prev**2))
        m = a + b * yhat[a1:b1]
        e = y_sqrt[a1:b1] - m
        s2 = np.empty(b1 - a1)
        cur = seed
        for i in range(b1 - a1):
            s2[i] = cur
            cur = LAM * cur + (1.0 - LAM) * e[i] ** 2
        f_raw = (m**2 + s2) * base[a1:b1]
        y_raw = y_sqrt[a1:b1] ** 2 * base[a1:b1]
        r = y_raw / f_raw
        ok = (y_raw > 0) & (r > 0) & np.isfinite(r)
        tmp = np.full(b1 - a1, np.nan)
        tmp[ok] = r[ok] - np.log(r[ok]) - 1.0
        loss[a1:b1] = tmp
    return loss


def main() -> None:
    A = _load("a0")
    B = _load("blk2")
    assert np.array_equal(A["rows"], B["rows"]), "a0/blk2 row ranges differ"
    K = A["yhat"].shape[1]
    t = pd.to_datetime(A["t"], utc=True).tz_convert("America/New_York")
    hod = t.hour + t.minute / 60.0
    rth = (hod >= 10.0) & (hod <= 15.0)  # RTH origins whose next K bars sit in the session
    print(f"stitched {len(A['rows'])} origins, K={K}, RTH origins {int(rth.sum())}", flush=True)

    rows = []
    LA = np.empty((len(A["rows"]), K))
    LB = np.empty_like(LA)
    for k in range(K):
        # sqrt-scale truth at t+k+1 is y_true[:,k]; baseline B_k[:,k]; identical for both arms
        ysq = A["y_true"][:, k].astype(np.float64)
        Bk = A["B_k"][:, k]
        LA[:, k] = _contract_losses(A["yhat"][:, k].astype(np.float64), ysq, Bk)
        LB[:, k] = _contract_losses(B["yhat"][:, k].astype(np.float64), ysq, Bk)
        for era, m in (("all", np.ones(len(ysq), bool)), ("rth", rth)):
            ok = m & np.isfinite(LA[:, k]) & np.isfinite(LB[:, k])
            d = LB[ok, k] - LA[ok, k]
            dm = float(su.dm_test(LB[ok, k], LA[ok, k], h=1)["dm"])
            rows.append({"era": era, "k": k + 1, "n": int(ok.sum()), "q_a0": float(LA[ok, k].mean()), "q_blk2": float(LB[ok, k].mean()), "dq": float(d.mean()), "dm": dm})
    byh = pd.DataFrame(rows)
    byh.to_csv(os.path.join(RES, "dense_byhorizon.csv"), index=False)

    # cumulative path: raw-scale per-bar forecasts (calibrated m^2*B is inside the
    # loss function; for the path use the same MZ+EWMA raw forecast) -- reuse:
    # rebuild raw forecasts by re-running the calibration and keeping f_raw.
    def raw_forecast(yhat: np.ndarray, y_sqrt: np.ndarray, base: np.ndarray) -> np.ndarray:
        n = len(yhat)
        f_out = np.full(n, np.nan)
        bounds = [(i * CHUNK, min((i + 1) * CHUNK, n)) for i in range((n + CHUNK - 1) // CHUNK)]
        for kk in range(1, len(bounds)):
            a0_, b0 = bounds[kk - 1]
            a1, b1 = bounds[kk]
            s = np.isfinite(yhat[a0_:b0]) & np.isfinite(y_sqrt[a0_:b0])
            a, b = 0.0, 1.0
            if s.sum() >= 3:
                f = su._ols2(yhat[a0_:b0][s], y_sqrt[a0_:b0][s])
                if f is not None:
                    a, b = f
            e_prev = y_sqrt[a0_:b0] - (a + b * yhat[a0_:b0])
            seed = float(np.nanmean(e_prev**2))
            m = a + b * yhat[a1:b1]
            e = y_sqrt[a1:b1] - m
            s2 = np.empty(b1 - a1)
            cur = seed
            for i in range(b1 - a1):
                s2[i] = cur
                cur = LAM * cur + (1.0 - LAM) * e[i] ** 2
            f_out[a1:b1] = (m**2 + s2) * base[a1:b1]
        return f_out

    FA = np.column_stack([raw_forecast(A["yhat"][:, k].astype(np.float64), A["y_true"][:, k].astype(np.float64), A["B_k"][:, k]) for k in range(K)])
    FB = np.column_stack([raw_forecast(B["yhat"][:, k].astype(np.float64), A["y_true"][:, k].astype(np.float64), A["B_k"][:, k]) for k in range(K)])
    Y = A["rv_raw_k"]  # raw realized at t+1..t+K
    prow = []
    for kk in (2, 4, 6, 8, 11):
        sumY = Y[:, :kk].sum(1)
        for name, F in (("a0", FA), ("blk2", FB)):
            pass
        fa = FA[:, :kk].sum(1)
        fb = FB[:, :kk].sum(1)
        for era, m in (("all", np.ones(len(sumY), bool)), ("rth", rth)):
            ok = m & np.isfinite(fa) & np.isfinite(fb) & (sumY > 0) & (fa > 0) & (fb > 0)
            qa = sumY[ok] / fa[ok] - np.log(sumY[ok] / fa[ok]) - 1.0
            qb = sumY[ok] / fb[ok] - np.log(sumY[ok] / fb[ok]) - 1.0
            dm = float(su.dm_test(qb, qa, h=1)["dm"])
            prow.append({"era": era, "bars_summed": kk, "n": int(ok.sum()), "q_a0": float(qa.mean()), "q_blk2": float(qb.mean()), "dq": float((qb - qa).mean()), "dm": dm})
    path = pd.DataFrame(prow)
    path.to_csv(os.path.join(RES, "dense_path.csv"), index=False)

    pd.set_option("display.width", 200)
    print("\n=== per-horizon k (bar t+k), blk2 vs a0 ===", flush=True)
    print(byh.round(4).to_string(index=False), flush=True)
    print("\n=== cumulative path (sum of first kk bars) ===", flush=True)
    print(path.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
