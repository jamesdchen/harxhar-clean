"""Train-window ablation x penalty grid on a FIXED evaluation set (isolate the lookback effect).

SELF-CONTAINED (only numpy + sklearn; inlines the HAR-OLS+penalty FWL fit and Duan-QLIKE) so it runs
unmodified on CARC without touching the repo's resid_amortized.py. Reads results/covid_imp_rank/
all_buckets/ relative to CWD (run from the repo root).

Tests: does a LARGER train window help (more data) or HURT (nonstationarity -> stale data), and does the
optimal penalty strength go HEAVIER with more data (double-descent thesis) or LIGHTER (classical, p<<n)?
ALL windows score the SAME rows [EVAL_START:n] -- a fair lookback comparison (prior 0.134->0.130 was
confounded by the OOS period shifting). Rows = windows; cols = lasso/ridge/enet (best over each grid) +
HAR-only OLS. best-cfg recorded to read which way optimal shrinkage moves.

Resumable: checkpoints results/window_penalty_ablation.json after each window; re-run resumes. The eval
[EVAL_START:n] is ~2018-2024 (covid-inclusive) so absolutes (~0.14) sit above the [48000:] per-bucket
table -- a harder eval period, the price of a same-rows window comparison.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from sklearn.linear_model import ElasticNet, Ridge

B = "results/covid_imp_rank/all_buckets"
HAR = {"har_ma_1", "har_ma_5", "har_ma_25", "har_ma_125", "har_ma_625", "har_ma_3125"}
OUT = "results/window_penalty_ablation.json"

WINDOWS = [48_000, 96_000, 144_000]  # 1000 / 2000 / 3000 trading days
EVAL_START = max(WINDOWS)  # fixed eval [144000:n] for every window
REFIT = 30_000
ITER, TOL = 800, 3e-3
RIDGE_A = [30.0, 100.0, 300.0]
LASSO_A = [1e-3, 3e-3, 1e-2]
ENET_CFG = [(1e-3, 0.2), (3e-3, 0.3), (1e-2, 0.5)]


def qlike(preds, y, base):
    """Duan-smeared raw-variance QLIKE (inlined apply_duan_smearing + QLIKE)."""
    f, yt, b = preds.astype(np.float64), y.astype(np.float64), base.astype(np.float64)
    smear = np.mean((yt - f) ** 2)
    pr, tr = (f**2 + smear) * b, (yt**2) * b
    m = (tr > 0) & (pr > 0)
    r = tr[m] / pr[m]
    return float(np.mean(r - np.log(r) - 1))


def cadence_fit(X, y, tw, hm, alpha, l1, penalty):
    """Cadence-refit HAR-OLS (via FWL) + penalized exog. Inlined from _cadence_enet_har_unpen."""
    n, p = X.shape
    starts = list(range(tw, n, REFIT))
    hm = np.asarray(hm, dtype=bool)
    em = ~hm
    coefs = np.zeros((len(starts), p))
    intercepts = np.empty(len(starts))
    if penalty == "ridge":
        en = Ridge(alpha=alpha, fit_intercept=False)
    else:
        en = ElasticNet(
            alpha=alpha,
            l1_ratio=(1.0 if penalty == "lasso" else l1),
            fit_intercept=False,
            warm_start=True,
            max_iter=ITER,
            tol=TOL,
        )
    for i, t_r in enumerate(starts):
        Xw, yw = X[t_r - tw : t_r], y[t_r - tw : t_r]
        xbar, ybar = Xw.mean(0), float(yw.mean())
        Xc, yc = Xw - xbar, yw - ybar
        H = np.ascontiguousarray(Xc[:, hm])
        E = np.ascontiguousarray(Xc[:, em])
        Hpinv = np.linalg.pinv(H)
        bH_y, bH_E = Hpinv @ yc, Hpinv @ E
        en.fit(E - H @ bH_E, yc - H @ bH_y)
        bE = en.coef_
        beta = np.zeros(p)
        beta[hm] = bH_y - bH_E @ bE
        beta[em] = bE
        coefs[i], intercepts[i] = beta, ybar - xbar @ beta
    return np.asarray(starts), coefs, intercepts


def preds_on_eval(X, starts, coefs, intercepts):
    n = len(X)
    out = np.full(n - EVAL_START, np.nan)
    for i, t_r in enumerate(starts):
        t_end = int(starts[i + 1]) if i + 1 < len(starts) else n
        lo, hi = max(t_r, EVAL_START), t_end
        if lo < hi:
            out[lo - EVAL_START : hi - EVAL_START] = X[lo:hi] @ coefs[i] + intercepts[i]
    assert not np.isnan(out).any(), "eval rows not fully covered"
    return out


def cadence_ols(X, y, tw):
    n, p = X.shape
    starts = list(range(tw, n, REFIT))
    coefs = np.zeros((len(starts), p))
    intercepts = np.empty(len(starts))
    for i, t_r in enumerate(starts):
        Xw, yw = X[t_r - tw : t_r], y[t_r - tw : t_r]
        xb, yb = Xw.mean(0), float(yw.mean())
        b, *_ = np.linalg.lstsq(Xw - xb, yw - yb, rcond=None)
        coefs[i], intercepts[i] = b, yb - xb @ b
    return np.asarray(starts), coefs, intercepts


def best_penalty(X, y, base, tw, hm, pen):
    cfgs = (
        [(a, None) for a in RIDGE_A]
        if pen == "ridge"
        else [(a, 1.0) for a in LASSO_A]
        if pen == "lasso"
        else ENET_CFG
    )
    bq, bc = np.inf, None
    for a, l1 in cfgs:
        s, c, ic = cadence_fit(X, y, tw, hm, a, l1, pen)
        q = qlike(preds_on_eval(X, s, c, ic), y[EVAL_START:], base[EVAL_START:])
        if q < bq:
            bq, bc = q, (a, l1)
    return bq, bc


def load_ckpt():
    if os.path.exists(OUT):
        try:
            return {int(k): v for k, v in json.load(open(OUT))["results"].items()}
        except Exception:
            return {}
    return {}


def print_table(done):
    print("\n=== WINDOW ABLATION (fixed eval; best OOS-tuned QLIKE; HAR OLS) ===", flush=True)
    print("| train window | HAR-only | lasso | ridge | enet | best all_buckets |", flush=True)
    print("|---|---|---|---|---|---|", flush=True)
    for tw in WINDOWS:
        if tw not in done:
            continue
        r = done[tw]
        best = min(r["lasso"], r["ridge"], r["enet"])
        print(
            f"| {tw//48}d ({tw}) | {r['HAR_only']:.5f} | {r['lasso']:.5f} | {r['ridge']:.5f} | "
            f"{r['enet']:.5f} | {best:.5f} |",
            flush=True,
        )
    print("\nbest cfg by window (heavier vs lighter reg with more data?):", flush=True)
    for tw in WINDOWS:
        if tw in done:
            print(f"  {tw//48}d -> enet {done[tw]['enet_cfg']}  lasso {done[tw]['lasso_cfg']}  ridge {done[tw]['ridge_cfg']}", flush=True)


def main():
    done = load_ckpt()
    feats = json.load(open(f"{B}/meta.json"))["feats"]
    X = np.ascontiguousarray(np.load(f"{B}/X_imp.npy"))
    y = np.load(f"{B}/y.npy").astype(np.float64)
    base = np.load(f"{B}/base.npy").astype(np.float64)
    hm = [f in HAR for f in feats]
    har_idx = [i for i, f in enumerate(feats) if f in HAR]
    Xh = np.ascontiguousarray(X[:, har_idx])
    n = len(X)
    print(f"X{X.shape} FIXED eval [{EVAL_START}:{n}]={n-EVAL_START}  done={sorted(done)}", flush=True)

    for tw in WINDOWS:
        if tw in done:
            continue
        t0 = time.time()
        s, c, ic = cadence_ols(Xh, y, tw)
        row = {"days": tw // 48, "HAR_only": qlike(preds_on_eval(Xh, s, c, ic), y[EVAL_START:], base[EVAL_START:])}
        print(f"\n[tw={tw} ({tw//48}d)] HAR_only={row['HAR_only']:.5f}", flush=True)
        for pen in ("lasso", "ridge", "enet"):
            tp = time.time()
            q, cfg = best_penalty(X, y, base, tw, hm, pen)
            row[pen], row[f"{pen}_cfg"] = q, cfg
            print(f"  {pen:5s} best {q:.5f} cfg={cfg} ({time.time()-tp:.0f}s)", flush=True)
        done[tw] = row
        json.dump(
            {"eval_start": EVAL_START, "eval_rows": n - EVAL_START, "refit": REFIT,
             "results": {str(k): v for k, v in done.items()}},
            open(OUT, "w"), indent=2,
        )
        print(f"[tw={tw}] checkpointed ({time.time()-t0:.0f}s)", flush=True)

    print_table(done)
    if all(w in done for w in WINDOWS):
        print("ALL_WINDOWS_DONE", flush=True)


if __name__ == "__main__":
    main()
