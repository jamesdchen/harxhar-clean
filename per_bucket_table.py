"""DIAGNOSTIC table: best OUT-OF-SAMPLE-tuned QLIKE per (bucket, penalty), HAR held OLS.

Rows  = bucket variations (no-bucket / each exog family / all-buckets), on the local
        covid_imp_rank/all_buckets cache by column-subsetting (HAR mains + that family + its
        availability indicators).
Cols  = lasso / ridge / enet; each cell = MIN full-OOS QLIKE over that penalty's full alpha(/l1)
        grid -- i.e. the ORACLE ceiling (tuned against the test set on purpose; diagnostic only,
        NOT reportable OOS performance).

Fixed tw=48000 (1000d, all_buckets-optimal) for cross-row comparability. HAR-only row: no exog ->
all three penalties are the same HAR-OLS number. Penalty tuning is a non-lever (see
penalty_estimator_floor), so the columns will land within noise; the ROW deltas (information per
bucket) are the signal. Absolutes sit below deployed enetreg2 0.12314 (no regime/cumrv add-ons).
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())
from resid_amortized import PERIODS_PER_DAY, _cadence_enet_har_unpen  # noqa: E402
from src.evaluation.metrics import apply_duan_smearing  # noqa: E402

B = "results/covid_imp_rank/all_buckets"
HAR = {"har_ma_1", "har_ma_5", "har_ma_25", "har_ma_125", "har_ma_625", "har_ma_3125"}
OUT = "results/per_bucket_table.json"
TW = 1000 * PERIODS_PER_DAY  # 48000
REFIT = 20_000
ITER, TOL = 1500, 1e-3

# per-penalty grids (each cell = best over its grid, full-OOS)
RIDGE_A = [3.0, 10.0, 30.0, 100.0]
LASSO_A = [3e-4, 1e-3, 3e-3]
ENET_CFG = [(3e-4, 0.5), (1e-3, 0.2), (1e-3, 0.5), (3e-3, 0.5)]

BUCKETS = [
    "moments",
    "liquidity",
    "market_ew",
    "market_vw",
    "vol_demand",
    "sentiment",
    "implied_vol",
]


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


def _qlike(preds, y, base):
    pr, tr = apply_duan_smearing(preds, y, base)
    m = (tr > 0) & (pr > 0)
    r = tr[m] / pr[m]
    return float(np.mean(r - np.log(r) - 1))


def _oos(Xsub, starts, coefs, intercepts):
    n = len(Xsub)
    oos = np.empty(n - TW, dtype=np.float64)
    for i, t_r in enumerate(starts):
        t_end = int(starts[i + 1]) if i + 1 < len(starts) else n
        oos[t_r - TW : t_end - TW] = Xsub[t_r:t_end] @ coefs[i] + intercepts[i]
    return oos


def _cadence_ols(Xsub, y):
    n, p = Xsub.shape
    starts = list(range(TW, n, REFIT))
    coefs = np.zeros((len(starts), p))
    intercepts = np.empty(len(starts))
    for i, t_r in enumerate(starts):
        X = Xsub[t_r - TW : t_r]
        yt = y[t_r - TW : t_r]
        xb, yb = X.mean(0), float(yt.mean())
        b, *_ = np.linalg.lstsq(X - xb, yt - yb, rcond=None)
        coefs[i], intercepts[i] = b, yb - xb @ b
    return np.asarray(starts), coefs, intercepts


def _fit_q(Xsub, y, base, hm, a, l1, pen):
    s, c, ic = _cadence_enet_har_unpen(
        Xsub, y, TW, REFIT, hm, alpha=a, l1=l1, penalty=pen, max_iter=ITER, tol=TOL
    )
    return _qlike(_oos(Xsub, s, c, ic), y[TW:], base[TW:])


def _best(Xsub, y, base, hm, pen):
    """Best full-OOS QLIKE over the penalty's grid (oracle)."""
    if pen == "ridge":
        cfgs = [(a, None) for a in RIDGE_A]
    elif pen == "lasso":
        cfgs = [(a, 1.0) for a in LASSO_A]
    else:
        cfgs = ENET_CFG
    best_q, best_cfg = np.inf, None
    for a, l1 in cfgs:
        q = _fit_q(Xsub, y, base, hm, a, l1, pen)
        if q < best_q:
            best_q, best_cfg = q, (a, l1)
    return best_q, best_cfg


def main():
    t_all = time.time()
    feats = json.load(open(f"{B}/meta.json"))["feats"]
    fams = [fam(f) for f in feats]
    X = np.ascontiguousarray(np.load(f"{B}/X_imp.npy"))
    y = np.load(f"{B}/y.npy").astype(np.float64)
    base = np.load(f"{B}/base.npy").astype(np.float64)
    har_idx = [i for i, f in enumerate(feats) if f in HAR]
    print(f"loaded X{X.shape}  tw={TW} refit={REFIT}", flush=True)

    table = {}

    # no-bucket = HAR-only OLS (penalty irrelevant)
    Xh = np.ascontiguousarray(X[:, har_idx])
    s, c, ic = _cadence_ols(Xh, y)
    q_har = _qlike(_oos(Xh, s, c, ic), y[TW:], base[TW:])
    table["no_bucket (HAR only)"] = {
        "lasso": q_har,
        "ridge": q_har,
        "enet": q_har,
        "n_exog": 0,
    }
    print(f"\nno_bucket (HAR only): {q_har:.5f}", flush=True)

    for bk in BUCKETS + ["all_buckets"]:
        cols = (
            list(range(len(feats)))
            if bk == "all_buckets"
            else [i for i in range(len(feats)) if fams[i] in ("HAR", bk)]
        )
        Xsub = np.ascontiguousarray(X[:, cols])
        hm = [feats[cc] in HAR for cc in cols]
        n_exog = len(cols) - sum(hm)
        row = {"n_exog": n_exog}
        print(f"\n=== {bk} ({n_exog} exog) ===", flush=True)
        for pen in ("lasso", "ridge", "enet"):
            t0 = time.time()
            q, cfg = _best(Xsub, y, base, hm, pen)
            row[pen] = q
            row[f"{pen}_cfg"] = cfg
            print(
                f"  {pen:5s} best {q:.5f} cfg={cfg}  vs HAR {q - q_har:+.5f}  ({time.time() - t0:.0f}s)",
                flush=True,
            )
        table[bk] = row

    # markdown table
    print(
        "\n\n=== FINAL TABLE (best OOS-tuned QLIKE; HAR unpenalized; tw=48000) ===",
        flush=True,
    )
    print(f"HAR-only reference = {q_har:.5f}\n", flush=True)
    print("| bucket (n_exog) | lasso | ridge | enet | best vs HAR |", flush=True)
    print("|---|---|---|---|---|", flush=True)
    for name, row in table.items():
        best = min(row["lasso"], row["ridge"], row["enet"])
        dv = "" if name.startswith("no_bucket") else f"{best - q_har:+.5f}"
        print(
            f"| {name} ({row['n_exog']}) | {row['lasso']:.5f} | {row['ridge']:.5f} | {row['enet']:.5f} | {dv} |",
            flush=True,
        )

    json.dump(
        {"tw": TW, "refit": REFIT, "har_only": q_har, "table": table},
        open(OUT, "w"),
        indent=2,
    )
    print(f"\nwrote {OUT}  total {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
