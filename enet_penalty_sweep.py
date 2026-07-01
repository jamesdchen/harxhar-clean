"""Tuned ridge / lasso / enet on the EXOG block, with HAR held OLS -- the experiment that was
never cleanly run (the deployed base hardcodes alpha=1e-3, l1_ratio=0.2; the old DR sweep
`do_dimreduce_cell` penalized HAR, used unstandardized inputs, and had no pure Lasso).

Base = `_cadence_enet_har_unpen` (6 HAR mains OLS via FWL, exog penalized). Every config shares
that base; ONLY the exog-block penalty (type, alpha, l1_ratio) varies. QLIKE is Duan-smeared
raw-variance, identical recipe to `do_dimreduce_cell`.

Protocol (the harness lesson: tune cheap-on-a-slice, deploy fixed -- per-step alpha CV HURTS):
  Stage 1  slice grid   -- slice-OOS QLIKE for the full (type, alpha, l1) grid on an interior slice.
  Stage 2  full-OOS      -- re-score the top pick per family + the deployed incumbent on all rows.

Scientific hook: contribution 2 says the signal is DENSE, not low-rank (~120 eff dims / 529, PCA
truncation fails). Prediction: LASSO (sparse selection) should LOSE; RIDGE / low-l1 enet (dense
shrinkage) should win-or-tie. This tests that prediction through the penalty geometry directly.

Run:  /c/Users/james/miniconda3/envs/285J/python.exe enet_penalty_sweep.py
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

BUCKET = "all_buckets"
B = f"results/covid_imp_rank/{BUCKET}"
HAR = ("har_ma_1", "har_ma_5", "har_ma_25", "har_ma_125", "har_ma_625", "har_ma_3125")
OUT = "results/enet_penalty_sweep.json"

# ---- grid -----------------------------------------------------------------
# l1 axis = the sparsity question (0.2 incumbent -> 1.0 lasso). Pure-L2 / dense end is the RIDGE arm
# (closed-form, fast); near-ridge enet (l1<=0.1) is coordinate-descent's worst case (30s+, non-converging)
# and redundant with Ridge, so it is excluded. All remaining enet configs converge -> fair ranking.
ENET_L1 = [0.2, 0.5, 0.9, 1.0]  # 0.2 = incumbent ; 1.0 = lasso
ENET_A = [
    1e-3,
    3e-3,
    1e-2,
    3e-2,
]  # 1e-3 = incumbent (lower alpha probed in Stage 2 on the winner)
RIDGE_A = [
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
]  # ridge alpha is on its OWN scale
INCUMBENT = ("enet", 1e-3, 0.2)
LOWA_PROBE = (
    3e-4  # Stage-2 "is the base over-shrunk?" probe at the winner's l1 (converged)
)

# slice for tuning (interior, contiguous; pre-covid varied regime). Loose tol -> RANKING only.
SLICE_LO, SLICE_HI = 70_000, 150_000
SLICE_TW = 12_000  # ~250 trading days
SLICE_REFIT = 8_000
SLICE_ITER, SLICE_TOL = 500, 2e-3
# full-OOS confirm (matches do_dimreduce_cell: tw=1000d). Tight tol -> final numbers.
FULL_TW = 1000 * PERIODS_PER_DAY  # 48000
FULL_REFIT = 12_000
FULL_ITER, FULL_TOL = 3000, 1e-4


def _qlike(preds, y_oos, base_oos):
    """Duan-smeared raw-variance QLIKE (identical to do_dimreduce_cell)."""
    pr, tr = apply_duan_smearing(preds, y_oos, base_oos)
    m = (tr > 0) & (pr > 0)
    r = tr[m] / pr[m]
    return float(np.mean(r - np.log(r) - 1))


def _oos_preds(Xs, y, tw, refit, har_mask, alpha, l1, penalty, max_iter, tol):
    """Cadence-refit base -> OOS preds for rows [tw:n] (HAR OLS, exog `penalty`)."""
    starts, coefs, intercepts = _cadence_enet_har_unpen(
        Xs,
        y,
        tw,
        refit,
        har_mask,
        alpha=alpha,
        l1=l1,
        penalty=penalty,
        max_iter=max_iter,
        tol=tol,
    )
    n = len(Xs)
    oos = np.empty(n - tw, dtype=np.float64)
    for i, t_r in enumerate(starts):
        t_end = int(starts[i + 1]) if i + 1 < len(starts) else n
        oos[t_r - tw : t_end - tw] = Xs[t_r:t_end] @ coefs[i] + intercepts[i]
    return oos


def _score(Xs, y, base, tw, refit, har_mask, alpha, l1, penalty, max_iter, tol):
    preds = _oos_preds(Xs, y, tw, refit, har_mask, alpha, l1, penalty, max_iter, tol)
    return _qlike(preds, y[tw:], base[tw:])


def main():
    t_all = time.time()
    feats = json.load(open(f"{B}/meta.json"))["feats"]
    har_mask = [f in set(HAR) for f in feats]
    assert sum(har_mask) == 6, f"expected 6 HAR mains, got {sum(har_mask)}"
    Xs = np.ascontiguousarray(np.load(f"{B}/X_imp.npy").astype(np.float64))
    y = np.load(f"{B}/y.npy").astype(np.float64)
    base = np.load(f"{B}/base.npy").astype(np.float64)
    n = len(Xs)
    print(
        f"loaded X{Xs.shape} y{y.shape}  n={n}  HAR mains unpenalized (OLS)", flush=True
    )

    # ---- build config list ------------------------------------------------
    cfgs = []
    for a in ENET_A:
        for l1 in ENET_L1:
            cfgs.append(("lasso" if l1 == 1.0 else "enet", a, l1))
    for a in RIDGE_A:
        cfgs.append(("ridge", a, None))

    # ---- Stage 1: slice grid ---------------------------------------------
    Xsl = Xs[SLICE_LO:SLICE_HI]
    ysl = y[SLICE_LO:SLICE_HI]
    bsl = base[SLICE_LO:SLICE_HI]
    print(
        f"\n=== STAGE 1: slice grid  rows[{SLICE_LO}:{SLICE_HI}] tw={SLICE_TW} refit={SLICE_REFIT} "
        f"({len(cfgs)} configs) ===",
        flush=True,
    )
    slice_res = []
    for k, (pen, a, l1) in enumerate(cfgs):
        t0 = time.time()
        try:
            q = _score(
                Xsl,
                ysl,
                bsl,
                SLICE_TW,
                SLICE_REFIT,
                har_mask,
                a,
                l1,
                pen,
                SLICE_ITER,
                SLICE_TOL,
            )
        except Exception as e:  # noqa: BLE001
            q = float("nan")
            print(f"  [{k:2d}] {pen:5s} a={a:<7g} l1={l1} FAILED: {e}", flush=True)
            continue
        star = "  <-- incumbent" if (pen, a, l1) == INCUMBENT else ""
        slice_res.append({"penalty": pen, "alpha": a, "l1": l1, "qlike": q})
        print(
            f"  [{k:2d}] {pen:5s} a={a:<7g} l1={str(l1):5s} slice_qlike={q:.5f} "
            f"({time.time() - t0:.0f}s){star}",
            flush=True,
        )

    ok = [r for r in slice_res if np.isfinite(r["qlike"])]
    ok.sort(key=lambda r: r["qlike"])
    print("\n--- slice ranking (best first) ---", flush=True)
    for r in ok[:12]:
        print(
            f"  {r['penalty']:5s} a={r['alpha']:<7g} l1={str(r['l1']):5s} -> {r['qlike']:.5f}",
            flush=True,
        )

    # top pick per family + incumbent -> Stage 2
    def best_of(pred):
        c = [r for r in ok if pred(r)]
        return min(c, key=lambda r: r["qlike"]) if c else None

    best_enet = best_of(lambda r: r["penalty"] == "enet")
    picks = {
        "overall_best": ok[0],
        "best_ridge": best_of(lambda r: r["penalty"] == "ridge"),
        "best_enet": best_enet,
        "best_lasso": best_of(lambda r: r["penalty"] == "lasso"),
        "incumbent": next(
            (r for r in ok if (r["penalty"], r["alpha"], r["l1"]) == INCUMBENT),
            None,
        ),
        # "is the base over-shrunk?" -- less shrinkage than any grid alpha, at the winning enet l1
        "low_alpha_probe": (
            {"penalty": "enet", "alpha": LOWA_PROBE, "l1": best_enet["l1"]}
            if best_enet
            else None
        ),
    }
    stage2 = {}
    seen = set()
    for name, r in picks.items():
        if r is None:
            continue
        key = (r["penalty"], r["alpha"], r["l1"])
        if key in seen:
            continue
        seen.add(key)
        stage2[name] = r

    # ---- Stage 2: full-OOS confirm ---------------------------------------
    print(
        f"\n=== STAGE 2: full-OOS  tw={FULL_TW} refit={FULL_REFIT} "
        f"(baseline harunpen enetreg2 ~0.12316) ===",
        flush=True,
    )
    full_res = {}
    for name, r in stage2.items():
        t0 = time.time()
        q = _score(
            Xs,
            y,
            base,
            FULL_TW,
            FULL_REFIT,
            har_mask,
            r["alpha"],
            r["l1"],
            r["penalty"],
            FULL_ITER,
            FULL_TOL,
        )
        full_res[name] = {**r, "full_qlike": q}
        print(
            f"  {name:16s} {r['penalty']:5s} a={r['alpha']:<7g} l1={str(r['l1']):5s} "
            f"-> full_qlike={q:.5f}  ({time.time() - t0:.0f}s)",
            flush=True,
        )

    json.dump(
        {
            "slice": {"lo": SLICE_LO, "hi": SLICE_HI, "tw": SLICE_TW, "results": ok},
            "full": {"tw": FULL_TW, "results": full_res},
        },
        open(OUT, "w"),
        indent=2,
    )
    print(f"\nwrote {OUT}   total {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
