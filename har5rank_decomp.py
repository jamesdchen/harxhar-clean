"""Direct test (E) of the claim 'har5rank's -0.00021 is static magnitude curvature, not rank-specific'.

har5rank = causal per-slot ROLLING rank-Gauss of har_ma_5 (rank of today's vol among the trailing same-slot
window). The claim says this is reducible to a STATIC monotone transform of har_ma_5 (a spline). The rolling
window makes it history-dependent, so it may NOT be static. This script regresses har5rank on the BEST
static per-slot function of har_ma_5 (a decile-knot relu spline, per-slot in-sample fit = an UPPER BOUND on
static explainability) and reports R^2:
  R^2 ~ 1  -> har5rank IS static curvature -> a magnitude spline can reproduce the -0.00021 (claim holds)
  R^2 < 1  -> the residual is the ROLLING / relative-standardization part a static spline cannot reach
Reported on all rows and on the close rows (16-19, where har5rank_x_close actually lives).
"""

import json

import numpy as np

from src.features.transforms.rank_gauss import rolling_rank_gauss

PERIODS_PER_DAY = 48
TRAIN_WIN = 1000 * PERIODS_PER_DAY
QS = (10, 20, 30, 40, 50, 60, 70, 80, 90)


def spline_basis(x: np.ndarray) -> np.ndarray:
    taus = np.percentile(x, list(QS))
    return np.column_stack([np.maximum(x - float(t), 0.0) for t in taus])


def r2_static(
    rank: np.ndarray, h5: np.ndarray, slot: np.ndarray, mask: np.ndarray
) -> float:
    """Pooled per-slot best static fit: regress rank on [1, h5, spline(h5)] within each slot, sum SS."""
    ss_res = ss_tot = 0.0
    for s in np.unique(slot[mask]):
        sel = mask & (slot == s)
        if int(sel.sum()) < 100:
            continue
        ys = rank[sel]
        xb = np.column_stack([np.ones(int(sel.sum())), h5[sel], spline_basis(h5[sel])])
        beta, *_ = np.linalg.lstsq(xb, ys, rcond=None)
        res = ys - xb @ beta
        ss_res += float(np.sum(res**2))
        ss_tot += float(np.sum((ys - ys.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def main() -> None:
    b = "results/covid_imp_rank/all_buckets"
    m = json.load(open(f"{b}/meta.json"))["feats"]
    X = np.load(f"{b}/X_imp.npy", mmap_mode="r")
    h5 = np.asarray(X[:, m.index("har_ma_5")], dtype=np.float64)
    hour = np.asarray(X[:, m.index("hour")], dtype=np.float64)
    slot = np.rint(hour).astype(np.int64)

    # har5rank (ungated): causal per-slot rolling rank-Gauss -- replicates resid_amortized._har5rank_close
    win_slot = max(2, TRAIN_WIN // PERIODS_PER_DAY)
    rank = np.zeros(len(h5), dtype=np.float64)
    for s in np.unique(slot):
        idx = np.where(slot == s)[0]
        rank[idx] = rolling_rank_gauss(h5[idx][:, None], win_slot)[:, 0]

    allmask = np.ones(len(h5), dtype=bool)
    closemask = (hour >= 16) & (hour <= 19)
    r2_all = r2_static(rank, h5, slot, allmask)
    r2_close = r2_static(rank, h5, slot, closemask)
    print(
        "E-TEST: R2 of har5rank explained by the BEST static per-slot spline of har_ma_5",
        flush=True,
    )
    print(
        "  win_slot=%d  n=%d  n_close=%d" % (win_slot, len(h5), int(closemask.sum())),
        flush=True,
    )
    print("  R2 all rows   = %.4f" % r2_all, flush=True)
    print("  R2 close rows = %.4f" % r2_close, flush=True)
    print(
        "  (R2~1 => har5rank IS static curvature, spline reproducible; R2<1 => rolling/relative residual)",
        flush=True,
    )
    print("HAR5RANK_DECOMP_DONE", flush=True)


if __name__ == "__main__":
    main()
