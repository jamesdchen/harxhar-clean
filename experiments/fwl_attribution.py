"""Sequential-FWL attribution layer for the fitted enetreg2 base.

Decomposes enetreg2's predictive signal into interpretable feature BLOCKS via the FWL operationalization:
nested causal walk-forward re-fits (residualizing each block on the priors == adding it after them), same
fixed-alpha / cadence / train_win as enetreg2, so the full row reproduces the 0.12314 base-alone QLIKE.

Why nested re-fits and NOT decomposing the single fitted coefficient vector: the blocks are collinear, so
the enet splits shared variance arbitrarily across them -- the fitted coefficients mis-attribute. Nested
fits let each model use its blocks optimally, giving honest incremental contributions. FWL is exact in OLS;
under the enet these are penalized incremental contributions (consistent with the delivered model) -- the
clean additive decomposition lives in R^2 (variance, additive), QLIKE is the (non-additive) loss reference.

Blocks (ordered for the sequential / Type-I decomposition):
  HAR    -- the 6 nested sqrt-magnitude HAR MAs (har_ma_1..3125): baseline vol persistence
  REGIME -- HAR x {open,close} session-edge interactions (har_ma_*_x_{open,close})
  CUMRV  -- cumrv_x_close: the intraday vol-path accumulation
  EXOG   -- everything else (microstructure/sentiment/indicators/calendar): the "data" block
Reports Type-I (sequential, each block added in order) AND Type-III (leave-one-out: unique contribution
net of ALL others), plus the full-fit partial coefficients for REGIME+CUMRV (FWL: each is the partial
coefficient net of the rest).
"""

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)

import glob
import json
import os

import numpy as np

from resid_amortized import (
    CACHE_ROOT,
    PERIODS_PER_DAY,
    _HAR_COLS,
    _cadence_enet,
    _cadence_ridge_oos,
    _qlike,
)

TWD = 1000
REFIT = 480
TRAIN_WIN = TWD * PERIODS_PER_DAY
ORDER = ["HAR", "REGIME", "CUMRV", "EXOG"]


def block_of(f: str) -> str:
    if f in set(_HAR_COLS):
        return "HAR"
    if f == "cumrv_x_close":
        return "CUMRV"
    if f.endswith("_x_open") or f.endswith("_x_close"):
        return "REGIME"
    return "EXOG"


def main() -> None:
    cell = None
    for c in sorted(glob.glob(f"{CACHE_ROOT}/*_enetreg2_rf480_slim")):
        if os.path.exists(f"{c}/feats.json") and os.path.exists(f"{c}/Xs.npy"):
            cell = c
            break
    assert cell, "no enetreg2 cell with feats.json + Xs.npy found"
    Xs = np.ascontiguousarray(np.load(f"{cell}/Xs.npy").astype(np.float64))
    y = np.load(f"{cell}/y.npy")
    base = np.load(f"{cell}/base.npy")
    feats = json.load(open(f"{cell}/feats.json"))
    assert len(feats) == Xs.shape[1], "feats/Xs width mismatch %d/%d" % (
        len(feats),
        Xs.shape[1],
    )

    blocks: dict[str, list[int]] = {b: [] for b in ORDER}
    for i, f in enumerate(feats):
        blocks[block_of(f)].append(i)
    print("FWL attribution on cell: %s" % cell, flush=True)
    print(
        "block sizes: " + ", ".join("%s=%d" % (b, len(blocks[b])) for b in ORDER),
        flush=True,
    )

    yo = y[TRAIN_WIN:]
    sst = float(np.sum((yo - yo.mean()) ** 2))

    def fit(cols):
        Xsub = np.ascontiguousarray(Xs[:, sorted(cols)])
        starts, coefs, intercepts = _cadence_enet(Xsub, y, TRAIN_WIN, REFIT)
        oos = _cadence_ridge_oos(Xsub, TRAIN_WIN, starts, coefs, intercepts)
        q = _qlike(oos, y, base, TRAIN_WIN)
        r2 = 1.0 - float(np.sum((yo - oos) ** 2)) / sst
        return q, r2, np.asarray(coefs, dtype=np.float64).mean(axis=0)

    # ---- Type I: sequential (each block added in ORDER) ----
    print("\n=== TYPE I  (sequential / incremental) ===", flush=True)
    cum: list[int] = []
    prev_q = prev_r2 = None
    for b in ORDER:
        cum += blocks[b]
        q, r2, _ = fit(cum)
        dq = "" if prev_q is None else "  dQLIKE=%+.5f" % (q - prev_q)
        dr = "" if prev_r2 is None else "  dR2=%+.5f" % (r2 - prev_r2)
        print(
            "  +%-7s nfeat=%4d  QLIKE=%.5f  R2=%+.5f%s%s"
            % (b, len(cum), q, r2, dq, dr),
            flush=True,
        )
        prev_q, prev_r2 = q, r2

    full_cols = sum(blocks.values(), [])
    q_full, r2_full, coefs_full = fit(full_cols)
    print(
        "  FULL    nfeat=%4d  QLIKE=%.5f  R2=%+.5f   (sanity: enetreg2 base-alone = 0.12314)"
        % (len(full_cols), q_full, r2_full),
        flush=True,
    )

    # ---- Type III: leave-one-out (unique contribution net of ALL others) ----
    print("\n=== TYPE III  (leave-one-out / unique) ===", flush=True)
    full_set = set(full_cols)
    for b in ORDER:
        drop = set(blocks[b])
        cols = [i for i in full_cols if i not in drop]
        q, r2, _ = fit(cols)
        print(
            "  -%-7s nfeat=%4d  QLIKE=%.5f  unique_dQLIKE=%+.5f  unique_dR2=%+.5f"
            % (b, len(cols), q, q_full - q, r2_full - r2),
            flush=True,
        )

    # ---- Partial coefficients from the full fit (FWL: coef = partial, net of the rest) ----
    print(
        "\n=== PARTIAL COEFS (full fit, mean across refits) -- REGIME + CUMRV ===",
        flush=True,
    )
    pairs = [(feats[i], coefs_full[i]) for i in (blocks["REGIME"] + blocks["CUMRV"])]
    for nm, c in sorted(pairs, key=lambda t: -abs(t[1])):
        print("  COEF %-26s %+.5f" % (nm, c), flush=True)
    print("\nFWL_ATTRIBUTION_DONE", flush=True)


if __name__ == "__main__":
    main()
