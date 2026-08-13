"""Multi-horizon probe: the horizon profile of the exogenous increment.

The campaign is h=1 end to end. This probe re-walks the three rungs of the
paper's ladder --- a0_ols_har (benchmark), blk2_user (two-block ridge),
blk3_user (the final three-block model) --- at h > 1, under the identical
frozen panel and the identical scoring protocol (MZ mean calibration + EWMA
second moment, chunk grid), and asks whether the exogenous increment grows,
holds, or decays with the horizon.

One SGE task per horizon: --horizon h. Writes results/multih/h{h}.npz with
per-bar losses for every arm (paired rows by construction), and prints the
pooled table. The cross-horizon summary is experiments/summarize_multih.py.
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "experiments"))

import numpy as np

import unification as U
import score_unification as su

WINDOW = 24000
CHUNK = 2763            # the v3 chunk grid
LAM = 0.6               # the protocol's second-moment decay
ARMS = ("a0_ols_har", "blk2_user", "blk3_user")


def _designs(p, window):
    """(design, alpha, tag) per arm, reusing the campaign's own builders."""
    out = {}
    for tag in ARMS:
        spec = U.ARMS[tag]
        if spec.kind == "ols":
            F, kept, _ = U._ols_design(p, spec)
            out[tag] = (F, 0.0, kept)
        elif spec.kind == "blocks":
            F, a_ref = U._build_design(p, spec, window, arm=tag)
            out[tag] = (F, a_ref, None)
        else:
            raise SystemExit(f"probe: unsupported kind {spec.kind} for {tag}")
    return out


def _contract_losses(yhat, y_raw, base):
    """Per-bar QLIKE under the scoring protocol: window-tiled MZ calibration,
    EWMA second moment seeded per window, identical exclusion rule.
    yhat/y_raw/base are full-length arrays over the EVAL SPAN (not the panel);
    the first chunk is the calibration burn-in and is left NaN."""
    n = len(yhat)
    loss = np.full(n, np.nan)
    bounds = [(i * CHUNK, min((i + 1) * CHUNK, n)) for i in range((n + CHUNK - 1) // CHUNK)]
    prev_m = prev_y = None
    for k in range(1, len(bounds)):
        a0, b0 = bounds[k - 1]
        a1, b1 = bounds[k]
        s = np.isfinite(yhat[a0:b0]) & np.isfinite(y_raw[a0:b0])
        a, b = 0.0, 1.0
        if s.sum() >= 3:
            f = su._ols2(yhat[a0:b0][s], y_raw[a0:b0][s])
            if f is not None:
                a, b = f
        m_prev = a + b * yhat[a0:b0]
        e_prev = y_raw[a0:b0] - m_prev
        seed = float(np.nanmean(e_prev**2))
        m = a + b * yhat[a1:b1]
        e = y_raw[a1:b1] - m
        s2 = np.empty(b1 - a1)
        cur = seed
        for i in range(b1 - a1):
            s2[i] = cur
            cur = LAM * cur + (1.0 - LAM) * e[i] ** 2
        f_raw = (m**2 + s2) * base[a1:b1]
        r = (y_raw[a1:b1] ** 2 * base[a1:b1]) / f_raw
        ok = (y_raw[a1:b1] ** 2 > 0) & (r > 0) & np.isfinite(r)
        l = r[ok] - np.log(r[ok]) - 1.0
        tmp = np.full(b1 - a1, np.nan)
        tmp[ok] = l
        loss[a1:b1] = tmp
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--span", type=int, default=60000)
    ap.add_argument("--out", default="results/multih")
    args = ap.parse_args()
    h = args.horizon
    s = h - 1

    p = U._load_panel()
    n = len(p.y)
    print(f"panel n={n}, horizon={h} bars, eval span={args.span}")

    designs = _designs(p, WINDOW)
    y_raw_full = np.sqrt(p.rv_raw / p.baseline)
    # memory: drop the 2.6 GB panel once the designs are built
    p.X = None
    import gc; gc.collect()

    # horizon shift (apply_horizon_shift convention): features at row i
    # predict the target at row i + (h-1). Shifted frame length n - s.
    y_h = p.y[s:]
    yr_h = y_raw_full[s:]
    rv_h = p.rv_raw[s:]
    B_h = p.baseline[s:]
    n_h = n - s

    lo = max(WINDOW, n_h - args.span)
    hi = n_h
    print(f"shifted n={n_h}, walking rows [{lo}, {hi}) = {hi - lo} bars")

    losses = {}
    for tag, (F, alpha, _) in designs.items():
        Fh = F[:-s] if s else F
        yhat = np.full(n_h, np.nan)
        yhat[lo:hi] = U._walk_ridge(Fh, y_h, WINDOW, lo, hi, alpha=alpha) if alpha > 0 else _walk_mn(
            U, Fh, y_h, WINDOW, lo, hi
        )
        # score on the shifted raw-scale pair over the eval span
        span_slice = slice(lo, hi)
        loss = np.full(n_h, np.nan)
        l = _contract_losses(yhat[span_slice], yr_h[span_slice], B_h[span_slice])
        loss[lo + CHUNK : hi] = l[CHUNK:]  # burn-in chunk of the span unscored
        losses[tag] = loss
        print(f"{tag}: pooled QLIKE {np.nanmean(loss[lo:hi]):.5f}  (scored {np.isfinite(loss[lo:hi]).sum()} bars)")

    base_l = losses["a0_ols_har"]
    print(f"\n=== h={h} summary (paired vs a0, common scored bars) ===")
    for tag in ARMS:
        d = losses[tag] - base_l
        ok = np.isfinite(d)
        if tag == "a0_ols_har":
            print(f"{tag:12s} QLIKE {np.nanmean(base_l[ok]):.5f}")
            continue
        dm = su.dm_test(losses[tag][ok], base_l[ok], h=1)
        print(f"{tag:12s} QLIKE {np.nanmean(losses[tag][ok]):.5f}  dQ {np.nanmean(d[ok]):+.5f}  DM {dm['dm']:+.2f}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"h{h}.npz")
    np.savez(path, horizon=h, lo=lo, hi=hi, **{f"loss_{k}": v for k, v in losses.items()})
    print(f"wrote {path}")


def _walk_mn(U, F, y, window, lo, hi):
    """min-norm LS walk for the benchmark arm (alpha=0 path)."""
    yhat, _ = U._walk_ols(F, y, window, lo, hi, ["c"] * F.shape[1])
    return yhat


if __name__ == "__main__":
    main()
