"""Duan-beating variant study (offline, causal replay on persisted chunks).

For each variant we replay the chunk chain exactly as _score_chunk_causal
does, swap layer 2/3, and pool per-bar QLIKE. Variants:
  none        y^2 B
  duan        (y^2 + s2_fit_inwin) B              [oracle: in-window fit-scale]
  contract    (m^2 + s2_prev_eval) B              [current contract]
  contract_noMZ (y^2 + s2_prev_eval) B
  duan_prev   (y^2 + s2_fit_prev) B               [fully causal Duan]
  duan_prev_MZ (m^2 + s2_fit_prev) B              [causal Duan + MZ]
  ewma{L}_MZ  (m^2 + s2_ewma(t)) B, L in grid     [causal adaptive, eval-scale]
  ewma{L}     (y^2 + s2_ewma(t)) B                [no MZ]
  oracle_mz_duan (m^2 + s2_fit_inwin) B           [upper-bound diagnostic]
"""

import os
import sys
import numpy as np

sys.path.insert(0, "/u/scratch/j/jamesdc1/harxhar-clean")
sys.path.insert(0, "/u/scratch/j/jamesdc1/harxhar-clean/experiments")
import score_unification as su

ARM = sys.argv[1] if len(sys.argv) > 1 else "a0_ols_har"
ROOT = "/u/scratch/j/jamesdc1/harxhar-clean/results/unification"
files = sorted(f for f in os.listdir(os.path.join(ROOT, ARM)) if su._CHUNK_RE.match(f))
chunks = [su._load_chunk(os.path.join(ROOT, ARM, f)) for f in files]
print(f"{ARM}: {len(chunks)} chunks")

# burn-in state for the first present window, then chain
prev = su._burnin_state(chunks[0])
LAMBDAS = [0.90, 0.95, 0.98, 0.99, 0.995]


def qlike(f, rv):
    return su.qlike_per_bar(f, rv)


losses = {}


def add(name, l):
    losses.setdefault(name, []).append(l)


for c in chunks[1:]:
    yhat, rv_raw, B, valid = c["yhat"], c["rv_raw"], c["baseline"], c["valid"]
    y_fit = c["y_fit"]
    y_raw = su._y_raw_of(rv_raw, B)
    # prev-window fits (contract chain)
    s = prev["valid"] & np.isfinite(prev["y_raw"]) & np.isfinite(prev["yhat"])
    a, b = 0.0, 1.0
    if s.sum() >= 3:
        fit = su._ols2(prev["yhat"][s], prev["y_raw"][s])
        if fit is not None:
            a, b = fit
    m = a + b * yhat
    sp = s & np.isfinite(prev["m"])
    s2_prev_eval = (
        float(np.mean((prev["y_raw"][sp] - prev["m"][sp]) ** 2))
        if sp.sum() >= 3
        else np.nan
    )
    # prev-window FIT-scale residual scalar (causal Duan): needs prev window's y_fit/yhat
    # prev dict from burnin/scorer lacks y_fit; rebuild from chunk directly:
    # we stored it below in prev['y_fit_resid']
    s2_prev_fit = prev.get("s2_fit", np.nan)
    # in-window fit scalar (oracle)
    sv = valid & np.isfinite(y_fit) & np.isfinite(yhat)
    s2_fit_inwin = float(np.mean((y_fit[sv] - yhat[sv]) ** 2)) if sv.any() else np.nan
    okb = valid & np.isfinite(B)
    n = len(yhat)

    def emit(name, f):
        l = qlike(f, rv_raw)
        l[~valid] = np.nan
        add(name, l)

    emit("none", np.where(okb, yhat**2 * B, np.nan))
    emit("duan", np.where(okb, (yhat**2 + s2_fit_inwin) * B, np.nan))
    emit("contract", np.where(okb, (m**2 + s2_prev_eval) * B, np.nan))
    emit("contract_noMZ", np.where(okb, (yhat**2 + s2_prev_eval) * B, np.nan))
    emit("duan_prev", np.where(okb, (yhat**2 + s2_prev_fit) * B, np.nan))
    emit("duan_prev_MZ", np.where(okb, (m**2 + s2_prev_fit) * B, np.nan))
    emit("oracle_mz_duan", np.where(okb, (m**2 + s2_fit_inwin) * B, np.nan))
    # EWMA adaptive variants: causal per-bar update within window, seeded prev
    e_running = None
    for L in LAMBDAS:
        s2_t = np.empty(n)
        cur = s2_prev_eval
        e_prev = np.nan
        for i in range(n):
            if np.isfinite(e_prev):
                cur = L * cur + (1 - L) * e_prev
            s2_t[i] = cur
            # residual at bar i known AFTER rv observed -> used for bar i+1
            if valid[i] and np.isfinite(y_raw[i]) and np.isfinite(m[i]):
                e_prev = (y_raw[i] - m[i]) ** 2
        emit(f"ewma{L}_MZ", np.where(okb, (m**2 + s2_t) * B, np.nan))
        emit(f"ewma{L}", np.where(okb, (yhat**2 + s2_t) * B, np.nan))
    # chain state for next window (mirror scorer; keep fit residual scalar)
    s2f = float(np.mean((y_fit[sv] - yhat[sv]) ** 2)) if sv.any() else np.nan
    prev = {"yhat": yhat, "y_raw": y_raw, "m": m, "valid": valid, "s2_fit": s2f}

print(f"\n=== pooled QLIKE by variant ({ARM}) ===")
rows = []
for name, parts in losses.items():
    l = np.concatenate(parts)
    rows.append((float(np.nanmean(l)), name))
for v, name in sorted(rows):
    print(f"  {name:16s} {v:.5f}")
