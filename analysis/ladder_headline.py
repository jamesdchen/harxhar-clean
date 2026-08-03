"""Three-rung ladder against base-2 HAR. OLS, calendar only, per-bar refit.

A deliberately small experiment: no exogenous channels, no penalty, no tuning.
The only thing that varies is how many rungs the lag ladder has.

PER-BAR REFIT, EXACTLY. The paper's battery refits "at every time step via the
closed-form solution" while everything else in this session refits every 1,000
bars, and that difference has never been controlled. At these widths it does
not need to be approximated: a sliding window changes the Gram by rank two --
one row leaves, one enters -- so the inverse can be updated in O(p^2) by two
Sherman-Morrison steps instead of re-solved in O(p^3). At p around 20 that is
roughly 500 flops per bar, so all 219,000 refits cost less than a single
1,000-bar walk of the wide designs. The fastest method here is also the correct
protocol, which is why it is worth using.

Rank-2 updating accumulates floating-point error, so the inverse is recomputed
exactly every REFRESH bars and the drift at each checkpoint is reported rather
than assumed -- a rolling update whose error is unmeasured is how a fast path
quietly becomes a wrong one.

THE LADDERS. "Three rungs" is ambiguous on this panel, so both readings are
run rather than one being chosen silently:

  classic-3   {48, 240, 1056} bars = day, week, month at 48 bars/day, the
              HAR of Corsi (2009) as originally specified
  geo-3       {1, 45, 2048} bars, a three-rung geometric ladder spanning the
              same range as base-2, so the comparison isolates RUNG COUNT
              rather than confounding it with the span
  base-2      {1, 2, 4, ... 2048}, twelve rungs, this project's ladder

Only geo-3 against base-2 is a clean count comparison. classic-3 differs in
both count and placement and is included because it is the ladder the
literature actually uses.

TARGET. Diurnally adjusted, sqrt-transformed, winsorized at 1/99 -- the
interior optimum measured under a fixed truth, where the project's default
5/95 costs 0.00304 and removing the clip entirely helps not at all. Every arm
is scored against ONE truth, the unclipped series, so the comparison is of
fits and not of questions.
"""

from __future__ import annotations
import json, os, re, sys, time
import numpy as np
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import dm_test                       # noqa: E402
import src.features.transforms.target as TGT        # noqa: E402
from src.data.loading import load_raw_data          # noqa: E402

OUT_DIR = os.path.join(ROOT, "writeup", "stats")
CACHE = os.environ.get("LH_CACHE", "b2_mmap_warm")
W = 24000
REFRESH = 5000
NBLOCK = 8
LADDERS = {"classic-3": [48, 240, 1056],
           "geo-3": [1, 45, 2048],
           "base-2": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]}


def qlike_bar(rv, p):
    r = rv / p
    return r - np.log(r) - 1.0


def build_target(df, lo, hi):
    o_lo, o_hi, o_w = TGT.WINSOR_LOWER_Q, TGT.WINSOR_UPPER_Q, TGT.rolling_winsorize
    try:
        if lo is None:
            TGT.rolling_winsorize = lambda s, **k: s
        else:
            TGT.WINSOR_LOWER_Q, TGT.WINSOR_UPPER_Q = lo, hi
        y, b = TGT.robust_transform(df, "RV", use_transform=True, use_diurnal=True,
                                    winsor_window=240, is_target=True)
    finally:
        TGT.WINSOR_LOWER_Q, TGT.WINSOR_UPPER_Q = o_lo, o_hi
        TGT.rolling_winsorize = o_w
    return y.to_numpy(np.float64), b.to_numpy(np.float64)


def per_bar_ols(F, y, b, W, refresh=REFRESH):
    """Sliding-window OLS refit at EVERY bar by rank-2 inverse updates.

    Removing row u and adding row v each change X'X by a rank-one term, so

        Ainv += (Ainv u)(u' Ainv) / (1 - u' Ainv u)      remove
        Ainv -= (Ainv v)(v' Ainv) / (1 + v' Ainv v)      add

    keeps the inverse current in O(p^2). Recomputed exactly every `refresh`
    bars, with the worst observed relative drift returned so the fast path is
    checked rather than trusted.
    """
    n, p = F.shape
    A = np.hstack([np.ones((n, 1)), F])
    pp = p + 1
    out = np.full(n - W, np.nan)
    G = A[:W].T @ A[:W] + 1e-10 * np.eye(pp)
    c = A[:W].T @ y[:W]
    Ainv = np.linalg.inv(G)
    drift = 0.0
    for t in range(W, n):
        cf = Ainv @ c
        rr = y[t - W:t] - A[t - W:t] @ cf
        out[t - W] = ((A[t] @ cf) ** 2 + float(rr @ rr) / W) * b[t]
        u = A[t - W]; v = A[t]
        Au = Ainv @ u
        Ainv += np.outer(Au, Au) / (1.0 - float(u @ Au))
        Av = Ainv @ v
        Ainv -= np.outer(Av, Av) / (1.0 + float(v @ Av))
        c = c - u * y[t - W] + v * y[t]
        if (t - W + 1) % refresh == 0:
            lo = t - W + 1
            Ge = A[lo:lo + W].T @ A[lo:lo + W] + 1e-10 * np.eye(pp)
            Ae = np.linalg.inv(Ge)
            drift = max(drift, float(np.max(np.abs(Ainv - Ae))
                                     / max(1e-12, np.max(np.abs(Ae)))))
            Ainv = Ae
            c = A[lo:lo + W].T @ y[lo:lo + W]
    return out, drift


def main():
    t0 = time.time()
    D = os.path.join(ROOT, "results", CACHE)
    Xc = np.load(os.path.join(D, "X.npy"), mmap_mode="r")
    names = [str(s) for s in np.load(os.path.join(D, "names.npy"), allow_pickle=True)]
    n = len(np.load(os.path.join(D, "y.npy")))
    CALN = ["DOW_0", "DOW_1", "DOW_2", "DOW_3", "DOW_4", "hour",
            "is_overnight", "is_open", "is_close"]
    cal = np.asarray(Xc[:, [names.index(c) for c in CALN if c in names]], np.float64)
    del Xc

    raw = load_raw_data(os.path.join(ROOT, "data"), allow_missing=True).dropna(
        subset=["RV"]).reset_index(drop=True)
    off = len(raw) - n
    y99, b99 = build_target(raw, 0.01, 0.99)
    y_no, _ = build_target(raw, None, None)
    y = y99[off:off + n]; b = b99[off:off + n]
    truth = y_no[off:off + n] ** 2 * b
    print(f"cache {CACHE}: {n:,} rows; target winsorized 1/99, scored against "
          f"the UNCLIPPED truth", flush=True)

    # bar-level adjusted RV, from which every ladder is built identically
    s = pd.Series(y ** 2)
    lag = {}
    for L in sorted({v for vv in LADDERS.values() for v in vv}):
        lag[L] = np.sqrt(np.maximum(
            s.rolling(L, min_periods=L).mean().shift(1).to_numpy(), 0.0))

    res, preds = {}, {}
    for nm, rungs in LADDERS.items():
        F = np.column_stack([lag[L] for L in rungs] + [cal])
        ok = np.isfinite(F).all(1)
        F = np.where(np.isfinite(F), F, 0.0)
        pr, drift = per_bar_ols(F, y, b, W)
        pr[~ok[W:]] = np.nan
        preds[nm] = pr
        res[nm] = {"rungs": rungs, "cols": F.shape[1], "drift": drift}
        print(f"    {nm:>10} rungs={rungs} cols={F.shape[1]} "
              f"inverse drift {drift:.2e} ({time.time()-t0:.0f}s)", flush=True)

    tr = truth[W:]
    good = (tr > 0) & np.isfinite(tr)
    for k in preds: good &= np.isfinite(preds[k]) & (preds[k] > 0)
    idx = np.flatnonzero(good)
    L = {k: qlike_bar(tr[idx], preds[k][idx]) for k in preds}
    Q = {k: float(L[k].mean()) for k in L}
    print(f"\n  scored {len(idx):,} bars, per-bar refit ({n - W:,} refits each)\n")
    print(f"  {'ladder':>10}{'rungs':>7}{'cols':>6}{'QLIKE':>11}"
          f"{'vs base-2':>12}{'t':>8}")
    out = {"cache": CACHE, "n": int(len(idx)), "qlike": Q, "arms": res, "dm": {}}
    for nm in LADDERS:
        if nm == "base-2":
            print(f"  {nm:>10}{len(LADDERS[nm]):>7}{res[nm]['cols']:>6}"
                  f"{Q[nm]:>11.5f}{'--':>12}{'--':>8}")
            continue
        r = dm_test(L[nm], L["base-2"], h=1)
        out["dm"][nm] = {"diff": float(r["mean_diff"]), "t": float(r["t"]),
                         "p": float(r["p"])}
        print(f"  {nm:>10}{len(LADDERS[nm]):>7}{res[nm]['cols']:>6}"
              f"{Q[nm]:>11.5f}{r['mean_diff']:>+12.5f}{r['t']:>8.2f}")
    nb = len(idx) // NBLOCK
    print(f"\n  by era (positive = worse than base-2):")
    for nm in LADDERS:
        if nm == "base-2": continue
        d = L[nm] - L["base-2"]
        em = [float(d[i*nb:(i+1)*nb if i < NBLOCK-1 else len(idx)].mean())
              for i in range(NBLOCK)]
        out[f"era_{nm}"] = em
        print(f"    {nm:>10}: {['%+.5f' % v for v in em]}  "
              f"base-2 better in {sum(1 for v in em if v > 0)}/{NBLOCK}")
    print(f"\n  geo-3 vs base-2 isolates RUNG COUNT (same span);")
    print(f"  classic-3 differs in count AND placement.")
    with open(os.path.join(OUT_DIR, "ladder_headline.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote ladder_headline.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
