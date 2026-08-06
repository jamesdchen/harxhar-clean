"""Multiplicity-corrected inference for the two claims §10 flagged as max-picked.

§5's "vol-regime conditioning pays" (ΔR² +0.0024, DM-t +5.0) and §7.1's "interactions add +35%"
(DM-t +2.71) are both the **maximum of a search**, and a per-arm Diebold-Mariano t is the wrong
statistic for a maximum. This module recomputes each arm family, keeps the per-arm loss
differentials, and applies the two standard corrections:

* **Hansen SPA / White Reality Check** — the null that *no* arm beats the benchmark. The test
  statistic is ``max_j t_j``; its distribution comes from a bootstrap of the *recentred* loss
  differentials, so the p-value prices in the fact that the winner was chosen by looking.
* **Romano–Wolf step-down** — per-arm adjusted p-values that control the family-wise error rate,
  obtained from the bootstrap distribution of ``max_j |t_j|``, stepping down through the ordered
  arms. Strictly less conservative than Bonferroni and, unlike it, respects the very high
  cross-arm correlation here (neighbouring K's share most of their predictions).

Both bootstrap with a **circular block bootstrap**, block length one month (1,008 bars), blocks
drawn jointly across arms so cross-arm dependence is preserved. Blocks matter: 219k 30-min bars
are ~4,500 independent days, and an i.i.d. bootstrap would understate the sampling variation of
the loss differentials by roughly the square root of the block length.

Two honest limits on what this can deliver:

1. **The arm families here are a subset of what was actually searched.** Claim 1's family is the
   26 granularity-ladder arms (13 axis x K combinations, each as a separate fit and as a 50/50
   blend); the 13 gain-channel arms of §5 were searched against a related benchmark and are not
   re-run. Claim 2's family is the 8 group-penalty arms (4 penalties x {dynamic, static}); the 12
   clipped-ladder arms are on a different specification. Adding arms can only widen the null of
   the maximum, so **every adjusted p reported here is a lower bound** on the fully-corrected one.
2. **No amount of correction recovers a held-out sample.** These p-values price in the search
   *within each family*, not the ~100 specifications this study scored overall. A clean number
   needs the pre-registered protocol in §10 (search on <= 2020, decide once on 2021-2024).

Usage
-----
    python analysis/multiplicity.py --stage conditioning   # claim 1: does vol-regime survive?
    python analysis/multiplicity.py --stage interactions   # claim 2: does the +35% survive?
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.alpha_manifestation import (  # noqa: E402
    BUCKETS,
    COMBINE_WINDOW_DAYS,
    REFIT_EVERY,
    TW,
    _bin_walk_forward,
    _coarsen,
    _vol_regime,
)
from analysis.alpha_panel import CACHE_DIR, load_panel  # noqa: E402
from analysis.nl_sparsity import (  # noqa: E402
    REFIT,
    RIDGE_ALPHA,
    _pair_ic,
    _products,
    _upper,
    base_columns,
)
from analysis.wf import r2_oos, walk_forward  # noqa: E402
from src.features.transforms.target import PERIODS_PER_DAY  # noqa: E402

BLOCK = 21 * PERIODS_PER_DAY  # one month
N_BOOT = 2000
OUT = "results/alpha_manifestation"


def _hac_se(d: np.ndarray, lags: int = 480) -> np.ndarray:
    """Newey-West standard error of the mean, per column of ``d``."""
    n = len(d)
    dc = d - d.mean(0)
    s = (dc * dc).sum(0)
    for L in range(1, lags + 1):
        s += 2.0 * (1.0 - L / (lags + 1.0)) * (dc[L:] * dc[:-L]).sum(0)
    return np.sqrt(np.maximum(s, 0.0)) / n


def _block_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap index vector of length ``n``."""
    n_blocks = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(BLOCK)[None, :]).ravel() % n
    return idx[:n]


def spa_and_romano_wolf(
    d: np.ndarray, names: list[str], seed: int = 0
) -> tuple[float, pd.DataFrame]:
    """(SPA p-value, per-arm Romano-Wolf adjusted p-values) for loss differentials ``d``.

    ``d[:, j] > 0`` means arm ``j`` beats the benchmark on that bar (squared-error reduction).
    """
    n, J = d.shape
    mu, se = d.mean(0), _hac_se(d)
    t = mu / np.where(se > 0, se, np.inf)
    dc = d - mu  # recentred: the null is "no arm beats the benchmark"
    rng = np.random.default_rng(seed)
    boot_t = np.empty((N_BOOT, J))
    for b in range(N_BOOT):
        idx = _block_indices(n, rng)
        boot_t[b] = dc[idx].mean(0) / np.where(se > 0, se, np.inf)

    spa_p = float(np.mean(boot_t.max(axis=1) >= t.max()))

    order = np.argsort(-t)
    adj = np.ones(J)
    remaining = list(order)
    prev = 0.0
    while remaining:
        j = remaining[0]
        m = np.array(remaining)
        p = float(np.mean(boot_t[:, m].max(axis=1) >= t[j]))
        prev = max(prev, p)  # enforce monotonicity down the step-down sequence
        adj[j] = prev
        remaining = remaining[1:]
    return spa_p, pd.DataFrame(
        {"arm": names, "mean_loss_diff": mu, "dm_t": t, "rw_adj_p": adj}
    ).sort_values("dm_t", ascending=False)


def stage_conditioning() -> None:
    """Claim 1: the vol-regime conditioning gain, corrected for the 26-arm ladder search."""
    os.makedirs(OUT, exist_ok=True)
    p = load_panel()
    e = np.load(os.path.join(CACHE_DIR, "har_resid.npz"))["e"][TW:]
    sig = dict(np.load(os.path.join(CACHE_DIR, "bucket_signals.npz")))
    ts = pd.Series(pd.to_datetime(p.t[TW + TW :]))
    n = len(e)
    S = np.column_stack([sig[b] for b in BUCKETS])
    cw = COMBINE_WINDOW_DAYS * PERIODS_PER_DAY

    base = np.full(n, np.nan)
    base[cw:] = walk_forward(S, e, cw, alpha=1.0, refit_every=REFIT_EVERY)
    slot = ((ts.dt.hour * 2 + (ts.dt.minute >= 30)).to_numpy() - 19) % PERIODS_PER_DAY

    arms: dict[str, np.ndarray] = {}
    for label, bins_full, Ks in [
        ("tod", slot, [2, 4, 6, 8, 12, 16, 24, 48]),
        ("dow", ts.dt.dayofweek.to_numpy(), [5]),
        ("vol", None, [2, 3, 5, 10]),
    ]:
        for K in Ks:
            bins = _coarsen(bins_full, K) if bins_full is not None else _vol_regime(p, ts, K)
            pred = _bin_walk_forward(S, e, bins, K, COMBINE_WINDOW_DAYS)
            arms[f"{label}_K{K}_separate"] = pred
            arms[f"{label}_K{K}_blend"] = 0.5 * (pred + base)
            print(f"  built {label} K={K}", flush=True)

    m = np.isfinite(base)
    for v in arms.values():
        m &= np.isfinite(v)
    names = list(arms)
    d = np.column_stack(
        [(e[m] - base[m]) ** 2 - (e[m] - arms[a][m]) ** 2 for a in names]
    )
    print(f"\n  {len(names)} arms, {int(m.sum())} common rows, benchmark R2 {r2_oos(e[m], base[m]):+.5f}")
    spa_p, tab = spa_and_romano_wolf(d, names)
    tab["r2_gain"] = [
        r2_oos(e[m], arms[a][m]) - r2_oos(e[m], base[m]) for a in tab["arm"]
    ]
    print(f"\n  Hansen SPA p-value (H0: no arm beats pooled) = {spa_p:.4f}")
    print("\n  top arms with Romano-Wolf adjusted p-values:")
    print(tab.head(8).round(5).to_string(index=False))
    tab.insert(0, "spa_p_family", spa_p)
    tab.to_csv(f"{OUT}/multiplicity_conditioning.csv", index=False)
    print(f"\nwrote {OUT}/multiplicity_conditioning.csv")


def stage_interactions() -> None:
    """Claim 2: the interaction gain, corrected for the 8-arm penalty x selection search."""
    os.makedirs(OUT, exist_ok=True)
    p = load_panel()
    e = np.load(os.path.join(CACHE_DIR, "har_resid.npz"))["e"]
    bc, bn = base_columns(p)
    keep = np.array([i for i, nm in enumerate(bn) if "voldemand" not in nm])
    X = np.ascontiguousarray(p.X[TW:, bc[keep]], dtype=np.float64)
    n = len(e)
    ii, jj = _upper(X.shape[1])

    def floored(Ptr, Pte):
        sd = Ptr.std(0)
        sd = np.maximum(sd, 0.1 * np.median(sd[sd > 0]) if (sd > 0).any() else 1.0)
        return Ptr / sd, Pte / sd

    def fit(Xtr, ytr, Xte, n_base, a_prod):
        pen = np.concatenate(
            [np.full(n_base, RIDGE_ALPHA), np.full(Xtr.shape[1] - n_base, a_prod)]
        )
        b = np.linalg.solve(Xtr.T @ Xtr + np.diag(pen), Xtr.T @ (ytr - ytr.mean()))
        return Xte @ b + ytr.mean()

    grid = (3e2, 3e3, 3e4, 3e5)
    base = np.full(n - TW, np.nan)
    arms = {f"{tag}_alpha{int(a)}": np.full(n - TW, np.nan) for a in grid for tag in ("dyn", "stat")}
    ss: np.ndarray | None = None
    for t0 in range(TW, n, REFIT):
        tr = slice(t0 - TW, t0)
        t1 = min(t0 + REFIT, n)
        out = slice(t0 - TW, t1 - TW)
        mu = X[tr].mean(0)
        Ztr, Zte = X[tr] - mu, X[t0:t1] - mu
        ytr = e[tr]
        nb = Ztr.shape[1]
        base[out] = fit(Ztr, ytr, Zte, nb, RIDGE_ALPHA)
        sel = np.argsort(-np.abs(np.nan_to_num(_pair_ic(Ztr, ytr)[ii, jj])))[:100]
        if ss is None:
            ss = sel.copy()
        for tag, sl in (("dyn", sel), ("stat", ss)):
            Ptr, Pte = floored(
                _products(Ztr, ii[sl], jj[sl]), _products(Zte, ii[sl], jj[sl])
            )
            Atr, Ate = np.hstack([Ztr, Ptr]), np.hstack([Zte, Pte])
            for a in grid:
                arms[f"{tag}_alpha{int(a)}"][out] = fit(Atr, ytr, Ate, nb, a)

    y = e[TW:]
    names = list(arms)
    d = np.column_stack([(y - base) ** 2 - (y - arms[a]) ** 2 for a in names])
    print(f"\n  {len(names)} arms, benchmark (no products) R2 {r2_oos(y, base):+.5f}")
    spa_p, tab = spa_and_romano_wolf(d, names)
    tab["r2_gain"] = [r2_oos(y, arms[a]) - r2_oos(y, base) for a in tab["arm"]]
    print(f"\n  Hansen SPA p-value (H0: no arm beats the linear base) = {spa_p:.4f}")
    print("\n  arms with Romano-Wolf adjusted p-values:")
    print(tab.round(5).to_string(index=False))
    tab.insert(0, "spa_p_family", spa_p)
    tab.to_csv(f"{OUT}/multiplicity_interactions.csv", index=False)
    print(f"\nwrote {OUT}/multiplicity_interactions.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["conditioning", "interactions"], required=True)
    a = ap.parse_args()
    {"conditioning": stage_conditioning, "interactions": stage_interactions}[a.stage]()
