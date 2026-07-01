"""Model Confidence Set (Hansen, Lunde & Nason, 2011) -- the multi-model companion to Diebold-Mariano.

DM is pairwise; comparing many models pairwise inflates false positives. The MCS instead returns the
SET of models that are statistically indistinguishable from the best at confidence 1-alpha, by
iteratively testing equal predictive ability (EPA) and eliminating the worst model until EPA is not
rejected. A moving-block bootstrap handles the autocorrelation in per-bar losses.

Statistic: T_max = max_i t_i with t_i = dbar_{i.} / sqrt(boot-var), dbar_{i.} = model i's mean loss
minus the cross-model mean (so t_i > 0 = worse than average). Eliminate argmax_i t_i when the bootstrap
p-value < alpha. MCS p-value of an eliminated model = the running max of the test p-values (monotone);
survivors have MCS p-value >= alpha. Reference: Hansen, Lunde & Nason (2011, Econometrica).

numpy-only. losses: dict{name: per-bar loss array on the SAME OOS rows}. Lower loss = better.
"""

from __future__ import annotations

import numpy as np


def _boot_col_means(L: np.ndarray, B: int, block: int, seed: int) -> np.ndarray:
    """Moving-block bootstrap: return (B, m) where entry [b, i] = mean over a resampled time index of
    column i. The SAME resampled index is used across all models within a replicate (preserves the
    cross-model correlation). Indices shared across MCS rounds (computed once for the full model set)."""
    T, m = L.shape
    rng = np.random.RandomState(seed)
    nblk = int(np.ceil(T / block))
    starts = rng.randint(0, T, size=(B, nblk))
    offs = np.arange(block)
    out = np.empty((B, m), dtype=np.float64)
    for b in range(B):
        idx = ((starts[b][:, None] + offs).ravel() % T)[
            :T
        ]  # length-T circular block index
        out[b] = L[idx].mean(axis=0)
    return out


def model_confidence_set(
    losses: dict,
    alpha: float = 0.10,
    B: int = 2000,
    block: int | None = None,
    seed: int = 0,
) -> dict:
    """MCS at confidence 1-alpha. Returns dict:
    mcs        -- list of surviving model names (the confidence set),
    pvals      -- {name: MCS p-value}; in the set iff p-value >= alpha,
    eliminated -- names in elimination order (worst first),
    T, block, B.
    """
    names = list(losses)
    L = np.column_stack([np.asarray(losses[n], dtype=np.float64) for n in names])
    fin = np.all(np.isfinite(L), axis=1)
    L = L[fin]
    T, m = L.shape
    if block is None:
        block = max(
            1, int(round(T ** (1.0 / 3.0)))
        )  # ~ persistence-scaled block length
    col_mean = L.mean(axis=0)  # full-sample mean loss per model
    Lstar = _boot_col_means(
        L, B, block, seed
    )  # (B, m) bootstrap col means -- computed ONCE

    active = list(range(m))
    eliminated: list[str] = []
    pvals: dict[str, float] = {}
    running_p = 0.0
    while len(active) > 1:
        a = np.asarray(active, dtype=int)
        dbar = col_mean[a] - col_mean[a].mean()  # dbar_{i.} : loss vs set average
        # bootstrap variance & null dist of the centered statistic
        lbar_star = Lstar[:, a].mean(
            axis=1
        )  # (B,) set-average bootstrap mean per replicate
        dstar = Lstar[:, a] - lbar_star[:, None]  # (B, k)
        var = ((dstar - dbar) ** 2).mean(axis=0)  # (k,) bootstrap var of dbar_{i.}
        var = np.maximum(var, 1e-300)
        t = dbar / np.sqrt(var)  # (k,)
        t_star = (dstar - dbar) / np.sqrt(var)  # (B, k)
        tmax = float(t.max())
        tmax_star = t_star.max(axis=1)  # (B,)
        p = float(np.mean(tmax_star >= tmax))
        running_p = max(running_p, p)
        if p >= alpha:
            break
        worst_local = int(np.argmax(t))
        worst = active[worst_local]
        pvals[names[worst]] = running_p
        eliminated.append(names[worst])
        del active[worst_local]

    mcs = [names[i] for i in active]
    for i in active:
        pvals[names[i]] = max(running_p, alpha)
    return {
        "mcs": mcs,
        "pvals": pvals,
        "eliminated": eliminated,
        "T": T,
        "block": block,
        "B": B,
    }
