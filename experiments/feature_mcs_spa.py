"""SPA, Reality Check and StepM against HAR + calendar over the feature-MCS candidates (2026-10-12).

User (2026-10-12): Hansen's (2005) test for superior predictive ability and the Romano-Wolf (2005) StepM,
benchmark = HAR + calendar, on the existing 576 forecasts of the feature MCS (results/feature_mcs_2026-10-08/).

Families
  per estimator (enet, ridge, lasso, lgbm, xgb, rf)
            benchmark = that estimator's `baseline` (HAR + calendar); alternatives = its other 95 designs
            (all_features, 47 add-one, 47 leave-one-out)
  grand     benchmark = the paper's HAR + calendar OLS: the master table's `sub_ols_baseline`, whose
            adjusted-scale forecast is the 2026-10-03 export's `exog_penalty/ols_baseline`
            (results/close_studies_2026-10-03/forecasts/forecasts.parquet; its plain forecast equals the
            master table's daily plain for sub_ols_baseline on the 866 days, gated); alternatives = all 576
Losses: the eight of experiments/feature_mcs_report.py (QLIKE, MSE fit, ES(1), trade, log-MSE, hit, and the
winsorized QLIKE / trade).  The winsorization is the MCS's: a family's daily differentials against its
cross-candidate mean, clipped at the 1 % / 99 % points of the pooled differentials, computed on the MCS
scope's candidates (an estimator's 96; the grand 576); the grand benchmark (the OLS, not an MCS candidate) is
clipped with the grand scope's daily mean and clip points.
Tests, on d[t, k] = loss of the benchmark - loss of alternative k (positive = the alternative is better):
  SPA      Hansen (2005): T = max(0, max_k sqrt(n) dbar_k / omega_k), omega_k^2 = n x the bootstrap variance of
           dbar_k; p-values consistent / lower / upper (recentring g_k = dbar_k 1{dbar_k >= -sqrt(omega_k^2 / n
           2 log log n)} / max(dbar_k, 0) / dbar_k) -- experiments/close_kfull_testlib.spa_rc, read-only
  RC       White (2000) Reality Check, max_k sqrt(n) dbar_k against its recentred bootstrap (same function)
  StepM    Romano & Wolf (2005), studentized: reject H_k: E d_k <= 0 when t_k = sqrt(n) dbar_k / omega_k
           exceeds the (1 - alpha) quantile of max over the not-yet-rejected k of sqrt(n) (dbar*_k - dbar_k) /
           omega_k; repeat on the rest until nothing new is rejected; FWER alpha = 0.05 and 0.10
Bootstrap: the circular block bootstrap of src/evaluation/model_confidence_set.py's resample=True path (the
same RandomState(seed) block starts, blocks of `block` rows wrapping mod n, truncated to n rows), written out
here as resampling COUNTS (B x n) so the replicate means are one matrix product; B = 10,000, blocks 5 / 10 /
21, seed 20261008 (the MCS's).  Gates: the index of every replicate checked is not the identity (the repo's
own helpers atm_straddle_lib.circular_block_bootstrap_idx / the module's default return the sample); the
counts reproduce the module's resample=True replicate means.  Alternatives identical to the benchmark on a
loss (d = 0 every day, omega = 0) are dropped from that test and counted.

    python experiments/feature_mcs_spa.py      (after experiments/feature_mcs_report.py)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (
    REPO,
    REPO / "experiments",
    REPO / "notebooks",
    REPO / "specs",
    REPO / "writeup",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import close_linear_pretune as CLP  # noqa: E402
import feature_mcs_report as R  # noqa: E402
import feature_mcs_runner as FR  # noqa: E402
from close_kfull_testlib import spa_rc  # noqa: E402

from src.evaluation import model_confidence_set as MCSM  # noqa: E402

OUT = R.OUT
FWER = (0.05, 0.10)
OLS_SID = "exog_penalty/ols_baseline"
COL = {"trade": "trade_loss"}  # losses.parquet column of a loss


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cb_counts(n: int, B: int, block: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(B x n) resampling counts of the MCS module's circular block bootstrap (resample=True) and its indices."""
    rng = np.random.RandomState(seed)
    nblk = int(np.ceil(n / block))
    starts = rng.randint(0, n, size=(B, nblk))
    idx = ((starts[:, :, None] + np.arange(block)).reshape(B, -1) % n)[:, :n]
    flat = (idx + (np.arange(B) * n)[:, None]).ravel()
    W = np.bincount(flat, minlength=B * n).reshape(B, n).astype(np.float64)
    return W, idx


def stepm(
    d: np.ndarray, Dstar: np.ndarray, alpha: float
) -> tuple[list[int], list[float]]:
    """Romano-Wolf StepM (studentized): the rejected alternatives (in the order rejected) and the critical values."""
    n = d.shape[0]
    dbar = d.mean(axis=0)
    om = np.sqrt(n * ((Dstar - dbar) ** 2).mean(axis=0))
    t = np.sqrt(n) * dbar / om
    Z = np.sqrt(n) * (Dstar - dbar) / om
    active = np.ones(d.shape[1], bool)
    rej: list[int] = []
    crit: list[float] = []
    while active.any():
        a = np.flatnonzero(active)
        c = float(np.quantile(Z[:, a].max(axis=1), 1.0 - alpha))
        crit.append(c)
        new = a[t[a] > c]
        if not len(new):
            break
        new = new[np.argsort(-t[new])]
        rej += [int(x) for x in new]
        active[new] = False
    return rej, crit


def main() -> None:  # noqa: C901
    t0 = time.time()
    from close_murphy_splits_headline import elementary

    gates: list[dict] = []

    def gate(g: str, item: str, value: str, ok: bool) -> None:
        gates.append(dict(gate=g, item=item, value=value, ok=bool(ok)))

    Ld = pd.read_parquet(OUT / "losses.parquet")
    md = pd.read_csv(FR.INP / "models.csv")["model"].tolist()
    ests = [s for s in R.SHORTS if (Ld["est"] == s).any()]
    S = CLP.Scorer()
    days = pd.DatetimeIndex(S.days)
    n = len(days)
    # ---- the grand benchmark: the paper's HAR + calendar OLS
    inp = json.loads((FR.INP / "inputs.json").read_text())
    W_ = int(inp["W"])
    y_adj = np.load(FR.INP / "bar_y.npy")[W_:]
    day_row = np.load(FR.INP / "ref_armf.npz")["day_row"]
    ex = pd.read_parquet(
        CLP.EXPORT / "forecasts.parquet", columns=["series_id", "row", "pred_adj"]
    )
    ols = (
        ex[ex["series_id"] == OLS_SID]
        .sort_values("row")["pred_adj"]
        .to_numpy(np.float64)
    )
    do = S.daily(ols)
    rel = float(np.max(np.abs(do["plain"] / S.ols_plain - 1)))
    gate(
        "grand benchmark: exog_penalty/ols_baseline plain == master table daily plain of sub_ols_baseline",
        "866 days",
        f"max rel {rel:.1e}",
        rel < 1e-9,
    )
    ols_loss = {
        "qlike": do["ql"],
        "trade": -do["mid"],
        "mse_fit": (ols[day_row] - y_adj[day_row]) ** 2,
        "es1": elementary(do["plain"] / S.iv, S.target / S.iv, 1.0),
        "log_mse": (np.log(do["plain"]) - np.log(S.target)) ** 2,
        "hit": (do["q"] != np.where(S.target > S.iv, 1.0, -1.0)).astype(np.float64),
    }

    # ---- candidate loss matrices (days x candidates) in the MCS scope order
    def mat(est_list: list[str], lo: str) -> tuple[np.ndarray, list[str]]:
        cols, names = [], []
        for e in est_list:
            sub = Ld[Ld["est"] == e].pivot(
                index="day", columns="model", values=COL.get(lo, lo)
            )
            sub.index = pd.DatetimeIndex(sub.index)
            sub = sub.reindex(days)[md]
            cols.append(sub.to_numpy(np.float64))
            names += [f"{e}:{m}" for m in md]
        return np.hstack(cols), names

    # gate: losses.parquet reproduces the scorer for one candidate (elastic net HAR + calendar)
    fc = pd.read_parquet(
        OUT / "forecasts.parquet", columns=["est", "model", "row", "pred_adj"]
    )
    pb = (
        fc[(fc["est"] == ests[0]) & (fc["model"] == "baseline")]
        .sort_values("row")["pred_adj"]
        .to_numpy()
    )
    db = S.daily(pb)
    Lq, nq = mat([ests[0]], "qlike")
    gate(
        "losses.parquet == the scorer recomputed from forecasts.parquet",
        f"{ests[0]}:baseline QLIKE",
        f"max abs diff {np.abs(Lq[:, nq.index(f'{ests[0]}:baseline')] - db['ql']).max():.1e}",
        np.allclose(
            Lq[:, nq.index(f"{ests[0]}:baseline")], db["ql"], rtol=0, atol=1e-15
        ),
    )

    # ---- bootstrap counts per block, gated
    Wb = {}
    for blk in R.BLOCKS:
        Wc, idx = cb_counts(n, R.B_BOOT, blk, R.SEED)
        ident = int((idx == np.arange(n)).all(axis=1).sum())
        gate(
            "bootstrap: no replicate's index is the identity (the repo's switched-off helpers return the sample)",
            f"block {blk}, B = {R.B_BOOT}",
            f"{ident} identity replicates; first index starts {idx[0][:6].tolist()}",
            ident == 0,
        )
        Xg = Lq[:, :4]
        mod = MCSM._boot_col_means(Xg, R.B_BOOT, blk, R.SEED, resample=True)
        mine = Wc @ Xg / n
        gate(
            "bootstrap: the counts reproduce the MCS module's resample=True replicate means",
            f"block {blk}",
            f"max rel {np.max(np.abs(mine / mod - 1)):.1e}",
            np.allclose(mine, mod, rtol=1e-12, atol=0),
        )
        gate(
            "bootstrap: replicate means vary (sd > 0 in every column checked)",
            f"block {blk}",
            f"min sd {mine.std(axis=0).min():.2e}",
            bool((mine.std(axis=0) > 0).all()),
        )
        Wb[blk] = Wc

    # ---- the families
    fam = []
    for e in ests:
        fam.append((e, [e], f"{e}:baseline"))
    if len(ests) == len(R.SHORTS):
        fam.append(("grand", list(ests), "ols:sub_ols_baseline"))
    spa_rows, stp_rows, t_rows = [], [], []
    for sc, est_list, bench in fam:
        for lo in R.ALL_LOSSES:
            base_lo = R.WINSOR.get(lo, lo)
            L, names = mat(est_list, base_lo)
            if sc == "grand":
                Lb = ols_loss[base_lo]
            if lo in R.WINSOR:  # the MCS scope's winsorization
                mu = L.mean(axis=1, keepdims=True)
                D = L - mu
                q1, q9 = np.quantile(D, R.WIN_Q)
                Lw = mu + np.clip(D, q1, q9)
                assert np.array_equal(Lw, R.winsorize(L)[0])
                L = Lw
                if sc == "grand":
                    Lb = mu[:, 0] + np.clip(Lb - mu[:, 0], q1, q9)
            if sc != "grand":
                jb = names.index(bench)
                Lb = L[:, jb]
                keep = [j for j in range(len(names)) if j != jb]
                L, names = L[:, keep], [names[j] for j in keep]
            d = Lb[:, None] - L
            zero = np.all(d == 0, axis=0)
            d, names_k = d[:, ~zero], [nm for nm, z in zip(names, zero) if not z]
            for blk in R.BLOCKS:
                Ds = Wb[blk] @ d / n
                r = spa_rc(d, Ds)
                tt = np.sqrt(n) * d.mean(axis=0) / r["omega"]
                spa_rows.append(
                    dict(
                        scope=sc,
                        loss=lo,
                        block=blk,
                        benchmark=bench,
                        n_alt=d.shape[1],
                        n_identical_dropped=int(zero.sum()),
                        stat_spa=r["stat_spa"],
                        p_consistent=r["p_consistent"],
                        p_lower=r["p_lower"],
                        p_upper=r["p_upper"],
                        best=names_k[r["best"]],
                        best_t=float(tt[r["best"]]),
                        stat_rc=r["stat_rc"],
                        p_rc=r["p_rc"],
                        n_beat_point=int((d.mean(axis=0) > 0).sum()),
                    )
                )
                for al in FWER:
                    rej, crit = stepm(d, Ds, al)
                    stp_rows.append(
                        dict(
                            scope=sc,
                            loss=lo,
                            block=blk,
                            fwer=al,
                            n_rejected=len(rej),
                            rejected=json.dumps([names_k[j] for j in rej]),
                            crit_first=crit[0] if crit else np.nan,
                            steps=len(crit),
                        )
                    )
                for k, nm in enumerate(names_k):
                    t_rows.append(
                        dict(
                            scope=sc,
                            loss=lo,
                            block=blk,
                            alternative=nm,
                            dbar=float(d[:, k].mean()),
                            t=float(tt[k]),
                        )
                    )
            say(
                f"{sc} {lo}: SPA p (consistent, blocks {R.BLOCKS}) "
                + " / ".join(f"{x['p_consistent']:.3f}" for x in spa_rows[-3:])
                + f"; StepM 5 % rejections {[x['n_rejected'] for x in stp_rows[-6:] if x['fwer'] == 0.05]}"
            )
    pd.DataFrame(spa_rows).to_csv(OUT / "spa_rc.csv", index=False)
    pd.DataFrame(stp_rows).to_csv(OUT / "stepm.csv", index=False)
    pd.DataFrame(t_rows).to_csv(OUT / "spa_tstats.csv", index=False)
    G = pd.DataFrame(gates)
    G.to_csv(OUT / "spa_gates.csv", index=False)
    say(f"done in {time.time() - t0:.0f}s; gates {int(G['ok'].sum())} / {len(G)} ok")


if __name__ == "__main__":
    main()
