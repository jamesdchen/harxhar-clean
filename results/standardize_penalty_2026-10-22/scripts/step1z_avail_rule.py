"""STEP 1-prime (scaled coordinates) of the data-relative availability rule (owner, 2026-10-22): test before implementing.  Prototype only; the
walk code (experiments/pool_linear_pretune.py at 0297e94) is used read-only.

The rule, for availability averages <v>_avail_ma_<k>, k > 1, live after the existing masks (identifiability; the
evidence rule on the k = 1 column), under STANDARDIZE_PENALTY: p = the column's window mean, se = sqrt(p (1 - p) / k);
(i) near-constant: masked if its window s.d. <= se; (ii) near-duplicate: in design order, masked if against an earlier
KEPT availability average of the same k the window s.d. of the difference <= sqrt(2) se_pair, se_pair =
sqrt(pbar (1 - pbar) / k), pbar = (p_i + p_j) / 2.  k = 1 flags, value averages, HAR and calendar exempt.

At each origin o (window = rows o - W .. o - 1 of the flag-on w1000 design, unobs d_on): the KKT-certified batch fits of
the committed standardized machinery (fit_std: weights, existing masks; the lasso / enet batch residualizes on the
intercept, i.e. centred) for the lasso (1e-3) and elastic net (1e-2, l1 0.5), without and with the rule; the flag-off
lasso / enet (batch_theta_pf, penalty factors only) for reference.  Active-set condition numbers as diag_cond_names.py
(the matrix the warm path solves: active columns + intercept, + the L2 term; centred = the walk's state since 58dca0c,
uncentred also given).

usage: python step1_avail_rule.py <out json> <origin index list "i,j,..." or "all">
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path(r"C:\Users\james\AppData\Local\Temp\std_wt")
DZ = Path(
    r"C:\Users\james\AppData\Local\Temp\claude\C--Users-james-CC-Allowed-harxhar-0dte-professor"
    r"\bd4224e0-12aa-4614-b6a6-3fe50535ade1\scratchpad\unobs\d_on\_work\design_all_features"
)
sys.path.insert(0, str(WT / "experiments"))
sys.path.insert(0, str(WT))
import pool_linear_pretune as P  # noqa: E402

P.SAMPLE, P.TW_DAYS, P.PERIODS_PER_DAY, P.W, P.HAR_BASE = "rth13", 1000, 13, 13000, "5"
P.EX_FLAGS = dict(
    DROP_ACTIVITY_MA=True,
    CALENDAR_THROUGH_OVERNIGHT=True,
    DROP_OVERNIGHT_INDICATOR=True,
    LEAN_GLOBAL_LAGS=True,
    LATE_START_WARMUP=True,
)
ns = P.spec_section("reclasso", "all_features", 0, -1, 0)
P.STANDARDIZE_PENALTY = True
meta = json.loads((DZ / "design.json").read_text())
names = meta["names"]
P.check_meta(meta, "all_features")
bb = P.backbone_mask(names)
X = np.load(DZ / "X.npy", mmap_mode="r")
y = np.load(DZ / "y.npy")
W = 13000
p = X.shape[1]
locked = np.zeros(p + 1, dtype=bool)
locked[-1] = True
pf = np.ones(p + 1)
pf[:-1][bb] = 0.01
er = P.evid_ref(names, bb)
nm = names + ["intercept"]
AV = [j for j, c in enumerate(names) if "_avail_ma_" in c and int(c.rsplit("_", 1)[1]) > 1]
KOF = {j: int(names[j].rsplit("_", 1)[1]) for j in AV}
VAL = [j for j, c in enumerate(names) if c.startswith("adj_")]
EST = dict(lasso=(1e-3, 1.0), enet=(1e-2, 0.5))

# origins: the first row of every January and July 2012-01 .. 2024-01, and the 2016 / 2020 diagnostic origins
d = pd.read_csv(DZ / "dates_oos.csv", parse_dates=["date"])["date"]
orig = []
for yr_ in range(2012, 2025):
    for mo in (1, 7):
        if (yr_, mo) > (2024, 4):
            continue
        i = int(np.flatnonzero((d.dt.year == yr_) & (d.dt.month == mo))[0])
        orig.append((f"{yr_}-{mo:02d}", W + i))
orig += [("diag_2016", 44250 + 250 * 60), ("diag_2020", 44250 + 250 * 120)]
sel = sys.argv[2]
todo = orig if sel == "all" else [orig[int(i)] for i in sel.split(",")]


def new_rule(Xa: np.ndarray, out0: np.ndarray) -> tuple[np.ndarray, list]:
    """The rule on the live availability averages (k > 1): (mask, records)."""
    n = Xa.shape[0]
    mask = np.zeros(p + 1, dtype=bool)
    recs = []
    kept: dict = {}
    for j in AV:  # design order
        if out0[j]:
            continue
        k = KOF[j]
        x = Xa[:, j]
        pj = float(x.mean())
        sd = float(x.std())
        se = float(np.sqrt(max(pj * (1.0 - pj), 0.0) / k))
        if sd <= se:
            mask[j] = True
            recs.append(dict(col=nm[j], kind="near-constant", sd=sd, se=se, p=pj))
            continue
        dup = None
        for i in kept.get(k, []):
            pi = float(Xa[:, i].mean())
            pbar = 0.5 * (pi + pj)
            sep = float(np.sqrt(max(pbar * (1.0 - pbar), 0.0) / k))
            sdd = float((x - Xa[:, i]).std())
            if sdd <= np.sqrt(2.0) * sep:
                dup = dict(col=nm[j], kind="near-duplicate", partner=nm[i], sd=sd, se=se, sd_diff=sdd,
                           threshold=float(np.sqrt(2.0) * sep), p=pj, p_partner=pi)
                break
        if dup is not None:
            mask[j] = True
            recs.append(dup)
            continue
        kept.setdefault(k, []).append(j)
    assert n == W
    return mask, recs


def cond_active(Xu: np.ndarray, th: np.ndarray, a: float, l1: float, w2: np.ndarray) -> dict:
    A = np.flatnonzero((th != 0) | locked)
    ei = np.flatnonzero(~locked[A])
    lam2 = W * a * (1.0 - l1)
    r = {}
    for cen in (False, True):
        XA = Xu[:, A] - (np.where(locked[A], 0.0, Xu[:, A].mean(axis=0)) if cen else 0.0)
        M = XA.T @ XA
        M[ei, ei] += lam2 * w2[A][ei]
        r["centred" if cen else "uncentred"] = float(np.linalg.cond(M))
    return r


def fit_z(Xu: np.ndarray, yr: np.ndarray, out: np.ndarray, kind: str, a: float, l1: float):
    """Textbook standardization: z_j = (x_j - m_j) / s_j (window mean and s.d. of the fit's rows; the masked columns 0),
    the intercept column 1, the uniform penalty (pf only, no s weights), KKT-certified batch (batch_theta_pf);
    back-transform beta_j = gamma_j / s_j, beta_0 = gamma_0 - m' beta.  Returns (beta, gamma, Z)."""
    m = np.where(locked | out, 0.0, Xu.mean(axis=0))
    s = np.where(locked | out, 1.0, Xu.std(axis=0))
    Z = (Xu - m) / s
    Z[:, out] = 0.0
    g = P.batch_theta_pf(ns, Z, yr, locked, a, l1, pf)
    beta = g / s
    beta[-1] = g[-1] - m @ beta
    return beta, g, Z


def cond_z(Z: np.ndarray, g: np.ndarray, a: float, l1: float) -> float:
    """The active-set matrix the scaled solver solves: active z columns + the intercept, + the L2 term (pf)."""
    A = np.flatnonzero((g != 0) | locked)
    ei = np.flatnonzero(~locked[A])
    M = Z[:, A].T @ Z[:, A]
    M[ei, ei] += W * a * (1.0 - l1) * pf[A][ei]
    return float(np.linalg.cond(M))


res = []
for lab, o in todo:
    t0 = time.time()
    Xa = np.hstack([np.asarray(X[o - W : o]), np.ones((W, 1))])
    yr = y[o - W : o]
    probe = ns["RollingTunedLinear"]()
    probe._locked = locked
    probe._recompute_mask(Xa)
    Xa[:, probe._maskout] = 0.0
    s0, out0 = P.std_weights(Xa, locked, er)  # the existing masks (identifiability + evidence)
    mask, recs = new_rule(Xa, out0)
    Xb = Xa.copy()
    Xb[:, mask] = 0.0
    s1, out1 = P.std_weights(Xb, locked, er)
    assert not (mask & ~out1).any()
    xn = np.append(np.asarray(X[o]), 1.0)
    r: dict = dict(origin=lab, row=o, date=str(d[o - W].date()), masked_existing=int(out0[:-1].sum()),
                   new_rule_masked=recs, n_new_rule=len(recs),
                   value_columns_masked_by_rule=int(mask[VAL].sum()),
                   avail_k_gt1_live_without=int(sum(not out0[j] for j in AV)),
                   avail_k_gt1_live_with=int(sum(not out1[j] for j in AV)))
    for m, (a, l1) in EST.items():
        th0 = P.fit_std(Xa, yr, locked, m, a, l1, pf, er)[0]
        th1 = P.fit_std(Xb, yr, locked, m, a, l1, pf, er)[0]
        bz0, gz0, Z0 = fit_z(Xa, yr, out0, m, a, l1)
        bz1, gz1, Z1 = fit_z(Xb, yr, out1, m, a, l1)

        def eqv(u, v):
            return dict(coef_normwise=float(np.abs(u - v).max() / np.abs(v).max()),
                        forecast_rel=float(abs(xn @ u - xn @ v) / abs(xn @ v)),
                        support_equal=bool(np.array_equal(u[:-1] != 0, v[:-1] != 0)))
        thf = P.batch_theta_pf(ns, Xa, yr, locked, a, l1, pf)  # flag off (identifiability mask only)
        act0 = (th0 != 0) & ~locked
        long_low = [nm[j] for j in AV if act0[j] and float(Xa[:, j].std()) <= np.sqrt(
            max(Xa[:, j].mean() * (1 - Xa[:, j].mean()), 0.0) / KOF[j])]
        f0, f1 = float(xn @ th0), float(xn @ th1)
        r[m] = dict(
            n_active_without=int(act0.sum()),
            n_active_with=int(((th1 != 0) & ~locked).sum()),
            n_active_flag_off=int(((thf != 0) & ~locked).sum()),
            cond_without=cond_active(Xa, th0, a, l1, pf * s0 * s0),
            cond_with=cond_active(Xb, th1, a, l1, pf * s1 * s1),
            cond_flag_off=cond_active(Xa, thf, a, l1, pf),
            cond_z_without=cond_z(Z0, gz0, a, l1),
            cond_z_with=cond_z(Z1, gz1, a, l1),
            n_active_z_with=int(((gz1 != 0) & ~locked).sum()),
            z_vs_weighted_without=eqv(bz0, th0),
            z_vs_weighted_with=eqv(bz1, th1),
            forecast_without=f0,
            forecast_with=f1,
            forecast_rel_change=abs(f1 - f0) / abs(f0),
            active_avail_k_gt1_sd_le_se_without=long_low,
            P1_all_removed=all((th1[nm.index(c)] == 0) for c in long_low),
            vix3m_3125_active_without=bool(th0[nm.index("vix3m_avail_ma_3125")] != 0),
            vvix_3125_active_without=bool(th0[nm.index("vvix_avail_ma_3125")] != 0),
            vix3m_3125_active_with=bool(th1[nm.index("vix3m_avail_ma_3125")] != 0),
            vvix_3125_active_with=bool(th1[nm.index("vvix_avail_ma_3125")] != 0),
        )
        r[m]["P2_cond_with_le_flag_off_centred"] = r[m]["cond_with"]["centred"] <= r[m]["cond_flag_off"]["centred"]
        r[m]["P2z_vs_flag_off_centred"] = r[m]["cond_z_with"] <= r[m]["cond_flag_off"]["centred"]
        r[m]["P2z_vs_flag_off_uncentred"] = r[m]["cond_z_with"] <= r[m]["cond_flag_off"]["uncentred"]
        r[m]["P2_cond_with_le_flag_off_uncentred"] = (
            r[m]["cond_with"]["uncentred"] <= r[m]["cond_flag_off"]["uncentred"]
        )
    r["vix3m_vvix_3125_rule"] = [x for x in recs if x["col"] in ("vix3m_avail_ma_3125", "vvix_avail_ma_3125")
                                 or x.get("partner") in ("vix3m_avail_ma_3125", "vvix_avail_ma_3125")]
    r["sec"] = time.time() - t0
    res.append(r)
    print(lab, o, len(recs), {m: (round(r[m]["cond_flag_off"]["centred"]), round(r[m]["cond_z_without"]),
                                  round(r[m]["cond_z_with"])) for m in EST}, round(r["sec"]), flush=True)
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
