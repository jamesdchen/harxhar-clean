"""The linear factor tests with PENALIZED estimators (results/beyond_trees_2026-10-27/RULE_linear_pen.txt, committed
a38439f before any computation; requested by the author after the OLS factor tests; not pre-registered).

Author (28 Oct): "they're cheap, and they make the results apples to apples. run it."  Follow-up: "how are we sure that
the pca isn't handicapping the factor model in comparison to the other ablations?" (controls C1-C3).

TESTS     A  base = the HAR set (12), block = the value averages of the 43 non-FOMC series (258; tab:rh_factor's block)
          B  base = add_rm_fomc (132: HAR + realized moments + FOMC), block = the 36 series of ew, vw, liq, iv, sent, vd
             (216; the trees' factor test's block)
COMPONENTS exactly the existing tests' (A: factor_har_2026_10_26.day_map on the rolled window sums; B:
          beyond_factor_linear.compute_map on the materialized window), once a trading day, here with every active
          eigenvector kept (C1); stored by the maps mode (scratchpad memmaps), read by the walk.
CLASSES   ridge (1e3), lasso (1e-4), elastic net (1e-3, l1 0.5): the add-arm convention (the full set's frozen penalty,
          r = 0.01 on the HAR block, 1 elsewhere, the scores included), pool_linear_pretune.py's estimator semantics
          (STANDARDIZE_PENALTY, the identifiability / evidence masks, ridge per-bar weights, lasso / elastic-net weights
          held between cold seeds and the runner's data-driven reseeds), solved exactly at every bar on the window Gram
          (ridge: the linear system; lasso / elastic net: an active-set solve warm-started from the previous bar's
          support, certified by the runner's KKT conditions, the runner's batch solve as the fallback).  OLS (min-norm,
          Table 1's walk) alongside as the reference class.  Test A also walks k = 0 with the HAR arm's own settings
          (<cls>_h: the baseline penalty, r = 1) for gate G1.
ARMS      k0, pca{1,2,3,5,10,20}, pls{1,2,3}, c1 (all active components), c2 (the block's value averages themselves);
          C3 (pca3_bar, pca10_bar: loadings at every bar) on the C3 days; gate helpers c2act (OLS every bar; ridge at the
          days' first bars, with c1iso).

usage (repository root, the 285J environment, one thread):
  python experiments/factor_pen_2026_10_28.py maps  --test A|B --design D --work W
  python experiments/factor_pen_2026_10_28.py walk  --test A|B --design D --work W --classes ridge,lasso,enet[,ols][,h]
                                                    [--max-bars N] [--kkt F.json]
  python experiments/factor_pen_2026_10_28.py gates --test A|B --design D --work W
  python experiments/factor_pen_2026_10_28.py score --test A|B
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.linalg as sla  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT), str(ROOT / "notebooks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ablate_arms as AA  # noqa: E402
import beyond_factor_linear as BFL  # noqa: E402
import factor_har_2026_10_26 as FH  # noqa: E402
import pool_linear_pretune as PLP  # noqa: E402

# the 13-bar rerun's configuration (run_info.json runner_cfg: --sample rth13 --har-base 5 ...): backbone_mask's branch
# for it (har_ma_* and the calendar columns the flags leave)
PLP.SAMPLE, PLP.HAR_BASE = "rth13", "5"

LABEL = "requested by the author after the OLS factor tests; not pre-registered"
RULE = "results/beyond_trees_2026-10-27/RULE_linear_pen.txt (a38439f)"
W = FH.W
TUNE_PER = FH.TUNE_PER
NPC, NPLS = 20, 3
K_PCA = (1, 2, 3, 5, 10, 20)
K_PLS = (1, 2, 3)
ARMS = ["k0"] + [f"pca{k}" for k in K_PCA] + [f"pls{k}" for k in K_PLS] + ["c1", "c2"]
C3_ARMS = ["pca3_bar", "pca10_bar"]
PEN = {"ridge": (1e3, 0.0), "lasso": (1e-4, 1.0), "enet": (1e-3, 0.5)}  # all_features frozen penalties
PEN_H = {"ridge": (1e-3, 0.0), "lasso": (1e-8, 1.0), "enet": (1e-7, 0.5)}  # baseline (HAR arm) frozen penalties
R_HAR = 0.01
KKT_TOL_REL = 1e-9
G1_TOL, G2_TOL, GC1_TOL = 1e-10, 1e-8, 1e-8
G2_DAYS = ["2014-01-02", "2016-01-27", "2018-02-16", "2020-03-16", "2020-03-17", "2022-04-06", "2024-04-30"]
N_SCORED, N_DAYS, N_EVENT = 33_527, 2_579, 81
HAC_LAG = 8
SPANS = ("ordinary", "all", "event")
TESTS: dict[str, dict[str, Any]] = {
    "A": dict(
        out=ROOT / "results" / "channels_2026-10-26" / "factor_pen",
        ols_ref=ROOT / "results" / "channels_2026-10-26" / "factor" / "forecasts.parquet",
        title="Penalized linear factor test, base = the HAR set (block: 43 non-FOMC series)",
    ),
    "B": dict(
        out=ROOT / "results" / "beyond_trees_2026-10-27" / "linear_pen",
        ols_ref=ROOT / "results" / "beyond_trees_2026-10-27" / "linear" / "forecasts.parquet",
        title="Penalized linear factor test beyond HAR + realized moments + FOMC (block: 36 series)",
    ),
}
BASE_B_FC = ROOT / "results" / "ablations_2014start_w3000" / "forecasts" / "linear_wide_rmfomc.parquet"
MAPS_B_REF = "210da341ee6c73b2a3b71e361ab4a188"


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ============================================================================ design
def load(test: str, design: Path) -> dict:
    """[base | block] columns of the D13 design parts (the OLS tests' rebuild), the target and the dates."""
    P = FH.Parts(design)
    order = P.order()
    AA.check_full_names(order)
    gs = AA.group_series()
    if test == "A":
        base = list(AA.HAR12)
        Z = FH.block_names(order, gs)
        assert len(Z) == 258, len(Z)
    else:
        base = AA.arm_names("add_rm_fomc", order)
        Z = BFL.block_names(order)
        assert len(base) == 132 and len(Z) == 216, (len(base), len(Z))
    assert not set(base) & set(Z)
    X = P.cols(base + Z)  # C order (one row a bar)
    d = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(P.parts[0] / "dates_oos.csv", dtype=str)["date"]))
    cfg = P.metas[0]["config"]
    assert int(cfg["window"]) == W and cfg["periods_per_day"] == 13, cfg
    assert X.shape[0] == W + len(d) == len(P.y)
    zser = [re.match(r"^adj_(.+)_ma_\d+$", z).group(1) for z in Z]  # type: ignore[union-attr]
    ser_list = list(dict.fromkeys(zser))
    k1 = np.array([Z.index(f"adj_{s}_ma_1") for s in ser_list])
    col_ser = np.array([ser_list.index(s) for s in zser])
    sg = FH.series_group(gs)
    pos = {nm: i for i, nm in enumerate(order)}
    return dict(
        X=X,
        y=np.ascontiguousarray(P.y, dtype=np.float64),
        dates=d,
        base=base,
        Z=Z,
        names=base + Z,
        nb=len(base),
        p=len(Z),
        k1=k1,
        col_ser=col_ser,
        ser_list=ser_list,
        zgrp=np.array([sg[s] for s in zser]),
        cache_pos=np.array([pos[nm] for nm in base + Z]),
        design=str(design),
    )


def walk_rows(D: dict) -> tuple[int, int]:
    """(Table 1's walk start S0, the end row): solves S0 .. n_rows - 1 (33,997)."""
    return FH.walk_start(D["dates"]), D["X"].shape[0]


# ============================================================================ maps
def map_full_sums(S, c, sx, sy, ev_ok, nb: int, p: int) -> dict:
    """factor_har_2026_10_26.day_map's arithmetic (test A's route: the rolled window sums), every active eigenvector
    kept: P = [V / sd | R / sd] on the active rows, V all n_act eigenvectors (descending; signed to a non-negative
    sum, as day_map signs the first 20), R the 3 SIMPLS weight vectors."""
    n = float(W)
    mZ = sx[nb:] / n
    G = S[nb:, nb:] - n * np.outer(mZ, mZ)
    sd = np.sqrt(np.maximum(np.diag(G) / n, 0.0))
    act = sd > FH.STD_MIN
    if ev_ok is not None:
        act &= ev_ok
    a = np.flatnonzero(act)
    sa = sd[a]
    R = G[np.ix_(a, a)] / (n * np.outer(sa, sa))
    R = 0.5 * (R + R.T)
    lam, V = np.linalg.eigh(R)
    lam, V = lam[::-1], V[:, ::-1]
    V = V * np.where(V.sum(axis=0) < 0, -1.0, 1.0)
    gy = c[nb:][a] - n * mZ[a] * (sy / n)
    Rp = FH.simpls(R, gy / (n * sa), NPLS)
    return dict(a=a, sd=sd, lam=lam, Pv=V / sa[:, None], Pr=Rp / sa[:, None], trace=float(lam.sum()))


def map_full_window(Zw: np.ndarray, yw: np.ndarray, k1: np.ndarray, col_ser: np.ndarray) -> dict:
    """beyond_factor_linear.compute_map's arithmetic (test B's route: the materialized window), every active
    eigenvector kept."""
    m = Zw.mean(axis=0)
    sd = Zw.std(axis=0)
    off = W - BFL.mode_count(Zw[:, k1])
    act = (sd > FH.STD_MIN) & (off[col_ser] >= FH.STD_MIN_EVIDENCE)
    a = np.flatnonzero(act)
    Zs = (Zw[:, a] - m[a]) / sd[a]
    C = Zs.T @ Zs / W
    C = 0.5 * (C + C.T)
    lam, V = np.linalg.eigh(C)
    lam, V = lam[::-1], V[:, ::-1]
    V = V * np.where(V.sum(axis=0) < 0, -1.0, 1.0)
    s = Zs.T @ (yw - yw.mean()) / W
    Rp = FH.simpls(C, s, NPLS)
    return dict(a=a, sd=sd, lam=lam, Pv=V / sd[a][:, None], Pr=Rp / sd[a][:, None], trace=float(lam.sum()))


class SumsA:
    """Test A's map route: factor_har's chunk (rows from S0 - W) with its Evidence tracker on the k = 1 columns."""

    def __init__(self, D: dict, S0: int):
        self.lo = S0 - W
        nb = D["nb"]
        self.ev = FH.Evidence(D["X"][self.lo :, nb + D["k1"]])
        self.zk1 = D["col_ser"]

    def ev_ok(self, t: int) -> np.ndarray:
        return self.ev.off_mode(t - self.lo)[self.zk1] >= FH.STD_MIN_EVIDENCE


def cmd_maps(a) -> int:
    t_wall = time.time()
    D = load(a.test, Path(a.design).resolve())
    X, y, dates = D["X"], D["y"], D["dates"]
    nb, p = D["nb"], D["p"]
    S0, n_rows = walk_rows(D)
    fdates = dates[S0 - W :]
    fday = fdates.normalize()
    first = [i for i in range(len(fdates)) if i == 0 or fday[i] != fday[i - 1]]
    nd = len(first)
    wk = Path(a.work) / f"maps_{a.test}"
    wk.mkdir(parents=True, exist_ok=True)
    mm = {
        "Pv": np.lib.format.open_memmap(wk / "Pv.npy", mode="w+", dtype=np.float64, shape=(nd, p, p)),
        "Pr": np.lib.format.open_memmap(wk / "Pr.npy", mode="w+", dtype=np.float64, shape=(nd, p, NPLS)),
    }
    act = np.zeros((nd, p), dtype=bool)
    nact = np.zeros(nd, dtype=np.int64)
    lam = np.zeros((nd, p))
    sd = np.zeros((nd, p))
    trace = np.zeros(nd)
    sa = SumsA(D, S0) if a.test == "A" else None
    S = c = sx = None
    sy = 0.0
    j = 0
    for i in range(len(fdates)):
        t = S0 + i
        if a.test == "A":  # factor_har's rolled sums over [HAR12, Z]
            if i % TUNE_PER == 0:
                Aw, yw = X[t - W : t], y[t - W : t]
                S, c, sx, sy = Aw.T @ Aw, Aw.T @ yw, Aw.sum(axis=0), float(yw.sum())
            else:
                u, v = X[t - 1], X[t - 1 - W]
                S += np.outer(u, u) - np.outer(v, v)
                c += u * float(y[t - 1]) - v * float(y[t - 1 - W])
                sx += u - v
                sy += float(y[t - 1]) - float(y[t - 1 - W])
        if j < nd and i == first[j]:
            if a.test == "A":
                mp = map_full_sums(S, c, sx, sy, sa.ev_ok(t), nb, p)  # type: ignore[union-attr]
            else:
                mp = map_full_window(np.array(X[t - W : t, nb:]), y[t - W : t], D["k1"], D["col_ser"])
            ia = mp["a"]
            na = len(ia)
            Pv = np.zeros((p, p))
            Pv[ia, :na] = mp["Pv"]
            Pr = np.zeros((p, NPLS))
            Pr[ia] = mp["Pr"]
            mm["Pv"][j] = Pv
            mm["Pr"][j] = Pr
            act[j, ia] = True
            nact[j] = na
            lam[j, :na] = mp["lam"]
            sd[j] = mp["sd"]
            trace[j] = mp["trace"]
            j += 1
            if j % 250 == 0:
                say(f"  map {j} / {nd} ({fday[i].date()}); {time.time() - t_wall:.0f} s")
    for v in mm.values():
        v.flush()
    np.savez(
        wk / "meta.npz",
        first_row=np.array([S0 + i for i in first]),
        days=np.array([str(fday[i].date()) for i in first]),
        act=act,
        n_active=nact,
        lam=lam,
        sd=sd,
        trace=trace,
    )
    info = dict(test=a.test, days=nd, p=p, wall_sec=round(time.time() - t_wall, 1), host=platform.node())
    (wk / "maps.json").write_text(json.dumps(info, indent=1))
    say(f"maps done: {info}")
    return 0


class Maps:
    def __init__(self, work: Path, test: str):
        wk = work / f"maps_{test}"
        self.Pv = np.load(wk / "Pv.npy", mmap_mode="r")
        self.Pr = np.load(wk / "Pr.npy", mmap_mode="r")
        m = np.load(wk / "meta.npz")
        self.first_row = m["first_row"]
        self.days = m["days"]
        self.act = m["act"]
        self.n_active = m["n_active"]
        self.lam = m["lam"]
        self.sd = m["sd"]
        self.idx = {int(r): j for j, r in enumerate(self.first_row)}

    def P_all(self, j: int) -> tuple[np.ndarray, int]:
        """[Pv (n_act columns) | Pr (3)] of day j, and n_act."""
        na = int(self.n_active[j])
        return np.ascontiguousarray(np.hstack([self.Pv[j][:, :na], self.Pr[j]])), na


# ============================================================================ the estimator's masks and seeds
class Tracker:
    """pool_linear_pretune's mask and seed state for one class on one column set (the arm's non-score columns, in the
    full cache's order): the identifiability mask at every tune, the evidence tracker, the lasso's between-tune
    mask-addition trackers, the held weights of the lasso / elastic net (np.std of the masked seed window).  The
    binomial availability rule is asserted moot (every availability average with k > 1 is masked at every seed)."""

    def __init__(self, kind: str, gidx: np.ndarray, names: list[str], X: np.ndarray, kkt: set[int]):
        self.kind = kind
        self.g = np.asarray(gidx, dtype=np.int64)
        self.names = names
        self.X = X
        self.p = len(gidx)
        bb = PLP.backbone_mask(names)
        self.bb = bb
        self.ref = PLP.evid_ref(names, bb)
        self.av = PLP.avail_k(names, bb)
        self.locked = np.zeros(self.p + 1, dtype=bool)
        self.locked[-1] = True
        self.maskout = np.zeros(self.p + 1, dtype=bool)
        self.evid_only = np.zeros(self.p + 1, dtype=bool)
        self.track = kind == "lasso"
        self.kkt = kkt
        self.seed_t = -1
        self.t0 = -1  # the walk start (set by the walk)
        self.seeds: list[tuple[int, str]] = []
        self.w = np.ones(self.p + 1)

    def raw(self, t0: int, t1: int) -> np.ndarray:
        A = np.empty((t1 - t0, self.p + 1))
        A[:, :-1] = self.X[t0:t1][:, self.g]
        A[:, -1] = 1.0
        return A

    def row(self, t: int) -> np.ndarray:
        return np.append(self.X[t, self.g], 1.0)

    def ident_mask(self, Xraw: np.ndarray) -> np.ndarray:
        """The spec's _recompute_mask: non-locked columns constant in the window or byte-duplicates of an earlier
        kept column."""
        mo = np.zeros(self.p + 1, dtype=bool)
        seen: dict = {}
        for j in range(self.p + 1):
            if self.locked[j]:
                continue
            col = Xraw[:, j]
            if col.max() == col.min():
                mo[j] = True
                continue
            key = col.tobytes()
            if key in seen:
                mo[j] = True
            else:
                seen[key] = j
        return mo

    def init(self, t: int) -> None:
        Xraw = self.raw(t - W, t)
        if self.track:  # the spec's _init_runs
            n = Xraw.shape[0]

            def trailing(ok):
                bad = ~ok[::-1]
                return np.where(bad.any(axis=0), bad.argmax(axis=0), n)

            self.run_const = trailing(Xraw == Xraw[-1]).astype(np.int64)
            self.run_eq = np.empty((self.p + 1, self.p + 1), dtype=np.int64)
            for j in range(self.p + 1):
                self.run_eq[j] = trailing(Xraw == Xraw[:, j : j + 1])
        self.seed(t, "tune", Xraw)

    def seed(self, t: int, why: str, Xraw: np.ndarray | None = None) -> None:
        """_tune's mask (why == 'tune') and _seed_std on the window [t - W, t)."""
        if Xraw is None:
            Xraw = self.raw(t - W, t)
        if why == "tune":
            self.maskout = self.ident_mask(Xraw)
            self.evid_only = np.zeros(self.p + 1, dtype=bool)
        elif self.kind != "ridge" and self.evid_only.any():
            self.maskout = self.maskout & ~self.evid_only
        self.evid_only = np.zeros(self.p + 1, dtype=bool)
        Xa = Xraw.copy()
        Xa[:, self.maskout] = 0.0
        sw, out, v, ev, _, _ = PLP.std_weights(Xa, self.locked, self.ref, self.av, full=True)
        ai = self.av[0]
        assert (out | self.maskout)[ai].all(), "binomial rule not moot: a live availability average"
        new = out & ~self.maskout
        self.ev_v, self.ev_d = v, ev.astype(np.int64)
        if self.kind == "ridge":
            self.sd_run = PLP._trailing_const(Xa)
        else:
            if new.any():
                self.maskout = self.maskout | new
                self.evid_only = new
            self.w = sw.copy()
        self.seed_t = t
        self.seeds.append((t, why))

    def step(self, t: int) -> None:
        """The roll into forecast t's window (entering row t - 1, leaving row t - 1 - W), then the solve's tune / KKT
        reseed.  Called for every t > S0."""
        ua = self.row(t - 1)
        ur = self.row(t - 1 - W)
        prev = self.row(t - 2)
        n = W
        seeded = False
        if self.kind == "ridge":
            self.sd_run = np.where(ua == prev, np.minimum(self.sd_run + 1, n), 1)
            um, rm = ua.copy(), ur.copy()
            um[self.maskout] = 0.0
            rm[self.maskout] = 0.0
            self.ev_d += (um != self.ev_v).astype(np.int64) - (rm != self.ev_v).astype(np.int64)
        else:
            if self.track:
                self.run_const = np.where(ua == prev, np.minimum(self.run_const + 1, n), 1)
                self.run_eq = np.where(ua[:, None] == ua[None, :], np.minimum(self.run_eq + 1, n), 0)
                const = self.run_const >= n
                dup = np.tril(self.run_eq >= n, -1).any(axis=1)
                gone = (const | dup) & ~self.locked & ~self.maskout
                if gone.any():
                    self.maskout = self.maskout | gone
                    self.seed(t, "mask_add")
                    seeded = True
            if not seeded:
                um, rm = ua.copy(), ur.copy()
                um[self.maskout] = 0.0
                rm[self.maskout] = 0.0
                self.ev_d += (um != self.ev_v).astype(np.int64) - (rm != self.ev_v).astype(np.int64)
                ev = PLP.group_evidence(self.ev_d, self.ref)
                low = (ev < PLP.STD_MIN_EVIDENCE) & ~self.maskout & ~self.locked
                if low.any():
                    self.maskout = self.maskout | low
                    self.evid_only = self.evid_only | low
                    self.seed(t, "evidence")
                    seeded = True
        if (t - self.t0) % TUNE_PER == 0:
            self.seed(t, "tune")
        elif self.kind != "ridge" and not seeded and t in self.kkt:
            self.seed(t, "kkt")

    def active_weights(self, Gc_diag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(positions of the live columns in self.g, their weights) for the solve at the current bar.  Ridge: s of the
        bar's window (Gc_diag = the window's centred sums of squares of self.g), a column out if s < STD_MIN (s = 0 on an
        exactly constant column), its evidence < STD_MIN_EVIDENCE or masked; lasso / elastic net: the held seed weights
        on the unmasked columns."""
        if self.kind == "ridge":
            s = np.sqrt(np.maximum(Gc_diag / W, 0.0))
            s[self.sd_run[:-1] >= W] = 0.0
            s[self.maskout[:-1]] = 0.0
            ev = PLP.group_evidence(self.ev_d, self.ref)[:-1]
            out = (s < PLP.STD_MIN) | (ev < PLP.STD_MIN_EVIDENCE)
            keep = np.flatnonzero(~out)
            return keep, s[keep]
        keep = np.flatnonzero(~self.maskout[:-1])
        return keep, self.w[:-1][keep]


# ============================================================================ solves
def ridge_pred(G, c, x, w, pfa, alpha, ybar) -> float:
    """(Z'Z + diag(alpha pf)) gamma = Z'y in the window-standardized coordinates z = (x - m) / w (centred window: the
    free intercept), forecast ybar + ((x_t - m) / w)' gamma."""
    Gz = G / np.outer(w, w)
    Gz[np.diag_indices_from(Gz)] += alpha * pfa
    g = sla.solve(Gz, c / w, assume_a="pos", check_finite=False)
    return float(ybar + (x / w) @ g)


def kkt_ok(G, c, mu, lam2, b, tol) -> bool:
    r = c - G @ b - lam2 * b
    on = b != 0.0
    if np.any(np.abs(r[on] - mu[on] * np.sign(b[on])) > tol):
        return False
    return not np.any(np.abs(r[~on]) > mu[~on] + tol)


STATS = {"active_set": 0, "fallback": 0, "fallback_fail": 0, "iters": 0}


def enet_exact(G, c, mu, lam2, A0: list, sg0: list, tol: float):
    """min 0.5 b'Gb - c'b + sum mu_j |b_j| + 0.5 lam2_j b_j^2: the active-set solve from the warm support (A0, sg0):
    the exact solve on the support with its signs, sign flips dropped, the worst KKT violator added (the runner's
    _active_repair, warm); None if it does not certify within 60 steps."""
    H = G.copy()
    H[np.diag_indices_from(H)] += lam2
    A, sg = list(A0), list(sg0)
    x = np.zeros(0)
    for _ in range(60):
        STATS["iters"] += 1
        if A:
            Ai = np.array(A)
            try:
                x = np.linalg.solve(H[np.ix_(Ai, Ai)], c[Ai] - mu[Ai] * np.array(sg))
            except np.linalg.LinAlgError:
                return None
            keep = np.sign(x) == np.array(sg)
            if not keep.all():
                A = [q for q, k in zip(A, keep) if k]
                sg = [s for s, k in zip(sg, keep) if k]
                continue
        b = np.zeros(len(c))
        if A:
            b[np.array(A)] = x
        r = c - H @ b
        v = np.abs(r) - mu
        if A:
            v[np.array(A)] = -np.inf
        jw = int(np.argmax(v))
        if v[jw] <= tol:
            return b if kkt_ok(G, c, mu, lam2, b, tol) else None
        A.append(jw)
        sg.append(1.0 if r[jw] > 0 else -1.0)
    return None


def enet_pred(G, c, x, w, pfa, alpha, l1, ybar, keys, warm: dict) -> tuple[float, dict]:
    """The lasso / elastic net in z = (x - m) / w with mu_j = n alpha l1 pf_j, lam2_j = n alpha (1 - l1) pf_j (the
    runner's units), solved exactly (enet_exact warm; else the runner's batch solve enet_coef_w: homotopy from scratch,
    KKT-certified and repaired); returns the forecast and the new warm support {key: sign}."""
    Gz = G / np.outer(w, w)
    cz = c / w
    mu = W * alpha * l1 * pfa
    lam2 = W * alpha * (1.0 - l1) * pfa
    tol = KKT_TOL_REL * max(float(np.abs(cz).max()), 1.0)
    pos = {k: i for i, k in enumerate(keys)}
    A0 = [pos[k] for k in warm if k in pos]
    sg0 = [warm[k] for k in warm if k in pos]
    b = enet_exact(Gz, cz, mu, lam2, A0, sg0, tol)
    if b is None:
        STATS["fallback"] += 1
        rep = dict(PLP.REPAIR)
        b = PLP.enet_coef_w(Gz, cz, W, alpha, l1, pfa, pfa)
        if PLP.REPAIR["fail"] > rep["fail"]:
            STATS["fallback_fail"] += 1
    else:
        STATS["active_set"] += 1
    nz = np.flatnonzero(b)
    return float(ybar + (x / w) @ b), {keys[i]: float(np.sign(b[i])) for i in nz}


# ============================================================================ the walk
def c3_days(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """C3's sample: the evaluation days at the quantiles i / 20 (nearest), plus 2020-03-16."""
    import ablate_score as AS

    days = AS.evaluation_days(AS.core_rv())
    idx = sorted({int(round(q * (len(days) - 1))) for q in np.linspace(0, 1, 21)})
    out = [days[i] for i in idx]
    if pd.Timestamp("2020-03-16") not in out:
        out.append(pd.Timestamp("2020-03-16"))
    return sorted(out)


def cmd_walk(a) -> int:
    t_wall, t_cpu = time.time(), time.process_time()
    test = a.test
    D = load(test, Path(a.design).resolve())
    X, y, dates = D["X"], D["y"], D["dates"]
    nb, p = D["nb"], D["p"]
    S0, n_rows = walk_rows(D)
    if a.max_bars:
        n_rows = S0 + int(a.max_bars)
    nt = n_rows - S0
    fdates = dates[S0 - W : n_rows - W]
    fday = fdates.normalize()
    M = Maps(Path(a.work), test)
    k0_only = bool(a.k0_only)  # the KKT-reseed identification walks (RULE amendment B): k = 0 only
    classes = [c for c in a.classes.split(",") if c]
    if "h" in classes:
        assert test == "A", "the HAR arm's settings (G1) are test A's"
        classes = [c for c in classes if c != "h"] + ["ridge_h", "lasso_h", "enet_h"]
    kkt = {k: set(v) for k, v in (json.loads(Path(a.kkt).read_text()).items() if a.kkt else [])}
    say(f"test {test}: base {nb} + block {p}; solves {nt} from row {S0} ({fdates[0]}); classes {classes}")
    # column sets in the full cache's order: base (k-arms, c1, c3, k0) and base + block (c2)
    cp = D["cache_pos"]
    gK = np.argsort(cp[:nb], kind="stable")
    gC2 = np.argsort(cp, kind="stable")
    names = D["names"]
    bbK = PLP.backbone_mask([names[i] for i in gK])
    bbC2 = PLP.backbone_mask([names[i] for i in gC2])
    trk: dict[str, dict[str, Tracker]] = {}
    for cl in classes:
        if cl == "ols":
            continue
        kind = cl.split("_")[0]
        ks = kkt.get(cl, set())
        trk[cl] = {"K": Tracker(kind, gK, [names[i] for i in gK], X, ks)}
        if not cl.endswith("_h") and not k0_only:
            trk[cl]["C2"] = Tracker(kind, gC2, [names[i] for i in gC2], X, kkt.get(cl + "|c2", set()))
        for tr in trk[cl].values():
            tr.t0 = S0
    c3 = {d for d in c3_days(dates)} if not (a.max_bars or k0_only) else set()
    g2_days = {pd.Timestamp(d) for d in G2_DAYS}
    cols: list[str] = []
    for cl in classes:
        arms = ["k0"] if (cl.endswith("_h") or k0_only) else ARMS + C3_ARMS
        if k0_only:
            pass
        elif cl == "ols":
            arms = arms + ["c2act"]
        if cl == "ridge":
            arms = arms + ["c1iso", "c2act"]
        cols += [f"{cl}|{m}" for m in arms]
    ci = {cn: j for j, cn in enumerate(cols)}
    preds = np.full((nt, len(cols)), np.nan)
    warm: dict[tuple[str, str], dict] = {}
    rec_g2: list[dict] = []
    n = float(W)
    S = c = sx = None
    sy = 0.0
    sw_t = -1  # the bar the lasso / elastic-net score weights were last set (the day's first bar)
    sF: dict[str, np.ndarray] = {}  # held score weights per class
    Pa = np.zeros((p, 0))
    na = 0
    t_map = t_solve = t_trk = 0.0
    sums_a = SumsA(D, S0) if test == "A" else None
    for i in range(nt):
        t = S0 + i
        if i % TUNE_PER == 0:
            Aw, yw = X[t - W : t], y[t - W : t]
            S, c, sx, sy = Aw.T @ Aw, Aw.T @ yw, Aw.sum(axis=0), float(yw.sum())
        else:
            u, v = X[t - 1], X[t - 1 - W]
            S += np.outer(u, u) - np.outer(v, v)
            c += u * float(y[t - 1]) - v * float(y[t - 1 - W])
            sx += u - v
            sy += float(y[t - 1]) - float(y[t - 1 - W])
        ts = time.perf_counter()
        for cl, tt in trk.items():
            for tr in tt.values():
                if i == 0:
                    tr.init(t)
                else:
                    tr.step(t)
        t_trk += time.perf_counter() - ts
        new_day = i == 0 or fday[i] != fday[i - 1]
        if new_day:
            j = M.idx[t]
            Pa, na = (np.zeros((p, 0)), 0) if k0_only else M.P_all(j)
            act_day = np.flatnonzero(M.act[j])
            sw_t = t
        mu = sx / n
        ybar = sy / n
        Gc = S - n * np.outer(mu, mu)
        cc = c - n * mu * ybar
        dx = X[t] - mu
        ts = time.perf_counter()
        Gbb, GbZ, GZZ = Gc[:nb, :nb], Gc[:nb, nb:], Gc[nb:, nb:]
        GZP = GZZ @ Pa
        GFF = Pa.T @ GZP
        GbF = GbZ @ Pa
        cF = Pa.T @ cc[nb:]
        xF = Pa.T @ dx[nb:]
        dF = np.diag(GFF).copy()
        sFbar = np.sqrt(np.maximum(dF / n, 0.0))
        # score subsets (positions in Pa): pca k, pls k, c1
        fsets = {"k0": np.arange(0)}
        for k in K_PCA:
            fsets[f"pca{k}"] = np.arange(min(k, na))
        for k in K_PLS:
            fsets[f"pls{k}"] = na + np.arange(k)
        fsets["c1"] = np.arange(na)
        fkeys = {m: [("f", int(q)) if q < na else ("r", int(q - na)) for q in fs] for m, fs in fsets.items()}
        t_map += time.perf_counter() - ts
        is_c3 = fday[i] in c3
        c3_P: dict[str, np.ndarray] = {}
        if is_c3:  # the bar's own map (loadings from the window ending at the previous bar)
            ts = time.perf_counter()
            if new_day:
                Pb = Pa[:, :NPC]
            elif test == "A":
                mpb = map_full_sums(S, c, sx, sy, sums_a.ev_ok(t), nb, p)  # type: ignore[union-attr]
                Pb = np.zeros((p, NPC))
                Pb[mpb["a"]] = mpb["Pv"][:, :NPC]
            else:
                mpb = map_full_window(np.array(X[t - W : t, nb:]), y[t - W : t], D["k1"], D["col_ser"])
                Pb = np.zeros((p, NPC))
                Pb[mpb["a"]] = mpb["Pv"][:, :NPC]
            for k in (3, 10):
                c3_P[f"pca{k}_bar"] = Pb[:, :k]
            t_map += time.perf_counter() - ts
        ts = time.perf_counter()
        for cl in classes:
            if cl == "ols":
                for m in ARMS:
                    if m == "c2":
                        ix = np.arange(nb + p)
                        preds[i, ci[f"{cl}|{m}"]] = FH.minnorm_pred(Gc, cc, dx, ybar)[0]
                        continue
                    fs = fsets[m]
                    G = np.block([[Gbb, GbF[:, fs]], [GbF[:, fs].T, GFF[np.ix_(fs, fs)]]])
                    preds[i, ci[f"{cl}|{m}"]] = FH.minnorm_pred(
                        G, np.r_[cc[:nb], cF[fs]], np.r_[dx[:nb], xF[fs]], ybar
                    )[0]
                ix = np.r_[np.arange(nb), nb + act_day]
                preds[i, ci["ols|c2act"]] = FH.minnorm_pred(Gc[np.ix_(ix, ix)], cc[ix], dx[ix], ybar)[0]
                for m, Pb_ in c3_P.items():
                    GbB = GbZ @ Pb_
                    G = np.block([[Gbb, GbB], [GbB.T, Pb_.T @ GZZ @ Pb_]])
                    preds[i, ci[f"{cl}|{m}"]] = FH.minnorm_pred(
                        G, np.r_[cc[:nb], Pb_.T @ cc[nb:]], np.r_[dx[:nb], Pb_.T @ dx[nb:]], ybar
                    )[0]
                continue
            kind = cl.split("_")[0]
            alpha, l1 = (PEN_H if cl.endswith("_h") else PEN)[kind]
            r_har = 1.0 if cl.endswith("_h") else R_HAR
            trK = trk[cl]["K"]
            gk = gK
            keepK, wK = trK.active_weights(np.diag(Gc)[gk])
            bK = gk[keepK]  # positions in [base | block] (base part: < nb)
            pfb = np.where(bbK[keepK], r_har, 1.0)
            Gbb_a = Gc[np.ix_(bK, bK)]
            cb, xb = cc[bK], dx[bK]
            if kind != "ridge":
                if new_day or trK.seed_t == t:
                    sF[cl] = sFbar.copy()  # the scores' weights: the later of the day's first bar and the last seed
                wF = sF[cl]
            else:
                wF = sFbar
            arms = ["k0"] if (cl.endswith("_h") or k0_only) else ARMS
            for m in arms:
                if m == "c2":
                    trC = trk[cl]["C2"]
                    keepC, wC = trC.active_weights(np.diag(Gc)[gC2])
                    ix = gC2[keepC]
                    pfa = np.where(bbC2[keepC], r_har, 1.0)
                    G, cv, xv, w = Gc[np.ix_(ix, ix)], cc[ix], dx[ix], wC
                    keys = [("g", int(q)) for q in ix]
                else:
                    fs = fsets[m]
                    GbFa = GbF[np.ix_(bK, fs)] if fs.size else np.zeros((len(bK), 0))
                    G = np.block([[Gbb_a, GbFa], [GbFa.T, GFF[np.ix_(fs, fs)]]])
                    cv, xv = np.r_[cb, cF[fs]], np.r_[xb, xF[fs]]
                    w = np.r_[wK, wF[fs]]
                    pfa = np.r_[pfb, np.ones(len(fs))]
                    keys = [("g", int(q)) for q in bK] + fkeys[m]
                if kind == "ridge":
                    preds[i, ci[f"{cl}|{m}"]] = ridge_pred(G, cv, xv, w, pfa, alpha, ybar)
                else:
                    pr, warm[(cl, m)] = enet_pred(G, cv, xv, w, pfa, alpha, l1, ybar, keys, warm.get((cl, m), {}))
                    preds[i, ci[f"{cl}|{m}"]] = pr
            if cl.endswith("_h") or k0_only:
                continue
            for m, Pb_ in c3_P.items():  # C3: the bar's own loadings; the scores' weights at this bar
                GbB = GbZ[bK] @ Pb_
                GBB = Pb_.T @ GZZ @ Pb_
                G = np.block([[Gbb_a, GbB], [GbB.T, GBB]])
                cv, xv = np.r_[cb, Pb_.T @ cc[nb:]], np.r_[xb, Pb_.T @ dx[nb:]]
                w = np.r_[wK, np.sqrt(np.maximum(np.diag(GBB) / n, 0.0))]
                pfa = np.r_[pfb, np.ones(Pb_.shape[1])]
                keys = [("g", int(q)) for q in bK] + [("f", q) for q in range(Pb_.shape[1])]
                if kind == "ridge":
                    preds[i, ci[f"{cl}|{m}"]] = ridge_pred(G, cv, xv, w, pfa, alpha, ybar)
                else:
                    pr, warm[(cl, m)] = enet_pred(G, cv, xv, w, pfa, alpha, l1, ybar, keys, warm.get((cl, m), {}))
                    preds[i, ci[f"{cl}|{m}"]] = pr
            if cl == "ridge" and new_day:  # GC1 (ridge): unstandardized full rotation vs the active block columns
                fs = fsets["c1"]
                GbFa = GbF[np.ix_(bK, fs)]
                G = np.block([[Gbb_a, GbFa], [GbFa.T, GFF[np.ix_(fs, fs)]]])
                w = np.r_[wK, np.ones(len(fs))]
                preds[i, ci["ridge|c1iso"]] = ridge_pred(
                    G, np.r_[cb, cF[fs]], np.r_[xb, xF[fs]], w, np.r_[pfb, np.ones(len(fs))], alpha, ybar
                )
                ix = np.r_[bK, nb + act_day]
                wz = np.sqrt(np.maximum(np.diag(Gc)[nb + act_day] / n, 0.0))
                preds[i, ci["ridge|c2act"]] = ridge_pred(
                    Gc[np.ix_(ix, ix)], cc[ix], dx[ix], np.r_[wK, wz], np.r_[pfb, np.ones(len(act_day))], alpha, ybar
                )
        t_solve += time.perf_counter() - ts
        if fday[i] in g2_days and (new_day or i == nt - 1 or fday[i + 1] != fday[i]):
            for cl, tt in trk.items():
                rr = dict(cls=cl, t=t, date=str(fdates[i]), score_w_t=sw_t)
                for nm, tr in tt.items():
                    rr[f"seed_{nm}"] = tr.seed_t
                    rr[f"mask_{nm}"] = [int(q) for q in np.flatnonzero(tr.maskout[:-1])]
                    if tr.kind == "ridge":
                        kp, _ = tr.active_weights(np.diag(Gc)[tr.g])
                        rr[f"keep_{nm}"] = [int(q) for q in kp]
                rec_g2.append(rr)
        if i % 1000 == 0:
            el = time.time() - t_wall
            say(
                f"  bar {i:,} / {nt:,} ({fdates[i]}); {el:.0f} s (trackers {t_trk:.0f}, gram {t_map:.0f}, solves "
                f"{t_solve:.0f}); {STATS}"
            )
    sec = time.time() - t_wall
    sfx = "_smoke" if a.max_bars else ("_k0" if k0_only else "")
    out = Path(a.work) / (a.out_name or f"walk_{test}_{'-'.join(classes)}{sfx}")
    out.mkdir(parents=True, exist_ok=True)
    f = pd.DataFrame({"date": fdates, "design_row": np.arange(S0, n_rows), "true_adj": y[S0:n_rows]})
    for cn, j in ci.items():
        f[cn] = preds[:, j]
    f.to_parquet(out / "forecasts.parquet", index=False)
    info = dict(
        rule=RULE,
        label=LABEL,
        test=test,
        design=D["design"],
        classes=classes,
        walk_start_row=S0,
        first=str(fdates[0]),
        last=str(fdates[-1]),
        bars=nt,
        seeds={cl: {nm: tr.seeds for nm, tr in tt.items()} for cl, tt in trk.items()},
        kkt_reseeds_applied={k: sorted(v) for k, v in kkt.items()},
        c3_days=[str(d.date()) for d in sorted(c3)],
        solver=dict(STATS),
        runner_repair=dict(PLP.REPAIR),
        wall_sec=round(sec, 1),
        cpu_sec=round(time.process_time() - t_cpu, 1),
        tracker_sec=round(t_trk, 1),
        gram_sec=round(t_map, 1),
        solve_sec=round(t_solve, 1),
        host=platform.node(),
        python=platform.python_version(),
        numpy=np.__version__,
        threads="1",
        g2_records=rec_g2,
    )
    (out / "walk.json").write_text(json.dumps(info, indent=1, default=str))
    say(f"walk done in {sec:.0f} s -> {out}; solver {STATS}; repair {PLP.REPAIR}")
    return 0


# ============================================================================ the merged walk
def walk_dirs(work: Path, test: str) -> list[Path]:
    return sorted(
        d
        for d in work.glob(f"walk_{test}_*")
        if not d.name.endswith(("_smoke", "_k0"))
        and "_k0_it" not in d.name
        and (d / "forecasts.parquet").is_file()
        and (d / "walk.json").is_file()
    )


def merged(work: Path, test: str) -> tuple[pd.DataFrame, dict]:
    """Every class's walk table joined on the forecast bars (one row a solve, 33,997)."""
    f = None
    infos = {}
    for d in walk_dirs(work, test):
        g = pd.read_parquet(d / "forecasts.parquet")
        infos[d.name] = json.loads((d / "walk.json").read_text())
        if f is None:
            f = g
        else:
            assert np.array_equal(f["date"].to_numpy(), g["date"].to_numpy()), d
            assert np.array_equal(f["true_adj"].to_numpy(), g["true_adj"].to_numpy()), d
            dup = [c for c in g.columns if c in f.columns and c not in ("date", "design_row", "true_adj")]
            assert not dup, (d, dup)
            f = pd.concat([f, g.drop(columns=["date", "design_row", "true_adj"])], axis=1)
    assert f is not None, f"no walk of test {test}"
    return f, infos


def templates() -> pd.DataFrame:
    return pd.read_parquet(
        FH.WS_FCST, columns=["model", "predictor_set", "date", "pred_adj", "true_adj", "B", "plain", "rv_actual"]
    )


def base_ref(test: str, cls: str, T: pd.DataFrame) -> pd.Series:
    """The class's own BASE walk (pred_adj): test A, Table 1's HAR arm; test B, the ablation's add_rm_fomc."""
    if test == "A":
        t = T[(T["model"] == cls) & (T["predictor_set"] == "baseline")]
        return pd.Series(t["pred_adj"].to_numpy(float), index=pd.DatetimeIndex(t["date"]))
    b = pd.read_parquet(BASE_B_FC, columns=["date", f"{cls}|add_rm_fomc"])
    return pd.Series(b[f"{cls}|add_rm_fomc"].to_numpy(float), index=pd.DatetimeIndex(b["date"]))


# ============================================================================ gates
def direct_bar(D: dict, M: Maps, r: int, rec: dict, cls: str, maps_t0: int) -> list[dict]:
    """G2 at the design row r for one class: every arm's forecast from a direct batch fit on the materialized window
    (the map from scratch: SVD PCA, NIPALS PLS, the entry rule from scratch; C1 with the walk's eigenvectors; the
    weights recomputed from the materialized windows; pool_linear_pretune.fit_std's ridge / batch_z route)."""
    X, y = D["X"], D["y"]
    nb = D["nb"]
    kind = cls
    alpha, l1 = PEN[kind]
    t0 = maps_t0
    Zw = np.array(X[t0 - W : t0, nb:])
    mu, sd = Zw.mean(axis=0), Zw.std(axis=0)
    _, cnt = PLP.col_mode(Zw)
    ev_ok = (W - cnt)[D["k1"]][D["col_ser"]] >= FH.STD_MIN_EVIDENCE
    a_ = np.flatnonzero((sd > FH.STD_MIN) & ev_ok)
    Zs = (Zw[:, a_] - mu[a_]) / sd[a_]
    del Zw
    _, sv, Vt = np.linalg.svd(Zs, full_matrices=False)
    Vp = Vt[:NPC].T
    yw = y[t0 - W : t0]
    Xd, yv = Zs.copy(), yw - yw.mean()
    Wl, Pl = [], []
    for _ in range(NPLS):
        w_ = Xd.T @ yv
        w_ /= np.linalg.norm(w_)
        tt = Xd @ w_
        t2 = float(tt @ tt)
        pl = Xd.T @ tt / t2
        yv = yv - float(yv @ tt) / t2 * tt
        Xd -= np.outer(tt, pl)
        Wl.append(w_)
        Pl.append(pl)
    del Xd, Zs
    Wm, Pmat = np.array(Wl).T, np.array(Pl).T
    j = M.idx[t0]
    act_w = np.flatnonzero(M.act[j])
    same_act = bool(np.array_equal(act_w, a_))
    na = int(M.n_active[j])
    Pv_walk = np.asarray(M.Pv[j][:, :na])  # C1: the walk's eigenvectors (V / sd on the active rows)
    cp = D["cache_pos"]
    gK = np.argsort(cp[:nb], kind="stable")
    gC2 = np.argsort(cp, kind="stable")
    names = D["names"]
    bbK = PLP.backbone_mask([names[i] for i in gK])
    bbC2 = PLP.backbone_mask([names[i] for i in gC2])
    lo = min(r, rec.get("seed_K", r), rec.get("seed_C2", r), max(rec["score_w_t"], rec.get("seed_K", -1))) - W
    rows = np.arange(lo, r + 1)
    Xr = np.asarray(X[lo : r + 1])
    Zall = (Xr[:, nb:][:, a_] - mu[a_]) / sd[a_]

    def scores(m: str) -> np.ndarray:
        if m == "k0":
            return np.zeros((len(rows), 0))
        if m == "c1":
            return (Xr[:, nb:] - mu) @ Pv_walk
        k = int(m[3:])
        if m.startswith("pca"):
            return Zall @ Vp[:, :k]
        return Zall @ (Wm[:, :k] @ np.linalg.inv(Pmat[:, :k].T @ Wm[:, :k]))

    cur = slice(r - W - lo, r - lo)  # the bar's window inside Xr

    def win_std(A: np.ndarray, t_: int) -> np.ndarray:
        return A[t_ - W - lo : t_ - lo].std(axis=0)

    out = []
    for m in ARMS:
        if m == "c2":
            pos = gC2
            if kind == "ridge":
                keep = np.array(rec["keep_C2"], dtype=np.int64)
            else:
                keep = np.setdiff1d(np.arange(len(gC2)), rec["mask_C2"])
            gi = pos[keep]
            Bm = Xr[:, gi]
            s_b = win_std(Bm, r) if kind == "ridge" else win_std(Bm, rec["seed_C2"])
            pf_b = np.where(bbC2[keep], R_HAR, 1.0)
            F = np.zeros((len(rows), 0))
            s_f = np.zeros(0)
        else:
            if kind == "ridge":
                keep = np.array(rec["keep_K"], dtype=np.int64)
            else:
                keep = np.setdiff1d(np.arange(len(gK)), rec["mask_K"])
            gi = gK[keep]
            Bm = Xr[:, gi]
            s_b = win_std(Bm, r) if kind == "ridge" else win_std(Bm, rec["seed_K"])
            pf_b = np.where(bbK[keep], R_HAR, 1.0)
            F = scores(m)
            ws = r if kind == "ridge" else max(rec["score_w_t"], rec["seed_K"])
            s_f = win_std(F, ws)
        A = np.column_stack([Bm, F, np.ones(len(rows))])
        s_all = np.r_[s_b, s_f, 1.0]
        locked = np.zeros(A.shape[1], dtype=bool)
        locked[-1] = True
        pf = np.r_[pf_b, np.ones(F.shape[1]), 1.0]
        th, _, _ = PLP.fit_std(
            A[cur].copy(), y[r - W : r], locked, kind, alpha, l1, pf, None, None, sw=(s_all, np.zeros_like(locked))
        )
        out.append(dict(cls=cls, model=m, design_row=r, direct=float(A[r - lo] @ th)))
    return out + [
        dict(cls=cls, model="_map", design_row=r, same_active_set=same_act, sv20_over_sv21=float(sv[19] / sv[20]))
    ]


def cmd_gates(a) -> int:
    import ablate_score as AS

    test = a.test
    o = TESTS[test]["out"]
    o.mkdir(parents=True, exist_ok=True)
    work = Path(a.work)
    f, infos = merged(work, test)
    f.to_parquet(o / "forecasts.parquet", index=False)
    (o / "walk.json").write_text(json.dumps(infos, indent=1, default=str))
    fi = f.set_index("date")
    T = templates()
    rec: dict = {"rule": RULE, "label": LABEL, "test": test, "walks": sorted(infos), "fail": []}
    # G0: the OLS arms = the earlier OLS factor test's forecasts; test B also the stored day maps
    ref = pd.read_parquet(TESTS[test]["ols_ref"]).set_index("date")
    g0: dict = {"reference": str(TESTS[test]["ols_ref"].relative_to(ROOT).as_posix()), "max_rel": {}}
    for m in ["k0", *[f"pca{k}" for k in K_PCA], *[f"pls{k}" for k in K_PLS]]:
        x = fi[f"ols|{m}"].reindex(ref.index)
        g0["max_rel"][m] = float((np.abs(x - ref[m]) / np.abs(ref[m])).max())
    g0["rows"] = len(ref)
    g0["pass"] = bool(max(g0["max_rel"].values()) <= G1_TOL and fi.index.isin(ref.index).sum() == len(ref))
    if test == "B":
        mz = Path(a.design).resolve().parent / "maps.npz"
        z = np.load(mz)
        M = Maps(work, test)
        assert np.array_equal(z["first_row"], M.first_row[-len(z["first_row"]) :]), "map days"
        off = len(M.first_row) - len(z["first_row"])
        mx = 0.0
        act_eq = True
        for jj in range(len(z["first_row"])):
            j = jj + off
            Pw = np.hstack([M.Pv[j][:, :NPC], M.Pr[j]])
            Pz = z["P"][jj]
            nrm = np.linalg.norm(Pz, axis=0)
            mx = max(mx, float((np.abs(Pw - Pz).max(axis=0) / nrm).max()))
            act_eq &= bool(np.array_equal(M.act[j], z["act"][jj]))
        g0["maps"] = dict(
            npz=str(mz),
            md5=md5_file(mz),
            md5_expected=MAPS_B_REF,
            days=len(z["first_row"]),
            max_rel_col=mx,
            active_sets_equal=act_eq,
        )
        g0["pass"] = bool(g0["pass"] and mx <= G1_TOL and act_eq and md5_file(mz) == MAPS_B_REF)
    rec["G0"] = g0
    # G1: k = 0 = the class's existing walk on every bar
    ws_ols = T[(T["model"] == "ols") & (T["predictor_set"] == "baseline")].set_index("date")
    g1: dict = {}
    for cls in PEN:
        col = f"{cls}_h|k0" if test == "A" else f"{cls}|k0"
        b = base_ref(test, cls, T)
        x = fi[col].reindex(b.index)
        rel = np.abs(x - b) / np.abs(b)
        ty = fi["true_adj"].reindex(b.index)
        rel_y = np.abs(ty - ws_ols["true_adj"].reindex(b.index)) / np.abs(ws_ols["true_adj"].reindex(b.index))
        # amendment C: an isolated one-bar difference (both neighbours within 1e-10) is the runner's warm solution
        # accepted within its KKT tolerance; counted separately, passes if <= 1e-8
        rv_ = rel.to_numpy()
        hi = rv_ > G1_TOL
        nb_lo = np.r_[False, hi[:-1]]
        nb_hi = np.r_[hi[1:], False]
        iso = hi & ~nb_lo & ~nb_hi
        rest = np.where(iso, 0.0, rv_)
        kk = sorted({int(s) for inf in infos.values() for s in inf["kkt_reseeds_applied"].get(col.split("|")[0], [])})
        g1[cls] = dict(
            kkt_reseeds_identified=kk,
            n_kkt_reseeds=len(kk),
            isolated_bars=[(str(d), float(v)) for d, v in zip(rel.index[iso], rv_[iso])],
            max_rel_excluding_isolated=float(np.nanmax(rest)),
            walk=col,
            reference="Table 1 HAR arm (w3000 fomcfix forecasts.parquet, baseline)"
            if test == "A"
            else f"{BASE_B_FC.relative_to(ROOT).as_posix()} [{cls}|add_rm_fomc]",
            rows=len(b),
            covered=int(x.notna().sum()),
            max_rel=float(rel.max()),
            median_rel=float(rel.median()),
            n_above_1e_12=int((rel > 1e-12).sum()),
            argmax=str(rel.idxmax()),
            max_rel_true_adj=float(rel_y.max()),
            pass_=bool(
                x.notna().all()
                and np.nanmax(rest) <= G1_TOL
                and (rv_[iso].max() if iso.any() else 0.0) <= 1e-8
                and rel_y.max() <= 1e-14
            ),
        )
    rec["G1"] = g1
    # GC1: the rotation (OLS every bar; ridge, unstandardized scores, at the days' first bars)
    r_o = (np.abs(fi["ols|c1"] - fi["ols|c2act"]) / np.abs(fi["ols|c2act"])).dropna()
    x = fi[["ridge|c1iso", "ridge|c2act"]].dropna()
    r_r = np.abs(x["ridge|c1iso"] - x["ridge|c2act"]) / np.abs(x["ridge|c2act"])
    # amendment D: OLS bars above the tolerance, with the day map's smallest eigenvalue (numerical rank)
    mt = np.load(work / f"maps_{test}" / "meta.npz")
    lmin = {str(d_): float(mt["lam"][j_, mt["n_active"][j_] - 1]) for j_, d_ in enumerate(mt["days"])}
    over = r_o[r_o > GC1_TOL]
    rec["GC1"] = dict(
        ols=dict(
            bars=len(r_o),
            max_rel=float(r_o.max()),
            median_rel=float(r_o.median()),
            n_above_1e_10=int((r_o > 1e-10).sum()),
            above_tol=[(str(d_), float(v_), lmin[str(d_.date())]) for d_, v_ in over.items()],
            median_day_min_eigenvalue=float(np.median(list(lmin.values()))),
            pass_=bool(len(r_o) == len(fi) and r_o.max() <= GC1_TOL),
            blocking=False,
        ),
        ridge_iso=dict(bars=len(r_r), max_rel=float(r_r.max()), pass_=bool(r_r.max() <= GC1_TOL)),
        note="under the add-arm convention every column is standardized, the scores included: ridge's penalty on the full"
        " rotation is alpha sum_k lambda_k c_k^2 = alpha gamma' R gamma in the block's standardized coordinates, not"
        " isotropic, so ridge C1 is not a rotation of C2 (no gate); the rotation itself is gated on the variant with"
        " unstandardized scores (penalty alpha ||c||^2) against ridge on [BASE, the day's active block columns] at every"
        " day's first bar, where the two windows coincide",
    )
    # G2: direct batch fits on the spread days
    D = load(test, Path(a.design).resolve())
    M = Maps(work, test)
    g2rows = []
    recs = [r_ for inf in infos.values() for r_ in inf["g2_records"]]
    for r_ in recs:
        cls = r_["cls"]
        if cls not in PEN:
            continue
        t = int(r_["t"])
        t0 = int(M.first_row[np.searchsorted(M.first_row, t, side="right") - 1])
        for d_ in direct_bar(D, M, t, r_, cls, t0):
            if d_["model"] == "_map":
                g2rows.append(d_)
                continue
            wv = float(fi.loc[fi["design_row"] == t, f"{cls}|{d_['model']}"].iloc[0])
            d_.update(date=r_["date"], walk=wv, rel=abs(wv - d_["direct"]) / abs(d_["direct"]))
            g2rows.append(d_)
        say(f"  G2 {cls} {r_['date']}")
    g2 = pd.DataFrame([x_ for x_ in g2rows if x_["model"] != "_map"])
    mp = [x_ for x_ in g2rows if x_["model"] == "_map"]
    rec["G2"] = dict(
        days=G2_DAYS,
        checks=len(g2),
        max_rel=float(g2["rel"].max()),
        max_rel_by_model=g2.groupby("model")["rel"].max().to_dict(),
        max_rel_by_class=g2.groupby("cls")["rel"].max().to_dict(),
        active_sets_equal=bool(all(x_["same_active_set"] for x_ in mp)),
        min_sv20_over_sv21=float(min(x_["sv20_over_sv21"] for x_ in mp)),
        rows=g2.to_dict(orient="records"),
        tol=G2_TOL,
        pass_=bool(len(g2) and g2["rel"].max() <= G2_TOL),
        note="direct: the window materialized, the map from scratch (SVD PCA, NIPALS PLS, the entry rule from scratch;"
        " C1 with the walk's eigenvectors, since ridge / lasso / elastic net on standardized scores are not invariant to"
        " rotations inside nearly tied eigenspaces), the weights recomputed from the materialized windows (ridge: the"
        " bar's; lasso / elastic net: the seed window's, the scores' at the later of the day's first bar and the seed),"
        " the walk's masks, pool_linear_pretune.fit_std (ridge_keep / batch_z: the batch homotopy, KKT-certified)",
    )
    # G3: the scored set
    rv = AS.core_rv()
    days = AS.evaluation_days(rv)
    ev = AS.event_days(days)
    panel = {c_: fi[c_] ** 2 for c_ in fi.columns if "|" in c_ and not c_.endswith(("_bar", "c1iso", "c2act"))}
    idx = AS.scored({k: v.dropna() for k, v in panel.items()}, rv, days)
    sdd = pd.DatetimeIndex(idx.normalize())
    bpd = pd.Series(1, index=sdd).groupby(level=0).size()
    g3 = dict(
        n_bars=len(idx),
        n_days=int(sdd.nunique()),
        all_days_13_bars=bool((bpd == 13).all()),
        n_event_days=len(ev),
        event_bars=int(sdd.isin(ev).sum()),
        ordinary_days=int(sdd.nunique()) - len(ev),
    )
    g3["pass_"] = bool(
        g3["n_bars"] == N_SCORED
        and g3["n_days"] == N_DAYS
        and g3["n_event_days"] == N_EVENT
        and g3["event_bars"] == 13 * N_EVENT
    )
    rec["G3"] = g3
    for k, v in (
        ("G0", g0["pass"]),
        ("G1", all(x_["pass_"] for x_ in g1.values())),
        ("G2", rec["G2"]["pass_"]),
        ("G3", g3["pass_"]),
        ("GC1", rec["GC1"]["ridge_iso"]["pass_"]),  # amendment D: the OLS check is reported, not blocking
    ):
        if not v:
            rec["fail"].append(k)
    rec["pass"] = not rec["fail"]
    (o / "gates.json").write_text(json.dumps(rec, indent=1, default=str))
    say(
        f"gates {test}: G0 {g0['pass']} {g0['max_rel']}; G1 "
        f"{ {k: (v['pass_'], v['max_rel']) for k, v in g1.items()} }; G2 {rec['G2']['pass_']} {rec['G2']['max_rel']:.2e}; "
        f"G3 {g3}; GC1 {rec['GC1']['ols']['max_rel']:.2e} / {rec['GC1']['ridge_iso']['max_rel']:.2e}; fail {rec['fail']}"
    )
    return 0 if rec["pass"] else 3


# ============================================================================ scores
TREES_R = dict(  # the trees' factor test and within-block ladder (ORDINARY days), for the summary only
    factor="results/beyond_trees_2026-10-27/factor/reading.json",
    ladder="results/ablations_2014start_w3000/STATUS.md (17:55, 27 Oct): LightGBM R(5) 0.72, XGBoost R(5) 0.91",
)
SHORT = [f"pca{k}" for k in K_PCA if k <= 3] + [f"pls{k}" for k in K_PLS if k <= 3]


def cmd_score(a) -> int:
    import ablate_score as AS
    from src.evaluation.diebold_mariano import dm_test

    test = a.test
    o = TESTS[test]["out"]
    g = json.loads((o / "gates.json").read_text())
    assert g["pass"], f"gates failed: {g['fail']} -- no scoring"
    work = Path(a.work)
    f, infos = merged(work, test)
    fi = f.set_index("date")
    T = templates()
    ws = T[(T["model"] == "ols") & (T["predictor_set"] == "baseline")].set_index("date").reindex(fi.index)
    Bv, rv_a = ws["B"].to_numpy(float), ws["rv_actual"]
    assert np.isfinite(Bv).all()
    rv = AS.core_rv()
    days = AS.evaluation_days(rv)
    ev = AS.event_days(days)
    plain: dict[str, pd.Series] = {}

    def put(name: str, pa: pd.Series) -> None:
        plain[name] = pd.Series(pa.reindex(fi.index).to_numpy(float) ** 2 * Bv, index=fi.index)

    for cls in PEN:
        put(f"{cls}|BASE", base_ref(test, cls, T))
        t = T[(T["model"] == cls) & (T["predictor_set"] == "all_features")]
        put(f"{cls}|FULL", pd.Series(t["pred_adj"].to_numpy(float), index=pd.DatetimeIndex(t["date"])))
        for m in ARMS:
            put(f"{cls}|{m}", fi[f"{cls}|{m}"])
    if test == "A":
        ob = T[(T["model"] == "ols") & (T["predictor_set"] == "baseline")]
        put("ols|BASE", pd.Series(ob["pred_adj"].to_numpy(float), index=pd.DatetimeIndex(ob["date"])))
    else:
        b = pd.read_parquet(BASE_B_FC, columns=["date", "ols|add_rm_fomc"])
        put("ols|BASE", pd.Series(b["ols|add_rm_fomc"].to_numpy(float), index=pd.DatetimeIndex(b["date"])))
    for m in ARMS:
        put(f"ols|{m}", fi[f"ols|{m}"])
    idx = AS.scored(plain, rv, days)
    assert len(idx) == N_SCORED, len(idx)
    dday = pd.DatetimeIndex(idx.normalize())
    span_ix = {"ordinary": idx[~dday.isin(ev)], "all": idx, "event": idx[dday.isin(ev)]}
    yv = rv_a

    def loss(name: str, ix: pd.DatetimeIndex) -> np.ndarray:
        pr, yy = plain[name].reindex(ix).to_numpy(float), yv.reindex(ix).to_numpy(float)
        r = yy / pr
        return r - np.log(r) - 1.0

    rows = []
    readings: dict = {}
    for cls in [*PEN, "ols"]:
        for sp, ix in span_ix.items():
            L = {m: loss(f"{cls}|{m}", ix) for m in ["BASE", *ARMS] + (["FULL"] if cls != "ols" else [])}
            if cls == "ols":  # the earlier tests' L_full: the lower of Table 1's ridge / elastic-net full sets
                lr, le = loss("ridge|FULL", ix), loss("enet|FULL", ix)
                L["FULL"] = lr if lr.mean() <= le.mean() else le
                full_ref = "ridge" if lr.mean() <= le.mean() else "enet"
            else:
                full_ref = cls
            q = {m: float(v.mean()) for m, v in L.items()}
            gain = q["BASE"] > q["FULL"]
            dmf = dm_test(AS.daily(L["FULL"], ix), AS.daily(L["BASE"], ix), hac_lag=HAC_LAG)
            for m in ["BASE", *ARMS, "FULL"]:
                row: dict = dict(
                    test=test,
                    cls=cls,
                    span=sp,
                    model=m,
                    n_bars=len(ix),
                    n_days=int(pd.DatetimeIndex(ix.normalize()).nunique()),
                    qlike=q[m],
                    pct_vs_base=100.0 * (q[m] / q["BASE"] - 1.0),
                    full_ref=full_ref,
                    full_beats_base=bool(gain),
                    full_vs_base_dm=float(dmf["dm"]),
                    full_vs_base_p=float(dmf["p"]),
                    R=(q["BASE"] - q[m]) / (q["BASE"] - q["FULL"]) if gain else np.nan,
                    R_info=(q["BASE"] - q[m]) / (q["BASE"] - q["FULL"]),
                )
                if m != "BASE":
                    r1 = dm_test(AS.daily(L[m], ix), AS.daily(L["BASE"], ix), hac_lag=HAC_LAG)
                    row.update(dm_base=float(r1["dm"]), p_base=float(r1["p"]), diff_base=float(r1["mean_diff"]))
                if m != "FULL":
                    r2 = dm_test(AS.daily(L[m], ix), AS.daily(L["FULL"], ix), hac_lag=HAC_LAG)
                    row.update(dm_full=float(r2["dm"]), p_full=float(r2["p"]), diff_full=float(r2["mean_diff"]))
                rows.append(row)
    sc = pd.DataFrame(rows)
    for cls in [*PEN, "ols"]:
        od = sc[(sc["cls"] == cls) & (sc["span"] == "ordinary")].set_index("model")
        gain = bool(od.loc["FULL", "full_beats_base"])
        rd: dict = dict(
            full_beats_base=gain,
            full_vs_base_pct=float(od.loc["FULL", "pct_vs_base"]),
            full_vs_base_dm=float(od.loc["FULL", "full_vs_base_dm"]),
            full_vs_base_p=float(od.loc["FULL", "full_vs_base_p"]),
            R={m: (None if np.isnan(od.loc[m, "R"]) else float(od.loc[m, "R"])) for m in ARMS},
            R_info={m: float(od.loc[m, "R_info"]) for m in ARMS},
        )
        if not gain:
            rd["reading"] = "NO GAIN TO RECOVER"
        else:
            shared = [
                m
                for m in SHORT
                if od.loc[m, "R"] >= 0.75 and not (od.loc[m, "p_full"] < 0.05 and od.loc[m, "diff_full"] > 0)
            ]
            full_beats_10 = bool(od.loc["pca10", "p_full"] < 0.05 and od.loc["pca10", "diff_full"] > 0)
            dense = bool(od.loc["pca10", "R"] < 0.5 and full_beats_10)
            w_ = "SHARED FACTOR" if shared else ("DENSE BEYOND A FEW FACTORS" if dense else "INTERMEDIATE")
            if od.loc["FULL", "full_vs_base_p"] >= 0.05:
                w_ += " (gain not significant; R imprecise)"
            rd.update(
                reading=w_,
                shared_arms=shared,
                dense_condition=dense,
                full_beats_pca10_significantly=full_beats_10,
                best_short=max(SHORT, key=lambda m: float(od.loc[m, "R"])),
            )
        if cls == "ols":
            rd["reading"] = "(reference class; the earlier tests' reading: " + rd.get("reading", "") + ")"
        readings[cls] = rd
    # C3: loadings at every bar on the C3 days (scored bars of those days)
    c3d = pd.DatetimeIndex(sorted({pd.Timestamp(d) for inf in infos.values() for d in inf["c3_days"]}))
    c3ix = idx[pd.DatetimeIndex(idx.normalize()).isin(c3d)]
    c3: dict = {"days": [str(d.date()) for d in c3d], "bars": len(c3ix)}
    for cls in [*PEN, "ols"]:
        for k in (3, 10):
            d_, b_ = fi[f"{cls}|pca{k}"].reindex(c3ix), fi[f"{cls}|pca{k}_bar"].reindex(c3ix)
            rel = (np.abs(d_ - b_) / np.abs(d_)).to_numpy()
            yy = yv.reindex(c3ix).to_numpy(float)
            Bc = pd.Series(Bv, index=fi.index).reindex(c3ix).to_numpy(float)
            ql = [
                float(np.mean(yy / (x * x * Bc) - np.log(yy / (x * x * Bc)) - 1.0))
                for x in (d_.to_numpy(), b_.to_numpy())
            ]
            c3[f"{cls}|pca{k}"] = dict(
                max_rel=float(np.nanmax(rel)),
                mean_rel=float(np.nanmean(rel)),
                qlike_daily=ql[0],
                qlike_bar=ql[1],
                pct=100.0 * (ql[1] / ql[0] - 1.0),
            )
    # amendment B (b): C1's extreme bars (information; no trimming)
    c1x: dict = {}
    oix = span_ix["ordinary"]
    for cls in [*PEN, "ols"]:
        c1, k0 = fi[f"{cls}|c1"].reindex(idx), fi[f"{cls}|k0"].reindex(idx)
        q = c1 / k0
        keep = oix[(fi[f"{cls}|c1"].reindex(oix) > 0).to_numpy()]
        lb, lc, lf = (loss(f"{cls}|{m}", keep).mean() for m in ("BASE", "c1", "FULL"))
        c1x[cls] = dict(
            n_extreme=int((np.abs(q - 1) > 0.5).sum()),
            n_nonpositive=int((c1 <= 0).sum()),
            min_pred=float(c1.min()),
            max_pred=float(c1.max()),
            worst_bar=str(q.sub(1).abs().idxmax()),
            ordinary_qlike_without_nonpositive=float(lc),
            ordinary_R_without_nonpositive=float((lb - lc) / (lb - lf)),
        )
    sc.to_csv(o / "scores.csv", index=False, float_format="%.17g")
    tr = json.loads((ROOT / TREES_R["factor"]).read_text())
    out = dict(
        label=LABEL,
        rule=RULE,
        test=test,
        readings=readings,
        c3=c3,
        c1_extremes=c1x,
        trees_factor=tr["models"],
        trees_ladder=TREES_R["ladder"],
    )
    (o / "reading.json").write_text(json.dumps(out, indent=1, default=str))
    summary(test, sc, readings, c3, g, infos, tr, c1x)
    say(f"score {test}: { {k: v['reading'] for k, v in readings.items()} }")
    return 0


def summary(test: str, sc: pd.DataFrame, rd: dict, c3: dict, g: dict, infos: dict, tr: dict, c1x: dict) -> None:
    o = TESTS[test]["out"]

    def f_(v, nd=3):
        return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"

    names = {"ridge": "Ridge", "lasso": "Lasso", "enet": "Elastic net", "ols": "OLS (reference)"}
    base_txt = "Table 1's HAR arm of the class" if test == "A" else "the ablation's add_rm_fomc walk of the class"
    L = [
        f"# {TESTS[test]['title']}",
        "",
        f"*{LABEL[0].upper() + LABEL[1:]}.*  Rule: `{RULE}`, committed before any computation.  Code:"
        " `experiments/factor_pen_2026_10_28.py` (maps, walk, gates, score); generated by its score mode.",
        "",
        f"Each class against its OWN base ({base_txt}) and its OWN full set (Table 1's patched full-set walk of the"
        " class); arms under the add-arm convention (the full set's frozen penalty, r = 0.01 on the HAR block, 1 on the"
        ' component scores and the block\'s columns).  R = (L_BASE - L_arm) / (L_BASE - L_FULL); "no gain to recover"'
        " where L_FULL >= L_BASE.",
        "",
        "## Readings (ORDINARY days)",
        "",
        "| class | FULL vs BASE (% QLIKE, DM, p) | reading | R pca1 / pca3 / pls3 / pca10 / pca20 | R C1 / C2 |",
        "|---|---|---|---|---|",
    ]
    for cls in [*PEN, "ols"]:
        r = rd[cls]
        Rr = r["R"] if r["full_beats_base"] else r["R_info"]
        tag = "" if r["full_beats_base"] else " (info: no gain)"
        L.append(
            f"| {names[cls]} | {r['full_vs_base_pct']:+.2f}%, {r['full_vs_base_dm']:+.2f}, {r['full_vs_base_p']:.3f} |"
            f" {r['reading']} | {' / '.join(f_(Rr[m]) for m in ('pca1', 'pca3', 'pls3', 'pca10', 'pca20'))}{tag} |"
            f" {f_(Rr['c1'])} / {f_(Rr['c2'])}{tag} |"
        )
    L += [
        "",
        "OLS row: the earlier OLS test's design (OLS base and arms; L_full = the lower of Table 1's ridge / elastic-net"
        " full sets), recomputed by this walk (gate G0) and shown for comparison.",
        f"Trees (RULE.txt section 1, the same 36-series block, base HAR + RM + FOMC): R(PCA 10) LightGBM"
        f" {tr['lgbm']['R_pca10']:.3f}, XGBoost {tr['xgb']['R_pca10']:.3f}; best k <= 3 {tr['lgbm']['best_short']}"
        f" {tr['lgbm']['R_best_short']:.3f} / {tr['xgb']['best_short']} {tr['xgb']['R_best_short']:.3f}; within-block ladder"
        " R(top-5 series) LightGBM 0.72, XGBoost 0.91 (ORDINARY days).",
        "",
        "## Scores",
        "",
        "QLIKE % against the class's BASE; DM (HLN, daily means, lag 8) against BASE and against FULL (negative: the row's"
        " loss is lower); R where FULL beats BASE on the span (else --).",
    ]
    for cls in [*PEN, "ols"]:
        for sp in SPANS:
            s = sc[(sc["cls"] == cls) & (sc["span"] == sp)].set_index("model")
            L += [
                "",
                f"### {names[cls]}, {sp} ({int(s['n_days'].iloc[0])} days, {int(s['n_bars'].iloc[0])} bars;"
                f" FULL = {s['full_ref'].iloc[0]})",
                "",
                "| model | QLIKE | % vs BASE | DM vs BASE | p | DM vs FULL | p | R |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for m in ["BASE", *ARMS, "FULL"]:
                x = s.loc[m]
                L.append(
                    f"| {m} | {x['qlike']:.5f} | {x['pct_vs_base']:+.2f} | {f_(x.get('dm_base'), 2)} |"
                    f" {f_(x.get('p_base'))} | {f_(x.get('dm_full'), 2)} | {f_(x.get('p_full'))} | {f_(x['R'])} |"
                )
    L += ["", "## Controls (the author's follow-up: does the PCA construction handicap the factor arms?)", ""]
    L += [
        "- C1 = BASE + all active components of the day; C2 = BASE + the block's value averages themselves (the add-arm"
        " estimator with its own masks).  FULL - C2 = what the components cannot see by construction"
        + (
            " (the availability averages and the FOMC group)"
            if test == "A"
            else " (the 36 series' availability averages)"
        )
        + "; C2 - C1 = the component basis (for the penalized classes the standardization of every score, i.e. a"
        " Mahalanobis rather than isotropic penalty, plus the active-column rule); C1 - pca k = truncation.",
        "",
        "| class | span | R pca10 | R pca20 | R C1 | R C2 | QLIKE C1 vs C2 (%) | QLIKE C2 vs FULL (%) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cls in [*PEN, "ols"]:
        for sp in ("ordinary", "all"):
            s = sc[(sc["cls"] == cls) & (sc["span"] == sp)].set_index("model")
            L.append(
                f"| {names[cls]} | {sp} | {f_(s.loc['pca10', 'R_info'])} | {f_(s.loc['pca20', 'R_info'])} |"
                f" {f_(s.loc['c1', 'R_info'])} | {f_(s.loc['c2', 'R_info'])} |"
                f" {100 * (s.loc['c1', 'qlike'] / s.loc['c2', 'qlike'] - 1):+.2f} |"
                f" {100 * (s.loc['c2', 'qlike'] / s.loc['FULL', 'qlike'] - 1):+.2f} |"
            )
    L += [
        "",
        "(R here = R_info, the ratio even where FULL does not beat BASE; read only where it does.)",
        "",
        "C1's extreme bars (amendment B (b); every scored bar; information, nothing trimmed): "
        + "; ".join(
            f"{names[c]}: {x['n_extreme']} bars with |C1 / k0 - 1| > 0.5, {x['n_nonpositive']} with pred_adj <= 0"
            f" (min {x['min_pred']:.3g}, max {x['max_pred']:.3g}, worst {x['worst_bar']}); ORDINARY R without the"
            f" non-positive bars {x['ordinary_R_without_nonpositive']:.3f}"
            for c, x in c1x.items()
        )
        + ".",
        "",
        f"C3 (loadings at every bar instead of once a day; {len(c3['days'])} days, {c3['bars']} scored bars):",
        "",
        "| class | arm | max rel diff | mean rel diff | QLIKE daily | QLIKE every bar | % |",
        "|---|---|---|---|---|---|---|",
    ]
    for cls in [*PEN, "ols"]:
        for k in (3, 10):
            x = c3[f"{cls}|pca{k}"]
            L.append(
                f"| {names[cls]} | pca{k} | {x['max_rel']:.2e} | {x['mean_rel']:.2e} | {x['qlike_daily']:.5f} |"
                f" {x['qlike_bar']:.5f} | {x['pct']:+.2f} |"
            )
    L += ["", "## Gates (gates.json)", ""]
    L.append(
        f"- G0 {'PASS' if g['G0']['pass'] else 'FAIL'}: the OLS arms vs {g['G0']['reference']}, max relative"
        f" {max(g['G0']['max_rel'].values()):.2e}"
        + (
            f"; day maps vs maps.npz (md5 {g['G0']['maps']['md5']}): max"
            f" {g['G0']['maps']['max_rel_col']:.2e} of the column norm, active sets equal {g['G0']['maps']['active_sets_equal']}"
            if "maps" in g["G0"]
            else ""
        )
        + "."
    )
    for cls, x in g["G1"].items():
        L.append(
            f"- G1 {cls} {'PASS' if x['pass_'] else 'FAIL'}: {x['walk']} vs {x['reference']}, {x['rows']:,} bars,"
            f" max relative {x['max_rel_excluding_isolated']:.2e} (tolerance 1e-10) with {x['n_kkt_reseeds']} KKT"
            f" reseeds of the existing walk identified from its forecasts (amendment B); isolated one-bar differences"
            f" (amendment C): {x['isolated_bars'] or 'none'}."
        )
    L.append(
        f"- G2 {'PASS' if g['G2']['pass_'] else 'FAIL'}: {g['G2']['checks']} direct batch fits on {len(G2_DAYS)} days,"
        f" max relative {g['G2']['max_rel']:.2e} (tolerance 1e-8); active sets equal {g['G2']['active_sets_equal']}."
    )
    L.append(
        f"- G3 {'PASS' if g['G3']['pass_'] else 'FAIL'}: {g['G3']['n_bars']:,} bars, {g['G3']['n_days']:,} days,"
        f" {g['G3']['n_event_days']} EVENT days."
    )
    L.append(
        f"- GC1 ridge {'PASS' if g['GC1']['ridge_iso']['pass_'] else 'FAIL'}: ridge with unstandardized scores vs"
        f" ridge on [BASE, the active block columns] at the days' first bars, max relative"
        f" {g['GC1']['ridge_iso']['max_rel']:.2e}.  GC1 OLS (amendment D, reported, not blocking): OLS C1 vs OLS on"
        f" [BASE, the active block columns], every bar, median relative {g['GC1']['ols']['median_rel']:.1e},"
        f" {g['GC1']['ols']['n_above_1e_10']} bars above 1e-10; the bars above 1e-8 (date, relative difference, the"
        f" day map's smallest correlation eigenvalue; the median over days is"
        f" {g['GC1']['ols']['median_day_min_eigenvalue']:.1e}): {g['GC1']['ols']['above_tol']}.  {g['GC1']['note']}."
    )
    L += ["", "## Provenance", ""]
    for k, inf in infos.items():
        L.append(
            f"- {k}: {inf['bars']:,} solves {inf['first']} .. {inf['last']}, {inf['wall_sec']:.0f} s wall on"
            f" {inf['host']} (numpy {inf['numpy']}, one thread); solver {inf['solver']}; runner repair"
            f" {inf['runner_repair']}; non-tune seeds"
            f" { {cl: {nm: [s_ for s_ in v if s_[1] != 'tune'] for nm, v in tt.items()} for cl, tt in inf['seeds'].items()} }."
        )
    L.append(
        "- Design: Table 1's D13 rebuilt on the laptop in 12 column parts (scratchpad fl/design_parts), as the OLS"
        " tests; maps: the walk's maps mode (scratchpad fpen/maps_<test>)."
    )
    (o / "SUMMARY.md").write_text("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("maps", "walk", "gates", "score"))
    ap.add_argument("--test", choices=("A", "B"), required=True)
    ap.add_argument("--design", default="")
    ap.add_argument("--work", default="")
    ap.add_argument("--classes", default="ridge,lasso,enet")
    ap.add_argument("--max-bars", type=int, default=0)
    ap.add_argument("--kkt", default="")
    ap.add_argument("--k0-only", action="store_true")
    ap.add_argument("--out-name", default="")
    a = ap.parse_args()
    return {"maps": cmd_maps, "walk": cmd_walk, "gates": cmd_gates, "score": cmd_score}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
