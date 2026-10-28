"""The LINEAR factor test beyond HAR + realized moments + FOMC (results/beyond_trees_2026-10-27/RULE_linear.txt with its
Amendment A; requested by the author after the beyond-base results; not pre-registered).  Local (laptop) run.

QUESTION  Does OLS on the add_rm_fomc base (HAR set + the futures' realized moments + the FOMC group) gain from a few
          components of the remaining predictors?  The earlier linear factor test (experiments/factor_har_2026_10_26.py;
          channels RULE item 4) with only the base changed: HAR set -> add_rm_fomc, and with it the component block ->
          the series outside the base (the six groups ew, vw, liq, iv, sent, vd).

DESIGN    Table 1's (D13, FOMC-patched; results/window_sweep_2014start_fomcfix/w3000/run_info.json): 13 regular-hours
          bars pooled, base-5 HAR ladder on the 48-interval grid, W = 39,000 rows (3,000 days), forecasts 2 January 2014
          10:00 to 30 April 2024 16:00 (33,970 bars).
  Base      BASE_LIN: the 132 columns of experiments/ablate_arms.arm_names("add_rm_fomc") in the cache's order (HAR12,
            the realized moments' and the FOMC series' value and availability averages); the ablation's OLS walk on
            them is results/ablations_2014start_w3000/forecasts/linear_wide_rmfomc.parquet "ols|add_rm_fomc".
  Block     the value averages adj_<s>_ma_<k> (k = 1 .. 3125) of the 36 series of ew, vw, liq, iv, sent, vd (216
            columns, the cache's order): the tree version's block (beyond_factor_trees.block_names; RULE.txt CARC
            addendum 1 (b)).
  Map       the tree version's construction (beyond_factor_trees.compute_map, here with 20 principal components): once
            a trading day, directly on the materialized design rows [t0 - W, t0), t0 = the day's first bar: m, s (ddof
            0); active = s > 1e-12 and the series' k = 1 column has >= 250 rows off its window mode (mode_count =
            pool_linear_pretune.col_mode's count); eigenvectors V of the active columns' correlation matrix
            (np.linalg.eigh, descending, each signed to a non-negative sum); SIMPLS weights R of the standardized active
            block against the window's target (y - its window mean).  P = diag(1/s) [V_20 | R_3] on the active rows
            (zero elsewhere); scores f = (z - m) P, held for the day's bars (every training row of a bar's window and
            the bar's own row take the day's map).
  Fit       min-norm OLS with a free intercept, re-estimated at every bar on the bar's 39,000-row window, exactly as
            Table 1's walk (pool_linear_pretune.make_ols_class): eigh pseudo-inverse of the centred Gram (eigenvalues
            below 1e-15 x the largest dropped) times the centred cross-product.  The window sums of [base, block] and y
            roll rank-one every bar and are rebuilt exactly every 250 solves from Table 1's walk start; the centred Gram
            of [base, scores] is Q' G Q of the rolled centred Gram G, Q = blockdiag(I_132, P) (the Gram route of the
            earlier test, base block 132 columns instead of 12).  k = 0 is BASE_LIN.
  Arms      pca<k>, k in {1, 2, 3, 5, 10, 20}; pls<k>, k in {1, 2, 3} (Amendment A: the earlier test's sets).

GATES (before any scoring; gates.json)
  G1  k = 0 reproduces the ablation's OLS add_rm_fomc pred_adj to 1e-10 relative on every one of its 33,970 bars; the
      target equals Table 1's true_adj.
  G2  on 7 spread days (the earlier test's: quantiles 0, .2, .4, .6, .8, 1 of the forecast days, and 2020-03-16), the
      first and last bar, every arm: the walk's forecast equals a direct computation on the materialized window
      within 1e-8 relative -- the entry rule from scratch (pool_linear_pretune.col_mode), PCA by np.linalg.svd of the
      standardized active block, PLS by NIPALS, scores materialized for the window and the bar, and np.linalg.lstsq of
      the centred target on the centred [base, scores] (the model's free intercept; rcond = sqrt(1e-15), the singular-
      value form of the walk's eigenvalue cut).
  G3  the scored set is Table 1's 33,527 bars (2,579 sessions with all 13 bars; ablate_score.evaluation_days and
      scored); EVENT = the 81 scheduled FOMC statement days (ablate_score.event_days); ORDINARY = the other 2,498.

SCORING   QLIKE (plain = pred_adj^2 B against the bar's rv_actual = data/core_stats.parquet sumret2) and MSE; ORDINARY
          days (primary), every bar, EVENT days; Diebold-Mariano (src/evaluation/diebold_mariano.dm_test: HLN, daily
          mean losses, HAC lag 8, two-sided) against BASE_LIN and against L_full (the lower-QLIKE of Table 1's patched
          full-set ridge and elastic net on the span; FULL_LIN on ORDINARY days); R = (L_BASE - L_arm) / (L_BASE -
          L_full).  FULL_LIN vs BASE_LIN on ORDINARY days is reported next to R, as information only.
READING   (ORDINARY days; the earlier test's rule verbatim) SHARED FACTOR if some k <= 3 (PCA or PLS) has R >= 0.75 and
          its loss is not significantly above FULL_LIN's; DENSE BEYOND A FEW FACTORS if R(PCA 10) < 0.5 and FULL_LIN
          beats BASE_LIN + 10 components significantly; otherwise INTERMEDIATE.  Descriptive: the block's variance shares
          and the first component's loadings (first PLS weights) summed by predictor group.

usage (repository root, the 285J environment):
  python experiments/factor_har_2026_10_26.py design --out D      Table 1's design D13, 12 column parts (one at a time)
  python experiments/beyond_factor_linear.py walk --design D [--maps-out F.npz]
  python experiments/beyond_factor_linear.py gates --design D
  python experiments/beyond_factor_linear.py score
Outputs: results/beyond_trees_2026-10-27/linear/ (forecasts.parquet, factor_days.parquet, walk.json, gates.json,
scores.csv, reading.json, SUMMARY.md).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT), str(ROOT / "notebooks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ablate_arms as AA  # noqa: E402
import factor_har_2026_10_26 as FH  # noqa: E402

OUT = ROOT / "results" / "beyond_trees_2026-10-27" / "linear"
WS_FCST = FH.WS_FCST  # Table 1's patched linear forecasts (window_sweep_2014start_fomcfix/w3000)
BASE_FC = ROOT / "results" / "ablations_2014start_w3000" / "forecasts" / "linear_wide_rmfomc.parquet"
BASE_COL = "ols|add_rm_fomc"
BASE_ARM = "add_rm_fomc"
REMAIN = ("ew", "vw", "liq", "iv", "sent", "vd")
N_BASE, N_BLOCK, N_SERIES = 132, 216, 36
W = FH.W  # 39,000
TUNE_PER = FH.TUNE_PER
FIRST_DAY = FH.FIRST_DAY
PINV_RCOND = FH.PINV_RCOND
STD_MIN, STD_MIN_EVIDENCE = FH.STD_MIN, FH.STD_MIN_EVIDENCE
K_PCA = (1, 2, 3, 5, 10, 20)
K_PLS = (1, 2, 3)
NPC, NPLS = max(K_PCA), max(K_PLS)
ARMS = [f"pca{k}" for k in K_PCA] + [f"pls{k}" for k in K_PLS]
MODELS = ["k0", *ARMS]
G1_TOL, G2_TOL = 1e-10, 1e-8
N_FC, N_SCORED, N_DAYS, N_EVENT = 33_970, 33_527, 2_579, 81
HAC_LAG = 8
SPANS = ("ordinary", "all", "event")
LABEL = "requested by the author after the beyond-base results; not pre-registered"


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ============================================================================ the tree version's construction
def block_names(names: list[str]) -> list[str]:
    """beyond_factor_trees.block_names: the value averages of the six remaining groups' series, in the cache's order."""
    gs = AA.group_series()
    ser = {s for g in REMAIN for s in gs[g]}
    out = []
    for n in names:
        m = re.match(r"^adj_(.+)_ma_(\d+)$", n)
        if m and not n.startswith("har_ma_") and m.group(1) in ser:
            out.append(n)
    assert len(out) == N_BLOCK and len(ser) == N_SERIES, (len(out), len(ser))
    return out


def mode_count(A: np.ndarray) -> np.ndarray:
    """beyond_factor_trees.mode_count: the count of each column's modal value (pool_linear_pretune.col_mode's)."""
    n, p = A.shape
    S = np.sort(A, axis=0)
    new = np.empty((n, p), dtype=bool)
    new[0] = True
    np.not_equal(S[1:], S[:-1], out=new[1:])
    idx = np.arange(n)[:, None]
    run = idx - np.maximum.accumulate(np.where(new, idx, 0), axis=0) + 1
    return run.max(axis=0)


def compute_map(Zw: np.ndarray, yw: np.ndarray, k1: np.ndarray, col_ser: np.ndarray) -> dict:
    """beyond_factor_trees.compute_map with NPC = 20 (the earlier linear test's largest k), plus the descriptive
    vectors of the earlier test (v1: the first eigenvector; r1: the first SIMPLS weight vector, unit norm, signed to a
    non-negative sum); P is computed exactly as the tree version's."""
    m = Zw.mean(axis=0)
    sd = Zw.std(axis=0)
    off = W - mode_count(Zw[:, k1])
    act = (sd > STD_MIN) & (off[col_ser] >= STD_MIN_EVIDENCE)
    a = np.flatnonzero(act)
    Zs = (Zw[:, a] - m[a]) / sd[a]
    C = Zs.T @ Zs / W
    C = 0.5 * (C + C.T)
    lam, V = np.linalg.eigh(C)
    lam, V = lam[::-1], V[:, ::-1]
    V = V[:, :NPC] * np.where(V[:, :NPC].sum(axis=0) < 0, -1.0, 1.0)
    s = Zs.T @ (yw - yw.mean()) / W
    R = FH.simpls(C, s, NPLS)
    P = np.zeros((Zw.shape[1], NPC + NPLS))
    P[a, :NPC] = V / sd[a][:, None]
    P[a, NPC:] = R / sd[a][:, None]
    r1 = R[:, 0] / np.linalg.norm(R[:, 0])
    r1 = r1 * (1.0 if r1.sum() >= 0 else -1.0)
    return dict(m=m, P=P, act=act, lam=lam[:NPC].copy(), trace=float(lam.sum()), n_active=len(a), v1=V[:, 0], r1=r1)


# ============================================================================ design
def load(design: Path) -> dict:
    """The base and block columns of the D13 design parts (factor_har_2026_10_26.py design), in memory."""
    P = FH.Parts(design)
    order = P.order()
    AA.check_full_names(order)
    base = AA.arm_names(BASE_ARM, order)
    Z = block_names(order)
    assert len(base) == N_BASE and not set(base) & set(Z), (len(base), set(base) & set(Z))
    X = P.cols(base + Z)
    d = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(P.parts[0] / "dates_oos.csv", dtype=str)["date"]))
    cfg = P.metas[0]["config"]
    assert int(cfg["window"]) == W and cfg["periods_per_day"] == 13, cfg
    assert X.shape[0] == W + len(d) == len(P.y), (X.shape, len(d), len(P.y))
    zser = [re.match(r"^adj_(.+)_ma_\d+$", z).group(1) for z in Z]  # type: ignore[union-attr]
    ser_list = list(dict.fromkeys(zser))
    k1 = np.array([Z.index(f"adj_{s}_ma_1") for s in ser_list])
    col_ser = np.array([ser_list.index(s) for s in zser])
    sg = FH.series_group(AA.group_series())
    return dict(
        X=X,
        y=np.ascontiguousarray(P.y, dtype=np.float64),
        dates=d,
        base=base,
        Z=Z,
        ser_list=ser_list,
        k1=k1,
        col_ser=col_ser,
        zgrp=np.array([sg[s] for s in zser]),
        design=str(design),
    )


def model_index() -> list[np.ndarray]:
    nb = N_BASE
    ix = [np.arange(nb)]
    ix += [np.r_[np.arange(nb), nb + np.arange(k)] for k in K_PCA]
    ix += [np.r_[np.arange(nb), nb + NPC + np.arange(k)] for k in K_PLS]
    return ix


# ============================================================================ the walk
def cmd_walk(a) -> int:
    t_wall, t_cpu = time.time(), time.process_time()
    D = load(Path(a.design).resolve())
    X, y, dates = D["X"], D["y"], D["dates"]
    n_rows = X.shape[0]
    S0 = FH.walk_start(dates)  # Table 1's walk start (the 250-solve boundary at or before 2014-01-02 10:00)
    f0 = W + int(np.flatnonzero(dates >= FIRST_DAY)[0])  # design row of the first forecast bar
    if a.max_bars:  # a smoke run (--out-dir elsewhere): the first max_bars forecast bars only
        n_rows = f0 + int(a.max_bars)
    nf = n_rows - f0
    assert nf == N_FC or a.max_bars, nf
    fdates = dates[f0 - W : n_rows - W]
    fday = fdates.normalize()
    say(
        f"design {D['design']} rows {n_rows}; walk start row {S0} ({dates[S0 - W]}); forecasts {fdates[0]} .. "
        f"{fdates[-1]} ({nf}); base {N_BASE} + block {N_BLOCK} columns"
    )
    nb, p = N_BASE, N_BLOCK
    ixs = model_index()
    preds = np.empty((nf, len(MODELS)))
    ranks = np.empty((nf, len(MODELS)), dtype=np.int16)
    days: list[dict] = []
    maps: dict[str, list] = {k: [] for k in ("days", "first_row", "m", "P", "act", "lam", "trace", "n_active")}
    Pm = np.zeros((p, NPC + NPLS))
    nq = nb + NPC + NPLS
    Gq, cq, xq = np.empty((nq, nq)), np.empty(nq), np.empty(nq)
    n = float(W)
    S = c = sx = None
    sy = 0.0
    t_map = t_solve = 0.0
    for t in range(S0, n_rows):
        if (t - S0) % TUNE_PER == 0:
            Aw, yw = X[t - W : t], y[t - W : t]
            S, c, sx, sy = Aw.T @ Aw, Aw.T @ yw, Aw.sum(axis=0), float(yw.sum())
        else:
            u, v = X[t - 1], X[t - 1 - W]
            S += np.outer(u, u) - np.outer(v, v)
            c += u * float(y[t - 1]) - v * float(y[t - 1 - W])
            sx += u - v
            sy += float(y[t - 1]) - float(y[t - 1 - W])
        if t < f0:
            continue
        i = t - f0
        if i == 0 or fday[i] != fday[i - 1]:  # the day's first bar: its map from the rows [t - W, t)
            ts = time.perf_counter()
            mp = compute_map(np.array(X[t - W : t, nb:]), y[t - W : t], D["k1"], D["col_ser"])
            Pm = mp["P"]
            rec = dict(day=fday[i], first_row=t, n_active=mp["n_active"], trace=mp["trace"])
            rec.update({f"lam{j + 1}": float(mp["lam"][j]) for j in range(NPC)})
            act = mp["act"]
            for g in REMAIN:
                gm = D["zgrp"][act] == g
                rec[f"n_{g}"] = int(gm.sum())
                rec[f"pc1_abs_{g}"] = float(np.abs(mp["v1"][gm]).sum())
                rec[f"pc1_sq_{g}"] = float((mp["v1"][gm] ** 2).sum())
                rec[f"pls1_abs_{g}"] = float(np.abs(mp["r1"][gm]).sum())
                rec[f"pls1_sq_{g}"] = float((mp["r1"][gm] ** 2).sum())
            days.append(rec)
            for k in ("m", "P", "act", "lam", "trace", "n_active"):
                maps[k].append(mp[k])
            maps["days"].append(str(fday[i].date()))
            maps["first_row"].append(t)
            t_map += time.perf_counter() - ts
        ts = time.perf_counter()
        mu = sx / n
        ybar = sy / n
        Gc = S - n * np.outer(mu, mu)
        cc = c - n * mu * ybar
        dx = X[t] - mu
        Gq[:nb, :nb] = Gc[:nb, :nb]
        Gq[:nb, nb:] = Gc[:nb, nb:] @ Pm
        Gq[nb:, :nb] = Gq[:nb, nb:].T
        Gq[nb:, nb:] = Pm.T @ (Gc[nb:, nb:] @ Pm)
        cq[:nb], cq[nb:] = cc[:nb], Pm.T @ cc[nb:]
        xq[:nb], xq[nb:] = dx[:nb], Pm.T @ dx[nb:]
        for j, ix in enumerate(ixs):
            preds[i, j], ranks[i, j] = FH.minnorm_pred(Gq[np.ix_(ix, ix)], cq[ix], xq[ix], ybar)
        t_solve += time.perf_counter() - ts
        if i % 2500 == 0:
            el = time.time() - t_wall
            say(f"  bar {i:,} / {nf:,} ({fdates[i]}); {len(days)} maps; {el:.0f} s (maps {t_map:.0f} s)")
    sec = time.time() - t_wall
    OUT.mkdir(parents=True, exist_ok=True)
    f = pd.DataFrame({"date": fdates, "design_row": np.arange(f0, n_rows), "true_adj": y[f0:n_rows]})
    for j, mname in enumerate(MODELS):
        f[mname] = preds[:, j]
    f.to_parquet(OUT / "forecasts.parquet", index=False)
    pd.DataFrame(days).to_parquet(OUT / "factor_days.parquet", index=False)
    maps_info: dict = {}
    if a.maps_out:
        mo = Path(a.maps_out)
        mo.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            mo,
            days=np.array(maps["days"]),
            first_row=np.array(maps["first_row"]),
            m=np.array(maps["m"]),
            P=np.array(maps["P"]),
            act=np.array(maps["act"]),
            lam=np.array(maps["lam"]),
            trace=np.array(maps["trace"]),
            n_active=np.array(maps["n_active"]),
        )
        maps_info = dict(maps_npz=str(mo), maps_md5=md5_file(mo))
    # rank: the weekday dummies sum to the intercept (one null direction at least); each arm adds k columns
    r0 = ranks[:, 0].astype(int)
    add_k = bool(all((ranks[:, j].astype(int) == r0 + int(m_[3:])).all() for j, m_ in enumerate(MODELS) if j > 0))
    na = np.array(maps["n_active"])
    info = dict(
        rule="results/beyond_trees_2026-10-27/RULE_linear.txt (Amendment A)",
        label=LABEL,
        design=D["design"],
        design_rows=n_rows,
        walk_start_row=S0,
        walk_start=str(dates[S0 - W]),
        first=str(fdates[0]),
        last=str(fdates[-1]),
        bars=nf,
        maps=len(days),
        base_columns=N_BASE,
        base=D["base"],
        block_columns=p,
        block_series=len(D["ser_list"]),
        block=D["Z"],
        models=MODELS,
        n_active_min=int(na.min()),
        n_active_max=int(na.max()),
        rank_k0_min=int(r0.min()),
        rank_k0_max=int(r0.max()),
        rank_arm_eq_k0_plus_k_every_bar=add_k,
        min_rank={mname: int(ranks[:, j].min()) for j, mname in enumerate(MODELS)},
        max_rank={mname: int(ranks[:, j].max()) for j, mname in enumerate(MODELS)},
        wall_sec=round(sec, 1),
        cpu_sec=round(time.process_time() - t_cpu, 1),
        map_sec=round(t_map, 1),
        solve_sec=round(t_solve, 1),
        host=platform.node(),
        python=platform.python_version(),
        numpy=np.__version__,
        threads="1 (OMP / OpenBLAS / MKL_NUM_THREADS=1)",
        **maps_info,
    )
    (OUT / "walk.json").write_text(json.dumps(info, indent=1))
    say(f"walk done: {json.dumps({k: v for k, v in info.items() if k not in ('base', 'block')})}")
    return 0


# ============================================================================ gates
def templates() -> pd.DataFrame:
    return pd.read_parquet(
        WS_FCST, columns=["model", "predictor_set", "date", "pred_adj", "true_adj", "B", "plain", "rv_actual"]
    )


def base_lin() -> pd.Series:
    b = pd.read_parquet(BASE_FC, columns=["date", BASE_COL])
    return pd.Series(b[BASE_COL].to_numpy(float), index=pd.DatetimeIndex(b["date"]))


def scoring_frame() -> pd.DataFrame:
    """One row per forecast bar (the 33,970): B, rv_actual, true_adj; the plain forecasts of Table 1's patched linear
    templates (ols_har, <m>_full), of BASE_LIN (base_lin) and of the walk's models."""
    T = templates()
    f = pd.read_parquet(OUT / "forecasts.parquet").set_index("date")
    base = T[(T["model"] == "ols") & (T["predictor_set"] == "baseline")].set_index("date").reindex(f.index)
    out = pd.DataFrame({"B": base["B"], "rv_actual": base["rv_actual"], "true_adj": base["true_adj"]}, index=f.index)
    assert np.isfinite(out[["B", "true_adj"]].to_numpy()).all(), "templates cover every forecast bar"
    tag = {"baseline": "har", "all_features": "full"}
    for (m, s), g in T.groupby(["model", "predictor_set"]):
        gi = g.set_index("date").reindex(f.index)
        assert np.array_equal(gi["B"].to_numpy(), out["B"].to_numpy()), (m, s)
        out[f"{m}_{tag[s]}"] = gi["plain"].to_numpy()
    bl = base_lin()
    assert bl.index.equals(f.index), "BASE_LIN rows = the walk's forecast bars"
    out["base_lin"] = bl.to_numpy() ** 2 * out["B"].to_numpy()
    for mname in MODELS:
        out[mname] = f[mname].to_numpy() ** 2 * out["B"].to_numpy()
    return out


def direct_check(D: dict, f: pd.DataFrame, day: pd.Timestamp) -> list[dict]:
    """G2 on one day: every model's forecast of the day's first and last bar, computed directly on the materialized
    window (independent of the walk's map and Gram route)."""
    import pool_linear_pretune as PLP

    X, y = D["X"], D["y"]
    nb = N_BASE
    fd = pd.DatetimeIndex(f["date"]).normalize()
    rows = f["design_row"].to_numpy()[fd == day]
    t0 = int(rows[0])
    Zw = np.array(X[t0 - W : t0, nb:])
    mu, sd = Zw.mean(axis=0), Zw.std(axis=0)
    _, cnt = PLP.col_mode(Zw)  # the entry rule from scratch
    ev_ok = (W - cnt)[D["k1"]][D["col_ser"]] >= STD_MIN_EVIDENCE
    a = np.flatnonzero((sd > STD_MIN) & ev_ok)
    Zs = (Zw[:, a] - mu[a]) / sd[a]
    del Zw
    _, sv, Vt = np.linalg.svd(Zs, full_matrices=False)  # PCA by SVD
    Vp = Vt[:NPC].T
    yw = y[t0 - W : t0]
    Xd, yv = Zs.copy(), yw - yw.mean()  # NIPALS PLS1
    Wl, Pl = [], []
    for _ in range(NPLS):
        w = Xd.T @ yv
        w /= np.linalg.norm(w)
        tt = Xd @ w
        t2 = float(tt @ tt)
        pl = Xd.T @ tt / t2
        yv = yv - float(yv @ tt) / t2 * tt
        Xd -= np.outer(tt, pl)
        Wl.append(w)
        Pl.append(pl)
    del Xd, Zs
    Wm, Pmat = np.array(Wl).T, np.array(Pl).T
    out = []
    for r in (int(rows[0]), int(rows[-1])):
        Zt = (np.asarray(X[r - W : r + 1, nb:])[:, a] - mu[a]) / sd[a]
        Bt = np.asarray(X[r - W : r + 1, :nb])
        yt = y[r - W : r]
        for mname in MODELS:
            if mname == "k0":
                F = np.zeros((W + 1, 0))
            elif mname.startswith("pca"):
                F = Zt @ Vp[:, : int(mname[3:])]
            else:
                k = int(mname[3:])
                F = Zt @ (Wm[:, :k] @ np.linalg.inv(Pmat[:, :k].T @ Wm[:, :k]))
            Dm = np.column_stack([Bt, F])
            Dc = Dm - Dm[:W].mean(axis=0)
            coef = np.linalg.lstsq(Dc[:W], yt - yt.mean(), rcond=float(np.sqrt(PINV_RCOND)))[0]
            direct = float(yt.mean() + Dc[W] @ coef)
            walk = float(f.loc[f["design_row"] == r, mname].iloc[0])
            out.append(
                dict(
                    day=str(day.date()),
                    design_row=r,
                    date=str(D["dates"][r - W]),
                    model=mname,
                    walk=walk,
                    direct=direct,
                    rel=abs(walk - direct) / abs(direct),
                    n_active=len(a),
                    sv20_over_sv21=float(sv[NPC - 1] / sv[NPC]),
                )
            )
    return out


def subset_md5(design: Path) -> dict:
    """md5 of the rebuilt design's arm subsets har, add_rm, add_fomc and add_<g> for the six remaining groups (np.save
    files in F order, streamed) against the Hoffman2 arm caches the ablation ran on (results/ablations_2014start_w3000/
    runs/linear_subsets_fix.csv): which column groups of the rebuild are bit-identical to theirs."""
    P = FH.Parts(design)
    order = P.order()
    ref = pd.read_csv(ROOT / "results" / "ablations_2014start_w3000" / "runs" / "linear_subsets_fix.csv")
    ref = ref.set_index("arm")["x_md5"]
    out: dict = {"reference": "results/ablations_2014start_w3000/runs/linear_subsets_fix.csv"}
    for arm in ["har_sub", "add_rm", "add_fomc", *[f"add_{g}" for g in REMAIN]]:
        names = AA.arm_names("har" if arm == "har_sub" else arm, order)
        hdr = {"descr": "<f8", "fortran_order": True, "shape": (P.n, len(names))}
        got = FH._stream_md5(hdr, (P.X[P.src[nm][0]][:, P.src[nm][1]] for nm in names))
        out[arm] = dict(n_cols=len(names), md5=got, equal=bool(got == ref[arm]))
    return out


def cmd_gates(a) -> int:
    import ablate_score as AS

    rec: dict = {"rule": "results/beyond_trees_2026-10-27/RULE_linear.txt (Amendment A)", "label": LABEL, "fail": []}
    design = Path(a.design).resolve()
    dck = design / "design_check.json"
    if dck.is_file():
        rec["design_check"] = json.loads(dck.read_text())
    rec["design_subsets"] = subset_md5(design)  # information: which column groups are bit-identical to Hoffman2's
    f = pd.read_parquet(OUT / "forecasts.parquet")
    fi = f.set_index("date")
    # G1: k = 0 = the ablation's OLS add_rm_fomc, every bar; the target = Table 1's true_adj
    bl = base_lin()
    T = templates()
    ols = T[(T["model"] == "ols") & (T["predictor_set"] == "baseline")].set_index("date").reindex(fi.index)
    same_rows = bool(fi.index.equals(bl.index))
    smoke = len(fi) != N_FC  # a smoke run (walk --max-bars): G1 on its bars, G2; no G3
    blr = bl.reindex(fi.index).to_numpy()
    rel = np.abs(fi["k0"].to_numpy() - blr) / np.abs(blr)
    rel_y = np.abs(fi["true_adj"].to_numpy() - ols["true_adj"].to_numpy()) / np.abs(ols["true_adj"].to_numpy())
    g1: dict = dict(
        reference=f"{BASE_FC.relative_to(ROOT).as_posix()} [{BASE_COL}]",
        rows=len(fi),
        same_rows=same_rows,
        first=str(fi.index[0]),
        last=str(fi.index[-1]),
        max_rel_pred=float(rel.max()),
        median_rel_pred=float(np.median(rel)),
        n_rel_above_1e_12=int((rel > 1e-12).sum()),
        max_rel_true_adj=float(np.nanmax(rel_y)),
        tol=G1_TOL,
    )
    g1["pass"] = bool(
        same_rows and len(fi) == N_FC and g1["max_rel_pred"] <= G1_TOL and g1["max_rel_true_adj"] <= 1e-14
    )
    rec["G1"] = g1
    if not g1["pass"]:
        rec["fail"].append("G1")
    say(f"G1: max rel {g1['max_rel_pred']:.2e} on {len(fi)} bars")
    # G2: direct computation on materialized windows, 7 spread days, every model
    D = load(design)
    ud = pd.DatetimeIndex(pd.DatetimeIndex(f["date"]).normalize().unique())
    pick = [ud[int(q * (len(ud) - 1))] for q in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    covid = ud[ud >= pd.Timestamp("2020-03-16")]
    pick = sorted(set(pick) | ({covid[0]} if len(covid) else set()))
    rows: list[dict] = []
    for day in pick:
        rows += direct_check(D, f, day)
        say(f"G2 {day.date()}: max rel {max(r['rel'] for r in rows if r['day'] == str(day.date())):.2e}")
    del D
    mx = max(r["rel"] for r in rows)
    fdy = pd.read_parquet(OUT / "factor_days.parquet").set_index("day")["n_active"]
    act_ok = all(int(fdy.loc[pd.Timestamp(r["day"])]) == r["n_active"] for r in rows)
    rec["G2"] = dict(
        days=[str(d.date()) for d in pick],
        models=MODELS,
        checks=rows,
        max_rel=mx,
        max_rel_by_model={m_: max(r["rel"] for r in rows if r["model"] == m_) for m_ in MODELS},
        tol=G2_TOL,
        n_active_walk_eq_direct=act_ok,
        **{"pass": bool(len(pick) >= 5 and mx <= G2_TOL and act_ok)},
    )
    if not rec["G2"]["pass"]:
        rec["fail"].append("G2")
    if smoke:
        rec["fail"].append("G3 (smoke run: not computed)")
        (OUT / "gates.json").write_text(json.dumps(rec, indent=1, default=str))
        say(f"smoke gates: G1 max rel {g1['max_rel_pred']:.2e}; G2 max rel {mx:.2e} {rec['G2']['max_rel_by_model']}")
        return 3
    # G3: the scored set and the EVENT days
    fr = scoring_frame()
    rv = AS.core_rv()
    days = AS.evaluation_days(rv)
    tcols = [c for c in fr.columns if c.endswith("_har") or c.endswith("_full")]
    panel_t = {c: fr[c] for c in tcols}
    panel = dict(panel_t, base_lin=fr["base_lin"], **{mname: fr[mname] for mname in MODELS})
    idx_t = AS.scored(panel_t, rv, days)
    idx = AS.scored(panel, rv, days)
    ev = AS.event_days(days)
    sd_days = pd.DatetimeIndex(idx.normalize())
    bars_per_day = pd.Series(1, index=sd_days).groupby(level=0).size()
    rva = fr["rv_actual"].reindex(idx).to_numpy()
    g3 = dict(
        n_bars=len(idx),
        n_bars_templates=len(idx_t),
        same_as_templates=bool(idx.equals(idx_t)),
        n_days=int(sd_days.nunique()),
        all_days_13_bars=bool((bars_per_day == 13).all()),
        n_event_days=len(ev),
        event_days_eq_earlier_test=bool(ev.equals(FH.event_days())),
        event_days_scored=int(pd.DatetimeIndex(ev).isin(bars_per_day.index).sum()),
        event_bars=int(sd_days.isin(ev).sum()),
        ordinary_days=int(sd_days.nunique()) - len(ev),
        rv_actual_eq_core_sumret2=bool(np.array_equal(rva, rv.reindex(idx).to_numpy())),
        event_days=[str(d.date()) for d in ev],
    )
    g3["pass"] = bool(
        g3["n_bars"] == N_SCORED
        and g3["same_as_templates"]
        and g3["n_days"] == N_DAYS
        and g3["all_days_13_bars"]
        and g3["n_event_days"] == N_EVENT
        and g3["event_days_eq_earlier_test"]
        and g3["event_days_scored"] == N_EVENT
        and g3["event_bars"] == 13 * N_EVENT
        and g3["rv_actual_eq_core_sumret2"]
    )
    rec["G3"] = g3
    if not g3["pass"]:
        rec["fail"].append("G3")
    rec["pass"] = not rec["fail"]
    (OUT / "gates.json").write_text(json.dumps(rec, indent=1, default=str))
    say(
        f"gates: G1 {g1['pass']} (max rel {g1['max_rel_pred']:.2e}), G2 {rec['G2']['pass']} (max rel {mx:.2e}), "
        f"G3 {g3['pass']} ({g3['n_bars']} bars, {g3['n_event_days']} EVENT days); fail {rec['fail']}"
    )
    return 0 if rec["pass"] else 3


# ============================================================================ scores
def cmd_score(a) -> int:
    import ablate_score as AS
    from src.evaluation.diebold_mariano import dm_test

    g = json.loads((OUT / "gates.json").read_text())
    assert g["pass"], f"gates failed: {g['fail']} -- no scoring"
    rv = AS.core_rv()
    days = AS.evaluation_days(rv)
    ev = AS.event_days(days)
    fr = scoring_frame()
    tcols = [c for c in fr.columns if c.endswith("_har") or c.endswith("_full")]
    idx = AS.scored({c: fr[c] for c in [*tcols, "base_lin", *MODELS]}, rv, days)
    assert len(idx) == N_SCORED
    dday = pd.DatetimeIndex(idx.normalize())
    span_ix = {"all": idx, "event": idx[dday.isin(ev)], "ordinary": idx[~dday.isin(ev)]}
    y_rv = fr["rv_actual"]
    rows = []
    info_full: dict = {}

    def loss(col: str, ix: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        pr, yy = fr[col].reindex(ix).to_numpy(float), y_rv.reindex(ix).to_numpy(float)
        r = yy / pr
        return r - np.log(r) - 1.0, (yy - pr) ** 2

    for sp, ix in span_ix.items():
        cols = ["base_lin", "ols_har", "ridge_full", "enet_full", "lasso_full", *MODELS]
        L = {cn: loss(cn, ix) for cn in cols}
        q = {cn: float(v[0].mean()) for cn, v in L.items()}
        full = "ridge_full" if q["ridge_full"] <= q["enet_full"] else "enet_full"
        q_base, q_full = q["base_lin"], q[full]
        mse_base = float(L["base_lin"][1].mean())

        def dm(c1: str, c2: str, li: int) -> dict:
            return dm_test(AS.daily(L[c1][li], ix), AS.daily(L[c2][li], ix), hac_lag=HAC_LAG)

        for cn in [*MODELS, "base_lin", "ols_har", "ridge_full", "enet_full", "lasso_full"]:
            meth = re.match(r"^(pca|pls|k)(\d+)$", cn)
            row: dict = dict(
                model=cn,
                method=("base" if cn == "k0" else meth.group(1)) if meth else cn,
                k=int(meth.group(2)) if meth else None,
                span=sp,
                n_bars=len(ix),
                n_days=int(pd.DatetimeIndex(ix.normalize()).nunique()),
                qlike=q[cn],
                mse_e10=float(L[cn][1].mean()) * 1e10,
                qlike_pct_vs_base=100.0 * (q[cn] / q_base - 1.0),
                mse_pct_vs_base=100.0 * (float(L[cn][1].mean()) / mse_base - 1.0),
                full_ref=full.split("_")[0],
                R=(q_base - q[cn]) / (q_base - q_full),
            )
            for li, ln in ((0, "qlike"), (1, "mse")):
                if cn != "base_lin":
                    r1 = dm(cn, "base_lin", li)
                    row[f"dm_base_{ln}"], row[f"p_base_{ln}"] = float(r1["dm"]), float(r1["p"])
                    row[f"diff_base_{ln}"] = float(r1["mean_diff"])
                if cn != full:
                    r2 = dm(cn, full, li)
                    row[f"dm_full_{ln}"], row[f"p_full_{ln}"] = float(r2["dm"]), float(r2["p"])
                    row[f"diff_full_{ln}"] = float(r2["mean_diff"])
                    row["T_days"] = int(r2["T"])
            rows.append(row)
        rf = next(r for r in rows if r["span"] == sp and r["model"] == full)
        info_full[sp] = dict(
            full_ref=full.split("_")[0],
            qlike_base=q_base,
            qlike_full=q_full,
            pct=rf["qlike_pct_vs_base"],
            dm=rf["dm_base_qlike"],
            p=rf["p_base_qlike"],
            full_beats_base_significantly=bool(rf["p_base_qlike"] < 0.05 and rf["diff_base_qlike"] < 0),
        )
    sc = pd.DataFrame(rows)
    o = sc[sc["span"] == "ordinary"].set_index("model")
    small = [f"pca{k}" for k in K_PCA if k <= 3] + [f"pls{k}" for k in K_PLS if k <= 3]
    shared = [
        m_
        for m_ in small
        if o.loc[m_, "R"] >= 0.75 and not (o.loc[m_, "p_full_qlike"] < 0.05 and o.loc[m_, "diff_full_qlike"] > 0)
    ]
    r10 = float(o.loc["pca10", "R"])
    full_beats_10 = bool(o.loc["pca10", "p_full_qlike"] < 0.05 and o.loc["pca10", "diff_full_qlike"] > 0)
    dense = bool(r10 < 0.5 and full_beats_10)
    word = "SHARED FACTOR" if shared else ("DENSE BEYOND A FEW FACTORS" if dense else "INTERMEDIATE")
    reading = dict(
        reading=word,
        shared_factor_models=shared,
        shared_condition_met=bool(shared),
        dense_condition_met=dense,
        R10=r10,
        full_beats_pca10_significantly=full_beats_10,
        full_ref_ordinary=str(o.loc["k0", "full_ref"]),
        R_ordinary={m_: float(o.loc[m_, "R"]) for m_ in MODELS},
        best_small_k=max(small, key=lambda m_: float(o.loc[m_, "R"])),
        full_vs_base=info_full,
        k0_qlike_rel_to_base_lin=float(o.loc["k0", "qlike"] / o.loc["base_lin", "qlike"] - 1.0),
    )
    sc.to_csv(OUT / "scores.csv", index=False, float_format="%.17g")
    desc = descriptive(days)
    (OUT / "reading.json").write_text(
        json.dumps(dict(label=LABEL, reading=reading, descriptive=desc), indent=1, default=str)
    )
    summary(sc, reading, desc)
    say(f"score: {word}; R (ordinary) {reading['R_ordinary']}; FULL_LIN vs BASE_LIN (ordinary) {info_full['ordinary']}")
    return 0


def descriptive(days: pd.DatetimeIndex) -> dict:
    fd = pd.read_parquet(OUT / "factor_days.parquet")
    fd = fd[pd.DatetimeIndex(fd["day"]).isin(days)]
    lam = fd[[f"lam{j}" for j in range(1, NPC + 1)]].to_numpy()
    share = lam / fd["n_active"].to_numpy()[:, None]
    cum = np.cumsum(share, axis=1)
    return dict(
        n_days=len(fd),
        n_active_min=int(fd["n_active"].min()),
        n_active_max=int(fd["n_active"].max()),
        share_pc={str(j + 1): float(share[:, j].mean()) for j in range(5)},
        share_cum={str(k): float(cum[:, k - 1].mean()) for k in K_PCA},
        share_cum3_min=float(cum[:, 2].min()),
        share_cum3_max=float(cum[:, 2].max()),
        lam_mean={str(j + 1): float(lam[:, j].mean()) for j in range(5)},
        n_cols={g: float(fd[f"n_{g}"].mean()) for g in REMAIN},
        pc1_abs={g: float(fd[f"pc1_abs_{g}"].mean()) for g in REMAIN},
        pc1_sq={g: float(fd[f"pc1_sq_{g}"].mean()) for g in REMAIN},
        pls1_abs={g: float(fd[f"pls1_abs_{g}"].mean()) for g in REMAIN},
        pls1_sq={g: float(fd[f"pls1_sq_{g}"].mean()) for g in REMAIN},
    )


def summary(sc: pd.DataFrame, r: dict, d: dict) -> None:
    g = json.loads((OUT / "gates.json").read_text())
    wk = json.loads((OUT / "walk.json").read_text())
    dck = g.get("design_check", {})
    fo = r["full_vs_base"]["ordinary"]

    def f_(v, nd=2):
        return "--" if v is None or pd.isna(v) else f"{v:.{nd}f}"

    L = [
        "# Linear factor test beyond HAR + realized moments + FOMC",
        "",
        f"*{LABEL[0].upper() + LABEL[1:]}.*  Rule: `results/beyond_trees_2026-10-27/RULE_linear.txt` (163eecb) with"
        " Amendment A (027bf5f), both committed before any computation.  Code: `experiments/beyond_factor_linear.py`"
        " (walk, gates, score); generated by its score mode.",
        "",
        f"## Reading (ORDINARY days): **{r['reading']}**",
        "",
        f"- SHARED FACTOR condition (some k <= 3, PCA or PLS, R >= 0.75 and not significantly worse than FULL_LIN):"
        f" {'met by ' + ', '.join(r['shared_factor_models']) if r['shared_factor_models'] else 'not met'}; best k <= 3:"
        f" {r['best_small_k']} (R = {r['R_ordinary'][r['best_small_k']]:.3f}).",
        f"- DENSE condition (R(PCA 10) < 0.5 and FULL_LIN significantly better than BASE_LIN + 10 components):"
        f" R = {r['R10']:.3f}, FULL_LIN significantly better: {r['full_beats_pca10_significantly']}"
        f" -> {'met' if r['dense_condition_met'] else 'not met'}.",
        f"- FULL_LIN on ORDINARY days: Table 1's patched full-set {r['full_ref_ordinary']} (the lower of ridge and"
        " elastic net).",
        f"- Information (not a reading condition): FULL_LIN vs BASE_LIN on ORDINARY days {fo['pct']:+.2f}% QLIKE,"
        f" DM {fo['dm']:.2f}, p {fo['p']:.3f} (significantly better: {fo['full_beats_base_significantly']}); R's"
        f" denominator L_BASE - L_FULL = {fo['qlike_base'] - fo['qlike_full']:.2e}.",
        *[
            f"- {({'all': 'Every bar', 'event': 'EVENT days'})[sp_]}: the full set ({v['full_ref']}) is"
            f" {'above' if v['pct'] > 0 else 'below'} BASE_LIN, {v['pct']:+.2f}% QLIKE, DM {v['dm']:.2f}, p {v['p']:.3f}"
            + (
                " -- R's denominator is negative there, so R is not a recovered share on that span."
                if v["pct"] > 0
                else "."
            )
            for sp_, v in r["full_vs_base"].items()
            if sp_ != "ordinary"
        ],
        "- k0 equals BASE_LIN to the G1 tolerance; its DM against BASE_LIN is computed on loss differences of rounding"
        " size (~1e-12) and carries no information.",
        "",
        "## Scores",
        "",
        "QLIKE change in percent against BASE_LIN (the ablation's OLS add_rm_fomc), DM (HLN, daily means, lag 8) against"
        " BASE_LIN and against the full set (negative: the row's loss is lower), R = (L_BASE - L_arm) / (L_BASE -"
        " L_full).",
        "",
    ]
    for sp in SPANS:
        s = sc[sc["span"] == sp].set_index("model")
        L += [
            f"### {sp} ({int(s['n_days'].iloc[0])} days, {int(s['n_bars'].iloc[0])} bars; full = {s['full_ref'].iloc[0]})",
            "",
            "| model | QLIKE | % vs base | DM vs base | p | DM vs full | p | R | MSE % vs base | DM MSE vs base | p |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for m_ in [*MODELS, "base_lin", "ridge_full", "enet_full", "lasso_full", "ols_har"]:
            x = s.loc[m_]
            L.append(
                f"| {m_} | {x['qlike']:.5f} | {x['qlike_pct_vs_base']:.2f} | {f_(x.get('dm_base_qlike'))} |"
                f" {f_(x.get('p_base_qlike'), 3)} | {f_(x.get('dm_full_qlike'))} | {f_(x.get('p_full_qlike'), 3)} |"
                f" {x['R']:.3f} | {x['mse_pct_vs_base']:.2f} | {f_(x.get('dm_base_mse'))} | {f_(x.get('p_base_mse'), 3)} |"
            )
        L.append("")
    L += [
        f"## Descriptive ({d['n_days']:,} scored days; the day's map)",
        "",
        f"- Active block columns a day: {d['n_active_min']} .. {d['n_active_max']} of {N_BLOCK}.",
        "- Mean share of the block's variance (eigenvalue / active columns): "
        + ", ".join(f"PC{k} {100 * v:.1f}%" for k, v in d["share_pc"].items())
        + "; cumulative "
        + ", ".join(f"1-{k} {100 * v:.1f}%" for k, v in d["share_cum"].items())
        + f" (components 1-3: {100 * d['share_cum3_min']:.1f}% .. {100 * d['share_cum3_max']:.1f}% across days).",
        "",
        "| group | columns (mean) | PC1 sum abs loading | PC1 share of squared loadings | PLS1 sum abs weight |"
        " PLS1 share of squared weights |",
        "|---|---|---|---|---|---|",
    ]
    for gr in REMAIN:
        L.append(
            f"| {AA.GROUP_LABEL[gr]} | {d['n_cols'][gr]:.1f} | {d['pc1_abs'][gr]:.3f} | {100 * d['pc1_sq'][gr]:.1f}% |"
            f" {d['pls1_abs'][gr]:.3f} | {100 * d['pls1_sq'][gr]:.1f}% |"
        )
    g1, g2, g3 = g["G1"], g["G2"], g["G3"]
    L += [
        "",
        "Loadings: the unit-norm eigenvector of the standardized block's correlation matrix (PC1) and the unit-norm"
        " first SIMPLS weight vector (PLS1), summed by predictor group, averaged over days.",
        "",
        "## Gates (gates.json)",
        "",
        f"- G1 {'PASS' if g1['pass'] else 'FAIL'}: k = 0 vs the ablation's OLS add_rm_fomc ({g1['reference']}),"
        f" {g1['rows']:,} bars, max relative difference {g1['max_rel_pred']:.2e} (tolerance 1e-10); target max"
        f" relative {g1['max_rel_true_adj']:.1e}.",
        f"- G2 {'PASS' if g2['pass'] else 'FAIL'}: every model vs a direct computation on the materialized window (SVD"
        f" PCA, NIPALS PLS, the entry rule from scratch, centred lstsq), {len(g2['checks'])} checks on"
        f" {', '.join(g2['days'])}; max relative difference {g2['max_rel']:.2e} (tolerance 1e-8); active sets equal:"
        f" {g2['n_active_walk_eq_direct']}.",
        f"- G3 {'PASS' if g3['pass'] else 'FAIL'}: {g3['n_bars']:,} scored bars on {g3['n_days']:,} days (= Table 1's),"
        f" {g3['n_event_days']} EVENT days ({g3['event_bars']} bars), all scored; {g3['ordinary_days']:,} ORDINARY days.",
        "",
        "## Provenance",
        "",
        f"- Design: Table 1's D13 rebuilt on the laptop in 12 column parts ({dck.get('design', '?')};"
        f" `{dck.get('command', '?')}`); md5 against the CARC patched build: {dck.get('md5_equal')}; the 12 HAR-set"
        f" columns md5-equal to Table 1's baseline X: {dck.get('har12_md5_eq_carc_baseline_X')}. Arm subsets against the"
        " Hoffman2 caches the ablation ran on (linear_subsets_fix.csv), bit-identical: "
        + ", ".join(f"{k} {v['equal']}" for k, v in g.get("design_subsets", {}).items() if isinstance(v, dict))
        + ". G1 bounds the effect of the last-bit differences on BASE_LIN; the block's columns differ from the CARC"
        " build in last bits too (no per-column reference here), immaterial at G2's and the scores' precision.",
        f"- Walk: {wk['bars']:,} bars {wk['first']} .. {wk['last']} (window sums from Table 1's walk start, row"
        f" {wk['walk_start_row']}, {wk['walk_start']}), {wk['maps']:,} daily maps computed on the materialized window;"
        f" {wk['wall_sec']:.0f} s wall ({wk['map_sec']:.0f} s maps, {wk['solve_sec']:.0f} s solves) on {wk['host']}"
        f" (numpy {wk['numpy']}, one thread). Rank: BASE_LIN {wk['rank_k0_min']} .. {wk['rank_k0_max']} of"
        f" {N_BASE} (the 60 availability averages of the realized-moment and FOMC series are constant in the window;"
        " the weekday dummies sum to the intercept; adj_fomc_day_ma_1 and adj_fomc_day_ma_5 are collinear; checked on"
        f" the last window; min-norm as the ablation's walk); every arm = BASE_LIN's rank + k at every bar: {wk['rank_arm_eq_k0_plus_k_every_bar']}.",
        f"- Maps: {wk.get('maps_npz', '(not stored)')} (md5 {wk.get('maps_md5', '-')}).",
        "- Templates: results/window_sweep_2014start_fomcfix/w3000/forecasts.parquet (B, rv_actual; full-set ridge,"
        " elastic net, lasso on the patched design; OLS-HAR for reference).",
        "- As in the earlier test: the full sets are penalized and include every availability average; BASE_LIN + k"
        " components is OLS on value averages of the block. The comparison is of forecast accuracy, not a nesting.",
        "",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("walk", "gates", "score"))
    ap.add_argument("--design", default="")
    ap.add_argument("--maps-out", default="")
    ap.add_argument("--max-bars", type=int, default=0, help="smoke run: the first N forecast bars (with --out-dir)")
    ap.add_argument("--out-dir", default="", help="output directory (default results/beyond_trees_2026-10-27/linear)")
    a = ap.parse_args()
    global OUT
    if a.out_dir:
        OUT = Path(a.out_dir).resolve()
    assert not a.max_bars or a.out_dir, "a smoke run writes elsewhere (--out-dir)"
    return {"walk": cmd_walk, "gates": cmd_gates, "score": cmd_score}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
