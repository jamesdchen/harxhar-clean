"""Variable importance of the POOLED intraday linear models (lasso, ridge; all features) at a monthly re-estimation
cadence (note E19, writeup/rehaul/08_feature_importance.tex:49).

Author's decisions (2026-10-19):
  * one re-estimation at the first trading day of each calendar month, scored on the month it forecasts;
  * both E15 samples: all 48 intervals (window 96,000 intervals) and the 13 regular-hours bars, bar-end labels
    10:00 .. 16:00 (window 26,000 intervals); 2,000 trading days in both;
  * span: the evaluation span of Table tab:rh_oos (\\rhoosSpanStart .. \\rhoosSpanEnd);
  * penalties: the frozen pre-sample picks of the pooled linear reruns (frozen_penalties.json of each sample).

The measures are the closing-interval analysis's (experiments/rehaul_fi_compute.py, linear stage and reducer;
formulas: writeup/rehaul/08_feature_importance.tex:51-325): lasso selection counts K_A, K_j, the HAR benchmark
block's share of sum |beta|, ||beta_{k,A}||_1 and the on/off indicator at each re-estimation, Kendall's W; ridge
shares S_A, the standardized share S^sd_A, S_{k,A}, Kendall's W; the window s.d. by predictor kind; the random probes
at every third re-estimation.

Estimator at each re-estimation k: the batch solution of the pooled model on the W sample rows before the first row
of the month's first trading day t_k, at the frozen alpha, with the penalty weight r on the HAR benchmark block
(the HAR components and the calendar dummies) and 1 on every exogenous predictor, the intercept unpenalized, and
the identifiability mask of the spec (constant or byte-duplicate columns in the window get a zero coefficient).
This is the pooled walk's cold seed (experiments/pool_linear_pretune.py: batch_theta_pf, the ridge solve of _seed),
the coefficients the walk starts from at a re-estimation; between re-estimations the walk re-solves every interval,
this analysis holds the month's fit.

Stages:  python experiments/rehaul_fi_pooled_compute.py <stage> ...
  design   --out D [--bucket all_features]   the pooled design cache (X, y, column names, the label of EVERY row);
           the design flags come from the environment (HAR_BASE etc., read by specs/causal_tune_linear.py)
  linear   --design D --sample 48|13 --penalties F --out O [--start --end]   coefficient paths, window scales,
           probes -> O/linear.npz, O/run.json
  reduce   --out O                           -> O/*.csv, O/summary.json
Env: RHFP_THREADS (default 4) BLAS threads.
"""

from __future__ import annotations

import os

_T = os.environ.get("RHFP_THREADS", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = _T
os.environ["NUMBA_NUM_THREADS"] = _T

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TW_DAYS = 2000
SAMPLES = {"48": 48, "13": 13}  # intervals of a trading day in the sample
RTH = {f"{h:02d}:{m:02d}" for h in range(10, 17) for m in (0, 30)} - {"16:30"}  # the 13 regular-hours bar-end labels
SPAN = ("2020-01-03", "2024-04-30")  # \rhoosSpanStart .. \rhoosSpanEnd (writeup/generated/rehaul_oos_numbers.tex)
MODELS = ("lasso", "ridge")
EST = {"lasso": ("reclasso", 1.0), "ridge": ("ridge", 0.0)}  # spec estimator, l1 ratio
NZ_TOL = 0.0  # nonzero: |beta| > 0 (the homotopy zeroes exactly)
_HAR = re.compile(r"har_ma_\d+(_x_(open|close))?$")
_EXOG = re.compile(r"^(?:adj_)?(.+?)(?:_(avail|active))?_ma_\d+$")


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ design cache
def design(a) -> None:
    """experiments/pool_linear_pretune.py design() plus the date label of every design row (the 13-interval
    sample needs the labels of the window rows, which that cache does not keep)."""
    import pool_linear_pretune as PLP
    import src.backtest.executor as EX

    cap: dict = {}
    orig = EX.apply_horizon_shift

    def spy(X, y, dates, baselines, horizon):
        out = orig(X, y, dates, baselines, horizon)
        cap["dates"] = pd.Series(out[2]).astype(str).to_numpy()
        return out

    EX.apply_horizon_shift = spy
    try:
        PLP.design(argparse.Namespace(bucket=a.bucket, data=a.data, out=a.out))
    finally:
        EX.apply_horizon_shift = orig
    out = Path(a.out)
    n = len(np.load(out / "y.npy", mmap_mode="r"))
    assert len(cap["dates"]) == n, (len(cap["dates"]), n)
    oos = pd.read_csv(out / "dates_oos.csv", dtype=str)["date"].to_numpy()
    assert np.array_equal(cap["dates"][n - len(oos) :], oos)
    pd.DataFrame({"date": cap["dates"]}).to_csv(out / "dates_all.csv", index=False)
    meta = json.loads((out / "design.json").read_text())
    meta["env"] = {k: v for k, v in os.environ.items() if k.startswith(("HAR_", "HPC_KW_"))}
    (out / "design.json").write_text(json.dumps(meta))
    say(f"design: {n} rows, {len(meta['names'])} columns, {cap['dates'][0]} .. {cap['dates'][-1]}")


# ============================================================================ groups
def group_of(name: str, series: set[str]) -> str:
    """The predictor group of a column: one group per exogenous series (its averages at every horizon, its
    missing-data and activity indicators), the HAR components, the calendar dummies."""
    if _HAR.fullmatch(name):
        return "har_ma"
    m = _EXOG.fullmatch(name)
    if m and m.group(1) in series:
        return m.group(1)
    assert re.fullmatch(r"DOW_\d|hour|is_\w+|days_to_opex", name), f"no predictor group for {name}"
    return "calendar"


def kind_of(name: str) -> str:
    """Predictor kind for the scale check (rehaul_fi_compute.reduce's rule)."""
    if re.fullmatch(r"har_ma_\d+", name):
        return "har"
    if re.fullmatch(r"adj_.+_ma_\d+", name):
        return "value"
    if re.fullmatch(r".+_(avail|active)_ma_\d+", name):
        return "flag"
    return "calendar"


def groups(names: list[str]) -> dict:
    from src.data import loading as LD

    bmap: dict = {}
    for b, ss in LD.SUBGROUPS.items():
        if b in ("baseline", "all_features") or b not in (
            "moments", "liquidity", "market_ew", "market_vw", "sentiment", "implied_vol", "vol_demand", "fomc"
        ):
            continue
        for s in ss:
            bmap[s] = b
    gcol = [group_of(n, set(bmap)) for n in names]
    order = ["har_ma", "calendar"] + sorted(
        {g for g in gcol if g not in ("har_ma", "calendar")},
        key=lambda s: (list(bmap.values()).index(bmap[s]), LD.SUBGROUPS[bmap[s]].index(s)),
    )
    order = [g for g in order if g in gcol]
    bmap.update(har_ma="backbone", calendar="backbone")
    G = np.zeros((len(names), len(order)))
    G[np.arange(len(names)), [order.index(g) for g in gcol]] = 1.0
    return dict(units=order, gcol=gcol, bmap=bmap, G=G, kind=[kind_of(n) for n in names])


# ============================================================================ penalties
def load_penalties(path: Path) -> dict:
    """{lasso, ridge: alpha; r: the penalty weight on the HAR benchmark block; columns; n_bb} of the all-features
    design from a pooled rerun's frozen_penalties.json ({model: {predictor set: alpha}, designs: {predictor set:
    {columns, n_har_benchmark_block, penalty_factor_on_har_benchmark_block}}}); the tiny case's flat form
    {ridge: {alpha}, lasso: {alpha}, r, columns} is accepted too."""
    z = json.loads(path.read_text())
    if "designs" in z:
        d = z["designs"]["all_features"]
        return dict(
            lasso=float(z["lasso"]["all_features"]), ridge=float(z["ridge"]["all_features"]),
            r=float(d["penalty_factor_on_har_benchmark_block"]), columns=list(d["columns"]),
            n_bb=int(d["n_har_benchmark_block"]), file=rel(path), info=z.get("info", {}),
        )
    return dict(lasso=float(z["lasso"]["alpha"]), ridge=float(z["ridge"]["alpha"]), r=float(z["r"]),
                columns=z.get("columns"), n_bb=None, file=rel(path), info={})


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


HAR_BASE5 = {f"har_ma_{k}" for k in (1, 5, 25, 125, 625, 3125)}


def check_landed(pen: dict) -> None:
    """Author's rule (2026-10-19, 22:45 UTC): a frozen_penalties.json counts as landed only if its all-features
    column list has HAR components exactly har_ma_{1,5,25,125,625,3125} (base 5) and contains the FOMC series."""
    from src.data.loading import FOMC_FEATURES

    cols = pen["columns"] or []
    har = {c for c in cols if _HAR.fullmatch(c) and "_x_" not in c}
    assert har == HAR_BASE5, f"{pen['file']}: HAR columns {sorted(har)} are not the base-5 ladder"
    fomc = [s for s in FOMC_FEATURES if not any(_EXOG.fullmatch(c) and _EXOG.fullmatch(c).group(1) == s for c in cols)]
    assert not fomc, f"{pen['file']}: FOMC series missing from the column list: {fomc}"


def row_labels(dz: Path, n: int) -> pd.Series:
    """The date label of every design row: dates_all.csv (this script's design stage), else the runner cache's
    dates_oos.csv, which labels the rows from W on (the earlier rows get "")."""
    if (dz / "dates_all.csv").is_file():
        return pd.read_csv(dz / "dates_all.csv", dtype=str)["date"]
    oos = pd.read_csv(dz / "dates_oos.csv", dtype=str)["date"].to_numpy()
    return pd.Series(np.r_[np.full(n - len(oos), ""), oos])


# ============================================================================ linear
def anchors(dates: pd.Series, rows: np.ndarray, start: str, end: str) -> tuple[np.ndarray, np.ndarray, list]:
    """Re-estimation k: the first sample row of the first trading day of each calendar month in [start, end];
    its block O_k: the sample rows from there to the next re-estimation (or to the end of the span)."""
    d = pd.to_datetime(dates.iloc[rows].str[:10]).to_numpy()
    inspan = (d >= np.datetime64(start)) & (d <= np.datetime64(end))
    idx = np.flatnonzero(inspan)
    days = pd.DatetimeIndex(d[idx])
    month = days.year * 12 + days.month
    first = np.r_[True, month[1:] != month[:-1]]
    a = idx[first]  # positions in `rows`
    b = np.r_[a[1:], idx[-1] + 1]
    return a, b, [str(pd.Timestamp(x).date()) for x in d[a]]


def solve(ns: dict, X: np.ndarray, y: np.ndarray, model: str, alpha: float, pf: np.ndarray) -> np.ndarray:
    """The pooled walk's cold seed at a fixed alpha: X carries the intercept as its last column (locked)."""
    import pool_linear_pretune as PLP

    locked = np.zeros(X.shape[1], bool)
    locked[-1] = True
    if model == "ridge":
        G = X.T @ X
        G[np.diag_indices_from(G)] += np.where(locked, 0.0, alpha * pf)
        return np.linalg.solve(G, X.T @ y)
    return PLP.batch_theta_pf(ns, X, y, locked, alpha, EST[model][1], pf)


def linear(a) -> None:
    import pool_linear_pretune as PLP
    import feature_importance_1530_trees as FT

    t_wall = time.time()
    dz, out = Path(a.design), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((dz / "design.json").read_text())
    names_all = list(meta["names"])
    pen = load_penalties(Path(a.penalties))
    if not a.no_check:
        check_landed(pen)
    if pen["columns"] is not None:
        miss = sorted(set(pen["columns"]) - set(names_all))
        assert not miss, f"penalty-file columns missing from the design: {miss[:10]}"
        cidx = np.array([names_all.index(c) for c in pen["columns"]])
    else:
        cidx = np.arange(len(names_all))
    names = [names_all[j] for j in cidx]
    gr = groups(names)
    bb = np.array([gr["bmap"][g] == "backbone" for g in gr["gcol"]])
    assert pen["n_bb"] in (None, int(bb.sum())), (pen["n_bb"], int(bb.sum()))
    X = np.load(dz / "X.npy", mmap_mode="r")
    y = np.load(dz / "y.npy")
    dates = row_labels(dz, len(y))
    per = SAMPLES[a.sample]
    lab = dates[dates != ""]
    if per == 48 or lab.str[11:16].isin(RTH).all():  # all rows, or a design of the 13 bars only
        rows = np.arange(len(y))
    else:
        rows = np.flatnonzero(dates.str[11:16].isin(RTH).to_numpy())
    W = a.tw_days * per
    A, B, adates = anchors(dates, rows, a.start, a.end)
    assert A[0] >= W and (dates.iloc[rows[A[0] - W :]] != "").all(), (A[0], W)
    K = len(A)
    p = len(names)
    say(f"sample {a.sample}: {len(rows)} rows, W {W}, p {p}, {len(gr['units'])} groups, K {K} "
        f"({adates[0]} .. {adates[-1]}), lasso {pen['lasso']}, ridge {pen['ridge']}, r {pen['r']}")
    ns = PLP.spec_section("reclasso", "all_features", 0, -1, 0)
    os.chdir(REPO)
    probe = ns["RollingTunedLinear"]()
    pf = np.ones(p + 1)
    pf[:-1][bb] = pen["r"]
    theta = {m: np.zeros((K, p + 1)) for m in MODELS}
    mse = {m: np.zeros(K) for m in MODELS}
    pk = [k for k in range(K) if k % FT.PROBE_EVERY == 0]
    probe_theta = {m: np.zeros((len(pk), p + 4)) for m in MODELS}
    sd, iqr = np.zeros((K, p)), np.zeros((K, p))
    masked = np.zeros((K, p), bool)
    sec = {m: [] for m in MODELS}
    sec_probe = []
    for k in range(K):
        t0 = time.time()
        win = rows[A[k] - W : A[k]]
        blk = rows[A[k] : B[k]]
        Xw = np.empty((W, p + 1))
        Xw[:, :-1] = X[win][:, cidx]
        Xw[:, -1] = 1.0
        yw = y[win]
        probe._locked = np.r_[np.zeros(p, bool), True]
        probe._recompute_mask(Xw)  # the spec's identifiability mask (constant / byte-duplicate columns)
        mk = probe._maskout
        masked[k] = mk[:-1]
        sd[k] = Xw[:, :-1].std(axis=0)
        q75, q25 = np.percentile(Xw[:, :-1], [75, 25], axis=0)
        iqr[k] = q75 - q25
        Xm = Xw.copy()
        Xm[:, mk] = 0.0
        Xb = np.hstack([X[blk][:, cidx], np.ones((len(blk), 1))])
        Xb[:, mk] = 0.0
        for m in MODELS:
            t1 = time.time()
            th = solve(ns, Xm, yw, m, pen[m], pf)
            sec[m].append(time.time() - t1)
            theta[m][k] = th
            mse[m][k] = float(np.mean((Xb @ th - y[blk]) ** 2))
        if k % FT.PROBE_EVERY == 0:
            # three noise predictors (rehaul_fi_compute.linear: continuous, PROBE_LEVELS-valued, binary), robust-
            # scaled on the window (median / IQR), exogenous (weight 1), not masked; inserted before the intercept
            t1 = time.time()
            q = pk.index(k)
            rng = np.random.default_rng([FT.PROBE_SEED, k])
            noise = np.column_stack(
                [rng.random(W), np.floor(rng.random(W) * FT.PROBE_LEVELS), (rng.random(W) < 0.5).astype(np.float64)]
            )
            med = np.median(noise, axis=0)
            n75, n25 = np.percentile(noise, [75, 25], axis=0)
            noise = (noise - med) / np.where(n75 > n25, n75 - n25, 1.0)
            Xp = np.hstack([Xm[:, :-1], noise, Xm[:, -1:]])
            pfp = np.r_[pf[:-1], np.ones(3), pf[-1:]]
            for m in MODELS:
                probe_theta[m][q] = solve(ns, Xp, yw, m, pen[m], pfp)
            sec_probe.append(time.time() - t1)
        say(f"k {k:2d} {adates[k]}: masked {int(mk.sum())}, lasso nz {int((theta['lasso'][k, :-1] != 0).sum())}, "
            f"block {len(blk)} rows, mse lasso {mse['lasso'][k]:.4f} ridge {mse['ridge'][k]:.4f}, "
            f"{time.time() - t0:.1f}s")
    np.savez_compressed(
        out / "linear.npz",
        **{f"{m}_theta": theta[m] for m in MODELS},
        **{f"{m}_block_mse": mse[m] for m in MODELS},
        **{f"{m}_probe_theta": probe_theta[m] for m in MODELS},
        probe_k=np.array(pk),
        win_sd=sd,
        win_iqr=iqr,
        masked=masked,
        anchor_row=rows[A],
        block_rows=B - A,
    )
    info = dict(
        sample=a.sample, window=W, p=p, K=K, start=a.start, end=a.end, anchors=adates, names=names,
        penalties=pen | dict(columns=None, info=None), pen_info=pen["info"], design=str(dz), design_env=meta.get("env", {}),
        design_shape=meta["shape"], probe_levels=int(FT.PROBE_LEVELS), probe_every=int(FT.PROBE_EVERY),
        sec_fit={m: dict(mean=float(np.mean(sec[m])), total=float(np.sum(sec[m]))) for m in MODELS},
        sec_probe=dict(mean=float(np.mean(sec_probe)), total=float(np.sum(sec_probe))),
        wall_sec=time.time() - t_wall, threads=_T, repair=dict(PLP.REPAIR), host=platform.node(),
        numpy=np.__version__,
    )
    (out / "run.json").write_text(json.dumps(info, indent=1))
    say(f"linear done: {info['wall_sec']:.0f}s")


def gate(a) -> None:
    """The batch fit of linear() at the frozen penalties == the pooled runner's walk forecast at its cold seeds
    (rows S = --row + 250 m, the runner's re-seed rows).  The walk forecasts come from the rerun's committed
    forecasts.parquet (--forecasts; rows matched by date label, predictor set all_features) or from runner walk
    output directories (--walks lasso=DIR,ridge=DIR; their first row is --row).  Writes gates.csv."""
    import pool_linear_pretune as PLP

    dz, out = Path(a.design), Path(a.out)
    pen = load_penalties(Path(a.penalties))
    if not a.no_check:
        check_landed(pen)
    names_all = list(json.loads((dz / "design.json").read_text())["names"])
    cidx = np.array([names_all.index(c) for c in pen["columns"]]) if pen["columns"] else np.arange(len(names_all))
    names = [names_all[j] for j in cidx]
    gr = groups(names)
    bb = np.array([gr["bmap"][g] == "backbone" for g in gr["gcol"]])
    X = np.load(dz / "X.npy", mmap_mode="r")
    y = np.load(dz / "y.npy")
    W, p = a.tw_days * SAMPLES[a.sample], len(names)
    ns = PLP.spec_section("reclasso", "all_features", 0, -1, 0)
    os.chdir(REPO)
    probe = ns["RollingTunedLinear"]()
    probe._locked = np.r_[np.zeros(p, bool), True]
    pf = np.ones(p + 1)
    pf[:-1][bb] = pen["r"]
    walk: dict = {m: {} for m in MODELS}  # model -> {row: walk forecast}
    if a.forecasts:
        lab = row_labels(dz, len(y))
        row_of = {d: i for i, d in enumerate(lab) if d}
        fc = pd.read_parquet(a.forecasts)
        for m in MODELS:
            f = fc[(fc["model"] == m) & (fc["predictor_set"] == "all_features")]
            assert np.allclose(f["alpha"], pen[m]), (m, f["alpha"].unique(), pen[m])
            walk[m] = {row_of[str(d)]: float(v) for d, v in zip(f["date"].astype(str), f["pred_adj"])}
    else:
        for kv in a.walks.split(","):
            m, d = kv.split("=")
            f = next(Path(d).glob("causal_tune_linear/*/*/results*.csv"))
            walk[m] = {a.row: float(pd.read_csv(f)["pred_adj"].iloc[0])}
    seeds = [r for r in range(a.row, len(y), 250) if all(r in walk[m] for m in MODELS)]
    rows = []
    for S0 in seeds:
        Xw = np.hstack([X[S0 - W : S0][:, cidx], np.ones((W, 1))])
        probe._recompute_mask(Xw)
        Xw[:, probe._maskout] = 0.0
        xs = np.r_[X[S0][cidx], 1.0]
        xs[probe._maskout] = 0.0
        for m in MODELS:
            th = solve(ns, Xw, y[S0 - W : S0], m, pen[m], pf)
            gap = abs(float(xs @ th) - walk[m][S0])
            rows.append(dict(model=m, row=S0, walk=walk[m][S0], gap=gap, tol=a.tol,
                             ok=gap <= a.tol * max(1.0, abs(walk[m][S0]))))
    g = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    g.to_csv(out / ("gates_forecasts.csv" if a.forecasts else "gates_runner.csv"), index=False)
    for m in MODELS:
        gm = g[g.model == m]
        say(f"gate {m}: batch fit == runner walk forecast at {len(gm)} cold-seed rows, max abs gap {gm.gap.max():.1e}")
    assert len(g) and g["ok"].all(), g[~g["ok"]]


# ============================================================================ reduce
def reduce(a) -> None:  # noqa: C901 -- one reducer
    from scipy.stats import spearmanr

    from rehaul_fi_compute import kendall_w, rank_table

    out = Path(a.out)
    info = json.loads((out / "run.json").read_text())
    Z = np.load(out / "linear.npz")
    names = info["names"]
    gr = groups(names)
    units, bmap, G = gr["units"], gr["bmap"], gr["G"]
    gcol, kind = np.array(gr["gcol"]), np.array(gr["kind"])
    U, p, K = len(units), len(names), info["K"]
    exog_u = [u for u in range(U) if bmap[units[u]] != "backbone"]
    is_bb = np.array([bmap[g] == "backbone" for g in gcol])
    rdates = info["anchors"]
    S: dict = dict(
        sample=info["sample"], window=info["window"], n_refits=K, p=p, n_units=U, n_exog=len(exog_u),
        n_buckets=len({bmap[units[u]] for u in exog_u}), first=rdates[0], span=[info["start"], info["end"]],
        n_block_rows=int(Z["block_rows"].sum()), r_bb=info["penalties"]["r"],
        probe_levels=info["probe_levels"], n_probe_cols=p + 3, penalties_file=info["penalties"]["file"],
        wall_sec=info["wall_sec"], sec_fit=info["sec_fit"], sec_probe=info["sec_probe"],
        masked_median=float(np.median(Z["masked"].sum(axis=1))),
    )
    base_cols = dict(unit=units, bucket=[bmap[s] for s in units], n_columns=G.sum(axis=0).astype(int))
    sd = Z["win_sd"]
    pk = Z["probe_k"]
    for name in MODELS:
        th = Z[f"{name}_theta"][:, :-1]  # (K, p), intercept dropped
        absb = np.abs(th)
        ser_abs = absb @ G  # ||beta_{k,A}||_1
        nz = absb > NZ_TOL
        nzA = (nz.astype(float) @ G) > 0
        df_s, R = rank_table(ser_abs, units)
        df_s = df_s.merge(pd.DataFrame(base_cols), on="unit")
        df_s["sum_abs"] = df_s["unit"].map(dict(zip(units, ser_abs.sum(axis=0))))
        df_s["sum_abs_share"] = df_s["sum_abs"] / ser_abs.sum()  # S_A
        sabs_sd = ((absb * sd) @ G).sum(axis=0)
        df_s["sum_abs_sd"] = df_s["unit"].map(dict(zip(units, sabs_sd)))
        df_s["sum_abs_sd_share"] = df_s["sum_abs_sd"] / sabs_sd.sum()  # S^sd_A
        df_s["refits_any_nonzero"] = df_s["unit"].map(dict(zip(units, nzA.sum(axis=0))))  # K_A
        df_s["selection_frequency"] = df_s["refits_any_nonzero"] / K
        df_s["mean_active_cols"] = df_s["unit"].map(dict(zip(units, (nz.astype(float) @ G).mean(axis=0))))
        df_s["W"] = kendall_w(R)
        df_s.to_csv(out / f"{name}_series.csv", index=False)
        dcol = pd.DataFrame(
            dict(column=names, series=gcol, bucket=[bmap[g] for g in gcol], kind=kind, refits_nonzero=nz.sum(axis=0),
                 sum_abs=absb.sum(axis=0), mean_abs=absb.mean(axis=0), sum_abs_sd=(absb * sd).sum(axis=0))
        )
        dcol.to_csv(out / f"{name}_columns.csv", index=False)
        pd.DataFrame(ser_abs, columns=units).assign(date=rdates, k=np.arange(K)).to_csv(
            out / f"{name}_series_by_refit.csv", index=False)
        share_k = ser_abs / ser_abs.sum(axis=1, keepdims=True)  # S_{k,A}
        pd.DataFrame(share_k, columns=units).assign(date=rdates, k=np.arange(K)).to_csv(
            out / f"{name}_share_by_refit.csv", index=False)
        pd.DataFrame(nzA.astype(int), columns=units).assign(date=rdates, k=np.arange(K)).to_csv(
            out / f"{name}_active_by_refit.csv", index=False)
        tot = absb.sum(axis=1)
        top = df_s.sort_values("sum_abs_share", ascending=False)
        top_sd = df_s.sort_values("sum_abs_sd_share", ascending=False)
        S[name] = dict(
            alpha=info["penalties"][name],
            W=float(kendall_w(R)),
            total_nonzero_median=float(np.median(nz.sum(axis=1))),
            total_nonzero_min=int(nz.sum(axis=1).min()),
            total_nonzero_max=int(nz.sum(axis=1).max()),
            exog_nonzero_median=float(np.median(nz[:, ~is_bb].sum(axis=1))),
            series_ever=int(nzA.any(axis=0).sum()),
            series_always=int(nzA.all(axis=0).sum()),
            exog_series_ever=int(sum(nzA[:, u].any() for u in exog_u)),
            exog_series_majority=int(sum(nzA[:, u].mean() >= 0.5 for u in exog_u)),
            exog_series_always=int(sum(nzA[:, u].all() for u in exog_u)),
            exog_always_list=[units[u] for u in exog_u if nzA[:, u].all()],
            cols_ever=int(nz.any(axis=0).sum()),
            cols_majority=int((nz.mean(axis=0) >= 0.5).sum()),
            cols_always=int(nz.all(axis=0).sum()),
            har_block_always=bool(nz[:, is_bb].any(axis=1).all()),
            backbone_share_median=float(np.median(absb[:, is_bb].sum(1) / tot)),
            har_share=float(df_s.loc[df_s.unit == "har_ma", "sum_abs_share"].iloc[0]),
            calendar_share=float(df_s.loc[df_s.unit == "calendar", "sum_abs_share"].sum()),
            top_share=[[u, float(v)] for u, v in zip(top["unit"][:5], top["sum_abs_share"][:5])],
            top_share_sd=[[u, float(v)] for u, v in zip(top_sd["unit"][:5], top_sd["sum_abs_sd_share"][:5])],
            top_mean_rank=df_s["unit"][:5].tolist(),
            spearman_abs_vs_abs_sd=float(spearmanr(df_s["sum_abs"], df_s["sum_abs_sd"])[0]),
            spearman_meanrank_vs_sum=float(spearmanr(df_s["mean_rank"], -df_s["sum_abs"])[0]),
            all_groups_selected=int(nzA.all(axis=0).sum()),
            block_mse_median=float(np.median(Z[f"{name}_block_mse"])),
        )
        pt = Z[f"{name}_probe_theta"]
        pb = pt[:, p : p + 3]
        thp = np.abs(pt[:, :p])
        S[name]["probe_n"] = int(len(pb))
        S[name]["probe_nonzero"] = [int((np.abs(pb[:, q]) > NZ_TOL).sum()) for q in range(3)]
        S[name]["probe_abs_median"] = [float(np.median(np.abs(pb[:, q]))) for q in range(3)]
        beat = (thp > np.abs(pb[:, [0]])).mean(axis=0) >= 0.5  # |beta_j| above the continuous probe's, most probe fits
        S[name]["cols_beat_probe"] = int(beat.sum())
        S[name]["series_beat_probe"] = len({gcol[j] for j in np.flatnonzero(beat) if not is_bb[j]})
        rate = S[name]["probe_nonzero"][0] / len(pb)
        sel = nz.mean(axis=0) > rate  # selected more often than the continuous probe
        S[name]["probe_rate"] = float(rate)
        S[name]["cols_above_probe_rate"] = int(sel.sum())
        S[name]["series_above_probe_rate"] = len({gcol[j] for j in np.flatnonzero(sel) if not is_bb[j]})
        S[name]["top_columns"] = dcol.sort_values(["refits_nonzero", "sum_abs"], ascending=False)["column"][:5].tolist()
        pd.DataFrame(
            dict(k=pk, date=[rdates[k] for k in pk], **{f"probe_{q}": pb[:, q] for q in range(3)},
                 cols_above_continuous=(thp > np.abs(pb[:, [0]])).sum(axis=1))
        ).to_csv(out / f"{name}_probes.csv", index=False)
    S["spearman_lasso_ridge_meanrank"] = float(
        spearmanr(
            pd.read_csv(out / "lasso_series.csv").set_index("unit").loc[units, "mean_rank"],
            pd.read_csv(out / "ridge_series.csv").set_index("unit").loc[units, "mean_rank"],
        )[0]
    )
    iqr = Z["win_iqr"]
    S["scale"] = {}
    for kd in ("value", "flag", "har", "calendar"):
        live = (sd > 0) & (kind == kd)[None]
        if not live.any():
            S["scale"][kd] = dict(n_cols=int((kind == kd).sum()))
            continue
        S["scale"][kd] = dict(
            n_cols=int((kind == kd).sum()),
            sd_median=float(np.median(sd[live])),
            sd_p05=float(np.percentile(sd[live], 5)),
            sd_p95=float(np.percentile(sd[live], 95)),
            iqr_median=float(np.median(iqr[live])),
        )
    pd.DataFrame(
        dict(column=names, kind=kind, group=gcol, sd_median=np.median(sd, axis=0), iqr_median=np.median(iqr, axis=0))
    ).to_csv(out / "column_scales.csv", index=False)
    (out / "summary.json").write_text(json.dumps(S, indent=1))
    say(f"reduce done: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("design", "linear", "gate", "reduce"))
    ap.add_argument("--row", type=int, default=0, help="gate: the cold-seed row of the runner walks")
    ap.add_argument("--walks", default="", help="gate: lasso=DIR,ridge=DIR")
    ap.add_argument("--forecasts", default="", help="gate: a rerun's forecasts.parquet")
    ap.add_argument("--tol", type=float, default=1e-9, help="gate: relative tolerance")
    ap.add_argument("--out", required=True)
    ap.add_argument("--design", default="")
    ap.add_argument("--bucket", default="all_features")
    ap.add_argument("--data", default="data")
    ap.add_argument("--sample", choices=tuple(SAMPLES), default="48")
    ap.add_argument("--penalties", default="")
    ap.add_argument("--start", default=SPAN[0])
    ap.add_argument("--end", default=SPAN[1])
    ap.add_argument("--no-check", action="store_true", help="skip check_landed (tests only)")
    ap.add_argument("--tw-days", type=int, default=TW_DAYS, help="window in trading days (tests only)")
    a = ap.parse_args()
    a.out = str(Path(a.out).resolve())
    if a.design:
        a.design = str(Path(a.design).resolve())
    if a.penalties:
        a.penalties = str(Path(a.penalties).resolve())
    os.chdir(REPO)
    {"design": design, "linear": linear, "gate": gate, "reduce": reduce}[a.stage](a)


if __name__ == "__main__":
    main()
