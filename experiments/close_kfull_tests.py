"""Statistical tests: LightGBM on the last k bars ending 16:00 (k = 1 .. 13) against the linear 16:00 model.

Design (the user's methodology note): a linear model trained on the 16:00 bar against LightGBM models
trained on backward-expanding pools of the last k half-hour bars ending 16:00, all scored only on the
16:00 forecast.  This script runs the test suite on the stored forecasts.  Every forecast goes through
the master table's research scorer (experiments/dense_vs_sparse_1530.py research_frame / deck_frame /
deck_panel: 16:00-bar recalibration (f^2 + s) B with s from the forecast's own trailing errors) on the
866 trade days 2020-01-03 .. 2024-04-30, so every loss is one value a day.

Forecasts
* Linear baseline (16:00 bar only): ridge with the HAR + calendar backbone unpenalized
  (exog_penalty/_work/runs/ridge_bb0.npz, primary); lasso_bb0, ridge_single, lasso_single (secondary).
* Trees: the bar-count ladder's stored forecasts (trees_morebars/_work, trees_datasize/_work for k = 1, 2;
  no bar column) and, when present, the files listed in trees_kfull/manifest.csv (the other agent's
  arms: k = 6, 8 .. 12, the column bar_end_minute for every k > 1, seeds 42 / 43 / 44).  For each k the
  tested forecast is the average of the available seeds' forecasts on the adjusted scale, then scored.

Losses: QLIKE (primary) and MSE on the variance level.  d_t = L_linear,t - L_tree,t (positive = the
tree has lower loss).

Tests (helpers in experiments/close_kfull_testlib.py)
1. DM (HLN-corrected, src/evaluation/diebold_mariano.dm_test), one- and two-sided, Newey-West lags 0, 5,
   10, 21, the dm_test automatic lag floor(4 (T/100)^(2/9)) and newey_west_lag floor(1.5 T^(1/3)); fixed-b
   critical values and p-values of the Bartlett-kernel t (Kiefer-Vogelsang) by simulation.
2. Giacomini-White: unconditional and conditional (instruments 1, d_{t-1}, the previous session's VIX /
   the previous trade day's implied variance), chi-square(q), HAC.
3. Multiple comparisons across k: Hansen's SPA (consistent / lower / upper p) and White's Reality Check
   with the stationary bootstrap; Holm-Bonferroni on the DM p-values; the model confidence set
   (src/evaluation/model_confidence_set.py) over {ridge, lasso, every tree pool}.  RESAMPLING: the repo's
   circular block bootstrap helper is switched off (commit b761b28: atm_straddle_lib.
   circular_block_bootstrap_idx returns the original order, and model_confidence_set._boot_col_means
   returns the sample in every replicate), so this script draws its own stationary-bootstrap indices
   and hands the MCS its own replicate means (the module's elimination rule is unchanged).
4. Encompassing (k* = lowest seed-mean QLIKE on the 2019 forecast rows, before the trade days; and the
   average of all pools): joint Mincer-Zarnowitz and its Patton-Sheppard weighted form, the Fair-Shiller /
   HLN (1998) combination regression, the QLIKE encompassing moment, a real-time combination
   (lambda on past trade days only) and equal weights, each against the ridge by DM.
5. Stability: Giacomini-Rossi fluctuation test (mu = 0.3, critical values by simulation), subperiods,
   previous-day VIX median split, FOMC days excluded, early-close days excluded.
6. Calibration: Mincer-Zarnowitz for each forecast.
7. Seed spread for each k and the curve of seed-averaged forecasts.
8. Bar column use, read from what the other agent records.
9. Within-day dependence of the adjusted target across the last 13 bars, design effect, effective n.

Stages:
  python experiments/close_kfull_tests.py gate                 reproduce trees_morebars/average.csv
  python experiments/close_kfull_tests.py run [interim|existing|full|auto]   gate, every test, CSVs, SUMMARY.md
        interim = the ladder's stored forecasts only; existing = plus every manifest file on disk;
        full = the same, asserting the manifest covers k = 1 .. 13 x seeds 42 / 43 / 44; auto = full if so, else existing
  python experiments/close_kfull_tests.py summary              SUMMARY.md from the CSVs
Outputs: results/close_studies_2026-10-03/kfull_tests/ (env KFT_OUT; KFT_KFULL = the folder holding manifest.csv).
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "experiments", REPO / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import close_kfull_testlib as T  # noqa: E402
import close_trees_morebars as mb  # noqa: E402  (arms, loaders; imports close_trees_datasize as mb.c)
import dense_vs_sparse_1530 as dvs  # noqa: E402
from src.evaluation import model_confidence_set as mcs_mod  # noqa: E402
from src.evaluation.diebold_mariano import _nw_lag, dm_test  # noqa: E402

import atm_straddle_lib as asl  # noqa: E402

c = mb.c
STUDIES = REPO / "results" / "close_studies_2026-10-03"
OUT = Path(os.environ.get("KFT_OUT", str(STUDIES / "kfull_tests")))
KFULL = Path(os.environ.get("KFT_KFULL", str(STUDIES / "trees_kfull")))  # the other agent's folder (manifest, bar-column record)
MANIFEST = KFULL / "manifest.csv"
LIN_DIR = STUDIES / "exog_penalty" / "_work" / "runs"
DESIGN13 = c.DESIGN / f"design_lastbars13_{c.BUCKET}.npz"
AVERAGE_CSV = mb.OUT / "average.csv"

LINEAR = {
    "ridge_bb0": "ridge, HAR + calendar unpenalized",
    "lasso_bb0": "lasso, HAR + calendar unpenalized",
    "ridge_single": "ridge, one penalty",
    "lasso_single": "lasso, one penalty",
}
BASE = "ridge_bb0"
LASSO = "lasso_bb0"
KS = tuple(range(1, 14))
SEEDS = (42, 43, 44)
FIRST_TRADE = pd.Timestamp("2020-01-03")
SPLIT_DATE = pd.Timestamp("2022-01-01")
FIXED_LAGS = (0, 5, 10, 21)
FIXEDB_REPS, FIXEDB_GRID = 50000, 2000
FLUCT_MU, FLUCT_REPS, FLUCT_GRID = 0.3, 50000, 2000
B_BOOT = 10000
MEAN_BLOCKS = (10, 5, 21)  # stationary bootstrap mean block length; 10 ~ T^(1/3) (the MCS module's default block) is the main one
MAIN_BLOCK = 10
BOOT_SEED = 20261004
MCS_ALPHAS = (0.10, 0.25)
RT_START = 250  # real-time combination: lambda from past trade days, first one after 250 days
LAMBDA_GRID = np.linspace(0.0, 1.0, 1001)
SESSIONS = 2000  # n_1: training sessions of every pool
REPRO_TOL = 1e-12
LOSSES = ("qlike", "mse")
FAM_LABEL = {
    "nobar": "LightGBM, last k bars, no bar column",
    "bar": "LightGBM, last k bars, column bar_end_minute for k > 1",
}


# ============================================================================ forecasts
def arm_path(arm: str) -> Path:
    s = mb.ARMS[arm]
    return (mb.DS_WORK / f"{s['reuse']}.npz") if s["reuse"] else (mb.WORK / f"{arm}.npz")


def interim_arms() -> list[dict]:
    """The bar-count ladder's stored LightGBM forecasts (2000 sessions)."""
    rows = []
    for n in mb.LADDER:
        base = f"lgbm_bars{n}_r{n * mb.SESSIONS}"
        for seed, arm in ((42, base), (43, f"{base}_seed43")):
            if arm in mb.ARMS and arm_path(arm).is_file():
                rows.append(dict(k=n, seed=seed, barcol=False, path=arm_path(arm), source=f"trees_morebars arm {arm}"))
    arm = "lgbm_bars13_r26000_barmin"
    if arm_path(arm).is_file():
        rows.append(dict(k=13, seed=42, barcol=True, path=arm_path(arm), source=f"trees_morebars arm {arm}"))
    return rows


def manifest_arms() -> tuple[list[dict], pd.DataFrame | None]:
    """The other agent's manifest (experiments/close_trees_kfull.py manifest): one row for each forecast
    file; family main (k = 1 without a column, k >= 2 with bar_end_minute) or no_bar_column; linear rows
    are skipped (the linear forecasts are read from exog_penalty).  A stored arm the other agent refitted
    under another name ("refitted here as X") is replaced by X when X's file exists."""
    if not MANIFEST.is_file():
        return [], None
    m = pd.read_csv(MANIFEST)
    m = m[m["model"].astype(str).str.lower() == "lgbm"].copy()
    rows = []
    for _, r in m.iterrows():
        st = str(r["status"])
        if "refitted here as" in st:
            repl = st.split("refitted here as")[1].strip().split()[0].strip(";,.)")
            rr = m[m["arm"] == repl]
            if len(rr) and (REPO / str(rr["path"].iloc[0])).is_file():
                continue
        p = REPO / str(r["path"])
        rows.append(dict(k=int(r["k"]), seed=int(r["seed"]), barcol=str(r["bar_column"]).strip().lower() == "yes", path=p,
                         source=f"trees_kfull manifest: {r['arm']} ({r['family']}, {st})", arm=str(r["arm"]), mfamily=str(r["family"])))
    return rows, m


def full_complete(rows: list[dict]) -> bool:
    """Every LightGBM file the manifest lists is on disk, and the main family covers k = 1 .. 13 x seeds 42 / 43 / 44."""
    if not rows or not all(r["path"].is_file() for r in rows):
        return False
    have = {(r["k"], r["seed"]) for r in rows if r["barcol"] or r["k"] == 1}
    return all((k, s) in have for k in KS for s in SEEDS)


def load_pred(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)["pred"].astype(float)


def families(rows: list[dict]) -> dict[str, dict[int, list[dict]]]:
    """family -> k -> arms (one for each seed).  k = 1 has one bar and no bar column: it belongs to both."""
    fam: dict[str, dict[int, list[dict]]] = {"nobar": {}, "bar": {}}
    seen = set()
    for r in rows:
        key = (r["k"], r["seed"], r["barcol"])
        if key in seen:  # the same forecast listed twice (ladder and manifest): keep the first
            continue
        seen.add(key)
        for f in (("nobar", "bar") if r["k"] == 1 else (("bar",) if r["barcol"] else ("nobar",))):
            fam[f].setdefault(r["k"], []).append(r)
    for f in fam:
        fam[f] = {k: sorted(v, key=lambda a: a["seed"]) for k, v in sorted(fam[f].items())}
    # a family needs at least three values of k to be a family over k
    return {f: v for f, v in fam.items() if len([k for k in v if k > 1]) >= 2}


# ============================================================================ scoring
class Scorer:
    """Every forecast through the research scorer; losses on the 866 trade days and on the 2019 rows."""

    def __init__(self) -> None:
        S = c.sources()
        self.n_fc = S["n_fc"]
        self.sub = np.arange(c.TREE_START, self.n_fc)
        self.dz = {k: v[self.sub] for k, v in S["dz"].items()}
        self.dk = dvs.deck_frame()
        self.days = self.dk.index
        self.n = len(self.days)
        self.F: dict[str, np.ndarray] = {}
        self.ql: dict[str, np.ndarray] = {}
        self.mse: dict[str, np.ndarray] = {}
        self.pnl: dict[str, np.ndarray] = {}
        self.pre: dict[str, np.ndarray] = {}
        self.pre_dates = None
        self.RV = None

    def add(self, name: str, pred: np.ndarray) -> None:
        assert pred.shape == (self.n_fc,) and np.isfinite(pred[self.sub]).all(), name
        fr = dvs.research_frame(self.dz, pred[self.sub])
        P = dvs.deck_panel([fr], self.dk)
        f = fr["pred_clock"].set_axis(fr.index.normalize())
        rv = fr["true_raw"].set_axis(fr.index.normalize())
        F = f.reindex(self.days).to_numpy(float)
        RV = rv.reindex(self.days).to_numpy(float)
        assert np.allclose(P["ql"][:, 0], T.qlike(RV, F), rtol=1e-12, atol=0)
        if self.RV is None:
            self.RV = RV
        assert np.array_equal(self.RV, RV)
        self.F[name], self.ql[name], self.mse[name], self.pnl[name] = F, P["ql"][:, 0], (RV - F) ** 2, P["pnl"][:, 0]
        m = (f.index < FIRST_TRADE) & np.isfinite(f.to_numpy(float))
        if self.pre_dates is None:
            self.pre_dates = f.index[m]
        assert self.pre_dates.equals(f.index[m]), name
        self.pre[name] = T.qlike(rv[m].to_numpy(float), f[m].to_numpy(float))

    def loss(self, kind: str, name: str) -> np.ndarray:
        return self.ql[name] if kind == "qlike" else self.mse[name]


# ============================================================================ gate
def gate(write: bool = True) -> pd.DataFrame:
    """Reproduce trees_morebars/average.csv with this script's scorer."""
    ref = pd.read_csv(AVERAGE_CSV).set_index("forecast")
    p = {a: load_pred(arm_path(a)) for a in mb.ARMS if mb.ARMS[a]["model"] == "lgbm" and arm_path(a).is_file()}
    lad = [f"lgbm_bars{n}_r{n * mb.SESSIONS}" for n in mb.LADDER]
    ridge = load_pred(LIN_DIR / "ridge_bb0.npz")
    avg = np.mean([p[a] for a in lad], axis=0)
    combos = {
        "lgbm, 1 bar (control)": p[mb.CTRL],
        "lgbm, 2 bars": p[mb.TWO],
        "lgbm, 4 bars (best single pool)": p[lad[3]],
        "average of lgbm pools N = 1, 2, 3, 4, 5, 7, 13": avg,
        "ridge, backbone unpenalized (linear reference)": ridge,
        "equal weights: ridge backbone unpenalized + average of lgbm pools, all N": 0.5 * ridge + 0.5 * avg,
        "equal weights: ridge backbone unpenalized + lgbm 4 bars": 0.5 * ridge + 0.5 * p[lad[3]],
    }
    sc = Scorer()
    for k, v in combos.items():
        sc.add(k, v)
    rows = []
    rk = "ridge, backbone unpenalized (linear reference)"
    for k in combos:
        sh = sc.pnl[k].mean() / sc.pnl[k].std(ddof=1) * float(np.sqrt(asl.PERIODS_PER_YEAR))
        vals = [("qlike", float(sc.ql[k].mean())), ("sharpe_mid", float(sh))]
        if k != rk:
            vals.append(("dm_vs_ridge", float(dm_test(sc.ql[k], sc.ql[rk])["dm"])))
        for what, v in vals:
            r0 = float(ref.loc[k, what])
            rows.append(dict(forecast=k, quantity=what, value=v, average_csv=r0, abs_diff=abs(v - r0), passed=bool(abs(v - r0) < REPRO_TOL)))
    g = pd.DataFrame(rows)
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        g.to_csv(OUT / "gate.csv", index=False)
    print(g.to_string(index=False), flush=True)
    assert g["passed"].all(), "GATE failed: average.csv not reproduced"
    return g


# ============================================================================ instruments and subsets
def daily_vix_prev(days: pd.DatetimeIndex) -> np.ndarray:
    """The VIX at the bar ending 16:00 of the session before each trade day (last value at or before 16:00);
    NaN when the session before has no value."""
    v = pd.read_parquet(REPO / "data" / "vix_and_voldemand.parquet", columns=["endbartime", "vix"])
    v["endbartime"] = pd.to_datetime(v["endbartime"])
    v = v[(v["endbartime"].dt.hour * 60 + v["endbartime"].dt.minute <= 960) & v["vix"].notna()]
    close = v.groupby(v["endbartime"].dt.normalize())["vix"].last()
    close.index = pd.DatetimeIndex(close.index).as_unit("ns")
    # sessions = the dates of the 16:00 rows (every session of the forecast span)
    sess = pd.DatetimeIndex(pd.to_datetime(c.sources()["dz"]["date"])).normalize().as_unit("ns")
    pos = sess.searchsorted(days, side="left") - 1
    assert (pos >= 0).all() and (sess[pos] < days).all()
    return close.reindex(sess[pos]).to_numpy(float)  # NaN when that session has no VIX value (the file ends 2024-02-12)


def fomc_days() -> tuple[set, pd.Timestamp]:
    r = pd.read_parquet(REPO / "data" / "releases.parquet", columns=["endbartime", "fomc release"])
    r["endbartime"] = pd.to_datetime(r["endbartime"])
    f = r.loc[r["fomc release"] > 0, "endbartime"]
    return set(f.dt.normalize()), f.max()


# ============================================================================ tests
def dm_rows(sc: Scorer, fam: str, trees: dict[str, dict], fixb: dict, lags: dict) -> list[dict]:
    rows = []
    n = sc.n
    corr = math.sqrt((n - 1) / n)  # the HLN factor at h = 1
    for loss in LOSSES:
        for base in LINEAR:
            La = sc.loss(loss, base)
            for name, info in trees.items():
                Lt = sc.loss(loss, name)
                for lab, lag in lags.items():
                    if base != BASE and lab not in ("auto", "nw"):
                        continue
                    r = dm_test(La, Lt, hac_lag=lag)
                    t_unc = r["dm"] / corr
                    draws = fixb[lag]
                    rows.append(
                        dict(
                            family=fam, loss=loss, baseline=base, forecast=name, **info, lag_rule=lab, lag=r["hac_lag"],
                            mean_linear=float(La.mean()), mean_tree=float(Lt.mean()), mean_d=r["mean_diff"], dm=r["dm"],
                            p_two=r["p"], p_one=T.norm_sf(r["dm"]), fixedb_b=(r["hac_lag"] + 1) / n,
                            fixedb_cv_one_05=float(np.quantile(draws, 0.95)), fixedb_cv_two_05=float(np.quantile(np.abs(draws), 0.95)),
                            fixedb_cv_two_10=float(np.quantile(np.abs(draws), 0.90)),
                            fixedb_p_one=float(np.mean(draws >= t_unc)), fixedb_p_two=float(np.mean(np.abs(draws) >= abs(t_unc))), n_days=r["T"],
                        )
                    )
    return rows


def gw_rows(sc: Scorer, fam: str, trees: dict[str, dict], vix_prev: np.ndarray, hac_lag: int) -> list[dict]:
    iv = sc.dk["iv_var"].to_numpy(float)
    rows = []
    for loss in LOSSES:
        La = sc.loss(loss, BASE)
        for name, info in trees.items():
            d = La - sc.loss(loss, name)
            sets = {
                "1": (d, np.ones((sc.n, 1))),
                "1, d(t-1)": (d[1:], np.column_stack([np.ones(sc.n - 1), d[:-1]])),
                "1, d(t-1), implied variance(t-1)": (d[1:], np.column_stack([np.ones(sc.n - 1), d[:-1], iv[:-1]])),
            }
            ok = np.isfinite(vix_prev[1:])
            sets["1, d(t-1), VIX(t-1) (days with a VIX value)"] = (d[1:][ok], np.column_stack([np.ones(sc.n - 1), d[:-1], vix_prev[1:]])[ok])
            for lab, (dd, H) in sets.items():
                g = T.gw_test(dd, H, hac_lag)
                rows.append(dict(family=fam, loss=loss, baseline=BASE, forecast=name, **info, instruments=lab, q=g["q"], stat=g["stat"], p=g["p"], n_days=g["n"], hac_lag=hac_lag))
    return rows


_W: dict[int, np.ndarray] = {}


def counts(n: int, block: int) -> np.ndarray:
    """(B, n) resampling counts of the stationary bootstrap with this mean block (cached; shared by SPA, RC, MCS)."""
    if block not in _W:
        idx = T.stationary_idx(n, B_BOOT, block, np.random.default_rng([BOOT_SEED, block]))
        W = np.zeros((B_BOOT, n))
        flat = (idx + np.arange(B_BOOT)[:, None] * n).ravel()
        W.ravel()[:] = np.bincount(flat, minlength=B_BOOT * n)
        _W[block] = W
    return _W[block]


def _mcs_boot_col_means(L: np.ndarray, B: int, block: int, seed: int) -> np.ndarray:
    """Replacement for model_confidence_set._boot_col_means (switched off in the module): stationary bootstrap means."""
    del seed
    W = counts(L.shape[0], block)
    assert W.shape == (B, L.shape[0])
    return W @ L / L.shape[0]


def multiple_rows(sc: Scorer, fam: str, kn: dict[int, str], dm_df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    mult, mcs_rows = [], []
    names = [kn[k] for k in sorted(kn)]
    for loss in LOSSES:
        La = sc.loss(loss, BASE)
        d = np.column_stack([La - sc.loss(loss, x) for x in names])
        for block in MEAN_BLOCKS:
            Ds = counts(sc.n, block) @ d / sc.n
            r = T.spa_rc(d, Ds)
            mult.append(dict(family=fam, loss=loss, test="SPA (Hansen 2005)", mean_block=block, B=B_BOOT, stat=r["stat_spa"], p_consistent=r["p_consistent"], p_lower=r["p_lower"], p_upper=r["p_upper"], best=names[r["best"]], n_models=len(names)))
            mult.append(dict(family=fam, loss=loss, test="Reality Check (White 2000)", mean_block=block, B=B_BOOT, stat=r["stat_rc"], p_consistent=np.nan, p_lower=np.nan, p_upper=r["p_rc"], best=names[int(np.argmax(d.mean(0)))], n_models=len(names)))
        sub = dm_df[(dm_df.family == fam) & (dm_df.loss == loss) & (dm_df.baseline == BASE) & (dm_df.lag_rule == "auto")].set_index("forecast").loc[names]
        for side in ("one", "two"):
            adj = T.holm(sub[f"p_{side}"].to_numpy())
            for nm, p0, pa in zip(names, sub[f"p_{side}"], adj):
                mult.append(dict(family=fam, loss=loss, test=f"Holm-Bonferroni on DM p ({side}-sided, dm_test automatic lag)", forecast=nm, p_raw=p0, p_holm=pa, n_models=len(names)))
        losses = {BASE: La, LASSO: sc.loss(loss, LASSO), **{x: sc.loss(loss, x) for x in names}}
        for alpha in MCS_ALPHAS:
            res = mcs_mod.model_confidence_set(losses, alpha=alpha, B=B_BOOT, block=MAIN_BLOCK, seed=BOOT_SEED)
            order = {nm: i + 1 for i, nm in enumerate(res["eliminated"])}
            for nm in losses:
                mcs_rows.append(dict(family=fam, loss=loss, alpha=alpha, model=nm, mean_loss=float(losses[nm].mean()), in_mcs=nm in res["mcs"], mcs_p=res["pvals"][nm], eliminated_rank=order.get(nm, np.nan), mean_block=MAIN_BLOCK, B=B_BOOT, n_days=res["T"]))
    return mult, mcs_rows


def encompassing_rows(sc: Scorer, fam: str, cands: dict[str, str], hac_lag: int) -> list[dict]:
    y, fL = sc.RV, sc.F[BASE]
    rows = []
    for name, choice in cands.items():
        fT = sc.F[name]
        base = dict(family=fam, forecast=name, choice=choice, n_days=sc.n, hac_lag=hac_lag)
        for form, yy, X in (
            ("joint Mincer-Zarnowitz, level", y, np.column_stack([np.ones(sc.n), fL, fT])),
            ("joint Mincer-Zarnowitz, divided by f_L (Patton-Sheppard)", y / fL, np.column_stack([1.0 / fL, np.ones(sc.n), fT / fL])),
        ):
            b, V, _ = T.ols_hac(yy, X, hac_lag)
            se = np.sqrt(np.diag(V))
            w = T.wald(b, V, np.eye(3), np.array([0.0, 1.0, 0.0]))
            rows.append(dict(**base, test=form, a=b[0], b_L=b[1], b_T=b[2], se_a=se[0], se_b_L=se[1], se_b_T=se[2],
                             t_b_T_eq_0=b[2] / se[2], p_b_T_eq_0=2 * T.norm_sf(abs(b[2] / se[2])),
                             t_b_L_eq_0=b[1] / se[1], p_b_L_eq_0=2 * T.norm_sf(abs(b[1] / se[1])),
                             wald_a0_bL1_bT0=w[0], p_wald=w[2]))
        b, V, _ = T.ols_hac(y - fL, (fT - fL)[:, None], hac_lag)
        se = math.sqrt(V[0, 0])
        rows.append(dict(**base, test="combination regression y - f_L = lambda (f_T - f_L) (Fair-Shiller / HLN 1998)", lam=b[0], se_lam=se,
                         t_lam_eq_0=b[0] / se, p_lam_eq_0_one=T.norm_sf(b[0] / se), p_lam_eq_0_two=2 * T.norm_sf(abs(b[0] / se)),
                         t_lam_eq_1=(b[0] - 1) / se, p_lam_eq_1_two=2 * T.norm_sf(abs((b[0] - 1) / se))))
        for lab, f0, f1 in (("QLIKE encompassing E[(1/f_L - y/f_L^2)(f_T - f_L)] = 0", fL, fT), ("QLIKE encompassing, reverse E[(1/f_T - y/f_T^2)(f_L - f_T)] = 0", fT, fL)):
            m = (1.0 / f0 - y / f0**2) * (f1 - f0)
            t = T.hac_t(m, hac_lag)
            rows.append(dict(**base, test=lab, moment_mean=float(m.mean()), t_moment=t, p_moment_one=T.norm_sf(-t), p_moment_two=2 * T.norm_sf(abs(t))))
    return rows


def realtime_rows(sc: Scorer, fam: str, cands: dict[str, str], eq_adj: dict[str, str]) -> tuple[list[dict], pd.DataFrame]:
    y, fL = sc.RV, sc.F[BASE]
    rows, paths = [], {}
    late = np.arange(sc.n) >= RT_START
    for name, choice in cands.items():
        fT = sc.F[name]
        lam = T.realtime_lambda(y, fL, fT, RT_START, LAMBDA_GRID)
        paths[f"{name}"] = lam
        fc = {
            "real-time lambda (QLIKE on past trade days, expanding)": fL + np.nan_to_num(lam) * (fT - fL),
            "equal weights, variance level": 0.5 * (fL + fT),
            "equal weights, adjusted scale, rescored": sc.F[eq_adj[name]],
        }
        for lab, f in fc.items():
            for sample, msk in (("trade days 251 .. 866", late), ("all 866 trade days", np.ones(sc.n, bool))):
                if lab.startswith("real-time") and sample.startswith("all"):
                    continue  # lambda needs 250 earlier trade days
                for loss in LOSSES:
                    Lc = T.qlike(y, f) if loss == "qlike" else (y - f) ** 2
                    La = sc.loss(loss, BASE)
                    r = dm_test(La[msk], Lc[msk])
                    rows.append(dict(family=fam, forecast=name, choice=choice, combination=lab, sample=sample, loss=loss, n_days=int(msk.sum()),
                                     mean_linear=float(La[msk].mean()), mean_combination=float(Lc[msk].mean()), mean_d=r["mean_diff"], dm=r["dm"],
                                     p_two=r["p"], p_one=T.norm_sf(r["dm"]), lag=r["hac_lag"],
                                     lambda_first=float(lam[RT_START]) if lab.startswith("real-time") else (0.5),
                                     lambda_last=float(lam[-1]) if lab.startswith("real-time") else 0.5,
                                     lambda_min=float(np.nanmin(lam)) if lab.startswith("real-time") else 0.5,
                                     lambda_max=float(np.nanmax(lam)) if lab.startswith("real-time") else 0.5,
                                     lambda_median=float(np.nanmedian(lam)) if lab.startswith("real-time") else 0.5))
    lp = pd.DataFrame({"date": sc.days[RT_START:], **{k: v[RT_START:] for k, v in paths.items()}})
    lp.insert(0, "family", fam)
    return rows, lp


def stability_rows(sc: Scorer, fam: str, trees: dict[str, dict], vix_prev: np.ndarray, fomc: set, hac_lag: int) -> list[dict]:
    days = sc.days
    early = set(pd.to_datetime(list(asl.EARLY_CLOSE_DATES)))
    is_fomc = np.asarray(days.isin(list(fomc)))
    is_early = np.asarray(days.isin(list(early)))
    okv = np.isfinite(vix_prev)
    vmed = np.median(vix_prev[okv])
    iv_prev = np.concatenate([[np.nan], sc.dk["iv_var"].to_numpy(float)[:-1]])
    oki = np.isfinite(iv_prev)
    imed = np.median(iv_prev[oki])
    late = np.asarray(days >= SPLIT_DATE)
    all_days = np.ones(sc.n, bool)
    subsets = {
        "all 866 trade days": all_days,
        "2020-2021": ~late,
        "2022-2024": late,
        "previous-day VIX at or above its median": okv & (vix_prev >= vmed),
        "previous-day VIX below its median": okv & (vix_prev < vmed),
        "previous trade day implied variance at or above its median": oki & (iv_prev >= imed),
        "previous trade day implied variance below its median": oki & (iv_prev < imed),
        "FOMC days excluded (flags end 2023-11-01)": ~is_fomc,
        "early-close days excluded": ~is_early,
    }
    # (rows used, group dummy)
    splits = {
        "2022-2024 vs 2020-2021": (all_days, late),
        "VIX high vs low": (okv, vix_prev >= vmed),
        "implied variance high vs low": (oki, iv_prev >= imed),
        "FOMC day vs other days": (all_days, is_fomc),
    }
    rows = []
    for loss in LOSSES:
        La = sc.loss(loss, BASE)
        for name, info in trees.items():
            Lt = sc.loss(loss, name)
            d = La - Lt
            for lab, m in subsets.items():
                r = dm_test(La[m], Lt[m])
                rows.append(dict(family=fam, loss=loss, forecast=name, **info, subset=lab, n_days=int(m.sum()), mean_linear=float(La[m].mean()), mean_tree=float(Lt[m].mean()),
                                 mean_d=r["mean_diff"], dm=r["dm"], p_two=r["p"], p_one=T.norm_sf(r["dm"]), lag=r["hac_lag"]))
            for lab, (ok, D) in splits.items():
                b, V, _ = T.ols_hac(d[ok], np.column_stack([np.ones(int(ok.sum())), D[ok].astype(float)]), hac_lag)
                t = b[1] / math.sqrt(V[1, 1])
                rows.append(dict(family=fam, loss=loss, forecast=name, **info, subset=f"difference in mean d: {lab}", n_days=int(ok.sum()), n_in_group=int(D[ok].sum()),
                                 mean_d=float(b[1]), dm=t, p_two=2 * T.norm_sf(abs(t)), lag=hac_lag))
    return rows


def fluctuation_rows(sc: Scorer, fam: str, trees: dict[str, dict], cv: dict, hac_lag: int) -> tuple[list[dict], list[pd.DataFrame]]:
    m = int(round(FLUCT_MU * sc.n))
    rows, paths = [], []
    for loss in LOSSES:
        La = sc.loss(loss, BASE)
        for name, info in trees.items():
            d = La - sc.loss(loss, name)
            sig = math.sqrt(float(T.hac_cov(d - d.mean(), hac_lag)[0, 0]))
            F = T.fluctuation_path(d, m, sig)
            ends = sc.days[m - 1 :]
            rows.append(dict(family=fam, loss=loss, forecast=name, **info, window=m, mu=FLUCT_MU, hac_lag=hac_lag, max_stat=float(F.max()), max_window_end=ends[int(F.argmax())].date(),
                             min_stat=float(F.min()), min_window_end=ends[int(F.argmin())].date(), sup_abs=float(np.abs(F).max()), **cv,
                             reject_two_05=bool(np.abs(F).max() > cv["cv_two_05"]), reject_one_05=bool(F.max() > cv["cv_one_05"])))
            paths.append(pd.DataFrame(dict(family=fam, loss=loss, forecast=name, window_end=ends, stat=F)))
    return rows, paths


def calibration_rows(sc: Scorer, fam: str, names: dict[str, dict], hac_lag: int) -> list[dict]:
    y = sc.RV
    rows = []
    for name, info in names.items():
        f = sc.F[name]
        for form, yy, X in (("level y = a + b f", y, np.column_stack([np.ones(sc.n), f])), ("divided by f (Patton-Sheppard) y/f = a/f + b", y / f, np.column_stack([1.0 / f, np.ones(sc.n)]))):
            b, V, u = T.ols_hac(yy, X, hac_lag)
            se = np.sqrt(np.diag(V))
            w = T.wald(b, V, np.eye(2), np.array([0.0, 1.0]))
            r2 = 1.0 - float(u @ u) / float(((yy - yy.mean()) ** 2).sum())
            rows.append(dict(family=fam, forecast=name, **info, form=form, a=b[0], b=b[1], se_a=se[0], se_b=se[1], t_b_eq_1=(b[1] - 1) / se[1], p_b_eq_1=2 * T.norm_sf(abs((b[1] - 1) / se[1])),
                             wald_a0_b1=w[0], p_wald=w[2], r2=r2, n_days=sc.n, hac_lag=hac_lag))
    return rows


def within_day_rows() -> list[dict]:
    z = np.load(DESIGN13, allow_pickle=False)
    d = pd.DatetimeIndex(pd.to_datetime(z["date"]))
    y = pd.Series(z["y"].astype(float), index=d)
    day = d.normalize()
    full = pd.Series(1, index=day).groupby(level=0).size()
    full = full[full == 13].index
    keep = np.asarray(day.isin(full))
    Y = y[keep].to_frame("y").assign(day=day[keep], hm=d[keep].strftime("%H:%M")).pivot(index="day", columns="hm", values="y")
    Y = Y[sorted(Y.columns)]
    assert Y.shape[1] == 13 and Y.columns[-1] == "16:00" and Y.notna().all().all()
    rows = []
    samples = {
        f"the {SESSIONS} full sessions before {FIRST_TRADE.date()}": Y[Y.index < FIRST_TRADE].iloc[-SESSIONS:],
        "all full sessions of the design": Y,
    }
    for lab, Yk in samples.items():
        C = np.corrcoef(Yk.to_numpy().T)
        for k in KS:
            sl = slice(13 - k, 13)
            sub = C[sl, sl]
            rho = float((sub.sum() - k) / (k * (k - 1))) if k > 1 else np.nan
            deff = 1.0 + (k - 1) * rho if k > 1 else 1.0
            rows.append(dict(sample=lab, n_sessions=len(Yk), first=Yk.index[0].date(), last=Yk.index[-1].date(), k=k, added_bar=Yk.columns[13 - k],
                             corr_added_bar_with_1600=float(C[13 - k, 12]), mean_pairwise_corr=rho, design_effect=deff,
                             n_rows=SESSIONS * k, effective_n=SESSIONS * k / deff, quantity="adjusted target"))
    return rows


def bar_column_rows() -> list[dict]:
    """What the other agent records about the use of bar_end_minute (and hour): trees_kfull/bar_column_by_k.csv,
    over the seeds and refits of each k (split count, split share, gain share, gain rank among the columns kept)."""
    f = KFULL / "bar_column_by_k.csv"
    if not f.is_file():
        return []
    df = pd.read_csv(f)
    df.insert(0, "source_file", str(f.relative_to(REPO)) if f.is_relative_to(REPO) else str(f))
    return df.to_dict("records")


# ============================================================================ run
def run(which: str = "auto") -> None:  # noqa: C901 - one linear driver
    t0, c0 = time.time(), time.process_time()
    OUT.mkdir(parents=True, exist_ok=True)
    gate()
    mrows, manifest = manifest_arms()
    complete = full_complete(mrows) if mrows else False
    if which == "auto":
        which = "full" if complete else "existing"
    if which == "full":
        assert complete, "the manifest does not list every k = 1 .. 13 x seeds 42 / 43 / 44 yet"
    # interim = the ladder's stored forecasts only; existing / full = every manifest file on disk (the manifest lists the stored ones too)
    rows = interim_arms() if (which == "interim" or not mrows) else [r for r in mrows if r["path"].is_file()]
    fams = families(rows)
    nk = {f: len(v) for f, v in fams.items()}
    primary = "bar" if ("bar" in fams and nk["bar"] >= nk.get("nobar", 0)) else "nobar"
    print(f"set: {which}; manifest rows {len(mrows)} (complete: {complete}); families {[(f, sorted(v)) for f, v in fams.items()]}; primary {primary}", flush=True)

    sc = Scorer()
    for nm in LINEAR:
        sc.add(nm, load_pred(LIN_DIR / f"{nm}.npz"))
    inv = []
    for nm in LINEAR:
        inv.append(dict(forecast=nm, family="linear", k=np.nan, seed=np.nan, barcol=False, kind="linear (16:00 bar)", file=str((LIN_DIR / f"{nm}.npz").relative_to(REPO)), source=LINEAR[nm]))
    preds_k: dict[str, dict[int, np.ndarray]] = {}
    tree_info: dict[str, dict[str, dict]] = {}
    single_info: dict[str, dict[str, dict]] = {}
    for fam, kd in fams.items():
        preds_k[fam], tree_info[fam], single_info[fam] = {}, {}, {}
        for k, arms in kd.items():
            ps = []
            for a in arms:
                nm = f"{fam}:k{k}:s{a['seed']}"
                p = load_pred(a["path"])
                ps.append(p)
                if nm not in sc.F:
                    sc.add(nm, p)
                single_info[fam][nm] = dict(k=k, seed=str(a["seed"]), n_seeds=1)
                inv.append(dict(forecast=nm, family=fam, k=k, seed=a["seed"], barcol=a["barcol"], kind="single seed", file=str(a["path"].relative_to(REPO)), source=a["source"]))
            preds_k[fam][k] = np.mean(ps, axis=0)
            nm = f"{fam}:k{k}"
            sc.add(nm, preds_k[fam][k])
            seeds = "/".join(str(a["seed"]) for a in arms)
            tree_info[fam][nm] = dict(k=k, seed=f"mean of {seeds}", n_seeds=len(arms))
            inv.append(dict(forecast=nm, family=fam, k=k, seed=f"mean of {seeds}", barcol=fam == "bar" and k > 1, kind="seed average (adjusted scale)", file="", source=""))
        nm = f"{fam}:avg"
        sc.add(nm, np.mean([preds_k[fam][k] for k in sorted(preds_k[fam])], axis=0))
        tree_info[fam][nm] = dict(k=np.nan, seed="seed averages", n_seeds=np.nan)
        inv.append(dict(forecast=nm, family=fam, k=np.nan, seed="", barcol=fam == "bar", kind=f"average of the seed averages, k = {', '.join(map(str, sorted(preds_k[fam])))}", file="", source=""))
    # extra forecasts outside the families (e.g. the ladder's k = 13 bar-column arm when the bar family is incomplete)
    extra: dict[str, dict] = {}
    for a in rows:
        if a["barcol"] and a["k"] > 1 and "bar" not in fams:
            nm = f"extra:k{a['k']}:s{a['seed']}:barcol"
            sc.add(nm, load_pred(a["path"]))
            extra[nm] = dict(k=a["k"], seed=str(a["seed"]), n_seeds=1)
            inv.append(dict(forecast=nm, family="extra", k=a["k"], seed=a["seed"], barcol=True, kind="single seed, column bar_end_minute", file=str(a["path"].relative_to(REPO)), source=a["source"]))

    # ---------------------------------------------------------------- k* from the 2019 rows
    sel_rows, kstar = [], {}
    for fam in fams:
        for k in sorted(preds_k[fam]):
            singles = [nm for nm in single_info[fam] if single_info[fam][nm]["k"] == k]
            sel_rows.append(dict(family=fam, k=k, n_seeds=len(singles), qlike_2019_seed_mean=float(np.mean([sc.pre[s].mean() for s in singles])),
                                 qlike_2019_of_seed_average=float(sc.pre[f"{fam}:k{k}"].mean()), n_rows=len(sc.pre_dates),
                                 first_row=sc.pre_dates[0].date(), last_row=sc.pre_dates[-1].date()))
        s = pd.DataFrame([r for r in sel_rows if r["family"] == fam])
        kstar[fam] = int(s.loc[s["qlike_2019_seed_mean"].idxmin(), "k"])
        for r in sel_rows:
            if r["family"] == fam:
                r["chosen"] = r["k"] == kstar[fam]
                r["lowest_of_seed_average"] = r["k"] == int(s.loc[s["qlike_2019_of_seed_average"].idxmin(), "k"])
    for nm in LINEAR:
        sel_rows.append(dict(family="linear", k=np.nan, n_seeds=np.nan, qlike_2019_seed_mean=np.nan, qlike_2019_of_seed_average=float(sc.pre[nm].mean()), n_rows=len(sc.pre_dates), first_row=sc.pre_dates[0].date(), last_row=sc.pre_dates[-1].date(), chosen=False, lowest_of_seed_average=False, forecast=nm))
    pd.DataFrame(sel_rows).to_csv(OUT / "kstar_selection.csv", index=False)
    # equal weights on the adjusted scale (the average.csv construction), rescored
    eq_adj = {}
    for fam in fams:
        ridge = load_pred(LIN_DIR / f"{BASE}.npz")
        for nm, p in ((f"{fam}:k{kstar[fam]}", preds_k[fam][kstar[fam]]), (f"{fam}:avg", np.mean([preds_k[fam][k] for k in sorted(preds_k[fam])], axis=0))):
            e = f"{nm}:eq_ridge"
            sc.add(e, 0.5 * ridge + 0.5 * p)
            eq_adj[nm] = e
    for r in inv:
        nm = r["forecast"]
        r.update(qlike=float(sc.ql[nm].mean()), mse=float(sc.mse[nm].mean()), qlike_2019=float(sc.pre[nm].mean()), n_days=sc.n)
    pd.DataFrame(inv).to_csv(OUT / "forecasts.csv", index=False)

    # ---------------------------------------------------------------- simulated critical values
    auto_lag = max(0, _nw_lag(sc.n))  # dm_test's default at h = 1
    nw_lag = asl.newey_west_lag(sc.n)
    lags = {**{str(L): L for L in FIXED_LAGS}, "auto": auto_lag, "nw": nw_lag}
    lag_vals = sorted(set(lags.values()))
    bmap = {L: (L + 1) / sc.n for L in lag_vals}
    draws = T.fixedb_draws(list(bmap.values()), n_grid=FIXEDB_GRID, reps=FIXEDB_REPS, seed=BOOT_SEED)
    fixb = {L: draws[bmap[L]] for L in lag_vals}
    pd.DataFrame([dict(lag=L, b=bmap[L], bandwidth_M=L + 1, n_grid=FIXEDB_GRID, reps=FIXEDB_REPS, cv_one_10=np.quantile(fixb[L], 0.90), cv_one_05=np.quantile(fixb[L], 0.95),
                       cv_one_01=np.quantile(fixb[L], 0.99), cv_two_10=np.quantile(np.abs(fixb[L]), 0.90), cv_two_05=np.quantile(np.abs(fixb[L]), 0.95), cv_two_01=np.quantile(np.abs(fixb[L]), 0.99))
                  for L in lag_vals]).to_csv(OUT / "fixedb_critical_values.csv", index=False)
    fa, fs = T.fluctuation_draws(FLUCT_MU, n_grid=FLUCT_GRID, reps=FLUCT_REPS, seed=BOOT_SEED + 1)
    fcv = dict(cv_two_05=float(np.quantile(fa, 0.95)), cv_two_10=float(np.quantile(fa, 0.90)), cv_one_05=float(np.quantile(fs, 0.95)), cv_one_10=float(np.quantile(fs, 0.90)))
    pd.DataFrame([dict(mu=FLUCT_MU, n_grid=FLUCT_GRID, reps=FLUCT_REPS, **fcv)]).to_csv(OUT / "fluctuation_critical_values.csv", index=False)

    vix_prev = daily_vix_prev(sc.days)
    fomc, fomc_last = fomc_days()
    mcs_mod._boot_col_means = _mcs_boot_col_means  # the module's own bootstrap is switched off; see the docstring

    dm_all, gw_all, mult_all, mcs_all, enc_all, rt_all, lam_all, st_all, fl_all, flp_all, cal_all = ([] for _ in range(11))
    for fam in fams:
        trees = dict(tree_info[fam])
        trees_and_singles = {**trees, **single_info[fam], **(extra if fam == primary else {})}
        dm_all += dm_rows(sc, fam, trees_and_singles, fixb, lags)
    dm_df = pd.DataFrame(dm_all)
    dm_df.to_csv(OUT / "dm.csv", index=False)
    for fam in fams:
        trees = dict(tree_info[fam])
        kn = {tree_info[fam][nm]["k"]: nm for nm in trees if not nm.endswith(":avg")}
        gw_all += gw_rows(sc, fam, trees, vix_prev, nw_lag)
        m1, m2 = multiple_rows(sc, fam, kn, dm_df)
        mult_all += m1
        mcs_all += m2
        cands = {f"{fam}:k{kstar[fam]}": f"k* = {kstar[fam]} (lowest seed-mean QLIKE on the 2019 rows)", f"{fam}:avg": "average of all pools (no choice)"}
        for k in sorted(preds_k[fam]):
            nm = f"{fam}:k{k}"
            if nm not in cands:
                cands[nm] = f"k = {k} (not chosen in advance)"
        enc_all += encompassing_rows(sc, fam, cands, nw_lag)
        rr, lp = realtime_rows(sc, fam, {x: cands[x] for x in list(cands)[:2]}, eq_adj)
        rt_all += rr
        lam_all.append(lp)
        st_all += stability_rows(sc, fam, trees, vix_prev, fomc, nw_lag)
        a, b = fluctuation_rows(sc, fam, trees, fcv, nw_lag)
        fl_all += a
        flp_all += b
        cal_all += calibration_rows(sc, fam, trees, nw_lag)
    cal_all = calibration_rows(sc, "linear", {nm: dict(k=np.nan, seed="", n_seeds=np.nan) for nm in LINEAR}, nw_lag) + cal_all
    pd.DataFrame(gw_all).to_csv(OUT / "gw.csv", index=False)
    pd.DataFrame(mult_all).to_csv(OUT / "multiple_comparisons.csv", index=False)
    pd.DataFrame(mcs_all).to_csv(OUT / "mcs.csv", index=False)
    pd.DataFrame(enc_all).to_csv(OUT / "encompassing.csv", index=False)
    pd.DataFrame(rt_all).to_csv(OUT / "realtime_combination.csv", index=False)
    pd.concat(lam_all).to_csv(OUT / "realtime_lambda_path.csv", index=False)
    pd.DataFrame(st_all).to_csv(OUT / "stability.csv", index=False)
    pd.DataFrame(fl_all).to_csv(OUT / "fluctuation.csv", index=False)
    pd.concat(flp_all).to_csv(OUT / "fluctuation_path.csv", index=False)
    pd.DataFrame(cal_all).to_csv(OUT / "calibration.csv", index=False)

    # ---------------------------------------------------------------- seed spread
    ss, curve = [], []
    for fam in fams:
        for nm, info in single_info[fam].items():
            ss.append(dict(family=fam, k=info["k"], seed=int(info["seed"]), qlike=float(sc.ql[nm].mean()), mse=float(sc.mse[nm].mean()), qlike_2019=float(sc.pre[nm].mean())))
        sdf = pd.DataFrame([r for r in ss if r["family"] == fam])
        prev = None
        for k in sorted(preds_k[fam]):
            g = sdf[sdf.k == k]
            row = dict(family=fam, k=k, n_seeds=len(g), seeds="/".join(map(str, g.seed)))
            for loss in LOSSES:
                row[f"{loss}_seed_mean"] = float(g[loss].mean())
                row[f"{loss}_seed_sd"] = float(g[loss].std(ddof=1)) if len(g) > 1 else np.nan
                row[f"{loss}_of_seed_average"] = float(sc.loss(loss, f"{fam}:k{k}").mean())
                if prev is not None:
                    step = row[f"{loss}_of_seed_average"] - prev[f"{loss}_of_seed_average"]
                    sds = [x for x in (row[f"{loss}_seed_sd"], prev[f"{loss}_seed_sd"]) if np.isfinite(x)]
                    pooled = float(np.sqrt(np.mean(np.square(sds)))) if sds else np.nan
                    row[f"{loss}_step_from_k{prev['k']}"] = step
                    row[f"{loss}_step"] = step
                    row[f"{loss}_sd_of_the_two_ends"] = pooled
                    row[f"{loss}_step_exceeds_sd"] = bool(abs(step) > pooled) if np.isfinite(pooled) else np.nan
                    row[f"{loss}_step_dm"] = dm_test(sc.loss(loss, f"{fam}:k{prev['k']}"), sc.loss(loss, f"{fam}:k{k}"))["dm"]
            row["previous_k"] = prev["k"] if prev is not None else np.nan
            curve.append(row)
            prev = row
    pd.DataFrame(ss).to_csv(OUT / "seed_spread.csv", index=False)
    cdf = pd.DataFrame(curve)
    cdf = cdf[[x for x in cdf.columns if "_step_from_k" not in x]]
    cdf.to_csv(OUT / "seed_curve.csv", index=False)

    pd.DataFrame(bar_column_rows()).to_csv(OUT / "bar_column.csv", index=False)
    pd.DataFrame(within_day_rows()).to_csv(OUT / "within_day.csv", index=False)

    info = dict(
        set=which,
        manifest=str(MANIFEST.relative_to(REPO)) if MANIFEST.is_relative_to(REPO) else str(MANIFEST),
        manifest_present=manifest is not None,
        manifest_rows=0 if manifest is None else len(manifest),
        manifest_complete=complete,
        primary_family=primary,
        families={f: {str(k): [a["seed"] for a in v] for k, v in kd.items()} for f, kd in fams.items()},
        kstar=kstar,
        n_trade_days=sc.n,
        first_trade_day=str(sc.days[0].date()),
        last_trade_day=str(sc.days[-1].date()),
        dm_auto_lag=auto_lag,
        newey_west_lag=nw_lag,
        fomc_flags_last=str(fomc_last.date()),
        fomc_days_in_trade_days=int(np.asarray(sc.days.isin(list(fomc))).sum()),
        early_close_days_in_trade_days=int(np.asarray(sc.days.isin(list(pd.to_datetime(list(asl.EARLY_CLOSE_DATES))))).sum()),
        vix_median=float(np.nanmedian(vix_prev)),
        vix_days=int(np.isfinite(vix_prev).sum()),
        vix_last_day_with_previous_value=str(sc.days[np.isfinite(vix_prev)][-1].date()),
        iv_prev_median=float(np.median(sc.dk["iv_var"].to_numpy(float)[:-1])),
        boot=dict(kind="stationary bootstrap (Politis-Romano)", B=B_BOOT, mean_blocks=list(MEAN_BLOCKS), main=MAIN_BLOCK, seed=BOOT_SEED),
        cpu_sec=time.process_time() - c0,
        wall_sec=time.time() - t0,
        written=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    (OUT / "run_info.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps(info, indent=1, default=str), flush=True)
    summary()


# ============================================================================ SUMMARY.md
def _f(x, nd=4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return f"{x:.{nd}f}"


def _p(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return f"{x:.3f}" if x >= 0.001 else f"{x:.1e}"


def _e(x) -> str:
    return f"{x:.3e}"


def summary() -> None:  # noqa: C901 - one linear report
    info = json.loads((OUT / "run_info.json").read_text())
    rd = lambda n: pd.read_csv(OUT / n)  # noqa: E731
    gate_df, inv, dm, gw, mult, mcs = rd("gate.csv"), rd("forecasts.csv"), rd("dm.csv"), rd("gw.csv"), rd("multiple_comparisons.csv"), rd("mcs.csv")
    enc, rt, st, fl, cal = rd("encompassing.csv"), rd("realtime_combination.csv"), rd("stability.csv"), rd("fluctuation.csv"), rd("calibration.csv")
    curve, ss, sel, wd, fbcv, flcv = rd("seed_curve.csv"), rd("seed_spread.csv"), rd("kstar_selection.csv"), rd("within_day.csv"), rd("fixedb_critical_values.csv"), rd("fluctuation_critical_values.csv")
    try:
        bc = rd("bar_column.csv")
    except pd.errors.EmptyDataError:
        bc = pd.DataFrame()
    fam = info["primary_family"]
    other = [f for f in info["families"] if f != fam]
    ks = sorted(int(k) for k in info["families"][fam])
    kst = info["kstar"][fam]
    L = []
    w = L.append
    w("# Tests: LightGBM on the last k bars ending 16:00 against the linear 16:00 model (close study, 2026-10-04)")
    w("")
    w(f"Written by `experiments/close_kfull_tests.py` from the CSVs in this folder ({info['written']}); every number below is in them.")
    w("")
    w("## Forecasts and set")
    full = info["set"] == "full"
    w(f"- Set: **{info['set']}**. Manifest `{info['manifest']}`: " + ("absent" if not info["manifest_present"] else f"{info['manifest_rows']} rows, complete: {info['manifest_complete']}") + ".")
    for f, kd in info["families"].items():
        w(f"- Family `{f}` ({FAM_LABEL[f]}): " + "; ".join(f"k = {k}: seeds {'/'.join(map(str, v))}" for k, v in kd.items()) + ("  (primary)" if f == fam else ""))
    if not full:
        w("- This is the set that existed when the script ran: k = 6 and 8 .. 12, the bar-column arms for k = 2 .. 12 and seed 44 were not there yet. "
          "For each k the tested forecast is the average of the seeds listed (one seed for some k).")
    w(f"- Linear baseline: `ridge_bb0` (ridge, HAR + calendar unpenalized; QLIKE {_f(inv.set_index('forecast').loc['ridge_bb0', 'qlike'])}); secondary `lasso_bb0`, `ridge_single`, `lasso_single`.")
    w(f"- Scorer: research convention (16:00-bar recalibration (f^2 + s) B from each forecast's own errors), {info['n_trade_days']} trade days {info['first_trade_day']} .. {info['last_trade_day']}, one loss a day. "
      "For each k, the seed forecasts are averaged on the adjusted scale and the average is scored. Losses: QLIKE (primary), MSE on the variance level. d = L_linear - L_tree, positive = the tree has lower loss.")
    gq = gate_df[gate_df.quantity == "qlike"].set_index("forecast")["value"]
    w(f"- Gate: trees_morebars/average.csv reproduced, {int(gate_df.passed.sum())} / {len(gate_df)} quantities (QLIKE, Sharpe mid, DM against the ridge) within {REPRO_TOL:g} (max |difference| {gate_df.abs_diff.max():.1e}); "
      f"e.g. QLIKE lgbm 4 bars {_f(gq['lgbm, 4 bars (best single pool)'])}, ridge_bb0 {_f(gq['ridge, backbone unpenalized (linear reference)'])}, equal weights ridge + 4-bar pool {_f(gq['equal weights: ridge backbone unpenalized + lgbm 4 bars'])}.")
    w(f"- Resampling: the repo's circular block bootstrap helper is switched off (commit b761b28: `atm_straddle_lib.circular_block_bootstrap_idx` returns the original order; `model_confidence_set._boot_col_means` likewise returns the sample in every replicate). "
      f"SPA, Reality Check and the MCS here use this script's own stationary bootstrap (Politis-Romano), B = {info['boot']['B']}, mean block {info['boot']['main']} (also {', '.join(str(b) for b in info['boot']['mean_blocks'] if b != info['boot']['main'])} in the CSV); the MCS keeps the module's elimination rule with these replicate means.")
    w("")

    # ---------------- DM
    w("## 1. Diebold-Mariano against the ridge (HLN-corrected), QLIKE")
    d0 = dm[(dm.family == fam) & (dm.loss == "qlike") & (dm.baseline == BASE)]
    dmain = d0[d0.forecast.str.fullmatch(rf"{fam}:k\d+") | (d0.forecast == f"{fam}:avg")]
    order = [f"{fam}:k{k}" for k in ks] + [f"{fam}:avg"]
    labs = {f"{fam}:k{k}": str(k) for k in ks} | {f"{fam}:avg": "average of all pools"}
    lag_rules = ["0", "5", "auto", "10", "nw", "21"]
    lagv = {r: int(dmain[dmain.lag_rule == r].lag.iloc[0]) for r in lag_rules}
    w(f"DM statistic (positive = tree lower loss) at Newey-West lag 0 / 5 / {lagv['auto']} (dm_test automatic) / 10 / {lagv['nw']} (newey_west_lag) / 21; one-sided normal p (H1: tree lower loss) and fixed-b one-sided p (Bartlett, b = (lag+1)/T, simulated) at the automatic lag.")
    w("")
    w("| k | seeds | QLIKE tree | QLIKE ridge | mean d | " + " | ".join(f"DM lag {lagv[r]}" for r in lag_rules) + " | p one (auto) | p two (auto) | fixed-b p one (auto) | fixed-b p two (auto) |")
    w("|" + "---|" * (9 + len(lag_rules)))
    for nm in order:
        g = dmain[dmain.forecast == nm].set_index("lag_rule")
        a = g.loc["auto"]
        w(f"| {labs[nm]} | {a.seed} | {_f(a.mean_tree)} | {_f(a.mean_linear)} | {a.mean_d:+.4f} | " + " | ".join(f"{g.loc[r, 'dm']:.2f}" for r in lag_rules) + f" | {_p(a.p_one)} | {_p(a.p_two)} | {_p(a.fixedb_p_one)} | {_p(a.fixedb_p_two)} |")
    w("")
    fb = fbcv.set_index("lag")
    w("Fixed-b critical values (simulated, " + f"{int(fbcv.reps.iloc[0])} draws, grid {int(fbcv.n_grid.iloc[0])}): " + "; ".join(f"lag {int(L)} (b = {fb.loc[L, 'b']:.4f}): one-sided 5% {fb.loc[L, 'cv_one_05']:.3f}, two-sided 5% {fb.loc[L, 'cv_two_05']:.3f}" for L in fb.index) + ".")
    w("")
    dm_mse = dm[(dm.family == fam) & (dm.loss == "mse") & (dm.baseline == BASE) & (dm.lag_rule == "auto")].set_index("forecast")
    w("MSE (variance level), automatic lag: " + "; ".join(f"k = {labs[nm]}: DM {dm_mse.loc[nm, 'dm']:.2f} (p one {_p(dm_mse.loc[nm, 'p_one'])})" for nm in order) + ".")
    w("")
    d2 = dm[(dm.family == fam) & (dm.loss == "qlike") & (dm.lag_rule == "auto") & dm.forecast.isin(order)]
    w("Secondary baselines (QLIKE, automatic lag), DM for each k:")
    w("")
    w("| baseline | QLIKE baseline | " + " | ".join(f"k = {labs[nm]}" for nm in order) + " |")
    w("|" + "---|" * (2 + len(order)))
    for base in LINEAR:
        g = d2[d2.baseline == base].set_index("forecast")
        w(f"| {base} | {_f(g.mean_linear.iloc[0])} | " + " | ".join(f"{g.loc[nm, 'dm']:.2f}" for nm in order) + " |")
    w("")
    sgl = d0[(d0.lag_rule == "auto") & d0.forecast.str.contains(":s") & ~d0.forecast.str.startswith("extra")]
    if len(sgl):
        w("Single seeds (QLIKE, automatic lag): " + "; ".join(f"k = {int(r.k)} seed {r.seed}: DM {r.dm:.2f}" for r in sgl.sort_values(["k", "seed"]).itertuples()) + ".")
        ex = dm[(dm.forecast.str.startswith("extra")) & (dm.loss == "qlike") & (dm.baseline == BASE) & (dm.lag_rule == "auto")]
        for r in ex.itertuples():
            w(f"Outside the family: k = {int(r.k)}, column bar_end_minute, seed {r.seed}: QLIKE {_f(r.mean_tree)}, DM {r.dm:.2f} (p one {_p(r.p_one)}).")
        w("")

    # ---------------- GW
    w("## 2. Giacomini-White against the ridge (HAC, newey_west_lag " + f"{info['newey_west_lag']})")
    g0 = gw[(gw.family == fam) & (gw.forecast.isin(order))]
    insts = list(dict.fromkeys(g0.instruments))
    for loss in LOSSES:
        w("")
        w(f"{loss.upper()}: statistic (p), chi-square with q df.")
        w("")
        w("| k | " + " | ".join(f"{i} (q = {int(g0[g0.instruments == i].q.iloc[0])})" for i in insts) + " |")
        w("|" + "---|" * (1 + len(insts)))
        for nm in order:
            g = g0[(g0.loss == loss) & (g0.forecast == nm)].set_index("instruments")
            w(f"| {labs[nm]} | " + " | ".join(f"{g.loc[i, 'stat']:.2f} ({_p(g.loc[i, 'p'])})" for i in insts) + " |")
    w("")
    w(f"Conditional sets use trade days 2 .. {info['n_trade_days']} (d(t-1) is the previous trade day's d; the deck's trade days are not every session); implied variance(t-1) = the deck's iv_var on the previous trade day; "
      f"VIX(t-1) = VIX at the bar ending 16:00 of the previous session, on the {int(g0[g0.instruments.str.startswith('1, d(t-1), VIX')].n_days.iloc[0])} days with a value (the VIX file ends 2024-02-12).")
    w("")

    # ---------------- multiple
    w("## 3. Multiple comparisons across k (against the ridge)")
    w("SPA and Reality Check: H0 max over k of E[d(k)] <= 0 (no pool has lower expected loss than the ridge); d(k) for the seed-averaged forecast of each k.")
    for loss in LOSSES:
        m = mult[(mult.family == fam) & (mult.loss == loss)]
        spa = m[(m.test.str.startswith("SPA"))].set_index("mean_block")
        rc = m[(m.test.str.startswith("Reality"))].set_index("mean_block")
        b0 = info["boot"]["main"]
        w(f"- {loss.upper()}: SPA statistic {spa.loc[b0, 'stat']:.3f} (largest studentized mean d at {spa.loc[b0, 'best']}), p consistent / lower / upper = {_p(spa.loc[b0, 'p_consistent'])} / {_p(spa.loc[b0, 'p_lower'])} / {_p(spa.loc[b0, 'p_upper'])} (mean block {b0}); "
          + "; ".join(f"block {int(b)}: {_p(spa.loc[b, 'p_consistent'])} / {_p(spa.loc[b, 'p_lower'])} / {_p(spa.loc[b, 'p_upper'])}" for b in spa.index if b != b0)
          + f". Reality Check p = {_p(rc.loc[b0, 'p_upper'])} (block {b0}); " + ", ".join(f"block {int(b)} {_p(rc.loc[b, 'p_upper'])}" for b in rc.index if b != b0) + ".")
        h = m[m.test.str.startswith("Holm") & m.test.str.contains("one-sided")].set_index("forecast")
        w("  Holm-Bonferroni, one-sided DM p (automatic lag), adjusted: " + "; ".join(f"k = {labs[nm]} {_p(h.loc[nm, 'p_holm'])}" for nm in order if nm in h.index) + f"; smallest adjusted {_p(h.p_holm.min())}.")
    w("")
    w(f"Model confidence set over {{ridge_bb0, lasso_bb0, every tree pool}} (T_max rule of `src/evaluation/model_confidence_set.py`, stationary bootstrap, mean block {info['boot']['main']}):")
    w("")
    for loss in LOSSES:
        for alpha in MCS_ALPHAS:
            mm = mcs[(mcs.family == fam) & (mcs.loss == loss) & (np.isclose(mcs.alpha, alpha))]
            mem = mm[mm.in_mcs]
            pretty = lambda x: x if x in LINEAR else ("k = " + x.split(":k")[1] if ":k" in x else x)  # noqa: E731
            el = mm[~mm.in_mcs].sort_values("eliminated_rank")
            w(f"- {loss.upper()}, alpha {alpha:.2f}: {len(mem)} of {len(mm)} in the set: " + ", ".join(f"{pretty(r.model)} (p {_p(r.mcs_p)})" for r in mem.sort_values("mean_loss").itertuples())
              + ("; eliminated in order: " + ", ".join(f"{pretty(r.model)} (p {_p(r.mcs_p)})" for r in el.itertuples()) if len(el) else "") + ".")
    w("")

    # ---------------- encompassing
    w("## 4. Encompassing")
    s0 = sel[sel.family == fam]
    w(f"k* = {kst}: lowest seed-mean QLIKE on the {int(s0.n_rows.iloc[0])} 16:00 rows {s0.first_row.iloc[0]} .. {s0.last_row.iloc[0]} (2019, before the trade days; the recalibration needs 63 earlier errors): "
      + "; ".join(f"k = {int(r.k)} {r.qlike_2019_seed_mean:.4f}" for r in s0.itertuples()) + f". The QLIKE of the seed-averaged forecast on the same rows is lowest at k = {int(s0[s0.lowest_of_seed_average].k.iloc[0])}.")
    w("")
    w("y = realized 16:00 variance, f = recalibrated forecasts (pred_clock), f_L = ridge_bb0, f_T = tree; HAC lag " + f"{info['newey_west_lag']}.")
    w("")
    w("| tree forecast | joint MZ b_L (t of 0) | b_T (t of 0) | Wald (a, b_L, b_T) = (0, 1, 0) p | weighted MZ b_L (t) | b_T (t) | Wald p | lambda (se) | t lambda = 0 | t lambda = 1 | QLIKE moment t | reverse moment t |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for nm in [f"{fam}:k{kst}", f"{fam}:avg"] + [x for x in order if x not in (f"{fam}:k{kst}", f"{fam}:avg")]:
        e = enc[(enc.family == fam) & (enc.forecast == nm)].set_index("test")
        mz = e[e.index.str.startswith("joint Mincer-Zarnowitz, level")].iloc[0]
        ps = e[e.index.str.contains("Patton")].iloc[0]
        cr = e[e.index.str.startswith("combination")].iloc[0]
        qm = e[e.index.str.startswith("QLIKE encompassing E")].iloc[0]
        qr = e[e.index.str.contains("reverse")].iloc[0]
        lab = {f"{fam}:k{kst}": f"k* = {kst}", f"{fam}:avg": "average of all pools"}.get(nm, f"k = {labs[nm]} (not chosen in advance)")
        w(f"| {lab} | {mz.b_L:.3f} ({mz.t_b_L_eq_0:.2f}) | {mz.b_T:.3f} ({mz.t_b_T_eq_0:.2f}) | {_p(mz.p_wald)} | {ps.b_L:.3f} ({ps.t_b_L_eq_0:.2f}) | {ps.b_T:.3f} ({ps.t_b_T_eq_0:.2f}) | {_p(ps.p_wald)} | {cr.lam:.3f} ({cr.se_lam:.3f}) | {cr.t_lam_eq_0:.2f} | {cr.t_lam_eq_1:.2f} | {qm.t_moment:.2f} | {qr.t_moment:.2f} |")
    w("")
    w("QLIKE moment t < 0: moving from f_L toward f_T lowers QLIKE at lambda = 0 (f_L does not encompass f_T); reverse moment t < 0: moving from f_T toward f_L lowers it.")
    w("")
    w("Combinations against the ridge (DM, automatic lag; positive = combination lower loss):")
    w("")
    w("| tree forecast | combination | sample | loss | mean ridge | mean combination | DM | p one | lambda first / median / last (min .. max) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in rt[(rt.family == fam)].itertuples():
        lam = f"{r.lambda_first:.3f} / {r.lambda_median:.3f} / {r.lambda_last:.3f} ({r.lambda_min:.3f} .. {r.lambda_max:.3f})" if r.combination.startswith("real-time") else "0.5"
        lab = f"k* = {kst}" if r.forecast == f"{fam}:k{kst}" else "average of all pools"
        mf = _f if r.loss == "qlike" else _e
        w(f"| {lab} | {r.combination} | {r.sample} | {r.loss} | {mf(r.mean_linear)} | {mf(r.mean_combination)} | {r.dm:.2f} | {_p(r.p_one)} | {lam} |")
    w("")

    # ---------------- stability
    w("## 5. Stability (QLIKE, against the ridge)")
    f0 = fl[(fl.family == fam) & (fl.loss == "qlike")].set_index("forecast")
    cv = flcv.iloc[0]
    w(f"Giacomini-Rossi fluctuation test, window {int(f0.window.iloc[0])} trade days (mu = {cv.mu}), statistic = rolling sum of d / (full-sample HAC sd x sqrt(window)); simulated critical values ({int(cv.reps)} draws): two-sided 5% {cv.cv_two_05:.3f} / 10% {cv.cv_two_10:.3f}, one-sided 5% {cv.cv_one_05:.3f} / 10% {cv.cv_one_10:.3f}.")
    w("")
    w("| k | max (window end) | min (window end) | sup abs > two-sided 5% | max > one-sided 5% |")
    w("|---|---|---|---|---|")
    for nm in order:
        r = f0.loc[nm]
        w(f"| {labs[nm]} | {r.max_stat:.2f} ({r.max_window_end}) | {r.min_stat:.2f} ({r.min_window_end}) | {r.reject_two_05} | {r.reject_one_05} |")
    w("")
    s1 = st[(st.family == fam) & (st.loss == "qlike")]
    subs = [x for x in dict.fromkeys(s1.subset) if not x.startswith("difference")]
    w("DM (automatic lag) in subsets; n days in the header:")
    w("")
    w("| k | " + " | ".join(f"{x} ({int(s1[s1.subset == x].n_days.iloc[0])})" for x in subs) + " |")
    w("|" + "---|" * (1 + len(subs)))
    for nm in order:
        g = s1[s1.forecast == nm].set_index("subset")
        w(f"| {labs[nm]} | " + " | ".join(f"{g.loc[x, 'dm']:.2f}" for x in subs) + " |")
    w("")
    w("Mean QLIKE of the ridge in each subset: " + "; ".join(f"{x} {s1[s1.subset == x].mean_linear.iloc[0]:.4f}" for x in subs) + ".")
    w("")
    diffs = [x for x in dict.fromkeys(s1.subset) if x.startswith("difference")]
    w("Difference in mean d between groups (d on a constant and the group dummy, HAC t of the dummy): " + "; ".join(
        f"{x.removeprefix('difference in mean d: ')}: " + ", ".join(f"k = {labs[nm]} {s1[(s1.subset == x) & (s1.forecast == nm)].dm.iloc[0]:.2f}" for nm in order) for x in diffs) + ".")
    w("")
    w(f"FOMC flags in `data/releases.parquet` stop at {info['fomc_flags_last']}, so FOMC days after that date stay in the 'FOMC days excluded' subset; {info['fomc_days_in_trade_days']} flagged FOMC days are trade days. "
      f"Early-close days among the trade days: {info['early_close_days_in_trade_days']} (the 15:30 straddle deck has none), so that subset equals the full sample. "
      f"VIX median split at {info['vix_median']:.2f} on the {info['vix_days']} trade days whose previous session has a VIX value (`data/vix_and_voldemand.parquet` ends 2024-02-12; last such trade day {info['vix_last_day_with_previous_value']}); "
      f"the implied-variance split uses the deck's iv_var of the previous trade day (median {info['iv_prev_median']:.3e}) on trade days 2 .. {info['n_trade_days']}.")
    w("")

    # ---------------- calibration
    w("## 6. Calibration: Mincer-Zarnowitz on the variance level, y = a + b f (HAC)")
    w("")
    w("| forecast | a | b (se) | t of b = 1 | Wald (a, b) = (0, 1) p | R^2 | weighted b (t of b = 1) | weighted Wald p |")
    w("|---|---|---|---|---|---|---|---|")
    for nm in list(LINEAR) + order:
        g = cal[cal.forecast == nm]
        lv = g[g.form.str.startswith("level")].iloc[0]
        wt = g[g.form.str.startswith("divided")].iloc[0]
        lab = nm if nm in LINEAR else f"k = {labs[nm]}" if nm != f"{fam}:avg" else "average of all pools"
        w(f"| {lab} | {lv.a:.2e} | {lv.b:.3f} ({lv.se_b:.3f}) | {lv.t_b_eq_1:.2f} | {_p(lv.p_wald)} | {lv.r2:.3f} | {wt.b:.3f} ({wt.t_b_eq_1:.2f}) | {_p(wt.p_wald)} |")
    w("")

    # ---------------- seeds
    w("## 7. Seed spread and the curve of seed-averaged forecasts (QLIKE)")
    w("")
    w("| k | seeds | QLIKE for each seed | seed mean (sd) | QLIKE of the seed average | step from the previous k | sd of the two ends | step exceeds sd | DM of the step |")
    w("|---|---|---|---|---|---|---|---|---|")
    c0 = curve[curve.family == fam]
    for r in c0.itertuples():
        each = ", ".join(f"{q:.4f}" for q in ss[(ss.family == fam) & (ss.k == r.k)].sort_values("seed").qlike)
        sd = f"{r.qlike_seed_sd:.4f}" if np.isfinite(r.qlike_seed_sd) else "one seed"
        step = f"{r.qlike_step:+.4f} (k = {int(r.previous_k)})" if np.isfinite(r.previous_k) else ""
        sdt = _f(r.qlike_sd_of_the_two_ends) if np.isfinite(r.previous_k) else ""
        ex = "" if not (np.isfinite(r.previous_k) and np.isfinite(r.qlike_sd_of_the_two_ends)) else str(r.qlike_step_exceeds_sd)
        dms = f"{r.qlike_step_dm:.2f}" if np.isfinite(r.previous_k) else ""
        w(f"| {r.k} | {r.seeds} | {each} | {r.qlike_seed_mean:.4f} ({sd}) | {r.qlike_of_seed_average:.4f} | {step} | {sdt} | {ex} | {dms} |")
    w("")
    w("sd = sample sd over seeds (ddof 1); sd of the two ends = root mean square of the two k's seed sd's (one end when the other has one seed); DM of the step: QLIKE of the previous k's seed average minus this k's (positive = this k lower).")
    w("")

    # ---------------- bar column
    w("## 8. Use of the bar column (bar_end_minute)")
    if len(bc):
        w(f"From `{bc.source_file.iloc[0]}` (the other agent's record, written by `experiments/close_trees_kfull.py`; copied to bar_column.csv): at every refit the booster's split count and gain of each column; "
          "share = the column's splits (gain) over all splits (gain) of that fit; rank 1 = most gain among the columns the fit kept; over the seeds and refits of each k.")
        w("")
        w("| family | k | column | arms | refits | % refits with a split | mean splits | split share | gain share mean (min .. max) | gain rank median (best .. worst) | columns kept (median) |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in bc.sort_values(["family", "column", "k"]).itertuples():
            w(f"| {r.family} | {int(r.k)} | {r.column} | {int(r.arms)} | {int(r.refits)} | {r.pct_refits_split:.1f} | {r.splits_mean:.1f} | {r.split_share_mean:.4f} | {r.gain_share_mean:.4f} ({r.gain_share_min:.4f} .. {r.gain_share_max:.4f}) | "
              f"{_f(r.rank_gain_median, 0)} ({_f(r.rank_gain_best, 0)} .. {_f(r.rank_gain_worst, 0)}) | {_f(r.n_columns_kept_median, 0)} |")
        w("")
        bm = bc[bc.column == "bar_end_minute"]
        never = sorted(int(k) for k in bm[bm.pct_refits_split == 0].k)
        w(("bar_end_minute: no split in any refit at k = " + ", ".join(map(str, never)) + ". ") if never else
          f"bar_end_minute: split on in {bm.pct_refits_split.min():.1f} .. {bm.pct_refits_split.max():.1f} % of the refits across k = {', '.join(str(int(k)) for k in sorted(bm.k))}; no k without a split. ")
        w("")
    else:
        w("`results/close_studies_2026-10-03/trees_kfull/bar_column_by_k.csv` did not exist when the script ran (the other agent writes it from the importances of its new forecast files).")
    w("")

    # ---------------- within-day
    w("## 9. Within-day dependence of the adjusted target (lastbars13 design)")
    w("")
    w("Mean pairwise correlation rho across the last k bars of the same session (columns = bars, rows = sessions with all 13 bars), design effect 1 + (k - 1) rho, effective n = 2000 k / design effect.")
    w("")
    w("| k | added bar | corr of added bar with 16:00 | rho | design effect | rows | effective n | rho (all sessions) | effective n (all sessions) |")
    w("|---|---|---|---|---|---|---|---|---|")
    samp = list(dict.fromkeys(wd["sample"]))
    a, b = wd[wd["sample"] == samp[0]].set_index("k"), wd[wd["sample"] == samp[1]].set_index("k")
    for k in KS:
        r, r2 = a.loc[k], b.loc[k]
        w(f"| {k} | {r.added_bar} | {r.corr_added_bar_with_1600:.3f} | {_f(r.mean_pairwise_corr, 3)} | {r.design_effect:.2f} | {int(r.n_rows)} | {r.effective_n:.0f} | {_f(r2.mean_pairwise_corr, 3)} | {r2.effective_n:.0f} |")
    w("")
    w(f"Main columns: {samp[0]} ({int(a.n_sessions.iloc[0])} sessions, {a['first'].iloc[0]} .. {a['last'].iloc[0]}); 'all sessions': {int(b.n_sessions.iloc[0])} sessions, {b['first'].iloc[0]} .. {b['last'].iloc[0]}. The trees forecast only the 16:00 row, so no tree errors exist on the earlier bars; the target is used.")
    w("")

    if other:
        w("## Other family")
        for f in other:
            ks2 = sorted(int(k) for k in info["families"][f])
            o2 = [f"{f}:k{k}" for k in ks2] + [f"{f}:avg"]
            g = dm[(dm.family == f) & (dm.loss == "qlike") & (dm.baseline == BASE) & (dm.lag_rule == "auto")].set_index("forecast")
            m = mult[(mult.family == f) & (mult.loss == "qlike") & mult.test.str.startswith("SPA") & (mult.mean_block == info["boot"]["main"])].iloc[0]
            w(f"- `{f}` ({FAM_LABEL[f]}), QLIKE, DM against the ridge (automatic lag): " + "; ".join(f"k = {x.split(':')[1][1:] if ':k' in x else 'average'} {g.loc[x, 'mean_tree']:.4f} / {g.loc[x, 'dm']:.2f}" for x in o2) + f". SPA p consistent / lower / upper {_p(m.p_consistent)} / {_p(m.p_lower)} / {_p(m.p_upper)}; k* = {info['kstar'][f]}.")
        w("")

    w("## Files")
    w("- `gate.csv`, `forecasts.csv` (every scored forecast, file, QLIKE, MSE), `kstar_selection.csv`, `dm.csv`, `fixedb_critical_values.csv`, `gw.csv`, `multiple_comparisons.csv`, `mcs.csv`, `encompassing.csv`, "
      "`realtime_combination.csv`, `realtime_lambda_path.csv`, `stability.csv`, `fluctuation.csv`, `fluctuation_path.csv`, `fluctuation_critical_values.csv`, `calibration.csv`, `seed_spread.csv`, `seed_curve.csv`, `bar_column.csv`, `within_day.csv`, `run_info.json`.")
    w(f"- CPU time of the run: {info['cpu_sec'] / 60:.1f} min (one process), wall {info['wall_sec'] / 60:.1f} min.")
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print(f"wrote {OUT / 'SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "run"
    if stage == "gate":
        gate()
    elif stage == "run":
        run(sys.argv[2] if len(sys.argv) > 2 else "auto")
    elif stage == "summary":
        summary()
    else:
        raise SystemExit(__doc__)
