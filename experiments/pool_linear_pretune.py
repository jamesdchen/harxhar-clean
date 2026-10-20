"""The POOLED linear models (one coefficient vector over all 48 half-hour bars) pre-tuned, backbone 1/100.

User (2026-10-06): "use the pre-tuned version for all linear models."  The 16:00-bar version is
results/linear_pretune_2026-10-06/ (experiments/close_linear_pretune.py).  This is its pooled twin,
the runner the clusters execute (the local scoring and SUMMARY.md: experiments/pool_linear_pretune_report.py).

Model and walk: the spec's own, specs/causal_tune_linear.py with segment ``none`` (the stored pooled arms
pool_<ridge|lasso|enet>_<bucket>): one model over all bars, window 2000 sessions = 96,000 bars, refit
every bar (ridge: Sherman-Morrison rank-one add / drop; lasso / elastic net: two warm Garrigues-El Ghaoui
online-homotopy updates a bar), the identifiability mask and a cold seed at every 250th solve, the penalty
re-chosen there on the window's any-clock validation tail (fit = the first 95,850 bars, a 25-bar embargo,
the last 125 bars), intercept unpenalized, the spec's grids (ridge 1e-2 .. 1e3, lasso / elastic net
1e-6 .. 1e-2, elastic net l1_ratio 0.5).  The spec's estimator section is EXECUTED here read-only
(spec_section) and run through the spec's own run_executor invocation (same data prep, same prescale,
same chunk seam); the only new code is a subclass of the spec's RollingTunedLinear that

  * carries glmnet-style penalty factors pf (penalty on column j = alpha pf_j (l1 |b_j| + (1 - l1)/2 b_j^2)):
    pf = r on the HAR + calendar backbone (the 34 columns of the pooled baseline design: the target's HAR
    ladder har_ma_*, its session-edge interactions har_ma_*_x_open / _x_close, the calendar and expiry
    columns), pf = 1 on every exogenous column; r = 1 executes exactly the spec's floating-point
    operations (a * 1.0 == a), so arm (a) is the spec; the batch elastic net with pf != 1 is solved by
    column scaling and certified by the unscaled problem's KKT conditions (repaired by support changes,
    else coordinate descent + an exact KKT solve: the C port of experiments/close_exogpen_kernel.c);
  * records, at every re-choice, the validation-tail residuals of every grid alpha (the light-retune
    margin rule of arm f is decided from them, offline);
  * or, given a schedule {block: alpha}, skips the re-choice and seeds the scheduled alpha (arm f's walk).

A block's forecasts depend only on its alpha and the window (cold seed at the block start), so a chunk
that starts at a re-choice boundary with a 96,000-bar halo reproduces the whole-series run (the chunk seam
the stored pooled campaigns gated bit for bit).

Modes  (python experiments/pool_linear_pretune.py <mode> ...; root = the deployment root, data = ./data)
  walk     --bucket B --est E --r R [--sched causal|<json {block: alpha}>] --start S --end E --halo H --out D
           D/causal_tune_linear/<est>/<bucket>/results.csv (the spec's table), results_rth.csv (10:00-16:00),
           trace.npz (alpha, mask count, validation residuals of every grid alpha at each re-choice),
           run.json (seconds, host, numpy).
  pretune  --bucket B --est E --r R --start S --halo H --out D
           4 folds x 12,000 bars (250 sessions) tiling the 48,000 bars before row S (the first bar of the
           first trade day), each fit on the 96,000 bars before a 1,200-bar (25-session) embargo -- the
           16:00-bar study's fold scheme in bars; every grid alpha by the spec's batch solution on the fit
           block (identifiability mask from that block); validation residuals of every row stored (fold.npz).
  dates    --out D: the row dates of the pooled design (baseline bucket; every bucket shares the index
           space: impute_indicate), dates.csv.

--rerun (2026-10-19, user: the pooled linear rerun, results/pooled_linear_rerun_2026-10-19_48/; default off,
every path above unchanged) switches every mode to the rerun design:
  * HAR ladder base 5, k = 1, 5, 25, 125, 625, 3125 (HPC_KW_HAR_BASE=5, asserted; author 2026-10-20: "base 5
    and FOMC included", base 5 for every model, the OLS included);
    [2026-10-20 (user): parked -- was "HAR ladder base 2, k = 1, 2, 4, ..., 2048 (HPC_KW_HAR_BASE=2; the printed
    run never set it: base 5)"]
  * src/backtest/executor.py DROP_ACTIVITY_MA (E6) and CALENDAR_THROUGH_OVERNIGHT (E9) on: no *_active_ma_*
    columns, calendar DOW_0 .. DOW_4, hour, is_overnight, no HAR x open/close products;
  * all_features WITH the four FOMC series (author 2026-10-20; src/data/loading.py FOMC_FEATURES, FOMC_CALENDAR_FIX
    on; E6 still drops their activity averages; asserted);
    [2026-10-20 (user): parked -- was "all_features without the FOMC series (the paper's text: not in the pooled
    model; src/data/loading.py:136 appends them to ALL_FEATURES)"]
  * the backbone (penalty factor r) is the HAR benchmark set: har_ma_1 .. har_ma_3125 + the 7 calendar columns;
  * --est ols: the spec's ridge path at alpha 0 (the pooled OLS on the HAR benchmark set, a0's successor);
  * --sched may hold one number: that penalty at every re-choice (the frozen walk, no margin re-tune).
  design --part K/N builds the design of the K-th of N contiguous parts of the bucket's raw columns (memory: the
  48-interval all_features design, ~1,050 columns, does not fit 15 GB in one pass); merge --parts D1,D2,.. --out D
  stacks the parts' columns in the one-pass design's order (target HAR, every value ladder, every availability
  ladder, calendar).  Every step of the data prep and of the scaling acts column by column, so the merged matrix is
  the one-pass matrix (gated bit for bit on a small bucket, results/pooled_linear_rerun_2026-10-19_48/SUMMARY.md).
"""

from __future__ import annotations

import os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(
    os.environ.get("PLP_ROOT", str(Path(__file__).resolve().parents[1]))
).resolve()
TW_DAYS = 2000
PERIODS_PER_DAY = 48
W = TW_DAYS * PERIODS_PER_DAY  # 96,000 bars
PRE_FOLDS, PRE_FOLD_SESS, PRE_EMBARGO_SESS = (
    4,
    250,
    25,
)  # the 16:00-bar study's pre-tune, in sessions
BACKBONE = frozenset(
    [f"har_ma_{k}" for k in (1, 5, 25, 125, 625, 3125)]
    + [
        f"har_ma_{k}_x_{e}"
        for k in (1, 5, 25, 125, 625, 3125)
        for e in ("open", "close")
    ]
    + [f"DOW_{d}" for d in range(5)]
    + ["hour", "is_overnight", "is_open", "is_close"]
    + [
        "is_opex",
        "is_opex_week",
        "is_quad_witch",
        "is_rebalance_close",
        "is_month_end",
        "is_quarter_end",
        "days_to_opex",
    ]
)
# 2026-10-19 (user): the rerun's backbone, the HAR benchmark set of the rerun design (--rerun)
# 2026-10-20 (user): base 5.  Parked -- was base 2:
# HAR_LADDER_RERUN = [2**j for j in range(12)]  # 1 .. 2048 (resolve_har_lags(base=2), cap 3125)
HAR_LADDER_RERUN = [5**j for j in range(6)]  # 1 .. 3125 (resolve_har_lags(base=5))
BACKBONE_RERUN = frozenset(
    [f"har_ma_{k}" for k in HAR_LADDER_RERUN]
    + [f"DOW_{d}" for d in range(5)]
    + ["hour", "is_overnight"]
)
RERUN = False  # set by main from --rerun
PART: tuple[int, int] | None = None  # design --part K/N (1-based)
_EXOG = re.compile(r"_ma_\d+$")
RTH = {f"{h:02d}:{m:02d}" for h in range(10, 17) for m in (0, 30)} - {
    "16:30"
}  # 13 bars, 10:00 .. 16:00


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def backbone_mask(names: list[str]) -> np.ndarray:
    """True on the HAR + calendar backbone (the pooled baseline design's 34 columns)."""
    bb = np.array(
        [not (_EXOG.search(n) and not n.startswith("har_ma_")) for n in names]
    )
    got = {n for n, b in zip(names, bb) if b}
    want = BACKBONE_RERUN if RERUN else BACKBONE
    assert got == want, sorted(got ^ want)
    return bb


# ============================================================================ the spec, read-only
def spec_section(est: str, bucket: str, start: int, end: int, halo: int) -> dict:
    """Execute specs/causal_tune_linear.py from its estimator section's first import to just before the
    arm loop (constants, RollingTunedLinear, fit_predict_lin_tuned), with the pooled arm's environment."""
    os.environ.update(
        {
            "HPC_KW_SEGMENT": "none",
            "HPC_KW_LAG_SCOPE": "global",
            "HPC_KW_ESTIMATOR": est,
            "HPC_KW_EXOG_BUCKET": bucket,
            "HPC_KW_TRAIN_WIN": str(TW_DAYS),
            "HPC_KW_START": str(start),
            "HPC_KW_END": str(end),
            "HPC_KW_HALO": str(halo),
        }
    )
    if RERUN:
        # 2026-10-20 (user): base 5; parked -- was os.environ["HPC_KW_HAR_BASE"] = "2"
        os.environ["HPC_KW_HAR_BASE"] = "5"
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    src = (ROOT / "specs" / "causal_tune_linear.py").read_text(encoding="utf-8")
    a = src.index(
        "import numpy as np\nimport pandas as pd\n\nfrom src.models.reclasso_har"
    )
    b = src.index("arm_results: dict = {}")
    ns: dict = {"__name__": "causal_tune_linear_section", "os": os}
    exec(compile(src[a:b], str(ROOT / "specs" / "causal_tune_linear.py"), "exec"), ns)
    # the pooled OLS (--est ols): the spec's ridge path at alpha 0 (G = X'X, intercept free)
    ns["ESTIMATOR_GRIDS"]["ols"] = [("ridge", 0.0, 0.0)]
    if RERUN:
        import src.backtest.executor as EX
        import src.data.loading as LD

        EX.DROP_ACTIVITY_MA = True
        EX.CALENDAR_THROUGH_OVERNIGHT = True
        assert ns["HAR_LAGS"] == HAR_LADDER_RERUN == [1, 5, 25, 125, 625, 3125], ns[
            "HAR_LAGS"
        ]
        # 2026-10-20 (user): the FOMC series stay in all_features, with the calendar fix
        assert set(LD.FOMC_FEATURES) <= set(ns["get_bucket"]("all_features")), "FOMC"
        assert LD.FOMC_CALENDAR_FIX, "FOMC_CALENDAR_FIX off"
        # 2026-10-20 (user): parked -- the FOMC series left all_features:
        # gb = ns["get_bucket"]
        #
        # def get_bucket_rerun(name: str) -> list[str]:
        #     cols = gb(name)
        #     return (
        #         [c for c in cols if c not in LD.FOMC_FEATURES]
        #         if name == "all_features"
        #         else cols
        #     )
        #
        # ns["get_bucket"] = get_bucket_rerun
    if PART is not None:
        gb_p = ns["get_bucket"]

        def get_bucket_part(name: str) -> list[str]:
            cols = gb_p(name)
            k, n = PART
            return [c for i, c in enumerate(cols) if i * n // len(cols) == k - 1]

        ns["get_bucket"] = get_bucket_part
    assert (
        ns["SEGMENT"] is None
        and ns["LAG_SCOPE"] == "global"
        and ns["TRAIN_WIN"] == TW_DAYS
    )
    assert (ns["TUNE_PER"], ns["VAL_TAIL"], ns["EMBARGO"], ns["REFIT_FREQUENCY"]) == (
        250,
        125,
        25,
        1,
    )
    return ns


def run_spec_executor(
    ns: dict,
    est: str,
    bucket: str,
    fit_predict,
    out_csv: str,
    start: int,
    end: int,
    halo: int,
    data_path: str,
) -> None:
    """The spec's run_executor invocation (its arm loop), line for line, with another fit_predict."""
    ns["run_executor"](
        method_name=f"lin_tuned_{est}",
        fit_predict=fit_predict,
        hyperparams={"_refit_frequency": ns["REFIT_FREQUENCY"]},
        data_path=data_path,
        output_file=out_csv,
        horizon=ns["HORIZON"],
        train_window=ns["TRAIN_WIN"],
        start=start,
        end=end,
        halo=halo,
        exog_cols=ns["get_bucket"](bucket),
        segment=ns["SEGMENT"],
        lag_scope=ns["LAG_SCOPE"],
        har_lags=ns["HAR_LAGS"],
        add_calendar=True,
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=False,
        overnight_fill=True,
        impute_indicate=True,
        diurnal_mode="divide",
        prescale=True,
        seed=ns["SEED"],
    )


# ============================================================================ penalty factors
REPAIR = {"kkt_ok": 0, "active_repair": 0, "cd_polish": 0, "fail": 0}


def _kkt_ok(G, c, mu, lam2, b, tol) -> bool:
    r = c - G @ b - lam2 * b
    on = b != 0.0
    if np.any(np.abs(r[on] - mu[on] * np.sign(b[on])) > tol):
        return False
    return not np.any(np.abs(r[~on]) > mu[~on] + tol)


def _active_repair(G, c, mu, lam2, b, tol):
    """Support changes: solve the KKT system on the support with its signs, drop sign flips, add the
    worst violator, until every condition holds (close_exogpen_kernel.c active_repair)."""
    act = list(np.flatnonzero(b != 0.0))
    sg = list(np.sign(b[act]))
    m = len(c)
    for _ in range(200):
        if act:
            A = np.array(act)
            M = G[np.ix_(A, A)] + np.diag(lam2[A])
            try:
                x = np.linalg.solve(M, c[A] - mu[A] * np.array(sg))
            except np.linalg.LinAlgError:
                return None
            keep = np.sign(x) == np.array(sg)
            if not keep.all():
                act = [a for a, k in zip(act, keep) if k]
                sg = [s for s, k in zip(sg, keep) if k]
                continue
        bt = np.zeros(m)
        if act:
            bt[np.array(act)] = x
        r = c - G @ bt
        v = np.abs(r) - mu
        v[bt != 0.0] = -np.inf
        jw = int(np.argmax(v))
        if v[jw] <= tol:
            return bt if _kkt_ok(G, c, mu, lam2, bt, tol) else None
        act.append(jw)
        sg.append(1.0 if r[jw] > 0 else -1.0)
    return None


def _cd_polish(G, c, mu, lam2, b, tol):
    """Covariance-form coordinate descent from b, then the exact KKT solve on its support."""
    b = b.copy()
    g = c - G @ b
    ctol = 1e-5 * max(float(np.linalg.norm(c)), 1.0)
    dg = np.diag(G).copy()
    for _ in range(8):
        for _sweep in range(20000):
            big = 0.0
            for j in range(len(c)):
                den = dg[j] + lam2[j]
                if den <= 0.0:
                    continue
                z = g[j] + dg[j] * b[j]
                nb = (
                    (z - mu[j]) / den
                    if z > mu[j]
                    else (z + mu[j]) / den
                    if z < -mu[j]
                    else 0.0
                )
                d = nb - b[j]
                if d != 0.0:
                    b[j] = nb
                    g -= G[j] * d
                    big = max(big, abs(d) * np.sqrt(dg[j]))
            if big < ctol:
                break
        A = np.flatnonzero((b != 0.0) | (mu == 0.0))
        if len(A):
            M = G[np.ix_(A, A)] + np.diag(lam2[A])
            try:
                x = np.linalg.solve(M, c[A] - mu[A] * np.sign(b[A]))
            except np.linalg.LinAlgError:
                x = None
            if x is not None and np.all((np.sign(x) == np.sign(b[A])) | (mu[A] == 0.0)):
                bt = np.zeros(len(c))
                bt[A] = x
                if _kkt_ok(G, c, mu, lam2, bt, tol):
                    return bt
        ctol *= 1e-2
    return None


def enet_coef_pf(enet_coef, lasso_homotopy, g, c, n, alpha, l1, pf) -> np.ndarray:
    """reclasso_har.enet_coef with penalty factors; pf all 1 -> exactly enet_coef."""
    if np.all(pf == 1.0):
        return enet_coef(g, c, n, alpha, l1)
    lam2 = n * alpha * (1.0 - l1)
    mu = n * alpha * l1
    d = 1.0 / pf
    gr = g * np.outer(d, d)
    gr[np.diag_indices_from(gr)] += lam2 * d
    th, _, _ = lasso_homotopy(gr, c * d, mu)
    b = th * d
    muv, l2v = mu * pf, lam2 * pf
    tol = 1e-9 * max(float(np.abs(c).max()), 1.0)
    if _kkt_ok(g, c, muv, l2v, b, tol):
        REPAIR["kkt_ok"] += 1
        return b
    bt = _active_repair(g, c, muv, l2v, b, tol)
    if bt is not None:
        REPAIR["active_repair"] += 1
        return bt
    bt = _cd_polish(g, c, muv, l2v, b, tol)
    if bt is not None:
        REPAIR["cd_polish"] += 1
        return bt
    REPAIR["fail"] += 1
    return b


def batch_theta_pf(ns: dict, Xa, yr, locked, a, l1, pf) -> np.ndarray:
    """The spec's _batch_theta (FWL on the locked block + the batch elastic net on the rest) with
    penalty factors; pf all 1 -> the spec's floating-point operations exactly."""
    exog = ~locked
    Hh, Ee = Xa[:, locked], Xa[:, exog]
    Hp = np.linalg.pinv(Hh)
    bHy, bHE = Hp @ yr, Hp @ Ee
    Eres, yres = Ee - Hh @ bHE, yr - Hh @ bHy
    pfE = pf[exog]
    if np.all(pfE == 1.0):
        bE = ns["enet_coef"](Eres.T @ Eres, Eres.T @ yres, len(yr), a, l1)
    else:
        import src.models.reclasso_har as RH

        bE = enet_coef_pf(
            ns["enet_coef"],
            RH.lasso_homotopy,
            Eres.T @ Eres,
            Eres.T @ yres,
            len(yr),
            a,
            l1,
            pfE,
        )
    th = np.zeros(Xa.shape[1])
    th[locked] = bHy - bHE @ bE
    th[exog] = bE
    return th


# ============================================================================ the walk
class ConstSched(dict):
    """A schedule with one penalty for every block (--sched holding one number)."""

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def __missing__(self, key):
        return self.alpha


def make_class(ns: dict, r: float, sched: dict | None, g0: int, rec: list):
    """The spec's RollingTunedLinear with penalty factors, the residual record, and an optional schedule."""
    Base = ns["RollingTunedLinear"]
    fws = ns["forward_window_split"]
    VAL_TAIL, EMBARGO, TUNE_PER = ns["VAL_TAIL"], ns["EMBARGO"], ns["TUNE_PER"]

    class PFTuned(Base):
        bb = None  # bool over the feature columns, set by fit_predict from the column names

        def init_window(self, X_win, y_win):
            pf = np.ones(np.asarray(X_win).shape[1] + 1)
            pf[:-1][self.bb] = r
            self._pf = pf
            return super().init_window(X_win, y_win)

        def _seed(self):
            """The spec's _seed with alpha * pf_j in place of alpha."""
            Xa, yr = self._masked_window()
            tw = len(yr)
            kind, a, l1 = self.kind_, self.alpha_, self.l1_
            pf = self._pf
            if kind == "ridge":
                D = np.where(self._locked, 0.0, a * pf)
                G = Xa.T @ Xa
                G[np.diag_indices_from(G)] += D
                self._K = np.linalg.inv(G)
                self._c = Xa.T @ yr
                self._th = self._K @ self._c
            else:
                mu, lam2 = tw * a * l1, tw * a * (1.0 - l1)
                self._mu_vec = np.where(self._locked, 0.0, mu * pf)
                Gr = Xa.T @ Xa
                ei = np.where(~self._locked)[0]
                Gr[ei, ei] += lam2 * pf[ei]
                self._Gr = Gr
                self._c = Xa.T @ yr
                th = batch_theta_pf(ns, Xa, yr, self._locked, a, l1, pf)
                self._A = list(np.where((np.abs(th) > 1e-9) | self._locked)[0])
                self._s = [
                    0.0 if self._locked[j] else float(np.sign(th[j])) for j in self._A
                ]
                self._th = th

        def _tune(self):
            """The spec's _tune (mask, forward split, grid argmin of the tail MSE, cold seed) with
            penalty factors and the tail residuals of every grid alpha recorded; with a schedule, the
            scheduled alpha is seeded instead of the argmin (no grid fits)."""
            Xraw, _ = self._window()
            self._recompute_mask(Xraw)
            Base.mask_trace.append(int(self._maskout.sum()))
            blk = (g0 + self._n_solve) // TUNE_PER
            t0 = time.process_time()
            if sched is None:
                Xa, yr = self._masked_window()
                n = len(yr)
                fit_lo, fit_hi, val_lo, val_hi = fws(n, n, VAL_TAIL, EMBARGO)
                Xf, yf = Xa[fit_lo:fit_hi], yr[fit_lo:fit_hi]
                Xv, yv = Xa[val_lo:val_hi], yr[val_lo:val_hi]
                best = None
                res = []
                pf = self._pf
                for kind, a, l1 in Base.grid:
                    if kind == "ridge":
                        D = np.where(self._locked, 0.0, a * pf)
                        G = Xf.T @ Xf
                        G[np.diag_indices_from(G)] += D
                        th = np.linalg.solve(G, Xf.T @ yf)
                    else:
                        th = batch_theta_pf(ns, Xf, yf, self._locked, a, l1, pf)
                    res.append(Xv @ th - yv)
                    mse = float(np.mean((Xv @ th - yv) ** 2))
                    if best is None or mse < best[0]:
                        best = (mse, kind, a, l1)
                _, self.kind_, self.alpha_, self.l1_ = best
                resid = np.array(res)
            else:
                a_s = float(sched[blk])
                hit = [g for g in Base.grid if g[1] == a_s]
                assert len(hit) == 1, (blk, a_s)
                self.kind_, self.alpha_, self.l1_ = hit[0]
                resid = None
            Base.trace.append((self.kind_, self.alpha_, self.l1_))
            self._seed()
            rec.append(
                dict(
                    block=int(blk),
                    solve=int(g0 + self._n_solve),
                    alpha=float(self.alpha_),
                    masked=int(self._maskout.sum()),
                    resid=resid,
                    cpu=time.process_time() - t0,
                )
            )

    return PFTuned


def design(a) -> None:
    """Design cache: the spec's run_executor on the whole series with a no-op walk, keeping what its
    _backtest_and_save hands the walk and the smearing step (the prescaled X, y, the column names, the OOS
    rows' baselines and date labels).  The all_features data prep (load, transforms, whole-series rolling
    robust scaling of ~640 columns) costs most of an hour; every walk and pre-tune then reads the cache."""
    ns = spec_section("ridge", a.bucket, 0, -1, 0)
    import src.backtest.executor as EX

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cap: dict = {}
    orig = EX.apply_duan_smearing

    def spy(preds, y_oos, b_oos):
        cap["b_oos"] = np.array(b_oos, dtype=np.float64)
        return orig(preds, y_oos, b_oos)

    def fit_predict(X_chunk, y_chunk, train_win_periods, hyperparams):
        cap["X"], cap["y"], cap["names"] = (
            X_chunk,
            y_chunk,
            list(hyperparams["_feature_names"]),
        )
        cap["hp"] = {k: v for k, v in hyperparams.items() if k != "_feature_names"}
        return np.zeros(len(X_chunk) - train_win_periods)

    EX.apply_duan_smearing = spy
    try:
        run_spec_executor(
            ns,
            "ridge",
            a.bucket,
            fit_predict,
            str(out / "dummy" / "results.csv"),
            0,
            -1,
            0,
            a.data,
        )
    finally:
        EX.apply_duan_smearing = orig
    d = pd.read_csv(out / "dummy" / "results.csv", usecols=["date"], dtype=str)
    # the matrix as the pipeline holds it: pandas' .values of a float block is column-major, and the walk's
    # BLAS calls on F-ordered operands sum in another order than on a C-ordered copy (last-bit differences
    # that the lasso / elastic net warm path can amplify), so the layout is kept (np.save records it)
    X = cap["X"]
    assert X.dtype == np.float64 and (X.flags.f_contiguous or X.flags.c_contiguous), (
        X.flags
    )
    assert X.shape[0] == W + len(d) and len(cap["b_oos"]) == len(d), (X.shape, len(d))
    np.save(out / "X.npy", X)
    np.save(out / "y.npy", np.ascontiguousarray(cap["y"], dtype=np.float64))
    np.save(out / "b_oos.npy", cap["b_oos"])
    d.to_csv(out / "dates_oos.csv", index=False)
    (out / "design.json").write_text(
        json.dumps(
            dict(
                bucket=a.bucket,
                rerun=RERUN,
                part=list(PART) if PART else None,
                raw_cols=ns["get_bucket"](a.bucket),
                names=cap["names"],
                hp=cap["hp"],
                shape=list(X.shape),
                f_order=bool(X.flags.f_contiguous),
                host=platform.node(),
                numpy=np.__version__,
            )
        )
    )
    (out / "dummy" / "results.csv").unlink()
    say(f"design done: {a.bucket} X {X.shape}")


def merge(a) -> None:
    """Stack the column parts of design --part K/N into the one-pass design (see the module docstring)."""
    parts = [Path(d).resolve() for d in a.parts.split(",")]
    metas = [json.loads((d / "design.json").read_text()) for d in parts]
    assert [m["part"] for m in metas] == [
        [k + 1, len(parts)] for k in range(len(parts))
    ], [m["part"] for m in metas]
    assert len({m["bucket"] for m in metas}) == 1 and all(
        m["rerun"] == RERUN for m in metas
    )
    ref = parts[0]
    y, b = np.load(ref / "y.npy"), np.load(ref / "b_oos.npy")
    d0 = (ref / "dates_oos.csv").read_bytes()
    for d in parts[1:]:  # rows and target are the parts' common columns
        assert np.array_equal(np.load(d / "y.npy"), y) and np.array_equal(
            np.load(d / "b_oos.npy"), b
        )
        assert (d / "dates_oos.csv").read_bytes() == d0
    raw = [c for m in metas for c in m["raw_cols"]]
    common = [
        n for n in metas[0]["names"] if not _EXOG.search(n) or n.startswith("har_ma_")
    ]
    har = [n for n in common if n.startswith("har_ma_")]
    cal = [n for n in common if not n.startswith("har_ma_")]
    for m in metas:
        assert [
            n for n in m["names"] if not _EXOG.search(n) or n.startswith("har_ma_")
        ] == common
    lad = [int(n.rsplit("_", 1)[1]) for n in har]
    exog = {n for m in metas for n in m["names"] if n not in common}
    # load_and_transform's order: every adj_<col>, then every <col>_avail, then every <col>_active
    order = har
    for suf in ("adj_{}", "{}_avail", "{}_active"):
        order += [
            f"{suf.format(c)}_ma_{k}"
            for c in raw
            for k in lad
            if f"{suf.format(c)}_ma_{k}" in exog
        ]
    order += cal
    assert sorted(order) == sorted(set(order)) and set(order) == exog | set(common), (
        "column order"
    )
    src = {}
    for d, m in zip(parts, metas):
        for j, n in enumerate(m["names"]):
            src.setdefault(n, (d, j))
    X = np.empty(
        (len(y), len(order)),
        dtype=np.float64,
        order="F" if metas[0]["f_order"] else "C",
    )
    for d in parts:
        Xp = np.load(d / "X.npy", mmap_mode="r")
        for j, n in enumerate(order):
            if src[n][0] == d:
                X[:, j] = Xp[:, src[n][1]]
        del Xp
    for d, m in zip(parts[1:], metas[1:]):  # the common columns agree across parts
        Xp = np.load(d / "X.npy", mmap_mode="r")
        for n in common:
            assert np.array_equal(Xp[:, m["names"].index(n)], X[:, order.index(n)]), n
        del Xp
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", X)
    np.save(out / "y.npy", y)
    np.save(out / "b_oos.npy", b)
    (out / "dates_oos.csv").write_bytes(d0)
    meta = dict(
        metas[0],
        part=None,
        raw_cols=raw,
        names=order,
        shape=list(X.shape),
        merged_from=[str(d) for d in parts],
    )
    (out / "design.json").write_text(json.dumps(meta))
    say(f"merge done: {meta['bucket']} X {X.shape} from {len(parts)} parts")


def cached_backtest(a, fit_predict, out_csv: Path) -> None:
    """_backtest_and_save's slice, walk and table on the design cache (the same arrays it would build)."""
    import src.backtest.executor as EX

    dz = Path(a.design)
    meta = json.loads((dz / "design.json").read_text())
    assert meta["bucket"] == a.bucket, (meta["bucket"], a.bucket)
    assert meta.get("rerun", False) == RERUN, (meta.get("rerun"), RERUN)
    # the whole matrix in memory, in the pipeline's layout, and the chunk a VIEW of it, as _backtest_and_save
    # slices it (a C-ordered copy is not bit-identical to the spec's path)
    # --rerun: memory-mapped (the same bytes and layout; the concurrent walks share one page-cache copy)
    X = np.load(dz / "X.npy", mmap_mode="r" if RERUN else None)
    assert bool(X.flags.f_contiguous) == meta["f_order"], X.flags
    y = np.load(dz / "y.npy")
    b_oos = np.load(dz / "b_oos.npy")
    dts = pd.read_csv(dz / "dates_oos.csv", dtype=str)["date"].to_numpy()
    load_start = max(0, a.start - a.halo)
    actual_end = len(X) if a.end < 0 else a.end
    X_chunk = X[load_start:actual_end]
    y_chunk = y[load_start:actual_end]
    preds = fit_predict(
        X_chunk, y_chunk, W, dict(meta["hp"], _feature_names=list(meta["names"]))
    )
    y_oos = y_chunk[W:]
    sl = slice(load_start, actual_end - W)  # OOS row i <-> design row W + i
    pred_raw, true_raw = EX.apply_duan_smearing(preds, y_oos, b_oos[sl])
    res = pd.DataFrame(
        {
            "date": dts[sl],
            "horizon": 1,
            "true_adj": y_oos,
            "pred_adj": preds,
            "true_raw": true_raw,
            "pred_raw": pred_raw,
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)


def walk(a) -> None:
    t_wall = time.time()
    ns = spec_section(a.est, a.bucket, a.start, a.end, a.halo)
    g0 = max(0, a.start - a.halo)
    assert g0 % ns["TUNE_PER"] == 0, (
        f"chunk start {a.start} - halo {a.halo} is not a re-choice boundary"
    )
    assert a.halo in (0, W), a.halo
    sched = None
    if a.sched != "causal":
        js = json.loads(Path(a.sched).read_text())
        if isinstance(
            js, (int, float)
        ):  # one penalty at every re-choice (the frozen walk)
            sched = ConstSched(float(js))
        else:
            sched = {int(k): float(v) for k, v in js.items()}
    rec: list = []
    Cls = make_class(ns, a.r, sched, g0, rec)
    Base = ns["RollingTunedLinear"]
    Base.grid = ns["ESTIMATOR_GRIDS"][a.est]
    names_seen: list = []

    def fit_predict(X_chunk, y_chunk, train_win_periods, hyperparams):
        assert train_win_periods == W, train_win_periods
        names = list(hyperparams["_feature_names"])
        names_seen.append(names)
        Cls.bb = backbone_mask(names)
        Base.trace, Base.mask_trace, Base.reseed_trace = [], [], []
        bt = ns["MultiStageBacktest"](
            residualizer=ns["IdentityResidualizer"](),
            regressor_factory=Cls,
            refit_frequency=int(hyperparams.get("_refit_frequency", 1)),
        )
        return bt.run(X_chunk, y_chunk, train_win_periods, desc="lin_tuned")

    out = Path(a.out).resolve()
    out_csv = out / "causal_tune_linear" / a.est / a.bucket / "results.csv"
    t_cpu = time.process_time()
    if a.design:
        cached_backtest(a, fit_predict, out_csv)
    else:
        run_spec_executor(
            ns,
            a.est,
            a.bucket,
            fit_predict,
            str(out_csv),
            a.start,
            a.end,
            a.halo,
            a.data,
        )
    cpu = time.process_time() - t_cpu
    res = pd.read_csv(out_csv, usecols=["date"])
    # the regular-hours rows (bar-end labels 10:00 .. 16:00), byte for byte (no float re-parse)
    nl = bytes([10])
    lines = out_csv.read_bytes().split(nl)
    keep = [lines[0]] + [ln for ln in lines[1:] if ln and ln[11:16].decode() in RTH]
    (out / "results_rth.csv").write_bytes(nl.join(keep) + nl)
    G = len(Base.grid)
    blocks = np.array([x["block"] for x in rec])
    resid = (
        np.stack([x["resid"] for x in rec])
        if sched is None
        else np.zeros((len(rec), G, 0))
    )
    np.savez_compressed(
        out / "trace.npz",
        block=blocks,
        solve=np.array([x["solve"] for x in rec]),
        alpha=np.array([x["alpha"] for x in rec]),
        masked=np.array([x["masked"] for x in rec]),
        tune_cpu=np.array([x["cpu"] for x in rec]),
        resid=resid,
        grid=np.array([g[1] for g in Base.grid]),
        reseed=np.array(Base.reseed_trace, dtype=np.int64) + g0,
        names=np.array(names_seen[0]),
        backbone=backbone_mask(names_seen[0]),
    )
    import numpy as _np

    info = dict(
        mode="walk",
        rerun=RERUN,
        bucket=a.bucket,
        est=a.est,
        r=a.r,
        sched=a.sched,
        start=a.start,
        end=a.end,
        halo=a.halo,
        g0=g0,
        rows=int(len(res)),
        first=str(res["date"].iloc[0]),
        last=str(res["date"].iloc[-1]),
        blocks=[int(blocks.min()), int(blocks.max())],
        n_tunes=len(rec),
        n_cols=len(names_seen[0]),
        n_backbone=int(backbone_mask(names_seen[0]).sum()),
        reseeds=len(Base.reseed_trace),
        repair=dict(REPAIR),
        cpu_sec=cpu,
        wall_sec=time.time() - t_wall,
        host=platform.node(),
        numpy=_np.__version__,
        python=platform.python_version(),
    )
    (out / "run.json").write_text(json.dumps(info, indent=1))
    say(f"walk done: {json.dumps(info)}")


# ============================================================================ the pre-tune
def pretune(a) -> None:
    t_wall = time.time()
    ns = spec_section(a.est, a.bucket, a.start, a.start + 1, a.halo)
    fold_n, emb = PRE_FOLD_SESS * PERIODS_PER_DAY, PRE_EMBARGO_SESS * PERIODS_PER_DAY
    assert a.halo == PRE_FOLDS * fold_n + emb + W, a.halo
    grid = ns["ESTIMATOR_GRIDS"][a.est]
    Base = ns["RollingTunedLinear"]
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    store: dict = {}

    def fit_predict(X_chunk, y_chunk, train_win_periods, hyperparams):
        names = list(hyperparams["_feature_names"])
        bb = backbone_mask(names)
        X = np.asarray(X_chunk, dtype=np.float64)
        y = np.asarray(y_chunk, dtype=np.float64)
        n = len(X) - 1  # the rows before the start row
        Xall = np.hstack([X, np.ones((len(X), 1))])
        locked = np.zeros(Xall.shape[1], dtype=bool)
        locked[-1] = True
        pf = np.ones(Xall.shape[1])
        pf[:-1][bb] = a.r
        e_all, info = [], []
        for f in range(PRE_FOLDS):
            v1 = n - (PRE_FOLDS - 1 - f) * fold_n
            v0 = v1 - fold_n
            f0, f1 = v0 - emb - W, v0 - emb
            assert f0 >= 0
            probe = Base()
            probe._locked = locked
            probe._recompute_mask(
                Xall[f0:f1]
            )  # the spec's identifiability mask, from the fit block
            mk = probe._maskout
            Xf = Xall[f0:f1].copy()
            Xf[:, mk] = 0.0
            Xv = Xall[v0:v1].copy()
            Xv[:, mk] = 0.0
            yf, yv = y[f0:f1], y[v0:v1]
            t0 = time.process_time()
            res = []
            if a.est == "ridge":
                G0 = Xf.T @ Xf
                cf = Xf.T @ yf
                for _, al, _ in grid:
                    G = G0.copy()
                    G[np.diag_indices_from(G)] += np.where(locked, 0.0, al * pf)
                    res.append(Xv @ np.linalg.solve(G, cf) - yv)
            else:
                for _, al, l1 in grid:
                    res.append(Xv @ batch_theta_pf(ns, Xf, yf, locked, al, l1, pf) - yv)
            e_all.append(np.array(res))
            info.append(
                dict(
                    fold=f,
                    fit=[f0, f1],
                    val=[v0, v1],
                    masked=int(mk.sum()),
                    cpu=time.process_time() - t0,
                )
            )
            say(
                f"fold {f}: fit {f0}..{f1} val {v0}..{v1} masked {int(mk.sum())} "
                f"mse {np.round((np.array(res) ** 2).mean(axis=1), 5).tolist()}"
            )
        store.update(resid=np.array(e_all), info=info, names=names, bb=bb)
        return np.zeros(len(X_chunk) - train_win_periods)

    t_cpu = time.process_time()
    if a.design:  # the same rows of the design cache
        dz = Path(a.design)
        meta = json.loads((dz / "design.json").read_text())
        assert meta["bucket"] == a.bucket, (meta["bucket"], a.bucket)
        assert meta.get("rerun", False) == RERUN, (meta.get("rerun"), RERUN)
        sl = slice(a.start - a.halo, a.start + 1)
        fit_predict(
            np.load(dz / "X.npy", mmap_mode="r" if RERUN else None)[
                sl
            ],  # a view, as in the pipeline (--rerun: memory-mapped)
            np.load(dz / "y.npy")[sl],
            W,
            dict(meta["hp"], _feature_names=list(meta["names"])),
        )
    else:
        run_spec_executor(
            ns,
            a.est,
            a.bucket,
            fit_predict,
            str(out / "dummy" / "results.csv"),
            a.start,
            a.start + 1,
            a.halo,
            a.data,
        )
    # global row of local row i: a.start - a.halo + i
    lo = a.start - a.halo
    np.savez_compressed(
        out / "fold.npz",
        resid=store["resid"],
        grid=np.array([g[1] for g in grid]),
        fit=np.array([[x["fit"][0] + lo, x["fit"][1] + lo] for x in store["info"]]),
        val=np.array([[x["val"][0] + lo, x["val"][1] + lo] for x in store["info"]]),
        masked=np.array([x["masked"] for x in store["info"]]),
        names=np.array(store["names"]),
        backbone=store["bb"],
    )
    import numpy as _np

    info = dict(
        mode="pretune",
        rerun=RERUN,
        bucket=a.bucket,
        est=a.est,
        r=a.r,
        start=a.start,
        halo=a.halo,
        folds=store["info"],
        repair=dict(REPAIR),
        cpu_sec=time.process_time() - t_cpu,
        wall_sec=time.time() - t_wall,
        host=platform.node(),
        numpy=_np.__version__,
    )
    (out / "run.json").write_text(json.dumps(info, indent=1))
    say(f"pretune done: {json.dumps(info)}")


def dates(a) -> None:
    """Row dates of the pooled design: the spec's run_executor on baseline with a no-op walk."""
    ns = spec_section("ridge", "baseline", 0, -1, 0)
    out = Path(a.out).resolve()

    def fit_predict(X_chunk, y_chunk, train_win_periods, hyperparams):
        return np.zeros(len(X_chunk) - train_win_periods)

    run_spec_executor(
        ns,
        "ridge",
        "baseline",
        fit_predict,
        str(out / "dates_raw" / "results.csv"),
        0,
        -1,
        0,
        a.data,
    )
    d = pd.read_csv(out / "dates_raw" / "results.csv", usecols=["date"])
    d.insert(0, "row", np.arange(W, W + len(d)))
    d.to_csv(out / "dates.csv", index=False)
    say(f"dates: {len(d)} OOS rows, {d['date'].iloc[0]} .. {d['date'].iloc[-1]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("walk", "pretune", "dates", "design", "merge"))
    ap.add_argument("--bucket", default="baseline")
    ap.add_argument(
        "--est", default="ridge", choices=("ridge", "reclasso", "reclasticnet", "ols")
    )
    ap.add_argument("--r", type=float, default=1.0)
    ap.add_argument("--sched", default="causal")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--halo", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument(
        "--design", default="", help="a design cache directory (mode design's output)"
    )
    ap.add_argument(
        "--rerun",
        action="store_true",
        help="the 2026-10-19 rerun design (module docstring); default: the printed run's",
    )
    ap.add_argument(
        "--part", default="", help="design: K/N, the K-th of N column parts"
    )
    ap.add_argument(
        "--parts",
        default="",
        help="merge: the part design directories, K = 1 .. N in order",
    )
    a = ap.parse_args()
    global RERUN, PART
    RERUN = a.rerun
    if a.part:
        PART = tuple(int(v) for v in a.part.split("/"))
    a.data = (
        str(Path(a.data).resolve()) if a.data != "data" else "data"
    )  # "data": the root's, after chdir
    a.out = str(Path(a.out).resolve())
    if a.sched != "causal":
        a.sched = str(Path(a.sched).resolve())
    if a.design:
        a.design = str(Path(a.design).resolve())
    {
        "walk": walk,
        "pretune": pretune,
        "dates": dates,
        "design": design,
        "merge": merge,
    }[a.mode](a)


if __name__ == "__main__":
    main()
