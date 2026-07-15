<!-- hpc-render audit_id: causal_tune_linear -->
<!-- hpc-render section: feature-construction -->
<!-- hpc-render section_sha: c1176a6cf68c1624958226b094d2b65eb53bc50e57eb6e6a7932c1f040cc110e -->
<!-- hpc-render view_sha: fc2e841f49c5f49c81628fb54adaa2e674b6703a8afd049f140da4b52acf2750 -->

## section: feature-construction  [tier: human_required]

- classification: modified
- section_sha: c1176a6cf68c1624958226b094d2b65eb53bc50e57eb6e6a7932c1f040cc110e
- view_sha: fc2e841f49c5f49c81628fb54adaa2e674b6703a8afd049f140da4b52acf2750

### diff-from-template

```diff
--- template:feature-construction
+++ source:feature-construction
@@ -1,11 +1,351 @@
 # %%
 # hpc-audit-section: feature-construction
-# [VARIABLE — REPLACE this cell body; it will diff and route to human
-# sign-off. For run #10: ONE existing exog family (e.g. the market_ew
-# moment columns), transformed the production way (robust_transform per
-# column, diurnal_mode as configured), assembled with the horizon-1 shift
-# (apply_horizon_shift — features at t, target at t+1; show the shift).
-# A candidate walk-forward forecast comes from the pinned engine
-# (src/backtest/multi_stage.py::MultiStageBacktest) — no hand-rolled loop.
-# END STATE: contributes to pred_raw / true_raw per the header contract.]
-
+# [VARIABLE — the causally-tuned CANDIDATE arms, one estimator kind per arm,
+# run through the PRODUCTION scaffold: src.backtest.executor.run_executor
+# does the data prep (load_and_transform, HAR + calendar build, burn-in
+# drop, horizon shift, scale guards + prescale, Duan raw-space
+# reconstruction, results table) with the EXACT invariants of
+# src.models.ridge.run — nothing re-derived here. The pinned cells above
+# demonstrate the loader + target; run_executor re-runs that same prep
+# internally (the same functions). The ONLY new code: RollingTunedLinear, a
+# fit_predict regressor exposing the engine's rank-1 rolling protocol
+# (init_window + roll + solve) so MultiStageBacktest's incremental fast path
+# drives it — the verified rolling_linear_tune machinery SPECIALIZED to one
+# estimator per arm. Between tunings the solution slides WARM every bar:
+# reclasso / reclasticnet through TWO src.models.reclasso_har.enet_online
+# calls (Garrigues online homotopy: add the entering row, remove the leaving
+# row; bit-exact, no fallback), ridge through the Sherman-Morrison rank-1
+# update of the ridged inverse. Tuning is PERIODIC AND CAUSAL: every TUNE_PER
+# solves, a leakage-clean forward split of the CURRENT training window
+# (forward_window_split: fit block, EMBARGO gap, VAL_TAIL holdout tail —
+# only rows inside the training window are ever read) scores the estimator's
+# LOG-SCALE alpha grid — ridge closed-form, reclasso / reclasticnet via the
+# FWL batch seed (enet_coef) — and the argmin alpha holds until the next
+# tuning, whose COLD RESEED is also the exact re-anchor bounding float
+# drift. The intercept rides as a locked, unpenalized augmented column.
+# IDENTIFIABILITY MASK (the lam2=0 pure-lasso singularity fix): at every
+# tune boundary — where the cold reseed already re-anchors — non-locked
+# columns that are CONSTANT within the current training window, or exact
+# byte-duplicates of an earlier kept column, are masked: zeroed in the fit
+# design and in the rank-1 update rows until the next tune re-evaluates the
+# mask from the raw ring. Availability-indicator columns of a late-start
+# source collapse to ONE IDENTICAL step function in windows straddling the
+# source's go-live bar; with lam2=0 (the reclasso arm) that identical block
+# makes the active-set subproblem singular and the warm homotopy can ride
+# the non-unique branch to overflow (the 8 blown rolling_linear_tune chunks,
+# QLIKE 48-178, all at exog data-start boundaries; float-path dependent —
+# CARC blew up, local did not, same data bit-exact). Masking is
+# THRESHOLD-FREE and SCALE-FREE ("has dispersion in window and is not a
+# copy"); masked columns keep coefficient 0 so predictions are unaffected
+# except through conditioning (validated locally: all 8 chunks back to
+# healthy 0.09-0.14 QLIKE, delta +0.000025 on a healthy control chunk).
+# EMBARGO=25 covers the har_ma_25 window; longer HAR lags (125+) cannot be
+# embargoed inside a 500-bar window — stated honestly, they are slow-moving
+# levels. PARALLEL: cluster map-reduce — tasks map over bucket x estimator x
+# chunk (EXOG_BUCKET + ESTIMATOR + the SliceSpec start, end, halo fed as
+# HPC_KW_* env into run_executor's chunk seam; the training window +
+# VAL_TAIL holdout is the warm-up halo, replayed without emitting; prescale
+# keeps chunks fungible), the framework's chunk-reduce combines the
+# per-chunk pieces save_chunk_reduce writes. Output tables land under
+# RESULTS_ROOT — $HPC_RESULT_DIR under the dispatcher, a write target the
+# executes-live lint cannot verify (no output_roots are recorded for this
+# audit; disclosed, not masked).
+# END STATE: per-bucket-per-estimator executor results tables + selection
+# traces in `arm_results` for `baseline`.]
+import numpy as np
+import pandas as pd
+
+from src.models.reclasso_har import enet_coef, enet_online, forward_window_split
+from src.backtest.executor import run_executor
+from src.backtest.multi_stage import MultiStageBacktest
+from src.data.loading import SUBGROUPS, get_bucket
+from src.features.transforms.residualizer import IdentityResidualizer
+
+HORIZON = 1
+TRAIN_WIN = 500        # deployed default (src.models.ridge.run train_window=500)
+REFIT_FREQUENCY = 1    # coefficients re-solved EVERY BAR (rank-1 fast path)
+TUNE_PER = 250         # penalty re-selected every 250 solves (the periodic cadence)
+VAL_TAIL = 125         # forward validation holdout inside each training window
+EMBARGO = 25           # kills har_ma_25 fit-to-val bleed (see section note)
+SEED = 42
+
+# One estimator kind per arm; alpha grids on a LOG SCALE (sklearn units:
+# Ridge(alpha=...) for ridge, ElasticNet(alpha=..., l1_ratio=...) for the
+# recursive lasso / elastic net).
+ESTIMATOR_GRIDS: dict = {
+    "ridge": [("ridge", float(a), 0.0) for a in np.logspace(-2, 3, 6)],
+    "reclasso": [("lasso", float(a), 1.0) for a in np.logspace(-6, -2, 5)],
+    "reclasticnet": [("enet", float(a), 0.5) for a in np.logspace(-6, -2, 5)],
+}
+
+# MAP-REDUCE seam: the hpc-agent dispatcher fans tasks out over bucket x
+# estimator x chunk — resolve() kwargs arrive as HPC_KW_* env, and start,
+# end, halo are run_executor's SliceSpec chunk fields. NOTE run_executor's
+# train_window is in DAYS: TRAIN_WIN=500 days = 24000 bars, so a mid-series
+# chunk's halo is 24000 warm-up bars (the canary-caught unit fix; the
+# VAL_TAIL holdout lives INSIDE the training window, so the same halo covers
+# it). Each task's save_chunk_reduce piece is combined by the framework's
+# reduce. Local defaults (no env) = whole series, all buckets x estimators.
+def _env(name: str, default: str) -> str:
+    return os.environ.get(f"HPC_KW_{name}", os.environ.get(name, default))
+
+
+EXOG_BUCKET = _env("EXOG_BUCKET", "")
+ESTIMATOR = _env("ESTIMATOR", "")
+START = int(_env("START", "0"))
+END = int(_env("END", "-1"))
+HALO = int(_env("HALO", "0"))
+BUCKETS = [EXOG_BUCKET] if EXOG_BUCKET else list(SUBGROUPS)  # all 9, incl. the
+# empty-exog `baseline` (no-buckets) arm and `all_features`
+ESTIMATORS = [ESTIMATOR] if ESTIMATOR else list(ESTIMATOR_GRIDS)
+CONTRACT_BUCKET = "all_features" if "all_features" in BUCKETS else BUCKETS[0]
+CONTRACT_ESTIMATOR = ESTIMATORS[0]
+
+RESULTS_ROOT = os.path.join(
+    os.environ.get("HPC_RESULT_DIR", "results"), "causal_tune_linear"
+)  # write target ($HPC_RESULT_DIR under the dispatcher)
+
+
+def _batch_theta(Xa, yr, locked, a, l1):
+    """Cold seed: FWL on the locked (intercept) block + enet_coef on the rest
+    — the verified winablate_r1 batch_theta pattern, sklearn ElasticNet units."""
+    exog = ~locked
+    Hh, Ee = Xa[:, locked], Xa[:, exog]
+    Hp = np.linalg.pinv(Hh)
+    bHy, bHE = Hp @ yr, Hp @ Ee
+    Eres, yres = Ee - Hh @ bHE, yr - Hh @ bHy
+    bE = enet_coef(Eres.T @ Eres, Eres.T @ yres, len(yr), a, l1)
+    th = np.zeros(Xa.shape[1])
+    th[locked] = bHy - bHE @ bE
+    th[exog] = bE
+    return th
+
+
+class RollingTunedLinear:
+    """Rolling linear model, ONE estimator kind: periodic causal alpha
+    re-selection with WARM rank-1 per-bar updates between tunings (the
+    verified rolling_linear_tune / winablate_r1 every-bar machinery).
+
+    Implements the MultiStageBacktest incremental protocol (init_window,
+    roll, solve, predict_one). Between tunings every bar slides the solution
+    exactly: reclasso / reclasticnet through TWO enet_online calls
+    (Garrigues warm homotopy — add the entering row, remove the leaving
+    row), ridge through the Sherman-Morrison rank-1 update of the ridged
+    inverse. Every TUNE_PER solves, forward_window_split re-selects alpha
+    from the arm's log-scale grid on the leakage-clean fit-EMBARGO-VAL_TAIL
+    split of the current window (only rows inside the training window are
+    ever read) and the warm state is COLD-RESEEDED on the full window — the
+    exact re-anchor that also bounds float drift. The intercept rides as a
+    locked, unpenalized augmented column.
+
+    IDENTIFIABILITY MASK: each tune recomputes, from the RAW ring, the set
+    of non-locked columns that are constant within the current window or
+    exact byte-duplicates of an earlier kept column, and those columns are
+    zeroed in the fit design and in the rank-1 update rows until the next
+    tune re-evaluates the mask. A masked column's coefficient is 0 by
+    construction (zero column => zero gradient => never enters the active
+    set), so prediction is unaffected except through conditioning; the
+    duplicate-step block that makes the lam2=0 active-set subproblem
+    singular can no longer form.
+    """
+
+    reinit_every = 0  # drift handled by the cold reseed at every tuning
+    grid: list = []   # the active arm's (kind, alpha, l1) log-scale grid
+    trace: list = []  # class-level record of selections (evidence)
+    mask_trace: list = []  # class-level record of masked-column counts (evidence)
+
+    def init_window(self, X_win, y_win):
+        X = np.asarray(X_win, dtype=np.float64)
+        self._X = np.hstack([X, np.ones((X.shape[0], 1))])  # augmented intercept
+        self._y = np.asarray(y_win, dtype=np.float64).copy()
+        self._ptr = 0
+        self._n_solve = 0
+        self._locked = np.zeros(self._X.shape[1], dtype=bool)
+        self._locked[-1] = True  # intercept: unpenalized, never exits the active set
+        self._maskout = np.zeros(self._X.shape[1], dtype=bool)
+        self._tune()
+        return self
+
+    def _window(self):
+        k = self._ptr
+        return (
+            np.vstack([self._X[k:], self._X[:k]]),
+            np.concatenate([self._y[k:], self._y[:k]]),
+        )
+
+    def _recompute_mask(self, Xraw):
+        """Mask non-locked columns that are constant in the window or exact
+        duplicates of an earlier kept column — threshold-free, scale-free."""
+        ncol = Xraw.shape[1]
+        maskout = np.zeros(ncol, dtype=bool)
+        seen: dict = {}
+        for j in range(ncol):
+            if self._locked[j]:
+                continue
+            col = Xraw[:, j]
+            if col.max() == col.min():
+                maskout[j] = True
+                continue
+            key = col.tobytes()
+            if key in seen:
+                maskout[j] = True
+            else:
+                seen[key] = j
+        self._maskout = maskout
+
+    def _masked_window(self):
+        Xa, yr = self._window()  # fresh arrays (vstack/concatenate) — safe to zero
+        Xa[:, self._maskout] = 0.0
+        return Xa, yr
+
+    def _seed(self):
+        """Cold seed (also the exact re-anchor) of the warm state on the current window."""
+        Xa, yr = self._masked_window()
+        tw = len(yr)
+        kind, a, l1 = self.kind_, self.alpha_, self.l1_
+        if kind == "ridge":
+            D = np.where(self._locked, 0.0, a)  # sklearn Ridge units, free intercept
+            G = Xa.T @ Xa
+            G[np.diag_indices_from(G)] += D
+            self._K = np.linalg.inv(G)
+            self._c = Xa.T @ yr
+            self._th = self._K @ self._c
+        else:
+            mu, lam2 = tw * a * l1, tw * a * (1.0 - l1)  # sklearn ElasticNet units
+            self._mu_vec = np.where(self._locked, 0.0, mu)
+            Gr = Xa.T @ Xa
+            ei = np.where(~self._locked)[0]
+            Gr[ei, ei] += lam2
+            self._Gr = Gr
+            self._c = Xa.T @ yr
+            th = _batch_theta(Xa, yr, self._locked, a, l1)
+            self._A = list(np.where((np.abs(th) > 1e-9) | self._locked)[0])
+            self._s = [0.0 if self._locked[j] else float(np.sign(th[j])) for j in self._A]
+            self._th = th
+
+    def roll(self, x_in, y_in, x_out, y_out):
+        ua = np.append(np.asarray(x_in, dtype=np.float64), 1.0)
+        ur = np.append(np.asarray(x_out, dtype=np.float64), 1.0)
+        self._X[self._ptr] = ua  # ring stores RAW rows; the mask re-derives from raw
+        self._y[self._ptr] = float(y_in)
+        self._ptr = (self._ptr + 1) % len(self._y)
+        if self._maskout.any():  # masked columns stay zero in the update rows
+            ua = ua.copy()
+            ur = ur.copy()
+            ua[self._maskout] = 0.0
+            ur[self._maskout] = 0.0
+        if self.kind_ == "ridge":  # Sherman-Morrison: add entering, drop leaving
+            Ku = self._K @ ua
+            self._K -= np.outer(Ku, Ku) / (1.0 + ua @ Ku)
+            self._c += ua * float(y_in)
+            Kv = self._K @ ur
+            self._K += np.outer(Kv, Kv) / (1.0 - ur @ Kv)
+            self._c -= ur * float(y_out)
+        else:  # warm Garrigues homotopy: one rank-1 data change per call
+            self._th, self._A, self._s = enet_online(
+                self._Gr, self._c, self._mu_vec, ua, float(y_in), +1.0,
+                self._th, self._A, self._s, self._locked,
+            )
+            self._th, self._A, self._s = enet_online(
+                self._Gr, self._c, self._mu_vec, ur, float(y_out), -1.0,
+                self._th, self._A, self._s, self._locked,
+            )
+
+    def _tune(self):
+        Xraw, _ = self._window()
+        self._recompute_mask(Xraw)
+        RollingTunedLinear.mask_trace.append(int(self._maskout.sum()))
+        Xa, yr = self._masked_window()
+        n = len(yr)
+        fit_lo, fit_hi, val_lo, val_hi = forward_window_split(n, n, VAL_TAIL, EMBARGO)
+        Xf, yf = Xa[fit_lo:fit_hi], yr[fit_lo:fit_hi]
+        Xv, yv = Xa[val_lo:val_hi], yr[val_lo:val_hi]
+        best = None
+        for kind, a, l1 in RollingTunedLinear.grid:
+            if kind == "ridge":
+                D = np.where(self._locked, 0.0, a)
+                G = Xf.T @ Xf
+                G[np.diag_indices_from(G)] += D
+                th = np.linalg.solve(G, Xf.T @ yf)
+            else:
+                th = _batch_theta(Xf, yf, self._locked, a, l1)
+            mse = float(np.mean((Xv @ th - yv) ** 2))
+            if best is None or mse < best[0]:
+                best = (mse, kind, a, l1)
+        _, self.kind_, self.alpha_, self.l1_ = best
+        RollingTunedLinear.trace.append((self.kind_, self.alpha_, self.l1_))
+        self._seed()
+
+    def solve(self):
+        if self._n_solve > 0 and self._n_solve % TUNE_PER == 0:
+            self._tune()
+        elif self.kind_ == "ridge":
+            self._th = self._K @ self._c
+        self._n_solve += 1
+
+    def predict_one(self, x):
+        return float(np.append(np.asarray(x, dtype=np.float64), 1.0) @ self._th)
+
+
+def fit_predict_lin_tuned(X_chunk, y_chunk, train_win_periods, hyperparams):
+    """Mirrors src.models.ridge.fit_predict_ridge with the rolling-tuned
+    wrapper as the regressor; the engine detects the rank-1 rolling protocol
+    the wrapper exposes and takes its incremental fast path."""
+    RollingTunedLinear.trace = []
+    RollingTunedLinear.mask_trace = []
+    backtest = MultiStageBacktest(
+        residualizer=IdentityResidualizer(),
+        regressor_factory=RollingTunedLinear,
+        refit_frequency=int(hyperparams.get("_refit_frequency", 1)),
+    )
+    return backtest.run(X_chunk, y_chunk, train_win_periods, desc="lin_tuned")
+
+
+arm_results: dict = {}
+for estimator in ESTIMATORS:
+    RollingTunedLinear.grid = ESTIMATOR_GRIDS[estimator]
+    for bucket in BUCKETS:
+        out_csv = os.path.join(RESULTS_ROOT, estimator, bucket, "results.csv")
+        # the run_executor invocation of src.models.ridge.run with ONE
+        # user-ruled deviation: impute_indicate=True + dropna_with_exog=False
+        # — exog NaN are ffilled with availability-INDICATOR features instead
+        # of dropping the leading edge, so every bucket keeps the SAME index
+        # space (the bucket x estimator x chunk grid is bucket-invariant).
+        # Other invariants unchanged: calendar on, diurnal target, winsor
+        # 240, prescale=True (whose scale guards are exactly the
+        # indicator-safe machinery).
+        run_executor(
+            method_name=f"lin_tuned_{estimator}",
+            fit_predict=fit_predict_lin_tuned,
+            hyperparams={"_refit_frequency": REFIT_FREQUENCY},
+            data_path=DATA_PATH,
+            output_file=out_csv,
+            horizon=HORIZON,
+            train_window=TRAIN_WIN,
+            start=START,
+            end=END,
+            halo=HALO,
+            exog_cols=get_bucket(bucket),
+            segment=None,
+            lag_scope="global",
+            add_calendar=True,
+            target_use_diurnal=True,
+            target_winsor_window=240,
+            dropna_with_exog=False,
+            overnight_fill=True,
+            impute_indicate=True,
+            diurnal_mode="divide",
+            prescale=True,
+            seed=SEED,
+        )
+        arm_results[(estimator, bucket)] = dict(
+            results=pd.read_csv(out_csv), trace=list(RollingTunedLinear.trace)
+        )
+        alphas = [a for _, a, _ in arm_results[(estimator, bucket)]["trace"]]
+        masked = RollingTunedLinear.mask_trace
+        print(f"{estimator}/{bucket}: "
+              f"{len(arm_results[(estimator, bucket)]['results'])} OOS rows; "
+              f"{len(alphas)} tunings, alpha path "
+              f"min={min(alphas):.2e} max={max(alphas):.2e}; "
+              f"masked cols per tune min={min(masked)} max={max(masked)}")
+
```

### assertions

(none declared)

### lint flags

(none)
