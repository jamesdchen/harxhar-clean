<!-- hpc-render audit_id: causal_tune_linear -->
<!-- hpc-render section: baseline -->
<!-- hpc-render section_sha: 40026e3e324d010849f4f8e79c205d07a8678b8ede65d26019e30309ad542f73 -->
<!-- hpc-render view_sha: b9272b9ef121a013710f4b23dc42f9ac9b592d3068ccc2b83c92b44700c55ada -->

## section: baseline  [tier: human_required]

- classification: modified
- section_sha: 40026e3e324d010849f4f8e79c205d07a8678b8ede65d26019e30309ad542f73
- view_sha: b9272b9ef121a013710f4b23dc42f9ac9b592d3068ccc2b83c92b44700c55ada

### diff-from-template

```diff
--- template:baseline
+++ source:baseline
@@ -1,11 +1,81 @@
 # %%
 # hpc-audit-section: baseline
-# [CONTRACT — replace with your lab's cell.] Reproduce the deployed baseline LIVE
-# in this run: run the deployed baseline's frozen config through the SAME
-# walk-forward engine and CITE that config's commit sha in the cell. A baseline
-# number is NEVER quoted from results/ — it is recomputed here or it does not
-# count. This is the known-answer check that anchors the whole audit.
-# END STATE: pred_raw, true_raw, baseline_pred_raw all defined — three aligned
-# 1-D raw-scale arrays (the interface contract) — and the baseline's headline
-# metric printed for the known-answer comparison.
+# [VARIABLE-BY-CONTENT, PINNED-BY-IDENTITY — will diff; human sign-off
+# expected. DELIBERATE DEVIATION from the template prose (which cites
+# configs/ridge_market_ew_prod.yaml): the comparator for the causal-tuning
+# question is OLS ON THE NO-BUCKET ARM (user-ruled) — ONE incumbent shared
+# by every bucket x estimator arm: the production fit_predict_ridge at
+# alpha=0.0 (RollingLeastSquares rank-1 is EXACT OLS at zero penalty,
+# per-bar refit) on the empty-exog `baseline` bucket (HAR + calendar only),
+# through the IDENTICAL run_executor invocation, CALLED, never re-derived —
+# reproduced LIVE here, never a number quoted from results tables on disk.
+# The impute_indicate=True + dropna_with_exog=False invariants keep the
+# incumbent on the SAME index space as every tuned arm (bucket-invariant
+# grid), so the per-bar loss series align bar-for-bar.
+# THE CITABLE TABLE: per bucket x estimator, the FULL beat4-writeup spread
+# from src.evaluation.metrics.forecast_metrics (qlike primary, mse / rmse /
+# mae, hmse / hmae, oos_r2 vs the incumbent, MZ calibration mz_beta /
+# mz_r2, n) PLUS the Diebold-Mariano tuned-vs-incumbent test on per-bar
+# QLIKE losses (src.evaluation.diebold_mariano.qlike_per_bar + dm_test,
+# HAC lag auto) — units from src, never inline; the table is persisted to
+# RESULTS_ROOT/metrics_table.csv (per-bar losses live in the executor
+# results tables, kept per the persist-raw-outputs ruling). END STATE:
+# pred_raw, true_raw, baseline_pred_raw — three aligned 1-D raw-variance
+# arrays read from the executor's own Duan-reconstructed results tables —
+# the CONTRACT_ESTIMATOR x CONTRACT_BUCKET arm, per the module header.]
+from src.evaluation.diebold_mariano import dm_test, qlike_per_bar
+from src.evaluation.metrics import forecast_metrics as _fm
+from src.models.ridge import fit_predict_ridge
 
+# alpha=0.0 -> RollingLeastSquares(alpha=0.0): exact OLS, zero penalty
+BASELINE_HP = dict(alpha=0.0, _refit_frequency=1, _incremental=True)
+INCUMBENT_BUCKET = "baseline"  # the empty-exog no-bucket arm (HAR + calendar)
+
+incumbent_csv = os.path.join(RESULTS_ROOT, "incumbent_ols", "results.csv")
+run_executor(
+    method_name="ols_incumbent",
+    fit_predict=fit_predict_ridge,
+    hyperparams=dict(BASELINE_HP),
+    data_path=DATA_PATH,
+    output_file=incumbent_csv,
+    horizon=HORIZON,
+    train_window=TRAIN_WIN,
+    start=START,
+    end=END,
+    halo=HALO,
+    exog_cols=get_bucket(INCUMBENT_BUCKET),
+    segment=None,
+    lag_scope="global",
+    add_calendar=True,
+    target_use_diurnal=True,
+    target_winsor_window=240,
+    dropna_with_exog=False,
+    overnight_fill=True,
+    impute_indicate=True,
+    diurnal_mode="divide",
+    prescale=True,
+    seed=SEED,
+)
+incumbent = pd.read_csv(incumbent_csv)
+p_i = incumbent["pred_raw"].values
+
+rows = []
+for (estimator, bucket), r in arm_results.items():
+    tuned = r["results"]
+    p_t, t_t = tuned["pred_raw"].values, tuned["true_raw"].values
+    m = _fm(p_t, t_t, benchmark=p_i)
+    dm = dm_test(qlike_per_bar(p_t, t_t), qlike_per_bar(p_i, t_t), h=HORIZON)
+    rows.append(dict(estimator=estimator, bucket=bucket, **m,
+                     incumbent_qlike=_fm(p_i, t_t)["qlike"],
+                     dm=dm["dm"], dm_p=dm["p"], dm_mean_diff=dm["mean_diff"],
+                     dm_better=dm.get("better", "")))
+metrics_table = pd.DataFrame(rows)
+os.makedirs(RESULTS_ROOT, exist_ok=True)
+metrics_table.to_csv(os.path.join(RESULTS_ROOT, "metrics_table.csv"), index=False)
+print(metrics_table.to_string(index=False))
+
+pred_raw = arm_results[(CONTRACT_ESTIMATOR, CONTRACT_BUCKET)]["results"]["pred_raw"].values
+true_raw = arm_results[(CONTRACT_ESTIMATOR, CONTRACT_BUCKET)]["results"]["true_raw"].values
+baseline_pred_raw = p_i
+print(f"contract arrays = {CONTRACT_ESTIMATOR} x {CONTRACT_BUCKET} arm vs OLS no-bucket incumbent")
+
```

### assertions

(none declared)

### lint flags

(none)
