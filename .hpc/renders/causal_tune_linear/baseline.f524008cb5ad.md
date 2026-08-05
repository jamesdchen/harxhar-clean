<!-- hpc-render audit_id: causal_tune_linear -->
<!-- hpc-render section: baseline -->
<!-- hpc-render section_sha: c7dc85d7781d4b0bdb5c840798b7d73e1d23345d52f47bb7b18ba469a3bc0127 -->
<!-- hpc-render view_sha: f524008cb5ad32267b2808e4cfc4721860185ffaf38b2507b7dfe96baf886cfe -->

## section: baseline  [tier: human_required]

- classification: modified
- section_sha: c7dc85d7781d4b0bdb5c840798b7d73e1d23345d52f47bb7b18ba469a3bc0127
- view_sha: f524008cb5ad32267b2808e4cfc4721860185ffaf38b2507b7dfe96baf886cfe

### diff-from-template

```diff
--- template:baseline
+++ source:baseline
@@ -1,10 +1,78 @@
 # %%
 # hpc-audit-section: baseline
 # [VARIABLE-BY-CONTENT, PINNED-BY-IDENTITY — will diff; human sign-off
-# expected. The honest baseline reproduced LIVE in this run: the deployed
-# ridge config (configs/ridge_market_ew_prod.yaml — cite its commit sha in
-# the cell), same walk-forward engine, never a number quoted from results/.
-# END STATE: pred_raw, true_raw, baseline_pred_raw all defined — three
-# aligned 1-D raw-variance arrays (the header contract) — and the
-# baseline's QLIKE printed for the known-answer check.]
+# expected. DELIBERATE DEVIATION from the template prose (which cites
+# configs/ridge_market_ew_prod.yaml): the comparator for the
+# causal-tuning question is the FIXED-HP INCUMBENT on the SAME machinery —
+# the production fit_predict_ridge (RollingLeastSquares rank-1,
+# DEFAULT_RIDGE_PARAMS alpha=1.0, per-bar refit) through the IDENTICAL
+# run_executor invocation, CALLED, never re-derived, per bucket (the
+# incumbent is estimator-invariant: one fixed-HP arm per bucket) —
+# reproduced LIVE here, never a number quoted from results tables on disk.
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
+from src.models.ridge import DEFAULT_RIDGE_PARAMS, fit_predict_ridge
 
+BASELINE_HP = dict(DEFAULT_RIDGE_PARAMS, _refit_frequency=1, _incremental=True)
+
+incumbent_results: dict = {}
+for bucket in BUCKETS:
+    out_csv = os.path.join(RESULTS_ROOT, "incumbent_fixed", bucket, "results.csv")
+    run_executor(
+        method_name="ridge",
+        fit_predict=fit_predict_ridge,
+        hyperparams=dict(BASELINE_HP),
+        data_path=DATA_PATH,
+        output_file=out_csv,
+        horizon=HORIZON,
+        train_window=TRAIN_WIN,
+        start=START,
+        end=END,
+        halo=HALO,
+        exog_cols=get_bucket(bucket),
+        segment=None,
+        lag_scope="global",
+        add_calendar=True,
+        target_use_diurnal=True,
+        target_winsor_window=240,
+        dropna_with_exog=False,
+        overnight_fill=True,
+        impute_indicate=True,
+        diurnal_mode="divide",
+        prescale=True,
+        seed=SEED,
+    )
+    incumbent_results[bucket] = pd.read_csv(out_csv)
+
+rows = []
+for (estimator, bucket), r in arm_results.items():
+    tuned, incumbent = r["results"], incumbent_results[bucket]
+    p_t, t_t = tuned["pred_raw"].values, tuned["true_raw"].values
+    p_i = incumbent["pred_raw"].values
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
+baseline_pred_raw = incumbent_results[CONTRACT_BUCKET]["pred_raw"].values
+print(f"contract arrays = {CONTRACT_ESTIMATOR} x {CONTRACT_BUCKET} arm (module header)")
+
```

### assertions

(none declared)

### lint flags

(none)
