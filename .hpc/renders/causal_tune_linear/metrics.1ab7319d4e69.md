<!-- hpc-render audit_id: causal_tune_linear -->
<!-- hpc-render section: metrics -->
<!-- hpc-render section_sha: a3dd19072ce69eecb514723082d4645afd90cbd0e996c7f3543065d5cc5c457b -->
<!-- hpc-render view_sha: 1ab7319d4e69debe9956cb1a4585da4a61e6f8983bf5d6b76a644b5b84ef054d -->

## section: metrics  [tier: human_required]

- classification: modified
- section_sha: a3dd19072ce69eecb514723082d4645afd90cbd0e996c7f3543065d5cc5c457b
- view_sha: 1ab7319d4e69debe9956cb1a4585da4a61e6f8983bf5d6b76a644b5b84ef054d

### diff-from-template

```diff
--- template:metrics
+++ source:metrics
@@ -1,13 +1,12 @@
 # %%
 # hpc-audit-section: metrics
-# [CONTRACT — replace with your lab's cell.] Compute the claimed units with the
-# lab's metrics MODULE, never inline arithmetic — the same functions production
-# scores with, so the audit and the deployment measure identically.
-#
-# DEATH BY CONSTRUCTION: the terminal line below references the three interface
-# arrays the VARIABLE sections above must define. An unfilled or half-filled
-# draft NameErrors HERE — deliberately — so it can never emit a vacuous report.
-# Keep a terminal reference to all three when you replace this cell with your
-# metrics call.
-_ = (pred_raw, true_raw, baseline_pred_raw)  # noqa: F821 — unfilled draft dies here (NameError by construction)
+# [PINNED — copy verbatim; auto-clears GIVEN the interface contract.
+# Claimed units computed by src/evaluation/metrics.py, never inline:
+# QLIKE raw-space (primary), hmse/oos_r2 (level-robust), MZ calibration.
+# A NameError below means the variable sections above were not filled —
+# DELIBERATE: an unfilled draft must die here, never emit a vacuous report.]
+from src.evaluation.metrics import forecast_metrics
 
+report = forecast_metrics(pred_raw, true_raw, benchmark=baseline_pred_raw)
+print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in report.items()})
+
```

### assertions

(none declared)

### lint flags

(none)
