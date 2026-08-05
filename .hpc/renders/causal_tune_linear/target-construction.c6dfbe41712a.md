<!-- hpc-render audit_id: causal_tune_linear -->
<!-- hpc-render section: target-construction -->
<!-- hpc-render section_sha: a386f3d29c4d49a9cc968b7c0fe11c4bda586fb0ce680171ce0c02b3931034a9 -->
<!-- hpc-render view_sha: c6dfbe41712a3b6315afd2cbd1cd2a98c9faca1ec1482475628725631736fb79 -->

## section: target-construction  [tier: human_required]

- classification: modified
- section_sha: a386f3d29c4d49a9cc968b7c0fe11c4bda586fb0ce680171ce0c02b3931034a9
- view_sha: c6dfbe41712a3b6315afd2cbd1cd2a98c9faca1ec1482475628725631736fb79

### diff-from-template

```diff
--- template:target-construction
+++ source:target-construction
@@ -1,9 +1,13 @@
 # %%
 # hpc-audit-section: target-construction
-# [CONTRACT — replace with your lab's cell.] Construct the prediction target by
-# CALLING the production target transform — the exact invariant transform used in
-# deployment, never re-derived here. If the transform also yields a reconstruction
-# baseline (e.g. a scale factor to return to raw units), keep it for the
-# raw-scale reconstruction the interface contract needs. Evidence: show the
-# non-null count of the constructed target.
+# [PINNED — copy verbatim; auto-clears. Mirrors the production call
+# (src/backtest/executor.py prepare step): the invariant transform CALLED,
+# never re-derived — diurnal -> sqrt -> winsorize, is_target=True; the
+# multiplicative baseline kept for raw-space reconstruction.]
+from src.features.transforms.target import robust_transform
 
+adj_rv, rv_baseline = robust_transform(df, "RV", is_target=True)
+df["adj_RV"] = adj_rv
+df["baseline"] = rv_baseline
+print(f"target adj_RV: non-null={int(adj_rv.notna().sum())} (diurnal->sqrt->winsorize, is_target=True)")
+
```

### assertions

(none declared)

### lint flags

(none)
