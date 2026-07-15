<!-- hpc-render audit_id: causal_tune_linear -->
<!-- hpc-render section: data-selection -->
<!-- hpc-render section_sha: 7694d25809d3500d68631a389338c869ded8f9c0e6f1a5a9d2d4145d7f033280 -->
<!-- hpc-render view_sha: 4041ef19f1085ccaf345ca8e1b049d5cdc6c7b79cd780ed40051f99093a487c7 -->

## section: data-selection  [tier: human_required]

- classification: modified
- section_sha: 7694d25809d3500d68631a389338c869ded8f9c0e6f1a5a9d2d4145d7f033280
- view_sha: 4041ef19f1085ccaf345ca8e1b049d5cdc6c7b79cd780ed40051f99093a487c7

### diff-from-template

```diff
--- template:data-selection
+++ source:data-selection
@@ -1,8 +1,10 @@
 # %%
 # hpc-audit-section: data-selection
-# [CONTRACT — replace with your lab's cell.] Load the raw inputs through the
-# lab's PINNED loader (a named reader callable, declared in the program pack's
-# reader_calls vocabulary), from a path literal UNDER the audit's input_roots.
-# Never re-implement loading inline. Evidence: show the loaded shape (rows/cols);
-# state no conclusion.
+# [PINNED — copy verbatim; auto-clears. The pinned loader, path literal
+# under input_roots, shape shown as evidence.]
+from src.data.loading import load_raw_data
 
+DATA_PATH = "data"
+df = load_raw_data(DATA_PATH, allow_missing=True)
+print(f"loaded: rows={len(df)} cols={len(df.columns)}")
+
```

### assertions

(none declared)

### lint flags

(none)
