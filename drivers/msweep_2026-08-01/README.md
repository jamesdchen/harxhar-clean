# Multi-horizon + parsimony campaign drivers — 2026-08-01 (session 2)

Session record: `writeup/SESSION_HANDOFF_2026-08-01B.md` (verdicts, laws, frozen list).
Raw verdict logs: `verdicts/` (committed copies of the gitignored `logs/*.log`).
Per-bar predictions: `results/geo_preds/*.npz`, `results/straddle_*.npz`,
`results/sesskern_*.npy` (session-conditional operators).

## Prerequisite: the shared mmap export

All drivers read `results/b2_mmap/{X,y,b,names}.npy` (NOT committed — 2.1GB,
regenerable). Rebuild once from the b2 prep cache:

    python -c "
    import numpy as np
    z = np.load('results/prep_cache_all_features_b2.npz', allow_pickle=True)
    np.save('results/b2_mmap/X.npy', np.ascontiguousarray(z['X'], dtype=np.float64))
    np.save('results/b2_mmap/y.npy', np.asarray(z['y'], dtype=np.float64))
    np.save('results/b2_mmap/b.npy', np.asarray(z['baselines'], dtype=np.float64))
    np.save('results/b2_mmap/names.npy', z['names'])"

(The prep cache itself is rebuilt by `run_geometry_local.prepare_full("all_features")`
with `HAR_BASE=2`, ~30 min.) Every process opens X with `mmap_mode='r'` so
concurrent drivers share one physical copy (Windows page cache).

Run everything from the repo root with `PYTHONPATH=<repo>` and this directory
on `sys.path` (the launchers do both). `OMP_NUM_THREADS=2..3` per process.

## Engine

- `block_ridge.py` — sliding-window ridge via block gram updates; one solve
  per decision; multi-alpha and per-feature-penalty solves share the gram;
  intercept unpenalized (== Ridge(fit_intercept=True)); exact rebuild every
  250 slides. The 10x latency cut vs per-bar rank-1 when predictions are
  only needed at decision points.

## Drivers (env knobs in each docstring)

Multi-step / product (honest labels: window [t-H-W, t-H)):
- `straddle_test.py`, `straddle_alpha.py`, `straddle_v3.py` — daily 16:00
  forecast-vs-implied vs OptionMetrics SPX chain; v3 = VIX/organ arms +
  measured SPY half-spread friction (secid 109820 = SPY: strike vs spot/10).
- `ann_shape.py` — intraday-H battery [24048,72048): inc/full/organ arms,
  ALPHA_SWEEP + INC_SHAPE modes. THE product-model harness.
- `product_tune.py` — 2-block ridge grid (light HAR / heavy exog) + organ
  + VIX ablation + window probe.
- `hybrid_daily.py`, `hybrid_intraday.py` — the Hawkes dictionary ported to
  product horizons (mixture cores, sliding probe, USE_ANN / USE_TEX knobs).

h=1 battery (screening chunk / tile 2 via PB_START/PB_END):
- `parsimony_battery.py` — the arm language (PB_ARMS): alpha/K/cadence/pool/
  proj/prune/rawcells/texture/products/panel arms. See parser at bottom.
- `causal_rawc.py` — per-segment causal cell re-selection (the honest rawc).
- `tile_corr.py` / `rff_corr.py` / `rawc_corr.py` / `hybrid_corr.py` /
  `hybrid_corr_dict.py`* — correction+stack variants (kNN / RFF / bases).
- `pb_enet.py`, `pb_tree.py`, `pb_tree_fresh.py`, `pb_ebm_resid.py`,
  `pb_rf_resid.py` — penalty-family and tree-reader arms (all dead ends,
  verdicts in the handoff).
- `enet_zeros.py`, `pcovr_loadings.py` / `pcovr_loadings80.py` — the dead-
  family extraction and the correction-loading (unspanned directions) dumps.
- `session_kernels.py` — session-conditional operator analysis (the
  regimes-as-amplitude-motion figure data; writes results/sesskern_*.npy).

*hybrid_corr_dict.py's "vs raw-query" DM printed against the wrong file;
the correct comparison is in the handoff.

`launch_*.sh` — the wave launchers (use these, not inline compound
commands: markers written only on success).
