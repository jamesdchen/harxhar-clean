# rehaul_fi_pooled13_2026-10-22

Feature importance of the pooled 13-bar linear models on the full predictor set (564 columns), on
`results/pooled_linear_rerun_2026-10-22_13_2012_w1000_latestart_nofomcrel` (LATE_START_WARMUP on, fomc_release dropped,
window 13,000 rows, frozen penalties ridge 1e3 / lasso 1e-3 / elastic net 1e-2 with l1_ratio 0.5, penalty factor 0.01
on the 12 HAR + calendar columns). Note E19: the pooled intraday model with a monthly refit cadence.

- **Re-estimations:** K = 151, one every 21 sessions from 2012-01-03 10:00 (origin = the session's first row) to
  2024-04-30 10:00; blocks of 264–273 rows (76 sessions have fewer than 13 bars); the last block is 1 session. Origins
  in `run_info.json` and `fi_refits.csv`. Neither E19 nor `experiments/rehaul_fi_compute.py` defines "monthly".
- **Coefficients:** the walk's own at each origin, replayed from the last 250-solve tune boundary with the imported
  runner code; every replayed row equals the published `pred_adj` (≤ 8.9e-16).
  - Batch refits at the origins match for ridge (1.8e-12 relative) but not at 31/151 lasso and 4/151 elastic-net
    origins (max 5.4e-3 and 8.5e-5 relative): the batch solutions meet the KKT conditions to 1e-14, the walk's
    warm-homotopy coefficients violate them at exactly those origins (up to 2.1e-3 and 1.0e-2 relative to max|X'y|).
    Using the batch fits instead moves lasso K_A by at most 1 in 7 groups and S_A by at most 1.1e-4.
- **Measures and conventions** as in `experiments/rehaul_fi_compute.py` (NZ_TOL 0; groups = raw series, har_ma,
  calendar; sd = window np.std; ranks by ||β_A||₁; Kendall W with ties):
  - S_A = Σ_k Σ_{j∈A}|β_kj| / Σ_k Σ_j|β_kj|; S^sd_A the same with |β_kj|·sd_kj (as a share, as `make_rehaul_fi.py:560`);
  - K_A = re-estimations with a nonzero coefficient in group A; K_live = re-estimations with at least one unmasked
    column in the group (`fi_groups.csv` also gives K_A/K_live).
  - Differences from the earlier compute: 48 groups (fomc_release dropped; 46 exogenous in 8 categories, plus har_ma and
    calendar), cadence 21 sessions instead of 10, no FI_OOS_START cut (the pre-tune folds end before 2012).
- **Files:** `fi_groups.csv`, `fi_predictors.csv`, `fi_refits.csv`, `fi_time_{ridge,lasso,enet}.csv` (long: origin ×
  group), `fi_coefs_{lasso,ridge,enet}.csv`, `fi_window_sd.csv`, `run_info.json` (design, penalties, origins, gates,
  summaries, timings, code SHAs: ls_wt a707349, main 7ae6b9a).
- **Code:** `code/fi13_fit.py`, `code/fi13_replay.py`, `code/fi13_reduce.py` (they import the runner code of the
  late-start worktree `C:\Users\james\AppData\Local\Temp\ls_wt` read-only; paths inside are absolute).

## Headline numbers

- Lasso top 10 by K_A/K (all 1.000; ties by S_A): har_ma (.285), vix (.091), sumret (.075), calendar (.051), sumret4
  (.038), sumabsret_vwstock (.032), sumautocov (.028), sumvolume (.023), fomc_until_inv (.016), sumret3_ewstock (.010).
  K_A/K saturates at α = 1e-3: 43 of 46 exogenous groups are selected in at least half the re-estimations.
- Ridge top 5: by S_A har_ma .199, sumret .050, vix .039, calendar .032, sumret4 .027; by S^sd_A har_ma .229, sumret
  .050, sumret4 .041, sumabsret_vwstock .034, sumabsret_ewstock .033. Spearman(S_A, S^sd_A): ridge 0.59, lasso 0.90,
  elastic net 0.98.
- HAR-block share (median per re-estimation, raw / sd-scaled): ridge .231 / .241; lasso .337 / .353; elastic net .689.
- Nonzero coefficients per re-estimation: lasso median 91 (65–104; 80 exogenous), 273 of 564 columns ever selected,
  62 in at least half; ridge 322 (every unmasked column); elastic net 32 (21 exogenous).
- Kendall's W: lasso 0.563 (0.507 exogenous only); ridge 0.620 (0.672 sd-scaled); elastic net 0.640.

## Open

- About 242 of the 564 columns are masked at each re-estimation, and 213 of the 276 availability averages are never
  live, so group importance rests almost entirely on the value averages. stocktwits_sentcount and three voldemand_*
  groups are live in only 114–121 re-estimations.
- Probe benchmark (as the earlier FI): needs no new design columns — append the 3 noise columns (uniform, 20-level,
  binary; median/IQR-scaled; penalty factor 1; unmasked) at each origin and batch-solve (~5 min on 6 workers); it is a
  batch fit, not the walk, and real columns are compared with the probe fit's own coefficients.
