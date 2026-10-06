# Reproduction check of the close studies (2026-10-03 / 04)

Written by `experiments/close_studies_repro_check.py check` on 2026-10-06 20:57 UTC, commit `2ecea61699` (a clean `git archive HEAD` under `_repro_scratch/checkout`), 9.4 min wall. Python 3.11.15, LightGBM 4.7.0, XGBoost 2.1.4, numpy 1.26.4, pandas 2.3.3; CPU Intel(R) Xeon(R) Processor @ 2.10GHz (AVX-512 yes). Every rerun went to the scratch folder; one child process at a time, single-threaded.

**21 pass, 5 fail, 2 informative.** pass = bitwise identical (scores: within 1e-12); info = an informative check with no expected outcome.

**The 5 failing rows of b, e are resolved by the rows b-fix, e-fix.** Run as the committed scripts run on this machine, the C-kernel arms (ridge: validation MSEs only, forecasts identical) and the numpy-BLAS linear models differ from the stored arms in the last bits only (the scored QLIKE / Sharpe agree to 1e-16), because this machine's numpy OpenBLAS kernel and gcc tuning differ from those of the machine that wrote the runs (environment.json, reproduction_settings). The same commands with the settings of REPRODUCE.md (OPENBLAS_CORETYPE=SkylakeX, C kernel built for -march=haswell) reproduce them bit for bit.

| check | target | result | max difference | seconds |
|---|---|---|---|---|
| env | this machine vs environment.json (python, packages, CPU flags, OpenBLAS kernels, gcc, LAPACK) | info |  |  |
| a0 | 17 cached designs (results/close_design/_work) vs design_hashes.json | pass |  | 22.5 |
| a | design_bar1600_all_features.npz rebuilt (clean checkout, CLOSE_DESIGN_DIR, 1 thread) vs the cache: X, y, names, date, baseline, true_raw, W | pass | 0 | 64.7 |
| a | rebuilt design vs design_hashes.json (sha256 of every array) | pass |  |  |
| a2 | bar1600 all_features rebuilt with numpy's AVX-512 code paths off (what an AVX2 machine computes) vs the cache | info | 2.78e-11 | 55.5 |
| b | exog_penalty ridge_bb0 (as the scripts run it here): 1469 forecasts, coefficients, validation MSEs, alphas, masks vs _work/runs/ridge_bb0.npz | fail | 0 | 42.7 |
| b | exog_penalty lasso_bb0 (as the scripts run it here): 1469 forecasts, coefficients, validation MSEs, alphas, masks vs _work/runs/lasso_bb0.npz | fail | 1.33e-15 |  |
| b | exog_penalty ols_baseline (HAR + calendar OLS, numpy lstsq; OpenBLAS own choice) vs _work/runs/ols_baseline.npz | fail | 2.44e-15 |  |
| b | ridge_bb0 rerun (as the scripts run it here) scored on the 866 trade days vs exog_penalty/headline.csv: QLIKE, Sharpe mid, Sharpe crossed | pass | 6.94e-17 | 0.6 |
| b | lasso_bb0 rerun (as the scripts run it here) scored on the 866 trade days vs exog_penalty/headline.csv: QLIKE, Sharpe mid, Sharpe crossed | pass | 1.39e-17 |  |
| b-fix | exog_penalty ridge_bb0 (with the settings of REPRODUCE.md): 1469 forecasts, coefficients, validation MSEs, alphas, masks vs _work/runs/ridge_bb0.npz | pass | 0 | 40.1 |
| b-fix | exog_penalty lasso_bb0 (with the settings of REPRODUCE.md): 1469 forecasts, coefficients, validation MSEs, alphas, masks vs _work/runs/lasso_bb0.npz | pass | 0 |  |
| b-fix | exog_penalty ols_baseline (HAR + calendar OLS, numpy lstsq; OpenBLAS SkylakeX) vs _work/runs/ols_baseline.npz | pass | 0 |  |
| c | trees_datasize lgbm_bar1600_w2000 (shipped configuration, k = 1), first 3 refits (30 forecasts) vs _work/lgbm_bar1600_w2000.npz | pass | 0 | 20.6 |
| c | trees_pretune walk control, first 3 refits (30 forecasts) vs _work/local/lgbm/walk_control_r10.npz | pass | 0 | 46 |
| c | trees_pretune walk frozen_k10, first 3 refits (30 forecasts) vs _work/local/lgbm/walk_frozen_k10_r10.npz | pass | 0 |  |
| c | trees_pretune walk frozen_k45 (+ retune_any, retune_margin: the same fits before row 380), first 3 refits (30 forecasts) vs _work/local/lgbm/walk_frozen_k45_r10.npz | pass | 0 |  |
| c | trees_pretune pre-tune trial 0 (the shipped configuration, 4 fold fits): validation MSE, QLIKE, early-stopped rounds vs pretune_s0.npz / pretune_s1.npz | pass | 0 | 33.6 |
| c | trees_lineartree lt_all_r10_lam10 (linear_tree, linear_lambda 10), first 3 refits (30 forecasts, leaf coefficients) vs _work/lt_all_r10_lam10.npz | pass | 0 | 20.5 |
| c | trees_morebars lgbm_bars4_r8000 (the k = 4 pool, 8000 rows), first 3 refits (30 forecasts) vs _work/lgbm_bars4_r8000.npz | pass | 0 | 26.3 |
| c | trees_kfull lgbm_bars13_r26000_barmin_seed43 (k = 13, column bar_end_minute, random_state 43), first 3 refits (30 forecasts, importances) vs _work/lgbm_bars13_r26000_barmin_seed43.npz | pass | 0 | 48.5 |
| d | kfull_tests run existing (KFT_OUT scratch): 20 CSVs, run_info.json, SUMMARY.md vs results/close_studies_2026-10-03/kfull_tests/ | pass | 0 | 20.4 |
| d | kfull_tests key numbers | pass |  |  |
| e | trees_datasize ridge_bar1600_w2000 (the linear spec's Python model, 1469 solves; OpenBLAS own choice) vs _work/ridge_bar1600_w2000.npz | fail | 4.98e-12 | 56.9 |
| e | trees_datasize lasso_bar1600_w2000 (the linear spec's Python model, 1469 solves; OpenBLAS own choice) vs _work/lasso_bar1600_w2000.npz | fail | 9.9e-14 |  |
| e-fix | trees_datasize ridge_bar1600_w2000 (the linear spec's Python model, 1469 solves; OPENBLAS_CORETYPE=SkylakeX) vs _work/ridge_bar1600_w2000.npz | pass | 0 | 54.1 |
| e-fix | trees_datasize lasso_bar1600_w2000 (the linear spec's Python model, 1469 solves; OPENBLAS_CORETYPE=SkylakeX) vs _work/lasso_bar1600_w2000.npz | pass | 0 |  |
| safe | committed results files and the design cache unchanged by the check (size and mtime of 339 files) | pass |  |  |

Details:

- **env**, this machine vs environment.json (python, packages, CPU flags, OpenBLAS kernels, gcc, LAPACK): same
- **a0**, 17 cached designs (results/close_design/_work) vs design_hashes.json: every array's sha256 equal
- **a**, design_bar1600_all_features.npz rebuilt (clean checkout, CLOSE_DESIGN_DIR, 1 thread) vs the cache: X, y, names, date, baseline, true_raw, W: bitwise identical
- **a**, rebuilt design vs design_hashes.json (sha256 of every array): all equal
- **a2**, bar1600 all_features rebuilt with numpy's AVX-512 code paths off (what an AVX2 machine computes) vs the cache: differs: X (max |diff| 2.78e-11); 120 of 628 columns of X differ
- **b**, exog_penalty ridge_bb0 (as the scripts run it here): 1469 forecasts, coefficients, validation MSEs, alphas, masks vs _work/runs/ridge_bb0.npz: differ: val_mse (max |diff| 5.55e-17); kernel compiled by the script (gcc -march=native), OpenBLAS's own kernel choice; the script compiled the kernel: True; cpu 5s (stored 9s)
- **b**, exog_penalty lasso_bb0 (as the scripts run it here): 1469 forecasts, coefficients, validation MSEs, alphas, masks vs _work/runs/lasso_bb0.npz: differ: pred (max |diff| 1.33e-15), val_mse (max |diff| 1.79e+51), exact_pred (max |diff| 7.77e-16), LU fallbacks 41 vs 37; kernel compiled by the script (gcc -march=native), OpenBLAS's own kernel choice; the script compiled the kernel: True; cpu 33s (stored 57s)
- **b**, exog_penalty ols_baseline (HAR + calendar OLS, numpy lstsq; OpenBLAS own choice) vs _work/runs/ols_baseline.npz: differ: pred (max |diff| 2.44e-15), theta_bb (max |diff| 2.07e-15)
- **b**, ridge_bb0 rerun (as the scripts run it here) scored on the 866 trade days vs exog_penalty/headline.csv: QLIKE, Sharpe mid, Sharpe crossed: QLIKE 0.094586 vs 0.094586, Sharpe mid 1.4966 vs 1.4966 (tolerance 1e-12: the scorer's column order moves the last bits)
- **b**, lasso_bb0 rerun (as the scripts run it here) scored on the 866 trade days vs exog_penalty/headline.csv: QLIKE, Sharpe mid, Sharpe crossed: QLIKE 0.095689 vs 0.095689, Sharpe mid 1.5204 vs 1.5204 (tolerance 1e-12: the scorer's column order moves the last bits)
- **b-fix**, exog_penalty ridge_bb0 (with the settings of REPRODUCE.md): 1469 forecasts, coefficients, validation MSEs, alphas, masks vs _work/runs/ridge_bb0.npz: bitwise identical; kernel prebuilt for -march=haswell, OPENBLAS_CORETYPE=SkylakeX; the script compiled the kernel: False; cpu 5s (stored 9s)
- **b-fix**, exog_penalty lasso_bb0 (with the settings of REPRODUCE.md): 1469 forecasts, coefficients, validation MSEs, alphas, masks vs _work/runs/lasso_bb0.npz: bitwise identical; kernel prebuilt for -march=haswell, OPENBLAS_CORETYPE=SkylakeX; the script compiled the kernel: False; cpu 33s (stored 57s)
- **b-fix**, exog_penalty ols_baseline (HAR + calendar OLS, numpy lstsq; OpenBLAS SkylakeX) vs _work/runs/ols_baseline.npz: bitwise identical
- **c**, trees_datasize lgbm_bar1600_w2000 (shipped configuration, k = 1), first 3 refits (30 forecasts) vs _work/lgbm_bar1600_w2000.npz: bitwise identical; 30 forecasts written
- **c**, trees_pretune walk control, first 3 refits (30 forecasts) vs _work/local/lgbm/walk_control_r10.npz: bitwise identical, same configuration and rounds
- **c**, trees_pretune walk frozen_k10, first 3 refits (30 forecasts) vs _work/local/lgbm/walk_frozen_k10_r10.npz: bitwise identical, same configuration and rounds
- **c**, trees_pretune walk frozen_k45 (+ retune_any, retune_margin: the same fits before row 380), first 3 refits (30 forecasts) vs _work/local/lgbm/walk_frozen_k45_r10.npz: bitwise identical, same configuration and rounds
- **c**, trees_pretune pre-tune trial 0 (the shipped configuration, 4 fold fits): validation MSE, QLIKE, early-stopped rounds vs pretune_s0.npz / pretune_s1.npz: val MSE 0.06158, rounds [257.0, 177.0, 956.0, 332.0]
- **c**, trees_lineartree lt_all_r10_lam10 (linear_tree, linear_lambda 10), first 3 refits (30 forecasts, leaf coefficients) vs _work/lt_all_r10_lam10.npz: bitwise identical
- **c**, trees_morebars lgbm_bars4_r8000 (the k = 4 pool, 8000 rows), first 3 refits (30 forecasts) vs _work/lgbm_bars4_r8000.npz: bitwise identical
- **c**, trees_kfull lgbm_bars13_r26000_barmin_seed43 (k = 13, column bar_end_minute, random_state 43), first 3 refits (30 forecasts, importances) vs _work/lgbm_bars13_r26000_barmin_seed43.npz: bitwise identical
- **d**, kfull_tests run existing (KFT_OUT scratch): 20 CSVs, run_info.json, SUMMARY.md vs results/close_studies_2026-10-03/kfull_tests/: 20 of 20 CSVs byte-identical; run_info.json (without times) same; SUMMARY.md (without its date and CPU-time lines) same; code = HEAD: yes
- **d**, kfull_tests key numbers: QLIKE ridge_bb0 0.0945858 (committed 0.0945858); QLIKE seed average k = 4 (bar column) 0.0979511 (committed 0.0979511); SPA p (bar, QLIKE) 0.732 (committed 0.732); Reality Check p (bar, QLIKE) 0.9138 (committed 0.9138)
- **e**, trees_datasize ridge_bar1600_w2000 (the linear spec's Python model, 1469 solves; OpenBLAS own choice) vs _work/ridge_bar1600_w2000.npz: differ: pred (max |diff| 4.98e-12)
- **e**, trees_datasize lasso_bar1600_w2000 (the linear spec's Python model, 1469 solves; OpenBLAS own choice) vs _work/lasso_bar1600_w2000.npz: differ: pred (max |diff| 9.9e-14)
- **e-fix**, trees_datasize ridge_bar1600_w2000 (the linear spec's Python model, 1469 solves; OPENBLAS_CORETYPE=SkylakeX) vs _work/ridge_bar1600_w2000.npz: bitwise identical
- **e-fix**, trees_datasize lasso_bar1600_w2000 (the linear spec's Python model, 1469 solves; OPENBLAS_CORETYPE=SkylakeX) vs _work/lasso_bar1600_w2000.npz: bitwise identical
- **safe**, committed results files and the design cache unchanged by the check (size and mtime of 339 files): unchanged

Seconds = wall time of the rerun behind the row (blank: the row reads the rerun of the row above). Logs: `_repro_scratch/logs/`. Commands for a full rerun: `REPRODUCE.md`.
