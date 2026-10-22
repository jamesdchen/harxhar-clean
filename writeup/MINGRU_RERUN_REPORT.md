# Pooled MinGRU rerun on the 13-bar design (2026-10-22)

Run log: `results/pooled_mingru_rerun13_2026-10-22/STATUS.md`. The selection rule there was written before any job ran.
The NN content of the paper is still parked; nothing in the paper was changed.

## Configuration

- **Design.** The pooled linear 13-bar rerun's design caches, the CARC build the pooled trees used
  (`/scratch1/jc_905/harxhar-trees13/.../pooled_linear_rerun_2026-10-19_13_w1000/_work`): 84,747 rows; 12
  HAR-benchmark columns; 576 full-set columns with 48 FOMC columns (fomc_release included). Names and order are
  byte-identical to the linear rerun's `frozen_penalties.json`; nothing was rebuilt. CARC vs cloud build: round-off
  differences, plus one column, `adj_voldemand_spx_open_only_ma_5` (max 4.4e-3), as the trees document records.
- **Trainer.** The original pooled study's native_g9 binary (`.cu` af1964d0, binary 44ab6fff), on Tesla P100 for every
  job.
- **Carried over from `pooled_rinit_study`:**
  - network: one MinGRU layer with highway gates, softplus link, zero-initialised decoder, linear shortcut started at
    its window's closed-form ridge;
  - training: Muon, cosine learning rate over 100 epochs, batch 128, QLIKE loss, early stopping on validation QLIKE
    (patience 10), window mask on, xstd = 0;
  - penalty factors: 0.01 on the HAR block (har_ma_* plus calendar), 1 on exogenous columns;
  - grid: H {64, 256} × L {20, 48} × lr {0.001, 0.003} × wd_enc {0, 3, 15} × wd_lin {0.03, 0.3, 3} × lr_lin {0, 1e-5},
    144 configurations;
  - seeds: 3 (20261012/13/14), as the original pooled study (the brief said 5, which is the per-bar MinGRU);
  - selection criterion: QLIKE of the seed-mean validation forecast against the training target y;
  - refit every 21 sessions, each seed for its own epoch count; the forecast is the seed mean of f, plain = f²B.
- **Changed, as in the linear models' protocol:**
  - tuning once, on the linear rerun's 4 pre-2012 folds (each fits on 13,000 bars, then a 325-bar embargo, then 3,250
    validation bars; validation spans 2008-01-15 to 2011-12-30);
  - the pick is the lowest mean of the 4 fold QLIKEs, one configuration per set, frozen for the whole walk;
  - refit epochs per seed: the median of its 4 fold best epochs, rounded half up;
  - window 13,000 bars; walk from design row 44,250 (2011-12-28 14:00) to 2024-04-30 16:00: 151 refit origins × 3
    seeds × 2 sets = 906 refits.

## Tuning (`pretune_qlike.csv`)

| set | frozen configuration | mean val QLIKE (folds 0–3) | epochs per seed | grid range (best..worst) |
|---|---|---|---|---|
| HAR | L20 H64 lr0.003 we15 wl0.3 ll1e-5 | 0.11766 (0.1183, 0.1075, 0.1226, 0.1222) | 8 / 12 / 9 | 0.11766..0.11905 |
| full | L20 H64 lr0.003 we3 wl3 ll1e-5 | 0.10832 (0.1099, 0.1045, 0.1105, 0.1084) | 16 / 10 / 9 | 0.10832..0.11504 |

- On the HAR set the next 5 configurations are within 0.0001; on the full set ranks 2–5 are within 0.0004.
- Grid edges: L 20, H 64, lr 0.003 and lr_lin 1e-5 are at an edge for both sets; wd_enc 15 is the top for the HAR set;
  wd_lin 3 is the top for the full set.
- Per-fold best epochs vary a lot (for example 2, 25, 4 and 11 for seed 20261012 on the HAR set).

## Compute

- CARC Discovery, `--constraint=p100`. 34 GPU jobs: 30 completed, 4 failed on `/scratch1` Lustre client errors (a short
  read, a bus error, a failed write, "transport endpoint shutdown"). Fixes: data, inits and binary staged on the node's
  local disk; outputs written locally then copied with an md5 check; the work directory moved to `/home1/jc_905/pm13`,
  every file md5-verified.
- 3.36 P100-hours allocated (0.89 in the failed jobs). Trainer time summed over concurrent models: 14.8 h for tuning,
  0.47 h for the walk. CPU about 0.1 core-hours.
- Job ids (all CARC): prep 13303035, init check 13303036, canary 13303107, tune 13303108–13303123 plus reruns
  13303257/58 and 13303656/57, select 13303658 (failed on a pandas lookup bug; the rule did not change) and 13305260,
  walk 13303659–13303670, duplicate 13303671, verify 13306008; failures 13303119, 13303120, 13303123, 13303257. Full
  list in `jobs.csv`.
- Hoffman2 was not used: the work needed 3.4 P100-hours, Hoffman2 has no native_g9 build, and its GPUs are a different
  model (bit-identical runs are established only within one GPU model).

## Checks

1. The ridge-init code reproduces the original study's shipped inits bit for bit: 24 files, both sets (job 13303036).
2. The folds equal the linear rerun's (rows and first/last bars asserted); the design columns equal its 12 and 576; the
   exported files match their sha256.
3. Canary: the trainer keeps the same columns as the init (338 of 576 at the first origin; 320–350 across the walk); the
   forecast rows are exactly [origin, next origin).
4. Repeats are bit-identical across P100 nodes: tune HAR fold 0 (432 jobs, 1.40M validation forecasts); tune HAR fold 3
   (the 52 jobs before the bus error); walk full-set part 0 (76 refits).
5. The forecast bars and rv_actual equal the linear rerun's 40,497 walk bars; all forecasts finite and positive; seed
   spread (sd/mean of f) median 3.1% on the full set and 1.8% on the HAR set.
6. Scorer gates: the linear 16:00 QLIKE equals `scores.csv` (1e-10) and the every-bar QLIKE equals
   `scores_all_bars.csv`; the context Sharpe ratios match the economics generator; the 2012–2024 Table 1 analog
   reproduces the linear DMs (−7.86, −8.07, −10.45).
7. After ruff and mypy cleanup the select, merge and score outputs reproduce byte for byte.

## Results (full predictor set vs HAR benchmark)

| model | QLIKE 16:00 Δ% (DM, p) | QLIKE all bars, 1,082 days Δ% (DM) | all bars 2012–24 Δ% (DM) | SR mid HAR → full (LW p; fixed-b t, p) | SR bid/ask HAR → full (LW p) |
|---|---|---|---|---|---|
| **MinGRU** | **−2.76 (−0.64, 0.52)** | **−7.05 (−5.72)** | **−6.78 (−9.81)** | **0.86 → 1.09 (0.49; 0.75, 0.47)** | **0.53 → 0.75 (0.49)** |
| elastic net | −0.87 (−0.31, 0.76) | −3.97 (−4.58) | −4.91 (−10.45) | 1.30 → 1.13 (0.75; 0.33, 0.75) | 0.96 → 0.79 (0.75) |
| lasso | +2.18 (0.58, 0.56) | −3.96 (−2.77) | −5.66 (−8.07) | 1.28 → 1.02 (1.00; −0.01, 0.99) | 0.93 → 0.68 (1.00) |
| ridge | +2.63 (0.61, 0.54) | −4.21 (−2.85) | −5.61 (−7.86) | 1.30 → 1.09 (0.96; −0.05, 0.96) | 0.96 → 0.75 (0.97) |
| LightGBM | −1.64 (−0.60, 0.55) | −10.48 (−7.29) | −9.70 (−11.67) | 0.96 → 1.07 (0.29; 1.14, 0.28) | 0.63 → 0.74 (0.29) |
| XGBoost | −1.66 (−0.62, 0.53) | −13.29 (−6.39) | −11.91 (−10.78) | 1.03 → 1.34 (0.18; 1.45, 0.17) | 0.69 → 1.00 (0.19) |
| random forest | −4.23 (−1.73, 0.08) | −7.03 (−7.05) | −6.73 (−11.86) | 1.21 → 1.03 (0.99; 0.02, 0.99) | 0.86 → 0.69 (0.99) |

- Every all-bar DM above has p < 0.01. Sharpe ratios: per-year annualized, 614 post-warmup days (the paper's
  convention); the linear and tree rows are the 1,000-day design of Section 7.
- MinGRU squared error rises with the full set: 16:00 MSE +121% (DM 1.15, p 0.25); every-bar R² falls from 63.4% to
  26.3% over 2020–24 and from 63.2% to 32.8% over 2012–24, but rises from 62.3% to 64.4% over 2012–19; the 5 largest
  2020–24 squared errors are March 2020 bars and carry 40% of the total.
- MinGRU positions: long on 208 (HAR) and 196 (full) of 614 days; the two sets agree on 510 days.
- MinGRU Sharpe by year, HAR → full: 2021 −0.01 → 0.66, 2022 0.44 → 1.18, 2023 1.74 → 1.32, 2024 0.02 → 0.49.
- Context, the original 48-interval pooled MinGRU (866 days, old conventions): QLIKE −3.0% (p 0.47); 866-day SR
  0.78 → 0.34.

## Same story?

- **MinGRU alone: yes on direction.** The full set lowers QLIKE at the 16:00 bar (−2.8%) and over every bar (−7.1% on
  2020–24, −6.8% on 2012–24), and raises the Sharpe ratio by +0.23 at mid and at bid/ask.
- **Significance:** only the every-bar QLIKE reductions are significant (DM −5.7 and −9.8). The 16:00 QLIKE reduction
  is not (p 0.52); the Sharpe increases are not (LW p 0.49; fixed-b p 0.47).
- **All seven pooled classes on this design:** the full set raises the Sharpe ratio in 3 (LightGBM, XGBoost, MinGRU)
  and lowers it in 4 (elastic net, lasso, ridge, random forest), none significant (LW p ≥ 0.18); it lowers the 16:00
  QLIKE in 5 of 7, none at 5%; it lowers every-bar QLIKE in 7 of 7, all p < 0.01.
- These are the 1,000-day-window numbers for the linear models; see `writeup/LINEAR_SWITCH_AUDIT.md` section 6 for the
  window sensitivity, and the open question of which Sharpe statistic to report (per-year vs pooled vs equal years).

## Open questions for James

1. **Change the paper?** Not done. Should the parked NN content be un-parked with these numbers, and with which
   accuracy view (16:00 or every bar)?
2. Several frozen hyperparameters sit at grid edges. Widen the grid (a pre-2012-only retune)?
3. 3 seeds, as the original pooled study, or 5, as the per-bar MinGRU?
4. Monthly refit (every 21 sessions), versus every bar for the linear models and trees. A daily refit (about 19k refits)
   is feasible in a few P100-hours.
5. Should the full-set MSE increase from March 2020 be noted?
6. If the window changes for the linear models and trees (`LINEAR_SWITCH_AUDIT.md` section 6), the MinGRU would need the
   same window for one design (a few P100-hours).
