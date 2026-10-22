# Pooled ridge-init MinGRU, 13-bar design: status log

Runner `experiments/pooled_mingru_rerun_13.py`; Slurm files `cluster/pooled_mingru13_gpu_carc.sbatch`,
`cluster/pooled_mingru13_cpu_carc.sbatch`. CARC working directory `/scratch1/jc_905/pm13/`.

## Selection rule (written 2026-10-22, before any job ran)

Per predictor set, one configuration of the 144 (H {64, 256} x L {20, 48} x lr {0.001, 0.003} x wd_enc {0, 3, 15}
x wd_lin {0.03, 0.3, 3} x lr_lin {0, 1e-5}, the original pooled study's grid), frozen over the whole walk: the
lowest mean over the 4 pre-2012 folds (the pooled linear rerun's folds, validation 2008-01-15 .. 2011-12-30) of the
fold's validation QLIKE (r = y^2 / f^2, r - log r - 1, rows with y > 0) of the forecast averaged over the 3 seeds
(20261012, 20261013, 20261014); configurations with a non-finite fold value are skipped; sorted name order, ties:
the first. Refit epochs of seed s: the median over the 4 folds of seed s's best (early-stopped) epoch in the frozen
configuration, rounded half up. Refit cadence: the original's monthly refit, every 21 sessions from the walk start
(design row 44,250, 2011-12-28 14:00). Forecast: the seed mean of f, plain = f^2 B. Nothing from 2012-01-03 on
enters any choice.

## Log
- 2026-10-22 07:23 PDT: CARC prep job 13303035 (CPU, a03-03): export (both design caches -> data/pm13_<set>/), plan
  (4 folds = the linear rerun's; 151 refit origins over 3,154 sessions, 2011-12-28 14:00 .. 2024-04-25 10:00), 1,860
  ridge-init files, 3,456 tune lines in 16 files. Check job 13303036: the init code reproduces the original study's
  shipped inits bit for bit (block 0, r24000 and tune, L 20 / 48, every wd_lin, both sets).
- 2026-10-22 07:35 PDT: submitted (cluster/pooled_mingru13_submit_carc.sh): canary 13303107; tune 13303108-13303123
  (all features f0-f3 x s0-s2, HAR benchmark f0-f3); select 13303124 (afterok tune); walk 13303125-13303136 (all
  features p0-p5, HAR benchmark p0-p5; afterok select + canary). All GPU jobs --constraint=p100.
- 2026-10-22 07:29 PDT: canary 13303107 COMPLETED (d23-16, P100, 3.3 s): 4 refit lines (both sets, first and last
  origin), status ok, forecast rows exactly [origin, next origin), the trainer's kept columns = the init's (338 of 576
  at the first origin on all features, 12 of 12 on the HAR benchmark).
- 2026-10-22 07:40 PDT: determinism duplicate 13303142 (walk_all_features.p0 again, afterok select + canary).
- 2026-10-22 ~08:00 PDT: tune 13303119 (all features f3 s2, e22-08) FAILED at start, "bad arrays in
  data/pm13_all_features" (a short read of X.f32 on /scratch1); tune 13303123 (HAR benchmark f3, e21-01) FAILED after
  51 of 432 jobs with a bus error. Their run directories moved to runs_failed/. The GPU sbatch now stages the jobs
  file's design, its ridge inits and the binary on the node's local disk (sizes checked, binary md5 checked). Reruns
  13303257 (all features f3 s2) and 13303258 (HAR benchmark f3), both --exclude=e22-08,e21-01. Select 13303124's
  dependency updated to the running and rerun shards. Walk jobs 13303125-13303136 and duplicate 13303142 cancelled
  (pending, not started) and resubmitted with the staging script: walk 13303259-13303264 (all features p0-p5),
  13303265-13303270 (HAR benchmark p0-p5), duplicate 13303271 (walk_all_features.p0), all afterok select 13303124.
- 2026-10-22 ~08:20 PDT: tune 13303120 (HAR benchmark f0, e22-14) finished all 432 jobs (status ok) at 07:40 but the
  job FAILED (exit 1) at 07:44: pool_summary.csv empty, steps.csv not compressed (a write on /scratch1). Run directory
  moved to runs_failed/ (kept for a determinism comparison); rerun 13303400 (--exclude=e22-08,e21-01,e22-14). The GPU
  sbatch now also writes the trainer's outputs on the local disk and copies them to the run directory with an md5
  check. Select 13303124 now afterok 13303257, 13303258, 13303400. Walk jobs 13303259-13303271 cancelled (pending)
  and resubmitted with this script: walk 13303401-13303406 (all features p0-p5), 13303407-13303412 (HAR benchmark
  p0-p5), duplicate 13303413.
- Design check: data/pm13_<set>/columns.txt (from the CARC design caches the trees used) is byte-identical to the
  linear rerun's frozen_penalties.json designs.<set>.columns: 12 and 576 columns, same names and order, 48 FOMC
  columns including adj_fomc_release_ma_* (the design as built on 2026-10-20; nothing was rebuilt, so later changes to
  src/data/loading.py do not reach it).
- 2026-10-22 ~08:40 PDT: the rerun 13303257 (all features f3 s2, d23-14) FAILED while staging: "cp: cannot stat
  .../pm13_lin_init/...: Input/output error", "Cannot send after transport endpoint shutdown" (the node's Lustre
  client; /scratch1 is Lustre). 13303258 (HAR benchmark f3) COMPLETED. Pending jobs cancelled (13303400 tune rerun,
  select 13303124, walk 13303401-13303413). Working directory moved to /home1/jc_905/pm13 (NFS; the original pooled
  study ran from $HOME): code, data, inputs, bin and the 14 completed tune runs + the canary copied, every file's md5
  equal to its source, the exported design files equal to their sha256 in data/pm13_export.json. The sbatch files
  default to ROOT=/home1/jc_905/pm13. Resubmitted: tune 13303656 (all features f3 s2), 13303657 (HAR benchmark f0);
  select 13303658 (afterok both); walk 13303659-13303664 (all features p0-p5), 13303665-13303670 (HAR benchmark
  p0-p5), duplicate 13303671 (walk_all_features.p0), afterok select.
- 2026-10-22 08:54 / 09:42 PDT: tune 13303656 (all features f3 s2, e22-06) and 13303657 (HAR benchmark f0, e22-10)
  COMPLETED. All 16 (set, fold, shard) tune runs present.
- 2026-10-22 09:52 PDT: select 13303658 FAILED on a bug in st_select (a pandas lookup of the frozen configuration's
  mean on a column that is the index; it had written pretune_qlike_folds.csv, nothing else). Fixed (u.loc[best,
  "qlike_mean"]; the rule is unchanged), checked on a synthetic frame, shipped (md5 0ad355a7); select resubmitted as
  13305260; the walk jobs 13303659-13303670 and duplicate 13303671 re-pointed (scontrol) to afterok 13305260.
- 2026-10-22 09:56 PDT: select 13305260 COMPLETED (a02-18): HAR benchmark L20_H64_lr0.003_we15_wl0.3_ll1e-05 (mean
  validation QLIKE 0.117659, epochs 8 / 12 / 9), full set L20_H64_lr0.003_we3_wl3_ll1e-05 (0.108322, epochs 16 / 10
  / 9); 453 refits a set in 6 files.
- 2026-10-22 09:57-10:01 PDT: walk 13303659-13303670 and duplicate 13303671 COMPLETED (P100; 20 s to 1 min 50 s
  each); every refit status ok; every run's outputs match their local md5 after the copy. Duplicate check: walk
  all features p0 (13303659) vs 13303671, 76 refits, every forecast bit-identical. Tune repeats bit-identical: HAR
  benchmark f0 13303120 (failed after finishing) vs 13303657, 432 jobs; HAR benchmark f3 13303123 (52 jobs before the
  bus error) vs 13303258.
- 2026-10-22 10:06 PDT (local): merge -> forecasts.parquet (2 x 40,497 bars), forecasts_seeds.parquet; score ->
  scores_accuracy*.csv, econ_full_vs_har.csv; experiments/score_13bar_span.py -> table1_span/.
- 2026-10-22 10:20-10:40 PDT: the new .py files ruff-formatted and type-annotated (no logic change); re-run: merge and
  score outputs equal (DataFrame.equals, cmp); select re-run on CARC with the formatted runner (13306008; an srun try
  13305831 was cancelled) reproduces pretune_qlike.csv, pretune_qlike_folds.csv, frozen_config.json and the 13 walk
  files byte for byte.
- Compute: 34 GPU jobs (30 COMPLETED, 4 FAILED), 3.36 P100-hours allocated (0.89 in the failed jobs); trainer
  training time summed over concurrent models 14.8 h (tune) and 0.47 h (walk); CPU jobs about 0.1 core-hours.
  Hoffman2 not used. jobs.csv holds sacct of every job that ran.
