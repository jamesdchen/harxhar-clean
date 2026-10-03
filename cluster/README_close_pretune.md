# Trees pre-tune study on Hoffman2 (2026-10-03 close studies)

The full-size run of `experiments/close_trees_pretune.py`: a heavy Optuna pre-tune of the 16:00-bar
LightGBM and XGBoost (all_features, 16:00 bar) on sessions before 2020-01-03, then a light retune
every 250 sessions, scored on the 866 trade days. The local reduced run and its report are in
`results/close_studies_2026-10-03/trees_pretune/SUMMARY.md`; this run uses the same code with larger
sizes.

## What it runs

| array | tasks | one task | default size |
|---|---|---|---|
| `ctp_pre` | 8 (`cluster/close_pretune_tasks_pretune.txt`: lgbm / xgb x seeds 0-3) | one Optuna TPE study (sampler seed 20261003 + seed, trial 0 = the shipped configuration), the 7 folds of each trial in parallel on 7 of 8 slots | 2000 trials; 7 folds x 250 sessions (1750 pre-2020 validation sessions, 2013-01 .. 2019-12), each fitted on the 2000 sessions before a 25-session embargo |
| `ctp_walk` (held on `ctp_pre`) | 4 (`cluster/close_pretune_tasks_walk.txt`: lgbm / xgb x refit every 10 / every session) | merges the model's 4 studies; light retunes every 250 sessions from 2020 (15 trials around the incumbent, 250 validation sessions as 2 folds of 125); walk-forward refits of the arms control / frozen (best of the first 50, 200, 1000, all trials of every study) / retune_any / retune_margin | 1469 forecast rows |

Parallel trials: **independent seeded studies, merged afterwards** (no shared storage). Each study
is sequential and its folds are combined in fold order, so a study is bit-reproducible on one CPU
class and one library build whatever the slot count; the merged result is a deterministic function
of the studies. Each study keeps an Optuna JournalFileStorage on scratch: a task stopped by its time
guard (45 min before h_rt) or killed resumes from the journal on resubmission; a resumed study
reseeds its sampler with seed + 7919 x completed trials (logged in `pretune_s<seed>_resumes.json`),
so it is reproducible given its resume points, but not identical to an uninterrupted study. Pin one
CPU architecture (`ARCH=...`) for every task: LightGBM's selections change across CPU classes.

Runtime: the CARC Python stack relayed into `/u/scratch/j/jamesdc1/harxhar-optuna` by
`cluster/optuna_h2_runtime_setup.sh` (LightGBM 4.6.0 / XGBoost 3.2.0, the stored campaign's builds)
when present; otherwise `source /u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh && conda
activate hpc-pi` (older LightGBM / XGBoost: the run's numbers then stand against its own control
arm). Every fit is single-threaded.

## Laptop commands (PowerShell or cmd; Git Bash for the `bash` scripts)

1. Ship (code, specs, src, the two design caches; md5-checked on Hoffman2; re-run after a scratch purge):

       bash cluster/close_pretune_ship_h2.sh

2. Smoke run on a compute node (2 trials, 1 fold; 2 retune trials, refit rows 370-399; the same run took 2.5 min of one core locally):

       C:\Windows\System32\OpenSSH\ssh.exe hoffman2 "bash -lc 'cd /u/scratch/j/jamesdc1/harxhar-close-pretune && bash cluster/submit_close_pretune_h2.sh smoke smoke1'"

3. Full run (pin a CPU architecture; check `qhost -F arch` for the values):

       C:\Windows\System32\OpenSSH\ssh.exe hoffman2 "bash -lc 'cd /u/scratch/j/jamesdc1/harxhar-close-pretune && ARCH=intel-gold-6240 bash cluster/submit_close_pretune_h2.sh all h2full'"

4. Status (one poller, every >= 10 min):

       C:\Windows\System32\OpenSSH\ssh.exe hoffman2 "bash -lc 'qstat -u jamesdc1; ls /u/scratch/j/jamesdc1/harxhar-close-pretune/results/close_studies_2026-10-03/trees_pretune/_work/h2full/*/'"

5. If studies stopped at the time guard before 2000 trials (no `PRETUNE_DONE_s<seed>` flag), resume (finished tasks exit at once; the walk array is re-held on the new pre-tune array):

       C:\Windows\System32\OpenSSH\ssh.exe hoffman2 "bash -lc 'cd /u/scratch/j/jamesdc1/harxhar-close-pretune && ARCH=intel-gold-6240 bash cluster/submit_close_pretune_h2.sh resume h2full'"

6. Pull and report (writes `SUMMARY_h2full_<model>.md` and the `h2full_<model>_*.csv` tables):

       bash cluster/close_pretune_pull_h2.sh h2full

Sizes can be changed with env on the submit command (`TRIALS`, `FOLDS`, `FOLD_LEN`, `RETUNE_TRIALS`,
`CHECKPOINTS`, `SLOTS`, `H_RT`, `H_DATA`). The CPU-hour estimate is in the local SUMMARY.md.
