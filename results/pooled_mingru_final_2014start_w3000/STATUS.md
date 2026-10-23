# Final pooled ridge-init MinGRU (W 3,000, first day 2014-01-02, fold 3): status log

Rule: `RULE_gru.txt` (written 2026-10-22 21:45 PDT, before any tuning job). Runner
`experiments/pooled_mingru_final_2014_w3000.py`; gate `experiments/pooled_mingru_final_gate.py`; GPU job check
`experiments/pooled_mingru_final_post_check.py`; Slurm `cluster/pooled_mingru_final_gpu_carc.sbatch`,
`cluster/pooled_mingru_final_cpu_carc.sbatch`, `cluster/pooled_mingru_final_submit_carc.sh`; scorer
`experiments/pooled_mingru_final_score.py`. CARC root `/home1/jc_905/pmf` (NFS, as the previous run after its Lustre
errors); binary copied from `/home1/jc_905/pm13/bin` (md5 44ab6fff).

## Log
- 2026-10-22 21:38 PDT: gate jobs submitted: 13319779 (tune, the previous frozen configurations on fold 3, 3 seeds a
  set) and 13319780 (refit, the first walk origin row 44,250, 3 seeds a set), both on the previous design's data
  (`/home1/jc_905/pm13/data`), P100, CONC 6, POST = the gate check.
- 2026-10-22 21:43 PDT: CPU prep 13319882 (b04-01): export of the W = 3,000 2014-start design caches (84,747 x 12 and
  x 564, HAR block 12 a set; dates of rows >= 39,000 equal the previous design's, y equal to 8.9e-16); plan: fold 3
  fit [8,202, 47,202), validation [47,527, 50,777) = 2013-01-02 15:30 .. 2013-12-31 16:00; 2,643 daily origins
  2014-01-02 .. 2024-04-30 (13 bars on 2,579 sessions, 7 on 61, 8 on 1, 4 on 2); fold-3 ridge inits for every ladder
  L x wd_lin (keep 293 of 564 columns on the full set, 12 of 12 on HAR); round 0: 1,296 configurations x 5 seeds =
  6,480 tune jobs a set in 19 files a set; canary jobs.
- 2026-10-22 21:49 PDT: canary on the new design, 13319943 (tune: one round-0 configuration a set, 1 seed) and
  13319944 (refit at the first and the last daily origin, 2 epochs), POST = check-run. Round-0 tune jobs
  13319945-13319982 (38 files) submitted held afterok on the two gate jobs and the two canaries: no tune job can start
  before the gate and the canary pass. All P100s of the gpu partition busy at submission (1,143 pending GPU jobs).
- 2026-10-22 21:57 PDT: owner (relayed): "Test mingru 3.5k as well". RULE_gru.txt addendum (same rule at W = 3,000 and
  W = 3,500, tuned separately); the runner takes the window from PMF_WINDOW_DAYS (fold 3 fit = [47,202 - W, 47,202));
  W = 3,500 in /home1/jc_905/pmf35 and results/pooled_mingru_final_2014start_w3500/ (its own STATUS.md). The W = 3,000
  jobs keep priority (W = 3,500 GPU jobs submitted with a Slurm nice value).
- 2026-10-22 22:45 PDT: owner (relayed): "All models need to refit every period. Don't cut corners." RULE_gru.txt
  ADDENDUM 2 (both windows, before any walk job): a refit at every bar from 2014-01-02 10:00 (33,970 bars an arm),
  5 seeds, from scratch (as every tuning job and every previous refit: weights_init + the window's ridge init, no
  init_from). Runner: bar-origin blocks of 1,000 (34 GPU job files an arm), per-block ridge-init subdirectories
  (INITSUB staging); timing probe cluster/pooled_mingru_final_probe_carc.sbatch (+ experiments/
  pooled_mingru_final_probe.py) at concurrency C, 2C, 4C with a bit-identity check. Projection sent to the
  coordinator (from the previous run's P100 walk logs scaled by window rows): about 175 P100-hours for the four bar
  walks at 10 epochs (W 3,000 full ~46, HAR ~34; W 3,500 full ~54, HAR ~39), more if the tuned H or L is large.
