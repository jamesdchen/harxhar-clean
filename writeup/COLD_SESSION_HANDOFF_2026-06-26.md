# COLD-SESSION HANDOFF — residualized-EBM campaign is DRIVING (2026-06-26)

You are picking up a campaign that is **already running and already proven correct**. Two autonomous
local driver processes (`drive_campaign.py carc` and `drive_campaign.py hoffman2`) are tuning an EBM
residualizer over the slim all_buckets matrix, full-OOS, via the hpc-agent chunk-array path, one
Optuna trial per iteration, 25 trials per cluster. Your job is to **babysit + report the result**, not
to rebuild anything.

> Supersedes the earlier `CAMPAIGN_HANDOFF.md` ("ready to drive" — it was WRONG; 5 fixes were needed,
> see `CAMPAIGN_DRIVE_STATUS_2026-06-26.md`). Framework bugs are in `HPC_AGENT_BUGS_2026-06-26.md`.

---

## 0. State at handoff (verify it, then act)

- **Proven correct:** CARC trial 0 → full-OOS residualized QLIKE **0.12424** (82 chunks, 194934 rows =
  n_oos exactly; beats enet base **0.12516**; matches the independent manual `ebm_verify` 0.12429).
- **CARC driver** (`drive_campaign.py carc`): at trial 1 (`ebm_resid-e584fa49`, SLURM job 9639562) as of
  14:06. Will run trials 1→24, then exit.
- **Hoffman2 driver** (`drive_campaign.py hoffman2`): cold-started; first **SGE** submit (trial 0) was
  in progress at handoff. **VERIFY this landed** — it's the one path not yet seen succeed this session.
- **Bar to beat:** enet base 0.12516. Headline = best (min) trial QLIKE per cluster. Trial 0 (untuned)
  already clears it; tuning targets toward the lgbm-subset proxy 0.12346.

## 1. FIRST ACTIONS — check progress (read-only)

Tail the driver logs:
- `.hpc/campaigns/ebm_all_buckets_carc/drive.log`
- `.hpc/campaigns/ebm_all_buckets_hoffman2/drive.log`

Query a study (finished trials + best QLIKE). Run with the uv-tool python from the repo root:
```
"C:/Users/james/AppData/Roaming/uv/tools/hpc-agent/Scripts/python.exe" -c "import optuna; s=optuna.load_study(study_name='resid_ebm_all_buckets', storage='sqlite:///.hpc/campaigns/ebm_all_buckets_carc/optuna.db'); fin=[(t.number,t.value) for t in s.get_trials(deepcopy=False) if t.state.is_finished()]; print('n_done',len(fin),'best',min((v for _,v in fin), default=None)); print(sorted(fin))"
```
(swap `carc`→`hoffman2` for the other study.) `n_done == 25` per cluster = done.

## 2. ARE THE DRIVERS STILL ALIVE? (the one thing that can break on context-clear)

The drivers are **local background processes**. Clearing context may or may not kill them; a laptop
sleep / SSH drop **will** pause them. Check: if `drive.log` hasn't advanced in ~20+ min and a study's
running trial's array is terminal on the cluster, the driver is dead. **They are fully resumable** —
just relaunch (the allowlist rule `Bash(*drive_campaign.py*)` in `.claude/settings.json` pre-approves it):

```
"C:/Users/james/AppData/Roaming/uv/tools/hpc-agent/Scripts/python.exe" "C:/Users/james/CC Allowed/harxhar-clean/drive_campaign.py" carc       # run_in_background:true
"C:/Users/james/AppData/Roaming/uv/tools/hpc-agent/Scripts/python.exe" "C:/Users/james/CC Allowed/harxhar-clean/drive_campaign.py" hoffman2    # run_in_background:true
```

On relaunch the driver reads `.hpc/campaigns/<cid>/drive_state.json` + the optuna study and continues
from the last submitted trial (waits for its array if still in flight, else pulls + tells + submits the
next). **Do NOT delete the optuna.db or drive_state.json** — that's the resume state. If you must hand-
fix state, `drive_state.json` = `{"run_id":..., "job_id":..., "trial": N}` for the in-flight array.

## 3. What is already set up (DO NOT redo)

- **hpc-agent uv-tool venv has the scientific stack** (numpy/optuna 4.9/numba/pandas/sklearn/scipy
  installed into `AppData/Roaming/uv/tools/hpc-agent`). Required so `load_tasks_module` can import
  `tasks.py` for local enumeration. (hpc-agent BUG 1/2 in the bugs doc; this is the workaround.)
- **Local cache** `results/resid_prep/ebm_all_buckets_tw1000_enet_rf480_slim/cell.json` (fetched from
  CARC; model-independent → shared by both campaigns). The ~1GB `Xs.npy` stays cluster-side; `results/`
  is deploy-excluded so it's never clobbered.
- **Cluster caches** present on CARC + Hoffman2 (handoff-built). Don't rebuild.
- **Code fixes on disk** (keep them): `.hpc/tasks.py` — (a) campaign-env import guard, (b) EBM
  `n_estimators` guard, (c) reuse-in-flight-trial determinism, (d) per-chunk `blk_*` output subdir +
  `_optuna_trial_number = trial_blk0_blk1`. `src/models/resid_tree.py` — substitutes `{run_id}` from
  `HPC_RUN_ID`. `src/backtest/tree_space.py` — unchanged (EBM space already omits n_estimators).

## 4. How one iteration works (so a manual step matches the driver)

Per cluster, the driver loops: **wait** the in-flight array (poll `hpc-agent batch-status`) → **scp-pull**
`results/<run_id>` → **enumerate** (`hpc-agent compute-run-id` imports `tasks.py` → `_tell_from_chunks`
tells the finished trial + reuses/asks the next) → read the running trial N from the study → **build**
the submit spec (`result_dir_template = results/{run_id}/ebm_all_buckets/trial_<N>/blk_{_optuna_trial_number}`)
→ `resolve-submit-inputs` → `submit-flow` (canary off; path proven) → record state → repeat to 25.

**Campaign env that MUST be exported for any manual enumerate/submit** (hpc-agent does NOT derive these
from the manifest — BUG 4):
```
PYTHONPATH=C:/Users/james/CC Allowed/harxhar-clean
HPC_CAMPAIGN_ID=ebm_all_buckets_carc        # or ..._hoffman2
HPC_KW_MODEL=ebm HPC_KW_BUCKET=all_buckets HPC_KW_OBJECTIVE=residualized HPC_KW_BASE=enet
HPC_KW_ALPHA=1 HPC_KW_ARM=resid_subset HPC_KW_PIPE=slim HPC_KW_TREE_REFIT=480
HPC_KW_TRAIN_WIN_DAYS=1000 HPC_KW_CHUNKS=90 HPC_KW_K=1 HPC_KW_MAX_TRIALS=200 HPC_KW_N_EST_CAP=300
```
A submit spec ALSO needs these `HPC_KW_*` inside `submit.extra_env` (the cluster `tasks.py` reads them),
plus `modules/conda_source/conda_env` per cluster (build-submit-spec doesn't auto-read clusters.yaml —
minor bug). CARC: `conda` / `/apps/conda/miniforge3/25.3.0/etc/profile.d/conda.sh` / `harxhar`, SLURM,
`jc_905@discovery2.usc.edu`, `/scratch1/jc_905/harxhar-clean`. Hoffman2: `""` /
`/u/local/apps/anaconda3/2023.03/etc/profile.d/conda.sh` / `hpc-pi`, SGE, `jamesdc1@hoffman2.idre.ucla.edu`,
`/u/scratch/j/jamesdc1/harxhar-clean`. (All baked into `drive_campaign.py`'s `CLUSTERS` dict — read it.)

## 5. Compute/verify a trial's QLIKE directly from chunks (audit a number)

Pull `results/<run_id>` from the cluster (`scp -r usc-discovery:/scratch1/jc_905/harxhar-clean/results/<run_id> results/`),
then concat its `trial_<N>/blk_*/chunk_*.csv` (cols `k,pred_adj,y_true,base`), drop_duplicates on `k`,
check `len == 194934` & contiguous, `apply_duan_smearing(pred_adj,y_true,base)`, QLIKE =
`mean(rr - log(rr) - 1)` over `rr = y_true/pred_adj` on positives. (Pattern: the verified
`scratchpad/verify_iter1.py` gave 0.12424 for trial 0.) This is the audit the professor-facing number
must pass — `_tell_from_chunks` does the same fold to tell the study.

## 6. Final deliverable

When both studies hit 25 finished trials: report **best (min) QLIKE per cluster** and the winning EBM
config (`study.best_trial.params`), vs base 0.12516 / lgbm-subset 0.12346 / production 0.12807. Merge
the two clusters' bests (independent studies, §2 of the architecture). Then it's notebook-write-up time
(verify-FIRST, professor-facing — see the `verify-before-interpret` discipline).

## 7. Rules / watch-fors

- **No `scancel` by the agent.** Stale runs are human-only (handoff convention).
- **Drive cluster ops through hpc-agent / the driver**, not hand-rolled ssh *polling* (CARC banhammer);
  single one-shot `ssh`/`scp` of small files via `usc-discovery` / `hoffman2` is fine (used all session).
- **Gates:** manifests set max-iters 25, max-core-hours 400 per cluster. The driver stops at 25 finished.
- **Hoffman2 SGE is the least-exercised path** — if its driver exited early, read its `drive.log` tail
  for `submit FAILED` / the error, fix, relaunch.
- The first EBM submit on CARC left a harmless dead `ridge_imp-e3b0c442-canary` journal entry (different
  cmd_sha) — ignore it.
