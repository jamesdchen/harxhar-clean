# Campaign runbook — driving the COVID-slice tree-tuning campaigns (cold-start)

**Audience:** a fresh session/person with NO prior context. This is self-contained.
Goal: drive 8 auditable hpc-agent campaigns that tune lgbm/xgb tree hyperparameters
to their best **fit-once QLIKE on the COVID slice**, then validate the winners at
refit=1. The Optuna studies are already **warm-started** from ~1000 top trials each
(a prior bespoke fleet), so they converge in a few iterations.

---

## 0. TL;DR — the loop

```bash
cd "C:/Users/james/CC Allowed/harxhar-clean"      # the experiment repo (has .hpc/)
hpc-agent load-context --experiment-dir .          # shows the next step per in-flight run
```
For EACH of the 8 campaigns, repeat one tick until it stops (max_iters=25 or plateau):
drive it with the **`hpc-campaign` skill** (preferred) — `Skill(hpc-campaign)` with
`{experiment_dir: ".", campaign_id: "<cid>"}` — which does validate → submit → monitor
→ aggregate → `campaign advance` for you. Or do it by hand with the CLI verbs in §4.

The 8 campaign_ids:
```
covid_lgbm_sentiment   covid_lgbm_implied_vol   covid_lgbm_vol_demand   covid_lgbm_all_buckets
covid_xgb_sentiment    covid_xgb_implied_vol    covid_xgb_vol_demand    covid_xgb_all_buckets
```

---

## 1. What's already built (the state you inherit)

- **8 campaign manifests** at `.hpc/campaigns/<cid>/manifest.json` (created via
  `hpc-agent campaign init`; metric=`qlike`, direction=`minimize`, max_iters=25,
  `strategy.params` = `{MODEL, BUCKET, K:8, MAX_TRIALS:200}`).
- **8 seeded Optuna studies** at `.hpc/campaigns/<cid>/optuna.db` (study name
  `camp_<MODEL>_<BUCKET>`), each warm-started with the **top ~1000** fleet trials.
  Seed bests (fit-once QLIKE, rank-only): lgbm implied_vol 0.19479 · xgb implied_vol
  0.18781 · lgbm sentiment 0.21032 · lgbm all_buckets 0.23542 · xgb all_buckets 0.22698.
- **`.hpc/tasks.py`** — the path-B strategy. Per iteration it (a) **tells** the study
  from the prior iteration's synced trial results (`_tell_from_results`), (b) **asks**
  K=8 trials from TPE+constant_liar over the widened space (`tree_space.widened_space`),
  (c) writes each trial as a `run.py` config targeting the COVID slice.
- **Slice exposed** in `src/models/lightgbm.py` / `xgboost.py` `run()` via
  `start/end/halo`; a trial scores through the **validated `run_executor`** (same code
  as the production pipeline — nothing bespoke).
- **`src/backtest/tree_space.py`** — single source of the widened search space (used by
  both `tasks.py` and the seeder, so seeds are in-distribution).

The COVID slice = emit window `X[189713:192023]` (2020-02-25..04-30) on the CARC tree
matrix (nX=242934, 48 bars/day); a trial fits ONCE on the preceding `train_win`
(250d=12000 bars) and predicts the slice (`refit_frequency = 2310 = fit-once`).

Cluster: **carc** (`ssh usc-discovery`, user jc_905, SLURM), env
`/home1/jc_905/.conda/envs/harxhar`, remote repo `/scratch1/jc_905/harxhar-clean`.

---

## 2. FIRST: make sure the local studies are present

Only 3/8 `optuna.db` were pulled local; pull the rest (they live on CARC):
```bash
cd "C:/Users/james/CC Allowed/harxhar-clean"
for cid in covid_lgbm_sentiment covid_lgbm_implied_vol covid_lgbm_vol_demand covid_lgbm_all_buckets \
           covid_xgb_sentiment covid_xgb_implied_vol covid_xgb_vol_demand covid_xgb_all_buckets; do
  scp usc-discovery:/scratch1/jc_905/harxhar-clean/.hpc/campaigns/$cid/optuna.db .hpc/campaigns/$cid/optuna.db
done
```
Verify each has ~1000 trials:
```python
python -c "import sqlite3; print(sqlite3.connect('.hpc/campaigns/covid_xgb_implied_vol/optuna.db').execute('select count(*) from trials').fetchone()[0])"
```
If any are missing on the cluster, re-seed (§6).

---

## 3. Drive a campaign (preferred: the skill)

For each `cid`, invoke `Skill(hpc-campaign)` with `args` naming the experiment_dir and
campaign_id. It runs ONE tick (submit OR monitor OR aggregate OR advance — whatever
`load-context` says is next) and returns. Call it repeatedly until the campaign's
`advance` returns `stop_converged` / `stop_over_budget` / max_iters. The skill composes
`hpc-submit` / `hpc-status` / `hpc-aggregate`; read each sub-skill's return via
`hpc-agent fetch-skill-return --skill <sub> --experiment-dir .`.

Drive all 8 by looping the skill over the 8 cids (they're independent; CARC has the
slots — 8 × K=8 = 64 trials/iteration fits its ~50–100 cap). **Do NOT use Hoffman2** —
it's `max_concurrent_jobs: 2` and on a different env than its matrices; not worth it.

---

## 4. Drive a campaign (manual CLI, if not using the skill)

```bash
cd "C:/Users/james/CC Allowed/harxhar-clean"
CID=covid_lgbm_all_buckets
hpc-agent load-context --experiment-dir .                      # next step for this campaign
# step == submit:
#   1) hpc-submit resolves the spec from tasks.py + manifest and submits K trials.
#      Use Skill(hpc-submit) {experiment_dir:".", campaign_id:"$CID"}  (it returns a run_id)
#   2) hpc-agent campaign advance --campaign-id $CID --run-id <new-run-id> --experiment-dir .
# step == monitor:  Skill(hpc-status) for the in-flight run_id; wait until terminal.
# step == aggregate: Skill(hpc-aggregate) for the terminal run_id (folds per-trial QLIKE).
# then advance again. Repeat.
hpc-agent campaign status --campaign-id $CID                   # per-iteration reduced metrics
```
The per-trial QLIKE comes from each trial's `results.csv` (written by `run_executor`
under `results/<run_id>/<MODEL>_<BUCKET>/trial_<n>/`). `tasks.py._tell_from_results`
reads those (after the framework syncs them local) and `study.tell`s before the next ask.

---

## 5. Gotchas (read before driving)

- **`scancel` is denied to the agent.** A cold session also cannot cancel cluster jobs —
  ask the human to run `ssh usc-discovery "scancel -u jc_905 --name=<jobname>"`. The
  campaign does NOT auto-resubmit between ticks (one iteration per `advance`), so a
  stalled campaign just stops — it is not a runaway. Submitted trials are short
  (fit-once, minutes).
- **Known framework bug:** `tune_tree._compute_trial_qlike` imports `hpc_agent.template`
  (removed in 0.10.65). `tasks.py` avoids it (uses `_compute_qlike` inline) — do not
  reintroduce that import path.
- **`_optuna_trial_number`** is attached to every task and is load-bearing: it keeps
  `cmd_sha` unique so submit-flow doesn't dedupe identical configs and silently collapse
  the campaign. `validate-campaign`'s `missing_stochastic_marker` guards this.
- **Login-node SSH throttles** burst connections (banner-exchange timeouts / fail2ban) —
  space SSH calls; don't retry-storm.
- **Stale orphan run** `ridge_imp-c6c887dc` may show in `load-context` as in-flight; it's
  dead (job 9580235, Jun 24) — reconcile/ignore it, it's not a campaign.

---

## 6. Re-seed / refresh a study from the fleet (if needed)

Cluster-side (the fleet studies live there): export top trials → seed top-N → pull local.
```bash
ssh usc-discovery 'cd /scratch1/jc_905/harxhar-clean && PY=/home1/jc_905/.conda/envs/harxhar/bin/python && \
  $PY export_fleet_trials.py <MODEL> <BUCKET> results/fleet_seed/<MODEL>_<BUCKET>.json && \
  rm -f .hpc/campaigns/covid_<MODEL>_<BUCKET>/optuna.db && \
  N_SEED=1000 $PY seed_campaign.py covid_<MODEL>_<BUCKET> <MODEL> <BUCKET> results/fleet_seed/<MODEL>_<BUCKET>.json'
scp usc-discovery:/scratch1/jc_905/harxhar-clean/.hpc/campaigns/covid_<MODEL>_<BUCKET>/optuna.db .hpc/campaigns/covid_<MODEL>_<BUCKET>/optuna.db
```
`N_SEED` controls how many top trials to seed (1000 is the sweet spot — more slows both
the sqlite import and every TPE `ask`).

---

## 7. After the campaigns converge — validate the winners at refit=1

Fit-once QLIKE is **rank-only** (Part 7). Re-score each campaign's best config(s) at
refit=1 for a comparable number — top-K guards the fit-once argmin flip:
```bash
ssh usc-discovery 'cd /scratch1/jc_905/harxhar-clean && /home1/jc_905/.conda/envs/harxhar/bin/python \
  validate_covid_topk.py <MODEL> <BUCKET> 250 <SEARCH_REFIT> 3'
```
(Best config is in the study: `optuna.load_study(...).best_trial.params`, or
`.hpc/campaigns/<cid>/best.json` if `tasks.py` wrote one.) Compare to the Ridge
baselines (Part 8 W3): tuned Ridge all_buckets ≈ 0.205 — the bar a tree must beat.

---

## 8. Definition of done

All 8 campaigns reach `stop_converged` or max_iters (25); each best config validated at
refit=1; the table of refit=1 tree-best vs Ridge-alone (Part 8) updated in the session
notes. The open scientific question this answers: **does a heavily-tuned tree finally
beat Ridge on the COVID slice, or does the dense-weak linear core win even after
exhaustive tuning?** (Through Part 8, Ridge wins; the campaigns are the final check.)
