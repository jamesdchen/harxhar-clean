# Full-Stack Residualized-Tree Campaign — Cold-Start Handoff

**Purpose.** Drive, from a cold session, the campaign that tunes an **EBM residualizer** on top of an
**elastic-net linear base**, over the **slim + pruned** all_buckets feature matrix, **full-OOS**, via the
**amortized chunked scorer**, **hpc-agent-driven**, across **CARC (SLURM)** + **Hoffman2 (SGE)**.

> Supersedes `writeup/CAMPAIGN_RUNBOOK.md` (that was the COVID-slice lgbm/xgb tuning). This is the
> full-stack, full-OOS campaign. Status markers: **[built]** = on disk + verified; **[pending]** =
> remaining build (see §9); **[TBD]** = waiting on the in-flight A/B (§8).

---

## 1. TL;DR — the one decision the campaign answers

Does an EBM, fit on the **nonlinear residual** of a strong **elastic-net** linear base, add QLIKE over the
linear base alone, **on the calm full-OOS sample** (not just the COVID stress slice)? The bar to beat is the
**elastic-net base = 0.12530** (all_buckets, full-OOS, fixed-OOS region). Ridge-α100 = 0.12565; production
all-buckets = 0.12807. If `resid_ebm − enet_base < 0` robustly, the tree earns its place off the extreme.

---

## 2. The architecture (the full stack)

| layer | choice | why |
|---|---|---|
| **matrix** | `covid_imp_rank/all_buckets` — per-slot **rank**-diurnal, **no robust-scale, no semantic** | ablation `C_norob_all`: rank kills the divide-by-degenerate-std bug class; robust-scale & semantic are dead/harmful |
| **prune** | drop the **224 signalless indicators** → ~305 features | dead to BOTH enet-all-windows AND a deep residual tree (the `_avail/_active` MAs of always-present stock stats are constant) |
| **base** | **elastic net** α≈0.001, l1_ratio≈0.2 (residual = `y − enet`) | beats Ridge-α100 (0.12530 vs 0.12565); supervised light-sparsity is the DR that works (PCA/PLS truncation fails — dense signal) |
| **tree** | **EBM only** (`ExplainableBoostingRegressor`) — Optuna-tuned | bagged boosted GAM = the smooth/additive/bagged residual shape the tuning kept pointing at; interpretable shape functions |
| **window** | base/cache `train_win = 1000d`; tree refit cadence `480` (~10 trading days) | all_buckets wants ~1000d for the base (Part-6); coarse tree cadence is rank-preserving (Part-7) |
| **search** | regular TPE (constant_liar OFF), seeded with "bagging" configs | under a 100-job cap, telling real results beats lying about in-flight trials |
| **saturation** | each trial fans across **CHUNKS≈90 time-chunks** (cadence-block partition) | K=1 sequential TPE + 90 concurrent chunk-jobs → full surrogate quality AND ~100-slot saturation, per cluster |
| **clusters** | EBM campaign on **CARC** + a mirror on **Hoffman2** | both ~100 concurrent; no shared FS → two independent studies, merge bests |

---

## 3. Verified facts & lessons (fold-in)

- **Amortized scorer is exact** — `max|amortized − naive MultiStage| = 5.06e-13` (self-check, slim all_buckets).
  The base is **config-independent → precomputed once per cell**; trials fit only trees.
- **Chunk fan-out is exact** — `max|whole − concat-of-chunks| = 0.000` (cadence-block partition ⇒ each tree-refit
  in exactly one chunk, no repeated work). Smearing is **deferred to collect (GLOBAL)** — per-chunk smearing would
  use a per-chunk retransformation factor and not equal the whole-backtest QLIKE.
- **Cadence-refit base** (refit=480) replaces every-bar: the every-bar incremental Ridge on all_buckets·tw1000·529
  is too slow; cadence is internally consistent (base + tree both refit on cadence) and the amortization makes it free.
- **α / base ranking** (full-OOS, fixed-OOS, cadence): Ridge α-scan → **α\*=100 → 0.12565**; enet `α0.001 l10.2`
  → **0.12530** (best); enet heavy-L1 and PLS/PCA truncation all worse. PCA causal (leak ~1e-13) + low-rank
  (K5=72%, K20=90% var) but truncation **loses** at proper α (best pca20 = 0.12890) — the signal is **dense**.
- **enet sparsity**: keeps **~98–120 / 529** (kills 77–81%) — so the effective dimension is ~120, not 529 (full)
  and not 10–40 (PCA). Warm-started CD makes the rolling enet refits cheap (no exact rank-1 for L1).
- **Signalless prune**: **224** features dead to both enet (all windows) AND a deep residual tree — the
  `_avail/_active` MA indicators of the **always-present** stock-microstructure families (31: `turnover`,
  `buy/sellturnover`, `spread`, `effspread`, `sumret{,2,3,4}`, `sumabsret`, `sumbipow`, `sumpret2`, `sumautocov`,
  `sumvolume`, `numobs` × raw/ew/vw). Survivors: all `adj_*` values, the missing-data indicators (`vix/vvix/vix3m`,
  `stocktwits_*`, `voldemand_*`), HAR, calendar.
- **EBM speed**: ~47 s/fit (48000×~290, max_rounds=500, outer_bags=4) → **5.3 hr/trial serial (infeasible)** but
  **~3.5 min/trial chunked** (407 blocks / 90 chunks). Caps `max_rounds=200, outer_bags=2` cut it ~4×.
- **fixed_cols bool-mask bug** (durable): `list(map(int, mask))` → `[0,0,1,1,…]` int list → numpy reads it as
  **fancy indices, not a bool mask** → only cols 0,1 "fixed" → indicator MAs blew up under robust-scale →
  full-OOS Ridge wrecked (slice/trees/production were safe). Fix: store `np.where(mask)[0]` indices. Diagnostic
  signature: flat-across-eras QLIKE + one bucket surviving = structural bug, not signal.
- **Cluster facts**: CARC = SLURM (`discovery2.usc.edu`, env `harxhar`), Hoffman2 = SGE
  (`hoffman2.idre.ucla.edu`, env `hpc-pi`). Both `max_concurrent_jobs=100`, `max_array_size=100` (config fixed from
  the stale `2`/`50`). **Use hpc-agent for cluster ops** — raw-ssh polling trips CARC's login banhammer; `rsync`
  absent in Git Bash (hpc-agent uses scp/tar fallback); hand-rolled `#SBATCH` breaks on Hoffman2's SGE.

---

## 4. Artifacts (files + locations)

**Repo (local + both clusters):**
- `resid_amortized.py` — the scorer/CLI (modes in §5). The heart.
- `src/features/transforms/rolling_pca.py` — causal rolling PCA (built; PCA shelved as base, may revisit for tree).
- `src/models/resid_tree.py` — `run.py`-dispatchable trial (whole + single-chunk mode).
- `.hpc/tasks.py` — campaign strategy: residualized objective branch, chunk fan-out, regular TPE, `_tell_from_chunks`
  (global smear). `OBJECTIVE=residualized` selects it.
- `src/backtest/tree_space.py` — `widened_space("lgbm"|"xgb"|"ebm")` (single source; **ebm** space added).
- `seed_resid.py` — enqueue "bagging" seed configs from an existing study into a residual study.
- `covid_setup_imp_features.py` — matrix builder (slim/rank via `diurnal_mode="rank"`; the fixed_cols fix; `use_semantic`).

**Cluster paths:** CARC `/scratch1/jc_905/harxhar-clean`; Hoffman2 `/u/scratch/j/jamesdc1/harxhar-clean` (`data` symlinked → `../harxhar/data`).

**Slim matrix** (byte-identical on both): `results/covid_imp_rank/all_buckets/{X_imp.npy(~1GB), y.npy, base.npy, meta.json}`.

**Amortized cache** (per cell): `results/resid_prep/<cell_id>/` with `Xs.npy`, `y.npy`, `base.npy`, `ridge_oos.npy`
(base OOS preds), `cadence.npz` (`starts`, `coefs`, `intercepts`), `masks.npy` (per-block enet survivors),
`prunable.npy` (224-mask), `cell.json`. `cell_id = <model>_<bucket>_tw<TWD>_a<alpha>_rf<refit>_<pipe>`.
Built so far: `lgbm_all_buckets_tw1000_a100_rf480_slim` (Ridge base). **[pending]** enet-base + ebm cells (§9).

---

## 5. The scorer CLI (`resid_amortized.py <mode> …`)

- `prep MODEL BUCKET TWD ALPHA REFIT PIPE` — build the amortized cache (cadence base + oos). **[pending]** `base=enet` variant.
- `enet_masks CELL_ID` — add rolling enet survivor masks + the prunable (224) mask to a built cache.
- `chunk_task CELL_ID ARM IDX NCHUNKS` — one chunk (array task); `ARM ∈ {residualized, resid_pruned, resid_subset, raw_tree, ridge_alone}`; tree cfg via env `TREE_CFG`.
- `chunk_collect CELL_ID ARM` — concat chunk CSVs → **global** smear → full-OOS QLIKE.
- `alphascan / pca_alpha_compare / dimreduce_cell / signalless_scan / pcawindowsweep / chunkcheck / selfcheck` — the analyses behind §3 (reproducible).

---

## 6. Cold-start drive procedure

### Step 0 — environment
```
cd <repo>                                  # CARC: /scratch1/jc_905/harxhar-clean
hpc-agent --version                        # expect 0.10.65+
hpc-agent clusters list                    # carc (slurm), hoffman2 (sge)
hpc-agent preflight --cluster carc         # all_ok:true (multiplexed ssh; do NOT raw-ssh poll)
hpc-agent preflight --cluster hoffman2     # all_ok:true
```
Cluster env python: CARC `/home1/jc_905/.conda/envs/harxhar/bin/python`; Hoffman2 `~/.conda/envs/hpc-pi/bin/python`
(numpy 2.2.6 / sklearn 1.7.2 / lgbm 3.3.5 / xgb 3.0.5 / optuna 4.9 / interpret 0.7.8).

### Step 1 — prerequisites (once per cell, per cluster)
1. Slim matrix present (`results/covid_imp_rank/all_buckets/`). If not: `python covid_setup_imp_features.py all_buckets rank` (needs `data/`).
2. **[pending]** Build the **enet-base** cache: `python resid_amortized.py prep ebm all_buckets 1000 <enet-as-base> 480 slim` (the `base=enet` prep — §9).
3. Masks + prune: `python resid_amortized.py enet_masks <cell_id>`.
4. Seeds: `python seed_resid.py <campaign_id> ebm all_buckets .hpc/campaigns/<seed_study>/optuna.db <seed_study_name> 50`.

### Step 2 — campaign init (per cluster)
```
hpc-agent campaign init --campaign-id ebm_all_buckets_<cluster> \
  --metric qlike --direction minimize --strategy-name covid_resid_tune \
  --max-iters 25 --max-core-hours <budget> \
  --strategy-params-json '{"OBJECTIVE":"residualized","MODEL":"ebm","BUCKET":"all_buckets",
     "TRAIN_WIN_DAYS":1000,"BASE":"enet","ALPHA":1,"TREE_REFIT":480,"PIPE":"slim","ARM":"resid_subset",
     "CHUNKS":90,"K":1,"MAX_TRIALS":200}'
```
(`K=1` = pure-sequential regular TPE; each asked trial fans into `CHUNKS` chunk-tasks. **`PIPE=slim`** is the MATRIX
(→ cache cell `ebm_all_buckets_tw1000_enet_rf480_slim`); **`ARM=resid_subset`** is the tree column selection (the
rolling ~112 enet survivors, §8). `BASE=enet`; `ALPHA=1` is an ignored placeholder under `BASE=enet` (enet base is
fixed α≈0.001, l1≈0.2).)

### Step 3 — drive the loop (per cluster, repeat until converged/over-budget)
Preferred — one tick via the skill: `Skill(hpc-campaign){experiment_dir:".", campaign_id:"ebm_all_buckets_<cluster>"}`
(runs validate → submit-flow → monitor-flow → aggregate → `campaign advance`). Repeat until `advance` returns
`stop_converged` / `stop_over_budget` / max_iters. The submit-flow deploys code (scp/tar) + emits the scheduler-correct
array (SLURM on CARC, SGE on Hoffman2). **Never raw-ssh poll** (banhammer); **never `scancel`** (human-only).

### Step 4 — collect / validate
Per-trial QLIKE is closed in `_tell_from_chunks` (concat chunk CSVs → global smear → tell). After convergence, the
winning EBM config's full-OOS residualized QLIKE vs the enet base (0.12530) is the headline.

### Hoffman2 specifics
Cache is **model-independent** → copy the CARC cache to the xgb/ebm cell (via the DTN `h2dtn`, the right endpoint for the
~1 GB `Xs.npy`), not rebuilt. SGE: `max_walltime` short — keep chunks ≤ the per-job limit. Space login-node connections.

---

## 7. Gates / safety

- `--max-core-hours` + `--max-iters` + convergence (the Part-15 fix for the fleet's silent 8× over-run).
- The chunk arrays respect `max_array_size=100` (CHUNKS≈90 fits).
- **No `scancel` by the agent** — the human cancels (e.g. stale `covid_stack`, `resid_val`).
- Verify-first: `selfcheck` (scorer exactness) + `chunkcheck` (chunk concat) must pass before trusting numbers.

---

## 8. [RESOLVED] — A/B verdict: `PIPE=subset`

lgbm residual tree, fixed bagging config, Ridge-α100 base, full-OOS fixed region:

| arm | features | QLIKE |
|---|---|---|
| `resid_subset` (rolling ~112 enet survivors) | ~120 | **0.12346** |
| `resid_pruned` (224-prune) | ~305 | 0.12381 |
| ridge base | 305 | 0.12565 |

- Residual tree **beats the base** by ~0.002 (both arms) — the tree adds value off the COVID extreme.
- **`subset` ≤ `pruned`** (−0.00035) → restricting the tree to the ~120 enet survivors loses **no** nonlinear-only
  signal (it denoises). **Decision: `PIPE=subset`** — low-dim, adaptive, best. Confirms data-hunger ∝ dimensionality:
  enet's supervised selection is the DR that works for the tree (PCA/PLS truncation did not).
- (This A/B used the lgbm tree as a fast proxy for the feature-set choice; the campaign tunes EBM on the same subset.)

---

## 9. Launch-ready status — START HERE for a cold drive

**Everything is deployed + init'd. The cold session's job is to DRIVE.** Ready:
- **[built]** enet-base EBM cache `ebm_all_buckets_tw1000_enet_rf480_slim` on **CARC** (enet base_alone QLIKE
  **0.12516** = the EBM's bar; 112 survivors, 218 prunable, masks+prunable saved) and on **Hoffman2** (same cache;
  `Xs.npy` symlinked to the shared slim matrix, small files copied).
- **[built]** EBM (interpret 0.7.8) + latest code (incl. `tasks.py` with `BASE`/`ARM` knobs) on **both** clusters.
- **[built]** Both campaigns **init'd**: `.hpc/campaigns/ebm_all_buckets_{carc,hoffman2}/manifest.json`
  (strategy `covid_resid_tune`; params `MODEL=ebm BASE=enet ARM=resid_subset PIPE=slim ALPHA=1 TREE_REFIT=480
  CHUNKS=90 K=1 MAX_TRIALS=200 TRAIN_WIN_DAYS=1000`; gates `max-iters=25 max-core-hours=400`).
- **[verified]** scorer exact (selfcheck 5e-13), chunk fan-out exact (chunkcheck 0.000), lgbm A/B (subset 0.12346 <
  pruned 0.12381 < base 0.12565).

**The one UNPROVEN step (do this FIRST in the cold session):** a campaign **tick** has never run, so the
EBM-through-`tasks.py`→`run.py`→`resid_tree`→chunk→`_tell_from_chunks` path + the tick's cache-read locus are
untested end-to-end. The standalone **EBM chunk verify** (`ebm_verify.sbatch`, job `9629325`) was in flight at handoff
— **check its collect first**: `python resid_amortized.py chunk_collect ebm_all_buckets_tw1000_enet_rf480_slim
resid_subset` (vs the 0.12516 base). If that's sane, drive CARC via §6 Step 3 (`hpc-campaign` tick) — the first tick
is the real integration test; watch for cache-not-found (locus) or arm/cfg errors, then Hoffman2.

**Cold drive = §6 Step 0 (preflight) → Step 3 (`hpc-campaign` tick loop), per cluster.** `campaign init` (Step 2) is
already done — skip it. No `scancel`; no raw-ssh polling.

---

## 10. Decisions log (why, not just what)

- **enet base over Ridge**: marginally better (0.12530 vs 0.12565) + supervised sparsity is the only DR that helped;
  PCA/PLS truncation failed (dense signal). Light L1 only — heavy L1 hurts.
- **EBM over lgbm/xgb for the residualizer**: the residual is smooth/additive/dense; EBM is a bagged additive GAM
  (the tuning's `colsample→0.1` bagging was groping toward exactly this). Interpretable shape functions can show
  whether the residual leans on leverage/semivariance-type features.
- **prune the 224 indicators**: zero-loss (dead to both linear & nonlinear), smaller matrix, cleaner.
- **slim (rank, no-robust, no-semantic)**: rank removes the divide-by-degenerate-std failure class outright;
  robust-scale & semantic were dead/harmful in ablation.
- **chunk-the-time-axis**: the only way to saturate under a 100-job cap with regular (non-constant_liar) TPE — and
  it's exact (0.0 vs whole).
- **cadence base**: every-bar incremental is too slow at 529-dim/1000d; cadence is consistent + amortizable.

Cluster: `ssh usc-discovery` (CARC, env `harxhar`); Hoffman2 via `ssh hoffman2` / `ssh h2dtn` (DTN for big files), env `hpc-pi`.
No `scancel` from the agent.
