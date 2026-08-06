# Part 8 — COVID-slice tree tuning, window ablation, and the Ridge-residualizer (2026-06-26)

A fast, hard-regime redirect of the tree work: instead of the multi-day full-OOS
campaign (Part 7), tune every tree combo on a **single high-vol slice** (COVID
2020), run the **train-window ablation** there, and stand up the **residualized
multi-stage pipeline** (Ridge-as-residualizer vs raw tree). The slice makes the
search cheap enough to do the *proper* thing (re-tune at every window) rather than
the biased fixed-config shortcut.

> Status at write time: numbers below tagged **[search]** are coarse-refit,
> rank-only (Part 7); **[r1]** are refit=1 validated and comparable. Tables with
> TBD are pending the sweep + validation.

## The COVID slice — anchored by calendar, not a transplanted chunk index

Part 7's "chunk 38 ≈ COVID" was on the **Hoffman2** feature build; on the CARC
tree matrix (nX=242,934, 48 bars/day) chunk-38-of-100 lands on a **calm** period.
So the slice was re-located by **peak rolling realized vol in 2020**:

- **Emit window `X[189713:192023]` = 2020-02-25 09:30 .. 2020-04-30 18:00** (2310
  bars ≈ 48 trading days), the highest-vol 2020 window. (The global vol peak is
  2008-10 GFC; COVID is the 2020 peak, the intended stress regime.)
- It is late in the 2005–2024 sample, so **all ablation windows fit** with huge
  headroom: even train_win=2000d (96,000 bars) leaves `emit_start − 96000 = 93,713`
  rows of history. Decoupling train_win from the slice position (the slice is FIXED;
  only the preceding training window grows) is the Part-6 `FIXED_OOS_DAYS` lesson.

## Method — one 32-cell sweep = tuning (W1) + window ablation (W2)

`{lgbm, xgb} × {sentiment, implied_vol, vol_demand, all_buckets} × {250,500,1000,2000}d`,
each a **self-contained Optuna tuner** on the slice (`tune_covid.py`: ask →
fit-one-slice → QLIKE → tell, resumable sqlite study, n_est ≤ 300). One fit per
trial, no chunk-array/async machinery (single-slice removes the reason it existed).

- **Re-tune at every window, not a fixed config.** Fixing a 250d-tuned config
  confounds the window effect with config-suboptimality: tree capacity (depth,
  n_est) should grow with training data, so a config tuned for 12k rows is
  handicapped at 96k and biases the result *toward* 250d — the very thing under
  test. Re-tuning lets each window compete at its own best config (unbiased).
- **Search at a window-scaled coarse refit, validate winners at refit=1.** Part 7:
  tree-config *ranking* is ~refit-invariant (Spearman ≥0.93 on this COVID slice up
  to refit≈100–250). Search refit = {250:10, 500:20, 1000:40, 2000:80} keeps
  per-trial compute ≈ constant across windows (231×12k ≈ 29×96k row-fits) so even
  2000d cells finish in walltime. Coarse-refit *absolute* QLIKE is rank-only, so
  every cell's winning config is re-scored at **refit=1** (`validate_covid.py`).
- **HAR-only control arm** (`tune_features/har`, p=13): tree on HAR+calendar alone,
  so "do exog help the tree on COVID at all?" is answerable.

**Honest scope caveat:** a single-slice optimum is regime-specific. This answers
"is 250d enough *under a vol explosion*" — a pointed stress test, not the
full-sample optimum (that needs a full-OOS ablation like Part 6's Ridge).

### Results — tuning + window ablation (refit=1 validated) — TBD

| model | bucket | 250d | 500d | 1000d | 2000d | argmin |
|---|---|---|---|---|---|---|
| lgbm | har | … | … | … | … | … |
| lgbm | sentiment | … | … | … | … | … |
| lgbm | implied_vol | … | … | … | … | … |
| lgbm | vol_demand | … | … | … | … | … |
| lgbm | all_buckets | … | … | … | … | … |
| xgb | … | … | … | … | … | … |

Open hypothesis (from Part 6's Ridge result): single buckets ~250d; high-dim
**all_buckets wants a longer window (~1000d)** — trees on all_buckets at 250d
(~23 rows/feature) are likely data-starved. The ablation tests this for trees.

## W3 — Ridge-as-residualizer vs raw tree (the residualized multi-stage pipeline)

Three arms, all on the **scaled impute-indicate matrix** (row-identical to the
tree matrix — impute fills, doesn't drop; nX=242,934, y identical), so
raw-vs-residualized is apples-to-apples (`covid_residualizer.py` via
`MultiStageBacktest`):

- `ridge_alone`    = MultiStage(Identity + Ridge α=1)         — the linear baseline
- `raw_tree`       = MultiStage(Identity + tree)              — tree models y
- `residualized`   = MultiStage(Residualizer(Ridge) + tree)   — tree models y − ridge

The Ridge baseline is sklearn Ridge(α=1) refit every bar on the full 529-feature
scaled matrix (≈ the production 0.12807 model; RollingLeastSquares is just a faster
solver for the same fit). Tree = the W1 COVID-tuned best, same config in both arms.

### Results — 6 single-bucket arms, refit=1, tuned configs

| model · bucket | ridge_alone | raw_tree | residualized | resid−raw |
|---|---|---|---|---|
| lgbm · implied_vol | 0.2120 | 0.2301 | 0.2236 | −0.0065 |
| lgbm · sentiment | 0.2181 | 0.2266 | 0.2202 | −0.0064 |
| lgbm · vol_demand | 0.2165 | 0.2336 | 0.2279 | −0.0056 |
| xgb · implied_vol | 0.2120 | 0.2464 | 0.2193 | −0.0271 |
| xgb · sentiment | 0.2181 | 0.2343 | 0.2186 | −0.0157 |
| xgb · vol_demand | 0.2165 | 0.2376 | 0.2302 | −0.0073 |

**The headline, robust across all 8 cells (refit=1, incl. lgbm all_buckets
0.207/0.235/0.229):** `raw_tree > ridge_alone` **everywhere** — the raw tree *loses*
to the linear model on the COVID slice. Residualizing **always helps** the tree
(`resid − raw` = −0.006 to −0.027, xgb most since raw xgb is worst) — **but never
enough to overtake Ridge**: `tree_adds_over_ridge` = `resid − ridge` is **positive
in all 8 cells** (+0.0005 to +0.022), so **Ridge alone is the best of the three
arms**. (My earlier "stacked tree beats Ridge" was from a partial-tune coarse smoke;
the refit=1 result flips it.) This is the dense-weak-linear-core story at full
strength: the linear hyperplane is so dominant on the COVID stress slice that the
tree is pure drag — residualizing only *shrinks* the drag. Honest caveat: this is the
COVID extreme; on calmer/full-OOS the tree may add genuine value (the open
full-OOS check).

## Reconciling boosting + residualization with Xiu's "dense but weak"

No contradiction — the two findings are the **linear core and the nonlinear
residual of the same dense-weak signal**, and the residualized architecture is
what separates them.

1. **The dense-weak thesis applies to the *exog increment over HAR*, not the whole
   problem.** GKX/Kozak-Nagel-Santosh is about cross-sectional *return* premia (R²
   ~0.4%/mo, low-SNR). Realized vol is highly predictable and HAR already captures
   the bulk. The weak, dense part is what the 41 buckets add on top — and it behaves
   exactly like "Shrinking the Cross-Section": no single bucket dominates (best
   single = moments −0.0035; all stacked = −0.0065 > any few, < naive sum →
   complementary + redundant), captured by **dense shrinkage** (α=1 Ridge over 529
   features), and **data-hungry ∝ dimensionality** (Part 6: all_buckets wants ≈1000d).
2. **The residualization ordering is the reconciliation, and it's in the data.**
   `ridge_alone < raw_tree` (the tree can't beat Ridge) and only `residualized` wins.
   A dense-weak signal is a diffuse weighted sum — worst-case geometry for
   axis-aligned splits — so a from-scratch tree wastes capacity relinearizing the
   HAR + 529-dim hyperplane. Hand it the Ridge hyperplane and it models only the
   thin nonlinear/interaction remainder — the GKX nonlinear increment, second-order
   to the linear core. `Ridge → tree` is the correct inductive bias for
   "dense-weak-*linear* core + small nonlinear residual," which is why it beats both
   pure approaches.
3. **Why raw trees win outright in GKX but lose here.** GKX predicts returns with no
   strong linear baseline, so trees' nonlinearity has the field. We have a dominant
   linear baseline (HAR autoregression), so a raw tree fights Ridge on its home turf;
   residualizing removes the handicap and isolates what trees are for.

**Caveats:** COVID-slice + (currently) coarse-refit numbers; GKX genuinely credits
nonlinearity (we *locate* it on the residual, not deny it). If the full-OOS raw
tree closes the gap on Ridge, the linear-core dominance is partly a COVID-stress
artifact — worth a full-OOS residualizer check.

## Artifacts (cluster `/scratch1/jc_905/harxhar-clean/`)

- `covid_slice_locate.py` — calendar-anchors the slice (saves `dates.npy`).
- `tune_covid.py` + `run_covid_tune.sbatch` — the per-cell slice tuner.
- `validate_covid.py` + `run_covid_validate.sbatch` — refit=1 re-score.
- `covid_setup_imp_features.py` — scaled impute-indicate Ridge matrices (W3).
- `covid_setup_har.py` — HAR-only control matrix.
- `covid_residualizer.py` + `run_covid_resid.sbatch` — the 3-arm residualizer.
- `covid_stacked.py` + `run_covid_stacked.sbatch` — the elastic stacker on the slice.
- `tune_covid_aggressive.py` + `run_covid_aggr.sbatch` — aggressive fit-once tuner.
- `covid_ridge_rank.py` + `run_covid_rank.sbatch` — robust-scale vs rank-Gauss A/B.
- `validate_covid_topk.py` + `run_covid_validate_topk.sbatch` — top-K refit=1 validation.
- `covid_collect.py` → `results/covid_tree_sweep.csv` + `results/covid_residualizer.csv`.

New repo models/transforms (lint-clean): `src/models/stacked.py` (elastic stacker),
`src/features/transforms/rank_gauss.py` (causal rolling rank-Gauss).

---

# Part 9 — The elastic stacker model (`src/models/stacked.py`)

A first-class pipeline model, not a one-off: **Ridge base + a non-negative ridge
stack over heterogeneous tree learners on the residual.**

    final = ridge.predict(X) + Σ_k w_k · base_k(X → y − ridge),   w_k ≥ 0

Reuses the existing harness unchanged: `MultiStageBacktest(Residualizer(Ridge) +
StackingResidualRegressor)`. The stacker is **causal/leakage-free by construction**:
the walk-forward already yields out-of-sample base predictions, and within each fit
the meta-weights are solved on a **held-out tail** of the training window (bases
trained on the head, predicting the tail) with an optional embargo — non-negative
ridge via NNLS on a ridge-augmented design. The zero-floor *is* the elastic knob:
weights can shrink any base (including all → 0 = pure Ridge), so the data picks the
bias-variance blend between tail-chasing boosting and variance-controlled RF/ridge.
`weight_log` records per-refit weights → we can see *where rf earns weight vs gbrt*.
Verified: synthetic `STACKER OK preds finite=True`, ruff clean. Running on the slice
(4 buckets × 4 windows).

**Why this is the right architecture (the dense-weak synthesis):** GKX/KNS say the
exog increment over HAR is weak + dense + *linearly* structured → Ridge captures it;
the W3 result (`raw_tree > ridge` everywhere) confirms trees can't extract it alone.
The stacker puts Ridge first (the dense linear core) and lets trees mop up only the
thin nonlinear residual — the correct inductive bias for "dense-weak-linear + small
nonlinear." RF/DART are *subsumed* as base learners, not separate models.

# Part 10 — Rank-Gauss & a preprocessing audit (`rank_gauss.py`)

Causal rolling **rank-Gauss** `Φ⁻¹(F_t(x))`: pins a feature's *whole* marginal to
N(0,1) in every window — invariant to location, scale **and shape**. Motivated by
the professor's point on **non-stationary shape drift**: affine robust-scale tracks
location/scale drift but skew/kurtosis are affine-invariant, so a feature reshaping
symmetric→fat-tailed across regimes still drifts under robust-scale; rank-Gauss
pins it. That's **first-order** in finance (regime-dependent higher moments) and
exactly the cross-regime transfer problem the COVID slice embodies. It's the
standard empirical-AP move (GKX rank chars per period) for **non-stationarity
control**, not just outlier robustness — and in a non-stationary world the
*percentile* is often the more stable signal than the raw level (rebutting the
"rank discards magnitude" objection).

**Two structural wins:** (1) **division-free** → eliminates the entire
divide-by-degenerate-scale failure class (the Part-3 bugs); the std-floor patch
becomes moot. (2) It **subsumes the marginal layers** (semantic root/log, winsorize,
robust-scale + IQR-guard) into one transform — a redundancy cleanup. It does **NOT**
subsume diurnal adjustment (conditional on time-of-slot, orthogonal — apply diurnal
FIRST, then rank the de-seasonalised residual) nor the hurdle occurrence indicators
(genuinely additive). No-op for trees (monotone). A/B running: Ridge robust-scale vs
rank-Gauss, 4 buckets × {250,1000}d.

**Preprocessing redundancy found:** every semantic rule is monotone (sqrt/cbrt/log/
4th-root/identity), so for the **tree path the entire feature-value transform stack
is a no-op** — only the hurdle/availability indicators + NaN-handling + target
transforms matter. On the Ridge path the per-feature root/log + winsorize + IQR-guard
are three *overlapping* tail-handling layers that rank-Gauss would collapse.

**A/B RESULT — the *marginal* rank-Gauss as a Ridge scaler HURTS, decisively.**
Ridge on the slice, robust-scale vs `rolling_rank_gauss` (pooled marginal rank),
4 buckets × {250,1000}d: rank − robust = **+0.013 to +0.125** (worse everywhere,
badly at 1000d — e.g. implied_vol 0.207→0.332). The magnitude-loss objection wins
empirically: pooled rank discards the *level* that vol forecasting needs, and the
shape-drift gain doesn't compensate. **Caveat / not yet refuted:** this is the
*marginal* (pooled-over-slot) rank. The *per-slot conditional* diurnal-rank
(`diurnal_mode="rank"`, Part 13) is a different transform — it removes only the
diurnal seasonality and keeps more cross-slot magnitude — so it must be A/B'd
separately before concluding rank is dead for the linear path.

**Per-slot rank wired (Part 13).** `diurnal_rank` + `diurnal_mode` are now in the
pipeline (`target.py`/`executor.py`/`ridge.py`/`stacked.py`), default `"divide"`.
Validated on voldemand: std 84 → 0.50, division-free, target baseline intact, both
indicator families preserved; the hurdle active-scaling auto-disabled under rank.
The dedicated COVID-slice A/B (build the impute matrix with `diurnal_mode="rank"`,
compare Ridge QLIKE) is the open follow-up.

# Part 14 — Cluster saturation + distributed constant_liar (2026-06-26, cont.)

The sequential slice tuners never fired `constant_liar` (it only decorrelates
*concurrent* in-flight trials; ask-fit-tell has none). Fix: **batched-parallel asks
under constant_liar** (`tune_covid_batched.py`, K=4 GIL-releasing threads) + a fat
random startup, then **distributed** — many worker jobs sharing ONE study via
`JournalStorage` (NFS-safe; sqlite locks flakily on scratch), so constant_liar
decorrelates cluster-wide. **CARC saturated**: 64 workers (8/combo) → 92/100 running.
**Result: it works** — the fleet ran 1,300–2,000 trials/combo and found markedly
better fit-once configs (xgb_implied_vol 0.191 vs the sequential-aggressive 0.246).
Hoffman2 brought up as a second front (optuna 4.9 installed; matrix build → slice
re-anchor → lgbm fleet, spacing SSH for the login-node throttle). The widened space
opened the binding bounds (min_child→500, colsample→0.1, lr→0.003) the first sweep
hit at the boundary. Next: top-K refit=1 validation of the fleet's best configs —
does a heavily-tuned tree finally beat Ridge on the slice?

# Part 11 — Aggressive fit-once tuning (`tune_covid_aggressive.py`)

Fit-once (no intra-slice refit, ~1 fit/trial, validated rank-preserving in Part 7)
makes trials nearly free, so we tune **aggressively**: 200 trials/combo on a
**widened** search space that opens the bounds the first sweep hit at the boundary —
`min_child_samples` was pinned at the 100 cap, `colsample_bytree` at the 0.30 floor,
lr low. Those binding constraints said the optimum wants *more* regularization than
allowed, so the aggressive space opens `min_child→500`, `colsample→0.1`, `lr→0.003`.
8 combos running; winners re-validated at refit=1 via the top-K validator (fit-once
argmin can flip, so validate the top *cluster*, not top-1).

# Part 12 — Methodology threads (this session)

- **Selection / in-sample bias on the slice.** Per-config walk-forward is causally
  clean; the optimism lives in the **argmin over configs evaluated on the scoring
  sample** (winner's curse, `E[min L̂] < min L`), amplified by the short,
  autocorrelated, fat-tailed slice. Affects *levels*; *rankings* largely differenced
  out. refit=1 validation fixes the coarse-refit bias but **not** the config-selection
  bias — only a held-out / full-OOS eval does. (Same logic kills the λ-blend as an
  *in-sample* tuned scalar; causal/held-out λ is fine.)
- **Why not test residualizer × window (18 jobs):** the residualized model's window
  behavior ≈ inherited from its two stage curves (Ridge ~1000d Part 6; raw-tree from
  W2). Residualizing makes the *tree* less data-hungry (fits a tree-friendlier
  residual), so the "does residualizing help" benefit is a small-data crutch that
  *shrinks* with window — not grows. Redundant to sweep.
- **DART:** unfavorable for us — slower + wants more trees (worsens the refit-bound
  cost), adds RNG noise (more selection bias) + hyperparams, and is largely redundant
  with the RF-ward regularization the tuner already chose. Test only as a cheap
  fit-once arm; don't default.
- **Diurnal-std floor critique:** the `0.1×typical` fix uses a *global* median
  (mild look-ahead) vs its causal-expanding IQR sibling, is a hard kink (a soft
  `√(std²+floor²)` is cleaner), and is a band-aid on the deeper signed-÷-degenerate-
  std mis-design that rank-Gauss removes outright.

---

# Part 15 — Auditable-campaign pivot, fleet-seeding, and the validate-not-drive call (HANDOFF)

This part is the **resume-from-cold state**. Read `writeup/CAMPAIGN_RUNBOOK.md` for the
step-by-step drive procedure; this section is the *why* + the in-flight results.

## The pivot (trust) and the seeding (efficiency)

The bespoke distributed fleet (Part 14) was un-auditable (no manifest / validate /
budget gates). Pivoted the tree tuning to a proper **hpc-agent campaign**: 8 campaigns
`covid_{lgbm,xgb}_{sentiment,implied_vol,vol_demand,all_buckets}`, each a path-B Optuna
strategy (`.hpc/tasks.py`) that asks TPE+constant_liar over the widened space
(`src/backtest/tree_space.py`, single source) and scores each trial on the COVID slice
through the **validated `run_executor`** (slice exposed via new `start/end/halo` args in
`lightgbm.run`/`xgboost.run`; fit-once). Inter-iteration `tell` is wired
(`tasks.py._tell_from_results`, reads combo-scoped synced results). `campaign init`
written for all 8; manifests at `.hpc/campaigns/<cid>/manifest.json`.

To not discard the fleet's compute, each campaign's Optuna study was **warm-started**
from the fleet's trials — `export_fleet_trials.py` (snapshot the dist/aggr/batch
studies) → `seed_campaign.py` (top-`N_SEED=1000` by value → `add_trial`). All 8 studies
seeded (1000 trials each) at `.hpc/campaigns/<cid>/optuna.db` (local + cluster). Seed
bests (fit-once, rank-only): lgbm {sent 0.2103, iv 0.1948, vd 0.2154, all 0.2354};
xgb {sent 0.2195, iv 0.1878, vd 0.2003, all 0.2270}.

## Two bugs / lessons (durable)

1. **Fleet over-ran 8× (budget bug).** `tune_covid_batched.py` tracked the stop
   condition with a LOCAL `done += b` instead of re-reading the shared study's
   finished-count, so each of the 8 distributed workers ran the full `MAX_TRIALS=3000`
   → ~24k trials/combo. The search was valid (shared JournalStorage TPE, real asks);
   only the stopping was wrong. **This is exactly the silent bug the campaign's
   `max_jobs`/`max_core_hours`/convergence gates would have caught** — the strongest
   concrete argument for the pivot. `tasks.py` re-reads the study count correctly.
2. **Seeder needed `mkdir`** for the campaign dir before sqlite write (5 combos failed
   silently until added).

## "Do we still need the campaign?" — No (for THIS question)

With ~24k trials already explored, the campaign as a *search engine* is redundant — the
best configs are in the seeded studies. The real remaining step the fleet never did is
**refit=1 validation of the best config per combo** (fit-once QLIKE is rank-only).
Decision: **skip driving the campaigns; validate the 8 seeded bests at refit=1** and
compare to Ridge. (The campaign infra + runbook remain the auditable substrate for
*future* tuning.) honest meta: the trust-resolving step is the held-out validation, not
re-running a done search.

## Plateau analysis — corrected an overconfident claim

`best_so_far_analysis.py` (best-so-far vs trial-number per combo): the search did **NOT**
plateau early — convergence (within 0.5% of final) at **28–97%** of trials; several
combos (vd 97%, xgb iv 89%, all 80%) improved nearly to the end, some sizeably
(`lgbm_vol_demand` −0.018 after trial 1000). So "the 24k was wasted" was **wrong on the
slice metric**. BUT: fit-once-slice best-so-far decreases mechanically with #trials
(winner's curse), so late gains are real-signal + slice-overfit in unknown proportion —
the refit=1 validation is the arbiter. (Lesson: reason from the curve, not a vibe.)

## Tuned-Ridge baseline (the bar trees must beat)

`covid_ridge_alpha.py` — Ridge α was frozen at 1.0 (unfair vs tuned trees). Tuning α:
single buckets ~unchanged (α≈1 near-optimal); **`all_buckets` wants α≈3160, gaining
−0.0020 → 0.20495** (it was badly under-shrunk — dense-weak/KNS: high-dim needs strong
shrinkage). So the fair bar is Ridge-alone ≈ **0.205 (all_buckets)** to 0.218 (sentiment).

## IN-FLIGHT when cleared (check these results on resume)

- **8 refit=1 validations of seeded bests** — jobs `9624274–81`. Output:
  `results/seeded_validation/<m>_<b>.json` (cluster) + `SEEDVAL` log lines. THIS is the
  headline answer: does a tuned tree beat Ridge (0.205–0.218) at refit=1 on the slice?
  Driver: `validate_seeded_best.py M B 3` (reads the seeded study top-3, refit=1).
- **Full-OOS rank A/B/C** — job `9623818`. Builds `results/covid_imp_rank/` (per-slot
  rank-diurnal matrices) then `full_oos_ridge_rank_ab.py` per bucket →
  `results/full_oos_rank_ab/<bucket>_tw250.json` (A=divide+robust, B=marginal-rank,
  C=**the wired per-slot diurnal-rank** — its real, full-OOS test, vs the slice A/B
  where marginal rank lost +0.08–0.12).
- **Refit=1 sweep validations** (#4) — `covid_val` jobs finishing.

## Pending (non-cluster)

- **`/sync`** — ALL session code uncommitted: `src/models/stacked.py`,
  `src/features/transforms/rank_gauss.py`, `src/backtest/tree_space.py`, the
  `diurnal_mode` wiring (`target.py`/`executor.py`/`ridge.py`/`stacked.py`), the slice
  exposure (`lightgbm.py`/`xgboost.py`), `.hpc/tasks.py` + manifests, `seed_campaign.py`,
  and the writeups. **Commit before relying on any of it.**
- Optional: re-run W3/stacker with tuned-α Ridge base (Part 8 used α=1).
- `matplotlib` absent in the `harxhar` env — the best-so-far plot didn't render (table
  in `best_so_far_analysis.py` output carries it).

## To resume after clear

1. Check the IN-FLIGHT results above (the SEEDVAL refit=1 numbers are the answer).
2. `/sync` the code.
3. If you want the auditable campaign actually driven: follow `CAMPAIGN_RUNBOOK.md`
   (but per "do we need it", validation already gives the science).
Cluster: `ssh usc-discovery`, env `harxhar`, repo `/scratch1/jc_905/harxhar-clean`.
No `scancel` from the agent — the human runs it.
