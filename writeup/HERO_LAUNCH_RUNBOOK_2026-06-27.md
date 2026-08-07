# Hero A / Hero B launch runbook (2026-06-27)

Executable checklist so the heroes fire from settled decisions, not improvisation. Cluster =
CARC (`usc-discovery`, `/scratch1/jc_905/harxhar-clean`, env `harxhar`,
`PY=/home1/jc_905/.conda/envs/harxhar/bin/python`, account `pollok_1603`, partition `main`).

## Decision log (settled)
- **Base = fixed-α `enetreg2`** (HAR×{open,close} + cumrv×close, α=0.001 / l1=0.2).
  - online-λ (`enetreg2_olam`) = 0.12324 ≈ fixed-α 0.12314 → **refuted, dropped** (data-driven μ adds
    selection noise, no signal).
  - ratio (`enetreg2_ratio`) = 0.12364 → **refuted** (collapses info the additive pair keeps).
  - cumrv ⟂ HAR×close are **complementary, keep both** (ablation: dropping close-half costs +0.00226).
- **cumrv STACKS with the dig**: EBM 0.12332→0.12276, XGB-d6 0.12215→**0.12146** on enetreg2. New best 0.12146.
- **K = 24** (deliberate; real CARC caps are 100 jobs / 2000 cores, so 24 = TPE-quality + headroom).
- **Search space verified** (`widened_space("xgb")`): `max_depth(2,16)`, `colsample(0.1,1.0)` already
  cover the d6-interp's deep + low-colsample region. No edit needed.
- **d6 interp**: residual is ~90% interaction, order-≥3, diffuse, edge-localized (h17/18/9/19/16);
  HAR+close+hour backbone, cross-sectional vol/flow moments as high-gain refinements ⇒ tuning should
  favor **deep + low-colsample**; EBM's pairwise ceiling is real (relevant to Hero B).

## Pending verdicts (gate the final base form)
- **item-2a** (`enetreg2_real` = post-diurnal physical cumrv, robust-scaled; + `enetreg2_opex`;
  + orthogonalized): does cumrv_real beat the proxy 0.12314? does OPEX add? → picks the cumrv form.
- **backfit A/B**: does AlternatingBackfit beat single-pass 0.12146? K=1 sanity + K* + orthogonality
  metrics → decides whether the heroes use backfit (final-pass) at all.

### Base-form selection rule
1. cumrv form: **RESOLVED → proxy `enetreg2` (0.12314)**. cumrv_real lost (0.12436 = no-cumrv level),
   OPEX no gain (0.12314), ratio worse (0.12364). So `BASE = xgb_all_buckets_tw1000_enetreg2_rf480_slim`.
2. Backfit: **RESOLVED → NO backfit, single-pass** (A/B job 9661151: K*=1 in all 407 blocks; K≥3
   degrades; cross_corr only 0.115 at K=1 → linear & tree already orthogonal). Hero A runs the standard
   single-pass amortized objective (`resid_amortized.py trial`). BASE = `xgb_all_buckets_tw1000_enetreg2_rf480_slim`.

## Hero A — XGB tuning campaign (global)
Prereqs (verified present for `enetreg2`; re-run for `enetreg2_real` if it wins):
- base cell `results/resid_prep/<BASE>/` with `Xs.npy`, `cadence.npz`, `masks.npy`, `feats.json` ✓
  (enetreg2). If `enetreg2_real` wins: its prep is built by item-2a, then run
  `$PY resid_amortized.py enet_masks <BASE>` to create `masks.npy` (the `resid_subset` arm needs it).
- seed study `.hpc/campaigns/covid_xgb_all_buckets/optuna.db` ✓ (warm-start; shallow depth-2 seeds —
  fine, TPE moves deep over the budget; their low-colsample is on-target).

Launch (controller runs CLUSTER-SIDE — login process or a controller SLURM job, so `sbatch` + the
results dir are local; it submits one job per trial and polls the JSONs):
```
$PY async_tune.py --cell <BASE> --arm resid_subset --k 24 --n-trials 200 --no-dry-run
```
- K=24 in flight, each trial = `resid_amortized.py trial <BASE> resid_subset <outname>` (TREE_CFG env).
- Monitor: `results/resid_tree/*.json` (per-trial qlike), the optuna study, `squeue`. Keep total jobs
  < 100 (K=24 leaves headroom for other work).
- **Interpret the winner** (answers bagging-vs-boosting): read its hyperparameters — deep + few-rounds
  + low-colsample = bagging-ward; shallow + many-rounds = boosting-ward (d6 interp predicts
  deep+low-colsample). Plus importances / H-stat. No separate RF needed.

## Hero B — untuned EBM regime stage gated on Hero A (B1 cascade)
Blocked on: (a) Hero A's winning XGB cfg, (b) item-2b (the `preds_chunk` backfit/regime arm).
- **Cascade**: `final_B = base + xgb_winner(X) + 1[h∈16-19]·ebm_regime(X)`, where `ebm_regime` is fit
  on the leftover residual `y − base − xgb_winner` using **h16-19 rows only**, predictions gated to h16-19.
- **EBM cfg (untuned — user: no EBM tuning success)**: `learning_rate=0.02, max_leaves=3,
  interactions=10, max_bins=256, min_samples_leaf=10, max_rounds=500, outer_bags=4` (the `do_ebmsmoke`
  point; sensible mid of `widened_space("ebm")`).
- **Score**: full-OOS QLIKE + an **h16-19-masked** QLIKE (the regime lift). Deferred Duan smearing at
  collect (like `do_chunk_collect`).
- Expectation: per the d6 interp, EBM (pairwise) leaves order-≥3 content on the table, so the regime
  EBM's job is the additive+pairwise slice of the h16-19 residual the global XGB didn't take.

## item-2b spec — `preds_chunk` regime-cascade arm (Hero B machinery)
New arm in `resid_amortized.preds_chunk` (e.g. `arm="resid_regime"`), edit serialized AFTER item-2a.
Per cadence block `i` over `[t_r, t_end)`, with `tw=train_win`, `hr = Xs[:, feats.index("hour")]`:
```
# 1. base (existing): out = ridge_oos[k0:k1]   (cadence matvec, already in cache)
# 2. global stage — XGB winner on the resid_subset survivors (existing residualized path):
Xtr = Xs[t_r-tw:t_r];  cols = masks[i]                      # resid_subset arm
r1  = y[t_r-tw:t_r] - (Xtr@coefs[i] + intercepts[i])
g   = xgb_winner.fit(Xtr[:,cols], r1)                        # cfg = Hero-A winner via env GLOBAL_CFG
out[blk] += g.predict(Xs[t_r:t_end][:,cols])
# 3. regime stage — EBM on the LEFTOVER, h16-19 train rows only, gated prediction:
m_tr  = (hr[t_r-tw:t_r] >= 16) & (hr[t_r-tw:t_r] <= 19)
r2    = r1 - g.predict(Xtr[:,cols])                          # leftover after global
e     = ebm.fit(Xtr[m_tr][:,cols], r2[m_tr])                 # cfg = REGIME_CFG (untuned EBM, §Hero B)
m_blk = (hr[t_r:t_end] >= 16) & (hr[t_r:t_end] <= 19)
pe    = e.predict(Xs[t_r:t_end][:,cols]); pe[~m_blk] = 0.0    # gate to h16-19
out[blk] += pe
```
- Two cfgs via env: `GLOBAL_CFG` (Hero-A winner, XGB) + `REGIME_CFG` (untuned EBM). `_tree_factory`
  builds both (model arg per stage — add a `model_regime` or pass both factories).
- Guard: if `m_tr.sum()` is tiny in an early block, skip the regime stage that block (predict 0) —
  but ~42k/195k OOS rows are h16-19, so per-block training samples are ample.
- Collect: reuse `do_chunk_collect` for full-OOS QLIKE; add an h16-19-masked variant (filter `k`
  rows whose `hour∈[16,19]`) for the regime-lift number. Deferred Duan smearing unchanged.
- Backfit (only if backfit A/B wins): wrap step 2 in `AlternatingBackfit.fit_window` with K* from
  `select_rounds_forward`, replacing the single `g.fit`; step 3 unchanged.
- Sanity: with the regime stage disabled, this arm must reproduce the single-pass `resid_subset` number.

## Open items / dependencies
- item-2a → final cumrv form (gates `BASE`).
- backfit A/B → whether backfit is in the stack.
- item-2b → the `preds_chunk` backfit/regime arm (edits `resid_amortized.py`; serialized after item-2a).
- Repo is local-only (no git remote) — no `/sync`.
