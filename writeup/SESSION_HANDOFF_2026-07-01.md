# Session handoff — penalty-floor + every-bar homotopy + significance suite + eval_sim (2026-07-01)

Cold-resume. Branch `edge-features-legibility`. Repo LOCAL-ONLY (no remote). Supersedes
`SESSION_HANDOFF_2026-06-30.md`. Everything below is committed locally.

## READ FIRST — live cluster jobs to harvest (they finish AFTER this clear)
Two Hoffman2 SGE arrays are running; cells persist on Hoffman2 regardless of local harvesters.
- **Hoffman2 `13859217` → `results/winablate_full/`** — THE DEFINITIVE every-bar run (full metric battery
  + per-bar losses for DM/MCS). Small-bucket cells done fast; the **6 all_buckets cells are the ~4-5 h
  bottleneck**. HARVEST when done:
  `ssh hoffman2 "cd /u/scratch/j/jamesdc1/harxhar-clean && tar cf - results/winablate_full/cell_*.json results/winablate_full/cell_*_lossbar.npz" | tar xf - -C .`
  then `python assemble_winablate.py` (→ window×penalty×bucket battery table) and
  `python dm_ablation.py` (→ DM + MCS: which windows/buckets are in the 90% confidence set).
  Fold both into a new §7 of `writeup/penalty_estimator_floor_2026-07-01.md` and commit.
- **Hoffman2 `13858612` → `results/winablate_r1/`** — QLIKE-only, SUPERSEDED by 13859217 (battery push
  landed after tasks dispatched). Ignore/discard; kept only as a QLIKE preview.
- Local harvesters `bmohd4vda`/`bj3cn61z9` (bash bg loops) may die on clear — doesn't matter, re-harvest
  via the ssh tar above. Check status: `ssh hoffman2 "bash -lc 'qstat -u jamesdc1'"`.

## ASK THE USER (agent is blocked from qdel/scancel)
- Hoffman2: `qdel 13858612` (superseded QLIKE-only run — frees slots for 13859217).
- CARC (when USC VPN back — it was DOWN this session): `scancel 9804496 9804863` (superseded refit=30000
  / refit=480 arrays). CARC login `usc-discovery`/`discovery2.usc.edu` unreachable = VPN off.

## What this session established
1. **The linear penalty/estimator is a NON-LEVER — proven 5 ways** (`penalty_estimator_floor_2026-07-01.md`,
   commits 3116fe6 / 891ccf9). ridge/lasso/enet on the exog with **HAR held OLS** (never cleanly tested
   before): pure **Ridge is the structural loser** (can't zero the 235 dead availability indicators →
   blows up); enet-incumbent (a=1e-3,l1=0.2) near-optimal; slice-tuning doesn't transfer; **Optuna
   full-OOS ORACLE ceiling ≤ −0.00015** (diagnostic, test-set-overfit), ridge +0.00038; online-λ ties
   (rejected); **coefficient-MA DEAD (monotone hurt → coef drift is real signal = gamma-dilution)**.
   Signal is **sparse-in-basis but NOT low-rank** (L1 yes, PCA no). §6 per-bucket table (fixed eval
   [48000:], ~0.126): moments −0.00191 leads; ridge wins ONLY on implied_vol (dense, no junk).
2. **Every-bar (refit=1) is real.** Implemented the missing **Garrigues online-observation homotopy**
   (`reclasso_har.enet_online`) — warm per-bar active-set update, finite-terminating, **no fallback**;
   verified **bit-exact vs batch incl. pure lasso** (Δθ~1e-13). `winablate_r1.py` = every-bar cell
   (enet_online for lasso/enet, Sherman-Morrison rank-1 ridge, periodic refresh), verified vs per-bar
   batch (enet 4.6e-14, ridge 5.5e-11). Commit 36520d3. Deployed as 13859217 (battery version).
3. **Significance + metric suite** (commits 50cfa99 / 23a69f4): `src/evaluation/diebold_mariano.py` (DM,
   NW-HAC + HLN), `model_confidence_set.py` (MCS, block bootstrap — the multi-model companion),
   `metrics.forecast_metrics` (QLIKE + MSE/MAE/RMSE + HMSE + **OOS-R²** + Mincer-Zarnowitz). All verified.
   `dm_ablation.py` runs DM+MCS over the ablation's per-bar losses.
4. **`eval_sim/` package** (commit ef1e7d4) — Monte Carlo prop-eval first-passage sim (trailing drawdown),
   library-first, Slurm/SGE-ready. PnL from a **simple strategy on the cumrv×close edge**
   (`build_cumrv_pnl.py` → `eval_sim/data/cumrv_close_pnl.npy`; corr(cumrv,r)=−0.30 realizes the edge;
   frictionless/optimistic Sharpe — a shape, not a track record). Gambler's-ruin verified (4/4),
   worker→aggregate end-to-end on a 36-coord grid.

## Window-ablation QLIKE PREVIEW (from 13858612, 48/54 cells; fixed eval [168000:], covid-incl ~0.147)
- **Larger train windows do NOT help** (your thesis refuted on the fixed eval): flat 1000→2000d, WORSE
  toward 3500d; optimum ~1500-2000d; whole-range spread only ~0.0002-0.0003 → **likely not significant**
  (MCS should keep most windows — the battery run's DM/MCS will confirm).
- Bucket ranking (this covid-inclusive eval): **implied_vol≈liquidity (~0.1466) < moments (0.1468)** <
  sentiment < market_vw/ew < vol_demand (0.1482) < HAR-only (0.1486). all_buckets row was still running.
  (NB ranking is eval-period-dependent: moments led on the [48000:] eval; VIX/liquidity lead through covid.)

## KEY FILES / QUESTIONS ANSWERED
- cumrv vs HAR: HAR = order-INVARIANT multi-scale level means (NOT sequence modeling); cumrv = within-day
  cumsum (also order-invariant, novel via window/clock). Order-dependent functionals ≈ near-null (DIN/BST
  −0.00037). VWAP: NOT in the exog (`vwstock` = value-weighted cross-section, not volume-weighted price).
- Deployed base refit = **480 bars (10 days)** cadence (`harunpen_prep.sbatch`), NOT every-bar; enet has
  no rank-1 so cadence is what's been done. Every-bar is only feasible via the new online homotopy.

## NEXT
1. Harvest 13859217 → assemble battery table + DM/MCS → writeup §7 + commit (the main deliverable pending).
2. qdel/scancel the superseded jobs (user).
3. Optional: eval_sim with a real 0DTE fills log (swap BlockBootstrap on it); calibrate the cumrv PnL.
4. Still queued from 6/30: notebook alignment to the 5 contributions; LSTM/GRU scaffold (design settled).

## MEMORY
Entries: `penalty-estimator-floor` (updated), plus existing `oos-robust-feature-harness`,
`ctr-lessons-cross-architecture-test`, etc. New this session: see `evaluation-suite-and-every-bar`.
