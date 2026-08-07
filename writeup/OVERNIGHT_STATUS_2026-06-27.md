# Overnight autonomous run — status (started 2026-06-27, user asleep)

Single source of truth for what ran overnight. The /loop driver updates this as results land.
Base is LOCKED = fixed-α single-pass `enetreg2` (0.12314); best dig so far XGB-d6 = 0.12146.

## ⏩ PICKUP AFTER CLEAR (read this first)
**Session work COMPLETE; FINAL best = 0.12050 (Hero B). Full ladder/findings in the WAKE-UP SUMMARY below.**
Cluster: CARC `usc-discovery`, `/scratch1/jc_905/harxhar-clean`, env harxhar, PY=`/home1/jc_905/.conda/envs/harxhar/bin/python`. cell = `xgb_all_buckets_tw1000_enetreg2_rf480_slim`. Chunk preds: `results/resid_ab/<cell>/<sub>/chunk_*.csv` (cols k,pred_adj,y_true,base); subs `resid_subset_heroA_d8` (d8 global 0.12129), `resid_regime_heroB` (Hero B 0.12050).

**Two agents were in-flight at the clear (a /clear likely killed them — harvest their CLUSTER outputs, or re-run):**
1. **EBM-shapes + multi-metric/day-night eval — ✅ DONE** (scripts ebm_shapes.py / ebm_eval_panel.py; outputs results/ebm_shapes.json + ebm_eval_panel.{json,txt}; findings in the "EBM SHAPES + DAY/NIGHT" section below).
2. **Rank-construction diagnosis + lag placebo.** Why proxy (cumsum cache `har_ma_1`, 0.12314) beats rankcum (cumsum fresh `diurnal_rank`, 0.12429) — NOT segmentation (calendar==hour-wrap byte-identical); suspects: rank-then-shift vs shift-then-rank, adj-vs-raw RV, cache rank window. May have added base_kinds (`enetreg2_rankcum2`, `enetreg2_lag`) to resid_amortized.py + submitted base-alone preps — check `logs/*rankcum2*`/`*lag*` on CARC + `grep -c enetreg2_lag resid_amortized.py`. **Lag placebo** `enetreg2_lag` (proxy cumrv lagged 1 session): ≈0.12314 → slow cross-session regime; →0.12436 → genuinely intraday (causal either way — cumrv is NOT look-ahead: har_ma_1 is shift(1), cumsum is a prefix not a day-total, day boundary uses only past hours).

**Cleanup:** kill the bonus Optuna campaign (won't beat 0.12050): `ssh usc-discovery "pkill -f async_tune; scancel -n heroA_xgb_enetreg2"`.

**IN-FLIGHT (linear-distill test, post-EBM-shapes):** base_kinds `enetreg2_distill` (enetreg2 + {sumvolume,numobs,voldemand}×close) and `enetreg2_distill_harsq` (+ nonlinear `har_ma_5²×close`) — base-alone preps on CARC vs 0.12314 / Hero B 0.12050. Harvest from `logs/*distill*` or `results/resid_prep/ebm_all_buckets_tw1000_enetreg2_distill*`. Tests whether the regime EBM's top features (HAR-damp convex, sumvolume↓/numobs↑/voldemand↓ at close) distill into a few linear/quadratic close-interactions, or are genuinely nonlinear-only. NOTE: scaling `har×close` by a constant is a no-op in the enet (scale-invariant) — that's why the nonlinear `har²×close` is used to capture the convex/aggressive slope.

**➜ ON RESUME, CHECK THIS FIRST: distill prep = job 9675076 — ✅ HARVESTED (COMPLETED).** Verdict: the regime does **NOT** distill linearly. `enetreg2_distill` (+{sumvolume,numobs,voldemand}×close) = **0.12310** (−0.00004 vs base 0.12314); `enetreg2_distill_harsq` (+nonlinear `har_ma_5²×close`) = **0.12309** (−0.00001 more → no real convexity). vs full regime-EBM stage −0.00264 (Hero B). ⇒ the close/AH correction is genuinely high-order/interaction-heavy, not compressible into a few linear/quadratic terms. `har²×close` showed no convexity → **target-transform experiment SKIPPED.** Cluster verified clean (0 jobs, no async_tune controller — bonus campaign wound down). **SESSION COMPLETE; every cheap lever rejected → ACCEPT the floor (0.12050), write up mechanism + data-to-buy.**

**Open experiment — target transform (only if `har²×close` shows REAL convexity):** test sqrt→log target to see if the close-damping is *multiplicative* (linear-in-log) vs genuinely convex. CAVEAT: RV has **exact zeros** (overnight/illiquid bars) → naïve `log(RV)` = −∞ breaks. Use **Yeo-Johnson** (the Box-Cox variant defined at 0/negatives) for the λ sweep — or `log1p(adj_RV)` / `log(adj_RV+ε)`. Skip entirely if `har²×close` already captures the convexity (we're at the modeling floor + multiple-testing budget; the log re-prep is a big change for a refinement).

**Strategic fork (post-harvest):** lean = ACCEPT the floor (every smarter lever rejected; 4th-decimal gains; multiple-testing risk is real) and write up the **mechanism (auction/session microstructure) + the data-to-buy case** rather than keep tuning. EBM shapes + day/night metrics feed that writeup.

---

## EBM SHAPES + DAY/NIGHT METRICS (agent 1 — DONE 2026-06-27)
Scripts `ebm_shapes.py` / `ebm_eval_panel.py`; data `results/ebm_shapes.json`, `results/ebm_eval_panel.{json,txt}`.

**Mechanism (EBM shapes; in-sample probe, 34,357 h16-19 OOS rows; target = adj leftover `y_true−pred_adj` after the d8 global; 269 survivor cols; sign: +score pushes close forecast UP, −score DOWN):**
The regime correction's signature is **HAR DAMPING AT THE CLOSE.** `har_ma_5` and its close-gated twin `har_ma_5_x_close` both show a clean **monotone-DOWN** shape (Spearman −0.99, ~equal magnitude): high short-horizon realized vol → a **negative** correction → the regime stage pulls the close/AH forecast **below** HAR's vol-persistence extrapolation. **This is the documented late-day sign-flip of HAR persistence, recovered directly from the EBM shape** — the auction/session-transition mechanism made legible. Modulators: trade-count (`numobs`) + `effspread` push vol UP; 1-bar `volume` + `voldemand` damp; `stocktwits` attention tilts up. Pairwise (all small): the close-HAR damping is **liquidity/turnover-regime dependent** (`har_ma_125_x_close × long buy-turnover`); `bipow × ret⁴` = continuous-vs-jump. (EBM in-sample R² on the leftover = 0.069 — small, as expected for a residual-of-residual.)

**Multi-metric × time-of-day (QLIKE / MSE_raw / MSE_adj / OOS-R²; full table in ebm_eval_panel.txt):**
1. **Metric agreement:** on ALL, every metric agrees base<d8<HeroB. At BUCKET level they SPLIT — raw-MSE / raw-R² **flip** in RTH+overnight (d8/HeroB look worse) while QLIKE + adj-space metrics still rank d8/HeroB better everywhere. Cause: raw-MSE is dominated by extreme-vol points and the global Duan-smear constant inflates raw error in low-vol buckets even as QLIKE (the proportional loss actually optimized) improves. ⇒ **trust QLIKE ⇄ adj-MSE ⇄ adj-R² (agree); raw-MSE is the dissenter off-close.**
2. **Most predictable:** by R² (variance explained), **close/AH h16-19 is HIGHEST** (R²_raw_vs_mean 0.748→0.769 vs RTH 0.67, overnight 0.60); by QLIKE, **overnight is lowest-loss** (0.110) and **close/AH is the HARDEST** (0.158-0.167). Informative tension: close/AH has the biggest, most variable vol (hard in proportional/QLIKE terms) yet the model captures the largest variance SHARE there — which is why the regime stage has room.
3. **Hero B's lift is ~entirely in h16-19** (ΔQLIKE −0.00441 close/AH vs −0.00003 RTH, −0.00002 overnight; HeroB pred byte-identical to d8 off-gate) — the gate works as designed.

---

## Live checklist
- [x] **rankcum array built** — `build_rankcum.py` → `results/intraday_feats/cumrv_realrankcum.npy` (range −89..+92).
- [~] **rankcum base-alone prep** — local run was killed by the harness bg-reaper; **re-submitted as CLUSTER job 9663910** (`logs/rankcum_prep_*.out`, grep PREP). The breadth/segmentation pivot vs proxy 0.12314 / real 0.12436.
- NOTE (kill scare, resolved): harness reaped the local bg shells at turn end. Campaign controller survived (nohup, detached on cluster, pid verified). rankcum moved to cluster. Loop uses foreground ssh polls (not bg shells), so it's unaffected.
- [x] **realrank** — `enetreg2_realrank` = **0.12366** (local). rank-Gauss is the right SPACE (recovered ~57% of robust-scale's loss 0.12436→0.12366) but doesn't beat the proxy 0.12314. Residual gap = hour(24) vs 48-slot resolution + construction order. ⇒ base stays the proxy.
- [~] **Hero A — deep-frontier sweep** (proven chunked machinery, NOT the unverified async_tune campaign): 6 configs submitted. Collects → ladder → winner.
  - d8@cs0.3 (dig 9662742 / col 9662743) — label heroA_d8
  - d10@cs0.3 (9662744/45) — heroA_d10
  - d12@cs0.3 (9662746/47) — heroA_d12
  - d6@cs0.1 (9662748/49) — heroA_d6cs1
  - d8@cs0.1 (9662750/51) — heroA_d8cs1
  - d10@cs0.1 (9662752/53) — heroA_d10cs1
  - anchor (already have): d6@cs0.3 = **0.12146**
- [ ] **Hero A winner picked** → run `interp_winner` on it (importances / H-stat / by-hour; bagging-vs-boosting read).
- [ ] **Hero B** — untuned-EBM regime cascade on the winner: `bash submit_heroB.sh <winnerDepth> <winnerCS>`. Reports full-OOS + h16-19-masked QLIKE (the regime lift).
- [ ] **Final** — fold all numbers into `session_notes_2026-06-27...md`; write the wake-up summary below.

## Why Hero A is a curated sweep, not the 200-trial campaign
The async_tune smoke (1 un-chunked trial) ran 33+ min without completing → a single full trial is slow and the controller loop was unverified, with no persistent controller job. For unattended overnight, the robust choice is the PROVEN chunked dig/collect arrays (which produced 0.12146), targeted at the deep + low-colsample frontier the d6 interp localized. async_tune (full Optuna campaign) remains available once its chunking/controller-job is debugged.

## Decision rules the loop follows
- Winner = min QLIKE across {0.12146 (d6), the 6 swept}. Ties → shallower/higher-colsample (parsimony).
- Interpret winner's hyperparameters: deep+low-colsample = bagging-ward (dense-weak); shallow = boosting-ward.
- Hero B uses the winner as GLOBAL_CFG + the untuned EBM (lr0.02/leaves3/inter10/bins256/minleaf10/rounds500/bags4) as REGIME_CFG.
- If any Hero A dig errors (check logs/heroA_dig_*.out), resubmit that config; if Hero B dig errors, record the issue and leave Hero A as the deliverable.

## Progress log
### Tick 06:56
- **rankcum DONE = 0.12429** — rank-then-sum over CALENDAR days ≈ no-cumrv level, vs proxy (same construction, HOUR-WRAP days) = 0.12314. ⇒ **the proxy's cumrv edge is the hour-wrap SEGMENTATION** (trading-session-aligned accumulation window), NOT breadth-vs-magnitude. Chain: real(magnitude)=0.12436, realrank(sum-then-rank, hr)=0.12366, rankcum(rank-then-sum, calendar)=0.12429, proxy(rank-then-sum, hour-wrap)=0.12314. Only the hour-wrap boundary recovers the edge → the feature is construction-entangled, not a clean physical signal.
- **Hero A SWEEP DONE — winner `d8@cs0.3` = 0.12129** (new best, −0.00017 vs d6 0.12146). Ladder: d8 **0.12129** < d10 0.12139 < d8cs1 0.12147 < d10cs1 0.12152 < d6cs1 0.12154 < d12 0.12169. Depth peaks at d8 (d12 overfits); colsample 0.3 ≥ 0.1 (low-colsample slightly worse — contra the slice-study prior).
- **Hero A CAMPAIGN**: 20/200 trials, best-so-far 0.12235 (still shallow seeds; deep trials running). Continuing — may refine the winner.
- **interp_winner FIRED** on d8@cs0.3 (job 9664198) — importances/H-stat/by-hour.
- **Hero B FIRED** on d8@cs0.3 global + untuned EBM regime (dig 9664199 / collect 9664200) — will report full-OOS + h16-19 regime lift.

### Tick 07:26
- **Campaign 42/200, best 0.12235 (d3)** — WORSE than sweep d8@cs0.3 (0.12129); deep campaign trials (d9–15) overfit (~0.1229). ⇒ **SWEEP winner d8@cs0.3 = 0.12129 remains Hero A winner**; the 200-trial Optuna campaign is not beating the targeted 6-config sweep (a finding in itself — the deep+moderate-reg sweet spot the sweep hit isn't easily improved by free hyperparams).
- **Hero B + interp FAILED then FIXED+RE-FIRED**: root cause = `sbatch --export` comma-splits a JSON cfg value (cfg arrived as `{"n_estimators":200`). The sweep worked because heroA_dig built the JSON *inside* the script. Fix = hardcode cfg inside dedicated sbatch (heroB_d8.sbatch / interp_d8.sbatch). Re-fired: Hero B re-dig **9664849** / collect **9664850**; interp **9664851**.

## WAKE-UP SUMMARY (complete — 2026-06-27 ~08:00)

**Headline: new best = 0.12050** (Hero B cascade), down from the 0.12414 plateau you started above — a **−0.00364** improvement over the session.

### Final QLIKE ladder
| model | QLIKE | Δ |
|---|---|---|
| plain enet | 0.12516 | |
| enetreg2 BASE (HAR×{open,close}+cumrv, fixed-α, locked) | 0.12314 | base |
| XGB-d6 on enetreg2 (prior best) | 0.12146 | −0.00168 |
| **XGB-d8@cs0.3 (Hero A winner, sweep)** | **0.12129** | −0.00185 |
| **Hero B: d8 global + untuned EBM regime@h16–19** | **0.12050** | **−0.00264 (BEST)** |
| (campaign 200-trial Optuna, 66+/200) | 0.12206 best | did NOT beat the sweep |

### Hero A (global)
Winner = **XGB depth-8, colsample 0.3** (n_est200, lr0.03, subsample0.7, mcw50, reg_lambda2) = **0.12129**, from the targeted deep-frontier sweep. Depth peaks at d8 (d10 0.12139, d12 0.12169 overfit); colsample 0.3 ≥ 0.1. The **200-trial Optuna campaign did NOT beat it** (best 0.12206; deep+free-hyperparam configs overfit relative to the sweep's deep+moderate-reg sweet spot) — *curated beat blind here*. interp (d8): **~90% interaction / order-≥3 / diffuse** (max H 0.088, top pairs `cumrv_x_close × {numobs, sumbipow}`), depth helps most at the session **edges [18,17,9,19,16]** — the bagging-ward / dense-weak signature.

### Hero B (h16–19 regime cascade) — WORKS
The untuned EBM regime stage on the h16–19 leftover adds **−0.00079** over the d8 global (0.12129 → **0.12050**). h16–19 in-window QLIKE = 0.15788. ⇒ the close/AH regime carries signal the global model misses, and a dedicated stage captures it.

### Diagnostics (the "why")
- **realrank = 0.12366** — rank-Gauss is the right space (recovered ~57% of robust-scale's loss) but physical cumrv still loses to the proxy.
- **rankcum = 0.12429** — CORRECTED: NOT a segmentation effect. Verified on the data (24h grid) that calendar-day == hour-wrap segmentation is **byte-identical** (6051 day-starts each, 0 disagreements). So rankcum's loss vs the proxy (0.12314) is the **per-bar rank SOURCE** — the proxy sums the cache's `har_ma_1` (per-slot rank-Gauss of the full diurnal→sqrt→winsor→shift pipeline); rankcum summed a fresh `diurnal_rank(RV)`. The cumrv edge is **construction-specific to `har_ma_1`**; cumrv_real (physical) accumulates the RIGHT bars but loses physical-magnitude vs rank-space.

### What this points toward
Every "smarter" lever was tested and **rejected** (online-λ, backfit, ratio, physical cumrv, OPEX, and now blind Optuna vs the curated sweep); the gains that stuck are small and the residual is dense/high-order/edge-localized. We're **near the modeling floor on this data** — the d8+EBM-regime stack (0.12050) is about as far as features+method push it. The remaining edge is an **auction/session-transition microstructure regime**; extracting more likely needs **direct mechanism data** (auction imbalance / GEX / OFI), not more model capacity. The campaign-vs-sweep result reinforces this: free hyperparameter search couldn't beat a 6-point sweep aimed by the interpretation — there's no rich landscape left to search.

### Open / FYI
- Campaign still running (nohup, ~66/200) as a bonus; it won't beat 0.12050. Kill with `scancel` on its `heroA_xgb_enetreg2` jobs + `pkill -f async_tune` if you want the cores back.
- Everything reproducible: sweep `submit_heroA.sh`, Hero B `heroB_d8.sbatch`+`heroB_collect.sbatch`, interp `interp_d8.sbatch`, diagnostics `build_rankcum.py`/`rankcum_prep.sbatch`. Verdicts also folded into `session_notes_2026-06-27_edge_features_recursive_en_dates.md`.
