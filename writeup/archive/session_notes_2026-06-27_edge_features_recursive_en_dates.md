# Session notes 2026-06-27 — edge features, recursive elastic net, date reconstruction

Continues the intraday-regime work (`intraday_regime_findings_2026-06-26.md`). All QLIKE are
full-OOS Duan-smeared on the slim `all_buckets` cell, `resid_subset` arm, unless noted. Cluster =
CARC (`/scratch1/jc_905/harxhar-clean`, env `harxhar`).

## TL;DR
- **FINAL BEST = 0.12050** (overnight Hero A/B; full detail in `OVERNIGHT_STATUS_2026-06-27.md`). Hero A
  global winner = XGB **d8@cs0.3 = 0.12129** (targeted deep-frontier sweep; depth peaks d8, d12 overfits;
  the 200-trial Optuna campaign did NOT beat it — curated > blind). Hero B = +untuned EBM regime stage on
  h16–19 → **0.12050** (−0.00079, the close/AH regime cascade WORKS). Diagnostics: realrank 0.12366
  (rank-Gauss is the right space but proxy still wins), rankcum 0.12429 (CORRECTED: NOT
  segmentation — calendar==hour-wrap is byte-identical on the 24h grid, 0 disagreements; the proxy's edge
  is specific to the cache's `har_ma_1` rank construction, which a fresh `diurnal_rank(RV)` cumsum doesn't
  reproduce; cumrv_real accumulates the RIGHT bars, it just loses physical-magnitude vs rank-space). Read:
  near the modeling floor; remaining edge = auction/session
  microstructure → needs direct mechanism data, not more capacity.
- **The regime does NOT distill linearly (final lever, job 9675076).** Adding the EBM's top features as
  linear close-interactions to the enet base — `enetreg2_distill` = +{sumvolume,numobs,voldemand}×close —
  moves base-alone QLIKE only 0.12314 → **0.12310** (−0.00004); adding the nonlinear `har_ma_5²×close`
  (`enetreg2_distill_harsq`) → **0.12309** (−0.00001 more, i.e. ~no real convexity). vs the full regime-EBM
  stage's −0.00264 (Hero B). ⇒ the close/AH correction is genuinely high-order/interaction-heavy and cannot
  be compressed into a few linear or simple-quadratic features. Since `har²×close` showed no convexity, the
  log/Yeo-Johnson **target-transform experiment is skipped** (a global target reshape can't capture a
  regime-localized interaction). **Every cheap lever is now tested and rejected → ACCEPT the floor;** the
  deliverable is the mechanism (auction/session microstructure) + the data-to-buy case.
- **cumrv edge = the SQRT, now ISOLATED (job 9676656).** Earlier the "sqrt variance-stabilization" claim was
  inferred from the confounded proxy-vs-real gap. The control `enetreg2_realsqrt` = `cumsum(sqrt(per-bar adj
  RV))×close` (else byte-identical to `enetreg2_real`) = **0.12317**, recovering **97.5%** of the real→proxy
  gap (real 0.12436 → realsqrt 0.12317 ≈ proxy 0.12314). sqrt (98%) > rank (`realrank` 0.12366, 57%): the
  cap-mixture keeps the magnitude rank discards while refusing to let the tail saturate the cumsum. Verified,
  not inferred. See `threshold-mixture-vol-transforms` memory.
- **First positive linear lever: rank-space `har_ma_5×close` (job 9676656).** `enetreg2_har5rank` =
  enetreg2 + causal per-slot rank-Gauss(`har_ma_5`)×close = **0.12293 = −0.00021 vs base** — rank space is a
  better monotone basis for the close HAR-damping shape than the base's sqrt-space column. Small (~⅛ of the
  d6 tree edge) and only the trend (Spearman −0.99 = monotone, not linear-in-rank → residual curvature still
  wants the tree). Open: does it stack through the d8 tree / Hero B (the EBM found this shape in the
  post-tree leftover, so plausibly complementary).
- **Overnight cumrv at the open = null (job 9676339).** `enetreg2_open` (cross-boundary overnight cumrv
  {20…9}×open) = **0.12315** (+0.00001). Session edges are asymmetric: the intraday path linearizes into the
  close, the overnight does NOT into the open (¼ the support + near-zero per-row signal; the open regime is a
  discrete auction/gap effect, tree-only).
- **Penalty is NOT basis-invariant; fix = unpenalize the HAR block.** Reparameterizing the 6 nested HAR MAs
  into differenced bands (span-identical in OLS) cost QLIKE under the fixed-α enet: `enetreg2_canon` = 0.12423
  (+0.00109), and robust-scaling the bands made it WORSE (`canon_scaled` 0.12584 — scaling destroys the
  natural signal-strength prior; differencing concentrates noise into the high bands). L2 is invariant only
  under orthonormal maps, L1 under none. **Fix:** `enetreg2_harunpen` (6 HAR mains UNPENALIZED via exact
  FWL/OLS profile-out, L1+L2 on rest) = **0.12316 ≈ 0.12314** — the L1 was doing nothing to the dense HAR
  block; unpenalizing makes the HAR basis a free legibility choice. Undistorted HAR coefs: mains decay and go
  negative at the longest horizon (short persist / long revert); `har_ma_5×close` = −0.05 (close damps short
  vol). This is the clean residual→tree base.
- **FWL block attribution (`fwl_attribution.py`; FULL=0.12314 ✓).** Type-III unique contributions (net of
  all): HAR **−0.02208** (R²=0.594 alone of 0.617 full → HAR = 96% of explainable variance), EXOG(359)
  −0.00449, REGIME −0.00225, CUMRV(1) **−0.00122 = exactly the famous cumrv edge.** ⇒ one engineered feature
  ≈ **27× the per-feature value** of raw exog (the data-to-buy case). cumrv↔exog ~28% overlap.
- **har5rank decomposed = ⅔ static / ⅓ rolling.** E-test: R²=0.95 of `har5rank` explained by a static per-slot
  spline of `har_ma_5`. Isolated dense spline (`har5spl`) recovers −0.00015 of the −0.00021 (~70%); stacking
  (`har5rank_spl` 0.12291) ≈ har5rank alone (~85% redundant). The static ⅔ = the documented HAR-damping
  curvature (redundant with the tree); the rolling ⅓ = a STANDARDIZED VOL INNOVATION (today vs recent same-slot
  history) — genuinely new-to-tree.
- **NEW BEST 0.12046 — the rolling/relative signal is the only post-Hero-A lever.** `enetreg2_rel` (enetreg2 +
  rolling per-slot RANK-GAUSS innovations `zr_har_{1,5,25,125}`; the raw z-score hurt +0.00049 via heavy tails,
  the bounded rank-Gauss form is base-neutral 0.12323) → +d8 = **0.12121** (−0.00008 vs Hero A 0.12129) → +EBM
  regime = **0.12046** (−0.00004 vs Hero B 0.12050). Gain matches har5rank's rolling-⅓ estimate (−0.00008). NOTE:
  a fresh dig cell needs `enet_masks` before the resid_subset/resid_regime dig.
- **Per-bucket space (implied_vol rank vs magnitude) MATTERS — magnitude wins −0.00013.** `enetreg2_ivmag` (swap
  adj_{vix,vvix,vix3m}_ma_* rank→robust-scale magnitude) base-alone = **0.12301 (−0.00013)**, 2nd-best linear
  lever after har5rank. **BUG caught:** the first attempt used `_floored_robust_scale` (self-computed ref) →
  degenerate-IQR million-scale blowup → fake null 0.12314; FIX = STD-pipe SAVED ref_iqr + fixed_cols. Confirms
  VIX²~variance wants magnitude. But it does NOT carry through the tree: ivmag +d8 = **0.12138 vs Hero A 0.12129
  (+0.00009)** — the tree already captures implied-vol's level via thresholds (monotone-invariant to space), so
  the linear magnitude gain is redundant (slight worsening = enet-survivor-mask perturbation). Same pattern as
  har5rank-static-⅔ / canon coefs (linear gains subsumed by the tree). ⇒ per-bucket space is a LINEAR/legibility
  finding, NOT a deployed-tree lever; rel Hero B 0.12046 stays best.
- **NEW BEST 0.12035 — the PATH-SHAPE facet (cumrv-sibling program).** Engineering more cumrv-style intraday-path
  features (FWL said cumrv = 27× the per-feature value of raw exog): cumret (directional signed path) NULL,
  gapopen (overnight directional gap) NULL — the close effect is a VOL-regime/TIMING phenomenon, not directional.
  But the within-day vol-path SHAPE — `cumrvshape` = afternoon vol share (cumrv_now−cumrv_midday)/cumrv_now ×close,
  i.e. WHEN the day's vol happened, front/back-loaded — is **orthogonal to the saturated cross-day regime factor**:
  d8 = 0.12109 (−0.00020 vs Hero A, best d8). The two orthogonal facets STACK: `shaperel` (afternoon-share +
  rolling-relative) d8 = 0.12095 → **+ EBM regime = 0.12035 = NEW BEST** (−0.00015 vs Hero B; h16-19 in-window
  0.15737). (Richer shape block d8 0.12099 but Hero B 0.12055 — didn't carry as well; the single share + regime is
  the sweet spot.) Path-shape = the hand-computable time-channel low-order path-signature terms. #3 cumrv×liquidity
  tree-covered (interp), #5 VWAP no data.
- **kNN/relevance regression: works PROPERLY but DOMINATED by the EBM regime.** Naive kNN (K=25 local MEAN, add raw)
  HURT (+0.00058, over-variance). PROPER (Cartea et al. ssrn-4652980: large-K + local LINEAR ridge, recency-history
  similarity) flips it on the close leftover — local-ALL 0.12111 (−0.00018), local-HAR −0.00009, V-only −0.00007,
  intercept ~null, SHUFFLE placebo +0.00026 (gains real). CORRECTION: the "only locally-fit regime-varying V"
  hypothesis was WRONG — full-local > V-only; stable features don't hurt with large-K+ridge. BUT the EBM regime
  already reaches −0.00079 ≫ the analog's −0.00018 → kNN is a weaker cousin, NOT deployable. coef-stability:
  har_ma_1/5, sumbipow, stocktwits regime-varying (coefs sign-flip with vol); cumrv/long-HAR stable. Subsuming
  model class = sequence-attention (PatchTST) / signature-linear, but data-hungry at ~195k rows → hand-features are
  the legible proxy. Use kernel-WEIGHTING not hard-k (embargo+causality starve early OOS at large k).
- **Broke the 0.12414 plateau.** Folding the open/close regime + intraday vol-path into the LINEAR
  base, then digging nonlinearly, lands at **0.12146** — −0.00268 under the old plateau.
- **Biggest single win is a LINEAR feature**: `cumrv_today × close` (intraday vol accumulated, gated
  to h16–19) → base **0.12436 → 0.12314** (−0.00122, sign-perfect). It linearizes most of what the
  tree was finding at the edge, and it STACKS with the dig (the tree did not already have it).
- **Base form LOCKED = fixed-α single-pass `enetreg2`.** Both user-requested enhancements were built,
  run, and REJECTED: data-driven λ (online-λ ties fixed-α, +0.00010) and alternating backfit
  (K\*=1 in all 407 blocks, K≥3 degrades). Simpler than planned → Hero A tunes on the cheap amortized
  single-pass objective.
- **Date reconstruction SOLVED** (§ Dates): cache = `load_raw_data("data")[3125:]` exactly
  (offset = HAR burn-in; hour/DOW match 1.0000; anchor 2020-02-25 ✓). OPEX/calendar are now real
  features, saved to `results/date_reconstruct/`.
- **Recursive elastic net (`reclasso_har.py`) built + validated to 1e-13** (Stages 1–3) and now fully
  exercised on the cluster — both frontier questions (online-λ, backfit) answered above.

## QLIKE ladder — every variant tested this session
Δ are vs the locked base `enetreg2` (0.12314), lower = better.

| variant | QLIKE | verdict |
|---|---|---|
| plain enet | 0.12516 | baseline |
| enetreg (HAR×{open,close}, capped h16–19) | 0.12436 | step, −0.00080 |
| **enetreg2 (+ cumrv×close, LINEAR)** | **0.12314** | **BASE — LOCKED**, −0.00122 |
| EBM on enetreg2 | 0.12276 | accepted dig, −0.00038 |
| **XGB-d6 on enetreg2** | **0.12146** | **best dig**, −0.00168 (−0.00268 vs old plateau) |
| old plateau (EBM on plain enet, campaign best) | 0.12414 | prior best — BEATEN |
| EBM on enetreg / XGB-d6 on enetreg | 0.12332 / 0.12215 | superseded by enetreg2 base |
| online-λ `enetreg2_olam` (data-driven μ) | 0.12324 | REJECTED — ties fixed-α, +0.00010 |
| ratio `enetreg2_ratio` ((cumrv/har)×close) | 0.12364 | REJECTED, +0.00050 |
| `cumrv_real` `enetreg2_real` (physical, robust-scaled) | 0.12436 | REJECTED — = no-cumrv level, +0.00122 |
| OPEX `enetreg2_opex` (+7 calfeats) | 0.12314 | REJECTED — no change |
| backfit K≥3 (vs K=1 = 0.12145) | 0.12156 / 0.12163 | REJECTED — K\*=1 optimal |
| ablation `enetcum` (cumrv only, no interactions) | 0.12539 | complementarity check |
| ablation `enetreg2_noclose` (open + cumrv, drop close half) | 0.12540 | complementarity check, +0.00226 |

## Decision log
- **Base LOCKED = fixed-α single-pass `enetreg2`** (HAR×{open,close} capped h16–19, + cumrv×close;
  α=0.001 / l1=0.2). Base cell: `xgb_all_buckets_tw1000_enetreg2_rf480_slim`.
- **Linear cumrv STACKS with the nonlinear dig** (jobs 9659045/46 EBM, 9659047/48 XGB-d6, COMPLETED
  ~03:30): folding `cumrv×close` into the base improved BOTH digs over their enetreg (no-cumrv)
  counterparts — EBM 0.12332→0.12276 (−0.00056), XGB-d6 0.12215→0.12146 (−0.00069). The tree did not
  already fully capture the linear term.
- **cumrv ⟂ HAR×close are COMPLEMENTARY, not redundant** (job 9659892, base-alone): `enetcum`
  (cumrv only) = 0.12539 and `enetreg2_noclose` (open + cumrv, drop close half) = 0.12540 — BOTH
  *worse* than plain enet (0.12516). cumrv delivers its −0.00122 ONLY in the presence of `HAR×close`:
  the close interaction is the close-regime HAR base level; `cumrv×close` is the mean-reversion
  correction *to* it; unanchored, cumrv is just noise/collinearity. ⇒ keep the full open+close
  interactions + cumrv; dropping the close half costs +0.00226. Refutes the earlier "cumrv×close
  shares the h16–19 window so the close interaction is redundant" hypothesis.
- **Data-driven λ (online-λ) REJECTED** (jobs 9660569/70, base-alone): `enetreg2_olam` (per-cadence-block
  μ via reclasso `select_l1_forward` on a leakage-clean forward split, embargo=3125) = 0.12324 vs
  fixed-α 0.12314 — +0.0001, ties within refit/forward-block noise. Fixed α=0.001/l1=0.2 is already
  near-optimal for this cell; making the elastic param data-driven adds selection variance, not signal.
  Sanity: online single-μ collapses to fixed-α to 1.8e-14.
- **Ratio cumrv REJECTED** (same job batch): `enetreg2_ratio` = `(cumrv/(har_ma_25+ε))×close` single
  column = 0.12364 (+0.0005). The ratio collapses info the additive `HAR×close` + `cumrv×close` pair
  keeps; consistent with the d6 probe showing both terms rank independently (two signals — regime level
  + mean-reversion surprise — not one).
- **Physical `cumrv_real` LOSES to the rank-space proxy** (job 9660826): `enetreg2_real` = 0.12436 =
  *exactly* the no-cumrv `enetreg` level ⇒ the enet found ~zero usable linear signal in robust-scaled
  cumrv_real and effectively zeroed it (+0.00122 vs proxy). Suspect the robust-scaling / diurnal-adjust
  flattened it, or the proxy's value is tied to living in the same rank-Gauss space as `har_ma_1`.
  One-shot follow-up open: rank-Gauss cumrv_real instead of robust-scaling.
- **OPEX linear add REJECTED**: `enetreg2_opex` (+7 OPEX calfeats) = 0.12314 = no change ⇒ OPEX adds
  nothing to the linear base. (Nonlinear OPEX×edge still untried — see next steps.)
- **Backfit REJECTED** (job 9661151, full A/B on enetreg2): `AlternatingBackfit` selects K\*=1 in all
  407 blocks (1-SE OOS). K=1=0.12145 (≈single-pass), K=2 flat, K≥3 DEGRADES (0.12156 / 0.12163 = the
  overfit fixed point). WHY: `cross_corr(f_L,f_N)`=0.115 at K=1 and RISES with rounds → the linear base
  and the d6 tree are ALREADY orthogonal (not fighting), so backfit has nothing to gain — confirms the
  d6 high-order/diffuse finding. Sanity exact: cached single-pass=0.12146, base-only reclasso enet=0.12314.
- ⇒ **Heroes use the SINGLE-PASS fixed-α enetreg2 base, no online-λ, no backfit.** Single-pass is
  amortizable → Hero A tunes on the cheap amortized objective.

## Feature findings (linear distills into the enetreg base 0.12436)
- **`cumrv_today × close (h16–19)` → 0.12314 (−0.00122), coef −0.007, sign-consist 1.00.** Intraday
  vol accumulated today (cumsum of `har_ma_1` within day, days segmented by `hour` drops), gated to
  close+after-hours. Negative coef = intraday mean-reversion (high-vol day → close/AH residual vol
  *below* HAR base). `×h17–19`=0.12398, `×h18`=0.12421 (broader close window best). **The standout.**
- **Leverage `sumret₅ × bipow₁₂₅` → 0.12426 (−0.00010), coef −0.040, consist 1.00.** Real but tiny.
  Window sweep: short-return × *long*-vol wins (NOT matched windows) ⇒ slow-vol-conditional return /
  state-dependent leverage. Canonical `sumret3` (realized skewness) adds **nothing**.
- **Corsi–Renò LHAR underperforms the data:** additive `r⁻` cascade (0.12429) < product; long-horizon
  `r⁻` terms sign-UNSTABLE (no persistence at this resolution); downside-gating *hurts* (0.12437) ⇒
  effect is symmetric-in-sign, not one-sided. **HARQ (`sumret4×har`) HURTS** (0.12473).
- **Dead ends (linear):** `DOW_4×close` = 0.12436 (nothing); `OFI×edge`, `voldemand×close`,
  market-vol×close, sentiment×close all ≈0. The edge linear content is exhausted by open/close + cumrv.

## Interpretation — why the nonlinear power (d6 ≈ ~90% interaction / order-≥3 / diffuse / edge-localized)
- **Depth ladder = irreducibly high-order** (XGB on enetreg, d1/d2/d3/d4/d6/d8 =
  0.12392 / 0.12348 / 0.12308 / 0.12272 / 0.12215 / 0.12194): monotone improvement to d8, no low-order
  cutoff. Additivity gap (EBM-pairwise vs XGB-d8) = **0.00138** = the order-≥3 content.
- **Error-localization**: d6 beats d2 **concentrated at the session edges** — open (h9, +0.0029) and
  close+AH (h17–19, peak **h18 +0.0095**). Same regime as the open/close linear feature; two
  independent localizations agreeing.
- **H-statistic (d6)**: interactions distributed (max H 0.20), dominated by market-wide vol moments
  (`*_ewstock/_vwstock`), `DOW_4`, sentiment, quarticity.
- **Regime-local EBM on h17–19**: EBM explains +3.4%, d6 +5.2% (high-order gap 1.77%). Top terms:
  day-of-week + liquidity/activity (`sumvolume`, `numobs`, `effspread` × hour) + order flow.
- **nocov A/B**: dropping coverage-artifact indicators 0.12332 → 0.12341 ⇒ ~91% real, ~9% artifact.
- **Verify-before-interpret wins/corrections**: `voldemand_active` = data-COVERAGE artifact (temporal
  on-frac `[0,0,0,.28,.28,.28,.29,.29,.52,.87]`), NOT gamma. `sumret×bipow` IS leverage-direction
  (fitted coef −, sign-perfect) though the crude grid-corr had misled.

## Dates — reconstruction SOLVED
- `data/time_categories.parquet` + `core_stats.parquet` carry `endbartime` (full datetime).
- The cache is **exactly** `load_raw_data("data", allow_missing=True)[3125:]` (offset 3125 = HAR
  `max_lag` burn-in; hour-match = DOW-match = **1.0000**; anchor cache[189713] = 2020-02-25 ✓).
- Data range 2005-03-31 .. 2024-04-30, 5975 trading days.
- **Saved `results/date_reconstruct/`**: `dates.npy` (242934 datetime64[D]), `calfeats.npy` (7 cols)
  + `names.json`: `is_opex, is_quad_witch, days_to_opex, days_since_opex, is_eom, is_eoq, dom_norm`.
  9389 OPEX bars (3.86%), 229 expiries. Script: `date_reconstruct2.py` (cluster-side, job 9659049 ✓).

## Recursive elastic net (`reclasso_har.py`) — built, validated, run
- Exact RecLasso homotopy re-authored from the (flattened/unusable) chat transcript. EN-as-Lasso-on-
  Gram (`G = XᵀX + λ₂I`, robust to collinearity). API: `enet_coef`, `lasso_path_coefs`, `GramState`
  (incremental sliding-window), `forward_window_split` + `select_l1_forward` (leakage-clean online-λ),
  `GramState.coef_intercept[_at_mu]`.
- **Validated**: batch + sliding + intercept vs sklearn ElasticNet to **1e-13**; reproduces production
  `_cadence_enet` base **0.12516 exactly** (`reclasso_validate.py`, drift 2e-13 over full slide).
- **Stage 3 backfit** (`backfit.py`, `AlternatingBackfit`): 13 local checks pass; orthogonalizes
  linear⇄tree, OOS-selected K (1-SE rule), degeneracy guard holds. A/B harness: `backfit_ab_DRAFT.py`.
- **Both frontier questions now CLOSED** (see Decision log): online-λ base ≈ fixed α (refuted, job
  9660569/70); alternating backfit A/B = no gain (refuted, job 9661151). The machinery stays available;
  neither enhancement enters the hero stack.

## Hero A / Hero B plan
Full executable checklist: **`writeup/HERO_LAUNCH_RUNBOOK_2026-06-27.md`**. Settled inputs from the
Decision log feed it directly.
- **Hero A — XGB tuning campaign (global).** Base = `xgb_all_buckets_tw1000_enetreg2_rf480_slim`,
  single-pass amortized objective, K=24 in flight, ~200 trials, warm-started study. Search space
  (`widened_space("xgb")`: `max_depth(2,16)`, `colsample(0.1,1.0)`) already covers the d6-interp's
  deep + low-colsample region. Interpreting the winner's hyperparameters answers bagging-vs-boosting
  (d6 interp predicts deep + low-colsample). Launch: cluster-side controller `async_tune.py`.
- **Hero B — untuned EBM regime stage, gated on Hero A** (B1 cascade): `final_B = base + xgb_winner(X)
  + 1[h∈16–19]·ebm_regime(X)`, the regime EBM fit on the leftover residual using h16–19 rows only,
  predictions gated to h16–19. Untuned EBM cfg per the runbook (no EBM tuning success historically).
  Machinery = the `preds_chunk` `resid_regime` arm spec'd in the runbook (item-2b).

## Code changes (local working tree — all on disk)
- `resid_amortized.py`: `base_kind` **enetreg** (`_regime_interactions`, HAR×{open,close}, capped
  h16–19), **enetreg2** (`_cumrv_close`); `cov_mask` + arm `resid_subset_nocov`; `chunk_task/collect`
  `label` param. (committed 63a2a80 = the 3 open/close files; LATER edits uncommitted but on disk.)
- `src/features/extractors/calendar.py`: `is_open`/`is_close` gates (upstream, permanent).
- `src/backtest/executor.py`: HAR×{open,close} products upstream.
- New: `reclasso_har.py`, `backfit.py`, `backfit_ab_DRAFT.py`, `date_reconstruct2.py`.
- Repo is **local-only (no git remote)** — commits stay local.

## Cluster state / job ledger (CARC, jobs COMPLETED 2026-06-27)
- Cells prepped: `{ebm,xgb}_all_buckets_tw1000_{enetreg,enetreg2}_rf480_slim`; base cell for heroes =
  `xgb_all_buckets_tw1000_enetreg2_rf480_slim`.
- **Audit trail (finding → job):**
  - enetreg2 base 0.12314 ← prep `9659044` ✓
  - EBM on enetreg2 0.12276 ← dig `9659045` → collect `9659046` ✓
  - XGB-d6 on enetreg2 0.12146 ← dig `9659047` → collect `9659048` ✓
  - date reconstruction ← `9659049` ✓
  - complementarity ablation (`enetcum`/`enetreg2_noclose`) ← `9659892` ✓
  - online-λ + ratio (`enetreg2_olam`/`enetreg2_ratio`) ← `9660569` / `9660570` ✓
  - `cumrv_real` + OPEX (`enetreg2_real`/`enetreg2_opex`) ← `9660826` ✓
  - backfit A/B ← `9661151` ✓

## Next steps (priority)
1. **Launch Hero A** (XGB tuning campaign) per the runbook; interpret the winner's hyperparameters
   (bagging-vs-boosting), importances, H-stat.
2. **Hero B** (EBM regime cascade) gated on Hero A's winning cfg + the `preds_chunk` `resid_regime` arm.
3. **Real intraday features** now that dates work — true within-session cumulative RV / time-since-open
   / overnight gap from `endbartime`, replacing the rank-space `cumrv` proxy (likely stronger). One-shot
   variant: rank-Gauss `cumrv_real` instead of robust-scaling (the rank-space proxy's edge may be the
   shared rank-Gauss space with `har_ma_1`).
4. **OPEX/flow-calendar × edge** NONLINEAR distills (`calfeats` saved and ready; linear OPEX already
   shown to add nothing, so the value, if any, is in interactions).
5. Deep frontier — d8 still improving on enetreg; tune/deepen the tree (subsumed by Hero A's
   `max_depth(2,16)` sweep).
