# Intraday Volatility Regimes & Auction Microstructure — Findings (2026-06-26)

Scope: what the residualized-EBM campaign on the slim `all_buckets` cell actually
learned, and the investigation it triggered. All numbers are full-OOS QLIKE (Duan-
smeared) on the cached cell `ebm_all_buckets_tw1000_enet_rf480_slim` unless noted.
Reproducibility scripts listed at the bottom (all cluster-side, `/scratch1/jc_905/harxhar-clean`).

## TL;DR
- Campaign stopped deliberately at best **QLIKE 0.12414** (vs 0.12422 baseline, 0.12516 enet base) — plateau-limited; gains are sub-0.0001 on a proven-flat surface.
- The residual the tuned EBM extracts is dominated by an **intraday REGIME**: HAR volatility-persistence **sign-flips in the late-day/auction regime** (not a leftover U-shape).
- Mechanism = **auction / session-transition microstructure**: the reversal localizes sharply at the **open (09:30)** and **close + after-hours (16:00–19:00)**, peak −0.21; it is **clock-anchored**, not vol-state-anchored.
- It **distills**: an explicit `HAR × late-day` linear feature recovers **~58%** of the black-box tree's edge.

---

## 1. Campaign outcome & the plateau
- Best **0.12414** (carc & hoffman2 trial 8, two distinct configs at the same floor). Baseline `resid_subset` 0.12422; enet base 0.12516; ridge/prod higher.
- 19 completed trials across clusters; stopped because the surface is flat (spectrum analysis below) and the campaign driver was self-poisoning its hpc-agent journal (ops issue, separate doc).

## 2. Dimensionality is NOT the bottleneck — the spectrum is
- p=529 but **γ = p/N = 0.0022**; covariance **participation ratio ≈ 6.8**; eigenspectrum decays **~ i^−3.13**; numerically real rank **317** (212 degenerate directions).
- PC1 (variance 48) carries **~0% of the target** (unsupervised PCA is misaligned). Verdict: steep spectrum, no resolvable tail → plateau. More data/features won't move it.

## 3. A third of the matrix is dead weight
- Block structure: **6 HAR + 270 availability indicators (`_avail/_active`, 51%) + 253 value exog**.
- **164 of 529 (31%) are CONSTANT (==1) always-on indicators** — flags for never-missing series (sumret, sumvolume, the ew/vw-stock aggregates).
- Proven **QLIKE-neutral to drop**: the enet cadence coefficient is **exactly 0 on all 164 across all 407 refits**; trees never split on a constant. Filter added to `do_prep` (commit `a30f4c5`).
- 106 indicators are *live* (sentiment-type, on-frac 0.17–0.94).

## 4. The event (availability) channel — blunt A/B vs clean extraction
- **Tree-inclusion hurts.** Adding the 106 live indicators to the EBM's feature set (arm `resid_subset_ind`) monotonically worsens OOS: `resid_subset` 0.12422 → `resid_subset_ind` 0.12443 → `residualized` (all 529) 0.12469.
- **But explicit L1-selected gates help, marginally.** Window-matched `value · availability` products (`adj_{s}_ma_{w} × {s}_avail_ma_{w}`) added to the enet: **0.12516 → 0.12510**, L1 keeps **~24 of 106** gates/refit consistently.
- Lesson: the blunt A/B drowned ~24 signal gates in ~80 noise ones with no per-gate regularization; the sample-efficient extraction (explicit product → L1) finds them. The `enet-survivor` selection (`resid_subset`) correctly discards the channel for the *tree* (only ~5% of live indicators survive it), because their signal is interaction-only.

## 5. What the tuned EBM learned (interpretation; carc t8 config, 4 bags)
- Top main effects: HAR (`har_ma_1/5/25` — top for *both* enet and EBM) + **`hour`** (enet \|coef\| rank **105 → EBM rank 7**, i.e. *purely nonlinear*) + `voldemand…_active`.
- **3 of the 5 pairwise interactions involve `hour`** (`sumabsret × hour` strongest at 0.00723; `voldemand × hour` ×2).
- All top shape functions are **stable across bags (SNR 9–19)** → real structure, not overfit. (Plateau = small magnitude, not noise.)

## 6. It is intraday REGIME, not the diurnal U-shape
- The diurnal rank adjustment already extracts **77% of the diurnal variance** of y; **23% leaks** into the residual (48% of amplitude) **but only ~4% of total residual noise**. Not worth driving to 0.
- The real signal: **HAR persistence sign-FLIPS late-day. 5 of 6 HAR windows flip** at the late-day bucket (`har_ma_1,5,25,125,625`; only the slow monthly `har_ma_3125` doesn't — too sluggish for intraday structure).
- Honest OOS test (train/test split on the residual, predictor×regime interactions):
  - hour dummies (leftover U main effect): +0.0014 OOS R²
  - **+ clock-regime interactions: +0.0095** (so regime interactions add **+0.008**)
  - + state-regime (RV vs diurnal-norm) interactions: **−0.0027** (FAILS)
  - clock-interactions alone: +0.0079 → the regime signal **does not need** the U-control.
- => intraday regimes are **clock-anchored, not vol-state-anchored**.

## 7. Mechanism: auction / session-transition microstructure
Hour mapping confirmed from `src/features/extractors/calendar.py`: `hour = t.dt.hour`, RTH 09:30–16:00 ET ⇒ hour 9 = open, hour 16 = close, ≥16 = after-hours.
- **Test 2 (localize) — the decisive result.** `corr(har_ma_5, residual)` by hour is +0.05…+0.13 through the continuous session, then **−0.085 at hour 9 (open)** and **−0.085/−0.21/−0.14/−0.03 at hours 16–19 (close + after-hours, peak −0.21 at 17)**. **Sharp, not gradual; not the last slot** → auction/session-transition mechanics, *symmetric* at the open.
- **Test 1 (gamma / hedging demand) — weak.** In the flip region the reversal is modestly stronger when `voldemand` is high (−0.128 vs −0.106); outside it, no effect. A ~20% modulation — real but secondary.
- **Test 3 (month/quarter-end rebalancing) — not yet run.** Blocked by date alignment (see §10).
- **Verdict:** primarily an auction/session-transition (clock-locked) phenomenon, *symmetric* open+close, with dealer-gamma/hedging-demand a minor modulator. The user's "closing rebalancing" instinct is the load-bearing half; "gamma" is real but minor.

## 8. Distillation — ~60% of the tree collapses to one interpretable line
| model | QLIKE | % of tree's edge |
|---|---|---|
| enet base | 0.12516 | — |
| **+ HAR × late-day (6 feats, L1 keeps ~4)** | **0.12457** | **58%** |
| + HAR × 6 regimes (36 feats) | 0.12453 | 62% |
| residualized EBM (the tree) | 0.12414 | 100% |
- The minimal feature (`HAR × late-day indicator`) *is* the bucket-4 sign-flip and carries 58% of the gain; the marginal regimes beyond late-day add only 4%. The remaining **~38% is genuinely nonlinear/higher-order** (the part a tree still earns; EBM-vs-XGB test pending to see if it's pairwise or higher-order).

## 9. Best data to predict this regime (priority)
1. **Auction imbalance feeds** (NYSE Order Imbalances / Nasdaq NOII) — paired/imbalance shares + side + indicative price at the open & close crosses. Most direct.
2. **Dealer gamma (GEX), 0DTE flow, charm/vanna** — the hedging modulator; 0DTE is now the dominant late-day gamma driver.
3. **Order flow imbalance (OFI)** — distinct from auction imbalance (continuous signed flow). *Constructible NOW* from `buyturnover`/`sellturnover` as `(buy−sell)/(buy+sell)`; the depth-change (Cont–Kukanov–Stoikov) version needs L2.
4. **Flow calendar** — OPEX/quad-witch, index reconstitution, month/quarter-end. Free, derivable from the date (we have it).
5. **Intraday-resolved liquidity** (spread/depth by slot, after-hours volume) for the 16–19 thin-book reversal; **overnight gap/news** for the open (hour 9).

## 10. Date availability & alignment (resolved understanding)
- The date **does exist**: `t` (from `endbartime`) in `data/*.parquet`; `hour`/`DOW` are derived from it; `tasks.py` even anchors **row 189713 = 2020-02-25**.
- BUT the cached rank-matrix (**242,934 rows, 48 half-hourly bars/trading-day, ~2004–2023**) was built from a **fuller history than the `data/` parquets currently on the cluster** (`load_raw_data` → **36,891 rows, 2012–2023, RTH ~13 bars/day**). Hour-alignment finds no offset → can't map row→date with the current parquets alone.
- To finish Test 3 / OPEX features: either (a) the **full source panel** used to build `covid_imp_rank`, or (b) **reconstruct via NYSE trading calendar from the row-189713 anchor**, validating against the cache's `DOW` column.

## 11. Reality check — why isn't this already traded away?
It largely **is** — and consistently so:
- The reversal is the **footprint** of dealer gamma hedging + MOC/auction liquidity provision, i.e. equilibrium residue of trading already happening, not undiscovered alpha.
- It's **tiny** (plateau; ~0.0001 QLIKE) — what's left after arbitrage; a *big* tradeable edge from a public HAR feature would be a red flag.
- It's a **vol-forecast** regularity, not a directional trade (informs option pricing / execution / risk).
- It's **strongest where costs are highest** (thin after-hours) → low capacity.
- The desks with **NOII/GEX** already took the tradeable part; we rediscovered a documented microstructure fact with public HAR features.
- Value = marginally better forecast + mechanistic understanding + a map of which data to buy.

## 12. Predicting on OPEX
- Cheap calendar features (deterministic, known in advance): `is_opex` (3rd Friday), `is_quad_witch`, `days_to_opex`, `days_since_opex` (post-OPEX gamma-roll-off regime).
- Model as a **regime overlay**: `HAR × late-day × is_opex` — expect the reversal **amplified** on OPEX (expiring gamma unwinds into the close).
- Scale by **expiring OI/gamma** + **pinning** (spot distance to max-OI strike; suppresses RV near the pin) — needs OCC data.
- 0DTE caveat: daily expiries have **diluted** the monolithic monthly-OPEX effect → weight recent history / time-vary the OPEX term.

---

## Open items / next steps
- **Date alignment** (§10) → finishes Test 3 (rebalance days) + OPEX/calendar features.
- **OFI test** from `buyturnover/sellturnover` (§9.3) — same-cache, no new data; the natural next probe of the order-flow story before buying NOII.
- **EBM-vs-XGB(d2/d6)** head-to-head — *pending* (job kept timing out single-threaded); tests whether the residual ~38% is pairwise or higher-order interactions.
- **Wire `HAR × late-day`** into the feature pipeline as a permanent feature (turns the tree's regime discovery into an interpretable single-stage term).

## Reproducibility (scripts, cluster-side)
- Spectrum / dead indicators: `analyze_spectrum.py`, `colcheck.py`, `manifest.py`, `verify_drop.py`, `mask_check.py`
- Event channel: `gate_test.py` (L1 gates) | EBM interpretation: `ebm_interpret.py`
- Regimes: `regime_study.py`, `har_flip.py` | Mechanism: `tests123.py` | Distillation: `distill_regime.py` | Dates: `test3_dates.py`
- Code: `resid_amortized.py` (filter `a30f4c5`; `resid_subset_ind` ARM `e3c3d40`).
