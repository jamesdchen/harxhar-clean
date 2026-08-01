# Session handoff — 2026-08-01 session 2 (honest multi-step + parsimony closure + operator structure)

Continuation of the 08-01 morning clear. Drivers: `drivers/msweep_2026-08-01/`
(README there; raw logs in its `verdicts/`). Predictions: `results/geo_preds/`,
`results/straddle_*.npz`, `results/sesskern_*.npy`. Memory:
`honest-multistep-and-parsimony-closure-2026-08-01.md`.

USER SCOPE RULING: the product is INTRADAY straddles (H<48). H>=48 descoped.

## 1. THE LEAK (invalidates the multistep pilot's table)

Direct-H training on windows [t-W, t) leaves the last H-1 rows' labels
unrealized at prediction time and overlapping the target by up to (H-1)/H —
near-duplicate answers a wide model interpolates and a 29p HAR cannot.
"Breadth halves QLIKE, growing with horizon" was the leak growing with
horizon. h=1 immune. Honest convention everywhere now: [t-H-W, t-H).

## 2. PRODUCT MODEL (settled, honest, 4y eval [24048,72048))

**Two-block ridge + announcement organ. No VIX. W=24000.**
- Light block (alpha~100): HAR base-2 ladder + calendar (the incumbent, nested).
- Heavy block (alpha=1e4 at H<=8, 1e5 at H=16): all exog rungs, shrunk never selected.
- Organ (28 cols, rides light): per release type since/until + count-in-window
  + bars-until-first-in-window (ex-ante; the window-aware pair is the lever —
  plain since/until is worthless).

| H | incumbent | composite | gain |
|---|-----------|-----------|------|
| 4 | 0.152224 | 0.142102 | -0.0101 p~0 |
| 8 | 0.129500 | 0.120045* | -0.0095 p~0 |
| 16 | 0.112560 | 0.102967 | -0.0096 p~0 |

*H=8 number includes VIX; VIX ablation negative at H=4/16 (and inc+VIX
significantly worse at all three) -> drop VIX; H=8 no-VIX backfill pending.

Laws: alpha scales with horizon AND capacity (dictionary ~1e2-1e3, 1077-col
~1e4-1e5; h=1 wants ~0.1-1). Light-block minimalism: only the persistence
backbone goes lightly shrunk — dictionary-as-light-block underperforms
(0.1232 vs 0.1200) and the heavy overlay contributes nothing on top of it.
Organ is base-independent: -0.005..-0.0067 on incumbent, wide, dictionary.

Hawkes dictionary at product horizons (fair alpha): beats incumbent
(H=4 -0.0029 p=.047; H=8+organ 0.123142 p=.006) but trails the wide
composite by ~0.003 at every H — breadth carries rung-level info the
5-shape compression discards. Dictionary+organ = the legible twin, price ~0.003.

Straddle economics (H=48/240 runs, pre-descope; methods reusable):
known-answer VRP ratios pass (1.37/1.14); net of measured SPY half-spreads
(median 0.15 volpt @1-2dte) inc+VIX+ann net t=+6.5 — the edge survives
friction at the daily horizon. P&L = variance-swap proxy per unit variance
notional, NOT straddle execution. Intraday version blocked on intraday
option quotes (OM is EOD) — data purchase question.

## 3. h=1 PARSIMONY CLOSURE (chunk tile1 [24000,26189); tile2 [26189,28378))

Champion form (validated BOTH tiles): **prune32 + kNN correction + stack**.
- prune32: 203p / 32-of-41 channels == hybrid_dp 0.165145 EXACTLY (both
  variants). Dead families (enet, unanimous): stocktwits x3, vvix, vix3m,
  voldemand x4. vix survives.
- kNN correction+stack: -0.0016 t1 (p=.089) / -0.0003 t2 (harmless) — the
  only residual harvester that never hurts. RFF read-out (256 feats on the
  16-dim PCovR space, per-bar-fresh): ~60% of the increment, microsecond
  latency — fallback, not upgrade.
- avg(champion, legchamp) = 0.161193 — best tile-1 number (all 4 combo
  pairings improve; 2-way > 3-way). NEITHER member t2-validated; the old
  champion (s_stack1000.npz = 0.162099) never ran on t2.

**Meta-law: every sub-0.001 tile-1 scalar-knob gain flipped or vanished on
tile 2** — rawc cells (fixed AND causal), alpha (0.1 t1 / 10 t2, sign flip),
curvature, texture overlay. Survivors of fresh-data contact: the prune
(exact), the correction (adaptive), the organ + shrinkage laws (large).
Tile 1 = comparator, never selector.

Adaptive-alpha closed two-sided: unconstrained 3-expert combiner explodes
(+0.0028 p=6e-4); inverse-MSE simplex weights toothless. Adaptivity pays
only over decorrelated alternatives (correction, model combination).

Trees dead in BOTH seats on the dictionary basis (pure: +0.08 extrapolation
clamp; residual: +0.0096 p=3e-4 vs kNN's -0.0016). Enet: light-alpha ties
(60-col free prune), selection hurts. EBM stalled (default config hours/fit).

## 4. OPERATOR STRUCTURE (the paper's fourth act)

- Probe (ridge of y on [O|ladder], 24k window) -> Km (41x12 transfer
  kernels) -> EWMA-10 pool -> SVD -> 5 shapes. sv = [.65 .54 .39 .22 .18].
- THE 5 QUESTIONS (pooled tile-1 frame, rungs 1..2048 bars): (1) slow
  vol-complex curvature (+2wk -1mo +6wk; vix, vw-vs-ew sumret2);
  (2) tails-vs-priced (ew kurtosis vs vix); (3) quarter-vs-fortnight tilt;
  (4) day-vs-week band-pass; (5) this-bar-vs-8h participation surprise
  (turnover, sellturnover). Shapes are CONTRASTS because the probe controls
  for HAR — the exog dictionary is the orthogonal complement of the HAR
  approximation.
- Separability K(t) ~ sum a_k(t) u_k v_k' receipts: pool memory eff~10-20
  is the ONLY shape lever (cadence non-lever at matched memory; stabilizer
  form non-lever; gauge proven irrelevant BIT-EXACT: proj_e1 == Kbar_e1);
  tshape (estimated temporal modes of the self-kernel) ties fixed power-law
  basis on t1, worse t2 -> kernel deforms WITHIN the power-law family.
- "Not low-rank" (data covariance; PCR fails, plateau spectrum) and
  "rank-5 operator" are about DIFFERENT matrices: compression is on the
  lag axis only; predictive subspace is ~205-dim in data space — dense in
  coordinates because 5 questions are asked 41 ways.
- SESSION-CONDITIONAL OPERATOR (results/sesskern_*.npy, full-sample
  descriptive): target kernel ROTATES through the day (cos vs unconditional:
  overnight +.83, open -.55, close -.21) = the June regime finding as kernel
  geometry. Shape amplitudes swing 3-4x and change sign across session bins;
  span coverage of the pooled 5-frame: 57-76% (vs 42% random), max midday,
  min at transitions (early's 0.58 at n=55k is genuine deformation, not
  noise). Claim: intraday regimes are MOSTLY amplitude motion on shared lag
  structure; the auctions bend the geometry itself.

## 5. PAPER INTEGRATION (main.tex)

Story: OLS-on-exog -> dense-but-weak -> NEW SECTION 5 "the structure of the
dense-weak field" (HAR-as-quadrature -> operator SVD/5 questions ->
separability -> parsimony pricing -> session-conditional geometry), between
linear_vs_nonlinear and algorithm_design; PCovR correction folds INTO
algorithm_design (continues its spectral-kNN arc). Multi-step/product =
PAPER 2. BLOCKER: paper's panel is 218,934 bars / incumbent 0.13415 /
base-5; all new numbers are the 242,934-row b2 panel / 0.17856. The paper's
own one-panel commitment requires the at-scale rerun to be designed on the
paper's convention (or a marked second regime). base-2-beats-base-5 goes
INSIDE the new section (keep the audited incumbent frozen).

## 6. FROZEN LIST v4 (at-scale cluster run — the claims machine)

prune32 control / prune32+kNN-corr+stack / legchamp / old champion
(s_stack1000) / avg(champ,legchamp) / rawc-as-regime-probe / RFF variant.
Pooled per-bar DMs vs matched controls; design the panel to speak the
paper's convention (sec. 5 blocker). All local knobs chunk-selected — the
freeze is the discipline.

## 6b. CODA — THE PLATEAU (final hours of the session)

- **CLOCK ORGAN**: Fourier-modulated shape amplitudes (C x 4 daily-phase
  harmonics), column-scaled. Fourier >> gates both tiles (continuous
  cycle). Penalty curve: s=1 t1 -0.00248 p=.039 / t2 +0.0025; s=0.3
  -0.00247 p=.0069 / +0.0019; s=0.1 -0.00149 p=.0075 / +0.0007 —
  monotone trade improvement, t2 never crosses. Loop is FULL-SAMPLE real
  (session operators); harvest is regime-priced.
- **THREE-WAY TIE AT THE TOP (tile 1)**: champion 0.162099 (s_stack1000)
  == legchamp+clock 0.162104 (p=.998) == prune32+clock+corr 0.162237
  (p=.95). Two of three fully nameable. Compositions all SUB-ADDITIVE
  (clock eats the correction's harvest: corr increment -0.0016 -> -0.0004
  on clocked anchor). avg(pcc, champion) = 0.16039 = best-ever number
  (diversification, unvalidated). t2: pcc +0.0015 (clock tax as
  predicted); prune32+corr remains the cross-tile-robust form.
  **LOCAL SCREEN DECLARED SATURATED** — every remaining difference is
  sub-chunk-resolution; compositions converge on the same residual.
- **DIVIDE-VS-RANK RACED**: DIURNAL_MODE env added to run_geometry_local
  (rank cache prep_cache_all_features_b2_rank.npz built; b2rank_mmap
  exported). Rank +0.00995 WORSE t1 (p=.001), tie t2 — threshold-mixture
  law at the adjustment layer; divide certified.
- lcc t2 log caveat: its two cross-model DM lines compared tile-1 preds
  cross-period (length-guard insufficient) — invalid; t2-internal valid.
- Drivers added: legchamp_clock.py, prune_clock_corr.py (in drivers/
  msweep_2026-08-01/ or scratchpad; pcc script = tile_corr + clock block).
- **PAPER RESTRUCTURED (user directive)**: Hawkes-first 5-section arc —
  see writeup/paper_restructure_2026-08-01.md. Key implication: the
  BUCKET TABLE must be re-run on the b2/kernel convention (fold into the
  cluster campaign). Multi-horizon = PAPER 2 (banked in the blueprint).

## 6c. FROZEN LIST v5 (supersedes v4)

prune32 control / prune32+corr+stack (robust champion) / legchamp+clock /
prune32+clock+corr (pcc) / old champion (s_stack1000) / avg combinations /
clock organ pure at s in {0.1, 0.3} / rawc-as-regime-probe / RFF variant
/ + PER-BUCKET ARMS on the b2 kernel backbone (the S2 table). Pooled
per-bar DMs vs matched controls; design panel to the paper's convention.

## 7. NEXT

1. Commit (this doc + drivers + preds) and /sync. 2. Cluster run (VPN,
human step). 3. Fresh-period validation: 2-block alphas + organ on 2012+/
0DTE era. 4. H=8 no-VIX composite backfill. 5. Intraday option data
decision (product implied side). 6. Macro SURPRISE values — ask user.
7. Paper: draft sections/operator_structure.tex + figures (5-shape
profiles, amplitude daily loop, coverage-by-session).
