# Session handoff — 2026-08-01 (geometry/Hawkes/multi-step campaign)

Continuation of the 07-30/31 geometry campaign. Memory file
`spectral-geometry-pls-2026-07-31.md` holds the full running record;
`writeup/geometry_campaign_lessons_2026-07-31.md` the lessons/evidence table;
`writeup/hawkes_rv_notes_2026-07-31.md` the Hawkes literature + results.
This file: what's true now, what's running, what's next.

## The scoreboard (screening chunk [24000,26189), incumbent 0.17856)

h=1: champion stack 0.16210 (anchor + PCovR local-linear p=0.0072 + learned
combiner p=0.0046) · full anchor 0.16425 · Hawkes-native mixture 0.16743
(216->220p; freshness by kernel reweighting) · 3-param Hawkes 0.18020.

**Multi-step (the product — user trades straddles on all-48-bar forecasts):**
direct full-stack ladder H=8/16/48 = 0.1584/0.1312/**0.0902** vs
incumbent-class 0.2109/0.1966/0.1962 — all significant; breadth grows
monotonically with horizon. Direct-per-horizon is the working architecture;
propagator v1 starved (state-126, iterated~direct p=0.86, both 0.229) but
fitted spectral radius **0.9983** = structural near-criticality confirmation.

## Final-sweep verdicts (landed just before clear)

- **LEGIBILITY CHAMPION 0.163202** — mixture-HN anchor + PCovR local-linear
  + stack; correction -0.00423 vs its anchor at **p=0.0023** (strongest
  controlled increment of the campaign). PCovR survives -> build the
  statistic-space rolling rotation next.
- **Shape pools: EWMA eff-10 = 0.16594 beats expanding 0.16743** (p=0.098):
  "stale shapes" law amended — shapes get a forgetting factor too.
- **Hybrid-257 = 0.16666** (beats 216p, p=0.069) — best parsimonious form.
- Factor-VAR spectral radius 0.9994 (second structural near-criticality hit).

## Post-handoff landings (later than the block above)

- **Factor-VAR v2: PROPAGATION REFUTED at matched information** — 32
  supervised factors: direct 0.2288, iterated 0.2376 (+0.0088, p=0.09;
  iteration compounds error through a rho=0.9994 system). With v1's
  starved tie: **direct-per-horizon is the settled straddle architecture.**
  (Caveat: factors fit vs h=1 target — horizon-mismatched; the sign held
  across both designs. Both factor models FAR below full-X direct 0.0902 —
  compression tax explodes with horizon too.)
- Discounted fitting: eff-24000 = 0.167437 vs flat 0.167430 — exact tie
  (the sanity control: lambda ~ window IS the flat window). eff-12k/6k are
  the live rows, still computing.
- LAUNCHED: `hybrid_dp` — hybrid-257 with the EWMA-10 discounted pool (the
  two amended laws composed; additive prediction ~0.1645 = anchor parity at
  1/4 params). Log `logs/geo_hybrid_dp.log`, marker hybrid_dp_done.marker;
  its 216p replica shares the discounted pool for the nested DM.

## Still in flight at clear (read these logs first)

- `logs/geo_shapepool.log` — expanding vs EWMA-40 vs EWMA-10 shape pools
  (control 0.167430 posted). The proper "stable AND fresh" shapes test.
- `logs/geo_legchamp.log` — mixture-HN anchor + PCovR correction + stack
  (the legibility-edition champion). Decides the rolling-rotation build.
- recovery chain (waits on the two markers above) -> hybrid-257 (shared
  shapes + private kernel per channel; OOM'd in the parallel wave) and
  EW-discounted fitting (eff 24k/12k/6k vs flat; last freshness axis).
- `logs/geo_factorvar.log` — propagator v2 (32 supervised factors as state).
- Markers: `logs/*_done.marker` per run. RAM ceiling: max ~4 concurrent
  3GB screen processes (hybrid OOM'd at 5).

## Hard-won conventions (do not rediscover)

- Alignment: horizon=1 shift is a NO-OP; cache row i <-> raw row 3125+i.
- Cumulative targets: y_H = sqrt(fwdsumRV/fwdsumBaseline); Duan unchanged;
  DM with h=H (HAC).
- Screen harness: run_geometry_local.py, prep caches, arm syntax
  tag:geom:W:k:d:resid:decay:cap:loc:navma:bands:omega; env HAR_BASE,
  HAR_KERNEL, PREP_ROWS (trimmed preps), FAST_ANCHOR, STACK_WINDOW, LL_TAU.
- Rank-1 everything: RollingRidgeResidualizer(fast_inverse=True) =
  block-Woodbury O(p^2)/bar, verified 1e-11. Statistic-space rolling
  PLS/PCovR designed (memory file) but UNBUILT — the one missing per-bar
  piece, needed only if legchamp keeps PCovR.
- Announcements: data/releases.parquet (7 types); since/until features;
  "until" is ex-ante-legal (published schedules); neutral at h=1, aimed at
  intraday-H shape forecasting.
- Laws with receipts: supervision x leash x hygiene; dense-weak ->
  shrink-over-everything, never select; direction != relevance; pool
  structure / refresh amplitudes; texture = coefficient-field detail
  (rungs x time); parsimony prices measured at every K.

## Next queue

1. Read the five verdicts; fold into memory + lessons file.
2. legchamp keeps PCovR? -> build rolling rotation. Else drop it.
3. Multi-step productization: finer H ladder, announcement SHAPE organ,
   forecast-vs-implied straddle signal construction.
4. Cluster frozen list (h=1 claims) unchanged; consider multi-step arms.
5. Unswept: ridge alpha=1 (everywhere). Unbuilt: lgbm anchor. Parked: GEX.
6. Ask user for macro data beyond the 7 release flags (surprise values?).

All local numbers are single-chunk; knobs chunk-selected; frozen-list +
at-scale pooled DMs remain the only claims machine.
