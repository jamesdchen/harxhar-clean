# Session handoff — VRP attribution program + venue verification (2026-07-02)

Cold-resume. Branch `edge-features-legibility`, LOCAL-ONLY. Supersedes `SESSION_HANDOFF_2026-07-01.md`.
The full narrative lives in auto-memory (`evaluation-suite-and-every-bar` — read it first); this file is
the commit map + in-flight/next list.

## Commit map (this session, chronological)
- `c8e9ca4` §7 every-bar window×penalty×bucket battery + DM/MCS (54/54) → `penalty_estimator_floor_2026-07-01.md`
- `0ddd683` eval_sim control package (hjb/hjb_pde/mc_fast/adp) + StructuredPolicy + reclasso empty-active-set fix
- `ac46c88` Risk-Constrained Kelly (Busseti-Ryu-Boyd) base + RCK policy sweep
- `bf01579` derived-policy pipeline (RCK + PDE free boundary); Browne graft tested & rejected
- `bd22177` honest-baseline cumrv PnL (close edge later RETRACTED in raw units — transform artifact)
- `b0984b9` open-window long-vol PnL (later verdict: DECAYED, recent-era ≈ 0)
- `b5a21c2` **VRP attribution stage-1**: Hadamard screen (38 col-bundles) + design factorial → survivors = microstructure flow
- `ec51062` **stage-2 full 2^8 ×2 clusters**: SPARSE (~4 latent factors); activity pair = 78% of lift; best6 +0.80 last1k, +0.61 @×0.8
- `0263938` window ladder: window ≈ non-lever for P&L too (fixed common eval); stage-2's 48k "win" = period luck
- `981d639` P(pass) rescoring: Sharpe proxy validated (Spearman 0.90-0.95); ×0.8-strike CLIFF (activity cells 0.78-0.89 vs ~0 without)

## THE RESULT (professor headline)
Tradeable content of all_buckets vs the implied strike = **sparse microstructure flow** (sumabsret ≫,
sumvolume, voldemand open-only, vvix, Stocktwits pair jointly) on the Corsi HAR core; moments/spreads/
sentiment-alone = QLIKE-only. **QLIKE↔P&L inversion**: dense info predicts RV; the strike subtracts what
the market knows; the residual edge is sparse. Methodology chain (transform artifact, weak-baseline trap,
raw-units validation, designed experiments) is itself a deliverable.

## VENUES (all verified vs primary sources — see memory for refs)
- Topstep & ALL futures/CFD props: CLOSED (options banned; VX/VOLQ not on enumerated list; delta-one
  replication tested dead; flat 3:10 PM CT).
- Vanquish: long calls/puts ONLY (own site — aggregators wrong). PTG: marketing site, no verifiable program.
- LIVE candidates (traditional model): **Maverick Trading** (spreads/condors, ~$5k bond + training),
  **Black Eagle FG** (US equities+options real capital day 1, Sterling; short-premium claim = own blog only,
  CONFIRM BY PHONE +1 833-253-2453), **T3 Trading** (SIE+Series 57, multi-leg spreads), **SMB Capital** (job-like).
  Phone script (6 questions + "is systematic execution permitted?") in conversation/memory.
- Personal options account / SVIX / VX = non-prop expressions.

## LOST ON CLEAR — re-run if needed
Deep-research workflow `wccbgm5sf` (full multi-source venue sweep with adversarial verification) was
in-flight at clear. Script preserved at
`~/.claude/projects/C--Users-james-CC-Allowed-harxhar-clean/ec9e2f66-.../workflows/scripts/deep-research-wf_5911a93d-c37.js`
(+ transcript dir sibling). If its report is needed, re-invoke the deep-research skill with the same args
(full prompt embedded in the script file).

## NEXT (priority order)
1. Professor package: through-line notebook (`vrp_pipeline.ipynb`) + finding notebooks A (transform-artifact
   retraction + weak-baseline decomposition) & B (factorial attribution) + `writeup/vrp_attribution_2026-07.md`
   (verify-first convention; all numbers final & committed).
2. **VIX1D / 0DTE implied history** — the ONE external datum (converts relative Sharpes → absolute; the
   ×0.8-strike cliff makes this decision-critical).
3. Venue phone calls (Maverick / Black Eagle) with the question script; then P(pass) re-run on their exact
   rule set via `simulate_topstep` params + `results/vrp_pnl/` component dumps (any strike/rules post-hoc).
4. Still queued from 6/30: notebook alignment to the 5 contributions; LSTM/GRU scaffold.

## Cluster state
Both queues should be empty (all arrays completed; Hoffman2 13877481 task 16 was a phantom — CELLS=15,
array 1-16 off-by-one, harmless). Verify once: `ssh hoffman2 "bash -lc 'qstat -u jamesdc1'"` and
`ssh usc-discovery "squeue -u jc_905"`. hpc-pi env now has pyarrow (was the instant-crash cause).
Workers/results all committed: `vrp_{screen,stage2,stage2b,design,winladder,pnl}_worker.py` + `results/vrp_*`.

---

## ADDENDUM (2026-07-02 PM session — meeting prep, beat-4 tables)

**Deliverable: `writeup/beat4_meeting_2026-07-02.{tex,pdf}`** (+ `beat4_metric_skill.png`,
generator `writeup/beat4_tex_generator.py` — its paths point at a dead scratchpad; the numbers are
baked into the tex and re-derivable from committed JSONs).
- **Table 1** = §7 bucket battery + completed **MZ column** (exact rank-1 recompute of each winning
  config; values in `results/winablate_full/beat4_mz_by_bucket.jsonl`). Eval dated **Jan 2018 – May
  2025** (74,934 bars; month precision via NYSE-calendar reconstruction from the tasks.py anchor row
  189713=2020-02-25; exact-day mapping blocked by special closures — DOW cols validate the anchor only).
- **Table 2 = NEW: bundle-combo factorial with FORECAST metrics** (the 2^8 stage-2 design re-scored on
  QLIKE/MAE/OOS-R²/MZ + DM). `vrp_stage2_fm_worker.py` (TW env; persists per-bar loss + preds npz
  CLUSTER-SIDE) + `vrp_fm_assemble.py` (cluster-side DM vs tid-0 + exact main effects); submit
  `vrp_fm.sbatch` (CARC tw48k, jobs 9885324-26) / `vrp_fm.sub` (Hoffman2 tw144k, 13886330-32; SGE
  needs `bash -lc` + explicit conda activate — plain ssh gets python2/no qsub). 512/512 in ~25 min;
  pulled summaries = `results/vrp_stage2_fm_tw{48k,144k}/summary.json`; per-cell JSON+npz stay
  cluster-side.
- **Findings**: absret main effect −0.199/−0.257 (tw48k/tw144k) ≈ 2× volume, ~10× rest; best combo
  {absret,volume,vdem_all,bipow} QLIKE 0.34598 (DM −40.9) BEATS all-8 (0.35156) → sparse-beats-dense
  replicates in QLIKE units on the raw panel; absret×volume interaction +0.094/+0.064 = the
  substitution, replicated across windows AND response units. CAVEAT (in captions): raw research
  panel (no diurnal adjustment) → levels not comparable to Table 1 (HAR QLIKE 0.786 vs 0.149); the
  Table-1 cache removes the seasonal channel that inflates Table-2 bundle gains. VRP panel n=246,059
  bars (build() ffills vd_vix BEFORE dropna — naive replication gives 85,625); tw48k eval
  2008-10-09→2024-04-30, tw144k 2016-04-25→2024-04-30.
- **`winablate_r1.py` patched**: battery() now computes MZ inline + persists winning-config
  predictions in the npz (the original run saved only per-bar QLIKE loss → MZ was unrecoverable
  offline; the 54 §7 cells on disk predate the patch — a re-run would backfill).
- **Dead end, kept**: `winbundle_r1.py` + partial `results/winbundle/` (per-bundle-over-HAR worker,
  killed mid-run — superseded by the factorial; deletion proposed, declined for now).
- **UNCOMMITTED**: everything above + modified `winablate_r1.py`.
