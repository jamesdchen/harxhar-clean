# Session handoff, 2026-10-06 (updated 2026-10-07 18:56, resumed after a pause)

Everything below is UNCOMMITTED in this worktree (branch insurance-selling-2026-09-19, at 0a1cdd1 + local changes).
Decisions: `writeup/MASTER_TABLE_REVIEW_2026-10-01.md` items 44-81. Paper edits: `writeup/D1_CHANGELOG_2026-09-29.md`
sections 8-12 (8.16 last). Scratch builds of the paper are clean (79 pages, 0 undefined, 0 overfull, no "??"); `writeup/main.pdf`
and the tracked aux files are untouched.

## Decisions (user)
- No recalibration anywhere: every forecast is the plain f^2 * B. No intervals (point estimates; DM / HAC t stay).
- No `adj_` prefix in presented names. No headline model. Live-feasible and LSTMs are out of the paper (LSTMs untrusted).
- The 65 main-results bars with the B = 1 placeholder baseline are not scored (`experiments/score_unification.py`,
  `DIURNAL_NO_HISTORY_BASELINE`; `--keep-no-history-bars` reproduces the old numbers).
- Input sets in the paper: HAR + calendar, all features, and the lean set `lean_vol_vix` (signed, absolute, quartic
  return, autocovariance, ES volume, VIX), presented as a follow-up to the importance study (new subsection
  `sections/appendix_lean_set.tex`). moments_vix and moments_vol_vix are parked (`results/spxw_pnl/parked_*`).
- Every per-bar linear model is pre-tuned (arm f of `results/linear_pretune_2026-10-06/`: penalty chosen before 2020,
  HAR + calendar at 1/100, margin-guarded retune); paper tables `yhat_subpt_*`, GUI replaces `sub_*` in memory.
- D.14 parked; D.16-D.18 re-based on the 163-forecast model set (`results/close_master_table/paper_model_set.csv`).
- Paper figure `writeup/figures/fig_qlike_sharpe.pdf` (`writeup/make_fig_qlike_sharpe.py`): HAR + cal. -> all features,
  51 pairs, no model dropped, caption macros `\fqs...`.
- GUI (`writeup/master_table_gui/`, untracked): only HAR + cal. and all features; sizing rules sign(s), robust clip,
  full long / scaled short, rank of |s|; reads its own master-table run `results/close_master_table_gui/`.

## Paused 2026-10-07 17:50-18:56 PDT (laptop power); resumed 18:56 -- search complete, winner trial 255 (0.108456)

Cluster jobs keep running on their own; only the local agents, the poller and the GUI server stopped.
Agents' own notes: `results/rf_devmenu_2026-10-06/PAUSED_2026-10-07.md`, `results/trees_rfdev_preview163/PAUSED_2026-10-07.md`,
`results/trees_xgbtop_1600/PAUSED_2026-10-07.md` (if they finished writing them).

1. **Forest 400-trial search** (CARC, journal `/scratch1/jc_905/harxhar-rf-devmenu/results/rf_devmenu_2026-10-06/_work/tree_menu_journal_rf.log`):
   at 17:50 all trial numbers 0..399 were started, 28 still running (expected complete by ~18:40). Uncapped worker arrays
   12777812 / 12790010 / 12790252 (time limits cut to end ~19:15 if the user ran the scontrol line); capped 12792938 tasks exit by themselves.
   On resume: confirm every trial < 400 is COMPLETE, `scancel 12777812 12790010 12790252` if anything is left, then
   `bash cluster/slurm/pull_rf_devmenu_carc.sh` and the freeze `--order number --limit 400 --out results/rf_devmenu_2026-10-06/budget400`
   (see the search agent's PAUSED note for the exact command), write `budget400/RF400_DONE`. Leader at 17:17: trial 255 (0.108456).
2. **Forest fleet on the final entry** (token `rf400`): pre-built by the tree agent; launch after RF400_DONE. CPUs: CARC
   `epyc-7513|epyc-7542` (byte-identical); xeon-4116 / AVX-512 OUT; Hoffman2 AVX nodes (E5-2670 / E5-2650 v2) OK for fixed-settings
   cells; tuned families on Hoffman2 only if the H2-AVX tuned-canary test was identical. Then swap into GUI (item 75 rows become the
   final entry) and paper (numbers agent, then text agent: 87 -> 400 trials, settings, XGBoost disclosure removed if item 76 lands).
3. **XGBoost on xgb_opt_00** (item 76; `results/trees_xgbtop_1600/`): gate / canaries / fleets submitted to CARC (and maybe Hoffman2);
   phase 1 scores + scope check of other xgb_opt_06 uses still to report.
4. **GUI**: restart with `pythonw writeup\master_table_gui\server.py --port 8765` from the repo (env 285J). It shows the trial-163
   forest as the random forest (item 75). The rebuild watcher (scratchpad `gui_watcher.py`) and cluster poller (`cluster_poller.py`;
   delete `cluster_status/STOP` first) can be restarted.
5. Nothing committed. Paper scratch build clean at the last round (79 pages).

## Paper text to fix when the tree round is picked up (text agent, 2026-10-07 23:00; NOT done -- user deferred paper work)
- Methods, trees paragraph: XGBoost look-ahead disclosure goes; forest caveat about library defaults revisited.
- Methods, "Refit contract": still says the tree columns read menu indices 00 and 16 -- WRONG since 8.17 (pooled XGBoost reads 10).
- D.19 Optuna paragraph: forest's first trial = library defaults; mask sentences (D.19, 5.4): library-default forest qualifier.
- D.19 cross-CPU: "these checks predate" sentence; D.19 importance table: XGBoost (and maybe forest) old -> new row mixes the
  settings change with the design clean-up.

## Pooled forest walk (every bar), state 2026-10-08 ~00:00
- Gates passed (repeat canaries bit-identical with 1-thread predict; epyc-7542 at 4 threads = epyc-7513 at 8 threads).
- CARC piece fleets: baseline 12811053 (32 x 20 CPU), lean 12811054 (140 x 20), all_features 12811055 (800 x 20) on
  main (epyc-7513|7542, 4:30 limits) + 12811220 (3 x 21 CPU, largemem, 56 h). Queue order by nice: op_rf400 first,
  then XGBoost lean (50), forest baseline / lean / all_features (100 / 110 / 120). CARC cap: 100 running jobs/user,
  2,000 CPUs/user. Pieces write `.claim`; `revoked_pieces.txt` moves unstarted pieces between clusters.
- Hoffman2 forest 1-slot arrays (chunks 99 downward) submitted once the cross-class checks 15099759 / 15099760 match.
- Projection: baseline ~6 h, lean ~28 h, all_features ~8-10 days (2.5 days if CARC gave the full 2,000-CPU cap).
- Outputs `results/pooled_trees_2026-10-07/yhat_pooltree_<est>_<set>.parquet`; the GUI adds them on restart.

## Pooled forest: finishing steps (agent paused 2026-10-08 15:30 after low-memory kills; jobs run on their own)
Status 15:26: baseline 1,609 / 1,900 pieces (ETA ~3-5 h), lean 4,698 / 8,400 (~12 h), all_features CARC 894 / 22,638 +
Hoffman2 35 / 1,106 (~3-3.5 days). Pooled LightGBM / XGBoost tables done and scored (`score_{lgbm,xgb}/SUMMARY.md`).
1. Timeouts: `bash cluster/slurm/pooltree_sweep.sh <array> <task file> cluster/pooltree_pieces_rf_<set>_sweepN.txt 24`,
   then `submit_pooltrees.sh pieces rf <set> <file>` with TAG=sweepN (CARC, from a shell with no conda env).
2. When a set is complete (baseline 1,900 / lean 8,400 pieces): `TAG=rf_<set> bash cluster/slurm/submit_pooltrees.sh dump`.
3. Before the all_features dump: `bash cluster/transfer_pooltrees_h2_to_carc.sh` (local); revoke Hoffman2's unstarted
   pieces if it lags (15100611 -> `revoked_pieces.txt`; 15101135 -> `revoked_pooltree_pieces_rf_all_features_h2r.txt`;
   15101593 -> `..._h2r2.txt`) and rerun them on CARC.
4. Pull + score: `bash cluster/slurm/pull_pooltrees.sh "yhat_pooltree_rf_*.parquet"`, `python experiments/pooled_trees_score.py --est rf`;
   restart the GUI to show them.

## 2026-10-08 evening state
- Lean set PARKED everywhere (review item 84; changelog 8.18): paper model set 112 forecasts (9 + 52 HAR + cal. + 51 all
  features); scratch build `nolean1` clean, 74 pages. Per-bar tree changes (xgbtop / rf400) still deferred.
- Feature MCS (elastic-net probe, add-one + leave-one-out, QLIKE + trade loss) running: `results/feature_mcs_2026-10-08/`.
- Importance PDF final (review item 83): trial-255 forest, clusters / de-dup / gates commented out, linear over-time figure.
- Pooled forest: HAR + cal. ~done tonight, lean later, all features ~3 days; agent dumps / scores each set; GUI restart then.

## 2026-10-09 state
- Feature MCS done (linear 288 + trees 288 = 576 candidates, four losses): no joint MCS can exclude HAR + calendar or pick
  an input set; pairwise, ES volume and cross-sectional absolute returns help. `results/feature_mcs_2026-10-08/SUMMARY.md`.
  Not in the paper yet (the lean subsection it replaces is parked, changelog 8.18).
- Pooled trees: LightGBM / XGBoost on all three sets and the forest on HAR + cal. and lean done (GUI shows HAR + cal. and
  all features; lean hidden). Pooled forest on all features (every bar) runs unattended in waves: ~6-8+ days at our
  CARC fair-share (0.039); resume the pooled agent ~daily with `results/pooled_trees_2026-10-07/RESUME.md`.
  Open user decision: keep every-bar or switch this one table to refit every 10 bars (~1 day).
- Importance PDF final (trial-255 forest; clusters / de-dup / gates commented out; linear over-time figure).
- Deferred: per-bar tree changes into the paper (xgbtop / rf400), "Refit contract" sentence (bank entry 16 -> 10).

## 2026-10-13 CARC incident (read before touching CARC)
- /scratch1 reads hang (D state) for many files under /scratch1/jc_905/{harxhar-pooltrees,harxhar-mcstrees,harxhar-feature-mcs}
  (OSTs 9, 10, 13, 15 at least). Stuck jc_905 processes on discovery1 and discovery2; discovery2 at the per-user process
  limit. A CARC ticket (draft in the session) is the user's to file. NO file reads / md5sums from CARC login nodes until
  fixed -- any file check inside a batch job only.
- Pooled forest all features: CARC 7,944 pieces done (old root), Hoffman2 4,911 / 5,530; recovery steps at the end of
  `results/pooled_trees_2026-10-07/RESUME.md` (fresh root, batch-job copy of readable pieces, then waves).
- Bucket study: linear done (story fails on QLIKE; `market_vw` the one bucket that helps; trade directional only);
  XGBoost / forest bucket trees on Hoffman2; LightGBM bucket trees wait for CARC.
- PufferLib 5.0 native MinGRU (supervised, sqrt-space target, no PPO / PyTorch): Phase 1 feasibility running.
- A stuck old-root submit may create job `fmcs_bk` when /scratch1 recovers -- cancel it (duplicate).

## 2026-10-13 15:25 coordination
- CARC is back (another user session operates it: pooled-forest wave 7 at NICE=0, a pf_torch GPU timing job, PufferLib
  Phase 2 = Hoffman2 job 15152085). This session does NOT touch CARC while that session is active.
- All of this session's cluster agents stopped at the account's weekly API limit (~01:40) before releasing holds, so
  Hoffman2 sat idle ~14 h. 15:25: pooled-forest Hoffman2 holds released (15101135 / 15101593, priority -100); the
  bucket fleet 15145176 is still held because its canary 15145174 ran out of memory at h_data 5G -- the tree agent
  resubmits with more memory; the old 15145176 needs `qdel` by the user. MCS agent waits for BUCKETS_XGB_RF_DONE.
- CARC ticket: dropped (user, 2026-10-13). CARC works again for the other session.

## 2026-10-13 15:40 -- the other session is stopped; this session owns everything again
- The other session submitted nothing on CARC (no jobs since 14:30): its eager-PyTorch timing job is superseded by the
  Hoffman2 benchmark (native vs eager / torch.compile / batched, one GPU); its WAVE=7 refill never ran.
- PufferLib Phase 2 (Hoffman2 15152085, cluster/puffer_forecast_phase2_h2.sh: 5 blocks, re-tune every 250 sessions,
  cold refit, 48 concurrent, L40S) -> PufferLib agent watches, pulls, scores vs the paper models and the old LSTM.
- CARC pooled forest -> pooled agent (batch-job read check of the old root, fresh root if needed, sweep, waves at
  nice >= 100). LightGBM bucket packs -> tree agent on CARC (fresh root harxhar-mcstrees-b, nice 0).

## Still running / open
| item | state | next |
|---|---|---|
| Random forest on the 1998-2002 dev-span tree menu (`results/rf_devmenu_2026-10-06/`) | DONE: frozen at 87 trials; per-bar re-run (`results/trees_rfdev_1600/`); swapped into GUI (item 73) and paper (model set, Tables 7/8, D.19 / 5.4 tree counts; `MACROS_RFDEV2_DONE`; text 8.16 incl. XGBoost disclosure) | the forest's RS / Optuna runs start from library defaults, not the dev settings (stated in the text); masked-vs-unmasked keeps the default forest; CARC job 12777812 (400-trial search) still running -- user to cancel or keep |
| Forest 400-trial search + trial-163 work (2026-10-07 afternoon) | search agent adding workers to job 12777812's journal (freeze `--limit 400` -> `results/rf_devmenu_2026-10-06/budget400/RF400_DONE`); tree agent: trial-163 preview (CARC epyc-7513), speculative 13 re-tuning families + unmasked from trial 163 (key token `rfp163`), Hoffman2 cross-CPU check of the dev forest; all in `results/trees_rfdev_preview163/` | if the freeze picks trial 163, the rfp163 runs replace rfdev + the default-start tuned forests in paper / GUI (user OK needed); else re-run with the new entry |
| XGBoost on its top dev entry `xgb_opt_00` (item 76) | agent running: fixed T1 / T10 + 13 re-tuning families + unmasked, 3 sets, CARC epyc-7513 (token `xgbtop`, `results/trees_xgbtop_1600/`, flags XGBTOP_P1_DONE / P2_DONE); scope check of other xgb_opt_06 uses | integrate into paper / GUI (drops the look-ahead disclosure) |
| Trees on top entries + pooled trees (items 76-81, 2026-10-07 night) | xgbtop per-bar DONE (15 families); rf400 per-bar fixed + RS DONE, Optuna + unmasked running; 15:30 pooled XGBoost -> arm 10 DONE (changelog 8.17); pooled LightGBM / XGBoost on HAR + cal. and lean running (every bar); pooled forest every bar on all three sets (~141k CPU-h) starting on CARC + Hoffman2 AVX | paper integration NOT now (user, 2026-10-07 23:00: focus on cluster parallelism); GUI picks up new tables on restart |
| Pooled linear pre-tune | DONE: paper and GUI use arm (f) (`poolpt_*` in the paper; GUI swaps `pool_*` in memory) | -- |
| RF / XGBoost 2016-19 pre-tune study | DONE (`results/trees_pretune_2026-10-06/`) | superseded for the forest by the dev-span search; XGBoost keeps its shipped settings |
| Live close-signal card | switched to plain in the working tree; tests 76/76, deck gate 866/866 | push `live/close_signal/forecast.py` together with `notebooks/atm_straddle_lib.py`; re-estimate `DEFAULT_LONG_FRACTION` (user to decide) |
| GUI rebuild watcher and cluster poller | background | stop with the STOP files in the scratchpad |

## Pending user decisions
1. CARC job 12777812 (the forest's 400-trial search, up to ~4,000 CPU-hours, ends ~20:20 PDT 2026-10-07): cancel
   (recommended; 87 trials = XGBoost's surviving count) or keep (result to `results/rf_devmenu_2026-10-06/budget400`).
(Answered 2026-10-07: change plot not the main figure; live card not re-estimated or pushed for now; dev-tuned forest
swapped in. Earlier: live card -> plain in the working tree; D.16 extra bars not run.)

## Known breakages (pre-existing)
- `experiments/dvs_dedup_1530.py` fails at import since b761b28 (`LF, AF = BUCKETS`).
- `experiments/linear_dedup_plan.py` asserts 76 forecasts; the new tables make it fail if re-run.
- CARC jobs left to cancel by the user (this session cannot): 12767830-33, 12768625, 12768626, and mvv_s2 / mvv_m2.
