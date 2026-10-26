# fomcfix_2026-10-26: status

## Jobs (CARC Discovery, account pollok_1603, partition main, constraint epyc-7513|epyc-7542)
| when (PDT) | job | what |
|---|---|---|
| 10-25 23:22 | 13457116 | design, cell A (both FOMC flags 0): hung on /scratch1, CANCELLED |
| 10-25 23:22 | 13457117 | design, cell B (FOMC flags 1): hung on /scratch1, CANCELLED |
| 10-25 23:22 | 13457118 | design, cell C (holidays dropped): hung on /scratch1, CANCELLED |
| 10-25 23:22 | 13457119 | design, cell D (both): hung on /scratch1, CANCELLED |
| 10-25 23:22 | 13457120 | design, P1 only (gate G3b): hung on /scratch1, CANCELLED |
| 10-25 23:24 | 13457128 | gate T1, EPYC 7513 (6 tasks, 8 threads): COMPLETED |
| 10-25 23:24 | 13457129 | gate T1, EPYC 7542 (6 tasks, 8 threads): COMPLETED |
| 10-26 00:31 | 13457892-96 | designs A, B, C, D, P1 again on /home1 (addendum C), 8 CPUs, 48 GB |
| 10-26 01:02 | 13458194 | linear walks cell A (gate L), 14 tasks |
| 10-26 01:02 | 13458195 | linear walks cell B (full set), 10 tasks, after design B |
| 10-26 01:02 | 13458196 | linear walks cell C, 14 tasks, after design C |
| 10-26 01:02 | 13458197 | linear walks cell D, 14 tasks, after design D |
| 10-26 01:02 | 13458198 | 7(a) neighbour walks, 20 tasks, after design B |
| 10-26 01:02 | 13458199-209 | 7(b) pretune + select, years 2014 .. 2024 (4 CPUs each), after design B |

## Gates
- G1 (frame, laptop): PASS (gate_G1_frame.json).
- G2 (design, CARC login node): PASS -- the cell-A build (both FOMC flags 0) md5-equal to the final design (X, y,
  b_oos, dates_oos of both sets; reference = the md5-verified copy /home1/jc_905/trees_final/designs/w3000).
- T1 (trees, CARC): PASS -- 12 of 12 pieces (LightGBM, XGBoost, rf_skx; 2014-01-02 and 2024-04-30; EPYC 7513 and
  7542) bit-identical to the templates' pred_adj, true_adj and B (gate_T1.json).
- S (scoring, laptop, cell A against Table 1, the window sweep's part2 and the tree runs' scores): PASS, 96 checks,
  largest relative difference 8e-15.

## Incidents
- 10-26 13:56: rf_skx piece 85 slow on node a02-19 (2.3x the template's seconds a refit, would have hit the 8 h limit);
  cancelled (scancel 13458389_85) and resubmitted as 13465922_85 excluding a02-19; the piece resumes from its checkpoint
  (173 refits kept; rf_skx is bit-identical across EPYC 7513 / 7542, RULE_rf_skx.txt gate).
- 10-26 00:27: design jobs 13457116-20 hung in /scratch1 I/O (failing mount); cancelled (scancel); addendum C moves the
  linear staging to /home1.
- G3 (design B vs A, login node): PASS -- baseline md5-equal; full set: y, b_oos, dates equal, only the 12 value
  averages adj_fomc_{until,since}_inv_ma_k differ (gates/G3.json).
- G3b (P1-only build vs A): PASS -- the same 12 columns at most, nothing before 2020-01-30 10:00 (the first design row
  after a changed bar, through the averages' one-bar shift and the clip); through the rolling scaler every later row
  of those columns moves, after 2020-07-31 by at most 0.08 (since_ma_3125: 0.23) in scaled units, median 0.002-0.03.
- GC (design C vs A): PASS -- exactly the holiday rows dropped (676 design rows, 443 in 2014-2024).
- GD (design D vs C): PASS -- as G3.
- L (linear walks of cell A vs w3000): PASS -- pred_adj and true_adj bit for bit, 7 models x 33,997 rows (gates/L.json).

## STEP 2 (linear, scores_linear/): MATERIAL (B vs A; also D vs C); coordinator told 10-26 01:22 and 01:40 PDT.
| 10-26 01:19 | 13458387 | trees cell B, LightGBM full set, 14 pieces: COMPLETED (247 CPU-h) |
| 10-26 01:19 | 13458388 | trees cell B, XGBoost full set, 29 pieces: COMPLETED (380 CPU-h) |
| 10-26 01:19 | 13458389 | trees cell B, rf_skx full set, 87 pieces: 86 COMPLETED; piece 85 ran at 83 s a refit on a02-19 (template 36 s) and was cancelled at 13:56 after 173 of 384 refits (4.06 h), checkpoint kept |
| 10-26 13:56 | 13465922 | rf_skx piece 85 resubmitted (--array=85, --exclude=a02-19), resumes from its checkpoint |
| 10-26 01:19 | 13458390 | 7(c) pre-sample tree check, 37 tasks: COMPLETED (286 CPU-h) |
| 10-26 01:19 | 13458518 | 7(b) annual year walks, 66 tasks: COMPLETED |

## Results so far
- STEP 2 (linear factorial): scores_linear/ -- MATERIAL (B vs A and D vs C).
- 7(a), 7(b): scores_step4_linear/ (neighbours nbr_*, annual ANN; vs HAR of cell B and vs the frozen arm of cell B).
- 7(c): tune2013/ (tune2013_tables.md, tune2013_scores.csv, verdict json; piece metas in tune2013/pieces/).
- Trees: LightGBM and XGBoost merged (forecasts/forecasts_gb.parquet); the forest pending.

## Where the FOMC effect comes from (linear, cell B - A; information)
Sum of the bar-level QLIKE differences (B - A, full set) over the 33,527 scored bars, split by period, as a share of
the mean: every bar ridge -0.00128 = -0.00030 (before 2020-01-29) - 0.00025 (2020-01-29 .. 04-30) - 0.00073 (after);
lasso -0.00517 = -0.00244 - 0.00061 - 0.00213; enet -0.00124 = -0.00037 - 0.00013 - 0.00075.  At 16:00 the +0.0012 ..
+0.0014 is spread over the years (before the window +0.0009, after +0.0004 .. +0.0006, the window about 0).  The
effect is spread over the whole sample, so most of it is the winsorizing change (P2), which touches every bar; P1
acts only through the 2020 window and, by the scaler, a little afterwards.  (No P1-only walk was run: not in the rule.)

## To finish the forest (when 13465922_85 is COMPLETED)
On the CARC login node (one command each, verbatim):
  cd /home1/jc_905/fomcfix_trees/code && PYTHONNOUSERSITE=1 PYTHONPATH=$PWD:$PWD/experiments OMP_NUM_THREADS=1     OPENSSL_CONF=/home1/jc_905/envs/harxhar-r13/ssl/openssl.cnf /home1/jc_905/trees_final/venv_skx/bin/python     experiments/pooled_trees_rerun_13.py merge --design /home1/jc_905/fomcfix_lin/runs/B --out /home1/jc_905/fomcfix_trees/runs/B     --window-days 3000 --refit bar --first-day 2014-01-02 --last-day 2024-04-30 --est rf_skx --set all_features     --require-all --dest /home1/jc_905/fomcfix_trees/runs/B/forecasts_rf.parquet
then fetch forecasts_rf.parquet into results/fomcfix_2026-10-26/forecasts/, concatenate with forecasts_gb.parquet into
forecasts_trees_B.parquet, and score (laptop):
  python results/fomcfix_2026-10-26/code/fomcfix_score.py --main <main worktree> --out results/fomcfix_2026-10-26/scores_trees     --lin A=results/window_sweep_2014start_2026-10-22/w3000/forecasts.parquet --trees A=templates     --trees B=results/fomcfix_2026-10-26/forecasts/forecasts_trees_B.parquet

## Forest, provisional (86 of 87 pieces, 33,586 of 33,970 bars; not scored)
Against the template (rf_skx, cell A): true_adj and B equal on every row; pred_adj relative difference median 0.28 %,
90th percentile 0.77 %, largest 41 %; correlation 0.99986.  Pieces on EPYC 7513 and 7542 (one family).

## Forest done (10-26 16:55)
13465922_85 COMPLETED (resumed from checkpoint); 87 pieces merged; results/pooled_rf_skx_2014start_fomcfix_w3000 built and scored.
