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
| 10-26 01:19 | 13458389 | trees cell B, rf_skx full set, 87 pieces: 83 COMPLETED, 4 RUNNING at 13:05 |
| 10-26 01:19 | 13458390 | 7(c) pre-sample tree check, 37 tasks: COMPLETED (286 CPU-h) |
| 10-26 01:19 | 13458518 | 7(b) annual year walks, 66 tasks: COMPLETED |

## Results so far
- STEP 2 (linear factorial): scores_linear/ -- MATERIAL (B vs A and D vs C).
- 7(a), 7(b): scores_step4_linear/ (neighbours nbr_*, annual ANN; vs HAR of cell B and vs the frozen arm of cell B).
- 7(c): tune2013/ (tune2013_tables.md, tune2013_scores.csv, verdict json; piece metas in tune2013/pieces/).
- Trees: LightGBM and XGBoost merged (forecasts/forecasts_gb.parquet); the forest pending.
