# fomcfix_2026-10-26: status

## Jobs (CARC Discovery, account pollok_1603, partition main, constraint epyc-7513|epyc-7542)
| when (PDT) | job | what |
|---|---|---|
| 10-25 23:22 | 13457116 | design, cell A (both FOMC flags 0) |
| 10-25 23:22 | 13457117 | design, cell B (FOMC flags 1) |
| 10-25 23:22 | 13457118 | design, cell C (holidays dropped) |
| 10-25 23:22 | 13457119 | design, cell D (both) |
| 10-25 23:22 | 13457120 | design, P1 only (gate G3b) |
| 10-25 23:24 | 13457128 | gate T1, EPYC 7513 (6 tasks, 8 threads): COMPLETED |
| 10-25 23:24 | 13457129 | gate T1, EPYC 7542 (6 tasks, 8 threads): COMPLETED |

## Gates
- G1 (frame, laptop): PASS (gate_G1_frame.json).
- T1 (trees, CARC): PASS -- 12 of 12 pieces (LightGBM, XGBoost, rf_skx; 2014-01-02 and 2024-04-30; EPYC 7513 and
  7542) bit-identical to the templates' pred_adj, true_adj and B (gate_T1.json).
- S (scoring, laptop, cell A against Table 1, the window sweep's part2 and the tree runs' scores): PASS, 96 checks,
  largest relative difference 8e-15.

## Incidents
- 10-26 00:27: design jobs 13457116-20 hung in /scratch1 I/O (failing mount); cancelled (scancel); addendum C moves the
  linear staging to /home1.
