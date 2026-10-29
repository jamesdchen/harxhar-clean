# Ladder claims map: monthly ladder (as printed) vs the every-bar ladder (RULE addendum K)

Repo: `C:\Users\james\CC Allowed\harxhar-0dte-professor` (branch insurance-selling-2026-09-19-e34, HEAD 2f61ba3). Read only.

- Monthly source: `results/ladder_2026-10-26/fomcfix/{scores,readings,posthoc_vs_walk,rank_primary}.csv`, macros from
  `writeup/generated/rehaul_ladder_numbers.tex` (make_rehaul_ladder.py), plus `rehaul_ladder_fig.tex` (caption) and
  `rehaul_channels_numbers.tex` / `rehaul_channels_sparsity_table.tex` (Panel B, the monthly ladder by day type, written by make_rehaul_channels.py).
- Every-bar target: `results/ablations_2014start_w3000/scores_fomcfix/ladall_readings.csv`, macros to `writeup/generated/rehaul_ladder_numbers_everybar.tex`.
  Every-bar ladder has lgbm, xgb, ridge, lasso, enet (NO forest); spans every bar, ORDINARY, EVENT (NO 16:00 span); L_0 = Table 1's HAR walk, L_46 = Table 1's patched full walk; labelled "not pre-registered".

## Phase 1: live claims that state or rely on a ladder result

Scope: live text of `writeup/paper_rehaul.tex` and `writeup/rehaul/*.tex` (comment lines and `\iffalse...\fi` skipped), plus the two generated floats they input.
`\editc` boxes print nothing in the submission build (`\authornotesfalse`); 99_author_notes.tex compiles only under `\authornotestrue`.

### Monthly reference values (primary ranking, span "all")

| model | R1 | R3 | R5 | R10 | R20 | k50/k75/k90 | verdict | best k | p(10 vs 46) | p(5 vs 46) | p(20 vs 46) | gain x1e3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | 3 | 34 | 34 | 95 | 96 | 10/10/10 | INTERMEDIATE | 46 | 0.002 | <0.001 | 0.001 | 25.53 |
| xgb | 4 | 27 | 36 | 98 | 99 | 10/10/10 | INTERMEDIATE | 46 | 0.107 | <0.001 | 0.611 | 25.73 |
| rf_skx | 6 | 42 | 40 | 98 | 101 | 10/10/10 | INTERMEDIATE | 20 | 0.273 | <0.001 | 0.477 | 20.10 |
| ridge | 56 | 84 | 145 | 188 | 171 | 1/3/5 | CONCENTRATED | 10 | <0.001 | 0.031 | <0.001 | 6.86 |
| lasso | 69 | 63 | 108 | 155 | 139 | 1/5/5 | CONCENTRATED | 10 | 0.001 | 0.751 | 0.030 | 11.36 |
| enet | 55 | 58 | 99 | 130 | 120 | 1/5/5 | CONCENTRATED | 10 | 0.008 | 0.911 | 0.007 | 9.95 |

Aggregates as printed: `\rhldTreeRTenMin/Max` 95/98 (lgbm 95, xgb 98, rf 98); `\rhldTreeRFiveMin/Max` 34/40 (lgbm 34, xgb 36, rf 40);
`\rhldLinRTenMin/Max` 130/188; `\rhldLinRFiveMin/Max` 99/145; `\rhldLinPTenFullMax` 0.008; `\rhldLinKbestMin/Max` 10/10;
`\ladderTopK` 10, `\ladderTreeRetainRng` 95--98 (= TreeRTen range, 3 trees), `\ladderLinKstarRng` 10 (= LinKbest range);
`\rhldTreeRankFomcUntilMin/Max` 7/8 (lgbm 7, xgb 8, rf 7; rank_primary.csv, shared by both ladders);
post hoc `\rhldLinPTenVsWalkMin/Max` 0.096/0.795 (monthly top-10 vs Table 1 every-bar full walk);
`\rhld<M>PctMonthlyVsWalk` lgbm 1.0, xgb 1.4, rf 1.2, ridge 2.6, lasso 3.3, enet 1.6 (%).

### A. Directly ladder-dependent claims

| ID | file:line | sentence (short) | macros (monthly value) | reading it rests on | forest? |
|---|---|---|---|---|---|
| L1 | rehaul/A_fi.tex:371-377 | "we refit each model on its top k exogenous series only, k = 0, 1, 3, 5, 10, 20 and 46, at the origins of Appendix app:rh_fi_compute [monthly] and with the frozen hyperparameters, and score every rung on the 33,527 bars ... the reading rule was fixed before scoring." | `\rhfipNExog`=46, `\rhorNIntB`=33,527, `\rhorPenYears`=2013 | design: monthly refits at 126 origins (app:rh_fi_compute); pre-registered RULE sec. 6 | implicit ("each model", 6 incl. forest) |
| L2 | rehaul/A_fi.tex:382-383 | "The figure and the numbers below, like the same ladder by day type in Panel B of Table tab:rh_channels_sparse, are on the design with the FOMC fixes" | none | both ladders patched; "the same ladder" = Panel B is the monthly ladder | no |
| L3 | rehaul/A_fi.tex:385-386 | "Trees: the top 10 series keep 95--98% of the gain, the top 5 only 34--40%" | `\rhldTreeRTenMin/Max`=95/98; `\rhldTreeRFiveMin/Max`=34/40 | R10, R5 of lgbm, xgb, rf (all) | YES (range spans 3 trees; RFiveMax 40 is the forest's) |
| L4 | rehaul/A_fi.tex:386-388 | "fomc_until_inv, ranked 7--8 by the trees, enters between the two rungs" | `\rhfipFomcUntil`; `\rhldTreeRankFomcUntilMin/Max`=7/8 | rank_primary.csv (ranking, not refit); implies the R5->R10 jump | YES (range over 3 trees; without rf still 7--8) |
| L5 | rehaul/A_fi.tex:388-389 | "The other series add little (top 10 against the full set: DM p = 0.002, 0.107 and 0.273)" | `\rhldLgbmPTenFull`=0.002, `\rhldXgbPTenFull`=0.107, `\rhldRfPTenFull`=0.273 | DM p of k=10 vs 46, span all | YES (third p is the forest's) |
| L6 | rehaul/A_fi.tex:396-397 | "Linear models: with 10 series they keep 130--188% of the gain, more than the full set (top 10 against the full set: DM p <= 0.008)" | `\rhldLinRTenMin/Max`=130/188; `\rhldLinPTenFullMax`=0.008 | R10 > 100 and p10 < 0.01 for ridge, lasso, enet | no |
| L7 | rehaul/A_fi.tex:397-398 | "and with 5 series 99--145%, as much as the full set or more" | `\rhldLinRFiveMin/Max`=99/145 | R5 >= ~100 or p5 not significant (generator assert: R5 > 100 or p5 > 0.10) | no |
| L8 | rehaul/A_fi.tex:398-399 | "the lasso's single top series, fomc_until_inv, alone keeps 69%" | `\rhldLassoTopOne`=fomc_until_inv; `\rhldLassoBarROne`=69 | lasso R1, span all | no |
| L9 | rehaul/A_fi.tex:399-402 | "Part of the full set's shortfall is the cost of holding a fit on 564 predictors for a month: re-estimated at every bar (Table 1), the full sets are as accurate as the monthly top-10 rungs (post hoc; DM p from 0.096 to 0.795)" | `\rhfipNCols`=564; `\rhldLinPTenVsWalkMin/Max`=0.096/0.795 | posthoc_vs_walk.csv (monthly top-10 vs Table 1 every-bar full walk, linear) | no |
| L10 | rehaul/A_fi.tex:410-411 | "On every bar, then, the exogenous signal is concentrated, not dense and weak, in both model classes" | none (rests on the R values above) | verdicts: linear CONCENTRATED (k90=5), trees INTERMEDIATE (k90=10) -- note: already loose for the trees under the RULE's words; holds as "not DENSE AND WEAK" | implicit ("both model classes") |
| L11 | rehaul/01_introduction.tex:125 | heading "The gain is concentrated, not dense and weak." | none | ablations (group shares) + ladder verdicts (no DENSE AND WEAK) | no |
| L12 | rehaul/01_introduction.tex:133-136 | "Along a ladder of nested predictor sets ranked before the evaluation span, the 10 highest-ranked series retain 95--98% of the trees' full-set gain, and the penalized linear models do best with 10 series (App. app:rh_fi_ladder, Fig. fig:rh_ladder)" | `\ladderTopK`=10; `\ladderTreeRetainRng`=95--98; `\ladderLinKstarRng`=10 | tree R10 range (3 trees); linear best rung (argmin QLIKE over k) = 10 for all 3 | YES (implicit: "the trees" = 3, range includes rf 98) |
| L13 | rehaul/01_introduction.tex:137-138 | "On the dense-versus-sparse question, the exogenous signal sits in a few series rather than spreading thinly over many." | none | verdicts (k90 <= 10 everywhere, no DENSE AND WEAK) + linear best k = 10 | no |
| L14 | rehaul/09_conclusion.tex:46-50 | "The gain is concentrated ... the 10 highest-ranked series retain 95--98% of it [the trees' gain], and the linear models do best with 10 series. The signal sits in a few series rather than spreading thinly over many." | `\ladderTopK`=10; `\ladderTreeRetainRng`=95--98; `\ladderLinKstarRng`=10 | as L12 + L13 (first clause, FOMC/RM, is the ablation) | YES (implicit, as L12) |
| L15 | rehaul/A_fi.tex:369 -> generated/rehaul_ladder_fig.tex (Figure fig:rh_ladder caption) | "How many exogenous series carry the gain. Each model is refitted on its top k ... ranked once ... on 2013 data ... QLIKE of the rung relative to the same model's k = 0 rung ... on the 33,527 bars (left) and on their 16:00 bar (right). Monthly refits at 126 origins, each held for 21 sessions, window W = 3,000 ..., hyperparameters frozen. FOMC-patched design." | `\rhldNSeries`=46, `\rhldRankYear`=2013, `\rhldNBars`=33,527, `\rhldNOrigins`=126, `\rhldHoldSessions`=21, `\rhldWindowDays`=3,000 | monthly design; 16:00 span; 6 model lines | YES (legend: 6 models incl. random forest) |
| L16 | rehaul/08_feature_importance.tex:192 -> generated/rehaul_channels_sparsity_table.tex (tab:rh_channels_sparse caption + Panel B + note) | caption "...the share of the full set's gain kept by the top k series"; Panel B R_k by day type (EVENT / ORDINARY) for 6 models; note: "Panel B: each model refitted monthly on its k highest-ranked exogenous series (... the feature-count ladder of Section sec:rh_fi), the FOMC-patched design" | `\rhch<M>{Event,Ord}{LadFullPct,ROne..RTwenty}` (e.g. LightGBM Event R10 100, Ord R5 93; Ridge Ord R5 824, R10 1391; Lasso Ord "--") | monthly ladder by day type (make_rehaul_channels.py item 3 (iii)) | YES (Random forest row) |

Panel B monthly values (R_k %, Full = % QLIKE k46 vs k0):

| model | EVENT Full | R1 | R3 | R5 | R10 | R20 | ORD Full | R1 | R3 | R5 | R10 | R20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ridge | -23.2 | 66 | 95 | 96 | 101 | 99 | -0.3 | -70 | -75 | 824 | 1391 | 1168 |
| Lasso | -46.9 | 64 | 67 | 92 | 96 | 97 | +1.0 | -- | -- | -- | -- | -- |
| Elastic net | -23.0 | 90 | 99 | 97 | 102 | 98 | -2.2 | -7 | -16 | 102 | 180 | 159 |
| LightGBM | -62.0 | -1 | 5 | 3 | 100 | 100 | -5.5 | 12 | 88 | 93 | 86 | 90 |
| XGBoost | -63.5 | 0 | 10 | 3 | 99 | 100 | -5.1 | 13 | 64 | 108 | 95 | 99 |
| Random forest | -44.4 | -3 | 3 | 6 | 100 | 100 | -4.6 | 20 | 109 | 97 | 95 | 102 |

### B. Ladder-adjacent claims (design/consistency or the window x k grid, which is the MONTHLY ladder at four windows)

| ID | file:line | sentence (short) | macros (value) | reading it rests on | forest? |
|---|---|---|---|---|---|
| G1 | rehaul/A_fi.tex:443-447 | "The feature-count ladder of Appendix app:rh_fi_ladder at rolling windows of W = 1,000, 2,000, 3,000 and 3,500 trading days" | `\rhcxW*` | identity: grid = the appendix ladder at 4 windows (true only while the appendix ladder is monthly) | no |
| G2 | rehaul/A_fi.tex:457-459 | "Each rung is refitted at the 126 monthly origins ... The full rung equals the every-bar walk at every origin (gates)" | `\rhcxNOrigins`, `\rhfipOriginEvery`=21, `\rhcxNGatesPass/NGates` | grid design (monthly) | no |
| G3 | rehaul/A_fi.tex:464-466 | "At the main window the trees' gain rises with the number of series and is nearly complete at 10 (LightGBM at W = 3,000: 8.73 at k = 5, 24.28 at k = 10, 25.53 for the full set)" | `\rhcxGainLgbmBarMainFive/Ten/Full`=8.73/24.28/25.53 (x1e3; = monthly ladder lgbm) | monthly ladder at W=3,000 (grid cell) | no (LightGBM only) |
| G4 | rehaul/A_fi.tex:466-468 | "The linear models' gain is hump-shaped in the number of series (ridge: 9.94, 12.92 and 6.86)" | `\rhcxGainRidgeBarMainFive/Ten/Full`=9.94/12.92/6.86 | ridge R5 145, R10 188, full 100 (monthly) | no |
| G5 | rehaul/A_fi.tex:411-414, 479-490 | "Across windows ... the best number of series rises with the window ... only for LightGBM, XGBoost and lasso"; per-model k* at four windows (main window: lgbm 46, xgb 46, ridge/lasso/enet 10) | `\rhcxWhoKstarRisesBar`, `\rhcxKstar*BarMain` | grid across windows (monthly); the every-bar ladder has only W = 3,000 | no (grid has no forest) |
| G6 | rehaul/A_fi.tex:494-496 | caveat "The refits are monthly and held for 21 trading days, not the every-bar walks of Table 1; holding a fit costs the full linear set most." | `\rhfipOriginEvery`=21 | posthoc PctMonthlyVsWalk (linear 1.6-3.3% vs trees 1.0-1.4%) | no |

### C. Pointers / framing (no ladder number; checked for consistency only)

| ID | file:line | sentence (short) | note |
|---|---|---|---|
| P1 | rehaul/08_feature_importance.tex:147-150 | "Concentration: the lasso ... ; the feature-count ladder of Maisch et al. (Table tab:rh_channels_sparse; Appendix app:rh_fi_ladder); and a factor-augmented HAR" | evidence pointer |
| P2 | rehaul/08_feature_importance.tex:180-181 | "the lasso's top-ranked series, `\rhldLassoTopOne` [fomc_until_inv], is the proximity to the next statement" | ranking (rank_primary.csv), shared by both ladders |
| P3 | rehaul/A_fi.tex:298-300 | "the paper relies instead on the feature-count ladder (App. app:rh_fi_ladder; by day type, Table tab:rh_channels_sparse) and the ridge--lasso comparison" | pointer |
| P4 | rehaul/A_fi.tex:336-337 | "The feature-count ladder (App. app:rh_fi_ladder) and the grid of App. app:rh_fi_scaling address that question [which series a longer window brings in]" | pointer; only the grid varies W |
| P5 | rehaul/A_results.tex:376-377 | "Table tab:rh_channels_sparse [reports] the lasso against the ridge and the elastic net and the feature-count ladder by day type" | pointer to Panel B |
| P6 | rehaul/10_appendix.tex:52-54 | overview: "the feature-count ladder on every bar (Appendix app:rh_fi_ladder)" | "on every bar" = scored span |
| P7 | rehaul/02_related_work.tex:79-82 | "That our exogenous signal is dense and weak ... is a hypothesis we test ... across the number of exogenous series and the length of the training window" | framing, no result |
| P8 | paper_rehaul.tex:282-283 (abstract) | "The gain concentrates in the FOMC calendar and the futures' realized moments" | group-level (ablations); the source comment (l. 80-81) cites "the feature-count ladder" as co-backing for "concentrated" |
| P9 | rehaul/A_fi.tex:403-408 | Ridge against lasso (`\rhldDmLassoRidge`=-4.82, `\rhldPLassoRidge`<0.001) | named \rhld but from Table 1 walks (rehaul_oos_vs_ols_numbers.tex); not a ladder number |

### D. Not the feature-count ladder (checked, out of scope)

The "beyond" ladder (Table tab:rh_beyond_ladder, `\rhby...`; HAR+RM+FOMC base + top k remaining series, already every-bar, ablations dir):
08_feature_importance.tex:132, 264-273, 274-282 ("under the reading fixed before the ladder was scored XGBoost's remainder is concentrated ..."), 312-316, 359-364;
A_results.tex:428-439 ("Together with the ladder: ..." = the beyond ladder); 01_introduction.tex:139-143; 09_conclusion.tex:53-57. None reads the monthly ladder.
Author notes 99_author_notes.tex:41-44, 66-75 (provenance only; author build only).

---

## Phase 2: the claims checked against the every-bar ladder (RULE addendum K)

Read 2026-10-29 14:37 PDT: `results/ablations_2014start_w3000/scores_fomcfix/ladall_readings.csv`, `ladall_scores.csv`, `ladall_checks.json`
(`pass: true`; label "every-bar refit, requested by the author after the monthly-vs-every-bar comparison; not pre-registered").
Spans `all` / `ord` (2,498 days, 32,474 bars) / `ev` (81 days, 1,053 bars); no 16:00 span; no forest. k = 0 / 46 = Table 1's walks
(match `posthoc_vs_walk.csv`, e.g. lgbm 0.187511 / 0.161058). Every-bar macro names: the generator's `--everybar` mode
(uncommitted in writeup/make_rehaul_ladder.py) inserts EB after the prefix, `\rhld<X>` -> `\rhldEB<X>`, `\ladder<X>` -> `\ladderEB<X>`,
Tree = LightGBM + XGBoost; it writes no origins, post hoc, secondary-ranking or 16:00 names. The rounded values below follow its `num(x, 0)` formatting.
`rehaul_ladder_numbers_everybar.tex` was not on disk when this was read.

### Every-bar values (primary ranking)

| model | span | R1 | R3 | R5 | R10 | R20 | k50/k75/k90 | verdict | best k | p(5 vs 46) | p(10 vs 46) | p(20 vs 46) | Full % vs k0 | gain x1e3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lgbm | all | 4 | 36 | 36 | 97 | 98 | 10/10/10 | INTERMEDIATE | 46 | <0.001 | 0.008 | 0.031 | -14.1 | 26.45 |
| xgb | all | 7 | 29 | 38 | 96 | 97 | 10/10/10 | INTERMEDIATE | 46 | <0.001 | 0.005 | 0.010 | -14.6 | 27.56 |
| ridge | all | 34 | 51 | 87 | 115 | 107 | 3/5/10 | INTERMEDIATE | 10 | 0.084 | 0.030 | 0.244 | -6.0 | 11.43 |
| lasso | all | 46 | 45 | 84 | 105 | 102 | 5/5/10 | INTERMEDIATE | 10 | 0.007 | 0.361 | 0.673 | -9.0 | 16.97 |
| enet | all | 43 | 46 | 78 | 104 | 104 | 5/5/10 | INTERMEDIATE | 20 | <0.001 | 0.522 | 0.231 | -6.7 | 12.67 |
| lgbm | ord | 14 | 91 | 94 | 90 | 94 | 3/3/3 | CONCENTRATED | 46 | 0.067 | 0.004 | 0.042 | -6.0 | 10.01 |
| xgb | ord | 19 | 67 | 102 | 91 | 94 | 3/5/5 | CONCENTRATED | 5 | 0.668 | 0.020 | 0.057 | -6.1 | 10.15 |
| ridge | ord | -6 | -5 | 73 | 129 | 116 | 5/10/10 | INTERMEDIATE | 10 | 0.085 | 0.033 | 0.195 | -3.2 | 5.41 |
| lasso | ord | -9 | -28 | 59 | 140 | 124 | 5/10/10 | INTERMEDIATE | 10 | 0.088 | 0.101 | 0.332 | -2.4 | 4.06 |
| enet | ord | -3 | -6 | 60 | 105 | 110 | 5/10/10 | INTERMEDIATE | 20 | <0.001 | 0.625 | 0.094 | -4.0 | 6.63 |
| lgbm | ev | -1 | 4 | 3 | 100 | 100 | 10/10/10 | INTERMEDIATE | 10 | <0.001 | 0.460 | 0.307 | -62.3 | 533.6 |
| xgb | ev | -0 | 8 | 2 | 99 | 99 | 10/10/10 | INTERMEDIATE | 46 | <0.001 | 0.009 | 0.001 | -63.9 | 564.5 |
| ridge | ev | 67 | 98 | 98 | 103 | 100 | 1/3/3 | CONCENTRATED | 10 | 0.677 | 0.507 | 0.955 | -22.4 | 197.1 |
| lasso | ev | 63 | 67 | 92 | 95 | 96 | 1/5/5 | CONCENTRATED | 46 | <0.001 | <0.001 | 0.003 | -47.2 | 415.0 |
| enet | ev | 91 | 100 | 98 | 102 | 98 | 1/1/1 | CONCENTRATED | 10 | 0.532 | 0.499 | 0.624 | -22.6 | 198.9 |

Headline shifts against the monthly ladder (span all): all five models are INTERMEDIATE (k90 = 10). The linear models move from CONCENTRATED (k90 = 5) because R5 falls from 99-145 to 78-87 and R10 from 130-188 to 104-115.
Their 10-rung no longer beats the full set significantly except in the ridge (p 0.030). The boosted trees are about unchanged (R10 96-97, R5 36-38), but their top 10 now trail the full set significantly (p 0.008 / 0.005).
By day type the readings split. On ordinary days the boosted trees are CONCENTRATED (k90 3 and 5) and the linear models INTERMEDIATE. On statement days the linear models are CONCENTRATED (k90 1-5) and the trees INTERMEDIATE (k90 10; fomc_until_inv ranks 7-8).

### Verdicts

| ID | verdict | monthly -> every-bar | macro it would switch to |
|---|---|---|---|
| L1 | FLIPPED | the design is no longer "at the origins of App. app:rh_fi_compute" (126 monthly origins, 21-session hold): every rung is re-estimated at every bar as in Table 1. There are 5 models, not 6 (no forest). "The reading rule was fixed before scoring" still holds for the rule (ladder RULE sec. 6) but the run itself is "not pre-registered" and needs that label. The linear k = 0 is Table 1's HAR walk under its own baseline penalty, while the rungs k >= 1 carry the full set's penalty. | none (prose); `\rhfipNExog`, `\rhorNIntB`, `\rhorPenYears` unchanged |
| L2 | UNCLEAR | "FOMC fixes": TRUE (both patched). "The same ladder by day type in Panel B" holds only if Panel B is switched to the every-bar ord/ev spans as well (see L16). | none |
| L3 | TRUE | top 10: 95--98 -> 96--97 (lgbm 97, xgb 96); top 5: 34--40 -> 36--38. "The trees" now means the two boosted trees (forest gone; RFiveMax 40 was the forest's). | `\rhldEBTreeRTenMin/Max`, `\rhldEBTreeRFiveMin/Max` |
| L4 | TRUE | ranks 7--8 -> 7--8 (same rank_primary.csv). The R5 -> R10 jump is intact (lgbm 36 -> 97, xgb 38 -> 96; on event days 3 -> 100, 2 -> 99). | `\rhldEBTreeRankFomcUntilMin/Max` |
| L5 | UNCLEAR | "add little" still holds in size (R10 96-97 %; the top-10 QLIKE is 0.6 % / 0.7 % above the full set's). But the cited DM p 0.002 / 0.107 / 0.273 become 0.008 / 0.005 (forest's gone): both boosted trees' full sets beat their top 10 significantly at 1 %. The parenthesis now reads as "small but significant", which matches Section 7's "only the boosted trees capture [the remainder]". | `\rhldEBLgbmPTenFull`, `\rhldEBXgbPTenFull` (no Rf) |
| L6 | FLIPPED | R10 130--188 -> 104--115 (ridge 115, lasso 105, enet 104): still above 100 in point estimate. But "DM p <= 0.008" -> max p 0.522 (ridge 0.030, lasso 0.361, enet 0.522): only the ridge's top 10 beats its full set at 5 %. The monthly generator's assert `max p10 < 0.05` would fail. | `\rhldEBLinRTenMin/Max`, `\rhldEBLinPTenFullMax` |
| L7 | FLIPPED | R5 99--145 -> 78--87 (ridge 87, lasso 84, enet 78). The 5-series rungs are worse than the full set, significantly for the lasso (p 0.007) and elastic net (p <0.001); ridge p 0.084. "As much as the full set or more" is false. | `\rhldEBLinRFiveMin/Max` (+ p: `\rhldEB<M>BarPFiveFull`) |
| L8 | TRUE | the lasso's top series is still fomc_until_inv; R1 69 -> 46 (the sentence only states the number; note R1 46 > R3 45) | `\rhldEBLassoTopOne`, `\rhldEBLassoBarROne` |
| L9 | TRUE (needs reframing) | the every-bar ladder confirms it: re-estimated at every bar, the linear 10-rungs keep 104-115 % instead of 130-188 %, so most of the monthly "shortfall" was the hold. The post hoc p 0.096-0.795 is unchanged as a fact. Under an every-bar report the sentence becomes a monthly-vs-every-bar contrast; a residual shortfall remains for the ridge (R10 115, p 0.030). EB mode writes no post hoc macro. | keep `\rhldLinPTenVsWalkMin/Max` (monthly) or rewrite on `\rhldEBLinRTenMin/Max` + `\rhldEB<M>BarRMonthlyTen` |
| L10 | FLIPPED | "concentrated ... in both model classes": monthly linear CONCENTRATED (k90 5), trees INTERMEDIATE (k90 10). Every bar: 0 of 5 CONCENTRATED, all INTERMEDIATE (k90 10). "Not dense and weak" is still TRUE (no DENSE AND WEAK on any span). The sentence was already loose for the trees; now it fits no model's RULE verdict. | `\rhldEB<M>BarVerdict`, `\rhldEB<M>BarKninety`, `\rhldEBTreeKninetyMin/Max`, `\rhldEBLinKninetyMin/Max` (10/10) |
| L11 | TRUE | heading; "concentrated" is the ablations' group reading (unchanged); no model is DENSE AND WEAK on every bar, ordinary or event days | none |
| L12 | FLIPPED | `\ladderTreeRetainRng` 95--98 -> 96--97 (TRUE, but "the trees'" = the boosted trees only). `\ladderLinKstarRng` 10 -> 10--20: the elastic net's best rung is k = 20 (QLIKE 0.176083 vs 0.176133 at k = 10; R 104.1 vs 103.7). "Do best with 10 series" is false as printed; with the EB macro it prints "10--20" and reads true. "Do best" beats the full set significantly only in the ridge (p 0.030; lasso k10 0.361, enet k20 0.231). | `\ladderEBTopK` (10), `\ladderEBTreeRetainRng` (96--97), `\ladderEBLinKstarRng` (10--20) |
| L13 | TRUE | k90 = 10 of 46 in all five models (monthly: 5 for linear, 10 for trees); no DENSE AND WEAK on any span. "A few series" now means ten for every model. | none (`\rhldEBLinKninetyMin/Max` if quantified) |
| L14 | FLIPPED | as L12: "the linear models do best with 10 series" -> 10--20 (elastic net k = 20). The tree clause is TRUE (96--97, boosted trees only). The last sentence is TRUE (as L13). | `\ladderEBTopK`, `\ladderEBTreeRetainRng`, `\ladderEBLinKstarRng` |
| L15 | FLIPPED | caption: "Monthly refits at 126 origins, each held for 21 sessions" is false for the every-bar run. "Their 16:00 bar (right)": the every-bar ladder has no 16:00 span (ordinary / event instead). The legend has 6 models -> 5 (no forest). The EB generator writes a separate Figure fig:rh_ladder_everybar (every-bar solid, monthly dashed, all-bars panel only). | `generated/rehaul_ladder_fig_everybar.tex`; `\rhldEBNBars` etc. |
| L16 | FLIPPED | Panel B note "each model refitted monthly" -> every bar; Random forest row gone. Lasso ordinary days: full +1.0 % ("--", no gain) -> -2.4 % with R_k -9 / -28 / 59 / 140 / 124. Ridge ordinary: full -0.3 % -> -3.2 %, R_k -70 / -75 / 824 / 1391 / 1168 -> -6 / -5 / 73 / 129 / 116. Elastic net ordinary: R5 102 -> 60, R10 180 -> 105, R20 159 -> 110. XGBoost ordinary: R5 108 -> 102, R10 95 -> 91, R20 99 -> 94. LightGBM ordinary: 12 / 88 / 93 / 86 / 90 -> 14 / 91 / 94 / 90 / 94. The event-day panel is nearly unchanged (e.g. LightGBM -62.0 -> -62.3 %, R10 100 -> 100; ridge R3 95 -> 98). Stale pointer in the note: "the feature-count ladder of Section sec:rh_fi" (it is app:rh_fi_ladder). | `\rhch<M>{Event,Ord}R<k>` / `LadFullPct` -> `\rhldEB<M>{Ev,Ord}R<k>` (Full % has no EB name in the docstring; `\rhldEB<M><Span>GainFull` is x1e3) |
| G1 | FLIPPED (consistency) | "The feature-count ladder of App. app:rh_fi_ladder at rolling windows W = ..." holds only while that appendix ladder is the monthly one; the grid stays monthly | none (prose: "its monthly version") |
| G2 | TRUE | grid design unchanged (monthly, gated) | none |
| G3 | TRUE | every bar at W = 3,000, LightGBM gain x1e3 is 9.60 (k5), 25.53 (k10), 26.45 (full) against grid 8.73 / 24.28 / 25.53: same shape, nearly complete at 10 | none (grid macros stay) |
| G4 | TRUE (weaker) | ridge every bar 9.90 / 13.13 / 11.43 (k5 / k10 / full) against grid 9.94 / 12.92 / 6.86: still hump-shaped, but the peak is 115 % of the full set, not 188 %. Lasso (84 / 105 / 102 at k5 / k10 / k20) and elastic net (78 / 104 / 104) are nearly flat above k = 10. | none |
| G5 | UNCLEAR | not checkable: the every-bar ladder has one window. Its W = 3,000 cell gives k* lgbm 46, xgb 46, ridge 10, lasso 10, enet 20 (grid main-window enet k* = 10). | none |
| G6 | TRUE | the hold costs linear models most: monthly vs every-bar full set +1.6-3.3 % (linear) vs +1.0-1.4 % (trees); linear gains x1e3 6.86 / 11.36 / 9.95 -> 11.43 / 16.97 / 12.67 | `\rhld<M>PctMonthlyVsWalk` (monthly; unchanged) |
| P1 | TRUE | pointer | none |
| P2 | TRUE | the lasso's top-ranked series is unchanged (fomc_until_inv) | `\rhldEBLassoTopOne` (same value) |
| P3 | TRUE | pointer (Panel B caveat as L2) | none |
| P4 | TRUE | pointer | none |
| P5 | TRUE | pointer (Panel B content changes per L16) | none |
| P6 | TRUE | "the feature-count ladder on every bar" (truer still if refit at every bar) | none |
| P7 | TRUE | framing | none |
| P8 | TRUE | group-level (ablations); no ladder number | none |
| P9 | TRUE | Table 1 walks; the every-bar k = 46 rows are those walks | none |

### Counts

- All 31 mapped rows: TRUE 19, FLIPPED 9, UNCLEAR 3.
- The 16 direct ladder claims (L1-L16): TRUE 6 (L3, L4, L8, L9, L11, L13), FLIPPED 8 (L1, L6, L7, L10, L12, L14, L15, L16), UNCLEAR 2 (L2, L5).
- The ones that change the story: L6, L7, L10 (the linear models are no longer "concentrated": 5 series keep 78-87 %, not >= 99 %; the 10-rung beats the full set significantly only in the ridge). L12 / L14: the elastic net's best rung is 20 (the "10" in the introduction and conclusion becomes "10--20"). L5: the boosted trees' remaining 36 series add a small but significant gain.
- Forest-dependent rows (no every-bar counterpart): L3 (RFiveMax 40 is the forest's), L4 (range), L5 (third p), L12 / L14 ("the trees'"), L15 (legend), L16 (forest row). In L3 and L4 the claim survives without the forest.
