# D1 changelog — tightening the close-option story (2026-09-29, agent W11)

One change per commit, each revertible on its own (`git revert <sha>`; later
changes that build on an earlier one are marked **depends on**). Every number
quoted below is read from the executed output named beside it; in the paper every
new number is a `\cm...` macro written by `writeup/make_table_close_main.py` from
those files (nothing typed by hand), with the source named in a comment on the line.

Build of record before any change (commit `1f5c6be`): `main.pdf` 67 pages, 12
overfull boxes (none in Section 5.4; the largest are the appendix protocol table and
the strategy-variation longtables), one multiply-defined label
(`tab:app_strategy_variations_res_short`, inside `generated/strategy_variations.tex`,
not touched here), no undefined references.

---

## 1. Map of the story before editing

### 1.1 What the front matter promises about the close

| where | promise | scorer / source |
|---|---|---|
| Abstract | nothing (the option reading was parked from the abstract on 2026-08-18) | — |
| Introduction, 4th finding | the sign of the gap $s=\widehat{RV}-\mathrm{IV}^2$ is the tradable content; forecast and implied disagree on 32–41 % of days; sign(s) improves on always short for all eight forecast columns, "exactly one of the eight is resolved at 95 %, marginally"; sizing on the magnitude does not help; headline column = the block-diagonal ridge on the FOMC design, four comparators on the earlier design | deck (session-bar recalibration), typed numbers |
| Related works, last paragraph | "what the forecast adds is **size** on that short rather than a directional flip" | **contradicts** the results: only the sign is used and sizing does not help (Section 5.4; Appendix D.11; `results/close_kelly/`) |
| Conclusion | same claims as the introduction | deck, typed |

### 1.2 What Section 5.4 (`sections/results_close_option.tex`) actually shows

- **One table** (`generated/table_rule_by_strategy_paper.tex`, from
  `results/atm_straddle_0dte_1530/rule_by_strategy_*.csv`, rv_iv notebook): the paper's
  eight forecasts, all fitted on all 48 bars of the day, **deck scorer** (Mincer–Zarnowitz
  map over the regular-session bars, 250 sessions), **midpoint only**. Block-diagonal
  ridge 1.34; the column ordering is "not asserted".
- **Prose, every number deck scorer and mostly on the block-diagonal ridge**: control
  (0.014/day, Sharpe 0.20); paired improvements 0.76–1.31 with 7 of 8 intervals
  including zero; headline 1.34 [0.27, 2.42]; FOMC-column gap +0.07; sample splits
  (1.51 without Q1-2020; 0.98 / 1.63 around 16 May 2022); pin days (43 % of the return);
  eight rows = 1.8 bets; the threshold decomposition of QLIKE and the Murphy diagram
  (`generated/table_murphy_1530.tex`, macros `\mur...`, study 59); same-day regression
  (figure `regression_R_on_signal_blk2.png`); crossed spread 1.34 → 0.87; shift test
  0.48 / 1.34 / 3.11; compounding at 3 % of wealth (7.2×).
- **Not in the paper at all**: the 15:30-specific (per-bar) linear models, the per-bar
  trees (untuned and causally tuned), the master table, the P&L decomposition, the
  strategy variations (Appendix D hook only), the post-2024 record (Appendix D.5 only).

### 1.3 Where the evidence is not apples-to-apples

| # | issue | evidence |
|---|---|---|
| 1 | **Two scorers.** The paper's table and prose use the deck's session-bar map; every per-bar, tree and master-table comparison made since 2026-09-20 uses the 16:00-bar recalibration $(f^2+s)B$. The same forecast reads 1.34 (deck) and 1.00 (16:00-bar) — so nothing tonight can be set beside the paper's table. | `results/close_master_table/SUMMARY.md` (reference row); `results/close_pnl_decomp/SUMMARY.md` (1.34 deck) |
| 2 | **Headline forecast.** The section's headline is the block-diagonal ridge; the master table recommends the per-bar ridge on the live-feasible set (1.90 / 1.44; +0.90 [+0.19, +1.66] vs the block-diagonal ridge). | `master_table.csv` |
| 3 | **Several things move at once** between the paper's forecast and the per-bar models (design, window, 48 bars vs one bar, inputs, estimator). The master table carries a pooled twin (same specification, all 48 bars) that isolates one of them; the paper shows none. | `master_table.csv` family "pooled twin" |
| 4 | **Fills.** The table is midpoint only; the crossed spread appears for one forecast in prose and as the last column of the Murphy table, so the two fills never sit side by side for all forecasts. | `table_rule_by_strategy_paper.tex`, `table_murphy_1530.tex` |
| 5 | **Intervals.** The paired intervals quoted (vs always short) are deck-scorer; none against the paper's own forecast; no interval for the per-bar vs 48-bar comparison. | prose lines 60–68 |
| 6 | **Numbers typed in prose** (all of 5.4 except the `\mur` macros), from the notebook's executed output; not re-derivable from a script at build time. | `results_close_option.tex` |
| 7 | **Four of the eight columns on an earlier design panel** (†), and two tree columns from a lost menu — the paper's own forecast set is not like-for-like; the per-bar family is (one design, one window, one target). | methods, "Which panel produced which row"; `results/spxw_pnl/MANIFEST.md` |
| 8 | **A result stated inside the methods**: "implied variance exceeds the forecast on 59 to 68 % of days (60 % for the headline)" — deck numbers, and the headline changes. | `methods_close_option.tex` "Scoring and control" |
| 9 | **The sample's end.** Section 5.4 says the 413 days after April 2024 "appear nowhere below", but Appendix D.5 now reports the trade's live form on them (no edge); the main text never says so. | `sections/appendix_running.tex` D.5 |
| 10 | **Wording.** "package" throughout (the wording rule is *straddle*, defined once); the straddle is defined in the methods only as the case where the two strikes coincide. | methods / results |
| 11 | **Appendix D conventions** point to Table 7 for the 866 days and the $\sqrt{252}$ convention and take the block-diagonal ridge as their forecast; they are W7's and are not edited here — the label `tab:close_option_rules` is kept on the new table so their references stay true (same days, same annualization). | `appendix_running.tex` lines 33–46 |

### 1.4 Plan (smallest and safest first; one commit each)

1. Methods: define the per-bar forecasts (with design column names and a key) and the
   16:00-bar recalibration — definitions only, nothing in the results changes.
2. (a) One scorer: the results table becomes selected rows of the master table
   (`make_table_close_main.py` → `generated/table_close_main.tex`), the deck table is
   parked (commented, dated); the prose that reads the table is re-read on the new table.
   The headline stays the paper's forecast in this step.
3. (b) Headline forecast = per-bar ridge on the live-feasible set, and why.
4. (c) Tuned trees (A2b).
5. (d) Where the P&L comes from (tail trade; month-ends; difference to always short).
6. (e) Strategy variations → Appendix D.
7. (f) Park the deck-scorer diagnostics into Appendix D (`sections/appendix_running_parked.tex`).
8. (g) Introduction / conclusion reconciled; the related-works error fixed as its own change.
9. Wording: package → straddle.

---

## 2. Changes

### Change 1 — Methods: define the per-bar forecasts and the 16:00-bar recalibration

- **What.** `sections/methods_close_option.tex` gains two paragraphs after "What the
  16:00 target is": *Forecasts fitted for the 16:00 bar alone* (rows, window, target,
  inputs with the design column names `har_ma_k`, `adj_x_ma_k` and the key of the three
  input sets `baseline` / `all_features` / `live_feasible` with the 16 live-feasible
  series named; OLS / ridge / lasso / elastic net and their causal penalty grids; per-bar
  LightGBM / XGBoost / random forest, untuned and causally tuned; the pooled twin) and
  *Two recalibrations* (the session-bar map already in the paper vs the 16:00-bar
  recalibration $(f^2+\bar e)B$, new Equation `eq:close_opt_recal1600`; the paired DM
  and block-bootstrap intervals).
- **Why.** The results cannot use one scorer, or show the 15:30-specific models, until
  both are defined; nothing in the results changes in this commit.
- **Before / after numbers.** None (definitions only). Constants quoted are the code's:
  grids `ESTIMATOR_GRIDS`, `TUNE_PER = 250`, `VAL_TAIL = 125`, `EMBARGO = 25`
  (`specs/causal_tune_linear.py`); the 16 live-feasible series
  (`src/data/loading.py`, `SUBGROUPS["live_feasible"]`); refit every 10 sessions, 32
  candidates (`specs/causal_tune_trees.py`, `specs/causal_tune_trees_tuned.py`);
  `SMEAR_W = 250`, `SMEAR_MIN = 63` (`experiments/score_linear_subsection_causal.py`);
  DM lag 6 on 866 days, blocks of 21 days, 2,000 draws
  (`results/close_master_table/master_table.csv` column `dm_hac_lag`,
  `experiments/master_table_close.py`).
- **Build.** 68 pages (+1), 12 overfull boxes (unchanged; the new list is set
  `\sloppy`), 0 undefined references.

### Change 2 — Wording: the straddle, defined once; "package" retired

- **What.** `methods_close_option.tex`, paragraph *Instrument*: the trade is now the
  **straddle** — the nearest out-of-the-money call plus the nearest out-of-the-money
  put, same-day expiry, one position — defined once there (the two strikes coincide
  when the index sits on a strike, otherwise they are one strike apart), and holding
  it from 15:30 to the close is named the **last-30-min trade**. Every rendered
  "package" in `methods_close_option.tex` and `results_close_option.tex` (7) becomes
  "straddle". Parked (`\iffalse`) text is untouched.
- **Why.** The wording rule of the professor-facing write-up (straddle, never
  package); the methods previously called only the coincident-strike case a straddle,
  which clashed with Appendix D and the master table, where "straddle" is the pair.
- **Before / after numbers.** None.
- **Build.** 68 pages, 12 overfull boxes (unchanged), 0 undefined references.

### Change 3 — (a) One scorer in Section 5.4: the table and the prose that reads it

- **What.** New `writeup/make_table_close_main.py` selects rows of
  `results/close_master_table/master_table.csv` (no number typed) and writes
  `generated/table_close_main.tex` (Table 7, 28 forecasts + always short, four panels:
  A the paper's eight 48-bar forecasts, B the pooled ridge twins, C the per-bar linear
  family on `baseline` / `all_features` / `live_feasible`, D the untuned per-bar trees
  on `live_feasible` / `all_features`; columns QLIKE, % and DM vs R, Sharpe mid and
  crossed, ΔSharpe vs R and vs always short with paired intervals, buy share) and
  `generated/close_main_numbers.tex` (`\cm...` macros for every number of the new prose;
  each qualitative claim asserted in the script). In `results_close_option.tex` the
  table and the four paragraphs that read it are rewritten on Table 7; the deck-scorer
  table and those paragraphs are **parked verbatim** (commented, dated) right below the
  new block; the deck crossed-spread paragraph is parked in place and re-read on
  Table 7. A boundary paragraph states that everything after it (sample splits, pins,
  Murphy diagram, same-day regression, shift test, compounding) is still the
  session-bar recalibration on the block-diagonal ridge, not comparable with Table 7.
  The label `tab:close_option_rules` moves to the new table (Appendix D's references
  to its 866 days and its $\sqrt{252}$ convention stay true).
- **Why.** Two scorers were mixed: nothing computed since 2026-09-20 (per-bar models,
  trees, master table) could be set beside the paper's table. The headline is
  deliberately **not** changed here (next change), so this commit moves one thing: the
  scorer.
- **Before → after** (sources: parked deck text / `master_table.csv`,
  `research_scorer/hit_payoff.csv`):

  | quantity | before (session-bar map) | after (16:00-bar recalibration) |
  |---|---|---|
  | forecasts in the table | 8 (mid only) | 28 + always short (mid and crossed) |
  | always short, Sharpe mid / crossed | 0.20 / −0.27 | 0.20 [−0.63, 1.14] / −0.27 [−1.10, 0.63] (forecast-free: unchanged) |
  | block-diagonal ridge, Sharpe mid / crossed | 1.34 / 0.87 | 1.00 / 0.53 |
  | paper's eight, Sharpe mid | 0.97 – 1.51 | 1.00 – 1.54 |
  | paper's eight vs always short | +0.76 – +1.31; 1 of 8 interval above 0 (XGBoost, lower bound 0.017) | +0.80 – +1.33; 1 of 8 above 0 (LightGBM, +1.27 [+0.05, +2.41]) |
  | FOMC columns | ridge with them higher by +0.07 [−0.49, 0.63] | ridge without them higher by +0.12 [−0.35, +0.63] |
  | best paper column | fixed lasso 1.51, "ordering not asserted" | fixed lasso 1.54, above the block-diagonal ridge by +0.54 [+0.07, +1.06] |
  | per-bar linear / untuned per-bar trees | not in the paper | 1.22 – 1.90 / 1.26 – 1.70 |
  | intervals vs the block-diagonal ridge above zero | — | 3 of 26 tabled forecasts; 9 of 99 in the master table |
  | Sharpe lost to the crossed spread | 0.47 (block-diagonal ridge) | 0.46 – 0.48 (every tabled forecast) |
- **Build.** 68 pages, 12 overfull boxes (unchanged), 0 undefined references.

### Change 4 — (b) The headline forecast: per-bar ridge on `live_feasible`, and why

- **What.** A new paragraph block in Section 5.4 ("The headline forecast, and why this
  one") after the table reading; the headline row is marked H (bold) in Table 7; the
  script gains `headline_macros()` (reads `master_table.csv` row
  `sub_ridge_live_feasible`, gated on the master table's own `is_headline`, and
  `master_table_vs_headline.csv`). Claims asserted in the script: the QLIKE gain is not
  significant; both Sharpe intervals (vs R, vs always short, mid and crossed) are above
  zero; the lasso, the elastic net, the all-features ridge and the table's top Sharpe
  forecast are all indistinguishable from the headline; none of the 99 others trades
  above it with an interval above zero.
- **Why.** The master table's recommendation (`results/close_master_table/SUMMARY.md`
  (a)): live-feasible at 15:30, the 15:30 study's forecast of record, chosen on
  feasibility and not as the maximum of 100 Sharpe ratios (the maximum is the per-bar
  lasso on `free_vix_only`, 1.91).
- **Before → after.** Headline: block-diagonal ridge (deck 1.34) → per-bar ridge
  `live_feasible`: QLIKE 0.1005 (−5.0 % vs R, DM −1.42, p 0.156, daily-difference
  interval [−0.0113, +0.0005]); Sharpe 1.90 mid / 1.44 crossed; mean 0.134; hit 54.6 %;
  long 40.2 %; vs R +0.90 [+0.19, +1.66] mid, +0.91 [+0.19, +1.67] crossed; vs always
  short +1.70 [+0.09, +3.11] mid, +1.71 [+0.12, +3.12] crossed; rank 2 / 100 on Sharpe,
  55 / 100 on QLIKE. Not separated from: lasso same inputs −0.08 [−0.89, +0.79] (89 %
  same side, QLIKE −4.1 %, DM −1.20); elastic net −0.49 [−1.08, +0.14]; ridge
  all features −0.24 [−1.07, +0.54]; top-Sharpe forecast +0.01 [−0.75, +0.79]. 0 of 99
  above, 14 of 99 below. All from `master_table.csv` / `master_table_vs_headline.csv`.
- **Depends on** change 3 (Table 7 and the macros file).
- **Build.** 69 pages (+1), 12 overfull boxes, 0 undefined references.

### Change 5 — One change at a time: Table 8 (paired steps from the paper's forecast to the headline)

- **What.** New Table 8 (`generated/table_close_ladder.tex`, label `tab:close_option_ladder`)
  and a paragraph "One change at a time" in Section 5.4. Eleven paired steps, each
  "to minus from" on the same 866 days: the paper's block-diagonal ridge → the per-bar
  specification pooled on all 48 bars; pooled twin → per-bar on each input set; inputs
  of the per-bar ridge; estimator on `live_feasible` (lasso, elastic net, three untuned
  trees). QLIKE % with day-block interval and DM, ΔSharpe mid and crossed with paired
  intervals, share of days on the same side. Computed by `make_table_close_main.py`
  (`ladder()`) from `results/close_master_table/master_table_daily.parquet` with the
  master table's own functions (`paired_sharpe`, `day_block_ci`, `dm_test`); the pairs
  are written to `writeup/generated/close_main_ladder.csv`. **Gate:** 8 of the 11 steps are
  pairs the master table already stores (vs the reference or vs the headline) and are
  reproduced to 1e-9; the 3 pooled → per-bar steps are new pairs.
- **Why.** The professor's "apples-to-apples, moving one thing at a time": the headline
  differs from the paper's forecast in several things at once; this shows which step
  carries the difference.
- **Numbers (new).** Specification step −0.01 [−0.73, +0.76]; per-bar step +0.68
  [−0.12, +1.45] (`all_features`), +0.78 [−0.08, +1.52] (`live_feasible`), −0.03
  [−0.66, +0.64] on `baseline` where QLIKE gains a resolved −11.2 % [−15.7, −6.9]
  (DM −4.59); inputs all → live +0.24 [−0.54, +1.07], baseline → live +0.69
  [−0.23, +1.70]; estimator from the headline −0.08 (lasso) to −0.49 (elastic net).
  **No step's interval excludes zero** (asserted); the three steps from the paper's
  forecast to the headline add up exactly to its resolved +0.90 lead.
- **Depends on** changes 3 and 4.
- **Build.** 70 pages (+1), 12 overfull boxes, 0 undefined references.

### Change 6 — (c) The causally tuned per-bar trees

- **What.** Paragraph "Trees, untuned and causally tuned" in Section 5.4, and three
  rows in Table 8 (tuned LightGBM / XGBoost / random forest on `live_feasible`, MSE
  selection = the arm of record). Script: `tuned_macros()` reads
  `results/linear_subsection_trees_tuned/a2b/qlike_1600_paired.csv` and
  `trade_1600_paired.csv` (866 deck days, same 16:00-bar recalibration) and gates all
  nine tuned-tree Sharpe ratios against `master_table.csv` (1e-9); the new Table 8 rows
  are gated against `master_table_vs_headline.csv`. The ladder's claim is updated and
  asserted: the only step whose interval excludes zero is the tuned LightGBM, below the
  headline; the three path steps sum to the headline's lead over R (asserted to 1e-9).
- **Why.** Checklist A2b finished tonight; the trees paragraph must say whether tuning
  rescues the trees (it does not).
- **Numbers.** Tuned vs the per-bar ridge on the same inputs, 16:00 QLIKE: 0 of 18
  intervals below zero, 6 above; closest LightGBM `all_features` +0.7 % [−5.4, +7.4].
  Tuned vs untuned (MSE rule): 0 of 9 below zero, 4 above; LightGBM `live_feasible`
  +7.3 % [+2.1, +12.6] (QLIKE rule: 0 below, 2 above). Trade: tuned LightGBM
  `live_feasible` 1.14 vs the ridge 1.90, −0.76 [−1.26, −0.24]; best tuned tree
  (LightGBM `all_features`, 1.81) +0.15 [−0.60, +0.90] vs its ridge, −0.09
  [−0.85, +0.63] vs the headline.
- **Depends on** change 5 (Table 8).
- **Build.** 70 pages, 12 overfull boxes, 0 undefined references.

### Change 7 — (d) Where the P&L comes from: a tail trade (+ Appendix D.14)

- **What.** Paragraph "Where the P&L comes from: a tail trade" in Section 5.4, and a new
  subsection D.14 "Where the Headline's P&L Comes From" (`sections/appendix_close_pnl.tex`,
  input from `main.tex` right after `appendix_running`, so it continues Appendix D) with
  Table 33 (`generated/table_close_pnl.tex`: the headline, always short and their
  difference by cell — all days, month-end T / T−1 / T+1, more than one session from a
  month-end, FOMC days, VIX terciles, years). Script: `pnl_macros()` and
  `write_pnl_table()` read `results/close_pnl_decomp/research_scorer/*.csv` — the B1
  decomposition re-run under the **same** 16:00-bar recalibration as Table 7 (gate: its
  headline Sharpe mid / crossed and buy count equal Table 7's row). The notebook-scorer
  version (`results/close_pnl_decomp/SUMMARY.md`, e.g. best 20 days = 73 %,
  difference +90.4 [−1.5, +184.6]) is deliberately **not** quoted: it is the other scorer.
- **Why.** The three facts that change the story's reading: the edge is a tail trade;
  month-ends are not its source; the difference to always short and its interval.
- **Numbers (16:00-bar recalibration, midpoint).** Total +116.2 premium units (87.4
  crossed); best 10 days 42 %, best 20 days 68 % (86 % crossed), all 20 buys, all 20
  worst sells; without the best 20 +37.6 (Sharpe 0.76); 40 days (4.6 %) make the whole
  total; bought 14 of the straddle's 20 biggest payoffs vs 8.0 expected (p 0.006);
  month-ends (52) −6.0 [−26.1, +12.7] with always short −24.5 [−43.5, −7.3] there;
  735 days away from month-ends +116.9 [+59.8, +174.1]; difference to always short
  +103.7 [+10.7, +200.1] (crossed +104.9 [+12.1, +201.8]), without its 10 best days
  +6.0 [−59.8, +72.7]; high-VIX tercile difference +2.3 [−39.0, +48.3]. Each claim
  asserted in the script.
- **Depends on** change 4 (headline).
- **Build.** 72 pages (+2), 12 overfull boxes (the new table is resized to the text
  width), 0 undefined references.

### Change 8 — (e) Strategy variations: one paragraph, pointing to Appendix D.13

- **What.** Paragraph "Other structures" in Section 5.4 (before the boundary paragraph),
  pointing to `sec:app_strategy_variations` (W10's table, hooked into Appendix D by W7).
  Script: `sv_macros()` reads `results/strategy_variations/{headline_per_straddle_premium,
  paired_vs_straddle,cost_line}.csv`, rows of the headline rule (per-bar ridge
  `live_feasible`, research scorer), per straddle premium; gate: the straddle row equals
  Table 7's headline row (mid to 1e-9; crossed to 1e-6 — F1 prices the crossed fill leg
  by leg, the master table from summed quotes; they differ by 7e-9).
- **Why.** The professor asked for the variations (strangles, butterflies, iron condors,
  …); the answer belongs in the main text as one paragraph: none beats the straddle,
  and wings cost at every width once the spread is paid.
- **Numbers.** 31 alternatives; 0 paired intervals above zero at either fill (asserted);
  closest at mid iron butterfly w50 1.94, +0.04 [−0.04, +0.14]; at crossed fills the
  straddle ranks first (asserted: every crossed difference negative); wing rows (iron
  butterfly + wing hedge) below the straddle with interval excluding zero 6 of 6 crossed,
  2 of 6 mid; median entry cost of crossing 2.9 % of the straddle's premium vs 4.0–6.1 %
  of an iron butterfly's.
- **Depends on** change 4 (headline).
- **Build.** 72 pages, 12 overfull boxes, 0 undefined references.

### Change 9 — (f) Park the session-bar diagnostics into Appendix D.15; closing summary re-read

- **What.** Everything in Section 5.4 after the one-scorer block that was computed under
  the session-bar recalibration moves, **verbatim**, to a new file
  `sections/appendix_running_parked.tex` (input from `main.tex` after
  `appendix_close_pnl`, so it is Appendix D.15 "Parked from Section 5.4: The Paper's
  Forecast under the Session-Bar Recalibration", with a dated "why it is parked"
  paragraph): the two sample splits, settlement pins, "eight rows are one or two
  bets", the threshold decomposition of QLIKE + Table (Murphy scores) + Murphy figure,
  "what a variance forecast can reach", "neither score orders the forecasts", the
  same-day regression + its figure, "magnitude carries information", the shift
  (dating) test, compounding at 3 % of wealth, and the already-parked `\iffalse`
  vol-target / iron-fly blocks that sat among them. Only five cross-references change
  in the moved text (they pointed to the deck-scorer table, now parked: they now name
  "the eight columns", Section 5.4, or the earlier design panel). In Section 5.4 the
  boundary paragraph becomes a pointer paragraph ("Diagnostics parked in Appendix D",
  listing what moved and pointing to D.11 for sizing), and the closing summary is
  rewritten on the one-scorer results; the session-bar closing summary is kept
  commented (dated) below it.
- **Why.** One scorer throughout the close-option results; the moved paragraphs are
  about the paper's forecast under the other recalibration and cannot sit beside
  Table 7. They are parked, not deleted, because three of them (the shift test, what a
  variance forecast can reach, why QLIKE and the trade rank differently) answer
  questions a reader will ask; they have **not** been re-run under the 16:00-bar
  recalibration (proposal for the user, see the report).
- **Before → after (closing summary).** Before (session-bar): "implied variance above
  the forecast on 59 to 68 % of days (60 for the headline) … improves on always short in
  every one of the eight columns and survives a crossed spread … only one column's
  improvement clears zero, by a knife edge … sizing does not improve on the sign".
  After (16:00-bar, macros): headline 1.90 (1.44 crossed), +0.90 [+0.19, +1.66] vs the
  paper's forecast, +1.70 [+0.09, +3.11] vs always short; accuracy gain not significant
  (DM −1.42, p 0.156); no single step resolved alone, the largest fitting the traded bar
  alone; not separated from the lasso (−0.08 [−0.89, +0.79]); tail trade (20 best days
  68 %, all buys; +6.0 [−59.8, +72.7] vs always short without the 10 best days); sample
  ends April 2024, no edge after it (Appendix D.5).
- **Depends on** changes 3–8.
- **Build.** 73 pages (+1: the parked text is now in the appendix with its own header),
  12 overfull boxes, 0 undefined references.

### Change 10 — Methods: stale session-bar statements re-pointed (one scorer)

- **What.** `methods_close_option.tex`: (i) after the session-bar map's description, one
  sentence saying where that map is still used (the parked diagnostics, D.15) and that
  every table of Section 5.4 uses the 16:00-bar recalibration; (ii) "the 413 days after
  April 2024 … are not scored anywhere below" → not scored in Section 5.4, with a pointer
  to Appendix D.5, where the trade's live form on them **is** reported (the old sentence
  had become false); (iii) the compounding paragraph says the exercise ran under the
  session-bar map and is in D.15; (iv) the sell shares re-read under the 16:00-bar
  recalibration (macros `\cmSellPaperMin/Max`, `\cmSellHead` from `master_table.csv`
  `pct_buy`); (v) "scaling … raised no portfolio's Sharpe ratio" corrected to "… with an
  interval above zero", with the session-bar qualifier and a pointer to D.11 — Appendix
  D.11 reports the rank rule *tying* sign(s) with some point estimates above it, so the
  old wording over-stated the negative result.
- **Why.** The methods carried results computed under the other scorer and two
  statements the new appendix made false or too strong.
- **Before → after.** Sell share, paper's eight: 59–68 % (session-bar) → 62–73 %
  (16:00-bar); headline: 60 % (block-diagonal ridge, session-bar) → 60 % (per-bar ridge,
  16:00-bar). Sources: `master_table.csv` (`pct_buy`); `appendix_running.tex` D.11 /
  `results/atm_straddle_0dte_1530/vrp_sized_rules.csv`.
- **Build.** 73 pages, 12 overfull boxes (a mid-paragraph `\input` that first produced a
  164 pt box was moved outside the paragraph), 0 undefined references.

### Change 11 — (g) Introduction, conclusion (and a proposed abstract sentence) reconciled

- **What.** Introduction, end of the fourth finding: the close-option promise rewritten on
  Section 5.4 as it now stands (macros); the old passage kept commented and dated.
  Roadmap: one clause on Appendix D (side analyses, P&L by cell, parked diagnostics).
  Conclusion: the close-option paragraph rewritten (no numbers); old one kept commented.
  Abstract: **not changed in the rendered paper** — the authors held the option reading
  out of the abstract on 2026-08-18; a proposed sentence sits in `main.tex` inside
  `\iffalse` ("PROPOSED 2026-09-29"), to render by flipping it to `\iftrue`.
- **Why.** The front matter promised the session-bar reading (eight columns all improving
  on always short, "exactly one resolved, marginally", block-diagonal ridge as headline).
  No over-claiming: the new text says the accuracy gain is not significant (p 0.156),
  the trade gain's interval excludes zero vs the paper's forecast and vs always short, no
  single step is resolved alone, the headline is not separated from the lasso and was
  chosen for feasibility "not as the best of the table", the edge is a tail trade, and the
  trade's live form has shown no edge after April 2024.
- **Before → after.** Intro: "disagree on 32 to 41 % of days … improves on always short for
  every one of the eight … exactly one resolved at 95 %, marginally … headline column is
  the block-diagonal ridge" → "every one of the paper's forecasts improves … in point
  estimate, one of the eight with an interval above zero; headline per-bar ridge on the 16
  live-feasible inputs: 1.90 (1.44 crossed), +0.90 [+0.19, +1.66] vs the paper's forecast,
  +1.70 [+0.09, +3.11] vs always short; not significantly more accurate (p = 0.156); 20 best
  days carry 68 % of its P&L; no edge after April 2024".
- **Depends on** changes 4–9 (the macros and the appendix sections it points to).
- **Build.** 74 pages (+1), 12 overfull boxes, 0 undefined references.

### Change 12 — Related works: an error corrected (the forecast adds direction, not size)

- **What.** `related_works.tex`, "The clock of the variance risk premium": the sentence
  "what the forecast adds is size on that short rather than a directional flip" is
  replaced by what Section 5.4 shows: under the headline forecast implied variance exceeds
  the forecast on 60 % of days (so the reading is mostly a short), sizing on the gap does
  not improve on its sign (Appendix D.11), and the rule differs from always short only on
  the days it buys, which are worth +103.7 [+10.7, +200.1] premium units to it.
- **Why.** The old sentence contradicted the paper's own results (only the sign is used;
  sizing does not help) — an error, fixed as its own change with the evidence.
- **Evidence.** `master_table.csv` (`pct_buy` 40.2 → sells 60 %);
  `results/close_pnl_decomp/research_scorer/diff_attribution.csv` ("buy days" = "all
  days" = +103.7 [+10.7, +200.1], asserted); Appendix D.11 /
  `results/atm_straddle_0dte_1530/vrp_sized_rules.csv` (no sized rule with an interval
  above zero).
- **Build.** 74 pages, 12 overfull boxes, 0 undefined references.

### Change 13 — Polish: small counts as words; the three path steps named

- **What.** `make_table_close_main.py` writes four prose counts as words ("none", "one")
  instead of digits; `results_close_option.tex` names the three steps of Table 8 whose sum
  is the headline's lead (they are rows 1, 3 and 5 of the table, not "the first three",
  as change 5/6 said — a wording error), and two sentences are re-worded for the words.
- **Why.** Readability, and one factual slip in my own earlier text (the path steps).
- **Numbers.** None change.
- **Build.** 74 pages, 12 overfull boxes, 0 undefined references.
