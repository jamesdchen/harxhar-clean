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

### Change 14 — Re-runnability: Table 8 falls back to its committed pairs

- **What.** `make_table_close_main.py` `ladder()`: when
  `results/close_master_table/master_table_daily.parquet` (3.4 MB, written by
  `experiments/master_table_close.py`, **not committed**) is absent, Table 8 is rebuilt from
  the committed `writeup/generated/close_main_ladder.csv` (written and gated by the last
  run that had the frame). Generated files unchanged.
- **Why.** So the script runs on a clean checkout; the recomputation and its gates run
  whenever the frame is present.

---

## 3. The story as it now reads (three sentences)

Under one recalibration and on the same 866 days, every forecast improves on selling the
straddle every day in point estimate, and the headline — a ridge fitted for the 15:30–16:00
bar alone on the 16 inputs a forecaster can rebuild at 15:30, chosen for that feasibility and
not as the table's maximum — is resolved above both always short (+1.70 [+0.09, +3.11]) and
the paper's own block-diagonal ridge (+0.90 [+0.19, +1.66]), at Sharpe 1.90 mid / 1.44
crossed. The accuracy gain behind it is not significant (QLIKE −5.0 %, DM −1.42, p 0.156), no
single change on the way from the paper's forecast is resolved alone (the largest is fitting
the traded bar on its own), and it is not separated from the lasso on the same inputs or
from any tree, tuned or untuned. The edge is a tail trade — 68 % of the P&L on 20 buy days,
month-ends not the source, +6.0 [−59.8, +72.7] over always short without its 10 best days —
no alternative structure beats the straddle, wings cost at every width once the spread is
paid, and the trade's live form has shown no edge after April 2024.

## 4. Commits (each revertible; later ones depend on earlier ones as marked above)

| # | commit | change |
|---|---|---|
| 0 | `0e8fb86` | this changelog, section 1 (the map) |
| 1 | `4b4afbe` | methods: per-bar forecasts and the 16:00-bar recalibration defined |
| 2 | `d8b04be` | wording: straddle defined once, "package" retired |
| 3 | `c8a9dd3` | (a) one scorer: Table 7 from the master table; deck table parked |
| 4 | `80af1ff` | (b) headline = per-bar ridge `live_feasible`, and why |
| 5 | `aee75a8` | one change at a time: Table 8 |
| 6 | `38c6682` | (c) causally tuned trees |
| 7 | `fa30b25` | (d) P&L decomposition: a tail trade; Appendix D.14 |
| 8 | `412e3a5` | (e) strategy variations paragraph → Appendix D.13 |
| 9 | `728f15e` | (f) session-bar diagnostics parked → Appendix D.15; closing summary |
| 10 | `2a2b56c` | methods: stale session-bar statements re-pointed |
| 11 | `a926fcd` | (g) introduction / conclusion reconciled; abstract sentence proposed (not rendered) |
| 12 | `daf4b31` | related works: "size, not direction" corrected |
| 13 | `64d9c83` | polish: counts as words; path steps named |
| 14 | this commit | Table 8 re-runnable without the uncommitted daily frame |

## 5. Not changed — for the user to decide

1. **Abstract.** A proposed sentence is in `main.tex` inside `\iffalse` (the authors held the
   option reading out of the abstract on 2026-08-18). Flip to `\iftrue` to render.
2. **Thesis vs headline.** The paper's thesis model is the block-diagonal ridge; the trade's
   headline is a per-bar ridge. The title/abstract do not say the paper has a second
   contribution; whether Section 5.4 stays a "reading" of the paper's forecast or becomes a
   result of its own is an authors' call.
3. **The size of the lead depends on the recalibration.** The 16:00-bar recalibration has no
   level map, so a 48-bar forecast keeps its 16:00 level bias (the block-diagonal ridge reads
   1.00 here, 1.34 under the session-bar map); Table 8's first step (−0.01) and the pooled
   twins suggest the bias is not the whole story, but a robustness row with a 16:00-bar
   Mincer–Zarnowitz map for every forecast (a third recalibration, in the master table) would
   settle it. Not run (new analysis).
4. **Parked diagnostics not re-run.** The shift (dating) test, the threshold/Murphy analysis,
   "what a variance forecast can reach", the same-day regression, pins, sample splits and
   compounding (Appendix D.15) are on the block-diagonal ridge under the session-bar map. The
   dating test at least should be re-run for the headline before submission.
5. **Appendix D conventions (W7's file, not edited).** Its conventions paragraph names the
   block-diagonal ridge and points to Table 7 for the days and the √252 convention (both still
   true) but does not name its recalibration; most of D.1–D.12 use the session-bar map.
   Proposed sentence for `sections/appendix_running.tex`: "Unless a subsection says
   otherwise, a recalibrated forecast in this appendix uses the session-bar recalibration of
   Section 4.5, and no number here is set beside Table 7."
6. **Kelly-type sizing (A3)** is not in the paper (session-bar scorer, `results/close_kelly/`);
   D.11 covers proportional and rank sizing only.
7. **Pre-existing:** multiply-defined label `tab:app_strategy_variations_res_short` in
   `generated/strategy_variations.tex` (W10's generator); 12 overfull boxes outside Section 5.4.

## 6. Re-running

After the master table is re-run (H2, with the LSTM rows):
`python writeup/make_table_close_main.py` (needs `master_table_daily.parquet` for Table 8's
recomputation; without it Table 8 is read from its committed pairs), then pdflatex → bibtex →
pdflatex ×2. Every qualitative claim in the new prose is asserted in the script: if a claim
flips (e.g. a new forecast trades above the headline with an interval above zero), the script
stops instead of writing numbers that contradict the text.

---

## 7. 2026-09-30 (I4b, agent L1): Section 5.4 after the 16:00 campaign

Input: the master table rebuilt on the de-duplicated per-bar design and the campaign's tables
(commit `e4b83a0`, `results/close_master_table/`: 220 table-A forecasts in 20 families + 9 check
rows; headline per-bar ridge `live_feasible` unchanged, QLIKE 0.1005, Sharpe 1.90 / 1.44; none of
219 trades above it). Campaign plan: `writeup/CAMPAIGN_16H_2026-09-29.md`. One commit (below);
files: `writeup/make_table_close_main.py`, `writeup/generated/{table_close_main,
close_main_numbers,table_close_ladder}.tex`, `writeup/generated/close_main_ladder.csv`,
`writeup/sections/{results,methods}_close_option.tex`, this changelog.
`generated/table_close_pnl.tex` is rewritten by the same run and is byte-identical.

### 7.1 Script (`make_table_close_main.py`)

- **Table 7 row selection.** Panel D was the six untuned per-bar trees (three trees ×
  `live_feasible` / `all_features`). It is now **one row per tree / LSTM family** of the
  campaign, all on `live_feasible` and LightGBM for the trees, so consecutive rows differ in
  one thing: T10 (`subtree_lgbm_live_feasible`), T1 (`subtree_daily_…`), RS10
  (`subtree_tuned_…`), RS1 (`subtree_tuned_daily_…`), Optuna TUNE_PER = 1 best-of-50
  (`subtree_optuna_tp1_…`), per-bar LSTM refit daily (`lstm_live_feasible`), intraday LSTM
  (`lstm_intraday_live_feasible`). Asserted: each key's master-table family; one row per
  family; each family's provenance (`master_table_provenance.csv`) says per-window mask and
  CPU class epyc-7513; every tree / LSTM family of table A has its row except the Optuna
  ablations (other TUNE_PER, best of 10 / 25), which Table 8 and the prose read. Panels A–C
  unchanged (paper's eight, pooled twins, per-bar linear). Table 7: 27 → 28 forecasts.
- **Table 8: five campaign blocks, 12 rows** (26 steps in all), LightGBM / LSTM on
  `live_feasible`: design de-dup (headline, from `before_after.csv`); per-window mask,
  masked − unmasked for T10, T1, RS10, RS1 and the LSTM refit every 10 (from agent H's
  `results/trees_mask_1600/mask_pairs.csv`; the masked levels are gated against the master
  table); refit cadence T1 − T10 (recomputed) and LSTM daily − every 10 (H's CSV); tuning
  RS1 − T1, Optuna tp1 − T1, Optuna tp1 − tp250 (recomputed); intraday − per-bar LSTM
  (recomputed). **Gate:** every recomputed campaign pair (QLIKE difference, DM, ΔSharpe and
  its interval) reproduces the campaign CSV that first reported it (H `mask_pairs.csv`,
  C `report/tuneper_oos.csv`, I `score/lstmi_pairs.csv`) to 5e-6 (their six-digit precision),
  i.e. the same days, scorer and resampled days; the 14 earlier steps still reproduce the
  master table's stored pairs to 1e-9.
- **`tuned_macros` (A2b) replaced by `campaign_macros`.** The master table's `subtree_tuned_*`
  rows are now H's masked, single-class RS10 re-run, so the first-pass gate "a2b's tuned-tree
  Sharpe = the master table's" no longer holds (e.g. LightGBM `live_feasible` 1.14 → 1.47);
  the trees paragraph now reads the campaign's CSVs (list in the script's docstring). New
  macros: design counts (agent A `results/linear_subsection_dedup/gates.csv`), kept columns
  (H `kept_counts.csv`), CPU class (H `results/linear_subsection_trees_mask/class/`, C
  `xclass*/xclass_gate.csv`, `chunk_hosts.csv`), all counts of the trees / LSTM paragraph.
  Every claim asserted (see 7.2).

### 7.2 Assertions that fired, and what the prose now says (before → after)

| # | assertion (script) | before | after | source | prose change |
|---|---|---|---|---|---|
| 1 | `ladder_macros`: lowest estimator step from the headline is the elastic net | elastic net −0.49 [−1.08, +0.14]; untuned XGBoost −0.21 [−0.82, +0.43] | untuned XGBoost (T10, masked, one CPU class) −0.63 [−1.51, +0.16]; elastic net unchanged −0.49 | `close_main_ladder.csv` (= `master_table_vs_headline.csv`) | "between −0.08 (the lasso) and −0.49 (the elastic net)" → "… −0.63 (XGBoost)"; "untuned tree" → "tree with the shipped settings". Assertion now: argmin = `subtree_xgb_live_feasible`. |
| 2 | `ladder_macros`: the only step with a Sharpe interval excluding zero is the tuned LightGBM, below the headline | tuned LightGBM `live_feasible` − headline −0.76 [−1.26, −0.24] (first pass, old design, unmasked) | RS10 LightGBM −0.43 [−1.22, +0.35]; no step of the 26 excludes zero | `close_main_ladder.csv` | "Of the 14 steps, one has a Sharpe interval that excludes zero: the causally tuned LightGBM, below the headline" → "Of all 26 steps of the table, none has a Sharpe interval that excludes zero." Assertion now: `len(res) == 0`. |
| 3 | `tuned_macros`: a2b tuned-tree Sharpe = master table (gate) | a2b = master table | master table = H's masked RS10 (LightGBM `live_feasible` 1.47, was 1.14) | `master_table.csv` vs `results/linear_subsection_trees_tuned/a2b/trade_1600_paired.csv` | Paragraph "Trees, untuned and causally tuned" replaced by "Trees and LSTMs, every rung" (7.3). |

Claims that still hold on the new tables and stay asserted: headline accuracy gain not
significant (DM −1.42, p 0.156); headline Sharpe intervals vs R and vs always short above
zero; lasso / elastic net / ridge `all_features` / top-Sharpe forecast not separated from the
headline; none of 219 trades above it; the three path steps sum to its +0.90 lead; the largest
step is fitting the 16:00 bar alone; per-bar beats pooled on QLIKE on `baseline` (−11.2 %
[−15.7, −6.9]) but not on the trade; estimator changes all cost (lasso −0.08 the smallest).

Numbers that moved with no assertion firing (macros; prose unchanged): forecasts in the master
table 106 → 220 (other 105 → 219); vs R intervals above zero 9 → 10 (master table), 3 of 26 → 2
of 27 (Table 7); headline QLIKE rank 55 → 63 of 220 (Sharpe rank 2 unchanged); top-Sharpe
forecast per-bar lasso `free_vix_only` → `live_vix_only` (1.91 both), vs headline +0.01
[−0.75, +0.79] → +0.01 [−0.76, +0.81]; trading below the headline with an interval below zero
18 of 105 → 43 of 219; panel D Sharpe range 1.26–1.70 (six untuned trees) → 0.87–1.63 (seven
rungs). Headline, paper's eight, pooled twins, P&L decomposition, strategy variations:
unchanged.

### 7.3 Prose added or rewritten (all numbers `\cm…` macros)

- **Results, new paragraph "The per-bar tables, rebuilt"** (before "The forecasts built for the
  traded bar"): design 34 / 244 / 640 → 22 / 232 / 628 columns (12 session-edge columns zero or
  copies of `har_ma_k` in a one-bar-per-session series); per-window mask on every tree / LSTM
  fit, median kept 17 / 22, 138 / 232, 369 / 628; daily refits; Optuna beside the random search;
  CPU class: across classes LightGBM's best-of-50 pick changed on 18 of 20 tuning points and a
  LightGBM forecast by up to 5.5 %, on the same class 20 of 20 points bit-identical for all three
  trees, all 541 chunks behind the tree and per-bar LSTM tables on one class; linear forecasts:
  headline positions unchanged and QLIKE changed only at rounding (6e-12 %), 55 per-bar linear
  forecasts ≤ 0.09 % QLIKE, ≤ two positions, no Sharpe interval excluding zero. Points to
  Appendix `sec:app_campaign_16h` (L4's file; wrapped in `\IfFileExists`).
- **Results, "One change at a time"**: a second paragraph reads the campaign blocks of Table 8:
  mask on LightGBM T10 +1.0 [−0.4, +2.3] % QLIKE, −0.17 [−0.80, +0.39] Sharpe; daily refits
  LightGBM −3.6 [−7.4, −0.8] % (DM −2.73), trade −0.17 [−1.11, +0.59]; LSTM −2.5 [−7.8, +3.2] %;
  random search vs shipped +4.1 [−0.2, +8.9] %; Optuna tuned daily vs shipped +7.4 [+1.4, +16.3] %
  (resolved loss); tuning every session vs every 250 −0.55 [−1.42, +0.30] Sharpe; intraday vs
  per-bar LSTM −7.5 [−20.9, +4.7] % QLIKE, +0.62 [−0.26, +1.54] Sharpe. Asserted: among the
  campaign steps only the daily-refit and Optuna-vs-shipped QLIKE intervals exclude zero.
- **Results, "Trees and LSTMs, every rung"** (replaces "Trees, untuned and causally tuned";
  before: 0 of 18 tuned trees below the ridge on QLIKE, 6 above, LightGBM `all_features` +0.7 %
  [−5.4, +7.4]; tuned vs untuned 0 / 9 below, 4 above, LightGBM `live_feasible` +7.3 % [+2.1,
  +12.6]; tuned LightGBM `live_feasible` 1.14 vs ridge 1.90, −0.76 [−1.26, −0.24]; best tuned
  tree LightGBM `all_features` 1.81, +0.15 [−0.60, +0.90] vs its ridge, −0.09 [−0.85, +0.63] vs
  the headline). After: trees vs the per-bar ridge 0 of 72 QLIKE intervals below zero, 32 above,
  Sharpe 0 above, two below (H `masked - ridge` + C `vs ridge`); T1 − T10 lowers QLIKE for 9 / 9,
  three below zero, one trade resolved (XGBoost `all_features` +0.52 [+0.10, +1.02]); RS1 − T1
  6 / 9 above zero; Optuna − T1 19 / 36 above, 0 below; no tuned trade resolved; TUNE_PER 1/5/25
  vs 250: 8 / 27 below, trade never resolved; best of 50 vs 10 / 25: 0 / 36 below, one above,
  while the best validation trial falls in the last ten at 27–40 % of points (20 % if
  exchangeable); mask: median |ΔQLIKE| 0.7 %, 5 / 36 above, 0 below, one Sharpe below; LSTM daily
  − every 10: −1.9 / −2.5 / −10.1 %, two below zero; per-bar LSTM vs ridge +6.8 … +30.0 %;
  intraday vs per-bar LSTM resolved only on `all_features` −14.3 [−26.9, −2.0] %, vs ridge +9.2 …
  +11.5 %, all above zero; no LSTM trade resolved vs ridge; best tree / LSTM on the trade
  (random-search RF `live_feasible`, QLIKE rule, refit every 10) 1.90, −0.00 [−0.56, +0.59] vs
  the headline; none of 147 above the headline.
- **Results, closing summary**: one sentence added — none of the 147 tree and LSTM forecasts
  trades above the headline with an interval above zero.
- **Table 7 caption**: panel D described; **Table 8 caption**: the campaign blocks, which rows
  are read vs recomputed, "DM ---: not stored".
- **Methods, "Forecasts fitted for the 16:00 bar alone"** (only where a fact changed): the
  session-edge columns left out of the per-bar design (kept in the pooled twins); trees refit
  every ten sessions *or every session* (was: every ten); tuning = the random search (unchanged
  definition) *and* Optuna (50 TPE trials, first trial the shipped settings, TUNE_PER 1 / 5 / 25
  / 250, best of 10 / 25 / 50, QLIKE-rule twins); the two LSTMs defined (sessions L ∈ {5, 20};
  half-hour bars N ∈ {13, 48, 96} with bar-level inputs; shared grid, 250-session tuning, five
  seeds, daily refits); the per-window mask for every per-bar fit; one processor model for every
  tree and LSTM table. Constants from `specs/causal_tune_trees*.py`, `specs/causal_tune_lstm*.py`.

### 7.4 Build and commit

Build (pdflatex → bibtex → pdflatex ×2, with L4's `sections/appendix_campaign_16h.tex` present):
99 pages, 0 undefined references, 0 multiply-defined labels, 1 overfull box (a 1.6 pt `\vbox`,
not in Section 5.4). Commit: this commit (`I4b`).

### 7.5 Not changed — for the user

1. Panel D shows LightGBM for the trees; XGBoost and the random forest of every rung are only
   counted in the prose (the master table has all 147 rows).
2. The design row of Table 8 shows the headline only; the other 54 per-bar linear forecasts are
   summarised in the prose (max 0.09 % QLIKE, ≤ two positions).
3. The introduction and conclusion (not this agent's files) make no tree-specific claim and
   need no change; their macros (`\cmH…`, `\cmPnl…`, `\cmNdays`, `\cmSellHead`) are unchanged.

## 8. 2026-10-06: the closing study without recalibration (text only)

**Decision (user):** "get rid of recalibration; it's too complicated to keep in the paper and it's
not the main result." Also: no intervals anywhere (the bootstrap is off), point estimates only;
DM and HAC t statistics may stay. Later rule from the user, same day: "remove adj_ in the names of
any variable being presented."

**New convention for the closing study:** every forecast is used as its plain back-transform
f^2 x B (f = the forecast of sqrt(RV/B) at the 16:00 bar, B = the time-of-day baseline),
`eq:close_opt_plain`. The 15:30 sign(s) trade buys one straddle when f^2 x B is above the 15:30
implied variance and sells one otherwise. QLIKE and every other loss use the same forecast.

**Scope of this pass:** only `.tex` files under `writeup/sections/`: Section 5.4 (methods and
results), Appendices D.14, D.16 to D.19, and the closing-study sentences in the introduction,
conclusion, related works and data preparation. No number was typed. Old text is kept as a comment
under `% 2026-10-06 parked (recalibration removed, user decision):` (for the adj_ rule, the header
says `(presentation rule, user decision: no adj_ prefix in variable names)`). Wherever a number or
comparison has to be checked against the regenerated macros or tables, a `% TODO-PLAIN:` line sits
directly above it. All edited files have LF endings. `appendix_close_murphy_splits.tex` was CRLF in
the working tree; it was converted to LF, and the index was already LF, so this adds no diff.

### 8.1 Inventory before editing (11 in-scope files)

The grep pass found 267 lines with hits: 236 in prose and 31 in comments. Hits by pattern:

| Pattern | Hits |
|---|---|
| `recalibrat*` | 45 |
| `Mincer` | 4 |
| `smear` | 10 |
| `(f^2` / `\hat y_k^2` | 2 |
| `250 sessions` | 11 |
| `95` | 14 |
| `interval` | 90 |
| `excludes zero` | 14 |
| `wholly above/below` | 8 |
| `resolv*` | 21 |
| `bootstrap` | 10 |

Interval macros in prose:

- **Pure interval-bound macros** (`\cm...Ci`, `\camp...CI`, the shift test's Lo/Hi, `\hmurRho...Lo/Hi`, `\hspl...Lo/Hi`): 30 references.
- **Macros whose generated value carries `[lo, hi]`** (for example `\cmHVsRefMid`, `\cmLd...`, `\cmPnlDiff`): 53 references.

The full per-line list is in the scratch file `inventory_before.txt`; the main hits are quoted in the edits below.

Hits that were not about recalibration or intervals, and were left alone:

- the `250 sessions` tuning cadences;
- "5th to 95th percentile" in the implied-volatility check;
- `k_{95}`;
- "the tie ... resolved short";
- the campaign appendix's `...Lo/...Hi` range macros (for example `\campTOneTenQLo`), which are minimums and maximums, not intervals;
- `\hmurLeaderAtOne` (its "[all_features]" is a label);
- the main results' smearing protocol in the introduction, related works and data preparation (out of scope, see 8.5).

### 8.2 Edits by file

Line numbers are those of the current file. Each edit starts with its parked block.

**`sections/methods_close_option.tex`**

- **:68, "Forecast remaining variance".** Parked: the 15:30 row mapped by the session-bar Mincer–Zarnowitz map, (m^2 + sigma^2) B, its three differences from the protocol, and "session-bar recalibration … 16:00-bar recalibration defined below". New text: the 15:30 row is turned into a variance by the plain back-transform of `eq:close_opt_plain`, the same for every forecast. One sentence says that the Section 4 retransformation protocol (`sec:smearing_contract`) is not used here and that no number is carried between the two.
- **:107.** Before: "Two conventions matter for honesty", covering Alignment, Weighting, and the regular-session mask (69,298 / 1,083 bars). After: "One convention matters for honesty", Alignment only. The weighting and the mask belonged to the map.
- **:139, B contamination.** Kept: early closes depress B for 20 sessions, and 94 days are affected. Parked: the cancellation argument written in terms of (m^2 + sigma^2) B. New text: the audit bound (under a tenth of a Sharpe ratio, a handful of flips) was measured under an earlier back-transform and has not been repeated with f^2 B.
- **:168.** Parked: "The weighting addresses a level effect …".
- **:239, "What the 16:00 target is".** Before: "the fit mask … keeps [those sessions] out of both the calibration and the trade". After: "the sessions on which they are not aligned are not traded: a cash-market holiday has no expiration, and an early close is dropped by the filter of the instrument paragraph".
- **:267, adj_ rule.** Before: `\texttt{adj\_}$x$\texttt{\_ma\_}$k$ (… after the diurnal adjustment)`. After: `$x$\texttt{\_ma\_}$k$`, plus: "Every exogenous input is divided by its time-of-day baseline (Section~\ref{sec:dp_prep}), and the HAR columns are means of the adjusted target." This was the only `adj_` name in the in-scope files.
- **:344, new paragraph "The variance forecast".** Adds `eq:close_opt_plain`, RV_t = f_t^2 B_t. Nothing is estimated on top of it: no level map and no second-moment term. The same forecast feeds the rule and every loss. A one-line Jensen note says that squaring tends to forecast RV low, which tilts the comparison toward selling, and that the tilt is left in. QLIKE uses the per-bar target, and forecasts are compared by DM (Newey–West, Bartlett lag 6, HLN correction). Sharpe ratios and their differences are point estimates, and no interval is computed. Parked at :373: "Two recalibrations", including `eq:close_opt_recal1600`, the 250-session window, and the 95 % circular block bootstrap.
- **:534, compounding.** Before: "run on the block-diagonal ridge under the session-bar recalibration … Appendix~\ref{sec:app_run_parked}". After: "reported for the headline forecast … in Appendix~\ref{sec:app_compounding_headline}".
- **:578, sell shares and sizing.** "under the 16:00-bar recalibration" becomes "with the plain forecast of Equation~\eqref{eq:close_opt_plain}".
  - Sizing, before: "under the session-bar recalibration … raised no portfolio's Sharpe ratio with an interval above zero".
  - Sizing, after: "in the notebook study of Appendix~\ref{sec:app_run_sizing} under that study's own back-transform … in point estimate, after a year of history, scaling by |s_t| lowered every forecast's Sharpe ratio and scaling by its rank left it about where it was". This restates the point estimates of Appendix D.11.

**`sections/results_close_option.tex`**

- **:8–11.** Added a dated header comment.
- **:25, opening.** "the 16:00-bar recalibration of eq:close_opt_recal1600" becomes "the plain back-transform RV_t = f_t^2 B_t of eq:close_opt_plain".
- **:43 and :56, Table 7 caption.** "recalibrated forecast" becomes "that forecast"; "with a 95% paired interval (circular block bootstrap …)" becomes "a point estimate".
- **:78, control.** Dropped `\cmShortShCiMid` and `\cmShortShCiX`. Before: "a premium that 866 days cannot separate from zero". After: "a small premium at the midpoint, and one the spread consumes". Always short does not depend on the forecast.
- **:90, the paper's forecasts.**
  - Dropped "with \cmPaperVsShortAboveN of the eight intervals above zero (…)".
  - The FOMC sentence, before: "trades higher by \cmFomcD, an interval that includes zero: … not resolved". After: "differs from it by \cmFomcD in Sharpe ratio".
  - Before: "Under this recalibration … keeps its own level". After: "With no level map … keeps whatever level it has".
  - Before: "One of them separates from it: the fixed lasso … above … by". After: "The fixed lasso … \cmLassoFVsRef against …".
- **:167.** Dropped "with no Sharpe difference resolved".
- **:188.** Before: "have a Sharpe-difference interval wholly above/below zero" and "its intervals are read as such". After: "trade above/below it" and "its differences are read as such".
- **:224, headline.**
  - Dropped: "with a day-block interval … \cmHQCi", "Its trade is resolved against both references" and "the second interval clears zero narrowly".
  - "What it is not separated from" becomes "How close its neighbours come".
  - The counts become point-estimate counts.
  - Before: "Table resolves that it trades above … does not resolve an order". After: "it trades above the paper's forecast and above the control, and differences of the size that separate it from the other per-bar forecasts are not read as an order among them".
- **:290, one change at a time.** "a resolved gain (DM …)" becomes "a gain at DM …". Before: "The headline's resolved lead … none of which is resolved alone; the largest …". After: "The headline's lead … ; the largest …".
- **:342, campaign rungs.** Before: "moves no rung's QLIKE or Sharpe ratio with an interval that excludes zero (LightGBM T10 …)". After: the LightGBM T10 change with its DM, pointing to Table 8 for the other masked rungs. "not resolved" becomes "(DM …)" and "a resolved loss of accuracy" becomes "(DM …), a loss of accuracy". Parked: "Of all \cmLdN steps, \cmLdResolved has a Sharpe interval that excludes zero".
- **:382 and :404, Table 8 caption.** "under the 16:00-bar recalibration", "with a 95% day-block interval" and "with 95% paired intervals" become "every forecast its plain back-transform f^2 B" and "a point estimate". "same scorer and resampling" becomes "same scorer".
- **:420, trees and LSTMs.** Every interval count is now a point-estimate count with the same macro.
  - Not restated: "No tree is more accurate than the per-bar ridge". In point estimate it is already false: Appendix D.19 has T1 LightGBM `live_feasible` at `\campRidgeBestTreeQ` = −5.0 % against the ridge.
  - Parked: "no tuned tree's trade differs … interval", "never resolved on the trade", "neither LSTM's trade differs … interval", "every interval above zero", and the budget clause "more accurate … in none of".
  - "resolved more accurate … only on all_features" becomes "more accurate … on all_features (\cmIntraAllQ)".
  - The XGBoost clause becomes "moves the trade of XGBoost on all_features by \cmCadShUp".
- **:510, P&L.** Dropped "— an interval that excludes zero, and that of the Sharpe difference in Table 7 does too —".
- **:568, other structures.** Parked: "None beats the straddle: \cmSvAbove of their paired Sharpe-difference intervals lies above zero at either fill". "\cmSvBestMid above it" becomes "\cmSvBestMid against it", because the point estimate is +0.04 for the iron butterfly. "with an interval that excludes zero at crossed fills" becomes "at crossed fills".
- **:608, "Diagnostics parked in Appendix D".** Before: "under the session-bar recalibration / the other recalibration / too, was run under the session-bar recalibration". After: "under a different back-transform, since removed … kept, verbatim … repeated for the headline forecast in Appendices D.16–D.18 … the notebook study, under its own back-transform".
- **:743, closing summary.**
  - "Under one recalibration" becomes "With every forecast used as its plain back-transform".
  - Before: "no single change … is resolved on its own, the largest being". After: "the largest change … is".
  - Before: "not separated from the lasso (\cmVsHLasso)". After: "the lasso on the same inputs trades \cmVsHLasso against it".
  - Before: "None of the 147 … trades above it with an interval above zero". After: "Of the 147 …, \cmNlVsHAbove trade above it".

**`sections/appendix_close_pnl.tex` (D.14)**

- **:9–10.** Added a header note.
- **:19.** Before: "under the 16:00-bar recalibration … Intervals are 95% circular block bootstrap intervals …". After: "its forecast used as the plain back-transform of Table 7 … every figure is a point estimate".
- **:47.** Added a TODO only.
- **:72.** Before: "the difference to always short and its interval". After: "the difference to always short".

**`sections/appendix_close_shift_test.tex` (D.16)**

- **:11–12.** Added a header note.
- **:26.** "(ŷ_k^2 + ŝ_16:00) B_16:00, with the smear ŝ and profile B" becomes "ŷ_k^2 B_16:00, with the profile B". "Scorer, trade, fills and intervals" becomes "Scorer, trade and fills".
- **:51.** Dropped the three `[\shift…Lo, \shift…Hi]` bounds and the interval counts (`\shiftNStaleIntervalBelow/Above`, `\shiftNNextIntervalAbove`). Dropped "in point estimate".

**`sections/appendix_close_murphy_splits.tex` (D.17)**

- **:11–12.** Added a header note.
- **:21.** "repeats that scoring under the 16:00-bar recalibration" becomes "repeats that scoring, every forecast used as its plain back-transform".
- **:61, Spearman.** Dropped the "day-block bootstrap 95 % interval" and every `\hmurRho…Lo/Hi`.
- **:83, splits.** Dropped "interval \hsplH…Lo to …Hi" three times and kept the t statistics.
- **:111, figure caption.** "under the 16:00-bar recalibration" becomes "every forecast used as its plain back-transform".

**`sections/appendix_close_compounding.tex` (D.18)**

- **:11–12.** Added a header note.
- **:23.** Before: "under the 16:00-bar recalibration … under the other recalibration". After: "its forecast used as the plain back-transform of Table 7 … under a different back-transform, since removed". The reference prefix "Section~" becomes "Appendix~".
- **:60, figure caption.** Same change as :23.

**`sections/appendix_campaign_16h.tex` (D.19)**

- **:12–14.** Added a header note.
- **:42, conventions.** Before: "the research scorer (the 16:00 bar recalibrated on its own)" with "95 % paired day-block bootstrap interval" and "with its interval". After: "every forecast as its plain back-transform f^2 B (Eq.)", every difference a point estimate, and every count a count of point estimates.
- **Interval counts rewritten as point-estimate counts with the same macros:**
  - :156 the mask, which also drops `\campMaskLSTMAllQCI`;
  - :197 cadence, which also drops `\campLSTMCad{Base,Live,All}CI`;
  - :222 against the ridge. Before: "none … has a QLIKE interval below zero". The new text also drops `\campRidgeBestTreeQCI`;
  - :259 random search against T;
  - :292 the out-of-sample budget;
  - :304 how often to tune. The list `\campTPBelowWho` is no longer quoted;
  - :322 Optuna against the other rungs. The list `\campOptRSBelowWho` is no longer quoted;
  - :427 the intraday LSTM;
  - :496 single class against mixed. `\campSMSharpeExcl` is replaced by `\campSMSBelow/\campSMSAbove`.
- **Captions** of the mask table (:179), the Optuna-vs table (:386), the repro table (:512) and the master table (:570) changed from "[95 % interval]", "whose 95 % interval lies wholly …", "with 95 % paired intervals" and "Sharpe-change intervals wholly above / below" to point-estimate wording.
- **:476, CPU class.** Dropped `\campClass{QTen,QOne,STen,SOne}CI`.
- **:525, master before and after.**
  - Dropped "with the same resampled days for every interval"; "recalibrated forecast" becomes "forecast".
  - No longer quoted: `\campMTSigN`/`\campMTSigWho` and `\campMTQBetterFamily`/`\campMTQBetterWho`, which are interval-only.
  - The counts become point-estimate counts: "trade above/below it" and "a lower/higher QLIKE".
- **:602, density.** Dropped `\campDvsLgbm…CI`.

**`sections/introduction.tex`**

- **:94, parked.** Before: "Scored under one recalibration … in point estimate, one of the eight with an interval above zero". After: "With every forecast used as its plain back-transform and scored on the same \cmNdays days, every one … improves".
- Before: "no single change … is resolved on its own, and it is not separated from the lasso". After: "and the lasso on the same inputs trades \cmVsHLasso against it".

**`sections/conclusion.tex`**

- **:42, parked.** "Under one recalibration" becomes "With every forecast used as its plain back-transform"; "in point estimate" is dropped; "is resolved above both" becomes "trades above both".
- Before: "is therefore not the one to trade". After: "need not be the one to trade". The claim now rests on point estimates only.
- Before: "no single step … is resolved alone (the largest is …)". After: "the largest step … is fitting the traded bar on its own".

**`sections/related_works.tex`**

- **:110.** Added a TODO only.

**`sections/data_prep.tex`**

- **:29.** Before: "excludes them by an explicit calendar". That calendar was the fit mask of the removed map. After: "does not trade them: a cash-market holiday has no expiration, and an early close is dropped by an explicit test".

### 8.3 TODO-PLAIN lines (48)

Each line names the macros or tables to re-read under f^2 B with point estimates only, and the
typed claims to re-check.

**methods_close_option.tex** (4)

- :158: the B-contamination audit bound
- :429: the 0.06 bound for dropping censored-IV days
- :592: `\cmSellPaperMin/Max`, `\cmSellHead`, and "leans short"
- :603: the sizing test not re-run with f^2 B

**results_close_option.tex** (20)

- :60: table_close_main point estimates only
- :111: all eight above the control
- :122: `\cmRefVsShort`, `\cmFomcD`
- :132: whether the fixed lasso is still the one to single out
- :174: `\cmDedupQMax/PosMax`
- :196: `\cmTabVsRef…` and `\cmMasterVsRefAbove` as point counts
- :215: the headline is not the maximum, and its ranks
- :249: "more accurate … not significantly" (DM, p)
- :257: `\cmHVsRef…`, `\cmHVsShort…` and their signs
- :264: `\cmVsH…` and the lasso's QLIKE
- :273: `\cmVsHAbove/Below` as point counts
- :309: the `\cmLd…` ladder claims ("moves nothing", "largest step", HAR gain)
- :327: the headline still leads, and the largest of the three steps
- :356: the `\cmLdC…` claims; `\cmLdResolved` has no point counterpart
- :390: table_close_ladder point estimates only
- :459: the trees and LSTMs counts and typed claims
- :528: the `\cmPnl…` claims
- :551: `\cmXLossMin/Max`
- :585: the `\cmSv…` claims
- :763: the closing summary

**appendix_close_pnl.tex** (1)

- :47: table_close_pnl and the `\cmPnl…` claims

**appendix_close_shift_test.tex** (2)

- :37: the shift script must use ŷ_k^2 B (no smear); k = 0 must reproduce Table 7
- :71: the "falls" and best-stale-row claims

**appendix_close_murphy_splits.tex** (4)

- :34: the headline's QLIKE, score at the threshold, and ranks
- :46: the Murphy leaders
- :70: the four Spearman values
- :91: the splits, pins and sign split

**appendix_close_compounding.tex** (1)

- :38: the `\cmp…` macros, "sits inside both", and the ranks

**appendix_campaign_16h.tex** (13)

- :50: the tables drop their intervals, and the counts count point estimates
- :163, :209, :231, :265, :296, :311, :330, :436, :499: the counts must count point estimates, then fix singular and plural
- :481: `\campClass…`
- :544: the master table before and after
- :608: `\campDvsLgbm…`

**introduction.tex** (1)

- :107: the claims in the paper's summary of Section 5.4

**conclusion.tex** (1)

- :57: the claims in the paper's summary of Section 5.4

**related_works.tex** (1)

- :110: `\cmSellHead`, and `\cmPnlDiff` as a point estimate

### 8.4 Build

Two scratch builds were compared, neither touching `writeup/main.*`:

- **Edited tree:** `pdflatex -jobname=plainnew -aux-directory=<scratch> -output-directory=<scratch>`, then bibtex and three more pdflatex passes.
- **Baseline:** the same run on a scratch copy holding the HEAD versions of the 11 edited files.

A first attempt with only `-output-directory` gave false "undefined" warnings for `eq:close_opt_plain`. MiKTeX had read the stale tracked `writeup/main.aux` from the working directory. Using a distinct `-jobname` together with `-aux-directory` fixes this.

The edited tree builds to 96 pages with no errors, no undefined references or citations, no multiply defined labels, and 0 overfull boxes. The baseline builds to 99 pages with 1 overfull box, the known 1.6 pt `\vbox`. No new overfull box was introduced.

Both builds used the generated files as they were at about 14:46. Other agents were regenerating them at the time: `generated/table_close_pnl.tex` changed at 14:41.

In the rendered PDF the in-scope sections still show interval brackets in two places:

- the macros that still carry `[lo, hi]`: 49 references, all under a TODO-PLAIN;
- the generated table bodies and captions.

Both clear when the regenerated macros land.

### 8.5 Out of scope: places that still mention recalibration or intervals (not changed)

1. **The main results' retransformation protocol.** This covers `sections/methods.tex` (`sec:smearing_contract`, 14 "smear" hits), `generated/table_smear_sensitivity.tex`, `sections/results.tex`, `sections/appendix.tex`, the introduction's second finding (:31–44, "Mincer–Zarnowitz mean map"), related works "Retransformation" (:80–93), and data preparation :64, :100 and :266–285 (`eq:dp_duan`).
2. **The notebook running appendix D.1–D.13 (`appendix_running.tex`):** 4 recalibration and 26 interval hits.
   - :358 "a recalibration per entry clock".
   - :571–599 "research scorer, which recalibrates the 16:00 bar on its own; the notebook's all-bar recalibration".
   - The sizing appendix (`sec:app_run_sizing`, intervals).
   - The post-2024 appendix.
   - The strategy-variations appendix (`sec:app_strategy_variations`, whose generated caption says "16:00-bar recalibration").
3. **`appendix_running_parked.tex` (D.15).** The whole appendix is session-bar recalibration with intervals. Its first paragraph (:14–25) says Section 5.4 "now scores every forecast under the 16:00-bar recalibration of Table 7". That is stale after this pass and needs one sentence from its owner.
4. **`writeup/main.tex` :101–111.** The proposed abstract sentence (inside `\iffalse`, not rendered) says "resolved above both selling it every day and trading the paper's own forecast". This file is outside `sections/` and was not touched.
5. **Multi-horizon sections** (`methods_multihorizon.tex`, `results_multihorizon.tex`) mention only "smear".
6. **Generated files, owned by the regenerating agents.** These captions and headers say "16:00-bar recalibration" and/or "95 % interval":
   - `close_main_numbers.tex` (header),
   - `table_close_main.tex`, `table_close_ladder.tex`, `table_close_pnl.tex`,
   - `appendix_close_{shift_test,murphy_splits,compounding}.tex`,
   - `appendix_campaign_16h_*.tex`,
   - `strategy_variations.tex`,
   - `table_master_close*.tex`,
   - `appendix_running_restofday_trading.tex` ("recalibrated per entry time").
7. **A comment, not prose:** `appendix_campaign_16h.tex` (compute-table source comment, :656) names the clusters ("Cluster A = CARC (usc-discovery), cluster B = Hoffman2"). This breaks the no-lab-names rule only in the source.

### 8.6 Second pass, 2026-10-06 (afternoon): TODO-PLAIN lines resolved on the regenerated numbers

**Inputs.** The re-checks used:

- `results/close_master_table/PLAIN_DONE` (14:53:58);
- `results/close_shift_test/PLAIN_HEADLINE_DONE` (15:01:23);
- `writeup/generated/MACROS_PLAIN_DONE` (15:32:50), covering `close_main_numbers.tex`, `table_close_{main,ladder,pnl}.tex` and `appendix_campaign_16h_*`;
- the D.16–D.18 macro files, regenerated at 14:56–14:58, after PLAIN_DONE.

Every claim was checked against:

- those macros;
- the claim list at the end of `close_main_numbers.tex` (25 statements FLIPPED);
- `results/close_master_table/SUMMARY.md`;
- `results/close_pnl_decomp/research_scorer/tail_days.csv`, which confirms that the 20 best days are all buys and the 20 worst all sells.

Rules applied:

- Counts are now point-estimate counts. The significant count (`…Sig`, 5 %, HAC t for Sharpe, DM for QLIKE) is given beside each one.
- No parked macro (the `…Ci`/`…CI`/`…Lo`/`…Hi`/`…Interval…`/`…Who`/`MTSig*`/`MTQBetter*` stubs) is referenced; the scratch script `macrocheck.py` was run on every in-scope file.
- The first-pass (unreviewed) wording was replaced in place. The pre-2026-10-06 text stays parked as in 8.2.

What happened to the 48 TODO-PLAIN lines:

- **Resolved:** 44.
- **Dropped:** 1, with the dense-vs-sparse text (8.7).
- **Still open:** 3, all in `methods_close_option.tex`; their sources were not regenerated (see 8.10).

**FLIPPED claims and what the text now says.** Several of these were parked afterwards with the headline framing (8.8).

**Section 5.4, the paper's forecasts**

- All eight still beat always short. Two do so significantly: LightGBM and the causally tuned lasso (`\cmPaperVsShortAboveSig{N,Who}`).
- R is not significantly above always short (`\cmRefVsShortT`). Neither is the FOMC gap (`\cmFomcDT`).
- FLIPPED: the fixed lasso no longer stands out. Its sentence now names:
  - the column furthest from R: `\cmPaperFarLabel`, the OLS baseline, at `\cmPaperFarVsRef`, HAC t `\cmPaperFarVsRefT`;
  - the highest of the eight: `\cmPaperBestLabel`, the causally tuned lasso, at `\cmPaperBestSh`.

**Section 5.4, the comparison set against R** (`\cmTabVsRef…`, `\cmMasterVsRef…`)

- Point counts, with the significant counts beside them.
- Added: the Sharpe ratios rank forecasts by point estimate, not by an established order. The maximum of `\cmMasterN` (`\cmTopShMid`) would select on noise.
- Added the user-requested sentence: in panel C the elastic net and the lasso on `all_features` trade highest (up to `\cmSubShMax`), but no Sharpe difference among the per-bar linear forecasts is significant, and their lead is concentrated in 2020–2021.
  - The source is cited in a comment: the main-session analysis found every pairwise HAC t at or below 1.4.
  - No macro exists for this.

**Section 5.4, the headline paragraphs** (later parked, 8.8)

- FLIPPED: the headline was *less* accurate than R in point estimate (QLIKE +10.9 %, DM +0.60).
- FLIPPED: the per-bar ridge on `live_vix_only` traded significantly above it (HAC t 2.46).
- FLIPPED: 24 of 219 forecasts traded above it, 2 significantly.

**Section 5.4, Table 8 and the ladder**

- Table 8's caption now names the HAC t column.
- The ladder keeps:
  - the specification step;
  - the per-bar step: a significant QLIKE gain on HAR + calendar, while the trade moves by `\cmLdPerBarBase` (HAC t `\cmLdPerBarBaseT`);
  - the count of significant steps (`\cmLdResolved`, `\cmLdQResolved`).
- FLIPPED: the costliest estimator change is the random forest, not XGBoost (later parked with the headline path).

**Section 5.4, trees and LSTMs**

- FLIPPED: trees beat the ridge on QLIKE in 48 of 72 cases. The text gives the counts with their significant counts.
- FLIPPED and parked: XGBoost on `all_features` as the trade that daily refits move most.
- FLIPPED: "Tuning does not make … more accurate" became "Tuning mostly makes … less accurate". Optuna is more accurate in 2 of 36 cases.
- FLIPPED: no "never" claims remain on more frequent tuning or on the best of 50 trials.
- FLIPPED: the intraday LSTM against the ridge is a signed range.
- FLIPPED: the best tree is now the Optuna XGBoost (`\cmNlBestLabel`). This sentence was later parked (8.8).

**Section 5.4, P&L and other structures** (later parked, 8.8)

- P&L: the top-20 payoff days were softened to "more, but not significantly at 5 % (p `\cmPnlMktTwentyP`)".
- FLIPPED: "None beats the straddle".
- FLIPPED: the closest structure at the midpoint is now the strangle one strike wide.
- FLIPPED: at the crossed spread the straddle no longer ranks first.

**Other files**

- **`related_works.tex`.** `\cmSellHead` and `\cmPnlDiff` were first re-read, then replaced in 8.8.
- **`methods_close_option.tex`.** The sell shares hold: 72–83 % for the paper's eight forecasts, so "leans short" holds.
- **D.14.** The top-20 payoff days were softened as in Section 5.4. Before: "in the high-VIX tercile the two make the same". After: "only `\cmPnlDiffHighVix` of it is made in the high-VIX tercile" (+11.9 of +97.0).
- **D.16.** The script now uses ŷ_k^2 B, and k = 0 reproduces Table 7. Added `\shiftNStaleBelow` and `\shiftNNextAbove`.
- **D.17.** Macro-only. Ranks are labelled as ranks of point estimates.
- **D.18.** f = 0.03 sits inside both ruin bounds (0.15, 0.10). Ranks are labelled as ranks of point estimates.
- **D.19.** All 12 count TODOs resolved with the `…Sig` twins.
  - The conventions paragraph defines "significant".
  - The Optuna-vs and master-table captions name the significant counts.
  - The master-table caption reads "median and range (minimum to maximum)".
- **D.15 (`appendix_running_parked.tex`), coordinator item 3.**
  - The stale sentence now reads "now uses every forecast as its plain back-transform f^2 B (Equation~\eqref{eq:close_opt_plain}), with no recalibration".
  - The last clause now reads "not been re-run with the plain forecast".
  - The session-bar recalibration is described as "a Mincer–Zarnowitz map fitted on the regular-session bars, since removed from Section~\ref{sec:close_option_methods}".
  - The old paragraph is parked.

### 8.7 Dense-vs-sparse write-up parked (user decision, 2026-10-06)

`appendix_campaign_16h.tex`, under `% 2026-10-06 parked (dense-vs-sparse write-up removed, user decision):`. Parked:

- the intro clause "importance and density studies", which now reads "importance study";
- the subsection title, which now reads "Feature importance on the new design" (label kept);
- the opening clause "and the dense-versus-sparse study";
- the density paragraph (`\campDvs…`, with its TODO-PLAIN);
- the density table (`tab:app_camp_dvs`).

Nothing else in `sections/` or `main.tex` references them. The paper's "dense but weak" theme and the multi-horizon `dense_numbers` / `table_dense_*` were left alone.

### 8.8 No headline model; live-feasible removed (user decisions, 2026-10-06)

Old text is parked under one of three headers:

- `% 2026-10-06 parked (no headline model, user decision):`
- `% … (live-feasible removed from the paper, user decision):`
- the combined header, for passages that fall under both decisions.

Every number in the new text is an existing macro. Where a model-set macro is missing, the text says nothing about it and a `% TODO-SET:` line names the macro needed.

**Section 5.4 (`results_close_option.tex`)**

- **Table 7 caption.** Dropped the panel D "each LightGBM or the LSTM on live_feasible" and "H: the headline forecast (bold)". TODO-SET: regenerate the table without the H marker and the live_feasible rows.
- **Per-bar tables.** The column counts are now `baseline` and `all_features` only. The de-dup sentence no longer says "the headline keeps every position".
- **Comparison set.** Kept, with the ranking caveat and the maximum (`\cmTopShMid`) as an example of selection on noise. TODO-SET lists the model-set macros that are missing:
  - above always short, point and significant: `\cmMasterVsShortAbove(Sig)`;
  - medians: `\cmMasterMed{Q,Sh,ShX}`;
  - the rank correlation of QLIKE with Sharpe: `\cmMasterRankCorr`. `\hmurRhoQAll` exists only in D.17's file, which is input after Section 5.4.
- **Parked:**
  - the three headline paragraphs ("The headline forecast, and why this one", its accuracy and trade, "How close its neighbours come");
  - the ladder's headline path and its live_feasible steps (`\cmLdPerBarLive`, `\cmLdInAll`/`\cmLdInBase`, the estimator changes from the headline, the "headline's lead is the sum of three steps");
  - the campaign-rung paragraph of Table 8 (all on live_feasible);
  - Table 8's caption sentence on the campaign blocks;
  - the P&L tail-trade paragraph (the headline's P&L);
  - "Other structures" (scored under the headline);
  - the diagnostics paragraph's pointer "repeated for the headline forecast in Appendices D.16–D.18";
  - the headline closing summary.
- **The ladder paragraph now opens:** "Table 8 pairs forecasts of Table 7 that differ in one thing". It keeps the specification step, the per-bar step on `all_features` and HAR + calendar, and the significance count.
- **Trees and LSTMs.** Dropped the `live_feasible` LSTM and intraday parts and the best-tree-against-the-headline sentence. TODO-SET: the counts still include the live_feasible arms.
- **New closing summary, about the model set:**
  - the paper's eight forecasts are all above always short, `\cmPaperVsShortAboveSigN` significantly;
  - across the `\cmMasterN` forecasts, `\cmMasterVsRefAbove` trade above R and `\cmMasterVsRefBelow` below, `\cmMasterVsRefAboveSig`/`\cmMasterVsRefBelowSig` significantly;
  - no step of Table 8 moves the Sharpe ratio significantly;
  - a better forecast is not by itself a better trade;
  - there is no edge after April 2024.
- **TODO-SET lines:**
  - Table 8 to be regenerated as model-set steps;
  - a model-set tail measure;
  - strategy variations over the model set;
  - `\cmMasterVsShort…` for the summary.

**`methods_close_option.tex`**

- The live_feasible input set (`\item`) is parked. "Three input sets" became "Two".
- "This is the headline rule" became "This is the rule the section studies".
- The compounding pointer no longer says "for the headline forecast", with a TODO-HEADLINE (D.18 is the former headline's).
- The sell shares drop the headline's `\cmSellHead`. TODO-SET: `\cmSellMaster…`.

**Front matter**

- **`introduction.tex`.** The headline passage is parked and replaced by the model-set statement:
  - every paper forecast beats always short, `\cmPaperVsShortAboveSigN` significantly;
  - across the `\cmMasterN` forecasts against R, the point counts and significant counts;
  - no order among the models;
  - a better forecast is not a better trade;
  - no edge after April 2024;
  - sizing does not help.
- **`conclusion.tex`.** The "ridge fitted for the bar alone, on inputs a forecaster can rebuild live" passage is parked and replaced by the model-set statement.
- **`related_works.tex`.** "under the headline forecast … `\cmSellHead`" became "for the paper's eight forecasts … `\cmSellPaperMin` to `\cmSellPaperMax`". The clause on `\cmPnlDiff` (the headline's P&L) is parked.

**Appendix D.19 (`appendix_campaign_16h.tex`)** follows b761b28: every live_feasible reference and `\camp…Live` use is parked.

- **Parked live_feasible references and `\camp…Live` uses:**
  - the input-set key;
  - the design counts;
  - the kept-column counts;
  - the LSTM cadence and LSTM-against-ridge values;
  - the picks caption;
  - the intraday-LSTM kept columns;
  - the AVX-512 column difference;
  - the whole-run CPU-class check, run on LightGBM live_feasible only (TODO-SET);
  - `\campRidgeBestTree`, a live_feasible arm (TODO-SET).
- **Parked headline uses:**
  - the master table's "The headline forecast, the per-bar ridge on live_feasible, is unchanged …";
  - "Against the headline, of the 219 other forecasts …".
- **TODO-SET at the conventions paragraph:** every count of the appendix still includes the live_feasible arms; regenerate over `baseline` and `all_features`.

**Appendices D.14, D.16, D.17, D.18** were left as they are. Each has `% TODO-HEADLINE: appendix built on the headline; user to decide (park, or re-base on the model set)` on its first line.

### 8.9 Build (second pass, final)

Scratch build: `pdflatex -jobname=plainp4`, aux and output in the scratch directory, then bibtex and three more pdflatex passes. Result:

- 79 pages;
- no errors;
- 0 overfull boxes;
- 0 undefined references or citations;
- no multiply defined labels;
- no "??" in the PDF text.

The tracked `writeup/main.*` were not touched. The page count fell from 96 because of:

- other agents' parking in `methods.tex`, `appendix.tex` and the multi-horizon sections;
- this pass's parking (dense-vs-sparse, the headline paragraphs, P&L, other structures).

Generated Tables 7 and 8 still show live_feasible rows, the H marker and the headline path until they are regenerated; their TODO-SET lines say so.

### 8.10 Still open after the second pass

**TODO-PLAIN (3), `methods_close_option.tex`**

- :158, the B-contamination audit bound;
- :433, the 0.06 censored-IV bound;
- :616, the sizing test (notebook Appendix D.11, not re-run with f^2 B).

**TODO-HEADLINE (5)**

- the first line of D.14 (`appendix_close_pnl.tex`), D.16 (`appendix_close_shift_test.tex`), D.17 (`appendix_close_murphy_splits.tex`) and D.18 (`appendix_close_compounding.tex`): user to decide whether to park them or re-base them on the model set;
- `methods_close_option.tex`:548, the compounding pointer to D.18.

**TODO-SET (11): macros or regenerations needed**

`results_close_option.tex`:

- :38, Table 7 without the H marker and the live_feasible rows;
- :209, `\cmMasterVsShortAbove(Sig)`, `\cmMasterMed{Q,Sh,ShX}`, `\cmMasterRankCorr`, and every count without live_feasible;
- :409, Table 8 regenerated as model-set steps;
- :497, the tree and LSTM counts without live_feasible;
- :585, a model-set tail measure (`\cmMasterTopTwentyShareMed`, …);
- :638, strategy variations over the model set;
- :835, `\cmMasterVsShortAbove(Sig)` for the closing summary.

`methods_close_option.tex`:

- :612, `\cmSellMaster{Min,Max}`.

`appendix_campaign_16h.tex`:

- :66, every count without live_feasible;
- :260, `\campRidgeBestTree` without live_feasible;
- :523, the CPU-class whole-run check on a non-live arm.

**Outside this pass's files, still headline- or live-feasible-based**

- Appendix D.13 strategy variations (`generated/strategy_variations.tex`, still dated 09-29 and recalibrated);
- `appendix_running.tex`;
- `main.tex`'s `\iffalse` abstract proposal;
- the headline-diagnostic generated tables.
### 8.11 Third pass, 2026-10-06 (evening): main-results sentences, LSTMs, D.5 and sizing, the model-set macros

Old text is parked under one of these headers:

- `% 2026-10-06 parked (main results re-scored on the plain back-transform, user decision):`
- `% … (LSTM not yet trustworthy, user decision):`
- `% … (running appendix re-run on the plain forecast: D.5 held out, sizing re-run):`
- `% … (model-set numbers regenerated, user decisions):`

Numbers appear only through macros:

- `\unif…` from `generated/campaign_numbers.tex`, 15:36;
- `\cm…` and `\camp…` from MACROS_SET_DONE, 17:11.

**Main-results sentences** (the coordinator's list; data_prep's B = 1 sentinel sentence at :250–252 is untouched)

- **`introduction.tex`**
  - **Second finding** (Duan / Mincer–Zarnowitz / GARCH) parked. It is now one sentence: a convention, every forecast scored as its plain back-transform ŷ²B (Section 4.2).
  - **"ridge beats lasso on every design"** became "penalty families not separated under one causal tuning rule", with the lasso–ridge gap `\unifIncrBTwoLassoTunedDQ` at DM `\unifIncrBTwoLassoTunedDM`, and "heavier ridge shrinkage lowers the loss steadily".
  - **Fourth finding:**
    - Dropped: "an order of magnitude smaller", "on all nine feature sets" and "additive-plus-pairwise".
    - New text: the hedged tree bank beats the two-block ridge by `\unifTreeBankHedgeVsBlkTwoDQ` (DM `\unifTreeBankHedgeVsBlkTwoDM`), more than the ridge beats the benchmark (`\unifBlkTwoUserDelta`). Most explainable variance is additive and HAR.
  - **"specification statement"** dropped. The ladder ends at the two-block ridge: fixed penalties `\unifBlkTwoUserQLIKE` / DM `\unifBlkTwoUserDM`, causally tuned `\unifBlkTwoTunedQLIKE` / DM `\unifBlkTwoTunedDM`.
  - **"discipline beats tuning"** became "what tuning buys is measured":
    - weighting beats selecting for the trees: `\unifIncrTreeHedgeDQ`, DM `\unifIncrTreeHedgeDM`;
    - the tuned elastic net beats tuned ridge: `\unifIncrBThreeEnetTunedDQ`, DM `\unifIncrBThreeEnetTunedDM`;
    - the tuned two-block beats the fixed two-block.
  - **Roadmap:** "back-transformation protocol" became "back-transform".
- **`conclusion.tex`**
  - "small beside … pairwise" and "strong paired significance" parked. The text now says the nonlinearity is not small (tree bank against two-block, with macros). The two-block ridge is `\unifBlkTwoUserDelta` / DM `\unifBlkTwoUserDM` with fixed penalties, short of significance, and `\unifBlkTwoTunedDelta` / DM `\unifBlkTwoTunedDM` with tuned penalties.
  - The Duan finding is deleted (parked). The tuning finding is restated as estimator-dependent.
  - Limitations: the legacy protocol, the conditional-variance panel and the GARCH extension are parked. One extension remains.
- **`related_works.tex`**
  - The "sparse pairwise products written back as columns" claim became "trees measure what the linear design leaves; a decomposition locates the explainable variance".
  - The retransformation paragraph became "the forecast is used uncorrected, as ŷ²B (Section 4.2)".
- **`data_prep.tex`**
  - "calibration burn-in" (twice) became "the first walk-forward window of each arm is not scored".
  - The FOMC bucket no longer beats the baseline. The text says its QLIKE is higher, not significantly (see Table `tab:fomc_bucket`).
  - The Duan / second-order paragraph and `eq:dp_duan` became the plain inverse pointing to `eq:backtransform`. `eq:dp_duan` was referenced nowhere else.
- **`introduction.tex` layout fix.** The close-study macro file is now input right after `\label{sec:intro}`, in vertical mode. Input mid-paragraph, the line ends of its 274 definitions had produced a 1019 pt overfull box.

**LSTMs parked** (user decision: not yet trustworthy)

- **`methods_close_option.tex`.** The two-LSTM description became one sentence: an LSTM was fitted, and it is not reported because its training setup is unsettled. The mask and CPU-class sentences now say "linear or tree" and "trees".
- **Section 5.4**
  - The opening, the Table 7 caption, the rebuilt paragraph and the panel D range now say trees only.
  - The paragraph title became "Trees, every rung".
  - The per-bar and intraday LSTM sentences are parked.
- **D.19**
  - The intro has one line saying the LSTMs are not reported.
  - Parked: the LSTM design move, the LSTM mask counts, LSTM10, the LSTM cadence and the LSTM-against-ridge sentences, the intraday-LSTM subsection (with `tab:app_camp_lstmi`) and the intraday-LSTM class re-run.
  - The cadence caption, the CPU sentence, the master-table "new keys" sentence and the compute sentence now say trees only. One compute stage also ran the LSTMs, which are not reported.

**Appendix D.5 and sizing** (from the running-appendix agent)

- **D.5:** it is now cited only as the live card's record, not scored with the plain forecast and not reported. Changed at `methods_close_option.tex` :100–105, `results_close_option.tex` opening and closing, the introduction and the conclusion.
- **Sizing:**
  - "does not improve on its sign" and "its own back-transform" became the re-run's point estimates, in words: after a year of history the robust clip and the rank of |s| each trade above sign(s) for most of the forecasts; from the second day sign(s) trades above both for most of them.
  - Changed in methods (the sizing TODO-PLAIN is resolved), in the Section 5.4 diagnostics paragraph, in related works, in the introduction and in the conclusion.
  - The `\runappSz…` macros cannot be quoted in the main text. `appendix_running.tex` inputs `generated/appendix_running_numbers.tex` unguarded, after the main text, so inputting it earlier would redefine the macros. One TODO-SET in methods (:648) says so.

**TODO-SET and TODO-PLAIN resolved on the model-set macros** (model set: 142 forecasts, no headline, no live_feasible, no LSTM)

- **Methods**
  - **B-contamination bound (FLIPPED).** The new sentence gives:
    - affected days: `\cmBContamDays`;
    - the largest move of a forecast: `\cmBContamMaxRel` %;
    - |ΔSharpe| median `\cmBContamMedDSh`, maximum `\cmBContamMaxDSh` at the midpoint (`\cmBContamMaxDShX` crossed), maximum `\cmBContamPaperMaxDSh` for the paper's eight;
    - flips: median `\cmBContamMedFlips`, maximum `\cmBContamMaxFlips`.

    "Under a tenth of a Sharpe ratio" is parked.
  - **Censored IV (FLIPPED at the crossed fill).** `\cmCensIvDays` days; median `\cmCensIvMedDSh`, maximum `\cmCensIvMaxDSh` at the midpoint and `\cmCensIvMaxDShX` crossed.
  - **Sell shares over the model set:** `\cmSellMasterMin`–`\cmSellMasterMax` %, median `\cmSellMasterMed` %. This also repairs a sentence that the earlier TODO comment had cut ("What the forecast contributes …").
- **Section 5.4**
  - **TODO lines** for `\cmNlFamN`, Table 7, `\cmClassChunks` and `\cmNlShMin/Max`: resolved and removed. The macros are now trees only.
  - **`\cmClassMaxRelMax`** is worded as a whole-run check on a LightGBM forecast outside the paper's input sets.
  - **The table's maximum (FLIPPED).** The highest of the 142, the per-bar ridge on `free_vix_only` (`\cmTopShLabel`, `\cmTopShInputs`, `\cmTopShMid`), is also the one forecast significantly above R.
  - **New paragraph, "The model set as a whole":**
    - medians and ranges (`\cmMasterMed{Sh,ShX,Q}`, `\cmMasterSh{Min,Max}`);
    - against always short: `\cmMasterVsShortAbove(Sig)` of `\cmMasterVsShortN`, median `\cmMasterVsShortMed`;
    - QLIKE against R: `\cmMasterQVsRef{Below,Above}(Sig)`;
    - rank correlation of QLIKE with Sharpe: `\cmMasterRankCorr`.
  - **Table 8 (model-set steps).** The caption names the HAC t column and the campaign blocks (LightGBM on `all_features`). The ladder paragraph adds:
    - the input step (`\cmLdIn…`);
    - the estimator range on `all_features`, from `\cmLdEstMin` (LightGBM, shipped settings) to `\cmLdEstMax` (elastic net);
    - the campaign rungs (`\cmLdC…`: mask, cadence, random search, Optuna, how often to tune).
  - **Trees:** "three input sets" became "two input sets" in both places.
  - **New paragraph, "A tail trade", over the model set:**
    - the 20 best days are buys for `\cmMasterTopTwentyBuysN` and the 20 worst are sells for `\cmMasterBottomTwentySellsN` of `\cmMasterPosTotalN`;
    - the top-20 share has median `\cmMasterTopTwentyShareMed` % (range `\cmMasterTopTwentyShare{Min,Max}`);
    - the difference to always short is positive for `\cmMasterDiffPosN`, median `\cmMasterDiffMed`;
    - without its 10 best days it is below half for `\cmMasterDiffExTenHalfN` and negative for `\cmMasterDiffExTenNegN` (median `\cmMasterDiffExTenMed`).
  - **New "Other structures" over the model set** (FLIPPED: "straddle first at the crossed spread for every forecast" and "the iron butterfly is closest"):
    - `\cmSvSetPairsN` pairs; above the straddle at the midpoint `\cmSvSetAboveMid` (`\cmSvSetAboveMidSig` significantly);
    - the straddle is first for `\cmSvSetStraddleFirstMid` forecasts at the midpoint and `\cmSvSetStraddleFirstX` crossed;
    - the best alternative by median is the `\cmSvSetBestMidLabel` (median `\cmSvSetBestMidMed`, above for `\cmSvSetBestMidAboveN`, `\cmSvSetBestMidAboveSig` significantly);
    - wings lie below the straddle in `\cmSvSetWingXBelow` of `\cmSvSetWingN` pairs crossed and `\cmSvSetWingMidBelow` at the midpoint.
  - **Closing summary:** adds `\cmMasterVsShortAbove` of `\cmMasterVsShortN` above always short (`\cmMasterVsShortAboveSig` significantly).
- **D.19**
  - The live_feasible count TODO is resolved; the macros now cover `baseline` and `all_features`.
  - "lowest relative QLIKE is `\campRidgeBestTree`" is restored; it is now T1 LightGBM on `all_features`.
  - The `\campClass…` whole-run check is not restated; a comment says why (live_feasible only).
  - The `\campMTKeys…` TODO is resolved (no LSTM keys).
  - The compute TODO is resolved by the wording above.

**Build (scratch, `-jobname=plainp7`)**

- 73 pages;
- 0 errors;
- 0 overfull boxes;
- 0 undefined references or citations;
- no multiply defined labels;
- no "??" in the PDF text.

Section 5.4 and D.19 contain no LSTM, live-feasible or headline text, apart from the one-line "LSTMs not reported" notes.

**Still open**

- **TODO-HEADLINE (5), for the user:**
  - the first line of D.14, D.16, D.17 and D.18;
  - `methods_close_option.tex` :573, the compounding pointer.
- **TODO-SET (1):** `methods_close_option.tex` :648, the `\runappSz…` guard; it needs `appendix_running.tex`'s input to be `\ifdefined`-guarded.
- **TODO-PLAIN:** none in this workstream's files.
### 8.12 Fourth pass, 2026-10-06: the last TODO-SET closed (sizing counts by macro)

- `sections/appendix_running.tex`: its `\input{generated/appendix_running_numbers}` (the file that defines the `\runappSz…` sizing macros) is now guarded with `\ifdefined\runappRpNdays\else … \fi`, the same pattern `methods_multihorizon` uses for `dense_numbers`. The table file `appendix_running_sizing.tex` defines no macros and needs no guard.
- `sections/introduction.tex`: the file is input once, early, beside the close-study macros, right after `\label{sec:intro}` (vertical mode).
- `methods_close_option.tex`: the TODO-SET block (:648) is removed. The sizing sentence now quotes the counts by macro:
  - after a year of history (`\runappSzDays` days): robust clip above sign(s) for `\runappSzClipAbove` of `\runappSzForecasts` forecasts (median `\runappSzClipMed`), and rank of |s| for `\runappSzRankAbove` (median `\runappSzRankMed`);
  - from the second day (`\runappSzDaysTwo` days): `\runappSzClipAboveTwo` and `\runappSzRankAboveTwo` (medians `\runappSzClipMedTwo`, `\runappSzRankMedTwo`).

  The worded version is parked.
- `results_close_option.tex`, diagnostics paragraph: the same counts by macro (`\runappSzClipAbove`, `\runappSzRankAbove` of `\runappSzForecasts`; from day 2 `\runappSzClipAboveTwo`, `\runappSzRankAboveTwo`). The worded version is parked.
- The introduction, conclusion and related works state no sizing number ("turns on the sample") and are unchanged.
- **Build** (scratch, `-jobname=plainp8`): 73 pages, 0 errors, 0 overfull boxes, 0 undefined references or citations, no multiply defined labels, no "??" in the PDF text.
- **Open in this workstream:** only the 5 TODO-HEADLINE lines, for the user. These are the first line of D.14, D.16, D.17 and D.18, and `methods_close_option.tex` (the compounding pointer to D.18). No TODO-SET or TODO-PLAIN remains in this workstream's files.
### 8.13 Fifth pass, 2026-10-06 (night): lean set, pre-tuned linear models, D.14 parked, D.16–D.18 re-based, the QLIKE–Sharpe figure, B = 1 bars

Each user decision below is applied with macros only, and the old text is parked under a dated header.

**Lean set as the third input set** ("make the lean set the third input set in the paper, but it has to be a follow up to the feature importance study").

- **New file `sections/appendix_lean_set.tex`** (Appendix subsection "A Lean Input Set, Chosen from the Importance Study", label `sec:app_lean`). It is input at the end of `appendix_campaign_16h.tex`, after D.19's compute subsection, so D.19 keeps its campaign framing. It has four parts:
  - **What the importance study finds.** The `\fl…` macros give:
    - the HAR lead: median rank `\flRankMedHar`, top-10 in `\flTopTenHar` of `\flCells` cells;
    - the next four series: signed return, absolute return, E-mini volume and autocovariance, at median ranks `\flRankMedSumret`, `\flRankMedSumabsret`, `\flRankMedSumvolume` and `\flRankMedSumautocov`;
    - the bipower / upside-squared cluster with the absolute and quartic return, with the smallest within-cluster correlation `\flClusterCorrMin`–`\flClusterCorrMax` across the lags;
    - the constituents' returns, which rank but duplicate the E-mini moments;
    - VIX, which counts more for the linear models (median rank `\flRankMedVix`).

    Two claims have no macro, and a comment cites their source in `results/feature_importance_1530_dedup/SUMMARY.md`: that the constituents "duplicate" the E-mini moments, and that VIX matters more for the linear models than for the trees.

    Table `tab:app_lean_importance` is `generated/featimp_lean_importance`.
  - **The lean set** (`lean_vol_vix`, six series, no `adj_`); `\flLeanInTopTenN` of `\flLeanN` are in the top ten.
  - **Its evaluation** against `all_features` and `baseline`, over the `\flSetN` pairs, including the pooled twins: medians, counts with their significant twins, and median differences. Table `tab:app_lean_results` is `generated/featimp_lean_results`. A sentence says the pairs are not independent.
  - **A selection caveat**, per the main-session analysis:
    - the pre-2020 refits (`\flPreRefits`, `\flPreTailDays` days) pick the same five return and volume series (`\flPreRank…`);
    - the cluster already holds on the pre-2020 rows (`\flPreClusterMinCorrLag…`);
    - VIX is in the set by design, not by its rank;
    - the set is reproducible from refits trained before the trade period, but the selection rule was written after the full-sample results had been seen.
- **`methods_close_option.tex`:**
  - "Two input sets" became "Three input sets".
  - A new `\item` for `lean_vol_vix` points to `sec:app_lean`.
- **Section 5.4:**
  - The model-set paragraph gives medians by input set (`\cmMasterMed{Sh,Q}{Base,All,Lean}`, `\cmMasterN{Base,All,Lean}`) and says the lean rows carry the selection.
  - The ladder gains the lean steps: `\cmLdInBaseLean`, `\cmLdInLeanAll` and their QLIKE changes.
  - The tree counts now say "three input sets".
  - The top-Sharpe sentence was corrected. It is now the per-bar elastic net on `lean_vol_vix` (`\cmTopSh…`) and is **not** significantly above R.
  - The panel C sentence (elastic net and lasso on `all_features` highest; no pairwise gap significant; 2020–21) is parked. Its source analysis predates the pre-tuned models, and the panel's top rows changed.
- **D.19:** the conventions name the three input sets, say the design and mask comparisons cover the first two, and point to `sec:app_lean`.
- **Introduction:** "three input sets, the third a lean set chosen after the importance study".
- **Moments + VIX:** no such text existed in these files (user: "get rid of moments + VIX only everywhere").

**Pre-tuned linear models** ("use the pre-tuned version for all linear models"). In `methods_close_option.tex`, the causal re-choice description is parked. In its place, the pre-tune is described once with the `\cmPt…` macros:

- `\cmPtFolds` folds of `\cmPtFoldLen` sessions, `\cmPtValFirst`–`\cmPtValLast`;
- fit window `\cmPtFitWin`, embargo `\cmPtEmbargo`;
- the HAR + calendar backbone at `\cmPtRatio` of the other inputs' penalty;
- blocks of `\cmPtBlock` sessions, with neighbour challenges under the trees' margin rule;
- `\cmPtFiredTotal` of `\cmPtFiredOf` challenges cleared the margin over `\cmPtRetuneN` re-tuning points.

OLS stays OLS. The reason for the change ("the causal re-choice switched often, mostly on sub-margin gains and between grid edges, and moved positions") is stated without numbers. No macro exists for the 75 of 165 re-choices; a comment cites `results/linear_pretune_2026-10-06/SUMMARY.md`.

**Bounds:** both use their macros. The B-contamination maximum is `\cmBContamMaxDSh`; the censored-IV maximum is `\cmCensIvMaxDSh` at the midpoint and `\cmCensIvMaxDShX` crossed. There is no typed 0.06.

**D.14 parked** ("comment out D.14"). `sections/appendix_close_pnl.tex` is commented out whole under one header and renders nothing; the file stays input from `main.tex`. Nothing live referenced `sec:app_close_pnl`.

**D.16–D.18 re-based on the model set** (163 forecasts; MODELSET_DONE). The old headline text is parked whole, and the TODO-HEADLINE lines are removed. The labels are kept.

- **D.16, shift test.** Only `\shiftNShifted` forecasts carry full-session rows: `\shiftNShiftedPaper` paper forecasts and `\shiftNShiftedPooled` pooled twins. The `\shiftNNotShifted` per-bar forecasts carry only the 16:00 row and cannot be shifted; the text says so. The new text gives:
  - the medians at k = 0, k = −1 and k = +1;
  - the below-k = 0 counts and their ranges;
  - the best stale row;
  - the realized-variance oracle and always short.

  The figure is `shift_test_modelset.png`.
- **D.17, threshold score and splits.**
  - Group counts and medians of QLIKE and of the score at the threshold.
  - Which group leads the Murphy diagram, by threshold count.
  - Rank agreement with Sharpe: QLIKE `\hmurRhoQMid` against the threshold score `\hmurRhoESMid` (point estimates; the threshold score tracks the trade better).
  - Splits over the model set: positive counts, medians, t counts, pins, and the same-day sign split.

  The figure is `murphy_modelset.png`.
- **D.18, compounding.**
  - `\cmpNPosMid` (mid) and `\cmpNPosX` (crossed) of `\cmpNSet` forecasts grow wealth.
  - Medians and ranges of terminal wealth and log-growth.
  - Drawdowns and the worst day.
  - The ruin bound is at least `\cmpRuinMin`, and f sits below it for `\cmpNRuinOK` forecasts.
  - Per-year counts.
  - No model is singled out.

  The figure is `wealth_modelset.png`.
- **Methods:** the compounding pointer TODO-HEADLINE is resolved: "reported for every forecast of the model set".
- **Section 5.4:** the diagnostics paragraph now points to D.16–D.18 as repeated for the model set.

**The QLIKE–Sharpe figure** (`figures/fig_qlike_sharpe.pdf`, `\fqs…` macros), placed in Section 5.4 after the model-set paragraph.

- The macro file is input in vertical mode beside the close-study macros. The label is `fig:qlike_sharpe`.
- The caption, built from macros, covers:
  - **what is drawn:** pairs `baseline` → `all_features` (`\fqsPairs`); LSTMs and the lean set are not drawn;
  - **adding the inputs:** it raises Sharpe for `\fqsShHigher` (`\fqsShHigherSig` significant) and lowers QLIKE for `\fqsQLower` (`\fqsQLowerSig`), with none significantly worse (`\fqsQHigherSig`, `\fqsShLowerSig`); where QLIKE does not improve, its change is within noise;
  - **the threshold:** the threshold-score change correlates at `\fqsCorrES` with the t of the P&L change, the QLIKE change at `\fqsCorrQ`;
  - **independence:** few gains are significant individually, and the counts are not independent (the models share the data).
- No model is dropped or singled out.

**B = 1 placeholder bars excluded** (user decision). In `data_prep.tex`:

- the scored count changes from 273,554 to 273,489 at three places, typed, because no `\unif` count macro exists yet;
- the baseline sentence now says "bars without a valid baseline are not scored".

The introduction and conclusion carry no typed counts and no "4.27".

**Build** (scratch, `-jobname=plainp11`):

- 78 pages;
- 0 errors;
- 0 overfull boxes;
- 0 undefined references or citations;
- no multiply defined labels;
- no "??" in the PDF text.

The only "headline" left in the PDF is in D.15, the parked session-bar appendix, which is out of scope.

**Open in this workstream:** none (no TODO-PLAIN, TODO-SET, TODO-LEAN or TODO-HEADLINE in these files).

**Missing macros:**

- the causal re-choice record (75 of 165);
- a `\unif` macro for the scored count (273,489).
### 8.14 2026-10-06 (late): counts and the re-choice record by macro

- `data_prep.tex`: the scored-bar count in all three places (:75, :122, and the FOMC table caption at :187) is now `\unifAzeroN` (273,489) from `generated/campaign_numbers.tex`, replacing the typed number. No other typed panel counts are in these files.
- `methods_close_option.tex`: the reason sentence for pre-tuning now quotes the paper-set macros. The causal re-choice switched the penalty at `\cmPtCausalSwitchesPaper` of its `\cmPtCausalChoicesPaper` re-choices on the paper's input sets. Of those, `\cmPtCausalMarginPassPaper` would have cleared the margin. `\cmPtCausalEdgePicksPaper` of its `\cmPtCausalPicksPaper` picks sat at an edge of the grid. A switch flipped `\cmPtSwitchFlipPctPaper` % of the positions in the block it governed. The worded version is parked.
- Lean scope confirmed by the coordinator: the lean set stays in Section 5.4, the methods and Tables 7/8; only the QLIKE–Sharpe figure excludes it.
- **Build** (scratch, `-jobname=plainp12`): 78 pages, 0 errors, 0 overfull boxes, 0 undefined references or citations, no multiply defined labels, no "??" in the PDF text. No TODO line remains in this workstream's files.

### 8.15 2026-10-06 (night): the pooled twins pre-tuned

- `methods_close_option.tex` (pre-tuning paragraph, after "two is the per-bar fit alone."): the pooled twins are pre-tuned the same way, on all 48 bars of the same folds. One sentence on the result: pre-tuning neither helps nor hurts significantly; against the causal re-choice QLIKE is worse in `\cmPoolPtQWorse` of `\cmPoolPtN` pairs (8 of 9), `\cmPoolPtQPctMean`% on average (+1.2), no DM or HAC t beyond 1.96 (largest `\cmPoolPtDMMax`, `\cmPoolPtTMax`). One sentence on the likely reason: the larger validation tail (all 48 bars) makes causal re-tuning less noisy; the margin rule replaces the penalty `\cmPoolPtRetuneSwMin`–`\cmPoolPtRetuneSwMax` times per model over `\cmPoolPtRetunes` re-tuning points. No new section.
- `results_close_option.tex` (ladder paragraph): "moves nothing on the trade (`\cmLdSpec`)" parked; now "moves the trade by `\cmLdSpec` (HAC t `\cmLdSpecT`)" (claim list: FLIPPED). The per-bar step on `all_features` is reported by macro only (`\cmLdPerBarAll` = -0.03, HAC t `\cmLdPerBarAllT`); no sentence says per-bar beats pooled on the trade. The lean per-bar step (`\cmLdPerBarLean`, `…T`, `…Q`, `…DM`) added in the same sentence. Pooled Sharpe range stays `\cmPoolShMin`–`\cmPoolShMax` (0.91–1.44).
- `appendix_close_shift_test.tex` (D.16, "What is tested"): old wording (paper forecasts plus pooled twins shifted) parked; now only the `\shiftNShiftedPaper` (9) paper forecasts are shifted, and the other `\shiftNNotShifted` (154) carry only the 16:00 row: `\shiftNNotShiftedPerbarLinear` per-bar linear, `\shiftNNotShiftedTrees` per-bar trees and `\shiftNNotShiftedPooled` pre-tuned pooled twins. The Reading paragraph quotes only regenerated `\shift…` macros (no wording change needed).
- Checked against the claim list: the lean appendix reports set-level medians and pair counts (`\flSet…`) only, never "lean beats all features" for the pooled group; tree claims (XGBoost refit move, Optuna) are counts by macro, none typed.
- **Build** (scratch, `-jobname=plainp13`): 78 pages, 0 errors, 0 overfull boxes, 0 undefined references or citations, no multiply defined labels, no "??" in the PDF text.

### 8.16 2026-10-07: the dev-tuned random forest and the tree-settings provenance

- `methods_close_option.tex`, trees paragraph: the old sentence "LightGBM and XGBoost take the settings of the paper's two tree columns ... and the random forest the library defaults" is parked. The new text says once where each tree's shipped settings come from. They come from an Optuna search per tree on the 1998–2002 sessions only, over the pooled all-bars `all_features` design, frozen as a menu, with no look-ahead into the trade period. Leaf minimums are scaled to the 2,000-row window (`\cmTreeProvLgbmLeafDev`→`\cmTreeProvLgbmLeaf`, `\cmTreeProvRfLeafDev`→`\cmTreeProvRfLeaf`). LightGBM and the random forest take their top entry. The forest's search was run later by the same procedure and stopped at `\cmTreeProvRfTrials` trials (nominal `\cmTreeProvRfBudget`), the count XGBoost's search reached in its only surviving record.
- XGBoost disclosure: XGBoost uses the `\cmTreeProvXgbDevRank`th-ranked of `\cmTreeProvXgbArms` entries (`\cmTreeProvXgbName`). It was picked by hand by `\cmTreeProvXgbPickDate`, with no recorded reason, after the menu had been scored on 2003+ sessions, so the pick cannot be certified free of look-ahead. It ranks last on accuracy from 2003 on (`\cmTreeProvXgbPostRank`th) and on the paper's QLIKE (`\cmTreeProvXgbQRank`th).
- Same paragraph: the random search and Optuna were run before the forest's dev search, so for the forest the grid's centre and Optuna's first trial are the library defaults (checked in `specs/causal_tune_trees_tuned.py` and `specs/causal_tune_trees_optuna.py`).
- `methods_close_option.tex`, "Refit contract" paragraph: "the hyperparameters behind those two columns cannot be recovered; ... basis of the pick is not recorded" is parked. The settings are stored with the forecast chunks, and the paragraph now points to the provenance text.
- `appendix_campaign_16h.tex` (D.19):
  - Optuna paragraph: trial 1 for the forest is the library defaults.
  - Mask subsection: both sides of the forest's masked-minus-unmasked comparison are the library-default forest, because the dev forest was run masked only (MACROS_RFDEV2_DONE).
  - Reproducibility across CPUs: the checks predate the forest's dev settings.
- `results_close_option.tex` (5.4), mask sentence: the same library-default-forest qualifier.
- Moved by macros only (MACROS_RFDEV_DONE, MACROS_RFDEV2_DONE): Tables 7/8, the tree-vs-ridge, cadence and tuning counts, the model-set medians, the lean evaluation, and the figure caption (`\fqsShHigher` 47, `\fqsQLower` 37, `\fqsCorrQ` −0.12, `\fqsCorrES` −0.41).
- Checked: "Tuning mostly makes a tree less accurate" still holds (RS 9 of 9, Optuna 35 of 36). "Daily refits lower every tree's QLIKE" still holds. Every FLIPPED tree claim is phrased as a count. The caption's "where QLIKE does not improve its change is within noise" still holds (`\fqsQHigherSig` 0). No sentence says no model improved QLIKE without Sharpe or that the worse-on-both pairs are all forests. The change figure is not used; the scatter stays.
- **Build** (scratch, `-jobname=plainp16`): 79 pages, 0 errors, 0 overfull, 0 undefined references/citations, no multiply defined labels, no "??".

### 8.17 2026-10-07: the pooled XGBoost column on its top dev-span entry (xgb_opt_00)

- **Decision (user, 2026-10-07; MASTER_TABLE_REVIEW item 76):** the pooled XGBoost column reads bank arm 10 (`xgb_opt_00`, the top entry of the 1998–2002 dev-span search, the rule LightGBM and the random forest follow) instead of arm 16 (`xgb_opt_06`, 7th-ranked, hand-picked after post-2002 scoring). Dump `results/spxw_pnl/yhat_tree10.parquet` (`results/atm_straddle_0dte_1530/TREE10_DONE`).
- **Code:** `notebooks/atm_straddle_lib.py` `yhat_paths["xgb"]` → `yhat_tree10.parquet` (old line parked); `experiments/master_table_close.py` `PAPER_SWITCHED` (asserted sha256 gate on the table; the campaign before/after scores the switched row on its own table; other `yhat_tree<NN>` dumps are not scored); `results/spxw_pnl/MANIFEST.md` row; `experiments/feature_importance_1530_trees.py` XGBoost settings → `xgb_opt_00` (old dict parked), `BUCKET_ID` pinned to the stored draws' ids. The live card does not read the `xgb` tag (live tests 76 passed).
- **Regenerated (MACROS_XGB10_DONE, MODELSET_XGB10_DONE):** master table (paper and GUI; 1,912 asserted gates, 0 failed), bounds, both makers, D.16–D.18 (27/27, 13/13, 7/7), the importance study's XGBoost refits on three input sets (CARC epyc-7513) with aggregate and before/after, the lean-set macros, the QLIKE–Sharpe figure (unchanged), the rv_iv notebook, the rule-by-strategy tables (15:30, re-pick, hold to close) and the running appendix.
- **Moved by macros only:** Table 7 panel A XGBoost row (QLIKE 0.1076 → 0.1063, Sharpe 1.12 / 0.62 → 1.26 / 0.77); `\cmPaperVsShortAboveSigN` two → three (XGBoost now significantly above always short); model-set medians and counts; D.16–D.18, lean-set and running-appendix XGBoost numbers; `\campFISpearmanTrees` 0.90 → 0.87.
- **Not regenerated:** the Murphy 15:30 table (`table_murphy_1530`, parked Section-5.4 appendix, frozen under the session-bar recalibration). **Text not edited** (methods sentences on XGBoost's settings: text agent).
- **Build** (scratch, `-jobname=xgb10`): see the report of this change.

### 8.18 2026-10-08: the lean set parked (replaced by an MCS over the features)

- **Decision (user, 2026-10-08):** the lean set (`lean_vol_vix`) is parked everywhere. The professor suggested a Model Confidence Set over the features instead, which is being built separately; there is no text for it yet. The deferred tree-text round (tree-top entries) was NOT applied.
- Every parked line is headed `% 2026-10-08 parked (lean set; replaced by an MCS over the features):`. Changes, after MACROS_NOLEAN_DONE (model set 112: paper 9, baseline 52, all_features 51):
  - `appendix_lean_set.tex`: every line commented out.
  - `appendix_campaign_16h.tex` (D.19): the `\input{sections/appendix_lean_set}` and the "follow-up to the importance study" comment are parked (the input was in D.19, not in `main.tex`). The conventions now name two input sets, back to the pre-lean wording.
  - `methods_close_option.tex`: "Three input sets" becomes "Two input sets", and the `lean_vol_vix` item is parked.
  - `introduction.tex`: "on three input sets --- the third a lean set chosen after the importance study" becomes "on two input sets".
  - `results_close_option.tex` (5.4):
    - model-set medians by input set: the lean median and the selection caveat are parked;
    - figure caption: "and the lean set are not drawn" is dropped;
    - ladder: the lean per-bar step (`\cmLdPerBarLean…`) and the lean input steps (`\cmLdInBaseLean…`, `\cmLdInLeanAll…`) are parked;
    - tree counts: "three input sets" becomes "two input sets" (twice).
  - D.16–D.18 had no lean wording; their numbers moved by macro.
- Checked: no live `lean_vol_vix`, `\fl…`, `…Lean…` or `sec:app_lean` remains in `sections/`, and the PDF text has no "lean set". The top-Sharpe forecast is named by macro (now the Optuna per-bar XGBoost on `all_features`, still not significantly above the ridge, claim list "holds"). Every FLIPPED tree claim is a count in the text.
- **Build** (scratch, `-jobname=nolean1`): 74 pages, 0 errors, 0 overfull, 0 undefined references/citations, no multiply defined labels, no "??".

### 8.19 2026-10-13: the LSTM parked everywhere (double-standardisation bug)

- **Decision (user, 2026-10-13):** "get rid of LSTM for now". The per-bar LSTM (`specs/causal_tune_lstm.py`,
  `_jobs.py`) re-standardises, inside each window, a design the executor has already robust-prescaled
  (`standardise()`, `causal_tune_lstm_jobs.py:92-96`, guards only sd == 0). Near-constant columns such as the
  `*_avail_ma_*` fractions (kept raw by the prescale, `src/backtest/executor.py:161-164`) blow up (window sd ~1e-5
  -> |z| > 1,000; measured on the MinGRU trainer, `experiments/puffer_forecast/phase2b.py`). Parked header:
  `% 2026-10-13 (user): LSTM parked for now (double-standardisation bug):`.
- **Intraday LSTM (lstmi):** shares the bug in part. It calls the same `standardise()`
  (`causal_tune_lstm_intraday_jobs.py:71-78, 115, 160`) with no floor or clip, but on a step matrix that is never
  prescaled and has no `*_avail_ma_*` columns (`causal_tune_lstm_intraday.py:365-449`). Its 0/1 indicators are
  bounded (|z| at most ~150-310 for its sequence lengths); its median-filled continuous `adj_<x>` columns could blow
  up the same way on a feed's sparse stretches (not measured). Already parked since 2026-10-06; nothing renders.
- **What still rendered (2026-10-06 had parked the rest) and is parked now:**
  - `methods_close_option.tex`: "An LSTM was also fitted for the 16:00 bar; it is not reported ...".
  - `results_close_option.tex`, figure caption: "LSTMs are not drawn".
  - `appendix_campaign_16h.tex` (D.19): the intro's "the LSTMs of the re-run are not reported ..."; the compute
    paragraph's "(one stage also ran the LSTMs, which are not reported)".
  - D.19 compute table: the masked stage's row "trees and LSTM with the mask" becomes "trees with the mask"
    (`make_appendix_campaign_16h_tex.py` `_sacct(..., drop_stage="lstm")`: jobs `lstm_mask*`, `tm_gates_lstm`,
    `tm_gates_h_lstm` dropped; the full stage still checked against the agent's SUMMARY): allocated CPU-h
    186.4 -> 115.2, used 102.5 -> 57.3, peak 408 and wall clock unchanged. The repro table's source line of the
    parked intraday-LSTM row is parked.
  - Standalone master table documents (`make_master_table_close_tex.py`, not the paper): LSTM rows, the
    "(trees, LSTM)" figure group, SUMMARY lines naming an LSTM and the "both LSTMs" text parked; regenerated with the
    master table run of 8.20.
- **Statistics whose set could include the LSTM (checked):** every model-set statistic (counts, medians, ranks, top
  Sharpe, best tree, bounds, strategy variations, figure pairs, D.16-D.18) reads `camp.in_paper_set`, which drops
  every LSTM family (since 2026-10-06; `paper_model_set.csv` has 0 LSTM rows); D.19 counts are trees only
  (`TREES`, `campMTKeysNew`, `campHChunks` filter LSTM); no MCS / SPA / StepM, combination or pool in the paper
  contained an LSTM (the bucket study, 8.21, has none: asserted). Only the compute row above moved.
- **Not in the paper, left as they are:** `generated/appendix_campaign_16h_plain_pairs.csv` (the re-score record
  still holds the LSTM pairs); the standalone master table's "all table-A forecasts" rank correlation and its
  SUMMARY counts are still computed by `experiments/master_table_close.py` over table A including the LSTM rows.
- **Build:** see 8.21.

### 8.20 2026-10-13: the per-bar trees on their top entries (xgbtop, rf400)

- **Decision (user; MASTER_TABLE_REVIEW items 76-79, deferred 2026-10-07/08):** per-bar XGBoost on its top dev entry
  `xgb_opt_00` (token xgbtop, `results/trees_xgbtop_1600/`, 15 families) and the per-bar random forest on trial 255
  of the 400-trial search (token rf400, `results/trees_rf400_1600/`, 15 families) replace XGBoost `xgb_opt_06` (all
  families), the 87-trial dev forest `rfdev` (T1 / T10) and the default-start forest (RS / Optuna), in one
  regeneration. Both batches are complete locally (fixed, RS, Optuna, unmasked T1 / T10). Pooled trees untouched.
- **Code:**
  - `experiments/master_table_close.py`: `PERBAR_SWITCH` (paper run only: active when `OUT` is
    `results/close_master_table`; the GUI's run redirects `OUT` and is untouched). Every per-bar xgb / rf key on
    baseline / all_features / lean_vol_vix keeps its key and label and reads the replacement table (the GUI's rule,
    `rows.py` `_tree_swap_table`); 90 keys switched (asserted gate); new column `perbar_switch` in `master_table.csv`
    and `before_after.csv`; agent C's Optuna QLIKE gate is reported, not asserted, for switched keys. Unswitched
    rows equal the previous run exactly (QLIKE, Sharpe mid / crossed).
  - `make_appendix_campaign_16h_tex.py`: `in_paper_set` takes the `rf` keys back and drops `rfdev` (parked rule kept);
    `switched()` / `switched_unmasked()` replace `rfdev_table()` in every re-scored pair (masked ladder, masked -
    ridge, masked - unmasked where an unmasked top-entry run exists, C's Optuna pairs); a switched table must
    re-score to its master-table row (asserted). Masked - unmasked RS10 / RS1 of XGBoost / forest have no unmasked
    top-entry run: 8 pairs left out (`---`, `\campMaskDroppedN`). Budget / picks tables: LightGBM from agent C's
    report, the forest from the rf400 stage-1 records (`optuna_trees_report_1600.budget_tables` / `picks_table`,
    on the run's own re-centred depth ladder {3, 5, 10, 20, 40, None}); XGBoost's rows parked (xgbtop stage-1
    records not local). D.19 master before / after leaves out the 12 switched rows (`\campMTSwitchedN`). Picks
    table heads "base / live / all" corrected to "base / all".
  - `make_table_close_main.py`: Table 8's forest step reads `subtree_rf_all_features` ("shipped settings");
    `RF_CONFIG` -> `budget400/rf_config.json`; new `\cmTreeProvXgbTopName`, `\cmLdEstMinWho`, `\cmLdEstMaxWho`;
    mask count and budget rows follow D.19 (`camp.budget_points_paper()`).
- **Regenerated:** master table (1,877 asserted gates, 0 failed; standalone PDFs), bounds, Tables 7 / 8 and
  `close_main_numbers.tex`, D.19, the QLIKE-Sharpe figure, D.16 (27/27 gates), D.17 (13/13; run with a numpy-1.26
  shim `np.trapezoid = np.trapz`, the same rule), D.18 (7/7).
- **Key moves (before -> after, plain QLIKE / Sharpe mid):** XGBoost T1 baseline 0.0975 / 1.33 -> 0.0976 / 0.95,
  all_features 0.0942 / 1.64 -> 0.0935 / 1.12; T10 0.0985 / 1.11 -> 0.0974 / 1.20, 0.0967 / 1.75 -> 0.0970 / 1.55.
  Forest T1 (rfdev -> rf400) 0.0979 / 1.07 -> 0.0968 / 0.66, 0.0954 / 0.85 -> 0.0966 / 0.85; T10 0.0988 / 0.75 ->
  0.0982 / 0.98, 0.0974 / 1.75 -> 0.0982 / 1.27. Model set: median Sharpe 1.10 -> 1.08, rank correlation QLIKE vs
  Sharpe -0.32 -> -0.18, top Sharpe now the per-bar elastic net on all_features (still not significantly above R);
  Table 8 estimator range minimum LightGBM -0.05 -> random forest -0.15; figure: Sharpe higher 47 -> 50, QLIKE
  lower 37 -> 38, ES-change correlation -0.41 -> -0.26, QLIKE-change correlation -0.12 -> 0.02.
- **Text (parked under `% 2026-10-13 parked (user; ... top entries ...)`):**
  - methods: the XGBoost look-ahead disclosure parked; "LightGBM, XGBoost and the random forest each take their
    search's top entry"; the forest's search "over \cmTreeProvRfTrials{} trials" (87 -> 400); "Both were run before
    the random forest's search ... library defaults" parked. Refit-contract paragraph: "indices 00 and 16 ... why the
    pick of index 16 cannot be certified" (stale since 8.17, and its pointer target parked) -> "two entries of a
    frozen twenty-arm menu, the top LightGBM and the top XGBoost entry".
  - 5.4: claim "daily refits lower every tree's QLIKE" FLIPPED -> "lowers the QLIKE of \cmCadQBelow{} of the
    \cmCadN{}"; mask qualifier (library-default forest) -> "XGBoost and the random forest were re-run unmasked with
    the shipped settings only"; trial budget "of LightGBM and the random forest"; estimator range names its ends by
    macro (was typed "LightGBM").
  - D.19: intro sentence on the re-run of XGBoost / forest; mask paragraph and caption (`---`); Optuna trial 1 (no
    forest qualifier); budget / picks paragraphs and captions (LightGBM and forest; `reg_lambda` parked); repro
    "these checks predate XGBoost's and the random forest's current settings"; master before / after exclusion
    sentence; importance: "the new side runs their current settings ... mixes the settings with the design and
    the mask" (`results/feature_importance_1530_dedup/SUMMARY.md` settings line).
- **Pending:** XGBoost's in-sample budget and picks rows need the xgbtop stage-1 records
  (`results/linear_subsection_trees_optuna_mask_xgbtop/stage1/{baseline,all_features}/bar1600/xgb/tw2000/trials_bar1600.npz`,
  on CARC under `/scratch1/jc_905/harxhar-xgbtop/`); unmasked RS10 / RS1 of both trees were never run.

### 8.21 2026-10-13: the bucket story check (new appendix subsection, B.17)

- **Decision (user):** write up the bucket results (MASTER_TABLE_REVIEW item 92; `results/feature_mcs_2026-10-08/`
  "The bucket story check"). New `writeup/make_buckets_tex.py` (reads `buckets_{pairwise,stepm,spa,verdicts,gates}.csv`,
  `bucket_losses.parquet`, `mcs_sets.csv`; asserts every qualitative claim: the story holds in no cell, StepM names
  all features nowhere, every named bucket is `market_vw` on a linear family, the trees' significant trade gains
  are single buckets above all features, no LSTM in any estimator set or pool) -> `generated/buckets_numbers.tex`
  (90 `\bk…` macros) and `generated/table_buckets_story.tex`.
- **Text:** new `sections/appendix_buckets.tex` (Question / Design / Tests / Finding / Reading, Table
  `tab:app_buckets`), input at the end of `appendix_campaign_16h.tex` where the parked lean subsection was; pointer
  paragraph "Bucket by bucket." in 5.4 after the QLIKE-Sharpe figure (macro file input in vertical mode at the top).
- **Numbers (CSV):** 0 of 30 family-loss cells hold the story; StepM (5 %, every block) names `market_vw` in 4 linear
  cells, all features in none; lasso + `market_vw` QLIKE -3.4 % (DM -3.75, SPA p <= 0.002); XGBoost trade 0.95
  (HAR + cal.) / 1.12 (all, t 0.30) / 1.67 (liquidity, t 2.05) / 1.65 (`market_ew`, t 2.39; StepM at blocks 10 and
  21, not 5); forest 0.66 / 0.85 (t 0.34) / 1.47 (moments, t 2.35) / 1.47 (`market_ew`, t 2.38); six-estimator
  pool all features DM -0.65, Sharpe 1.67 vs 1.22 (t 1.00); pool without LightGBM, dropping liquidity 1.64 -> 2.03
  (t 2.73). The review's "the forest's all-features design is its weakest trade design" does not hold in the CSV
  (HAR + cal. 0.66 and two add-bucket designs 0.68 are lower) and is not in the paper.
- **Build** (scratch, `-jobname=after7` + this pass): 77 pages (74 before the three changes), 0 errors, 0 overfull,
  0 undefined references / citations / control sequences, no multiply defined labels, no "??", no "LSTM" in the
  PDF text.

### 8.22 2026-10-14: the bucket write-up parked (user)

- `sections/appendix_campaign_16h.tex`: the `\input{sections/appendix_buckets}` line is commented out;
  `sections/results_close_option.tex`: the "Bucket by bucket." paragraph is commented out. Both carry the dated note.
  The generator, macros, table and section file from 8.21 are kept, so un-parking means un-commenting the two lines.
- **Build** (scratch, `-jobname=main_nobuckets_2026-10-14`): 75 pages, 0 errors, 0 overfull, 0 undefined
  references / citations / control sequences, no multiply defined labels.

### 8.23 2026-10-14: XGBoost's Optuna trial records on its top entry (xgbtop) back in D.19 and 5.4

- **Input:** the xgbtop stage-1 records, pulled from CARC and md5-verified by the coordinator:
  `results/linear_subsection_trees_optuna_mask_xgbtop/stage1/{baseline,all_features,lean_vol_vix}/bar1600/xgb/tw2000/trials_bar1600.npz`
  (md5 07fe18af… / f0699a03… / db004fd0…; local md5sum agrees). This closes the item left pending in 8.20.
- **Code:** `make_appendix_campaign_16h_tex.py` `_top_entry_trials()` reads every top-entry run (`TOP_OPT`: xgb ->
  xgbtop, n_estimators 367 asserted; rf -> rf400, 485) with agent C's `budget_tables` / `picks_table`, each on its
  run's own search space from the records' meta (xgbtop's `reg_lambda` floor is 1e-5, rf400's depth ladder is
  re-centred; the campaign's default spaces would misplace the picks). The budget and picks tables take XGBoost's
  rows again (`TREES`), the lambda assertion is back to `lambda_l2` + `reg_lambda`, the source lines name both runs.
  `make_table_close_main.py`: the budget count asserts all three trees again (`n_tree * nb`). The 2026-10-13 lines
  are parked, not deleted.
- **Re-run:** `make_appendix_campaign_16h_tex.py`, `make_table_close_main.py` (same 11 FLIPPED claims as before,
  none new; the "late trials win more than chance" assertion holds with XGBoost in).
- **Numbers:** XGBoost's rows (best in trials 41-50: 36.5 % baseline / 33.6 % all_features; shipped best at
  0.3 / 0.2 %) fall inside the LightGBM / forest ranges, so one macro moves:
  `\campOptGainTwentyFiveMedHi` 1.1 -> 1.2. `\cmBudLastMin/Max` (27 / 37), `\campOptLate…`, `\campOptPickDiff…`,
  `\campPickLambdaHighMax` (0.0) and `\campPickLambdaPriorLo` (5.1) are unchanged.
- **Text (the 2026-10-13 interim wording parked under `% 2026-10-14 (coordinator; xgbtop stage-1 records pulled)`):**
  D.19 "Are 50 trials enough?" back to "each arm"; "Where the picks sit" back to "The penalties `lambda_l2` and
  `reg_lambda` …; LightGBM's `min_child_samples` …"; budget caption back to the original; picks caption "Each tree's
  axes are its own run's search space."; 5.4 trial-budget sentence back to "the best validation score falls in the
  last ten of the 50 trials". The bucket write-up stays parked (8.22): its two lines were not touched.
- **Build** (scratch, `-jobname=main_xgbtrials_2026-10-14`): 75 pages, 0 errors, 0 overfull, 0 undefined
  references / citations / control sequences, no multiply defined labels, 21 underfull boxes (unchanged), no "??",
  no "LSTM" in the PDF text.

## 9. 2026-10-06: the multi-horizon subsections without recalibration

**Decision (user), same as section 8:** no recalibration anywhere (including the Duan smearing
back-transform); every forecast is used as its plain back-transform; no intervals (point
estimates; DM statistics may stay); no `adj_` in shown variable names.

**Scope:** `sections/methods_multihorizon.tex`, `sections/results_multihorizon.tex`, their
generated inputs and the producers of those inputs. Both sections are still parked from the
rendered paper (`\iffalse` around their `\input` in `methods.tex` and `results.tex`, since
2026-08-28). That parking was not changed.

### 9.1 Inventory

None of the numbers in these subsections had an interval, so no interval macro needed parking.
Recalibration entered the numbers in three places.

| Object | Producer chain | Where recalibration entered | Action |
|---|---|---|---|
| Rest of session (M0 to M3, `\mh...`, Tables `multihorizon_pooled`/`_byhour`) | `experiments/multihorizon_ttc.py` → `results/multihorizon/ttc_*.csv` → `experiments/multihorizon_tex.py` | `_causal_scale`: every construction, M0 included, was multiplied by c_h, the expanding mean of realized/forecast over earlier sessions. The per-bar inputs were already plain (`yhat^2 * baseline`). | Plain by default. Re-run from the stored per-bar exports `results/spxw_pnl/yhat_{a0,blk2}.parquet`. The cheap log-linear maps were refitted; no model was refitted. |
| Direct per-bar path (`\dmh...`, `tab:dense_byk`) | stored shards `{a0,blk2}/shard_*of8.npz` (untracked; read from the sibling checkout `harxhar-clean/results/direct_mh`) → `experiments/direct_multihorizon_reduce.py` → `experiments/direct_mh_tex.py` | The old contract: a per-window MZ mean map on the square-root scale, an EWMA(0.6) second moment, and a first chunk used only as calibration burn-in. The per-horizon target was the winsorized `y_true^2 * B`. | Re-scored from the stored `yhat`: `f = yhat_k^2 * B_k` against the unwinsorized `rv_raw_k` (the paper's `qlike_per_bar` rule), on every origin. |
| Direct h-step profile (`\mhp...`, `tab:multih_profile`, already parked 2026-08-19) | `results/multih/h*.npz` → `experiments/multih_probe_tex.py` | Scored under the contract. The npz files store only contract losses, not forecasts. The probe was also retired on 2026-09-17 for a leak. | Cannot be re-scored without refitting. The `\input{generated/multih_numbers}` is parked, and the generated file is left untouched. |

### 9.2 Producers changed

- **`experiments/multihorizon_ttc.py`**
  - The default is now plain: exp() of the fitted log value, with no rescale.
  - `--recal` (which requires `--tag`) reproduces the archived run. Re-running it to scratch gives `ttc_{byhour,pooled,coefs}.csv` byte-identical to the archive, and an identical forecasts parquet.
  - Archive of the old default outputs: `results/multihorizon/recal_2026-09-24/`.
  - The re-run used `--save-forecasts`. `multihorizon_frem_check.py` passes byte for byte.
  - The full-sample coefficients are unchanged.
- **`experiments/direct_multihorizon_reduce.py`**
  - The default is now the plain scoring.
  - `--recal` (which requires `--out`) reproduces the archived CSVs exactly (max |diff| 0).
  - New flag `--shard-dir`.
  - New diagnostics: medians, the share of the ridge's loss carried by its ten largest losses, and `dense_nonpos.csv`.
  - Archive: `results/direct_mh/recal_2026-08-18/`.
- **`experiments/direct_mh_tex.py`**
  - New macros: `\dmh[Rth]{NK,BlkWorseN,BlkSigN,MedBlkWorseN,TopTenMin,TopTenMax}`, `\dmhAzeroNonposN`, `\dmhBlkNonposN` and `\dmhBlkNonpos{First,Last}`.
  - No macro was renamed or removed.
- **`experiments/multihorizon_tex.py`:** header comments only. The macro set is unchanged (94).

**Shared-producer note.** The running appendix's `ttc_close_fomc1_*` variant (owned elsewhere) is
built by the same `multihorizon_ttc.py`. Re-running its usual command now gives the plain version.
It was not re-run here, so its CSVs and parquet are still recalibrated.

### 9.3 Key numbers, old (recalibrated) → new (plain)

Rest of session, pooled, all sessions:

| Quantity | Old | New |
|---|---|---|
| Sessions | 5,587 | 5,650 (no 63-session rescale burn-in) |
| Benchmark QLIKE | 0.8409 | 1.0006 |
| M1, a0 / blk2 | 0.1837 / 0.1740 | 0.1932 / 0.1839 |
| M2, a0 / blk2 | 0.1814 / 0.1719 | 0.1892 / 0.1796 |
| M3, a0 / blk2 | 0.1761 / 0.1674 | 0.1841 / 0.1755 |
| DM blk2 vs a0, M1 / M2 / M3 / M2 pooled | −4.2 / −5.4 / −5.9 / −5.3 | −3.7 / −5.0 / −5.4 / −4.9 |
| DM blk2 vs benchmark | about −20 | about −15.3 |
| Pooled vs per-hour (blk2) | +0.3 | +0.5 |
| By hour (M2): blk2 better at | 11/11, DM −6.3 to −2.8 | 11/11, DM −5.6 to −2.5 |
| 2016–2024: benchmark | 0.9044 | 1.0799 |
| 2016–2024: DM blk2 vs a0, M1 / M2 / M3 | −3.2 / −3.9 / −4.0 | −3.0 / −3.7 / −3.7 |

Every claim in the text still holds.

Direct per-bar path:

| Quantity | Old | New |
|---|---|---|
| Origins (n) | 273,371 | 275,688 (no burn-in chunk) |
| a0 QLIKE, t+1 / t+11 | 0.1928 / 0.2160 | 0.2803 / 0.3326 |
| blk2 QLIKE, t+1 / t+7 / t+11 | 0.1979 / 0.2229 / 0.2283 | 0.4667 / 40.9865 / 3.3048 |
| DM range | +11.2 to +21.0 | +1.0 to +2.6 |
| DM range, RTH origins | +3.1 to +11.7 | +1.0 to +3.7 |
| Horizons significant at 5 % | 11/11 | 3/11 (RTH 4/11) |

- **Why the ridge's plain mean is erratic.** At 91 origins, from February 2018 to April 2020, the ridge's square-root-scale forecast is ≤ 0 at some horizon. The incumbent has none. Squared, such a forecast puts almost no variance on the bar, and QLIKE's penalty for under-forecasting is unbounded. Depending on the horizon, the ten largest losses carry 15–99 % of the ridge's summed loss.
- **The median orders the arms the same way** at 11 of 11 horizons.
- **Cumulative path (11 bars), not in the text:** DM +15.9 → +7.3.

Parked claim, re-checked on the plain forecast in scratch. The per-bar-refit diagnostic
(`blk2_refit1`, first eighth of the panel) used to read "narrows the gap … does not change its
sign". Plain, the per-bar refit gives the ridge the lower loss at t+1 and t+2 (dQ −0.0028 and
−0.0031, DM −1.1 and −1.0). From t+3 on its loss is higher (DM +0.9 to +2.6), against +0.0 to
+3.0 under the daily refit. The claim fails as worded, so it is parked. No macro exists for it.

### 9.4 Text edits

Old text is kept under `% 2026-10-06 parked (recalibration removed, user decision):`. No number
was typed.

**`methods_multihorizon.tex`**

- Header: the plain convention.
- `\input{generated/multih_numbers}` parked (9.1).
- A dated note on the parked profile paragraph A.
- "The rest-of-session panel":
  - \mhNDays now reads "scored on … every complete session after the first 63".
  - The per-bar inputs are "each used as its plain back-transform … ŷ²B" (before: "exactly as scored in Section three_block_results").
- Constructions: the causal QLIKE-optimal factor becomes "exponentiation alone … no smearing factor, rescale or other recalibration", plus a one-line Jensen note: "tends to forecast low; the tilt is left in".
- Benchmark: "back-transformed the same way — the best constant forecast" becomes "exponentiated in the same way and with nothing on top — a constant forecast".
- Per-bar path: "Scoring is the paper's contract (sec:smearing_contract)" becomes the plain back-transform against the target bar's realized variance. "returned to in the results" is dropped.

**`results_multihorizon.tex`**

- Header comment, and a dated note on the parked profile.
- "The rest of the session" adds "every forecast its plain back-transform with no rescale".
- Table `multihorizon_pooled`:
  - The caption reads "exponentiated, no rescale".
  - The tabular is now wrapped in `\resizebox{\textwidth}`. Unparked, it overran the text width by 199 pt; this predates the change.
- The per-bar path paragraph is rewritten on the new macros:
  - mean worse at `\dmhBlkWorseN`/`\dmhNK`;
  - DM `\dmhDMmin`–`\dmhDMmax` (RTH `\dmhRthDMmin`–`\dmhRthDMmax`);
  - significant at `\dmhBlkSigN` (`\dmhRthBlkSigN`);
  - the near-zero-forecast diagnosis (`\dmhBlkNonpos*`, `\dmhTopTen*`);
  - the median check `\dmhMedBlkWorseN`;
  - one sentence that the refit diagnostic is not re-measured.
- Parked from that paragraph: "in every era of the panel separately", which no producer output supports, and the refit parenthetical.
- Table `dense_byk` caption: "QLIKE under the evaluation protocol" becomes the plain back-transform, with a note that the two-block column is dominated by a few near-zero forecasts.

### 9.5 Outside these two sections (not edited)

These sentences describe the multi-horizon rescale or its intervals:

- **`sections/appendix_running.tex`:**
  - :636–640 (`sec:app_run_multihorizon`, rendered): "fitted per entry clock on strictly earlier days (at least 63) and rescaled by the running mean of realized over forecast". Its numbers come from `ttc_close_fomc1_*`, which are still recalibrated.
  - :153–160: the rest-of-day maps are "the constructions of Section app_run_multihorizon", so they inherit that rescale.
  - :173 and :193: the maps' table caption ("95 % moving-block bootstrap percentile interval") and `\runappMapIntervals`.
  - :338–342 and :358–360: the direct rest-of-day trading test ("a recalibration per entry clock", interval counts).
  - :646 is a comment that calls the default ttc outputs "the plain variant". Its meaning clashes with the new convention.
- **`sections/methods_options.tex` :64–80** (parked via `\iffalse` in `methods.tex`): `eq:opt_ttc` with "c_h … the causal QLIKE-optimal multiplicative rescale".

### 9.6 Build

All builds went into the scratchpad with their own `-jobname` and `-aux-directory`. `writeup/main.*` and the tracked aux files were not touched.

| Build | Pages | Errors | Undefined refs/cites | Multiply defined | Overfull |
|---|---|---|---|---|---|
| Paper as it stands (`mhplain_main`) | 95 | 0 | 0 | 0 | 0 |
| Scratch copy with both sections unparked (`mhplain_unparked`), after the resize | 100 | 0 | 0 | 0 | 0 |

The generated files of other workstreams were taken as they stood at about 15:15.

## 10. 2026-10-06: the main results without recalibration (re-scored from stored outputs)

User decision 2026-10-06: recalibration and intervals leave the paper; every forecast is its plain
back-transform `f = yhat^2 * B` (adjusted-scale forecast squared times the diurnal baseline). This
section covers the MAIN RESULTS only (methods.tex, results.tex, appendix.tex, the abstract, their
generated inputs). Same day: live-feasible removed from the paper (user decision) -- nothing in this
scope reported a live-feasible row; the only reference (the unrendered proposed abstract sentence about
the last-30-min headline) is parked with that tag.

**Re-score (no re-fitting).** `experiments/score_unification.py` gains `--convention {plain,contract}`
(default `plain`) and `--arms` (regex; a full two-root pass holds ~6 GB of per-bar arrays). Under
`plain` the scored forecast is `yhat^2 * B` from the stored per-chunk npz arrays (yhat, baseline,
rv_raw); DM tests, increments, the tree-bank selector and hedge (fed plain tail losses) all read it;
the conditional-variance sidecar arms are not assembled; the smear-sensitivity table and its tau
macros are not written. Row set unchanged (first present window of each arm still held out), so every
n in the text stands. Run on the paper's arms over the two harvest roots (read from the paper2
checkout, raw-harvest cache hard-linked into `results/.harvest_cache/`):
`results/unification_plain_2026-10-06/` (scores CSV, increments, tree selection, `plain_vs_contract.csv`
= the old -> new audit, `score.log`; `fomc1/` = the FOMC root). New exhibits: tree-bank macros
(`\unifTreeBank...`, experts split by their frozen menu name read from the chunk meta),
`\unifAzeroMedAbsErr`/`\unifAzeroMaxAbsErr`, and the paired penalty-family tests
`\unifIncrBTwoLassoTuned{DM,DQ}`, `\unifIncrBThreeEnetTuned{DM,DQ}`.
Check: `results/spxw_pnl/yhat_a0.parquet` equals the stored benchmark chunks position for position
(yhat, baseline, rv_raw bitwise); `yhat^2 * baseline` from the table equals the scorer's plain
forecast exactly; pooled plain QLIKE recomputed independently 0.240974 = the macro.
Generated: `writeup/generated/campaign_numbers.tex` (plain; arms outside the run render `\pending`,
none of them in live text), `table_buckets/lasso_ridge/blocks.tex` (not input anywhere);
`writeup/generated_fomc1/table_fomc_bucket.tex` via `writeup/make_fomc1_data_table.py` (new `--src`,
default the plain FOMC CSV). `table_smear_sensitivity.tex` left as is (parked, not input).

**Headline numbers, old (calibrated) -> new (plain).** Benchmark 0.22521 -> 0.24097. Two-block (1,100)
0.22350 (DM -3.1) -> 0.23885 (DM -1.8, no longer significant); two-block per-block tuned 0.22224 (-5.6)
-> 0.23633 (-3.8). Tuned ridge (fine) 0.22452 (-1.1) -> 0.24306 (+1.0); tuned lasso 0.22566 (+0.5) ->
0.24238 (+0.6); tuned enet 0.22464 (-0.6) -> 0.24008 (-0.4); lasso - ridge paired -0.00068 (DM -0.5),
enet - ridge -0.00298 (DM -2.0): ridge no longer beats lasso. Fixed-ridge sweep 0.1..300: 0.28959,
0.26750, 0.25403, 0.24698, 0.24303, 0.24101, 0.24046, 0.24204 (minimum at alpha = 100, the only point
below the benchmark, DM -0.3); tuned ridge lands at the alpha = 10 level. Trees: hedge 0.21296 (-15.3)
-> 0.22207 (-17.7), selection 0.21710 (-4.5) -> 0.22813 (-5.6), hedge - selection -0.00606 (DM -5.8);
hedge - best expert +0.00113 (DM +1.2: the hedge no longer beats its best member, which is a Sobol
control); hedge - two-block ridge on common rows -0.01131 (DM -10.6). Experts: 18 of 20 beat the
benchmark (all 16 Optuna-selected, 2 of 4 Sobol). Unpenalized joint fit 0.22939 -> 0.92974; FOMC
bucket (OLS) 0.22400 (-3.6) -> 0.24409 (+0.5). Kendall tau of the plain vs old arm ranking 0.60.

**Text.** methods.tex: the protocol defines the back-transform once, new subsection "The
Back-Transform" (`eq:backtransform`, plain `yhat^2 B`; the labels `sec:smearing_contract`,
`sec:smearing`, `sec:smearing_bias`, `sec:appendix_backtransform` live there so surviving
cross-references resolve); the calibrated protocol, three-convention comparison and its table parked;
typed error figures -> macros; the block-ridge motivation's battery `t = +5.0` parked. results.tex:
head-to-head rewritten (families not separated; enet below both; paired row added to Table
lasso_vs_ridge); sweep typed levels -> macros, reading rewritten; ridge-beats-lasso diagnosis and the
principal-component loss claim parked; tree bank rewritten on macros (table too); "bank beats its
members" -> "weighting beats selecting"; comparison against the linear ladder now paired against the
two-block ridge; expressivity ladder, product distillation and the pairwise specification statement
parked (earlier builds, retired back-transform, not re-scorable); block-ridge paragraph states the
non-significant DM and the tuned variant. appendix.tex: back-transformation appendix, protocol note,
conditional second-moment rows and the convention-sensitivity appendix parked; master-table wording.
Abstract: plain back-transform sentence; Duan finding, "optimally", "small beside that information
effect ... pairwise order" and "strong paired significance" parked and rewritten.

**Open (not changed here).** 65 scored bars carry the diurnal-baseline sentinel B = 1 (no history)
and score per-bar QLIKE ~12.7; on the plain benchmark they lift pooled QLIKE from 0.23806 to 0.24097.
The paper2 checkout's scorer has an uncommitted `_valid_of` exclusion for them; adopting it changes
every n. Introduction / conclusion / related works / data_prep sentences on smearing are listed for
their owner in the agent report, not edited here.

## 11. 2026-10-06: the running appendix (D.1–D.13) without recalibration, intervals or live-feasible

**Decisions (user, 2026-10-06):** no recalibration anywhere (for the running appendix, the
notebooks' own session-bar Mincer–Zarnowitz map); no intervals (point estimates; DM / HAC t and
Holm p stay); sizing study = sign(s), robust clip `clip(s / median|s| of earlier days, -1, 1)`,
rank of |s| (proportional and centred rank parked); live-feasible is no longer part of the paper;
headline references marked `TODO-HEADLINE`.

**Library switch** (`notebooks/atm_straddle_lib.py`): `RECALIBRATE = False`; every loader takes
`recal=None|True|False` and returns the plain `rv_hat = yhat^2 * B` by default
(`plain_back_transform`, `plain_mz`: m = yhat, s2 = 0). The map (`_mz_day_coefs`,
`second_order_raw`, `second_order_mz`) is parked with a dated comment and callable through
`recal=True`. Exception: `load_yhat_1530_mz_cached` keeps `recal=True` by default because the live
card calls it (`live/close_signal/forecast.recalibrate`); the live tests pass (7/7).

**Re-executed** (chain lock held, one at a time): `atm_straddle_rv_iv` (141 s), `atm_straddle_intraday`
(34 s), `atm_straddle_intraday_holdclose` (43 s); `make_rule_by_strategy_intraday_tex.py
--start-1000` (73 s; new opt-in flag keeps the re-pick bundle at 10:00–15:30) and `--dh-holdclose`
(111 s); `multihorizon_ttc.py --through-close --blk2-file yhat_blk2_fomc1.parquet --tag
_close_fomc1 --save-forecasts` (5 min, now plain); `restofday_trading_test.py --out
.../trading_test_plain --score-csv .../linear_subsection_restofday_dedup/score_rest_of_day.csv`
(2 min 45 s); `strategy_variations_1530.py` (33 s); `make_rule_by_strategy_tex.py`;
`make_appendix_running_tex.py`. Snapshots of the recalibrated outputs:
`results/{atm_straddle_0dte_1530,atm_straddle_intraday,atm_straddle_intraday_holdclose}/recal_2026-09-24/`,
`results/multihorizon/recal_2026-09-24_close_fomc1/`, `results/strategy_variations/recal_2026-09-29/`.

**Gates converted or repaired:** rv_iv §10 GATE 1 no longer compares stored bootstrap intervals
(bootstrap commented out 2026-10-03); GATE 2 holds to 1e-6 relative (the per-bar tables were rebuilt on
the de-duplicated design on 2026-09-29); new old-against-new cell (`recal_vs_plain_sign_s.csv`). Intraday
`RECORDED` forecast-dependent values are printed old → new, not asserted. The intraday maker's
deck gate reads the re-run deck CSV and compares only the rows it has.

**Key numbers, recalibrated → plain:** block-diagonal ridge sign(s) at 15:30 1.338 → 1.281 (buys
40.0% → 27.0% of days); re-pick sign(s^m) 2.34 → 2.57 (crossed −1.90 → −1.69); hold-to-close
sign(s) 2.40 / 1.35 → 2.98 / 1.96 (always short 2.93 / 2.21, unchanged) — sign(s) now ahead at the
midpoint, behind crossed; rest-of-day maps: stretch QLIKE 0.2018 → 0.2629, M1 / M2 / M3 0.2093 /
0.2132 / 0.2009 → 0.2393 / 0.2378 / 0.2235 (now all better than the stretch), M1 ΔSharpe −0.60 →
−0.09; per-bar vs pooled session QLIKE −9.0..−3.3% (7) → −10.0..+99.8% (4; the all-features ridge
has one extreme bar); multi-horizon blk2 M1 / M3 0.1435 / 0.1367 → 0.1555 / 0.1463; sizing (ridge,
after a year) sign(s) 1.54 → 1.55, robust clip 1.64 (+0.09), rank 1.59 (+0.04; was +0.05), clip
ahead for 8 of 12 forecasts, rank 7 of 12; strategy variations straddle (block ridge) 1.34 / 0.87 →
1.28 / 0.80; the trading test's plain rows are unchanged (they never used the map).

**Text** (`sections/appendix_running.tex`, old text parked under `% 2026-10-06 parked (...)`):
conventions state the plain back-transform and point estimates; D.2 finding and "why" rewritten;
D.4 two column sets (dedup scores), trading test plain-only, no intervals; D.5 reduced to its heading,
label and one paragraph (the main text cites it; the card studies are not re-scored); D.6 no longer
cites D.7; D.7 (decision time) and D.8 (VIX alone) parked whole; D.9 DM-based counts, live-feasible
rows out, rewritten finding; D.10 no rescale; D.11 the two new sizes; D.12 point estimates, the
live-feasible column out; D.13 regenerated, the headline (live-feasible) table parked inside the
generated file under `TODO-HEADLINE`. Parked macros: every `…Ci`, `runappMapExcl/Intervals`,
`runappPbSessCi/SixteenCi`, `runappPwGainCi/BestLo`, `runappCo{ExA,ExB,One,OneCi}`,
`runappTt{RidgeIntervals,RidgeZero,ExclFor,ExclAgainst}`, `runappTtLf*`, `runappRdLive*`, and the
D.5 / D.7 / D.8 macro families (their maker functions are no longer called).

**Build:** scratch `-jobname=runapp1006` with `-aux-directory`; bibtex and four passes; 74 pages,
0 undefined references or citations, no multiply defined labels, 0 overfull boxes.

**Not changed, for their owners:** the main-text sentences that cite D.5 as reporting the live form
(methods_close_option.tex :100–105; results_close_option.tex :21–24, :849–850; introduction.tex
:160–161; conclusion.tex :91) and that state sizing "does not improve on its sign" under "that study's
own back-transform" (methods_close_option.tex :616–623; results_close_option.tex :682–684;
related_works.tex :133–135; introduction.tex :161–162; conclusion.tex :92). Readers of
`vrp_sized_rules.csv` that still ask for the parked rules: `make_master_table_close_tex.py`
(`vrp_sizing_table`), `experiments/close_kelly_sizing.py`, `make_vrp_sizing_tex.py`; the GUI
(`master_table_gui/rows.py`) maps rule strings, and the clip rows are named `(d) robust clip s/med|s|`.


## 12. 2026-10-06: main results without the 65 no-history bars

User decision 2026-10-06 ("yes, exclude the 65 bars"): bars whose persisted diurnal baseline is the
target transform's no-history fill (B = 1: fewer than five earlier same-slot observations, the first
five appearances of 13 slots that began printing after the panel start) are not scored.

- `experiments/score_unification.py`: named rule `DIURNAL_NO_HISTORY_BASELINE` / `_scored_mask`
  (valid &= baseline != that value), applied in every arm before scoring and in the burn-in state, so
  the bars leave every pooled loss and DM sample; `--keep-no-history-bars` reproduces the earlier
  row set. New macros: `\unifAzeroN`, `\unifAzeroNQlike`, `\unifBlkTwoUserNQlike`, `\unifTreeHedgeN`
  (row counts, no longer typed), `\unifAzeroTopTenSSEPct`, and `\unifAzeroFit{Rsq,MZalpha,MZbeta,MSE,MAE}`
  (the benchmark table's square-root-scale rows, against the unwinsorized realization sqrt(RV/B); the
  typed 0.417 / -0.034 / 1.039 / 0.1397 / 0.2208 came from an earlier build and do not reproduce from
  the stored outputs).
- Re-run (main + FOMC roots) -> `results/unification_plain_2026-10-06/`; regenerated
  `writeup/generated/campaign_numbers.tex`, `table_{buckets,lasso_ridge,blocks}.tex`,
  `writeup/generated_fomc1/table_fomc_bucket.tex`. Gate passed: all 978 macros shared with
  `results/sentinel_b_impact_2026-10-06/main_without/tex/campaign_numbers.tex` identical; scores CSVs
  equal `main_without` (123 rows) and `fomc_without` (6 rows) at max |dQLIKE| = |dDM| = |dn| = 0.
- Key moves: benchmark 0.24097 -> 0.23806; two-block 0.23885 (DM -1.8) -> 0.23603 (-1.7); tuned
  two-block DM -3.8 unchanged; tree hedge vs two-block -0.01131 -> -0.01148 (DM -10.7); maximum
  absolute error 4.27 (a no-history bar, 2003-07-06 20:30) -> 1.7e-3; n 273,554 -> 273,489,
  272,971 -> 272,907, 263,782 -> 263,732. No sentence's reading changed (penalty sweep still
  falls to alpha = 100, the only point below the benchmark; tuned ridge ties alpha = 10; tree-bank
  counts unchanged).
- Text (methods / results only): the exclusion rule stated once in the back-transform subsection;
  typed n values -> macros (methods eval panel and table; results Table lasso_vs_ridge, tree bank,
  linear-ladder comparison, Table block_ladder); the error sentence now quotes the ten-largest share.
  data_prep's counts are for its owner.
