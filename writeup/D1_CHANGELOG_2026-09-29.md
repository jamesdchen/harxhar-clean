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
