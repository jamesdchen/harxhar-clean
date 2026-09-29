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
