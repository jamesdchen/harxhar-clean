# Master-table GUI review — 2026-10-01 (meeting with the professors)

Changes decided while going through `writeup/master_table_gui/` row by row, one item per decision, in the
order made. From item 3 on the user said to apply each change as it is listed (no batch at the end); items
1-3 were applied at that point. The GUI dir is untracked and shared with the other session on this branch:
nothing here is committed. The server must be restarted after a .py change (the page reloads index.html itself).

## 1. Add a t-statistic for the mean daily return (column) — APPLIED

- **Decision (user):** the trade columns need the t value of the mean return.
- **Which t:** Newey-West HAC t of the mean daily midpoint return against zero, from `asl.newey_west_t`
  (`notebooks/atm_straddle_lib.py:1098`). It is the function the master table already uses for
  `t_hac_mid_vs_ref`, and its automatic lag is 14 at 866 days. The plain t is not used: it equals
  Sharpe x sqrt(n/252) exactly, so it repeats the Sharpe column.
- **Columns:** `t_mean` (header "t(mean)", group trade, default on, placed after `mean`) and `t_mean_x`
  (header "t(mean×)", default off, placed after `mean×`). Format `+.2f`. Sort: descending first.
- **Where it comes from:** computed by rows.py, not read from the CSV. Add both fields to
  `_series_stats` (rows.py:419); that one function feeds the sign(s) rows, always short and every
  sizing row, so each row gets the t of its own position series and its own days (865 for sizing rows).
  Register the formatters in columns.py and add `t_mean` / `t_mean_x` to `SORT` in static/index.html.
- **Glossary (proposed):** "Newey-West t of the mean daily midpoint return against zero, lag 14 at
  866 days. Computed from the row's daily returns; not in the master-table CSV." Crossed: "Same t when
  the entry pays the spread."
- **Reference values** (computed 2026-10-01 in a scratch process from `master_table_daily.parquet`
  and, for always short, the quote book `daily_a0.parquet`; 866 days, lag 14):

  | row | mean | t HAC mid | t HAC crossed | plain t mid (= Sharpe x sqrt(866/252)) |
  |---|---:|---:|---:|---:|
  | Always short | +0.0145 | +0.43 | -0.58 | +0.38 |
  | HAR and calendar, OLS (a0) | +0.0829 | +2.53 | +1.48 | +2.17 |
  | Block-diagonal ridge (blk2, reference) | +0.0710 | +1.94 | +1.02 | +1.86 |
  | Per-bar ridge, live-feasible (headline) | +0.1342 | +3.54 | +2.66 | +3.53 |

## 2. Crossed-fill columns default off — APPLIED

- **Decision (user):** every crossed-fill column is off in the default view.
- **Change:** `"default": False` in `writeup/master_table_gui/columns.py` for `sharpe_crossed`
  (Sharpe×, line 156), `mean_crossed` (mean×, line 235), `ret_std_x` (std×, line 242) and
  `dsharpe_crossed_vs_short` (ΔSharpe× vs short, line 295). `d_vs_sign_x` is already off.
  The new `t_mean_x` from item 1 starts off.
- **Effect:** the default view drops from 22 to 18 columns (19 with item 1's `t_mean`). The crossed
  columns stay in the Trade picker and in the CSV export when ticked.

## 3. Rows: switch off the pooled 48-bar fits in one click (row filter) — APPLIED, option 1

- **Decision (user):** the families selection is wanting. It must be possible to turn off the models
  fitted on all 48 bars of the day, so only the closing regressions (fitted on the 16:00 bar) are compared.
- **Today:** the family boxes are include-only (nothing ticked = every family). Removing the pooled fits
  means ticking the other 17 families one by one. The pooled fits sit in two families: `paper`
  (a0, blk2, blk2_inc, lgbm, xgb, lasso_t, lasso_f, enet, b2lasso) and `pooled twin` (6 served; the 3
  live_feasible twins are already filtered out). That is 15 of the 153 sign(s) rows, plus the 24 sizing
  rows of the 8 paper forecasts that have them.
- **Change (a), a Fit facet:** a new "Fit" block in the Rows panel with two boxes, both ticked by default:
  "16:00 bar only" and "All 48 bars (pooled)". Unticking the second keeps only the closing regressions.
  rows.py derives a `fit` field once at load: family `paper` or `pooled twin` gives "All 48 bars
  (pooled)", every other family gives "16:00 bar only". Add `fits` to `facets()` and to `select_rows()`
  (same pattern as `rules`), and a "Fit" block plus a `fits` entry in `ROW_KEYS` in static/index.html.
  Classification notes a professor may ask about: the direct rest-of-day check rows are fitted at the
  15:30 clock and are bit-identical to their per-bar twins, so they are "16:00 bar only"; the
  intraday-sequence LSTM reads the last 13/48/96 half-hour bars as inputs but its target is the 16:00
  bar alone, so it is "16:00 bar only".
- **Change (b), families untick to remove:** every family box starts ticked and unticking removes that
  family, so any family can be dropped in one click. The server's family filter already keeps exactly
  the ticked list; only the page's initial state changes. Update the keys hint in index.html
  ("Clear those boxes to keep every listed key") to match.
- **Change (c), a Fit column:** identity group, default off, showing "16:00 bar" or "48 bars", so a
  mixed view still shows which rows are pooled.
- **Open, for the user — the pinned lead row when the pooled fits are off.** The pinned lead of every
  block, a0, is itself a pooled fit, and the GUI's DM column is computed against it (rows.py
  `_attach_accuracy`, `_BASELINE_KEY = "a0"`). Unticking the pooled fits would drop the DM zero while
  every DM cell is still measured against it. Options:
  1. Keep a0 pinned whatever the Fit filter says, marked as pooled (recommended: the DM column keeps
     one fixed meaning across every view).
  2. When the pooled fits are off, lead with the per-bar OLS on HAR + calendar (`sub_ols_baseline`,
     family per-bar linear: same estimator and inputs as a0, fitted on the 16:00 bar only; Sharpe mid
     1.33, recalibrated QLIKE 0.0975) and compute DM against it in that view. The same row's DM would
     then differ between views.

- **Applied as:** (a), (b) with All / None buttons on the Families block (unticking every box is refused with
  a status message, since an empty list means "no filter" to the server), (c). Open point resolved by the
  recommendation, option 1: a0 is exempt from the Fit filter only (unticking the paper family still drops it).
  Checked through the API after the restart: default view 238 rows as before; Fit "16:00 bar only" 203 rows
  (sign(s) 139, each sizing rule 21), a0 still leading all 4 blocks.

## 4. Live-feasible rows: a toggle, default off — APPLIED

- **Decision (user):** the live_feasible rows, which rows.py dropped at load, become toggleable, off by default.
- **Applied:** rows.py loads them and flags them (`live_feasible`); `select_rows` hides them unless the spec has
  `show_live_feasible: true` (a request with no spec hides them too). index.html: a "Show live-feasible" toggle
  (unticked at load) in the Rows panel, carried in the spec and the CSV export. With it on: 332 rows
  (sign(s) 223, each sizing rule 36). The T10 / RS10 tree families stay dropped at load, unchanged.
- **Note:** the headline (`sub_ridge_live_feasible`) is a live-feasible row, so it is hidden until the toggle is
  on; "Only headline" with the toggle off shows no rows. The HAR-ladder variants family is entirely
  live-feasible: its box is listed but shows rows only with the toggle on.

## 5. Explain the families paper, VIX-only, direct rest-of-day, implied-vol representations, chain-period buckets

- **Asked (user):** an explanation, given in the session from the code (research subagent). No GUI change decided.

## 6. Drop the "rule" family — APPLIED

- **Decision (user):** remove the "rule" family; the view is already grouped by rule.
- **Applied:** always short's family is blanked at load, so `facets()` no longer lists "rule"; the family filter
  keeps rows with no family, so always short is governed only by "Hide always-short".

## 7. Description column explains the rule — APPLIED

- **Decision (user):** the Description column must say clearly what each rule does.
- **Applied:** Description = the forecast's sentence (describe.py) followed by the rule's sentence (rows.py
  `_RULE_SENTENCE`, `_RULE_CORE`); always short shows its rule sentence alone. Sizing rows say where their gap s
  comes from: rows read from `vrp_sized_rules.csv` use the 15:30 deck's forecast (its 13-bar recalibration) and
  compare with the deck's own sign(s) on 865 days; per-bar lassos computed in rows.py use the sign(s) block's
  forecast on their own days. Description glossary now defines the straddle.
- Rule sentences as shown:
  - always short: sell one straddle at 15:30 every day and hold it to the close; no forecast enters.
  - sign(s): at 15:30 buy one straddle when this forecast of the 15:30-16:00 variance (recalibrated on the
    16:00 bar) is above the 15:30 implied variance, sell one otherwise; hold it to the close.
  - proportional: same direction as sign(s), but hold s / rms straddles (rms = root-mean-square of s over every
    earlier day); no cap.
  - rank of |s|: same direction as sign(s), sized 0 to 1 straddle by the share of earlier days whose |s| was at
    or below today's.
  - centred rank: hold 2F - 1 straddles, F the share of earlier days whose s was at or below today's; can buy
    when the forecast is below implied variance, and sell when above.

Checks after items 1-7: ruff check passes on rows.py / columns.py; node --check passes on the page script; the
mypy errors at rows.py `_score_sizes` (Optional sign Sharpe) and `_source_stamp` (tuple type) and the
ruff-format line-length diffs are in code that predates these edits, left as they are. Row-cache build at
server start is now ~8 s (was ~2 s): 79 more rows through the Diebold-Mariano tests and the sizing arithmetic.

## 8. Every stats / trade number from the de-duplicated pass — CHECKED, 4 sizing sources switched (APPLIED)

- **Request (user):** make sure the stats / trade numbers are from the de-duplicated pass.
- **Checked:** `master_table.csv` is the run `master_table_close.py --arm-root linear_subsection_dedup` of
  2026-09-30 08:30 (commit e4b83a0; working file = HEAD). Its gates: 2399, every asserted one passes; the 26 not
  ok are report-only comparisons with first-pass stored numbers (T10 trees, VIX-only trade), as SUMMARY.md says.
  `master_table_daily.parquet` (untracked, same run, 08:30) reproduces every row's days, Sharpe mid / crossed and
  recalibrated QLIKE to 2e-16. No `yhat_*.parquet` is newer than the run. The other session's uncommitted edit to
  the maker (one line, a `live_feasible` entry commented out) has not been run: the CSV is unchanged.
- **Found:** the sizing rows of four per-bar keys were read from the 15:30 deck's `vrp_sized_rules.csv` and
  `daily_<tag>.parquet` of 2026-09-24, i.e. the pre-dedup tables. Pre-dedup vs de-dup 16:00 forecast on the 866
  days: ridge baseline 1.2e-10, ridge all_features 4.9e-9, lasso all_features 9.1e-4 (70 days > 1e-6), elastic
  net all_features 6.5e-2 (53 days); no master-table position changed for any of them. The paper's eight sizing
  rows are pooled models the campaign did not touch: current.
- **Applied:** those four keys are parked out of `_VRP_TAG` (commented, dated) and sized in rows.py from
  `master_table_daily.parquet`, the path the per-bar lassos already use (`_DEDUP_RESIZED`). They are now on the
  research scorer, like their sign(s) rows; the paper rows' sizing stays on the deck scorer. Most of the change
  is that scorer switch (the ridge forecasts are equal to 5e-9). No sizing rule beats sign(s), before or after.
  Before -> after, Sharpe mid / gap to sign(s):

  | key | Proportional | Rank of abs(s) | Centred rank |
  |---|---|---|---|
  | sub_ridge_baseline | -0.48 -> -0.50 / -1.75 -> -1.74 | 1.26 -> 1.15 / -0.01 -> -0.09 | 0.92 -> 0.87 / -0.35 -> -0.36 |
  | sub_ridge_all_features | -0.00 -> -0.32 / -1.80 -> -2.00 | 1.34 -> 1.37 / -0.45 -> -0.32 | 0.64 -> 0.61 / -1.15 -> -1.07 |
  | sub_lasso_all_features | -0.36 -> -0.47 / -1.82 -> -1.94 | 1.37 -> 1.30 / -0.09 -> -0.17 | 0.99 -> 0.94 / -0.47 -> -0.52 |
  | sub_enet_all_features | -0.24 -> -0.46 / -1.83 -> -1.81 | 1.49 -> 1.33 / -0.10 -> -0.02 | 1.10 -> 0.94 / -0.49 -> -0.40 |

- **Not done (needs the user):** re-running the 15:30 deck notebook on the de-dup tables, which would regenerate
  `vrp_sized_rules.csv` and the deck books (many tracked files) and keep these rows on the deck scorer.
- **Also found (research subagent, describe.py wording vs code), not yet changed:** blk2 / blk2_inc say "a separate
  coefficient block for each time of day" but block-diagonal = two penalty blocks (HAR backbone at 1, exogenous at
  100) on one pooled coefficient vector; the rest-of-day sentence says "rather than a forecast of the 16:00 bar"
  but at 15:30 it is that bar; mfiv_only / ivslice_only also carry the HAR ladder and calendar; ivrep_target_scale
  keeps VVIX and VIX3M as levels; ivrep_innovations has change columns for VIX only; free_vix_only is cut from
  free_feasible_vol (keeps ES volume); a0 omits the base-2 ladder and session-edge interactions.

## 9. Park five families; new baseline row — APPLIED

- **Decision (user):** comment out paper, VIX-only family, direct rest-of-day at 15:30 (check), implied-vol
  representations and chain-period buckets: old, or not what the closing study is about now.
- **Fact checked for the user:** the paper models ARE the pooled 48-bar fits, and the 16:00 campaign did not
  re-run them (SUMMARY.md provenance: "not re-run", before/after "nothing"; tables from August). What was re-run
  is the per-bar (16:00-bar) linear models, trees and LSTMs. The pooled twin family (pooled 48-bar versions of the
  per-bar linear models, also not re-run) stays, and the Fit filter still controls it.
- **Applied:** rows.py `_PARKED_FAMILIES` (dated) drops the five at load, so their boxes, rows and computed sizing
  rows go. Because the paper family held the pinned lead row and the DM zero (a0) and the reference (blk2):
  `_BASELINE_KEY` (rows.py) and `BASELINE_KEY` (index.html) are now `sub_ols_baseline`, OLS on HAR + calendar
  fitted to the 16:00 bar alone; it also gets sizing rows so it leads every block (`_DEDUP_RESIZED`). DM glossary
  updated. The "Only reference" toggle is parked (its row is gone). describe.py no longer lower-cases OLS.
- **Result:** 179 rows loaded (was 332); default view 112 (sign(s) 93, each sizing rule 6), live-feasible on 179.
  Families left: per-bar linear, pooled twin, the ten daily tree rungs, LSTM, intraday LSTM, HAR-ladder variants
  (all live-feasible).
- **Note for the meeting:** every DM now compares with the per-bar OLS. The headline's DM is +0.69: its
  recalibrated QLIKE, 0.1005, is higher (worse) than the per-bar OLS's 0.0975, not significantly. Its Sharpe is
  1.90 against the per-bar OLS's 1.33.

## 10. Sizing rows for every tree and LSTM — APPLIED

- **Why they were missing (asked):** the GUI only showed sizing rows the 15:30 notebook had stored (its only
  nonlinear models were the paper's pooled LightGBM / XGBoost, parked in item 9) plus its own fill-in for per-bar
  lassos. Not an HPC limit: a rule needs only the forecast, the 15:30 implied variance and the straddle quotes, all in
  `master_table_daily.parquet` and the quote book.
- **Applied:** rows.py `_SIZED_FAMILY_PREFIXES = ("per-bar tree", "LSTM")` added to the computed-sizing selection
  (`_attach_lasso_rules`). Same arithmetic, scorer and 865 days as the other computed rows; point estimates only, no
  intervals. Each forecast's sign(s) positions from its own gap reproduce its master-table row exactly (checked).
- **Result:** 539 rows loaded; default view 352 (each sizing rule 86), live-feasible on 539 (each 130). Gap to sign(s)
  over the 80 default-view trees / LSTMs: proportional median -1.50, 0 of 80 above; rank of abs(s) median +0.00,
  41 of 80 above, best +0.48 (Optuna random forest, TUNE_PER 25, best of 25, HAR + calendar); centred rank median
  -0.34, 7 of 80 above. Live-feasible 40: -1.82 (0/40), +0.00 (20/40), -0.46 (0/40).

## 11. Rule explanation in each block's shaded header row, not in every row — APPLIED (supersedes item 7's layout)

- **Decision (user):** the rule's description belongs in the shaded header row of its block, in the cell under the
  Description column; each forecast row's Description describes the forecast only.
- **Applied:** rows.py `rule_texts()` (rule name -> text, from `_RULE_SENTENCE` / `_RULE_CORE` plus one shared note
  for the sizing rules); server.py returns it as `rule_text` with every table; index.html builds the header row with
  one shaded cell per column and puts the text under Description (also as the rule name's tooltip, for when the
  Description column is off). Per-row `rule_sentence` removed; columns.py `_description` = the forecast's sentence,
  plus a source note only on sizing rows read from the deck's stored file (none while the paper family is parked).
  Description glossary updated. The CSV export has no header rows, so it carries the forecast sentences only.

## 12. Header rows show formulas, shorter — APPLIED

- **Decision (user):** the block header descriptions show formulas and say less.
- **Applied:** rows.py `_RULE_HTML` / `rule_html()` (replaces the prose `_RULE_SENTENCE`, `_RULE_CORE`,
  `_COMPUTED_SOURCE`, `rule_texts()`); server.py sends `rule_html`; index.html sets it as the header cell's markup
  (constant strings from the server; italics and subscripts, serif for symbols) and its plain text as the rule
  name's tooltip. `_DECK_SOURCE` (row note for deck-file sizing rows, none shown now) shortened. Texts:
  - Always short: q_t = -1. Sell one straddle every day; no forecast.
  - sign(s): q_t = sign(s_t), s_t = f_t - IV_t. f_t: the row's 16:00 variance forecast, recalibrated; IV_t: the
    15:30 implied variance; q_t: straddles held from 15:30 to the close (+ buy, - sell).
  - Proportional: q_t = s_t / rms_t, rms_t = sqrt(mean of s_u^2 over u < t). No cap.
  - Rank of abs(s): q_t = sign(s_t) P_t(abs(s_t)), P_t(x) = share of days u < t with abs(s_u) <= x. abs(q_t) from 0 to 1.
  - Centred rank: q_t = 2 P_t(s_t) - 1, P_t(x) = share of days u < t with s_u <= x. q_t from -1 to +1; can buy
    when s_t < 0.
  - The three sizing rules add: Days 2 to 866 (865); Delta vs sign compares with sign(s) on those days.

## 13. Baseline row stands out; "Forecast" column renamed "Model" — APPLIED

- **Decision (user):** make the pinned baseline row stand out (e.g. say "baseline" in its name cell) and rename the
  first column from "Forecast" to "Model".
- **Applied:** index.html marks the pinned row with a "baseline" badge next to its name (tooltip: pinned first in
  every block, the DM column is measured against it), a blue-grey tint, a heavier lower rule and a bold name; the
  first column header and its tooltip say Model. The server now sends `baseline_key` (rows.py public `BASELINE_KEY`)
  and the page takes the pinned row from it, so the key lives in one place. CSV export: header "Model", and the
  baseline row's name ends "[baseline]".
- **Housekeeping:** the clock times once written after "APPLIED" in this file were estimates, not read from file
  times, and are removed.

## 14. QLIKE-vs-Sharpe chart below the table — APPLIED

- **Request (user):** a chart below the master table (before any feature-importance section) of QLIKE against Sharpe
  for the selected models and rules, in the style of artificialanalysis.ai's index-vs-cost scatter; models as
  different shapes, a model's rules joined by dotted lines. Built in the background, in parallel (user's request).
- **Data side, applied:** rows.py `estimator_of(key)` and `chart_point(row)`; server.py adds `row.chart` =
  {qlike_raw, qlike_recal, sharpe_mid, sharpe_crossed, estimator, fit, is_headline} to every table row. Checked on
  the live server: 539 of 539 rows carry it; only always short has no QLIKE.
- **Chart side:** index.html only (a background agent): plots the rows visible in the table (filters, per-rule limit,
  sort); x = QLIKE (plain, as in the table; or recalibrated), y = Sharpe (mid or crossed); shape = estimator, colour =
  rule (validated palette), hollow = pooled 48-bar fit, dotted grey = same model across rules; always short as a
  dashed horizontal reference; shaded "lower QLIKE and higher Sharpe than the baseline" quadrant; labels on the
  baseline and headline only, tooltips on every point; no external libraries.
- **Revised (user):** no headline label; the point popup shows the model name and rule only, on hover, and stays
  when the point is clicked (click it again or empty space to clear). Keep it simple and fast.
- **Revised again (user):** no shape by model and no colour by rule; every point the same circle in one colour; the
  popup reads "model name (rule)". Hollow-for-pooled and the legends dropped; dotted same-model lines, always-short
  line, baseline quadrant and label, and the axis selectors kept.
- **Built (background agent, index.html only):** panel "QLIKE against Sharpe" after the table; x/y selectors; one
  colour (#2a78d6, validator pass on #fffcf8); dotted same-model lines; always-short dashed line; baseline label and
  shaded better-than-baseline corner; popup "model (rule)" on hover, pinned on click. Plots exactly the visible table
  rows (hooks: `chartRowOf.set` in renderTable, `chartRefresh()` at the end of applyLimit). Tested in a Node harness
  and headless Chrome on mock and live payloads (default view: 351 points from 93 models). Not done: keyboard access
  to the popup; on the live-feasible view three models at QLIKE 0.21-0.23 squeeze the cluster (linear axis kept).

## 15. Default tree rung; chart coloured by model type — APPLIED

- **Decision (user):** tree estimator and inputs fully expanded, tuning = Optuna over 50 trials; each model a
  different colour in the chart; the chart should be uncrowded under the new default.
- **Retune frequency (assumed, user had not picked):** every 5 sessions, the Optuna retune frequency with the lowest
  recalibrated QLIKE in the most estimator x input arms (5 of 9, best of 50, squared-error pick); every session
  (the paper's Table 7 rung) is the alternative. Squared-error pick, refit every session.
- **Applied:** rows.py `_DEFAULT_TREE` (`subtree_optuna_tp5_<bucket>_<est>`); every other tree row is flagged
  `tree_ablation` and hidden unless the spec has `show_tree_ablations: true`; index.html "Show tree ablations"
  toggle (off by default, carried in the spec and the CSV export). Default view 88 rows (was 352): 27 models, 6 of
  them trees; ablations on 352; live-feasible on 143.
- **Chart:** colour per estimator (`CHART_EST`): OLS (baseline) ink, ridge blue, lasso violet, elastic net green,
  LightGBM orange, XGBoost aqua, random forest magenta, LSTM red, intraday LSTM yellow (dataviz reference hues);
  legend of the plotted estimators above the plot; pooled and per-bar fits of one estimator share its colour.
  Validator, all pairs on #fffcf8: FAILS (red/orange normal-vision dE 7.1, green/orange protan dE 3.2; no 8-hue set
  separates every pair on one scatter), so identity also rests on the popup's model name; aqua, yellow and magenta
  are below 3:1 contrast (the table is the relief view). Default chart: 87 points from 27 models (was 351 from 93).
  Checked in a headless-Chrome screenshot of the live page.
- **Left as is:** the Families box still lists all ten tree families; with ablations off only the TUNE_PER=5 family
  has rows (proposal items 1, 2, 4 and 5 of the tree reorganisation not applied).

## 16. Default tree rung: no tuning — APPLIED (replaces item 15's Optuna default)

- **Decision (user):** default the trees to tuning off.
- **Applied:** rows.py `_DEFAULT_TREE` = `subtree_daily_<bucket>_<est>`, the shipped configuration refit every
  session (T1), every estimator and input set; item 15's Optuna pattern kept as a dated comment. Random search and
  every Optuna rung are behind "Show tree ablations" (toggle tooltip updated). Default view still 88 rows, 27
  models, 6 trees. Chart unchanged (it follows the visible rows).

## 17. Optuna knobs as Rows-panel sections — APPLIED

- **Decision (user):** make the tree knobs sections of the Rows block: under Families, "Optuna retune every" with
  1, 5, 25; under it, "Optuna trials kept" with 10, 25, 50. Nothing else for now.
- **Applied:** index.html two checkbox groups after Families (retune: none ticked; trials: 50 ticked), carried in
  the spec (`optuna_retune`, `optuna_trials`) and the CSV export, restored by Apply spec. rows.py flags the Optuna
  rows with the squared-error pick and retune 1 / 5 / 25 (`optuna_button`, `optuna_tp`, `optuna_k`); such a row
  shows when its retune AND its trials are both ticked; these rows no longer depend on "Show tree ablations", which
  now covers random search, Optuna retune 250 and the Optuna QLIKE picks (tooltip updated).
- **Checked through the API:** default 88 rows, no Optuna; retune 5 + trials 50 adds 6; retune 1 and 25 with all
  three budgets adds 36; retune 5 + trials 10 adds none (retune 5 ran only at best of 50); ablations on adds random
  search 12, Optuna retune 250 6 and the QLIKE picks 6. Rows panel checked in a headless-Chrome screenshot.

## 18. Sizing rules scored after a 252-day warm-up — APPLIED

- **Question (user):** were the proportional Sharpes (all negative) computed correctly? **Answer:** yes, the GUI's
  arithmetic equals the 15:30 notebook's stored value (a0, from day 2: -0.436462 both). The negatives came from scoring
  from day 2: rms_t had only a few quiet January 2020 days behind it, so Jan 31, Feb 28, Mar 2 and Mar 13-18 2020 got
  positions of 20-37 straddles and lost; the 10 largest P&L days carried more than the whole negative total.
- **Decision (user):** skip the warm-up.
- **Applied:** rows.py `_SIZING_WARMUP = 252`: positions still use every earlier day, but every sizing rule (and its
  sign(s) and always-short comparison) is scored on days 253-866 (614); computed rows and the deck-file path alike
  (`_VRP_BLOCK` = "after 252-day warm-up"). Checked first: the GUI's arithmetic reproduces the notebook's stored
  warm-up block for all 15 of its forecasts and every rule to 2.2e-16. Header note and seven column glossaries now
  say 614 days after the warm-up. The sign(s) block itself stays on 866 days.
- **Result (live-feasible on, 31 models per rule; point estimates, no intervals):** proportional Sharpe median +1.05,
  gap to sign(s) -0.23 (7 above 0); rank of abs(s) +1.48, gap +0.12 (26 above 0); centred rank +1.11, gap -0.21 (5 above 0).
  No sizing Sharpe is negative now.

## 19. No tree families in the Families box — APPLIED

- **Decision (user):** remove the tree family options from the Families section.
- **Applied:** rows.py `_TREE_FAMILY_PREFIX = "per-bar tree"`: `facets()` no longer lists the ten tree families, and
  the family filter lets tree rows pass (they are chosen by the untuned default, the Optuna boxes and "Show tree
  ablations"). Families offered: per-bar linear, pooled twin, LSTM, LSTM (intraday sequence), HAR-ladder variants.
  Default view unchanged at 88 rows (6 trees). No switch hides the untuned trees now.

## 20. Drop "Δ vs sign compares with sign(s) on those days." — APPLIED

- **Decision (user):** that sentence is unneeded everywhere. It existed only in rows.py `_SIZING_NOTE` (the three
  sizing header rows); the note now reads "Scored on days 253 to 866 (614), after a 252-day warm-up that only builds
  the scale." The Δ vs sign column glossary (its definition) is unchanged.

## 21. "the full column set" -> "all features" — APPLIED

- **Decision (user):** say "all features" everywhere instead of "on the full column set".
- **Applied (describe.py, the only source):** model names (`_short_bucket`: "... on all features"), description
  sentences (`_BUCKET`: "on all features, including the cross-section, StockTwits, and Cboe volume") and the
  intraday-LSTM sentence ("all features, adjusted per bar, and their availability flags, plus the calendar and a
  half-hour clock"). No "full column set" left in the GUI. Checked through the API after a restart.

## 22. "the HAR and calendar set" -> "the HAR and cal. ind. features" — APPLIED

- **Decision (user):** model names say "on the HAR and cal. ind. features". Applied in describe.py `_short_bucket`
  (the only source); the description sentences ("the HAR ladder and the calendar") are unchanged.

## 23. No "daily refit" in labels — APPLIED

- **Decision (user):** erase "daily refit" from every label; every model shown is refit daily.
- **Applied (describe.py `_tree_phrase`):** "Daily-refit per-bar X on Y" -> "Per-bar X on Y"; "Daily squared-error-tuned
  per-bar ..." -> "Squared-error-tuned per-bar ..."; "Daily QLIKE-tuned ..." -> "QLIKE-tuned ...". Description
  sentences keep "refit every session". The parked every-10-session random-search rows would now share the
  random-search labels; they are not loaded.

## 24. HAR-ladder variants parked — APPLIED

- **Decision (user):** comment them out unless Sharpe rises monotonically as the ladder gets finer.
- **Check (master_table.csv, Sharpe mid, live_feasible; base 5 = production -> base 3 -> base 2 | same-clock):**
  ridge 1.90, 1.71, 1.34 | -0.02; lasso 1.83, 1.60, 1.79 | 0.19; elastic net 1.41, 1.57, 1.75 | 0.36. Monotonic
  rise only for elastic net (1 of 3, point estimates); not a property of finer ladders.
- **Applied:** "HAR-ladder variants" added to rows.py `_PARKED_FAMILIES` (dated comment with the numbers). Families
  offered now: per-bar linear, pooled twin, LSTM, LSTM (intraday sequence).

## 25. Every model gets sizing rows — APPLIED

- **Decision (user):** size every model instead of a list.
- **Applied:** rows.py `_attach_lasso_rules` now sizes every row except always short (and any deck-file key, none
  loaded); the earlier lists (per-bar lassos, `_DEDUP_RESIZED`, `_SIZED_FAMILY_PREFIXES`) are superseded. Same
  arithmetic, warm-up (614 days) and scorer as before. Pooled twins, the headline per-bar ridge, and the per-bar
  elastic nets now have sizing rows. Default view 109 rows (27 per rule block); live-feasible on 161 (40 per block);
  no model without sizing rows. Headline (live-feasible ridge) on the 614 days: proportional 0.65 (gap -1.11), rank
  1.43 (-0.33), centred 1.04 (-0.73). Pooled ridge HAR + calendar: proportional 0.834 and centred 0.829 both round to
  0.83 (checked: different positions, correlation 0.76).

## 26. "pooled twin" shown as "pooled" — APPLIED

- **Decision (user):** the family name "pooled twin" becomes "pooled". Applied at load in rows.py (the CSV keeps its
  name); `_POOLED_FAMILIES` includes "pooled" so the Fit filter still treats them as 48-bar fits.
- **Answered:** every nonlinear model in the GUI is per-bar (target = the 16:00 bar alone, one row per session):
  108 tree rows (LightGBM, XGBoost, random forest; every rung) and 12 LSTM rows (6 per-bar over the last 5 or 20
  sessions of the 16:00 bar, 6 intraday-sequence whose inputs are the last 13 / 48 / 96 half-hour bars but whose
  target is the 16:00 bar). The pooled family holds only ridge, lasso and elastic net (9). The only pooled nonlinear
  models were the paper's pooled LightGBM and XGBoost, parked in item 9.

## 27. Zoomable chart — APPLIED

- **Request (user):** make the graph zoomable.
- **Applied (index.html chart section):** drag a box over the plot to zoom into it (drag again to zoom further);
  double-click the plot or press "Reset zoom" (shown only while zoomed) to see every point. Lines and dots are
  clipped at the frame; only points in view can be hovered; the always-short line and the baseline label show only
  when inside the zoomed range; a click (no drag) still pins a popup; changing an axis measure or building a new
  table resets the zoom. The note reads "Drag a box to zoom." or "N of M points ... Zoomed: double-click or Reset
  zoom to see all."
- **Checked:** headless Chrome on a scratch copy of the page fed the live default payload, with real pointer events
  (`scratchpad/zoom_test.js`): 11/11 checks pass (drag zooms 108 -> 42 points in view, all inside the frame; the
  drag's click does not pin; a plain click still works; double-click and the button reset to 108).

## 28. Page title "0DTE Closing Straddle Trade" — APPLIED

- **Decision (user):** rename the page from "Master table". Applied to the browser-tab `<title>` and the page
  heading `<h1>` in index.html.
- **Housekeeping:** my Python-scripted edits had rewritten rows.py, columns.py and describe.py with Windows CRLF line
  endings (the folder's files are LF); restored to LF, contents unchanged (ruff passes). Later scripted edits write
  bytes so line endings are kept.

## 29. Families box: per-bar linear / pooled linear / nonlinear — APPLIED

- **Decision (user):** add a "nonlinear" box under Families for the trees and LSTMs; rename "pooled" to "pooled linear".
- **Applied (rows.py):** each row gets `family_group` = "nonlinear" for every per-bar tree and both LSTM families,
  otherwise its family; the Families box lists the groups in the order per-bar linear, pooled linear, nonlinear, and
  the family filter uses them (the item-19 tree exemption is gone: unticking nonlinear now hides the trees too). The
  pooled family displays as "pooled linear" (Family column and box). The Family column keeps each row's detailed
  family. Checked through the API: all ticked 109 rows; nonlinear unticked 53 (13 linear models); only nonlinear 57
  (6 trees, 8 LSTMs). With per-bar linear unticked the pinned baseline row is hidden too (DM still measured against it).

## 30. "per-bar linear" box renamed "linear" — APPLIED

- **Decision (user):** the Families box label "per-bar linear" becomes "linear". Applied in rows.py (`family_group` and
  `_FAMILY_GROUP_ORDER`): the box now reads linear, pooled linear, nonlinear. The Family column still says
  "per-bar linear" for those rows.

## 31. Rules boxes moved to the chart; chart only — APPLIED

- **Decision (user):** move the Rules section down to the graph and make it work for the graph; the table always shows
  every rule.
- **Applied (index.html):** the Rules boxes sit in the chart panel under the axis selectors, all ticked at first; a
  change redraws the chart at once (no rebuild). The table request never carries a rule filter; the selection is kept
  in the spec as `chart_rules` (an older spec's `rules` is read as a fallback). Keys hint updated.
- **Checked (headless Chrome, live default payload):** unticking Proportional takes the chart from 108 to 81 points and
  leaves the table at 109 rows; unticking Always short removes its line; re-ticking restores 108; zoom checks 11/11.

## 32. Every loss column uses the recalibrated forecast — APPLIED

- **Decision (user):** all losses on the recalibrated forecast; nothing renamed.
- **Applied:** the "QLIKE" column (id kept `qlike_raw`) now shows `qlike_recal`; MSE, MAE and OOS R2 are computed from
  `pred_clock` (rows.py `_attach_accuracy`); DM already used the recalibrated QLIKE. Glossaries for QLIKE and MSE now
  say "recalibrated". The chart's x default is the recalibrated QLIKE, its option labelled "(scored, as in the table)";
  the plain QLIKE stays as the second option, labelled "QLIKE (plain)" (the old "as in the table" would be untrue).
- **Checked:** for the per-bar OLS, the headline ridge, the pooled ridge (live) and per-bar XGBoost (all features), the
  served QLIKE equals the CSV's qlike_recal and MSE / MAE / R2 equal values recomputed from pred_clock and the target.

## 33. Chart x axis named "QLIKE" — APPLIED

- **Decision (user):** name the x axis "QLIKE" instead of "QLIKE recalibrated ...". Applied to the selector option and
  the axis title (`CHART_X.qlike_recal`); the title keeps ", lower is better (left)". The plain option stays
  "QLIKE (plain)".

## 34. Chart's dotted lines: HAR + calendar to all features — APPLIED

- **Decision (user):** instead of tracing a model's sizing rules, the dotted lines trace HAR + cal. ind. features to
  all features.
- **Applied (index.html `chartModel`):** a line joins the same model and rule on the two input sets (key with
  `_baseline` / `_all_features` swapped); models without both (the OLS baseline, live-feasible versions) get none.
  Note text and code comment updated. Checked (headless Chrome, live default payload): 52 links, 13 per rule, each
  joining the same model and rule; the rules (9/9) and zoom (11/11) suites still pass.

## 35. "on the HAR and cal. ind. features" -> "on HAR and cal. ind. features" — APPLIED

- **Decision (user):** drop the "the". Applied in describe.py `_short_bucket` (model names, 12 of them in the default
  view). Description sentences ("on the HAR ladder and the calendar") unchanged.

## 36. Which end is which: lines fade from HAR + calendar to all features — APPLIED

- **Decision (user):** show which end of each line is all features and which is HAR + cal. ind. features; an arrowhead
  was tried first, then replaced at the user's suggestion by line opacity.
- **Applied (index.html `chartDraw`):** each link is a solid 1.5px grey line with its own gradient, opacity 0.08 at the
  HAR + calendar end and 0.9 at the all-features end (a plain line if the two ends coincide). Note: "Each line joins
  a model on HAR + cal. ind. features (faint end) to the same model on all features (dark end)." The "always short"
  label moved to the left end of its line (the right end met the LSTM points).
- **Checked (headless Chrome, live default payload):** 52 links, all with a gradient; rules 9/9 and zoom 11/11 still
  pass; screenshot looked at (`scratchpad/shot_fade.png`). The links test's note check failed only because the test
  still looked for the old note wording; the screenshot shows the new note.

## 37. Baseline label and shading survive unticking sign(s) — APPLIED

- **Reported (user):** the baseline label and shaded region disappear when sign(s) is unticked in the chart's Rules.
- **Cause:** both were anchored only to the baseline's sign(s) point.
- **Applied (index.html `chartModel` / `chartRefresh`):** they anchor to the baseline's sign(s) point, or, when sign(s)
  is not plotted, to its point under the first plotted rule in the order Proportional, Rank of |s|, Centred rank; the
  note then says "... than the baseline under <rule>." With only Always short plotted there is no baseline point and
  no shading.
- **Checked (headless Chrome, live default payload):** label, shading and note correct with all rules, with sign(s)
  unticked (Proportional), with sign(s) and Proportional unticked (Rank of |s|), and after re-ticking; links 8/8,
  rules 9/9, zoom 11/11.

## 38. Pooled models drawn as squares in the chart — APPLIED

- **Decision (user):** pooled dots become squares.
- **Applied (index.html):** a point whose `row.chart.fit` is "All 48 bars (pooled)" is an 8px square in its estimator's
  colour (same paper ring); per-bar points stay 9px circles. The legend adds "square = pooled (all 48 bars)" when any
  pooled point is plotted.
- **Checked (headless Chrome, live default payload):** 24 squares (the pooled ridge / lasso / elastic net on HAR + cal.
  and all features, 4 rules each) and 84 circles; legend entry present; screenshot looked at; base 13/13, links 8/8,
  rules 9/9, zoom 11/11.

## 39. Rows-panel tick boxes apply at once (Families now work on the graph) — APPLIED

- **Reported (user):** the Families boxes do not work on the graph.
- **Cause:** Rows-panel boxes only took effect on "Build table"; the chart's Rules boxes act at once, so the Families
  boxes looked broken.
- **Applied (index.html):** any tick box in the Rows panel (Fit, Families, Optuna retune / trials, the toggles) and the
  Families All / None buttons rebuild the table, and so the chart, 150 ms after the change (one request for several
  quick clicks). The search and keys text fields still wait for "Build table". A rebuild resets the chart zoom and any
  column sort, as a manual build did.
- **Checked (headless Chrome; the fetch stub filters by family group as rows.py does):** unticking nonlinear takes the
  table from 109 to 53 rows and the chart from 108 to 52 points, with no tree or LSTM point left; All restores 109 / 108.
  The live server returns the same 53 rows for that filter (item 29). Base 13/13, links 8/8, rules 9/9, zoom 11/11.

## 40. "Show LSTMs" toggle, off by default — APPLIED

- **Decision (user):** a button to turn the LSTMs on, off by default.
- **Applied:** rows.py `select_rows` hides every row of the two LSTM families (per-bar and intraday-sequence) unless the
  spec has `show_lstm: true`; index.html adds the "Show LSTMs" toggle (unticked) after "Show live-feasible", carried in
  the spec and the CSV export; like every Rows-panel tick box it rebuilds the table and chart at once (item 39).
- **Checked through the API:** default 77 rows (19 models, no LSTM); with the toggle 109 rows (27 models, 8 LSTMs).

## 41. Baseline pinned in every view again — APPLIED

- **Reported (user):** the per-bar OLS on HAR + cal. ind. features is no longer pinned under each rule header.
- **Cause:** it still led every block whenever "linear" was ticked, but it is a per-bar linear model, and the family
  filter rebuilt for the family groups (item 29) did not exempt it, so unticking "linear" hid it (noted in item 29 as a
  consequence; now treated as a defect, since the DM column is measured against it).
- **Applied (rows.py `select_rows`):** the family filter keeps `_BASELINE_KEY` whatever is ticked, as the Fit filter has
  since item 3. Checked through the API: with only nonlinear ticked (29 rows), only pooled linear, and nonlinear with
  16:00-bar-only, the baseline is first in every block.

## 42. Skew and kurtosis columns — APPLIED

- **Request (user):** a kurtosis and a skew column.
- **Applied:** rows.py `_skew_kurt` (skewness m3 / m2^1.5, excess kurtosis m4 / m2^2 - 3, moments about the mean with
  n in the denominator; NaN below 4 days) inside `_series_stats`, so every row gets them from its own daily midpoint
  return (sign(s), always short, every sizing row), with crossed-fill versions. columns.py: "skew" and "kurt" after std
  (default on), "skew×" / "kurt×" after std× (default off); index.html sort entries. Checked against scipy's biased
  estimators on the daily file for the OLS baseline, the headline ridge, per-bar XGBoost and always short.

## 43. Sortino as a chart y axis — APPLIED

- **Request (user):** a y axis that shows Sortino.
- **Applied:** rows.py `_sortino` (mean daily return over the downside deviation sqrt(mean(min(r, 0)^2)), target 0,
  times sqrt(252)) in `_series_stats`, so every row (sign(s), always short, every sizing row) gets mid and crossed
  values; `chart_point` sends them; index.html adds "Sortino (mid)" and "Sortino (crossed)" to the y selector with
  their axis titles. Not added as a table column (not asked).
- **Checked:** served values equal a recomputation from the daily file (OLS baseline 2.09, headline ridge 3.19,
  per-bar XGBoost 2.72, always short 0.24); headless Chrome: switching the y axis retitles it, keeps every point and
  the always-short line; base, links, rules and zoom suites still pass.

## 44. No recalibration anywhere in the GUI — APPLIED (GUI only; the paper is a separate decision)

- **Decision (user):** get rid of recalibration; it is too complicated for the paper and not the main result.
- **Applied (GUI):** every number now uses the plain forecast f^2 * B. rows.py `_attach_plain_trade` recomputes each
  forecast row's sign(s) trade (Sharpe mid / crossed, means, hit rates, buys, gap to always short, the return
  distribution, t, skew, kurtosis, Sortino) as sign(plain - iv_var) times the quote book's straddle, replacing the
  stored research-scorer numbers (after checking the quote book reproduces the stored trade); `_attach_accuracy` uses
  the plain forecast for MSE / MAE / OOS R2 and the plain daily QLIKE for DM; the sizing rules' gap is plain - iv_var;
  the QLIKE column shows `qlike_raw`; the old `_attach_return_dist` is parked. Glossaries, the sign(s) header ("the
  row's 16:00 variance forecast", no "recalibrated") and the chart (x axis QLIKE = plain only) updated.
- **Checked:** served Sharpe equals an independent sign(plain - iv) recomputation for seven models; browser suites
  sortino 6/6, base 13/13, links 8/8, rules 9/9, zoom 11/11 on the new payload.
- **What moved (40 models, live-feasible and LSTMs on):** median sign(s) Sharpe 1.26 -> 1.31; 20 of 40 higher. Headline
  per-bar ridge live 1.90 -> 1.79 (QLIKE 0.1184, DM +1.17 vs the baseline); top by Sharpe now per-bar elastic net all
  features 2.04, per-bar lasso all features 2.00, per-bar lasso live 1.87, then the headline ridge 1.79.
- **Not done (needs the user):** removing recalibration from the paper and the master-table pipeline.

## 45. Intervals commented out in the GUI — APPLIED

- **Decision (user):** comment out all of the intervals.
- **Applied:** the GUI already displayed none (every interval column was commented out before this session); the four
  interval formatting helpers in columns.py (`_qlike_interval`, `_ci`, `_vs_ref`, `_vs_short`, unused) are now
  commented out with a dated note. The repo's bootstrap itself was commented out in commit b761b28 (2026-10-03).

## 46. No `adj_` in any presented variable name — IN PROGRESS (agents)

- **Decision (user):** remove `adj_` from the names of any variable being presented; it clogs the presentation.
- **Why it is safe:** the HAR columns are already adjusted (trailing means of `adj_RV`, named `har_ma_*` by
  `src/features/extractors/har.py:47`); every exogenous input is divided by its time-of-day baseline. Stripping the
  prefix leaves all 628 per-bar design names unique (checked on the cached designs).
- **Scope:** display only (makers, keys, figure labels); data files, keys and code identifiers keep their names.
  Feature-importance documents (both passes), model diagnostics, dense-vs-sparse, paper figures, the master-table
  documents and the paper text. The GUI shows no variable names. The key says once that every exogenous input is
  divided by its time-of-day baseline.
- **Wording rule (supersedes the handoff's `adj_vix_ma_*` example):** design column names in figures and tables
  without the `adj_` prefix.

## 47. Shrinkage of the HAR + calendar backbone — no change (information)

- **Question (user):** do all GUI results shrink the baseline HAR variables? **Yes:** every per-bar ridge / lasso /
  elastic net row comes from `specs/causal_tune_linear.py`, where only the intercept is unpenalized
  (`_locked[-1] = True`, line 300); the OLS baseline has no penalty at all.
- The 2026-10-03 study `results/close_studies_2026-10-03/exog_penalty/` left the backbone unpenalized (all
  features, 16:00 bar). Re-scored here on the plain forecast (866 trade days; its one-penalty arms reproduce the
  GUI rows exactly): ridge 0.1164 / 1.71 -> 0.0960 / 1.65; lasso 0.0973 / 2.00 -> 0.0935 / 1.90; elastic net
  0.0986 / 2.04 -> 0.0976 / 1.79 (QLIKE / sign(s) Sharpe mid, one penalty -> backbone unpenalized).

## 48. A third input set, moments + VIX, for every model — IN PROGRESS (runs)

- **Decision (user):** on top of HAR + cal. ind. features and all features, a version of every model on the return
  moments + the VIX level alone; use both clusters as much as possible.
- **Bucket:** `moments_vix` = HAR + calendar + `SUBGROUPS["moments"]` (sumret, sumabsret, sumret3, sumret4, sumpret2,
  sumbipow, sumautocov) + `vix` (`src/data/loading.py:394`). No VVIX / VIX3M, no FOMC, no volume.
- **GUI (applied ahead of the data):** label "moments and VIX" (describe.py `_BUCKET`, `_short_bucket`,
  `_LSTM_INTRA_COLS`); the default-tree pattern includes it (rows.py `_DEFAULT_TREE`). The master table files it under
  per-bar linear (it is not one of the parked VIX-only buckets).
- **Runs:** per-bar linear locally; default trees + ablations, Optuna trees, LSTMs (CARC, Hoffman2 where shown
  identical) and pooled linear (Hoffman2) on the clusters, each gated on a stored run first.

## 49. Robust clip replaces proportional; centred rank removed — APPLIED

- **Decision (user):** get rid of the centred rank rule; replace the proportional rule with the robust clip (the
  "sizing by size" rule that behaves like rank of |s|; proposed in the clipping discussion of 2026-10-02).
- **Rule:** q_t = clip(s_t / m_t, -1, +1), m_t = median |s_u| over the earlier days u < t: proportional below a typical
  gap, one straddle beyond it. Scored after the same 252-day warm-up (614 days).
- **Applied:** rows.py `_position_sizes` (robust clip added; proportional and centred rank parked), `_SIZING_ROWS`,
  `_RULE_ORDER`, `_RULE_HTML` (formula header); index.html `RULE_ORDER` and the baseline-anchor rule list. Blocks are
  now Always short, sign(s), Robust clip, Rank of |s|.
- **Checked:** the OLS baseline's served robust-clip Sharpe equals an independent recomputation (1.5492, 614 days);
  browser suites sortino 6/6, base 13/13, links 8/8, rules 9/9, zoom 11/11.
- **Numbers (139 forecasts, plain forecast, same 614 days):** robust clip median Sharpe 1.44, above sign(s) in 112
  (median gap +0.30), median kurtosis 4.9; rank of |s| 1.45, 114 above (+0.30), kurtosis 5.0. Point estimates only.

## 50. "Full long, scaled short" sizing block — APPLIED

- **Decision (user):** add the "full long, scaled short" rule to the master table.
- **Rule:** q_t = +1 if s_t > 0; q_t = max(s_t / m_t, -1) if s_t <= 0, m_t = median |s_u| over u < t (the robust clip's
  scale). Buys one straddle; sells the robust clip's amount. Same 252-day warm-up (614 days).
- **Applied:** rows.py `_position_sizes` ("lfss"), `_SIZING_ROWS`, `_RULE_ORDER`, `_RULE_HTML`; index.html
  `RULE_ORDER` and the anchor list. Block order: Always short, sign(s), Robust clip, Full long scaled short, Rank of |s|.
- **Checked:** OLS baseline served 1.6934 = independent recomputation; browser suites all pass (sortino 6, base 13,
  links 8, rules 9, zoom 11).
- **Numbers (139 forecasts, same 614 days, medians):** Sharpe 1.34, above sign(s) in 104 (gap +0.16), Sortino 2.15,
  skew +2.05, kurtosis 22.5, worst day -3.55, best day +10.32 (robust clip: 1.44, Sortino 1.99, skew -0.79,
  kurtosis 4.9, worst -3.55, best +3.22).

## 51. Dense-vs-sparse write-up commented out of the paper — IN PROGRESS (text agent)

- **Decision (user):** comment out the dense vs sparse write-up. In the paper it is the density part of D.19
  (`sec:app_camp_featimp`: the opening clause, the density paragraph, Table `tab:app_camp_dvs`); the subsection
  becomes "Feature importance on the new design". Not affected: the paper's "dense but weak" theme and the
  multi-horizon `dense_numbers` / `table_dense_byk` (direct multi-horizon origins).

## 52. Backbone-unpenalized linear rows in the GUI — APPLIED

- **Decision (user):** add the backbone-unpenalized rows to the GUI.
- **Rows:** per-bar ridge / lasso / elastic net on all features with the HAR + calendar columns unpenalized (the
  2026-10-03 study `results/close_studies_2026-10-03/exog_penalty/`, arm `bb0`), keys `sub_<est>_all_features_bb0`,
  family "per-bar linear, HAR unpenalized" under the linear box, each right after its one-penalty twin, label
  "Per-bar <est> on all features, HAR + cal. unpenalized".
- **How:** rows.py builds their daily series from the study's forecast export (plain f^2 * B on the master table's 866
  days, the same quote book) and `_read_daily` feeds them to every scorer, so they get the same columns and the
  four rule blocks. No new data file; the master table is not re-run.
- **Gate:** the export's one-penalty arms equal the master table's rows: ridge and lasso to 1e-11; the elastic net
  differs on 132 days by up to 1.3 % on variance with the same position on all 866 days (documented in the study's
  SUMMARY.md gate 1: its warm homotopy branches on the floating-point path), so its check is same positions and
  <= (1.0065^2 - 1).
- **Numbers (plain, 866 days):** ridge 0.0960 / 1.65, lasso 0.0935 / 1.90, elastic net 0.0976 / 1.79 (QLIKE / Sharpe mid),
  vs one penalty 0.1164 / 1.71, 0.0973 / 2.00, 0.0986 / 2.04.

## 53. The all-features linear rows carry the backbone-unpenalized numbers, original names — APPLIED (supersedes 52)

- **Decision (user):** replace the all_features rows with the unpenalized numbers; keep the original name.
- **Applied:** rows.py: per-bar ridge / lasso / elastic net on all features (`sub_<est>_all_features`) now carry the
  HAR-unpenalized forecasts of the 2026-10-03 study under their own keys and names. `_read_daily` drops their
  one-penalty series and feeds the unpenalized ones to every scorer; `_add_exogpen_rows` blanks the stored numbers
  and re-scores (QLIKE, MSE / MAE / R2, DM, the trade, every rule block). The separate `_bb0` rows of item 52 are gone.
  The description ends: "The HAR and calendar columns carry no penalty; only the other inputs are shrunk."
- **Checked:** served numbers = the earlier independent scoring (ridge 0.0960 / 1.65, lasso 0.0935 / 1.90, elastic net
  0.0976 / 1.79); the export gate of item 52 still runs; browser suites all pass.
- **Not changed:** the other input sets (HAR + cal., live-feasible, moments + VIX) keep one penalty for every column
  (no unpenalized run exists for them); the paper's master table is unchanged.

## 54. All-features linear rows: backbone penalized separately at 1/100 — APPLIED (supersedes 53)

- **Decision (user):** change out unpenalized for separately, much more lightly, penalized.
- **Applied:** rows.py reads the 2026-10-03 study's `bbfix` arm (HAR + calendar penalized at 1/100 of the exogenous
  penalty, the paper's 2-block ridge ratio) for `sub_<est>_all_features`, same keys and names; description: "The HAR
  and calendar columns are penalized separately, at 1/100 of the other inputs' penalty."
- **Next (user):** pre-tune the linear penalty on the data before the trade period, then retune lightly and sparsely,
  as the trees' pre-tune study did: study dispatched (`results/linear_pretune_2026-10-06/`).

## 55. No headline model; live-feasible out of the paper — APPLIED (GUI), IN PROGRESS (paper)

- **Decisions (user):** "live-feasible is not a part of the paper anymore"; asked which model replaces the per-bar
  ridge on live-feasible as the headline: "nothing, the paper is not about a specific headline sharpe".
- **GUI:** the "Only headline" toggle is parked (index.html: the checkbox, the two key lists, the spec); the
  live-feasible toggle's tooltip says it is not part of the paper (still off by default).
- **Paper:** the text agent removes the headline framing from Section 5.4 (the model set instead) and parks
  live-feasible mentions; D.19 follows b761b28. Appendices D.14 and D.16-D.18 dissect the headline model: held,
  user to decide (park them, or re-base them on the model set).

## 56. Every moments + VIX model in the GUI; the GUI reads its own master-table run — APPLIED

- All 55 moments_vix forecast tables (per-bar linear 3, pooled 3, default trees + ablations 18, Optuna 27, LSTMs 4)
  are stacked in `results/spxw_pnl/`. `experiments/master_table_close.py` input-set pattern gains `moments_vix` (line
  ~493). The GUI reads a separate run, `results/close_master_table_gui/` (275 table-A forecasts, 1795 gates, 0 failed;
  scratchpad wrapper `master_table_gui_run.py` = the script with OUT redirected, `--no-pdf`), so the paper's
  `results/close_master_table/` stays as the paper's macros use it (moments_vix in the paper: user to decide).
- rows.py `_DEFAULT_CSV` / `_DAILY` point to the GUI run.

## 57. LSTMs not trustworthy — APPLIED (GUI), IN PROGRESS (paper)

- **Decision (user):** the LSTM numbers are not to be trusted; how to do an LSTM properly for this data is unsettled.
- **GUI:** the toggle reads "Show LSTMs (untrusted)", off by default, tooltip says their numbers are not results.
- **Paper:** LSTMs excluded from every model-set count / table (macros agent); LSTM results parked from the text.

## 58. Input sets after the importance study: moments + volume + VIX, then the lean set as the paper's third — IN PROGRESS

- **Decisions (user):** add ES volume to moments + VIX (`moments_vol_vix`, 9 columns); run the lean set (`lean_vol_vix`:
  signed, absolute and quartic return, autocovariance, ES volume, VIX; 6 columns); "make the lean set the third input set
  in the paper, but it has to be a follow up to the feature importance study".
- **Why lean:** in the all-features importance study (37 model x measure cells, 49 series) signed return (median rank
  3), absolute return (5), ES volume (5) and autocovariance (12) lead; bipower (31) and upside squared returns (32)
  sit in one cluster with absolute and quartic return at every lag (|corr| >= 0.82); cubic return is weak (20).
- **Per-bar linear so far (plain, QLIKE):** ridge all 0.1164 / moments+VIX 0.1379 / +volume 0.1180 / lean 0.0966;
  lasso 0.0973 / 0.0950 / 0.0939 / 0.0938; enet 0.0986 / 0.0965 / 0.0949 / 0.0945 (OLS on HAR + cal. 0.0946).
- **Paper (default model set, user may amend):** the paper's own forecasts + per-bar linear, pooled linear and trees on
  HAR + cal., all features and lean; out: live-feasible, LSTMs, moments_vix / moments_vol_vix, VIX-only family, IV
  representations, chain-period buckets. New appendix subsection after the importance study; selection caveat stated
  (the lean set was chosen after an importance study on the same period).
- **Code:** `src/data/loading.py` (`moments_vol_vix`, `lean_vol_vix`), `experiments/master_table_close.py` input-set
  pattern, GUI labels "moments, volume and VIX" / "lean moments, volume and VIX".

## 59. Every per-bar linear model is the pre-tuned version — APPLIED (GUI), IN PROGRESS (paper)

- **Decision (user):** use the pre-tuned version for all linear models.
- **Version:** arm (f) of `results/linear_pretune_2026-10-06/`: penalty chosen once on the sessions before 2020 (forward
  folds), HAR + calendar penalized at 1/100 of the other inputs' penalty, retuned every 250 sessions only when the
  validation gain clears a margin (the margin fired 4 of 72 times, all on HAR + cal.).
- **Why:** causal re-choice switched the penalty at 75 of 165 re-choices; only 3 of the 75 passed a margin; 132 of
  198 picks sat at a grid edge; a switch moved the forecast a median 2.6 % and flipped 8 % of the block's positions.
- **GUI:** rows.py reads arm (f) for every `sub_<est>_<bucket>` row the master table has (HAR + cal., all features,
  live-feasible, moments + VIX; moments + volume + VIX and lean once in the GUI master table), same keys and names,
  description note added. Gate: the study's one-penalty arm (a) equals the master table's rows (ridge to 1e-9; lasso /
  elastic net within the documented warm-path gap, same positions). 12 of 12 rows equal the study's arm (f).
- **Paper:** the macros agent brings the pre-tuned forecasts into the paper master table as their own key family;
  the text agent describes the tuning once in methods.

## 60. Moments + VIX removed everywhere; the GUI rebuilds itself as new tables land — APPLIED

- **Decision (user):** get rid of moments + VIX only everywhere; make sure the GUI has the lean set.
- **Why its one-penalty ridge was bad:** four crash-recovery days (2020-03-25 .. 2020-04-08) where the forecast fell to
  4-21 % of realized variance (2020-04-06: QLIKE 20.2 on one day = 17 % of its total); without its 10 worst days it was
  0.0876 vs the OLS's 0.0884. The pre-tuned version (HAR at 1/100) scored 0.0938.
- **Applied:** the 55 `yhat_*moments_vix*` tables moved to `results/spxw_pnl/parked_moments_vix_2026-10-06/` (README);
  rows.py `_PRETUNE_BUCKETS` without moments_vix; every agent told (paper text and macros: no moments_vix anywhere).
- **GUI rebuild watcher** (scratchpad `gui_watcher.py`, log `gui_watcher.log`): when the set of
  `results/spxw_pnl/yhat_*.parquet` changes and stays quiet one 2-minute cycle, it re-runs the GUI master table
  (`results/close_master_table_gui/`) and restarts the GUI, so the lean and moments + volume + VIX rows appear as the
  cluster agents stack them.

## 61. Chart lines through the lean set — APPLIED

- **Decision (user):** lines connecting the lean set to the other two: HAR + cal. to lean as a dotted version of the
  current line, then lean to all features.
- **Applied (index.html):** links are [from, to, dotted]: HAR + cal. -> lean dotted (`stroke-dasharray 1 4`, same
  faint-to-dark fade), lean -> all features solid with the fade; a model with no lean version keeps the solid
  HAR + cal. -> all features line. Caption updated.
- **Also:** rows.py skips the paper's `subpt_*` copies of the pre-tuned linear forecasts (they duplicated the sub_*
  rows and crashed the labeler, HTTP 500 after the watcher's rebuild).
- **Checked:** new lean-link test (12 dotted, 12 lean -> all solid on the default view) and the updated link test pass;
  all six browser suites pass.
- **Selection check (for the paper's caveat):** importance from pre-2020 refits only (2018-19 tails, 381 days) picks
  the same six lean columns; the absolute-return cluster already holds before 2020; VIX is in by design (rank 19 in
  both samples).

## 62. Moments + volume + VIX (the non-lean version) removed everywhere — APPLIED

- **Decision (user):** comment out the non-lean results everywhere (read as the 9-column moments + volume + VIX set;
  HAR + cal. and all features stay as the paper's other two input sets).
- **Applied:** its 3 per-bar linear tables moved to `results/spxw_pnl/parked_moments_vol_vix_2026-10-06/` (README);
  rows.py `_PRETUNE_BUCKETS` without it; the cluster agents stop moments_vol_vix work (anything already produced goes to
  the parked folder) and put the capacity into lean; the paper agents carry nothing about it. The GUI watcher rebuilds
  on the file change.
- **Per-bar ridge on all features (pre-tuned) trails its lean and lasso versions:** QLIKE 0.0958 vs 0.0916 (ridge lean)
  and 0.0914 (lasso all); Sharpe 1.42 vs 1.88. Not crash days (worst-10 share 14 %, as the others). Ridge cannot drop
  inputs: with ~600 columns it keeps small coefficients on all of them, its forecast runs about 1.6 % below the lean
  ridge's (median ratio 0.984) and lower still on high-variance days (2020-03-13/25, 2020-04-06: 22-28 % of realized vs
  27-40 %), so it buys less (27 % vs 30 %) and differs from the lean ridge on 79 days. Lasso and the elastic net drop
  most columns and land near the lean set.

## 63. Headline appendices: D.14 parked, D.16-D.18 re-based on the model set — IN PROGRESS

- **Decision (user):** comment out D.14 (where the headline's P&L comes from); re-base D.16 (shift / dating test),
  D.17 (score at the trade's threshold + sample splits) and D.18 (fixed-fraction compounding) on the paper's model set.
- **Who:** the diagnostics agent rewrites the three scripts for every model-set forecast (the list written by the macros
  agent to `results/close_master_table/paper_model_set.csv`; flag `results/close_shift_test/MODELSET_DONE`); the macros
  agent parks the `\cmPnl...` macros; the text agent parks D.14 and its references and rewrites D.16-D.18.
- **Sentinel-B bars (information, user to decide):** the 65 bars are the first five appearances of 13 slots that began
  printing after the panel starts (2002 evening, Jul-Aug 2003 overnight); `target.py:297` fills B = 1; each scores QLIKE
  10-17. Excluding them lowers every main-results level by ~0.0029, flips no claim and no DM sign, fixes the methods'
  "maximum absolute error 4.27" (a sentinel bar; 1.7e-3 without them); 110 macros and ~10 typed counts would change.
  Details: `results/sentinel_b_impact_2026-10-06/`.

## 64. The 65 sentinel-B bars excluded from the main results — IN PROGRESS

- **Decision (user):** yes, exclude the 65 bars.
- **How:** a scoring-time validity rule in `experiments/score_unification.py` (bars whose time-of-day baseline has
  fewer than five earlier same-slot observations, i.e. B = the 1.0 fill, are not scored; the paper2 checkout's
  `_valid_of`); the target transform is unchanged. Gate: the regenerated macros equal
  `results/sentinel_b_impact_2026-10-06/main_without` / `fomc_without`.
- **Text:** the methods' "maximum absolute error 4.27" (a sentinel bar) becomes 1.7e-3 via a macro; typed sample counts
  273,554 -> 273,489, 272,971 -> 272,907, 263,782 -> 263,732; the exclusion stated once in methods and data_prep.
  No claim flips (checked).

## 65. The GUI shows only HAR + cal. ind. features and all features — APPLIED

- **Decision (user):** comment out everything that isn't all features or HAR + cal. ind. in the GUI; the QLIKE vs
  Sharpe plot is to become a paper figure telling "regressing on exogenous variables increases Sharpe and decreases
  QLIKE across the models".
- **Applied:** rows.py keeps a row only if its key is on `baseline` or `all_features` (`_GUI_INPUT_SETS`) or is always
  short; the live-feasible toggle is parked (hidden input kept for the page code). Lean, live-feasible and every other
  input set are out of the GUI; the links fall back to HAR + cal. -> all features. Browser suites pass.
- **What the data say (every model on both sets, LSTMs excluded; pre-tuned per-bar linear):**
  - default view, 9 pairs: Sharpe higher with all features in 9 (0 significant), QLIKE lower in 7 (1 significant);
    exceptions per-bar ridge (+0.0011, DM 0.23) and random forest (+0.0003, DM 0.08).
  - every tree rung and Optuna variant, 42 pairs: Sharpe higher in 40 (4 significant, 0 significantly lower), QLIKE
    lower in 30 (3 significant, 0 significantly higher); median dQLIKE -0.0016, dSharpe +0.52.
  - Keep every model in the figure; the text should say "for most models", give the counts, and note that model-level
    counts are not independent evidence (the models share the data).

## 66. GUI text brought up to date — APPLIED

- **Decision (user):** the GUI text needs to be updated.
- **Applied (index.html):** subtitle (was "Numbers are the stored master table, not recomputed") now says the GUI shows
  every model on HAR + cal. ind. and all features, every number on the plain forecast over the 866 trade days (sizing
  rules: 614 days after the warm-up), per-bar linear models pre-tuned before 2020; the keys hint no longer mentions
  "Show live-feasible"; the chart caption describes only what is drawn (HAR + cal. -> all features; the lean wording
  only if a lean link exists). Row descriptions, rule headers and glossaries checked: current (plain forecast, rule
  formulas, warm-up, pre-tuned note on the per-bar linear rows). Browser suites pass.

## 67. The paper figure: QLIKE against Sharpe, HAR + cal. -> all features (no lean) — IN PROGRESS

- **Decision (user):** the figure stays HAR + cal. ind. -> all features; the lean set is not in it (it stays in the
  paper as the importance follow-up only).
- **Figure:** `writeup/make_fig_qlike_sharpe.py` -> `writeup/figures/fig_qlike_sharpe.pdf`, from the paper model set
  (sign(s), LSTMs excluded, pre-tuned per-bar linear), the GUI's colors, squares for pooled fits, a faint-to-dark line
  per pair; caption numbers as macros (`\fqs...`): the counts of lower QLIKE / higher Sharpe with their significant
  counts, the medians, and the threshold-score vs QLIKE correlation with the Sharpe change. Every model on both sets is
  shown; the caption gives the counts. The text agent places it in Section 5.4.

## 68. Where the trees' fixed settings come from — APPLIED (GUI text)

- **Question (user):** weren't LightGBM and XGBoost out of the box? **No:** their per-bar settings were copied from the
  main results' pooled tree bank (`lgbm_opt_00`, `xgb_opt_06`), chosen by an Optuna search on the 1998-2002
  development span only (`experiments/tree_menu_dev.py`, DEV_END_DATE 2003-01-01; no look-ahead into 2020-24): shallow,
  slow, subsampled, minimum leaf sizes. The random forest had no precedent and runs sklearn's defaults
  (`specs/causal_tune_trees.py` PROVENANCE). So the forest is not compared like for like; the RF / XGBoost pre-tune study
  gives it its own pre-2020 tuning.
- **Applied:** describe.py: the untuned tree rows' descriptions now say where their settings come from (LightGBM /
  XGBoost: tuned on 1998-2002 for the pooled bank, not retuned here; random forest: sklearn defaults, no earlier tuning).

## 69. The random forest gets its settings from the same 1998-2002 procedure as LightGBM / XGBoost — IN PROGRESS

- **Facts:** LightGBM's and XGBoost's per-bar settings (`lgbm_opt_00`, `xgb_opt_06`) come from `experiments/tree_menu_dev.py`
  (Optuna, pooled all-bars design, dev span before 2003, sqrt-space MSE); the forest was never in that search (sklearn
  defaults). The 2016-19 pre-tune study (`results/trees_pretune_2026-10-06/`) used the 16:00 bar only: not like for like.
- **Decision (user):** run the forest through the same dev-span procedure (same objective, span, window, trial budget,
  freeze rule), then re-run the per-bar forest (T1, T10) on HAR + cal., all features and lean with those settings, under
  new keys (`*_rfdev`) for review before anything is swapped. The search agent also reports how `xgb_opt_06` (not the
  top XGBoost entry) was picked, and flags any look-ahead in that pick.

## 70. Pooled linear rows pre-tuned too (one protocol for all linear models) — APPLIED (GUI), IN PROGRESS (paper)

- **Decision (user):** do it -- apply the pre-specified protocol to the pooled models although the pooled study shows no
  gain (QLIKE worse in 8 of 9 pairs, mean +1.2 %, nothing significant; Sharpe higher on all features, lower on lean /
  HAR + cal.); reported honestly in methods.
- **GUI:** rows.py replaces every `pool_<est>_<bucket>` row with arm (f) of `results/pool_linear_pretune_2026-10-06/`
  (Hoffman2), using the stored pooled tables' own 16:00 baseline; gate: the study's causal arm (a) equals the stored
  pooled rows (1e-6, same positions); skips the paper's `poolpt_*` copies. 6 of 6 GUI pooled rows equal the study.
- **Paper:** `poolpt_*` key family in the master table, paper model set, macros `\cmPoolPt...`, both figures re-run;
  D.16 then shifts only the paper's 9 forecasts.

## 71. Open decisions answered — RECORDED

- **Change plot:** not the main figure; the scatter (`fig_qlike_sharpe.pdf`) stays the paper figure. The change plot
  stays a file only (`writeup/figures/fig_qlike_sharpe_change.*`).
- **Live card:** no re-estimate of `DEFAULT_LONG_FRACTION` and no push for now (the plain switch stays in the working tree).
- **Random forest:** yes -- when the dev-span re-run (`*_rfdev`) lands, it replaces the default-settings forest in the
  paper and the GUI.

## 72. The forest's dev-span settings frozen; XGBoost's config history — RECORDED

- **Forest:** `rf_opt_00` (top 1998-2002 dev entry, LightGBM's rule): 404 trees, depth 19, min leaf 24 (scaled to 2
  for the 2,000-row per-bar window, as the spec scales LightGBM 98 -> 8), max_features 0.896, max_samples 0.780;
  dev MSE 0.108644 vs lgbm_opt_00 0.107449, xgb_opt_06 0.108076. Frozen at 87 trials (XGBoost's surviving count;
  LightGBM 398); a 400-trial job (CARC 12777812) is still running -- user to decide.
- **XGBoost's config:** `xgb_opt_06` is the 7th-ranked dev entry, hand-picked about 2026-08-14 (for the 15:30 deck,
  `yhat_tree16`) with no recorded reason, after the bank was scored on 2003+ data: cannot be certified free of
  look-ahead, but it ranks 8th of 8 XGBoost arms on post-2002 MSE and the paper's QLIKE, so not performance-chosen.
  To state in the paper's methods (D.19 / trees).

## 73. The dev-tuned random forest replaces the default forest — APPLIED (GUI); paper in progress

- Tables `yhat_subtree_daily_<set>_rfdev` (T1) and `yhat_subtree_rfdev_<set>` (T10), sets baseline / all_features /
  lean_vol_vix; scores `results/trees_rfdev_1600/SUMMARY.md`. QLIKE below the default forest in 6 of 6 cells (DM
  significant on all features T1 / T10 and lean T10); against LightGBM / XGBoost no DM significant. Sharpe mixed
  (all features T1: 0.85 vs 1.52 default).
- GUI (`rows.py` `_RFDEV_PAIRS`, `_rfdev_pairs`, `_swap_rfdev_rows`; `_read_daily` swaps the series): the forest rows
  keep their keys and names and carry the dev-tuned series; `*_rfdev` rows are never listed; the description says where
  the settings come from (`_RF_NOTE_DEV`; `_RF_NOTE_DEFAULT` if a dev table is missing). Verified on the live GUI:
  all features 0.0954 / Sharpe 0.85, HAR + cal. 0.0979 / 1.07, equal to the tree agent's scores.
- Paper: numbers agent swaps the model set and re-runs (flag `writeup/generated/MACROS_RFDEV_DONE`); text agent writes
  the settings provenance and the XGBoost disclosure.

## 74. Trial-163 forest shown next to the dev-tuned forest (preview, temporary) — APPLIED (GUI only)

- Rows `subtree_daily_<set>_rfp163` ("... (trial 163 preview)"), refit every session, built in `rows.py`
  (`_preview_daily`, `_add_preview_rows`) from `results/trees_rfdev_preview163/` (outside spxw_pnl; the master table
  never sees them). Gate: the same construction on the dev forest's tables reproduces the master table's rows.
- Served (sign(s) / robust clip): HAR + cal. 0.0982 / Sharpe 1.00 / 1.36; all features 0.0951 / 1.31 / 1.54 (the dev
  forest beside it: 0.0979 / 1.07 / 1.40 and 0.0954 / 0.85 / 1.44).
- Temporary: when the 400-trial freeze lands, these rows are removed (or become the forest rows if trial 163 wins).

## 75. The trial-163 forest is the GUI's random forest (preview); the dev-tuned forest is off — APPLIED (GUI only)

- `rows.py`: `_read_daily` puts the trial-163 series under the forest's own keys (`subtree_daily_<set>_rf`), replacing
  the dev-tuned (rf_opt_00) series; `_add_preview_rows` now updates those rows' QLIKE and description
  (`_PREVIEW_NOTE`) instead of adding rows (item 74's separate rows and colour are gone from the chart).
- Served: HAR + cal. 0.0982 / Sharpe 1.00, all features 0.0951 / 1.31 (sign(s)); pink forest dots, 76 points / 19 models.
- The paper is unchanged (its forest is still rf_opt_00). When the 400-trial freeze lands, the GUI forest becomes the
  final entry (trial 163's series stay if it wins).

## 76. XGBoost takes its top 1998-2002 entry (xgb_opt_00) — DECIDED (user, 2026-10-07); runs in progress

- Replaces `xgb_opt_06` (7th-ranked, hand-picked about 2026-08-14 with no recorded reason after post-2002 scoring),
  so all three trees follow one rule (the top dev-span entry) and the look-ahead disclosure goes away.
- Per-bar close study: fixed settings T1 / T10 and the 13 re-tuning families + unmasked on baseline, all_features,
  lean_vol_vix, from xgb_opt_00 (dev MSE 0.107894). Other places that use xgb_opt_06 (main results' deck etc.) are
  scoped first and reported before any re-run.

## 77. The forest's 400-trial search frozen — RECORDED

- `results/rf_devmenu_2026-10-06/budget400/rf_config.json`: top entry of the completed trials numbered 0..399 = trial
  255 (n_estimators 485, max_depth 20, min_samples_leaf 12 -> 1 for the 2,000-row window, max_features 0.9013,
  max_samples 0.4130, bootstrap); dev MSE 0.10845556 (trial 163 0.10847958; the 87-trial pick 0.10864383).
- Every trial 0..399 ran on EPYC 7513. 6,224 allocated CPU-hours for the whole search. The 87-trial config is kept.
- Next: the per-bar forest batch on this entry (token `rf400`), then it replaces the forest in GUI and paper.

## 78. The GUI's forest switches to trial 255 automatically — APPLIED (GUI only)

- `rows.py` `_preview_source()`: the shown forest rows (refit every session) read `results/trees_rf400_1600/
  yhat_subtree_daily_<set>_rf400.parquet` once those exist for HAR + cal. and all features, else the trial-163 preview
  (items 74-75). Description `_PREVIEW_NOTE_400` for trial 255. A scratchpad watcher (`rf400_gui_watch.py`) restarts the
  GUI when the tables land.

## 79. GUI: XGBoost on its top entry everywhere; forest on trial 255 — APPLIED (GUI only)

- `rows.py` `_tree_swap_table` / `_preview_daily` / `_add_preview_rows`: every per-bar XGBoost and forest row (sets
  baseline / all_features / lean_vol_vix) whose replacement table exists carries it under its own key and name:
  XGBoost from `results/trees_xgbtop_1600/` (6 fixed-settings + 39 re-tuned keys), the forest from
  `results/trees_rf400_1600/` (6 fixed-settings keys now; the re-tuned ones as they land), else the trial-163 preview.
  Gate per key: the same construction on its master-table table reproduces the master table's daily series.
- Served (sign(s)): XGBoost T1 0.0976 / 0.95 (HAR + cal.), 0.0935 / 1.12 (all features); forest 0.0968 / 0.66, 0.0966 / 0.85.
- Paper unchanged so far: both trees go in together (one regeneration) after the forest's re-tuning batch.

## 80. Pooled trees in the GUI; 15:30 XGBoost to the top entry; pooled runs for the missing trees — DECIDED (user) / in progress

- GUI: the master table's pooled `lgbm` / `xgb` rows (the main results' pooled trees, all 48 bars, all features) are
  shown as "pooled tree" rows `pooltree_lgbm_all_features` / `pooltree_xgb_all_features` (renamed before the parked
  "paper" family filter; sized from the plain daily series, not the deck's recalibrated file). New pooled runs in
  `results/pooled_trees_2026-10-07/yhat_pooltree_<est>_<set>.parquet` are added automatically (template row).
  Served: pooled LightGBM QLIKE 0.1060 / Sharpe 1.43; pooled XGBoost (arm 16 until the switch) 0.1076 / 1.12.
- 15:30 switch to bank arm 10 (xgb_opt_00): `results/spxw_pnl/yhat_tree10.parquet` dumped and verified; an agent
  switches `notebooks/atm_straddle_lib.py` and regenerates Table 7, bounds, D.16-D.18, running-portfolio, Murphy 15:30,
  the feature-importance study and the figure.
- Pooled walks submitted for the missing trees: forest (trial 255) on all features; LightGBM, XGBoost (xgb_opt_00) and
  forest on HAR + cal. and lean.

## 81. Pooled random forest refits every bar — DECIDED (user, 2026-10-07)

- The pooled forest (trial 255, the 400-trial top entry, no leaf scaling) runs the bank's exact walk: refit every bar
  on the trailing 24,000 bars, on HAR + cal. (52-column pooled backbone), lean (196) and all features (1,144);
  ~141,000 CPU-h (measured on CARC: 5.8 / 27.4 / 197 s per refit at 8 threads). Forest predicts single-threaded
  (sklearn's multi-threaded predict sums trees in thread-finish order; the fit is reproducible).
- Pooled LightGBM / XGBoost on HAR + cal. and lean: new every-bar runs (~2,100 CPU-h); no re-runs of the existing
  all-features pooled rows and no once-per-session variants (user: "why do the gradient boosting methods need to be
  rerun again?" -- they do not).
- 15:30 switch to bank arm 10 done (agent report; changelog 8.17): Table 7 XGBoost QLIKE 0.1076 -> 0.1063, Sharpe
  1.12 -> 1.26; the frozen 15:30 Murphy table in the parked running appendix is left as is.

## 82. Feature-importance study: forest to trial 255; meeting PDF rebuilt — DECIDED (user, 2026-10-08); in progress

- The 16:00 importance study's trees now follow the top-entry rule too: XGBoost xgb_opt_00 (refit 2026-10-07 21:15-21:36
  by the 15:30-switch agent), forest -> trial 255 of the 400-trial search (min leaf 12 -> 1; predict single-threaded),
  baseline and all_features refits; stored default-forest results kept.
- `writeup/feature_importance_1530_dedup.pdf` rebuilt all_features only: interim now (forest still defaults), final
  after the forest refits; the 2026-10-06 PDF backed up. Paper macros untouched (paper integration deferred).
- On disk now (stale in the other thread's notes): XGBoost tail MSE HAR+cal 0.06337 vs all features 0.06509 (wins
  89/147), the mean driven by the 2020-02-28..2020-03-26 crash tails (refits 42-43); without the worst 3 tails all
  features is 9% lower. LightGBM (0.06588 vs 0.06220) and the default forest (0.06627 vs 0.06584) reproduce.

## 83. Importance PDF final: trial-255 forest; clusters / de-dup / gates commented out; linear over-time figure — DONE

- `writeup/feature_importance_1530_dedup.pdf` (7 pages, 0 errors): all features only; XGBoost xgb_opt_00; forest trial 255
  (refits `_work/trees/trees_*_rf255_*.npz`, gates `gates_rf255.csv`; default-forest files kept). Commented out in
  `writeup/make_feature_importance_1530_tex.py`: the cluster unit, the clusters column, section 5 (dilution), the de-dup
  intro and title line, the gates and before / after sections, the cluster-rank appendix; sections renumbered 1-7.
- `writeup/feature_importance_1530_dedup_sections.py`: `over_time_figure` generalised; second figure
  `writeup/figures/feature_importance_1530_over_time_linear.*` (ridge / lasso: SHAP and |beta x sd| shares and ranks);
  caption states one point per refit (every 10 sessions) on its held-out tail.
- Tail loss HAR + cal. vs all features (147 refits): forest trial 255 MSE 0.06152 vs 0.06352 (81 wins), QLIKE 0.12740 vs
  0.12567 (79); without crash refits 42-43 MSE 0.05920 vs 0.05804. har_ma share (MDI / SHAP): forest 58.7 / 61.4 %.
- Caveat: the study's linear models are the old one-penalty, causally re-tuned ones; the lasso's har_ma share flips
  60 % -> 15 % -> 60 % with its 250-session penalty re-choices (2021-23: small penalty, 115-120 columns).

## 84. Lean set parked everywhere; an MCS over the features instead — DECIDED (user, 2026-10-08, professor's suggestion)

- Lean set: removed from the paper model set and commented out in the text / tables / macros (numbers agent ->
  `MACROS_NOLEAN_DONE`, then the text agent). The GUI already hides lean. Per-bar tree changes stay deferred.
- Feature MCS (Hansen, Lunde & Nason 2011): candidates = HAR + cal. and all features (anchors) plus, for every exogenous
  series X, ADD-ONE (HAR + cal. + X) and LEAVE-ONE-OUT (all features - X); probe estimator = the paper's pre-tuned
  per-bar elastic net (arm f), each design with its own pre-2020 penalty; all on one CPU class. Losses: QLIKE (primary)
  and the sign(s) trade loss (minus daily mid P&L). MCS alpha 0.10 / 0.25, B = 10,000, blocks 5 / 10 / 21; the MCS
  bootstrap restored behind an explicit argument (defaults unchanged). Trees decided after the probe.
  Outputs `results/feature_mcs_2026-10-08/`.

## 85. Feature MCS: elastic-net result; more powerful losses; ridge and lasso — RESULT / in progress

- Elastic net (96 candidates, arm f, EPYC 7513 reference-BLAS kernel, gate bit for bit): QLIKE MCS keeps all 96 at 10 % /
  25 % (T_max, every block); the trade-loss MCS drops a few add-one designs (buy / sell / total turnover, ES volume, VIX).
  Pairwise DM: `sumabsret_vwstock` the only input significant both ways (add -2.9 %, DM -3.38; remove +1.5 %, DM +3.03).
- User 2026-10-08: add fit-space squared error and the threshold score (c = 1, IV-scaled) as MCS losses; run the same
  96-design analysis for the pre-tuned ridge and lasso, plus a joint MCS over all 288 linear candidates.

## 86. Feature MCS: linear result final; trees next (CARC) — RESULT / in progress

- Joint MCS over all 288 linear candidates (ridge, lasso, elastic net x 96 designs): keeps every candidate on QLIKE and
  fit-space MSE; drops 2-5 add-one turnover designs on ES(1) / trade. Power limited by the 866 independent days (top 5 %
  of days carry 31-43 % of every loss). Pairwise, consistent across estimators: ES volume and value-weighted stock
  absolute returns help added alone (DM -2.4 .. -3.6); removing the latter costs QLIKE in every estimator.
- User 2026-10-08: run the trees on CARC -- the same 96 designs for LightGBM (lgbm_opt_00), XGBoost (xgb_opt_00) and the
  forest (trial 255), fixed settings, refit every session, 866 trade days; then per-tree, joint-288-tree and grand-576
  MCS on the four losses. Outputs `results/feature_mcs_2026-10-08/trees/`.

## 87. Feature MCS with trees: grand MCS over 576 candidates — RESULT

- Trees (LightGBM lgbm_opt_00, XGBoost xgb_opt_00, forest trial 255; fixed settings, refit every session, 96 designs
  each; anchors bit-identical to the stored T1 runs; 875 CPU-h on CARC): no joint MCS (288 trees, 576 all) singles out
  an input set on any loss. The grand MCS has power only on fit-space MSE, where it drops weak LightGBM add-one designs
  (trees fit the adjusted target worse: median MSE 0.054-0.055 vs 0.050-0.051 linear) -- a model-family effect, not an
  input effect. Steadiest single exclusion: `add_ofi_ewstock` (LightGBM / XGBoost, QLIKE).
- Pairwise DM, consistent across linear models and trees: ES volume and the stock cross-section's absolute returns help.
- Results `results/feature_mcs_2026-10-08/SUMMARY.md`; nothing in the paper yet.

## 88. SPA / Reality Check / StepM against HAR + calendar — RESULT (2026-10-12)

- MCS with tail-tamed losses (log-MSE, winsorized QLIKE / trade) gains power but only drops weak single-series add-ones;
  HAR + calendar never excluded (the MCS eliminates from the bottom). The hit loss has no power.
- SPA (Hansen 2005; studentized, circular block bootstrap B = 10,000, blocks 5 / 10 / 21; gate: bootstrap is not the
  identity) REJECTS "nothing beats HAR + calendar" for ridge, lasso, elastic net and the grand set vs the paper's OLS
  on QLIKE, MSE fit and winsorized QLIKE (grand QLIKE p 0.016-0.033); log-MSE for enet / lasso / LightGBM / XGBoost;
  trade / hit only for XGBoost; never on ES(1); the forest never at 5 %. White's RC (unstudentized) rarely rejects.
- Romano-Wolf StepM (FWER 5 %) names add-one designs: `add_sumabsret_vwstock` (28 times), `add_sumvolume` (28),
  `add_sumabsret_ewstock` (8), VVIX (6), `add_sumpret2` (6), ...; never all features or a leave-one-out design.
  Grand vs the OLS: enet + `sumabsret_vwstock` and ridge + `sumvolume` beat it on QLIKE and winsorized QLIKE.
- Files: `experiments/feature_mcs_spa.py`, `results/feature_mcs_2026-10-08/{spa_rc,stepm,spa_tstats,spa_gates}.csv`.

## 89. Importance PDF: split count, P1 and the linear QLIKE permutation commented out — DONE (user, 2026-10-12)

- Why: split count counts splits without gain, is summed over a series' columns (voldemand 18 vs har_ma 6) and tracks
  cardinality -- the trial-255 forest's split rank put voldemand_spx above har_ma (0.031 vs 0.031) while its MDI / SHAP /
  P2 ranks are 18 / 28 / 25; P1 (shuffle within the 10-session tail) is blind to slow inputs; the linear QLIKE
  permutation is dominated by single tails. Scale plays no role (tree measures are scale-free; linear uses |beta sd|).
- `writeup/make_feature_importance_1530_tex.py`: TREE_M / LIN_M without them; the measures paragraph, the cardinality,
  noise-probe and agreement tables, the reading paragraph; the ranked-bars and rank-heatmap figures commented out (they
  include the dropped measures); section 5 (linear QLIKE permutation) commented out; sections 1-6. PDF 5 pages, clean.
- 2026-10-12 follow-up (user): the ranked-bars and rank-heatmap figures are REDRAWN without split count, P1 and the
  linear QLIKE permutation (`experiments/feature_importance_1530.py` figures(): trees MDI / perm P2 dMSE / perm P2
  dQLIKE / SHAP, linear |beta sd| / drop-column / perm P2 dMSE / SHAP; redrawn from the stored by_measure tables, no
  aggregate re-run) and restored in the PDF with updated captions. PDF 7 pages, 0 errors / overfull / undefined.

## 90. Importance PDF Section 1 reframed: "How much do the inputs below the leader matter?" — DONE (user, 2026-10-12)

- User: the cardinality-bias question ignores how much the inputs below the leader matter in absolute terms. New
  Table 2 (computed from by_measure + noise_probes): permutation P2 per model and loss -- har_ma's value; below-HAR
  series with interval > 0 and above the pure-noise probe; their summed and largest importance as % of har_ma's.
  Trees: 7-8 series significant on QLIKE (4-6 above the noise probe), together 4-11 % of the ladder (1.5-4.4 % on
  squared error), the largest at most 3.2 %; ridge 20 series / 77 %, lasso 20 / 24 % (the penalty spreads the
  ladder's weight). The cardinality tables follow as a note ("Cardinality bias"). PDF 8 pages, clean.

## 91. Per-bar bucket-by-bucket study (professor's story) — RUNNING (user, 2026-10-12)

- Story to test: "each bucket alone didn't help much, all features together did". Evidence so far is mixed: on QLIKE
  single series (ES volume, value-weighted abs. returns) beat HAR + cal. (SPA / StepM) while all features is never
  named; on the trade all features is higher in 48 / 51 models (4 significant).
- Designs: for the 8 buckets partitioning all features (moments, liquidity, market_ew, market_vw, sentiment,
  implied_vol, vol_demand, fomc): add-bucket (HAR + cal. + bucket) and leave-bucket-out (all features - bucket);
  6 estimators (pre-tuned ridge / lasso / enet on EPYC 7513; LightGBM, XGBoost, forest trial 255 fixed, refit every
  session); 866 trade days. Analysis: DM / HAC t vs HAR + cal. and vs all features on QLIKE / winsorized QLIKE / MSE
  fit / Sharpe; SPA + StepM over {8 add-bucket + all features} vs HAR + cal., and over the 8 leave-outs vs all features.
- Pooled forest all features: CARC queue drained since 10-10 (4,110 pieces done, 13 timeouts, 28 held on a CARC
  env-retrieval failure); the pooled agent is refilling (nice >= 100 so the bucket study goes first).

## 92. PufferLib neural forecaster in place of the LSTM — DECIDED (user, 2026-10-12); Phase 1 running

- User: use PufferLib (PuffeRL PPO + LSTM, vectorized envs) for iteration speed; vmap was tried and rejected.
- Framing: forecasting as a contextual bandit -- episode = a training window's sessions in order, observation = the
  session's features, action = the adjusted-scale forecast, reward = minus its loss (fit-space squared error; QLIKE
  option); causal walk-forward on the 2,000-session window; exploration noise measured / policy mean at evaluation.
- Phase 1 (feasibility, cluster GPU node): install, ForecastEnv, train one window (baseline, all features), speed and
  accuracy vs the per-bar linear / LightGBM forecasts, projection for the walk-forward. Phase 2 after review.
- 2026-10-12 (user, firm): the model is PufferLib's NATIVE minGRU network, trained directly on the forecasting loss through the native module (the bandit / PuffeRL framing only if the native minGRU can only be driven through PufferLib's trainer).
- 2026-10-12 (user): PPO dropped entirely (no PuffeRL arm, no bandit env). PufferLib's native minGRU trained purely supervised, keeping PufferLib's speed (native network / kernels, batching; many walk-forward windows trained concurrently where its RL-only components don't apply).
- 2026-10-12 (user, verified): PufferLib 5.0's native C/CUDA trainer (MinGRU + highway nets, cuBLAS matmuls), NO PyTorch; the PPO objective in the native backend is replaced by the supervised forecasting loss (minimal vendored fork if needed).
- 2026-10-12 (user): robust gradients -- log-variance (or positive, floored) output; fit-space squared error default, QLIKE with a per-sample gradient cap; global grad-norm clipping + warmup; fp32 loss / reductions with NaN-Inf guards; finite-difference gradient check vs numpy and a toy end-to-end check; causal standardization; deterministic reductions (bit-identical re-runs on the same GPU).
- 2026-10-13 (user): the target is the repo's SQRT-space adjusted target (log space blew up when sumret2 <= 0); output f in sqrt space, plain v = f^2 B; default loss squared error in sqrt space; QLIKE optional with the repo's floor convention. Robustness requirements apply only to code we introduce (loss kernel, gradient, data feeding, glue); PufferLib 5.0's own kernels / optimizer are trusted and left as shipped.
- 2026-10-13 linear result (ridge / lasso / enet x 16 bucket designs, EPYC 7513, gates bit for bit): the story FAILS on
  QLIKE and winsorized QLIKE -- all features never beats HAR + cal. (pooled DM -0.23 / -0.66) while `addb_market_vw`
  (value-weighted cross-sectional returns) does (lasso DM -3.75, SPA p 0.002; winsorized pooled DM -2.96), and removing
  market_vw from all features is the only removal that hurts. Trade: all features higher Sharpe (pooled 1.86 vs 1.37)
  but HAC t 0.97 -- directional only. `addb_liquidity` is significantly WORSE than HAR + cal. (QLIKE +8..15 %).
  Trees pending (BUCKETS_DONE).
- 2026-10-13 Phase 1 DONE: native supervised MinGRU in PufferLib 5.0 (no PyTorch, no PPO) works, verified vs float64 reference, bit-identical, ~0.1 s per fit; projection ~2 h for 96 designs (refit every session, tune every 250). Accuracy on one window mixed. Summary: `results/puffer_forecast_2026-10-12/SUMMARY.md`. Phase 2 awaits the user.
- 2026-10-13 Phase 2 submitted on Hoffman2 as job 15152085 (`pf_phase2`, one L40S, 2 h). Walk-forward of the two Phase 1 designs only (`baseline`, `all_features`): 1,088 origins from 2020-01-02 through the last 16:00 row, cold refit every session, full 12-config grid re-tuned every 250 sessions, 5 seeds, fc_n=1. Block 0's selector, pointed at the Phase 1 preds, reproduces the Phase 1 configs (baseline L20_H16_lr0.01, all_features L20_H16_lr0.03). The duplicate Phase 1 job 15144888 was deleted while still queued. CARC run of record remains runs/13037177.
- 2026-10-13 (user): speed comparison must be apples to apples (not the old CPU LSTM): one job on one GPU timing native PufferLib MinGRU (K = 1/8/24/48 concurrent), PyTorch eager, torch.compile (reduce-overhead / max-autotune) and batched PyTorch (vmap / stacked, eager + compiled) on the same jobs and epoch counts; Hoffman2 GPU (the other session owns CARC and its eager pf_torch job). Results -> results/puffer_forecast_2026-10-12/bench/.
- 2026-10-13 INTERIM (ridge, lasso, enet + LightGBM): the story still fails. LightGBM alone: no StepM rejection on any
  loss; only `addb_moments` significant pairwise (QLIKE -5.7 %, DM -2.01); all features -4.7 % (DM -1.43, n.s.); Sharpe
  0.79 (HAR + cal.) -> 1.33 all features (t 1.11) / 1.61 moments (t 1.48), n.s. Pooled over 4: no StepM rejection;
  `market_vw` still the pairwise standout (add DM -2.05; remove +2.14 winsorized); all features DM -0.54, Sharpe 1.91
  vs 1.33 (t 1.18). XGBoost / forest pending (Hoffman2).
- 2026-10-13 FULL bucket story check (6 estimators + pools; results/feature_mcs_2026-10-08/SUMMARY.md "The bucket
  story check", buckets_*.csv): the story fails on every loss, every estimator, every pool. All features never beats
  HAR + cal. jointly (StepM). Significant gains are single buckets: market_vw on QLIKE (linear); trade, trees,
  pairwise only: XGBoost liquidity (Sharpe 1.67, t 2.05) / market_ew (1.65, t 2.39) vs all features 1.12 (t 0.30)
  vs HAR + cal. 0.95; forest moments / market_ew (1.47, t 2.35-2.38) vs all features 0.85 (t 0.34) vs 0.66 -- the
  forest's all-features design is its weakest trade design. Six-estimator pool: all features DM QLIKE -0.65, Sharpe
  1.67 vs 1.22 (t 1.00). Five-estimator pool: SPA says removing a bucket hurts on QLIKE (implied_vol, market_vw
  pairwise) and dropping liquidity raises the trade Sharpe 1.64 -> 2.03 (t 2.73).

## 93. LSTM parked everywhere in the paper (double-standardisation bug) — APPLIED (paper; GUI untouched)

- **Decision (user, 2026-10-13):** "get rid of LSTM for now". The per-bar LSTM's `standardise()`
  (`specs/causal_tune_lstm_jobs.py:92-96`, guard only at sd == 0) re-standardises the executor's prescaled design per
  window; near-constant `*_avail_ma_*` fractions blow up (window sd ~1e-5 -> |z| > 1,000).
- **lstmi:** same `standardise()`, no floor; its step matrix is not prescaled and has no `*_avail_ma_*` columns, so the
  known trigger is absent (0/1 indicators bounded at |z| ~150-310), but median-filled continuous `adj_<x>` columns can
  blow up the same way (not measured). Parked either way (already so since 2026-10-06).
- **Paper:** the four remaining "not reported" LSTM sentences parked (methods, 5.4 figure caption, D.19 intro and
  compute); D.19 compute row recomputed without the LSTM jobs (186.4 -> 115.2 allocated CPU-h). Every model-set
  statistic already excluded the LSTM (2026-10-06); no MCS / SPA / StepM, pool or combination in the paper held one.
  Standalone master-table PDFs: LSTM rows / lines dropped. Changelog 8.19.

## 94. Per-bar trees on their top entries in the paper (xgbtop, rf400) — APPLIED (paper; GUI untouched)

- Items 76-79 carried into the paper in one regeneration: every per-bar XGBoost key reads `results/trees_xgbtop_1600/`
  (xgb_opt_00), every per-bar forest key `results/trees_rf400_1600/` (trial 255); the `rfdev` rows leave the model set
  (still 112 forecasts). Paper-only switch in `experiments/master_table_close.py` (`PERBAR_SWITCH`, active only when
  `OUT` is the paper's `results/close_master_table`); D.19 re-scores its pairs on the same tables.
- XGBoost look-ahead disclosure parked; methods now say all three trees take their search's top entry; the forest's
  tuned rungs start from its shipped settings (re-centred grid, Optuna trial 1). Refit-contract sentence (indices
  00 / 16) corrected to "the top LightGBM and the top XGBoost entry" (stale since 8.17).
- Moved: XGBoost T1 all_features 0.0942 / 1.64 -> 0.0935 / 1.12; forest T1 all_features (rfdev) 0.0954 / 0.85 ->
  0.0966 / 0.85; model-set median Sharpe 1.10 -> 1.08; "daily refits lower every tree's QLIKE" FLIPPED (5 of 6) and
  reworded by macro.
- Pending: XGBoost's in-sample Optuna budget / picks rows (xgbtop stage-1 `trials_bar1600.npz` not pulled); unmasked
  RS10 / RS1 on the top entries never run (8 mask pairs shown as ---). Changelog 8.20.

## 95. Bucket story check written up (new appendix subsection B.17 + 5.4 pointer) — APPLIED (paper)

- `writeup/make_buckets_tex.py` -> `generated/buckets_numbers.tex`, `generated/table_buckets_story.tex`;
  `sections/appendix_buckets.tex` at the end of D.19 (where the lean subsection was); "Bucket by bucket." in 5.4.
- Reading: the story fails in all 30 family-loss cells; StepM never names all features; the significant gains are
  single buckets (`market_vw` on the linear models' QLIKE; moments / `market_ew` / liquidity on the trees' trade,
  pairwise only; XGBoost `market_ew` named by StepM at blocks 10 and 21, not 5). Correction to item 92: the forest's
  all-features design is not its weakest trade design (HAR + cal. 0.66, two add-bucket designs 0.68). Changelog 8.21.

## 96. Bucket write-up parked (user, 2026-10-14) — APPLIED (paper)

- `sections/appendix_campaign_16h.tex`: `\input{sections/appendix_buckets}` commented out; `sections/results_close_option.tex`:
  the "Bucket by bucket." paragraph commented out (dated notes). The generator `writeup/make_buckets_tex.py`, the
  macros `generated/buckets_numbers.tex` (still read, silently), the table and `sections/appendix_buckets.tex` stay in
  place, so un-parking is two lines. Scratch build: 75 pages (77 with the appendix), 0 errors / overfull / undefined.

## 97. XGBoost's Optuna trial rows on its top entry back in the paper — APPLIED (paper)

- The xgbtop stage-1 records (`results/linear_subsection_trees_optuna_mask_xgbtop/stage1/<set>/bar1600/xgb/tw2000/
  trials_bar1600.npz`, pulled from CARC, md5-verified) close item 94's pending part: D.19's budget and picks tables
  carry LightGBM, XGBoost (xgb_opt_00) and the forest (trial 255) again, each tree's picks on its own run's search
  space. Text back to "each arm" / "The penalties lambda_l2 and reg_lambda" (interim wording parked).
- Only `\campOptGainTwentyFiveMedHi` moves (1.1 -> 1.2); the 5.4 budget range (27-37 %) is unchanged. Bucket
  write-up still parked (item 96). Scratch build 75 pages, clean. Changelog 8.23.

## 98. Per-bar PufferLib MinGRU rows in the GUI — APPLIED (GUI only)

- **User (2026-10-14):** "can I get the gui updated with the per bar NN numbers?" Three walks, each on HAR + cal. and
  all features, family "neural net (MinGRU)" (nonlinear group, on by default, brown on the chart): minGRU (identity)
  = Phase 2c (job 13082282; walk_identity_p100, job 13089745, has the same forecasts up to 1 ulp in one row);
  minGRU (softplus) = walk_softplus_p100 (job 13089746); minGRU (softplus + wd, tuned) = wd_study walk_wdgrid
  (job 13135342, wd in each block's tune grid {0, 0.01, 0.03, 0.1, 0.3, 1}). All on one Tesla P100, forecast = the
  5-seed mean f.
- **rows.py:** `_nn_daily()` builds plain = f^2 x B from each run's phase2_score `forecasts.csv` (row -> date through the
  feature study's inputs.json, B from the close-study export's targets.parquet, target / iv_var / quotes as every other
  row); `_add_nn_rows()` adds the rows; every GUI column (QLIKE, MSE, MAE, OOS R², DM, Sharpe, return distribution,
  the sizing rules) comes from the GUI's own scorers. Keys `mingru_{identity,softplus,wdtuned}_{baseline,all_features}`.
  describe.py: one sentence per walk (native MinGRU + Muon, 12-config grid, 5 seeds, causal re-tune every 250
  origins, cold refits, P100) plus the link / wd note. index.html: chart colour for "minGRU".
- **Gate:** the GUI's QLIKE, Sharpe mid / crossed, % buy and n, and the fit-space MSE rebuilt from the GUI's plain
  series, reproduce each run's phase2_score `scores.csv` to 1.6e-15 relative (stop line 1e-9); every row has the
  baseline row's 866 trade days. The fixed-wd walks (wd_study A) are not shown: the GUI has no collapsible group for
  them.
- **Also fixed (needed for the restart):** an interim pooled-forest table (`yhat_pooltree_rf_all_features_INTERIM1600`,
  2026-10-13 22:08) was being added as a row with no phrase in describe.py, so every `/api/table` request failed after a
  restart; `_preview_daily` now reads only `yhat_pooltree_<est>_<set>.parquet`.

## 99. GUI: pooled weight-decay MinGRU on all features; the non-weight-decay MinGRU rows parked — APPLIED (GUI only)

- **User (2026-10-15):** "update the gui with the pooled weight decay all features. also get rid of the non weight decay
  versions of NN". rows.py `_NN_RUNS`: the softplus (no wd) entry parked with a dated note, as the identity one was;
  the per-bar `wdtuned` rows stay (both input sets).
- **New row** `mingru_pooled_wdtuned_all_features`, "minGRU pooled (softplus + wd, tuned)", family "neural net (MinGRU)",
  fit "All 48 bars (pooled)" (a square on the chart): the pooled walk with wd chosen in each block's tune grid
  (selection B, `results/puffer_forecast_2026-10-12/pooled_wd_study/runs/pwalk_wdgrid_13138604`, CARC job 13138604,
  P100, trainer native_g6 md5 15cd4521). Built as `experiments/puffer_forecast/pooled_score.py` view 2 does: the seed
  mean of every block's refit forecasts at the panel's 16:00 bar, plain = f^2 x that bar's baseline
  (`_work/pooled_walk/rows.csv`), on the 866 trade days. describe.py: one sentence (pooled 48-bar panel, trailing 24,000
  bars, 8 configurations x 6 wd at each block start on a 2,400-bar tail, 3 seeds, refits every 21 sessions, softplus,
  P100, scored at 16:00).
- **Gate:** the GUI's QLIKE, Sharpe mid / crossed, % buy, n and the rebuilt fit-space MSE equal `at1600_scores.csv`
  (B, all_features: 0.1174 / 1.83 / 1.32, fit MSE 0.0554) and the per-bar wdtuned rows equal
  `wd_study/score_wdgrid/scores.csv`, to 8.9e-16 relative (stop line 1e-9).
- **GUI:** restarted the launch.pyw way; the catalog serves exactly mingru_wdtuned_baseline, mingru_wdtuned_all_features
  and mingru_pooled_wdtuned_all_features.
- **2026-10-15 follow-up (user: "can I get pooled mingru har+cal back on there for comparison?"):** + the same pooled
  selection-B walk on HAR + cal. (`mingru_pooled_wdtuned_baseline`, same source and construction; `_NN_POOLED` sets
  now baseline and all_features). It equals `at1600_scores.csv`'s B baseline row (QLIKE 0.1134, fit MSE 0.0607, Sharpe
  1.04 / 0.55) to 1.1e-15 relative on the same 866 days; it is joined to the all-features point on the chart. GUI
  restarted the launch.pyw way; the catalog serves the four MinGRU keys (wdtuned and pooled_wdtuned, baseline and
  all_features).

## 100. GUI: per-bar MinGRU with the exogenous-block decay (wd_enc, fix #1), honest selection walk — APPLIED (GUI only)

- **User (2026-10-15):** "update gui". Rows `mingru_wdenc_baseline` / `mingru_wdenc_all_features`, "minGRU (softplus +
  exogenous decay, tuned)", family "neural net (MinGRU)", per-bar: `_NN_RUNS["wdenc"]` reads
  `results/puffer_forecast_2026-10-12/wd_enc_study/score_wegrid/forecasts.csv` (CARC job 13142129, P100, trainer
  native_g7, .cu md5 12617b66d25f705e39ea5978cd261412), built exactly like the per-bar wdtuned rows. The walk: softplus,
  the Phase 2c 12-config grid x wd_enc in {0, 0.03, 0.3, 1, 3, 6, 15} chosen at each block start by the causal
  validation rule, 5 seeds; wd_enc decays only the encoder's input columns, scaled by the linear arms' penalty factors
  (exogenous 1, HAR + calendar backbone 0.01); bias, MinGRU and decoder not decayed. describe.py: the per-bar
  sentence plus one clause on the decay.
- **Gate:** QLIKE, Sharpe mid / crossed, % buy, n and the rebuilt fit-space MSE equal `score_wegrid/scores.csv` to
  1.6e-15 relative on the 866 days (all_features 0.1054 / 1.47 / 0.97; baseline 0.0990 / 1.49 / 1.01).
- The four earlier MinGRU rows stay. GUI restarted the launch.pyw way; the catalog serves the six MinGRU keys.

## 101. Per-bar uniform weight-decay MinGRU rows parked (user, 2026-10-15) — APPLIED (GUI)

- `master_table_gui/rows.py`: `_NN_RUNS["wdtuned"]` commented out ("get rid of non exogenous weight decay"). The GUI now
  serves `mingru_wdenc_{baseline,all_features}` (per-bar, exogenous-block decay) and
  `mingru_pooled_wdtuned_{baseline,all_features}` (pooled, uniform wd: the pooled study has no exogenous-decay run).
  wdenc provenance updated: re-run on the final g7 build (job 13143151) with byte-identical forecasts; gate 7 PASS.

## 102. GUI: per-bar MinGRU variant (b), ridge-initialised shortcut, honest selection walk — APPLIED (GUI only)

- **User (2026-10-15):** "can we update the gui?". Rows `mingru_rinit_baseline` / `mingru_rinit_all_features`, "minGRU
  (ridge-init shortcut, tuned)", family "neural net (MinGRU)", per-bar: `_NN_RUNS["rinit"]` reads
  `results/puffer_forecast_2026-10-12/ridge_init_study/score_rinit/forecasts.csv` (walk B, CARC job 13148540, P100,
  trainer native_g8, .cu md5 4ce6f65b8f1596afd526854dcbb2f332, binary f9f7d18b), built like the other per-bar rows.
  Model: f = softplus(c + s [v x_t + MinGRU branch]); v starts at each window's closed-form ridge (penalty factors HAR
  + calendar 0.01, exogenous 1), fine-tuned under L2 with its own Adam; the branch starts at zero with decoupled
  decay on its exogenous inputs; chosen in each block's tuning from 216 candidates (12 configs x wd_enc {0, 3, 15}
  x wd_lin {0.02, 0.2, 2} x lr_lin {0.0003, 0.001}). describe.py: one sentence.
- **Gate:** QLIKE, Sharpe mid / crossed, % buy, n and the rebuilt fit-space MSE equal `score_rinit/scores.csv` to
  1.6e-15 relative on the 866 days (all_features 0.1021 / 0.78 / 0.30; baseline 0.0987 / 1.47 / 0.99).
- Kept: wdenc and pooled_wdtuned (both designs). GUI restarted the launch.pyw way; the catalog serves the six MinGRU
  keys.

## 103. GUI: only the ridge-init MinGRU walks (MSE- and QLIKE-tuned); the other NN rows parked — APPLIED (GUI only)

- **User (2026-10-15):** "I also need the non ridge init stuff commented out (since we're updating the gui)". rows.py:
  `_NN_RUNS["wdenc"]` and `_NN_POOLED["pooled_wdtuned"]` commented out with dated notes (the pooled code path stays,
  now with an empty registry, for the pooled ridge-init walk being run). `rinit` relabelled "minGRU (ridge-init
  shortcut, MSE-tuned)".
- **New rows** `mingru_rinitq_baseline` / `mingru_rinitq_all_features`, "minGRU (ridge-init shortcut, QLIKE-tuned)":
  option 2 of the QLIKE round (trained on QLIKE, early-stopped on validation QLIKE, selected on validation QLIKE),
  `results/puffer_forecast_2026-10-12/qlike_study/score_rinitq/forecasts.csv` (CARC jobs 13150849 / 50 / 61 / 62 / 63,
  one per block; trainer native_g9, .cu md5 af1964d01576005365f553b98fb91d0c, binary 44ab6fff; P100). describe.py:
  the rinit sentence plus "trained, early-stopped and selected on QLIKE", trainer g9.
- **Gate:** QLIKE, Sharpe mid / crossed, % buy, n and the rebuilt fit-space MSE equal `score_rinitq/scores.csv` (and
  `ridge_init_study/score_rinit/scores.csv` for rinit) to 1.6e-15 relative on the 866 days (rinitq all_features
  0.0948 / 1.53 / 1.05; baseline 0.0985 / 1.04 / 0.57).
- GUI restarted the launch.pyw way; the catalog serves exactly the four MinGRU keys (rinit, rinitq x 2 designs).
- **Addendum (user, 2026-10-15: "comment out the mse-tuned stuff now as well"):** `_NN_RUNS["rinit"]` (MSE-tuned
  ridge-init, job 13148540) parked with a dated note too. After one restart the catalog serves exactly two MinGRU keys,
  `mingru_rinitq_baseline` and `mingru_rinitq_all_features` (still equal to `score_rinitq/scores.csv` to 1.1e-15). The
  pooled ridge-init QLIKE rows will be added later through `_NN_POOLED`.

## 104. GUI: the pooled ridge-init QLIKE-tuned MinGRU, both designs — APPLIED (GUI only)

- **User (2026-10-15):** add the pooled ridge-init QLIKE-tuned MinGRU. rows.py `_NN_POOLED["pooled_rinitq"]` reads the
  composite walk `results/puffer_forecast_2026-10-12/pooled_rinit_study/runs/pwalk_rinitq_composite` (CARC all_features
  blocks 13151980-99, baseline 13152000-09; P100; trainer native_g9, .cu af1964d01576005365f553b98fb91d0c, binary
  44ab6fff) at the pooled panel's 16:00 bar, as `experiments/puffer_forecast/pooled_rinit.py` score() does
  (pooled_score.walk_forecasts; plain = f[pos]^2 B[pos]). Keys `mingru_pooled_rinitq_{baseline,all_features}`,
  "minGRU pooled (ridge-init shortcut, QLIKE-tuned)", fit "All 48 bars (pooled)". describe.py: one sentence (shortcut
  at the window's ridge with pf 0.01 / 1, zero-init branch with exogenous decay, loss = early stop = selection =
  QLIKE, 8 configs x wd_enc {0, 3, 15} x wd_lin {0.03, 0.3, 3} x lr_lin {0, 1e-5}, 3 seeds, monthly refits on
  24,000 bars, P100, g9).
- **Gate:** QLIKE, fit-space MSE and Sharpe mid / crossed equal `pooled_rinit_study/at1600_scores.csv` to 1.1e-15
  relative, n = 866 (all_features 0.1121 / 0.34 / -0.13; baseline 0.1151 / 0.78 / 0.31); % buy is not in that file.
- GUI restarted the launch.pyw way (the first start now takes about a minute while the row cache builds); the catalog
  serves exactly the four MinGRU keys (rinitq and pooled_rinitq, both designs).
