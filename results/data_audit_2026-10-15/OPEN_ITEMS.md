# Data problems left open (2026-10-15)

On 2026-10-15 the user said "fix all 8 + 10" and "save the other data problems for me to investigate another day".

- **Adopted:** bugs 1-8 and #10, scoring against actual RV. They are being implemented in `src/` behind flags.
- **Left open:** everything below. Nothing here is fixed in the data, and none of it is in the re-run cascade.
- **Evidence:** the CSVs in this folder, produced by `experiments/data_audit_2026_10_15.py` and `experiments/data_audit_stages.py`.

## 9. Stock-file spikes, 2015-2018 (the largest open item)

- **What:** bars where the EW or VW stock-file "average squared return" jumps 50x to 7,000x relative to ES while ES is flat.
  - 2018-08-20 13:00: `sumret2_vwstock` 2.6e-3 vs ES `sumret2` 3.4e-7, about 7,500x. That implies a typical stock moving about 5 % in half an hour while futures moved 0.06 %.
  - 2015-09-03 12:30: 1.2e-2 vs 3.4e-6, about 3,600x.
  - Reading: consistent with one constituent's bad price dominating the average. The files are already averaged, so the stock and the cause can't be recovered from our side.
- **Size** (`stockfile_contamination_by_year.csv`):
  - The daily correlation of ES RV with VW-stock RV is 0.84-0.96 in every other year but 0.25-0.45 in 2015-2018.
  - The p90 of stock RV / ES RV is about 5x normally and 48-50x in 2016-2018.
- **Candidate rule** (`stock_spike_flags`, switch `spk`):
  - Flag a stock-file bar when log(stock RV / ES RV) has a robust z > 4 against all earlier history for the same half hour, while ES's own robust z is below 3 in absolute value. The rule looks only at earlier data.
  - Flags per year: 266, 543, 540 and 400 in 2015-2018; 60 and 42 in 2019-2020; 0-14 in every other year.
  - Blanking the flagged bars restores the correlation to 0.78-0.84.
- **Why it matters:** each 2020-2024 forecast trains on the previous 2,000 sessions, so 2015-2018 are in every test-period training window. 16:00 is the most-hit clock slot (116 of the 1,211 flagged regular-hours bars in 2015-2018). The pooled panel is affected too.
- **Open questions:**
  - Is it a vendor-file problem? Ask the data source / re-pull 2015-2018 from the per-stock data if possible.
  - Are the thresholds (4 and 3) reasonable? They must be fixed before any model result is looked at.
  - Blank the whole bar, or only the stock-moment columns (`sumret2/3/4`, `sumabsret`, `sumbipow`, `sumpret2`), keeping spreads and turnover?
  - Measured effect on the models: not run yet. Switch `spk` on top of the adopted bundle: `python experiments/data_audit_2026_10_15.py designs <bundle>+spk`, then `linear` and `score`.
- **Includes the old item (d):** the 2018-11-12 11:00 bar of both stock files. Its `spread_vwstock` was -0.000318 (the print that flipped the series to "signed"). VW `sumret2` is 3.9e-4 vs about 2e-5 on either side, and `sumret4` is 3.1e-5 vs about 1e-9. Blanking the invalid spread print is adopted; blanking the whole bar is open. See `../data_fix_2026-10-15/bad_print_neighbours.csv`.

## 11a. Isolated bad ticks in ES and stock series

- **Counts** (`bad_tick_summary.csv`, `bad_tick_candidates.csv`, `bad_tick_top.csv/png`): likely data errors, meaning isolated spikes while sibling series are flat.

  | Series | Likely errors |
  |---|---|
  | ES `RV` | 37 |
  | `sumabsret` | 28 |
  | `sumvolume` | 112 |
  | `sumabsret_vwstock` | 27 |
  | `effspread_vwstock` | 2 |
  | `spread_vwstock` | 2 |
  | `sumret2/4_ewstock` | 2-3 |

  The rest are market events (COVID, Volmageddon, the 2010-2015 flash events, the AP hack) or moves shared with siblings. There are also 51-78 "unclear" cases for RV and `spread_vwstock`.
- **Sensitive:** ES `RV` is both the target and the HAR feature. Options:
  - leave everything as it is;
  - blank these prints in the features only, keeping the target untouched;
  - also edit the target. Not recommended: the trade pays on the actual move.
- **To check first:** whether any of the 37 RV errors is a 16:00 bar inside the scored window 2020-01-03 .. 2024-04-30.

## 11b. Duplicate columns in the pooled panel

- **What** (`pooled_dup_groups.csv`, `pooled_dup_windows.csv`): each training window has 312-336 exact-copy availability-ladder columns, up to 24 copies of one column.
- **Effect** (`pooled_blk2_dedup.csv`):
  - Dropping the copies makes blk2 worse: QLIKE +0.44 %, DM 2.51, 4 positions changed.
  - The gate (dedup machinery reproduces the published blk2) passes at 5.8e-12.
  - Reading: the copies act as a lighter penalty on the availability block, so this is a modelling choice rather than a data error.
- **Open:** keep and document, or replace the copies with one column and an explicit penalty weight?

## 11c. Thin bars kept in the 30-minute grid

- **What** (`grid_audit.csv`, `grid_closed_window_bars_by_year.csv`):
  - 6,035 Friday bars after the 17:00 ET CME close (bar ends 17:30-20:00): median volume 38 vs 1,382 at Mon-Thu 19:00.
  - 9,039 bars in the daily 17:00-18:00 maintenance window: median volume 578 at 17:30 and 68 at 18:00.
- **Effect:** these enter the pooled panel as rows, and the HAR / MA windows in bar space count them.
- **Removing them** changes the 46-bar session grid (`BARS_PER_SESSION`) and every MA window. That is a structural change and needs a full rebuild.
- **Also seen:**
  - 935 bars with RV == 0, 2 of them at 16:00.
  - 8 NYSE full sessions without an ES 16:00 bar, all in 1998-2003, so outside the per-bar design.
  - 36 regular-hours bars missing from the ES tape, none from 2010-07-12 on.

## Minor, probably nothing

- **DST weeks:** 7 of 35 DST-transition weeks in 2007-2024 have the stock-file/ES correlation peaking off lag 0 (`align_checks.csv`). The stock-file/ES correlation is only about 0.37 at lag 0, so this is likely noise. The SPX-spot/ES check peaks at lag 0 in 9 of 12 weeks; corr 0.9985. Worth one look at those 7 weeks.
- **vvix missing sessions:** vvix has 14 missing sessions in 2013-2014 and vix3m has 3 in 2017 (`feed_missing_sessions.csv`). All are non-trade days, so there's no test-period effect.

## Added after the audit finished (2026-10-15)

- **Correction to 11a:** its counts came from the first bad-tick scan (`bad_tick_summary.csv`), which was mostly float
  zeros and holiday bars. The final scan (`bad_tick_candidates.csv`) finds 22 ES candidates, 11 of them likely errors.
  **None is a 16:00 bar in the scored window**, which answers 11a's "check first" question.
- **VIX family:** 74 jump-and-revert prints, 62 not explained by a known event.
- **Stock moments:** on pre-market bars in 2009-2011 they hold values of 1e-32 to 1e-64 where 0 is meant.
- **The 7 off-lag DST weeks** are all in 2015-2020, so they point to #9 (stock-file contamination), not DST.
- **Live signal:** pinned to the pre-fix chain (`live/close_signal/forecast.py: PIN_PRE_AUDIT_SRC = True`). Unpinning
  moves the 16:00 forecast by a median 5 % recently, and by up to 51 % over the last year. The user decides when.

## What has already been checked and found clean

- Time zones (UTC/ET) and DST: 866/866 entry stamps carry the right clock.
- The 16:00 bar matches the option's close: corr 0.9945, peaking at lag 0.
- Option expirations and early-close flags match the NYSE calendar.
- No 16:00 bar is missing after 2010.
- asinh routing is the same at every cutoff.
- First-window scaling touches no test rows.
