# Data audit 2026-10-15: register of adopted fixes (1-8) and their implementation

- **Decision:** the user said "fix all 8 + 10". #10 (score against actual RV) was done separately.
- **Not touched:** #9, #11, `spk` and `d1118`; see `OPEN_ITEMS.md`.
- **Implementation:** all 8 fixes are in `src/`. Each has its own module flag, default True, and the replaced code is parked with
  a dated note.
- **Status:** nothing committed; no cluster job run.
- **Detail:**
  - `register.csv`, `src_gates.csv`, `linear_scores.csv`, `linear_marginals.csv`, `live_flag_check.csv`;
  - the pooled v2 diff;
  - code in `experiments/data_audit_2026_10_15.py` and `experiments/data_audit_stages.py`.

## Register

| # | Problem | Fix | Flag (file:line) | Counts |
|---|---|---|---|---|
| 1 | Invalid prints; sign from the full sample | Null at load; sign from the series name | `loading.py:299 NULL_INVALID_PRINTS`, `target.py:170 SIGN_BY_NAME` | 21,939 prints nulled; 34 of 628 16:00 columns change |
| 2 | FOMC 0/1 flags logged and winsorized: fomc_release ladders constant; fomc_day 27.6x its indicator | Pass the flags through unchanged (identity) | `target.py:200 FOMC_FLAGS_IDENTITY` | 249 statement bars |
| 3a | `FFILL_LIMIT=26` never passed | `ffill_limit="auto"` by default, so it also reaches `prepare_full` | `executor.py:93 APPLY_FFILL_LIMIT` | voldemand ladders move on every test row (mean change 0.15) |
| 3b | VIX family ends 2024-02-12; on all 54 later trade days the availability ladders sat outside their training range | Yahoo hourly splice, historical panel only | `loading.py:419 SPLICE_VIX_FREE_FEED` | 1,476 / 702 / 648 bars; matches the panel on 96 / 88 / 100 % of overlapping stamps |
| 3c | `releases.parquet` is a stale pull with 3 FOMC statements missing, leaving a 123-trade-day dead tail | Add the 3 statements in code; 2024-05-01 as the next event | `loading.py:402 FOMC_CALENDAR_FIX` | 6,134 bars now observed |
| 4 | `_active` flags read off the forward-filled column; voldemand's mode-inflated class existed only because of the fill | Flag from observed prints; unobserved rows take the flag's observed mode | `executor.py:100 ACTIVE_FROM_OBSERVED` | voldemand loses 24 columns; StockTwits gains 18 |
| 5a | Neutral fill from the full-sample median | Expanding median of earlier observations | `executor.py:115 CAUSAL_NEUTRAL_FILL` | The fill moves by at most 0.22 IQR |
| 5b | Signed s.d. floor from the full sample | Expanding median of the lagged per-slot s.d. | `target.py:208 CAUSAL_SD_FLOOR` | 0-13k test rows per series |
| 5c | Mode-inflation class and mode from the full sample | Expanding over earlier prints, at least 250 | `executor.py:108 CAUSAL_MODE_INFLATION` | StockTwits series become hurdle series for 2010-16 |
| 5d | Guard point-mass test over the whole column (the vvix case) | Tested on earlier rows only, at each 480-row grid point | `executor.py:120 CAUSAL_GUARD_MODE` | 7 ladder decisions differ |
| 5e | asinh routing from the full sample | Pinned share counted from the series' first print, at least 250 rows; warm-up rows get 0 | `target.py:215 CAUSAL_ASINH_ROUTING` | Routing unchanged in practice |
| 5f | First window's scaler statistics | Kept: frozen to the first window, so no look-ahead | -- | 0 test rows |
| 6 | Early closes depress B; 22 post-close 16:00 rows were training rows | B from regular bars only; per-bar designs drop early-close sessions; the pooled panel keeps the rows | `executor.py:125 B_REGULAR_BARS_ONLY`, `:129 DROP_EARLY_CLOSE_SESSIONS` | B changes on 94 trade days (max 36 %); 28 sessions dropped |
| 7 | Per-bar erosion window counted in sessions | Count it in bars | `executor.py:135 ERODE_IN_BARS` | ref_iqr changes on 224 of 282 columns; columns stuck at the warm-up floor fall from 46 to 9 |
| 8 | Burn-in applied after slicing to one bar a session | Burn-in on the bar grid | `executor.py:140 BURNIN_IN_BAR_SPACE` | 16:00 design grows from 3,469 to 6,462 sessions, starting 1998-06-04; first forecast 2006-06 |

## Gates (`src_gates.csv`)

- **a. All flags off reproduces the old code:**
  - The per-bar 16:00 designs, all_features and baseline, equal the 2026-10-06 rebuild bit for bit. They differ from the
    design of record by 5.403e-11, as before.
  - The pooled panel's 216 affected columns equal the git-HEAD build bit for bit.
  - The 2026-08-08 pooled cache differs from any laptop rebuild, old or new code, by up to 1.7e-10. That is rounding.
- **b. Flags on reproduces the audit's variant designs:** bit for bit (6,462 x 628, plus baseline). The four new causal
  statistics (5b-5e) were unit-checked; together they move 68 columns.
- **c. Live tests:** 79 pass, including 3 new pin tests.
- **d. Lint and types:** ruff check, ruff format and mypy are clean on the changed src files and
  `live/close_signal/forecast.py`.

## Live signal

- **Without a pin, the fixes move the live 16:00 forecast a lot.**
  - Over the last 5 sessions the median move is 5.2 % and the max 7.8 %.
  - The max is 13.6 % over the last 20 sessions and 51 % over the last 250.
- **So live is pinned:**
  - `live/close_signal/forecast.py` has `PIN_PRE_AUDIT_SRC = True`, which sets all 14 new flags False in each run's src
    copy. That is exactly the pre-fix chain (gate a).
  - The VIX splice cannot run in live.
  - The user flips this one switch to move live to the fixed chain.

## Bundle v2 (all 8 on), pre-tuned f arm, scored on actual RV

Record and v2 use the same post-#10 Scorer.

| Model | QLIKE record -> v2 | DM | Sharpe mid | Sharpe crossed | Positions changed |
|---|---|---|---|---|---|
| ridge | 0.0975 -> 0.0969 | -0.62 | 1.42 -> 2.12 | 0.93 -> 1.63 | 35 |
| lasso | 0.0931 -> 0.0933 | +0.27 | 1.88 -> 1.69 | 1.40 -> 1.22 | 23 |
| enet | 0.0958 -> 0.0957 | -0.14 | 1.92 -> 2.01 | 1.43 -> 1.53 | 40 |
| baseline | 0.0967 -> 0.0967 | ~0 | 1.37 -> 1.06 | 0.89 -> 0.58 | 32 |

- **No QLIKE change is significant.** The Sharpe moves come from 23-40 flipped positions, so they are noise-sized.
- **Retune dates also move:** with burn-in in bar space the forecasts start in 2006, so the 250-session retune blocks fall on
  different dates than in the record.
- **Marginals:**
  - B fix: QLIKE -0.18 to -0.33 %, not significant.
  - FOMC calendar + VIX splice: ridge QLIKE -0.23 %.
  - Invalid-print nulling + name sign (fix 1): about 0.

## Re-run cascade (not launched; costs)

1. Per-bar designs v2 for the other buckets: 1-2 minutes each.
2. Per-bar linear models, all arms: about 1 hour per bucket at 4 workers.
3. MinGRU per-bar walks: GPU, minutes. Parked in the paper (user, 2026-10-15).
4. Per-bar LSTM specs: they call `_backtest_and_save` directly and miss fixes 6 and 8, so they need a small code change
   first.
5. Per-bar trees: about 1,000 CARC core-hours, after an untuned probe of about 100.
6. Pooled panel v2 under a new name: 1-3 hours on a cluster node. The column set changes: voldemand `_active` columns
   out, StockTwits `_active` columns in.
7. The paper's eight on pooled v2, blk2 first.
8. Pooled twins: about 77 CPU-hours. Pooled NN: parked in the paper.
9. Pooled forest: about 750 core-hours for the 16:00 interim; about 140,000 for the full table.
10. Master table (#10 scoring), then the paper generators.
