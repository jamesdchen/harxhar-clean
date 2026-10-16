# Data fix 2026-10-15: invalid prints nulled at load, sign of a series read from its name

User (2026-10-15): "it shouldn't count as signed though, it's clearly a data problem." Produced by
`experiments/data_fix_2026_10_15.py` (stages audit, designs, diff, linear fix / fix_fomc, score, pooled). Rebuilt
designs, walks and logs are in `_work/` (gitignored). No cache read by a running job was touched; no commits; no cluster
jobs.

## 1. Root cause

- **The print:** `data/vwstock_stats.parquet`, the bar ending 2018-11-12 11:00 ET, `spread_vwstock` = -0.000318.
  - It is the only negative value in 211,639 prints.
  - Neighbours: 10:30 = 0.00225, 11:30 = 0.00145. The 11:00 slot over the whole sample: n 7,886, median 0.00113,
    1st percentile 0.000283.
  - Details in `bad_print_neighbours.csv`.
- **The same bar in other columns:**
  - VW moments spike: `sumret2_vwstock` 3.9e-4 (vs about 2e-5 either side), `sumret4` 3.1e-5 (vs about 1e-9),
    `sumret3` -1.07e-4.
  - The EW side moves too: `sumret2_ewstock` 1.7e-4; `spread_ewstock` dips 0.0030 -> 0.0017.
  - ES RV is ordinary.
  - Reading: one crossed quote. It fits a -29 % one-minute return in one stock (about 0.44 % VW weight).
  - The positive moment prints are kept by the domain rule: flagged, not nulled. Nulling the whole 11:00 bar in both
    stock files is a user decision, item (d) below.
- **The code** (`src/features/transforms/target.py::robust_transform`, before the fix):
  - `has_negatives = bool((series.dropna() < 0).any())` over the whole forward-filled 1998-2024 panel.
  - A 2018 print sets the treatment of every bar back to 1998, so it is also a look-ahead.
  - With True, the series is divided by the same-slot rolling s.d. and mapped by the identity. The right treatment is
    the same-slot mean and then log.
  - The dictionary records the series as "x and s.d." while its EW twin is "log x and mean".

## 2. Audit (66 series)

Files: `audit.csv`, `audit_checks.csv`, `audit_fomc.csv`, `audit_design_record_fomc.csv`.

### Invalid prints, now nulled at load (21,939)

| series | rule | nulled | where | why invalid | old effect |
|---|---|---:|---|---|---|
| spread_vwstock | > 0 | 1 | 2018-11-12 11:00 | a VW quoted spread below 0 needs crossed quotes | whole series treated as signed |
| sumvolume | > 0 | 21,852 | 1998-01-05 .. 2003-06-30, mostly overnight / pre-open | each bar has 1-30 one-minute returns (98.7 % with a price change); volume was unrecorded and coded 0 | log(1e-12) = -27.6 on those bars |
| effspread_ewstock / _vwstock | > 0 | 43 each | 1998-2012, mostly 08:30 / 09:00 | exactly 0 across the cross-section while quoted spreads are 0.4-0.8 %; trades stamped at the mid with no prevailing quote | a fabricated zero-cost bar |

### Other checks

- **Impossible values: zero cases** for each of:
  - buy or sell above total, buy + sell above total;
  - OFI outside [-1, 1];
  - numobs outside 1..30 or not an integer;
  - sentiment outside [-1, 1];
  - FOMC flags outside {0, 1};
  - inverse distances outside (0, 1].
- **Zero total:** buy + sell = 0 on 26 bars, which are exactly the zero-effective-spread bars. OFI is already NaN there.
- **Sign classification:** every series the old rule called signed has thousands of negative prints, except
  spread_vwstock (1). The next smallest are vix_slope_3m (3,652) and stocktwits_sentiment (4,261). After the nulling,
  the name rule and the data rule agree on all 66 series.
- **FOMC 0/1 flags (a bug, contained):**
  - Rule 6 applies `log(clip(x, 1e-12))`, so 0 -> -27.631 and 1 -> 0.
  - The rolling 1/99 winsoriser clips all 206 statement bars of fomc_release to -27.631. The six
    `adj_fomc_release_ma_*` columns of the 16:00 design are therefore **constant**.
  - `adj_fomc_day_ma_1` is 27.63 x the flag (the IQR guard falls back to 1.0). It duplicates `fomc_day_active` at
    27.6x scale, so penalized models penalize it about 27.6x (lasso) or 760x (ridge) less.
  - Trees are unaffected.
  - Not fixed here; the proposal is identity and no winsorising for 0/1 flags (item c).
- **FOMC dead tail (a second bug):**
  - After the last statement (2023-11-01 14:30) the feed is NaN, and the raw column is forward-filled without limit.
  - The hurdle reads the filled column, so `fomc_{release,day}_active_ma_*` = 1 on all 124 forecast sessions from
    2023-11-01 to 2024-04-30, against 0 on about 97 % of training sessions.
  - Measured as variant `fix_fomc` (indicators set to their mode, 0, on unobserved rows).
  - voldemand's tail is unchanged (its mode is 1).

## 3. The fix

- **`src/data/loading.py`:**
  - A domain registry, l.183-312: `STRICTLY_POSITIVE_SERIES`, `NON_NEGATIVE_SERIES`, `SIGNED_SERIES`.
  - `NULL_INVALID_PRINTS = True`; off reproduces the old panel bit for bit.
  - `null_invalid_prints()` sets prints <= 0 (or < 0 for the non-negative set) to NaN. RV is exempt.
  - It writes one log line: `nulled 21939 invalid print(s): sumvolume 21852, effspread_ewstock 43, effspread_vwstock
    43, spread_vwstock 1`.
  - It is called at load step 6a, before OFI and FOMC are derived, so the NaNs go through the existing missing-data
    path.
- **`src/features/transforms/target.py`:**
  - `SIGN_BY_NAME = True` with `series_is_signed()`: registry names decide; unregistered names keep the data rule.
  - The old line is parked with a dated note.
  - On current data the name rule changes no output beyond the nulling.
- **Live service:** `live/close_signal` copies `src/` on every run. Its bucket changes only at the 1e-15 level.

## 4. Impact

### 4a. The 16:00 all-features design (`design_diff.csv`, `design_gates.csv`)

- **Gates:** with both flags off, the rebuild equals the 2026-10-06 local rebuild bit for bit, which is within 5.4e-11
  of the design of record. Baseline designs are bit-identical.
- **34 of 628 columns change:**
  - `adj_spread_vwstock_ma_*` (6): all 1,469 forecast rows; max |d| 1.2-21.9; mean |d| 0.46-0.64; corr old/new
    0.30-0.64.
  - spread_vwstock avail (4): 2018-11 to 2019-02 only.
  - effspread (12 adj + 8 avail): max |d| <= 0.175, mean <= 6.5e-4.
  - sumvolume (4): floating-point noise only (<= 3.6e-15).
- `fix_fomc` adds 12 FOMC activity columns (1 -> 0 on 124 rows).

### 4b. The per-bar linear arms of record (`linear_scores.csv`, `linear_gates.csv`, `linear_forecasts.parquet`)

Gates: the 2026-10-06 walk recomputed is bit-identical, and the old f arms reproduce the master table to < 1e-9.

| arm | variant | QLIKE old -> new | DM vs old (p) | Sharpe mid | Sharpe crossed | trade days changed |
|---|---|---|---|---|---|---:|
| subpt_enet_all_features | fix (= fix_fomc) | 0.0942 -> 0.0942 | -1.20 (0.23) | 1.92 -> 1.92 | 1.43 -> 1.43 | 0 |
| subpt_lasso_all_features | fix (= fix_fomc) | 0.0914 -> 0.0914 | +0.04 (0.97) | 1.88 -> 1.88 | 1.40 -> 1.40 | 0 |
| subpt_ridge_all_features | fix | 0.0958 -> 0.0957 | -0.36 (0.72) | 1.42 -> 1.37 | 0.93 -> 0.89 | 8 |
| subpt_ridge_all_features | fix_fomc | 0.0958 -> 0.0958 | -0.04 (0.96) | 1.42 -> 1.34 | 0.93 -> 0.86 | 9 |

- **Arm a** (the one-penalty `sub_*`):
  - ridge 0.1164 -> 0.1149 (DM -1.51);
  - enet 0.0986 -> 0.0983 (DM -1.76);
  - lasso 0.0973 -> 0.0972.
- **Every change is within noise.**

### 4c. The pooled panel (cache not rebuilt; `pooled_diff.csv`, `pooled_gates.csv`)

- **Gate:** the old-flag rebuild of the 96 affected columns equals the cache exactly.
- **Under `fix`, 96 of 1,144 columns change; about 14 change materially in 2020-2024:**
  - `adj_spread_vwstock_ma_*` (12): corr 0.45-0.67 from 2020 on.
  - `adj_sumvolume_ma_*`: heavily changed in 1998-2003. From 2020 on, only ma_2048 (corr 0.987) and ma_1024 change,
    through the masked scaler's expanding IQR floor.
- The pooled forest's 2001-2005 forecasts train on the changed 1998-2003 rows.

## 5. Proposed cascade (nothing launched)

0. **Settle the bundle once:**
   - (a) nulling + name sign (committed in src, uncommitted in git);
   - (b) the FOMC dead-tail activity fix;
   - (c) 0/1 FOMC flags as identity, not winsorised (not measured);
   - (d) whether to null the whole 2018-11-12 11:00 stock-file bar.
1. **Per-bar 16:00 designs v2** under new names: about 4 minutes per design.
2. **Per-bar linear models:** about 20 minutes. Expect no change for enet and lasso; ridge within noise.
3. **MinGRU per-bar walks:** GPU, minutes per design.
4. **Per-bar trees on all features:** about 1,000 CARC core-hours. Probe the untuned shipped configurations first
   (about 100 core-hours).
5. **Pooled panel v2** under a new name: 1-3 hours on a cluster node.
6. **The paper's eight on the pooled panel**, blk2 first: it is the master-table reference and the deck's `daily_blk2`.
7. **Pooled twins:** about 77 Hoffman2 CPU-hours.
8. **Pooled NN:** re-export from v2, then a GPU walk.
9. **The pooled forest (running):**
   - The full table costs about 140,000 core-hours; the 16:00 interim about 750.
   - Recommendation: don't stop the running campaign. Re-run the 16:00 interim on v2, and regenerate the full table
     only if the 16:00 rows move beyond noise.

## Files

- `audit*.csv`, `bad_print_neighbours.csv`
- `design_diff.csv`, `design_gates.csv`
- `linear_scores.csv`, `linear_gates.csv`, `linear_forecasts.parquet`
- `pooled_diff.csv`, `pooled_gates.csv`
- `_work/`
- Script: `experiments/data_fix_2026_10_15.py`
- Source changes: `src/data/loading.py`, `src/features/transforms/target.py`
