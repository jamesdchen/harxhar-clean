# The long-only 15:30 close signal (Google Calendar, free feeds, GitHub Actions)

Every SPXW session at 15:30 ET this job rebuilds the research panel's row for
the day, runs the **same** per-bar forecast arm the research measured, turns
the sign rule into a **break-even straddle price**, sizes the buy by premium,
and posts the instruction as a Calendar event on the operator's phone.  The
operator executes on Robinhood (XSP index options; the SPX equivalent is on
the card).  **The short side of sign(s) is not traded**: the book is the
Heaviside part 1{s > 0} plus the month-end long.

```
12:17 ET   Actions fires (16:17 / 17:17 / 18:17 UTC; the guard keeps the first)
15:00      PREP event (its own calendar key): the day, the budget, the row
           near the index -- what is known half an hour ahead
15:15      PRECOMPUTE: the 15:00 panel (the free ES delay is ~10 min), the
           warm arm server (arms_shared --serve), a canary pass through
           the whole forecast; nothing it computes is reused (below)
15:30:30   fetch the five feeds at once, measure delays, build today's rows
           forecast = specs/causal_tune_linear.py (13 bars, ridge, tw 2000,
           free_vix_only) in FULL on the extended panel, one shared pass
           + the causal MZ map                                     -> rv_hat
           P* = Black-76 straddle price at sqrt(rv_hat)                -> the card
~15:31     Calendar event at 15:30 with a popup; state committed back
```

**Why the forecast itself cannot start before 15:30.**  The 15:30 row moves
full-sample statistics inside the spec's transform -- the diurnal std floor of
the signed moments (a median over every row), the median fills of unobserved
cells -- so appending it changes 28 of the 232 transformed columns over the
whole history (adj_sumret3 on 29,837 rows, the HAR means of adj_sumret to the
last row) and with them every arm's past forecasts: on 2026-09-23 all 27,940
prior-session rows moved, by up to 6.3e-6 relative, and the MZ map is fitted
on those rows.  A 15:15 forecast is a different number.  What moves ahead of
the stamp is everything around the arithmetic: the interpreter and imports,
the spawned workers, the numba kernels, the code copy, the vendor files.

**The shared pass** (`arms_shared.py`).  With `LAG_SCOPE=global` the 13 arms
load, transform and HAR the SAME panel; the path of record does it 13 times
(plus the spec's incumbent OLS run and metrics table, which the card never
reads).  The shared pass runs the spec's own `run_executor` once, over the 13
bar segments, with the spec's own definitions and keywords read from its
source, and hands the 13 slices to the SAME `_backtest_and_save` in 4 spawned
workers; the per-column `robust_transform` calls and the expiry extractor run
in those workers too, each on exactly the inputs it reads.  The gate,
`python -m live.close_signal.fastpath_check --check`, requires the 13 results
CSVs to be bit-identical to the 13-process path's, rv_hat to be equal, and the
card (P\*, strikes, counts, decision) to be equal, on 2026-09-23, 2026-09-24
and the 2026-08-31 month-end.  `run.py --full-arms` still forecasts through
the path of record.

## The rule, in the operator's units

Research rule: buy iff `rv_hat > iv_var`, where `iv_var` is the Black-76
package variance of the nearest-OTM straddle over the last bar (study 73: the
deck's `iv_var` is exactly that re-inversion).  Because the package price is
increasing in volatility, the same statement is

    buy  iff  ask(straddle)  <=  P*  =  package_price( sqrt(rv_hat), S, Kc, Kp )

so the card carries P\* and the operator compares it with the quote.  No option
feed is needed.  XSP = SPX / 10; each break-even is computed on its own listed
nearest-OTM strikes (XSP's 1-point grid is 10 SPX points, coarser than SPX's
5), so `P*_xsp` is close to, not exactly, `P*_spx / 10`.

The card (one example):

```
BUY ONLY IF the XSP 654C / 653P straddle ASK <= 0.62
  then buy N = floor($2,310 / (ask x 100)) straddles (37 at the break-even),
  limit at the ask, chase up to +5%, hold to cash settlement.
  SPX equivalent: 6535C / 6530P, ask <= 6.21, N = 3.
  If the ask is above the break-even: NO TRADE (the short side is not traded).
Behind it: spot 6,532.40; forecast last-bar vol 0.250% (rv_hat 6.25e-06); ...
```

Month-end closes (proposal 55): **buy regardless of the forecast**, limit at
the ask, chase +5 %.  Third Fridays: sign(s) as usual (study 76 found the
third-Friday buys losing on 11 days -- too few to act on).

## Sizing (study 76, Kelly of the long leg; premium units)

| leg | fraction of capital as premium | note |
|---|---|---|
| general long leg (days with s > 0) | **0.033** (half-Kelly; f\* = 0.066 rests on five days) | `--long-fraction` |
| month-end long | **0.15** (the operator's choice after study 77: half-Kelly on the observed mean is 0.22, full Kelly one SE below it; 0.15 keeps P(75 % drawdown) ≤ 0.07 in every version of the mean at +80 %/yr); **0.22** is the ceiling, earned when the live ledger's month-end mean holds | `--month-end-fraction` |

`N = floor(capital x fraction / (ask x 100))` -- the card prints the count at
P\*; the operator recomputes at the actual ask.  Capital comes from the
`CLOSE_SIGNAL_CAPITAL` secret.

## Input modes (`--input-mode`)

Yahoo's published delays (read 2026-09-23): **Cboe indices 15 minutes, CME
futures 10 minutes, the S&P 500 index real-time.**  At 15:30:30 the free feeds
therefore do NOT carry the 15:00-15:30 ES bar or the 15:30 Cboe prints.  The
run measures every symbol's delay (`state/latency.parquet`) and never uses a
stale bar as the 15:30 stamp without saying so.

| mode | 15:00-15:30 ES moments | Cboe prints at 15:30 | status |
|---|---|---|---|
| `ibkr` | real-time ES bars through live/ibkr's broker | real-time | the research construction; the reference mode. **Seam not wired in this package** (`run.py` raises until it is). |
| `free_substitute` (default) | the real-time `^GSPC` 1-minute bar's return moments stand in for ES; `sumvolume`/`numobs` NaN for that stamp | the latest available print (~15:15), lag journaled | **A MODEL CHANGE.** The research features are ES-based and the VIX print is the 15:30 one; substituting the index bar and a 15-minute-old VIX must be VALIDATED on the purchased history before it is trusted: a degradation study (SPX-based last-bar moments + 15-minute-lagged Cboe prints vs the research features, 2024-04..2026-09, same arm, same scoring). Not done. |
| `free_delayed` | wait for the ES bar (~15:40) | wait for the prints (~15:45) | the research inputs, 15 minutes late; the card is marked **LATE** -- part of the bar is gone and the settlement is what it is. Deadline `--delayed-deadline 15:50:00`. |

## Setup

1. **Google Calendar.** Create an OAuth client (Desktop app) with the
   Calendar API enabled; run once locally
   `python -m live.close_signal.auth_once --client-id ... --client-secret ...`;
   paste the printed token into the repository secrets `GCAL_REFRESH_TOKEN`,
   plus `GCAL_CLIENT_ID`, `GCAL_CLIENT_SECRET`, `GCAL_CALENDAR_ID` (`primary`
   for the main calendar) and `CLOSE_SIGNAL_CAPITAL` (dollars).
2. **History once.**
   * The Cboe gap (2024-02-13 .. today) is **already ingested** from the free
     Yahoo snapshots in `data/free_feed/` (`state/cboe_gap_yahoo.parquet`, 45,792
     grid rows, and the same three columns in `state/panel_free.parquet` under
     source `yahoo_cboe_hourly`).  To redo it:
     `python -m live.close_signal.ingest cboe-yahoo`; the overlap gate:
     `python -m live.close_signal.ingest gate-yahoo`.  No FirstRate purchase.
   * The ES gap needs the one purchased file: Databento `GLBX.MDP3` `ohlcv-1m`,
     symbol `ES.v.0` (volume-ranked continuous, `stype_in="continuous"`),
     2024-04-01 .. today (`end` exclusive), one CSV (UTC ns bar-START
     `ts_event`, decimal or fixed-point prices) -- cents inside the signup
     credit (`client.metadata.get_cost` first):
     `python -m live.close_signal.ingest es-databento <es.csv>`
   Everything shifts to the panel's bar-END naive ET before joining.  The roll
   is read off `instrument_id`; the moments are built per contract segment and
   summed by stamp, so only the one cross-contract minute is lost.  The ingest
   prints the seam report (log ratios vs the free feed's rows on the stamps
   both carry) before the purchased rows overwrite them.  The portal's
   request builder has no continuous symbology -- it sells the whole `ES`
   product (every contract, ~79 MB / $5 for the span): such a file is
   accepted too, `ingest.select_front_by_volume` applying Databento's own `.v`
   rule (the outright with the highest volume on the previous trading day;
   spreads excluded) before the same per-contract build.
   `ingest_cboe` (FirstRate 1-minute CSVs) stays as an optional cross-check.
   * **Done 2026-09-23** through `pull_databento_es.py` (streaming API; the
     batch queue never moved): 29,338 rows 2024-03-31 .. 2026-09-23 in the
     store.  Vendor gate on 2024-04: overnight stamps identical to machine
     precision on half the month, RTH sumret2 +1.2 % / sumvolume -7 % level
     shifts (written down in the audit ledger, absorbed by the rolling scaler);
     `features.SESSION_BREAK` (no return across the maintenance hour or the
     weekend) came out of that gate.
   * **Row rule** (`forecast._extend`): a panel row exists iff the bar had ES
     prints -- the vendor's own convention (no Saturday, Sunday from 18:30,
     holiday sessions end with the prints, never the spring-forward 02:00).
     The store holds the Cboe carry on every calendar stamp; only stamps with
     a realized `sumret2` become rows.  A vendor VIX cell that is NaN at a
     stamp the store has a print for is filled (2024-02-13 .. 04-30), a finite
     vendor cell never overwritten.  Before the arms run, the ES history from a
     month before the vendor's last day to the session must have no gap wider
     than 5 calendar days (the vendor's widest closure since 2010 is 4) --
     otherwise a NO SIGNAL card names the hole.
3. **FOMC dates**: fill `state/fomc_statement_dates.csv` from the Fed's calendar
   (the vendor's releases feed ends 2023-11-01).
4. **Commit the state directory** and enable the workflow.

## History and refits -- the fidelity rule

The operator's rule (2026-09-23): **the highest-fidelity computation, so the
regression never has to be re-validated against the true data.**  Concretely:

* **The daily forecast runs the FULL research arms -- all 13 of them.**
  `forecast.run_arms` calls `specs/causal_tune_linear.py` once per regular-hours
  bar (`bar1000 .. bar1600`) on the whole extended panel -- START 0, END -1,
  no halo, the same `ridge / tw 2000 / free_vix_only` settings the CARC
  campaign measured -- so every rolling object (the robust scaler, the
  availability masks, the impute medians, the diurnal baseline, the 21-session
  refits) sees exactly the rows it saw on the cluster.  There is no chunking,
  no incremental update and no frozen coefficient: today's rv_hat is the
  research forecast on a longer panel.  One spec process per bar is the path
  of record (`--full-arms`, about 5 minutes); the card uses the shared pass
  above, the same 13 results files from one load and transform.
* **The table and the loader are the research's.**  The 13 arms' rows are
  stacked into the notebook's yhat table exactly as
  `experiments/build_subsection_yhat.py` builds the research tables (`t` = the
  ET stamp in UTC at microsecond resolution, `yhat` = pred_adj, `baseline` =
  true_raw / true_adj^2, `rv_raw` = the PANEL's realized variance at the stamp,
  never the arm's winsorized `true_raw`), and today's 16:00 row goes through
  the deck's own loader `asl.load_yhat_1530_mz_cached` -- the causal
  second-order MZ map fitted on the session's regular-hours rows -- to
  `rv_hat`.  Why every bar when only the close is traded: the recalibration is
  fitted on the whole session.  Measured 2026-09-23: on the 16:00 rows alone
  the map is 4-5 % off the deck's rv_hat and disagrees with sign(s) on 89 of
  866 days; on the 13-bar table it reproduces the deck exactly.
* **The gate that proves it** -- `python -m live.close_signal.run --gate-deck`
  (`forecast.assert_reproduces_deck`): the 13 arms with the research bucket
  (`live_feasible`) on the vendor files alone must equal the research table
  `results/spxw_pnl/yhat_sub_ridge_live_feasible.parquet` on every one of its
  20,044 rows (yhat to 1e-9 relative, baseline and rv_raw exactly) and the
  loader's rv_hat must equal `daily_sub_live_ridge.parquet` on 866/866 deck
  days with every sign(s) agreeing.  **Result on 2026-09-23** (13 arms, two at
  a time on the operator's Windows machine, 15 minutes): 20,044 of 20,044 rows
  matched; baseline and rv_raw exactly equal; yhat max relative deviation
  5.5e-7 (bar1530; the traded 16:00 bar 7.3e-12; five bars below 1e-9);
  rv_hat on 866/866 deck days, max relative deviation 9.6e-9; **signs 866/866**.
  The yhat deviations are the known cross-machine float-path effect of the
  ridge solves (CARC's BLAS vs the local one, same rows, bit-identical inputs);
  the gate's tolerances sit one decade above the measured numbers (yhat 1e-6,
  rv_hat 1e-7) and the deterministic columns and the signs must be exact.
  `--reuse-arms` re-checks the assembly and the loader without re-running the
  arms.
* **History once, then daily appends.**  The vendor panel ends 2024-04-30 (its
  Cboe feed 2024-02-12).  The gap is filled once: ES 1-minute bars from
  Databento (the one purchased file, inside the signup credit) and the Cboe
  prints from the free Yahoo hourly snapshots.  After that the job appends each
  session's rows from the free feeds and the full arm re-runs on them.
* **The Yahoo-hourly Cboe gap, measured.**  On the vendor overlap 2023-12-07 ..
  2024-02-12 the vendor's value at stamp T is the print standing just before T,
  and the Yahoo hourly bar that ENDS at T closes at exactly that print: log
  error MAD 0.0000 on 629 / 270 / 315 stamps (VIX / VVIX / VIX3M; VIX max
  0.0039, VIX3M p95 0.0026).  The other half-hour stamps carry the last print
  of the day: MAD 0.31 / 0.39 / 0.27 %, p95 1.4 / 1.6 / 1.3 %.  Stamps before
  the day's first print and days without hourly bars take the previous
  session's daily close (never the same day's): MAD 1.9 / 1.4 / 0.9 % on the
  few overlap rows that needed it.  The vendor-panel study puts the forecast
  cost of a 30-minute-stale VIX at R^2 0.8211 -> 0.8200.  Provenance per row
  in `state/cboe_gap_yahoo.parquet` (`<col>_source`).
* **FirstRate is not needed.**  VVIX and VIX3M add nothing at the per-bar arm
  (the 2026-09-23 vixonly campaign), and the VIX gap is covered above; the
  FirstRate reader stays only as an optional cross-check.

## Validation before anything is trusted (30 days)

1. **Latency at 15:30** -- `state/latency.parquet` after each run: the measured
   delay per symbol against Yahoo's nominal 10 / 15 / 0 minutes.
2. **Purchased history vs the vendor panel** on April 2024 (the vendor's last
   month): `features.assert_parity(ours, vendor, rel_tol=...)` on the ES
   moments and the Cboe prints.  This is the gate on the DEFINITIONS in
   `features.py` (written from the column names; the vendor's builder is not
   in the repository) and on the print convention.  Pre-register the
   tolerance; a failed gate is a finding.
3. **Yahoo vs purchased** on the 30-day overlap Yahoo's 1-minute history allows:
   the same gate between the two sources.
4. **Placeholder invariance**: `python -m live.close_signal.run --self-test 2024-04-29`
   runs the 13 arms with the true 16:00 target and with the placeholder and
   requires identical `pred_adj` and baseline on every row of the assembled
   table up to and including that session -- the claim the daily design rests
   on (the features are shift(1), the diurnal baseline is shift(1), the MZ map
   uses prior sessions).
4b. **Deck reproduction**: `python -m live.close_signal.run --gate-deck` (above);
   run it after any change to the spec, the panel files or this package's
   assembly.
5. `free_substitute` degradation study (above) before that mode is the default
   in anger; until then `free_delayed` is the honest mode.

## What "NO SIGNAL" means

A `FeedError` (empty or malformed download, a bar that never arrived before the
deadline) or a forecast failure posts an event titled `CLOSE TRADE: NO SIGNAL
(feed failure)` with the error text and exits 2.  No event at all on a session
day means the workflow did not run -- check the Actions log.

## Limits, stated

* Yahoo is unofficial: rate limits, occasional empty days, delayed Cboe and
  CME data.  The measured-delay log is the record; `free_delayed` is the mode
  that reproduces the research inputs.
* Actions cron jitter (5-20 min) is absorbed by firing hours early and
  sleeping to 15:30:30 in the job; a firing later than 15:25 ET is skipped.  A
  firing after 15:00 posts the prep at once, after 15:25 none; closer than two
  minutes to the stamp the precompute is skipped and the card path starts its
  server cold (slower: worker start-up, numba compilation, transforms inline).
* The forecast needs a continuous panel: 65 sessions for the HAR ladder,
  2,000 for the training window.  Until the gap is ingested the arm cannot run.
* `numobs` in the vendor panel is a tick count; from 1-minute bars it is 30, so
  the free bucket drops it.  `sumvolume` IS reproducible (Yahoo's ES=F bars
  carry volume) and it mattered: `free_feasible` (both dropped, 14 cols) cost
  ~2 % QLIKE at the close, `free_feasible_vol` (numobs only, 15 cols) is within
  0.5 % of `live_feasible` with the trade unchanged -- the service forecasts with
  `free_feasible_vol`.  `free_vix_only` (13 cols: also without vvix / vix3m) is
  staged in `cluster/slurm/submit_freevix.sh` and replaces it if it holds.
* XSP strike spacing near the money is taken as 1 point; verify on the chain
  the first live day.

## Files

`feeds.py` Yahoo adapters + delays; `features.py` the 30-minute moments and
prints, `assert_parity`; `ingest.py` Databento / Yahoo-hourly Cboe gap / FirstRate (cross-check); `state.py` the
committed parquets; `forecast.py` the spec arm on the extended panel + MZ;
`arms_shared.py` the 13 arms in one shared pass (+ the warm server);
`fastpath_check.py` its gate against the path of record;
`signal.py` P\*, sizing, the card; `calendar_push.py`, `auth_once.py`;
`schedule.py` the guard and the prep / precompute timing; `run.py` the entry
point; `close_signal.workflow.proposed.yml` the workflow proposed for main;
`tests/`.
