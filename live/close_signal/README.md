# The long-only 15:30 close signal (Google Calendar, free feeds, GitHub Actions)

Every SPXW session at 15:30 ET this job rebuilds the research panel's row for
the day, runs the **same** per-bar forecast arm the research measured, turns
the sign rule into a **break-even straddle price**, sizes the buy by premium,
and posts the instruction as a Calendar event on the operator's phone.  The
operator executes on Robinhood (XSP index options; the SPX equivalent is on
the card).  **The short side of sign(s) is not traded**: the book is the
Heaviside part 1{s > 0} plus the month-end long.

```
15:00 ET   Actions fires (19:00 or 20:00 UTC; the guard keeps the right one)
15:30:30   fetch feeds, measure delays, build today's 48 panel rows
           forecast = specs/causal_tune_linear.py (bar1600, ridge, tw 2000,
           free_feasible) on the extended panel + the causal MZ map  -> rv_hat
           P* = Black-76 straddle price at sqrt(rv_hat)                -> the card
15:31      Calendar event at 15:30 with a popup; state committed back
```

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
2. **History once** (the purchased files; pricing by the parent's pricing agent):
   * Databento `GLBX.MDP3` `ohlcv-1m`, symbols `ES.FUT` (all months) and `ES.v.0`
     (volume-ranked continuous), 2024-04-01 .. today, CSV export (UTC ns
     bar-START `ts_event`):
     `python -c "from live.close_signal.ingest import *; from live.close_signal.state import StateStore; from pathlib import Path; ingest_es(StateStore(Path('live/close_signal/state')), Path('<es.csv>'))"`
   * FirstRate Data 1-minute CSVs (US/Eastern bar-START) for VIX, VVIX, VIX3M
     (SPX optional), 2024-02-01 .. today:
     `... ingest_cboe(store, {'vix': Path('VIX.csv'), 'vvix': Path('VVIX.csv'), 'vix3m': Path('VIX3M.csv')})`
   Both shift to the panel's bar-END naive ET before joining.  Roll days of the
   continuous ES symbol are dropped (the stitched jump is not a return).
3. **FOMC dates**: fill `state/fomc_statement_dates.csv` from the Fed's calendar
   (the vendor's releases feed ends 2023-11-01).
4. **Commit the state directory** and enable the workflow.

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
   runs the arm with the true 16:00 target and with the placeholder and
   requires identical `pred_adj` and baseline -- the claim the daily design
   rests on (the features are shift(1), the diurnal baseline is shift(1), the
   MZ map uses prior sessions).
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
* Actions cron jitter (5-20 min) is absorbed by firing at 15:00 and sleeping
  to 15:30:30 in the job; a firing later than 15:25 ET is skipped.
* The forecast needs a continuous panel: 65 sessions for the HAR ladder,
  2,000 for the training window.  Until the gap is ingested the arm cannot run.
* `numobs` in the vendor panel is a tick count; from 1-minute bars it is 30.
  The free bucket therefore drops the two liquidity columns (`free_feasible`,
  14 columns; CARC campaign staged in `cluster/slurm/submit_free.sh`).
* XSP strike spacing near the money is taken as 1 point; verify on the chain
  the first live day.

## Files

`feeds.py` Yahoo adapters + delays; `features.py` the 30-minute moments and
prints, `assert_parity`; `ingest.py` Databento / FirstRate; `state.py` the
committed parquets; `forecast.py` the spec arm on the extended panel + MZ;
`signal.py` P\*, sizing, the card; `calendar_push.py`, `auth_once.py`;
`schedule.py` the guard; `run.py` the entry point; `tests/`.
