# 08. One straddle, held from an early stamp to 15:30, then sign(s)

Read-only study. Script `writeup/intraday_proposals/08_hold_to_close.py`; tables and
figure in `results/atm_straddle_intraday/proposals/08/`
(`08_results_grid.csv`, `08_diagnostics_1530.csv`, `08_drift_decomposition.csv`,
`08_placebo_random_sign.csv`, `08_bootstrap_vs_close.csv`,
`08_unconstructible_days.csv`, `08_daily_pnl_points.csv`, `08_cum_points.png`).
Nothing is wired into any notebook. Every number below is printed by the script.

> **Provenance.** The deck's per-day file `daily_blk2.parquet` was rebuilt by the
> RV–IV notebook while this study was running, several times, and the script
> therefore fingerprints the version it scores. The numbers below are scored on
> forecast fingerprint **`a5c65541dcf0`** (first written 2026-09-05 01:24, rewritten
> unchanged at 01:26 and 01:34), on which
> the deck's rule table reads sign(s) Sharpe **1.3383** ($t$ 2.4810), always short
> 0.2038 ($t$ 0.3778), 39.95% of days long. The previous version of that file gave
> sign(s) 1.4179 ($t$ 2.6285) and 40.07% long. The grid was computed on both. Every
> verdict is the same; the sign-dependent cells move by at most 0.14 in Sharpe (at
> 15:00, FLIP 2.06 → 1.94 at the midpoint and 1.24 → 1.12 crossed, against the close
> trade's 1.42 → 1.34 and 0.95 → 0.87). Comparator (a), the diagnostics and the
> drift decomposition do not use the signal and are identical on both. The script
> fingerprints whichever version it reads, so any re-run is traceable. The intraday
> notebook's own `rule_table_intraday_crossed_blk2.csv`, quoted once below for
> context, was refreshed in the same wave; the script re-reads it at run time and
> the values in the next paragraph are that refreshed table's.

**Caveat, up front.** A held straddle is not the sum of re-picked one-bar shorts.
The held strikes drift away from the index while a re-picked package keeps its
premium fresh. The two paths are different portfolios over the same bars, and the
midpoint P&L of one is not the midpoint P&L of the other. Section 4 measures the
gap rather than assuming it away.

## Idea

The twelve-bar re-pick loses at the crossed spread because it crosses 16 to 17
times a day (`rule_table_intraday_crossed_blk2.csv`, re-read by the script at run
time: always short 16.41 crossings, sign(s) 17.03, hybrid 16.62; the hybrid falls
from 3.17 at the midpoint to −0.97 crossed, sign(s) from 1.72 to −2.24, always
short from 1.90 to −1.92). This proposal replaces the twelve one-bar shorts with
one short. Sell the nearest out-of-the-money straddle once, early. Hold those
strikes. At 15:30 apply the deck's own signal, and only if it says long is there
anything to do. The day then crosses the spread once, twice or three times instead
of seventeen.

## Exact construction

- **Days.** The deck's 866 scored days, 2020-01-03 to 2024-04-30, joined on the
  index of `results/atm_straddle_0dte_1530/daily_blk2.parquet` (block-diagonal
  ridge). The twelve half sessions are dropped by the shared rule
  (`asl.drop_early_close`, `hours_to_expiration <= 0` at the 15:30 stamp).
- **Frame.** Rebuilt from `data/spxw_chain.parquet` with the library helpers:
  `asl.quote_mid` for the no-quote sentinel, `asl.stamp_spot`,
  `asl.pick_nearest_otm_guarded` on `("expiration", "timestamp")`,
  `asl.attach_iv_hourly_as_30min`, `asl.settle_package` against the official
  `^GSPC` close. 10,387 packages on the deck's 866 days, twelve stamps
  10:00–15:30. The 16:00 stamp is excluded outright.
- **Entry.** Stamp $E \in \{10{:}00, 11{:}00, 12{:}00, 13{:}00, 14{:}00, 15{:}00\}$.
  10:00 is the first stamp with a vendor underlying price and live mids; 9:30 is
  unusable. Short the nearest out-of-the-money call and put at $E$ at the midpoint
  (base case) or at the bid (crossed case).
- **Hold.** Keep those strikes to 15:30 and mark them at the 15:30 quotes of the
  same strikes.
- **Decision at 15:30.** Read the deck's signal $s$ for that day. It is not
  recomputed here: it is the 16:00-row forecast issued at 15:30 against the 15:30
  at-the-money implied variance.
  - $s \le 0$ (520 of the 866 days): **nothing happens at 15:30.** The position
    opened at $E$ is carried to cash settlement at the official close on its own
    strikes, whatever they have drifted to. No exit, no re-pick, no 15:30
    transaction of any kind. The day has exactly one crossing, the entry.
  - $s > 0$ (346 days), **FLAT**: buy the held strikes back at 15:30 (midpoint, or
    the ask) and stay flat. Two crossings.
  - $s > 0$, **FLIP**: buy the held strikes back and buy the 15:30 nearest
    out-of-the-money straddle (midpoint, or the ask), settled at the close. Three
    crossings.

  Only a long signal triggers a 15:30 transaction. The two variants differ only in
  what they do on the 346 long days; on the 520 short days they are the same
  series, and both are identical to comparator (a).
- **Comparators.** (a) short at $E$ and hold to settlement unconditionally — that
  is exactly the construction above with the signal ignored, one crossing; (b)
  short at $E$ and close at 15:30 unconditionally, two crossings; (c) the deck's
  close-only sign(s) — `pos` against `R` in `daily_blk2` — one crossing; (d) the
  deck's always short at 15:30, one crossing.
- **Units.** Per unit of entry premium (the $E$ midpoint in the midpoint case, the
  $E$ bid in the crossed case, which is the convention of
  `asl.crossed_premium_return`), and in index points per straddle. Statistics:
  mean, $t = \sqrt{n}\,\bar r / s_r$, annualized Sharpe $\times\sqrt{252}$,
  maximum drawdown of the running points total, worst day, crossings a day, and
  the break-even half-spread — the uniform proportional half-spread per crossing
  that zeroes the midpoint mean.
- **Crossed spread.** Every crossing pays the touch: sell at the bid, buy at the
  ask. Cash settlement pays no exit spread.

## Gates

- **Reproduction.** The independently picked 15:30 package equals the deck's on
  all 866 days: $K_c$, $K_p$ and $S$ to 0, the midpoint entry to 9.5e-07 (the
  chain's float32 quotes), the implied variance to 0. The deck's rule table is
  reproduced exactly: sign(s) Sharpe 1.3383 ($t$ 2.4810), always short 0.2038
  ($t$ 0.3778), 39.95% of days long.
- **Provenance of the signal.** The deck's `signal` equals `rv_hat - iv_var` to 0,
  and `hours_to_expiration` on that row is 0.5 — the signal is a 15:30 object.
  `pos <= 0` equals `signal <= 0` on every day, so the short branch is the deck's
  own sign with no re-derivation: 520 short days, 346 long days.
- **The short branch books nothing at 15:30.** On the 520 short days FLAT, FLIP
  and comparator (a) are the same series to 0.000 points at both fills, and the
  day carries one crossing.
- **Causality.** The latest stamp anywhere in the frame is 15:30, so the 16:00
  stamp's censored implied volatility is never read. Every decision input (entry
  price, held strikes, 15:30 marks, the deck's position) is built in
  `entry_context`, which reads no stamp after 15:30. The only field dated after
  15:30 is the official close, and it enters only through `asl.settle_package`.
  Permuting that close across days: FLAT days that end flat at 15:30 change on 0
  of 2,075; days that run to settlement change on 3,118 of 3,118. The cached
  official close equals the deck's own `S_close` on every day (max difference 0),
  and settling the 15:30 strikes at it reproduces the deck's `exit` to 0.
- **Cost identity.** Crossed points equal midpoint points minus the half-spreads
  paid, to 9.5e-07 over all 26 constructions. Comparator (c) crossed reproduces
  `asl.crossed_premium_return` to 4.4e-16, with 0 untradeable rows.
- **Algebraic identity.** On the 39.95% of days the signal is long, FLIP is
  exactly comparator (b) plus comparator (c), to 0 points. FLIP is therefore the
  30-minute short plus the deck's close trade on long days, and the stale short
  run to settlement on short days.
- **Placebo, bootstrap.** Section 5 and section 6.

## Results

The whole grid: six entry stamps, four constructions each (FLAT, FLIP, and
comparators (a) and (b) at that stamp), plus the two close comparators. `mean`,
`t` and the two Sharpes are per unit of entry premium; `pts` columns are index
points per straddle; `be` is the break-even half-spread as a percentage of the
premium crossed, `real` the half-spread the quotes actually charge on the same
crossings.

| construction | n | mean | t | Sharpe mid | Sharpe crossed | pts/day mid | pts/day crossed | maxDD pts | worst pts | crossings/day | be % | real % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10:00 hold+flat | 864 | −0.0110 | −0.36 | −0.19 | −0.78 | −0.019 | −0.772 | −632 | −102.3 | 1.40 | −0.06 | 2.45 |
| 10:00 hold+flip | 864 | −0.0001 | −0.00 | −0.00 | −0.64 | 0.094 | −0.746 | −631 | −102.3 | 1.80 | 0.29 | 2.55 |
| 10:00 (a) hold uncond | 864 | −0.0161 | −0.50 | −0.27 | −0.66 | −0.079 | −0.636 | −821 | −102.3 | 1.00 | −0.34 | 2.44 |
| 10:00 (b) close uncond | 864 | 0.0097 | 0.36 | 0.19 | −0.72 | 0.456 | −0.599 | −324 | −104.3 | 2.00 | 1.01 | 2.33 |
| 11:00 hold+flat | 865 | −0.0077 | −0.24 | −0.13 | −0.56 | −0.040 | −0.500 | −463 | −109.3 | 1.40 | −0.15 | 1.73 |
| 11:00 hold+flip | 865 | 0.0058 | 0.17 | 0.09 | −0.40 | 0.064 | −0.485 | −444 | −109.3 | 1.80 | 0.22 | 1.90 |
| 11:00 (a) hold uncond | 865 | −0.0117 | −0.33 | −0.18 | −0.42 | 0.017 | −0.261 | −588 | −109.3 | 1.00 | 0.09 | 1.43 |
| 11:00 (b) close uncond | 865 | 0.0036 | 0.13 | 0.07 | −0.69 | 0.157 | −0.574 | −230 | −89.9 | 2.00 | 0.41 | 1.89 |
| 12:00 hold+flat | 866 | −0.0005 | −0.01 | −0.01 | −0.44 | 0.081 | −0.326 | −394 | −111.6 | 1.40 | 0.35 | 1.75 |
| 12:00 hold+flip | 866 | 0.0124 | 0.37 | 0.20 | −0.30 | 0.184 | −0.311 | −398 | −111.6 | 1.80 | 0.72 | 1.95 |
| 12:00 (a) hold uncond | 866 | 0.0050 | 0.14 | 0.08 | −0.17 | 0.217 | −0.034 | −477 | −111.6 | 1.00 | 1.28 | 1.48 |
| 12:00 (b) close uncond | 866 | 0.0099 | 0.37 | 0.20 | −0.61 | 0.278 | −0.394 | −264 | −85.8 | 2.00 | 0.83 | 2.00 |
| 13:00 hold+flat | 866 | −0.0148 | −0.42 | −0.23 | −0.66 | −0.048 | −0.438 | −460 | −96.1 | 1.40 | −0.24 | 1.90 |
| 13:00 hold+flip | 866 | 0.0003 | 0.01 | 0.00 | −0.50 | 0.055 | −0.423 | −478 | −96.1 | 1.80 | 0.24 | 2.11 |
| 13:00 (a) hold uncond | 866 | −0.0272 | −0.70 | −0.38 | −0.63 | −0.113 | −0.357 | −543 | −96.1 | 1.00 | −0.75 | 1.63 |
| 13:00 (b) close uncond | 866 | −0.0110 | −0.38 | −0.21 | −1.02 | 0.072 | −0.555 | −389 | −86.5 | 2.00 | 0.24 | 2.10 |
| 14:00 hold+flat | 866 | 0.0320 | 0.98 | 0.53 | −0.12 | 0.486 | −0.015 | −290 | −77.6 | 1.40 | 2.80 | 2.89 |
| 14:00 hold+flip | 866 | 0.0522 | 1.47 | 0.79 | 0.08 | 0.589 | −0.000 | −297 | −77.6 | 1.80 | 3.01 | 3.01 |
| 14:00 (a) hold uncond | 866 | 0.0021 | 0.06 | 0.03 | −0.36 | 0.130 | −0.244 | −448 | −77.6 | 1.00 | 1.01 | 2.89 |
| 14:00 (b) close uncond | 866 | 0.0306 | 1.30 | 0.70 | −0.55 | 0.547 | −0.161 | −138 | −69.7 | 2.00 | 2.16 | 2.80 |
| **15:00 hold+flat** | 866 | 0.0898 | 3.37 | **1.82** | **1.00** | 0.536 | 0.183 | −225 | −88.1 | 1.40 | 4.32 | 2.85 |
| **15:00 hold+flip** | 866 | 0.1199 | 3.60 | **1.94** | **1.12** | 0.639 | 0.198 | −267 | −88.1 | 1.80 | 4.36 | 3.01 |
| 15:00 (a) hold uncond | 866 | 0.0509 | 1.39 | 0.75 | 0.35 | 0.299 | 0.053 | −339 | −88.1 | 1.00 | 3.18 | 2.61 |
| 15:00 (b) close uncond | 866 | 0.0598 | 4.64 | 2.51 | 0.16 | 0.541 | 0.018 | −53 | −26.3 | 2.00 | 2.96 | 2.87 |
| **(c) close sign(s)** | 866 | 0.0947 | 2.48 | **1.34** | **0.87** | 0.227 | −0.005 | −273 | −78.4 | 1.00 | 3.34 | 3.41 |
| (d) close always short | 866 | 0.0145 | 0.38 | 0.20 | −0.27 | 0.020 | −0.212 | −329 | −78.4 | 1.00 | 0.30 | 3.41 |

Read the grid before reading any cell of it. Twenty-one of the twenty-four entry
constructions are below the close trade at the midpoint, and every construction at
every stamp before 15:00 is below it at both fills. The pattern is otherwise
monotone in $E$ — the later the entry, the better the cell, at every variant, with
one dip at 13:00 in all four variants. That is the same shape reports 04 and 06
found, and it is the shape the diagnostics below explain.

**Crossings.** FLAT crosses 1.40 times a day, FLIP 1.80, comparators (a), (c) and
(d) once, (b) twice. The notebook's re-picked day crosses 16.4 to 17.0 times. The
proposal does what it was meant to do.

**The early entries.** At 10:00 the held short is flat to negative at the midpoint
(FLAT −0.19, FLIP −0.00) and clearly negative crossed (−0.78, −0.64), with a
maximum drawdown of 632 points against the close trade's 273. Its break-even
half-spread is −0.06% and 0.29% of premium against a realized 2.45% and 2.55%, so
it does not clear its own spread even at 1.4 crossings. 11:00, 12:00 and 13:00 are
the same story with smaller numbers. The 14:00 row is the first that is positive
at the midpoint (FLAT 0.53, FLIP 0.79) and roughly flat crossed (−0.12, 0.08),
with a break-even of 2.80% and 3.01% against a realized 2.89% and 3.01% — exactly
at the spread.

**The 15:00 row.** This is the one place the grid beats the close trade. FLIP
scores 1.94 at the midpoint and 1.12 crossed, against the close trade's 1.34 and
0.87; FLAT scores 1.82 and 1.00. The break-even half-spread is 4.32% and 4.36% of
premium against a realized 2.85% and 3.01%, so the trade clears its spread with
roughly a third of the premium to spare — which is why the crossed column stays
positive where the re-pick's does not. In points, FLIP earns 0.639 a day at the
midpoint and 0.198 crossed, against the close trade's 0.227 and −0.005. The
unconditional comparators at the same stamp do not do this: (a) hold uncond is
0.75 mid and 0.35 crossed, (b) close uncond is 2.51 mid — the highest midpoint
Sharpe in the grid — but 0.16 crossed, a 30-minute decay harvest that the spread
eats. So at 15:00 the sign, not the short, is again what pays.

Note that (c) here is the deck's close trade on its own 866 days, not the intraday
notebook's settlement leg (1.32 crossed), which scores the first 63 days flat and
so drops the COVID quarter. The comparison in this file is against the deck.

`08_cum_points.png` plots the cumulative points a day for both variants at every
entry stamp, with comparators (c) and (d), at midpoint fills. The early-entry
lines rise steeply through 2020 and 2021, give it all back in the first half of
2022, and end below where they peaked; the close trade's line is flatter and its
gradient is positive after 2022. That divergence is the drawdown column of the
table, and it is the reason the early stamps are rejected on more than their
Sharpe.

## What the held strikes look like at 15:30

Measured near-the-money strike spacing at 15:30: median 5.0 points, on 862 of the
866 days. Distances are from the index at 15:30 to the centre of the held pair.
This block does not use the signal.

| entry | n | premium at E | held pair at 15:30 | fresh 15:30 | mean distance (pts) | median | 90th pct | mean distance (strikes) | > 1 strike | > 2 strikes | mean abs delta proxy | held pair = 15:30 pair |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10:00 | 864 | 19.83 | 15.23 | 5.39 | 22.95 | 16.30 | 50.90 | 4.56 | 81.9% | 66.4% | 20.58 | 9.1% |
| 11:00 | 865 | 17.05 | 13.55 | 5.39 | 19.54 | 13.90 | 44.99 | 3.89 | 80.1% | 61.8% | 17.16 | 11.0% |
| 12:00 | 866 | 14.83 | 12.43 | 5.39 | 16.60 | 12.42 | 36.09 | 3.31 | 76.3% | 56.6% | 14.26 | 12.1% |
| 13:00 | 866 | 12.65 | 10.66 | 5.39 | 14.49 | 10.07 | 32.68 | 2.88 | 71.5% | 50.2% | 12.18 | 14.7% |
| 14:00 | 866 | 10.58 | 8.83 | 5.39 | 11.29 | 7.85 | 24.50 | 2.25 | 63.4% | 39.4% | 9.05 | 19.5% |
| 15:00 | 866 | 7.70 | 6.69 | 5.39 | 6.41 | 4.45 | 14.73 | 1.27 | 45.2% | 20.1% | 4.33 | 29.8% |

A straddle sold at 10:00 is, by 15:30, a directional position: 82% of the time it
sits more than one strike from the index and 66% of the time more than two, and
the call-minus-put intrinsic — the net delta proxy — averages 20.6 points in
absolute value against a mean of +0.59. It is not an at-the-money straddle any
more, and the 15:30 signal, which is a statement about at-the-money variance
against an at-the-money price, is not a statement about it. At 15:00 the same
package is 1.27 strikes away on average, 45% of days more than one strike, delta
proxy 4.33, and on 29.8% of days the held strikes are still the 15:30 strikes.

This is also what the short branch commits to. On the 520 short days the position
that settles is the one opened at $E$, not an at-the-money straddle: from 10:00
that is a package 23 points off the money on average, with a 90th percentile of
51 points.

Median half-spreads, as a percentage of premium: 1.27% to 2.29% at the entry
stamps (1.58% at 10:00, lowest 1.27% at 11:00, highest 2.29% at 15:00); 2.30% at
10:00 rising to 2.79% at 15:00 on the held pair marked at 15:30; 2.90% on the
fresh 15:30 package.

## Drift: the held short against the re-picked shorts

Both legs of this comparison are shorts at midpoints over the same bars. The held
short runs from $E$ to the 15:30 mark of the same strikes. The re-picked sum adds
the one-bar shorts at every stamp from $E$ to 15:00, each exiting at the next
stamp's mid of its own strikes — the notebook's construction, computed here from
the chain. This block does not use the signal either.

| entry | bars | n | held (pts) | re-picked (pts) | drift (pts) | t | held sd | re-picked sd | held (per prem) | re-picked (per prem) | drift (per prem) | t |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10:00 | 11 | 864 | 0.456 | 1.809 | −1.354 | −2.43 | 17.27 | 8.62 | 0.010 | 0.146 | −0.136 | −4.93 |
| 11:00 | 9 | 865 | 0.157 | 1.520 | −1.363 | −3.02 | 14.74 | 7.85 | 0.004 | 0.138 | −0.134 | −5.50 |
| 12:00 | 7 | 866 | 0.278 | 1.181 | −0.903 | −2.40 | 12.03 | 7.22 | 0.010 | 0.118 | −0.108 | −4.58 |
| 13:00 | 5 | 866 | 0.072 | 0.958 | −0.886 | −2.76 | 10.52 | 6.66 | −0.011 | 0.097 | −0.108 | −4.72 |
| 14:00 | 3 | 866 | 0.547 | 0.894 | −0.346 | −1.54 | 7.89 | 5.76 | 0.031 | 0.086 | −0.056 | −3.40 |
| 15:00 | 1 | 866 | 0.541 | 0.541 | 0.000 | — | 3.51 | 3.51 | 0.060 | 0.060 | 0.000 | — |

The one-bar case is an exact identity — at 15:00 the held short *is* the re-picked
short, drift 0.000 with a maximum absolute difference of 0.000 points — which is
the check that the two paths are built the same way. Everywhere else holding costs
money at midpoints: 1.35 points a day from 10:00, 1.36 from 11:00, falling to 0.35
from 14:00, all with $t$ between −1.5 and −3.0 in points and between −3.4 and −5.5
per premium. The held short also carries twice the dispersion of the re-picked sum
(17.3 against 8.6 points at 10:00), which is the same fact from the other side:
the stale strikes accumulate direction.

So the proposal buys its cost saving with a real loss of midpoint edge. That is
the trade-off the crossed column resolves: at 10:00 through 13:00 the saving is
not enough, and at 15:00 there is nothing to save because there is only one bar.

## Placebo: a random sign at 15:30

2,000 draws, `rng(0)`, replacing the 15:30 sign with a random one. The
equal-probability placebo draws short with probability one half; the rate-matched
one draws short at the rule's own rate (60.05% of days). Percentiles are of the
real construction's Sharpe among the draws.

| construction | fill | real Sharpe | pct (p = 1/2) | pct (rate-matched) | placebo median | placebo 95th |
|---|---|---|---|---|---|---|
| 10:00 hold+flat | mid | −0.19 | 14.8 | 25.1 | −0.05 | 0.19 |
| 10:00 hold+flip | mid | −0.00 | 67.5 | 74.1 | −0.09 | 0.23 |
| 11:00 hold+flat | mid | −0.13 | 34.2 | 40.1 | −0.07 | 0.20 |
| 11:00 hold+flip | mid | 0.09 | 81.7 | 84.9 | −0.10 | 0.27 |
| 12:00 hold+flat | mid | −0.01 | 20.6 | 22.7 | 0.13 | 0.41 |
| 12:00 hold+flip | mid | 0.20 | 73.2 | 72.9 | 0.06 | 0.43 |
| 13:00 hold+flat | mid | −0.23 | 67.4 | 69.9 | −0.30 | −0.02 |
| 13:00 hold+flip | mid | 0.01 | 93.1 | 93.7 | −0.37 | 0.05 |
| 14:00 hold+flat | mid | 0.53 | 86.7 | 92.3 | 0.28 | 0.66 |
| 14:00 hold+flip | mid | 0.79 | 96.3 | 97.9 | 0.19 | 0.72 |
| **15:00 hold+flat** | mid | 1.82 | **96.5** | **99.1** | 1.09 | 1.73 |
| **15:00 hold+flip** | mid | 1.94 | **99.3** | **99.6** | 0.79 | 1.56 |
| 10:00 hold+flat | crossed | −0.78 | 22.4 | 22.2 | −0.68 | −0.45 |
| 10:00 hold+flip | crossed | −0.64 | 76.5 | 72.7 | −0.78 | −0.46 |
| 11:00 hold+flat | crossed | −0.56 | 43.2 | 36.1 | −0.53 | −0.28 |
| 11:00 hold+flip | crossed | −0.40 | 86.9 | 83.7 | −0.64 | −0.27 |
| 12:00 hold+flat | crossed | −0.44 | 30.4 | 22.7 | −0.36 | −0.08 |
| 12:00 hold+flip | crossed | −0.30 | 81.9 | 73.6 | −0.51 | −0.14 |
| 13:00 hold+flat | crossed | −0.66 | 77.1 | 69.4 | −0.79 | −0.51 |
| 13:00 hold+flip | crossed | −0.50 | 96.1 | 93.7 | −0.93 | −0.52 |
| 14:00 hold+flat | crossed | −0.12 | 91.9 | 92.7 | −0.43 | −0.06 |
| 14:00 hold+flip | crossed | 0.08 | 98.1 | 98.1 | −0.60 | −0.08 |
| **15:00 hold+flat** | crossed | 1.00 | **98.0** | **98.7** | 0.27 | 0.85 |
| **15:00 hold+flip** | crossed | 1.12 | **99.5** | **99.5** | −0.06 | 0.73 |

The real sign clears the 95th percentile of random signs under both placebos only
at 14:00 (FLIP) and 15:00 (both variants), at both fills; 13:00 FLIP crossed
clears one of the two (96.1 equal-probability, 93.7 rate-matched) and nothing else
reaches the 95th under either placebo. Below 14:00 the placebo says the sign adds
nothing to a held
stale straddle — several early cells sit below their own placebo median. This is
the same message as the distance table: the 15:30 signal is about the 15:30
at-the-money package, and the further the held pair has drifted from that, the
less the signal has to say about it.

## Bootstrap against comparator (c)

Circular block bootstrap, block 21, B = 2,000, `rng(0)`, on the days the two
constructions share. `pct+` is the share of draws in which the difference is
positive; the two intervals are the percentile interval and the basic interval.

| construction | fill | ΔSharpe | pct+ | percentile 95% | basic 95% |
|---|---|---|---|---|---|
| 10:00 hold+flat | mid | −1.56 | 1.7 | [−2.97, −0.12] | [−2.99, −0.15] |
| 10:00 hold+flip | mid | −1.36 | 1.9 | [−2.64, −0.09] | [−2.64, −0.09] |
| 11:00 hold+flat | mid | −1.47 | 2.4 | [−2.89, −0.02] | [−2.93, −0.06] |
| 11:00 hold+flip | mid | −1.25 | 3.0 | [−2.50, +0.02] | [−2.52, +0.00] |
| 12:00 hold+flat | mid | −1.35 | 3.2 | [−2.69, +0.05] | [−2.75, −0.00] |
| 12:00 hold+flip | mid | −1.14 | 3.3 | [−2.32, +0.08] | [−2.36, +0.03] |
| 13:00 hold+flat | mid | −1.56 | 1.7 | [−2.87, −0.13] | [−3.00, −0.25] |
| 13:00 hold+flip | mid | −1.33 | 1.8 | [−2.50, −0.10] | [−2.57, −0.16] |
| 14:00 hold+flat | mid | −0.81 | 13.1 | [−2.05, +0.59] | [−2.21, +0.42] |
| 14:00 hold+flip | mid | −0.55 | 18.0 | [−1.56, +0.59] | [−1.68, +0.47] |
| **15:00 hold+flat** | mid | **+0.48** | 80.3 | [−0.72, +1.72] | [−0.77, +1.67] |
| **15:00 hold+flip** | mid | **+0.60** | 94.0 | [−0.17, +1.38] | [−0.18, +1.38] |
| 15:00 (a) hold uncond | mid | −0.59 | 25.8 | [−2.15, +1.10] | [−2.27, +0.97] |
| 15:00 (b) close uncond | mid | +1.17 | 91.0 | [−0.53, +3.10] | [−0.77, +2.86] |
| (d) close always short | mid | −1.13 | 5.9 | [−2.41, +0.33] | [−2.60, +0.14] |
| 14:00 hold+flip | crossed | −0.79 | 7.4 | [−1.78, +0.25] | [−1.84, +0.19] |
| **15:00 hold+flat** | crossed | **+0.13** | 62.9 | [−1.04, +1.35] | [−1.08, +1.31] |
| **15:00 hold+flip** | crossed | **+0.25** | 76.7 | [−0.48, +1.00] | [−0.49, +0.99] |
| 15:00 (a) hold uncond | crossed | −0.52 | 28.3 | [−2.08, +1.14] | [−2.18, +1.03] |
| 15:00 (b) close uncond | crossed | −0.71 | 21.3 | [−2.23, +0.95] | [−2.37, +0.81] |
| (d) close always short | crossed | −1.14 | 5.6 | [−2.39, +0.30] | [−2.59, +0.11] |

(The full table, all 50 rows, is `08_bootstrap_vs_close.csv`.)

Every entry from 10:00 to 13:00 is below the close trade at the midpoint with at
most 3.3% of bootstrap draws positive. Both intervals exclude zero at 10:00 and
13:00, for both variants, and for 11:00 FLAT; at 11:00 FLIP and at both 12:00
variants at least one of the two intervals reaches just past zero, the largest
upper end being +0.08. 14:00 is worse
but not resolved. 15:00 is the only stamp above the close trade, and its intervals
include zero at both fills: +0.60 mid with a percentile interval [−0.17, +1.38]
and 94.0% of draws positive, +0.25 crossed with [−0.48, +1.00] and 76.7%
positive. The comparison is also a best-of-24 read: no cell was pre-registered,
and the placebo does not correct for looking at the whole grid.

## Days that cannot be constructed

Three of the 5,196 entry-stamp cells, all from vendor outages already known to the
chain-integrity audit:

| entry | date | reason | live contracts |
|---|---|---|---|
| 10:00 | 2020-05-01 | no live quote at the stamp | 0 |
| 10:00 | 2022-02-22 | no put at or below the spot | 4 |
| 11:00 | 2022-02-22 | no put at or below the spot | 4 |

Over the whole 10:00–15:30 grid on the deck's days the guards refuse five cells:
those three plus 2022-02-22 at 10:30 (no put, 4 live contracts) and at 11:30
(strike gap 296.8 points, 6 live contracts). No held pair is ever missing a 15:30
midpoint: 0 of the 5,193 constructed entry-stamp days. So $E = 10{:}00$ scores 864
days, $E = 11{:}00$ scores 865, and the other four score all 866.

## Verdicts

- **The proposal as stated, for entries before 14:00: reject.** 10:00 through
  13:00 are below the close trade at both fills, with at most 3.3% of bootstrap
  draws positive and placebo percentiles at or below the median. The cost saving
  is real — 1.4 to 1.8 crossings a day against 16 to 17 — but the midpoint edge
  given up is larger: holding costs 0.9 to 1.4 points a day against the re-picked
  sum, and by 15:30 the held pair is four strikes from the money with a 20-point
  net delta proxy, which the 15:30 at-the-money signal has no claim on. On the 520
  short days the trade settles that stale package, which is what the drawdown
  column is measuring.
- **14:00: reject.** FLIP reaches 0.79 mid and 0.08 crossed with a placebo at the
  96th to 98th percentile, but it is still below the close trade and its interval
  excludes nothing.
- **15:00, both variants: needs more.** FLIP scores 1.94 mid and 1.12 crossed
  against the close trade's 1.34 and 0.87, on the same 866 days, with 1.80
  crossings; FLAT scores 1.82 and 1.00 with 1.40. The placebo is at the 96th to
  99.6th percentile at both fills, the break-even half-spread (4.36% of premium)
  is well above the realized one (3.01%), and the Sharpe difference against the
  close trade is positive at both fills. But neither interval excludes zero
  (percentile [−0.17, +1.38] mid, [−0.48, +1.00] crossed), the cell is the best of
  twenty-four, and the algebraic identity says what it is: on long days FLIP is
  exactly comparator (b) plus the deck's close trade, and on short days it is the
  15:00 short run to settlement with stale strikes. In other words the 15:00 row
  is the close trade with a 30-minute head start, not a new trade. Report 06
  already found that the 15:30 entry sits at the 99.8th percentile of random entry
  clocks; this file finds the adjacent stamp, and only the adjacent stamp,
  competitive with it.
- **What the file does settle.** The turnover question raised in 03's open
  questions — whether a hold rule keyed to something other than sign flips keeps
  the midpoint edge — is answered no for a hold keyed to the clock. The held
  package loses midpoint edge at a rate that grows with the length of the hold,
  and the only hold short enough to keep it is the one that barely holds at all.

## Open questions

- On 29.8% of days at $E = 15{:}00$ the held strikes *are* the 15:30 strikes. FLIP
  as specified still pays two spreads there (buy back the held pair, buy the fresh
  pair); a netted flip would cross once. The crossed column for FLIP is therefore
  conservative on those days — by how much is not measured here.
- The 15:00 cell needs a pre-registered forward test, or an era split, before it
  is anything more than the best of twenty-four. Is 2024-05-01 onward available on
  the same footing?
- Would a hold rule keyed to *moneyness* — re-pick when the held pair drifts more
  than one strike — sit between the two paths? The distance table says that
  trigger would fire on 45% of days from 15:00 and 82% from 10:00, so it becomes
  the re-pick again at the early stamps.
- The 15:00 unconditional close (comparator (b), 2.51 at the midpoint, the highest
  in the grid, 0.16 crossed) is a pure 30-minute decay harvest on an at-the-money
  package. Is any part of it executable at a fraction of the spread?
- The deck moved under this study (see the provenance note). Should proposal files
  record the forecast fingerprint of the `daily_*.parquet` they scored, the way
  this one now does?
