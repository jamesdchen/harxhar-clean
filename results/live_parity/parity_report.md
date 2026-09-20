# Live engine vs backtest: parity report

Trade cache: `trade_77439812_1786671584951439500_d4dd8708f6_9d76153205.parquet` (of 1 in the cache directory, newest by modification time)  
Days replayed at 11:00: 865  
Every reproduction below is ASSERTED at PARITY_TOL = 1e-09 (published Sharpes at 5e-08, half a unit in their last printed place) and raises `ParityError` on a miss; a missing input raises `FileNotFoundError` naming it, rather than skipping.  

Book of record: short one nearest-OTM 0DTE straddle at 11:00,
delta-hedged every 30 minutes, bought back at 15:30 with the futures
leg flattened at the same stamp (the EXIT book). The HOLD book is the
same entry and hedge carried through cash settlement.

## 1. Vendor-implied path (must be the same formula on the same inputs)

| book | max abs diff vs research | Sharpe (engine) | Sharpe (research) |
| --- | --- | --- | --- |
| exit (buy back at 15:30) | 2.079e-13 | 4.695674 | 4.695674 |
| hold (cash settlement) | 1.110e-15 | 4.200674 | 4.200674 |

Bars whose vendor implied volatility sat on a solver bracket node and was re-inverted from the straddle midpoint by the live engine: 13.
Entry premium rebuilt from the leg quotes, largest disagreement with the cached premium: 0.000e+00 points.

## 2. Mid-inverted path (what the live engine must do at every stamp)

| book | max abs diff | mean abs diff | Sharpe (mid-inverted) | Sharpe (vendor) |
| --- | --- | --- | --- | --- |
| exit | 0.014758 | 0.000333 | 4.700271 | 4.695674 |
| hold | 0.007136 | 0.000210 | 4.199967 | 4.200674 |

This is a finding, not a defect: the broker publishes no vendor implied
volatility, so the live delta is taken from a volatility bisected out of
the straddle's own midpoint.

## 3. Whole-contract futures hedge (ES, multiplier 50)

| straddles | exit Sharpe | exit max abs diff | hold Sharpe | hold max abs diff |
| --- | --- | --- | --- | --- |
| 1 | 4.359622 | 0.680629 | 3.945987 | 0.644696 |
| 3 | 4.679924 | 0.278432 | 4.156349 | 0.325835 |
| 5 | 4.625119 | 0.216697 | 4.121004 | 0.212560 |

Continuous hedge for comparison: exit 4.695674, hold 4.200674.

## 4. What the 15:30 buy-back actually costs

| buy-back | Sharpe (whole sample) | Sharpe (daily era) |
| --- | --- | --- |
| Black-76 model mark | 4.6957 | 4.2272 |
| quoted midpoint | 4.2661 | 3.8787 |
| quoted ask, entry at the bid | 2.2525 | 2.5131 |

Daily era is 2022-05-16 onward (489 days).
Median model mark minus quoted midpoint: -0.0913 points whole sample, -0.1181 points in the daily era.

## 5. The whole tape, and the strike selection

Sessions replayed: 1279 (2020-01-03 .. 2025-12-31, 12 half sessions dropped: the 15:30 row has already expired on them).  
Per-clock tapes built on 11 worker processes.  
Strike-selection parity, at the CHAIN level rather than on the two legs the backtest already chose: the nearest-OTM straddle is re-picked from the whole chain at every stamp with the research's guards, and its 11:00 strikes agree with the trade cache on all 1278 common sessions (0 disagreements, 4 stamps refused by the guards).  
Entries whose package had a leg with no bid: 0.

## 6. The 11:00 book, mid-inverted (proposal 43's GATE 0)

| sample | n | engine Sharpe | research | difference |
| --- | --- | --- | --- | --- |
| deck in-sample | 865 | 2.2530773 | 2.2530773 | 2.46e-08 |
| daily era | 489 | 2.5133605 | 2.5133605 | 4.14e-08 |

43's own GATE 0 and GATE 1 run inside this harness on the same tape, so the day-by-day reproduction of proposal 41's chain build and the 413 unseen sessions' mean and HAC t are asserted too.

## 7. Fixed entry clocks, held to settlement, on the 413 unseen sessions

| clock | unit | n | mean | Sharpe | HAC t | MaxDD | worst | vs 43 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13:30 hold | index points per contract | 413 | +1.173260 | 2.059741 | +3.210432 | -92.6627 | -39.4889 | 0.0e+00 |
| 13:30 hold | premium | 413 | +0.087721 | 2.515839 | +3.557952 | -3.9602 | -3.2529 | 4.4e-16 |
| 11:00 hold | index points per contract | 413 | +0.528453 | 0.773241 | +0.968293 | -215.6884 | -62.8409 | 0.0e+00 |
| 11:00 hold | premium | 413 | +0.049354 | 1.745826 | +2.107446 | -6.3193 | -2.8723 | 5.6e-17 |

The afternoon entry is the redesign's target: 13:30 held to the settlement earns +1.173260 index points a day per contract on the unseen sessions with a HAC t of +3.210432, where fixed 11:00 held is inside noise. Every cell is asserted against `proposals/43/a_clock_year.csv`.

## 8. The causal selector book (proposal 46, estimator E2, rolling 252)

Entry clock chosen each session by the trailing ratio of sums of implied over realized remaining-window variance per clock, relative to that clock's own expanding level, on sessions strictly earlier; held to the settlement. The pick sequence agrees with proposal 46 on every one of the 1279 sessions (63 warm-up sessions flat), and the daily series is pinned through 46's complete statistic set.

| sample | unit | n | mean | Sharpe | HAC t | MaxDD | vs 46 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| deck in-sample | pts | 866 | +0.779967 | 1.733164 | +3.136383 | -138.9267 | 0.0e+00 |
| deck in-sample | units | 866 | +0.063432 | 1.417360 | +2.727548 | -15.3269 | 8.3e-17 |
| unseen | pts | 413 | +0.838258 | 1.447708 | +1.867073 | -124.3912 | 7.1e-15 |
| unseen | units | 413 | +0.082061 | 2.038211 | +2.555090 | -8.5188 | 4.4e-16 |
| all | pts | 1279 | +0.798789 | 1.613267 | +3.618560 | -138.9267 | 7.1e-15 |
| all | units | 1279 | +0.069447 | 1.602326 | +3.676588 | -15.3269 | 2.2e-16 |

Clocks picked: 10:00 200, 10:30 1, 11:30 1, 13:00 108, 13:30 32, 14:00 3, 14:30 524, 15:00 347, flat 63.

Read it as proposal 46 does: the selector tracks the morning-to-afternoon migration about a quarter late, and it does NOT beat fixed 11:00 with an interval excluding zero, nor fixed 13:30. It is the defensible causal choice, not a demonstrated improvement.

## 9. The corrected delta (proposal 36's V9) on the 11:00 book

The hedge delta is taken from the implied volatility corrected by the ledger's trailing lagged per-clock mean of realized over implied variance (`pricing.corrected_total_vol`), with the research's fallback to the uncorrected implied wherever the factor is missing or the vendor implied sat on the solver's bracket node. Book: the crossed-quoted 15:30 exit of the 11:00 straddle.

| sample | n | Sharpe V0 | Sharpe V9 | dSharpe | 95% CI | vs 36 |
| --- | --- | --- | --- | --- | --- | --- |
| whole | 865 | 2.252547 | 2.383504 | +0.130957 | [+0.05267, +0.21936] | 5.3e-16 |
| daily_era | 489 | 2.513132 | 2.688684 | +0.175552 | [+0.06329, +0.30051] | 5.1e-16 |

Ledger: `C:\Users\james\CC Allowed\harxhar-0dte-professor\results\live_seed\premium_ledger.parquet` (1279 sessions). The V9 factor here is the ledger's own estimator fed proposal 36's implied and realized grids, so the reproduction is of the estimator as well as of the book. None of these numbers is charged the 0.5 bp hedge cost, and V9 raises turnover: proposal 27's reading, that the exit book nets about Sharpe 1.20 at 0.5 bp and V9 about 1.31, is the one to quote.
