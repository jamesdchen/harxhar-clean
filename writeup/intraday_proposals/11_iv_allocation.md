# 11. Reallocating the remaining implied variance across the day

Read-only study, **pre-registered on four share variants**. Script
`writeup/intraday_proposals/11_iv_allocation.py`; tables and figure in
`results/atm_straddle_intraday/proposals/11/` (`11_gate.csv`, `11_warmup.csv`,
`11_calibration_by_clock.csv`, `11_calibration_dispersion.csv`,
`11_shares_by_clock.csv`, `11_qlike_by_clock.csv`, `11_sign_rule.csv`,
`11_by_clock_rates.csv`, `11_paired.csv`, `11_placebo.csv`, `11_causality.csv`,
`11_calibration_by_clock.png`). Nothing is wired into any notebook. Every
number below is printed by the script.

> **Caveat, up front.** The deck prices the next 30-minute bar as a *share* of
> the remaining-session implied variance: `slice_t = iv_hourly_t^2 * h_t * w_t`.
> A share reallocates a fixed quantity. It cannot add one bit of day-by-day
> information. On any given day it only moves variance between clocks; the
> total is whatever the option market says it is. So the question a better
> share can answer is a **calibration** question --- is the bar-level price
> right at each clock --- and only after that a trading question. The scoring
> below is in that order: calibration by clock first, the sign rule second.
> A variant that improves the sign rule without improving calibration has not
> found a better price; it has changed how often the rule goes short.

> **Multiplicity.** Four variants, fixed before any of them was scored.
> Nothing is selected from a grid. V0 is the deck's current share and is the
> control.

> **Provenance.** Block-diagonal ridge forecast, bar-end labelled panel, the
> deck's 866 days from `daily_blk2.parquet`, trade cache
> `trade_77439812_1786671584951439500_f24191ffee_9d76153205.parquet`,
> 10,387 bars at twelve clocks 10:00--15:30.

## Gate

The study reproduces the intraday notebook's rule table under V0 before
anything else is reported. Midpoint fills, daily sums, 866 days.

| rule | reproduced | notebook | abs diff |
|---|---|---|---|
| always short | 1.9020389 | 1.9020389 | 0.0000000 |
| sign(s) V0 | 1.7205907 | 1.7205907 | 0.0000000 |
| hybrid: always short, sign(s) at 15:30 | 3.1714284 | 3.1714284 | 0.0000000 |

Gate passed to 0 at the printed precision, inside the 1e-3 tolerance.

## The four variants

All four are F_t-measurable. Every trailing statistic uses prior days only with
`shift(1)` and a 63-session minimum.

- **V0 unconditional.** `w_t` = the trailing expanding per-clock mean of
  realized bar variance, divided by the remaining-session sum of those means.
  The deck's current share.
- **V1 event-conditional.** Separate trailing per-clock profiles for FOMC
  statement days (33), month-end sessions (51), third-Friday and quad-witching
  expirations (51) and all other days (731); first match in that order. Each
  conditional profile is shrunk toward the unconditional one with weight
  `n_cond / (n_cond + 20)`. `is_fomc` is a nullable boolean and is NA beyond
  the release file's horizon; NA is read as "other", never as FOMC.
- **V2 market-implied.** On each prior day the forward implied variance of bar
  `t` is `IV_rem_t - IV_rem_{t+1}`, floored at zero (both stamps are known ex
  post for a prior day). The trailing expanding per-clock mean of
  `forward / IV_rem` is the market's own average allocation. It is turned into
  an allocation over the day, `a_c = r_c * prod_{j<c}(1 - r_j)`, and
  renormalized so the shares of the remaining clocks sum to one.
- **V3 same-day-conditioned.** A per-clock regression, fit on prior days only
  (expanding, refit each day, 63-day minimum), of the realized share of the
  remaining variance taken by this bar on: the log of today's cumulative
  realized variance so far relative to its trailing per-clock expectation; the
  log of `iv_hourly_t` relative to its trailing per-clock mean; and four
  day-of-week indicators. Predictions are clipped to (0.02, 0.98) and
  renormalized across the remaining clocks. At clock `t` the only day state
  that exists is the one measured at `t`, so every remaining clock's regression
  is evaluated at clock `t`'s regressors.

**At 15:30 there is one bar left, so `w = 1` for every variant by
construction.** Maximum `|w - 1|` at 15:30 is exactly 0 across all 866
settlement bars, and the settlement position is identical wherever all four
variants are warm. Every difference in this file lives in the daytime legs.

**Warm-up.** The variants warm up at different rates, so the paired work runs
on the common support: the 8,815 bars on 735 days that all four price.

| variant | warm-up bars | warm-up dates | days with a signal |
|---|---|---|---|
| V0 unconditional | 756 | 64 | 803 |
| V1 event-conditional | 756 | 64 | 803 |
| V2 market-implied | 815 | 68 | 798 |
| V3 same-day-conditioned | 1,572 | 132 | 735 |

A bar with no share sits flat, `q = 0`, and its zero stays in the daily sum.

## 1. Calibration first

`mean(RV_bar) / mean(slice)` by clock. 1.0 is fair; above 1 the slice is cheap.
Common support, 735 days. Figure: `11_calibration_by_clock.png`.

| clock | V0 | V1 | V2 | V3 |
|---|---|---|---|---|
| 10:00 | 1.070 | 1.057 | 0.871 | 0.879 |
| 10:30 | 0.989 | 0.983 | 0.749 | 0.801 |
| 11:00 | 1.038 | 1.039 | 0.853 | 0.853 |
| 11:30 | 0.931 | 0.930 | 0.705 | 0.844 |
| 12:00 | 1.017 | 1.021 | 0.816 | 0.869 |
| 12:30 | 0.928 | 0.940 | 0.723 | 0.880 |
| 13:00 | 1.060 | 1.078 | 0.878 | 0.947 |
| 13:30 | 0.853 | 0.867 | 0.883 | 0.893 |
| 14:00 | 1.071 | 1.050 | 1.086 | 0.826 |
| 14:30 | 1.100 | 1.076 | 1.231 | 0.937 |
| 15:00 | 1.067 | 1.068 | 1.014 | 0.836 |
| 15:30 | 0.793 | 0.793 | 0.793 | 0.793 |

`median(RV_bar / slice)` by clock runs 0.760--1.029 under V0, 0.760--0.972
under V1, 0.650--0.856 under V2 and 0.714--0.816 under V3; the full column is
in `11_calibration_by_clock.csv`.

Pooled calibration statistic --- the dispersion across clocks of
`mean(RV)/mean(slice)`, which a good allocation makes flat:

| variant | sd across clocks | range | mean abs deviation from 1 |
|---|---|---|---|
| V0 unconditional | 0.0971 | 0.3071 | 0.0774 |
| V1 event-conditional | 0.0919 | 0.2852 | 0.0731 |
| V2 market-implied | 0.1563 | 0.5263 | 0.1718 |
| V3 same-day-conditioned | 0.0480 | 0.1539 | 0.1368 |

Read the two columns together. V3 is by far the flattest across clocks --- and
uniformly too rich: its ratios sit near 0.85 at every clock, so it is further
from fair in level than V0 is. V1 improves both, slightly. V2 is worse on both.

QLIKE of the slice as a forecast of the bar's realized variance, with the
paired Diebold--Mariano t against V0 (autocorrelation-robust, lag
`floor(1.5 n^(1/3))`; a negative t favours the variant):

| clock | QLIKE V0 | QLIKE V1 | QLIKE V2 | QLIKE V3 | DM t V1 | DM t V2 | DM t V3 |
|---|---|---|---|---|---|---|---|
| 10:00 | 0.1539 | 0.1433 | 0.1557 | 0.1419 | -4.93 | +0.36 | -2.12 |
| 10:30 | 0.1602 | 0.1601 | 0.1792 | 0.1385 | -0.08 | +2.79 | -3.33 |
| 11:00 | 0.1814 | 0.1672 | 0.1871 | 0.1511 | -4.80 | +1.00 | -3.30 |
| 11:30 | 0.1649 | 0.1601 | 0.2080 | 0.1386 | -2.64 | +6.33 | -3.90 |
| 12:00 | 0.1803 | 0.1634 | 0.1955 | 0.1507 | -5.06 | +2.38 | -3.42 |
| 12:30 | 0.1861 | 0.1716 | 0.2097 | 0.1581 | -4.91 | +3.50 | -4.29 |
| 13:00 | 0.2395 | 0.2099 | 0.2246 | 0.1726 | -5.78 | -1.93 | -6.68 |
| 13:30 | 0.1980 | 0.1820 | 0.1893 | 0.1641 | -4.16 | -4.16 | -5.25 |
| 14:00 | 0.1755 | 0.1743 | 0.1745 | 0.1581 | -1.18 | -1.14 | -2.40 |
| 14:30 | 0.1620 | 0.1559 | 0.1587 | 0.1565 | -1.96 | -1.32 | -1.03 |
| 15:00 | 0.1008 | 0.0995 | 0.1045 | 0.1129 | -1.13 | +2.14 | +3.32 |
| 15:30 | 0.0980 | 0.0980 | 0.0980 | 0.0980 | --- | --- | --- |
| **pooled** | **0.1667** | **0.1571** | **0.1737** | **0.1451** | **-5.72** | **+3.37** | **-5.78** |

The 15:30 row is identical by construction, so its differential is exactly zero
and the t is undefined.

So on the forecast question the answer is clean. V1 and V3 are better bar-level
prices than V0, significantly and at almost every clock. V2 is worse. V3 is the
best QLIKE of the four and the flattest across clocks; it pays for that by
being rich everywhere.

## 2. Then the sign rule

`q = sign(rv_hat - slice_v)`. Pooled daily-sum Sharpe times sqrt(252).
`t` is the plain t of the daily sums; `t_HAC` is autocorrelation-robust.
maxDD is in index points on the daily sum of points.

**Common support (735 days), midpoint fills:**

| rule | Sharpe | mean/day | t | t_HAC | crossings/day | maxDD (pts) | % long |
|---|---|---|---|---|---|---|---|
| always short | +2.045 | +0.172 | +3.49 | +3.84 | 0.00 | -150.1 | 0.0 |
| sign(s) V0 | +1.464 | +0.122 | +2.50 | +2.46 | 0.00 | -145.6 | 56.9 |
| sign(s) V1 | +1.215 | +0.102 | +2.08 | +2.12 | 0.00 | -140.5 | 55.0 |
| sign(s) V2 | +1.791 | +0.152 | +3.06 | +2.85 | 0.00 | -176.4 | 41.9 |
| sign(s) V3 | +1.496 | +0.125 | +2.56 | +2.41 | 0.00 | -136.9 | 39.9 |
| hybrid (identical V0--V3) | +3.034 | +0.252 | +5.18 | +5.52 | 0.00 | -188.2 | 3.2 |

**Common support (735 days), crossed spread:**

| rule | Sharpe | mean/day | t | t_HAC | crossings/day | maxDD (pts) | % long |
|---|---|---|---|---|---|---|---|
| always short | -1.329 | -0.114 | -2.27 | -2.47 | 16.47 | -1750.3 | 0.0 |
| sign(s) V0 | -2.412 | -0.201 | -4.12 | -4.11 | 18.36 | -2567.0 | 56.9 |
| sign(s) V1 | -2.628 | -0.220 | -4.49 | -4.64 | 18.37 | -2719.5 | 55.0 |
| sign(s) V2 | -1.964 | -0.167 | -3.36 | -3.19 | 18.17 | -2318.1 | 41.9 |
| sign(s) V3 | -2.358 | -0.197 | -4.03 | -3.86 | 18.33 | -2580.7 | 39.9 |
| hybrid (identical V0--V3) | -0.500 | -0.042 | -0.85 | -0.90 | 16.75 | -1713.6 | 3.2 |

**On the full 866 days** (each variant carries its own warm-up flats): mid
1.721 / 1.405 / 1.953 / 1.378 for V0 / V1 / V2 / V3, crossed -2.243 / -2.538 /
-1.863 / -2.169. Always short is 1.902 mid and -1.918 crossed; the hybrid is
3.171 mid and -0.969 crossed. The full frame is in `11_sign_rule.csv`.

**The hybrid.** At 15:30 every variant collapses to `w = 1`, so the settlement
leg --- and therefore the hybrid `always short before the close, sign(s) at
15:30` --- is identical across the four. It is 3.171 mid and -0.969 crossed on
866 days no matter which share is used during the day. This file cannot move
it.

**% long by clock** (common support, `11_by_clock_rates.csv`) runs, across the
eleven daytime clocks, 48.8--70.9 under V0, 41.4--69.9 under V1, 20.4--69.0
under V2 and 23.8--68.0 under V3. At 15:30 it is 38.6 for all four.

**The oracle short rate under each variant's slice** --- the share of bars on
which the realized bar variance actually came in below the slice --- answers
the 13:00 question directly:

| clock | V0 | V1 | V2 | V3 |
|---|---|---|---|---|
| 10:00 | 54.4 | 57.1 | 69.5 | 72.3 |
| 10:30 | 56.5 | 56.4 | 76.7 | 74.8 |
| 11:00 | 56.9 | 60.5 | 70.3 | 74.8 |
| 11:30 | 62.3 | 64.2 | 80.8 | 75.9 |
| 12:00 | 55.7 | 60.8 | 71.3 | 73.3 |
| 12:30 | 59.0 | 62.0 | 75.8 | 70.3 |
| 13:00 | 48.2 | 52.5 | 61.4 | 64.8 |
| 13:30 | 65.0 | 71.0 | 63.7 | 70.5 |
| 14:00 | 66.5 | 65.4 | 65.9 | 66.9 |
| 14:30 | 72.4 | 65.6 | 63.7 | 67.5 |
| 15:00 | 59.5 | 57.0 | 64.1 | 73.2 |
| 15:30 | 72.4 | 72.4 | 72.4 | 72.4 |

Under V0 the 13:00 bar is the cheapest slice of the day at 48.2% short. V1
moves it to 52.5%. V3 moves it to 64.8% and squeezes the whole day into
64.8--75.9. So yes, the 13:00 under-allocation goes away --- but it goes away
because V3 makes *every* clock rich, not because it found the missing variance
at 13:00. That is the same fact as V3's 0.85-everywhere calibration column.

## 3. Paired differences against V0

Daily sums, common support, 735 days. Circular block bootstrap of the Sharpe
difference: block 21, B = 2000, rng(0), draws shared across variants.

| fill | variant | Sharpe V0 | Sharpe variant | mean daily diff | t | t_HAC | dSharpe | percentile 95% | basic 95% |
|---|---|---|---|---|---|---|---|---|---|
| mid | V1 | +1.464 | +1.215 | -0.0199 | -1.64 | -1.88 | -0.249 | [-0.498, -0.014] | [-0.484, +0.001] |
| mid | V2 | +1.464 | +1.791 | +0.0298 | +1.84 | +1.94 | +0.327 | [-0.051, +0.679] | [-0.025, +0.706] |
| mid | V3 | +1.464 | +1.496 | +0.0032 | +0.15 | +0.17 | +0.032 | [-0.380, +0.463] | [-0.398, +0.444] |
| crossed | V1 | -2.412 | -2.628 | -0.0198 | -1.62 | -1.86 | -0.216 | [-0.474, +0.032] | [-0.465, +0.041] |
| crossed | V2 | -2.412 | -1.964 | +0.0341 | +2.10 | +2.16 | **+0.448** | **[+0.057, +0.822]** | **[+0.073, +0.839]** |
| crossed | V3 | -2.412 | -2.358 | +0.0032 | +0.15 | +0.17 | +0.054 | [-0.367, +0.507] | [-0.400, +0.474] |

On the full 866 days the same table gives V1 -0.315 mid / -0.295 crossed (both
intervals excluding zero, against the variant), V2 +0.233 mid / +0.380 crossed
(percentile covering zero at both fills) and V3 -0.343 mid / +0.074 crossed
(covering zero). Full table in `11_paired.csv`.

One variant clears the crossed-spread bar with an interval excluding zero:
**V2**, on the common support, at both the percentile and the basic interval.
It is also the one variant whose calibration is *worse* than V0's.

## 4. Placebo

Both V2 and V3 beat V0 at the crossed spread on the common support, so both are
placebo-tested: 200 random reshufflings of that variant's shares across clocks
within each day, preserving the day's allocation sum to one. (The reshuffle
leaves `w = 1` at 15:30, so it perturbs exactly the daytime legs.)

| variant | real Sharpe crossed | placebo mean | placebo 95th | percentile of the real variant |
|---|---|---|---|---|
| V2 market-implied | -1.964 | -1.958 | -1.615 | **49.0** |
| V3 same-day-conditioned | -2.358 | -1.979 | -1.620 | 2.5 |

This is the finding that settles the file. V2's crossed-spread Sharpe sits at
the **49th percentile** of its own reshufflings: scrambling which clock gets
which share leaves the trade exactly where it was. V2's gain is not in the
allocation. It is in the level --- V2's shares are larger, the slice is richer,
and the rule goes short on 58.1% of bars instead of V0's 43.1%. At the crossed
spread always short (-1.329) beats sign(s) V0 (-2.412), so any variant that
shorts more looks better. That is a re-weighting toward the forecast-free rule,
not a better price. V3's real ordering is at the 2.5th percentile --- worse
than 97.5% of random reshufflings of its own shares.

## 5. Causality

Multiply one bar's realized variance by 10 and rebuild every share; nothing
used on the perturbed day may move. Ten cut points, drawn from day 200 onward.

- V0: 0 of 120 share cells moved on the perturbed day.
- V1: 0 of 120.
- V2: 0 of 120 (V2 never touches realized variance at all).
- V3: 0 of 65 bars at or before the perturbed bar; 45 of 55 bars after it.

That is exactly the assertion the pre-registration asks for. Only V3's same-day
cumulative term may move, and only for bars strictly after the perturbed one.
Per-cut detail in `11_causality.csv`.

## Verdict

The gate reproduces the deck's rule table under V0 to zero at the printed
precision, so the frame is the notebook's. **V1 event-conditional** improves
calibration --- dispersion across clocks 0.0919 against V0's 0.0971, mean
absolute deviation from fair 0.0731 against 0.0774, pooled QLIKE 0.1571 against
0.1667 with a paired Diebold--Mariano t of -5.72 --- and loses the sign(s)
trade at both fills: 1.215 against 1.464 at the midpoint and -2.628 against
-2.412 at the crossed spread, difference -0.216 with a percentile interval of
[-0.474, +0.032] that covers zero. **Kill.** **V2 market-implied** is the only
variant that beats V0 at the crossed spread with an interval excluding zero
(-1.964 against -2.412, difference +0.448, percentile [+0.057, +0.822], basic
[+0.073, +0.839]; +1.791 against +1.464 at the midpoint), and it is the one
variant whose calibration is *worse* --- dispersion 0.1563, mean absolute
deviation 0.1718, pooled QLIKE 0.1737 with a Diebold--Mariano t of +3.37
against it --- and its crossed-spread Sharpe sits at the 49th percentile of 200
reshufflings of its own shares across clocks, so the gain survives scrambling
the allocation entirely: it is a shift toward shorting (58.1% of bars against
V0's 43.1%) on a day where always short beats sign(s) at the crossed spread,
not a better price. **Kill.** **V3 same-day-conditioned** is the best bar-level
price in the file --- flattest across clocks at 0.0480 dispersion, best pooled
QLIKE at 0.1451, Diebold--Mariano t -5.78, and it removes the 13:00
under-allocation (oracle short rate 48.2% to 64.8%) --- but it buys that
flatness by making every clock uniformly rich (ratios near 0.85, mean absolute
deviation from fair 0.1368 against V0's 0.0774), it does not move the trade at
either fill (+1.496 mid, -2.358 crossed, difference +0.054 with a percentile
interval of [-0.367, +0.507]), its real clock ordering is at the 2.5th
percentile of its own reshufflings, and it costs 68 further warm-up days.
**Kill.** Nothing is adopted: the standing bar is beating V0 at the crossed
spread with an interval excluding zero **and** improving calibration, and no
variant does both. **V0 stands.** The deeper reading is the caveat made
concrete --- the allocation is not the binding constraint. Three of four
variants move the bar-level price a long way, the best of them cuts pooled
QLIKE by 13%, and the sign(s) trade does not move at all beyond what a richer
slice buys by shorting more. And at 15:30, where the slice is an actual price
and the trade actually survives the spread, all four variants are the same
number by construction. The frontier is still the last bar.
