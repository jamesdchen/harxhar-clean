# 12. The remaining-session oracle, held from clock t to the close

An oracle is a measurement, not a trade. It is a position sized by a quantity
that does not exist when the position is taken — here the realized variance of
the session still to run — and it exists to bound what any forecast of that
quantity could be worth, and to describe the shape of the thing being forecast.
It is not a rule, it cannot be run, and no part of it is executable. This study
adopts nothing, selects nothing and recommends nothing. It is a description of
one object: at each clock t in 10:00–15:30, the sign of the session's remaining
realized variance against the market's price of that same remainder, applied to
the nearest out-of-the-money straddle picked at t and held to cash settlement at
the official close. Every number below is printed by
`writeup/intraday_proposals/12_oracle_hold_to_close.py`; tables and figures are
in `results/atm_straddle_intraday/proposals/12/`.

## The object

At clock t, with h hours to the 16:00 close,

- `IV_rem = iv_hourly_t^2 * h` — the market's price of the remaining window;
- `RV_rem = ` the sum of the session's realized bar variance from t to 16:00;
- `q = sign(RV_rem − IV_rem)` — long when the remainder outran its price.

`q` is applied to the nearest out-of-the-money straddle picked at t and held to
settlement on those strikes. One crossing at entry, no re-pick, no exit spread.
The position drifts off the money as the index moves; nothing is done about it.

Keep this distinct from the thirty-minute slice oracle of proposal 10's Check 0,
which signs the next bar's realized variance against a share of the remaining
implied. That object prices thirty minutes. This one prices the rest of the
session. Its pooled short rate here (68.1–75.2% by clock) reproduces Check 0's
definition (3) (67.9–75.3%) up to the day frame: Check 0 scored the 803 days
that carry a signal, this file scores the deck's 866.

## Frame and provenance

Block-diagonal ridge forecast. The intraday notebook's cached package file
`trade_77439812_1786671584951439500_f24191ffee_9d76153205.parquet` joined to the
bar-end labelled forecast panel — the bar `[t, t+30]` and its realized variance
sit on the row stamped `t+30` — restricted to the deck's 866 days from
`results/atm_straddle_0dte_1530/daily_blk2.parquet` (forecast fingerprint
`a5c65541dcf0`), 2020-01-03 to 2024-04-30. **10,387 packages, twelve clocks
10:00–15:30.** The package guards refuse 5 of the 10,392 day-clock cells; 13
further rows carry no vendor implied volatility (solver-node censored), so
**10,374 rows carry an oracle.** Strike pairs: 10,314 one strike apart, 52
wider, 21 on the same strike.

The remaining realized variance is summed from the **panel**, not from the trade
rows. The panel prices all twelve session bars on all 866 days, so the five
refused package cells do not shorten anyone's remainder.

## Gates

1. **`IV_rem` is a price.** Black–Scholes–Merton on the picked pair at the
   vendor's hourly implied volatility over the remaining window reproduces the
   quoted package midpoint: median ratio **1.0000** at every clock, 5th–95th
   percentile 0.9995–1.0004 at 10:00 widening to 0.9972–1.0063 at 15:30.
2. **The 15:30 row is the deck's close trade.** Max absolute difference on 866
   days: strikes 0, official close 0, settlement payoff 0, entry midpoint
   9.5e-07 (the chain's float32 quotes), held return 2.0e-07, `IV_rem` against
   the deck's `iv_var` 2.7e-20.
3. **At 15:30 the remaining session is the entry bar**: max |`RV_rem` −
   `RV_bar`| = 0.
4. **The deck's own rule table on the file this run scored**: always short
   Sharpe 0.2038 (t 0.3778), sign(s) 1.3383 (t 2.4810), 39.95% of days long.
5. **The payoff identity.** The held package pays the terminal displacement from
   the centre of the strike pair, less half the pair's width, floored at zero:
   max absolute difference 0. This is what section 3 is about.

**The peek, made explicit.** Tripling the realized variance of every bar
*strictly after* the entry bar moves the oracle's sign on 67.1–71.1% of rows at
every daytime clock, and on **0** of 861 rows at 15:30, where there are no later
bars. The sign at 10:00 is made almost entirely of information that does not
exist at 10:00. That is the definition of the object, stated as a measurement.

## 1. The gap: how rich is the remainder, and when

`RV_rem / IV_rem` by entry clock. 1.0 means the remainder came in exactly at its
price. Figure `12_gap_by_clock.png`; table `12_gap_quantiles.csv`.

| clock | n | p10 | p25 | p50 | p75 | p90 | mean | pct short | mean gap / IV_rem | t of the mean gap |
|---|---|---|---|---|---|---|---|---|---|---|
| 10:00 | 864 | 0.523 | 0.661 | 0.835 | 1.072 | 1.376 | 0.927 | 68.98 | −0.073 | −2.95 |
| 10:30 | 865 | 0.511 | 0.649 | 0.833 | 1.062 | 1.395 | 0.922 | 68.90 | −0.078 | −3.82 |
| 11:00 | 865 | 0.519 | 0.651 | 0.838 | 1.070 | 1.350 | 0.924 | 69.36 | −0.076 | −3.98 |
| 11:30 | 864 | 0.527 | 0.644 | 0.834 | 1.052 | 1.358 | 0.917 | 70.72 | −0.083 | −3.84 |
| 12:00 | 866 | 0.515 | 0.642 | 0.837 | 1.085 | 1.361 | 0.930 | 68.13 | −0.070 | −3.60 |
| 12:30 | 866 | 0.525 | 0.637 | 0.828 | 1.052 | 1.344 | 0.926 | 69.28 | −0.074 | −3.44 |
| 13:00 | 866 | 0.532 | 0.640 | 0.817 | 1.061 | 1.339 | 0.929 | 69.40 | −0.071 | −3.42 |
| 13:30 | 866 | 0.532 | 0.639 | 0.795 | 1.028 | 1.320 | 0.910 | 72.17 | −0.090 | −3.93 |
| 14:00 | 864 | 0.526 | 0.639 | 0.801 | 1.031 | 1.316 | 0.899 | 71.30 | −0.101 | −4.50 |
| 14:30 | 865 | 0.508 | 0.625 | 0.784 | 1.008 | 1.291 | 0.871 | 73.53 | −0.129 | −4.72 |
| 15:00 | 862 | 0.507 | 0.623 | 0.776 | 0.998 | 1.291 | 0.860 | 75.17 | −0.141 | −4.73 |
| 15:30 | 861 | 0.463 | 0.590 | 0.757 | 1.029 | 1.318 | 0.855 | 72.82 | −0.145 | −5.97 |

The t is the autocorrelation-robust t of the mean gap against zero.

The remainder is short of its price at every clock. The median ratio is 0.835 at
10:00 and 0.757 at 15:30 — richness of 16.5% of the remaining variance in the
morning and 24.3% at the close. The short rate runs 68.1% to 75.2%. So the
answer to the question posed is: **the remaining implied is rich all day and the
richness builds only mildly into the close.** There is no clock at which it is
fair, and no clock at which it is 80% short.

The dispersion swamps the level and barely moves with the clock: the tenth
percentile sits at 0.51–0.53 and the ninetieth at 1.29–1.39 at every daytime
clock, and at 15:00 the seventy-fifth percentile is 0.998 — a quarter of days
come in at or above their price. A level this stable with a spread this wide is
the reason the short rate is a tilt and not a rule.

## 2. Does the day's sign settle early?

On the 858 days that price all twelve clocks. Tables `12_flips.csv`,
`12_persistence.csv`, `12_transitions.csv`, `12_transitions_pooled.csv`.

**The sign never flips across the twelve clocks on 378 of 858 days (44.06%)** —
334 all-short days and 44 all-long days.

| flips per day | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| days | 378 | 166 | 145 | 88 | 43 | 30 | 4 | 3 | 1 |
| pct | 44.06 | 19.35 | 16.90 | 10.26 | 5.01 | 3.50 | 0.47 | 0.35 | 0.12 |

Mean 1.276 flips a day, median 1, max 8.

Agreement with the sign at 15:30:

| clock | n | agrees with 15:30 % | short at t → short at 15:30 % | long at t → long at 15:30 % |
|---|---|---|---|---|
| 10:00 | 859 | 66.47 | 78.41 | 39.85 |
| 10:30 | 860 | 68.14 | 79.63 | 42.48 |
| 11:00 | 860 | 68.49 | 79.63 | 42.91 |
| 11:30 | 859 | 69.03 | 79.47 | 43.60 |
| 12:00 | 861 | 71.54 | 82.48 | 47.99 |
| 12:30 | 861 | 71.78 | 82.11 | 48.29 |
| 13:00 | 861 | 70.96 | 81.47 | 46.95 |
| 13:30 | 861 | 73.05 | 81.70 | 50.42 |
| 14:00 | 861 | 73.98 | 82.76 | 52.03 |
| 14:30 | 861 | 77.00 | 83.78 | 57.96 |
| 15:00 | 861 | 82.58 | 86.86 | 69.63 |
| 15:30 | 861 | 100.00 | 100.00 | 100.00 |

Transitions between consecutive clocks, pooled over the eleven pairs (counts,
then row percentages):

| | to short | to long |
|---|---|---|
| from short | 6,180 | 536 |
| from long | 568 | 2,222 |
| **from short %** | **92.02** | 7.98 |
| **from long %** | 20.36 | **79.64** |

Per pair, short → short runs 86.86% (15:00 → 15:30) to 94.58% (12:00 → 12:30),
and long → long runs 69.63% (15:00 → 15:30) to 86.56% (11:30 → 12:00).

Read the asymmetry, because it is the structure. **A short call is sticky and a
long call is not.** Once the remainder is running below its price it stays below
on 92% of consecutive pairs; once it is running above, it falls back on 20%. In
the same direction: a 10:00 short is still short at 15:30 on 78.4% of days, a
10:00 long is still long on only 39.9%. The 15:30 sign is therefore *mostly*
knowable early on short days and *not* knowable early on long days — as an
oracle fact about the day, not as a forecast anyone could issue. The last pair,
15:00 → 15:30, is the least persistent of the eleven in both directions, which
is the same statement report 09 made from the other side: the last bar is its
own object.

## 3. Variance against drift in the held straddle

Gate 5 says what the held package pays: the terminal displacement of the index
from the centre of the strike pair, less half the pair's width, floored at zero.
That is a statement about where the index *ends*, not about how much variance
the path delivered. Section 3 measures the difference. Table
`12_variance_vs_drift.csv`.

The held short's P&L in points splits exactly:

`entry − payoff = (entry − E|move|) + (E|move| − payoff)`,

with `E|move| = S_t sqrt(2 RV_rem / π)`, the displacement a driftless diffusion
with the day's *own* remaining realized variance would deliver. The first term
is the price against the value of the realized remainder — the variance term.
The second is that value against where the index actually landed — the
directional term. The two covariance shares of `var(entry − payoff)` sum to 1 to
2.2e-16.

The gap is positive when the remainder outran its price, so the informative sign
of `corr(sign gap, sign short P&L)` is negative.

| clock | corr(sign gap, sign short P&L) | oracle hit rate % | R² held return on the gap | variance share | directional share | corr(sign gap, sign variance term) | R² payoff on E\|move\| | oracle Sharpe, held pair | oracle Sharpe if it paid E\|move\| | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 10:00 | −0.155 | 60.42 | 0.158 | 0.050 | 0.950 | −0.615 | 0.258 | 3.31 | 10.62 | 0.31 |
| 10:30 | −0.138 | 60.23 | 0.176 | 0.055 | 0.945 | −0.586 | 0.261 | 3.25 | 10.16 | 0.32 |
| 11:00 | −0.107 | 58.15 | 0.231 | 0.053 | 0.947 | −0.541 | 0.253 | 3.43 | 9.19 | 0.37 |
| 11:30 | −0.091 | 58.33 | 0.258 | 0.052 | 0.948 | −0.523 | 0.299 | 3.06 | 8.55 | 0.36 |
| 12:00 | −0.117 | 59.24 | 0.268 | 0.057 | 0.943 | −0.539 | 0.310 | 3.25 | 8.49 | 0.38 |
| 12:30 | −0.142 | 60.16 | 0.284 | 0.056 | 0.944 | −0.509 | 0.299 | 3.60 | 7.54 | 0.48 |
| 13:00 | −0.168 | 61.32 | 0.321 | 0.063 | 0.937 | −0.493 | 0.331 | 3.87 | 6.86 | 0.56 |
| 13:30 | −0.163 | 61.32 | 0.304 | 0.060 | 0.940 | −0.434 | 0.350 | 3.92 | 6.04 | 0.65 |
| 14:00 | −0.201 | 63.54 | 0.275 | 0.078 | 0.922 | −0.431 | 0.404 | 4.55 | 5.74 | 0.79 |
| 14:30 | −0.157 | 62.20 | 0.173 | 0.056 | 0.944 | −0.389 | 0.411 | 4.20 | 4.74 | 0.89 |
| 15:00 | −0.214 | 65.78 | 0.136 | 0.057 | 0.943 | −0.332 | 0.324 | 4.72 | 3.51 | 1.35 |
| 15:30 | −0.172 | 62.83 | 0.140 | 0.067 | 0.933 | −0.301 | 0.342 | 4.54 | 3.31 | 1.37 |

R² of the held *points* on the gap runs 0.071 (10:00) to 0.142 (14:00) and back
to 0.076 (15:30); the return column above is the same regression per unit of
entry premium.

Three readings.

**The gap signs the held P&L weakly at every clock.** The sign correlation is
−0.09 to −0.21 and the oracle's hit rate on the held package is 58.2% to 65.8%.
An oracle that knows the exact remaining variance still gets the held straddle's
sign wrong on more than a third of days.

**The direction the index takes carries 92–95% of the held P&L's variance at
every clock.** The variance term's share never exceeds 7.8%. So the held
straddle is a bet on the terminal displacement, dressed as a variance position,
at every entry clock — including the close.

**What changes with the clock is how much of the variance signal survives that.**
The last three columns are the measurement. If the package paid `E|move|`
instead of its own payoff — a smooth function of the day's realized remainder,
priced at the same premium — the oracle would score 10.62 at 10:00 and 3.31 at
15:30. It actually scores 3.31 and 4.54. The ratio rises from 0.31 at 10:00 to
1.37 at 15:30, monotone but for one dip at 11:30: 0.31 at 10:00, 0.38 at noon,
0.56 at 13:00, 0.79 at 14:00, 1.35 at 15:00.
From a morning entry the held package delivers under a third of the Sharpe a
variance payoff would; from the last hour it delivers more, because the payoff's
zero floor is a gift to a short (22.4% of 15:30 packages settle worthless
against 7.9% of 10:00 packages) and the strikes have not had time to run away.
Caveat: the hypothetical is priced at the *out-of-the-money* pair's own premium,
which is below an at-the-money straddle's, so its level is optimistic; the shape
across clocks is what the column is for.

The same shape appears in the raw correlations. `corr(|S_close − S_t|, E|move|)`
— how tightly the terminal displacement tracks the day's own remaining
volatility — is 0.508 at 10:00, 0.637 at 14:00 and 0.587 at 15:30, and the R² of
the payoff on `E|move|` runs 0.258 → 0.411 → 0.342. And
`corr(sign gap, sign variance term)` decays from −0.615 at 10:00 to −0.301 at
15:30: early in the day the gap almost *is* the variance term, because the
premium is large and the six-hour remainder dominates it; at the close the two
have come apart.

## 4. The ceiling: what the oracle would have earned

One trade a day, entered at clock t and held to cash settlement. Midpoint, then
the crossed spread: one crossing at entry, none at settlement. Comparator: always
short the same package, held the same way. Sharpe is daily, annualized by
`sqrt(PERIODS_PER_YEAR)`. Points are index points per straddle. Table
`12_pnl_by_clock.csv`; figure `12_cum_points.png`. No row is untradeable at the
crossed spread (0 of 10,374).

| clock | n | pct long | oracle Sharpe mid | t | oracle Sharpe crossed | oracle pts/day mid | crossed | oracle maxDD mid | always short mid | crossed | always short pts/day mid | median premium | median half-spread % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10:00 | 864 | 31.02 | 3.31 | 6.12 | 2.90 | 3.06 | 2.50 | −481 | −0.27 | −0.66 | −0.08 | 19.83 | 1.58 |
| 10:30 | 865 | 31.10 | 3.25 | 6.03 | 3.00 | 3.11 | 2.80 | −425 | −0.01 | −0.27 | 0.23 | 18.30 | 1.22 |
| 11:00 | 865 | 30.64 | 3.43 | 6.35 | 3.20 | 3.11 | 2.83 | −417 | −0.18 | −0.42 | 0.02 | 17.05 | 1.27 |
| 11:30 | 864 | 29.28 | 3.06 | 5.67 | 2.82 | 2.33 | 2.06 | −262 | 0.07 | −0.18 | 0.20 | 16.05 | 1.35 |
| 12:00 | 866 | 31.87 | 3.25 | 6.02 | 3.01 | 2.60 | 2.34 | −336 | 0.08 | −0.17 | 0.22 | 14.83 | 1.38 |
| 12:30 | 866 | 30.72 | 3.60 | 6.68 | 3.37 | 2.87 | 2.61 | −312 | −0.04 | −0.29 | 0.22 | 13.68 | 1.46 |
| 13:00 | 866 | 30.60 | 3.87 | 7.18 | 3.63 | 3.26 | 3.01 | −213 | −0.38 | −0.63 | −0.11 | 12.65 | 1.54 |
| 13:30 | 866 | 27.83 | 3.92 | 7.27 | 3.67 | 2.91 | 2.67 | −209 | −0.18 | −0.45 | 0.07 | 11.70 | 1.66 |
| 14:00 | 864 | 28.70 | 4.55 | 8.43 | 4.18 | 3.16 | 2.79 | −180 | 0.02 | −0.37 | 0.09 | 10.55 | 1.86 |
| 14:30 | 865 | 26.47 | 4.20 | 7.77 | 3.85 | 2.57 | 2.33 | −237 | 0.55 | 0.20 | 0.15 | 9.30 | 1.96 |
| **15:00** | 862 | 24.83 | **4.72** | 8.72 | **4.35** | 2.54 | 2.31 | −154 | 0.78 | 0.39 | 0.41 | 7.65 | 2.29 |
| **15:30** | 861 | 27.18 | **4.54** | 8.40 | **4.10** | 1.76 | 1.56 | −159 | 0.25 | −0.21 | 0.17 | 5.35 | 2.89 |

The oracle is positive at every clock at both fills. Its midpoint Sharpe rises
from 3.06–3.43 in the morning to 4.55–4.72 in the last hour, and the crossing
costs it 0.23 to 0.45 of Sharpe — between 6% and 12% of the midpoint number,
and never enough to threaten the sign. **The ceiling survives the touch at every
entry clock**, which is the same conclusion proposal 10 reached for the
thirty-minute slice oracle at the re-picked frequency.

In points the shape is the other way round. The oracle earns 3.06 points a day
from a 10:00 entry and 1.76 from 15:30, because the morning premium is 19.8
points and the close's is 5.4. The Sharpe rises through the day while the points
fall: the late entries are a smaller, cleaner bet. The maximum drawdown says the
same — 481 points from 10:00 against 154 from 15:00.

Always short held to settlement, on the same rows, is between −0.38 and +0.78 at
the midpoint and between −0.66 and +0.39 crossed, positive only at 14:30 and
15:00. The rich remainder measured in section 1 does not pay a short who holds
it: the straddle pays the terminal displacement, and the fat right tail eats the
many small wins. **What the oracle is worth is the sign, not the short.**

The two sides scored alone, flat on the other days (`12_legs_by_clock.csv`):

| clock | long days | short days | long leg Sharpe mid | crossed | short leg Sharpe mid | crossed |
|---|---|---|---|---|---|---|
| 10:00 | 268 | 596 | 2.33 | 2.18 | 2.30 | 1.88 |
| 11:00 | 265 | 600 | 2.29 | 2.20 | 2.55 | 2.31 |
| 12:00 | 276 | 590 | 1.96 | 1.86 | 2.74 | 2.46 |
| 13:00 | 265 | 601 | 2.54 | 2.45 | 3.04 | 2.75 |
| 14:00 | 248 | 616 | 2.73 | 2.60 | 3.87 | 3.38 |
| 15:00 | 214 | 648 | 2.50 | 2.39 | 4.19 | 3.74 |
| 15:30 | 234 | 627 | 2.69 | 2.54 | 3.76 | 3.23 |

Knowing the sign buys about the same thing on the long side at every clock —
1.88 to 2.73 at the midpoint, no trend — and an increasing amount on the short
side, 2.30 at 10:00 rising to 4.19 at 15:00. The whole of the clock profile in
the headline table is the short leg. That is consistent with section 2: the
short call is the sticky, well-populated one (590–648 days of 861–866), and it
gets sharper as the horizon shortens.

## 5. Conditional structure

Descriptive only. Terciles are cut on the full sample, no flag was chosen from a
search, and nothing here is a filter. Tables `12_conditional_*.csv`.

Columns are pooled over the twelve clocks unless a clock is named.

**By day of week.**

| day | days | median ratio | pct short pooled | pct short 10:00 | pct short 15:00 | pct short 15:30 |
|---|---|---|---|---|---|---|
| Monday | 199 | 0.774 | 73.49 | 78.89 | 74.62 | 69.54 |
| Tuesday | 120 | 0.863 | 67.02 | 67.23 | 71.67 | 63.33 |
| Wednesday | 225 | 0.804 | 69.49 | 64.89 | 70.98 | 75.89 |
| Thursday | 109 | 0.856 | 64.68 | 62.39 | 69.73 | 66.97 |
| Friday | 213 | 0.744 | 76.89 | 68.40 | 84.91 | 81.04 |

Friday and Monday are the rich ends of the week; Friday's afternoon is the
richest cell in the table (84.9% short at 15:00). The day counts are uneven:
Tuesday and Thursday carry 120 and 109 sessions against 199 to 225 for the other
three.

**By calendar flag** (FOMC statement days first, then month ends, else other;
`asl.fomc_and_monthend`, no date beyond the release file's horizon).

| flag | days | median ratio | pct short pooled | 10:00 | 12:00 | 15:00 | 15:30 |
|---|---|---|---|---|---|---|---|
| FOMC | 33 | 0.743 | 84.62 | 87.88 | 90.91 | 75.00 | 84.38 |
| month end | 51 | 0.863 | 61.28 | 60.78 | 60.78 | 60.78 | 62.75 |
| other | 782 | 0.799 | 71.37 | 68.72 | 67.65 | 76.12 | 73.01 |

FOMC days are the one strong conditional cell: 84.6% short pooled on 33 days,
with the morning richer than the afternoon (87.9% at 10:00, 90.9% at noon,
75.0% at 15:00) — the mirror image of the unconditional profile. Month ends run
the other way, 61.3% against 71.4%. Thirty-three days is thirty-three days.

**By VIX tercile**, the VIX at the 10:00 stamp (bar-end labelled, so the
09:30–10:00 bar), 812 of 866 days priced; tercile edges 12.13 / 17.97 / 23.68 /
72.22.

| tercile | days | median ratio | pct short pooled | 10:00 | 15:00 | 15:30 |
|---|---|---|---|---|---|---|
| low VIX | 271 | 0.872 | 63.28 | 63.47 | 64.58 | 63.10 |
| mid VIX | 270 | 0.761 | 77.04 | 74.07 | 83.33 | 76.30 |
| high VIX | 271 | 0.776 | 75.40 | 69.89 | 80.90 | 81.20 |
| no VIX | 54 | 0.863 | 62.04 | 66.67 | 59.26 | 62.96 |

Not monotone: the middle tercile is the richest, the low tercile clearly the
least rich. A quiet index is priced closer to what it delivers.

**By year.**

| year | days | median ratio | pct short pooled | 10:00 | 15:00 | 15:30 |
|---|---|---|---|---|---|---|
| 2020 | 158 | 0.764 | 71.54 | 71.34 | 73.38 | 73.86 |
| 2021 | 158 | 0.746 | 76.11 | 70.25 | 80.38 | 77.85 |
| 2022 | 219 | 0.776 | 78.74 | 71.56 | 84.93 | 85.85 |
| 2023 | 248 | 0.858 | 65.52 | 66.94 | 69.76 | 61.29 |
| 2024 | 83 | 0.895 | 59.04 | 61.45 | 59.04 | 61.45 |

The richness decays through the sample: median ratio 0.746 in 2021 and 0.895 in
2024, short rate 78.7% in 2022 and 59.0% in the four months of 2024. The daily
0DTE listing arrives in the middle of that decline, and this table cannot
separate the two.

**The morning against the afternoon.** Morning realized is the four bars
10:00–12:00 (2.0 hours); its price at 10:00 is `iv_hourly(10:00)^2 x 2.0`. Both
are known at 12:00. The oracle they condition is not. Pooled columns cover the
afternoon clocks 12:00–15:30 only. Table `12_morning_conditional.csv`.

| conditioning | tercile | days | median of the variable | median ratio | pct short pooled | 12:00 | 13:00 | 14:00 | 15:00 | 15:30 |
|---|---|---|---|---|---|---|---|---|---|---|
| morning realized | low | 289 | 6.46e-06 | 0.780 | 69.76 | 70.24 | 75.09 | 68.17 | 68.51 | 66.78 |
| morning realized | mid | 288 | 1.58e-05 | 0.801 | 69.44 | 65.62 | 63.54 | 71.53 | 75.00 | 71.53 |
| morning realized | high | 289 | 4.54e-05 | 0.802 | 74.90 | 68.51 | 69.55 | 74.22 | 82.11 | 80.28 |
| morning realized / morning implied | low | 288 | 0.672 | 0.706 | 78.82 | 76.74 | 76.74 | 79.44 | 80.77 | 80.42 |
| morning realized / morning implied | mid | 288 | 1.008 | 0.820 | 69.11 | 67.01 | 68.06 | 67.60 | 74.48 | 68.42 |
| morning realized / morning implied | high | 288 | 1.542 | 0.859 | 66.18 | 60.76 | 63.54 | 67.01 | 69.44 | 69.44 |

(Two days carry no vendor implied at 10:00 and so no morning price; they sit in
a seventh row of `12_morning_conditional.csv`.)

The raw level of the morning's realized variance says almost nothing: 69.8%,
69.4%, 74.9% short across the terciles, and the median afternoon ratio moves
from 0.780 to 0.802 the *wrong* way for a "quiet morning, rich afternoon" story.
Scaled by its own price it says something: a morning that realized 0.67 of what
it was priced at is followed by an afternoon remainder that is short of its price
on **78.8%** of clocks with a median ratio of 0.706, against **66.2%** and 0.859
after a morning that realized 1.54 times its price. The effect is present at
every afternoon clock (80.8% against 69.4% at 15:00).

This is descriptive and it is not a signal. The morning ratio and the afternoon
ratio share the day's implied volatility in their denominators, so a day on which
the market's implied is simply too high scores low on both by construction. The
honest statement is that the *pricing error* is persistent within the day, which
is the same fact section 2 measured as sign persistence, seen through a
conditioning variable that happens to be measurable at noon.

## 6. Where the held pair sits at the close

This is why section 3 reads the way it does. Distances are from the official
close to the centre of the held pair; one strike is the pair's own width. Table
`12_strike_offset.csv`.

| clock | median premium | median \|S_close − K\| | mean | p90 | median in strikes | > 1 strike | > 2 strikes | settles worthless |
|---|---|---|---|---|---|---|---|---|
| 10:00 | 19.83 | 18.56 | 25.37 | 54.48 | 3.70 | 84.2% | 69.6% | 7.9% |
| 10:30 | 18.30 | 16.73 | 23.13 | 51.50 | 3.35 | 82.3% | 68.4% | 8.1% |
| 11:00 | 17.05 | 14.82 | 21.79 | 49.50 | 2.96 | 80.0% | 63.5% | 9.5% |
| 11:30 | 16.05 | 14.11 | 20.36 | 45.08 | 2.80 | 80.1% | 61.9% | 10.0% |
| 12:00 | 14.83 | 13.07 | 19.12 | 38.78 | 2.60 | 80.7% | 60.4% | 8.2% |
| 12:30 | 13.68 | 12.49 | 18.09 | 40.35 | 2.49 | 75.6% | 56.8% | 12.8% |
| 13:00 | 12.65 | 12.14 | 17.44 | 39.62 | 2.42 | 76.2% | 55.9% | 13.4% |
| 13:30 | 11.70 | 10.94 | 16.25 | 34.41 | 2.19 | 74.7% | 52.8% | 13.4% |
| 14:00 | 10.55 | 10.17 | 15.04 | 32.95 | 2.03 | 73.1% | 50.6% | 14.7% |
| 14:30 | 9.30 | 9.09 | 13.37 | 29.58 | 1.81 | 69.3% | 45.4% | 16.9% |
| 15:00 | 7.65 | 7.49 | 11.08 | 24.13 | 1.50 | 64.5% | 38.4% | 19.1% |
| 15:30 | 5.35 | 6.27 | 8.66 | 18.63 | 1.26 | 59.4% | 29.9% | 22.4% |

(The 21 same-strike packages are dropped from the strike columns only; 0 to 4 a
clock.)

A straddle picked at 10:00 sits 3.7 strikes from the close at the median and more
than two strikes away on 69.6% of days. It is not an at-the-money package by
settlement and the remaining-session variance is not what decides its payoff. At
15:30 the same measurement is 1.26 strikes and 29.9%, and 22.4% of packages
settle worthless — the short's cleanest outcome, and three times as frequent as
from a 10:00 entry. The monotone decline down this table is the monotone rise of
the last column of section 3.

## Summary

The remaining-session oracle is a short tilt, not a short. It calls short on
68.1% to 75.2% of days at every clock, and the remainder comes in at a median of
0.835 of its price from a 10:00 entry and 0.757 from 15:30: the richness is
there all day and builds only mildly into the close, under a distribution wide
enough (tenth percentile 0.46–0.53, ninetieth 1.29–1.39) that the tilt is never
a rule. The sign is largely settled early, and asymmetrically so — it never
flips across the twelve clocks on 44.1% of days, one flip on a further 19.4%,
and from short it stays short on 92.0% of consecutive pairs against 79.6% from
long, so a 10:00 short is still short at 15:30 on 78.4% of days while a 10:00
long survives on 39.9%. The money is late and on the short side: held to
settlement the oracle scores 3.31 at 10:00 and 4.72 at 15:00 at the midpoint,
2.90 and 4.35 at the crossed spread — the ceiling survives the touch at every
clock, at a cost of 6% to 12% of its Sharpe — against always short's −0.27 and
+0.78, and its short leg alone climbs from 2.30 at 10:00 to 4.19 at 15:00 while
its long leg sits flat near 2 to 2.7 all day; in points the ranking inverts,
3.06 a day from 10:00 against 1.76 from 15:30, because the morning sells 3.7
times the premium at three times the drawdown. And the held straddle stops being
a variance instrument in the *morning*, not the afternoon: it pays the terminal
displacement from strikes picked hours earlier — 3.70 strikes away at the median
from 10:00 against 1.26 from 15:30, worthless on 7.9% of days against 22.4% —
and it delivers 31% of the Sharpe a payoff proportional to the day's realized
volatility would deliver from a 10:00 entry, 56% from 13:00, and more than all
of it from 15:00 onward. Where the index happens to land carries 92% to 95% of
the held P&L's variance at every clock; what the clock changes is how much of
the variance signal survives it.
