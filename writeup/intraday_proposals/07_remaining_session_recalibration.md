# 07 — Recalibrating the remaining-session forecast against the realized remainder

Read-only study; nothing wired into a notebook. Scripts, log and CSVs:
`scratchpad/remaining_recal/{study.py, study.log, calibration_by_clock.csv,
per_clock_rules.csv, boot_ci.csv, era_split.csv, costs_by_clock.csv,
work_recal.parquet}`. Block-diagonal ridge forecast throughout. Frame: the 04
study's 803 days (2020-05-26 to 2024-04-30, the days on which the causal
window share exists; the 413 cache days after 2024-04-30 have no forecast and
are excluded), reduced to 740 days (from 2020-10-16) once each recalibration's
own 250-day trailing window has at least 63 rows.

## Idea

The audit (04a) found that the remaining-session implied variance is a real
price and is rich at every clock (median realized-remaining / implied 0.76-0.85;
a peeking oracle sign(RV_rem − IV_rem) earns 3-4.7 from any entry), while the
remaining-session forecast F_rem = rv_hat / w runs 15-50% above the realized
remainder at the median, so sign(F_rem − IV_rem) buys on 43-62% of days against
the oracle's 25-32%. The one-bar recalibration targets a mean; the sign rule
needs the median crossing. This study re-calibrates the multi-bar forecast
causally against the realized remainder and asks whether any clock before
15:00 becomes a significant, placebo-beating, cost-surviving trade.

## Exact construction

At bar t with h_rem hours to the close, w_t the causal per-clock share and
B_rem,t the causal sum of the profile over the remaining bars (the same
expanding, one-day-lagged per-clock means of realized variance that define w):

- normalized forecast x_t = sqrt(F_rem,t / B_rem,t) = sqrt(rv_hat_t / prof_t),
- normalized realized remainder y_t = sqrt(RV_rem,t / B_rem,t), with
  RV_rem,t the sum of the day's realized variance from t to 16:00 (known only
  at the close; every remaining bar must be present),
- fits on the trailing 250 trading days of rows strictly before day d, three
  ways: (i) **mean, per clock** — weighted least squares y = a + b x with the
  deck's weights 1/max(x, q10)², residual variance s², F̂ = (m² + s²)·B_rem;
  (ii) **median, per clock** — least-absolute-deviation (quantile 0.5) fit of
  the same line, F̂ = m²·B_rem (no variance term: the median of RV given x is
  the squared conditional median of y); (iii) **mean, pooled** — one weighted
  fit across all clocks with the clock's share w as an extra regressor;
- signal ŝ = F̂ − IV_rem with IV_rem = IV_hourly²·h_rem; rule sign(ŝ), one
  trade per day per entry clock, package held to the close, cash-settled; the
  un-recalibrated sign(s_rem) and always short on the same days as references.

Which is principled: (ii). A sign decision asks whether the realized remainder
will exceed the price, i.e. a median crossing; a mean map on a right-skewed
target sits above its median by construction, so (i) cannot remove the level
bias — and the table shows it does not. The deck's mean map is right for its
own purpose (a QLIKE-scored mean forecast), wrong for this one.

## Results

### 1. Calibration and buy shares (median forecast / realized remainder)

| clock | oracle buy | raw | mean per-clock | median per-clock | pooled |
|---|---|---|---|---|---|
| 10:00 | 0.32 | 1.31 · buy 0.62 | 1.21 · 0.56 | **1.03 · 0.32** | 1.22 · 0.57 |
| 11:00 | 0.31 | 1.29 · 0.59 | 1.23 · 0.54 | **0.99 · 0.24** | 1.23 · 0.57 |
| 12:00 | 0.32 | 1.34 · 0.61 | 1.31 · 0.61 | **0.98 · 0.23** | 1.30 · 0.61 |
| 13:00 | 0.30 | 1.53 · 0.71 | 1.43 · 0.68 | **1.06 · 0.31** | 1.47 · 0.69 |
| 14:00 | 0.28 | 1.48 · 0.67 | 1.45 · 0.65 | **1.03 · 0.32** | 1.45 · 0.66 |
| 15:00 | 0.25 | 1.24 · 0.47 | 1.05 · 0.30 | **0.95 · 0.21** | 1.24 · 0.49 |
| 15:30 | 0.28 | 1.15 · 0.38 | 1.09 · 0.34 | **0.98 · 0.22** | 1.15 · 0.42 |

The median map does what it is built to do: the forecast sits at the realized
remainder's median (0.95-1.08) at every clock and the buy share drops to the
oracle's rate. The mean maps leave the level 5-49% high and the buy share
almost unchanged.

### 2. Per entry clock, one trade per day, held to the close (740 days)

Sharpe · Newey-West t · buy share · hit rate against the oracle sign:

| clock | raw sign(s_rem) | mean per-clock | median per-clock | pooled | always short |
|---|---|---|---|---|---|
| 10:00 | 0.36 · 0.6 · 0.62 · 0.48 | 0.42 · 0.7 · 0.56 · 0.51 | 0.59 · 1.0 · 0.32 · 0.61 | 0.56 · 1.0 · 0.57 · 0.50 | −0.57 · −1.2 |
| 10:30 | 0.42 · 0.8 · 0.57 · 0.53 | 0.03 · 0.0 · 0.51 · 0.53 | 0.05 · 0.1 · 0.26 · 0.62 | 0.75 · 1.4 · 0.54 · 0.53 | −0.08 · −0.2 |
| 11:00 | 0.55 · 1.0 · 0.59 · 0.51 | 0.65 · 1.2 · 0.54 · 0.51 | 0.64 · 1.2 · 0.24 · 0.65 | 0.66 · 1.3 · 0.57 · 0.52 | −0.22 · −0.5 |
| 11:30 | 0.36 · 0.7 · 0.51 · 0.54 | 1.14 · 2.1 · 0.57 · 0.50 | 0.79 · 1.4 · 0.27 · 0.65 | 0.76 · 1.4 · 0.49 · 0.53 | −0.08 · −0.2 |
| 12:00 | 0.58 · 1.1 · 0.61 · 0.50 | 0.89 · 1.7 · 0.61 · 0.50 | −0.31 · −0.7 · 0.23 · 0.62 | 0.79 · 1.5 · 0.61 · 0.50 | −0.13 · −0.3 |
| 12:30 | 0.51 · 0.9 · 0.58 · 0.52 | 1.21 · 2.3 · 0.66 · 0.47 | 0.12 · 0.2 · 0.26 · 0.63 | 1.39 · 2.6 · 0.57 · 0.51 | −0.37 · −0.7 |
| 13:00 | 0.35 · 0.7 · 0.71 · 0.47 | 0.76 · 1.5 · 0.68 · 0.47 | −0.33 · −0.6 · 0.31 · 0.61 | 0.56 · 1.1 · 0.69 · 0.46 | −0.55 · −1.1 |
| 13:30 | 0.35 · 0.7 · 0.49 · 0.57 | 0.46 · 0.9 · 0.70 · 0.44 | 0.69 · 1.3 · 0.30 · 0.65 | 0.36 · 0.6 · 0.52 · 0.53 | −0.12 · −0.2 |
| 14:00 | −0.03 · −0.1 · 0.67 · 0.47 | 0.13 · 0.3 · 0.65 · 0.48 | 0.73 · 1.2 · 0.32 · 0.64 | −0.14 · −0.3 · 0.66 · 0.47 | 0.06 · 0.1 |
| 14:30 | 0.35 · 0.6 · 0.52 · 0.54 | 0.79 · 1.5 · 0.41 · 0.57 | 1.07 · 2.0 · 0.28 · 0.66 | 0.23 · 0.5 · 0.56 · 0.52 | 0.39 · 0.7 |
| 15:00 | 1.19 · 2.0 · 0.47 · 0.57 | 0.90 · 1.6 · 0.30 · 0.66 | 0.75 · 1.5 · 0.21 · 0.69 | 0.98 · 1.7 · 0.49 · 0.57 | 0.77 · 1.4 |
| 15:30 | 1.64 · 3.1 · 0.38 · 0.67 | 1.77 · 3.2 · 0.34 · 0.68 | 1.70 · 3.0 · 0.22 · 0.72 | 0.68 · 1.2 · 0.42 · 0.64 | 0.27 · 0.5 |

Worst days and drawdowns move with the buy share, not with the Sharpe: the
median map's worst days are the raw ones (it sells more), the mean maps' are
smaller (−4 to −6 vs −8 to −12) because they buy more.

### 3. Paired tests and the two placebos

Block-bootstrap 95% intervals for ΔSharpe against the raw signal / against
always short, and the entry-clock placebo (2,000 random entry-clock-per-day
series; the percentile of each clock's Sharpe):

| clock | mean per-clock: Δ vs raw · Δ vs AS · placebo pct | median per-clock: Δ vs raw · Δ vs AS · placebo pct |
|---|---|---|
| 11:30 | [−0.07, +1.87] · [−0.44, +2.76] · 79 | [−0.55, +1.52] · [−0.28, +1.97] · 70 |
| 12:30 | [−0.50, +1.98] · [−0.07, +3.08] · 83 | [−1.39, +0.67] · [−0.58, +1.65] · 19 |
| 14:30 | [−0.56, +1.61] · [−1.15, +1.83] · 50 | [−0.28, +1.91] · [−0.52, +1.73] · 87 |
| 15:30 | [−0.47, +0.74] · [−0.00, +2.72] · 99 | [−0.73, +0.88] · [+0.10, +2.54] · 99 |

No clock before 15:00 has an interval that excludes zero against the raw
signal; the single-clock placebo percentiles reach 83 (mean, 12:30) and 87
(median, 14:30); the pooled fit reaches 96 at 12:30. Family-wise — the best of
the ten clocks before 15:00 against the best of ten random entry-clock series
— the best recalibrated clock sits at the 16th (mean per-clock), 23rd (median)
and 64th (pooled) percentile: what looks like a good clock is what the best of
ten random series looks like.

### 4. Era split (Sharpe before 2022-05-16 · daily-0DTE era)

| clock | raw | mean per-clock | median per-clock | pooled | always short |
|---|---|---|---|---|---|
| 11:30 | 1.75 · −0.43 | 1.36 · 1.02 | 2.22 · −0.03 | 0.59 · 0.86 | −0.81 · 0.34 |
| 12:30 | 2.16 · −0.36 | 2.14 · 0.71 | 1.29 · −0.50 | 2.09 · 1.02 | −0.95 · −0.07 |
| 14:30 | 1.23 · −0.13 | 0.47 · 0.97 | 0.13 · 1.60 | 0.18 · 0.26 | −0.86 · 1.08 |
| 15:00 | 2.97 · 0.09 | 0.66 · 1.06 | 0.17 · 1.13 | 1.49 · 0.66 | −0.99 · 1.91 |
| 15:30 | 1.05 · 1.97 | 0.84 · 2.30 | 0.68 · 2.27 | 0.52 · 0.76 | −0.56 · 0.73 |

In the daily-0DTE era, where the close trade earns its significance, no
recalibrated clock before 15:00 exceeds 1.1 and most sit between −0.7 and 0.7.

### 5. Costs (one crossing per day) and the close

Crossed-spread Sharpe: mean per-clock 12:30 1.00, 11:30 0.94, all other clocks
before 15:00 at most 0.68 (median map at most 0.76; pooled at most 1.19 at
12:30); 15:30 raw 1.23, mean-recalibrated 1.37. At the close the per-clock
mean recalibration gives 1.77 against 1.64 raw (paired t 0.46, interval
[−0.47, +0.74]) with 93% sign agreement with the deck's signal — noise, and the
deck's own fit (pooled over the session's bars) stays the reference. The pooled
multi-clock fit breaks the close (0.68, t −2.34 against raw), which rules it out
as a construction whatever it does at 12:30.

## Gates

- Reconciliation: on the 740 common days the raw signal agrees with the deck's
  in sign on 100% of days and its Sharpe equals the deck's (1.637).
- Causality: perturbing the realized remainder on all days after ten random cut
  dates leaves every past fitted value unchanged (0 violations).
- Tests: Newey-West with lag floor(1.5·n^(1/3)); block bootstrap B=2000,
  block length ceil(n^(1/3)), rng(0); placebos per clock and family-wise.
- No clips or thresholds beyond the standing 250-day window and 63-row minimum.

## Verdict

Reject. The median recalibration fixes exactly what the audit diagnosed — the
forecast's level lands on the realized remainder and the rule buys at the
oracle's rate — and it changes nothing that matters: the hit rate against the
oracle stays at 61-72% everywhere, no clock before 15:00 beats the raw signal,
always short, or a random entry clock once the ten clocks are treated as one
family, and the daily-0DTE era has no intraday edge under any map. The
remaining-session miss is the forecast's ranking across days, not its level; a
rescaled one-bar forecast cannot supply it under any calibration.

## Open questions

- A genuinely multi-step forecast of the remainder (the main repository's
  direct per-bar horizon forecasts) is the only remaining input that could
  change the ranking; it is not in this repository.
- The pooled fit's collapse at the close (0.68) says the share regressor
  absorbs the close's own calibration; any pooled construction must exclude the
  final bar or model it separately.
