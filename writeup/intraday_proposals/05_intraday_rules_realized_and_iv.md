# 05 - Other intraday rules: realized persistence, implied-variance changes, forecast revisions, remaining-session richness

Read-only study of the intraday trade (nearest-OTM package re-picked every 30-minute
bar from 10:00, one-bar hold to the next midpoint, last bar cash-settled; 866 days,
10,387 bars, block-diagonal ridge forecast). Scripts, logs and CSVs:
`scratchpad/intraday_rules_a/{study.py, study.log, results.csv, per_clock_sharpe.csv,
hold_to_close.csv, e1_placebo.py, e1_placebo.log}`. Nothing touches the notebook.

## Idea

The diagnosis (00) found that the per-bar implied slice is not a price, so the
forecast-versus-slice sign has no edge before the close - but that the 30-minute
marks do respond to realized variance (a peeking oracle earns 4-4.7 per clock).
This study asks whether any *other* causal rule - one that reads information
available at the bar other than the forecast-versus-slice sign - is successful in
the intraday regime. Every rule below uses only quantities known at the bar's entry;
first-bar-of-day positions are zero where a rule needs a completed bar.

Gate: the rebuilt frame reproduces the notebook's daily-sum Sharpes exactly
(always short 1.9020, sign(s) 1.9160).

## Exact constructions

Per bar $t$ of a day: $R_t$ the one-bar per-premium return; $rv_t$ the bar's realized
variance (known only after the bar); $\widehat{rv}_t$ the forecast for the bar,
issued at its start; $\mathrm{IV}^2_t$ the package's implied variance (hourly units);
$h_t$ the hours remaining; $w_t$ the causal window share; the bar's slice
$\ell_t = \mathrm{IV}^2_t\,h_t\,w_t$; the remaining-session implied
$\mathrm{IV}^{\mathrm{rem}}_t = \mathrm{IV}^2_t\,h_t$ and the causal expectation of
remaining realized variance $E^{\mathrm{rem}}_t = \widehat{rv}_t / w_t$.

- (a1) realized persistence, one bar: $q_t = \mathrm{sign}(rv_{t-1} - \ell_{t-1})$.
- (a2) two bars: $q_t = \mathrm{sign}\big((rv_{t-1}+rv_{t-2}) - (\ell_{t-1}+\ell_{t-2})\big)$.
- (a3) expanding within the day: $q_t = \mathrm{sign}\big(\sum_{u<t} rv_u - \sum_{u<t} \ell_u\big)$.
- (b1) implied-variance change, mean reversion: $q_t = -\mathrm{sign}(\mathrm{IV}^2_t - \mathrm{IV}^2_{t-1})$; (b2) momentum: the opposite sign.
- (c) forecast revision: $q_t = \mathrm{sign}(\widehat{rv}_t - \widehat{rv}_{t-1})$.
- (d) last bar's forecast error: $q_t = \mathrm{sign}(rv_{t-1} - \widehat{rv}_{t-1})$.
- (e1) short only when the remaining session is rich: $q_t = -1$ if $\mathrm{IV}^{\mathrm{rem}}_t > E^{\mathrm{rem}}_t$, else $0$.
- (e2) sign of the remaining-session comparison: $q_t = \mathrm{sign}(E^{\mathrm{rem}}_t - \mathrm{IV}^{\mathrm{rem}}_t)$.

Daily series = sum of $q_t R_t$ over the day's bars. Sharpe is the daily-sum Sharpe
annualized by $\sqrt{252}$. Paired tests against always short and against sign(s):
Newey-West with lag $\lfloor 1.5\,n^{1/3}\rfloor$ on the daily difference, and a
block bootstrap ($B=2000$, block $\lceil n^{1/3}\rceil$, seed 0) for the Sharpe
difference. Placebo for the signed rules: 2,000 random $\pm1$ sequences with the
rule's buy share on the rule's active bars. Placebo for (e1), which only chooses
*when* to be short: 2,000 random subsets of the same number of bars, plain and
with the same per-clock counts.

## Results (daily sums, block-diagonal ridge, 866 days)

| rule | active bars | buy share | Sharpe | mean/day | worst day | maxDD | dSharpe vs AS [95% CI] | NW t vs AS | dSharpe vs sign(s) [95% CI] | placebo pct |
|---|---|---|---|---|---|---|---|---|---|---|
| always short | 10387 | 0.00 | 1.90 | +0.160 | -11.4 | -16.8 | - | - | -0.01 [-1.30, +1.68] | - |
| sign(s) | 9631 | 0.56 | 1.92 | +0.153 | -6.3 | -14.6 | +0.01 [-1.74, +1.38] | -0.12 | - | 100 |
| (a1) last bar realized vs slice | 8828 | 0.40 | 1.67 | +0.132 | -5.1 | -24.2 | -0.23 [-1.91, +0.97] | -0.50 | -0.24 [-1.68, +0.86] | 99.5 |
| (a2) last two bars | 8025 | 0.42 | 1.52 | +0.119 | -6.1 | -32.0 | -0.38 [-2.06, +0.78] | -0.73 | -0.40 [-1.72, +0.71] | 99.0 |
| (a3) today so far | 8828 | 0.51 | -0.08 | -0.007 | -6.5 | -68.3 | -1.98 [-3.90, -0.64] | -2.78 | -2.00 [-3.35, -0.71] | 46 |
| (b1) IV change, mean reversion | 9521 | 0.45 | 0.93 | +0.078 | -11.6 | -35.2 | -0.97 [-2.18, +0.07] | -1.95 | -0.99 [-2.23, +0.47] | 91 |
| (b2) IV change, momentum | 9521 | 0.55 | -0.93 | -0.078 | -4.7 | -107.9 | -2.83 [-5.03, -1.46] | -3.29 | -2.85 [-4.58, -1.43] | 11 |
| (c) forecast revision | 9521 | 0.44 | 0.46 | +0.038 | -4.7 | -33.3 | -1.44 [-3.58, +0.04] | -1.68 | -1.45 [-2.99, -0.31] | 69 |
| (d) last bar forecast error | 9521 | 0.33 | 1.44 | +0.120 | -7.1 | -25.2 | -0.47 [-2.04, +0.79] | -0.69 | -0.48 [-2.02, +0.91] | 94 |
| **(e1) short only when remaining implied is rich** | **4283** | 0.00 | **2.69** | +0.151 | **-6.0** | **-8.2** | **+0.79 [-0.30, +1.84]** | -0.29 | **+0.78 [+0.14, +1.63]** | see below |
| (e2) sign(remaining forecast - remaining implied) | 9631 | 0.56 | 1.92 | +0.153 | -6.3 | -14.6 | +0.01 | -0.12 | 0.00 (identical) | 100 |

Per-clock Sharpe (daily series of each clock's bar):

| rule | 10:00 | 10:30 | 11:00 | 11:30 | 12:00 | 12:30 | 13:00 | 13:30 | 14:00 | 14:30 | 15:00 | 15:30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| always short | 0.26 | 0.64 | 0.46 | 2.19 | 1.12 | 1.78 | 0.23 | 0.79 | 0.50 | 1.00 | 2.51 | 0.20 |
| sign(s) | 0.14 | 0.10 | -0.03 | 1.01 | 0.82 | -0.01 | 0.79 | 0.63 | -0.27 | 0.20 | 0.08 | 1.82 |
| (a1) | 0.00 | -0.04 | 0.49 | 1.55 | 0.44 | 0.48 | 0.78 | -0.42 | 0.45 | 0.31 | 1.37 | 0.92 |
| (d) | 0.00 | -0.29 | 0.54 | 1.42 | 0.99 | 1.28 | 0.94 | -0.19 | 0.12 | 0.39 | 1.25 | 0.63 |
| (e1) | 0.08 | 0.40 | 0.44 | 2.43 | 1.38 | 1.05 | 1.24 | 1.06 | 0.12 | 0.74 | 1.81 | 1.54 |

Hold-to-close ladder (every bar's position held to the cash settlement, daily sum):
always short 0.06, sign(s) 1.00, (a1) 0.47, (a2) 0.50, (a3) 0.06, (b1) 0.31,
(b2) -0.31, (c) 0.31, (d) 0.29, (e1) 0.54, (e2) 1.00 - no rule improves under a
hold-to-close exit; the twelve overlapping settlement positions per day dominate.

## Two findings

**(e2) is sign(s).** $E^{\mathrm{rem}}_t - \mathrm{IV}^{\mathrm{rem}}_t =
(\widehat{rv}_t - \mathrm{IV}^2_t h_t w_t)/w_t = s^{\mathrm{matched}}_t / w_t$ with
$w_t > 0$, so the remaining-session comparison has the same sign as the matched
signal on every bar: identical positions (9,631 active bars, Sharpe 1.916). The
diagnosis' statement that the per-bar slice is not a price is the same statement as:
the remaining-session forecast-versus-implied comparison, applied to a one-bar hold,
has no edge before the close. A remaining-session study can only differ from
sign(s) through the exit (hold to close), not through the signal.

**(e1) halves the risk of always short at the same return.** Being short only on
the 41% of bars where the remaining-session implied exceeds the causal expectation
of remaining realized variance keeps the mean (0.151 vs 0.160 per day) and cuts the
daily standard deviation from 1.34 to 0.89: Sharpe 2.69 vs 1.90, worst day -6.0 vs
-11.4, maxDD -8.2 vs -16.8. It beats random subsets of the same size (99.9th
percentile, placebo median 1.24) and random subsets with the same per-clock counts
(100th percentile, median 1.21), so the selection is not a clock effect. By year:
2020 +4.19 vs +1.64, 2021 +2.82 vs +1.33, 2022 +2.25 vs +2.95, 2023 +3.43 vs +2.81,
2024 +0.03 vs -0.73. The bootstrap interval for the Sharpe difference against always
short includes zero ([-0.30, +1.84]); against sign(s) it does not ([+0.14, +1.63]).
The gain is variance, not mean: the Newey-West $t$ on the mean difference is -0.29.

## Gates

- Reproduction: always short 1.9020, sign(s) 1.9160 (exact).
- Paired tests and bootstrap intervals as tabulated.
- Placebos: signed rules against random signs with the same buy share; (e1) against
  random same-size and same-clock-mix subsets (99.9th / 100th percentile).
- Causality: all quantities after four cut dates perturbed by 50% noise; positions
  before each cut identical for all nine rules (PASS).

## Verdicts

- (a1)-(a3) realized persistence: **reject** - (a1) beats random signs but not always short; (a3) loses outright.
- (b1)/(b2) implied-variance change: **reject** - neither direction beats always short; momentum loses heavily.
- (c) forecast revision: **reject** - 0.46, below both benchmarks.
- (d) forecast-error sign: **reject** - 1.44, placebo 94th, below both benchmarks.
- (e2) remaining-session sign: **not a new rule** - identical to sign(s) by construction.
- (e1) short only when the remaining session is rich: **needs more, as a risk filter, not a return claim** - same mean as always short with half the variance and half the drawdown, robust to clock-mix placebos and positive in every year, but the Sharpe gain over always short is not resolved at 95% and the mean is unchanged.

## Open questions

- Costs: (e1) trades 41% of bars (about 9 crossings a day instead of 21-23). With
  the costs fork's median half-spread of 1.7% of premium and a mean of 0.15 per day,
  it is unlikely to survive crossed fills either; to be checked with that fork's
  cost model before any adoption.
- Composition with the clock-selection result: (e1) before the close plus sign(s)
  at 15:30 is the natural composite; not computed here (it would be an in-sample
  pairing of two adopted pieces and needs its own placebo).
- Whether the (e1) condition is mostly "implied above forecast" (a VRP filter) or
  carries timing beyond that: compare with sitting out on the bars where the
  package's implied variance is below its own trailing per-clock median.
