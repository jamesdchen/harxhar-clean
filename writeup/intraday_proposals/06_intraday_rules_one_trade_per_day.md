# 06. Intraday rules that survive costs: one trade per day, held to the close

Read-only study on the intraday cached rows (`results/atm_straddle_intraday/cache/book_*.parquet`,
block-diagonal ridge forecast); scripts, log and CSV in the session scratchpad `intraday_rules_b/`
(`study.py`, `addendum.py`, `results.csv`, `study.log`). Nothing is wired into any notebook.

## Idea

The twelve-bar re-pick loses at the crossed spread because it crosses 21–23 times a day (file 03).
The family that can survive costs opens **one** position per day and holds it to the cash-settled
close: one crossing at entry, none at exit. The question is whether any member of that family
— entering earlier than 15:30, entering on the day's first signal, selling the whole session's
premium from the open, comparing the forecast with a remaining-session price, or skipping
scheduled event days — beats the close trade (sign(s) at 15:30), which is the family's member
already in the deck.

## Exact construction

- Days: the 864 days on which all twelve half-hour bars 10:00–15:30 are present (866 in the
  notebook; two lack a bar). Package at each bar: the nearest out-of-the-money call and put
  picked at that bar, midpoint entry $P_c$; exit is the cash settlement of that package at the
  official close, $X_c$; hold-to-close return $R^{hold}_c = X_c/P_c - 1$.
- Bar signal at clock $c$: $s^{bar}_c = \widehat{RV}_c - \mathrm{IV}^2_{hr,c}\, h_{rem}\, w_c$, exactly the
  notebook's window-matched signal ($w_c$ the causal per-clock share of remaining realized variance,
  expanding mean lagged one day, minimum 63 days; $h_{rem}$ the hours to the close).
- Remaining-session signal: $s^{rem}_c = \widehat{RV}_c / w_c - \mathrm{IV}^2_{hr,c}\, h_{rem}$ — the
  bar forecast scaled to the whole remainder against the remainder's implied variance. At 15:30 it
  is the close signal.
- Position: sign of the signal (flat if the signal does not yet exist, i.e. the 63-day warm-up), or
  $-1$ for always short. Daily return $R' = q\,R^{hold}_c$. Statistics: mean, annualized Sharpe
  ($\times\sqrt{252}$), $t$ with Newey–West lags $\lfloor 1.5\,n^{1/3}\rfloor$, worst day,
  maximum drawdown of the running sum, buy share. Paired test against the close trade (sign(s) at
  15:30): $t$ of the daily difference and a block-bootstrap 95% interval for the Sharpe difference
  (B = 2000, block $\lceil n^{1/3}\rceil$, seed 0). Placebo for the clock ladders: 2000 draws of a
  random entry clock per day. Crossed spread: the entry leg pays the ask (long) or receives the bid
  (short) from the chain cache; settlement is cash.
- Reference on these 864 days: sign(s) at 15:30 — mean 0.123, Sharpe 1.83, $t$ 3.6, worst −5.4,
  maxDD −12.5, buy share 34% (7% of days flat in the warm-up); crossed 1.41.

## Results

### (a) Entry-clock ladder, one trade per day, held to the close

| entry | sign(s): mean | Sharpe | t | worst | maxDD | buy % | crossed | Δmean vs 15:30 (t) | ΔSharpe 95% |
|---|---|---|---|---|---|---|---|---|---|
| 10:00 | 0.011 | 0.19 | 0.4 | −4.3 | −24.6 | 56 | −0.16 | −0.112 (−2.8) | [−2.91, −0.46] |
| 10:30 | 0.030 | 0.50 | 1.0 | −7.8 | −28.5 | 55 | 0.28 | −0.093 (−2.2) | [−2.67, +0.01] |
| 11:00 | 0.030 | 0.47 | 0.9 | −8.7 | −21.1 | 56 | 0.26 | −0.093 (−2.0) | [−2.81, +0.17] |
| 11:30 | 0.021 | 0.32 | 0.6 | −9.4 | −32.9 | 47 | 0.11 | −0.102 (−2.1) | [−3.01, −0.17] |
| 12:00 | 0.045 | 0.70 | 1.4 | −10.8 | −19.8 | 57 | 0.48 | −0.078 (−1.7) | [−2.54, +0.21] |
| 12:30 | 0.035 | 0.50 | 1.0 | −10.7 | −29.3 | 52 | 0.29 | −0.088 (−1.8) | [−2.89, +0.12] |
| 13:00 | 0.023 | 0.32 | 0.6 | −11.8 | −19.4 | 62 | 0.09 | −0.101 (−2.1) | [−2.91, −0.20] |
| 13:30 | 0.008 | 0.11 | 0.2 | −11.3 | −22.3 | 43 | −0.13 | −0.115 (−2.6) | [−3.00, −0.55] |
| 14:00 | −0.003 | −0.04 | −0.1 | −7.6 | −27.7 | 62 | −0.40 | −0.126 (−2.7) | [−3.30, −0.54] |
| 14:30 | 0.018 | 0.29 | 0.6 | −9.5 | −28.9 | 48 | −0.02 | −0.105 (−2.5) | [−2.97, −0.34] |
| 15:00 | 0.083 | 1.25 | 2.2 | −5.2 | −22.0 | 46 | 0.93 | −0.040 (−1.1) | [−1.81, +0.45] |
| **15:30** | **0.123** | **1.83** | **3.6** | −5.4 | −12.5 | 34 | **1.41** | — | — |

| entry | always short: mean | Sharpe | t | worst | maxDD | crossed | Δmean vs 15:30 sign(s) (t) |
|---|---|---|---|---|---|---|---|
| 10:00 | −0.016 | −0.27 | −0.6 | −6.8 | −35.1 | −0.66 | −0.139 (−3.0) |
| 11:00 | −0.012 | −0.18 | −0.4 | −8.7 | −28.0 | −0.41 | −0.135 (−2.7) |
| 12:00 | 0.003 | 0.05 | 0.1 | −10.8 | −20.6 | −0.20 | −0.120 (−2.5) |
| 13:00 | −0.030 | −0.41 | −0.9 | −11.8 | −38.5 | −0.66 | −0.153 (−3.1) |
| 14:00 | 0.001 | 0.01 | 0.0 | −11.3 | −28.0 | −0.38 | −0.122 (−2.5) |
| 14:30 | 0.033 | 0.51 | 1.1 | −9.5 | −30.3 | 0.16 | −0.090 (−1.9) |
| 15:00 | 0.049 | 0.72 | 1.4 | −9.9 | −26.7 | 0.32 | −0.074 (−1.4) |
| 15:30 | 0.014 | 0.19 | 0.4 | −10.3 | −19.3 | −0.29 | −0.109 (−2.3) |

(Full twelve-clock table for both rules in `results.csv`.)

Placebo — a random entry clock each day: sign(s) median Sharpe 0.53, 95th percentile 1.30; the
15:30 entry is at the 99.8th percentile and is also the best clock. Always short: median 0.05, 95th
percentile 0.64; the 15:30 entry is at the 66th percentile, the best clock (15:00, 0.72) at the
97th — an in-sample pick of one clock out of twelve, not a finding.

### (b) The remaining-session signal

Identical to the bar signal on every bar: $s^{rem}_c = s^{bar}_c / w_c$ with $w_c > 0$, so the two
have the same sign on 100.00% of bars and every row of the sign($s^{rem}$) ladder equals the
sign($s^{bar}$) row above. Scaling a one-bar forecast to the remainder by the same profile share
that allocates the implied variance to the bar cannot change which side of the price the
forecast lands on. A remaining-session comparison that could differ needs a remaining-session
forecast that is not a rescaling of the bar forecast (a multi-step forecast of each remaining
bar); the intraday cache carries none.

### (c) First-fire entries (enter at the first bar whose signal has the required sign)

| rule | fires on | median entry | mean | Sharpe | t | worst | maxDD | crossed | Δmean vs 15:30 (t) |
|---|---|---|---|---|---|---|---|---|---|
| first sell | 86% of days | 10:30 (42% at 10:00) | −0.023 | −0.39 | −0.9 | −7.8 | −27.7 | −0.66 | −0.146 (−3.4) |
| first buy | 90% of days | 10:00 (63% at 10:00) | 0.019 | 0.32 | 0.7 | −1.0 | −18.3 | 0.02 | −0.105 (−2.8) |

Same under $s^{rem}$ (identity above). The first signal of the day arrives in the first hour on
most days, i.e. these rules are early-entry rules in disguise, and early entries have no edge.

### (d) Selling the whole session from the open vs selling the last half hour

| | premium (median, pts) | points/day short | t |
|---|---|---|---|
| sell at 10:00, hold to close | 19.8 | −0.079 | −0.1 |
| sell at 15:30, hold to close | 5.4 | +0.016 | +0.0 |

The full-session premium sold at the open is not positive on this sample (Sharpe −0.27,
maxDD −35 premium units); always short is not positive at any entry before 14:30 and only
marginal after. Whatever variance premium the close trade collects is a last-half-hour
phenomenon, and it is the sign, not the short, that pays there (always short at 15:30 is 0.19,
sign(s) 1.83).

### (e) Causal calendar: flat on FOMC statement days and month-ends (83 of 864 days)

| rule at 15:30 | Sharpe unfiltered → filtered | t (filtered) | worst | maxDD | crossed | Δmean vs unfiltered (t) | ΔSharpe 95% vs close trade | placebo pct |
|---|---|---|---|---|---|---|---|---|
| sign(s) | 1.83 → **2.30** | 4.4 | −4.6 | −16.3 | 1.92 | +0.022 (+1.7) | [+0.02, +0.87] | 99.9 |
| always short | 0.19 → 0.69 | 1.4 | −10.3 | −17.4 | 0.22 | +0.032 (+2.7) | [−2.38, +0.36] | 99.6 |

Event days lose for both rules: sign(s) averages −0.225 on event days against +0.160 on the rest
(FOMC −0.144, month-end −0.253). The placebo removes 83 random days; the realized filter beats
99.9% of such draws. By era: pre-2022-05-16 1.62 → 1.82 (paired t 0.7, event-day mean −0.08);
daily-0DTE era 1.97 → 2.66 (paired t 1.6, event-day mean −0.40); by year 2020 3.47 → 3.69,
2021 1.01 → 0.89, 2022 0.36 → 1.08, 2023 2.30 → 3.30, 2024 3.14 → 3.66. The improvement comes
from 2022 onward.

## Gates

- Reproduction: the 15:30 sign(s) row reproduces the intraday notebook's close leg (Sharpe 1.83 on
  these days; the notebook's rule table is reproduced to 1e-6 by the shared frame builder).
- Causality: with realized variance on all days after ten random cut dates perturbed, the bar and
  remaining-session signals on and before each cut date are unchanged (rtol 1e-12) — PASS.
- Placebo: entry clocks (2000 random-clock draws) and event days (2000 random 83-day removals),
  reported above.
- No magic numbers: the only constants are the deck's standing 63-day minimum and the scheduled
  calendar; no thresholds on signal size.
- Pre-registration caveat on (e): the FOMC/month-end flags were first identified by tail forensics
  on an earlier version of the close trade (the parked event filter in the paper), so this is a
  confirmation on the same sample, not an out-of-sample test. The era split is the only
  independent check offered here.

## Verdicts

- (a) Entering before the last half hour, one trade per day: **reject** — every earlier clock is
  worse than the close trade for both rules, most of them significantly; the close is the 99.8th
  percentile of random entry clocks.
- (b) Remaining-session signal: **not a distinct rule** — identical in sign to the bar signal under
  the share-based construction; needs a genuine multi-step forecast to be testable.
- (c) First-fire entries: **reject** — they are early entries.
- (d) Selling the session from the open: **reject** — the full-session premium is not positive; the
  edge is the close and it is the sign, not the short.
- (e) Flat on FOMC and month-end days at the close: **needs more** — 1.83 → 2.30 with a Sharpe
  interval above zero and a 99.9th-percentile placebo, but the paired mean gain is at t 1.7 and
  the flags were found in-sample on an earlier trade; adopt only after an out-of-sample era or a
  pre-registered forward test.

## Open questions

- A remaining-session forecast that is not a rescaling: the paper's multi-horizon pipeline
  (direct forecasts of each remaining bar) would give $s^{rem}$ a sign of its own; is it available
  at the intraday clocks for these days?
- Why event days lose for the close trade (both rules): is it the settlement move or the 15:30
  price? A dispersion reading (the parked paper paragraph) says the former.
- The always-short 15:00 entry (0.72, 97th percentile of the placebo) is the one early-clock
  number worth a pre-registered look, not a claim.
