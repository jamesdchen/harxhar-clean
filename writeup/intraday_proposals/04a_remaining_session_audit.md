# 04a. Remaining-session trade: adversarial audit

Independent rebuild of the remaining-session study (forecast of the variance
remaining to the close against the remaining-session implied variance, package
picked at the entry clock, held to the close, cash-settled). Built from the
intraday cache and the forecast panel only; no notebook code reused beyond the
library's loader and recalibration. Block-diagonal ridge forecast throughout.
Scripts and logs: session scratchpad `remaining_session_audit/`
(`audit.py`, `extra.py`, `audit_log.txt`, `extra_log.txt`, `per_clock_htc.csv`,
`per_clock_costs.csv`).

Construction audited: at bar t, F_rem = rv_hat_t / w_t (rv_hat the recalibrated
one-bar forecast issued at t; w_t the entry bar's share of remaining-session
variance under the expanding per-clock mean of realized bar variance over prior
days only, minimum 63 days, lagged one day); IV_rem = IV_hourly,t^2 x h_rem;
s_rem = F_rem - IV_rem; R = settlement / entry - 1.

## 1. Reconciliation at 15:30 - exact

| quantity | max |intraday - close deck| | days |
|---|---|---|
| signal s_rem vs the deck's signal | 8.7e-12 (sign agreement 100%) | 803 (days with a share) |
| rv_hat | 0.0 | 866 |
| IV_rem vs the deck's iv_var | 1.5e-11 | 866 |
| hold-to-close return vs the deck's R | 0.0 | 866 |
| entry premium | 0.0 | 866 |

Frame: the intraday cache has 1,279 days at 15:30, the deck 871; 866 are common.
The 5 deck-only days are half sessions the intraday trade drops. The 413
intraday-only days lie after the forecast panel ends (2024-04-30) and carry no
signal; they must be excluded, not scored as anything. The share exists from
2020-05-26 (63-day warm-up), so every per-clock row below has 801-803 days.

## 2. Measurability - pass, with one construction choice to state

- Share w: perturbing realized variance on all later days and on the same day's
  bars at or after t leaves w_t unchanged, 10 of 10 cut points.
- rv_hat: perturbing every panel row after the fresh stamp t+30 (later days and
  later same-day bars) leaves rv_hat_t unchanged, 10 of 10 (the recalibration
  window ends before the bar).
- Alignment: the joined realized variance correlates with the bar's own squared
  30-minute log return at 0.63 against 0.60 for the one-bar-stale join, so the
  fresh row is the one attached.
- IV_rem uses the two picked legs' quotes at t (verified equal to their mean).
- Against the main repo's `ft_remaining.py`: the two share estimates differ by
  up to 0.18 because that profile expands from the panel's start in 2001 while
  the notebook's expands from the trade rows' start in 2020, and the main repo's
  numerator is the raw yhat^2 x baseline rather than the recalibrated rv_hat
  (median ratio 1.14). Both are measurable at t; the sign of s_rem agrees
  between them on 84% of rows. The builder must say which one it uses, because
  the per-clock results move with the choice (section 5, last table).

## 3. Units at every clock - exact

Pricing the picked package with Black-Scholes-Merton at total volatility
IV_hourly x sqrt(h_rem) reproduces the quoted midpoint at every clock: median
ratio 1.0000 from 10:00 to 15:30, 5th-95th percentiles within +-0.07% at 10:00
and +-0.6% at 15:30. IV_rem is the implied variance of the remaining window.

## 4. The implied side is rich at every clock; the forecast side is biased high

| clock | median RV_rem/IV_rem | ratio of means | P[RV_rem > IV_rem] | oracle Sharpe sign(RV_rem - IV_rem) | median F_rem/RV_rem | median F_rem/IV_rem | P[s_rem > 0] | hit vs oracle |
|---|---|---|---|---|---|---|---|---|
| 10:00 | 0.835 | 0.891 | 0.31 | 3.31 | 1.31 | 1.08 | 0.56 | 0.48 |
| 11:00 | 0.838 | 0.926 | 0.31 | 3.43 | 1.31 | 1.08 | 0.56 | 0.51 |
| 12:00 | 0.837 | 0.930 | 0.32 | 3.25 | 1.35 | 1.10 | 0.57 | 0.50 |
| 13:00 | 0.817 | 0.912 | 0.31 | 3.88 | 1.50 | 1.21 | 0.62 | 0.48 |
| 14:00 | 0.801 | 0.887 | 0.29 | 4.57 | 1.47 | 1.20 | 0.62 | 0.46 |
| 15:00 | 0.777 | 0.861 | 0.25 | 4.72 | 1.26 | 1.00 | 0.46 | 0.56 |
| 15:30 | 0.759 | 0.917 | 0.27 | 4.58 | 1.15 | 0.91 | 0.34 | 0.68 |

The remaining-session implied variance sits above the realized remaining
variance on 69-75% of days at every clock, not only at the close, and an
oracle that signs realized-minus-implied earns 3.1-4.7 from any entry clock:
the information is priced everywhere. What the rule lacks is the forecast side.
F_rem over-predicts the remaining variance by 15-50% at the median (the
one-bar recalibration itself sits at a median rv_hat/rv_raw of 1.09-1.48: it
targets the mean of a right-skewed variable, so the median forecast exceeds the
median realized), which puts F_rem above IV_rem on 43-62% of intraday days
against an oracle 25-31%. Half of the intraday buys are false; the rule's hit
rate against the oracle is 46-58% before 15:00 and 68% at the close.

## 5. Per-clock hold-to-close, one trade per day (803 days, 2020-05-26 to 2024-04-30)

| clock | sign(s_rem) Sharpe | NW t | worst | maxDD | always short Sharpe (same days) | diff NW t | P[buy] |
|---|---|---|---|---|---|---|---|
| 10:00 | 0.19 | 0.35 | -4.3 | -24.6 | -0.49 | 0.78 | 0.61 |
| 10:30 | 0.52 | 0.99 | -7.8 | -28.5 | -0.08 | 0.79 | 0.59 |
| 11:00 | 0.48 | 0.92 | -8.7 | -21.1 | -0.22 | 0.93 | 0.60 |
| 11:30 | 0.34 | 0.65 | -9.4 | -32.9 | -0.10 | 0.63 | 0.50 |
| 12:00 | 0.72 | 1.42 | -10.8 | -19.8 | -0.11 | 1.21 | 0.61 |
| 12:30 | 0.51 | 0.93 | -10.7 | -29.3 | -0.25 | 1.12 | 0.57 |
| 13:00 | 0.31 | 0.61 | -11.8 | -19.4 | -0.55 | 1.10 | 0.67 |
| 13:30 | 0.10 | 0.21 | -11.3 | -22.3 | -0.25 | 0.48 | 0.46 |
| 14:00 | -0.05 | -0.09 | -7.6 | -28.0 | -0.11 | 0.07 | 0.67 |
| 14:30 | 0.29 | 0.54 | -9.5 | -28.9 | 0.40 | -0.14 | 0.52 |
| 15:00 | 1.29 | 2.24 | -5.2 | -22.0 | 0.75 | 0.58 | 0.50 |
| 15:30 | 1.89 | 3.62 | -5.4 | -12.5 | 0.31 | 2.22 | 0.37 |

First-fire rules (one trade per day): long at the first clock with s_rem > 0,
else short from the first clock: Sharpe 0.34; the short analogue -0.22.
Always short held to the close from an early clock is negative despite the
variance richness: the straddle pays |move|, and the fat right tail outweighs
the many small wins.

Era split (Sharpe): before 2022-05-16 the mid-morning clocks look strong
(10:30-12:30: 1.4-2.1); in the daily-0DTE era (489 days) every clock is
negative or zero except 15:30 (1.97). Per-year at 10:00: -1.4, -0.4, 0.8, 0.5,
0.6. The mid-morning full-sample numbers are 2020-21.

Sensitivity to the construction (Sharpe, same days):

| clock | rv_hat / w (trade-row profile) | rv_hat / w_next (panel profile) | main-repo F_rem (raw yhat^2 x baseline) |
|---|---|---|---|
| 10:00 | 0.19 | -0.02 | -0.40 |
| 12:00 | 0.72 | 0.38 | 0.13 |
| 13:00 | 0.31 | 0.44 | -0.11 |
| 15:00 | 1.29 | 1.13 | 0.50 |
| 15:30 | 1.89 | 1.89 | 1.40 |

## 6. Costs - one crossing per day

Median half-spread over premium rises from 1.2% at 10:30 to 2.8% at 15:30.
Crossed-spread Sharpe of sign(s_rem): -0.17 to +0.49 before 15:00, 0.95 at
15:00, 1.46 at 15:30; always short crossed is negative at every clock except
14:30 and 15:00.

## Verdict

The construction is honest - it reconciles with the close trade to machine
precision, every input is known at the entry bar, and the implied term is the
market's price of the remaining window - but it does not generalize the edge
intraday: sign(s_rem) is insignificant at every clock before 15:00, negative in
the daily-0DTE era everywhere but the close, and dependent on which of two
equally honest profile estimates is used. The implied side is rich at every
clock and an oracle earns 3-4.7 from any entry, so the miss is on the forecast
side: the remaining-session forecast runs 15-50% above realized at the median
and buys twice as often as it should. Any revival has to fix the multi-bar
level calibration of the forecast, not the pairing.
