# 07a — Audit of the remaining-session recalibration (independent rebuild)

Adversarial audit of the sibling study 07 (a causal recalibration of the
remaining-session forecast before the sign rule). Read-only; nothing wired into
a notebook. Script, log and CSV: `scratchpad/remaining_recal_audit/{audit.py,
audit_log.txt, per_clock_crosscheck.csv}`. Block-diagonal ridge forecast.

## What was audited (frozen spec)

At bar $t$: $RV_{\mathrm{rem},t}$ = realized variance from $t$ to the close;
raw forecast $F_{\mathrm{rem}}=\widehat{RV}_t/w_t$ (the intraday notebook's
causal share $w$, minimum 63 days, lagged one day); $B_{\mathrm{rem}}$ = the
panel's time-of-day profile summed over the remaining bars. A causal
Mincer–Zarnowitz fit on the normalized scale,
$\sqrt{RV_{\mathrm{rem}}/B_{\mathrm{rem}}}=a+b\sqrt{F_{\mathrm{rem}}/B_{\mathrm{rem}}}$,
on the trailing 250 days strictly before day $d$: per clock, and pooled across
clocks with the share $w$ as a third regressor. Mean-targeting variant
$\hat F=(m^2+\hat\sigma^2)B_{\mathrm{rem}}$; median-targeting $\hat F=m^2B_{\mathrm{rem}}$.
Signal $\hat s=\hat F-\mathrm{IV}_{\mathrm{rem}}$ with
$\mathrm{IV}_{\mathrm{rem}}=\mathrm{IV}^2_{\mathrm{hr}}h_{\mathrm{rem}}$; sign rule; one
trade per day per entry clock, package held to the close and cash-settled.
Frame: the 803 days with forecasts and shares (2020-05-26 to 2024-04-30; the
413 cache days after 2024-04-30 have no forecast and are excluded); the
recalibration needs 250 prior days, so the **scored frame is 553 days,
2021-12-27 to 2024-04-30** — almost entirely the daily-0DTE era. References
(sign($s_{\mathrm{rem}}$), always short) are scored on the same days.

Reconciliation: at 15:30 the raw $F_{\mathrm{rem}}$ equals the deck's
$\widehat{RV}$ exactly, $s_{\mathrm{rem}}$ equals the deck's signal to
$9\times10^{-12}$, and the hold-to-close return equals the deck's $R$ exactly
(803 days).

## 1. Leakage — pass

- The target $RV_{\mathrm{rem}}$ of day $d$ is known only at that day's close.
  The fit for day $d$ uses rolling sums shifted by one day (per clock, and per
  day for the pooled fit), so day $d$ never enters its own fit.
- Perturbation: realized variance multiplied by 9 on every day $\ge d$ and on
  every bar of day $d$ at or after the entry clock (which changes $w$ for later
  days, $RV_{\mathrm{rem}}$ for day $d$, and both fits for later days) leaves
  $\hat F(d,c)$ unchanged for both the per-clock and the pooled fit: **0
  violations in 10 random cut points** (relative tolerance $10^{-10}$).
- $B_{\mathrm{rem}}$ is a sum of the forecast file's fixed profile column; its
  causality was audited with the one-bar recalibration and is not re-tested
  here.

## 2. Calibration — the overshoot is only partly removed

Median $\hat F/RV_{\mathrm{rem}}$ by clock (scored frame):

| clock | raw $F_{\mathrm{rem}}$ | per-clock mean | per-clock median | pooled mean | pooled median |
|---|---|---|---|---|---|
| 10:00 | 1.29 | 1.16 | 1.07 | 1.22 | 1.10 |
| 11:00 | 1.24 | 1.19 | 1.09 | 1.20 | 1.06 |
| 12:00 | 1.33 | 1.26 | 1.11 | 1.26 | 1.12 |
| 13:00 | 1.56 | 1.40 | 1.23 | 1.47 | 1.33 |
| 14:00 | 1.51 | 1.45 | 1.21 | 1.46 | 1.31 |
| 15:00 | 1.29 | 1.14 | 1.06 | 1.25 | 1.12 |
| 15:30 | 1.15 | 1.14 | 1.07 | 1.19 | 1.07 |

By year, all clocks pooled: raw 1.22 / 1.31 / 1.32 / 1.21 (2021 partial year
to 2024); per-clock mean 1.87 / 1.26 / 1.28 / 1.15; per-clock median 1.41 /
1.13 / 1.14 / 1.05. The mean-targeting variants stay 15–45% above the
realized remainder at mid-day because the target is right-skewed and the
variance term $\hat\sigma^2$ (large for multi-bar remainders) is added back;
the median variants land nearer 1.05–1.10 at the open and the close but still
1.2–1.3 at 13:00–14:00. No variant delivers a level-calibrated remaining
forecast across the day, and none introduces look-ahead through the target.

## 3. Per-clock cross-check (one trade per day, hold to close, 553 days)

sign($\hat s$), per-clock mean variant, against the references on the same days:

| clock | buy share $\hat s$ / $s_{\mathrm{rem}}$ | Sharpe $\hat s$ | $t$ | worst | maxDD | Sharpe $s_{\mathrm{rem}}$ | Sharpe AS | $\Delta$ vs $s_{\mathrm{rem}}$: $t$, 95% CI | $\Delta$ vs AS: $t$, 95% CI | crossed $\hat s$ | placebo pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10:00 | 0.47 / 0.62 | 0.79 | 1.34 | −4.3 | −20.2 | 0.60 | −0.46 | 0.37, [−0.91, +1.13] | 1.43, [−0.69, +2.76] | 0.55 | 72 |
| 10:30 | 0.51 / 0.54 | −0.28 | −0.48 | −7.8 | −21.3 | −0.40 | 0.14 | 0.25, [−0.86, +0.90] | −0.56, [−2.03, +0.97] | −0.48 | 10 |
| 11:00 | 0.53 / 0.55 | −0.09 | −0.15 | −5.6 | −27.5 | −0.27 | −0.29 | 0.24, [−1.33, +1.40] | 0.21, [−1.98, +1.86] | −0.28 | 17 |
| 11:30 | 0.56 / 0.51 | 0.25 | 0.42 | −5.2 | −16.2 | −0.52 | −0.13 | 1.17, [−0.18, +2.06] | 0.42, [−1.54, +2.04] | 0.05 | 36 |
| 12:00 | 0.60 / 0.61 | 0.73 | 1.14 | −8.0 | −19.3 | 0.19 | −0.24 | 0.73, [−0.66, +1.82] | 0.93, [−1.05, +2.76] | 0.53 | 68 |
| 12:30 | 0.69 / 0.63 | −0.05 | −0.07 | −9.0 | −26.7 | −0.25 | −0.46 | 0.28, [−1.12, +1.61] | 0.43, [−1.63, +2.26] | −0.24 | 21 |
| 13:00 | 0.70 / 0.75 | 1.05 | 1.71 | −4.4 | −16.2 | 0.17 | −0.60 | 1.40, [−0.15, +2.03] | 1.49, [−0.59, +3.49] | 0.86 | 85 |
| 13:30 | 0.72 / 0.57 | 0.33 | 0.52 | −4.2 | −17.5 | −0.04 | −0.02 | 0.51, [−1.04, +1.73] | 0.30, [−2.24, +2.34] | 0.11 | 43 |
| 14:00 | 0.69 / 0.72 | 0.24 | 0.41 | −7.6 | −20.1 | −0.53 | 0.29 | 1.83, [−0.04, +1.87] | −0.05, [−2.12, +1.77] | −0.02 | 37 |
| 14:30 | 0.50 / 0.59 | −0.13 | −0.23 | −9.5 | −25.7 | −0.17 | 0.72 | 0.10, [−0.70, +0.83] | −1.12, [−2.49, +0.72] | −0.43 | 17 |
| 15:00 | 0.40 / 0.52 | 0.89 | 1.54 | −3.9 | −12.7 | 0.26 | 1.59 | 1.31, [−0.32, +1.50] | −0.76, [−2.77, +1.01] | 0.53 | 78 |
| **15:30** | 0.39 / 0.41 | **1.59** | **2.58** | −5.4 | −12.0 | **1.78** | 0.62 | −0.59, [−0.86, +0.44] | 1.15, [−0.81, +2.46] | 1.17 | **98** |

Placebo = 2,000 series that pick a random entry clock each day (median 0.43,
95th percentile 1.38): only 15:30 clears it. No clock before 15:30 is
significant on its own ($t\le1.71$), no difference against
sign($s_{\mathrm{rem}}$) or always short is resolved at 95% at any clock, and
at the close the recalibrated rule is slightly *below* the raw one (1.59 vs
1.78, $t$ −0.59). Crossed at one spread per day: only 15:30 (1.17) and 13:00
(0.86) exceed 0.5. Era split: the pre-2022-05-16 sub-sample is 95 days and
reads ±2 at several clocks (noise); the daily-0DTE era gives 0.0–1.0 before
the close and 1.8 at 15:30.

Sharpe by clock across all five variants shows how fragile the mid-day signs
are: at 12:00 the per-clock mean variant reads +0.73 and the per-clock median
−1.10; at 15:00 +0.89, +1.54 and −0.02 (per-clock mean, per-clock median,
pooled mean). The sign of $\hat s$ at mid-day is set by the level term, and
the level term is the part the recalibration does not pin down.

## 4. The trap — not the mechanism, because there is no gain to explain

The feared outcome (a median-calibrated forecast against a rich implied
saying "sell" on 70–75% of days, so that any gain is always short with fewer
buys) does not occur: the mean-targeting variant *raises* the buy share at
mid-day (0.69–0.72 at 12:30–14:00) because the added residual variance pushes
$\hat F$ up. The buy side carries information only where the raw signal
already had it: mean $R$ on buy days minus sell days is +0.085 vs −0.120 at
15:30 ($t$ 2.56), +0.089 vs −0.056 at 13:00 ($t$ 1.53), +0.080 vs −0.019 at
10:00 ($t$ 1.51), and zero or wrong-signed elsewhere. "Short only when
$\hat F<\mathrm{IV}_{\mathrm{rem}}$, flat otherwise" never beats always short
significantly ($t\le1.49$).

## 5. Robustness

- Per-clock vs pooled fit: the two variants' Sharpes across clocks correlate
  0.71; they disagree in sign at 15:00 and 11:30.
- Window 500 vs 250 (303 common days): signs flip at 14:30 (−0.05 vs +0.47)
  and the level changes by 0.4–0.8 at 13:00–15:00; only 15:30 is stable
  (2.35 vs 2.61 on that sub-frame).
- At 15:30 the re-fit is a near-identity on top of the deck's own
  recalibration ($a=-0.0007$, $b=0.963$; median $\hat F/F_{\mathrm{rem}}$
  0.975; sign agreement with $s_{\mathrm{rem}}$ 95.1%), so it does not
  reproduce the deck exactly — the raw $F_{\mathrm{rem}}$ does — and the 5%
  of flipped signs cost a little.

## Verdict

The construction is honest — the fit uses only prior days, future
perturbations leave it untouched, and it reconciles with the close trade —
but recalibrating the level of the remaining-session forecast does not
generalize the edge intraday: no entry clock before 15:30 is significant or
beats a random entry clock, the mid-day signs flip across equally admissible
variants and windows, and at the close the extra fit slightly hurts. The
remaining-session implied is rich at every clock (04a), yet the forecast's
day-by-day sign information before the last hour is what is missing, and a
level correction cannot supply it.

## Open questions

- A remaining-session forecast that is not $\widehat{RV}_t/w_t$ (the main
  repository's direct multi-horizon per-bar forecasts) is the only untested
  input; this repository does not carry those files.
- The scored frame is the daily-0DTE era by construction (250-day window on
  top of the 63-day warm-up); a shorter standing window would widen the frame
  but is a tuning choice.
