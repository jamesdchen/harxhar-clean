# 15. Baseline attribution: why the baseline beats a daily HAR

Read-only study, **pre-registered on eight fixed designs**. Script
`writeup/intraday_proposals/15_baseline_attribution.py`; tables and the figure in
`results/atm_straddle_intraday/proposals/15/` (`15_main_table.csv`,
`15_paired_vs_n5.csv`, `15_steps.csv`, `15_daily_forecasts.csv`,
`15_causality.csv`, `15_sharpe_ladder.png`). Nothing is wired into any notebook.
Every number below is printed by the script.

## The question

The baseline (HAR + calendar OLS) beats a traditional daily HAR on the 15:30
trade. Why? Is it overnight persistence, the intraday rungs, the calendar block,
or the diurnal profile?

The baseline is arm `a0_ols_har` of the main repository. Minimum-norm ordinary
least squares on 52 columns of a 24-hour 30-minute futures panel: the base-2 HAR
lag ladder $\mathrm{har\_ma}_r$, $r\in\{1,2,4,\dots,2048\}$, on the
profile-adjusted target, shifted one bar; that ladder's 24 session-edge
interactions `_x_open` and `_x_close`; and a 16-column calendar block (`DOW_0`
to `DOW_4`, `hour`, `is_overnight`, `is_open`, `is_close`, `is_opex`,
`is_opex_week`, `is_quad_witch`, `is_rebalance_close`, `is_month_end`,
`is_quarter_end`, `days_to_opex`). The window is 24,000 bars ending strictly
before the target bar and the fit is refit at every bar. The target is the
winsorized $\sqrt{RV/B}$ with $B$ the per-slot trailing-20 diurnal profile.
Stamps are naive ET and bar-end labelled, so the 16:00-stamp row is the
15:30-to-close bar the trade prices.

## The eight designs

All eight are minimum-norm OLS on the same window, the same target, the same
profile, refit at every stamp that is scored or that the recalibration reads.
**They differ only in the regressors.** No design was added, dropped or
re-specified after a number was seen.

| design | columns | regressors |
|---|---|---|
| N0 | 3 | daily HAR only: the trailing 1-day, 5-day and 22-day means of $RV$, that is **48, 240 and 1,056 bars** of the 24-hour grid (48 bars = 1 day), shifted one bar, plus an intercept. Means of the **raw** realized variance: no calendar, no interactions, no profile in the regressors. |
| N1 | 7 | the daily-and-above rungs the forecaster actually uses: $\mathrm{har\_ma}_{32,64,128,256,512,1024,2048}$ on the profile-adjusted target (32 bars = 16 hours, 64 = 1.3 days, 1024 = 21 days). |
| N2 | 10 | N1 plus the session rungs $\mathrm{har\_ma}_{4,8,16}$ -- two, four and eight hours. |
| N3 | 12 | N2 plus the last-hour rungs $\mathrm{har\_ma}_{1,2}$ -- the last half hour and hour. |
| N4 | 36 | N3 plus the 24 session-edge interactions. |
| N5 | 52 | N4 plus the calendar block = **the full baseline**. Must equal the gate. |
| N6 | 52 | N5 with the ladder recomputed on **session bars only** (stamps 10:00..16:00). Rung $r$ is the mean of the last $r$ session bars, so each rung spans the same number of session bars as before and no overnight bar enters. |
| N7 | 52 | N5 with the ladder computed on **overnight bars only** (every stamp outside 10:00..16:00). |

The target is already $\sqrt{RV/B}$, so N0 uses it too and N0's forecast is
profile-scaled like every other design's. The step N0 to N1 is therefore the
diurnal profile moving from the target alone into the regressors, on the panel's
own base-2 ladder, and it is the only step that changes two things at once (the
series the rung averages and the rung lengths). Every other step adds columns to
its predecessor.

> **Provenance.** The panel is the main repository's own build
> (`unification._load_panel`, which calls `run_geometry_local.prepare_full` on
> the cached prep matrix) with `FEATURE_SET_TAG` cleared: `a0` is fitted on the
> incumbent panel, not the FOMC one (`results/spxw_pnl/MANIFEST.md`). 300,317
> rows and 52 backbone columns; the arm's dedup drops no column, and the script
> refuses to run if it ever does. Forecasts are produced at the 16,383 fit-mask
> stamps (10:30..16:00) from 2018-12-19, which is 260 sessions of margin before
> the first scored day -- more than the 250 sessions the library's
> recalibration window reads. On the 24-hour grid a
> session carries **13** bars at stamps 10:00..16:00 on the modal day, 12.84 on
> average (half sessions carry fewer), against 30.56 overnight bars on average.
> Each design's forecasts are written to a parquet in the deck's format and read
> back through `asl.load_yhat_1530`, so the Mincer-Zarnowitz map is **identical
> for every design**. The trade, the 866 days and the quotes are the deck's
> `daily_a0.parquet`.

## Gates

| gate | target | reproduced | difference |
|---|---|---|---|
| the 36 ladder + edge columns rebuilt from the target and rescaled by the pipeline's scaler | the panel's own | matches | **4.619e-14** |
| N5 against the shipped `yhat_a0` table, all 16,383 fit rows | -- | -- | **4.882e-13** relative, 0 unmatched stamps |
| N5's recalibrated $\widehat{RV}$ against the deck's | -- | -- | **1.359e-16** |
| sign(s) Sharpe | 0.967310 | 0.967310 | 4.55e-07 |
| always short Sharpe | 0.203779 | 0.203779 | 1.28e-07 |

866 days, sign(s) mean 0.068589, $t$ 1.793181, 33.03% long. No row of any rule
is untradeable at the crossed spread: **0 of 866**. **Gates passed.**

**Causality.** On ten cut days from 2020-01-03 to 2024-04-02, everything at or
after the day's 16:00 stamp is multiplied ($RV$ by 9, the target by 3), every
ladder is rebuilt through the same pipeline and every design is refit. Day $d$'s
own 16:00 forecast moves on **0 of 50** design-days; a session 20 days later
moves on **50 of 50**, by 0.0004 to 2.88 in the forecast, so the perturbation has
teeth and the one-bar shift is doing the work.

## The table

866 days. QLIKE is of $\widehat{RV}$ against `rv_raw` at the 16:00 stamps; the DM
$t$ is paired against N5, autocorrelation-robust at the Bartlett lag
$\lfloor 1.5n^{1/3}\rfloor=14$, and **positive means the design loses to N5**.
dSharpe is design minus N5 with a circular block bootstrap (block 21, B 2000,
`rng(0)`, draws shared across designs and fills); the interval shown is the
percentile one, and the script prints both alongside the
autocorrelation-robust $t$ of each paired difference
(`15_paired_vs_n5.csv`). $\rho$ are Spearman rank correlations of the 16:00
forecast against the 15:00-15:30 realized variance (the last bar the forecast
can see) and against the prior session's realized variance.

| design | cols | QLIKE | DM $t$ vs N5 | Sharpe mid | Sharpe crossed | dSharpe mid [95%] | dSharpe crossed [95%] | % long | agrees N5 | $\rho$ last bar | $\rho$ yesterday |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N0 daily HAR | 3 | 0.2617 | 9.43 | 0.5951 | 0.1344 | -0.372 [-1.698, +0.969] | -0.357 [-1.685, +0.999] | 58.08 | 58.31 | 0.6924 | 0.7686 |
| N1 + profile | 7 | 0.1812 | 8.04 | 0.2655 | -0.2052 | -0.702 [-1.680, +0.302] | -0.696 [-1.674, +0.313] | 52.54 | 69.86 | 0.7896 | 0.8523 |
| N2 + session rungs | 10 | 0.1192 | 3.19 | 0.6453 | 0.1691 | -0.322 [-1.171, +0.512] | -0.322 [-1.167, +0.513] | 36.61 | 84.64 | 0.9083 | 0.7723 |
| N3 + last hour | 12 | 0.1081 | 0.13 | **1.0423** | **0.5638** | +0.075 [-0.475, +0.631] | +0.073 [-0.475, +0.626] | 31.41 | 90.76 | **0.9464** | 0.7543 |
| N4 + edges | 36 | 0.1094 | 1.71 | 0.9147 | 0.4352 | -0.053 [-0.435, +0.317] | -0.056 [-0.437, +0.311] | 31.52 | 97.34 | 0.9374 | 0.7850 |
| **N5 = the baseline** | 52 | **0.1078** | -- | 0.9673 | 0.4909 | -- | -- | 33.03 | 100.00 | 0.9346 | 0.7832 |
| N6 session bars only | 52 | 0.1298 | 4.73 | 0.6392 | 0.1609 | -0.328 [-1.083, +0.422] | -0.330 [-1.077, +0.410] | 27.48 | 84.06 | 0.8961 | 0.7995 |
| N7 overnight bars only | 52 | 0.2346 | 10.58 | 0.0190 | -0.4506 | -0.948 [-2.190, +0.208] | -0.942 [-2.187, +0.223] | 55.54 | 64.09 | 0.7146 | 0.8215 |

Always short on the same 866 days: **0.2038** mid, **-0.2749** crossed. The two
horizons the correlations use are themselves 0.6984 correlated in rank, so the
last two columns are related but are not the same measurement.

Mean and $t$ of the daily return, both fills:

| design | mean mid | $t$ mid | mean crossed | $t$ crossed |
|---|---|---|---|---|
| N0 | 0.042245 | 1.103 | 0.009388 | 0.249 |
| N1 | 0.018862 | 0.492 | -0.014439 | -0.380 |
| N2 | 0.045800 | 1.196 | 0.012057 | 0.313 |
| N3 | 0.073882 | 1.932 | 0.040092 | 1.045 |
| N4 | 0.064869 | 1.696 | 0.031053 | 0.807 |
| N5 | 0.068589 | 1.793 | 0.034975 | 0.910 |
| N6 | 0.045374 | 1.185 | 0.011552 | 0.298 |
| N7 | 0.001353 | 0.035 | -0.031675 | -0.835 |

## The ladder, step by step

| step | adds | dSharpe mid | dSharpe crossed | dQLIKE |
|---|---|---|---|---|
| N0 to N1 | the diurnal profile in the regressors | **-0.3295** | -0.3396 | **-0.0805** |
| N1 to N2 | the session rungs (4, 8, 16 bars) | **+0.3797** | +0.3743 | -0.0619 |
| N2 to N3 | the last-hour rungs (1, 2 bars) | **+0.3970** | +0.3947 | -0.0111 |
| N3 to N4 | the session-edge interactions | -0.1276 | -0.1286 | +0.0013 |
| N4 to N5 | the calendar block | +0.0527 | +0.0558 | -0.0016 |
| N5 to N6 | the ladder on session bars only | -0.3281 | -0.3301 | +0.0220 |
| N5 to N7 | the ladder on overnight bars only | -0.9483 | -0.9416 | +0.1268 |

The five steps from N0 to N5 sum to the whole gap: +0.372 mid, +0.357 crossed.

**The loss and the trade do not move together.** The single largest QLIKE step is
N0 to N1, worth -0.0805 -- and it *costs* the trade 0.33 Sharpe. The two steps
that pay at the trade, N1 to N2 and N2 to N3, are worth -0.0619 and -0.0111 in
loss, the second of them a tenth of the first step's improvement. A design that
forecasts the bar better is not automatically a design that ranks the days
better against the implied.

**The intervals.** Every paired interval in the table covers zero, at both fills,
percentile and basic alike. The basic intervals differ from the percentile ones
by at most 0.09 (N0 mid [-1.713, +0.954], N1 [-1.705, +0.277], N3 [-0.481,
+0.625], N6 [-1.078, +0.427], N7 [-2.104, +0.293]), and the autocorrelation-robust
$t$ of the paired difference runs -1.58 to +0.27. **Nothing here is a candidate
and nothing is adopted.** This is an attribution of a difference that exists in
the point estimates, on the deck's own days, not a claim that any design is
significantly better than any other at the trade.

## The figure

`15_sharpe_ladder.png`: sign(s) Sharpe at both fills across N0 to N5, with N6 and
N7 beside them and always short as the dotted lines.

## The overnight question

N6's rungs count session bars only, so no overnight bar enters. It loses
decisively on the forecast -- QLIKE 0.1298 against 0.1078, DM $t$ 4.73 -- and
about a third of a Sharpe at the trade, -0.328 mid and -0.330 crossed, with the
interval covering zero. N7, the complement, forecasts at the level of a
traditional daily HAR -- QLIKE 0.2346 against N0's 0.2617 and the baseline's
0.1078, DM $t$ 10.58 -- and trades at 0.019 mid, -0.451 crossed, below always
short at both fills.

**One confound, stated.** The pre-registration fixed N6 so that "each rung spans
the same number of session bars as before". A rung of $r$ session bars therefore
reaches $48/13 = 3.7$ times further back in calendar time than a rung of $r$
panel bars, and N7's reaches $48/30.6 = 1.6$ times further. So N6 removes
overnight information *and* stretches the ladder's calendar reach, and part of
its loss is reach rather than overnight content. The design that isolates reach
-- session-bar rungs shortened to the same calendar span -- was not run, because
it was not pre-registered.

## The attribution

The baseline's advantage over a traditional daily HAR at the 15:30 trade is
**+0.372 Sharpe at the midpoint and +0.357 at the crossed spread** (N0 0.595 /
0.134 against the baseline's 0.967 / 0.491), and the bootstrap interval covers
zero at both fills. Where that difference comes from is not ambiguous: **the
intraday rungs, and nothing else**. The two-to-eight-hour rungs are worth +0.380
mid and +0.374 crossed; the last half hour and hour are worth +0.397 and +0.395.
The five rungs shorter than a day carry +0.777 mid between them, more than the
whole gap. Everything else subtracts or does nothing. Moving the daily
information onto the forecaster's own profile-adjusted base-2 ladder costs -0.330
mid, the 24 session-edge interactions cost -0.128, and the 16-column calendar
block adds +0.053 -- so the calendar is not the answer and the diurnal profile in
the regressors is not the answer. The bare twelve-rung ladder N3, with no
interactions and no calendar, is the best design in the study at the trade (1.042
mid, 0.564 crossed, +0.075 over the baseline with the interval covering zero) and
a dead heat with the baseline on the loss (QLIKE 0.1081 against 0.1078, DM $t$
0.13), so the 40 columns the baseline adds to the ladder buy nothing here. The
correlation column says the same thing in one line: the rank correlation between
the forecast and the 15:00-15:30 realized variance climbs 0.69, 0.79, 0.91, 0.95
across N0, N1, N2, N3 and stays there, while the correlation with yesterday's
realized variance sits between 0.75 and 0.85 in every design including the daily
HAR. **The baseline beats a daily HAR because it can see the last bar**, and it
converts that into the trade by turning short: 58% long at N0, 37% at N2, 31% at
N3 and 33% at the baseline, against a rule that is always short scoring 0.204 mid
and -0.275 crossed. Removing the overnight bars costs the forecast decisively (DM
$t$ 4.73) and the trade about a third of a Sharpe, with the interval covering
zero, so overnight persistence is worth something but is not the mechanism; and
the overnight bars on their own carry a daily HAR's worth of information and no
more, forecasting at 0.2346 against the daily HAR's 0.2617 and the baseline's
0.1078, and trading below always short at both fills. The frontier the earlier
files found at the last bar is visible here in the regressors: the baseline's
edge over the daily HAR is the last hour of the ladder.

## Open questions

- The step N0 to N1 changes the series the rungs average *and* the rung lengths.
  The profile and the ladder cannot be told apart from it. A design with the
  daily 48/240/1056 rungs on the profile-adjusted target would separate them and
  was not pre-registered.
- N6 confounds removing overnight bars with stretching the ladder's calendar
  reach by 3.7. Nothing here decides how much of its -0.33 is which.
- The two steps that pay at the trade improve the loss by -0.0619 and -0.0111,
  and the step that improves the loss most (-0.0805) costs the trade. The map
  from forecast loss to trade Sharpe is not monotone in this study and no file in
  this series has isolated why.
- N3 beats the shipped baseline at the trade at both fills and ties it on the
  loss, every interval covering zero. It is 12 columns against 52. This study
  does not propose replacing the baseline -- the difference is not significant
  and the baseline is the audited comparator -- but a pre-registered head-to-head
  on fresh days would settle it.
