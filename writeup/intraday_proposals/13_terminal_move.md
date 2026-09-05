# 13. The terminal move: forecasting $r^2$ instead of $RV$

Read-only study, **pre-registered on four cells**. Script
`writeup/intraday_proposals/13_terminal_move.py`; tables and figures in
`results/atm_straddle_intraday/proposals/13/` (`13_gate.csv`,
`13_bias_by_clock.csv`, `13_bias_by_year.csv`, `13_tape_check.csv`,
`13_calibration.csv`, `13_expected_payoff.csv`, `13_t3_coef_by_year.csv`,
`13_t3_coef_path_by_year.csv`, `13_t3_incremental.csv`,
`13_t3_retransformation.csv`, `13_t3_retransformation_trade.csv`,
`13_oracles.csv`, `13_cells.csv`, `13_cells_t3_support.csv`, `13_paired.csv`,
`13_placebo.csv`, `13_causality.csv`, `13_positions.csv`, `13_all_tags.csv`,
`13_daily_pnl_points.csv`, `13_cum_points.png`, `13_bias_by_clock.png`).
Nothing is wired into any notebook. Every number below is printed by the script.

The 15:30 straddle held to cash settlement pays on the terminal move
$|S_{\text{close}}-K|$, that is on the squared thirty-minute return $r^2$, not on
the sum of squared one-minute returns $RV$. The forecast pipeline and the deck's
recalibration target $RV$, so under a martingale the two have the same mean but
different realizations, and the sign rule may be comparing the wrong quantity
with the implied. This file asks whether re-targeting the recalibration at the
terminal variance is worth anything.

> **Cells T0--T3 were fixed before any number was seen.** Nothing is selected
> from a grid. T3 is the only cell with more than one degree of freedom and its
> feature list is pre-registered exactly as written below; nothing was added to
> it. Section 4b reports one **post-hoc** diagnostic, labelled as such
> throughout and not adoptable.

> **Multiplicity.** Four pre-registered cells (T2 carries a hurdle variant and
> an `rv_hat` secondary, both fixed in advance, so seven position series in
> all). The eight-forecast table in section 10 is the same cells under the other
> seven forecasts, not a search over forecasts.

> **Provenance.** Block-diagonal ridge as the primary forecast, bar-end labelled
> panel `results/spxw_pnl/yhat_blk2_fomc1.parquet`, the deck's 866 days from
> `daily_blk2.parquet`. The terminal move comes from the main repository's
> minute-bar summary panel `data/core_stats.parquet`: `sumret` is the bar's
> terminal log return and `sumret2` its realized variance, which reproduces the
> forecast panel's `rv_raw` to exactly 0 on all 276,317 rows.

## The four cells

**T0, the control.** The deck's close trade. $\widehat{RV}$ from the
$RV$-targeted Mincer--Zarnowitz map, position $q=\mathrm{sign}(s)$ with
$s=\widehat{RV}-\mathrm{IV}^2_{\mathrm{hr}}/2$.

**T1, the terminal-variance target.** The same weighted Mincer--Zarnowitz map --
same weights $1/\max(\hat y, q_{10})^2$, same 250-session window, same session
fit mask 10:30--16:00 -- with the target $y=\sqrt{r^2/B}$ in place of
$\sqrt{RV/B}$. The back-transform $(m^2+s^2)B$ is then $\widehat{TV}$, a
forecast of the bar's squared terminal return, and
$q=\mathrm{sign}(\widehat{TV}-\mathrm{iv\_var})$. Only the target column
differs, so the weights, the window, the mask and the session ranking are
identical by construction.

**T2, expected payoff against premium.** With $\widehat{TV}$, the expected
payoff of the *actual* nearest out-of-the-money package -- call at $K_c$, put at
$K_p$, index at $S_{15:30}$ -- under a normal terminal log return with variance
$\widehat{TV}$, in closed form (Black-76 at a zero rate, forward $=S_{15:30}$,
total variance $\widehat{TV}$; the library's own `_bsm_package_price`). Long
when the expected payoff exceeds the midpoint premium, short otherwise. The
hurdle variant **T2h** is long only above the ask, short only below the bid,
flat in between. **T2b** and **T2bh** are the same two rules on $\widehat{RV}$,
pre-registered as the secondary.

**T3, terminal-move regressors.** An expanding, prior-days-only least-squares
fit, minimum 250 sessions, refit every day, of $\log r^2$ at the 16:00 stamp on
$\log\widehat{TV}$ plus five features known at 15:30:

1. $|r|$ over 15:00--15:30 (the stamp-15:30 bar's terminal log return);
2. the day's net log return from the 10:00 stamp to the 15:30 stamp;
3. the log of the day's realized variance so far (stamps 10:30--15:30) over its
   trailing prior-days-only mean;
4. the distance of $S_{15:30}$ to the nearest 25-point strike as a fraction of
   the implied move $S_{15:30}\sqrt{\mathrm{iv\_var}}$ (pinning);
5. a month-end / FOMC indicator (`asl.fomc_and_monthend`; NA read as other).

The forecast is $\exp(\widehat{\log r^2} + s^2/2)$ and
$q=\mathrm{sign}(\text{forecast}-\mathrm{iv\_var})$.

**Fills.** The midpoint case enters at the quoted midpoint. The crossed case
pays the touch at entry -- buy at the ask, sell at the bid -- and cash
settlement pays no exit spread (`asl.crossed_premium_return`). No row of any
cell is untradeable at the crossed spread: 0 for all seven position series.

## Gate

The deck's rule table for the block-diagonal ridge is reproduced before anything
else is reported, and the recalibration is rebuilt from the panel independently.

| figure | target | reproduced | abs diff |
|---|---|---|---|
| sign(s) mean | 0.094736 | 0.094736 | 3.5e-07 |
| sign(s) $t$ | 2.480957 | 2.480957 | 4.2e-07 |
| sign(s) Sharpe | 1.338322 | 1.338322 | 4.0e-07 |
| always short Sharpe | 0.203779 | 0.203779 | 1.3e-07 |
| $\widehat{RV}$ rebuilt from the panel | 0 | 0 | **0** |
| deck position against $\mathrm{sign}(\widehat{RV}-\mathrm{iv\_var})$ | 0 days | 0 days | **0** |

866 days, worst gate difference 4.2e-07. **Gate passed.**

**The tape check.** `sumret` at the 16:00 stamp against the price path, on the
866 traded days:

| against | mean abs diff | median | max | corr |
|---|---|---|---|---|
| $\log(S_{\text{close}}/S_{15:30})$, official close | 0.000267 | 0.000178 | 0.004074 | 0.9945 |
| $\log(S_{16:00\ \text{tape}}/S_{15:30})$, chain tape | 0.000181 | 0.000115 | 0.002785 | 0.9972 |

The standard deviation of `sumret` is 0.004119 and of
$\log(S_{\text{close}}/S_{15:30})$ 0.004072, so the disagreement is about 6% of
one standard deviation: the futures bar and the cash close differ slightly, as
expected, and the futures bar sits closer to the 16:00 tape print than to the
official close.

**Causality.** Perturbing everything at or after a day's 16:00 stamp -- the
traded bar's own $r$ and $RV$, and every later row -- triples realized variance
and the terminal return on that set and rebuilds every map. On ten cut days from
2021-12-27 to 2024-04-30, $\widehat{TV}$ on the cut day moves on **0 of 10**,
the T3 forecast on **0 of 10**, the T3 position on **0 of 10**. Nothing at or
before 15:30 on the cut day is touched, which is exactly the set a 15:30
decision may read.

## 1. The bias, and where it reverses

$\mathrm{mean}(RV)/\mathrm{mean}(r^2)$ by clock, all 68,215 session bars,
bar-end stamps. Figure `13_bias_by_clock.png`.

| stamp | $n$ | ratio of means | median ratio | % $r^2>RV$ |
|---|---|---|---|---|
| 10:30 | 5,683 | 1.083 | 2.293 | 30.3 |
| 11:00 | 5,683 | 1.055 | 2.194 | 30.9 |
| 11:30 | 5,684 | 1.108 | 2.281 | 29.6 |
| 12:00 | 5,685 | 1.147 | 2.311 | 28.8 |
| 12:30 | 5,685 | 1.127 | 2.419 | 28.0 |
| 13:00 | 5,685 | 1.147 | 2.590 | 26.4 |
| 13:30 | 5,685 | 1.202 | 2.442 | 28.1 |
| 14:00 | 5,685 | 1.110 | 2.430 | 28.7 |
| 14:30 | 5,685 | 1.195 | 2.343 | 29.9 |
| 15:00 | 5,685 | 1.137 | 2.378 | 29.1 |
| 15:30 | 5,685 | 0.975 | 2.376 | 29.5 |
| **16:00** | **5,685** | **0.765** | **1.959** | **34.7** |

Pooled over all session bars the ratio is **1.0468** -- the premise of the file,
and close to the 1.09 that motivated it. **It reverses at the bar the trade
owns.** On the 16:00 stamp -- the traded bar, 15:30 to the close -- realized
variance is only 0.765 of the squared terminal move, and on the 866 traded days
it is **0.676**. The close bar's move is not mean-reverting within the bar; it
is the opposite, a directional push into the settlement. By year the pooled
ratio runs 0.916--1.260 and the close bar's runs 0.575--1.288, below 1 in 19 of
24 years (`13_bias_by_year.csv`).

So a map calibrated to $RV$ is not biased *high* as a forecast of the terminal
variance of the traded bar. It is biased **low**. The file's premise is right in
sign for the day and wrong in sign for the bar that pays.

## 2. Calibration: does the terminal map forecast $r^2$ better?

QLIKE $=y/f-\log(y/f)-1$. A negative Diebold--Mariano $t$ favours the terminal
map; the $t$ is autocorrelation-robust at lag $\lfloor 1.5\,n^{1/3}\rfloor$.

| frame | $n$ | QLIKE $\widehat{RV}$ | QLIKE $\widehat{TV}$ | DM $t$ | mean $\widehat{RV}$ / mean $y$ | mean $\widehat{TV}$ / mean $y$ |
|---|---|---|---|---|---|---|
| 866 traded bars, target $r^2$ | 866 | **1.4350** | 1.4869 | **+2.92** | 0.7528 | 0.6335 |
| all session bars, target $r^2$ | 67,281 | **1.6091** | 1.6273 | **+8.36** | 1.0792 | 0.9350 |
| 866 traded bars, target $RV$ | 866 | 0.1100 | 0.1305 | +4.97 | 1.1144 | 0.9378 |
| all session bars, target $RV$ | 67,462 | 0.1593 | 0.1895 | +17.50 | 1.0317 | 0.8938 |

181 session bars have $r^2$ exactly 0 and are dropped from the $r^2$ rows; no
traded day is.

**The terminal map is worse at forecasting the terminal move than the map that
never looked at it.** On the traded bars QLIKE goes from 1.4350 to 1.4869 with a
Diebold--Mariano $t$ of $+2.92$ against it, and on all session bars 1.6091 to
1.6273 with $t=+8.36$. Two things do it. The target is one squared draw instead
of thirty summed squares, so fitting the line to $\sqrt{r^2/B}$ injects far more
estimation noise than the 4.7% level bias it was meant to remove. And the
pooled fit spends eleven twelfths of its rows on daytime bars where $r^2$ is
*smaller* than $RV$, so the terminal map lands 16% below the $RV$ map in level
($\widehat{TV}/\widehat{RV}=0.842$) at exactly the bar where it needed to be
higher. Both maps under-price the traded bar's terminal variance -- 0.753 and
0.633 of $\mathrm{mean}(r^2)$ -- and the terminal one under-prices it more.

## 3. T2: the expected payoff of the actual package

The closed form is validated against the market first. Pricing the actual
package at the quoted implied variance reproduces the quoted midpoint: ratio
mean **1.0005**, 5th--95th percentile 0.9972--1.0063. So the strike offset and
the two-strike geometry are handled correctly and the only thing T2 changes
relative to T1 is the strike offset and, in the hurdle variant, the flat zone.

Index points per straddle, 866 days:

| mean midpoint premium | mean bid | mean ask | mean $E$[payoff] at $\widehat{TV}$ | at $\widehat{RV}$ | mean realized exit | median half-spread |
|---|---|---|---|---|---|---|
| 6.806 | 6.574 | 7.038 | 5.953 | 6.364 | 6.785 | 2.90% of premium |

Both expected payoffs sit below the realized average exit of 6.785, the
terminal map's by more. T2 differs from T0 on **142** days and T2h on **193**;
T2b differs from T0 on **3** days -- the strike offset alone -- and T2bh on
**110**, which is the flat zone.

## 4. T3: the terminal-move regression

Six regressors and a constant. The trailing mean in feature 3 needs 63 prior
sessions and the regression 250 more, so the fit fires on **553** of the 866
days, 2021-12-27 to 2024-04-30. 84 of the 866 days carry the event flag. The
expanding fit's in-sample $R^2$ on $\log r^2$ is **0.104**.

Per-year in-sample coefficients (a stability check, never traded), with
heteroskedasticity-robust $t$ in brackets:

| term | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|
| const | $-3.73$ [$-0.35$] | $10.87$ [$1.91$] | $-6.69$ [$-1.32$] | $-5.12$ [$-1.02$] | $6.28$ [$0.28$] |
| $\log\widehat{TV}$ | $0.66$ [$0.68$] | $1.99$ [$4.04$] | $0.56$ [$1.23$] | $0.71$ [$1.71$] | $1.63$ [$0.93$] |
| $\lvert r\rvert$ 15:00--15:30 | $6.1$ [$0.04$] | $14.2$ [$0.14$] | $117.0$ [$1.91$] | $116.4$ [$1.13$] | $-213.3$ [$-0.63$] |
| net return 10:00--15:30 | $-23.7$ [$-0.70$] | $-55.9$ [$-2.01$] | $-9.9$ [$-0.72$] | $-19.3$ [$-0.81$] | $-36.2$ [$-0.69$] |
| $\log$ RV so far, relative | $0.81$ [$1.54$] | $-0.13$ [$-0.39$] | $0.21$ [$0.58$] | $-0.20$ [$-0.69$] | $-0.64$ [$-1.06$] |
| pinning | $-0.28$ [$-0.41$] | $0.37$ [$1.16$] | $0.05$ [$0.10$] | $-0.33$ [$-1.27$] | $-0.32$ [$-0.54$] |
| event | $0.63$ [$1.32$] | $0.29$ [$0.72$] | $1.12$ [$2.35$] | $0.94$ [$1.70$] | $0.38$ [$0.28$] |

Sign stability: $\log\widehat{TV}$, the net return and the event flag keep their
sign in 5 of 5 years; $|r|$ in 4 of 5; the relative realized variance and the
pinning distance in 3 of 5, which is a coin. Only one coefficient clears $|t|=2$
more than once, and only $\log\widehat{TV}$ is ever significant with the right
sign twice. The expanding path tells the same story: the $|r|$ coefficient runs
2.1, 7.1, 66.0, 77.1 across 2021--2024 and the pinning coefficient changes sign
in 2024 (`13_t3_coef_path_by_year.csv`).

**Incremental calibration on T3's 553 days:**

| QLIKE T0 $\widehat{RV}$ | QLIKE T1 $\widehat{TV}$ | QLIKE T3 | DM $t$ (T3 $-$ T1) | DM $t$ (T3 $-$ T0) |
|---|---|---|---|---|
| 1.3958 | 1.4254 | **1.8753** | **+6.77** | **+7.98** |

T3 is far worse than either map it was built on top of, and it buys on 100% of
its days.

### 4b. Why -- a post-hoc diagnostic, not a cell

$\log r^2$ is the log of a *squared* normal, not a normal, so the Gaussian
retransformation $\exp(s^2/2)$ is the wrong constant. The in-window residual
variance averages **4.655**, close to $\pi^2/2=4.935$, the variance of the log
of a squared standard normal. So the correction factor is $\exp(s^2/2)=10.29$
where the honest smearing factor -- the in-window mean of $\exp(\text{residual})$,
also prior days only -- is **4.014**. The pre-registered forecast therefore sits
at 5.53 times $\mathrm{mean}(r^2)$ and is above the implied variance on every
single day.

Replacing the Gaussian constant by the smearing factor is **post-hoc and is not
adopted**. Reported once, so the reader can see what the regressors do when the
level is not broken: QLIKE 1.4356 (against T0's 1.3958 on the same days), 64.9%
long, Sharpe 1.824 at the midpoint and 1.485 at the crossed spread against T0's
1.447 and 1.089, paired $\Delta$Sharpe $+0.376$ / $+0.396$ with percentile
intervals $[-1.055,+1.742]$ and $[-1.046,+1.773]$ that cover zero, autocorrelation-robust
$t$ of the daily difference 0.50, placebo 99.95th percentile. It fails the
standing rule on calibration -- 1.4356 against 1.3958 -- and its interval covers
zero. It is a diagnostic, not a candidate.

## 5. The two oracles

Neither is a trade. Both peek at the bar they price.

| oracle | $n$ | % long | corr with $\mathrm{sign}(R)$ | hit rate | Sharpe mid | Sharpe crossed |
|---|---|---|---|---|---|---|
| $\mathrm{sign}(RV-\mathrm{iv\_var})$ | 866 | 27.14 | **0.1728** | 0.6282 | 4.531 | 4.072 |
| $\mathrm{sign}(r^2-\mathrm{iv\_var})$ | 866 | 28.41 | **0.7285** | 0.8695 | 14.647 | 14.413 |

The two oracles agree on **69.17%** of days. This is the number the file was
built to explain. A variance oracle that knows the bar's realized variance
perfectly is only 0.17 correlated with the sign of the held straddle's profit
and right on 63% of days; a terminal-move oracle that knows $r^2$ is 0.73
correlated and right on 87%, and earns more than three times the Sharpe on the
same marks at both fills. **The payoff is a function of the terminal move, and
$RV$ is a weak proxy for it.** The residual gap from 1.00 is the strike offset:
$R$ compares $\max(S_{\text{close}}-K_c,0)+\max(K_p-S_{\text{close}},0)$ with the
paid premium, while the oracle compares $r^2$ with the at-the-money implied
variance, and the two disagree near the boundary.

Levels: mean $r^2/\mathrm{iv\_var}$ 1.030 with median 0.386; mean
$RV/\mathrm{iv\_var}$ 0.855 with median 0.757. The terminal move is the far more
skewed of the two.

## 6. The cells scored

Per unit of the entry premium actually paid. `t` is the plain $t$; Sharpe is
$\bar r/s_r\sqrt{252}$. All 866 days, T3 sitting flat through its warm-up.

**Midpoint fills:**

| cell | $n$ | mean | $t$ | Sharpe | % buy | flat | differ from T0 | pts/day |
|---|---|---|---|---|---|---|---|---|
| **T0** $\mathrm{sign}(\widehat{RV}-\mathrm{iv})$ | 866 | 0.09474 | 2.481 | **1.3383** | 39.95 | 0 | 0 | $+0.227$ |
| T1 $\mathrm{sign}(\widehat{TV}-\mathrm{iv})$ | 866 | 0.03730 | 0.974 | 0.5253 | 33.26 | 0 | 140 | $-0.154$ |
| T2 payoff > mid ($\widehat{TV}$) | 866 | 0.03311 | 0.864 | 0.4663 | 33.26 | 0 | 142 | $-0.157$ |
| T2h payoff vs touch ($\widehat{TV}$) | 866 | 0.04415 | 1.232 | 0.6645 | 27.83 | 100 | 193 | $-0.093$ |
| T2b payoff > mid ($\widehat{RV}$) | 866 | 0.09332 | 2.444 | 1.3182 | 40.30 | 0 | 3 | $+0.225$ |
| T2bh payoff vs touch ($\widehat{RV}$) | 866 | 0.09191 | 2.567 | **1.3850** | 33.95 | 110 | 110 | $+0.214$ |
| T3 $\mathrm{sign}(f_3-\mathrm{iv})$ | 866 | $-0.02634$ | $-0.911$ | $-0.4914$ | 63.86 | 313 | 634 | $-0.296$ |

**Crossed spread:**

| cell | $n$ | mean | $t$ | Sharpe | % buy | flat | differ from T0 | pts/day |
|---|---|---|---|---|---|---|---|---|
| **T0** | 866 | 0.06137 | 1.612 | **0.8696** | 39.95 | 0 | 0 | $-0.005$ |
| T1 | 866 | 0.00340 | 0.087 | 0.0471 | 33.26 | 0 | 140 | $-0.386$ |
| T2 | 866 | $-0.00079$ | $-0.020$ | $-0.0110$ | 33.26 | 0 | 142 | $-0.389$ |
| T2h | 866 | 0.01471 | 0.404 | 0.2177 | 27.83 | 100 | 193 | $-0.293$ |
| T2b | 866 | 0.05996 | 1.575 | 0.8496 | 40.30 | 0 | 3 | $-0.007$ |
| T2bh | 866 | 0.06344 | 1.783 | **0.9616** | 33.95 | 110 | 110 | $+0.022$ |
| T3 | 866 | $-0.04128$ | $-1.462$ | $-0.7888$ | 63.86 | 313 | 634 | $-0.403$ |

**On T3's 553 days**, so every cell is read on one frame
(`13_cells_t3_support.csv`): T0 1.447 mid / 1.089 crossed; T1 1.355 / 0.992;
T2 1.409 / 1.046; T2h 1.511 / 1.171; T2b 1.469 / 1.110; T2bh 1.564 / 1.228;
T3 $-0.615$ / $-0.987$ at 100% long. Read the two frames before reading either:
dropping the first 313 days lifts the close trade from 1.338 to 1.447 at the
midpoint, the same frame effect report 09 measured.

## 7. The paired differences against T0

Daily, on each cell's own tradeable days, both fills. Autocorrelation-robust $t$
at lag $\lfloor 1.5\,n^{1/3}\rfloor$. Sharpe difference by circular block
bootstrap, block 21, $B=2{,}000$, `rng(0)`, draws shared across cells.

| fill | cell | $n$ | mean diff | $t$ | robust $t$ | $\Delta$Sharpe | percentile 95% | basic 95% |
|---|---|---|---|---|---|---|---|---|
| mid | T1 | 866 | $-0.0574$ | $-1.37$ | $-1.49$ | $-0.813$ | $[-1.799,+0.218]$ | $[-1.844,+0.173]$ |
| mid | T2 | 866 | $-0.0616$ | $-1.46$ | $-1.61$ | $-0.872$ | $[-1.831,+0.127]$ | $[-1.871,+0.086]$ |
| mid | T2h | 866 | $-0.0506$ | $-1.25$ | $-1.33$ | $-0.674$ | $[-1.676,+0.353]$ | $[-1.701,+0.329]$ |
| mid | T2b | 866 | $-0.0014$ | $-0.44$ | $-0.44$ | $-0.020$ | $[-0.113,+0.063]$ | $[-0.104,+0.072]$ |
| mid | T2bh | 866 | $-0.0028$ | $-0.21$ | $-0.19$ | $+0.047$ | $[-0.404,+0.475]$ | $[-0.382,+0.497]$ |
| mid | T3 | 553 | $-0.1380$ | $-2.16$ | $-2.33$ | $\mathbf{-2.062}$ | $\mathbf{[-3.943,-0.302]}$ | $\mathbf{[-3.823,-0.181]}$ |
| crossed | T1 | 866 | $-0.0580$ | $-1.38$ | $-1.50$ | $-0.823$ | $[-1.811,+0.206]$ | $[-1.851,+0.166]$ |
| crossed | T2 | 866 | $-0.0622$ | $-1.47$ | $-1.62$ | $-0.881$ | $[-1.838,+0.114]$ | $[-1.875,+0.077]$ |
| crossed | T2h | 866 | $-0.0467$ | $-1.15$ | $-1.22$ | $-0.652$ | $[-1.640,+0.371]$ | $[-1.674,+0.336]$ |
| crossed | T2b | 866 | $-0.0014$ | $-0.44$ | $-0.44$ | $-0.020$ | $[-0.112,+0.064]$ | $[-0.104,+0.072]$ |
| crossed | T2bh | 866 | $+0.0021$ | $+0.15$ | $+0.14$ | $+0.092$ | $[-0.359,+0.514]$ | $[-0.330,+0.543]$ |
| crossed | T3 | 553 | $-0.1374$ | $-2.15$ | $-2.32$ | $\mathbf{-2.077}$ | $\mathbf{[-3.976,-0.301]}$ | $\mathbf{[-3.852,-0.177]}$ |

Only T3's interval excludes zero, and it excludes it on the losing side at both
fills. Every other difference covers zero, and every point estimate on a
$\widehat{TV}$ cell is negative.

## 8. Placebo

One cell beats T0 at the crossed spread: **T2bh**, 0.9616 against 0.8696. Its
sign is placebo-tested: 2,000 draws, `rng(0)`, its own long / short / flat rates
(33.95% long, 12.70% flat) permuted across days.

| fill | real Sharpe | percentile | placebo median | 5th | 95th |
|---|---|---|---|---|---|
| mid | 1.3850 | **99.45** | $+0.044$ | $-0.833$ | $+0.922$ |
| crossed | 0.9616 | **99.55** | $-0.393$ | $-1.255$ | $+0.476$ |

T2bh's sign is not noise. It is also, forecast for forecast, T0's sign: T2bh
uses $\widehat{RV}$, so its calibration against $r^2$ is T0's exactly and the
standing rule's second condition cannot be met by construction. What it changes
is the flat zone -- it stands aside on 110 days when the expected payoff sits
between the bid and the ask -- and $\Delta$Sharpe $+0.092$ crossed, with
percentile $[-0.359,+0.514]$ and basic $[-0.330,+0.543]$, does not resolve at
95%.

## 9. The same cells under all eight forecasts

`13_all_tags.csv` carries every row. The calibration verdict is unanimous:

| tag | QLIKE $\widehat{TV}$ | QLIKE $\widehat{RV}$ | DM $t$ | T1 Sharpe mid | T1 Sharpe crossed | $\Delta$Sharpe crossed |
|---|---|---|---|---|---|---|
| a0 | 1.5011 | 1.4584 | $+2.22$ | $-0.015$ | $-0.484$ | $-0.975$ |
| **blk2** | 1.4869 | **1.4350** | $+2.92$ | $0.525$ | $0.047$ | $-0.823$ |
| blk2_inc | 1.4866 | 1.4336 | $+2.99$ | $0.516$ | $0.039$ | $-0.766$ |
| lgbm | 1.4693 | 1.4341 | $+2.01$ | $0.726$ | $0.243$ | $-0.738$ |
| xgb | 1.4668 | 1.4367 | $+1.73$ | $0.771$ | $0.287$ | $-0.734$ |
| lasso_t | 1.4761 | 1.4274 | $+2.89$ | $0.615$ | $0.134$ | $-0.800$ |
| lasso_f | 1.4917 | 1.4389 | $+2.96$ | $0.474$ | $-0.004$ | $-1.048$ |
| enet | 1.4807 | 1.4320 | $+2.81$ | $0.767$ | $0.284$ | $-0.261$ |

The $RV$ map beats the terminal map on QLIKE against $r^2$ on **all eight**
forecasts, with a Diebold--Mariano $t$ between $+1.73$ and $+2.99$ against the
terminal map every time. T1 loses to T0 at the crossed spread on all eight, by
0.26 to 1.05 in Sharpe; no interval excludes zero. T3 collapses to always long
under every forecast, so its rows are the identical $-0.615$ / $-0.987$ eight
times over -- the clearest possible tell that the cell is a level artefact.

## Figures

`13_cum_points.png`: cumulative index points a day, T0 against the
best-calibrated cell, at the midpoint and at the crossed spread. The second line
is chosen on **QLIKE against $r^2$, not on Sharpe**; cells sharing a forecast tie
exactly on QLIKE and the tie is broken toward the cell whose position differs
from T0 on the most days, never on the return. The pooled QLIKE ranking is
$\widehat{RV}$ 1.4350 $<$ $\widehat{TV}$ 1.4869 $<$ $f_3$ 1.8753, so the
best-calibrated cell other than T0 is T2bh, and T1 -- the cell the file is named
for -- is drawn dashed for context.

`13_bias_by_clock.png`: $\mathrm{mean}(RV)/\mathrm{mean}(r^2)$ by clock, all
session bars, with the reversal at the 16:00 stamp.

## Verdict

**T1, the terminal-variance map. Kill.** Calibration against $r^2$ gets *worse*,
not better: QLIKE 1.4869 against T0's 1.4350 on the traded bars with a
Diebold--Mariano $t$ of $+2.92$ against it, and 1.6273 against 1.6091 on all
67,281 scored session bars with $t=+8.36$. Sharpe 0.525 at the midpoint and
0.047 at the crossed spread against T0's 1.338 and 0.870; $\Delta$Sharpe
$-0.813$ and $-0.823$, percentile $[-1.811,+0.206]$ crossed, covering zero.
Fails both conditions.

**T2, expected payoff against the midpoint premium ($\widehat{TV}$). Kill.**
Same forecast as T1, so the same calibration miss; 0.466 mid and $-0.011$
crossed, $\Delta$Sharpe $-0.872$ / $-0.881$, percentile $[-1.838,+0.114]$
crossed, covering zero. The closed form is right -- it reprices the quoted
midpoint to 1.0005 -- and repricing does not rescue a worse forecast.

**T2h, the hurdle variant ($\widehat{TV}$). Kill.** 0.664 mid and 0.218 crossed;
$\Delta$Sharpe $-0.674$ / $-0.652$, percentile $[-1.640,+0.371]$ crossed,
covering zero. Standing aside on 100 days inside the spread recovers about a
fifth of T1's loss and no more.

**T2b, expected payoff against the midpoint premium ($\widehat{RV}$). Kill.**
It is T0 with the strike offset priced properly: 3 days differ, Sharpe 1.318 mid
and 0.850 crossed, $\Delta$Sharpe $-0.020$ at both fills with percentile
$[-0.112,+0.064]$. No calibration change by construction. Nothing to adopt.

**T2bh, the hurdle variant ($\widehat{RV}$). Kill, but note it.** The only cell
that beats T0 at the crossed spread: 0.9616 against 0.8696, $\Delta$Sharpe
$+0.092$, percentile $[-0.359,+0.514]$ and basic $[-0.330,+0.543]$, both
covering zero; 1.385 against 1.338 at the midpoint; placebo 99.55th percentile
crossed. It uses T0's own forecast, so it cannot improve calibration against
$r^2$ -- the standing rule's second condition is unmeetable by construction --
and its interval covers zero. Not adopted. What it says is narrow and worth
saying: on the 110 days when the package's expected payoff sits inside the
quoted spread, standing aside costs nothing and saves the crossing.

**T3, the terminal-move regressors. Kill.** The pre-registered forecast is above
the implied variance on 100% of its 553 days, because the Gaussian
retransformation $\exp(s^2/2)=10.29$ is the wrong constant for the log of a
squared normal. QLIKE 1.8753 against T0's 1.3958 on those days, Diebold--Mariano
$t$ $+7.98$; Sharpe $-0.615$ mid and $-0.987$ crossed; $\Delta$Sharpe $-2.062$
and $-2.077$ with percentile intervals $[-3.943,-0.302]$ and $[-3.976,-0.301]$
that exclude zero **against** it. The regressors themselves are weak and
unstable: in-sample $R^2$ on $\log r^2$ of 0.104, and of six terms only
$\log\widehat{TV}$, the net session return and the event flag hold their sign in
all five years, with the pinning distance and the relative realized variance
holding in three of five. The post-hoc smeared level (section 4b) is reported
once and is not adoptable: it is still worse-calibrated than T0, 1.4356 against
1.3958, and its Sharpe interval covers zero.

**The standing rule.** Nothing is adopted unless it beats T0 at the crossed
spread with an interval excluding zero **and** improves calibration against
$r^2$. No cell does both. **T0 stands.** The finding worth carrying out of this
file is not any of the cells but the pair of numbers underneath them. The
straddle pays on the terminal move, and the terminal-move oracle is 0.73
correlated with the sign of the trade's profit against the variance oracle's
0.17 -- so there is a very large amount of information in $r^2$ that $RV$ does
not carry. But re-targeting the recalibration at $r^2$ does not reach it: one
squared draw is a much noisier regressand than thirty summed squares, the pooled
fit across twelve clocks pushes the level the wrong way at exactly the bar that
pays (the mean ratio is 1.05 across the day and 0.68 on the traded bar), and the
$RV$-targeted map is the better forecast of $r^2$ on $r^2$'s own loss under all
eight forecasts. The terminal move is the right target and the recalibration is
the wrong instrument for it.

## Open questions

- The reversal at the close is a fact about the tape, not about the trade: the
  ratio $\mathrm{mean}(RV)/\mathrm{mean}(r^2)$ is 1.06--1.20 at every daytime
  stamp and 0.765 on the 16:00 stamp, in 20 of 24 years. Whatever produces it --
  the closing auction, the settlement imbalance -- it is the one bar of the day
  whose realized variance understates its own terminal move, and no file in this
  series has priced it directly.
- A clock-conditional recalibration would let the terminal map fit the close bar
  on the close bar's own rows instead of pooling twelve clocks. Report 02
  rejected per-clock recalibration for the $RV$ target on Sharpe grounds; it was
  never run against $r^2$, where the pooling bias is measurable and
  one-directional. That is the one variant this file's failure actually points
  at, and it was not pre-registered here.
- The terminal-move oracle earns 14.6 at the midpoint and 14.4 at the crossed
  spread -- the spread bill is 0.2 of it. As in report 10, the ceiling is
  enormous and the instrument is missing.
