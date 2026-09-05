# 14. The close-bar scalar: one trailing number between $RV$ and $r^2$

Read-only study, **pre-registered on one cell**. Script
`writeup/intraday_proposals/14_close_bar_scalar.py`; tables and the figure in
`results/atm_straddle_intraday/proposals/14/` (`14_gate.csv`, `14_fact.csv`,
`14_c_by_year.csv`, `14_c_series.csv`, `14_flips.csv`, `14_calibration.csv`,
`14_oracles.csv`, `14_rules.csv`, `14_paired.csv`, `14_placebo.csv`,
`14_causality.csv`, `14_positions.csv`, `14_all_tags.csv`,
`14_daily_pnl_points.csv`, `14_cum_points.png`). Nothing is wired into any
notebook. Every number below is printed by the script.

Report 13 measured the fact this file acts on. On the traded 16:00-stamp bars
$\mathrm{mean}(RV)/\mathrm{mean}(r^2)=0.6756$, where $RV$ is the realized
variance the forecast is calibrated to and $r^2$ is the squared thirty-minute
terminal return the settled straddle pays on; the eleven daytime stamps run
0.975--1.202 and the 16:00 stamp is below 1 in 19 of 24 years. So $\widehat{RV}$
is biased low as a forecast of the close bar's terminal variance by roughly a
third, and sign(s) compares the implied -- which prices the terminal move --
against it. **The one cell is a single trailing scalar that removes the level
error and changes nothing else**: $c_t$ is the ratio of means of $r^2$ to $RV$
over the previous 250 16:00-stamp sessions, prior sessions only, and the
position becomes $q=\mathrm{sign}(c_t\widehat{RV}_t-\mathrm{iv\_var}_t)$. **The
cell, its window, its comparators and the standing rule were fixed before any
number was seen.**

> **One cell, one window.** No other window was run. No per-clock version, no
> shrinkage, no variant of any kind. Section "The urge" records what was not
> done.

> **Provenance.** Block-diagonal ridge as the primary forecast, bar-end labelled
> panel `results/spxw_pnl/yhat_blk2_fomc1.parquet`, the deck's 866 days from
> `daily_blk2.parquet`. $r^2$ comes from the main repository's minute-bar
> summary panel `data/core_stats.parquet`: `sumret` is the bar's terminal log
> return and `sumret2` its realized variance, which reproduces the forecast
> panel's `rv_raw` to exactly 0. The close bar's $RV$ and $r^2$ are the tape and
> not the forecast: the script asserts they are identical on all eight forecast
> panels, which is why one scalar serves all eight tags.

## The cell

$$c_t=\frac{\text{mean of }r^2\text{ over the previous 250 close bars}}
{\text{mean of }RV\text{ over the same bars}},\qquad
\widehat{TV}_t=c_t\,\widehat{RV}_t,\qquad
q_t=\mathrm{sign}\!\left(\widehat{TV}_t-\mathrm{iv\_var}_t\right).$$

The 250-session window is closed at $t-1$ and a full 250 prior sessions are
required; before that the cell sits flat, $q=0$. It never sits flat here. The
first traded day is session 4,604 of the panel's 5,685 16:00-stamp sessions, so
the warm-up is spent long before the deck starts: $c_t$ exists on **866 of 866**
days and **0** days are flat. $+1$ is long, $-1$ is short, and $s=0$ is short.
The trade is the deck's: the 15:30 nearest-OTM package, cash-settled.

**Comparators.** T0, the deck's rule $q=\mathrm{sign}(\widehat{RV}-\mathrm{iv\_var})$,
and always short.

**Fills.** The midpoint case enters at the quoted midpoint. The crossed case
pays the touch at entry -- long pays the ask, short receives the bid -- and cash
settlement pays no exit spread. No row of any rule is untradeable at the crossed
spread: 0 for all three.

## Gate

| figure | target | reproduced | abs diff |
|---|---|---|---|
| sign(s) mean | 0.094736 | 0.094736 | 3.5e-07 |
| sign(s) $t$ | 2.480957 | 2.480957 | 4.2e-07 |
| sign(s) Sharpe | 1.338322 | 1.338322 | 4.0e-07 |
| always short Sharpe | 0.203779 | 0.203779 | 1.3e-07 |
| $\widehat{RV}$ rebuilt from the panel | 0 | 0 | **0** |
| deck position against $\mathrm{sign}(\widehat{RV}-\mathrm{iv\_var})$ | 0 days | 0 days | **0** |

866 days, worst gate difference 4.2e-07. **Gate passed.**

**Causality.** Day $d$'s own close-bar $RV$ and terminal return are tripled,
together with every later session's, and the scalar is rebuilt. On ten cut days
from 2020-01-03 to 2024-04-30, $c_t$ on the cut day moves on **0 of 10** and the
cell's position on **0 of 10**. Later sessions move on **9 of the 9** cuts that
have later sessions -- the largest displacement is 4.63 in $c$ -- so the
perturbation has teeth and the shift is doing the work.

## 1. The scalar

$\mathrm{mean}(RV)/\mathrm{mean}(r^2)$, and the scalar it implies:

| frame | $n$ | ratio of means | implied $c$ |
|---|---|---|---|
| all session bars, 11 daytime stamps | 62,530 | 1.1042 | 0.906 |
| the 16:00 stamp, all sessions | 5,685 | 0.7646 | 1.308 |
| the 16:00 stamp, the traded days | 866 | 0.6756 | **1.480** |

$c_t$ on the traded days, by year:

| year | $n$ | mean | min | max |
|---|---|---|---|---|
| 2020 | 158 | 1.4347 | 0.9443 | 1.8120 |
| 2021 | 158 | 1.1662 | 0.8648 | 1.4576 |
| 2022 | 219 | 1.1719 | 1.0798 | 1.3344 |
| 2023 | 248 | 1.0783 | 0.9880 | 1.1693 |
| 2024 | 83 | 1.1612 | 0.9842 | 1.3267 |

Over the 866 traded days $c_t$ has mean 1.1910, standard deviation 0.1680, and
runs 0.8648 to 1.8120; over all 5,435 sessions that carry a value it has mean
1.2183 and runs 0.7033 to 1.8998. It is above 1 on 96.5% of traded days.

**The trailing scalar is smaller than the fact it is built from.** The full
sample says 1.480 on the traded bars and 1.308 on all 16:00 stamps; the trailing
window delivers 1.19 on average and falls to 1.08 through 2023. The 250-session
window that ends inside the deck's own years is carrying 2020's crisis close
bars at the start (1.43 in 2020) and a quiet trailing window later. So the cell
applies about a fifth of the level correction the pooled fact advertises, and it
applies most of it in the year the trade least needs it.

## 2. What the scalar does to the position

| | |
|---|---|
| days | 866 |
| T0 long / short | 346 / 520 |
| cell long / short / flat | 508 / 358 / 0 |
| days the position differs | **164** (18.94%) |
| short $\to$ long | **163**, which is 31.35% of T0's shorts |
| long $\to$ short | 1 |

The pre-registered expectation holds almost exactly: a third of T0's shorts move
toward long, and essentially nothing moves the other way. Buying goes from
39.95% of days to 58.66%.

## 3. Calibration against $r^2$

QLIKE $=y/f-\log(y/f)-1$. A negative Diebold--Mariano $t$ favours the rescaled
forecast; the $t$ is autocorrelation-robust at lag $\lfloor 1.5\,n^{1/3}\rfloor=14$.

| target | $n$ | QLIKE $\widehat{RV}$ | QLIKE $c\widehat{RV}$ | DM $t$ | mean $\widehat{RV}$ / mean $y$ | mean $c\widehat{RV}$ / mean $y$ |
|---|---|---|---|---|---|---|
| $r^2$, the terminal variance | 866 | **1.4350** | 1.4381 | $+0.25$ | 0.7528 | **1.0715** |
| $RV$, reference | 866 | **0.1100** | 0.1376 | $+4.39$ | 1.1144 | 1.5860 |

**The scalar fixes the level and does not improve the loss.** The mean forecast
goes from 0.753 of $\mathrm{mean}(r^2)$ to 1.071 -- the level error the fact
identified is gone -- and QLIKE moves the wrong way by 0.0031, with a
Diebold--Mariano $t$ of $+0.25$: a dead heat. Against $RV$, which is what the
map was fitted to, the rescaling is unambiguously worse (0.1376 against 0.1100,
$t=+4.39$). Fixing the first moment of a forecast of $r^2$ buys nothing on
$r^2$'s own loss, because $r^2$ is one squared draw and its QLIKE is dominated
by the shape of the ratio, not by its mean.

## 4. The oracle bound

Neither of these is a trade; all three read the bar they price. The question is
whether rescaling closes the 0.17-versus-0.73 gap report 13 measured.

| oracle | % long | agree with $\mathrm{sign}(r^2-\mathrm{iv})$ | corr with $\mathrm{sign}(R)$ | hit rate | Sharpe mid | Sharpe crossed |
|---|---|---|---|---|---|---|
| $\mathrm{sign}(RV-\mathrm{iv\_var})$ | 27.14 | 69.17% | **0.1728** | 0.6282 | 4.531 | 4.072 |
| $\mathrm{sign}(c\,RV-\mathrm{iv\_var})$ | 40.88 | **62.36%** | **0.1495** | 0.5924 | 3.865 | 3.432 |
| $\mathrm{sign}(r^2-\mathrm{iv\_var})$ | 28.41 | 100% | **0.7285** | 0.8695 | 14.647 | 14.412 |

**The scalar moves the oracle away from the terminal oracle, not toward it.**
Agreement with $\mathrm{sign}(r^2-\mathrm{iv\_var})$ falls from 69.17% to 62.36%,
correlation with the sign of the trade's profit falls from 0.173 to 0.150, the
hit rate falls from 0.628 to 0.592, and the peeking Sharpe falls from 4.53 to
3.86 at the midpoint. This is the bound on what the cell could have done, and it
is negative before the forecast is involved at all. The gap between the two
oracles is not a level gap. A scalar multiplies every day by the same number and
so cannot reorder $RV$ against $\mathrm{iv\_var}$ in the way the terminal move
does; it only moves the threshold, and the threshold moves the wrong way.

## 5. The cell scored

Per unit of the entry premium actually paid. `t` is the plain $t$; Sharpe is
$\bar r/s_r\sqrt{252}$. The 866 days and the days carrying $c_t$ are the same
866 days, so the two frames are identical and only one is printed here.

**Midpoint fills:**

| rule | $n$ | mean | $t$ | Sharpe | % buy | hit rate | pts/day |
|---|---|---|---|---|---|---|---|
| cell $\mathrm{sign}(c\widehat{RV}-\mathrm{iv})$ | 866 | 0.03285 | 0.858 | 0.4627 | 58.66 | 0.485 | $+0.024$ |
| **T0** $\mathrm{sign}(\widehat{RV}-\mathrm{iv})$ | 866 | 0.09474 | 2.481 | **1.3383** | 39.95 | 0.544 | $+0.227$ |
| always short | 866 | 0.01448 | 0.378 | 0.2038 | 0.00 | 0.618 | $+0.020$ |

**Crossed spread:**

| rule | $n$ | mean | $t$ | Sharpe | % buy | hit rate | pts/day |
|---|---|---|---|---|---|---|---|
| cell | 866 | $-0.00001$ | $-0.000$ | $-0.0001$ | 58.66 | 0.485 | $-0.208$ |
| **T0** | 866 | 0.06137 | 1.612 | **0.8696** | 39.95 | 0.544 | $-0.005$ |
| always short | 866 | $-0.02028$ | $-0.510$ | $-0.2749$ | 0.00 | 0.618 | $-0.212$ |

The cell's hit rate is 0.485, below a coin and below T0's 0.544.

## 6. The paired difference against T0

Daily, both fills. Autocorrelation-robust $t$ at lag
$\lfloor 1.5\,n^{1/3}\rfloor=14$. Sharpe difference by circular block bootstrap,
block 21, $B=2{,}000$, `rng(0)`, draws shared.

| fill | $n$ | mean diff | $t$ | robust $t$ | $\Delta$Sharpe | percentile 95% | basic 95% |
|---|---|---|---|---|---|---|---|
| mid | 866 | $-0.0619$ | $-2.39$ | $-2.49$ | $\mathbf{-0.8757}$ | $\mathbf{[-1.627,-0.148]}$ | $\mathbf{[-1.604,-0.125]}$ |
| crossed | 866 | $-0.0614$ | $-2.37$ | $-2.47$ | $\mathbf{-0.8697}$ | $\mathbf{[-1.627,-0.138]}$ | $\mathbf{[-1.601,-0.112]}$ |

Both intervals exclude zero, and both exclude it on the losing side. 0.85% of
the 2,000 bootstrap draws are positive at the midpoint and 0.90% at the crossed
spread.

## 7. Placebo

2,000 draws, `rng(0)`, the cell's own long share (58.66%) permuted across days.

| fill | real Sharpe | percentile | placebo median | 5th | 95th |
|---|---|---|---|---|---|
| mid | 0.4627 | **83.2** | $-0.044$ | $-0.905$ | $+0.807$ |
| crossed | $-0.0001$ | **83.3** | $-0.509$ | $-1.370$ | $+0.348$ |

The cell's sign is better than a coin thrown at its own rate -- the 83rd
percentile at both fills -- and much worse than the sign it replaced. Most of
what is left is the rate itself: buying on 59% of days is a different portfolio
from T0's 40%, and the placebo says the day-by-day ordering adds little on top
of it.

## 8. All eight forecasts

`14_all_tags.csv` carries every row. Crossed spread:

| tag | T0 Sharpe | cell Sharpe | days differ | $\Delta$Sharpe | percentile 95% |
|---|---|---|---|---|---|
| a0 | 0.491 | $-0.122$ | 153 | $-0.613$ | $[-1.291,+0.032]$ |
| **blk2** | **0.870** | $-0.000$ | 164 | $\mathbf{-0.870}$ | $\mathbf{[-1.627,-0.138]}$ |
| blk2_inc | 0.805 | 0.059 | 170 | $-0.746$ | $[-1.413,-0.104]$ |
| lgbm | 0.982 | 0.479 | 160 | $-0.502$ | $[-1.371,+0.381]$ |
| xgb | 1.022 | 0.549 | 151 | $-0.473$ | $[-1.290,+0.321]$ |
| lasso_t | 0.935 | 0.048 | 177 | $-0.886$ | $[-1.454,-0.325]$ |
| lasso_f | 1.044 | 0.204 | 159 | $-0.840$ | $[-1.566,-0.067]$ |
| enet | 0.545 | 0.228 | 176 | $-0.317$ | $[-0.912,+0.310]$ |

Calibration against $r^2$ on the traded bars:

| tag | QLIKE $\widehat{RV}$ | QLIKE $c\widehat{RV}$ | DM $t$ |
|---|---|---|---|
| a0 | 1.4584 | 1.4511 | $-0.57$ |
| **blk2** | **1.4350** | 1.4381 | $+0.25$ |
| blk2_inc | 1.4336 | 1.4380 | $+0.35$ |
| lgbm | 1.4341 | 1.4297 | $-0.38$ |
| xgb | 1.4367 | 1.4316 | $-0.45$ |
| lasso_t | 1.4274 | 1.4279 | $+0.05$ |
| lasso_f | 1.4389 | 1.4419 | $+0.24$ |
| enet | 1.4320 | 1.4310 | $-0.09$ |

The cell loses to its control at the crossed spread on **all eight** forecasts,
by 0.32 to 0.89 in Sharpe; four of the eight intervals exclude zero against it
and none excludes zero in its favour. Calibration splits four to four with every
Diebold--Mariano $t$ inside $\pm0.58$: rescaling is a coin on QLIKE, uniformly a
loss on the trade. The cell's Sharpe at the midpoint runs 0.34 (a0) to 1.01
(xgb) and at the crossed spread $-0.12$ to 0.55.

## Figure

`14_cum_points.png`: cumulative index points a day, the cell against T0 and
against always short, at the midpoint and at the crossed spread, with a third
panel of $c_t$ over the full 5,435 sessions that carry it (the 866 traded days
shaded, the full-sample 1.48 marked). The third panel is the file's other
finding on its own: $c_t$ spends 2010--2019 near and below 1 and only the crisis
windows push it toward 1.5, so the "roughly a third" of the pooled fact is not a
stable feature of the tape.

## Verdict

**Kill.** The scalar does exactly what it was built to do to the level and
nothing it was hoped to do to the trade. Calibration against $r^2$ does not
improve: QLIKE 1.4381 against T0's 1.4350 with a Diebold--Mariano $t$ of $+0.25$
on 866 days, a dead heat, even though the mean forecast moves from 0.753 of
$\mathrm{mean}(r^2)$ to 1.071 and the level error is genuinely removed. The
trade is worse at both fills: Sharpe 0.463 at the midpoint against T0's 1.338
and $-0.000$ at the crossed spread against T0's 0.870, with $\Delta$Sharpe
$-0.876$ and $-0.870$ and percentile intervals $[-1.627,-0.148]$ and
$[-1.627,-0.138]$ that **exclude zero against the cell** at both fills, robust
$t$ of the daily difference $-2.49$ and $-2.47$. The cell's own sign is not
noise -- the 83rd placebo percentile at both fills -- it is simply worse than
the sign it replaced, hit rate 0.485 against 0.544. Under the standing rule an
adoption needs a win at the crossed spread with an interval excluding zero
**and** better calibration against $r^2$; this cell has neither, and its interval
excludes zero in the wrong direction. **T0 stands.** The reason is visible before
the forecast is involved: rescaling the variance oracle moves its agreement with
the terminal oracle from 69.17% to 62.36% and its correlation with the sign of
the trade's profit from 0.173 to 0.150, so the ceiling a scalar can reach is
already below where the trade already is. A scalar can move a threshold; it
cannot reorder the days, and the days are where the 0.17-versus-0.73 gap lives.

## The urge

The 2023 mean of $c_t$ is 1.08 against a full-sample 1.48, which is an obvious
invitation to try a shorter window, or a window taken only from the deck's own
years, or a shrinkage toward 1.48. **The urge was recorded and not acted on.**
No second window was run, no per-clock version, no shrinkage. One cell was
pre-registered and one cell is reported. The next file that wants to test a
window should pre-register the grid and pay for it.

## Open questions

- $c_t$ is not stationary in the way the pooled fact implies: it runs near and
  below 1 through 2010--2019 and above 1.5 only in crisis windows
  (`14_c_series.csv`). The pooled 0.68 on the traded bars is a statement about
  2020--2024, not about the tape.
- The level error is real and removing it costs the trade. That is the finding:
  the sign rule is not reading $\widehat{RV}$ as an unbiased forecast of the
  terminal variance. It is reading $\widehat{RV}-\mathrm{iv\_var}$ as a ranking,
  and the ranking's threshold is where the deck's edge sits. What the threshold
  is doing has never been isolated in this series.
- Report 13's open question stands untouched: the reversal at the close --
  daytime stamps at 1.05--1.20 and the 16:00 stamp at 0.765 -- is a fact about
  the closing auction that no file has priced directly.
