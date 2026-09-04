# 09. The last hour: the close trade against a 15:00 entry

Read-only study, **pre-registered on two cells**. Script
`writeup/intraday_proposals/09_last_hour_boundary.py`; tables and figure in
`results/atm_straddle_intraday/proposals/09/` (`09_gate.csv`, `09_cells.csv`,
`09_paired.csv`, `09_signal_agreement.csv`, `09_placebo.csv`,
`09_decomposition.csv`, `09_all_tags.csv`, `09_daily_pnl_points.csv`,
`09_cum_points.png`). Nothing is wired into any notebook. Every number below is
printed by the script.

> **Caveat, up front.** The 15:00 entry is not a new idea here. Report 08 found
> it as the best of twenty-four cells and refused to adopt it on that ground.
> This file is its **pre-registered replication**: two cells, fixed before the
> data were read, one of them the deck's own close trade. Nothing is selected.

> **Multiplicity.** Two pre-registered cells, nothing selected. The eight-tag
> table in section 8 is the same two cells under the other seven forecasts, not
> a search over forecasts.

> **Provenance.** Scored on `daily_blk2.parquet`, forecast fingerprint
> `a5c65541dcf0` — the same version report 08 scored.

## The two cells

**Cell A, entry 15:30.** The deck's close trade exactly. Position
$q=\mathrm{sign}(s)$ with $s=\widehat{RV}-\mathrm{IV}^2_{\mathrm{hr}}/2$: short
when the forecast is at or below the implied variance, long otherwise. The
nearest out-of-the-money call and put are picked at 15:30 and cash-settled at
the official close. One crossing.

**Cell B, entry 15:00.** The nearest out-of-the-money call and put are picked at
15:00 by the same guarded picker. Position $q=\mathrm{sign}(s_{\mathrm{rem}})$
with

$$s_{\mathrm{rem}}=\frac{\widehat{RV}_{15:00}}{w_{15:00}}-\mathrm{IV}^2_{\mathrm{hr},15:00}\,h,
\qquad h=1.0\ \text{hour}.$$

$\widehat{RV}_{15:00}$ is the fresh one-bar forecast for $[15{:}00,15{:}30]$ —
the panel row stamped 15:30, issued at 15:00, through the library's
recalibration. $w_{15:00}$ is the trailing share of that bar in the remaining
hour's variance: the expanding per-clock mean of realized bar variance over
prior days only, minimum 63 days, lagged one day — the intraday notebook's own
`w_slice`. $\mathrm{IV}_{\mathrm{hr},15:00}$ is the at-the-money vendor hourly
implied volatility of the two picked legs at 15:00, censored of the solver's
bracket nodes. **Those same strikes are held to cash settlement at the official
close: one crossing, and no 15:30 transaction of any kind.**

**Comparators, same days.** Always short at 15:00 held to settlement; always
short at 15:30 (the deck's row).

**Units and fills.** Per unit of the entry premium actually paid, and in index
points per straddle. $t=\sqrt{n}\,\bar r/s_r$; Sharpe $=\bar r/s_r\times
\sqrt{252}$. The midpoint case enters at the quoted midpoint; the crossed case
pays the touch at entry — buy at the ask, sell at the bid — and cash settlement
pays no exit spread (`asl.crossed_premium_return`).

## Gates

**Reproduction (the gate that had to pass first).** The 15:30 package was picked
independently from `data/spxw_chain.parquet`, the forecast read from the panel,
the settlement taken from the official close cache. Against the deck's own
per-day file: $K_c$, $K_p$, $S$, $\widehat{RV}$ and the settlement exit agree to
0; the midpoint entry to 9.5e-07 (the chain's float32 quotes); the implied
variance to 2.2e-11; the per-premium return to 2.0e-07; the position on 0 of 866
days. The deck's rule table is reproduced:

| rule | n | mean | $t$ | Sharpe | buy |
|---|---|---|---|---|---|
| sign(s) | 866 | 0.094736 | 2.480957 | 1.338322 | 39.9538% |
| always short | 866 | 0.014475 | 0.377762 | 0.203779 | 0% |

Target and reconstruction agree to 4.1e-07 on every one of the four gated
figures. **Gate passed.**

**Causality.** The 15:00 decision reads three things: the forecast issued at
15:00 (the panel row stamped 15:30), the 15:00 quotes, and a trailing profile.
The script asserts it. On ten cut days spread from 2020-05-26 to 2024-04-30,
tripling realized variance on day $d$ and on every later day moves $w_{15:00}(d)$
on **0 of 10**; tripling realized variance on every panel row from day $d$'s
15:30 stamp onward moves $\widehat{RV}_{15:00}(d)$ on **0 of 10**. The third
input is the 15:00 quote itself, which is a 15:00 object by construction.

**Frame.** The deck's 866 days, the twelve half sessions already dropped by the
shared rule. The profile needs 63 prior days, so cell B trades **803** of them,
2020-05-26 to 2024-04-30. No day is lost to a censored implied volatility (862
of 866 days carry one at 15:00, and the four that do not are inside the
warm-up), and the guards refuse no cell at either entry stamp. The paired
statistics and the main table are on those 803 common days; the deck's 866-day
rows are shown separately.

**The share.** Median $w_{15:00}=0.3466$, 5th–95th percentile 0.2892–0.3769: the
15:00–15:30 bar is about a third of the last hour's variance, so the remaining
forecast is about three times the one-bar forecast. Median
$\widehat{RV}_{\mathrm{rem}}/\mathrm{IV}_{\mathrm{rem}}=1.0082$, and cell B buys
on 51.56% of its days.

## Results

803 common days. `mean`, `t`, `Sharpe` per unit of entry premium; `pts` columns
and the drawdown in index points per straddle.

| construction | fill | n | mean | $t$ | Sharpe | pts/day | $t$ pts | Sharpe pts | buy | maxDD pts | worst pts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A 15:30 sign(s)** | mid | 803 | 0.1144 | 2.92 | **1.6357** | 0.5366 | 2.24 | 1.2567 | 38.11% | −140.9 | −49.18 |
| **A 15:30 sign(s)** | crossed | 803 | 0.0840 | 2.15 | **1.2042** | 0.3524 | 1.47 | 0.8243 | 38.11% | −162.3 | −49.43 |
| **B 15:00 sign(s_rem)** | mid | 803 | 0.0748 | 1.95 | **1.0908** | 0.6390 | 1.85 | 1.0342 | 51.56% | −158.5 | −67.27 |
| **B 15:00 sign(s_rem)** | crossed | 803 | 0.0513 | 1.35 | **0.7536** | 0.4334 | 1.25 | 0.7018 | 51.56% | −195.7 | −67.77 |
| always short 15:00 to settlement | mid | 803 | 0.0514 | 1.33 | 0.7476 | 0.3849 | 1.11 | 0.6221 | 0% | −338.6 | −67.27 |
| always short 15:00 to settlement | crossed | 803 | 0.0271 | 0.68 | 0.3834 | 0.1793 | 0.52 | 0.2898 | 0% | −401.8 | −67.77 |
| always short 15:30 (deck) | mid | 803 | 0.0217 | 0.55 | 0.3080 | 0.2479 | 1.03 | 0.5790 | 0% | −153.3 | −49.18 |
| always short 15:30 (deck) | crossed | 803 | −0.0098 | −0.24 | −0.1343 | 0.0636 | 0.27 | 0.1486 | 0% | −197.2 | −49.43 |

The same two 15:30 rows on the deck's full 866 days, for reference:

| construction | fill | n | mean | $t$ | Sharpe | pts/day | maxDD pts |
|---|---|---|---|---|---|---|---|
| A 15:30 sign(s) | mid | 866 | 0.0947 | 2.48 | 1.3383 | 0.2271 | −273.1 |
| A 15:30 sign(s) | crossed | 866 | 0.0614 | 1.61 | 0.8696 | −0.0049 | −326.7 |
| always short 15:30 | mid | 866 | 0.0145 | 0.38 | 0.2038 | 0.0203 | −329.2 |
| always short 15:30 | crossed | 866 | −0.0203 | −0.51 | −0.2749 | −0.2118 | −439.2 |

Read the two frames before reading either. Dropping the 63 warm-up days — the
first quarter of 2020 — lifts the close trade from 1.34 to 1.64 at the midpoint
and from 0.87 to 1.20 crossed. That is a frame effect, not a result, and it is
why cell B is compared only with cell A on the same 803 days.

On those days **A beats B at both fills**: 1.64 against 1.09 at the midpoint,
1.20 against 0.75 crossed. B earns more points a day (0.639 against 0.537 at the
midpoint) because it sells a bigger premium, and it pays for them with more
dispersion and a worse tail: worst day −67.3 against −49.2, drawdown −158.5
against −140.9. Both cells clear both comparators. Always short from 15:00 held
to settlement scores 0.75 mid and 0.38 crossed with a −338.6 drawdown; the
deck's always short scores 0.31 and −0.13. So at 15:00, as at 15:30, it is the
sign that pays, not the short.

## The paired difference B − A

Same days, daily, both fills. HAC $t$ at lag $\lfloor 1.5\,n^{1/3}\rfloor=13$.
Sharpe difference by circular block bootstrap, block 21, $B=2{,}000$, `rng(0)`.

| fill | unit | mean B−A | $t$ | HAC $t$ | ΔSharpe | draws + | percentile 95% | basic 95% |
|---|---|---|---|---|---|---|---|---|
| mid | per premium | −0.0396 | −0.91 | −0.97 | **−0.5449** | 15.5% | [−1.7170, +0.4946] | [−1.5844, +0.6272] |
| mid | points | +0.1024 | 0.28 | 0.29 | −0.2225 | 35.3% | [−1.4794, +0.8810] | [−1.3260, +1.0344] |
| crossed | per premium | −0.0327 | −0.75 | −0.80 | **−0.4506** | 20.3% | [−1.6325, +0.5947] | [−1.4959, +0.7312] |
| crossed | points | +0.0811 | 0.22 | 0.23 | −0.1224 | 40.6% | [−1.3739, +0.9721] | [−1.2170, +1.1290] |

The difference is negative in Sharpe at both fills and in both units, and no
interval excludes zero in either direction. The mean difference per premium is
negative and the mean difference in points is positive — the head start sells
more premium for a slightly worse risk-adjusted return — and neither is
significant. Report 08 measured the same comparison the other way round, on 866
days and against the deck's 1.34, and found +0.60 mid with an interval that also
included zero. On the 803 days both cells can trade, the point estimate reverses
sign. That is what a pre-registered replication is for.

## The two signals

| signal | window it prices | n | hit rate | signal long | oracle long |
|---|---|---|---|---|---|
| 15:00 sign(s_rem) | [15:00, 16:00] | 803 | **0.5467** | 51.56% | 24.66% |
| 15:30 sign(s) | [15:30, 16:00] | 803 | **0.6737** | 38.11% | 27.15% |
| 15:00 sign(s_rem) | [15:30, 16:00] | 803 | 0.5616 | 51.56% | 27.15% |
| 15:30 sign(s) | [15:00, 16:00] | 803 | 0.6463 | 38.11% | 24.66% |

The two signals agree on **66.13%** of the 803 days. Against the realized sign of
remaining realized variance minus remaining implied variance — the quantity each
one is a bet on — the 15:30 signal is right 67.4% of the time and the 15:00
signal 54.7%. The 15:00 signal also buys far too often: 51.6% against an oracle
24.7%. This is the level bias reports 04a and 07 measured, one bar earlier and
undiminished: dividing a one-bar forecast by a trailing share does not give it
day-by-day information about the remaining hour.

## Placebo

2,000 draws, `rng(0)`, the 15:00 sign replaced by a random sign at cell B's own
long rate (51.56%).

| fill | real Sharpe | percentile | placebo median | 5th | 95th |
|---|---|---|---|---|---|
| mid | 1.0908 | **98.35** | −0.0183 | −0.9706 | 0.8802 |
| crossed | 0.7536 | **98.35** | −0.3613 | −1.3100 | 0.5398 |

Cell B's sign is not noise. It clears the 95th percentile of random signs at
both fills. It is simply worse than the sign half an hour later.

## Decomposition: which bar carries cell B

Index points per straddle, cell B split at the 15:30 mark of its own held
strikes. The two legs add to the total exactly (maximum gap 0.000 points).

| fill | bar | mean pts | $t$ | sd pts | share of total | Sharpe pts |
|---|---|---|---|---|---|---|
| mid | 15:00–15:30 | 0.0439 | 0.35 | 3.57 | **6.9%** | 0.19 |
| mid | 15:30–16:00 | 0.5952 | 2.00 | 8.45 | **93.1%** | 1.12 |
| mid | total | 0.6390 | 1.85 | 9.81 | 100% | 1.03 |
| crossed | 15:00–15:30 | −0.1617 | −1.28 | 3.58 | −37.3% | −0.72 |
| crossed | 15:30–16:00 | 0.5952 | 2.00 | 8.45 | 137.3% | 1.12 |
| crossed | total | 0.4334 | 1.25 | 9.80 | 100% | 0.70 |

This is the answer to the question the file was built to ask. The head start
contributes 0.044 points a day at the midpoint, 6.9% of cell B's total, with a
$t$ of 0.35. The settlement bar contributes 0.595, and it is the only leg with a
$t$ above 2. At the crossed spread the first bar is **negative**: the entry
half-spread is paid there, and the extra half hour does not earn it back
(−0.162 points a day, $t$ −1.28). Cell B is the settlement bar plus a leg that
costs money once the spread is paid. (The crossed case charges the whole entry
half-spread to the first bar, which is where it is paid; cash settlement pays no
exit spread.)

## The same two cells under all eight forecasts (secondary)

`09_all_tags.csv` carries the full rows. 803 common days for every tag.

| tag | Sharpe A mid | Sharpe B mid | ΔSharpe mid | 95% mid | Sharpe A crossed | Sharpe B crossed | ΔSharpe crossed | 95% crossed | buy B |
|---|---|---|---|---|---|---|---|---|---|
| a0 | 1.1011 | 0.3148 | −0.7863 | [−2.10, +0.43] | 0.6601 | −0.0272 | −0.6872 | [−2.01, +0.52] | 53.55% |
| **blk2** | **1.6357** | **1.0908** | **−0.5449** | [−1.72, +0.49] | **1.2042** | **0.7536** | **−0.4506** | [−1.63, +0.59] | 51.56% |
| blk2_inc | 1.5976 | 1.1966 | −0.4010 | [−1.67, +0.77] | 1.1700 | 0.8611 | −0.3088 | [−1.58, +0.88] | 54.30% |
| lgbm | 1.6083 | 0.8059 | −0.8024 | [−1.94, +0.27] | 1.1681 | 0.4702 | −0.6979 | [−1.83, +0.40] | 58.90% |
| xgb | 1.5628 | 0.6271 | −0.9357 | [−2.19, +0.27] | 1.1215 | 0.2899 | −0.8316 | [−2.09, +0.39] | 59.78% |
| lasso_t | 1.8110 | 1.2061 | −0.6049 | [−1.82, +0.58] | 1.3762 | 0.8713 | −0.5049 | [−1.74, +0.69] | 55.17% |
| lasso_f | 1.8598 | 1.2298 | −0.6300 | [−1.77, +0.41] | 1.4329 | 0.8926 | −0.5404 | [−1.69, +0.51] | 51.68% |
| enet | 1.4202 | 1.0437 | −0.3765 | [−1.45, +0.65] | 0.9837 | 0.7076 | −0.2761 | [−1.37, +0.76] | 55.55% |

Cell A beats cell B on **all eight** forecasts, at both fills, by 0.28 to 0.94 in
Sharpe. No interval excludes zero. The baseline (HAR + calendar OLS) is the
worst of the eight at 15:00 — B scores 0.31 mid and −0.03 crossed there — and
the two trees, whose 15:00 buy shares are the highest in the table (58.9% and
59.8%), lose the most. The direction is the same everywhere; only its size moves
with the forecast.

## Figure

`09_cum_points.png`: cumulative index points a day, cell A against cell B with
both comparators, at the midpoint and at the crossed spread, on the 803 common
days.

## Verdict

The head start is not real. On the 803 days both cells can trade, the 15:00
entry scores 1.09 at the midpoint and 0.75 at the crossed spread against the
close trade's 1.64 and 1.20. The paired Sharpe difference B − A is −0.54 at the
midpoint and −0.45 crossed, and every interval covers zero: percentile
[−1.63, +0.59] and basic [−1.50, +0.73] at the crossed spread. The interval
includes zero, so the frontier is the last bar and cell B is not adopted. The
decomposition says the same thing arithmetically: the extra half hour is 6.9% of
cell B's midpoint points with a $t$ of 0.35, and at the crossed spread it is
negative. The 15:00 signal is not noise — it clears the 98th percentile of
random signs at both fills — but it is a worse signal than the one issued thirty
minutes later, right on 54.7% of days against 67.4%, and buying on 51.6% of days
against an oracle's 24.7%. All eight forecasts agree on the direction. Report
08's 15:00 cell was the best of twenty-four; pre-registered against the close
trade on the days both can trade, it loses.

## Open questions

- The 63-day warm-up costs the close trade 0.30 in Sharpe (1.34 on 866 days,
  1.64 on 803). That is the first quarter of 2020, not a construction choice.
  Any future comparison of an intraday cell with the close trade has to be run
  on the cell's own days, as this one is.
- Cell B's positions and the deck's disagree on a third of days (66.13%
  agreement), and B's are worse. The disagreement is not random — B buys 51.6%
  against A's 38.1% — so it is the level of the remaining forecast again, the
  same miss reports 04a and 07 measured. A direct multi-horizon forecast for the
  remaining hour, which this repository does not carry, is still the only thing
  that could change the reading.
