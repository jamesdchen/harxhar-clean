# 10. Check 0, and the ideas that had not been run

Read-only study. Script `writeup/intraday_proposals/10_unrun.py`; tables and
figures in `results/atm_straddle_intraday/proposals/10/` (`10_gate.csv`,
`10_causality.csv`, `10_check0_by_clock.csv`, `10_check0_pooled.csv`,
`10_check0_gap_quantiles.csv`, `10_check0_oracle_cost.csv`,
`10_check0_oracle_by_clock.csv`, `10_A_unconstructible.csv`,
`10_B_synthetic.csv`, `10_B_diagnostics.csv`, `10_C_hurdle.csv`,
`10_D_sparse.csv`, `10_D_clock_selection.csv`, `10_E_unconstructible.csv`,
`10_F_fade_iv.csv`, `10_G_two_line.csv`, `10_books.csv`,
`10_check0_pct_short.png`, `10_cum_crossed.png`). Nothing is wired into any
notebook. Every number below is printed by the script.

**Frame.** The intraday notebook's own cached package file
(`results/atm_straddle_intraday/cache/trade_77439812_1786671584951439500_f24191ffee_9d76153205.parquet`,
15,343 packages) joined to the block-diagonal ridge forecast panel exactly as
sections 4–5b of `notebooks/_write_0dte_intraday_nb.py` join it: bar-end
labelled stamps shifted back one bar, 4,956 packages dropped for having no
forecast row, **10,387 bars on 866 days**, twelve clocks 10:00–15:30. The causal
diurnal share `w` is the expanding per-clock mean of realized bar variance over
prior days only, minimum 63 days, lagged one day; 64 dates carry an incomplete
profile, 757 bars carry no matched signal, and **803 days** can trade the
sign(s) rule at all. Both day counts are reported for every book.

**Gate.** The rebuild reproduces the notebook's two rule tables — Sharpe,
mean/day, crossed Sharpe and crossings a day on all four rules — to
**8.6e-08**:

| rule | Sharpe mid | mean/day mid | Sharpe crossed | crossings/day |
|---|---|---|---|---|
| always short | 1.902039 | 0.160346 | −1.918099 | 16.406467 |
| always short, flat at 15:30 | 3.429979 | 0.145871 | −3.620670 | 16.002309 |
| sign(s) | 1.720591 | 0.138054 | −2.243217 | 17.033487 |
| always short, sign(s) close | 3.171428 | 0.251961 | −0.968889 | 16.615473 |

**Causality.** Ten cut days from 2020-05-27 to 2024-04-30: tripling realized bar
variance on the cut day and every later day moves the trailing profile share on
**0 of 10**.

**Fills.** The midpoint case marks entry and exit at the quoted midpoints. The
crossed case pays the touch on both sides of every crossing, with the intraday
notebook's hold-through exemption — a re-pick that lands on the same two strikes
while the position keeps its sign is a hold, not a round trip. The 15:30 bar
cash-settles at the official close and pays no exit spread. Per-premium returns
divide crossed points by the midpoint entry premium, the notebook's own
convention. No bar in any book is untradeable at the crossed spread (0 of
10,387 in every row of `10_books.csv`).

---

## Lead: what is positive at the crossed spread, and F_t-measurable

Six books. Five of them are the settlement leg, or the settlement leg with
something attached; the sixth is a single 15:00 short.

| book | Sharpe crossed (866) | Sharpe crossed (803) | crossings/day |
|---|---|---|---|
| **G both lines: 15:00 always short + 15:30 sign(s)** | **+1.2706** | **+1.3755** | 2.61 |
| G line 2 only: 15:30 sign(s), cash-settled | +1.1545 | +1.1991 | 0.93 |
| D settlement leg only, second half of the frame | +0.8622 | +0.8954 | 0.50 |
| D pre-registered {11:30, 12:30, 15:00} + sign(s) close | +0.7491 | +0.8902 | 6.61 |
| G line 1 only: 15:00 always short, 30-minute hold | +0.1783 | +0.3428 | 2.00 |

**Exactly one book's point estimate exceeds 15:30 sign(s) alone at the crossed
spread on both frames: G, the two-line book.** Its paired Sharpe difference
against the settlement leg alone is **+0.1162** crossed with a percentile 95%
interval of **[−0.299, +0.518]** and 72.9% of draws positive — *the interval
includes zero*. At the midpoint the same difference is **+0.7665**, percentile
95% **[+0.358, +1.184]**, basic **[+0.349, +1.175]** — that interval excludes
zero. So the two lines *do* stack at the midpoint, and at the crossed spread the
addition is not resolved. G is not adopted; it is the one thing in this file
worth a pre-registered forward test.

Everything else that touches more than the last bar is negative crossed. The
instruction stands unchanged: **trade 15:30 sign(s).**

---

## Check 0 — is the oracle almost always short before 15:30?

Four definitions, on the 10,387-bar frame, by clock and on both day counts. The
position is `sign(RV − price)`: *short* means the realized quantity came in
below the priced one.

**Pooled short rate, and mean realized over priced** (`10_check0_pooled.csv`,
803 signal days; the 866-day column differs only in the third decimal):

| definition | what it compares | n | pct short | mean ratio |
|---|---|---|---|---|
| **(1)** `sign(RV_bar − slice)` | the bar against `IV²·h·w`, the published oracle | 9,630 | **60.98** | 1.0127 |
| (2) `sign(RV_bar − IV_rem)` | the bar against the whole remainder (tenor mismatch) | 9,631 | 97.54 | 0.2090 |
| **(3)** `sign(RV_rem − IV_rem)` | remaining against remaining, the honest remainder | 9,631 | **70.72** | 0.9080 |
| (4) `sign(RV_bar − IV²·h)` | the bar against an equal 30-minute slice, `w ≡ 1` | 9,631 | 70.81 | 0.8804 |

**Short rate by clock, 803 signal days** (`10_check0_by_clock.csv`; figure
`10_check0_pct_short.png` plots columns (1) and (3)):

| clock | (1) slice | (2) remainder | (3) rem vs rem | (4) flat share |
|---|---|---|---|---|
| 10:00 | 55.06 | 100.00 | 67.96 | 23.44 |
| 10:30 | 56.36 | 100.00 | 68.33 | 45.64 |
| 11:00 | 56.48 | 99.88 | 68.95 | 55.49 |
| 11:30 | 63.05 | 100.00 | 70.41 | 68.66 |
| 12:00 | 55.79 | 100.00 | 68.37 | 75.22 |
| 12:30 | 60.02 | 100.00 | 69.36 | 84.81 |
| 13:00 | **49.56** | 99.88 | 69.74 | 79.95 |
| 13:30 | 66.13 | 99.88 | 72.48 | 88.67 |
| 14:00 | 66.13 | 99.63 | 71.36 | 82.57 |
| 14:30 | 71.48 | 99.25 | 73.47 | 85.93 |
| 15:00 | 58.78 | 99.13 | **75.34** | 86.43 |
| 15:30 | 72.85 | 72.85 | 72.85 | 72.85 |

**Mean realized over priced by clock** (the median is in
`10_check0_by_clock.csv` and is the number to compare with report 04a):

| clock | (1) mean / median | (3) mean / median | (4) mean |
|---|---|---|---|
| 10:00 | 1.077 / 0.943 | 0.930 / 0.846 | 1.593 |
| 11:30 | 0.968 / 0.856 | 0.919 / 0.835 | 0.886 |
| 12:30 | 1.005 / 0.840 | 0.929 / 0.824 | 0.666 |
| 13:30 | 0.942 / 0.815 | 0.910 / 0.795 | 0.610 |
| 14:30 | 0.914 / 0.761 | 0.874 / 0.790 | 0.686 |
| 15:00 | 1.010 / 0.902 | 0.862 / 0.785 | 0.691 |
| 15:30 | 0.856 / 0.756 | 0.856 / 0.756 | 0.856 |

The `t` of the mean gap (`10_check0_by_clock.csv`) tells the same story in one
column: on definition (1) it runs −3.04 to +2.99 across the daytime clocks and
−11.72 at 15:30; on definition (3) it is between −4.43 and −11.72 at *every*
clock.

**The gap of definition (1), variance units and premium units**
(`10_check0_gap_quantiles.csv`, 803 signal days). A one-bar variance surprise
adds to the variance still to run, and an at-the-money package prices its square
root, so the first-order premium response is `gap / (2·IV²·h)` — the gap in
units of the premium itself:

| clock | mean var | 10 / 50 / 90 var | mean prem | 10 / 50 / 90 prem |
|---|---|---|---|---|
| 10:00 | +5.32e-07 | −3.99e-06 / −1.96e-07 / +5.88e-06 | +0.0046 | −0.0318 / −0.0034 / +0.0495 |
| 12:30 | −2.47e-07 | −2.02e-06 / −2.19e-07 / +1.77e-06 | +0.0002 | −0.0270 / −0.0075 / +0.0332 |
| 14:30 | +5.72e-07 | −2.61e-06 / −5.03e-07 / +2.68e-06 | −0.0106 | −0.0735 / −0.0299 / +0.0639 |
| 15:30 | −1.77e-06 | −5.93e-06 / −1.11e-06 / +1.37e-06 | −0.0718 | −0.2675 / −0.1220 / +0.1643 |

The daytime gap is a symmetric ±3% of premium at the deciles with a median near
zero. The 15:30 gap is −12% of premium at the median and −27% at the tenth
percentile. That is the whole distinction between a mispricing and a noise term,
in one table.

### The one sentence on the short-rate hypothesis

**The hypothesis is false for the slice-oracle and only a tilt for the
remaining-session one: `sign(RV_bar − slice)` is short on 49.6–72.9% of daytime
bars (60.98% pooled, and it *buys* half the bars at 13:00), so the daytime
oracle is genuinely mixed and not a disguised always-short; `sign(RV_rem −
IV_rem)` is short on 67.9–75.3% of bars at every clock — a real short tilt with
a median realized-over-implied of 0.76–0.90, but nowhere near the 80%-plus that
would make the remaining-session reading a pure short.**

These are two different objects and the deck should stop quoting them as one.
The slice-oracle prices the *next thirty minutes* and finds it fair on average
(mean ratio 1.0127, median 0.76–1.01 by clock) with real dispersion either way;
its edge is day-by-day, not a level. The remaining-session oracle prices *the
rest of the session* and finds it persistently rich (median ratio 0.785–0.846
before 15:00), which is a level and which the always-short line already
harvests — that is the same rich remainder reports 04a and 07 measured, and
recalibrating a level does not supply a day-by-day sign. Definition (4) is the
control that shows how much of any short rate is an artefact of the slicing: an
equal 30-minute share says the 10:00 bar is *cheap* (short on 23.4% of bars,
mean ratio 1.593) and the 15:00 bar is rich (86.4%), because it ignores the
diurnal profile entirely.

### The oracle costed — never done before

The published oracle peeks at the bar's own realized variance. It is not a book;
it is the ceiling. The question this file settles is whether the ceiling is
*executable* (`10_check0_oracle_cost.csv`):

| oracle | Sharpe mid (866) | **Sharpe crossed (866)** | Sharpe mid (803) | Sharpe crossed (803) | mean/day mid | mean/day crossed | crossings/day | buy rate | sign placebo |
|---|---|---|---|---|---|---|---|---|---|
| (1) `sign(RV_bar − slice)` | 9.83 | **+5.91** | 10.36 | +6.17 | 0.7887 | 0.4708 | 17.15 | 39.0% | 100.0 |
| (2) `sign(RV_bar − IV_rem)` | 6.26 | +2.04 | 6.20 | +2.26 | 0.4959 | 0.1651 | 16.57 | 2.4% | 100.0 |
| (3) `sign(RV_rem − IV_rem)` | 10.86 | +6.54 | 10.87 | +6.90 | 0.8518 | 0.5163 | 16.88 | 29.2% | 100.0 |
| (4) `sign(RV_bar − IV²·h)` | 10.75 | +6.22 | 10.88 | +6.70 | 0.8536 | 0.4997 | 17.84 | 28.6% | 100.0 |

Per clock, oracle (1) at both fills (`10_check0_oracle_by_clock.csv`): mid 3.52
to 4.51, crossed **positive at every clock** — 0.36 at 10:00, 1.92 at 10:30,
1.08–1.57 through the middle of the day, 2.18 at 14:00, 2.88 at 15:00 and 4.04
at 15:30.

**The ceiling survives the touch.** Seventeen crossings a day cost the oracle
0.318 per unit of premium and it still keeps 0.471 of the 0.789 it earns at the
midpoint — a crossed Sharpe of 5.91 against the settlement leg's 1.15. Knowing
the next bar's variance is worth three times the spread bill. So the intraday
job is a **nowcast**, not a better slice: the marks pay for information about the
bar in flight and they pay it after costs. That is the single most useful thing
in this file, and it is also the thing this repository cannot act on (see A).

---

## A — nowcast, not forecast

**Construction.** The feasible version of the oracle: elapsed realized variance
from the first five or ten minutes of the bar, against the implied slice for the
minutes still unstamped, decided at ten or twenty past and held to the next
half-hour mark.

**Mid / crossed.** Not computed. `10_A_unconstructible.csv`: the finest spacing
in `data/spxw_chain.parquet` is **30 minutes** (14 stamps a session), in
`data/spxw_spot.parquet` **30 minutes**, and `data/core_stats.parquet` stores
per-bar *sums* over at most 30 intra-bar observations — the observations
themselves are not in this repository.

**Verdict: UNCONSTRUCTIBLE — skipped, not leaked.** The only object here with
the content of the bar in flight is the finished bar, and using it is the peek
Check 0 has just measured and costed. The honest statement is the one Check 0
makes: a nowcast is worth up to 5.91 crossed, and it needs sub-30-minute data
this repository does not carry. That is a data purchase, not a modelling choice.

---

## B — synthesize a 30-minute implied, do not slice one

**Construction.** At `t` the package midpoint prices the whole window still to
run. The market's price of the next thirty minutes is what that price gives up
when the clock advances one bar and the implied moves the way it usually moves
at this clock:
`σ²_syn = IV_t²·h_t − (g_t·IV_t)²·(h_t − ½)`, with `g_t` the trailing per-clock
mean of `IV_{t+1}/IV_t` over prior days only. Two gates pass: the Black-76
package price at the vendor implied over the remaining window reproduces the
quoted midpoint (median ratio **1.0000**, 5th–95th 0.9991–1.0012), and at 15:30
the synthetic collapses to `IV²/2` exactly (max |ratio − 1| = **0.00e+00**), the
deck's own close-trade implied. No synthetic slice is non-positive (0 of
10,387). Signal `rv_hat − σ²_syn`, one-bar hold.

**Mid.** 1.6783 on 866 days, 1.8638 on 803, mean/day +0.1414, t +3.11, maxDD
−19.31, 49.3% long — against the realized-profile slice's 1.7206 / 1.7876.

**Crossed.** **−2.6315** on 866 days, −2.1489 on 803, mean/day −0.2227, t −4.88,
maxDD −195.59, 18.23 crossings a day — worse than the deck's −2.2432.

**Verdict: KILL, and the reason is collinearity.** The synthetic correlates
**+0.9909** in level and +0.9925 in logs with the realized-profile slice it was
meant to replace, and its median ratio to it is 1.0668: the implied term
structure and the realized diurnal profile agree on the *shape* of the day and
disagree by about seven percent on the *level* of the next bar's share (implied
share 0.106–0.369 by clock against a realized `w` of 0.095–0.347). Synthesizing
a 30-minute implied from the market's own term structure produces the same
object the notebook already builds. The rate-matched sign placebo says the
signal is not noise (99.9th percentile mid, 100.0 crossed) — it is simply the
same signal, and it costs one more crossing a day to carry.

---

## C — hurdle against the quoted half-spread

**Construction.** Trade bar `t` only when the expected edge clears `k` quoted
half-spreads, else flat. The signal is a variance mispricing, so it is mapped to
points the same way Check 0 maps it: `E_t = P_t·|s_t| / (2·IV_t²·h_t)`. (The
brief's `|s|·premium` is not a points quantity; this is that comparison made
dimensionally honest.) The always-short line is run against two readings of
"expected decay": the frozen-spot decay of book B, which is the package's
**theta** and not its expectation — under the market's own implied the package
is a martingale and the gamma term offsets the theta, which is why the median
theta is **1.2196 points** against a realized always-short profit of **0.1529
points a bar** — and the honest one, the trailing per-clock mean of the realized
always-short profit over prior days only (median **0.1674 points**). Median
half-spread 0.2000 points, 1.69% of the midpoint premium. `k = 1.5`
pre-registered.

**Mid.** sign(s) 1.7206 unfiltered → 1.4591 (k=1), 1.1869 (k=1.5), 1.2427 (k=2).
Always short 1.9020 → 1.9519 (trailing-decay k=1.5), 1.8070 (k=2). The theta
hurdle keeps 96–99% of bars and changes nothing.

**Crossed.** Every cell negative. sign(s) −2.2432 → −0.8399 / −0.7476 / −0.3003
as `k` rises, turnover 17.03 → 9.31 / 6.92 / 5.24 crossings a day. Always short
with the trailing-decay hurdle −1.9181 → −2.0939 / −0.9668 / **−0.3103**, at
2.54 crossings a day. Nothing reaches zero.

**Verdict: KILL.** The hurdle works exactly as designed — it cuts turnover by
two thirds to five sixths and cuts the loss by the same proportion — and it
never crosses into profit. That is the arithmetic of report 03 restated: the
break-even half-spread is 0.81–1.52% of premium against a quoted 1.69%, so
there is no subset of bars whose edge pays its own spread, only subsets whose
edge pays less of it. The pre-registered cell's entry-pattern placebo is at the
100.0th percentile at the midpoint and the 79.5th crossed — the filter picks
good bars, and good bars are still not good enough. (`C sign(s)` at k=1.5 sits
at the 92.0th mid, 99.5th crossed.)

---

## D — sparse clocks, not twelve

**Construction.** Two sets, both with `sign(s)` at 15:30. The pre-registered one
is **{11:30, 12:30, 15:00} always short**, fixed before any per-clock Sharpe in
report 03 was re-read. The honest one is chosen on the first half of the frame
(days before 2022-08-05) by a cost criterion rather than a return criterion — a
clock's break-even half-spread must exceed the half-spread actually quoted
there — capped at three daytime clocks and evaluated on the second half.

**Mid.** Pre-registered set **2.7427** (866) / 2.7099 (803), mean/day +0.1946,
t +5.08, maxDD −11.12, 3.93 trades and 6.61 crossings a day, against the
settlement leg alone at 1.5746 and the deck's twelve-bar hybrid at 3.1714. The
paired difference against the settlement leg alone is **+1.1681** in Sharpe,
percentile 95% **[+0.686, +1.659]**, basic **[+0.678, +1.650]**, mean +0.0885 a
day, t +6.23, autocorrelation-robust t +5.81 — *the interval excludes zero.*

**Crossed.** Pre-registered set **+0.7491** (866) / +0.8902 (803), mean/day
+0.0533, t +1.39, maxDD −19.19 — positive, but the paired difference against the
settlement leg alone is **−0.4053**, percentile 95% **[−0.868, +0.043]**, mean
−0.0245 a day, t −1.65, autocorrelation-robust t −1.51. The entry-pattern
placebo, with the settlement leg held fixed and the three daytime entries
redrawn among the eleven daytime bars, puts the real set at the **100.0th**
percentile at both fills (median draw +2.048 mid, +0.100 crossed).

**Verdict: KILL on the pre-registered comparison — but the honest half of this
is the clock screen, and it selects nothing.** On the first half of the frame
**no clock** has a break-even half-spread above its own quoted half-spread; the
best is 15:00 at 2.46% against 3.23%, and the gap widens monotonically down the
ladder to 13:30 at −0.22% against 2.33% (`10_D_clock_selection.csv`). So the
honest sparse set is the empty set, and the pre-registered set — which is a
genuine midpoint improvement, significant and at the top of its placebo — is a
cost of 0.40 in Sharpe at the touch. Three daytime trades are three too many.

---

## E — 0DTE against 1DTE as a market-priced 30-minute window

**Construction.** If quotes for a second expiration existed at the same stamps,
`remaining_0DTE − remaining_1DTE·(τ₀/τ₁)` would be a traded claim on the front
window, and the signal could be scored on that calendar rather than on a sliced
0DTE package.

**Mid / crossed.** Not computed. `10_E_unconstructible.csv`: all **8,198,778**
rows of `data/spxw_chain.parquet` have zero days to expiration.

**Verdict: UNCONSTRUCTIBLE.** There is no 1DTE or next-weekly quote at any
30-minute stamp in this chain, so there is no calendar to price. This is the
second data purchase Check 0 implies, and the cheaper of the two: a second
expiration would make the 30-minute window a *quoted* object instead of a
constructed one, which is precisely the mis-specification report 00 diagnosed.

---

## F — fade the implied, not realized minus implied

**Construction.** The next mark's change in implied volatility is not `F_t`.
Two causal stand-ins: the trailing per-clock mean drift of the implied over
prior days only, and the day's own last thirty-minute implied change (faded, and
followed). The package is long vega, so a book that expects the implied to fall
is short the package: `q = sign(expected change in implied)`.

**Mid.** Trailing drift **−1.9498**; fade the last change +0.9073; follow it
−0.9073; always short the package +1.9020. The peeking line that reads the next
mark's implied change — not `F_t`, listed for the ceiling only — scores
**+10.1642**.

**Crossed.** Trailing drift **−8.3881**; fade −3.1689; follow −5.0237; always
short −1.9181; the peeking line +2.1842.

**Verdict: KILL, and it is emphatically not the same book as always short.**
Position agreement with always short is **35.98%** for the trailing-drift book
and 55.42% for the fade — the vendor's *hourly* implied volatility rises through
the afternoon even as the remaining variance falls (trailing drift by clock:
−0.000129 at 10:00 through +0.000495 at 15:00, negative at only **4 of 12**
clocks), so a book that shorts when the implied is expected to fall is mostly
*long* the package during the day and loses at both fills. The vega term is real
and tradeable — the peeking line earns 10.16 at the midpoint and still +2.18
crossed — and no causal predictor of it in this frame captures any of that.
Fading the implied is a different book from always short, and a much worse one.

---

## G — the 15:00 bar and the settlement leg, as a two-line book

**Construction.** Line 1: short the 15:00 nearest out-of-the-money straddle,
30-minute hold, marked out at the 15:30 midpoint. Line 2: the deck's close
trade, `sign(s)` at 15:30 cash-settled. Then both, on the same days, with the
hold-through exemption netting the boundary whenever the 15:30 pick lands on the
15:00 strikes with the same sign — which happens on **15.7%** of days and makes
the joint book **+0.00831 per unit of premium a day cheaper** than the sum of
the two lines.

**Mid.** Line 1 **2.5050** (mean/day +0.0598, t +4.64, maxDD −5.95, 2.00
crossings); line 2 **1.5746** (+0.1061, t +2.92, maxDD −12.61, 0.93 crossings);
**both 2.3411** (+0.1659, t +4.34, maxDD −10.57, 2.61 crossings). Paired
difference both − line 2: **+0.7665** in Sharpe, 99.9% of draws positive,
percentile 95% **[+0.358, +1.184]**, basic **[+0.349, +1.175]**, mean +0.0598 a
day, t +4.64, autocorrelation-robust t +4.63 — **excludes zero**.

**Crossed.** Line 1 **+0.1783** (866) / +0.3428 (803), mean/day +0.0044; line 2
**+1.1545** / +1.1991, mean/day +0.0778; **both +1.2706 / +1.3755**, mean/day
+0.0904, t +2.36, maxDD −14.34. Paired difference both − line 2: **+0.1162** in
Sharpe, 72.9% of draws positive, percentile 95% **[−0.299, +0.518]**, basic
**[−0.285, +0.531]**, mean +0.0127 a day, t +0.95, autocorrelation-robust
t +0.93 — **includes zero**.

**Verdict: KEEP as the one unresolved candidate; do not adopt.** The expected
answer was that the lines do not stack. They do at the midpoint, significantly,
and the mechanism is visible: the 15:00 short is a two-crossing trade whose exit
is netted against the settlement entry on one day in six, so the pair costs 2.61
crossings rather than 2.93 and the drawdown *falls* from the settlement leg's
−12.61 to −10.57 at the midpoint. At the crossed spread the addition is +0.12 in
Sharpe with an interval covering zero. The entry-pattern placebo — settlement
leg held fixed, the single daytime short redrawn among the eleven daytime bars,
200 draws — puts 15:00 at the **100.0th** percentile at both fills, with a median
draw of +1.735 mid and **+0.758 crossed**: a *random* daytime short damages the
settlement leg at the touch (0.758 against 1.155) while the 15:00 one does not.
That is a clock effect, not a size effect, and it is the same last-hour boundary
report 09 found from the other side. Report 09 measured a 15:00 entry *held to
settlement* and rejected it; this measures a 15:00 entry *closed at 15:30* and
finds it the only daytime bar that pays its own spread. The two findings are
consistent: what is worth having at 15:00 is one bar of carry, not a head start
on the signal.

---

## Everything in one table

`10_books.csv` carries every column for every book; the figure
`10_cum_crossed.png` plots the crossed cumulative of the surviving trade against
the four candidates. Pre-registered comparison: **15:30 sign(s) alone at the
crossed spread, +1.1545 on 866 days and +1.1991 on 803.**

| book | F_t | Sharpe mid (866) | Sharpe crossed (866) | Sharpe crossed (803) | mean/day crossed | t crossed | maxDD crossed | crossings/day |
|---|---|---|---|---|---|---|---|---|
| oracle (1) `sign(RV_bar − slice)` | no | +9.83 | **+5.91** | +6.17 | +0.4708 | +10.96 | −9.89 | 17.15 |
| oracle (3) `sign(RV_rem − IV_rem)` | no | +10.86 | **+6.54** | +6.90 | +0.5163 | +12.13 | −10.77 | 16.88 |
| F peeking, the next implied change | no | +10.16 | +2.18 | +3.00 | +0.0985 | +4.05 | −50.23 | 18.38 |
| **G both lines** | yes | +2.34 | **+1.27** | **+1.38** | +0.0904 | +2.36 | −14.34 | 2.61 |
| G line 2, 15:30 sign(s) | yes | +1.57 | +1.15 | +1.20 | +0.0778 | +2.14 | −15.39 | 0.93 |
| D pre-registered sparse set | yes | +2.74 | +0.75 | +0.89 | +0.0533 | +1.39 | −19.19 | 6.61 |
| G line 1, 15:00 always short | yes | +2.51 | +0.18 | +0.34 | +0.0044 | +0.33 | −9.62 | 2.00 |
| C sign(s), hurdle k=2 | yes | +1.24 | −0.30 | −0.31 | −0.0180 | −0.56 | −31.91 | 5.24 |
| C always short, trailing-decay hurdle k=2 | yes | +1.81 | −0.31 | −0.32 | −0.0085 | −0.58 | −17.09 | 2.54 |
| D the deck's hybrid, twelve bars | yes | +3.17 | −0.97 | −0.69 | −0.0788 | −1.80 | −91.99 | 16.62 |
| always short, twelve bars | yes | +1.90 | −1.92 | −1.66 | −0.1663 | −3.56 | −144.60 | 16.41 |
| sign(s), twelve bars | yes | +1.72 | −2.24 | −2.33 | −0.1795 | −4.16 | −156.60 | 17.03 |
| B synthetic 30-minute implied | yes | +1.68 | −2.63 | −2.15 | −0.2227 | −4.88 | −195.59 | 18.23 |
| F fade the last implied change | yes | +0.91 | −3.17 | −2.95 | −0.2682 | −5.87 | −233.37 | 17.32 |
| F trailing implied drift | yes | −1.95 | −8.39 | −8.81 | −0.3608 | −15.55 | −312.46 | 15.47 |

---

## Verdict

1. **Check 0 separates two oracles the deck had been quoting as one.** The
   slice-oracle is *mixed*: short on 49.6–72.9% of daytime bars, mean realized
   over slice 1.0127, and it buys half the bars at 13:00. The
   remaining-session oracle is a *tilt to short*: 67.9–75.3% at every clock,
   median realized over implied 0.785–0.846 before 15:00 — persistently rich,
   which is a level the always-short line already collects, not a day-by-day
   sign. "Daytime is always short, you just could not see it" is false; the
   leftover juice is mixed long and short, and it needs a nowcast.
2. **The ceiling is executable, and that is new.** Costed at the crossed spread
   for the first time, the published oracle keeps a Sharpe of **5.91** on 866
   days (6.17 on 803) after seventeen crossings a day; it is positive at every
   clock, from 0.36 at 10:00 to 4.04 at 15:30. Information about the bar in
   flight is worth roughly three times the spread bill. The intraday job is a
   nowcast, not a better slice.
3. **This repository cannot run that nowcast.** The finest series it carries is
   30 minutes (A), and it carries only one expiration, so the 30-minute window
   cannot be quoted rather than constructed (E). Both gaps are data purchases,
   and they are the two purchases Check 0 argues for.
4. **Every book that touches more than the last bar is negative at the crossed
   spread.** Synthesizing the implied from the term structure reproduces the
   realized-profile slice (correlation 0.99) and costs one more crossing a day;
   hurdles cut turnover by five sixths and the loss by the same proportion
   without reaching zero; the honest cost screen selects **no** daytime clock;
   fading the implied is a different book from always short and a much worse
   one.
5. **One candidate is unresolved.** The 15:00 always-short bar stacked on the
   15:30 `sign(s)` settlement scores **+1.2706** crossed against the settlement
   leg's +1.1545, with a paired interval of [−0.299, +0.518]. It stacks
   significantly at the midpoint (+0.7665, [+0.358, +1.184]), its drawdown is
   smaller than the settlement leg's, and 15:00 sits at the 100th percentile of
   random daytime entry clocks at both fills while the median random clock
   *hurts*. It is worth a pre-registered forward test and nothing more.

**The instruction stands: trade 15:30 `sign(s)`.**

## Open questions

- The two-line book G is the third time 15:00 has surfaced (08's best-of-24,
  09's rejected cell B, now a two-crossing carry line). Each time it has been
  measured on the same 866 days. Is 2024-05-01 onward available on the same
  footing for the forward test G actually needs?
- Check 0 prices the nowcast at up to 5.91 crossed. What sub-30-minute series
  would it take — one-minute index returns for 2020-2024 would do — and does
  the vendor's 30-minute option tape have an intra-bar sibling?
- A second expiration at the same stamps would turn the 30-minute window into a
  quoted object. Report 00 named the constructed slice as *the* mis-specification;
  E says the fix is a chain purchase, not a modelling one. Which is cheaper?
- Definition (2) is short on 97.5% of bars with a mean ratio of 0.21. It is a
  pure tenor artefact and it earns a crossed Sharpe of 2.04. Nothing in the deck
  should ever pair a one-bar forecast with a remaining-session implied, and the
  retired 30-minute pairing in §6 of the notebook is the same mistake with the
  opposite sign — worth one sentence in the notebook's prose.
