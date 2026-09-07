# 18. Flat on FOMC statement days and month-end sessions at the close

Adversarial audit of one candidate rule change to the deck's 15:30 close trade
(`notebooks/atm_straddle_rv_iv.ipynb`, per-day files
`results/atm_straddle_0dte_1530/daily_<tag>.parquet`). Script
`18_event_day_flat.py`, outputs `results/atm_straddle_intraday/proposals/18/`.
Nothing here is wired into a notebook. Every number below is printed by that
script.

## The candidate

One rule, nothing else changed — same package, same 15:30 entry, same cash
settlement at the official close, same forecast:

$$q_t = 0 \text{ on FOMC statement days and month-end sessions},\qquad
q_t = \mathrm{sign}(s_t) \text{ otherwise.}$$

The flags are `asl.fomc_and_monthend(index, asl.find_repo())`: `is_fomc`
(nullable boolean, NA beyond the knowledge horizon treated as False here) and
`is_me` (the last session of each calendar month). On the deck's 866 days they
mark 84 sessions: 33 FOMC, 52 month-end, 1 overlap.

**Gate.** The rebuilt frame reproduces the deck's rule table exactly: sign(s)
Sharpe 1.338322 against 1.338322, |diff| 2.2e-16. Untradeable crossed rows: 0.
866 days common to all seven forecasts, 2020-01-03 to 2024-04-30.

## The prior registration

Proposal 06 section (e), `06_intraday_rules_one_trade_per_day.md`, and §7 of the
intraday notebook. What 06 **pre-declared and ran**:

- the same two flags, from the same library call, on the intraday notebook's
  close leg — 83 of 864 days, block-diagonal ridge only;
- the midpoint fill only, plus one crossed number without an interval (1.92);
- sign(s) 1.83 → 2.30 and always short 0.19 → 0.69 on that frame; the paired
  mean difference at $t$ 1.73; the Sharpe interval [+0.02, +0.87] percentile and
  [+0.00, +0.85] basic — 06 called the basic bound "a knife edge";
- one placebo: 2000 random 83-day removals, the rule at the 99.9th percentile;
- an era split (pre-2022-05-16 1.62 → 1.82, daily-0DTE era 1.97 → 2.66) and a
  by-year table;
- the caveat that the flags were first found by tail forensics on an earlier
  version of the close trade, so 06 is a confirmation, not an out-of-sample test;
- the verdict **"needs more — adopt only after an out-of-sample era or a
  pre-registered forward test."**

What is **new here**: the deck's own 866-day frame instead of the intraday
frame's 864; both fills with paired intervals at each; all seven scored
forecasts (`asl.MODEL_ORDER` minus `blk2_inc`) instead of one; the rule applied
to always short and to the two heaviside legs; the buy/sell mean decomposition;
the information ratio; the calendar-neighbour placebos; the leave-one-year-out
and k-worst-day concentration curves; the mechanism reading of the last bar, the
close bar, the close-bar implied and the squared terminal move; and a corollary
with a genuine holdout on twenty unscored months.

What is **still not new**: the days. This is the same sample 06 confirmed on,
scored at a different aggregation. It is **not** the forward test 06 asked for
and does not become one by being more thorough. The forecast panels stop
2024-04-30; the only genuinely out-of-sample number in this file is the
corollary's holdout, which needs no forecast.

## 1. Causality of the flags

| item | number |
|---|---|
| warnings raised by `fomc_and_monthend` on this index | 0 |
| FOMC knowledge horizon (`flags.attrs["fomc_known_until"]`) | 2025-12-10 |
| NA rows for `is_fomc` inside the frame | 0 of 866 |
| `releases.parquet` "fomc release" flags | 246 distinct days, 1993-02-03 .. 2023-11-01 |
| hard-coded `asl.FOMC_STATEMENT_DAYS` | 2020-03-03 .. 2025-12-10 |
| frame ends | 2024-04-30 |
| flags re-computed identically | True |
| exchange last-sessions inside the frame vs flagged month-ends | 52 vs 52, mismatches 0 |

**Both flags are pure calendar.** `is_fomc` is a set membership test on the date
against the release file's flagged days union the hard-coded tuple; `is_me` is
the maximum date within each (year, month) of the session list. Neither reads a
price, a return, a realized variance or a forecast column. The release feed dies
2023-11-01, which is why the tuple exists; the tuple runs to 2025-12-10, well
past the frame's end, so no scored day sits beyond the horizon and no day is
scored on a defaulted flag.

The month-end flag is taken from the traded index (`sessions=None`). Because
SPXW always lists an end-of-month expiry, that is the exchange's last trading
day of the month on every one of the 52 months here — checked against the
session list in `data/spxw_spot.parquet`: 52 dates, 0 mismatches. FOMC statement
days are published a year ahead; the last trading day of a month is known from
the exchange calendar. Nothing in either flag is knowable only after 15:30.

**2020-03-16.** 2020-03-03 (the other 2020 emergency action) is a Tuesday and is
not in the traded frame — SPXW listed Monday/Wednesday/Friday before 2022-06.
2020-03-15 was a Sunday and not a session. The library's convention is the
statement day, or the first session after an off-hours statement, so the flag
falls on 2020-03-16. The statement went out around 17:00 ET on Sunday; at 15:30
on Monday it was about 22.5 hours old and public, so the flag is known before
the entry and the day is causal. It was not *scheduled* in advance, so the
strict pre-scheduled reading drops it. That day: R +0.6963, sign(s) short,
sign(s) $R'$ −0.6963 (the frame's worst sign(s) day is −5.4225). Dropping it
moves the event set 84 → 83 and the rule from 1.7834 to 1.7707 at the midpoint
and from 1.3488 to 1.3338 at the crossed spread — a difference of +0.0127 and
+0.0150. **Nothing in this report turns on the emergency day.**

## 2. Reproduction, seven forecasts, both fills

Every coordinator figure reproduces to the printed precision (block-diagonal
ridge, 866 days):

| rule | n flat | mean mid | std mid | Sharpe mid | mean crossed | std crossed | Sharpe crossed | worst | hit rate |
|---|---|---|---|---|---|---|---|---|---|
| sign(s) | 0 | 0.0947 | 1.1237 | **1.3383** | 0.0614 | 1.1203 | **0.8696** | −5.4225 | 0.5439 |
| flat on 84 event days | 84 | 0.1184 | 1.0540 | **1.7834** | 0.0891 | 1.0481 | **1.3488** | −4.6267 | 0.4931 |
| flat on month-end only | 52 | 0.1125 | 1.0710 | 1.6669 | 0.0817 | 1.0657 | 1.2171 | −4.6267 | 0.5150 |
| flat on FOMC only | 33 | 0.0996 | 1.1073 | 1.4280 | 0.0677 | 1.1033 | 0.9739 | −5.4225 | 0.5208 |
| always short | 0 | 0.0145 | 1.1276 | 0.2038 | −0.0203 | 1.1709 | −0.2749 | −10.3160 | 0.6178 |
| always short, flat on events | 84 | 0.0474 | 1.0596 | 0.7103 | 0.0168 | 1.1018 | 0.2417 | −10.3160 | 0.5727 |
| heaviside long only | 520 | 0.0401 | 0.8330 | 0.7648 | 0.0252 | 0.8018 | 0.4997 | −1.0000 | 0.1628 |
| heaviside long only, flat on events | 539 | 0.0355 | 0.8190 | 0.6881 | 0.0217 | 0.7885 | 0.4361 | −1.0000 | 0.1490 |
| heaviside short only | 346 | 0.0546 | 0.7571 | 1.1449 | 0.0361 | 0.7835 | 0.7320 | −5.4225 | 0.3811 |
| heaviside short only, flat on events | 411 | 0.0829 | 0.6679 | **1.9706** | 0.0674 | 0.6926 | **1.5446** | −4.6267 | 0.3441 |

Claim by claim: sign(s) 1.338/0.870 against 1.3383/0.8696; flat 1.783/1.349
against 1.7834/1.3488; month-end only 1.667/1.217 against 1.6669/1.2171; FOMC
only 1.428/0.974 against 1.4280/0.9739; always short −0.275 → +0.242 against
−0.2749 → +0.2417. Largest gap 0.0004.

**Where the gain comes from.** At the crossed spread the mean rises 0.0614 →
0.0891 (+45%) and the standard deviation falls 1.1203 → 1.0481 (−6.4%). This is
mostly a mean effect, not a variance effect: the rule stops taking a losing bet,
it does not mainly shrink risk.

**The legs.** The rule *hurts* the long-only leg (0.4997 → 0.4361 crossed) and
*helps* the short-only leg hard (0.7320 → 1.5446). The two legs partition
sign(s) day by day, so this says exactly where the trouble is: the shorts on
event days.

### Per forecast, crossed spread

| forecast | sign(s) | flat | ΔSharpe | percentile 95% | basic 95% | t of the daily difference | bootstrap mass above zero |
|---|---|---|---|---|---|---|---|
| baseline (HAR + calendar OLS) | 0.4909 | 0.9895 | **+0.4986** | [+0.155, +0.884] | [+0.113, +0.842] | 2.3414 | 0.9970 |
| block-diagonal ridge | 0.8696 | 1.3488 | **+0.4792** | [+0.117, +0.914] | [+0.045, +0.842] | 2.0923 | 0.9940 |
| LightGBM | 0.9818 | 1.1716 | +0.1898 | [−0.202, +0.590] | [−0.211, +0.582] | 0.6277 | 0.8275 |
| XGBoost | 1.0215 | 1.3788 | +0.3573 | [−0.018, +0.727] | [−0.013, +0.733] | 1.4470 | 0.9675 |
| lasso (causally tuned) | 0.9346 | 1.3308 | **+0.3962** | [+0.011, +0.828] | [−0.035, +0.781] | 1.6700 | 0.9770 |
| lasso (fixed 1e-4) | 1.0444 | 1.4029 | +0.3585 | [−0.044, +0.813] | [−0.096, +0.761] | 1.4286 | 0.9590 |
| elastic net (causally tuned) | 0.5449 | 0.8709 | +0.3260 | [−0.077, +0.768] | [−0.116, +0.729] | 1.4674 | 0.9460 |

Midpoint gains run +0.1581 (LightGBM) to +0.4697 (baseline); the crossed range
is +0.1898 to +0.4986, matching the coordinator's "+0.19 to +0.50". Block 21,
B 2000, rng 0.

**One correction to the brief.** The coordinator reported four of seven
percentile intervals excluding zero, naming baseline, ridge, XGBoost and tuned
lasso. At rng 0 I count **three**: baseline, ridge and tuned lasso. XGBoost's
lower bound is **−0.018**, inside zero by less than 2.5% of the interval's
width. It is a coin flip on the seed:

| forecast | seeds 0..9 whose percentile interval excludes zero | min lower bound | max lower bound |
|---|---|---|---|
| baseline (HAR + calendar OLS) | 10 | +0.1511 | +0.1752 |
| block-diagonal ridge | 10 | +0.0933 | +0.1168 |
| LightGBM | 0 | −0.2020 | −0.1678 |
| XGBoost | **4** | −0.0279 | +0.0063 |
| lasso (causally tuned) | 10 | +0.0113 | +0.0328 |
| lasso (fixed 1e-4) | 0 | −0.0496 | −0.0221 |
| elastic net (causally tuned) | 0 | −0.0789 | −0.0456 |

Three forecasts clear the bar under every seed, three fail under every seed, one
is on the line. Basic intervals exclude zero for two of seven (baseline, ridge).

### Mean decomposition of sign(s)

| days | n | buy share | E[R\|buy] | contrib buy | E[−R\|sell] | contrib sell | mean R' | t | mean R |
|---|---|---|---|---|---|---|---|---|---|
| all 866 | 866 | 0.3995 | 0.1004 | 0.0401 | 0.0909 | 0.0546 | 0.0947 | 2.4810 | −0.0145 |
| non-event | 782 | 0.4182 | 0.0940 | 0.0393 | 0.1578 | 0.0918 | **0.1311** | 3.3080 | −0.0525 |
| event | 84 | 0.2262 | 0.2112 | 0.0478 | **−0.3771** | **−0.2918** | **−0.2440** | −1.8451 | 0.3396 |
| month-end | 52 | 0.1731 | 0.5090 | 0.0881 | −0.4633 | −0.3831 | −0.2950 | −1.5803 | 0.4712 |
| FOMC | 33 | 0.3333 | 0.0337 | 0.0112 | −0.2086 | −0.1390 | −0.1278 | −0.7541 | 0.1503 |

The event-day loss is entirely the sell contribution: −0.2918 against +0.0918
elsewhere. The buy contribution on event days is *better* than elsewhere
(+0.0478 against +0.0393). The rule removes a leg that is losing, and pays for
it by also removing a leg that is winning.

### Information ratio against always short

Same 866 days, zeros on event days for the portfolio and for the benchmark:

| line | mean active | te (ann.) | IR (ann.) | t of the active mean (library lag 14) | correlation to benchmark |
|---|---|---|---|---|---|
| flat on events, both legs, midpoint | 0.0710 | 26.0011 | **0.6881** | 1.3856 | −0.2011 |
| unfiltered, midpoint | 0.0803 | 26.4467 | **0.7648** | 1.5656 | −0.0952 |
| flat on events, both legs, crossed spread | 0.0723 | 26.0453 | 0.6993 | 1.4086 | −0.1643 |
| unfiltered, crossed spread | 0.0816 | 26.4921 | 0.7766 | 1.5907 | −0.0606 |

**The information ratio falls.** Against a benchmark that sits out the same
days, the rule adds nothing: 0.6993 against 0.7766 at the crossed spread. The
rule's whole gain is the days it *stops trading*, which the benchmark also stops
trading in this comparison. It is a filter on the trade, not on the forecast.

## 3. Multiplicity, neighbours and random flats

2000 random 84-day flats, rng 0: median gain −0.0437, 95th percentile +0.2271,
maximum +0.4990 at the crossed spread (−0.0677 / +0.2018 / +0.4695 at the
midpoint).

| rule | n flat | overlap with the real flags | Sharpe crossed | gain crossed | percentile 95% | reading | random-flat percentile |
|---|---|---|---|---|---|---|---|
| flat on 84 event days (the rule) | 84 | 84 | 1.3488 | **+0.4792** | [+0.117, +0.914] | excludes zero | **99.90** |
| flat on month-end only | 52 | 52 | 1.2171 | +0.3475 | [+0.015, +0.714] | excludes zero | 99.15 |
| flat on FOMC only | 33 | 33 | 0.9739 | +0.1043 | [−0.073, +0.321] | includes zero | 80.90 |
| placebo: flat the session BEFORE each event day | 84 | 8 | 1.0798 | +0.2102 | [−0.129, +0.568] | includes zero | 93.80 |
| placebo: flat the session AFTER each event day | 83 | 8 | 0.7643 | −0.1053 | [−0.379, +0.169] | includes zero | 35.65 |
| placebo: BEFORE, real event days excluded | 76 | 0 | 1.0174 | +0.1478 | [−0.178, +0.494] | includes zero | 87.05 |
| placebo: AFTER, real event days excluded | 75 | 0 | 0.8083 | −0.0613 | [−0.313, +0.197] | includes zero | 45.30 |

**The effect is sharp on the day.** Two of 2000 random draws beat it. The day
after is null and slightly negative (45.3rd percentile with the overlap removed).
The day before is *not* clean — it carries +0.1478 at the 87.1st percentile with
zero overlap, about 31% of the rule's gain — but its interval covers zero and it
does not reach the 95th percentile. A day-before shadow of this size is what a
pre-event vol build would look like, and it is the only smear in the placebo
table. It is not enough to call the flag mis-dated: the day-of gain is 3.2 times
larger and the day-after is negative.

**Multiplicity.** Three sub-rules, and only two of them fire. FOMC alone is
null: +0.1043, interval [−0.073, +0.321], 80.9th percentile. Month-end alone
carries the rule: +0.3475, interval [+0.015, +0.714], 99.15th percentile. The
joint rule is not two effects; it is one effect plus 33 days of noise that
happen to also lose.

## 4. Concentration

### Leave one year out

| year | n days | event days | sign(s) crossed | flat crossed | gain within | gain leaving the year out |
|---|---|---|---|---|---|---|
| 2020 | 158 | 19 | 0.5582 | 0.8891 | +0.3309 | +0.5148 |
| 2021 | 158 | 20 | 0.4395 | 0.3698 | **−0.0696** | +0.6553 |
| 2022 | 219 | 20 | −0.3321 | 0.3891 | +0.7213 | +0.4275 |
| 2023 | 248 | 20 | 1.8560 | 2.8676 | +1.0116 | +0.3208 |
| 2024 | 83 | 5 | 1.8022 | 2.2477 | +0.4455 | +0.4842 |

The gain survives dropping any single year: +0.3208 (without 2023) to +0.6553
(without 2021). It is negative only inside 2021 (−0.0696), and that year is the
smallest within-year gain by a wide margin.

### The k worst event days removed

| k | days left | worst removed | its sign(s) R' | gain mid | gain crossed |
|---|---|---|---|---|---|
| 0 | 866 | — | — | +0.4451 | **+0.4792** |
| 1 | 865 | 2023-11-30 | −5.4225 | +0.3365 | +0.3731 |
| 2 | 864 | 2024-04-30 | −3.8139 | +0.2625 | +0.3014 |
| 3 | 863 | 2023-03-22 | −3.5509 | +0.1931 | **+0.2332** |
| 4 | 862 | 2023-01-31 | −3.2222 | +0.1301 | +0.1705 |
| 5 | 861 | 2022-11-30 | −2.2967 | +0.0865 | +0.1259 |
| 6 | 860 | 2023-09-20 | −2.1666 | +0.0453 | +0.0849 |
| 7 | 859 | 2022-03-31 | −1.7493 | +0.0125 | +0.0521 |
| 8 | 858 | 2021-09-30 | −1.5078 | −0.0156 | +0.0230 |
| 9 | 857 | 2022-12-30 | −1.3926 | −0.0415 | **−0.0033** |
| 10 | 856 | 2021-10-29 | −1.3725 | −0.0671 | −0.0298 |

**This is the audit's worst finding.** Three days halve the crossed gain
(+0.4792 → +0.2332, the coordinator's +0.23). **Nine days of 84 carry all of
it**: at k = 9 the gain is −0.0033 and it is negative at every larger k. Eight
of those ten worst days are month-ends. The rule is not a shift in the whole
event-day distribution; it is the removal of a handful of large losses that
happened to fall on flagged dates.

### Month-end versus FOMC

Crossed: joint +0.4792, month-end only +0.3475 (72.5%), FOMC only +0.1043
(21.8%), sum of the parts +0.4518. Month-end carries roughly three-quarters.

### The distribution of the event-day return

| days | n | mean R | t | share R>0 | q05 | q10 | q25 | q50 | q75 | q90 | q95 | mean R' | t of R' | sign(s) short share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all 866 | 866 | −0.0145 | −0.3778 | 0.3811 | −1.0000 | −1.0000 | −0.9281 | −0.3274 | 0.4486 | 1.4103 | 2.1000 | 0.0947 | 2.4810 | 0.6005 |
| non-event | 782 | −0.0525 | −1.3169 | 0.3645 | −1.0000 | −1.0000 | −0.9574 | −0.3538 | 0.3920 | 1.3975 | 1.9970 | 0.1311 | 3.3080 | 0.5818 |
| event | 84 | **+0.3396** | 2.6182 | 0.5357 | −1.0000 | −0.9840 | −0.4456 | 0.0694 | 0.7493 | 1.6768 | 2.3197 | −0.2440 | −1.8451 | 0.7738 |
| month-end | 52 | **+0.4712** | 2.6256 | 0.5385 | −1.0000 | −0.9191 | −0.3300 | **+0.1121** | 0.9240 | 2.0798 | 2.7281 | −0.2950 | −1.5803 | **0.8269** |
| FOMC | 33 | +0.1503 | 0.8896 | 0.5455 | −1.0000 | −0.9893 | −0.6742 | 0.0281 | 0.5715 | 0.9541 | 1.5023 | −0.1278 | −0.7541 | 0.6667 |

The +0.4712 month-end mean is **both** a shift and a tail. The whole
distribution moves: q25 −0.9574 → −0.3300, the median −0.3538 → +0.1121, q75
0.3920 → 0.9240, q95 1.9970 → 2.7281. A long on month-ends wins on 53.85% of
days against 36.45% elsewhere. But the mean is still tail-carried: removing the
three largest month-end returns takes the mean from +0.4712 to +0.2458, roughly
half. The forecast is short on 82.69% of month-ends against 58.18% elsewhere.

## 5. Mechanism

Bar-end stamps: the 15:30 stamp is the 15:00–15:30 bar the forecast reads, the
16:00 stamp is the traded close bar. 866 of 866 rows joined for each.

| days | n | rv last bar (×1e-6) | iv_var close bar (×1e-6) | rv close bar (×1e-6) | rv_hat (×1e-6) | close/last | close/implied | rv_hat/implied | share close > last | share close > implied |
|---|---|---|---|---|---|---|---|---|---|---|
| non-event | 782 | 2.8273 | 5.2565 | 4.0230 | 4.5869 | 1.4659 | 0.7550 | 0.9380 | 0.7967 | 0.2698 |
| event | 84 | 6.5310 | 9.4987 | 8.8874 | 7.5094 | 1.6223 | 0.7894 | 0.7146 | 0.6429 | 0.2857 |
| month-end | 52 | 3.8487 | 8.6480 | 7.9322 | 5.0397 | **2.0615** | **0.8868** | **0.6775** | 0.8654 | 0.3846 |
| FOMC | 33 | **12.7274** | 11.1229 | 8.8580 | 10.1365 | **0.6428** | 0.6759 | 0.8115 | 0.2727 | 0.1515 |

**Month-ends and FOMC days are two different objects.** Say which fails:

- **On month-ends, the forecast's information set cannot see it and the market
  underprices it too, but the forecast fails harder.** The last bar is 1.36
  times the non-event median (3.8487 / 2.8273) while the close bar is 1.97 times
  it (7.9322 / 4.0230). The forecast, which reads the last bar, lands at only
  1.10 times its non-event median (5.0397 / 4.5869). The implied does better —
  1.65 times (from `iv_var`) — but not enough: the close bar realizes 0.8868 of
  the implied against 0.7550 elsewhere, so the market's cushion is thinner too.
  The result is that `rv_hat / iv_var` falls from 0.9380 to 0.6775 exactly on
  the days when the close bar is largest: the forecast turns *more* short into
  the one bar it should be least short into.
- **On FOMC days it is the opposite: nothing is hidden, the market overprices,
  and the trade still loses.** The last bar is 4.5 times the non-event median
  (12.7274 / 2.8273) — the 14:00 statement and the 14:30 press conference are
  already in the forecast's window. The close bar is 8.8580, *smaller* than the
  last bar (median close/last 0.6428 against 1.4659), and it realizes only
  0.6759 of the implied against 0.7550 elsewhere. The forecast is right on the
  variance and the trade still loses −0.1278 a day, at t −0.7541.

### What the package actually pays on

The settled straddle pays the squared terminal move $r^2 = ((S_{close}-S)/S)^2$,
not the close bar's realized variance (proposal 13's fact).

| days | n | median r² (×1e-6) | mean r² / mean close-bar RV | median r²/iv_var | share r² > iv_var | share close-bar RV > iv_var |
|---|---|---|---|---|---|---|
| non-event | 782 | 1.9736 | 1.4057 | 0.3642 | 0.2711 | 0.2698 |
| event | 84 | 5.7395 | 1.7206 | 0.7153 | 0.3929 | 0.2857 |
| month-end | 52 | 5.9907 | 1.5754 | **0.7438** | **0.4423** | 0.3846 |
| FOMC | 33 | 5.5807 | 1.8619 | 0.7204 | 0.3333 | 0.1515 |

This closes the FOMC puzzle. On FOMC days the close bar's realized variance
beats the implied on only 15.15% of days, but the *terminal move* beats it on
33.33% — the index goes somewhere, and where it lands is what settles. The
event-day $r^2$/close-bar-RV ratio is 1.72 against 1.41 elsewhere: event days
trend within the last half hour rather than chop. The forecast is a variance
forecast; it cannot see that, and neither the flag nor the flat rule fixes it.

### Position mix, all seven forecasts

| forecast | short, all | short, non-event | short, month-end | short, FOMC | mean R' event | mean R' non-event |
|---|---|---|---|---|---|---|
| baseline (HAR + calendar OLS) | 0.6697 | 0.6573 | 0.8462 | 0.6667 | −0.2787 | 0.1059 |
| block-diagonal ridge | 0.6005 | 0.5818 | 0.8269 | 0.6667 | −0.2440 | 0.1311 |
| LightGBM | 0.6651 | 0.6560 | 0.8462 | 0.5758 | −0.0441 | 0.1191 |
| XGBoost | 0.6778 | 0.6701 | 0.8846 | 0.5152 | −0.1558 | 0.1341 |
| lasso (causally tuned) | 0.6293 | 0.6151 | 0.8462 | 0.6061 | −0.1866 | 0.1308 |
| lasso (fixed 1e-4) | 0.5866 | 0.5678 | 0.8077 | 0.6667 | −0.1532 | 0.1347 |
| elastic net (causally tuned) | 0.6455 | 0.6330 | 0.8269 | 0.6364 | −0.1585 | 0.0973 |

Every one of the seven turns short on month-ends: 80.77% to 88.46% against
56.78% to 67.01% elsewhere. All seven lose money on event days and make it
elsewhere. LightGBM loses least (−0.0441) and gains least from the rule
(+0.1898) — consistent.

### Do the baseline's calendar columns move it?

| forecast | rv_hat/iv_var non-event | rv_hat/iv_var month-end | rv_hat/iv_var FOMC | median rv_hat month-end / non-event |
|---|---|---|---|---|
| baseline (HAR + calendar OLS) | 0.8758 | 0.6690 | 0.8179 | 1.1769 |
| block-diagonal ridge | 0.9380 | 0.6775 | 0.8115 | 1.0987 |
| LightGBM | 0.8768 | 0.6710 | 0.9563 | 1.1415 |
| XGBoost | 0.8617 | 0.6616 | 0.9930 | 1.1003 |
| lasso (causally tuned) | 0.9194 | 0.6648 | 0.8248 | 1.1569 |
| lasso (fixed 1e-4) | 0.9446 | 0.6720 | 0.7604 | 1.0832 |
| elastic net (causally tuned) | 0.9189 | 0.6725 | 0.8280 | 1.1730 |

**The baseline carries an `is_month_end` column and it does almost nothing.**
The baseline's forecast rises 1.1769 times on month-ends. The implied rises
1.6452 times and the realized close bar rises 1.9717 times. A calendar dummy
worth 18% against a 97% move in the thing being forecast is not a fix; every
forecast in the table, with or without the dummy, lands `rv_hat / iv_var` near
0.67 on month-ends. The FOMC channels are equally inert: the ridge with the FOMC
columns forecasts a median 1.0260 times the ridge without them on FOMC days
(0.9901 on other days), and the two disagree about the position on 3 of 33 FOMC
days.

## 6. The corollary — pre-registered, run once, NOT adopted

**Selection warning, stated plainly.** The corollary was formulated *after*
seeing the month-end mean R of +0.4712 in section 4. It is not an independent
hypothesis: it is the same fact read as a trade. It therefore carries selection
risk and is reported, not adopted. Two cells, no variants, no grid, run once.

| rule | positions changed | Sharpe mid | vs sign(s) Δmid | percentile | vs flat Δmid | percentile | Sharpe crossed | vs sign(s) Δcrossed | percentile | vs flat Δcrossed | percentile |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 long on month-ends | 43 | 1.9969 | **+0.6586** | [+0.163, +1.244] | +0.2136 | [−0.144, +0.541] | **1.5352** | **+0.6657** | [+0.167, +1.256] | +0.1864 | [−0.167, +0.517] |
| C2 long on FOMC days | 22 | 1.4893 | +0.1509 | [−0.151, +0.525] | −0.2941 | [−0.700, +0.082] | 1.0226 | +0.1530 | [−0.149, +0.522] | −0.3263 | [−0.727, +0.039] |

C1 beats sign(s) with an interval excluding zero at both fills (+0.6657 crossed,
[+0.167, +1.256]) and beats the flat rule by +0.1864 with an interval covering
zero. C2 beats nothing.

### The holdout

The long-on-month-end leg needs **no forecast**: the position is +1 by the
calendar. The twenty unscored months 2024-05 to 2025-12 are therefore a genuine
out-of-sample test of C1, though not of the flat rule (which needs sign(s)).
Built from the chain's 15:30 quotes and the official close by the same picker
the deck uses. Two of the twenty listed month-ends are half sessions (2024-11-29,
2025-11-28) whose 15:30 row sits after the close, and are dropped, leaving 18.

**Gate:** the same builder rebuilds all 866 deck days with max |R rebuilt −
R deck| = 0.00e+00 before it is pointed out of sample.

| line | n | mean R | median | hit rate | t |
|---|---|---|---|---|---|
| long the month-end package, midpoint | 18 | **+0.4503** | −0.2398 | 0.4444 | +1.194 |
| long the month-end package, crossed spread | 18 | **+0.4315** | −0.2475 | 0.4444 | +1.158 |
| control: every session in the same window | 413 | **−0.0457** | −0.3804 | 0.3487 | −0.827 |
| in-sample month-ends | 52 | +0.4712 | +0.1121 | 0.5385 | — |
| in-sample non-month-ends | 814 | −0.0455 | — | — | — |

Placebo: 2000 random 18-session draws from the same window give a median mean R
of −0.0678 and a 95th percentile of +0.4188; the month-end draw sits at the
**96.0th percentile**.

**What it settles, and what it does not.** The *level* replicates almost
exactly: +0.4503 against the in-sample +0.4712, with a same-window control of
−0.0457 against the in-sample non-month-end −0.0455. The split is the same size
out of sample as in. But the *shape* does not replicate. The holdout median is
−0.2398 and the hit rate is 0.4444 — a long on month-ends now wins *less* often
than a coin, where in sample it won 53.85% of the time. The mean is carried by
one day: 2024-05-31 returns +5.0084, which is 61.8% of the total; without it the
mean is +0.1822 (n 17), and without the two largest it is +0.0454. Eighteen
observations at the 96.0th placebo percentile with 62% of the mean in one day is
a *supportive*, not a *settling*, holdout.

## Figure

`results/atm_straddle_intraday/proposals/18/cum_points_blk2.png` — cumulative
index points, sign(s) against flat-on-events against long-on-month-ends,
block-diagonal ridge, midpoint and crossed spread. Note the units: the figure is
in index points, the Sharpe ratios above are on the per-premium return, so the
two need not track day by day.

## Verdict

**The flat-on-event-days rule passes the standing gate for the block-diagonal
ridge and for three of the seven forecasts, and the audit finds one defect that
keeps it from being an alpha.** At the crossed spread the ridge goes from 0.8696
to 1.3488, a gain of +0.4792 whose percentile interval [+0.117, +0.914] and
basic interval [+0.045, +0.842] both exclude zero, and whose lower bound stays
positive under all ten bootstrap seeds. The baseline (+0.4986, [+0.155, +0.884])
and the causally tuned lasso (+0.3962, [+0.011, +0.828]) also clear it; XGBoost
is on the line at [−0.018, +0.727] and excludes zero under 4 of 10 seeds, so the
honest count is three of seven, not the four the brief reported; LightGBM, the
fixed lasso and the elastic net do not clear it under any seed. Causality is
clean: both flags are pure calendar, the FOMC horizon runs to 2025-12-10 against
a frame ending 2024-04-30, all 52 month-ends are the exchange's own last
sessions, and the one unscheduled day, 2020-03-16, was public 22.5 hours before
the entry and is worth +0.0150 of the +0.4792. The placebos say the effect is a
day effect: two of 2000 random 84-day flats beat it, the session after each
event day is null and negative (45.3rd percentile), and the session before
carries a third of the gain at the 87.1st percentile with an interval covering
zero — a shadow, not a mis-dating. The rule is really one flag: month-end alone
is +0.3475 at the 99.15th percentile, FOMC alone is +0.1043 at the 80.9th and
resolves nothing. The defect is concentration. Three event days halve the gain
and **nine of the 84 carry all of it** — at k = 9 the crossed gain is −0.0033 —
and eight of the ten worst are month-ends; the gain is also negative inside 2021
(−0.0696). Against an always-short benchmark that sits out the same days the
information ratio *falls*, 0.7766 to 0.6993, so nothing has been added to the
forecast: this is a rule about which days to trade, not a better signal. The
mechanism says why it works and why it will not generalize: on month-ends the
close bar realizes 1.97 times its usual size while the last bar the forecast
reads moves only 1.36 times, so the forecast cannot see it and drives
`rv_hat / iv_var` down to 0.6775 exactly when it should be up, while the implied
also underprices (close/implied 0.8868 against 0.7550); on FOMC days the
opposite holds — the last bar is 4.5 times normal, the close bar is *smaller*
than it, the implied is rich, and the trade still loses, because the package
settles on the squared terminal move, which beats the implied on 33.3% of FOMC
days against a close-bar variance that beats it on 15.2%. The baseline's own
`is_month_end` column lifts its forecast by 1.1769 against a 1.9717 lift in the
realized close bar, so the calendar channel already in the panel is not a
substitute for the rule. On the corollary: long the package on month-ends beats
sign(s) by +0.6657 crossed with an interval [+0.167, +1.256] excluding zero, but
it was formulated after seeing the month-end mean and is not adopted here; its
holdout on the eighteen scorable month-ends of 2024-05 to 2025-12 replicates the
*level* (mean R +0.4503 against +0.4712 in sample, same-window control −0.0457
against −0.0455) at the 96.0th percentile of 2000 random 18-session draws, but
not the *shape* (median −0.2398, hit rate 0.4444 against 0.5385 in sample, with
61.8% of the mean in 2024-05-31 alone). **Adopt the flat rule for the ridge and
the baseline as a risk filter on the trade; do not present it as an improvement
to the forecast, do not claim it for all seven, and state the nine-day
concentration wherever the 1.35 is quoted.** Proposal 06 asked for an
out-of-sample era or a pre-registered forward test before adoption; this study
is neither — it is the same days, more thoroughly scored — and the only genuinely
out-of-sample number in it, the corollary's holdout, is 18 observations. The
outstanding test is unchanged: score the forecast panels past 2024-04-30 and
run the flat rule forward.

## Files

- `18_event_day_flat.py`, `18_event_day_flat.md`
- `results/atm_straddle_intraday/proposals/18/`: `repro_blk2.csv`,
  `per_forecast.csv`, `seed_sensitivity.csv`, `mean_decomposition.csv`,
  `information_ratio.csv`, `subrules_placebos.csv`, `concentration_by_year.csv`,
  `concentration_k_worst.csv`, `event_distribution.csv`, `mechanism_bars.csv`,
  `mechanism_terminal_move.csv`, `position_mix.csv`, `calendar_channel.csv`,
  `corollary.csv`, `holdout_monthend.csv`, `holdout_all_sessions.csv`,
  `cum_points_blk2.png`
