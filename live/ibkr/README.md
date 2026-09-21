# `live/ibkr` — execution layer for the **afternoon** delta-hedged SPX 0DTE straddle

Sell the nearest-OTM SPXW 0DTE straddle in the **afternoon**, at a clock chosen
each session from a causal premium ledger; delta-hedge every 30 minutes through
15:30 with ES for the bulk and MES for the remainder, off the implied volatility
**corrected** by that clock's trailing realized-over-implied factor; **hold the
body to cash settlement**; flatten the futures at 16:00:00 ET; reconcile the
official settlement the next morning. Size by a stress table, never by margin.
On the last trading session of the month, sell nothing and **buy** the 15:30
straddle instead, held to settlement (§1.6). On the monthly-expiration session
(the third Friday), sell more (§1.7).

```
preflight   calendar (§1.6, §1.7)  →  ledger  →  regime (size multiplier) and
            entry clock                            (§1.5; selector, or the fixed clock)
13:30 ET    SELL the body, size = stress x m (x t on a third Friday, §1.7)
            (combo first, then the futures hedge)
14:00 ..    re-quote, re-invert, CORRECT,     (ES bulk + MES remainder)
15:30       re-hedge, journal the residual
16:00:00    flatten ES + MES, let the body    (passive 5 s, then cross)
            cash-settle; write pending_settlement.json
next am     --reconcile: official settlement → final P&L → append the ledger
```

---

## 0. Status: nothing here has touched a live socket

No TWS and no IB Gateway is installed on the development machine. Every
`IBBroker` code path in this package is **written, reviewed and type-checked but
never executed**. The replay path (`FakeBroker`) is exercised hard — including
refusals, rejections, partial fills, disconnects and stale snapshots — and that
is the only thing the green tests prove. §11 and §12 say exactly what is
untested and what was guessed.

The research verdict this package implements is the 2026-09-18 audit, sections
4–7. The short version: **the 11:00 book should not be traded as designed.** On
the 413 sessions the deck never saw, the 11:00 exit book earns +0.025 premium
units/day but **−0.02 index points per contract**; the implied/realized
remaining-window curve changed shape (0.78–0.84 at 11:00–13:00 in 2025, 1.16–1.25
after 13:30). The morning premium is gone; the afternoon/settlement premium
remains. Six afternoon clocks are positive per contract with t > 2 out of sample,
and **hold-to-settlement is the only variant with t > 2** (Sharpe 1.75, t 2.11).

Two honesty notes that belong at the top, not in a footnote:

* The **selector is a defensible causal rule, not a demonstrated improvement.**
  Proposal 46 adopted 0 of 14 cells: the selector tracks the migration about a
  quarter late, beats neither fixed 11:00 nor fixed 13:30 with an interval
  excluding zero. `--entry-mode fixed --entry-clock 13:30` is an equally
  defensible configuration and is one flag away.
* Everything in the research is measured **gross of the hedge cost**. At 0.5 bp
  the exit book nets about Sharpe 1.20; restate any headline you quote.

---

## 1. The decision layer

Four objects, each owned by one module, each causal by construction.

### 1.1 The premium ledger — `premium_ledger.py`

One row per `(session, clock)`:

| column | what |
| --- | --- |
| `implied_rem_var` | the square of the total volatility bisected out of that stamp's straddle package midpoint — the market's variance over the REMAINING session |
| `realized_rem_var` | the variance actually realized over that same window: summed squared log spot returns from the stamp through 15:30, plus the last step into the settlement |
| `spot`, `premium_mid`, `kc`, `kp` | the stamp itself, so a row can be audited back to the tape |

Two files, both read, the live one winning per session on overlap:

* `results/live_seed/premium_ledger.parquet` — built from the research tape (`--ledger`)
* `results/live_journal/premium_ledger_live.parquet` — appended by `--reconcile`, one session a day (`--ledger-live`)

Preflight checks that the merged ledger's **last session is the previous trading
session**. If it is not, the runner shouts, falls back to the **seed alone**, and
journals `ledger {ok: false, used: "seed"}`. A stale live ledger is a stale
selector, and the runner says so rather than pretending.

### 1.2 The selector — `selector.py` (proposal 46, estimator E2)

    score_c(today) = ratio_of_sums(c, today, 252) / ratio_of_sums(c, today, None)

each clock's recent premium **relative to its own long-run level**, both legs
read off sessions strictly before today. Argmax over `candidate_clocks`, ties to
the earlier clock. Dividing by the clock's own expanding level removes the fixed
effect that a one-bar remaining window is a different object from a six-bar one —
which is what made proposal 44's mean-of-logs pick 15:00 on 1216 of 1216
sessions.

Default candidates are the **afternoon**, `13:00 … 15:00`; `--candidates all`
opens the full `10:00 … 15:00`. During the warm-up (fewer than
`--selector-min-sessions`, default 63, prior sessions at every candidate) there
is no pick: the runner uses `--entry-clock` and journals
`selector {mode: "warmup_fixed", warmup: true}`.

### 1.3 Sizing — `sizing.py`, the stress table

Proposal 31 asked whether the book could be sized by margin: no — the
margin-sized row is unbounded (positions to 25,591×) and loses anyway. The
constraint that binds is the tail this book is actually exposed to, a jump in the
**last bar**, which settles before the hedge can be reset:

    contracts = floor( capital × stress_fraction / loss per contract )

    loss per contract = worse of { |settlement intrinsic| − premium }  over ±jump
                      + |delta| × jump × S            (the futures leg, charged adverse)

both in dollars at $100 an index point. The convention is deliberately
conservative and stated rather than buried: the option leg is charged its worse
side and the futures leg the adverse side of the *same* jump, although one jump
cannot be the worse side for both.

**Worked example — the 13:30 book, measured on the replay of 2025-06-20:**

| | |
| --- | --- |
| index at 13:30 | 5959.66 |
| strikes Kc / Kp | 5960 / 5955 |
| package mid | 18.30 pts |
| package delta (corrected) | +0.068 |
| option leg on a +5 % last-bar jump | $27,934 |
| futures leg on the same jump | $2,026 |
| **stress loss per contract** | **$29,961** |
| at `--stress-fraction 0.10` | **one contract per ≈ $300k of capital** |

The audit's design figure for the same book is ≈ $30,600 per contract and one
contract per ≈ $306k; the replay reproduces it to 2 %. A quiet morning is
cheaper — the same table on 2023-11-30 at 11:00 gives $21,429 and one contract
per $214k — which is exactly why the size is computed per session rather than
configured.

`--n` overrides the table. It is honoured, and the runner **prints and journals
the stress fraction that override implies** so that a hand-set size is never
silently a different risk budget.

`--max-straddles` is a hard cap on top. With no `--capital` and no `--n`,
preflight **aborts**: this runner does not guess a size. Whatever number comes
out of this table, §1.5's regime multiplier is applied to it, then on the
monthly-expiration session §1.7's multiplier, and `--max-straddles` caps the
result again.

### 1.4 The corrected delta — proposal 36's V9

The hedge delta is taken at

    corrected_total_vol(tv, ledger.realized_over_implied_mean(clock, today, window))
      = sqrt( tv² × factor )

with `window=None` (expanding) by default — the lagged mean of realized over
implied remaining variance **at that clock**, recomputed at every rebalance for
the clock being hedged. V9 is the one hedge-volatility variant whose era interval
on the primary book excludes zero (+0.176, [+0.063, +0.301]) and it survives
Bonferroni across all 24 variants. Where the factor is missing or non-positive —
the whole warm-up — the hedge uses the implied volatility unchanged, so the
corrected book hedges the same days and differs only where the correction has
something to say. `--no-delta-correction` turns it off.

### 1.5 Deleveraging (design candidate from proposal 50)

**This is a design candidate, not an adopted result.** Proposal 50 measured it in
its part B2 and adopted it nowhere; it is wired in here behind a flag so the
runner can be driven with it or without it, and because a filter that is only
described is never tested. `--no-delever` restores the old size exactly.

The rule, each session *d*, before sizing:

    RV_d        the session's realized variance
    RV21_d      the mean of RV over the 21 sessions strictly before d
    p90, p97.5  the EXPANDING quantiles of RV21 over the sessions strictly
                before d, taken only once 252 of them carry an RV21

    m_d = 1.0   RV21_d <= p90              full size
        = 0.5   p90 < RV21_d <= p97.5      half size
        = 0.0   RV21_d > p97.5             no position
        = 1.0   fewer than 252 prior sessions -- the rule is INACTIVE and the
                day is journaled `state: "warmup"`

    n_d = floor( n_stress_d x m_d )

`--n` is multiplied too: an explicit size is a full-size *intent*, not an
exemption from the regime, and the journal carries `n_stress`, `multiplier` and
`n` side by side. `n_d == 0` ends the day flat — a zero multiplier refuses it in
**preflight**, before a clock is chosen, with `deleveraging: regime above the
97.5th percentile (RV21 … > …)` on the summary and exit code **2**.

**The session RV.** The live runner has one realized-variance tape: the ledger.
`RV_d` is `realized_rem_var` at the **first entry clock, 10:00**, whose remaining
window is 10:00 through the settlement print — the session's own realized
variance over all but its first half hour. Proposal 50 measured its rule on a
different object: the 30-minute panel's `sumret2` summed over the 13
regular-hours bars, 09:30 through 16:00, each bar itself a sum of squared
**one-minute** returns. **The two do not agree**, and the numbers are these. On
the 866 chain sessions the panel also covers, the ledger's 10:00 variance against
the panel's 10:00–16:00 sum of squared **30-minute** returns has correlation
0.997 and median ratio 0.996 but a largest absolute gap of 3.9e-4 — the ledger's
last step runs into the settlement print and the panel's into its own 16:00
close. Against the panel's 09:30–16:00 one-minute RV, the object proposal 50
actually used, the median ratio is 0.743: that window is half an hour longer and
sampled thirteen times as finely. The ledger's column is what the runner *has* —
the research panel ends in 2024 and covers only 866 of the 1279 chain sessions,
and the runner never reads it. What is asserted exactly, session by session, is
the **rule**: proposal 50's `expanding_pct` helper and its three constants,
imported by path and evaluated on the same trailing-variance series, reproduce
this ledger's multiplier on all 1279 sessions
(`test_the_multiplier_matches_proposal_50_session_by_session`).

**What it does, on the replay.** Proposal 50's part B2, on its 6,593 replayed
panel sessions:

| | no rule | deleveraging rule |
| --- | --- | --- |
| full / half / zero sessions | 6,593 / 0 / 0 | **6,134 / 318 / 141** (93 % full) |
| mean, pts/session | −0.5418 | **−0.1884** |
| sd, pts | 12.95 | **9.45** (−27 %) |
| 1-in-200 session, pts | −61.75 | −47.00 |
| worst session, pts | −359.79 | −140.01 |
| **worst 20-session run**, as a fraction of the no-rule 5 % capital | **−0.99** | **−0.32** |

The mean improves, the spread falls by a quarter, and the 20-session drawdown
that took essentially the whole capital budget takes a third of it. The caveat is
the size of the claim: **this was measured on proposal 50's replay — a repriced,
frozen-strike reconstruction of 1998–2024 — and not on the realized book.** No
session of the live book has been traded through it, and 50 is an upper-bound
exercise, not a P&L.

On the shipped seed ledger (2020-01-03 … 2025-12-31, 1279 chain sessions) the
same rule gives **1223 full / 38 half / 18 zero**. It is inactive over the whole
of 2020 and 2021: the expanding percentiles begin at the ledger's own first
session, so the 252-session minimum is not met until 2021-09-27 and **March 2020
is warm-up** — this rule would not have stopped the COVID crash, because on this
history it had not yet seen one. The first half-size session is **2022-02-22**;
the first flat session is **2025-04-09**, and the replay of that day is the
canary:

```
$ python -m live.ibkr.run_day --date 2025-04-09 --entry-mode fixed \
      --entry-clock 13:30 --capital 1000000
  PREFLIGHT ABORT: deleveraging: regime above the 97.5th percentile
                   (RV21 2.0981e-04 > 2.0201e-04 over 1077 prior sessions):
                   the day is FLAT
  FLAT: deleveraging: regime above the 97.5th percentile ...          [exit 2]

$ ...  --no-delever                      # the same session, the old size
  straddles                                 6
  day P&L (premium units)             -0.3559
```

Flags: `--no-delever`; the window, the two percentiles and the 252-session
minimum are `Config.delever_window`, `delever_half_pct`, `delever_zero_pct` and
`delever_min_sessions`. Every session journals one `regime` record — `rv21`,
`p_half`, `p_zero`, `n_prior`, `state` (`full` / `half` / `zero` / `warmup` /
`disabled` / `no_ledger`) and `multiplier` — whether or not the rule bites.

### 1.6 The month-end override — `calendar_guard.py` (proposals 54 and 55)

**The rule.** On the **last trading session of a calendar month** the short
program does not sell. In the default mode it **buys** instead:

| `--month-end-mode` | on a month-end |
| --- | --- |
| **`override`** (default) | no short, no futures. At 15:30 BUY the nearest-OTM straddle at the quoted ask, one per straddle the short program would have sold that session, and hold it unhedged to cash settlement |
| `sit_out` | the day ends FLAT in preflight, before the ledger is read and before any order is built (the old guard) |
| `off` | the calendar is not consulted; the short book trades as on any other day |

`--no-calendar-guard` is kept as an alias of `--month-end-mode off`. On every
other session the three modes are the same run, and a test holds them to it.

**The evidence** (`writeup/intraday_proposals/54_month_end_close.py` and
`55_calendar_overrides_1530.py`; 70 month-end sessions of the 1279 in the
ledger, index points per contract):

| on the last session of the month | month-end sessions | every other session |
| --- | --- | --- |
| short straddle sold 11:00, held to settlement | −1.8 | +1.4 (t 5.7) |
| short straddle sold 13:30, held to settlement | −1.8 | +1.1 (t 4.7) |
| short straddle sold 14:30, held to settlement | −2.3 | +0.8 (t 3.5) |
| short straddle sold 13:30, bought back at 15:30 | −0.7 | +0.5 (t 3.0) |
| **long 15:30 straddle bought at the ask, held** | **+2.9** (+0.43 premium units, t 2.7) | −0.05 units at the mid |

The 13:30 hold book with each treatment, index points per contract per day
(the insurance-book check, same tape):

| 13:30 hold book | all 1279 sessions: mean / Sharpe | holdout 2024-05 .. 2025-12: Sharpe |
| --- | --- | --- |
| as is (short on month-ends too) | +0.92 / 1.81 | 2.06 |
| sit out month-ends | +1.02 / 2.07 | 2.49 |
| **override: sit out and buy the 15:30 straddle** | **+1.18 / 2.26** | **2.55** |

Month-end rebalancing is executed at the closing auction: since 1998 the last
half hour's variance is 1.5× the bar before it on month-ends against 1.1×
otherwise (t 7.9, 264 month-ends before any option in the study), and the
option market does not price it — the long straddle is cheap on exactly the
day the short is expensive. On the 866-day deck proposal 55 forced the 15:30
sign(s) trade long on month-ends: +0.67 crossed Sharpe, interval [+0.15, +1.21],
both placebos p ≤ 0.001, on all eight forecasts.

**How it is sized** (`month_end_long_size="match_short"`, the only value
offered). At the clock the selector picked, the runner quotes the straddle the
short program would have sold and runs the short program's own stress table —
capital × fraction over the stress loss per contract, capped at
`--max-straddles`, `--n` overriding it — and buys that many at 15:30. The
**deleveraging multiplier is NOT applied**: it is a brake on short-tail risk and
the long can lose only its premium; the regime is still journaled, a zero
multiplier does not refuse an override day, and the sizing record carries
`multiplier_not_applied`. A hard check refuses the buy if the premium outlay
(`n × ask × 100`) exceeds the loss budget (`capital × stress_fraction`); with no
`--capital` there is no budget and no long.

At $1m and 10 % on the 70 replayed month-ends (seed ledger, selector clock)
this buys 2–9 straddles, mean 4.3; the premium outlay is at most 26 % of the
$100k loss budget, so the check never binds at the default fraction.

**The untested alternative, not enabled.** Sizing the long to the same loss
budget as the short — the long's worst case is its premium, $250–$2,910 per
straddle on those days, against the short's stress loss of $10–36k (median
$22k) per contract — would buy a **median 28×** as many straddles (mean 31×,
range 4–96×). Nothing here says the 15:30 book is that deep, so it is
described, not offered.

**The order.** One limit at the quoted **ask**, placed with
`max_cross_ticks=0`: it rests for `passive_wait_s` and is cancelled if it has
not filled. It never chases. No valid two-sided quote at 15:30, or no fill,
ends the day FLAT with the reason journaled; a partial fill is a real position
and is held to settlement. The long settles through the same provisional-settle
→ `--reconcile` path as the book; the summary takes its sign from the entry
fill's action (a `body_entry` BUY), so the P&L is `payoff − ask`.

**The ledger does not get a hole.** From the clock the selector picked through
15:30 the override quotes and inverts the nearest-OTM straddle at every stamp,
without an order, and hands those rows to the morning in
`pending_settlement.json` exactly as the book does — even on a day it ends
flat. (A `sit_out` day, like any preflight refusal, leaves no rows.)

**Parity.** `python -m live.ibkr.parity --month-end` replays every month-end
through `DayRunner` + `FakeBroker` in override mode at `--n 1`, reconciles it at
the official close the research settles at, and gates it against proposal 54:
**70 of 70** month-ends traded, per-contract P&L max |difference| 3.6e-15
points, premium units 4.4e-16; mean +2.9306 points / +0.4301 units. The replay's
provisional settlement (the tape's 16:00 print) differs from the official close
by 0.90 points a day on average — that is what `--reconcile` is for.

**Its limits.** Month-end was **found by inspecting the sample** (the book's
single worst day, 2023-11-30, is one), so the deck-period numbers were not a
test. The data that played no part in finding it agree: the underlying before
2020, and the **18 held-out month-ends** from 2024-05, whose long-straddle mean
at the ask is the same +0.43 (t 1.2 on 18 days — the same mean, not a
significant one). 2025's month-ends are flat (−0.02). Every `IBBroker` path is
untested (§11), and **buying at 15:30 is a new order path for this package**:
walk it on paper before trusting it.

**The calendar.** Live trading needs the answer for a *future* date, so the
calendar reads no ledger: `is_last_session_of_month` computes the NYSE session
calendar from the exchange's holiday rules (Rule 7.2 observance, Good Friday
from `dateutil.easter`). Neither `exchange_calendars` nor
`pandas_market_calendars` is installed in the `285J` environment and none is
needed — `dateutil` already ships with pandas. Two parity checks pin it
(`tests/test_calendar_guard.py`):

* on the seed ledger's 1279 sessions the rule flags **70** month-ends, exactly
  proposal 54's empirical 70, with **0** disagreements. The ledger spans 72
  months; the two missing month-ends are 2024-11-29 and 2025-11-28, 13:00
  half sessions the ledger never carried;
* against 26 years of the 30-minute panel (1998-01-05 .. 2024-04-30): no day
  the rule calls a holiday has a 16:00 bar, and every day it calls a session
  without one is a 13:00 early close.

**The calendar's limits.** A *special* closure announced at short notice (a
day of mourning, a storm) cannot be computed. The ones on record are listed in
`SPECIAL_CLOSURES`; a new one has to be added by hand. The failure is
one-sided: a special closure on a month's last weekday would make the calendar
miss the true last session, the day before. A half session that is also a
month's last session is refused earlier, by the liquid-hours check, so it ends
flat in every mode — the override is never tried on a 13:00 close. A position
found open in preflight outranks the calendar — that is the louder alarm.

**The shape.** `NO_SHORT_CALENDARS` is a registry of *named* calendars and
`Config.no_short_calendars` picks from it (`("month_end",)` today). Another
class of days is a registry entry plus its evidence — for **both** halves of
the override: that the short loses into that close and that the long is paid
for it. Proposal 55 tested FOMC days and found the second half missing (the
long earns +0.10 premium at the ask, t 0.6), so none is registered.

```
$ python -m live.ibkr.run_day --date 2023-11-30 --capital 1000000
  *** month-end override: 2023-11-30 is the last trading session of the
      month -- no short into this close; the runner BUYS the 15:30 straddle
      and holds it to cash settlement ***
  side                         LONG (month-end)
  entry clock                           15:30
  straddles                                 4   (the short's stress count at 14:30)
  entry premium (pts)                  3.6500
  exit (settlement, provisional) pts   23.7900
  day P&L ($)                        +8056.02
  day P&L (premium units)             +5.5178
  # reconciled at the official close 4567.80: +19.1498 pts per straddle,
  # +5.2465 units -- proposal 54's number

$ ...  --month-end-mode sit_out
  PREFLIGHT ABORT: calendar guard: 2023-11-30 is the last trading session of
                   the month -- the short book does not hold into this close:
                   the day is FLAT                                    [exit 2]

$ ...  --month-end-mode off              # the short book, as on any day
  entry clock                           14:30
  straddles                                 4
  day P&L (premium units)             -4.7275          (-$10,778.81)
```

Every session journals one `calendar` record — `mode`, `session`,
`calendars`, `hits`, `decision` (`short` / `sit_out` / `override`), `flat`,
`override`, `reason`, `evidence`, `long_size`, and the third-Friday fields of
§1.7 — whether or not it bites.

### 1.7 The third-Friday size — `calendar_guard.py` (proposal 58; multiplier from study 66)

**The rule.** On the **monthly-expiration session** — the third Friday of the
month, or the session before it when that Friday is a holiday (Good Friday
2025-04-18 → Thursday 2025-04-17; Juneteenth 2026-06-19 → Thursday 2026-06-18)
— the short program sells more. In the order the sizing record journals it:

    n_stress  = min( stress table, --max-straddles )    (or --n)          §1.3
    n_d       = floor( n_stress × m_d )                 the brake         §1.5   n_before_third_friday
    n         = min( floor( n_d × t ), --max-straddles )                         n

| flag | effect |
| --- | --- |
| `--third-friday-multiplier t` | the multiplier `t` (default **1.1**, `config.THIRD_FRIDAY_SIZE_MULTIPLIER`) |
| `--no-third-friday` | `t = 1`: the ordinary size every session |

* **The month-end calendar is read first** (§1.6). `t` scales a SHORT day only:
  an override or sit-out day is never scaled. No real session is both — the
  expiration session falls on the 14th–21st — and a test with a stand-in
  calendar holds the order anyway.
* **`--n` is never multiplied.** A hand-set count is the count (it is still
  braked, §1.5); the calendar record says `--n sets the size`.
* **`--max-straddles` caps after the multiplier**, so `t` can never lift a day
  above it.
* **The brake still wins.** A zero regime ends the day flat in preflight; a count
  the brake floors to zero stays zero (`0 × t = 0`).
* `t` must be a finite number ≥ 1; the product is floored in decimal arithmetic
  (`sizing.scaled_contracts`: in binary, 25 × 1.16 floors to 28).
* The ES/MES split is recomputed from the new count at every rebalance.

**Floored, 1.1 changes nothing below ten straddles** — `floor(n × 1.1) = n` for
`n ≤ 9` — and at the default `--max-straddles 10` the cap takes an 11 back to 10.
So at the default cap the default multiplier never changes a count; it acts only
with `--max-straddles` above 10 and a table count of 10 or more (at 10 % and a
≈ $29k stress loss, about $2.9m of capital). Study 66 measured it as a continuous
fraction of capital (+0.17 %/yr on the 2020–25 live book, 9.49 % → 9.66 %);
whole contracts deliver that only at size.

**The evidence** (`writeup/intraday_proposals/58_third_friday_short.py`, 72
expiration sessions of the 1279 in the ledger; index points per contract, or
premium units at the quoted bid):

| | third Friday | every other session |
| --- | --- | --- |
| short straddle sold 13:30, held to settlement | **+3.00** (t 5.3) | +0.80 (t 3.4) |
| short straddle sold 11:00, held | +2.97 (t 3.4) | +1.10 (t 4.4) |
| 15:30 straddle sold at the bid, held | **+0.33 units** (t 3.9) | — |
| … deck 2020-01 .. 2024-04 / holdout 2024-05 .. 2025-12 | +0.25 (52) / +0.55 (20) | |

By year (15:30 at the bid, 12 sessions each, 2020 → 2025): −0.08, +0.26, +0.32,
+0.43, +0.64, +0.43. The sharp placebo does not carry it on all sessions: the
other Fridays of the month +0.04 (t 0.5), the third Thursday +0.05, the third
Wednesday −0.16 — the expiration, not the Friday (the holdout's first Fridays,
+0.51 on 19 sessions, are the one exception). The lead met all four criteria
written before 58 ran.

**Where 1.1 comes from** (`writeup/intraday_proposals/66_kelly_sizing.py`,
`results/atm_straddle_intraday_holdclose/proposals/66/c_multiplier.csv`). Log
growth adds across days and the day type is known in advance, so the Kelly
fraction can be set per day type. The ratio of Kelly fractions, third Friday
over other sessions, with the pre-2020 replay tail, is **1.86** at the point
estimate; the rule written before 66 ran takes the **lower 2.5 % end** of its
bootstrap interval (1.15), rounded down to a tenth, floored at 1: **1.1**.

**Its limits.**

* **Both samples had been seen** before 58's criteria were written: proposal 56
  had printed the third-Friday rows of the deck *and* the holdout (on the long
  side, where they failed in the opposite direction). 58 is a test of a lead
  found in the data, not an out-of-sample confirmation. The next truly new
  evidence is 2026.
* **The close is not calmer on third Fridays.** 58's mechanism check: over
  2020–2024 the last half hour's realized variance was 1.25× its own trailing
  median on third Fridays against 1.02× on the other Fridays (t 1.2; before 2020,
  0.91 against 1.03). The tail per contract is **not** smaller on those days, so a
  larger count is a proportionally larger stress loss: at `t` the day's stress
  budget is `t × --stress-fraction` of capital (the sizing record's
  `fraction_implied` shows it).
* 58 and 66 measured the **hold** book. `--terminal flatten` gets the same
  multiplier with no evidence of its own for the exit variant.
* Every `IBBroker` path is untested (§11); this rule adds no new order path —
  it changes a count.

**The calendar.** `is_third_friday_session` walks back from the third Friday to
the first session on or before it, on the same rule-based NYSE calendar as §1.6
(no ledger is read). Parity (`tests/test_calendar_guard.py`): on the seed
ledger's 1279 sessions it flags **72**, exactly 58's, 12 in each year 2020–2025,
**0** disagreements with 58's own `nth_weekday_sessions` imported by path; two
of them are Thursdays (2022-04-14 and 2025-04-17, Good Friday weeks). Checked
once by hand, not in the suite: on 58's full empirical calendar (the panel's
16:00 days joined with the chain's, 1998-01-05 .. 2025-12-31, 7,024 sessions)
the rule and 58 flag the same 336 sessions.

```
$ python -m live.ibkr.run_day --date 2025-06-20 --capital 1000000
  third-Friday size: 2025-06-20 is the monthly-expiration session -- the short
  program's contract count is multiplied by 1.1
  straddles                                 3     (3 x 1.1 = 3.3, floored)
  day P&L ($)                         +750.94     (the §5 replay, unchanged)

$ ...  --third-friday-multiplier 2
  straddles                                 6     (fraction_implied 17.3 %)
  day P&L ($)                        +1479.87     (+0.2183 units)

$ ...  --capital 4000000 --max-straddles 20      # 13 in the table
  straddles                                14     (13 x 1.1 = 14.3)
  day P&L ($)                        +3473.83     (--no-third-friday: 13, +3270.24)
```

The `calendar` record carries `third_friday`, `third_friday_multiplier` (the
configured `t`), `third_friday_applied` (the `t` used: 1 off an expiration, on a
non-short day, or under `--n`), `third_friday_reason` and
`third_friday_evidence`; the `sizing` record carries `n_stress`, `multiplier`
(the brake), `n_before_third_friday`, `third_friday_multiplier`,
`third_friday_capped` and `n`.

---

## 2. The daily cycle

```bash
# during the session
python -m live.ibkr.run_day --mode paper --capital 1000000 --terminal hold

# the next morning, before the next session
python -m live.ibkr.run_day --mode paper --reconcile
# or, if the broker cannot supply the official print:
python -m live.ibkr.run_day --mode paper --reconcile --settlement 5966.93
```

**During the session.** Preflight → selector → entry → a rebalance at every half
hour through 15:30 → at 16:00:00 flatten ES and MES (passive five seconds, then
cross) → journal the **provisional** settlement (the last index print) → write
`<journal-dir>/pending_settlement.json`.

`pending_settlement.json` carries everything the morning needs: the date, the
journal path, the entry clock, the strikes, the entry fill, the provisional
settlement, and the **observed path** — one row per clock the runner actually
saw, with its spot, its total volatility and its package mid.

**The next morning, `--reconcile`.** Reads the pending file, gets the official
settlement (SPXW is PM-settled, so it is the official 16:00 SPX close) from the
broker's daily bar, or from `--settlement` when the broker cannot supply it,
then:

1. journals a `reconcile` record — the payoff at the official print, next to the
   provisional one;
2. rewrites `<date>_summary.json` from the journal, now with `provisional: false`;
3. builds this session's `ClockRecord`s from the observed path (implied variance
   = `total_vol²`; realized variance = the summed squared log steps from that
   clock through the settlement) and **appends them to the live ledger**;
4. renames the pending file to `pending_settlement.<date>.json`, so the same day
   cannot be reconciled twice.

It returns 0 only if the day is flat. Without an official settlement it finalises
**nothing** and returns 2.

### The flatten variant

`--terminal flatten --exit-clock 15:30` keeps the old book: buy the body back as
one combo at the exit clock, then flatten the futures in the same minute. It is
retained, it is tested, and it is not the book of record.

---

## 3. What to install

| Thing | Setting |
| --- | --- |
| TWS or IB Gateway | Gateway is preferred for an unattended run (no chart GUI, fewer restarts) |
| API | Global Configuration → API → Settings → **Enable ActiveX and Socket Clients** |
| Socket port | **7497** paper, **7496** live; IB Gateway is **4002** paper, **4001** live |
| Read-Only API | **off** — orders must be able to go out (the runner still connects `readonly=True` in `--mode dry`) |
| Trusted IPs | add **127.0.0.1** |
| Master API client ID | leave blank; this runner uses `--client-id 17` |
| Auto-restart | set the daily restart *outside* 09:30–16:15 ET |
| Precautionary settings | raise or clear the "size limit"/"percentage limit" warnings, or combo orders will sit in a confirmation dialog |
| Python | `ib_async == 2.1.0` (the `ib_insync` successor), plus `pandas`, `pyarrow`, `numpy` |

### Market data subscriptions

All three are required; missing any one makes the run refuse in preflight or,
worse, quote garbage.

* **OPRA (US Options Exchanges)** — top of book for SPXW.
* **CME Real-Time (NP,L2)** or at least CME Level 1 — ES/MES quotes for the
  hedge. Without it `futures_reference()` returns NaN, every hedge order is
  refused **and the kill switch disarms itself and says so** (it will not
  compute a loss it cannot mark).
* **S&P Index Data / Cboe Streaming Market Indexes** — the **SPX index itself**.
  The book picks strikes off the index, not off the futures.

Delayed data is not enough. `spx_spot()` reads `last`/`markPrice` only and
**never falls back to `close`** — a delayed feed now looks like an outage, which
is what it is.

### Account permissions

* **Uncovered index options** (a short straddle is a naked short call plus a
  naked short put) — IB permission level 4.
* **Futures** (CME) for ES **and** MES: the lot rule uses both.
* **Portfolio margin** — strongly recommended. Under Reg-T the naked short index
  straddle ties up several times the margin portfolio margin asks for. Note the
  sizing here does not consult margin at all (§1.3); margin decides whether the
  size *fits*, the stress table decides what it *is*.
* If you cannot get uncovered permission, run the **defined-risk variant**:
  `--wings 0.015`. The wings are *not* quotable in replay (§5), so the fly can
  only be validated forward in paper; proposal 45 also priced them on the
  afternoon book and found 0 of 10 cells worth it (1 % wings cap 3.9 units of
  tail for −1.27 Sharpe).

---

## 4. The commands

```bash
# 1. replay a recorded session (no network, simulated clock) — the smoke test
#    (2023-11-30 is a month-end session: by default it is the month-end
#    override, a long 15:30 straddle; --month-end-mode off replays the short book, §1.6)
python -m live.ibkr.run_day --date 2023-11-30 --entry-mode fixed --entry-clock 11:00 --n 1 --month-end-mode off
python -m live.ibkr.run_day --date 2023-11-30 --capital 1000000
python -m live.ibkr.run_day --date 2025-06-20 --capital 1000000

# 2. connected, but no orders: read-only socket, every order logged and a
#    synthetic fill at the limit returned
python -m live.ibkr.run_day --mode dry --capital 1000000

# 3. paper
python -m live.ibkr.run_day --mode paper --capital 1000000

# 4. live — needs the env var AND the explicit flag AND, at a terminal, a typed
#    confirmation, or it refuses to build
HARXHAR_LIVE=I_UNDERSTAND python -m live.ibkr.run_day --mode live --capital 1000000
```

Decision-layer flags: `--entry-mode selector|fixed`, `--entry-clock HH:MM`,
`--candidates afternoon|all`, `--selector-window`, `--selector-min-sessions`,
`--ledger`, `--ledger-live`, `--no-delta-correction`, `--no-delever`,
`--month-end-mode override|sit_out|off` (`--no-calendar-guard` = `off`),
`--third-friday-multiplier t` (`--no-third-friday` = 1).
Book flags: `--terminal hold|flatten`, `--exit-clock`, `--wings`.
Sizing flags: `--capital`, `--stress-fraction`, `--stress-jump`, `--n`,
`--max-straddles`.
Plumbing: `--mode`, `--port`, `--client-id`, `--account`, `--journal-dir`,
`--date`, `--fill`, `--reconcile`, `--settlement`.

**The live gate keys on the mode AND the port.** 7496 and 4001 are live
listeners: `--mode paper --port 7496` is **refused**, and so is `--mode live`
pointed at 7497. `--mode live` additionally needs `HARXHAR_LIVE=I_UNDERSTAND` in
the environment and `--mode live` (or `--mode=live`) typed on the command line;
at an interactive terminal it also asks for a typed confirmation. In paper mode
preflight asserts the account id is a `DU*`.

Exit codes: `0` flat and done (a month-end override that settled, or that
refused its buy and stayed flat, is a `0`) · `1` fatal · `2` preflight refused (a
half session, an open position, no size — or the deleveraging rule, §1.5; or a
`sit_out` month-end, §1.6; or nothing to
reconcile) · **`3` a position or a futures leg is still open** — a supervisor
must treat 3 as "go and look".

---

## 5. What replay can and cannot do

`--date YYYY-MM-DD` runs the whole session against `data/spxw_chain.parquet`
through `FakeBroker`, with a simulated clock and no socket at all.

**Can.** Any session from 2020-01-03 to 2025-12-31. Any stamp the tape carries —
09:30 through 15:30 on the 30-minute grid, plus the 16:00 settlement print — so
every afternoon entry clock and the hold terminal both replay. One filtered chain
read per session (pyarrow pushes `expiration == date` into the file), cached
under `results/live_replay_cache/<date>_<hash>.parquet`, where the hash is over
the chain path and the band — **not** in the live journal tree, and not keyed on
the date alone.

It can also be told to go **wrong**: `FakeFaults(refuse=…, reject=…, partial=…,
disconnect_at=…, stale_at=…)` makes any order rest unfilled, come back
`Rejected`, fill part of its quantity, drop the socket at a clock (with a
configurable reconnect delay, so lateness is exercised) or serve a stale
snapshot. Those five are what the test suite drives.

**Cannot.** Wings: the tape is cut to a ±1 % band, so 1.5 %-of-spot wings have no
quote and `--wings` ends the day flat in replay by construction. Intermediate
clocks: the tape is 30-minute, so `--exit-clock 15:40|15:50` is refused with
`--date`. Its futures are a simulation: one price for ES and MES at the index
**plus a basis** (never the index itself — marking the hedge at SPX is the
phantom-P&L bug the audit ranked fifth), filled at that reference with no
slippage unless you ask for some.

Two replays, `--fill cross` (sell the package at its bid, buy it at its ask),
against the shipped seed ledger:

```
$ python -m live.ibkr.run_day --date 2023-11-30 --entry-mode fixed       --entry-clock 11:00 --n 1 --month-end-mode off

day summary  2023-11-30   book=afternoon-hold-to-cash-settlement  mode=dry
  entry clock                           11:00
  straddles                                 1
  strikes (Kc/Kp)                   4555/4550
  entry premium (pts)                 11.2000
  exit (settlement, provisional) pts   13.7900
  option P&L (pts)                    -2.5900
  hedge P&L (pts)                    -25.7220
  day P&L ($)                        -2831.21
  day P&L (premium units)             -2.5279
  OPEN futures                              0
  worst residual delta         -0.024 @ 13:00

$ python -m live.ibkr.run_day --date 2025-06-20 --capital 1000000

  [selector] 14:30   scores 13:00 0.809  13:30 1.012  14:00 1.054
                            14:30 1.069  15:00 1.004
day summary  2025-06-20   book=afternoon-hold-to-cash-settlement  mode=dry
  entry clock                           14:30
  straddles                                 3
  strikes (Kc/Kp)                   5975/5970
  entry premium (pts)                 11.3000
  exit (settlement, provisional) pts    3.0698
  option P&L (pts)                    +8.2302
  hedge P&L (pts)                     -5.7271
  futures fills                             7
  rebalances                                3
  stress loss / contract ($)           28,812
  day P&L ($)                         +750.94
  day P&L (premium units)             +0.2215
  OPEN futures                              0
  worst residual delta         +0.007 @ 15:30
```

The second one is the whole design in one screen: the ledger chose 14:30 (not
the configured fallback), the stress table chose three straddles off $1m at 10 %,
the hedge ran to 15:30 in ES + MES, and the body cash-settled.

2023-11-30 is the book's worst day and the whole reason the exit variant exists:
the same session flattened at 15:30 ends at **−0.3381** premium units instead of
−2.53. It is also the reason the hold book is sized by the stress table rather
than by conviction. (With `--no-delta-correction` the same hold replay is
−2.4805: the V9 factor moves the hedge, and only the hedge.) It is also the
last trading session of November 2023, which is why the replay above needs
`--month-end-mode off`: by default the runner now sells nothing that day and
buys the 15:30 straddle instead (§1.6), which made +5.52 premium units.

---

## 6. The fee and lot rule: ES for the bulk, MES for the remainder

Against `n` straddles, one **ES** is `50 / (100 n) = 0.5/n` straddle-deltas and
one **MES** is `0.05/n`. Rounding to a whole contract therefore leaves at most
half of that:

| hedge | granularity | residual after rounding | ticket cost at n = 4 |
| --- | --- | --- | --- |
| ES only | 0.5 / n | 0.25 / n | 4 × $0.85 = **$3.40** |
| MES only | 0.05 / n | 0.025 / n | 40 × $0.25 = **$10.00** |
| **ES bulk + MES remainder** | **0.05 / n** | **0.025 / n** | ≤ 4 ES + 9 MES ≈ **$5.65** |

`target_lots` takes the integer part **toward zero** of the dollar delta divided
by the ES multiplier — so the big contract never overshoots — and rounds the
remainder into MES. You get MES granularity at close to ES fees, and the runner
places the two legs as two orders and journals each with its own symbol and
multiplier.

The previous README claimed "at n = 4, ES granularity (±0.06) is as good as MES
at n = 1". **That was false** by its own formulas — ES at n = 4 is 0.0625 against
MES's 0.025, 2.5× coarser; parity with MES-at-one needs ES at n = 10. The fee
half of the claim was right. The lot rule above is the fix.

---

## 7. The canary sequence

Do these in order. Each one closes a specific guess in §12; do not skip to the
next until the previous one printed what it should.

1. **Replay.** `--date 2023-11-30 --entry-mode fixed --entry-clock 11:00 --n 1
   --month-end-mode off` and `--date 2025-06-20 --capital 1000000`. Both must end
   flat, both must write a summary and a `pending_settlement.json`. Then
   `--date 2023-11-30 --capital 1000000` with no mode flag: it must sell
   nothing, buy 4 straddles at 15:30 at 3.65 and settle them, `side LONG`; and
   with `--month-end-mode sit_out` it must refuse in preflight with the
   month-end reason and exit 2.
2. **Reconcile the replay.** `--date 2025-06-20 --reconcile` (the fake serves the
   16:00 print as the "statement"). The summary must lose `provisional` and the
   live ledger must gain that session's rows.
3. **Connect dry.** `--mode dry`. Preflight must name the account, the SPXW
   chain, the ES **and** MES front months, the liquid hours (16:00), and a live
   SPX print. This closes guesses #1 (SMART routing) and #5 (front month).
4. **One paper MES round trip.** Buy one MES and sell it back by hand, or with
   `--mode paper --capital <small>` on a day you are willing to lose. Check the
   limit landed at the touch and the fill came back with a status. Guesses #6, #8.
5. **One paper bag SELL, one straddle, odd nickel.** Watch three things: the leg
   directions IB reports, the **sign of `avgFillPrice`**, and whether a limit at
   an odd nickel above $3.00 is rejected. Guesses #2, #3, #4 — and #4 is the one
   that matters, because a rejected *exit* is the failure mode this whole package
   is built around. Then **one paper bag BUY at the ask with no crossing** — the
   month-end override's order: it must rest, fill or cancel after
   `passive_wait_s`, and never be repriced.
6. **A full paper session, `--terminal hold`.** Let it run to 16:00:00 and
   reconcile it the next morning against the real statement. Compare
   `<date>_summary.json` against IB's activity statement line by line.
7. **Only then**, live, at `--n 1` or the smallest size the stress table allows,
   for a week, with the journal read every morning.

---

## 8. The morning reconciliation

Every session leaves two files next to each other:

* `results/live_journal/<date>.jsonl` — every decision, in order, one JSON per
  line, flushed **and fsynced** after each. A second run of the same date rolls
  to `<date>.1.jsonl` rather than appending to a finished journal.
* `results/live_journal/<date>_summary.json` — the day recomputed **from the
  fills alone**, so it is an independent check on IB's statement.

The ritual, in order:

1. `python -m live.ibkr.run_day --mode paper --reconcile` (or with
   `--settlement`), then read the printed summary.
2. **`OPEN futures` must read `0`.** It is printed on every summary, always, and
   shouts when it is not zero. If it is not zero the P&L line is still computed —
   but marked at the futures reference the runner journaled, with
   `incomplete: "open futures … marked at …"` — and the leg is yours to flatten
   by hand.
3. **`straddles open` must be absent.** If a straddle is still short, the summary
   NaNs the P&L rather than reporting one, and the journal carries
   `manual_flatten_required`.
4. Check `worst residual delta` against `--max-residual-delta` (default 0.6) and
   the `late_clock` alerts: the research says a one-bar lag zeroes the edge.
5. Reconcile `day P&L ($)` against IB's activity statement. They are two
   independent computations of the same number; a divergence is a bug in one of
   them, not a rounding difference.

What the summary refuses to do, each because it once produced a confident wrong
number: mark an open futures leg at **zero**; count a **rejected** order (it
contributes on its filled quantity, so a rejection contributes nothing and a
partial contributes exactly what it filled); count the **same order twice**
(fills are deduplicated by `order_id`); default a **multiplier** (without a
`config` record there is no P&L at all).

---

## 9. Kill switches and the safety batch

* **`--max-loss-units` (default 6.0)** — mark-to-market loss in premium units,
  with the futures leg marked at the **futures reference**, not at the index. If
  there is no futures reference the switch **disarms and says so**: an unknown
  loss is not a zero loss.
* The kill buys the body back first and flattens the futures **only if that
  fill came back complete**. An unfilled or partial kill leaves the hedge on,
  alerts, and the session keeps hedging the open straddle — it re-attempts the
  buy-back at every remaining clock. Stripping the hedge off an open short
  straddle into the close is the accident this book exists to avoid.
* **`--max-residual-delta` (default 0.6)** — a monitoring alarm on the level of
  unhedged delta, not a trading rule.
* **A non-invertible volatility never becomes a delta of zero.** The runner
  carries the last total volatility scaled by `sqrt(h_now/h_prev)` if it can, and
  the **last delta** if it cannot, and alerts either way.
* **A stale, halted or crossed snapshot is not traded on.** Every `quotes()` call
  returns a `QuoteHealth`; a sick one skips the rebalance and refuses the entry
  and the kill.
* **Every market-data request has a timeout** (10 s). A stuck snapshot raises
  rather than hanging the runner past its clock with a position on.
* **`positions()` never swallows an exception.** A position report that failed
  aborts preflight; it never reads as "flat".
* **Everything after the entry is inside a `try/finally`** that journals a `mark`
  (the futures reference every open lot is marked at) and, if a straddle is still
  open, prints and journals `MANUAL FLATTEN REQUIRED` — and does **not** touch
  the futures.

---

## 10. The two non-negotiables

1. **No market orders.** Every order rests at a limit and is repriced at most
   `ceil(spread / tick) + 1` ticks through it, capped by `--max-cross-ticks`
   (default 12). The cap is sized off the **live spread**: the old fixed
   `2 × $0.05 = $0.10` was below the spread of the module's own fixture day at
   both ends, which made "the exit did not fill" the expected path rather than an
   edge case.
2. **The option tick follows the price.** SPXW quotes in $0.05 below $3.00 and
   $0.10 at and above it. A package at 6.475 rounds to **6.50**, not to the
   sub-tick 6.45 that IB rejects outright.

---

## 11. What is untested without TWS

Everything that needs a socket, which is all of `IBBroker`:

* `connect` / `ensure_connected` and the `readonly` flag in dry mode
* `reqTickers` (and its `asyncio.wait_for` timeout), `reqSecDefOptParams`,
  `reqContractDetails`, `reqHistoricalData`
* bag construction, `placeOrder`, the reprice loop, `cancelOrder`, and every
  `orderStatus` field the `Fill` reads (`status`, `filled`, `avgFillPrice`)
* `positions()`, `accountSummary()`, `liquidHours` / `timeZoneId` parsing, the
  futures front-month scan and its roll rule
* `official_settlement` — the daily SPX bar `--reconcile` reads when no
  `--settlement` is passed
* the whole `--wings` iron-fly path end to end: the replay serves a ±0.75 % band,
  so a 1.5 % wing is never quotable and the four-leg bag is never built. The
  refusal is clean and tested; the fly itself is not.
* the month-end override's BUY at the ask with `max_cross_ticks=0`: a bag BUY
  that rests and cancels without repricing has never been sent to IB.

The 220 green tests exercise the replay, the journal, the summary arithmetic, the
config gates and the decision layer. They say nothing about IB.

---

## 12. The guessed ib_async details, ranked by the audit

| # | guess | if wrong | canary |
| --- | --- | --- | --- |
| 1 | **option tick by price level** — 0.05 below $3.00, 0.10 at/above, applied to the package limit | a limit on an odd nickel is rejected outright; an **exit** rejection is the worst case in this package | §7 step 5 — one paper limit at an odd nickel above $3 |
| 2 | **`NonGuaranteed` on the SMART combo** (`smartComboRoutingParams`) | if SMART will not leg the bag without it, every combo is rejected, including the exit | §7 step 5, first paper bag |
| 3 | **bag orientation and a positive limit** — `action="SELL"` at a positive package price; `avgFillPrice` assumed positive | if IB returns it credit-signed the entry looks unfilled with a real short straddle open (the runner now branches on `quantity`, not on the price, which blunts this) | §7 step 5 — read the leg signs and the sign of `avgFillPrice` |
| 4 | **`exchange="SMART"` for SPXW**, with a CBOE fall-back | SMART empty and CBOE empty is reported as "not an SPXW expiry"; the fall-back makes a routing fault less likely to read as a calendar fault | §7 step 3 |
| 5 | **front month by `realExpirationDate`, rolled 8 calendar days out** | wrong contract in the roll week: bad fills, never a wrong sign | §7 step 3, and re-check during a roll week |
| 6 | **`liquidHours` format and `timeZoneId`** — `YYYYMMDD:HHMM-YYYYMMDD:HHMM`, read in the contract's own zone, widest segment wins | a format mismatch raises and aborts (fail-closed); a *parseable but wrong* segment is silent | §7 step 3, and on an early-close day |
| 7 | **futures limit at `ref ± tick/2`** with `ref` the mid | lands off the touch on a wide or one-sided book | §7 step 4 |
| 8 | **reprice = `placeOrder` again with the same `Order`** | verified correct against ib_async 2.1.0 (`orderId` is kept, so it modifies rather than duplicating); residual risk is a race on `Trade.isDone()` | §7 step 4 |
| 9 | **`ib.positions(account or "")`** | verified correct: the signature is `positions(self, account: str = '')` and `''` is the all-accounts default | §7 step 3 |
| 10 | **`Ticker.time` / `.halted`** for the freshness check | if `time` is absent the age check silently never fires (the halt and crossed checks still do) | §7 step 3 — print one `Ticker` |
| 11 | **`reqHistoricalData` daily bar = the official settlement** | `--reconcile` falls back to `--settlement`, loudly, and finalises nothing without one | §7 step 6 |

---

## 13. Files

| file | what |
| --- | --- |
| `config.py` | `Config` + the CLI; the live gate (mode **and** port), the tick-by-price rule, the spread-sized crossing cap, the rebalance ladder, the deleveraging knobs |
| `broker.py` | `Broker` protocol, `IBBroker`, `FakeBroker` + `FakeFaults`, `Fill`, `QuoteHealth`, `load_replay_day` |
| `journal.py` | `Journal` (JSONL, flushed and fsynced, rolls on re-run) + `DaySummary.from_journal` |
| `run_day.py` | `DayRunner` (preflight → calendar → ledger → regime → selector → sizing → entry → hedge → settle; or on a month-end, observe → size → buy at 15:30 → settle) , `reconcile_day`, `main` |
| `calendar_guard.py` | the NYSE session calendar by rule, `is_last_session_of_month`, the registry of named no-short calendars, and the month-end modes (proposals 54 and 55); `is_third_friday_session` and the third-Friday multiplier's verdict (proposal 58) |
| `premium_ledger.py` | the `(session, clock)` tape, its two causal estimators and the deleveraging multiplier |
| `selector.py` | proposal 46's E2 entry-clock selector |
| `sizing.py` | the stress table, `contracts_for`, and `scaled_contracts` (the third-Friday floor) |
| `pricing.py` | Black-76 package price, delta, volatility inversion, the V9 correction |
| `strikes.py` | nearest-OTM selection, the no-quote sentinel, the outage guards |
| `hedge.py` | `target_lots` (ES bulk + MES remainder), `rebalance_lots`, `residual_delta_lots` |
| `parity.py` | the engine-against-research parity harness; `--month-end` gates the override against proposal 54 |
| `tests/` | 220 tests, all replay-only, no network |
