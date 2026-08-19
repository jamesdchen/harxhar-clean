# Seed prompt for a fresh, repo-less agent — meeting prep with my professor

*Everything below the rule is self-contained: paste it into a new session that has no access to my code, files, or GitHub. All numbers are inlined; the agent should reason from what is here and ask me for anything it needs.*

---

## Your role

I have a meeting tomorrow with my professor (Chris Jones, USC Marshall — co-authors on the paper are Austin Pollok and Mihai Cucuringu). Help me (1) address each concern he raised, (2) organise what was built into a 15-minute walkthrough with a clear headline and honest caveats, (3) anticipate his follow-up questions, and (4) rehearse. Be direct: tell me where my framing is weak, where a claim outruns the evidence, and what he is likely to push on. You have no access to my repository or data — every number you need is in this brief; if something is missing, ask me and I will pull it, do not invent it.

## Background: the paper

We forecast 30-minute-ahead S&P 500 realized variance (RV) on a near-24-hour panel, 1998–2024, ~275,000 bars. Incumbent: OLS on the HAR lags plus calendar dummies ("a0"). Our contribution: a two-block ridge ("blk2") that shrinks 41 exogenous channels (order flow, liquidity, sentiment, implied vol, cross-section moments, etc.) under one penalty and the HAR block under another, penalties differing by two orders of magnitude; then a frozen pairwise-product block distilled from tree fits ("blk3"). Scoring: causal calibrated second-moment protocol, QLIKE loss (lower better), Diebold–Mariano t against a0 (negative favours the arm), all on the same rows.

| arm | QLIKE | DM vs a0 | n bars |
|---|---|---|---|
| a0 — OLS on HAR + calendar | 0.22521 | — | 273,554 |
| blk2 — two-block ridge | 0.22350 | −3.1 | 272,971 |
| blk3 — three-block (adds product block) | 0.21486 | −4.2 | 248,280 |
| product-block increment, paired blk3 vs blk2 | −0.00079 | −2.8 | 248,280 |

These held to the last digit after this week's complete cluster harvest and re-score. The exogenous increment is real but small — about 0.8% relative QLIKE for blk2 over a0.

The professor's question, from a research-spec document he wrote in mid-August 2026: can this forecast accuracy be turned into a **0DTE SPX options strategy**, and what is the right design?

## The professor's spec — his concerns, item by item

1. **Instrument.** SPX (cash-settled at the close, no share unwinding), 0DTE, near-the-money. **Delta-hedged calls or puts, separately — not straddles**: hedging is cheap, gives a larger sample of options, and mispricing may differ by leg. (He noted straddles could signal non-directionality to a market maker for better fills, but chose single legs to see calls and puts separately.)
2. **Cadence.** Real-time but no more than once a minute for data cost; 30-minute periods for simplicity.
3. **Signal.** At each period, forecast RV to the close ("time-till-close"), measure IV to the close, form the variance risk premium as **ln(IV/RV)** (a price, not an expected return). **Sell + hedge if conditionally sufficiently positive; buy + hedge if sufficiently negative; do nothing near zero.**
4. **Forecast structure.** He needs a *sequence* of time-till-close forecasts that update every period; a one-shot multivariate forecast at the open "is a bad idea because it doesn't update." Is summing multi-step forecasts the right construction? Is there a model that naturally "predicts the rest of the series conditional on the current time" (he floated diffusion models, weather nowcasting / GraphCast)? Should the model be conditioned on IV: RV(t+kD | F_t) = f(X(t), k, time-of-day, IV(t,T))?
5. **IV measurement.** Black–Scholes for time-till-close IV — "why is this sufficient, and why not another model?"
6. **A second strategy.** "Trading the bumps" — relative value / vol-surface anomalies using an IV forecast, distinct from the RV-value strategy.
7. **Roadmap.** Single-period forecast first; extend HAR to an Attention-based HAR (time-series or cross-sectional attention); then multi-period.

## What was built against each item

Data: every SPX 0DTE (expiration-day) option chain 2020–2025 at 30-minute stamps (8.2M quotes: bid/ask/mid, underlying); the paper's per-bar forecasts (a0 and blk2) aligned to the same bars; realized per-bar variance from our panel. Everything below is causal (all calibrations, regressions, z-scores and thresholds fit on an expanding window of past days, minimum 63, applied shifted one day), all trading books are scored at mids and "crossed" (buy at the ask, sell at the bid, plus 0.5 bp on the underlying per hedge rebalance), PnL is aggregated to one number per day, Sharpe = mean/std × √252 on those daily numbers, and every book is run alongside an always-short (sell-vol) control and an a0-in-place-of-blk2 "swap test."

### Items 1 + 3 — the strategy itself, exactly as specified

74,000 near-ATM single-leg contracts (Black–Scholes |delta| 0.3–0.7, both sides quoted) over 670 expiration days 2020–Apr 2024. Per contract: BS time-till-close IV from the mid (τ = hours to 16:00 ET); remaining-variance forecast = sum of the model's per-bar forecasts over the remaining bars, calibrated per entry hour; signal s = ln(IV / √RV̂); dead-zone θ; delta-hedged with the BS delta, rebalanced every 30 min on the underlying, held to settlement at intrinsic.

| dead-zone θ | days traded | % long | Sharpe (mids) | **Sharpe (crossed)** | hit |
|---|---|---|---|---|---|
| 0 (always act) | 100% | 34% | 6.1 | 2.5 | 63% |
| 0.05 | 78% | 24% | 6.4 | 3.0 | 66% |
| **0.10** | 58% | 15% | **6.8** | **3.6** | 70% |
| 0.20 | 27% | 4% | 5.4 | 3.1 | 72% |
| *always-short control* | 100% | 0% | 3.8 | — | 65% |

- Beats the always-short control while going long a quarter to a third of the time — something always-short cannot do.
- Positive after crossing the spread at every θ, both legs, both eras.
- Calls and puts agree (θ=0.10: calls 5.8 / 3.0, puts 6.5 / 3.6) — the spec's reason for splitting the legs is vindicated.
- Holds in the "daily 0DTE" era (every weekday expiring, May 2022 onward, 433 days): 6.6 mids / 3.5 crossed.
- Hit rate rises monotonically with θ (63 → 72%): the dead-zone is doing what he said it would.
- It is a **morning phenomenon**: entries 10:00–13:00 ET earn 5–6.5 vs the control's 3.3–4.3; from 15:00 the edge is ~0 as the premium fraction of the option price rises to ~0.2 and the model goes long only 10% of the time. The smile is fairly priced late, mispriced early.

### Item 3 — the dead-zone threshold is measured, not guessed

Quote-cost surface from the same chains: the round-trip crossing cost expressed as ln(IV_ask / IV_bid) is ≈ 0.015 at 11:00 ET for |delta| 0.3–0.5, ≈ 0.02 for 0.5–0.7, rising to 0.03–0.04 by 15:00; for |delta| > 0.7 it is 0.07–0.13 (untradeable at any forecast quality). The absolute half-spread is pinned at the tick (0.05 / 0.10 / 0.25 index points by moneyness), so all the time-of-day and era variation is premium thinning, not spreads widening. So "conditionally sufficiently positive" has a floor of about 0.015–0.04 depending on hour, and θ = 0.05–0.10 sits above it everywhere — that is why it is the sweet spot. Availability is not binding: every moneyness bucket is live in ~100% of snapshots, 2–10 contracts per snapshot per bucket.

### Regime and exit-timing on the same book (θ = 0.10, blk2)

- **No market regime concentrates the edge.** Every causal VIX quintile pays (2.4–6.1 at mids, non-monotone), contango vs backwardation ties, day-of-week is noise, median |signal| barely moves with VIX (0.11 → 0.14) so a state-dependent θ is not warranted; a signal × VIX interaction is t = +0.55 once clustered by day.
- **One anti-regime: FOMC days.** On the 23 announcement sessions the book goes 97% short at 5× the normal signal magnitude — the remaining-variance forecast cannot see the announcement — and earns 0.33 mids / −2.20 crossed. Excluding them lifts the book to 7.21 / 4.01. Release days in general are weaker (4.96 vs 7.65 mids on non-release days).
- **Exit rule.** Exit when the re-measured signal crosses back through zero: 7.63 mids / 4.09 crossed vs hold-to-settlement's 6.78 / 3.62, exits early on 27% of trades, beats hold at every entry hour. Fixed holding periods (1, 2, 4 bars) look fine at mids and are poison crossed (1 bar: −4.7) — the second option round-trip eats a partial move. Realized-fraction curve: a 10:00 entry has captured 30% of its settlement PnL after 1 bar, 71% after 4, ~100% by 9 — you exit on information, not on the clock.

### The composed final book (ex-FOMC + signal-cross exit): both rules add

| θ = 0.10, both legs, all days | Sharpe mids | Sharpe crossed | hit (crossed) |
|---|---|---|---|
| hold, all days (baseline) | 6.78 | 3.62 | 64% |
| + signal-cross exit | 7.64 | 4.09 | 65% |
| + ex-FOMC | 7.02 | 3.82 | 66% |
| **+ both** | **7.86** | **4.28** | **67%** |

Trades 55% of days, 28% long, 74% hit at mids. θ = 0.05 is the crossed optimum for the composed book — **8.34 / 4.32** on 77% of days — because the exit rule now does part of the dead-zone's job. Holds in the daily-0DTE era (7.79 / 4.36 at θ = 0.10). 23 FOMC days excluded; the release-flag series ends 2024-01-09 so 67 later days are kept as "unknown," not treated as non-FOMC.

### Item 4 — forecast structure (his modelling questions)

- **Should IV be in the information set? Yes.** A direct expanding regression of log remaining RV on log(model sum) alone buys nothing over the sum (DM +1.0); adding log MFIV (the model-free implied variance from the strip) is worth **DM −3.6** (−4.3 in the daily-0DTE era); adding today's realized-so-far another −3.1. The IV weight rises monotonically 10:00 → 14:00 (0.035 → 0.118 full sample; 0.14 → 0.22 daily-0DTE) then is discarded in the last hour. His f(X, k, time-of-day, IV) intuition is confirmed by the data.
- **Is summing the right construction, or should we predict the "rest of the series"?** Tested the cheap honest analogues of his diffusion/nowcast idea: a deterministic diurnal shape, a direct multi-output regression per entry bar, and an analog-ensemble (50 nearest past days on the day's partial state, rescaled). None beats the sum on the *total* remaining variance (the analog level is decisively worse, DM +7.3, because porting a neighbour day's path onto today is a high-variance level estimator; the direct regression only wins after 14:00, DM −2.7 to −4.4). But the analog days do know the *allocation* across remaining bars — per-bar QLIKE 0.303 vs the sum's 0.324. Answer: the sum's level is already right; its shape across the remaining bars is improvable; the concrete proposal is a small shape model *on top of* the sum, not a replacement.
- **One pooled model conditioned on k vs one fit per entry hour?** Twelve per-hour fits beat one pooled fit with time-of-day dummies for entries before 13:00 (t +2 to +6); pooled wins only in the last two bars. The slopes genuinely move with k, so "one model, condition on time-of-day" is not free.

### Item 5 — why Black–Scholes suffices

The payoff is settled at intrinsic, which is model-free. BS enters only as (a) the quoting map from a price to a comparable time-till-close vol and (b) the hedge ratio, a second-order term over a few hours. Any smile-consistent model gives the same intrinsic PnL and differs only in delta. (Implementation checked: the vectorized BS-IV solver matches a root-finder to 5e-7; the vectorized hedge accounting matches a contract-by-contract scalar loop to 1e-14.)

### Item 6 — "trading the bumps"

Built exactly as an IV relative-value book: quadratic smile fit per snapshot and side (R² ≈ 0.99 per side), buy strikes with anomalously low IV / sell anomalously high at z ∈ {1, 1.5, 2}, delta-hedged to settlement, plus an intraday temporal-anomaly variant and a premium-neutral pairing. At mids the cross-sectional books earn 0.4–0.9 Sharpe; **crossed, every one is negative** (−0.9 to −10.6): the bump lives inside the bid–ask spread. The temporal variant's 1.7 at mids is a disguised 80%-short position that the always-short control dominates outright. Null after costs. (One useful by-product: the delta-hedged always-short near-ATM control is crossed-positive, +1.1 all / +2.2 daily-0DTE — the VRP measured on his instrument, and the baseline the value strategy has to beat.)

### Item 7 — Attention-HAR

Started, then stopped by me to prioritise the strategy work. Not built. The single-period step it depends on is exactly the paper's existing result.

## The caveats I must raise myself, in this order

**A. The book monetizes HAR-family accuracy, not the exogenous block.** The swap test — a0 in place of blk2, everything else identical — matches or beats blk2 at every θ (paired daily t from −0.1 to −2.2 favouring a0 on the plain book; 0.24 / 0.03 / −0.05 on the final composed book — indistinguishable). blk2 goes long more often (34% vs 24%), and those extra longs are the trades that lose. So the paper's ~0.8% relative-QLIKE increment does not register in 670 days of trading. Framing: *the trading result demonstrates the forecast family against 0DTE quotes; the paper's contribution stays a forecasting one.* I must not claim the exogenous edge trades.

**B. A clock bug invalidated an earlier round of results; it is fixed and retracted.** Our per-bar forecast panel carries naive Eastern-time stamps that one export script labelled as UTC; when the intraday option chain (true UTC) was merged against it, every option snapshot was joined to the model's bar 4 hours later (5 in winter). Everything intraday built before the fix was void — including an ATM-straddle "two-sided book" I had reported at 1.5–2.9 Sharpe uniformly across the day. The 10:00 → close analysis and everything chain-only (bumps, quote costs) were unaffected. The bug was found by one of the checks (the per-bar RV profile collapsed after 12:30, and the summer/winter RV ratio by clock label swung 0.4 → 15.8 across the day, impossible with a correct clock), fixed at the source, verified two ways (RV now peaks at the 10:00 opening bar and collapses overnight; the summer/winter ratio is ~1 flat), and everything was rerun. On the corrected panel the straddle book is 1.5–2.5 mids / 1.2–2.3 crossed for morning entries and ~0 to negative from 14:30 — consistent with the leg book's morning-only edge. **State this proactively; it is a strength of the process, and he will find it otherwise.**

**C. The remaining-variance forecast in the strategy backtest "peeks."** The sum of per-bar forecasts over the remaining bars uses each bar's own forecast — but bar t+3's forecast is made at t+3, with information the trader at bar t does not have. A strictly F_t-measurable version (the one-step forecast standing at t, or a direct regression of remaining variance on information at t) scores ~3× worse in QLIKE (0.19 vs 0.063 pooled); that gap is the size of the peek. **The 6.8 / 3.6 leg book uses the peeking sum.** This is the most important open item: the book must be rerun with the F_t forecast before any number is called tradeable (it is a one-line swap in the ledger builder). Expect the mids and crossed numbers to fall. Two things already known inside the F_t class: IV still helps by the same DM −3.6, and the sum-vs-direct ranking is unchanged.

**D. Data limits that shape what is possible.** Only expiration-day (0DTE) chains are on disk — no calendars. Within a single expiry, vega/gamma = S²σT for every strike, so gamma cannot be isolated from smile marks without a second expiry; the per-bar *timing* edge (blk2's largest one-bar advantage) is unexpressable in 0DTE-only data as a matter of algebra. The chain's early-close flag is set on every row (useless — half-days were detected by clock instead); the 09:30 stamp has no underlying price or delta anywhere, so the opening bar is untradeable from this file. VIX and release-day series end Jan/Feb 2024. Concrete data asks: intraday non-0DTE chains (unlocks calendars), pre-2020 expiration-day chains (more tape — the daily-0DTE structural break is May 2022, only 433 days after it).

**E. Costs modelled, capacity not.** Mids and crossed at quoted spreads, 0.5 bp on the underlying; there is no size data, so no market-impact or capacity claim.

## Suggested 15 minutes

1. (2) His spec → what was built, one line per item; headline: **the design works** — 6.8 / 3.6 → 7.9 / 4.3 composed, two-sided, beats always-short, positive after costs, calls = puts, holds in daily-0DTE.
2. (3) The three things his design got right that our earlier straddle test did not: single legs (larger sample, per-leg agreement); the dead-zone with a measured cost floor; delta-hedging (removes the late-day long-bias loss the unhedged straddle shows).
3. (3) His modelling questions: IV in the information set — yes (DM −3.6); path models — level no, shape yes; per-hour beats pooled; why BS suffices.
4. (3) Caveats A–C, as findings, swap test first.
5. (2) Bumps null; regime/exit; FOMC.
6. (2) Next steps: F_t rerun first; calendars data ask; Attention-HAR scope; capacity.

## Questions he is likely to ask

- "Is 6.8 believable?" → It's at mids on daily-aggregated PnL over 670 days; crossed 3.6; the always-short control alone is 3.8 at mids, so the *incremental* Sharpe over just selling vol is the honest number; and the F_t rerun is pending (caveat C).
- "So the exogenous features don't matter for trading?" → Not at this sample; the swap test is flat. The book tests the family's level accuracy against the smile — a fair result, not a failure — and it is exactly why the paper's claim is forecasting rather than trading.
- "Why did the straddle look good and then not?" → the clock bug, one sentence; the two-plot tell.
- "Why hedge with entry IV rather than re-marking each bar?" → standard for hold-to-settle 0DTE; documented assumption; second order over a few hours; testable.
- "What about the diffusion-model idea?" → tested the cheap analogues; the sum's level is right; the shape is where the room is; a shape model on top of the sum is the concrete proposal.
- "Attention-HAR?" → not built; the single-period step is the paper; propose scope.
- "How much is the FOMC exclusion doing?" → +0.2 crossed; the exit rule +0.5; the base book without either is 3.6 crossed. Small, additive, pre-specifiable.
- "Could this be a paper section?" → As an economic reading of the forecast family, yes — QLIKE on the 10:00 → close claim, the delta-hedged book with controls, and the swap test as the honest boundary of the claim. Not as an exogenous-edge trading result.
