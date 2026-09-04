# Proposal 03 — How the intraday Sharpe is aggregated, and what costs do to it

Scope: the intraday trade of `atm_straddle_intraday.ipynb` (nearest-OTM package re-picked every 30-minute bar from 10:00, one-bar hold to the next midpoint, 15:30 bar cash-settled; window-matched signal; block-diagonal ridge forecast; 10,387 bars on 866 days). Nothing here touches a notebook. All numbers are computed from the notebook's own cached rows plus the chain's bid/ask for the picked legs (`scratchpad/intraday_prop_agg/{build,analysis}.py`, `results.json`).

**Gate (all candidates).** The rebuild reproduces the notebook's rule table to 1e-6: always short 1.9020, sign(s) 1.9160, unit-median 1.8808, always-short-with-unit-median-close 3.0525 (annualized Sharpe of the daily sum of per-bar returns).

**Headline for the "why is the intraday Sharpe low" question.** It is not low relative to what the bars can deliver; it is low relative to a projection that assumes every bar carries the 15:30 edge. (1) Bars within a day are uncorrelated (implied within-day correlation 0.002–0.004; effective independent bars per day 11.5–12.3), so the daily Sharpe equals the per-bar Sharpe scaled by the square root of bars per day — there is no correlation penalty. (2) The sign(s) signal's edge sits almost entirely on the settlement bar: the 15:30 leg alone scores 1.82 annualized on 866 days, the other eleven bars score between −0.27 and 1.01 (six of them below 0.2), and 80% of the rule's per-premium mean comes from the 15:30 bar. Adding eleven near-zero-mean bars to one good one leaves the pooled Sharpe at 1.92 — the signed quadrature sum of the twelve per-clock Sharpes is 2.46, and the negative clocks pull it down from there. A "×√12" projection of the 15:30 result assumes twelve copies of the 15:30 edge; the data contain one. (3) Always short is the opposite case: it earns at every bar (per-clock Sharpe 0.20–2.51), which is why the pooled always-short rises from 0.20 at the settlement leg to 1.90 across the day, and why the hybrid (always short intraday, unit-median at the close) reaches 3.05. (4) The equal-premium aggregation over-weights the cheapest bars, and crossing the spread up to 23 times a day destroys everything except the settlement leg — candidates (b)–(d) below.

---

## (a) Which Sharpe to report: per-bar, daily sum, or premium-weighted

**Idea.** Three annualizations of the same rows disagree, and the disagreement is informative: the per-bar Sharpe annualized as if bars were independent, the notebook's daily-sum Sharpe, and a premium-weighted daily return (the day's dollar P&L divided by the day's dollar premium, which is one contract per bar).

**Construction.** Per-bar: mean/std of the stacked R′ = q·R, times √(252·12). Daily sum: Σ_bars R′ per day, mean/std ×√252 (the notebook). Premium-weighted: Σ q(exit−entry) / Σ |q|·entry per day, ×√252. Variance ratio var(daily sum)/(n̄·var(bar)) gives the implied within-day correlation and the effective bar count. Paired tests on the daily series after standardizing each by its own standard deviation (units differ); block-bootstrap ΔSharpe (B = 2000, rng 0, block ⌈n^{1/3}⌉).

| | always short | sign(s) | hybrid |
|---|---|---|---|
| per-bar Sharpe (mean/sd of R′) | 0.035 | 0.035 | 0.055 |
| per-bar × √(252·12) | 1.94 | 1.93 | 3.02 |
| daily sum × √252 (notebook) | **1.90** | **1.92** | **3.05** |
| premium-weighted daily × √252 | 3.39 | 1.86 | 4.34 |
| variance ratio / implied corr / n_eff | 1.04 / 0.004 / 11.5 | 1.02 / 0.002 / 11.8 | 0.98 / −0.002 / 12.3 |
| premium-weighted − daily: ΔSharpe CI, NW t | +1.49 [+0.86, +1.94], +6.0 | −0.12 [−0.60, +0.33], −0.5 | +1.29 [+0.76, +1.83], +5.6 |

Per-clock Sharpe (×√252 as one bar per day) and the median premium: always short 0.26, 0.64, 0.47, 2.19, 1.13, 1.78, 0.23, 0.79, 0.50, 1.00, 2.51, 0.20 from 10:00 to 15:30; sign(s) 0.14, 0.10, −0.03, 1.01, 0.82, −0.01, 0.79, 0.63, −0.27, 0.20, 0.08, 1.82; median premium falls from 19.8 points at 10:00 to 5.4 at 15:30. Within-day autocorrelation of R′ at lag one: 0.01 (sign(s)), 0.02 (always short); average cross-clock correlation of the daily contributions 0.001–0.002.

**Gates.** Rule table reproduced. No new information enters — every line is a re-aggregation of the same rows, so causality is inherited. Placebo not applicable (no fitted choice).

**Verdict: adopt the daily-sum Sharpe as the headline and print the premium-weighted one beside it, labelled by the sizing it implies.** The daily sum is the honest number for a trader who commits one unit of premium to every bar; it is not depressed by correlation (the naive per-bar annualization agrees with it to 0.04). The premium-weighted number is the honest number for one contract per bar, and it is much higher for always short and the hybrid because their edge lives in the expensive morning bars while the per-premium sum spends 80% of its weight on the 3.5%-of-premium settlement bar. Neither is a "projection" of the 15:30 result: sign(s) has one bar of edge, not twelve.

**Open questions.** Should the deck report both sizings side by side, or pick one convention and state it once? Is the 15:30-only sign(s) Sharpe (1.82 here vs 1.63 in the close-trade deck, five fewer days and warm-up bars scored flat) the right anchor for any intraday comparison?

---

## (b) Equal-premium versus equal-contract sizing across bars

**Idea.** The current construction puts one unit of premium in every bar; the 10:00 package costs 3.7 times the 15:30 package, so the equal-premium series holds nearly four times as many contracts at the close as at the open. Equal contracts is the other natural convention.

**Construction.** Equal premium (current): daily Σ q·R. Equal contract: daily Σ q·(exit−entry) in index points per contract. Equal dollar risk per day spread evenly across bars is the identity with equal premium, so it is not a third case. Comparison in Sharpe (unit-free) with the block bootstrap; a paired NW t on the standardized daily series.

| | always short | sign(s) | hybrid |
|---|---|---|---|
| Sharpe equal-premium (current) | 1.90 | 1.92 | 3.05 |
| Sharpe equal-contract | **2.49** | 1.80 | 2.44 |
| ΔSharpe (contract − premium), CI, NW t | +0.58 [−0.34, +1.31], +1.7 | −0.12 [−0.78, +0.51], −0.4 | −0.62 [−1.23, +0.04], −2.0 |
| worst day (per-prem / contract per day's mean premium) | −11.4 / −5.0 | −6.3 / −3.6 | −9.7 / −4.2 |
| max drawdown (same units) | −16.8 / −8.5 | −14.6 / −10.6 | −22.0 / −12.4 |
| skew, excess kurtosis (per-prem → contract) | −1.9 → −1.3, 8.7 → 5.0 | 0.5 → −0.1, 9.1 → 4.5 | −0.6 → −1.3, 6.8 → 11.7 |

Median share of the day's premium by clock: 12.9% at 10:00 falling monotonically to 3.5% at 15:30. Share of sign(s)'s total P&L from the 15:30 bar: 80% per premium, 48% per contract.

**Gates.** Rule table reproduced. Causal: sizing uses only the entry premium known at the bar. Placebo not applicable.

**Verdict: needs more — no significant difference in Sharpe for any rule; report the sizing convention explicitly rather than switching it.** Equal contracts lifts always short (+0.58, not significant at 95%) and lowers the hybrid (−0.62, borderline), because the hybrid's value lives in the cheap settlement bar that equal-premium over-weights; per-premium tails look worse only because the denominator is small on the late bars.

**Open questions.** Whether a fixed-dollar-per-day convention (one budget per day, spread by contracts) is the professor's preferred unit; whether the deck's rule table should state "one unit of premium per bar" in its header.

---

## (c) Transaction costs at the crossed spread

**Idea.** The intraday trade crosses the spread up to 23 times a day (entry and exit on eleven bars, entry only on the settled bar). The 15:30 trade crosses once. Any per-bar edge must clear the per-bar spread.

**Construction.** Crossed fills: a long pays the ask at entry and receives the bid at the next bar; a short receives the bid and pays the ask; the 15:30 bar settles in cash. Half-spread variant: charge ½(ask−bid) per crossing against the midpoint (identical here, the quotes are symmetric around the mid). Break-even half-spread: the uniform cost per crossing that zeros the daily mean, per unit premium.

| | always short | sign(s) | hybrid |
|---|---|---|---|
| Sharpe mid → crossed | 1.90 → **−3.54** | 1.92 → **−3.09** | 3.05 → **−2.10** |
| mean/day mid → crossed (premium units) | 0.160 → −0.304 | 0.153 → −0.247 | 0.272 → −0.189 |
| cost/day (premium units; NW t) | 0.465 (19.7) | 0.400 (17.8) | 0.461 (21.0) |
| cost/day in points vs mean/day in points | 6.7 vs 1.8 | 5.4 vs 1.2 | 6.7 vs 2.5 |
| crossings/day | 23.0 | 21.3 | 22.9 |
| break-even half-spread, % of premium | 0.70 | 0.72 | 1.19 |
| median half-spread, % of premium | 1.69 (1.2% at 10:30 rising to 2.9% at 15:30) | | |

Per-clock crossed Sharpe: every bar before 15:30 is negative for every rule (−1.5 to −4.1); the 15:30 bar survives for sign(s) (1.82 → 1.40) and for the hybrid (1.59 → 1.21). The 15:30 trade in the close deck: 1.63 at the midpoint, 1.11 crossed.

**Gates.** Rule table reproduced. Causal: entry quotes are the decision-time quotes; exit quotes are the next bar's; the settlement leg pays no exit spread. Placebo not applicable.

**Verdict: reject the intraday re-pick trade at the crossed spread; only the settlement leg clears its spread.** The daily cost is 2.5–3.5 times the daily mean; the break-even half-spread is 0.7–1.2% of premium against an actual 1.7%.

**Open questions.** Whether fills at a fraction of the spread (the experimental notebook's λ-fill sensitivity) rescue any morning bar; whether the vendor's quotes are stale enough at some clocks that the crossed spread overstates the executable cost.

---

## (d) Reducing turnover: hold the package while sign(s) does not flip

**Idea.** Re-picking every bar is what makes the trade cross the spread twenty-odd times. Keep the held strikes while the sign(s) signal keeps its sign; trade only on a flip.

**Construction.** Each day, enter the first bar's package on its sign; at each later bar, if sign(s) matches the held position, hold (no trade); if it flips or goes flat, unwind the held package at that bar's midpoint of the held strikes and enter the new bar's package. P&L per bar = q·(value of the held strikes at the next bar − value at this bar), midpoints; the last bar's value is the cash settlement of the held strikes. Per-premium return divides by the held package's entry premium, so a one-bar hold reproduces R exactly (identity check: max difference 0.0 on the 63 days where every bar flips). Crossed costs: one half-spread per entry (new strikes) and per unwind (held strikes at that bar, from the chain). Placebo: 200 random entry patterns per day with the same number of entries as hold-through, first bar always an entry. Causality: perturbing the signal after bar t leaves all decisions up to t unchanged (10 random days).

| | re-pick every bar | hold-through |
|---|---|---|
| trades (entries) per day | 12.0 | 3.8 (plus 2.9 unwinds) |
| crossings per day | 21.3 | 6.7 |
| within-day flip rate of sign(s) | — | 23–31% of bars at every clock after 10:00 |
| Sharpe at midpoints | **1.92** | 1.35 (ΔSharpe −0.57, CI [−1.33, +0.24], NW t −1.6) |
| Sharpe crossed | −3.09 | −0.56 (ΔSharpe +2.53, CI [+1.68, +3.51], NW t +6.3) |
| cost/day (premium units) | 0.400 | 0.147 |
| worst day / max drawdown (mid) | −6.3 / −14.6 | −8.9 / −18.1 |
| placebo percentile of hold-through (mid / crossed) | — | 84th / 84th (random patterns: median 0.96 / −0.94) |

**Gates.** Rule table reproduced; identity check exact; causality passed; placebo 84th percentile — below the 95th the deck requires.

**Verdict: reject as an improvement and reject as a rescue.** Holding gives up the midpoint edge (a re-picked nearest-OTM package keeps its premium fresh; a held package drifts in or out of the money and its next-bar mark carries a different exposure), and after costs the trade is still under water at −0.56. The cost saving is real (0.40 → 0.15 per day) but there is not enough intraday edge to pay even a third of the spread bill.

**Open questions.** Whether a hold rule keyed to moneyness (re-pick when the held package drifts more than one strike) rather than to sign flips would keep the midpoint edge; whether the only viable intraday trade is the settlement leg plus at most the 15:00 bar (the one intraday bar with a crossed Sharpe near zero for always short).

---

### Summary for the intraday deck (no changes proposed to it)

| candidate | verdict | one line |
|---|---|---|
| (a) which Sharpe | adopt (daily sum) + print premium-weighted | no correlation penalty; sign(s) has one bar of edge, always short has twelve small ones |
| (b) sizing | needs more | no significant Sharpe change; state the convention |
| (c) costs | reject intraday re-pick | 2.5–3.5× the mean in costs; only the 15:30 leg clears its spread |
| (d) hold-through | reject | loses midpoint edge, still negative after costs, placebo 84th |
