# Door 3 quantified: the eval-prop as a barrier option on the *firm's fee schedule*, not on our edge

**2026-07-02. Branch `edge-features-legibility`, local-only. Worker `door3_worker.py`, results
`results/door3/summary.json`. Env 285J.**

## Question
The signed-return channel is null (commit `5881635`): our information set has zero first-moment content,
so no futures-only (delta-one) strategy can express the variance edge. The *only* literal way to sit at a
retail **futures** eval-prop and use the vol forecast at all is to stop treating the prop as a market and
treat it as a **contract**: the evaluation fee buys a barrier option (reach +target before the trailing
drawdown), and a realized-vol forecast is an odds-timer for that option. Door 3 asks: **priced honestly,
what is that worth, and does forecast quality move it?**

## Design (pre-registered, no tuning)
- **Env = real SPX 30-min session bars (9:00–15:00 ET, 1993–2024), returns globally DEMEANED** → a
  zero-drift tape *by construction*. This is the null hypothesis made into an environment: any P(pass) > 0
  here is produced by the **barrier geometry**, not by any edge. 7 950 trading days, 1 350 overlapping
  attempts (start every 5th day, dependence acknowledged).
- **Mechanics = the verified `eval_sim.adp.simulate_topstep` rules** (Topstep 50K Combine): initial
  \$50 000, target +\$3 000, trailing drawdown \$2 000 with EOD ratchet + breakeven lock
  (`floor = min(M_eod − D, initial)`), \$1 000 daily-loss-limit day-stop, consistency 0.5 (best day ≤ 50 %
  of target or the effective target rises), min 2 trading days, 200-day window.
- **Two executions bracket reality**: `barclose` = the DLL stop fills at the breaching *bar's close*
  (full overshoot — **honest/conservative**); `exact` = the stop fills exactly at −\$1 000 (perfect
  intraday management — **optimistic fantasy**). Truth is nearer `barclose`.
- **The lever under test = the vol forecast**, three ways: `perfect` (√ of *realized* day-variance,
  look-ahead), `naive` (√ of yesterday's RV), `HAR` (causal daily HAR on log-RV, expanding refit). Plus
  vol-**timing** (trade only when σ̂ is in the top-50/top-20 or bottom-20 of its trailing 1000-day
  distribution), sizing grids (vol-target and fixed-notional), and day-target grids.

## Result — the forecast is worth nothing to the eval contract

| config | P(pass) honest / optimistic | cal-days | note |
|---|---|---|---|
| vol-target \$4k, τ=\$1.5k, **HAR σ̂** | 0.276 / 0.367 | 3.9 | the deployable version |
| vol-target \$4k, **naive σ̂** | 0.272 / 0.371 | 3.9 | ≈ HAR |
| vol-target \$4k, **PERFECT σ̂ (look-ahead)** | **0.276 / 0.348** | 3.9 | **no better than naive** |
| σ̂-timing top-50 | 0.238 / 0.317 | 37.3 | worse + 10× slower |
| σ̂-timing top-20 | 0.224 / 0.279 | 80.2 | worse + 20× slower |
| σ̂-timing bottom-20 | 0.170 / 0.207 | 69.2 | worst |
| fixed 1 ES, τ=\$1.5k | 0.284 / 0.368 | 8.7 | sizing ≫ forecast |
| day-target τ=\$1k / \$3k / none | 0.199 / 0.184 / 0.127 (honest) | — | τ (a *rule* choice) dominates |

**The decisive line is `perfect σ̂`.** Giving the trader *look-ahead knowledge of realized variance* does
**not** raise P(pass) above the naive lag (0.276 vs 0.272). Vol-**timing** — the natural way to "use" the
forecast — strictly **lowers** P(pass) and inflates time-to-resolution 10–20× (sitting out most days means
rarely reaching target inside the window). What *does* move P(pass) is entirely **contract geometry and
sizing**: the day-target τ, the notional, the execution assumption. **The eval contract pays a coin-flipper
exactly what it pays us.**

Mechanistically this is inevitable: pass-probability of a driftless first-passage problem is a functional
of the barrier ratio and the *step-size distribution*, not of any conditioning that leaves the mean at
zero. Our forecast conditions the **variance**; the barrier problem is (to first order) variance-scaled, so
a better variance forecast changes *how fast* you resolve, not *which way*.

## Funded stage — bounded, mostly zero
After a pass, the funded account (no target, floor locked at initial, withdraw down to a \$2k cushion
whenever cushion ≥ \$2.9k, run to bust or 500 days):

| execution | mean extracted | median | P(extract = \$0) |
|---|---|---|---|
| honest (barclose) | \$1 087 | **\$0** | 0.72 |
| optimistic (exact) | \$1 708 | **\$0** | 0.63 |

Both sit under the **analytic ceiling of \$2 000** (a zero-drift account's optional-stopping value = its
initial cushion above the floor). The **median funded account extracts nothing**; the positive mean is a
lucky right tail. Friction (the day-stop overshoot) is the gap from \$2 000 down to \$1 087.

## Economics (the honest chain)
Per funded account you pay for ≈ 1 / 0.28 ≈ **3.6 combine attempts** (honest P(pass)). At a ~\$150
combine + reset that is ~**\$540 of fees** to reach one funded account whose **median** payout is **\$0**
and whose **mean** is ~\$1 100. The expectation is a small positive number carried entirely by the tail,
dominated by terms we did **not** supply — reset price, payout buffer, minimum winning days, the real
consistency rule — and it **does not contain our edge**.

## Verdict
Door 3 is real but it is **not a deployment of the variance edge**. Its EV is a property of the firm's
**fee-and-payout schedule** (funded by other applicants' failed fees), to which our forecast contributes
**≈ 0**. It is the "gambling style" trading firms explicitly police (TOS-fragile), and the funded stage is
mostly sim. **Recommendation: not a venue for the research edge.** It belongs in the writeup as the clean
demonstration that *a prop evaluation is a barrier option whose value is edge-independent* — the same
QLIKE↔P&L / second-moment story from the other side.

## Caveats to flag in any external use
- **Daily/bar resolution**: the `barclose`↔`exact` gap (0.28→0.37) is pure intraday-management fantasy;
  honest = `barclose`. Intraday MLL touches that recover by the close are invisible → favorable bias.
- **Overlapping starts** → the P(pass) point estimate is not iid; but the *invariance to σ̂ quality* is a
  within-sample paired comparison, robust to that dependence.
- **Topstep 50K params only.** Real firms differ on the levers that actually matter here (reset price,
  payout buffer, min days, consistency) — those, not the forecast, are the decision variables. That is the
  datum to get, and `simulate_topstep` prices any of them post-hoc from `results/vrp_pnl/` component dumps.
