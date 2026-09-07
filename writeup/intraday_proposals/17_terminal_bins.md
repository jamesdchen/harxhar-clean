# 17 - Terminal-move correction by session trend and pin proximity (one cell): kill

**Claim.** Whether the close lands beyond the package's premium is predictable at 15:30 from two things a variance forecast does not carry: how strongly the session has trended into 15:30, and where the index sits inside its 5-point strike interval.
**Mechanism.** The estimand closer to settlement (analysis point 1); the market is wrong about which side of the coin the variance sits on (point 5). Not a slope on s, not sizing, not moneyness, not a cutoff.
**Estimand.** P(|S_close - S_15:30| > entry premium | F_15:30), realized against the Gaussian value implied by the variance forecast, in causal bins of trend tercile x pin-proximity bin; the decision stays a comparison of E[R | F] to the mid.
**Action.** q = sign(E[R | F]) where E[R] is the Gaussian expected payoff of the actual two strikes with the forecast sd scaled by the trailing cell ratio mean(m^2)/mean(sd^2), normalized to mean one across cells each day (a level shift is already dead: proposal 14). Prior days only, 63-day minimum for the tercile cuts, 30 observations per cell.
**Falsifier.** A flat pre-check table; or flipped days with E[R] near zero (a flattened step).

## Pre-check (864 days, block-diagonal ridge)

Realized P(beyond premium) 0.537 vs Gaussian 0.548. Realized / Gaussian by session-trend tercile: weak 1.09, mid 0.94, strong 0.91; E[R | buy] +0.17 / +0.15 / -0.02. By pin proximity: at a strike 0.89, near 0.95, middle 1.03; E[R | buy] +0.01 / +0.01 / +0.20. The close bar continues the session's direction on 50.0% of days (no momentum into the close; if anything exhaustion). Each cell has about 120 buy days with a standard error near 0.12 on the mean: one standard error deep.

## Result (midpoint; crossed in brackets)

| rule | buy share | mean | Sharpe | E[R|buy] | E[-R|sell] | contrib buy / sell | IR vs always short | flips vs T0 |
|---|---|---|---|---|---|---|---|---|
| always short | 0% | 0.014 | 0.19 [-0.29] | - | 0.014 | 0 / 0.014 | - | - |
| sign(s), T0 | 40% | 0.096 | 1.36 [0.89] | +0.104 | +0.092 | 0.041 / 0.055 | 0.79 | 0 |
| Gaussian payoff vs mid, no correction | 40% | 0.093 | 1.31 [0.84] | +0.098 | +0.089 | 0.040 / 0.053 | 0.75 | 4 |
| the cell | 38% | 0.082 | 1.15 [0.68] | +0.091 | +0.076 | 0.034 / 0.048 | 0.68 | 153 |

Paired cell minus T0: -0.21 Sharpe at both fills, interval [-1.07, +0.56]. Trailing cell ratios: mean 1.00, sd 0.36, range 0.31-2.76.

## The failure, against the mixture

It flattened the sign step: the 66 days flipped short-to-long have E[R] = -0.06 and the 87 flipped long-to-short have E[R] = +0.03; the decile plot of the corrected estimand lost the step (raw s deciles -0.13, -0.01, +0.12, +0.22 across the sign; corrected -0.22, -0.06, +0.18, +0.07, -0.04). It stole from both contributions (sell 0.055 -> 0.048, buy 0.041 -> 0.034). The population effect (ratios 0.89-1.09) is real and smaller than the noise of its causal cell-level estimate (sd 0.36), so near the boundary the correction reorders days at random. The uncorrected Gaussian payoff changes four days: E[RV] minus the implied variance already is the classifier, and the strike geometry adds no decision.

## Stop

The terminal-side information exists (the terminal oracle is 0.73 correlated with the trade's sign against 0.17 for the variance oracle) and is not reachable from realized-variance forecasts by recalibration (13), by a level (14), or by causal bins on trend and pin proximity (17). The next single test is a forecast trained on the terminal move upstream, scored on the twenty unscored months as a holdout. Script: `17_terminal_bins.py`.
