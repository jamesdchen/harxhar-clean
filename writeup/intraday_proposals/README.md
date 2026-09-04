# Intraday Sharpe: diagnosis and proposals (2026-09-04)

Read-only studies of the intraday trade (`notebooks/atm_straddle_intraday.ipynb`).
Nothing here is wired into the notebook; every study rebuilds the trade from the
cached rows and reproduces the notebook's rule table (always short 1.90,
sign(s) 1.92, unit-median 1.88, hybrid 3.05; block-diagonal ridge, 866 days)
before changing anything. Scripts and logs live in the session scratchpad.

## The question

The close trade (RV-IV notebook) scores sign(s) at Sharpe 1.63. A naive
projection - twelve independent half-hour legs at that edge - would put the
intraday trade near 6. It scores 1.92.

## The answer (00_diagnosis_sharpe_gap.md)

1. **Reconciliation is exact.** The intraday 15:30 leg is the RV-IV portfolio:
   identical strikes, entries, exits, returns and forecasts on all 866 shared
   days; the matched signal equals the close signal to 1e-11. Its 1.82 vs 1.63
   is the 63-day warm-up dropping the COVID quarter.
2. **Independence is not the problem; the edge is.** Cross-bar correlation is
   about zero (effective independent legs 5.8 of 12 by variance share). The
   eleven intraday legs together add Sharpe 0.10 to the close leg's 1.82; the
   close leg holds 80% of the P&L. Per-bar Sharpe 0.115 at 15:30, 0.00-0.07
   elsewhere.
3. **The intraday implied slice is the wrong price.** The forecast is as
   accurate intraday as at the close (correlation with realized 0.75-0.90 at
   every clock), but the quantity it is compared with - implied variance x
   half hour x causal window share - is fair to cheap during the day
   (realized / slice 1.02-1.11) and rich only at 15:30 (0.795), where the slice
   is an actual price. So sign(s) buys 50-67% of intraday bars (oracle 28-50%)
   with a buy-minus-sell spread of at most 0.02 (0.27 at the close) and forgoes
   the decay always short harvests (3.43 on those legs). A peeking oracle earns
   4-4.7 per clock, so the marks pay for information; the implied term is what
   is mis-specified, not the forecast.
4. Secondary: vega noise on the 30-minute mark dilutes but does not cap;
   premium falls 19.8 -> 5.4 points over the day, concentrating variance in the
   settlement leg. Ruled out: alignment, units (pricing ratio 1.0000 at every
   clock), window-share noise.

## Proposals and verdicts

| file | candidate | Sharpe (mid) | gates | verdict |
|---|---|---|---|---|
| 01 clock selection | trailing-edge clock filter | 1.92 -> 1.60 | t -2.7, placebo 32nd pct | reject |
| 01 | causally chosen late start | 1.80 | t -2.4, placebo 37-61st | reject |
| 01 | **always short before 15:30, sign(s) at 15:30** | **3.37** | dSharpe +1.47 vs all-bars always short, CI [+0.14, +2.54]; positive every year | **adopt (mid fills)** |
| 01 | per-clock sign(s)/always-short switch | 2.81 | placebo 99.4th, but dominated by the row above | reject |
| 02 signal & exit | bar-matched implied by BSM pricing | 1.83 (profile share); 2.26 (flat share) | vendor IV already = BSM implied (ratio 1.0000); flat share CI includes 0, placebo 70th | reject |
| 02 | per-clock recalibration | 1.54 | t -1.2, placebo 19th | reject |
| 02 | hold-to-close / first-clock / first-fire exits | 1.00 / 0.19 / 0.34 | all significantly worse; 15:30-only entry 1.82 is the 99.7th pct of entry clocks | reject; the paper's exit is right |
| 02 | sign(s) vs unit-median by clock | 3.37 vs 3.05 | one construction fewer; best-per-clock composite does not transfer split-half | adopt the simplification |
| 03 aggregation | daily-sum vs per-bar Sharpe | 1.92 vs 1.93 | bars uncorrelated; no correlation penalty | report the daily sum |
| 03 | equal-contract sizing | 1.80 (sign(s)), 2.49 (always short) | CIs include 0 | state the convention, no claim |
| 03 | **crossed-spread costs** | **-3.1 sign(s), -3.5 always short, -2.1 hybrid** | 21-23 crossings/day cost 0.40-0.47 premium units vs mean 0.15-0.27 (t 18-21); break-even half-spread < 1.2% vs median 1.7% | **the intraday re-pick does not survive costs; only the settlement leg does (1.82 -> 1.40)** |
| 03 | hold through unchanged signs | 1.35 mid, -0.56 crossed | worse mid, still negative crossed | reject |
| 04 remaining session | forecast of the variance left to the close vs the remaining-session implied, hold to close, per entry clock | 0.19 (10:00) ... 1.29 (15:00), 1.89 (15:30) | same SIGN as the bar signal by algebra (s_rem = s_bar / w); entries before 15:00 at the 11th-63rd pct of random entry clocks; first-fire rules significantly worse | reject as an intraday generalization |
| 04a audit | independent rebuild of 04 | matches | F_t-measurable; IV_rem is a price (pricing ratio 1.0000 every clock); the remaining implied is RICH at every clock (realized/implied 0.76-0.84) - the miss is the forecast's level (F_rem overshoots the realized remainder by 15-50%) | honest; the lever is the forecast's level calibration (07) |
| 05 other causal rules | realized persistence, implied-variance change, forecast revision, forecast-error sign | 1.67 / 0.93 / 0.46 / 1.44 | all below always short (1.90) and sign(s) (1.92) | reject |
| 05 | short-only sign(s): short when s<0, flat otherwise | 2.69 | same mean as always short, worst day -6.0 vs -11.4, placebo 99.9th; Sharpe gain not resolved at 95%; ~9 crossings/day | a risk filter, unlikely to survive costs |
| 06 one trade per day | entry-clock ladder, first-fire, sell-the-session-from-the-open | 15:30 entry 1.83 is the 99.8th pct of entry clocks; first-fire -0.39 / 0.32; sell-from-open -0.27 | every early entry loses; the session premium is the close, and there it is the sign not the short | reject |
| 06 | flat at the close on FOMC / month-end days (83 of 864) | 1.83 -> 2.30 | t 4.4; dSharpe CI [+0.02, +0.87]; placebo 99.9th; gain in the daily-0DTE era; flags found in-sample on the parked event filter | needs a forward test |
| 07 remaining-session recalibration | causal level calibration of the multi-bar forecast against the realized remainder (in flight) | - | - | pending |

## Recommendation

- At midpoint fills the best intraday construction is always short on every
  bar before the close plus sign(s) at 15:30 (3.37), replacing the unit-median
  hybrid (3.05) with one construction fewer.
- At realistic fills the intraday legs are a cost, not an edge: every bar before
  15:30 is negative for every rule once the spread is paid. The trade that
  survives costs is the close trade of the RV-IV notebook.
- The remaining-session implied IS a price and it is rich at every clock (04a);
  the intraday miss is the forecast's level, not the pairing - a causal
  recalibration of the multi-bar forecast against the realized remainder is
  the one lever left (07). The calendar filter at the close (06) is the one
  candidate that improves the surviving trade, pending a forward test.
