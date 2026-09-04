# Intraday Sharpe: diagnosis and proposals (2026-09-04)

Read-only studies of the intraday trade (`notebooks/atm_straddle_intraday.ipynb`).
Nothing here is wired into the notebook; every study rebuilds the trade from the
cached rows and reproduces the notebook's rule table before changing anything.
Scripts and logs live in the session scratchpad.

**The notebook's rule table (re-executed 2026-09-04, block-diagonal ridge,
866 days / 10,387 bars):** always short 1.90 (t 3.53), sign(s) 1.84 (t 3.41),
hybrid `always short, sign(s) close` 3.23 (t 5.98). Source:
`results/atm_straddle_intraday/rule_table_intraday_blk2.csv`.

> **Provenance.** Every number below that is sourced to the notebook or to
> `results/atm_straddle_intraday/*.csv` has been refreshed against that run.
> The per-proposal Sharpe ratios in the verdict table were computed by the
> scratchpad studies *before* the 2026-09-04 fix wave (the five frozen
> half-sessions, the 250-session recalibration window, the implied-volatility
> node censoring, the strike-gap guard) and have **not** been re-run. Their
> verdicts stand --- every one is a comparison inside a single frame --- but
> read their levels as indicative, not as current numbers. Rows marked
> `(notebook)` in the table below carry the re-executed numbers; rows marked
> `(pre-fix)`, and every unmarked row, carry the earlier study's.

## The question

The close trade (RV-IV notebook) scores sign(s) at Sharpe 1.42. A naive
projection - twelve independent half-hour legs at that edge - would put the
intraday trade near 5. It scores 1.84.

## The answer (00_diagnosis_sharpe_gap.md)

1. **Reconciliation is exact.** The intraday 15:30 leg is the RV-IV portfolio:
   identical strikes, entries, exits, returns and forecasts on all 866 shared
   days; the matched signal equals the close signal to 1e-11. Its settlement
   leg scores 1.67 against the close notebook's 1.42 because the matched signal
   sits flat on the first 64 dates, which drops the COVID quarter (the close
   notebook on the same 802 post-warm-up days scores 1.69).
2. **Independence is not the problem; the edge is.** Cross-bar correlation is
   about zero (effective independent legs 5.8 of 12 by variance share). The
   eleven intraday legs together add a fraction of a point of Sharpe to the
   close leg's 1.67; the close leg holds the great majority of the P&L. Per-bar
   Sharpe is an order of magnitude larger at 15:30 than at any other clock
   (pre-fix study numbers: 0.115 against 0.00-0.07).
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
| 01 | **always short before 15:30, sign(s) at 15:30** | **3.23** (notebook) | dSharpe vs all-bars always short positive, CI excludes zero; positive every year | **adopt (mid fills)** |
| 01 | per-clock sign(s)/always-short switch | 2.81 | placebo 99.4th, but dominated by the row above | reject |
| 02 signal & exit | bar-matched implied by BSM pricing | 1.83 (profile share); 2.26 (flat share) | vendor IV already = BSM implied (ratio 1.0000); flat share CI includes 0, placebo 70th | reject |
| 02 | per-clock recalibration | 1.54 | t -1.2, placebo 19th | reject |
| 02 | hold-to-close / first-clock / first-fire exits | 1.00 / 0.19 / 0.34 | all significantly worse; 15:30-only entry 1.82 is the 99.7th pct of entry clocks | reject; the paper's exit is right |
| 02 | sign(s) vs unit-median by clock | 3.23 vs 3.05 (pre-fix) | one construction fewer; best-per-clock composite does not transfer split-half; the notebook now carries only the sign(s) hybrid | adopt the simplification |
| 03 aggregation | daily-sum vs per-bar Sharpe | 1.92 vs 1.93 | bars uncorrelated; no correlation penalty | report the daily sum |
| 03 | equal-contract sizing | 1.80 (sign(s)), 2.49 (always short) | CIs include 0 | state the convention, no claim |
| 03 | **crossed-spread costs** (notebook) | **-2.13 sign(s), -1.92 always short, -0.89 hybrid** | 16.4-17.0 crossings/day (the re-pick holds the same strikes on 29.9% of holds and is booked at mid); break-even half-spread 0.86-1.55% of premium vs a median half-spread of 1.69% | **the intraday re-pick does not survive costs; at the crossed spread only the settlement leg does, and only when sized by sign (sign(s) +1.32, hybrid +1.31, always short -0.12)** |
| 03 | hold through unchanged signs | 1.35 mid, -0.56 crossed | worse mid, still negative crossed | reject |
| 04 remaining session | forecast of the variance left to the close vs the remaining-session implied, hold to close, per entry clock | 0.19 (10:00) ... 1.29 (15:00), 1.89 (15:30) | same SIGN as the bar signal by algebra (s_rem = s_bar / w); entries before 15:00 at the 11th-63rd pct of random entry clocks; first-fire rules significantly worse | reject as an intraday generalization |
| 04a audit | independent rebuild of 04 | matches | F_t-measurable; IV_rem is a price (pricing ratio 1.0000 every clock); the remaining implied is RICH at every clock (realized/implied 0.76-0.84) - the miss is the forecast's level (F_rem overshoots the realized remainder by 15-50%) | honest; the lever is the forecast's level calibration (07) |
| 05 other causal rules | realized persistence, implied-variance change, forecast revision, forecast-error sign | 1.67 / 0.93 / 0.46 / 1.44 (pre-fix) | all below always short (1.90) and sign(s) (1.84) | reject |
| 05 | short-only sign(s): short when s<0, flat otherwise | 2.69 (pre-fix) | same mean as always short, worst day -6.0 vs -11.4, placebo 99.9th; Sharpe gain not resolved at 95%; ~9 crossings/day | a risk filter, unlikely to survive costs |
| 06 one trade per day | entry-clock ladder, first-fire, sell-the-session-from-the-open | 15:30 entry 1.83 is the 99.8th pct of entry clocks; first-fire -0.39 / 0.32; sell-from-open -0.27 | every early entry loses; the session premium is the close, and there it is the sign not the short | reject |
| 06 | flat at the close on FOMC / month-end days (84 of 864) (notebook) | sign(s) 1.67 -> 2.13; always short 0.19 -> 0.70; hybrid 3.23 -> 3.68 | sign(s): t of the daily difference 1.73, dSharpe 95% percentile [+0.07, +0.92] but basic [+0.00, +0.85], a knife edge; always short: t 2.75, both intervals exclude zero; flags found in-sample on the parked event filter | needs a forward test |
| 07 remaining-session recalibration | causal level calibration of the multi-bar forecast against the realized remainder (mean, median, pooled maps) | best pre-close clock 1.1-1.4 mid, <= 1.0 crossed | the median map lands level and buy rate on the oracle's, yet no clock before 15:00 is significant, no interval excludes zero, family-wise best-of-ten at the 16th-64th pct; the pooled map breaks the close | reject |
| 07a audit | independent rebuild of 07 | matches | fit uses prior days only (0/10 perturbation violations); mid-day signs flip across admissible variants and windows; the miss is the forecast's day-by-day ranking, not its level | reject |
| 08 hold to close | short one nearest-OTM straddle at E in {10:00..15:00}, hold the same strikes to 15:30, then sign(s): short days hold to settlement with no 15:30 transaction (1 crossing), long days buy back and stay flat or flip (2-3 crossings) (study, 866 days) | before 15:00: -0.4 to +0.8 at the midpoint, negative at the crossed spread; 15:00: 1.8-1.9 mid / 1.0-1.1 crossed vs the close trade alone 1.34 / 0.87 | REJECTED before 15:00: a held straddle is not the sum of re-picked one-bar shorts (drift cost t -2.4 to -3.0; a 10:00 straddle is 23 pts, 4.6 strikes, off the money by 15:30 on the median day). 15:00 is the close trade plus one bar of always-short carry (identity on long days): placebo 99th percentile but the gain over the close trade has intervals including zero at both fills, best of 24 cells - unresolved, not adopted. Files 08_hold_to_close.{py,md}, results/atm_straddle_intraday/proposals/08/. |

## Recommendation

- At midpoint fills the best intraday construction is always short on every
  bar before the close plus sign(s) at 15:30 (3.23 on 866 days, t 5.98),
  replacing the unit-median hybrid with one construction fewer.
- At realistic fills the intraday legs are a cost, not an edge: every rule is
  negative across the day once the spread is paid (always short -1.92,
  sign(s) -2.13, hybrid -0.89). The trade that survives costs is the close
  trade of the RV-IV notebook: at the crossed spread the settlement leg alone
  scores +1.32 for sign(s) and +1.31 for the hybrid, against -0.12 for always
  short --- the settlement leg survives the spread only when it is sized by
  the sign.
- The remaining-session implied IS a price and it is rich at every clock (04a),
  and recalibrating the forecast's level to the realized remainder does not
  help (07/07a): before the last hour the forecast lacks day-by-day sign
  information, which no rescaling of a one-bar forecast can supply. Only a
  genuinely multi-step remainder forecast (the main repository's direct
  per-bar horizons, not in this repo) could reopen the question.
- The calendar filter at the close (06) is the one candidate that improves the
  surviving trade, pending a forward test.
