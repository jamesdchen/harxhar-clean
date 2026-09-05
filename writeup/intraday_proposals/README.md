# Intraday Sharpe: diagnosis and proposals (2026-09-04)

> Provenance note (2026-09-05): rows marked (notebook) quote the intraday notebook run of 2026-09-04. The notebook was re-executed on 2026-09-05 after the library fix wave (session-date fit mask, 2001-2025 early-close calendar): pooled always short 1.90, sign(s) 1.72, hybrid 3.17 at the midpoint; -1.92 / -2.24 / -0.97 at the crossed spread; the close leg's calendar test 1.58 -> 2.04. Refresh of the table rows pending.

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
| 09 last-hour boundary | **pre-registered two-cell replication of 08's 15:00 cell**: cell A = the deck's close trade at 15:30 (sign(s), settled at the official close); cell B = nearest-OTM straddle picked at 15:00, position sign(s_rem) with s_rem = rv_hat/w - IV_hr^2 h, the SAME strikes held to settlement (1 crossing, no 15:30 transaction); comparators always short at each stamp (803 common days) | A 1.64 mid / 1.20 crossed, B 1.09 / 0.75 | gate: the deck's rule table reproduced to 4.1e-07 (sign(s) 1.338322, t 2.480957, always short 0.203779); causality 0/10 perturbation violations; B - A dSharpe -0.54 mid / -0.45 crossed, every interval covers zero (crossed percentile [-1.63, +0.59], basic [-1.50, +0.73]); B's sign at the 98th placebo percentile at both fills; sign agreement 66.1%, hit rate 0.55 (15:00) vs 0.67 (15:30); A wins on all eight forecasts | **REJECT: the frontier is the last bar.** The 15:00-15:30 bar is 6.9% of B's midpoint points (t 0.35) and NEGATIVE crossed (-0.16 pts/day, t -1.28); 93% of B comes from the settlement bar. The head start sells more premium, not more information. Files 09_last_hour_boundary.{py,md}, results/atm_straddle_intraday/proposals/09/. |
| 10 unrun (Check 0 + A-G) | **Check 0**: the four oracles separated by clock on the notebook's own 10,387-bar frame, then the published oracle COSTED at the crossed spread for the first time; then seven books never run - A nowcast of the bar in flight, B a 30-min implied synthesized from the implied term structure, C a hurdle against the quoted half-spread, D sparse clocks (pre-registered {11:30,12:30,15:00}) plus an honest cost screen, E a 0DTE-1DTE calendar, F fading the implied, G the 15:00 always-short bar stacked on the 15:30 sign(s) settlement | oracle (1) 9.83 mid / **5.91 crossed**; G both lines 2.34 / **1.27** against the settlement leg 1.57 / 1.15; D sparse 2.74 / 0.75; B 1.68 / -2.63; C best -0.30 crossed; F -3.17 to -8.39 crossed | gate: the notebook's two rule tables reproduced to 8.6e-08; causality 0/10 perturbation violations; slice-oracle short on 49.6-72.9% of daytime bars (60.98% pooled), remaining-vs-remaining short on 67.9-75.3% (70.72% pooled, median realized/implied 0.785-0.846); G dSharpe against the close trade +0.7665 mid [+0.358, +1.184] EXCLUDES zero, +0.1162 crossed [-0.299, +0.518] includes zero; G's 15:00 at the 100th placebo percentile of daytime clocks at both fills (median draw +0.758 crossed); D dSharpe +1.17 mid [+0.686, +1.659] excludes zero, -0.41 crossed [-0.868, +0.043]; the first-half cost screen selects NO clock; A and E unconstructible (nothing finer than 30 min; every chain row is 0DTE) | **The hypothesis dies and the ceiling lives.** The daytime oracle is MIXED, not a disguised always-short, and the remaining-session oracle is a short TILT (not 80%+) - two different objects. Costed, the oracle keeps 5.91 crossed and is positive at every clock: information about the bar in flight is worth about three times the spread bill, so the job is a NOWCAST, which this repository has no data to run. Every F_t book that touches more than the last bar is negative crossed. One unresolved candidate: G, 15:00 always short + 15:30 sign(s), 1.27 crossed against 1.15, interval covers zero - forward test, not adoption. **Instruction stands: trade 15:30 sign(s).** Files 10_unrun.{py,md}, results/atm_straddle_intraday/proposals/10/. |
| 11 IV allocation | **pre-registered four-variant study of the implied share w_t**: V0 unconditional (the deck's), V1 event-conditional (FOMC / month-end / third-Friday profiles shrunk n/(n+20) toward it), V2 market-implied (trailing per-clock mean of (IV_rem_t - IV_rem_t+1)/IV_rem_t, renormalized), V3 same-day-conditioned (per-clock regression on today's cumulative RV, iv_hourly and day-of-week, clipped to (0.02, 0.98) and renormalized); 8,815 bars on 735 common days | V0 1.46 / V1 1.22 / V2 1.79 / V3 1.50 mid; -2.41 / -2.63 / -1.96 / -2.36 crossed | gate: the notebook's rule table reproduced to 0 at seven decimals (sign(s) 1.7205907, always short 1.9020389, hybrid 3.1714284); causality 0/10 violations for V0/V1/V2, V3 moves only after the perturbed bar (0 of 65 at or before it, 45 of 55 after); calibration dispersion across clocks 0.0971 (V0) against 0.0919 / 0.1563 / 0.0480; pooled QLIKE 0.1667 against 0.1571 / 0.1737 / 0.1451, DM t -5.72 / +3.37 / -5.78; V2 crossed dSharpe +0.448, percentile [+0.06, +0.82], basic [+0.07, +0.84], but the 49th placebo percentile | **REJECT all three: the allocation is not the constraint.** A share reallocates a fixed remaining implied variance and cannot add day-by-day information. V1 and V3 are better bar-level prices (DM t -5.7, -5.8) and do not move the trade; V2 wins the crossed spread only by shorting more (58.1% of bars against V0's 43.1%) and its Sharpe survives scrambling its own clocks. At 15:30 w = 1 for every variant, so the hybrid (3.171 mid / -0.969 crossed) is untouched. Files 11_iv_allocation.{py,md}, results/atm_straddle_intraday/proposals/11/. |

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
