# Weights as a function of the forecast gap: Kelly-type sizing of the 15:30 last-30-min straddle trade (A3), 2026-09-29

Sources: `kelly_rules.csv` (one row per forecast × rule × Kelly fraction × fill), `kelly_summary.csv` (counts of paired intervals above / below the sign(s) rule per window × cap × rule), `kelly_robustness.csv` (every window × cap × bin variant), `kelly_bins.csv` (the conditional mean and variance by gap bin at the last refit); script `experiments/close_kelly_sizing.py`. Every number below is read from those files.

## Setup
- 15 forecasts: the paper's 8 and the 7 per-bar linear ones. Scorer: the deck's (§7 recalibration, the one §10 and §15 use) — not the research scorer, so these Sharpes sit next to §10/§15, not next to the master table.
- Every rule is scored on the same 614 of the 866 days, after a 252-day warm-up (2021-08-11 → 2024-04-30), at midpoint and crossed fills. All quantities at day t use days before t only.
- Rules: (a) Kelly from a causal estimate of the trade return's conditional mean and variance given the gap s — by |s| terciles and by sign, or by a regression of the return on s and |s| (with and without a sign intercept); (b) Kelly by the sign of s alone (log-optimal, and mean-variance); (c) proportional to s at the sign(s) Kelly stake; (d) always-short Kelly; plus the sign(s) rule itself levered at its own Kelly stake. Full, ½ and ¼ Kelly. Primary cap: the history ruin bound (the stake that would have ruined on the worst day seen so far); also the exchange-margin cap and no cap; windows expanding, trailing 504, trailing 252.
- Intervals: paired circular block bootstrap (block 21, 2,000 draws, seed 0), the same draws as `vrp_sized_rules.csv`.

## Results: paired intervals vs the sign(s) rule
- Primary specification (expanding window, history-ruin-bound cap): of 720 Kelly rows (8 rules × full/½/¼ × 2 fills × 15 forecasts), **0 have a paired Sharpe interval above sign(s) and 375 have one below** (`kelly_summary.csv`).
- Every window × cap × bin variant: 0 of 6,750 above, 2,677 below (`kelly_robustness.csv`).
- Growth against sign(s) at §15's fixed 3 % of wealth: 0 of 6,750 above, 2,279 below.

Block-diagonal ridge, ½-Kelly, midpoint (ΔSharpe vs sign(s) at 1.54, 95 % paired interval; `kelly_rules.csv`):

| rule | Sharpe | ΔSharpe vs sign(s) |
|---|---|---|
| sign(s), Kelly-levered | 1.35 | −0.19 [−0.40, +0.04] |
| (b) Kelly by sign of s, log-optimal | 1.16 | −0.38 [−0.74, +0.03] |
| (b′) Kelly by sign of s, mean-variance | 1.19 | −0.35 [−0.79, +0.19] |
| (a) sign × \|s\| terciles | 0.06 | −1.48 [−2.35, −0.59] |
| (a) regression on s, \|s\| | −0.18 | −1.72 [−3.18, −0.19] |
| (a) same regression with a sign intercept | 1.22 | −0.32 [−0.78, +0.27] |
| (c) proportional to s at the sign(s) Kelly stake | 0.80 | −0.74 [−2.05, +0.74] |
| (d) always-short Kelly | −0.16 | −1.70 [−3.23, −0.11] |

- Crossed fills: sign(s) 1.18; Kelly-levered sign(s) 0.80 (−0.38 [−0.96, +0.21]); (b) 0.49.
- Per-bar ridge, live-feasible inputs: Kelly-levered sign(s) 1.33 vs sign(s) 1.51, −0.19 [−0.33, −0.03] — an interval below zero.

## The conditional payoff by gap size (`kelly_bins.csv`, last refit)
- On buy days the largest-gap tercile is the weakest of the three in 11 of 15 forecasts; the terciles rise with the gap in 0 of 15.
- On sell days the middle tercile pays the short most in 12 of 15; the largest-gap tercile in 0.
- Block-diagonal ridge, long-straddle mean return by tercile: +0.146 / +0.171 / −0.016 on buy days; −0.095 / −0.154 / −0.047 on sell days.
- Magnitude-driven rules take the side opposite to sign(s) on many days: 24 % for the tercile rule, 61 % for the regression without a sign intercept (which cannot represent the jump at s = 0). This agrees with §10 (proportional and rank sizing) and study 71.

## Growth and drawdown
- Full-Kelly sign(s) has the highest median growth rate (0.838 vs 0.674 for sign(s) at 3 %; higher in 13 of 15 forecasts), but no interval is above zero, and its median maximum drawdown is −0.76 of wealth vs −0.33.
- The ½-Kelly stake averages 3.4 % of wealth, so §15's fixed 3 % is about half-Kelly.

## Other findings
- At crossed fills zero-sized days appear where both sides lost on average: 18 % of days for (b); 84 % for always-short Kelly, which also loses (Sharpe −1.15).
- Only one day ruins accounts under the history-ruin-bound cap: 2021-11-22, worse than any earlier deck day. It ruins 16 binned-rule paths across windows; in the primary specification, 1.
- Robustness: 0 intervals above sign(s) in every window (expanding / trailing 504 / trailing 252) × cap (history ruin bound / exchange margin / none). The exchange-margin cap binds on a median 59 % of days.

## Checks (all passed)
- The 15 books share the 866 days.
- sign(s) reproduces §10's Sharpe on all 866 days for every forecast.
- sign(s) and the proportional rule reproduce `results/atm_straddle_0dte_1530/vrp_sized_rules.csv` to 1e-9 (same draws).
- Multiplying the future by ×50 leaves every fraction and cap up to that day unchanged (nothing after the entry time enters a stake).
- A rerun reproduced the CSVs byte for byte.

## Not done
- The research scorer (16:00-bar recalibration) was not run here; the master table's headline Sharpes are on that scorer.
- These rules are not yet in the paper; `writeup/D1_CHANGELOG_2026-09-29.md` §5 lists that as an open item for the user.
