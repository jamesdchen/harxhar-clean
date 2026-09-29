# Strategy variations of the last-30-min trade — summary (F1), 2026-09-29

Sources: `headline_per_straddle_premium.csv` (the full tables), `summary_all.csv` (both return frames), `paired_vs_straddle.csv`, `cost_line.csv`, `variation_catalog.csv`, `VARIATIONS.md` (the definitions); script `experiments/strategy_variations_1530.py`; figure `sharpe_by_variation.png`. Every number below is read from those files.

## Setup
- The on-screen list is not in the repository; the standard set was used (`VARIATIONS.md`): the straddle; strangles one strike wide and 0.5 %, 1 % and 2 % out of the money; iron butterflies and iron condors with wings w10 / w25 / w50; the put vertical and the call vertical alone; call and put butterflies at the money; put and call 1×2 ratio spreads; the straddle with wings on selling days only ("straddle + wing hedge"). 32 same-day-expiry variations, entered at the 15:30 quotes and cash-settled at the official close, on the straddle's 866 days (2020-01-03 to 2024-04-30). Calendars (not same-day) and broken-wing structures were not run.
- Each is bought when s > 0 and sold when s < 0; the wing hedge holds the straddle on buying days and the short iron butterfly on selling days. Return is P&L over that day's straddle premium at the same fill, so the straddle row is the straddle trade's return (`summary_all.csv` also carries an own-notional frame). Fills: mid and crossed. Intervals: 95 % paired circular block bootstrap (block 21, 2000 draws) against the straddle under the same rule and fill.
- Two forecasts, kept in separate tables and never compared with each other: the per-bar ridge on the live-feasible inputs (16:00-bar recalibration, the research scorer) and the block-diagonal ridge as in the paper (deck scorer). Checks: the straddle rows reproduce the research scorer (1.9025 mid / 1.4396 crossed) and the deck (1.3383 / 0.8696). The forecast's large values need no treatment: only its sign enters (max forecast / implied ratio 2.2).
- One pass over the chain produced `legs_1530_1600.parquet` (2.4 MB, not committed).

## Results by variation (paired vs the straddle; no variation's interval is above zero under either forecast or fill)
- Per-bar ridge, mid: iron butterfly w50 1.94 (+0.04 [−0.04, +0.14]); straddle 1.90 [0.87, 2.92]; wing hedge w50 1.85 (−0.05 [−0.12, +0.06]); condor w50 1.85 (−0.05 [−0.49, +0.34]); iron butterfly w25 1.84 (−0.06 [−0.23, +0.12]); condor w25 1.82; one-strike strangle 1.81 (−0.10 [−0.54, +0.31]).
- Per-bar ridge, crossed: straddle 1.44 [0.38, 2.48] is first; one-strike strangle 1.37 (−0.07 [−0.53, +0.34]); iron butterfly w50 1.31 (−0.13 [−0.21, −0.03]); wing hedge w50 1.30 (−0.14 [−0.22, −0.03]); condor w50 1.18 (−0.26 [−0.72, +0.14]).
- Call side alone: call vertical mid Δ −1.08 [−2.00, −0.13], −1.03 [−1.88, −0.24], −0.94 [−1.78, −0.17] at w10 / w25 / w50; put vertical mid 1.09 / 1.31 / 1.36 (intervals include zero).
- Paper's forecast: the put vertical w50 is the best row, 1.51 mid (+0.18 [−0.57, +0.90]) and 1.14 crossed (+0.27 [−0.48, +1.01]) against the straddle's 1.34 / 0.87; call vertical w50 0.07 (−1.27 [−2.21, −0.37]). The put side carries more of the difference.
- Strangles out of the money, mid / crossed: 0.5 % 1.19 / 0.33; 1 % 0.04 / −1.42; 2 % −0.57 / −3.01.
- 1×2 ratio spreads: −0.28 to −1.37 mid, all 12 intervals below zero (per-bar ridge).
- Butterflies, mid: call 1.07 / 1.48 / 1.68, put 1.05 / 1.51 / 1.74; crossed: call −1.33 / −0.68 / −0.68, put −1.50 / −0.81 / −1.13, all intervals below zero. The median crossing cost of the three strikes is 0.525–0.800 index points, against 0.150 for the straddle.
- Tail: the straddle's best 21 days carry 0.70 of total P&L at mid; the other days average +0.042 (+0.012 crossed). Iron butterfly w10 at mid: worst day −2.43 vs −5.42, maximum drawdown −8.60 vs −12.47. At w25 / w50 the worst day is unchanged at −5.44. Crossed, the w10 drawdown is −23.97 vs −14.74.

## Cost line (`cost_line.csv`, `paired_vs_straddle.csv`) — the earlier "cost at every width" reading, re-measured
- Crossed: all 6 wing rows (iron butterfly and wing hedge at w10 / w25 / w50) sit below the straddle with intervals excluding zero, under the per-bar ridge, the paper's forecast and always short. Iron butterfly −0.86 [−1.39, −0.37], −0.31 [−0.48, −0.13], −0.13 [−0.21, −0.03].
- Mid: the cost is resolved in 2 of 6 rows for the per-bar ridge (wing hedge w10 −0.46 [−0.85, −0.10], w25 −0.21 [−0.38, −0.01]), 0 of 6 for the paper's forecast, 6 of 6 for always short. At w50 the wing is nearly free: the close lands beyond the wings on 22.3 / 4.0 / 1.0 % of days at w10 / w25 / w50.
- Median entry cost as a share of the structure's own premium: straddle 2.9 %; one-strike strangle 4.5 %; strangles 0.5 / 1 / 2 % OTM 25 / 67 / 100 %; iron butterfly 6.1 / 4.3 / 4.0 %; iron condor 10.4 / 7.7 / 7.0 %; verticals 4.0–6.5 %; ratios 5.1–14.3 %. Butterflies cost 9.7–13.5 % of the straddle premium.
- Zero wing bid: the w50 wing on 781 of 866 days; the 2 % strangle on 845.
- Sharpe lost from mid to crossed: straddle 0.46; iron butterfly 1.08 / 0.71 / 0.63; condor 1.13 / 0.75 / 0.67; butterflies 2.16–2.87.

## Always short (no forecast)
- Straddle 0.20 mid / −0.27 crossed.
- 0.5 % OTM strangle sold every day: 1.78 [0.37, 4.49] mid (Δ vs the straddle +1.57 [+0.23, +3.99]; hit rate 0.946; worst day −3.65), but 0.83 [−0.45, 3.11] crossed.
- 2 % OTM strangle 2.64 mid / −0.55 crossed.
- Iron butterfly, condor, credit put vertical and butterflies are all ≤ −0.26 crossed.
- Front put ratio w10 1.42 [0.45, 2.37] mid / 0.51 [−0.46, 1.45] crossed; hit rate 0.363.

## Caveats
- Crossed is a worst-case quoted fill (pay the ask, hit the bid on every leg).
- The own-notional frame (`summary_all.csv`) is degenerate for far strangles: crossed keeps 656 / 401 days.
- Butterflies at w25 / w50 are scored on 861 days.
- The close and the 16:00 tape disagree on a straddle leg on 29 days; settlement is the official close.
- `writeup/generated/strategy_variations.tex` (Appendix D.13) defines one label twice — harmless, to be de-duplicated.
