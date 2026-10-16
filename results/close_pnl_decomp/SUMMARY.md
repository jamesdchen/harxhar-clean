# Where the P&L of the last-30-min straddle trade comes from

Checklist item B1 (2026-09-29). Script: `experiments/close_pnl_decomposition.py`; every number below is read by that script from the CSVs in this directory.

## Set-up

- **The trade.** At 15:30 ET one **straddle** — the nearest out-of-the-money call plus the nearest out-of-the-money put, same-day expiry, one position — held to the cash settlement at the official close. The **sign(s)** rule buys it when the forecast of the 15:30–16:00 realized variance exceeds the variance implied by the 15:30 quotes, and sells it otherwise. **Always short** sells it every day and uses no forecast.
- **Days.** The same 866 days for every series, 2020-01-03 to 2024-04-30 (the rv_iv notebook's frame, 200 trade days a year).
- **Units.** Premium units: one unit of premium staked every day, daily returns summed, never compounded (the notebook's §13 convention). A day's P&L is q(X/P − 1) with q = ±1, X the settlement value and P the entry price. Sharpe ratios are annualized by √252 per trade day, as in the deck.
- **Fills.** Midpoint, and crossed spread (buy at the ask, sell at the bid — the whole quoted spread once; a bound, not an estimate).
- **Forecasts.** Per-bar ridge on the live-feasible set (the 16 columns a 15:30 forecaster can rebuild; the lead series below), per-bar ridge on all features, and the block-diagonal ridge (the paper's headline forecast). The paper's HAR + calendar OLS baseline is in every CSV as a fourth sign(s) series.
- **Scorer.** The rv_iv notebook's (recalibration on all 13 session bars), because the per-day tables come from it. The two per-bar ridges are run separately on their plain 16:00 forecast f²·B (no recalibration; 2026-10-06, user decision) in `research_scorer/` and summarized in §7; the two are never mixed in one table.
- **No intervals.** Every number is a point estimate (the repository's bootstrap is disabled, commit b761b28; the CSVs carry no `_lo` / `_hi` columns).

| series | P&L, mid | Sharpe, mid | P&L, crossed | Sharpe, crossed | buy days | max drawdown, mid |
|---|---:|---:|---:|---:|---:|---:|
| sign(s): per-bar ridge (live-feasible) | +103.0 | 1.68 | +74.1 | 1.22 | 348 | -15.1 |
| sign(s): per-bar ridge (all features) | +108.5 | 1.77 | +79.6 | 1.31 | 316 | -10.6 |
| sign(s): block-diagonal ridge (paper headline) | +82.0 | 1.34 | +53.1 | 0.87 | 346 | -17.7 |
| sign(s): HAR + calendar OLS (paper baseline) | +59.4 | 0.97 | +30.3 | 0.49 | 286 | -16.6 |
| always short | +12.5 | 0.20 | -17.6 | -0.27 | 0 | -19.3 |

## 1. Tail days: a few days carry the P&L — the market's tail, the rule's selection

- The live-feasible sign(s) makes 103.0 premium units at the midpoint. Its 10 best days carry 46 % of that and its 20 best 73 %; without the 20 best days the other 846 days make +28.0 (Sharpe 0.56, t 1.02). At the crossed spread the 20 best days carry 97 % and the other days make +2.4.
- 12 days make half of the total, 24 make 80 %, and 34 days (3.9 % of 866) add up to the whole of it: the remaining days net to zero.
- Every one of the 20 best days of each of the three sign(s) series is a **buy** day, and every one of the 20 worst is a **sell** day. Of the live-feasible's 20 best days 2 are month-ends and 0 FOMC days (`tail_days.csv` lists them).
- **Whose tail is it?** Always short shows the market's side of it: it earns at most +1 a day (the premium, kept whole on 193 days) and its 20 worst days make -83.5 against a total of +12.5; without them it would make +96.1. The concentration comes from the straddle's payoff — a few large moves in the last half hour — not from the rule. What the rule adds is *which* of those days it owns: of the 20 days the straddle paid most, the live-feasible sign(s) bought 12 against 8.0 expected at its buy rate (hypergeometric p = 0.056); of the 50, 31 against 20.1 (p = 0.001).

| series (mid) | P&L | best 10: share | best 20: share | P&L without best 20 | Sharpe without best 20 | worst 20 | days for 100 % | biggest-20 payoff days bought (expected; p) | biggest-50 (expected; p) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| per-bar ridge (live-feasible) | +103.0 | 46 % | 73 % | +28.0 | 0.56 | -57.9 | 34 | 12 (8.0; 0.056) | 31 (20.1; 0.001) |
| per-bar ridge (all features) | +108.5 | 45 % | 72 % | +30.6 | 0.62 | -55.0 | 36 | 13 (7.3; 0.008) | 30 (18.2; 0.000) |
| block-diagonal ridge (paper headline) | +82.0 | 56 % | 89 % | +9.0 | 0.18 | -60.0 | 25 | 11 (8.0; 0.124) | 31 (20.0; 0.001) |
| HAR + calendar OLS (paper baseline) | +59.4 | 73 % | 113 % | -7.7 | -0.15 | -66.2 | 17 | 7 (6.6; 0.509) | 22 (16.5; 0.063) |
| always short | +12.5 | ties at +1 | ties at +1 | -7.5 | -0.12 | -83.5 | 13 | 0 (never buys) | 0 (never buys) |

## 2. Calendar: the P&L comes from ordinary days, not from month-ends

- **Month-ends** (52 last sessions of a month in the frame): the straddle's long side pays on them on average — always short makes -24.5 there — and the rule sells most of them (the live-feasible sign(s) buys 10), so it makes -10.7 on month-ends. Month-ends are not the source of the sign(s) P&L: on the 735 days more than one session away from a month-end it makes +106.8, 104 % of its total.
- **FOMC statement days** (33 in the frame; the release file's flags through 2023-11-01 plus the Federal Reserve's published schedule after it): -0.6 for the live-feasible sign(s), -5.0 for always short — nothing measurable.
- **Day of week.** Thursday is the one weekday on which all three sign(s) series lose: per-bar ridge (live-feasible) -10.2; per-bar ridge (all features) -27.7; block-diagonal ridge (paper headline) -9.3 over 109 Thursdays (five weekdays times three series were looked at, so read it as a flag, not a finding). Mondays pay most for the live-feasible series (+35.8; always short -24.8).
- **Years.** Live-feasible sign(s), P&L per day at the midpoint: 2020 +0.135; 2021 +0.097; 2022 +0.016; 2023 +0.198; 2024 +0.167. Always short: 2020 -0.030; 2021 -0.041; 2022 +0.102; 2023 +0.064; 2024 -0.173.
- **Listing era** (every-session expirations from 2022-05-13 in this frame): Mon/Wed/Fri + month-end listings 1.50 Sharpe; every-session listings 1.84 Sharpe.

| cell (mid) | days | live-feasible: buys | live-feasible: P&L | all features: P&L | block-diagonal ridge: P&L | always short: P&L |
|---|---:|---:|---:|---:|---:|---:|
| month-end (last session) | 52 | 10 | -10.7 | -3.6 | -15.3 | -24.5 |
| month-end T-1 | 35 | 13 | -2.6 | -2.8 | -3.7 | +6.7 |
| month-end T+1 | 44 | 18 | +9.4 | +6.8 | +4.9 | +10.6 |
| not within one session of a month-end | 735 | 307 | +106.8 | +108.1 | +96.2 | +19.7 |
| FOMC statement day | 33 | 11 | -0.6 | -1.8 | -4.2 | -5.0 |
| quarterly opex (3rd Fri Mar/Jun/Sep/Dec) | 17 | 4 | +2.0 | +3.8 | +3.8 | +5.5 |
| monthly opex (other 3rd Fridays) | 35 | 7 | +7.7 | +5.0 | +9.8 | +9.0 |
| day of week: Mon | 199 | 93 | +35.8 | +44.4 | +46.8 | -24.8 |
| day of week: Thu | 109 | 46 | -10.2 | -27.7 | -9.3 | +8.1 |

## 3. Regimes (known at the previous close)

- **VIX tercile**: the previous session's VIX close against the terciles of every VIX close since 1990 up to that session (cut-offs 14.71–15.25 and 20.53–21.22 over the frame). **Realized-vol tercile**: the S&P's 21-session close-to-close realized volatility up to the previous close, against its own terciles since 1990 (cut-offs 10.3–10.6 % and 15.8–16.5 % a year). The terciles are levels against a long history, so the 2020–2024 frame is unbalanced across them (172, 299, 395 days by VIX tercile).
- In the **high-VIX tercile** (395 days) the live-feasible sign(s) makes +0.055 a day and always short +0.054; the paired difference over those days is +0.4. The low- and mid-VIX terciles carry 41 % and 59 % of the rule's advantage over always short; in stressed markets the two make the same.

| regime (mid) | days | live-feasible: per day | live-feasible: Sharpe | block-diagonal ridge: Sharpe | always short: per day | always short: Sharpe |
|---|---:|---:|---:|---:|---:|---:|
| VIX tercile: low | 172 | +0.132 | 1.61 | -0.21 | -0.081 | -0.99 |
| VIX tercile: mid | 299 | +0.195 | 2.48 | 2.58 | +0.017 | 0.21 |
| VIX tercile: high | 395 | +0.055 | 0.96 | 1.07 | +0.054 | 0.94 |
| S&P 21-day realized vol tercile: low | 156 | +0.268 | 2.81 | 2.89 | -0.054 | -0.56 |
| S&P 21-day realized vol tercile: mid | 321 | +0.098 | 1.42 | 0.96 | +0.028 | 0.40 |
| S&P 21-day realized vol tercile: high | 389 | +0.076 | 1.29 | 0.76 | +0.031 | 0.52 |

## 4. Hit rate against payoff

- The live-feasible sign(s) is right on 0.546 of days — less often than always short (0.618), which wins every day the straddle settles below its premium. The difference is the payoff: sign(s) makes 0.870 on a winning day and loses 0.787 on a losing day (payoff ratio 1.105, so it breaks even at a 0.475 hit rate); always short makes 0.685 and loses 1.072 (payoff 0.639, break-even 0.610).
- **Buy days** (348): hit rate 0.411, payoff ratio 1.89, P&L +45.2 — a long-shot profile. **Sell days** (518): hit rate 0.637, payoff ratio 0.76, P&L +57.8. Both legs earn. On the days the rule sells, it and always short hold the same position and make +0.112 a day; on the days the rule buys, always short makes -0.130 a day — the days it moves to the long side are the ones that hurt the seller.

| series (mid) | hit rate | payoff ratio | break-even hit | buy days: P&L per day | buy days: hit | sell days: P&L per day | sell days: hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| per-bar ridge (live-feasible) | 0.546 | 1.11 | 0.475 | +0.130 | 0.411 | +0.112 | 0.637 |
| per-bar ridge (all features) | 0.553 | 1.09 | 0.478 | +0.152 | 0.411 | +0.110 | 0.635 |
| block-diagonal ridge (paper headline) | 0.544 | 1.05 | 0.487 | +0.100 | 0.408 | +0.091 | 0.635 |
| HAR + calendar OLS (paper baseline) | 0.551 | 0.96 | 0.510 | +0.082 | 0.399 | +0.062 | 0.626 |
| always short | 0.618 | 0.64 | 0.610 | — | — | +0.014 | 0.618 |

## 5. Path: drawdowns, stability, serial dependence

- Live-feasible sign(s), midpoint: deepest drawdown -15.1 premium units (peak 2021-06-09, trough 2021-11-08, recovered 2022-01-05, 92 sessions), 13.5 daily standard deviations; longest time under water 137 sessions (2022-08-24 to 2023-03-15). Always short's deepest drawdown is -19.3 and its longest spell under water 435 sessions (2020-01-06 to 2022-08-11).
- Rolling Sharpe (trailing windows, midpoint): the 252-session Sharpe of the live-feasible sign(s) is positive in 100.0 % of windows (minimum 0.33, window ending 2022-11-14); the 126-session one in 96.6 %; the 63-session one in 78.7 % (minimum -3.25, ending 2021-10-29). Always short: 73.3 % of 252-session windows positive.
- Serial dependence: the three sign(s) series show none — smallest Ljung–Box p over lags 5, 10, 21: per-bar ridge (live-feasible) 0.174, per-bar ridge (all features) 0.618, block-diagonal ridge (paper headline) 0.198; variance ratios (Lo–MacKinlay, heteroskedasticity-robust z) with |z| > 1.96: HAR + calendar OLS (paper baseline): VR(10) 0.76 (z -2.21); always short: VR(2) 0.93 (z -2.18), VR(5) 0.86 (z -2.03), VR(10) 0.75 (z -2.27), VR(21) 0.63 (z -2.26). A variance ratio below one means multi-day P&L varies less than the daily P&L implies: losing days tend to be followed by recovering ones. Autocorrelations outside ±1.96/√866 at lags 1–10: per-bar ridge (live-feasible) 1 (lag 8), per-bar ridge (all features) 0, block-diagonal ridge (paper headline) 0 of 10 each — about what one in twenty by chance gives.

| series | fill | max drawdown | peak → trough → recovery | sessions to recover | 252-session Sharpe > 0 | Ljung–Box p (5/10/21) | VR(21), z |
|---|---|---:|---|---:|---:|---|---:|
| per-bar ridge (live-feasible) | mid | -15.1 | 2021-06-09 → 2021-11-08 → 2022-01-05 | 92 | 100.0 % | 0.80 / 0.17 / 0.63 | 1.06, 0.34 |
| per-bar ridge (live-feasible) | crossed | -18.1 | 2021-06-09 → 2021-11-08 → 2022-01-21 | 99 | 99.7 % | 0.78 / 0.17 / 0.64 | 1.07, 0.41 |
| per-bar ridge (all features) | mid | -10.6 | 2022-08-24 → 2023-04-10 → 2023-07-07 | 215 | 100.0 % | 0.62 / 0.79 / 0.85 | 0.78, -1.33 |
| per-bar ridge (all features) | crossed | -14.4 | 2022-08-24 → 2023-04-10 → 2023-08-02 | 233 | 99.0 % | 0.62 / 0.80 / 0.83 | 0.78, -1.30 |
| block-diagonal ridge (paper headline) | mid | -17.7 | 2020-01-08 → 2020-05-04 → 2020-10-12 | 122 | 100.0 % | 0.31 / 0.20 / 0.68 | 1.09, 0.53 |
| block-diagonal ridge (paper headline) | crossed | -22.2 | 2020-01-08 → 2020-06-15 → 2020-12-02 | 143 | 94.5 % | 0.31 / 0.20 / 0.67 | 1.12, 0.74 |
| always short | mid | -19.3 | 2023-09-19 → 2024-04-30 → not recovered | n/a | 73.3 % | 0.12 / 0.34 / 0.23 | 0.63, -2.26 |
| always short | crossed | -35.9 | 2020-01-06 → 2022-04-01 → 2023-05-08 | 620 | 62.1 % | 0.13 / 0.37 / 0.26 | 0.64, -2.19 |

## 6. The difference sign(s) − always short, day by day

The two portfolios hold the same short straddle on every day sign(s) sells, so the paired difference is zero on those days and 2R on the days sign(s) buys (at the midpoint; at the crossed spread it is the ask-bought long minus the bid-sold short). The whole difference is therefore a statement about the buy days.

- Live-feasible sign(s) minus always short: +90.4 premium units over 866 days (+0.104 a day), all of it on the 348 buy days.
- Without its 10 best days the difference is -3.6; those 10 days add +94.0 (104 % of the total).
- Without its 20 best days the difference is -59.4; those 20 days add +149.9 (166 % of the total).
- Month-ends add +13.8 (15 %; 10 buys on 52 month-ends); days more than one session from a month-end add +87.1 (96 %).

| cell (live-feasible, mid) | days | buys | difference | share | difference without the cell |
|---|---:|---:|---:|---:|---:|
| all days | 866 | 348 | +90.4 | 100 % | — |
| top 10 days of the difference | 10 | 10 | +94.0 | 104 % | -3.6 |
| top 20 days of the difference | 20 | 20 | +149.9 | 166 % | -59.4 |
| bottom 20 days of the difference | 20 | 20 | -40.0 | -44 % | +130.4 |
| month-end (last session) | 52 | 10 | +13.8 | 15 % | +76.6 |
| month-end T-1 | 35 | 13 | -9.3 | -10 % | +99.8 |
| month-end T+1 | 44 | 18 | -1.2 | -1 % | +91.6 |
| not within one session of a month-end | 735 | 307 | +87.1 | 96 % | +3.3 |
| FOMC statement day | 33 | 11 | +4.4 | 5 % | +86.1 |
| day of week: Mon | 199 | 93 | +60.7 | 67 % | +29.8 |
| day of week: Thu | 109 | 46 | -18.3 | -20 % | +108.8 |
| year 2020 | 158 | 69 | +26.0 | 29 % | +64.4 |
| year 2021 | 158 | 50 | +21.8 | 24 % | +68.7 |
| year 2022 | 219 | 51 | -18.9 | -21 % | +109.3 |
| year 2023 | 248 | 135 | +33.3 | 37 % | +57.1 |
| year 2024 | 83 | 43 | +28.2 | 31 % | +62.3 |
| VIX tercile: low | 172 | 106 | +36.7 | 41 % | +53.7 |
| VIX tercile: mid | 299 | 117 | +53.3 | 59 % | +37.2 |
| VIX tercile: high | 395 | 125 | +0.4 | 0 % | +90.0 |
| S&P 21-day realized vol tercile: low | 156 | 75 | +50.3 | 56 % | +40.1 |
| S&P 21-day realized vol tercile: mid | 321 | 157 | +22.6 | 25 % | +67.9 |
| S&P 21-day realized vol tercile: high | 389 | 116 | +17.6 | 19 % | +72.9 |

| sign(s) series | fill | difference vs always short | per day | without its 10 best days | month-ends | high-VIX tercile |
|---|---|---:|---:|---:|---:|---:|
| per-bar ridge (live-feasible) | mid | +90.4 | +0.104 | -3.6 | +13.8 | +0.4 |
| per-bar ridge (live-feasible) | crossed | +91.7 | +0.106 | -2.5 | +13.9 | +1.1 |
| per-bar ridge (all features) | mid | +95.9 | +0.111 | -1.7 | +20.9 | -3.9 |
| per-bar ridge (all features) | crossed | +97.1 | +0.112 | -0.8 | +21.0 | -3.3 |
| block-diagonal ridge (paper headline) | mid | +69.5 | +0.080 | -22.3 | +9.2 | +2.8 |
| block-diagonal ridge (paper headline) | crossed | +70.7 | +0.082 | -21.2 | +9.2 | +3.4 |
| HAR + calendar OLS (paper baseline) | mid | +46.9 | +0.054 | -39.9 | +4.2 | -2.5 |
| HAR + calendar OLS (paper baseline) | crossed | +47.9 | +0.055 | -39.0 | +4.2 | -2.0 |

## 7. The same decomposition on the plain 16:00 forecast (no recalibration)

The 15:30 sign(s) rule reads the per-bar ridge's 16:00 forecast as its plain back-transform f²·B (f the adjusted-scale forecast, B the diurnal baseline), never recalibrated: q = sign(f²·B − implied variance). Only the two per-bar ridges have rows here; full tables in `research_scorer/` (folder name kept for its readers).

| series | same position as the notebook scorer | Sharpe, mid | Sharpe, crossed | best-20 share, mid | days for 100 % | biggest-20 payoff days bought | difference vs always short, mid | without its 10 best days |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| per-bar ridge (live-feasible) | 87.6 % | 1.79 | 1.31 | 64 % | 44 | 10 (6.2; 0.058) | +97.0 | +9.8 |
| per-bar ridge (all features) | 89.6 % | 1.71 | 1.22 | 65 % | 44 | 8 (5.5; 0.161) | +91.9 | +5.1 |

## What this does not cover

- The frame ends 2024-04-30. Studies 85–90 (other session; `writeup/AUDIT_2026-09-18.md`) found no daily edge after the research span; nothing here speaks to those years.
- The rule holds one unit whatever the size of the forecast, so a rare extreme forecast can only turn a day into a buy; over these days the recalibrated forecast is at most 2.14× (per-bar ridge (live-feasible)), 3.72× (per-bar ridge (all features)), 2.43× (block-diagonal ridge (paper headline)), 2.29× (HAR + calendar OLS (paper baseline)) the implied variance — no extreme values enter.
- Premium units weight every day alike. The dollar and index-point versions are in the notebook's §13.

## Files

- `tail_concentration.csv` — block 1: shares of the best/worst k days, P&L without them, the market's biggest-payoff days the rule bought
- `tail_days.csv` — the 20 best and 20 worst days of each series with their flags
- `calendar_cells.csv` — block 2: month-end (T, T−1, T+1), FOMC, opex, weekday, year, listing era
- `regime_cells.csv` — block 3: VIX and realized-vol terciles, years
- `hit_payoff.csv` — block 4: hit rate, payoff ratio, buy/sell split, days for 50/80/100 %
- `path_summary.csv, path_drawdowns.csv, path_rolling_sharpe.csv, path_autocorr.csv` — block 5: drawdowns, rolling Sharpe, autocorrelation, Ljung–Box, variance ratios
- `diff_attribution.csv` — block 6: sign(s) − always short by cell, paired intervals
- `daily_pnl_and_flags.csv` — the day-level table every block is cut from
- `market_daily_closes.csv` — ^GSPC and ^VIX closes since 1990 (session calendar and regimes)
- `fig_cum_pnl_tails.png, fig_regime_bars.png, fig_drawdown.png` — figures
- `research_scorer/` — the same CSVs on the plain 16:00 forecast f²·B, no recalibration (two per-bar ridges; folder name kept)
