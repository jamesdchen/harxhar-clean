# Why the intraday Sharpes fall short of the 15:30 projection — diagnosis

Date 2026-09-04. Read-only diagnosis; no notebook was changed. Scripts and logs:
`scratchpad\intraday_gap\{build_work, recon_projection, mechanisms, oracle_vega, forecast_skill}.py` + `.log`,
tables `per_clock.csv`, `signal_quality_by_clock.csv`, `exit_convention_by_clock.csv`, `oracle_vega_by_clock.csv`,
`forecast_skill_by_clock.csv`. Block-diagonal ridge forecast throughout; the intraday `work` frame was rebuilt
exactly as the notebook's sections 4–5b build it (10,387 bars, 866 days, 12 clocks, 9,631 matched-signal rows;
the 15:30 collapse check is exactly 0).

## 1. Reconciliation at 15:30 — no discrepancy

| check (866 common days) | result |
|---|---|
| days: intraday 866 vs RV–IV 871 | 5 days only in RV–IV: 2020-11-27, 2020-12-24, 2022-11-25, 2023-07-03, 2023-11-24 (half sessions, dropped by the intraday notebook) |
| strikes, entry, exit, R identical | 100% of days (|difference| < 1e-9) |
| forecast rv_hat identical | 100%; matched signal = RV–IV signal to 8.7e-12, same sign on every day it exists |
| matched signal exists | 803 of 866 days — the first 63 days sit flat (q = 0) in the intraday portfolios |
| sign(s) on identical days (803) | intraday 15:30 leg 1.891 = RV–IV 1.891 |
| headline numbers | intraday 15:30 leg 1.820 (866 days, warm-up flat) vs RV–IV 1.631 (871 days) / 1.557 on the 866 common days |

The 15:30 leg *is* the RV–IV portfolio. Its higher printed Sharpe (1.82 vs 1.63) is the 63-day warm-up sitting flat,
which removes the COVID quarter — the same effect the RV–IV audit reported (warm-up-matched 1.90).

## 2. The projection and what actually happens

- Naive projection: twelve legs a day, each like the 15:30 leg → √12 × 1.82 = **6.3** (or √12 × 1.63 = 5.6).
- Actual daily-sum Sharpe (the notebook's convention): sign(s) **1.92** (t +3.5); always short 1.90 (t +3.8).
- The 15:30 leg alone: **1.82**. The eleven legs 10:00–15:00 together: sign(s) **0.75** (t +1.2) — they add +0.10 to the daily Sharpe. Always short on the same eleven legs: **3.43**.
- The independence assumption was fine: mean cross-bar correlation of sign(s) returns within a day +0.001 (max +0.08); var(daily sum)/Σvar = 1.02.
- What fails is the per-leg edge: per-bar Sharpe (mean/std, unannualized) 0.115 at 15:30 vs 0.00–0.07 on the other legs. Per-leg annualized Sharpe, sign(s): 0.14, 0.10, −0.03, 1.05, 0.86, −0.01, 0.82, 0.66, −0.28, 0.21, 0.08, then 1.82 at 15:30; only 11:30 has |t| > 1.9.
- Concentration: the 15:30 leg carries 80% of sign(s) P&L and 72% of daily variance (premium 5.4 pts at 15:30 vs 19.8 at 10:00; per-premium std 1.07 vs 0.15). Effective independent legs (Σσ)²/var(sum) = 5.8 of 12 — so even eleven *good* legs could not deliver √12.

## 3. Mechanisms, ranked

**1. The intraday implied "slice" is mis-specified — this is the gap.** The forecast is as good intraday as at the close: corr(log forecast, log realized) 0.75–0.90 at every clock (0.88 at 15:30). What the sign rule compares it with intraday is not a market price but a construction, IV²·h·w, the vendor's remaining implied variance allocated to the next bar by the *realized* profile share w. That allocation is wrong in level: mean realized/slice is 1.02–1.11 at most intraday clocks (0.85–0.94 at 11:30, 12:30, 13:30) but **0.795 at 15:30**, where the slice is the market's own price for the last half hour. The slice therefore tells the rule the intraday bars are fair or cheap while the market's marks say they are rich (always short earns Sharpe 0.2–2.5 at every intraday clock, 3.43 pooled). The consequences on the same rows:
- the rule buys 50–67% of intraday bars (37% at 15:30; an oracle that knows the bar's realized variance buys 28–50%), paying the bar-by-bar decay that always short collects;
- hit rate 0.44–0.52 intraday vs 0.56 at 15:30; mean-return spread buy-minus-sell ≤ 0.02 intraday vs 0.27 at 15:30;
- the ceiling is not the problem: a peeking oracle sign(realized − slice) earns Sharpe 4.0–4.7 at *every* clock (daily sum 10.4; 13.2 on the eleven intraday legs alone) — the marks do pay for knowing the bar's variance;
- decisive test: replacing w by the *same day's* realized profile share (a better allocation by construction, though it peeks at the day's shape) makes sign(s) **worse**, −0.6 to −2.0 at every intraday clock and unchanged 1.89 at 15:30. Any realized-profile slice flags "buy" exactly when the bar's share is small. The forecast-versus-profile-slice comparison is anti-informative intraday by construction; the rule works at 15:30 only because w ≡ 1 there and the slice is a real price.

**2. Mark-to-market noise (vega) dilutes but does not cap.** On the 30-minute mark, the bar's realized variance explains 14–33% of the return's variance and the change in implied volatility a further 20–30% (R² 0.36–0.59 with both; t on the vega term 15–24 vs 8–15 on realized). A hold-to-close exit does not rescue sign(s) (0.19–1.29 by clock) and turns always short negative from morning entries. This is why per-leg Sharpes are lower than the settlement leg's even for the oracle relative to its hit rate — but the oracle's 4+ per clock shows the mark is tradable.

**3. Variance concentration.** Premium falls 19.8 → 5.4 points across the day; the settlement leg holds 72% of daily variance; effective independent legs 5.8 of 12. This bounds what any fix can add: the oracle's daily sum is 10.4, not 12 × anything.

**4. Not mechanisms.** Reconciliation (section 1) is exact. Alignment: the bar's own realized variance is the lag-0 maximum of the |R| correlations (lag −1: −0.03, 0: +0.02, +1: 0.00) — weak because of vega, not misaligned. The BSM check passes at every clock: pricing the package over the remaining window at the vendor hourly IV reproduces the quoted mid to 1.0000 (5th–95th pct 0.9996–1.0007), so the units are right and the failure is the *allocation* of that implied variance across bars, not its level. w's own noise is small (coefficient of variation 0.01–0.10 by clock).

## 4. Verdict

The projection assumes the intraday legs are twelve copies of the 15:30 trade. They are not: the 15:30 leg supplies 1.82 of the 1.92, the eleven intraday legs supply +0.10, and the whole shortfall against 5.6–6.3 sits in one place — the intraday sign is computed against a realized-profile slice of the implied variance that is fair-to-cheap intraday and rich only at the close, so the rule buys more than half of the intraday bars and pays the premium that always short harvests (3.43 on those legs). The forecast itself is not at fault (0.85 correlation with its target at every clock), the exit convention is not at fault (an oracle earns 4+ per clock on the same marks), and the join is not at fault. Concentration of variance in the settlement leg (72%) then caps any repair at roughly half the √12 the projection imagined. Decomposition: 6.3 (projection) → 1.82 (settlement leg, as the RV–IV portfolio) + 0.10 (achieved by the intraday legs) = 1.92; the oracle bound on the intraday legs, 10.4 − 1.82, is the room a correctly specified intraday implied term would compete for.
