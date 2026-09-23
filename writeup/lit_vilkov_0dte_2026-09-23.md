# Vilkov, "0DTE Trading Rules: Tail Risk, Implementation, and Tactical Timing" (SSRN 4641356) — read (2026-09-23)

Read by a subagent from the public package (github.com/vilkovgr/0dte-strategies: docs/paper/paper-annotated.md, March
2026 text; code/build_data.py; code/analysis/*.py; output/tables/*.tex; KNOWN-ISSUES.md; README.md). The SSRN page
returned 403; the LFS data parquets are pointers, so nothing was recomputed. "verified" = read in the text or code;
"inferred" = read off a figure or deduced.

**Read first — the package supersedes the paper text.** KNOWN-ISSUES.md (Aug 2026): the bid-ask half-spread was
charged at 1/100 of its true size in every net-of-cost table and in the binary target of the conditional models.
Code and output/tables are refreshed; the paper PDF and the Table 9 baskets are stale. README: "no strategy or
basket retains a positive net Sharpe ratio." The paper's headline conditional Sharpes (1.18 / 0.93, 1.12 / 0.82,
strangle 0.56 / 0.39) are pre-fix and must not be quoted.

## 1. Data and construction

| Item | Finding | Status |
|---|---|---|
| Options | Cboe 30-min option bars (NBBO, sizes, OHLC, volume, underlying), root SPXW, proprietary | verified |
| Underlying | ThetaData 1-min SPX and VIX bars | verified |
| Sample | 09/2016–01/2026 (OOS to 2026-02-02); Tue/Thu expiries added 2022-04/05 | verified |
| Entry clocks | strategies at 10:00 (main), 13:00, 15:00, 16:00-previous-day (appendix); IV/RV at every 30-min bar-end 10:00..15:30 | verified |
| Moneyness window | keep K/S in [0.98, 1.02]; Akima interpolation to a 0.001 grid per (date, time, type) | verified |
| Zero-bid / min-bid filter | **none** in paper or code; only the ±2% window (bid = 0 -> mid = ask/2 still enters); negative time value clamped to 0 | verified absent in code; inferred absent upstream |
| IV to expiry | VIX formula 2 Σ ΔK/K² Q(K) − (F/K0 − 1)² on the interpolated OTM mids; code: r = 0, F from the strike minimising \|C − P\|, ΔK = 0.001·S; semivariances IV_up / IV_dn, IS = IV_up − IV_dn | verified |
| RV to close | Σ of 1-min close-to-close log returns² from the bar to 16:00; whether the 16:00 settlement print is the last term is unstated | formula verified |
| Units | not annualised; same-day payoff units; PnL in % of spot | verified |
| Straddle | synthetic exactly-ATM (both legs at interpolated moneyness 1.000), not listed nearest-OTM strikes; long side reported | verified |
| Costs | entry only (cash settlement); mid; mid − Σ half-spreads; − 0.5 bp fixed; mean half-spread ≈ 2.2 bp of spot per structure; net = sign·gross − c | verified |
| Hedging | "All strategies in this paper are unhedged and held to expiration" | verified |

## 2. VRP by time of day

- Fig. vrp (§3.1): four panels by 30-min bar (variance swap rate, VRP = IV − RV, vol swap rate, vol RP), mean with
  NW(3) 95% CI and quartiles, units variance × 100, not annualised.
- Numbers in the text: only "median realized VRP from 10:00 ET to expiration ≈ 0.0011% of underlying", "positive and
  statistically significant", monetisable upper bound ≈ 0.20% of spot. **No per-bar magnitudes or t-stats anywhere;
  the shipped vrp_bytime_bars.pdf is blank.** The paper does not claim the close is richest or otherwise.
- Indirect (read off strangle_reth_und_bymnes_bars.pdf, inferred): long exactly-ATM straddle PnL in % of spot —
  10:00 mean ≈ 0.00, median −0.09, IQR [−0.29, +0.24]; 13:00 mean ≈ −0.005, median −0.073; 15:00 mean ≈ −0.007,
  median −0.045, IQR [−0.13, +0.06]: mean/IQR worsens into the afternoon.
- **Strongest clock-specific fact for us** (appendix tab:strat_ret_expl_1500, 15:00 → close, pooled strangle
  combos, combo FE, date-clustered t): long straddle/strangle PnL on IV −0.042 (t −2.68), RV +0.064 (t 2.47),
  adj. R² 0.201; with IS/RS: IV −0.038 (t −3.79), RV +0.056 (t 5.24), R² 0.313. At 10:00: IV −0.013 (t −2.27),
  RV +0.013 (t 1.36), R² 0.056. Near-equal-and-opposite coefficients: last-hour straddle PnL ≈ β (RV − IV) — the
  sign(s) construct, ex post, with RV known.

## 3. Straddle returns (long, mid, hold to 16:00, % of spot; 10:00 entry; n = 1318 days) — verified

| Config | Mean | Vol | Min | p1 | Med | p99 | Max | Skew | SR p.a. |
|---|---|---|---|---|---|---|---|---|---|
| 1/1 straddle | −0.0011 | 0.55 | −2.69 | −0.99 | −0.08 | 1.83 | 6.04 | 2.11 | −0.03 |
| 0.995/1.005 | −0.0009 | 0.47 | −2.70 | −0.83 | −0.06 | 1.76 | 6.02 | 3.07 | −0.03 |
| 0.99/1.01 | −0.0067 | 0.38 | −2.75 | −0.78 | −0.02 | 1.59 | 5.93 | 4.69 | −0.28 |

Subperiods (10:00 straddle, older 952-day panel): 2016–19 mean +0.0055, SR +0.20; **2020–21 mean −0.0305, SR
−0.78**; 2022–23 mean +0.0263, SR +0.71; 01–04/2024 −0.0379, SR −1.45. Other clocks: figures only, "main
qualitative findings unchanged"; **15:30 entry is not analysed as a trade.**

Implementable / tail (10:00, equal-weight strangle basket, 953 days, corrected costs): mean mid −0.0059, net
−0.0215; SR mid −0.27, **net −0.97**; turnover 24.1 bp; ES1% 1.61% of spot; worst day −2.67%; worst 5 days −7.77%;
max DD 25.9; loss probability of the long straddle 69.9%. "Tail dominates carry": |mean net| ≈ 0.02%/day vs ES1%
0.77–1.64% across templates.

## 4. Tactical timing — everything tested

| Rule | Result | Status |
|---|---|---|
| 10:00 0DTE-IV terciles (ex-post cutoffs), long PnL by regime | straddle H−L −0.0142, Welch t −0.44; only the put ratio spread is significant (H−L +0.0777, t 2.50) | verified |
| Logistic sign(p̂ − 0.5), features at 10:00 (IV, IS, moneyness slopes, lagged RV/RS/return, lagged PnL, 5-d mean/sd), expanding or rolling 252, OOS 04/2019–02/2026 (n 682) | corrected: expanding hit 68.5%, gross SR 0.33, **net −0.29**; rolling gross SR 0.69, net +0.07 | verified |
| Model zoo (ridge/enet logit, RF, LGBM, XGB, CatBoost, NN) | binary-hard best, "SR slightly above 1.0" — but with the flat 0.5 bp cost and no half-spread (upper bound) | verified |
| Baskets (top-3 by mean / SR, all) rolling 252 | corrected top-3 net −0.82; straddle sleeve long share **8.3%** (the classifier is "short the straddle" 92% of days), hit 69.7%, ES1% −268 bp, worst day −602 bp | verified; table stale |
| Realized vol / term structure / time of day as rules | not tested (RV only as a lagged feature; 0DTE only) | verified absent |
| Day-of-week, FOMC, month-end, opex / third Friday | **not tested**; "extend event conditioning" is future work | verified absent |
| Realized-vs-implied sign rule / forecast switch | not tested | verified absent |

## 5. Implementation

Sizing: one structure per unit of spot; soft map w = 2p̂ − 1; equal-weight baskets; no vol target / Kelly;
prescription: size against ES1%, worst day, drawdown. No delta hedging. Slippage: half-spread per leg + 0.5 bp of
spot, entry only; no impact model. Capacity/liquidity and margin not analysed (ES1% as capital at risk).

## 6. Relation to our results

(a) By-bar VRP shape: neither confirmed nor contradicted — the per-bar figure is blank and the text carries no
    numbers; what exists (the 15:00 regression's R² 0.20–0.31 vs 0.06 at 10:00; the afternoon mean/IQR read-off)
    points the same way as our "the close's slice is rich, the intraday slices fair-to-cheap".
(b) Sign rule: none. Nearest analogues: the ex-post regression (PnL ≈ β (RV − IV)) and a classifier that is 92%
    short the straddle — the opposite default to our long-biased tail trade.
(c) Month-end / third Friday: not examined.
(d) Strip: his "VIX methodology" runs only on interpolated mids within ±2% moneyness — no zero-bid truncation, but
    the window removes the far-OTM minimum-tick asks that make our full-chain strip 9x the slice; back-of-envelope
    inflation inside ±2% at 15:30 is a few percent. So his IV is a near-ATM truncated strip, much closer to our ATM
    slice than to our strip, and his VRP is, if anything, understated on the tails.
(e) Replicate / gate: IV_up / IV_dn semivariance split; a percent-of-spot units guard (his 100x bug is a pts-vs-%
    slip); the sign·gross − c convention (our crossed book prices the short at the bid and the long at the ask, so
    cost never adds back); his 2022-04-14 / 2022-05-11 regime split. Gate against: Akima-interpolated mids and the
    negative-time-value clamp (we trade listed strikes); F from min|C − P| with r = 0; one-sided ΔK at the window
    edges; the 16:00-labelled bar in RV.

## 7. Synthesis

1. Cite for the definition stack (VIX-formula IV to expiry at 30-min bar-ends, RV from 1-min returns to 16:00,
   VRP = IV − RV, unannualised, ±2% window) and for "positive but economically small 0DTE VRP" (median 0.0011% of
   spot from 10:00; upper bound 0.20%). Cite the 15:00 → close regression as the cleanest published support for the
   sign(s) construct at the close (PnL loads on RV − IV, R² 0.20–0.31).
2. Cite the tail message: ES1% 1.61% of spot vs |mean| 0.02%/day — our "20 of 866 days = 89% of P&L" in his units.
3. Mild contradiction: his unconditional exactly-ATM 10:00 straddle has SR −0.03 (mid), 2020–21 long SR −0.78, and
   the conditional sleeve is 92% short; he never tests 15:30 entry, so nothing meets our 15:30 result directly. Do
   not quote his 1.18 / 0.93 or 1.12 / 0.82 — retracted by the Aug-2026 correction; corrected best net SR ≈ 0.07.
4. His IV sidesteps our 9x problem by moneyness truncation, not zero-bid handling; neither of us has a per-bar VRP
   figure with zero-bid discipline — the open comparison.
5. To check in our data: (i) our 15:30 implied variance on his ±2% interpolated-mid window vs the ATM slice and the
   full strip (expect within ~5% of the slice); (ii) his Panel-C regression (PnL on IV, RV, date-clustered) at 15:30
   on the deck, for a comparable R²; (iii) ES1%, worst day, worst 5 days, max DD for sign(s) in % of spot; (iv) his
   69.9% loss probability of the long straddle as the short-hit-rate benchmark; (v) month-end / third Friday are
   ours alone.
6. Sample caveat: the main straddle table is n = 1318 (to 01/2026); the implementable, tail, regime and subperiod
   tables use the 952/953-day panel (to 2024-05-01); the OOS table is n = 682.

Sources: https://github.com/vilkovgr/0dte-strategies/blob/main/docs/paper/paper-annotated.md ;
https://github.com/vilkovgr/0dte-strategies/blob/main/code/build_data.py ;
https://github.com/vilkovgr/0dte-strategies/blob/main/KNOWN-ISSUES.md ;
https://github.com/vilkovgr/0dte-strategies/blob/main/README.md ;
https://github.com/vilkovgr/0dte-strategies/tree/main/output/tables ; https://ssrn.com/abstract=4641356 (403).
