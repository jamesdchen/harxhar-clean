# Mechanical Dealer-Hedging Channel: Literature Sweep

**Scope.** End-of-day / intraday dealer gamma rebalancing and its effect on realized volatility — especially the last 30–60 minutes and 0DTE options. Papers that (1) measure or model dealer rebalancing flow as a driver of intraday realized vol or variance risk premium, (2) test whether gamma-driven hedging creates predictable mechanical price pressure near expiry, (3) decompose VRP or intraday returns into a mechanical-hedging component vs. information/risk-premium component, and (4) ask whether the channel is already priced in or crowded by 0DTE traders. Sources: Scholar, SSRN, arXiv, journal and exchange sites; emphasis on Dim–Eraker–Vilkov and 2023–2026 follow-ups, plus the Ni / Barbon / Gayda measurement line.

**Compiled:** 2026-09-28 (voice lit sweep).

---

## Headline finding

The mechanical dealer-hedging channel is **real** in the literature. It shows up most clearly for **last-thirty-minute momentum** and for **stock-level gamma imbalances**.

For **SPX 0DTE** specifically, the weight of recent work says market-maker gamma is usually **positive** and **dampens** intraday volatility — not the opposite. **Brogaard–Han–Won** is the main dissenting **volume-based** result (0DTE volume share raises vol). One **Princeton senior thesis (Shi 2026)** argues the canonical gamma-feedback channel is **too weak** to explain intraday variance even though 0DTE activity correlates with vol.

Almost nobody cleanly splits equity **VRP** (or intraday returns) into a mechanical-hedge-flow component versus an information / risk-premium component. The closest **flow-premium / limits-to-arbitrage** frames are:

- **Gârleanu–Pedersen–Poteshman** (demand-based option pricing)
- **Terstegge** (shadow gamma; overnight inventory / gap risk)
- **Almeida–Freire–Hizmeri** (fading 0DTE mispricing / arb profits as the market integrated)

---

## Annotated list

Organized in the chunks returned by the search.

### Chunk 1 — Canonical mechanical-hedging papers (incl. last half-hour)

1. **Ni, Pearson, Poteshman, White (2021).** *Does Option Trading Have a Pervasive Impact on Underlying Stock Prices?* **Review of Financial Studies.**  
   Stock-level dealer gamma from classified OI; hedge rebalancing explains a meaningful share of daily absolute-return variation — the template for later mechanical-channel tests.  
   Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2867461

2. **Baltussen, Da, Lammers, Martens (2021).** *Hedging Demand and Market Intraday Momentum.* **Journal of Financial Economics.**  
   Last-thirty-minute futures return predicted by earlier-day return across asset classes; explicitly linked to gamma hedging by option market makers and leveraged ETFs. Core last-hour paper.  
   Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3760365

3. **Barbon and Buraschi.** *Gamma Fragility.* Working paper / SSRN.  
   Aggregate dealer gamma imbalance → intraday momentum when short gamma, reversal when long; stronger in illiquid names; more flash-crash-like events when gamma very negative. Frames returns as non-informational price pressure / limits to liquidity provision.  
   Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3725454  
   PDF (author site): https://www.abarbon.com/assets/Barbon_Buraschi_2021_Gamma_Fragility.pdf

4. **Barbon.** *Liquidity Provision to Rebalancing Flows from Leveraged ETFs and Equity Options.*  
   Compares LETF end-of-day rebalancing vs option delta-hedging; option hedging responds faster to intraday jumps, LETF almost purely end-of-day; both create temporary EOD pressure that reverses — classic mechanical flow.  
   PDF: https://www.abarbon.com/assets/Liquidity_Provision_to_Rebalancing_Flows_from_Leveraged_ETFs_and_Equity_Options.pdf

5. **Gayda, Gruenthaler, Harren (2022).** *Option Liquidity and Gamma Imbalances.* SSRN.  
   AGI = \(S^2 \sum\) MM residual OI × γ; negative AGI widens option spreads; when AGI is forced near zero in stress, option expensiveness and VRP rise — inventory / limits-to-arb pricing of gamma risk.  
   Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4138512

6. **O’Donovan, Yu, Zhang.** *Option Market Maker Hedging and Stock Market Liquidity.* SSRN.  
   Open–Close NetΓ from customer residual predicts stock liquidity; hedging demand as mechanical liquidity consumer.  
   Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4567604

7. **Egebjerg and Kokholm.** Working paper, SSRN 4936978.  
   SPX OMMs’ required delta hedge in the last thirty minutes predicts EOD SPX futures returns; hedge split into a **gamma** piece and an **inventory-change** piece — both matter, and focusing only on gamma understates impact when they reinforce.  
   Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4936978

8. **Anderegg, Ulmann, Sornette (2022).** *The impact of option hedging on the spot market volatility.* **Journal of International Money and Finance.**  
   Model and evidence that delta-hedging frictions raise spot vol.  
   Link: https://doi.org/10.1016/j.jimonfin.2022.102635

### Chunk 2 — 0DTE and near-expiry (2023–2026)

9. **Dim, Eraker, Vilkov (2023/rev. 2025).** *0DTEs: Trading, Gamma Risk and Volatility Propagation.* SSRN.  
   MM inventory gamma on average positive and negatively related to future intraday SPX vol; positive gamma strengthens reversal, negative strengthens momentum; consistent with delta-hedging, not information. Caveat: sample when MMs were mostly long gamma.  
   Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4692190  
   WFA PDF host: https://westernfinance-portal.org/viewpaper?n=950096

10. **Adams, Fontaine, Ornthanalai (2024).** *The Market for 0DTE: The Role of Liquidity Providers in Volatility Attenuation.* SSRN.  
    Exploiting expiration-day variation, index vol falls about 60–90 annualized basis points on 0DTE days; attenuation comes mainly from longer-dated positions rolling into 0DTE, not same-day flow alone.  
    Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4881008

11. **Adams, Dim, Eraker, Fontaine, Ornthanalai, Vilkov.** *Do S&P500 Options Increase Market Volatility? Evidence from 0DTEs.* SSRN 5641974.  
    Merged follow-up subsuming Dim–Eraker–Vilkov; hedging needs predict stronger order-flow reversals, lower momentum, lower vol. (Subsumes paper 9 per Dim research site; RFS track.)  
    Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5641974

12. **Brogaard, Han, Won (2023/rev. 2024).** *Does 0DTE Options Trading Increase Volatility?* SSRN.  
    Opposite side: instrumented 0DTE volume share raises ETF realized vol substantially. Volume-based, not gamma-signed — the main dissent.  
    Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4426358

13. **Pearson, Poteshman, White et al. (Cboe research).** *0DTE Index Options and Market Volatility: How Large Is Their Impact?*  
    Proprietary OMM gamma at one-minute frequency; typical impact dampens vol; max impact when gamma negative roughly 3.3 annualized points daily, 6.4 over thirty minutes — economically modest vs normal vol moves.  
    PDF: https://cdn.cboe.com/resources/education/research_publications/gammasqueezes.pdf  
    Related insight note: https://www.cboe.com/insights/posts/volatility-insights-evaluating-the-market-impact-of-spx-0-dte-options

14. **Shi, Junhe (2026).** *0DTE Options and Intraday Volatility.* Princeton ORFE senior thesis.  
    Structural variance map split into exogenous flow vs endogenous hedge flow; canonical gamma-feedback adds little QLIKE or correlation predictive power for intraday variance — so 0DTE relates to vol, but not primarily via the textbook hedge channel.  
    Record: https://theses-dissertations.princeton.edu/entities/publication/61115cd1-cb09-4a8b-9230-ff488fc2a386

15. **Almeida, Freire, Hizmeri.** *0DTE Asset Pricing.* SSRN.  
    High compensation for upside variance, U-shaped pricing kernel; many 0DTEs violate risk-averse SD bounds; delta-hedged ATM arb was very profitable then faded as daily 0DTEs integrated — closest “crowded / priced-in” paper for the 0DTE complex (about option mispricing more than whether the underlying hedge-flow premium got crowded).  
    Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4701401

### Chunk 3 — Flow premium, limits-to-arb frames, and gaps

16. **Gârleanu, Pedersen, Poteshman (2009).** *Demand-Based Option Pricing.* **Review of Financial Studies.**  
    End-user demand pressure with intermediary risk aversion; the classic limits-to-arbitrage / flow-premium setup for options. Not about intraday realized vol per se, but the conceptual parent of “returns as compensation for warehousing unbalanced gamma.”  
    Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=963442

17. **Terstegge.** Shadow-gamma / overnight option risk-premium paper (FMA Derivatives 2025 materials).  
    Dealers’ short put inventory creates gap risk overnight when hedges can’t be adjusted; local gamma misses it; shadow gamma predicts overnight delta-hedged option returns. Explicit inventory / limits-to-arb pricing of unhedgeable moves — VRP concentrated overnight.  
    PDF: https://www.fma.org/assets/docs/Derivatives2025/Terstegge.pdf

18. **Ni-line / Gayda inventory pricing.** When AGI is forced toward zero in stress, VRP and liquidity risk premia rise — dealers charge more to avoid mechanical rebalancing: a priced inventory channel rather than a clean decomposition of equity VRP into hedge-flow versus information.

19. **Near-expiry pinning (cousin channel).** Avellaneda–Lipkin; **Golez and Jackwerth (2012),** *Pinning in the S&P 500 futures,* **Journal of Financial Economics.** Mechanical gravitating toward high-OI strikes into expiry — pin-risk cousin of gamma rebalancing; more about strike clustering than last-hour vol amplification.  
    Link: https://doi.org/10.1016/j.jfineco.2012.06.010

### What the sweep did **not** find cleanly

- No peer-reviewed paper that formally decomposes equity VRP or intraday returns into a mechanical-hedging component versus an information/risk-premium component with a tight identification design.
- Closest substitutes: Baltussen last-thirty-minute momentum as temporary price pressure; Barbon LETF-vs-options reversals; Dim/Adams “consistent with delta-hedging not information”; Terstegge / Gârleanu for option-side premia.
- No strong academic paper titled as “0DTE hedge channel is crowded”; Almeida’s fading arb profits is the best proxy.

### Industry / exchange notes (not journals)

- Cboe Volatility Insights, *Much Ado About 0DTEs* — directionally aligned with Dim / Adams / Cboe OMM gamma.
- SpotGamma / Quant Memo explainers — same directional long-vs-short gamma story for intraday damping/amplification.

---

## Quick reference: identification styles

| Paper | Measure of hedging | Horizon | Sign / effect on vol or returns |
|-------|--------------------|---------|----------------------------------|
| Ni et al. 2021 | Classified OI × Γ | Daily | Short gamma → more |return| variation |
| Baltussen et al. 2021 | Implied hedge demand | Last 30 min | Short gamma / hedge demand → EOD momentum |
| Barbon–Buraschi | Call−put Γ / flow INV | Intraday | Neg Γ → momentum & fragility |
| Gayda et al. | AGI = S² Σ netOI_MM Γ | Daily | Neg AGI → wider option spreads |
| Egebjerg–Kokholm | Δ-hedge = γ + inventory | Last 30 min SPX | Hedge predicts EOD futures return |
| Dim–Eraker–Vilkov | MM inventory gamma | Intraday SPX | Avg +Γ dampens vol / aids reversal |
| Adams et al. | Hedging needs + 0DTE day IV | Intraday / day | 0DTE presence attenuates vol |
| Brogaard–Han–Won | 0DTE volume share | Daily | More 0DTE vol share → higher vol |
| Cboe OMM gamma | Proprietary OMM Γ | 1-min / 30-min | Typical dampen; modest max amplify |
| Shi 2026 | Structural hedge-flow var | Intraday | Canonical channel weak predictively |

---

## Addendum: GEX comparison on this repo (harxhar-clean / grok/0dte-professor-notes)

Read-only checks against `results/gex/gex_daily.parquet` (built by `experiments/gex.py` from OptionMetrics IvyDB EOD SPX) and RV–IV notebook buy days in `results/atm_straddle_0dte_1530/daily_blk2.parquet` (`pos > 0`, 346 of 866 days, 2020-01-03 → 2024-04-30).

| Finding | Detail |
|---------|--------|
| Buy days vs lag-1 net-negative dealer GEX | **Not enriched.** P(neg \| buy) = **40.17%** vs base P(neg) = **45.15%**; enrichment **0.89**. Contingency: buy∧neg 139, buy∧posΓ 207, sell∧neg 252, sell∧posΓ 268. |
| Same-day (unshifted) check | Nearly identical: P(neg \| buy) 39.88%, base 44.57%, enrichment 0.895 — consistent with sticky regimes. |
| Gamma measure | OptionMetrics IvyDB EOD SPX dealer GEX: `gex_level = Σ s_i · γ_i · OI_i · 100 · S² · 0.01`. IvyDB `gamma` when present, else BS from IV. Caller lags one trading session (`gex_regime_test.lag_regime` / `options_features` go-live). |
| Regime persistence | Sticky: autocorr(`gex_level`) lag1…5 ≈ 0.89, 0.79, 0.72, 0.65, 0.60. `regime_long_gamma` stays the same next session **~86.6%** of the time (flip ~13.4%); median run length 3 sessions, mean ~7.4. |
| Intraday / OPRA GEX | **No OPRA tick GEX series** in this repo. Only daily `gex_daily.parquet`. `gex_0dte` is still EOD. `gex_bar_test` broadcasts lagged daily regime to 30-min bars; `structure_gex_live` re-evaluates the **prior** EOD chain at live spot — not a live OPRA gamma feed. |

Lag used for the overlap was full-calendar `gex_level.shift(1)` then reindexed to the buy frame (agrees with last GEX session strictly before the trade date and with `options_features` go-live). Do **not** `shift(1)` on the sparse 866-day expiration-only index alone — that mis-lags on ~223 days that skip intervening GEX sessions.
