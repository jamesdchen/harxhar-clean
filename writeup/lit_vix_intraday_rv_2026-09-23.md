# Implied volatility as an exogenous predictor of intraday realized variance — literature check (2026-09-23)

Question: have others established that the VIX (and VVIX / VIX3M / VIX9D) or option-implied variance is a useful
exogenous predictor of INTRADAY realized variance of the S&P 500 in HAR/GARCH-type models, beyond the RV lags?
Our setting: 30-minute S&P 500 RV one bar ahead (and to the close from 15:30), a HAR ladder of RV lags plus ES
return moments, Cboe VIX / VVIX / VIX3M prints at the 30-minute stamp as exogenous. We find the VIX family is the
only exogenous bucket that improves QLIKE at the close, and log VIX at 15:30 has correlation 0.81 with log RV of the
15:30–16:00 bar.

Run by a subagent (12 web searches, 9 open-PDF reads). "V" = title/abstract verified from a search result or fetched
PDF; "R" = recalled detail, not verified. Paywalled pages (Wiley, ScienceDirect, SSRN) returned 403 — marked.

## A. Daily-horizon HAR / GARCH + implied vol (the canonical strand)

| Paper | Year | Venue | Market / freq | Horizon | How IV enters | Headline | Closeness to us | Verified |
|---|---|---|---|---|---|---|---|---|
| Blair, Poon & Taylor, "Forecasting S&P 100 volatility: the incremental information content of implied volatilities and high-frequency index returns" | 2001 | J. Econometrics 105(1) | S&P 100; daily, VXO, 5-min RV | 1–20 d | VIX as exogenous term in ARCH variance | in-sample "nearly all relevant information is provided by the VIX"; OOS VIX most accurate at all horizons | daily only; same direction as our bucket result | V |
| Poon & Granger, "Forecasting volatility in financial markets: a review" | 2003 | J. Econ. Literature 41(2) | survey (93 papers) | daily–monthly | survey | IV forecasts generally beat historical-vol models | background | V |
| Koopman, Jungbacker & Hol, "Forecasting daily variability of the S&P 100 … historical, realised and implied volatility" | 2005 | J. Empirical Finance 12(3) | S&P 100; 5-min RV, VXO | 1 d | IV beside RV (ARFIMA/UC) | RV-based models dominate; IV adds but is not sufficient (ranking R) | "RV lags + IV" precedent | V (ranking R) |
| Jiang & Tian, "The model-free implied volatility and its information content" | 2005 | RFS 18(4) | S&P 500 options | ~1 month | MFIV strip in encompassing regressions | MFIV subsumes BS-IV and past RV | monthly strip; our 0DTE strip is 9× the slice (zero-bid strikes) — theirs is not | V |
| Becker, Clements & White, "Does implied volatility provide any information beyond that captured in model-based volatility forecasts?" | 2007 | J. Banking & Finance 31(8) | S&P 500 daily; VIX vs model set | 22 d (R) | VIX vs combination of GARCH/SV/RV forecasts | VIX adds nothing beyond the model set | the dissent | V |
| Busch, Christensen & Nielsen, "The role of implied volatility in forecasting future realized volatility and jumps in FX, stock, and bond markets" | 2011 | J. Econometrics 160(1) | $/DM, S&P 500, T-bond; HF RV | monthly | HAR-RV-IV; continuous + jump split; VecHAR | IV strong predictor of RV and jumps beyond HAR lags | the canonical HAR-RV-IV paper | V |
| Kambouroudis, McMillan & Tsakou, "Forecasting stock return volatility: a comparison of GARCH, implied volatility, and realized volatility models" | 2016 | J. Futures Markets 36(12) | international indices, daily | 1 d+ | IV-index forecasts vs GARCH vs RV | IV-augmented models tend to win (R) | daily | V (result R) |
| Kambouroudis, McMillan & Tsakou, "Forecasting realized volatility: the role of implied volatility, leverage effect, overnight returns and volatility of realized volatility" | 2021 | J. Futures Markets 41 | 10 indices; daily RV | 1 d+ | HAR + local IV index + leverage + overnight + vol-of-RV | the IV-augmented model "provides a significantly better forecast than more sophisticated models … that exclude the information backed out from option prices" | closest daily analog to "VIX is the only bucket that helps" | V (WP read) |
| Clements, Liao & Tang, "Moving beyond VIX: HARnessing the term structure of implied volatility" | 2022 | J. Forecasting 41(1) | S&P 500 (SPY 5-min), 2000–2019 | d / w / m | log-RV HAR + IV term-structure level / slope / curvature as exogenous | slope and curvature beat HAR and HAR+VIX at all horizons | the direct precedent for a VIX9D/VIX/VIX3M bucket; daily; OptionMetrics IVs, not Cboe prints | V (PDF read) |
| Dutta (+co-author), "Forecasting realized volatility: new evidence from time-varying jumps in VIX" | 2022 | J. Futures Markets | SPX 5-min RV; VIX sole exogenous | daily | HAR-RV + VIX level + VIX jumps | VIX jumps improve HAR-RV forecasts | "VIX as sole exogenous" mirrors ours; daily | V (snippet) |

## B. Intraday-horizon models — do they use implied vol?

| Paper | Year | Venue | Market / freq | Horizon | How IV enters | Headline | Closeness | Verified |
|---|---|---|---|---|---|---|---|---|
| Andersen & Bollerslev, "Intraday periodicity and volatility persistence in financial markets" | 1997 | J. Empirical Finance 4 | DM/$, S&P 500 futures 5-min | intraday | none | periodicity distorts HF dynamics; FFF filter | the diurnal pattern our per-clock work addresses | V |
| Engle & Sokalska, "Forecasting intraday volatility in the US equity market. Multiplicative component GARCH" | 2012 | J. Financial Econometrics 10(1) | ~2,700 US stocks, 10-min | next bars | none (daily component from an external daily forecast, R) | daily × diurnal × intraday GARCH | intraday benchmark WITHOUT IV — defines the gap | V ("no IV" R) |
| Stroud & Johannes, "Bayesian modeling and forecasting of 24-hour high-frequency volatility" | 2014 | arXiv 1211.2961 / JFEc (R) | S&P 500 futures, 5-min, 24 h | intraday–daily | none | multi-factor SV + jumps + seasonals + announcements; RV forecasts up to 50% better | ES intraday, no IV | V |
| Lyócsa, Molnár & Výrost, "Stock market volatility forecasting: do we need high-frequency data?" | 2021 | IJF 37(3) | 18 indices; OHLC vs HF | 1 d–1 m | none (range estimators) | HF beats low-frequency only at short horizons | tangential | V |
| Degiannakis et al. (ARFIMA intraday 2008; "Forecasting VIX"; "Trading VIX on volatility forecasts") | 2008–2025 | MPRA; J. Forecasting | intraday RV → VIX | 1–10 d | VIX is the TARGET | HF-based forecasts of VIX | reverse direction | V (titles) |
| Ougou, "IHAV: Intraday Heterogeneous Autoregression with Anchored Volatility — minute-level RV forecasting for the S&P 500" | 2026 | SSRN 6752822 (unrefereed; 403) | SPX minute RV | 5–60 min | HAR features + 0DTE implied vol + time-of-day dummies | OOS R² 0.815 at h = 30 min vs 0.792 pure HAR | THE closest precedent: intraday HAR + 0DTE IV for SPX at 30 min; R² not QLIKE; no close-specific result | V (snippet only) |
| "Forecasting the realized variance in the presence of intraday periodicity" (authors R) | 2024 | J. Banking & Finance | intraday RV | intraday/daily | unknown (403) | periodicity-aware RV forecasting | possibly relevant to the per-clock design | title only |
| "Modeling and forecasting intraday spot volatility" | 2025 | IJF | intraday spot vol | intraday | unknown (403) | — | — | title only |

## C. 0DTE variance risk premium / IV vs realized, intraday

| Paper | Year | Venue | Market / freq | Horizon | How IV enters | Headline | Closeness | Verified |
|---|---|---|---|---|---|---|---|---|
| Vilkov, "0DTE Trading Rules: Tail Risk, Implementation, and Tactical Timing" | 2026 draft | SSRN 4641356; GitHub vilkovgr/0dte-strategies | SPXW 0DTE, Cboe 30-min bars, 2016–2026 | each 30-min bar → 16:00 | VIX methodology at each 30-min bar to 16:00; RV from 1-min to close; VRP by bar; ATM straddle hold-to-expiry by entry clock | positive, significant 0DTE VRP; "economic magnitude, however, is small"; PnL "dominated by tail risk rather than by stable mean carry" | same instrument and geometry; IV as a premium measure, not a HAR regressor; consistent with "sign(s) is a tail trade" | V (annotated paper) |
| Almeida, Freire & Hizmeri, "0DTE Asset Pricing" | 2025 draft | WP (FMA Derivatives 2025) | SPX 0DTE + HF returns | intraday → expiry | intraday pricing kernel; VRP from 0DTE prices | high VRP driven by upside-risk compensation; mispricing dissipates after daily 0DTE listing (2022) | intraday risk premia, not RV forecasting | V (PDF) |
| Dim, Eraker & Vilkov, "0DTEs: Trading, Gamma Risk and Volatility Propagation" | 2024 | WP (WFA) | SPX 0DTE intraday | intraday | gamma exposure | 0DTE gamma does not propagate volatility | peripheral | V |
| Amaya, Garcia-Ares, Pearson & Vasquez, "0DTE Index Options and Market Volatility: How Large is Their Impact?" | 2025 | Cboe-hosted WP | SPX/SPXW trades, 30-min windows | 30 min | OMM gamma vs index vol | max impact ≈ +6.4 pp annualized 30-min vol (snippet) | 30-min SPX vol is the unit others use | V |
| Bollerslev, Tauchen & Zhou, "Expected stock returns and variance risk premia" | 2009 | RFS | S&P 500 monthly | monthly | VIX² − RV predicts returns | VRP predicts returns | foundational VRP definition | V |

## Synthesis

1. Established at the DAILY horizon: implied vol carries strong incremental information over RV lags in HAR models for
   S&P indices — Busch et al. 2011, Kambouroudis et al. 2021, Clements et al. 2022, with Blair et al. 2001 as the
   ARCH-era classic. Dissent: Becker et al. 2007 (nothing beyond a rich model set at 22 days). Clements et al. 2022 is
   the direct precedent for a VIX-family bucket (term structure beyond VIX), though with OptionMetrics interpolated
   IVs rather than the Cboe prints.
2. NOT established: the intraday-horizon literature (Andersen–Bollerslev 1997, Engle–Sokalska 2012, Stroud–Johannes
   2014) models diurnal periodicity and forecasts 5–10-min or intraday RV with no option-implied regressor at all.
   No refereed paper found puts VIX / VVIX / VIX3M at a 30-minute stamp into an intraday HAR, and none reports QLIKE
   for the 15:30–16:00 bar or the close.
3. The only located intraday HAR + implied-vol precedent is Ougou (SSRN 2026, unrefereed, R² metric) — cite as a
   contemporaneous working paper, not as established. Nobody found reports a log-VIX-to-log-RV correlation for the
   last half hour; our 0.81 at 15:30 appears to be new.
4. The 0DTE VRP strand (Vilkov 2026; Almeida–Freire–Hizmeri 2025) confirms implied variance from 30-min bars exceeds
   realized-to-close with time-of-day structure, but treats it as a premium, not a forecast input; Vilkov's small,
   tail-dominated unconditional carry is consistent with our "sign(s) is a tail trade".
5. Degiannakis and co-authors run the opposite direction (HF data → VIX): adjacent, not a precedent.
6. Cite first: Busch, Christensen & Nielsen 2011 (HAR-RV-IV); Clements, Liao & Tang 2022 (VIX family / term structure
   as exogenous on the S&P 500); Engle & Sokalska 2012 (the intraday benchmark that omits IV — the gap). Then Blair–
   Poon–Taylor 2001 and Becker–Clements–White 2007 as the pro/con pair, and Vilkov 2026 for the 30-min-to-close VRP
   geometry.

Sources: Blair et al. https://econpapers.repec.org/RePEc:eee:econom:v:105:y:2001:i:1:p:5-26 ; Busch et al.
https://econpapers.repec.org/article/eeeeconom/v_3a160_3ay_3a2011_3ai_3a1_3ap_3a48-57.htm ; Becker et al.
https://econpapers.repec.org/article/eeejbfina/v_3a31_3ay_3a2007_3ai_3a8_3ap_3a2535-2549.htm ; Kambouroudis 2016
https://onlinelibrary.wiley.com/doi/abs/10.1002/fut.21783 ; Kambouroudis 2021 WP
https://rahwebdav.swan.ac.uk/repec/pdf/WP2019-03.pdf ; Clements–Liao–Tang AAM
https://research-management.mq.edu.au/ws/portalfiles/portal/182398117/168458596AAM.pdf ; Koopman et al.
https://ideas.repec.org/a/eee/empfin/v12y2005i3p445-475.html ; Jiang–Tian
https://academic.oup.com/rfs/article-abstract/18/4/1305/1595745 ; Poon–Granger
https://ideas.repec.org/a/aea/jeclit/v41y2003i2p478-539.html ; Engle–Sokalska
https://academic.oup.com/jfec/article-abstract/10/1/54/755620 ; Stroud–Johannes https://arxiv.org/abs/1211.2961 ;
Lyócsa et al. https://www.sciencedirect.com/science/article/abs/pii/S0169207020301874 ; Andersen–Bollerslev
https://econpapers.repec.org/RePEc:eee:empfin:v:4:y:1997:i:2-3:p:115-158 ; Ougou IHAV
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6752822 ; Dutta 2022
https://onlinelibrary.wiley.com/doi/full/10.1002/fut.22372 ; Vilkov https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4641356
and https://github.com/vilkovgr/0dte-strategies/blob/main/docs/paper/paper-annotated.md ; Almeida–Freire–Hizmeri
https://www.fma.org/assets/docs/Derivatives2025/Almeida.pdf ; Dim–Eraker–Vilkov
https://westernfinance-portal.org/viewpaper?n=950096 ; Amaya et al.
https://cdn.cboe.com/resources/education/research_publications/gammasqueezes.pdf ; Degiannakis MPRA
https://mpra.ub.uni-muenchen.de/96307/8/MPRA_paper_96307.pdf
