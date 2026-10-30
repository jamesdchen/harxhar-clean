# Macro-announcement calendars and intraday RV: literature notes

Compiled 2026-10-29 for the JFEc paper (half-hour RV of S&P 500 futures; HAR / penalized linear / GBRT; FOMC calendar
features; adding six 8:30 ET release calendars: CPI, Employment Situation, PPI, GDP, trade balance, retail sales).

**Bib status** is checked against `writeup/references.bib` by author name (2026-10-29).
- **IN BIB** means the key exists.
- **NOT IN BIB** means no entry under that author.

None of the announcement-effect papers are in the bib. The only related entries are the HAR/forecasting canon (Corsi2009,
AndersenBollerslevDiebold2007, BollerslevPattonQuaedvlieg2016, AndersenBollerslev1997, christensen2023machine,
EngleSokalska2012, AndersenSuTodorovZhang2024, RahimikiaPoon2020, CaporinPoli2017, AudrinoSigristBallinari2020).

**Verification level**
- [FT]: checked against the full text (working paper or journal PDF).
- [ABS]: checked against the abstract or bibliographic record only.
- [MEM]: from memory, not re-verified here. Check before citing.

Paper context checked in the repo:
- Sample ends 30 April 2024. The MCS sets cover 2014-2024 and the R2_OOS benchmark history starts 5 January 1998.
- Lag averages run through the overnight grid, so the 8:30-9:00 and 9:00-9:30 futures bars feed the HAR averages of
  the first RTH bar.
- Scored bars end at 10:00 through 16:00 ET.
- The target is diurnally scaled. The calendar vector holds weekday dummies and the hour as an integer.
- FOMC features: a day indicator plus 1/(1+n+) and 1/(1+n-) in *intervals*.
- The March 2020 emergency action is already handled (beats/full/03_data.tex).

---

## 0. Key-paper table

| # | Paper | One-line finding | Bib |
|---|---|---|---|
| 1 | Andersen & Bollerslev (1998), JF 53(1) 219-265 | 5-min DM/$: scheduled US releases (employment report largest) are a distinct, short-lived volatility component separate from the intraday pattern; announcement effects modeled as dummies with post-release decay | NOT IN BIB |
| 2 | Ederington & Lee (1993), JF 48(4) 1161-1191 | Scheduled releases explain most time-of-day/day-of-week volatility patterns in rate and FX futures; prices adjust within ~1 min, volatility stays elevated ~15 min and modestly higher for several hours | NOT IN BIB |
| 3 | Andersen, Bollerslev, Diebold & Vega (2007), JIE 73(2) 251-277 | S&P 500 futures (GLOBEX from 8:20 ET): responses to surprises are almost entirely in the first 5 min; equity sign is state-dependent; the *mere presence* of an announcement raises volatility regardless of the surprise size (fn. 11) | NOT IN BIB |
| 4 | Lucca & Moench (2015), JF 70(1) 329-371 | Pre-FOMC drift; on FOMC days realized volatility and E-mini volume are *lower* before the statement and jump at release ("quiet before the storm") | NOT IN BIB |
| 5 | Savor & Wilson (2013), JFQA 48(2) 343-375 | Market returns and Sharpe ratios are much higher on days with scheduled CPI/PPI, employment and FOMC releases (1958-2009) | NOT IN BIB |
| 6 | Bomfim (2003), JBF 27(1) 133-151 | Stock-market conditional volatility is abnormally low the day before scheduled FOMC decisions (post-1994); target surprises raise it in the short run | NOT IN BIB |
| 7 | Jones, Lamont & Lumsdaine (1998), JFE 47(3) 315-337 | Treasury volatility is higher on employment/PPI days, with "calm before the storm", and announcement-day shocks do *not* persist | NOT IN BIB |
| 8 | Rangel (2011), JBF 35(5) 1263-1276 | S&P 500 jump intensity: of the announcement days, only employment days raise it; inflation (PPI) surprises have lasting effects, FOMC/employment surprises brief ones; news terms lower OOS RMSE on event days with no foresight of surprises | NOT IN BIB |
| 9 | Martens, van Dijk & de Pooter (2009), IJF 25(2) 282-303 | Daily RV of S&P 500 futures: adding day-of-week, holiday and macro-announcement effects (with long memory, level shifts, leverage) improves OOS forecasts up to 20 days | NOT IN BIB |
| 10 | Lahaye, Laurent & Neely (2011), JAE 26(6) 893-921 | Nonfarm payrolls and fed-funds target announcements are the most important sources of jumps and cojumps across stock-index futures, bonds and FX | NOT IN BIB |
| 11 | Gilbert, Scotti, Strasser & Vega (2017), JME 92 78-95 | Price impact differs across releases because of "intrinsic value": relation to fundamentals and timeliness; NFP ranks top | NOT IN BIB |
| 12 | Kroner (2025), FEDS 2025-022 | CPI-release reactions rose sharply in 2021-2023 (attention); reactions to other releases largely unchanged | NOT IN BIB |
| 13 | Bollerslev, Li & Xue (2018), RES 85(4) 2005-2041 | Most releases come before the open; the most important *intraday* ones are FOMC, ISM Mfg/Non-Mfg and Consumer Confidence (10:00); volume-volatility elasticity at news is below 1 | NOT IN BIB |
| 14 | Christensen, Siggaard & Veliyev (2023), JFEc 21(5) 1680-1727 | ML vs HAR for DJIA-stock RV; the predictors include a *pre-scheduled* earnings-announcement dummy, which ranks low in average importance but is "arguably very important" near the event | IN BIB (christensen2023machine) |
| 15 | Andersen, Thyrsgaard & Todorov (2019), JASA 114(528) 1695-1707 | Intraday periodicity is time-varying; FOMC days show a sharp post-statement rise; RTH releases (ISM, sentiment) affect the pattern far less | NOT IN BIB |
| 16 | Kurov, Sancetta, Strasser & Wolfe (2019), JFQA 54(1) 449-479 | E-mini and Treasury futures drift ~30 min *before* several releases (leakage or superior forecasting) | NOT IN BIB |

Secondary (cite as needed): ABDV (2003) AER; Bollerslev-Cai-Song (2000) JEF; Ederington-Lee (1996) JFQA; Savor-Wilson
(2014) JFE; Ai-Bansal (2018) Ecta; Cieslak-Morse-Vissing-Jorgensen (2019) JF; Hu-Pan-Wang-Zhu (2022) JFE; Evans (2011)
JBF; Lee (2012) RFS; Flannery-Protopapadakis (2002) RFS; Miao-Ramchander-Zumwalt (2014) JFM; Vortelinos (2015) RFE;
Vortelinos-Gkillas (2018) IJBAAF; Plíhal (2021) JoF; Lyócsa-Molnár-Plíhal (2019) JIFMIM; Wright (2025) JTSA;
Fisher-Martineau-Sheng (2022) RFS; Boguth-Grégoire-Martineau (2019) JFQA; Kurov-Wolfe-Gilbert (2021) FRL;
Gürkaynak-Kısacıkoğlu-Wright (2020) AER; Engle-Sokalska (2012) JFEc (IN BIB).

---

## 1. Scheduled US macro announcements and intraday volatility

### 1a. Foundations: FX and bond futures
- **Ederington, L.H. & Lee, J.H. (1993).** "How Markets Process Information: News Releases and Volatility." *Journal of
  Finance* 48(4), 1161-1191. doi:10.2307/2329034. https://ideas.repec.org/a/bla/jfinan/v48y1993i4p1161-91.html [ABS]
  - Covers T-bond, Eurodollar and DM futures.
  - Scheduled releases account for most of the time-of-day and day-of-week volatility pattern.
  - Prices adjust within the first minute. Volatility stays well above normal for ~15 minutes and slightly elevated
    for several hours.
  - The releases with the largest impact include the employment report, PPI, CPI, retail sales and durable goods
    [MEM on the exact ranking].
  - Relevance: this is the empirical basis for a *decaying* post-release effect, which the paper's interval-count
    proximity 1/(1+n-) approximates hyperbolically.
- **Ederington, L.H. & Lee, J.H. (1996).** "The Creation and Resolution of Market Uncertainty: The Impact of
  Information Releases on Implied Volatility." *JFQA* 31(4), 513-539. doi:10.2307/2331358.
  https://ideas.repec.org/a/cup/jfinqa/v31y1996i04p513-539_02.html [ABS]
  - Implied volatility already reflects the variance of scheduled releases and falls after them.
  - The Friday-down/Monday-up IV pattern comes from the weekday timing of releases.
  - Relevance: VIX and VIX3M in the paper's IV block partly price the scheduled event variance. A referee may ask
    whether the calendars add to IV.
- **Andersen, T.G. & Bollerslev, T. (1998).** "Deutsche Mark-Dollar Volatility: Intraday Activity Patterns,
  Macroeconomic Announcements, and Longer Run Dependencies." *Journal of Finance* 53(1), 219-265.
  doi:10.1111/0022-1082.85732. https://scholars.duke.edu/publication/761296 [ABS; model structure MEM]
  - Five-minute returns over one year.
  - Volatility splits into an intraday periodic pattern (FFF), calendar and announcement effects, and ARCH-type
    persistence.
  - Announcement effects enter as release-time dummies with a parametric decay over the following interval.
    The employment report has the largest effect.
  - The effects are large at 5 minutes but short-lived, and their significance at the daily level is "tenuous"
    (Rangel 2011's reading).
  - This is the template for "intraday volatility model + announcement dummies".
  - The repo cites only AndersenBollerslev1997 (JEF, intraday periodicity).
- **Bollerslev, T., Cai, J. & Song, F.M. (2000).** "Intraday Periodicity, Long Memory Volatility, and Macroeconomic
  Announcement Effects in the US Treasury Bond Market." *Journal of Empirical Finance* 7(1), 37-55.
  doi:10.1016/S0927-5398(00)00002-5. https://scholars.duke.edu/publication/644374 [ABS]
  - 5-min T-bond futures, 1994-1997.
  - Scheduled releases are a significant source of intraday volatility, the employment report most of all. Long
    memory remains after the periodicity and announcement adjustment.
- **Jones, C.M., Lamont, O. & Lumsdaine, R.L. (1998).** "Macroeconomic News and Bond Market Volatility." *JFE*
  47(3), 315-337. https://ideas.repec.org/a/eee/jfinec/v47y1998i3p315-337.html [ABS + Rangel's and Lucca-Moench's
  summaries]
  - Volatility is higher on employment and PPI release days, with a "calm before the storm" beforehand.
  - Announcement-day shocks are transitory and do not persist.
  - **Relevance for HAR:** a release-day RV spike enters the HAR lags and inflates the next forecasts, although the
    shock does not persist. Calendar features let the model both raise the release-day forecast and *discount* the
    lagged spike afterwards. The paper's "since last" proximity does this naturally.
  - Contrast: Li & Engle (1998) find less persistence on announcement days but not zero [MEM; via Rangel 2011].
- **Andersen, Bollerslev, Diebold & Vega (2003).** "Micro Effects of Macro Announcements: Real-Time Price Discovery
  in Foreign Exchange." *American Economic Review* 93(1), 38-62. doi:10.1257/000282803321455151.
  https://www.nber.org/system/files/working_papers/w8959/w8959.pdf [ABS]
  - Surprises cause conditional-mean jumps.
  - Bad news has larger effects than good news.
  - Volatility rises on announcement days.

### 1b. Equities: S&P 500 futures and the index
- **Andersen, Bollerslev, Diebold & Vega (2007).** "Real-Time Price Discovery in Global Stock, Bond and Foreign
  Exchange Markets." *Journal of International Economics* 73(2), 251-277. doi:10.1016/j.jinteco.2007.02.004.
  WP: https://repec.econ.au.dk/repec/creates/rp/07/rp07_20.pdf [FT, WP]
  - Data: S&P 500 futures. GLOBEX supplies 8:20-9:30 ET, because "most significant U.S. macroeconomic news
    announcements are released at 8:30 [ET] when the futures markets are open, but the equity markets closed."
  - Systematic responses are "almost exclusively restricted to the five-minute interval following the news
    release".
  - The equity response is state-dependent: good real news lowers prices in expansions and raises them in
    contractions.
  - Fn. 11: "the mere presence of an announcement, quite apart from the size of the corresponding surprise, tend[s]
    to boost volatility". Their best volatility fit used announcement *dummies*, not |surprise|.
  - **This is the most direct published support for calendar indicators over surprise measures in a volatility
    equation.**
  - Calls nonfarm payroll employment "arguably [the] most important U.S. macroeconomic announcement".
- **Flannery, M.J. & Protopapadakis, A.A. (2002).** "Macroeconomic Factors Do Influence Aggregate Stock Returns."
  *RFS* 15(3), 751-782. [MEM; summary confirmed via Rangel 2011's literature review, FT]
  - GARCH with announcement effects.
  - CPI and PPI move the *level* of returns.
  - The balance of trade, employment/unemployment and housing starts move the *conditional volatility*.
  - Relevance: support for including trade balance.
- **Bomfim, A.N. (2003).** "Pre-Announcement Effects, News Effects, and Volatility: Monetary Policy and the Stock
  Market." *JBF* 27(1), 133-151. doi:10.1016/S0378-4266(01)00211-4.
  https://ideas.repec.org/a/eee/jbfina/v27y2003i1p133-151.html [ABS]
  - Conditional volatility is abnormally low on the day before scheduled FOMC decisions. The effect is significant
    only after the 1994 disclosure change.
  - Target surprises raise volatility, positive surprises more than negative.
- **Lucca, D.O. & Moench, E. (2015).** "The Pre-FOMC Announcement Drift." *JF* 70(1), 329-371.
  doi:10.1111/jofi.12196. WP: NY Fed SR 512 [FT, WP]
  - The drift is ~49 bp in the 24 hours before scheduled statements.
  - Section 3.5: on FOMC days "prior to the announcement, realized volatility is somewhat lower ... realized
    volatility jumps at 2:15 pm". E-mini volume follows the same pattern ("quiet-before-the-storm").
  - Data: scheduled meetings only, from the Fed's calendar. Before 11 Sept 2001 the meeting dates went to the press on
    information sheets; afterwards they were posted on the website.
  - Unscheduled meetings and conference calls are excluded and identified explicitly.
  - Relevance: supports the FOMC "until" feature, which captures lower pre-statement RV. Read alongside
    Kurov-Wolfe-Gilbert (2021, FRL 40), who find the drift disappeared after 2015.
- **Savor, P. & Wilson, M. (2013).** "How Much Do Investors Care About Macroeconomic Risk? Evidence from Scheduled
  Economic Announcements." *JFQA* 48(2), 343-375. doi:10.1017/S002210901300015X.
  https://ideas.repec.org/a/cup/jfinqa/v48y2013i02p343-375_00.html [ABS]
  - Announcement days are those with scheduled inflation (CPI/PPI), employment and FOMC releases.
  - Average returns and Sharpe ratios are much higher on those days (1958-2009).
  - Their "announcement day" set is close to the paper's (CPI, PPI, Employment + FOMC).
- **Savor, P. & Wilson, M. (2014).** "Asset Pricing: A Tale of Two Days." *JFE* 113(2), 171-201.
  doi:10.1016/j.jfineco.2014.04.005. https://ideas.repec.org/a/eee/jfinec/v113y2014i2p171-201.html [ABS]
  - Beta is priced on announcement days and not on other days.
- **Ai, H. & Bansal, R. (2018).** "Risk Preferences and the Macroeconomic Announcement Premium." *Econometrica* 86(4),
  1383-1430. doi:10.3982/ECTA14607. [ABS]
  - Returns around pre-scheduled releases (employment report, FOMC) account for ~55% of the equity premium.
- **Cieslak, A., Morse, A. & Vissing-Jorgensen, A. (2019).** "Stock Returns over the FOMC Cycle." *JF* 74(5),
  2201-2248. doi:10.1111/jofi.12818. [ABS]
  - The equity premium, 1994-2016, was earned in even weeks of FOMC-cycle time.
  - **Directly relevant to a "time since last statement" feature.** It is the natural citation for cycle-time
    effects of FOMC distance.
- **Hu, G.X., Pan, J., Wang, J. & Zhu, H. (2022).** "Premium for Heightened Uncertainty: Explaining Pre-Announcement
  Market Returns." *JFE* 145(3), 909-936. doi:10.1016/j.jfineco.2021.09.015. https://www.nber.org/papers/w25817 [ABS]
  - VIX builds up over several days before FOMC and resolves at the release.
  - Similar pre-announcement returns appear for NFP, ISM and GDP, without abnormal variance.
- **Bollerslev, T., Li, J. & Xue, Y. (2018).** "Volume, Volatility, and Public News Announcements." *Review of
  Economic Studies* 85(4), 2005-2041. doi:10.1093/restud/rdy003.
  https://public.econ.duke.edu/~boller/Published_Papers/restud_18.pdf [FT]
  - Data: SPY one-minute bars, 2001-2014.
  - "Most of these announcements occur before the market opens". They study only the RTH releases and drop the
    employment report.
  - The most important intraday releases are FOMC, ISM Manufacturing, ISM Non-Manufacturing and Consumer Confidence
    (all 10:00 ET except FOMC).
  - The volume-volatility elasticity at news is below 1.
- **Andersen, T.G., Thyrsgaard, M. & Todorov, V. (2019).** "Time-Varying Periodicity in Intraday Volatility." *JASA*
  114(528), 1695-1707. doi:10.1080/01621459.2018.1512864. WP: CREATES RP 2018-05 [FT, WP]
  - The intraday periodic component varies over time and with the volatility level.
  - FOMC days show "a sharp increase immediately after the announcement".
  - ISM and Consumer Sentiment "have a much smaller impact on the intraday volatility pattern than the FOMC
    announcement".
  - Constant periodicity is still rejected when FOMC days are excluded, so "scheduled macro announcements cannot
    explain the variation in the periodic component".
  - Relevance: the paper's diurnal factor b(u) and the integer hour regressor do not capture event-day shape changes.
    That is the job of the calendar features.

### 1c. Jumps and news
- **Evans, K.P. (2011).** "Intraday Jumps and US Macroeconomic News Announcements." *JBF* 35(10), 2511-2527.
  doi:10.1016/j.jbankfin.2011.02.018. [ABS]
  - About one-third of intraday jumps in US futures (incl. E-mini S&P 500) coincide with macro releases.
  - Surprises explain much of the jump size.
- **Lahaye, J., Laurent, S. & Neely, C.J. (2011).** "Jumps, Cojumps and Macro Announcements." *J. Applied
  Econometrics* 26(6), 893-921. doi:10.1002/jae.1149. [ABS]
  - Nonfarm payrolls and fed-funds target announcements matter most across asset classes. Payrolls matter most for
    stock-index and bond futures.
  - Many releases cause no jump, and many jumps are unrelated to releases.
- **Lee, S.S. (2012).** "Jumps and Information Flow in Financial Markets." *RFS* 25(2), 439-479.
  doi:10.1093/rfs/hhr084. [ABS]
  - Jumps tend to follow Fed announcements, nonfarm payrolls and *weekly jobless claims*.
  - Jobless claims is an 8:30 Thursday release that the paper's list omits.
- **Miao, H., Ramchander, S. & Zumwalt, J.K. (2014).** "S&P 500 Index-Futures Price Jumps and Macroeconomic News."
  *J. Futures Markets* 34(10), 980-1001. https://ideas.repec.org/a/wly/jfutmk/v34y2014i10p980-1001.html [ABS]
  - More than three-quarters of 8:30-8:35 jumps coincide with 8:30 releases.
  - Most 10:00-10:05 jumps coincide with 10:00 releases.
  - NFP and Consumer Confidence are significant.
- **Rangel, J.G. (2011).** "Macroeconomic News, Announcements, and Stock Market Jump Intensity Dynamics." *JBF* 35(5),
  1263-1276. https://ideas.repec.org/a/eee/jbfina/v35y2011i5p1263-1276.html. WP: Banco de México 2009-15 [FT, WP]
  - Data: daily S&P 500, 1992-2008. Releases covered: CPI, PPI, FFR, NFP and unemployment.
  - Fn. 15: CPI, PPI and employment come out "early in the morning (before the market opens)".
  - RV is higher on FFR and NFP days. The day-before "calm" is not significant at daily frequency.
  - On the day itself, only employment days raise jump intensity.
  - OOS results are in Section 2.

### 1d. Which announcements matter for US equity volatility
- **Tier 1:**
  - FOMC statement: LLN 2011, Lee 2012, ATT 2019, BLX 2018, Lucca-Moench.
  - Employment Situation (NFP), the consensus top 8:30 release: ABDV 2007, LLN 2011, Rangel 2011, Gilbert et al.
    2017, Miao et al. 2014, Lee 2012.
  - CPI: always important for inflation (Savor-Wilson; Rangel's persistent inflation effect). Its *equity* impact
    jumped in 2021-2023 (Kroner 2025, FEDS 2025-022,
    https://www.federalreserve.gov/econres/feds/files/2025022pap.pdf [FT p.1-2]), squarely inside the paper's OOS
    span.
- **Tier 2:**
  - PPI: Ederington-Lee 1993; Rangel's PPI surprises persistent; Vortelinos 2015 ranks PPI and trade balance as the
    most significant in OOS forecasts.
  - Retail sales.
  - GDP advance estimate: Hu et al. 2022 use it; the second and third estimates move markets far less [MEM].
- **Tier 3:**
  - Trade balance: Flannery-Protopapadakis find a volatility effect; Vortelinos 2015 too. Most event studies find
    small equity responses.
- **Missing but relevant:**
  - ISM Manufacturing / Non-Manufacturing (10:00 ET, *inside* the 10:00-10:30 bar).
  - Conference Board Consumer Confidence (10:00).
  - Weekly initial jobless claims (8:30 Thursday; Lee 2012).
  - FOMC minutes (14:00, three weeks after the meeting).
  - BLX 2018 and Miao et al. 2014 show the 10:00 releases move the E-mini *during RTH*.
- **Why the ranking differs by release:** Gilbert, Scotti, Strasser & Vega (2017), "Is the Intrinsic Value of a
  Macroeconomic News Announcement Related to Its Asset Price Impact?", *JME* 92, 78-95,
  doi:10.1016/j.jmoneco.2017.09.008 [ABS]. Impact tracks the release's relation to GDP, inflation and fed-funds
  fundamentals, plus timeliness.
- **Attention and time variation:**
  - Fisher, Martineau & Sheng (2022), *RFS* 35(11), 5057-5093, doi:10.1093/rfs/hhac011 [ABS]: attention to
    unemployment and FOMC predicts announcement premia and implied-volatility changes.
  - Boguth, Grégoire & Martineau (2019), *JFQA* 54(6), 2327-2353 [ABS]: meetings with a press conference attract
    more attention and carry more information. Relevant to the 2011-2018 "press-conference vs not" FOMC
    heterogeneity.

### 1e. Timing: 8:30 ET, the 9:30 open and the first RTH bars
- All six series in the paper are released at 8:30 ET, during the CME Globex session and before the cash open
  (ABDV 2007; Rangel 2011 fn. 15; BLX 2018).
- The jump happens in the first minutes (ABDV 2007; Miao et al. 2014: >3/4 of 8:30-8:35 jumps are news) and therefore
  in an *unscored* overnight bar (8:30-9:00).
- What reaches the scored bars:
  1. Elevated but decaying post-release volatility (Ederington-Lee: ~15 min high, then modestly elevated for hours).
  2. The cash open processing the overnight information.
  3. Lower dispersion later in the day once the uncertainty is resolved (cf. Ederington-Lee 1996 on implied
     volatility).
- Expect any gain to concentrate in the 10:00 and 10:30 bars.
- **Design implication.** The paper's lag averages include the overnight bars, so the 8:30-9:30 RV is in
  y^(1)/y^(5) for the 10:00 bar.
  - The HAR benchmark already "sees" the release reaction before forecasting the first RTH bar.
  - The calendar's *incremental* value on release days therefore has three possible sources:
    (a) the shape of the decay;
    (b) the next-day discount (Jones-Lamont-Lumsdaine transitory shocks);
    (c) any pre-release calm on the eve or the morning.
  - Expect smaller gains than for FOMC, whose 14:00 statement falls *inside* RTH. The paper should say so ex ante.
- **Kurov, A., Sancetta, A., Strasser, G. & Wolfe, M.H. (2019).** "Price Drift before U.S. Macroeconomic News:
  Private Information about Public Announcements?" *JFQA* 54(1), 449-479. [ABS]
  - E-mini and Treasury futures begin moving ~30 min before several releases, so for 8:30 releases the drift lies in
    the overnight 8:00-8:30 bar.
  - Pre-release leakage is about *direction*. It is not a lookahead problem for a calendar feature.

---

## 2. Volatility forecasting with announcement information

### 2a. HAR and daily RV with announcement dummies (closest antecedents)
- **Martens, M., van Dijk, D. & de Pooter, M. (2009).** "Forecasting S&P 500 Volatility: Long Memory, Level Shifts,
  Leverage Effects, Day-of-the-Week Seasonality, and Macroeconomic Announcements." *International Journal of
  Forecasting* 25(2), 282-303. doi:10.1016/j.ijforecast.2009.01.010.
  https://econpapers.repec.org/RePEc:eee:intfor:v:25:y:2009:i:2:p:282-303 [ABS; the specific release list was not
  retrieved, full text paywalled]
  - Daily RV of S&P 500 futures.
  - Adding day-of-week, holiday and macro-announcement effects (with level shifts, leverage and a high-order AR
    instead of ARFIMA) improves OOS forecasts up to 20 days ahead.
  - **The closest published precedent for "HAR-type + announcement dummies" on S&P futures. Must cite.**
- **Rangel (2011)**, above. One-day-ahead recursive forecasts of S&P 500 volatility, OOS 2004-2008, scored by RMSE on
  high-low range volatility on event days (WP Table 8) [FT, WP].

  | Event days | Baseline RMSE | Announcement-only ("no foresight") RMSE |
  |---|---|---|
  | PPI | 2.069 | 1.989 |
  | FFR | 2.558 | 2.524 |
  | NFP/UMP | 1.658 | 1.590 |

  - Gains are roughly 1-4%, with no formal test.
  - "Perfect foresight" of the FFR surprise lowers the FFR RMSE to 2.268. The WP shows this as an infeasible
    benchmark.
- **Vortelinos, D.I. (2015).** "Out-of-Sample Evaluation of Macro Announcements, Linearity, Long Memory, Heterogeneity
  and Jumps in Mini-Futures Markets." *Review of Financial Economics* 27, 58-67. doi:10.1016/j.rfe.2015.09.001.
  https://ideas.repec.org/a/eee/revfin/v27y2015icp58-67.html [ABS]
  - Mini-futures, including the E-mini.
  - Announcements improve OOS RV forecasts for only a few release categories. Trade balance and PPI are the most
    significant.
  - Long memory matters far more.
- **Vortelinos, D.I. & Gkillas, K. (2018).** "Intraday Realised Volatility Forecasting and Announcements."
  *International Journal of Banking, Accounting and Finance* 9(1). doi:10.1504/IJBAAF.2018.089426.
  https://www.inderscience.com/offer.php?id=89426 [ABS]
  - **Intraday RV forecasting** in equity, energy and commodity mini-futures, OOS 2009-2011.
  - Accuracy is evaluated separately on announcement days and on all days.
  - Some releases are forecast better by nonlinear and combined models on announcement days.
  - Low-tier journal, but the closest *intraday* precedent.
- **Plíhal, T. (2021).** "Scheduled Macroeconomic News Announcements and Forex Volatility Forecasting." *Journal of
  Forecasting* 40(8), 1379-1397. doi:10.1002/for.2773.
  https://ideas.repec.org/a/wly/jforec/v40y2021i8p1379-1397.html [ABS]
  - HAR on EUR/USD, GBP/USD and USD/JPY RV, 2009-2017, with scheduled-announcement terms.
  - OOS accuracy improves by up to 12.4%.
- **Lyócsa, Š., Molnár, P. & Plíhal, T. (2019).** "Central Bank Announcements and Realized Volatility of Stock Markets
  in G7 Countries." *JIFMIM* 58, 117-135. doi:10.1016/j.intfin.2018.09.010.
  https://ideas.repec.org/a/eee/intfin/v58y2019icp117-135.html [ABS]
  - HAR-type models of G7 index RV, 2006-2016.
  - RV rises significantly on domestic rate-announcement days and falls in the five days after. QE announcements have
    no effect.
  - Relevance: an analogue of the FOMC "since" decay.
- **Halousková, M. & Lyócsa, Š. (2025).** "Forecasting U.S. Equity Market Volatility with Attention and Sentiment to
  the Economy." arXiv:2503.19767. https://arxiv.org/abs/2503.19767 [ABS]
  - Attention and sentiment for ten scheduled macro variables improve RV forecasts for 404 US stocks, with the
    largest gains (~15%) on extreme days.
  - Working paper.

### 2b. Intraday volatility models with announcement components
- Andersen-Bollerslev (1998), the FFF with announcement dummies, is the canonical intraday model.
- **Engle, R.F. & Sokalska, M.E. (2012).** "Forecasting Intraday Volatility in the US Equity Market. Multiplicative
  Component GARCH." *JFEc* 10(1), 54-83. doi:10.1093/jjfinec/nbr005. **IN BIB (EngleSokalska2012)**
  - Daily × diurnal × intraday-GARCH components for 10-min returns.
  - No explicit announcement component [MEM]. That gap is what the paper fills.
- ATT (2019) and Andersen, Su, Todorov & Zhang (2024, JASA 119(546), IN BIB) treat the periodic component, with the
  event-day shape separated out.

### 2c. ML and HAR forecasting with event indicators
- **Christensen, K., Siggaard, M. & Veliyev, B. (2023).** "A Machine Learning Approach to Volatility Forecasting."
  *JFEc* 21(5), 1680-1727. doi:10.1093/jjfinec/nbac020. **IN BIB.** WP: CREATES RP 2021-03 [FT, WP]
  - The extended set M3 includes "an indicator for earnings announcements (EA)", "a dummy ... equal to one if the
    company makes a pre-scheduled earnings announcement on a given day".
  - EA is in the BIC-selected linear HAR-X (M2).
  - "The EA variable has a low VI score ... Earnings information is released infrequently, but it is arguably very
    important for making short-run predictions of realized variance in the days surrounding the announcement."
  - The introduction notes that the HAR extensions (ABD 2007, Corsi-Renò 2012, BPQ 2016) "are solely based on past
    return histories ..., which is surprising, since a large amount of research documents an intimate relationship
    between news announcements and volatility".
  - **This matches the paper's FOMC finding: a low aggregate importance score but a concentrated gain on event days.
    Cite it to frame the event-day split.**
- **Corsi (2009)** and **BPQ (2016)** contain no announcement terms. Both are IN BIB.
  - So no canonical HAR paper includes announcements.
  - The paper's own claim to novelty is the *intraday, penalized and tree-based, out-of-sample* use of
    lookahead-free announcement *calendars*, with day-split evaluation.
- Rahimikia & Poon (IN BIB) use news counts. Caporin & Poli (2017) and Audrino-Sigrist-Ballinari (2020) (IN BIB) use
  news and sentiment measures. They are complements, not calendars.

### 2d. Do announcement calendars improve RV forecasts out of sample?
The evidence is mixed and small.
- Martens et al.: yes, daily, S&P.
- Rangel: 1-4% RMSE on event days, untested.
- Vortelinos: only a few categories (PPI, trade).
- Plíhal: up to 12% for FX.
- No published study I found reports a *DM/MCS-tested* intraday S&P RV gain from 8:30 calendars split by release
  type. That is a gap the paper can claim cautiously.
- Expect small effects for releases before the open: the lagged pre-open RV already embeds them (Section 1e).

---

## 3. Lookahead and scheduling (scheduled vs actual dates)

### 3a. Calendars are public ex ante
- **US releases.** OMB Statistical Policy Directive No. 3 governs Principal Federal Economic Indicators.
  - Agencies send OMB their release schedule for the next calendar year, and OMB publishes the consolidated calendar
    (historically in the autumn).
  - Each release announces the date and time of the next one.
  - Any change must be announced and explained as soon as it is known.
  - Archived OMB notices: https://obamawhitehouse.archives.gov/omb/pei_1999 and
    https://clintonwhitehouse4.archives.gov/textonly/OMB/pubpress/pei98.html [ABS]
  - So a *vintage* calendar, the schedule as known at t, is available. The "next release" distance is lookahead-free
    when built from it.
- **FOMC.** Meeting schedules are published a year ahead. Lucca-Moench (fn. 4) note the press information sheets
  before 2001 and the website afterwards [FT].
  - Lucca-Moench separate unscheduled meetings and calls using the Greenbook criterion. Savor-Wilson and Lucca-Moench
    use scheduled announcements only.

### 3b. Where "actual" differs from "scheduled" inside the paper's sample (1998-2024 training, 2014-2024 OOS)
- **Oct 2013 shutdown (1-16 Oct).** The BLS release schedule was revised.
  - Sept 2013 Employment Situation: scheduled Fri 4 Oct, released Tue 22 Oct.
  - Sept CPI: scheduled ~16 Oct, released 30 Oct.
  - Sources: BLS blog https://www.bls.gov/blog/2013/post-government-shutdown-updates-new-bls-k-12-pages.htm and news
    reports [ABS].
  - PPI, retail sales, trade and Census/BEA releases were also rescheduled [MEM on exact dates].
- **Dec 2018 - Jan 2019 shutdown (Commerce closed; BLS funded, so CPI, PPI and jobs stayed on time).** BEA
  (https://www.bea.gov/news/blog/2019-01-28/bea-economic-reports-delayed [FT]) postponed:
  - Q4 2018 GDP advance: scheduled Wed 30 Jan 2019.
  - Nov 2018 trade: scheduled 8 Jan.
  - Dec 2018 trade: scheduled Tue 5 Feb.
  - Dec personal income: scheduled 31 Jan.

  Reported new dates (news reports [ABS]; confirm against the Census/BEA notices):
  - Dec 2018 retail sales: 14 Feb 2019.
  - Q4 GDP: a combined "initial" estimate on 28 Feb 2019.
  - Dec trade: early March 2019.

  **Consequence:**
  - A "next release" feature built from *actual* dates knows, in early January, a later date than the market knew,
    so it uses future information.
  - One built from the *original schedule* is wrong but lookahead-free: "known schedule as of t, updated when the
    agency announces a change".
  - The latter is the correct vintage rule, the same treatment the paper already gives the 15 March 2020 FOMC action.
- **Unscheduled FOMC:** 3 Mar 2020 (10:00 ET) and 15 Mar 2020 (Sunday). Both are already handled.
- **Good Friday.** The Employment Situation is often released while the NYSE is closed and CME equity futures trade
  an abbreviated session: 2 Apr 2021 and 7 Apr 2023 [ABS]; 3 Apr 2015 and 6 Apr 2012 [MEM]. These are "NYSE holiday"
  days whose bars the paper forecasts but does not score. The "since last" counter must still count them.
- **Co-releases.** CPI and retail sales, PPI and retail sales, and the trade balance with other 8:30 data often share
  a morning, and GDP sometimes coincides with other releases.
  - The six indicators will be collinear on those days.
  - Report the co-release frequency, or pool the releases into "any 8:30 tier-1/tier-2" alongside the separate series.
- **GDP vintages.** The advance, second and third estimates are all scheduled at 8:30. Market impact is concentrated
  in the advance estimate. State which vintages the GDP calendar includes, or split them.

### 3c. Calendar vs surprise, and what is lookahead
- **Rangel (2011)** separates "no foresight", where surprises are set to zero and only the announcement-day terms are
  used, from "perfect foresight", where realized surprises are used and which is infeasible ex ante [FT, WP]. The
  calendar approach is the "no foresight" case.
- **Intraday nuance (important).** The paper scores bars from 10:00 to 16:00, and the 8:30 number is public by 9:30.
  - The realized surprise (actual minus consensus) is therefore *in the information set* for every scored RTH bar of a
    release day.
  - Using |surprise| would not be lookahead for RTH-bar forecasts made after 8:30, though it would be for
    prior-day/overnight bars.
  - This needs time-stamped consensus data (Bloomberg/MMS). It is the natural referee follow-up to the calendar
    features. ABDV 2007 fn. 11 suggests the dummy captures most of the volatility effect anyway.
- **Missing-event issue.** Gürkaynak, Kısacıkoğlu & Wright (2020), "Missing Events in Event Studies: Identifying the
  Effects of Partially Measured News Surprises", *AER* 110(12), 3871-3912, doi:10.1257/aer.20181470 [ABS].
  - Headline surprises miss much of the news in a release, which supports calendar/latent-factor approaches over
    headline-surprise regressors.
- **Pre-release drift (Kurov et al. 2019)** is a price-direction effect before 8:30. It does not compromise a
  calendar feature.

---

## 4. What a referee would expect

### Citations
- **Core set:**
  - Andersen-Bollerslev 1998; Ederington-Lee 1993 (+1996 for IV); ABDV 2003/2007.
  - Bollerslev-Cai-Song 2000 or Jones-Lamont-Lumsdaine 1998, for transitory announcement shocks.
  - Bomfim 2003 and Lucca-Moench 2015, for the pre-FOMC calm.
  - Savor-Wilson 2013 and Ai-Bansal 2018, for why scheduled days matter.
  - Martens-van Dijk-de Pooter 2009 and Rangel 2011, for the OOS forecasting precedent.
  - Christensen et al. 2023, for an event dummy inside ML RV forecasting.
  - Gilbert et al. 2017 and LLN 2011, for which releases matter.
  - Bollerslev-Li-Xue 2018 and ATT 2019, for RTH vs pre-open timing and FOMC-day intraday shape.
  - Cieslak-Morse-Vissing-Jorgensen 2019, for FOMC cycle time.
  - Kroner 2025, for the time-varying CPI importance.

### Tests
1. **Per-release attribution.**
   - Six calendars entered jointly and one at a time (add-one / drop-one), with DM/HLN on the *release-day subsample*
     for each series.
   - Expect NFP and CPI to carry the gain and GDP and trade to add nothing (cf. Vortelinos 2015, which favors PPI and
     trade).
   - Report the number of release days per series in the OOS span. Power is low: about 120 releases per series in
     10 years.
2. **Bar-level profile.** Gains on release days by bar (10:00, 10:30, ..., 16:00). Ederington-Lee and the 8:30 timing
   predict concentration in the first one or two bars. A flat profile would suggest the features pick up something
   else, such as a day-of-week confound: NFP is almost always a Friday.
3. **Incremental over the pre-open reaction.**
   - Show the calendars add beyond the overnight HAR lags: drop them, or interact them with the 8:30-9:30 RV.
   - Show they add beyond VIX and VIX3M (Ederington-Lee 1996 and Wright 2025 show options price event variance).
4. **Day-after and eve effects.** Jones-Lamont-Lumsdaine predict lower forecast error the day after; Bomfim and
   Lucca-Moench predict a calm on the eve. The "until" and "since" proximities should capture both. Show the
   eve-of-release and day-after splits.
5. **Calendar vs surprise.**
   - Calendar-only vs calendar + |surprise| on RTH bars (lookahead-free after 8:30).
   - Or show that release-day forecast errors correlate with |surprise| (Rangel's foresight comparison).
6. **Robustness to the schedule vintage.** Rebuild the features from the as-scheduled calendar with change
   announcements (2013 and 2018-19 shutdowns) vs actual dates, and show the results are unchanged. This preempts the
   lookahead objection.
7. **Subsamples.** Pre-2021 vs 2021-2024 (the inflation regime; Kroner 2025), which should show CPI gaining weight.
   Also the post-2015 FOMC drift decline (Kurov-Wolfe-Gilbert 2021).
8. **Omitted releases.** ISM, Consumer Confidence (10:00, *inside* RTH; BLX 2018, Miao et al. 2014), weekly jobless
   claims (Lee 2012) and FOMC minutes. Either justify leaving them out or add the 10:00 group as a robustness check.
9. **Inference.**
   - MCS (HansenLundeNason2011, IN BIB) on the release-day subsample.
   - A Giacomini-White (2006, *Econometrica* 74(6), 1545-1578; NOT IN BIB) conditional predictive ability test,
     conditioning on the release-day indicator. This is the formal version of "the gain is on statement days".
10. **Model-class comparison.** Day-level indicators in a linear model shift all bars equally on the diurnally scaled
    target. Trees learn indicator × hour interactions. A referee will ask whether the linear models get an
    indicator × early-bar interaction, or whether the tree advantage on release days is just that interaction.

---

## Sources (URLs used)
- AB1998: https://scholars.duke.edu/publication/761296
- Ederington-Lee 1993: https://ideas.repec.org/a/bla/jfinan/v48y1993i4p1161-91.html
- Ederington-Lee 1996: https://ideas.repec.org/a/cup/jfinqa/v31y1996i04p513-539_02.html
- ABDV 2003: https://www.nber.org/system/files/working_papers/w8959/w8959.pdf
- ABDV 2007: https://repec.econ.au.dk/repec/creates/rp/07/rp07_20.pdf
- BCS 2000: https://scholars.duke.edu/publication/644374
- JLL 1998: https://ideas.repec.org/a/eee/jfinec/v47y1998i3p315-337.html
- Bomfim 2003: https://ideas.repec.org/a/eee/jbfina/v27y2003i1p133-151.html
- Lucca-Moench 2015: https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr512.pdf
- Savor-Wilson 2013: https://ideas.repec.org/a/cup/jfinqa/v48y2013i02p343-375_00.html
- Savor-Wilson 2014: https://ideas.repec.org/a/eee/jfinec/v113y2014i2p171-201.html
- Ai-Bansal 2018: https://www.econometricsociety.org/doi/10.3982/ECTA14607
- CMVJ 2019: https://econpapers.repec.org/RePEc:bla:jfinan:v:74:y:2019:i:5:p:2201-2248
- Hu-Pan-Wang-Zhu 2022: https://www.nber.org/papers/w25817
- Kurov-Wolfe-Gilbert 2021: https://ideas.repec.org/a/eee/finlet/v40y2021ics1544612320315956.html
- Kurov et al. 2019: https://econpapers.repec.org/RePEc:cup:jfinqa:v:54:y:2019:i:01:p:449-479_00
- BLX 2018: https://public.econ.duke.edu/~boller/Published_Papers/restud_18.pdf
- ATT 2019: https://repec.econ.au.dk/repec/creates/rp/18/rp18_05.pdf
- ASTZ 2024: https://ideas.repec.org/a/taf/jnlasa/v119y2024i546p1181-1191.html
- Evans 2011: https://ideas.repec.org/a/eee/jbfina/v35y2011i10p2511-2527.html
- LLN 2011: https://ideas.repec.org/a/wly/japmet/v26y2011i6p893-921.html
- Lee 2012: https://ideas.repec.org/a/oup/rfinst/v25y2012i2p439-479.html
- Miao et al. 2014: https://ideas.repec.org/a/wly/jfutmk/v34y2014i10p980-1001.html
- Rangel 2011: https://ideas.repec.org/a/eee/jbfina/v35y2011i5p1263-1276.html ; WP https://ideas.repec.org/p/bdm/wpaper/2009-15.html
- Gilbert et al. 2017: https://econpapers.repec.org/RePEc:eee:moneco:v:92:y:2017:i:c:p:78-95
- Kroner 2025: https://www.federalreserve.gov/econres/feds/files/2025022pap.pdf
- Martens et al. 2009: https://econpapers.repec.org/RePEc:eee:intfor:v:25:y:2009:i:2:p:282-303
- Vortelinos 2015: https://ideas.repec.org/a/eee/revfin/v27y2015icp58-67.html
- Vortelinos-Gkillas 2018: https://www.inderscience.com/offer.php?id=89426
- Plíhal 2021: https://ideas.repec.org/a/wly/jforec/v40y2021i8p1379-1397.html
- LMP 2019: https://ideas.repec.org/a/eee/intfin/v58y2019icp117-135.html
- Halousková-Lyócsa 2025: https://arxiv.org/abs/2503.19767
- Christensen et al. 2023: https://repec.econ.au.dk/repec/creates/rp/21/rp21_03.pdf
- Engle-Sokalska 2012: https://ideas.repec.org/a/oup/jfinec/v10yi1p54-83.html
- Fisher-Martineau-Sheng 2022: https://ideas.repec.org/a/oup/rfinst/v35y2022i11p5057-5093..html
- Boguth et al. 2019: https://ideas.repec.org/a/cup/jfinqa/v54y2019i6p2327-2353_2.html
- Wright 2025 (Event-day options, JTSA 46(6) 1085-1097): https://econpapers.repec.org/RePEc:bla:jtsera:v:46:y:2025:i:6:p:1085-1097
- GKW 2020: https://econpapers.repec.org/RePEc:aea:aecrev:v:110:y:2020:i:12:p:3871-3912
- OMB PEI calendar / Directive 3: https://obamawhitehouse.archives.gov/omb/pei_1999
- BEA 2019 delays: https://www.bea.gov/news/blog/2019-01-28/bea-economic-reports-delayed
- BLS 2013 shutdown: https://www.bls.gov/blog/2013/post-government-shutdown-updates-new-bls-k-12-pages.htm
