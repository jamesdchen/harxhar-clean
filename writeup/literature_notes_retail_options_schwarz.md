# Literature notes: Christopher Schwarz (UC Irvine) and retail options trading

Compiled 2026-09-04 from public sources (SSRN abstracts blocked; details taken from
the Fed working-paper PDF, the Traders Magazine write-up, the Merage CV, the
Committee on Capital Markets Regulation staff report, and the Bogousslavsky-Muravyev
paper PDF). Numbers are as reported by those sources.

## Who

Christopher G. Schwarz, Professor of Finance, Paul Merage School of Business, UC
Irvine; Faculty Director, Center for Investment and Wealth Management. Research
areas: retail trading, market microstructure, investment funds, regulation.
Frequent co-authors: Xing Huang (WashU), Philippe Jorion (UCI), Brad Barber (UC
Davis), Terrance Odean (Berkeley), Jeongmin "Mina" Lee (Fed).

## The paper on retail OPTIONS

**Huang, Jorion, Schwarz - "Some Anonymous Options Trades Are More Equal than
Others" (SSRN 4951825; first circulated 2024, updated June 2025).**
Design: the authors placed ~7,000 simultaneous, identical market orders in listed
options across six retail brokers from mid-March to end-June 2024, in 18 tickers
(~45% of option volume), chosen so that not every wholesaler could route to an
exchange with an affiliated designated market maker.
Findings: average round-trip execution cost ranges from 0% to 7% across brokers
for the same trades at the same time. The dispersion is created by wholesalers,
who vary both the execution method (auto-execution vs auction etc.) and the pricing
within each method by broker; it is primarily explained by payment for order flow
(two of the six brokers take none). Wholesalers with a routing choice are 69% more
likely than random to route auto-executions to an exchange with an affiliated DMM,
without worse execution on that account. PFOF in options was $1.6 billion in 2023.
Policy: extend equity-style execution-quality disclosure (Rules 605/606-type) to
options. Cited in the Committee on Capital Markets Regulation staff report
"Empirical Research on U.S. Retail Options Markets" (June 2025).

## The equity companions (same method, useful for context)

- **"The 'Actual Retail Price' of Equity Trades"** (Schwarz, Barber, Huang, Jorion,
  Odean; Journal of Finance 2025). 85,000 simultaneous market orders across six
  accounts at five brokers; mean account-level round-trip cost from -0.07% to
  -0.46% before commissions; dispersion comes from wholesalers giving different
  prices to different brokers for the same trade.
- **"Who Is Minding the Store? Order Routing and Competition in Retail Trade
  Execution"** (Huang, Jorion, Lee, Schwarz; Fed FEDS 2024-080, July 2024).
  150,000 actual trades; substantial, persistent dispersion in execution cost
  across wholesalers within a broker; many brokers hardly change routing and
  keep sending more flow to the more expensive wholesalers.
- **"Attention-Induced Trading and Returns: Evidence from Robinhood Users"**
  (Barber, Huang, Odean, Schwarz; Journal of Finance 2022). Intense Robinhood
  buying predicts negative returns (-4.7% 20-day abnormal for the top-bought
  stocks).
- **"A (Sub)Penny for Your Thoughts: Tracking Retail Investor Activity in TAQ"**
  (Barber, Huang, Jorion, Odean, Schwarz). The sub-penny retail identification
  method and its limits.

## Adjacent retail-options papers the same literature cites

- **Bogousslavsky & Muravyev, "An Anatomy of Retail Option Trading"** (Aug 2024).
  Trader-level data, $15 billion of retail stock and option trades. Options are
  over a third of all trades, concentrated in a few underlyings (especially the
  S&P 500 index), dominated by short-term purchases; median days to expiry fell
  from four (2020) to one (2022), and 0DTE are 23% of option trades. Losses are
  surprisingly small despite wide spreads overall, but 0DTE trades lose 4.7%
  relative to other option trades (t = -10; -2.95% with controls) and index/ETF
  option trades earn -3% vs +0.15% for single-stock options.
- **de Silva, Smith & So, "Losing is Optional: Retail Option Trading and Expected
  Announcement Volatility"** (2023-25). Retail loses 5-9% on options around
  earnings announcements, 10-14% when expected volatility is high; retail demands
  liquidity without private information.
- **Bryzgalova, Pavlova & Sikorskaya, "Retail Trading in Options and the Rise of
  the Big Three Wholesalers"** (Journal of Finance 2023). Retail share of option
  volume and the concentration of wholesaling.

## Why it matters for this project

The 0DTE close trade lives or dies on fills. Schwarz's options experiment puts
retail round-trip option execution cost at 0-7% depending on broker; our own
crossed-spread accounting (median half-spread ~1.7% of premium at 15:30, higher
intraday) sits inside that range, and it is exactly the mechanism that turns the
intraday re-pick negative (21-23 crossings a day) while the single close trade
survives at a reduced Sharpe (1.63 midpoint -> 1.11 crossed). The broker choice
is therefore a first-order parameter of any retail implementation, and the
Bogousslavsky-Muravyev 0DTE loss figures are the base rate our trade has to beat.
