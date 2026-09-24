# One-time intraday history purchase for the close-signal service (priced 2026-09-23)

Purpose: fill the research panel's gap (ES 1-minute bars 2024-04 -> today; Cboe VIX / VVIX / VIX3M intraday
2024-02 -> today), after which the free daily feeds append one session at a time and the per-bar ridge updates
incrementally. Priced by a subagent from the vendors' own pages (V = verified on the page; I = inferred / behind a
JavaScript cart, confirm at checkout). Nothing was signed up for or bought.

## Shopping list

| Line | What | Cost | Status |
|---|---|---|---|
| 1 | Databento `GLBX.MDP3`, schema `ohlcv-1m`, symbols `ES.FUT` (every contract month) + `ES.v.0` (volume-roll continuous), 2024-04-01 -> 2026-09-23, all Globex hours; DBN or CSV via the Python client / batch download | **$0 out of pocket**: the new-account credit is "$125 in free credits that can be used towards historical data", expires 6 months after signup (V); the order is ~50 MB front month / 100-200 MB all months, single- to low-double-digit dollars at any plausible per-GB rate (I -- run `Historical().metadata.get_cost()` before submitting). Personal use: instant approval on the usage-based plan (V) | buy when ready to pull (the credit clock starts at signup) |
| 1b | Same span, `VX.FUT` ohlcv-1m (VIX futures) | inside the same credit | optional proxy / cross-check for the VIX leg |
| 2 | FirstRate Data "US Indices" bundle (126 tickers; lists VIX, VVIX, VIX3M, VIX9D, VIX1D, SKEW, SPX), 1-min / 5-min / 30-min / 1-h / 1-day OHLC, zipped CSV, immediate download. Spans: VIX and VIX3M from 2008-01-02, VVIX from 2012-03-14, SPX from 2008-01-02, all to 2026-09-23 (V) | one-time price rendered only in the cart (I: low hundreds of dollars for the bundle, singles under $100); "1 month of free updates are included. Updates are $59.95 per month thereafter" (V) -- decline the updates, the free feeds take over | if singles are cheaper in the cart, buy VIX + VVIX + VIX3M (+ SPX optional) |
| 3 | SPX 1-min fallback | $0 marginal -- inside line 2 | optional |
| alt 2 | Cboe DataShop "Main Channel Tick Data" (SPX, VIX, XSP) and "Cboe Global Indices Channel Tick Data" (the other CGIF indices, where VVIX / VIX3M sit): every disseminated value, 15-second native, from January 2004 (V); price configured in the cart per symbol x span (I). The $1000/month figure on the site is the live FEED licence, not the file | audit-grade upgrade if the FirstRate 1-min VVIX / VIX3M look oddly aggregated |

Ruled out: Polygon/Massive Indices ($49/mo 15-min delayed, $99/mo real-time) has "1+ year" of history only -- does not
reach 2024-02; Tiingo and Nasdaq Data Link carry no Cboe index intraday history; Kibot's futures bundles start at
$400 (ES 1-min included, no per-symbol fee) -- dominated by the Databento credit; Portara / Barchart OnDemand / CME
DataMine are quote-only or raw ticks.

No free source of 1-minute VVIX or VIX3M history exists (Cboe's own site: daily closes 1990 -> present; FRED,
Macroption, Kaggle, WRDS: daily). The retail-priced sources are exactly FirstRate Data and Cboe DataShop.

## Yahoo Finance delays (Yahoo's exchange-delay page, read 2026-09-23) -- the live-feed constraint

| Symbol | Exchange line on Yahoo's page | Delay |
|---|---|---|
| ^VIX, ^VVIX, ^VIX3M | "Cboe Indices ... 15 min ... ICE Data Services" | **15 minutes** |
| ES=F | "Chicago Mercantile Exchange (CME) ... 10 min" | **10 minutes** |
| ^GSPC | "S & P Indices ... Real-time" | real-time |

So at 15:30 the free feeds do not carry the 15:00-15:30 ES bar (arrives ~15:40) or the 15:30 Cboe prints (~15:45).
The close-signal service therefore has three input modes (live/close_signal/README.md): `ibkr` (real-time ES and
Cboe prints through the existing broker code -- the research construction; IBKR non-professional market data is a
few dollars a month with an account), `free_substitute` (the real-time ^GSPC 1-minute bars for the last bar's return
moments in place of ES, the Cboe prints as of ~15:15 with the lag journaled -- a MODEL CHANGE to be validated on the
purchased history before it is trusted), and `free_delayed` (wait for the 15:30 bar and prints, ~15:45, and trade
late -- not the research object). The daily APPEND of the panel is unaffected by the delays (it runs after the
close).

## Checks on receipt

1. Timestamps: Databento `ts_event` is UTC nanoseconds and marks the bar OPEN; FirstRate stamps are US Eastern,
   also bar-start (I). The panel is naive-ET bar-END labelled: shift +1 minute, strip the zone, then join (the
   2026-08-17 misjoin rule).
2. ES continuous: Databento continuous series are unadjusted; `.c` rolls at expiry, `.v` by volume, `.n` by open
   interest -- `.v` matches a live front month; or take `ES.FUT` and roll by the panel's own rule. FirstRate's roll
   rule is not stated; its absolute / ratio "adjusted" files alter levels -- use only unadjusted per-contract files.
3. Sessions: Databento ohlcv-1m emits a bar only for minutes with trades (gaps, not zeros, in back months and the
   17:00-18:00 ET halt). VIX has a 03:15-09:15 ET extended session; VVIX / VIX3M are RTH-only prints -- confirm the
   FirstRate files' session coverage and the 16:00-16:15 tail.
4. Quality: FirstRate index bars are aggregated from 15-second prints -- expect flat / duplicate minutes on quiet
   stretches, missing early-close afternoons, a meaningless volume column. Spot-check ~5 days against Cboe's free
   daily OHLC and against Databento `VX` moves before trusting the VVIX / VIX3M legs.
5. Parity gates (the build carries them): purchased ES 1-min -> the panel's moment columns vs the vendor panel on the
   2024-04 overlap; Yahoo's last 30 days vs the purchased files on their overlap.

Sources: https://databento.com/pricing ; https://databento.com/catalog/cme/GLBX.MDP3/futures/ES ;
https://firstratedata.com/b/8/us-index-historic-intraday ; https://firstratedata.com/i/index/VVIX ;
https://firstratedata.com/i/index/VIX3M ; https://firstratedata.com/i/futures/ES ;
https://datashop.cboe.com/main-channel-tick-data ; https://datashop.cboe.com/cboe-global-indices-channel-tick-data ;
https://massive.com/pricing?product=indices ; https://help.yahoo.com/kb/SLN2310.html ;
https://www.cboe.com/tradable-products/vix/vix-historical-data ;
https://www.kibot.com/Historical_Data/Top_10_Futures_Historical_Intraday_Data.aspx
