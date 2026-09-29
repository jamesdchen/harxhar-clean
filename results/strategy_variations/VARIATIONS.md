# Strategy variations of the last-30-min trade (checklist F1) — written down

**Source of the list.** The professor's on-screen list (written about a week before
2026-09-29) is not in the repository, and the user was not available to supply it.
This file uses the standard set his note names — strangles, butterfly spreads, iron
condors, "different combos" — completed with the iron butterfly, the credit
verticals, the 1×2 ratio spreads and the straddle-plus-wing combination that the
earlier (parked) iron-fly and condor work explored. If the on-screen list differs,
add its rows to `VARIATIONS` in `experiments/strategy_variations_1530.py`; the
scoring is generic in the legs.

Scored by `experiments/strategy_variations_1530.py`; results in `SUMMARY.md` and the
CSVs beside this file. LaTeX twin: `writeup/generated/strategy_variations.tex`.

## What every variation shares

- **Instrument.** SPXW options with same-day expiry (0DTE; calendars are out of
  scope), European, cash-settled. Entered at the 15:30 ET quote, held to the close,
  settled at intrinsic value against the official S&P 500 close — the settlement of
  the straddle trade. No exit trade and no exit spread. One position a day.
- **Days.** The 866 scored days of the straddle trade (2020-01-03 to 2024-04-30),
  the same days for every variation; a variation that cannot be priced on a day
  reports its coverage and is compared with the straddle on the days it has.
- **Quotes and strikes.** Only 15:30 quotes with a live midpoint (finite and
  positive — the straddle trade's filter). $S$ is the 15:30 index print.
  - $K_c$: smallest live call strike $\ge S$; $K_p$: largest live put strike
    $\le S$ (the **straddle**: nearest out-of-the-money call + nearest
    out-of-the-money put, same-day expiry, one position).
  - "One strike further": the next live strike beyond $K_c$ (calls) or below
    $K_p$ (puts) — the earlier condor lab's strangle body.
  - Moneyness strangles: smallest live call strike $\ge S(1+x)$ and largest live
    put strike $\le S(1-x)$, $x \in \{0.5\%, 1\%, 2\%\}$.
  - Wings: the nearest live strike at least $w$ index points beyond the body
    strike (the rule of `asl.pick_wings`), $w \in \{10, 25, 50\}$. 25 and 50 are
    the earlier lab's widths; 10 is added as the narrowest width of the earlier
    width ladder — the width at which the wing actually binds: the close lands
    beyond the iron butterfly's wings on 22.3 % of the days at $w=10$, 4.0 % at
    $w=25$ and 1.0 % at $w=50$ (`cost_line.csv`,
    `share_close_beyond_outermost_strike`).
  - Butterflies: centre $K_0$ = the straddle strike nearer to $S$ ($K_c$ on a
    tie); wings at exactly $K_0 \pm w$ (a butterfly needs equal wings).
- **The sign(s) rule.** $s=\widehat{RV}-\mathrm{IV}_{30}^2$. Every variation is
  written in its **buy-side form $L$** — the structure bought when $s>0$. The position is $q\,L$ with $q=\mathrm{sign}(s)$: $q=+1$ buys $L$,
  $q=-1$ sells it (holds $-L$). **Always short** is $q=-1$ every day (no
  forecast). The combination "straddle, wings on selling days" holds different
  structures on the two sides (below).
- **Forecasts for sign(s)**, two blocks that are never compared with each other:
  1. the **per-bar ridge on the live-feasible set** (16 series a 15:30 forecaster
     can rebuild in real time), scored with the research scorer (16:00-bar
     recalibration: causal per-clock second-moment term, 250-session window) — the
     headline;
  2. the **block-diagonal ridge** (the paper's headline forecast), scored as in the
     paper (the deck's recalibration over the session bars).
- **Fills, at the same 15:30 quotes; positions identical under both.**
  - *mid*: every leg at its midpoint;
  - *crossed*: every leg bought at the ask and sold at the bid (a zero bid sells
    for zero). A worst-case quoted fill, not a market-impact model.
- **Return frames.**
  1. *Per straddle premium* (primary, every variation): the day's P&L of one unit
     of the held structure, in index points, divided by that day's straddle
     premium at the same fill (mid: $\mathrm{mid}_c+\mathrm{mid}_p$; crossed: the
     ask sum on buying days, the bid sum on selling days). The straddle row is the
     straddle trade's return exactly (mid, and the crossed return of the rule
     tables). All variations share the denominator on a given day and fill, so a
     paired difference isolates the structure. This is the "per body premium"
     frame of the parked iron-fly section.
  2. *Own notional* (secondary): the structure's own premium at the fill for the
     premium structures (straddle, strangles: $R=\mathrm{payoff}/P-1$ bought,
     $1-\mathrm{payoff}/P$ sold; undefined when $P\le 0$); capital at risk for the
     defined-risk structures — the larger actual wing gap (iron butterfly, iron
     condor, verticals) or $w$ (single-type butterflies). None for the ratio
     spreads (one leg is uncovered) and for the straddle-plus-wings combination
     (its two sides live in different units).
- **Statistics.** Mean, annualized Sharpe ratio ($\times\sqrt{252}$ per trade day),
  hit rate (share of days with $R'>0$), maximum drawdown of cumulative $R'$
  (no compounding), worst day, the **top-21 share** (sum of the best 21 days —
  one trading month — over the sum of all days), share of buying days. Paired
  circular block bootstrap of each variation minus the straddle, same rule and
  fill (block 21 days, 2,000 draws, seed 0: the rule tables' convention), for the
  Sharpe difference and the mean difference; Newey–West $t$ of the daily
  difference.
- **Cost line.** Bid-ask cost of entering = crossed price minus mid price, as a
  share of the structure's own $|\text{mid premium}|$ and of the straddle premium;
  and the erosion of mean and Sharpe from mid to crossed.

## The variations

$C(K)$, $P(K)$: one call, one put at strike $K$; $+$ bought, $-$ sold, in the
buy-side form $L$ (held when $s>0$; $-L$ is held when $s<0$). Row names in the
CSVs: `straddle`, `strangle, one strike wide`, `strangle 0.5% OTM` (1 %, 2 %),
`iron butterfly, w10` (w25, w50), `iron condor, wW`, `put vertical, wW`,
`call vertical, wW`, `call butterfly, wW`, `put butterfly, wW`,
`put ratio 1x2, wW`, `call ratio 1x2, wW`, `straddle + wing hedge, wW`.

| # | Variation | $L$ (bought when $s>0$) | $-L$ (sold when $s<0$) | Own notional | Purpose |
|---|---|---|---|---|---|
| 1 | **straddle** (baseline) | $+C(K_c)+P(K_p)$ | short straddle | premium | convexity both ways at the money |
| 2 | strangle, one strike wide | $+C(K_c^{+})+P(K_p^{-})$ (next strike beyond each leg) | short strangle | premium | all time value, cheaper, less gamma at the money |
| 3–5 | strangle ±0.5 %, ±1 %, ±2 % | $+C(\ge S(1+x))+P(\le S(1-x))$ | short strangle | premium | tail-only exposure; sold, a small premium for the tails |
| 6–8 | iron butterfly, $w$ | $+C(K_c)+P(K_p)-C(\ge K_c+w)-P(\le K_p-w)$ | short straddle + long wings (credit) | larger wing gap | tail cap on the short side; long side capped too |
| 9–11 | iron condor, $w$ | body = strangle one strike wide, wings $w$ beyond each body leg | short strangle + long wings (credit) | larger wing gap | cheaper body, tail cap |
| 12–14 | put vertical, $w$ | $+P(K_p)-P(\le K_p-w)$ | **credit put vertical** | wing gap | premium from the put side alone, capped |
| 15–17 | call vertical, $w$ | $+C(K_c)-C(\ge K_c+w)$ | **credit call vertical** | wing gap | premium from the call side alone, capped |
| 18–20 | call butterfly at the money, $w$ | short butterfly $-C(K_0-w)+2C(K_0)-C(K_0+w)$ | long butterfly (debit; pays on a pin at $K_0$) | $w$ | defined-risk short volatility with a pin payoff; one wing in the money |
| 21–23 | put butterfly at the money, $w$ | same with puts | long put butterfly | $w$ | as 18–20; tests the in-the-money wing's spread |
| 24–26 | put ratio 1×2, $w$ | backspread $-P(K_p)+2P(\le K_p-w)$ | front ratio $+P(K_p)-2P(K_p-w)$ (one put uncovered) | none | convexity in the crash tail (bought) vs premium from it (sold) |
| 27–29 | call ratio 1×2, $w$ | backspread $-C(K_c)+2C(\ge K_c+w)$ | front ratio $+C(K_c)-2C(K_c+w)$ | none | as 24–26 on the upside |
| 30–32 | straddle + wing hedge (combination), $w$ | the straddle $+C(K_c)+P(K_p)$ | short iron butterfly $-C(K_c)-P(K_p)+C(\ge K_c+w)+P(\le K_p-w)$ | — (per straddle premium only) | cap the short side's tail, keep the long side's convexity (the parked defined-risk design) |

Notes.
- The iron butterfly here is centred on the straddle's two strikes, so when $S$
  sits between strikes its body is one strike wide (as the straddle itself is).
- A long call (put) butterfly and a short iron butterfly on the same strikes have
  the same settlement payoff up to a constant; they differ in their legs' quotes —
  the single-type butterfly carries one in-the-money wing — which is what the
  crossed fill prices.
- Not in the set: calendars and diagonals (not 0DTE), the volatility-target sizing
  overlay (a sizing rule, parked earlier), broken-wing and skewed structures.
