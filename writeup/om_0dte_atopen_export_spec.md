# OptionMetrics / intraday export spec — 0–1 DTE at-open implied, to price the §29.1 session VRP

**Purpose (the datum-to-buy):** turn §29.1's remaining-session forecast (integrated vol
10:00→16:00, the 0DTE-at-open object, cached in `straddle_eod.npz`) from a QLIKE proxy into a
**measured VRP** and a friction-priced intraday straddle P&L. Companion to
`writeup/om_friction_export_spec.md` (prop branch) — same WRDS flow, same landing conventions,
same consumer style as `drivers/msweep_2026-08-01/straddle_v3.py`. Decision this export informs:
does forecast-vs-implied at the session horizon clear measured half-spreads in the 0DTE era? If
the Tier-1 (EOD) answer is yes at base costs, buy the Tier-2 (intraday) sample; if no, the
intraday-straddle VRP route closes by measurement.

## The honest problem first

OptionMetrics IvyDB is **EOD-only** — there is no at-open quote in `opprcd`, and the §29.1
object is priced at ~09:30–10:00. No single free export solves this. The spec is therefore
three tiers, ordered by availability:

| Tier | Source | What it prices | Gap |
|---|---|---|---|
| 1 | OM EOD (WRDS, have access) | dte=1 quote at close *t−1* → implied var over [16:00 t−1, 16:00 t] | includes overnight; 17.5h stale at the open |
| 2 | CBOE DataShop / LiveVol intraday (purchase) | same-day-expiry ATM straddle at 09:30–09:45 | exact object; costs money |
| 3 | Cboe VIX1D daily history (free download, 2011+) | 1-calendar-day constant-maturity SPX implied | window mismatch; QC anchor only |

Tier 1 is computable now and becomes the session VRP via the **overnight decomposition**
(below). Tier 2 is the purchase the Tier-1 readout gates.

**Note before re-querying:** the prior landing was `data/om_friction/chain_109820_dte0_10.parquet`
— **dte 0–10, so the 0–1 dte EOD rows already exist in that file** wherever it lives (it is not
in this repo; 2.1GB-class data land is gitignored). If the original parquet is retrievable, Tier
1 needs no new WRDS pull at all — only the widened fields below argue for a re-export.

## Tier 1 — WRDS query (OM `opprcd` + `secprd`)

Two secids, one query each (mirroring the base spec):

- **SPY = `109820`** (primary: pipeline continuity — same schema `straddle_v3` already joins;
  tightest quoted spreads; strikes ≈ SPX/10). Caveat recorded: American exercise / physical
  settlement; early-exercise premium at ≤1 dte near-ATM is negligible for straddle IV, and
  friction realism is the point of this leg.
- **SPX = `108105`** (secondary: SPXW PM-settled European cash — the theoretically clean 16:00
  window). **Filter `am_settlement = 0`** — AM-settled monthlies expire at the *open* and price
  the wrong window; this filter is load-bearing.

**Fields:** `secid, date, exdate, cp_flag, strike_price, best_bid, best_offer, volume,
open_interest, impl_volatility, delta, vega, gamma, ss_flag, am_settlement` (adds
`volume, open_interest, ss_flag, am_settlement` to the base-spec field list — QC needs them).

**Filters (in-query, keep the export small):**
- calendar days to expiry **0–4** (0–1 is the object; 2–4 covers weekend/holiday gaps —
  Friday close → Monday expiry is 3 calendar days; the consumer selects "same or next
  *trading* day");
- moneyness 0.90–1.10 vs same-day `secprd` close (do **not** use a delta band here — at 0–1 dte
  delta collapses toward a step function and the base spec's 0.20–0.80 band drops ATM rows);
- do **NOT** filter `impl_volatility` non-null: OM's IV field is frequently null at ultra-short
  dte. Export raw bid/offer regardless; the consumer backs out ATM IV from the straddle mid via
  Brenner–Subrahmanyam (σ ≈ straddle_mid / (S·√(2τ/π))), which is exact enough ATM at these
  maturities and immune to OM's IV-fitting dropouts. Keep zero-bid rows, flagged, for coverage QC.
- `ss_flag = 0` (standard settlement only).

**Dates:** 2004-01-01 → latest. **Underlying:** `secprd` same secids (`date, close, return`).
**Land as:** `data/om_friction/chain_{secid}_dte0_4.parquet`, reuse
`data/optionm_spx_spot.parquet` conventions (v3's `spot = close/10` for SPY vs SPX-scale
strikes stays as-is).

## Coverage epochs (sets the eval design — expected match rates, not surprises)

Daily-expiry coverage is a **regime**, not a constant:

| Epoch | Expiry calendar | Expected dte≤1 match rate |
|---|---|---|
| ≤ 2010 | monthlies (+ early Friday weeklies from 2005) | ~1 day/month → thin |
| 2010 → 2016 | Friday weeklies established | ~20% of days |
| 2016 → 2022-11 | M/W/F SPXW/SPY | ~60% of days |
| 2022-11 → | daily (Tue/Thu added) — the 0DTE era | ~100% of days |

The full-strength session-VRP series exists **only in the 0DTE era**, which is precisely where
§24's era table puts the fading cadence edge and §26's rot detector fired. The eval must report
the 2022-11+ era separately and first; earlier epochs are M/W/F-conditional subsamples (and
Mon/Wed/Fri days are not exchangeable with Tue/Thu — expiry-day effects exist; keep the
day-of-week split in the readout).

## The session-window decomposition (Tier 1 → §29.1's object)

A dte=1 quote at close t−1 prices [16:00 t−1 → 16:00 t] = overnight + session. Do **not**
convert with a flat 13/48 time fraction — variance is not uniform over the 24h day (that is what
the diurnal baseline exists to say). Two decompositions, both reported:

1. **Model overnight (primary):** implied_session(t) = IV_var_closeclose(t) − E[RV_overnight(t)],
   with E[RV_overnight] forecast by the same embargoed walk machinery on the overnight window
   (rows labeled 16:30 t−1 → 10:00 t exclusive; one more target column in
   `analysis/straddle_horizon.py`, same 679 design). Model-dependent; the dependence is the
   price of EOD data.
2. **Baseline share (secondary, model-free):** implied_session = IV_var_closeclose ×
   (Σ baseline over session bars / Σ baseline over all 48), trailing-year share. Crude but
   assumption-transparent; disagreements between (1) and (2) are themselves a diagnostic.

Units convention, to match v3: OM `impl_volatility` is annualized; close-to-close 1-trading-day
implied variance = iv²/252. HSV = (Σ ask − Σ bid)/2 / Σ vega over the two ATM legs — half-spread
in decimal vol, straddle-level (v3's exact formula).

## Tier 2 — the intraday purchase (spec'd so the quote request is one email)

- **Product:** CBOE DataShop historical intraday option quotes (or LiveVol equivalent), SPX +
  SPXW roots (add SPY if Tier-1 friction says SPY executes better).
- **Slice:** one quote snapshot per day per option, **09:35 ET** (5 min after open — post-auction,
  quotes firm), strikes within ±3% of spot, expiry = same trading day (plus next-day as control).
  A single daily snapshot keeps the purchase small; do not buy full-day NBBO until the edge
  survives the snapshot version.
- **Fields:** timestamp, root, exdate, strike, cp_flag, bid, ask, bid/ask size, underlying.
- **Range:** 2016-01 → latest (before SPXW M/W/F there is nothing to buy); 2022-11+ is the part
  that matters — if budget forces a choice, buy 2022-11+ only.
- **Sizing rule for the purchase decision:** buy iff Tier-1 base-tier net edge (net of HSV, the
  v3 `|edge| > half_spread` rule) has t ≥ 2 and positive mean in the 2022-11+ era.

## Tier 3 — VIX1D (free QC anchor)

Cboe publishes VIX1D daily history (backfilled to 2011). Window is 1 *calendar* day constant
maturity — not the session object — so it is a **cross-check, never the implied leg**:
corr(Tier-1 close-close implied vol, VIX1D at close) should exceed ~0.95 on overlap; a break
localizes an export bug (settlement mapping, expiry selection) faster than any unit test.

## Known-answer gates (run before any research readout; §16 philosophy)

1. **VRP ratio:** mean(implied var)/mean(realized close-close var) at dte=1 within [1.05, 1.6]
   — the prior runs' known-answer band (1.37 at H=48, 1.14 at H=240 on 4–9 dte).
2. **Match rates per epoch** within the coverage table above (±10pts); failures = expiry-calendar
   mapping bug, not market fact.
3. **HSV sanity:** median half-spread at 0–1 dte ≥ the 1–2 dte median from the prior export
   (0.15 vol pts) and below ~5× it; day-of-expiry spreads widen into the close — the 09:35
   snapshot (Tier 2) and the close t−1 quote (Tier 1) must be kept distinct in the readout.
4. **Settlement convergence:** dte=0 straddle mid at 16:00 ≈ intrinsic (PM-settled) — a
   nonzero systematic residual means exdate/settlement mapping is wrong (the `am_settlement`
   filter above is the usual culprit).
5. **Join integrity:** every matched OM `date` t−1 attaches to exactly one panel 10:00-labeled
   row of trading date t; count must equal the epoch match rate times panel days.

## Consumer (to be written when data lands): `analysis/vrp_eod.py`

Joins: `straddle_eod.npz` open-slice forecast (h=13 rows) + Tier-1 implied (decomposed) +
HSV → per-day edge = forecast_session_var − implied_session_var; readouts mirror v3: gross/net
per cost tier, era-split (2022-11+ first), day-of-week split, |edge| > HSV trade rule, plus the
§28.1(b) sizing overlay (vega budget ∝ trailing e² concentration — the amplitude finding's
legitimate use). Every DM/t HAC per the study's conventions; the open-decision series is
one-per-day and non-overlapping, so plain NW at 63 lags suffices.
