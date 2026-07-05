# OptionMetrics export spec — friction pricing of the cumrv×close edge

**Purpose (the datum-to-buy):** decide whether the honest vol edge (Sharpe 0.271/day,
frictionless, `eval_sim/data/cumrv_close_pnl_honest.npy`) survives real option execution costs.
Decision rule: net Sharpe ≥ ~0.1/day at base costs → options eval-prop venue live
(Imperial-class); below → the venue class is moot. Consumer: `price_option_friction.py`.

## WRDS web query (same flow as the GEX pull that produced `prep_om_chain.py` inputs)

Two exports, **SPY and XSP** (run as two queries or one with both secids):

### 1. Option prices — OptionMetrics → "Option Prices" (`opprcd`)
- **secid:** SPY = `109820`; XSP — look up via Security file (`securd`, ticker XSP, index flag).
  Add SPX (`108105`) only if XSP coverage is thin pre-2010.
- **Date range:** 2004-01-01 → latest available (panel is ~5,915 trading days ≈ 2000s→2023;
  the overlap window is what prices).
- **Fields:** `secid, date, exdate, cp_flag, strike_price, best_bid, best_offer, volume,
  open_interest, impl_volatility, delta, vega, gamma, theta`
- **Filters (apply in-query to keep the export small):**
  - days to expiry 1–10 (we trade DTE 1–5; keep 10 for roll flexibility)
  - `impl_volatility` non-null
  - optional: 0.20 < |delta| < 0.80 (ATM neighborhood; straddle legs are |delta|≈0.5)
- **Format:** csv.gz, one file per underlying.

### 2. Underlying prices — OptionMetrics → "Security Prices" (`secprd`)
- Same secids/date range; fields: `secid, date, close, return`.

Expected size: DTE≤10 + delta band keeps SPY to ~a few hundred MB gz. Land files in
`data/om_friction/{spy,xsp}_opprcd.csv.gz`, `{spy,xsp}_secprd.csv.gz`.

## Step 0 before the join — date axis for the signal

The panel has **no calendar dates** (`build_cumrv_pnl.py` uses a synthetic hour-wrap `day_id`).
Extend `build_cumrv_pnl.py` (prop branch) to also emit `eval_sim/data/cumrv_signal_dated.parquet`
with columns `date, signal` (the causal expanding-mean-adjusted cumrv at the last pre-close bar,
i.e. the value known at 16:00). The date axis comes from the raw loader (`src/data/loading.py`
knows the source timestamps); wire it through the same causal path — do NOT re-derive the signal
with new machinery (rank-1 rule: identical verified machinery).

Use the **honest** signal variant (causal fitted HAR+VIX/VVIX baseline, bd22177 lineage), since
0.271 is the number the decision threshold is calibrated to.

## Cost model tiers (implemented in the script)

| Tier | Option fills | Rationale |
|---|---|---|
| optimistic | mid | lower bound |
| base | mid ∓ 25% of quoted spread each way | effective/quoted ≈ 0.5 (penny-pilot SPY lit.) |
| conservative | sell at bid / buy at ask | full quoted half-spread |

Plus: residual delta hedged in underlying at entry and exit, 1 bp each way; $0.65/contract
commission (prop sims typically pass through ~retail rates — confirm with venue later).

## Addendum 2026-07-04: VIX-options export (the last unclosed residual)

Tests the ONE untested channel: expressing the signed overnight ΔVIX prediction (NW-t +6.59,
gross 0.376 pts/traded-day) via listed VIX index options — the only VIX-derivative an
ETNA-class equities/options venue could carry. Expectation: negative (futures-beta dilution
~0.5 + vol-of-vol premium + ~0.05/leg spreads), but it closes by measurement not arithmetic.

1. **secid lookup:** OptionMetrics Security File (`securd`), ticker `VIX`, index_flag=1
   (likely 102456 — verify in the form).
2. **Option prices (`opprcd`):** that secid, 2006-02-24 (VIX options listing) → latest,
   same fields as before, **DTE ≤ 40** (monthlies pre-2015; weeklies after).
3. **Security prices (`secprd`):** same secid (VIX index level).
4. Land as `data/om_friction/vix_opprcd.csv.gz`, `vix_secprd.csv.gz`.

Driver: `vix_option_friction.py` — DIRECTIONAL (not straddle): signal>0 → long ATM puts,
signal<0 → long ATM calls, sized w/|delta|, enter close t exit close t+1 at real mids ± tier
spreads. Note VIX options price the FORWARD (VX future) — the forward's overnight move is
~0.4–0.5 of spot ΔVIX; the test measures that dilution implicitly since quotes are real.

## Readouts

1. **Gross option-space Sharpe** (mid fills) — tests RV−HAR ⇒ RV−implied transfer
   (edge measured orthogonal-to-implied, so expected to survive; this is the proof).
2. Net Sharpe per tier per underlying; breakeven spread-fraction.
3. Post-2016 subsample (0DTE-era = live regime).
4. If base-tier net Sharpe ≥ 0.1/day: save net daily PnL as
   `eval_sim/data/cumrv_option_net_pnl.npy` (rescaled to $400/day std) and re-run
   `imperial_pass_calc.py` with it → real chain P.
