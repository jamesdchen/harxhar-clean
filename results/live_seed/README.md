# `premium_ledger.parquet` — the live engine's premium ledger, seeded from the research tape

`live.ibkr.premium_ledger.PremiumLedger` reads and writes this file. It is the
only state the live entry-clock selector and the corrected hedge delta carry
between sessions: one row per `(session, clock)`, 15,348 rows over
**1,279 sessions, 2020-01-03 .. 2025-12-31**, at the twelve 30-minute clocks
`10:00 … 15:30`.

## Provenance

Built once from `writeup/intraday_proposals/43_causal_entry_over_time.py`
(imported by path; its `main()` is not run), which is the forecast-free
entry-clock study of the 2026-09-18 audit. 43's builders were used unchanged —
re-deriving them here would have made this a second construction rather than a
seed.

| what | where it comes from |
| --- | --- |
| sessions | every expiration in `data/spxw_chain.parquet` **less the 12 half sessions** whose 15:30 row has already expired (`hours_to_expiration <= 0`); the chain's own `early_close` column is degenerate and is never read |
| stamps, spot | `data/spxw_spot.parquet`, pivoted to (session, `HH:MM`) in naive Eastern, bar-END labelled as the panel is |
| straddle | `atm_straddle_lib.pick_nearest_otm_guarded` re-picked at **every** stamp from one pyarrow-filtered read of `data/spxw_chain.parquet` — the call is the smallest live strike at or above spot, the put the largest at or below, with the research's outage guards (`bid == ask == 0` is no quote; 4 stamps of 15,348 refused) |
| implied volatility | bisected out of that stamp's own package midpoint by `live.ibkr.pricing.invert_total_vol`; 43 gates its vectorised inversion against that scalar function on all 15,348 cells (max abs difference 0.0, 5 cells NaN on both sides) |
| settlement | `results/atm_straddle_intraday_holdclose/cache/gspc_ohlc.parquet` close |

Rebuild it — and re-run all three gates below — with

```
python -m live.ibkr.parity --seed
```

(`live.ibkr.parity.seed_premium_ledger`, about 20 s). The ledger itself depends
on nothing but the chain, the spot tape and the settlement tape: no forecast
panel, no model, no trade cache. The third gate reads further, because it checks
the estimator against proposal 36's own grids.

## The columns, exactly

| column | definition |
| --- | --- |
| `session` | the expiration date, tz-naive midnight |
| `clock` | the Eastern stamp, `"10:00" … "15:30"` |
| `implied_rem_var` | **total** variance over the REMAINING session: `total_vol ** 2`, where `total_vol = sigma * sqrt(T)` is bisected out of the nearest-OTM straddle's package midpoint at that stamp (Black-76, `r = 0`, forward taken as spot — the vendor's own convention) |
| `realized_rem_var` | the variance actually realized over that same remaining window: `sum of (log(S_next / S)) ** 2` over the 30-minute steps from the stamp through the 15:30 stamp, **plus the last step from the 15:30 spot into the settlement print**. NaN unless every step of the window is present — a partial window is not a smaller window |
| `spot` | the index at that stamp (the chain's `underlying_price`, which agrees with the spot file) |
| `premium_mid` | the straddle's package midpoint at that stamp, in index points |
| `kc`, `kp` | the nearest-OTM call and put strikes at that stamp |

`implied_rem_var` and `realized_rem_var` are variances over the **same** window,
so their ratio is dimensionless and directly comparable across clocks and years.

## What the ledger computes, and the masking rule

A cell is **usable** only when its implied and its realized remaining variance
are both finite and strictly positive. Both are then masked together, so every
sum below runs over exactly the same sessions — proposal 46's shared mask.
Both estimators read sessions **strictly before** `asof`: a record for `asof`
itself never enters its own statistic, and both refuse (NaN) below
`min_sessions = 63`, the repo's warm-up. Nothing is filled, clipped or floored.

* `ratio_of_sums(clock, asof, window)` — the summed implied over the summed
  realized remaining variance, per clock, over the `window` sessions strictly
  before `asof` (`None` expands). A **ratio of sums**, never a mean of per-day
  ratios: proposal 44 showed a mean of logs is monotone in the clock (a one-bar
  remaining window makes `log(implied/realized)` explode) and picks 15:00 on
  every session.
* `realized_over_implied_mean(clock, asof, window)` — the lagged mean of
  realized over implied at that clock; proposal 36's V9 bias correction, which
  `live.ibkr.pricing.corrected_total_vol` applies to the implied **variance**
  before the delta is taken.

## Gates this seed passed

| gate | result |
| --- | --- |
| `ratio_of_sums` per clock over each of the 10 calendar periods reproduces proposal 43's `a_premium_curve` `implied_over_realised` | 110 cells, max abs deviation **4.441e-16** (bar 1e-9) |
| the migration itself | 10:00 **1.165730 (2020) → 0.856771 (2025)**; 11:00 **1.043430 → 0.767877**; 14:00 **0.870278 → 1.102734** |
| `selector.pick_entry_clock(window=252, min_sessions=63)` reproduces proposal 46's E2 rolling-252 pick sequence | **0 disagreements on 1,279 sessions**, 63 warm-up sessions flat |
| the same at `window=126` against 46's rolling-126 sequence | **0 disagreements on 1,279 sessions** |
| `realized_over_implied_mean` reproduces proposal 36's V9 factor (`expanding_lagged_mean(realized / var0, 63)`) on the 865 deck days x 10 book stamps, fed 36's own implied and realized grids | 8,020 live cells, max abs deviation **4.441e-16** (bar 1e-9); era median variance factor **0.9047 at 11:00 … 0.8047 at 15:30** (volatility factor 0.951 … 0.897) |

The V9 gate is run on proposal 36's own grids because 36's implied slice is the
**vendor** hourly volatility squared times the hours remaining and its realized
window is the forecast panel's bar variances — a different tape from this
ledger's, which is mid-inverted and built on the 30-minute spot steps. The gate
therefore proves the **estimator** is 36's, to floating point. On this ledger's
own tape the same factor is larger (era median 0.9207 at 11:00, 1.0351 at
15:30) because the two tapes measure different things; that difference is a
property of the inputs, not of the estimator.

## Reading it

The premium walked from the morning to the afternoon over 2020-2025. That is
what the first gate's three rows say, and it is why the live engine no longer
hard-codes an 11:00 entry. The selector that follows the migration is a
**defensible causal choice, not a demonstrated improvement**: proposal 46 found
it does not beat fixed 11:00 with an interval excluding zero, and does not beat
fixed 13:30. Nothing in this ledger is costed — charge the 0.5 bp hedge cost
before quoting any Sharpe built on it.
