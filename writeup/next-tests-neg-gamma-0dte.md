# Next tests: net-negative GEX and 0DTE straddle pricing

**Scope.** Follow-up experiments suggested by the lag-1 net-negative dealer GEX / half-hour 0DTE ATM straddle analysis on `grok/0dte-professor-notes`. Read-only findings so far; this note does not change code or data.

**Data anchors (unchanged):**

- Panel: `results/atm_straddle_0dte_1530/daily_blk2.parquet` — 866 days, 2020-01-03 → 2024-04-30.
- GEX: `results/gex/gex_daily.parquet` from `experiments/gex.py` (OptionMetrics IvyDB EOD SPX). Measure: `gex_level = Σ s_i γ_i OI_i · 100 · S² · 0.01`.
- Regime: full-calendar lag via `experiments/gex_regime_test.py` `lag_regime` (do **not** `shift(1)` on the sparse 866-day index alone). Neg-gamma day ⟺ lagged `gex_level < 0` (391 / 866); pos (475 / 866).
- Half-hour package: absolute log-spot move `|Δlog S|`, entry premium, `iv_var`, premium-normalized return `R`.

**Compiled:** 2026-09-28 (voice follow-ups).

---

## 1. Context summary (what we already know)

### Full-sample magnitude pattern

On lag-1 **net-negative** dealer GEX days vs positive:

| Quantity | Pattern |
| --- | --- |
| Half-hour `|Δlog S|` | ~**1.7–1.9×** larger on neg (across mean / median / high percentiles) |
| Entry premium | ~**2×** richer on neg |
| `iv_var` | ~**4.5×** on neg |
| Premium-normalized `R` | **Not** fatter-tailed on neg (pos has higher kurtosis / extremes); bigger spot moves are largely absorbed by the richer entry IV (`|ΔS|/entry` is actually *lower* on neg) |

So the raw “neg-gamma → bigger moves” story is real in levels, but the straddle’s entry IV already embeds most of that extra move size — `R` does not blow out.

### Subperiod split: pricing accuracy is **not** stable

Suggested splits: 2020–2021 vs 2022–2024; plus calendar years when `n` allows.

| Window | Neg / pos `n` | Move ratio (neg/pos mean `|Δ|` pts) | Entry-premium ratio | Entry / move | Mean `R` (neg) | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **2020–21** | 95 / 221 | ~**2.50** | ~**2.46** | ~**0.98** | **+0.11** | Roughly one-for-one premium-to-move pricing |
| **2022–24** | 296 / 254 | ~**1.45** | ~**1.81** | ~**1.25** | **−0.075** | Premium markup exceeds move markup |
| 2020 | 51 / 107 | ~3.00 | ~2.83 | ~0.94 | +0.13 | Largest real move differential |
| 2021 | 44 / 114 | ~1.74 | ~1.82 | ~1.05 | +0.09 | Still roughly matched |
| 2022 | 169 / 50 | ~1.45 | ~1.55 | ~1.07 | −0.08 | Few pos-gamma days |
| 2023 | 101 / 147 | ~**1.07** | ~**1.37** | ~**1.29** | −0.13 | Move gap nearly gone; premium gap remains |
| 2024 (thin) | 26 / 57 | ~1.10 | ~1.35 | ~1.23 | +0.14 | Small sample |

**Reading.** Early sample: richer IV on neg-gamma days roughly **justifies** the larger half-hour moves (mean `R` on neg positive). Later sample: the **move gap compressed** toward ~1.1–1.5× while the **premium gap stayed** ~1.3–1.8× — so relative to subsequent moves, neg-gamma IV looks **rich**, and mean `R` on neg days flipped negative.

### Coincidence with 0DTE growth (Dim–Eraker–Vilkov volume-explosion)

In-repo proxy (not exchange volume): share of days with nonzero `gex_0dte`, and mean `|gex_0dte|`.

| Year | P(`gex_0dte` ≠ 0) | Mean `|gex_0dte|` |
| --- | ---: | ---: |
| 2020 | ~9% | ~1.4e8 |
| 2021 | ~8% | ~2.5e8 |
| 2022 | ~**62%** | ~8.8e8 |
| 2023 | ~**79%** | ~2.1e9 |
| 2024 | ~**80%** | ~2.9e9 |

The 2021→2022 jump lines up with (a) compression of the neg/pos half-hour move ratio and (b) entry/move rising into the 1.23–1.29 range in 2023–24. As 0DTE came to dominate the gamma surface, the “absorbed by richer IV” story flipped from **matched** pricing toward **over-pricing** relative to realized half-hour moves.

**Caveats already noted:** 2024 has only 26 neg days; 2022 has only 50 pos days; `gex_0dte` is EOD OptionMetrics 0-DTE gamma exposure, not OPRA volume; analysis is half-hour package only; no causal ID.

---

## 2. Test 1 — Jump-risk compensation

**Question.** Do the rare huge-move neg-gamma days **justify** the average IV cushion (so that left-tail / jump losses on long premium, or right-tail wins, make up for the later-sample overpricing), or do the tails **fail** to cover the average overpricing?

**Motivation.** Full-sample `R` is not fatter on neg days, and 2022–24 mean `R` on neg is negative. That could still be rational if the **conditional** distribution has rare jumps that long-vol (or short-vol) cares about, and those jumps cluster on neg-gamma days. Alternatively, the premium markup may simply be too large relative to both body and tails.

**Design sketch (read-only on existing `daily_blk2` + lag regime):**

1. Within neg vs pos, report **quantiles** of `|Δlog S|` and of `R` (e.g. p50 / p90 / p95 / p99 / max), plus mean of top-`k` days.
2. **Contribution decomposition:** share of total sum of positive `R` (and of negative `R`) coming from days with `|Δlog S|` above a high threshold (e.g. p95 of the pooled sample, or a fixed move size).
3. **Premium vs ex-post move on jump days only:** on the top-`k` `|Δ|` days in each regime, is entry premium still rich / cheap relative to realized `|ΔS|`? Report mean `R` and mean `|ΔS|/entry` on those days alone.
4. **Subperiod interaction:** repeat (1)–(3) for 2020–21 vs 2022–24. Hypothesis from context: early sample, jump days help explain the matched pricing; late sample, even jump days may not rescue mean `R` if the body is overpriced and jumps are less differential.
5. Optional: exceedance regression — `R` on an indicator for “large move” × regime, with and without year FE.

**Pass / interesting outcomes:**

- **Justifies cushion:** top-tail neg days have substantially higher mean `R` (or cover a large share of cumulative positive `R`), and that effect is stable or stronger post-2022.
- **Does not cover overpricing:** late-sample mean `R` on neg stays negative even after restricting to, or overweighting, jump days; or neg/pos tail `R` gap is small while entry remains elevated.

**Do not:** modify `daily_blk2` or GEX files; keep lag definition as `lag_regime`.

---

## 3. Test 2 — Same-day intraday gamma (live chain at spot)

**Question.** Is the stale EOD lag-1 `gex_level` the right conditioning variable for the **15:30** half-hour window, or does **live** net gamma at the window’s spot change the overlap / pricing story?

**Motivation.** Forced-hedging flow depends on where spot sits in the gamma **profile now**. Lag-1 EOD sign can disagree with same-day intraday gamma, especially as 0DTE mass concentrates near the money. Repo already has the causal machinery:

- `experiments/structure_gex_live.py` — re-prices the **prior session’s** OptionMetrics chain at live spot (SPY→SPX daily ratio), arms `live_shortG` / `live_longG` / `live_deepshortG` vs stale `eod_shortG` / `eod_longG`.
- Related: `gex_bar_test` broadcasts lagged daily regime; panel OPTIONS_FEATURES include `opt_gex_*` but they are **not** in the frozen ALL_FEATURES panel.

**Design sketch:**

1. For each `daily_blk2` date, evaluate live net gamma at (or just before) the 15:30 entry using the prior EOD chain at contemporaneous spot — same causal convention as `structure_gex_live.py` (`net_gamma(S, d)` with prior-session chain).
2. Define regimes: `live_neg` (`G(S)<0`), optional `live_deep_neg` (bottom tercile of `G(S)`), vs lag-1 EOD neg.
3. **Contingency:** P(live_neg | EOD_neg), P(EOD_neg | live_neg), disagreement rate — full sample and by year (expect more disagreement or more near-zero live G as 0DTE share rises).
4. Redo the magnitude / pricing tables (mean/median `|Δlog S|`, entry, `iv_var`, mean `R`, P(`R>0`), move ratio, entry ratio, entry/move) under **live** regime instead of EOD lag-1.
5. Split by agreement: EOD_neg ∩ live_neg vs EOD_neg ∩ live_pos (stale short-gamma, live long) — which cell carries the large-move / rich-IV pattern?

**Pass / interesting outcomes:**

- Live short-gamma **sharpens** the move differential and/or restores matched entry/move in 2022–24 → stale EOD was the wrong clock for 15:30.
- Live and EOD agree and late-sample overpricing remains → the drift is not an artifact of stale sign.
- Disagreement concentrates in high-`gex_0dte` years → 0DTE makes EOD lag less informative for the close window.

**Constraints:** prior-chain only (no same-day OI/IV refresh); no OPRA tick GEX in-repo — do not invent it. Prefer extending / calling the existing live helper over a one-off reimplementation.

---

## 4. Test 3 — Extend the sample through 2025–26

**Question.** Did the post-2022 pattern (compressed move gap, persistent premium gap, negative mean `R` on neg-gamma, high 0DTE gamma share) **continue**, reverse, or stabilize as 0DTE kept growing after April 2024?

**Motivation.** `daily_blk2` ends **2024-04-30**. In-sample 2024 is thin (26 neg days). Dim–Eraker–Vilkov-style volume growth and the in-repo `gex_0dte` share were still rising into 2024; out-of-sample years are the cleanest check that the regime shift is durable rather than a 2022–23 artifact.

**Design sketch:**

1. Rebuild (or extend) the half-hour 0DTE ATM straddle package and OptionMetrics EOD GEX through the latest available date (target: into **2025–26**), same definitions as `daily_blk2` / `gex.py` / `lag_regime`.
2. Replicate the full-sample magnitude table and the **year-by-year** and **2020–21 vs 2022+** splits, adding 2024 (full), 2025, 2026YTD.
3. Track the 0DTE proxy: P(`gex_0dte` ≠ 0) and mean `|gex_0dte|` by year — does it plateau or keep rising, and does entry/move stay elevated?
4. Optional: rolling 252-day windows of (move ratio, entry ratio, entry/move, mean `R` | neg) to see whether the break is a step in 2022 or a gradual drift.

**Pass / interesting outcomes:**

- **Continuation:** 2025–26 look like 2023 (move ratio ~1.1, entry/move ≳ 1.2, mean `R` on neg ≤ 0) → overpricing relative to moves is the new steady state under 0DTE-heavy gamma.
- **Mean reversion:** move ratio and/or entry/move return toward ~1 and mean `R` on neg recovers → mid-sample was transitional.
- **Breakdown of GEX signal:** neg/pos move differential disappears entirely while premium differential remains (or vice versa) → revisit Test 2 (live gamma) and measurement.

**Dependencies:** OptionMetrics / IvyDB refresh and the same 15:30 construction pipeline used for `daily_blk2`; keep file layout compatible (`results/gex/gex_daily.parquet`, `results/atm_straddle_0dte_1530/…`).

---

## 5. Suggested order and shared hygiene

1. **Test 1** first — pure post-processing on existing parquet; cheapest falsification of “jumps justify the IV.”
2. **Test 2** next — needs live `G(S)` at 15:30 but reuses `structure_gex_live` / prior-chain convention; clarifies whether EOD lag is the right label for the window we price.
3. **Test 3** last — data refresh; then re-run Tests 1–2 on the extended sample.

**Shared rules (from the conversation):**

- Always lag with full-calendar `lag_regime`, not sparse-index `shift(1)`.
- Report both raw move ratios and entry/move (premium ratio ÷ move ratio).
- Flag thin cells (`n` neg or pos ≲ 50).
- Do not treat `gex_0dte` as exchange volume; say so when citing Dim–Eraker–Vilkov coincidence.
- No claims of causal identification from these descriptive splits alone.

---

## 6. Pointers

| Artifact | Path / note |
| --- | --- |
| Half-hour straddle panel | `results/atm_straddle_0dte_1530/daily_blk2.parquet` |
| Daily GEX | `results/gex/gex_daily.parquet` (`experiments/gex.py`) |
| Official lag helper | `experiments/gex_regime_test.py` → `lag_regime` |
| Live prior-chain gamma | `experiments/structure_gex_live.py` |
| Related lit note | `writeup/mechanical-dealer-hedging-channel-literature.md` (PR into professor-notes) |
