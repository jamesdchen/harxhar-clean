# Next tests: net-negative GEX and 0DTE straddle pricing

**Scope.** Follow-up experiments suggested by the lag-1 net-negative dealer GEX / half-hour 0DTE ATM straddle analysis on `grok/0dte-professor-notes`. Read-only findings so far; this note does not change code or data.

**Data anchors (unchanged):**

- Panel: `results/atm_straddle_0dte_1530/daily_blk2.parquet` — 866 days, 2020-01-03 → 2024-04-30.
- GEX: `results/gex/gex_daily.parquet` from `experiments/gex.py` (OptionMetrics IvyDB EOD SPX). Measure: `gex_level = Σ s_i γ_i OI_i · 100 · S² · 0.01`.
- Regime: full-calendar lag via `experiments/gex_regime_test.py` `lag_regime` (do **not** `shift(1)` on the sparse 866-day index alone). Neg-gamma day ⟺ lagged `gex_level < 0` (**391 / 866**); pos (**475 / 866**). Base rate P(neg) = **45.15%**.
- Half-hour package: absolute log-spot move `|Δlog S|`, entry premium, `iv_var`, premium-normalized return `R`.
- Buy / long-vol days: from the same `daily_blk2` RV–IV notebook output (`notebooks/atm_straddle_rv_iv.ipynb`), **`pos > 0`** → **346 / 866** buys.

**Compiled:** 2026-09-28 (voice follow-ups; expanded with full period-split and overlap numbers).

---

## 0. Thesis / story we were trying to defend

**Claim.** Net-negative dealer gamma days are **mechanically higher-vol** days in the **last ~30 minutes**, because dealers who are short gamma must chase the underlying as spot moves — amplifying realized moves near the close. That mechanical channel should create a **predictable long-vol edge** for a **0DTE ATM straddle** (here: enter ~15:30) via the **variance risk premium** channel: if the market underprices that extra close-window variance on short-gamma days, buying the straddle on those days should earn positive premium-normalized `R`.

**Literature anchors.** Dim–Eraker–Vilkov (0DTE volume explosion and its pricing implications); the mechanical-hedging / last-thirty-minute line (Ni–Pearson–Poteshman–White; Baltussen–Da–Lammers–Martens; Barbon–Buraschi; see `writeup/mechanical-dealer-hedging-channel-literature.md`).

**What the in-repo tests were meant to check.**

1. Do lag-1 net-negative `gex_level` days actually show larger half-hour `|Δlog S|`?
2. Is that extra move **priced** in entry IV (so `R` stays flat), or **underpriced** (long-vol edge)?
3. Are the strategy’s **buy / long-vol days** enriched for neg-gamma (consistent with the VRP / mechanical story selecting the same days)?
4. Did the pattern survive the **0DTE volume explosion** after ~2021, or did pricing and/or the move differential change?

**Headline from the conversation:** (1) yes, moves are ~1.7–1.9× larger on neg-gamma; (2) entry IV is ~2× richer, so `R` does **not** blow out — full-sample long-vol edge on neg-gamma is **not** supported by fatter `R` tails; (3) buys are **not** enriched for neg-gamma (enrichment **0.89**); (4) the premium-to-move match held in **2020–21** (mean `R` on neg **positive**) and **broke** in **2022–24** (move gap compressed, premium gap stayed wide, mean `R` on neg **negative**), coinciding with the in-repo 0DTE gamma-share jump.

---

## 1. Context summary (what we already know)

### 1a. Buy days vs lag-1 neg GEX (headline overlap)

| Cell | Count |
| --- | ---: |
| Buy ∩ neg | **139** |
| Buy ∩ pos | 207 |
| Non-buy ∩ neg | 252 |
| Non-buy ∩ pos | 268 |
| Buys total | **346 / 866** |

- P(neg \| buy) = **40.17%**
- Base P(neg) = **45.15%**
- Enrichment = 40.17 / 45.15 = **0.89** (buys are slightly *under*-represented on neg-gamma, not enriched)

So the long-vol buy calendar and the mechanical short-gamma calendar are **not** the same days. That weakens a simple “strategy is just harvesting neg-gamma VRP” reading of the existing buy rule.

### 1b. Full-sample magnitude and `R` distribution

On lag-1 **net-negative** dealer GEX days vs positive:

| Quantity | Neg-gamma | Pos-gamma | Pattern |
| --- | --- | --- | --- |
| Half-hour `|Δlog S|` | — | — | ~**1.7–1.9×** larger on neg (mean / median / high percentiles) |
| Entry premium | — | — | ~**2×** richer on neg |
| `iv_var` | — | — | ~**4.5×** on neg |
| `R` std | **0.961** | **1.25** | Pos more dispersed |
| `R` kurtosis | **2.71** | **12.2** | Pos has the extreme right tail (max `R` ~**10.32**) |
| Premium-normalized `R` | — | — | **Not** fatter-tailed on neg; bigger spot moves absorbed by richer entry (`|ΔS|/entry` actually *lower* on neg) |

So the raw “neg-gamma → bigger moves” story is real in **levels**, but the straddle’s entry IV already embeds most of that extra move size — `R` does not blow out on neg days; if anything, the **extreme** long-vol wins sit on **pos**-gamma days.

### 1c. Subperiod split: pricing accuracy is **not** stable

`entry/move` = (neg/pos mean entry-premium ratio) ÷ (neg/pos mean `|Δ|` points ratio). Above 1 ⇒ IV markup exceeds the move markup.

#### Main split

**2020–2021** (316 days; **95 neg / 221 pos**):

| Metric | Neg | Pos | Ratio / note |
| --- | ---: | ---: | --- |
| Mean `|Δlog S|` | 5.50e-3 | 1.80e-3 | move ratio **3.05** |
| Median `|Δlog S|` | 2.50e-3 | 1.40e-3 | — |
| Mean entry | 11.63 | 4.74 | entry ratio **2.46** |
| Mean `iv_var` ratio | — | — | ~8.96 |
| Mean `R` | **+0.113** | **+0.002** | — |
| P(`R > 0`) | **45.3%** | **37.1%** | — |
| `|ΔS|` points ratio | — | — | ~2.50 |
| **entry / move** | — | — | **0.98** (premium ≈ matches moves) |
| `|ΔS|/entry` | 1.38 | 1.54 | — |

**2022–2024** (550 days; **296 neg / 254 pos**):

| Metric | Neg | Pos | Ratio / note |
| --- | ---: | ---: | --- |
| Mean `|Δlog S|` | 2.36e-3 | 1.52e-3 | move ratio **1.55** |
| Median `|Δlog S|` | 1.81e-3 | 1.11e-3 | — |
| Mean entry | 8.57 | 4.74 | entry ratio **1.81** |
| Mean `iv_var` ratio | — | — | ~2.99 |
| Mean `R` | **−0.075** | **−0.006** | — |
| P(`R > 0`) | **36.5%** | **38.2%** | — |
| `|ΔS|` points ratio | — | — | ~1.45 |
| **entry / move** | — | — | **1.25** (premium **overshoots** move gap) |
| `|ΔS|/entry` | 1.21 | 1.51 | — |

#### Year by year

| Year | Neg / pos `n` | Move ratio | Entry ratio | entry/move | Mean `R` (neg) | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **2020** | 51 / 107 | **3.00** | **2.83** | **0.94** | **+0.133** | Largest real move differential; P(`R>0`) 43% vs 41% |
| **2021** | 44 / 114 | **1.74** | **1.82** | **1.05** | **+0.090** | Still roughly matched; P(`R>0`) 48% vs 33% |
| **2022** | 169 / 50 | **1.45** | **1.55** | **1.07** | **−0.077** | Few pos-gamma days (`n=50`); mean `R` pos also −0.187 |
| **2023** | 101 / 147 | **1.07** | **1.37** | **1.29** | **−0.126** | Move gap nearly gone; premium gap remains; P(`R>0`) 29% vs 37% |
| **2024** (thin) | 26 / 57 | **1.10** | **1.35** | **1.23** | **+0.138** | Small sample; P(`R>0`) 50% vs 39%; mean `R` pos +0.188 |

**Reading.** Early sample: richer IV on neg-gamma days roughly **justifies** the larger half-hour moves (mean `R` on neg positive; entry/move ≈ 1). Later sample: the **move gap compressed** toward ~1.1–1.5× while the **premium gap stayed** ~1.3–1.8× — relative to subsequent moves, neg-gamma IV looks **rich**, and mean `R` on neg days flipped negative. That undercuts the simple “buy straddle on neg-gamma for VRP” thesis in the 0DTE-heavy regime.

### 1d. 0DTE volume-explosion coincidence (Dim–Eraker–Vilkov)

In-repo proxy (**not** exchange volume): share of days with nonzero `gex_0dte`, and mean `|gex_0dte|`.

| Year | P(`gex_0dte` ≠ 0) | Mean `|gex_0dte|` |
| --- | ---: | ---: |
| 2020 | **8.9%** | **1.4e8** |
| 2021 | **8.2%** | **2.5e8** |
| 2022 | **62%** | **8.8e8** |
| 2023 | **79%** | **2.1e9** |
| 2024 | **80%** | **2.9e9** |

The 2021→2022 jump lines up with (a) compression of the neg/pos half-hour move ratio from ~3 toward ~1.1 and (b) entry/move rising from ~0.94–1.05 into the 1.23–1.29 range in 2023–24. As 0DTE came to dominate the gamma surface, the “absorbed by richer IV” story flipped from **matched** pricing toward **over-pricing** relative to realized half-hour moves — and mean `R` on neg-gamma days went from positive (2020–21) to negative (2022–23).

**Caveats:** 2024 has only 26 neg days; 2022 has only 50 pos days; `gex_0dte` is EOD OptionMetrics 0-DTE gamma exposure, not OPRA volume; half-hour package only; no causal identification.

---

## 2. Test 1 — Jump-risk compensation

**Question.** Do the rare huge-move neg-gamma days **justify** the average IV cushion (so that left-tail / jump losses on long premium, or right-tail wins, make up for the later-sample overpricing), or do the tails **fail** to cover the average overpricing?

**Motivation.** Full-sample `R` is not fatter on neg days (pos has kurtosis 12.2 and max ~10.32); 2022–24 mean `R` on neg is negative. That could still be rational if the **conditional** distribution has rare jumps that long-vol cares about and those jumps cluster on neg-gamma days. Alternatively, the premium markup may simply be too large relative to both body and tails — and the extreme `R` wins may sit on the wrong regime.

**Design sketch (read-only on existing `daily_blk2` + lag regime):**

1. Within neg vs pos, report **quantiles** of `|Δlog S|` and of `R` (e.g. p50 / p90 / p95 / p99 / max), plus mean of top-`k` days.
2. **Contribution decomposition:** share of total sum of positive `R` (and of negative `R`) coming from days with `|Δlog S|` above a high threshold (e.g. p95 of the pooled sample, or a fixed move size).
3. **Premium vs ex-post move on jump days only:** on the top-`k` `|Δ|` days in each regime, is entry premium still rich / cheap relative to realized `|ΔS|`? Report mean `R` and mean `|ΔS|/entry` on those days alone.
4. **Subperiod interaction:** repeat (1)–(3) for 2020–21 vs 2022–24. Hypothesis: early sample, jump days help explain matched pricing; late sample, even jump days may not rescue mean `R` if the body is overpriced and jumps are less differential.
5. Optional: exceedance regression — `R` on an indicator for “large move” × regime, with and without year FE.

**Pass / interesting outcomes:**

- **Justifies cushion:** top-tail neg days have substantially higher mean `R` (or cover a large share of cumulative positive `R`), and that effect is stable or stronger post-2022.
- **Does not cover overpricing:** late-sample mean `R` on neg stays negative even after restricting to, or overweighting, jump days; or neg/pos tail `R` gap is small while entry remains elevated (consistent with pos already owning the extreme right tail of `R`).

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
3. **Contingency:** P(live_neg | EOD_neg), P(EOD_neg | live_neg), disagreement rate — full sample and by year (expect more disagreement or more near-zero live G as 0DTE share rises). Also: P(live_neg | buy) vs P(EOD_neg | buy).
4. Redo the magnitude / pricing tables (mean/median `|Δlog S|`, entry, `iv_var`, mean `R`, P(`R>0`), move ratio, entry ratio, entry/move) under **live** regime instead of EOD lag-1.
5. Split by agreement: EOD_neg ∩ live_neg vs EOD_neg ∩ live_pos (stale short-gamma, live long) — which cell carries the large-move / rich-IV pattern?

**Pass / interesting outcomes:**

- Live short-gamma **sharpens** the move differential and/or restores matched entry/move in 2022–24 → stale EOD was the wrong clock for 15:30.
- Live and EOD agree and late-sample overpricing remains → the drift is not an artifact of stale sign.
- Disagreement concentrates in high-`gex_0dte` years → 0DTE makes EOD lag less informative for the close window.
- Live neg **does** enrich buys even though EOD neg does not → strategy may be tracking something closer to same-day gamma.

**Constraints:** prior-chain only (no same-day OI/IV refresh); no OPRA tick GEX in-repo — do not invent it. Prefer extending / calling the existing live helper over a one-off reimplementation.

---

## 4. Test 3 — Extend the sample through 2025–26

**Question.** Did the post-2022 pattern (compressed move gap, persistent premium gap, negative mean `R` on neg-gamma, high 0DTE gamma share, buy/neg enrichment < 1) **continue**, reverse, or stabilize as 0DTE kept growing after April 2024?

**Motivation.** `daily_blk2` ends **2024-04-30**. In-sample 2024 is thin (26 neg days). Dim–Eraker–Vilkov-style volume growth and the in-repo `gex_0dte` share were still rising into 2024; out-of-sample years are the cleanest check that the regime shift is durable rather than a 2022–23 artifact.

**Design sketch:**

1. Rebuild (or extend) the half-hour 0DTE ATM straddle package and OptionMetrics EOD GEX through the latest available date (target: into **2025–26**), same definitions as `daily_blk2` / `gex.py` / `lag_regime`.
2. Replicate the full-sample magnitude table, buy∩neg contingency, `R` moments (std / kurtosis / max), and the **year-by-year** and **2020–21 vs 2022+** splits, adding 2024 (full), 2025, 2026YTD.
3. Track the 0DTE proxy: P(`gex_0dte` ≠ 0) and mean `|gex_0dte|` by year — does it plateau or keep rising, and does entry/move stay elevated?
4. Optional: rolling 252-day windows of (move ratio, entry ratio, entry/move, mean `R` | neg, enrichment P(neg|buy)/P(neg)) to see whether the break is a step in 2022 or a gradual drift.

**Pass / interesting outcomes:**

- **Continuation:** 2025–26 look like 2023 (move ratio ~1.1, entry/move ≳ 1.2, mean `R` on neg ≤ 0, enrichment ≲ 1) → overpricing relative to moves is the new steady state under 0DTE-heavy gamma; the original long-vol-on-neg-GEX thesis stays weak.
- **Mean reversion:** move ratio and/or entry/move return toward ~1 and mean `R` on neg recovers → mid-sample was transitional.
- **Breakdown of GEX signal:** neg/pos move differential disappears entirely while premium differential remains (or vice versa) → revisit Test 2 (live gamma) and measurement.

**Dependencies:** OptionMetrics / IvyDB refresh and the same 15:30 construction pipeline used for `daily_blk2`; keep file layout compatible (`results/gex/gex_daily.parquet`, `results/atm_straddle_0dte_1530/…`).

---

## 5. Suggested order and shared hygiene

1. **Test 1** first — pure post-processing on existing parquet; cheapest falsification of “jumps justify the IV,” and reconciles with pos-gamma owning the extreme `R` right tail.
2. **Test 2** next — needs live `G(S)` at 15:30 but reuses `structure_gex_live` / prior-chain convention; clarifies whether EOD lag is the right label for the window we price (and for the buy overlap).
3. **Test 3** last — data refresh; then re-run Tests 1–2 on the extended sample.

**Shared rules (from the conversation):**

- Always lag with full-calendar `lag_regime`, not sparse-index `shift(1)`.
- Report both raw move ratios and entry/move (premium ratio ÷ move ratio).
- Flag thin cells (`n` neg or pos ≲ 50).
- Do not treat `gex_0dte` as exchange volume; say so when citing Dim–Eraker–Vilkov coincidence.
- No claims of causal identification from these descriptive splits alone.
- Buy days = `pos > 0` on `daily_blk2`, not an unrelated `position_sign` strategy column.

---

## 6. Pointers

| Artifact | Path / note |
| --- | --- |
| Half-hour straddle panel | `results/atm_straddle_0dte_1530/daily_blk2.parquet` |
| Daily GEX | `results/gex/gex_daily.parquet` (`experiments/gex.py`) |
| Official lag helper | `experiments/gex_regime_test.py` → `lag_regime` |
| Live prior-chain gamma | `experiments/structure_gex_live.py` |
| RV–IV / buy definition | `notebooks/atm_straddle_rv_iv.ipynb` (`pos > 0`) |
| Related lit note | `writeup/mechanical-dealer-hedging-channel-literature.md` |
