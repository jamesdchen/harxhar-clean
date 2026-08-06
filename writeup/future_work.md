# harxhar — Future Work Tracker

This is the durable tracker for known-but-deferred work across harxhar modules. Each entry has a stable ID; module docstrings cross-reference IDs (e.g., `"See writeup/future_work.md#STRAT-03"`) so the canonical description lives in one place. Deleting an entry should be followed by a grep for its ID to find and clean up stale references in code. New modules append entries with their own ID prefix (e.g., `EVAL-01`, `LOAD-01`) — no restructuring required.

## Summary

| ID | Module | Subject | Blocker |
|---|---|---|---|
| [STRAT-01](#strat-01--surface-dynamics--iv-evolution) | `src/strategy_eval.py` | Surface dynamics / IV evolution | Real chain data + research time on the surface model |
| [STRAT-02](#strat-02--real-option-chain-ingestion) | `src/strategy_eval.py` | Real option-chain ingestion | Data subscription / source decision |
| [STRAT-03](#strat-03--absolute-spy-price-level) | `src/strategy_eval.py` | Absolute SPY price level | None — implementable today |
| [STRAT-04](#strat-04--delta-hedge-rebalancing-frequency) | `src/strategy_eval.py` | Delta-hedge rebalancing frequency | Unblocked — purely implementation |
| [STRAT-05](#strat-05--continuously-rebalanced-multi-period-strategies) | `src/strategy_eval.py` | Continuously-rebalanced multi-period strategies | Unblocked but research-heavy |
| [STRAT-06](#strat-06--strike-policies-beyond-atm-at-open) | `src/strategy_eval.py` | Strike policies beyond ATM-at-open | Unblocked, additive |
| [STRAT-07](#strat-07--transaction-cost-calibration) | `src/strategy_eval.py` | Transaction cost calibration | Depends on `STRAT-02` |
| [STRAT-08](#strat-08--per-horizon-qlike-vs-strategy-sharpe-consistency-study) | `src/strategy_eval.py` | Per-horizon QLIKE-vs-strategy-Sharpe consistency study | Depends on `STRAT-01` / `STRAT-02` (real data) |
| [AM-01](#am-01--panel-refresh-through-the-present) | `analysis/*` (alpha-manifestation study) | Panel refresh through the present (2024-03 → now is unseen OOS) | Current raw bar data |
| [AM-02](#am-02--land-the-options-chain--run-tier-1-vrp) | `analysis/vrp_eod.py` | Land the options chain + run Tier-1 VRP | Data transfer (chain parquets exist elsewhere) |
| [AM-03](#am-03--qlike-to-straddle-pnl-translation) | `analysis/vrp_eod.py`, `src/evaluation/strategy.py` | QLIKE → straddle-P&L translation (real-data STRAT-08) | AM-02 |
| [AM-04](#am-04--gamma-path-vs-integrated-variance-wedge) | `src/evaluation/strategy.py` | Gamma-path vs integrated-variance wedge | Unblocked (scaffold exists) |
| [AM-05](#am-05--tail-treatment-and-distributional-forecast-head) | `analysis/straddle_horizon.py` | Tail/winsorization audit at horizon + quantile forecast head | Unblocked — runnable today |
| [AM-06](#am-06--announcement-organ-port) | `analysis/straddle_horizon.py` | Announcement-organ port (msweep's settled lever) | `data/releases.parquet` absent in this environment |
| [AM-07](#am-07--per-horizon-shrinkage-from-the-msweep-alpha-law) | `analysis/straddle_horizon.py` | Per-horizon shrinkage imported from the msweep α-law | Unblocked — runnable today |
| [AM-08](#am-08--cross-market-replication) | study-wide | Cross-market replication of dense-weak + the meta-law | A second market's bar data |
| [AM-09](#am-09--session-conditional-operator) | `analysis/straddle_horizon.py` | Session-conditional (per-clock) remaining-session model | §29.2 verdict first |
| [AM-10](#am-10--multiplicity-sweep-over-the-s27-32-family) | `analysis/multiplicity.py` | Romano–Wolf sweep over the §27–§32 claim family | All §27–32 verdicts in |
| [AM-11](#am-11--finer-resolution-intraday-flow) | `analysis/cucuringu.py` | Intraday flow at 1–5 min bars (sub-bar dynamics behind the −0.28 jitter) | Finer-resolution bar data |
| [AM-12](#am-12--cross-sectional-panel-cucuringu-as-estimator) | new module | Cross-sectional panel: lead-lag portfolios, cross-impact OFI, SPONGE sectors | Cross-section of names |
| [AM-13](#am-13--battery-rerun-on-the-fixed-panel) | battery campaign | Causal-tune battery rerun on the FIXED panel (+ tree parity rematch on the 699 set) | Cluster time |
| [AM-14](#am-14--transmission-as-execution-timing) | trading layer | Transmission arrows as straddle ENTRY timing (flow leads implied repricing) | Tier-2 intraday option quotes |

---

### STRAT-01 — Surface dynamics / IV evolution

- **Module:** `src/strategy_eval.py`
- **Symptom / why it matters:** The scaffold freezes IV at session open and ignores intraday level changes, term structure, and skew. Once strategies extend beyond same-day ATM (multi-day holds, OTM strikes, calendar spreads, skew trades), the frozen-ATM-IV assumption breaks.
- **Concrete change required:** `IVProvider` gains methods like `get_iv_surface(t, strikes, tenors)` and `get_atm_iv_intraday(t)`; `compute_delta_hedged_atm_straddle_pnl` uses bar-level IV; new strategies plug in. Likely first concrete model: a Heston-style stochastic vol or an SVI-fit static surface refitted nightly — discussion deferred until real chain data is in hand.
- **Blocker:** Real chain data + research time on the surface model.
- **Where flagged in code:** `IVProvider` protocol docstring (first paragraph), `compute_delta_hedged_atm_straddle_pnl` "Simplifying assumptions" block (item i), module docstring.

### STRAT-02 — Real option-chain ingestion

- **Module:** `src/strategy_eval.py`
- **Symptom / why it matters:** `OptionChainProvider` is a `NotImplementedError` stub today. The scaffold cannot produce strategy P&L against real implied vol until a chain provider lands.
- **Concrete change required:** Implement `OptionChainProvider` against one of the candidate sources: OptionMetrics IvyDB US, CBOE DataShop, ORATS. Honors the `IVProvider` protocol — date → ATM mid IV at session open.
- **Blocker:** Data subscription / source decision.
- **Where flagged in code:** `OptionChainProvider` stub docstring lists the candidate sources; module docstring.

### STRAT-03 — Absolute SPY price level

- **Module:** `src/strategy_eval.py`
- **Symptom / why it matters:** `_reconstruct_underlying_prices` builds the underlying via `S = S0 × exp(cumsum(sumret))` with `S0 = 100` because no actual price column lives in the repo data. Output JSON tags this with `"underlying_source": "reconstructed_from_sumret_S0_100"`. Sharpe, hit rate, t-stat, and any ratio-based metric are **scale-invariant** and unaffected. Raw dollar P&L magnitudes are in normalized units — a "$1 P&L" reported by the scaffold is `$1 / (real_S0 / 100)` of real-world dollar P&L. Anyone reading dollar magnitudes without remembering this will misinterpret them.
- **Concrete failure mode:** A stakeholder asks "what's the dollar P&L of this strategy on $10M notional" — answer is wrong by a factor of `real_S0 / 100`.
- **Concrete change required:** Wire a real $-denominated SPY series (yfinance, vendor feed, or a corrections file) into a new `underlying_source='real'` mode of `_reconstruct_underlying_prices`. Use real opens at session open as `S0` for that day; subsequent bars still rebuild via `cumsum(sumret)` to stay consistent with the rest of the data. No other strategy code changes — `K`, `Γ$`, and per-bar P&L all just inherit the new scale.
- **Blocker:** None, beyond "actually getting a SPY level series." Implementable today.
- **Where flagged in code:** `_reconstruct_underlying_prices` docstring (first paragraph), output-JSON warning string, and notebook banner.

### STRAT-04 — Delta-hedge rebalancing frequency

- **Module:** `src/strategy_eval.py`
- **Symptom / why it matters:** The current P&L computation implicitly assumes a delta hedge is rebalanced *every* 30-min bar. Each bar's contribution `(1/2) × Γ$ × (RV − IV²)Δt` is the discrete-time approximation of the continuous-hedge integral. Real options traders rebalance on event-based or cost-aware cadences (e.g., when delta drifts > X, or every N hours), not every bar. The scaffold's reported P&L corresponds to a frictionless 30-min-rebalance regime. Real strategies with sparser hedging will produce noisier P&L (un-hedged delta exposure between rebalances injects directional return into the eval); strategies with hedging too dense will pay too much in costs.
- **Concrete failure mode:** Comparing the scaffold's P&L to a real trader's monthly P&L and concluding the strategy is uneconomic when the real difference is hedging-cadence specification, not signal quality.
- **Concrete change required:** Add a `hedge_freq: int | 'event_band'` parameter to `compute_delta_hedged_atm_straddle_pnl`.
  - `hedge_freq=1` is the current behavior.
  - `hedge_freq=N` aggregates `N` bars between rebalances — un-hedged delta-driven P&L (`Δ × ΔS`) accumulates between rebalances and is added to the gamma term.
  - `'event_band'` triggers rebalances when `|Δ_current − Δ_at_last_rebalance| > threshold`.
  - Transaction costs are charged per rebalance (not per bar), so `cost_bps` becomes meaningful.
- **Blocker:** Unblocked — purely implementation; needs a clean derivation written into the docstring so the math is auditable.
- **Where flagged in code:** `compute_delta_hedged_atm_straddle_pnl` docstring (in the "Simplifying assumptions" block as item iv after the existing three), and the bar-level `bar_pnl_df` returned by the function carries a `hedge_freq` column reading 1 today so the assumption is in any saved diagnostic.

### STRAT-05 — Continuously-rebalanced multi-period strategies

- **Module:** `src/strategy_eval.py`
- **Symptom / why it matters:** The filter already produces an intraday path of forecasts; today only the `i=0` row drives an open-time decision. A follow-up uses the full path for intraday rebalancing.
- **Concrete change required:** Add a position-sizing rule, intraday cost model, and re-evaluation of the look-ahead lag. Consume the full `path_df` rather than only the session-open row.
- **Blocker:** Unblocked but research-heavy.
- **Where flagged in code:** `filter_intraday_estimate` docstring; module docstring.

### STRAT-06 — Strike policies beyond ATM-at-open

- **Module:** `src/strategy_eval.py`
- **Symptom / why it matters:** Strike policy is `ATM-at-open, frozen for the day` (`K = S(b_0(D))`). Other policies (rolling-ATM (re-strike each bar), fixed-strike (rolling expiries), vol-targeted notional) are all deferred.
- **Concrete change required:** Extend the `strike_policy` parameter on `compute_delta_hedged_atm_straddle_pnl` to handle `'rolling_atm'`, `'fixed_strike'`, `'vol_targeted'`. Each policy plugs in additively without disturbing the base ATM-at-open path.
- **Blocker:** Unblocked, additive.
- **Where flagged in code:** `compute_delta_hedged_atm_straddle_pnl` docstring (`strike_policy` parameter); module docstring.

### STRAT-07 — Transaction cost calibration

- **Module:** `src/strategy_eval.py`
- **Symptom / why it matters:** `cost_bps` defaults to 0; without real chain bid-ask values, transaction costs cannot be calibrated to a meaningful number.
- **Concrete change required:** Once chain data is available (via `STRAT-02`), derive realistic `cost_bps` from bid-ask spreads at the strikes/tenors used by each strategy.
- **Blocker:** Depends on `STRAT-02`.
- **Where flagged in code:** `compute_delta_hedged_atm_straddle_pnl` docstring (`cost_bps` parameter).

### STRAT-08 — Per-horizon QLIKE-vs-strategy-Sharpe consistency study

- **Module:** `src/strategy_eval.py`
- **Symptom / why it matters:** Once real numbers exist, sanity-check that QLIKE-better models tend toward higher Sharpe on the strategy eval. Not a unit test — a research item.
- **Concrete change required:** Run the strategy eval across all executors at multiple horizons against real IV data; rank by QLIKE and by strategy Sharpe; report rank correlation and any regimes where the two diverge.
- **Blocker:** Depends on `STRAT-01` / `STRAT-02` (real data) before the study is meaningful.
- **Where flagged in code:** Module docstring of `src/strategy_eval.py`.


---

### AM-01 — Panel refresh through the present

- **Module:** the whole alpha-manifestation study (`analysis/*`, `writeup/alpha_manifestation_findings_2026-08-04.md`).
- **Symptom / why it matters:** The panel ends 2024-03; as of 2026-08 there are ~2.5 years of genuinely unseen data. Every §22–§32 claim (the per-bar composite, the era-GROWTH of the edge, the §28.2 phase sliver, the §26 rot-detector firing, the §32 re-draw) is validated only by internal splits. The rot detector fired AT the panel edge — whether the map recovered (2018 pattern) or broke permanently is the single most decision-relevant unknown and is unanswerable on this panel.
- **Concrete change required:** Rebuild the panel through the present with the existing loaders; re-run the §22 final model, the rot-detector trace, the era table, and the §28.2/§32 gates on the 2024-03+ span as true OOS. No new methodology — this is pure adjudication.
- **Blocker:** Current raw bar data (exists in the user's environment, not in this container).

### AM-02 — Land the options chain + run Tier-1 VRP

- **Module:** `analysis/vrp_eod.py` (consumer, built and waiting), `writeup/om_0dte_atopen_export_spec.md` (spec).
- **Symptom / why it matters:** Every trade-shaped conclusion is QLIKE-proxy. The measured VRP, the `|edge| > half-spread` trade rule, and the Tier-2 purchase decision all wait on two files.
- **Concrete change required:** Land `data/om_friction/chain_109820_dte0_*.parquet` + `data/optionm_spx_spot.parquet`; run `--stage gates` then `--stage vrp`. The overnight-decomposition cache (`straddle_overnight.npz`) is already computed.
- **Blocker:** Data transfer — the prior dte0_10 landing already contains the 0–1 dte rows.

### AM-03 — QLIKE-to-straddle-PnL translation

- **Module:** `analysis/vrp_eod.py`, `src/evaluation/strategy.py`.
- **Symptom / why it matters:** DM units do not answer "is the edge monetizable." STRAT-08 posed this; real implied (AM-02) makes it answerable: rank models by QLIKE and by net straddle Sharpe, report where the rankings diverge (a small DM edge can be un-monetizable after half-spreads; conversely the veto-overlay can pay more than its DM suggests).
- **Blocker:** AM-02.

### AM-04 — Gamma-path vs integrated-variance wedge

- **Module:** `src/evaluation/strategy.py` (the delta-hedged straddle eval already computes path-dependent P&L).
- **Symptom / why it matters:** A delta-hedged straddle earns dollar-gamma-weighted variance along the price path; our targets are flat integrated variance. On trending days the wedge is systematic. Unmeasured for the midday-entry hold-to-close shape (§29.1's recommended trade).
- **Concrete change required:** On the synthetic-IV scaffold (real IV when AM-02 lands), compare variance-swap P&L vs path-dependent straddle P&L for the same signals; quantify the wedge by regime and day-of-week.
- **Blocker:** None — scaffold exists today.

### AM-05 — Tail treatment and a distributional forecast head

- **Module:** `analysis/straddle_horizon.py`, `src/features/transforms/target.py`.
- **Symptom / why it matters:** Targets are built from winsorized bar RV — defensible for regression, but a straddle payoff is convex in exactly the clipped tail. Unaudited at horizon. Deeper: the study forecasts the MEAN of RV; a straddle buyer prices the DISTRIBUTION (right tail especially). A quantile/distributional head is the most payoff-aligned modeling idea not yet tried on this panel.
- **Concrete change required:** (a) audit: recompute §29 ladder QLIKE on unwinsorized reconstruction, report divergence; (b) pilot: quantile regression (same 679 design, pinball loss at τ = 0.75/0.9/0.95) on y_H at H = 8, gate on out-of-half pinball skill vs a scaled-mean baseline.
- **Blocker:** None — runnable today.

### AM-06 — Announcement-organ port

- **Module:** `analysis/straddle_horizon.py` (feature block), source machinery in `drivers/msweep_2026-08-01/straddle_v3.py` lines 54–77.
- **Symptom / why it matters:** The msweep session's settled lever: window-aware announcement features worth −0.005…−0.0067 QLIKE at H = 4–16 — larger than most §29-era increments — base-independent, and absent from our 543 columns.
- **Concrete change required:** Port the 4-per-release-type organ (since/until/count-in-window/bars-until-first) onto the panel; one pre-registered arm at H = 8.
- **Blocker:** `data/releases.parquet` absent in this environment.

### AM-07 — Per-horizon shrinkage from the msweep α-law

- **Module:** `analysis/straddle_horizon.py`.
- **Symptom / why it matters:** The msweep law: α scales with horizon AND capacity (their 1077-col basis wanted 1e4 at H ≤ 8, 1e5 at H = 16). Our ladder used §22's h=1 penalties unretuned, so the composite's H = 8/16 numbers are floors.
- **Concrete change required:** Import the law as a PRIOR (no bake-off): one pre-registered α per horizon scaled from their measurement to our 679-col capacity; single arm per horizon on the fast engine; gate DM ≥ +2.0 vs the unretuned twin.
- **Blocker:** None — runnable today; cheapest known upside on the ladder.

### AM-08 — Cross-market replication

- **Module:** study-wide.
- **Symptom / why it matters:** Every law here (dense-but-weak, the meta-law's 12+ casualties, the 35-day cycle, the intraday flow) is n = 1 market. One replication on a second series would upgrade or kill the program's generality claims more than any further instrument on this panel.
- **Concrete change required:** Rebuild the panel machinery on a second market (NQ, rates vol, or a liquid single name); re-run §22 + §26 + §30 core objects only.
- **Blocker:** A second market's bar data.

### AM-09 — Session-conditional operator

- **Module:** `analysis/straddle_horizon.py`.
- **Symptom / why it matters:** One set of weights serves 10:00, 15:30, and 03:00; time-of-day enters only as features. The msweep session found session-conditional kernels; §29.2 (in flight) adjudicates whether the open-decision failure is clock-conditioning.
- **Concrete change required:** If §29.2 implicates the morning information set: per-slot remaining-session models (13 small ridges, fast engine), pre-registered against the pooled model.
- **Blocker:** §29.2 verdict first.

### AM-10 — Multiplicity sweep over the §27–§32 family

- **Module:** `analysis/multiplicity.py`.
- **Symptom / why it matters:** The §24 sweep covered the model's spine; since then a new claim family accumulated (§28.2 sliver +2.09, §29 ladder, §30 objects, §32 re-draw). Marginal survivors must clear FWER before anything ships, or be labeled provisional — the smear claim died exactly this way.
- **Concrete change required:** Romano–Wolf step-down over the family's loss differentials once all verdicts are in.
- **Blocker:** All §27–32 verdicts in (nearly there).

### AM-11 — Finer-resolution intraday flow

- **Module:** `analysis/cucuringu.py`.
- **Symptom / why it matters:** The intraday phase increments anticorrelate at −0.28 per 30-min bar — the signature of coherent dynamics (or microstructure noise) BELOW bar resolution. The flow at 1–5 min lags is unexplored and is exactly the frequency Cucuringu's lead-lag program targets.
- **Concrete change required:** Rebuild factor scores at finer bars (or on raw quotes), re-run `--stage intraflow` + the kill-tests at 1–5 min lags.
- **Blocker:** Finer-resolution bar data.

### AM-12 — Cross-sectional panel: Cucuringu as estimator

- **Module:** new module (the study's designated "next unit of progress").
- **Symptom / why it matters:** On one series his methods are diagnostics; on thousands of names they are estimators with literature-verified economics: lead-lag portfolios (laggards traded on leaders), cross-impact of OFI (Cont–Cucuringu–Zhang — the OFI features already exist), SPONGE sector clustering for relative-vol trades, per-sector flow networks whose differences are tradable.
- **Concrete change required:** Cross-sectional bar panel of names; port the frame/map/flow machinery per sector; pre-register the lead-lag portfolio as the first scored arm.
- **Blocker:** Cross-section of names (bar-level).


### AM-13 — Battery rerun on the fixed panel

- **Module:** the causal-tune battery campaign (`writeup/causal_tune_battery_table.tex`).
- **Symptom / why it matters:** the battery (buckets × estimators, causal-tuned, per-bar tree
  refits) ran on the PRE-§16 panel; every alpha-manifestation number is on the fixed panel
  (HAR anchors 0.1330 vs 0.13275). No table may juxtapose them until the battery is re-run on
  the fixed panel — the third instance of the paper's panel-unification blocker.
- **Concrete change required:** re-run the battery on the fixed panel (cluster). Fold in the
  TREE PARITY REMATCH as its headline row: per-bar LightGBM/XGB on the full 699-column final
  feature set (products + transmission as inputs), hyperparameters causal-tuned as before,
  gate DM ≥ +2.0 vs the 699 per-bar ridge. Prior (from §19/msweep saturation-at-pairwise):
  trees lose; a pass would be a major result.
- **Blocker:** cluster time (pairs naturally with the §34.7 LSTM run).


### AM-14 — Transmission as execution timing

- **Module:** trading layer (post-VRP).
- **Symptom / why it matters:** the intraday flow says order-flow factors lead the VIX complex
  by ~1 bar (§30, +0.99 stable). For the midday hold-to-close straddle, that is an ENTRY-timing
  edge: buy before the implied repricing the arrows predict, not after. Not a QLIKE object —
  an implied-vs-forecast timing spread, invisible to everything measured on this panel.
- **Concrete change required:** with Tier-2 intraday quotes (AM-02's purchase), test whether
  flow-conditioned entry (enter when transmission predicts VIX-complex catch-up) beats
  clock-fixed midday entry, net of half-spreads.
- **Blocker:** Tier-2 intraday option quotes.
