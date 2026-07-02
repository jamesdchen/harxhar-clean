# PROP-TRADING EXPLORATION — HANDOFF (2026-07-02)  ⟨PARKED — NOT the research⟩

**This is a self-contained side-quest, fully parked. A fresh session should IGNORE it unless the user
explicitly says "resume the prop / directional / breakout work."** The 4pm-meeting research is a
SEPARATE bucket — see the bottom of this file.

## Where it lives (so it's cleanly separable)
- **Branch**: `prop-exploration-2026-07-02` (a label at HEAD); also committed on `edge-features-legibility`
  as commits `5881635 … 7e0e013` (all prefixed `data(vrp)/data(gex)/data(dir)/feat(dir)/feat(gex)`).
- **Memory**: `vrp-doors-verdict-2026-07-02.md` (the full arc; retraction included).
- **Scripts (all repo-root)**: `signed_channel_worker.py`, `signed_tree_worker.py`, `door3_worker.py`,
  `flow_dir_pilot.py`, `gex.py`, `gex_regime_test.py`, `gex_bar_test.py`, `gex_rv_test.py`, `gex_vrp.py`,
  `gex_flow_dir.py`, `gex_reversion.py`, `prep_om_chain.py`, `fairvalue_test.py`, `opening_reversal_test.py`,
  `jj_backtest.py`, `structure_setup.py`, `structure_gex.py`, `structure_gex_live.py`, `breakout_runner.py`,
  `breakout_maker.py`, `breakout_boost.py`, `breakout_dig.py`, `breakout_wf.py`.
- **Results**: `results/{signed_channel,door3,flow_dir_pilot,gex,gex_regime,gex_bar,gex_rv,gex_vrp,
  gex_flow_dir,gex_reversion,fairvalue,opening_reversal}/`.
- **Data (gitignored)**: `data/optionm_*.parquet` (OptionMetrics SPX chain/spot, regenerate via
  `prep_om_chain.py`); free QQQ/SPY intraday lived in the scratchpad (not committed).

## The question and the honest verdict
**Q: where/how to trade the variance (VRP) edge on futures prop firms, and is there a directional edge?**

1. **Futures rail closed BY MEASUREMENT.** VRP needs an instrument embedding *implied* variance; linear
   futures don't (spanning wall). And the signed-return channel is null — linear (`5881635`) AND nonlinear
   (`ae2218f`): our feature set has zero first-moment content. Two independent kills.
2. **Door-3 (trade the eval firm's payout convexity):** priced (`0dd4757`) — P(pass) is barrier geometry +
   sizing, *forecast-independent* (perfect σ̂ = naive). Not a deployment of the edge.
3. **Venue reframe:** the edge's real home is a **pro futures brokerage (member-rate) on CFE vol products
   (VX/VXM/VA), capitalized by first-loss capital (Topwater lead)** — binding constraint = a **live audited
   track record** (we have paper only). Deep-research `wv2jmmn5w` (synthesis errored on session limit; re-run).
4. **GEX (dealer gamma) from OptionMetrics** (`bf08c32`): Fork-B directional = NULL (gamma predicts VARIANCE
   not DIRECTION; bar autocorrelation doesn't flip with regime, `656d440`); Fork-A = GEX adds +0.022 OOS R²
   to a HAR+VIX RV forecast (real) but does NOT improve the tradeable VRP P(pass) (QLIKE↔P&L inversion,
   `16e3930`).
5. **JJ-Simon "fair value" / breakout-continuation** (`a2617a7 … 7e0e013`): the directional thread. The
   MECHANICS insights are real (maker break-retest earns the spread; structural stop; a fixed target beats
   run-to-close; charge transaction costs; volume-confirmation concept). **BUT the EDGE is UNVALIDATED:**
   the exciting numbers (maker flip, target=3R, +volume/+trend P(pass) 0.9+) were all **in-sample
   selection**. **End-to-end WALK-FORWARD** (`breakout_wf.py`, `7e0e013`) on 60d of a QQQ-5m *proxy*:
   adaptive config-selection = **net-negative** OOS; a pre-committed config = **~flat** (Sharpe +0.011).
   No demonstrable OOS edge. 60 days overfits everything.

**Bottom line: NO validated directional edge (measured six ways). The variance edge is real but needs a
vol-permitting venue + a live track record.**

## If ever resumed (do NOT before this)
- Data: **1-min NQ/ES over YEARS** (institutional; prof setting up access). NDX-options GEX for NQ.
- Method: build the eval **walk-forward from the first line** (like `feature_cv.py`); config committed
  *before* seeing the test. Reusable: `breakout_wf.py` (WF harness), `gex.py` (GEX), `eval_sim.simulate_topstep`
  (P(pass)), the maker/target/cost mechanics.

---

## ⟨SEPARATE BUCKET⟩ — THE ACTUAL RESEARCH (4pm-meeting focus) — UNTOUCHED this session
The **volatility-forecasting research** (HAR/EBM, feature investigation, the intraday-regime program) was
**not modified by the prop side-quest**. It lives in `src/`, `notebooks/results/edge_features/`,
`ebm_*.py`, `writeup/` (non-prop), and the research memories (`evaluation-suite-and-every-bar`,
`intraday-regime-findings`, `ebm-campaign-*`, `clean-feature-pipeline-and-fwl`, etc.). Pre-existing
uncommitted research WIP (README, `src/*`, `ebm_shapes.py`, `writeup/session_notes_*`, notebook deletions)
was already in the working tree at session start and is **unrelated to the prop work**. Start the meeting
prep fresh there; nothing above competes with it.
