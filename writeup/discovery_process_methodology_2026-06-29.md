# How the close regime was discovered — the repeatable recipe (2026-06-29)

Task: comb the notes, reconstruct *exactly* how the intraday close regime was found so cleanly, and
distill a **reusable discovery loop** that generalizes to finding more features. This is the process
distillation behind `intraday_regime_findings_2026-06-26.md` and `mechanism_and_data_to_buy_2026-06-28.md`.

The discovery was not luck — it was a **fixed 6-step loop**, run once on one conditioning axis (`hour`).
The loop is the asset; below it is reconstructed step-by-step (with the real tool / tell / output), then
generalized, then connected to the regime-MoE (which automates its hardest steps).

---

## 1. The loop that found the close regime

| # | Step | Tool / tell | What it produced |
|---|---|---|---|
| 1 | **Get a black box that beats the linear base** | tuned EBM on the residual: **0.12414** vs enet **0.12516** | proof the residual *has* structure worth chasing |
| 2 | **Read the black box's anatomy** | `ebm_interpret.py` — main effects + pairwise | `hour` jumps enet \|coef\| rank **105 → EBM rank 7** (purely nonlinear); **3 of 5 top interactions involve `hour`** ⇒ the signal is *hour-conditioned* |
| 3 | **Name a regime axis, test the interaction OOS** | `regime_study.py` / `har_flip.py`; train/test split, predictor×regime | clock-regime interactions **+0.0095 OOS R²**; vol-state regime **−0.0027 (fails)**; HAR persistence **sign-flips, 5/6 windows** |
| 4 | **Localize sharply** (the decisive plot) | `tests123.py` — `corr(har_ma_5, resid)` **by hour** | **−0.085 at h9 (open)**, **−0.21 at h17 (close/AH)**, +0.05…0.13 mid-session — *sharp, not gradual* ⇒ auction/session-transition, not a smooth U |
| 5 | **Distill to one interpretable feature** | `distill_regime.py` — `HAR × late-day` | recovers **58%** of the tree's edge ⇒ regime real and mostly low-order |
| 6 | **Falsification gauntlet** | diurnal-U control (survives without it); gamma (weak +20%); real-transform-space; FORCE past L1; fill-swap-verify | confounds ruled out; the residual ~38% that doesn't distill = genuine higher-order |

**Why it was clean:** every step is *honest-OOS and falsifiable*. The "discovery" plot (step 4) was only
trusted because step 3 had already shown the interaction generalizes OOS and step 6 killed the obvious
confounds. No plot was believed divorced from a real run (the standing methodology rule).

## 2. The compass — the tree-subsumption law
A gradient-boosted tree / EBM is **invariant to monotone transforms of the current feature row**. So the
*only* thing that can beat a fitted tree is a **functional of history/sequence the row doesn't contain.**
This is the organizing law that makes the loop efficient:
- it tells you **where to look** — at what the tree itself still can't reach (history-dependent regime /
  path / relevance functionals), not at re-encodings of existing columns (those get absorbed);
- it tells you **when you're done** — when the only surviving lever is a history functional whose OOS
  gain is 5th-decimal, the residual is saturated and the next lever is *data*, not features.

Confirmed repeatedly: `har5rank` (⅔ static curvature) → absorbed; implied-vol magnitude beats rank at the
linear base but is *worse through the tree*; the banded-HAR reparam is legible but penalty-entangled. The
**one** thing that beat the tree was a rolling-relative vol innovation (today's vol vs its own recent
same-slot history) — a history functional, exactly as the law predicts.

## 3. Generalizing the loop — sweep other conditioning axes through the same gauntlet
The loop is **axis-agnostic**: step 2 names a candidate conditioning variable; steps 3-6 test it. The close
regime used `hour`. To find *more* features, point the same gauntlet at other axes. Status of the sweep
(from `mechanism_and_data_to_buy_2026-06-28.md` §10 and the persist_* families):

| candidate axis | step-3 OOS test | verdict |
|---|---|---|
| clock / `hour` (open, close/AH) | +0.0095 | **the win** (deployed) |
| vol-state (RV vs diurnal-norm) | −0.0027 | fails (clock dominates) |
| dealer-gamma / `voldemand` modulation | −0.128 vs −0.106 in-region | weak (~20% modulator, secondary) |
| regime dwell / run-length (F3) | null OOS | dies |
| cross-timescale cascade `har_5/har_125 × close` (F2) | in-sample **#2 of 291**, but 87% spanned by close terms | dies OOS (re-expresses existing close-damping) |
| nested calendar / Friday / OPEX | blocked on date-alignment; Friday null | inconclusive / null |

**The falsification gates that every candidate must pass** (this *is* the generalized method):
1. **Beats the tree** — only history/sequence functionals can (the compass). Re-encodings are absorbed.
2. **Survives OOS** — train/test split with predictor×axis interactions, not in-sample importance.
3. **Localizes sharply** — a real mechanism concentrates (open/close), a confound smears.
4. **Distills** — partial collapse to a legible feature; the un-distilled remainder is the higher-order part.
5. **Not spanned** — OLS-R² of the candidate on the features already in the base; >~85% spanned ⇒ dead OOS.
6. **Clip-free / fill-robust / forced-past-L1** — rule out artifacts (a biting clip or a wrong-type fill can *manufacture* a robust-looking signal; an L1-zeroed feature ≠ no signal until forced).

## 4. The honest finding — the *manual* loop is near-exhausted on price-only data
Run faithfully, the gauntlet now **rejects** new price-derived axes: the close state is **low-dimensional,
already saturated, and ~93% noise** (h16-19 in-sample R²≈0.07). The cleanest in-sample axis (the fast/slow
vol cascade) is 87% spanned and dies OOS. Per the compass, this is the *expected* terminal state: the lever
is **information (the auction cross / GEX / OFI feed), not more feature encodings.** The single sharpest
quantification: `cumrv` (one mechanism-targeted engineered feature) is **~27× more valuable per feature**
than the 359 raw exog — direct measurement of the mechanism dominates piling on indirect series.

## 5. The generative extension — the regime-MoE *automates* steps 2-4
Steps 2-4 were done **by hand**: interpret the EBM, *guess* `hour`, plot corr-by-hour. The regime-MoE
soft-tree gate replaces that guess with **gradient descent over the partition across all features at once**:
- the **learned gate hyperplane = steps 2-3 automated** — it discovers the conditioning axis instead of an
  analyst naming it;
- **reading the gate = step 4 automated** — the hyperplane's top weights are the corr-by-hour localization,
  made legible. It either **recovers `hour`** (validates the manual discovery) or finds a partition the clock
  **smeared** (extends it — the one outcome the manual loop, anchored on a single guessed axis, cannot reach).

That readout is the deliverable even if QLIKE doesn't move, and it is *exactly* the architecture to point at
the auction-imbalance data once acquired (gate = regime, experts = the new microstructure state). So the two
tasks are one arc: **Task-1's manual discovery loop, automated and generalized, is Task-2's MoE gate.**

## 6. Checklist — running the loop on a NEW candidate (reusable)
1. Build the candidate as a **causal, bounded, history-dependent** functional (no magic clips/epsilons;
   rank-Gauss / Bowley / 1/√n stabilization).
2. **Inject where it can act** — into the regime EBM via the `regime_extra` conduit (it BYPASSES the global
   enet's fixed-α, which mis-penalizes sparse close-gated columns) and **FORCE past L1**; do *not* judge it
   folded into the global base (it will spuriously "hurt").
3. Score **full-OOS QLIKE + the h16-19-masked QLIKE** (the regime lift), deferred-smeared.
4. Apply the **six gates** (§3). Compute the **spanning R²** before believing any in-sample importance.
5. If it survives all six → distill to a legible term and add it to the base. If it dies → record *which*
   gate killed it (that's the finding). Most will die at gate 2 or 5 — that is the saturation, honestly measured.
