# The linear penalty/estimator is a non-lever — five angles (2026-07-01)

**Question.** The deployed residual base hardcodes an elastic-net penalty on the exog block
(`alpha=1e-3, l1_ratio=0.2`) with the 6 HAR mains held OLS. Was that penalty ever *properly* tuned —
ridge vs lasso vs enet, strength swept — with HAR unpenalized? **No.** The only prior comparison
(`do_dimreduce_cell`) penalized HAR, used unstandardized inputs, and had no pure Lasso; the deployed
`alpha/l1` were carried over and frozen. This note closes that gap from five angles. All agree: **the
linear estimator's penalty/shrinkage is not a lever — it is at the floor. The lever is information.**

**Scope / caveat.** All runs use the raw slim base (529 feats; 6 HAR mains OLS via FWL; exog penalized),
*without* the deployed `enetreg2` add-ons (HAR×regime interactions + cumrv×close). Absolute QLIKE therefore
sits ~0.1253, *below* the deployed `enetreg2`/`linbest` 0.12314 — **the penalty *ranking* is the
deliverable, not the absolute number.** Incumbent = enet `a=1e-3, l1=0.2` (raw slim). Small incumbent
differences across tables (0.12534 / 0.12540 / 0.12546) are refit-cadence only; each table is
internally consistent.

## 1. Penalty type × strength grid, HAR held OLS (`enet_penalty_sweep.py`)
Slice-tune → full-OOS confirm. Full-OOS (tw=48000, refit=12000):

| config | full-OOS QLIKE |
|---|---|
| incumbent enet a=1e-3, l1=0.2 | **0.12546** |
| low-α probe enet a=3e-4, l1=0.5 | 0.12532 (best, −0.00014) |
| slice-winner enet a=3e-3, l1=0.5 | 0.12644 |
| best lasso a=1e-3 | 0.12629 |
| **best ridge (pure L2) a=100** | 0.12590 |
| ridge a=0.03 (low α) | 0.31 (blows up) |

- **Pure Ridge is the structural loser** — at low α it blows up (near-OLS on 523 collinear exog);
  even at heavy shrinkage (α=100) it loses to the incumbent enet.
- **Slice-tuning does not transfer**: every slice-tuned winner loses to the fixed incumbent on full-OOS.
- **Reverses the naive prediction, and sharpens it.** Dense-not-low-rank was read as "ridge should win";
  in fact the signal is **sparse-in-basis but not low-rank** — ~400 of 529 columns are prunable noise,
  which L1 zeros and L2 cannot. Use L1 in the raw basis; never PCA.

## 2. Optuna full-OOS oracle ceiling (`enet_penalty_optuna.py`) — DIAGNOSTIC ONLY
Objective *is* full-OOS QLIKE (test-set overfit by design) → the upper bound on what tuning could *ever*
buy. tw=48000, refit=20000; incumbent ref 0.12540.

| penalty | oracle best | params | vs incumbent |
|---|---|---|---|
| ridge | 0.12578 | α=266 | **+0.00038 (loses even as an oracle)** |
| lasso | 0.12524 | α=1.5e-4 | −0.00015 (α at search floor) |
| enet | 0.12529 | α=3.4e-4, l1=0.33 | −0.00011 |

Even cheating on the test set, the **ceiling is −0.00015** (sub-floor noise), ridge can't buy anything,
and every winner sits at **α→0 (minimal shrinkage)**. A proper CV would capture a fraction of −0.00015 —
i.e. nothing. **Do not report these as OOS performance; they are a ceiling.**

## 3. Online shrinkage (prior work, `enetreg2_olam` / `_cadence_enet_online`)
Leakage-clean per-block online μ-selection (embargoed forward split). Base-alone: **0.12324 ≈ fixed-α
0.12314 (+0.00010) — REJECTED, ties.** Adapting the penalty online adds nothing over a fixed α.

## 4. Coefficient moving-average (`coef_ma_test.py`) — the one conceptually-new idea
Fit the base once; causally smooth per-block coefficients (trailing MA over K, or EWMA λ) before predicting.

| scheme | full-OOS Δ vs no-smoothing |
|---|---|
| MA K=2 / 3 / 5 / 8 / 13 | +0.00006 / +0.00008 / +0.00011 / +0.00017 / +0.00028 |
| EWMA λ=0.3 / 0.5 / 0.7 / 0.9 | +0.00001 / +0.00001 / +0.00003 / +0.00036 |

**DEAD — monotone hurt.** No sweet spot: any weight on past blocks costs, growing with strength. This is
a *positive* corroboration — if the coefficient drift were noise, small-K averaging would help; it never
does, so **the drift is real signal** (matching the explainability audit's 0DTE/OPEX gamma-dilution:
fast HAR×close decays, slow strengthens). Smoothing blurs a real time-varying effect → pure bias.

## 5. What L1 actually kills, and when (`lasso_support.py`)
Support of the oracle lasso (a=1.5e-4) across walk-forward blocks: kills 304/523 exog cols.

- **WHAT: 235 of the 304 kills (77%) are availability indicators** — the known dead/constant missing-data
  flags (only 11 of 246 carry signal). Informative families (moments, liquidity, implied_vol, sentiment,
  market moments) are mostly **kept**; the 6 calendar/clock features are **always kept** (the regime anchor).
  **This is the mechanism behind §1's "ridge loses": ridge can't zero those 235 dead indicators, so they
  inject variance and it blows up; L1 zeros them cleanly.**
- **WHEN: the support is 87% flicker** — 190 of 219 ever-kept features drop and re-enter block-to-block;
  only 29 are a stable core (calendar + a few moments/liquidity/IV). The −0.00015 oracle "win" rides on
  that regime-dependent flicker → fragile, would not survive CV.
- Lasso ≈ incumbent enet: support differs by only ~19 of ~220 cols — the same base, lighter shrinkage.

## Verdict
| angle | result |
|---|---|
| penalty type (grid) | enet-incumbent near-optimal; **ridge loses** (dead availability indicators) |
| penalty tuning (slice→OOS) | doesn't transfer |
| penalty tuning (oracle ceiling) | ≤ −0.00015 even test-set-overfit; ridge +0.00038 |
| online shrinkage | ties fixed-α — rejected |
| coefficient-MA | dead, monotone hurt (coef drift is real signal) |

**The linear base's penalty/shrinkage is settled: a non-lever.** Ridge fails for a concrete, legible
reason (dead availability indicators it cannot zero); L1's marginal edge is unstable, regime-dependent
flicker. Consistent with the project floor from every angle — the lever is information, not the estimator.

## Reproduce
Code: `_cadence_enet_har_unpen` gained `penalty`∈{enet,lasso,ridge} + `max_iter`/`tol` params (backward-
compatible; default reproduces the deployed base). Drivers + outputs:
`enet_penalty_sweep.py` → `results/enet_penalty_sweep.{json,log}`;
`enet_penalty_optuna.py` → `results/enet_penalty_optuna.{json,log,db}`;
`coef_ma_test.py` → `results/coef_ma_test.{json,log}`;
`lasso_support.py` → `results/lasso_support.log`.
Run with `/c/Users/james/miniconda3/envs/285J/python.exe`.
