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

## 6. Per-bucket diagnostic table (`per_bucket_table.py`)
Best **OUT-OF-SAMPLE-tuned** QLIKE per (bucket, penalty) — each cell is the min over that penalty's full
α(/l1) grid (oracle ceiling; test-set-tuned, DIAGNOSTIC ONLY). HAR held OLS; fixed tw=48000 for cross-row
comparability. **HAR-only reference = 0.12998.**

| bucket (n_exog) | lasso | ridge | enet | best vs HAR |
|---|---|---|---|---|
| no-bucket (HAR only, 0) | 0.12998 | 0.12998 | 0.12998 | — |
| moments (84) | 0.12808 | 0.12819 | **0.12807** | −0.00191 |
| implied_vol (36) | 0.12914 | **0.12887** | 0.12897 | −0.00111 |
| liquidity (144) | 0.12885 | 0.12892 | **0.12884** | −0.00114 |
| market_vw (72) | **0.12915** | 0.12920 | 0.12915 | −0.00083 |
| market_ew (72) | **0.12919** | 0.12927 | 0.12920 | −0.00079 |
| sentiment (36) | **0.12933** | 0.12941 | 0.12933 | −0.00065 |
| vol_demand (72) | 0.12974 | 0.12993 | **0.12975** | −0.00024 |
| **all_buckets (523)** | 0.12553 | 0.12587 | **0.12527** | **−0.00472** |

- **Information ranking (rows):** moments (−0.00191) ≫ liquidity / implied_vol (~−0.0011) > market_vw/ew
  (~−0.0008) > sentiment (−0.00065) > vol_demand (−0.00024). Every bucket beats HAR; **all_buckets −0.00472
  is complementary but sublinear** (< the sum of singles → the buckets overlap).
- **Penalty (columns) is a non-lever:** within every row lasso/ridge/enet agree to ≤0.0003. Best configs all
  want minimal shrinkage (lasso α at the 3e-4 grid floor; enet α 3e-4–1e-3, l1 0.2–0.5).
- **The one structured exception — penalty choice tracks the bucket's junk fraction.** Ridge is worst where
  there are dead/collinear columns it can only shrink, not zero (all_buckets: ridge +0.0006 vs enet, the
  largest gap); but **ridge WINS on implied_vol** (0.12887, dense all-informative VIX levels, ~no junk →
  dense L2 shrinkage is ideal and L1 zeroing discards correlated signal). Direct corroboration of §1/§5:
  L1's advantage = zeroing junk; where there is none, L2 wins.
- **Consistency:** the all_buckets row reconciles with the §2 Optuna oracle (ridge 0.12587≈0.12578,
  lasso 0.12553≈0.12524, enet 0.12527≈0.12529; small deltas = grid coarseness).
- **Window footnote:** HAR-only is 0.12998 at tw=48000 (1000d, all_buckets-optimal); at shorter windows it
  is **0.13299 (250d) / 0.13403 (125d)** — the classic ~0.134 HAR baseline. The absolute level is a train-
  window effect; the *ranking/deltas* are the deliverable and are window-robust.

## Reproduce
Code: `_cadence_enet_har_unpen` gained `penalty`∈{enet,lasso,ridge} + `max_iter`/`tol` params (backward-
compatible; default reproduces the deployed base). Drivers + outputs:
`enet_penalty_sweep.py` → `results/enet_penalty_sweep.{json,log}`;
`enet_penalty_optuna.py` → `results/enet_penalty_optuna.{json,log,db}`;
`coef_ma_test.py` → `results/coef_ma_test.{json,log}`;
`lasso_support.py` → `results/lasso_support.log`;
`per_bucket_table.py` → `results/per_bucket_table.{json,log}` (the §6 table).
Run with `/c/Users/james/miniconda3/envs/285J/python.exe`.
