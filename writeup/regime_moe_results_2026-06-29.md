# Regime-MoE & temporal-kNN on the d8 residual — live results (2026-06-29)

DL/kNN attempt at the Hero-B regime slot on `linbest` + d8@cs0.5 global. Bar = EBM regime **0.12033**.
Updated as the chain lands. Notebooks: `notebooks/results/edge_features/edge_04_regime_moe.ipynb`,
`edge_05_temporal_knn.ipynb` (read `results/moe_ladder/{summary,knn_d8}.csv`).

## Anchor — PASSED
`ctrl_ebm` (EBM regime on the identical d8@cs0.5 global) = **0.12033** full-OOS, h16-19 0.15736, n=194934.
Reproduces the established ceiling exactly → the harness path (cache, global stage, collect, smearing) is
sound and every MoE/kNN row below is directly comparable.

## Regime-MoE ladder (in flight)
| rung | full-OOS QLIKE | h16-19 | vs EBM 0.12033 | read |
|---|---|---|---|---|
| ctrl_ebm (anchor) | **0.12033** | 0.15736 | — | EBM ceiling reproduced |
| moe_d0 (depth-0 NAM) | **0.15200** | 0.33329 | **+0.0317** | badly loses |
| moe_d1 (depth-1 gate) | *running* | | | |
| moe_d2 (depth-2 gate) | *pending* | | | |
| moe_d1diag (+probe) | *pending* | | | |

**Rung 0 fails hard, and informatively.** A from-scratch per-block NAM is **0.152** — worse than doing
nothing (cs0.5 global, no regime stage, 0.12081) and worse than the plain enet base (0.12516). The damage
is localized to the regime rows: h16-19 QLIKE **0.333 vs 0.157** for the EBM. The NAM's residual
predictions are badly miscalibrated for *raw-space QLIKE* — a few large-magnitude predictions inflate the
series-wide Duan smear (the documented failure mode). This is the strongest possible confirmation of the
prior: **the EBM regime stage was never an interpretability compromise** — its additive + pairwise +
**bagging** is the correct regularizer for the few-thousand-row, ~93%-noise h16-19 sample; an un-bagged
per-block DL fit has no such variance control. (Caveat: part of this is a *calibration* gap — output
shrinkage / bagging the NAM might recover the EBM level; but it cannot plausibly *beat* it given the
small-sample evidence. The gate rungs d1/d2 test whether a learned partition changes the verdict.)

## Temporal kNN on the same d8 leftover (DONE) — placebo-clean, dominated by EBM
Hero A (linbest d8@cs0.5) base: full 0.12081, h16-19 0.16070.

| variant | best full | best h16-19 | placebo (should ~0) | verdict |
|---|---|---|---|---|
| `knn_analog` (naive weighted-mean) | 0.12149 (**+0.00067**) | 0.16390 (+0.0032) | SHUFFLE +0.00119 (NOT clean) | **hurts** — pure variance injection |
| `knn_local` (large-K local-linear ridge) | **0.12071** (−0.00011, `V` set) | 0.15982 (**−0.00088**) | SHUFFLE +0.00028 (clean: real<0, placebo>0) | helps, but **< EBM 0.12033** |

- The **naive analog over-adds variance and hurts** (the shuffle placebo hurts as much as the real → no
  signal, just noise injection) — exactly the documented kNN failure.
- The **proper local-linear ridge helps** (−0.00088 on h16-19, sign-clean vs placebo) — reproducing the
  prior −0.00018-class result, now apples-to-apples on `linbest`. But its best (0.12071) is **dominated by
  the EBM regime (0.12033)**: kNN is a *validated method, not a deployable lever here*.
- **Convergent story with the MoE:** a fixed-metric kNN (hand HAR-state metric) and a learned-metric gate
  (the MoE) both fail to beat the EBM on this slot. The close leftover is low-dimensional, noise-dominated,
  and the EBM already extracts its thin additive+pairwise signal. The lever is **information, not local/
  relevance/DL modelling** of the existing state.
- Minor bug: `knn_analog` crashes for K≥50 (`argpartition(dist, K)` needs `ptr > K`; guard is `ptr < K`).
  K=25 result valid; fix is a one-char guard change if we want the K-sweep.

## Status
ctrl_ebm ✓, moe_d0 ✓, kNN ✓. Awaiting moe_d1 / moe_d2 / moe_d1diag (single incremental waiter re-armed).

## sat_d0 (clean input-saturation) + gate readout (2026-06-29, later)

**Input-saturation WORKS structurally, but the additive NAM still can't replicate.**
| config | full-OOS | h16-19 | vs no-regime 0.12081 / EBM 0.12033 |
|---|---|---|---|
| moe_d0 unbounded SiLU NAM | 0.15200 | 0.33329 | catastrophic (Duan-smear blowup) |
| **sat_d0 input-saturation (n_bags4, additive)** | **0.12757** | 0.19761 | bounding FIXED the blowup, but still +0.0068 vs no-regime |
| no-regime (cs0.5 global) | 0.12081 | 0.16070 | — |
| EBM regime | 0.12033 | 0.15736 | the bar |

- **Diagnosis confirmed:** per-feature input saturation took the NAM 0.152 -> 0.12757 (the unbounded-SiLU
  extrapolation was the blowup cause). But the bounded *additive* NAM STILL ADDS HARM (h16-19 0.198 vs the
  base 0.161) -> it overfits the tiny noisy close sample AND lacks the EBM's pairwise interactions. Exactly
  the expressiveness story: a GAM can't match a GA2M, and an under-regularized one overfits. -> motivates the
  reg'd (shrunk + feature-subsampled) sweep and the FM (native pairwise) sweep.

**Gate readout (the interpretable payoff):** the learned soft gate partitions the close window by **VOL
STATE** — top |w| = `har_ma_125`, `har_ma_1`, `voldemand`, `vix`, `har5rank_x_close`; **`hour` is rank
14/26/69**, NOT a top splitter. So the gate "finds a partition the clock smeared" — a vol-regime sub-
structure WITHIN h16-19 (where `hour` is ~constant). BUT this vol-state partition does not improve QLIKE,
**consistent with the original discovery** (intraday_regime_findings §6: state-regime interactions FAIL,
-0.0027). The gate rediscovered the vol-state axis the hand-analysis already rejected -> coherent, not a new
lever. The interpretable readout is the deliverable; the QLIKE confirms the floor.
