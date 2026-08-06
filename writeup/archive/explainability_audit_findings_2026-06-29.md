# Explainability audit — findings (2026-06-29)

Mining the explanations the pipeline already emits (not fitting new models). The walk-forward refits ~407
times, so every coefficient/mask/shape is a **time series** we have always collapsed to one number. Notebook:
`notebooks/results/edge_features/edge_06_explainability_audit.ipynb`. Cell `xgb_all_buckets_tw1000_enetreg2_linbest_rf480_slim`.

## Finding 1 — the close regime ROTATES across timescales (coefficient trajectory)

`coef_trajectory.py` → the per-block linbest base coefficient on each HAR×{open,close} interaction, early
(first 10 blocks) vs late (last 10):

| term | mean | first10 | last10 | read |
|---|---|---|---|---|
| `har_ma_1×close` (fast) | +0.0055 | +0.0305 | +0.0110 | **fades toward 0** |
| `har_ma_5×close` (fast) | +0.0014 | +0.0157 | +0.0000 | **fades toward 0** |
| `har_ma_25×close` (slow) | +0.0806 | +0.0391 | +0.0577 | strengthens |
| `har_ma_125×close` (slow) | +0.0865 | +0.0495 | **+0.1171** | strengthens most (|Δ|=0.068) |
| `har5rank×close` (rank-space damp) | −0.0466 | −0.0508 | −0.0457 | **stable** (largest close term) |
| `cumrv×close` | −0.0046 | −0.0038 | −0.0053 | stable |
| `har_ma_1×open` | −0.0436 | −0.0417 | −0.0661 | open damp strengthens |

**Read:** not a uniform fade — a **cross-timescale rotation**. The *fast* close-damping (1-/5-bar × close)
decays toward zero across the sample while the model shifts close-regime weight onto *slower* HAR terms
(25-/125-bar × close). The fast-term decay is the **0DTE/OPEX-dilution signature** (intraday_regime_findings
§12): daily expiries spread the fast gamma unwind, so the model's reliance on the bar-scale close interaction
fades; the persistent close pressure migrates to multi-day terms. The dominant close coefficient
(`har5rank×close`, the rank-space damp) is **stationary** — the *level* regime is stable; only its
*timescale composition* drifts.

**Caveat (and the clean follow-up):** these are **L1-penalized** linbest coefficients, so "→0.0000" partly
reflects L1 zeroing the fast term in late blocks rather than a true coefficient decay. The confound-free test
is to re-run `coef_trajectory.py` on the **unpenalized harunpen base** (`base_kind=enetreg2_harunpen`, where
the dense HAR block is not penalized) — if the fast×close terms still fade there, the dilution is real.

## Finding 2 — the learned gate hyperplane (gate readout)
*Pending — `explain.sbatch` gate_readout depth 1/2 still running; folds in here on completion.*

## Probes still on the table (pipeline-native, specified in edge_06 §3)
Survivor-mask evolution (regime-change detector), H-stat/SHAP interaction localization on the d8 global
(is the order-≥3 structure concentrated → distillable, or diffuse?), final-cascade residual diagnostics
(is the floor white noise?), EBM bag-disagreement confidence map.
