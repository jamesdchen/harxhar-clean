# P-Model / Spectral-Prior Program: Results Brief for Writeup
<!-- Cold-session handoff. Written 2026-08-10. Everything below is final unless marked pending. -->

## 0. TL;DR

The spectral/transmission block in the HAR + exogenous ridge contains **no new
information** — it is a reparametrization of shrinkage geometry. The frozen-score
version is *exactly* a two-block generalized ridge with a spiked prior covariance
\(P^{-1}=\lambda_Z^{-1}I+AD_G^{-1}A^\top\) on the exogenous coefficients.
A static \(P\) recovers only ~19% of the no-product spectral gain; the rest is the
**trailing-SD gate**, which no time-invariant Gaussian prior can express. The
correct forecasting-model framing is a **gated low-rank coefficient adapter**
(LoRA-type): \(\delta_t = \beta_Z + A\,\mathrm{diag}(s_t)\,\gamma\), rank 40,
gate \(s_t\) = trailing realized SD of each factor direction. The exogenous block
itself is the dominant source of QLIKE (~80% of total gain over HAR-only);
everything in this program is a second-order correction on top of it.

## 1. Repo / data / compute layout

- Remote: `jamesdc1@dtn1.hoffman2.idre.ucla.edu`
- Project: `/u/scratch/j/jamesdc1/harxhar-clean`
- Python: `/u/home/j/jamesdc1/.conda/envs/hpc-pi/bin/python`
- Scheduler: UGE/SGE, `/u/local/bin/qsub|qstat|qdel`
- Chunking contract (`oos_mult=2`): `WARMUP=24000`, `N_CHUNKS=100`,
  legal chunks are **9–99** (91 chunks, 248,686 OOS rows). Chunk bounds:
  `start = WARMUP + span*i//N_CHUNKS`, `end = WARMUP + span*(i+1)//N_CHUNKS`.
- One-slot array runner: `jobs/sge/exog_spike_single.sge` (set `ARM=...`,
  `SOLVER=rank2`). Runner `experiments/run_unification_batch.py` skips existing
  chunk NPZs, so re-running an array only fills gaps.
- Scoring: `experiments/score_unification.py --roots results/unification --out results/wave_scores_2026-08-10.csv`
  (also rewrites `results/unification_increments.csv` and `writeup/generated/*.tex`).
- Arm registry + all designs: `src/unification.py` (`ARM_REGISTRY`).

## 2. The scoreboard (QLIKE, n = 248,686 OOS rows, 91/91 chunks unless noted)

| Arm | Description | QLIKE |
|---|---|---|
| `a0_ols_har` | HAR-only OLS baseline | 0.226907 (204,474 rows) |
| `blk2_gated_tuned` | backbone + broad exog ridge (the workhorse) | 0.219971 |
| `blk3_exogSpikeFrozen_tuned` | = static P, exog frame, frozen | 0.219902 |
| `blk3_exogSpikeTrailSd_tuned` | exog frame, trailing SD | 0.219796 |
| `blk3_prodSpikeFrozen_tuned` | = static P, product-base frame, frozen | 0.220039 |
| `blk3_trailGShapedTrailSd` | product-base frame + trailing SD (no-product champion) | 0.219609 |
| `blk4_prodSpikeFrozen_tuned` | prod-frame frozen spike + product block | 0.219348 |
| `blk4_trailGShapedTrailSd` | full champion | 0.218861 |

Mechanism cells (all prod-frame, frozen, no product block unless stated):

| Arm | QLIKE | vs `blk3_prodSpikeFrozen_tuned` (ΔQ, DM) |
|---|---|---|
| `blk3_prodNoCrossSpikeFrozen_tuned` (cross-block cov deleted) | 0.220069 | −0.000030, **DM = −2.48** |
| `blk3_prodNoHarSpikeFrozen_tuned` | 0.220038 | +0.000001, DM = +0.02 |
| `blk3_prodNoSessionSpikeFrozen_tuned` | 0.220050 | −0.000011, DM = −0.36 |
| `blk3_prodValuesSpikeFrozen_tuned` | 0.220023 | +0.000016, DM = +0.35 |
| `blk3_prodBlockPermSpikeFrozen_tuned` (cross-family time sync destroyed) | 0.220002 | +0.000036, DM = +1.15 |

Key increments (ΔQLIKE, DM; negative = first arm better):

- Exog block value: `blk2_gfull_tuned` vs `a0_ols_har` (n=179,606): **−0.00462, DM = −10.3**
- Frame × scaling factorial (no product block anywhere):
  - exog trail − exog frozen: −0.000106, DM = −0.84
  - prod frozen − exog frozen: +0.000137, DM = +1.21
  - **prod trail − prod frozen: −0.000430, DM = −3.17**  ← the interaction
  - prod trail − exog trail: −0.000187, DM = −1.37
- Product block increment: `blk4_trailGShapedTrailSd` − `blk3_trailGShapedTrailSd`:
  −0.000748, DM = −1.76 (≈ 67% of champion's total gain over `blk2_gated_tuned`;
  spectral ≈ 33%)
- Static P vs full no-product champion: static P captures 0.000068 of 0.000361 ≈ **19%**.

Full machine-readable tables: `results/wave_scores_2026-08-10.csv`,
`results/unification_increments.csv`.

## 3. The math (what a writeup can state as proved)

### 3.1 Exact absorption (frozen scores)

Three-block ridge: \(y=a+B\beta_B+Z\beta_Z+G\gamma+\varepsilon\), \(G=ZA\)
(frozen deterministic map), penalties \(\lambda_Z\|\beta_Z\|^2+\gamma^\top D_G\gamma\).
Substituting \(\delta=\beta_Z+A\gamma\) gives the two-block generalized ridge

\[
y=a+B\beta_B+Z\delta+\varepsilon,\qquad
\delta\sim N(0,P^{-1}),\quad
P^{-1}=\lambda_Z^{-1}I+AD_G^{-1}A^\top .
\]

Identical normal equations ⇒ identical forecasts. So the frozen three-block arms
*are* the P-model fits; no separate empirical two-block fit is needed.
Bayesian reading: \(\delta=\nu+A\gamma\), \(\nu\sim N(0,\lambda_Z^{-1}I)\),
\(\gamma\sim N(0,D_G^{-1})\) — a base isotropic prior plus a rank-40 covariance spike.
If \(A^\top A=\mathrm{diag}(h_i)\), spike eigenvalues are \(\eta_i=h_i/d_i\) and
prior-variance inflation along direction \(i\) is \(R_i=1+\lambda_Z h_i/d_i\).

### 3.2 Why trailing SD escapes P

Trailing-SD scores are \(G=D_sZA\) with \(D_s\) a diagonal time-varying gate.
No constant \(A'\) satisfies \(D_sZA=ZA'\), so the model is not expressible as any
time-invariant prior. It *is* a time-varying coefficient path

\[
\delta_t=\beta_Z+A\,\mathrm{diag}(s_t)\,\gamma
\]

— a gated low-rank (LoRA-type) adapter on the exogenous loadings, with
observable volatility gates. Local shrinkage on direction \(i\) at time \(t\):
\(\kappa_i(t)=s_{t,i}^2/(s_{t,i}^2+d_i/W)\).

### 3.3 Joint-state prior / cross-block covariance

Write the joint state \(X_t=[B_t;Z_t]\), \(\delta=[\beta_B;\beta_Z]\). The spectral
block induces \(C=C_0+AD_G^{-1}A^\top\) with cross-block term
\(C_{BZ}=A_BD_G^{-1}A_Z^\top\). Deleting it (split spike on backbone and exog
parts separately, `blk3_prodNoCrossSpikeFrozen_tuned`) removes the only
statistically detectable frozen-frame effect (DM = −2.48, though tiny ΔQ).

### 3.4 Canonical numerical-rank cutoff (full-rank diagnosis, earlier phase)

Naive full-rank trailing-SD failed via division by numerical-null directions.
Fix: keep \(\lambda_i > \tau = p\,\varepsilon\,\lambda_{\max}\) (float64 machine
precision). Full-rank findings: frozen scaling helps, trailing SD hurts at full
rank, exog vs all-panel frames tie, product frame worse, smooth power-law spectra
do **not** replace the hard \(K=40\) projection.

## 4. Empirical mechanism conclusions (each backed by a run)

1. **Redundancy**: spectral factors are linearly spanned by backbone+exog;
   residual squared-norm share after projection ≈ 3.6e-30 (float64 roundoff).
   The block changes estimator geometry, not information.
2. **Pure exogenous P fails**: ΔQ = −0.000068 vs plain exog ridge, DM = −0.76.
   A rank-40 spike over exogenous coefficients alone captures nothing.
3. **Frame is the mechanism**: coefficient-direction alignment of the top-40
   frame subspace vs rolling ridge coefficients (5 windows,
   `results/frame_alignment_diagnostics.csv`):
   prod-family ≈ 1.5–4.5 (relative alignment), **exog ≈ 0.02–0.07**,
   all-panel ≈ 0.007–0.04. The exog frame's top-40 subspace misses the
   predictive direction almost entirely.
4. **Interaction, not main effect**: trailing SD pays off only on the
   product-base frame (DM = −3.17 vs DM = −0.84 on exog). Frame alone (frozen)
   does not help.
5. **Cross-block covariance is real but small**: DM = −2.48, ΔQ ≈ −3e-5.
   HAR-only or session-only frame deletions are nulls; within-family block
   permutation is also ~null (DM = +1.15) — the joint *mixed-family* directions
   matter, fine-grained time synchrony does not.
6. **Product block is a separate nonlinear channel**: ΔQ = −0.000748,
   DM = −1.76 beyond the linear spectral block; ≈ 2/3 of the champion's gain.
7. **Where the QLIKE lives**: exog block ≈ −0.0046 (DM ≈ −10) over HAR-only;
   all spectral machinery ≈ −0.0011 total; static-P portion ≈ −0.00007.

## 5. Framing for the paper/talk

- Forecasting model: HAR backbone + broad exogenous block, coefficients shrunk
  by a **gated low-rank adapter**: \(\delta_t=\beta_Z+A\,\mathrm{diag}(s_t)\gamma\),
  rank 40, \(A\) = frozen-window frame eigenvectors, \(s_t\) = trailing realized
  SD per adapter direction (causal, observable at forecast time).
- The static \(P\) model is the constant-gate special case, exactly equivalent
  to the frozen three-block regression.
- Honest magnitudes: gains over backbone+exog are ~1e-4–1e-3 QLIKE on a base of
  ~0.22; DM values are single-path and there is a multiple-testing caveat across
  ~dozens of arms. The contribution is identification of the mechanism
  (geometry vs gate vs nonlinearity), not a large forecast improvement.

## 6. Reproduce / verify

```bash
cd /u/scratch/j/jamesdc1/harxhar-clean
PY=/u/home/j/jamesdc1/.conda/envs/hpc-pi/bin/python
# score everything (reads chunk NPZs under results/unification/<arm>/)
$PY experiments/score_unification.py --roots results/unification \
    --out results/wave_scores_2026-08-10.csv
# rerun any missing chunks for an arm (91 one-slot tasks):
ARM=blk3_prodSpikeFrozen_tuned /u/local/bin/qsub jobs/sge/exog_spike_single.sge
```

Diagnostics already harvested (do not rerun unless needed):
`results/p_hypothesis_analysis.json`, `results/p_hypothesis_penalty_diagnostics.csv`,
`results/p_hypothesis_factor_signal.csv`, `results/p_hypothesis_exog_frame_signal.csv`,
`results/exog_spike_p_prior_diagnostics.csv`, `results/frame_alignment_diagnostics.csv`.

Scripts: `experiments/analyze_p_hypotheses.py`, `experiments/analyze_p_exog_signal.py`,
`experiments/analyze_frame_alignment.py` (job `jobs/sge/frame_align.sge`).

## 7. Caveats to keep in the writeup

- Single asset, single OOS path; DM statistics are path-dependent.
- Multiple comparisons: only the exog-block value (DM ≈ −10), the frame×scaling
  interaction (DM = −3.17) and the alignment contrast are robust to that concern;
  cross-covariance (DM = −2.48) and product increment (DM ≈ −1.8) are marginal.
- Generated CSVs may trigger `git diff --check` trailing-whitespace warnings; cosmetic.
