#!/bin/bash
# Submit the GRID-FREE SHRINKAGE arms (2026-08-07): 4 arms x 100 chunks.
# ZERO tuned hyperparameters in any of them — that is the entire point.
#
# THE PRINCIPLE. Every estimator in this campaign allocates shrinkage by a
# hand-designed structure and then SELECTS its level on a 125-bar validation
# tail. Our own diagnostics say that machinery is failing on its own terms:
# tuned ridge pins its modal alpha at the grid top in 8/8 bucket designs, the
# elastic net pins 43%, the PCR block 74% bimodally, backbone blocks 53%. These
# arms replace SELECTION with ESTIMATION. Each fitted coefficient carries its
# own standard error — beta_hat ~ N(beta, sigma^2 (X'X)^-1) — so the
# risk-minimizing shrinkage is computable from the TRAINING WINDOW ALONE.
#
# THE NEGATIVE RESULT DELIBERATELY AVOIDED: empirical Bayes with a SINGLE prior
# variance tau^2 gives shrinkage d_i tau^2/(d_i tau^2 + sigma^2), which is
# EXACTLY ridge at lambda = sigma^2/tau^2. A common prior IS the overarching
# shrinkage we already have. The content is in a NON-CONSTANT prior, ESTIMATED
# rather than parameterized as a shape — and the shape family is already tested
# and dead (power/exponential/step interchangeable; the tilt LOSES where it is
# actually applied, DiD z = -4.15).
#
# ARMS (all on blk3_tuned's exact design: backbone + exog + product, window
# 24000, per-bar refit, oos_mult=2, legality range(9)). The backbone (HAR
# ladder + session interactions + calendar) is left UNSHRUNK in every arm and
# its coefficients are bit-identical to the plain least-squares solve: the JS
# risk guarantee is a statement about the shrunk block, and the backbone is the
# incumbent's own audited basis.
#   blk3_js_tuned          positive-part James-Stein, EXACT block covariance.
#                          beta_hat_S ~ N(beta_S, sigma^2 C), C = [(X'X)^-1]_SS
#                          obtained as the Schur complement
#                          C^-1 = G_SS - G_SB G_BB^-1 G_BS. Whitening is a
#                          scalar map, so only the Wald form
#                          ||gamma||^2 = beta_S' C^-1 beta_S is ever formed.
#                          Unknown-variance convention:
#                          f = (1 - [(k-2)/(m+2)] m sigma2_hat/||gamma||^2)_+
#                          with m the residual dof.
#   blk3_npeb_tuned        nonparametric empirical Bayes. z_i = beta_i/se_i is
#                          unit-variance BY CONSTRUCTION, and Tweedie gives the
#                          posterior mean with no prior parameterization:
#                          E[mu|z] = z + dlog f(z)/dz. f and f' come from a
#                          Gaussian KDE evaluated ANALYTICALLY at the sample
#                          points (so f > 0 always and f'/f is bounded),
#                          bandwidth by Silverman's rule — a deterministic
#                          function of the data, nothing selected.
#   blk3_js_pcbasis_tuned  DIAGONAL-whitened JS with the exog block rotated
#                          into the frozen base-feature eigenbasis.
#   blk3_jsDiag_tuned      the same diagonal-whitened JS WITHOUT the rotation.
#
# WHY THE PC ARM IS DIAGONAL-WHITENED, AND WHY blk3_jsDiag_tuned EXISTS. The
# EXACT James-Stein factor is INVARIANT under any orthogonal rotation of the
# shrunk block: it depends on the data only through beta_S' C^-1 beta_S, and
# under beta -> A beta, C -> A C A' with A orthogonal that form is unchanged.
# So "exact JS in the PC basis" would be a BIT-IDENTICAL DUPLICATE of
# blk3_js_tuned — 100 chunks for a provable no-op. (The synthetic asserts the
# invariance to 1e-8 rather than asserting a difference that cannot exist.)
# What DOES depend on the basis is the DECOUPLING APPROXIMATION — whitening by
# the diagonal alone is exact only when the gram is diagonal, and rotating into
# the frozen frame is what makes it nearly so. Hence the 2x2: PC-vs-raw at
# matched whitening isolates the BASIS; either against blk3_js_tuned measures
# what the decoupling COSTS.
#
# sigma^2 IS CAUSAL AND HONEST: RSS = y_c'y_c - beta'X_c'y_c read straight off
# the rolling sufficient statistics (no residual vector formed), divided by
# n - p with p = design columns + 1 for the centered intercept. Not reduced
# further for the shrinkage — the factor is estimated FROM this sigma^2, so a
# shrinkage-aware dof would be circular. At n=24000, p~1250 the correction is
# ~5%, not cosmetic.
#
# CONDITIONING FALLBACK: the Cholesky is rank-revealing for a PSD gram, so a
# LinAlgError IS the near-singularity signal; on failure the solve drops to the
# MIN-NORM pseudo-inverse (the path _walk_ols already takes for the
# rank-deficient OLS arms) and every downstream consumer falls back explicitly
# instead of propagating a bad factor. Fallback bars are counted and persisted,
# so a run that leaned on it can never be mistaken for one that did not.
#
# NO TUNING MACHINERY AT ALL: these arms never touch BLOCK_TUNE_GRIDS or
# ESTIMATOR_GRIDS, meta.tuned_alphas/tuned_grids stay empty by construction
# (the shape summary skips arms with no descriptors, so nothing downstream sees
# a malformed entry), and the exhibit rides in meta.shrink_profile ->
# results/shrinkage_profile.csv: per boundary, the DISTRIBUTION of the factors
# the estimator derived. A FLAT decile profile on the NPEB arm would mean the
# estimated prior is effectively constant, i.e. ridge in disguise — which is
# exactly the negative result above, and would be worth knowing.
#
# WHY THIS MATTERS: if a zero-hyperparameter estimator matches or beats the
# tuned arms, the whole tuning apparatus — and the endpoint pinning that
# contaminates it — was unnecessary, and the paper's methodological conclusion
# turns from cautionary into constructive.
#
# HOFFMAN2 (intended target; CARC is saturated by the tree bank):
#   cd /u/scratch/j/jamesdc1/harxhar-clean
#   for arm in blk3_js_tuned blk3_npeb_tuned blk3_js_pcbasis_tuned \
#       blk3_jsDiag_tuned; do
#     qsub -N "unif_$arm" -v ARM="$arm" -tc 60 jobs/sge/unification_serial.sge
#   done
# This script is the CARC-shaped equivalent, kept for parity with the rest of
# jobs/submit/. Run ON the CARC login node from /scratch1/jc_905/harxhar-clean.
set -e
cd /scratch1/jc_905/harxhar-clean
for arm in blk3_js_tuned blk3_npeb_tuned blk3_js_pcbasis_tuned blk3_jsDiag_tuned; do
  sbatch --export=ALL,ARM="$arm" --job-name="unif_$arm" jobs/slurm/unification.sbatch
done
squeue -u "$USER" -h | wc -l
