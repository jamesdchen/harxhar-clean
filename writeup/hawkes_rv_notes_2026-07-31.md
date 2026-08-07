# Hawkes processes and intraday S&P RV — literature position + our results (2026-07-31)

Screening slice: 2,189 OOS 30-min bars; walk-forward, per-bar (mu, alpha) /
cadence beta; QLIKE in raw variance space via the production Duan scaffold.
Comparators: OLS-HAR incumbent 0.17856 (~40 coefs), base-2 per-bar ridge on
the full panel 0.16425 (1,077 coefs).

## What the literature has

- **Hawkes-driven stochastic volatility on S&P**: goodness-of-fit work on
  Hawkes-SV specifications; best fit when intensity is a linear transform of
  the variance process (consistent with our K-filter regression form).
  [Annals of OR 2022](https://link.springer.com/article/10.1007/s10479-022-04924-9)
- **Hawkes volatility estimation from tick data**: intraday Hawkes vol as an
  estimator; found to carry predictive power for DAILY vol (closest cousin
  to our exercise, but estimation-then-forecast, not a direct walk-forward
  forecaster benchmarked at the intraday horizon).
  [Lee 2024, ASMBI](https://onlinelibrary.wiley.com/doi/10.1002/asmb.2892) /
  [arXiv:2207.05939](https://arxiv.org/pdf/2207.05939)
- **Marked Hawkes price dynamics + vol estimation**:
  [arXiv:1907.12025](https://arxiv.org/pdf/1907.12025)
- **Crash-period stock vol forecasting with Hawkes**:
  [ResearchGate 2023](https://www.researchgate.net/publication/369663384_Forecasting_Stock_Volatility_during_the_Stock_Market_Crash_PeriodThe_role_of_Hawkes_process)
- **Theory**: rough volatility as the scaling limit of nearly-unstable
  Hawkes (Jaisson-Rosenbaum; El Euch-Fukasawa-Rosenbaum) — the license for
  treating a power-kernel Hawkes as the generative model of RV memory.
- **HAR-benchmark intraday RV forecasting**: standard comparisons at 5-min /
  intraday horizons (e.g. [HAR intraday forecasting thesis](https://thesis.eur.nl/pub/73357/Thesis_Bachelor_Olivier_van_Wel_577295ow.pdf));
  HAR + wavelet variants
  ([ScienceDirect 2026](https://www.sciencedirect.com/science/article/pii/S1062940826000276)).

**Gap we don't find covered**: (i) a power-kernel Hawkes FILTER as a direct
walk-forward 30-min-ahead S&P RV forecaster scored head-to-head against HAR;
(ii) the kernel-filter <-> HAR-ladder-regression equivalence stated and
exploited; (iii) any dense-weak analysis of multivariate excitation; (iv)
the horizon law measured INSIDE the generative class.

## Our results (all this slice, walk-forward, registered predictions)

1. **3 parameters match 40**: power-kernel Hawkes (mu, alpha, beta; U=256)
   QLIKE 0.18020 vs incumbent 0.17856 — DM p=0.50, statistical tie. The
   complete target-only story on an index card.
2. **Kernel-shape law confirmed**: exponential kernel 0.19140 (+0.011) —
   memory is power-law, decisively (generative-side confirmation of the
   pol1 log-linearity result).
3. **Horizon law in the generative class**: U=2048 -> 0.25161 (+0.071!). A
   normalized kernel cannot zero its own tail, so even a 3-parameter model
   needs the ~5-day hygiene cut. Strongest form of the horizon law measured.
4. **Fitted beta = 1.36, not the rough-vol 0.6** (registered miss): a free
   intercept absorbs the slow component, so the single kernel fits only the
   fast remainder and steepens. Recovering structural H needs the kernel to
   own the slow part (no intercept / two-kernel fit).
5. **Parsimony has a measured price at every level of breadth**:
   top-3 channels as kernels (5p): captures ~0 of the 0.016 breadth gap.
   ALL 42 channels, 1 fitted kernel each (43p): ~1.5%.
   Level+tilt per channel (85p): ~19%.
   Full ladder (1,077p): 100%.
   The breadth factor is dense-weak in channels AND timescales JOINTLY.
   (Metric-vs-regression inversion: the same pol1 compression keeps ~90% of
   a METRIC's value but ~19% of the REGRESSION's — different consumers.)
6. **Why Hawkes cannot hold the cross-section (SVD of the fitted 41x12
   kernel matrix)**: shape space needs 4-5 factors (rank-2 = 70%, rank-4 =
   91%), and the leading shapes OSCILLATE (8 sign changes; adjacent long
   rungs +0.60/-0.59) — the regression internally builds signed band
   contrasts (differences of adjacent MAs). Positive monotone excitation
   kernels are the wrong class for exog transfer; the proper model is a
   **Hawkes core (self-channel: positive, power-law, near-critical) plus
   general signed linear covariate filters** — i.e., the ridge anchor IS
   the nonparametric estimator of the correct hybrid. Smallest legible
   middle form: a 4-5 shape dictionary with per-channel loadings (~200p),
   untested.
   (Also resolves the bands paradox: pre-standardized band FEATURES failed;
   fitted band WEIGHTS are what the regression builds itself.)
7. **Ridge on excitation has a physical meaning**: near-criticality (the
   rough-vol regime) puts fitted excitation near the stability boundary;
   L2 shrinkage of amplitudes is insurance against estimating a
   supercritical (explosive) process. Hierarchical pooling of per-channel
   exponents and shrink-to-diagonal excitation structure are the two
   principled upgrades if the parametric route is pursued.

## Placement

The paper-shaped claim: *intraday S&P RV behaves as a near-critical
self-exciting system whose own-history is 3-parameter compressible
(power kernel, ~5-day horizon), while its cross-sectional drivers enter as
signed, oscillating, jointly-dense linear filters that admit no parsimonious
excitation representation — the measured boundary between the Hawkes story
and the breadth story.* Fits between the Hawkes-SV fitting literature
(which stops at goodness-of-fit) and the HAR-forecasting literature (which
never uses the generative frame), with the rough-vol theory as the bridge.
