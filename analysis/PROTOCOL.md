# Protocol and retraction ledger

Written after a session that produced seven retracted claims and one paper
rewritten around a data artifact. The purpose is to fix the conventions before
any further result is quoted, and to record what has been withdrawn so a dead
claim cannot be silently reinherited.

## 1. Why the retractions happened

Every one traces to one of four causes. They are listed because the protocol
below exists to make each of them impossible rather than unlikely.

**One factor at a time.** The stripped-down model beat the elaborate one on the
pre-fix cache and lost on the corrected one — a design x data-version
interaction, discovered by accident after two paper rewrites. The same error
then repeated inside the prep fix: median substitution alone looked harmful,
masked estimation alone looked useless, the IQR floor alone looked inert, and
all the value was in the interaction. Three attempts failed for one reason.

**Proxy metrics never validated.** Scaled magnitude in rolling-IQR units was
used as a stand-in for forecast harm, and it was wrong in both directions: it
called `sumbipow_ewstock` at 84.6 IQRs fat tails when the composed fix takes it
to 17.7, and it called effspread at 932 a regression whose QLIKE cost was never
measured. No claim denominated in IQRs is a claim about forecasting.

**Selection contaminating the estimate.** The MIDAS shape parameter was chosen
against the reported out-of-sample loss (+0.0033); the exponent sweep's argmin
was the same error until caught; penalties were inherited across a dataset
change.

**Numbers quoted across incompatible footings.** QLIKE 0.13280 (pre-fix cache)
was tabled against 0.13177 (fixed cache) without labels, and 504 vs 505 columns
were used interchangeably.

## 2. Fixed conventions

Anything violating these is not reportable.

| | convention |
|---|---|
| panel | `results/b2_mmap*`, 242,934 rows; walk begins at bar 24,000 |
| protocol | rolling 24,000-bar training window, refit every 1,000 bars |
| loss | QLIKE on raw RV after Duan smearing; MSE reported alongside for any effect under 0.005 |
| selection | on a region strictly preceding the evaluation region, or on a tail held out inside each training window; never on evaluation rows |
| oracle arms | permitted, always labelled `ORACLE`, never quoted as achievable |
| design size | 41 channels x 12 rungs = 492 exogenous columns; + 12 HAR = **504 features**; + intercept = 505 parameters. Use "504 features" in tables |
| labelling | every quoted number carries its cache version |
| stability | any effect below 0.005 is reported by era, not pooled |

## 3. Pre-registered design

Stated before running, so the analysis cannot drift into whatever the data
rewards.

```
A  cache     none / composed-fix          (2 levels, was 3 before the
                                            indicator-only build was
                                            superseded)
B  design    backbone (12) / amplitude b in {0.5, 1, 2} (53) /
             amplitude b fitted per window (53) / raw 492 (504)
blocks       8 eras
```

**Hypotheses, with the direction each would have to fall to count as
confirmed:**

- H1 The composed prep fix removes the October 2023 failure. *Confirmed if*
  era-8 ridge-vs-backbone goes from +0.03321 to negative.
- H2 The prep fix is worth more than any design choice. *Confirmed if* the
  cache main effect exceeds the design main effect.
- H3 The winner depends on the cache. *Confirmed if* `argmin` over designs
  differs between cache levels. This is the interaction that has already
  reversed a conclusion twice.
- H4 The amplitude exponent is a non-lever. *Confirmed if* the three fixed-beta
  cells lie within 0.0005 of each other.

**What would falsify the stripped-down design's case:** raw 492 beating it on
the composed-fix cache by more than 0.0005 with `|t| > 3`. On the pre-fix cache
that comparison went the other way, which is why it is stated in advance here.

## 4. Retraction ledger

Claims made and withdrawn in this session. None may reappear without new
evidence.

| claim | status |
|---|---|
| the effective kernel is negative, n = -0.038 | withdrawn: wrong estimand and wrong basis |
| the kernel estimate is not reproducible | withdrawn: rebuilt cache reproduces at corr +1.0000 |
| the trial space is a manifold, not a subspace | withdrawn: the width framing does not apply to a single-solution problem |
| one nonlinear degree of freedom beats twelve linear | withdrawn: oracle selection artifact, +0.0033 |
| a manifold-drawn linear space cannot reach the nonlinear fit | withdrawn: arithmetically guaranteed reachable |
| QLIKE rewards attenuated forecasts (as a loss-geometry claim) | withdrawn: two attenuation operators disagree; inconclusive |
| the kernel ceiling is 0.26337 | withdrawn: an estimator ceiling, beaten at 0.26181 |
| directed shrinkage makes 41 weak channels usable | withdrawn: it is tail-risk control, then withdrawn again as a data artifact |
| the stripped-down model beats the elaborate one | withdrawn: reversed on the corrected cache |
| the raw parameterisation is actively harmful | withdrawn: it faithfully carried a corrupted feature |
| the exponent's usable range is [-1, -0.5] | withdrawn: flat over [0.2, 3.0] |
| 23 of 41 channels show the voldemand pathology | withdrawn: the detector conflated collapsed scale with fat tails |
| the honest indicator is the whole of the bug fix | withdrawn: it changes no feature value |
| the median substitution is actively harmful | withdrawn: harmful alone, correct as half of a pair |

## 5. Order of execution

No step may start before the previous one passes.

1. Build the composed-fix cache.
2. `prep_invariants.py` across all caches — must show the composed cache
   clean on classes A, B and E.
3. `factorial_design.py` — the pre-registered design above.
4. `audit_paper_numbers.py` — every quoted number traces to a JSON.
5. Rewrite the paper, against results that have passed 1-4 and carrying the
   ledger in Section 4 as a published limitation rather than a private note.
