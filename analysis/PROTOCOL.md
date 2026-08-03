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
| significance | DM with Newey-West HAC and the HLN correction; there is no blanket `\|t\|` threshold. Any effect whose differential is concentrated (top 1% of bars carrying most of the mass) or confined to one era must additionally survive a block bootstrap and a leave-one-era-out sweep, because a normal-theory statistic on such a series is not trustworthy at any `\|t\|` |
| selection | on a region strictly preceding the evaluation region, or on a tail held out inside each training window; never on evaluation rows |
| oracle arms | permitted, always labelled `ORACLE`, never quoted as achievable |
| design size | 41 channels x 12 rungs = 492 exogenous columns; + 12 HAR = **504 features**; + intercept = 505 parameters. Use "504 features" in tables |
| arm names | `raw492` and `ridge492` are **the same arm** -- 504 features, identity penalty on the 492 exogenous columns -- and agree to 12 significant figures wherever both appear (0.1361736957106712 vs 0.13617369571068358). One design, two labels, in the same class as the 504/505 confusion above. `shaped492` is genuinely different: same 504 features, shared-shape matrix penalty instead of the identity. Prefer **`ridge-504`** and **`shaped-504`**; never name an arm after a number that is not its width |
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

## 3a. Results, read against the hypotheses as stated

Run on `b2_mmap_warm`, 218,909 bars common to every cell. Recorded before any
of it was written into the paper.

| | outcome |
|---|---|
| **H1** the composed fix removes the October 2023 failure | **CONFIRMED**. Era-8 raw492 vs backbone 0.13159 vs 0.13511 = -0.00352, from +0.03321 |
| **H2** the prep fix is worth more than any design choice | **REJECTED**. Cache main effect 0.00067, design main effect 0.00311. Holds on the affected window and in era 8, not pooled |
| **H3** the winner depends on the cache | **CONFIRMED**. none -> amp b fitted (0.13248); composed -> raw492 (0.13146) |
| **H4** the amplitude exponent is a non-lever | **CONFIRMED**. Fixed-beta cells span 0.00016 on composed, 0.00030 on none, both under 0.0005 |
| **falsification condition** raw492 beats the stripped-down design on the composed cache by >0.0005 with \|t\|>3 | **MET**: -0.00126, t -5.91, p 3.4e-09. The stripped-down design is withdrawn |

### H2 was badly posed, and the verdict stands anyway

H2 is rejected and stays rejected; the paper reports it as a failure. What
follows is a post-mortem on the hypothesis, not a re-scoring of it. Nothing
below changes what is claimed.

Five of the six designs are immune to the cache factor by construction:

| design | none | composed | delta |
|---|---|---|---|
| backbone | 0.13571 | 0.13571 | +0.00000 |
| amp b=0.5 | 0.13294 | 0.13301 | -0.00007 |
| amp b=1.0 | 0.13264 | 0.13285 | -0.00022 |
| amp b=2.0 | 0.13265 | 0.13286 | -0.00021 |
| amp b fitted | 0.13248 | 0.13272 | -0.00024 |
| **raw492** | 0.13617 | 0.13146 | **+0.00472** |

The backbone uses no exogenous columns and cannot respond; the amplitude
designs average 12 rungs and so dilute the corrupted channel about twelvefold.
The cache "main effect" is therefore five zeros and one 0.00472 divided by six.

Three flaws, all identifiable before the answer was known:

1. **Averaged over units that cannot receive the treatment.** A main effect
   averages over the other factor's levels, which is only meaningful if those
   levels form a sensible population. Here they do not.
2. **Not like-for-like.** The cache effect is a difference of two means; the
   design effect is a *range over six levels*. Ranges grow with level count.
3. **A pooled mean for a concentrated failure.** 27 bars carry 47% of the
   damage. Worse, section 2 of this document already requires that "any effect
   below 0.005 is reported by era, not pooled" -- H2 was stated pooled on
   effects below 0.005, so it violated our own convention.

Under better-posed estimands:

| framing | cache | design | holds |
|---|---|---|---|
| as pre-registered | 0.00066 | 0.00311 | no |
| like-for-like, immune backbone dropped | 0.00066 | 0.00121 | no |
| exposure-conditional (raw492) | 0.00472 | 0.00425 | yes, 1.1x |
| interaction | 0.00486 | 0.00311 | yes, 1.6x |
| era 8, where the defect binds | 0.03653 | 0.00311 | yes, 11.8x |

Fixing flaw 2 alone does **not** rescue H2 -- design still exceeds cache. So
the pooled claim genuinely fails and only the conditional one is supported,
which is what the paper says.

**Restatement for any future panel**, to be fixed before running: among designs
that actually use the affected columns, the data-version effect on an
era-level or worst-decile loss exceeds the largest design contrast, with the
same statistic computed on both sides.

**That restatement was then run here, post-hoc, and it does not cleanly hold**
(`analysis/exposure_conditional.py`, `writeup/stats/exposure_conditional.json`).
Exposure is defined structurally as `r/12`, the lag dimensions a design keeps,
so it is fixed before any loss is computed -- defining "exposed" as "responds
to the cache" would have been circular.

| estimand | cache effect (exposed) | design range (exposed) | holds | rho(exposure, effect) |
|---|---|---|---|---|
| T1 pooled | +0.00325 | 0.00009 | yes | +0.771 |
| T2 era-8 | +0.02329 | 0.00015 | yes | +0.829 |
| T3 worst decile by RV | **-0.00004** | 0.00015 | **no** | +0.371 |

T3 was the framing predicted to be strongest and it is the one that fails: the
October 2023 bars cluster in *time*, not in *volatility*, so "damage
concentrated in the tail" and "damage concentrated in high-volatility bars" are
different claims and only the first is true.

Two reservations against the test itself, recorded so it is not cited as
stronger than it is:

- the a priori cut admits only two designs, so the design range in the
  denominator is small almost by construction -- the mirror image of the flaw
  this restatement was meant to fix;
- `amp rank-3` on the corrupted cache is an outlier (0.14402 pooled, 0.23536 on
  era 8, against roughly 0.132-0.136 elsewhere) and inflates the rank
  correlation more than the smooth trend does.

Pooled DM on the one unambiguously exposed design: raw492 composed vs none
-0.00472, t -2.95, p 0.0032. An earlier version of this note called that
"below the |t| > 3 bar this protocol uses elsewhere", which was wrong and is
retracted: `|t| > 3` appears once in this document, as the pre-registered
falsification threshold for one specific comparison, and was never a general
convention. Inventing a house rule and then failing a result against it is the
same class of error as the unvalidated proxy metrics in section 1.

The reliability question underneath it is real and is answered separately in
`analysis/dm_robustness.py`, which tests the differential four ways rather than
against a threshold.

**Nothing here enters the paper as a confirmation.** The paper's claim remains
the conditional one: the defect dominates where it binds, and pooled averages
hide it.

Two further results, both on the clean cache:

- lag-axis rank 3 reaches 0.13149 in 135 columns against raw492's 0.13146 in
  504 -- almost all of the full design's advantage is lag shape, not
  channel-specific detail.
- the supervised d=6 retrieval metric moved from an exact null (+0.00015,
  Holm p 1.0) to beating the 516-dimensional ambient view by -0.00540
  (t -8.9). A distance weights every coordinate by its scale, so the collapsed
  denominator dominated the metric outright. The earlier "every estimated
  geometry loses" claim now holds only for the unsupervised ones.

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

## 4a. Gate log

Changes made to the pipeline *after* seeing a gate fail. Recorded because a
threshold moved after a failure is indistinguishable from a threshold tuned to
pass unless the reasoning is written down at the time.

**2026-08-03, gate 2 FAIL on `b2_mmap_indonly`: class B = 11.**

Class A went 31 -> 0, so the availability half of the composed fix is
confirmed. Class B (scale collapse) did not clear.

First hypothesis, that class B was another bad proxy, was tested and
*rejected*: on the composed cache the B-flagged channels are exactly the
high-magnitude ones (worst flagged 336.5 against worst unflagged 42.6,
Spearman -0.365). The gate failure is real. (Class B does miss effspread's 932.6
on `b2_mmap_fix`, but that cache's regression comes from the double-scaling
wiring bug, a different mechanism.)

Localised: the composed fix cures voldemand (2260.4 -> 6.7) and *introduces*
blow-ups in six channels that were previously fine -- stocktwits_sentcount
9.0 -> 336.5, vix3m 8.1 -> 307.6, stocktwits_attention 10.4 -> 52.3,
vvix 5.0 -> 32.6, numobs 80.9 -> 120.2, vix 3.5 -> 10.0. Each channel's worst
value sits 13 to 116 days after its feed's *first* print, so this is a
feed-initiation transient -- the same defect family as the voldemand
termination bug, at the other end of the feed's life. `vix` is the exception
and is not a defect: its worst is 2020-03-19, a real market move.

Cause: `masked_rolling_scale_col` accepted a median/IQR estimate from as few as
8 observed values, so a feed's opening fortnight set the scale for everything
after it. Separately, a coarse integer count (numobs, 30 distinct values in 19
years) yields a near-zero IQR in a quiet window that the next ordinary move
then divides by.

Change: `MASKED_MIN_OBS = 512` (from 8) and a causal running-maximum IQR floor
at `MASKED_IQR_FLOOR_FRAC = 0.01`; rows without a trustworthy scale emit the
neutral 0.0, the same treatment a dead feed already gets. Both constants are
set from the mechanism -- 512 is ~4% quantile error and about two months of a
session-limited feed -- not by searching for a value that passes the gate. No
threshold in `prep_invariants.py` was touched.

An emulation on EWMA-smoothed raw series showed the change helping two channels
(numobs 1.6e11 -> 189, stocktwits_attention 123 -> 10.6) and harming none, but
that emulation omits the diurnal adjustment and the real `adj_` construction,
so it is not evidence. The measurement that counts is a rebuild plus a real
invariants run.

## 4b. Baseline error: this session's designs dropped the calendar block

Recorded because it inflated a headline by a factor of three and conditions
every contrast measured in this session.

The paper's shared incumbent is **OLS HAR base-5 + calendar**, QLIKE 0.13415.
Calendar features were always in it. Every design fitted in this session is
`har(12) + lad(492) = 504` and excludes the nine calendar columns
(`DOW_0..DOW_4`, `hour`, `is_overnight`, `is_open`, `is_close`).

The interaction ladder therefore decomposes as:

| step | QLIKE | what it is |
|---|---|---|
| ridge-504, no calendar | 0.13146 | this session's baseline, weaker than the paper's |
| + cal | 0.13129 | repairs the omission -- not a finding |
| + har x cal | 0.13048 | the paper's regime block, already credited at -0.00225 |
| + exog x cal4 | 0.12987 | the genuinely new part |

So the novel increment is **0.00061**, not the 0.00159 first quoted against the
handicapped baseline. The 8-of-8 era stability stands; the magnitude claim was
a threefold overstatement of the same kind as several entries in section 4 --
measured against a base of our own construction rather than the paper's.

Consequence for everything else here: the exogenous block's 0.00425, the
winsorization contrasts, and the tree and kernel arms were all measured against
`har + lad` with no clock. Those contrasts are internally consistent, but none
is against the paper's actual incumbent and none may be quoted beside the
paper's numbers without restating its base.

**And it is worse than the calendar block alone.** The cache holds 1,077
columns; every design in this session used 504:

| block | count | used |
|---|---|---|
| `adj_*_ma_*` exogenous ladder | 492 | yes |
| **`*_avail_ma_*` availability indicators** | **492** | **no** |
| `*_active_ma_*` occurrence indicators | 48 | no |
| `har_ma_*` | 12 | yes |
| other, including 9 calendar | 33 | no |

Against the battery table: the paper's `ridge / all_features` is **0.12788**,
its LightGBM 0.12604, its dense elastic net 0.12530, its deployed `enetreg2`
0.12314. This session's `ridge-504` is **0.13146** -- 0.00358 worse than the
paper's plain ridge, which is larger than every effect measured here. The best
arm found, `+ exog x cal4` at 2,589 columns and 0.12987, is still worse than
the paper's ordinary ridge.

The omission is pointed. This session's central finding is that availability
bookkeeping was dishonest and that repairing it matters; the impute-and-
indicate encoding exists so the model can tell a real value from a filled one.
Every experiment here then dropped all 492 availability indicators.

**What this invalidates.** Not only the wins. The NULLS are equally affected --
"shock breadth adds nothing", "smooth nonlinearity adds nothing", "the channel
axis is not low rank" were each measured on a design missing the block that
encodes missingness, which is where breadth and regime information would sit.
No null here is evidence about the full design.

**Conventions added:**

1. Any arm compared to a published number is fitted on the published
   incumbent's feature set, or the difference in base is stated in the same
   sentence.
2. Before any arm is fitted, its column count is reconciled against the
   cache's total and the difference accounted for explicitly. `504 of 1,077`
   should have stopped this on the first run.

## 5. Order of execution

No step may start before the previous one passes.

1. Build the composed-fix cache.
2. `prep_invariants.py` across all caches — must show the composed cache
   clean on classes A, B and E.
3. `factorial_design.py` — the pre-registered design above.
4. `audit_paper_numbers.py` — every quoted number traces to a JSON.
5. Rewrite the paper, against results that have passed 1-4 and carrying the
   ledger in Section 4 as a published limitation rather than a private note.
