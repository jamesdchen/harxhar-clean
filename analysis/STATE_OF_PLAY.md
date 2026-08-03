# State of play

What is measured, what it rests on, and what the paper can be built with.
Companion to `PROTOCOL.md`, which holds the conventions, the pre-registered
design, the retraction ledger and the gate log. This document is the inventory.

> **UPDATE.** The re-race is done. Axes B, D and E have been re-run on the
> clean cache `b2_mmap_warm` and the pre-registered factorial has been read
> against H1-H4 (`PROTOCOL.md` section 3a). Headlines: exogenous data **works**
> (raw492 0.13146 vs backbone 0.13571, winning in all 8 eras); the
> stripped-down design is **falsified** by its own pre-registered condition
> (-0.00126, t -5.91); the tree-premium negative **replicates**; the
> supervised retrieval metric **reverses** from null to a win. H2 is
> **rejected**. Sections 1-2 below are kept as written at the time and the
> per-axis verdicts are updated in place.

## 0. The one thing to read first

**Every modelling number in this project ~~currently sits~~ *sat* on a cache
that has since been superseded.** The design comparisons were run on `b2_mmap` (pre-fix)
or `b2_mmap_fix` (which carries a double-scaling wiring bug). The clean cache,
`b2_mmap_warm`, was built after them, and the re-race on it is now complete
(see the update above).

That was not a footnote. The cache level **reversed the sign** of the headline
exogenous result: raw-492 loses to the backbone by +0.00046 on `b2_mmap` and
beats it by −0.00425 on `b2_mmap_warm`, against a backbone that is
byte-identical across both. Every design conclusion in this project had to be
re-derived on the clean cache before it could be quoted, and one of them
(the stripped-down design) did not survive.

### The four caches

| cache | what it is | status |
|---|---|---|
| `b2_mmap` | original | defective: availability recorded after the fill, voldemand at 2260 rolling IQRs |
| `b2_mmap_fix` | availability + masked scaling, wired wrong | defective: double scaling, effspread at 933 |
| `b2_mmap_indonly` | composed fix, correct wiring | defective: feed-initiation transient, worst channel 336 |
| `b2_mmap_warm` | + warm-up guard and IQR floor | **current**, worst channel 96, invariants A 0 / B 1 / D 7 |

---

## 1. Clean separation

### WORKS — validated, footing current

| finding | evidence |
|---|---|
| **The prep defect is real and large** | voldemand's feed terminates 2023-08-31; October 2023 print rate exactly 0.0000; the unlimited fill carried a dead value for 8,453 bars. ridge492 on that window: 6.37170 → 0.12391 |
| **Availability bookkeeping was dishonest and is now honest** | class-A violations 31 → 0. Indicator 0.6983 → 0.2224 against a true print rate of 0.2197 |
| **The publication calendars are coherent** | every channel group matches its session: 22 equity channels 0.6233 vs implied 0.6667; index bars 0.9800 on 24h; vix family 0.21–0.35 on 10:00–16:30 |
| **Three more feeds terminate inside the evaluation window** | vix, vix3m, vvix all stop 2024-02-12, ~2,746 dead bars each. Newly identified, not previously accounted for |
| **The scale pathology is bounded** | worst channel across the panel 2260.4 → 96.2 |

These do not depend on which model wins. They are properties of the data and
the pipeline.

### RESOLVED — the re-race, on `b2_mmap_warm`

| result | outcome |
|---|---|
| exogenous data helps | **yes**: raw492 0.13146 vs backbone 0.13571, and it wins in all 8 eras. On the pre-fix cache it lost |
| the stripped-down design | **falsified** by the pre-registered condition: -0.00126, t -5.91, p 3.4e-09 |
| lag rank | **saturates at r=4**; both bases agree and are flat over r∈[4,12] (range 0.00045). The r=3 dip is **basis-specific** — power 0.13117 vs log-poly 0.13246 — so it is about *which shapes*, not *how many dimensions* |
| tree premium reachable by an RBF kernel | **no**, replicates: gain 0.00098 vs premium 0.00184-0.00358 |
| supervised retrieval metric | **reverses**: null (+0.00015, Holm p 1.0) -> beats 516-dim ambient by -0.00540, t -8.9 |
| unsupervised retrieval metrics | still lose: PCA +0.00552, Laplacian +0.00450 |
| H2 (prep beats any design choice) | **rejected**: cache main effect 0.00067 vs design 0.00311 |

### DEAD — retracted

Fourteen claims in `PROTOCOL.md` §4. Eleven died from one of two causes:
selection touching the reported loss, or an unvalidated proxy standing in for
forecast harm. Neither is a modelling error, which is why the protocol targets
those two specifically.

---

## 2. Every idea, by the axis it was attacking

Footing column: `pre` = `b2_mmap`, `fix` = `b2_mmap_fix`, `warm` = clean.
Lower QLIKE is better throughout. **Numbers are only comparable within an
axis** — the axes use different estimands and different evaluation sets.

### Axis A — Lag weighting: how the 12 rungs combine

Backbone-only, no exogenous data. The estimand is the effective kernel
φ(u) = Σ_{r_k ≥ u} w_k / r_k.

| idea | QLIKE | footing | verdict |
|---|---|---|---|
| boxcar | 0.26492 | pre | baseline |
| spline-192 | 0.26355 | pre | no gain over MIDAS |
| MIDAS / Beta, validation-selected | 0.26337 | pre | PENDING |
| spline shrunk toward Beta centre | 0.26181 | pre | PENDING — beat the claimed "ceiling" |
| **tapered power law, β=1.5, u_c=8** | **0.25869** | pre | PENDING — best on this axis, t −3.79 vs Beta centre |
| greedy / POD reduced basis | — | pre | DEAD as a width story; the gap was criterion mismatch, not representation |
| manifold-distance penalty | 0.26291 | pre | DEAD — loses to the point prior |
| power law β=1.033 / β=1.506 / exponential | — | pre | DEAD — λ\*=0 or harmful |

*Reading:* the only thing that ever helped on this axis is a prior centred on a
sensible shape, and tapering it. The elaborate machinery (reduced bases,
manifold penalties, kernel families) contributed nothing.

### Axis B — Cross-channel structure: how 41 channels share strength

Full panel, 218,909 scored bars.

| idea | cols | QLIKE | footing | verdict |
|---|---|---|---|---|
| backbone (HAR only) | 12 | 0.13571 | pre & fix | reference — identical across caches |
| raw 492 (ridge) | 504 | 0.13617 | **pre** | loses to backbone |
| raw 492 (ridge) | 504 | **0.13146** | **warm** | **LIVE — wins by 0.00425, all 8 eras** |
| amplitude rank-1 | 53 | 0.13280 | pre | superseded |
| amplitude rank-1 | 53 | **0.13285** | **warm** | LIVE |
| amplitude rank-2 | 94 | **0.13199** | **warm** | LIVE |
| **power basis r=3** | 135 | **0.13117** | **warm** | LIVE — beats raw492 by 0.00029 (t −3.00), validation-selected r. But **basis-dependent**: log-polynomials at r=3 give 0.13246 |
| power basis r=8 | 340 | 0.13145 | **warm** | LIVE — indistinguishable from raw492's 0.13146 |
| rank sweep r=4..12, both bases | — | flat, range 0.00045 | **warm** | LIVE — rank beyond 4 buys nothing |
| shaped 492 | 504 | 0.13415 / 0.13487 | pre / fix | PENDING |

Rolling operator variants on the 41×12 block (different evaluation, not
comparable to the rows above):

| penalty | QLIKE | verdict |
|---|---|---|
| **shared shape** | 0.14878 | best of the family |
| backbone | 0.15242 | |
| ridge | 0.17906 | |
| pooled | 0.18907 | |
| smooth | 0.21664 | |
| rank-1 | 0.30711 | worst |

And how the lag direction is obtained:

| source of the direction | QLIKE |
|---|---|
| **imposed rung^−1.0** | **0.14878** |
| imposed rung^−0.5 | 0.14980 |
| rank-2 imposed | 0.15042 |
| in-window SVD | 0.17112 |
| rank-2 frozen | 0.19943 |
| frozen SVD | 0.20902 |

*Reading:* imposing the shape beats estimating it, by a wide margin. The
in-window first singular vector has consecutive |cos| of 0.893 on average but a
**minimum of 0.002** — it flips. That instability is the whole story of this
axis, and it is the strongest surviving argument for the amplitude construction.

### Axis C — Penalty and prior geometry

| idea | result | footing | verdict |
|---|---|---|---|
| primal–dual identity β′Pβ ⟺ k(x,x′)=x′P⁻¹x′ | verified to 3.1e-15 | — | **WORKS** — apparatus, not a claim |
| effective df = tr((G+λP)⁻¹G) | ridge ~95 df vs shape ~54 | fix | **WORKS** as a diagnostic |
| where ridge spends its df | PC1-5 4.94, PC6-20 14.12, **PC21+ 72.08** (shape: 0.30/0.13/0.07) | fix | **WORKS** — the sharpest single diagnostic in the project |
| df into the failing era | 95.4 → 104.7, flat | fix | **WORKS** — rules out overfitting, points at distribution shift |
| directed shrinkage | — | pre | DEAD — twice: first re-read as tail control, then as a data artifact |
| amplitude exponent β as a lever | 29/36 points within 0.0005 over β ∈ [0.20, 3.00] | fix | **WORKS** as a negative: it is not a lever |

*Reading:* this axis produced no wins and the project's best diagnostics. The
PC21+ number explains *why* the raw parameterisation was fragile without any
appeal to column count.

### Axis D — Nonlinearity

| idea | cols | QLIKE | footing | verdict |
|---|---|---|---|---|
| amp + RFF 128 | 181 | **0.13240** | **warm** | LIVE |
| amp + RFF 512 | 565 | **0.13187** | **warm** | LIVE |
| gain from nonlinearity | — | **0.00098** | **warm** | vs a tree premium of 0.00184–0.00358 — replicates |

*Reading:* a clean **negative**. An RBF kernel with 512 random features on the
41 amplitude coordinates cannot reach even the *linear* raw-492 model, let alone
the tree premium. "Trees win" does not restate as "the linear model was in the
wrong space" — not this space. Worth publishing as a closed direction.

### Axis E — Retrieval metric geometry (kNN, W=24, k=100)

| metric | QLIKE | vs ambient | verdict |
|---|---|---|---|
| **path-only** | **0.16347** | −0.01476, t −15.5 | LIVE (warm) — the best arm |
| path + amplitudes | 0.16667 | | PENDING |
| ambient | 0.17823 | reference | |
| amplitudes β=0.5 / 1.0 / 1.5 | 0.17266 / 0.17299 / 0.17300 | | PENDING |
| **operator SVD d6 (supervised)** | **0.17283** | **−0.00540, t −8.9** | LIVE (warm) — REVERSED from null |
| SVD coordinates | 0.17593 | | |
| PCA d6 | 0.18375 | +0.00552, t +12.8 | worse |
| Laplacian eigenmaps d6 | 0.18274 | +0.00450, t +9.8 | worse |
| anchor | 0.18322 | | worse |

*Reading (revised after the re-race):* the earlier blanket negative was wrong,
and instructively so. **Unsupervised** reduction still loses — PCA +0.00552,
Laplacian +0.00450 at d=6. But the **supervised** projection onto the operator's
singular triplets now *beats* the 516-dimensional ambient view by 0.00540
(t −8.9) in six dimensions, having been an exact null on the corrupted panel. A
distance weights every coordinate by its scale, so a channel standardised by a
collapsed denominator dominated the metric outright; the supervised projection
was spending its budget on an artefact. The plain lag path is still the single
best arm at 0.16347.

### Axis F — Selection methodology

| rule | QLIKE | verdict |
|---|---|---|
| A — fit SSE | 0.26640 | wrong objective |
| B — fit QLIKE | 0.26484 | |
| **C — validation QLIKE** | **0.26337** | **WORKS** — the honest rule |
| D — test QLIKE | 0.26321 | ORACLE, not achievable |

Objective mismatch (A − B) = **+0.00157**. Selection optimism (C − D) =
+0.00017.

*Reading:* choosing the criterion correctly is worth roughly ten times more than
the residual selection optimism, and more than most of the modelling ideas in
Axis B. **WORKS** — and it is a methodological contribution in its own right.

### Axis G — Data representation and prep

| intervention | effect | verdict |
|---|---|---|
| availability recorded before the fill | class A 31 → 0 | **WORKS** |
| bounded fill (`FFILL_LIMIT = 26`) | kills an 8-month propagation of a dead feed | **WORKS** |
| neutral median alone | effspread 17.6 → 932.6 | harmful alone |
| masked scale estimation alone | voldemand unchanged | inert alone |
| **the two composed** | voldemand 2260.4 → 6.7 | **WORKS** — the interaction is the whole effect |
| warm-up guard + running-max IQR floor | worst channel 336.5 → 96.2; class B 11 → 1 | **WORKS** |
| ±20 IQR clip | not implemented, by instruction | — |

*Reading:* the largest effects in the entire project are on this axis, by three
orders of magnitude on the worst window and roughly ten times pooled.

---

## 3. What the paper can be built with

### The spine

1. **A defect in exogenous data preparation, and what it cost.** Feed
   termination carried forward by an unlimited fill, established from
   publication calendars rather than inferred from magnitudes. 6.37 → 0.12
   QLIKE on the affected window.
2. **Why it was invisible.** The pipeline never compared its own bookkeeping
   against the raw inputs. One assertion — indicator mean vs raw print rate —
   catches it. This generalises beyond this dataset and is the most portable
   thing here.
3. **The fix, and that it is an interaction.** Neither half works alone; one is
   actively harmful alone. Three prior attempts failed because each shipped one
   half.
4. **Where the signal actually lives.** Axis B's lag-rank result plus Axis C's
   PC21+ diagnostic — *pending re-race*.
5. **A closed negative.** Axis D: the tree premium is not smooth in amplitude
   space.

### Supporting, publishable as methodology

- Axis F: selection criterion worth +0.00157, more than most modelling effects.
- Axis C: effective df as the right way to count a penalised design's capacity.
- The retraction ledger itself, published as a limitations section. Fourteen
  withdrawn claims is the strongest available evidence that what remains was
  actually tested.

### What must NOT go in

- ~~Any Axis B design winner, until re-raced.~~ Resolved: raw492 wins on the
  clean cache, in all 8 eras, and rank-3 matches it at a quarter the width.
- The amplitude design as a centrepiece. Promoted twice, lost both times, and
  now formally falsified by its own pre-registered condition. It survives as a
  *diagnostic* about lag-direction instability, not as a recommended estimator.
- The unconditional "data representation dominates" claim. H2 was rejected:
  pooled main effects are 0.00067 (cache) vs 0.00311 (design). Only the
  conditional form is supported.
- Any table mixing the 0.26xxx (Axis A, backbone-only) and 0.13xxx (Axis B, full
  panel) families. Different estimands.

---

## 4. Immediate next steps, in order

1. ~~Re-race Axis B and Axis D on `b2_mmap_warm`.~~ **Done** — see the update
   at the top and `PROTOCOL.md` section 3a.
2. ~~Re-run Axis E's retrieval race on the clean cache.~~ **Done** — it
   reversed.
3. Re-estimate the `vol_demand` bucket effect (−0.00108, t −4.1) on observed
   bars only — it is currently suspended, being the very channel the defect was
   in.
4. Characterise the one residual class-B violation (`vix`, ratio 0.0000, \|max\|
   10.0) — benign on magnitude, but not waived.
5. Only then, rewrite.

## 5. Open and unexplored

- Nyström with r ≈ df landmarks.
- Eigenvectors of K = XP⁻¹X′ on the observation side.
- Era-by-era read of amplitude rank-3, whose λ\* = 0.1 against 100 for ranks 1–2
  and worst-in-table MSE suggest it trades tail calibration for bulk accuracy.
- Whether the three newly-found feed terminations (2024-02-12) change block 8.
