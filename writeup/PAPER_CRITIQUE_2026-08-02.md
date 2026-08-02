# Critique of the current paper draft (2026-08-02)

Scope: `writeup/main.tex` + `writeup/sections/*.tex` as of `0155319`, checked
against the evidence actually in the repo (`writeup/metrics_table_causal_tune_plus_spectral.csv`,
`writeup/causal_tune_battery_table.tex`, `writeup/paper_restructure_2026-08-01.md`).

Bottom line: the **abstract, introduction and conclusion are a strong paper**
— honest, unusually well-disciplined about negative results, and the headline
numbers in them do reconcile with the 98-arm battery CSV. The **body is a
different, older paper**, and the two contradict each other on the single
claim the new framing is built around. Nothing here is a fatal scientific
error; the problems are (a) assembly, (b) a benchmark that is knowingly
beatable, and (c) inference that is claimed but not run.

---

## 1. Blocking: the document does not compile as described

`main.tex:35-38` inputs four section files that do not exist:

```
\input{sections/descriptive_analysis}   % missing
\input{sections/marginal_contribution}  % missing
\input{sections/linear_vs_nonlinear}    % missing
\input{sections/algorithm_design}       % missing
```

What *does* exist in `sections/` — `data.tex`, `methodology.tex`,
`results.tex`, `discussion.tex`, `related_work.tex`, `master_table*.tex`,
`tree_story.tex` — is never inputted. So the compiled artifact is
**abstract + introduction + conclusion**, with every `\ref` in those three
files dangling (`sec:descriptive`, `sec:marginal`, `sec:linvsnonlin`,
`sec:algorithms`, `tab:model_comparison`, `tab:ridge_har_subgroup`).

Consequences worth naming explicitly:

- Every quantitative claim in the abstract is currently **unsupported inside
  the document** — the tables that back them (`causal_tune_battery_table.tex`,
  `master_table.tex`) are not in the build.
- `related_work.tex` is a two-line stub. The introduction picks a fight with
  `christensen2023machine` and `gu2020empirical` and then never engages the
  literature.
- The abstract promises Diebold–Mariano and MCS inference throughout; the
  build contains **zero DM statistics and zero MCS results**.

This is the first thing to fix, because several items below are only
*apparently* contradictions — they're really "two drafts in one directory".

---

## 2. The body contradicts the front matter on the paper's central claim

The new framing (abstract, intro §4, conclusion ¶3) says:

> "the edge is not a tuning story — causally tuned hyperparameters trail
> fixed defaults" … "tuning is a non-lever from every angle we measured"

`sections/results.tex:33-104` and `sections/discussion.tex:7` say the exact
opposite, at length and with a table:

> "The default-hyperparameter tree results would, on their own, support a
> reading that linear HAR is uniquely well-suited… After tuning, that reading
> disappears… tuned LightGBM beating ridge outright by 0.028 units —
> reversing the apparent ordering that the default configuration suggested."

`tab:model_comparison` reports tuned LightGBM at QLIKE **0.1099**, a number
that appears nowhere in the current evidence base and is ~0.024 better than
the best arm in the 98-arm battery. The reason is stated in the old
methodology and never reconciled: `methodology.tex:162` —

> "The objective is to minimize QLIKE (Section 4.3) on the **full walk-forward
> backtest**."

That is hyperparameter selection on the evaluation panel. `discussion.tex:12`
half-admits it ("the tuning objective is identical to the evaluation metric…
the ranking between models is the defensible quantity") but the ranking is
*precisely* what such selection corrupts — 125 XGBoost trials vs 30 LightGBM
vs 16 RF means the models are ranked partly by how many draws each got from
the test set. The whole `results.tex` §5.2 / `discussion.tex` §6.1 narrative
is an artifact of that protocol, and the causal-tuning battery was built to
replace it.

**Fix:** delete or quarantine `results.tex` §5.2, `discussion.tex` §6.1–6.2,
and `tab:best_tuned_params` / `tab:model_comparison`. Keep the leak as a
*methodological result* if you want it — "tuning on the evaluation panel
manufactures a 0.03 QLIKE model-class reversal" is a genuinely useful
cautionary finding, and it directly supports the intro's claim that published
disagreements are "mechanical". But it cannot sit in the paper as a result.

---

## 3. "Bar-for-bar on the identical panel" is true of the battery, false of the body

Verified: all 98 rows of `metrics_table_causal_tune_plus_spectral.csv` carry
`n = 218,934`. Good — the abstract's claim holds for the new evidence.

But the tables currently in the document do not. `tab:ridge_har_subgroup` and
`tab:ridge_pca_subgroup` compare rows with N = 221,915 / 222,058 / 217,214 /
203,433 / 200,687 / 138,279 / 136,849, and compute ΔQLIKE **across
different panels** — the Sentiment and Vol Demand rows lose ~38% of the
sample relative to the MA baseline they are differenced against. The
sentiment/vol-demand rows in particular span a different calendar era
(StockTwits and options-flow coverage start late), so their ΔQLIKE is
partly a regime effect. `results.tex:6-8` waves at this ("Differences in N
across subgroups reflect both variable availability windows and warm-up
periods") but the table then does arithmetic across them anyway.

Also inconsistent: the naive/MA baseline is 0.3518 in `results.tex`, 0.1955
in `master_table.tex`, and the shared incumbent is 0.13415 in the front
matter. Three benchmarks, all called "the baseline".

---

## 4. The shared incumbent is beatable by a one-line change, and this eats the headline effects

This is the most substantive scientific issue, and the restructure blueprint
already flags it (`paper_restructure_2026-08-01.md:131` — "PANEL
UNIFICATION (the blocker)"). Spelling out the damage:

The incumbent is OLS on the **base-5** HAR ladder, QLIKE 0.13415. The same
CSV contains 21 ladder arms. The best is base-2:

| arm | QLIKE | DM vs incumbent |
|---|---|---|
| `ols_har b2 cap3125` | 0.133323 | −10.58 |
| `ols_har b2 cap240`  | 0.133330 | −10.51 |
| incumbent (b5)       | 0.134153 | — |

So the benchmark is **0.00082 worse than an OLS model differing only in the
lag grid**, with DM ≈ −10.6. Now re-read the headline effects against a
correctly specified linear benchmark:

- **"Pure nonlinearity premium of 0.00297"** (abstract; = baseline LightGBM
  0.13118 vs incumbent 0.13415). Against `b2` OLS it is **0.00215** — about
  **28% of the claimed premium is HAR lag-grid quadrature error, not
  nonlinearity.** And the comparison is anyway asymmetric: a *causally tuned*
  LightGBM against an *untuned* OLS. The matched linear arm on the same
  features (baseline ridge causal-tuned, 0.13445) is *worse* than the
  incumbent, so "nonlinearity premium" is measured against whichever linear
  arm happens to look best.
- **Every bucket ΔQLIKE** in §2 of the paper is inflated by the same 0.00082,
  which is 45% of the tree-vs-linear gap the paper calls its second headline.

The blueprint's recommendation (re-run the bucket table on the b2 backbone)
is the right call and should be treated as a **correctness prerequisite, not
a polish item**. Until it lands, the honest framing is: "all exogenous and
model-class gains are reported against a base-5 OLS incumbent that a base-2
OLS beats by 0.0008."

---

## 5. Two claims that the repo's own data contradict

**(a) `conclusion.tex:10` — the spectral-kNN arm "trails every causally
tuned parametric arm."** It does not. The winning arm (0.130964) beats
**25 of the 45** causally tuned arms, including baseline LightGBM (0.13118),
baseline XGBoost (0.13162), baseline ridge/EN/Lasso (0.13445/0.13470/0.13474),
and every ridge arm outside `all_features`. The defensible statement is that
it trails the best arms on the informative buckets (`all_features`,
`moments`, `liquidity`), which is a weaker and more interesting claim.

**(b) Column arithmetic doesn't reconcile.** `introduction.tex:6` says 41
exogenous predictors; `introduction.tex:8` says a 529-column design matrix;
`conclusion.tex:6` says a "359-column exogenous block". `methodology.tex:88`
states the rule: 6 rolling-mean features per exogenous variable → 41 × 6 =
246, not 359, and 529 − 359 = 170 ≠ 6 HAR + calendar. The paper needs one
explicit table: channels → lag expansion → design columns, per bucket.

---

## 6. Methodology issues that need to be argued, not just described

Ordered by how much they could move a referee.

**Winsorization of the target at rolling 5th/95th percentiles**
(`methodology.tex:53-64`). By construction this clips **10% of observations**,
and on a heavy-tailed volatility target the clipped mass is exactly the
spikes the forecast exists to catch. Models are trained on a clipped target
and evaluated on the unclipped one, and the paper acknowledges the transform
is not invertible (`methodology.tex:24`, `:237`) without quantifying the
resulting bias. At minimum: an ablation at 1st/99th and at no winsorization,
on the shared panel. A referee will ask whether the "dense but weak" finding
survives when the tails are not truncated — plausibly the strongest signal
lives there.

**The diurnal divisor does forecasting work.** Stage 1 divides by a *rolling*
20-observation same-slot mean (`eq:diurnal_mean`), not a fixed seasonal
profile. That is a slowly adapting level estimate, i.e. a forecast, and it is
multiplied back in at evaluation (`eq:duan`). Every model therefore inherits a
common, non-trivial predictive component, which inflates all QLIKE-vs-naive
and OOS-R² numbers and compresses differences *between* models. The paper
should report what the divisor alone achieves as a standalone forecast.

**500-observation training window.** `methodology.tex:217` — ~10.4 trading
days, against a 529-column design. That is p ≈ n. The old §5.2 then selects
XGBoost at depth 6 / 300 trees on it. Note the causal-tuning battery uses a
24,000-bar window, so the paper contains two incompatible window conventions;
only one should survive.

**Refit-cadence asymmetry** (every bar for linear, every 5 for trees) is
noted as a caveat in `discussion.tex:12` but it confounds the model-class
comparison in the direction of the paper's conclusion in the old results and
against it in the new battery (where "model refit is every bar throughout").
State which convention the shipped numbers use, once.

**Duan's smearing** (`eq:duan`, `methodology.tex:239`) is written as the analytic correction
ŷ² + σ̂², not the empirical smearing average over residuals — fine, but then
don't call it Duan's estimator without a line of derivation, and note σ̂² is
estimated on the winsorized scale while the evaluation target is not
winsorized.

**QLIKE description** (`methodology.tex:254`): "strictly consistent for the
conditional mean under the assumption that realized volatility is an unbiased
estimator of integrated variance" is a garbled version of Patton (2011). The
point is robustness of the *ranking* to noise in the volatility proxy.

---

## 7. Inference: claimed but not run

- **No pairwise DM anywhere.** The entire model-class conclusion rests on
  0.12604 (LightGBM) vs 0.12788 (ridge), both compared only to a *shared third
  model*. From DM t = −15.4 and −19.9 against a common benchmark you cannot
  infer that 0.00184 is distinguishable from zero. `conclusion.tex:14` admits
  this ("pairwise Diebold–Mariano tests and a model confidence set over the
  45 arms are pending") — but the abstract and intro state the model-class
  ordering as established. **This is the single largest gap between claim and
  evidence in the draft**; one MCS run over the 45 arms closes it.
- **No multiplicity control.** 98 arms, all DM'd against one benchmark, and
  the minimum is quoted as the result. For the spectral-kNN section
  specifically: **32 kNN cells** were run and the best (0.130964, DM −17.6) is
  the headline. That is a winner's-curse number, selected on the same panel it
  is evaluated on. Either report the DM for a pre-registered cell or add a
  holdout confirmation.
- **The 90% MCS "singleton"** claim is stated in the abstract and conclusion
  with no table. With n = 218,934 an MCS collapsing to a singleton is close to
  automatic; report the p-values and the elimination order or the claim reads
  as an artifact of sample size.
- **Metric disagreement is unreported.** In the battery, `lgbm/all_features`
  has OOS-R² +0.059 while `ridge/all_features` is **−0.058** and
  `xgb/all_features` (rank 2 on QLIKE) is **−0.062**. So under MSE the
  second-best QLIKE arm is *worse than the incumbent*. The table caption
  dismisses MSE as "dominated by a small number of COVID bars" — that is a
  post-hoc metric choice defending the paper's own ordering, and it should be
  faced in the text with a COVID-excluded MSE column rather than in a caption.
- **Mincer–Zarnowitz β runs 0.84–0.94 for every arm** (t for β=1 down to
  −162 in `master_table.tex`). Every model in the paper is systematically
  miscalibrated in the same direction, and the better-QLIKE arms are the
  *more* miscalibrated ones (lgbm β=0.891 vs incumbent β=0.929). That is a
  substantive result about what QLIKE is rewarding here, and it is currently
  a silent column.

---

## 8. Framing and overclaiming

- **The abstract sells the spectral-kNN result by omission.** "beats the
  incumbent (QLIKE 0.13096, DM −17.6)" is true; what it omits is that the
  winning configuration has **embedding dimension 0** — i.e. the "spectral"
  step is switched off, the method is a residual-anchored kNN, and the novel
  component is the one that fails. The conclusion states this honestly; the
  abstract should too, in the same sentence. As written a reader takes away
  "novel manifold estimator wins".
- **"Nearly two orders of magnitude more skill per feature"** (abstract).
  Per-feature normalization is doing all the rhetorical work here: dividing a
  block's *unique* (post-FWL) contribution by its column count penalizes
  redundancy twice. The underlying fact (−0.00122 for one engineered feature
  vs −0.00449 for 359) is interesting; the normalization inflates it.
- **"Established three independent ways"** (abstract, of
  information > model class > hyperparameters). The three routes share a
  panel, a preprocessing pipeline, a benchmark and a loss. They are three
  *measurements*, not three independent confirmations.
- **The effect sizes deserve an economic-significance sentence.** The whole
  paper lives in 0.126–0.134 QLIKE, i.e. ~6% relative improvement over a
  benchmark, and no result is translated into anything a user of the forecast
  would notice. Given the paper's own honesty commitments, one paragraph on
  "is 0.00184 worth anything" belongs in the conclusion.

---

## 9. Missing scholarship and reproducibility

- `related_work.tex` is empty. Minimum expected engagement: HARQ /
  measurement-error HAR (Bollerslev–Patton–Quaedvlieg), realized GARCH,
  HAR-J/CHAR, intraday-periodicity literature (Andersen–Bollerslev), and the
  ML-for-vol papers the intro already cites. The Hawkes framing in the
  restructure blueprint also demands Bacry–Muzy–Hawkes-for-volatility.
- No code/data availability statement, no seeds, no software versions.
- The old body is written against **in-flight results** ("at the time of
  writing, the completed trial counts are 125/30/16", "[to be filled when RF
  trials complete]"). Whatever survives must be frozen.
- Section 5.2's PatchTST results are promised in `results.tex:4` and never
  delivered.

---

## 10. What is genuinely good, and should not be lost in the rewrite

Stated plainly so the rewrite doesn't sand it off:

1. **The negative results are the contribution.** Manifold embedding hurting
   on every matched cell, with a *measured mechanism* (Nyström extension error
   at the embedding scale), is better science than most positive results in
   this literature. Same for tuner-trails-default and the oracle-penalty
   ceiling of 0.00015.
2. **The causal-tuning protocol** (fit block / 25-bar embargo / 125-bar
   validation tail, re-selected every 250 solves) is a real methodological
   asset and is described precisely in `causal_tune_battery_table.tex`. It
   should be a numbered subsection with its own figure, not a table caption.
3. **The one-panel discipline in the battery is real** (verified: all 98 arms
   at n = 218,934) — it just needs to be extended to the sections that
   currently violate it.
4. **The h=1 target is immune to the multi-horizon label leak** documented in
   `SESSION_HANDOFF_2026-08-01B.md:10-16`. Say so explicitly in the paper; a
   referee who reads the follow-up work will otherwise wonder.

---

## 11. Prioritized fix list

**P0 — correctness / assembly**
1. Make `main.tex` build: either write the four missing sections or repoint
   the inputs at the existing files.
2. Remove the test-set-tuned results (`results.tex` §5.2, `discussion.tex`
   §6.1–6.2, `tab:model_comparison`, `tab:best_tuned_params`) or rewrite them
   as a documented cautionary finding.
3. Re-run the bucket table on the base-2 backbone, or restate every ΔQLIKE
   against `b2` and revise the 0.00297 premium to 0.00215.
4. Fix `conclusion.tex:10` ("trails every causally tuned parametric arm" —
   false, it beats 25/45).
5. Reconcile 41 / 359 / 529 with one feature-accounting table.

**P1 — inference**
6. Pairwise DM + MCS over the 45 causally tuned arms; report whether
   0.12604 vs 0.12788 survives.
7. Multiplicity statement for the 98-arm battery; holdout or pre-registration
   for the spectral-kNN headline cell.
8. Add a COVID-excluded MSE column and address the QLIKE/MSE sign flip in
   text; discuss the MZ-β pattern.

**P2 — robustness a referee will demand**
9. Winsorization ablation (5/95 vs 1/99 vs none) on the shared panel.
10. Standalone forecast skill of the diurnal divisor.
11. Single training-window convention (500 vs 24,000) and a single refit
    cadence.

**P3 — presentation**
12. Write `related_work.tex`.
13. Abstract: state the embedding-dim-0 fact in the same sentence as the
    kNN win; drop or defend "per feature" and "three independent ways".
14. Economic-significance paragraph; reproducibility statement.
