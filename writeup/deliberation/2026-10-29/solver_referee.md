# Solver: JFEc referee and editor lens (29 Oct 2026)

Independent solver for the fresh version's conflicts C1–C7 and §8.9 Q8. I did not read the other solvers' files.
Read: `writeup/INTEGRATION_PLAN_2026-10-26.md` (D1–D20, budget), `writeup/fresh/paper_fresh_beats.tex` (7 `% CONFLICT`),
`writeup/beats/top/*.tex`, the live `rehaul/01,06,08,07,A_results,A_fi,99_author_notes`, the generated tables and
macros, `results/beyond_trees_2026-10-27/linear_pen/SUMMARY.md`, `results/channels_2026-10-26/factor_pen/SUMMARY.md`,
`results/beyond_trees_2026-10-27/linear/ADDENDUM_pen_split.md`, `results/tree_shap_w3000_2014start_fomcfix_2026-10-26/`,
`SESSION_HANDOFF_2026-10-26.md`, `rehaul/drafts/MAP_06_07.md`, and the commit log of 26–28 Oct (author quotes in commit
messages). I found no literature note on ablation presentation in `writeup/` or `results/`. I checked the closest JFEc
precedents directly (sources at the end).

## Verdicts

| Item | Answer | Tier | Conf. |
|---|---|---|---|
| C1 | Keep the every-bar full-set gain against the OLS-HAR as the first result (D1, D2, D7), but state D18's two-level question in the research-question beat. Order the contributions: gain and MSE caveat, then the two channels, then what lies beyond HAR + RM + FOMC. Scope "beats augmented HAR models" (see F1). | T2 | 0.75 |
| C2 | One every-bar Table 1. For every row, including the trees' HAR-set rows, report % and DM against the OLS-HAR. Keep the within-class DM. Mark 90% MCS membership. Move the 16:00 columns of `tab:rh_oos_vsols` to `app:rh_closing`. | T0 (tests) / T1 (layout) | 0.90 |
| C3 | Compare each class with its own base. Claim that survives: with the estimator held fixed, the 36 series add no significant accuracy to any penalized linear model at every bar or on ordinary days. The cross-estimator numbers (−2.62%, +2.74%) measure penalization, not series. Disclose that OLS on HAR + RM + FOMC is the best linear every-bar forecast. | T1 | 0.90 |
| C4 | No ladder figure in the main text. D20's approved map already puts `fig:rh_ladder` in the appendix (F.6), so D20 and D16 supersede the budget row. Text cites the every-bar rungs by macro. | T0 | 0.90 |
| C5 | Add a D18 grouping to the SHAP generator: HAR + calendar + RM + FOMC against the six remaining groups, with the same bootstrap. Keep the pre-registered standard / non-standard split in the appendix. | T0 | 0.90 |
| C6 | Not incoherent. The author decided the application with the D16 constraint in view. Add one pointer sentence, with no 16:00 numbers, to the appendix that reports the 16:00 accuracy and its relation to the Sharpe ratios. | T1 | 0.85 |
| C7 | The closing-interval models go to the appendix. D16 (later on 26 Oct), the plan's own appendix list and the author notes all place them there. D3's corralling now governs only the appendix section's content. | T0 | 0.93 |
| Q8 | Keep the current bullets as the first L2 draft; do not reset. For L1, use the fresh version's claim-form top beats, after correcting its six lock or evidence errors. The current top layer is mostly headings. | T2 | 0.70 |

No item needs T3 in my reading. C1 would become T3 only if the adjudicator does not accept the author's 28 Oct in-place
abstract edit as closing the open item "Headline under D18". A fallback question is given under C1.

---

## C1: Introduction framing

**Answer.** These are not two competing leads. D1 fixes the target (every bar), D2 the benchmark (OLS-HAR) and D7 the
MSE clause. D18 fixes the question ("what the other exogenous series add beyond HAR + RM + FOMC"). All four can be met:

1. The research-question beat states D18's two levels: (a) do exogenous series improve every-bar forecasts against
   the OLS-HAR, and (b) what do the series other than the well-studied ones add?
2. The first result is the full-set gain against the OLS-HAR, −6.0 to −9.0% (linear) and −11.2 to −14.9% (trees),
   6 of 6 significant. The D7 clause follows: MSE is higher in 6 of 6, significant in 0, and 5 crash days carry it.
3. Contribution 2 gives the two channels. Statement days are 3.1% of bars but carry 49–77% (linear) and 60–64%
   (trees) of the gain. On ordinary days the realized moments carry 77–88% of the trees' gain.
4. Contribution 3 is the D18 answer. HAR + RM + FOMC reaches 89–101% of each class's gain. Beyond it there is no
   linear gain, and the boosted trees gain −1.6% / −1.8%, sparse across a few series from several groups.
5. Contribution 4 is the application.

This is the order of both the current abstract and the fresh version. The real defect is a claim-scope problem, not the
order (F1).

**Referee logic.** A referee who reads a 6–15% headline and then learns that 49–77% of it sits on 81 scheduled FOMC
days will call the headline inflated, unless the decomposition is in the abstract's next sentence. D18's own words
("those are well studied") are the referee's objection made in advance. Leading the contributions with the
decomposition and the beyond-base answer is what makes the full-set number defensible.

**Evidence**
- `INTEGRATION_PLAN_2026-10-26.md`:36–37 (D1, D2), :43 (D7), :54 (D18), :55 (D20).
- `SESSION_HANDOFF_2026-10-26.md`:41 lists "Headline under D18 … once the add_rm_fomc numbers are in" as an open
  author decision.
- Later author actions that close it:
  - ee3838e (27 Oct, author): "Section 6.2 rewritten around one statement: two groups … carry the gain; what remains
    is small, trees only, a few series".
  - 3c6ccc4 (28 Oct, author's alternative-data edit): the abstract clause becomes "beyond them, only boosted trees
    gain, slightly". It is the third sentence, after the full-set gain, which the author left first
    (`paper_rehaul.tex`:278–284).
- Numbers:
  - `rhchLinShareEventMin–Max` = 49–77, `rhchTreeShareEventMin–Max` = 60–64, `rhchEventBarPct` = 3.1.
  - `rhabBaseShareMin–Max` = 89–101.
  - `rhabLgbm/XgbFullVsAddRmFomcAllPct` = −1.6 / −1.8 (DM −4.92 / −5.26).
  - `rhcaNMseHigherSigBarSpan` = 0.

**If adopted**
- `rehaul/01_introduction.tex`: the research-question beat is rewritten to the two-level form.
- Contribution 1 is retitled. Drop "beyond richer HAR models" (line 100), which F1 refutes for the linear classes.
- Contribution order becomes gain, channels, beyond the base, application.
- The fresh version's `% CONFLICT` at lines 240–242 is resolved, and the fresh intro already has this order.
- "Under squared error the comparison reverses" (`06_results.tex`:146) becomes "is higher, in no class
  significantly", because 0 of 6 MSE increases are significant.

**Fallback T3 question (only if the adjudicator escalates):** "Should the abstract and introduction keep the
full-set gain against the OLS-HAR as the first result, with what the other 36 series add beyond HAR + RM + FOMC as
the stated question and the third result? Recommended default: yes, the current abstract's order."

---

## C2: Tests in the main text (Table 1, `tab:rh_oos_vsols`)

**Answer.** D2's text already prescribes the tests: "Every main-text forecast (incl. the trees' full sets) is tested
against the RTH13 OLS-HAR; the within-class full-vs-HAR-set pairs stay as the paired comparison"
(`INTEGRATION_PLAN`:37). The main text should carry one every-bar Table 1:

- MSE, QLIKE and R²_OOS (as now).
- "% vs OLS-HAR" and "DM (p) vs OLS-HAR" for all 12 rows. This includes the three tree HAR-set rows: LightGBM's HAR set
  is −0.9% with p = 0.012, which the text's "within 0.9%" needs.
- The within-class DM, full against its own HAR set. For the linear rows it equals the vs-OLS DM to two decimals
  (−6.28, −6.34, −7.36). For the trees it is −10.26 / −9.94 / −9.74 against −10.35 / −9.95 / −9.65 vs OLS.
- A marker for 90% MCS membership. Only LightGBM full and XGBoost full survive (`rhmcsBarQlikeTmaxSetNinety`).

`tab:rh_oos_vsols` is not `\input` anywhere today. Its 16:00 columns belong in the closing-interval appendix's accuracy
table, so D16 holds. The paired D18 tests (each class against its own base) stay in `tab:rh_ablation` Panel D.

Two small required notes:
- The penalized HAR-set rows "coincide" with the OLS (0.0%), yet have DM 8.50 / 4.40 / 4.85 (p < 0.001) against it.
  The loss differences are tiny and nearly constant. A note must say so, or the three rows should collapse into
  "OLS (= ridge, lasso, elastic net on the HAR set)". Otherwise a referee will read the table as contradicting itself.
- Cite Giacomini and White (2006) in the methods. Every paired comparison here is nested (full ⊃ HAR, full ⊃ base),
  and with a fixed-length rolling window DM-type tests of equal predictive ability of forecasting methods are valid
  for nested models. Today the paper is silent on nesting (no hit for nested, Clark or Giacomini in 05_methods,
  10_statistical_tests or A_methods), and Clark–McCracken / Clark–West is the first thing a JFEc referee raises. The
  pending Clark–West appendix item (plan :62) complements this but does not replace it.

**Literature**
- Christensen, Siggaard and Veliyev (2023, JFEc 21(5)) report every model's out-of-sample loss *relative to one HAR
  benchmark* across nested information sets, with an OLS on the extended set (HAR-X) as a column. They report MCS
  inclusion rates (CREATES RP 2021-03, Table 2, Fig. 4). That is D2's design.
- Zhang, Zhang, Cucuringu and Qian (2024, JFEc 22(2)) mark MCS membership in the main tables and put the full DM
  tables in an appendix (Table C.1). They average intraday losses over intervals and days, with no time-of-day
  breakdown in the main out-of-sample tables. That precedent backs D16's every-bar-only main text, and Cucuringu is a
  coauthor of this paper.

**Evidence**
- `generated/rehaul_results_oos.tex` (DM column: own HAR set).
- `generated/rehaul_results_oos_vsols.tex` (every-bar plus 16:00 columns).
- `generated/rehaul_oos_vs_ols_numbers.tex`.
- `rehaul/06_results.tex`:121–135, whose "DM column of Table 1" is the paired test.

**If adopted**
- `make_rehaul_oos_rerun.py` writes the vs-OLS every-bar columns into `tab:rh_oos` and stops emitting the 16:00
  columns into any main-text float. The 16:00 pair goes to the A_closing accuracy table.
- `06_results.tex` "Against the primary benchmark" points to Table 1's vs-OLS columns.
- One methods sentence and a bib entry for GiacominiWhite2006.
- The fresh version's `% CONFLICT` at lines 393–397 is resolved.

---

## C3: The linear models beyond the base: which comparison

**Answer: each class against its own base (estimator held fixed).** Comparing the penalized full set with the *OLS*
base changes two things at once, the estimator and the information set. The exact decomposition already exists. All
figures are % of the OLS base's QLIKE (`ADDENDUM_pen_split.md`):

| Elastic net | total (full vs OLS base) | penalization (enet base vs OLS base) | series (enet full vs enet base) |
|---|---|---|---|
| ordinary days | −2.62 (p .005) | −2.06 (p .018) | −0.57 (p .194) |
| every bar | +2.74 (p .021) | +3.37 (p .003) | −0.61 (p .161) |
| statement days | +71.2 | +72.7 | −0.9 (p .459) |

The −2.62% and +2.74% are the estimator: shrinkage of the 132 base columns helps on ordinary days and wrecks the
FOMC coefficients on statement days. They are not the 36 series. Using them to answer D18 would be a confounded
comparison.

**Claim that survives a referee**
1. With the estimator held fixed, the remaining 36 series do not significantly improve the ridge, lasso or elastic
   net:
   - every bar: +0.1 / −0.6 / −0.6%, p ≥ 0.161 (`rhabLinFullVsAddRmFomcAllPMin`);
   - ordinary days: +0.2 / 0.0 / −0.6%, p ≥ 0.194.
   - The penalized factor test B agrees: "no gain to recover" (ridge, lasso); "gain not significant" (elastic net)
     (`linear_pen/SUMMARY.md`:9–14).
2. Scope it "at every bar and on ordinary days". On statement days the lasso's full set is −6.4% below its own base
   (DM −3.58, p < 0.001; `rhabLassoFullVsAddRmFomcEvP`). An unscoped "no significant accuracy" (fresh version :230,
   :264–265) is literally false for one class on one day type.
3. The frozen-penalty convention cannot manufacture this null. The base inherits the penalty tuned for the full set.
   If that penalty is wrong for the base, it handicaps the base and biases the comparison *toward* a gain, so finding
   none is conservative.
4. Disclose (F1) that OLS on HAR + RM + FOMC is the most accurate linear every-bar forecast. It is −9.2% against the
   OLS-HAR (DM −5.53, `rhabOlsAddRmFomcVsHarAllPct`). Against it, the ridge full set is +3.46% (p .003), the elastic
   net full set +2.74% (p .021) and the lasso full set +0.24% (p .743). Penalization costs the linear models their
   statement-day gain.

**Evidence**
- Author, 28 Oct, commissioning the penalized tests: "they're cheap, and they make the results apples to apples. run
  it." (commit a38439f). The test was built so that each class faces its own base.
- Coordinator, 27 Oct: "the linear factor test … is not wired: its R compares the penalized full set with OLS on the
  base, and that gap is mostly penalization of the base columns, not the six groups" (`rehaul/A_results.tex`:439–442).
- D2's logic: the within-class pairs measure "what the exogenous predictors add" (`INTEGRATION_PLAN`:37).
- Christensen et al. (2023) separate the two comparisons the same way: the information set across rows, the estimator
  across columns (HAR-X against ridge on the same set).

**If adopted**
- The fresh version's beat at :467–469 ("Against the OLS on the base, the elastic-net full set …") leaves the main
  text. The decomposition table goes to the appendix as the explanation.
- Main text states claim 1 with the scope of claim 2, plus claim 4 as one sentence.
- `tab:rh_ablation` gains an OLS row for HAR + RM + FOMC, in Panel C or Panel D's OLS column (now "--").
- The fresh version's `% CONFLICT` at :463–466 is resolved.

---

## C4: The ladder figure's 16:00 panel

**Answer.** The figure stays in the appendix.
- D16's commit 7f7894a moved `fig:rh_ladder` to `A_fi`.
- D20 (approved 28 Oct) adopted the v2 map, which lists "Fig. `fig:rh_ladder` | to appendix | −0.5" and "Figure
  `fig:rh_ladder` moves to the appendix" (`MAP_06_07.md`:84, :101).
- `A_fi.tex`:366–369 now hosts it as F.6 "Feature-count ladder".
- The budget row "feature-count ladder (1 figure)" (`INTEGRATION_PLAN`:21) predates both.

Under D18 the main-text "how many series" evidence is in any case the *beyond-base* ladder. The over-HAR ladder's
top-10 rungs (95–98%) mostly repeat the base-share result: the top-ranked series include the FOMC and RM series. Main
text cites rungs by macro. If the author later wants a main-text figure, an every-bar-only panel is a one-option change
to `make_rehaul_ladder.py` (today it draws both panels unconditionally, line 359).

**If adopted:** nothing moves (status quo). The fresh version's `% CONFLICT` at :486–488 is resolved.

---

## C5: SHAP grouping against D18's base

**Answer: mechanical regrouping.** Panel A of `tab:rh_fip_shares` is by bucket, and its six non-base buckets are
exactly D18's six groups:

| Bucket | Series |
|---|---|
| Liquidity | 14, including the futures volume and trade count |
| EW market | 6 |
| VW market | 6 |
| Sentiment | 3 |
| Implied volatility | 3, including the VIX |
| Volatility demand | 4 |
| **Total** | **36** (`buckets_numbers.tex`; = `rhbyFLNSeries`) |

The pre-registered split instead classes VIX, futures volume and trade count as "standard" and FOMC as "non-standard"
(`RULE_shap.txt`:77–85). It answers the professor's registered question, so it stays in the appendix as registered.

Point estimates from `m1_buckets.csv` (fomcfix run): the six groups hold **LightGBM 22.8%, XGBoost 21.3%, forest
20.0%** of the trees' total attribution. Their accuracy beyond the base is −1.6 / −1.8 / −0.5%. That is the clean
D18 version of "use is not accuracy". The fresh version's "use" beat (:499–501) is a non-sequitur without it.

**If adopted**
- `make_rehaul_fi_pooled.py` and `shap_reduce.py` take a D18 class map with the same circular block bootstrap over
  origins (block 12, B = 2,000). The reduction runs over saved aggregates; nothing is refit.
- New macros give the six-group share range and its interval. Panel B gains (or switches to) "Base: HAR + calendar +
  RM + FOMC / Six remaining groups". Panel C's tree-minus-linear D is recomputed on that split and labelled a
  relabelling of the registered measure.
- `\rhPending` at fresh :498 is filled. The fresh version's `% CONFLICT` at :493–495 is resolved.

---

## C6: The application trades on the 16:00 forecast while D16 bans 16:00 accuracy

**Answer: not incoherent, with one pointer sentence.**
- The author made the choice knowing the tension. Commit 7f7894a: "The economic application (15:30 straddle on the
  16:00 forecast) is unchanged pending the author's decision". The author's answer: "keep the econ application as
  is" (D16, `INTEGRATION_PLAN`:51).
- A referee accepts an economic evaluation at the one bar with a traded payoff ending at the bar's end: SPXW 0DTE
  settles at the 16:00 close. Every-bar statistical evaluation is the headline.
- Trading profit is a different loss function from QLIKE. Statistical and economic rankings routinely diverge (Leitch
  and Tanner 1991). The paper's own precedents, Engle, Hong, Kane and Noh (1993) and Noh, Engle and Kane (1994), value
  forecasts through straddle trades.
- The referee's question will be "how accurate is the forecast you trade on?". The answer must be one click away.

**The divergence is real.**
- At 16:00 the linear full sets are *not* more accurate than the OLS-HAR: +1.8 / +4.2 / +0.6%, n.s. LightGBM and
  XGBoost are (−5.3 / −5.8%, p < 0.001).
- Yet all six full-set strategies have higher Sharpe ratios than the OLS-HAR strategy (+0.58 to +0.92). Only the
  forest's gain is significant (p 0.039), and its 16:00 QLIKE gain is −1.8% (n.s.).
- The sign trade depends on where σ̂² falls relative to IV², not on QLIKE.

**Fix that stays inside D16.** Add one sentence in Section 7, "Trading strategy" (no 16:00 numbers): "The trade uses
each model's forecast of the 16:00 bar; that bar's forecast accuracy against the OLS-HAR, and its relation to the
strategies' Sharpe ratios, are in Appendix~\ref{app:rh_closing} and Appendix~\ref{…forecast loss vs Sharpe…}." The
explanation of the divergence (sign accuracy, the confusion table `appendix_running_confusion`) belongs in that
appendix. Today `07_economics.tex` has no pointer to the 16:00 accuracy (only "forecast loss against the Sharpe ratio"
among further results, :305). A main-text qualitative clause ("the Sharpe ranking does not follow the QLIKE ranking at
that bar") would be a 16:00-accuracy result. Default: leave it out (D16 literal).

**If adopted:** one sentence and one appendix paragraph. The fresh version's `% CONFLICT` at :518–521 is resolved; the
fresh version's beat 1 at :523–525 already does this.

---

## C7: The closing-interval models: final main-text section or appendix

**Answer: appendix.**
- D3 (plan of 26 Oct, commit 2f0353a) is superseded the same evening by D16 (7f7894a, 18:00, "why is the 1600 bar
  still in the main paper??"). The closing-interval models (CI-HAR, close-trained linear and trees) *are* 16:00-bar
  accuracy results.
- The plan's own budget table has no main-text closing section. Its online-appendix list names "CI-HAR and
  closing-interval section" (`INTEGRATION_PLAN`:26–28).
- The author notes, which the author reads in the draft build, already restate D3 as "the closing-interval models live
  only in the final appendix section (D3)" (`rehaul/99_author_notes.tex`:17–18).
- `MAP_06_07.md` retires `sec:rh_results_close` ("its content is in App. app:rh_closing").

D3's binding content is the corralling. Every closing-interval model sits in one place, rebuilt on the pre-registered
three-way CLOSE arm plus the rebuilt CI-HAR, with the old master-table set dropped. That now governs `A_closing.tex`.
The JFEc main text (about 22–24 pp in this template) has no room for it.

**If adopted:** status quo (`A_closing.tex` is the last appendix section). The fresh version's `% CONFLICT` at
:601–602 is resolved. Optionally reword D3's row to "final appendix section" so the register stops conflicting with
itself.

---

## Q8: Restart from the fresh version, or keep the current bullets as the first L2 draft?

**Answer: keep the current bullets as the first L2 draft (no reset); serve a corrected fresh top layer as L1.**

The current text is the more *correct* of the two. It carries author locks made after or outside D1–D20 that the fresh
version, written without reading it, misses or contradicts:

| # | Fresh version | Lock or evidence it breaks |
|---|---|---|
| 1 | "a small, **dense** gain to boosted trees" (:218); "spread over the series" (:266–268, :574–575) | Author edit 3c6ccc4: "sparse, not dense"; `08_feature_importance.tex`:270. Within-block ladder: the top 5 series recover 0.72 (LightGBM) and 0.91 (XGBoost) of the ordinary-day remainder; readings "intermediate" and "concentrated". "Dense beyond a few factors" only says that a few *common components* do not recover it. |
| 2 | A separate section, "Beyond HAR, realized moments and the FOMC calendar" (:454) | Author, 28 Oct: "sections 6 and 7 were never meant to be split. it needs to be one experiments section" (`paper_rehaul.tex`:325). D20 dropped v2's separate beyond subsection. |
| 3 | `tab:rh_fip_shares` and the beyond table in the main text (:510–512) | D20 map: SHAP table to the appendix (`MAP_06_07.md`:83, :100). |
| 4 | OLS-base comparison of the elastic net as a main-text beat (:467–469) | C3: confounded. The coordinator's 27 Oct decision not to wire it. |
| 5 | "about as much as the penalized linear full sets" (:440–442) | OLS on HAR + RM + FOMC is −9.2%, ≥ every penalized full set and significantly better than ridge and elastic net (F1). |
| 6 | "no significant accuracy to penalized linear models", unscoped (:230, :264) | Lasso on statement days, −6.4%, p < 0.001. |

The fresh version is the better *top layer*: a governing claim, beats that are claims, and D18-ordered contributions.
The current L1 (`beats/top/01`, `06`, `09`) is mostly bold labels with no claim ("Design.", "Benchmark and
evaluation.", "Contributions."). L1 in §8.1 is "top-level claims only". The current text also has its own referee
defects to fix in L2: F1, "reverses" under MSE, and "beyond richer HAR models".

Under A1, author judgments lock and propagate. A reset would discard the 27–28 Oct locks and force them to be
re-applied. A2's "simple, unedited" is about not polishing before the author reads, and the current bullets are plain.
The author framed the fresh version as a comparator: "let's do a version where we start over just to compare" (89c359e).

**If adopted**
- L1 = the fresh top beats with errors 1–6 corrected and C1–C7 applied as above. The author reads it once.
- L2 = the current bullets, reconciled to the locked L1. Bullets contradicting L1 are parked, not rewritten freely.
- §8.9 Q8 is recorded as T2 (inferred), vetoable in one line.

---

## Cross-cutting referee findings (not among C1–C7, but they change claims)

- **F1. "Beats augmented HAR models" overclaims for the linear classes.**
  - OLS on HAR + RM + FOMC is −9.2% against the OLS-HAR (DM −5.53) at every bar. That is the same or larger than every
    penalized full set: −6.0 / −9.0 / −6.7%.
  - Against it, ridge full is +3.46% (p .003), elastic net full +2.74% (p .021) and lasso full +0.24% (p .743)
    (`ADDENDUM_pen_split.md`).
  - The number is printed nowhere in the paper. `rhabOlsAddRmFomcVsHar*` is defined but unused, and Panel D's OLS
    column is "--".
  - Under D18 this is the paper's own "well-studied" HAR-X. A referee who knows Christensen et al. (2023) asks for
    exactly this OLS-on-the-extended-set row.
  - Fix: add the row (Panel C, "HAR + RM + FOMC (OLS)"), and scope the abstract's clause to "HAR models augmented with
    the realized moments or with semivariance, quarticity and leverage terms", or drop it.
  - Only the trees beat every augmented HAR. Their base alone is about 4.6–4.8% below the OLS base, computed from
    `rhab*AddRmFomcVsHarAllPct` and the HAR-set QLIKEs; this is nonlinear use of the FOMC and RM information.
  - Tier T1.
- **F2. Nested comparisons:** cite Giacomini–White (2006) (see C2). Tier T1.
- **F3. "Coincide" against DM 8.50:** see C2. Tier T0.
- **F4. MSE "reverses":** none of the 6 MSE increases is significant. Say "higher, in no class significantly" (D7's
  clause stands). Tier T0.

## Sources (literature checked for this note)

- Christensen, Siggaard and Veliyev (2023), "A Machine Learning Approach to Volatility Forecasting", *JFEc* 21(5),
  1680–1727, doi:10.1093/jjfinec/nbac020. Working paper: [CREATES RP 2021-03](https://repec.econ.au.dk/repec/creates/rp/21/rp21_03.pdf)
  (Table 2: MSE relative to HAR for HAR-X, ridge, lasso, EN, trees, by nested dataset; Fig. 4: MCS inclusion).
  [RePEc record](https://econpapers.repec.org/RePEc:oup:jfinec:v:21:y:2023:i:5:p:1680-1727.)
- Zhang, Zhang, Cucuringu and Qian (2024), "Volatility Forecasting with Machine Learning and Intraday Commonality",
  *JFEc* 22(2), 492–530, doi:10.1093/jjfinec/nbad005. [arXiv 2202.08962 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2202.08962)
  (MCS asterisks in Table 3; DM tests in Appendix Table C.1; losses averaged over intervals and days).
  [RePEc record](https://econpapers.repec.org/RePEc:oup:jfinec:v:22:y:2024:i:2:p:492-530.)
- From the literature, not re-fetched: Giacomini and White (2006), *Econometrica* 74(6), 1545–1578; Clark and
  McCracken (2001), *J. Econometrics* 105; Clark and West (2007), *J. Econometrics* 138; Leitch and Tanner (1991),
  *AER* 81(3); Hansen, Lunde and Nason (2011), *Econometrica* 79(2). Engle, Hong, Kane and Noh (1993) and Noh, Engle
  and Kane (1994) are already cited by the paper.
