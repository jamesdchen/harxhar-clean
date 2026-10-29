# Deliberation 2026-10-29: solver EVIDENCE (lens: results, generated numbers, tests, JFEc norms)

Independent solver; no other solver's output was read. No paper file edited, nothing committed, no compute launched.
Sources read: `writeup/INTEGRATION_PLAN_2026-10-26.md` (D1-D20), `INTEGRATION_SPEC_2026-10-26.md`,
`SESSION_HANDOFF_2026-10-26.md`, `writeup/fresh/paper_fresh_beats.tex` (7 `% CONFLICT`), `writeup/beats/{top,full}/*`,
`writeup/rehaul/*.tex`, `writeup/generated/*` (Table 1, vs-OLS table, ablation, ladder, SHAP), the two penalized factor
SUMMARYs, `results/beyond_trees_2026-10-27/linear/ADDENDUM_pen_split.md`, `RULE_linear_pen.txt`, the SHAP rule and
outputs, `writeup/reports/AI layered paper writing system.md` (A1-A6, Q8), git log 26-28 Oct.

## Verdicts

| Item | Answer (short) | Conf. | Tier |
|---|---|---|---|
| C1 | No real conflict. State D18's question as the research question. Report the full-set gain against the OLS-HAR first (D2) with the MSE clause (D7). Then give the base share (89-101%), then what lies beyond the base. | 0.65 | T2 |
| C2 | Main text: both tests, every bar only. Table 1 gets "% vs OLS-HAR" and "DM vs OLS-HAR" for every row, keeping the within-class DM. The 16:00 columns of tab:rh_oos_vsols go to App. rh_closing. | 0.85 | T1 |
| C3 | Use within-class comparisons (each class against its own base). Claim: the 36 series add nothing significant to any penalized linear model at every bar or on ordinary days. The gaps against OLS on the base come from the penalty, not the series. | 0.85 | T1 |
| C4 | Already resolved. The figure is in App. F.6 (A_fi.tex:369), where a 16:00 panel is allowed. The main text uses every-bar ladder macros only. Nothing to do. | 0.9 | T0 |
| C5 | Keep the pre-registered standard/non-standard split in the appendix (its rule forbids revising it). Add a D18 grouping, labelled post hoc, and use it in the main text: the six groups hold 20-23% of the trees' attribution but 4-11% of their gain. | 0.75 | T1 |
| C6 | No conflict. D16 itself keeps the application "as is" and bans only 16:00 accuracy. Add a pointer to App. rh_closing with no numbers and no direction words. | 0.8 | T1 |
| C7 | Appendix. A later and more specific lock (D16), the author-approved spec and the 17-Oct parking all put the closing-interval models in the final appendix section, which is already built (A_closing.tex). | 0.9 | T0 |
| Q8 | Do not restart from the fresh version. Keep the current bullets as the first L2 draft. Fill the 14 header-only top beats with claim sentences adapted from the fresh version. Use the fresh version as a checklist. | 0.7 | T2 |

---

## C1. Introduction framing

**Answer.** These are not alternatives. Read by scope, the locks compose:
- D18 sets the question.
- D1 sets the target (every regular-hours bar).
- D2 sets the benchmark (OLS-HAR).
- D7 requires the MSE clause in the abstract, introduction and conclusion.

Recommended intro and abstract order:
1. Question: what the other exogenous series add to every-bar realized-variance forecasts beyond the HAR terms, the
   futures' own realized moments (RM) and the FOMC calendar (D18, on D1's target).
2. Finding 1: the full set lowers QLIKE against the OLS-HAR by 6.0-9.0% (linear) and 11.2-14.9% (trees), significant in
   6 of 6 classes (D2). It raises MSE through a few crash days (D7).
3. Finding 2: HAR + RM + FOMC alone delivers 89-101% of each class's full-set gain (D17/D18 channels).
4. Finding 3, the answer to the question: the other 36 series add nothing significant to the penalized linear models
   and add 1.6-1.8% to the boosted trees, through a few series from several groups (sparse, not dense).
5. Application (2 sentences).

This is the fresh version's order; its `% CONFLICT` (fresh l.240-242) dissolves once the locks are read by scope. The
"reverse" (opening with the beyond-base result) is not supported. It would bury the only large and robust result, and
D2/D7 explicitly place the full-set gain and the MSE clause in the abstract and introduction.

**Decisive evidence.**
- D18 (INTEGRATION_PLAN l.54, 26 Oct): the author "quotients out the HAR, the futures' realized moments and the FOMC
  calendar ('those are well studied'): the question is what the other exogenous series add beyond HAR + RM + FOMC".
  D18 is later than D1/D2/D7 (l.36, 37, 42) and governs "the question". D1/D2/D7 govern target, benchmark and the MSE
  clause, and none of them is contradicted.
- Numbers (rehaul_ablation_numbers):
  - Base share `\rhabBaseShareMin/Max` = 89/101. By class: ridge 101, lasso 94, enet 91, LightGBM 90, XGBoost 89,
    forest 96.
  - OLS on HAR + RM + FOMC: -9.2% vs the OLS-HAR (`\rhabOlsAddRmFomcVsHarAllPct`, DM -5.53). That is as large as the
    penalized linear full sets (-6.0 / -9.0 / -6.7%; rehaul_oos_vs_ols_numbers).
  - So the headline gain sits almost entirely in channels the author calls "well studied".
  - Under the current generic question ("whether exogenous predictors improve forecasts", 01_introduction.tex
    Research-question bullet), the answer a JFEc referee reads is a known result.
- The author has moved toward D18 since 26 Oct:
  - 27 Oct, "consolidate everything into a clear concise statement" (06_results.tex:217): "two groups carry the gain;
    what remains is small, only the trees capture it...".
  - 28 Oct, commit 3c6ccc4: "the claim is about the 36 series beyond HAR + RM + FOMC (no linear gain, small boosted-tree
    gain from a few series across groups: sparse, not dense)". This edited the intro and conclusion but left the lead
    in place, which is weak evidence that the full-set-first order is acceptable.
- Reservation: SESSION_HANDOFF_2026-10-26.md:41 lists "Headline under D18 (the three-part reference) once the
  add_rm_fomc numbers are in" under "Open author decisions". The numbers are now in (add_rm_fomc arm and both factor
  tests).

**If adopted.**
- rehaul/01_introduction.tex: the Research-question bullet states D18's question. The D18 sub-bullet of contribution 2
  ("Beyond the FOMC calendar...", l.139ff) becomes its own contribution.
- paper_rehaul.tex abstract: the first sentence names the three known channels. Same 100-word budget: the current
  third sentence already carries the answer.
- rehaul/09_conclusion.tex: the first bullet.
- Macros all exist: `\rhorVsOlsRedLinRng`, `\rhorVsOlsRedTreeRng`, `\rhabBaseShareMin/Max`,
  `\rhabLgbm/XgbFullVsAddRmFomcAllPct`, `\rhabLinFullVsAddRmFomcAllPMin`, `\rhabOlsAddRmFomcVsHarAllPct`,
  `\rhbyFLNSeries`. No generator or title change.

**Tier.** T2: the author's dated quotes imply the answer, and no lock conflicts. If the adjudicator treats the 26-Oct
handoff reservation as binding, the T3 question would be: "Should the introduction ask D18's question (what the other
36 series add beyond HAR, the futures' realized moments and the FOMC calendar), with the full-set gain against the
OLS-HAR reported first as the gain to be decomposed? Default: yes."

---

## C2. Table 1's tests in the main text

**Answer.** Every main-text accuracy table and claim tests at every bar only, with two tests:
- (a) each forecast against the OLS-HAR (D2), for every row including the trees' HAR-set rows;
- (b) each class's full set against its own HAR set, the within-class pair that D2 explicitly keeps.

Table 1 (tab:rh_oos) should carry, for every row, "% vs OLS-HAR" and "DM vs OLS-HAR (p)", and keep the within-class DM
on the full-set rows. The 16:00 columns of tab:rh_oos_vsols leave the main text and go to App. rh_closing (D16). No
conclusion changes: both tests reject at p < 0.001 for all six full sets.

**Decisive evidence.**
- D2 consequence column (PLAN l.37): "Every main-text forecast (incl. the trees' full sets) is tested against the
  RTH13 OLS-HAR; the within-class full-vs-HAR-set pairs stay as the paired comparison". This literally specifies both
  tests.
- D16 (l.51, 26 Oct, "why is the 1600 bar still in the main paper??") rules out the 16:00 columns.
- Numbers (generated/rehaul_results_oos.tex; rehaul_results_oos_vsols.tex):

  | Forecast | Within-class DM | DM vs OLS-HAR |
  |---|---|---|
  | Ridge full | -6.28 | -6.28 |
  | Lasso full | -6.34 | -6.34 |
  | Elastic net full | -7.36 | -7.36 |
  | LightGBM full | -10.26 | -10.35 |
  | XGBoost full | -9.94 | -9.95 |
  | Random forest full | -9.74 | -9.65 |

  - The two DMs are identical for the linear classes because their HAR-set fits equal the OLS to four decimals of
    QLIKE.
  - The trees' HAR sets against the OLS: LightGBM -0.9% (DM -2.52, p 0.012), XGBoost -0.3% (p 0.363), forest +0.1%
    (p 0.752). These rows are main-text forecasts and currently have no test in any table.
  - 16:00 against the OLS (to be moved out): ridge +1.8%, lasso +4.2%, enet +0.6% (all n.s.), LightGBM -5.3%,
    XGBoost -5.8%, forest -1.8%.
- Current state:
  - Section 6.1 already quotes the trees' DMs against the OLS in the text (`\rhorVsOlsDmLgbmFull`, ...;
    beats/full/06_results.tex "Against the primary benchmark").
  - Table 1 shows only the within-class DM, and its note says "minus that of the HAR benchmark of the same model".
  - tab:rh_oos_vsols is `\input` nowhere in the paper (grep). So the abstract's "significantly in all 6 classes"
    against the OLS has no backing table.
- JFEc norm: the benchmark test sits in the main accuracy table; nested ablations are reported beside it.

**If adopted.**
- writeup/make_rehaul_oos_rerun.py (writes both tables and both number files):
  - tab:rh_oos becomes a long format: Model x Set rows; MSE, QLIKE, R2_OOS, % vs OLS-HAR, DM vs OLS-HAR (p), DM vs own
    HAR set (p).
  - It could instead stay wide with two DM columns plus a tree-HAR row test.
  - tab:rh_oos_vsols keeps only its 16:00 columns and is `\input` in rehaul/A_closing.tex.
  - Update the notes in both tables.
- Text: rehaul/06_results.tex 6.1 cites Table 1 for both tests. Update the backing comment for the abstract's claim
  in paper_rehaul.tex.
- Macros exist (`\rhorVsOls{Pct,Dm,P}<Model><Set>`). Page-neutral. A presentation option: footnote that the linear
  HAR-set rows coincide with the OLS (max |change| 0.00%, though their DM against the OLS is 4.40-8.50 from
  fifth-decimal differences), rather than printing a "significant" 0.0%.

---

## C3. Linear models beyond the base: which comparison

**Answer.** The question "what do the 36 series add?" is about information. The estimator must be held fixed, so the
correct test is each penalized class's full set against the same class on HAR + RM + FOMC (its own base). Setting a
penalized full set against OLS on the base mixes two changes, the estimator (penalty) and the information set. The
numbers show the penalty explains nearly all of the OLS-base gaps.

Claim for the paper:
- At every bar and on ordinary days, the 36 series add no significant accuracy to any penalized linear model (each
  class against its own base):
  - every bar: ridge +0.1%, lasso -0.6%, enet -0.6%, p >= 0.161;
  - ordinary days: +0.2%, 0.0%, -0.6%, p >= 0.194.
- The penalized factor test confirms this ("no gain to recover" for ridge and lasso; the enet's gain is not
  significant).
- One sentence of disclosure: OLS on the base is as accurate at every bar as the penalized full sets (-9.2% against
  the OLS-HAR). The penalized models' distance from it is the penalty's: about -1 to -2% on ordinary days and +25 to
  +73% on statement days, where the frozen penalties shrink the FOMC terms.
- Scope caveat: the lasso does gain beyond its base on statement days (-6.4%, p < 0.001). The forest gains on ordinary
  days (-0.6%, p 0.037). So write "at every bar" and keep "only boosted trees" scoped to every bar.

**Decisive evidence.**
- The author, 28 Oct, quoted verbatim in RULE_linear_pen.txt:5: "they're cheap, and they make the results apples to
  apples. run it."
- RULE_linear_pen.txt:15-18, "WHY (the mismatch)": the earlier linear tests "fit the base and the component arms by
  OLS but measure R against a PENALIZED full set ... so R mixed the components' contribution with the penalty's".
- The author's commit 3c6ccc4 (28 Oct): "no linear gain".
- ADDENDUM_pen_split.md (existing forecasts), ordinary days:

  | Class | Total vs OLS base | Penalization | Series |
  |---|---|---|---|
  | Ridge | -1.88% (p .031) | -2.03% (p .018) | +0.15% (p .752) |
  | Lasso | -1.06% | -1.07% | 0.00% (p .995) |
  | Elastic net | -2.62% (p .005) | -2.06% (p .018) | -0.57% (p .194) |

  At every bar, the enet's +2.74% (p .021) against the OLS base is +3.37% penalization and -0.61% series. On statement
  days, penalization accounts for +72.7% (ridge, enet) and +24.8% (lasso).
- linear_pen/SUMMARY.md l.11-14, readings on ordinary days:
  - ridge +0.15% (p .752) NO GAIN TO RECOVER;
  - lasso 0.00% (p .995) NO GAIN TO RECOVER;
  - enet -0.57% (p .194) DENSE (gain not significant, R imprecise).
- The same macros already exist (rehaul_ablation_numbers):
  - `\rhab<Class>FullVsAddRmFomc{All,Ord,Ev}{Pct,P}`, the own-base test;
  - `\rhab<Class>FullVsOlsAddRmFomc...`, the OLS-base test;
  - `\rhabOlsAddRmFomcAllQlike` = 0.1719, against the full sets' 0.1778 / 0.1723 / 0.1766.
- The trees' test uses one model for base, arms and full set (RULE.txt section 1). Within-class is the like-for-like
  counterpart.

**If adopted.**
- The current main text already makes the within-class claim (08_feature_importance.tex 6.3.3 "Small, and only in the
  boosted trees"; 01_introduction.tex l.139; 09_conclusion.tex l.55). Keep it. Add "at every bar" scoping, the lasso
  statement-day caveat and one OLS-on-base sentence.
- Drop the fresh beat that sets the enet against the OLS base (`\rhbyFLFull{Ord,All}{Pct,P}`, fresh l.463-469) from
  the main text.
- New generator (or an extension of make_rehaul_beyond*.py) emitting macros from:
  - results/beyond_trees_2026-10-27/linear_pen (reading.json, scores.csv);
  - results/channels_2026-10-26/factor_pen;
  - linear/pen_split.csv.

  No generated macro reads these yet.
- Appendix:
  - tab:rh_beyond_factor_lin (A_results; OLS base against a penalized full set, i.e. the mismatched R) is replaced or
    joined by the penalized test B.
  - `\rhbyFLReading` ("dense beyond a few factors") is superseded for the penalized classes.
  - The same principle applies to test A (tab:rh_factor, Section 6.3.2). Penalized readings are ridge DENSE, lasso
    INTERMEDIATE, enet DENSE, with R(pca10) 0.40 / 0.45 / 0.31 against OLS 0.27. The section's "k <= 3 recovers none"
    needs softening. The best k <= 3 arms are ridge pls3 0.23 (not significant against the base, p 0.39), enet 0.07
    and lasso 0.01, so write "little" rather than "none" for the penalized classes.

---

## C4. Ladder figure with a 16:00 panel

**Answer.** Already resolved; no action. fig:rh_ladder is an appendix figure (rehaul/A_fi.tex:369, App. F.6
app:rh_fi_ladder). D16 bans 16:00 accuracy only from the main text, so its right-hand 16:00 panel may stay. The main
text uses only every-bar ladder numbers:
- `\ladderTreeRetainRng` = 95-98 = `\rhldTreeRTenMin/Max`, "Tree models, every bar";
- `\ladderLinKstarRng` = 10, "Lin, every bar";
- Table tab:rh_channels_sparse, which has no 16:00 column.

**Decisive evidence.**
- The JFEc budget row (PLAN l.21, 26 Oct) put "feature-count ladder (1 figure)" in the VI section. Two later locks
  supersede it:
  - D16 / commit 7f7894a (26 Oct): "No 16:00-bar accuracy results in the main text (author): moved to Appendix
    app:rh_closing; ... ladder figure moved". The handoff, SESSION_HANDOFF:40, says "the ladder figure already moved
    for D16".
  - D20 / commit b0477ae (28 Oct, v2 approved): "F.6 app:rh_fi_ladder (the every-bar ladder, moved)".
- The fresh `% CONFLICT` (l.486-488) came from reading the budget row without the later locks.

**If adopted.** Nothing in the paper. Annotate the budget row in INTEGRATION_PLAN as superseded by D16/D20. If the
author later wants the figure in the main text: once addendum K's every-bar refit ladder lands, make_rehaul_ladder.py
can emit a one-panel (every-bar) variant.

---

## C5. SHAP grouping against D18's base

**Answer.** Regroup by adding a grouping, not by replacing one.
- Appendix: keep the pre-registered standard/non-standard Panels B-C of tab:rh_fip_shares, labelled pre-registered.
- Add a D18 grouping (HAR; calendar; base exogenous = RM + FOMC; the six remaining groups), labelled "added after
  D18, post hoc", as the rule requires.
- Main text: use the D18 grouping. It gives the sharpest use-versus-accuracy contrast in the paper. The six remaining
  groups hold 20-23% of the trees' total attribution, about half of their exogenous attribution, yet deliver 4-11% of
  their accuracy gain.

**Decisive evidence.**
- RULE_shap.txt:150 (23 Oct, before D18): "The measures, thresholds and classes above are not revised after seeing
  results; anything added later is labelled post hoc". A_fi.tex:75 says the same ("Classes, fixed before any SHAP value
  was computed"). Dropping or redefining the classes would break a registered rule.
- The two splits are both 10/36 but differ by three series:
  - "Standard" holds the VIX, futures volume and trade count (fomcfix code/shap_common.py l.128-145;
    RULE_shap.txt:76-79).
  - D18's base holds the 3 FOMC series (`\rhbyFLNSeries` = 36: ew, vw, liq, iv, sent, vd).
  - So no current number gives the D18 share, hence the fresh version's `\rhPending` (l.496-498).
- A1 (28 Oct, report l.31): a top-layer decision "needs to be reflected everywhere those results are reported or used
  to back up any claim". D18 is such a decision.
- Computed now from results/tree_shap_w3000_2014start_fomcfix_2026-10-26/m1_buckets.csv. Shares add, so the point sums
  are exact:

  | Model | Six groups, % of total | Six groups, % of exogenous | RM + FOMC, % of total | Gain beyond base |
  |---|---|---|---|---|
  | LightGBM | 22.8 | 48.0 | 24.6 | 10% |
  | XGBoost | 21.3 | 45.2 | 25.8 | 11% |
  | Random forest | 20.0 | 47.4 | 22.1 | 4% |

  The gain beyond the base is 100 minus `\rhab<Class>BaseShareAll` (90 / 89 / 96).

**If adopted.**
- writeup/make_rehaul_fi_pooled.py:
  - a D18 panel in tab:rh_fip_shares (A_fi.tex:153);
  - macros such as `\rhfipBeyondMin/Max` (20.0-22.8) and `\rhfipBeyondExogMin/Max` (45.2-48.0).
- Bootstrap intervals for the sum need the per-origin arrays (shap_reduce.py's work dir; not found under
  %TEMP%\shap_work on the laptop, so they may need retrieval from CARC). That is a re-reduction with a third class
  map, not a new SHAP computation. Without it, print point shares in the text.
- Main text: one clause in 6.3.5 "Attribution ranks the parts the wrong way round" (08_feature_importance.tex). This
  resolves the fresh `\rhPending`.

---

## C6. The application trades on the 16:00 forecast

**Answer.** There is no conflict to resolve. D16 itself decides both halves: "The economic application stays AS IS
(author: 'keep the econ application as is'): the 15:30 straddle signed by the pooled forecast of the 16:00 bar" and "No
16:00-bar accuracy results in the main text; they live in Appendix app:rh_closing".

Add a one-clause pointer in Section 7, e.g. "the accuracy of the closing-bar forecast is reported in
Appendix~\ref{app:rh_closing}". It has no numbers and no direction words, so it is not a result, and it does not
breach D16. The introduction's Organization bullet already points to app:rh_closing, so the pointer has precedent.

**Decisive evidence.**
- PLAN l.51 (D16, 26 Oct; commit 7f7894a).
- Why a reader needs the pointer: at 16:00 the linear full sets are not better than the OLS-HAR (+1.8 / +4.2 / +0.6%,
  n.s.), while LightGBM and XGBoost are (-5.3 / -5.8%, p < 0.001). The Sharpe gain against the OLS-HAR strategy is
  significant only for the forest (`\rhemFullVsOlsSigWord` = one, `\rhemPDShRfFullVsOls` = 0.039), whose 16:00 accuracy
  gain is not significant (-1.8%, p 0.200).
- A JFEc referee will ask how accurate the signal is at the traded bar.
- The current Section 7 has no such pointer (beats/full/07_economics.tex, opening and Signal bullets,
  07_economics.tex:102-109).

**If adopted.** One clause in rehaul/07_economics.tex (opening bullet or Signal bullet). No macro, generator or table
change.

---

## C7. Closing-interval models: final main-text section or appendix

**Answer.** Appendix, as already built: rehaul/A_closing.tex, "the final section of the online appendix: the models
estimated on the closing half hour alone (decision D3)". D3's words "the final section ('A final refinement...',
07:495ff)" point to a subsection the author had already moved to the appendix on 17 Oct. D16, decided later the same
day and more specific, bans 16:00-bar accuracy from the main text. The closing-interval models are scored only at the
16:00 bar, so a main-text section on them would be exactly that.

**Decisive evidence.**
- rehaul/07_economics.tex:761, 15 Oct: "I need the last 30 min scheme to be moved to the end of the economic
  application section as a final cherry on top".
- 07_economics.tex:769, 17 Oct: "parked -- ... moved verbatim to Appendix sec:rh_econ_last30".
- D16 (PLAN l.51; 7f7894a, 26 Oct): "why is the 1600 bar still in the main paper??".
- INTEGRATION_SPEC_2026-10-26.md:23-24, the "author-approved framing": closing-interval models "live ONLY in the final
  appendix section".
- 99_author_notes.tex:18: "the closing-interval models live only in the final appendix section (D3)".
- The current intro's Organization bullet already sends them to App. rh_closing.
- JFEc budget: the main text is about 22 pp and already at budget.

**If adopted.** Nothing in the paper. Amend D3's row in INTEGRATION_PLAN to read "final appendix section
(rehaul/A_closing.tex, app:rh_closing)" and record the precedence D16 > D3's "07:495ff".

---

## Q8. Restart from the fresh version, or keep the current bullets as the first L2 draft?

**Answer.** Keep the current bullets as the first L2 (bullet-layer) draft. Do not restart from the fresh version. Its
value is as:
- (a) a source of claim sentences for the top layer where the current top beats are bare headers (14 of 106: 5 of 7
  in the introduction, 4 of 8 in the conclusion);
- (b) a checklist of omissions, e.g. the OLS-on-base sentence and the trees-beat-linear MCS beat.

Build L1 from the current top beats with those headers filled. Reconcile every adopted fresh sentence with the later
locks and this deliberation (C1-C7) before serving it.

**Decisive evidence.**
- The current sources carry 921 dated "(user|author, ...)" correction comments from 15-28 Oct, 126 of them from
  26-28 Oct. Examples:
  - 50 "prune 'pooled'";
  - 32 "conventional titles";
  - 14 "Sections 6 and 7 are one section";
  - 6 alt-data edits;
  - v2 approval (D20).
- Under A1 (28 Oct: a judgment "needs to ... stay locked in for every lower layer") these are de facto locks. A
  restart discards them unless each is re-derived.
- The fresh version was written "without reading the current text" (commit 89c359e). It contradicts at least six
  later locks or results:
  1. It has separate "Forecast accuracy" and "Beyond HAR, RM and FOMC" sections (fresh l.388, l.454). Against: author,
     28 Oct, "sections 6 and 7 were never meant to be split. it needs to be one experiments section"
     (06_results.tex:33), and D20, under which v2's beyond-base 6.3 was dropped as a duplicate.
  2. "The gain is spread over the series" (fresh l.267, l.575). Against: the author's "sparse, not dense" (3c6ccc4,
     28 Oct) and the evidence (top-5 series recover 0.72 / 0.91 of the trees' remainder;
     `\rhbyLXgbReading` = concentrated).
  3. "pooled" twice in the abstract and passim (fresh l.214-215, 227, 244, 249). Against: the 28-Oct prune
     (ae0f526), which keeps it once in the abstract.
  4. Its section titles do not follow the 28-Oct conventional titles (3f57938).
  5. It lacks the author-added bullets "StockTwits sentiment in XGBoost" and "A weak test of volatility demand"
     (3c6ccc4).
  6. It uses the OLS-based linear factor reading (`\rhbyFLReading`, fresh l.482), superseded by today's penalized test
     (C3).

  Minor: it drops the Related-work section that the budget keeps at 1.5 pp, and it has 2 `\rhPending`.
- The author's request was "let's do a version where we start over just to compare" (89c359e), a comparison rather
  than a commitment.
- A2 ("start from a simple unedited state") governs the order of reading and correcting. It does not require
  discarding corrections already made. The report's own starting point (§8.5, "Serve the current top beats, seeded
  from D1-D20, as the simple starting state") agrees.

**If adopted.**
- make_paper_beats.py output (beats/top, beats/full) is the L1/L2 base.
- In rehaul/01_introduction.tex and 09_conclusion.tex, the header-only bold leads ("Design.", "Benchmark and
  evaluation.", "Contributions.", "Application...") get one claim sentence each, from the fresh beats as reconciled
  under C1-C7.
- decisions.yaml is seeded with D1-D20 plus the dated quotes above.
- Nothing is deleted.
- The fresh file stays as a comparison artifact. Its `% CONFLICT` notes are answered by this deliberation.

**Tier.** T2 (inferred from A1, A2 and the corrections already embedded; the author can veto in one line). The author
commissioned the comparison personally, so the digest should show the six contradictions above. If the adjudicator
holds that the author reserved this choice, the T3 question would be: "Keep the current bullets as the first draft of
the bullet layer (the default), filling the 14 header-only top beats from the fresh version, or regenerate the bullets
from the fresh beats?"
