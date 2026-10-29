# Solver: the author's record (deliberation of 29 October 2026)

Lens: what the author has said and chosen before (dated quotes, decisions, reversals, rejections), so that each answer is
the one the author would most likely give. Items: the seven `% CONFLICT:` comments of `writeup/fresh/paper_fresh_beats.tex`
(C1-C7) and Q8 (restart from the fresh version or keep the current bullets as the first draft of the bullet layer).
This solver did not read the other solvers' files.

## Sources and method

- `writeup/INTEGRATION_PLAN_2026-10-26.md` D1-D20 (lines 36-55), budget table (lines 13-27).
- `writeup/reports/AI layered paper writing system.md`: A1-A6 (lines 31-47), §8.1 layer sequence, §8.9 Q8 (line 1403),
  §8.10 tiers, Phase 2 plan ("serve the current top beats, seeded from D1-D20, as the simple starting state").
- Dated `(user|author, ...)` comments in `writeup/paper_rehaul.tex` and `writeup/rehaul/*.tex` (about 970 tags; Oct 26-28
  tags concentrate on: Sections 6-7 merged, conventional titles, "pooled" pruned, the alternative-data edit, v2).
- Commit messages since 24 Oct (`git log --format=%B`).
- The author's own chat messages, extracted by a streaming script from the 9 transcripts in
  `C:\Users\james\.claude\projects\C--Users-james-CC-Allowed-harxhar-0dte-professor\*.jsonl` (910 typed messages plus
  308 queued ones; tool results, agent hand-backs and system text skipped). The transcript timestamps are UTC; the local
  time (PDT, UTC-7) is given first below. Where an author reply answers an agent question, the question was read too.

## Precedence timeline (the dated decisions these items turn on)

| Local time (PDT) | Author's words (verbatim) | Where |
|---|---|---|
| 12 Oct 23:19 | "my professor wants to tell the following story: we want to say that each individual bucket didn't see much improvement, but all features saw a discernible difference." | transcript 531849d7, 2026-10-13T06:19Z |
| 13 Oct 19:47 | "fuck the headline. we need to write the paper to the prof's specification" | 531849d7, 2026-10-14T02:47Z |
| 15 Oct 15:49 | "there needs to be a very clean table in section 6 on the out of sample performance" | 531849d7, 2026-10-15T22:49Z |
| 15 Oct 18:11 | "I need the last 30 min scheme to be moved to the end of the economic application section as a final cherry on top instead of competing with the exogenous feature signal story" | 531849d7, 2026-10-16T01:11Z |
| 22 Oct 14:00 | "we aren't presenting the closing-interval models in the experiments section. It should only be mentioned in the last subsection where we describe the closing-interval models and their results" | bd4224e0, 2026-10-22T21:00Z |
| 25 Oct 12:32 | "I want the language of our paper to focus on the alternative data aspect of our features. Don't over do it ... emphasize the option trading is an application of these RV forecasts ... In the conclusion, we should definitely compare and contrast our trading results with the trading results of their paper" | bd4224e0, 2026-10-25T19:32Z |
| 25 Oct 18:23 | "We need a clean apples to apples comparison ... one consistent training design for all models" | 2026-10-26T01:23Z |
| 25 Oct 18:35 | "why is closing bar the paper's target??" | 2026-10-26T01:35Z |
| 25 Oct 19:14 | "1. every bar" (D1) | 2026-10-26T02:14Z |
| 25 Oct 23:35 | "no, we can't have it be the primary benchmark. we need all closing interval models to be corralled into that extra section" (D3) | 2026-10-26T06:35Z |
| 25 Oct 23:36 | "the primary benchmark is OLS on har for rth13" (D2) | 2026-10-26T06:36Z; plan line 37 |
| 26 Oct 14:48 | "yes, set the budget for jfec" (budget table; online appendix lists "CI-HAR and closing-interval section") | 2026-10-26T21:48Z; plan lines 13-27, commit ee1e900 14:51 |
| 26 Oct 17:13 | "wait, the qlike story is the pooled story remember? stop framing the last 30 min as the main story" | 2026-10-27T00:13Z |
| 26 Oct 17:23 | "should we also do ablations that correct for this quirk so that the paper is able to claim better usual predictive power from the other variables?" | 2026-10-27T00:23Z |
| 26 Oct 17:56 | "why is the 1600 bar still in the main paper??" (D16) | 2026-10-27T00:56Z; commit 7f7894a 18:00 |
| 26 Oct 18:58 | "let's quotient out futures' realized moments too then. those are well studied. how do the exogenous features contribute when these 3 (the har baseline, the well-studied moments, and the FOMC days) are accounted for?" (D18) | queued message 2026-10-27T01:58Z |
| 26 Oct 19:02 | "stop the forest run, run the HAR+RM+FOMC arm" | 2026-10-27T02:02Z |
| 26 Oct 19:05 | "keep the econ application as is" (answer to option (a) "Keep it as the application, with no 16:00 accuracy statistics around it") | 2026-10-27T02:05Z; agent question 2026-10-27T01:02Z |
| 27 Oct 12:02 | "The abstract wording should be neither. Both miss the point of the bigger story" | 2026-10-27T19:02Z |
| 27 Oct 18:08 | "Why are we doing PLS? Don't change up the methodology, otherwise we aren't systematically comparing apples to apples. The linear no-moments factor test must use exactly the same design as the earlier linear factor test and the tree version; only the base changes." | 2026-10-28T01:08Z |
| 27 Oct 22:51 | "I feel like there needs to not just be a rampage of things we explored and we need to consolidate everything into a clear concise statement" | 2026-10-28T05:51Z |
| 28 Oct 00:42 | "I approve v2 preliminarily. make sure it's easy to revert if I find something wrong" (D20; v2 moved the attribution table and the ladder figure to the appendix) | 2026-10-28T07:42Z |
| 28 Oct 15:09-16:15 | "so what exactly is the story?" -> "when I say alternativbe data, I'm not necessarily talking about those 3 series" -> "draft the edit" -> "both" (abstract and §6.3 edit applied: commit 3c6ccc4 16:18) | 2026-10-28T22:09Z-23:15Z |
| 28 Oct 16:30 | "can we add the tables to top beats?" | 2026-10-28T23:30Z |
| 28 Oct 16:40 | "sections 6 and 7 were never meant to be split. it needs to be one experiments section" | queued 2026-10-28T23:40Z; commit 7ca4268 |
| 28 Oct 16:52 | "in the tables, we need to make it abundantly clear what positive and negative deltas mean" | 2026-10-28T23:52Z |
| 28 Oct 17:01-17:04 | "we need to rerun the ladder every bar to have qlike stats no?" -> "go" (addendum K, commit 1152482) | 2026-10-29T00:01Z, 00:04Z |
| 28 Oct 17:06 | "also, why aren't there any tables on attribution numbers of feature importance?" | queued 2026-10-29T00:06Z |
| 28 Oct 17:10 | "make the section and subsection titles normal like you'd find in the literature, not with ai prose" | 2026-10-29T00:10Z; commit 3f57938 |
| 28 Oct 17:50 | "I need a better reconciliation of the attribution with the ablations" | 2026-10-29T00:50Z |
| 28 Oct 17:51 | "prune the wording 'pooled' again" | 2026-10-29T00:51Z; commit ae0f526 |
| 28 Oct 17:57 | "'In the linear test the realized moments are folded in with 36 other series' they're cheap, and they make the results apples to apples. run it." | 2026-10-29T00:57Z; quoted in commit a38439f |
| 28 Oct 23:22 | A1: "There needs to essentially be a way for a human's judgement at any top layer to trickle down and stay locked in for every lower layer. For example, if I decide that the main story is to compare on every RTH bar over 20 years of sample, that decision needs to be reflected everywhere those results are reported or used to back up any claim or compared against, etc" | 2026-10-29T06:22Z; report line 31 |
| 28 Oct 23:26 | A2: "thus, the paper needs to start from a simple unedited state and, through my reading and correction of the simple version, end in a finished edited state" | 2026-10-29T06:26Z; report line 34 |
| 28 Oct 23:32 | "let's do a version where we start over just to compare" | 2026-10-29T06:32Z; commit 89c359e |
| 28 Oct 23:37 | A4: "no" (Rule NT never applies to text the author typed) | 2026-10-29T06:37Z; report line 40 |

---

## C1. Introduction framing

**Answer.** Lead with the every-bar question and the full-set gain against the OLS-HAR, with the MSE clause (D1, D2, D7).
The opening question is the general one, "what exogenous predictors add to every-bar forecasts", not D18's. D18's
question and answer come second, as the decomposition contribution: the FOMC calendar and the futures' realized moments
carry at least 89% of each class's gain; beyond them the other 36 series add nothing to the linear models and a small,
sparse gain to the boosted trees. This is the order of the current abstract and introduction; the fresh version's first
beat (line 244) should be generalized.

**Confidence.** 0.75. **Tier.** T2 (inferred preference; the author can veto in one line).

**Decisive evidence.**
- The last framing the author approved is this order. On 28 Oct he asked "so what exactly is the story?" (15:09) and got
  "Exogenous predictors improve ... significantly in every model class. But almost all of the gain comes from two
  well-known sources ... Everything else adds little." His replies corrected the definition of alternative data and the
  denseness reading, not the order. He then approved the edit ("both", 16:15; commit 3c6ccc4). That abstract opens "We
  ask what exogenous predictors ... add" and gives the full-set gain against the OLS-HAR before the decomposition
  (`writeup/paper_rehaul.tex:278-284`).
- A1 (28 Oct 23:22) names the main story: "the main story is to compare on every RTH bar over 20 years of sample". That
  describes the D1 comparison, not the beyond-base question.
- The budget the author asked for (26 Oct, "set the budget for jfec"; plan line 15): "Every-bar question; 3 numbered
  contributions with effect sizes".
- The professor's specification (12-13 Oct): "all features saw a discernible difference", and "we need to write the
  paper to the prof's specification". The title has not changed: "Exogenous Predictors in Pooled Intraday Forecasts".
- D18's source quote is a question about an analysis ("how do the exogenous features contribute when these 3 ... are
  accounted for?"), not an instruction to reframe the paper. D20 (28 Oct) then folded v2's separate "Beyond HAR + RM +
  FOMC" section into the predictor analysis. The beyond-base result sits as a subsection, §6.3.3, under the full-set
  headline.

**Counter-evidence weighed.**
- On 26 Oct at 19:02 the agent asked "Reframe the headline around the three-part reference?", and the author replied
  "stop the forest run, run the HAR+RM+FOMC arm". That ran the arm without answering the framing question.
- "The abstract wording should be neither. Both miss the point of the bigger story" (27 Oct) was read by the agent as
  "the bigger story is the decomposition". The author never confirmed that reading. The abstract he later approved
  still puts the decomposition after the headline.
- The 25 Oct wish to "focus on the alternative data aspect" came with "Don't over do it". The beyond-base answer is
  null for the linear models, so leading with it would make the paper's first claim a near-null.

**What changes if adopted.** The current intro and abstract are unchanged. In the fresh version, beat 1 (line 244)
becomes the general every-bar question, and D18's question opens contribution 2 (lines 257-268). Open dependency, not
decided here: the agent's 27 Oct proposal to compute strategies on the HAR + RM + FOMC forecasts, so that the economics
sentence answers the decomposition question, was never approved and has not been run.

---

## C2. Table 1's tests

**Answer.** Every main-text test is on every bar, and each forecast gets two tests:
1. Against the OLS-HAR, the primary test, for every row. This includes the trees' HAR-set forecasts: LightGBM's HAR
   set is -0.9% against the OLS, DM -2.52, p 0.012.
2. Each class's full set against its own HAR set, the paired test that D2 keeps.

Put both in Table 1. Add "% vs OLS-HAR" and "DM vs OLS-HAR" columns, every bar, next to the existing within-class DM
column. The every-bar columns of `tab:rh_oos_vsols` merge into Table 1. Its 16:00 columns move to Appendix
`app:rh_closing`.

**Confidence.** 0.85 on the content (both tests, every bar only); 0.7 on the layout (one table rather than two).

**Tier.** T1 on the content; the layout is T2.

**Decisive evidence.**
- D2 (plan line 37; author 25 Oct 23:36, "the primary benchmark is OLS on har for rth13"): "Every main-text forecast
  (incl. the trees' full sets) is tested against the RTH13 OLS-HAR; the within-class full-vs-HAR-set pairs stay as the
  paired comparison of what the exogenous predictors add". Both tests are named, so there is no conflict to resolve.
- A1 (28 Oct 23:22): a decision "needs to be reflected everywhere those results are reported or used to back up any
  claim or compared against". Today the trees' vs-OLS tests appear only in the text (`06_results.tex:121-134`) and in a
  generated table that no file loads.
- D16 (26 Oct 17:56, "why is the 1600 bar still in the main paper??"; commit 7f7894a) bars the 16:00 columns from the
  main text. D16 sends them to `app:rh_closing`.
- Layout: the author asked for "a very clean table in section 6 on the out of sample performance" (15 Oct). The main
  text runs to about 27 pp against the 22-24 pp budget, so adding columns is better than adding a float. The 28 Oct
  rule also applies: "make it abundantly clear what positive and negative deltas mean" (a sign key in the notes).

**What changes if adopted.**
- `writeup/make_rehaul_oos_rerun.py`: Table 1 gains two columns and a sign key. `tab:rh_oos_vsols` keeps only the 16:00
  columns and is loaded in `A_closing.tex`.
- The "Against the primary benchmark" bullet (`06_results.tex:121`) cites the new columns.
- The ridge, lasso and elastic-net HAR-set rows print 0.0% with DM 8.50 / 4.40 / 4.85 against the OLS. The differences
  are below 0.005% (`\rhorVsOlsPctLinHarMaxAbs` = 0.00) but one-signed. The table note should say so, so that the
  printed DM does not read as a real difference. The fresh version's Forecast-accuracy beats then cite Table 1
  directly.

---

## C3. Linear models beyond the base

**Answer.** Compare each class with its own penalized base: the same estimator and frozen hyperparameters on HAR + RM +
FOMC. That is the ablation's add_rm_fomc arm, confirmed by the penalized factor test B. The claim is:

> Beyond HAR + RM + FOMC the other 36 series add no significant accuracy to the penalized linear models.

The numbers behind it:

| | Ridge | Lasso | Elastic net |
|---|---|---|---|
| Ordinary days | +0.15%, p 0.75 | 0.00%, p 0.995 | -0.57%, p 0.19 |
| Every bar | +0.1% | -0.6% | -0.6% |

None is significant; the smallest every-bar p is 0.161.

The comparison with the OLS on the base (elastic net -2.62% on ordinary days, +2.74% on every bar) leaves the main text.
At most it is reported in the appendix as the split of commit d7e3b58: penalizing the base's columns accounts for -2.06%
(p 0.018) and the series for -0.56% (p 0.19). That is an estimator effect, not a series effect.

**Confidence.** 0.85. **Tier.** T1.

**Decisive evidence.**
- 27 Oct 18:08: "Don't change up the methodology, otherwise we aren't systematically comparing apples to apples ... only
  the base changes." The trees' beyond-base test sets each tree against its own base, so the linear test must do the
  same.
- 28 Oct 17:57: "they're cheap, and they make the results apples to apples. run it." This commissioned exactly the
  penalized test, "each class against its own base and full set" (commit a38439f; `RULE_linear_pen.txt`). The author
  paid for this comparison so that it would replace the OLS-base one.
- The same preference appears throughout: "We need a clean apples to apples comparison" (25 Oct), "can we run the base 5
  no fomc to do proper apples to apples?" (26 Oct), and "not apples to apples" on MinGRU (13 Oct).
- D2's primary benchmark is the OLS on the HAR set, not the OLS on HAR + RM + FOMC. D2 also keeps within-class pairs as
  "the paired comparison of what the exogenous predictors add". The beyond-base question is the same kind of question.
  The fresh version's conflict comment reads D2 as covering an OLS base; that reading is not supported.

**What changes if adopted.**
- Main text: §6.3.3's "Small, and only in the boosted trees" and "Sparse, not dense" (`08_feature_importance.tex:252,
  270`) already compare within class, so they stay. The fresh version drops beat 467 (the OLS-base bullet).
- A generator for `results/beyond_trees_2026-10-27/linear_pen` writes macros and a table for test B, replacing the
  unwired OLS-base `tab:rh_beyond_factor_lin` (whose reading, "DENSE", the penalized test overturns for ridge and
  lasso).
- By the same author rule, the main-text factor-augmented HAR (`tab:rh_factor`, fitted by OLS) should report the
  penalized test A beside or instead of the OLS version. Its readings: ridge DENSE, lasso INTERMEDIATE, enet DENSE
  (commit 30b0470). This is T2, inferred from the same quote.

---

## C4. The ladder figure's 16:00 panel

**Answer.**
- The 16:00 panel leaves `fig:rh_ladder` and goes to `app:rh_closing`. This is T0: D16 says 16:00 accuracy results
  "live in Appendix app:rh_closing".
- The figure keeps only the every-bar panel. It is redrawn from the every-bar refits of addendum K when they land; until
  then it uses the current refits, labelled monthly.
- Placement: keep the figure in the appendix (§F.6), as approved in v2 (D20). The main text carries the every-bar rungs
  through the sparsity table (`tab:rh_channels_sparse`, Panel B, rebuilt on the every-bar ladder) and the text.

**Confidence.** 0.95 that no 16:00 panel appears outside `app:rh_closing`; 0.7 on placement.

**Tier.** T1 overall (T0 for the panel, T2 for placement).

**Decisive evidence.**
- D16 (26 Oct 17:56; plan line 51; commit 7f7894a, "ladder figure moved").
- A1 (28 Oct 23:22): every-bar "everywhere those results are reported".
- 28 Oct 17:01: "we need to rerun the ladder every bar to have qlike stats no?", then "go" (addendum K, about 860 CPU-h,
  commit 1152482). The author paid to put the ladder on Table 1's footing, so the every-bar version is the one to show.
- Placement: on 26 Oct 17:52 the agent asked "Can the attribution table and the ladder figure move to the appendix?".
  The author approved v2, which made that move (D20, 28 Oct 00:42). That is later than the budget table (26 Oct), which
  had put "feature-count ladder (1 figure)" in the main text. His later complaint about missing tables (28 Oct 17:06)
  named attribution, not the ladder.

**What changes if adopted.**
- `writeup/make_rehaul_ladder.py` writes two figures: `fig:rh_ladder` (every bar, `A_fi.tex` §F.6) and a 16:00 figure
  loaded in `A_closing.tex`.
- The caption drops "and on their 16:00 bar (right)".
- The intro's pointer (`01_introduction.tex:136-137`) stays.
- Panel B of `tab:rh_channels_sparse` is regenerated from the every-bar ladder once addendum K is scored.

---

## C5. SHAP grouping against D18's base

**Answer.**
- Re-aggregate the existing TreeSHAP values, and the linear models' coefficient shares, on the ablation's partition:
  HAR, calendar, and the eight groups. Report the D18 split: HAR + calendar | RM + FOMC | the six remaining groups,
  each with the same moving-block bootstrap interval. This needs no refit.
- Build the reconciliation table the author asked for, in §6.3.5 of the main text. Each predictor group gets one row:
  its attribution share (trees on all bars, statement days and ordinary days; linear models) next to the change when the
  group is added alone and when it is removed, by day type.
- Keep the standard/non-standard split in `tab:rh_fip_shares` in Appendix F, labelled as the pre-registered answer to
  the professor's question (`results/tree_shap_w3000_2014start_2026-10-23/RULE_shap.txt`).
- The six-group share the fresh version leaves pending is computable now from Panel A. For LightGBM it is 8.8 + 5.5 +
  5.1 + 0.8 + 2.3 + 0.3 = 22.8%.

**Confidence.** 0.8. **Tier.** T1.

**Decisive evidence.**
- 28 Oct 17:06: "also, why aren't there any tables on attribution numbers of feature importance?" The agent answered
  that the table had moved to the appendix in v2 and proposed "one compact table to §6.3.5. One row per predictor group
  ... its attribution share ... its accuracy contribution". The author did not answer the "Want that table?" question,
  but 44 minutes later (17:50) wrote: "I need a better reconciliation of the attribution with the ablations".
- Reconciling attribution with the ablations requires both on the same grouping, and the ablations' grouping is D18's.
- A1 (28 Oct 23:22): a top-layer decision, here D18's base, must "trickle down and stay locked in for every lower
  layer".
- The standard/non-standard classes come from a rule fixed on 23 Oct for the professor's question ("how do the
  nonlinear models respond to the dataset's non-standard variables"). They predate D18 (26 Oct). The author's line
  "write the paper to the prof's specification" (13 Oct) argues for keeping them in the appendix, not deleting them.

**What changes if adopted.**
- `writeup/make_rehaul_fi_pooled.py`: Panel B, the D18 partition with intervals, replaces standard/non-standard in the
  main-text version. Panel C (tree minus linear) is redone on the six-group share. New macros cover the six-group share,
  which removes the fresh version's `\rhPending` (line 498).
- A new reconciliation-table generator, with no compute, reads the ablation and channel numbers.
- Optional, not decided here: the agent's "bridge" measure (permute each group at the 126 SHAP origins, no refit, about
  50 CPU-h on CARC). It needs launch approval under "Don't launch anything new without asking".

---

## C6. The application trades on the 16:00 forecast

**Answer.** Keep the application as it is: the 15:30 straddle signed by the forecast of the closing bar, with no 16:00
accuracy statistics in the main text. Add at most one sentence without numbers in §7.1 that points to `app:rh_closing`
for the accuracy of the forecast that signs the trade.

**Confidence.** 0.9 that the application stays unchanged; 0.7 on the pointer sentence.

**Tier.** T1 (the pointer is a T2 detail).

**Decisive evidence.**
- 26 Oct 18:02 the agent asked: "(a) Keep it as the application, with no 16:00 accuracy statistics around it. My
  recommendation ... (b) Move the whole application to the appendix ... (c) Redesign it around every bar". The author
  answered at 19:05: "keep the econ application as is". He chose (a), whose wording itself resolves the conflict, 69
  minutes after D16. D16's row records both (plan line 51).
- The author knowingly set the forecast target apart from the trade: "why is closing bar the paper's target??" (25 Oct)
  and "stop framing the last 30 min as the main story" (26 Oct). On 25 Oct: "the option trading is an application of
  these RV forecasts".
- Pointer: the author's pattern is to disclose rather than omit (D8 window "disclosed", D14 "disclose only", D15 "keep
  and disclose"). A sentence without numbers prints no 16:00 statistic.

**What changes if adopted.** `07_economics.tex`, Trading strategy: one pointer sentence; nothing else. The fresh
version's beat 523 already does this.

---

## C7. Closing-interval models: final main-text section or appendix

**Answer.** Appendix only (`app:rh_closing`, `A_closing.tex`), which is where the current paper already has them. D3's
aim, keeping every closing-interval model corralled in one place apart from the main story, is met by that appendix
section. The main text keeps only the intro's pointer to it.

**Confidence.** 0.9. **Tier.** T1 (a later decision supersedes the earlier one).

**Decisive evidence.**
- D3's words (25 Oct 23:35): "we need all closing interval models to be corralled into that extra section". They follow
  23:33, "the ci har stuff is only important for the last subsection that details concentrating training on closing
  days". The concern was isolation from the benchmark, not placement in the main text.
- Two later decisions put the material in the appendix:
  - the budget the author asked for (26 Oct 14:48), whose online-appendix line lists "CI-HAR and closing-interval
    section" (plan line 26);
  - D16 (26 Oct 17:56), "why is the 1600 bar still in the main paper??". The closing-interval models report only
    16:00-bar results.
- The 15 Oct "final cherry on top" wish and the 22 Oct "last subsection" both predate D16 and the budget, so they are
  superseded.

**What changes if adopted.** Nothing in the current paper. In the fresh version, the conflict at line 601 is resolved
as written (appendix only).

---

## Q8. Restart from the fresh version, or keep the current bullets as the first draft of the bullet layer?

**Answer (recommended default).** Keep the current bullets as the first draft of the bullet layer (L2). Use the fresh
version as the author's comparison for the top-beat layer (L1). Port from it only what the author marks after
comparing, for example its one-paragraph "Governing claim" and the leaner beat order.

**Confidence.** 0.6. **Tier.** T3. The author reserved the comparison ("just to compare"), §8.9 lists it as his
decision, and A2 (start simple) pulls against A1/A4 (locks and protected author text) with no later quote to settle
them.

**Decisive evidence.**
- "let's do a version where we start over just to compare" (28 Oct 23:32): the fresh version was commissioned as a
  comparison, not as a restart.
- The fresh version was written from D1-D20 only (commit 89c359e), so it breaks at least five later instructions:
  - It has two results sections ("Forecast accuracy", "Beyond HAR, realized moments and the FOMC calendar"), against
    "sections 6 and 7 were never meant to be split. it needs to be one experiments section" (28 Oct 16:40).
  - About 10 prose uses of "pooled", two in the abstract, against "prune the wording 'pooled' again" (28 Oct 17:51;
    commit ae0f526 kept one use in the abstract).
  - Prose-style titles ("Beyond HAR, ...", "Economic value: 0DTE straddles"), against "make the section and subsection
    titles normal like you'd find in the literature" (28 Oct 17:10).
  - No Related-work section; the budget has one (1.5 pp), and the author gave related-work direction on 24 Oct.
  - No comparison with Maisch et al. in the conclusion or the economics, against "In the conclusion, we should
    definitely compare and contrast our trading results with the trading results of their paper" (25 Oct; "Maisch"
    appears on 12 lines of the current `09_conclusion.tex`).
- A1 says a judgment "stay[s] locked in for every lower layer", and A4 says author-supplied text is protected verbatim.
  The current bullets carry the author's corrections since 15 Oct (about 970 dated tags), including the v2 approval
  (D20), the alternative-data edit, the D19 claim check, the merged Results section, the titles and the "pooled" pruning.
  A restart would have to re-apply every one of them.
- The report the author asked to keep "as the research you pull from" plans Phase 2 as "serve the current top beats,
  seeded from D1-D20, as the simple starting state" (report line 1351).
- For restarting: A2 (28 Oct 23:26), "the paper needs to start from a simple unedited state". The current bullets are
  an edited state, and the author had complained of "a rampage of things we explored" (27 Oct) and of length ("the
  paper wo reference or appendix is at like 50 pages", 26 Oct).

**What changes if adopted (default).** The current `paper_rehaul.tex` / `rehaul/*.tex` stay the source of L2. The fresh
file stays in `writeup/fresh/` as an L1 candidate, and C1-C7 are applied to the current text.

**Question for the author:**

> The fresh version (commit 89c359e) was written from D1-D20 alone. It breaks five of your later instructions: one
> Results section, "pooled" pruned, conventional titles, a Related-work section, and the Maisch comparison in the
> conclusion. The current bullets carry your edits since 15 October. Keep the current bullets as the first draft of the
> bullet layer and take from the fresh version only what you mark after comparing (default)? Or restart: regenerate a
> simple version under every lock (D1-D20 plus your dated quotes) and re-apply your edits to it?

---

## Summary

| Item | Answer | Conf. | Tier |
|---|---|---|---|
| C1 | Lead with the every-bar full-set gain vs the OLS-HAR plus the MSE clause; D18's beyond-base answer is contribution 2 | 0.75 | T2 |
| C2 | Two every-bar tests per forecast: vs the OLS-HAR (all rows) and full vs own HAR set; both in Table 1; the 16:00 columns go to `app:rh_closing` | 0.85 | T1 (layout T2) |
| C3 | Each class vs its own penalized base; "no significant linear gain beyond HAR + RM + FOMC"; the OLS-base comparison only as the appendix estimator-effect split | 0.85 | T1 |
| C4 | No 16:00 panel outside `app:rh_closing`; every-bar ladder (addendum K) in Appendix F.6 per v2, rungs in the main-text sparsity table | 0.95 / 0.7 | T1 |
| C5 | Re-aggregate the SHAP shares on D18's partition (no refit) plus a main-text attribution-vs-ablation table; standard/non-standard stays in the appendix | 0.8 | T1 |
| C6 | Keep the application as is (author chose option (a)); at most one sentence without numbers pointing to `app:rh_closing` | 0.9 | T1 |
| C7 | Appendix only, as now; D16 and the budget supersede D3's "final section" | 0.9 | T1 |
| Q8 | Default: keep the current bullets as L2; the fresh version is the author's L1 comparison | 0.6 | T3 |

Scripts used, read-only, in the scratchpad (`sar_*.py`, `extract_user.py`, `queued.py`). The scratchpad is shared with
other solvers, and one generic script name (`ctx.py`) was overwritten by another process mid-run. This solver's
outputs were re-run under `sar_` names.
