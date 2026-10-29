# Adjudication, deliberation of 29 October 2026

Adjudicator under A6 and §8.10 of `writeup/reports/AI layered paper writing system.md`. Inputs: the five solver files
in this folder (evidence, author record, referee for C1-C7 and Q8; process risk and process goals for Q10-Q13). Nothing
in the paper, the report, the sync code or CLAUDE.md was edited; nothing was committed, pushed, pulled or launched.

**What was checked against the repo** (beyond reading the solvers):
- `INTEGRATION_PLAN_2026-10-26.md` D1-D20; `SESSION_HANDOFF_2026-10-26.md:41`; commits 3c6ccc4, 5554817, 7f7894a,
  a38439f, 30b0470, d7e3b58.
- The live abstract (`paper_rehaul.tex:258-284`), introduction, `06_results.tex`, `08_feature_importance.tex`,
  `A_results.tex`, `A_fi.tex`, `A_closing.tex`, `07_economics.tex`.
- Generated numbers: `rehaul_ablation_numbers.tex`, `rehaul_ablation_table.tex`, `rehaul_oos_vs_ols_numbers.tex`,
  `rehaul_close_accuracy_numbers.tex`. SHAP shares recomputed from `m1_buckets.csv`.
- Penalized-test results (`linear_pen/`, `factor_pen/`, `pen_split.csv`) and every `.py` that reads them.
- Sync code: `overleaf_git.example.sh` (full read), `flatten.py` (map write order), `unflatten.py`, `autopush.sh`. The
  live `overleaf_git.sh` differs from the example only in the token comment line (diffed with `olp_*` masked).
- The local Overleaf clone's log (`../harxhar-overleaf`), and Overleaf's git-integration docs (fetched).
- The author's typed messages and the agent's questions, 26-29 Oct, from transcript `bd4224e0…jsonl`, read-only with
  scratchpad scripts `adj/adj_*.py` (UTC timestamps; PDT = UTC-7).

**Rules applied.** Tiers as in §8.10. Precedence: a later dated author decision or quote supersedes an earlier one. A
majority answer was kept only when a refutation attempt against the repo failed.

---

## Summary

| Item | Final answer (short) | Conf. | Tier | vs. solvers |
|---|---|---|---|---|
| C1 | Research-question beat states two levels (gain vs OLS-HAR on every bar; what the 36 series beyond HAR + RM + FOMC add). First result: full-set gain vs OLS-HAR plus the MSE clause. Then base share, then the 36-series answer. Abstract order unchanged. | 0.75 | T2 | Synthesis; all three agree on the result order |
| C2 | Every Table 1 row tested against the OLS-HAR on every bar, plus the within-class full-vs-HAR DM. 16:00 columns to `app:rh_closing`. | 0.90 | T1 (layout T2) | Unanimous |
| C3 | Each class against its own base. Scope "on every bar and on ordinary days". Disclose OLS on HAR + RM + FOMC. Main-text factor test moves to the penalized version. | 0.90 | T1 | Unanimous; one extension adopted |
| C4 | No main-text ladder figure (D16, D20 supersede the budget row). 16:00 panel moves to `app:rh_closing` when addendum K's figure is drawn. | 0.90 / 0.60 | T0 / T2 | Merges both readings |
| C5 | Add a D18 grouping (labelled post hoc) to the SHAP generator; pre-registered split stays in the appendix; build the reconciliation table the author asked for. | 0.85 / 0.70 | T1 / T2 | Unanimous core; table adopted |
| C6 | Application unchanged; one no-number pointer to `app:rh_closing`. | 0.85 | T1 | Unanimous |
| C7 | Appendix only (status quo); amend D3's row. | 0.93 | T0 | Unanimous |
| Q8 | Recommended: L1 from the fresh version's claim beats, corrected; L2 = current bullets reconciled to the locked L1. | 0.65 | **T3** | Majority tier T2 refuted |
| Q10 | Keep the default. Held edits stay visible as a draft-only `\heldedit` note (final build fails until resolved), not reverted. | 0.75 / 0.65 | T2 | Held-edit split resolved for `\heldedit` |
| Q11 | All human text pulled from Overleaf is exempt from scrub and Rule NT; no byte-lock. Provenance in a sidecar. | 0.85 / 0.65 | T1 / T2 | Mechanism split resolved for sidecar |
| Q12 | Visible text in; suggestions arrive as accepted text; agents never act on review state or settings. Comment digest deferred to the first coauthor comment. One read-only setting check. | 0.75 / 0.60 | T1 / T2; conditional T3 | Both kept, sequenced |
| Q13 | Keep git for all agent writes; no claudeleaf, no bot account. | 0.85 | T1 | Unanimous |
| X1 (F1) | Confirmed with a qualification: "beats augmented HAR models" is too broad. | 0.85 | T1 | Confirmed |
| X2 (F4) | Confirmed: "reverses" needs "in none significantly". | 0.90 | T0 | Confirmed |
| X3 | Partly confirmed: the vs-OLS DMs are printed in the text, but no table carries them. | 0.90 | T1 (with C2) | Partly confirmed |
| X4 | Confirmed: no generator reads the 28 Oct penalized tests or the split. | 0.95 | T0 | Confirmed |
| X5a, X5b | Both sync defects confirmed; two related defects found. | 0.90 / 0.85 | T0 | Confirmed |
| X6 | Confirmed: the bridge names the most recent editor of each batch; §8.5's "unknown" is wrong. | 0.85 | T0 | Confirmed |
| X7 | Confirmed conflict between the CLAUDE.md Overleaf rule and A5/A6. Needs the author's OK. | 0.90 | author OK | Confirmed |
| X8 | Confirmed: base-forecast strategies proposed 27 Oct, conditional on an unanswered question, never run. | 0.95 / 0.60 | T2 | Confirmed |

Only Q8 reaches the author as a question. X7 reaches him as a wording change that needs his OK. Q12 carries one
conditional question, asked only if a read-only check finds "track changes for everyone" on.

---

## C1. Introduction framing

| Solver | Answer | Tier |
|---|---|---|
| Evidence | D18's question is *the* research question; then the full-set gain vs OLS-HAR with the MSE clause; then base share; then the D18 answer | T2 (0.65) |
| Author record | General question ("what exogenous predictors add"); lead with the every-bar full-set gain vs OLS-HAR and the MSE clause; D18's question and answer as contribution 2 | T2 (0.75) |
| Referee | Research-question beat states D18's two levels; first result is the full-set gain vs OLS-HAR plus MSE; contributions: gain, channels, beyond base, application | T2 (0.75) |

**Where they actually differ.** All three put the full-set gain vs OLS-HAR (with the MSE clause) first among the
results, and all three put the D18 answer after the base share. The disagreement is one sentence: what the
research-question beat asks.

**Refutation attempts.**
1. *Does a later author act make D18 the paper's lead?* D18's source quote (26 Oct 18:58 PDT) asks about an analysis:
   "how do the exogenous features contribute when these 3 … are accounted for?". On 27 Oct (19:08Z) the agent asked
   "Decision 1: the bigger story: is it the decomposition?". No author message answers it (all typed author messages
   27 Oct 19:08Z to 29 Oct 09:00Z checked). On 28 Oct the author asked "so what exactly is the story?" (22:09Z). The
   reply opened with the full-set gain ("Exogenous predictors improve … significantly in every model class. But almost
   all of the gain comes from two well-known sources …"). He corrected the alternative-data scope and the denseness
   reading, not the order, said "draft the edit", then "both" (23:15Z; "both" = apply the draft and fix flatten).
   That is commit 3c6ccc4, which kept "We ask what exogenous predictors … add" and the full-set gain first, and made
   the third sentence the D18 answer. **Leading with D18 alone is refuted** as the author's latest choice.
2. *Does the handoff's open item "Headline under D18 … once the add_rm_fomc numbers are in" (`SESSION_HANDOFF:41`)
   keep this open?* The numbers came in 27 Oct; the 28 Oct approval is later and edits exactly that headline. Closed
   in effect.
3. *Does keeping a generic question ignore D18 and the 28 Oct alternative-data quote ("when I say alternativbe data,
   I'm not necessarily talking about those 3 series", 22:14Z)?* Partly yes: the live research-question bullet
   (`01_introduction.tex:49`) asks only "whether exogenous predictors improve forecasts". Under D18 that question's
   answer is largely a known result (OLS on HAR + RM + FOMC alone: -9.2%, `\rhabOlsAddRmFomcVsHarAllPct`). The
   25 Oct "focus on the alternative data aspect … Don't over do it" argues for naming the D18 question without
   making it the only one. **The author-record answer, unchanged, is refuted on this sentence.**

**Final answer.**
- Research-question beat, two levels: (a) do the exogenous series improve every-bar forecasts against the OLS-HAR
  (D1, D2); (b) what do the 36 series beyond the well-studied HAR terms, futures' realized moments and FOMC calendar
  add (D18).
- Three numbered contributions (the budget's "3 numbered contributions"):
  1. the full-set gain vs OLS-HAR, 6.0-9.0% (linear) and 11.2-14.9% (trees), significant in 6 of 6, with the D7
     MSE clause;
  2. the two channels: HAR + RM + FOMC delivers 89-101% of each class's gain (FOMC on statement days, RM on
     ordinary days);
  3. beyond the base: nothing significant in the penalized linear models on every bar; -1.6% / -1.8% in LightGBM /
     XGBoost, from a few series across groups (sparse, not dense).
- The application stays in two sentences.
- Abstract: order unchanged. Only X1's scope fix applies, which frees words.
- "Alternative data" keeps its narrow meaning (StockTwits, volatility demand). The 36 series are "the other
  exogenous series". This is what the applied 28 Oct edit did: its "Your call" on the definition went unanswered, and
  the author applied the draft. Calling VIX, spreads and constituents' returns "alternative data" would draw a
  referee objection.

**Tier T2** (0.75): wording of the research question is taste, implied by D18, the 28 Oct approval and the 25/28 Oct
alternative-data quotes.

**Actions.**
- `rehaul/01_introduction.tex`:
  - Research-question bullet (l.49): the two-level question.
  - Contributions: renumber to gain, channels, beyond base. The MSE item folds into contribution 1. The 36-series
    sub-bullet (l.139) becomes contribution 3. Contribution 1's title loses "beyond richer HAR models" (X1).
- `rehaul/09_conclusion.tex`: first bullet mirrors the order.
- No macro, generator or table change: `\rhorVsOlsRedLinRng`, `\rhorVsOlsRedTreeRng`, `\rhabBaseShareMin/Max`,
  `\rhabLgbm/XgbFullVsAddRmFomcAllPct`, `\rhabLinFullVsAddRmFomcAllPMin`, `\rhbyFLNSeries` exist.
- Fresh version: the `% CONFLICT` at l.240-242 is answered by this item.

---

## C2. Table 1's tests in the main text

| Solver | Answer | Tier |
|---|---|---|
| Evidence | Both tests on every bar for every row (vs OLS-HAR; full vs own HAR); 16:00 columns of `tab:rh_oos_vsols` to `app:rh_closing` | T1 (0.85) |
| Author record | Same; two columns added to Table 1; sign key | T1 content, T2 layout |
| Referee | Same, plus 90% MCS markers and a Giacomini-White sentence for nested comparisons | T0 tests, T1 layout |

**Refutation attempts.**
1. *Does D2 really ask for the trees' HAR-set rows?* D2 (`INTEGRATION_PLAN:37`): "Every main-text forecast (incl.
   the trees' full sets) is tested against the RTH13 OLS-HAR; the within-class full-vs-HAR-set pairs stay". The
   trees' HAR rows are main-text forecasts. Holds.
2. *Would the extra columns breach "a very clean table" (15 Oct)?* That is a layout matter, hence T2 for layout. A
   second float costs more pages; main text runs about 27 pp against 22-24.
3. *Numbers.* All six full sets reject against the OLS-HAR at p < 0.001
   (`rehaul_oos_vs_ols_numbers.tex:5-67`). LightGBM's HAR set is -0.9% (DM -2.52, p 0.012), so the text's "within
   0.9%" holds; XGBoost -0.3% (p 0.363); forest +0.1% (p 0.752). Linear HAR rows: 0.0% but DM 8.50 / 4.40 / 4.85.
   No conclusion changes.

**Final answer.** As the solvers. Table 1 gets "% vs OLS-HAR" and "DM vs OLS-HAR (p)" for all 12 rows and keeps the
within-class DM. A note says the linear HAR rows equal the OLS up to fifth-decimal differences, so a referee does not
read DM 8.50 as a real gap; alternatively, collapse them into one row "OLS (= ridge, lasso, elastic net on the HAR
set)". The 16:00 columns go to the A_closing accuracy table. Add one methods sentence with Giacomini and White (2006)
for nested comparisons under a fixed rolling window. The paper is silent on nesting: no `nested`, Clark or Giacomini
in `05_methods`, `10_statistical_tests`, `A_methods` or `references.bib`. MCS markers are optional (T2, low priority).

**Tier T1** (0.90): D2's text and the generated numbers settle the content. Layout T2.

**Actions.**
- `writeup/make_rehaul_oos_rerun.py`:
  - add the two columns to `tab:rh_oos` with a sign key;
  - stop emitting the 16:00 columns into any main-text float;
  - emit an A_closing 16:00 table (or `\input` the trimmed `tab:rh_oos_vsols` in `rehaul/A_closing.tex`).
- `rehaul/06_results.tex:121-133` ("Against the primary benchmark") cites the new columns.
- `paper_rehaul.tex:267`: the backing comment still says `\rhPending{rhorVsOlsLowerSig}`; the macro is 6 (T0).
- `rehaul/05_methods.tex`: the GW sentence. Fetch the BibTeX deterministically (rule 6 of the report).

---

## C3. Linear models beyond the base

| Solver | Answer | Tier |
|---|---|---|
| Evidence | Each class vs its own base; scope "every bar"; lasso statement-day caveat; one OLS-on-base sentence; penalized test A should replace the OLS one ("little", not "none") | T1 (0.85) |
| Author record | Each class vs its own penalized base; OLS-base split only in the appendix; penalized test A for `tab:rh_factor` (T2) | T1 (0.85) |
| Referee | Same; scope "every bar and ordinary days"; disclose OLS on HAR + RM + FOMC; add that row to `tab:rh_ablation` | T1 (0.90) |

**Refutation attempts.**
1. *The 27 Oct quote "must use exactly the same design as the earlier linear factor test" (OLS-based) argues for the
   OLS base.* Superseded. The 28 Oct quote "they're cheap, and they make the results apples to apples. run it."
   commissioned the penalized one-model test. `RULE_linear_pen.txt` states its purpose: "so that the linear and the
   trees' recovery shares are on the same footing".
2. *Numbers.* Own-base, every bar: ridge +0.1% (p 0.906), lasso -0.6% (0.312), enet -0.6% (0.161). Ordinary days:
   +0.2 / 0.0 / -0.6%, p >= 0.194. Statement days: lasso -6.4%, p < 0.001 (`rehaul_ablation_numbers.tex:3155-3157`).
   An unscoped "no linear gain" is literally false for one class on one day type. The scope survives; the claim does
   not survive unscoped.
3. The appendix already uses the within-class reading (`A_results.tex:434-442`, with the 27 Oct coordinator note
   that the OLS-base table is not wired). Consistent.

**Final answer.** As the solvers, with the referee's scope and the evidence solver's extension to test A:
- **Main text.** "On every bar and on ordinary days the other 36 series add no significant accuracy to any
  penalized linear model."
  - Plus one sentence: OLS on HAR + RM + FOMC is as accurate on every bar as the penalized full sets (-9.2% vs
    OLS-HAR), and penalization costs them the statement-day gain.
- **`tab:rh_factor` (§6.3.2).** Switch to the penalized test A, or report it beside the OLS version. Readings: ridge
  DENSE, lasso INTERMEDIATE, enet DENSE. Best k <= 3 is ridge pls3, R 0.23, not significant vs base (p 0.39), so
  "recovers none" becomes "little".

**Tier T1** (0.90).

**Actions.**
- New generator `writeup/make_rehaul_linear_pen.py` (or an extension of `make_rehaul_beyond_factor_lin.py`). It reads
  `results/beyond_trees_2026-10-27/linear_pen/{reading.json,scores.csv}`, `results/channels_2026-10-26/factor_pen/`
  and `linear/pen_split.csv`, and writes macros plus a table for test B, replacing the unwired OLS-base
  `tab:rh_beyond_factor_lin`.
- `writeup/make_rehaul_ablations.py`: fill the OLS cell for HAR + RM + FOMC (Panel D's OLS column is "--") or add a
  Panel C row "HAR + RM + FOMC (OLS)" from `\rhabOlsAddRmFomcVsHarAll*`.
- Scope fixes in `01_introduction.tex:139`, `09_conclusion.tex:55` and `08_feature_importance.tex`: "on every bar".
- `08_feature_importance.tex:236-243` rewritten on the penalized test A.
- Fresh version: drop beat l.467-469.

---

## C4. Ladder figure with a 16:00 panel

| Solver | Answer | Tier |
|---|---|---|
| Evidence | Already resolved: figure is in App. F.6; a 16:00 panel is allowed in the appendix; nothing to do | T0 |
| Author record | 16:00 panel leaves `fig:rh_ladder` for `app:rh_closing` (D16 says 16:00 results "live in" it); every-bar panel redrawn from addendum K | T1 (T0 panel / T2 placement) |
| Referee | Figure stays in the appendix (D16, D20 supersede the budget row) | T0 |

**Refutation attempts.**
1. *Does D16 require every 16:00 result to be in `app:rh_closing`?* D16's own implementation (7f7894a, listed in the
   D16 row as "Done") moved the two-panel figure to `A_fi`, not `A_closing`, and D20 then put it in F.6. So D16 as
   executed bans only the main text. The author-record T0 claim fails.
2. *Does anything still put the figure in the main text?* Only the budget row (26 Oct 14:48 PDT), superseded by D16
   (17:56) and D20 (28 Oct). Holds.
3. *A1* ("reflected everywhere those results are reported") favours an every-bar figure. Addendum K, every-bar refits,
   was approved 28 Oct ("go").

**Final answer.** No main-text action (T0). When addendum K is scored, `make_rehaul_ladder.py` draws `fig:rh_ladder`
on every bar only. The 16:00 panel becomes its own figure in `A_closing.tex` (T2, 0.60: tidier under A1, not
required by D16). Annotate the budget row as superseded.

---

## C5. SHAP grouping against D18's base

| Solver | Answer | Tier |
|---|---|---|
| Evidence | Add a D18 grouping, labelled post hoc; keep the pre-registered split in the appendix; one main-text clause | T1 (0.75) |
| Author record | Re-aggregate on D18's partition with bootstrap intervals; keep the registered split in the appendix; build a main-text reconciliation table (attribution vs add/remove, by group) | T1 (0.80) |
| Referee | Mechanical regrouping in the generator; keep the registered split | T0 (0.90) |

**Refutation attempts.**
1. *Would replacing the classes break the registered rule?* `RULE_shap.txt:150`: classes "not revised after seeing
   results; anything added later is labelled post hoc". So add, don't replace. Holds.
2. *Numbers.* Recomputed from `m1_buckets.csv`. Six groups as a share of total attribution: LightGBM 22.8%, XGBoost
   21.3%, forest 20.0%. As a share of exogenous attribution: 48.0 / 45.2 / 47.4%. Gain beyond the base:
   100 - BaseShare = 10 / 11 / 4%. Matches the solvers.
3. *Is the reconciliation table still open?* Yes. The author wrote 28 Oct 17:50 PDT: "I need a better reconciliation
   of the attribution with the ablations". The agent proposed a no-compute table (29 Oct 00:51Z) and asked "Which of
   these do you want?" (01:23Z). No answer, and no commit since builds it.

**Final answer.**
- **T1** (0.85). D18 grouping added as a post hoc panel with macros (`\rhfipBeyondMin/Max` 20.0-22.8;
  `\rhfipBeyondExogMin/Max` 45.2-48.0). Point shares now. Intervals after re-reducing the per-origin arrays with a
  third class map; this needs no SHAP refit, but the arrays may have to be fetched from CARC. The registered split
  stays in Appendix F, labelled pre-registered.
- **T2** (0.70). The reconciliation table in §6.3.5, one row per group on the D18 partition: attribution share
  (trees, all / statement / ordinary days; linear) next to the add-alone and drop-one changes. No compute. Page cost:
  about 1/3 page.

**Actions.**
- `writeup/make_rehaul_fi_pooled.py`: D18 panel in `tab:rh_fip_shares` and the new macros.
- New `writeup/make_rehaul_fi_reconcile.py`, reading existing ablation and channel macros/CSVs.
- `08_feature_importance.tex` §6.3.5: one clause and the table.
- Fresh `\rhPending` at l.498 resolved.

---

## C6. The application trades on the 16:00 forecast

All three: keep the application as is; add one sentence with no numbers pointing to `app:rh_closing`.

**Refutation attempt.** Does a pointer breach D16? The author chose option (a), "Keep it as the application, with no
16:00 accuracy statistics around it" ("keep the econ application as is", 26 Oct 19:05 PDT). A pointer prints no
statistic. `07_economics.tex` has no pointer today (grep for `app:rh_closing` finds none). The refutation fails.

**Final answer.** As the solvers. **T1** (0.85).

**Action.** One sentence in `07_economics.tex`, Trading strategy (l.79ff): the trade uses each model's forecast of the
16:00 bar, whose accuracy against the OLS-HAR is in Appendix~\ref{app:rh_closing}.

---

## C7. Closing-interval models: final main-text section or appendix

All three: appendix (status quo, `A_closing.tex`).

**Precedence.** D3 (25 Oct 23:35 PDT, "corralled into that extra section") is followed by the budget (26 Oct 14:48,
online appendix lists "CI-HAR and closing-interval section") and D16 (26 Oct 17:56). The closing-interval models are
scored only at 16:00, so a main-text section would be 16:00 accuracy. The refutation (that D3's "final section"
binds) fails on precedence.

**Final answer.** **T0** (0.93). Action: reword D3's row in `INTEGRATION_PLAN_2026-10-26.md` to "final appendix
section (`A_closing.tex`, `app:rh_closing`)", noting D16 > D3.

---

## Q8. Restart from the fresh version, or keep the current bullets as the first L2 draft?

| Solver | Answer | Tier |
|---|---|---|
| Evidence | Keep current bullets as L2; fill the header-only top beats from the fresh version; fresh = checklist | T2 (0.70) |
| Author record | Keep current bullets as L2; the fresh version is the author's L1 comparison; port what he marks | **T3** (0.60) |
| Referee | Keep current bullets as L2; L1 = fresh claim beats after fixing its six errors | T2 (0.70) |

**Facts verified.** The fresh version (89c359e) breaks later locks:
1. It has separate "Forecast accuracy" and "Beyond HAR, …" sections (fresh l.388, l.454), against "sections 6 and 7
   were never meant to be split" (28 Oct).
2. "a small, dense gain" (l.218) and "spread over the series" (l.267, l.575), against "sparse, not dense" (3c6ccc4).
3. 20 uses of "pooled", against the 28 Oct prune.
4. Prose titles, against "make the section and subsection titles normal" (28 Oct).
5. No Related-work section and no Maisch comparison in the conclusion, against the budget and the 25 Oct quote.
6. Unscoped "no significant accuracy to linear models" (C3).

**Refutation of the majority tier (T2).** T2 needs the author's quotes to imply the answer. They pull both ways:
- **For a restart.** A2 (28 Oct 23:26 PDT): "the paper needs to start from a simple unedited state". The current
  bullets are neither simple (main text about 27 pp) nor unedited (about 970 dated edit tags). Six minutes later the
  author asked "let's do a version where we start over just to compare" (23:32). That is the latest quote. He asked
  for it after the report (3c87fa6, 23:07) had already proposed serving the current top beats, and it reserves the
  comparison for him.
- **For keeping.** A1 (judgments lock and propagate), A4, and the 28 Oct locks embedded in the current text.
- The report's Phase 2 line ("serve the current top beats … as the simple starting state", report l.1306, 1351) is
  agent-written, not an author quote.

No later quote settles A2 against A1. **The refutation succeeds; Q8 is T3** under §8.10's "own locks conflict with no
precedence rule or later quote".

**Recommendation** (0.65), and default if no reply:
- **L1.** The author's first read is a one-page top layer built from the fresh version's claim sentences. Its six lock
  breaks are fixed and C1-C7 applied. This gives him the "start over" version, made lock-compliant.
- **L2.** After he locks L1, the bullet layer is the current bullets reconciled to it. Bullets that contradict L1 are
  parked, not freely rewritten.
- **Alternative.** Regenerate L2 from the locked L1 and re-apply his edits. This costs more agent work, re-derives
  about 970 edits and risks losing some.

**Consequences (where used).** Serving the default needs nothing deleted: the current bullets are not touched until L1
is locked, and the fresh file stays in `writeup/fresh/`. Choosing "regenerate" makes every dated `(user|author, …)`
correction a lock to re-apply, so seeding them into `decisions.yaml` comes first.

---

## Q10. Collaborator authority (A5)

| Topic | Risk solver | Goals solver | Adjudicated |
|---|---|---|---|
| Who locks | Only the author; no named-coauthor locking | Same | Same |
| Held edit | Not applied. Next push **reverts it on Overleaf**. Text parked as a dated `%` comment at the spot; one conflict line to the author | Kept **verbatim in the source** inside a visible `\heldedit{id}{lock}{text}`. Draft PDFs print it; the final build fails while any remain | `\heldedit` |
| Editor | Overleaf commit author, "last editor of the batch" | Same | Same |
| Adoption | One-line chat message, may cover many edits | Adopted by the author's single read of the layer; later meaning changes "adopt"/"drop" | Read adopts, if the pull digest listed the edit before the read |
| Author's Overleaf account | (implicit) | Not a locking channel; stray keystrokes reverted (T0) | Same |
| Tier | T2 (0.80) | T2 (0.80) | T2 |

**Refutation attempts on the held-edit split.**
1. *Risk solver: carrying held text breaks `push --check`'s identity of Overleaf's PDF and the source PDF.* Refuted
   when the marker lives in the source: the flat file is generated from the source, so both builds show it.
2. *Risk solver: the alternative to reverting is blocking pushes.* `\heldedit` does not block pushes.
3. *Against reverting.*
   - A3: the author reads the PDF, and a `%` comment never reaches it.
   - The collaborators are his advisor and senior coauthors (`MEETING_PREP_SEED_2026-08-18.md:9`). Reverting their
     text on Overleaf before the author decides is a unilateral rejection in his name. The co-authoring norm cited by
     the goals solver is "never accept or reject another author's tracked change unilaterally".
   - The draft PDF already has a colored-note convention (`paper_rehaul.tex:292`, the `\editc` legend).
4. *Against `\heldedit`.* It is new markup in `main2.tex`, and a coauthor's next edit across it becomes a structural
   conflict. Held edits are by definition rare (lock conflicts only), so the cost is small.

**Final answer.**
- Keep the default.
- A held prose edit goes into the source as a draft-only `\heldedit` note beside the locked text, worded neutrally
  ("held: conflicts with D1; pending the author's decision"), and is listed in the pull digest. It is resolved by one
  chat line. The C4 gate (no `\rhPending`, `\editc`) extends to `\heldedit`.
- Number and generated-table conflicts keep SPEC §5's behaviour (reported), plus a digest line.
- The editor is recorded as the bridge commit author (X6).
- The author's own Overleaf account is not a locking channel. Header keystrokes are auto-reverted as now.
- A standing rule "adopt whatever X writes" is available if the author ever states it.

**Tier T2.** Overall 0.75; held-edit form 0.65.

**Actions** (sync-code session, after X5a):
- `unflatten.py`: emit `\heldedit` for lock conflicts in prose; write the provenance sidecar (Q11).
- `paper_rehaul.tex` preamble: define `\heldedit` (draft-only; errors in the final layer).
- `check_pending.py` / the future `check_jfec.py`: count `\heldedit`.
- One digest line per pull.
- Optional T0 guard: hold a pulled insertion that restores text the agent's own pushes removed since the last pull
  (a "resurrected suggestion").

---

## Q11. Does A4 cover collaborators' text?

Both solvers: yes for the scrub and Rule NT, for all human text pulled from Overleaf; no C26 byte-lock; the author's
chat edits and the propagation of locked decisions can still change it; gates raise conflicts, never auto-fixes.

**Refutation attempt on the core.** Could the scrub safely run on coauthor text while sparing only the author's own?
No. The bridge names only the last editor of each pull batch (X6), and A4 is binding for the author's own Overleaf
text. The only rule that never breaks A4 protects all human Overleaf text. **T1** (0.85).

**Mechanism split.** Risk solver: `git blame` on `Overleaf pull …` commits. Goals solver: a sidecar (file + sentence
hash → overleaf, editor, date).
- *Against blame.* Any later mechanical touch of the line moves blame to an agent commit and silently drops the
  protection: a macro re-link, a lock-propagation scope edit, a park or a file move (Sections 6 and 7 were merged on
  28 Oct). Lines hold several sentences until the reflow.
- *Against the sidecar.* It must be maintained, but the same tools that touch a protected sentence can re-hash it.
  The report already plans pulled-edit provenance in `decisions.yaml` (§8.5).

**Final answer.** A sidecar written by `unflatten.py` at each pull (file, sentence hash, pull SHA, editor). Tools that
edit a protected sentence update its entry. Blame of pull commits serves as an audit cross-check. No in-text markup.
**T2** (0.65).

---

## Q12. Tracked changes and comments on Overleaf

| Topic | Risk solver | Goals solver | Adjudicated |
|---|---|---|---|
| What git carries | Text only; no comments; pending suggestions exported as accepted (open-source project-history, `filterTrackedDeletes: true`) | Same | Same (strong inference; hosted bridge unverified) |
| Rule | Pull visible text; suggestions arrive as edits under Q10; minimal-diff pushes; fixed file names | Same, plus agents never accept, reject, resolve or change settings without the author's word | Both |
| Comments | Never an input (§8.9 Q1) | Read-only digest of open comments via the author's Chrome session, built at the first coauthor comment | Deferred digest (T2) |
| Track-changes setting | Keep it off for the token account; read-only check; conditional T3 if "for everyone" is on | (not raised) | Read-only check (T0); question only if on |
| Tier | T1, conditional T3 | T2 | T1 rules, T2 digest |

**Refutation attempts.**
1. *Does a comment digest contradict §8.9 Q1 ("comments are still not an input")?* No. Q1 means comments are not an
   edit channel. A digest line becomes an edit only through the author's chat reply. Without it, a senior coauthor
   who comments rather than edits is never heard, since the author reads the PDF (A3).
2. *Is building it now justified?* No coauthor has edited or commented yet. Every Overleaf-side commit (85500a8,
   aba8c24, be31a78, ae201a4) is a stray keystroke in `beats_top.tex` from the author's own account; none touched
   `main2.tex`. So defer.
3. *Does pushing with tracked changes on turn agent pushes into suggestions?* This is inferred from the TPDS upsert
   path, not shown for the bridge. A read-only look at the project's Review panel settles it.

**Final answer.**
- **T1** (0.75):
  - read Overleaf's visible text only; pending suggestions are ordinary collaborator edits under Q10;
  - agents never accept, reject or resolve anything on Overleaf and never change project settings without the
    author's chat instruction (the coordinator's 28 Oct precedent);
  - minimal-diff pushes, fixed flat-file names, the one sentence-per-line reflow before coauthors start.
- **T2** (0.60): a read-only comment digest via the author's Chrome session, built at the first coauthor comment.
- **T0**: one read-only check of the track-changes setting. Optionally, a 5-minute test in a throwaway project (never
  the paper's) of the suggestion-export semantics.
- **Conditional T3**, asked only if the check finds "track changes for everyone" on: "Every agent push would become
  suggestions under your name. Turn it off for the project (affects coauthors; recommended, with a one-line note to
  them), or stop auto-pushing `main2.tex`?"

---

## Q13. Agent edit route

Both solvers: keep git for every agent write; no claudeleaf suggestion route; no bot account. The goals solver's
read-only use of the live editor is the Q12 comment digest, deferred.

**Refutation attempt.** Would suggestion mode serve A1 better through attribution? No:
- it makes every agent edit an accept/reject item, the per-edit review A1 rejected;
- the pull would read pending bot suggestions as accepted text (Q12);
- the author named the route (A3, "you use the overleaf git").

The refutation fails. **T1** (0.85). No action.

---

## Extra findings

### X1 (F1). "Beats augmented HAR models" overclaims for the linear classes. Confirmed, with a qualification

- **Verified.**
  - OLS on HAR + RM + FOMC: -9.2% vs OLS-HAR, DM -5.53 (`rehaul_ablation_numbers.tex:3041-3043`).
  - Against it: ridge full +3.5% (DM 2.99, p 0.003), enet full +2.7% (DM 2.30, p 0.021), lasso full +0.2% (p 0.743)
    (`:3067-3069, 3203-3205, 3135-3137`).
  - The `\rhabOlsAddRmFomc*` and `*FullVsOlsAddRmFomc*` macros appear nowhere in `paper_rehaul.tex` or `rehaul/*.tex`,
    not even in comments. Panel D's OLS column is "--" (`rehaul_ablation_table.tex:41-44`).
- **Qualification.** The abstract's backing comment (`paper_rehaul.tex:267-268`) means HAR + RM and SHAR/HARQ/LHAR,
  and those it does beat (`\rhabNClassesBeatAddRm` = six). The intro scopes contribution 1 to "a richer volatility
  model built from the same one-minute futures returns". Read unscoped, though, the abstract clause clashes with its
  own next clause ("beyond them, only boosted trees gain"). It is also false against the paper's own well-studied base
  for the ridge and elastic net. The clause is agent text from 5554817 (26 Oct), written before the add_rm_fomc arm
  existed; the 28 Oct approval changed only the next clause.
- **Final.** T1 (0.85):
  - drop "and beats augmented HAR models" from the abstract (frees 5 of the 100 words), or scope it to "HAR models
    augmented with the futures' realized moments";
  - drop "beyond richer HAR models" from the intro's contribution-1 title (`01_introduction.tex:100`);
  - add the OLS HAR + RM + FOMC cell or row (C3) and the one disclosure sentence (C3).

### X2 (F4). "The comparison reverses" under MSE. Confirmed

`06_results.tex:146`. MSE is higher in 6 of 6 and significantly in 0 (`rehaul_close_accuracy_numbers.tex:1610-1611`).
The point estimates do flip, so the word is not false, but it is asymmetric next to significant QLIKE gains. Agent text
(5554817). **T0** (a comparison word set by a generated count): "Under squared error the full set's loss is higher in
all six classes, in none significantly". The intro already says "(significantly in 0)".

### X3. The abstract's "significantly in all 6" has no backing table. Partly confirmed

`tab:rh_oos_vsols` is `\input` nowhere (grep of `paper_rehaul.tex`, `rehaul/`, `beats/`). But the text prints each
full set's DM against the OLS (`06_results.tex:127-133`), and `\rhorVsOlsLowerSig` = 6. The backing exists in the
text; no table carries it. Resolved by C2 (T1). The stale `\rhPending{rhorVsOlsLowerSig}` in the backing comment is T0.

### X4. No generated macros for the 28 Oct penalized factor tests or the penalization split. Confirmed

Only the producers (`experiments/factor_pen_2026_10_28.py`, `experiments/beyond_factor_linear_pensplit.py`) reference
`linear_pen`, `factor_pen` or `pen_split`. No `writeup/*.py` and no `writeup/generated/*` does. **T0**. Action under
C3.

### X5. Sync defects in `writeup/overleaf_sync/overleaf_git.example.sh` (the live `overleaf_git.sh` is identical apart from the token line). Confirmed

- **X5a. Deadlock after a real pull.**
  - `pull` runs `unflatten.py`, which writes source files and `REPORT.md` but never the map (writes at
    `unflatten.py:781, 880, 1048`; no map write).
  - `push` then compares Overleaf's `main2.tex` with `map.json`'s `flat_text` and dies while they differ (`:117-119`).
  - `--first` dies too when the file exists (`:120-122`).
  - The documented flow, "pull … Then rebuild the paper, and run `push`" (`:26-27`), cannot complete.
  - **Added:** `push-all` runs `push` then `push-beats` under `set -e` (`:238-241`), so the deadlock also stops the
    review layers, and with them `autopush.sh`'s whole Overleaf update.
  - Fix, T0: at pull time record a watermark (Overleaf commit SHA and the text consumed). `push` accepts Overleaf
    content equal to the last flatten or to the last pulled text. Needed for A5 to work at all; independent of Q10's
    form.
- **X5b. `map.json` is advanced before the push lands.**
  - `flatten.py --map "$MAP"` writes the repo's `map.json` (`:134-135`; `flatten.py:1212-1214`) before `og push`
    (`:151`).
  - `og` retries only on 502/503/504 and connection errors (`:71`), so a non-fast-forward rejection exits under
    `set -e`. The Overleaf clone keeps an unpushed commit, the next `pull --ff-only` fails (`:85`), and `map.json`
    holds a flat text Overleaf never received.
  - **Added:** the map is also written before `--check` runs (`flatten.py:1212` precedes `check` at `:1228`). A failed
    check leaves an uncommitted `main2.tex` in the clone (next run: "uncommitted changes", `:84`) and a wrong
    `map.json` in the repo.
  - Fix, T0: write the map to a temporary path and move it into place only after the push succeeds. On rejection,
    reset the clone to origin and report. An automatic pull-and-retry waits for the X7 wording.
- No collaborator has edited `main2.tex` yet, so neither defect has fired.

### X6. §8.5 says the editor of a pulled change is "unknown". Confirmed wrong

Overleaf's git-integration page: "the author of the most recent change will be listed as the author of the commit";
commits are created when someone pulls. The local clone's Overleaf-side commits carry author `jamesdchen01` and
committer `node`, consistent with that, though one account cannot discriminate. §8.5 (report l.1157) cites the
GitHub-sync bundling rule instead. **T0** correction: "editor = the bridge commit's author, i.e. the last editor of each
pull batch; exact when one person edited between pulls". Whether two editors are split into separate commits is
unverified; one test edit settles it.

### X7. The CLAUDE.md Overleaf rule conflicts with A5/A6. Confirmed

- CLAUDE.md (author, 22 Oct): "If push refuses (Overleaf edits not pulled back), run `pull --dry-run`, report, and
  ask." Under A5 (collaborator edits flow back) and A6 (only T3 reaches the author), every coauthor edit would become
  a question. A5 and A6 are later (28-29 Oct), so precedence favours them in substance.
- Only the author can authorize a CLAUDE.md change, so it goes to him as a wording change, not a question.
- Precondition: X5a fixed, else a pull still cannot be followed by a push.
- Until he agrees, the current rule stands.

### X8. Strategies on the HAR + RM + FOMC forecasts. Confirmed open

- Proposed 27 Oct 19:02Z ("no strategy is built on the HAR + RM + FOMC forecasts … a local computation, no cluster")
  and listed at 19:08Z as "If decision 1 is yes, compute the trading strategies on the HAR + RM + FOMC forecasts".
- Decision 1 was never answered. No econ generator reads `add_rm_fomc`, and no result folder holds base-forecast
  strategies.
- Weighed:
  - for: the author's 27 Oct 19:02Z "The abstract wording should be neither. Both miss the point of the bigger story",
    said of the economics sentence; and a referee will ask whether the 36 series add trading value;
  - against: "there needs to not just be a rampage of things we explored" (28 Oct 05:51Z), and "Don't launch anything
    new without asking" (28 Oct 00:48Z, said of the clusters).
- **T2** (0.60):
  - a local computation from existing 16:00 forecasts, run only after the digest (veto line);
  - a rule file is committed first, labelled "proposed after the author's 27 Oct remark; not pre-registered";
  - results go to the economics appendix;
  - the abstract's trading sentence stays on the author's pending check (`99_author_notes.tex:136-138`).

---

## Records for `decisions.yaml` (source `inferred`; the file does not exist yet, Phase 2)

| ID | Decision | Tier | Evidence |
|---|---|---|---|
| I-C1 | Two-level research question; result order: gain vs OLS-HAR plus MSE, base share, beyond base; "alternative data" stays narrow | T2 | D18; 3c6ccc4; transcript 28 Oct 22:09-23:15Z |
| I-C2 | Every Table 1 row tested vs OLS-HAR on every bar, plus within-class DM; 16:00 to `app:rh_closing` | T1 | D2; D16 |
| I-C3 | Beyond-base linear claims are within-class, scoped to every bar and ordinary days; OLS-on-base disclosed | T1 | a38439f; `pen_split.csv`; `:3155-3157` |
| I-C4 | Ladder figure appendix only; every-bar after addendum K; 16:00 panel to A_closing | T0 / T2 | D16 (7f7894a), D20 |
| I-C5 | D18 SHAP grouping (post hoc) plus reconciliation table | T1 / T2 | `RULE_shap.txt:150`; 28 Oct 17:50 PDT |
| I-C6 | Application as is plus a no-number pointer | T1 | D16 option (a) |
| I-C7 | Closing-interval models appendix only | T0 | D16 > D3 |
| I-Q10 | Author is the only lock authority; held edits shown as `\heldedit` | T2 | A1-A5; co-author norm |
| I-Q11 | Overleaf-pulled human text exempt from scrub and Rule NT; sidecar provenance | T1 / T2 | A4; X6 |
| I-Q12 | Visible text only; no agent action on review state or settings; comment digest deferred | T1 / T2 | git-bridge code; 28 Oct precedent |
| I-Q13 | Git route only | T1 | A1, A3 |
| I-X1 | Abstract and intro scope fix of "augmented HAR" | T1 | `:3041-3205` |
| I-X8 | Base-forecast strategies, appendix, after the veto window | T2 | 27 Oct 19:02Z |

## Consolidated actions by file (T0-T2; none applied by this adjudication)

| File | Change | Item |
|---|---|---|
| `paper_rehaul.tex` | Abstract: drop or scope "and beats augmented HAR models"; refresh the backing comment (l.267); define `\heldedit` | X1, X3, Q10 |
| `rehaul/01_introduction.tex` | Two-level research question; three contributions; drop "beyond richer HAR models"; "on every bar" scope | C1, X1, C3 |
| `rehaul/06_results.tex` | Cite Table 1's vs-OLS columns; MSE "in none significantly" | C2, X2 |
| `rehaul/08_feature_importance.tex` | OLS-on-base sentence; penalized test A ("little"); D18 SHAP clause plus reconciliation table | C3, C5 |
| `rehaul/07_economics.tex` | One pointer sentence to `app:rh_closing` | C6 |
| `rehaul/09_conclusion.tex` | Order as C1; "on every bar" scope | C1, C3 |
| `rehaul/05_methods.tex`, `references.bib` | Giacomini-White sentence (BibTeX fetched) | C2 |
| `rehaul/A_closing.tex` | 16:00 accuracy table vs OLS; later the 16:00 ladder panel | C2, C4 |
| `make_rehaul_oos_rerun.py` | Two Table 1 columns, sign key, note; 16:00 columns out of main-text floats | C2 |
| `make_rehaul_ablations.py` | OLS HAR + RM + FOMC cell or row | C3, X1 |
| new `make_rehaul_linear_pen.py` | Macros and table for tests A/B and the split | C3, X4 |
| `make_rehaul_fi_pooled.py`, new `make_rehaul_fi_reconcile.py` | D18 panel and macros; reconciliation table | C5 |
| `make_rehaul_ladder.py` | Every-bar figure after addendum K; separate 16:00 figure | C4 |
| `overleaf_sync/overleaf_git.example.sh` (and live copy) | Pull watermark; map written after the push lands; reset on rejection | X5 |
| `overleaf_sync/unflatten.py` | `\heldedit` for prose lock conflicts; provenance sidecar | Q10, Q11 |
| `INTEGRATION_PLAN_2026-10-26.md` | D3 row reworded; budget ladder row marked superseded | C7, C4 |
| `reports/AI layered paper writing system.md` | §8.5 editor attribution; §8.9 Q8 and Q10-Q13 tiers and status | X6 |
