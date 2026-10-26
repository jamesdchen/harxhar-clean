# Integration and pruning pass: shared spec for the section agents (2026-10-26)

Read first: `writeup/INTEGRATION_PLAN_2026-10-26.md` (author decisions D1-D15 and the JFEc page budget) and the audit
findings in `C:\Users\james\AppData\Local\Temp\claude\C--Users-james-CC-Allowed-harxhar-0dte-professor\bd4224e0-12aa-4614-b6a6-3fe50535ade1\scratchpad\audit_2026-10-25.md`
(89 confirmed findings; fix the ones in your files).

## The story (author-approved framing)

The paper belongs to the realized-volatility literature. Question: what do exogenous predictors (constituents' trading
measures, the FOMC calendar, the futures' own realized moments, implied volatility, and alternative data: StockTwits,
volatility demand) add to pooled intraday realized-variance forecasts of the E-mini S&P 500, on every regular-hours
bar? Design: one model per class, estimated on the 13 regular-hours bars (I_rh, bar ends 10:00-16:00 ET), predictors
(HAR ladder and trailing averages) computed on the full 48-bar grid incl. overnight; W = 3,000 days; frozen
hyperparameters. Primary benchmark: OLS on the HAR set, pooled on RTH13 (Table 1's OLS row). Findings to carry:
- Every bar: the full set beats HAR significantly in all six classes (linear about -6 to -9%, trees about -14%, patched
  design), and beats HAR + the futures' realized moments and the classical SHAR/HARQ/LHAR extensions.
- The gain is concentrated, not dense and weak: the FOMC calendar is about two thirds of the trees' gain, the futures'
  realized moments second; ~10 series carry ~95% of the trees' gain (feature-count ladder); linear models do best with
  5-10 series.
- Alternative data add little even when live (ablations on live windows); their pooled SHAP share understates use;
  use is not accuracy.
- Under MSE the full set is not more accurate (a handful of March-2020 crash days); QLIKE is the primary loss.
- Closing half hour: no significant full-set gain over HAR + realized moments; closing-interval models (CI-HAR rebuilt
  under the current data chain, the three-way CLOSE arm) live ONLY in the final appendix section.
- Application (short): one 0DTE ATM straddle a day signed by the closing-bar forecast vs implied variance. Full-set
  strategies beat passive selling; their gain over the HAR strategy is not significant; profits come from long days and
  rest on about ten days; the strategy has no market beta (beta appendix). The straddle pays on the index move, not
  on realized variance; no "pinning" claim.

## Page budget (JFEc): main text about 22 pages, abstract <= 100 words. See the plan's budget table.

## File ownership (no two agents edit the same file)

| Agent | Owns (edit only these) | Creates (new appendix file for moved-out material) |
|---|---|---|
| FRONT | writeup/paper_rehaul.tex, rehaul/01_introduction.tex, rehaul/09_conclusion.tex, rehaul/99_author_notes.tex | -- |
| RELATED | rehaul/02_related_work.tex | -- |
| DATA | rehaul/03_data.tex, rehaul/03_data_floats.tex, rehaul/04_features.tex, rehaul/04_features_floats.tex | rehaul/A_data_predictors.tex |
| METHODS | rehaul/05_methods.tex, rehaul/05_methods_floats.tex | rehaul/A_methods.tex |
| RESULTS | rehaul/06_results.tex, rehaul/06_results_floats.tex | rehaul/A_results.tex |
| VI | rehaul/08_feature_importance.tex, rehaul/08_feature_importance_floats.tex, rehaul/08b_fi_measures_floats.tex | rehaul/A_fi.tex |
| ECON | rehaul/07_economics.tex, rehaul/07_economics_floats.tex | rehaul/A_econ.tex |
| APPENDIX | rehaul/10_appendix.tex, rehaul/10_appendix_floats.tex, rehaul/10_statistical_tests.tex | rehaul/A_closing.tex, rehaul/A_robustness.tex |
| NUMBERS | generators (writeup/make_*.py, experiments/master_table_close.py, writeup/final_run.py), writeup/generated/*, writeup/references.bib, writeup/scholar_bib/* | -- |

Section order after the pass (FRONT wires it in paper_rehaul.tex): 1 Introduction, 2 Related work, 3 Data, 4 Predictors,
5 Methods, 6 Results (accuracy + ablations), 7 Variable importance (promoted from 6.5 to its own section; VI keeps the
label sec:rh_fi), 8 Economic application, 9 Conclusion; references; online appendix (APPENDIX wires A_*.tex files in).

## Conventions

- House rules: every number in text from a generated macro (no hard-coded numbers); replaced or removed text is PARKED
  in dated `%` comments (`% 2026-10-26 (integration, JFEc): parked -- ... Was: "..."`), never deleted; generated files
  keep the AUTO-GENERATED header and are written only by generators.
- Moving material to the appendix: cut it from your section (park a one-line pointer comment), paste it, lightly
  edited, into your `rehaul/A_*.tex` file under a `\subsection{...}` with a `\label{app:...}`, and leave a short
  cross-reference in the main text ("Appendix~\ref{app:...}"). Floats moved with their text go into the same A_ file.
- Resolve visible editorial notes (`\editc{...}`) in your files: park resolved ones as comments; an open one goes to
  rehaul/99_author_notes.tex only via FRONT (tell FRONT in your report; do not edit that file yourself).
- Drafts to integrate (writeup/drafts/): 2026-10-25_econ_main, 2026-10-25_econ_decomp (ECON); 2026-10-25_close_accuracy
  (RESULTS: the MSE/QLIKE and level-bias points; APPENDIX: the closing-bar tables); 2026-10-25_mcs (RESULTS one
  sentence; APPENDIX the tables); 2026-10-25_ablations (RESULTS); 2026-10-26_complexity (VI main text one sentence +
  APPENDIX/A_fi the scaling grid, framing B only: data and feature scaling, NOT the complexity/KMZ argument);
  2026-10-26_econ_beta (ECON one bullet; A_econ the subsection); 2026-10-26_train_arms (APPENDIX).
  Generated macro files these drafts need must be \input in the preamble: tell FRONT the exact file names.
- The numbers will be regenerated on the FOMC-patched design later; write direction words ("lowers", "significant")
  carefully and prefer macro-driven counts where a direction could change.
- Do NOT compile in writeup/ (shared aux files); if you must check LaTeX, copy writeup/ to your own scratch dir.
  Laptop memory is low: no heavy jobs. Do NOT git commit, push, or touch Overleaf; the coordinator commits.
- Report back: what moved where (with labels), pages you estimate for your part, any macro/label the others need,
  open issues.
