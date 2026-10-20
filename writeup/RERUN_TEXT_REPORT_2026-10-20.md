# Prose numbers on the pooled linear reruns (2026-10-20)

Branch `insurance-selling-2026-09-19-prose-numbers`. Nothing in `writeup/rehaul/*.tex`, `writeup/paper_rehaul.tex`
or an existing generated file was edited.

## What this is

The abstract, introduction, results and conclusion quote macros of `writeup/make_rehaul_results.py`
(`writeup/generated/rehaul_results_numbers.tex`). They were computed from the old pooled forecasts: the six classes
(three linear, three tree ensembles), scored on the closing half hour only, against the closing-interval OLS. This
report takes every live sentence that quotes such a macro, or states such a claim without one, and gives its
like-for-like counterpart on the two pooled linear reruns. The counterparts come from
`writeup/make_rehaul_rerun_text_numbers.py`. It writes `writeup/generated/rehaul_rerun_text_numbers.tex` (217
`\rhrr...` macros, not loaded by the preamble) and `results/rerun_text_numbers_2026-10-20/{numbers,dm}.csv`.

**Scoring.** The scoring is Table 1's: the generator imports `writeup/make_rehaul_oos_rerun.py`, so the scored
intervals, the truth (`core_stats` sumret2), the losses and the daily averages are the same.
- Days: the 1,082 trading days, 3 January 2020 to 30 April 2024. Early closes are dropped, and so are intervals with a
  non-positive RV.
- Tests: Diebold-Mariano (`dm_test`, its default lag, here 6) on the daily average loss. "Significant" means
  p < 0.05.

**Framings and panels.**

| | Panel A: trained on all 48 intervals | Panel B: trained on the 13 regular-hours bars |
|---|---|---|
| **(a) every interval** of the panel's sample | 50,386 intervals | 14,066 intervals |
| **(b) the closing half hour only** (16:00 row = the 15:30 forecast) | 1,082 intervals | 1,082 intervals |

Under (b) a day holds one interval, so the daily average loss is that interval's loss.

**Macro names.** `\rhrr<Name><Framing><Panel>`, with Framing = `All` | `Close` and Panel = `A` | `B`. Examples:
`\rhrrAfLowerAllA`, `\rhrrAfLowerSigCloseB`. LaTeX macro names cannot contain digits, so the brief's `...A48` and
`...Close13` would not compile. I used Table 1's panel letters (`\rhorNIntA`, `\rhorNIntB`) instead.

**Classes.** These are the linear classes only: ridge, lasso and elastic net, each on the HAR benchmark and on all
features, plus the OLS on the HAR benchmark. That makes 7 forecasts and 3 comparisons of all features against the HAR
benchmark (the old text has 12 and 6). The tree ensembles are held off.

**"The OLS".** The old counts (`\rhrNLowerPool`, `\rhrVsBmPool...`, `\rhrOlsQ`) compare against the
*closing-interval* OLS. Its forecasts are in neither rerun nor in the repo
(`results/linear_subsection_dedup/...` is absent). The counterpart computed here is against the *pooled* OLS on the
HAR benchmark of the same panel. That is a different benchmark, so each such row says so.

**Not computable here.** Sharpe ratios and the rank correlations with Sharpe ratios (`\rhrAfRfPoolDSh`,
`\rheRankCorrQPool`, `\rheRankCorrESPool`, the "Sharpe higher in every class" claims) need the option data and the
economics pipeline, and neither is in the repo. Those rows are listed with "--".

## Headline counts (all features against the HAR benchmark, same model; 3 linear classes)

| Count | Old macro (value) | (a) A | (a) B | (b) A | (b) B |
|---|---|---|---|---|---|
| classes compared | `\rhrAfPoolPairs` (6) | 3 | 3 | 3 | 3 |
| QLIKE lower with all features | `\rhrAfPoolLower` (6) | 3 | 3 | **0** | 1 |
| ... and significant at 5% | `\rhrAfPoolLowerSig` (4) | **0** | **3** | 0 | 0 |
| QLIKE higher and significant | `\rhrAfRevSig` (0) | 0 | 0 | 0 | 0 |
| MSE higher with all features | `\rhrAfPoolMseHigher` (3) | 3 | 3 | 3 | 3 |
| ... and significant | `\rhrAfPoolMseHigherSig` (0) | 0 | 0 | 0 | 0 |
| QLIKE % change, range | -- | -1.0 to -0.1 | -3.5 to -3.1 | +1.6 to +2.0 | -0.4 to +6.7 |

### Each class: QLIKE % change, all features vs HAR (DM, p)

| Class | Old (`\rhrAf<M>Pool{Pct,Dm,Pval}`) | (a) A | (a) B | (b) A | (b) B |
|---|---|---|---|---|---|
| Ridge | -5.0 (-1.77, 0.077) | -0.1 (-0.19, 0.847) | -3.4 (-2.35, 0.019) | +2.0 (+0.67, 0.503) | +6.7 (+1.49, 0.137) |
| Lasso | -3.0 (-1.13, 0.26) | -1.0 (-1.35, 0.178) | -3.1 (-2.54, 0.011) | +1.8 (+0.68, 0.495) | +5.2 (+1.38, 0.168) |
| Elastic net | -5.6 (-2.10, 0.036) | -0.8 (-1.00, 0.317) | -3.5 (-4.43, < 0.001) | +1.6 (+0.56, 0.573) | -0.4 (-0.18, 0.857) |
| LightGBM / XGBoost / RF | -13.0 / -12.0 / -11.9 (all p < 0.001) | held off | held off | held off | held off |

### Mean QLIKE

| Forecast | Old | (a) A | (a) B | (b) A | (b) B |
|---|---|---|---|---|---|
| OLS, HAR | `\rhrOlsQ` 0.1033 (closing-interval OLS) | 0.2360 | 0.1926 | 0.1116 | 0.1152 |
| Ridge HAR / Full | 0.1143 / 0.1086 | 0.2359 / 0.2357 | 0.1926 / 0.1860 | 0.1114 / 0.1137 | 0.1152 / 0.1230 |
| Lasso HAR / Full | 0.1135 / 0.1101 | 0.2359 / 0.2336 | 0.1926 / 0.1866 | 0.1114 / 0.1133 | 0.1152 / 0.1212 |
| Elastic net HAR / Full | 0.1142 / 0.1078 | 0.2356 / 0.2338 | 0.1926 / 0.1859 | 0.1108 / 0.1125 | 0.1152 / 0.1148 |

(a) and (b) average QLIKE over different intervals, so their levels are not comparable with each other. Within a
column they are.

### Against the pooled OLS on the HAR benchmark (6 penalized forecasts)

| Count | Old macro (vs closing-interval OLS, 12 forecasts) | (a) A | (a) B | (b) A | (b) B |
|---|---|---|---|---|---|
| lower QLIKE than the OLS | `\rhrNLowerPool` 0 | 6 | 3 | 3 | 4 |
| ... and significant | `\rhrNLowerSigPool` 0 | 3 (the HAR ones) | 3 (the Full ones) | 3 (the HAR ones) | 3 (the HAR ones) |
| higher and significant | `\rhrNWorseSigPool` 9 | 0 | 3 (the HAR ones) | 0 | 0 |
| Full vs OLS, QLIKE % range | `\rhrVsBmPoolLinFullPct` +4.4 to +6.7 | -1.0 to -0.1 | -3.5 to -3.1 | +0.8 to +1.9 | -0.4 to +6.7 |
| HAR vs OLS, QLIKE % range | `\rhrVsBmPoolHarPct` +10.0 to +26.0 | -0.2 to 0.0 | 0.0 to 0.0 | -0.7 to -0.1 | 0.0 to 0.0 |

**Caution, Panel B.** With their frozen penalties, the penalized HAR-benchmark models of Panel B are numerically
almost the OLS: QLIKE equal to 4 decimals, and MSE equal to within about 1e-15. Their DM tests against the OLS are
still significant (|DM| 2.9 to 4.9), because the daily loss differences are tiny but very regular. These "significant"
differences are of order 1e-6 in QLIKE. They should not be read as a result.

## Inventory and suggested rewording

24 live sentences or notes. Rows 1-12 quote a macro computed from pooled forecasts. Rows 13-24 make the same kind of
claim with no macro, or sit in an author or draft note.

Abbreviations:
- "Old": the current value in `rehaul_results_numbers.tex`.
- "T": the wording stays true. "F": it is false. "R": it needs rewording (a number or a scope changes, but the
  direction holds).
- The order of the true/false column is (a) A, (a) B, (b) A, (b) B.

Suggested rewordings use the new macros. They state only what the numbers show, with no model ranking. Where the
framings differ, there are two variants.

| # | Sentence (file:line) | Macros: old value | (a) A | (a) B | (b) A | (b) B | True? aA/aB/bA/bB | Neutral rewording |
|---|---|---|---|---|---|---|---|---|
| 1 | Abstract: "Under the QLIKE loss, the full predictor set yields lower average out-of-sample loss than the HAR benchmark across all \rhrAfPoolPairs{} intraday model classes, with \rhrAfPoolLowerSig{} of the \rhrAfPoolPairs{} reductions statistically significant at the 5% level." (`paper_rehaul.tex:118-124`) | AfPoolPairs 6, AfPoolLowerSig 4 | 3 of 3 lower, 0 sig | 3 of 3 lower, 3 sig | 0 of 3 lower | 1 of 3 lower, 0 sig | R / R / F / F | (a): "Under QLIKE, the full predictor set has a lower average loss than the HAR benchmark in all \rhrrAfPairsAllB{} linear model classes, with \rhrrAfLowerSigAllB{} of the reductions significant at the 5% level." (Panel A: "... none significant at the 5% level"). (b): "Under QLIKE, at the closing half hour, the full predictor set does not lower the average loss of the linear models significantly: the change ranges from \rhrrAfPctMinCloseA{} to \rhrrAfPctMaxCloseA\%, none significant at the 5% level." |
| 2 | Abstract E34 note: "... the counts of intraday model classes in this abstract and in the introduction (\rhrAfPoolPairs{} classes, \rhrAfPoolLowerSig{} significant) ..." (`paper_rehaul.tex:140`) | 6, 4 | 3, 0 | 3, 3 | 3, 0 (and 0 lower) | 3, 0 (1 lower) | R / R / R / R | A draft note: replace with "(linear classes only: \rhrrAfPairsAllA{} classes; see rehaul_rerun_text_numbers.tex for the counts under each framing)" once the framing is chosen. |
| 3 | Intro: "Incorporating the full predictor set reduces the average out-of-sample QLIKE loss relative to the HAR benchmark across every intraday model class (\rhrAfPoolPairs{} classes in total, with \rhrAfPoolLowerSig{} of these reductions statistically significant at the 5% level; ...)" (`01_introduction.tex:64-66`) | 6, 4 | 3/3, 0 sig | 3/3, 3 sig | 0/3 | 1/3, 0 sig | R / R / F / F | (a): "The full predictor set lowers the average QLIKE loss relative to the HAR benchmark in each of the \rhrrAfPairsAllB{} linear model classes, \rhrrAfLowerSigAllB{} of them significantly at the 5% level." (b): "At the closing half hour, the full predictor set changes the average QLIKE loss of the linear models by \rhrrAfPctMinCloseA{} to \rhrrAfPctMaxCloseA\% relative to the HAR benchmark, none significantly at the 5% level." |
| 4 | Results bullet: "Augmenting the HAR benchmark with the full predictor set reduces the average out-of-sample QLIKE loss across every intraday model class ($\Delta\% < 0$ in all \rhrAfPoolPairs{} classes), with \rhrAfPoolLowerSig{} of these reductions statistically significant at the 5% level" (`06_results.tex:310-313`) | 6, 4 | 3, 0 | 3, 3 | 0, 0 | 1, 0 | R / R / F / F | As row 3. The bold heading "Consistent QLIKE reductions across all model classes" holds only under (a). For (b), a neutral heading: "QLIKE: full predictor set against the HAR benchmark". |
| 5 | Results: "Regarding the tree ensembles, we observe the largest relative improvements: $\Delta\% = \rhrAfLgbmPoolPct$ for LightGBM ($\DM = \rhrAfLgbmPoolDm$), $\rhrAfXgbPoolPct$ for XGBoost ..., and $\rhrAfRfPoolPct$ for the random forest ..., each of which is statistically significant at the 5% level." (`06_results.tex:315-318`) | -13.0 (-4.39), -12.0 (-4.29), -11.9 (-4.75) | -- | -- | -- | -- | n/a (trees held off) | Park until the trees are refit. |
| 6 | Results: "Regarding the penalized linear models, the elastic net achieves a statistically significant reduction of $\Delta\% = \rhrAfEnetPoolPct$ ($\DM = \rhrAfEnetPoolDm$), while ridge regression and the lasso yield reductions of $\rhrAfRidgePoolPct$ (...) and $\rhrAfLassoPoolPct$ (...), respectively." (`06_results.tex:319-322`) | Enet -5.6 (-2.10, p 0.036); Ridge -5.0 (-1.77, p 0.077); Lasso -3.0 (-1.13, p 0.26) | Enet -0.8 (-1.00, .317); Ridge -0.1 (-0.19, .847); Lasso -1.0 (-1.35, .178) | Enet -3.5 (-4.43, <.001); Ridge -3.4 (-2.35, .019); Lasso -3.1 (-2.54, .011) | Enet +1.6; Ridge +2.0; Lasso +1.8 (all p > 0.49) | Enet -0.4 (.857); Ridge +6.7 (.137); Lasso +5.2 (.168) | F / R / F / F | Generic: "The full predictor set changes the average QLIKE loss by $\rhrrAfEnet<F><P>Pct$\% for the elastic net ($\DM = \rhrrAfEnet<F><P>Dm$, \rhrrAfEnet<F><P>Pval), $\rhrrAfRidge<F><P>Pct$\% for ridge (...) and $\rhrrAfLasso<F><P>Pct$\% for the lasso (...)." Under (a) B, add: "each significant at the 5% level". |
| 7 | Results: "Under the full predictor set, the linear models attain average QLIKE losses between \rhrPoolQLinFullMin{} and \rhrPoolQLinFullMax{}—specifically, \rhrQEnetPoolFull{} for the elastic net, \rhrQRidgePoolFull{} for ridge, and \rhrQLassoPoolFull{} for the lasso, representing the three most accurate specifications among all \rhrPoolN{} pooled intraday forecasts—compared with \rhrPoolQTreeFullMin--\rhrPoolQTreeFullMax{} for the tree ensembles." (`06_results.tex:469-473`) | 0.1078-0.1101 (Enet .1078, Ridge .1086, Lasso .1101); PoolN 12; trees .1126-.1146 | .2336-.2357 (Enet .2338, Ridge .2357, Lasso .2336); 7 forecasts | .1859-.1866 (.1859, .1860, .1866) | .1125-.1137 (.1125, .1137, .1133) | .1148-.1230 (.1148, .1230, .1212) | R / R / F / R | Drop the ranking clause ("three most accurate") and the tree comparison (trees held off). Under (b) A, all three Full forecasts have a higher QLIKE than the HAR ones, so the "most accurate" clause is false. Rewording: "On the full predictor set, the average QLIKE loss of the linear models ranges from \rhrrPoolQLinFullMin<F><P>{} to \rhrrPoolQLinFullMax<F><P>{} (elastic net \rhrrQEnetFull<F><P>, ridge \rhrrQRidgeFull<F><P>, lasso \rhrrQLassoFull<F><P>)." |
| 8 | Results: "Under the HAR benchmark, the linear specifications achieve average QLIKE losses of \rhrPoolQLinHarMin--\rhrPoolQLinHarMax, outperforming the tree ensembles at \rhrPoolQTreeHarMin--\rhrPoolQTreeHarMax." (`06_results.tex:474-477`) | .1135-.1143; trees .1279-.1301 | .2356-.2359 | .1926-.1926 | .1108-.1114 | .1152-.1152 | R / R / R / R | "On the HAR benchmark, the average QLIKE loss of the linear models ranges from \rhrrPoolQLinHarMin<F><P>{} to \rhrrPoolQLinHarMax<F><P>{} (pooled OLS: \rhrrOlsQ<F><P>)." Drop the tree clause. |
| 9 | Results: "Every pooled intraday specification incurs a higher average QLIKE loss than the closing-interval HAR benchmark ($\overline{\QLIKE} = \rhrOlsQ$)" (`06_results.tex:480-482`) | OlsQ 0.1033 (closing-interval OLS) | -- | -- | -- | -- | not computable (closing-interval OLS not in the reruns) | Under (a) this comparison has no meaning: the closing-interval OLS forecasts one interval. Under (b) it needs the closing-interval OLS refit on the rerun design. Counterpart against the pooled OLS (rows 10-11) is below. |
| 10 | Results: "Under the full predictor set, the average QLIKE loss of the pooled linear models exceeds that of the closing-interval HAR benchmark by $+\rhrVsBmPoolLinFullPctMin$ to $+\rhrVsBmPoolLinFullPctMax\%$, although none of these differences is statistically significant at the 5% level; for the pooled tree ensembles, the excess loss ranges from $+\rhrVsBmPoolTreeFullPctMin$ to $+\rhrVsBmPoolTreeFullPctMax\%$, each significant at the 5% level." (`06_results.tex:484-489`) | +4.4 to +6.7 (linear, n.s.); +9.0 to +11.0 (trees, sig) | vs pooled OLS: -1.0 to -0.1, 0 sig | -3.5 to -3.1, 3 sig | +0.8 to +1.9, 0 sig | -0.4 to +6.7, 0 sig | different benchmark: F / F / R / R | If the pooled OLS becomes the reference: "Against the pooled OLS on the HAR benchmark, the full-predictor-set linear models change the average QLIKE loss by \rhrrVsOlsFullPctMin<F><P>{} to \rhrrVsOlsFullPctMax<F><P>\%; \rhrrNLowerSigFull<F><P>{} of the \rhrrAfPairs<F><P>{} reductions are significant at the 5% level and \rhrrNWorseSigFull<F><P>{} of the increases." Drop the tree clause. |
| 11 | Results: "Under the HAR benchmark, the excess QLIKE loss of the pooled intraday models ranges from $+\rhrVsBmPoolHarPctMin$ to $+\rhrVsBmPoolHarPctMax\%$, with every difference statistically significant at the 1% level." (`06_results.tex:490-492`) | +10.0 to +26.0, all sig at 1% | vs pooled OLS: -0.2 to 0.0; 3 lower, sig | 0.0 to 0.0; 3 higher, "sig" (see caution) | -0.7 to -0.1; 3 lower, sig | 0.0 to 0.0; 3 lower, sig (see caution) | different benchmark: F / F / F / F | "On the HAR benchmark, the penalized linear models change the average QLIKE loss by \rhrrVsOlsHarPctMin<F><P>{} to \rhrrVsOlsHarPctMax<F><P>\% relative to the pooled OLS." In Panel B, state that the penalized and OLS forecasts are numerically almost identical, rather than citing the DM tests. |
| 12 | Author notes: "\rhrNLowerPool{} of the \rhrPoolN{} pooled intraday forecasts have lower QLIKE than the daily HAR benchmark, \rhrNLowerSigPool{} significantly (Diebold--Mariano, 5% level); closing-interval tests are left out of this count" (`99_author_notes.tex:14`) | 0 of 12, 0 sig (vs closing-interval OLS) | vs pooled OLS: 6 of 6, 3 sig | 3 of 6, 3 sig (and 3 sig higher) | 3 of 6, 3 sig | 4 of 6, 3 sig | different benchmark: F / F / F / F | "\rhrrNLower<F><P>{} of the \rhrrNSet<F><P>{} penalized pooled forecasts have a lower QLIKE than the pooled OLS on the HAR benchmark, \rhrrNLowerSig<F><P>{} significantly; \rhrrNWorseSig<F><P>{} are significantly higher." |
| 13 | Abstract: "Furthermore, although every intraday specification remains less accurate under QLIKE than a HAR-type model ..., estimated by OLS on the closing-interval observations ..., incorporating the exogenous predictors narrows this gap across all model classes." (`paper_rehaul.tex:124-128`) | no macro (\rhrVsBmPool...) | -- | -- | -- | -- | not computable against the closing-interval OLS | Against the pooled OLS, "narrows the gap" is the same statement as "QLIKE lower with all features" (row 1): T under (a), F under (b) A (0 of 3), F under (b) B (1 of 3). Suggest dropping the clause until the closing-interval OLS is rerun. |
| 14 | Abstract: "... a sign-based trading strategy achieves a higher Sharpe ratio under the full predictor set than under the HAR benchmark across every intraday model class." (`paper_rehaul.tex:128-130`) | implicit \rhrAfPoolShHigher 5 of 6 | -- | -- | -- | -- | not computable here | Needs the economics rerun on the rerun forecasts. |
| 15 | Intro: "The tree-based ensembles exhibit the largest relative improvements, while among the linear specifications, the elastic net achieves a statistically significant reduction." (`01_introduction.tex:66-68`) | from \rhrAfEnetPoolPval 0.036 | Enet p 0.317 | all three p < 0.05 | Enet +1.6 (higher) | Enet -0.4, p 0.857 | F / R / F / F | Drop the tree clause. (a) B: "... and the reduction is significant at the 5% level in all three linear classes." Otherwise drop the sentence. |
| 16 | Intro: "Every pooled intraday forecast incurs a higher average QLIKE loss than the closing-interval HAR benchmark. While this performance gap is statistically insignificant at the 5% level for the linear models, it is significant for the tree ensembles. Notably, augmenting the intraday models with the full predictor set narrows this gap across all model classes" (`01_introduction.tex:75-78`) | no macro (\rhrVsBmPool...) | -- | -- | -- | -- | not computable (as rows 9, 13) | As row 13. |
| 17 | Intro: "Using the full predictor set increases the annualized Sharpe ratio of this strategy relative to the HAR benchmark in every intraday model class" (`01_introduction.tex:84-86`) | implicit | -- | -- | -- | -- | not computable here | As row 14. |
| 18 | Intro note N15c: "... pooled random forest $\Delta\SR = \rhrAfRfPoolDSh$ ..." (`01_introduction.tex:87`) | -0.19 | -- | -- | -- | -- | n/a (trees held off; Sharpe not computable) | Keep as a note until the trees and the economics are rerun. |
| 19 | Results: "Comparing across model classes in Table~\ref{tab:rh_oos}, the penalized linear models consistently achieve lower average QLIKE loss than the tree ensembles on both predictor sets" (`06_results.tex:465-467`) | no macro (\rhrPoolQ...) | -- | -- | -- | -- | n/a (trees held off; also a ranking claim) | Drop. |
| 20 | Results: "Consequently, incorporating the exogenous covariates narrows the accuracy gap relative to the closing-interval HAR benchmark across every intraday model class." (`06_results.tex:493-495`) | no macro | -- | -- | -- | -- | not computable (as row 13) | As row 13. |
| 21 | Conclusion: "Relative to the HAR benchmark, augmenting the model with the full exogenous predictor set lowers mean QLIKE across all \rhrAfPoolPairs{} pooled intraday model classes, with \rhrAfPoolLowerSig{} of these reductions statistically significant at the 5% level (...). The largest relative reductions occur in the tree ensembles, while among the penalized linear specifications the elastic net achieves a statistically significant reduction." (`09_conclusion.tex:19`) | 6, 4 | 3, 0 | 3, 3 | 0, 0 | 1, 0 | R / R / F / F | As rows 3 and 15. Its bold heading "Exogenous predictors reduce QLIKE loss in pooled intraday models" holds under (a) only. |
| 22 | Conclusion: "Every pooled intraday specification incurs a higher mean QLIKE than the closing-interval HAR benchmark, ... Incorporating the full exogenous predictor set narrows this loss gap in every model class, leaving the gap statistically insignificant at the 5% level for all three penalized linear models while remaining significant for the tree ensembles" (`09_conclusion.tex:23`) | no macro (\rhrVsBmPool...) | -- | -- | -- | -- | not computable (as row 13) | As row 13. |
| 23 | Conclusion: "... adding the exogenous predictors increases the sign strategy's post-warmup Sharpe ratio in every intraday model class. Across the \rhrPoolN{} pooled intraday forecasts, average QLIKE loss and $\overline{\mathrm{ES}}_1$ ... are both negatively rank-correlated with straddle Sharpe ratios, at \rheRankCorrQPool{} and a weak \rheRankCorrESPool{} ..." (`09_conclusion.tex:30`) | PoolN 12; -0.52; -0.10 | PoolN 7; correlations -- | 7; -- | 7; -- | 7; -- | not computable here (Sharpe) | Needs the economics rerun. With 7 forecasts, a rank correlation rests on very few points. |
| 24 | Conclusion note N15c: "... pooled random forest $\Delta\SR = \rhrAfRfPoolDSh$ ..." (`09_conclusion.tex:31`) | -0.19 | -- | -- | -- | -- | n/a | As row 18. |

`<F><P>` in a rewording stands for the chosen framing and panel, e.g. `AllA`, `AllB`, `CloseA` or `CloseB`.

## What the numbers say, without a framing choice

- **Every interval.**
  - All features lower the average QLIKE in all three linear classes in both panels.
  - The reductions are small and not significant in Panel A (-0.1 to -1.0%).
  - In Panel B they are -3.1 to -3.5%, and all three are significant at 5%.
- **Closing half hour.**
  - All features raise the average QLIKE in all three classes in Panel A (+1.6 to +2.0%, none significant).
  - In Panel B they raise it in two classes and lower it in one (elastic net -0.4%). None of these is significant.
  - None of the old text's direction ("lower in every class") survives under this framing.
- **MSE.** All features raise the MSE in every class, framing and panel. None of the increases is significant.
- **Against the pooled OLS.**
  - In Panel A the penalized HAR-benchmark models have a significantly lower QLIKE than the OLS under both framings
    (differences of 0.1-0.7%).
  - In Panel B they coincide with the OLS (see the caution above).
  - Of the full-set forecasts, only those of Panel B under (a) are significantly below the OLS.

## Checks

- Panel A's closing-half-hour numbers reproduce the 48-interval rerun's own `SUMMARY.md` table, for example:
  - QLIKE: OLS 0.11159, ridge HAR 0.11144, enet Full 0.11251;
  - DM against the OLS: ridge HAR -3.84, lasso Full +0.61.
- Panel B's 13-bar closing numbers are scored on the same 1,082 days. The every-interval sets (50,386 and 14,066
  intervals) are Table 1's (`scored_set` imported).
- Run time: 21 min. Almost all of it is Table 1's `slot_benchmark`, called four times.
