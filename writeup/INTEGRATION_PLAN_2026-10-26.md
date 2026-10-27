# Integration plan, Monday 2026-10-26 (author decisions of 2026-10-25/26)

One pass over the paper after the tree ablations land. House rules apply: numbers only via generated macros, replaced
text parked in dated comments, Overleaf push with `--check` after the commit.

## Target journal and page budget (author, 2026-10-26: "set the budget for jfec")

Journal of Financial Econometrics: submissions "should typically not exceed 40 double-spaced pages"; abstract at most
100 words; replication files required. In this template (single-spaced, 1-inch margins) that is about 22-24 pages of
main text including tables and figures, references excluded; everything else goes to an online appendix.

| Part | Now (pp.) | Budget (pp.) | Main-text content |
|---|---|---|---|
| Title + abstract | 1 | 1 | Abstract <= 100 words; no editorial notes |
| 1 Introduction | 2 | 2.5 | Every-bar question; 3 numbered contributions with effect sizes; the application in 2 sentences |
| 2 Related work | 3 | 1.5 | RV lineage, pooled intraday forecasting (regular session + overnight predictors), ML for RV, alternative data, economic evaluation incl. Maisch et al.; reading lists cut |
| 3 Data | 4 | 1.5 | Sources, sample, target; details to Appendix |
| 4 Predictors | 5 | 2 | Groups, HAR ladder on the full grid, late-start rule in one bullet; construction steps to Appendix |
| 5 Methods | 5 | 2.5 | Design (RTH13 pooled, W, refits), models, benchmark (RTH13 OLS-HAR), losses and tests |
| 6 Results | 9 (incl. VI 6) | 4 | Table 1; ablations / augmented-HAR table; window pointer; MSE clause |
| 7 Variable importance | (in 6) | 3 | SHAP shares (1 table or figure), feature-count ladder (1 figure), live-window bullet, use vs accuracy |
| 8 Economic application | 15 | 3 | Main table + by-year figure; profits from long days / few days; beta in 1 bullet |
| 9 Conclusion | 2 | 1 | |
| Total main text | ~46 | ~22 | |

Online appendix: implementation details, data and predictor construction, tuning disclosure (D14), holiday
disclosure (D15), robustness (windows, training arms, CI-HAR and closing-interval section, MCS/SPA/StepM), economics
details (sizing, compounding, shift test, Sharpe definitions, break-evens, beta and allocation, profit decomposition),
VI details (shapes, interactions, conditional, scaling grid), statistical tests. Reproducibility appendix (9 pp) becomes
a replication-package README with a half-page pointer.

## Decisions (author)

| # | Decision | Consequence for the text |
|---|---|---|
| D1 | Headline target = every regular-hours bar (intraday RV), not the closing half hour | Rewrite title, abstract, research question; the closing half hour becomes the application and the final section |
| D2 | Primary benchmark = OLS on the HAR benchmark set, pooled on RTH13 (one forecast: Table 1's OLS row; the HAR-set ridge/lasso/enet coincide with it). The closing-interval HAR (CI-HAR) is NOT a primary benchmark (author, 2026-10-26: "the primary benchmark is OLS on har for rth13") | Intro (01:80-82, 01:112), Section 6 "Benchmark forecast" (06:125) and every `\rhrVsBm`/`\rhrOls`/`\rhrNLower*` comparison against CI-HAR move out of Sections 1, 6, 7 and 9. Every main-text forecast (incl. the trees' full sets) is tested against the RTH13 OLS-HAR; the within-class full-vs-HAR-set pairs stay as the paired comparison of what the exogenous predictors add; main-text economics compare against the RTH13 OLS-HAR strategy and the short straddle |
| D3 | All closing-interval models (CI-HAR, close-trained linear and trees) are corralled into the final section ("A final refinement: training on the last 30 minutes", 07:495ff) and its appendix | That section is rebuilt on the pre-registered three-way CLOSE arm (current data chain) plus the rebuilt CI-HAR; the old master-table closing-interval model set (old data rules) is dropped |
| D4 | One training design for every main-text model: RTH13 estimation, predictors on the full 48-bar grid incl. overnight (precedent: Engle-Sokalska 2012, Zhang et al. 2024; overnight information: Stroud-Johannes 2014 etc.) | Done in text (commit 297a603); ALL48 / CLOSE / equal-row arms go to the robustness appendix (D5) |
| D5 | Training-arms robustness appendix | `writeup/make_rehaul_train_arms.py` + `drafts/2026-10-26_train_arms.tex` (built, gated) |
| D6 | Integrate E1/E2/A1/A2 + ablations (linear done; trees Monday) | drafts `2026-10-25_{econ_main,econ_decomp,close_accuracy,mcs,ablations}.tex`; re-run their generators on the rebuilt CI-HAR (D9) and drop CI-HAR from the main-text economics (D2) |
| D7 | Restore the MSE clause in abstract/intro/conclusion with the crash-days explanation (A1) | |
| D8 | Window choice disclosed; all windows reported in the appendix (audit #5) | Window sweep exists (results/window_sweep_2014start_2026-10-22) |
| D9 | CI-HAR rebuilt under the current data chain (results/ci_har_rebuild_2026-10-26) | Rebuild master table; re-run every consumer (list in the rebuild report); CI-HAR only in the final section |
| D10 | Beta / optimal-allocation analysis in the appendix | `make_rehaul_econ_beta.py` + `drafts/2026-10-26_econ_beta.tex`; CI-HAR row moves with D2/D3; trim macros to those used; bib: TreynorMazuy1966, GibbonsRossShanken1989 |
| D11 | Live-window reading of the alternative data | Done (commit c4d5392) |
| D12 | Cite Andersen-Su-Todorov-Zhang (round-the-clock E-mini diurnal curve) in the diurnal bullet | verify metadata first |
| D13 | FOMC fixes ADOPTED (author 2026-10-26): the paper's numbers come from the patched design (FOMC_SCHEDULE_2020_FIX, FOMC_PROXIMITY_NO_WINSOR; branch -fomcfix) | Patched linear + LightGBM/XGBoost done, forest last piece pending. Rerun on the patched design: linear ablations, FOMC tree arms (add_fomc x3, lo_fomc LightGBM/XGBoost), ladder + scaling grid, tree SHAP (and the live-window shares built on it). Other tree ablation arms and the three-way / window robustness arms stay pre-patch, disclosed |
| D14 | Tree tuning: disclose only (author: tuning only needs to put off-the-shelf models in the forecasting regime) | Appendix: transferred configurations, 2013 pre-sample neighbourhood check, penalty neighbours, annual re-tuning, oneDAL forest engine |
| D15 | NYSE-holiday sessions: KEEP and disclose (2 sentences + the measured linear 2x2 as robustness). **NOTE FOR OFFICIAL PUBLICATION: rerun with DROP_HOLIDAY_SESSIONS on (all models) before the final version.** | |
| D16 | No 16:00-bar accuracy results in the main text (author, 26 Oct: "why is the 1600 bar still in the main paper??"); they live in Appendix app:rh_closing. The economic application stays AS IS (author: "keep the econ application as is"): the 15:30 straddle signed by the pooled forecast of the 16:00 bar | Done 7f7894a (intro, methods, results, conclusion, ladder figure moved) |
| D17 | Two channels of the pooled gain: FOMC statement days vs ordinary days; tests fixed in results/channels_2026-10-26/RULE.txt (86ee222) | Items 1-3 done (04d9822); factor test running; Sections 6-7 draft (b81283d) to be redrafted on D18 |
| D19 | Primary Sharpe ratio = the standard pooled ratio (author, 26 Oct: "if standard pooled is proper, then do it"): each post-warmup option day weighs 1/T, annualized by sqrt(average option days per calendar year, ~225); Ledoit-Wolf test. D6 (equal-year mean of SR_y) and the per-year N_y-weighted ratio become robustness columns | NUMBERS agent regenerating the economics generators; claim check pending |
| D18 | The author quotients out the HAR, the futures' realized moments and the FOMC calendar ("those are well studied"): the question is what the other exogenous series add beyond HAR + RM + FOMC. Forest lo_fomc STOPPED (cost 3,550 vs 2,500 CPU-h); new arm add_rm_fomc (HAR + RM + FOMC, all six classes + OLS) on the patched design | Exploratory read (26 Oct, existing arms): beyond HAR + RM on ordinary days, linear +0.2 to -0.5% (n.s.), trees -1.2 to -1.6% (DM -3.2 to -4.8), forest -0.7% (DM -2.2). Arm running (ablation agent) |

## Pending author decisions

- Tuning-provenance checks (audit #6): DECIDED, disclose only (D14).
- NYSE-holiday sessions (audit #3): DECIDED, keep + disclose; rerun with them dropped before official publication (D15).
- Complexity / scaling framing (Kelly-Malamud-Zhou 2024): grid and draft `drafts/2026-10-26_complexity.tex` coming after the ladder; author decides whether to make the argument.
- Clark-West on MSE as appendix robustness: proposed, awaiting go.
- Hoffman2 Xeon bit-for-bit CI-HAR gate: blocked by permissions; bundle staged; author to run or skip.

## Audit text fixes (no decision needed)

All confirmed findings in the audit file (scratchpad `audit_2026-10-25.md`; 89 findings), notably: economics significance
counts and "pinning"; the straddle pays on the move, not the variance; T_oos defined two ways and the DM lag; the
scaler description (rolling robust scaler, not the masked one); stale 48-interval text; FOMC appendix item; forward-fill
cap; SHAP sample 33,970 vs 33,527; FOMC statement days 81 not 78; dense-and-weak premise as a hypothesis; alternative-data
novelty claim; hard-coded constants.
