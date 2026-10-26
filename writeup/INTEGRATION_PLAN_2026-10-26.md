# Integration plan, Monday 2026-10-26 (author decisions of 2026-10-25/26)

One pass over the paper after the tree ablations land. House rules apply: numbers only via generated macros, replaced
text parked in dated comments, Overleaf push with `--check` after the commit.

## Decisions (author)

| # | Decision | Consequence for the text |
|---|---|---|
| D1 | Headline target = every regular-hours bar (intraday RV), not the closing half hour | Rewrite title, abstract, research question; the closing half hour becomes the application and the final section |
| D2 | Primary benchmark = the pooled HAR benchmark (each model class's own HAR-set forecast on the 13-bar design). The closing-interval HAR (CI-HAR) is NOT a primary benchmark | Intro (01:80-82, 01:112), Section 6 "Benchmark forecast" (06:125) and every `\rhrVsBm`/`\rhrOls`/`\rhrNLower*` comparison move out of Sections 1, 6, 7 and 9 |
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
| D13 | FOMC fixes (missing 18 Mar 2020 meeting; no winsorizing of the proximity ramps) | Running (branch -fomcfix): linear 2x2 with holidays, then full-set trees; ablations/robustness rerun only if material (author decides) |

## Pending author decisions

- Tuning-provenance checks (audit #6): results coming from the FOMC-fix agent (penalty neighbours, annual re-tune, tree pre-sample neighbourhood); decide after.
- NYSE-holiday sessions (audit #3): measured in the 2x2; disclose or drop after.
- Complexity / scaling framing (Kelly-Malamud-Zhou 2024): grid and draft `drafts/2026-10-26_complexity.tex` coming after the ladder; author decides whether to make the argument.
- Clark-West on MSE as appendix robustness: proposed, awaiting go.
- Hoffman2 Xeon bit-for-bit CI-HAR gate: blocked by permissions; bundle staged; author to run or skip.

## Audit text fixes (no decision needed)

All confirmed findings in the audit file (scratchpad `audit_2026-10-25.md`; 89 findings), notably: economics significance
counts and "pinning"; the straddle pays on the move, not the variance; T_oos defined two ways and the DM lag; the
scaler description (rolling robust scaler, not the masked one); stale 48-interval text; FOMC appendix item; forward-fill
cap; SHAP sample 33,970 vs 33,527; FOMC statement days 81 not 78; dense-and-weak premise as a hypothesis; alternative-data
novelty claim; hard-coded constants.
