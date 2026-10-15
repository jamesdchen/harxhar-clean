# Generator-side table and figure fixes (from the float-caption pass, 2026-10-15)

- **Source:** the floats editor found these when applying NOTATION_GUIDE sections 1-8 to `writeup/rehaul/*_floats.tex`.
  Generator agents apply them in their own generator. Guide section 6b overrides anything here.
- **Defaults for every table:**
  - bold → stars (`*`, `**`, `***` = 10/5/1%), stated in the notes;
  - drop $p$ columns; test statistics in parentheses;
  - panels "Panel A: Closing-interval models" and "Panel B: Pooled intraday models";
  - the guide's vocabulary;
  - typed numbers (866, 112, 51, 614, dates) → macros;
  - notes of about five sentences, referring to Methods instead of repeating it.
- **Decided symbols:**
  - $\mathrm{IV}_t$ for the implied volatility (not $IV^{30}_t$);
  - position $q_t = \sign(\hat\sigma^2_t - \mathrm{IV}^2_t)$, with no $s_t$; "the sign strategy";
  - Murphy threshold $\theta$, with elementary score $\mathrm{ES}_\theta$ (Ehm et al. 2016);
  - day index in sums: $t'$.

## make_rehaul_data_methods.py

- **rh_dm_sources:**
  - Caption: "Data sources."
  - Drop the `src/...` paths from the notes. "bar-end stamp" → "interval-end time stamp".
  - The `[vendor]` placeholders are still open; leave them and list them.
- **rh_dm_samples:**
  - Caption: "Estimation and evaluation samples."
  - Rows "Bar grid", "Per-bar 16:00 design", "Linear pre-tune rows", "Test period: 16:00 rows" and "Trade days" → guide
    terms.
  - Role column: walk-forward → out-of-sample, embargo → gap, shipped → fixed hyperparameters.
  - Panels: pooled intraday / closing interval / out-of-sample.
- **rh_dm_design:**
  - Caption: "Composition of the predictor sets."
  - Headers: "Per bar (16:00)" → "Closing-interval model"; Avail./Active → "Missing-data indicators" / "Activity
    indicators".
  - "HAR ladder of the target" → "HAR components"; HAR + calendar / All features → HAR benchmark / Full predictor set.
  - Notes: $\bar x_{t,k}$ → $x^{(k)}_t$; no code names (numobs, voldemand_*, fomc_*); refer to Section 4.
- **rh_dm_dictionary** (longtable):
  - Series names → guide/addenda symbols; drop the "Source file" column; "Diurnal" → "Periodicity adjustment".
  - Cut the 8+ sentence notes and the `src/...`, `apply_semantic_transform` and `robust_transform` mentions.
- **rh_dm_hyper and rh_dm_hyper_search:**
  - `\texttt` parameter names → words or symbols ($\eta$, $\lambda_2^{\mathrm{tree}}$).
  - Penalty $\alpha$ → $\lambda$; "$\rho$ (L1 share)" → $\alpha$; "$R_{BB}$, backbone" → $\underline\omega$ on the
    HAR-benchmark block.
  - Drop `lgbm_opt_00`, `xgb_opt_00` and `rf_opt_00`; "top entry"/"shipped" → fixed hyperparameters.
  - $L$ → $\ell$; $w_{\mathrm{enc}}, w_{\mathrm{lin}}$ → $\nu_{\mathrm{enc}}, \nu_{\mathrm{lin}}$;
    $\lceil 1/\mathrm{lr}\rceil$ → $\lceil 1/\eta\rceil$.
  - Drop `tp25_q`, the `specs/` and `experiments/` paths, and "sklearn units". Shorten the notes.
- **rh_dm_tuning:**
  - Caption: "Hyperparameter tuning, by model class."
  - Headers: Family / Refit / "Validation tail / embargo" → Model class / Re-estimation / Validation sample / gap.
  - Move the seed values out of the table. In the notes, $\sqrt{RV/B}$ → $y_{t,i}$.
  - The 7 prose `p{}` columns need trimming.

## make_rehaul_results.py

- **t1_axis1:**
  - Caption: "Forecast accuracy: HAR benchmark versus full predictor set."
  - Drop $p$ (8 → 7 columns).
  - Headers: HAR + cal. / All / $\Delta$SR → HAR / Full / $\Delta\SR$.
  - Notes: say actual realized variance (not winsorised); $\hat y_t^2B_t$ → $\hat y_t^{\,2}s^2_{t,c}$; American
    spelling; "Interim" → preliminary; "earlier paper's" → earlier draft's. Cut from about 9 sentences.
- **t2_pooled:**
  - Caption: "Forecast accuracy: closing-interval versus pooled intraday models."
  - Drop both $p$ columns (9 → 7). Same note fixes as t1.
- **t3_ranking:**
  - Caption: "Ranking of the main forecasts by mean QLIKE."
  - Drop $p$.
  - Plain model names. Replace "XGBoost, fixed, daily refit", "MinGRU (ridge-init, QLIKE-tuned)" and "Block-diagonal
    ridge, earlier panel". Fit "per-bar" → "Closing interval".
- **t3_full** (longtable): drop the file path in the notes and the "rand. search (SE)" and "refit 10" labels; move it
  to the appendix.
- **Figure rh_cumulative** (kept):
  - y-label → "Cumulative QLIKE loss differential vs closing-interval OLS".
  - Legend "Lasso, all features" → "Lasso".

## make_rehaul_economics.py

- **e1_control:**
  - Caption: "Returns of the sign strategy."
  - 9 columns: move Hit/Buys out, or go landscape. $t$ values in parentheses.
  - Mid / Crossed → Mid-quotes / Bid–ask; $\Delta$Sharpe → $\Delta\SR$.
  - Rows "OLS, HAR + calendar" / "..., all features" → guide terms; panel "Per-bar linear (16:00 bar)" →
    "Closing-interval linear models".
  - Notes: from 8+ sentences plus a/b notes to about five; "annualised per trade day" → American spelling, trading day;
    typed 0.20, 14 and 866 → macros.
- **e2_allfeat:**
  - Caption: "Sharpe ratios: HAR benchmark versus full predictor set."
  - Drop `\texttt{baseline}` from the notes. Move the "0/15" counts out of the $t$ columns.
- **e3_pooled:**
  - 9 columns: drop the bid–ask $\Delta$ pair (almost identical to mid), or split into panels.
  - State $q$ instead of "automatic Newey–West lag".
- **e4_sizing:**
  - Caption: "Sharpe ratios by position-sizing rule". The typed "253--866 (614 days)" → macros.
  - 9 columns → Panel A mid-quotes / Panel B bid–ask. $IV^{30}_t$ → $\mathrm{IV}_t$.
- **fig_cumpnl** (kept):
  - Caption: "closing trade", "at mid" and "family (all features)" → guide terms; typed +104.7, 2.99, 10 and 102 of 112
    → macros.
  - Axis: "(premia, mid)" → "(straddle premia, mid-quotes)".
  - FAM_LABEL: "Per-bar ..." → "Closing-interval ..."; "Pooled (48 bars)" → "Pooled intraday".

## make_rehaul_fi.py

- **fi_summary:**
  - Caption: "Variable importance by model."
  - HAR ladder → HAR components; `\texttt` series codes → symbols.
  - Notes: $\alpha$ → $\lambda$; backbone → HAR-benchmark block; refits/sessions → re-estimations / trading days.
- **fi_linear:**
  - 9 columns: split lasso / ridge, or drop Active and Mean rank.
  - The Unit column has code names → predictor-group names.
  - The bucket codes BB/Mom/Liq are undefined: spell them out.
  - Shorten the notes.
- **fi_trees:**
  - 11 columns: panel by model, or keep only $\bar r$ and Top 5.
  - Drop `lgbm_opt_00` and "de-dup study" from the notes.
- **fi_inter:**
  - Column pairs like `\texttt{har\_ma\_1}` × `\texttt{adj\_sumabsret\_ma\_1}` → $\bar y^{(1)}_t \times$ (addenda
    symbol).
  - Inline "Lundberg et al. 2020" / "Friedman–Popescu" → `\citet{lundberg2020local}` / `\citet{FriedmanPopescu2008}`
    (the bib agent is adding the latter).
  - Shorten the notes.
- **fi_fig_ranks** (kept):
  - Caption: per-refit → per re-estimation; units → predictor groups; held-out rows / tail → out-of-sample block;
    centred → centered.
  - Legend labels are code names: use the predictor-group names.

## make_rehaul_appendix.py (bodies inside the appendix wrappers)

- **tune_ladder:** relabel the rows. Current labels: "block-diag. ridge", "pooled twin", "shipped, refit every 10",
  "Campaign, design: ... session-edge duplicates", "per-bar ridge, one penalty (campaign's)".
- **tune_cadence:** $\Delta$SR → $\Delta\SR$; panels → "Panel A: HAR benchmark / Panel B: Full predictor set".
- **tune_mask and tune_budget_oos:** 9 columns each; split into stacked panels.
- **tune_budget:** "shipped" → "fixed"; typed "50 trials" → `\rhappOptTrials`.
- **tune_optuna_vs:** `TUNE\_PER` → "Re-tuning frequency (trading days)"; Arms → Specifications.
- **tune_picks:**
  - 10 columns: split H/A, or go landscape.
  - `\texttt` axis names → words. "rounds $\le$ cap(lr)" → $B \le \bar B(\eta)$.
- **mingru_dev and mingru_all:**
  - Drop $p$ (9 → 7 columns); put stars on $\Delta$%.
  - Walk → Specification. Drop the code tags (g8), (b), (fix 1).
  - "Ridge-initialised" → "Ridge-initialized".
  - "Pooled walk (every bar), read at 16:00" → guide terms.
  - $w$, $w_{\mathrm{enc}}$ → $\nu$, $\nu_{\mathrm{enc}}$.
  - The mingru_dev wrapper notes, in 10_appendix_floats.tex, contain a "$p$: ..." sentence, to be replaced by a star
    sentence in the final consolidation.
- **mingru_dayclass:** the "Walk" header and the "(b)" tag; typed $n$ = 43/60/763/866 → macros.
- **featimp_harma:** "Perm. P2 $\Delta$QLIKE" → "Permutation $\Delta$QLIKE"; `\texttt{\_x\_close}` share →
  "Close-interaction share".

## experiments/close_murphy_splits_headline.py (fig:rh_app_murphy, kept, currently a PNG)

- "threshold c" / "ES_c" → $\theta$ / $\mathrm{ES}_\theta$.
- Drop "plain f²B" from the panel title.
- Write a PDF instead of the raster.

## Open for the final consolidation pass

- **Section labels:** sections have no `\label`; add them, so notes can say Section~\ref{...} instead of "see Methods".
- **Floats captions still say `q_t = \sign(s_t)`:** rename the signal per guide 6b.
- **The floats use $u$ as a dummy day index:** unify to $t'$.
- **"Fit-space QLIKE"** is now worded "mean QLIKE on the scale of $y_{t,i}$". It is likely computed against the winsorized
  $\tilde y$; confirm, and re-score under decision #10 if so.
