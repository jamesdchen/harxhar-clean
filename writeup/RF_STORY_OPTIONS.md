# N15c and the pooled random forest: diagnosis and options (2026-10-22)

This is a read-only note. Nothing was rerun, and no generator, generated file, result or `.tex` file was changed.
Scripts: the session scratchpad `rf_story/{diag,diag2,timing}.py`. They import `writeup/pooled_trees_13.py`,
`writeup/pooled_linear_13.py`, `asl.sharpe_ann_by_year` and `hi.lw_sharpe_test`.
Trading: q = +1 if plain > iv_var, otherwise −1, with return qR. R is ret_mid·q of `sub_ols_baseline`. Sharpe ratios are at
mid, on the 614 post-warmup days (2021-08-11..2024-04-30), annualized year by year and weighted by N_y.
**These numbers describe the evaluation sample. None of them was used, or may be used, to choose a specification.**

## 0. The main point: fixing the RF alone would not restore the story

The coordinator is moving the six pooled linear rows of Section 7 to the 13-bar rerun on the trees' design
(`results/pooled_linear_rerun_2026-10-20_13_2012_w1000`, `pooled_linear_13.py`). On that shared design **the full
predictor set lowers the Sharpe ratio in 4 of the 6 pooled classes (the three linear models and the RF), not 1**.
No change to the RF can make "every intraday model class" true again. Tuning anything until the sentence holds would
be specification search on the evaluation sample.

| class | SR HAR | SR full | ΔSR | Ledoit–Wolf p (block 10 / 5 / 20) | t of mean differential (LLSW) | ΔSR, fixed-factor √252 SR |
|---|---|---|---|---|---|---|
| Elastic net | 1.30 | 1.13 | −0.17 | 0.754 / 0.771 / 0.770 | 0.33 | +0.11 |
| Lasso | 1.28 | 1.02 | −0.26 | 0.996 / 0.998 / 0.997 | −0.01 | −0.00 |
| Ridge | 1.30 | 1.09 | −0.22 | 0.963 / 0.962 / 0.966 | −0.05 | −0.02 |
| LightGBM | 0.96 | 1.07 | +0.11 | 0.285 / 0.273 / 0.296 | 1.14 | +0.55 |
| XGBoost | 1.03 | 1.34 | +0.31 | 0.181 / 0.177 / 0.176 | 1.44 | +0.69 |
| Random forest | 1.21 | 1.03 | −0.18 | 0.985 / 0.988 / 0.989 | 0.02 | +0.01 |

- All six values match the coordinator's figures.
- The old 48-interval, 2,000-day linear rows reproduce as 0.74 → 1.56, 0.74 → 1.46 and 0.75 → 1.44.
- **No difference is significant (Ledoit–Wolf p from 0.18 to 0.996).** For lasso, ridge and the RF, the mean daily returns of
  the full-set and HAR strategies differ by at most 0.0016 (|t| ≤ 0.05).
- The sign of ΔSR is noise:
  - **5 days decide it.** Removing each class's 5 best differential days makes the summed differential negative in
    5 of 6 classes; removing the 5 worst makes it positive in all 6.
  - **Elastic net and RF flip with the annualization rule.** Under the fixed-factor ratio used until 2026-10-18, their
    ΔSR is positive.
- Switching back to the fixed-factor ratio for that reason is **not** an option (see F below).
- **QLIKE on this design.** On the 16:00 rows of the 1,082-day span, the full set *raises* QLIKE for lasso
  (0.12030 → 0.12292, +2.2 %) and ridge (0.12045 → 0.12362, +2.6 %). If T1 moves to these rows, the abstract's
  QLIKE claim "across all 6" also becomes 4 of 6. The all-6 assert in `make_rehaul_results.py:3073` is already parked.

## 1. Why the RF differs: the hypotheses tested

| hypothesis | fact (config / code) | evidence (614 days, descriptive) | verdict |
|---|---|---|---|
| Frozen hyperparameters | `rf400_rf_opt_00` is trial 255 of a 400-trial TPE search (6,224 CPU-h). That search used the 48-interval grid, the base-2 full set with 1,144 columns, rows before 2003, a 24,000-bar window, a refit every 250 bars, and sqrt-space MSE. **The HAR-set RF was never tuned.** LightGBM and XGBoost come frozen from the same menu. Only the linear models were re-tuned, on pre-2012 folds of this design. | — | Real, but it applies to **all three trees**, not only the RF |
| Leaf size unscaled | min_samples_leaf is 12, as tuned for 24,000 rows. With max_samples 0.413 a tree sees 9,911 → 5,369 rows, so a leaf grows from 0.12 % to 0.22 % of a tree's rows. Scaled to the window it would be 6.5. The GBMs are unscaled too: min_child_samples 98 (scaled 53) and min_child_weight 50.2 (scaled 27.2). max_depth 20 does not bind, since a tree has at most about 450 leaves. | Both RF arms carry the same leaf. There is no over-smoothing: the log-MZ slope is 0.891 (HAR) and 0.892 (full). | Not a cause of the full-vs-HAR gap |
| Feature subsampling at 576 columns | max_features 0.901 means 519 of 576 candidates per split (10 of 12 on the HAR set). P(a given column is left out of a split) = 0.099. P(all 6 HAR columns are left out) ≈ 1e-6. | — | Rejected: the HAR columns are not diluted |
| Shrinkage toward the mean | — | sd of log f is 0.910 → 0.919 on the 614 days and 0.915 → 0.960 over 2012–2019. log f_full = 0.008 + 1.001·log f_HAR. | Rejected: the full-set forecast is slightly *more* dispersed |
| Bias or scaling | plain = pred_adj²·B for all 12 forecasts | mean log(f/rv): −0.014 / −0.016. Mean log(f/iv): −0.241 / −0.242. Long days: 21.0 % / 21.5 %. | Rejected: the level is the same |
| Noise | — | The two strategies disagree on 69 days (36 full-long, 33 full-short). The differential sums to **+0.33** (mean 0.0840 vs 0.0835 a day). By year: +6.32 (2021, 64 d), −3.53 (2022, 219 d), −8.87 (2023, 248 d), +6.41 (2024, 83 d). The N_y weights put 76 % of the weight on 2022–23. | **The cause.** ΔSR = −0.18 is the per-year weighting of a zero mean differential |

The full-set RF changes its HAR twin less than any other class does: the correlation of the two log forecasts is
0.9911, against 0.983–0.990 for the other classes. It also gains accuracy (QLIKE 0.1190 → 0.1140, −4.2 %).
The GBMs' positive ΔSR is not robust either. Their HAR arms forecast too high (log bias +0.023 / +0.019, 26 % long
days), and their gains rest on a few days: without the best 5 days, the LightGBM differential is −2.12.

**Pre-sample evidence.**
- **The trees have no pre-2012 out-of-sample output on this design.** The walk starts on 2012-01-03.
- **The RF's own tuning menu is flat.** The top 8 of 400 trials lie within 0.10 % of each other (MSE 0.108456–0.108565),
  with leaf sizes from 10 to 27 and max_samples from 0.22 to 0.88. At 87 trials the pick was leaf 24 with
  max_samples 0.78.
- **A pre-tune of the closing-interval RF on the same fold template** (`results/trees_pretune_2026-10-06`):
  - Validation MSE, best/shipped: 0.940 (full set) and 0.907 (HAR set). Held out: 0.988 and 0.954.
  - The trials' fold-rank correlations average 0.16, with a minimum of −0.06.
  - Out of sample, the tuned arms moved the RF's Sharpe ratio (866 days, fixed-factor) by −0.29 to +0.26, with
    |HAC t| ≤ 1.09:
    - full set, refit every 10: 1.67 → 1.55–1.82;
    - HAR set, refit every 10: 1.00 → 0.71–0.81;
    - HAR set, refit every session: 1.06 → 1.13–1.32.
  - **So a re-tune moves the RF's Sharpe ratio by about the size of the gap, in either direction.**

## 2. Options

Compute basis: the E34 walk of the trees on CARC used jobs 13262219, 13265608 and 13265651, plus the merge 13278970.
- **Wall time:** 2026-10-20 12:27 → 22:35 (10.1 h, including 81 tasks that failed on Intel nodes and were rerun); the
  longest task took 5.3 h.
- **Allocated:** 5,654 CPU-h.
- **Piece CPU-h by arm:** RF full **4,995** (190 pieces, 53 s a refit on 8 EPYC-7513 threads), RF HAR 105,
  LightGBM 159 + 16, XGBoost 278 + 12 (total 5,565).
- **One RF full-set configuration on the 4 pre-2012 folds** (4 fits on 13,000 rows each) costs about 0.5 CPU-h.

| | option | (a) justification, whatever the outcome | (b) what changes | (c) compute | (d) risks |
|---|---|---|---|---|---|
| A | **Revise the claim to a count.** The macros already exist: `\rhrAfPoolShHigher` of `\rhrAfPoolPairs`, `\rhrAfPoolShHigherSig`, `\rhrAfPoolDShMin/Max`. | The text must describe the numbers. | Abstract (`paper_rehaul.tex:134`), `01_introduction.tex:92`, `09_conclusion.tex:33`; `07_economics.tex:297` already prints the count. Possible wording: "raises the Sharpe ratio in 2 of the 6 classes; none of the six differences is significant at 5 % (Ledoit–Wolf)". | 0 (the generators are being rerun anyway) | None. It weakens a headline. |
| B | **Report the RF as it is**, with one diagnostic sentence (equal mean returns, LW p 0.985) | The RF ran under the same frozen rule as LightGBM and XGBoost, and §1 finds nothing broken | One sentence in §7 | 0 | None. Goes together with A. |
| C | **Re-tune all three trees, on both sets, on the 4 pre-2012 folds** (the linear template: `frozen_penalties.json` folds, 325-row embargo, lowest mean validation MSE over all 13 bars) | The 12 pooled forecasts would then all be tuned the same way on one design. The HAR-set trees were never tuned at all. | New configurations for 6 tree arms, then a full re-walk. Every tree number in §§6–7 changes. | Tune: RF full ≈ 200 CPU-h for 400 TPE trials (≤ 500 at the box's corners) or ≈ 6 CPU-h for a 12-point grid fixed in advance; other arms 4–11 CPU-h each. Walk: 5,565 CPU-h, ≈ 10 h wall. **Total ≈ 5,800 CPU-h, about a day.** | Must be pre-registered: search space, budget, seed, rule, rounding, and **adoption whatever the result**. Expect each tree's SR to move by about −0.3 to +0.3, in either direction. It cannot restore "every class". If it is launched *because of* N15c, it is a forking path. |
| D | Re-tune the RF only | None: every argument for it also applies to LightGBM and XGBoost | RF rows only | ≈ 5,300 CPU-h | It is chosen by its outcome, so it is specification search. **Not acceptable.** |
| E | Rescale every leaf minimum to the window by rule (RF 12 → 6.5, rounding fixed in advance; LightGBM 98 → 53; XGBoost 50.2 → 27.2) | `specs/causal_tune_trees.py` already scales the GBM leaf minimum by window/24,000 | The 6 tree arms are re-walked | ≈ 5,600–6,100 CPU-h (a smaller leaf adds about 1 tree level) | A weaker version of C (one axis only), with the same story risk |
| F | **Not acceptable:** another entry from the menu (rf_opt_01..07), the "7th entry" rule, going back to the fixed-factor SR (it would give 4 of 6), changing the warmup, sample or fills, dropping days | — | — | — | Specification search on the evaluation sample |

**Recommendation: A + B now.**
- **Why:** the three linear classes break the story whatever happens to the RF. No class's difference is significant.
  The RF shows no defect.
- **Why not a re-tune:** about 5,800 CPU-h, and its effect on the sign is a coin flip.
- **When to do C:** only as a separate consistency decision, pre-registered, after the text is already in count form,
  so that no result of it is needed to save the story.

## 3. Questions for James

1. Do you approve A + B: replace "every intraday model class" in the abstract, introduction and conclusion with the
   count? To print "none significant", the generator needs a new count of classes with p < 0.05 in either direction
   (`\rhrAfPoolShHigherSig` counts only ΔSR > 0). That is a change to `make_rehaul_results.py`. Should I make it?
2. Given that 0 of 6 differences are significant, should the abstract keep a full-vs-HAR Sharpe sentence at all?
3. Do you want C (re-tune all three trees, on both sets, on the pre-2012 folds) as a consistency fix in its own right,
   independent of N15c? If so, please approve the pre-registration and about 5,800 CPU-h (about a day) on CARC.
   D (the RF alone) is not offered.
4. Should the QLIKE claim ("lower in all 6") get the same count treatment once T1 uses the 13-bar linear rows?
   Lasso and ridge then show higher QLIKE with the full set.
   *Update (coordinator, same day):* done in this pass. T1 now uses the 13-bar linear rows, and the abstract,
   introduction, Section 6 and conclusion print `\rhrAfPoolLower` (4) of `\rhrAfPoolPairs` (6), with
   `\rhrAfPoolLowerSig` (0) significant. The old text is parked. Only the Sharpe sentence (N15c) is left for you.
