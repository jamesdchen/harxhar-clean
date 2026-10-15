# Notation and language guide for the rehauled paper (2026-10-15)

- **User's request:** "use standard symbols and language. Model it after the papers in the financial and financial economic
  fields."
- **This file is the core standard.** Every editor applies it. If a case isn't covered, choose the most standard usage in
  financial econometrics (J. Finance, JFE, RFS, J. Econometrics, J. Financial Econometrics, JBES) and record the choice
  in your final reply under "new decisions".
- **Shared macros** are defined in `writeup/paper_rehaul.tex`: `\RV`, `\E`, `\Var`, `\Cov`, `\SR`, `\DM`, plus the
  existing `\QLIKE`, `\argmin`, `\softplus`, `\clip`. Do not define macros in section files.

## House rules (unchanged)

- **Structure:** keep the bullet structure and formulas. Change symbols and wording, not structure.
- **Numbers:** every number stays a generated macro (`\rh...`, `\cm...`, `\fqs...`). Never type a number in place of a
  macro, and never rename a macro.
- **Labels:** never rename a `\label` / `\ref` key.
- **Park, don't delete:** when a sentence or block is replaced in substance, comment out the old one with
  `% 2026-10-15 (notation pass): parked -- <why>`.
- **Comments:** keep `%` code-pointer comments. Update only the symbols they mention.
- **Spelling:** American (stabilized, initialized, penalized, modeling).

## 1. Indices and timing

| Concept | Old | New | Source |
|---|---|---|---|
| Trading day | t (session), "session", "trade day" | $t = 1, \dots, T$; "trading day" | standard |
| Intraday 30-minute interval | t (bar), "bar", "slot" | $i = 1, \dots, M$; "intraday interval"; "interval $i$" | Andersen and Bollerslev (1997) |
| The 15:30-16:00 interval (target) | "16:00 bar", "the bar stamped 16:00" | the closing interval $i = c$; "the closing half hour (15:30-16:00 ET)" | |
| Forecast origin | "15:30", "entry" | forecast origin 15:30 ET; information set $\mathcal F_{t}$ (all data through 15:30 on day $t$) | |
| Conditional expectation | $\mathbb E_t$ | $\E[\,\cdot \mid \mathcal F_t]$ or $\E_t[\cdot]$ (define once) | |
| Generic row in a window | $i$ (row) | $s$ (rows of the estimation window), so $i$ stays the intraday interval | |
| Out-of-sample length | $n$ scored days | $T_{\mathrm{oos}}$ (use the existing macro for its value) | |

When only the closing interval is modeled, write $\RV_t \equiv \RV_{t,c}$. State this once in Methods.

## 2. Volatility quantities

| Concept | Old | New | Source |
|---|---|---|---|
| One-minute log return | (implicit) | $r_{t,i,k}$, $k = 1, \dots, 30$ | ABDL (2003) |
| Interval realized variance | $RV_t$, $RV_j$ | $\RV_{t,i} = \sum_k r_{t,i,k}^2$ (upright `\RV`) | ABDL (2003); Barndorff-Nielsen and Shephard (2002) |
| Diurnal baseline | $B_t$, "diurnal baseline", "B" | intraday periodicity factor $s^2_{t,i}$: the mean of $\RV$ in interval $i$ over the previous 20 trading days; "periodicity factor" | Andersen and Bollerslev (1997); Taylor and Xu (1997) |
| Target | $y_t = \sqrt{RV_t/B_t}$ | $y_{t,i} = \sqrt{\RV_{t,i} / s^2_{t,i}}$, "periodicity-adjusted realized volatility" | |
| Winsorized target (fit only) | $\tilde y_t$ | $\tilde y_{t,i}$; "winsorized at the rolling 1st/99th percentiles" | |
| Fitted / forecast | $\hat y_t$ | $\hat y_{t}$; variance forecast $\hat\sigma^2_{t} = \hat y_t^{\,2}\, s^2_{t,c}$ | |
| Scoring | QLIKE vs the winsorized target | QLIKE vs the actual $\RV_{t,c}$ (user decision 2026-10-15, data audit #10): $L_t = \QLIKE(\RV_t, \hat\sigma^2_t)$ | Patton (2011) |
| Loss function | $\QLIKE(RV,\hat\sigma^2)$ | $\QLIKE(\RV, \hat\sigma^2) = \RV/\hat\sigma^2 - \log(\RV/\hat\sigma^2) - 1$ | Patton (2011) |
| Squared-error loss | "MSE" | $\mathrm{MSE}$; "squared-error loss" | |

## 3. Predictors and models

| Concept | Old | New | Source |
|---|---|---|---|
| Predictor vector | $x_t \in \mathbb R^p$, "features" | $\bm x_t \in \mathbb R^{p}$; "predictors" in prose ("features" only for ML methods) | Gu, Kelly and Xiu (2020) |
| HAR averages | $\bar y_{t,k}$, "ladder", "ma_k" | $\bar y^{(k)}_{t}$, $k \in \mathcal K$; "HAR components at horizons $k$" | Corsi (2009) |
| Calendar dummies | $c_t$, coefficient $\gamma$ | $\bm d_t$, coefficient $\bm\delta$ ($\gamma$ is reserved for the tree split penalty) | |
| Regression coefficients | $\beta$ | $\bm\beta$, intercept $\beta_0$ | |
| Penalty strength | $\alpha$ (scikit-learn units) | $\lambda$. Say once that the grid is in the scikit-learn parametrization of the objective shown | Tibshirani (1996); Hoerl and Kennard (1970) |
| Elastic-net mixing | $\rho$ (lasso 1, enet 0.5) | $\alpha \in [0,1]$ (lasso $\alpha = 1$, elastic net $\alpha = 0.5$) | Zou and Hastie (2005); Friedman, Hastie and Tibshirani (2010) |
| Per-column penalty weight | $\mathrm{pf}_j$ | $\omega_j$, "penalty weight", defined at first use | Zou (2006) (adaptive lasso weights) |
| Block-diagonal penalty | "block-diagonal Tikhonov", "blk2" | "block-penalized ridge": $\omega_j = \underline\omega$ on the HAR-benchmark block, $1$ on the additional predictors, $0$ on the intercept | |
| Estimation window | $\mathcal T_t$, $T_W$ | rolling estimation window $\mathcal W_t$ of length $W$ | |
| Fit / embargo / validation | $\mathcal F_t, \mathcal E_t, \mathcal V_t$ ($\mathcal F$ clashes with the information set) | $\mathcal W^{\mathrm{est}}_t$, $\mathcal W^{\mathrm{gap}}_t$, $\mathcal W^{\mathrm{val}}_t$; "estimation sample", "gap", "validation sample" | Gu, Kelly and Xiu (2020) |
| Generic parameter | $\theta$ | $\bm\theta$; hyperparameters $\bm\vartheta \in \Theta$ | |
| Boosting rounds | $m$, $F_m$, $f_m$ | $b = 1, \dots, B$, $F_b$, $f_b$; learning rate $\eta$ | Friedman (2001); Chen and Guestrin (2016) |
| Tree regularization | $\gamma J + \lambda_1 \lVert w\rVert_1 + \tfrac12\lambda_2\lVert w\rVert_2^2$ | keep. These are XGBoost's own symbols; write $\lambda_1^{\mathrm{tree}}, \lambda_2^{\mathrm{tree}}$ if next to the linear $\lambda$ | Chen and Guestrin (2016) |
| Random forest | $N$ trees $T_m$ | $N$ trees $\mathcal T_b$, $b = 1, \dots, N$ | Breiman (2001) |
| Logistic function | $\sigma(\cdot)$ (clashes with volatility) | $\Lambda(a) = 1/(1+e^{-a})$ | econometrics convention |
| Network hidden state / gates | $h_\tau, z_\tau, \omega_\tau$ (clashes with $\omega_j$) | $\bm h_\tau$, update gate $\bm z_\tau$, highway gate $\bm\psi_\tau$ | |
| Network look-back | $L$ (clashes with the loss $L_t$) | look-back length $\ell$, written $\bm x_{t-\ell+1:t}$; per-observation training loss $\mathcal L$ | |
| Weight decay | $\mathrm{wd}$, $\mathrm{wd}_{\mathrm{enc}}$, $\mathrm{wd}_{\mathrm{lin}}$ | $\nu$, $\nu_{\mathrm{enc}}$, $\nu_{\mathrm{lin}}$; "weight decay" | |
| Training moments | $\mu_y, \sigma_y$ | keep | |
| Seeds, epochs | $S$, $E_{\max}$ | keep | |

## 4. Evaluation and tests

| Concept | Old | New | Source |
|---|---|---|---|
| Loss differential | $d_t = L^A_t - L^B_t$ | keep: "loss differential" | Diebold and Mariano (1995) |
| DM statistic | DM, DM$^\ast$ | $\DM$; "the Diebold-Mariano statistic with the Harvey-Leybourne-Newbold small-sample correction" | Harvey, Leybourne and Newbold (1997) |
| HAC variance | $\hat\Omega$, bandwidth $\ell$ | $\hat\Omega$; Newey-West lag truncation $q$ (since $\ell$ is now the look-back) | Newey and West (1987) |
| Sharpe ratio | "Sharpe", "Sharpe mid / crossed" | $\SR$; "annualized Sharpe ratio at mid-quotes" / "...net of the quoted bid-ask spread" | |
| Sharpe test | "Newey-West $t$ (lag 14)" | "HAC $t$-statistic on the daily return differential (Newey-West, $q = $ macro)" | |
| MCS / SPA / StepM | keep | "model confidence set", "superior predictive ability", "StepM" | Hansen, Lunde and Nason (2011); Hansen (2005); Romano and Wolf (2005) |
| Murphy diagram | keep | "Murphy diagram (elementary scores)" | Ehm, Gneiting, Jordan and Krüger (2016) |
| Significance wording | "bold = p < 0.05", "significant" | "significant at the 5% level"; in tables, "boldface: significant at the 5% level" | |

## 5. Terminology (code jargon to standard)

| Old | New |
|---|---|
| per-bar model, "per bar" | **closing-interval model** (estimated on the closing half hour only, one observation per trading day) |
| pooled model, "pooled" | **pooled intraday model** (estimated on every 30-minute interval) |
| HAR + calendar (input set), "baseline" | **HAR benchmark** (HAR components and calendar dummies) |
| all features, "all_features" | **full predictor set** |
| arm, f arm, subpt, sub_* | specification, or drop the code name and describe it |
| walk, walk-forward backtest | rolling-window out-of-sample exercise; "re-estimated daily" |
| test period, test window | out-of-sample period |
| trade days | trading days |
| shipped settings, untuned | fixed hyperparameters |
| pre-tuning, pre-tune | hyperparameter selection on a pre-sample |
| fit block / validation tail / embargo | estimation sample / validation sample / gap |
| gate | replication check |
| ladder (HAR ladder, avail ladder) | HAR components; missing-data indicators |
| avail, `_active`, `_avail` | missing-data (availability) indicator |
| plain forecast | forecast |
| ridge-init, QLIKE-tuned MinGRU | MinGRU network (ridge-initialized, trained and selected on QLIKE) |
| family (linear / trees / MinGRU) | model class |
| straddle trade | delta-neutral at-the-money straddle position, entered at 15:30 and held to expiration |
| crossed | at quoted bid/ask prices |
| dagger / interim | keep the dagger; "preliminary" in prose |
| `adj_x_ma_k` (column names) | $x^{(k)}_t$, "series $x$ averaged over $k$ intervals" in typewriter-free notation. Example: `adj_vix_ma_25` becomes $\mathrm{VIX}^{(25)}_t$. Series symbols: $\mathrm{VIX}$, $\mathrm{VIX3M}$, $\mathrm{VVIX}$, $\mathrm{OFI}^{\mathrm{EW}}$, $\mathrm{OFI}^{\mathrm{VW}}$, $\mathrm{RV}^{\mathrm{EW}}$, $\mathrm{RV}^{\mathrm{VW}}$, $\mathrm{Spread}^{\mathrm{VW}}$, $\mathrm{FOMC}$ |

## 6b. Amendments from the editors (binding)

- **$\varphi$, $\phi$:** reserved for SHAP values ($\phi_{t,A}$). The MinGRU candidate activation is renamed
  $\Lambda^{+}(a)$, defined as $a + \tfrac12$ for $a \ge 0$ and $\Lambda(a)$ otherwise (Methods and Appendix).
  - Correction to an earlier note: not $\chi$, because $\chi_{t,i}$ is the activity indicator (Predictors).
  - The network's recurrent branch $g(\bm x_{t-\ell+1:t})$ is renamed $f^{\mathrm{NN}}(\bm x_{t-\ell+1:t})$.
  - $g(\cdot)$ is the variance-stabilizing map (Predictors).
- **Symbols fixed by the Data and Predictors sections** (03/04, done; their header comments hold the full map):
  - one-minute returns $r_{t,i,k}$, $k = 1, \dots, N_{t,i}$, with $N_{t,i} \le 30$;
  - $(t, i-j)$ is the $j$-th retained interval before $(t,i)$, possibly on an earlier day; $\mathcal I$ is the set of
    retained intervals;
  - empirical quantile $Q_p(\cdot)$;
  - availability indicator $a_{t,i}$; activity indicator $\chi_{t,i}$; asinh scale $\kappa_{t,i}$;
  - IQR floor $\mathrm{IQR}^{\ast(k)}_t$; screened columns $\mathcal J_t$;
  - realized semivariance, bipower variation $\mathrm{BV}_{t,i}$, realized autocovariance $\mathrm{AC}_{t,i}$;
    $\mathrm{OFI}^{\mathrm{EW/VW}}_{t,i}$;
  - indicator averages $\bar a^{(k)}_t(u)$, $\bar\chi^{(k)}_t(u)$; EW/VW superscripts on constituent aggregates;
  - $\mathrm{VIX}^{(25)}_t$ is the average over the 25 intervals ending at 15:30.
  - The section title is "Predictors" (was "Features").
  - Generators and the addenda's lookup follow these choices.
- **Out-of-sample sets:** written $\mathcal O$, with $|\mathcal O|$ days, defined per exercise. The headline period
  (2020-2024) and the variable-importance period (2018-2024) differ, so each section defines its own.
  $T_{\mathrm{oos}}$ is used only for the headline period.
- **$W$:** Kendall's $W$ keeps its symbol. The window length is $W$ only in Methods; never use both in the same
  section.
- **Refit index:** $k$ in the variable-importance section, subscripted ($\hat\beta_{k,j}$). HAR horizons use the
  superscript $(k)$.
- **Standard deviation of a predictor:** $\mathrm{sd}_{k,j}$ ($\hat\sigma$ is the variance forecast).
- **Section 8's title:** "Variable importance". Terms: "predictor group"; "selection frequency" (lasso); "absolute
  standardized coefficient" (ridge); "mean absolute SHAP value" (trees); "interaction effects".
- **Series without a symbol in section 5:** the return-moment, liquidity and sentiment series (sumret, sumabsret,
  sumautocov, sumvolume, numobs, stocktwits_*, ...). Their names come from the addenda's lookup
  (`NOTATION_GUIDE_ADDENDA_2026-10-15.md`), never `\texttt{...}` code names.

- **Trading-strategy symbols (fixed by the Introduction):**
  - $\mathrm{IV}_t$ is the closing-half-hour implied volatility, so $\mathrm{IV}_t^2$ is the implied variance.
  - Position $q_t = \sign(\hat\sigma^2_t - \mathrm{IV}^2_t)$.
  - $R_t$ is the straddle's simple return.
  - The unconditional short-straddle benchmark is $q_t \equiv -1$.
  - No $s_t$ for the signal; $s$ is the periodicity factor.
- **Out-of-sample day set:** $\mathcal T_{\mathrm{oos}}$, with $|\mathcal T_{\mathrm{oos}}| = T_{\mathrm{oos}}$ (headline).
  Variable importance uses its own $\mathcal O$.
- **Forecast index:** $f_{m,\mathcal X,\mathcal I}$, where $m$ is the model, $\mathcal X \in \{\text{HAR}, \text{Full}\}$
  the predictor set, and $\mathcal I \in \{\{c\}, \{1,\dots,M\}\}$ the estimation sample (closing interval or pooled).
- **Row index inside an estimation window:** $t'$ (sums $\sum_{t' \in \mathcal W_t}$), not $s$, since $s$ is the
  periodicity factor. The minute index $k$ in $r_{t,i,k}$ is a local summation index only.

## 7. Figures parked (user, 2026-10-15)

The user said "comment out most figures. A lot of them are non-standard. I want to focus on making the tables legible
for now."

- **Kept, the four standard ones:**
  - `fig:rh_cumulative`: cumulative QLIKE difference against the benchmark (Welch and Goyal 2008 style).
  - `fig:rh_econ_cumpnl`: cumulative straddle P&L.
  - `fig:rh_fi_ranks`: variable-importance ranks (Gu, Kelly and Xiu 2020 style).
  - `fig:rh_app_murphy`: Murphy diagram.
- **Parked, every other figure:** comment out the whole figure environment, or its `\input` line, with
  `% 2026-10-15 (user): parked -- non-standard figure; focus on legible tables`. The parked labels are:
  - `fig:rh_axis1_forest`, `fig:rh_pooled_dumbbell`, `fig:rh_se_share`;
  - `fig:rh_econ_qlike_sharpe`, `fig:rh_econ_byyear`, `fig:rh_econ_sizing`;
  - `fig:rh_fi_linear`, `fig:rh_fi_inter`;
  - `fig:rh_app_fi_ranked`, `fig:rh_app_fi_heat_series`, `fig:rh_app_fi_heat_cluster`, `fig:rh_app_fi_time_trees`,
    `fig:rh_app_fi_time_linear`;
  - `fig:rh_app_shift`, `fig:rh_app_wealth`.
- **References to parked figures:** a `\ref` in the text to a parked figure points instead to the table with the same
  numbers. If there is none, park the sentence. No `??` may remain.

## 8. Tables (finance-journal style; the current focus)

- **Caption:** a short title only, e.g. "Forecast accuracy: HAR benchmark versus full predictor set". Everything else
  goes in the notes.
- **Notes:** below the table, in `threeparttable` `tablenotes`, starting `\textit{Notes:}`.
  - At most about five sentences: the sample (out-of-sample period, number of trading days, as macros), the loss, the
    test and its lag, the significance markers, and any symbol not defined in the text.
  - Method details belong in Methods. Refer to them ("see Section~\ref{...}") instead of repeating them.
- **Significance:** finance convention. Use stars on the statistic or estimate: `$^{*}$`, `$^{**}$`, `$^{***}$` for
  the 10%, 5% and 1% levels, stated in the notes. Drop separate $p$-value columns; the stars carry them. Test
  statistics go in parentheses, either in their own column or under the estimate.
- **Columns:** at most about 7. Group headers with `\cmidrule`. Units go in the header ("$\Delta$ (%)").
  - Use the same decimals throughout a column: QLIKE 4, percentages 1, DM and $t$ 2, $\SR$ 2.
  - Use right-aligned `r` columns with fixed decimals, so the decimals line up. Do NOT use siunitx: the local MiKTeX
    can't load it.
- **Rows:**
  - Model names in words: OLS, Ridge, Lasso, Elastic net, LightGBM, XGBoost, Random forest, MinGRU.
  - Group the rows in panels: "Panel A: Closing-interval models", "Panel B: Pooled intraday models".
  - No code names, keys, file names, `adj_`/`ma_` column names or arm labels anywhere in a table.
- **Size:** never `\scriptsize` or `\tiny`; `\small` at most.
  - A table too wide at `\small` either splits into panels or goes landscape (`\begin{landscape}` from `pdflscape`).
  - Long listings (every forecast, the feature dictionary, every tuning trial) go to the appendix as `longtable`s, or are
    parked if nothing in the text uses them.
- **Formatting:** minus signs in math mode (`$-0.4$`); "n/a" or "--" for empty cells, consistently; no vertical rules;
  booktabs only.

## 6. Prose rules

- **Voice and tense:** "We" and active voice. Present tense for what the paper does and finds; past tense for what
  others found.
- **Claims:** state a result, then its test: "...lowers QLIKE by \rh..\% (DM $= $ \rh.., $p = $ \rh..)."
- **Statistical vs economic significance:** keep them distinct ("statistically significant", "economically large").
- **Wording:** no code identifiers, file names or function names in body text; they may stay in `%` comments. No
  exclamation marks, no "huge", no "clearly"; prefer "substantially", "materially", "modestly".
- **Definitions:** define each symbol at its first use, in the same bullet, and never use a symbol before its
  definition.
