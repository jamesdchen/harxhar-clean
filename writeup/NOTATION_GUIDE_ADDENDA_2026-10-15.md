# Addenda to NOTATION_GUIDE_2026-10-15.md (2026-10-15)

- **Status:** companion to the core guide. The core guide wins wherever the two agree. Where they disagree, section B
  says so and gives a recommended fix. Items marked **[coordinator]** need a decision before the editors apply them.
- **Line numbers:** they refer to the files as they stood *before* the notation pass (rehaul/*.tex and the five
  generators are untracked, so there is no git anchor). Every item also quotes the text, so grep for the quote.
- **Contents:**
  - A. Predictor-name lookup (every series in the 16:00 design and the pooled panel). Delivered first.
  - B. Disagreements with the core guide, clashes it misses, symbols used before their definition.
  - C. Generator inventory, one self-contained section per generator, tables first.
  - D. References the guides rely on that `references.bib` lacks, with BibTeX.

---

## A. Predictor-name lookup

### A.1 The rule (apply mechanically)

Column names come in five patterns. `S` is the series symbol from table A.2, `k` the horizon in 30-minute intervals.

| Column-name pattern (regex) | Math name in tables | In prose |
|---|---|---|
| `^har_ma_(\d+)$` | `$\bar y^{(k)}$` | "HAR component at horizon $k$" (Corsi 2009) |
| `^har_ma_(\d+)_x_open$` | `$\bar y^{(k)}\cdot\mathrm{Open}$` | "HAR component times the opening-interval dummy" (pooled only) |
| `^har_ma_(\d+)_x_close$` | `$\bar y^{(k)}\cdot\mathrm{Close}$` | "HAR component times the closing-interval dummy" (pooled only) |
| `^adj_(.+)_ma_(\d+)$` | `$S^{(k)}$` | "$S$ averaged over the previous $k$ intervals" |
| `^(.+)_avail_ma_(\d+)$` | `$\bar a^{(k)}(S)$` | "share of the previous $k$ intervals on which $S$ was observed" (availability indicator $a_{t,i}$; 04_features.tex) |
| `^(.+)_active_ma_(\d+)$` | `$\bar\chi^{(k)}(S)$` | "share of the previous $k$ intervals on which $S$ was off its modal value" (activity indicator $\chi_{t,i}$; 04_features.tex) |
| bare series name (a predictor group in the variable-importance tables, e.g. `sumret`) | `$S$` | the series' name in words (A.2, last column) |
| calendar column (A.3) | its symbol in A.3 | its words in A.3 |

- **Superscripts stack** as `group, horizon`: `adj_sumret2_ewstock_ma_25` becomes `$\mathrm{RV}^{\mathrm{EW},(25)}$`, and
  `adj_sumpret2_vwstock_ma_1` becomes `$\mathrm{RS}^{+,\mathrm{VW},(1)}$`.
- **Time subscripts** ($_t$, $_{t,i}$) are dropped in tables and kept in formulas.
- **Horizons:** $k \in \{1, 5, 25, 125, 625, 3125\}$ intervals in the closing-interval design; $k \in \{1, 2, 4, \dots,
  2048\}$ intervals in the pooled panel. One trading day has 48 intervals.
- **Agrees with the finished Predictors and Data sections** (04_features.tex header and lines 52-57; 03_data.tex
  header and lines 16-41): indicator averages $\bar a^{(k)}_t(u)$, $\bar\chi^{(k)}_t(u)$; $\mathrm{BV}$,
  $\mathrm{AC}$, $\mathrm{OFI}^{\mathrm{EW/VW}}$, EW/VW superscripts; one-minute count $N_{t,i}$.
  - 03_data.tex:30-31 names the positive realized semivariance in words only. Give it the symbol
    $\mathrm{RS}^{+}_{t,i}$ there, so the tables can use it.
- **Never** typeset a code name (`\texttt{sumret}`, `har\_ma\_1`, `adj\_...`) in a table, caption or note.

### A.2 Series symbols: all 47 exogenous series

Every series below is in the 16:00 design (628 columns). The pooled panel (1,144 columns) has the 43 series without
the four FOMC ones. The activity column (`_active`) exists for the seven series marked "yes" in the design, and for
the five non-FOMC ones among them in the pooled panel.

| Code name | Symbol (LaTeX) | Words (first use) | Source of the convention | Activity col. |
|---|---|---|---|---|
| *ES return moments (E-mini S&P 500 futures, one-minute log returns $r_{t,i,k}$)* | | | | |
| `sumret` | `r` | interval log return, $r_{t,i} = \sum_k r_{t,i,k}$ | ABDL (2003) | no |
| `sumabsret` | `\mathrm{RAV}` | realized absolute variation, $\sum_k \lvert r_{t,i,k}\rvert$ | Forsberg and Ghysels (2007) | no |
| `sumret3` | `\mathrm{RSK}` | realized skewness, unnormalized: $\sum_k r_{t,i,k}^3$ | Amaya, Christoffersen, Jacobs and Vasquez (2015) (they divide by $\RV^{3/2}/\sqrt n$; say so) | no |
| `sumret4` | `\mathrm{RQ}` | realized quarticity, unscaled: $\sum_k r_{t,i,k}^4$ | Barndorff-Nielsen and Shephard (2002); Bollerslev, Patton and Quaedvlieg (2016) (they scale by $n/3$; say so) | no |
| `sumpret2` | `\mathrm{RS}^{+}` | positive realized semivariance, $\sum_k r_{t,i,k}^2\,\mathbf 1\{r_{t,i,k} > 0\}$ (assumed reading; see 03_data.tex:25-27) | Barndorff-Nielsen, Kinnebrock and Shephard (2010); Patton and Sheppard (2015) | no |
| `sumbipow` | `\mathrm{BV}` | bipower variation, $\sum_{k\ge2}\lvert r_{t,i,k}\rvert\lvert r_{t,i,k-1}\rvert$ (without the $\pi/2$ factor; say so) | Barndorff-Nielsen and Shephard (2004) | no |
| `sumautocov` | `\mathrm{AC}` | first-order realized autocovariance, $\sum_{k\ge2} r_{t,i,k} r_{t,i,k-1}$ (03_data.tex) | Hansen and Lunde (2006) | no |
| *Liquidity (14)* | | | | |
| `sumvolume` | `\mathrm{Volume}` | ES contract volume | | no |
| `numobs` | `N` | number of one-minute returns in the interval, $N_{t,i} \le 30$ (03_data.tex:16-17) | | **yes** |
| `turnover_ewstock` | `\mathrm{TO}^{\mathrm{EW}}` | equal-weighted constituent turnover | | no |
| `buyturnover_ewstock` | `\mathrm{TO}^{\mathrm{EW}}_{\mathrm{buy}}` | equal-weighted buyer-initiated turnover | | no |
| `sellturnover_ewstock` | `\mathrm{TO}^{\mathrm{EW}}_{\mathrm{sell}}` | equal-weighted seller-initiated turnover | | no |
| `effspread_ewstock` | `\mathrm{EffSpread}^{\mathrm{EW}}` | equal-weighted effective spread | | no |
| `spread_ewstock` | `\mathrm{Spread}^{\mathrm{EW}}` | equal-weighted quoted spread | core guide section 5 | no |
| `turnover_vwstock` | `\mathrm{TO}^{\mathrm{VW}}` | value-weighted constituent turnover | | no |
| `buyturnover_vwstock` | `\mathrm{TO}^{\mathrm{VW}}_{\mathrm{buy}}` | value-weighted buyer-initiated turnover | | no |
| `sellturnover_vwstock` | `\mathrm{TO}^{\mathrm{VW}}_{\mathrm{sell}}` | value-weighted seller-initiated turnover | | no |
| `effspread_vwstock` | `\mathrm{EffSpread}^{\mathrm{VW}}` | value-weighted effective spread | | no |
| `spread_vwstock` | `\mathrm{Spread}^{\mathrm{VW}}` | value-weighted quoted spread | core guide section 5 | no |
| `ofi_ewstock` | `\mathrm{OFI}^{\mathrm{EW}}` | equal-weighted order-flow imbalance, $(\mathrm{buy}-\mathrm{sell})/(\mathrm{buy}+\mathrm{sell})$ of turnover | core guide section 5 | no |
| `ofi_vwstock` | `\mathrm{OFI}^{\mathrm{VW}}` | value-weighted order-flow imbalance | core guide section 5 | no |
| *Market, equal-weighted constituents (6)* | | | | |
| `sumret2_ewstock` | `\mathrm{RV}^{\mathrm{EW}}` | equal-weighted realized variance | core guide section 5 | no |
| `sumret3_ewstock` | `\mathrm{RSK}^{\mathrm{EW}}` | equal-weighted realized skewness (unnormalized) | as `sumret3` | no |
| `sumret4_ewstock` | `\mathrm{RQ}^{\mathrm{EW}}` | equal-weighted realized quarticity (unscaled) | as `sumret4` | no |
| `sumabsret_ewstock` | `\mathrm{RAV}^{\mathrm{EW}}` | equal-weighted realized absolute variation | as `sumabsret` | no |
| `sumbipow_ewstock` | `\mathrm{BV}^{\mathrm{EW}}` | equal-weighted bipower variation | as `sumbipow` | no |
| `sumpret2_ewstock` | `\mathrm{RS}^{+,\mathrm{EW}}` | equal-weighted positive semivariance | as `sumpret2` | no |
| *Market, value-weighted constituents (6)* | | | | |
| `sumret2_vwstock` | `\mathrm{RV}^{\mathrm{VW}}` | value-weighted realized variance | core guide section 5 | no |
| `sumret3_vwstock` | `\mathrm{RSK}^{\mathrm{VW}}` | value-weighted realized skewness (unnormalized) | | no |
| `sumret4_vwstock` | `\mathrm{RQ}^{\mathrm{VW}}` | value-weighted realized quarticity (unscaled) | | no |
| `sumabsret_vwstock` | `\mathrm{RAV}^{\mathrm{VW}}` | value-weighted realized absolute variation | | no |
| `sumbipow_vwstock` | `\mathrm{BV}^{\mathrm{VW}}` | value-weighted bipower variation | | no |
| `sumpret2_vwstock` | `\mathrm{RS}^{+,\mathrm{VW}}` | value-weighted positive semivariance | | no |
| *Sentiment, StockTwits (3)* | | | | |
| `stocktwits_attention` | `\mathrm{Attn}` | StockTwits attention (message volume) | | no |
| `stocktwits_sentiment` | `\mathrm{Sent}` | StockTwits net sentiment score | | no |
| `stocktwits_sentcount` | `\mathrm{NSent}` | number of sentiment-tagged StockTwits messages | | no |
| *Implied volatility (3)* | | | | |
| `vix` | `\mathrm{VIX}` | VIX index | core guide section 5 | no |
| `vvix` | `\mathrm{VVIX}` | VVIX index | core guide section 5 | no |
| `vix3m` | `\mathrm{VIX3M}` | VIX3M index | core guide section 5 | no |
| *Volatility demand, Cboe open-close data (4)* | | | | |
| `voldemand_spx_open_and_close` | `\mathrm{VD}^{\mathrm{SPX}}_{\mathrm{OC}}` | net volatility demand in SPX options, opening and closing trades | Ni, Pan and Poteshman (2008) | **yes** |
| `voldemand_spx_open_only` | `\mathrm{VD}^{\mathrm{SPX}}_{\mathrm{O}}` | net volatility demand in SPX options, opening trades only | Ni, Pan and Poteshman (2008) | **yes** |
| `voldemand_all_open_and_close` | `\mathrm{VD}^{\mathrm{All}}_{\mathrm{OC}}` | net volatility demand in all index options, opening and closing trades | Ni, Pan and Poteshman (2008) | **yes** |
| `voldemand_all_open_only` | `\mathrm{VD}^{\mathrm{All}}_{\mathrm{O}}` | net volatility demand in all index options, opening trades only | Ni, Pan and Poteshman (2008) | **yes** |
| *FOMC calendar (4; 16:00 design only)* | | | | |
| `fomc_release` | `\mathrm{FOMC}^{\mathrm{rel}}` | FOMC statement-interval dummy (14:30 ET) | core guide section 5 | **yes** |
| `fomc_day` | `\mathrm{FOMC}^{\mathrm{day}}` | FOMC statement-day dummy | | **yes** |
| `fomc_until_inv` | `\mathrm{FOMC}^{\mathrm{next}}` | $1/(1 + n^{+}_{t,i})$, $n^{+}_{t,i}$ the intervals to the next statement | | no |
| `fomc_since_inv` | `\mathrm{FOMC}^{\mathrm{last}}` | $1/(1 + n^{-}_{t,i})$, $n^{-}_{t,i}$ the intervals since the last statement | | no |

Examples: `adj_sumabsret_ma_1` $\to$ `$\mathrm{RAV}^{(1)}$`; `adj_numobs_ma_25` $\to$ `$N^{(25)}$`;
`numobs_active_ma_5` $\to$ `$\bar\chi^{(5)}(N)$`; `vix_avail_ma_3125` $\to$ `$\bar a^{(3125)}(\mathrm{VIX})$`;
`adj_voldemand_spx_open_only_ma_125` $\to$ `$\mathrm{VD}^{\mathrm{SPX},(125)}_{\mathrm{O}}$`.

### A.3 HAR block, calendar dummies and probes

| Code name | Symbol (LaTeX) | Words | Note |
|---|---|---|---|
| `har_ma_k` (unit name "HAR ladder") | `\bar y^{(k)}`; the group: "HAR components" | HAR component of the adjusted target at horizon $k$ | Corsi (2009) |
| `adj_RV` (target) | `y` (fit) / `\tilde y` (winsorized) | periodicity-adjusted realized volatility | core guide section 2 |
| calendar block (unit name "Calendar") | group: "Calendar dummies" | | |
| `DOW_0` ... `DOW_4` | `\mathrm{Mon}`, `\mathrm{Tue}`, `\mathrm{Wed}`, `\mathrm{Thu}`, `\mathrm{Fri}` | day-of-week dummies | |
| `hour` | `\mathrm{Hour}` | clock hour of the interval end | constant at the closing interval |
| `is_overnight` | `\mathrm{Night}` | outside 09:30-16:00 ET | constant at the closing interval |
| `is_open` | `\mathrm{Open}` | opening-interval dummy (09:30-10:00) | constant (0) at the closing interval |
| `is_close` | `\mathrm{Close}` | close and after-hours dummy (16:00-19:30) | constant (1) at the closing interval |
| `is_opex` | `\mathrm{OpEx}` | monthly option-expiration day | |
| `is_opex_week` | `\mathrm{OpExWk}` | option-expiration week | |
| `is_quad_witch` | `\mathrm{QW}` | quadruple-witching day | |
| `is_rebalance_close` | `\mathrm{QW}\cdot\mathrm{Close}` | quadruple-witching closing auction (index rebalancing) | |
| `is_month_end` | `\mathrm{ME}` | last trading day of the month | |
| `is_quarter_end` | `\mathrm{QE}` | last trading day of the quarter | |
| `days_to_opex` | `\mathrm{DOpEx}` | signed trading days to the nearest monthly expiration, clipped | |
| noise probe, continuous | `\varepsilon^{\mathrm{U}}` | uniform noise column | variable-importance section only |
| noise probe, 20 levels | `\varepsilon^{(20)}` | 20-level discrete noise column | |
| noise probe, binary | `\varepsilon^{\mathrm{B}}` | Bernoulli(1/2) noise column | |

- **The four constant calendar columns** (Hour, Night, Open, Close) never vary at the closing interval, so the window
  screen drops them in every closing-interval fit. Say so wherever "16 calendar dummies" is counted (04_features.tex:40-44,
  `rehaul_dm_design`), or the reader will look for their coefficients.

### A.4 Drop-in dictionary for the generators

Paste into a generator; it covers every column of both designs.

```python
import re

SERIES_TEX = {
    "sumret": r"r", "sumabsret": r"\mathrm{RAV}", "sumret3": r"\mathrm{RSK}", "sumret4": r"\mathrm{RQ}",
    "sumpret2": r"\mathrm{RS}^{+}", "sumbipow": r"\mathrm{BV}", "sumautocov": r"\mathrm{AC}",
    "sumvolume": r"\mathrm{Volume}", "numobs": r"N",
    "turnover": r"\mathrm{TO}", "buyturnover": r"\mathrm{TO}_{\mathrm{buy}}", "sellturnover": r"\mathrm{TO}_{\mathrm{sell}}",
    "effspread": r"\mathrm{EffSpread}", "spread": r"\mathrm{Spread}", "ofi": r"\mathrm{OFI}",
    "sumret2": r"\mathrm{RV}",
    "stocktwits_attention": r"\mathrm{Attn}", "stocktwits_sentiment": r"\mathrm{Sent}",
    "stocktwits_sentcount": r"\mathrm{NSent}",
    "vix": r"\mathrm{VIX}", "vvix": r"\mathrm{VVIX}", "vix3m": r"\mathrm{VIX3M}",
    "voldemand_spx_open_and_close": r"\mathrm{VD}^{\mathrm{SPX}}_{\mathrm{OC}}",
    "voldemand_spx_open_only": r"\mathrm{VD}^{\mathrm{SPX}}_{\mathrm{O}}",
    "voldemand_all_open_and_close": r"\mathrm{VD}^{\mathrm{All}}_{\mathrm{OC}}",
    "voldemand_all_open_only": r"\mathrm{VD}^{\mathrm{All}}_{\mathrm{O}}",
    "fomc_release": r"\mathrm{FOMC}^{\mathrm{rel}}", "fomc_day": r"\mathrm{FOMC}^{\mathrm{day}}",
    "fomc_until_inv": r"\mathrm{FOMC}^{\mathrm{next}}", "fomc_since_inv": r"\mathrm{FOMC}^{\mathrm{last}}",
}
CAL_TEX = {
    "DOW_0": r"\mathrm{Mon}", "DOW_1": r"\mathrm{Tue}", "DOW_2": r"\mathrm{Wed}", "DOW_3": r"\mathrm{Thu}",
    "DOW_4": r"\mathrm{Fri}", "hour": r"\mathrm{Hour}", "is_overnight": r"\mathrm{Night}", "is_open": r"\mathrm{Open}",
    "is_close": r"\mathrm{Close}", "is_opex": r"\mathrm{OpEx}", "is_opex_week": r"\mathrm{OpExWk}",
    "is_quad_witch": r"\mathrm{QW}", "is_rebalance_close": r"\mathrm{QW}\cdot\mathrm{Close}",
    "is_month_end": r"\mathrm{ME}", "is_quarter_end": r"\mathrm{QE}", "days_to_opex": r"\mathrm{DOpEx}",
}
GROUP_TEX = {"ewstock": r"\mathrm{EW}", "vwstock": r"\mathrm{VW}"}
GROUP_WORDS = {"har": "HAR components", "calendar": "Calendar dummies"}  # variable-importance group names


def _sup(base: str, extra: str) -> str:
    """Append `extra` to the superscript of `base` (brace-matched): RS^{+} -> RS^{+,EW}; TO_{buy} -> TO^{EW}_{buy}."""
    i = base.find("^{")
    if i >= 0:
        depth = 0
        for j in range(i + 1, len(base)):
            depth += {"{": 1, "}": -1}.get(base[j], 0)
            if depth == 0:
                return base[:j] + "," + extra + base[j:]
    i = base.find("_{")
    if i >= 0:
        return base[:i] + "^{" + extra + "}" + base[i:]
    return base + "^{" + extra + "}"


def series_tex(s: str) -> str:
    """Math name (no $) of a bare series code name, e.g. 'sumret2_ewstock' -> RV^{EW}."""
    if s in SERIES_TEX:
        return SERIES_TEX[s]
    stem, _, grp = s.rpartition("_")
    return _sup(SERIES_TEX[stem], GROUP_TEX[grp])


def column_tex(c: str) -> str:
    """Math name (with $) of any design column of the 16:00 design or the pooled panel."""
    if c in CAL_TEX:
        return f"${CAL_TEX[c]}$"
    if m := re.fullmatch(r"har_ma_(\d+)", c):
        return rf"$\bar y^{{({m.group(1)})}}$"
    if m := re.fullmatch(r"har_ma_(\d+)_x_(open|close)", c):
        return rf"$\bar y^{{({m.group(1)})}}\cdot\mathrm{{{m.group(2).title()}}}$"
    if m := re.fullmatch(r"adj_(.+)_ma_(\d+)", c):
        return f"${_sup(series_tex(m.group(1)), '(' + m.group(2) + ')')}$"
    if m := re.fullmatch(r"(.+)_(avail|active)_ma_(\d+)", c):
        f = r"\bar a" if m.group(2) == "avail" else r"\bar\chi"
        return f"${f}^{{({m.group(3)})}}({series_tex(m.group(1))})$"
    raise KeyError(c)
```

Spot checks of `column_tex`: `adj_sumret2_ewstock_ma_25` $\to$ `$\mathrm{RV}^{\mathrm{EW},(25)}$`;
`adj_sumpret2_vwstock_ma_1` $\to$ `$\mathrm{RS}^{+,\mathrm{VW},(1)}$`; `adj_buyturnover_ewstock_ma_5` $\to$
`$\mathrm{TO}^{\mathrm{EW},(5)}_{\mathrm{buy}}$`; `adj_sumautocov_ma_5` $\to$ `$\mathrm{AC}^{(5)}$`;
`adj_voldemand_spx_open_only_ma_125` $\to$ `$\mathrm{VD}^{\mathrm{SPX},(125)}_{\mathrm{O}}$`. Run it over both name
lists (`design_bar1600_all_features.npz`, `prep_cache_all_features_b2.npz`) and assert no `KeyError` before relying on
it.

---

## B. Disagreements with the core guide, missed clashes, use before definition

(filled in below)
