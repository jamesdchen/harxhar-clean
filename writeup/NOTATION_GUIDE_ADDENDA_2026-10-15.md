# Addenda to NOTATION_GUIDE_2026-10-15.md (2026-10-15)

- **Status:** companion to the core guide; the core guide wins where the two agree. Where they disagree, section B
  says so and gives a recommended fix with a priority (H, M, L). Section C.0 collects the cross-generator decisions
  and the items only the user can settle.
- **Line numbers:** section B quotes the section files *after* the editors' notation pass; section C quotes each
  generator as it stood when its inventory was written (14:30-14:56). The files are untracked, so there is no git
  anchor: every item also quotes the text, so grep for the quote.
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
- **Time subscript:** keep $_t$, as 04_features.tex:53-58 defines the names ($\mathrm{VIX}^{(25)}_t$,
  $\bar a^{(k)}_t(u)$, $\bar\chi^{(k)}_t(u)$); drop only the interval index ($i = c$). A symbol that already has a
  subscript takes $t$ inside it: $\mathrm{TO}^{\mathrm{EW},(5)}_{\mathrm{buy},t}$,
  $\mathrm{VD}^{\mathrm{SPX},(125)}_{\mathrm{O},t}$. The A.1 and A.2 examples omit $_t$ for brevity;
  `column_tex(c)` (A.4) adds it, and `column_tex(c, t=False)` leaves it off.
- **Horizons:** $k \in \{1, 5, 25, 125, 625, 3125\}$ intervals in the closing-interval design; $k \in \{1, 2, 4, \dots,
  2048\}$ intervals in the pooled panel. One trading day has 48 intervals.
- **Agrees with the finished Predictors and Data sections** (04_features.tex header and lines 52-57; 03_data.tex
  header and lines 16-41): indicator averages $\bar a^{(k)}_t(u)$, $\bar\chi^{(k)}_t(u)$; $\mathrm{BV}$,
  $\mathrm{AC}$, $\mathrm{OFI}^{\mathrm{EW/VW}}$, EW/VW superscripts; one-minute count $N_{t,i}$.
  - 03_data.tex:30-31 names the positive realized semivariance in words only. Give it the symbol
    $\mathrm{RS}^{+}_{t,i}$ there, so the tables can use it.
- **Row labels in variable-importance tables:** "words (symbol)" from A.2. Where the width binds, use the short words of section C.5 (it measured the widths).
- **Never** typeset a code name (`\texttt{sumret}`, `har\_ma\_1`, `adj\_...`) in a table, caption or note.

### A.2 Series symbols: all 47 exogenous series

> **2026-10-19 (user): the draft prints raw names for now.** `paper_rehaul.tex` (and `fi_bullets_standalone.tex`) set
> `\rawseriestrue` in the preamble, so every series below prints as its code name in typewriter type (the first
> column: the parquet column of `data/*.parquet`, or for `ofi_*` and `fomc_*`, the column the loader builds), not
> as its symbol. The names go through `\ser{code name}{symbol}`, or `\serx{code name}{raw superscript/subscript}{symbol}`
> when the horizon stacks into the symbol's superscript (`make_rehaul_data_methods.py` and `make_rehaul_fi.py`,
> `series_tex` / `column_tex`; `03_data.tex` and `04_features.tex` by hand). **To switch back,** change
> `\rawseriestrue` to `\rawseriesfalse` in both preambles and rebuild: the symbols of this table return unchanged.
> The figures (matplotlib) keep the symbols either way. Section A.1's rule "never typeset a code name" is suspended
> while the switch is on.

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
| `hour` | `\mathrm{Hour}` | clock hour of the interval's end stamp | constant at the closing interval |
| `is_overnight` | `\mathrm{Night}` | intervals ending before 09:00 or from 16:00 ET, and weekends (`src/features/extractors/calendar.py:27`) | constant at the closing interval |
| `is_open` | `\mathrm{Open}` | opening dummy: intervals ending 09:00 and 09:30 ET (clock hour 9) | constant at the closing interval |
| `is_close` | `\mathrm{Close}` | close and after-hours dummy: intervals ending 16:00 to 19:30 ET | constant at the closing interval |
| `is_opex` | `\mathrm{OpEx}` | monthly option-expiration day | |
| `is_opex_week` | `\mathrm{OpExWk}` | option-expiration week | |
| `is_quad_witch` | `\mathrm{QW}` | quadruple-witching day | |
| `is_rebalance_close` | `\mathrm{QW}\cdot\mathrm{Close}` | quadruple-witching day times the close dummy (index-rebalancing auction) | equals $\mathrm{QW}$ at the closing interval, so the screen drops one of the two |
| `is_month_end` | `\mathrm{ME}` | last trading day of the month | |
| `is_quarter_end` | `\mathrm{QE}` | last trading day of the quarter | |
| `days_to_opex` | `\mathrm{DOpEx}` | signed trading days to the nearest monthly expiration, clipped | |
| noise probe, continuous | `\varepsilon^{\mathrm{U}}` | uniform noise column | variable-importance section only |
| noise probe, 20 levels | `\varepsilon^{(20)}` | 20-level discrete noise column | |
| noise probe, binary | `\varepsilon^{\mathrm{B}}` | Bernoulli(1/2) noise column | |

- **Five calendar columns never enter a closing-interval fit:** Hour, Night, Open and Close are constant at the
  closing interval, and $\mathrm{QW}\cdot\mathrm{Close}$ is an exact copy of $\mathrm{QW}$ there, so the window screen
  ($\mathcal J_t$) drops all five at every fit. Verified on `design_bar1600_all_features.npz`: the four columns are
  constant over all 3,469 rows, and the two quadruple-witching columns are equal row for row. Say so wherever "16 calendar dummies" is counted (04_features.tex:40-44,
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


def _sub_t(base: str) -> str:
    """Add the day subscript t: merges into a trailing subscript (TO^{EW}_{buy} -> TO^{EW}_{buy,t}), else appends _t."""
    if base.endswith("}"):
        depth, j = 0, len(base) - 1
        while j >= 0:
            depth += {"}": 1, "{": -1}.get(base[j], 0)
            if depth == 0:
                break
            j -= 1
        if j >= 1 and base[j - 1] == "_":  # the last group is a subscript
            return base[:-1] + ",t}"
    return base + "_t"


def column_tex(c: str, t: bool = True) -> str:
    """Math name (with $) of any column of the 16:00 design or the pooled panel; t adds the day subscript."""
    T = _sub_t if t else (lambda base: base)
    if c == "is_rebalance_close":
        return "$" + T(r"\mathrm{QW}") + r"\cdot " + T(r"\mathrm{Close}") + "$"
    if c in CAL_TEX:
        return "$" + T(CAL_TEX[c]) + "$"
    if m := re.fullmatch(r"har_ma_(\d+)", c):
        return "$" + T(r"\bar y^{(" + m.group(1) + ")}") + "$"
    if m := re.fullmatch(r"har_ma_(\d+)_x_(open|close)", c):
        edge = r"\mathrm{" + m.group(2).title() + "}"
        return "$" + T(r"\bar y^{(" + m.group(1) + ")}") + r"\cdot " + T(edge) + "$"
    if m := re.fullmatch(r"adj_(.+)_ma_(\d+)", c):
        return "$" + T(_sup(series_tex(m.group(1)), "(" + m.group(2) + ")")) + "$"
    if m := re.fullmatch(r"(.+)_(avail|active)_ma_(\d+)", c):
        f = r"\bar a" if m.group(2) == "avail" else r"\bar\chi"
        return "$" + T(f + "^{(" + m.group(3) + ")}") + "(" + series_tex(m.group(1)) + ")$"
    raise KeyError(c)
```

Spot checks of `column_tex(c, t=False)`: `adj_sumret2_ewstock_ma_25` $\to$ `$\mathrm{RV}^{\mathrm{EW},(25)}$`;
`adj_sumpret2_vwstock_ma_1` $\to$ `$\mathrm{RS}^{+,\mathrm{VW},(1)}$`; `adj_buyturnover_ewstock_ma_5` $\to$
`$\mathrm{TO}^{\mathrm{EW},(5)}_{\mathrm{buy}}$`; `adj_sumautocov_ma_5` $\to$ `$\mathrm{AC}^{(5)}$`;
`adj_voldemand_spx_open_only_ma_125` $\to$ `$\mathrm{VD}^{\mathrm{SPX},(125)}_{\mathrm{O}}$`. Run it over both name
lists (`design_bar1600_all_features.npz`, `prep_cache_all_features_b2.npz`) and assert no `KeyError` before relying on
it.

---

## B. Disagreements with the core guide, missed clashes, use before definition

Audited against the section files **after** the editors' pass (read at about the time this section was written).
"Line" is the current line in `writeup/rehaul/`; the quote is what to grep for. Priority: **H** fix before
circulating, **M** fix in this pass if cheap, **L** optional.

### B.1 Clashes the core guide creates or leaves

| # | Where (file:line, quote) | Problem | Recommended fix | Pri. |
|---|---|---|---|---|
| 1 | 05_methods:438-452 "lag truncation $q$"; 06_results:71, 120, 129, 157 "$q = \fqsHACLag$"; 07_economics:122, 133; 10_appendix_floats:48, 51, 227-228, 303 | Core section 4 renames the Newey-West lag to $q$, but $q_t$ is the position (01:35, 07 throughout, 09:28, every economics table). 07:133 puts both in one sentence. | Keep $q$ only inside the Methods formulas (no position there). Everywhere else, and in every table note, write it in words: "Newey--West variance with \fqsDMLag{} lags" / "with \fqsHACLag{} lags", no symbol. | H |
| 2 | 05_methods:444, 452 "which gives $q = 6$", "$q = 14$" | Typed numbers that have macros. | `\fqsDMLag`, `\fqsHACLag` (generated/fig_qlike_sharpe_numbers.tex:36-37). | H |
| 3 | 05_methods:139, 146 "$\bm\delta^{\top} \bm d_t$"; 04_features:60 "$\bm d_{t,i}$" vs 05_methods:420 "$d_t = L_t^{A} - L_t^{B}$" | Core section 3 gives $\bm d_t$ to the calendar dummies while section 4 keeps $d_t$ for the DM loss differential; both appear in Methods. | Calendar dummies $\bm D_{t,i}$ (the usual dummy-variable letter), coefficients $\bm\delta$. If the coordinator prefers to keep $\bm d$, accept it as a bold-vector/scalar distinction and say so in the Methods notation bullet. | M |
| 4 | 05_methods:202 "$\gamma J$" vs 05_methods:440-441 "$\hat\gamma_k$", 06_results:63-67 "$\hat\gamma_j$" | Core section 3 reserves $\gamma$ for the tree split penalty, but the autocovariances $\hat\gamma_j$ (the stronger convention, Newey and West 1987) also use it, in the same section. | Keep $\hat\gamma_l$ for autocovariances; write the leaf penalty $\gamma^{\mathrm{tree}}$, matching $\lambda_1^{\mathrm{tree}}, \lambda_2^{\mathrm{tree}}$. Use one lag index in both formulas (05 uses $k$, 06 uses $j$; $k$ and $j$ are taken): $l$. | M |
| 5 | Core section 5 row "avail, `_active`, `_avail` -> missing-data (availability) indicator" | Wrong for `_active`: the activity indicator is $\mathbf 1\{u_{t,i} \ne \operatorname{mode}(u)\}$ (zero-inflation / hurdle encoding), not missingness. 04_features already gets it right ($\chi_{t,i}$). | Amend the core row: `_avail` -> "availability indicator $a_{t,i}$"; `_active` -> "activity indicator $\chi_{t,i}$". Generators must not label activity columns "missing-data". | H |
| 6 | Core section 5 "straddle trade -> delta-neutral at-the-money straddle"; 01_introduction:29 "delta-neutral and at the money" | Inaccurate twice: when the index lies between strikes the position is a one-strike strangle ($K_C = \min\{K \ge S\}$, $K_P = \max\{K \le S\}$, 07_economics), and it is never hedged; its delta is near zero only at entry. | "an at-the-money straddle (a one-strike strangle when the index lies between strikes), opened at 15:30 and held unhedged to the 16:00 settlement". | H |
| 7 | Core 6b "$\mathcal I$ is the set of retained intervals" vs 01:61-71, 05:105-107, 429-431, 06:115-117 "$f_{m,\mathcal X,\mathcal I}$, $\mathcal I \in \{\{c\},\{1,\dots,M\}\}$" | Two meanings for $\mathcal I$. 03_data does not actually use $\mathcal I$ (grep). | Drop the 6b sentence (or rename the retained-interval set $\mathcal G$, "the interval grid"); keep $\mathcal I$ as the estimation-sample index. | L |
| 8 | 02_related_work:15-16 "$\RV^{(k)}_t$ ... $k \in \{1,5,22\}$ trading days" vs 04/05 "$\bar y^{(k)}$, $k$ in 30-minute intervals" | The superscript $(k)$ carries two units (days in Related work, intervals elsewhere). | In Related work use Corsi's own $\RV^{(d)}_t, \RV^{(w)}_t, \RV^{(m)}_t$: $\E_t[\RV_{t+1}] = \beta_0 + \beta_d \RV^{(d)}_t + \beta_w \RV^{(w)}_t + \beta_m \RV^{(m)}_t$. | M |
| 9 | 08_feature_importance:162-163, 206-209 "$\Phi_{t,AB}$" vs 05_methods:376-377 "$\Phi$: the standard normal distribution function" | $\Phi$ has two meanings. | $\phi_{t,AB}$ (core 6b reserves $\phi$ for SHAP; it extends $\phi_{t,A}$, $\phi_{t,jl}$). | M |
| 10 | 07_economics:140, 207 "$\mathrm{ES}_t$", "$\overline{\mathrm{ES}}$"; 10_appendix_floats:390 "$\mathrm{ES}_\theta$" vs 03_data:13 "E-mini S\&P 500 futures (ES)" | ES is the futures ticker in Data and reads as expected shortfall to a finance referee. | Ehm et al. (2016)'s own notation $S_\theta(\hat\varrho_t, \varrho_t)$, mean $\bar S_\theta$; "elementary score" in words. | M |
| 11 | 10_appendix_floats:392 "$q_t = \sign(s_t)$" | Leftover $s_t$ (core 6b: no $s_t$). | $q_t = \sign(\hat\sigma^2_t - \mathrm{IV}^2_t)$. | H |
| 12 | 09_conclusion:42-43 "$\sum_{s \in \mathcal W^{\mathrm{est}}_t}$ ... $s$ the trading days" | $s$ as the window row index (core 6b: $t'$). | $t'$. | M |
| 13 | 05_methods:34, 140 (lag $j$), 158-185 (predictor $j$), 207-211 (leaf $j$: $w^\ast_j, G_j, H_j, n_j, \mathcal R_j$), 361-364 (fold $j$), 391-404 (seed $j$) | $j$ has five meanings in Methods. | Keep $j$ for predictors and for the interval lag $(t, i-j)$ (both fixed by 6b). Leaf index $l$ ($w^\ast_l, G_l, H_l, n_l, \mathcal R_l$); validation fold $v = 1, \dots, 4$; seed index $n = 1, \dots, S$ ($\hat y_t^{(n)}$, $e^\ast_n$). | M |
| 14 | 05_methods:212-215 "average of $N$ trees $\mathcal T_b$" vs 05_methods:31 "$N_{t,i} \le 30$" | $N$ twice in one section. | $B$ trees, $b = 1, \dots, B$ (Hastie, Tibshirani and Friedman 2009, Algorithm 15.1); the boosting bullet already uses $b$. | L |
| 15 | 05_methods:149 "the HAR components $x^{(k)}_t$ ... of each additional series" vs 04_features:30-35 "$\bar u^{(k)}_{t,i}$" | Two names for the same object; $x^{(k)}_t$ also reads as an element of $\bm x_t$. | $\bar u^{(k)}_t$, as in Predictors. | M |
| 16 | 05_methods:284-287 "$\bm W \in \mathbb R^{R \times C}$", "$\sqrt{\max(1, R/C)}$" | $R$ is the straddle return, $\bm W$ sits beside the window length $W$ (05:103), $\bm M$ beside $M$ intervals (05:28). | $\bm W \in \mathbb R^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$, $\sqrt{\max(1, d_{\mathrm{out}}/d_{\mathrm{in}})}$; momentum $\bm M$ may stay (bold, local). | L |
| 17 | 05_methods:420-422, 06_results:71 "$h = 1$", "$\max(h-1, \cdot)$" | $h$ is also the GRU state $\bm h_\tau$ and the boosting Hessian $h_{t'}$ in the same section. | Substitute $h = 1$: HLN factor $\sqrt{(|\mathcal O| - 1)/|\mathcal O|}$ and $q = \lfloor 4(|\mathcal O|/100)^{2/9} \rfloor$; say "one-step-ahead forecasts" in words. | L |
| 18 | 07_economics:153-155 "$1 + f\,q_t R_t$" and "$1 + f\,r_t$" in one display | Same quantity written two ways. | $r_t$ in both. | L |
| 19 | 04_features:78-81 "$\rho_{t,i}$" (band centre); 07_economics:205-207 "$\rho_S$"; 08_feature_importance_floats cluster "$|\rho|$" | Three uses of $\rho$, in different sections. | Acceptable as local symbols; if touched, rename the band centre $\bar s^{(u)}_{t,i}$ (it is a median of series-specific periodicity factors). | L |
| 20 | 07_economics header comment "kappa_0, kappa_1 the MinGRU output link" vs 05_methods:251 "$\varpi_0$, $\varpi_1$" | Stale comment. | Update the comment to $\varpi_0, \varpi_1$. | L |

### B.2 Used before defined, or never defined

| # | Where | Problem | Fix | Pri. |
|---|---|---|---|---|
| 21 | 01_introduction:24 "$\sum_{k=1}^{30} r_{t,c,k}^2$" | Upper limit 30 contradicts Data and Methods ($N_{t,i} \le 30$ varies). | $\sum_{k=1}^{N_{t,c}}$, "with $N_{t,c} \le 30$". | H |
| 22 | 05_methods:157-161 (ridge, lasso, enet over $\mathcal W_t$) vs 05_methods:120-124 (estimation objective over $\mathcal W^{\mathrm{est}}_t$) | Which window the linear fit uses is stated two ways (the editor flagged it, math unchanged). | Decide from the code, then use one set; the forecast-time refit uses $\mathcal W_t$, the tuning fits $\mathcal W^{\mathrm{est}}_t$. Say so in one sentence. | M |
| 23 | 05_methods:365 "a light re-tune at each block" | "block" is used before it is defined (it is the 250-day re-tuning interval, defined later at 05:395). | "at each re-tuning date (every 250 trading days)". | M |
| 24 | 08_feature_importance (Kendall's $W$ quoted at ~92 and 123, defined at ~138) | Used before its definition. | Define $W$ at first use, or write "Kendall's $W$ (defined below)". | L |
| 25 | 07_economics:207 "$\rho_S(\overline{\mathrm{ES}}, \SR) = -0.50$" | Typed number; the generated macro `\hmurRhoESX` has the opposite sign convention ($-$ES vs Sharpe) and is not loaded. | Add a macro with this sign convention to the Murphy generator (for example `\hmurRhoESLossX` = correlation of mean ES with Sharpe), or write $\rho_S(-\bar S_1, \SR) = \hmurRhoESX$ and load `generated/appendix_close_murphy_splits.tex`. Also note that these correlations were computed on the winsorized target (scoring caveat). | H |
| 26 | Typed results in prose: 06_results:124 "12--14\%", "about $-4.5$"; 06:137 "3.8\%"; 06:148 "$+12$--$13\%$", "about 3.4"; 06:172-173 "0.0914 ... 0.0948", "3.4\%", "$-1.22$"; 06:180 "0.121--0.124"; 07:150 "$f = 0.03$ (3\%)"; 07:181 "1.88"; 07:187 "$+0.55$", "$+0.50$"; 07:226 "4.4", "1.8" | House rule: numbers come from macros. These survived the pass. | Replace with the existing `\rhr...`, `\rhe...`, `\rhapp...` (`\rhappCompF` = 3) or `\cm...` macros where one exists; otherwise add them to the generator's numbers file. Several (the QLIKE levels) are in `rehaul_results_t*` but have no macro yet. | H |
| 27 | Plain-text citations: 05_methods:197 "Friedman 2001"; 05:212 "Breiman 2001"; 05:223-224 "Feng et al.\ 2024", "Jordan et al.\ 2024"; 05:438 "Newey and West 1987"; 05:454-455 "Hansen 2005", "Romano and Wolf 2005"; 06_results:61 "Newey and West (1987)"; 07_economics:134 "Ledoit and Wolf 2008" | Not `\citet`/`\citep`, because the keys are missing. | Add the BibTeX of section D, then `\citet{...}`. | M |
| 28 | 02_related_work:65 "$\bm y - X\bm\beta$" | $X$ not bold (05_methods:180 uses $\bm X_t$). | $\bm X$. | L |

### B.3 Language

| # | Where | Problem | Fix |
|---|---|---|---|
| 29 | 99_author_notes:11, 15 "per-bar OLS", "trade days" | Old terms (author notes, dropped before circulating). | "closing-interval OLS", "trading days", if the notes are kept. |
| 30 | 08_feature_importance_floats:144, 148 "opening- and closing-session dummies", "closing-session interactions" | "Session" for the opening/closing-interval dummies. | "opening- and closing-interval dummies" (A.3: Open, Close). |
| 31 | 10_appendix_floats:171 "shipped: the share of tuning dates ..." | Leftover "shipped". | "Fixed: the share of tuning dates at which the best trial is the fixed-hyperparameter configuration". |
| 32 | Core section 5 "HAR benchmark" | "Benchmark" also names the reference model (the closing-interval OLS on the HAR benchmark) and the short-straddle strategy. | Keep "HAR benchmark" for the predictor set only; call the OLS "the reference forecast" and the strategy "the unconditional short-straddle strategy" (07 already uses "benchmark" for it: pick one and keep it). |

### B.4 Scoring caveat (user decision 2026-10-15: QLIKE against the actual RV)

- Methods (05:64, 81-93) and Results now describe scoring against the actual $\RV_t$. Until the scoring agent re-runs,
  every generated number and these table notes still use the winsorized target:
  - `make_rehaul_results.py`: the T1, T2 and T3 notes ("$RV_t$ the winsorised 15:30--16:00 realised variance").
  - The Murphy correlations (`\hmurRho...`) and the elementary scores (07_economics:145-147 already says so).
- Change those strings in the same commit as the re-scored numbers, not before (section C repeats this per
  generator).

---

## C. Generator inventory (tables first)

- **One self-contained section per generator** (C.1-C.5). Each lists, for every generated table: its label, the
  problems against core section 8, a proposed layout (caption, columns, panels, notes, what moves or is parked), and
  line-numbered string edits; then the kept figures, the macro values with visible text, and open questions.
- **Shared by all five:** sections A (names), B.4 (scoring caveat) and D (references) of this file; core sections 5,
  6b, 7 and 8. Quotes are exact, so grep for them; line numbers are those of each generator when its section was
  written (14:30-14:56).
- **Concurrent edits:** a "table legibility" pass (`TABLE_FIXES_FROM_FLOATS_2026-10-15.md`) rewrote
  `make_rehaul_economics.py`, `make_rehaul_data_methods.py` and `make_rehaul_fi.py` while C.2, C.3 and C.5 were being
  written. Each of those sections audits the rewritten version and opens with a fixed / open status table. Keep one
  editor per generator from here on, so no one restores parked code.

### C.0 Decisions that cut across the generators (apply the same way in all five)

| # | Issue | Decision (default) |
|---|---|---|
| 1 | Stars on the estimate (C.2) or on the statistic (C.1, C.4) | **On the test statistic** (DM or HAC $t$, in parentheses) whenever the table has one; on the estimate only when no statistic is shown. Same everywhere. |
| 2 | Lag in notes | Words, no symbol: "Newey--West variance with \fqsDMLag{} lags" (DM), "with \fqsHACLag{} lags" (HAC $t$). Use these two existing macros; do not add generator-specific copies such as `\rheHACLag` unless the generator also asserts they are equal. |
| 3 | "Always short" | Row label "Short straddle ($q_t \equiv -1$)"; prose "the unconditional short-straddle benchmark". Also in the kept figure `fig:rh_econ_cumpnl` (legend, axis label, caption). |
| 4 | Sizing-rule names | Sign, Clipped, Asymmetric, Rank in tables (E4) and in 07_economics prose (which still says "robust clip", "full long, scaled short"). |
| 5 | Significance level words | "significant at the 5\% level"; notes state "$^{*}$, $^{**}$, $^{***}$: significant at the 10\%, 5\%, 1\% levels". |
| 6 | Days | Headline tables: "$T_{\mathrm{oos}} = $ macro trading days, 3 January 2020 to 30 April 2024". Tuning tables count tuning dates from 25 June 2018 (`\rhappOptPoints`), so their notes must name that period (C.4). |
| 7 | Missing section labels | Add `\label`s (additions, not renames) for Data, Predictors, Variable importance and the trading-strategy subsection of 07 (`sec:rh_data`, `sec:rh_predictors`, `sec:rh_vi`, `sec:rh_econ_strategy`), so notes can say "see Section~\ref{...}". Methods labels already exist (`sec:rh_methods_*`). |
| 8 | Re-run order | `make_table_close_main.py` and `make_appendix_campaign_16h_tex.py` before `make_rehaul_appendix.py` (C.4: it copies their cells and currently fails its re-scoring asserts); then every `make_rehaul_*.py`. Every `generated/rehaul_*` file read for this inventory was stale relative to its generator. |
| 9 | Time subscript in predictor names | Keep $_t$ (A.1, 04_features.tex:53-58). |
| 10 | `^\ast` used for "not in the model set" (C.1, longtable) | Becomes $^\ddagger$, since stars now mean significance. $^\dagger$ stays "preliminary". |

### C.0b Open items for the user (collected from C.1-C.5)

- **`[vendor]`:** the data vendors are not named anywhere in the repo (C.3); the placeholders stay until the user
  supplies them.
- **Earlier-draft forecasts:** nine pooled forecasts from the earlier draft sit in the 112-forecast model set and feed
  `\rheModelSetN`, `\rheTailNeg`, `\rheSigVsShort` and the E4 model-set rows (C.2), and rows of the full ranking (C.1,
  Panel C). Describe them in one sentence, or drop them from the set.
- **Appendix tables nobody cites (C.4):** park `tab:rh_app_tune_ladder`, `tab:rh_app_mingru_all`,
  `tab:rh_app_mingru_dayclass`, `tab:rh_app_fi_harma` (MDI is not a measure of the paper); merge mask into cadence and
  out-of-sample budget into tuning frequency. Needs the user's yes.
- **Duplicated Sharpe columns:** T1 (`tab:rh_axis1`) and E2 (`tab:rh_econ_allfeat`) print the same $\Delta\SR$ and
  $t$; C.1 recommends dropping them from T1, but E2 lacks some rows (specific trees, pooled trees, pooled MinGRU).
- **Random-forest depth grid:** {3, 5, 10, 20, 40, none} in the picks table vs {2, 4, 8, 16, 32, none} in
  `tab:rh_dm_hyper_search` (C.4); one of them is wrong.
- **"trial 255 of 400":** confirm it is the best trial of the forest's development search (C.3).

### C.1 `make_rehaul_results.py`

Writes: `generated/rehaul_results_t1_axis1.tex` (tab:rh_axis1), `generated/rehaul_results_t2_pooled.tex` (tab:rh_axis2),
`generated/rehaul_results_t3_ranking.tex` (tab:rh_rank), `generated/rehaul_results_t3_full.tex` (tab:rh_rank_full,
longtable), `generated/rehaul_results_numbers.tex` (`\rhr...` macros); figures `figures/rehaul_results_f{1_forest,
2_dumbbell,3_cumulative,5_se_share}.pdf` (+ .png). Wrappers: `rehaul/06_results_floats.tex` `\input`s all four tables
(lines 21, 48, 72, 146) and the numbers file (line 17). Table captions and notes all live in the generator. The figure
captions are typed in the wrapper: F3 at lines 85-93; F1, F2 and F5 are parked there.

**Read first: the generated files are stale.** All five `generated/rehaul_results_*.tex` files and the F3 PDF date
from 02:02. The generator was last edited at 14:16 and has since moved to scoring against the actual RV (lines 163, 271,
309, 327 use `rv_unclipped`).
- The files on disk still print the trimmed-target numbers (OLS QLIKE 0.0946; the generator now asserts 0.0967 at
  line 1265) and the "winsorised" note wording.
- `rehaul_results_numbers.tex` has 26 macros. The generator now writes 52 (the second block, lines 1695-1739, is
  missing from the file).
- The editor must re-run `python writeup/make_rehaul_results.py` after the edits. The wording change and the rescored
  numbers then land together (scoring caveat satisfied).

Shared helpers used by every table:

| Line | Current | Replacement |
|---|---|---|
| 482-486 | `num(x, nd, bold=False, sign=True)` wraps in `\bm{...}` when `bold` | Drop the `bold` parameter. Add `stars(p)` -> `"***"` if p<0.01, `"**"` if p<0.05, `"*"` if p<0.10, else `""`, and `stat(x, p, nd=2)` -> `f"${s}^{{{st}\\phantom{{{'*'*(3-len(st))}}}}}$"`. The phantom pads every cell to three stars, so the decimals stay aligned in `r` columns. Optional: drop the leading `+` (finance tables usually omit it); if so, do it in every column. |
| 489-494 | `pval()` (bold `p`, `{<}0.001`, the 0.0497 special case) | Unused once the $p$ columns go; delete it or leave it uncalled. Keep computing `p` and `p_t` in `pair()` (lines 338, 345): the asserts and the stars need them. |
| 507-513 | `NOTE_CONV`: "QLIKE is the mean over the {days} trade days (2020-01-03 to 2024-04-30, early closes excluded) of the daily loss $L_t = RV_t/\hat\sigma^2_t - \log(RV_t/\hat\sigma^2_t) - 1$, with the plain forecast $\hat\sigma^2_t = \hat y_t^2 B_t$ (no recalibration) and $RV_t$ the actual (not winsorised) 15:30--16:00 realised variance; the models are fitted on the winsorised target." (line 510 holds the parked "winsorised" version) | "The out-of-sample period is 3 January 2020 to 30 April 2024 ($T_{\mathrm{oos}} = {days}$ trading days). QLIKE is the average loss $\bar L$, with $L_t = \QLIKE(\RV_t, \hat\sigma^2_t)$, $\RV_t$ the realized variance of the closing half hour and $\hat\sigma^2_t = \hat y_t^{\,2} s^2_{t,c}$ the variance forecast." **Scoring caveat:** valid only after regeneration (the generator already scores against the actual RV). Drop "plain", "(no recalibration)", "early closes excluded" and "fitted on the winsorised target": Methods and the Results bullet "Forecast and realization" cover them. |
| 514-517 | `NOTE_DM`: "DM is the Diebold--Mariano statistic of the daily QLIKE difference {what}, with a Bartlett Newey--West variance at lag 6 and the Harvey--Leybourne--Newbold correction; $p$ is two-sided (normal). " | "$\DM$ is the Diebold--Mariano statistic of the daily loss differential {what}, with the Harvey--Leybourne--Newbold correction and a Newey--West variance with \fqsDMLag{} lags. " (\fqsDMLag = 6 is loaded by the wrapper at line 18; the generator asserts lag 6 at line 325.) |
| new | -- | `NOTE_STARS = "$^{*}$, $^{**}$ and $^{***}$ denote significance at the 10\\%, 5\\% and 1\\% levels. "` |
| 560, 624, 678, 721 | `\textit{Notes.}` | `\textit{Notes:}` |

#### Tables

##### `tab:rh_axis1` (generated/rehaul_results_t1_axis1.tex; generator lines 520-580, rows built at 1277-1303)
- **Problems vs core section 8:**
  - The caption is two sentences of method text.
  - 8 columns, including a separate $p$ column and two Sharpe columns ($\Delta$SR, $t$).
  - Bold marks significance on DM, $p$ and $t$ (`num(..., sig)`, `pval`).
  - The notes run to about 12 sentences. They carry Methods detail (pre-tuning, seed counts, refit schedules), code
    provenance ("the earlier paper's pooled trees") and model-set counts that belong in the text.
  - Code jargon: per-bar, pooled, "one model per bar", HAR + cal., All, plain, trade days, session, walk, interim,
    midpoint, $B_t$, $RV_t$, "lag 6" / "lag 14", British spellings.
  - The row groups are italic titles, not "Panel A/B".
  - The stale file still says "winsorised".
  - Decimals are fine (QLIKE 4, % 1, DM and $t$ 2, SR 2), and so is the `\small` body.
  - `\caption` sits outside `threeparttable` (minor: move it inside, as tab:rh_econ_allfeat does, so it takes the
    table's width).
- **Sharpe-change columns:** move them to the economics tables. tab:rh_econ_allfeat (`generated/rehaul_econ_e2_allfeat.tex`,
  make_rehaul_economics.py) already prints $\Delta$SR at mid-quotes and its $t$ for these rows, with the same numbers
  (Ridge +0.05/0.09, Lasso +0.50/1.47, Elastic net +0.55/1.34, MinGRU +0.49/0.95, and the pooled linear rows):
  - E2 lacks only the closing-interval trees' single specification (fixed hyperparameters, re-estimated daily; E2
    gives medians over the 15 specifications), the pooled intraday trees and the pooled MinGRU.
  - If the text needs those rows, add them to E2 (another generator; see Open questions).
  - Keep computing `dsh` and `t`: the asserts at lines 1346-1356 and 1376-1379 use them.
- **Proposed layout:**
  - Caption: "Forecast accuracy: HAR benchmark versus full predictor set"
  - Columns (5): Model | HAR | Full | $\Delta$ (\%) | $\DM$, with `\cmidrule(lr){2-3}` "QLIKE" over HAR and Full, and
    `\cmidrule(l){4-5}` "Full vs HAR" over the last two. Tabular `@{}l rr rr@{}`. Stars on $\DM$.
    Fallback if the Sharpe columns stay: 7 columns, adding $\Delta\SR$ | $t$ (stars on $t$).
  - Panels / rows: "Panel A: Closing-interval models" (Ridge, Lasso, Elastic net, LightGBM, XGBoost, Random forest,
    MinGRU); "Panel B: Pooled intraday models" (the same seven; Random forest$^\dagger$). Keep `\quad` indentation
    and `\addlinespace` between the panels.
  - Notes (5 sentences): "\textit{Notes:} The out-of-sample period is 3 January 2020 to 30 April 2024
    ($T_{\mathrm{oos}} = \rhrDays$ trading days). QLIKE is the average loss $\bar L$, with
    $L_t = \QLIKE(\RV_t, \hat\sigma^2_t)$, $\RV_t$ the realized variance of the closing half hour and
    $\hat\sigma^2_t$ the variance forecast; $\Delta = 100(\bar L^{\text{Full}}/\bar L^{\text{HAR}} - 1)$. $\DM$ is the
    Diebold--Mariano statistic of the daily loss differential (full predictor set minus HAR benchmark, same model),
    with the Harvey--Leybourne--Newbold correction and a Newey--West variance with \fqsDMLag{} lags; negative values
    favor the full predictor set. $^{*}$, $^{**}$ and $^{***}$ denote significance at the 10\%, 5\% and 1\% levels.
    Closing-interval trees use fixed hyperparameters and are re-estimated daily, and the MinGRU is ridge-initialized,
    trained and selected on QLIKE (see Section~\ref{sec:rh_methods_models}); $^\dagger$preliminary: the pooled
    intraday random forest's closing-interval forecasts on the full predictor set."
    Fallback sentence if $\Delta\SR$ stays: "$\Delta\SR$: change in the annualized Sharpe ratio of the sign strategy
    at mid-quotes; $t$: HAC $t$-statistic of the daily return differential (Newey--West variance with \fqsHACLag{}
    lags)."
  - Moves to notes / appendix / parked:
    - To the text: the counts sentence ("Across all \rhrAfPairs{} input-set pairs ... \rhrAfMseHigherSig{} ...").
      06_results.tex already says it at lines 126-131.
    - To Methods, already there: the seed counts ("5 per-bar, \rhrNNPooledSeeds{} pooled"; 05_methods.tex states
      "5 seeds (3 in the pooled intraday model)"), "Per-bar linear models are pre-tuned" and the pooled refit schedule.
    - To a `%` comment in the generated file: "The pooled LightGBM and XGBoost on all features are the earlier
      paper's pooled trees" (see Open questions).
    - To the economics tables: $\Delta$SR and $t$.
- **String edits:**

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 526-527 | `\caption{Axis 1: HAR + calendar against all features, within each family. QLIKE of the two input sets, ... Sharpe ratio.}` | `\caption{Forecast accuracy: HAR benchmark versus full predictor set}` (inside `threeparttable`) |
| 530 | `\begin{tabular}{@{}l rr rrr rr@{}}` | `\begin{tabular}{@{}l rr rr@{}}` |
| 532 | ` & \multicolumn{2}{c}{QLIKE} & \multicolumn{3}{c}{All vs HAR + cal.} & \multicolumn{2}{c}{Sharpe change} \\` | ` & \multicolumn{2}{c}{QLIKE} & \multicolumn{2}{c}{Full vs HAR} \\` |
| 533 | `\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(l){7-8}` | `\cmidrule(lr){2-3}\cmidrule(l){4-5}` |
| 534 | `Model & HAR + cal. & All & $\Delta$ (\%) & DM & $p$ & $\Delta$SR & $t$ \\` | `Model & HAR & Full & $\Delta$ (\%) & $\DM$ \\` |
| 538 | `("per-bar", "Per-bar (one model per bar)")` | `("per-bar", "Panel A: Closing-interval models")` |
| 539 | `("pooled", "Pooled (one model for all 48 bars)")` | `("pooled", "Panel B: Pooled intraday models")` |
| 541 | `\multicolumn{8}{@{}l}{\textit{...}}` | `\multicolumn{5}{@{}l}{\textit{...}}` |
| 546 | `\multicolumn{7}{l}{\textit{pending (walk running)}}` (dead branch: every `pool` is set) | `\multicolumn{4}{c}{--}` |
| 549 | `sig, sig_t = r["p"] < SIG, r["p_t"] < SIG` | delete (stars come from `p`) |
| 551-552 | `{q4(qb)} & {q4(qa)} & {num(pct,1)} & {num(dm,2,sig)} & {pval(p)} & {num(dsh,2)} & {num(t,2,sig_t)}` | `{q4(qb)} & {q4(qa)} & {num(pct,1)} & {stat(dm, p)}` |
| 559 | `\begin{tablenotes}[flushleft]\footnotesize` | keep |
| 560-563 | `\item \textit{Notes.} ` + NOTE_CONV + ` $\Delta = 100(\overline{L}^{\,\text{all}}/\overline{L}^{\,\text{HAR+cal.}} - 1)$. ` + NOTE_DM(what="(all features minus HAR + calendar, same model)") | the proposed notes above: `\textit{Notes:}`, $\Delta = 100(\bar L^{\text{Full}}/\bar L^{\text{HAR}} - 1)$, what = "(full predictor set minus HAR benchmark, same model)" |
| 564 | `Negative $\Delta$ and DM favour all features.` | "; negative values favor the full predictor set." |
| 564-566 | `$\Delta$SR is the change in the annualised Sharpe ratio of the midpoint straddle trade ($\sign$ of forecast minus implied variance); $t$ is the Newey--West $t$ (lag 14) of the daily return difference.` | delete (columns moved); fallback wording above |
| 566 | `Bold: $p < 0.05$ (DM, or the $t$ against the normal).` | NOTE_STARS |
| 567-570 | `Per-bar linear models are pre-tuned; the per-bar trees use fixed settings refit every session; the MinGRU is the ridge-initialised network ... (seed mean: 5 per-bar, \rhrNNPooledSeeds{} pooled). The pooled MinGRU is refit monthly, each refit on its trailing 24,000 bars; the pooled trees are refit at every bar.` | "Closing-interval trees use fixed hyperparameters and are re-estimated daily, and the MinGRU is ridge-initialized, trained and selected on QLIKE (see Section~\ref{sec:rh_methods_models});" |
| 570 | `The pooled LightGBM and XGBoost on all features are the earlier paper's pooled trees.` | delete from the notes (keep it as a `%` comment line in the generated .tex) |
| 571-572 | `$^\dagger$Interim: the pooled forest's 16:00 forecasts on all features; its full every-bar walk is still running.` | "$^\dagger$preliminary: the pooled intraday random forest's closing-interval forecasts on the full predictor set." |
| 572-574 | `Across all \rhrAfPairs{} input-set pairs of the model set, all features lower QLIKE in \rhrAfLower{} (\rhrAfLowerSig{} with $p < 0.05$) and raise variance-space MSE in ...` | delete (in the text at 06_results.tex:126-131) |
| 1296 | `"mark": "$^\\dagger$" if pool[1] in INTERIM else ""` | keep (the dagger stays) |

##### `tab:rh_axis2` (generated/rehaul_results_t2_pooled.tex; generator lines 583-641, rows built at 1381-1405)
- **Problems vs core section 8:**
  - The caption is a sentence of method text.
  - 9 columns, two of them $p$ columns.
  - Bold marks significance on DM, $p$ and $t$.
  - Code jargon: per-bar, pooled, "Axis 2", family, "the bank's walk", bar, session, "pre-tuned", interim, midpoint,
    "(24,000-bar window)", lag 6 / lag 14, $B_t$, $RV_t$, British spellings.
  - The notes run to about 10 sentences.
  - $t$ (the trade's return differential) is an economic statistic in an accuracy table, as in T1.
  - The row groups are fine (by model class; Panel A/B does not apply because each row compares the two samples).
- **Proposed layout:**
  - Caption: "Forecast accuracy: pooled intraday versus closing-interval models"
  - Columns (5): Model | $\Delta$ (\%) | $\DM$ | $\Delta$ (\%) | $\DM$, with `\cmidrule(lr){2-3}` "HAR benchmark" and
    `\cmidrule(l){4-5}` "Full predictor set". Tabular `@{}l rr rr@{}`. Stars on $\DM$.
    Fallback keeping the trade's $t$: 7 columns, Model | $\Delta$ (\%) | $\DM$ | $t$ | $\Delta$ (\%) | $\DM$ | $t$,
    with stars on $t$.
  - Panels / rows: italic group rows "Linear models" (Ridge, Lasso, Elastic net), "Tree ensembles" (LightGBM,
    XGBoost, Random forest$^\dagger$), "Neural network" (MinGRU).
  - Notes (5 sentences): "\textit{Notes:} Sample, loss and test as in Table~\ref{tab:rh_axis1}.
    $\Delta = 100(\bar L^{\mathrm{pool}}/\bar L^{\mathrm{close}} - 1)$ and $\DM$ compare the pooled intraday model
    with the closing-interval model of the same class on the same predictor set (pooled minus closing-interval);
    positive values favor the closing-interval model. $^{*}$, $^{**}$ and $^{***}$ denote significance at the 10\%,
    5\% and 1\% levels. The pooled intraday trees are re-estimated at every interval and the pooled MinGRU monthly,
    each on its most recent 24,000 intervals (see Section~\ref{sec:rh_methods_models}). $^\dagger$Preliminary (see
    Table~\ref{tab:rh_axis1})."
    Fallback sentence if $t$ stays: "$t$: HAC $t$-statistic of the daily differential of the sign strategy's
    mid-quote returns (pooled minus closing-interval; Newey--West variance with \fqsHACLag{} lags)."
    $\bar L^{\mathrm{pool}}$ and $\bar L^{\mathrm{close}}$ match 06_results.tex:144.
  - Moves to notes / appendix / parked: drop $t$ here and add the trees and MinGRU to tab:rh_econ_pooled (E3, which
    now covers linear models only), or keep $t$ under the 7-column fallback. 06_results.tex:155-157 cites this $t$
    for linear models and trees (see Open questions).
- **String edits:**

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 589-590 | `\caption{Axis 2: pooled against per-bar, by family and input set. QLIKE change (pooled minus per-bar), its Diebold--Mariano test and the $t$ of the trade's return difference.}` | `\caption{Forecast accuracy: pooled intraday versus closing-interval models}` (inside `threeparttable`) |
| 593 | `\begin{tabular}{@{}l rrrr rrrr@{}}` | `\begin{tabular}{@{}l rr rr@{}}` |
| 595 | ` & \multicolumn{4}{c}{HAR + calendar} & \multicolumn{4}{c}{All features} \\` | ` & \multicolumn{2}{c}{HAR benchmark} & \multicolumn{2}{c}{Full predictor set} \\` |
| 596 | `\cmidrule(lr){2-5}\cmidrule(l){6-9}` | `\cmidrule(lr){2-3}\cmidrule(l){4-5}` |
| 597 | `Model & $\Delta$ (\%) & DM & $p$ & $t$ & $\Delta$ (\%) & DM & $p$ & $t$ \\` | `Model & $\Delta$ (\%) & $\DM$ & $\Delta$ (\%) & $\DM$ \\` |
| 600 | `((LIN, "Linear"), (TREE, "Trees"), (NN, "Neural network"))` | `((LIN, "Linear models"), (TREE, "Tree ensembles"), (NN, "Neural network"))` |
| 602 | `\multicolumn{9}{@{}l}{...}` | `\multicolumn{5}{@{}l}{...}` |
| 607-612 | `num(pct,1), num(dm,2,p<SIG), pval(p), num(t,2,p_t<SIG)` | `num(pct,1), stat(dm, p)` |
| 624-628 | `\item \textit{Notes.} ` + NOTE_CONV + ` $\Delta = 100(\overline{L}^{\,\text{pooled}}/\overline{L}^{\,\text{per-bar}} - 1)$, the same estimator and input set. ` + NOTE_DM(what="(pooled minus per-bar)") | the proposed notes above |
| 629-630 | `Positive $\Delta$ and DM favour the per-bar fit. $t$ is the Newey--West $t$ (lag 14) of the daily difference of the midpoint straddle returns (pooled minus per-bar). Bold: $p < 0.05$.` | "positive values favor the closing-interval model." + NOTE_STARS (+ the fallback $t$ sentence) |
| 631-634 | `Per-bar trees: fixed settings refit every session; pooled trees: the bank's walk refit at every bar (24,000-bar window); linear models are pre-tuned in both fits. The pooled MinGRU is refit monthly ... both MinGRU fits are ridge-initialised and trained and selected on QLIKE.` | "The pooled intraday trees are re-estimated at every interval and the pooled MinGRU monthly, each on its most recent 24,000 intervals (see Section~\ref{sec:rh_methods_models})." |
| 635 | `$^\dagger$The pooled forest on all features is interim (16:00 forecasts; the full walk is still running).` | "$^\dagger$Preliminary (see Table~\ref{tab:rh_axis1})." |
| 1400 | `"mark": "$^\\dagger$" if ...` | keep |

##### `tab:rh_rank` (generated/rehaul_results_t3_ranking.tex; generator lines 644-694, row keys at 1555-1568)
- **Problems vs core section 8:**
  - The caption is two sentences.
  - 8 columns, including a $p$ column and a "Fit" column that panels make redundant.
  - Bold marks significance on DM and $p$, and also the reference row ("\textbf{OLS} (reference)", bold QLIKE).
  - The OLS row has empty Δ, DM and $p$ cells (use "--").
  - Code-style row labels: "XGBoost, fixed, daily refit"; "MinGRU (ridge-init, QLIKE-tuned)"; "Block-diagonal
    ridge, earlier panel"; "OLS, earlier panel"; Inputs "HAR + cal." / "All"; Fit "per-bar" / "pooled".
  - The notes run to about 10 sentences, with "canonical rung", "earlier paper's" and model-set counts.
- **Proposed layout:**
  - Caption: "Forecast accuracy relative to the HAR benchmark"
  - Columns (6): Rank | Model | Predictors | QLIKE | $\Delta$ (\%) | $\DM$, with `\cmidrule(l){5-6}` "vs HAR
    benchmark". Tabular `@{}r l l r rr@{}`. Stars on $\DM$.
  - Panels / rows:
    - "Panel A: Closing-interval models": OLS (benchmark), Lasso, XGBoost, LightGBM, Elastic net, MinGRU, Ridge and
      Random forest on Full, then the trees and the MinGRU on HAR, sorted by QLIKE.
    - "Panel B: Pooled intraday models": 14 rows, sorted by QLIKE.
    - Rank stays the global rank among the \rhrRankN{} forecasts. In the current numbers every pooled row ranks
      below every closing-interval row shown, so the split keeps the sort order.
    - Benchmark row: `OLS (benchmark) & HAR & <QLIKE> & -- & --`, no bold.
    - Earlier-draft rows (`a0`, `blk2`): park them from this table by changing line 1555 to `head = [OLS]`, and keep
      them in tab:rh_rank_full Panel C (see Open questions).
  - Notes (5 sentences): "\textit{Notes:} Sample, loss and test as in Table~\ref{tab:rh_axis1}; here $\Delta = 100(\bar
    L/\bar L^{\mathrm{OLS}} - 1)$ and $\DM$ compare each forecast with the closing-interval OLS on the HAR predictor
    set, and negative values favor the forecast. Rank is the position among the \rhrRankN{} forecasts of
    Table~\ref{tab:rh_rank_full}. Closing-interval trees use fixed hyperparameters and are re-estimated daily; the
    closing-interval ridge, lasso and elastic net on the HAR predictor set are within \rhrLinBaseMaxPct\% of the OLS
    and are omitted. $^{*}$, $^{**}$ and $^{***}$ denote significance at the 10\%, 5\% and 1\% levels.
    $^\dagger$Preliminary (see Table~\ref{tab:rh_axis1})."
  - Moves to notes / appendix / parked:
    - To the text: the count sentence ("Of the \rhrNSet{} model-set forecasts ... \rhrNWorseSig{} are significantly
      worse"). 06_results.tex:174-175 has the first half; \rhrNWorseSig is used only here.
    - To Table~\ref{tab:rh_axis2}'s note: "The pooled MinGRU is refit monthly, the pooled trees at every bar."
- **String edits:**

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 651-652 | `\caption{The headline forecasts ranked by QLIKE, against the per-bar OLS on HAR + calendar. Rank is the position among all \rhrRankN{} forecasts of Table~\ref{tab:rh_rank_full}.}` | `\caption{Forecast accuracy relative to the HAR benchmark}` (inside `threeparttable`) |
| 655 | `\begin{tabular}{@{}r l l l r rrr@{}}` | `\begin{tabular}{@{}r l l r rr@{}}` |
| 657 | ` & & & & & \multicolumn{3}{c}{vs per-bar OLS} \\` | ` & & & & \multicolumn{2}{c}{vs HAR benchmark} \\` |
| 658 | `\cmidrule(l){6-8}` | `\cmidrule(l){5-6}` |
| 659 | `Rank & Model & Fit & Inputs & QLIKE & $\Delta$ (\%) & DM & $p$ \\` | `Rank & Model & Predictors & QLIKE & $\Delta$ (\%) & $\DM$ \\` |
| 662 | `sub = rank.loc[keys].sort_values("qlike")` | Split by `r["fit"]` into two panels: `\multicolumn{6}{@{}l}{\textit{Panel A: Closing-interval models}} \\`, then Panel B; sort each by QLIKE; `\addlinespace` between them. |
| 664 | `tex_escape(r["name"]) + ("$^\\dagger$" if k in INTERIM else "")` | keep (the names change in `rung_label`, below) |
| 667 | `{rank} & \textbf{{name}} (reference) & {fit} & {inputs} & \textbf{{q4}} & & & \\` | `{rank} & OLS (benchmark) & {inputs} & {q4} & -- & -- \\` |
| 671-672 | `... & {r['fit']} & {r['inputs']} & {q4} & {num(pct,1)} & {num(dm,2,p<SIG)} & {pval(p)} \\` | `{rank} & {name} & {inputs} & {q4} & {num(pct,1)} & {stat(dm, p)} \\` |
| 678-681 | `\item \textit{Notes.} ` + NOTE_CONV + ` $\Delta = 100(\overline{L}/\overline{L}^{\,\text{OLS}} - 1)$. ` + NOTE_DM(what="(model minus the per-bar OLS)") | the proposed notes above |
| 682 | `Negative $\Delta$ and DM favour the model. Bold: $p < 0.05$.` | "negative values favor the forecast." + NOTE_STARS |
| 682-684 | `Rows: the per-bar OLS, the canonical rung of each family on each input set (per-bar trees with fixed settings refit every session) and the earlier paper's pooled OLS and block-diagonal ridge.` | "Closing-interval trees use fixed hyperparameters and are re-estimated daily;" (with the earlier-draft rows parked) |
| 684-685 | `The per-bar ridge, lasso and elastic net on HAR + calendar are within \rhrLinBaseMaxPct\,\% of the OLS and are not shown.` | "the closing-interval ridge, lasso and elastic net on the HAR predictor set are within \rhrLinBaseMaxPct\% of the OLS and are omitted." |
| 685 | `$^\dagger$Interim (see Table~\ref{tab:rh_axis1}).` | `$^\dagger$Preliminary (see Table~\ref{tab:rh_axis1}).` |
| 686 | `The pooled MinGRU is refit monthly, the pooled trees at every bar.` | delete (in T2's notes) |
| 687-688 | `Of the \rhrNSet{} model-set forecasts other than the OLS, \rhrNLower{} have lower QLIKE (\rhrNLowerSig{} with $p < 0.05$) and \rhrNWorseSig{} are significantly worse.` | delete (to the text) |
| 1555 | `head = [OLS, "a0", "blk2"]` | `head = [OLS]` (park the earlier-draft rows here) |

##### `tab:rh_rank_full` (generated/rehaul_results_t3_full.tex, longtable; generator lines 697-747; wrapper 06_results_floats.tex:143-146)
- **Placement:** move it to the appendix. Park lines 143-146 of 06_results_floats.tex ("A candidate for the
  appendix.") and `\input` it from `rehaul/10_appendix_floats.tex`:
  - add `\clearpage` before it;
  - add the guard `\ifdefined\rhrDays\else\input{generated/rehaul_results_numbers}\fi` (10_appendix_floats already
    loads the fqs macros at line 12);
  - add one bullet in 10_appendix.tex introducing it.
  - The `\ref{tab:rh_rank_full}` in tab:rh_rank's notes works across sections; the label is unchanged.
- **Problems vs core section 8:**
  - The body is `\footnotesize` (line 701), which section 8 forbids; it must be `\small`.
  - 8 columns, including $p$ and "Fit".
  - Bold marks significance; the reference row is bold.
  - $^\ast$ marks "not in the model set". It collides with the significance stars and must change.
  - The notes name a file path ("results/close\_master\_table/paper\_model\_set.csv").
  - Row labels are code-style: "XGBoost, Optuna (SE), tune 25, top 10", "rand.\ search (QLIKE), refit 10", "fixed,
    daily refit", "MinGRU (ridge-init, QLIKE-tuned)", "..., earlier panel".
  - Two different forecasts carry the identical label "Lasso (fixed penalty), earlier panel" (`b2lasso` and `lasso_f`;
    stale file rows 96 and 112).
  - The caption is a long sentence.
- **Proposed layout:**
  - Caption: "Forecast accuracy of all forecasts"
  - Columns (7): Rank | Model | Specification | Predictors | QLIKE | $\Delta$ (\%) | $\DM$. Longtable
    `@{}r l l l r rr@{}`, `\small`, `\tabcolsep` 4pt. Stars on $\DM$.
    - Model = the estimator in words (OLS, Ridge, Lasso, Elastic net, LightGBM, XGBoost, Random forest, MinGRU).
    - Specification = the tuning specification, "--" where there is none.
    - Width estimate: about 465pt at `\small` with the longest Specification cell (about 33 characters), against a
      469pt text width. Check for an overfull `\hbox` after the build. If it overflows, use `\tabcolsep` 3pt or
      `pdflscape`; do not shrink the font.
  - Panels / rows (each panel sorted by QLIKE; Rank = the global rank):
    - "Panel A: Closing-interval models" (99 rows): OLS, closing-interval linear, the 90 tree specifications, MinGRU.
    - "Panel B: Pooled intraday models" (14 rows): pooled linear (`poolpt_*`), pooled trees (`pooltree_*`, plus the
      master table's `lgbm` and `xgb`), pooled MinGRU.
    - "Panel C: Pooled intraday models, earlier specification" (7 rows): `a0`, `blk2`, `blk2_inc`, `lasso_f`,
      `b2lasso`, `lasso_t`, `enet`. Park this panel instead if the coordinator decides so (Open questions).
    - Implementation: `rung_label` (lines 411-476) returns (model, specification, panel, predictors). The panel is
      "A" for per-bar, "C" for the keys of the `paper` dict at lines 429-442 and "B" for every other pooled key.
    - Benchmark row: `OLS (benchmark) & -- & HAR & <QLIKE> & -- & --`, no bold.
    - Markers: $^\dagger$ stays; replace $^\ast$ with $^\ddagger$. The interim forest becomes "Random
      forest$^{\dagger\ddagger}$".
  - Specification vocabulary for the closing-interval trees (the 15 specifications per estimator):

    | Key pattern | Current label (`rung_label`) | Specification cell |
    |---|---|---|
    | `subtree_<est>_<set>` | "fixed, refit 10" | "Fixed, every 10 d" |
    | `subtree_daily_<set>_<est>` | "fixed, daily refit" | "Fixed, daily" |
    | `subtree_tuned_<est>` | "rand.\ search (SE), refit 10" | "Random search (MSE), every 10 d" |
    | `subtree_tuned_daily_<est>` | "rand.\ search (SE), daily refit" | "Random search (MSE), daily" |
    | `subtree_tunedq[_daily]_<est>` | "rand.\ search (QLIKE), ..." | "Random search (QLIKE), every 10 d" / "..., daily" |
    | `subtree_optuna_tp<n>[_k<k>]_<est>` | "Optuna (SE), tune n, top k" | "Optuna (MSE), n d, best of k" ("daily" for n = 1; k = 50 when no `_k` token) |
    | `subtree_optunaq_tp25_<est>` | "Optuna (QLIKE), tune 25, top 50" | "Optuna (QLIKE), 25 d, best of 50" |
    | pooled trees (B) | -- | "Fixed, every interval" |
    | `a0` | "OLS, earlier panel" | Model "OLS", Spec "--" |
    | `blk2` | "Block-diagonal ridge, earlier panel" | Model "Ridge", Spec "Block penalty" |
    | `blk2_inc` | "Block-diag.\ ridge, no FOMC, earlier panel" | Model "Ridge", Spec "Block penalty, no FOMC" |
    | `lasso_f` | "Lasso (fixed penalty), earlier panel" | Model "Lasso", Spec "Fixed penalty" |
    | `b2lasso` | "Lasso (fixed penalty), earlier panel" (duplicate) | Model "Lasso", Spec "Fixed penalty, no FOMC" (verify; Open questions) |
    | `lasso_t` | "Lasso (causally tuned), earlier panel" | Model "Lasso", Spec "Re-tuned penalty" |
    | `enet` | "Elastic net (causally tuned), earlier panel" | Model "Elastic net", Spec "Re-tuned penalty" |

  - Notes (5 sentences, in the `\endlastfoot` `p{\linewidth}` cell, `\footnotesize`): "\textit{Notes:} Sample, loss,
    test and significance markers as in Table~\ref{tab:rh_axis1}; $\Delta$ and $\DM$ compare each forecast with the
    HAR benchmark, as in Table~\ref{tab:rh_rank}. Closing-interval trees (Panel A) use fixed hyperparameters or random
    search (re-tuned every 250 trading days) on squared-error loss (MSE) or QLIKE, re-estimated daily or every 10
    trading days (d); or Optuna (TPE) on MSE or QLIKE, re-tuned every 1, 5, 25 or 250 trading days, keeping the best of
    the first 10, 25 or 50 trials (see Section~\ref{sec:rh_methods_tuning}). Panel C: pooled intraday linear models of
    an earlier specification, with a block penalty, a fixed penalty or a re-tuned penalty; ``no FOMC'' omits the FOMC
    predictors. The pooled intraday trees are re-estimated at every interval and the pooled MinGRU monthly.
    $^\ddagger$Not in the model set counted in the text; $^\dagger$preliminary."
  - Moves to notes / appendix / parked: the whole table to the appendix. Delete the file path, the code words
    ("tune", "top", "rung", "earlier panel") and "HLN".
- **String edits:**

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 698 | `head = "Rank & Model & Fit & Inputs & QLIKE & $\Delta$ (\%) & DM & $p$ \\"` | `"Rank & Model & Specification & Predictors & QLIKE & $\\Delta$ (\\%) & $\\DM$ \\\\"` |
| 701 | `{\footnotesize` | `{\small` |
| 702 | `\setlength{\tabcolsep}{3.5pt}` | `\setlength{\tabcolsep}{4pt}` (check the width) |
| 703 | `\begin{longtable}{@{}r l l l r rrr@{}}` | `\begin{longtable}{@{}r l l l r rr@{}}` |
| 704-706 | `\caption{Every forecast ranked by QLIKE, against the per-bar OLS on HAR + calendar (\rhrRankN{} forecasts: the \rhrNModelSet{}-forecast model set, the per-bar and pooled MinGRU and the pooled trees outside the master table).}` | `\caption{Forecast accuracy of all forecasts}` |
| 712 | `\multicolumn{8}{@{}l}{\textit{Table~\ref{tab:rh_rank_full}, continued}} \\` | `\multicolumn{7}{@{}l}{\textit{Table~\ref{tab:rh_rank_full}, continued}} \\` |
| 718 | `\multicolumn{8}{r@{}}{\textit{continued on the next page}} \\` | `\multicolumn{7}{r@{}}{\textit{Continued on next page}} \\` |
| 721 | `\multicolumn{8}{@{}p{\linewidth}@{}}{\footnotesize\raggedright\textit{Notes.} Conventions as in Table~\ref{tab:rh_rank}:` | `\multicolumn{7}{@{}p{\linewidth}@{}}{\footnotesize\raggedright\textit{Notes:} ` + the proposed notes |
| 722-723 | `plain $\hat y_t^2 B_t$ forecasts on \rhrDays{} days; DM against the per-bar OLS (lag 6, HLN), negative favours the model; bold $p < 0.05$.` | delete (covered by "as in Table~\ref{tab:rh_axis1}") |
| 723-724 | `$^\ast$Not in the master table's model set (results/close\_master\_table/paper\_model\_set.csv).` | `$^\ddagger$Not in the model set counted in the text;` |
| 724 | `$^\dagger$Interim.` | `$^\dagger$preliminary.` |
| 724-727 | `Per-bar tree rungs: fixed settings refit every 10 sessions or every session (daily refit); random search on squared error (SE) or QLIKE, re-run every 250 sessions; Optuna on SE or QLIKE re-run every 1, 5, 25 or 250 sessions (tune), keeping the best of the first 10, 25 or 50 trials (top).` | the Panel A sentence of the proposed notes |
| 727 | `Earlier panel: the pooled 48-bar models of the earlier draft.` | the Panel C sentence (or delete it if Panel C is parked) |
| 727-728 | `The pooled MinGRU is refit monthly, the pooled trees at every bar.` | "The pooled intraday trees are re-estimated at every interval and the pooled MinGRU monthly." |
| 731-736 | loop over `rank.sort_values("qlike")`; marker `("$^\\ast$" if not r["in_set"] else "")` | loop per panel; marker `"$^{\\ddagger}$"`; combine with the dagger as `$^{\\dagger\\ddagger}$` |
| 739 | `{rank} & \textbf{{name}} (reference) & {fit} & {inputs} & \textbf{{q4}} & & & \\` | `{rank} & OLS (benchmark) & -- & {inputs} & {q4} & -- & -- \\` |
| 743-744 | `... & {r['fit']} & {r['inputs']} & {q4} & {num(pct,1)} & {num(dm,2,p<SIG)} & {pval(p)} \\` | `{rank} & {model} & {spec} & {inputs} & {q4} & {num(pct,1)} & {stat(dm, p)} \\` |
| 413 | `sets = {"baseline": "HAR + cal.", "all_features": "All"}` | `{"baseline": "HAR", "all_features": "Full"}` (also used by tab:rh_rank) |
| 416 | `"MinGRU (ridge-init, QLIKE-tuned)"` | `"MinGRU"` (described once in tab:rh_axis1's notes) |
| 418, 426, 448, 450 | fit strings `"per-bar"` / `"pooled"` | internal only once the Fit column goes; map them to the panels A/B/C |
| 430-441 | the `paper` dict labels ("OLS, earlier panel", "Block-diagonal ridge, earlier panel", "Block-diag.\\ ridge, no FOMC, earlier panel", 2x "Lasso (fixed penalty), earlier panel", "Lasso (causally tuned), earlier panel", "Elastic net (causally tuned), earlier panel") | (Model, Specification) pairs from the table above; panel C |
| 445-446 | `lgbm`, `xgb`: `EST_NAME[key], TREE, "pooled", "All"` | Model = EST_NAME, Spec "Fixed, every interval", panel B, "Full" |
| 448 | `return "OLS", LIN, "per-bar", "HAR + cal."` | Model "OLS (benchmark)", Spec "--", panel A, "HAR" |
| 462 | `how = "fixed, refit 10"` | `"Fixed, every 10 d"` |
| 464 | `how = "fixed, daily refit"` | `"Fixed, daily"` |
| 466, 472 | `loss = "QLIKE" if ... else "SE"` | `... else "MSE"` |
| 467-469 | `f"rand.\\ search ({loss}), " + ("daily refit" if ... else "refit 10")` | `f"Random search ({loss}), " + ("daily" if ... else "every 10 d")` |
| 475 | `f"Optuna ({loss}), tune {tp}, top {k}"` | `f"Optuna ({loss}), {'daily' if tp == '1' else tp + ' d'}, best of {k}"` |
| 476 | `return f"{EST_NAME[est]}, {how}", TREE, "per-bar", sets[b]` | return the Model (`EST_NAME[est]`) and the Specification (`how`) separately |

#### Kept figures

`fig:rh_cumulative` (figures/rehaul_results_f3_cumulative.pdf; `fig_f3`, lines 1113-1162). The caption is typed in the
wrapper, 06_results_floats.tex:85-93.

| Line | Current | Replacement |
|---|---|---|
| 1576 | `("subpt_lasso_all_features", LIN, "Lasso, all features")` | label `"Lasso"` (the caption says every line is on the full predictor set). The end label at line 1126 takes `label.split(",")[0]`, so it still works. |
| 1577 | `"XGBoost, all features"` | `"XGBoost"` |
| 1578 | `"MinGRU, all features"` | `"MinGRU"` (these labels also feed the parked `fig_f5`; harmless) |
| 1132 | `"Cumulative QLIKE difference\nvs per-bar OLS (sum over days)"` | `"Cumulative QLIKE loss differential\nvs. HAR benchmark"` |
| 1145 | `"below 0: the model has lower total QLIKE than the OLS"` | delete the annotation (lines 1142-1151; the caption says it), or `"Below 0: lower cumulative loss than the HAR benchmark"` |
| 1153 | `label="24 Feb–30 Apr 2020"` | keep (or `"24 February–30 April 2020"`) |
| 1129-1131 | end-of-line labels (Lasso / XGBoost / MinGRU) | keep. With the shortened legend they duplicate it; optionally keep only the shaded-window patch in the legend. |

- Wrapper caption (lines 85-93) is already in standard terms. Two caveats:
  - "Losses are computed against actual realized variance" matches the rescored generator. The PDF and the
    `\rhrCum*` values on disk are pre-rescore, so regenerate.
  - Optional wording: "against $\RV_t$, the realized variance of the closing half hour".
- Parked (skip): `fig:rh_axis1_forest` (F1), `fig:rh_pooled_dumbbell` (F2), `fig:rh_se_share` (F5). The generator
  still draws them (lines 1573-1574, 1584); that is harmless.

#### Macro values with user-visible text

| Macro | Line | Current value | Replacement |
|---|---|---|---|
| (all `\rhr...`) | 1620-1739 | numeric only (`\ensuremath{\pm}` signs via `pm()`); no value holds words or code names | none |
| `\rhrPoolTreeBaseQMin`, `\rhrPoolTreeBaseQMax` | 1736-1737 | 3 decimals (`f"{...:.3f}"`) | optional: `q4(...)`, to match the 4-decimal QLIKE rule if the prose quotes them |

- The `%` description comments ("trade days scored", "per-bar", "HAR + cal.", "sign(s) Sharpe mid") are not printed.
  Optionally update their wording.
- The file header comment at line 1742 ("plain f^2 B, 866 days, DM lag 6 HLN") is a comment.
- Two macros lose their only table use under this plan:
  - `\rhrNWorseSig`: move it into the 06_results.tex text.
  - `\rhrNNPooledSeeds`: 05_methods.tex types "3"; it could use this macro.

#### Other user-visible strings
- Panel and group titles: T1 lines 538-539; T2 line 600; T3 and T3-full new panel rows (tables above).
- Footnote markers: `$^\dagger$` (lines 664, 734, 1296, 1400) stays, worded "Preliminary". `$^\ast$` (line 735)
  becomes `$^\ddagger$`, because the stars now mean significance.
- "pending (walk running)": T1 line 546 is a dead branch; the copy at line 901 is in parked F1.
- Longtable continuation lines 712 and 718 (above).
- Wrapper `rehaul/06_results_floats.tex` (comments only):
  - lines 8-9: "bold = p < 0.05" becomes "stars at 10/5/1 %, no p columns";
  - lines 143-146: move the longtable to the appendix (see tab:rh_rank_full).
- `rehaul/06_results.tex:46-47`, the "REFERENCE = blk2" comment (comment only), is stale twice over:
  - the line is now `master_table_close.py:247`;
  - the "porting" it asks for is done, because this generator computes the % change and DM against the
    closing-interval OLS itself (`pair(by[k], by[OLS])`, lines 1526-1528).
  - Proposed comment: "% Code: master_table_close.py:247 sets REFERENCE = "blk2" for its own *_vs_ref columns; the % and
    DM against the closing-interval OLS in tab:rh_rank / tab:rh_rank_full come from make_rehaul_results.py (pair(...,
    by[OLS]), lines 1526-1528)."
- `rehaul/06_results.tex`, the hand-typed duplicate of T1 (now at lines 92-113, not ~64-83): already parked with a
  dated `% 2026-10-15 (notation pass): parked` note in favor of tab:rh_axis1. Nothing to do.

#### Open questions
1. **Sharpe columns.** Drop $\Delta$SR and $t$ from tab:rh_axis1, and $t$ from tab:rh_axis2 (recommended: the accuracy
   tables carry accuracy only)? The economics generator would then need rows it lacks:
   - E2 (tab:rh_econ_allfeat): the closing-interval trees' fixed, re-estimated-daily specification, the pooled trees
     and the pooled MinGRU;
   - E3 (tab:rh_econ_pooled): the trees and the MinGRU (it is linear-only now);
   - 06_results.tex:155-157 cites T2's $t$.
   Otherwise keep them, with the 7-column fallbacks given.
2. **Earlier-draft pooled models.** Park `a0` and `blk2` from tab:rh_rank (recommended), and either keep them as
   Panel C of the appendix longtable or park Panel C too. The appendix ladder (tab:rh_app_tune_ladder) starts from "the
   earlier draft's pooled intraday forecast", which argues for keeping Panel C.
3. **Panel attribution.** `atm_straddle_lib.py:35-40` says `blk2` and `lasso_f` are on the FOMC panel, and `blk2_inc`,
   `lasso_t`, `enet`, `lgbm`, `xgb` and `b2lasso` are on the earlier panel without FOMC columns. The generator labels
   all of them "earlier panel". Verify this before using "no FOMC" in the Specification cells. The duplicate label for
   `b2lasso` and `lasso_f` must be disambiguated either way.
4. **Pooled LightGBM and XGBoost on the full predictor set.** These are the master table's `lgbm` and `xgb` (earlier
   panel). Their HAR-benchmark twins come from `pooled_trees_2026-10-07`. If the two runs differ in specification
   beyond the predictor set, Methods must say so; the T1 note sentence at line 570 is being removed from the notes.

### C.2 `make_rehaul_economics.py`

Writes: `generated/rehaul_econ_e1_control.tex`, `rehaul_econ_e2_allfeat.tex`, `rehaul_econ_e3_pooled.tex`,
`rehaul_econ_e4_sizing.tex`, `rehaul_econ_fig_cumpnl.tex` (+ `figures/rehaul_econ_cumpnl.pdf`),
`rehaul_econ_fig_sizing.tex` and `rehaul_econ_fig_byyear.tex` (both parked, + their PDFs), and `rehaul_econ_numbers.tex`
(`\rhe...` macros). Wrappers: `rehaul/07_economics_floats.tex` `\input`s E1 (l.14), fig_cumpnl (l.17), E2 (l.20),
E3 (l.23) and E4 (l.56). It parks fig_sizing (l.57-58) and fig_byyear (l.61-62). `fig:rh_econ_qlike_sharpe` is a
commented-out float in the wrapper itself, drawn by `make_fig_qlike_sharpe.py`, so it is skipped here. Every caption and
note comes from the generator; the wrapper holds only `\input` lines and comments. `07_economics.tex` l.20 inputs the
numbers file.

**State when this was written (read this first).** A concurrent "2026-10-15 (table legibility)" pass, working from
`TABLE_FIXES_FROM_FLOATS_2026-10-15.md`, rewrote this generator between 14:17 and 14:30 today: 1,444 lines became
2,071. It parked the old `table_e1..e4`, `NOTE_FILLS`, `NOTE_SHARPE`, `NOTE_HAC`, `panel` and `tree_rung`, and added
finance-style helpers (l.459-541), stars, panels and about 20 new `\rhe...` macros. Every line number below refers to
that 2,071-line version (mtime 14:30:10). The generated `.tex` files are from 14:26:32, before its last edits, so E3's
caption and header in `generated/` still differ from the code (l.1035, 1041-1042). Re-run the script after editing.
Below, items marked **done** need no action; the rest are what remains.

Status of the brief's specific issues:

| Brief issue | Status in the current generator |
|---|---|
| E1 nine columns, bold for significance | **done**: 7 columns, stars, $t$ under the estimate (l.646-723) |
| E1 tnote a "selected on the evaluation sample" | **done**: kept as "so selected ex post on the out-of-sample period" (l.715-716); the spec wording still needs fixing (l.403) |
| NOTE_FILLS / NOTE_SHARPE / NOTE_HAC repeated | **done**: parked (l.437-456), replaced by shared `HAC_NOTE` + `STAR_NOTE` (l.465-474). Remaining: `HAC_NOTE` writes "$q = $" for the lag (l.472) |
| "lfss" / "Long/scaled" | partly done: the E4 header now reads "Full long, scaled short", which must become "Asymmetric" (l.1280-1281). The `RULES` labels are still `Long/scaled` (l.1068). The macro name `\rheSzLfssMed` stays |
| "sign(s)" headers | E4 header **done** ("Sign"). The `RULES` labels (l.1066, 1069) still hold `$\sign(s)$` and `Rank $|s|$`; no active table uses them, but they should still change |
| E2 `\texttt{baseline}` | **done** (gone) |
| E2 tree rows "0/15" | **not done**: now `$[0/15]$` in brackets under the medians (`count()`, l.514-516, 858-863) |
| E3 "per-bar minus pooled" | partly done: header "Closing-interval minus pooled intraday" (l.1039), but it spans $\Delta\QLIKE$, which is a ratio; the direction for $\Delta\SR$ is never stated |
| E4 "days 253--866 (614 days)" | **done** in the table: `\rheSzWarmup`, `\rheSzDays`, `\rheNdays` (l.1289-1291). Remaining: the out-of-sample dates of the sizing sample, and the typed 252 / 614 in `07_economics.tex` l.90-91, 110-111 |
| "annualised per trade day (... about 200 days a year)" | **done** in every active float. It survives only in the parked fig_byyear caption (l.1818) |
| QLIKE scoring caveat | E3's QLIKE (`qlike_raw`) is already scored against the actual $\RV_t$: I checked `master_table_daily.parquet`, where `qlike_raw` equals the QLIKE of `rv_unclipped`/`plain` exactly and differs from the `target` (winsorized) one by up to 0.62. The MinGRU is scored on `rv_unclipped` (l.233-234) and is not in E3. No "winsorised" string is left in any active note. The post-rescore wording can go in now (E3 notes) |

#### Tables

##### `tab:rh_econ_control` (generated/rehaul_econ_e1_control.tex; generator lines 632-723, helpers 459-541, tree_spec 383-406)
- **Problems vs core section 8:**
  - **Done:** 7 columns; stars; $t$ in parentheses under the estimate; Panel A/B; `\small`; booktabs; minus signs in math; fixed decimals (mean 1, $\SR$ 2, $t$ 2, % 1); macros for the sample; the tree-selection disclosure is kept.
  - "Always short" is still the benchmark's name in the row label (l.673) and the notes (l.709, 712). Guide: "unconditional short-straddle benchmark" $q_t \equiv -1$, with "Short" allowed once it is defined.
  - The lag is written "$q = \rheHACLag$" through `HAC_NOTE` (l.472). Core rule: no $q$ for the lag in a table note ($q_t$ is the position). Write "\rheHACLag{} lags". The value equals `\fqsHACLag` (14) either way.
  - The notes repeat the benchmark's $\SR$ "(\rheShortSh)" (l.713), which the first row already shows.
  - Tree spec vocabulary (l.403): "Optuna search (10 trials), re-tuned every 25 trading days" should read "Optuna (TPE), best of the first 10 trials, re-tuned every 25 trading days".
  - Header "Bid--ask" (l.701) while the notes say "quoted bid/ask prices". Use "Bid/ask" for one term throughout (E1-E3).
  - Row order: Elastic net, Lasso, Ridge (`PERBAR_ALL` l.81-85, `POOLED_ALL` l.86-90). The results tables (`rehaul_results_t1_axis1.tex`) use Ridge, Lasso, Elastic net. Optional: reorder for consistency.
  - Stars sit on the estimate, with $t$ under it (documented at l.33-36, 459-463). Core section 8 allows this ("on the statistic or estimate"); the brief says "on the statistic". See Open questions.
- **Proposed layout:**
  - Caption: "Returns of the sign strategy" (already l.695).
  - Columns (7): Model | Mean return (%) | $\SR$: Mid | $\SR$: Bid/ask | Hit rate (%) | Long days (%) | $\Delta\SR$, with `\cmidrule(lr){3-4}` under "$\SR$" (as now).
  - Panels / rows:
    - First row, outside the panels: "Short ($q_t \equiv -1$)".
    - Panel A, closing-interval models: OLS, HAR benchmark; Ridge; Lasso; Elastic net; LightGBM$^a$; XGBoost$^a$; Random forest$^a$; MinGRU$^b$.
    - Panel B, pooled intraday models: Ridge; Lasso; Elastic net.
    - HAC $t$ in parentheses under Mean and under $\Delta\SR$.
  - Notes (5 sentences): "\textit{Notes:} Daily returns over the out-of-sample period, \rheSampleStart{} to
    \rheSampleEnd{} ($T_{\mathrm{oos}} = \rheNdays$ trading days), of the sign strategy,
    $q_t = \sign(\hat\sigma^2_t - \mathrm{IV}^2_t)$, and of the unconditional short-straddle benchmark (Short),
    $q_t \equiv -1$; every model except the OLS uses the full predictor set. Mean return is at mid-quotes, in percent of
    the premium; $\SR$ is the annualized Sharpe ratio at mid-quotes and at quoted bid/ask prices; hit rate and long days
    are the shares of days with $r_t > 0$ at mid-quotes and with $q_t = +1$. $\Delta\SR$ is the $\SR$ at mid-quotes minus
    the benchmark's. In parentheses, the HAC $t$-statistic of the mean return and, under $\Delta\SR$, the HAC
    $t$-statistic on the daily return differential (Newey--West, \rheHACLag{} lags;
    Section~\ref{sec:rh_methods_tests}). $^{*}$, $^{**}$ and $^{***}$ denote significance at the 10\%, 5\% and 1\%
    levels."
    - tnote a: "For each tree estimator, the specification with the highest $\SR$ at mid-quotes among its
      \rheTreeSpecs{} on the full predictor set, so selected ex post on the out-of-sample period. {tree_note}."
      `{tree_note}` is generated by `tree_spec`, with l.403 fixed.
    - tnote b: unchanged (l.717).
  - Moves to notes / appendix / parked: nothing further. The Sharpe formula, fills and settlement rules are already in
    `07_economics.tex` (Performance measures / Execution).
- **String edits** (generator line -> current -> replacement):

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 472 | `(Newey--West, $q = \rheHACLag$; ` (in `HAC_NOTE`, shared by E1-E3) | `(Newey--West, \rheHACLag{} lags; ` |
| 403 | `f"Optuna search ({k} trials), re-tuned {every}, selected on {loss}"` | `f"Optuna (TPE), best of the first {k} trials, re-tuned {every}, selected on {loss}"` |
| 673 | `body = row("Always short", SHORT)` | `body = row(r"Short ($q_t \equiv -1$)", SHORT)` |
| 701 | `& return (\%) & Mid-quotes & Bid--ask & rate (\%)` | `& return (\%) & Mid & Bid/ask & rate (\%)` |
| 707-709 | `Daily returns of the straddle position over the out-of-sample period, ... The sign strategy holds $q_t = \sign(...)$ straddles and always short holds $q_t = -1$;` | the first sentence of the proposed notes above |
| 712-713 | `$\Delta\SR$ is $\SR$ at mid-quotes minus that of always short (\rheShortSh); in parentheses, ...` | `$\Delta\SR$ is the $\SR$ at mid-quotes minus the benchmark's. In parentheses, ...` |
| 715-716 | `The specification with the highest $\SR$ at mid-quotes among the estimator's \rheTreeSpecs{{}} ...` | `For each tree estimator, the specification with the highest $\SR$ at mid-quotes among its \rheTreeSpecs{{}} ...` (optional) |
| 81-90 | `PERBAR_ALL` / `POOLED_ALL` order enet, lasso, ridge | ridge, lasso, enet (optional; matches results T1) |
| 690 | `float_head("Table E1", "the sign strategy and always short")` (a `%` comment line) | optional: `"the sign strategy and the short-straddle benchmark"` |

##### `tab:rh_econ_allfeat` (generated/rehaul_econ_e2_allfeat.tex; generator lines 830-944)
- **Problems vs core section 8:**
  - **Done:** 5 columns; stars; $t$ under the estimate; Panel A/B; no `\texttt`; short caption; macros for the counts.
  - **Panel C still shows "0/15"** as `$[0/15]^{\phantom{***}}$` under each median (`count()`, l.514-516; called at
    l.861-862). A count in a $\Delta\SR$ column breaks the column's meaning. The bid/ask counts equal the mid-quote counts
    in every row (0, 3, 1, 4, 4), so the second bracket repeats the first.
  - Short-header rule: "HAR benchmark & Full set" (l.919) should be "HAR & Full"; "Mid-quotes & Bid--ask" should be "Mid & Bid/ask".
  - Panel C labels: "Medians across pairs" (l.878), "All trees" (l.883) and "Model set (\rhePairs)" (l.884). The last
    row is all 51 HAR/Full pairs, not the model set.
  - `HAC_NOTE` lag "$q =$" (via l.928).
  - Notes are 5 sentences plus tnote a. Fine apart from the Panel C sentence, which changes with the fix below.
  - Panel A/B show the same $\Delta\SR$ and $t$ as `tab:rh_axis1` (results T1, last two columns). See Open questions.
- **Proposed layout** (cleaner Panel C: medians only; the counts move into one note sentence):
  - Caption: "Sharpe ratios: HAR benchmark versus full predictor set" (already l.913).
  - Columns (5): Model | $\SR$ at mid-quotes: HAR | Full | $\Delta\SR$: Mid | Bid/ask. Use `\cmidrule(lr){2-3}` and
    `\cmidrule(l){4-5}`, and the group header "$\Delta\SR$ (Full $-$ HAR)".
  - Panels / rows:
    - Panel A, closing-interval models: Ridge, Lasso, Elastic net, MinGRU$^a$.
    - Panel B, pooled intraday models: Ridge, Lasso, Elastic net.
    - Panel C, "Medians across specifications": LightGBM (\rheTreePairsEach), XGBoost (\rheTreePairsEach), Random
      forest (\rheTreePairsEach), All tree models (\rheTreePairs), All pairs (\rhePairs). One line each, no row under
      it, phantom-padded cells as now.
    - Alternative if the counts must stay in the table: give Panel C its own header row ("Pairs | Median HAR | Median
      Full | Median $\Delta\SR$ | Significant at 5\%"), with the count as a plain integer in its own column.
  - Notes (5 sentences): "\textit{Notes:} Each row estimates one model on the HAR benchmark and on the full predictor
    set, with all else unchanged, over the out-of-sample period ($T_{\mathrm{oos}} = \rheNdays$ trading days);
    $\Delta\SR$ is the full-set minus the HAR-benchmark $\SR$ of the sign strategy, with the HAC $t$-statistic on the
    daily return differential in parentheses (Newey--West, \rheHACLag{} lags; Section~\ref{sec:rh_methods_tests}). On
    the HAR benchmark the closing-interval ridge, lasso and elastic net hold the same position as the closing-interval
    OLS on every day, so they share its $\SR$. Panel C reports medians across the \rheTreePairsEach{} specifications of
    each tree estimator and across all \rhePairs{} pairs of the model set. At mid-quotes, \rheTreeSigLgbm,
    \rheTreeSigXgb{} and \rheTreeSigRf{} of the LightGBM, XGBoost and random-forest pairs, and \rhePairsSig{} of all
    \rhePairs{} pairs, have a $\Delta\SR$ significant at the 5\% level, and none is significantly negative. $^{*}$,
    $^{**}$ and $^{***}$ denote significance at the 10\%, 5\% and 1\% levels."
    - New macros `\rheTreeSigLgbm`/`Xgb`/`Rf` are computed in `summary()` as `int((v.t_mid > SIG).sum())`. Current
      values 0/3/1; `\rhePairsSig` = 4.
    - Assert that the `t_x` counts equal the `t_mid` counts, or say "at mid-quotes and at quoted bid/ask prices".
  - Moves to notes / appendix / parked: the bracket counts move into the notes; park `count()`.
- **String edits:**

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 514-516 | `def count(k, n): return f"$[{k}/{n}]^{{\\phantom{{***}}}}$"` | park (`# 2026-10-15: parked -- counts move to the notes`) |
| 858-864 | `under = ["", "", count(int((v.t_mid > SIG).sum()), n), count(int((v.t_x > SIG).sum()), n)]` | `under = ["", "", "", ""]` (`rows2` then skips the row); return the counts for the new macros |
| 878 | `panel_head("Panel C: Medians across pairs", 5)` | `panel_head("Panel C: Medians across specifications", 5)` |
| 883 | `r"All trees (\rheTreePairs)"` | `r"All tree models (\rheTreePairs)"` |
| 884 | `r"Model set (\rhePairs)"` | `r"All pairs (\rhePairs)"` |
| 917 | `\multicolumn{{2}}{{c}}{{$\Delta\SR$}}` | `\multicolumn{{2}}{{c}}{{$\Delta\SR$ (Full $-$ HAR)}}` |
| 919 | `& HAR benchmark & Full set & Mid-quotes & Bid--ask \\` | `& HAR & Full & Mid & Bid/ask \\` |
| 925-933 | `Each pair is one model estimated ... with the number of pairs whose differential is significantly positive at the 5\% level in brackets (none is significantly negative at that level).` | the proposed notes above |
| 872, 876 | `for e in ("enet", "lasso", "ridge"):` | `("ridge", "lasso", "enet")` (optional, order) |

##### `tab:rh_econ_pooled` (generated/rehaul_econ_e3_pooled.tex; generator lines 1002-1061)
- **Problems vs core section 8:**
  - **Done:** 6 columns; stars from the DM test's own $p$ (`dmr["p"]`, l.1021); $t$ and DM under the estimates; panels by predictor set. These panels are the right exception: the table compares the two model classes, so Panel A is the full predictor set and Panel B the HAR benchmark. Short caption (l.1035; not yet in `generated/`).
  - **Sign convention:** "Closing-interval minus pooled intraday" (l.1039) spans $\Delta\QLIKE$, which is
    `100*(closing/pooled - 1)` (l.1013), a relative difference. It is a minus only when stated "in percent of the
    pooled intraday model's". The notes give the favorable sign only for QLIKE/DM ("negative values favor"), never
    for $\Delta\SR$.
  - Header l.1041-1042: "$\Delta\SR$, & $\Delta\SR$," with trailing commas over "mid-quotes & bid--ask". Use one
    `\multicolumn{2}{c}{$\Delta\SR$}` with a `\cmidrule`.
  - Lags: "$q = $" via `HAC_NOTE` (l.1052) and "Newey--West $q = \rheDMLag$" (l.1054). `\rheDMLag` = 6 = `\fqsDMLag`
    (dm_test's automatic `floor(4 (T/100)^{2/9})` at $T = 866$).
  - No statement of the QLIKE realization. It is already the actual $\RV_t$ (verified), so the post-rescore wording can go in.
  - Optional: the bid/ask $\Delta\SR$ is within 0.01 of the mid one in every row (also true in E2). If dropped, add
    one note sentence giving the computed max $|\Delta\SR^{\mathrm{mid}} - \Delta\SR^{\mathrm{ba}}|$ (not typed).
- **Proposed layout:**
  - Caption: "Closing-interval versus pooled intraday models" (l.1035).
  - Columns (6): Model | $\SR$ at mid-quotes: Pooled intraday | Closing interval | $\Delta\SR$: Mid | Bid/ask |
    $\Delta\QLIKE$ (%).
    - Header row 1: `& \multicolumn{2}{c}{$\SR$ at mid-quotes} & \multicolumn{3}{c}{Closing interval minus pooled intraday}`,
      then `\cmidrule(lr){2-3}\cmidrule(l){4-6}`.
    - Row 2: `& Pooled & Closing & \multicolumn{2}{c}{$\Delta\SR$} & $\Delta\QLIKE$`, then `\cmidrule(lr){4-5}`.
    - Row 3: `& intraday & interval & Mid & Bid/ask & (\%)`.
  - Panels / rows: Panel A: Full predictor set; Panel B: HAR benchmark. Each holds Ridge, Lasso, Elastic net (or the
    current enet/lasso/ridge order), with $t$ and DM in parentheses underneath.
  - Notes (5 sentences): "\textit{Notes:} Each row compares the closing-interval and the pooled intraday model of the
    same estimator and predictor set over the out-of-sample period ($T_{\mathrm{oos}} = \rheNdays$ trading days); the
    pooled intraday model is evaluated at the closing interval, and both have hyperparameters selected on a pre-sample.
    $\Delta\SR$ is the closing-interval minus the pooled intraday $\SR$ of the sign strategy, with the HAC $t$-statistic
    on the daily return differential in parentheses (Newey--West, \rheHACLag{} lags). $\Delta\QLIKE$ is the
    closing-interval minus the pooled intraday mean QLIKE, in percent of the pooled intraday model's, with QLIKE
    computed against $\RV_t$, the realized variance of the closing half hour; in parentheses, the Diebold--Mariano
    statistic with the Harvey--Leybourne--Newbold correction (Newey--West variance with \rheDMLag{} lags;
    Section~\ref{sec:rh_methods_tests}). Positive $\Delta\SR$ and negative $\Delta\QLIKE$ favor the closing-interval
    model; $^{*}$, $^{**}$ and $^{***}$ denote significance at the 10\%, 5\% and 1\% levels."
  - Moves to notes / appendix / parked: nothing; the bid/ask $\Delta\SR$ is optional (see above).
- **String edits:**

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 1039 | `\multicolumn{{3}}{{c}}{{Closing-interval minus pooled intraday}}` | `\multicolumn{{3}}{{c}}{{Closing interval minus pooled intraday}}` (keep; the notes define both differences) |
| 1041 | `& Pooled & Closing & $\Delta\SR$, & $\Delta\SR$, & \\` | `& Pooled & Closing & \multicolumn{{2}}{{c}}{{$\Delta\SR$}} & $\Delta\QLIKE$ \\` then `\cmidrule(lr){{4-5}}` |
| 1042 | `& intraday & interval & mid-quotes & bid--ask & $\Delta\QLIKE$ (\%) \\` | `& intraday & interval & Mid & Bid/ask & (\%) \\` |
| 1050-1052 | `... $\Delta\SR$ is the closing-interval minus the pooled intraday $\SR$, at mid-quotes and at quoted bid/ask prices, with the {HAC_NOTE} in parentheses.` | as in the proposed notes (the lag via the fixed `HAC_NOTE`) |
| 1052-1055 | `$\Delta\QLIKE$ is the closing-interval model's mean QLIKE relative to the pooled intraday model's, in percent, with the Diebold--Mariano statistic (Harvey--Leybourne--Newbold correction, Newey--West $q = \rheDMLag$) in parentheses; negative values favor the closing-interval model.` | `$\Delta\QLIKE$ is the closing-interval minus the pooled intraday mean QLIKE, in percent of the pooled intraday model's, with QLIKE computed against $\RV_t$, the realized variance of the closing half hour; in parentheses, the Diebold--Mariano statistic with the Harvey--Leybourne--Newbold correction (Newey--West variance with \rheDMLag{{}} lags; Section~\ref{{sec:rh_methods_tests}}). Positive $\Delta\SR$ and negative $\Delta\QLIKE$ favor the closing-interval model.` |
| 1010 | `for e in ("enet", "lasso", "ridge"):` | optional reorder (ridge, lasso, enet) |

##### `tab:rh_econ_sizing` (generated/rehaul_econ_e4_sizing.tex; generator lines 1065-1070, 1190-1300)
- **Problems vs core section 8:**
  - **Done:** 5 columns; Panel A mid-quotes / Panel B quoted bid/ask prices, each with "Closing-interval models" and "Pooled intraday models" sub-heads; macros for 252/614/866/112; short caption; "no test" stated.
  - **Rule names** are not the settled ones. Header l.1280-1281 reads "Robust clip" and "Full long, scaled short" on two
    rows. Settled: "Sign", "Clipped", "Asymmetric", "Rank".
  - `RULES` labels l.1066-1069 (`$\sign(s)$`, `Clip`, `Long/scaled`, `Rank $|s|$`) are unused by the active table but are the grep-able source of the old names.
  - Benchmark row l.1224: "Always short & \multicolumn{4}{c}{... under every rule}". It should say "Short ($q_t \equiv -1$)", and "under every rule" moves to the notes.
  - "Model set, number above sign" (l.1244): "Sign" is now a rule name, so capitalize it.
  - The notes say "defined in the text" with no pointer (`07_economics.tex` has no `\label` on its strategy subsection).
  - The sizing sample is given as a day count only; its first date is not reported. It is `days[WARMUP]` = 11 August 2021 (read from `master_table_daily.parquet`; index 252 of 866). The end is `\rheSampleEnd`.
  - The model-set rows include the nine earlier-draft pooled forecasts (see Open questions).
- **Proposed layout:**
  - Caption: "Sharpe ratios by position-sizing rule" (l.1276).
  - Columns (5): Model | Sign | Clipped | Asymmetric | Rank, on one header row.
  - Panels / rows:
    - Panel A, $\SR$ at mid-quotes, and Panel B, $\SR$ at quoted bid/ask prices.
    - Each panel opens with the row "Short ($q_t \equiv -1$) & \multicolumn{4}{c}{value}", followed by:
      - Closing-interval models: OLS, HAR benchmark; Ridge; Lasso; Elastic net; LightGBM; XGBoost; Random forest; MinGRU.
      - Pooled intraday models: Ridge; Lasso; Elastic net.
      - "Model set, median" and "Model set, number above Sign".
  - Notes (5 sentences): "\textit{Notes:} Annualized Sharpe ratios of the straddle position $q_t$ under four sizing
    rules, each a function of $\hat\sigma^2_{t'} - \mathrm{IV}^2_{t'}$ on days $t' \le t$ only: Sign is the sign
    strategy, and Clipped, Asymmetric and Rank scale it by the size of the gap relative to earlier days (see
    Section~\ref{sec:rh_econ_strategy}). The first \rheSzWarmup{} trading days of the out-of-sample period only estimate
    that scale, so every rule is evaluated on the remaining \rheSzDays{} trading days, \rheSzStart{} to
    \rheSampleEnd{}; at quoted bid/ask prices the spread is paid on $|q_t|$ straddles. Every model except the OLS uses
    the full predictor set, the tree models are the specifications of Table~\ref{tab:rh_econ_control}, and the
    short-straddle benchmark (Short, $q_t \equiv -1$) is the same under every rule. The last two rows of each panel give
    the median across the \rheModelSetN{} forecasts of the model set (the MinGRU is not in it) and the number of them
    whose $\SR$ exceeds that of their own sign strategy. Point estimates; no test is reported."
    - If the label `sec:rh_econ_strategy` is not added, keep "defined in the text".
    - New macro `\rheSzStart` = `f"{days[WARMUP].day} {days[WARMUP]:%B %Y}"` (11 August 2021).
  - Moves to notes / appendix / parked: nothing.
- **String edits:**

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 1066 | `("sign", r"$\sign(s)$"),` | `("sign", "Sign"),` |
| 1067 | `("clip", "Clip"),` | `("clip", "Clipped"),` |
| 1068 | `("lfss", "Long/scaled"),` | `("lfss", "Asymmetric"),` (key `lfss` stays: it indexes `gui._score_sizes`) |
| 1069 | `("rank", r"Rank $|s|$"),` | `("rank", "Rank"),` |
| 1224 | `rf"Always short & \multicolumn{{4}}{{c}}{{{cell(short_v, starred=False)} under every rule}} \\"` | `rf"Short ($q_t \equiv -1$) & \multicolumn{{4}}{{c}}{{{cell(short_v, starred=False)}}} \\"` |
| 1244 | `r"Model set, number above sign & -- & "` | `r"Model set, number above Sign & -- & "` |
| 1280-1281 | ` & & Robust & Full long, & \\` / ` & Sign & clip & scaled short & Rank \\` | ` & Sign & Clipped & Asymmetric & Rank \\` (or build it from `RULES` labels: `" & ".join(lab for _, lab in RULES)`) |
| 1287-1294 | `Annualized Sharpe ratios ... under four sizing rules (defined in the text), ... Point estimates; no test is reported.` | the proposed notes above |
| (new, in `main` near l.1953) | -- | `nums += mac("rheSzStart", f"{days[WARMUP].day} {days[WARMUP]:%B %Y}", "sizing: first scored day")` |

#### Kept figures

`fig:rh_econ_cumpnl` (drawn in `fig_cumpnl`, l.1350-1447; float caption l.1773-1792; legend labels from `key4`, l.1711-1716).
- **Done:** the `key4` labels "Closing-interval elastic net", "Closing-interval LightGBM", "Closing-interval MinGRU", "Pooled intraday elastic net"; the ax1 y-label (l.1405); the ax2 marker legend "each model's {TOP_DAYS} days of largest gain" (l.1440); caption numbers as macros; benchmark differential written $r_{t'} + R_{t'}$.
- Remaining: "Always short" (legend, axis, caption), and a $t$/$t'$ index slip in the caption.
- `FAM_LABEL` (l.114-120) is not used by `fig_cumpnl`; only the parked `fig_sizing` (l.1555-1561) uses it. Update it anyway so no old term survives.

| Line | Current | Replacement |
|---|---|---|
| 119 | `"short": "Always short",` (FAM_LABEL) | `"short": "Short-straddle benchmark",` |
| 1406 | `ax2.set_ylabel("Cumulative gain over always\nshort (straddle premia)")` | `ax2.set_ylabel("Cumulative gain over the short\nbenchmark (straddle premia)")` |
| 1413 | `labels = [...] + ["Always short"]` | `+ ["Short-straddle benchmark"]` (check the 3-column legend still fits with `--preview`) |
| 1779 | `\caption{Cumulative returns of the sign strategy. The top panel plots` | keep the title; optional "Panel A (top) plots" / "Panel B (bottom) plots" |
| 1780 | `in units of the straddle premium, for always short and for one model of each class` | `in units of the straddle premium, for the unconditional short-straddle benchmark ($q_t \equiv -1$) and for one model of each class` |
| 1782 | `The bottom panel plots the cumulative gain over always short,` | `The bottom panel plots the cumulative gain over the benchmark,` |
| 1784 | `of them days with $q_t = +1$ and $R_t \ge \rheTopRMin$` | `of them days with $q_{t'} = +1$ and $R_{t'} \ge \rheTopRMin$` |
| 1785-1788 | the 8 gain macros in two lists, then the tail sentence | keep. Optional: the caption is 6 sentences, so move the two gain lists to the prose (the macros already exist) and keep the tail sentence |

The axis tick labels are years (`DateFormatter("%Y")`, l.1408); there is no x-label and there are no panel titles.

Parked, not inventoried: `fig:rh_econ_sizing` (code l.1450-1575, caption l.1794-1809) and `fig:rh_econ_byyear`
(l.1577-1664, caption l.1810-1823). Both still write old terms ($\sign(s_t)$, "crossed", "per-bar", "annualised per
trade day", typed 112, days 253-866) into `generated/`, but they are not `\input`. Skip them.

#### Macro values with user-visible text
No `\rhe...` macro holds code text. The only text values are dates, and they are fine. The `%` comments after the
macros still use the old terms ("per-bar enet", "sizing lfss", "clip above sign(s)", "HAR + cal.", "614 days", "of
112"; l.1843-1909). They are not rendered; tidying them is optional.

| Macro | Line | Current value | Replacement |
|---|---|---|---|
| `\rheSampleStart` | 1927-1929 | `3 January 2020` | keep |
| `\rheSampleEnd` | 1930-1932 | `30 April 2024` | keep |
| `\rheSzLfssMed` | 1880-1890 | `1.12` (name says Lfss) | keep the name (never rename); comment "Asymmetric rule" |
| *(new)* `\rheSzStart` | after 1954 | -- | `days[WARMUP]`, currently "11 August 2021" (E4 notes) |
| *(new)* `\rheTreeSigLgbm` / `\rheTreeSigXgb` / `\rheTreeSigRf` | after 1952 | -- | E2 Panel C counts, `t_mid > SIG` (0 / 3 / 1 now) |
| *(new)* `\rheLassoSh` | after 1859 | -- | `m2(st["subpt_lasso_all_features"]["sh_mid"])` (1.88), for the typed "1.88" in `07_economics.tex` l.181 |
| *(new)* `\rhePairEnetD`, `\rhePairLassoD` | after 1879 | -- | `m2(e2i["pairs"]["subpt_enet_baseline"]["d_mid"], plus=True)` (+0.55) and the lasso one (+0.50), for `07_economics.tex` l.187 |
| *(new)* `\rheShortNegYears`, `\rheNYears` | after 1730 | -- | `sum(v < 0 for v in yrs[SHORT])` (3) and `len(yrs["years"])` (5), for "3 of the 5 calendar years" (`07_economics.tex` l.167). By-year short $\SR$: $-0.43, -0.48, 1.94, 0.88, -2.06$ |

These new macros remove the "Still typed, no loadable macro" list in the `07_economics.tex` header (l.16-17). That
file's l.90-91 and 110-111 can switch to the existing `\rheSzWarmup` / `\rheSzDays`. Those edits belong to the `.tex`
editor.

#### Other user-visible strings
- Panel titles (all fine): "Panel A: Closing-interval models" / "Panel B: Pooled intraday models" (E1 l.674, 681;
  E2 l.871, 875); "Panel A: Full predictor set" / "Panel B: HAR benchmark" (E3 l.1006-1007); "Panel A: $\SR$ at
  mid-quotes" / "Panel B: $\SR$ at quoted bid/ask prices" (E4 l.1250, 1252); sub-heads "Closing-interval models" /
  "Pooled intraday models" (E4 l.1227, 1234). Change "Panel C: Medians across pairs" (E2 l.878) as above.
- Footnote markers: E1 uses `\tnote{a}` (trees) and `\tnote{b}` (MinGRU); E2 uses `\tnote{a}` (MinGRU). The same
  letter means different things in the two tables. That is acceptable, since each table defines its own.
- `HEADER` and `float_head` (l.435, 539-540) write `%` comment lines that say "always short" (l.690) and "scored
  exactly as the GUI (rows._score_sizes)" (l.1270). They are comments only.
- No `\texttt`, no code names (`subpt_`, `poolpt_`, `baseline`) and no `\footnotesize` table bodies remain in the
  active floats. Every table body is `\small`; the notes are `\footnotesize`.

#### Open questions
1. **Star placement.** This generator stars the estimate and puts the HAC $t$ or DM under it (finance convention;
   core section 8 allows "statistic or estimate"). The brief says "on the statistic". Choose one paper-wide and make
   the other generators match.
2. **Section label for the strategy definitions.** `07_economics.tex` has no `\label` on "The straddle strategy" or
   "Performance measures". Adding `\label{sec:rh_econ_strategy}` (a new label, not a rename) would let E4's notes, and
   E1's if wanted, point to it. Until then, keep "defined in the text".
3. **Sizing-rule names in the text.** `07_economics.tex` l.77-80 still says "robust clip" and "full long, scaled short".
   The tables will say "Clipped" and "Asymmetric", so the text must define the rules under those names.
4. **Model set.** The 112-forecast model set includes the nine earlier-draft pooled forecasts (`family == "paper"`:
   `a0, blk2, blk2_inc, lgbm, xgb, lasso_t, lasso_f, enet, b2lasso`; `family_of` maps them to "pooled", l.167-181).
   They enter E4's model-set rows, `\rheModelSetN`, `\rheTailNeg` and `\rheSigVsShort`. They have no HAR pair, so they
   are not in E2's 51. Per the cross-cutting rule, describe them in one note sentence or drop them from the model set;
   the second changes the asserted outline numbers (l.1689-1701).
5. **Overlap with results T1.** E2 Panels A/B repeat the $\Delta\SR$ and $t$ columns of `tab:rh_axis1`
   (`rehaul_results_t1_axis1.tex`: elastic net +0.55 (1.34), lasso +0.50 (1.47), ...). Keep both, or drop the columns
   from T1?

### C.3 `make_rehaul_data_methods.py`

Writes: `generated/rehaul_dm_sources.tex` (`tab:rh_dm_sources`), `rehaul_dm_samples.tex` (`tab:rh_dm_samples`),
`rehaul_dm_design.tex` (`tab:rh_dm_design`), `rehaul_dm_dictionary.tex` (`tab:rh_dm_dictionary`, a landscape `longtable`),
`rehaul_dm_hyper.tex` (`tab:rh_dm_hyper` and `tab:rh_dm_hyper_search`, two floats in one file), `rehaul_dm_tuning.tex`
(`tab:rh_dm_tuning`). Wrappers: `rehaul/03_data_floats.tex:7-8` (sources, samples); `rehaul/04_features_floats.tex:7`
(design; its dictionary `\input` is parked at :11); `rehaul/10_appendix_floats.tex:456` (dictionary);
`rehaul/05_methods_floats.tex:7-8` (hyper, tuning). **Every caption and note lives in the generator**; the wrappers
hold only `%` comments and `\input` lines. Text references: `03_data.tex:11` (sources), `:101` (samples);
`04_features.tex:17` (dictionary), `:179` (design). `tab:rh_dm_hyper`, `tab:rh_dm_hyper_search` and `tab:rh_dm_tuning`
have no `\ref` in `05_methods.tex` (Open question 5).

> **State of the generator (read first).** While this inventory was being written (14:26 to 14:55:47 on 2026-10-15),
> another agent ran a "table legibility" pass on this generator. It parked the six old writers as `#` comments (D1 at
> line 1086, D2 1135, F1 1188, F2 1242,
> M1 1605, M2 1807) and wrote new ones: `write_sources` 1978-2111,
> `write_samples` 2114-2246, `write_design` 2249-2402, `write_dictionary` 2405-2542,
> `write_hyper` 2545-2902 and `write_tuning` 2905-3116. Along the way it took up most of this section's
> draft proposals, re-ran the generator (`generated/rehaul_dm_*.tex` at 14:48:05) and formatted the file. A later
> "consolidation" edit (14:55) changed the pooled-tree row of `tab:rh_dm_tuning`; the `.tex` on disk predates that
> edit. Line numbers below are for the file as finalized here (mtime 14:55:47, 3180 lines). "Current" strings come
> from the new writers and the regenerated `.tex`. **Only the residual items below remain.** The coordinator should keep
> one agent on the six writers from here on.

#### Status of the brief's items

| Brief item | Status now | Where |
|---|---|---|
| Dictionary "Series" column: `\texttt` code names | fixed: symbol `series_tex()` + words `desc_of()`; code names only as `% data field:` comments | 2539 |
| `STAT` / `BUCKET_LABEL` replacement content | fixed: `DESC`, `DESC_XS`, `GROUP_WORDS`, `desc_of` (A.2 words); `BUCKET_LABEL` = the Predictors names | 746, 637 |
| Dictionary "Source file" column | renamed "Source", in words (`SOURCE_WORDS`) | 787 |
| Very long longtable notes, file paths | fixed: 5 sentences, no paths | 2405-2542 |
| `[vendor]` placeholders | **open**: 6 cells + one notes sentence | 1983 |
| `src/data/loading.py` etc. in notes | fixed | -- |
| Hyper tables: `\texttt` hyperparameter names | fixed: `TREE_WORDS` / `RF_WORDS` / `FIXED_WORDS` (words + Methods symbols; libraries named in the notes) | 1890 |
| $R_{BB}$ -> $\underline\omega$ | fixed | 2651 |
| "pre-tune", "margin retune", "menu", `*_opt_00` | fixed ("pre-sample", "re-selection", "best trial of an Optuna search") | -- |
| `tp25_q` | fixed ("in one specification, QLIKE") | -- |
| "TPE 20260929 + row" seeds | fixed: "all random seeds are fixed" | 3102 |
| Design "Avail." / "Active" headers | **partly**: now "Missing-data indicators $\bar a^{(k)}$", "Activity indicators $\bar\chi^{(k)}$"; the columns count *averages* | 2272 |
| Samples rows ("Per-bar forecast rows", "walk-forward", "Linear pre-tune rows ...") | fixed; panels in core-8 order; $T_{\mathrm{oos}}$ marked | 2114-2246 |
| Tuning "Tree bank (20 experts)" | fixed by a consolidation edit: "Trees, fixed hyperparameters (one of 20 configurations per method)", no re-selection; label wording still to trim | 3059 |
| MinGRU of record (ridge-initialized, trained and selected on QLIKE) | fixed: both samples; the pooled run (`NN["pooled_rq"]`, 144 configurations) now read with its own selection files | 2819, 2931 |
| 7-column limit | no table exceeds 7 (check at the end) | -- |

#### Tables

##### `tab:rh_dm_sources` (generated/rehaul_dm_sources.tex; generator lines 1978-2111)
- **Problems vs core section 8:**
  - `[vendor]`: `V = r"\textit{[vendor]}"` is appended to 6 source cells ("E-mini S\&P 500 futures (ES), \textit{[vendor]}",
    "Index constituents, equal- and value-weighted, \textit{[vendor]}", "StockTwits, ...", "Cboe, ...", "FOMC calendar,
    ...", "SPXW options on expiration days, ..."). The 5th notes sentence reads "\textit{[vendor]}: the vendor is not yet named in any data file or
    document (the option chain is attributed to OptionMetrics only in a code comment)." `TABLE_FIXES_FROM_FLOATS`
    said "leave them and list them"; the brief lists them as a problem (Open question 2).
  - Otherwise compliant: 6 columns (Source | Series | First | Last | Obs. | Use), `\small`, words only, 5 sentences,
    caption "Data sources.".
- **Proposed layout:**
  - Caption: "Data sources." (as now)
  - Columns (6): Source | Series | First | Last | Obs. | Use (as now)
  - Rows: as now, without "[vendor]".
  - Notes (4 sentences): sentences 1-4 as now; the vendor sentence is dropped. The open item stays as the `\todo`
    in `03_data.tex:54`.
- **String edits:**

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 1983 | `V = r"\textit{[vendor]}"` | delete, and drop `, {V}` from the 6 source cells (lines 1991, 2002, 2013, 2033, ...) |
| 2105 | `{V}: the vendor is not yet named in any data file or document (the option chain is attributed to OptionMetrics only in a code comment).` | delete |

##### `tab:rh_dm_samples` (generated/rehaul_dm_samples.tex; generator lines 2114-2246)
- **Problems vs core section 8:** none left. The rows use the guide's terms. Panels are A closing-interval models,
  B pooled intraday models, C out-of-sample period, with the grid row above them. Units sit in the panel titles,
  and "Evaluation days ($T_{\mathrm{oos}}$)" is marked. 5 columns, 5 sentences.
- **Proposed layout:** as now. Optional: mark the 1,469 closing-interval forecasts as the variable-importance sample
  (core 6b).
- **String edits:**

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 2156 | `Rolling-window forecasts` (Panel A) | optional: `Rolling-window forecasts (variable-importance sample $\mathcal O$)` |

##### `tab:rh_dm_design` (generated/rehaul_dm_design.tex; generator lines 2249-2402)
- **Problems vs core section 8:**
  - Headers "Missing-data / indicators / $\bar a^{(k)}$" and "Activity / indicators / $\bar\chi^{(k)}$": the columns
    count the *averages* of the indicators. 04_features.tex:51-52 and 141 say "the same averages of the availability
    indicator" and "averages of an activity indicator", and the brief asks for "availability / activity averages".
    The notes already say "the same averages of its availability (missing-data) and activity indicators".
  - Width: `@{}lrrrrrr@{}` with an `l` first column ("HAR components $\times$ open, close", "Additional predictors")
    and three 3-line headers. Check the compile log for an overfull `\hbox`.
  - Otherwise compliant: 7 columns (the limit), no bold, "--" throughout, 5 sentences, the four constant calendar
    dummies stated (addenda A.3).
- **Proposed layout:**
  - Caption: "Composition of the predictor sets." (as now)
  - Columns (7): Predictor group | Series | Series averages $u^{(k)}$ | Availability averages $\bar a^{(k)}$ | Activity
    averages $\bar\chi^{(k)}$ | Total | Total, with the "Closing-interval model" spanner over 3-6 and "Pooled
    intraday" over 7 (as now).
  - Rows / notes: as now.
- **String edits:**

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 2272 | `stack("Missing-data", "indicators", r"$\bar a^{(k)}$")` | `stack("Availability", "averages", r"$\bar a^{(k)}$")` |
| 2273 | `stack("Activity", "indicators", r"$\bar\chi^{(k)}$")` | `stack("Activity", "averages", r"$\bar\chi^{(k)}$")` |

##### `tab:rh_dm_dictionary` (generated/rehaul_dm_dictionary.tex, landscape `longtable`, Appendix; generator lines 2405-2542)
- **Problems vs core section 8:** none left. The 6 columns are Variable | Description | Source | Periodicity
  adjustment | Transform $g(v)$ | Predictors, $k \in \mathcal K$. Symbols come from `series_tex`, words from `desc_of`,
  and $g(v)$ / $\operatorname{asinh}(u/\kappa)$ match 04_features.tex. The body is `\small` (notes `\footnotesize`),
  there are 5 notes sentences with no paths, and code names appear only as trailing `% data field: ...` comments
  (core section 6 allows code in comments). Caption: "Predictor definitions.".
- **Proposed layout:** as now.
- **Replacement dict content** (the brief asks for it). The generator now holds it; quoted here for the record
  (746-797). Each ES series has words + the 03_data.tex formula: "Interval log return,
  $r_{t,i} = \sum_k r_{t,i,k}$", "Realized absolute variation, $\sum_k |r_{t,i,k}|$", "Realized skewness, unnormalized,
  $\sum_k r_{t,i,k}^3$", "Realized quarticity, unscaled, ...", "Positive realized semivariance, ...", "Bipower
  variation, $\mathrm{BV}_{t,i} = ...$", "First-order realized autocovariance, $\mathrm{AC}_{t,i} = ...$", "ES contract
  volume", "Number of one-minute returns, $N_{t,i} \le 30$". Constituent series get "Equal-weighted" / "Value-weighted"
  + words only (`DESC_XS`). The FOMC rows read "FOMC announcement-interval dummy (14:30 ET)", "FOMC announcement-day
  dummy", "$1/(1 + n^{\pm}_{t,i})$, ...". `BUCKET_LABEL` (637): "Realized moments", "Liquidity",
  "Equal-weighted market", "Value-weighted market", "Sentiment", "Implied volatility", "Volatility demand", "FOMC" (=
  04_features.tex:15-17). Optional wording polish: the VD rows say "opening trades" where A.2 says "opening trades
  only".
- **String edits:** none required.

##### `tab:rh_dm_hyper` (generated/rehaul_dm_hyper.tex, first float; generator lines 2545-2902, part (i))
- **Problems vs core section 8:** none left. The 5 columns are Model | Hyperparameter | Search space | Value: HAR
  benchmark | Value: Full predictor set. Panel A is penalized linear models ($\lambda$, $\alpha$, $\underline\omega$);
  Panel B is the tree ensembles with fixed hyperparameters, in words with Methods symbols ($J$, $\eta$, $B$, $n_{\min}$,
  $\lambda_1^{\mathrm{tree}}$, $\lambda_2^{\mathrm{tree}}$, $\gamma$, $N$). The notes have 5 sentences, name the
  libraries (\citep{Ke2017}, \citep{Chen2016}, scikit-learn) and give the macro-consistent re-selection count (4 of 24).
- **Proposed layout:** as now. One content check: notes sentence 3 says "random forest: trial 255 of 400"
  (2763), next to "the best trial" for the boosters (Open question 4).
- **String edits:** none required.

##### `tab:rh_dm_hyper_search` (generated/rehaul_dm_hyper.tex, second float; generator lines 2545-2902, part (ii))
- **Problems vs core section 8:** none left. The 4 columns are Model | Hyperparameter | Search space | Selected (count).
  Panel A covers the re-tuned trees; Panel B is the MinGRU network of record (ridge-initialized, trained and selected
  on QLIKE): the closing-interval model (216 configurations) and the pooled intraday model (144 configurations,
  `NN["pooled_rq"]`). The squared-error development networks are parked as comments (2847). The notes give the round cap
  $\lceil 4 B^{\mathrm{fix}}\eta^{\mathrm{fix}}/\eta \rceil$, depth "unrestricted" and the computed number of predictor
  sets.
- **Proposed layout:** as now. If the appendix MinGRU development table (`make_rehaul_appendix.py`,
  `tab:rh_app_mingru_dev`) needs the squared-error grids, give them in its notes.
- **String edits:** none required.

##### `tab:rh_dm_tuning` (generated/rehaul_dm_tuning.tex; generator lines 2905-3116)
- **Problems vs core section 8:**
  - A later "consolidation" edit (3049) settled the pooled-tree
    row as one fixed configuration per method ("single arms of that bank", as in Methods 05_methods.tex:221-222). The
    row label still reads "Trees, fixed hyperparameters (one of 20 configurations per method)". The "20
    configurations" means nothing to a reader, since the bank is not described anywhere in the paper.
  - The last notes sentence keeps "(part of them only every 10th)". With one configuration per method it should say
    which method is re-estimated every 10th interval, or drop the clause.
  - Otherwise compliant: 7 columns (the limit), short cells, units in the panel titles, "switch if $\DM <
    -z_{0.05/|\mathcal C|}$", QLIKE criteria for both MinGRU rows (the pooled one from the run of record: "12
    re-estimations (5 blocks)"), no seed values, 4 notes sentences.
- **Proposed layout:** as now.
- **String edits:**

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 3059 | `Trees, fixed hyperparameters (one of {20} configurations per method)` | `Trees, fixed hyperparameters` |
| 3105 | `the pooled trees use one fixed configuration per method, re-estimated at every interval (part of them only every {10}th).` | name the method(s) re-estimated every {10}th interval, or end at "re-estimated at every interval." |

#### Column-limit check

| Table | Columns | Note |
|---|---|---|
| `tab:rh_dm_sources` | 6 | the old "Freq." column is gone |
| `tab:rh_dm_samples` | 5 | -- |
| `tab:rh_dm_design` | 7 | at the limit; check for an overfull `\hbox` |
| `tab:rh_dm_dictionary` | 6 (landscape `longtable`) | "Source file" -> "Source"; "Data field" -> `%` comments |
| `tab:rh_dm_hyper` | 5 | -- |
| `tab:rh_dm_hyper_search` | 4 | -- |
| `tab:rh_dm_tuning` | 7 | at the limit; the "Seeds" values are gone, only the count $S$ remains |

No table exceeds 7 columns; no split or landscape page is needed beyond the dictionary's.

#### Kept figures
None drawn here (the generator draws no figures).

#### Macro values with user-visible text
| Macro | Line | Current value | Replacement |
|---|---|---|---|
| -- | -- | -- | -- |

None. The generator defines no number or text macros. `HEAD` (179) only `\providecommand`s the layout
helpers `\rdmL` / `\rdmR`, which only the parked writers use.

#### Other user-visible strings
- Panel headings (`panel_row()`, 257) are all in the guide's terms: "Panel A: Closing-interval models (trading
  days)", "Panel B: Pooled intraday models (30-minute intervals)", "Panel C: Out-of-sample period (trading days)",
  "Panel A: Penalized linear models, selected on the pre-sample", "Panel B: Tree ensembles, fixed hyperparameters
  (selected on the `{y0}`--`{y1}` pre-sample)", "Panel A: Tree ensembles, closing-interval models (`{50}` Optuna trials per
  re-selection)", "Panel B: MinGRU network (ridge-initialized, trained and selected on QLIKE)"; dictionary group heads
  "`{BUCKET_LABEL}` (`{n}`)", with "FOMC (4; closing-interval model only)".
- Longtable continuation strings "Table \ref{tab:rh_dm_dictionary} (continued)" and "continued on the next page":
  fine.
- Dates print as ISO `\mbox{1998-01-05}` (`D_()`): acceptable.
- Dead code the rewrite left (not printed): `STAT` (919), `stat_of`, `TRANSFORM_TEX` (970),
  `CALENDAR_ORDER` (805), `row()` / `group()` / `colspec()`; park or delete in a later cleanup.

#### Open questions
1. **One editor per file.** The "table legibility" pass has already applied almost all of this section. The next
   editor should apply only the residual edits above (vendor, design headers, pooled-tree label and notes clause)
   and must not restore parked code.
2. **Vendors** (`[vendor]` in 6 cells and a notes sentence): the repo does not name them; the chain is OptionMetrics
   only per a code comment. Either the user names them, or the table drops the placeholders and keeps the `\todo` in
   03_data.tex:54. `TABLE_FIXES_FROM_FLOATS_2026-10-15.md` says "leave them and list them", which conflicts with the
   brief; the coordinator should pick one.
3. **Pooled trees** (settled by the consolidation edit: one fixed configuration per method, i.e. the bank arms
   `tree_expert_00` = LightGBM and `tree_expert_16` = XGBoost, plus `results/pooled_trees_2026-10-07` for the
   forest). Still to check: which pooled tree is re-estimated only every 10th interval (the notes' "part of them").
4. **Random forest "trial 255 of 400":** confirm it is the best (rank-1) trial of the pre-sample search, so the notes
   can say "the best trial" for all three models. 05_methods.tex:340 also writes "trial 255 of 400".
5. **Unreferenced floats:** add "(Table~\ref{tab:rh_dm_hyper})", "(Table~\ref{tab:rh_dm_hyper_search})" and
   "(Table~\ref{tab:rh_dm_tuning})" to the bullets under `sec:rh_methods_hyper` / `sec:rh_methods_tuning` in
   05_methods.tex, or move `tab:rh_dm_hyper_search` to the appendix.
6. **No `\label` on Data / Predictors:** notes say "see the Data and Predictors sections" in words. New labels
   (`sec:rh_data`, `sec:rh_predictors`; additions, not renames) would allow `\ref`.
7. **Pre-fix data chain:** the run pins the pre-2026-10-15 chain (`DOCUMENT_PRE_FIX_CHAIN = True`, 3128;
   the data-audit fix flags are switched off for this run). The dictionary's signed-$\mathrm{Spread}^{\mathrm{VW}}$
   sentence, the asinh routing and the FOMC dummies' $\log v$ describe that chain. They change when the results are
   re-run on the fixed chain, at the same time as the QLIKE re-scoring.

### C.4 `make_rehaul_appendix.py`

Writes (all under `writeup/generated/`): `rehaul_app_tune_ladder.tex`, `rehaul_app_tune_cadence.tex`,
`rehaul_app_tune_mask.tex`, `rehaul_app_tune_tuneper.tex`, `rehaul_app_tune_budget.tex`,
`rehaul_app_tune_budget_oos.tex`, `rehaul_app_tune_optuna_vs.tex`, `rehaul_app_tune_picks.tex`,
`rehaul_app_featimp_harma.tex`, `rehaul_app_mingru_dev.tex`, `rehaul_app_mingru_all.tex`,
`rehaul_app_mingru_dayclass.tex`, `rehaul_app_numbers.tex` (`\rhapp...` macros). Wrappers: `rehaul/10_appendix_floats.tex`
inputs eleven tables plus the macro file; `rehaul/08_feature_importance_floats.tex:120-153` inputs `featimp_harma`.

- **What the generator writes, and where the rest lives:**
  - It writes only `tabular` bodies: headers, panel titles, row labels and cells. Every caption, every note and the
    `\small` / `\setlength{\tabcolsep}` settings live in the wrappers.
- **The eight `tune_*` bodies and `featimp_harma` copy their cells verbatim from other generated files.**
  - The source files are `generated/table_close_ladder.tex` (from `make_table_close_main.py`) and
    `generated/appendix_campaign_16h_{cadence,mask,tuneper,budget,budget_oos,optuna_vs,picks,featimp}.tex` (from
    `make_appendix_campaign_16h_tex.py`).
  - Their row labels and cells (`\texttt{num\_leaves}`, "shipped settings", `\emph{all}`, `---`) come in from those
    files. Remap them inside this generator, as `LADDER_LABEL`, `relabel()` and `_space()` already do. Keep the
    parse-side keys (upstream strings, `DESIGN_TT`, `DASH` for parsing, the `LADDER_PANEL` keys) unchanged, or the
    asserts fail.
- **BLOCKER (verified 2026-10-15): the generator does not run now.**
  - Importing it fails with an `AssertionError` at line 647, called from line 661 (CHAIN (2)).
  - Cause: `walk_softplus_p100`, `walk_identity_p100`, `wd_study`, `qlike_study` and `pooled_rinit_study` scores
    were rewritten on 2026-10-15 at 14:18-14:23, apparently the re-scoring against the actual $\RV_t$.
    `wd_enc_study`, `lin_study`, `ridge_init_study` and `pooled_wd_study` are still dated 2026-10-14.
  - The elastic-net reference no longer matches the contrasts. On disk, the softplus specification now scores
    0.1108 / 0.1027; the assert holds 0.1082 / 0.1002.
  - Fix: re-score every study first. Then regenerate the hard-coded asserts from the new outputs: `KNOWN` (670-677),
    the `POOLED_RINIT` checks (726-729) and the `ALLBARS` checks (761-762).
  - The `tune_*` sources are dated 2026-10-14 too. Re-run `make_table_close_main.py` and
    `make_appendix_campaign_16h_tex.py` before this generator.

#### Tables

**Use in the text (grep of `rehaul/*.tex` for each `\ref`):**
- No section text (`rehaul/01`-`09`, `10_appendix.tex`) references any of the 12 tables or `fig:rh_app_murphy`.
- The only live references are between floats:
  - the mask, tuneper and budget_oos notes point to `tab:rh_app_tune_cadence`;
  - the mingru_all and dayclass notes point to `tab:rh_app_mingru_dev`.
- The outline `10_appendix.tex` names the topics but cites no table:
  - item B: "fixed hyperparameters, Optuna budgets, selected trials", "re-selection frequency ... re-estimation
    frequency", the network grid;
  - item D: "Murphy diagrams".
- Default recommendation (the user decides; labels are never renamed; a parked float takes its label with it; no `??`
  results because nothing in the text cites these tables):

| Label | Recommendation | Why |
|---|---|---|
| `tab:rh_app_tune_ladder` | **Park** (or keep Panels A-D only) | Panels F-H repeat the cadence, mask and tuneper cells (LightGBM, full set); B-D repeat the main T1/T2/T3 contrasts; E is a no-op check (+0.0, 100% same position) |
| `tab:rh_app_tune_cadence` | **Keep; merge `tab:rh_app_tune_mask` into it** as extra panels (5-column layout) | the other tables' notes point here; it supports Appendix B, "re-estimation frequency" |
| `tab:rh_app_tune_mask` | **Merge** into cadence (park its float) | 9 columns; same cells as cadence |
| `tab:rh_app_tune_tuneper` | **Keep; merge `tab:rh_app_tune_budget_oos` into it** (5-column layout) | supports Appendix B, "re-selection frequency" and "Optuna budgets" |
| `tab:rh_app_tune_budget_oos` | **Merge** into tuneper (park its float) | 9 columns |
| `tab:rh_app_tune_budget` | Keep | Appendix B, "Optuna budgets" (in-sample) |
| `tab:rh_app_tune_optuna_vs` | Keep | the only tuning table with significance counts |
| `tab:rh_app_tune_picks` | Keep | Appendix B, "selected trials" |
| `tab:rh_app_mingru_dev` | Keep; add a `\ref` from the 06_results MinGRU bullet and from Section~\ref{sec:rh_methods_models} (text editors) | development path to the reported specification (6) |
| `tab:rh_app_mingru_all` | **Park** | a 35-row listing of every variant; nothing in the text uses it (core section 8: such listings are parked) |
| `tab:rh_app_mingru_dayclass` | **Park** | it diagnoses the full-set loss of specifications (1)-(4); for the reported (6) the full set is better (-0.0037) |
| `tab:rh_app_fi_harma` | **Park** | 3 of its 5 measures are MDI-based, and 08_feature_importance.tex:218-219 says "MDI is not used"; it is a before/after of the earlier study (permutation P2, also parked) |

- **If the merges are done:**
  - Merged table 1 is cadence plus mask, 17 rows in 7 panels. Merged table 2 is tuneper plus budget_oos, 21 rows in
    7 panels.
  - Each fits on one page at `\small`. A single merge of all four would run to about 52 rows and overflow the page.
- **If the merges are not done:** give each of the four the same 5-column layout below.
- **Lags:** every wrapper note that says "Newey--West lag $q = \fqsDMLag$" or "$q = \fqsHACLag$" (wrapper 10:48, 51,
  227, 228, 303) must say "a Newey--West variance with \fqsDMLag{} lags" / "(Newey--West, \fqsHACLag{} lags)". This
  is a cross-cutting rule: no $q$ in notes.
- **Method pointers:** "see Methods (\emph{Tuning procedures})" and the like can now be real references. The labels
  exist: `sec:rh_methods_setup`, `sec:rh_methods_models`, `sec:rh_methods_hyper`, `sec:rh_methods_tuning`,
  `sec:rh_methods_tests` (05_methods.tex:22-414).

**Shared 5-column layout** (cadence, mask, tuneper, budget_oos):
- Columns: Model | HAR benchmark: $\Delta$QLIKE (\%), $\Delta\SR$ | Full predictor set: $\Delta$QLIKE (\%), $\Delta\SR$.
- Header code:
  `& \multicolumn{2}{c}{HAR benchmark} & \multicolumn{2}{c}{Full predictor set} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}`
  then `Model & $\Delta$QLIKE (\%) & $\Delta\SR$ & $\Delta$QLIKE (\%) & $\Delta\SR$ \\`. Colspec `@{}l*{4}{r}@{}`.
- Panels: one per comparison ("Panel A: ..."), rows LightGBM, XGBoost, Random forest.
- Implementation:
  - Replace `design_panels` (262-270) with a transposing helper.
  - For comparison $g$ and model $m$, the row is `data["baseline"][m][2g:2g+2] + data["all_features"][m][2g:2g+2]`.
  - Drop a model's row in a panel when all four of its cells are dashes.
- Significance: the cells are point estimates. Stars are possible from
  `generated/appendix_campaign_16h_plain_pairs.csv` (columns `dm_p`, `t_p`; sources `H mask_pairs`,
  `C tuneper_oos`, `A change_by_arm`). Otherwise keep "point estimates" in the notes (open question 4).

##### `tab:rh_app_tune_ladder` (generated/rehaul_app_tune_ladder.tex; generator lines 183-186, 471-520)
- **Problems vs core section 8:**
  - Row labels are code jargon. Some come from upstream, some from `LADDER_LABEL`: "block-diag.\ ridge", "pooled
    twin", "per-bar", "shipped", "refit every 10" (no unit), "one penalty (campaign's)", "16:00 bar", "the paper's
    forecast" (= the earlier draft), and lower-case "lasso", "elastic net", "random forest".
  - "shipped settings" survives in Panels G-H because no `LADDER_LABEL` pair covers it.
  - Panel E's title has a typed "12" (`\rhappNsessionEdge`) and the jargon "session-edge duplicates".
  - The panel titles have no "Panel A:" prefix; there are 8 panels and 22 rows.
  - Headers: "$\Delta$\%" has no unit form; "DM" should be `$\DM$`; "$\Delta$, mid" / "$\Delta$, crossed" should
    read bid/ask; "Same / side (\%)" is split over two header rows.
  - No significance markers: DM and HAC $t$ are plain numbers.
  - Empty cells are "---", not "--". Plus signs are text ("+0.16", from upstream) beside `\ensuremath{-}`.
  - "Same side" is an integer % (other % columns: 1 decimal).
  - 7 columns, which is OK. The wrapper sets `\tabcolsep` to 3.5pt (wrapper 10:25); it is not needed after the
    shorter labels.
  - Caption and notes are in the wrapper (10:32, 44-52). The notes use $q$ for the lags.
- **Proposed layout** (if kept):
  - Caption: "Decomposition of the closing-interval results" (wrapper; keep).
  - Columns (7): Change | QLIKE: $\Delta$ (\%), $\DM$ | Sharpe ratio: $\Delta\SR$ mid, $t$, $\Delta\SR$ bid--ask |
    Same position (\%); `\cmidrule(lr){2-3}\cmidrule(lr){4-6}`. Stars on $\DM$ and $t$, from $|{\rm stat}| \ge$ 1.645 /
    1.960 / 2.576, the two-sided normal convention of the results tables.
  - Panels / rows:
    - A: "From the earlier draft's model to the closing-interval ridge" (1 row);
    - B: "Ridge: pooled intraday $\to$ closing-interval" (HAR benchmark; Full predictor set);
    - C: "Closing-interval ridge: predictor set" (1 row);
    - D: "Full predictor set: closing-interval ridge $\to$ other model" (8 rows).
  - Notes (<= 5 sentences): "Each row is the change from the first to the second forecast (second minus first), over
    the \rhappNdays{} trading days of the out-of-sample period. $\Delta$: $100\,(\overline{\QLIKE}_{2}/
    \overline{\QLIKE}_{1} - 1)$; $\DM$: Diebold--Mariano statistic of the daily QLIKE loss differential, with the
    Harvey--Leybourne--Newbold correction and a Newey--West variance with \fqsDMLag{} lags (--: not stored).
    $\Delta\SR$: change in the annualized Sharpe ratio of the sign strategy at mid-quotes and at quoted bid/ask
    prices; $t$: HAC $t$-statistic on the daily return differential at mid-quotes (Newey--West, \fqsHACLag{} lags).
    $^{*}$, $^{**}$, $^{***}$: significant at the 10\%, 5\% and 1\% levels. Same position: share of trading days on
    which the two forecasts imply the same position $q_t$; the earlier draft's model is the block-penalized ridge
    estimated on all intraday intervals (Section~\ref{sec:rh_methods_models})."
  - Moves / parked:
    - Panel E becomes one sentence in Appendix A (dropping the duplicated interactions leaves the ridge forecast
      unchanged).
    - Panels F-H are parked: they repeat the merged cadence/mask and tuneper cells.
    - Default for the whole table: park.
- **String edits:**

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 473-474 | `r"block-diag.\ ridge $\to$ pooled ridge, all features"` | `r"Block-penalized ridge $\to$ ridge, pooled intraday, full predictor set"` |
| 476 | `r", shipped, refit every 10"` | `r", fixed hyperparameters, every 10 days"` |
| 477 | `r", random search, refit every 10"` | `r", random search, every 10 days"` |
| 478 | `r", shipped, refit daily"` | `r", fixed hyperparameters, daily"` |
| 479 | `r", random search, refit daily"` | `r", random search, daily"` |
| 482 | `r"per-bar ridge, one penalty (campaign's)"` | `r"Closing-interval ridge, single penalty"` (or drop with Panel E) |
| 484 (add pairs, applied last) | upstream `shipped settings`; `Optuna, tuned daily`; `Optuna: tuned every 250 sessions $\to$ daily`; `\quad lasso`; `\quad elastic net`; `\quad random forest` | `fixed hyperparameters`; `Optuna, re-tuned daily`; `Optuna, re-tuned every 250 trading days $\to$ daily`; `\quad Lasso`; `\quad Elastic net`; `\quad Random forest` |
| 486 (value) | `From the paper's forecast to the per-bar specification` | `Panel A: From the earlier draft's model to the closing-interval ridge` |
| 487 (value) | `Fit the 16:00 bar alone: pooled twin $\to$ per-bar ridge` | `Panel B: Ridge, pooled intraday $\to$ closing-interval` |
| 488 (value) | `Inputs of the per-bar ridge` | `Panel C: Closing-interval ridge, predictor set` |
| 489 (value) | `Estimator on all features: per-bar ridge $\to$ alternative` | `Panel D: Full predictor set, closing-interval ridge $\to$ other model` |
| 490 (value) | `Campaign, design: the 12 session-edge duplicates dropped` | park panel (if kept: `Panel E: Dropping the \rhappNsessionEdge{} duplicated HAR interactions`) |
| 491 (value) | `Campaign, per-window mask: masked minus unmasked` | park (if kept: `Within-window screening (LightGBM): screened minus unscreened`) |
| 492 (value) | `Campaign, refit cadence: every 10 sessions $\to$ every session` | park (if kept: `Re-estimation: every 10 trading days $\to$ daily`) |
| 493 (value) | `Campaign, tuning (LightGBM, refit daily)` | park (if kept: `Tuning (LightGBM, re-estimated daily)`) |
| 184-186 | `relabel`: `"HAR + calendar"`, `"all features"` | `"HAR benchmark"`, `"full predictor set"`; map `\quad \texttt{all\_features}` first to `\quad Full predictor set` (row start) |
| 512 | cells copied verbatim (`c[1:]`) | star `c[2]` (DM) and `c[4]` ($t$); `---` -> `--`; text `+` -> `\ensuremath{+}` |
| 513 | `& \multicolumn{3}{c}{Sharpe ratio} & Same \\` | `& \multicolumn{3}{c}{Sharpe ratio} & Same \\` (keep "Same" as the top half of the "Same position" header) |
| 515 | `Step (to minus from) & $\Delta$\% & DM & $\Delta$, mid & HAC $t$ & $\Delta$, crossed & side (\%)` | `Change & $\Delta$ (\%) & $\DM$ & $\Delta\SR$, mid & $t$ & $\Delta\SR$, bid--ask & position (\%)` |
| wrapper 10:44-52 | notes ("$q = \fqsDMLag$", "Same side", "see Methods (\emph{Tuning procedures})") | the proposed notes above |
| wrapper 10:25 | `\setlength{\tabcolsep}{3.5pt}` | delete |

##### `tab:rh_app_tune_cadence` (generated/rehaul_app_tune_cadence.tex; generator lines 44, 218-230, 262-280)
- **Problems vs core section 8:**
  - Group headers are code names: "T1 $-$ T10", "RS1 $-$ RS10", "RS1 $-$ T1" (275).
  - The sub-header "QLIKE" sits over a % difference, with no "$\Delta$" and no unit. "$\Delta$SR" should be
    `$\Delta\SR$` (219).
  - Panels are titled "HAR + calendar" / "All features" (`DLABEL`, 44), with no "Panel A:".
  - Point estimates only: no test, no stars.
  - 7 columns, which is OK. The wrapper notes (10:79-85) explain the code names T1/T10/RS1/RS10 and must be rewritten.
  - Caption and notes are in the wrapper (10:67, 79-85).
- **Proposed layout** (5 columns, shared layout; merged with mask):
  - Caption: "Re-estimation frequency, random search and column screening" (wrapper; was "Re-estimation frequency
    and random search").
  - Columns (5): Model | HAR benchmark: $\Delta$QLIKE (\%), $\Delta\SR$ | Full predictor set: $\Delta$QLIKE (\%),
    $\Delta\SR$.
  - Panels / rows (3 rows each):
    - A: "Fixed hyperparameters: re-estimated daily minus every 10 trading days";
    - B: "Random search: re-estimated daily minus every 10 trading days";
    - C: "Re-estimated daily: random search minus fixed hyperparameters";
    - D-G: the mask panels (see below).
  - Notes (<= 5 sentences): "Differences between specifications of the closing-interval tree models over the
    \rhappNdays{} trading days of the out-of-sample period; point estimates. For a panel `A minus B', $\Delta$QLIKE
    is $100\,(\overline{\QLIKE}_A/\overline{\QLIKE}_B - 1)$ and $\Delta\SR$ the difference in the annualized Sharpe
    ratio of the sign strategy at mid-quotes. HAR benchmark: \rhappNcolsBase{} predictors; full predictor set:
    \rhappNcolsAll{}. Panels D--G compare forecasts with and without the within-window screen, which drops
    predictors that are constant on the estimation window or copies of an earlier column
    (Section~\ref{sec:rh_methods_models}); --: XGBoost and the random forest were not estimated without the screen
    at their current hyperparameters. Tuning procedures: Section~\ref{sec:rh_methods_tuning}."
  - Moves: none; the T1/RS1 legend in the notes is no longer needed.
- **String edits:**

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 44 | `DLABEL = {"baseline": "HAR + calendar", "all_features": "All features"}` | `{"baseline": "HAR benchmark", "all_features": "Full predictor set"}` (also add `DSHORT = {"baseline": "HAR", "all_features": "Full"}` for the picks table) |
| 219 | `sub=("QLIKE", r"$\Delta$SR")` | `sub=(r"$\Delta$QLIKE (\%)", r"$\Delta\SR$")` |
| 275 | `pair_header([r"T1 $-$ T10", r"RS1 $-$ RS10", r"RS1 $-$ T1"], "Model")` | header = the HAR benchmark / Full predictor set groups; the three comparisons become panel titles A-C (text above) |
| 267 | `panel_line(DLABEL[d], ncol)` | the transposing helper: `panel_line(f"Panel {L}: {title}", 5)` |
| 278 | `r"@{}l*{6}{r}@{}"` | `r"@{}l*{4}{r}@{}"` |
| wrapper 10:67 | `\caption{Re-estimation frequency and random search}` | `\caption{Re-estimation frequency, random search and column screening}` (if merged) |
| wrapper 10:79-85 | `T1 $-$ T10: fixed hyperparameters, ...` | the proposed notes |

##### `tab:rh_app_tune_mask` (generated/rehaul_app_tune_mask.tex; generator lines 252-259, 283-290)
- **Problems vs core section 8:**
  - 9 columns (288).
  - Code-name groups "T10", "T1", "RS10", "RS1" (285).
  - "QLIKE" / "$\Delta$SR" sub-headers (219).
  - `\multicolumn{2}{c}{---}` cells (256), with "---" instead of "--".
  - The wrapper narrows `\tabcolsep` to 5pt (10:94).
  - Caption and notes are in the wrapper (10:100, 110-113). The notes point to the cadence table for cell
    definitions.
- **Proposed layout:** merged into `tab:rh_app_tune_cadence` as panels D-G (park this float, `10:90-116`). Standalone
  fallback: the same 5 columns with Panels A-D.
  - Caption: (fallback) "Within-window column screening" (wrapper; keep).
  - Columns (5): Model | HAR benchmark: $\Delta$QLIKE (\%), $\Delta\SR$ | Full predictor set: $\Delta$QLIKE (\%),
    $\Delta\SR$.
  - Panels / rows (all "screened minus unscreened"):
    - D: "Fixed hyperparameters, re-estimated every 10 trading days" (3 rows);
    - E: "Fixed hyperparameters, re-estimated daily" (3);
    - F: "Random search, re-estimated every 10 trading days" (LightGBM only);
    - G: "Random search, re-estimated daily" (LightGBM only).
    - Drop the all-dash XGBoost and random-forest rows; the note explains.
  - Notes: folded into the cadence notes (above). Fallback (<= 5 sentences): "Screened minus unscreened forecasts of
    the closing-interval tree models over the \rhappNdays{} trading days of the out-of-sample period; point
    estimates. At every estimation, the screen drops predictors that are constant on the estimation window or
    copies of an earlier column (Section~\ref{sec:rh_methods_models}). Cells as in
    Table~\ref{tab:rh_app_tune_cadence}. XGBoost and the random forest were not estimated without the screen at
    their current hyperparameters. The unscreened runs were not restricted to a single CPU class (Appendix E)."
  - Moves: the CPU-class remark moves to Appendix E (reproducibility).
- **String edits:**

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 285 | `pair_header(["T10", "T1", "RS10", "RS1"], "Model")` | the 5-column header; the groups become panel titles D-G |
| 288 | `r"@{}l*{8}{r}@{}"` | `r"@{}l*{4}{r}@{}"` (or write into the cadence file) |
| 256 | `rf"\multicolumn{{2}}{{c}}{{{DASH}}}"` | output `--` per cell (keep `DASH = "---"` for parsing upstream; add `OUT_DASH = "--"`) |
| wrapper 10:94 | `\setlength{\tabcolsep}{5pt}` | delete |
| wrapper 10:110-113 | notes | fold into the cadence notes (or the fallback above) |

##### `tab:rh_app_tune_tuneper` (generated/rehaul_app_tune_tuneper.tex; generator lines 307-327)
- **Problems vs core section 8:**
  - 8 columns (325).
  - Group headers "1 vs 250", "5 vs 250", "25 vs 250" (311): terse and unit-less.
  - "QLIKE, 250" (319) is a level column inside a table of differences.
  - "QLIKE" over a % difference (320); "$\Delta$SR" (320).
  - Panels are `DLABEL`.
  - Point estimates only.
  - Caption and notes are in the wrapper (10:127, 137-141). The notes' "QLIKE, 250" sentence goes with the column.
- **Proposed layout** (5 columns; merged with budget_oos):
  - Caption: "Hyperparameter re-tuning frequency and trial budget" (wrapper; was "Hyperparameter re-tuning
    frequency").
  - Columns (5): Model | HAR benchmark: $\Delta$QLIKE (\%), $\Delta\SR$ | Full predictor set: $\Delta$QLIKE (\%),
    $\Delta\SR$.
  - Panels / rows (3 rows each):
    - A: "Optuna re-tuned every trading day minus every 250 trading days";
    - B: "... every 5 trading days minus every 250";
    - C: "... every 25 trading days minus every 250";
    - D-G: the budget_oos panels (see below).
  - Notes (<= 5 sentences): "Differences between Optuna specifications of the closing-interval tree models over the
    \rhappNdays{} trading days of the out-of-sample period; point estimates; every specification is re-estimated
    daily. At each tuning date, a tree-structured Parzen estimator study runs \rhappOptTrials{} trials, starting from
    the fixed hyperparameters, and keeps the configuration with the lowest validation MSE
    (Section~\ref{sec:rh_methods_tuning}). Panels A--C vary the re-tuning frequency; panels D--G compare the best of
    \rhappOptTrials{} trials with the best of the first 10 or 25. Cells as in Table~\ref{tab:rh_app_tune_cadence}.
    The mean QLIKE of every specification is in Table~\ref{tab:rh_rank_full}."
  - Moves: the "QLIKE, 250" level column moves out; the levels are in `tab:rh_rank_full`.
- **String edits:**

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 309 | `lambda c: [c[0], *stacked_cells(c[1:])]` | drop `c[0]` (the level) |
| 311 | `groups = [r"1 vs 250", r"5 vs 250", r"25 vs 250"]` | panel titles A-C (text above) |
| 312-322 | `top`, `rules`, `bottom = r"Model & QLIKE, 250 & " + ...` | the shared 5-column header |
| 325 | `r"@{}l*{7}{r}@{}"` | `r"@{}l*{4}{r}@{}"` |
| wrapper 10:127 | `\caption{Hyperparameter re-tuning frequency}` | `\caption{Hyperparameter re-tuning frequency and trial budget}` (if merged) |
| wrapper 10:137-141 | notes incl. "QLIKE, 250: the mean QLIKE of ..." | the proposed notes |

##### `tab:rh_app_tune_budget` (generated/rehaul_app_tune_budget.tex; generator lines 330-356)
- **Problems vs core section 8:**
  - 8 columns with 3 header rows.
  - Typed "50" in "Best of 50 trials" and "Val.\ MSE gain to 50 trials (\%)" (337); it should be `\rhappOptTrials`.
  - "shipped" (342) should be "fixed". "in 41--50" (342) has typed numbers. "Pick / changes" (338, 344) is jargon.
    "Val." is abbreviated.
  - Cells hold two numbers each ("5.22\,/\,4.42", mean / median, 334) in `c` columns (353). These % cells have 2
    decimals; other % columns have 1 or 0 ("shipped" 1, "Pick changes" 0).
  - Panels are `DLABEL`, with no "Panel A:".
  - Caption and notes are in the wrapper (10:157, 169-175). The notes say "shipped:" and "in 41--50".
  - Period: 'over all \rhappOptPoints{} (= 1,469) tuning dates'. This exceeds the 866 out-of-sample days, so the
    notes should state the tuning period (open question 3).
  - The wrapper narrows `\tabcolsep` to 4pt.
- **Proposed layout:**
  - Caption: "Adequacy of the Optuna trial budget" (wrapper; keep).
  - Columns (8, all short numerics): Model | Best of \rhappOptTrials{} trials: Median trial, Fixed (\%), Last ten
    (\%) | Validation MSE reduction (\%): From 10, From 25, From 40 | Changed selection (\%);
    `\cmidrule(lr){2-4}\cmidrule(lr){5-7}`.
    - Option A (the guide's ~7 limit): show the means only, which removes the "a / b" cells.
    - Option B: keep "mean\,/\,median" cells (current) and use `r` columns.
  - Panels / rows: "Panel A: HAR benchmark", "Panel B: Full predictor set"; LightGBM, XGBoost, Random forest.
  - Notes (<= 5 sentences): "Position of the best of \rhappOptTrials{} Optuna trials, and the gain from the last
    trials, over the \rhappOptPoints{} tuning dates of each specification [state the period]. Median trial: the
    median index of the trial that attains the best of \rhappOptTrials{}; Fixed: the share of tuning dates at which
    that trial is the fixed-hyperparameter configuration (trial 0); Last ten: the share at which it is among the last
    ten trials (\rhappOptExch\% if its position were exchangeable). Validation MSE reduction: the relative reduction
    in the best validation MSE from the first 10, 25 or 40 trials to all \rhappOptTrials{}, mean [\,/\,median] across
    tuning dates. Changed selection: the share of tuning dates at which the best of the first 25 trials differs from
    the best of \rhappOptTrials{}. Tuning procedure: Section~\ref{sec:rh_methods_tuning}."
  - Moves: none.
- **String edits:**

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 337 | `\multicolumn{3}{c}{Best of 50 trials}` | `\multicolumn{3}{c}{Best of \rhappOptTrials{} trials}` |
| 337 | `\multicolumn{3}{c}{Val.\ MSE gain to 50 trials (\%)}` | `\multicolumn{3}{c}{Validation MSE reduction (\%)}` |
| 338 | `\multicolumn{1}{c}{Pick}` | `\multicolumn{1}{c}{Changed}` |
| 342 | `{median}` / `{shipped}` / `{in 41--50}` | `{Median}` / `{Fixed}` / `{Last ten}` |
| 343 | `{from 10}` / `{from 25}` / `{from 40}` | `{From 10}` / `{From 25}` / `{From 40}` |
| 344 | `{changes}` | `{selection}` |
| 346 | `{trial}` `{(\%)}` `{(\%)}` | keep |
| 347 | `{mean\,/\,med.}` x3 | Option A: `{(mean)}`; Option B: `{mean\,/\,median}` |
| 348 | `{(\%)}` | keep |
| 334 | `x.replace(" / ", "\\,/\\,")` | Option A: keep the part before " / " |
| 353 | `r"@{}l*{3}{r}*{3}{c}r@{}"` | `r"@{}l*{7}{r}@{}"` |
| 267 | `panel_line(DLABEL[d], ncol)` | prefix "Panel A: " / "Panel B: " |
| wrapper 10:150 | `\setlength{\tabcolsep}{4pt}` | delete (Option A) |
| wrapper 10:169-175 | notes ("shipped: ...", "in 41--50", "Pick changes") | the proposed notes |

##### `tab:rh_app_tune_budget_oos` (generated/rehaul_app_tune_budget_oos.tex; generator lines 359-380)
- **Problems vs core section 8:**
  - 9 columns (375) with 3 header rows.
  - "Tuned every session" / "Tuned every 25 sessions" (361): "sessions" should be "trading days"; "tuned" should be
    "re-tuned".
  - "50 vs 10" / "50 vs 25" (366) have typed 50s. "QLIKE" / "$\Delta$SR" (371).
  - The wrapper narrows `\tabcolsep` to 5pt (10:184).
  - Caption and notes are in the wrapper (10:190, 198-200).
- **Proposed layout:** merged into `tab:rh_app_tune_tuneper` as panels D-G (park this float, `10:180-203`). Standalone
  fallback: the same 5 columns with Panels A-D.
  - Caption: (fallback) "Optuna trial budget, out of sample" (wrapper; keep).
  - Columns (5): Model | HAR benchmark: $\Delta$QLIKE (\%), $\Delta\SR$ | Full predictor set: $\Delta$QLIKE (\%),
    $\Delta\SR$.
  - Panels / rows (3 rows each):
    - D: "Re-tuned daily: best of \rhappOptTrials{} minus best of the first 10 trials";
    - E: "Re-tuned daily: best of \rhappOptTrials{} minus best of the first 25 trials";
    - F: "Re-tuned every 25 trading days: best of \rhappOptTrials{} minus best of the first 10";
    - G: "Re-tuned every 25 trading days: best of \rhappOptTrials{} minus best of the first 25".
  - Notes: folded into the tuneper notes.
  - Moves: none.
- **String edits:**

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 361 | `\multicolumn{4}{c}{Tuned every session} & \multicolumn{4}{c}{Tuned every 25 sessions}` | drop; panel titles D-G (text above) |
| 366 | `["50 vs 10", "50 vs 25"] * 2` | drop; panel titles |
| 371 | `r"Model & " + "QLIKE & $\Delta$SR" x4` | the shared 5-column header |
| 375 | `r"@{}l*{8}{r}@{}"` | `r"@{}l*{4}{r}@{}"` |
| wrapper 10:184 | `\setlength{\tabcolsep}{5pt}` | delete |
| wrapper 10:198-200 | notes | fold into the tuneper notes |

##### `tab:rh_app_tune_optuna_vs` (generated/rehaul_app_tune_optuna_vs.tex; generator lines 383-422)
- **Problems vs core section 8:**
  - Code identifiers: "TUNE\_PER" and "Arms" (417); "T1" / "RS1", "shipped settings", "refit every session" and
    "per-bar ridge" in the panel titles (384-386).
  - `\emph{all}` row label (upstream; printed at 414).
  - "Sharpe, mid" (415); "below 0" / "above 0".
  - The "Arms" column is constant (6 per frequency, 24 in "all"): it belongs in the notes.
  - 6 columns, which is OK.
  - Caption and notes are in the wrapper (10:214, 224-229). The notes use $q$ for the lags.
- **Proposed layout:**
  - Caption: "Optuna against simpler tuning and the closing-interval ridge" (wrapper; keep).
  - Columns (5): Re-tuned every (trading days) | QLIKE: $<0$, $>0$ | $\Delta\SR$, mid-quotes: $<0$, $>0$;
    `\cmidrule(lr){2-3}\cmidrule(lr){4-5}`.
  - Panels / rows (rows 1, 5, 25, 250, All):
    - A: "Optuna minus fixed hyperparameters, both re-estimated daily";
    - B: "Optuna minus random search, both re-estimated daily";
    - C: "Optuna minus the closing-interval ridge".
  - Notes (<= 5 sentences): "Optuna (best of \rhappOptTrials{} trials) minus the comparator, for the three tree
    models on the two predictor sets, over the \rhappNdays{} trading days of the out-of-sample period. Each cell
    counts the specifications whose difference lies below or above zero; in parentheses, the number significant at
    the 5\% level. QLIKE: Diebold--Mariano test with the Harvey--Leybourne--Newbold correction and a Newey--West
    variance with \fqsDMLag{} lags; Sharpe ratio: HAC $t$-statistic on the daily return differential at
    mid-quotes (Newey--West, \fqsHACLag{} lags). All: the four re-tuning frequencies pooled."
  - Moves: the "Arms" column moves to the notes, in words ("three tree models, two predictor sets"), with no typed
    6/24.
- **String edits:**

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 384 | `"Optuna minus the shipped settings, refit every session (T1)"` | `"Panel A: Optuna minus fixed hyperparameters, both re-estimated daily"` |
| 385 | `"Optuna minus the random search, refit every session (RS1)"` | `"Panel B: Optuna minus random search, both re-estimated daily"` |
| 386 | `"Optuna minus the per-bar ridge"` | `"Panel C: Optuna minus the closing-interval ridge"` (the keys `"T1"`, `"RS1"`, `"per-bar ridge"` are the upstream parse keys: keep them) |
| 400 | `key, tp = order[-1], r"\emph{all}"` | `tp = "All"` (keep the `== r"\emph{all}"` test at 399) |
| 406 | `panels[key].append((tp, c[2:]))` | `c[3:]` (drops Arms) |
| 412 | `panel_line(OPT_VS[k], 6)` | `panel_line(OPT_VS[k], 5)` |
| 415 | `& & \multicolumn{2}{c}{QLIKE} & \multicolumn{2}{c}{Sharpe, mid}` | `& \multicolumn{2}{c}{QLIKE} & \multicolumn{2}{c}{$\Delta\SR$, mid-quotes}` |
| 416 | `\cmidrule(lr){3-4}\cmidrule(lr){5-6}` | `\cmidrule(lr){2-3}\cmidrule(lr){4-5}` |
| 417 | `TUNE\_PER & Arms & below 0 & above 0 & below 0 & above 0` | `Re-tuned every (days) & $<0$ & $>0$ & $<0$ & $>0$` |
| 420 | `r"@{}lrcccc@{}"` | `r"@{}lcccc@{}"` |
| wrapper 10:224-229 | notes ("$q = \fqsDMLag$", "The \emph{all} rows") | the proposed notes |

##### `tab:rh_app_tune_picks` (generated/rehaul_app_tune_picks.tex; generator lines 425-468)
- **Problems vs core section 8:**
  - 10 columns (466).
  - Hyperparameter names in `\texttt` (`num\_leaves`, `min\_child\_samples`, `feature\_fraction`,
    `bagging\_fraction`, `lambda\_l2`, `learning\_rate`, `max\_depth`, `min\_child\_weight`, `subsample`,
    `colsample\_bytree`, `reg\_lambda`, `max\_features`, `min\_samples\_leaf`): copied from upstream as `c[0]` at 455.
  - `\texttt{rounds\_cap}$^{\dagger}$` with range "rounds $\le$ cap(lr)" (426): "lr" is a code name.
  - Headers: "Axis", "H" / "A", "prior", "Median pick", "At the low end" / "At the high end" (459-463).
  - Range strings "log, int" (429-433).
  - Panel titles are bare model names (446).
  - The wrapper narrows `\tabcolsep` to 4pt (10:238).
  - Caption and notes are in the wrapper (10:244, 254-258). The notes define H/A and "pseudo-axis".
  - **Data conflict:** the random forest's max-depth set here is {3, 5, 10, 20, 40, none} (the run's own space). In
    `tab:rh_dm_hyper_search` it is {2, 4, 8, 16, 32, none}.
- **Proposed layout:**
  - Caption: "Optuna-selected hyperparameters within the search space" (wrapper; keep).
  - Columns (9): Hyperparameter | Lower end (\%): HAR, Full, Prior | Upper end (\%): HAR, Full, Prior | Median
    selection: HAR, Full; `\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}`.
    - "Search range" moves to `tab:rh_dm_hyper_search`, which lists the same spaces; resolve the random-forest
      depth conflict first.
    - With words for the names and short numerics, 9 columns fit portrait at `\small` (about 130pt label + 8 x
      40pt). Otherwise use `landscape` (pdflscape is loaded) and keep the range column (10).
  - Panels / rows: "Panel A: LightGBM" (7), "Panel B: XGBoost" (7), "Panel C: Random forest" (3).
  - Notes (<= 5 sentences): "Share of the best-of-\rhappOptTrials{} selections in the outermost twentieth of each
    hyperparameter's search range, at its lower and upper end, for the HAR benchmark (HAR) and the full predictor set
    (Full), beside the share of the random start-up draws in the same region (Prior), and the median selection.
    Search ranges: Table~\ref{tab:rh_dm_hyper_search}; each tree model's own search space
    (Section~\ref{sec:rh_methods_hyper}). $^{\dagger}$Not searched directly: the early-stopped number of boosting
    rounds $B$, capped at $\bar B(\eta)$, a function of the learning rate $\eta$."
  - Moves: the search range moves to `tab:rh_dm_hyper_search`.
- **String edits** (new dict `AXIS_WORDS`, applied to `c[0]` at 455; the keys are the upstream strings):

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 455 | `\texttt{num\_leaves}` | `Maximum number of leaves` |
| 455 | `\texttt{min\_child\_samples}` | `Minimum observations per leaf` |
| 455 | `\texttt{feature\_fraction}` | `Share of predictors per tree` |
| 455 | `\texttt{bagging\_fraction}` | `Share of observations per tree` |
| 455 | `\texttt{lambda\_l2}` | `$L_2$ penalty $\lambda_2^{\mathrm{tree}}$` |
| 455 | `\texttt{learning\_rate}` (LightGBM, XGBoost) | `Learning rate $\eta$` |
| 455 | `\texttt{rounds\_cap}$^{\dagger}$` | `Boosting rounds $B$$^{\dagger}$` |
| 455 | `\texttt{max\_depth}` (XGBoost, random forest) | `Maximum depth` |
| 455 | `\texttt{min\_child\_weight}` | `Minimum Hessian weight per leaf` |
| 455 | `\texttt{subsample}` | `Share of observations per tree` |
| 455 | `\texttt{colsample\_bytree}` | `Share of predictors per tree` |
| 455 | `\texttt{reg\_lambda}` | `$L_2$ penalty $\lambda_2^{\mathrm{tree}}$` |
| 455 | `\texttt{max\_features}` | `Share of predictors per split` |
| 455 | `\texttt{min\_samples\_leaf}` | `Minimum observations per leaf` |
| 426 | `r"rounds $\le$ cap(lr)"` | `r"$B \le \bar B(\eta)$"` (only if the range column stays) |
| 429-433 | `", log, int"`, `"], log"`, `"], int"` | `" (log, integer)"`, `"] (log)"`, `"] (integer)"` (only if the range column stays) |
| 446 | `panel_line(MODEL_CAP[r["title"]], 10)` | `panel_line(f"Panel {L}: {MODEL_CAP[...]}", 9)` |
| 455 | `rf"\quad {c[0]} & {_space(c[1])} & {lo[0]} & {lo[1]} & {c[4]} & ..."` | drop `_space(c[1])` (range column moved) |
| 459-460 | `{At the low end (\%)}`, `{At the high end (\%)}`, `{Median pick}` | `{Lower end (\%)}`, `{Upper end (\%)}`, `{Median selection}` |
| 462 | `\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}` | `\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}` |
| 463 | `Axis & Search range & H & A & prior & H & A & prior & H & A` | `Hyperparameter & HAR & Full & Prior & HAR & Full & Prior & HAR & Full` |
| 466 | `r"@{}ll*{8}{r}@{}"` | `r"@{}l*{8}{r}@{}"` |
| wrapper 10:238 | `\setlength{\tabcolsep}{4pt}` | delete (if 9 columns fit) |
| wrapper 10:254-258 | notes ("H: HAR benchmark; A: full predictor set", "axes", "pseudo-axis") | the proposed notes |

##### `tab:rh_app_mingru_dev` (generated/rehaul_app_mingru_dev.tex; generator lines 575-581, 659-666, 765-834)
- **Problems vs core section 8:**
  - 9 columns (826). There is a separate $p$ column (`fp`, 775), which must become stars on $\Delta$.
  - Headers: "Walk"; "$\Delta$\%" (unit form); "Sharpe" should be `$\SR$` (809-810).
  - Group headers are `DLABEL` (804).
  - Panel titles: "Per-bar walk (the 16:00 bar)" (817), "Pooled walk (every bar), read at 16:00" (819), "Per-bar
    linear references" (822).
  - Row labels carry code tags: "Walk of record", "(fix 1, B)", "(g8, B)", "(b)", British "Ridge-initialised" (660-665,
    820-821).
  - The tag "B" (per block) clashes with $B$, the number of boosting rounds (core section 3).
  - "(1) Walk of record" is no longer the reported specification; (6) is (QLIKE 0.0948 = the 06_results bullet).
  - Empty cells are "---" (792-793).
  - The wrapper narrows `\tabcolsep` to 3pt (10:276).
  - Caption and notes are in the wrapper (10:282, 300-311). The notes are long, describe (1)-(8) in full and use
    $q$ for the lag.
  - "mean QLIKE on the scale of $y_{t,i}$" (the `rhappAllbars*` macros, `qlike_fit`) is likely scored against the
    winsorized $\tilde y$. Re-check after the re-scoring (TABLE_FIXES "open").
  - The DM statistic exists only in the `walk_*` contrasts (`dm_qlike`), so stars must come from `p_qlike`.
- **Proposed layout:**
  - Caption: "Development of the closing-interval MinGRU network" (wrapper; keep).
  - Columns (7): Specification | HAR benchmark: QLIKE, $\Delta$ (\%), $\SR$ | Full predictor set: QLIKE, $\Delta$
    (\%), $\SR$; `\cmidrule(lr){2-4}\cmidrule(lr){5-7}`; stars on $\Delta$ from `p_qlike` (10/5/1\%).
  - Panels / rows:
    - A: "Closing-interval models" (1)-(6);
    - B: "Pooled intraday models, evaluated at the closing interval" (7)-(8);
    - C: "Closing-interval linear models" (Elastic net (comparator), Ridge).
  - Notes (<= 5 sentences): "Each row is a rolling-window out-of-sample exercise over the \rhappNdays{} trading days
    of the out-of-sample period. $\Delta$: $100\,(\overline{\QLIKE}/\overline{\QLIKE}^{\mathrm{EN}} - 1)$ relative
    to the closing-interval elastic net on the same predictor set; $^{*}$, $^{**}$, $^{***}$: Diebold--Mariano test
    (Harvey--Leybourne--Newbold correction, Newey--West variance with \fqsDMLag{} lags) significant at the 10\%, 5\%
    and 1\% levels; no test for the ridge. $\SR$: annualized Sharpe ratio of the sign strategy at mid-quotes.
    Specification (6), ridge-initialized and trained and selected on QLIKE, is the reported MinGRU network; each
    earlier row adds one change (Section~\ref{sec:rh_methods_models}), and "selected per block" means the added
    penalty is re-selected at each of the \rhappNblocks{} block starts. Over all intraday intervals, the mean QLIKE
    on the scale of $y_{t,i}$ is \rhappAllbarsRinitAll{} (full predictor set) and \rhappAllbarsRinitBase{} (HAR
    benchmark) for (8), against \rhappAllbarsWdBAll{} and \rhappAllbarsWdBBase{} for (7)."
  - Moves: the row-by-row description of (1)-(8) leaves the notes (the row labels carry it; the details go to
    Section~\ref{sec:rh_methods_models} / Appendix F).
- **String edits:**

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 660 | `"(1) Walk of record (softplus output)"` | `"(1) Softplus output, MSE"` |
| 661 | `"(2) Exogenous decay (fix 1, B)"` | `r"(2) + encoder weight decay $\nu_{\mathrm{enc}}$"` |
| 662 | `"(3) Linear shortcut (g8, B)"` | `"(3) + linear shortcut, zero-initialized"` |
| 663 | `"(4) Ridge-initialised shortcut (b), MSE"` | `"(4) + shortcut ridge-initialized"` |
| 664 | `"(5) (4), QLIKE selection"` | `"(5) (4), selected on QLIKE"` |
| 665 | `"(6) (4), QLIKE training and selection"` | `"(6) (4), trained and selected on QLIKE"` |
| 775 | `[fq(x["qlike"]), fpct(x["dpct"]), fp(x["p"]), fsr(x["sharpe"])]` | `[fq(...), fpct(x["dpct"]) + stars(x["p"]), fsr(...)]` (new `stars()`: `$^{***}$` p<0.01, `$^{**}$` <0.05, `$^{*}$` <0.10) |
| 772 | `\multicolumn{4}{c}{\emph{not run}}` | `\multicolumn{3}{c}{--}` (mingru_all only) |
| 778 | `\multicolumn{{8}}{{c}}{{\emph{{{pending}}}}}` | `\multicolumn{6}{c}{--}` (dead path; or delete) |
| 784 | `"Elastic net (the comparator)"` | `"Elastic net (comparator)"` |
| 792-793 | `DASH` (`---`) for the comparator's $\Delta$ and $p$ | `--` for $\Delta$; drop the $p$ cell |
| 804 | `DLABEL[d]` in `\multicolumn{4}` | `\multicolumn{3}{c}{DLABEL[d]}` (new DLABEL) |
| 807 | `\cmidrule(lr){2-5}\cmidrule(lr){6-9}` | `\cmidrule(lr){2-4}\cmidrule(lr){5-7}` |
| 809-810 | `r"Walk & " + "QLIKE & $\Delta$\% & $p$ & Sharpe"` | `r"Specification & " + r"QLIKE & $\Delta$ (\%) & $\SR$"` |
| 817 | `"Per-bar walk (the 16:00 bar)"`, ncol 9 | `"Panel A: Closing-interval models"`, 7 |
| 819 | `"Pooled walk (every bar), read at 16:00"` | `"Panel B: Pooled intraday models, evaluated at the closing interval"` |
| 820 | `r"\quad (7) Weight decay (B)"` | `r"\quad (7) Weight decay $\nu$, selected per block"` |
| 821 | `r"\quad (8) Ridge-initialised shortcut, QLIKE"` | `r"\quad (8) Ridge-initialized shortcut, trained and selected on QLIKE"` |
| 822 | `"Per-bar linear references"` | `"Panel C: Closing-interval linear models"` |
| 826 | `r"@{}l*{8}{r}@{}"` | `r"@{}l*{6}{r}@{}"` |
| 670-682 | `KNOWN = [(0.1082, 0.1002), ...]` | regenerate after the re-scoring (blocker) |
| wrapper 10:276 | `\setlength{\tabcolsep}{3pt}` | delete |
| wrapper 10:300-311 | notes ("$p$: $p$-value ...", "$q = \fqsDMLag$", "(1) softplus output ... B: the new penalty ...") | the proposed notes |

##### `tab:rh_app_mingru_all` (generated/rehaul_app_mingru_all.tex; generator lines 840-934)
- **Problems vs core section 8:**
  - 9 columns and 35 rows; a `[p]` float with `\addlinespace[2pt]` (919) to squeeze onto one page; the wrapper sets
    `\tabcolsep` to 3pt (10:319).
  - A $p$ column, as in mingru_dev.
  - Weight decay written $w$ / $w_{\mathrm{enc}}$; the guide uses $\nu$ / $\nu_{\mathrm{enc}}$.
  - Code tags "(fix 1)", "(g8)", "(b)", "(B)".
  - Jargon: "branch frozen", "check, identity link", "grid at", "= (n)".
  - Cells read `\emph{not run}` (772); the guide wants "--".
  - "Pooled walk (every bar), read at 16:00".
  - Nothing in the text uses the table. The live mingru_dev notes no longer point to it; only a parked comment does.
  - Caption and notes are in the wrapper (10:325, 338-343).
- **Proposed layout:**
  - Default: **park** the float (`10:316-346`) and `\rhappNvariants` stays defined but unused.
  - If kept:
    - Caption: "All estimated MinGRU specifications".
    - Columns (7): as mingru_dev.
    - Panels A-G as below, `--` for the HAR-benchmark cells not estimated, in a `longtable` or `[p]` at `\small`.
    - Notes (<= 5 sentences): "\rhappNvariants{} specifications, grouped by the modification each tests; columns,
      comparator and stars as in Table~\ref{tab:rh_app_mingru_dev}, whose rows are (1)--(8). $\nu$: weight decay on
      all weights; $\nu_{\mathrm{enc}}$: weight decay on the encoder's input columns only; a fixed value is held in
      every block, or the value is selected per block. Shortcut only: the recurrent branch $f^{\mathrm{NN}}$ is set to
      zero, so the forecast is the linear shortcut; checks: the shortcut held at the window's ridge estimate, with
      the identity or the softplus output. --: estimated on the full predictor set only."
- **String edits** (if kept):

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 843 | `"Output link"` | `"Panel A: Output link"` |
| 845 / 846 | `"identity"` / `"softplus = (1)"` | `"Identity"` / `"Softplus [row (1)]"` |
| 850 | `r"Uniform weight decay $w$ on every weight"` | `r"Panel B: Weight decay $\nu$ on all weights"` |
| 852 | `rf"$w = {w}$"` | `rf"$\nu = {w}$"` |
| 855 | `r"$w$ per block (B)"` | `r"$\nu$ selected per block"` |
| 858 | `r"Exogenous decay $w_{\mathrm{enc}}$ (fix 1)"` | `r"Panel C: Encoder weight decay $\nu_{\mathrm{enc}}$"` |
| 860 | `rf"$w_{{\mathrm{{enc}}}} = {w}$"` | `rf"$\nu_{{\mathrm{{enc}}}} = {w}$"` |
| 865 | `r"$w_{\mathrm{enc}}$ per block (B) = (2)"` | `r"$\nu_{\mathrm{enc}}$ selected per block [row (2)]"` |
| 871 | `"Linear shortcut, zero start (g8)"` | `"Panel D: Linear shortcut, zero-initialized"` |
| 873 | `r"shortcut alone, branch frozen"` | `r"Shortcut only ($f^{\mathrm{NN}} \equiv 0$)"` |
| 874 / 875 | `r"grid at $w_{\mathrm{enc}} = 0$"` / `... = 15$"` | `r"$\nu_{\mathrm{enc}} = 0$"` / `r"$\nu_{\mathrm{enc}} = 15$"` |
| 876 | `r"per block (B) = (3)"` | `r"Penalty selected per block [row (3)]"` |
| 880 | `"Shortcut started at the window's ridge (b)"` | `"Panel E: Linear shortcut, ridge-initialized"` |
| 882 / 883 | `r"check, identity link"` / `r"check, softplus link"` | `r"Check: ridge shortcut only, identity output"` / `r"Check: ridge shortcut only, softplus output"` |
| 885 | `r"grid at $w_{\mathrm{enc}} = 15$"` | `r"$\nu_{\mathrm{enc}} = 15$"` |
| 888 | `r"per block (B) = (4)"` | `r"Penalty selected per block [row (4)]"` |
| 892 | `"QLIKE in selection and training"` | `"Panel F: QLIKE in selection and training"` |
| 894-897 | `"(1), QLIKE selection"`, `"(2), QLIKE selection"`, `"(4), QLIKE selection = (5)"`, `"(4), QLIKE training = (6)"` | `"(1), selected on QLIKE"`, `"(2), selected on QLIKE"`, `"(4), selected on QLIKE [row (5)]"`, `"(4), trained and selected on QLIKE [row (6)]"` |
| 901 | `"Pooled walk (every bar), read at 16:00"` | `"Panel G: Pooled intraday models, evaluated at the closing interval"` |
| 903 | `rf"$w = {w}$"` | `rf"$\nu = {w}$"` |
| 907 | `r"$w$ per block (B) = (7)"` | `r"$\nu$ selected per block [row (7)]"` |
| 908 | `"ridge start, QLIKE = (8)"` | `"Ridge-initialized shortcut, QLIKE [row (8)]"` |
| 920 | `panel_line(title, 9)` | `panel_line(title, 7)` |
| 930 | `r"@{}l*{8}{r}@{}"` | `r"@{}l*{6}{r}@{}"` |
| wrapper 10:316-346 | the whole float | park (`% 2026-10-15 (...): parked -- exhaustive listing; nothing in the text uses it`) |

##### `tab:rh_app_mingru_dayclass` (generated/rehaul_app_mingru_dayclass.tex; generator lines 937-1052)
- **Problems vs core section 8:**
  - 5 columns, which is OK.
  - Header "Walk" (1028).
  - The day counts "($n = 43$)" etc. (1028-1036) are printed from data. The macros `\rhappNmonthEnd`,
    `\rhappNextreme` and `\rhappNother` exist (1045-1047) but are unused; use them, and `\rhappNdays`, in the header.
  - "Extreme inputs" (1027): the notes say "extreme predictor values".
  - Row labels "(1) Walk of record", "(B)", "(b)", "Ridge start" (954-964).
  - Panel titles "QLIKE, HAR + calendar", "QLIKE, all features", "All features minus HAR + calendar" (1013-1018).
  - The label "(1) Walk of record" is also a dict key at line 987: rename both.
  - Caption and notes are in the wrapper (10:430, 442-448). The notes are 5 sentences and OK.
  - The table diagnoses specification (1), not the reported (6).
- **Proposed layout:**
  - Default: **park** (`10:422-451`).
  - If kept:
    - Caption: "MinGRU accuracy by type of trading day".
    - Columns (5): Specification | Month-ends (\rhappNmonthEnd{} days) | Extreme predictors (\rhappNextreme) | Other
      days (\rhappNother) | All days (\rhappNdays).
    - Panels: "Panel A: HAR benchmark", "Panel B: Full predictor set", "Panel C: Full predictor set minus HAR
      benchmark".
    - Notes: the wrapper text, with "specification (1)" kept (it is a description, not "walk of record").
- **String edits** (if kept):

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 954 + 987 | `"(1) Walk of record"` (label and dict key) | `"(1) Softplus output, MSE"` (both places) |
| 958 | `"(2) Exogenous decay (B)"` | `r"(2) + encoder weight decay $\nu_{\mathrm{enc}}$"` |
| 961 | `"(3) Linear shortcut (B)"` | `"(3) + linear shortcut, zero-initialized"` |
| 962 | `"(4) Ridge start (b), MSE"` | `"(4) + shortcut ridge-initialized"` |
| 963 | `"(5) (4), QLIKE selection"` | `"(5) (4), selected on QLIKE"` |
| 964 | `"(6) (4), QLIKE training"` | `"(6) (4), trained and selected on QLIKE"` |
| 1013-1015 | `f"QLIKE, {DLABEL[d].lower() if d == 'all_features' else DLABEL[d]}"` | `f"Panel {'AB'[i]}: {DLABEL[d]}"` |
| 1018 | `"All features minus HAR + calendar"` | `"Panel C: Full predictor set minus HAR benchmark"` |
| 1027 | `& Month-ends & Extreme inputs & Other days & All days` | `& Month-ends & Extreme predictors & Other days & All days` |
| 1028-1036 | `r"Walk & ($n = " + str(int(n["month_end"])) + ...` | `r"Specification & (\rhappNmonthEnd{} days) & (\rhappNextreme{} days) & (\rhappNother{} days) & (\rhappNdays{} days) \\"` |

##### `tab:rh_app_fi_harma` (generated/rehaul_app_featimp_harma.tex; generator lines 523-541)
- **Problems vs core section 8:**
  - Three of the five measure columns rest on MDI ("MDI share", "MDI rank", and `\texttt{\_x\_close}` = a share of
    MDI). The core text (08_feature_importance.tex:218-219) says "MDI is not used; SHAP values and coefficients only".
  - "TreeSHAP" names an algorithm, not a measure; "Perm.\ P2" is a code name; `\texttt{\_x\_close}` is a code name.
  - Two numbers per cell ("57 $\to$ 52", old $\to$ new) in `c` columns (539).
  - The table compares the first-pass design with the current one (development history). Its permutation study is
    parked in section 8.
  - The wrapper notes still open with "Draft, measures under review" (08:142).
  - Caption and notes are in `08_feature_importance_floats.tex:130, 142-150`.
- **Proposed layout:**
  - Default: **park** the float (`08:120-153`).
  - If kept:
    - Caption: "Importance of the HAR components in the tree models".
    - Columns (5): Model | Mean absolute SHAP share (\%): First pass, Current | Permutation $\Delta$QLIKE (\%):
      First pass, Current; `\cmidrule(lr){2-3}\cmidrule(lr){4-5}`.
    - Drop the MDI share, the MDI rank and the close-interaction share (or, if the user wants it, label it "Share on
      $\bar y^{(k)}\cdot\mathrm{Close}$ (\%)", addenda A.1).
    - Rows: LightGBM, XGBoost, Random forest.
    - Notes (<= 5 sentences): "Closing-interval tree models on the full predictor set. First pass: the earlier
      design, with the \rhappNsessionEdge{} interactions $\bar y^{(k)}\cdot\mathrm{Open}$ and
      $\bar y^{(k)}\cdot\mathrm{Close}$ and no within-window screen; current: those interactions dropped and the
      screen applied. SHAP share: the HAR components' share of the total mean absolute SHAP value; permutation
      $\Delta$QLIKE: the increase in QLIKE when their values are replaced by rows drawn from the estimation window,
      in percent of the out-of-sample loss. XGBoost and the random forest also changed hyperparameters between the
      two passes."
- **String edits** (if kept):

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 531 | `rf"{MODEL_CAP[c[1]]} & " + " & ".join(c[2:])` | split `c[3]`, `c[4]` at `$\to$` into two cells each; drop `c[2]` (MDI share), `c[5]` (close share), `c[6]` (MDI rank) |
| 533 | `& MDI & TreeSHAP & Perm.\ P2 & \texttt{\_x\_close} & MDI` | `& \multicolumn{2}{c}{Mean absolute SHAP share (\%)} & \multicolumn{2}{c}{Permutation $\Delta$QLIKE (\%)}` + `\cmidrule(lr){2-3}\cmidrule(lr){4-5}` |
| 535 | `Model & share (\%) & share (\%) & $\Delta$QLIKE (\%) & share (\%) & rank` | `Model & First pass & Current & First pass & Current` |
| 539 | `r"@{}lccccc@{}"` | `r"@{}l*{4}{r}@{}"` |
| wrapper 08:120-153 | the whole float | park (`% 2026-10-15 (...): parked -- MDI not used (Section 8); earlier study's before/after`) |
| wrapper 08:142-150 | notes ("Draft, measures under review", "MDI and TreeSHAP", "Close-interaction share", "MDI rank") | the proposed notes (if kept) |

#### Kept figures

**`fig:rh_app_murphy`** (wrapper `10_appendix_floats.tex:378-395`).
- **Sources:**
  - The PNG `results/close_murphy_splits/murphy_modelset.png` is drawn by
    `experiments/close_murphy_splits_headline.py`. Its in-image strings ("threshold c", "ES_c", "plain f²B") are that
    script's; see TABLE_FIXES lines 148-152.
  - This generator supplies only the caption macros (lines 1082-1103): `\rhappModelSetN` (112),
    `\rhappModelSetPaper` (9), `\rhappModelSetPooledLin` (6), `\rhappModelSetPerBarLin` (7),
    `\rhappModelSetPerBarTrees` (90), `\rhappMurphyNc` (33) and `\rhappNdays`. Their values are numbers; nothing
    changes.
- **Caption strings** (wrapper 10:387-393):

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 10:387 | `Murphy diagram (elementary scores) for the model set (` | `Murphy diagrams (elementary scores; \citet{Ehm2016}) for the model set (` |
| 10:387-389 | `earlier draft \rhappModelSetPaper, pooled intraday linear \rhappModelSetPooledLin, closing-interval linear \rhappModelSetPerBarLin, closing-interval trees \rhappModelSetPerBarTrees` | `\rhappModelSetPaper{} pooled intraday forecasts of the earlier draft, \rhappModelSetPooledLin{} pooled intraday linear models, \rhappModelSetPerBarLin{} closing-interval linear models and \rhappModelSetPerBarTrees{} closing-interval tree specifications` |
| 10:391 | `in units of the 15:30 implied variance` | `in units of the implied variance $\mathrm{IV}^2_t$` |
| 10:391-392 | `the threshold used by the sign strategy, $q_t = \sign(s_t)$` | `the threshold of the sign strategy, $q_t = \sign(\hat\sigma^2_t - \mathrm{IV}^2_t)$` |
| 10:392 | `QLIKE equals the integral of the elementary scores weighted by $\theta^{-2}$` | keep |

- **Text pointer:** nothing references the figure. `10_appendix.tex:47` ("Murphy diagrams (elementary scores)") could
  cite it (text editor).

#### Macro values with user-visible text
| Macro | Line | Current value | Replacement |
|---|---|---|---|
| (none) | 1162-1168 | every `\rhapp...` value is numeric (e.g. `\rhappOptPoints` = `1{,}469`, `\rhappGapShareOther` = `\ensuremath{-}25`) | -- |

- **Unused macros** (keep all; never delete or rename):
  - `\rhappNmonthEnd`, `\rhappNextreme`, `\rhappNother`: use them in the dayclass header if that table is kept.
  - `\rhappFiColsEligible`, `\rhappFiDraws`, `\rhappFiConstCols`: used nowhere.
  - `\rhappShift*`, `\rhappCompF`, `\rhappFi*`: used only in parked figures.
- **Possible new macro:** `\rhappNdays` (866) equals `\rhrDays`. The notes could add the dates "3 January 2020 to
  30 April 2024" only through a date macro, if one exists.

#### Other user-visible strings
- **`DASH = "---"` (83):**
  - It is both the parse token for the upstream cells (153, 255-256) and the output for empty cells (256, 792-793).
  - Keep `"---"` for parsing; add an output constant `"--"`. The ladder copies upstream `---` verbatim (512): replace
    it on output.
- **`panel_line` (197-198):** prints `\emph{title}`. Callers pass "Panel A: ..." titles.
- **`fp` (75-76):** prints `$<$0.001`. It becomes dead once the $p$ columns go; reuse its thresholds in `stars()`.
- **`vrow(..., pending=...)` (778, 924):** prints `\emph{pending}`. No variant is `None` now, so the path is dead.
  Output "--" if it is kept.
- **Plus signs:**
  - Upstream ladder cells print text `+`; this generator's `fpct` prints `\ensuremath{+}`.
  - Optionally normalize the ladder's signs at 512.
- **`\addlinespace[2pt]` (919):** only needed because mingru_all fills a page.
- **`write()` header (56) and source lines:** `%` comments, not visible. Code names may stay there.
- **Wrapper `\tabcolsep` overrides to remove once the tables have ≤ 7 columns:**
  - 10:25 (3.5pt), 10:94 (5pt), 10:150 (4pt), 10:184 (5pt), 10:238 (4pt), 10:276 (3pt), 10:319 (3pt).
  - Every wrapper already uses `\small` for the body and `\footnotesize` for the notes, as the guide allows.

#### Open questions
1. **Blocker (see top):** re-score the remaining studies under the actual-$\RV_t$ convention. Then regenerate the
   hard-coded asserts (`KNOWN` 670-677, 726-729, 761-762) from the new outputs, and re-run the two upstream
   generators before this one.
2. **Parking:** confirm the default to park `tab:rh_app_tune_ladder`, `tab:rh_app_mingru_all`,
   `tab:rh_app_mingru_dayclass` and `tab:rh_app_fi_harma`, and to merge mask into cadence and budget_oos into
   tuneper.
3. **Tuning period:** `\rhappOptPoints` = 1,469 tuning dates per specification, against 866 out-of-sample days.
   Over which period does the budget table count tuning dates? Its notes should say.
4. **Stars for the tree comparisons:** add stars from `appendix_campaign_16h_plain_pairs.csv` (`dm_p`, `t_p`), or
   keep "point estimates" only?
5. **Budget table:** show means only (8 narrow columns) or keep mean / median? Should the 2-decimal % cells be
   re-rounded to 1 decimal? Re-round from the source CSV, not from the printed strings.
6. **Random-forest depth set:** {3, 5, 10, 20, 40, none} in picks versus {2, 4, 8, 16, 32, none} in
   `tab:rh_dm_hyper_search`. Which is right?
7. **MinGRU development chain:** is it cumulative, so that (3) keeps (2)'s $\nu_{\mathrm{enc}}$? The proposed "+"
   labels assume so (mingru_all's g8 panel suggests yes). The puffer-study DM lag is not stored (`hac_lag` = 14
   only), so is it \fqsDMLag?

### C.5 `make_rehaul_fi.py`

Writes:
- `generated/rehaul_fi_summary.tex` (tab:rh_fi_summary)
- `generated/rehaul_fi_linear.tex` (tab:rh_fi_linear)
- `generated/rehaul_fi_trees.tex` (tab:rh_fi_trees)
- `generated/rehaul_fi_inter.tex` (tab:rh_fi_inter)
- `generated/rehaul_fi_fig_ranks.tex` and `figures/rehaul_fi_tree_ranks.pdf` (fig:rh_fi_ranks, kept)
- `generated/rehaul_fi_fig_linear.tex`, `generated/rehaul_fi_fig_inter.tex` and their PDFs (fig:rh_fi_linear and fig:rh_fi_inter, both parked)
- `generated/rehaul_fi_numbers.tex` (`\rhfi...` prose macros)

Wrappers:
- `rehaul/08b_fi_measures_floats.tex` loads the numbers file, inputs the four tables and `rehaul_fi_fig_ranks`, and comments out the two parked figures (lines 15-16 and 28-29).
- The wrapper holds no caption or note text. Every caption and note is in the generator.
- Each generated table also loads the numbers file itself (generator line 1095), so its notes can use `\rhfi...` macros.

**State of the file (read this first).** Another session rewrote this generator while this inventory was written,
between 14:21 and 14:29 (2026-10-15). It then had 1,780 lines; the last write was 14:29:32. The outputs were
regenerated at 14:29:35.
- Line numbers below refer to that version. Quotes are exact, so grep for them.
- The old writers are kept as comments: `tt`/`ttc` at lines 245-254, the four old table writers at lines 748-1077,
  and the old figure caption at 1657-1673.
- Most of the brief's issues are already fixed. Fixed since the 03:58 build:

| Brief issue | Status (line) |
|---|---|
| "Unit" header | Fixed: "Predictor group" (1272, 1349); "Group" in the summary (1184, 1203) |
| `BUCKET_ABBR` column (BB, Mom, Liq, EW, VW, Sent, IV, VD) | Dropped from every table. The rows now fall in "Panel A: HAR benchmark block" and "Panel B: Exogenous series" (1274/1277, 1351/1354). `BUCKET_ABBR` (85) is used only by the parked `fig_linear` (1584) |
| `MLABEL` / `MMAC` | `MLABEL` (77) holds words only (LightGBM, XGBoost, Random forest, Lasso, Ridge), which is fine. `MMAC` (84) is a macro-name infix and never printed. No change |
| `\texttt` series names | Gone. `grp()` (213) and `col()` print symbols. Remaining: row labels need the "words (symbol)" form (below) |
| "de-dup study's stored values of the cluster refits", `lgbm_opt_00` | Gone from the notes (old text only at 963-965 in the parked comment; 08 text keeps it as a `%` comment, lines 38-39) |
| "fit space" | Now "on the scale of $y_t$" (1450) |
| "Lundberg et al. 2020" as plain text | Now `\citet{lundberg2020local}` (1367, 1444) |
| Friedman--Popescu as plain text | Now `\citet{FriedmanPopescu2008}` (1455); the key is in `references.bib:981` |
| Borda column | Parked (1362-1365); the text keeps `\rhfi...SpBorda` |
| 9 columns (linear) and 11 columns (trees) | Both now 7 columns: `@{}lrrrrrr@{}` (1313, 1382) |
| Inter table: per-model $W$ row | Moved to the notes as macros (1438-1441, 1462-1463) |
| Inter table: "Probe floor" rows, mixed group/predictor pairs, `--` cells | **Still open** (see tab:rh_fi_inter) |
| `\mathrm{AC}_1`, `\mathrm{NObs}` | Fixed to `\mathrm{AC}` and `N` (115-125, "addenda revision") |
| "16 calendar dummies" without the five that never enter a fit (A.3) | Fixed in the linear notes (1283-1289, 1302) |

#### Tables

##### `tab:rh_fi_summary` (generated/rehaul_fi_summary.tex; generator lines 1149-1238)
- **Problems vs core section 8 (remaining):**
  - The cells name groups by symbol only ($r$, $\mathrm{RAV}^{\mathrm{VW}}$, $\mathrm{AC}$, $\mathrm{EffSpread}^{\mathrm{VW}}$,
    $\mathrm{Volume}$, $\mathrm{RAV}^{\mathrm{EW}}$). Data and Predictors define none of these except $\mathrm{AC}$.
  - The note points to Table~\ref{tab:rh_dm_dictionary} for the symbols (1229). That dictionary
    (`make_rehaul_data_methods.py`, C.3) still prints `\texttt{sumret}` code names and has no symbol column, so the
    pointer explains nothing yet.
  - Symbols are right in these cells: "words (symbol)" measured 596pt against a 470pt `\textwidth` (three group
    columns). Fix the pointer instead.
  - The notes are five sentences but about 170 words. Sentence 2 (1216-1221) holds all four definitions.
  - "(1 = most important)" (1221) differs from "1 = largest" in the other notes and the figure.
  - Otherwise compliant: 7 columns `@{}clrlrlr@{}`; two panels, each with its own header; fixed decimals ($N_A$
    integer, % to 1 decimal, $\bar r_A$ to 1, $W$ to 2); `\small`; booktabs; no stars (no tests). Caption: "Variable
    importance by model".
- **Proposed layout:** keep the current one.
  - Caption: "Variable importance by model" (unchanged).
  - Columns (7): Rank | Lasso: Group, $N_A$ | Ridge: Group, $S_A$ (%) | Ridge, standardized: Group,
    $S^{\mathrm{sd}}_A$ (%). Panel B repeats the layout with LightGBM, XGBoost and Random forest: Group, $\bar r_A$.
  - Panels: "Panel A: Linear models", "Panel B: Tree ensembles". Each panel has ranks 1-5, then the footer rows
    "Kendall's $W$" and "Series above probe".
  - Notes (5 sentences):
    "Closing-interval models on the full predictor set ($p = \rhfiP$ predictors in \rhfiNUnits{} predictor groups),
    $K = \rhfiNRefits$ refits, one every \rhfiRefitEvery{} trading days, over the out-of-sample period \rhfiFirst{}
    to \rhfiLast{} ($|\mathcal O| = \rhfiNOos$ trading days). The five leading groups by the lasso's selection count
    $N_A$ (refits with a nonzero coefficient on group $A$, of $K$), the ridge's share $S_A$ of the absolute
    coefficients and $S^{\mathrm{sd}}_A$ of the absolute standardized coefficients, and the tree ensembles' mean rank
    $\bar r_A$ by mean absolute SHAP value (1 = largest). $W$: Kendall's coefficient of concordance of the $K$
    per-refit rankings. Series above probe: exogenous series with a predictor more important than the continuous noise
    probe $\varepsilon^{\mathrm{U}}$ in the \rhfiNProbe{} probe fits (lasso: selected more often; ridge and trees: a
    larger $|\hat\beta_{k,j}|$ or mean $|\phi_{t,j}|$ in at least half of them). Group names as in
    Tables~\ref{tab:rh_fi_linear} and~\ref{tab:rh_fi_trees}."
  - Moves out of the notes: the two $\sum_k\sum_j$ formulas (1219-1220; they are in the 08 text, lines 95-98 and
    107-109) and the tie-break clause (it stays in the linear notes).
- **String edits:**

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 1216-1221 | `Each panel lists the five leading groups by the measure in the column header: ... (1 = most important). ` | sentence 2 of the proposed notes |
| 1222 | `Kendall's $W$ is the concordance of the $K$ per-refit rankings ($W = 1$: the same ranking at every refit). ` | `$W$: Kendall's coefficient of concordance of the $K$ per-refit rankings. ` |
| 1223-1228 | `Series above probe: exogenous series with a predictor that beats ` + PROBE + `, added in \rhfiNProbe{} probe fits (lasso: selected in a larger share of refits than the probe in its fits; ridge and trees: ...)` | sentence 4 of the proposed notes |
| 1229 | `Group symbols as in Table~\ref{tab:rh_dm_dictionary}.` | `Group names as in Tables~\ref{tab:rh_fi_linear} and~\ref{tab:rh_fi_trees}.` (or keep the dictionary pointer once C.3 adds a symbol column) |
| 1190, 1209 | `"Series above probe"` | keep (optionally "Series above noise probe") |

##### `tab:rh_fi_linear` (generated/rehaul_fi_linear.tex; generator lines 1241-1316)
- **Problems vs core section 8 (remaining):**
  - Row labels in Panel B are symbols only (`grp(u)`, 1265), e.g. `\ensuremath{\mathrm{TO}^{\mathrm{VW}}_{\mathrm{sell}}}`.
    The brief requires "words (symbol)".
  - Measured at `\small`, `\tabcolsep` 4pt, against a 469.8pt `\textwidth`:
    - Full A.2 words, longest current row "Value-weighted seller-initiated turnover ($\mathrm{TO}^{\mathrm{VW}}_{\mathrm{sell}}$)": 446pt.
    - Short words with an EW/VW prefix: 398pt.
    - Full words for a volatility-demand series ("net volatility demand in all index options, opening and closing
      trades") would overflow. The rows are data-driven, so use the short words (dictionary below).
  - Row "$N$" (number of one-minute returns, 45 / 0.3 / ...) sits under the header "$N_A$". The words in the row
    label remove the ambiguity.
  - Notes, sentence 1 (1291): "Lasso and ridge on the full predictor set" does not say closing-interval. The value of
    $\underline\omega$ is not given.
  - Otherwise compliant: 7 columns; `\cmidrule` groups Lasso (2-4) and Ridge (5-7); Panels A/B; fixed decimals;
    notes of 5 sentences; caption "Lasso selection frequency and ridge coefficient shares".
- **Proposed layout:**
  - Caption: "Lasso selection frequency and ridge coefficient shares" (unchanged).
  - Columns (7): Predictor group | Lasso: $N_A$, Nonzero, $S_A$ (%) | Ridge: $S_A$ (%), $S^{\mathrm{sd}}_A$ (%), $\bar r_A$.
  - Panels: "Panel A: HAR benchmark block" (HAR components, Calendar dummies); "Panel B: Exogenous series" (the 18 with
    the largest $N_A$), with "words (symbol)" labels such as "VW realized absolute variation ($\mathrm{RAV}^{\mathrm{VW}}$)".
  - Notes (5 sentences): the current text with these edits.
    - Sentence 1: "Closing-interval lasso and ridge on the full predictor set, <SAMPLE>; penalty weight
      $\underline\omega = \rhfiBackbonePf$ on the HAR benchmark block (Panel A)."
    - Sentence 2 ends: "...; EW and VW: equal- and value-weighted index constituents."
- **String edits:**

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 1265 | `f"{grp(u)} & {int(a['refits_any_nonzero'])} & ..."` | `f"{grp_row(u)} & ..."`, with new helper `grp_row(u)` = `GROUP_WORDS[u]` for HAR/calendar, else `f"{SERIES_WORDS(u)} (\ensuremath{{{series_tex(u)}}})"` |
| 1291-1292 | `r"Lasso and ridge on the full predictor set, " + SAMPLE + r"; penalty weight "` / `r"$\omega_j = \underline{\omega}$ on the HAR benchmark block (Panel A). "` | `r"Closing-interval lasso and ridge on the full predictor set, " + SAMPLE + r"; penalty weight $\underline{\omega} = \rhfiBackbonePf$ on the HAR benchmark block (Panel A). "` |
| 1298 | `r" of the \rhfiNUnits{} groups are never selected. "` | `r" of the \rhfiNUnits{} groups are never selected; EW and VW: equal- and value-weighted index constituents. "` |
| 1302 | `({word(len(cal_drop))} of them constant or duplicated at the closing interval, so never in a fit)` | keep. Optionally name them: "(Hour, Night, Open, Close and $\mathrm{QW}\cdot\mathrm{Close}$: constant or duplicated at the closing interval, so never in a fit)" |

##### `tab:rh_fi_trees` (generated/rehaul_fi_trees.tex; generator lines 1319-1385)
- **Problems vs core section 8 (remaining):**
  - Row labels are symbols only (`grp(u)`, 1335).
  - With full A.2 words the table measured 467pt of 470pt for "Value-weighted seller-initiated turnover". It overflows
    for $\mathrm{NSent}$ ("number of sentiment-tagged StockTwits messages ($\mathrm{NSent}$)"), which is in the current
    15 rows. Short words: 418pt.
  - The notes do not say that the trees run at fixed hyperparameters.
  - Otherwise compliant:
    - 7 columns; `\cmidrule` per model; Panels A/B.
    - Top 5 is now to 1 decimal (core: percentages 1). $\bar r_A$ to 1 decimal.
    - Footer: "Kendall's $W$" and "Series above probe", as `\multicolumn{2}{c}` cells.
    - Notes: 5 sentences.
    - Borda parked.
    - Caption: "Tree ensembles: mean SHAP rank by predictor group".
- **Proposed layout:**
  - Caption: unchanged.
  - Columns (7): Predictor group | LightGBM: $\bar r_A$, Top 5 (%) | XGBoost: same | Random forest: same.
  - Panels: "Panel A: HAR benchmark block", "Panel B: Exogenous series" (13 rows), with "words (symbol)" labels.
  - Notes (5 sentences):
    "Path-dependent TreeSHAP values of \citet{lundberg2020local} for the closing-interval tree ensembles at fixed
    hyperparameters (Section~\ref{sec:rh_methods_hyper}) on the full predictor set, <SAMPLE>. At refit $k$, group $A$
    is ranked ($r_{k,A}$, 1 = largest) among the \rhfiNUnits{} groups by its mean absolute SHAP value on the
    out-of-sample block $\mathcal O_k$. $\bar r_A$: mean of $r_{k,A}$ over the refits; Top 5: share of refits with
    $r_{k,A} \le 5$. Rows: the 15 groups with the lowest $\bar r_A$ averaged over the three models; EW and VW: equal-
    and value-weighted index constituents. Kendall's $W$ and series above probe as in Table~\ref{tab:rh_fi_summary}."
  - Moves to the text: the formula $I_{k,A} = |\mathcal O_k|^{-1}\sum_{t\in\mathcal O_k}|\phi_{t,A}|$ (1370). It is
    already in 08, lines 127-130.
- **String edits:**

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 1335 | `return f"{grp(u)} & " + " & ".join(cells) + r" \\"` | `return f"{grp_row(u)} & " + ...` |
| 1367 | `for the closing-interval tree ensembles on the "` | `for the closing-interval tree ensembles at fixed hyperparameters (Section~\ref{sec:rh_methods_hyper}) on the "` |
| 1369-1371 | `by its mean absolute SHAP value on the out-of-sample block, $I_{k,A} = |\mathcal O_k|^{-1}\sum_{t\in\mathcal O_k}|\phi_{t,A}|$ (rank $r_{k,A}$, 1 = largest). ` | `is ranked ($r_{k,A}$, 1 = largest) among the \rhfiNUnits{} groups by its mean absolute SHAP value on the out-of-sample block $\mathcal O_k$. ` |
| 1375 | `r" groups with the lowest $\bar r_A$ averaged over the three models. "` | `r" groups with the lowest $\bar r_A$ averaged over the three models; EW and VW: equal- and value-weighted index constituents. "` |

##### `tab:rh_fi_inter` (generated/rehaul_fi_inter.tex; generator lines 1388-1472)
- **Problems vs core section 8 (remaining):**
  - One 6-column header serves two kinds of rows.
    - "Group pairs" fill all five statistics.
    - "Predictor pairs" print `-- & --` in the middle columns $\bar r_{AB}$ and Top 10 (1431).
  - "\quad Probe floor" (1435-1437) is a row with one number and four blank cells, in each of the three model panels.
  - The panels are models ("Panel A: LightGBM" ... "Panel C: Random forest", 1414). Each panel has unlabeled
    sub-headers "Group pairs" (1415) and "Predictor pairs" (1426).
  - The float is about a page tall. The generator comment at 1438 records that one more row per panel overflowed by 20pt.
  - Predictor pairs print `\ensuremath{\bar y^{(1)}_t} $\times$ \ensuremath{\mathrm{RAV}^{(1)}_{t}}`: two math groups,
    a time subscript (see Open questions), and inconsistent `_t` / `_{t}`. A.1 form: `$\bar y^{(1)} \times \mathrm{RAV}^{(1)}$`.
  - Notes: 6 sentences (more than 5).
  - $H^2$ (%) uses 2 decimals while core section 8 says 1 for percentages. The values run from 0.05 to 1.16, so 1
    decimal would print 0.1 / 0.0. Keep 2 as this column's fixed decimals.
  - Group-pair labels use symbols ("HAR components $\times$ $\mathrm{RAV}^{\mathrm{EW}}$"). That is right here: with
    words the column measured 496pt against 470pt. Every symbol in these pairs is a row label of tab:rh_fi_trees.
- **Proposed layout (recommended, A): panels by pair type, models as italic sub-blocks, all five statistics kept.**
  - Caption: "Tree ensembles: SHAP interaction effects" (unchanged).
  - Columns (6): Pair | $\bar I$ ($\times 10^{3}$) | $H^2$ (%) | $H^2_{\min}$ | $\bar r_{AB}$ | Top 10 (%).
    - Group header: `& \multicolumn{3}{c}{Over $\mathcal O$} & \multicolumn{2}{c}{Per refit} \\`.
    - Rules: `\cmidrule(lr){2-4}\cmidrule(l){5-6}`.
    - The per-refit columns come last, so a row without them ends in "--" instead of having gaps in the middle.
  - "Panel A: Pairs of predictor groups": for each model, an italic sub-header (`\textit{LightGBM}`, ...) and its 5
    pairs. 18 lines.
  - "Panel B: Pairs of predictors": for each model, an italic sub-header and its 3 pairs, written
    `$\bar y^{(1)} \times \bar y^{(5)}$`, with "--" in the two per-refit columns. 12 lines.
  - Drop the three "Probe floor" rows. The floors go into the notes through the existing macros
    `\rhfiLgbmInterFloor`, `\rhfiXgbInterFloor`, `\rhfiRfInterFloor`, as was done for $W$.
  - The table loses about 3 lines net (33 instead of 36), which also eases the page-height problem.
  - Notes (5 sentences):
    "SHAP interaction values \citep{lundberg2020local} of the re-estimated closing-interval tree ensembles over the
    out-of-sample period $\mathcal O$ ($|\mathcal O| = \rhfiNOos$ trading days, $K = \rhfiNRefits$ refits), on the
    scale of $y_t$. $\bar I$: the mean over $\mathcal O$ of $|\Phi_{t,AB}|$ for a pair of predictor groups (Panel A:
    the five largest of \rhfiNPairs) and of $|\phi_{t,jl} + \phi_{t,lj}|$ for a pair of predictors (Panel B: the three
    largest); group symbols as in Table~\ref{tab:rh_fi_trees}. $H^2$: the $H$-statistic of \citet{FriedmanPopescu2008},
    in percent, and $H^2_{\min}$, the same numerator over the smaller single-group partial-dependence variance, both
    pooled over \rhfiHRefits{} refits. $\bar r_{AB}$ and Top 10: the pair's mean per-refit rank and the share of refits
    in which it ranks in the top ten (group pairs only); Kendall's $W$ of the per-refit rankings of the top
    \rhfiTopWPairs{} pairs: \rhfiLgbmInterW{}, \rhfiXgbInterW{} and \rhfiRfInterW{} (LightGBM, XGBoost, random forest).
    Noise floor, the median over the \rhfiNProbe{} probe fits of the largest $\bar I$ between the continuous noise
    probe $\varepsilon^{\mathrm{U}}$ and any group: \rhfiLgbmInterFloor{}, \rhfiXgbInterFloor{} and \rhfiRfInterFloor."
  - Moves to the text: "computed for each model's top \rhfiHTopSeries{} group pairs and top five predictor pairs"
    (1456-1459). 08 already states it, lines 182-183.
- **Alternative (B), if the page height still bites: models across, as in tab:rh_fi_trees.**
  - Columns (7): Pair | LightGBM: $\bar I$, Top 10 (%) | XGBoost: same | Random forest: same. Measured 403pt.
  - Rows:
    - Panel A: the union of each model's top five group pairs (6 rows with the current data).
    - Panel B: the union of each model's top three predictor pairs (5 rows).
    - Footer, in `\multicolumn{2}{c}` cells: "Noise floor", "Kendall's $W$".
  - About 17 lines in all. $H^2$, $H^2_{\min}$ and $\bar r_{AB}$ move to the text, which already reports their ranges
    (`\rhfi...HTop`, `...HMin`/`HMax`, `...HMinTop`, `...HMinLo`/`Hi`, `...HColTop`, `...InterSpGlobal`).
  - The union rows need $\bar I$ for pairs outside a model's top 5. `T[m+"_pairs"]` holds all \rhfiNPairs{} group
    pairs. For predictor pairs, check that `{m}_inter_top_columns.csv` covers the union; otherwise print "--".
- **String edits (layout A):**

| Line | Current (quote, shortened) | Replacement |
|---|---|---|
| 1397 | `r"Pair & $\bar I_{AB}$ ($\times 10^{3}$) & $\bar r_{AB}$ & Top 10 (\%) & $H^2$ (\%) & $H^2_{\min}$ \\"` | `r" & \multicolumn{3}{c}{Over $\mathcal O$} & \multicolumn{2}{c}{Per refit} \\"`, `r"\cmidrule(lr){2-4}\cmidrule(l){5-6}"`, `r"Pair & $\bar I$ ($\times 10^{3}$) & $H^2$ (\%) & $H^2_{\min}$ & $\bar r_{AB}$ & Top 10 (\%) \\"` |
| 1400-1414 | one loop over `MODELS`, with `panel(f"Panel {'ABC'[mi]}: {MLABEL[m]}", 6)` | two passes: `panel("Panel A: Pairs of predictor groups", 6)` then a loop over `MODELS` with `rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{{MLABEL[m]}}}}} \\"`; then `panel("Panel B: Pairs of predictors", 6)` and the same loop |
| 1415 | `r"\multicolumn{6}{@{}l}{Group pairs} \\"` | delete (the panel title carries it) |
| 1420-1422 | `{Ī} & {r̄} & {Top10} & {H2} & {H2min}` | `{Ī} & {H2} & {H2min} & {r̄} & {Top10}` (reorder) |
| 1426 | `r"\multicolumn{6}{@{}l}{Predictor pairs} \\"` | delete |
| 1430-1432 | `rf"\quad {col(cz['col_a'][i])} $\times$ {col(cz['col_b'][i])} & " f"{...} & -- & -- & {H2} & {H2min} \\\\"` | `rf"\quad ${column_tex(a)} \times {column_tex(b)}$ & {Ī} & {H2} & {H2min} & -- & -- \\"` (one math group) |
| 1435-1437 | `rf"\quad Probe floor & {f(1e3 * IQ['floor_ser_median'], 2)} & & & & \\"` | delete; floors go into the notes as macros |
| 1443-1463 | notes (6 sentences) | proposed notes (5 sentences) |
| 1459 | `predictor pairs (--: not computed). ` | gone; "(group pairs only)" in sentence 4 |
| 1460-1463 | `Probe floor: median over ... Kendall's $W$ of the per-refit rankings of each model's top \rhfiTopWPairs{} pairs: ...` | sentences 4-5 of the proposed notes |

**Short words for "words (symbol)" row labels.** Add a `SERIES_WORDS` dict next to `SERIES_TEX` (line 108). It holds
the A.2 words shortened, with "EW"/"VW" as a prefix for the constituent aggregates (`_ewstock`/`_vwstock`), applied
the same way `AGG_TEX` is:

| Code stem | Row words | Code stem | Row words |
|---|---|---|---|
| sumret | Interval log return | turnover | (EW/VW) turnover |
| sumabsret | Realized absolute variation | buyturnover | (EW/VW) buyer-initiated turnover |
| sumret3 | Realized skewness | sellturnover | (EW/VW) seller-initiated turnover |
| sumret4 | Realized quarticity | effspread | (EW/VW) effective spread |
| sumpret2 | Positive realized semivariance | spread | (EW/VW) quoted spread |
| sumbipow | Bipower variation | ofi | (EW/VW) order-flow imbalance |
| sumautocov | Realized autocovariance | sumret2 | (EW/VW) realized variance |
| sumvolume | ES contract volume | stocktwits_attention | StockTwits attention |
| numobs | One-minute return count | stocktwits_sentiment | StockTwits sentiment |
| vix / vvix / vix3m | VIX index / VVIX index / VIX3M index | stocktwits_sentcount | StockTwits sentiment count |
| voldemand_spx_open_and_close | SPX volatility demand (all trades) | voldemand_spx_open_only | SPX volatility demand (opening) |
| voldemand_all_open_and_close | Index volatility demand (all trades) | voldemand_all_open_only | Index volatility demand (opening) |
| fomc_release | FOMC statement interval | fomc_day | FOMC statement day |
| fomc_until_inv | Inverse time to next FOMC statement | fomc_since_inv | Inverse time since last FOMC statement |

- Row label: `f"{words} (\ensuremath{{{series_tex(u)}}})"`, e.g. "VW seller-initiated turnover
  ($\mathrm{TO}^{\mathrm{VW}}_{\mathrm{sell}}$)".
- Prose form (`grp_prose`): the words in lower case, except proper nouns and EW/VW, followed by the symbol without
  parentheses, e.g. "realized absolute variation $\mathrm{RAV}$".
- Add a gate next to the existing name gates (352-368): every unit has words, and the words are distinct.

#### Kept figures

`fig:rh_fi_ranks` (`fig_ranks`, lines 1624-1692; `generated/rehaul_fi_fig_ranks.tex`; `figures/rehaul_fi_tree_ranks.pdf`,
regenerated 14:29). There are three stacked panels, one per model, and no title.

| Line | String | Current | Replacement |
|---|---|---|---|
| 1641-1643 | legend labels `label=grp_mpl(u)` | "HAR components", "$r$", "$\mathrm{RAV}$", "$\mathrm{Volume}$", "$\mathrm{RAV}^{\mathrm{EW}}$", "$\mathrm{RAV}^{\mathrm{VW}}$" (current data; was "HAR ladder", "sumret", ... via `short()`) | keep the symbols: 3 legend columns cannot hold the words. Define them through the caption pointer below |
| 1650 | y label `f"{MLABEL[m]}\nrank $r_{{k,A}}$"` | "LightGBM / rank $r_{k,A}$" (and XGBoost, Random forest) | keep |
| 1647-1648 | y ticks `[1, 2, 5, 10, 20, 49]`, labels `["1", ..., "49"]` | "49" typed (= `S["n_units"]`) | keep the text. Optionally derive the last tick from `S["n_units"]` |
| -- | x axis | year labels (`DateFormatter("%Y")`), no x label | keep |
| 1676 | caption, sentence 1: `Mean absolute SHAP rank of the leading predictor groups over time.` | -- | `Variable-importance ranks of the tree ensembles over time.` (the plot shows the rank by mean absolute SHAP value, not a mean of ranks) |
| 1684 | caption end: `ensembles; Kendall's $W$ is in Table~\ref{tab:rh_fi_summary}.` | -- | `ensembles, named as in Table~\ref{tab:rh_fi_trees}; Kendall's $W$ is in Table~\ref{tab:rh_fi_summary}.` |

The rest of the caption is fine: refit $k$, \rhfiRefitEvery, \rhfiNUnits, $\mathcal O_k$, $t_k$, "centered", "trading
days", and "about 110 trading days", computed as `roll * S["refit_every"]`.

Parked, not inventoried: `fig:rh_fi_linear` (`fig_linear`, 1521-1621) and `fig:rh_fi_inter` (`fig_inter`, 1695-1753).
They still write their `.tex` and `.pdf` with the old wording ("units", "sessions", "pre-tuned", "colour",
`short()` tick labels "HAR ladder", `BUCKET_ABBR` side labels). The wrapper does not input them.

#### Macro values with user-visible text

Names are generator logic, so a replacement is a rule. The values shown are for the current data:
- "before": the 03:58 build.
- "now": the 14:29 build.
- "proposed": `grp_prose()` (lowercase words + symbol), with no `_t` in names (see Open questions).

The HAR and calendar words ("HAR components", "calendar dummies") are already right. Where the 08 text uses a macro,
the line is given.

| Macro | Line | Before (03:58) | Now (14:29) | Proposed |
|---|---|---|---|---|
| `\rhfiLassoAlwaysList` (08:78) | 504-505 | `\texttt{sumret}, \texttt{sumabsret\_vwstock}, \texttt{sumautocov}` | `\ensuremath{r}, \ensuremath{\mathrm{RAV}^{\mathrm{VW}}}, \ensuremath{\mathrm{AC}}` | `interval log return $r$, VW realized absolute variation $\mathrm{RAV}^{\mathrm{VW}}$ and realized autocovariance $\mathrm{AC}$` (join the last item with " and ") |
| `\rhfiRidgeTopOne` (08:113) | 536 | `Calendar` | `calendar dummies` | keep |
| `\rhfiRidgeTopTwo` (08:113) | 536 | `HAR ladder` | `HAR components` | keep |
| `\rhfiRidgeTopThree` (08:114) | 536 | `\texttt{sumret}` | `\ensuremath{r}` | `interval log return $r$` |
| `\rhfiRidgeTopFour` (unused) | 536 | `\texttt{sumabsret\_vwstock}` | `\ensuremath{\mathrm{RAV}^{\mathrm{VW}}}` | `VW realized absolute variation $\mathrm{RAV}^{\mathrm{VW}}$` |
| `\rhfiRidgeSdTopOne` (08:110) | 545 | `HAR ladder` | `HAR components` | keep |
| `\rhfiRidgeSdTopTwo` (08:110) | 545 | `Calendar` | `calendar dummies` | keep |
| `\rhfiRidgeSdTopThree` (08:111) | 545 | `\texttt{sumret}` | `\ensuremath{r}` | `interval log return $r$` |
| `\rhfiRidgeTopExog` (unused) | 548 | `\texttt{sumret}` | `\ensuremath{r}` | `interval log return $r$` |
| `\rhfiLassoTopCol` (unused) | 573 | `\texttt{is\_month\_end}` | `\ensuremath{\mathrm{ME}_t}` | `month-end dummy $\mathrm{ME}$` (A.3) |
| `\rhfiLgbmTopOne`, `\rhfiXgbTopOne`, `\rhfiRfTopOne` (08:146, 148, 150) | 611 | `HAR ladder` | `HAR components` | keep |
| `\rhfiLgbmTopTwo`, `\rhfiXgbTopTwo` (08:146, 148) | 611 | `\texttt{sumret}` | `\ensuremath{r}` | `interval log return $r$` |
| `\rhfiRfTopTwo` (08:150) | 611 | `\texttt{sumabsret}` | `\ensuremath{\mathrm{RAV}}` | `realized absolute variation $\mathrm{RAV}$` |
| `\rhfiLgbmTopThree`, `\rhfiXgbTopThree` (08:147, 148) | 611 | `\texttt{sumabsret}` | `\ensuremath{\mathrm{RAV}}` | `realized absolute variation $\mathrm{RAV}$` |
| `\rhfiRfTopThree` (08:150) | 611 | `\texttt{sumret}` | `\ensuremath{r}` | `interval log return $r$` |
| `\rhfiLgbmTopCol`, `\rhfiXgbTopCol`, `\rhfiRfTopCol` (unused) | 637-638 | `\texttt{har\_ma\_1}` | `\ensuremath{\bar y^{(1)}_t}` | `$\bar y^{(1)}$` (A.1) |
| `\rhfiLgbmTopColTwo`, `\rhfiXgbTopColTwo`, `\rhfiRfTopColTwo` (unused) | 642 | `\texttt{har\_ma\_5}` | `\ensuremath{\bar y^{(5)}_t}` | `$\bar y^{(5)}$` |
| `\rhfiLgbmInterTopA`, `\rhfiXgbInterTopA`, `\rhfiRfInterTopA` (08:170-172) | 669 | `HAR ladder` | `HAR components` | keep |
| `\rhfiLgbmInterTopB`, `\rhfiXgbInterTopB`, `\rhfiRfInterTopB` (08:170-172) | 670 | `\texttt{sumabsret}` | `\ensuremath{\mathrm{RAV}}` | `realized absolute variation $\mathrm{RAV}$` |
| `\rhfiLgbmWithinTop`, `\rhfiXgbWithinTop`, `\rhfiRfWithinTop` (08:176-177) | 705 | `HAR ladder` | `HAR components` | keep |

- Why words + symbol in prose: Data and Predictors define $\mathrm{AC}$, $\mathrm{BV}$, $N$, $\mathrm{OFI}$ and the
  EW/VW $\RV$, but not $r$ as a series, $\mathrm{RAV}$, $\mathrm{RQ}$, $\mathrm{RSK}$, $\mathrm{Volume}$ or
  $\mathrm{Sent}$. A bare symbol in 08 (e.g. "LightGBM: HAR components (1.0), $r$ (2.9), $\mathrm{RAV}$ (4.8)") would
  be undefined.
- Other macro values are numbers, dates (`\rhfiFirst` = `25~June~2018`, `\rhfiLast` = `30~April~2024`) or
  `\ensuremath{10^{...}}` (`\rhfiRidgeAlpha`, `\rhfiLassoAlpha`); none is a code name.
- The `% ...` comments in the numbers file already use the guide's terms (lines 444-445).

#### Other user-visible strings

- **Panel and footer labels:**
  - Fine as they are: "Panel A: Linear models" (1181), "Panel B: Tree ensembles" (1198), "Panel A: HAR benchmark
    block" (1274, 1351), "Panel B: Exogenous series" (1277, 1354), "Rank" and "Group" (1184, 1203), "Kendall's $W$"
    (1188, 1207, 1357).
  - "Series above probe" (1190, 1209, 1359): optionally "Series above noise probe".
  - Inter labels "Panel A/B/C: <model>", "Group pairs", "Predictor pairs", "Probe floor" (1414, 1415, 1426, 1436):
    replace per the inter layout above.
- **`SAMPLE` (1123-1126) and `PROBE` (1127):** shared note fragments. They read "$K = \rhfiNRefits$ refits, one every
  \rhfiRefitEvery{} trading days, over the out-of-sample period \rhfiFirst{} to \rhfiLast{} ($|\mathcal O| = \rhfiNOos$
  trading days)" and "the continuous noise probe $\varepsilon^{\mathrm{U}}$". Both fine (core 6b, $\mathcal O$; A.3, $\varepsilon^{\mathrm{U}}$).
- **`CAL_TEX["is_rebalance_close"]` (169):** `(\mathrm{QW}\cdot\mathrm{Close})` has parentheses that A.3 lacks. Keep
  them while a `_t` or `^{(k)}` follows; drop them if names lose `_t`. The column never enters a fit (A.3), so it
  never prints.
- **Comments only, not printed:**
  - Line 746: a literal TAB in "code names, <TAB>exttt units" (a `\t` escape ate `\t` of `\texttt`).
  - Docstring lines 13-15: the `rehaul_fi_numbers.tex` entry sits under the 2026-10-15 paragraph instead of the file
    list.
  - `rehaul/08b_fi_measures_floats.tex` lines 8, 18: "leading units", "units above the noise bar".
  - `rehaul/08_feature_importance.tex` line 20: "The generated floats ... still print the old symbols ($T_k$,
    $\beta^{(k)}_j$, "unit", code names)". This is now stale; update the comment, not the text.

#### Open questions

1. **Time subscript in predictor names.** `column_tex` (201, 203, 206, 209) appends `_t`: $\bar y^{(1)}_t$,
   $\mathrm{RAV}^{(1)}_{t}$, $\mathrm{ME}_t$. The generator comment (101-107) follows core section 5 and
   04_features.tex:51-55 ($\mathrm{VIX}^{(25)}_t$). Addenda A.1 (line 34) and the brief drop time subscripts in tables
   ($\bar y^{(1)}$). This decides tab:rh_fi_inter Panel B and the `\rhfi...TopCol*` / `\rhfiLassoTopCol` values. One
   line each in `column_tex`.
2. **Inter table:** layout A (all five statistics, pair-type panels, about 33 lines) or B (models across, about 17
   lines, $H^2$ and ranks to the text)?
3. **Row words:** short (EW/VW prefix, this section's dictionary) or the full A.2 words? Full words fit the current
   linear rows (446pt of 470pt) but not the trees table ($\mathrm{NSent}$) or long volatility-demand names.
4. **Section label:** `08_feature_importance.tex` has no `\label` after `\section{Variable importance}` (line 25). The
   notes can point to `sec:rh_methods_hyper` but not to section 8's definitions. Adding `\label{sec:rh_vi}` is not a
   rename; the coordinator decides.
5. **Dictionary dependency:** if tab:rh_fi_summary keeps "Group symbols as in Table~\ref{tab:rh_dm_dictionary}", C.3
   (`make_rehaul_data_methods.py`) must add the A.2 symbol column to the dictionary.

---

## D. References missing from `writeup/references.bib`

- **Checked against** the 75 keys in `references.bib` (2026-10-15). Already present and used by the guides:
  `AndersenBollerslevDieboldLabys2003`, `AndersenBollerslev1997`, `BarndorffNielsenShephard2004`, `Corsi2009`,
  `Patton2011`, `DieboldMariano1995`, `HarveyLeybourneNewbold1997`, `HansenLundeNason2011`, `Ehm2016`, `ZouHastie2005`,
  `Tibshirani1996`, `HoerlKennard1970`, `gu2020empirical`, `lundberg2017unified`, `lundberg2018consistent`,
  `lundberg2020local`, `Duan1983`, `Ke2017`, `Chen2016`, `Akiba2019`, `BollerslevPattonQuaedvlieg2016`,
  `vandeWiel2021multipenalty`, `Garrigues2008`.
- **Key-name oddity, not an error:** `BollerslevHoodHussLebedev2018` holds Bollerslev, Hood, Huss and *Pedersen*
  (2018). The key is fine to keep; do not "fix" the author list to match it.
- **D.1** lists what the two guides, the edited sections or section A rely on. **D.2** lists the draft's own
  `\todo{to add}` items (02_related_work:50, 60), for completeness. Append D.1 to `references.bib`; verify the D.2
  working papers against SSRN before adding them.

### D.1 Required by the guides and the edited sections

```bibtex
@article{BarndorffNielsenShephard2002,
  author  = {Barndorff-Nielsen, Ole E. and Shephard, Neil},
  title   = {Econometric Analysis of Realized Volatility and its Use in Estimating Stochastic Volatility Models},
  journal = {Journal of the Royal Statistical Society: Series B (Statistical Methodology)},
  year    = {2002},
  volume  = {64},
  number  = {2},
  pages   = {253--280}
}

@article{TaylorXu1997,
  author  = {Taylor, Stephen J. and Xu, Xinzhong},
  title   = {The Incremental Volatility Information in One Million Foreign Exchange Quotations},
  journal = {Journal of Empirical Finance},
  year    = {1997},
  volume  = {4},
  number  = {4},
  pages   = {317--340}
}

@article{NeweyWest1987,
  author  = {Newey, Whitney K. and West, Kenneth D.},
  title   = {A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix},
  journal = {Econometrica},
  year    = {1987},
  volume  = {55},
  number  = {3},
  pages   = {703--708}
}

@article{Zou2006,
  author  = {Zou, Hui},
  title   = {The Adaptive Lasso and Its Oracle Properties},
  journal = {Journal of the American Statistical Association},
  year    = {2006},
  volume  = {101},
  number  = {476},
  pages   = {1418--1429}
}

@article{FriedmanHastieTibshirani2010,
  author  = {Friedman, Jerome and Hastie, Trevor and Tibshirani, Robert},
  title   = {Regularization Paths for Generalized Linear Models via Coordinate Descent},
  journal = {Journal of Statistical Software},
  year    = {2010},
  volume  = {33},
  number  = {1},
  pages   = {1--22}
}

@article{Friedman2001,
  author  = {Friedman, Jerome H.},
  title   = {Greedy Function Approximation: A Gradient Boosting Machine},
  journal = {Annals of Statistics},
  year    = {2001},
  volume  = {29},
  number  = {5},
  pages   = {1189--1232}
}

@article{Breiman2001,
  author  = {Breiman, Leo},
  title   = {Random Forests},
  journal = {Machine Learning},
  year    = {2001},
  volume  = {45},
  number  = {1},
  pages   = {5--32}
}

@book{HastieTibshiraniFriedman2009,
  author    = {Hastie, Trevor and Tibshirani, Robert and Friedman, Jerome},
  title     = {The Elements of Statistical Learning: Data Mining, Inference, and Prediction},
  edition   = {2},
  publisher = {Springer},
  address   = {New York},
  year      = {2009}
}

@article{Hansen2005,
  author  = {Hansen, Peter Reinhard},
  title   = {A Test for Superior Predictive Ability},
  journal = {Journal of Business \& Economic Statistics},
  year    = {2005},
  volume  = {23},
  number  = {4},
  pages   = {365--380}
}

@article{White2000,
  author  = {White, Halbert},
  title   = {A Reality Check for Data Snooping},
  journal = {Econometrica},
  year    = {2000},
  volume  = {68},
  number  = {5},
  pages   = {1097--1126}
}

@article{RomanoWolf2005,
  author  = {Romano, Joseph P. and Wolf, Michael},
  title   = {Stepwise Multiple Testing as Formalized Data Snooping},
  journal = {Econometrica},
  year    = {2005},
  volume  = {73},
  number  = {4},
  pages   = {1237--1282}
}

@article{LedoitWolf2008,
  author  = {Ledoit, Oliver and Wolf, Michael},
  title   = {Robust Performance Hypothesis Testing with the {S}harpe Ratio},
  journal = {Journal of Empirical Finance},
  year    = {2008},
  volume  = {15},
  number  = {5},
  pages   = {850--859}
}

@article{FlemingKirbyOstdiek2001,
  author  = {Fleming, Jeff and Kirby, Chris and Ostdiek, Barbara},
  title   = {The Economic Value of Volatility Timing},
  journal = {Journal of Finance},
  year    = {2001},
  volume  = {56},
  number  = {1},
  pages   = {329--352}
}

@article{WelchGoyal2008,
  author  = {Welch, Ivo and Goyal, Amit},
  title   = {A Comprehensive Look at the Empirical Performance of Equity Premium Prediction},
  journal = {Review of Financial Studies},
  year    = {2008},
  volume  = {21},
  number  = {4},
  pages   = {1455--1508}
}

@article{CampbellThompson2008,
  author  = {Campbell, John Y. and Thompson, Samuel B.},
  title   = {Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average?},
  journal = {Review of Financial Studies},
  year    = {2008},
  volume  = {21},
  number  = {4},
  pages   = {1509--1531}
}

@article{BollerslevTauchenZhou2009,
  author  = {Bollerslev, Tim and Tauchen, George and Zhou, Hao},
  title   = {Expected Stock Returns and Variance Risk Premia},
  journal = {Review of Financial Studies},
  year    = {2009},
  volume  = {22},
  number  = {11},
  pages   = {4463--4492}
}

@article{CarrWu2009,
  author  = {Carr, Peter and Wu, Liuren},
  title   = {Variance Risk Premiums},
  journal = {Review of Financial Studies},
  year    = {2009},
  volume  = {22},
  number  = {3},
  pages   = {1311--1341}
}

@incollection{BarndorffNielsenKinnebrockShephard2010,
  author    = {Barndorff-Nielsen, Ole E. and Kinnebrock, Silja and Shephard, Neil},
  title     = {Measuring Downside Risk: Realised Semivariance},
  booktitle = {Volatility and Time Series Econometrics: Essays in Honor of {R}obert {F}. {E}ngle},
  editor    = {Bollerslev, Tim and Russell, Jeffrey R. and Watson, Mark W.},
  publisher = {Oxford University Press},
  address   = {Oxford},
  year      = {2010},
  pages     = {117--136}
}

@article{PattonSheppard2015,
  author  = {Patton, Andrew J. and Sheppard, Kevin},
  title   = {Good Volatility, Bad Volatility: Signed Jumps and the Persistence of Volatility},
  journal = {Review of Economics and Statistics},
  year    = {2015},
  volume  = {97},
  number  = {3},
  pages   = {683--697}
}

@article{AmayaChristoffersenJacobsVasquez2015,
  author  = {Amaya, Diego and Christoffersen, Peter and Jacobs, Kris and Vasquez, Aurelio},
  title   = {Does Realized Skewness Predict the Cross-Section of Equity Returns?},
  journal = {Journal of Financial Economics},
  year    = {2015},
  volume  = {118},
  number  = {1},
  pages   = {135--167}
}

@article{ForsbergGhysels2007,
  author  = {Forsberg, Lars and Ghysels, Eric},
  title   = {Why Do Absolute Returns Predict Volatility So Well?},
  journal = {Journal of Financial Econometrics},
  year    = {2007},
  volume  = {5},
  number  = {1},
  pages   = {31--67}
}

@article{HansenLunde2006,
  author  = {Hansen, Peter R. and Lunde, Asger},
  title   = {Realized Variance and Market Microstructure Noise},
  journal = {Journal of Business \& Economic Statistics},
  year    = {2006},
  volume  = {24},
  number  = {2},
  pages   = {127--161}
}

@article{NiPanPoteshman2008,
  author  = {Ni, Sophie X. and Pan, Jun and Poteshman, Allen M.},
  title   = {Volatility Information Trading in the Option Market},
  journal = {Journal of Finance},
  year    = {2008},
  volume  = {63},
  number  = {3},
  pages   = {1059--1091}
}

@article{FriedmanPopescu2008,
  author  = {Friedman, Jerome H. and Popescu, Bogdan E.},
  title   = {Predictive Learning via Rule Ensembles},
  journal = {Annals of Applied Statistics},
  year    = {2008},
  volume  = {2},
  number  = {3},
  pages   = {916--954}
}

@article{KendallBabingtonSmith1939,
  author  = {Kendall, M. G. and Babington Smith, B.},
  title   = {The Problem of $m$ Rankings},
  journal = {Annals of Mathematical Statistics},
  year    = {1939},
  volume  = {10},
  number  = {3},
  pages   = {275--287}
}

@misc{Feng2024minGRU,
  author        = {Feng, Leo and Tung, Frederick and Ahmed, Mohamed Osama and Bengio, Yoshua and Hajimirsadeghi, Hossein},
  title         = {Were {RNN}s All We Needed?},
  year          = {2024},
  eprint        = {2410.01201},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

@misc{Jordan2024Muon,
  author       = {Jordan, Keller and Jin, Yuchen and Boza, Vlado and You, Jiacheng and Cesista, Franz and Newhouse, Laker and Bernstein, Jeremy},
  title        = {Muon: An Optimizer for Hidden Layers in Neural Networks},
  year         = {2024},
  howpublished = {\url{https://kellerjordan.github.io/posts/muon/}},
  note         = {Blog post}
}

@inproceedings{KingmaBa2015,
  author    = {Kingma, Diederik P. and Ba, Jimmy},
  title     = {Adam: A Method for Stochastic Optimization},
  booktitle = {3rd International Conference on Learning Representations (ICLR)},
  year      = {2015}
}

@inproceedings{LoshchilovHutter2017,
  author    = {Loshchilov, Ilya and Hutter, Frank},
  title     = {{SGDR}: Stochastic Gradient Descent with Warm Restarts},
  booktitle = {5th International Conference on Learning Representations (ICLR)},
  year      = {2017}
}

@inproceedings{LoshchilovHutter2019,
  author    = {Loshchilov, Ilya and Hutter, Frank},
  title     = {Decoupled Weight Decay Regularization},
  booktitle = {7th International Conference on Learning Representations (ICLR)},
  year      = {2019}
}

@inproceedings{BergstraBardenetBengioKegl2011,
  author    = {Bergstra, James and Bardenet, R{\'e}mi and Bengio, Yoshua and K{\'e}gl, Bal{\'a}zs},
  title     = {Algorithms for Hyper-Parameter Optimization},
  booktitle = {Advances in Neural Information Processing Systems 24},
  year      = {2011},
  pages     = {2546--2554}
}

@article{BergstraBengio2012,
  author  = {Bergstra, James and Bengio, Yoshua},
  title   = {Random Search for Hyper-Parameter Optimization},
  journal = {Journal of Machine Learning Research},
  year    = {2012},
  volume  = {13},
  pages   = {281--305}
}
```

Where each is used: BNS 2002 and Taylor and Xu 1997 (core section 2); Newey and West 1987 (05:438, 06:61, every
HAC note); Zou 2006 and Friedman, Hastie and Tibshirani 2010 (core section 3, penalty weights and $\alpha$); Friedman
2001 (05:197); Breiman 2001 (05:212); Hastie, Tibshirani and Friedman 2009 (B.1 #14); Hansen 2005, White 2000,
Romano and Wolf 2005 (05:454-455, 02:60); Ledoit and Wolf 2008 (07:134); Fleming, Kirby and Ostdiek 2001 (economic
value of volatility forecasts, Economic application); Welch and Goyal 2008 (core section 7, cumulative-loss figure);
Campbell and Thompson 2008 (06, out-of-sample $R^2$); Bollerslev, Tauchen and Zhou 2009 and Carr and Wu 2009
(variance risk premium, 01:51-53, 02, 07); BNKS 2010, Patton and Sheppard 2015, Amaya et al. 2015, Forsberg and
Ghysels 2007, Hansen and Lunde 2006, Ni, Pan and Poteshman 2008 (section A series names; 03_data:30-31 already names
BNKS 2010 in plain text); Friedman and Popescu 2008, Kendall and Babington Smith 1939 (08); Feng et al. 2024, Jordan
et al. 2024 (05:223-224); Kingma and Ba 2015, Loshchilov and Hutter 2017 and 2019 (05:279-307: Adam, cosine
schedule, decoupled weight decay); Bergstra et al. 2011 and Bergstra and Bengio 2012 (TPE and random search, 05:383-387
and the tuning tables).

### D.2 The draft's own to-add list (02_related_work:50, 60): verify before adding

```bibtex
@article{Bucci2020,
  author  = {Bucci, Andrea},
  title   = {Realized Volatility Forecasting with Neural Networks},
  journal = {Journal of Financial Econometrics},
  year    = {2020},
  volume  = {18},
  number  = {3},
  pages   = {502--531}
}

@article{ZhangZhangCucuringuQian2024,
  author  = {Zhang, Chao and Zhang, Yihuang and Cucuringu, Mihai and Qian, Zhongmin},
  title   = {Volatility Forecasting with Machine Learning and Intraday Commonality},
  journal = {Journal of Financial Econometrics},
  year    = {2024},
  volume  = {22},
  number  = {2},
  pages   = {492--530}
}
```

- **Working papers on the draft's list:** Bandi, Fusari and Renò (2023, 0DTE option pricing); Beckmeyer, Branger
  and Gayda (2023, retail 0DTE trading); Brogaard, Han and Won (2023, 0DTE trading and underlying volatility); Dim,
  Eraker and Vilkov (2024, 0DTE gamma risk); Almeida, Freire and Hizmeri (2025); Rahimikia and Poon (2020, ML for
  realized volatility). All are SSRN working papers whose current titles and versions should be taken from SSRN, so
  no BibTeX is given here.
