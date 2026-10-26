# Variable importance: meeting brief (2026-10-26)

Final pooled design: 13 regular-hours bars, predictors on the full 48-bar grid, W = 3,000 days, 564 columns
(12 HAR-benchmark + 46 exogenous series x 12). FOMC features are pre-patch (a fix for the missing scheduled
18 Mar 2020 meeting and the winsorized proximity ramps is being measured); FOMC numbers are preliminary.

## 1. What the trees use (TreeSHAP, 126 monthly origins, 2014-24; Section 8)

- HAR components 52-58% of the attribution; the futures' own realized moments 21-24%.
- Non-standard series 19-22% of the total (about 45% of the exogenous part), concentrated in five: constituent
  absolute returns (value- and equal-weighted), time to the next FOMC statement, order-flow imbalance, effective spread.
- Trees lean on the non-standard series LESS than the linear models (45% vs 56-74% of the exogenous total).
- Conditional: the non-standard share doubles on FOMC statement days (21% -> 40%), is highest at the first bar and at
  14:30-15:00, lower in volatile markets; responses are monotone and one-sided; about half acts through interactions.

## 2. Alternative data, read in the years they are live (Figure fi_livewindow.png)

| Group | Share before live | Share in full live years | Live share of training window at the last origin |
|---|---|---|---|
| StockTwits | 0.5-0.6% | 0.8-1.3% (2022-24) | 27% |
| Volatility demand | 0.1-0.2% | 0.6-0.9% (2023) | 7% |
| VIX3M | 0.2-0.3% | 0.4-1.6% (2022-24) | 24% |
| VIX | 1.1-1.7% | 0.8-1.2% (no rise) | 51% |

The pooled 2014-24 share understates them: the late-start rule keeps their values at zero until a full window
postdates their first print. SHAP measures use, not accuracy. Linear ablations on the live windows: StockTwits
-1.0 to -1.3% QLIKE (n.s.); VIX / VIX3M about 0 to +1.4% (n.s.); volatility demand HURTS (+2.8 to +4.3%, DM about +2.3).

## 3. Feature-count ladder (Maisch et al. style; Figure ladder.png)

Each model refit on only its top-k exogenous series (ranked on the 2013 pre-sample, no look-ahead, as Maisch et al.
rank on their first training window), k = 46 / 20 / 10 / 5 / 3 / 1 / 0 (= HAR), at the 126 monthly origins.
R_k = share of the full set's QLIKE gain over HAR kept with k series (every bar).

| Model | R20 | R10 | R5 | R3 | R1 | Reading |
|---|---|---|---|---|---|---|
| LightGBM | 96% | 95% | 34% | 34% | 3% | ~10 series |
| XGBoost | 99% | 97% | 38% | 28% | 4% | ~10 series |
| Random forest | 100% | 98% | 41% | 43% | 6% | ~10 series |
| Ridge | 200% | 216% | 211% | 94% | 70% | 3-10 series, fewer is better |
| Lasso | 219% | 231% | 190% | 122% | 105% | 1-10 series |
| Elastic net | 125% | 134% | 132% | 91% | -2% | 3-10 series |

- **Not dense and weak for either model class.** No model needs more than 10 of the 46 series for ~95% of its gain.
- Trees: the top 10 keep 95-98%; the top 5 only 34-41%. Going 5 -> 10 adds time-to-FOMC and the futures' realized skew
  in all three trees (plus VIX in the boosters). The other 36 series add a significant ~4% in LightGBM, nothing
  detectable in XGBoost or the forest.
- Linear: 5-10 series give about twice the full set's gain; the lasso with time-to-FOMC alone matches the full set.
  Caveat: part of this is the instability of a 564-column linear fit held for a month (the every-bar walks of
  Table 1 are as good as the reduced sets).
- As in Maisch et al., the full set is not the best set for the linear models; for the trees it is, by a hair.
- 16:00 bar: the trees' gain is small there and concentrated (3-5 series).

Caveats: monthly refits held 21 sessions (not the every-bar walks); FOMC pre-patch (it leads the linear rankings and is
in every tree's top 10); one 2013 ranking; many DM comparisons, no multiplicity correction.

## 4. Ablations (pre-patch design; QLIKE % change, every bar, 2014-24, 33,527 bars)

| | LightGBM | XGBoost | Forest | Linear (ridge/lasso/enet) |
|---|---|---|---|---|
| Full vs HAR | -14.0 | -13.8 | -11.1 | -5.4 / -6.2 / -6.0 |
| Full vs HAR + futures' realized moments | -11.3 | -10.8 | -8.2 | -2.8 / -4.2 / -3.4 |
| Full vs OLS SHAR+HARQ+LHAR | -12.4 | -11.6 | -8.4 | -2.7 / -3.6 / -3.4 |
| Remove FOMC calendar | +10.1 | +9.3 | (not run) | +2.3 to +3.6 |
| Remove realized moments | +1.5 | +1.6 | +1.5 | +1.3 to +1.9 |
| Remove any other group | +-0.2 | +-0.2 | +-0.2 | about 0 |
| Add FOMC calendar alone to HAR | -9.0 | -8.8 | -7.5 | -2.4 to -3.8 |
| Add realized moments alone | -3.0 | -3.4 | -3.2 | -2.1 to -2.7 |
| Add StockTwits / implied vol / vol demand | -0.9 to -1.7 / -0.2 to -1.6 / -0.6 to -1.2 | | | vol demand hurts |

- The exogenous gain survives every augmented-HAR benchmark (trees 8-12%, linear 3-4%), all significant.
- About two thirds of the trees' gain is the FOMC calendar (a scheduled-announcement effect HAR cannot see); the
  futures' realized moments are second; the constituent-market and liquidity groups help alone but overlap with them.
- Alternative data add little even when live: StockTwits -1.0 to +0.1 (n.s.), VIX about -1 (XGBoost significant),
  volatility demand about 0 for trees, harmful for linear models.
- SHAP shares understate the FOMC calendar (1-2% of attribution, ~9% of accuracy): use is not accuracy.
- Next: report full vs HAR + FOMC explicitly; rerun the FOMC arms on the patched design (the FOMC fix is material
  for the linear models: every-bar gains rise to -6.0 / -9.0 / -6.7%; trees -14.1 / -14.6%).

## 5. Next

Window x feature-count grid (data and feature scaling; "complexity" P/T in the sense of Kelly, Malamud and Zhou 2024)
running now; FOMC patch measurement; integration into Section 8.
