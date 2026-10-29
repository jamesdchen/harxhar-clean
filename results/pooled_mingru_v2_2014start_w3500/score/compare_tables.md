### MinGRU v2, W = 3,500: full set vs HAR set (PRE-FOMC-FIX PREVIEW for every full-set number)

Negative % / DM = the full set more accurate. DM: HLN, daily mean loss, NW lag 8 (EVENT: automatic lag in brackets).

| span | QLIKE full [PREVIEW] | QLIKE HAR | % QLIKE (DM, p) [PREVIEW] | % MSE (DM, p) [PREVIEW] |
|---|---|---|---|---|
| every bar | 0.16154 | 0.18000 | -10.25 (-10.82, <0.001) | +29.26 (+1.09, 0.274) |
| 16:00 bar | 0.11327 | 0.12037 | -5.90 (-3.16, 0.002) | +19.56 (+0.90, 0.368) |
| 16:00 bar 2020-24 (Table 1's column) | 0.10644 | 0.10905 | -2.39 (-0.61, 0.541) | +23.44 (+0.97, 0.333) |
| ORDINARY days | 0.15233 | 0.16275 | -6.41 (-8.46, <0.001) | +32.18 (+1.12, 0.263) |
| EVENT days (81) | 0.44581 | 0.71189 | -37.38 (-7.93, <0.001) [lag 3: -7.67, <0.001] | -9.81 (-1.73, 0.083) |

Gain shares [PRE-FOMC-FIX PREVIEW] (Section 6.2's definition; sum over bars of HAR loss - full loss = 618.79): EVENT days 45.3 %, EVENT afternoons (bars ending 14:30-16:00) 46.2 %, ORDINARY days 54.7 % (EVENT afternoons hold 0.97 % of the bars).

### Context, W = 3,500: full vs HAR, % QLIKE (DM, p); linear classes against the OLS on the HAR set (D2)

| model | design | every bar | 16:00 | 16:00 2020-24 | ORDINARY | EVENT | EVENT pm share % |
|---|---|---|---|---|---|---|---|
| Ridge | pre-patch | -6.33 (-7.06, <0.001) | -3.07 (-1.55, 0.120) | +1.96 (+0.49, 0.627) | -4.11 (-4.70, <0.001) | -19.13 (-7.18, <0.001) | 49.3 |
| Lasso | pre-patch | -7.62 (-6.39, <0.001) | -1.55 (-0.69, 0.488) | +4.10 (+0.87, 0.386) | -3.52 (-3.15, 0.002) | -31.41 (-7.54, <0.001) | 63.7 |
| Elastic net | pre-patch | -6.69 (-8.45, <0.001) | -3.89 (-2.15, 0.032) | +1.06 (+0.30, 0.767) | -4.75 (-6.40, <0.001) | -17.93 (-7.26, <0.001) | 44.2 |
| LightGBM | pre-patch | -13.98 (-10.02, <0.001) | -2.38 (-1.62, 0.106) | -2.13 (-0.81, 0.418) | -5.74 (-10.36, <0.001) | -62.78 (-8.36, <0.001) | 64.6 |
| XGBoost | pre-patch | -13.45 (-9.97, <0.001) | -3.25 (-2.19, 0.029) | -2.86 (-1.02, 0.309) | -5.80 (-10.64, <0.001) | -57.61 (-6.70, <0.001) | 62.6 |
| Random forest | pre-patch | -11.19 (-9.40, <0.001) | -1.81 (-1.15, 0.251) | +0.55 (+0.18, 0.858) | -5.08 (-9.75, <0.001) | -45.21 (-8.84, <0.001) | 60.7 |
| MinGRU v2 | v2 [PRE-FOMC-FIX PREVIEW] | -10.25 (-10.82, <0.001) | -5.90 (-3.16, 0.002) | -2.39 (-0.61, 0.541) | -6.41 (-8.46, <0.001) | -37.38 (-7.93, <0.001) | 46.2 |

### QLIKE levels, W = 3,500 (every bar / 16:00)

| model | design | full every bar | full 16:00 | HAR every bar | HAR 16:00 |
|---|---|---|---|---|---|
| OLS | pre-patch | - | - | 0.18967 | 0.12723 |
| Ridge | pre-patch | 0.17767 | 0.12333 | 0.18967 | 0.12723 |
| Lasso | pre-patch | 0.17520 | 0.12525 | 0.18967 | 0.12723 |
| Elastic net | pre-patch | 0.17698 | 0.12229 | 0.18967 | 0.12723 |
| LightGBM | pre-patch | 0.16134 | 0.12037 | 0.18756 | 0.12331 |
| XGBoost | pre-patch | 0.16356 | 0.11918 | 0.18899 | 0.12318 |
| Random forest | pre-patch | 0.16854 | 0.12414 | 0.18977 | 0.12642 |
| MinGRU v2 | v2 (full: PRE-FOMC-FIX PREVIEW) | 0.16154 | 0.11327 | 0.18000 | 0.12037 |

### Economics, W = 3,500: the sign strategy, paper Sharpe ratio (D19) on the 614 post-warmup days; differences tested by Ledoit-Wolf (p; paired block bootstrap p_boot alongside)

Short straddle: mid 0.322, bid/ask -0.032.

| model | design | test | side | SR a | SR b | delta | p (LW) | p_boot |
|---|---|---|---|---|---|---|---|---|
| Ridge | pre-patch | full - HAR | mid | 1.837 | 1.005 | +0.832 | 0.069 | 0.066 |
| Ridge | pre-patch | full - short | mid | 1.837 | 0.322 | +1.515 | 0.026 | 0.012 |
| Ridge | pre-patch | har - short | mid | 1.005 | 0.322 | +0.683 | 0.320 | 0.281 |
| Ridge | pre-patch | full - HAR | bid/ask | 1.481 | 0.653 | +0.828 | 0.068 | 0.065 |
| Ridge | pre-patch | full - short | bid/ask | 1.481 | -0.032 | +1.513 | 0.027 | 0.011 |
| Ridge | pre-patch | har - short | bid/ask | 0.653 | -0.032 | +0.685 | 0.313 | 0.275 |
| Lasso | pre-patch | full - HAR | mid | 1.797 | 1.005 | +0.792 | 0.101 | 0.104 |
| Lasso | pre-patch | full - short | mid | 1.797 | 0.322 | +1.475 | 0.033 | 0.016 |
| Lasso | pre-patch | har - short | mid | 1.005 | 0.322 | +0.683 | 0.320 | 0.281 |
| Lasso | pre-patch | full - HAR | bid/ask | 1.441 | 0.653 | +0.788 | 0.101 | 0.104 |
| Lasso | pre-patch | full - short | bid/ask | 1.441 | -0.032 | +1.473 | 0.033 | 0.014 |
| Lasso | pre-patch | har - short | bid/ask | 0.653 | -0.032 | +0.685 | 0.313 | 0.275 |
| Elastic net | pre-patch | full - HAR | mid | 2.004 | 1.005 | +0.998 | 0.027 | 0.023 |
| Elastic net | pre-patch | full - short | mid | 2.004 | 0.322 | +1.682 | 0.012 | 0.003 |
| Elastic net | pre-patch | har - short | mid | 1.005 | 0.322 | +0.683 | 0.320 | 0.281 |
| Elastic net | pre-patch | full - HAR | bid/ask | 1.648 | 0.653 | +0.995 | 0.027 | 0.022 |
| Elastic net | pre-patch | full - short | bid/ask | 1.648 | -0.032 | +1.680 | 0.012 | 0.003 |
| Elastic net | pre-patch | har - short | bid/ask | 0.653 | -0.032 | +0.685 | 0.313 | 0.275 |
| LightGBM | pre-patch | full - HAR | mid | 1.540 | 0.590 | +0.950 | 0.035 | 0.029 |
| LightGBM | pre-patch | full - short | mid | 1.540 | 0.322 | +1.218 | 0.065 | 0.040 |
| LightGBM | pre-patch | har - short | mid | 0.590 | 0.322 | +0.268 | 0.682 | 0.665 |
| LightGBM | pre-patch | full - HAR | bid/ask | 1.187 | 0.242 | +0.945 | 0.036 | 0.030 |
| LightGBM | pre-patch | full - short | bid/ask | 1.187 | -0.032 | +1.219 | 0.063 | 0.037 |
| LightGBM | pre-patch | har - short | bid/ask | 0.242 | -0.032 | +0.274 | 0.673 | 0.653 |
| XGBoost | pre-patch | full - HAR | mid | 1.347 | 0.777 | +0.570 | 0.151 | 0.129 |
| XGBoost | pre-patch | full - short | mid | 1.347 | 0.322 | +1.025 | 0.106 | 0.078 |
| XGBoost | pre-patch | har - short | mid | 0.777 | 0.322 | +0.455 | 0.499 | 0.472 |
| XGBoost | pre-patch | full - HAR | bid/ask | 0.994 | 0.429 | +0.564 | 0.155 | 0.131 |
| XGBoost | pre-patch | full - short | bid/ask | 0.994 | -0.032 | +1.026 | 0.102 | 0.073 |
| XGBoost | pre-patch | har - short | bid/ask | 0.429 | -0.032 | +0.461 | 0.486 | 0.462 |
| Random forest | pre-patch | full - HAR | mid | 1.828 | 0.802 | +1.026 | 0.010 | 0.009 |
| Random forest | pre-patch | full - short | mid | 1.828 | 0.322 | +1.506 | 0.023 | 0.009 |
| Random forest | pre-patch | har - short | mid | 0.802 | 0.322 | +0.480 | 0.469 | 0.444 |
| Random forest | pre-patch | full - HAR | bid/ask | 1.470 | 0.452 | +1.018 | 0.010 | 0.009 |
| Random forest | pre-patch | full - short | bid/ask | 1.470 | -0.032 | +1.502 | 0.023 | 0.009 |
| Random forest | pre-patch | har - short | bid/ask | 0.452 | -0.032 | +0.484 | 0.461 | 0.435 |
| MinGRU v2 | v2 [final] | har - short | mid | 0.882 | 0.322 | +0.560 | 0.457 | 0.434 |
| MinGRU v2 | v2 [final] | har - short | bid/ask | 0.538 | -0.032 | +0.570 | 0.443 | 0.421 |
| MinGRU v2 | v2 [PRE-FOMC-FIX PREVIEW] | full - short | mid | 1.449 | 0.322 | +1.127 | 0.109 | 0.090 |
| MinGRU v2 | v2 [PRE-FOMC-FIX PREVIEW] | full - short | bid/ask | 1.103 | -0.032 | +1.135 | 0.103 | 0.087 |
| MinGRU v2 | v2 [PRE-FOMC-FIX PREVIEW] | full - HAR | mid | 1.449 | 0.882 | +0.567 | 0.270 | 0.253 |
| MinGRU v2 | v2 [PRE-FOMC-FIX PREVIEW] | full - HAR | bid/ask | 1.103 | 0.538 | +0.565 | 0.273 | 0.256 |
