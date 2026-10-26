# fomcfix scores (main worktree 2d1b4b0); Table 1's 33,527 bars on 2,579 days, 16:00 on 2,579 days

QLIKE full / HAR, % vs HAR = 100 (full / HAR - 1), DM (HLN) full vs HAR (negative = full lower) with p; D6 Sharpe (614 post-warmup option days) full / HAR mid, bid/ask full, bootstrap p of full - HAR (mid).

| family | class | cell | QLIKE full bar | HAR bar | % bar | DM bar (p) | QLIKE full 16:00 | HAR 16:00 | % 16:00 | DM 16:00 (p) | D6 full / HAR mid | D6 b/a full / HAR | ΔD6 mid (p) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| extra | Elastic net | nbr_enet_0.0001 | 0.17281 | 0.18927 | -8.70 | -5.35 (0.000) | 0.13354 | 0.12687 | +5.25 | +1.54 (0.123) | 1.76 / 0.98 | 1.41 / 0.62 | +0.78 (0.195) |
| extra | Elastic net | nbr_enet_0.01 | 0.17860 | 0.18927 | -5.64 | -10.98 (0.000) | 0.12075 | 0.12687 | -4.82 | -3.67 (0.000) | 1.78 / 0.98 | 1.42 / 0.62 | +0.80 (0.101) |
| extra | Lasso | nbr_lasso_0.001 | 0.17613 | 0.18927 | -6.94 | -8.74 (0.000) | 0.12498 | 0.12687 | -1.49 | -0.74 (0.462) | 2.01 / 0.98 | 1.65 / 0.62 | +1.03 (0.095) |
| extra | Lasso | nbr_lasso_1e-05 | 0.17355 | 0.18927 | -8.31 | -3.57 (0.000) | 0.13609 | 0.12687 | +7.26 | +1.85 (0.065) | 1.76 / 0.98 | 1.40 / 0.62 | +0.77 (0.208) |
| extra | Ridge | nbr_ridge_100 | 0.17688 | 0.18927 | -6.54 | -4.81 (0.000) | 0.13208 | 0.12687 | +4.10 | +1.21 (0.226) | 1.64 / 0.98 | 1.28 / 0.62 | +0.66 (0.273) |
| extra | Ridge | nbr_ridge_10000 | 0.17849 | 0.18927 | -5.69 | -8.34 (0.000) | 0.12546 | 0.12687 | -1.12 | -0.61 (0.544) | 1.93 / 0.98 | 1.57 / 0.62 | +0.94 (0.090) |
| linear | Elastic net | A | 0.17784 | 0.18927 | -6.04 | -6.98 (0.000) | 0.12622 | 0.12687 | -0.52 | -0.21 (0.832) | 1.94 / 0.98 | 1.58 / 0.62 | +0.96 (0.135) |
| linear | Elastic net | ANN | 0.17756 | 0.18927 | -6.18 | -5.80 (0.000) | 0.12429 | 0.12687 | -2.04 | -1.00 (0.319) | 1.51 / 0.98 | 1.14 / 0.62 | +0.52 (0.211) |
| linear | Elastic net | B | 0.17660 | 0.18927 | -6.69 | -7.36 (0.000) | 0.12764 | 0.12687 | +0.60 | +0.24 (0.807) | 1.95 / 0.98 | 1.59 / 0.62 | +0.97 (0.132) |
| linear | Lasso | A | 0.17747 | 0.18927 | -6.23 | -4.72 (0.000) | 0.13104 | 0.12687 | +3.28 | +1.04 (0.300) | 1.72 / 0.98 | 1.36 / 0.62 | +0.74 (0.227) |
| linear | Lasso | ANN | 0.17719 | 0.18927 | -6.38 | -5.28 (0.000) | 0.12450 | 0.12687 | -1.87 | -0.81 (0.418) | 1.53 / 0.98 | 1.17 / 0.62 | +0.55 (0.252) |
| linear | Lasso | B | 0.17230 | 0.18927 | -8.97 | -6.34 (0.000) | 0.13225 | 0.12687 | +4.24 | +1.35 (0.178) | 1.75 / 0.98 | 1.40 / 0.62 | +0.77 (0.206) |
| linear | Ridge | A | 0.17912 | 0.18927 | -5.36 | -5.63 (0.000) | 0.12789 | 0.12687 | +0.80 | +0.31 (0.759) | 1.85 / 0.98 | 1.49 / 0.62 | +0.86 (0.161) |
| linear | Ridge | ANN | 0.17368 | 0.18927 | -8.24 | -6.40 (0.000) | 0.12902 | 0.12687 | +1.69 | +0.68 (0.499) | 1.74 / 0.98 | 1.39 / 0.62 | +0.76 (0.219) |
| linear | Ridge | B | 0.17783 | 0.18927 | -6.04 | -6.28 (0.000) | 0.12913 | 0.12687 | +1.78 | +0.67 (0.500) | 1.75 / 0.98 | 1.40 / 0.62 | +0.77 (0.217) |

## Effects (differences of the quantities above)

| family | class | effect | ΔQLIKE full bar | Δ% vs HAR bar | ΔDM bar | ΔQLIKE full 16:00 | Δ% vs HAR 16:00 | ΔDM 16:00 | ΔD6 mid full | ΔD6 mid HAR |
|---|---|---|---|---|---|---|---|---|---|---|
| linear | Ridge | B-A | -0.001284 | -0.679 | -0.656 | +0.001235 | +0.973 | +0.367 | -0.092 | +0.000 |
| linear | Lasso | B-A | -0.005174 | -2.734 | -1.613 | +0.001206 | +0.951 | +0.309 | +0.031 | +0.000 |
| linear | Elastic net | B-A | -0.001242 | -0.656 | -0.380 | +0.001416 | +1.116 | +0.457 | +0.012 | +0.000 |

## Materiality (RULE.txt section 5)

```
{
 "fomc_B_vs_A": {
  "cells": "B vs A",
  "classes": {
   "ridge": {
    "sign_flip_bar": false,
    "cross_5pct_bar": false,
    "qlike_full_change_pct_bar": -0.7170753597033341,
    "qlike_change_gt_0.5pct_bar": true,
    "sign_flip_c16": false,
    "cross_5pct_c16": false,
    "qlike_full_change_pct_c16": 0.9654574428357154,
    "qlike_change_gt_0.5pct_c16": true,
    "d6_mid_full_change": -0.09212529978649142,
    "d6_ba_full_change": -0.08949906830395649,
    "d6_change_gt_0.10": false,
    "material": true
   },
   "lasso": {
    "sign_flip_bar": false,
    "cross_5pct_bar": false,
    "qlike_full_change_pct_bar": -2.915359458754263,
    "qlike_change_gt_0.5pct_bar": true,
    "sign_flip_c16": false,
    "cross_5pct_c16": false,
    "qlike_full_change_pct_c16": 0.9205784825019547,
    "qlike_change_gt_0.5pct_c16": true,
    "d6_mid_full_change": 0.031016563225716,
    "d6_ba_full_change": 0.03156188041643082,
    "d6_change_gt_0.10": false,
    "material": true
   },
   "enet": {
    "sign_flip_bar": false,
    "cross_5pct_bar": false,
    "qlike_full_change_pct_bar": -0.6984819371780415,
    "qlike_change_gt_0.5pct_bar": true,
    "sign_flip_c16": true,
    "cross_5pct_c16": false,
    "qlike_full_change_pct_c16": 1.1215444403051,
    "qlike_change_gt_0.5pct_c16": true,
    "d6_mid_full_change": 0.011537840994519932,
    "d6_ba_full_change": 0.010928124492784796,
    "d6_change_gt_0.10": false,
    "material": true
   }
  },
  "material": true
 },
 "fomc_D_vs_C": {
  "cells": "D vs C",
  "classes": {},
  "material": false
 },
 "holiday_C_vs_A_information": {
  "cells": "C vs A",
  "classes": {},
  "material": false
 },
 "holiday_D_vs_B_information": {
  "cells": "D vs B",
  "classes": {},
  "material": false
 },
 "trees_B_vs_A_information": {
  "cells": "B vs A",
  "classes": {},
  "material": false
 }
}
```

## Gates S: PASS (60 checks)
