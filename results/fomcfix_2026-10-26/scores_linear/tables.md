# fomcfix scores (main worktree 2d1b4b0); Table 1's 33,527 bars on 2,579 days, 16:00 on 2,579 days

QLIKE full / HAR, % vs HAR = 100 (full / HAR - 1), DM (HLN) full vs HAR (negative = full lower) with p; D6 Sharpe (614 post-warmup option days) full / HAR mid, bid/ask full, bootstrap p of full - HAR (mid).

| family | class | cell | QLIKE full bar | HAR bar | % bar | DM bar (p) | QLIKE full 16:00 | HAR 16:00 | % 16:00 | DM 16:00 (p) | D6 full / HAR mid | D6 b/a full / HAR | ΔD6 mid (p) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| linear | Elastic net | A | 0.17784 | 0.18927 | -6.04 | -6.98 (0.000) | 0.12622 | 0.12687 | -0.52 | -0.21 (0.832) | 1.94 / 0.98 | 1.58 / 0.62 | +0.96 (0.135) |
| linear | Elastic net | B | 0.17660 | 0.18927 | -6.69 | -7.36 (0.000) | 0.12764 | 0.12687 | +0.60 | +0.24 (0.807) | 1.95 / 0.98 | 1.59 / 0.62 | +0.97 (0.132) |
| linear | Elastic net | C | 0.17809 | 0.18927 | -5.91 | -6.71 (0.000) | 0.12488 | 0.12648 | -1.27 | -0.56 (0.579) | 1.67 / 1.01 | 1.31 / 0.65 | +0.66 (0.255) |
| linear | Elastic net | D | 0.17682 | 0.18927 | -6.58 | -7.04 (0.000) | 0.12642 | 0.12648 | -0.05 | -0.02 (0.984) | 1.61 / 1.01 | 1.25 / 0.65 | +0.60 (0.310) |
| linear | Lasso | A | 0.17747 | 0.18927 | -6.23 | -4.72 (0.000) | 0.13104 | 0.12687 | +3.28 | +1.04 (0.300) | 1.72 / 0.98 | 1.36 / 0.62 | +0.74 (0.227) |
| linear | Lasso | B | 0.17230 | 0.18927 | -8.97 | -6.34 (0.000) | 0.13225 | 0.12687 | +4.24 | +1.35 (0.178) | 1.75 / 0.98 | 1.40 / 0.62 | +0.77 (0.206) |
| linear | Lasso | C | 0.17774 | 0.18927 | -6.09 | -4.76 (0.000) | 0.13001 | 0.12648 | +2.79 | +0.93 (0.353) | 2.22 / 1.01 | 1.86 / 0.65 | +1.21 (0.082) |
| linear | Lasso | D | 0.17287 | 0.18927 | -8.67 | -6.08 (0.000) | 0.13159 | 0.12648 | +4.04 | +1.33 (0.185) | 2.14 / 1.01 | 1.79 / 0.65 | +1.13 (0.108) |
| linear | Ridge | A | 0.17912 | 0.18927 | -5.36 | -5.63 (0.000) | 0.12789 | 0.12687 | +0.80 | +0.31 (0.759) | 1.85 / 0.98 | 1.49 / 0.62 | +0.86 (0.161) |
| linear | Ridge | B | 0.17783 | 0.18927 | -6.04 | -6.28 (0.000) | 0.12913 | 0.12687 | +1.78 | +0.67 (0.500) | 1.75 / 0.98 | 1.40 / 0.62 | +0.77 (0.217) |
| linear | Ridge | C | 0.17939 | 0.18927 | -5.22 | -5.39 (0.000) | 0.12626 | 0.12648 | -0.17 | -0.07 (0.943) | 1.93 / 1.01 | 1.58 / 0.65 | +0.92 (0.194) |
| linear | Ridge | D | 0.17806 | 0.18927 | -5.92 | -6.03 (0.000) | 0.12764 | 0.12648 | +0.91 | +0.37 (0.712) | 1.79 / 1.01 | 1.44 / 0.65 | +0.78 (0.248) |

## Effects (differences of the quantities above)

| family | class | effect | ΔQLIKE full bar | Δ% vs HAR bar | ΔDM bar | ΔQLIKE full 16:00 | Δ% vs HAR 16:00 | ΔDM 16:00 | ΔD6 mid full | ΔD6 mid HAR |
|---|---|---|---|---|---|---|---|---|---|---|
| linear | Ridge | B-A | -0.001284 | -0.679 | -0.656 | +0.001235 | +0.973 | +0.367 | -0.092 | +0.000 |
| linear | Ridge | D-C | -0.001326 | -0.700 | -0.641 | +0.001373 | +1.085 | +0.440 | -0.137 | +0.000 |
| linear | Ridge | C-A | +0.000266 | +0.138 | +0.235 | -0.001628 | -0.976 | -0.378 | +0.086 | +0.028 |
| linear | Ridge | D-B | +0.000225 | +0.117 | +0.250 | -0.001490 | -0.864 | -0.306 | +0.041 | +0.028 |
| linear | Ridge | (D-C)-(B-A) | -0.000041 | -0.022 | +0.015 | +0.000138 | +0.112 | +0.073 | -0.045 | +0.000 |
| linear | Lasso | B-A | -0.005174 | -2.734 | -1.613 | +0.001206 | +0.951 | +0.309 | +0.031 | +0.000 |
| linear | Lasso | D-C | -0.004867 | -2.571 | -1.320 | +0.001578 | +1.248 | +0.397 | -0.075 | +0.000 |
| linear | Lasso | C-A | +0.000267 | +0.139 | -0.035 | -0.001028 | -0.494 | -0.108 | +0.495 | +0.028 |
| linear | Lasso | D-B | +0.000574 | +0.301 | +0.257 | -0.000657 | -0.197 | -0.021 | +0.389 | +0.028 |
| linear | Lasso | (D-C)-(B-A) | +0.000307 | +0.162 | +0.293 | +0.000372 | +0.297 | +0.087 | -0.106 | +0.000 |
| linear | Elastic net | B-A | -0.001242 | -0.656 | -0.380 | +0.001416 | +1.116 | +0.457 | +0.012 | +0.000 |
| linear | Elastic net | D-C | -0.001276 | -0.674 | -0.331 | +0.001546 | +1.222 | +0.535 | -0.064 | +0.000 |
| linear | Elastic net | C-A | +0.000252 | +0.131 | +0.272 | -0.001342 | -0.754 | -0.343 | -0.266 | +0.028 |
| linear | Elastic net | D-B | +0.000218 | +0.113 | +0.322 | -0.001212 | -0.647 | -0.265 | -0.341 | +0.028 |
| linear | Elastic net | (D-C)-(B-A) | -0.000034 | -0.018 | +0.049 | +0.000131 | +0.107 | +0.079 | -0.075 | +0.000 |

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
  "classes": {
   "ridge": {
    "sign_flip_bar": false,
    "cross_5pct_bar": false,
    "qlike_full_change_pct_bar": -0.7389657278364603,
    "qlike_change_gt_0.5pct_bar": true,
    "sign_flip_c16": true,
    "cross_5pct_c16": false,
    "qlike_full_change_pct_c16": 1.0873829731314455,
    "qlike_change_gt_0.5pct_c16": true,
    "d6_mid_full_change": -0.137391002400159,
    "d6_ba_full_change": -0.13932179734847194,
    "d6_change_gt_0.10": true,
    "material": true
   },
   "lasso": {
    "sign_flip_bar": false,
    "cross_5pct_bar": false,
    "qlike_full_change_pct_bar": -2.7382807001074427,
    "qlike_change_gt_0.5pct_bar": true,
    "sign_flip_c16": false,
    "cross_5pct_c16": false,
    "qlike_full_change_pct_c16": 1.2138202354257333,
    "qlike_change_gt_0.5pct_c16": true,
    "d6_mid_full_change": -0.07498222278609878,
    "d6_ba_full_change": -0.07626633621150591,
    "d6_change_gt_0.10": false,
    "material": true
   },
   "enet": {
    "sign_flip_bar": false,
    "cross_5pct_bar": false,
    "qlike_full_change_pct_bar": -0.7163294899407369,
    "qlike_change_gt_0.5pct_bar": true,
    "sign_flip_c16": false,
    "cross_5pct_c16": false,
    "qlike_full_change_pct_c16": 1.2381257370493781,
    "qlike_change_gt_0.5pct_c16": true,
    "d6_mid_full_change": -0.06386054437510724,
    "d6_ba_full_change": -0.06151802228112446,
    "d6_change_gt_0.10": false,
    "material": true
   }
  },
  "material": true
 },
 "holiday_C_vs_A_information": {
  "cells": "C vs A",
  "classes": {
   "ridge": {
    "sign_flip_bar": false,
    "cross_5pct_bar": false,
    "qlike_full_change_pct_bar": 0.14837430156238174,
    "qlike_change_gt_0.5pct_bar": false,
    "sign_flip_c16": true,
    "cross_5pct_c16": false,
    "qlike_full_change_pct_c16": -1.2731288177295075,
    "qlike_change_gt_0.5pct_c16": true,
    "d6_mid_full_change": 0.08626766735897329,
    "d6_ba_full_change": 0.08883953713903314,
    "d6_change_gt_0.10": false,
    "material": true
   },
   "lasso": {
    "sign_flip_bar": false,
    "cross_5pct_bar": false,
    "qlike_full_change_pct_bar": 0.15036029710173704,
    "qlike_change_gt_0.5pct_bar": false,
    "sign_flip_c16": false,
    "cross_5pct_c16": false,
    "qlike_full_change_pct_c16": -0.7847261667438987,
    "qlike_change_gt_0.5pct_c16": true,
    "d6_mid_full_change": 0.4949453014759593,
    "d6_ba_full_change": 0.49791303306033474,
    "d6_change_gt_0.10": true,
    "material": true
   },
   "enet": {
    "sign_flip_bar": false,
    "cross_5pct_bar": false,
    "qlike_full_change_pct_bar": 0.1417082937851255,
    "qlike_change_gt_0.5pct_bar": false,
    "sign_flip_c16": false,
    "cross_5pct_c16": false,
    "qlike_full_change_pct_c16": -1.0636143343008553,
    "qlike_change_gt_0.5pct_c16": true,
    "d6_mid_full_change": -0.26583751399127875,
    "d6_ba_full_change": -0.26756794101791703,
    "d6_change_gt_0.10": true,
    "material": true
   }
  },
  "material": true
 },
 "holiday_D_vs_B_information": {
  "cells": "D vs B",
  "classes": {
   "ridge": {
    "sign_flip_bar": false,
    "cross_5pct_bar": false,
    "qlike_full_change_pct_bar": 0.12629311500045315,
    "qlike_change_gt_0.5pct_bar": false,
    "sign_flip_c16": false,
    "cross_5pct_c16": false,
    "qlike_full_change_pct_c16": -1.1539065963063977,
    "qlike_change_gt_0.5pct_c16": true,
    "d6_mid_full_change": 0.04100196474530571,
    "d6_ba_full_change": 0.039016808094517685,
    "d6_change_gt_0.10": false,
    "material": true
   },
   "lasso": {
    "sign_flip_bar": false,
    "cross_5pct_bar": false,
    "qlike_full_change_pct_bar": 0.33303081409157276,
    "qlike_change_gt_0.5pct_bar": false,
    "sign_flip_c16": false,
    "cross_5pct_c16": false,
    "qlike_full_change_pct_c16": -0.4964394639410563,
    "qlike_change_gt_0.5pct_c16": false,
    "d6_mid_full_change": 0.3889465154641445,
    "d6_ba_full_change": 0.390084816432398,
    "d6_change_gt_0.10": true,
    "material": true
   },
   "enet": {
    "sign_flip_bar": false,
    "cross_5pct_bar": false,
    "qlike_full_change_pct_bar": 0.12370973286295328,
    "qlike_change_gt_0.5pct_bar": false,
    "sign_flip_c16": true,
    "cross_5pct_c16": false,
    "qlike_full_change_pct_c16": -0.949552269684073,
    "qlike_change_gt_0.5pct_c16": true,
    "d6_mid_full_change": -0.3412358993609059,
    "d6_ba_full_change": -0.3400140877918263,
    "d6_change_gt_0.10": true,
    "material": true
   }
  },
  "material": true
 },
 "trees_B_vs_A_information": {
  "cells": "B vs A",
  "classes": {},
  "material": false
 }
}
```

## Gates S: PASS (60 checks)
