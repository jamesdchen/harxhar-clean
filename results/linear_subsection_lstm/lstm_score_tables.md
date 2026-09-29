# LSTM 16:00 bar -- score tables (written by experiments/score_lstm_subsection.py)

## QLIKE on the stamps every forecast of the bucket shares (clock = research scorer)

| bucket | forecast | n | first | last | QLIKE_clock | QLIKE_as_scored | QLIKE_plain | max_day_QLIKE_clock | max_day_QLIKE_plain |
|---|---|---|---|---|---|---|---|---|---|
| all_features | avg_lstm_ridge | 1406 | 2018-09-24 | 2024-04-30 | 0.1262 | 0.1256 | 0.1265 | 7.8605 | 8.9683 |
| all_features | lstm | 1406 | 2018-09-24 | 2024-04-30 | 0.1703 | 0.1696 | 0.1801 | 7.9819 | 9.4922 |
| all_features | lstm_qsel | 1406 | 2018-09-24 | 2024-04-30 | 0.1692 | 0.1686 | 0.1789 | 7.9337 | 9.4922 |
| all_features | lasso | 1406 | 2018-09-24 | 2024-04-30 | 0.1173 | 0.1159 | 0.1153 | 7.1940 | 8.0483 |
| all_features | enet | 1406 | 2018-09-24 | 2024-04-30 | 0.1147 | 0.1134 | 0.1150 | 6.8663 | 7.6538 |
| all_features | ridge | 1406 | 2018-09-24 | 2024-04-30 | 0.1191 | 0.1202 | 2.6419 | 7.5422 | 3516.8077 |
| all_features | lgbm | 1406 | 2018-09-24 | 2024-04-30 | 0.1195 | 0.1180 | 0.1146 | 6.2228 | 6.9255 |
| all_features | xgb | 1406 | 2018-09-24 | 2024-04-30 | 0.1215 | 0.1199 | 0.1158 | 6.1977 | 6.9164 |
| all_features | rf | 1406 | 2018-09-24 | 2024-04-30 | 0.1241 | 0.1224 | 0.1179 | 5.6322 | 6.3121 |
| all_features | lgbm_tuned | 1406 | 2018-09-24 | 2024-04-30 | 0.1222 | 0.1204 | 0.1170 | 6.2233 | 6.9420 |
| all_features | xgb_tuned | 1406 | 2018-09-24 | 2024-04-30 | 0.1250 | 0.1229 | 0.1198 | 7.9049 | 8.9337 |
| all_features | rf_tuned | 1406 | 2018-09-24 | 2024-04-30 | 0.1248 | 0.1233 | 0.1193 | 5.9927 | 6.7464 |
| baseline | avg_lstm_ridge | 1406 | 2018-09-24 | 2024-04-30 | 0.1194 | 0.1180 | 0.1160 | 6.7986 | 7.6526 |
| baseline | lstm | 1406 | 2018-09-24 | 2024-04-30 | 0.1270 | 0.1254 | 0.1235 | 7.2800 | 8.3367 |
| baseline | lstm_qsel | 1406 | 2018-09-24 | 2024-04-30 | 0.1256 | 0.1241 | 0.1221 | 7.2669 | 8.3367 |
| baseline | ols | 1406 | 2018-09-24 | 2024-04-30 | 0.1167 | 0.1157 | 0.1161 | 6.5028 | 7.2839 |
| baseline | lasso | 1406 | 2018-09-24 | 2024-04-30 | 0.1174 | 0.1162 | 0.1155 | 6.9582 | 7.8459 |
| baseline | enet | 1406 | 2018-09-24 | 2024-04-30 | 0.1165 | 0.1153 | 0.1150 | 6.7984 | 7.6414 |
| baseline | ridge | 1406 | 2018-09-24 | 2024-04-30 | 0.1166 | 0.1155 | 0.1151 | 6.3020 | 7.0357 |
| baseline | lgbm | 1406 | 2018-09-24 | 2024-04-30 | 0.1250 | 0.1237 | 0.1209 | 5.1785 | 5.8371 |
| baseline | xgb | 1406 | 2018-09-24 | 2024-04-30 | 0.1239 | 0.1228 | 0.1189 | 5.7057 | 6.4306 |
| baseline | rf | 1406 | 2018-09-24 | 2024-04-30 | 0.1246 | 0.1232 | 0.1204 | 5.3375 | 6.0637 |
| baseline | lgbm_tuned | 1406 | 2018-09-24 | 2024-04-30 | 0.1305 | 0.1284 | 0.1253 | 4.7255 | 5.3602 |
| baseline | xgb_tuned | 1406 | 2018-09-24 | 2024-04-30 | 0.1306 | 0.1283 | 0.1247 | 5.1978 | 5.9191 |
| baseline | rf_tuned | 1406 | 2018-09-24 | 2024-04-30 | 0.1309 | 0.1286 | 0.1251 | 5.5669 | 6.2581 |
| live_feasible | avg_lstm_ridge | 1406 | 2018-09-24 | 2024-04-30 | 0.1212 | 0.1209 | 0.1269 | 9.2370 | 10.6032 |
| live_feasible | lstm | 1406 | 2018-09-24 | 2024-04-30 | 0.1468 | 0.1472 | 0.1620 | 10.1388 | 14.6649 |
| live_feasible | lstm_qsel | 1406 | 2018-09-24 | 2024-04-30 | 0.1427 | 0.1427 | 0.1478 | 10.3764 | 12.2807 |
| live_feasible | lasso | 1406 | 2018-09-24 | 2024-04-30 | 0.1158 | 0.1146 | 0.1135 | 7.1044 | 7.9330 |
| live_feasible | enet | 1406 | 2018-09-24 | 2024-04-30 | 0.1135 | 0.1124 | 0.1131 | 7.1237 | 7.9519 |
| live_feasible | ridge | 1406 | 2018-09-24 | 2024-04-30 | 0.1192 | 0.1208 | 0.1459 | 8.2754 | 10.3362 |
| live_feasible | lgbm | 1406 | 2018-09-24 | 2024-04-30 | 0.1181 | 0.1162 | 0.1123 | 5.7264 | 6.3378 |
| live_feasible | xgb | 1406 | 2018-09-24 | 2024-04-30 | 0.1202 | 0.1187 | 0.1150 | 6.3755 | 7.1338 |
| live_feasible | rf | 1406 | 2018-09-24 | 2024-04-30 | 0.1189 | 0.1175 | 0.1138 | 6.3537 | 7.0843 |
| live_feasible | lgbm_tuned | 1406 | 2018-09-24 | 2024-04-30 | 0.1263 | 0.1240 | 0.1192 | 6.1051 | 6.8181 |
| live_feasible | xgb_tuned | 1406 | 2018-09-24 | 2024-04-30 | 0.1243 | 0.1226 | 0.1189 | 6.8397 | 7.7222 |
| live_feasible | rf_tuned | 1406 | 2018-09-24 | 2024-04-30 | 0.1202 | 0.1188 | 0.1156 | 6.0164 | 6.6999 |

## Adjusted scale (the fitted target), same stamps

| bucket | forecast | n | mse_adj | corr_with_target | mz_slope | sd_forecast | corr_shifted_one_session | corr_with_ridge_forecast |
|---|---|---|---|---|---|---|---|---|
| all_features | avg_lstm_ridge | 1406 | 0.0642 | 0.8312 | 1.0378 | 0.3645 | 0.5192 | 0.9729 |
| all_features | lstm | 1406 | 0.0882 | 0.7580 | 0.9994 | 0.3452 | 0.4767 | 0.8726 |
| all_features | lstm_qsel | 1406 | 0.0892 | 0.7548 | 0.9867 | 0.3481 | 0.4801 | 0.8717 |
| all_features | lasso | 1406 | 0.0580 | 0.8489 | 0.9748 | 0.3963 | 0.5202 | 0.9867 |
| all_features | enet | 1406 | 0.0575 | 0.8506 | 0.9619 | 0.4024 | 0.5187 | 0.9901 |
| all_features | ridge | 1406 | 0.0602 | 0.8438 | 0.9412 | 0.4081 | 0.5243 | 1.0000 |
| all_features | lgbm | 1406 | 0.0619 | 0.8389 | 0.9525 | 0.4009 | 0.5222 | 0.9610 |
| all_features | xgb | 1406 | 0.0599 | 0.8434 | 0.9973 | 0.3849 | 0.5290 | 0.9628 |
| all_features | rf | 1406 | 0.0657 | 0.8298 | 0.9301 | 0.4061 | 0.5188 | 0.9503 |
| all_features | lgbm_tuned | 1406 | 0.0617 | 0.8388 | 1.0551 | 0.3618 | 0.5280 | 0.9536 |
| all_features | xgb_tuned | 1406 | 0.0628 | 0.8355 | 1.0469 | 0.3632 | 0.5314 | 0.9473 |
| all_features | rf_tuned | 1406 | 0.0626 | 0.8356 | 1.0090 | 0.3769 | 0.5265 | 0.9552 |
| baseline | avg_lstm_ridge | 1406 | 0.0604 | 0.8418 | 0.9892 | 0.3874 | 0.5170 | 0.9955 |
| baseline | lstm | 1406 | 0.0655 | 0.8273 | 0.9721 | 0.3873 | 0.5110 | 0.9817 |
| baseline | lstm_qsel | 1406 | 0.0655 | 0.8276 | 0.9613 | 0.3918 | 0.5061 | 0.9809 |
| baseline | ols | 1406 | 0.0581 | 0.8487 | 0.9752 | 0.3961 | 0.5134 | 0.9988 |
| baseline | lasso | 1406 | 0.0588 | 0.8463 | 0.9879 | 0.3899 | 0.5169 | 0.9986 |
| baseline | enet | 1406 | 0.0582 | 0.8482 | 0.9837 | 0.3924 | 0.5163 | 0.9993 |
| baseline | ridge | 1406 | 0.0580 | 0.8486 | 0.9880 | 0.3909 | 0.5182 | 1.0000 |
| baseline | lgbm | 1406 | 0.0662 | 0.8260 | 0.9488 | 0.3962 | 0.5100 | 0.9719 |
| baseline | xgb | 1406 | 0.0620 | 0.8372 | 1.0152 | 0.3753 | 0.5098 | 0.9805 |
| baseline | rf | 1406 | 0.0658 | 0.8284 | 0.9354 | 0.4031 | 0.4989 | 0.9730 |
| baseline | lgbm_tuned | 1406 | 0.0689 | 0.8171 | 1.0359 | 0.3590 | 0.5127 | 0.9709 |
| baseline | xgb_tuned | 1406 | 0.0706 | 0.8124 | 1.0440 | 0.3542 | 0.5098 | 0.9683 |
| baseline | rf_tuned | 1406 | 0.0688 | 0.8175 | 1.0239 | 0.3634 | 0.5029 | 0.9731 |
| live_feasible | avg_lstm_ridge | 1406 | 0.0602 | 0.8424 | 1.0255 | 0.3739 | 0.5327 | 0.9810 |
| live_feasible | lstm | 1406 | 0.0742 | 0.8011 | 0.9847 | 0.3703 | 0.5090 | 0.9199 |
| live_feasible | lstm_qsel | 1406 | 0.0729 | 0.8054 | 0.9647 | 0.3800 | 0.5173 | 0.9185 |
| live_feasible | lasso | 1406 | 0.0570 | 0.8516 | 0.9835 | 0.3941 | 0.5230 | 0.9858 |
| live_feasible | enet | 1406 | 0.0564 | 0.8537 | 0.9707 | 0.4003 | 0.5257 | 0.9908 |
| live_feasible | ridge | 1406 | 0.0581 | 0.8483 | 0.9826 | 0.3929 | 0.5341 | 1.0000 |
| live_feasible | lgbm | 1406 | 0.0614 | 0.8412 | 0.9374 | 0.4084 | 0.5282 | 0.9656 |
| live_feasible | xgb | 1406 | 0.0589 | 0.8461 | 0.9973 | 0.3861 | 0.5305 | 0.9699 |
| live_feasible | rf | 1406 | 0.0611 | 0.8415 | 0.9477 | 0.4041 | 0.5236 | 0.9622 |
| live_feasible | lgbm_tuned | 1406 | 0.0628 | 0.8365 | 1.0700 | 0.3558 | 0.5231 | 0.9561 |
| live_feasible | xgb_tuned | 1406 | 0.0600 | 0.8429 | 1.0153 | 0.3779 | 0.5330 | 0.9659 |
| live_feasible | rf_tuned | 1406 | 0.0597 | 0.8438 | 1.0145 | 0.3786 | 0.5316 | 0.9719 |

## Paired QLIKE, LSTM minus comparator (clock, common stamps; day-block 95 % interval)

| bucket | lstm | comparator | n | QLIKE_lstm | QLIKE_comparator | pct | ci_lo | ci_hi | dm_stat |
|---|---|---|---|---|---|---|---|---|---|
| all_features | lstm | avg_lstm_ridge | 1406 | 0.17028 | 0.12616 | 34.97424 | 0.03206 | 0.05776 | 8.24953 |
| all_features | lstm | lstm_qsel | 1406 | 0.17028 | 0.16918 | 0.65277 | -0.00379 | 0.00697 | 0.50166 |
| all_features | lstm | lasso | 1406 | 0.17028 | 0.11728 | 45.19491 | 0.03776 | 0.07130 | 7.13077 |
| all_features | lstm | enet | 1406 | 0.17028 | 0.11470 | 48.45280 | 0.04041 | 0.07347 | 7.45980 |
| all_features | lstm | ridge | 1406 | 0.17028 | 0.11907 | 43.00671 | 0.03537 | 0.06979 | 6.62605 |
| all_features | lstm | lgbm | 1406 | 0.17028 | 0.11947 | 42.53312 | 0.03527 | 0.06908 | 6.88273 |
| all_features | lstm | xgb | 1406 | 0.17028 | 0.12147 | 40.18026 | 0.03355 | 0.06637 | 6.58120 |
| all_features | lstm | rf | 1406 | 0.17028 | 0.12411 | 37.20084 | 0.03069 | 0.06410 | 6.00332 |
| all_features | lstm | lgbm_tuned | 1406 | 0.17028 | 0.12217 | 39.38507 | 0.03320 | 0.06639 | 6.59912 |
| all_features | lstm | xgb_tuned | 1406 | 0.17028 | 0.12502 | 36.20022 | 0.02949 | 0.06390 | 6.17269 |
| all_features | lstm | rf_tuned | 1406 | 0.17028 | 0.12478 | 36.47001 | 0.03041 | 0.06326 | 5.99328 |
| all_features | lstm_qsel | lasso | 1406 | 0.16918 | 0.11728 | 44.25327 | 0.03722 | 0.06956 | 7.18267 |
| all_features | lstm_qsel | enet | 1406 | 0.16918 | 0.11470 | 47.49003 | 0.03998 | 0.07150 | 7.60758 |
| all_features | lstm_qsel | ridge | 1406 | 0.16918 | 0.11907 | 42.07926 | 0.03506 | 0.06780 | 6.75267 |
| all_features | lstm_qsel | lgbm | 1406 | 0.16918 | 0.11947 | 41.60874 | 0.03474 | 0.06754 | 6.82073 |
| all_features | lstm_qsel | xgb | 1406 | 0.16918 | 0.12147 | 39.27114 | 0.03275 | 0.06515 | 6.50516 |
| all_features | lstm_qsel | rf | 1406 | 0.16918 | 0.12411 | 36.31105 | 0.03046 | 0.06274 | 6.01638 |
| all_features | lstm_qsel | lgbm_tuned | 1406 | 0.16918 | 0.12217 | 38.48111 | 0.03127 | 0.06547 | 6.44664 |
| all_features | lstm_qsel | xgb_tuned | 1406 | 0.16918 | 0.12502 | 35.31691 | 0.02850 | 0.06213 | 6.09578 |
| all_features | lstm_qsel | rf_tuned | 1406 | 0.16918 | 0.12478 | 35.58495 | 0.02999 | 0.06180 | 6.00212 |
| all_features | avg_lstm_ridge | lasso | 1406 | 0.12616 | 0.11728 | 7.57231 | 0.00339 | 0.01486 | 3.03869 |
| all_features | avg_lstm_ridge | enet | 1406 | 0.12616 | 0.11470 | 9.98602 | 0.00591 | 0.01731 | 3.91010 |
| all_features | avg_lstm_ridge | ridge | 1406 | 0.12616 | 0.11907 | 5.95112 | -0.00070 | 0.01392 | 2.00446 |
| all_features | avg_lstm_ridge | lgbm | 1406 | 0.12616 | 0.11947 | 5.60024 | -0.00155 | 0.01497 | 1.79808 |
| all_features | avg_lstm_ridge | xgb | 1406 | 0.12616 | 0.12147 | 3.85705 | -0.00421 | 0.01312 | 1.15346 |
| all_features | avg_lstm_ridge | rf | 1406 | 0.12616 | 0.12411 | 1.64965 | -0.00697 | 0.01066 | 0.48405 |
| all_features | avg_lstm_ridge | lgbm_tuned | 1406 | 0.12616 | 0.12217 | 3.26791 | -0.00437 | 0.01264 | 0.99443 |
| all_features | avg_lstm_ridge | xgb_tuned | 1406 | 0.12616 | 0.12502 | 0.90831 | -0.00906 | 0.01071 | 0.26507 |
| all_features | avg_lstm_ridge | rf_tuned | 1406 | 0.12616 | 0.12478 | 1.10819 | -0.00672 | 0.00977 | 0.34593 |
| baseline | lstm | avg_lstm_ridge | 1406 | 0.12703 | 0.11941 | 6.38011 | 0.00500 | 0.01065 | 5.95679 |
| baseline | lstm | lstm_qsel | 1406 | 0.12703 | 0.12560 | 1.13569 | -0.00089 | 0.00436 | 1.20806 |
| baseline | lstm | ols | 1406 | 0.12703 | 0.11670 | 8.84808 | 0.00546 | 0.01578 | 4.24414 |
| baseline | lstm | lasso | 1406 | 0.12703 | 0.11743 | 8.17116 | 0.00454 | 0.01527 | 4.00345 |
| baseline | lstm | enet | 1406 | 0.12703 | 0.11646 | 9.07627 | 0.00580 | 0.01606 | 4.45114 |
| baseline | lstm | ridge | 1406 | 0.12703 | 0.11664 | 8.90148 | 0.00554 | 0.01585 | 4.32161 |
| baseline | lstm | lgbm | 1406 | 0.12703 | 0.12504 | 1.58931 | -0.00249 | 0.00683 | 0.76055 |
| baseline | lstm | xgb | 1406 | 0.12703 | 0.12391 | 2.51467 | -0.00088 | 0.00735 | 1.42845 |
| baseline | lstm | rf | 1406 | 0.12703 | 0.12455 | 1.98375 | -0.00286 | 0.00825 | 0.86064 |
| baseline | lstm | lgbm_tuned | 1406 | 0.12703 | 0.13054 | -2.68986 | -0.01071 | 0.00261 | -0.98974 |
| baseline | lstm | xgb_tuned | 1406 | 0.12703 | 0.13055 | -2.70210 | -0.01341 | 0.00433 | -0.82216 |
| baseline | lstm | rf_tuned | 1406 | 0.12703 | 0.13094 | -2.98772 | -0.01202 | 0.00274 | -1.09592 |
| baseline | lstm_qsel | ols | 1406 | 0.12560 | 0.11670 | 7.62579 | 0.00462 | 0.01336 | 4.19742 |
| baseline | lstm_qsel | lasso | 1406 | 0.12560 | 0.11743 | 6.95646 | 0.00380 | 0.01294 | 3.95916 |
| baseline | lstm_qsel | enet | 1406 | 0.12560 | 0.11646 | 7.85141 | 0.00492 | 0.01380 | 4.46714 |
| baseline | lstm_qsel | ridge | 1406 | 0.12560 | 0.11664 | 7.67858 | 0.00478 | 0.01334 | 4.32506 |
| baseline | lstm_qsel | lgbm | 1406 | 0.12560 | 0.12504 | 0.44852 | -0.00427 | 0.00533 | 0.21623 |
| baseline | lstm_qsel | xgb | 1406 | 0.12560 | 0.12391 | 1.36349 | -0.00262 | 0.00604 | 0.77070 |
| baseline | lstm_qsel | rf | 1406 | 0.12560 | 0.12455 | 0.83854 | -0.00441 | 0.00686 | 0.37032 |
| baseline | lstm_qsel | lgbm_tuned | 1406 | 0.12560 | 0.13054 | -3.78260 | -0.01266 | 0.00127 | -1.39076 |
| baseline | lstm_qsel | xgb_tuned | 1406 | 0.12560 | 0.13055 | -3.79470 | -0.01539 | 0.00315 | -1.13977 |
| baseline | lstm_qsel | rf_tuned | 1406 | 0.12560 | 0.13094 | -4.07711 | -0.01343 | 0.00129 | -1.50954 |
| baseline | avg_lstm_ridge | ols | 1406 | 0.11941 | 0.11670 | 2.31995 | 0.00018 | 0.00546 | 2.16754 |
| baseline | avg_lstm_ridge | lasso | 1406 | 0.11941 | 0.11743 | 1.68363 | -0.00079 | 0.00500 | 1.56236 |
| baseline | avg_lstm_ridge | enet | 1406 | 0.11941 | 0.11646 | 2.53445 | 0.00043 | 0.00577 | 2.49025 |
| baseline | avg_lstm_ridge | ridge | 1406 | 0.11941 | 0.11664 | 2.37015 | 0.00048 | 0.00532 | 2.39468 |
| baseline | avg_lstm_ridge | lgbm | 1406 | 0.11941 | 0.12504 | -4.50348 | -0.00963 | -0.00165 | -2.51398 |
| baseline | avg_lstm_ridge | xgb | 1406 | 0.11941 | 0.12391 | -3.63362 | -0.00777 | -0.00127 | -2.60578 |
| baseline | avg_lstm_ridge | rf | 1406 | 0.11941 | 0.12455 | -4.13269 | -0.00977 | -0.00047 | -2.15895 |
| baseline | avg_lstm_ridge | lgbm_tuned | 1406 | 0.11941 | 0.13054 | -8.52601 | -0.01843 | -0.00501 | -3.26741 |
| baseline | avg_lstm_ridge | xgb_tuned | 1406 | 0.11941 | 0.13055 | -8.53752 | -0.02103 | -0.00331 | -2.62052 |
| baseline | avg_lstm_ridge | rf_tuned | 1406 | 0.11941 | 0.13094 | -8.80600 | -0.01917 | -0.00590 | -3.48418 |
| live_feasible | lstm | avg_lstm_ridge | 1406 | 0.14676 | 0.12116 | 21.12474 | 0.01856 | 0.03382 | 7.53074 |
| live_feasible | lstm | lstm_qsel | 1406 | 0.14676 | 0.14266 | 2.87039 | -0.00002 | 0.00868 | 1.76412 |
| live_feasible | lstm | lasso | 1406 | 0.14676 | 0.11578 | 26.75274 | 0.01963 | 0.04351 | 5.37215 |
| live_feasible | lstm | enet | 1406 | 0.14676 | 0.11346 | 29.34524 | 0.02194 | 0.04576 | 5.74622 |
| live_feasible | lstm | ridge | 1406 | 0.14676 | 0.11919 | 23.12288 | 0.01491 | 0.04070 | 4.55585 |
| live_feasible | lstm | lgbm | 1406 | 0.14676 | 0.11806 | 24.31016 | 0.01613 | 0.04236 | 4.49985 |
| live_feasible | lstm | xgb | 1406 | 0.14676 | 0.12020 | 22.09713 | 0.01431 | 0.04014 | 4.19241 |
| live_feasible | lstm | rf | 1406 | 0.14676 | 0.11893 | 23.39224 | 0.01508 | 0.04153 | 4.21318 |
| live_feasible | lstm | lgbm_tuned | 1406 | 0.14676 | 0.12632 | 16.17478 | 0.00698 | 0.03449 | 3.05023 |
| live_feasible | lstm | xgb_tuned | 1406 | 0.14676 | 0.12434 | 18.02841 | 0.00891 | 0.03661 | 3.41403 |
| live_feasible | lstm | rf_tuned | 1406 | 0.14676 | 0.12023 | 22.06418 | 0.01380 | 0.04039 | 3.98364 |
| live_feasible | lstm_qsel | lasso | 1406 | 0.14266 | 0.11578 | 23.21596 | 0.01559 | 0.03971 | 4.99803 |
| live_feasible | lstm_qsel | enet | 1406 | 0.14266 | 0.11346 | 25.73612 | 0.01792 | 0.04172 | 5.48141 |
| live_feasible | lstm_qsel | ridge | 1406 | 0.14266 | 0.11919 | 19.68739 | 0.01149 | 0.03671 | 4.21398 |
| live_feasible | lstm_qsel | lgbm | 1406 | 0.14266 | 0.11806 | 20.84154 | 0.01288 | 0.03783 | 4.23727 |
| live_feasible | lstm_qsel | xgb | 1406 | 0.14266 | 0.12020 | 18.69025 | 0.01070 | 0.03545 | 3.93308 |
| live_feasible | lstm_qsel | rf | 1406 | 0.14266 | 0.11893 | 19.94923 | 0.01149 | 0.03696 | 3.99108 |
| live_feasible | lstm_qsel | lgbm_tuned | 1406 | 0.14266 | 0.12632 | 12.93316 | 0.00379 | 0.03014 | 2.70953 |
| live_feasible | lstm_qsel | xgb_tuned | 1406 | 0.14266 | 0.12434 | 14.73506 | 0.00537 | 0.03204 | 3.10198 |
| live_feasible | lstm_qsel | rf_tuned | 1406 | 0.14266 | 0.12023 | 18.65823 | 0.01008 | 0.03561 | 3.70324 |
| live_feasible | avg_lstm_ridge | lasso | 1406 | 0.12116 | 0.11578 | 4.64645 | -0.00059 | 0.01183 | 1.73039 |
| live_feasible | avg_lstm_ridge | enet | 1406 | 0.12116 | 0.11346 | 6.78680 | 0.00207 | 0.01432 | 2.51152 |
| live_feasible | avg_lstm_ridge | ridge | 1406 | 0.12116 | 0.11919 | 1.64966 | -0.00730 | 0.00876 | 0.53838 |
| live_feasible | avg_lstm_ridge | lgbm | 1406 | 0.12116 | 0.11806 | 2.62987 | -0.00482 | 0.01206 | 0.73290 |
| live_feasible | avg_lstm_ridge | xgb | 1406 | 0.12116 | 0.12020 | 0.80280 | -0.00726 | 0.01025 | 0.22599 |
| live_feasible | avg_lstm_ridge | rf | 1406 | 0.12116 | 0.11893 | 1.87204 | -0.00641 | 0.01166 | 0.50733 |
| live_feasible | avg_lstm_ridge | lgbm_tuned | 1406 | 0.12116 | 0.12632 | -4.08666 | -0.01468 | 0.00483 | -1.05742 |
| live_feasible | avg_lstm_ridge | xgb_tuned | 1406 | 0.12116 | 0.12434 | -2.55632 | -0.01296 | 0.00665 | -0.68303 |
| live_feasible | avg_lstm_ridge | rf_tuned | 1406 | 0.12116 | 0.12023 | 0.77560 | -0.00716 | 0.01033 | 0.21232 |

## 15:30 sign(s) straddle trade, 866 deck days (95 % day-block intervals)

| bucket | forecast | deck_days | pct_buy | Sharpe_mid | Sharpe_mid_lo | Sharpe_mid_hi | Sharpe_crossed | Sharpe_crossed_lo | Sharpe_crossed_hi |
|---|---|---|---|---|---|---|---|---|---|
| all_features | avg_lstm_ridge | 866 | 36.605 | 1.106 | 0.290 | 2.020 | 0.626 | -0.184 | 1.533 |
| all_features | lstm | 866 | 41.801 | 0.531 | -0.407 | 1.525 | 0.059 | -0.880 | 1.046 |
| all_features | lstm_qsel | 866 | 41.917 | 0.373 | -0.632 | 1.389 | -0.098 | -1.101 | 0.930 |
| all_features | lasso | 866 | 39.607 | 1.446 | 0.523 | 2.319 | 0.981 | 0.023 | 1.883 |
| all_features | enet | 866 | 39.145 | 1.329 | 0.343 | 2.214 | 0.863 | -0.153 | 1.779 |
| all_features | ridge | 866 | 36.143 | 1.665 | 0.754 | 2.549 | 1.198 | 0.271 | 2.107 |
| all_features | lgbm | 866 | 38.222 | 1.346 | 0.416 | 2.316 | 0.873 | -0.068 | 1.853 |
| all_features | xgb | 866 | 39.261 | 1.258 | 0.368 | 2.136 | 0.791 | -0.124 | 1.681 |
| all_features | rf | 866 | 41.801 | 1.326 | 0.474 | 2.137 | 0.860 | -0.011 | 1.682 |
| all_features | lgbm_tuned | 866 | 39.376 | 1.813 | 1.007 | 2.615 | 1.349 | 0.523 | 2.172 |
| all_features | xgb_tuned | 866 | 38.106 | 1.453 | 0.602 | 2.269 | 0.987 | 0.121 | 1.819 |
| all_features | rf_tuned | 866 | 42.148 | 1.408 | 0.542 | 2.255 | 0.941 | 0.056 | 1.803 |
| baseline | avg_lstm_ridge | 866 | 37.875 | 0.824 | -0.021 | 1.661 | 0.352 | -0.511 | 1.206 |
| baseline | lstm | 866 | 39.607 | 0.613 | -0.375 | 1.535 | 0.141 | -0.863 | 1.090 |
| baseline | lstm_qsel | 866 | 39.030 | 0.926 | -0.020 | 1.805 | 0.454 | -0.537 | 1.354 |
| baseline | ols | 866 | 36.836 | 1.326 | 0.517 | 2.055 | 0.853 | 0.008 | 1.611 |
| baseline | lasso | 866 | 37.991 | 1.507 | 0.718 | 2.231 | 1.035 | 0.210 | 1.802 |
| baseline | enet | 866 | 37.644 | 1.290 | 0.486 | 2.021 | 0.817 | -0.028 | 1.577 |
| baseline | ridge | 866 | 37.875 | 1.217 | 0.408 | 1.954 | 0.744 | -0.094 | 1.495 |
| baseline | lgbm | 866 | 41.917 | 0.892 | -0.016 | 1.708 | 0.423 | -0.501 | 1.269 |
| baseline | xgb | 866 | 39.607 | 1.122 | 0.302 | 1.904 | 0.651 | -0.190 | 1.448 |
| baseline | rf | 866 | 41.686 | 1.176 | 0.222 | 2.135 | 0.698 | -0.250 | 1.686 |
| baseline | lgbm_tuned | 866 | 41.801 | 1.012 | 0.173 | 1.817 | 0.544 | -0.334 | 1.365 |
| baseline | xgb_tuned | 866 | 42.032 | 0.749 | -0.105 | 1.564 | 0.278 | -0.595 | 1.121 |
| baseline | rf_tuned | 866 | 43.764 | 0.581 | -0.437 | 1.638 | 0.108 | -0.919 | 1.167 |
| live_feasible | avg_lstm_ridge | 866 | 39.954 | 1.189 | 0.233 | 2.171 | 0.718 | -0.265 | 1.713 |
| live_feasible | lstm | 866 | 40.185 | 1.013 | 0.108 | 1.998 | 0.538 | -0.385 | 1.539 |
| live_feasible | lstm_qsel | 866 | 40.416 | 0.883 | 0.020 | 1.767 | 0.411 | -0.465 | 1.319 |
| live_feasible | lasso | 866 | 38.915 | 1.826 | 0.885 | 2.674 | 1.363 | 0.382 | 2.242 |
| live_feasible | enet | 866 | 37.760 | 1.411 | 0.456 | 2.312 | 0.943 | -0.029 | 1.859 |
| live_feasible | ridge | 866 | 40.185 | 1.902 | 0.891 | 2.938 | 1.440 | 0.408 | 2.471 |
| live_feasible | lgbm | 866 | 39.376 | 1.692 | 0.770 | 2.596 | 1.225 | 0.286 | 2.159 |
| live_feasible | xgb | 866 | 38.799 | 1.696 | 0.855 | 2.561 | 1.229 | 0.357 | 2.115 |
| live_feasible | rf | 866 | 39.954 | 1.582 | 0.688 | 2.393 | 1.114 | 0.197 | 1.950 |
| live_feasible | lgbm_tuned | 866 | 40.300 | 1.142 | 0.166 | 2.065 | 0.677 | -0.323 | 1.618 |
| live_feasible | xgb_tuned | 866 | 39.145 | 1.458 | 0.652 | 2.251 | 0.993 | 0.174 | 1.801 |
| live_feasible | rf_tuned | 866 | 39.607 | 1.559 | 0.758 | 2.392 | 1.092 | 0.253 | 1.934 |

## Sharpe difference, LSTM minus comparator, same days

| bucket | lstm | comparator | days | same_position_pct | dSharpe_mid | dSharpe_mid_lo | dSharpe_mid_hi | dSharpe_crossed | dSharpe_crossed_lo | dSharpe_crossed_hi |
|---|---|---|---|---|---|---|---|---|---|---|
| all_features | lstm | avg_lstm_ridge | 866 | 87.875 | -0.575 | -1.301 | 0.060 | -0.567 | -1.288 | 0.060 |
| all_features | lstm | lstm_qsel | 866 | 94.573 | 0.158 | -0.302 | 0.713 | 0.156 | -0.298 | 0.706 |
| all_features | lstm | lasso | 866 | 74.480 | -0.915 | -2.089 | 0.279 | -0.922 | -2.098 | 0.275 |
| all_features | lstm | enet | 866 | 73.557 | -0.798 | -2.018 | 0.441 | -0.804 | -2.036 | 0.436 |
| all_features | lstm | ridge | 866 | 74.018 | -1.134 | -2.396 | 0.100 | -1.140 | -2.406 | 0.097 |
| all_features | lstm | lgbm | 866 | 74.942 | -0.815 | -1.971 | 0.386 | -0.815 | -1.971 | 0.379 |
| all_features | lstm | xgb | 866 | 75.520 | -0.727 | -1.947 | 0.494 | -0.732 | -1.962 | 0.490 |
| all_features | lstm | rf | 866 | 73.672 | -0.795 | -2.073 | 0.519 | -0.801 | -2.092 | 0.508 |
| all_features | lstm | lgbm_tuned | 866 | 76.559 | -1.282 | -2.478 | -0.092 | -1.290 | -2.489 | -0.096 |
| all_features | lstm | xgb_tuned | 866 | 76.443 | -0.922 | -2.132 | 0.236 | -0.928 | -2.127 | 0.234 |
| all_features | lstm | rf_tuned | 866 | 77.252 | -0.877 | -2.063 | 0.299 | -0.883 | -2.080 | 0.294 |
| all_features | lstm_qsel | avg_lstm_ridge | 866 | 86.143 | -0.733 | -1.573 | -0.038 | -0.723 | -1.560 | -0.036 |
| all_features | lstm_qsel | lstm | 866 | 94.573 | -0.158 | -0.713 | 0.302 | -0.156 | -0.706 | 0.298 |
| all_features | lstm_qsel | lasso | 866 | 74.827 | -1.073 | -2.268 | 0.147 | -1.078 | -2.259 | 0.142 |
| all_features | lstm_qsel | enet | 866 | 74.365 | -0.955 | -2.190 | 0.318 | -0.960 | -2.192 | 0.318 |
| all_features | lstm_qsel | ridge | 866 | 74.596 | -1.292 | -2.574 | 0.013 | -1.296 | -2.564 | 0.007 |
| all_features | lstm_qsel | lgbm | 866 | 75.289 | -0.973 | -2.118 | 0.159 | -0.971 | -2.117 | 0.157 |
| all_features | lstm_qsel | xgb | 866 | 75.635 | -0.885 | -2.065 | 0.352 | -0.888 | -2.077 | 0.352 |
| all_features | lstm_qsel | rf | 866 | 74.711 | -0.953 | -2.206 | 0.344 | -0.957 | -2.210 | 0.334 |
| all_features | lstm_qsel | lgbm_tuned | 866 | 76.905 | -1.440 | -2.580 | -0.300 | -1.446 | -2.584 | -0.307 |
| all_features | lstm_qsel | xgb_tuned | 866 | 76.328 | -1.080 | -2.210 | 0.067 | -1.084 | -2.218 | 0.071 |
| all_features | lstm_qsel | rf_tuned | 866 | 77.136 | -1.035 | -2.115 | 0.099 | -1.039 | -2.121 | 0.097 |
| all_features | avg_lstm_ridge | lstm | 866 | 87.875 | 0.575 | -0.060 | 1.301 | 0.567 | -0.060 | 1.288 |
| all_features | avg_lstm_ridge | lstm_qsel | 866 | 86.143 | 0.733 | 0.038 | 1.573 | 0.723 | 0.036 | 1.560 |
| all_features | avg_lstm_ridge | lasso | 866 | 83.834 | -0.340 | -1.329 | 0.700 | -0.355 | -1.360 | 0.699 |
| all_features | avg_lstm_ridge | enet | 866 | 84.065 | -0.223 | -1.266 | 0.841 | -0.237 | -1.298 | 0.835 |
| all_features | avg_lstm_ridge | ridge | 866 | 86.143 | -0.559 | -1.612 | 0.445 | -0.572 | -1.639 | 0.443 |
| all_features | avg_lstm_ridge | lgbm | 866 | 80.370 | -0.240 | -1.271 | 0.859 | -0.247 | -1.281 | 0.851 |
| all_features | avg_lstm_ridge | xgb | 866 | 81.409 | -0.152 | -1.152 | 0.935 | -0.165 | -1.170 | 0.928 |
| all_features | avg_lstm_ridge | rf | 866 | 80.023 | -0.220 | -1.343 | 0.952 | -0.234 | -1.366 | 0.938 |
| all_features | avg_lstm_ridge | lgbm_tuned | 866 | 82.217 | -0.707 | -1.655 | 0.375 | -0.723 | -1.677 | 0.370 |
| all_features | avg_lstm_ridge | xgb_tuned | 866 | 82.333 | -0.347 | -1.270 | 0.581 | -0.361 | -1.296 | 0.571 |
| all_features | avg_lstm_ridge | rf_tuned | 866 | 81.986 | -0.302 | -1.324 | 0.790 | -0.316 | -1.344 | 0.779 |
| baseline | lstm | avg_lstm_ridge | 866 | 94.111 | -0.211 | -0.680 | 0.210 | -0.211 | -0.678 | 0.211 |
| baseline | lstm | lstm_qsel | 866 | 94.342 | -0.313 | -0.799 | 0.121 | -0.313 | -0.796 | 0.121 |
| baseline | lstm | ols | 866 | 87.760 | -0.714 | -1.413 | -0.052 | -0.712 | -1.409 | -0.049 |
| baseline | lstm | lasso | 866 | 87.298 | -0.894 | -1.709 | -0.148 | -0.895 | -1.707 | -0.147 |
| baseline | lstm | enet | 866 | 87.413 | -0.677 | -1.396 | -0.011 | -0.677 | -1.394 | -0.010 |
| baseline | lstm | ridge | 866 | 87.875 | -0.604 | -1.306 | 0.063 | -0.603 | -1.300 | 0.064 |
| baseline | lstm | lgbm | 866 | 85.450 | -0.280 | -1.130 | 0.588 | -0.282 | -1.136 | 0.589 |
| baseline | lstm | xgb | 866 | 87.529 | -0.509 | -1.340 | 0.275 | -0.511 | -1.349 | 0.277 |
| baseline | lstm | rf | 866 | 84.065 | -0.563 | -1.536 | 0.449 | -0.557 | -1.530 | 0.456 |
| baseline | lstm | lgbm_tuned | 866 | 87.413 | -0.400 | -1.110 | 0.266 | -0.404 | -1.116 | 0.268 |
| baseline | lstm | xgb_tuned | 866 | 86.490 | -0.136 | -0.933 | 0.652 | -0.137 | -0.943 | 0.652 |
| baseline | lstm | rf_tuned | 866 | 85.219 | 0.032 | -0.891 | 0.978 | 0.033 | -0.890 | 0.981 |
| baseline | lstm_qsel | avg_lstm_ridge | 866 | 93.995 | 0.102 | -0.346 | 0.542 | 0.102 | -0.346 | 0.542 |
| baseline | lstm_qsel | lstm | 866 | 94.342 | 0.313 | -0.121 | 0.799 | 0.313 | -0.121 | 0.796 |
| baseline | lstm_qsel | ols | 866 | 89.954 | -0.401 | -1.010 | 0.193 | -0.399 | -1.008 | 0.191 |
| baseline | lstm_qsel | lasso | 866 | 89.492 | -0.581 | -1.334 | 0.139 | -0.581 | -1.335 | 0.142 |
| baseline | lstm_qsel | enet | 866 | 89.376 | -0.364 | -0.994 | 0.264 | -0.363 | -0.997 | 0.263 |
| baseline | lstm_qsel | ridge | 866 | 89.607 | -0.291 | -0.891 | 0.330 | -0.290 | -0.891 | 0.328 |
| baseline | lstm_qsel | lgbm | 866 | 86.259 | 0.033 | -0.775 | 0.834 | 0.031 | -0.778 | 0.835 |
| baseline | lstm_qsel | xgb | 866 | 88.337 | -0.196 | -0.946 | 0.460 | -0.197 | -0.950 | 0.460 |
| baseline | lstm_qsel | rf | 866 | 84.180 | -0.250 | -1.229 | 0.809 | -0.244 | -1.228 | 0.817 |
| baseline | lstm_qsel | lgbm_tuned | 866 | 87.991 | -0.087 | -0.767 | 0.553 | -0.091 | -0.771 | 0.551 |
| baseline | lstm_qsel | xgb_tuned | 866 | 87.298 | 0.177 | -0.595 | 0.911 | 0.176 | -0.606 | 0.912 |
| baseline | lstm_qsel | rf_tuned | 866 | 85.104 | 0.345 | -0.618 | 1.353 | 0.346 | -0.623 | 1.347 |
| baseline | avg_lstm_ridge | lstm | 866 | 94.111 | 0.211 | -0.210 | 0.680 | 0.211 | -0.211 | 0.678 |
| baseline | avg_lstm_ridge | lstm_qsel | 866 | 93.995 | -0.102 | -0.542 | 0.346 | -0.102 | -0.542 | 0.346 |
| baseline | avg_lstm_ridge | ols | 866 | 93.649 | -0.502 | -0.984 | -0.005 | -0.501 | -0.984 | -0.002 |
| baseline | avg_lstm_ridge | lasso | 866 | 92.956 | -0.683 | -1.364 | -0.081 | -0.684 | -1.371 | -0.081 |
| baseline | avg_lstm_ridge | enet | 866 | 93.303 | -0.466 | -1.009 | 0.079 | -0.466 | -1.010 | 0.080 |
| baseline | avg_lstm_ridge | ridge | 866 | 93.764 | -0.392 | -0.920 | 0.148 | -0.392 | -0.919 | 0.150 |
| baseline | avg_lstm_ridge | lgbm | 866 | 88.106 | -0.068 | -0.857 | 0.740 | -0.071 | -0.860 | 0.740 |
| baseline | avg_lstm_ridge | xgb | 866 | 90.416 | -0.298 | -1.014 | 0.342 | -0.300 | -1.017 | 0.343 |
| baseline | avg_lstm_ridge | rf | 866 | 86.259 | -0.351 | -1.314 | 0.654 | -0.346 | -1.313 | 0.658 |
| baseline | avg_lstm_ridge | lgbm_tuned | 866 | 89.838 | -0.188 | -0.827 | 0.437 | -0.193 | -0.839 | 0.437 |
| baseline | avg_lstm_ridge | xgb_tuned | 866 | 88.453 | 0.075 | -0.636 | 0.778 | 0.074 | -0.642 | 0.775 |
| baseline | avg_lstm_ridge | rf_tuned | 866 | 86.490 | 0.243 | -0.694 | 1.242 | 0.244 | -0.694 | 1.252 |
| live_feasible | lstm | avg_lstm_ridge | 866 | 90.762 | -0.176 | -0.810 | 0.389 | -0.180 | -0.812 | 0.383 |
| live_feasible | lstm | lstm_qsel | 866 | 92.610 | 0.130 | -0.365 | 0.606 | 0.127 | -0.368 | 0.593 |
| live_feasible | lstm | lasso | 866 | 78.176 | -0.814 | -1.762 | 0.248 | -0.825 | -1.767 | 0.239 |
| live_feasible | lstm | enet | 866 | 77.021 | -0.398 | -1.280 | 0.514 | -0.405 | -1.285 | 0.507 |
| live_feasible | lstm | ridge | 866 | 76.674 | -0.890 | -1.922 | 0.248 | -0.902 | -1.948 | 0.238 |
| live_feasible | lstm | lgbm | 866 | 78.637 | -0.679 | -1.639 | 0.296 | -0.687 | -1.647 | 0.289 |
| live_feasible | lstm | xgb | 866 | 78.984 | -0.684 | -1.533 | 0.178 | -0.691 | -1.542 | 0.176 |
| live_feasible | lstm | rf | 866 | 78.984 | -0.569 | -1.425 | 0.367 | -0.576 | -1.438 | 0.363 |
| live_feasible | lstm | lgbm_tuned | 866 | 79.792 | -0.129 | -1.033 | 0.780 | -0.139 | -1.052 | 0.781 |
| live_feasible | lstm | xgb_tuned | 866 | 77.021 | -0.445 | -1.314 | 0.483 | -0.456 | -1.322 | 0.473 |
| live_feasible | lstm | rf_tuned | 866 | 77.714 | -0.546 | -1.382 | 0.343 | -0.554 | -1.388 | 0.340 |
| live_feasible | lstm_qsel | avg_lstm_ridge | 866 | 89.607 | -0.305 | -0.830 | 0.181 | -0.307 | -0.837 | 0.181 |
| live_feasible | lstm_qsel | lstm | 866 | 92.610 | -0.130 | -0.606 | 0.365 | -0.127 | -0.593 | 0.368 |
| live_feasible | lstm_qsel | lasso | 866 | 77.714 | -0.943 | -1.917 | 0.019 | -0.952 | -1.919 | 0.017 |
| live_feasible | lstm_qsel | enet | 866 | 77.021 | -0.528 | -1.410 | 0.326 | -0.532 | -1.417 | 0.327 |
| live_feasible | lstm_qsel | ridge | 866 | 77.598 | -1.019 | -2.028 | -0.007 | -1.029 | -2.047 | -0.008 |
| live_feasible | lstm_qsel | lgbm | 866 | 79.099 | -0.809 | -1.714 | 0.077 | -0.814 | -1.717 | 0.076 |
| live_feasible | lstm_qsel | xgb | 866 | 79.677 | -0.813 | -1.604 | -0.039 | -0.818 | -1.614 | -0.037 |
| live_feasible | lstm_qsel | rf | 866 | 78.984 | -0.699 | -1.493 | 0.199 | -0.703 | -1.503 | 0.196 |
| live_feasible | lstm_qsel | lgbm_tuned | 866 | 80.023 | -0.259 | -1.126 | 0.655 | -0.266 | -1.139 | 0.652 |
| live_feasible | lstm_qsel | xgb_tuned | 866 | 78.176 | -0.575 | -1.378 | 0.252 | -0.582 | -1.388 | 0.252 |
| live_feasible | lstm_qsel | rf_tuned | 866 | 78.868 | -0.676 | -1.484 | 0.123 | -0.681 | -1.486 | 0.122 |
| live_feasible | avg_lstm_ridge | lstm | 866 | 90.762 | 0.176 | -0.389 | 0.810 | 0.180 | -0.383 | 0.812 |
| live_feasible | avg_lstm_ridge | lstm_qsel | 866 | 89.607 | 0.305 | -0.181 | 0.830 | 0.307 | -0.181 | 0.837 |
| live_feasible | avg_lstm_ridge | lasso | 866 | 83.949 | -0.638 | -1.584 | 0.316 | -0.645 | -1.602 | 0.312 |
| live_feasible | avg_lstm_ridge | enet | 866 | 84.411 | -0.222 | -1.025 | 0.598 | -0.225 | -1.027 | 0.599 |
| live_feasible | avg_lstm_ridge | ridge | 866 | 85.912 | -0.714 | -1.601 | 0.167 | -0.721 | -1.611 | 0.166 |
| live_feasible | avg_lstm_ridge | lgbm | 866 | 84.642 | -0.503 | -1.328 | 0.362 | -0.507 | -1.334 | 0.366 |
| live_feasible | avg_lstm_ridge | xgb | 866 | 84.296 | -0.508 | -1.177 | 0.232 | -0.511 | -1.181 | 0.231 |
| live_feasible | avg_lstm_ridge | rf | 866 | 83.372 | -0.393 | -1.223 | 0.486 | -0.396 | -1.233 | 0.492 |
| live_feasible | avg_lstm_ridge | lgbm_tuned | 866 | 84.873 | 0.047 | -0.739 | 0.883 | 0.042 | -0.748 | 0.889 |
| live_feasible | avg_lstm_ridge | xgb_tuned | 866 | 82.102 | -0.269 | -1.047 | 0.528 | -0.275 | -1.049 | 0.530 |
| live_feasible | avg_lstm_ridge | rf_tuned | 866 | 82.794 | -0.370 | -1.096 | 0.394 | -0.374 | -1.097 | 0.396 |

## Chosen hyperparameters: share of tuning points

| bucket | rule | axis | value | tuning_points | share |
|---|---|---|---|---|---|
| all_features | mse | dropout | 0.000 | 1 | 0.167 |
| all_features | mse | dropout | 0.200 | 5 | 0.833 |
| all_features | mse | hidden | 16.000 | 2 | 0.333 |
| all_features | mse | hidden | 64.000 | 4 | 0.667 |
| all_features | mse | lr | 0.010 | 6 | 1.000 |
| all_features | mse | seq_len | 5.000 | 3 | 0.500 |
| all_features | mse | seq_len | 20.000 | 3 | 0.500 |
| all_features | qlike | dropout | 0.000 | 2 | 0.333 |
| all_features | qlike | dropout | 0.200 | 4 | 0.667 |
| all_features | qlike | hidden | 16.000 | 4 | 0.667 |
| all_features | qlike | hidden | 64.000 | 2 | 0.333 |
| all_features | qlike | lr | 0.010 | 6 | 1.000 |
| all_features | qlike | seq_len | 5.000 | 3 | 0.500 |
| all_features | qlike | seq_len | 20.000 | 3 | 0.500 |
| baseline | mse | dropout | 0.000 | 2 | 0.333 |
| baseline | mse | dropout | 0.200 | 4 | 0.667 |
| baseline | mse | hidden | 16.000 | 3 | 0.500 |
| baseline | mse | hidden | 64.000 | 3 | 0.500 |
| baseline | mse | lr | 0.001 | 2 | 0.333 |
| baseline | mse | lr | 0.010 | 4 | 0.667 |
| baseline | mse | seq_len | 5.000 | 3 | 0.500 |
| baseline | mse | seq_len | 20.000 | 3 | 0.500 |
| baseline | qlike | dropout | 0.000 | 3 | 0.500 |
| baseline | qlike | dropout | 0.200 | 3 | 0.500 |
| baseline | qlike | hidden | 16.000 | 1 | 0.167 |
| baseline | qlike | hidden | 64.000 | 5 | 0.833 |
| baseline | qlike | lr | 0.001 | 4 | 0.667 |
| baseline | qlike | lr | 0.010 | 2 | 0.333 |
| baseline | qlike | seq_len | 5.000 | 2 | 0.333 |
| baseline | qlike | seq_len | 20.000 | 4 | 0.667 |
| live_feasible | mse | dropout | 0.000 | 2 | 0.333 |
| live_feasible | mse | dropout | 0.200 | 4 | 0.667 |
| live_feasible | mse | hidden | 16.000 | 3 | 0.500 |
| live_feasible | mse | hidden | 64.000 | 3 | 0.500 |
| live_feasible | mse | lr | 0.001 | 1 | 0.167 |
| live_feasible | mse | lr | 0.010 | 5 | 0.833 |
| live_feasible | mse | seq_len | 5.000 | 4 | 0.667 |
| live_feasible | mse | seq_len | 20.000 | 2 | 0.333 |
| live_feasible | qlike | dropout | 0.000 | 2 | 0.333 |
| live_feasible | qlike | dropout | 0.200 | 4 | 0.667 |
| live_feasible | qlike | hidden | 16.000 | 1 | 0.167 |
| live_feasible | qlike | hidden | 64.000 | 5 | 0.833 |
| live_feasible | qlike | lr | 0.001 | 2 | 0.333 |
| live_feasible | qlike | lr | 0.010 | 4 | 0.667 |
| live_feasible | qlike | seq_len | 5.000 | 2 | 0.333 |
| live_feasible | qlike | seq_len | 20.000 | 4 | 0.667 |

## Single networks vs the seed average

| bucket | network | n | QLIKE_clock | deck_days | Sharpe_mid | Sharpe_crossed |
|---|---|---|---|---|---|---|
| all_features | seed 42 | 1406 | 0.2064 | 866 | 0.0129 | -0.4549 |
| all_features | seed 43 | 1406 | 0.1847 | 866 | 1.1085 | 0.6266 |
| all_features | seed 44 | 1406 | 0.1941 | 866 | 0.7755 | 0.3024 |
| all_features | seed 45 | 1406 | 0.2184 | 866 | 0.3104 | -0.1596 |
| all_features | seed 46 | 1406 | 0.2132 | 866 | 0.2203 | -0.2486 |
| all_features | seed average (of record) | 1406 | 0.1703 | 866 | 0.5310 | 0.0586 |
| baseline | seed 42 | 1406 | 0.1349 | 866 | 0.6552 | 0.1834 |
| baseline | seed 43 | 1406 | 0.1307 | 866 | 0.5129 | 0.0398 |
| baseline | seed 44 | 1406 | 0.1316 | 866 | 0.8561 | 0.3848 |
| baseline | seed 45 | 1406 | 0.1352 | 866 | 1.1901 | 0.7189 |
| baseline | seed 46 | 1406 | 0.1302 | 866 | 0.5336 | 0.0606 |
| baseline | seed average (of record) | 1406 | 0.1270 | 866 | 0.6127 | 0.1406 |
| live_feasible | seed 42 | 1406 | 0.1677 | 866 | 0.8881 | 0.4166 |
| live_feasible | seed 43 | 1406 | 0.1573 | 866 | 1.1207 | 0.6451 |
| live_feasible | seed 44 | 1406 | 0.1701 | 866 | 0.7894 | 0.3168 |
| live_feasible | seed 45 | 1406 | 0.1658 | 866 | 0.8864 | 0.4138 |
| live_feasible | seed 46 | 1406 | 0.1627 | 866 | 1.3190 | 0.8485 |
| live_feasible | seed average (of record) | 1406 | 0.1468 | 866 | 1.0127 | 0.5379 |

## Gates: 133 checked, 0 failed

