# A recalibration grid for the 0DTE 15:30 straddle portfolio

Everything below is printed by `writeup/recalibration_grid/recal_grid.py`. The
tables and the figure are written to `results/atm_straddle_0dte_1530/recal_grid/`.
Run time is about four minutes.

The study follows the library. Every convention that changes the fit is imported
from `notebooks/atm_straddle_lib.py` and never restated here: the forecast tables
with their labels and order, the stamp window, the early-close calendar, the
session rule, the window length, the start session, the truncation-lag rule, the
block bootstrap and the annualization. Only the fits themselves are written here,
and they are vectorized rather than looped. Gate 1 is what proves the two agree.
The imported constants are printed at the top of every run:

```
from the library  WINDOW_DAYS=250 MZ_START_DAY=63 FIT_MASK_MINUTES=(630, 960) EARLY_CLOSE_DATES=55
  a0        baseline (HAR + calendar OLS)                yhat_a0.parquet
  blk2      block-diagonal ridge                         yhat_blk2_fomc1.parquet
  blk2_inc  block-diagonal ridge, without the FOMC columns yhat_blk2.parquet
  lgbm      LightGBM                                     yhat_tree00.parquet
  xgb       XGBoost                                      yhat_tree16.parquet
  lasso_t   lasso (causally tuned)                       yhat_b2lasso_tuned.parquet
  lasso_f   lasso (fixed 1e-4)                           yhat_b2lasso_fomc1.parquet
  enet      elastic net (causally tuned)                 yhat_b3enet_tuned.parquet
```

## 1. What the grid is, and why nothing is chosen

The portfolio trades sign(s) at 15:30, with s = rv_hat - iv_var. The forecast
rv_hat is a recalibrated forecast of the 15:30-16:00 realized variance. The
recalibration is a Mincer-Zarnowitz line fitted causally on the square-root
scale. Write y = sqrt(rv_raw / B), with B the profile baseline. On a trailing
window of past sessions we fit m = a + b * yhat by weighted least squares, with
weights w = 1 / max(yhat, q10)^p. Here q10 is the 10th percentile of yhat inside
that window, floored at the smallest positive yhat if it is not positive. The
back-transform adds a variance term, because the mean does not commute with the
square: rv_hat = (m^2 + sigma^2) * B.

Three parts of that recipe are conventions, not results. The window. The weight
exponent p. The form of the variance term. The deck uses a flat window of 250
sessions, p = 2, and a variance term that is flat across the window. Each was
chosen once and never varied. This note varies all three at once and reports
every combination.

The three dimensions are:

- **Window**, six levels. Flat windows of 125, 250 and 500 sessions. Exponential
  windows with halflives of 125, 250 and 500 sessions. The halflives are the same
  three numbers as the flat lengths, so the six levels read as three lengths in
  two shapes; a halflife of h and a flat window of h are not the same amount of
  data, and no claim here depends on pairing them.
- **Weight exponent p**, three levels: 0, 1, 2. At p = 0 the fit is ordinary
  least squares. At p = 2 it is the map the deck uses.
- **Variance term**, two levels. FLAT is the current form: sigma^2 is the
  w-weighted mean squared residual of the window, ddof 0, and the same number is
  added to every row. PER-ROW takes the same model seriously. Weighting by
  w = g^(-p), with g = max(yhat, q10), is efficient when Var(e) is proportional
  to g^p. So estimate the common factor as sigma_w^2 = the plain mean over the
  window of (e / g^(p/2))^2, and give each row its own variance term:
  rv_hat = (m^2 + sigma_w^2 * g_row^p) * B.

Six windows times three exponents times two variance terms is 36 cells. The
current map is the cell (flat 250, p = 2, FLAT).

At p = 0 the weights are one, g^p is one, and the two variance terms are the same
number. Those cells are duplicates. The script checks this rather than assuming
it: the largest absolute difference between the FLAT and PER-ROW forecasts across
all eight forecasts and all six windows is 0.000e+00. So the 36 cells are 30
distinct maps in 6 duplicated pairs. The duplicates are kept in every table and
in the multiplicity draw, because a reader counting cells should see them.

Nothing here is chosen. A grid of 36 cells run on one sample will always have a
best cell. The question this note answers is how wide the grid is, in what
direction it is wide, and whether the width is larger than the noise in a single
cell. The current map's position is reported as a rank, not as a verdict.

Two implementation notes. The library's exponential path is unweighted, so p > 0
on an exponential window is a construction of this note. The combined weight is
the session decay times g^(-p). Under the exponential window q10 is the
decay-weighted 10th percentile of yhat over the prior history, taken as the
inverse of the weighted distribution with no interpolation; under a flat window
it is the plain 10th percentile of the window, which is what the library does.
Every quantity used from an exponential window is invariant to a common rescaling
of the weights, which is what makes the fits cheap.

Conventions taken from the library and not varied here: windows and the start
session are counted in sessions, that is, in dates that carry at least one
fit-mask row, not in calendar days; the first fitted session is session rank 63;
the fit mask is stamps 10:30 to 16:00 ET under bar-end labels; the 55 early-close
dates are excluded from every fit and from every output; and a session's
coefficients are applied to every row of that session.

## 2. Gate 1

Before anything else, the cell (flat 250, p = 2, FLAT) has to reproduce the
forecast the deck actually traded. The test is the 16:00 stamp on the 866 scored
days, against `results/atm_straddle_0dte_1530/daily_<tag>.parquet`, for all eight
forecasts, at a relative error below 1e-9.

```
  a0       n=866  matched=866  max_rel_err=0.000e+00  PASS
  blk2     n=866  matched=866  max_rel_err=0.000e+00  PASS
  blk2_inc n=866  matched=866  max_rel_err=0.000e+00  PASS
  lgbm     n=866  matched=866  max_rel_err=0.000e+00  PASS
  xgb      n=866  matched=866  max_rel_err=0.000e+00  PASS
  lasso_t  n=866  matched=866  max_rel_err=0.000e+00  PASS
  lasso_f  n=866  matched=866  max_rel_err=0.000e+00  PASS
  enet     n=866  matched=866  max_rel_err=0.000e+00  PASS
```

The agreement is bit-exact, not merely within tolerance. The fit here is written
from scratch and shares no code with the library's day loop, so this is a real
check on the window, the session counting, the mask, the weights and the
back-transform.

The same cell reproduces the traded portfolio. Its sign(s) Sharpe is 0.9673 for
the baseline (HAR + calendar OLS), 1.3383 for the block-diagonal ridge, 1.2707
for the ridge without the FOMC columns, 1.4594 for LightGBM, 1.4980 for XGBoost,
1.4131 for the causally tuned lasso, 1.5098 for the fixed lasso, and 1.0225 for
the elastic net. Always short is 0.2038 and does not depend on the map.

Every cell produces a usable forecast on every scored row. For each forecast
there are 12984 fit-mask rows on 1082 session dates in 2020-01-03 to 2024-04-30,
all 12984 are usable on all 36 cells, and all 866 traded 16:00 rows are present.
So every comparison below is on the same rows.

## 3. Forecast loss first

QLIKE is rv_raw/rv_hat - log(rv_raw/rv_hat) - 1. Lower is better. It is averaged
two ways. First over all 12984 fit-mask rows in the scored period. Second over
the 866 traded 16:00 rows only.

### QLIKE, all fit-mask rows in 2020-01-03 to 2024-04-30

```
                        a0    blk2  blk2_inc    lgbm     xgb  lasso_t  lasso_f    enet
flat125|p0|FLAT    0.20076 0.19036   0.19866 0.19528 0.19817  0.19893  0.19047 0.19867
flat125|p0|PER-ROW 0.20076 0.19036   0.19866 0.19528 0.19817  0.19893  0.19047 0.19867
flat125|p1|FLAT    0.19814 0.18846   0.19696 0.19229 0.19534  0.19703  0.18859 0.19682
flat125|p1|PER-ROW 0.19685 0.18756   0.19619 0.19072 0.19385  0.19626  0.18785 0.19597
flat125|p2|FLAT    0.19736 0.18847   0.19698 0.19126 0.19440  0.19680  0.18865 0.19653
flat125|p2|PER-ROW 0.19676 0.18791   0.19661 0.19081 0.19394  0.19667  0.18830 0.19618
flat250|p0|FLAT    0.20208 0.19149   0.20003 0.19650 0.19918  0.20011  0.19148 0.19965
flat250|p0|PER-ROW 0.20208 0.19149   0.20003 0.19650 0.19918  0.20011  0.19148 0.19965
flat250|p1|FLAT    0.19896 0.18862   0.19738 0.19257 0.19548  0.19735  0.18869 0.19691
flat250|p1|PER-ROW 0.19789 0.18752   0.19640 0.19081 0.19387  0.19639  0.18773 0.19585
flat250|p2|FLAT    0.19794 0.18819   0.19694 0.19113 0.19418  0.19667  0.18834 0.19622
flat250|p2|PER-ROW 0.19787 0.18780   0.19668 0.19060 0.19377  0.19651  0.18814 0.19588
flat500|p0|FLAT    0.20309 0.19155   0.19998 0.19650 0.19893  0.19975  0.19159 0.19930
flat500|p0|PER-ROW 0.20309 0.19155   0.19998 0.19650 0.19893  0.19975  0.19159 0.19930
flat500|p1|FLAT    0.20029 0.18865   0.19734 0.19258 0.19534  0.19691  0.18876 0.19654
flat500|p1|PER-ROW 0.20004 0.18795   0.19683 0.19112 0.19414  0.19628  0.18823 0.19579
flat500|p2|FLAT    0.19963 0.18820   0.19696 0.19128 0.19427  0.19633  0.18839 0.19594
flat500|p2|PER-ROW 0.20083 0.18836   0.19737 0.19118 0.19449  0.19675  0.18876 0.19608
ewma125|p0|FLAT    0.20213 0.19087   0.19929 0.19606 0.19863  0.19929  0.19089 0.19896
ewma125|p0|PER-ROW 0.20213 0.19087   0.19929 0.19606 0.19863  0.19929  0.19089 0.19896
ewma125|p1|FLAT    0.19907 0.18825   0.19696 0.19223 0.19508  0.19676  0.18836 0.19645
ewma125|p1|PER-ROW 0.19825 0.18737   0.19628 0.19065 0.19370  0.19602  0.18762 0.19561
ewma125|p2|FLAT    0.19818 0.18793   0.19668 0.19092 0.19395  0.19627  0.18810 0.19594
ewma125|p2|PER-ROW 0.19844 0.18771   0.19674 0.19063 0.19383  0.19643  0.18805 0.19588
ewma250|p0|FLAT    0.20380 0.19087   0.19925 0.19628 0.19857  0.19927  0.19087 0.19889
ewma250|p0|PER-ROW 0.20380 0.19087   0.19925 0.19628 0.19857  0.19927  0.19087 0.19889
ewma250|p1|FLAT    0.20063 0.18819   0.19688 0.19222 0.19501  0.19670  0.18827 0.19635
ewma250|p1|PER-ROW 0.20018 0.18746   0.19639 0.19079 0.19390  0.19610  0.18768 0.19566
ewma250|p2|FLAT    0.19993 0.18785   0.19661 0.19099 0.19407  0.19625  0.18799 0.19589
ewma250|p2|PER-ROW 0.20070 0.18792   0.19700 0.19098 0.19437  0.19666  0.18823 0.19608
ewma500|p0|FLAT    0.20435 0.19014   0.19850 0.19561 0.19776  0.19848  0.19013 0.19808
ewma500|p0|PER-ROW 0.20435 0.19014   0.19850 0.19561 0.19776  0.19848  0.19013 0.19808
ewma500|p1|FLAT    0.20178 0.18794   0.19664 0.19199 0.19482  0.19647  0.18802 0.19610
ewma500|p1|PER-ROW 0.20168 0.18748   0.19642 0.19081 0.19406  0.19615  0.18769 0.19567
ewma500|p2|FLAT    0.20145 0.18775   0.19652 0.19105 0.19424  0.19624  0.18786 0.19587
ewma500|p2|PER-ROW 0.20260 0.18811   0.19722 0.19128 0.19492  0.19693  0.18841 0.19632
```

### QLIKE, the 866 traded 16:00 rows

```
                        a0    blk2  blk2_inc    lgbm     xgb  lasso_t  lasso_f    enet
flat125|p0|FLAT    0.11056 0.11397   0.11427 0.10419 0.10493  0.10690  0.11505 0.10710
flat125|p0|PER-ROW 0.11056 0.11397   0.11427 0.10419 0.10493  0.10690  0.11505 0.10710
flat125|p1|FLAT    0.10789 0.11129   0.11163 0.10171 0.10281  0.10467  0.11249 0.10514
flat125|p1|PER-ROW 0.10635 0.10984   0.10990 0.10099 0.10236  0.10387  0.11111 0.10448
flat125|p2|FLAT    0.10686 0.11009   0.11046 0.10079 0.10212  0.10381  0.11133 0.10436
flat125|p2|PER-ROW 0.10550 0.10902   0.10903 0.10206 0.10365  0.10427  0.11040 0.10485
flat250|p0|FLAT    0.11154 0.11510   0.11536 0.10522 0.10574  0.10802  0.11633 0.10814
flat250|p0|PER-ROW 0.11154 0.11510   0.11536 0.10522 0.10574  0.10802  0.11633 0.10814
flat250|p1|FLAT    0.10882 0.11161   0.11203 0.10227 0.10331  0.10525  0.11290 0.10557
flat250|p1|PER-ROW 0.10769 0.11007   0.11019 0.10157 0.10299  0.10444  0.11142 0.10489
flat250|p2|FLAT    0.10782 0.10999   0.11045 0.10124 0.10260  0.10408  0.11128 0.10449
flat250|p2|PER-ROW 0.10722 0.10899   0.10911 0.10272 0.10453  0.10475  0.11043 0.10529
flat500|p0|FLAT    0.11166 0.11330   0.11327 0.10344 0.10410  0.10637  0.11470 0.10670
flat500|p0|PER-ROW 0.11166 0.11330   0.11327 0.10344 0.10410  0.10637  0.11470 0.10670
flat500|p1|FLAT    0.10966 0.10983   0.10988 0.10103 0.10242  0.10377  0.11119 0.10431
flat500|p1|PER-ROW 0.10968 0.10848   0.10828 0.10096 0.10286  0.10354  0.10991 0.10423
flat500|p2|FLAT    0.10925 0.10825   0.10827 0.10053 0.10237  0.10274  0.10954 0.10340
flat500|p2|PER-ROW 0.11042 0.10755   0.10734 0.10288 0.10543  0.10415  0.10900 0.10498
ewma125|p0|FLAT    0.11120 0.11394   0.11404 0.10393 0.10449  0.10668  0.11523 0.10693
ewma125|p0|PER-ROW 0.11120 0.11394   0.11404 0.10393 0.10449  0.10668  0.11523 0.10693
ewma125|p1|FLAT    0.10855 0.11063   0.11082 0.10123 0.10237  0.10410  0.11195 0.10454
ewma125|p1|PER-ROW 0.10779 0.10921   0.10915 0.10079 0.10234  0.10353  0.11058 0.10411
ewma125|p2|FLAT    0.10772 0.10912   0.10932 0.10042 0.10191  0.10307  0.11041 0.10360
ewma125|p2|PER-ROW 0.10767 0.10828   0.10821 0.10225 0.10424  0.10402  0.10971 0.10469
ewma250|p0|FLAT    0.11289 0.11388   0.11390 0.10381 0.10431  0.10666  0.11529 0.10696
ewma250|p0|PER-ROW 0.11289 0.11388   0.11390 0.10381 0.10431  0.10666  0.11529 0.10696
ewma250|p1|FLAT    0.11048 0.11031   0.11039 0.10127 0.10264  0.10402  0.11168 0.10452
ewma250|p1|PER-ROW 0.11041 0.10899   0.10886 0.10120 0.10312  0.10375  0.11043 0.10441
ewma250|p2|FLAT    0.11004 0.10872   0.10878 0.10081 0.10264  0.10303  0.11003 0.10364
ewma250|p2|PER-ROW 0.11102 0.10811   0.10796 0.10325 0.10578  0.10440  0.10956 0.10521
ewma500|p0|FLAT    0.11374 0.11323   0.11319 0.10318 0.10379  0.10611  0.11472 0.10645
ewma500|p0|PER-ROW 0.11374 0.11323   0.11319 0.10318 0.10379  0.10611  0.11472 0.10645
ewma500|p1|FLAT    0.11216 0.10987   0.10985 0.10125 0.10295  0.10387  0.11127 0.10442
ewma500|p1|PER-ROW 0.11266 0.10868   0.10846 0.10153 0.10394  0.10387  0.11014 0.10461
ewma500|p2|FLAT    0.11225 0.10841   0.10836 0.10122 0.10345  0.10309  0.10971 0.10376
ewma500|p2|PER-ROW 0.11400 0.10791   0.10766 0.10408 0.10724  0.10470  0.10936 0.10563
```

### Paired differences

Each cell's loss is differenced against the current cell's loss row by row, and
the mean difference is divided by an autocorrelation-robust standard error with a
Bartlett kernel at lag floor(1.5 n^(1/3)). That is lag 35 on the 12984 fit-mask
rows and lag 14 on the 866 daily losses. A negative t means the cell loses less
than the current map.

```
Diebold-Mariano t, all fit-mask rows, Bartlett lag 35
                      a0  blk2  blk2_inc  lgbm   xgb  lasso_t  lasso_f  enet
flat125|p0|FLAT     2.89  2.33      1.87  4.03  3.84     2.65     2.25  2.85
flat125|p0|PER-ROW  2.89  2.33      1.87  4.03  3.84     2.65     2.25  2.85
flat125|p1|FLAT     0.34  0.58      0.03  2.28  2.15     0.71     0.51  1.23
flat125|p1|PER-ROW -1.97 -0.90     -0.91 -0.90 -0.72    -0.58    -0.68 -0.38
flat125|p2|FLAT    -1.03  0.73      0.08  0.27  0.46     0.26     0.80  0.71
flat125|p2|PER-ROW -1.33 -0.26     -0.28 -0.34 -0.26    -0.01    -0.04 -0.04
flat250|p0|FLAT     4.39  3.29      3.18  4.79  4.60     3.74     3.07  3.73
flat250|p0|PER-ROW  4.39  3.29      3.18  4.79  4.60     3.74     3.07  3.73
flat250|p1|FLAT     2.86  1.04      1.10  3.37  3.13     1.82     0.81  1.86
flat250|p1|PER-ROW -0.12 -1.13     -0.80 -1.13 -1.06    -0.45    -0.99 -0.66
flat250|p2|FLAT      NaN   NaN       NaN   NaN   NaN      NaN      NaN   NaN
flat250|p2|PER-ROW -0.09 -0.36     -0.23 -0.62 -0.49    -0.16    -0.18 -0.36
flat500|p0|FLAT     4.51  3.13      2.72  4.32  4.01     2.46     3.05  2.61
flat500|p0|PER-ROW  4.51  3.13      2.72  4.32  4.01     2.46     3.05  2.61
flat500|p1|FLAT     3.21  0.73      0.57  2.24  1.84     0.31     0.69  0.44
flat500|p1|PER-ROW  2.29 -0.24     -0.10 -0.03 -0.06    -0.44    -0.12 -0.52
flat500|p2|FLAT     2.49  0.02      0.03  0.27  0.16    -0.53     0.09 -0.47
flat500|p2|PER-ROW  2.26  0.12      0.27  0.04  0.27     0.06     0.28 -0.12
ewma125|p0|FLAT     4.20  2.92      2.63  4.49  4.15     2.81     2.74  3.03
ewma125|p0|PER-ROW  4.20  2.92      2.63  4.49  4.15     2.81     2.74  3.03
ewma125|p1|FLAT     2.54  0.17      0.04  2.61  2.17     0.20     0.04  0.58
ewma125|p1|PER-ROW  0.59 -1.17     -0.82 -1.27 -1.25    -1.00    -1.01 -0.99
ewma125|p2|FLAT     0.88 -1.23     -1.03 -0.94 -0.99    -1.45    -1.18 -1.12
ewma125|p2|PER-ROW  0.56 -0.41     -0.15 -0.53 -0.37    -0.22    -0.24 -0.33
ewma250|p0|FLAT     4.91  2.92      2.48  4.40  4.00     2.52     2.75  2.71
ewma250|p0|PER-ROW  4.91  2.92      2.48  4.40  4.00     2.52     2.75  2.71
ewma250|p1|FLAT     3.72 -0.00     -0.12  2.11  1.66     0.04    -0.16  0.24
ewma250|p1|PER-ROW  2.67 -0.90     -0.59 -0.64 -0.50    -0.76    -0.81 -0.79
ewma250|p2|FLAT     3.10 -0.92     -0.75 -0.35 -0.26    -0.90    -0.98 -0.77
ewma250|p2|PER-ROW  2.36 -0.21      0.04 -0.15  0.17    -0.01    -0.08 -0.14
ewma500|p0|FLAT     5.01  2.30      1.76  3.93  3.46     1.82     2.14  1.97
ewma500|p0|PER-ROW  5.01  2.30      1.76  3.93  3.46     1.82     2.14  1.97
ewma500|p1|FLAT     4.15 -0.48     -0.51  1.49  1.12    -0.31    -0.66 -0.21
ewma500|p1|PER-ROW  3.46 -0.78     -0.49 -0.50 -0.17    -0.62    -0.71 -0.69
ewma500|p2|FLAT     3.86 -0.88     -0.70 -0.16  0.09    -0.70    -0.97 -0.62
ewma500|p2|PER-ROW  3.28 -0.06      0.18  0.12  0.61     0.21     0.05  0.08
```

```
Diebold-Mariano t, 866 traded days, Bartlett lag 14
                      a0  blk2  blk2_inc  lgbm   xgb  lasso_t  lasso_f  enet
flat125|p0|FLAT     1.60  2.31      2.48  1.41  1.10     1.98     2.15  1.81
flat125|p0|PER-ROW  1.60  2.31      2.48  1.41  1.10     1.98     2.15  1.81
flat125|p1|FLAT     0.08  1.56      1.60  0.49  0.21     0.71     1.46  0.82
flat125|p1|PER-ROW -1.65 -0.18     -0.62 -0.33 -0.30    -0.20    -0.21 -0.01
flat125|p2|FLAT    -1.03  0.18      0.01 -0.61 -0.59    -0.34     0.10 -0.18
flat125|p2|PER-ROW -1.40 -0.92     -1.18  0.57  0.71     0.13    -0.83  0.27
flat250|p0|FLAT     1.85  2.73      2.78  1.69  1.35     2.33     2.66  2.13
flat250|p0|PER-ROW  1.85  2.73      2.78  1.69  1.35     2.33     2.66  2.13
flat250|p1|FLAT     1.32  2.23      2.31  1.20  0.83     1.87     2.19  1.68
flat250|p1|PER-ROW -0.20  0.10     -0.35  0.69  0.82     0.46     0.17  0.54
flat250|p2|FLAT      NaN   NaN       NaN   NaN   NaN      NaN      NaN   NaN
flat250|p2|PER-ROW -0.39 -0.92     -1.17  1.04  1.30     0.47    -0.77  0.62
flat500|p0|FLAT     2.72  2.80      2.45  1.58  1.10     2.16     2.83  2.07
flat500|p0|PER-ROW  2.72  2.80      2.45  1.58  1.10     2.16     2.83  2.07
flat500|p1|FLAT     1.98 -0.22     -0.71 -0.28 -0.23    -0.41    -0.13 -0.26
flat500|p1|PER-ROW  1.33 -1.35     -1.74 -0.27  0.22    -0.42    -1.23 -0.23
flat500|p2|FLAT     1.43 -2.23     -2.48 -0.81 -0.24    -1.61    -2.31 -1.39
flat500|p2|PER-ROW  1.27 -1.66     -1.92  0.93  1.48     0.04    -1.57  0.30
ewma125|p0|FLAT     1.91  2.68      2.70  1.37  0.96     2.12     2.62  1.93
ewma125|p0|PER-ROW  1.91  2.68      2.70  1.37  0.96     2.12     2.62  1.93
ewma125|p1|FLAT     1.05  1.12      0.72 -0.01 -0.32     0.04     1.13  0.10
ewma125|p1|PER-ROW -0.03 -0.99     -1.49 -0.67 -0.38    -0.57    -0.89 -0.44
ewma125|p2|FLAT    -0.22 -2.78     -2.80 -1.81 -1.52    -2.29    -2.91 -2.13
ewma125|p2|PER-ROW -0.09 -1.49     -1.75  0.67  1.05    -0.04    -1.38  0.15
ewma250|p0|FLAT     2.66  3.06      2.96  1.44  0.98     2.36     3.10  2.21
ewma250|p0|PER-ROW  2.66  3.06      2.96  1.44  0.98     2.36     3.10  2.21
ewma250|p1|FLAT     2.25  0.51     -0.10  0.04  0.04    -0.09     0.63  0.05
ewma250|p1|PER-ROW  1.64 -1.03     -1.46 -0.04  0.51    -0.28    -0.88 -0.08
ewma250|p2|FLAT     1.94 -2.25     -2.49 -0.56  0.04    -1.47    -2.34 -1.33
ewma250|p2|PER-ROW  1.43 -1.42     -1.68  1.16  1.73     0.19    -1.31  0.48
ewma500|p0|FLAT     3.02  2.87      2.56  1.21  0.78     2.04     2.97  1.98
ewma500|p0|PER-ROW  3.02  2.87      2.56  1.21  0.78     2.04     2.97  1.98
ewma500|p1|FLAT     2.60 -0.17     -0.72  0.01  0.35    -0.25    -0.02 -0.10
ewma500|p1|PER-ROW  2.14 -1.16     -1.57  0.24  1.00    -0.16    -1.02  0.10
ewma500|p2|FLAT     2.46 -2.08     -2.36 -0.02  0.71    -1.02    -2.18 -0.86
ewma500|p2|PER-ROW  2.06 -1.43     -1.72  1.47  2.19     0.33    -1.34  0.70
```

There are 280 non-trivial comparisons, that is 35 cells times eight forecasts. On
all fit-mask rows 110 of them reach |t| > 2. Every one of those 110 is the cell
losing more than the current map; not one cell beats it at |t| > 2 on any
forecast, and the largest improvement anywhere is t = -1.97. On the 866 traded
days 81 reach |t| > 2: 67 worse and 14 better, with the largest improvement at
t = -2.91.

The direction is mostly one way, and where it is not, it is structured rather
than random. Of the 110 significant-worse comparisons on all rows, 88 are p = 0
cells. All 96 p = 0 comparisons are positive, running from t = 1.76 to t = 5.01
on all rows and from 0.78 to 3.10 on traded days. Dropping the
variance-stabilizing weight is the one change in the grid that reliably hurts the
forecast.

The 14 cells that beat the current map on the traded rows are not scattered
either. They are p = 2 FLAT cells on windows longer or smoother than the current
one, that is flat 500 and the three exponential windows, and they appear only on
the two ridges, the two lassos and the elastic net, never on the trees or on the
baseline (HAR + calendar OLS). The same fourteen do not beat it on all fit-mask
rows, where their t runs from -1.45 to +0.09 and not one of them is significant.
So that improvement lives in the 866-row subset, not in the forecast as a whole.

### Is the cell calibrated?

The mean map m is the same for the two variance terms, so there are 18 distinct
mean maps. For each one, y is regressed on m over the scored fit-mask rows. A
calibrated map has slope 1 and intercept 0.

```
slope
tag            a0   blk2  blk2_inc   lgbm    xgb  lasso_t  lasso_f   enet
flat125|p0 0.9820 0.9857    0.9866 0.9828 0.9763   0.9807   0.9850 0.9810
flat125|p1 0.9918 1.0109    1.0114 0.9752 0.9675   1.0019   1.0122 1.0024
flat125|p2 1.0041 1.0403    1.0414 0.9673 0.9610   1.0275   1.0427 1.0282
flat250|p0 0.9735 0.9856    0.9861 0.9829 0.9723   0.9756   0.9854 0.9764
flat250|p1 0.9798 1.0055    1.0053 0.9711 0.9588   0.9918   1.0073 0.9931
flat250|p2 0.9900 1.0317    1.0316 0.9606 0.9499   1.0152   1.0349 1.0162
flat500|p0 0.9636 0.9860    0.9847 0.9819 0.9679   0.9771   0.9866 0.9771
flat500|p1 0.9708 1.0048    1.0034 0.9695 0.9549   0.9917   1.0070 0.9924
flat500|p2 0.9830 1.0315    1.0302 0.9599 0.9486   1.0140   1.0354 1.0141
ewma125|p0 0.9753 0.9875    0.9865 0.9875 0.9753   0.9803   0.9878 0.9800
ewma125|p1 0.9809 1.0066    1.0054 0.9734 0.9605   0.9950   1.0085 0.9955
ewma125|p2 0.9925 1.0330    1.0319 0.9623 0.9520   1.0172   1.0364 1.0176
ewma250|p0 0.9802 0.9882    0.9864 0.9924 0.9740   0.9809   0.9889 0.9797
ewma250|p1 0.9826 1.0057    1.0040 0.9731 0.9563   0.9939   1.0078 0.9936
ewma250|p2 0.9950 1.0315    1.0298 0.9606 0.9487   1.0156   1.0349 1.0149
ewma500|p0 0.9877 0.9892    0.9868 0.9947 0.9725   0.9817   0.9900 0.9797
ewma500|p1 0.9861 1.0050    1.0029 0.9723 0.9533   0.9932   1.0069 0.9920
ewma500|p2 0.9976 1.0287    1.0267 0.9597 0.9467   1.0136   1.0318 1.0120
```

```
intercept
tag              a0     blk2  blk2_inc    lgbm     xgb  lasso_t  lasso_f     enet
flat125|p0  0.01873  0.01251   0.01168 0.01547 0.02130  0.01678  0.01308  0.01743
flat125|p1  0.00973 -0.01135  -0.01171 0.02225 0.02933 -0.00323 -0.01256 -0.00275
flat125|p2 -0.00049 -0.03528  -0.03639 0.02895 0.03489 -0.02403 -0.03727 -0.02395
flat250|p0  0.02944  0.01211   0.01177 0.01656 0.02642  0.02191  0.01202  0.02214
flat250|p1  0.02354 -0.00638  -0.00631 0.02740 0.03898  0.00678 -0.00828  0.00650
flat250|p2  0.01518 -0.02704  -0.02722 0.03645 0.04672 -0.01188 -0.02999 -0.01210
flat500|p0  0.04207  0.01328   0.01418 0.02015 0.03368  0.02214  0.01223  0.02322
flat500|p1  0.03537 -0.00391  -0.00299 0.03189 0.04612  0.00880 -0.00638  0.00913
flat500|p2  0.02532 -0.02486  -0.02410 0.03993 0.05151 -0.00872 -0.02856 -0.00804
ewma125|p0  0.02843  0.01106   0.01188 0.01319 0.02468  0.01818  0.01044  0.01944
ewma125|p1  0.02320 -0.00669  -0.00573 0.02628 0.03856  0.00460 -0.00880  0.00505
ewma125|p2  0.01366 -0.02769  -0.02698 0.03572 0.04583 -0.01317 -0.03089 -0.01274
ewma250|p0  0.02622  0.01046   0.01208 0.01007 0.02766  0.01858  0.00936  0.02055
ewma250|p1  0.02409 -0.00562  -0.00414 0.02816 0.04439  0.00674 -0.00791  0.00779
ewma250|p2  0.01404 -0.02594  -0.02463 0.03878 0.05096 -0.01048 -0.02927 -0.00921
ewma500|p0  0.02060  0.00990   0.01213 0.00859 0.03015  0.01891  0.00864  0.02147
ewma500|p1  0.02237 -0.00451  -0.00261 0.02979 0.04841  0.00844 -0.00670  0.01026
ewma500|p2  0.01307 -0.02338  -0.02166 0.04053 0.05417 -0.00785 -0.02648 -0.00575
```

Every map is close to calibrated. Slopes run from 0.9467 to 1.0427 and intercepts
from -0.03727 to 0.05417. The pattern is systematic rather than random. Raising p
raises the slope and lowers the intercept for both ridges, both lassos and the
elastic net, and lowers the slope and raises the intercept for LightGBM and
XGBoost. Lengthening the window moves both in the same direction as lowering p.
So the p = 2 maps sit closest to slope 1 for the linear forecasts, and the p = 0
maps sit closest for the tree forecasts. QLIKE still prefers p = 2 for the trees,
which says the loss is being driven by the variance term and the weighting, not
by the slope of the mean map.

## 4. Then the trade

The rule is sign(s): short the straddle when rv_hat is at or below iv_var, buy it
otherwise. iv_var and the return R are read from `daily_<tag>.parquet` and are
not recomputed. Only the forecast changes across cells. Sharpe is the daily mean
over the daily standard deviation, times sqrt(252), over the same 866 days.

### sign(s) Sharpe, 866 days

```
                       a0   blk2  blk2_inc   lgbm    xgb  lasso_t  lasso_f   enet
flat125|p0|FLAT    1.0556 1.0631    1.3286 1.0046 1.2923   1.3872   1.2282 1.0124
flat125|p0|PER-ROW 1.0556 1.0631    1.3286 1.0046 1.2923   1.3872   1.2282 1.0124
flat125|p1|FLAT    0.7534 1.2837    1.3897 1.1667 1.3218   1.7383   1.3476 0.8421
flat125|p1|PER-ROW 0.8798 1.3477    1.4686 1.3959 1.2973   1.2524   1.3230 1.1148
flat125|p2|FLAT    0.9348 1.6318    1.2438 1.1437 1.2881   1.6184   1.6062 1.1893
flat125|p2|PER-ROW 0.9394 1.4332    1.4002 1.4983 1.4669   1.2755   1.4669 1.2082
flat250|p0|FLAT    1.1035 1.3650    1.3452 1.0809 1.4328   1.1460   1.1183 0.6715
flat250|p0|PER-ROW 1.1035 1.3650    1.3452 1.0809 1.4328   1.1460   1.1183 0.6715
flat250|p1|FLAT    0.9184 1.1214    1.4032 1.2429 1.6257   1.3343   1.4878 0.9446
flat250|p1|PER-ROW 0.9687 1.1852    1.3935 1.3302 1.3924   1.3314   1.1721 1.0085
flat250|p2|FLAT    0.9673 1.3383    1.2707 1.4594 1.4980   1.4131   1.5098 1.0225
flat250|p2|PER-ROW 0.7452 1.3016    1.2216 1.3221 1.2882   1.1247   1.4126 0.8892
flat500|p0|FLAT    1.0499 1.1180    1.4384 1.0383 1.4298   1.2244   1.0281 0.8505
flat500|p0|PER-ROW 1.0499 1.1180    1.4384 1.0383 1.4298   1.2244   1.0281 0.8505
flat500|p1|FLAT    1.0127 1.2895    1.1570 1.2208 1.2616   1.2365   1.1648 0.9920
flat500|p1|PER-ROW 0.9627 1.2844    1.5480 1.2231 1.1844   0.8835   1.3966 0.9240
flat500|p2|FLAT    1.1772 1.1237    1.3026 1.2890 0.9043   1.5479   1.3143 1.1942
flat500|p2|PER-ROW 0.8387 1.2596    1.3685 1.3744 1.0504   1.1657   1.2591 1.0449
ewma125|p0|FLAT    1.0048 1.1719    1.3869 1.1410 1.1569   1.4640   1.1075 0.9061
ewma125|p0|PER-ROW 1.0048 1.1719    1.3869 1.1410 1.1569   1.4640   1.1075 0.9061
ewma125|p1|FLAT    1.0835 1.3467    1.4544 1.2213 1.3350   1.3887   1.4788 1.0152
ewma125|p1|PER-ROW 0.9821 1.2189    1.5607 1.2855 1.4237   1.1613   1.2932 0.8963
ewma125|p2|FLAT    1.0386 1.5702    1.4404 1.1797 1.2114   1.3996   1.6924 1.0945
ewma125|p2|PER-ROW 0.6417 1.1728    1.2109 1.2544 1.3541   0.9501   1.2344 1.0167
ewma250|p0|FLAT    1.0258 1.0218    1.3035 1.0231 1.3403   1.3277   1.0806 0.7305
ewma250|p0|PER-ROW 1.0258 1.0218    1.3035 1.0231 1.3403   1.3277   1.0806 0.7305
ewma250|p1|FLAT    1.0038 1.3620    1.1846 1.2607 1.2468   1.2595   1.2309 1.0617
ewma250|p1|PER-ROW 0.9160 1.0275    1.5886 1.1914 1.3090   0.9732   1.2261 1.0014
ewma250|p2|FLAT    1.0638 1.5132    1.3563 1.1990 1.1361   1.3900   1.4617 1.1559
ewma250|p2|PER-ROW 0.8503 1.1866    1.0974 1.1979 1.0728   0.9272   1.2432 1.0947
ewma500|p0|FLAT    0.9939 1.1055    1.1669 1.0876 1.1207   1.2282   0.9462 0.8225
ewma500|p0|PER-ROW 0.9939 1.1055    1.1669 1.0876 1.1207   1.2282   0.9462 0.8225
ewma500|p1|FLAT    1.0728 1.2458    1.2988 1.3551 1.2629   1.3602   1.1631 0.9331
ewma500|p1|PER-ROW 1.1601 0.8467    1.4868 1.1720 1.2316   0.8803   1.1238 1.0978
ewma500|p2|FLAT    1.1962 1.4372    1.4433 1.2100 0.9640   1.3966   1.4005 1.0394
ewma500|p2|PER-ROW 1.1486 1.3478    0.9146 1.2544 1.0777   0.9290   1.0966 1.0611
```

The Sharpe table has none of the structure the QLIKE table has. The p = 0 cells
lose on forecast loss for every forecast, but they are neither systematically
worst nor systematically best here. For the ridge without the FOMC columns a
p = 0 cell reaches 1.4384, above that column's current-cell 1.2707. For the
block-diagonal ridge no p = 0 cell exceeds 1.3650 while the column reaches
1.6318. The columns also disagree with each other about which cell is good: the
cell that tops the block-diagonal ridge column, flat125|p2|FLAT, is 0.9348 for
the baseline (HAR + calendar OLS) and 1.1437 for LightGBM. That is what a table
of noise looks like.

The mean return, the t statistic, the number of days the rule buys, and the
number of days whose side differs from the current map are in `cells_detail.csv`;
the script also prints the full per-cell table for the block-diagonal ridge. The
side flips are small.

```
days whose side differs from the current map, out of 866
                    a0  blk2  blk2_inc  lgbm  xgb  lasso_t  lasso_f  enet
flat125|p0|FLAT     51    48        45    50   56       51       43    56
flat125|p0|PER-ROW  51    48        45    50   56       51       43    56
flat125|p1|FLAT     28    19        31    35   36       30       29    33
flat125|p1|PER-ROW  30    42        40    33   26       45       39    42
flat125|p2|FLAT     32    21        28    30   27       30       22    28
flat125|p2|PER-ROW  44    44        47    51   39       49       48    60
flat250|p0|FLAT     53    52        51    52   53       56       45    58
flat250|p0|PER-ROW  53    52        51    52   53       56       45    58
flat250|p1|FLAT     21    27        25    20   19       18       21    21
flat250|p1|PER-ROW  26    34        34    21   13       36       30    39
flat250|p2|FLAT      0     0         0     0    0        0        0     0
flat250|p2|PER-ROW  45    46        44    50   39       43       34    54
flat500|p0|FLAT     66    62        55    58   56       60       55    65
flat500|p0|PER-ROW  66    62        55    58   56       60       55    65
flat500|p1|FLAT     50    35        45    35   28       35       32    39
flat500|p1|PER-ROW  51    44        50    40   36       51       43    52
flat500|p2|FLAT     38    26        30    32   23       30       21    28
flat500|p2|PER-ROW  63    56        57    63   55       59       49    70
ewma125|p0|FLAT     46    54        55    50   55       52       49    62
ewma125|p0|PER-ROW  46    54        55    50   55       52       49    62
ewma125|p1|FLAT     20    23        31    26   22       21       22    27
ewma125|p1|PER-ROW  39    39        44    26   25       39       36    53
ewma125|p2|FLAT     20    11        18    16   17       17        6    12
ewma125|p2|PER-ROW  49    52        53    56   48       52       43    65
ewma250|p0|FLAT     67    62        55    59   58       57       54    67
ewma250|p0|PER-ROW  67    62        55    59   58       57       54    67
ewma250|p1|FLAT     51    35        42    30   28       31       27    41
ewma250|p1|PER-ROW  50    44        48    34   34       44       40    57
ewma250|p2|FLAT     38    24        26    27   29       27       15    24
ewma250|p2|PER-ROW  62    57        56    60   56       64       49    69
ewma500|p0|FLAT     71    56        53    57   56       55       58    63
ewma500|p0|PER-ROW  71    56        53    57   56       55       58    63
ewma500|p1|FLAT     67    35        44    35   37       38       27    47
ewma500|p1|PER-ROW  61    46        52    43   44       51       44    58
ewma500|p2|FLAT     62    29        36    42   35       35       26    31
ewma500|p2|PER-ROW  76    58        62    65   62       62       54    74
```

The largest disagreement anywhere in the grid is 76 days out of 866, and for the
block-diagonal ridge it is 62 out of 866. So the 36 cells are 36 slightly
different versions of the same portfolio, and their Sharpes differ mostly because
a few dozen days moved.

## 5. Multiplicity

This section is the block-diagonal ridge only, stated so that the reader is not
left wondering whether it was picked after the fact.

The resampling is a circular block bootstrap with a block of 21 days, B = 2000
draws, seeded rng(0), on the 866 days. Each draw resamples days once and every
cell is recomputed on the same resampled days, so the cells stay comparable
within a draw.

```
  current cell Sharpe                 1.3383
  current cell bootstrap SE           0.5476
  current cell bootstrap 2.5/97.5%    0.2745 .. 2.4216
  observed max over the 36 cells      1.6318  (flat125|p2|FLAT)
  observed mean over the 36 cells     1.2379
  observed selection premium max-mean 0.3939
  bootstrap max-Sharpe distribution   mean 1.6962  sd 0.5069
    quantiles 5/25/50/75/95%          0.8517  1.3634  1.6756  2.0443  2.5320
  fraction of draws with max <= obs   0.4685
  bootstrap selection premium max-mean  mean 0.4699  sd 0.1157  95% 0.6751
```

Read the first line against the rest. A single cell's Sharpe carries a standard
error of 0.5476 on 866 days. The whole grid spans 0.8467 to 1.6318, a width of
0.7851, which is under one and a half standard errors of any one cell in it.

The last two lines are the multiplicity point. The best of 36 exceeds the average
of 36 by 0.4699 on average under resampling, with a 95th percentile of 0.6751,
purely because 36 correlated series were maximized over. The observed gap is
0.3939, below the average gap a maximum produces when nothing distinguishes the
cells. The observed maximum of 1.6318 sits at the 46.9th percentile of the
bootstrap maximum distribution. This resampling is centred on the sample, so it
is not a test of no skill; it is a measure of how far a maximum over 36 wanders,
and the answer is further than the spread actually seen.

The cells are also close to each other by construction. Fifteen of the 36 agree
with the current map on the side of the trade on more than 95% of the 866 days.
Of the 630 distinct pairs of cells, 394 agree on more than 95% of days. The least
similar cell to the current map still agrees on 92.84% of days. There are not 36
independent portfolios here.

## 6. Sensitivity

For the block-diagonal ridge, each dimension is grouped by level and the range
over the other dimensions is reported.

```
        dimension   level  n_cells  sharpe_min  sharpe_mean  sharpe_max  qlike_traded_min  qlike_traded_mean  qlike_traded_max  qlike_all_min  qlike_all_mean  qlike_all_max
           window flat125        6      1.0631       1.3038      1.6318            0.1090             0.1114            0.1140         0.1876          0.1889         0.1904
           window flat250        6      1.1214       1.2794      1.3650            0.1090             0.1118            0.1151         0.1875          0.1892         0.1915
           window flat500        6      1.1180       1.1989      1.2895            0.1076             0.1101            0.1133         0.1880          0.1894         0.1916
           window ewma125        6      1.1719       1.2754      1.5702            0.1083             0.1109            0.1139         0.1874          0.1888         0.1909
           window ewma250        6      1.0218       1.1888      1.5132            0.1081             0.1106            0.1139         0.1875          0.1889         0.1909
           window ewma500        6      0.8467       1.1814      1.4372            0.1079             0.1102            0.1132         0.1875          0.1886         0.1901
weight exponent p       0       12      1.0218       1.1409      1.3650            0.1132             0.1139            0.1151         0.1901          0.1909         0.1916
weight exponent p       1       12      0.8467       1.2133      1.3620            0.1085             0.1099            0.1116         0.1874          0.1880         0.1886
weight exponent p       2       12      1.1237       1.3597      1.6318            0.1076             0.1087            0.1101         0.1877          0.1880         0.1885
    variance term    FLAT       18      1.0218       1.2838      1.6318            0.1083             0.1112            0.1151         0.1877          0.1891         0.1916
    variance term PER-ROW       18      0.8467       1.1921      1.4332            0.1076             0.1105            0.1151         0.1874          0.1888         0.1916
```

```
  window             spread of level means:  Sharpe 0.1223   QLIKE(traded) 0.00169   QLIKE(all) 0.00079
  weight exponent p  spread of level means:  Sharpe 0.2188   QLIKE(traded) 0.00520   QLIKE(all) 0.00293
  variance term      spread of level means:  Sharpe 0.0918   QLIKE(traded) 0.00072   QLIKE(all) 0.00030
```

One dimension separates and two do not. The weight exponent moves the level-mean
QLIKE by 0.00293 on all rows and 0.00520 on traded days. The window moves it by
0.00079 and 0.00169. The variance term moves it by 0.00030 and 0.00072. The
exponent's effect is between three and ten times the other two, and the p = 0
level does not overlap the other two on QLIKE at all: its range is 0.1901 to
0.1916 on all rows, above the 0.1886 that is the worst p = 1 cell and the 0.1885
that is the worst p = 2 cell.

The Sharpe column shows the same ordering with much less separation. The exponent
moves the level-mean Sharpe by 0.2188, the window by 0.1223, the variance term by
0.0918. All three are far below the 0.5476 standard error of a single cell. The
within-level ranges overlap almost completely.

## 7. Where the current cell stands

```
     tag  rank_qlike_all_rows  rank_qlike_traded  rank_sharpe  sharpe_min  sharpe_max  sharpe_current  qlike_all_min  qlike_all_max  qlike_all_current
      a0                    6                  9           25      0.6417      1.1962          0.9673         0.1968         0.2043             0.1979
    blk2                   17                 18           12      0.8467      1.6318          1.3383         0.1874         0.1916             0.1882
blk2_inc                   15                 20           27      0.9146      1.5886          1.2707         0.1962         0.2000             0.1969
    lgbm                   14                 12            2      1.0046      1.4983          1.4594         0.1906         0.1965             0.1911
     xgb                   12                  8            2      0.9043      1.6257          1.4980         0.1937         0.1992             0.1942
 lasso_t                   16                 15            6      0.8803      1.7383          1.4131         0.1960         0.2001             0.1967
 lasso_f                   16                 18            3      0.9462      1.6924          1.5098         0.1876         0.1916             0.1883
    enet                   17                 12           13      0.6715      1.2082          1.0225         0.1956         0.1996             0.1962
```

Rank 1 is the lowest QLIKE or the highest Sharpe; ties take the lower rank.

## 8. Figure

`sharpe_vs_qlike_blk2.png` plots the sign(s) Sharpe against the QLIKE on the 866
traded rows, one point per cell, for the block-diagonal ridge, with the current
cell marked by a star. Colour is the weight exponent, shape is the window family,
and an open face is the PER-ROW variance term. The p = 0 pairs plot on top of
each other. The picture is a cloud running from the p = 0 cells at high loss
toward the p = 2 cells at low loss, with a vertical spread of several tenths of a
Sharpe at every loss level. The current cell sits inside the cloud.

## 9. Verdict

The grid does not overturn the map the deck uses, and it does not endorse it. On
QLIKE over all fit-mask rows the current cell ranks 17th of 36 for the
block-diagonal ridge, and between 6th and 17th across the eight forecasts; on
QLIKE over the 866 traded rows it ranks 18th for the block-diagonal ridge, and
between 8th and 20th across the eight. On the sign(s) Sharpe it ranks 12th of 36
for the block-diagonal ridge, 2nd for LightGBM and XGBoost, 3rd for the fixed
lasso, 6th for the causally tuned lasso, 13th for the elastic net, 25th for the
baseline (HAR + calendar OLS) and 27th for the ridge without the FOMC columns.
The forecast-loss spread is real and points mostly one way: all 96 p = 0
comparisons lose more than the current map, at t from 1.76 to 5.01 on all rows,
and of the 280 comparisons not one beats it at |t| > 2 on all fit-mask rows,
where the largest improvement is t = -1.97. Fourteen do beat it at |t| > 2 on the
866 traded rows, at up to t = -2.91, and they are a coherent set rather than a
scatter: p = 2 FLAT cells on the longer and smoother windows, on the two ridges,
the two lassos and the elastic net only, and the same cells do not beat it on the
full row set. The Sharpe spread is not real in the same sense: the current cell's
own Sharpe carries a bootstrap standard error of 0.5476, the 36 cells span 0.8467
to 1.6318 for the block-diagonal ridge, and the best of 36 exceeds the average of
36 by 0.4699 on average under resampling against an observed gap of 0.3939, so
the observed maximum sits at the 46.9th percentile of the maximum's own
distribution. Fifteen of the 36 cells take the same side as the current map on
more than 95% of the 866 days, and the least similar cell still agrees on 92.84%.
What the numbers support is narrow and worth stating plainly: the weight exponent
is the one convention of the three that moves the forecast loss, the window
length and the variance term move it barely, and none of the three moves the
portfolio by more than the noise in a single cell.

## 10. Files

All under `results/atm_straddle_0dte_1530/recal_grid/`.

| File | Contents |
| --- | --- |
| `gate1.csv` | gate 1 per forecast: days matched, max relative error, pass |
| `qlike_all_rows.csv` | 36 x 8 QLIKE, all fit-mask rows in the scored period |
| `qlike_traded_days.csv` | 36 x 8 QLIKE, the 866 traded 16:00 rows |
| `dm_t_all_rows.csv` | 36 x 8 paired t against the current cell, all rows |
| `dm_t_traded_days.csv` | 36 x 8 paired t against the current cell, traded days |
| `calibration_slope.csv` | 18 x 8 slope of y on m over the scored rows |
| `calibration_intercept.csv` | 18 x 8 intercept of y on m over the scored rows |
| `sharpe.csv` | 36 x 8 sign(s) Sharpe on the 866 days |
| `n_buy.csv` | 36 x 8 days the rule buys the straddle |
| `n_sign_diff_vs_current.csv` | 36 x 8 days whose side differs from the current map |
| `cells_detail.csv` | one row per cell and forecast: every quantity above |
| `bootstrap_blk2.csv` | the multiplicity numbers |
| `sign_agreement_blk2.csv` | per-cell agreement with the current map |
| `sensitivity_blk2.csv` | the per-dimension ranges |
| `current_cell_rank.csv` | the rank table of section 7 |
| `sharpe_vs_qlike_blk2.png` | the figure |
