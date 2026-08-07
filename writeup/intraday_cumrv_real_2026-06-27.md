# Intraday cumulative realized variance (cumrv_real) -- build + validation

Date: 2026-06-27

## What this is

A physical, variance-scale, causal replacement for the rank-space proxy in
`resid_amortized.py:_cumrv_close` (lines 89-100). Built by `build_intraday_cumrv.py`
(repo root). **`resid_amortized.py` is not modified.**

| step | proxy (`_cumrv_close`) | this feature (`cumrv_real`) |
|---|---|---|
| series | per-slot rank-Gauss `har_ma_1` (rank/sqrt+winsor compresses tails) | physical RV = `sumret2` |
| diurnal | rank-Gauss | divide by per-slot rolling MEAN, causal `.shift(1)`, window=20/min5 (= target default) |
| scale of cumsum | rank-Gauss (non-additive) | **variance scale** (additive; no sqrt/winsorize before cumsum) |
| day segmentation | hour-wrap heuristic `np.diff(hour) < 0` | **true calendar day** `t.dt.normalize()` |
| causality | free (`har_ma_1` pre-shifted) | explicit per-day `.shift(1)` of the cumsum |

The shift is done *within* each calendar day (first bar of a day -> 0) so the prior
session's total does not leak across the day boundary -- the Sunday 18:30 ET open is the
first bar of its calendar day yet sits inside the 16-19 close gate, so a global shift would
inject Friday's whole-day RV total into that gated bar once a week.

## Outputs

- `results/intraday_feats/cumrv_real.npy` -- ungated cumulative abnormal RV, shape `(242934,)`, `float64`
- `results/intraday_feats/cumrv_real_close.npy` -- gated to hours 16-19 (mirrors `cumrv_x_close`)
- `results/intraday_feats/cumrv_real_meta.json` -- alignment sidecar

## Alignment

Cache rows = `load_raw_data('data', allow_missing=True)[3125:]` (offset
3125 = HAR max-lag burn-in). Full load len = 246059;
sliced len = 242934.

Anchor: **sliced/cache index 189713** == full index
192838 == `2020-02-25 09:30:00` (date == 2020-02-25:
**True**). NB: the task phrased this as "full row 189713",
but 189713 is the *sliced* index; full index 189713
is a different (earlier) bar. The cache-space interpretation is the one that matters for
alignment and it checks out exactly (RTH 09:30 open, Tue).

## Validation run

```
==============================================================================
BUILD intraday cumrv_real (physical, variance-scale, causal)
==============================================================================
full load len            : 246059
offset (HAR burn-in)     : 3125
sliced array len         : 242934
dtype                    : float64
diurnal window/min_per   : 20 / 5 (target default, divide/mean)
close gate (hour)        : [16, 19]

[1] CAUSALITY (perturb future RV row, recompute)
    perturbed full row       : 246009  (ts 2024-04-29 23:00:00)
    ungated [:jf] identical   : True   max|diff| before jf = 0.000e+00
    close   [:jf] identical   : True
    first changed full index  : 246010  (>= jf=246009 required)  -> PASS

[2] ANCHOR (sliced/cache row space)
    sliced idx 189713 == full idx 192838
    timestamp                 : 2020-02-25 09:30:00  (DOW=1, hour=9)
    date == 2020-02-25      : True  -> PASS
    note: the task's 'full row 189713' is the SLICED index; full idx 189713 = 2019-11-27 09:00:00

[3] COMPARE vs proxy (per-slot rank-Gauss har_ma_1, hour-wrap day_id)
    Pearson  corr (ungated)   : +0.4742
    Spearman corr (ungated)   : +0.4293
    sign agreement            : 0.4517  (cumrv_real >= 0 by construction)
    Pearson corr (close gate) : +0.5498  (hours 16-19, n=42812)
    -> positive-but-imperfect expected: same RV path, restored ADDITIVE magnitude vs rank-compressed.

[4] FINITE / NO-NaN after [3125:] slice
    ungated all finite        : True   (min=0.0000 max=1772.0336)
    close   all finite        : True   (nonzero rows=41800)

SAVED results/intraday_feats/cumrv_real.npy            shape=(242934,)
SAVED results/intraday_feats/cumrv_real_close.npy      shape=(242934,)
SAVED results/intraday_feats/cumrv_real_meta.json
```

## Out of scope (cluster steps)

Verifying row alignment against the actual `results/covid_imp_rank/all_buckets/X_imp.npy`
and the QLIKE A/B (proxy vs real in the residualized EBM) require the covid_imp_rank cache,
which is not present locally.
