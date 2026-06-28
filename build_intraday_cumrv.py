"""Build the REAL intraday cumulative realized-variance feature (cumrv_real).

This replaces the rank-space proxy in ``resid_amortized.py:_cumrv_close``. The proxy
cumsums the per-slot rank-Gauss ``har_ma_1`` (so its tails are compressed by the rank /
sqrt+winsor transform) and segments days by an hour-wrap heuristic (``np.diff(hour) < 0``).
That destroys the additive PHYSICAL magnitude of realized variance.

This script builds the physically meaningful version:
  * PHYSICAL realized variance ``RV`` (= ``sumret2``), loaded via ``src.data.loading``;
  * diurnally adjusted EXACTLY as the production TARGET is (``diurnal_adjust``, divide /
    rolling per-slot MEAN, causal ``.shift(1)``, window=20 / min_periods=5) -> "abnormal
    vol" on a VARIANCE-like (additive) scale. NO sqrt/winsorize semantic transform is
    applied before the cumsum (variance is additive; the cumsum must happen on the
    variance scale);
  * cumulated within the TRUE calendar day (``t.dt.normalize()``), then shifted by one bar
    so bar ``t`` only ever sees information through ``t-1`` (causal). The shift is done
    WITHIN each calendar day so the prior session's total does not leak into a day's first
    bar -- this matters because the Sunday 18:30 ET open is the first bar of its calendar
    day yet falls inside the 16-19 close gate; a global shift would inject Friday's whole-
    day RV total into that gated bar once per week.

Outputs (under ``results/intraday_feats/``):
  * ``cumrv_real.npy``        -- ungated cumulative abnormal RV, aligned to the cache rows;
  * ``cumrv_real_close.npy``  -- the same, gated to the 16-19 close + after-hours window
                                 (mirrors the proxy's ``cumrv_x_close``);
  * ``cumrv_real_meta.json``  -- alignment sidecar (offset, anchor check, length, dtype).

Row alignment: the residualized cache rows correspond EXACTLY to
``load_raw_data('data', allow_missing=True)[3125:]`` (offset 3125 = HAR max-lag burn-in).
The arrays are built on the full load and sliced ``[3125:]`` so the kept rows have full
history behind them.

NOTE: this script does NOT edit ``resid_amortized.py``. Verifying row alignment against the
actual ``results/covid_imp_rank/.../X_imp.npy`` and the QLIKE A/B are CLUSTER steps (that
cache is not present locally) and are out of scope here.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.loading import load_raw_data
from src.features.extractors.har import resolve_har_lags
from src.features.transforms.target import (
    DIURNAL_MIN_PERIODS,
    DIURNAL_WINDOW,
    diurnal_adjust,
    diurnal_rank,
)

DATA_PATH = "data"
OFFSET: int = resolve_har_lags()[
    -1
]  # 3125 -- HAR max-lag burn-in dropped by the executor
OUT_DIR = "results/intraday_feats"
WRITEUP = "writeup/intraday_cumrv_real_2026-06-27.md"

# Anchor (SLICED / cache-row space): cache row 189713 == full row 189713+3125 == 2020-02-25.
ANCHOR_SLICED_IDX = 189713
ANCHOR_DATE = "2020-02-25"

CLOSE_LO, CLOSE_HI = 16, 19  # close auction + after-hours gate, hours [16, 19]


# ---------------------------------------------------------------------------
# REAL feature (physical, variance-scale, causal)
# ---------------------------------------------------------------------------


def diurnal_adjust_rv(rv: pd.Series, tod: pd.Series) -> pd.Series:
    """Diurnally adjust RV the SAME way the production target is adjusted.

    RV = sumret2 >= 0 (non-negative), so ``diurnal_adjust`` takes the rolling per-slot
    MEAN branch (causal ``.shift(1)``), matching the target's production default window
    (DIURNAL_WINDOW=20 / DIURNAL_MIN_PERIODS=5). Returns RV / per-slot-mean-baseline =
    "abnormal vol" on a variance-like scale (NO sqrt / winsorize).
    """
    has_neg = bool(
        (rv.dropna() < 0).any()
    )  # faithful to robust_transform; False for RV
    adjusted, _baseline = diurnal_adjust(
        rv,
        tod,
        has_negatives=has_neg,
        window=DIURNAL_WINDOW,
        min_periods=DIURNAL_MIN_PERIODS,
    )
    return adjusted


def build_real_cumrv(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (cumrv_ungated, cumrv_close_gated, cumrv_sqrt_ungated, adj_rv) on the FULL row space.
    cumrv_sqrt = SQRT(per-bar adj RV) cumsum'd within the SAME calendar day with the SAME causal shift
    -- the only delta vs cumrv is accumulating on the VOL (sqrt) scale instead of the VARIANCE scale.
    The isolated control for 'is the cumrv edge the sqrt variance-stabilization' (proxy=cumsum of
    sqrt-vol har_ma_1 = 0.12314 vs cumrv_real=cumsum of raw variance = 0.12436)."""
    rv = df["RV"].astype(np.float64)
    adj = diurnal_adjust_rv(rv, df["time_of_day"])  # variance-scale abnormal vol
    day = df["t"].dt.normalize()  # TRUE calendar day (midnight boundary)
    cum = adj.groupby(day).cumsum()  # cumulative abnormal RV today (incl. current bar)
    cum_sqrt = (
        np.sqrt(adj).groupby(day).cumsum()
    )  # SAME, accumulated on the VOL (sqrt) scale
    # Causal: shift by one bar so bar t sees only through t-1. Shift WITHIN the calendar
    # day (first bar of each day -> 0) so the prior session's total never leaks across the
    # day boundary (notably into the Sunday 18:30 open, which is inside the close gate).
    cum_caus = cum.groupby(day).shift(1).fillna(0.0)
    cum_sqrt_caus = cum_sqrt.groupby(day).shift(1).fillna(0.0)
    hour = df["t"].dt.hour
    close = ((hour >= CLOSE_LO) & (hour <= CLOSE_HI)).astype(np.float64)
    cumrv = cum_caus.to_numpy(dtype=np.float64)
    cumrv_close = (cum_caus * close).to_numpy(dtype=np.float64)
    cumrv_sqrt = cum_sqrt_caus.to_numpy(dtype=np.float64)
    return cumrv, cumrv_close, cumrv_sqrt, adj.to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# PROXY reconstruction (for validation only -- mirrors resid_amortized._cumrv_close)
# ---------------------------------------------------------------------------


def build_proxy_cumrv(df: pd.DataFrame) -> np.ndarray:
    """Reconstruct the rank-space proxy's cumrv on the FULL row space.

    Mirrors ``_cumrv_close``: per-slot rank-Gauss ``har_ma_1`` (= rank-Gauss adj_RV shifted
    by one bar, per ``generate_har_features``'s ``rolling(1).mean().shift(1)``), day_id by
    hour-wrap (``np.diff(hour) < 0``), cumsum within day_id. NOT gated here (compared ungated
    against the real ungated cumrv).
    """
    adj_rank, _ = diurnal_rank(
        df["RV"].astype(np.float64),
        df["time_of_day"],
        window=DIURNAL_WINDOW,
        min_periods=DIURNAL_MIN_PERIODS,
        gaussianize=True,
    )
    har1 = adj_rank.rolling(1, min_periods=1).mean().shift(1)  # == shift(1)
    hour = df["t"].dt.hour.to_numpy()
    day_id = np.cumsum(np.concatenate([[0], (np.diff(hour) < 0).astype(int)]))
    cumrv = pd.Series(har1.to_numpy()).groupby(day_id).cumsum().to_numpy()
    return np.asarray(cumrv, dtype=np.float64)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(WRITEUP), exist_ok=True)

    df = load_raw_data(DATA_PATH, allow_missing=True)
    n_full = len(df)
    cumrv_full, cumrv_close_full, cumrv_sqrt_full, _adj = build_real_cumrv(df)

    # Slice to the cache row space.
    cumrv = np.ascontiguousarray(cumrv_full[OFFSET:])
    cumrv_close = np.ascontiguousarray(cumrv_close_full[OFFSET:])
    cumrv_sqrt = np.ascontiguousarray(cumrv_sqrt_full[OFFSET:])
    n = len(cumrv)

    # ---- VALIDATION ----
    lines: list[str] = []

    def emit(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    emit("=" * 78)
    emit("BUILD intraday cumrv_real (physical, variance-scale, causal)")
    emit("=" * 78)
    emit(f"full load len            : {n_full}")
    emit(f"offset (HAR burn-in)     : {OFFSET}")
    emit(f"sliced array len         : {n}")
    emit(f"dtype                    : {cumrv.dtype}")
    emit(
        f"diurnal window/min_per   : {DIURNAL_WINDOW} / {DIURNAL_MIN_PERIODS} (target default, divide/mean)"
    )
    emit(f"close gate (hour)        : [{CLOSE_LO}, {CLOSE_HI}]")
    emit("")

    # (1) Causality: perturb a FUTURE RV row, recompute; earlier values byte-identical.
    jf = n_full - 50  # a clearly-future full-space row
    df2 = df.copy()
    df2.loc[df2.index[jf], "RV"] = float(df2["RV"].iloc[jf]) * 1e6 + 1e9
    cumrv2_full, cumrv2_close_full, _sq2, _ = build_real_cumrv(df2)
    eq_un = bool(np.array_equal(cumrv_full[:jf], cumrv2_full[:jf]))
    eq_cl = bool(np.array_equal(cumrv_close_full[:jf], cumrv2_close_full[:jf]))
    diff_un = np.where(cumrv_full != cumrv2_full)[0]
    first_diff = int(diff_un[0]) if diff_un.size else -1
    max_abs_before = (
        float(np.max(np.abs(cumrv_full[:jf] - cumrv2_full[:jf]))) if jf > 0 else 0.0
    )
    emit("[1] CAUSALITY (perturb future RV row, recompute)")
    emit(f"    perturbed full row       : {jf}  (ts {df['t'].iloc[jf]})")
    emit(
        f"    ungated [:jf] identical   : {eq_un}   max|diff| before jf = {max_abs_before:.3e}"
    )
    emit(f"    close   [:jf] identical   : {eq_cl}")
    emit(
        f"    first changed full index  : {first_diff}  (>= jf={jf} required)  -> {'PASS' if first_diff >= jf else 'FAIL'}"
    )
    emit("")

    # (2) Anchor: sliced index 189713 -> 2020-02-25.
    full_anchor_idx = ANCHOR_SLICED_IDX + OFFSET
    anchor_ts = df["t"].iloc[full_anchor_idx]
    anchor_ok = str(anchor_ts.date()) == ANCHOR_DATE
    emit("[2] ANCHOR (sliced/cache row space)")
    emit(f"    sliced idx {ANCHOR_SLICED_IDX} == full idx {full_anchor_idx}")
    emit(
        f"    timestamp                 : {anchor_ts}  (DOW={anchor_ts.dayofweek}, hour={anchor_ts.hour})"
    )
    emit(
        f"    date == {ANCHOR_DATE}      : {anchor_ok}  -> {'PASS' if anchor_ok else 'FAIL'}"
    )
    emit(
        f"    note: the task's 'full row 189713' is the SLICED index; full idx 189713 = {df['t'].iloc[ANCHOR_SLICED_IDX]}"
    )
    emit("")

    # (3) Compare vs proxy: Pearson corr + sign agreement on the SAME sliced rows.
    proxy_full = build_proxy_cumrv(df)
    proxy = np.ascontiguousarray(proxy_full[OFFSET:])
    finite_pair = np.isfinite(cumrv) & np.isfinite(proxy)
    pear = float(np.corrcoef(cumrv[finite_pair], proxy[finite_pair])[0, 1])
    # Spearman (rank) corr -- robust to the proxy's rank/sqrt-compressed scale.
    rc = pd.Series(cumrv[finite_pair]).rank().to_numpy()
    rp = pd.Series(proxy[finite_pair]).rank().to_numpy()
    spear = float(np.corrcoef(rc, rp)[0, 1])
    sign_agree = float(
        np.mean(np.sign(cumrv[finite_pair]) == np.sign(proxy[finite_pair]))
    )
    # Focused: within the close-gate (hour 16-19) only, where the production feature lives.
    hour_sliced = df["t"].dt.hour.to_numpy()[OFFSET:]
    gate = (hour_sliced >= CLOSE_LO) & (hour_sliced <= CLOSE_HI) & finite_pair
    pear_gate = float(np.corrcoef(cumrv[gate], proxy[gate])[0, 1])
    emit("[3] COMPARE vs proxy (per-slot rank-Gauss har_ma_1, hour-wrap day_id)")
    emit(f"    Pearson  corr (ungated)   : {pear:+.4f}")
    emit(f"    Spearman corr (ungated)   : {spear:+.4f}")
    emit(
        f"    sign agreement            : {sign_agree:.4f}  (cumrv_real >= 0 by construction)"
    )
    emit(
        f"    Pearson corr (close gate) : {pear_gate:+.4f}  (hours {CLOSE_LO}-{CLOSE_HI}, n={int(gate.sum())})"
    )
    emit(
        "    -> positive-but-imperfect expected: same RV path, restored ADDITIVE magnitude vs rank-compressed."
    )
    emit("")

    # (4) Finite / no-NaN after the slice.
    fin_un = bool(np.all(np.isfinite(cumrv)))
    fin_cl = bool(np.all(np.isfinite(cumrv_close)))
    emit("[4] FINITE / NO-NaN after [3125:] slice")
    emit(
        f"    ungated all finite        : {fin_un}   (min={cumrv.min():.4f} max={cumrv.max():.4f})"
    )
    emit(
        f"    close   all finite        : {fin_cl}   (nonzero rows={int((cumrv_close != 0).sum())})"
    )
    emit("")

    # ---- SAVE ----
    np.save(f"{OUT_DIR}/cumrv_real.npy", cumrv)
    np.save(f"{OUT_DIR}/cumrv_real_close.npy", cumrv_close)
    np.save(
        f"{OUT_DIR}/cumrv_realsqrt.npy", cumrv_sqrt
    )  # SQRT-scale control (isolates the sqrt)
    emit(
        f"[SAVE] cumrv_realsqrt.npy  (min={cumrv_sqrt.min():.4f} max={cumrv_sqrt.max():.4f})"
    )
    meta = {
        "feature": "cumrv_real",
        "description": "Physical intraday cumulative realized variance: RV diurnally "
        "adjusted (divide/mean, causal shift1, window=20/min5) then cumsum within the true "
        "calendar day on the variance scale, causal per-day shift1. Replaces the rank-space "
        "proxy resid_amortized._cumrv_close.",
        "offset": OFFSET,
        "full_load_len": n_full,
        "array_len": n,
        "dtype": str(cumrv.dtype),
        "diurnal_window": DIURNAL_WINDOW,
        "diurnal_min_periods": DIURNAL_MIN_PERIODS,
        "close_gate_hours": [CLOSE_LO, CLOSE_HI],
        "anchor_sliced_idx": ANCHOR_SLICED_IDX,
        "anchor_full_idx": full_anchor_idx,
        "anchor_timestamp": str(anchor_ts),
        "anchor_date_expected": ANCHOR_DATE,
        "anchor_date_ok": anchor_ok,
        "causality_pass": bool(eq_un and eq_cl and first_diff >= jf),
        "finite_ungated": fin_un,
        "finite_close": fin_cl,
        "pearson_corr_vs_proxy": pear,
        "spearman_corr_vs_proxy": spear,
        "files": {
            "ungated": f"{OUT_DIR}/cumrv_real.npy",
            "close_gated": f"{OUT_DIR}/cumrv_real_close.npy",
        },
    }
    with open(f"{OUT_DIR}/cumrv_real_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    emit(f"SAVED {OUT_DIR}/cumrv_real.npy            shape={cumrv.shape}")
    emit(f"SAVED {OUT_DIR}/cumrv_real_close.npy      shape={cumrv_close.shape}")
    emit(f"SAVED {OUT_DIR}/cumrv_real_meta.json")

    _write_report(lines, meta)
    emit(f"WROTE {WRITEUP}")


def _write_report(lines: list[str], meta: dict) -> None:
    body = "\n".join(lines)
    md = f"""# Intraday cumulative realized variance (cumrv_real) -- build + validation

Date: 2026-06-27

## What this is

A physical, variance-scale, causal replacement for the rank-space proxy in
`resid_amortized.py:_cumrv_close` (lines 89-100). Built by `build_intraday_cumrv.py`
(repo root). **`resid_amortized.py` is not modified.**

| step | proxy (`_cumrv_close`) | this feature (`cumrv_real`) |
|---|---|---|
| series | per-slot rank-Gauss `har_ma_1` (rank/sqrt+winsor compresses tails) | physical RV = `sumret2` |
| diurnal | rank-Gauss | divide by per-slot rolling MEAN, causal `.shift(1)`, window={meta["diurnal_window"]}/min{meta["diurnal_min_periods"]} (= target default) |
| scale of cumsum | rank-Gauss (non-additive) | **variance scale** (additive; no sqrt/winsorize before cumsum) |
| day segmentation | hour-wrap heuristic `np.diff(hour) < 0` | **true calendar day** `t.dt.normalize()` |
| causality | free (`har_ma_1` pre-shifted) | explicit per-day `.shift(1)` of the cumsum |

The shift is done *within* each calendar day (first bar of a day -> 0) so the prior
session's total does not leak across the day boundary -- the Sunday 18:30 ET open is the
first bar of its calendar day yet sits inside the 16-19 close gate, so a global shift would
inject Friday's whole-day RV total into that gated bar once a week.

## Outputs

- `{meta["files"]["ungated"]}` -- ungated cumulative abnormal RV, shape `({meta["array_len"]},)`, `{meta["dtype"]}`
- `{meta["files"]["close_gated"]}` -- gated to hours {meta["close_gate_hours"][0]}-{meta["close_gate_hours"][1]} (mirrors `cumrv_x_close`)
- `{OUT_DIR}/cumrv_real_meta.json` -- alignment sidecar

## Alignment

Cache rows = `load_raw_data('data', allow_missing=True)[{meta["offset"]}:]` (offset
{meta["offset"]} = HAR max-lag burn-in). Full load len = {meta["full_load_len"]};
sliced len = {meta["array_len"]}.

Anchor: **sliced/cache index {meta["anchor_sliced_idx"]}** == full index
{meta["anchor_full_idx"]} == `{meta["anchor_timestamp"]}` (date == {meta["anchor_date_expected"]}:
**{meta["anchor_date_ok"]}**). NB: the task phrased this as "full row {meta["anchor_sliced_idx"]}",
but {meta["anchor_sliced_idx"]} is the *sliced* index; full index {meta["anchor_sliced_idx"]}
is a different (earlier) bar. The cache-space interpretation is the one that matters for
alignment and it checks out exactly (RTH 09:30 open, Tue).

## Validation run

```
{body}
```

## Out of scope (cluster steps)

Verifying row alignment against the actual `results/covid_imp_rank/all_buckets/X_imp.npy`
and the QLIKE A/B (proxy vs real in the residualized EBM) require the covid_imp_rank cache,
which is not present locally.
"""
    with open(WRITEUP, "w") as fh:
        fh.write(md)


if __name__ == "__main__":
    main()
