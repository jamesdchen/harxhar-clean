"""Shared executor scaffold for ML walk-forward volatility backtests.

Lifts the ~90% duplicated `main()` body of the per-method executors
(`src/models/xgboost.py`, `src/models/lightgbm.py`, `src/models/random_forest.py`,
`src/models/ridge.py`) into one module. Per-method scripts supply only a
`fit_predict(X_chunk, y_chunk, train_win_periods, hyperparams)`
callable plus a default-hyperparam dict; everything else — CLI parsing,
loading, transforms, horizon shift, chunk slicing, smearing, reduce
JSON — is shared.

The CLI contract is now owned by ``.hpc/cli.py`` (the auto-generated
dispatcher) and ``.hpc/tasks.py`` FLAGS dict — the per-executor flag
list lives there, not in this module. ``run_executor`` and the
backtest plumbing remain here. The flags expected on ``args`` match
those declared in ``.hpc/tasks.py`` FLAGS for the calling executor's
module key (e.g. ``FLAGS["src/backtest/tune_tree.py`` (cmd_evaluate) passes via subprocess:
``--params-file --output-file --start --end --data-path --train-window``
``--horizon --exog-cols`` plus the Ridge-only ``--segment`` and
``--lag-scope`` and the optional ``--refit-frequency`` / ``--seed``.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import re
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.diagnostics import check_and_report
from src.evaluation.metrics import apply_duan_smearing, save_chunk_reduce
from src.data.loading import apply_overnight_fills, load_raw_data
from src.features.transforms.scaling import (
    masked_rolling_scale_col,
    rolling_robust_scale,
)
from src.backtest.segmentation import (
    SEGMENT_DEFINITIONS,
    compute_segment_train_window,
    slice_to_segment,
)
from src.features.extractors.calendar import add_calendar_features
from src.features.extractors.expiry import add_expiry_features
from src.features.extractors.har import generate_har_features, resolve_har_lags
from src.features.transforms.target import (
    PERIODS_PER_DAY,
    apply_horizon_shift,
    robust_transform,
)

FitPredict = Callable[[np.ndarray, np.ndarray, int, dict], np.ndarray]

# A feature with at least this fraction of its mass at a single *exact* value is treated as
# mode-inflated (a point mass + a continuous part) and gets the hurdle encoding: an occurrence
# indicator plus a magnitude scaled over its off-mode (active) values. Continuous features
# (returns, vols) never reach this.
#
# The test used to be on exact *zeros* specifically, which is the right test for the flow features
# it was written for (``voldemand``, ~60% zeros) but misses the same shape at a different location:
# ``numobs`` is capped at 30 observations per bar and *is* 30 on 78% of bars, so its raw IQR is
# exactly 0 and its adjusted IQR is 0.029 — which turns an ordinary holiday half-bar into
# ``adj_numobs_ma_25`` at |z| 79, with 986 rows past |z| 20. Keying on the modal value instead of on
# zero subsumes the old behaviour exactly (``voldemand``'s mode *is* 0) and, across the 43 raw exog,
# catches exactly one new column: ``numobs`` at 0.78. The next-highest modal share is
# ``stocktwits_sentiment`` at 0.12, comfortably below the threshold.
ZERO_INFLATED_FRAC: float = 0.2

# How many bars a forward fill may carry an exogenous value before the row is
# treated as unobserved again. 26 bars is two trading sessions at this panel's
# 30-minute frequency: long enough to bridge a missed print or a holiday edge,
# short enough that a stopped feed cannot masquerade as live data for months.
# ``None`` restores the unlimited fill.
FFILL_LIMIT: int | None = 26


def modal_value(vals: np.ndarray) -> float:
    """The single most common exact value of ``vals`` (NaNs dropped), or NaN if empty."""
    finite = vals[~np.isnan(vals)]
    if not finite.size:
        return float("nan")
    uniq, counts = np.unique(finite, return_counts=True)
    return float(uniq[int(np.argmax(counts))])


def mode_inflated(vals: np.ndarray) -> tuple[bool, float]:
    """``(is_mode_inflated, mode)`` under :data:`ZERO_INFLATED_FRAC`."""
    finite = vals[~np.isnan(vals)]
    if not finite.size:
        return False, float("nan")
    mode = modal_value(vals)
    return bool(float(np.mean(finite == mode)) >= ZERO_INFLATED_FRAC), mode


def _expanding_real_iqr(
    x: np.ndarray, mask: np.ndarray, grid_step: int = 480, warmup_floor: float = 1.0
) -> np.ndarray:
    """Causal expanding IQR of ``x`` over its *available* (``mask``) values.

    A slowly-varying reference scale, so it is computed on a coarse grid and
    forward-filled rather than re-evaluated every row. Excluding the imputed
    (unavailable) rows is the crux: the median-fill injects a flat constant
    that otherwise dominates — and collapses — the window IQR. Causal: row
    ``t`` only ever sees real values strictly before ``t``.

    ``warmup_floor`` is what the reference is worth *before* there is enough real history to
    estimate it (fewer than 8 real values on the grid). It used to be 0, i.e. "impose no floor" —
    and 0 is the worst possible answer, because those are exactly the rows where the trailing window
    is all imputed constant and the local IQR is therefore ~0 too. ``adj_stocktwits_sentiment_ma_*``
    reaches |z| 70 on 2010-06-02, one day into its coverage, with ``ref_iqr`` recorded as exactly
    0.0. An unmeasurable scale must not become an unbounded divisor: 1.0 leaves an adjusted feature
    (a log or a ratio, O(1) by construction) at its natural scale until its own dispersion is
    estimable, and the availability indicator already tells the model the feature is not live.
    """
    n = len(x)
    ref = np.full(n, float(warmup_floor), dtype=np.float64)
    real_idx = np.flatnonzero(mask)
    last = float(warmup_floor)
    for t in range(0, n, grid_step):
        past = real_idx[real_idx < t]
        if past.size >= 8:
            q25, q75 = np.percentile(x[past], (25.0, 75.0))
            iq = float(q75 - q25)
            if iq >= 1e-12:
                last = iq
        ref[t : t + grid_step] = last
    return ref


def _build_scale_guards(
    X: np.ndarray,
    df: pd.DataFrame,
    feature_names: list[str],
    zero_inflated_active: bool = True,
    erode_ma_window: bool = True,
    warmup_floor: float = 1.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Scale guards for the impute-and-indicate feature pathology.

    Returns ``(ref_iqr, fixed_cols)`` for :func:`rolling_robust_scale`, or
    ``(None, None)`` when no guarding is needed (no availability columns →
    not an ``impute_indicate`` run → scaling is bit-for-bit unchanged).

    * ``*_avail_ma_*`` / ``*_active_ma_*`` (availability + occurrence-indicator
      rolling means) → *fixed*: passed through raw; scale is known ([0, 1]).
    * ``adj_<col>_ma_*`` whose ``<col>_avail`` column exists (the exog was
      imputed) → IQR floored at the causal expanding IQR over the *real*
      (available) values, with the availability mask **eroded by the MA window**
      so a half-imputed long average does not set the scale; for a
      *mode-inflated* magnitude the scale is taken over off-mode (active)
      values too, so neither a regime-degenerate window nor a point mass can
      collapse the scale and blow up the forecast.
    * everything else (HAR, calendar) → untouched.
    """
    n, n_features = X.shape
    ref_iqr = np.zeros((n, n_features), dtype=np.float64)
    fixed = np.zeros(n_features, dtype=bool)
    touched = False
    for j, name in enumerate(feature_names):
        if "_avail_ma_" in name or "_active_ma_" in name:
            fixed[j] = True
            touched = True
            continue
        m = re.match(r"adj_(.+)_ma_(\d+)$", name)
        if m is None:
            continue
        avail_col = f"{m.group(1)}_avail"
        if avail_col in df.columns:
            mask = df[avail_col].to_numpy().astype(bool)[:n]
            # Erode the availability mask by the MA window. ``adj_<col>_ma_3125`` is a ~65-day
            # average, so for 3125 bars after the raw column first appears the MA is still mostly
            # the imputed median — a seam, not data. Taking the reference IQR over rows the raw
            # column merely *exists* on therefore measures the scale of a half-imputed average.
            # This is the whole of the ``adj_vix3m_ma_3125`` (|z| 50, 2009-11-06, three months into
            # vix3m coverage) and ``adj_vvix_ma_3125`` (|z| 30, 2012-06-19, three months into vvix
            # coverage) failures: only the long windows fail, and only near the seam.
            w = int(m.group(2))
            if erode_ma_window and w > 1:
                full = (
                    pd.Series(mask.astype(np.float64))
                    .rolling(w, min_periods=w)
                    .min()
                    .to_numpy()
                )
                eroded = np.nan_to_num(full) > 0.5
                if eroded.sum() >= 8:  # keep the raw mask if erosion leaves nothing to fit
                    mask = eroded
            xj = X[:, j]
            # Zero-inflated magnitude (e.g. voldemand): also estimate the scale
            # over ACTIVE (non-zero) values — the zero point-mass carries no
            # dispersion and collapses the IQR otherwise. The indicator half of
            # the hurdle encoding is the ``<col>_active`` columns added in
            # ``load_and_transform``.
            inflated, mode = mode_inflated(xj)
            if zero_inflated_active and inflated:
                mask = mask & (xj != mode)
            ref_iqr[:, j] = _expanding_real_iqr(xj, mask, warmup_floor=warmup_floor)
            touched = True
    if not touched:
        return None, None
    return ref_iqr, fixed



def apply_masked_scaling(
    X: np.ndarray, X_raw: np.ndarray, df: pd.DataFrame,
    feature_names: list[str], train_win: int
) -> np.ndarray:
    """Rescale every imputed exogenous column using OBSERVED rows only.

    Applied after the ordinary rolling scaler, replacing its output for the
    columns that have an availability mask. Columns without one (HAR, calendar,
    the indicators themselves) are left exactly as the ordinary path produced
    them, so a run with no imputed exog is bit-for-bit unchanged.

    This supersedes the value-side repairs -- the neutral-median substitution
    and the staleness bound -- both of which are off by default. Their measured
    behaviour is why: the median substitution fixed a 22%-coverage channel
    (voldemand 189.5 -> 7.4 rolling IQRs) and broke a 62%-coverage one
    (effspread 17.6 -> 932.6), because what helps a sparse series (stop using
    stale values) is exactly what degrades a dense one (introduce a constant).
    Masking the ESTIMATE has no such trade-off.
    """
    out = X
    n = X.shape[0]
    touched = 0
    # X_raw is the UNSCALED matrix. Masked scaling REPLACES the ordinary
    # scaler for these columns; composing the two would estimate the masked
    # median and IQR on a series the unmasked scaler had already distorted,
    # which is what a first version of this did (voldemand landed at 64.0
    # rolling IQRs instead of the 10.5 a single-column test predicted, and
    # numobs and vix went backwards).
    jobs = []
    for j, name in enumerate(feature_names):
        m = re.match(r"adj_(.+)_ma_(\d+)$", name)
        if m is None:
            continue
        obs_col = f"{m.group(1)}__obs"
        if obs_col not in df.columns:
            continue
        # A rung-r moving average is observed when the bars feeding it were;
        # require the window's centre bar rather than all r of them, which is
        # the same convention the availability rolling means already use.
        jobs.append((j, df[obs_col].to_numpy().astype(bool)[:n]))

    # Columns are independent, and each is a few hundred rolling-quantile
    # evaluations -- the dominant cost of a rebuild. Farm them out over
    # processes, which is exact: the workers run the same function on the same
    # inputs and only the scheduling changes. Threads would not do: numpy's
    # partition holds the GIL for the window sizes here.
    n_proc = max(1, min(os.cpu_count() or 1, 8))
    if len(jobs) > 8 and n_proc > 1:
        # Submitted in batches rather than all at once: each task carries a
        # column copy, so queueing all ~500 upfront would hold about a
        # gigabyte of pickled arrays alongside a rebuild that is already the
        # heaviest process on the box.
        batch = n_proc * 4
        with cf.ProcessPoolExecutor(max_workers=n_proc) as pool:
            for s in range(0, len(jobs), batch):
                chunk = jobs[s:s + batch]
                futs = {pool.submit(masked_rolling_scale_col, X_raw[:, j], mk,
                                    train_win): j for j, mk in chunk}
                for fut in cf.as_completed(futs):
                    out[:, futs[fut]] = fut.result()
                    touched += 1
        note = f" across {n_proc} workers"
    else:
        for j, mk in jobs:
            out[:, j] = masked_rolling_scale_col(X_raw[:, j], mk, train_win)
            touched += 1
        note = ""
    if touched:
        print(f"[scale] masked rescaling applied to {touched} exogenous "
              f"columns{note}")
    return out


def _backtest_and_save(
    df: pd.DataFrame,
    feature_names: list[str],
    fit_predict: FitPredict,
    hyperparams: dict,
    train_win_periods: int,
    horizon: int,
    start: int,
    end: int,
    halo: int,
    output_file: str,
    prescale: bool = False,
    diurnal_mode: str = "divide",
) -> None:
    """Run a prepared DataFrame through the walk-forward backtest and save.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``adj_RV``, ``baseline``, ``t``, plus all
        ``feature_names`` columns. The HAR-burn-in (``max_lag``) rows
        are dropped here before any extraction.
    feature_names : list[str]
        Column names in ``df`` that form the feature matrix.
    fit_predict : FitPredict
        Callable ``(X_chunk, y_chunk, train_win_periods, hyperparams)
        -> preds`` of shape ``(len(X_chunk) - train_win_periods,)``.
    hyperparams : dict
        Method-specific hyperparameters (typically a merge of method
        defaults and tuned overrides from ``--params-file``).
    train_win_periods : int
        Training window in 30-min periods.
    horizon : int
        Forecast horizon in 30-min periods.
    start, end, halo : int
        The framework slice for this task, in *post horizon-shift* index
        space (the fields of ``hpc_agent.template.SliceSpec``). ``[start,
        end)`` is the emit range; ``halo`` is the count of warm-up rows
        replayed before it, so the chunk processed is
        ``X[start - halo : end]`` and its first ``halo`` rows train the
        model without emitting predictions (``halo`` is sized to the
        training window). ``end < 0`` means "to the end"; a whole-series
        run is ``start=0, end=-1, halo=0``.
    output_file : str
        Output CSV path. ``<basename>_reduce.json`` is written
        alongside.
    prescale : bool, default False
        If True, rolling-robust-scale the whole feature matrix
        (``rolling_robust_scale``) *before* the chunk slice — the linear /
        distance-based methods (Ridge, kNN, spectral_knn) need feature
        standardization, and doing it whole-series keeps the scaler's
        look-back out of the backtest so a ``halo == train_win`` chunk is
        fungible. Tree methods leave this False (they neither scale nor
        prescale). The matching ``fit_predict`` receives an already-scaled
        ``X_chunk`` and passes it through to its
        :class:`~src.backtest.multi_stage.MultiStageBacktest`.
    """
    max_lag = resolve_har_lags()[-1]
    df = df.iloc[max_lag:].reset_index(drop=True)

    X = df[feature_names].values.astype(np.float64)
    y = df["adj_RV"].values.astype(np.float64)
    dates = df["t"]
    baselines = df["baseline"].values.astype(np.float64)

    X, y, dates, baselines = apply_horizon_shift(X, y, dates, baselines, horizon)

    # Rolling robust scaling is a bounded-window transform; doing it here,
    # whole-series, keeps its look-back out of the per-chunk backtest (see
    # rolling_robust_scale). The slice below then cuts the pre-scaled array.
    ref_iqr = None
    if prescale:
        ref_iqr, fixed_cols = _build_scale_guards(
            X, df, feature_names, zero_inflated_active=(diurnal_mode != "rank")
        )
        X = rolling_robust_scale(
            X, train_win_periods, ref_iqr=ref_iqr, fixed_cols=fixed_cols
        )

    # Feature-health invariants, on every build. Ten defects of the same class (a transform whose
    # divisor degenerates, a fill constant that manufactures a predictor out of missingness) have
    # been found in this pipeline by *symptom* — a blown QLIKE, days later — and every one of them
    # is a seconds-long check on the finished matrix. Advisory by default; FEATURE_HEALTH_STRICT=1
    # makes a FAIL fatal. See src/diagnostics.py and analysis/diagnostics_backtest.py.
    check_and_report(
        X,
        feature_names,
        df=df,
        raw_cols=[c for c in df.columns if f"adj_{c}" in df.columns],
        ref_iqr=ref_iqr,
        train_win=train_win_periods,
        output_path=f"{os.path.splitext(output_file)[0]}_feature_health.csv",
        label=os.path.basename(os.path.splitext(output_file)[0]),
    )

    load_start = max(0, start - halo)
    actual_end = len(X) if end < 0 else end
    X_chunk = X[load_start:actual_end]
    y_chunk = y[load_start:actual_end]
    dates_chunk = dates.iloc[load_start:actual_end].reset_index(drop=True)
    baselines_chunk = baselines[load_start:actual_end]

    if train_win_periods >= len(X_chunk):
        raise ValueError(
            f"train_window ({train_win_periods} periods) >= chunk size ({len(X_chunk)})"
        )

    # The column names of X_chunk, forwarded under the established ``_*``
    # control-key convention (``_refit_frequency``, ``_incremental``): a
    # fit_predict that selects columns by MEANING (e.g. spectral_knn's
    # view_exog state block, which excludes target-derived ``har_ma_*``)
    # needs names, and re-deriving them caller-side would fork the source of
    # truth. Every other fit_predict strips ``_*`` keys and is unaffected.
    hyperparams = dict(hyperparams, _feature_names=list(feature_names))

    preds = fit_predict(X_chunk, y_chunk, train_win_periods, hyperparams)

    oos_start = train_win_periods
    y_oos = y_chunk[oos_start:]
    dates_oos = dates_chunk.iloc[oos_start:].values
    baselines_oos = baselines_chunk[oos_start:]

    pred_raw, true_raw = apply_duan_smearing(preds, y_oos, baselines_oos)

    results = pd.DataFrame(
        {
            "date": dates_oos,
            "horizon": horizon,
            "true_adj": y_oos,
            "pred_adj": preds,
            "true_raw": true_raw,
            "pred_raw": pred_raw,
        }
    )

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    results.to_csv(output_file, index=False)
    save_chunk_reduce(results, output_file)
    print(f"Saved {len(results)} rows -> {output_file}")


def _build_har_and_calendar(df, exog_cols, add_calendar, har_lags=None):
    df, har_names = generate_har_features(
        df, target_col="adj_RV", exog_cols=exog_cols, lags=har_lags
    )
    if add_calendar:
        # add_expiry_features after add_calendar_features: it reads ``is_close`` to build the
        # quad-witch closing-auction gate, which is where the index-rebalance flow actually prints.
        feature_names = har_names + add_calendar_features(df)
        feature_names += add_expiry_features(df)
        # HAR x open/close session-edge interactions (the distilled intraday vol-persistence
        # sign-flip: high recent RV -> LOWER future RV at the 09:30 open and 16:00 close/AH).
        # Materialized upstream so every model + the cached matrix inherit them; causal (the
        # is_open/is_close gates are pure functions of the timestamp). Mirrors the no-rebuild
        # resid_amortized._regime_interactions, identical naming (har_ma_{w}_x_{open,close}).
        for h in [c for c in har_names if c.startswith("har_ma_")]:  # the 6 target-HAR cols
            for gate, suffix in (("is_open", "open"), ("is_close", "close")):
                name = f"{h}_x_{suffix}"  # matches resid_amortized: har_ma_{w}_x_{open,close}
                df[name] = df[h] * df[gate]
                feature_names.append(name)
    else:
        feature_names = har_names
    return df, feature_names


def load_and_transform(
    data_path: str,
    exog_cols: list[str],
    *,
    target_use_diurnal: bool,
    target_winsor_window: int | None,
    dropna_with_exog: bool,
    overnight_fill: bool = True,
    impute_indicate: bool = False,
    diurnal_mode: str = "divide",
    use_semantic: bool = True,
    ffill_limit: int | None = None,
    legacy_avail: bool = False,
    neutralise_unobserved: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Load raw data, apply RV + exog robust transforms, return (df, adj_exog).

    ``diurnal_mode`` (``"divide"`` | ``"rank"``) selects how the per-slot diurnal
    adjustment is done for the EXOG features: the default divide, or the
    division-free per-slot rank-Gauss (:func:`diurnal_rank`). The target's diurnal
    is always divide (its raw reconstruction needs the multiplicative baseline).

    Parameters
    ----------
    data_path : str
        Forwarded to ``load_raw_data``.
    exog_cols : list[str]
        Raw exog column names (pre-transform). May be empty.
    target_use_diurnal : bool
        If True, apply diurnal adjustment to the RV target transform.
        Tree methods: True. Ridge: False.
    target_winsor_window : int | None
        Winsorization window for the RV target. Tree methods: 240.
        Ridge: None (no winsorization beyond ``robust_transform``
        defaults).
    dropna_with_exog : bool
        Exog columns are forward-filled before this branch, so the only
        rows that can still have NaN exog are at the leading edge of the
        dataset (before any valid value). If True, drop those rows.
        Ridge: True (sklearn Ridge requires no NaN). Tree methods: False
        (trees handle NaN natively).
    overnight_fill : bool
        If True (default, legacy behaviour), fill matching exog columns with
        1.0 in their overnight window before ffill. This substring-matches
        moment columns (``sumret3_*`` etc.) and fills them with 1.0, which is
        a latent bug for non-ratio columns — set False for clean ffill-only.
    """
    df = load_raw_data(data_path, allow_missing=True)
    if exog_cols:
        if overnight_fill:
            apply_overnight_fills(df, exog_cols)
        # Availability is recorded BEFORE the forward fill. Recording it after
        # (the historical behaviour, kept under ``legacy_avail``) makes the
        # indicator mean "this series has started" rather than "this bar is
        # observed", because ffill carries the last value across every gap and
        # ``notna()`` is then True on stale rows too. Measured consequence on
        # the voldemand family: raw coverage ~33.6%, indicator mean 0.698 --
        # the indicator was 0 only for the leading years before the feed
        # begins. That defeats all three guards at once. The impute-and-
        # indicate median fill never fires (ffill left no NaN), the model
        # cannot tell fresh from stale, and ``_build_scale_guards`` computes
        # its "real values" IQR floor over the ffilled constants, so the floor
        # is itself degenerate and the rolling IQR collapses. The scaled
        # columns then reach 200+ IQRs, and on 13-15 October 2023 that
        # cancelled the backbone and drove the forecast two orders of
        # magnitude low (analysis/october_2023.py).
        obs_cols = {}
        for col in exog_cols:
            obs_cols[col] = df[col].notna().to_numpy(copy=True)
        # A finite staleness limit is the other half. Carrying a value forward
        # indefinitely is defensible for a short gap and indefensible for a
        # feed that stops: voldemand is 0% observed through 2024, so an
        # unlimited ffill presents a 2023 value as current for a whole year.
        # Past the limit the value returns to NaN and the impute-and-indicate
        # branch below replaces it with the neutral median.
        df[exog_cols] = df[exog_cols].ffill(limit=ffill_limit)
        for col in exog_cols:
            df[f"{col}__obs"] = obs_cols[col] if not legacy_avail \
                else df[col].notna().to_numpy()
        if dropna_with_exog and not impute_indicate:
            df = df.dropna(subset=["RV"] + exog_cols).reset_index(drop=True)
        else:
            # impute_indicate keeps exog-missing rows (filled + flagged below).
            df = df.dropna(subset=["RV"]).reset_index(drop=True)

    transform_kwargs: dict = {"is_target": True}
    if target_use_diurnal:
        transform_kwargs["use_diurnal"] = True
    if target_winsor_window is not None:
        transform_kwargs["winsor_window"] = target_winsor_window
    adj_rv, baseline = robust_transform(df, "RV", **transform_kwargs)
    df["adj_RV"] = adj_rv
    df["baseline"] = baseline

    adj_exog_cols: list[str] = []
    for col in exog_cols:
        adj_col = f"adj_{col}"
        adj_series, _ = robust_transform(df, col, use_transform=use_semantic, use_diurnal=True, diurnal_mode=diurnal_mode)
        df[adj_col] = adj_series
        adj_exog_cols.append(adj_col)

    if impute_indicate and exog_cols:
        # Impute-and-indicate: instead of dropping rows where an exog is
        # structurally absent (which shrinks the sample and breaks cross-arm
        # QLIKE comparability), fill its adjusted feature with 0 (neutral) and
        # add a ``<col>_avail`` availability indicator. Every row is then
        # scored and the model can down-weight the feature where it's absent.
        for col in exog_cols:
            # Neutral fill must be in-distribution in the *transformed* space:
            # the feature's own median. 0 is wrong (e.g. adj_vix = log(vix), so
            # fillna(0) imputes vix = exp(0) = 1, a wild value that blew the
            # Ridge forecast up to QLIKE ~2.4). Median ≈ neutral, prescales to 0.
            adj = df[f"adj_{col}"]
            # Unobserved rows take the neutral median. This handles the
            # NUMERATOR half of the pathology: a forward-filled value is stale,
            # and on a sparsely observed channel it drifts arbitrarily far from
            # the current regime, so no scale estimate can rescue it (masking
            # the estimate alone still leaves voldemand at 231.9 rolling IQRs).
            #
            # On its own this substitution BREAKS moderately covered channels,
            # because replacing varying values with a constant collapses the
            # rolling IQR: effspread_vwstock went 17.6 -> 932.6 when it shipped
            # alone. That is the DENOMINATOR half, and it is fixed by
            # scaling.masked_rolling_scale_col, which estimates location and
            # scale from observed rows only.
            #
            # The two compose and neither works without the other -- measured
            # peak rolling IQRs, ffill/median x plain/masked scaling:
            #
            #   voldemand (cov 0.22)   43705.7 / 231.9 / 11813.6 / 10.5
            #   effspread (cov 0.62)      17.3 /  20.2 /    34.1 / 14.1
            #   sumbipow  (cov 0.62)      84.6 /  95.5 /    71.6 / 17.7
            #   vix       (cov 0.35)    3296.4 /   6.0 /  7141.6 /  5.6
            #
            # Testing them one at a time is why this took three attempts to
            # get right: each is harmful or inert alone and all the value is in
            # the interaction.
            obs = df[f"{col}__obs"].to_numpy().astype(bool)
            if neutralise_unobserved:
                adj = adj.where(pd.Series(obs, index=adj.index), np.nan)
            df[f"adj_{col}"] = adj.fillna(adj.median())
            avail = f"{col}_avail"
            df[avail] = obs.astype("float64")
            adj_exog_cols.append(avail)

    # Hurdle (two-part) encoding for mode-inflated exog. A sparse signed flow
    # like voldemand is a mixture — a point mass at 0 ("no demand") plus a
    # continuous demand distribution — so a single robust scale is degenerate
    # (the point mass collapses the IQR). Add an occurrence indicator so the
    # model separates the extensive margin (is there demand at all) from the
    # intensive margin (how big); the magnitude's scale is taken over the
    # off-mode (active) values in ``_build_scale_guards``. The point mass need
    # not sit at zero — see ``ZERO_INFLATED_FRAC`` for ``numobs``.
    for col in exog_cols:
        vals = df[col].to_numpy(dtype=np.float64)
        inflated, mode = mode_inflated(vals)
        if inflated:
            active = f"{col}_active"
            # "off the point mass": for a zero-inflated flow that is "there was demand", for
            # ``numobs`` it is "this bar was not a full 30-observation bar" — i.e. a holiday or
            # half-session, which is exactly the information the 80-sigma tail was encoding.
            df[active] = (np.nan_to_num(vals, nan=mode) != mode).astype("float64")
            adj_exog_cols.append(active)

    return df, adj_exog_cols


def _iter_TOD_segment(
    df,
    *,
    segment,
    lag_scope,
    train_window,
    output_file,
    exog_cols,
    add_calendar,
    har_lags=None,
):
    """Yield (seg_name, job_df, feature_names, train_win_periods, job_output_file)
    for each time-of-day segment we need to backtest. ``seg_name`` is None when
    no segmentation is requested."""
    if segment is None:
        df, feature_names = _build_har_and_calendar(df, exog_cols, add_calendar, har_lags)
        yield None, df, feature_names, train_window * PERIODS_PER_DAY, output_file
        return

    segments = list(SEGMENT_DEFINITIONS) if segment == "all" else [segment]
    base, ext = os.path.splitext(output_file)

    if lag_scope == "global":
        df, feature_names = _build_har_and_calendar(df, exog_cols, add_calendar, har_lags)

    for seg_name in segments:
        seg_df = slice_to_segment(df, seg_name)
        if seg_df.empty:
            print(f"No data for segment '{seg_name}'. Skipping.")
            continue
        if lag_scope == "intra":
            seg_df, feature_names = _build_har_and_calendar(
                seg_df, exog_cols, add_calendar, har_lags
            )
        train_win_periods = compute_segment_train_window(seg_df["t"], train_window)
        yield (
            seg_name,
            seg_df,
            feature_names,
            train_win_periods,
            f"{base}_{seg_name}{ext}",
        )


def run_executor(
    method_name: str,
    fit_predict: FitPredict,
    hyperparams: dict,
    *,
    data_path: str,
    output_file: str,
    horizon: int,
    train_window: int,
    start: int,
    end: int,
    halo: int,
    exog_cols: list[str],
    segment: str | None,
    lag_scope: str,
    add_calendar: bool,
    target_use_diurnal: bool,
    target_winsor_window: int | None,
    dropna_with_exog: bool,
    overnight_fill: bool = True,
    impute_indicate: bool = False,
    diurnal_mode: str = "divide",
    prescale: bool = False,
    seed: int = 42,
    har_lags: list[int] | None = None,
) -> None:
    """Top-level scaffold. Loads data, builds features, dispatches backtest.

    Per-method scripts (`ml_xgboost.py`, etc.) call this from their
    ``run()`` after reading the framework slice. ``start`` / ``end`` /
    ``halo`` are the plain-int fields of the active
    ``hpc_agent.template.SliceSpec`` (the chunked-axis seam) — see
    ``_backtest_and_save`` for their meaning; a whole-series run passes
    ``start=0, end=-1, halo=0``.

    Notes
    -----
    * ``segment`` is None for tree methods (XGB/LGBM/RF). Ridge is the
      only method that uses segments.
    * ``add_calendar`` is True for tree methods, False for Ridge.
    * ``prescale`` is True only for the linear methods (Ridge): it
      rolling-robust-scales the feature matrix whole-series so a chunked
      backtest needs only a train-window halo. Tree methods leave it False.
    * ``method_name`` is informational — used in log lines only. The
      per-method data-prep invariants it once keyed a drift-check on are
      now inline literals in each executor's ``run()`` (B2).
    """
    del seed  # reserved; per-method scripts can wire seed into model_fn directly

    if har_lags is not None:
        burn_in = resolve_har_lags()[-1]
        if not har_lags or min(har_lags) < 1 or max(har_lags) > burn_in:
            raise ValueError(
                f"har_lags must be non-empty with rungs in [1, {burn_in}]: the "
                f"HAR burn-in drop is fixed at {burn_in} rows (arm-invariant "
                f"index space); a rung beyond it would be under-warmed."
            )

    df, adj_exog_cols = load_and_transform(
        data_path,
        exog_cols,
        target_use_diurnal=target_use_diurnal,
        target_winsor_window=target_winsor_window,
        dropna_with_exog=dropna_with_exog,
        overnight_fill=overnight_fill,
        impute_indicate=impute_indicate,
        diurnal_mode=diurnal_mode,
    )

    for seg_name, job_df, feature_names, train_win, out_file in _iter_TOD_segment(
        df,
        segment=segment,
        lag_scope=lag_scope,
        train_window=train_window,
        output_file=output_file,
        exog_cols=adj_exog_cols,
        add_calendar=add_calendar,
        har_lags=har_lags,
    ):
        if seg_name is not None:
            print(
                f"{'=' * 20} {method_name.upper()} SEGMENT: {seg_name.upper()} {'=' * 20}"
            )
            print(f"Window: {train_win} periods ({train_window} days)")
        _backtest_and_save(
            job_df,
            feature_names,
            fit_predict,
            hyperparams,
            train_win,
            horizon,
            start,
            end,
            halo,
            out_file,
            prescale,
            diurnal_mode,
        )
