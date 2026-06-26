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

import os
import re
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.evaluation.metrics import apply_duan_smearing, save_chunk_reduce
from src.data.loading import apply_overnight_fills, load_raw_data
from src.features.transforms.scaling import rolling_robust_scale
from src.backtest.segmentation import (
    SEGMENT_DEFINITIONS,
    compute_segment_train_window,
    slice_to_segment,
)
from src.features.extractors.calendar import add_calendar_features
from src.features.extractors.har import generate_har_features, resolve_har_lags
from src.features.transforms.target import (
    PERIODS_PER_DAY,
    apply_horizon_shift,
    robust_transform,
)

FitPredict = Callable[[np.ndarray, np.ndarray, int, dict], np.ndarray]

# A feature with at least this fraction of *exact* zeros is treated as
# zero-inflated (a point mass at 0 + a continuous part) and gets the hurdle
# encoding: an occurrence indicator plus a magnitude scaled over its non-zero
# (active) values. Continuous features (returns, vols) never reach this.
ZERO_INFLATED_FRAC: float = 0.2


def _expanding_real_iqr(
    x: np.ndarray, mask: np.ndarray, grid_step: int = 480
) -> np.ndarray:
    """Causal expanding IQR of ``x`` over its *available* (``mask``) values.

    A slowly-varying reference scale, so it is computed on a coarse grid and
    forward-filled rather than re-evaluated every row. Excluding the imputed
    (unavailable) rows is the crux: the median-fill injects a flat constant
    that otherwise dominates — and collapses — the window IQR. Causal: row
    ``t`` only ever sees real values strictly before ``t``.
    """
    n = len(x)
    ref = np.zeros(n, dtype=np.float64)
    real_idx = np.flatnonzero(mask)
    last = 0.0
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
    X: np.ndarray, df: pd.DataFrame, feature_names: list[str], zero_inflated_active: bool = True
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Scale guards for the impute-and-indicate feature pathology.

    Returns ``(ref_iqr, fixed_cols)`` for :func:`rolling_robust_scale`, or
    ``(None, None)`` when no guarding is needed (no availability columns →
    not an ``impute_indicate`` run → scaling is bit-for-bit unchanged).

    * ``*_avail_ma_*`` / ``*_active_ma_*`` (availability + occurrence-indicator
      rolling means) → *fixed*: passed through raw; scale is known ([0, 1]).
    * ``adj_<col>_ma_*`` whose ``<col>_avail`` column exists (the exog was
      imputed) → IQR floored at the causal expanding IQR over the *real*
      (available) values; for a *zero-inflated* magnitude the scale is taken
      over active (non-zero) values too, so neither a regime-degenerate window
      nor a zero point-mass can collapse the scale and blow up the forecast.
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
        m = re.match(r"adj_(.+)_ma_\d+$", name)
        if m is None:
            continue
        avail_col = f"{m.group(1)}_avail"
        if avail_col in df.columns:
            mask = df[avail_col].to_numpy().astype(bool)[:n]
            xj = X[:, j]
            # Zero-inflated magnitude (e.g. voldemand): also estimate the scale
            # over ACTIVE (non-zero) values — the zero point-mass carries no
            # dispersion and collapses the IQR otherwise. The indicator half of
            # the hurdle encoding is the ``<col>_active`` columns added in
            # ``load_and_transform``.
            if zero_inflated_active and float(np.mean(xj == 0.0)) >= ZERO_INFLATED_FRAC:
                mask = mask & (xj != 0.0)
            ref_iqr[:, j] = _expanding_real_iqr(xj, mask)
            touched = True
    if not touched:
        return None, None
    return ref_iqr, fixed


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
    if prescale:
        ref_iqr, fixed_cols = _build_scale_guards(
            X, df, feature_names, zero_inflated_active=(diurnal_mode != "rank")
        )
        X = rolling_robust_scale(
            X, train_win_periods, ref_iqr=ref_iqr, fixed_cols=fixed_cols
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


def _build_har_and_calendar(df, exog_cols, add_calendar):
    df, har_names = generate_har_features(df, target_col="adj_RV", exog_cols=exog_cols)
    if add_calendar:
        feature_names = har_names + add_calendar_features(df)
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
        df[exog_cols] = df[exog_cols].ffill()
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
            df[f"adj_{col}"] = adj.fillna(adj.median())
            avail = f"{col}_avail"
            df[avail] = df[col].notna().astype("float64")
            adj_exog_cols.append(avail)

    # Hurdle (two-part) encoding for zero-inflated exog. A sparse signed flow
    # like voldemand is a mixture — a point mass at 0 ("no demand") plus a
    # continuous demand distribution — so a single robust scale is degenerate
    # (the zero mass collapses the IQR). Add an occurrence indicator so the
    # model separates the extensive margin (is there demand at all) from the
    # intensive margin (how big); the magnitude's scale is taken over the
    # active (non-zero) values in ``_build_scale_guards``.
    for col in exog_cols:
        vals = df[col].to_numpy(dtype=np.float64)
        finite = vals[~np.isnan(vals)]
        if finite.size and float(np.mean(finite == 0.0)) >= ZERO_INFLATED_FRAC:
            active = f"{col}_active"
            df[active] = (np.nan_to_num(vals) != 0.0).astype("float64")
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
):
    """Yield (seg_name, job_df, feature_names, train_win_periods, job_output_file)
    for each time-of-day segment we need to backtest. ``seg_name`` is None when
    no segmentation is requested."""
    if segment is None:
        df, feature_names = _build_har_and_calendar(df, exog_cols, add_calendar)
        yield None, df, feature_names, train_window * PERIODS_PER_DAY, output_file
        return

    segments = list(SEGMENT_DEFINITIONS) if segment == "all" else [segment]
    base, ext = os.path.splitext(output_file)

    if lag_scope == "global":
        df, feature_names = _build_har_and_calendar(df, exog_cols, add_calendar)

    for seg_name in segments:
        seg_df = slice_to_segment(df, seg_name)
        if seg_df.empty:
            print(f"No data for segment '{seg_name}'. Skipping.")
            continue
        if lag_scope == "intra":
            seg_df, feature_names = _build_har_and_calendar(
                seg_df, exog_cols, add_calendar
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
