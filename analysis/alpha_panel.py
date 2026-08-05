"""Build (and cache) the production-space panel used by the alpha-manifestation study.

Reproduces the Ridge production feature space exactly as ``executor.load_and_transform``
+ ``_build_har_and_calendar`` + ``prescale=True`` build it — target
``adj_RV = winsorize(sqrt(RV / rolling-per-slot-diurnal-mean))``, HAR rolling means at
``[1,5,25,125,625,3125]`` for the target and every exog, calendar + the session-edge
``har_ma_w x {open,close}`` interactions, impute-and-indicate for structurally-absent
exog, the hurdle ``_active`` indicators for zero-inflated exog, and the rolling-robust
prescale with the ``_build_scale_guards`` floors.

Two reasons this is a separate module rather than a call into ``run_executor``:

1. The study needs the *unaggregated* matrix (features tagged by bucket, plus the
   timestamps) in memory, not a walk-forward metrics dict.
2. It is a second, independent implementation of the same prep, which is how the
   ``diurnal_mode`` ``TypeError`` in ``executor.load_and_transform`` (every exog run) was
   found and fixed.

The panel is cached as a single ``.npz`` (float32 matrix) keyed by the config; rebuild
cost is ~4 min, reload ~2 s.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.executor import _build_scale_guards, mode_inflated  # noqa: E402
from src.diagnostics import check_and_report  # noqa: E402
from src.data.loading import ALL_FEATURES, SUBGROUPS, load_raw_data  # noqa: E402
from src.features.extractors.calendar import add_calendar_features  # noqa: E402
from src.features.extractors.expiry import add_expiry_features  # noqa: E402
from src.features.extractors.har import generate_har_features, resolve_har_lags  # noqa: E402
from src.features.transforms.scaling import rolling_robust_scale  # noqa: E402
from src.features.transforms.target import PERIODS_PER_DAY, robust_transform  # noqa: E402

CACHE_DIR = os.environ.get(
    "ALPHA_PANEL_CACHE",
    "/tmp/claude-0/-home-user-harxhar-clean/8d6e56a6-934c-5a06-a38d-8fa3ada92453/scratchpad",
)

# Which raw exog each adjusted feature descends from -> bucket tag.
_BUCKET_OF_RAW: dict[str, str] = {}
for _name, _cols in SUBGROUPS.items():
    if _name in ("baseline", "all_features"):
        continue
    for _c in _cols:
        _BUCKET_OF_RAW.setdefault(_c, _name)


@dataclass
class Panel:
    """The prepared walk-forward panel.

    Attributes
    ----------
    X : (n, p) float32
        Prescaled feature matrix, columns ``names``.
    y : (n,) float64
        ``adj_RV`` target, already horizon-shifted (horizon=1 is a no-op: the
        ``.shift(1)`` lives in the HAR construction).
    t : (n,) datetime64
        Bar end timestamps.
    baseline : (n,) float64
        Diurnal baseline, for raw-space reconstruction.
    names : list[str]
        Feature names, aligned to ``X`` columns.
    kind : list[str]
        Per-column tag: ``har`` | ``calendar`` | ``regime`` | ``value`` | ``indicator``.
    bucket : list[str]
        Per-column exog bucket (``""`` for har/calendar/regime columns).
    raw : list[str]
        Per-column source raw exog name (``""`` for har/calendar).
    window : list[int]
        Per-column HAR window (``0`` for non-MA columns).
    """

    X: np.ndarray
    y: np.ndarray
    t: np.ndarray
    baseline: np.ndarray
    names: list[str]
    kind: list[str]
    bucket: list[str]
    raw: list[str]
    window: list[int]

    def cols(self, kind: str | None = None, bucket: str | None = None) -> np.ndarray:
        """Column indices matching a kind and/or bucket."""
        idx = [
            j
            for j in range(len(self.names))
            if (kind is None or self.kind[j] == kind)
            and (bucket is None or self.bucket[j] == bucket)
        ]
        return np.asarray(idx, dtype=np.int64)


def _classify(name: str, raw_of: dict[str, str]) -> tuple[str, str, str, int]:
    """(kind, bucket, raw, window) for one feature column."""
    if name.startswith("har_ma_"):
        if "_x_" in name:
            return "regime", "", "", int(name.split("_x_")[0].rsplit("_", 1)[1])
        return "har", "", "", int(name.rsplit("_", 1)[1])
    if "_ma_" in name:
        stem, w = name.rsplit("_ma_", 1)
        raw = raw_of.get(stem, "")
        bucket = _BUCKET_OF_RAW.get(raw, "")
        kind = (
            "indicator"
            if (stem.endswith("_avail") or stem.endswith("_active"))
            else "value"
        )
        return kind, bucket, raw, int(w)
    return "calendar", "", "", 0


def build_panel(
    data_path: str = "data",
    train_window: int = 250,
    exog: list[str] | None = None,
) -> Panel:
    """Build the panel from scratch (see module docstring for the mirrored pipeline)."""
    exog_cols = list(ALL_FEATURES if exog is None else exog)
    df = load_raw_data(data_path, allow_missing=True)
    df[exog_cols] = df[exog_cols].ffill()  # overnight gaps; RATIO_FILL_COLS is empty
    df = df.dropna(subset=["RV"]).reset_index(drop=True)

    adj_rv, baseline = robust_transform(
        df, "RV", is_target=True, use_diurnal=True, winsor_window=240
    )
    df["adj_RV"] = adj_rv
    df["baseline"] = baseline

    raw_of: dict[str, str] = {}
    adj_exog: list[str] = []
    for col in exog_cols:
        adj, _ = robust_transform(df, col, use_transform=True, use_diurnal=True)
        df[f"adj_{col}"] = adj
        adj_exog.append(f"adj_{col}")
        raw_of[f"adj_{col}"] = col

    # impute-and-indicate: median fill in the *transformed* space + availability flag
    for col in exog_cols:
        s = df[f"adj_{col}"]
        df[f"adj_{col}"] = s.fillna(s.median())
        df[f"{col}_avail"] = df[col].notna().astype("float64")
        adj_exog.append(f"{col}_avail")
        raw_of[f"{col}_avail"] = col

    # hurdle encoding for mode-inflated exog (see ``executor.ZERO_INFLATED_FRAC``)
    for col in exog_cols:
        v = df[col].to_numpy(dtype=np.float64)
        inflated, mode = mode_inflated(v)
        if inflated:
            df[f"{col}_active"] = (np.nan_to_num(v, nan=mode) != mode).astype("float64")
            adj_exog.append(f"{col}_active")
            raw_of[f"{col}_active"] = col

    df, har_names = generate_har_features(df, target_col="adj_RV", exog_cols=adj_exog)
    names = har_names + add_calendar_features(df) + add_expiry_features(df)
    for h in [c for c in har_names if c.startswith("har_ma_")]:
        for gate, suffix in (("is_open", "open"), ("is_close", "close")):
            df[f"{h}_x_{suffix}"] = df[h] * df[gate]
            names.append(f"{h}_x_{suffix}")

    df = df.iloc[resolve_har_lags()[-1] :].reset_index(drop=True)
    X = df[names].to_numpy(dtype=np.float64)
    ref_iqr, fixed = _build_scale_guards(X, df, names)
    X = rolling_robust_scale(
        X, train_window * PERIODS_PER_DAY, ref_iqr=ref_iqr, fixed_cols=fixed
    )

    # Same invariants the production build runs (see ``src.diagnostics``). This module exists
    # partly because it is a second implementation of the same prep — so it gets the same checks,
    # and a divergence between the two panels shows up here rather than in a forecast.
    check_and_report(
        X,
        names,
        df=df,
        raw_cols=exog_cols,
        ref_iqr=ref_iqr,
        train_win=train_window * PERIODS_PER_DAY,
        output_path=os.path.join(CACHE_DIR, f"feature_health_tw{train_window}.csv"),
        label="alpha_panel",
    )

    meta = [_classify(n, raw_of) for n in names]
    return Panel(
        X=X.astype(np.float32),
        y=df["adj_RV"].to_numpy(dtype=np.float64),
        t=df["t"].to_numpy(),
        baseline=df["baseline"].to_numpy(dtype=np.float64),
        names=names,
        kind=[m[0] for m in meta],
        bucket=[m[1] for m in meta],
        raw=[m[2] for m in meta],
        window=[m[3] for m in meta],
    )


def load_panel(
    data_path: str = "data", train_window: int = 250, rebuild: bool = False
) -> Panel:
    """Cached :func:`build_panel`."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"alpha_panel_tw{train_window}.npz")
    if os.path.exists(path) and not rebuild:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return Panel(
            X=z["X"],
            y=z["y"],
            t=z["t"],
            baseline=z["baseline"],
            names=meta["names"],
            kind=meta["kind"],
            bucket=meta["bucket"],
            raw=meta["raw"],
            window=meta["window"],
        )
    p = build_panel(data_path, train_window)
    np.savez_compressed(
        path,
        X=p.X,
        y=p.y,
        t=p.t,
        baseline=p.baseline,
        meta=json.dumps(
            {
                "names": p.names,
                "kind": p.kind,
                "bucket": p.bucket,
                "raw": p.raw,
                "window": p.window,
            }
        ),
    )
    return p


if __name__ == "__main__":
    p = load_panel(rebuild="--rebuild" in sys.argv)
    print(
        f"X {p.X.shape}  y {p.y.shape}  {pd.Timestamp(p.t[0])} .. {pd.Timestamp(p.t[-1])}"
    )
    kinds = pd.Series(p.kind).value_counts()
    print(kinds.to_string())
    print("buckets:", pd.Series([b for b in p.bucket if b]).value_counts().to_dict())
    print("nan/inf:", int(np.isnan(p.X).sum()), int(np.isinf(p.X).sum()))
    print("y nan:", int(np.isnan(p.y).sum()), " |X| max:", float(np.abs(p.X).max()))
