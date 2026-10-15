"""Pooled-panel inputs for the native MinGRU memorisation run (user, 2026-10-14).

The regular stack's pooled panel -- src.unification._load_panel: experiments/run_geometry_local.prepare_full
("all_features", har_base=2), cached as harxhar-clean/results/prep_cache_all_features_b2.npz (X 300,317 x 1,144
float64, y = y_fit, baselines, names) -- exactly as the pooled tree bank sees it (experiments/pooled_trees_walk.py):
one model for all 48 half-hour bars (round the clock, ET), refit on the trailing 24,000 bars.

  * scaling   the panel X is ALREADY the prep's rolling robust scale (prepare_full applies rolling_robust_scale +
              the scale guards before any arm sees it; the pooled linear arms take p.X straight, "apply NO further
              standardization", src/unification.py:1195-1212; _Panel.X is documented "prescaled features"), so the
              trainer feeds it unchanged: xstd=0.  mask=1 (window_keep on the fit window) as always.
  * sets      all_features = every column; baseline = src.unification._backbone_cols (HAR rungs + session-edge
              interactions + calendar, 52 columns), as experiments/pooled_trees_walk.set_columns.
  * target    y = the cache's y (y_fit: winsorised sqrt(RV / B)); checked > 0 on every row (softplus needs mu_y > 0).
  * window    the bank's 24,000 trailing bars ending at the last bar of 2019-12-31: panel rows [222053, 246053)
              (2018-01-30 22:00 .. 2019-12-31 20:00 ET), then 480 bars out of sample (2020-01-01 18:30 ..
              2020-01-15 14:00 ET).  Timestamps: the bank's own forecast table harxhar-clean/results/spxw_pnl/
              yhat_a0.parquet (panel row = table row + 24,000), checked against the panel's y (yhat table carries
              baseline / rv_raw: y_fit = sqrt(rv_raw / baseline) away from the winsorised tails).
  * export    rows [222053 - 47, 246053 + 480) only (L <= 48), as data/pooled_<set>/{X.f32, X.f64, y.f32, meta.txt}
              + rows.csv (slice row, panel row, t ET, half-hour slot 0..47); the X is read through a memory map of the
              npz's stored X.npy member (2.75 GB; this machine has ~2 GB of RAM).

    python experiments/puffer_forecast/prep_pooled.py          # -> results/puffer_forecast_2026-10-12/_work/pooled/
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
CLEAN = REPO.parent / "harxhar-clean"
CACHE = CLEAN / "results" / "prep_cache_all_features_b2.npz"
YHAT = CLEAN / "results" / "spxw_pnl" / "yhat_a0.parquet"
OUT = REPO / "results" / "puffer_forecast_2026-10-12" / "_work" / "pooled"
WIN0, WIN1, OOS, LMAX = 222053, 246053, 480, 48
WARMUP = 24_000  # panel row = bank forecast-table row + 24,000


def member_memmap(path: Path, name: str) -> np.ndarray:
    """np.memmap of a STORED (uncompressed) .npy member of an .npz."""
    z = zipfile.ZipFile(path)
    info = z.getinfo(name)
    assert info.compress_type == zipfile.ZIP_STORED, f"{name} is compressed"
    with open(path, "rb") as fh:
        fh.seek(info.header_offset)
        local = fh.read(30)
        n_name, n_extra = (
            int.from_bytes(local[26:28], "little"),
            int.from_bytes(local[28:30], "little"),
        )
        fh.seek(info.header_offset + 30 + n_name + n_extra)
        version = np.lib.format.read_magic(fh)
        shape, fortran, dtype = np.lib.format._read_array_header(fh, version)  # type: ignore[attr-defined]
        offset = fh.tell()
    return np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=offset,
        shape=shape,
        order="F" if fortran else "C",
    )


def main() -> None:
    import src.unification as U

    z = np.load(
        CACHE, allow_pickle=True
    )  # names is an object array (the stack's own cache)
    names = [str(s) for s in z["names"]]
    y = np.asarray(z["y"], np.float64)
    base = np.asarray(z["baselines"], np.float64)
    X = member_memmap(CACHE, "X.npy")
    n, p = X.shape
    assert n == len(y) == 300_317 and p == len(names) == 1_144, (
        X.shape,
        len(y),
        len(names),
    )
    assert (y >= 0).all()
    r0, r1 = WIN0 - (LMAX - 1), WIN1 + OOS
    # y_fit = 0 on 262 panel rows (RV printed 0), none in the exported rows: softplus can fit every row here
    assert (y[r0:r1] > 0).all(), f"y_fit <= 0 on {(y[r0:r1] <= 0).sum()} exported rows"
    # timestamps / target check from the bank's own table (panel row = table row + 24,000)
    tab = pd.read_parquet(YHAT, columns=["t", "baseline", "rv_raw"])
    assert len(tab) == n - WARMUP
    t = pd.DatetimeIndex(tab["t"].iloc[r0 - WARMUP : r1 - WARMUP]).tz_convert(
        "America/New_York"
    )
    tb = tab.iloc[r0 - WARMUP : r1 - WARMUP]
    assert np.allclose(tb["baseline"].to_numpy(), base[r0:r1], rtol=0, atol=0), (
        "baseline misaligned"
    )
    ychk = np.sqrt(tb["rv_raw"].to_numpy() / tb["baseline"].to_numpy())
    med = float(np.median(np.abs(ychk - y[r0:r1])))
    assert med == 0.0, f"y_fit vs sqrt(rv_raw / baseline): median |diff| {med}"
    slot = (t.hour * 2 + t.minute // 30).to_numpy()
    sets = {"all_features": np.arange(p), "baseline": U._backbone_cols(names)}
    Xw = np.asarray(X[r0:r1], np.float64)
    assert np.isfinite(Xw).all()
    OUT.mkdir(parents=True, exist_ok=True)
    info: dict = dict(
        panel_rows=[r0, r1],
        window_panel=[WIN0, WIN1],
        oos=OOS,
        w0=WIN0 - r0,
        w1=WIN1 - r0,
        first_window_bar=str(t[WIN0 - r0]),
        last_window_bar=str(t[WIN1 - r0 - 1]),
        first_oos_bar=str(t[WIN1 - r0]),
        last_oos_bar=str(t[-1]),
        window_days=int(pd.Index(t[WIN0 - r0 : WIN1 - r0].date).nunique()),
        window_days_with_1600=int(
            (
                (t[WIN0 - r0 : WIN1 - r0].hour == 16)
                & (t[WIN0 - r0 : WIN1 - r0].minute == 0)
            ).sum()
        ),
        y_zero_rows_panel=int((y == 0).sum()),
        y_min_window=float(y[WIN0:WIN1].min()),
        y_mean_window=float(y[WIN0:WIN1].mean()),
        y_sd_window=float(y[WIN0:WIN1].std()),
        scaling="prescaled (rolling robust scale in prepare_full); xstd=0",
        sets={},
    )
    for s, cols in sets.items():
        d = OUT / f"pooled_{s}"
        d.mkdir(exist_ok=True)
        Xs = np.ascontiguousarray(Xw[:, cols])
        Xs.astype("<f4").tofile(d / "X.f32")
        Xs.astype("<f8").tofile(d / "X.f64")
        y[r0:r1].astype("<f4").tofile(d / "y.f32")
        (d / "meta.txt").write_text(f"{r1 - r0} {len(cols)}\n")
        (d / "columns.txt").write_text("\n".join(names[c] for c in cols) + "\n")
        info["sets"][s] = dict(p=int(len(cols)), max_abs_x=float(np.max(np.abs(Xs))))
    pd.DataFrame(
        {
            "row": np.arange(r1 - r0),
            "panel_row": np.arange(r0, r1),
            "t_et": t.strftime("%Y-%m-%d %H:%M"),
            "slot": slot,
        }
    ).to_csv(OUT / "rows.csv", index=False)
    (OUT / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
