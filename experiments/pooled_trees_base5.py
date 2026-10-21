"""Pooled trees at base 5 (user decision 2026-10-19: "we need base 5 across the entire paper").

The printed pooled LightGBM, XGBoost and random forest (results/pooled_trees_2026-10-07; experiments/pooled_trees_walk.py)
were fitted on the base-2 panel (src/unification.py HAR_BASE = 2: prep_cache_all_features_b2.npz, 12 horizons
1 .. 2048, 1,144 columns).  This runner refits them on a base-5 pooled design built the way the linear reruns of
2026-10-19 build theirs:

  * panel   the repo's own prep (experiments/run_geometry_local.prepare_full, the bank's builder; executor flags at
            their current defaults: CAUSAL_MODE_INFLATION, CAUSAL_NEUTRAL_FILL, CAUSAL_GUARD_MODE, the 2026-10-15 data
            fixes = panel v2), HAR ladder base 5 = the production ladder {1, 5, 25, 125, 625, 3125} (k in samples of
            the full 48-interval grid, overnight included);
  * series  every all_features series, the FOMC group included (author decision 2026-10-19; data fix 3c,
            FOMC_CALENDAR_FIX, on).  2026-10-19 (user): parked -- "every all_features series except the FOMC group
            (bucket ``all_features_nofomc``, registered here only)": the exclusion was an artifact of the base-2
            panel predating the FOMC columns;
  * E6      the trailing averages of the activity indicators (``<series>_active_ma_<k>``) are dropped;
  * E9      the calendar stops at the overnight indicator: DOW_0 .. DOW_4, hour, is_overnight are kept; is_open,
            is_close, the expiry / month / quarter columns and the HAR x open / close products are dropped.  On the
            13-bar sample the overnight indicator goes too (E9: it stays only if the overnight intervals stay in the
            training sample).
  Dropping columns after the build is the same as not building them: every prep step after the HAR / calendar
  build (scale guards, rolling robust scale, masked scaling) acts column by column.

Samples (E15 still open): ``48`` = every interval of the grid; ``13`` = the 13 regular-hours bars (end stamps
10:00 .. 16:00 ET), features still built on the full grid.  Window: the trees' printed 24,000 rows of the sample,
refit at every forecast row, the printed fixed hyperparameters (cluster/pooltree_lgbm_config.json lgbm_opt_00,
cluster/xgbtop_xgb_config.json xgb_opt_00, cluster/pooltree_rf400_config.json rf400_rf_opt_00), seed 42, as
src.unification._walk_tree / pooled_trees_walk.walk_rf build them.

Forecast rows: the paper scores the pooled trees on 16:00 ET rows only (writeup/make_rehaul_oos_table.py,
make_rehaul_results.plain_16, 2020-01-03 .. 2024-04-30), so the walk forecasts exactly the 16:00 rows of that span
(each on its own trailing window, so the rows are independent and can be split into pieces).

Modes
  design  --sample 48|13          build (cached) the base-5 panel, write <out>/design.json (names, counts) and
                                  <out>/rows.csv (row_id, t of the forecast rows)
  walk    --sample S --est E --set B [--part i/n] [--threads T]
                                  <out>/pieces/<est>_<set>_<i>of<n>.parquet (row_id, t, yhat, baseline, rv_raw)
  merge   --sample S              every complete arm -> <out>/forecasts/yhat_pooltree_<est>_<set>.parquet
  time    --sample S --est E --set B --n N   wall seconds of N refits (the cost projection)

Sets: baseline (HAR + calendar), lean_vol_vix (+ the six lean series' value and availability averages),
all_features (every column).  Env: B5_CACHE = folder of the prep cache (default results/pooled_trees_base5_cache).

Run:  python experiments/pooled_trees_base5.py design --sample 48
      python experiments/pooled_trees_base5.py walk --sample 48 --est lgbm --set all_features --threads 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DATE = "2026-10-19"
# 2026-10-19 (user): parked -- BUCKET = "all_features_nofomc" (FOMC now in the pooled model)
BUCKET = "all_features"
WINDOW = 24_000
SPAN = ("2020-01-03", "2024-04-30")
SETS = ("baseline", "lean_vol_vix", "all_features")
CONFIGS = {
    "lgbm": ROOT / "cluster" / "pooltree_lgbm_config.json",
    "xgb": ROOT / "cluster" / "xgbtop_xgb_config.json",
    "rf": ROOT / "cluster" / "pooltree_rf400_config.json",
}
CAL_KEEP = ["DOW_0", "DOW_1", "DOW_2", "DOW_3", "DOW_4", "hour", "is_overnight"]
RTH_MIN = {h * 60 + m for h in range(10, 17) for m in (0, 30) if (h, m) <= (16, 0)}
CACHE = Path(os.environ.get("B5_CACHE", str(ROOT / "results" / "pooled_trees_base5_cache")))


def out_dir(sample: str) -> Path:
    return ROOT / "results" / f"pooled_trees_base5_{DATE}_{sample}"


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def classify(name: str) -> tuple[str, str, int]:
    """src/unification.py _classify (same regexes): kind har | regime | value | indicator | calendar."""
    m = re.match(r"^har_ma_(\d+)_x_(open|close)$", name)
    if m:
        return "regime", "", int(m.group(1))
    m = re.match(r"^har_ma_(\d+)$", name)
    if m:
        return "har", "", int(m.group(1))
    m = re.match(r"^(.+)_(avail|active)_ma_(\d+)$", name)
    if m:
        return ("activity" if m.group(2) == "active" else "indicator"), m.group(1), int(m.group(3))
    m = re.match(r"^adj_(.+)_ma_(\d+)$", name)
    if m:
        return "value", m.group(1), int(m.group(2))
    return "calendar", "", 0


def full_panel():
    """The base-5 panel with every column the builder makes (FOMC excluded), plus t / raw RV (src.unification
    _load_panel's row discipline: loader rows after the RV dropna, from row BURNIN_ROWS on)."""
    from src.data import loading as LD

    # 2026-10-19 (user): parked -- the FOMC-free bucket. Was:
    # LD.SUBGROUPS[BUCKET] = [f for f in LD.SUBGROUPS["all_features"] if f not in LD.FOMC_FEATURES]
    assert LD.FOMC_CALENDAR_FIX, "data fix 3c (FOMC_CALENDAR_FIX) must be on"
    import run_geometry_local as R
    from src.features.extractors.har import resolve_har_lags

    R.CACHE_DIR = str(CACHE)
    CACHE.mkdir(parents=True, exist_ok=True)
    X, y, baselines, names = R.prepare_full(BUCKET, har_base=5)
    n = len(y)
    burn = resolve_har_lags()[-1]
    raw = LD.load_raw_data("data", allow_missing=True).dropna(subset=["RV"]).reset_index(drop=True)
    rv_raw = raw["RV"].to_numpy(np.float64)[burn:][:n]
    t = pd.to_datetime(raw["t"]).to_numpy()[burn:][:n]
    if len(t) != n:
        raise SystemExit(f"panel n={n} but {len(t)} loader rows after burn-in")
    with np.errstate(divide="ignore", invalid="ignore"):
        gate = float(np.nanmedian(np.abs(y - np.sqrt(rv_raw / baselines))))
    if gate != 0.0:
        raise SystemExit(f"alignment gate: median |y - sqrt(rv_raw / baseline)| = {gate} != 0")
    return np.asarray(X), np.asarray(y, np.float64), np.asarray(baselines, np.float64), list(names), t, rv_raw


def design_columns(names: list[str], sample: str) -> list[int]:
    """E6 / E9 cut of the full base-5 panel (see the docstring)."""
    cal = CAL_KEEP if sample == "48" else [c for c in CAL_KEEP if c != "is_overnight"]
    keep = []
    for j, nm in enumerate(names):
        kind = classify(nm)[0]
        if kind in ("har", "value", "indicator") or (kind == "calendar" and nm in cal):
            keep.append(j)
    return keep


def set_columns(names: list[str], name: str) -> np.ndarray:
    from src.data.loading import SUBGROUPS

    idx = []
    for j, nm in enumerate(names):
        kind, stem, _ = classify(nm)
        if kind in ("har", "calendar"):
            idx.append(j)
        elif name == "all_features" or (name == "lean_vol_vix" and stem in set(SUBGROUPS["lean_vol_vix"])):
            idx.append(j)
    return np.asarray(idx, dtype=np.int64)


def load(sample: str):
    """(X, y, baselines, names, t, rv_raw) of the sample's design; t naive ET."""
    X, y, b, names, t, rv = full_panel()
    keep = design_columns(names, sample)
    X = X[:, keep]
    names = [names[j] for j in keep]
    if sample == "13":
        tt = pd.DatetimeIndex(t)
        rows = np.flatnonzero(np.isin(tt.hour * 60 + tt.minute, list(RTH_MIN)))
        X, y, b, t, rv = X[rows], y[rows], b[rows], t[rows], rv[rows]
    return np.ascontiguousarray(X, dtype=np.float64), y, b, names, t, rv


def forecast_rows(t) -> np.ndarray:
    tt = pd.DatetimeIndex(t)
    at16 = (tt.hour == 16) & (tt.minute == 0)
    inspan = (tt.normalize() >= pd.Timestamp(SPAN[0])) & (tt.normalize() <= pd.Timestamp(SPAN[1]))
    rows = np.flatnonzero(at16 & inspan)
    if rows.min() < WINDOW:
        raise SystemExit(f"first forecast row {rows.min()} inside the {WINDOW}-row window")
    return rows


def model_factory(est: str, threads: int):
    cfg = json.loads(CONFIGS[est].read_text(encoding="utf-8"))
    params = dict(cfg.get("params") or cfg.get("rf_params") or {})
    if est == "lgbm":
        import lightgbm as lgb

        return cfg, lambda: lgb.LGBMRegressor(**params, num_threads=threads, random_state=42, verbosity=-1)
    if est == "xgb":
        import xgboost as xgb

        return cfg, lambda: xgb.XGBRegressor(**params, n_jobs=threads, random_state=42, verbosity=0)
    from sklearn.ensemble import RandomForestRegressor

    return cfg, lambda: RandomForestRegressor(**params, random_state=42, n_jobs=threads)


def fit_predict(make, est, Xw, yw, x1) -> float:
    m = make()
    m.fit(Xw, yw)
    if est == "rf":
        m.set_params(n_jobs=1)  # pooled_trees_walk.walk_rf: a one-thread predict sums in tree order
    return float(m.predict(x1)[0])


def cmd_design(a) -> int:
    if a.sample == "48":
        # the base-5 pooled panel BEFORE the E6 / E9 cut (the structure the paper's design formula and
        # writeup/make_rehaul_data_methods.py describe): names and row count only
        _, y0, _, names0, _, _ = full_panel()
        od0 = out_dir("48")
        od0.mkdir(parents=True, exist_ok=True)
        (od0 / "panel_full.json").write_text(
            json.dumps({"bucket": BUCKET, "har_base": 5, "n_rows": int(len(y0)), "names": names0}, indent=0),
            encoding="utf-8",
        )
    X, y, b, names, t, rv = load(a.sample)
    rows = forecast_rows(t)
    od = out_dir(a.sample)
    od.mkdir(parents=True, exist_ok=True)
    kinds: dict[str, int] = {}
    for nm in names:
        k = classify(nm)[0]
        kinds[k] = kinds.get(k, 0) + 1
    rungs = sorted({classify(nm)[2] for nm in names if classify(nm)[0] == "har"})
    series = sorted({classify(nm)[1] for nm in names if classify(nm)[0] == "value"})
    tt = pd.DatetimeIndex(t)
    d = {
        "sample": a.sample,
        "n_rows": int(len(y)),
        "n_cols": len(names),
        "kinds": kinds,
        "har_rungs": rungs,
        "n_series": len(series),
        "series": series,
        "calendar": [nm for nm in names if classify(nm)[0] == "calendar"],
        "sets": {s: int(len(set_columns(names, s))) for s in SETS},
        "window_rows": WINDOW,
        "first_row_t": str(tt[0]),
        "forecast_rows": int(len(rows)),
        "forecast_span": [str(tt[rows[0]]), str(tt[rows[-1]])],
        "names": names,
    }
    (od / "design.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
    pd.DataFrame({"row_id": rows, "t": tt[rows].astype(str)}).to_csv(od / "rows.csv", index=False)
    say(f"design {a.sample}: {len(y)} rows x {len(names)} columns {kinds}; rungs {rungs}; sets {d['sets']}; "
        f"{len(rows)} forecast rows {d['forecast_span']}")
    return 0


def cmd_walk(a) -> int:
    X, y, b, names, t, rv = load(a.sample)
    rows = forecast_rows(t)
    i, n = (int(v) for v in a.part.split("/"))
    mine = rows[(i - 1) :: n] if n > 1 else rows
    if a.limit:
        mine = mine[: a.limit]
    cols = set_columns(names, a.set)
    F = np.ascontiguousarray(X[:, cols])
    cfg, make = model_factory(a.est, a.threads)
    od = out_dir(a.sample) / "pieces"
    od.mkdir(parents=True, exist_ok=True)
    tag = f"{a.est}_{a.set}_{i}of{n}" + (f"_first{a.limit}" if a.limit else "")
    path = od / f"{tag}.parquet"
    done = pd.read_parquet(path) if path.is_file() else pd.DataFrame(columns=["row_id", "yhat"])
    have = set(done["row_id"].astype(int).tolist())
    say(f"walk {a.sample} {a.est} {a.set}: {F.shape[1]} columns, {len(mine)} rows (part {i}/{n}), "
        f"{len(have)} on disk, {cfg['name']} {cfg['sha']}, threads {a.threads}")
    recs = done.to_dict("records")
    t0 = time.time()
    k = 0
    tt = pd.DatetimeIndex(t)
    for r in mine:
        if int(r) in have:
            continue
        yh = fit_predict(make, a.est, F[r - WINDOW : r], y[r - WINDOW : r], F[r : r + 1])
        recs.append({"row_id": int(r), "t": str(tt[r]), "yhat": yh, "baseline": float(b[r]), "rv_raw": float(rv[r])})
        k += 1
        if k % 25 == 0 or k == len(mine) - len(have):
            pd.DataFrame(recs).to_parquet(path.with_suffix(".part"), index=False)
            os.replace(path.with_suffix(".part"), path)
            say(f"  {k} new rows, {(time.time() - t0) / k:.2f} s a row")
    pd.DataFrame(recs).to_parquet(path.with_suffix(".part"), index=False)
    os.replace(path.with_suffix(".part"), path)
    meta = {"sample": a.sample, "est": a.est, "set": a.set, "part": a.part, "n_cols": int(F.shape[1]),
            "config": cfg["name"], "sha": cfg["sha"], "window": WINDOW, "threads": a.threads,
            "rows": int(len(mine)), "wall_sec": round(time.time() - t0, 1)}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    say(f"DONE {tag}: {k} rows in {meta['wall_sec']} s -> {path}")
    return 0


def cmd_time(a) -> int:
    X, y, b, names, t, rv = load(a.sample)
    rows = forecast_rows(t)
    F = np.ascontiguousarray(X[:, set_columns(names, a.set)])
    _, make = model_factory(a.est, a.threads)
    t0 = time.time()
    for r in rows[: a.n]:
        fit_predict(make, a.est, F[r - WINDOW : r], y[r - WINDOW : r], F[r : r + 1])
    sec = (time.time() - t0) / a.n
    say(f"TIME {a.sample} {a.est} {a.set} ({F.shape[1]} columns, {a.threads} threads): {sec:.2f} s a refit; "
        f"{len(rows)} rows -> {sec * len(rows) / 3600:.2f} h")
    return 0


def cmd_merge(a) -> int:
    od = out_dir(a.sample)
    want = pd.read_csv(od / "rows.csv")
    (od / "forecasts").mkdir(exist_ok=True)
    for est in CONFIGS:
        for s in SETS:
            parts = sorted((od / "pieces").glob(f"{est}_{s}_*of*.parquet"))
            parts = [p for p in parts if "_first" not in p.name]
            if not parts:
                continue
            df = pd.concat([pd.read_parquet(p) for p in parts]).drop_duplicates("row_id").sort_values("row_id")
            if not np.array_equal(df["row_id"].to_numpy(), want["row_id"].to_numpy()):
                say(f"{est} {s}: {len(df)} of {len(want)} rows, incomplete -- not merged")
                continue
            tt = pd.to_datetime(df["t"]).dt.tz_localize("America/New_York").dt.tz_convert("UTC")
            outp = pd.DataFrame({"t": tt.to_numpy(), "yhat": df["yhat"].to_numpy(), "baseline": df["baseline"].to_numpy(),
                                 "rv_raw": df["rv_raw"].to_numpy()})
            outp["t"] = pd.to_datetime(outp["t"], utc=True)
            outp.to_parquet(od / "forecasts" / f"yhat_pooltree_{est}_{s}.parquet", index=False)
            say(f"merged {est} {s}: {len(outp)} rows")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("design", "walk", "merge", "time"))
    ap.add_argument("--sample", required=True, choices=("48", "13"))
    ap.add_argument("--est", choices=sorted(CONFIGS))
    ap.add_argument("--set", choices=SETS)
    ap.add_argument("--part", default="1/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    return {"design": cmd_design, "walk": cmd_walk, "merge": cmd_merge, "time": cmd_time}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
