"""E32 gate: the every-session refits of the closing-interval tree rungs that refit every 10 sessions.

Note E32 (writeup/rehaul/10_appendix.tex): the closing-interval tree specifications that refit every 10 trading
days (REFIT_EVERY = 10, specs/causal_tune_trees.py) are to refit every session, with each specification's
hyperparameter rule unchanged.  The 16:00 campaign of 2026-09-29/30 (writeup/CAMPAIGN_16H_2026-09-29.md, the
masked ladder of results/trees_mask_1600/) ran every ten-day rung a second time at REFIT_EVERY = 1:

  rung   refit       table (results/spxw_pnl/yhat_<key>.parquet)   every-session twin
  T10    10          subtree_<model>_<bucket>                       T1   subtree_daily_<bucket>_<model>
  RS10   10          subtree_tuned_<bucket>_<model>                 RS1  subtree_tuned_daily_<bucket>_<model>
  RS10q  10          subtree_tunedq_<bucket>_<model>                RS1q subtree_tunedq_daily_<bucket>_<model>

This script checks, without refitting the walks, that each twin is the ten-day rung with only the refit cadence
changed:

  twins   every ten-day table against its twin: the same stamps, baseline and rv_raw; yhat bit for bit on the
          ten-day refit rows (forecast row % 10 == 0, where both fit on the same window [t - W, t)), and the count
          of the other rows that differ (the ten-day rung forecasts them with a model up to 9 sessions old).
  picks   the random-search rungs' tuning traces (tune_trace_bar1600.csv under
          results/linear_subsection_trees_tuned_mask/{rs10,rs1}/): every column but tune_sec equal, so the
          hyperparameters in force at every row are the same at both cadences.
  refit   fresh fits of the shipped configuration (the dicts of specs/causal_tune_trees.py, read by ast; the
          spec's leaf-minimum scaling; one thread; the per-window mask src.models.window_mask.window_keep) on the
          printed 16:00 all_features design (results/close_studies_2026-10-03/forecasts/
          design_bar1600_all_features.parquet, the de-duplicated design the trees were fitted on), at a few
          forecast rows: the cadence-1 fit (window ending at the row) against the printed T1 forecast and the
          cadence-10 fit (window ending at the row's ten-day anchor) against the printed T10 forecast, bit for bit.
          XGBoost grows a different model under MSVC's standard library (Windows) than under libstdc++ (Linux,
          where the tables were made), so its refit rows are only meaningful on Linux.  The exported design was
          captured on an AVX-512 Xeon (results/close_studies_2026-10-03/design_hashes.json) while the fleet built
          its design on epyc-7513 (AVX2); the two differ at ~1e-11 (results/trees_mask_1600/SUMMARY.md, CPU
          class), enough to move a LightGBM histogram bin or the window mask, so a refit row can differ for that
          reason alone: the refit part is evidence, not a gate (reported, never asserted).

Outputs (--out, default results/e32_cadence_gate/): twins.csv, picks.csv, refit_<platform>.csv.  Exit status 1
when the twins or picks gate fails.

Run:  python experiments/e32_cadence_gate.py [--data-root <checkout holding the untracked tables>]
          [--parts twins,picks,refit] [--models lgbm,rf,xgb] [--rows 0,3,9,10,737,1468]
"""

from __future__ import annotations

import argparse
import ast
import os
import platform
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.models.window_mask import window_keep  # noqa: E402

TREE_SPEC = REPO / "specs" / "causal_tune_trees.py"
MODELS = ("lgbm", "xgb", "rf")
BUCKETS = ("baseline", "all_features", "live_feasible", "lean_vol_vix")
CADENCE = 10  # the ten-day rungs' REFIT_EVERY
W = 2000  # the per-bar window (TRAIN_WIN default of the spec), asserted against the design below
DESIGN = (
    Path("results")
    / "close_studies_2026-10-03"
    / "forecasts"
    / "design_bar1600_all_features.parquet"
)
DESIGN_META = ("row", "date", "forecast_row", "y", "baseline", "true_raw")
TUNED_ROOT = Path("results") / "linear_subsection_trees_tuned_mask"
MODEL_SET = Path("results") / "close_master_table" / "paper_model_set.csv"


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def spec_constants(names: tuple[str, ...]) -> dict:
    """The first top-level literal assignment (plain or annotated) of each name in specs/causal_tune_trees.py."""
    out: dict = {}
    for node in ast.parse(TREE_SPEC.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for tgt in targets:
            if isinstance(tgt, ast.Name) and tgt.id in names and tgt.id not in out:
                out[tgt.id] = ast.literal_eval(value)
    missing = set(names) - set(out)
    assert not missing, f"{TREE_SPEC.name} no longer assigns {sorted(missing)}"
    return out


def pairs() -> list[dict]:
    """Every (ten-day key, every-session twin) of the ladder; the RF dev configuration (rfdev) has T10 / T1 only."""
    out = []
    for b in BUCKETS:
        for m in (*MODELS, "rfdev"):
            out.append(
                dict(
                    rung="T10",
                    model=m,
                    bucket=b,
                    k10=f"subtree_{m}_{b}",
                    k1=f"subtree_daily_{b}_{m}",
                )
            )
            if m == "rfdev":
                continue
            for q in ("", "q"):
                out.append(
                    dict(
                        rung=f"RS10{q}",
                        model=m,
                        bucket=b,
                        k10=f"subtree_tuned{q}_{b}_{m}",
                        k1=f"subtree_tuned{q}_daily_{b}_{m}",
                    )
                )
    return out


def part_twins(root: Path) -> tuple[pd.DataFrame, bool]:
    spx = root / "results" / "spxw_pnl"
    ms = (
        set(pd.read_csv(root / MODEL_SET)["key"])
        if (root / MODEL_SET).is_file()
        else set()
    )
    rows, ok = [], True
    for p in pairs():
        f10, f1 = spx / f"yhat_{p['k10']}.parquet", spx / f"yhat_{p['k1']}.parquet"
        if not (f10.is_file() and f1.is_file()):
            if f10.is_file() or f1.is_file():  # one side only: a twin is missing
                rows.append({**p, "present": False, "ok": False})
                ok = False
            continue
        a, d = pd.read_parquet(f10), pd.read_parquet(f1)
        same_t = len(a) == len(d) and bool(
            (a["t"].to_numpy() == d["t"].to_numpy()).all()
        )
        on = np.arange(len(a)) % CADENCE == 0
        ya, yd = a["yhat"].to_numpy(), d["yhat"].to_numpy()
        r = dict(
            **p,
            present=True,
            in_model_set=p["k10"] in ms if ms else np.nan,
            twin_in_model_set=p["k1"] in ms if ms else np.nan,
            rows=len(a),
            same_stamps=same_t,
            same_baseline=bool(
                np.array_equal(a["baseline"].to_numpy(), d["baseline"].to_numpy())
            ),
            same_rv_raw=bool(
                np.array_equal(
                    a["rv_raw"].to_numpy(), d["rv_raw"].to_numpy(), equal_nan=True
                )
            ),
            refit_rows=int(on.sum()),
            refit_rows_bit_equal=int((ya[on] == yd[on]).sum()),
            other_rows=int((~on).sum()),
            other_rows_equal=int((ya[~on] == yd[~on]).sum()),
            other_rows_max_abs_diff=float(np.max(np.abs(ya[~on] - yd[~on]))),
        )
        r["ok"] = bool(
            same_t
            and r["same_baseline"]
            and r["same_rv_raw"]
            and r["refit_rows_bit_equal"] == r["refit_rows"]
        )
        ok &= r["ok"]
        rows.append(r)
    return pd.DataFrame(rows), ok


def part_picks(root: Path) -> tuple[pd.DataFrame, bool]:
    base = root / TUNED_ROOT
    rows: list[dict] = []
    ok = True
    for b in BUCKETS:
        for m in MODELS:
            rel = (
                Path(b)
                / "bar1600"
                / m
                / "tw2000"
                / "causal_tune_trees"
                / m
                / b
                / "tune_trace_bar1600.csv"
            )
            f10, f1 = base / "rs10" / rel, base / "rs1" / rel
            if not (f10.is_file() and f1.is_file()):
                continue
            a, d = pd.read_csv(f10), pd.read_csv(f1)
            diff = [c for c in a.columns if c not in d.columns or not a[c].equals(d[c])]
            same = len(a) == len(d) and set(diff) <= {"tune_sec"}
            rows.append(
                dict(
                    model=m,
                    bucket=b,
                    tuning_points=int(a["tune_idx"].nunique()),
                    trace_rows=len(a),
                    columns_differing=";".join(diff),
                    ok=same,
                )
            )
            ok &= same
    return pd.DataFrame(rows), ok


def make_model(model: str, c: dict):
    if model == "lgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            **c["lgbm"], num_threads=1, random_state=c["SEED"], verbosity=-1
        )
    if model == "xgb":
        import xgboost as xgb

        return xgb.XGBRegressor(
            **c["xgb"], n_jobs=1, random_state=c["SEED"], verbosity=0
        )
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(**c["rf"], random_state=c["SEED"], n_jobs=1)


def part_refit(root: Path, models: list[str], rows: list[int]) -> pd.DataFrame:
    k = spec_constants(("LGBM_PARAMS", "XGB_PARAMS", "RF_PARAMS", "SEED"))
    # the spec's leaf-minimum scaling for a W-row window (specs/causal_tune_trees.py, POOLED_WINDOW_ROWS = 24000)
    scale = W / 24000
    lgbm = dict(k["LGBM_PARAMS"])
    lgbm["min_child_samples"] = max(1, round(lgbm["min_child_samples"] * scale))
    xgbp = dict(k["XGB_PARAMS"])
    xgbp["min_child_weight"] = xgbp["min_child_weight"] * scale
    c = dict(lgbm=lgbm, xgb=xgbp, rf=dict(k["RF_PARAMS"]), SEED=k["SEED"])
    d = pd.read_parquet(root / DESIGN)
    cols = [x for x in d.columns if x not in DESIGN_META]
    X = np.ascontiguousarray(d[cols].to_numpy(dtype=np.float64))
    y = np.ascontiguousarray(d["y"].to_numpy(dtype=np.float64))
    fr = d["forecast_row"].to_numpy()
    assert len(cols) == 628 and fr[0] == -W and int((fr >= 0).sum()) == len(d) - W, (
        len(cols),
        fr[0],
    )
    spx = root / "results" / "spxw_pnl"
    linux = platform.system() == "Linux"
    out: list[dict] = []
    for m in models:
        import importlib

        lib = {"lgbm": "lightgbm", "xgb": "xgboost", "rf": "sklearn"}[m]
        ver = importlib.import_module(lib).__version__
        printed = {
            "T1": pd.read_parquet(spx / f"yhat_subtree_daily_all_features_{m}.parquet")[
                "yhat"
            ].to_numpy(),
            "T10": pd.read_parquet(spx / f"yhat_subtree_{m}_all_features.parquet")[
                "yhat"
            ].to_numpy(),
        }
        fits: dict[int, tuple] = {}  # anchor row -> (model, keep)
        for r in rows:
            for rung, anchor in (("T1", r), ("T10", r - r % CADENCE)):
                if anchor not in fits:
                    t = W + anchor  # design index of the anchor's forecast row
                    keep = window_keep(X[t - W : t])
                    a = time.time()
                    mdl = make_model(m, c)
                    mdl.fit(X[t - W : t][:, keep], y[t - W : t])
                    fits[anchor] = (mdl, keep, time.time() - a)
                mdl, keep, sec = fits[anchor]
                pred = float(mdl.predict(X[W + r : W + r + 1][:, keep])[0])
                pv = float(printed[rung][r])
                rec = dict(
                    model=m,
                    rung=rung,
                    row=r,
                    anchor=anchor,
                    kept=len(keep),
                    fresh=pred,
                    printed=pv,
                    bit_equal=pred == pv,
                    rel_diff=abs(pred - pv) / abs(pv),
                    fit_sec=round(sec, 3),
                    platform=platform.system(),
                    library=f"{lib} {ver}",
                    comparable=m != "xgb"
                    or linux,  # XGBoost off Linux grows another model
                )
                out.append(rec)
                say(
                    f"{m} {rung} row {r} (anchor {anchor}): fresh {pred:.17g} printed {pv:.17g} equal {pred == pv}"
                )
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        default=str(REPO),
        help="checkout holding results/ (untracked tables included)",
    )
    ap.add_argument("--parts", default="twins,picks,refit")
    ap.add_argument("--models", default="lgbm,rf,xgb")
    ap.add_argument("--rows", default="0,3,9,10,737,1468")
    ap.add_argument("--out", default=str(REPO / "results" / "e32_cadence_gate"))
    a = ap.parse_args()
    root, out = Path(a.data_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = {x.strip() for x in a.parts.split(",") if x.strip()}
    rc = 0
    if "twins" in parts:
        t, ok = part_twins(root)
        t.to_csv(out / "twins.csv", index=False)
        say(f"twins: {int(t['ok'].sum())}/{len(t)} pairs ok -> {out / 'twins.csv'}")
        rc |= 0 if ok else 1
    if "picks" in parts:
        p, ok = part_picks(root)
        p.to_csv(out / "picks.csv", index=False)
        say(
            f"picks: {int(p['ok'].sum()) if len(p) else 0}/{len(p)} traces ok -> {out / 'picks.csv'}"
        )
        rc |= 0 if ok else 1
    if "refit" in parts:
        models = [m for m in a.models.split(",") if m]
        rows = [int(x) for x in a.rows.split(",") if x]
        r = part_refit(root, models, rows)
        tag = platform.system().lower()
        r.to_csv(out / f"refit_{tag}.csv", index=False)
        say(
            f"refit: {int(r['bit_equal'].sum())}/{len(r)} bit-equal ({tag}) -> {out / f'refit_{tag}.csv'}"
        )
    say(f"exit {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
