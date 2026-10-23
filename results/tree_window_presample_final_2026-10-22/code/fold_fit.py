"""Trees' pre-sample window check (scratch driver, 2026-10-22): one (window, fold, estimator, predictor set).

Final-design rerun (results/tree_window_presample_final_2026-10-22): the earlier check's driver
(results/tree_window_presample_2026-10-22/code/fold_fit.py) with two provenance fields added to the record
(code_commit, design_source, from the environment); the fit and the criteria are unchanged.  Designs: the window
sweep's caches built on 9d3fc05 (e34 e0f3a7a + --first-fold: LATE_START_WARMUP, UNOBSERVED_NEUTRAL,
EXCLUDE_FOMC_RELEASE on; UNOBSERVED_NEUTRAL is a module default of src/backtest/executor.py and so not in the
cache's config, which equals the earlier one's).  The earlier text follows.

Fits the frozen configuration once on the fold's fit rows of the window's fixed design (e34 45183a0:
LATE_START_WARMUP on, EXCLUDE_FOMC_RELEASE on; 564 / 12 columns), predicts the fold's 3,250 validation rows and
records validation MSE on the adjusted scale (the linear pre-tune's criterion) and validation QLIKE (plain =
pred_adj^2 x B against the bar's RV, rows with RV > 0).  Model construction and fit/predict are
experiments/pooled_trees_rerun_13.py's model_factory and fit_predict, imported unchanged (its N_COLS = 576 check is
not used: the column list is asserted against the w1000 nofomcrel run's frozen_penalties.json instead).

Fold rows as experiments/pooled_linear_rerun_2026_10_19_13.py st_pretune / st_select: S = the design row of the
first 2012-01-03 bar, PRE_HALO = 4 x 3,250 + 325 + W; fold f: fit [S - PRE_HALO + 3250 f, + W),
val [S - 13,000 + 3250 f, + 3,250).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PPD, FOLD_N, EMB, N_FOLDS = 13, 3250, 325, 4
FIRST_DAY = "2012-01-03"
FP_REL = "results/pooled_linear_rerun_2026-10-22_13_2012_w1000_latestart_nofomcrel/frozen_penalties.json"
EX_FLAGS = {"DROP_ACTIVITY_MA": True, "CALENDAR_THROUGH_OVERNIGHT": True, "DROP_OVERNIGHT_INDICATOR": True,
            "LEAN_GLOBAL_LAGS": True, "LATE_START_WARMUP": True}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def fold_rows(S: int, Wr: int, f: int) -> tuple[list[int], list[int]]:
    halo = N_FOLDS * FOLD_N + EMB + Wr
    fit = [S - halo + f * FOLD_N, S - halo + f * FOLD_N + Wr]
    val = [S - halo + Wr + EMB + f * FOLD_N, S - halo + Wr + EMB + (f + 1) * FOLD_N]
    return fit, val


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--design", required=True, help="a design_<set> directory")
    ap.add_argument("--window-days", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--est", choices=("lgbm", "xgb", "rf"), required=True)
    ap.add_argument("--set", choices=("baseline", "all_features"), required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    repo, dz, out = Path(a.repo), Path(a.design), Path(a.out)
    sys.path.insert(0, str(repo / "experiments"))
    import pooled_trees_rerun_13 as PT

    assert PT.ROOT.resolve() == repo.resolve(), (PT.ROOT, repo)
    Wr = a.window_days * PPD
    meta = json.loads((dz / "design.json").read_text())
    cfg = meta["config"]
    assert meta["bucket"] == a.set, meta["bucket"]
    assert cfg["sample"] == "rth13" and cfg["periods_per_day"] == PPD and cfg["har_base"] == "5", cfg
    assert cfg["window"] == Wr and cfg["drop_fomc"] is False and cfg["ex_flags"] == EX_FLAGS, cfg
    assert "keep_fomc_release" not in cfg, cfg
    fp = json.loads((repo / FP_REL).read_text())
    names = meta["names"]
    assert names == fp["designs"][a.set]["columns"], "columns differ from the w1000 nofomcrel run"
    assert len(names) == {"baseline": 12, "all_features": 564}[a.set], len(names)

    X = np.load(dz / "X.npy", mmap_mode="r")
    y = np.load(dz / "y.npy")
    b = np.load(dz / "b_oos.npy")
    d = pd.DatetimeIndex(pd.read_csv(dz / "dates_oos.csv", parse_dates=["date"])["date"])
    assert X.shape == tuple(meta["shape"]) and X.shape[1] == len(names)
    assert len(y) == X.shape[0] == Wr + len(d) and len(b) == len(d), (X.shape, len(y), len(d), len(b))
    S = Wr + int(np.flatnonzero(d >= pd.Timestamp(FIRST_DAY))[0])
    fit, val = fold_rows(S, Wr, a.fold)
    assert fit[0] >= 0, f"fold {a.fold} does not fit at W = {Wr}: fit {fit}"
    assert val[0] - Wr >= 0, ("validation rows before the first OOS row", val, Wr)
    lin = fp["info"]["folds"][a.fold]
    # validation dates are the linear run's for every window; at W = 1,000 the rows are the recorded ones exactly
    assert str(d[val[0] - Wr]) == lin["val_first"] and str(d[val[1] - 1 - Wr]) == lin["val_last"], (
        str(d[val[0] - Wr]), str(d[val[1] - 1 - Wr]), lin)
    if a.window_days == fp["info"]["window_days"]:
        assert fit == lin["fit"] and val == lin["val"], (fit, val, lin)
    f0, f1 = fit
    v0, v1 = val
    Xf = np.ascontiguousarray(X[f0:f1], dtype=np.float64)
    Xv = np.ascontiguousarray(X[v0:v1], dtype=np.float64)
    yf, yv = y[f0:f1], y[v0:v1]
    assert np.isfinite(Xf).all() and np.isfinite(Xv).all() and np.isfinite(yf).all() and np.isfinite(yv).all()

    tcfg, make = PT.model_factory(a.est, a.threads)
    t0, c0 = time.time(), time.process_time()
    pred = PT.fit_predict(make, a.est, Xf, yf, Xv)
    wall, cpu = time.time() - t0, time.process_time() - c0
    assert pred.shape == yv.shape and np.isfinite(pred).all()

    idx = np.arange(v0, v1) - Wr
    dates, B = d[idx], b[idx]
    plain = pred**2 * B
    rv = PT.rv_bars().reindex(dates).to_numpy(float)
    true_raw = yv**2 * B
    okr = np.isfinite(rv) & (rv > 0)
    ok = okr & np.isfinite(plain) & (plain > 0)
    r = rv[ok] / plain[ok]
    qlike = float(np.mean(r - np.log(r) - 1.0))
    mse_adj = float(np.mean((pred - yv) ** 2))
    tag = f"w{a.window_days}_f{a.fold}_{a.est}_{a.set}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(row=np.arange(v0, v1), date=dates, pred_adj=pred, true_adj=yv, B=B, plain=plain,
                      rv_actual=rv)).to_parquet(out / f"{tag}.parquet", index=False)
    rec = dict(
        window_days=a.window_days, window_rows=Wr, fold=a.fold, est=a.est, predictor_set=a.set,
        n_columns=len(names), fit=fit, val=val, val_first=str(dates[0]), val_last=str(dates[-1]), S=S,
        n_val=int(v1 - v0), mse_adj=mse_adj, qlike=qlike, n_qlike=int(ok.sum()), n_rv_nonpos=int((~okr).sum()),
        mse_plain_e10=float(np.mean((rv[okr] - plain[okr]) ** 2) * 1e10),
        target_rv_max_rel=float(np.max(np.abs(true_raw[okr] / rv[okr] - 1.0))) if okr.any() else None,
        fit_wall_sec=round(wall, 3), fit_cpu_sec=round(cpu, 3), threads=a.threads,
        config=tcfg["name"], config_sha=tcfg["sha"], params=tcfg.get("params"),
        config_file_sha256=PT.sha256(PT.CONFIGS[a.est]), design_config=cfg,
        x_md5=md5(dz / "X.npy"), y_md5=md5(dz / "y.npy"), host=platform.node(), cpu=PT.cpu_model(),
        versions=PT.versions(), slurm_job=os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "")),
        slurm_task=os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        code_commit=os.environ.get("CODE_COMMIT", ""), design_source=os.environ.get("DESIGN_SOURCE", ""),
    )
    (out / f"{tag}.json").write_text(json.dumps(rec, indent=1))
    print(f"{tag}: fit {fit} val {val} ({rec['val_first']} .. {rec['val_last']}); mse_adj {mse_adj:.6f} "
          f"qlike {qlike:.6f} (n {rec['n_qlike']}); fit+predict {wall:.1f} s wall, {cpu:.1f} s CPU, "
          f"{a.threads} threads", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
