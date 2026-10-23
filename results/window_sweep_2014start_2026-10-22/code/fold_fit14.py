"""Trees' pre-test fold fit for the 2014-start window sweep: one (window, fold, estimator, predictor set).

The driver of results/tree_window_presample_final_2026-10-22/code/fold_fit.py (model_factory and fit_predict of
experiments/pooled_trees_rerun_13.py imported unchanged, the frozen configurations, validation MSE on the adjusted
scale and QLIKE) with the first evaluation day 2014-01-02 instead of 2012-01-03: S = the design row of the first
2014-01-02 bar; fold f: fit [S - PRE_HALO + 3250 f, + W), val [S - 13,000 + 3250 f, + 3,250), PRE_HALO = 4 x 3,250 +
325 + W -- the rows of experiments/pooled_linear_rerun_2026_10_19_13.py st_pretune with --first-day 2014-01-02
(--first-fold only drops folds, the rows of the others are the same).  The validation dates are asserted equal to
folds14.json (computed from the design's dates with the same formula, W-independent).  The column list is asserted
against the w1000 nofomcrel run's frozen_penalties.json (564 / 12 names).
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
FIRST_DAY = "2014-01-02"
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
    ap.add_argument("--folds-json", required=True)
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
    folds = json.loads(Path(a.folds_json).read_text())

    X = np.load(dz / "X.npy", mmap_mode="r")
    y = np.load(dz / "y.npy")
    b = np.load(dz / "b_oos.npy")
    d = pd.DatetimeIndex(pd.read_csv(dz / "dates_oos.csv", parse_dates=["date"])["date"])
    assert X.shape == tuple(meta["shape"]) and X.shape[1] == len(names)
    assert len(y) == X.shape[0] == Wr + len(d) and len(b) == len(d), (X.shape, len(y), len(d), len(b))
    S = Wr + int(np.flatnonzero(d >= pd.Timestamp(FIRST_DAY))[0])
    assert S == folds["S"], (S, folds["S"])
    fit, val = fold_rows(S, Wr, a.fold)
    assert fit[0] >= 0, f"fold {a.fold} does not fit at W = {Wr}: fit {fit}"
    assert val[0] - Wr >= 0, ("validation rows before the first OOS row", val, Wr)
    exp = folds["folds"][a.fold]
    assert val == exp["val"] and str(d[val[0] - Wr]) == exp["val_first"] and str(d[val[1] - 1 - Wr]) == exp["val_last"], (
        val, str(d[val[0] - Wr]), str(d[val[1] - 1 - Wr]), exp)
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
        window_days=a.window_days, window_rows=Wr, first_day=FIRST_DAY, fold=a.fold, est=a.est, predictor_set=a.set,
        n_columns=len(names), fit=fit, val=val, val_first=str(dates[0]), val_last=str(dates[-1]), S=S,
        fit_first=str(d[f0 - Wr]) if f0 >= Wr else f"design row {f0} (before the first OOS row)",
        n_val=int(v1 - v0), mse_adj=mse_adj, qlike=qlike, n_qlike=int(ok.sum()), n_rv_nonpos=int((~okr).sum()),
        mse_plain_e10=float(np.mean((rv[okr] - plain[okr]) ** 2) * 1e10),
        target_rv_max_rel=float(np.max(np.abs(true_raw[okr] / rv[okr] - 1.0))) if okr.any() else None,
        fit_wall_sec=round(wall, 3), fit_cpu_sec=round(cpu, 3), threads=a.threads,
        config=tcfg["name"], config_sha=tcfg["sha"], params=tcfg.get("params"),
        config_file_sha256=PT.sha256(PT.CONFIGS[a.est]), design_config=cfg,
        x_md5=md5(dz / "X.npy"), y_md5=md5(dz / "y.npy"), host=platform.node(), cpu=PT.cpu_model(),
        versions=PT.versions(), slurm_job=os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "")),
        slurm_task=os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        code_commit=os.environ.get("CODE_COMMIT", ""), design_source=str(dz),
    )
    (out / f"{tag}.json").write_text(json.dumps(rec, indent=1))
    print(f"{tag}: fit {fit} val {val} ({rec['val_first']} .. {rec['val_last']}); mse_adj {mse_adj:.6f} "
          f"qlike {qlike:.6f} (n {rec['n_qlike']}); fit+predict {wall:.1f} s wall, {cpu:.1f} s CPU, "
          f"{a.threads} threads", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
