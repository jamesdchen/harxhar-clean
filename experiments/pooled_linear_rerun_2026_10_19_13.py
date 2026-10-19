"""The pooled linear models rerun on the 13 regular-hours bars, penalties frozen (2026-10-19).

Author decisions (2026-10-19; notes E6, E9, E15, E26, E28, N18, S4a-S4c): one pooled model over the 13 bars with
end stamps 10:00 .. 16:00 ET; the HAR components and every trailing average on the full 48-interval grid (lag scope
global), then the rows restricted; base-2 HAR ladder 1 .. 2048 intervals; window 2,000 trading days = 26,000 rows;
every data-fix flag of src/backtest/executor.py at its default (on); no FOMC series; no trailing averages of the
activity indicators (DROP_ACTIVITY_MA); calendar = DOW_0 .. DOW_4 and hour (CALENDAR_THROUGH_OVERNIGHT +
DROP_OVERNIGHT_INDICATOR); one penalty a model, picked once on the pre-sample and frozen over the evaluation span.
The sibling run on all 48 intervals is branch insurance-selling-2026-09-19-pooled-linear-48.

Runner: experiments/pool_linear_pretune.py (the spec's estimator section, specs/causal_tune_linear.py) with CFG
below; this script only orchestrates it and scores its tables.

Stages  (python experiments/pooled_linear_rerun_2026_10_19_13.py <stage> [--jobs N])
  design    the two design caches (HAR benchmark set = baseline, all features), _work/design_<set>/
  pretune   4 folds x 250 sessions (3,250 rows) tiling the 13,000 rows before the first 2020-01-03 row, each fit on
            the 26,000 rows before a 25-session (325-row) embargo; every grid penalty (ridge 1e-2 .. 1e3, lasso and
            elastic net 1e-6 .. 1e-2, one value a decade; elastic net l1_ratio 0.5); penalty factor r on the HAR
            benchmark block (r = 0.01 on all features, as the printed arms; r = 1 on the HAR benchmark set, where
            every column is in that block)
  select    the penalty with the lowest mean validation MSE (adjusted scale, all 13 bars, 4 folds pooled) ->
            frozen_penalties.json, pretune_mse.csv
  walk      every model over the evaluation span with --sched const:<frozen penalty> (the OLS: min-norm LS), from
            the re-choice boundary at or before the first 2020-01-03 row to the end of the data; --chunks K splits
            a walk at 250-solve boundaries (each chunk with a 26,000-row halo reproduces the whole walk)
  score     forecasts.parquet (every bar of the sample), scores.csv (16:00 rows: the 15:30 forecast of the closing
            half hour on the common span of tab:rh_oos, QLIKE / MSE / R2_OOS on actual RV as
            experiments/rolling_eval_2026_10_15.py), dm.csv (against the new OLS; against the printed rows when their
            forecast files are present), scores_all_bars.csv
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "pooled_linear_rerun_2026-10-19_13"
WORK = OUT / "_work"
RUNNER = REPO / "experiments" / "pool_linear_pretune.py"
CFG = [
    "--sample",
    "rth13",
    "--har-base",
    "2",
    "--drop-fomc",
    "--ex-flag",
    "DROP_ACTIVITY_MA=1",
    "--ex-flag",
    "CALENDAR_THROUGH_OVERNIGHT=1",
    "--ex-flag",
    "DROP_OVERNIGHT_INDICATOR=1",
    "--ex-flag",
    "LEAN_GLOBAL_LAGS=1",  # memory only: the same design bit for bit (checked on the sentiment bucket)
]
SETS = ("baseline", "all_features")
EST = {
    "ridge": "ridge",
    "lasso": "reclasso",
    "enet": "reclasticnet",
}  # paper name -> runner name
R_BB = {
    "baseline": 1.0,
    "all_features": 0.01,
}  # penalty factor on the HAR benchmark block
PPD, TW_DAYS, TUNE_PER = 13, 2000, 250
W = TW_DAYS * PPD  # 26,000 rows
PRE_FOLDS, PRE_FOLD_SESS, PRE_EMBARGO_SESS = 4, 250, 25
PRE_HALO = PRE_FOLDS * PRE_FOLD_SESS * PPD + PRE_EMBARGO_SESS * PPD + W  # 39,325
FIRST_DAY = "2020-01-03"
PRINTED = REPO / "results" / "rolling_eval_2026-10-15" / "metrics.csv"


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(args: list[str], log: Path) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    t0 = time.time()
    with open(log, "w") as fh:
        r = subprocess.run(
            [sys.executable, "-u", str(RUNNER), *args, *CFG],
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=REPO,
        )
    if r.returncode:
        raise RuntimeError(f"{' '.join(args[:3])} failed, see {log}")
    return time.time() - t0


def pool(jobs: list[tuple[list[str], Path]], n: int) -> None:
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        futs = {ex.submit(run, a, lg): (a, lg) for a, lg in jobs}
        for f in cf.as_completed(futs):
            _, lg = futs[f]
            say(f"done {lg.parent.name}/{lg.name} in {f.result():.0f} s")


def design_dir(s: str) -> Path:
    return WORK / f"design_{s}"


def oos_dates(s: str = "baseline") -> pd.Series:
    return pd.read_csv(design_dir(s) / "dates_oos.csv", parse_dates=["date"])["date"]


def first_row() -> int:
    """Design row of the first bar of the first evaluation day (the pre-tune's start row)."""
    d = oos_dates()
    return W + int(np.flatnonzero(d >= FIRST_DAY)[0])


def walk_start() -> int:
    """The re-choice boundary (250 solves from the first OOS row) at or before first_row()."""
    return (first_row() - W) // TUNE_PER * TUNE_PER + W


# ============================================================================ stages
def st_design(a) -> None:
    pool(
        [
            (
                ["design", "--bucket", s, "--out", str(design_dir(s))],
                WORK / "logs" / f"design_{s}.log",
            )
            for s in SETS
        ],
        a.jobs,
    )
    d0 = oos_dates("baseline")
    for s in SETS[1:]:
        assert oos_dates(s).equals(d0), (
            s
        )  # every set shares the row index (impute_indicate)


def pre_dir(s: str, m: str) -> Path:
    return WORK / "pretune" / s / m


def st_pretune(a) -> None:
    S = first_row()
    jobs = []
    for s in SETS:
        for m, e in EST.items():
            o = pre_dir(s, m)
            args = ["pretune", "--bucket", s, "--est", e, "--r", str(R_BB[s])]
            args += ["--start", str(S), "--halo", str(PRE_HALO), "--out", str(o)]
            args += ["--design", str(design_dir(s))]
            jobs.append((args, o / "run.log"))
    say(f"pretune: start row {S}, halo {PRE_HALO}, {len(jobs)} jobs")
    pool(jobs, a.jobs)


def st_select(a) -> None:
    rows, frozen, designs = [], {}, {}
    for s in SETS:
        meta = json.loads((design_dir(s) / "design.json").read_text())
        names = meta["names"]
        bb = [
            n for n in names if n.startswith("har_ma_") or not re.search(r"_ma_\d+$", n)
        ]
        designs[s] = dict(
            n_columns=len(names),
            n_har_benchmark_block=len(bb),
            penalty_factor_on_har_benchmark_block=R_BB[s],
            columns=names,
            rows=meta["shape"][0],
        )
        for m in EST:
            z = np.load(pre_dir(s, m) / "fold.npz")
            res, grid = z["resid"], z["grid"]  # (folds, grid, rows)
            mse_f = (res**2).mean(axis=2)  # (folds, grid)
            crit = (res**2).mean(
                axis=(0, 2)
            )  # equal fold sizes: the mean of the fold means
            k = int(np.argmin(crit))
            frozen.setdefault(m, {})[s] = float(grid[k])
            for j, al in enumerate(grid):
                rows.append(
                    dict(
                        model=m,
                        predictor_set=s,
                        alpha=float(al),
                        **{
                            f"mse_fold{f}": float(mse_f[f, j])
                            for f in range(len(mse_f))
                        },
                        mse_mean=float(crit[j]),
                        picked=j == k,
                        edge=k in (0, len(grid) - 1),
                    )
                )
    frozen["ols"] = {"baseline": None}
    pd.DataFrame(rows).to_csv(OUT / "pretune_mse.csv", index=False)
    S = first_row()
    d = oos_dates()
    info = dict(
        rule="one penalty a model, the lowest mean validation MSE (adjusted scale, all 13 bars) over 4 pre-sample "
        "folds; frozen over the evaluation span",
        sample="13 regular-hours bars, end stamps 10:00-16:00 ET; HAR components and trailing averages on the full "
        "48-interval grid",
        window_rows=W,
        window_days=TW_DAYS,
        har_ladder=[2**i for i in range(12)],
        folds=[
            dict(
                fit=[S - PRE_HALO + f * 3250, S - PRE_HALO + f * 3250 + W],
                val=[
                    S - PRE_HALO + W + 325 + f * 3250,
                    S - PRE_HALO + W + 325 + (f + 1) * 3250,
                ],
                val_first=str(d[S - PRE_HALO + W + 325 + f * 3250 - W]),
                val_last=str(d[S - PRE_HALO + W + 325 + (f + 1) * 3250 - 1 - W]),
            )
            for f in range(PRE_FOLDS)
        ],
        grids=dict(
            ridge="1e-2 .. 1e3", lasso="1e-6 .. 1e-2", enet="1e-6 .. 1e-2, l1_ratio 0.5"
        ),
        runner_cfg=CFG,
    )
    out = dict(
        **{m: frozen[m] for m in ("ridge", "lasso", "enet", "ols")},
        designs=designs,
        info=info,
    )
    (OUT / "frozen_penalties.json").write_text(json.dumps(out, indent=1))
    say(f"frozen penalties: { {m: frozen[m] for m in frozen} }")


def walk_dir(s: str, m: str, c: int | None = None) -> Path:
    base = WORK / "walk" / s / m
    return base if c is None else base / f"c{c}"


def walk_jobs(
    chunks: dict[tuple[str, str], int], only: list[str] | None = None
) -> list[tuple[list[str], Path, tuple]]:
    fp = OUT / "frozen_penalties.json"
    fr = json.loads(fp.read_text()) if fp.exists() else {}  # the OLS walk needs none
    S0 = walk_start()
    n = len(oos_dates()) + W  # design rows
    jobs = []
    models = [(s, m) for s in SETS for m in EST] + [("baseline", "ols")]
    for s, m in models:
        if only and f"{s}/{m}" not in only:
            continue
        k = chunks.get((s, m), 1)
        blocks = -(-(n - S0) // TUNE_PER)
        per = -(-blocks // k)
        for c in range(k):
            lo = S0 + c * per * TUNE_PER
            hi = min(n, S0 + (c + 1) * per * TUNE_PER)
            if lo >= n:
                break
            o = walk_dir(s, m, c if k > 1 else None)
            args = ["walk", "--bucket", s, "--est", EST.get(m, "ols")]
            if m != "ols":
                args += ["--r", str(R_BB[s]), "--sched", f"const:{fr[m][s]!r}"]
            args += [
                "--start",
                str(lo),
                "--end",
                str(-1 if hi == n else hi),
                "--halo",
                str(W),
            ]
            args += ["--out", str(o), "--design", str(design_dir(s))]
            jobs.append((args, o / "run.log", (s, m, c)))
    return jobs


def st_walk(a) -> None:
    chunks = {}
    for spec in a.chunks:  # set/model=K
        sm, k = spec.split("=")
        s, m = sm.split("/")
        chunks[(s, m)] = int(k)
    jobs = walk_jobs(chunks, a.only)
    if a.end_row:  # timing probe: a short span from the walk start
        jobs = [
            (
                j[0][: j[0].index("--end") + 1]
                + [str(a.end_row)]
                + j[0][j[0].index("--end") + 2 :],
                j[1],
                j[2],
            )
            for j in jobs
            if j[2][2] == 0
        ]
    say(f"walks: start row {walk_start()} ({len(jobs)} jobs)")
    pool([(j[0], j[1]) for j in jobs], a.jobs)


def walk_table(s: str, m: str) -> pd.DataFrame:
    """The walk's table, chunks stitched in order (each chunk's rows are its emit range)."""
    base = walk_dir(s, m)
    parts = sorted(
        base.glob("c*/causal_tune_linear/*/*/results.csv"),
        key=lambda p: int(p.parts[-5][1:]),
    )
    if not parts:
        parts = list(base.glob("causal_tune_linear/*/*/results.csv"))
    assert parts, base
    r = pd.concat(
        [pd.read_csv(p, parse_dates=["date"]) for p in parts], ignore_index=True
    )
    assert r["date"].is_monotonic_increasing and r["date"].is_unique, (s, m)
    return r


def st_score(a) -> None:
    sys.path.insert(0, str(REPO / "experiments"))
    sys.path.insert(0, str(REPO / "notebooks"))
    import rolling_eval_2026_10_15 as RE

    d = oos_dates()
    b_oos = np.load(design_dir("baseline") / "b_oos.npy")
    B = pd.Series(b_oos, index=pd.DatetimeIndex(d))
    core = pd.read_parquet(
        REPO / "data" / "core_stats.parquet", columns=["endbartime", "sumret2"]
    )
    rv_bar = pd.Series(
        core["sumret2"].to_numpy(float),
        index=pd.DatetimeIndex(pd.to_datetime(core["endbartime"])),
    )
    rv_bar = rv_bar[~rv_bar.index.duplicated()]
    fr = json.loads((OUT / "frozen_penalties.json").read_text())
    models = [(s, m) for s in SETS for m in EST] + [("baseline", "ols")]
    long, plain16 = [], {}
    for s, m in models:
        r = walk_table(s, m)
        b = B.reindex(pd.DatetimeIndex(r["date"])).to_numpy()
        assert np.isfinite(b).all()
        ok = r["true_adj"].to_numpy() > 0
        assert np.allclose(
            r["true_raw"][ok] / r["true_adj"][ok] ** 2, b[ok], rtol=1e-9
        ), (s, m)
        f = pd.DataFrame(
            dict(
                series_id=f"pooled_linear_rerun_13/{m}_{s}",
                model=m,
                predictor_set=s,
                alpha=np.nan if m == "ols" else fr[m][s],
                date=r["date"],
                pred_adj=r["pred_adj"],
                true_adj=r["true_adj"],
                B=b,
                plain=r["pred_adj"].to_numpy() ** 2 * b,
                pred_raw_duan=r["pred_raw"],
                true_raw=r["true_raw"],
                rv_actual=rv_bar.reindex(pd.DatetimeIndex(r["date"])).to_numpy(),
            )
        )
        long.append(f)
        k16 = (f["date"].dt.hour == 16).to_numpy()
        plain16[(s, m)] = pd.Series(
            f["plain"].to_numpy()[k16],
            index=pd.DatetimeIndex(f["date"][k16].dt.normalize()),
        )
    fc = pd.concat(long, ignore_index=True)
    fc.to_parquet(OUT / "forecasts.parquet", index=False)
    # ---- the closing half hour on the common span of tab:rh_oos
    hist = RE.history()
    cs = pd.read_csv(
        REPO / "results" / "rolling_eval_2026-10-15" / "common_span.csv"
    ).iloc[0]
    days = hist.index[(hist.index >= cs["first"]) & (hist.index <= cs["last"])]
    for p in plain16.values():
        days = days.intersection(p.index[np.isfinite(p.to_numpy())])
    bench = RE.ct_benchmark(hist, days)
    rv = hist.reindex(days)
    printed = pd.read_csv(PRINTED)
    printed = printed[printed["span"] == "common"].set_index("key")
    pkey = {(s, m): f"poolpt_{m}_{s}" for s in SETS for m in EST} | {
        ("baseline", "ols"): "a0"
    }
    rows, dms = [], []
    ols = plain16[("baseline", "ols")].reindex(days)
    for (s, m), p in plain16.items():
        p = p.reindex(days)
        mt = RE.metrics(rv, p, bench)
        pk = pkey[(s, m)]
        pr = printed.loc[pk] if pk in printed.index else None
        row = dict(
            model=m,
            predictor_set=s,
            alpha=np.nan if m == "ols" else fr[m][s],
            first=str(days[0].date()),
            last=str(days[-1].date()),
            n_days=len(days),
            qlike=mt["qlike"],
            mse_e10=mt["mse"] * 1e10,
            r2_oos_pct=mt["r2_oos_pct"],
            printed_key=pk,
            printed_qlike=np.nan if pr is None else pr["qlike"],
            printed_mse_e10=np.nan if pr is None else pr["mse_e10"],
            printed_r2_oos_pct=np.nan if pr is None else pr["r2_oos_pct"],
            printed_n=np.nan if pr is None else pr["n"],
        )
        rows.append(row)
        if m != "ols":
            dms.append(
                dict(
                    model=m,
                    predictor_set=s,
                    against="ols_rerun_13",
                    **RE.dm_pair(rv, p, ols),
                )
            )
        # against the printed row, when its forecast file is in the repo
        pth = RE.SPXW / ("yhat_a0.parquet" if pk == "a0" else f"yhat_{pk}.parquet")
        if pth.exists():
            q = RE.scored(pk)["plain"].reindex(days)
            ok = np.isfinite(q.to_numpy())
            dms.append(
                dict(
                    model=m,
                    predictor_set=s,
                    against=f"printed:{pk}",
                    **RE.dm_pair(rv[ok], p[ok], q[ok]),
                )
            )
        else:
            dms.append(
                dict(
                    model=m,
                    predictor_set=s,
                    against=f"printed:{pk}",
                    note=f"{pth.relative_to(REPO)} not in the repo",
                )
            )
    sc = pd.DataFrame(rows)
    sc.to_csv(OUT / "scores.csv", index=False)
    pd.DataFrame(dms).to_csv(OUT / "dm.csv", index=False)
    # ---- every bar of the sample (secondary): actual RV of the bar, finite positive prints, common days
    ab = []
    for (s, m), f in fc.groupby(["predictor_set", "model"], sort=False):
        g = f[f["date"].dt.normalize().isin(days) & (f["rv_actual"] > 0)]
        q = RE.qlike_daily(g["rv_actual"].to_numpy(float), g["plain"].to_numpy(float))
        ab.append(
            dict(
                model=m,
                predictor_set=s,
                n_bars=len(g),
                qlike=float(q.mean()),
                mse_e10=float(((g["rv_actual"] - g["plain"]) ** 2).mean() * 1e10),
            )
        )
    pd.DataFrame(ab).to_csv(OUT / "scores_all_bars.csv", index=False)
    say(f"scored {len(days)} days\n{sc.to_string(index=False)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("design", "pretune", "select", "walk", "score"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument(
        "--chunks",
        action="append",
        default=[],
        help="set/model=K, e.g. all_features/lasso=4",
    )
    ap.add_argument("--only", action="append", default=[], help="set/model")
    ap.add_argument(
        "--end-row",
        type=int,
        default=0,
        help="timing probe: end the walks at this design row",
    )
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {
        "design": st_design,
        "pretune": st_pretune,
        "select": st_select,
        "walk": st_walk,
        "score": st_score,
    }[a.stage](a)
    say(f"host {platform.node()}, numpy {np.__version__}")


if __name__ == "__main__":
    main()
