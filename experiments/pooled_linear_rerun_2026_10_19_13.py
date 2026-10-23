"""The pooled linear models rerun on the 13 regular-hours bars, penalties frozen (2026-10-19).

Author decisions (2026-10-19; notes E6, E9, E15, E26, E28, N18, S4a-S4c): one pooled model over the 13 bars with
end stamps 10:00 .. 16:00 ET; the HAR components and every trailing average on the full 48-interval grid (lag scope
global), then the rows restricted; window 2,000 trading days = 26,000 rows; every data-fix flag of
src/backtest/executor.py and src/data/loading.py at its default (on); no trailing averages of the activity indicators
(DROP_ACTIVITY_MA); calendar = DOW_0 .. DOW_4 and hour (CALENDAR_THROUGH_OVERNIGHT +
DROP_OVERNIGHT_INDICATOR); one penalty a model, picked once on the pre-sample and frozen over the evaluation span.
The sibling run on all 48 intervals is branch insurance-selling-2026-09-19-pooled-linear-48.
2026-10-19 22:45 UTC (author, two corrections relayed by the main session): the HAR ladder is base 5, k in {1, 5, 25,
125, 625, 3125} intervals of the 48-interval grid ("we need base 5 across the entire paper"), for every model including
the OLS; the FOMC series stay in the full predictor set (src/data/loading.py FOMC_FEATURES, FOMC_CALENDAR_FIX on).
The first run (base 2, FOMC out) is parked in results/pooled_linear_rerun_2026-10-19_13/base2_superseded/.

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

2026-10-20 (user): --window-days N (default 2000) and --first-day D (default 2020-01-03) for the window check and the
2012 rerun; every count above scales with them (pre-tune: the 1,000 sessions before D, each fold fit on the N-day
window before its embargo). The score stage still scores the common span of tab:rh_oos; forecasts.parquet holds every
bar from the walk start to the end of the data.

2026-10-22 (user: "Fix the late start series problems."): --ex-flag NAME=0|1 adds executor flags to CFG (e.g.
LATE_START_WARMUP, on by default in src/backtest/executor.py), and --keep-fomc-release keeps the FOMC statement-bar
dummy in the full set although src/data/loading.py EXCLUDE_FOMC_RELEASE is on; check_design asserts the FOMC series
of that set (it asserted all four, which fails at EXCLUDE_FOMC_RELEASE = True).  Without either option every stage
is as before.

2026-10-22 (James: "add the KKT check flag"): --walk-kkt-reseed 0|1 passes the runner's option of that name to the walks
(src/models/reclasso_har.py WALK_KKT_RESEED, on by default: after every warm lasso / elastic-net update the KKT
conditions of the window's problem, a cold reseed where they fail); without it the walks run at the module's default.

2026-10-22 (James: "yes, adopt standardization and hold the pretune"): --standardize-penalty 0|1 passes the runner's
option of that name to the pretunes and the walks (pool_linear_pretune.py STANDARDIZE_PENALTY, on by default: the
penalties act on the window-standardized coefficients); without it both run at the runner's default.  The penalty grid
rule of the new specification (RULE_std.txt), applied when the pretunes ran with the flag on (their run.json) unless
--grid-extend 0|1 says otherwise: a selected penalty at either end of its grid adds the next decade in that direction
(a pretune of that value only, pretune/<set>/<model>/ext_<alpha>/), until the selection is interior; stops early when
the new end's criterion is within 1e-6 relative of the old end's ("flat") or after 6 decades ("cap"); every extension in
frozen_penalties.json info.grid_extensions and pretune_mse.csv.  A walk whose penalty is off the spec's grid gets it as
--grid-alphas.  With the flag off every stage is as before.
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
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "pooled_linear_rerun_2026-10-19_13"
WORK = OUT / "_work"
RUNNER = REPO / "experiments" / "pool_linear_pretune.py"
CFG = [
    "--sample",
    "rth13",
    "--har-base",
    "5",
    # 2026-10-19 (author, correction): parked -- base 2 and no FOMC. Was: "--har-base", "2", "--drop-fomc",
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
HAR_LADDER = [1, 5, 25, 125, 625, 3125]  # base 5 (author, 2026-10-19 correction)


KEEP_FOMC_RELEASE = False  # 2026-10-22: set by main from --keep-fomc-release
WALK_KKT = ""  # 2026-10-22: set by main from --walk-kkt-reseed ("" = the runner's default)
# 2026-10-22: set by main from --standardize-penalty ("" = the runner's default, on) and --grid-extend ("" = on iff
# the pretunes ran standardized); the grid rule's safeguards (RULE_std.txt)
STD = ""
GRID_EXT = ""
GRID_EXT_FLAT, GRID_EXT_CAP = 1e-6, 6


def design_features() -> tuple[list[str], list[str]]:
    """2026-10-22 (user): (the full set's raw series, its FOMC series) as src/data/loading.py builds them under
    EXCLUDE_FOMC_RELEASE, with fomc_release put back before fomc_day under --keep-fomc-release (as
    pool_linear_pretune.py --keep-fomc-release)."""
    from src.data import loading as LD

    feats = list(LD.ALL_FEATURES)
    if KEEP_FOMC_RELEASE and "fomc_release" not in feats:
        feats.insert(feats.index("fomc_day"), "fomc_release")
    return feats, [f for f in LD.FOMC_FEATURES if f in feats]


def check_design(s: str) -> list[str]:
    """The design's columns: the target's HAR components exactly at HAR_LADDER, calendar DOW_0 .. DOW_4 and hour,
    and on the full predictor set every FOMC series present (value and availability averages, no activity
    averages)."""
    # 2026-10-22 (user): parked -- every FOMC series asserted, before src/data/loading.py EXCLUDE_FOMC_RELEASE went
    # on (it fails at HEAD). The series are now design_features(): the flag's, or fomc_release kept. Was:
    # from src.data.loading import ALL_FEATURES, FOMC_FEATURES
    ALL_FEATURES, FOMC_FEATURES = design_features()

    names = json.loads((design_dir(s) / "design.json").read_text())["names"]
    har = [n for n in names if n.startswith("har_ma_")]
    assert har == [f"har_ma_{k}" for k in HAR_LADDER], har
    cal = [n for n in names if not re.search(r"_ma_\d+$", n)]
    assert cal == [f"DOW_{d}" for d in range(5)] + ["hour"], cal
    assert not any("_active_ma_" in n for n in names)
    if s == "all_features":
        assert set(FOMC_FEATURES) <= set(ALL_FEATURES)
        for f in FOMC_FEATURES:
            for k in HAR_LADDER:
                assert f"adj_{f}_ma_{k}" in names and f"{f}_avail_ma_{k}" in names, (
                    f,
                    k,
                )
        exog = {n for n in names if n.startswith("adj_")}
        assert exog == {f"adj_{f}_ma_{k}" for f in ALL_FEATURES for k in HAR_LADDER}, (
            sorted(exog)[:5]
        )
    else:
        assert len(names) == len(HAR_LADDER) + 6, names
    return names


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
    S = W + int(np.flatnonzero(d >= FIRST_DAY)[0])
    assert S >= PRE_HALO, (
        S,
        PRE_HALO,
    )  # 2026-10-20: the pre-tune folds fit inside the design
    return S


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
    for s in SETS:
        check_design(s)
    d0 = oos_dates("baseline")
    for s in SETS[1:]:
        assert oos_dates(s).equals(d0), (
            s
        )  # every set shares the row index (impute_indicate)


def pre_dir(s: str, m: str) -> Path:
    return WORK / "pretune" / s / m


def pretune_args(s: str, m: str, o: Path, S: int) -> list[str]:
    args = ["pretune", "--bucket", s, "--est", EST[m], "--r", str(R_BB[s])]
    args += ["--start", str(S), "--halo", str(PRE_HALO), "--out", str(o)]
    args += ["--design", str(design_dir(s))]
    if STD:  # 2026-10-22 (--standardize-penalty)
        args += ["--standardize-penalty", STD]
    return args


def runner_std_default() -> bool:
    """2026-10-22: pool_linear_pretune.STANDARDIZE_PENALTY's default (what a run without --standardize-penalty does)."""
    if str(RUNNER.parent) not in sys.path:
        sys.path.insert(0, str(RUNNER.parent))
    import pool_linear_pretune as PLP

    return bool(PLP.STANDARDIZE_PENALTY)


def std_effective() -> bool:
    return STD == "1" if STD else runner_std_default()


# ---------------------------------------------------------------------------- the grid rule (2026-10-22)
def pre_folds(s: str, m: str) -> list[Path]:
    """The pretune fold files of set s / model m: the spec grid's, then every grid extension's."""
    base = pre_dir(s, m)
    return [base / "fold.npz"] + sorted(base.glob("ext_*/fold.npz"))


def pick(s: str, m: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(grid ascending, resid (folds, grid, rows), mean validation MSE, argmin) over every pretuned penalty of set s /
    model m (the spec grid and its extensions; without extensions exactly the spec grid's arrays)."""
    zs = [np.load(p) for p in pre_folds(s, m)]
    for z in zs[1:]:  # the same folds
        assert np.array_equal(z["fit"], zs[0]["fit"]) and np.array_equal(
            z["val"], zs[0]["val"]
        ), (s, m)
    if (
        len(zs) == 1
    ):  # the spec grid only: its arrays as stored (the reductions below then run as before)
        grid, res = zs[0]["grid"], zs[0]["resid"]
    else:
        grid = np.concatenate([z["grid"] for z in zs])
        o = np.argsort(grid, kind="stable")
        grid = grid[o]
        # (C order: a fancy-indexed res[:, o] is laid out grid-major and its mean sums in another order)
        res = np.ascontiguousarray(
            np.concatenate([z["resid"] for z in zs], axis=1)[:, o]
        )
    assert len(np.unique(grid)) == len(grid), (s, m, grid)
    crit = (res**2).mean(axis=(0, 2))  # equal fold sizes: the mean of the fold means
    return grid, res, crit, int(np.argmin(crit))


def std_pretuned(s: str, m: str) -> bool:
    """The pretune of set s / model m ran with STANDARDIZE_PENALTY on (its run.json; absent before the flag)."""
    return bool(
        json.loads((pre_dir(s, m) / "run.json").read_text()).get(
            "standardize_penalty", False
        )
    )


def grid_ext_on(s: str, m: str) -> bool:
    return GRID_EXT == "1" if GRID_EXT else std_pretuned(s, m)


def grid_state(s: str, m: str) -> dict:
    """The grid rule (RULE_std.txt) on set s / model m: status interior | extend (with the next alpha) | flat | cap,
    and the extensions made so far (ascending)."""
    base = np.load(pre_dir(s, m) / "fold.npz")["grid"]
    grid, _, crit, k = pick(s, m)
    added = [float(g) for g in grid if g < base.min() or g > base.max()]
    st: dict = dict(added=added, selected=float(grid[k]))
    if 0 < k < len(grid) - 1:
        return st | dict(status="interior")
    down = k == 0
    n_dir = sum((g < base.min()) if down else (g > base.max()) for g in added)
    end, nxt = (0, 1) if down else (len(grid) - 1, len(grid) - 2)
    if n_dir and abs(crit[end] - crit[nxt]) <= GRID_EXT_FLAT * crit[nxt]:
        return st | dict(status="flat", direction="down" if down else "up")
    if n_dir >= GRID_EXT_CAP:
        return st | dict(status="cap", direction="down" if down else "up")
    e = int(round(float(np.log10(grid[end])))) + (-1 if down else 1)
    return st | dict(
        status="extend", direction="down" if down else "up", alpha=float(f"1e{e}")
    )


def st_pretune(a) -> None:
    for s in SETS:
        check_design(s)
    S = first_row()
    jobs = []
    for s in SETS:
        for m in EST:
            o = pre_dir(s, m)
            jobs.append((pretune_args(s, m, o, S), o / "run.log"))
    say(f"pretune: start row {S}, halo {PRE_HALO}, {len(jobs)} jobs")
    pool(jobs, a.jobs)
    # 2026-10-22: the grid rule -- one decade a round for every arm selected at a grid end, until none is
    for rnd in range(1, 2 * GRID_EXT_CAP + 2):
        ext = []
        for s in SETS:
            for m in EST:
                if not grid_ext_on(s, m):
                    continue
                st = grid_state(s, m)
                if st["status"] != "extend":
                    continue
                o = pre_dir(s, m) / f"ext_{st['alpha']!r}"
                args = pretune_args(s, m, o, S) + ["--grid-alphas", repr(st["alpha"])]
                ext.append((args, o / "run.log"))
                say(
                    f"grid rule round {rnd}: {s}/{m} selected {st['selected']!r} at the "
                    f"{'lower' if st['direction'] == 'down' else 'upper'} end, adding {st['alpha']!r}"
                )
        if not ext:
            break
        pool(ext, a.jobs)


def st_select(a) -> None:
    rows, frozen, designs = [], {}, {}
    exts: dict = {}  # 2026-10-22: the grid rule's record
    off_grid: dict = {}
    std_on = {(s, m): std_pretuned(s, m) for s in SETS for m in EST}
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
            # 2026-10-22: pick() -- the spec grid and the grid rule's extensions (none: the arrays as before)
            # parked -- was: z = np.load(pre_dir(s, m) / "fold.npz"); res, grid = z["resid"], z["grid"];
            # crit = (res**2).mean(axis=(0, 2)); k = int(np.argmin(crit))
            grid, res, crit, k = pick(s, m)  # res (folds, grid, rows)
            mse_f = (res**2).mean(axis=2)  # (folds, grid)
            frozen.setdefault(m, {})[s] = float(grid[k])
            if grid_ext_on(s, m):
                st = grid_state(s, m)
                assert st["status"] != "extend", (s, m, st, "run the pretune stage")
                exts.setdefault(m, {})[s] = st
            base = np.load(pre_dir(s, m) / "fold.npz")["grid"]
            if float(grid[k]) not in [float(g) for g in base]:
                off_grid.setdefault(m, {})[s] = float(grid[k])
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
                        picked_at_grid_end=k in (0, len(grid) - 1),
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
        har_ladder=HAR_LADDER,
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
    if (
        any(std_on.values()) or exts
    ):  # 2026-10-22: recorded only with the flag on (else the file as before)
        assert len(set(std_on.values())) == 1, std_on
        info |= dict(
            standardize_penalty=all(std_on.values()),
            std_rule="RULE_std.txt: penalties on the window-standardized coefficients s_j b_j (s_j = np.std, ddof 0, "
            "over the fit's rows); grid rule: a selection at a grid end adds the next decade in that direction "
            f"until interior (stop: flat within {GRID_EXT_FLAT} relative, or {GRID_EXT_CAP} decades)",
            grid_extensions=exts,
            off_spec_grid=off_grid,
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
    # 2026-10-22: the walks standardize iff the penalties were selected standardized
    off_grid = fr.get("info", {}).get("off_spec_grid", {})
    if fr and any(m in fr for m in EST):
        sel_std = bool(fr.get("info", {}).get("standardize_penalty", False))
        assert sel_std == std_effective(), (
            f"frozen penalties selected with standardize_penalty={sel_std}, walks would run with "
            f"{std_effective()} (--standardize-penalty)"
        )
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
                if s in off_grid.get(
                    m, {}
                ):  # 2026-10-22: a grid-rule penalty, off the spec's grid
                    args += ["--grid-alphas", repr(off_grid[m][s])]
                if STD:  # 2026-10-22 (--standardize-penalty)
                    args += ["--standardize-penalty", STD]
            args += [
                "--start",
                str(lo),
                "--end",
                str(-1 if hi == n else hi),
                "--halo",
                str(W),
            ]
            args += ["--out", str(o), "--design", str(design_dir(s))]
            if WALK_KKT:  # 2026-10-22 (--walk-kkt-reseed)
                args += ["--walk-kkt-reseed", WALK_KKT]
            jobs.append((args, o / "run.log", (s, m, c)))
    return jobs


def st_walk(a) -> None:
    for s in SETS:
        check_design(s)
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
        # against the closing-interval OLS (the master table's DM benchmark), when its arm file is in the repo
        if RE.OLS_ARM.exists():
            q = RE.scored(RE.OLS)["plain"].reindex(days)
            ok = np.isfinite(q.to_numpy())
            dms.append(
                dict(
                    model=m,
                    predictor_set=s,
                    against=RE.OLS,
                    **RE.dm_pair(rv[ok], p[ok], q[ok]),
                )
            )
        else:
            dms.append(
                dict(
                    model=m,
                    predictor_set=s,
                    against=RE.OLS,
                    note=f"{RE.OLS_ARM.relative_to(REPO)} not in the repo",
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
    global OUT, WORK, TW_DAYS, W, PRE_HALO, FIRST_DAY, KEEP_FOMC_RELEASE, WALK_KKT, STD
    global GRID_EXT
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
    # 2026-10-20 (user): another output directory, e.g. a scratch reproduction; default the committed one
    ap.add_argument("--out", default="")
    # 2026-10-20 (user): the window in trading days and the first evaluation day; the defaults are the committed run.
    # The pre-tune folds tile the 1,000 sessions before --first-day, each fit on the window before its embargo.
    ap.add_argument("--window-days", type=int, default=TW_DAYS)
    ap.add_argument("--first-day", default=FIRST_DAY)
    # 2026-10-22 (user, the late-start rerun): more executor flags for every stage (NAME=0|1, repeatable, after CFG's
    # own, e.g. LATE_START_WARMUP=1), and fomc_release kept in the full set although src/data/loading.py
    # EXCLUDE_FOMC_RELEASE is on (pool_linear_pretune.py --keep-fomc-release); the defaults add nothing to CFG
    ap.add_argument("--ex-flag", action="append", default=[])
    ap.add_argument("--keep-fomc-release", action="store_true")
    # 2026-10-22 (James: "add the KKT check flag"): the walks' WALK_KKT_RESEED (pool_linear_pretune.py --walk-kkt-reseed)
    ap.add_argument("--walk-kkt-reseed", default="", choices=("", "0", "1"))
    # 2026-10-22 (James: "yes, adopt standardization and hold the pretune"): the pretunes' and walks'
    # STANDARDIZE_PENALTY (pool_linear_pretune.py --standardize-penalty), and the grid rule (default: on iff the
    # pretunes ran standardized)
    ap.add_argument("--standardize-penalty", default="", choices=("", "0", "1"))
    ap.add_argument("--grid-extend", default="", choices=("", "0", "1"))
    a = ap.parse_args()
    WALK_KKT = a.walk_kkt_reseed  # 2026-10-22
    STD, GRID_EXT = a.standardize_penalty, a.grid_extend  # 2026-10-22
    if a.out:
        OUT = Path(a.out).resolve()
        WORK = OUT / "_work"
    if a.window_days != TW_DAYS:
        TW_DAYS = a.window_days
        W = TW_DAYS * PPD
        PRE_HALO = PRE_FOLDS * PRE_FOLD_SESS * PPD + PRE_EMBARGO_SESS * PPD + W
        CFG.extend(["--window-days", str(TW_DAYS)])
    for kv in a.ex_flag:  # 2026-10-22
        CFG.extend(["--ex-flag", kv])
    if a.keep_fomc_release:  # 2026-10-22
        KEEP_FOMC_RELEASE = True
        CFG.append("--keep-fomc-release")
    FIRST_DAY = a.first_day
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
