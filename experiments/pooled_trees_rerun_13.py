"""Pooled trees on the design of the pooled linear 13-bar rerun (2026-10-20).

The pooled LightGBM, XGBoost and random forest fitted on exactly the X, y, B and rows that Table 1's pooled linear
13-bar rerun uses (experiments/pooled_linear_rerun_2026_10_19_13.py, results/pooled_linear_rerun_2026-10-19_13):
nothing is rebuilt. The design caches come from

    python experiments/pooled_linear_rerun_2026_10_19_13.py design --jobs 2 [--window-days N] --out <DIR>

-> <DIR>/_work/design_baseline/ (HAR benchmark, 12 columns) and <DIR>/_work/design_all_features/ (576 columns).
Each holds the prescaled X (W + len(dates_oos) rows), y, the column names, b_oos and dates_oos.csv; OOS row i is
design row W + i (experiments/pool_linear_pretune.py, design mode).

  * window  W = window_days x 13 rows (read from the design cache). The forecast of design row r is fitted on rows
            [r - W, r): the rows the linear walk uses at r.
  * models  the frozen configs cluster/pooltree_lgbm_config.json, cluster/xgbtop_xgb_config.json,
            cluster/pooltree_rf400_config.json, constructed as experiments/pooled_trees_base5.py does (seed 42;
            LightGBM num_threads = T, XGBoost n_jobs = T, random forest n_jobs = T to fit and n_jobs = 1 to predict).
  * refit   bar = a refit at every row (the linear models' cadence); day = a refit at the first row of each trading
            day, whose fit forecasts every row of that day (rows grouped by date: some days have fewer than 13).
  * span    --first-day .. --last-day (default the last design row) on dates_oos; --part i/n = the i-th of n
            contiguous blocks of trading days (tasks are independent, no warm state).

Modes
  walk    --est E --set S --refit bar|day [--first-day D] [--last-day D] [--part i/n] [--threads T]
          -> <OUT>/pieces/<est>_<set>_<refit>_<first>_<last>_<i>of<n>.parquet (model, predictor_set, date, pred_adj,
          true_adj, B, plain = pred_adj^2 x B, rv_actual, fit_row) + .json meta; resumable (checkpoints every
          --ckpt-sec seconds; rows already in the piece are skipped)
  merge   --refit R [--first-day D] [--last-day D] -> <OUT>/forecasts.parquet with the columns of the linear
          forecasts.parquet that Table 1 reads (model, predictor_set, date, pred_adj, true_adj, B, plain, rv_actual)
  time    --est E.. --set S.. --k K: seconds a refit of each arm, projected CPU-hours and wall hours of each --span
          under each cadence; appends <OUT>/timing/sec_refit.csv
  tasks   --sec-table CSV --target-hours H -> a task file, one line "<est> <set> <refit> <first> <last> <i>/<n>"

Run:  python experiments/pooled_trees_rerun_13.py walk --design results/pooled_trees_rerun13_2026-10-20 \
          --est lgbm --set all_features --refit day --first-day 2020-01-03 --last-day 2020-01-07 --threads 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# LightGBM's sklearn wrapper names the columns of an unnamed array; predicting on an unnamed array then warns
warnings.filterwarnings("ignore", message="X does not have valid feature names")
PPD = 13
SETS = ("baseline", "all_features")
N_COLS = {"baseline": 12, "all_features": 576}
CONFIGS = {
    "lgbm": ROOT / "cluster" / "pooltree_lgbm_config.json",
    "xgb": ROOT / "cluster" / "xgbtop_xgb_config.json",
    "rf": ROOT / "cluster" / "pooltree_rf400_config.json",
}
COLS = ["model", "predictor_set", "date", "pred_adj", "true_adj", "B", "plain", "rv_actual"]
PIECE_COLS = COLS + ["fit_row"]
SPANS = ("2020-01-03:2024-04-30", "2012-01-03:2019-12-31")
DEFAULT_OUT = ROOT / "results" / "pooled_trees_rerun13_2026-10-20"
DEFAULT_DESIGN = DEFAULT_OUT  # the linear runner's design stage run with --out DEFAULT_OUT (its _work/ is gitignored)


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================ design
def design_dir(root: Path, s: str) -> Path:
    return root / "_work" / f"design_{s}"


def check(root: Path, s: str, window_days: int) -> dict:
    """The linear runner's check_design (HAR columns exactly har_ma_{1,5,...,3125}, calendar, every FOMC series),
    the column count, and the window of the cache against --window-days."""
    import pooled_linear_rerun_2026_10_19_13 as LR

    LR.WORK = root / "_work"
    names = LR.check_design(s)
    assert len(names) == N_COLS[s], (s, len(names))
    meta = json.loads((design_dir(root, s) / "design.json").read_text())
    W = int(meta["config"]["window"])
    assert meta["config"]["periods_per_day"] == PPD, meta["config"]
    assert W == window_days * PPD, (W, window_days)
    return meta


def oos_dates(root: Path, s: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.read_csv(design_dir(root, s) / "dates_oos.csv", parse_dates=["date"])["date"])


def span_rows(d: pd.DatetimeIndex, first: str, last: str | None) -> tuple[np.ndarray, str]:
    """OOS rows of the span (first .. last day, last = the last design row by default) and the last day's label."""
    day = d.normalize()
    last = last or str(day[-1].date())
    i = np.flatnonzero((day >= pd.Timestamp(first)) & (day <= pd.Timestamp(last)))
    assert len(i), (first, last)
    return i, last


def part_rows(d: pd.DatetimeIndex, rows: np.ndarray, part: str) -> np.ndarray:
    """The i-th of n contiguous blocks of trading days of the span."""
    i, n = (int(v) for v in part.split("/"))
    assert 1 <= i <= n, part
    day = d.normalize()[rows]
    blocks = np.array_split(day.unique(), n)
    return rows[day.isin(blocks[i - 1])]


def refits_of(d: pd.DatetimeIndex, rows: np.ndarray, refit: str) -> int:
    return len(rows) if refit == "bar" else int(d.normalize()[rows].nunique())


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def rv_bars() -> pd.Series:
    """data/core_stats.parquet sumret2 by bar end (naive ET), as the linear runner's score stage."""
    core = pd.read_parquet(ROOT / "data" / "core_stats.parquet", columns=["endbartime", "sumret2"])
    rv = pd.Series(core["sumret2"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(core["endbartime"])))
    return rv[~rv.index.duplicated()]


# ============================================================================ models
def model_factory(est: str, threads: int):
    """experiments/pooled_trees_base5.py model_factory (seed 42; T threads)."""
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


def fit_predict(make, est, Xw, yw, Xp) -> np.ndarray:
    m = make()
    m.fit(Xw, yw)
    if est == "rf":
        m.set_params(n_jobs=1)  # pooled_trees_walk.walk_rf: a one-thread predict sums in tree order
    return np.asarray(m.predict(Xp), dtype=np.float64)


def versions() -> dict:
    import lightgbm
    import pyarrow
    import scipy
    import sklearn
    import xgboost

    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, lightgbm=lightgbm.__version__, xgboost=xgboost.__version__,
                pyarrow=pyarrow.__version__)


def cpu_model() -> str:
    try:
        for ln in Path("/proc/cpuinfo").read_text().splitlines():
            if ln.startswith("model name"):
                return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


class Arm:
    """One (estimator, set) on the design: X rows [lo, hi) in memory, C-ordered (the trees bin the values, so the
    layout does not change a fit)."""

    def __init__(self, root: Path, s: str, window_days: int, lo: int = 0, hi: int | None = None):
        self.meta = check(root, s, window_days)
        self.W = window_days * PPD
        dz = design_dir(root, s)
        X = np.load(dz / "X.npy", mmap_mode="r")
        hi = len(X) if hi is None else hi
        self.lo = lo
        self.X = np.ascontiguousarray(X[lo:hi], dtype=np.float64)
        self.y = np.load(dz / "y.npy")
        self.b = np.load(dz / "b_oos.npy")
        self.d = oos_dates(root, s)
        assert len(self.y) == self.W + len(self.d) == self.meta["shape"][0] and len(self.b) == len(self.d)

    def rows(self, r0: int, r1: int) -> np.ndarray:
        return self.X[r0 - self.lo : r1 - self.lo]


# ============================================================================ walk
def piece_path(out: Path, est: str, s: str, refit: str, first: str, last: str, part: str) -> Path:
    i, n = part.split("/")
    return out / "pieces" / f"{est}_{s}_{refit}_{first}_{last}_{i}of{n}.parquet"


def cmd_walk(a) -> int:
    root, out = Path(a.design).resolve(), Path(a.out).resolve()
    d = oos_dates(root, a.set)
    span, last = span_rows(d, a.first_day, a.last_day)
    mine = part_rows(d, span, a.part)
    W = a.window_days * PPD
    path = piece_path(out, a.est, a.set, a.refit, a.first_day, last, a.part)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_p, ckpt_p = path.with_suffix(".json"), path.with_suffix(".ckpt.json")
    day = d.normalize()
    groups = [mine[day[mine] == g] for g in day[mine].unique()] if a.refit == "day" else [mine[[k]] for k in
                                                                                             range(len(mine))]
    done = pd.read_parquet(path) if path.is_file() else pd.DataFrame(columns=PIECE_COLS)
    have = set(pd.DatetimeIndex(done["date"]).tolist())
    if meta_p.is_file() and len(done) == len(mine):
        say(f"complete: {path.name}")
        return 0
    arm = Arm(root, a.set, a.window_days, lo=int(mine[0]), hi=W + int(mine[-1]) + 1)  # design rows W + mine[0] - W ..
    cfg, make = model_factory(a.est, a.threads)
    rv = rv_bars()
    prior = json.loads(ckpt_p.read_text()) if ckpt_p.is_file() else {"wall_sec": 0.0, "refits": 0}
    say(f"walk {a.est} {a.set} {a.refit} {a.first_day}..{last} part {a.part}: {len(mine)} rows "
        f"{d[mine[0]]} .. {d[mine[-1]]}, {len(groups)} refits, {len(have)} rows on disk; W {W}, "
        f"{arm.X.shape[1]} columns, {cfg['name']} {cfg['sha']}, threads {a.threads}")
    recs = [done] if len(done) else []
    t0, t_ck, k = time.time(), time.time(), 0

    def save(final: bool) -> dict:
        f = pd.concat(recs, ignore_index=True)[PIECE_COLS].sort_values("date", ignore_index=True)
        f.to_parquet(path.with_suffix(".part"), index=False)
        os.replace(path.with_suffix(".part"), path)
        st = {"wall_sec": prior["wall_sec"] + time.time() - t0, "refits": prior["refits"] + k}
        if not final:
            ckpt_p.write_text(json.dumps(st))
        return st

    for g in groups:
        if all(t in have for t in d[g]):
            continue
        r = W + g  # design rows
        r0 = int(r[0])  # the fit of the group's first row
        yh = fit_predict(make, a.est, arm.rows(r0 - W, r0), arm.y[r0 - W : r0], arm.rows(r0, int(r[-1]) + 1))
        b = arm.b[g]
        recs.append(pd.DataFrame(dict(model=a.est, predictor_set=a.set, date=d[g], pred_adj=yh, true_adj=arm.y[r],
                                      B=b, plain=yh**2 * b, rv_actual=rv.reindex(d[g]).to_numpy(),
                                      fit_row=np.full(len(g), r0 - W, dtype=np.int64))))
        k += 1
        if time.time() - t_ck > a.ckpt_sec:
            save(False)
            t_ck = time.time()
            say(f"  {k} refits, {(time.time() - t0) / k:.2f} s a refit, through {d[g[-1]]}")
    st = save(True)
    f = pd.read_parquet(path)
    assert pd.DatetimeIndex(f["date"]).equals(d[mine]), "piece rows"
    meta = dict(est=a.est, predictor_set=a.set, refit=a.refit, span=[a.first_day, last], part=a.part,
                block=[str(d[mine[0]]), str(d[mine[-1]])], rows=int(len(mine)), refits=int(st["refits"]),
                wall_sec=round(st["wall_sec"], 1), config=cfg["name"], sha=cfg["sha"], params=cfg.get("params"),
                config_file=str(CONFIGS[a.est].relative_to(ROOT)), config_file_sha256=sha256(CONFIGS[a.est]),
                window_rows=W, window_days=a.window_days, n_columns=int(arm.X.shape[1]), threads=a.threads,
                host=platform.node(), cpu=cpu_model(), versions=versions(), design=str(design_dir(root, a.set)),
                slurm_job=os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "")),
                slurm_task=os.environ.get("SLURM_ARRAY_TASK_ID", ""))
    meta_p.write_text(json.dumps(meta, indent=1))
    ckpt_p.unlink(missing_ok=True)
    say(f"DONE {path.name}: {k} refits in {time.time() - t0:.1f} s ({meta['wall_sec']} s in all)")
    return 0


# ============================================================================ merge
def cmd_merge(a) -> int:
    root, out = Path(a.design).resolve(), Path(a.out).resolve()
    frames, used = [], {}
    for est in a.est or list(CONFIGS):
        for s in a.set or list(SETS):
            d = oos_dates(root, s)
            span, last = span_rows(d, a.first_day, a.last_day)
            pat = re.compile(rf"^{est}_{s}_{a.refit}_{a.first_day}_{last}_(\d+)of(\d+)\.parquet$")
            by_n: dict[int, dict[int, Path]] = {}
            for p in sorted((out / "pieces").glob(f"{est}_{s}_{a.refit}_{a.first_day}_{last}_*of*.parquet")):
                m = pat.match(p.name)
                if m and p.with_suffix(".json").is_file():
                    by_n.setdefault(int(m.group(2)), {})[int(m.group(1))] = p
            full = {n: ps for n, ps in by_n.items() if set(ps) == set(range(1, n + 1))}
            if a.n:
                full = {n: ps for n, ps in full.items() if n == a.n}
            if not full:
                say(f"{est} {s}: no complete set of pieces ({ {n: sorted(ps) for n, ps in by_n.items()} })")
                if a.require_all:
                    raise SystemExit(1)
                continue
            assert len(full) == 1, f"{est} {s}: complete splits {sorted(full)}; pick one with --n"
            n, ps = next(iter(full.items()))
            f = pd.concat([pd.read_parquet(ps[i]) for i in range(1, n + 1)], ignore_index=True)
            f = f.sort_values("date", ignore_index=True)
            assert f["date"].is_unique and pd.DatetimeIndex(f["date"]).equals(d[span]), f"{est} {s}: span rows"
            num = f[["pred_adj", "true_adj", "B", "plain", "rv_actual"]].to_numpy(float)
            assert np.isfinite(num).all(), f"{est} {s}: non-finite values"
            assert np.allclose(f["plain"], f["pred_adj"] ** 2 * f["B"], rtol=1e-12, atol=0)
            frames.append(f[COLS])
            used[f"{est}/{s}"] = [p.name for p in ps.values()]
            say(f"{est} {s}: {len(f)} rows from {n} pieces")
    assert frames, "nothing to merge"
    fc = pd.concat(frames, ignore_index=True)
    fc["date"] = pd.to_datetime(fc["date"]).astype("datetime64[ns]")
    for c in COLS[3:]:
        fc[c] = fc[c].astype(np.float64)
    dest = Path(a.dest).resolve() if a.dest else out / "forecasts.parquet"
    fc.to_parquet(dest, index=False)
    dest.with_suffix(".json").write_text(json.dumps(dict(refit=a.refit, span=[a.first_day, a.last_day],
                                                         window_days=a.window_days, pieces=used), indent=1))
    say(f"merged {len(fc)} rows -> {dest}")
    return 0


# ============================================================================ time / tasks
def project(sec: float, threads: int, refits: int) -> tuple[float, float]:
    """(CPU-hours, wall hours of one task) of `refits` refits at `sec` seconds a refit on `threads` threads."""
    return sec * refits * threads / 3600, sec * refits / 3600


def cmd_time(a) -> int:
    root, out = Path(a.design).resolve(), Path(a.out).resolve()
    W = a.window_days * PPD
    recs = []
    for s in a.set or list(SETS):
        d = oos_dates(root, s)
        rows0, _ = span_rows(d, a.spans[0].split(":")[0], None)
        r = W + rows0[: a.k]
        arm = Arm(root, s, a.window_days, lo=int(r[0]) - W, hi=int(r[-1]) + 1)
        for est in a.est or list(CONFIGS):
            _, make = model_factory(est, a.threads)
            t0 = time.time()
            for rr in r:
                fit_predict(make, est, arm.rows(rr - W, rr), arm.y[rr - W : rr], arm.rows(rr, rr + 1))
            sec = (time.time() - t0) / len(r)
            recs.append(dict(est=est, set=s, n_columns=int(arm.X.shape[1]), window_rows=W, threads=a.threads,
                             k=len(r), sec_refit=sec, host=platform.node(), cpu=cpu_model()))
            say(f"TIME {est} {s} ({arm.X.shape[1]} columns, W {W}, {a.threads} threads): {sec:.2f} s a refit")
    tab = pd.DataFrame(recs)
    proj = []
    for t in recs:
        d = oos_dates(root, t["set"])
        for sp in a.spans:
            f, l = sp.split(":")
            rows, _ = span_rows(d, f, l)
            for refit in ("bar", "day"):
                n = refits_of(d, rows, refit)
                cpu, wall = project(t["sec_refit"], t["threads"], n)
                proj.append(dict(est=t["est"], set=t["set"], span=sp, refit=refit, refits=n, threads=t["threads"],
                                 sec_refit=t["sec_refit"], cpu_hours=cpu, wall_hours_one_task=wall))
    proj = pd.DataFrame(proj)
    (out / "timing").mkdir(parents=True, exist_ok=True)
    p = out / "timing" / "sec_refit.csv"
    tab.to_csv(p, mode="a", header=not p.is_file(), index=False)
    proj.to_csv(out / "timing" / f"projection_{platform.node()}_{a.threads}t.csv", index=False)
    pd.set_option("display.width", 200)
    print(tab.drop(columns=["cpu"]).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(proj.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return 0


def cmd_tasks(a) -> int:
    root = Path(a.design).resolve()
    sec = pd.concat([pd.read_csv(p) for p in a.sec_table], ignore_index=True)
    sec = sec[sec["window_rows"] == a.window_days * PPD]
    lines, info = [], []
    for sp in a.spans:
        first, last = sp.split(":")
        for refit in a.refit_list:
            for est in a.est or list(CONFIGS):
                for s in a.set or list(SETS):
                    t = sec[(sec["est"] == est) & (sec["set"] == s)]
                    assert len(t), f"no seconds a refit for {est} {s} at W {a.window_days * PPD}"
                    t = t.iloc[-1]
                    d = oos_dates(root, s)
                    rows, last_ = span_rows(d, first, last)
                    n_ref = refits_of(d, rows, refit)
                    n_days = int(d.normalize()[rows].nunique())
                    n = max(1, min(n_days, math.ceil(t["sec_refit"] * n_ref / 3600 / a.target_hours)))
                    lines += [f"{est} {s} {refit} {first} {last_} {i}/{n}" for i in range(1, n + 1)]
                    info.append(f"# {est} {s} {refit} {first}..{last_}: {n_ref} refits x {t['sec_refit']:.2f} s "
                                f"({int(t['threads'])} threads) -> {n} tasks of ~{t['sec_refit'] * n_ref / n / 3600:.2f} h")
    Path(a.taskfile).parent.mkdir(parents=True, exist_ok=True)
    Path(a.taskfile).write_text("\n".join(lines) + "\n")
    print("\n".join(info))
    say(f"{len(lines)} tasks -> {a.taskfile}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("walk", "merge", "time", "tasks"))
    ap.add_argument("--design", default=str(DEFAULT_DESIGN), help="the linear runner's --out directory")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--window-days", type=int, default=2000)
    ap.add_argument("--est", action="append", choices=sorted(CONFIGS))
    ap.add_argument("--set", action="append", choices=SETS)
    ap.add_argument("--refit", choices=("bar", "day"), default="day")
    ap.add_argument("--first-day", default="2020-01-03")
    ap.add_argument("--last-day", default=None, help="default: the last design row")
    ap.add_argument("--part", default="1/1")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    ap.add_argument("--ckpt-sec", type=float, default=300.0)
    # merge
    ap.add_argument("--n", type=int, default=0, help="merge: the split to use when several are complete")
    ap.add_argument("--dest", default="")
    ap.add_argument("--require-all", action="store_true")
    # time / tasks
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--span", action="append", dest="spans", help="first:last (time, tasks)")
    ap.add_argument("--sec-table", action="append", default=[])
    ap.add_argument("--target-hours", type=float, default=3.0)
    ap.add_argument("--refits", default="day", help="tasks: bar, day or bar,day")
    ap.add_argument("--taskfile", default="cluster/pooltree_rerun13_tasks.txt")
    a = ap.parse_args()
    if a.mode == "walk":
        assert a.est and a.set and len(a.est) == len(a.set) == 1, "walk: one --est and one --set"
        a.est, a.set = a.est[0], a.set[0]
    if a.mode == "time" and not a.spans:
        a.spans = ["2020-01-03:2024-04-30", "2012-01-03:2024-04-30"]
    a.spans = a.spans or list(SPANS)
    a.refit_list = a.refits.split(",")
    return {"walk": cmd_walk, "merge": cmd_merge, "time": cmd_time, "tasks": cmd_tasks}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
