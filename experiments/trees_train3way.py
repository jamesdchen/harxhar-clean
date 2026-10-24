"""Three-way training-sample comparison of the pooled trees, 2014 start, 3,000-day window
(results/trees_train3way_2014start_w3000/RULE.txt, written before any job).

The final tree walks are the templates: results/pooled_trees_final_2014start_w3000 (LightGBM lgbm_opt_00, XGBoost
xgb_opt_00) and results/pooled_rf_skx_2014start_w3000 (the random forest rf400_rf_opt_00 on oneDAL, rf_skx), every one
refitted at every bar on W = 39,000 rows of the final 13-bar design.  Their model code is used unchanged:
experiments/pooled_trees_rerun_13.py model_factory / fit_predict (seed 42, the frozen configurations), rv_bars, the
oneDAL dispatch check and the one-CPU-family rule.  What changes is the training sample of each refit:

  arm         design (the linear comparison's caches)          sample of the window       W (rows)   origins
  RTH13       the rebuilt 13-bar design (= w3000, md5)          every row                  39,000     the template's
  CLOSE       the same design                                   its 16:00 rows only         3,000     its 16:00 rows
  RTH13_N3k   the same design                                   every row                   3,000     the template's
  ALL48       the 48-interval design (overnight indicator)      every row                 144,000     the template's
  ALL48_N3k   the same design                                   every row                   3,000     the template's
  ALL48_N39k  the same design                                   every row                  39,000     the template's

"The template's" origins: the 33,970 bar ends the template walks forecast (every row of the 13-bar design from
2014-01-02 10:00 to 2024-04-30 16:00); in the 48-interval design the rows with those end stamps (asserted present).
The forecast at an origin (design row r, position j of r in the arm's sample S) is fitted from scratch on the rows
S[j - W], .., S[j - 1] (the W sample rows before it) with the fixed seed, and predicts row r: no state is carried
from one refit to the next, so a walk that refits only at the origins gives the forecast a walk refitting at every
sample row would give there (gate 2 checks it).  RTH13 is not rerun: its forecasts are the templates' (gate 1 checks
that this code path reproduces slices of them).

usage: python experiments/trees_train3way.py <mode> [options]
  check   the design caches against the templates' design and the linear comparison's derived caches -> --out
          design_check.json (exit 3 on a failed check)
  walk    --arm A --est E --set S [--part i/n | --ends K | --from T0 --to T1 [--every-row]] [--tag X]
          -> <out>/pieces/<arm>_<est>_<set>_<i>of<n>.parquet (+ .json meta; resumable through .ckpt.json)
  merge   --arm A [--est E ...] -> <out>/<arm>/forecasts.parquet (complete piece sets only, --require-all to insist)
  time    --arm A [--est E ...] [--set S ...] --k K --threads T -> <out>/timing/sec_refit.csv (appended)
  tasks   --arm A ... --sec-table CSV --target-hours H -> --taskfile (lines "<arm> <est> <set> <i>/<n>") and
          <taskfile>.projection.csv (CPU-hours = seconds a refit x refits x threads)
  gate1   compare the gate-1 pieces (--ends K on RTH13) with the templates' forecasts -> <out>/gate1.json, exit 3
  gate2   compare the gate-2 pieces (every row against the scored origins only) -> <out>/gate2.json, exit 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pooled_trees_rerun_13 as T  # noqa: E402

FIRST_DAY = "2014-01-02"
SETS = ("baseline", "all_features")
EST = ("lgbm", "xgb", "rf_skx")
DESIGNS = {"RTH13": 39000, "ALL48": 144000}  # the design caches' own window (scaler, late-start warm-up)
ARMS: dict[str, dict[str, Any]] = {
    "RTH13": dict(design="RTH13", sample="all", W=39000, desc="13 regular-hours bars, 3,000 days (the template)"),
    "CLOSE": dict(design="RTH13", sample="close", W=3000, desc="the closing half hour: the 16:00 rows, 3,000 sessions"),
    "RTH13_N3k": dict(design="RTH13", sample="all", W=3000, desc="13 regular-hours bars, the last 3,000 rows"),
    "ALL48": dict(design="ALL48", sample="all", W=144000, desc="all 48 half-hour intervals, 3,000 days"),
    "ALL48_N3k": dict(design="ALL48", sample="all", W=3000, desc="all 48 half-hour intervals, the last 3,000 rows"),
    "ALL48_N39k": dict(design="ALL48", sample="all", W=39000, desc="all 48 half-hour intervals, the last 39,000 rows"),
}
# the templates' design caches (results/pooled_trees_final_2014start_w3000/run.json design.md5; the rf_skx run used the
# same files, results/pooled_rf_skx_2014start_w3000/design/)
TEMPLATE_MD5 = {
    "baseline": {
        "X.npy": "05da4b5b30ec4579df77f1a3b6dd57ed",
        "y.npy": "996504f16da7982d64656d1ba0d9722a",
        "b_oos.npy": "7670f01a3055265dfac28ebac1eca864",
        "dates_oos.csv": "d8a203fe273cd3e8e34987fe5a73048e",
    },
    "all_features": {
        "X.npy": "374bea97f35e33a02404dc47c82c3138",
        "y.npy": "996504f16da7982d64656d1ba0d9722a",
        "b_oos.npy": "7670f01a3055265dfac28ebac1eca864",
        "dates_oos.csv": "d8a203fe273cd3e8e34987fe5a73048e",
    },
}
TEMPLATE_FC = {
    "lgbm": ROOT / "results" / "pooled_trees_final_2014start_w3000" / "forecasts.parquet",
    "xgb": ROOT / "results" / "pooled_trees_final_2014start_w3000" / "forecasts.parquet",
    "rf_skx": ROOT / "results" / "pooled_rf_skx_2014start_w3000" / "forecasts.parquet",
}
FLAGS = {
    "RTH13": {
        "DROP_ACTIVITY_MA": True,
        "CALENDAR_THROUGH_OVERNIGHT": True,
        "DROP_OVERNIGHT_INDICATOR": True,
        "LEAN_GLOBAL_LAGS": True,
        "LATE_START_WARMUP": True,
    },
    "ALL48": {
        "DROP_ACTIVITY_MA": True,
        "CALENDAR_THROUGH_OVERNIGHT": True,
        "DROP_OVERNIGHT_INDICATOR": False,
        "LEAN_GLOBAL_LAGS": True,
        "LATE_START_WARMUP": True,
    },
}
PIECE_COLS = T.COLS + ["fit_first", "fit_last"]


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ============================================================================ designs, samples, origins
class Design:
    """One design cache: <root>/<name>/_work/design_<set>/ (X.npy, y.npy, b_oos.npy, dates_oos.csv, dates_all.csv,
    design.json).  Design row k has date dates_all[k]; OOS row i is design row Wd + i (Wd = the cache's window)."""

    def __init__(self, root: Path, name: str, s: str):
        self.name, self.set = name, s
        self.dir = root / name / "_work" / f"design_{s}"
        self.meta = json.loads((self.dir / "design.json").read_text())
        self.Wd = int(self.meta["config"]["window"])
        assert self.Wd == DESIGNS[name], (name, self.Wd)
        self.dates = pd.DatetimeIndex(pd.read_csv(self.dir / "dates_all.csv", parse_dates=["date"])["date"])
        oos = pd.DatetimeIndex(pd.read_csv(self.dir / "dates_oos.csv", parse_dates=["date"])["date"])
        assert self.dates[self.Wd :].equals(oos), name
        assert self.dates.is_unique and self.dates.is_monotonic_increasing, name
        self.n = len(self.dates)
        assert list(self.meta["shape"]) == [self.n, len(self.meta["names"])], (name, self.meta["shape"])

    def X(self) -> np.ndarray:
        return np.load(self.dir / "X.npy", mmap_mode="r")

    def y(self) -> np.ndarray:
        return np.load(self.dir / "y.npy")

    def b(self) -> np.ndarray:
        return np.load(self.dir / "b_oos.npy")


def template_stamps(root: Path) -> pd.DatetimeIndex:
    """The templates' 33,970 forecast rows: every OOS row of the 13-bar design from 2014-01-02 (the trees' span_rows
    with --first-day 2014-01-02 and the last design row)."""
    d = Design(root, "RTH13", "baseline")
    o = d.dates[d.Wd :]
    return o[o.normalize() >= pd.Timestamp(FIRST_DAY)]


def sample_rows(des: Design, arm: str) -> np.ndarray:
    """The arm's sample: design rows its windows are drawn from, ascending."""
    if ARMS[arm]["sample"] == "all":
        return np.arange(des.n)
    return np.flatnonzero(des.dates.strftime("%H:%M") == "16:00")


def arm_origins(root: Path, des: Design, arm: str) -> np.ndarray:
    """Design rows of the arm's scored origins (see the module docstring), ascending."""
    st = template_stamps(root)
    if ARMS[arm]["sample"] == "close":
        st = st[st.strftime("%H:%M") == "16:00"]
    r = des.dates.get_indexer(st)
    assert (r >= 0).all(), f"{arm}: {int((r < 0).sum())} origin stamps missing from {des.name}"
    assert (r >= des.Wd).all(), f"{arm}: origins before the first row with a stored B"
    return r


def positions(S: np.ndarray, rows: np.ndarray, W: int) -> np.ndarray:
    """Positions of rows in the sample S (each must be a sample row with W sample rows before it)."""
    p = np.searchsorted(S, rows)
    assert (p < len(S)).all() and np.array_equal(S[p], rows), "origin not in the arm's sample"
    assert (p >= W).all(), f"an origin has fewer than W = {W} sample rows before it"
    return p


# ============================================================================ walk
def piece_path(out: Path, arm: str, est: str, s: str, tag: str) -> Path:
    return out / "pieces" / f"{arm}_{est}_{s}_{tag}.parquet"


def select(a, root: Path, des: Design, arm: str) -> tuple[np.ndarray, str]:
    """The design rows this call refits and its piece tag."""
    orig = arm_origins(root, des, arm)
    if a.ends:  # gate 1: the first and last K origins
        k = a.ends
        return np.concatenate([orig[:k], orig[-k:]]), a.tag or f"ends{k}"
    if a.from_:  # gate 2: a stretch of stamps, every sample row or the scored origins only
        t0, t1 = pd.Timestamp(a.from_), pd.Timestamp(a.to)
        if a.every_row:
            S = sample_rows(des, arm)
            r = S[(des.dates[S] >= t0) & (des.dates[S] <= t1)]
        else:
            r = orig[(des.dates[orig] >= t0) & (des.dates[orig] <= t1)]
        assert len(r), (t0, t1)
        return r, a.tag or ("every" if a.every_row else "scored")
    i, n = (int(v) for v in a.part.split("/"))
    assert 1 <= i <= n, a.part
    day = des.dates[orig].normalize()
    blocks = np.array_split(day.unique(), n)
    return orig[day.isin(blocks[i - 1])], f"{i}of{n}"


def cmd_walk(a) -> int:
    root, out = Path(a.design_root).resolve(), Path(a.out).resolve()
    arm, est, s = a.arm, a.est[0], a.set[0]
    c = ARMS[arm]
    W = c["W"]
    des = Design(root, c["design"], s)
    rows, tag = select(a, root, des, arm)
    S = sample_rows(des, arm)
    P = positions(S, rows, W)
    path = piece_path(out, arm, est, s, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_p, ckpt_p = path.with_suffix(".json"), path.with_suffix(".ckpt.json")
    stamps = des.dates[rows]
    done = pd.read_parquet(path) if path.is_file() else pd.DataFrame(columns=PIECE_COLS)
    have = set(pd.DatetimeIndex(done["date"]).tolist())
    if meta_p.is_file() and len(done) == len(rows):
        say(f"complete: {path.name}")
        return 0
    # the sample rows the piece touches, in memory C-ordered float64 (as pooled_trees_rerun_13.Arm)
    base = int(P.min()) - W
    need = S[base : int(P.max()) + 1]
    Xm = des.X()
    if c["sample"] == "all":
        assert need[-1] - need[0] + 1 == len(need)
        X = np.ascontiguousarray(Xm[int(need[0]) : int(need[-1]) + 1], dtype=np.float64)
    else:
        X = np.ascontiguousarray(Xm[need], dtype=np.float64)
    y_all = des.y()
    y = np.ascontiguousarray(y_all[need])
    b_all = des.b()
    cfg, make = T.model_factory(est, a.threads)
    rv = T.rv_bars()
    prior: dict = json.loads(ckpt_p.read_text()) if ckpt_p.is_file() else {"wall_sec": 0.0, "refits": 0}
    cpus: list[str] = [str(x) for x in prior.get("cpus", [])] + [T.cpu_model()]
    if est in T.ONE_CPU and len({T.cpu_family(x) for x in cpus}) > 1:
        raise SystemExit(f"{est}: piece {path.name} was begun on {cpus[:-1]}, this node is {cpus[-1]}")
    say(
        f"walk {arm} {est} {s} {tag}: {len(rows)} refits {stamps[0]} .. {stamps[-1]}, {len(have)} on disk; design "
        f"{des.name} ({des.n} rows, {X.shape[1]} columns), sample {c['sample']} ({len(S)} rows), W {W}; "
        f"{cfg['name']} {cfg['sha']}, threads {a.threads}"
    )
    recs = [done] if len(done) else []
    t0, t_ck, k = time.time(), time.time(), 0
    skx_dispatch: list[str] = []

    def save(final: bool) -> dict:
        f = pd.concat(recs, ignore_index=True)[PIECE_COLS].sort_values("date", ignore_index=True)
        f.to_parquet(path.with_suffix(".part"), index=False)
        os.replace(path.with_suffix(".part"), path)
        st = {"wall_sec": prior["wall_sec"] + time.time() - t0, "refits": prior["refits"] + k, "cpus": cpus}
        if not final:
            ckpt_p.write_text(json.dumps(st))
        return st

    for r, p, t in zip(rows, P, stamps):
        if t in have:
            continue
        lo, hi = int(p) - W - base, int(p) - base  # positions in X / y of the window [S[p - W], S[p - 1]]
        yh = T.fit_predict(make, est, X[lo:hi], y[lo:hi], X[hi : hi + 1])
        b = float(b_all[int(r) - des.Wd])
        recs.append(
            pd.DataFrame(
                {
                    "model": est,
                    "predictor_set": s,
                    "date": [t],
                    "pred_adj": yh,
                    "true_adj": [float(y_all[int(r)])],
                    "B": [b],
                    "plain": yh**2 * b,
                    "rv_actual": [float(rv.get(t, np.nan))],
                    "fit_first": [int(need[lo])],
                    "fit_last": [int(need[hi - 1])],
                }
            )
        )
        k += 1
        if est == "rf_skx" and k == 1:  # the first fit ran on oneDAL, not stock scikit-learn (as the template)
            assert any("fit: running accelerated version" in m for m in T._SkxLog.msgs), T._SkxLog.msgs[:5]
            assert not any("fallback" in m.lower() for m in T._SkxLog.msgs), T._SkxLog.msgs[:5]
            skx_dispatch = sorted(set(T._SkxLog.msgs))
            logging.getLogger("sklearnex").setLevel(logging.WARNING)
        if time.time() - t_ck > a.ckpt_sec:
            save(False)
            t_ck = time.time()
            say(f"  {k} refits, {(time.time() - t0) / k:.2f} s a refit, through {t}")
    st = save(True)
    f = pd.read_parquet(path)
    assert pd.DatetimeIndex(f["date"]).equals(stamps), "piece rows"
    meta = {
        "arm": arm,
        "est": est,
        "predictor_set": s,
        "tag": tag,
        "design": str(des.dir),
        "design_window": des.Wd,
        "sample": c["sample"],
        "window_rows": W,
        "block": [str(stamps[0]), str(stamps[-1])],
        "rows": len(rows),
        "refits": int(st["refits"]),
        "wall_sec": round(st["wall_sec"], 1),
        "config": cfg["name"],
        "sha": cfg["sha"],
        "params": cfg.get("params"),
        "config_file": str(T.CONFIGS[est].relative_to(ROOT)),
        "config_file_sha256": T.sha256(T.CONFIGS[est]),
        "n_columns": int(X.shape[1]),
        "threads": a.threads,
        "host": platform.node(),
        "cpu": T.cpu_model(),
        "cpus": sorted(set(cpus)),
        "versions": T.versions(),
        "slurm_job": os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "")),
        "slurm_task": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
    }
    if est == "rf_skx":
        meta["skx_dispatch"] = skx_dispatch
        meta["skx_split"] = {k_: v for k_, v in make().get_params().items() if k_ in ("max_bins", "min_bin_size")}
    meta_p.write_text(json.dumps(meta, indent=1))
    ckpt_p.unlink(missing_ok=True)
    say(f"DONE {path.name}: {k} refits in {time.time() - t0:.1f} s ({meta['wall_sec']} s in all)")
    return 0


# ============================================================================ merge
def cmd_merge(a) -> int:
    import re

    root, out = Path(a.design_root).resolve(), Path(a.out).resolve()
    arm = a.arm
    frames, used, cms = [], {}, {}
    for est in a.est or list(EST):
        for s in SETS:
            des = Design(root, ARMS[arm]["design"], s)
            want = des.dates[arm_origins(root, des, arm)]
            pat = re.compile(rf"^{arm}_{est}_{s}_(\d+)of(\d+)\.parquet$")
            by_n: dict[int, dict[int, Path]] = {}
            for p in sorted((out / "pieces").glob(f"{arm}_{est}_{s}_*of*.parquet")):
                m = pat.match(p.name)
                if m and p.with_suffix(".json").is_file():
                    by_n.setdefault(int(m.group(2)), {})[int(m.group(1))] = p
            full = {n: ps for n, ps in by_n.items() if set(ps) == set(range(1, n + 1))}
            if not full:
                say(f"{arm} {est} {s}: no complete set of pieces ({ {n: sorted(ps) for n, ps in by_n.items()} })")
                if a.require_all:
                    raise SystemExit(1)
                continue
            assert len(full) == 1, f"{arm} {est} {s}: complete splits {sorted(full)}"
            n, ps = next(iter(full.items()))
            f = pd.concat([pd.read_parquet(ps[i]) for i in range(1, n + 1)], ignore_index=True)
            f = f.sort_values("date", ignore_index=True)
            assert f["date"].is_unique and pd.DatetimeIndex(f["date"]).equals(want), f"{arm} {est} {s}: rows"
            num = f[["pred_adj", "true_adj", "B", "plain"]].to_numpy(float)
            assert np.isfinite(num).all(), f"{arm} {est} {s}: non-finite values"
            assert np.allclose(f["plain"], f["pred_adj"] ** 2 * f["B"], rtol=1e-12, atol=0)
            c: set[str] = set()
            for p in ps.values():
                pm = json.loads(p.with_suffix(".json").read_text())
                assert (pm["arm"], pm["est"], pm["predictor_set"], pm["window_rows"]) == (arm, est, s, ARMS[arm]["W"])
                c |= set(pm.get("cpus") or [pm["cpu"]])
            if est in T.ONE_CPU:
                assert len({T.cpu_family(x) for x in c}) == 1, f"{arm} {est} {s}: CPU families {sorted(c)}"
            cms[f"{est}/{s}"] = sorted(c)
            used[f"{est}/{s}"] = [p.name for p in ps.values()]
            frames.append(f[T.COLS])
            say(f"{arm} {est} {s}: {len(f)} rows from {n} pieces")
    assert frames, "nothing to merge"
    fc = pd.concat(frames, ignore_index=True)
    fc["date"] = pd.to_datetime(fc["date"]).astype("datetime64[ns]")
    for col in T.COLS[3:]:
        fc[col] = fc[col].astype(np.float64)
    dest = out / arm / "forecasts.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(dest, index=False)
    dest.with_suffix(".json").write_text(
        json.dumps(dict(arm=arm, window_rows=ARMS[arm]["W"], pieces=used, cpu_models=cms), indent=1)
    )
    say(f"merged {len(fc)} rows -> {dest}")
    return 0


# ============================================================================ time / tasks
def cmd_time(a) -> int:
    root, out = Path(a.design_root).resolve(), Path(a.out).resolve()
    arm = a.arm
    c = ARMS[arm]
    W = c["W"]
    recs = []
    for s in a.set or list(SETS):
        des = Design(root, c["design"], s)
        orig = arm_origins(root, des, arm)[: a.k]
        S = sample_rows(des, arm)
        P = positions(S, orig, W)
        base = int(P.min()) - W
        need = S[base : int(P.max()) + 1]
        Xm = des.X()
        X = np.ascontiguousarray(
            Xm[int(need[0]) : int(need[-1]) + 1] if c["sample"] == "all" else Xm[need], dtype=np.float64
        )
        y = np.ascontiguousarray(des.y()[need])
        for est in a.est or list(EST):
            _, make = T.model_factory(est, a.threads)
            t0 = time.time()
            for p in P:
                lo, hi = int(p) - W - base, int(p) - base
                T.fit_predict(make, est, X[lo:hi], y[lo:hi], X[hi : hi + 1])
            sec = (time.time() - t0) / len(P)
            n_ref = len(arm_origins(root, des, arm))
            recs.append(
                dict(
                    arm=arm,
                    est=est,
                    set=s,
                    n_columns=int(X.shape[1]),
                    window_rows=W,
                    threads=a.threads,
                    k=len(P),
                    sec_refit=sec,
                    refits=n_ref,
                    cpu_hours=sec * n_ref * a.threads / 3600,
                    host=platform.node(),
                    cpu=T.cpu_model(),
                    slurm_job=os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "")),
                )
            )
            say(f"TIME {arm} {est} {s} (W {W}, {X.shape[1]} columns, {a.threads} threads): {sec:.2f} s a refit")
    tab = pd.DataFrame(recs)
    (out / "timing").mkdir(parents=True, exist_ok=True)
    job = os.environ.get("SLURM_ARRAY_JOB_ID", os.environ.get("SLURM_JOB_ID", "local"))
    tid = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    p = out / "timing" / f"sec_{arm}_{'-'.join(a.est or EST)}_t{a.threads}_{job}_{tid}.csv"
    tab.to_csv(p, index=False)
    print(tab.drop(columns=["cpu"]).to_string(index=False))
    return 0


def cmd_tasks(a) -> int:
    root = Path(a.design_root).resolve()
    files = list(a.sec_table) + (sorted(str(p) for p in Path(a.sec_dir).glob("sec_*.csv")) if a.sec_dir else [])
    sec = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    while True:
        lines, info, proj = plan(a, root, sec)
        if not a.max_lines or len(lines) <= a.max_lines:
            break
        say(f"{len(lines)} tasks > --max-lines {a.max_lines} at {a.target_hours:.2f} h a task: target x 1.25")
        a.target_hours *= 1.25
    Path(a.taskfile).parent.mkdir(parents=True, exist_ok=True)
    Path(a.taskfile).write_text("\n".join(lines) + "\n")
    pd.DataFrame(proj).to_csv(a.taskfile + ".projection.csv", index=False)
    print("\n".join(info))
    say(f"{len(lines)} tasks -> {a.taskfile}; {sum(r['cpu_hours'] for r in proj):.0f} CPU-h projected")
    return 0


def plan(a, root: Path, sec: pd.DataFrame) -> tuple[list[str], list[str], list[dict]]:
    """Task lines of the arms (--arm, in order; within an arm LightGBM, XGBoost, then the forest, the HAR set first),
    the pooled_trees_rerun_13 rule: n = ceil(projected wall hours / target hours) contiguous trading-day blocks."""
    lines: list[str] = []
    info: list[str] = []
    proj: list[dict] = []
    for arm in a.arm_list:
        c = ARMS[arm]
        for est in a.est or list(EST):
            for s in SETS:
                t = sec[(sec["arm"] == arm) & (sec["est"] == est) & (sec["set"] == s)]
                if est in a.threads_of:
                    t = t[t["threads"] == a.threads_of[est]]
                assert len(t), f"no seconds a refit for {arm} {est} {s}"
                t = t.iloc[-1]
                des = Design(root, c["design"], s)
                orig = arm_origins(root, des, arm)
                n_days = int(des.dates[orig].normalize().nunique())
                tot = float(t["sec_refit"]) * len(orig)
                n = max(1, min(n_days, math.ceil(tot / 3600 / a.target_hours)))
                lines += [f"{arm} {est} {s} {i}/{n}" for i in range(1, n + 1)]
                cpu = tot * int(t["threads"]) / 3600
                info.append(
                    f"# {arm} {est} {s}: {len(orig)} refits x {t['sec_refit']:.2f} s ({int(t['threads'])} threads) "
                    f"-> {n} tasks of ~{tot / n / 3600:.2f} h; {cpu:.1f} CPU-h"
                )
                proj.append(
                    dict(
                        arm=arm,
                        est=est,
                        set=s,
                        refits=len(orig),
                        threads=int(t["threads"]),
                        sec_refit=float(t["sec_refit"]),
                        tasks=n,
                        wall_hours=tot / 3600,
                        cpu_hours=cpu,
                    )
                )
    return lines, info, proj


def cmd_project(a) -> int:
    """The projected CPU-hours of every task file's projection in --taskdir (the new walks; RTH13 is reused), ->
    <taskdir>/projection.json; exit 3 when the total exceeds --cap."""
    d = Path(a.taskdir).resolve()
    pr = pd.concat([pd.read_csv(p) for p in sorted(d.glob("*.projection.csv"))], ignore_index=True)
    by = pr.groupby(["arm", "est"])["cpu_hours"].sum().round(1)
    tot = float(pr["cpu_hours"].sum())
    out = dict(
        cap=a.cap,
        total_new_cpu_hours=round(tot, 1),
        by_arm=pr.groupby("arm")["cpu_hours"].sum().round(1).to_dict(),
        by_arm_est={f"{k[0]}/{k[1]}": float(v) for k, v in by.items()},
        rth13_reused_actual_cpu_hours=dict(lgbm=288.8, xgb=410.3, rf_skx=2625.8),
        within_cap=bool(tot <= a.cap),
    )
    (d / "projection.json").write_text(json.dumps(out, indent=1))
    say(f"PROJECTION {json.dumps(out)}")
    return 0 if tot <= a.cap else 3


# ============================================================================ gates
def cmd_gate1(a) -> int:
    """Gate 1: the --ends pieces of RTH13 (this code path, every tag found) against the templates' forecasts:
    LightGBM / XGBoost text-identical (repr of pred_adj), the forest within 1e-14 relative (CPU paths; expected 0)."""
    out = Path(a.out).resolve()
    g: dict = {}
    ok = True
    covered = set()
    for est in EST:
        tf = pd.read_parquet(TEMPLATE_FC[est])
        for s in SETS:
            ref = tf[(tf.model == est) & (tf.predictor_set == s)].set_index("date")
            for p in sorted((out / "pieces").glob(f"RTH13_{est}_{s}_*.parquet")):
                m = json.loads(p.with_suffix(".json").read_text())
                f = pd.read_parquet(p).set_index("date")
                r = ref.reindex(f.index)
                assert r["pred_adj"].notna().all(), p.name
                a_, b_ = f["pred_adj"].to_numpy(float), r["pred_adj"].to_numpy(float)
                text = [repr(x) for x in a_] == [repr(x) for x in b_]
                rel = float(np.max(np.abs(a_ - b_) / np.abs(b_)))
                same_y = bool(np.array_equal(f["true_adj"].to_numpy(), r["true_adj"].to_numpy()))
                same_b = bool(np.array_equal(f["B"].to_numpy(), r["B"].to_numpy()))
                good = same_y and same_b and (text if est != "rf_skx" else rel <= 1e-14)
                g[p.stem] = dict(
                    origins=len(f),
                    first=str(f.index[0]),
                    last=str(f.index[-1]),
                    threads=m["threads"],
                    cpu=m["cpu"],
                    text_identical=text,
                    max_rel_diff=rel,
                    true_adj_equal=same_y,
                    B_equal=same_b,
                    pass_=bool(good),
                )
                ok &= good
                covered.add((est, s))
    g["pass"] = bool(ok) and covered == {(e, s) for e in EST for s in SETS}
    (out / "gate1.json").write_text(json.dumps(g, indent=1))
    say(f"GATE 1 {'PASS' if g['pass'] else 'FAIL'}: {json.dumps(g)}")
    return 0 if g["pass"] else 3


def cmd_gate2(a) -> int:
    """Gate 2: for each arm / est / set with both an 'every' and a 'scored' piece, the scored origins' forecasts of the
    every-row walk equal the scored-only walk's bit for bit (and date, true_adj, B)."""
    out = Path(a.out).resolve()
    g: dict = {}
    ok = True
    n = 0
    for p in sorted((out / "pieces").glob("*_scored.parquet")):
        # 2026-10-24 (--arm, optional): the arms named only (a piece is <arm>_<est>_<set>_<tag>)
        if a.arm_list and not any(p.name.startswith(f"{x}_{e}_") for x in a.arm_list for e in EST):
            continue
        q = p.with_name(p.name.replace("_scored.parquet", "_every.parquet"))
        assert q.is_file() and q.with_suffix(".json").is_file() and p.with_suffix(".json").is_file(), p.name
        fs, fe = pd.read_parquet(p).set_index("date"), pd.read_parquet(q).set_index("date")
        e = fe.reindex(fs.index)
        same = all(np.array_equal(fs[c].to_numpy(float), e[c].to_numpy(float)) for c in ("pred_adj", "true_adj", "B"))
        ms, me = (json.loads(x.with_suffix(".json").read_text()) for x in (p, q))
        ps, pe = fs["pred_adj"].to_numpy(float), e["pred_adj"].to_numpy(float)
        g[p.stem.replace("_scored", "")] = dict(
            every_rows=len(fe),
            scored_rows=len(fs),
            stamps_every=[str(fe.index[0]), str(fe.index[-1])],
            identical=bool(same),
            differing=[str(t) for t, x, y_ in zip(fs.index, ps, pe) if x != y_],
            max_rel_diff=float(np.max(np.abs(ps - pe) / np.abs(pe))),
            sec_refit_every=me["wall_sec"] / max(1, me["refits"]),
            sec_refit_scored=ms["wall_sec"] / max(1, ms["refits"]),
            threads=me["threads"],
            cpu=[me["cpu"], ms["cpu"]],
        )
        ok &= same and len(fe) > len(fs) > 0
        n += 1
    g["pass"] = bool(ok) and n > 0
    g["arms"] = a.arm_list or "all"
    (out / ("gate2.json" if not a.arm_list else f"gate2_{'_'.join(a.arm_list)}.json")).write_text(
        json.dumps(g, indent=1)
    )
    say(f"GATE 2 {'PASS' if g['pass'] else 'FAIL'}: {json.dumps(g)}")
    return 0 if g["pass"] else 3


# ============================================================================ check (designs)
def cmd_check(a) -> int:
    """The design caches of this comparison (copies of the linear comparison's, --design-root) against the templates'
    design and the linear comparison's own derived caches (--linear, read only)."""
    root, lin = Path(a.design_root).resolve(), Path(a.linear).resolve()
    g: dict = {}
    ok = True
    for s in SETS:
        r13, a48 = Design(root, "RTH13", s), Design(root, "ALL48", s)
        # (a) RTH13 = the templates' design, byte for byte
        mm = {fn: md5(r13.dir / fn) for fn in TEMPLATE_MD5[s]}
        g[f"a_RTH13_{s}_md5"] = dict(mine=mm, equal=mm == TEMPLATE_MD5[s])
        ok &= mm == TEMPLATE_MD5[s]
        # (b) configurations
        for d in (r13, a48):
            cf = d.meta["config"]
            good = (
                cf["sample"] == {"RTH13": "rth13", "ALL48": "48"}[d.name]
                and cf["har_base"] == "5"
                and cf["ex_flags"] == FLAGS[d.name]
                and int(cf["periods_per_day"]) == {"RTH13": 13, "ALL48": 48}[d.name]
            )
            g[f"b_{d.name}_{s}_config"] = dict(config=cf, shape=d.meta["shape"], pass_=bool(good))
            ok &= good
        n13, n48 = r13.meta["names"], a48.meta["names"]
        good = set(n48) - set(n13) == {"is_overnight"} and set(n13) <= set(n48) and len(n48) == len(n13) + 1
        g[f"b_names_{s}"] = dict(
            rth13=len(n13),
            all48=len(n48),
            all48_extra=sorted(set(n48) - set(n13)),
            order_equal=[x for x in n48 if x != "is_overnight"] == n13,
            pass_=bool(good),
        )
        ok &= good
        # (c) the origins, and y / B of ALL48 = RTH13's at every origin
        o13, o48 = arm_origins(root, r13, "RTH13"), arm_origins(root, a48, "ALL48")
        good = bool(
            len(o13) == len(o48) == 33970
            and r13.dates[o13].equals(a48.dates[o48])
            and np.array_equal(r13.y()[o13], a48.y()[o48], equal_nan=True)
            and np.array_equal(r13.b()[o13 - r13.Wd], a48.b()[o48 - a48.Wd], equal_nan=True)
        )
        g[f"c_origins_{s}"] = dict(
            n=len(o13),
            first=str(r13.dates[o13[0]]),
            last=str(r13.dates[o13[-1]]),
            all48_first_window_row=str(a48.dates[o48[0] - 144000]),
            y_B_equal=good,
        )
        ok &= good
        # (d) CLOSE: the linear comparison's CLOSE cache = this design's 16:00 rows, byte for byte, and the trees'
        # CLOSE windows lie inside it
        lc = lin / "CLOSE" / "_work" / f"design_{s}"
        rr = np.load(lc / "rth13_rows.npy")
        S16 = sample_rows(r13, "CLOSE")
        Xl, X13 = np.load(lc / "X.npy", mmap_mode="r"), r13.X()
        same = bool(np.array_equal(np.asarray(Xl), np.asarray(X13[rr]), equal_nan=True)) and np.array_equal(
            np.load(lc / "y.npy"), r13.y()[rr], equal_nan=True
        )
        oc = arm_origins(root, r13, "CLOSE")
        pc = positions(S16, oc, 3000)
        inside = bool(np.isin(S16[pc.min() - 3000 :], rr).all())
        g[f"d_CLOSE_{s}"] = dict(
            linear_rows=len(rr),
            X_y_equal=same,
            rows_are_16=bool(np.isin(rr, S16).all()),
            origins=len(oc),
            first_window_row=str(r13.dates[S16[pc.min() - 3000]]),
            windows_inside_linear_cache=inside,
        )
        ok &= same and inside and bool(np.isin(rr, S16).all())
        # (e) the equal-rows arms: the linear derived caches = the sibling's rows from W_sibling - N, byte for byte
        for arm, sib, N in (("RTH13_N3k", r13, 3000), ("ALL48_N3k", a48, 3000), ("ALL48_N39k", a48, 39000)):
            ld = lin / arm / "_work" / f"design_{s}"
            drop = sib.Wd - N
            Xl, Xs = np.load(ld / "X.npy", mmap_mode="r"), sib.X()
            same = Xl.shape == (sib.n - drop, Xs.shape[1])
            for j0 in range(0, Xs.shape[1], 64) if same else ():
                same &= bool(
                    np.array_equal(np.asarray(Xl[:, j0 : j0 + 64]), np.asarray(Xs[drop:, j0 : j0 + 64]), equal_nan=True)
                )
            same &= bool(np.array_equal(np.load(ld / "y.npy"), sib.y()[drop:], equal_nan=True))
            same &= bool(np.array_equal(np.load(ld / "b_oos.npy"), sib.b(), equal_nan=True))
            g[f"e_{arm}_{s}"] = dict(rows_dropped=drop, X_y_B_equal=bool(same))
            ok &= same
    g["pass"] = bool(ok)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "design_check.json").write_text(json.dumps(g, indent=1, default=str))
    say(f"DESIGN CHECK {'PASS' if ok else 'FAIL'}: {json.dumps(g, default=str)[:4000]}")
    return 0 if ok else 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("check", "walk", "merge", "time", "tasks", "project", "gate1", "gate2"))
    ap.add_argument("--design-root", default="", help="holds RTH13/ and ALL48/ (_work/design_<set>/)")
    ap.add_argument("--out", default="")
    ap.add_argument("--linear", default="", help="check: the linear comparison's results dir (read only)")
    ap.add_argument("--arm", action="append", dest="arm_list", choices=tuple(ARMS), default=[])
    ap.add_argument("--est", action="append", choices=EST)
    ap.add_argument("--set", action="append", choices=SETS)
    ap.add_argument("--part", default="1/1")
    ap.add_argument("--ends", type=int, default=0)
    ap.add_argument("--from", dest="from_", default="")
    ap.add_argument("--to", default="")
    ap.add_argument("--every-row", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--ckpt-sec", type=float, default=300.0)
    ap.add_argument("--require-all", action="store_true")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sec-table", action="append", default=[])
    ap.add_argument("--target-hours", type=float, default=3.0)
    ap.add_argument("--threads-of", default="", help="tasks: est=threads,... (the timing rows to use)")
    ap.add_argument("--taskfile", default="")
    ap.add_argument("--sec-dir", default="", help="tasks: every sec_*.csv of this directory (time's output)")
    ap.add_argument("--max-lines", type=int, default=0, help="tasks: raise the target hours until the lines fit")
    ap.add_argument("--taskdir", default="", help="project: the task files' directory")
    ap.add_argument("--cap", type=float, default=25000.0, help="project: CPU-hours")
    a = ap.parse_args()
    a.arm = a.arm_list[0] if a.arm_list else ""
    a.threads_of = dict((k, int(v)) for k, v in (x.split("=") for x in a.threads_of.split(",") if x))
    if a.mode == "walk":
        assert len(a.arm_list) == 1 and a.est and a.set and len(a.est) == len(a.set) == 1, "walk: one arm, est, set"
    if a.mode in ("merge", "time"):
        assert len(a.arm_list) == 1, f"{a.mode}: one --arm"
    return {
        "check": cmd_check,
        "walk": cmd_walk,
        "merge": cmd_merge,
        "time": cmd_time,
        "tasks": cmd_tasks,
        "project": cmd_project,
        "gate1": cmd_gate1,
        "gate2": cmd_gate2,
    }[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
