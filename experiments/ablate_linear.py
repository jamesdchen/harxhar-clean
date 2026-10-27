"""Linear walks of the ablation run (results/ablations_2014start_w3000/RULE.txt sections 3-5): ridge, lasso, elastic
net and the OLS on the arms' column-subset caches, through experiments/pool_linear_pretune.py walk unchanged, with the
final w3000 run's arguments (pooled_linear_rerun_2026_10_19_13.walk_jobs: CFG, frozen penalties, walk start, chunks).

usage: python experiments/ablate_linear.py <mode> [options]
  tasks  --out F                     the task file, one line "<arm> <model> <chunk> <n chunks>" (gate L's 14 walks
                                     first: arm full / har_sub, then the ablation walks)
  walk1  --tasks F --line N --root R --designs D [--full D]
                                     run line N: pool_linear_pretune.py walk -> R/linear/<arm>/<model>[/c<k>]/
  gate   --root R --out J            gate L: the full / har_sub walks = w3000/forecasts.parquet's pred_adj bit for bit
  merge  --root R --tasks F --designs D [--full D] --out P
                                     every finished (arm, model): the stitched walk table -> one parquet (arm, model,
                                     date, pred_adj, true_adj, B, plain)
The arm caches: D/<arm>/ (experiments/ablate_design.py subsets); arm "full" reads the full cache (--full).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ablate_arms as AA  # noqa: E402

W3000 = ROOT / "results" / "window_sweep_2014start_2026-10-22" / "w3000"
RUNNER = ROOT / "experiments" / "pool_linear_pretune.py"
PEN = ("ridge", "lasso", "enet")
# chunks of a walk (the final run: all_features ridge 2, lasso 4, enet 4); arms near the full set's size keep them
FULL_CHUNKS = {"ridge": 2, "lasso": 4, "enet": 4, "ols": 1}
_SETUP: dict = {}


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def orchestrator(full: Path, cell: str = ""):
    """pooled_linear_rerun_2026_10_19_13 set up as its main() does for the final run (--window-days 3000
    --first-day 2014-01-02 --first-fold 3 --ex-flag LATE_START_WARMUP=1), reading w3000's frozen_penalties.json and
    the full cache's dates."""
    import pooled_linear_rerun_2026_10_19_13 as orch

    if _SETUP:
        return orch
    orch.OUT = W3000
    orch.WORK = W3000 / "_work"
    orch.TW_DAYS = 3000
    orch.W = orch.TW_DAYS * orch.PPD
    orch.PRE_HALO = orch.PRE_FOLDS * orch.PRE_FOLD_SESS * orch.PPD + orch.PRE_EMBARGO_SESS * orch.PPD + orch.W
    orch.CFG.extend(["--window-days", "3000"])
    orch.FIRST_FOLD = 3
    orch.PRE_HALO = (orch.PRE_FOLDS - 3) * orch.PRE_FOLD_SESS * orch.PPD + orch.PRE_EMBARGO_SESS * orch.PPD + orch.W
    orch.CFG.extend(["--ex-flag", "LATE_START_WARMUP=1"])
    # 2026-10-26 addendum D: the cell's design flags after LATE_START_WARMUP (as fomcfix's walk_task.py), e.g.
    # FOMC_SCHEDULE_2020_FIX=1,FOMC_PROXIMITY_NO_WINSOR=1,DROP_HOLIDAY_SESSIONS=0; empty = the final design
    for f in [x for x in cell.replace("+", ",").split(",") if x]:  # "+" also separates (sbatch --export splits ",")
        orch.CFG.extend(["--ex-flag", f])
    orch.FIRST_DAY = "2014-01-02"
    orch.design_dir = lambda s: full  # dates_oos.csv only (both sets share the rows)
    _SETUP["done"] = True
    return orch


def n_chunks(arm: str, model: str) -> int:
    if model == "ols":
        return 1
    return FULL_CHUNKS[model] if (arm == "full" or arm.startswith("lo_")) else 1


def task_lines() -> list[tuple[str, str, int, int]]:
    out = []
    for m in PEN:  # gate L: the final run's 14 walks
        k = FULL_CHUNKS[m]
        out += [("full", m, c, k) for c in range(k)]
    out += [("har_sub", m, 0, 1) for m in (*PEN, "ols")]
    for arm in AA.LINEAR_ARMS:
        for m in PEN:
            k = n_chunks(arm, m)
            out += [(arm, m, c, k) for c in range(k)]
    out += [(arm, "ols", 0, 1) for arm in AA.OLS_ARMS]
    return out


def cmd_tasks(a) -> int:
    lines = task_lines()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(f"{arm} {m} {c} {k}\n" for arm, m, c, k in lines))
    say(f"{len(lines)} linear walks -> {a.out}")
    return 0


def walk_args(orch, arm: str, model: str, c: int, k: int, design: Path, out: Path) -> list[str]:
    """The final run's walk arguments for (set, model) with k chunks, chunk c, with --out / --design replaced.
    HAR benchmark (har_sub): the baseline set's job; every other arm: the all_features job (its penalty and r); the
    OLS on an arm with exogenous columns: the baseline OLS job with --bucket all_features (the arm cache's label)."""
    s = "baseline" if (arm == "har_sub" or model == "ols") else "all_features"
    jobs = orch.walk_jobs({(s, model): k}, [f"{s}/{model}"])
    assert len(jobs) == k, (arm, model, len(jobs), k)
    args = list(jobs[c][0])
    args[args.index("--out") + 1] = str(out)
    args[args.index("--design") + 1] = str(design)
    if model == "ols" and arm != "har_sub":
        args[args.index("--bucket") + 1] = "all_features"
    return args


def out_dir(root: Path, arm: str, model: str, c: int, k: int) -> Path:
    base = root / "linear" / arm / model
    return base if k == 1 else base / f"c{c}"


def design_of(arm: str, designs: Path, full: Path) -> Path:
    return full if arm == "full" else designs / arm


def cmd_walk1(a) -> int:
    lines = [ln.split() for ln in Path(a.tasks).read_text().splitlines() if ln.strip()]
    arm, model, c, k = lines[a.line - 1]
    c, k = int(c), int(k)
    orch = orchestrator(Path(a.full), a.cell)
    root = Path(a.root)
    o = out_dir(root, arm, model, c, k)
    args = walk_args(orch, arm, model, c, k, design_of(arm, Path(a.designs), Path(a.full)), o)
    log = o / "run.log"
    o.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    cmd = [sys.executable, "-u", str(RUNNER), *args, *orch.CFG]
    say(f"line {a.line}: {arm} {model} chunk {c + 1}/{k}: {' '.join(cmd[2:])}")
    t0 = time.time()
    with open(log, "w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    say(f"line {a.line}: exit {r.returncode} in {time.time() - t0:.0f} s ({log})")
    if r.returncode == 0:
        (o / "DONE").write_text(
            json.dumps(
                dict(line=a.line, arm=arm, model=model, chunk=c, chunks=k, sec=round(time.time() - t0, 1), cmd=cmd[2:])
            )
        )
    return r.returncode


def walk_table(root: Path, arm: str, model: str, k: int) -> pd.DataFrame:
    """The stitched walk table (orch.walk_table's rule: chunks in order, each chunk's emit range)."""
    parts = []
    for c in range(k):
        o = out_dir(root, arm, model, c, k)
        assert (o / "DONE").is_file(), f"{arm}/{model} chunk {c}: not done"
        ps = list(o.glob("causal_tune_linear/*/*/results.csv"))
        assert len(ps) == 1, (o, ps)
        parts.append(pd.read_csv(ps[0], parse_dates=["date"]))
    r = pd.concat(parts, ignore_index=True)
    assert r["date"].is_monotonic_increasing and r["date"].is_unique, (arm, model)
    return r


def cmd_gate(a) -> int:
    root = Path(a.root)
    ref = pd.read_parquet(
        W3000 / "forecasts.parquet", columns=["model", "predictor_set", "date", "pred_adj", "true_adj"]
    )
    if a.ref_full:  # 2026-10-26 addendum D: the full-set walks against another run's (fomcfix lin_B.parquet)
        rf = pd.read_parquet(a.ref_full, columns=["model", "predictor_set", "date", "pred_adj", "true_adj"])
        ref = pd.concat([ref[ref["predictor_set"] != "all_features"], rf], ignore_index=True)
    rec: dict = {"checks": [], "fail": []}
    for arm, s, ms in (("full", "all_features", PEN), ("har_sub", "baseline", (*PEN, "ols"))):
        for m in ms:
            k = FULL_CHUNKS[m] if arm == "full" else 1
            try:
                r = walk_table(root, arm, m, k)
            except AssertionError as e:
                rec["fail"].append(f"{arm}/{m}: {e}")
                continue
            t = ref[(ref["model"] == m) & (ref["predictor_set"] == s)].sort_values("date")
            same_dates = len(r) == len(t) and np.array_equal(r["date"].to_numpy(), t["date"].to_numpy())
            eq = same_dates and np.array_equal(r["pred_adj"].to_numpy(), t["pred_adj"].to_numpy())
            eq_y = same_dates and np.array_equal(r["true_adj"].to_numpy(), t["true_adj"].to_numpy())
            d = (
                float(np.max(np.abs(r["pred_adj"].to_numpy() - t["pred_adj"].to_numpy())))
                if same_dates
                else float("nan")
            )
            rec["checks"].append(
                dict(
                    arm=arm,
                    model=m,
                    set=s,
                    rows=len(r),
                    dates_equal=bool(same_dates),
                    pred_adj_bit_identical=bool(eq),
                    true_adj_equal=bool(eq_y),
                    max_abs_diff=d,
                )
            )
            if not (eq and eq_y):
                rec["fail"].append(f"{arm}/{m}: rows {len(r)} vs {len(t)}, max |diff| {d}")
    import platform

    from pooled_trees_rerun_13 import cpu_model

    rec.update(host=platform.node(), cpu=cpu_model(), numpy=np.__version__)
    rec["pass"] = not rec["fail"] and len(rec["checks"]) == 7
    Path(a.out).write_text(json.dumps(rec, indent=1))
    say(f"gate L {'PASS' if rec['pass'] else 'FAIL'} on {rec['cpu']}: {rec['fail']}")
    return 0 if rec["pass"] else 3


def cmd_merge(a) -> int:
    root = Path(a.root)
    full = Path(a.full)
    d = pd.DatetimeIndex(pd.read_csv(full / "dates_oos.csv", parse_dates=["date"])["date"])
    B = pd.Series(np.load(full / "b_oos.npy"), index=d)
    y = np.load(full / "y.npy")
    t = pd.read_parquet(W3000 / "forecasts.parquet", columns=["model", "predictor_set", "date", "true_adj"])
    t = t[(t["model"] == "ols") & (t["predictor_set"] == "baseline")]
    ty = pd.Series(t["true_adj"].to_numpy(), index=pd.DatetimeIndex(t["date"]))
    lines = [ln.split() for ln in Path(a.tasks).read_text().splitlines() if ln.strip()]
    pairs = sorted({(arm, m, int(k)) for arm, m, _, k in lines}, key=lambda x: (x[0], x[1]))
    frames, missing = [], []
    for arm, m, k in pairs:
        try:
            r = walk_table(root, arm, m, k)
        except AssertionError as e:
            missing.append(f"{arm}/{m}: {e}")
            continue
        dt = pd.DatetimeIndex(r["date"])
        b = B.reindex(dt).to_numpy()
        assert np.isfinite(b).all(), (arm, m)
        # 2026-10-25: the walk tables' true_adj went through to_csv / read_csv (pandas' default float parser, the
        # templates' path), so it equals the template's true_adj exactly and y.npy to the last bits.  Was:
        # assert np.array_equal(r["true_adj"].to_numpy(), y[39000 + d.get_indexer(dt)])
        assert np.allclose(r["true_adj"].to_numpy(), y[39000 + d.get_indexer(dt)], rtol=1e-14, atol=0), (arm, m)
        assert np.array_equal(r["true_adj"].to_numpy(), ty.reindex(dt).to_numpy()), (arm, m, "true_adj")
        frames.append(
            pd.DataFrame(
                dict(
                    arm=arm,
                    model=m,
                    date=dt,
                    pred_adj=r["pred_adj"].to_numpy(),
                    true_adj=r["true_adj"].to_numpy(),
                    B=b,
                    plain=r["pred_adj"].to_numpy() ** 2 * b,
                )
            )
        )
    f = pd.concat(frames, ignore_index=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    f.to_parquet(a.out, index=False)
    if a.wide:  # 2026-10-26 addendum F: the ablation walks (not gate L's full / har_sub), one pred_adj per model|arm
        g = f[~f["arm"].isin(["full", "har_sub"]) & (f["date"] >= pd.Timestamp("2014-01-02"))]
        w = g.pivot(index="date", columns=["model", "arm"], values="pred_adj").sort_index()
        w.columns = [f"{m}|{x}" for m, x in w.columns]
        w = w[sorted(w.columns)].reset_index()
        assert len(w) == 33970 and w.notna().all().all(), (len(w), int(w.isna().sum().sum()))
        w.to_parquet(a.wide, index=False)
        say(f"compact: {w.shape[1] - 1} walks x {len(w)} bars -> {a.wide}")
    say(f"merged {len(frames)} (arm, model) walks, {len(f)} rows -> {a.out}; missing {len(missing)}: {missing[:5]}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("tasks", "walk1", "gate", "merge"))
    ap.add_argument("--tasks", default="")
    ap.add_argument("--line", type=int, default=0)
    ap.add_argument("--root", default="")
    ap.add_argument("--designs", default="")
    ap.add_argument("--full", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--cell", default="", help="extra design flags NAME=v,... (addendum D)")
    ap.add_argument("--ref-full", default="", help="gate: the full-set walks' reference parquet (addendum D)")
    ap.add_argument("--wide", default="", help="merge: also the compact file of the ablation walks (addendum F)")
    a = ap.parse_args()
    return {"tasks": cmd_tasks, "walk1": cmd_walk1, "gate": cmd_gate, "merge": cmd_merge}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
