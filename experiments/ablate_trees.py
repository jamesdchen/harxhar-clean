"""Tree walks of the ablation run (results/ablations_2014start_w3000/RULE.txt sections 3-6): LightGBM, XGBoost and the
random forest (rf_skx) on column subsets of the final 13-bar design, refitted at every one of the templates' 33,970
bars (2014-01-02 10:00 .. 2024-04-30 16:00) on the W = 39,000 rows before it.

The templates' model code is used unchanged: experiments/pooled_trees_rerun_13.py model_factory / fit_predict (seed 42,
the frozen configurations), rv_bars, cpu_model, the oneDAL dispatch check and the one-CPU-family rule (ONE_CPU,
cpu_family).  LightGBM runs with deterministic=True and force_col_wise=True in every arm (RULE.txt section 3); gate T1
checks that this reproduces the templates.  What changes is the columns: the arm's columns (experiments/ablate_arms.py)
are selected from the full cache in memory, C-ordered float64, as pooled_trees_rerun_13.Arm holds the full set.

usage: python experiments/ablate_trees.py <mode> [options]
  tasks    --out F                       the task file (lines "<est> <arm> <i>/<n> <hours at EPYC speed>"), forest
                                         first, then XGBoost, LightGBM, the most expensive arms first; pieces of about
                                         --target-hours at 8 threads from the templates' measured cost
  walk     --arm A --est E (--part i/n | --ends K | --first K) --full D --out O [--threads T] [--tag X]
           -> O/pieces/<arm>_<est>_<tag>.parquet + .json (checkpoint .ckpt.json every --ckpt-sec s; resumable)
  gate1    --out O [--est E1,E2]         T1: O/pieces/{full,har}_<est>_ends10 vs the templates -> O/gate1_<ests>.json
                                         and O/passed_<est> (the CPU model) for every class that passes; exit 3 if any
                                         fails
  gate2    --gates O1,O2,..  --out J     T2: lo_rm's first-10 pieces of each class across the gate dirs (CPU models)
  project  --gates O1,..  --tasks F --out J [--cap 15000]
                                         seconds a refit on each gated model (gate pieces) -> projected CPU-hours of
                                         the task file at the slowest gated speed; drops lo_rm's forest, then the
                                         forest's add arms (most columns first) until <= cap -> J (+ rf_skip.txt);
                                         exit 3 if still above the cap
  merge    --arm A --est E --out O [--gate2 J] -> O/forecasts/<est>_<arm>.parquet (+ .json): one complete piece set,
                                         true_adj / B equal to the templates', the CPU models per piece
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import re
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
import pooled_trees_rerun_13 as T  # noqa: E402

FIRST_DAY = "2014-01-02"
W = 39000
EST = ("lgbm", "xgb", "rf_skx")
TEMPLATE_FC = {
    "lgbm": ROOT / "results" / "pooled_trees_final_2014start_w3000" / "forecasts.parquet",
    "xgb": ROOT / "results" / "pooled_trees_final_2014start_w3000" / "forecasts.parquet",
    "rf_skx": ROOT / "results" / "pooled_rf_skx_2014start_w3000" / "forecasts.parquet",
}
SET_OF = {"full": "all_features", "har": "baseline"}
# measured CPU-hours of the templates (CARC EPYC 7513 / 7542, 8 threads, 33,970 refits; RULE.txt section 6)
COST = {"lgbm": (39.3, 249.5), "xgb": (26.5, 383.8), "rf_skx": (50.3, 2575.5)}  # (HAR p = 12, full p = 564)
PIECE_COLS = ["model", "arm", "date", "pred_adj", "true_adj", "B", "plain", "rv_actual", "fit_row"]
RF_TOL = 1e-14


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cost_cpu_h(est: str, p: int) -> float:
    """CPU-hours of a whole arm (33,970 refits, 8 threads) at EPYC speed, linear in the column count p."""
    c12, c564 = COST[est]
    b = (c564 - c12) / 552.0
    return max(c12 * 0.5, c12 + b * (p - 12))


def arms_of(est: str) -> list[str]:
    return list(AA.RF_ARMS) if est == "rf_skx" else list(AA.TREE_ARMS)


# ============================================================================ design
class Full:
    """The full cache (X F-ordered on disk, read through a memory map), its names, y, b_oos and OOS dates."""

    def __init__(self, d: Path):
        self.dir = d
        self.meta = json.loads((d / "design.json").read_text())
        self.names = list(self.meta["names"])
        AA.check_full_names(self.names)
        assert int(self.meta["config"]["window"]) == W, self.meta["config"]
        self.X = np.load(d / "X.npy", mmap_mode="r")
        self.y = np.load(d / "y.npy")
        self.b = np.load(d / "b_oos.npy")
        self.d = pd.DatetimeIndex(pd.read_csv(d / "dates_oos.csv", parse_dates=["date"])["date"])
        assert len(self.y) == W + len(self.d) == self.X.shape[0] and len(self.b) == len(self.d)

    def origins(self) -> np.ndarray:
        """OOS rows of the templates' 33,970 bars (every OOS row from 2014-01-02 on)."""
        i = np.flatnonzero(self.d.normalize() >= pd.Timestamp(FIRST_DAY))
        assert len(i) == 33970, len(i)
        return i

    def cols(self, arm: str) -> list[int]:
        names = AA.arm_names(arm, self.names)
        assert len(names) == AA.n_cols(arm), (arm, len(names))
        return [self.names.index(n) for n in names]


def model(est: str, threads: int):
    """pooled_trees_rerun_13.model_factory; LightGBM with deterministic=True and force_col_wise=True added."""
    cfg, make = T.model_factory(est, threads)
    if est != "lgbm":
        return cfg, make
    import lightgbm as lgb

    params = dict(cfg.get("params") or {})
    return cfg, lambda: lgb.LGBMRegressor(
        **params, num_threads=threads, random_state=42, verbosity=-1, deterministic=True, force_col_wise=True
    )


def select(a, F: Full) -> tuple[np.ndarray, str]:
    o = F.origins()
    if a.ends:
        return np.concatenate([o[: a.ends], o[-a.ends :]]), a.tag or f"ends{a.ends}"
    if a.first:
        return o[: a.first], a.tag or f"first{a.first}"
    i, n = (int(v) for v in a.part.split("/"))
    assert 1 <= i <= n, a.part
    day = F.d[o].normalize()
    blocks = np.array_split(day.unique(), n)
    return o[day.isin(blocks[i - 1])], f"{i}of{n}"


def piece_path(out: Path, arm: str, est: str, tag: str) -> Path:
    return out / "pieces" / f"{arm}_{est}_{tag}.parquet"


# ============================================================================ walk
def cmd_walk(a) -> int:
    F = Full(Path(a.full))
    arm, est = a.arm, a.est
    assert est in EST, est
    assert arm in ("full", "har") or arm in arms_of(est), (arm, est)
    rows, tag = select(a, F)
    out = Path(a.out)
    path = piece_path(out, arm, est, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_p, ckpt_p = path.with_suffix(".json"), path.with_suffix(".ckpt.json")
    stamps = F.d[rows]
    done = pd.read_parquet(path) if path.is_file() else pd.DataFrame(columns=PIECE_COLS)
    have = set(pd.DatetimeIndex(done["date"]).tolist())
    if meta_p.is_file() and len(done) == len(rows):
        say(f"complete: {path.name}")
        return 0
    cols = F.cols(arm)
    lo, hi = int(rows.min()), W + int(rows.max()) + 1  # design rows [lo, hi): every window [r - W, r) and row r
    X = np.ascontiguousarray(F.X[lo:hi][:, cols], dtype=np.float64)
    cfg, make = model(est, a.threads)
    rv = T.rv_bars()
    prior: dict = json.loads(ckpt_p.read_text()) if ckpt_p.is_file() else {"wall_sec": 0.0, "refits": 0}
    cpus: list[str] = [str(x) for x in prior.get("cpus", [])] + [T.cpu_model()]
    if est in T.ONE_CPU and len({T.cpu_family(x) for x in cpus}) > 1:
        raise SystemExit(f"{est}: piece {path.name} was begun on {cpus[:-1]}, this node is {cpus[-1]}")
    say(
        f"walk {arm} {est} {tag}: {len(rows)} refits {stamps[0]} .. {stamps[-1]}, {len(have)} on disk; {len(cols)} "
        f"columns, W {W}; {cfg['name']} {cfg['sha']}, threads {a.threads}"
        f"{', LightGBM deterministic + force_col_wise' if est == 'lgbm' else ''}; {cpus[-1]}"
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

    for g, t in zip(rows, stamps):
        if t in have:
            continue
        r = W + int(g)  # design row
        yh = T.fit_predict(make, est, X[r - W - lo : r - lo], F.y[r - W : r], X[r - lo : r - lo + 1])
        b = float(F.b[g])
        recs.append(
            pd.DataFrame(
                {
                    "model": est,
                    "arm": arm,
                    "date": [t],
                    "pred_adj": yh,
                    "true_adj": [float(F.y[r])],
                    "B": [b],
                    "plain": yh**2 * b,
                    "rv_actual": [float(rv.get(t, np.nan))],
                    "fit_row": [r - W],
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
        "tag": tag,
        "n_columns": len(cols),
        "columns_md5": hashlib.md5("\n".join(F.names[j] for j in cols).encode()).hexdigest(),
        "design": str(F.dir),
        "window_rows": W,
        "block": [str(stamps[0]), str(stamps[-1])],
        "rows": len(rows),
        "refits": int(st["refits"]),
        "wall_sec": round(st["wall_sec"], 1),
        "config": cfg["name"],
        "sha": cfg["sha"],
        "params": cfg.get("params"),
        "config_file_sha256": T.sha256(T.CONFIGS[est]),
        "threads": a.threads,
        "host": platform.node(),
        "cpu": T.cpu_model(),
        "cpus": sorted(set(cpus)),
        "versions": T.versions(),
        "job": os.environ.get("SLURM_JOB_ID", os.environ.get("JOB_ID", "")),
    }
    if est == "lgbm":
        meta["lgbm_deterministic"] = True
        meta["lgbm_force_col_wise"] = True
    if est == "rf_skx":
        meta["skx_dispatch"] = skx_dispatch
        meta["skx_split"] = {k_: v for k_, v in make().get_params().items() if k_ in ("max_bins", "min_bin_size")}
    meta_p.write_text(json.dumps(meta, indent=1))
    ckpt_p.unlink(missing_ok=True)
    say(f"DONE {path.name}: {k} refits in {time.time() - t0:.1f} s ({meta['wall_sec']} s in all)")
    return 0


# ============================================================================ tasks
def cmd_tasks(a) -> int:
    lines = []
    for est in ("rf_skx", "xgb", "lgbm"):
        arms = sorted(arms_of(est), key=lambda x: -cost_cpu_h(est, AA.n_cols(x)))
        for arm in arms:
            wall = cost_cpu_h(est, AA.n_cols(arm)) / 8.0
            n = max(1, math.ceil(wall / a.target_hours))
            lines += [f"{est} {arm} {i}/{n} {wall / n:.2f}" for i in range(1, n + 1)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(ln + "\n" for ln in lines))
    tot = sum(cost_cpu_h(e, AA.n_cols(x)) for e in EST for x in arms_of(e))
    say(f"{len(lines)} lines -> {a.out}; {tot:,.0f} CPU-hours at EPYC speed")
    return 0


# ============================================================================ gates
def template(est: str, gb: str = "") -> pd.DataFrame:
    """The template forecasts of est; gb (addendum D): another file for LightGBM / XGBoost (fomcfix forecasts_gb, the
    full set on the patched design; its HAR rows are the templates')."""
    f = pd.read_parquet(TEMPLATE_FC[est])
    f = f[f["model"] == est]
    if gb and est in ("lgbm", "xgb"):
        g = pd.read_parquet(gb)
        g = g[(g["model"] == est) & (g["predictor_set"] == "all_features")]
        f = pd.concat([f[f["predictor_set"] != "all_features"], g], ignore_index=True)
    return f


def cmd_gate1(a) -> int:
    out = Path(a.out)
    rec: dict = {"dir": str(out), "classes": {}, "fail": []}
    cpu = None
    for est in [e for e in a.est.split(",") if e] or list(EST):
        tf = template(est, a.template_gb)
        res: list[dict] = []
        ok = True
        for arm in ("full", "har"):
            p = piece_path(out, arm, est, "ends10")
            if not (p.is_file() and p.with_suffix(".json").is_file()):
                ok = False
                res.append(dict(arm=arm, missing=True))
                continue
            f = pd.read_parquet(p)
            m = json.loads(p.with_suffix(".json").read_text())
            cpu = cpu or m["cpus"][0]
            if len(m["cpus"]) != 1:
                ok = False
            t = tf[tf["predictor_set"] == SET_OF[arm]].set_index("date").reindex(pd.DatetimeIndex(f["date"]))
            x, y = f["pred_adj"].to_numpy(), t["pred_adj"].to_numpy()
            rel = float(np.max(np.abs(x - y) / np.abs(y)))
            text = [repr(float(u)) == repr(float(v)) for u, v in zip(x, y)]
            eq_side = bool(np.array_equal(f["true_adj"].to_numpy(), t["true_adj"].to_numpy())) and bool(
                np.array_equal(f["B"].to_numpy(), t["B"].to_numpy())
            )
            good = eq_side and (all(text) if est != "rf_skx" else rel <= RF_TOL)
            ok &= good
            res.append(
                dict(
                    arm=arm,
                    rows=len(f),
                    text_identical=int(sum(text)),
                    max_rel=rel,
                    side_equal=eq_side,
                    cpus=m["cpus"],
                    sec_refit=m["wall_sec"] / max(1, m["refits"]),
                    pass_=good,
                )
            )
        rec["classes"][est] = dict(pass_=ok, checks=res)
        if ok and cpu:
            (out / f"passed_{est}").write_text(cpu + "\n")
        else:
            rec["fail"].append(est)
    rec["cpu"] = cpu
    rec["pass"] = not rec["fail"]
    (out / f"gate1_{'_'.join(rec['classes'])}.json").write_text(json.dumps(rec, indent=1))
    say(f"gate T1 {out.name} [{cpu}]: {'PASS' if rec['pass'] else 'FAIL ' + str(rec['fail'])}")
    return 0 if rec["pass"] else 3


def cmd_gate2(a) -> int:
    dirs = [Path(x) for x in a.gates.split(",") if x]
    rec: dict = {"classes": {}}
    for est in EST:
        ps = []
        for d in dirs:
            p = piece_path(d, "lo_rm", est, "first10")
            if p.is_file() and p.with_suffix(".json").is_file():
                m = json.loads(p.with_suffix(".json").read_text())
                ps.append((d.name, m["cpus"], pd.read_parquet(p)["pred_adj"].to_numpy()))
        if len(ps) < 2:
            rec["classes"][est] = dict(models=[x[1] for x in ps], identical=None)
            continue
        ref = ps[0][2]
        ident = all(np.array_equal(ref, x[2]) for x in ps[1:])
        rel = max(float(np.max(np.abs(ref - x[2]) / np.abs(ref))) for x in ps[1:])
        rec["classes"][est] = dict(models={x[0]: x[1] for x in ps}, identical=bool(ident), max_rel=rel)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    say(f"gate T2: {json.dumps({e: v.get('identical') for e, v in rec['classes'].items()})}")
    return 0


def cmd_project(a) -> int:
    """Seconds a refit per gated model and class (gate pieces: full and har ends10, lo_rm first10) -> a + b p fit per
    model -> CPU-hours of every task line at the slowest gated model's speed (8 threads)."""
    dirs = [Path(x) for x in a.gates.split(",") if x]
    speed: dict = {}
    for d in dirs:
        for est in EST:
            if not (d / f"passed_{est}").is_file():
                continue
            pts = []
            for arm, tag in (("full", "ends10"), ("har", "ends10"), ("lo_rm", "first10")):
                p = piece_path(d, arm, est, tag).with_suffix(".json")
                if p.is_file():
                    m = json.loads(p.read_text())
                    pts.append((AA.n_cols(arm), m["wall_sec"] / max(1, m["refits"]) * m["threads"] / 3600.0))
            if len(pts) >= 2:
                P, C = np.array([x[0] for x in pts], float), np.array([x[1] for x in pts], float)
                b, c0 = np.polyfit(P, C, 1)
                speed.setdefault(est, {})[d.name] = dict(a=float(c0), b=float(b), points=pts)
    lines = [ln.split() for ln in Path(a.tasks).read_text().splitlines() if ln.strip()]
    rows = []
    for est, arm, part, _h in lines:
        n = int(part.split("/")[1])
        ncol = AA.n_cols(arm)
        per = speed.get(est, {})
        if not per:
            rows.append(dict(est=est, arm=arm, part=part, cpu_h=float("nan")))
            continue
        worst = max(per.values(), key=lambda s: s["a"] + s["b"] * ncol)  # CPU-hours a refit, slowest gated model
        rows.append(dict(est=est, arm=arm, part=part, cpu_h=(worst["a"] + worst["b"] * ncol) * 33970 / n))
    P = pd.DataFrame(rows)
    by_arm = P.groupby(["est", "arm"], sort=False)["cpu_h"].sum()
    total = float(np.nansum(P["cpu_h"]))
    drop: list[str] = []
    order = ["lo_rm"] + sorted(AA.ADD, key=lambda x: -AA.n_cols(x))
    for arm in order:
        if total <= a.cap:
            break
        if ("rf_skx", arm) in by_arm.index:
            total -= float(by_arm[("rf_skx", arm)])
            drop.append(arm)
    rec = dict(
        cap=a.cap,
        total_cpu_h=total,
        total_all=float(np.nansum(P["cpu_h"])),
        rf_dropped=drop,
        by_est={e: float(np.nansum(P[P["est"] == e]["cpu_h"])) for e in EST},
        by_arm={f"{e}/{x}": float(v) for (e, x), v in by_arm.items()},
        speed=speed,
        missing_speed=sorted({e for e in EST if e not in speed}),
    )
    Path(a.out).write_text(json.dumps(rec, indent=1))
    (Path(a.out).parent / "rf_skip.txt").write_text("".join(f"rf_skx {x}\n" for x in drop))
    say(
        f"projection {total:,.0f} CPU-hours (all {rec['total_all']:,.0f}); forest arms dropped {drop}; cap {a.cap:,.0f}"
    )
    return 0 if total <= a.cap else 3


# ============================================================================ merge
def cmd_merge(a) -> int:
    out = Path(a.out)
    arm, est = a.arm, a.est
    pat = re.compile(rf"^{re.escape(arm)}_{est}_(\d+)of(\d+)\.parquet$")
    by_n: dict[int, dict[int, Path]] = {}
    for p in sorted((out / "pieces").glob(f"{arm}_{est}_*of*.parquet")):
        m = pat.match(p.name)
        if m and p.with_suffix(".json").is_file() and not p.with_suffix(".ckpt.json").exists():
            by_n.setdefault(int(m.group(2)), {})[int(m.group(1))] = p
    full = [n for n, ps in by_n.items() if set(ps) == set(range(1, n + 1))]
    assert len(full) == 1, f"{arm}/{est}: complete piece sets {full} of {sorted(by_n)}"
    n = full[0]
    frames, pcs = [], {}
    for i in range(1, n + 1):
        p = by_n[n][i]
        frames.append(pd.read_parquet(p))
        pm = json.loads(p.with_suffix(".json").read_text())
        assert pm["arm"] == arm and pm["est"] == est and pm["window_rows"] == W
        if est == "lgbm":
            assert pm.get("lgbm_deterministic") and pm.get("lgbm_force_col_wise"), p.name
        if est in T.ONE_CPU:
            assert len({T.cpu_family(c) for c in pm["cpus"]}) == 1, (p.name, pm["cpus"])
        pcs[p.name] = pm["cpus"]
    f = pd.concat(frames, ignore_index=True).sort_values("date", ignore_index=True)
    models = sorted({c for v in pcs.values() for c in v})
    if est != "rf_skx" and len(models) > 1:  # pieces of several CPU models only with T2's bit identity
        g2 = json.loads(Path(a.gate2).read_text()) if a.gate2 else {}
        assert g2.get("classes", {}).get(est, {}).get("identical") is True, f"{arm}/{est}: {models} without T2"
    tf = template(est)
    t = tf[tf["predictor_set"] == "all_features"].set_index("date")
    assert pd.DatetimeIndex(f["date"]).equals(pd.DatetimeIndex(t.index)), f"{arm}/{est}: rows"
    assert np.array_equal(f["true_adj"].to_numpy(), t["true_adj"].to_numpy()), "true_adj"
    assert np.array_equal(f["B"].to_numpy(), t["B"].to_numpy()), "B"
    o = out / "forecasts" / f"{est}_{arm}.parquet"
    o.parent.mkdir(parents=True, exist_ok=True)
    f[["date", "pred_adj"]].to_parquet(o, index=False)
    o.with_suffix(".json").write_text(
        json.dumps(dict(arm=arm, est=est, n_pieces=n, rows=len(f), cpus=models, piece_cpus=pcs), indent=1)
    )
    say(f"merged {arm}/{est}: {n} pieces, {len(f)} rows, CPU models {models}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("tasks", "walk", "gate1", "gate2", "project", "merge"))
    ap.add_argument("--arm", default="")
    ap.add_argument("--est", default="")
    ap.add_argument("--part", default="1/1")
    ap.add_argument("--ends", type=int, default=0)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--full", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--ckpt-sec", type=float, default=300.0)
    ap.add_argument("--target-hours", type=float, default=3.0)
    ap.add_argument("--gates", default="")
    ap.add_argument("--gate2", default="")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--cap", type=float, default=15000.0)
    ap.add_argument("--template-gb", default="", help="gate1: LightGBM / XGBoost full-set templates (addendum D)")
    a = ap.parse_args()
    return {
        "tasks": cmd_tasks,
        "walk": cmd_walk,
        "gate1": cmd_gate1,
        "gate2": cmd_gate2,
        "project": cmd_project,
        "merge": cmd_merge,
    }[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
