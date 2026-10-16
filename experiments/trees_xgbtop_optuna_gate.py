"""IDENTITY gate (and smoke) of the re-centred XGBoost Optuna rung: specs/causal_tune_trees_optuna_xgbtop.py vs the fixed
XGBoost of the same configuration.  experiments/trees_rfp163_optuna_gate.py (the random forest's twin) with the
XGBoost facts in place of the forest's; the structure and the comparisons are the same.

PARAMETERIZED by the configuration file and the key token (--config, --token; defaults = cluster/xgbtop_xgb_config.json,
xgbtop = xgb_opt_00).  The wrapper runs specs/causal_tune_trees_optuna.py unchanged with the configuration's XGBoost as
the shipped XGBoost (TREE_SPEC = specs/causal_tune_trees_<token>.py, written by experiments/trees_xgbtop_make_specs.py;
trial 0 = its configuration on the six searched axes; reg_lambda axis re-centred to [1e-5, 1e4]).  With IDENTITY=1 (one
trial = the shipped configuration, early stopping off) its stage 2 must reproduce the FIXED-settings XGBoost --
specs/causal_tune_trees_<token>.py itself, REFIT_EVERY 1, IMPORTANCE_EVERY 10, WINDOW_MASK 1, one thread -- bit for bit,
as the default Optuna rung reproduces the untuned spec (experiments/gate_trees_optuna.py "identity").

For one bucket at bar1600 / TRAIN_WIN 2000 and whole-series OOS rows [ROW0, ROW0 + ROWS) (ROW0 a multiple of 250,
so every path's tuning points lie in the slice), every spec runs as the cluster runs it (a subprocess with the
pack scripts' environment; HALO = the window):

  fixed     the fixed-settings XGBoost on the slice (one process, SLURM_CPUS_PER_TASK 1, cluster/slurm/
            trees_xgbtop_pack.sbatch's environment for rung t1) -- or --fixed-npz, a stored chunk npz of that run
            covering the slice (the phase-1 T1 chunk)
  stage 1   the wrapper, STAGE tune, IDENTITY 1, a spawned pool of --pool processes
  stage 2   the wrapper, STAGE refit, IDENTITY 1, from stage 1's records, the same pool
  compare   every path's trees_bar1600.npz vs the fixed run on the slice: date, pred_adj, true_adj, true_raw,
            baseline, e2_adj, native importance (NaN rows between the recorded rows), kept columns (n_kept vs
            kept_n), TreeSHAP (tp1), bit for bit (np.array_equal, NaN == NaN); every path's results CSV pred_adj;
            tp1's results CSV (the executor's own table on both sides) byte for byte with the fixed run's; every
            configuration row = trial 0 = the shipped vector.  The fixed run's forecasts / targets are taken EXACT
            from its results CSV (round-trip parse; its npz holds the default parser's values, an ULP off at
            times -- reported, informational)
  probe     the stage-1 records were computed in spawned workers (every point's pid != the wrapper's pid, read
            from its log) and saw the patches: trial 0 = the configuration's (max_depth, min_child_weight, subsample,
            colsample_bytree, reg_lambda, learning_rate) -- (5, 4.1834, 0.6551, 0.6538, 1.159e-5, 0.012337) for
            xgbtop --, rounds = its n_estimators (367; identity: early stopping off), meta space = the re-centred
            space, meta shipped_params = expected_params(config)

--smoke-trials N > 0 adds a NON-identity smoke on rows [ROW0, ROW0 + --smoke-rows): stage 1 with N trials (TPE,
early stopping), then stage 2 from those records.  Checks: trial 0 = the shipped vector, every parameter inside the
re-centred space (max_depth an integer), every trial's patience = ceil(1 / lr) and round cap = ceil(4 x 367 x
0.012337 / lr) with 1 <= rounds <= cap, every trial COMPLETE, the tp1 configuration rows = the best-of-N trials;
reports seconds per trial.

Writes <out>/gate_<bucket>.csv (one row per check) and, only when every check passes, touches --flag.
Exit 1 on any failure.

--verify <npz> (no run): check one finished chunk npz of the wrapper (stage 1 trials or stage 2 trees) for the
configuration's meta / trial 0 / space / rounds -- cluster/slurm/optuna_xgbtop_pack.sbatch runs it before DONE.

Usage: python experiments/trees_xgbtop_optuna_gate.py --bucket baseline --out <dir> [--row0 0 --rows 11]
           [--pool 4] [--fixed-npz <npz>] [--smoke-trials 3 --smoke-rows 2] [--flag <file>]
           [--config cluster/xgbtop_xgb_config.json --token xgbtop]
       python experiments/trees_xgbtop_optuna_gate.py --verify <chunk npz> [--config ... --token ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import trees_xgbtop_make_specs as MK  # noqa: E402 -- the default CONFIG / TOKEN, expected_params

W, SEG, MODEL = 2000, "bar1600", "xgb"
WRAPPER = "specs/causal_tune_trees_optuna_xgbtop.py"
CFG = {"config": MK.CONFIG, "token": MK.TOKEN}  # set from --config / --token in main()
PATHS = (
    "tp1",
    "tp5",
    "tp25",
    "tp250",
    "tp1_k10",
    "tp1_k25",
    "tp25_k10",
    "tp25_k25",
    "tp25_q",
)
S2_KEYS = (
    "date",
    "pred_adj",
    "true_adj",
    "true_raw",
    "baseline",
    "e2_adj",
    "importance",
)
AXES = (
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
    "learning_rate",
)
CAP_MULT = (
    4  # the job module's ROUNDS_CAP_MULT (the tuned spec's 4 x the shipped rounds)
)


def fixed_spec() -> str:
    """The fixed-settings XGBoost of the configuration: the generated spec itself."""
    return f"specs/causal_tune_trees_{CFG['token']}.py"


def expected() -> dict:
    """The configuration's facts, stated here independently of the wrapper: the per-bar parameters
    (expected_params), trial 0 on the six searched axes, and the re-centred space -- the job module's XGBoost space
    with reg_lambda's lower end at the power of ten at or below the shipped value (never above 1e-4)."""
    p = MK.expected_params(CFG["config"])
    lo = min(1e-4, 10.0 ** math.floor(math.log10(p["reg_lambda"])))
    space = {
        "max_depth": {"kind": "int", "low": 1, "high": 8, "log": False},
        "min_child_weight": {"kind": "float", "low": 0.1, "high": 256.0, "log": True},
        "subsample": {"kind": "float", "low": 0.15, "high": 1.0, "log": False},
        "colsample_bytree": {"kind": "float", "low": 0.05, "high": 1.0, "log": False},
        "reg_lambda": {"kind": "float", "low": lo, "high": 1e4, "log": True},
        "learning_rate": {"kind": "float", "low": 0.005, "high": 0.1, "log": True},
    }
    with open(ROOT / CFG["config"], encoding="utf-8") as f:
        cfg = json.load(f)
    return {
        "params": json.loads(json.dumps(p)),
        "vec": np.array([p[a] for a in AXES], dtype=np.float64),
        "shipped_config": {a: p[a] for a in AXES},
        "n_estimators": int(p["n_estimators"]),
        "lr": float(p["learning_rate"]),
        "space": space,
        "name": cfg["name"],
        "sha": cfg["sha"],
    }


def load(p: Path) -> dict | None:
    if not p.is_file():
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def max_rel(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype.kind not in "fiu":
        return float("nan")
    a, b = a.astype(float), b.astype(float)
    m = np.isfinite(a) & np.isfinite(b) & (b != 0)
    return float(np.max(np.abs(a[m] / b[m] - 1.0))) if m.any() else 0.0


def meta_checks(meta: dict, exp: dict) -> list[tuple[str, bool, str]]:
    """The configuration's facts every chunk of the wrapper must record."""
    space = meta.get("space", {})
    return [
        ("meta model xgb", meta.get("model") == MODEL, str(meta.get("model"))),
        (
            "meta window_mask",
            meta.get("window_mask") is True,
            str(meta.get("window_mask")),
        ),
        (
            "meta shipped_config",
            meta.get("shipped_config") == exp["shipped_config"],
            str(meta.get("shipped_config")),
        ),
        (
            "meta shipped_params",
            meta.get("shipped_params") == exp["params"],
            str(meta.get("shipped_params")),
        ),
        ("meta space = the re-centred space", space == exp["space"], str(space)),
        ("meta axes order", meta.get("axes") == list(AXES), str(meta.get("axes"))),
        (
            "meta provenance = the configuration",
            f"{exp['name']} sha {exp['sha']} ({CFG['config']};"
            in str(meta.get("provenance", "")),
            str(meta.get("provenance", ""))[:120],
        ),
    ]


def stage1_checks(st: dict, exp: dict, identity: bool) -> list[tuple[str, bool, str]]:
    p = st["params"]
    out = meta_checks(json.loads(str(st["meta"])), exp)
    inside = np.ones(p.shape[:2], dtype=bool)
    for j, a in enumerate(AXES):
        s = exp["space"][a]
        inside &= (p[:, :, j] >= s["low"]) & (p[:, :, j] <= s["high"])
    lr = p[:, :, AXES.index("learning_rate")]
    cap = np.ceil(CAP_MULT * exp["n_estimators"] * (exp["lr"] / lr))
    out += [
        (
            "trial 0 = shipped vector",
            bool(
                np.array_equal(
                    p[:, 0, :], np.broadcast_to(exp["vec"], p[:, 0, :].shape)
                )
            ),
            str(p[0, 0]),
        ),
        (
            "every parameter inside the re-centred space",
            bool(inside.all()),
            f"{int((~inside).sum())} outside",
        ),
        ("max_depth integer", bool((p[:, :, 0] == np.round(p[:, :, 0])).all()), ""),
        ("every trial COMPLETE", bool(np.asarray(st["complete"]).all()), ""),
    ]
    if identity:
        out += [
            (
                "rounds = shipped n_estimators (early stopping off)",
                bool((st["rounds"] == exp["n_estimators"]).all()),
                str(np.unique(st["rounds"])),
            )
        ]
    else:
        out += [
            (
                "patience = ceil(1 / lr)",
                bool((st["patience"] == np.ceil(1.0 / lr)).all()),
                "",
            ),
            (
                "round cap = ceil(4 x shipped rounds x shipped lr / lr)",
                bool((st["rounds_max"] == cap).all()),
                f"trial-0 cap {np.unique(st['rounds_max'][:, 0])}",
            ),
            (
                "1 <= rounds <= cap",
                bool(((st["rounds"] >= 1) & (st["rounds"] <= st["rounds_max"])).all()),
                "",
            ),
        ]
    return out


def verify(npz: Path) -> int:
    exp = expected()
    z = load(npz)
    if z is None:
        print(f"verify: {npz} missing")
        return 1
    if "val_mse" in z:
        checks = stage1_checks(z, exp, bool(json.loads(str(z["meta"])).get("identity")))
    else:
        checks = meta_checks(json.loads(str(z["meta"])), exp)
    bad = [c for c in checks if not c[1]]
    for name, ok, note in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {name} {note}")
    print(f"verify {npz}: {'OK' if not bad else 'FAILED'}")
    return 1 if bad else 0


def exact_reference(fx: dict, csv_path: Path, sl: slice) -> dict:
    """The fixed run's arrays on the slice with its forecasts and targets EXACT: pred_adj / true_adj / true_raw
    re-read from its results CSV with the round-trip parser (the executor writes repr floats; the untuned spec
    parses them with the default parser, off by an ULP at times, while the Optuna spec uses round_trip), baseline
    and e2_adj recomputed from them with both specs' formula; date, importance and TreeSHAP from the npz."""
    r = pd.read_csv(csv_path, parse_dates=["date"], float_precision="round_trip").iloc[
        sl
    ]
    ok = (r["true_adj"] > 0) & (r["true_raw"] > 0)
    return {
        "date": fx["date"][sl],
        "pred_adj": r["pred_adj"].to_numpy(float),
        "true_adj": r["true_adj"].to_numpy(float),
        "true_raw": r["true_raw"].to_numpy(float),
        "baseline": np.where(ok, r["true_raw"] / r["true_adj"].where(ok) ** 2, np.nan),
        "e2_adj": ((r["true_adj"] - r["pred_adj"]) ** 2).to_numpy(float),
        "importance": fx["importance"][sl],
        "shap": fx["shap"][sl],
        "csv_date": r["date"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy().astype("U19"),
    }


def clean_env() -> dict:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("HPC_KW_") and k != "HPC_RESULT_DIR"
    }
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
    )
    env.update(PYTHONUNBUFFERED="1", TQDM_DISABLE="1")
    env["HPC_KW_TREE_SPEC"] = f"specs/causal_tune_trees_{CFG['token']}.py"
    return env


def optuna_env(
    bucket, start, end, stage, out, pool, trials, identity, stage1=""
) -> dict:
    """cluster/slurm/optuna_xgbtop_pack.sbatch's run_spec environment."""
    env = clean_env()
    env.update(
        SLURM_CPUS_PER_TASK=str(pool),
        HPC_KW_MODEL=MODEL,
        HPC_KW_EXOG_BUCKET=bucket,
        HPC_KW_SEGMENT=SEG,
        HPC_KW_TRAIN_WIN=str(W),
        HPC_KW_LAG_SCOPE="global",
        HPC_KW_START=str(start),
        HPC_KW_END=str(end),
        HPC_KW_HALO=str(W),
        HPC_KW_STAGE=stage,
        HPC_KW_N_TRIALS=str(trials),
        HPC_KW_IDENTITY="1" if identity else "0",
        HPC_KW_RESUME="1",
        HPC_KW_STAGE1_TRIALS=stage1,
        HPC_RESULT_DIR=str(out),
        HPC_KW_WINDOW_MASK="1",
    )
    return env


def fixed_env(bucket, start, end, out) -> dict:
    """cluster/slurm/trees_xgbtop_pack.sbatch's run_line environment, rung t1."""
    env = clean_env()
    env.update(
        SLURM_CPUS_PER_TASK="1",
        HPC_KW_MODEL=MODEL,
        HPC_KW_EXOG_BUCKET=bucket,
        HPC_KW_SEGMENT=SEG,
        HPC_KW_TRAIN_WIN=str(W),
        HPC_KW_LAG_SCOPE="global",
        HPC_KW_HAR_BASE="",
        HPC_KW_START=str(start),
        HPC_KW_END=str(end),
        HPC_KW_HALO=str(W),
        HPC_KW_REFIT_EVERY="1",
        HPC_KW_IMPORTANCE_EVERY="10",
        HPC_KW_SHAP_SEGMENTS=SEG,
        HPC_KW_WINDOW_MASK="1",
        HPC_RESULT_DIR=str(out),
    )
    return env


def launch(script: str, env: dict, log: Path) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log, "w", encoding="utf-8")  # noqa: SIM115 -- closed by the caller's wait()
    return subprocess.Popen(
        [sys.executable, "-u", script],
        cwd=ROOT,
        env=env,
        stdout=fh,
        stderr=subprocess.STDOUT,
    )


def run(script: str, env: dict, log: Path) -> bool:
    t0 = time.time()
    rc = launch(script, env, log).wait()
    tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[
        -1:
    ] or [""]
    print(
        f"  {script} STAGE={env.get('HPC_KW_STAGE', '-')} -> rc {rc} in {time.time() - t0:.0f}s: {tail[0][:220]}",
        flush=True,
    )
    return rc == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", default="")
    ap.add_argument("--bucket", default="baseline")
    ap.add_argument("--row0", type=int, default=0)
    ap.add_argument("--rows", type=int, default=11)
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--pool", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))
    )
    ap.add_argument("--fixed-npz", default="")
    ap.add_argument("--smoke-trials", type=int, default=0)
    ap.add_argument("--smoke-rows", type=int, default=2)
    ap.add_argument("--flag", default="")
    ap.add_argument(
        "--config", default=MK.CONFIG, help="the configuration JSON (repo-relative)"
    )
    ap.add_argument(
        "--token",
        default=MK.TOKEN,
        help="the key token: TREE_SPEC = specs/causal_tune_trees_<token>.py",
    )
    a = ap.parse_args()
    CFG.update(config=MK.repo_relative(a.config), token=a.token)
    if not (ROOT / "specs" / f"causal_tune_trees_{a.token}.py").is_file():
        raise SystemExit(
            f"specs/causal_tune_trees_{a.token}.py missing (experiments/trees_xgbtop_make_specs.py)"
        )
    if a.verify:
        return verify(Path(a.verify))
    if not a.out:
        raise SystemExit("--out is required")
    if a.row0 % 250:
        raise SystemExit(
            f"--row0 {a.row0} must be a multiple of 250 (every path's tuning points in the slice)"
        )
    exp = expected()
    b, out = a.bucket, Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    inner = Path("causal_tune_trees") / MODEL / b
    start, end = W + a.row0, W + a.row0 + a.rows
    rows = np.arange(a.row0, a.row0 + a.rows)
    res: list[dict] = []

    def rec(
        stage: str,
        what: str,
        key: str,
        good: bool,
        rel: float = 0.0,
        note: str = "",
        gating: bool = True,
    ) -> None:
        res.append(
            dict(
                bucket=b,
                rows=f"{rows[0]}-{rows[-1]}",
                stage=stage,
                what=what,
                key=key,
                ok=bool(good),
                gating=gating,
                max_rel_diff=rel,
                note=note,
                host=socket.gethostname(),
            )
        )

    print(
        f"{a.token} Optuna gate ({CFG['config']}, trial 0 {exp['shipped_config']}, reg_lambda low {exp['space']['reg_lambda']['low']}): {b} rows {rows[0]}-{rows[-1]}, "
        f"pool {a.pool}, out {out}",
        flush=True,
    )
    t_all = time.time()
    # ---- fixed settings (concurrently with the identity stages), unless a stored chunk is given
    fixed_proc = None
    if a.fixed_npz:
        fixed_npz = Path(
            a.fixed_npz.replace("%b", b)
        )  # %b = the bucket (one fixed chunk per bucket)
    else:
        fixed_npz = out / "fixed" / inner / f"trees_{SEG}.npz"
        fixed_proc = launch(
            fixed_spec(), fixed_env(b, start, end, out / "fixed"), out / "fixed.log"
        )
    # ---- identity: stage 1 -> stage 2
    s1_ok = run(
        WRAPPER,
        optuna_env(b, start, end, "tune", out / "s1", a.pool, 1, True),
        out / "s1.log",
    )
    s1_npz = out / "s1" / inner / f"trials_{SEG}.npz"
    s2_ok = s1_ok and run(
        WRAPPER,
        optuna_env(b, start, end, "refit", out / "s2", a.pool, 1, True, str(s1_npz)),
        out / "s2.log",
    )
    if fixed_proc is not None:
        rc = fixed_proc.wait()
        print(f"  {fixed_spec()} (fixed settings) -> rc {rc}", flush=True)
        rec("fixed", "run", "rc", rc == 0)
    rec("1", "identity", "run", s1_ok)
    rec("2", "identity", "run", s2_ok)
    st = load(s1_npz)
    if st is not None:
        for name, ok, note in stage1_checks(st, exp, True):
            rec("1", "identity", name, ok, note=note)
        rec("1", "identity", "rows", bool(np.array_equal(st["row"], rows)))
        m = re.search(
            r"optuna_xgbtop: pid (\d+)",
            (out / "s1.log").read_text(encoding="utf-8", errors="replace"),
        )
        pids = []
        for pt in sorted((out / "s1" / inner / "points").glob("pt_*.npz")):
            with np.load(pt) as z:
                pids.append(int(z["pid"]))
        parent = int(m.group(1)) if m else -1
        spawned = (
            a.pool > 1 and len(pids) == len(rows) and parent > 0 and parent not in pids
        )
        rec(
            "1",
            "probe",
            "points computed in spawned workers",
            spawned or a.pool == 1,
            note=f"wrapper pid {parent}, worker pids {sorted(set(pids))}",
        )
    fx = load(fixed_npz)
    if fx is None:
        rec("fixed", "npz", "-", False, float("nan"), f"missing {fixed_npz}")
    else:
        fm = json.loads(str(fx["meta"]))
        rec(
            "fixed",
            "meta",
            "params = expected_params(config)",
            fm.get("params") == exp["params"],
            note=str(fm.get("params")),
        )
        rec(
            "fixed",
            "meta",
            "refit_every 1 / importance_every 10 / window_mask 1",
            (fm.get("refit_every"), fm.get("importance_every"), fm.get("window_mask"))
            == (1, 10, 1),
        )
        off = max(0, fm["slice"][0] - fm["slice"][2])
        sl = slice(a.row0 - off, a.row0 - off + a.rows)
        if a.row0 - off < 0 or a.row0 - off + a.rows > len(fx["pred_adj"]):
            rec(
                "fixed",
                "npz",
                "covers the slice",
                False,
                note=f"stored rows {off}..{off + len(fx['pred_adj']) - 1}",
            )
        ref = exact_reference(fx, fixed_npz.with_name(f"results_{SEG}.csv"), sl)
        rec(
            "fixed",
            "npz",
            "pred_adj = its CSV (round-trip parse)",
            same(fx["pred_adj"][sl], ref["pred_adj"]),
            max_rel(fx["pred_adj"][sl], ref["pred_adj"]),
            "the untuned spec parses its CSV without round_trip",
            gating=False,
        )
        rec(
            "fixed",
            "csv",
            "CSV stamps = npz stamps",
            same(ref["csv_date"], ref["date"]),
        )
        fixed_csv = fixed_npz.with_name(f"results_{SEG}.csv").read_bytes()
        for path in PATHS:
            g2 = load(out / "s2" / path / inner / f"trees_{SEG}.npz")
            if g2 is None:
                rec("2", path, "-", False, float("nan"), "missing")
                continue
            for k in S2_KEYS:
                rec("2", path, k, same(g2[k], ref[k]), max_rel(g2[k], ref[k]))
            rec("2", path, "n_kept = kept_n", same(g2["n_kept"], fx["kept_n"][sl]))
            csv_p = out / "s2" / path / inner / f"results_{SEG}.csv"
            got = pd.read_csv(csv_p, float_precision="round_trip")["pred_adj"].to_numpy(
                float
            )
            rec(
                "2",
                path,
                "results CSV pred_adj",
                same(got, ref["pred_adj"]),
                max_rel(got, ref["pred_adj"]),
            )
            if (
                path == "tp1" and not a.fixed_npz
            ):  # the executor's own table on both sides
                rec(
                    "2",
                    path,
                    "results CSV bytes = fixed CSV bytes",
                    csv_p.read_bytes() == fixed_csv,
                )
            rec(
                "2",
                path,
                "cfg = trial 0 = shipped",
                bool(
                    (g2["cfg_trial"] == 0).all()
                    and np.array_equal(
                        g2["cfg_params"],
                        np.broadcast_to(exp["vec"], g2["cfg_params"].shape),
                    )
                ),
            )
            if path == "tp1":
                rec(
                    "2",
                    path,
                    "shap",
                    same(g2["shap"], ref["shap"]),
                    max_rel(g2["shap"], ref["shap"]),
                )
                for name, ok, note in meta_checks(json.loads(str(g2["meta"])), exp):
                    rec("2", path, name, ok, note=note)
        z1 = load(out / "s2" / "tp1" / inner / f"trees_{SEG}.npz")
        if z1 is not None:
            print(
                f"  identity: fixed fit {np.mean(fx['fit_sec'][sl]):.2f}s/refit; Optuna stage-2 fit "
                f"{np.mean(z1['fit_sec']):.2f}s/refit; stage-1 trial {np.mean(st['sec']) if st is not None else float('nan'):.2f}s"
            )
    # ---- smoke: TPE trials past trial 0, then stage 2 from them
    if a.smoke_trials > 0:
        s_end = W + a.row0 + a.smoke_rows
        ok1 = run(
            WRAPPER,
            optuna_env(
                b, start, s_end, "tune", out / "k1", a.pool, a.smoke_trials, False
            ),
            out / "k1.log",
        )
        k1 = out / "k1" / inner / f"trials_{SEG}.npz"
        ok2 = ok1 and run(
            WRAPPER,
            optuna_env(
                b,
                start,
                s_end,
                "refit",
                out / "k2",
                a.pool,
                a.smoke_trials,
                False,
                str(k1),
            ),
            out / "k2.log",
        )
        rec("smoke", "stage 1", "run", ok1)
        rec("smoke", "stage 2", "run", ok2)
        sk = load(k1)
        if sk is not None:
            for name, ok, note in stage1_checks(sk, exp, False):
                rec("smoke", "stage 1", name, ok, note=note)
            rec(
                "smoke",
                "stage 1",
                "n_trials",
                sk["params"].shape[1] == a.smoke_trials,
                note=str(sk["params"].shape),
            )
            g2 = load(out / "k2" / "tp1" / inner / f"trees_{SEG}.npz")
            if g2 is not None:
                best = sk["params"][np.arange(len(sk["row"])), sk["best_mse_k50"]]
                rec(
                    "smoke",
                    "stage 2",
                    "tp1 cfg = best-of-N",
                    same(g2["cfg_params"], best),
                )
            print(
                f"  smoke: {sk['params'].shape[1]} trials x {len(sk['row'])} points, {np.mean(sk['sec']):.2f}s/trial "
                f"(trial 0 {np.mean(sk['sec'][:, 0]):.2f}s), best trials {sk['best_mse_k50'].tolist()}"
            )
            for r in range(len(sk["row"])):
                for t in range(sk["params"].shape[1]):
                    v = sk["params"][r, t]
                    cfg = ", ".join(f"{ax} {x:.4g}" for ax, x in zip(AXES, v))
                    print(
                        f"    row {int(sk['row'][r])} trial {t}: {cfg}  val_mse {sk['val_mse'][r, t]:.5f}  "
                        f"{sk['sec'][r, t]:.2f}s  rounds {int(sk['rounds'][r, t])} / cap {int(sk['rounds_max'][r, t])}"
                    )
    p = out / f"gate_{b}.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    bad = [
        f"{r['stage']}/{r['what']}/{r['key']}"
        for r in res
        if r["gating"] and not r["ok"]
    ]
    info = [
        f"{r['stage']}/{r['what']}/{r['key']} (max rel {r['max_rel_diff']:.1e})"
        for r in res
        if not r["gating"] and not r["ok"]
    ]
    if info:
        print("informational (not gating): " + ", ".join(info))
    n_gating = sum(r["gating"] for r in res)
    print(
        f"gate {b} rows {rows[0]}-{rows[-1]} on {socket.gethostname()}: {n_gating - len(bad)} of {n_gating} gating checks "
        f"passed ({time.time() - t_all:.0f}s); wrote {p}"
    )
    if bad:
        print("GATE FAIL: " + ", ".join(bad[:40]))
        return 1
    if a.flag:
        Path(a.flag).parent.mkdir(parents=True, exist_ok=True)
        Path(a.flag).touch()
        print(f"GATE OK: wrote {a.flag}")
    else:
        print("GATE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
