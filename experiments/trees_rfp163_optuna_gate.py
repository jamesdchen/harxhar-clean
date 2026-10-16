"""IDENTITY gate (and smoke) of the shipped-forest Optuna rung: specs/causal_tune_trees_optuna_rfp.py vs the fixed forest.

PARAMETERIZED by the configuration file and the key token (--config, --token; defaults = the worked instance rfp163,
cluster/rfprev163_rf_config.json, the dev-span search's trial 163).  The wrapper runs specs/causal_tune_trees_optuna.py
unchanged with the configuration's forest as the shipped forest (TREE_SPEC = specs/causal_tune_trees_<token>.py,
written by experiments/trees_rfp163_make_specs.py; trial 0 = its configuration; max_depth choices re-centred on its
depth: {d/8, d/4, d/2, d, 2d} rounded half up, >= 1, de-duplicated, + None -- {2, 4, 7, 14, 28, None} for d = 14).
With IDENTITY=1 (one trial = the shipped configuration) its stage 2 must reproduce the FIXED-settings forest --
specs/causal_tune_trees_rfdev.py with HPC_KW_RF_CONFIG=<config>, REFIT_EVERY 1, IMPORTANCE_EVERY 10, WINDOW_MASK 1,
one thread -- bit for bit, as the default forest's Optuna rung reproduces the untuned spec
(experiments/gate_trees_optuna.py "identity").

For one bucket at bar1600 / TRAIN_WIN 2000 and whole-series OOS rows [ROW0, ROW0 + ROWS) (ROW0 a multiple of 250,
so every path's tuning points lie in the slice), every spec runs as the cluster runs it (a subprocess with the
pack scripts' environment; HALO = the window):

  fixed     the fixed-settings forest on the slice (one process, SLURM_CPUS_PER_TASK 1, cluster/slurm/
            trees_rfdev_pack.sbatch's environment for rung t1) -- or --fixed-npz, a stored chunk npz of that run
            covering the slice (e.g. the rfprev163 T1 fleet's for rfp163)
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
            from its log) and saw the patches: trial 0 = the configuration's (max_features, min_samples_leaf,
            max_depth) -- (0.7356..., 2, 14) for rfp163 --, rounds = its n_estimators (355), meta space max_depth =
            the re-centred ladder, meta shipped_params = rf_params(config)

--smoke-trials N > 0 adds a NON-identity smoke on rows [ROW0, ROW0 + --smoke-rows): stage 1 with N trials (TPE),
then stage 2 from those records.  Checks: trial 0 = the shipped vector, every rounds = n_estimators, every max_depth
in the re-centred ladder, max_features in [0.02, 1], min_samples_leaf an integer in [1, 256], every trial
COMPLETE, the tp1 configuration rows = the best-of-N trials; reports seconds per trial and the depths sampled.

Writes <out>/gate_<bucket>.csv (one row per check) and, only when every check passes, touches --flag.
Exit 1 on any failure.

--verify <npz> (no run): check one finished chunk npz of the wrapper (stage 1 trials or stage 2 trees) for the
configuration's meta / trial 0 / rounds -- cluster/slurm/optuna_rfp_pack.sbatch runs it before writing DONE.

Usage: python experiments/trees_rfp163_optuna_gate.py --bucket baseline --out <dir> [--row0 0 --rows 11]
           [--pool 4] [--fixed-npz <npz>] [--smoke-trials 3 --smoke-rows 2] [--flag <file>]
           [--config cluster/rfprev163_rf_config.json --token rfp163]
       python experiments/trees_rfp163_optuna_gate.py --verify <chunk npz> [--config ... --token ...]
"""

from __future__ import annotations

import argparse
import csv
import json
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

import trees_rfp163_make_specs as MK  # noqa: E402 -- the default CONFIG / TOKEN, rfdev's rf_params

W, SEG, MODEL = 2000, "bar1600", "rf"
WRAPPER = "specs/causal_tune_trees_optuna_rfp.py"
FIXED = "specs/causal_tune_trees_rfdev.py"
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


def depth_ladder(d) -> list:
    """The re-centred max_depth choices, stated here independently of the wrapper: the default forest's ladder
    {2, 4, 8, 16, 32, None} around an unbounded shipped depth, else {d/8, d/4, d/2, d, 2d} rounded half up,
    >= 1, de-duplicated, ascending, + None ({2, 4, 7, 14, 28, None} for d = 14)."""
    if d is None:
        return [2, 4, 8, 16, 32, None]
    vals = sorted(
        {int(np.floor(d * f + 0.5)) for f in (1 / 8, 1 / 4, 1 / 2, 1, 2)} - {0}
    )
    return [*vals, None]


def expected() -> dict:
    p = MK.rf_params(CFG["config"])
    return {
        "depths": depth_ladder(p.get("max_depth")),
        "params": json.loads(json.dumps(p)),
        "vec": np.array(
            [p["max_features"], p["min_samples_leaf"], p["max_depth"]], dtype=np.float64
        ),
        "shipped_config": {
            a: p[a] for a in ("max_features", "min_samples_leaf", "max_depth")
        },
        "n_estimators": int(p["n_estimators"]),
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
        ("meta model rf", meta.get("model") == MODEL, str(meta.get("model"))),
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
        (
            "meta space max_depth",
            space.get("max_depth") == exp["depths"],
            str(space.get("max_depth")),
        ),
        (
            "meta space max_features / min_samples_leaf",
            space.get("max_features")
            == {"kind": "float", "low": 0.02, "high": 1.0, "log": False}
            and space.get("min_samples_leaf")
            == {"kind": "int", "low": 1, "high": 256, "log": True},
            f"{space.get('max_features')} {space.get('min_samples_leaf')}",
        ),
        (
            "meta provenance = the configuration",
            f"dev-span configuration {CFG['config']} ("
            in str(meta.get("provenance", "")),
            "",
        ),
    ]


def stage1_checks(st: dict, exp: dict) -> list[tuple[str, bool, str]]:
    p = st["params"]
    md = p[:, :, 2]
    out = meta_checks(json.loads(str(st["meta"])), exp)
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
            "rounds = shipped n_estimators",
            bool((st["rounds"] == exp["n_estimators"]).all()),
            str(np.unique(st["rounds"])),
        ),
        (
            "max_depth in the new choices",
            bool(
                np.isin(
                    np.where(np.isnan(md), -1, md),
                    [-1 if d is None else d for d in exp["depths"]],
                ).all()
            ),
            str(
                sorted(
                    {None if np.isnan(v) else int(v) for v in md.ravel()},
                    key=lambda v: (v is None, v or 0),
                )
            ),
        ),
        (
            "max_features in [0.02, 1]",
            bool(((p[:, :, 0] >= 0.02) & (p[:, :, 0] <= 1.0)).all()),
            "",
        ),
        (
            "min_samples_leaf integer in [1, 256]",
            bool(
                (
                    (p[:, :, 1] >= 1)
                    & (p[:, :, 1] <= 256)
                    & (p[:, :, 1] == np.round(p[:, :, 1]))
                ).all()
            ),
            "",
        ),
        ("every trial COMPLETE", bool(np.asarray(st["complete"]).all()), ""),
    ]
    return out


def verify(npz: Path) -> int:
    exp = expected()
    z = load(npz)
    if z is None:
        print(f"verify: {npz} missing")
        return 1
    if "val_mse" in z:
        checks = stage1_checks(z, exp)
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
    """cluster/slurm/optuna_rfp_pack.sbatch's run_spec environment."""
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
    """cluster/slurm/trees_rfdev_pack.sbatch's run_line environment, rung t1."""
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
        HPC_KW_RF_CONFIG=CFG["config"],
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
            f"specs/causal_tune_trees_{a.token}.py missing (experiments/trees_rfp163_make_specs.py)"
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
        f"{a.token} Optuna gate ({CFG['config']}, depth choices {exp['depths']}): {b} rows {rows[0]}-{rows[-1]}, "
        f"pool {a.pool}, out {out}",
        flush=True,
    )
    t_all = time.time()
    # ---- fixed settings (concurrently with the identity stages), unless a stored chunk is given
    fixed_proc = None
    if a.fixed_npz:
        fixed_npz = Path(a.fixed_npz)
    else:
        fixed_npz = out / "fixed" / inner / f"trees_{SEG}.npz"
        fixed_proc = launch(
            FIXED, fixed_env(b, start, end, out / "fixed"), out / "fixed.log"
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
        print(f"  {FIXED} (fixed settings) -> rc {rc}", flush=True)
        rec("fixed", "run", "rc", rc == 0)
    rec("1", "identity", "run", s1_ok)
    rec("2", "identity", "run", s2_ok)
    st = load(s1_npz)
    if st is not None:
        for name, ok, note in stage1_checks(st, exp):
            rec("1", "identity", name, ok, note=note)
        rec("1", "identity", "rows", bool(np.array_equal(st["row"], rows)))
        m = re.search(
            r"optuna_rfp: pid (\d+)",
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
            "params = rf_params(config)",
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
            for name, ok, note in stage1_checks(sk, exp):
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
                    md = "None" if np.isnan(v[2]) else int(v[2])
                    print(
                        f"    row {int(sk['row'][r])} trial {t}: max_features {v[0]:.4f} min_samples_leaf {int(v[1])} "
                        f"max_depth {md}  val_mse {sk['val_mse'][r, t]:.5f}  {sk['sec'][r, t]:.2f}s  rounds {int(sk['rounds'][r, t])}"
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
