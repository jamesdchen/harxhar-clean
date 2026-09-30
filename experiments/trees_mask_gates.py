"""Gates of the masked 16:00 tree and LSTM re-run (checklist I9: WINDOW_MASK=1) before any fleet.

The campaign (writeup/CAMPAIGN_16H_2026-09-29.md, "Restart with a per-window mask") re-runs
T10 / T1 / RS10 / RS1 (specs/causal_tune_trees.py, specs/causal_tune_trees_tuned.py) and the
LSTM (specs/causal_tune_lstm.py) with the env axis WINDOW_MASK=1: every fit sees only the
columns src.models.window_mask.window_keep keeps on the window it is trained on.  The
generic gates run UNEDITED with the mask on (WINDOW_MASK=1 in the environment, which every
spec reads; cluster/slurm/trees_mask_gates.sbatch):

  experiments/trees_cadence_gates.py   g0 design, g1 importance cadence, g2 / g2b chunks
                                       (t1, t10), g3 tuned TUNE_IDENTITY=1 = untuned, g4a
                                       process count, g4b pool size
  experiments/gate_lstm.py             determinism, pool, chunk

This script adds the gates of the mask itself.  Every run is the SHIPPED spec run end to
end (a subprocess, as the fleet runs it) on the live_feasible 16:00 slice of whole-series
OOS rows [OOS0, OOS0 + rows) (START = W + OOS0, HALO = W = 2000; OOS0 = 1000, a tuning row),
LightGBM / XGBoost / RF (+ the LSTM at the reduced FAST budget of gate_lstm.py):

  h1 pre-edit   WINDOW_MASK=0 = the spec BEFORE the edit (git f9a19b6, shipped to
                --ref-root, run from there) bit for bit: untuned at REFIT 1 / IMPORTANCE 10
                and REFIT 10 / IMPORTANCE 1, tuned (full grid, REFIT 1, TreeSHAP, pool),
                LSTM (REFIT_EVERY unset, and set to its default 10) -- every npz array
                but the timings, the npz key set, the meta but timings, the results CSVs
                byte for byte, the tuning tables but timing columns, the grid json
  h2 full rank  on the slice's design restricted to K = the columns kept on EVERY window
                the runs use (so no column of the restricted design is constant or a copy
                on any of them; a run_executor wrapper hands the spec X[:, K]):
                WINDOW_MASK=1 = WINDOW_MASK=0 bit for bit, and kept_n = |K| at every
                refit and tuning point (the LSTM at REFIT_EVERY 1 and 10)
  h3 the mask   WINDOW_MASK=1 on the slice: the kept set of every refit and tuning point =
                window_keep(X[t - W : t]) recomputed here from the captured design (the
                mask is its window's, nothing later); native importance and TreeSHAP are
                exactly 0 on dropped columns at recorded refits; TreeSHAP adds up to the
                forecast; kept_n < p; how far the masked forecasts move (recorded); the
                LSTM at REFIT_EVERY 1 and 10
  h4 chunks     WINDOW_MASK=1: untuned REFIT 1 in two chunks (seam at 7 rows) = the
                unchunked run (forecasts, kept, kept_n); tuned REFIT 10 over two tuning
                periods [OOS0, OOS0 + 250) + [OOS0 + 250, OOS0 + 260) merged by
                experiments/reduce_trees_tuned_chunks.py = unchunked (forecasts of both
                rules, refit rows, importance, candidate losses, picks), kept / tune_kept
                concatenated = unchunked; the merged QLIKE-rule CSV (the reducer re-parses
                the chunk CSVs) within MERGED_CSV_TOL of the unchunked one; the LSTM at
                REFIT_EVERY 1 over the same two tuning periods: the chunks' forecasts (both
                rules, every seed), tuning records and kept / tune_kept concatenated =
                unchunked

--parts trees / lstm runs one half (the cluster runs them as two jobs, each with its own
--out).  Outputs (--out): gate_rows.csv (one row per gate x model), gate_runs.csv (one row per run),
design_census.csv (kept count of every window of the slice), GATES_OK when every gate
passes.  Exit status 1 otherwise.

Usage (the cluster: cluster/slurm/trees_mask_gates.sbatch):
  python experiments/trees_mask_gates.py [--procs 16] [--pool 4] [--models lgbm,xgb,rf]
      [--ref-root gates_ref] [--out results/linear_subsection_trees_mask/gates/h]
Worker mode (internal): python experiments/trees_mask_gates.py --worker SPEC
      [--keep-cols K.npy] [--capture OUT.npz]  (the spec's run cell under a run_executor
      wrapper: the design restricted to K, or captured and not fitted)
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.models.window_mask import window_keep  # noqa: E402

SPECS = {
    "untuned": "specs/causal_tune_trees.py",
    "tuned": "specs/causal_tune_trees_tuned.py",
    "lstm": "specs/causal_tune_lstm.py",
}
BUCKET, SEG, W = "live_feasible", "bar1600", 2000
OOS0 = 1000  # whole-series OOS row of the slice's first forecast: a tuning row (x 250)
TUNE_PER = 250
U_ROWS = 24  # untuned slice (trees_cadence_gates' U)
C1_ROWS = 7  # untuned chunk seam (not a multiple of 10)
T_ROWS = 8  # tuned slice at REFIT 1 (one tuning point, 8 refits)
TC_ROWS = TUNE_PER + 10  # tuned chunk slice: two tuning points
L_ROWS = 30  # LSTM slice: one tuning point, three refit blocks
CAP_ROWS = TC_ROWS  # the captured design covers every slice
FAST = {
    "N_SEEDS": "2",
    "MAX_EPOCHS": "12",
    "PATIENCE": "3",
}  # gate_lstm.py's code-path budget
TIMING_KEYS = {"fit_sec", "shap_sec", "qsel_sec", "tune_sec", "cand_fit_sec", "meta"}
TIMING_META = {"walk_sec", "run_sec", "window_mask", "versions"}
TIMING_COLS = {"fit_sec", "tune_sec"}
MASK_KEYS = {"kept_n", "kept", "tune_kept_n", "tune_kept"}
SHAP_ADD_TOL = 1e-4  # float32 TreeSHAP rows summed in float64 vs the float64 forecast
# the tuned reducer re-reads the chunk CSVs with pandas' default float parser before writing the
# merged CSV, so a merged value may sit one ulp off the unchunked run's text (the npz arrays and
# the chunk CSVs themselves are compared bit for bit)
MERGED_CSV_TOL = 1e-12
LSTM_RE = (
    1,
    10,
)  # the LSTM's REFIT_EVERY values gated (1 = the campaign's, 10 = the default)


class _Captured(Exception):
    pass


# ----------------------------------------------------------------------------- worker mode
def worker(spec: str, keep_cols: str | None, capture: str | None) -> None:
    """Run the spec's run cell with run_executor wrapped (the design restricted to K, or
    captured and not fitted)."""
    import src.backtest.executor as ex

    real = ex.run_executor
    K = np.load(keep_cols) if keep_cols else None

    def wrapped(**kw):
        fp = kw["fit_predict"]

        def fp2(X, y, train_win_periods, hyperparams):
            names = list(hyperparams.get("_feature_names", []))
            if capture:
                np.savez(
                    capture,
                    X=np.asarray(X, dtype=np.float64),
                    y=np.asarray(y, dtype=np.float64),
                    W=int(train_win_periods),
                    names=np.array([str(c) for c in names]),
                )
                raise _Captured
            Xk = np.ascontiguousarray(np.asarray(X, dtype=np.float64)[:, K])
            hp = (
                dict(hyperparams, _feature_names=[names[j] for j in K])
                if names
                else hyperparams
            )
            return fp(Xk, y, train_win_periods, hp)

        kw["fit_predict"] = fp2
        return real(**kw)

    ex.run_executor = wrapped  # type: ignore[assignment]  # the specs import it from the module when they run
    try:
        runpy.run_path(spec, run_name="__main__")
    except _Captured:
        print(f"captured the design into {capture}")


# ----------------------------------------------------------------------------- runs
@dataclass
class Run:
    name: str
    kind: str  # untuned | tuned | lstm
    model: str
    start: int
    end: int
    out: Path
    env: dict = field(default_factory=dict)
    cpus: int = 1
    root: Path = ROOT  # the ref root for the pre-edit runs
    keep_cols: Path | None = None
    capture: Path | None = None
    res: dict = field(default_factory=dict)


_LOCK = threading.Lock()


def inner(r: Run) -> Path:
    if r.kind == "lstm":
        return r.out / "causal_tune_lstm" / "lstm" / BUCKET
    return r.out / "causal_tune_trees" / r.model / BUCKET


def launch(r: Run) -> Run:
    if r.out.exists():
        shutil.rmtree(r.out)
    r.out.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("HPC_KW_")}
    env.pop("WINDOW_MASK", None)
    env.update(
        HPC_KW_EXOG_BUCKET=BUCKET,
        HPC_KW_SEGMENT=SEG,
        HPC_KW_TRAIN_WIN=str(W),
        HPC_KW_LAG_SCOPE="global",
        HPC_KW_HAR_BASE="",
        HPC_KW_START=str(r.start),
        HPC_KW_END=str(r.end),
        HPC_KW_HALO=str(W),
        HPC_RESULT_DIR=str(r.out),
        SLURM_CPUS_PER_TASK=str(r.cpus),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        TQDM_DISABLE="1",
    )
    if r.kind != "lstm":
        env["HPC_KW_MODEL"] = r.model
    else:
        env.update({f"HPC_KW_{k}": v for k, v in FAST.items()})
    env.update(r.env)
    spec = str(r.root / SPECS[r.kind])
    if r.keep_cols is not None or r.capture is not None:
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", spec]
        if r.keep_cols is not None:
            cmd += ["--keep-cols", str(r.keep_cols)]
        if r.capture is not None:
            cmd += ["--capture", str(r.capture)]
    else:
        cmd = [sys.executable, "-u", spec]
    a = time.time()
    rss = np.nan
    with open(r.out / "run.log", "w", encoding="utf-8") as log:
        p = subprocess.Popen(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(r.root)
        )
        if hasattr(os, "wait4"):
            _pid, status, ru = os.wait4(p.pid, 0)
            p.returncode = os.waitstatus_to_exitcode(status)
            rss = ru.ru_maxrss / 2**20
        else:
            p.wait()
    r.res |= {"rc": p.returncode, "sec": time.time() - a, "max_rss_gib": rss}
    if p.returncode:
        text = (r.out / "run.log").read_text(encoding="utf-8", errors="replace").strip()
        print(
            f"RUN FAILED {r.name}: {text.splitlines()[-1][:300] if text else ''}",
            flush=True,
        )
    else:
        (r.out / "DONE").touch()
    with _LOCK:
        print(f"  {r.name}: rc {p.returncode}, {r.res['sec']:.0f}s", flush=True)
    return r


def load(r: Run) -> dict:
    d = inner(r)
    stem = "lstm" if r.kind == "lstm" else "trees"
    with np.load(d / f"{stem}_{SEG}.npz", allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out["_meta"] = json.loads(str(out["meta"]))
    out["_dir"] = d
    return out


def eq(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def compare_npz(a: dict, b: dict, skip: set[str]) -> list[str]:
    """Keys of a that differ in b (timings and the mask keys skipped)."""
    bad = []
    for k in a:
        if k.startswith("_") or k in skip:
            continue
        if k not in b:
            bad.append(f"{k} missing")
        elif not eq(a[k], b[k]):
            bad.append(k)
    return bad


def compare_tables(da: Path, db: Path, kind: str) -> list[str]:
    bad = []
    for f in (f"results_{SEG}.csv", f"results_qsel_{SEG}.csv", f"grid_{SEG}.json"):
        fa, fb = da / f, db / f
        if fa.is_file() != fb.is_file():
            bad.append(f"{f} on one side only")
        elif fa.is_file() and fa.read_bytes() != fb.read_bytes():
            bad.append(f"{f} bytes")
    for f in (f"tune_trace_{SEG}.csv", f"tune_candidates_{SEG}.csv"):
        fa, fb = da / f, db / f
        if fa.is_file() != fb.is_file():
            bad.append(f"{f} on one side only")
        elif fa.is_file():
            ta = pd.read_csv(fa, float_precision="round_trip")
            tb = pd.read_csv(fb, float_precision="round_trip")
            cols = [c for c in ta.columns if c not in TIMING_COLS]
            if list(ta.columns) != list(tb.columns) or not ta[cols].equals(tb[cols]):
                bad.append(f"{f} (but timings)")
    return bad


def meta_diff(a: dict, b: dict) -> list[str]:
    ma = {k: v for k, v in a["_meta"].items() if k not in TIMING_META}
    mb = {k: v for k, v in b["_meta"].items() if k not in TIMING_META}
    return sorted(k for k in set(ma) | set(mb) if ma.get(k) != mb.get(k))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default=None)
    ap.add_argument("--keep-cols", default=None)
    ap.add_argument("--capture", default=None)
    ap.add_argument("--models", default="lgbm,xgb,rf")
    ap.add_argument(
        "--parts",
        default="trees,lstm",
        help="trees and / or lstm (two jobs can split them)",
    )
    ap.add_argument(
        "--procs", type=int, default=16, help="concurrent CPUs for the spec runs"
    )
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument(
        "--ref-root", default="gates_ref", help="the pre-edit specs + src (git f9a19b6)"
    )
    ap.add_argument("--out", default="results/linear_subsection_trees_mask/gates/h")
    a = ap.parse_args()
    if a.worker:
        worker(a.worker, a.keep_cols, a.capture)
        return 0

    os.chdir(ROOT)
    out = (ROOT / a.out).resolve()
    scratch = out / "scratch"
    out.mkdir(parents=True, exist_ok=True)
    ref = (ROOT / a.ref_root).resolve()
    for f in SPECS.values():
        assert (ref / f).is_file(), f"ref root {ref} lacks {f}"
    if not (
        ref / "data"
    ).exists():  # the pre-edit specs read DATA_PATH "data" under their root
        os.symlink(ROOT / "data", ref / "data")
    parts = {x.strip() for x in a.parts.split(",") if x.strip()}
    models = (
        [m.strip() for m in a.models.split(",") if m.strip()]
        if "trees" in parts
        else []
    )
    do_lstm = "lstm" in parts
    s0 = W + OOS0

    # ---- phase 1: capture the slice's design (the untuned spec's own run_executor call)
    cap = scratch / "capture.npz"
    t0 = time.time()
    rc = launch(
        Run(
            "capture",
            "untuned",
            "lgbm",
            s0,
            s0 + CAP_ROWS,
            scratch / "capture_run",
            capture=cap,
        )
    )
    if rc.res["rc"] != 0 or not cap.is_file():
        print("capture failed")
        return 1
    with np.load(cap) as z:
        Xc, names = z["X"], z["names"]
    p = Xc.shape[1]
    keeps = [
        window_keep(Xc[i : W + i]) for i in range(CAP_ROWS)
    ]  # window of OOS row OOS0 + i
    census = pd.DataFrame(
        {
            "oos_row": OOS0 + np.arange(CAP_ROWS),
            "kept": [len(k) for k in keeps],
            "p": p,
        }
    )
    census.to_csv(out / "design_census.csv", index=False)
    span = max(U_ROWS, T_ROWS, L_ROWS)
    K = keeps[0]
    for k in keeps[1:span]:
        K = np.intersect1d(K, k)
    kfile = scratch / "K.npy"
    np.save(kfile, K)
    pd.DataFrame({"col": K, "name": names[K]}).to_csv(
        out / "full_rank_columns.csv", index=False
    )
    print(
        f"design {Xc.shape}, p {p}; kept per window {census['kept'].min()}..{census['kept'].max()}; "
        f"K = {len(K)} columns kept on all of the first {span} windows",
        flush=True,
    )

    # ---- phase 2: every run
    runs: dict[tuple[str, str], Run] = {}

    def add(r: Run) -> None:
        runs[(r.model, r.name)] = r

    for m in models:
        d = scratch / m

        def cad(re_: int, ie: int) -> dict:
            return {
                "HPC_KW_REFIT_EVERY": str(re_),
                "HPC_KW_IMPORTANCE_EVERY": str(ie),
                "HPC_KW_SHAP_SEGMENTS": SEG,
            }

        for re_, ie in ((1, 10), (10, 1)):
            tag = f"re{re_}"
            add(
                Run(
                    f"u_pre_{tag}",
                    "untuned",
                    m,
                    s0,
                    s0 + U_ROWS,
                    d / f"u_pre_{tag}",
                    cad(re_, ie),
                    root=ref,
                )
            )
            add(
                Run(
                    f"u_m0_{tag}",
                    "untuned",
                    m,
                    s0,
                    s0 + U_ROWS,
                    d / f"u_m0_{tag}",
                    cad(re_, ie) | {"HPC_KW_WINDOW_MASK": "0"},
                )
            )
            for mk in ("0", "1"):
                add(
                    Run(
                        f"u_k{mk}_{tag}",
                        "untuned",
                        m,
                        s0,
                        s0 + U_ROWS,
                        d / f"u_k{mk}_{tag}",
                        cad(re_, ie) | {"HPC_KW_WINDOW_MASK": mk},
                        keep_cols=kfile,
                    )
                )
        m1 = cad(1, 1) | {"HPC_KW_WINDOW_MASK": "1"}
        add(Run("u_m1_re1", "untuned", m, s0, s0 + U_ROWS, d / "u_m1_re1", m1))
        c10 = cad(1, 10) | {"HPC_KW_WINDOW_MASK": "1"}
        add(Run("u_c1_re1", "untuned", m, s0, s0 + C1_ROWS, d / "u_c1_re1", c10))
        add(
            Run(
                "u_c2_re1", "untuned", m, s0 + C1_ROWS, s0 + U_ROWS, d / "u_c2_re1", c10
            )
        )

        te = {
            "HPC_KW_REFIT_EVERY": "1",
            "HPC_KW_QSEL": "1",
            "HPC_KW_TUNE_IDENTITY": "0",
            "HPC_KW_SHAP_SEGMENTS": SEG,
        }
        add(
            Run(
                "t_pre",
                "tuned",
                m,
                s0,
                s0 + T_ROWS,
                d / "t_pre",
                te,
                cpus=a.pool,
                root=ref,
            )
        )
        add(
            Run(
                "t_m0",
                "tuned",
                m,
                s0,
                s0 + T_ROWS,
                d / "t_m0",
                te | {"HPC_KW_WINDOW_MASK": "0"},
                cpus=a.pool,
            )
        )
        add(
            Run(
                "t_m1",
                "tuned",
                m,
                s0,
                s0 + T_ROWS,
                d / "t_m1",
                te | {"HPC_KW_WINDOW_MASK": "1"},
                cpus=a.pool,
            )
        )
        for mk in ("0", "1"):
            add(
                Run(
                    f"t_k{mk}",
                    "tuned",
                    m,
                    s0,
                    s0 + T_ROWS,
                    d / f"t_k{mk}",
                    te | {"HPC_KW_WINDOW_MASK": mk},
                    cpus=a.pool,
                    keep_cols=kfile,
                )
            )
        tc = te | {
            "HPC_KW_REFIT_EVERY": "10",
            "HPC_KW_SHAP_SEGMENTS": "none",
            "HPC_KW_WINDOW_MASK": "1",
        }
        tcr = scratch / "tuned_chunks"
        arm = tcr / BUCKET / SEG / m / f"tw{W}" / "chunks"
        add(Run("tc_U", "tuned", m, s0, s0 + TC_ROWS, d / "tc_U", tc, cpus=a.pool))
        add(Run("tc_C1", "tuned", m, s0, s0 + TUNE_PER, arm / "c0", tc, cpus=a.pool))
        add(
            Run(
                "tc_C2",
                "tuned",
                m,
                s0 + TUNE_PER,
                s0 + TC_ROWS,
                arm / "c1",
                tc,
                cpus=a.pool,
            )
        )

    if do_lstm:
        dl = scratch / "lstm"
        lm = "lstm"
        add(
            Run(
                "l_pre",
                "lstm",
                lm,
                s0,
                s0 + L_ROWS,
                dl / "l_pre",
                cpus=a.pool,
                root=ref,
            )
        )
        add(
            Run(
                "l_m0",
                "lstm",
                lm,
                s0,
                s0 + L_ROWS,
                dl / "l_m0",
                {"HPC_KW_WINDOW_MASK": "0"},
                cpus=a.pool,
            )
        )
        add(
            Run(
                "l_m0_re10x",
                "lstm",
                lm,
                s0,
                s0 + L_ROWS,
                dl / "l_m0_re10x",
                {"HPC_KW_WINDOW_MASK": "0", "HPC_KW_REFIT_EVERY": "10"},
                cpus=a.pool,
            )
        )
        for re_ in LSTM_RE:
            r_env = {"HPC_KW_REFIT_EVERY": str(re_)}
            add(
                Run(
                    f"l_m1_re{re_}",
                    "lstm",
                    lm,
                    s0,
                    s0 + L_ROWS,
                    dl / f"l_m1_re{re_}",
                    r_env | {"HPC_KW_WINDOW_MASK": "1"},
                    cpus=a.pool,
                )
            )
            add(
                Run(
                    f"l_m0_re{re_}",
                    "lstm",
                    lm,
                    s0,
                    s0 + L_ROWS,
                    dl / f"l_m0_re{re_}",
                    r_env | {"HPC_KW_WINDOW_MASK": "0"},
                    cpus=a.pool,
                )
            )
            for mk in ("0", "1"):
                add(
                    Run(
                        f"l_k{mk}_re{re_}",
                        "lstm",
                        lm,
                        s0,
                        s0 + L_ROWS,
                        dl / f"l_k{mk}_re{re_}",
                        r_env | {"HPC_KW_WINDOW_MASK": mk},
                        cpus=a.pool,
                        keep_cols=kfile,
                    )
                )
        lc = {"HPC_KW_REFIT_EVERY": "1", "HPC_KW_WINDOW_MASK": "1"}
        add(Run("lc_U", "lstm", lm, s0, s0 + TC_ROWS, dl / "lc_U", lc, cpus=a.pool))
        add(Run("lc_C1", "lstm", lm, s0, s0 + TUNE_PER, dl / "lc_C1", lc, cpus=a.pool))
        add(
            Run(
                "lc_C2",
                "lstm",
                lm,
                s0 + TUNE_PER,
                s0 + TC_ROWS,
                dl / "lc_C2",
                lc,
                cpus=a.pool,
            )
        )

    # CPU-budgeted concurrency: a pooled run takes its a.pool CPUs at once
    cond = threading.Condition()
    free = [a.procs]

    def go(r: Run) -> Run:
        need = min(r.cpus, a.procs)
        with cond:
            cond.wait_for(lambda: free[0] >= need)
            free[0] -= need
        try:
            return launch(r)
        finally:
            with cond:
                free[0] += need
                cond.notify_all()

    order = sorted(
        runs.values(), key=lambda r: -r.cpus
    )  # pooled first, singles fill in
    print(f"phase 2: {len(order)} runs, {a.procs} CPUs", flush=True)
    with ThreadPoolExecutor(len(order)) as ex:
        list(ex.map(go, order))
    print(f"runs done in {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(
        [
            {
                "run": f"{r.model}/{r.name}",
                "kind": r.kind,
                "start": r.start,
                "end": r.end,
                "cpus": r.cpus,
                "ref": r.root == ref,
                "K": r.keep_cols is not None,
            }
            | r.res
            for r in [rc, *order]
        ]
    ).to_csv(out / "gate_runs.csv", index=False)

    # ---- the gates
    rows: list[dict] = []

    def emit(row: dict) -> None:
        print(
            ("PASS " if row["ok"] else "FAIL ") + json.dumps(row, default=str),
            flush=True,
        )
        rows.append(row)

    def need(*names, m):
        bad = [n for n in names if runs[(m, n)].res.get("rc") != 0]
        if bad:
            emit(
                {
                    "model": m,
                    "gate": "runs",
                    "ok": False,
                    "detail": f"failed runs {bad}",
                }
            )
        return not bad

    for m in [*models, *(["lstm"] if do_lstm else [])]:
        kind_runs = (
            [
                ("h1_pre_edit_re1", "u_pre_re1", "u_m0_re1"),
                ("h1_pre_edit_re10", "u_pre_re10", "u_m0_re10"),
                ("h1_pre_edit_tuned", "t_pre", "t_m0"),
            ]
            if m != "lstm"
            else [
                ("h1_pre_edit_lstm", "l_pre", "l_m0"),
                ("h1_pre_edit_lstm_re10_explicit", "l_pre", "l_m0_re10x"),
            ]
        )
        # h1: WINDOW_MASK=0 = pre-edit
        for gname, ra, rb in kind_runs:
            if not need(ra, rb, m=m):
                continue
            A, B = load(runs[(m, ra)]), load(runs[(m, rb)])
            bad = compare_npz(A, B, TIMING_KEYS)
            extra = sorted(set(B) - set(A) - {"_meta", "_dir"})
            tab = compare_tables(A["_dir"], B["_dir"], runs[(m, ra)].kind)
            md = meta_diff(A, B)
            emit(
                {
                    "model": m,
                    "gate": gname,
                    "arrays_compared": len(
                        [k for k in A if not k.startswith("_") and k not in TIMING_KEYS]
                    ),
                    "arrays_differ": bad,
                    "extra_keys_in_new": extra,
                    "tables_differ": tab,
                    "meta_differ": md,
                    "new_meta_window_mask": B["_meta"].get("window_mask"),
                    "ok": not bad
                    and not extra
                    and not tab
                    and not md
                    and B["_meta"].get("window_mask") == 0,
                }
            )
        # h2: full-rank design, mask 1 = mask 0
        pairs = (
            [
                ("h2_full_rank_re1", "u_k0_re1", "u_k1_re1"),
                ("h2_full_rank_re10", "u_k0_re10", "u_k1_re10"),
                ("h2_full_rank_tuned", "t_k0", "t_k1"),
            ]
            if m != "lstm"
            else [
                (f"h2_full_rank_lstm_re{r}", f"l_k0_re{r}", f"l_k1_re{r}")
                for r in LSTM_RE
            ]
        )
        for gname, ra, rb in pairs:
            if not need(ra, rb, m=m):
                continue
            A, B = load(runs[(m, ra)]), load(runs[(m, rb)])
            bad = compare_npz(A, B, TIMING_KEYS)
            tab = compare_tables(A["_dir"], B["_dir"], runs[(m, ra)].kind)
            kn = np.asarray(B.get("kept_n", []))
            tk = np.asarray(B.get("tune_kept_n", [])) if "tune_kept_n" in B else None
            full = bool(
                len(kn) and (kn == len(K)).all() and (np.asarray(B["kept"]) == 1).all()
            )
            emit(
                {
                    "model": m,
                    "gate": gname,
                    "K": len(K),
                    "n_features": B["_meta"].get("n_features"),
                    "arrays_differ": bad,
                    "tables_differ": tab,
                    "kept_n_all_K": full,
                    "tune_kept_n": None if tk is None else tk.tolist(),
                    "refits": len(kn),
                    "ok": not bad
                    and not tab
                    and full
                    and B["_meta"].get("n_features") == len(K)
                    and (tk is None or bool((tk == len(K)).all())),
                }
            )
        # h3: the mask on the slice
        cases = (
            ["u_m1_re1", "t_m1"] if m != "lstm" else [f"l_m1_re{r}" for r in LSTM_RE]
        )
        for rn in cases:
            if not need(rn, m=m):
                continue
            r = runs[(m, rn)]
            Z = load(r)
            i0 = r.start - s0
            refit = np.asarray(Z["refit_row"])
            want = np.array(
                [np.isin(np.arange(p), keeps[i0 + i]) for i in refit], dtype=np.uint8
            )
            kept = np.asarray(Z["kept"])
            row = {
                "model": m,
                "gate": f"h3_mask_{r.kind}"
                + (f"_{rn.rsplit('_', 1)[1]}" if m == "lstm" else ""),
                "run": rn,
                "refits": len(refit),
                "kept_n_min": int(Z["kept_n"].min()),
                "kept_n_max": int(Z["kept_n"].max()),
                "p": p,
                "kept_is_window_keep": eq(kept, want),
                "kept_n_is_row_sum": eq(Z["kept_n"], kept.sum(axis=1)),
            }
            ok = (
                row["kept_is_window_keep"]
                and row["kept_n_is_row_sum"]
                and row["kept_n_max"] < p
            )
            if "tune_kept" in Z:
                tr = np.asarray(Z["tune_row"])
                tw = np.array(
                    [np.isin(np.arange(p), keeps[i0 + i]) for i in tr], dtype=np.uint8
                )
                row["tune_kept_is_window_keep"] = eq(Z["tune_kept"], tw)
                ok = ok and row["tune_kept_is_window_keep"]
            if "importance" in Z:
                imp = np.asarray(Z["importance"], dtype=np.float64)
                rec = np.isfinite(imp).all(axis=1)
                row["importance_rows"] = int(rec.sum())
                row["importance_zero_on_dropped"] = bool(
                    (imp[rec][kept[rec] == 0] == 0).all()
                )
                ok = (
                    ok
                    and row["importance_zero_on_dropped"]
                    and row["importance_rows"] > 0
                )
            if "shap" in Z and np.asarray(Z["shap"]).size:
                sh = np.asarray(Z["shap"], dtype=np.float64)
                srow = np.isfinite(sh).all(axis=1)
                in_force = np.searchsorted(refit, np.arange(len(sh)), side="right") - 1
                dropped = kept[in_force] == 0
                row["shap_rows"] = int(srow.sum())
                row["shap_zero_on_dropped"] = bool(
                    (sh[:, :p][srow][dropped[srow]] == 0).all()
                )
                gap = float(
                    np.max(
                        np.abs(sh[srow].sum(axis=1) - np.asarray(Z["pred_adj"])[srow])
                    )
                )
                row["shap_additivity_max"] = gap
                row["shap_additivity_gap_meta"] = float(Z["shap_additivity_gap"])
                ok = (
                    ok
                    and row["shap_zero_on_dropped"]
                    and gap < SHAP_ADD_TOL
                    and row["shap_rows"] > 0
                )
            # how far the mask moves the forecasts (recorded, not a pass criterion)
            twin = {"u_m1_re1": None, "t_m1": "t_m0"}.get(
                rn, rn.replace("l_m1", "l_m0")
            )
            if twin and runs[(m, twin)].res.get("rc") == 0:
                Z0 = load(runs[(m, twin)])
                row["max_abs_pred_move_vs_mask0"] = float(
                    np.max(np.abs(Z["pred_adj"] - Z0["pred_adj"]))
                )
            row["ok"] = bool(ok)
            emit(row)
        if m == "lstm":
            # h4 (LSTM): REFIT_EVERY 1, two tuning periods as two chunks = one run
            if need("lc_U", "lc_C1", "lc_C2", m=m):
                U = load(runs[(m, "lc_U")])
                C = [load(runs[(m, n)]) for n in ("lc_C1", "lc_C2")]

                def cat(k: str) -> np.ndarray:
                    return np.concatenate([np.asarray(c[k]) for c in C])

                gl = {
                    "preds": eq(cat("pred_adj"), U["pred_adj"]),
                    "qsel_preds": eq(cat("pred_adj_qsel"), U["pred_adj_qsel"]),
                    "seed_preds": eq(cat("pred_adj_seeds"), U["pred_adj_seeds"]),
                    "refit_rows": eq(
                        np.concatenate(
                            [
                                np.asarray(c["refit_row"]) + int(c["oos_offset"]) - OOS0
                                for c in C
                            ]
                        ),
                        U["refit_row"],
                    ),
                    "cand_val_mse": eq(cat("cand_val_mse"), U["cand_val_mse"]),
                    "cand_epochs": eq(cat("cand_epochs"), U["cand_epochs"]),
                    "picks": eq(cat("chosen_mse"), U["chosen_mse"])
                    and eq(cat("chosen_qlike"), U["chosen_qlike"]),
                    "kept": eq(np.vstack([c["kept"] for c in C]), U["kept"]),
                    "kept_n": eq(cat("kept_n"), U["kept_n"]),
                    "tune_kept": eq(
                        np.vstack([c["tune_kept"] for c in C]), U["tune_kept"]
                    ),
                }
                refits = int(len(U["refit_row"]))
                emit(
                    {"model": m, "gate": "h4_chunks_lstm_re1"}
                    | gl
                    | {"refits": refits, "tunings": int(len(U["tune_row"]))}
                    | {
                        "ok": all(gl.values())
                        and refits == TC_ROWS
                        and len(U["tune_row"]) == 2
                    }
                )
            continue
        # h4: chunks with the mask on
        if need("u_c1_re1", "u_c2_re1", "u_m1_re1", m=m):
            C1, C2, U = (
                load(runs[(m, n)]) for n in ("u_c1_re1", "u_c2_re1", "u_m1_re1")
            )
            g = {
                "preds": eq(
                    np.concatenate([C1["pred_adj"], C2["pred_adj"]]), U["pred_adj"]
                ),
                "kept": eq(np.vstack([C1["kept"], C2["kept"]]), U["kept"]),
                "kept_n": eq(np.concatenate([C1["kept_n"], C2["kept_n"]]), U["kept_n"]),
            }
            emit(
                {"model": m, "gate": "h4_chunks_untuned_re1"}
                | g
                | {"ok": all(g.values())}
            )
        if need("tc_U", "tc_C1", "tc_C2", m=m):
            import reduce_trees_tuned_chunks as rtc

            tcr = scratch / "tuned_chunks"
            chunks = [(0, s0, s0 + TUNE_PER, W), (1, s0 + TUNE_PER, s0 + TC_ROWS, W)]
            gate = rtc.merge_arm(tcr, (BUCKET, SEG, m, W), chunks)
            row = {
                "model": m,
                "gate": "h4_chunks_tuned_re10",
                "reducer_ok": bool(gate["ok"]),
                "reducer_why": gate.get("why", ""),
            }
            if gate["ok"]:
                mdir = rtc.inner(rtc.arm_dir(tcr, BUCKET, SEG, m, W), m, BUCKET)
                with np.load(mdir / f"trees_{SEG}.npz", allow_pickle=False) as z:
                    M = {k: z[k] for k in z.files}
                U = load(runs[(m, "tc_U")])
                C = [load(runs[(m, n)]) for n in ("tc_C1", "tc_C2")]
                rt = "round_trip"
                mu = pd.read_csv(
                    inner(runs[(m, "tc_U")]) / f"results_qsel_{SEG}.csv",
                    float_precision=rt,
                )
                mc = pd.read_csv(mdir / f"results_qsel_{SEG}.csv", float_precision=rt)
                ccat = pd.concat(
                    [
                        pd.read_csv(
                            inner(runs[(m, n)]) / f"results_qsel_{SEG}.csv",
                            float_precision=rt,
                        )
                        for n in ("tc_C1", "tc_C2")
                    ],
                    ignore_index=True,
                )
                row["merged_qsel_csv_max_abs_vs_unchunked"] = float(
                    (mu["pred_adj"] - mc["pred_adj"]).abs().max()
                )
                gt: dict[str, bool | int] = {
                    "preds": eq(M["pred_adj"], U["pred_adj"]),
                    "qsel_preds_npz": eq(M["pred_adj_qsel"], U["pred_adj_qsel"]),
                    "qsel_chunk_csvs_eq_unchunked_csv": bool(
                        mu["pred_adj"].equals(ccat["pred_adj"])
                    ),
                    "merged_qsel_csv_within_tol": bool(
                        row["merged_qsel_csv_max_abs_vs_unchunked"] <= MERGED_CSV_TOL
                    ),
                    "refit_rows": eq(M["refit_row"], U["refit_row"]),
                    "importance": eq(M["importance"], U["importance"]),
                    "cand_val_mse": eq(M["cand_val_mse"], U["cand_val_mse"]),
                    "cand_val_qlike": eq(M["cand_val_qlike"], U["cand_val_qlike"]),
                    "picks": eq(M["chosen_mse"], U["chosen_mse"])
                    and eq(M["chosen_qlike"], U["chosen_qlike"]),
                    "kept": eq(np.vstack([c["kept"] for c in C]), U["kept"]),
                    "tune_kept": eq(
                        np.vstack([c["tune_kept"] for c in C]), U["tune_kept"]
                    ),
                    "tunings": int(len(U["tune_row"])),
                }
                row |= gt | {
                    "ok": all(v for k, v in gt.items() if k != "tunings")
                    and gt["tunings"] == 2
                }
            else:
                row["ok"] = False
            emit(row)

    tab = pd.DataFrame(rows)
    tab.to_csv(out / "gate_rows.csv", index=False)
    n_bad = int((~tab["ok"].astype(bool)).sum()) if len(tab) else 1
    failed = [f"{r.model}/{r.name}" for r in order if r.res.get("rc") != 0]
    print(
        f"\n{len(tab)} gate rows, {n_bad} failed; failed runs: {failed or 'none'}; {out / 'gate_rows.csv'}"
    )
    if n_bad == 0 and not failed:
        (out / "GATES_OK").write_text(
            time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8"
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
