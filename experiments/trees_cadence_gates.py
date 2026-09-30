"""Gates for the 16:00 tree cadence rungs (checklist I2: T10, T1, RS10, RS1) before any fleet.

Every run below is the SHIPPED spec run end to end as the cluster runs it (a
subprocess of specs/causal_tune_trees.py or specs/causal_tune_trees_tuned.py with its
env axes), on a smoke slice of the live_feasible 16:00 series (de-duplicated per-bar
design, commit 47f7f9c), for LightGBM, XGBoost and the random forest.  The slice U is
whole-series OOS rows [OOS0, OOS0 + U_ROWS) (START = W + OOS0, HALO = W = 2000);
OOS0 is a tuning row of the tuned spec (a multiple of its TUNE_PER = 250).  Every
untuned fit runs with SLURM_CPUS_PER_TASK = 1 (one thread; the spec passes that value
to the models), as every fleet process will.

  g0  design        the 16:00 live_feasible design has 232 columns and no HAR x
                    open / close interaction column (the de-duplicated design)
  g1  importance    REFIT_EVERY 1: IMPORTANCE_EVERY 10 leaves the forecasts and refit rows
                    bit-identical to IMPORTANCE_EVERY 1; importance and TreeSHAP rows are
                    the IMPORTANCE_EVERY 1 rows at every 10th whole-series refit and NaN
                    elsewhere (importance never feeds a forecast)
  g2  chunks (t1)   REFIT_EVERY 1, IMPORTANCE_EVERY 10: two chunks (HALO = W, the seam at
                    C1_ROWS, not a multiple of 10) merged by
                    experiments/trees_cadence_reduce_chunks.py = the unchunked slice bit
                    for bit (forecasts, targets, refit rows, importance incl. which refits
                    record it, TreeSHAP); pred_raw (a look-ahead smear, recomputed over the
                    merged rows) to float round-off
  g2b chunks (t10)  the same at REFIT_EVERY 10, IMPORTANCE_EVERY 1 (seam at 10 rows)
  g3  identity      the tuned spec at REFIT_EVERY 1 with TUNE_IDENTITY = 1 (a pool of
                    POOL workers) = the untuned spec at REFIT_EVERY 1 bit for bit
                    (forecasts, QLIKE-rule forecasts, importance, TreeSHAP)
  g4  processes     single-thread invariance: (a) the untuned chunk C2 run again ALONE
                    after everything else = its run among the concurrent gate processes,
                    bit for bit; every untuned run recorded threads = 1; (b) the tuned
                    spec, full grid, REFIT_EVERY 1, on a POOL_ROWS slice: a pool of 1 =
                    a pool of POOL, bit for bit (forecasts of both rules, every
                    candidate's validation losses and rounds, the picks)

Outputs (--out): gate_rows.csv (one row per gate x model), gate_runs.csv (one row per
spec run: seconds, fit seconds, the child's peak RSS on POSIX).
Exit status 1 when a gate fails.

Usage: python experiments/trees_cadence_gates.py [--models lgbm,xgb,rf] [--procs 4]
           [--pool 4] [--out results/linear_subsection_trees_dedup/gates]
           [--u-rows 24 --c1-rows 7 --pool-rows 8]
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
os.chdir(ROOT)
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

import trees_cadence_reduce_chunks as red  # noqa: E402

SPEC_U = "specs/causal_tune_trees.py"
SPEC_T = "specs/causal_tune_trees_tuned.py"
BUCKET, SEG, W = "live_feasible", "bar1600", 2000
OOS0 = 1000  # whole-series OOS row of the slice's first forecast: a tuning row (x 250)
N_COLS_DEDUP = 232  # live_feasible per-bar columns after commit 47f7f9c (244 before)
SESSION_EDGE = re.compile(r"_x_(open|close)$")
PRED_RAW_REL = (
    1e-12  # pred_raw is a smear over the run's rows: equal to float round-off
)


@dataclass
class Run:
    name: str
    spec: str
    model: str
    start: int
    end: int
    out: Path
    env: dict = field(default_factory=dict)
    cpus: int = 1
    res: dict = field(default_factory=dict)


def inner(out: Path, model: str) -> Path:
    return out / "causal_tune_trees" / model / BUCKET


_LOCK = threading.Lock()
_RUNNING = [0]


def launch(r: Run) -> Run:
    """Run the spec for r in a subprocess; r.res gets seconds, peak RSS, concurrency."""
    if r.out.exists():
        shutil.rmtree(r.out)
    r.out.mkdir(parents=True)
    env = dict(os.environ)
    env.update(
        HPC_KW_MODEL=r.model,
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
    env.update(r.env)
    cmd = [sys.executable, "-u", r.spec]
    with _LOCK:
        _RUNNING[0] += r.cpus
        r.res["concurrent_cpus_at_start"] = _RUNNING[0]
    a = time.time()
    rss = np.nan
    with open(r.out / "run.log", "w", encoding="utf-8") as log:
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        if hasattr(os, "wait4"):  # POSIX: this child's own peak RSS (Linux: kB)
            _pid, status, ru = os.wait4(p.pid, 0)
            p.returncode = os.waitstatus_to_exitcode(status)
            rss = ru.ru_maxrss / 2**20
        else:
            p.wait()
        rc = p.returncode
    with _LOCK:
        _RUNNING[0] -= r.cpus
    r.res |= {"rc": rc, "sec": time.time() - a, "max_rss_gib": rss}
    text = (r.out / "run.log").read_text(encoding="utf-8", errors="replace")
    if rc == 0:
        (r.out / "DONE").touch()
    else:
        print(
            f"RUN FAILED {r.name}: {text.strip().splitlines()[-1][:300] if text.strip() else ''}"
        )
    print(
        f"  {r.name}: rc {rc}, {r.res['sec']:.0f}s, rss {r.res['max_rss_gib']:.2f} GiB",
        flush=True,
    )
    return r


def load(r: Run) -> dict:
    d = inner(r.out, r.model)
    csv = pd.read_csv(d / f"results_{SEG}.csv")
    with np.load(d / f"trees_{SEG}.npz", allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out["meta"] = json.loads(str(out["meta"]))
    out["csv"] = csv
    tc = d / f"tune_candidates_{SEG}.csv"
    out["cands"] = pd.read_csv(tc) if tc.is_file() else None
    qc = d / f"results_qsel_{SEG}.csv"
    out["qsel_csv"] = pd.read_csv(qc) if qc.is_file() else None
    return out


def eq(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lgbm,xgb,rf")
    ap.add_argument(
        "--procs", type=int, default=4, help="concurrent single-thread spec runs"
    )
    ap.add_argument(
        "--pool", type=int, default=4, help="tuned-spec pool size (g3, g4b)"
    )
    ap.add_argument("--out", default="results/linear_subsection_trees_dedup/gates")
    ap.add_argument("--u-rows", type=int, default=24)
    ap.add_argument(
        "--c1-rows", type=int, default=7, help="t1 seam; not a multiple of 10"
    )
    ap.add_argument(
        "--t10-c1-rows", type=int, default=10, help="t10 seam; a multiple of 10"
    )
    ap.add_argument("--pool-rows", type=int, default=8)
    a = ap.parse_args()
    out = ROOT / a.out
    scratch = out / "scratch"
    out.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    s0 = W + OOS0
    u_end, c1_end, t10_c1_end = s0 + a.u_rows, s0 + a.c1_rows, s0 + a.t10_c1_rows
    assert a.c1_rows % 10 and 0 < a.c1_rows < a.u_rows and a.t10_c1_rows % 10 == 0
    ue = {"HPC_KW_SHAP_SEGMENTS": SEG}  # TreeSHAP on at 16:00 (the untuned default)

    def chunk_dir(rung: str, model: str, k: int) -> Path:
        return scratch / rung / BUCKET / SEG / model / f"tw{W}" / "chunks" / f"c{k}"

    def cad(re_: int, ie: int) -> dict:
        return ue | {"HPC_KW_REFIT_EVERY": str(re_), "HPC_KW_IMPORTANCE_EVERY": str(ie)}

    runs: dict[tuple[str, str], Run] = {}
    tuned: dict[tuple[str, str], Run] = {}
    for m in models:
        for name, start, end, o, env in (
            ("u_re1_ie1_U", s0, u_end, scratch / m / "u_re1_ie1_U", cad(1, 1)),
            ("u_re1_ie10_U", s0, u_end, scratch / m / "u_re1_ie10_U", cad(1, 10)),
            ("u_re1_ie10_C1", s0, c1_end, chunk_dir("t1", m, 0), cad(1, 10)),
            ("u_re1_ie10_C2", c1_end, u_end, chunk_dir("t1", m, 1), cad(1, 10)),
            ("u_re10_ie1_U", s0, u_end, scratch / m / "u_re10_ie1_U", cad(10, 1)),
            ("u_re10_ie1_C1", s0, t10_c1_end, chunk_dir("t10", m, 0), cad(10, 1)),
            ("u_re10_ie1_C2", t10_c1_end, u_end, chunk_dir("t10", m, 1), cad(10, 1)),
        ):
            runs[(m, name)] = Run(f"{m}/{name}", SPEC_U, m, start, end, o, env)
        te = {"HPC_KW_REFIT_EVERY": "1", "HPC_KW_QSEL": "1"}
        tuned[(m, "t_ident_U")] = Run(
            f"{m}/t_ident_U",
            SPEC_T,
            m,
            s0,
            u_end,
            scratch / m / "t_ident_U",
            te | ue | {"HPC_KW_TUNE_IDENTITY": "1"},
            cpus=a.pool,
        )
        for p in (1, a.pool):
            tuned[(m, f"t_grid_p{p}")] = Run(
                f"{m}/t_grid_p{p}",
                SPEC_T,
                m,
                s0,
                s0 + a.pool_rows,
                scratch / m / f"t_grid_p{p}",
                te | {"HPC_KW_TUNE_IDENTITY": "0", "HPC_KW_SHAP_SEGMENTS": "none"},
                cpus=p,
            )
    alone = {
        m: Run(
            f"{m}/u_re1_ie10_C2_alone",
            SPEC_U,
            m,
            c1_end,
            u_end,
            scratch / m / "u_re1_ie10_C2_alone",
            runs[(m, "u_re1_ie10_C2")].env,
        )
        for m in models
    }

    t0 = time.time()
    print(
        f"phase 1: {len(runs)} untuned runs, {a.procs} at a time; slice OOS rows "
        f"[{OOS0}, {OOS0 + a.u_rows}) of {BUCKET} {SEG}",
        flush=True,
    )
    with ThreadPoolExecutor(a.procs) as ex:
        list(ex.map(launch, runs.values()))
    print(
        f"phase 2: {len(tuned)} tuned runs, {max(1, a.procs // a.pool)} at a time",
        flush=True,
    )
    with ThreadPoolExecutor(max(1, a.procs // a.pool)) as ex:
        list(ex.map(launch, tuned.values()))
    print("phase 3: the C2 chunk again, alone", flush=True)
    for r in alone.values():
        launch(r)
    print(f"runs done in {time.time() - t0:.0f}s", flush=True)

    every = list(runs.values()) + list(tuned.values()) + list(alone.values())
    run_rows = []
    for r in every:
        row = {
            "run": r.name,
            "spec": r.spec,
            "model": r.model,
            "start": r.start,
            "end": r.end,
            "cpus": r.cpus,
        } | r.res
        if r.res.get("rc") == 0:
            z = load(r)
            row |= {
                "rows": len(z["csv"]),
                "refits": len(z["refit_row"]),
                "fit_sec_mean": float(np.mean(z["fit_sec"])),
                "threads_meta": z["meta"].get("threads"),
                "refit_every_meta": z["meta"].get("refit_every"),
                "n_features": z["meta"].get("n_features"),
                "walk_sec": z["meta"].get("walk_sec"),
                "run_sec": z["meta"].get("run_sec"),
            }
        run_rows.append(row)
    pd.DataFrame(run_rows).to_csv(out / "gate_runs.csv", index=False)
    failed_runs = [r.name for r in every if r.res.get("rc") != 0]

    rows: list[dict] = []

    def emit(row: dict) -> None:
        print(json.dumps(row, default=str), flush=True)
        rows.append(row)

    for m in models:
        need = [
            runs[(m, k)]
            for k in (
                "u_re1_ie1_U",
                "u_re1_ie10_U",
                "u_re1_ie10_C1",
                "u_re1_ie10_C2",
                "u_re10_ie1_U",
                "u_re10_ie1_C1",
                "u_re10_ie1_C2",
            )
        ]
        need += [
            tuned[(m, k)] for k in ("t_ident_U", "t_grid_p1", f"t_grid_p{a.pool}")
        ] + [alone[m]]
        bad = [r.name for r in need if r.res.get("rc") != 0]
        if bad:
            emit(
                {
                    "model": m,
                    "gate": "runs",
                    "ok": False,
                    "detail": f"failed runs: {bad}",
                }
            )
            continue
        A = load(runs[(m, "u_re1_ie1_U")])
        B = load(runs[(m, "u_re1_ie10_U")])

        # g0: the de-duplicated design
        names = [str(s) for s in A["feature_names"]]
        edge = [n for n in names if SESSION_EDGE.search(n)]
        hhmm = pd.to_datetime(A["csv"]["date"]).dt.strftime("%H:%M")
        emit(
            {
                "model": m,
                "gate": "g0_design",
                "n_features": len(names),
                "want": N_COLS_DEDUP,
                "session_edge_cols": len(edge),
                "rows": len(A["csv"]),
                "all_16_00": bool((hhmm == "16:00").all()),
                "first": str(A["csv"]["date"].iloc[0]),
                "ok": len(names) == N_COLS_DEDUP
                and not edge
                and len(A["csv"]) == a.u_rows
                and bool((hhmm == "16:00").all()),
            }
        )

        # g1: IMPORTANCE_EVERY never feeds a forecast
        whole = OOS0 + np.asarray(
            B["refit_row"]
        )  # REFIT_EVERY 1: refit index = OOS row
        rec = whole % 10 == 0
        imp_b = np.asarray(B["importance"])
        shp_a, shp_b = np.asarray(A["shap"]), np.asarray(B["shap"])
        ok_rows = np.isfinite(imp_b).all(axis=1)
        g1 = {
            "preds_bit_identical": eq(A["pred_adj"], B["pred_adj"]),
            "csv_pred_adj_identical": eq(A["csv"]["pred_adj"], B["csv"]["pred_adj"]),
            "refit_rows_identical": eq(A["refit_row"], B["refit_row"]),
            "recorded_refits": int(rec.sum()),
            "importance_rows_finite_exactly_at_recorded": bool(
                np.array_equal(ok_rows, rec)
            ),
            "importance_equal_at_recorded": eq(
                imp_b[rec], np.asarray(A["importance"])[rec]
            ),
            "shap_equal_at_recorded": eq(shp_b[rec], shp_a[rec]),
            "shap_nan_elsewhere": bool(np.isnan(shp_b[~rec]).all()),
            "ie1_all_recorded": bool(np.isfinite(np.asarray(A["importance"])).all()),
        }
        emit(
            {"model": m, "gate": "g1_importance_every"}
            | g1
            | {"ok": all(v for k, v in g1.items() if k != "recorded_refits")}
        )

        # g2 / g2b: chunks merged by the fleet's reducer = the unchunked slice
        for gname, rung, full, c1e in (
            ("g2_chunks_t1", "t1", B, c1_end),
            ("g2b_chunks_t10", "t10", load(runs[(m, "u_re10_ie1_U")]), t10_c1_end),
        ):
            re_, ie = red.RUNGS[rung]
            chunks = [(0, s0, c1e, W), (1, c1e, u_end, W)]
            gate = red.merge_arm(
                scratch / rung, (BUCKET, SEG, m, W), chunks, re_, ie, write=False
            )
            row = {
                "model": m,
                "gate": gname,
                "reducer_ok": bool(gate["ok"]),
                "reducer_why": gate["why"],
            }
            if gate["ok"]:
                res, mz = gate["_merged"]
                fc = full["csv"]
                rawrel = float(
                    np.max(
                        np.abs(
                            res["pred_raw"].to_numpy() / fc["pred_raw"].to_numpy() - 1
                        )
                    )
                )
                row |= {
                    "preds_bit_identical": eq(mz["pred_adj"], full["pred_adj"]),
                    "csv_identical_but_pred_raw": bool(
                        (
                            res["date"].astype(str).to_numpy()
                            == fc["date"].astype(str).to_numpy()
                        ).all()
                        and eq(res["true_adj"], fc["true_adj"])
                        and eq(res["true_raw"], fc["true_raw"])
                        and eq(res["pred_adj"], fc["pred_adj"])
                    ),
                    "refit_rows_identical": eq(mz["refit_row"], full["refit_row"]),
                    "importance_identical_incl_nan_rows": eq(
                        mz["importance"], full["importance"]
                    ),
                    "recorded_refits": int(
                        np.isfinite(np.asarray(mz["importance"])).all(axis=1).sum()
                    ),
                    "shap_identical": eq(mz["shap"], full["shap"]),
                    "pred_raw_max_rel": rawrel,
                }
                row["ok"] = bool(
                    row["preds_bit_identical"]
                    and row["csv_identical_but_pred_raw"]
                    and row["refit_rows_identical"]
                    and row["importance_identical_incl_nan_rows"]
                    and row["shap_identical"]
                    and rawrel < PRED_RAW_REL
                )
            else:
                row["ok"] = False
            emit(row)

        # g3: tuned spec, identity grid, REFIT_EVERY 1 = untuned spec, REFIT_EVERY 1
        T = load(tuned[(m, "t_ident_U")])
        g3 = {
            "tuned_refit_every": T["meta"].get("refit_every"),
            "tuned_workers": T["meta"].get("workers"),
            "preds_bit_identical": eq(T["pred_adj"], A["pred_adj"]),
            # the QLIKE-rule CSV against the untuned CSV: both parsed from text the same way
            # (the npz pred_adj is read back from the results CSV, pred_adj_qsel is the
            # in-memory array, so those two may differ in the last bit of the text round trip)
            "qsel_csv_preds_bit_identical": T["qsel_csv"] is not None
            and eq(T["qsel_csv"]["pred_adj"], A["csv"]["pred_adj"]),
            "npz_qsel_vs_pred_adj_max_abs": float(
                np.max(np.abs(T["pred_adj_qsel"] - T["pred_adj"]))
            ),
            "importance_bit_identical": eq(T["importance"], A["importance"]),
            "shap_bit_identical": eq(T["shap"], A["shap"]),
            "refits": len(T["refit_row"]),
        }
        emit(
            {"model": m, "gate": "g3_tuned_identity"}
            | g3
            | {
                "ok": bool(
                    g3["tuned_refit_every"] == 1
                    and g3["preds_bit_identical"]
                    and g3["qsel_csv_preds_bit_identical"]
                    and g3["importance_bit_identical"]
                    and g3["shap_bit_identical"]
                )
            }
        )

        # g4a: a process's results do not depend on how many processes run beside it
        C2, C2a = load(runs[(m, "u_re1_ie10_C2")]), load(alone[m])
        thr = [load(r)["meta"].get("threads") for r in need if r.spec == SPEC_U]
        g4 = {
            "concurrent_cpus_when_run": runs[(m, "u_re1_ie10_C2")].res.get(
                "concurrent_cpus_at_start"
            ),
            "alone_cpus_when_run": alone[m].res.get("concurrent_cpus_at_start"),
            "preds_bit_identical": eq(C2["pred_adj"], C2a["pred_adj"]),
            "importance_bit_identical": eq(C2["importance"], C2a["importance"]),
            "shap_bit_identical": eq(C2["shap"], C2a["shap"]),
            "untuned_threads_all_1": all(t == 1 for t in thr),
        }
        emit(
            {"model": m, "gate": "g4a_process_count"}
            | g4
            | {
                "ok": bool(
                    g4["preds_bit_identical"]
                    and g4["importance_bit_identical"]
                    and g4["shap_bit_identical"]
                    and g4["untuned_threads_all_1"]
                )
            }
        )

        # g4b: the tuned spec's pool size does not move a number
        P1, PP = load(tuned[(m, "t_grid_p1")]), load(tuned[(m, f"t_grid_p{a.pool}")])
        cols = [
            "tune_idx",
            "cand_idx",
            "rounds",
            "val_mse",
            "val_qlike",
            "chosen_mse",
            "chosen_qlike",
        ]
        g4b = {
            "candidates": int(len(P1["cands"])) if P1["cands"] is not None else 0,
            "preds_bit_identical": eq(P1["pred_adj"], PP["pred_adj"]),
            "qsel_preds_bit_identical": eq(P1["pred_adj_qsel"], PP["pred_adj_qsel"]),
            "candidate_records_identical": bool(
                P1["cands"] is not None
                and PP["cands"] is not None
                and P1["cands"][cols].equals(PP["cands"][cols])
            ),
            "cand_val_mse_bit_identical": eq(P1["cand_val_mse"], PP["cand_val_mse"]),
            "importance_bit_identical": eq(P1["importance"], PP["importance"]),
            "workers": [P1["meta"].get("workers"), PP["meta"].get("workers")],
            "refit_every": P1["meta"].get("refit_every"),
        }
        emit(
            {"model": m, "gate": "g4b_tuned_pool"}
            | g4b
            | {
                "ok": bool(
                    g4b["preds_bit_identical"]
                    and g4b["qsel_preds_bit_identical"]
                    and g4b["candidate_records_identical"]
                    and g4b["cand_val_mse_bit_identical"]
                    and g4b["importance_bit_identical"]
                    and g4b["refit_every"] == 1
                )
            }
        )

    tab = pd.DataFrame(rows)
    tab.to_csv(out / "gate_rows.csv", index=False)
    n_bad = int((~tab["ok"].astype(bool)).sum()) if len(tab) else 1
    print(
        f"\n{len(tab)} gate rows, {n_bad} failed; failed runs: {failed_runs or 'none'}; "
        f"{out / 'gate_rows.csv'}"
    )
    if n_bad == 0 and not failed_runs:
        (out / "GATES_OK").write_text(
            time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8"
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
