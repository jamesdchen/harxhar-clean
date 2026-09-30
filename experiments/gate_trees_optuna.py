"""Local gates for specs/causal_tune_trees_optuna.py before it is shipped (checklist I3).

Data: the spec's exact run_executor invocation on the live_feasible bar1600 series of the
de-duplicated design, sliced through run_executor's START / END / HALO seam (HALO = the
training window), the design matrix captured as the executor hands it to fit_predict.
Captures: the unchunked smoke slice U = whole-series OOS rows SLICE_OOS_START ..
+ SLICE_ROWS (SLICE_OOS_START a multiple of every TUNE_PER, so every path's tuning points
lie inside U; SLICE_ROWS spans two TUNE_PER = 5 points), and its two chunks C1 = the first
CHUNK_SPLIT rows, C2 = the rest.

Per model (lgbm, xgb, rf):
  coverage  the continuous search space contains every value of the random-search grid of
            specs/causal_tune_trees_tuned.py on the axes both search, and the shipped
            configuration (trial 0)
  identity  IDENTITY=1 (one trial = the shipped configuration, early stopping off) through
            both stages on U: every path's forecasts, native importance and TreeSHAP are
            bit-identical to the UNTUNED spec's fit_predict_tree with REFIT_EVERY = 1 and
            IMPORTANCE_EVERY = 10 (one thread) on U; constructor params equal
  smoke     stage 1 with N_TRIALS = 50 on U (pooled), stage 2 on U: seconds per trial, per
            tuning point, per refit; best trial per point; distinct fits
  determ    the first tuning point of U re-run in this process equals the pooled run's
            (params, losses, rounds of every trial, bit for bit)
  prefix    that point as a 25-trial study equals the first 25 trials of the 50-trial study
  chunk     stage 1 on C1 and C2 equals U's records bit for bit; stage 2 on C1 and C2 (from
            the chunked records) equals U's forecasts, importance, TreeSHAP and configuration
            rows on every path
  pool      stage 1 and stage 2 on C2 with one process equal the pooled runs bit for bit
  perturb   U with the target x PERTURB_FACTOR on every row from the forecast row t_p of an
            interior point on: every tuning record of a point <= t_p and every forecast of
            a row <= t_p is bit-identical on every path; later forecasts move

  mask_equiv (--mask 1 only) on the columns that are kept in EVERY window of U (so the design
            has no column constant on a window and no copy), stage 1 and stage 2 with the mask
            on equal the mask off bit for bit, and every kept count is the design's width

--mask 1 runs every gate with WINDOW_MASK=1 (user decision 2026-09-29: the per-window column
mask of src/models/window_mask.py); the identity comparator is then the untuned spec with
WINDOW_MASK=1 when it has that axis, else the untuned spec's own make_model /
native_importance / contributions fitted row by row on window_keep of each window (the same
rule written out independently of the Optuna spec's refit).  determ / prefix run the first
tuning point of U as three concurrent spawned processes (50, 50 and 25 trials).  The smoke
runs --smoke-trials trials; chunk / pool / perturb run --gate-trials (the smoke is re-run at
that count when they differ; a count above 10 runs TPE past its random start-up).

Usage: python experiments/gate_trees_optuna.py [--models lgbm,xgb,rf] [--mask 0|1]
           [--gates coverage,identity,smoke,determ,prefix,chunk,pool,perturb,mask_equiv]
           [--pool 4] [--smoke-trials 50] [--gate-trials 50] [--out DIR] [--scratch DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

SPEC = ROOT / "specs" / "causal_tune_trees_optuna.py"
TUNED_SPEC = ROOT / "specs" / "causal_tune_trees_tuned.py"
TREE_SPEC = ROOT / "specs" / "causal_tune_trees.py"
_TREE_RUN_CELL = '\nout_csv = os.path.join(OUT_DIR, "results.csv")\n'

BUCKET, SEGMENT, TW = "live_feasible", "bar1600", 2000
SLICE_OOS_START = (
    1000  # a multiple of 1, 5, 25 and 250: every path's tuning points lie in U
)
SLICE_ROWS = 6  # two TUNE_PER = 5 tuning points (rows 1000 and 1005)
CHUNK_SPLIT = 3  # C1 = the first 3 rows, C2 = the other 3
PERTURB_ROW = 3  # U-relative forecast row t_p: points / rows <= it must not move
PERTURB_FACTOR = 50.0  # the target x 50 on every row from t_p on
PREFIX_TRIALS = 25
W = TW  # one row per session at a one-bar segment
MASK = False  # --mask: WINDOW_MASK for every spec run of this invocation


def set_env(
    model: str,
    *,
    stage: str,
    identity: bool,
    start: int,
    end: int,
    cpus: int,
    trials: int,
    stage1: str = "",
) -> None:
    os.environ.update(
        HPC_KW_MODEL=model,
        HPC_KW_EXOG_BUCKET=BUCKET,
        HPC_KW_SEGMENT=SEGMENT,
        HPC_KW_TRAIN_WIN=str(TW),
        HPC_KW_LAG_SCOPE="global",
        HPC_KW_SHAP_SEGMENTS=SEGMENT,
        HPC_KW_START=str(start),
        HPC_KW_END=str(end),
        HPC_KW_HALO=str(W),
        HPC_KW_STAGE=stage,
        HPC_KW_IDENTITY="1" if identity else "0",
        HPC_KW_N_TRIALS=str(trials),
        HPC_KW_IMPORTANCE_EVERY="10",
        HPC_KW_REFIT_EVERY="1",  # read only by the untuned spec (the identity comparator)
        HPC_KW_RESUME="0",
        HPC_KW_STAGE1_TRIALS=stage1,
        HPC_KW_WINDOW_MASK="1" if MASK else "0",
        SLURM_CPUS_PER_TASK=str(cpus),
    )


def exec_spec() -> dict:
    ns: dict = {"__file__": str(SPEC), "__name__": "optuna_setup"}
    exec(compile(SPEC.read_text(encoding="utf-8"), str(SPEC), "exec"), ns)
    return ns


def exec_untuned() -> dict:
    head, sep, _ = TREE_SPEC.read_text(encoding="utf-8").partition(_TREE_RUN_CELL)
    assert sep
    ns: dict = {"__file__": str(TREE_SPEC), "__name__": "untuned_setup"}
    exec(compile(head, str(TREE_SPEC), "exec"), ns)
    return ns


def capture(scratch: Path, oos0: int, rows: int) -> dict:
    """X, y, W, names and OOS stamps of whole-series OOS rows [oos0, oos0 + rows)."""
    cache = scratch / f"capture_{BUCKET}_{SEGMENT}_{oos0}_{rows}.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        print(f"capture: cached {cache.name} X{d['X'].shape}")
        return d
    from src.backtest.executor import run_executor
    from src.data.loading import get_bucket

    start = W + oos0
    set_env(
        "lgbm",
        stage="tune",
        identity=True,
        start=start,
        end=start + rows,
        cpus=1,
        trials=1,
    )
    ns = exec_spec()
    got: dict = {}

    def grab(X_chunk, y_chunk, train_win_periods, hyperparams):
        got.update(
            X=np.array(X_chunk, dtype=np.float64),
            y=np.array(y_chunk, dtype=np.float64),
            W=int(train_win_periods),
            names=np.array([str(c) for c in hyperparams["_feature_names"]]),
        )
        return np.zeros(len(X_chunk) - int(train_win_periods))

    out_csv = scratch / f"capture_{oos0}_{rows}" / "results.csv"
    a = time.time()
    run_executor(  # the spec's __main__ invocation, argument for argument
        method_name="gate_capture",
        fit_predict=grab,
        hyperparams={},
        data_path=ns["T"]["DATA_PATH"],
        output_file=str(out_csv),
        horizon=ns["T"]["HORIZON"],
        train_window=TW,
        start=start,
        end=start + rows,
        halo=W,
        exog_cols=get_bucket(BUCKET),
        segment=SEGMENT,
        lag_scope="global",
        har_lags=None,
        add_calendar=True,
        target_use_diurnal=True,
        target_winsor_window=240,
        dropna_with_exog=False,
        overnight_fill=True,
        impute_indicate=True,
        diurnal_mode="divide",
        prescale=True,
        seed=ns["SEED"],
    )
    res = pd.read_csv(out_csv.with_name(f"results_{SEGMENT}.csv"))
    assert got["W"] == W and len(res) == len(got["X"]) - W == rows, (got["W"], len(res))
    got["dates"] = res["date"].astype(str).to_numpy().astype("U19")
    got["prep_sec"] = time.time() - a
    scratch.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **got)
    print(f"capture: {got['X'].shape} in {got['prep_sec']:.0f}s -> {cache.name}")
    return got


def stage1(
    model, d, oos0, *, identity=False, cpus=1, trials=50, y=None
) -> tuple[dict, float, dict]:
    """Stage 1 on a capture: (stacked records, seconds, namespace)."""
    start = W + oos0
    set_env(
        model,
        stage="tune",
        identity=identity,
        start=start,
        end=start + len(d["X"]) - W,
        cpus=cpus,
        trials=trials,
    )
    ns = exec_spec()
    a = time.time()
    pr = ns["fit_predict_tune"](
        d["X"],
        d["y"] if y is None else y,
        int(d["W"]),
        {"_feature_names": list(d["names"])},
    )
    assert np.isnan(pr).all()
    return ns["stack_records"](ns["SIDE"]["records"]), time.time() - a, ns


def write_records(ns: dict, st: dict, path: Path) -> str:
    """The stage-1 records as the reducer writes an arm's merged file (stage 2's input)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, meta=json.dumps(ns["run_meta"]({"oos_offset": int(st["row"][0])})), **st
    )
    return str(path)


def stage2(
    model, d, oos0, records: str, *, identity=False, cpus=1, y=None
) -> tuple[dict, float, dict]:
    start = W + oos0
    set_env(
        model,
        stage="refit",
        identity=identity,
        start=start,
        end=start + len(d["X"]) - W,
        cpus=cpus,
        trials=1 if identity else 50,
        stage1=records,
    )
    ns = exec_spec()
    a = time.time()
    pr = ns["fit_predict_refit"](
        d["X"],
        d["y"] if y is None else y,
        int(d["W"]),
        {"_feature_names": list(d["names"])},
    )
    side = dict(ns["SIDE"])
    assert np.array_equal(pr, side["paths"][ns["SHAP_PATH"]]["preds"])
    return side, time.time() - a, ns


REC_KEYS = (
    "row",
    "seed",
    "split",
    "params",
    "val_mse",
    "val_qlike",
    "rounds",
    "rounds_max",
    "patience",
    "complete",
    "n_kept",
)
PATH_KEYS = (
    "preds",
    "importance",
    "cfg_point",
    "cfg_trial",
    "cfg_rounds",
    "cfg_params",
    "n_kept",
)


def same_records(a: dict, b: dict, sl: slice | None = None) -> bool:
    sl = sl or slice(None)
    return all(
        np.array_equal(np.asarray(a[k])[sl], np.asarray(b[k])[sl], equal_nan=True)
        for k in REC_KEYS
    )


def same_paths(a: dict, b: dict, sl: slice | None = None) -> dict[str, bool]:
    sl = sl or slice(None)
    out = {}
    for path in a:
        ok = all(
            np.array_equal(
                np.asarray(a[path][k])[sl], np.asarray(b[path][k])[sl], equal_nan=True
            )
            for k in PATH_KEYS
        )
        if a[path]["shap"] is not None:
            ok = ok and np.array_equal(
                a[path]["shap"][sl], b[path]["shap"][sl], equal_nan=True
            )
        out[path] = bool(ok)
    return out


def concat_paths(a: dict, b: dict) -> dict:
    out = {}
    for path in a:
        out[path] = {k: np.concatenate([a[path][k], b[path][k]]) for k in PATH_KEYS}
        out[path]["shap"] = (
            None
            if a[path]["shap"] is None
            else np.vstack([a[path]["shap"], b[path]["shap"]])
        )
    return out


def masked_untuned(
    un: dict, d: dict, oos0: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """specs/causal_tune_trees.py's model, refit every row on window_keep of its window
    (REFIT_EVERY = 1, importance / TreeSHAP at whole-series rows that are multiples of 10),
    written out here independently of the Optuna spec's refit."""
    from src.models.window_mask import scatter, window_keep

    X, y = d["X"], d["y"]
    n, p = len(X) - W, X.shape[1]
    preds = np.empty(n)
    imp = np.full((n, p), np.nan, dtype=np.float32)
    shap = np.full((n, p + 1), np.nan, dtype=np.float32)
    for i in range(n):
        t = W + i
        keep = window_keep(X[t - W : t])
        m = un["make_model"]()
        m.fit(np.ascontiguousarray(X[t - W : t][:, keep]), y[t - W : t])
        Xb = np.ascontiguousarray(X[t : t + 1][:, keep])
        preds[i] = np.asarray(m.predict(Xb), dtype=np.float64)[0]
        if (oos0 + i) % 10 == 0:
            imp[i] = scatter(un["native_importance"](m, len(keep)), keep, p)
            shap[i] = scatter(un["contributions"](m, Xb), keep, p)[0]
    return preds, imp, shap


def first_point_job(ns: dict, d: dict, trials: int) -> dict:
    """The tuning job of the capture's first forecast row, as the spec builds it."""
    return dict(
        X=d["X"][0:W],
        y=d["y"][0:W],
        fit=(0, W - ns["VAL_TAIL"] - ns["EMBARGO"]),
        val=(W - ns["VAL_TAIL"], W),
        row=SLICE_OOS_START,
        seed=ns["OJ"].TPE_SEED_BASE + SLICE_OOS_START,
        n_trials=trials,
        early_stop=ns["EARLY_STOP"],
        mask=MASK,
    )


def old_grid(model: str) -> dict:
    """The random-search grid of specs/causal_tune_trees_tuned.py (its build_grid, executed)."""
    os.environ.update(
        HPC_KW_MODEL=model, HPC_KW_TUNE_IDENTITY="0", HPC_KW_REFIT_EVERY="10"
    )
    src = TUNED_SPEC.read_text(encoding="utf-8").partition(
        "\n# %%\nTRACE: list[dict] = []"
    )[0]
    ns: dict = {"__file__": str(TUNED_SPEC), "__name__": "tuned_grid"}
    exec(compile(src, str(TUNED_SPEC), "exec"), ns)
    return ns["AXES"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lgbm,xgb,rf")
    ap.add_argument(
        "--gates", default="coverage,identity,smoke,determ,prefix,chunk,pool,perturb"
    )
    ap.add_argument(
        "--pool",
        type=int,
        default=4,
        help="worker processes for the pooled runs (local cap 4)",
    )
    ap.add_argument(
        "--gate-trials",
        type=int,
        default=50,
        help="trials in the chunk / pool / perturb / mask_equiv runs",
    )
    ap.add_argument(
        "--smoke-trials", type=int, default=50, help="trials of the smoke run"
    )
    ap.add_argument("--mask", type=int, default=0, choices=(0, 1), help="WINDOW_MASK")
    ap.add_argument("--out", default="results/linear_subsection_trees_optuna/gates")
    ap.add_argument(
        "--scratch",
        default=os.environ.get(
            "GATE_SCRATCH", "results/linear_subsection_trees_optuna/gates/scratch"
        ),
    )
    a = ap.parse_args()
    global MASK
    MASK = bool(a.mask)
    sfx = "_mask" if MASK else ""
    out, scratch = ROOT / a.out, ROOT / a.scratch
    out.mkdir(parents=True, exist_ok=True)
    gates = {g.strip() for g in a.gates.split(",") if g.strip()}
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    tag = "_".join(models) + sfx
    rows: list[dict] = []

    def emit(row: dict, fatal: bool) -> None:
        print(json.dumps(row, default=str), flush=True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / f"gate_rows_{tag}.csv", index=False)
        if fatal and not row.get("ok", True):
            sys.exit(f"GATE {row['gate']} FAILED for {row['model']}: STOP")

    U = capture(scratch, SLICE_OOS_START, SLICE_ROWS)
    C1 = C2 = None
    if gates & {"chunk", "pool"}:
        C1 = capture(scratch, SLICE_OOS_START, CHUNK_SPLIT)
        C2 = capture(scratch, SLICE_OOS_START + CHUNK_SPLIT, SLICE_ROWS - CHUNK_SPLIT)
        ok = all(np.array_equal(C1[k], U[k][: len(C1[k])]) for k in ("X", "y")) and all(
            np.array_equal(C2[k], U[k][CHUNK_SPLIT:]) for k in ("X", "y")
        )
        ok = ok and np.array_equal(np.r_[C1["dates"], C2["dates"]], U["dates"])
        emit({"model": "-", "gate": "chunk_inputs", "ok": bool(ok)}, fatal=True)

    for model in models:
        key = {
            "model": model,
            "bucket": BUCKET,
            "segment": SEGMENT,
            "pool": a.pool,
            "window_mask": MASK,
        }
        if "coverage" in gates:
            grid = old_grid(model)
            set_env(
                model,
                stage="tune",
                identity=False,
                start=W + SLICE_OOS_START,
                end=W + SLICE_OOS_START + SLICE_ROWS,
                cpus=1,
                trials=50,
            )
            ns = exec_spec()
            space = ns["OJ"].SPACES[model]
            ship = ns["OJ"].shipped_config(model)
            bad = []
            for ax, vals in list(grid.items()) + [(k, [v]) for k, v in ship.items()]:
                if ax not in space:
                    bad.append(f"{ax} not searched")
                    continue
                s = space[ax]
                for v in vals:
                    if s[0] == "cat":
                        inside = v in s[1]
                    else:
                        inside = v is not None and s[1] <= v <= s[2]
                    if not inside:
                        bad.append(f"{ax}={v} outside {s}")
            emit(
                key
                | {
                    "gate": "coverage",
                    "old_grid": json.dumps(
                        {k: [str(x) for x in v] for k, v in grid.items()}
                    ),
                    "shipped": json.dumps(ship, default=str),
                    "outside": "; ".join(bad),
                    "ok": not bad,
                },
                fatal=True,
            )

        if "identity" in gates:
            st, s1, ns = stage1(
                model, U, SLICE_OOS_START, identity=True, cpus=a.pool, trials=1
            )
            ship_vec = ns["OJ"].encode(model, ns["OJ"].shipped_config(model))
            trial0_shipped = bool(
                np.array_equal(
                    st["params"][:, 0, :],
                    np.tile(ship_vec, (len(st["row"]), 1)),
                    equal_nan=True,
                )
            )
            rec = write_records(ns, st, scratch / f"identity_{model}" / "trials.npz")
            side, s2, ns2 = stage2(
                model, U, SLICE_OOS_START, rec, identity=True, cpus=a.pool
            )
            os.environ["SLURM_CPUS_PER_TASK"] = (
                "1"  # the untuned spec's threads; REFIT_EVERY=1, IMPORTANCE_EVERY=10 set above
            )
            un = exec_untuned()
            assert (
                un["REFIT_EVERY"] == 1
                and un["IMPORTANCE_EVERY"] == 10
                and un["N_THREADS"] == 1
            )
            un_has_mask = "WINDOW_MASK" in un
            same_params = (
                ns2["J"].build_model(ns2["OJ"].shipped_config(model), None).get_params()
                == un["make_model"]().get_params()
            )
            a0 = time.time()
            if MASK and not un_has_mask:
                comparator = (
                    "untuned make_model on window_keep, row by row (this script)"
                )
                p_un, imp_un, shap_un = masked_untuned(un, U, SLICE_OOS_START)
            else:
                comparator = "specs/causal_tune_trees.py fit_predict_tree" + (
                    " (WINDOW_MASK=1)" if MASK else ""
                )
                assert not MASK or bool(un["WINDOW_MASK"])
                p_un = np.asarray(
                    un["fit_predict_tree"](
                        U["X"], U["y"], W, {"_feature_names": list(U["names"])}
                    )
                )
                imp_un, shap_un = un["SIDE"]["importance"], un["SIDE"]["shap"]
            per_path = {
                path: bool(
                    np.array_equal(d["preds"], p_un)
                    and np.array_equal(d["importance"], imp_un, equal_nan=True)
                )
                for path, d in side["paths"].items()
            }
            shap_ok = bool(
                np.array_equal(
                    side["paths"][ns2["SHAP_PATH"]]["shap"], shap_un, equal_nan=True
                )
            )
            row = key | {
                "gate": "identity",
                "n": len(p_un),
                "trial0_is_shipped": trial0_shipped,
                "constructor_params_equal": bool(same_params),
                "paths_bit_identical": json.dumps(per_path),
                "shap_bit_identical": shap_ok,
                "comparator": comparator,
                "importance_rows_recorded": int(np.isfinite(imp_un[:, 0]).sum()),
                "n_kept_stage1": json.dumps(st["n_kept"].tolist()),
                "sec_stage1": s1,
                "sec_stage2": s2,
                "sec_untuned": time.time() - a0,
            }
            row["ok"] = bool(
                trial0_shipped and same_params and all(per_path.values()) and shap_ok
            )
            emit(row, fatal=True)

        if gates & {"determ", "prefix"}:
            set_env(
                model,
                stage="tune",
                identity=False,
                start=W + SLICE_OOS_START,
                end=W + SLICE_OOS_START + SLICE_ROWS,
                cpus=1,
                trials=50,
            )
            ns = exec_spec()
            jobs = [first_point_job(ns, U, 50), first_point_job(ns, U, 50)]
            jobs.append(first_point_job(ns, U, PREFIX_TRIALS))
            with ns["pool_for"](3) as pool:
                ra, rb, rc = list(pool.map(ns["OJ"].tune_point, jobs))
            fields = (
                "params",
                "val_mse",
                "val_qlike",
                "rounds",
                "rounds_max",
                "n_kept",
            )
            if "determ" in gates:
                ok = ra["pid"] != rb["pid"] and all(
                    np.array_equal(ra[f], rb[f], equal_nan=True) for f in fields
                )
                emit(
                    key
                    | {
                        "gate": "determ",
                        "row": SLICE_OOS_START,
                        "trials": 50,
                        "processes": f"{ra['pid']} vs {rb['pid']}",
                        "n_kept": int(ra["n_kept"]),
                        "ok": bool(ok),
                    },
                    fatal=True,
                )
            if "prefix" in gates:
                k = PREFIX_TRIALS
                ok = all(
                    np.array_equal(
                        rc[f], ra[f][:k] if np.ndim(ra[f]) else ra[f], equal_nan=True
                    )
                    for f in fields
                )
                ok = ok and int(ns["best_of"](rc["val_mse"][None, :], k)[0]) == int(
                    ns["best_of"](ra["val_mse"][None, :], k)[0]
                )
                emit(
                    key
                    | {
                        "gate": "prefix",
                        "row": SLICE_OOS_START,
                        "k": k,
                        "ok": bool(ok),
                    },
                    fatal=True,
                )
        need_u = gates & {"smoke", "chunk", "pool", "perturb"}
        if not need_u:
            continue
        stU, sU, ns = stage1(
            model, U, SLICE_OOS_START, cpus=a.pool, trials=a.smoke_trials
        )
        recU = write_records(ns, stU, scratch / f"smoke_{model}{sfx}" / "trials.npz")
        sideU, s2U, ns2 = stage2(model, U, SLICE_OOS_START, recU, cpus=a.pool)
        tl = ns["trials_long"](stU, list(U["dates"]))
        tl.to_csv(out / f"smoke_trials_{model}{sfx}.csv", index=False)
        if "smoke" in gates:
            sec = stU["sec"]
            fs = np.concatenate([d["fit_sec"] for d in sideU["paths"].values()])
            row = key | {
                "gate": "smoke",
                "points": len(stU["row"]),
                "trials": sec.shape[1],
                "sec_stage1_wall": sU,
                "sec_per_trial_mean": float(sec.mean()),
                "sec_per_trial_median": float(np.median(sec)),
                "sec_per_trial_max": float(sec.max()),
                "sec_per_point_mean": float(stU["study_sec"].mean()),
                "tpe_overhead_sec_per_point": float(
                    (stU["study_sec"] - sec.sum(axis=1)).mean()
                ),
                "rounds_mean": float(stU["rounds"].mean()),
                "rounds_max_seen": int(stU["rounds"].max()),
                "cap_hits": int(
                    (
                        (stU["rounds_max"] > 0) & (stU["rounds"] >= stU["rounds_max"])
                    ).sum()
                ),
                "best_mse_k50": json.dumps(stU["best_mse_k50"].tolist()),
                "best_mse_k25": json.dumps(stU["best_mse_k25"].tolist()),
                "best_mse_k10": json.dumps(stU["best_mse_k10"].tolist()),
                "best_qlike_k50": json.dumps(stU["best_qlike_k50"].tolist()),
                "val_mse_shipped": json.dumps(stU["val_mse"][:, 0].round(6).tolist()),
                "val_mse_best": json.dumps(
                    np.take_along_axis(stU["val_mse"], stU["best_mse_k50"][:, None], 1)[
                        :, 0
                    ]
                    .round(6)
                    .tolist()
                ),
                "stage2_distinct_fits": sideU["n_fits"],
                "stage2_rows_x_paths": len(U["y"]) - W,
                "sec_stage2_wall": s2U,
                "sec_per_refit_mean": float(np.mean(fs)),
                "shap_additivity_gap": sideU["shap_additivity_gap"],
            }
            emit(row, fatal=False)
        gt = a.gate_trials
        if gt != a.smoke_trials and gates & {"chunk", "pool", "perturb"}:
            stU, _, ns = stage1(model, U, SLICE_OOS_START, cpus=a.pool, trials=gt)
            recU = write_records(ns, stU, scratch / f"smoke{gt}_{model}" / "trials.npz")
            sideU, _, _ = stage2(model, U, SLICE_OOS_START, recU, cpus=a.pool)
        if "chunk" in gates:
            assert C1 is not None and C2 is not None
            sa, _, nsa = stage1(model, C1, SLICE_OOS_START, cpus=a.pool, trials=gt)
            sb, sec_b1, _ = stage1(
                model, C2, SLICE_OOS_START + CHUNK_SPLIT, cpus=a.pool, trials=gt
            )
            merged = {
                k: np.concatenate([np.asarray(sa[k]), np.asarray(sb[k])]) for k in sa
            }
            rec_ok = same_records(merged, stU)
            recC = write_records(nsa, merged, scratch / f"chunk_{model}" / "trials.npz")
            pa, _, _ = stage2(model, C1, SLICE_OOS_START, recC, cpus=a.pool)
            pb, sec_b2, _ = stage2(
                model, C2, SLICE_OOS_START + CHUNK_SPLIT, recC, cpus=a.pool
            )
            per_path = same_paths(
                concat_paths(pa["paths"], pb["paths"]), sideU["paths"]
            )
            row = key | {
                "gate": "chunk",
                "trials": gt,
                "tuning_records_bit_identical": bool(rec_ok),
                "paths_bit_identical": json.dumps(per_path),
            }
            row["ok"] = bool(rec_ok and all(per_path.values()))
            emit(row, fatal=True)
            if "pool" in gates:
                sc, sec_c1, _ = stage1(
                    model, C2, SLICE_OOS_START + CHUNK_SPLIT, cpus=1, trials=gt
                )
                pc, sec_c2, _ = stage2(
                    model, C2, SLICE_OOS_START + CHUNK_SPLIT, recC, cpus=1
                )
                per_path = same_paths(pc["paths"], pb["paths"])
                row = key | {
                    "gate": "pool",
                    "trials": gt,
                    "tuning_records_bit_identical": bool(same_records(sc, sb)),
                    "paths_bit_identical": json.dumps(per_path),
                    "sec_stage1_one_process": sec_c1,
                    "sec_stage1_pooled": sec_b1,
                    "sec_stage2_one_process": sec_c2,
                    "sec_stage2_pooled": sec_b2,
                }
                row["ok"] = bool(
                    row["tuning_records_bit_identical"] and all(per_path.values())
                )
                emit(row, fatal=True)
        if "perturb" in gates:
            y2 = U["y"].copy()
            t_p = (
                W + PERTURB_ROW
            )  # design row of the forecast row SLICE_OOS_START + PERTURB_ROW
            y2[t_p:] *= PERTURB_FACTOR
            sp, _, nsp = stage1(model, U, SLICE_OOS_START, cpus=a.pool, trials=gt, y=y2)
            recP = write_records(nsp, sp, scratch / f"perturb_{model}" / "trials.npz")
            pp, _, _ = stage2(model, U, SLICE_OOS_START, recP, cpus=a.pool, y=y2)
            keep = slice(0, PERTURB_ROW + 1)  # points / rows <= t_p
            rec_ok = same_records(sp, stU, keep)
            later_rec_moved = int(
                sum(
                    not np.array_equal(sp["val_mse"][j], stU["val_mse"][j])
                    for j in range(PERTURB_ROW + 1, len(sp["row"]))
                )
            )
            per_path = same_paths(pp["paths"], sideU["paths"], keep)
            moved = {
                path: int(
                    np.sum(
                        pp["paths"][path]["preds"][PERTURB_ROW + 1 :]
                        != sideU["paths"][path]["preds"][PERTURB_ROW + 1 :]
                    )
                )
                for path in pp["paths"]
            }
            row = key | {
                "gate": "perturb",
                "trials": gt,
                "t_p_oos_row": SLICE_OOS_START + PERTURB_ROW,
                "t_p_date": U["dates"][PERTURB_ROW],
                "factor": PERTURB_FACTOR,
                "tunings_checked": PERTURB_ROW + 1,
                "tunings_bit_identical": bool(rec_ok),
                "later_tunings_moved": later_rec_moved,
                "rows_checked": PERTURB_ROW + 1,
                "paths_bit_identical_before": json.dumps(per_path),
                "rows_after_moved": json.dumps(moved),
            }
            row["ok"] = bool(
                rec_ok
                and all(per_path.values())
                and all(v > 0 for v in moved.values())
                and later_rec_moved > 0
            )
            emit(row, fatal=True)
        if "mask_equiv" in gates and MASK:
            from src.models.window_mask import window_keep

            common: set | None = None
            for i in range(len(U["X"]) - W):  # every window of U: [i, i + W)
                kept = set(window_keep(U["X"][i : i + W]).tolist())
                common = kept if common is None else common & kept
            assert common is not None
            cols = np.array(sorted(common), dtype=np.int64)
            D = dict(U, X=np.ascontiguousarray(U["X"][:, cols]), names=U["names"][cols])
            got = {}
            for mask_on in (True, False):
                MASK = mask_on
                sd, _, nsd = stage1(model, D, SLICE_OOS_START, cpus=a.pool, trials=gt)
                recD = write_records(
                    nsd, sd, scratch / f"maskeq_{model}_{int(mask_on)}" / "trials.npz"
                )
                pd_, _, _ = stage2(model, D, SLICE_OOS_START, recD, cpus=a.pool)
                got[mask_on] = (sd, pd_)
            MASK = True
            (s_on, p_on), (s_off, p_off) = got[True], got[False]
            rec_ok = same_records(s_on, s_off)
            per_path = same_paths(p_on["paths"], p_off["paths"])
            n_on = {int(v) for v in s_on["n_kept"]} | {
                int(v) for d in p_on["paths"].values() for v in d["n_kept"]
            }
            row = key | {
                "gate": "mask_equiv",
                "trials": gt,
                "design_columns": len(cols),
                "of_columns": U["X"].shape[1],
                "kept_counts_mask_on": json.dumps(sorted(n_on)),
                "tuning_records_bit_identical": bool(rec_ok),
                "paths_bit_identical": json.dumps(per_path),
            }
            row["ok"] = bool(rec_ok and all(per_path.values()) and n_on == {len(cols)})
            emit(row, fatal=True)
    print("gates done")


if __name__ == "__main__":
    main()
