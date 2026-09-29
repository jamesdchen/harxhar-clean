"""Local gates for specs/causal_tune_trees_tuned.py before it is shipped.

Data: the spec's exact run_executor invocation on the live_feasible bar1600
series, sliced through run_executor's START / END / HALO seam (HALO = the
training window), the design matrix captured as the executor hands it to
fit_predict.  Three captures: the unchunked smoke slice U (whole-series OOS rows
SLICE_OOS_START .. + SLICE_ROWS: two tuning points and one refit block after the
second) and its two chunks C1 = the first TUNE_PER rows, C2 = the rest.

Per model:
  identity  TUNE_IDENTITY=1 (grid = the shipped config, rounds = the shipped
            n_estimators), pooled, against the UNTUNED spec's own fit_predict_tree
            (one thread, as the untuned GB fleet) on U: bit-identical required;
            also max |rel diff| against the CARC untuned arm on the same stamps
  smoke     the full grid on U, pooled: runtime per tuning point / candidate /
            refit, chosen configs, validation losses
  chunk     C1 and C2 run separately equal U bit for bit: the executor's arrays
            (X, y of each chunk = the matching rows of U's), the forecasts (both
            rules), TreeSHAP, and every tuning record (whole-series rows, split,
            every candidate's losses and rounds, picks)
  pool      C2 with one process (map in-process) equals C2 pooled bit for bit
  perturb   U with the target x PERTURB_FACTOR on every row from the second
            tuning point's forecast row t_p on: every tuning record at a forecast
            row <= t_p and every forecast of a model fitted before t_p's first
            block ends is bit-identical; later forecasts move

Usage: python experiments/gate_trees_tuned.py [--models lgbm,xgb,rf]
           [--gates identity,smoke,chunk,pool,perturb] [--pool 6]
           [--carc-root DIR] [--out DIR] [--scratch DIR]
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

TUNED_SPEC = ROOT / "specs" / "causal_tune_trees_tuned.py"
TREE_SPEC = ROOT / "specs" / "causal_tune_trees.py"
_TREE_RUN_CELL = '\nout_csv = os.path.join(OUT_DIR, "results.csv")\n'

BUCKET, SEGMENT, TW = "live_feasible", "bar1600", 2000
TUNE_PER, REFIT_EVERY = 250, 10  # asserted against the spec below
SLICE_OOS_START = 1000  # whole-series OOS row of U's first forecast (a tuning row)
SLICE_ROWS = (
    TUNE_PER + 2 * REFIT_EVERY
)  # tunings at U rows 0 and TUNE_PER + one block after
PERTURB_FACTOR = 50.0  # the target x 50 on every row from t_p on
W = TW  # one row per session at a one-bar segment


def set_env(
    model: str, identity: bool, shap: bool, start: int, end: int, cpus: int
) -> None:
    os.environ.update(
        HPC_KW_MODEL=model,
        HPC_KW_EXOG_BUCKET=BUCKET,
        HPC_KW_SEGMENT=SEGMENT,
        HPC_KW_TRAIN_WIN=str(TW),
        HPC_KW_LAG_SCOPE="global",
        HPC_KW_TUNE_IDENTITY="1" if identity else "0",
        HPC_KW_QSEL="1",
        HPC_KW_SHAP_SEGMENTS=SEGMENT if shap else "none",
        HPC_KW_START=str(start),
        HPC_KW_END=str(end),
        HPC_KW_HALO=str(W),
        SLURM_CPUS_PER_TASK=str(cpus),
    )


def exec_tuned() -> dict:
    ns: dict = {"__file__": str(TUNED_SPEC), "__name__": "tuned_setup"}
    exec(compile(TUNED_SPEC.read_text(encoding="utf-8"), str(TUNED_SPEC), "exec"), ns)
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
    set_env("lgbm", identity=True, shap=False, start=start, end=start + rows, cpus=1)
    ns = exec_tuned()  # the spec's own constants for the invocation below
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


def carc_preds(carc_root: Path, model: str, dates: np.ndarray) -> np.ndarray | None:
    p = (
        carc_root
        / BUCKET
        / SEGMENT
        / model
        / f"tw{TW}"
        / "causal_tune_trees"
        / model
        / BUCKET
        / f"results_{SEGMENT}.csv"
    )
    if not p.is_file():
        print(f"note: no CARC untuned arm at {p}")
        return None
    return pd.read_csv(p).set_index("date")["pred_adj"].reindex(dates).to_numpy(float)


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.nanmax(np.abs(a / b - 1.0)))


def run_spec(model, d, oos0, *, identity=False, shap=True, cpus=1, y=None):
    """Exec the spec for this slice and walk it; (preds, SIDE, seconds, namespace)."""
    start = W + oos0
    set_env(model, identity, shap, start, start + len(d["X"]) - W, cpus)
    ns = exec_tuned()
    assert ns["TUNE_PER"] == TUNE_PER and ns["REFIT_EVERY"] == REFIT_EVERY
    a = time.time()
    preds = ns["fit_predict_tuned"](
        d["X"],
        d["y"] if y is None else y,
        int(d["W"]),
        {"_feature_names": list(d["names"])},
    )
    return np.asarray(preds), dict(ns["SIDE"]), time.time() - a, ns


def same_record(r0: dict, r1: dict) -> bool:
    return bool(
        r0["pick_mse"] == r1["pick_mse"]
        and r0["pick_qlike"] == r1["pick_qlike"]
        and r0["split"] == r1["split"]
        and np.array_equal(r0["val_mse"], r1["val_mse"])
        and np.array_equal(r0["val_qlike"], r1["val_qlike"])
        and np.array_equal(r0["rounds"], r1["rounds"])
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lgbm,xgb,rf")
    ap.add_argument("--gates", default="identity,smoke,chunk,pool,perturb")
    ap.add_argument(
        "--pool", type=int, default=6, help="worker processes for the pooled runs"
    )
    ap.add_argument("--carc-root", default=os.environ.get("CARC_TREES_ROOT", ""))
    ap.add_argument("--out", default="results/linear_subsection_trees_tuned/gates")
    ap.add_argument(
        "--scratch",
        default=os.environ.get(
            "GATE_SCRATCH", "results/linear_subsection_trees_tuned/gates/scratch"
        ),
    )
    a = ap.parse_args()
    out, scratch = ROOT / a.out, Path(a.scratch)
    out.mkdir(parents=True, exist_ok=True)
    gates = {g.strip() for g in a.gates.split(",") if g.strip()}
    U = capture(scratch, SLICE_OOS_START, SLICE_ROWS)
    C1 = C2 = None
    if gates & {"chunk", "pool"}:
        C1 = capture(scratch, SLICE_OOS_START, TUNE_PER)
        C2 = capture(scratch, SLICE_OOS_START + TUNE_PER, SLICE_ROWS - TUNE_PER)
    rows: list[dict] = []

    def emit(row: dict, fatal: bool) -> None:
        print(json.dumps(row, default=str), flush=True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / "gate_rows.csv", index=False)
        if fatal and not row.get("ok", True):
            sys.exit(f"GATE {row['gate']} FAILED for {row['model']}: STOP")

    if C1 is not None:  # executor-level chunk invariance of the inputs (model-free)
        assert C2 is not None and U is not None, (
            "the chunked and unchunked captures run together"
        )
        ok = all(np.array_equal(C1[k], U[k][: len(C1[k])]) for k in ("X", "y")) and all(
            np.array_equal(C2[k], U[k][TUNE_PER:]) for k in ("X", "y")
        )
        ok = ok and np.array_equal(np.r_[C1["dates"], C2["dates"]], U["dates"])
        emit({"model": "-", "gate": "chunk_inputs", "ok": bool(ok)}, fatal=True)

    for model in [m.strip() for m in a.models.split(",") if m.strip()]:
        key = {"model": model, "bucket": BUCKET, "segment": SEGMENT, "pool": a.pool}
        if "identity" in gates:
            p_id, side_id, s_id, ns = run_spec(
                model, U, SLICE_OOS_START, identity=True, cpus=a.pool
            )
            assert len(ns["CANDIDATES"]) == 1 and not ns["EARLY_STOP"]
            os.environ["SLURM_CPUS_PER_TASK"] = (
                "1"  # the untuned GB fleet's thread count
            )
            un = exec_untuned()
            same_params = (
                ns["J"].build_model(ns["CANDIDATES"][0], None).get_params()
                == un["make_model"]().get_params()
            )
            a0 = time.time()
            p_un = np.asarray(
                un["fit_predict_tree"](
                    U["X"], U["y"], W, {"_feature_names": list(U["names"])}
                )
            )
            row = key | {
                "gate": "identity",
                "n": len(p_id),
                "constructor_params_equal": bool(same_params),
                "bit_identical_vs_local_untuned": bool(np.array_equal(p_id, p_un)),
                "max_rel_vs_local_untuned": rel(p_id, p_un),
                "shap_bit_identical": bool(
                    np.array_equal(side_id["shap"], un["SIDE"]["shap"], equal_nan=True)
                ),
                "importance_bit_identical": bool(
                    np.array_equal(side_id["importance"], un["SIDE"]["importance"])
                ),
                "sec_tuned_identity_pooled": s_id,
                "sec_untuned_1thread": time.time() - a0,
            }
            if a.carc_root:
                pc = carc_preds(Path(a.carc_root), model, U["dates"])
                if pc is not None:
                    row |= {
                        "n_carc": int(np.isfinite(pc).sum()),
                        "max_rel_vs_carc_untuned": rel(p_id, pc),
                    }
            row["ok"] = bool(
                row["bit_identical_vs_local_untuned"]
                and same_params
                and row["shap_bit_identical"]
            )
            emit(row, fatal=True)
        if not gates & {"smoke", "chunk", "perturb"}:
            continue
        p0, side0, s0, ns = run_spec(model, U, SLICE_OOS_START, cpus=a.pool)
        tr, cands = side0["trace"], ns["CANDIDATES"]
        trace_tab, cand_tab = ns["trace_tables"](pd.Series(U["dates"]))
        trace_tab.to_csv(out / f"smoke_trace_{model}.csv", index=False)
        cand_tab.to_csv(out / f"smoke_candidates_{model}.csv", index=False)
        if "smoke" in gates:
            row = key | {
                "gate": "smoke",
                "n": len(p0),
                "grid_size": ns["GRID_SIZE"],
                "candidates": len(cands),
                "tunings": len(tr),
                "sec_walk": s0,
                "sec_per_tuning": float(np.mean([r["tune_sec"] for r in tr])),
                "sec_per_candidate_fit": float(
                    np.mean([r["fit_sec"].mean() for r in tr])
                ),
                "sec_per_refit": float(np.mean(side0["fit_sec"])),
                "sec_per_shap_block": float(np.mean(side0["shap_sec"]))
                if len(side0["shap_sec"])
                else np.nan,
                "qsel_extra_refits": len(side0["qsel_sec"]),
                "sec_per_qsel_refit": float(np.mean(side0["qsel_sec"]))
                if len(side0["qsel_sec"])
                else 0.0,
                "mean_rounds": float(np.mean([r["rounds"].mean() for r in tr])),
                "max_rounds": int(max(r["rounds"].max() for r in tr)),
                "rounds_max": ns["ROUNDS_MAX"],
                "slice_mse_tuned": float(np.mean((U["y"][W:] - p0) ** 2)),
            }
            for j, r in enumerate(tr):
                gm, gq = r["pick_mse"], r["pick_qlike"]
                row |= {
                    f"t{j}_row": SLICE_OOS_START + r["i"],
                    f"t{j}_date": U["dates"][r["i"]],
                    f"t{j}_mse_pick": json.dumps(cands[gm]),
                    f"t{j}_mse_rounds": int(r["rounds"][gm]),
                    f"t{j}_val_mse_pick": float(r["val_mse"][gm]),
                    f"t{j}_val_mse_shipped": float(r["val_mse"][0]),
                    f"t{j}_val_mse_min_max": [
                        float(r["val_mse"].min()),
                        float(r["val_mse"].max()),
                    ],
                    f"t{j}_qlike_pick": json.dumps(cands[gq]),
                    f"t{j}_val_qlike_pick": float(r["val_qlike"][gq]),
                    f"t{j}_val_qlike_shipped": float(r["val_qlike"][0]),
                }
            emit(row, fatal=False)
        if "chunk" in gates:
            pa, sa, _, _ = run_spec(model, C1, SLICE_OOS_START, cpus=a.pool)
            pb, sb, sec_b, _ = run_spec(
                model, C2, SLICE_OOS_START + TUNE_PER, cpus=a.pool
            )
            trc = sa["trace"] + sb["trace"]
            rec_ok = (
                len(trc) == len(tr)
                and all(same_record(r0, r1) for r0, r1 in zip(tr, trc))
                and [sa["oos_offset"] + r["i"] for r in sa["trace"]]
                + [sb["oos_offset"] + r["i"] for r in sb["trace"]]
                == [side0["oos_offset"] + r["i"] for r in tr]
            )
            row = key | {
                "gate": "chunk",
                "preds_bit_identical": bool(np.array_equal(np.r_[pa, pb], p0)),
                "qsel_bit_identical": bool(
                    np.array_equal(np.r_[sa["pred_q"], sb["pred_q"]], side0["pred_q"])
                ),
                "shap_bit_identical": bool(
                    np.array_equal(
                        np.vstack([sa["shap"], sb["shap"]]),
                        side0["shap"],
                        equal_nan=True,
                    )
                ),
                "importance_bit_identical": bool(
                    np.array_equal(
                        np.vstack([sa["importance"], sb["importance"]]),
                        side0["importance"],
                    )
                ),
                "tuning_records_bit_identical": bool(rec_ok),
                "tunings": len(trc),
            }
            row["ok"] = bool(
                row["preds_bit_identical"]
                and row["qsel_bit_identical"]
                and row["shap_bit_identical"]
                and row["importance_bit_identical"]
                and rec_ok
            )
            emit(row, fatal=True)
            if "pool" in gates:
                pc, sc, sec_c, _ = run_spec(
                    model, C2, SLICE_OOS_START + TUNE_PER, cpus=1
                )
                row = key | {
                    "gate": "pool",
                    "preds_bit_identical": bool(np.array_equal(pc, pb)),
                    "qsel_bit_identical": bool(
                        np.array_equal(sc["pred_q"], sb["pred_q"])
                    ),
                    "tuning_records_bit_identical": bool(
                        all(
                            same_record(r0, r1)
                            for r0, r1 in zip(sc["trace"], sb["trace"])
                        )
                    ),
                    "sec_one_process": sec_c,
                    "sec_pooled": sec_b,
                }
                row["ok"] = bool(
                    row["preds_bit_identical"]
                    and row["qsel_bit_identical"]
                    and row["tuning_records_bit_identical"]
                )
                emit(row, fatal=True)
        if "perturb" in gates:
            y2 = U["y"].copy()
            t_p = W + TUNE_PER  # the forecast row of the second tuning point
            y2[t_p:] *= PERTURB_FACTOR
            p1, side1, s1, _ = run_spec(
                model, U, SLICE_OOS_START, shap=False, cpus=a.pool, y=y2
            )
            tr1 = side1["trace"]
            before = [j for j, r in enumerate(tr) if W + r["i"] <= t_p]
            cut = (
                TUNE_PER + REFIT_EVERY
            )  # rows forecast by models fitted on rows < t_p only
            moved = int(np.sum(p0[cut:] != p1[cut:]))
            row = key | {
                "gate": "perturb",
                "t_p_oos_row": SLICE_OOS_START + TUNE_PER,
                "t_p_date": U["dates"][TUNE_PER],
                "factor": PERTURB_FACTOR,
                "tunings_checked": len(before),
                "tunings_bit_identical": bool(
                    all(same_record(tr[j], tr1[j]) for j in before)
                ),
                "rows_checked": cut,
                "preds_bit_identical": bool(np.array_equal(p0[:cut], p1[:cut])),
                "qsel_preds_bit_identical": bool(
                    np.array_equal(side0["pred_q"][:cut], side1["pred_q"][:cut])
                ),
                "rows_after": len(p0) - cut,
                "rows_after_moved": moved,
                "sec_walk": s1,
            }
            row["ok"] = bool(
                row["tunings_bit_identical"]
                and row["preds_bit_identical"]
                and row["qsel_preds_bit_identical"]
                and moved > 0
            )
            emit(row, fatal=True)
    print("gates done")


if __name__ == "__main__":
    main()
