"""Local gates for specs/causal_tune_lstm.py before it is shipped.

  identity     per bucket: the design matrix, target, OOS stamps and column names
               the LSTM spec hands its model (its own run_executor invocation,
               whole series) equal -- sha256 of the bytes -- those of the per-bar
               LINEAR spec's own invocation (specs/causal_tune_linear.py executed
               with run_executor intercepted: its keyword arguments are captured,
               then replayed with a capturing fit_predict); the two invocations'
               keyword arguments are also compared (all but method_name /
               fit_predict / hyperparams / output_file must be equal)
  determinism  the walk on U (OOS rows 0 .. U_ROWS of live_feasible bar1600),
               pooled, run twice: forecasts (both rules, every seed) and every
               tuning record bit-identical
  pool         the same walk with one process (in-process map) == pooled, bit for bit
  chunk        end to end through the spec's run cell and the reducer: U as one
               chunk, and its two chunks C1 = [0, TUNE_PER), C2 = [TUNE_PER, U_ROWS)
               as separate runs merged by experiments/reduce_lstm_chunks.py; the
               merged pred_adj (both rules), stamps, target, tuning picks and every
               candidate's validation losses equal U's bit for bit
  smoke        the campaign settings (5 seeds, <= 200 epochs, patience 20) through
               the spec's run cell on SMOKE_ROWS OOS rows of live_feasible bar1600,
               pool of --pool: runtime per tuning point / fit / refit, picks

NO CHAIN LOAD: the loader reads every parquet of its directory (then skips the
ones without a bar key), so every data prep here reads BAR_DATA -- the bar-keyed
files of data/ (decided from the parquet schema alone), hard-linked (or copied)
into the scratch dir -- never data/spxw_chain.parquet.  The linear and LSTM
invocations are replayed on the same BAR_DATA; the target and stamps are also
checked against the linear campaign's own stored bar1600 results (ridge arm).

determinism / pool / chunk run with the reduced training budget FAST (fewer
seeds and epochs) -- they test the code path, not the fit; smoke runs the real one.

Usage: python experiments/gate_lstm.py [--gates identity,determinism,pool,chunk,smoke]
           [--buckets live_feasible,all_features,baseline] [--pool 4] [--scratch DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

LSTM_SPEC = ROOT / "specs" / "causal_tune_lstm.py"
LINEAR_SPEC = ROOT / "specs" / "causal_tune_linear.py"
SEGMENT, TW = "bar1600", 2000
W = TW  # one row per session at a one-bar segment
TUNE_PER = 250  # asserted against the spec
U_ROWS = (
    TUNE_PER + 30
)  # two tuning points (rows 0, TUNE_PER) and three refit blocks after the second
SMOKE_ROWS = 300  # the smoke slice: two tuning points, 30 refit blocks
FAST = {
    "N_SEEDS": "2",
    "MAX_EPOCHS": "12",
    "PATIENCE": "3",
}  # the code-path gates' budget
CAMPAIGN = {"N_SEEDS": "5", "MAX_EPOCHS": "200", "PATIENCE": "20"}
SKIP_KW = ("method_name", "fit_predict", "hyperparams", "output_file")


class _Captured(Exception):
    pass


LINEAR_STORED = (
    ROOT / "results" / "linear_subsection" / "arms_hoffman2"
)  # <bucket>/ridge/tw2000/results_<seg>.csv
BAR_DATA: Path | None = None


def bar_data(scratch: Path) -> Path:
    """A directory with the bar-keyed parquet files of data/ only (schema read, no data)."""
    import pyarrow.parquet as pq

    d = scratch / "data_barkeyed"
    d.mkdir(parents=True, exist_ok=True)
    kept, skipped = [], []
    for f in sorted((ROOT / "data").glob("*.parquet")):
        if "endbartime" not in pq.read_schema(f).names:
            skipped.append(f.name)
            continue
        dst = d / f.name
        if not dst.is_file() or dst.stat().st_size != f.stat().st_size:
            dst.unlink(missing_ok=True)
            try:
                os.link(f, dst)
            except OSError:
                import shutil

                shutil.copyfile(f, dst)
        kept.append(f.name)
    for f in d.glob("*.parquet"):
        if f.name not in kept:
            f.unlink()
    print(
        f"bar-keyed data dir {d}: {len(kept)} files; not bar-keyed (never read): {skipped}"
    )
    return d


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def set_env(
    bucket: str, start: int, end: int, halo: int, cpus: int, budget: dict
) -> None:
    for k in ("HAR_LAGS", "HAR_BASE", "DATA_PATH"):
        os.environ.pop(f"HPC_KW_{k}", None)
        os.environ.pop(k, None)
    os.environ.update(
        HPC_KW_EXOG_BUCKET=bucket,
        HPC_KW_SEGMENT=SEGMENT,
        HPC_KW_TRAIN_WIN=str(TW),
        HPC_KW_LAG_SCOPE="global",
        HPC_KW_QSEL="1",
        HPC_KW_START=str(start),
        HPC_KW_END=str(end),
        HPC_KW_HALO=str(halo),
        SLURM_CPUS_PER_TASK=str(cpus),
        **{f"HPC_KW_{k}": v for k, v in budget.items()},
    )


def exec_lstm() -> dict:
    ns: dict = {"__file__": str(LSTM_SPEC), "__name__": "lstm_setup"}
    exec(compile(LSTM_SPEC.read_text(encoding="utf-8"), str(LSTM_SPEC), "exec"), ns)
    return ns


def grab_into(got: dict):
    def grab(X_chunk, y_chunk, train_win_periods, hyperparams):
        got.update(
            X=np.array(X_chunk, dtype=np.float64),
            y=np.array(y_chunk, dtype=np.float64),
            W=int(train_win_periods),
            names=np.array([str(c) for c in hyperparams["_feature_names"]]),
        )
        return np.zeros(len(X_chunk) - int(train_win_periods))

    return grab


def replay(kwargs: dict, out_dir: Path) -> dict:
    """run_executor with these kwargs, the model replaced by a capture; + the OOS stamps."""
    from src.backtest.executor import run_executor

    got: dict = {}
    assert BAR_DATA is not None
    kw = dict(
        kwargs,
        fit_predict=grab_into(got),
        output_file=str(out_dir / "results.csv"),
        data_path=str(BAR_DATA),
    )
    run_executor(**kw)
    res = pd.read_csv(out_dir / f"results_{SEGMENT}.csv")
    got["dates"] = res["date"].astype(str).to_numpy().astype("U19")
    got["true_adj"] = res["true_adj"].to_numpy(float)
    return got


def capture_lstm(
    bucket: str, scratch: Path, start: int = 0, end: int = -1, halo: int = 0
) -> tuple[dict, dict]:
    cache = scratch / f"capture_lstm_{bucket}_{start}_{end}_{halo}.npz"
    set_env(bucket, start, end, halo, 1, FAST)
    ns = exec_lstm()
    kwargs = ns["executor_kwargs"](None, "")
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}, kwargs
    got = replay(kwargs, scratch / f"cap_lstm_{bucket}_{start}_{end}")
    np.savez(cache, **got)
    return got, kwargs


def capture_linear(bucket: str, scratch: Path) -> tuple[dict, dict]:
    """Execute the linear spec with run_executor intercepted at its first call."""
    import src.backtest.executor as ex
    import src.data.loading as ld

    os.environ.update(HPC_KW_ESTIMATOR="ridge")
    set_env(bucket, 0, -1, 0, 1, FAST)
    real, real_load = ex.run_executor, ld.load_raw_data
    seen: dict = {}

    def intercept(**kw):
        seen.update(kw)
        raise _Captured

    def load_bar_data(
        path, *args, **kw
    ):  # the spec's evidence cell reads "data": redirect
        return real_load(str(BAR_DATA), *args, **kw)

    ex.run_executor = intercept  # type: ignore[assignment]
    ld.load_raw_data = load_bar_data  # type: ignore[assignment]
    try:
        ns: dict = {"__file__": str(LINEAR_SPEC), "__name__": "linear_setup"}
        exec(
            compile(LINEAR_SPEC.read_text(encoding="utf-8"), str(LINEAR_SPEC), "exec"),
            ns,
        )
        raise AssertionError("the linear spec never called run_executor")
    except _Captured:
        pass
    finally:
        ex.run_executor = real  # type: ignore[assignment]
        ld.load_raw_data = real_load  # type: ignore[assignment]
        os.environ.pop("HPC_KW_ESTIMATOR", None)
    cache = scratch / f"capture_linear_{bucket}.npz"
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}, seen
    got = replay(seen, scratch / f"cap_linear_{bucket}")
    np.savez(cache, **got)
    return got, seen


def walk(d: dict, cpus: int, budget: dict, oos0: int = 0) -> tuple[dict, dict]:
    """Exec the spec for the slice [oos0, oos0 + len(X) - W) and walk it in-process."""
    start = W + oos0
    set_env("live_feasible", start, start + len(d["X"]) - W, W, cpus, budget)
    ns = exec_lstm()
    assert ns["TUNE_PER"] == TUNE_PER, ns["TUNE_PER"]
    a = time.time()
    preds = ns["fit_predict_lstm"](
        d["X"], d["y"], int(d["W"]), {"_feature_names": list(d["names"])}
    )
    side = dict(ns["SIDE"])
    side["preds"] = np.asarray(preds)
    side["sec"] = time.time() - a
    return side, ns


def same_walk(a: dict, b: dict) -> tuple[bool, str]:
    why = []
    for k in ("preds", "pred_q", "pred_seeds", "refit_row"):
        if not np.array_equal(a[k], b[k]):
            why.append(
                f"{k} max|diff| {np.max(np.abs(np.asarray(a[k], float) - np.asarray(b[k], float))):.2e}"
            )
    for ra, rb in zip(a["trace"], b["trace"]):
        for k in ("pick_mse", "pick_qlike", "split"):
            if ra[k] != rb[k]:
                why.append(f"trace {k} at row {ra['i']}")
        for k in ("val_mse", "val_qlike", "epochs", "seed_val_mse"):
            if not np.array_equal(ra[k], rb[k]):
                why.append(f"trace {k} at row {ra['i']}")
    if len(a["trace"]) != len(b["trace"]):
        why.append("number of tuning points")
    return not why, "; ".join(why)


def run_spec_main(out: Path, start: int, end: int, cpus: int, budget: dict) -> float:
    """The spec's run cell in a subprocess (as the cluster runs it).  An output dir
    that already carries DONE (an earlier invocation of this gate) is reused: delete
    the scratch dir to force a re-run."""
    if (out / "DONE").is_file():
        print(f"reusing {out} (DONE)")
        return float("nan")
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("HPC_KW_"):
            env.pop(k)
    env.update(
        HPC_KW_EXOG_BUCKET="live_feasible",
        HPC_KW_SEGMENT=SEGMENT,
        HPC_KW_TRAIN_WIN=str(TW),
        HPC_KW_LAG_SCOPE="global",
        HPC_KW_START=str(start),
        HPC_KW_END=str(end),
        HPC_KW_HALO=str(W),
        HPC_KW_QSEL="1",
        HPC_RESULT_DIR=str(out),
        HPC_KW_DATA_PATH=str(BAR_DATA),
        SLURM_CPUS_PER_TASK=str(cpus),
        PYTHONUNBUFFERED="1",
        TQDM_DISABLE="1",
        **{f"HPC_KW_{k}": v for k, v in budget.items()},
    )
    out.mkdir(parents=True, exist_ok=True)
    a = time.time()
    with open(out / "run.log", "w", encoding="utf-8") as log:
        rc = subprocess.run(
            [sys.executable, "-u", str(LSTM_SPEC)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode
    assert rc == 0, (
        f"spec run failed ({out / 'run.log'}): {(out / 'run.log').read_text(encoding='utf-8')[-800:]}"
    )
    (out / "DONE").touch()
    return time.time() - a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", default="identity,determinism,pool,chunk,smoke")
    ap.add_argument("--buckets", default="live_feasible,all_features,baseline")
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--scratch", default="results/linear_subsection_lstm/gates/scratch")
    ap.add_argument("--out", default="results/linear_subsection_lstm/gates")
    a = ap.parse_args()
    gates = {g.strip() for g in a.gates.split(",") if g.strip()}
    scratch, out = ROOT / a.scratch, ROOT / a.out
    scratch.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    def emit(row: dict) -> None:
        rows.append(row)
        print(
            ("PASS " if row["ok"] else "FAIL ")
            + json.dumps({k: v for k, v in row.items() if k != "ok"}, default=str),
            flush=True,
        )
        pd.DataFrame(rows).to_csv(out / "gate_lstm.csv", index=False)

    out.mkdir(parents=True, exist_ok=True)
    global BAR_DATA
    BAR_DATA = bar_data(scratch)
    if "identity" in gates:
        for bucket in [b.strip() for b in a.buckets.split(",") if b.strip()]:
            t = time.time()
            lin, lin_kw = capture_linear(bucket, scratch)
            mine, my_kw = capture_lstm(bucket, scratch)
            kw_diff = sorted(
                k
                for k in set(lin_kw) | set(my_kw)
                if k not in SKIP_KW and lin_kw.get(k) != my_kw.get(k)
            )
            hashes = {
                k: (sha(lin[k]), sha(mine[k]))
                for k in ("X", "y", "names", "dates", "true_adj")
            }
            stored = pd.read_csv(
                LINEAR_STORED / bucket / "ridge" / f"tw{TW}" / f"results_{SEGMENT}.csv"
            )
            st_dates = stored["date"].astype(str).to_numpy()
            same_dates = len(st_dates) == len(mine["dates"]) and bool(
                (st_dates == mine["dates"]).all()
            )
            st_gap = (
                float(
                    np.max(
                        np.abs(
                            stored["true_adj"].to_numpy(float) / mine["true_adj"] - 1.0
                        )
                    )
                )
                if same_dates
                else float("nan")
            )
            ok = (
                not kw_diff
                and all(x == y for x, y in hashes.values())
                and int(lin["W"]) == int(mine["W"])
                and same_dates
                and st_gap < 1e-12
            )
            emit(
                {
                    "gate": "identity",
                    "bucket": bucket,
                    "ok": ok,
                    "kwargs_differ": kw_diff,
                    "X_shape": list(mine["X"].shape),
                    "n_oos": len(mine["dates"]),
                    "stored_linear_same_stamps": same_dates,
                    "stored_linear_true_adj_max_rel": st_gap,
                    "first": str(mine["dates"][0]),
                    "last": str(mine["dates"][-1]),
                    **{
                        f"sha_{k}": v[1]
                        + ("" if v[0] == v[1] else f" != linear {v[0]}")
                        for k, v in hashes.items()
                    },
                    "sec": round(time.time() - t),
                }
            )
    if gates & {"determinism", "pool"}:
        whole, _ = capture_lstm("live_feasible", scratch)
        U = {
            "X": whole["X"][: W + U_ROWS],
            "y": whole["y"][: W + U_ROWS],
            "W": W,
            "names": whole["names"],
        }
        A, _ = walk(U, a.pool, FAST)
        if "determinism" in gates:
            B, _ = walk(U, a.pool, FAST)
            ok, why = same_walk(A, B)
            emit(
                {
                    "gate": "determinism",
                    "bucket": "live_feasible",
                    "ok": ok,
                    "why": why,
                    "rows": U_ROWS,
                    "sec": [round(A["sec"]), round(B["sec"])],
                    "budget": FAST,
                }
            )
        if "pool" in gates:
            S, _ = walk(U, 1, FAST)
            ok, why = same_walk(A, S)
            emit(
                {
                    "gate": "pool",
                    "bucket": "live_feasible",
                    "ok": ok,
                    "why": why,
                    "rows": U_ROWS,
                    "sec_pooled": round(A["sec"]),
                    "sec_serial": round(S["sec"]),
                    "pool": a.pool,
                }
            )
    if "chunk" in gates:
        croot = scratch / "chunk_e2e"
        arm = croot / "live_feasible" / SEGMENT / "lstm" / f"tw{TW}"
        tasks = croot / "tasks.txt"
        t = time.time()
        u_dir = scratch / "chunk_e2e_U"
        run_spec_main(u_dir, W, W + U_ROWS, a.pool, FAST)
        lines = []
        for k, (s, e) in enumerate(((W, W + TUNE_PER), (W + TUNE_PER, W + U_ROWS))):
            run_spec_main(arm / "chunks" / f"c{k}", s, e, a.pool, FAST)
            lines.append(f"live_feasible lstm {TW} {SEGMENT} {k} {s} {e} {W}")
        tasks.write_text("\n".join(lines) + "\n", encoding="utf-8")
        rc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "reduce_lstm_chunks.py"),
                "--root",
                str(croot),
                "--tasks",
                str(tasks),
            ]
        ).returncode
        inner = Path("causal_tune_lstm") / "lstm" / "live_feasible"
        bad: list[str] = [] if rc == 0 else [f"reducer exit {rc}"]
        if rc == 0:
            rt = "round_trip"  # parse the CSVs' floats exactly
            for f, cols in (
                (
                    f"results_{SEGMENT}.csv",
                    ["date", "true_adj", "pred_adj", "true_raw"],
                ),
                (f"results_qsel_{SEGMENT}.csv", ["date", "pred_adj"]),
            ):
                mu = pd.read_csv(u_dir / inner / f, float_precision=rt)
                mc = pd.read_csv(arm / inner / f, float_precision=rt)
                cat = pd.concat(
                    [
                        pd.read_csv(
                            arm / "chunks" / f"c{k}" / inner / f, float_precision=rt
                        )
                        for k in (0, 1)
                    ],
                    ignore_index=True,
                )
                for c in cols:
                    if not mu[c].equals(cat[c]):
                        bad.append(f"chunks {f}:{c}")
                    if not mu[c].equals(mc[c]):
                        bad.append(f"merged {f}:{c}")
                gap = float(np.max(np.abs(mu["pred_raw"] / mc["pred_raw"] - 1.0)))
                if gap > 1e-12:
                    bad.append(f"{f}:pred_raw rel gap {gap:.1e}")
            tu = pd.read_csv(
                u_dir / inner / f"tune_trace_{SEGMENT}.csv", float_precision=rt
            )
            tc = pd.read_csv(
                arm / inner / f"tune_trace_{SEGMENT}.csv", float_precision=rt
            )
            for c in (
                "tune_row",
                "rule",
                "cand_idx",
                "epochs",
                "val_mse",
                "val_qlike",
                "fit_lo",
                "val_hi",
            ):
                if not tu[c].equals(tc[c]):
                    bad.append(f"tune_trace:{c}")
            cu = pd.read_csv(
                u_dir / inner / f"tune_candidates_{SEGMENT}.csv", float_precision=rt
            )
            cc = pd.read_csv(
                arm / inner / f"tune_candidates_{SEGMENT}.csv", float_precision=rt
            )
            for c in ("tune_row", "cand_idx", "epochs", "val_mse", "val_qlike"):
                if not cu[c].equals(cc[c]):
                    bad.append(f"tune_candidates:{c}")
            with (
                np.load(u_dir / inner / f"lstm_{SEGMENT}.npz") as zu,
                np.load(arm / inner / f"lstm_{SEGMENT}.npz") as zc,
            ):
                for key in (
                    "pred_adj",
                    "pred_adj_qsel",
                    "pred_adj_seeds",
                    "true_adj",
                    "refit_row",
                    "tune_row",
                    "cand_epochs",
                    "cand_val_mse",
                    "cand_val_qlike",
                ):
                    if not np.array_equal(zu[key], zc[key]):
                        bad.append(f"npz:{key}")
        emit(
            {
                "gate": "chunk",
                "bucket": "live_feasible",
                "ok": not bad,
                "why": "; ".join(bad),
                "rows": U_ROWS,
                "chunks": 2,
                "sec": round(time.time() - t),
                "budget": FAST,
            }
        )
    if "smoke" in gates:
        sdir = scratch / "smoke"
        sec = run_spec_main(sdir, W, W + SMOKE_ROWS, a.pool, CAMPAIGN)
        inner = sdir / "causal_tune_lstm" / "lstm" / "live_feasible"
        tr = pd.read_csv(inner / f"tune_trace_{SEGMENT}.csv")
        cd = pd.read_csv(inner / f"tune_candidates_{SEGMENT}.csv")
        with np.load(inner / f"lstm_{SEGMENT}.npz") as z:
            meta = json.loads(str(z["meta"]))
            fit_sec, qsel_sec = z["fit_sec"], z["qsel_sec"]
            spread = float(np.mean(np.std(z["pred_adj_seeds"], axis=1)))
            pa = z["pred_adj"]
        res = pd.read_csv(inner / f"results_{SEGMENT}.csv")
        emit(
            {
                "gate": "smoke",
                "bucket": "live_feasible",
                "ok": bool(np.isfinite(pa).all()),
                "rows": len(res),
                "run_sec": round(sec),
                "walk_sec": round(meta["walk_sec"]),
                "pool": a.pool,
                "tune_sec": tr[tr["rule"] == "mse"]["tune_sec"].round().tolist(),
                "fit_process_sec_mean": round(float(cd["fit_sec"].mean()), 1),
                "fit_process_sec_max": round(float(cd["fit_sec"].max()), 1),
                "refit_process_sec_mean_all_seeds": round(float(fit_sec.mean()), 1),
                "qsel_refits": int(len(qsel_sec)),
                "picks_mse": tr[tr["rule"] == "mse"][
                    ["seq_len", "hidden", "dropout", "lr", "epochs"]
                ].to_dict("records"),
                "picks_qlike": tr[tr["rule"] == "qlike"][
                    ["seq_len", "hidden", "dropout", "lr"]
                ].to_dict("records"),
                "epoch_cap_hits": int(cd["epochs_cap_hits"].sum()),
                "seed_sd_of_forecast_mean": round(spread, 5),
                "fit_space_mse": round(
                    float(((res["true_adj"] - res["pred_adj"]) ** 2).mean()), 5
                ),
            }
        )
    tab = pd.DataFrame(rows)
    print(tab[["gate", "bucket", "ok"]].to_string(index=False))
    if len(tab) and not tab["ok"].all():
        sys.exit("a gate failed")


if __name__ == "__main__":
    main()
