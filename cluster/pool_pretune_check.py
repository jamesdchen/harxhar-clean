"""Gate of the pooled pre-tune study on Hoffman2 (2026-10-06; runner experiments/pool_linear_pretune.py).

gate   The canary re-ran, with the runner's penalty-factor subclass at r = 1 and the spec's causal
       re-choice (= arm a, the spec), one slice of nine STORED pooled arms -- all_features, baseline and
       lean_vol_vix x ridge / recursive lasso / recursive elastic net -- each on the CPU architecture its
       stored arm ran on: START = W + 199,250 (W = 96,000 bars; 199,250 = 797 x 250, a re-choice boundary
       of the whole-series run), END = -1, HALO = W, i.e. the stored arm's last 5,067 forecasts.  The
       stored arms: all_features / baseline in the 2026-09-20/21 root (the recursive lasso from the
       lassofix re-run), lean_vol_vix from today's fleet in the pool-momvix root.
       PASS = same stamps and pred_adj bit for bit on every row (all_features and baseline required;
       lean_vol_vix required when its stored arm exists, else recorded as pending); ridge where the stored
       arm ran on another CPU type (all_features: intel-E5-2670, n7134) within 1e-6 relative instead.  Also required:
       the smoke tasks (r = 0.01 causal walks, one pre-tune) finished with finite output and no
       uncertified penalty-factor batch solution.
       Writes results/pool_pretune/gate.csv and, on a pass, results/pool_pretune/GATE_OK; CACHE_OK when also the
       gatec slices (arm a through the all_features design cache, cluster/pool_pretune_tasks_cgate_*.txt) pass.
status One line per task directory under results/pool_pretune/<tag>/ (DONE, seconds, rows, host).

Run (cluster, deployment root):  python cluster/pool_pretune_check.py gate | status
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OLD = Path("/u/scratch/j/jamesdc1/harxhar-subsection/results")
MOMVIX = Path("/u/scratch/j/jamesdc1/harxhar-pool-momvix/results/pool_momvix/fleet")
OUT = Path("results/pool_pretune")
TW = 2000
SLICE_START = 295_250
N_OOS = 204_317
SLICE_ROWS = N_OOS - (SLICE_START - TW * 48)
RT = "round_trip"  # exact text -> double parse, so bit comparisons see every ulp
RIDGE_TOL = 1e-6  # ridge only, where the stored arm ran on another CPU type (all_features: intel-E5-2670)
# the canary, and the design-cache gate slices (tag gatec: the same nine-arm comparison through the cache)
TASKFILES = sorted(
    Path("cluster").glob("pool_pretune_tasks_canary_intel*.txt")
) + sorted(Path("cluster").glob("pool_pretune_tasks_cgate_*.txt"))


def stored(bucket: str, est: str) -> Path:
    if bucket == "lean_vol_vix":
        return (
            MOMVIX
            / bucket
            / "none"
            / est
            / f"tw{TW}"
            / "causal_tune_linear"
            / est
            / bucket
            / "results.csv"
        )
    root = "linear_subsection_lassofix" if est == "reclasso" else "linear_subsection"
    return (
        OLD
        / root
        / bucket
        / "none"
        / est
        / f"tw{TW}"
        / "causal_tune_linear"
        / est
        / bucket
        / "results.csv"
    )


def task_dir(tag: str, bucket: str, est: str, r: str, start: str) -> Path:
    return OUT / tag / bucket / est / f"r{r}" / f"s{start}"


def gate() -> None:
    rows, ok_all = [], True
    ok_cache = any(tf.name.startswith("pool_pretune_tasks_cgate_") for tf in TASKFILES)
    for tf in TASKFILES:
        for ln in tf.read_text().splitlines():
            if not ln.strip():
                continue
            tag, mode, bucket, est, r, sched, start, end, halo, *design = ln.split()
            if mode == "design":
                d = Path(design[0])
                rows.append(
                    dict(
                        taskfile=tf.name,
                        tag=tag,
                        mode=mode,
                        bucket=bucket,
                        est=est,
                        r=r,
                        start=start,
                        done=(d / "DONE").is_file(),
                        verdict="ok" if (d / "DONE").is_file() else "NOT DONE",
                        required=True,
                    )
                )
                ok_all = ok_all and rows[-1]["done"]
                continue
            d = task_dir(tag, bucket, est, r, start)
            row = dict(
                taskfile=tf.name,
                tag=tag,
                mode=mode,
                bucket=bucket,
                est=est,
                r=r,
                start=start,
                done=(d / "DONE").is_file(),
                seconds=(d / "SECONDS").read_text().strip()
                if (d / "SECONDS").is_file()
                else "",
            )
            info = (
                json.loads((d / "run.json").read_text())
                if (d / "run.json").is_file()
                else {}
            )
            row["host"] = info.get("host", "")
            row["repair"] = json.dumps(info.get("repair", {}))
            if not row["done"]:
                row.update(verdict="NOT DONE", required=True)
            elif tag.startswith("gate"):
                res = d / "causal_tune_linear" / est / bucket / "results.csv"
                new = pd.read_csv(res, float_precision=RT)
                sp = stored(bucket, est)
                required = bucket != "lean_vol_vix" or sp.is_file()
                row["stored"] = str(sp)
                if not sp.is_file():
                    row.update(verdict="stored pending", required=False)
                else:
                    ref = pd.read_csv(sp, float_precision=RT)
                    tail = ref.iloc[len(ref) - len(new) :]
                    a, b = (
                        new["pred_adj"].to_numpy(float),
                        tail["pred_adj"].to_numpy(float),
                    )
                    same_stamps = bool(
                        len(new) == SLICE_ROWS
                        and len(ref) == N_OOS
                        and (new["date"].to_numpy() == tail["date"].to_numpy()).all()
                    )
                    m16 = new["date"].str.endswith("16:00:00").to_numpy()
                    bit = int((a == b).sum())
                    row.update(
                        rows=len(new),
                        rows_stored=len(ref),
                        same_stamps=same_stamps,
                        n_1600=int(m16.sum()),
                        pred_adj_bitwise_rows=bit,
                        max_abs_diff=float(np.abs(a - b).max()),
                        max_rel_diff=float(np.abs(a / b - 1).max()),
                        max_rel_diff_1600=float(np.abs(a[m16] / b[m16] - 1).max()),
                        true_adj_bitwise=bool(
                            np.array_equal(
                                new["true_adj"].to_numpy(float),
                                tail["true_adj"].to_numpy(float),
                            )
                        ),
                        required=required,
                    )
                    good = same_stamps and bit == len(new) and row["true_adj_bitwise"]
                    row["verdict"] = "bitwise" if good else "DIFFERS"
                    ridge_ok = (
                        est == "ridge" and same_stamps and row["true_adj_bitwise"]
                    )
                    if not good and ridge_ok and row["max_rel_diff"] <= RIDGE_TOL:
                        # the stored all_features ridge ran on intel-E5-2670 (n7134, no AVX2 / FMA): ridge has
                        # no branch to take, another CPU moves it by float rounding only
                        row["verdict"] = f"ridge within {RIDGE_TOL:g}"
            else:  # smoke walks and the canary pre-tune: finished, finite, every batch solution certified
                rep = info.get("repair", {})
                if mode == "walk":
                    res = d / "causal_tune_linear" / est / bucket / "results.csv"
                    p = pd.read_csv(res, float_precision=RT)["pred_adj"].to_numpy(float)
                    finite = bool(np.isfinite(p).all())
                    row["rows"] = len(p)
                else:
                    z = np.load(d / "fold.npz")
                    finite = bool(np.isfinite(z["resid"]).all())
                    row["rows"] = int(z["resid"].shape[-1])
                good = finite and rep.get("fail", 0) == 0
                row.update(
                    verdict="ok" if good else "FAILED", required=True, finite=finite
                )
            if row["required"] and not (
                row["verdict"] in ("bitwise", "ok")
                or row["verdict"].startswith("ridge within")
            ):
                if tf.name.startswith("pool_pretune_tasks_cgate_"):
                    ok_cache = False
                else:
                    ok_all = False
            rows.append(row)
            print(
                {
                    k: row.get(k)
                    for k in (
                        "tag",
                        "bucket",
                        "est",
                        "r",
                        "verdict",
                        "rows",
                        "pred_adj_bitwise_rows",
                        "max_rel_diff",
                        "seconds",
                        "host",
                        "repair",
                    )
                }
            )
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "gate.csv", index=False)
    # GATE_OK: the canary (gates the walks that redo the data prep); CACHE_OK: the canary and the gatec slices
    # through the all_features design cache (gates the walks that read it)
    flag, cflag = OUT / "GATE_OK", OUT / "CACHE_OK"
    if ok_all and ok_cache:
        cflag.write_text("cache gate passed\n")
        print("CACHE GATE PASSED ->", cflag)
    else:
        cflag.unlink(missing_ok=True)
    if ok_all:
        flag.write_text("gate passed\n")
        print("GATE PASSED ->", flag)
    else:
        flag.unlink(missing_ok=True)
        print("GATE FAILED (no flag)")
        sys.exit(1)


def status() -> None:
    for d in sorted(OUT.glob("*/*/*/r*/s*")):
        info = (
            json.loads((d / "run.json").read_text())
            if (d / "run.json").is_file()
            else {}
        )
        sec = (d / "SECONDS").read_text().strip() if (d / "SECONDS").is_file() else "-"
        print(
            f"{d}  {'DONE' if (d / 'DONE').is_file() else 'running/failed'}  {sec}s  rows {info.get('rows', '-')}"
            f"  tunes {info.get('n_tunes', '-')}  {info.get('host', '')}"
        )


if __name__ == "__main__":
    {"gate": gate, "status": status}[sys.argv[1]]()
