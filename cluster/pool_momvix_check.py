"""Gate and collect steps of the pooled moments_vix campaign (Hoffman2, 2026-10-06).

The pooled twins (one coefficient vector over all 48 half-hour bars, segment ``none``, scored at the
16:00 bar) on the moments_vix bucket are run by ``cluster/pool_momvix_task.sh`` exactly as the stored
pooled arms were (``cluster/subsection_pack.sh``, 2026-09-20/21).  This script, run on the cluster
from the deployment root, has two steps.

gate     The canary re-ran one slice of six STORED pooled arms (buckets moments and baseline x ridge /
         recursive lasso / recursive elastic net, 2000-session window) with today's code and data,
         through run_executor's slice seam: START = W + 199,250 (W = 2000 x 48 = 96,000 bars; 199,250
         = 797 x 250, a tune boundary of the whole-series run, where the warm state is cold-reseeded,
         so the slice's first solve is the whole-series run's 797th tuning), END = -1, HALO = W.  Each
         slice's 5,067 forecasts are compared with the stored whole-series arm's last 5,067 rows (the
         stored results.csv in the old deployment root; ridge / elastic net from the first runs, the
         recursive lasso from the lassofix re-run, as the stored yhat_pool_* tables were built).
         PASS = same stamps and pred_adj bit for bit on every row.  If not bit for bit, the
         tolerance of the original lassofix canary is reported (max |pred_adj difference| < 1e-6,
         cluster/lassofix_canary.sh) and the flag is written only under it.
         Also checked: the regular-hours filter (bars labelled 10:00 .. 16:00) applied to the stored
         whole-series results reproduces the local results_none_rth.csv files the stored yhat_pool_*
         tables were built from (md5, cluster/pool_momvix_rth_ref_md5.txt).
         Writes results/pool_momvix/gate.csv, rth_filter_check.csv, and GATE_OK on a pass.
collect  For each fleet arm (moments_vix x three estimators, whole series): DONE, 204,317 rows on the
         stamps of the stored pooled arms, finite forecasts; writes results_none_rth.csv beside
         results.csv with the filter the gate verified, and results/pool_momvix/collect.csv,
         MD5SUMS, COLLECTED.  `collect <bucket>` (2026-10-06: moments_vol_vix) does the same for another
         bucket's fleet and suffixes those three file names with _<bucket>.

Run (cluster):  python cluster/pool_momvix_check.py gate | collect
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OLD = Path("/u/scratch/j/jamesdc1/harxhar-subsection/results")
OUT = Path("results/pool_momvix")
TW = 2000
N_OOS = 204_317  # rows of every stored whole-series pooled arm (tw2000)
SLICE_ROWS = N_OOS - 199_250  # the gate slice: solves 199,250 .. 204,316
TOL_ABS = 1e-6  # cluster/lassofix_canary.sh's float-path tolerance on pred_adj
RT = "round_trip"  # exact text -> double parse, so bit comparisons see every ulp
RTH = {f"{h:02d}:{m:02d}" for h in range(10, 17) for m in (0, 30)} - {"16:30"}
ESTS = ("ridge", "reclasso", "reclasticnet")
COLS = ("pred_adj", "pred_raw", "true_adj", "true_raw")


def stored(bucket: str, est: str) -> Path:
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


def ours(tag: str, bucket: str, est: str) -> Path:
    return (
        OUT
        / tag
        / bucket
        / "none"
        / est
        / f"tw{TW}"
        / "causal_tune_linear"
        / est
        / bucket
        / "results.csv"
    )


def rth_text(path: Path) -> bytes:
    """Header + the lines whose bar-end label is 10:00 .. 16:00, byte for byte."""
    lines = path.read_bytes().split(b"\n")
    keep = [lines[0]] + [ln for ln in lines[1:] if ln and ln[11:16].decode() in RTH]
    return b"\n".join(keep) + b"\n"


def rth_pandas(path: Path) -> bytes:
    r = pd.read_csv(path, float_precision="round_trip")
    hm = pd.to_datetime(r["date"]).dt.strftime("%H:%M")
    return r[hm.isin(RTH)].to_csv(index=False, lineterminator="\n").encode()


def md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def compare(new: pd.DataFrame, ref: pd.DataFrame) -> dict:
    tail = ref.iloc[len(ref) - len(new) :]
    out: dict = {
        "rows": len(new),
        "rows_stored": len(ref),
        "same_stamps": bool(
            len(new) <= len(ref)
            and (new["date"].to_numpy() == tail["date"].to_numpy()).all()
        ),
        "n_1600": int(new["date"].str.endswith("16:00:00").sum()),
    }
    for c in COLS:
        a, b = new[c].to_numpy(float), tail[c].to_numpy(float)
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        d = np.abs(a - b)
        out[f"{c}_bitwise_rows"] = int(same.sum())
        out[f"{c}_max_abs_diff"] = float(np.nanmax(d)) if len(d) else np.nan
        out[f"{c}_max_rel_diff"] = float(np.nanmax(d / np.abs(b))) if len(d) else np.nan
    return out


def gate() -> int:
    rows = []
    ok_all, tol_all = True, True
    for bucket in ("moments", "baseline"):
        for est in ESTS:
            p, s = ours("gate", bucket, est), stored(bucket, est)
            r = {"bucket": bucket, "estimator": est, "slice": str(p), "stored": str(s)}
            if not p.is_file() or not s.is_file():
                r.update(verdict="FAIL: missing file")
                ok_all = tol_all = False
                rows.append(r)
                continue
            r.update(
                compare(
                    pd.read_csv(p, float_precision=RT),
                    pd.read_csv(s, float_precision=RT),
                )
            )
            bit = (
                r["same_stamps"]
                and r["rows"] == SLICE_ROWS
                and r["pred_adj_bitwise_rows"] == r["rows"]
            )
            tol = (
                r["same_stamps"]
                and r["rows"] == SLICE_ROWS
                and r["pred_adj_max_abs_diff"] < TOL_ABS
            )
            r["verdict"] = (
                "bitwise" if bit else (f"within {TOL_ABS:g}" if tol else "FAIL")
            )
            ok_all &= bit
            tol_all &= tol
            rows.append(r)
            print(
                f"{bucket:8s} {est:12s} rows {r['rows']} (16:00 rows {r['n_1600']}) stamps "
                f"{r['same_stamps']}  pred_adj bitwise {r['pred_adj_bitwise_rows']}/{r['rows']} "
                f"max|d| {r['pred_adj_max_abs_diff']:.3e}  -> {r['verdict']}"
            )
    g = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    g.to_csv(OUT / "gate.csv", index=False)

    # the regular-hours filter vs the local results_none_rth.csv the stored tables came from
    ref = {}
    for ln in Path("cluster/pool_momvix_rth_ref_md5.txt").read_text().split("\n"):
        if ln.strip():
            m, b, e = ln.split()
            ref[(b, e)] = m
    fr = []
    for (b, e), m in sorted(ref.items()):
        s = stored(b, e)
        cands = [s] + (
            [
                OLD
                / "linear_subsection"
                / b
                / "none"
                / e
                / f"tw{TW}"
                / "causal_tune_linear"
                / e
                / b
                / "results.csv"
            ]
            if e == "reclasso"
            else []
        )
        for c in cands:
            if not c.is_file():
                continue
            mt, mp = md5(rth_text(c)), md5(rth_pandas(c))
            fr.append(
                {
                    "bucket": b,
                    "estimator": e,
                    "stored": str(c),
                    "local_md5": m,
                    "text_filter_md5": mt,
                    "pandas_filter_md5": mp,
                    "text_match": mt == m,
                    "pandas_match": mp == m,
                }
            )
            print(
                f"rth filter {b:12s} {e:12s} {c.parts[-9]}: text {mt == m}  pandas {mp == m}"
            )
    pd.DataFrame(fr).to_csv(OUT / "rth_filter_check.csv", index=False)

    if ok_all or tol_all:
        (OUT / "GATE_OK").write_text(
            "bitwise\n" if ok_all else f"within {TOL_ABS:g} (not bitwise)\n"
        )
        print("GATE_OK written:", "bitwise" if ok_all else f"within {TOL_ABS:g}")
        return 0
    print("GATE FAILED: no flag written")
    return 1


def collect(bucket: str = "moments_vix") -> int:
    ref_dates = pd.read_csv(stored("baseline", "ridge"), usecols=["date"])["date"]
    ref_rth = pd.read_csv(io_bytes(rth_text(stored("baseline", "ridge"))))["date"]
    rows, ok_all, sums = [], True, []
    for est in ESTS:
        p = ours("fleet", bucket, est)
        arm = p.parents[3]
        r: dict = {"estimator": est, "results": str(p)}
        if not (arm / "DONE").is_file() or not p.is_file():
            r["verdict"] = "FAIL: not done"
            ok_all = False
            rows.append(r)
            continue
        x = pd.read_csv(p, float_precision=RT)
        txt, pdt = rth_text(p), rth_pandas(p)
        (p.parent / "results_none_rth.csv").write_bytes(txt)
        rth = pd.read_csv(io_bytes(txt))
        r.update(
            rows=len(x),
            same_stamps_as_stored=bool(
                len(x) == len(ref_dates) and (x["date"] == ref_dates).all()
            ),
            finite_pred_adj=int(np.isfinite(x["pred_adj"]).sum()),
            rth_rows=len(rth),
            rth_same_stamps_as_stored=bool(
                len(rth) == len(ref_rth) and (rth["date"] == ref_rth).all()
            ),
            rth_text_equals_pandas=txt == pdt,
            seconds=(arm / "SECONDS").read_text().strip()
            if (arm / "SECONDS").is_file()
            else "",
        )
        good = (
            r["rows"] == N_OOS
            and r["same_stamps_as_stored"]
            and r["rth_same_stamps_as_stored"]
            and r["finite_pred_adj"] == N_OOS
        )
        r["verdict"] = "ok" if good else "FAIL"
        ok_all &= good
        rows.append(r)
        for f in (p, p.parent / "results_none_rth.csv"):
            sums.append(f"{md5(f.read_bytes())}  {f}")
        print(
            f"{est:12s} rows {r['rows']} stamps {r['same_stamps_as_stored']} finite "
            f"{r['finite_pred_adj']} rth {r['rth_rows']} ({r['rth_same_stamps_as_stored']}) "
            f"{r['seconds']} s -> {r['verdict']}"
        )
    sfx = (
        "" if bucket == "moments_vix" else f"_{bucket}"
    )  # the first bucket keeps its file names
    pd.DataFrame(rows).to_csv(OUT / f"collect{sfx}.csv", index=False)
    (OUT / f"MD5SUMS{sfx}").write_text("\n".join(sums) + "\n")
    if ok_all:
        (OUT / f"COLLECTED{sfx}").write_text("ok\n")
        return 0
    return 1


def io_bytes(b: bytes):
    import io

    return io.BytesIO(b)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "gate":
        sys.exit(gate())
    if step == "collect":
        sys.exit(collect(*sys.argv[2:3]))
    sys.exit("usage: python cluster/pool_momvix_check.py gate | collect")
