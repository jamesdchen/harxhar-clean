"""CARC twin of cluster/pool_momvix_check.py: gate, merge and collect of the pooled moments_vix arms.

Run on CARC from the deployment root /scratch1/jc_905/harxhar-pool-momvix (2026-10-06).

gate     The canary re-ran one slice of the three STORED pooled live_feasible arms of the CARC root
         (ridge / elastic net under results/linear_subsection, the recursive lasso under
         results/linear_subsection_lassofix; run 2026-09-22/23 on epyc-7513 with the same seven data
         files) with today's code, on epyc-7513: START = W + 199,250 (W = 96,000 bars; 199,250 =
         797 x 250 is a tune boundary of the whole-series run, where the warm state is
         cold-reseeded), END = -1, HALO = W.  PASS = the slice's 5,067 rows sit on the stored arm's
         last 5,067 stamps and pred_adj is equal bit for bit; otherwise the original lassofix
         canary's tolerance (max |pred_adj difference| < 1e-6) is reported and the flag is written
         only under it.  Also: the regular-hours filter (bar labels 10:00 .. 16:00) of each stored
         whole-series file vs the local results_none_rth.csv (md5 list shipped).
         Writes results/pool_momvix/gate.csv, rth_filter_check.csv, GATE_OK.
merge    <bucket> <est>: concatenates the fleet's tune-aligned chunks (results/pool_momvix/
         fleet_chunks/.../c00 .. cNN) in order into the whole-series results.csv (header once,
         every data line byte for byte), checks 204,317 rows on the stored arms' stamps, and writes
         results_none_rth.csv.  The chunks are the slices the gate validated: each starts at a tune
         boundary with a W-row halo, so chunk k reproduces rows [k L, (k + 1) L) of the whole-series
         run.
collect  every fleet arm merged and checked: results/pool_momvix/collect.csv, MD5SUMS, COLLECTED;
         `collect <bucket>` (2026-10-06: moments_vol_vix) suffixes the three names with _<bucket>.

Run (cluster):  python cluster/slurm/pool_momvix_check_carc.py gate | merge <bucket> <est> | collect
"""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OLD = Path("/scratch1/jc_905/harxhar-subsection/results")
OUT = Path("results/pool_momvix")
TW = 2000
N_OOS = 204_317
SLICE_ROWS = N_OOS - 199_250
TOL_ABS = 1e-6  # cluster/lassofix_canary.sh's float-path tolerance on pred_adj
RT = "round_trip"
RTH = {f"{h:02d}:{m:02d}" for h in range(10, 17) for m in (0, 30)} - {"16:30"}
ESTS = ("ridge", "reclasso", "reclasticnet")
COLS = ("pred_adj", "pred_raw", "true_adj", "true_raw")
GATE_BUCKET = "live_feasible"
FLEET_BUCKET = "moments_vix"


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


def arm_csv(arm: Path, bucket: str, est: str) -> Path:
    return arm / "causal_tune_linear" / est / bucket / "results.csv"


def arm_dir(tag: str, bucket: str, est: str) -> Path:
    return OUT / tag / bucket / "none" / est / f"tw{TW}"


def rth_text(path: Path) -> bytes:
    lines = path.read_bytes().split(b"\n")
    keep = [lines[0]] + [ln for ln in lines[1:] if ln and ln[11:16].decode() in RTH]
    return b"\n".join(keep) + b"\n"


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
        out[f"{c}_max_abs_diff"] = float(np.nanmax(d))
        out[f"{c}_max_rel_diff"] = float(np.nanmax(d / np.abs(b)))
    return out


def gate() -> int:
    rows, ok_all, tol_all = [], True, True
    for est in ESTS:
        p, s = (
            arm_csv(arm_dir("gate", GATE_BUCKET, est), GATE_BUCKET, est),
            stored(GATE_BUCKET, est),
        )
        r: dict = {
            "bucket": GATE_BUCKET,
            "estimator": est,
            "slice": str(p),
            "stored": str(s),
        }
        if not p.is_file() or not s.is_file():
            r["verdict"] = "FAIL: missing file"
            ok_all = tol_all = False
            rows.append(r)
            continue
        r.update(
            compare(
                pd.read_csv(p, float_precision=RT), pd.read_csv(s, float_precision=RT)
            )
        )
        base_ok = r["same_stamps"] and r["rows"] == SLICE_ROWS
        bit = base_ok and r["pred_adj_bitwise_rows"] == r["rows"]
        tol = base_ok and r["pred_adj_max_abs_diff"] < TOL_ABS
        r["verdict"] = "bitwise" if bit else (f"within {TOL_ABS:g}" if tol else "FAIL")
        ok_all &= bit
        tol_all &= tol
        rows.append(r)
        print(
            f"{GATE_BUCKET} {est:12s} rows {r['rows']} (16:00 rows {r['n_1600']}) stamps {r['same_stamps']}  "
            f"pred_adj bitwise {r['pred_adj_bitwise_rows']}/{r['rows']} max|d| {r['pred_adj_max_abs_diff']:.3e} "
            f"-> {r['verdict']}"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "gate.csv", index=False)
    fr = []
    for ln in Path("cluster/pool_momvix_rth_ref_md5_carc.txt").read_text().split("\n"):
        if not ln.strip():
            continue
        m, b, e = ln.split()
        s = stored(b, e)
        mt = md5(rth_text(s)) if s.is_file() else ""
        fr.append(
            {
                "bucket": b,
                "estimator": e,
                "stored": str(s),
                "local_md5": m,
                "text_filter_md5": mt,
                "match": mt == m,
            }
        )
        print(f"rth filter {b} {e}: {mt == m}")
    pd.DataFrame(fr).to_csv(OUT / "rth_filter_check.csv", index=False)
    if ok_all or tol_all:
        (OUT / "GATE_OK").write_text(
            "bitwise\n" if ok_all else f"within {TOL_ABS:g} (not bitwise)\n"
        )
        print("GATE_OK written:", "bitwise" if ok_all else f"within {TOL_ABS:g}")
        return 0
    print("GATE FAILED: no flag written")
    return 1


def merge(bucket: str, est: str) -> int:
    chunks = sorted(arm_dir("fleet_chunks", bucket, est).glob("c[0-9][0-9]"))
    assert chunks, "no chunks"
    for c in chunks:
        assert (c / "DONE").is_file(), f"chunk {c} not done"
    parts = [arm_csv(c, bucket, est).read_bytes() for c in chunks]
    header = parts[0].split(b"\n", 1)[0]
    body = []
    for b in parts:
        h, rest = b.split(b"\n", 1)
        assert h == header, "chunk headers differ"
        body.append(rest if rest.endswith(b"\n") else rest + b"\n")
    text = header + b"\n" + b"".join(body)
    out = arm_csv(arm_dir("fleet", bucket, est), bucket, est)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(text)
    x = pd.read_csv(io.BytesIO(text), float_precision=RT)
    ref = pd.read_csv(stored(GATE_BUCKET, "ridge"), usecols=["date"])["date"]
    ok = (
        len(x) == N_OOS
        and bool((x["date"] == ref).all())
        and int(np.isfinite(x["pred_adj"]).sum()) == N_OOS
    )
    (out.parent / "results_none_rth.csv").write_bytes(rth_text(out))
    secs = [int((c / "SECONDS").read_text()) for c in chunks]
    print(
        f"merge {bucket} {est}: {len(chunks)} chunks, {len(x)} rows, stamps = stored {ok}, "
        f"chunk seconds max {max(secs)} sum {sum(secs)}"
    )
    if ok:
        (arm_dir("fleet", bucket, est) / "DONE").write_text(
            f"{len(chunks)} chunks, cpu seconds {sum(secs)}\n"
        )
    return 0 if ok else 1


def collect(bucket: str = FLEET_BUCKET) -> int:
    ref = pd.read_csv(stored(GATE_BUCKET, "ridge"), usecols=["date"])["date"]
    ref_rth = pd.read_csv(io.BytesIO(rth_text(stored(GATE_BUCKET, "ridge"))))["date"]
    rows, ok_all, sums = [], True, []
    for est in ESTS:
        a = arm_dir("fleet", bucket, est)
        p = arm_csv(a, bucket, est)
        r: dict = {"estimator": est, "results": str(p)}
        if not (a / "DONE").is_file():
            r["verdict"] = "FAIL: not merged"
            ok_all = False
            rows.append(r)
            continue
        x = pd.read_csv(p, float_precision=RT)
        rth = pd.read_csv(p.parent / "results_none_rth.csv")
        r.update(
            rows=len(x),
            same_stamps_as_stored=bool(len(x) == len(ref) and (x["date"] == ref).all()),
            finite_pred_adj=int(np.isfinite(x["pred_adj"]).sum()),
            rth_rows=len(rth),
            rth_same_stamps_as_stored=bool(
                len(rth) == len(ref_rth) and (rth["date"] == ref_rth).all()
            ),
            done=(a / "DONE").read_text().strip(),
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
            f"{est:12s} rows {r['rows']} stamps {r['same_stamps_as_stored']} rth {r['rth_rows']} -> {r['verdict']}"
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


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "gate":
        sys.exit(gate())
    if step == "merge" and len(sys.argv) == 4:
        sys.exit(merge(sys.argv[2], sys.argv[3]))
    if step == "collect":
        sys.exit(collect(*sys.argv[2:3]))
    sys.exit(
        "usage: python cluster/slurm/pool_momvix_check_carc.py gate | merge <bucket> <est> | collect"
    )
