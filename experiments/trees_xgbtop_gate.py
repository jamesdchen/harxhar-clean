"""Reproduction gate of the per-bar XGBoost re-run with the top dev-span entry (xgbtop, user decision 2026-10-07).

The xgbtop run (specs/causal_tune_trees_xgbtop.py) is specs/causal_tune_trees.py with the XGB_PARAMS block replaced
by a configuration file (experiments/trees_xgbtop_make_specs.py).  Before it runs, the SAME generator with the SHIPPED
XGBoost's configuration (cluster/xgbtop_shipped_xgb.json = xgb_opt_06 -> specs/causal_tune_trees_xgb06.py) must
reproduce the stored masked XGBoost cells byte for byte on the CPU class the fleet runs on: then the xgbtop run
differs from the shipped XGBoost by XGB_PARAMS alone.

Gate line (--lines, one per line):  "<rung> <bucket> <chunk> <start> <nrows> <ref_root> <stored_root>"
  rung         t1 | t10 (cluster/slurm/trees_mask_pack.sbatch's cadences: t10 REFIT_EVERY 10 / IMPORTANCE_EVERY 1,
               t1 1 / 10; TreeSHAP at bar1600; WINDOW_MASK 1; one thread per fit)
  chunk/start  the stored chunk whose first nrows rows are re-run (END = start + nrows, HALO = 2000)
  ref_root     the deployment root holding it (read only): /scratch1/jc_905/harxhar-mask for baseline / all_features
               (the masked campaign), /scratch1/jc_905/harxhar-momvolvix-trees for lean_vol_vix
  stored_root  its results root under <ref_root>/results (linear_subsection_trees_mask[_momvolvix])

Compared, per line:
  1. every recorded array of the re-run chunk npz against the stored chunk npz, BYTE FOR BYTE -- the rules of
     experiments/trees_momvix_gate.py (imported, unedited: compare()): per row date / pred_adj / true_adj / true_raw /
     baseline / e2_adj / shap, per refit refit_row / importance / kept_n / kept, feature_names, meta params /
     refit_every / importance_every / threads / window_mask / versions;
  2. (--tables DIR) the re-run forecasts against the PUBLISHED table cells: yhat of
     DIR/yhat_subtree_daily_<bucket>_xgb.parquet (t1) or DIR/yhat_subtree_xgb_<bucket>.parquet (t10) on the re-run
     rows' stamps (naive ET -> UTC, the stacker's rule), float64 bytes equal after the published pipeline's CSV round
     trip (canonical_chain: the reducer's to_csv and the stacker's default-parser read_csv; pandas' default parser is
     not round-trip exact, so a published cell can sit an ULP or two from the model's forecast -- found on the first
     gate run, t10 lean_vol_vix 2018-06-27: text 1.8460233211517334, npz 1.8460233211517332, table
     1.8460233211517327); the direct npz-vs-table comparison is reported, not gating; and the re-run's results CSV TEXT
     (date, true_adj, pred_adj, true_raw) equal to the stored chunk's -- results/spxw_pnl/'s tables shipped to the
     deployment (the stacker drops rows with a non-positive target; those rows are counted, not compared).
Output (--out): one run dir per line, gate_rows.csv, gate_lines.csv, GATE_OK when every line passed both checks.
Exit status 0 iff GATE_OK.

Run:  python experiments/trees_xgbtop_gate.py --lines cluster/trees_xgbtop_gate_lines.txt --out <dir>
          [--tables ref_tables] [--spec specs/causal_tune_trees_xgb06.py] [--procs 20]
"""

from __future__ import annotations

import argparse
import csv
import io
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

import trees_momvix_gate as g  # noqa: E402 -- spec_env, compare, npz_of, same (unedited)

SPEC = "specs/causal_tune_trees_xgb06.py"
MODEL = "xgb"
TABLES = {
    "t1": "yhat_subtree_daily_{b}_xgb.parquet",
    "t10": "yhat_subtree_xgb_{b}.parquet",
}


def canonical_chain(pred: np.ndarray) -> np.ndarray:
    """The published pipeline's CSV round trip applied to a chunk's forecasts: the chunk npz holds pandas' DEFAULT
    parse of the chunk CSV (the spec's own read_csv), the reducer (experiments/trees_cadence_reduce_chunks.py) writes
    those values with to_csv and the stacker (build_subsection_yhat.read_arm) parses the merged CSV with the default
    parser again.  The default parser is not round-trip exact, so a published cell can sit an ULP or two from the
    forecast the model wrote (e.g. text 1.8460233211517334 -> 1.8460233211517332 -> 1.8460233211517327)."""
    text = pd.DataFrame({"pred_adj": pred}).to_csv(index=False)
    return pd.read_csv(io.StringIO(text))["pred_adj"].to_numpy(np.float64)


def table_check(
    npz: Path, rung: str, bucket: str, tables: Path, ref_npz: Path
) -> list[dict]:
    tab = tables / TABLES[rung].format(b=bucket)
    if not tab.is_file():
        return [{"key": "table", "identical": False, "detail": f"missing {tab}"}]
    t = pd.read_parquet(tab).set_index("t")
    z = np.load(npz, allow_pickle=False)
    stamps = (
        pd.DatetimeIndex(pd.to_datetime(z["date"]))
        .tz_localize("America/New_York")
        .tz_convert("UTC")
        .as_unit("us")
    )
    pred = np.asarray(z["pred_adj"], dtype=np.float64)
    hit = stamps.isin(t.index)
    ok_target = (z["true_adj"] > 0) & (z["true_raw"] > 0)
    rows = []
    if (hit != ok_target).any():
        rows.append(
            {
                "key": "table stamps",
                "identical": False,
                "detail": f"{int(hit.sum())} stamps in the table, {int(ok_target.sum())} rows with a positive target",
            }
        )
    got = t.loc[stamps[hit], "yhat"].to_numpy(np.float64)
    ok, why = g.same(canonical_chain(pred)[hit], got)
    rows.append(
        {
            "key": f"table yhat = the re-run through the published CSV chain ({tab.name}, {int(hit.sum())} cells, "
            f"{int((~hit).sum())} dropped rows)",
            "identical": ok and int(hit.sum()) > 0,
            "detail": why,
        }
    )
    raw_ok, raw_why = g.same(
        pred[hit], got
    )  # informational: the parse chain's ULP drift, not gating
    rows.append(
        {
            "key": "INFO table yhat vs the re-run's npz forecast directly",
            "identical": True,
            "detail": "identical" if raw_ok else f"(informational) {raw_why}",
        }
    )
    # the forecast TEXT the model wrote: the re-run's results CSV against the stored chunk's, row for row
    a = pd.read_csv(npz.with_name("results_bar1600.csv"), dtype=str)
    b = pd.read_csv(ref_npz.with_name("results_bar1600.csv"), dtype=str).iloc[: len(a)]
    cols = ["date", "true_adj", "pred_adj", "true_raw"]
    same_text = list(a.columns) == list(b.columns) and all(
        (a[c].to_numpy() == b[c].to_numpy()).all() for c in cols
    )
    rows.append(
        {
            "key": "results CSV text (date, true_adj, pred_adj, true_raw) = the stored chunk's",
            "identical": bool(same_text),
            "detail": ""
            if same_text
            else f"columns {list(a.columns)} vs {list(b.columns)}",
        }
    )
    return rows


def run_line(line: str, out: Path, spec: str, tables: Path | None) -> dict:
    rung, bucket, chunk, start, nrows, ref_root, stored = line.split()
    chunk_i, start_i, n = int(chunk), int(start), int(nrows)
    tag = f"{rung}_{bucket}_{MODEL}_c{chunk_i}_n{n}"
    run_dir = out / tag
    ref = g.npz_of(
        Path(ref_root)
        / "results"
        / stored
        / rung
        / bucket
        / g.SEG
        / MODEL
        / f"tw{g.TW}"
        / "chunks"
        / f"c{chunk_i}",
        MODEL,
        bucket,
    )
    rec = {
        "line": tag,
        "rung": rung,
        "bucket": bucket,
        "chunk": chunk_i,
        "start": start_i,
        "nrows": n,
        "ref": str(ref),
    }
    if not ref.is_file():
        return rec | {
            "ok": False,
            "why": f"no stored chunk npz {ref}",
            "seconds": 0.0,
            "_rows": [],
        }
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(run_dir / "run.log", "w", encoding="utf-8") as log:
        p = subprocess.run(
            [sys.executable, "-u", spec],
            cwd=ROOT,
            env=g.spec_env(rung, bucket, MODEL, start_i, start_i + n, run_dir, 1),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    sec = time.time() - t0
    new = g.npz_of(run_dir, MODEL, bucket)
    if p.returncode != 0 or not new.is_file():
        return rec | {
            "ok": False,
            "why": f"spec exit {p.returncode}",
            "seconds": sec,
            "_rows": [],
        }
    rows = g.compare(new, ref, n)
    if tables is not None:
        rows += table_check(new, rung, bucket, tables, ref)
    for r in rows:
        r.update(line=tag)
    bad = [r for r in rows if not r["identical"]]
    return rec | {
        "ok": not bad,
        "why": "; ".join(f"{r['key']}: {r['detail']}" for r in bad)[:500],
        "seconds": sec,
        "keys": len(rows),
        "_rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--spec", default=SPEC, help="the generated spec of the SHIPPED configuration"
    )
    ap.add_argument(
        "--tables",
        default="",
        help="folder with the published yhat_subtree*_xgb* tables",
    )
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "GATE_OK").unlink(missing_ok=True)
    tables = Path(a.tables).resolve() if a.tables else None
    host = socket.gethostname()
    lines = [
        ln.strip()
        for ln in Path(a.lines).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    for ln in lines:
        if ln.split()[0] not in TABLES or len(ln.split()) != 7:
            raise SystemExit(f"bad gate line: {ln!r}")
    print(
        f"xgbtop gate on {host}: {len(lines)} lines through {a.spec}, {a.procs} at once (1 thread each); tables {tables}",
        flush=True,
    )
    results = []
    with ThreadPoolExecutor(max(1, a.procs)) as ex:
        for r in ex.map(lambda ln: run_line(ln, out, a.spec, tables), lines):
            print(
                f"  {r['line']}: {'IDENTICAL' if r['ok'] else 'DIFFERS'} ({r['seconds']:.0f}s) {r['why']}",
                flush=True,
            )
            results.append(r)
    with open(out / "gate_rows.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["line", "key", "identical", "detail"])
        w.writeheader()
        for r in results:
            for x in r["_rows"]:
                w.writerow({k: x[k] for k in ("line", "key", "identical", "detail")})
    cols = [
        "line",
        "rung",
        "bucket",
        "chunk",
        "start",
        "nrows",
        "ok",
        "keys",
        "seconds",
        "why",
        "ref",
    ]
    with open(out / "gate_lines.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["host"])
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in cols} | {"host": host})
    n_ok = sum(bool(r["ok"]) for r in results)
    print(
        f"{n_ok} of {len(results)} lines identical (chunk npz{' + published table cells' if tables else ''})"
    )
    if results and n_ok == len(results):
        (out / "GATE_OK").write_text(
            f"{host} {time.strftime('%F %T')}\n", encoding="utf-8"
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
