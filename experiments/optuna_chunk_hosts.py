"""The node and CPU class of every chunk of the Optuna tree campaign (run on the CARC login node).

Why: the executor's design matrix X differs at rounding level between CPU vector classes
(agent D, 2026-09-29: AVX-512 nodes -- CARC xeon-4116 -- agree bit for bit with each other,
AVX2 nodes -- CARC epyc-7513 / epyc-7542 -- give a different X, live_feasible 28 of 232
columns, max abs 2.6e-11, the target identical), so the campaign's records are bit-identical
on the same CPU class and differ at rounding level across classes.  This script records, per
chunk-arm, where it ran:

  * from the chunk's host.txt (cluster/slurm/optuna_pack.sbatch writes it beside DONE since
    2026-09-29 23:15) when present, else
  * from the Slurm task logs logs/<job name>.<array job>.<task>.out of the given jobs (the
    "task N on <host>, stage K ...: <bucket> [<models>] ... chunk C" line and the model's
    "done in" line),

and the node's CPU class from `scontrol show node` (AvailableFeatures: epyc-7513 / epyc-7542
= AVX2, xeon-4116 = AVX-512).  Writes <root>/chunk_hosts.csv.

Usage (on CARC, repo root): python experiments/optuna_chunk_hosts.py --jobs 12480992,12480993,12480998,12481000
           [--root results/linear_subsection_trees_optuna_mask]
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_RE = re.compile(
    r"^task (\d+) on (\S+), stage (\d) \(\w+\), pool of \d+: (\S+) \[([\w,]+)\] tw(\d+) (\S+) chunk (\d+) rows"
)
DONE_RE = re.compile(r"^\s+(\w+) (\S+) c(\d+) (done in|already DONE)")
AVX512_CLASSES = ("xeon-4116",)  # AVX-512 (agent D); the epyc classes are AVX2


def node_class(host: str, cache: dict) -> str:
    short = host.split(".")[0]
    if short not in cache:
        try:
            out = subprocess.run(
                ["scontrol", "show", "node", short],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout
            m = re.search(r"AvailableFeatures=(\S+)", out)
            feats = m.group(1).split(",") if m else []
            cache[short] = next(
                (f for f in feats if f.startswith(("epyc", "xeon"))), "unknown"
            )
        except Exception:  # noqa: BLE001 -- a node that left the cluster
            cache[short] = "unknown"
    return cache[short]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/linear_subsection_trees_optuna_mask")
    ap.add_argument(
        "--jobs",
        required=True,
        help="comma-separated Slurm job ids (canary, stage 1, stage 2)",
    )
    a = ap.parse_args()
    root = ROOT / a.root
    rows: dict[tuple, dict] = {}
    for job in a.jobs.split(","):
        for log in sorted((ROOT / "logs").glob(f"*.{job.strip()}.*.out")):
            cur = None
            for line in log.read_text(errors="replace").splitlines():
                m = TASK_RE.match(line)
                if m:
                    cur = m.groups()
                    continue
                d = DONE_RE.match(line)
                if cur and d and d.group(4) == "done in":
                    task, host, stage, bucket, _models, tw, seg, chunk = cur
                    key = (int(stage), bucket, d.group(1), int(chunk))
                    rows[key] = {
                        "host": host,
                        "source": f"log {log.name}",
                        "task": task,
                        "seg": seg,
                        "tw": tw,
                    }
    cache: dict = {}
    out = []
    for stage in (1, 2):
        for done in sorted((root / f"stage{stage}").glob("*/*/*/tw*/chunks/c*/DONE")):
            d = done.parent
            chunk = int(d.name[1:])
            bucket, seg, model, tw = d.parts[-6], d.parts[-5], d.parts[-4], d.parts[-3]
            rec = rows.get((stage, bucket, model, chunk), {})
            host, source = rec.get("host", ""), rec.get("source", "")
            ht = d / "host.txt"
            if ht.is_file():
                kv = dict(
                    line.split("=", 1)
                    for line in ht.read_text().splitlines()
                    if "=" in line
                )
                host, source = kv.get("host", host), "host.txt"
            cls = node_class(host, cache) if host else "unknown"
            out.append(
                {
                    "stage": stage,
                    "bucket": bucket,
                    "segment": seg,
                    "model": model,
                    "train_win": tw,
                    "chunk": chunk,
                    "host": host,
                    "cpu_class": cls,
                    "avx512": cls in AVX512_CLASSES,
                    "source": source,
                }
            )
    p = root / "chunk_hosts.csv"
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]) if out else ["stage"])
        w.writeheader()
        w.writerows(out)
    by: dict[tuple, int] = {}
    for r in out:
        by.setdefault((r["stage"], r["cpu_class"]), 0)
        by[(r["stage"], r["cpu_class"])] += 1
    print(
        f"wrote {p}: {len(out)} DONE chunk-arms; by (stage, class): {dict(sorted(by.items()))}"
    )


if __name__ == "__main__":
    main()
