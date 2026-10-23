"""Tight Slurm time limits for the final pooled MinGRU's GPU jobs (2026-10-23, coordinator: our jobs start mostly by
backfill, which needs time limits close to the runtime).

For every jobs file of both windows' roots (tune files of every round, walk files): the expected GPU wall seconds =
jobs x per-job seconds, where the per-job seconds of a (set, H, L) come from the completed runs (the trainer's pool
phase / jobs, median over that cell's runs, the W = 3,500 root's own runs where it has them, else W = 3,000's x 45,500
/ 39,000); a cell not yet measured uses the runner's cost model (job_cost); walk files use the probe's seconds a refit
(--walk-s-per-refit).  Limit = (seconds x factor + 300 s staging) in minutes, at least 20; factor 1.3 for a measured
cell, 1.6 for a modelled one.  Writes <out> lines "<root>/inputs/<file> <minutes> <measured|model>", and with --apply
lowers the TimeLimit of our PENDING jobs whose file is known (scontrol update; never raises one, never touches a
running job).

    python pooled_mingru_final_tlim.py --out /home1/jc_905/pmf/tlim_map.txt [--apply] [--walk-s-per-refit 2.04]
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOTS = {3000: Path("/home1/jc_905/pmf"), 3500: Path("/home1/jc_905/pmf35")}
POOL_RE = re.compile(r"pool phase \d+: (\d+) jobs on \d+ slots in ([0-9.]+) s")
HL_RE = re.compile(r" L=(\d+) H=(\d+) ")
SET_RE = re.compile(r"_(baseline|all_features)[._]")


def first_hl(path: Path) -> tuple[int, int]:
    with open(path) as fh:
        m = HL_RE.search(fh.readline())
    assert m is not None, path
    return int(m.group(2)), int(m.group(1))  # (H, L)


def minutes(lim: str) -> int:
    """A Slurm time limit (d-hh:mm:ss, hh:mm:ss, mm:ss or mm) in whole minutes."""
    days, _, rest = lim.rpartition("-")
    parts = [int(x) for x in rest.split(":")]
    if len(parts) == 3:
        mins = parts[0] * 60 + parts[1]
    elif len(parts) == 2:
        mins = parts[0]
    else:
        mins = parts[0]
    return mins + (int(days) * 1440 if days else 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--walk-s-per-refit", type=float, default=2.04)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOTS[3000] / "code"))
    import pooled_mingru_final_2014_w3000 as m  # noqa: E402  (job_cost; W = 3,000 constants)

    meas: dict[tuple[int, str, int, int], list[float]] = {}
    for w, root in ROOTS.items():
        sub = {}
        for ln in (root / "inputs" / "submitted.txt").read_text().splitlines():
            p = ln.split()
            if len(p) >= 2:
                sub[p[1]] = p[0]
        for d in (root / "runs").glob("tune_*_r*_*"):
            log = d / "tune.log"
            if not log.is_file() or not (d / "tune" / "jobs_out.csv").is_file():
                continue
            pools = [
                p
                for p in (POOL_RE.search(ln) for ln in log.read_text().splitlines())
                if p is not None
            ]

            if not pools:
                continue
            nj = sum(int(p.group(1)) for p in pools)
            secs = sum(float(p.group(2)) for p in pools)
            s = SET_RE.search(d.name + "_")
            jid = d.name.rsplit("_", 1)[1]
            fname = sub.get(jid)
            if fname is None or s is None:
                continue
            H, L = first_hl(root / "inputs" / fname)
            meas.setdefault((w, s.group(1), H, L), []).append(secs / nj)
    out = []
    files = {}
    for w, root in ROOTS.items():
        for f in sorted((root / "inputs").glob("tune_*_r*.txt")) + sorted(
            (root / "inputs").glob("walk_*.b*.txt")
        ):
            s = SET_RE.search(f.name)
            if s is None:
                continue
            n = sum(1 for _ in open(f))
            sec: float
            if f.name.startswith("walk_"):
                sec, kind = n * a.walk_s_per_refit * (w / 3000), "probe"
            else:
                H, L = first_hl(f)
                own = meas.get((w, s.group(1), H, L))
                other = meas.get((3000, s.group(1), H, L))
                if own:
                    sec, kind = n * statistics.median(own), "measured"
                elif other:
                    sec, kind = n * statistics.median(other) * w / 3000, "measured"
                else:
                    sec, kind = n * m.job_cost(s.group(1), H, L) * w / 3000, "model"
            fac = 1.6 if kind == "model" else 1.3
            mins = max(20, math.ceil((sec * fac + 300) / 60))
            files[str(root / "inputs" / f.name)] = mins
            out.append(f"{root / 'inputs' / f.name} {mins} {kind}")
    Path(a.out).write_text("\n".join(out) + "\n")
    print(f"{len(out)} files; measured cells {len(meas)}")
    if a.apply:
        q = subprocess.run(
            ["squeue", "-u", "jc_905", "-h", "-t", "PD", "-o", "%i %j %l"],
            capture_output=True,
            text=True,
        )
        sub_all = {}
        for root in ROOTS.values():
            for ln in (root / "inputs" / "submitted.txt").read_text().splitlines():
                p = ln.split()
                if len(p) >= 2:
                    sub_all[p[1]] = str(root / "inputs" / p[0])
        changed = 0
        for ln in q.stdout.splitlines():
            jid, name, lim = ln.split()
            if (
                not name.startswith("pmf")
                or jid not in sub_all
                or sub_all[jid] not in files
            ):
                continue
            new = files[sub_all[jid]]
            cur = minutes(lim)
            if new < cur:
                r = subprocess.run(
                    ["scontrol", "update", f"jobid={jid}", f"TimeLimit={new}"],
                    capture_output=True,
                    text=True,
                )
                changed += r.returncode == 0
        print(f"time limits lowered on {changed} pending jobs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
