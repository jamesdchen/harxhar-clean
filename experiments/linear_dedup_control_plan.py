"""I1 attribution control: which arms moved, and is it the dropped columns or the float path?

After the de-dup fleet, some one-bar arms differ from their pre-dedup forecast by more than the
float-path tolerance (experiments/linear_dedup_gates.py FLOAT_PATH_TOL).  To attribute that change,
each such arm (bar) is run once more ON THE SAME CLUSTER AND ENVIRONMENT as the de-dup fleet with
the PRE-dedup executor (src/backtest/executor.py at 47f7f9c^ -- the only file of src/ the change
touched), everything else identical.  Then
  control vs old   = what the cluster / environment / software drift alone does (same design),
  control vs new   = what dropping the twelve session-edge columns alone does (same cluster).

Writes cluster/linear_dedup_control_tasks.txt (the fleet's line format, results root
results/linear_subsection_dedup_control/...) and results/linear_subsection_dedup/control_arms.csv.
Run:  python experiments/linear_dedup_control_plan.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from linear_dedup_gates import BARS, FLOAT_PATH_TOL, NEW  # noqa: E402
from linear_dedup_plan import PACK  # noqa: E402

OUT_TASKS = ROOT / "cluster" / "linear_dedup_control_tasks.txt"
CTRL = "results/linear_subsection_dedup_control"


def main() -> None:
    arms = pd.read_csv(NEW / "arm_list.csv")
    rows, lines = [], []
    for _, r in arms.iterrows():
        bars = BARS if int(r["segments_run"]) == len(BARS) else ("bar1600",)
        hit = []
        for bar in bars:
            pn = ROOT / r["run_path"].replace("bar1600", bar)
            po = ROOT / r["old_path"].replace("bar1600", bar)
            a, b = pd.read_csv(pn), pd.read_csv(po)
            rel = float(
                np.max(
                    np.abs(a["pred_adj"].to_numpy() / b["pred_adj"].to_numpy() - 1.0)
                )
            )
            if rel > FLOAT_PATH_TOL:
                hit.append(bar)
                rows.append(
                    {"forecast": r["key"], "bar": bar, "max_rel_new_vs_old": rel}
                )
        if not hit:
            continue
        root = r["results_root"].replace("results/linear_subsection_dedup", CTRL, 1)
        har = "-" if pd.isna(r["har_base"]) else str(int(r["har_base"]))
        for k in range(0, len(hit), PACK):  # packs of PACK one-bar arms, as the fleet
            lines.append(
                f"{root} {r['bucket']} {r['run_estimator']} {int(r['train_win'])} {har} "
                f"{r['lag_scope']} {','.join(hit[k : k + PACK])}"
            )
    lines = sorted(set(lines))
    OUT_TASKS.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    pd.DataFrame(rows).to_csv(NEW / "control_arms.csv", index=False)
    print(
        f"{len(rows)} (forecast, bar) pairs above {FLOAT_PATH_TOL:g}; {len(lines)} control task lines"
    )
    for ln in lines:
        print("  " + ln)


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


# --------------------------------------------------------------------------- same-arch pairs
# The control showed the stored arms are not reproduced by a re-run of the SAME code on another
# machine for most moved (forecast, bar) pairs: Hoffman2's nodes are several CPU generations
# (E5-2670 AVX .. Gold 6140 AVX-512 ..), and numpy's kernels round differently per instruction set.
# The clean attribution holds the CPU fixed: every 16:00 pair of the control is run again, pinned
# (qsub -l arch=...) to the architecture of the node its control ran on, (i) with the de-dup
# executor (results/linear_subsection_dedup_samearch) and (ii) with the pre-dedup executor again
# (results/linear_subsection_dedup_control_rep, the same-arch determinism check).
HOSTS_ARCH = NEW / "control_hosts_arch.txt"  # "<host> <arch>" (qhost -F arch)


def samearch() -> None:
    arch_of = dict(
        ln.split() for ln in HOSTS_ARCH.read_text().splitlines() if ln.strip()
    )
    lines = OUT_TASKS.read_text().splitlines()
    host_of: dict[int, str] = {}
    for f in (ROOT / CTRL / "logs").glob("lin_dedup_ctrl.o*"):
        for ln in f.read_text(errors="replace").splitlines():
            if ln.startswith("task "):
                t, host = ln.split()[1], ln.split()[3].rstrip(":")
                host_of[int(t)] = host
    by_arch: dict[str, list[str]] = {}
    for i, ln in enumerate(lines, start=1):
        if "bar1600" not in ln.split()[-1]:
            continue
        head = " ".join(ln.split()[:-1])
        by_arch.setdefault(arch_of[host_of[i]], []).append(f"{head} bar1600")
    for arch, ls in sorted(by_arch.items()):
        for tag, root in (
            ("samearch", "results/linear_subsection_dedup_samearch"),
            ("ctrlrep", "results/linear_subsection_dedup_control_rep"),
        ):
            f = ROOT / "cluster" / f"linear_dedup_{tag}_{arch}.txt"
            f.write_text(
                "\n".join(x.replace(CTRL, root, 1) for x in ls) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        print(f"{arch}: {len(ls)} 16:00 pairs")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--samearch":
    samearch()
