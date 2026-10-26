"""Task files of the tree walks of results/fomcfix_2026-10-26/RULE.txt (gate T1, section 6), written once.

T1: per CPU model (EPYC 7513, 7542) the first and the last evaluation day (2014-01-02, 2024-04-30) of LightGBM, XGBoost
and rf_skx on the full set of the cell-A build.  Section 6: the full-set lines of the templates' task files
(results/pooled_trees_final_2014start_w3000/tasks_gb.txt, results/pooled_rf_skx_2014start_w3000/tasks.txt), i.e. their
piece partition, on the cell-B build.  Lines for code/tree_task.sh: <out> <est> <set> <refit> <first> <last> <i>/<n>.

usage: python tree_tasks.py <code tree> <run root on CARC> <out dir for the task files (relative to the code tree)>
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE, RUN, OUTREL = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
OUT = CODE / OUTREL
OUT.mkdir(parents=True, exist_ok=True)


def write(name: str, lines: list[str]) -> None:
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(name, len(lines))


for cpu in ("e7513", "e7542"):
    write(
        f"tasks_T1_{cpu}.txt",
        [
            f"{RUN}/gateT1/{cpu} {est} all_features bar {d} {d} 1/1"
            for est in ("lgbm", "xgb", "rf_skx")
            for d in ("2014-01-02", "2024-04-30")
        ],
    )
gb = (CODE / "results" / "pooled_trees_final_2014start_w3000" / "tasks_gb.txt").read_text().split("\n")
rf = (CODE / "results" / "pooled_rf_skx_2014start_w3000" / "tasks.txt").read_text().split("\n")
for est, src, n in (("lgbm", gb, 14), ("xgb", gb, 29), ("rf_skx", rf, 87)):
    ls = [ln.strip() for ln in src if ln.startswith(f"{est} all_features bar ")]
    assert len(ls) == n and all(ln.endswith(f"/{n}") for ln in ls), (est, len(ls))
    write(f"tasks_B_{est}.txt", [f"{RUN}/B {ln}" for ln in ls])
