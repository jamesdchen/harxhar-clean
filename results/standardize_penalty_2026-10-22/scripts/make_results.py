"""Assemble results/standardize_penalty_2026-10-22/ in the std worktree from the scratch runs (small files only: no
walk forecasts, no out-of-sample score).

usage: python make_results.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

SP = Path(r"C:\Users\james\AppData\Local\Temp\claude\C--Users-james-CC-Allowed-harxhar-0dte-professor\bd4224e0-12aa-4614-b6a6-3fe50535ade1\scratchpad")
S = SP / "std"
WT = Path(r"C:\Users\james\AppData\Local\Temp\std_wt")
OUT = WT / "results" / "standardize_penalty_2026-10-22"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "scripts").mkdir(exist_ok=True)
(OUT / "d_select").mkdir(exist_ok=True)

shutil.copy2(S / "RULE_std.txt", OUT / "RULE_std.txt")
for f in (
    "test_unit.py",
    "run_orch.sh",
    "drive_walk.py",
    "compare_a.py",
    "d_on_provenance.py",
    "check_bc.py",
    "check_b_ridge_detail.py",
    "summarize_d.py",
    "diag_kkt.py",
    "diag_cond.py",
    "make_results.py",
):
    shutil.copy2(S / f, OUT / "scripts" / f)


def run(*args: str) -> dict:
    py = r"C:\Users\james\AppData\Local\Temp\w1500_venv\Scripts\python.exe"
    r = subprocess.run([py, "-I", *args], capture_output=True, text=True, check=True)
    return json.loads(r.stdout[r.stdout.index("{") :])


a_pre = run(str(S / "compare_a.py"), str(S / "a_off"))
a_walk = run(str(S / "compare_a.py"), str(S / "a_off"), "walk")
a_final = run(str(S / "compare_a.py"), str(S / "a_off_final"), "walk")
(OUT / "check_a_flag_off.json").write_text(
    json.dumps(
        dict(
            reference="results/pooled_linear_rerun_2026-10-22_13_2012_w1000_latestart_nofomcrel_kkt (its _work in the kk_wt "
            "worktree) and its base run B (ls_wt _work, pretune); designs: run B's caches (read-only junctions)",
            pretune_select=a_pre,
            walks=a_walk,
            walks_rerun_on_committed_code_baseline=a_final,
        ),
        indent=1,
    )
)
bc = json.loads((S / "check_bc.json").read_text())
bc["b_ridge_detail"] = json.loads((S / "check_b_ridge_detail.json").read_text())
(OUT / "check_b_c.json").write_text(json.dumps(bc, indent=1))
shutil.copy2(S / "d_on_provenance.json", OUT / "d_design_provenance.json")
for tag, d in (("flag_on", "d_on_run"), ("flag_off", "d_off_run")):
    shutil.copy2(S / d / "frozen_penalties.json", OUT / "d_select" / f"frozen_penalties_{tag}.json")
    shutil.copy2(S / d / "pretune_mse.csv", OUT / "d_select" / f"pretune_mse_{tag}.csv")
d = run(str(S / "summarize_d.py"), str(S / "d_on_run"), str(S / "d_off_run"), str(S / "a_off"), str(S / "d_summary.json"))
(OUT / "check_d.json").write_text(json.dumps(d, indent=1))
for f in ("diag_kkt_on.json", "diag_kkt_off.json", "diag_cond.json"):
    shutil.copy2(S / f, OUT / f)
print("assembled", OUT)
