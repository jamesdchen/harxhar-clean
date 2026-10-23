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
    "diag_cond_names.py",
    "make_results.py",
):
    shutil.copy2(S / f, OUT / "scripts" / f)


def run(*args: str) -> dict:
    py = r"C:\Users\james\AppData\Local\Temp\w1500_venv\Scripts\python.exe"
    r = subprocess.run([py, "-I", *args], capture_output=True, text=True, check=True)
    return json.loads(r.stdout[r.stdout.index("{") :])


a_pre = run(str(S / "compare_a.py"), str(S / "a_off2"))
a_walk = run(str(S / "compare_a.py"), str(S / "a_off2"), "walk")
a_final = run(str(S / "compare_a.py"), str(S / "a_off"), "walk")  # the pre-addendum code (5db45e8)
(OUT / "check_a_flag_off.json").write_text(
    json.dumps(
        dict(
            reference="results/pooled_linear_rerun_2026-10-22_13_2012_w1000_latestart_nofomcrel_kkt (its _work in the kk_wt "
            "worktree) and its base run B (ls_wt _work, pretune); designs: run B's caches (read-only junctions)",
            pretune_select=a_pre,
            walks=a_walk,
            walks_pre_addendum_code_5db45e8=a_final,
        ),
        indent=1,
    )
)
bc = json.loads((S / "check_bc2.json").read_text())
(OUT / "check_b_c.json").write_text(json.dumps(bc, indent=1))
PRE = OUT / "pre_evidence"
PRE.mkdir(exist_ok=True)
bc0 = json.loads((S / "check_bc.json").read_text())
bc0["b_ridge_detail"] = json.loads((S / "check_b_ridge_detail.json").read_text())
(PRE / "check_b_c.json").write_text(json.dumps(bc0, indent=1))
shutil.copy2(S / "d_summary.json", PRE / "check_d.json")
for f in ("diag_kkt_on.json", "diag_kkt_off.json", "diag_cond.json"):
    shutil.copy2(S / f, PRE / f)
shutil.copy2(S / "d_on_provenance.json", OUT / "d_design_provenance.json")
for tag, d in (("flag_on", "d_on_run2"), ("flag_off", "d_off_run")):
    shutil.copy2(S / d / "frozen_penalties.json", OUT / "d_select" / f"frozen_penalties_{tag}.json")
    shutil.copy2(S / d / "pretune_mse.csv", OUT / "d_select" / f"pretune_mse_{tag}.csv")
d = run(str(S / "summarize_d.py"), str(S / "d_on_run2"), str(S / "d_off_run"), str(S / "a_off2"), str(S / "d_summary2.json"))
(OUT / "check_d.json").write_text(json.dumps(d, indent=1))
for f in ("diag_cond2.json", "diag_cond2_names.json"):
    shutil.copy2(S / f, OUT / f)
for f in ("diag_kkt_on.json", "diag_kkt_off.json", "diag_cond.json"):
    (OUT / f).unlink(missing_ok=True)  # moved to pre_evidence/
print("assembled", OUT)
