"""Assemble results/standardize_penalty_2026-10-22/ for the combined change (addendum 2: k = 1 evidence + centring);
the addendum-1 results move to addendum1/ (the pre-evidence ones stay in pre_evidence/).  Small files only; no walk
forecasts, no out-of-sample score.

usage: python make_results4.py
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
A1 = OUT / "addendum1"
A1.mkdir(exist_ok=True)
for f in ("check_a_flag_off.json", "check_b_c.json", "check_d.json", "diag_cond2.json", "diag_cond2_names.json"):
    if (OUT / f).exists():
        shutil.move(str(OUT / f), str(A1 / f))
if (OUT / "d_select").exists() and not (A1 / "d_select").exists():
    shutil.move(str(OUT / "d_select"), str(A1 / "d_select"))
(OUT / "d_select").mkdir(exist_ok=True)
shutil.copy2(S / "RULE_std.txt", OUT / "RULE_std.txt")
for f in (
    "test_unit.py",
    "run_orch.sh",
    "drive_walk.py",
    "compare_a.py",
    "d_on_provenance.py",
    "check_bc.py",
    "summarize_d.py",
    "diag_cond_names.py",
    "make_results.py",
    "make_results4.py",
):
    shutil.copy2(S / f, OUT / "scripts" / f)


def run(*args: str) -> dict:
    py = r"C:\Users\james\AppData\Local\Temp\w1500_venv\Scripts\python.exe"
    r = subprocess.run([py, "-I", *args], capture_output=True, text=True, check=True)
    return json.loads(r.stdout[r.stdout.index("{") :])


(OUT / "check_a_flag_off.json").write_text(
    json.dumps(
        dict(
            reference="results/pooled_linear_rerun_2026-10-22_13_2012_w1000_latestart_nofomcrel_kkt (kk_wt _work) and its "
            "base run B (ls_wt _work, pretune); designs: run B's caches (read-only junctions); code 58dca0c",
            pretune_select=run(str(S / "compare_a.py"), str(S / "a_off4")),
            walks=run(str(S / "compare_a.py"), str(S / "a_off4"), "walk"),
        ),
        indent=1,
    )
)
shutil.copy2(S / "check_bc4.json", OUT / "check_b_c.json")
for tag, d in (("flag_on", "d_on_run4"), ("flag_off", "d_off_run")):
    shutil.copy2(S / d / "frozen_penalties.json", OUT / "d_select" / f"frozen_penalties_{tag}.json")
    shutil.copy2(S / d / "pretune_mse.csv", OUT / "d_select" / f"pretune_mse_{tag}.csv")
d = run(str(S / "summarize_d.py"), str(S / "d_on_run4"), str(S / "d_off_run"), str(S / "a_off4"), str(S / "d_summary4.json"))
d["conditioning"] = json.loads((S / "diag_cond4.json").read_text())
(OUT / "check_d.json").write_text(json.dumps(d, indent=1))
print("assembled", OUT)
