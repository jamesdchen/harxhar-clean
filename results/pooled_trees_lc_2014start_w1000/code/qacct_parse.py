"""UGE accounting (Hoffman2): `qacct -j <job>` text -> one CSV row per job / array task.

usage: python qacct_parse.py <out csv> <qacct text file> [...]
Columns: jobnumber, taskid, jobname, owner, hostname, qname, granted_pe, slots, qsub_time, start_time, end_time,
ru_wallclock (s), cpu (s), maxvmem, exit_status, failed, wallclock_h, cpu_h, slot_h (= ru_wallclock x slots / 3600).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

KEYS = (
    "jobnumber",
    "taskid",
    "jobname",
    "owner",
    "hostname",
    "qname",
    "granted_pe",
    "slots",
    "qsub_time",
    "start_time",
    "end_time",
    "ru_wallclock",
    "cpu",
    "maxvmem",
    "exit_status",
    "failed",
)


def records(text: str) -> list[dict]:
    out, cur = [], {}
    for ln in text.splitlines():
        if ln.startswith("====="):
            if cur:
                out.append(cur)
            cur = {}
            continue
        parts = ln.split(None, 1)
        if len(parts) == 2 and parts[0] in KEYS:
            cur[parts[0]] = parts[1].strip()
    if cur:
        out.append(cur)
    return out


rows = []
for p in sys.argv[2:]:
    rows += records(Path(p).read_text(encoding="utf-8", errors="replace"))
R = pd.DataFrame(rows, columns=list(KEYS))
for c in ("ru_wallclock", "slots", "exit_status"):
    R[c] = pd.to_numeric(
        R[c].astype(str).str.replace("s", "", regex=False), errors="coerce"
    )
R["cpu"] = pd.to_numeric(
    R["cpu"].astype(str).str.replace("s", "", regex=False), errors="coerce"
)
R["wallclock_h"] = R["ru_wallclock"] / 3600
R["cpu_h"] = R["cpu"] / 3600
R["slot_h"] = R["ru_wallclock"] * R["slots"] / 3600
R = R.drop_duplicates(subset=["jobnumber", "taskid", "start_time"]).sort_values(
    ["jobnumber", "taskid"]
)
R.to_csv(sys.argv[1], index=False)
print(
    f"{len(R)} accounting records, {R['jobnumber'].nunique()} jobs; {R['slot_h'].sum():.1f} slot-hours, "
    f"{R['cpu_h'].sum():.1f} CPU-hours; exit status {R['exit_status'].value_counts().to_dict()}"
)
