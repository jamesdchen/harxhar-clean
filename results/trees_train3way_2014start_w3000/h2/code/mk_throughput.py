"""Hoffman2 throughput for the trees' three-way comparison (owner, 02:40 PDT: lines go to Hoffman2 in waves sized by
measured throughput) -> the text written to CARC's runs/tasks/h2_throughput.txt.

Per gate tag (one CPU model): the accept verdict, the ALL48 forest full-set every-row gate piece's seconds a refit
(38 refits at its threads), CPU-seconds a refit against CARC's (145.9 s at 20 threads on EPYC 7513), the hours of a
91-refit line at those threads, the RTH13 full-set gate piece at 8 threads against CARC's 34.0 s probe, and the
8-slot tasks that fit in the free slots of that model's nodes now (a snapshot; 8-slot jobs started within minutes on
2026-10-24, 20-slot ones waited hours).

usage: python mk_throughput.py --summary JSON --accepted JSON --tags FILE --free8 FILE --out FILE
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

CARC_RTH13_T8 = 34.046971797943115  # timing/sec_RTH13_rf_skx_t8 (EPYC 7513, 3 refits, full set)
LINE_REFITS = 91


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("--summary", "--accepted", "--tags", "--free8", "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args()
    s: dict[str, Any] = json.loads(Path(a.summary).read_text())
    acc: dict[str, Any] = json.loads(Path(a.accepted).read_text())
    acc_cpu = {v["tag"]: cpu for cpu, v in acc.get("accepted", {}).items()}
    tags = {}
    for ln in Path(a.tags).read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            t, arch, _rx = ln.split(None, 2)
            tags[t] = arch
    free: dict[str, tuple[int, int]] = {}
    for ln in Path(a.free8).read_text().splitlines():
        if ln.strip():
            arch, h, c = ln.split()
            free[arch] = (int(h), int(c))
    carc = float(s["carc_sec_refit_all48_rf_full_t20"])
    carc_cpu = carc * 20
    out = [
        f"# Hoffman2 throughput, the ALL48 forest full set (rf_skx), written {time.strftime('%F %T %Z')} by the Hoffman2 "
        "executor (h2/code/mk_throughput.py)",
        f"# CARC reference: {carc:.1f} s a refit at 20 threads ({s.get('carc_cpu')}) = {carc_cpu:.0f} CPU-s a refit; "
        f"RTH13 full-set probe {CARC_RTH13_T8:.1f} s at 8 threads",
        "# Hoffman2 runs 8-thread tasks (-pe shared 8): 8-slot jobs start within minutes, 20-slot jobs waited > 1.75 h "
        "(2026-10-24 02:14-04:01); the forest's forecasts do not depend on threads (CARC rf_skx gate (d))",
        "# tag cpu_model accept threads sec_per_refit cpu_sec_per_refit cpu_ratio_to_carc hours_per_line_91_refits "
        "rth13_t8_sec rth13_t8_ratio_to_carc free_nodes_ge8 startable_8slot_tasks_now",
    ]
    total = 0
    for t, r in s["tags"].items():
        arch = tags.get(t, r.get("arch", "?"))
        ok = bool(r.get("accepted"))
        sec = r.get("sec_refit_all48_rf_full")
        nt = int(r.get("threads_all48_rf_full") or 0)
        rth = r["pieces"].get(f"RTH13_rf_skx_all_features_ends10_{t}_t8.parquet", {}).get("sec_refit")
        cpu = [acc_cpu[t]] if t in acc_cpu else ((r.get("accept") or {}).get("cpus") or ["?"])
        h, c = free.get(arch, (0, 0))
        if ok:
            total += c
        f = (
            f"{t} '{cpu[0]}' {'PASS' if ok else 'FAIL'} {nt} "
            + (
                f"{sec:.1f} {sec * nt:.0f} {sec * nt / carc_cpu:.2f} {LINE_REFITS * sec / 3600:.2f} "
                if sec
                else "- - - - "
            )
            + (f"{rth:.1f} {rth / CARC_RTH13_T8:.2f} " if rth else "- - ")
            + f"{h} {c}"
        )
        out.append(f)
    out.append(f"# startable 8-slot tasks now on passing models: {total} (a snapshot of free slots, not a reservation)")
    Path(a.out).write_text("\n".join(out) + "\n")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
