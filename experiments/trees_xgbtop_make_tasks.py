"""Task file of the per-bar XGBoost re-run with the top dev-span entry (xgbtop, user decision 2026-10-07): rungs T1 / T10
of the per-bar 16:00 XGBoost on baseline, all_features and lean_vol_vix, through cluster/slurm/trees_xgbtop_pack.sbatch
(the masked fleet, WINDOW_MASK=1, and the unmasked twin, WINDOW_MASK=0, read the same file).

The random forest's generator (experiments/trees_rfdev_make_tasks.py) with the model set to xgb: chunk geometry and
packing are the campaign generator's (experiments/trees_cadence_make_tasks.py: its ``chunks`` and ``sec_per_row`` are
called; chunks of T10 start on multiples of 10; chunking changes no number).  One-thread CARC seconds per refit of
xgb_opt_00 (--fit-sec bucket=seconds,...: the stored xgb_opt_06 chunks' mean fit_sec on epyc-7513 times the measured
cost ratio xgb_opt_00 / xgb_opt_06) size the chunks to TARGET_SEC seconds a process.

  cluster/trees_xgbtop_tasks_untuned.txt   "<pack> <rung> <bucket> xgb <tw> <seg> <chunk> <start> <end> <halo>"
      pack 0 = the canary (chunk c0 of the 6 arms); packs 1.. = the fleet, longest-processing-time first.
  cluster/trees_xgbtop_tasks_xcpu.txt      the cross-CPU cell (--xcpu-arm, default t1:all_features): every chunk of that
      arm in ONE pack (pack 0), re-run on epyc-7542 / xeon-4116 and compared with the epyc-7513 chunks byte for byte.

Run:  python experiments/trees_xgbtop_make_tasks.py --fit-sec baseline=0.3,all_features=5,lean_vol_vix=0.6
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import trees_cadence_make_tasks as tcm  # noqa: E402

BUCKETS = ("baseline", "all_features", "lean_vol_vix")
MODEL = "xgb"
TARGET_SEC = 600  # wall seconds a chunk process (the campaign's)
PREP_SEC = {
    "baseline": 10,
    "all_features": 45,
    "lean_vol_vix": 25,
}  # data preparation a process


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fit-sec",
        required=True,
        help="bucket=one-thread CARC seconds per refit of xgb_opt_00, comma list",
    )
    ap.add_argument(
        "--shap-sec",
        type=float,
        default=tcm.SHAP_SEC["xgb"],
        help="seconds per TreeSHAP call",
    )
    ap.add_argument(
        "--target-sec",
        type=int,
        default=TARGET_SEC,
        help="wall seconds a chunk process",
    )
    ap.add_argument(
        "--out",
        default="cluster/trees_xgbtop_tasks_untuned.txt",
        help="task file (repo-relative)",
    )
    ap.add_argument(
        "--xcpu-arm",
        default="t1:all_features",
        help="rung:bucket of the cross-CPU cell",
    )
    ap.add_argument("--xcpu-out", default="cluster/trees_xgbtop_tasks_xcpu.txt")
    a = ap.parse_args()
    fit = {k: float(v) for k, v in (x.split("=") for x in a.fit_sec.split(","))}
    assert set(fit) == set(BUCKETS), fit
    tcm.TARGET_SEC = a.target_sec
    tcm.SHAP_SEC[MODEL] = a.shap_sec
    for b in BUCKETS:
        tcm.FIT_SEC[(MODEL, b)] = fit[b]
        tcm.PREP_SEC[b] = PREP_SEC[b]
    procs = []
    for rung, (re_, _ie) in tcm.RUNGS.items():
        for b in BUCKETS:
            spr = tcm.sec_per_row(MODEL, b, rung)
            rows = int((a.target_sec - PREP_SEC[b]) / spr)
            for k, s, e in tcm.chunks(min(max(rows, 1), tcm.N_OOS), re_):
                n = (e if e >= 0 else tcm.W + tcm.N_OOS) - s
                procs.append((PREP_SEC[b] + n * spr, rung, b, MODEL, k, s, e))
    canary = sorted((p for p in procs if p[4] == 0), key=lambda p: (p[1] != "t1", p[2]))
    fleet = sorted((p for p in procs if p[4] != 0), key=lambda p: -p[0])
    n_packs = max(1, math.ceil(len(fleet) / tcm.PACK_PROCS))
    load, size, packed = [0.0] * n_packs, [0] * n_packs, []
    for p in fleet:
        j = min(
            (i for i in range(n_packs) if size[i] < tcm.PACK_PROCS),
            key=lambda i: load[i],
        )
        load[j] += p[0]
        size[j] += 1
        packed.append((j + 1, p))
    W, SEG = tcm.W, tcm.SEG
    lines = [
        f"0 {r} {b} {m} {W} {SEG} {k} {s} {e} {W}" for (_t, r, b, m, k, s, e) in canary
    ] + [
        f"{j} {r} {b} {m} {W} {SEG} {k} {s} {e} {W}"
        for j, (_t, r, b, m, k, s, e) in sorted(packed, key=lambda x: (x[0], x[1][1:]))
    ]
    f = ROOT / a.out
    f.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print(
        f"{f.name}: {len(canary)} canary processes (pack 0), {len(fleet)} fleet processes in packs 1-{n_packs}; "
        f"est max process {max(p[0] for p in procs):.0f}s (canary max {max(p[0] for p in canary):.0f}s), "
        f"pack sums {min(load):.0f}..{max(load):.0f}s, total {sum(p[0] for p in procs) / 3600:.2f} core-h"
    )
    for rung in tcm.RUNGS:
        for b in BUCKETS:
            ps = [p for p in procs if p[1] == rung and p[2] == b]
            print(
                f"  {rung:3s} {b:13s} {len(ps):3d} chunks, est {sum(p[0] for p in ps) / 3600:5.2f} h"
            )
    xr, xb = a.xcpu_arm.split(":")
    xs = sorted((p for p in procs if p[1] == xr and p[2] == xb), key=lambda p: p[4])
    if len(xs) > tcm.PACK_PROCS:
        raise SystemExit(
            f"cross-CPU arm {a.xcpu_arm} has {len(xs)} chunks > {tcm.PACK_PROCS} a pack"
        )
    xl = [f"0 {r} {b} {m} {W} {SEG} {k} {s} {e} {W}" for (_t, r, b, m, k, s, e) in xs]
    (ROOT / a.xcpu_out).write_bytes(("\n".join(xl) + "\n").encode("utf-8"))
    print(
        f"{a.xcpu_out}: {len(xl)} processes (the {a.xcpu_arm} arm, one pack), est max {max(p[0] for p in xs):.0f}s"
    )


if __name__ == "__main__":
    main()
