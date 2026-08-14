"""Re-sign every finished sweep part with a0 (and blk2) using mid-implied IV."""

from __future__ import annotations

import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
PARTS = os.path.join(ROOT, "results", "spxw_pnl", "parts")
YH = {
    "a0": os.path.join(ROOT, "results", "spxw_pnl", "yhat_a0.parquet"),
    "blk2": os.path.join(ROOT, "results", "spxw_pnl", "yhat_blk2.parquet"),
}


def main() -> None:
    files = sorted(glob.glob(os.path.join(PARTS, "h*_sweep.parquet")))
    script = os.path.join(ROOT, "experiments", "spxw_resign.py")
    for path in files:
        for lab, yh in YH.items():
            if not os.path.exists(yh):
                continue
            print(f"==== {os.path.basename(path)} {lab} ====", flush=True)
            subprocess.check_call(
                [
                    PY,
                    "-u",
                    script,
                    "--trades",
                    path,
                    "--yhat",
                    yh,
                    "--label",
                    f"{lab}-{os.path.basename(path)}",
                    "--chain",
                    os.path.join(ROOT, "data", "spxw_chain.parquet"),
                ]
            )


if __name__ == "__main__":
    main()
