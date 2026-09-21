"""Local subsection regressions of the production linear spec, one process per arm.

Each arm runs specs/causal_tune_linear.py on ONE time-of-day subsection
(src.backtest.segmentation) with lag_scope=global: the HAR lags and the
diurnal target are built on the full 48-bar series and only the fitted rows are
sliced, so a subsection arm sees exactly the information the pooled arm sees
and differs only in having its own coefficients. ``none`` is the pooled arm.

Usage:
  python experiments/run_linear_subsection.py \
      --segments none bar1530 bar1600 blk_close --estimators ridge \
      --train-wins 500 2000 --workers 7
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "specs" / "causal_tune_linear.py"
OUT = ROOT / "results" / "linear_subsection"


def run_arm(arm: tuple[str, str, int, str]) -> tuple[str, int, float]:
    seg, est, tw, bucket = arm
    out = OUT / bucket / seg / est / f"tw{tw}"
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        HPC_KW_SEGMENT=seg,
        HPC_KW_LAG_SCOPE="global",
        HPC_KW_ESTIMATOR=est,
        HPC_KW_EXOG_BUCKET=bucket,
        HPC_KW_TRAIN_WIN=str(tw),
        HPC_RESULT_DIR=str(out),
    )
    t0 = time.time()
    with open(out / "run.log", "w", encoding="utf-8") as log:
        rc = subprocess.run(
            [sys.executable, "-u", str(SPEC)],
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    return f"{bucket}/{seg}/{est}/tw{tw}", rc, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", nargs="+", required=True)
    ap.add_argument("--estimators", nargs="+", default=["ridge"])
    ap.add_argument("--train-wins", nargs="+", type=int, default=[500])
    ap.add_argument("--bucket", default="baseline")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    arms = [
        (s, e, tw, a.bucket)
        for s in a.segments
        for e in a.estimators
        for tw in a.train_wins
    ]
    rc = 0
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for name, code, secs in pool.map(run_arm, arms):
            print(f"done {name} rc={code} {secs:.0f}s", flush=True)
            rc = rc or code
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
