"""Local refit of linear + penalized linear models with overnight bars dropped.

Two slices (HAR lags rebuilt inside the slice, lag_scope=intra):
  rth    = 09:30-16:00 ET
  last30 = 15:30-16:00 ET

Estimators: ridge, reclasso (lasso), reclasticnet.
Bucket: baseline (HAR + calendar) first — the linear backbone.
Runs specs/causal_tune_linear.py with HPC_KW_* env, one process per arm.

Usage: python experiments/run_linear_no_overnight.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "specs" / "causal_tune_linear.py"
SEGMENTS = ("rth", "last30")
ESTIMATORS = ("ridge", "reclasso", "reclasticnet")
BUCKET = "baseline"


def main() -> None:
    py = sys.executable
    procs = []
    for seg in SEGMENTS:
        for est in ESTIMATORS:
            out = ROOT / "results" / "linear_no_on" / seg / est
            out.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["HPC_KW_SEGMENT"] = seg
            env["HPC_KW_LAG_SCOPE"] = "intra"
            env["HPC_KW_ESTIMATOR"] = est
            env["HPC_KW_EXOG_BUCKET"] = BUCKET
            env["HPC_RESULT_DIR"] = str(out)
            log = out / "run.log"
            print(f"start {seg}/{est} -> {out}", flush=True)
            procs.append(
                (
                    f"{seg}/{est}",
                    subprocess.Popen(
                        [py, str(SPEC)],
                        cwd=str(ROOT),
                        env=env,
                        stdout=open(log, "w", encoding="utf-8"),
                        stderr=subprocess.STDOUT,
                    ),
                )
            )
    rc = 0
    for name, p in procs:
        p.wait()
        print(f"done {name} rc={p.returncode}", flush=True)
        rc = rc or p.returncode
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
