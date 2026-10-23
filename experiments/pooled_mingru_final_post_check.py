"""POST step of a GPU job of the final pooled MinGRU (cluster/pooled_mingru_final_gpu_carc.sbatch POST=...): the
runner's check-run on the job's run directory (status ok, the trainer's kept columns = the ridge init's, forecast rows
exactly [origin, next origin), finite positive forecasts); a failure fails the job, so jobs held afterok on it do not
start.  The window's root is the run directory's grandparent (<root>/runs/<run>; /home1/jc_905/pmf for W = 3,000,
/home1/jc_905/pmf35 for W = 3,500).

    python pooled_mingru_final_post_check.py <run dir>
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

RUN = Path(sys.argv[1]).resolve()
ROOT = RUN.parents[1]
r = subprocess.run(
    [
        sys.executable,
        str(ROOT / "code" / "pooled_mingru_final_2014_w3000.py"),
        "check-run",
        "--inputs",
        str(ROOT / "inputs"),
        "--run",
        str(RUN),
    ],
    check=False,
)
sys.exit(r.returncode)
