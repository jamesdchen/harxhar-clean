"""Model module: one full-OOS residualized (Ridge->tree) trial on the amortized slim cache.

``run.py`` dispatches here (``model: resid_tree``). The expensive Ridge base is precomputed
ONCE per cell (``resid_amortized`` prep), so a trial only fits trees on ``y - ridge`` at the
refit cadence. Reads the asked tree config from ``params_file`` and writes ``results.csv``
(``true_raw``/``pred_raw``) next to ``output_file`` so the campaign's ``_compute_qlike`` /
``_tell_from_results`` close the TPE loop on the FULL-OOS residualized QLIKE.
"""
from __future__ import annotations

import json
import os
import sys


def run(cid: str, params_file: str, output_file: str, arm: str = "residualized",
        blk0=None, blk1=None, **_ignored) -> str:
    sys.path.insert(0, os.getcwd())
    import resid_amortized as ra

    cfg = json.load(open(params_file))
    # Under hpc-agent dispatch the config's output_file carries a literal
    # ``{run_id}`` placeholder (the run_id isn't known when tasks.py writes the
    # config); substitute it from HPC_RUN_ID so the written path matches the
    # framework's result_dir_template (and the canary writes under <run_id>-canary).
    _rid = os.environ.get("HPC_RUN_ID")
    if _rid and "{run_id}" in output_file:
        output_file = output_file.replace("{run_id}", _rid)
    d = os.path.dirname(output_file) or "."
    if blk0 is not None and blk1 is not None:  # one time-chunk (cadence-block range)
        return ra.run_chunk_to_csv(cid, arm, cfg, int(blk0), int(blk1), os.path.join(d, f"chunk_{int(blk0)}_{int(blk1)}.csv"))
    return ra.run_to_csv(cid, arm, cfg, os.path.join(d, "results.csv"))  # whole backtest
