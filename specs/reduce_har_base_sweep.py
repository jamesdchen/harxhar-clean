"""Run-level reducer for har_base_sweep (run har_base_sweep-53c27e42).

The framework's built-in no-combiner fallback is a JSON weighted mean and
cannot reduce this run's per-task CSV artifact — and a mean of per-chunk
metrics would be WRONG for the DM test anyway (DM needs the pooled per-bar
loss series). This reducer does the correct pooled reduce, mirroring the
audited baseline section of specs/har_base_sweep.py exactly:

per ladder-arm x rungs-cap cell: concatenate the 100 chunks' per-bar
executor results tables in chunk order (the chunked_series tiling is
disjoint and ordered, so concatenation IS the full-OOS series), likewise
the per-chunk OLS-HAR base-5 incumbent runs (each task runs its own
incumbent chunk in-process, so per-bar alignment is by construction), then
compute the beat4 spread (src.evaluation.metrics.forecast_metrics) and the
Diebold-Mariano arm-vs-incumbent test on per-bar QLIKE losses
(src.evaluation.diebold_mariano.qlike_per_bar + dm_test) ONCE per arm over
the pooled series. Every number is computed by those src functions; this
script only routes arrays. The b5_c3125 row carries ka_exact — pooled
byte-equality vs its incumbent (the per-task known-answer assert already
enforced this cluster-side; the column makes it visible in the table).

Registered as the run's aggregate_cmd (env-var I/O): reads
HPC_PER_TASK_RESULTS (the per-task mirror root) and HPC_AGGREGATE_OUT (the
output CSV path), with argv fallbacks for direct invocation.

Usage:
    python specs/reduce_har_base_sweep.py [per_task_results_root] [out_csv]
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.diebold_mariano import dm_test, qlike_per_bar  # noqa: E402
from src.evaluation.metrics import forecast_metrics  # noqa: E402

HORIZON = 1  # matches the audited source
_CHUNK_RE = re.compile(r"^chunk_(\d+)$")


def _label(arm: str, cap: str) -> str:
    """Arm dir '2_240' / 'explicit_960' -> results label 'b2_c240' / 'explicit_c960'."""
    return (f"b{arm}" if arm != "explicit" else "explicit") + f"_c{cap}"


def _chunk_dirs(arm_dir: Path) -> list[Path]:
    """Chunk task dirs under an arm, ordered by chunk_start (time order)."""
    out = []
    for d in arm_dir.iterdir():
        m = _CHUNK_RE.match(d.name)
        if d.is_dir() and m:
            out.append((int(m.group(1)), d))
    out.sort()
    return [d for _, d in out]


def _pooled_arm(root: Path, arm_dir_name: str):
    """Pooled (pred_raw, true_raw, incumbent_pred_raw, n_chunks) for one arm cell."""
    arm, cap = arm_dir_name.rsplit("_", 1)
    label = _label(arm, cap)
    arm_dir = root / arm_dir_name
    chunks = _chunk_dirs(arm_dir)
    if not chunks:
        raise FileNotFoundError(f"no chunk dirs under {arm_dir}")
    arm_parts: list[pd.DataFrame] = []
    inc_parts: list[pd.DataFrame] = []
    for chunk in chunks:
        arm_csv = chunk / "har_base_sweep" / label / "results.csv"
        inc_csv = chunk / "har_base_sweep" / "incumbent_base5" / "results.csv"
        arm_df = pd.read_csv(arm_csv)
        inc_df = pd.read_csv(inc_csv)
        if len(arm_df) != len(inc_df):
            raise ValueError(
                f"bar-count mismatch in {chunk.name} ({arm_dir_name}): "
                f"arm={len(arm_df)} incumbent={len(inc_df)}"
            )
        arm_parts.append(arm_df)
        inc_parts.append(inc_df)
    arm_all = pd.concat(arm_parts, ignore_index=True)
    inc_all = pd.concat(inc_parts, ignore_index=True)
    return (
        arm_all["pred_raw"].to_numpy(),
        arm_all["true_raw"].to_numpy(),
        inc_all["pred_raw"].to_numpy(),
        len(chunks),
    )


def main() -> int:
    root = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get(
            "HPC_PER_TASK_RESULTS",
            "_aggregated/har_base_sweep-53c27e42/_per_task_results",
        )
    )
    out_csv = Path(
        sys.argv[2]
        if len(sys.argv) > 2
        else os.environ.get(
            "HPC_AGGREGATE_OUT",
            "_aggregated/har_base_sweep-53c27e42/metrics_table.csv",
        )
    )
    arm_dir_names = sorted(d.name for d in root.iterdir() if d.is_dir())
    rows = []
    for arm_dir_name in arm_dir_names:
        arm, cap = arm_dir_name.rsplit("_", 1)
        label = _label(arm, cap)
        p_t, t_t, p_i, n_chunks = _pooled_arm(root, arm_dir_name)
        m = forecast_metrics(p_t, t_t, benchmark=p_i)
        dm = dm_test(qlike_per_bar(p_t, t_t), qlike_per_bar(p_i, t_t), h=HORIZON)
        rows.append(
            dict(
                arm=label,
                ladder_arm=arm,
                rungs_cap=int(cap),
                chunks=n_chunks,
                **m,
                incumbent_qlike=forecast_metrics(p_i, t_t)["qlike"],
                dm=dm["dm"],
                dm_p=dm["p"],
                dm_mean_diff=dm["mean_diff"],
                dm_better=dm.get("better", ""),
                ka_exact=bool(np.array_equal(p_t, p_i)) if label == "b5_c3125" else "",
            )
        )
    table = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    print(table.to_string(index=False))
    print(f"\nwrote {out_csv} ({len(table)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
