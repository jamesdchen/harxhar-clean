"""Run-level reducer for causal_tune_linear (run causal_tune_linear-de448128).

The framework's built-in no-combiner fallback is a JSON weighted mean and
cannot reduce this run's per-task CSV artifact — and a mean of per-chunk
metrics would be WRONG for the DM test anyway (DM needs the pooled per-bar
loss series). This reducer does the correct pooled reduce, mirroring the
audited baseline section of specs/causal_tune_linear.py exactly:

per estimator x bucket arm: concatenate the 100 chunks' per-bar executor
results tables in chunk order (the chunked_series tiling is disjoint and
ordered, so concatenation IS the full-OOS series), likewise the per-chunk
OLS no-bucket incumbent runs, then compute the beat4 spread
(src.evaluation.metrics.forecast_metrics) and the Diebold-Mariano
tuned-vs-incumbent test on per-bar QLIKE losses
(src.evaluation.diebold_mariano.qlike_per_bar + dm_test) ONCE per arm over
the pooled series. Every number is computed by those src functions; this
script only routes arrays.

Registered as the run's aggregate_cmd (env-var I/O): reads
HPC_PER_TASK_RESULTS (the per-task mirror root) and HPC_AGGREGATE_OUT (the
output CSV path), with argv fallbacks for direct invocation.

Usage:
    python specs/reduce_causal_tune_linear.py [per_task_results_root] [out_csv]
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.diebold_mariano import dm_test, qlike_per_bar  # noqa: E402
from src.evaluation.metrics import forecast_metrics  # noqa: E402

ESTIMATORS = ("ridge", "reclasso", "reclasticnet")
HORIZON = 1  # matches the audited source
_CHUNK_RE = re.compile(r"^chunk_(\d+)$")


def _chunk_dirs(arm_dir: Path) -> list[Path]:
    """Chunk task dirs under an arm, ordered by chunk_start (time order)."""
    out = []
    for d in arm_dir.iterdir():
        m = _CHUNK_RE.match(d.name)
        if d.is_dir() and m:
            out.append((int(m.group(1)), d))
    out.sort()
    return [d for _, d in out]


def _pooled_arm(root: Path, estimator: str, bucket: str):
    """Pooled (pred_raw, true_raw, incumbent_pred_raw) for one arm."""
    arm_dir = root / estimator / bucket
    chunks = _chunk_dirs(arm_dir)
    if not chunks:
        raise FileNotFoundError(f"no chunk dirs under {arm_dir}")
    tuned_parts: list[pd.DataFrame] = []
    inc_parts: list[pd.DataFrame] = []
    for chunk in chunks:
        tuned_csv = chunk / "causal_tune_linear" / estimator / bucket / "results.csv"
        inc_csv = chunk / "causal_tune_linear" / "incumbent_ols" / "results.csv"
        tuned = pd.read_csv(tuned_csv)
        inc = pd.read_csv(inc_csv)
        if len(tuned) != len(inc):
            raise ValueError(
                f"bar-count mismatch in {chunk.name} ({estimator}/{bucket}): "
                f"tuned={len(tuned)} incumbent={len(inc)}"
            )
        tuned_parts.append(tuned)
        inc_parts.append(inc)
    tuned_all = pd.concat(tuned_parts, ignore_index=True)
    inc_all = pd.concat(inc_parts, ignore_index=True)
    return (
        tuned_all["pred_raw"].to_numpy(),
        tuned_all["true_raw"].to_numpy(),
        inc_all["pred_raw"].to_numpy(),
    )


def main() -> int:
    root = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get(
            "HPC_PER_TASK_RESULTS",
            "_aggregated/causal_tune_linear-de448128/_per_task_results",
        )
    )
    out_csv = Path(
        sys.argv[2]
        if len(sys.argv) > 2
        else os.environ.get(
            "HPC_AGGREGATE_OUT",
            "_aggregated/causal_tune_linear-de448128/metrics_table.csv",
        )
    )
    buckets = sorted(d.name for d in (root / ESTIMATORS[0]).iterdir() if d.is_dir())
    rows = []
    for estimator in ESTIMATORS:
        for bucket in buckets:
            p_t, t_t, p_i = _pooled_arm(root, estimator, bucket)
            m = forecast_metrics(p_t, t_t, benchmark=p_i)
            dm = dm_test(qlike_per_bar(p_t, t_t), qlike_per_bar(p_i, t_t), h=HORIZON)
            rows.append(
                dict(
                    estimator=estimator,
                    bucket=bucket,
                    **m,
                    incumbent_qlike=forecast_metrics(p_i, t_t)["qlike"],
                    dm=dm["dm"],
                    dm_p=dm["p"],
                    dm_mean_diff=dm["mean_diff"],
                    dm_better=dm.get("better", ""),
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
