"""Merge chunked worker outputs -> a per-coordinate pass-rate surface.

The HPC worker writes per-path rows to ``*.parquet`` chunks (one array task -> one file), with columns
``path_id, coord_id, outcome, stop_step, terminal_equity, peak_equity, max_dd, days`` plus one column
per coord key. This reduces those rows to one tidy row per coordinate: the pass/fail/timeout rates and a
Wilson 95% CI on the pass-rate (the Monte Carlo sampling error on a binomial proportion). pandas/pyarrow
are imported lazily so importing this module (and ``grid``) never requires the HPC-layer deps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eval_sim.grid import COORD_KEYS

if TYPE_CHECKING:
    import pandas as pd

_OUTPUT_NAME = "aggregate.parquet"


def aggregate(results_dir: str) -> pd.DataFrame:
    """Concat every chunk in ``results_dir`` and reduce to one row per coordinate, sorted by pass-rate.

    Skips a prior ``aggregate.parquet`` so re-running is idempotent. Raises ``FileNotFoundError`` if no
    input chunks are present.
    """
    import glob
    import os

    import numpy as np
    import pandas as pd

    from eval_sim.metrics import wilson_ci
    from eval_sim.sim_core import FAIL, PASS, TIMEOUT

    paths = sorted(
        p
        for p in glob.glob(os.path.join(results_dir, "*.parquet"))
        if os.path.basename(p) != _OUTPUT_NAME
    )
    if not paths:
        raise FileNotFoundError(f"no chunk *.parquet files in {results_dir!r}")

    df = pd.concat(
        (pd.read_parquet(p, engine="pyarrow") for p in paths), ignore_index=True
    )

    coord_cols = [k for k in COORD_KEYS if k in df.columns]
    group_cols = ["coord_id", *coord_cols]

    df = df.assign(
        _pass=(df["outcome"] == PASS).astype(float),
        _fail=(df["outcome"] == FAIL).astype(float),
        _timeout=(df["outcome"] == TIMEOUT).astype(float),
        # 1-based step count on PASS paths only (NaN elsewhere -> skipped by mean)
        _steps_pass=np.where(df["outcome"] == PASS, df["stop_step"] + 1.0, np.nan),
    )

    out = (
        df.groupby(group_cols, sort=False, dropna=False)
        .agg(
            n_paths=("outcome", "size"),
            n_pass=("_pass", "sum"),
            n_fail=("_fail", "sum"),
            n_timeout=("_timeout", "sum"),
            mean_steps_pass=("_steps_pass", "mean"),
        )
        .reset_index()
    )

    out["pass_rate"] = out["n_pass"] / out["n_paths"]
    out["fail_rate"] = out["n_fail"] / out["n_paths"]
    out["timeout_rate"] = out["n_timeout"] / out["n_paths"]
    ci = [
        wilson_ci(int(k), int(n))
        for k, n in zip(out["n_pass"], out["n_paths"], strict=True)
    ]
    out["wilson_lo"] = [lo for lo, _ in ci]
    out["wilson_hi"] = [hi for _, hi in ci]

    cols = [
        *group_cols,
        "n_paths",
        "pass_rate",
        "fail_rate",
        "timeout_rate",
        "wilson_lo",
        "wilson_hi",
        "mean_steps_pass",
    ]
    return out[cols].sort_values("pass_rate", ascending=False).reset_index(drop=True)


def main() -> None:
    """CLI: aggregate a results dir, print the table, and write ``aggregate.parquet`` beside the chunks."""
    import argparse
    import os

    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", required=True, help="directory of chunk *.parquet files"
    )
    args = parser.parse_args()

    table = aggregate(args.results_dir)
    with pd.option_context("display.max_rows", None, "display.width", None):
        print(table.to_string(index=False))

    out_path = os.path.join(args.results_dir, _OUTPUT_NAME)
    table.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
