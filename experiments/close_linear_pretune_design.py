"""Design cache for experiments/close_linear_pretune.py (the 2026-10-06 linear pre-tune study).

Runs experiments/capture_design_close.py's capture() unchanged (imported read-only), with two
differences that do not change the design:

* data_path: a directory of hard links to every bar-keyed parquet of data/ (all but
  spxw_chain.parquet and spxw_spot.parquet).  src/data/loading.load_raw_data reads every
  parquet of the directory and then SKIPS the ones without an ``endbartime`` key; those two are
  the OptionMetrics chain / spot exports, which it skips, so the panel is the same.  The chain
  (80 MB) is never read.
* CLOSE_DESIGN_DIR: results/linear_pretune_2026-10-06/_work/design (gitignored by the study
  folder's own .gitignore), so the shared cache results/close_design/_work is not touched.

Usage: python experiments/close_linear_pretune_design.py <segment> <bucket> [<bucket> ...]
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STUDY = REPO / "results" / "linear_pretune_2026-10-06"
WORK = STUDY / "_work"
DESIGN_DIR = WORK / "design"
DATA_DIR = WORK / "data_nochain"
SKIP = (
    "spxw_chain.parquet",
    "spxw_spot.parquet",
)  # non-bar-keyed exports load_raw_data skips


def data_dir() -> Path:
    """Hard links (copies if linking fails) to data/*.parquet except SKIP."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for f in sorted((REPO / "data").glob("*.parquet")):
        if f.name in SKIP:
            continue
        dst = DATA_DIR / f.name
        if dst.is_file() and dst.stat().st_size == f.stat().st_size:
            continue
        if dst.exists():
            dst.unlink()
        try:
            os.link(f, dst)
        except OSError:
            shutil.copy2(f, dst)
    return DATA_DIR


def main(segment: str, buckets: list[str]) -> None:
    os.environ["CLOSE_DESIGN_DIR"] = str(DESIGN_DIR)
    src = str(data_dir())
    sys.path.insert(0, str(REPO / "experiments"))
    import capture_design_close as cdc  # noqa: E402  (reads CLOSE_DESIGN_DIR at import)

    assert cdc.OUT == DESIGN_DIR, cdc.OUT
    run = cdc.ex.run_executor

    def run_nochain(*args, **kw):
        assert kw.get("data_path") == "data", kw.get("data_path")
        kw["data_path"] = src
        return run(*args, **kw)

    cdc.ex.run_executor = run_nochain
    try:
        for b in buckets:
            cdc.capture(segment, b)
    finally:
        cdc.ex.run_executor = run


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2:])
