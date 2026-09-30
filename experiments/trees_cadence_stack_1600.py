"""Stack the 16:00 tree cadence rungs (checklist I2) into forecast tables, with the EXISTING builders.

The cadence campaign fitted only the 15:30-16:00 bar (campaign scope,
writeup/CAMPAIGN_16H_2026-09-29.md), so the builders
experiments/build_subsection_tree_yhat.py (untuned) and
experiments/build_subsection_tree_tuned_yhat.py (tuned: MSE and QLIKE rule) are run
unchanged with their bar list narrowed to bar1600 (their module constant BARS, which
both read at call time); every gate of theirs applies (profile B against the production
table to 1e-9, every stamp in it, the production rv_raw carried; the tuned tables'
stamps / B / rv_raw against the untuned table of the same model and bucket -- here the
untuned table of the SAME refit cadence).  The tables are 16:00-only, as the LSTM tables.

  rung  root                                              -> results/spxw_pnl/
  T10   results/linear_subsection_trees_dedup/t10         yhat_subtree_<model>_<bucket>          (the canonical name, overwritten)
  T1    results/linear_subsection_trees_dedup/t1          yhat_subtree_daily_<bucket>_<model>
  RS10  results/linear_subsection_trees_tuned_dedup/rs10  yhat_subtree_tuned[q]_<bucket>_<model> (overwritten)
  RS1   results/linear_subsection_trees_tuned_dedup/rs1   yhat_subtree_tuned[q]_daily_<bucket>_<model>

The builders write into a staging directory per rung (results/trees_cadence_1600/stack/),
the tables are then copied under the names above.

Run:  python experiments/trees_cadence_stack_1600.py [--out results/spxw_pnl]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_subsection_tree_tuned_yhat as bstt  # noqa: E402
import build_subsection_tree_yhat as bst  # noqa: E402

U = ROOT / "results" / "linear_subsection_trees_dedup"
T = ROOT / "results" / "linear_subsection_trees_tuned_dedup"
STAGE = ROOT / "results" / "trees_cadence_1600" / "stack"
BAR = "bar1600"


def run_builder(module, argv: list[str]) -> str | None:
    """module.main() with argv; its exit (a failed gate) is caught and returned."""
    saved = sys.argv
    try:
        sys.argv = [module.__file__, *argv]
        module.main()
    except SystemExit as e:
        if e.code not in (None, 0):
            return f"{Path(module.__file__).name}: {e.code}"
    finally:
        sys.argv = saved
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "spxw_pnl"))
    a = ap.parse_args()
    out = Path(a.out)
    bst.BARS = (BAR,)  # type: ignore[assignment]  # the campaign fitted the 16:00 bar only
    fails: list[str] = []
    copied: list[str] = []

    def copy(src: Path, name: str) -> None:
        if not src.is_file():
            print(f"not built: {src.name} (skipped by the builder)")
            fails.append(f"missing {src.name}")
            return
        shutil.copyfile(src, out / name)
        copied.append(name)
        print(f"  {src.relative_to(ROOT)} -> {name}")

    for rung, tag in (("t10", None), ("t1", "daily")):
        st = STAGE / rung
        st.mkdir(parents=True, exist_ok=True)
        print(f"== {rung}: build_subsection_tree_yhat on {U / rung}")
        f = run_builder(bst, ["--root", str(U / rung), "--out", str(st)])
        if f:
            fails.append(f"{rung} {f}")
        for m in bst.MODELS:
            for b in bst.BUCKETS:
                src = st / f"yhat_subtree_{m}_{b}.parquet"
                copy(
                    src,
                    src.name if tag is None else f"yhat_subtree_{tag}_{b}_{m}.parquet",
                )
    for rung, twin, tag in (("rs10", "t10", None), ("rs1", "t1", "daily")):
        st = STAGE / rung
        st.mkdir(parents=True, exist_ok=True)
        print(
            f"== {rung}: build_subsection_tree_tuned_yhat on {T / rung} (stamp gate vs the {twin} tables)"
        )
        f = run_builder(
            bstt,
            [
                "--root",
                str(T / rung),
                "--out",
                str(st),
                "--untuned-dir",
                str(STAGE / twin),
            ],
        )
        if f:
            fails.append(f"{rung} {f}")
        for m in bst.MODELS:
            for b in bst.BUCKETS:
                for rule in ("tuned", "tunedq"):
                    src = st / f"yhat_subtree_{rule}_{b}_{m}.parquet"
                    copy(
                        src,
                        src.name
                        if tag is None
                        else f"yhat_subtree_{rule}_{tag}_{b}_{m}.parquet",
                    )
    print(
        f"{len(copied)} tables written to {out}; {len(fails)} problems"
        + (f": {fails}" if fails else "")
    )
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
