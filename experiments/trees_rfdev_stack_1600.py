"""Stack the dev-tuned random forest's 16:00 arms under NEW keys (user decision 2026-10-06).

The campaign's untuned builder (experiments/build_subsection_tree_yhat.py, unedited: its gates -- profile B
against the production table to 1e-9, every stamp in it, the production rv_raw carried) is run with its module
lists narrowed to the 16:00 bar, the random forest and the three input sets of this run, on the rfdev roots;
the tables are written to a staging folder and copied under keys that no existing table has, so no default
random-forest table changes (the main session decides integration):

  rung  root                                                    -> results/spxw_pnl/
  T10   results/linear_subsection_trees_mask_rfdev/t10          yhat_subtree_rfdev_<bucket>.parquet
  T1    results/linear_subsection_trees_mask_rfdev/t1           yhat_subtree_daily_<bucket>_rfdev.parquet

Staging: results/trees_rfdev_1600/stack/<rung>/.  Written only once results/close_master_table/PLAIN_DONE exists.

Run:  python experiments/trees_rfdev_stack_1600.py [--out results/spxw_pnl]
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

import build_subsection_tree_yhat as bst  # noqa: E402

BUCKETS = ("baseline", "all_features", "lean_vol_vix")
U = ROOT / "results" / "linear_subsection_trees_mask_rfdev"
STAGE = ROOT / "results" / "trees_rfdev_1600" / "stack"
PLAIN_DONE = ROOT / "results" / "close_master_table" / "PLAIN_DONE"
NAMES = {
    "t10": "yhat_subtree_rfdev_{b}.parquet",
    "t1": "yhat_subtree_daily_{b}_rfdev.parquet",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "spxw_pnl"))
    a = ap.parse_args()
    out = Path(a.out)
    if (
        out.resolve() == (ROOT / "results" / "spxw_pnl").resolve()
        and not PLAIN_DONE.is_file()
    ):
        print(f"{PLAIN_DONE} missing")
        return 1
    bst.BARS = ("bar1600",)  # type: ignore[assignment]
    bst.MODELS = ("rf",)  # type: ignore[assignment]
    bst.BUCKETS = BUCKETS  # type: ignore[assignment]
    fails: list[str] = []
    for rung, pat in NAMES.items():
        st = STAGE / rung
        st.mkdir(parents=True, exist_ok=True)
        saved = sys.argv
        try:
            sys.argv = [bst.__file__, "--root", str(U / rung), "--out", str(st)]
            bst.main()
        except SystemExit as e:
            if e.code not in (None, 0):
                fails.append(f"{rung}: {e.code}")
        finally:
            sys.argv = saved
        for b in BUCKETS:
            src = st / f"yhat_subtree_rf_{b}.parquet"
            dst = out / pat.format(b=b)
            if not src.is_file():
                fails.append(f"missing {src.name} ({rung})")
                continue
            shutil.copyfile(src, dst)
            print(f"  {src.relative_to(ROOT)} -> {dst.name}")
    print(
        f"{len(NAMES) * len(BUCKETS) - len([f for f in fails if f.startswith('missing')])} tables written to {out}; problems: {fails or 'none'}"
    )
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
