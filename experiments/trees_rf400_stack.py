"""Stack the rf400 forest's 16:00 arms (the 400-trial dev-span search's final entry, trial 255) into
results/trees_rf400_1600/ (2026-10-07; NOT results/spxw_pnl/), each table written atomically (temp file + os.replace), with
the campaign's builders and their gates (experiments/build_subsection_tree_yhat.py; the tuned twin
build_subsection_tree_tuned_yhat.py, whose stamp gate compares each tuned table with the fixed-settings table of the same
cadence).  Columns as every forecast table: t, yhat, baseline, rv_raw.

  part fx      results/linear_subsection_trees_mask_<tok>/{t1,t10}     yhat_subtree_daily_<set>_<tok> / yhat_subtree_<tok>_<set>
  part nm      results/linear_subsection_trees_nomask_<tok>/{t1,t10}   the same names in the subfolder nomask/ (not picked up
               by the GUI; the layout of results/trees_xgbtop_1600/nomask/)
  part rs      results/linear_subsection_trees_tuned_mask_<tok>/{rs1,rs10}
               yhat_subtree_tuned[q]_daily_<set>_<tok> (RS1) / yhat_subtree_tuned[q]_<set>_<tok> (RS10)
  part optuna  results/linear_subsection_trees_optuna_mask_<tok>      yhat_subtree_optuna[q]_tp<N>[_k<K>]_<set>_<tok>
               (experiments/trees_rfp163_stack_optuna.py, unchanged, staged in stack/optuna/ and moved here)

The table names are the master table's keys with the estimator token replaced (the GUI swaps them in by name).

Run:  python experiments/trees_rf400_stack.py --part fx [--rungs t1] [--token rf400]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_subsection_tree_tuned_yhat as bstt  # noqa: E402
import build_subsection_tree_yhat as bst  # noqa: E402

BUCKETS = ("baseline", "all_features", "lean_vol_vix")


def put(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(
        tmp, dst
    )  # atomic on the same filesystem: a reader sees the old or the new table, never a partial one
    print(f"  {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def run(module, argv: list[str]) -> str | None:
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
    ap.add_argument("--part", required=True, choices=("fx", "nm", "rs", "optuna"))
    ap.add_argument(
        "--rungs", default="", help="comma list (t1,t10 or rs1,rs10); default all"
    )
    ap.add_argument("--token", default="rf400")
    ap.add_argument("--out", default=str(ROOT / "results" / "trees_rf400_1600"))
    a = ap.parse_args()
    tok, out = a.token, Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stage = out / "stack"
    bst.BARS = ("bar1600",)  # type: ignore[assignment]
    bst.MODELS = ("rf",)  # type: ignore[assignment]
    bst.BUCKETS = BUCKETS  # type: ignore[assignment]
    probs: list[str] = []
    if a.part in ("fx", "nm"):
        root = (
            ROOT
            / "results"
            / (
                f"linear_subsection_trees_mask_{tok}"
                if a.part == "fx"
                else f"linear_subsection_trees_nomask_{tok}"
            )
        )
        dst = out if a.part == "fx" else out / "nomask"
        dst.mkdir(parents=True, exist_ok=True)
        for rung in [
            r for r in ("t1", "t10") if not a.rungs or r in a.rungs.split(",")
        ]:
            st = stage / f"{a.part}_{rung}"
            st.mkdir(parents=True, exist_ok=True)
            e = run(bst, ["--root", str(root / rung), "--out", str(st)])
            if e:
                probs.append(f"{rung}: {e}")
            for b in BUCKETS:
                src = st / f"yhat_subtree_rf_{b}.parquet"
                if not src.is_file():
                    probs.append(f"missing {src.name} ({rung})")
                    continue
                name = (
                    f"yhat_subtree_daily_{b}_{tok}.parquet"
                    if rung == "t1"
                    else f"yhat_subtree_{tok}_{b}.parquet"
                )
                put(src, dst / name)
    elif a.part == "optuna":
        st = stage / "optuna"
        st.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "experiments" / "trees_rfp163_stack_optuna.py"),
            "--token",
            tok,
            "--root",
            str(ROOT / "results" / f"linear_subsection_trees_optuna_mask_{tok}"),
            "--out",
            str(st),
        ]
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode:
            probs.append(f"optuna stacker exit {r.returncode}")
        got = sorted(st.glob(f"yhat_subtree_optuna*_{tok}.parquet"))
        if len(got) != 27:
            probs.append(f"{len(got)} Optuna tables, expected 27 (9 paths x 3 sets)")
        if not probs:  # move only a complete, gated set
            for src in got:
                put(src, out / src.name)
    else:
        root = ROOT / "results" / f"linear_subsection_trees_tuned_mask_{tok}"
        for rung, twin in [
            (r, t)
            for r, t in (("rs1", "t1"), ("rs10", "t10"))
            if not a.rungs or r in a.rungs.split(",")
        ]:
            st = stage / f"rs_{rung}"
            st.mkdir(parents=True, exist_ok=True)
            e = run(
                bstt,
                [
                    "--root",
                    str(root / rung),
                    "--out",
                    str(st),
                    "--untuned-dir",
                    str(stage / f"fx_{twin}"),
                ],
            )
            if e:
                probs.append(f"{rung}: {e}")
            for b in BUCKETS:
                for rule in ("tuned", "tunedq"):
                    src = st / f"yhat_subtree_{rule}_{b}_rf.parquet"
                    if not src.is_file():
                        probs.append(f"missing {src.name} ({rung})")
                        continue
                    name = (
                        f"yhat_subtree_{rule}_daily_{b}_{tok}.parquet"
                        if rung == "rs1"
                        else f"yhat_subtree_{rule}_{b}_{tok}.parquet"
                    )
                    put(src, out / name)
    print(f"problems: {probs or 'none'}")
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
