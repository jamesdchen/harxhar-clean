"""Stack the Optuna per-bar tree forecasts (checklist I3) into yhat tables the research scorers read.

The Optuna twin of ``experiments/build_subsection_tree_tuned_yhat.py``.  The campaign
(``specs/causal_tune_trees_optuna.py``, chunks merged by ``experiments/reduce_trees_optuna.py
--stage 2``) writes one arm per (path, bucket, model) in the untuned layout

  <root>/paths/<path>/<bucket>/bar1600/<model>/tw<TW>/causal_tune_trees/<model>/<bucket>/results_bar1600.csv

Each is written in the notebooks' format -- ``t`` (UTC), ``yhat`` (the adjusted-scale
forecast pred_adj), ``baseline`` (the profile B) and ``rv_raw`` -- with the linear stacker's
own ``read_arm`` and ``with_production_rv`` (same gates: B agrees with the production table
to 1e-9, every stamp is in it; the production rv_raw is carried).  Nothing of the spec's
look-ahead ``pred_raw`` is carried.  THE TABLES HOLD THE 16:00 BAR ONLY (the forecast issued
at 15:30): a scorer that recalibrates on all 13 session bars cannot use them; the research
scorer's 16:00-bar recalibration can.

Tables (``--out``, default results/spxw_pnl/), <bucket> in all_features / baseline /
live_feasible, <model> in lgbm / xgb / rf:
  yhat_subtree_optuna_tp<N>_<bucket>_<model>.parquet       N = 1, 5, 25, 250: best of 50 trials
                                                           (validation MSE, the rule of record)
  yhat_subtree_optuna_tp<N>_k<k>_<bucket>_<model>.parquet  best of the first k = 10 / 25 trials
                                                           at N = 1 and 25
  yhat_subtree_optunaq_tp25_<bucket>_<model>.parquet       best of 50 by validation QLIKE, N = 25

Extra gate: every table of an arm carries the SAME stamps, B and production rv_raw as the
arm's tp1 table (one design, one target); a difference fails the exit status.

Run:  python experiments/build_subsection_tree_optuna_yhat.py
          [--root results/linear_subsection_trees_optuna_mask] [--out results/spxw_pnl]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_subsection_tree_yhat as bst  # noqa: E402
import build_subsection_yhat as bsy  # noqa: E402

OPTUNA_ROOT = ROOT / "results" / "linear_subsection_trees_optuna_mask"
SEG = "bar1600"
# path of the campaign -> table stem (the campaign file's names)
TABLES = {
    "tp1": "subtree_optuna_tp1",
    "tp5": "subtree_optuna_tp5",
    "tp25": "subtree_optuna_tp25",
    "tp250": "subtree_optuna_tp250",
    "tp1_k10": "subtree_optuna_tp1_k10",
    "tp1_k25": "subtree_optuna_tp1_k25",
    "tp25_k10": "subtree_optuna_tp25_k10",
    "tp25_k25": "subtree_optuna_tp25_k25",
    "tp25_q": "subtree_optunaq_tp25",
}


def table_name(path: str, bucket: str, model: str) -> str:
    return f"yhat_{TABLES[path]}_{bucket}_{model}.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(OPTUNA_ROOT))
    ap.add_argument("--out", default=str(bsy.OUT))
    ap.add_argument("--tw", type=int, default=bsy.TW)
    a = ap.parse_args()
    root, out = bst.resolve(a.root), bst.resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    written, failed = [], []
    for model in bst.MODELS:
        for bucket in bst.BUCKETS:
            ref = None
            for path in TABLES:
                name = table_name(path, bucket, model)
                p = bst.tree_arm_path(root / "paths" / path, bucket, SEG, model, a.tw)
                if p is None:
                    print(f"skip {name}: no merged arm under {root / 'paths' / path}")
                    continue
                try:
                    tab = bsy.read_arm(p).sort_values("t").reset_index(drop=True)
                    assert not tab["t"].duplicated().any(), name
                    tab = bsy.with_production_rv(tab, name)
                except AssertionError as e:
                    print(f"GATE FAIL {name}: {e!r} -- not written")
                    failed.append(name)
                    continue
                if ref is None:
                    ref = tab
                else:
                    same = len(tab) == len(ref) and bool(
                        (tab["t"].to_numpy() == ref["t"].to_numpy()).all()
                    )
                    same = same and np.array_equal(
                        tab["rv_raw"].to_numpy(), ref["rv_raw"].to_numpy()
                    )
                    db = (
                        float(
                            np.max(
                                np.abs(
                                    tab["baseline"].to_numpy()
                                    / ref["baseline"].to_numpy()
                                    - 1.0
                                )
                            )
                        )
                        if same
                        else np.inf
                    )
                    if not (same and db < bsy.GATE_REL):
                        print(
                            f"GATE FAIL {name}: stamps / rv_raw / B differ from the arm's first table (B {db:.1e})"
                        )
                        failed.append(name)
                        continue
                tab.to_parquet(out / name, index=False)
                et = tab["t"].dt.tz_convert("America/New_York")
                print(
                    f"wrote {name}: {len(tab):,} rows (bar ends {sorted(set(et.dt.strftime('%H:%M')))}), "
                    f"{et.min().date()} .. {et.max().date()}"
                )
                written.append(name)
    print(f"{len(written)} tables written to {out}")
    if failed:
        sys.exit(f"{len(failed)} table(s) failed a gate: {', '.join(failed)}")


if __name__ == "__main__":
    main()
