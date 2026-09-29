"""Assemble the causally TUNED per-bar tree forecasts into yhat tables (checklist A2b).

The tuned twin of ``experiments/build_subsection_tree_yhat.py`` (which is left
unchanged).  The tuned campaign (``specs/causal_tune_trees_tuned.py``, chunks
merged by ``experiments/reduce_trees_tuned_chunks.py``) writes one arm per
(bucket, bar, model) in the untuned layout

  <root>/<bucket>/<bar>/<model>/tw<TW>/causal_tune_trees/<model>/<bucket>/
      results_<bar>.csv        MSE-selected configuration (the arm of record)
      results_qsel_<bar>.csv   QLIKE-selected configuration (second rule)

Thirteen one-bar arms (10:00 .. 16:00) stacked are a forecast table in the
notebooks' format -- ``t`` (UTC), ``yhat`` (the adjusted-scale forecast
pred_adj), ``baseline`` (the profile B) and ``rv_raw`` -- built with the linear
stacker's own ``read_arm`` and ``with_production_rv`` exactly as the untuned
tables are (same gates: B agrees with the production table to 1e-9, every stamp
is in it; the production rv_raw is carried, not the spec's winsorized target).
Nothing of the spec's look-ahead ``pred_raw`` is carried: the reader applies its
own causal recalibration.

Tables (``--out``, default results/spxw_pnl/):
  yhat_subtree_tuned_<bucket>_<model>.parquet    MSE rule (the arm of record)
  yhat_subtree_tunedq_<bucket>_<model>.parquet   QLIKE rule
for <model> in lgbm / xgb / rf and <bucket> in all_features / baseline /
live_feasible.  Only sets whose 13 bar arms all exist are written.

Extra gate: every tuned table has the stamps and production rv_raw of the untuned
table of the same model and bucket (``--untuned-dir``, the
yhat_subtree_<model>_<bucket> parquets) exactly, and its profile B to 1e-9
relative (B is backed out of each run's CSV text), because both campaigns fit the
same design; a difference is printed and fails the exit status.

Run:  python experiments/build_subsection_tree_tuned_yhat.py
          [--root results/linear_subsection_trees_tuned] [--out results/spxw_pnl]
          [--untuned-dir results/spxw_pnl]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_subsection_tree_yhat as bst  # noqa: E402
import build_subsection_yhat as bsy  # noqa: E402

TUNED = ROOT / "results" / "linear_subsection_trees_tuned"
RULES = {
    "mse": ("results", "tuned"),
    "qlike": ("results_qsel", "tunedq"),
}  # file stem, table tag


def arm_file(
    root: Path, bucket: str, seg: str, model: str, tw: int, stem: str
) -> Path | None:
    """The rule's CSV next to the arm's results CSV (cluster or flattened layout)."""
    p = bst.tree_arm_path(root, bucket, seg, model, tw)
    if p is None:
        return None
    q = p.with_name(f"{stem}_{seg}.csv")
    return q if q.is_file() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(TUNED), help="tuned tree results root")
    ap.add_argument("--out", default=str(bsy.OUT), help="where the parquet tables go")
    ap.add_argument(
        "--untuned-dir",
        default=str(bsy.OUT),
        help="where yhat_subtree_<model>_<bucket> live",
    )
    ap.add_argument("--tw", type=int, default=bst.TW, help="training window (sessions)")
    a = ap.parse_args()
    root, out, udir = (
        bst.resolve(a.root),
        bst.resolve(a.out),
        bst.resolve(a.untuned_dir),
    )
    if not root.is_dir():
        print(f"no tuned tree results root at {root}; nothing to stack")
        return
    out.mkdir(parents=True, exist_ok=True)
    written, failed = [], []
    for model in bst.MODELS:
        for bucket in bst.BUCKETS:
            for rule, (stem, tag) in RULES.items():
                name = f"yhat_subtree_{tag}_{bucket}_{model}.parquet"
                paths = {
                    b: arm_file(root, bucket, b, model, a.tw, stem) for b in bst.BARS
                }
                missing = [b for b, p in paths.items() if p is None]
                if missing:
                    print(
                        f"skip {name}: {len(missing)} of {len(bst.BARS)} bar arms missing ({', '.join(missing)})"
                    )
                    continue
                try:
                    tab = (
                        pd.concat(
                            [bsy.read_arm(p) for p in paths.values() if p is not None]
                        )
                        .sort_values("t")
                        .reset_index(drop=True)
                    )
                    assert not tab["t"].duplicated().any(), name
                    tab = bsy.with_production_rv(tab, name)
                except AssertionError as e:  # a gate: loud, and the exit status says so
                    print(f"GATE FAIL {name}: {e!r} -- not written")
                    failed.append(name)
                    continue
                except Exception as e:  # noqa: BLE001 -- one unreadable set must not stop the others
                    print(f"skip {name}: {type(e).__name__}: {e}")
                    continue
                twin = udir / f"yhat_subtree_{model}_{bucket}.parquet"
                if twin.is_file():
                    u = pd.read_parquet(twin, columns=["t", "baseline", "rv_raw"])
                    same_t = len(u) == len(tab) and bool(
                        (u["t"].to_numpy() == tab["t"].to_numpy()).all()
                    )
                    # B is backed out of each run's own CSV text (true_raw / true_adj^2), so the
                    # two campaigns agree to float round-off, not bit for bit: the linear
                    # stacker's 1e-9 relative bound
                    db = (
                        float(
                            (
                                tab["baseline"].to_numpy() / u["baseline"].to_numpy()
                                - 1.0
                            )
                            .__abs__()
                            .max()
                        )
                        if same_t
                        else float("inf")
                    )
                    same_b = db < bsy.GATE_REL
                    same_rv = same_t and bool(
                        (u["rv_raw"].to_numpy() == tab["rv_raw"].to_numpy()).all()
                    )
                    print(
                        f"GATE  {name} vs {twin.name}: stamps identical {same_t} ({len(tab):,} vs {len(u):,}), "
                        f"baseline max rel diff {db:.1e}, production rv_raw identical {same_rv}"
                    )
                    if not (same_t and same_b and same_rv):
                        failed.append(name)
                else:
                    print(f"note: no untuned twin {twin.name}; stamp gate not run")
                tab.to_parquet(out / name, index=False)
                print(
                    f"wrote {name}: {len(tab):,} rows, {tab['t'].dt.normalize().nunique():,} sessions, "
                    f"{tab['t'].min().date()} .. {tab['t'].max().date()}"
                )
                written.append(name)
    print(f"{len(written)} tables written to {out}")
    if failed:
        sys.exit(f"{len(failed)} set(s) failed a gate: {', '.join(failed)}")


if __name__ == "__main__":
    main()
