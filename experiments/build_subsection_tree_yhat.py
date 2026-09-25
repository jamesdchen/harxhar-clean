"""Assemble the per-bar TREE forecasts into yhat tables the 0DTE notebooks read.

The tree twin of ``experiments/build_subsection_yhat.py``.  The per-bar tree
campaign (``specs/causal_tune_trees.py``: LightGBM / XGBoost / random forest on
exactly the per-bar linear design, SEGMENT = one regular-hours bar, TRAIN_WIN =
2000 sessions) writes one arm per (bucket, bar, model):

  <root>/<bucket>/<bar>/<model>/tw<TW>/causal_tune_trees/<model>/<bucket>/results_<bar>.csv

(the cluster pack's layout; a flattened pull ``<root>/<bucket>/<model>/tw<TW>/
results_<bar>.csv`` is read as well).  Thirteen one-bar arms (10:00 .. 16:00)
stacked are a forecast table in the notebooks' format -- ``t`` (UTC), ``yhat``
(the adjusted-scale forecast pred_adj), ``baseline`` (the profile B) and
``rv_raw`` -- built with the linear stacker's own ``read_arm`` and
``with_production_rv`` (same gates: B agrees with the production table to 1e-9,
every stamp is in it; the production rv_raw is carried, not the spec's
winsorized target).  Nothing of the spec's look-ahead ``pred_raw`` is carried:
the notebooks apply their own causal recalibration.

Tables (``--out``, default results/spxw_pnl/):
  yhat_subtree_<model>_<bucket>.parquet   for <model> in lgbm / xgb / rf and
                                          <bucket> in all_features / baseline /
                                          live_feasible
Only sets whose 13 bar arms all exist are written; the others print a skip line.

The arm discovery here (``tree_arm_path``, ``discover_tree_arms``) is shared with
``experiments/score_trees_subsection.py``.

Run:  python experiments/build_subsection_tree_yhat.py
          [--root results/linear_subsection/trees_carc] [--out results/spxw_pnl]
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_subsection_yhat as bsy  # noqa: E402

TREES = ROOT / "results" / "linear_subsection" / "trees_carc"
MODELS = ("lgbm", "xgb", "rf")
BUCKETS = bsy.BUCKETS
BARS = bsy.BARS
TW = bsy.TW
SPEC_DIR = "causal_tune_trees"
_RESULTS = re.compile(r"^results_(bar\d{4})\.csv$")


@dataclass(frozen=True)
class TreeArm:
    bucket: str
    seg: str
    model: str
    tw: int
    csv: Path

    @property
    def npz(self) -> Path:
        return self.csv.with_name(f"trees_{self.seg}.npz")

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.bucket, self.seg, self.model, self.tw)


def resolve(p: str | Path) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def tree_arm_path(
    root: Path, bucket: str, seg: str, model: str, tw: int
) -> Path | None:
    """The arm's results CSV in the cluster layout, else the flattened one, else None."""
    nested = (
        root
        / bucket
        / seg
        / model
        / f"tw{tw}"
        / SPEC_DIR
        / model
        / bucket
        / f"results_{seg}.csv"
    )
    flat = root / bucket / model / f"tw{tw}" / f"results_{seg}.csv"
    for p in (nested, flat):
        if p.is_file():
            return p
    return None


def discover_tree_arms(root: Path, tw: int | None = None) -> list[TreeArm]:
    """Every per-bar tree arm under ``root`` (both layouts; the cluster layout wins a tie).

    The flattened layout carries no spec marker, so it is only read for the tree
    models in MODELS -- a linear root passed by mistake is not taken for trees.
    """
    found: dict[tuple[str, str, str, int], TreeArm] = {}
    for p in sorted(root.glob(f"*/*/*/tw*/{SPEC_DIR}/*/*/results_bar*.csv")):
        m = _RESULTS.match(p.name)
        rel = p.relative_to(root).parts
        if m is None or len(rel) != 8:
            continue
        bucket, seg, model, twd, _spec, model2, bucket2, _f = rel
        if (model, bucket, seg) != (model2, bucket2, m.group(1)) or not twd[
            2:
        ].isdigit():
            print(f"skip {p}: path parts disagree ({rel})")
            continue
        arm = TreeArm(bucket, seg, model, int(twd[2:]), p)
        found.setdefault(arm.key, arm)
    for p in sorted(root.glob("*/*/tw*/results_bar*.csv")):
        m = _RESULTS.match(p.name)
        rel = p.relative_to(root).parts
        if (
            m is None
            or len(rel) != 4
            or rel[1] not in MODELS
            or not rel[2][2:].isdigit()
        ):
            continue
        arm = TreeArm(rel[0], m.group(1), rel[1], int(rel[2][2:]), p)
        if arm.key in found:
            print(f"note: {arm.key} in both layouts; the cluster layout is read")
            continue
        found[arm.key] = arm
    arms = sorted(found.values(), key=lambda a: (a.bucket, a.model, a.tw, a.seg))
    return [a for a in arms if tw is None or a.tw == tw]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(TREES), help="tree results root")
    ap.add_argument("--out", default=str(bsy.OUT), help="where the parquet tables go")
    ap.add_argument("--tw", type=int, default=TW, help="training window (sessions)")
    a = ap.parse_args()
    root, out = resolve(a.root), resolve(a.out)
    if not root.is_dir():
        print(f"no tree results root at {root}; nothing to stack")
        return
    out.mkdir(parents=True, exist_ok=True)
    written, failed = [], []
    for model in MODELS:
        for bucket in BUCKETS:
            name = f"yhat_subtree_{model}_{bucket}.parquet"
            paths = {b: tree_arm_path(root, bucket, b, model, a.tw) for b in BARS}
            missing = [b for b, p in paths.items() if p is None]
            if missing:
                have = len(BARS) - len(missing)
                print(
                    f"skip {name}: {len(missing)} of {len(BARS)} bar arms missing"
                    + (
                        f" (have {have}: {', '.join(b for b in BARS if b not in missing)})"
                        if have
                        else ""
                    )
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
