"""causal_tune_trees_mcs -- the per-bar 16:00 tree on ONE candidate design of the feature Model Confidence Set
(2026-10-08; results/feature_mcs_2026-10-08/, the linear candidates by experiments/feature_mcs_runner.py).

A wrapper, not a new spec: it executes a shipped tree spec UNCHANGED except for four lines (each must occur exactly
once, else it refuses):

  BASE  (HPC_KW_MCS_BASE)  specs/causal_tune_trees.py          LightGBM lgbm_opt_00 (the shipped spec)
                           specs/causal_tune_trees_xgbtop.py   XGBoost xgb_opt_00 (generated, xgbtop)
                           specs/causal_tune_trees_rf400.py    random forest, the 400-trial search's trial 255
  DESIGN (HPC_KW_MCS_DESIGN) one candidate of experiments/feature_mcs_runner.py: its column list, taken from that runner's
         load_inputs()["models"] (written to HPC_KW_MCS_DESIGNS by experiments/feature_mcs_trees.py prepare); the column
         names are the tree design's own (the all_features 16:00-bar design: the same 628 names in the same order --
         checked here at run time against the executor's feature names).

The spec runs with EXOG_BUCKET = all_features (its design, its rows, its prescaling: one index space for every bucket);
fit_predict receives the all_features columns and passes ONLY the design's columns on (order kept) to the spec's own
walk-forward, so the window mask, the refits and everything after see exactly the design -- a bucket run on those columns.
Rows: only the out-of-sample rows listed in HPC_KW_MCS_ROWS (the 866 trade days of the MCS sample) are refitted and
predicted; every other row's forecast is NaN.  Each refit is independent of the others (REFIT_EVERY 1, a fresh model
seeded 42 on its own trailing window), so a computed row is the row the full run computes.

Patches: (1) fit_predict = the column-subsetting wrapper of the spec's fit_predict_tree; (2) preds start as NaN; (3) the
refit loop visits the listed rows only; (4) the refit-count assert counts the listed rows; plus the design's name and
column count in the npz meta.  Run with the same env axes as the stored T1 runs (cluster/slurm/feature_mcs_trees_pack.sbatch).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _env(name: str, default: str = "") -> str:
    return os.environ.get(f"HPC_KW_{name}", os.environ.get(name, default))


PATCHES = (
    (
        "    preds = np.empty(n_test)\n",
        "    preds = np.full(n_test, np.nan)  # mcs: rows not listed stay NaN\n",
    ),
    (
        "    for i in range(0, n_test, REFIT_EVERY):\n",
        "    for i in [j for j in range(0, n_test, REFIT_EVERY) if (off + j) in _MCS_ROWS]:  # mcs: listed rows only\n",
    ),
    (
        "    fit_predict=fit_predict_tree,\n",
        "    fit_predict=_MCS_WRAP(fit_predict_tree),  # mcs: the design's columns\n",
    ),
    (
        'assert len(SIDE["refit_row"]) == -(-n_oos // REFIT_EVERY), (len(SIDE["refit_row"]), n_oos)\n',
        'assert len(SIDE["refit_row"]) == sum((max(0, START - HALO) + j) in _MCS_ROWS for j in range(n_oos)), '
        '(len(SIDE["refit_row"]), n_oos)  # mcs\n',
    ),
    (
        '            "bucket": EXOG_BUCKET,\n',
        '            "bucket": EXOG_BUCKET,\n            "mcs_design": _MCS_DESIGN,\n            "mcs_n_cols": len(_MCS_COLS),\n',
    ),
)


def main() -> None:
    base = HERE / Path(_env("MCS_BASE")).name
    design = _env("MCS_DESIGN")
    designs = json.loads(Path(_env("MCS_DESIGNS")).read_text(encoding="utf-8"))
    rows = json.loads(Path(_env("MCS_ROWS")).read_text(encoding="utf-8"))
    if base.name not in (
        "causal_tune_trees.py",
        "causal_tune_trees_xgbtop.py",
        "causal_tune_trees_rf400.py",
    ):
        raise SystemExit(
            f"MCS_BASE must be one of the three shipped specs, got {base.name!r}"
        )
    if design not in designs["models"]:
        raise SystemExit(
            f"MCS_DESIGN {design!r} is not a candidate of {_env('MCS_DESIGNS')}"
        )
    if (
        _env("EXOG_BUCKET") != "all_features"
        or _env("REFIT_EVERY") != "1"
        or _env("WINDOW_MASK") != "1"
    ):
        raise SystemExit(
            "the MCS trees run with EXOG_BUCKET=all_features, REFIT_EVERY=1, WINDOW_MASK=1"
        )
    names = list(designs["names"])
    cols = list(designs["models"][design]["cols"])
    idx = [names.index(c) for c in cols]
    if idx != sorted(idx) or len(set(idx)) != len(idx):
        raise SystemExit(
            f"{design}: the columns are not an order-preserving subset of the all_features design"
        )
    want = {int(r) for r in rows["oos_rows"]}

    def wrap(fp):
        def g(X_chunk, y_chunk, train_win_periods, hyperparams):
            got = [str(c) for c in hyperparams["_feature_names"]]
            if got != names:
                raise SystemExit(
                    "the executor's all_features columns are not the MCS design's 628 names"
                )
            X = np.asarray(X_chunk)[:, idx]
            hp = dict(hyperparams, _feature_names=[got[i] for i in idx])
            print(
                f"mcs: design {design}: {len(idx)} of {len(got)} columns, {X.shape[0]} rows in the chunk",
                flush=True,
            )
            return fp(X, y_chunk, train_win_periods, hp)

        return g

    src = base.read_text(encoding="utf-8")
    for old, new in PATCHES:
        if src.count(old) != 1:
            raise SystemExit(
                f"{base}: patch anchor found {src.count(old)} times, expected once: {old.strip()!r}"
            )
        src = src.replace(old, new)
    print(
        f"mcs: {base.name} on design {design} ({len(cols)} columns), {len(want)} listed OOS rows",
        flush=True,
    )
    sys.argv = [str(base)]
    g = {
        "__file__": str(base),
        "__name__": "__main__",
        "_MCS_WRAP": wrap,
        "_MCS_ROWS": want,
        "_MCS_DESIGN": design,
        "_MCS_COLS": cols,
    }
    exec(compile(src, str(base), "exec"), g)  # noqa: S102 -- a shipped spec's own source, the lines above patched


if __name__ == "__main__":
    main()
