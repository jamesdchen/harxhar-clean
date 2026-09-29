"""Extract the per-refit native importance and the TreeSHAP rows of the TUNED per-bar trees.

Checklist item A2b (causally tuned per-bar trees), for the feature-importance
workstream.  The tuned campaign (specs/causal_tune_trees_tuned.py, merged by
experiments/reduce_trees_tuned_chunks.py) stores, per arm (bucket x bar x model),
one npz with

  importance   (refits x p) native importance of every refit, in refit order
               (LightGBM split gain, XGBoost total_gain, random forest impurity
               decrease = sklearn feature_importances_, which sums to one per refit)
  refit_row    the OOS row each refit's block starts at (the model was fitted on
               the TRAIN_WIN rows strictly before it)
  shap         (OOS rows x p + 1) path-dependent TreeSHAP of every forecast row
               under the model in force, last column = the expected value; only on
               the SHAP bars (the spec's SHAP_SEGMENTS, default bar1600)
  date, pred_adj, feature_names

The npz files stay where they are (they are not pulled); this script turns them
into tables, so it is run where the npz files live:

  <out>/importance_refits_<bucket>_<model>.parquet
        one row per (segment, refit): segment, refit_idx, refit_row, date = the
        stamp of the first forecast row of the refit's block (naive ET, bar-end
        labelled), then one float32 column per design column (native importance)
  <out>/shap_rows_<seg>_<bucket>_<model>.parquet
        one row per forecast row of a SHAP bar: date, pred_adj, expected_value,
        then one float32 column per design column (TreeSHAP on the adjusted scale)
  <out>/importance_manifest.csv
        per arm: refits, features, importance type, SHAP rows, the largest
        additivity gap |sum(SHAP row) - pred_adj|, the npz path

Only merged arms (the unchunked layout) are read; the per-chunk npz files under
chunks/ are not.

Usage:  python experiments/a2b_extract_tree_importance.py
            [--root results/linear_subsection_trees_tuned] [--out <root>/importance] [--tw 2000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = "causal_tune_trees"
IMPORTANCE_TYPE = {
    "lgbm": "LightGBM split gain (importance_type='gain')",
    "xgb": "XGBoost total_gain",
    "rf": "random forest impurity decrease (feature_importances_, sums to 1 per refit)",
}


def resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def merged_npz(root: Path, tw: int) -> list[tuple[str, str, str, Path]]:
    """(bucket, segment, model, npz) of every merged arm; per-chunk files are skipped."""
    out = []
    for p in sorted(root.glob(f"*/*/*/tw{tw}/{SPEC_DIR}/*/*/trees_bar*.npz")):
        rel = p.relative_to(
            root
        ).parts  # bucket, seg, model, twNNNN, spec, model, bucket, file
        if len(rel) != 8:
            continue
        bucket, seg, model, _tw, _spec, model2, bucket2, name = rel
        if (model, bucket) != (model2, bucket2) or name != f"trees_{seg}.npz":
            print(f"skip {p}: path parts disagree")
            continue
        out.append((bucket, seg, model, p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/linear_subsection_trees_tuned")
    ap.add_argument("--out", default=None, help="default: <root>/importance")
    ap.add_argument("--tw", type=int, default=2000)
    a = ap.parse_args()
    root = resolve(a.root)
    out = resolve(a.out) if a.out else root / "importance"
    out.mkdir(parents=True, exist_ok=True)
    arms = merged_npz(root, a.tw)
    print(f"{len(arms)} merged arms under {root}")

    refit_parts: dict[tuple[str, str], list[pd.DataFrame]] = {}
    names_of: dict[tuple[str, str], list[str]] = {}
    manifest = []
    for bucket, seg, model, path in arms:
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        names = [str(s) for s in d["feature_names"]]
        p = len(names)
        dates = pd.to_datetime(pd.Series(np.asarray(d["date"]).astype(str)))
        imp = np.asarray(d["importance"], dtype=np.float32)
        rr = np.asarray(d["refit_row"], dtype=np.int64)
        assert imp.ndim == 2 and imp.shape == (len(rr), p), (
            path,
            imp.shape,
            len(rr),
            p,
        )
        assert (rr >= 0).all() and (rr < len(dates)).all(), path
        key = (bucket, model)
        if key in names_of and names_of[key] != names:
            raise SystemExit(
                f"{path}: feature names differ from the other bars of {key}"
            )
        names_of[key] = names
        head = pd.DataFrame(
            {
                "segment": seg,
                "refit_idx": np.arange(len(rr), dtype=np.int64),
                "refit_row": rr,
                "date": dates.iloc[rr].to_numpy(),
            }
        )
        refit_parts.setdefault(key, []).append(
            pd.concat([head, pd.DataFrame(imp, columns=names)], axis=1)
        )

        sh = np.asarray(d.get("shap", np.zeros((0, 0))), dtype=np.float32)
        shap_rows, gap = 0, float("nan")
        if sh.ndim == 2 and sh.size and sh.shape[1] == p + 1:
            pred = np.asarray(d["pred_adj"], dtype=np.float64)
            ok = np.isfinite(sh).all(axis=1)
            gap = float(np.abs(sh[ok].astype(np.float64).sum(axis=1) - pred[ok]).max())
            tab = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "date": dates.to_numpy(),
                            "pred_adj": pred,
                            "expected_value": sh[:, p],
                        }
                    ),
                    pd.DataFrame(sh[:, :p], columns=names),
                ],
                axis=1,
            )
            tab.to_parquet(
                out / f"shap_rows_{seg}_{bucket}_{model}.parquet", index=False
            )
            shap_rows = int(ok.sum())
        meta = json.loads(str(d.get("meta", "{}")))
        manifest.append(
            {
                "bucket": bucket,
                "model": model,
                "segment": seg,
                "refits": len(rr),
                "n_features": p,
                "importance_type": IMPORTANCE_TYPE.get(model, "native"),
                "refit_every": meta.get("refit_every"),
                "train_rows": meta.get("train_rows"),
                "oos_rows": len(dates),
                "first": str(dates.min()),
                "last": str(dates.max()),
                "shap_rows": shap_rows,
                "max_abs_additivity_gap": gap,
                "npz": path.relative_to(root).as_posix(),
            }
        )

    for (bucket, model), parts in sorted(refit_parts.items()):
        tab = pd.concat(parts, ignore_index=True)
        name = f"importance_refits_{bucket}_{model}.parquet"
        tab.to_parquet(out / name, index=False)
        print(
            f"wrote {name}: {len(tab)} refits x {tab.shape[1] - 4} features, {tab['segment'].nunique()} bars"
        )
    man = pd.DataFrame(manifest)
    man.to_csv(out / "importance_manifest.csv", index=False)
    print(
        f"wrote importance_manifest.csv ({len(man)} arms, {int((man['shap_rows'] > 0).sum())} with SHAP rows)"
    )


if __name__ == "__main__":
    main()
