"""I5 cross-cluster check of the Optuna per-bar tree campaign: Hoffman2 (CARC's runtime) vs CARC.

Reads the pulled check tree (cluster/pull_optuna_h2_xcheck.sh) under <root>/<tag>/:
  <side>/stage1/<bucket>/<seg>/<model>/tw<TW>/chunks/c<k>/causal_tune_trees/<model>/<bucket>/trials_<seg>.npz
      tuning points, every trial (Hoffman2 re-runs of a few points; CARC's fleet chunk records)
  <side>/stage2/<bucket>/<seg>/<model>/tw<TW>/.../<path>/causal_tune_trees/<model>/<bucket>/trees_<seg>.npz
      a refit block (Hoffman2 from CARC's own stage-1 records of that block; CARC = its canary's stage 2)
  <side>/probe/*.csv, <side>/probe_x/*.npz   design / target hashes, the per-window kept-column set, and
      the design itself (experiments/optuna_h2_design_probe.py), one file per machine
  hosts.csv                                  host -> CPU model / Slurm features (the pull writes it)
Every PAIR (--pair A:B, repeatable; default h2:carc) is compared, and every probe of every side is
tabulated (the design's dependence on the CPU).  Writes into <root>/<tag>/:
  stage1_trials_<A>_vs_<B>.csv   one row per (model, point, trial): both sides' validation MSE / QLIKE,
                                 rounds, cap, patience, parameters; bit-identical flags; rel. diff.
  stage1_points_<A>_vs_<B>.csv   per (model, point): trials bit-identical, the picks (best-of-10/25/50 by
                                 MSE, best-of-50 by QLIKE) on each side, kept-column count, hosts
  stage2_paths_<A>_vs_<B>.csv    per (model, path): forecasts bit-identical, max rel. diff, configuration
                                 in force equal, importance / TreeSHAP identical, kept counts equal
  qualify_<A>_vs_<B>.csv         per model: the rule and the verdict
  probe_hosts.csv                per (side, probe file): host, CPU, design / target / kept-set hashes
  design_diffs.csv               per (bucket, host pair): cells / columns that differ, max abs / rel diff
  SUMMARY.md                     generated from the CSVs above (and <root>/SUMMARY.md over every tag)
Relative difference = |a - b| / max(|a|, |b|) (0 when both are 0; NaN on both sides counts equal).

Usage: python experiments/optuna_h2_crosscheck.py --tag mask --pair h2:carc --pair h2avx:carc
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REL_TOL = 1e-12  # the campaign's rule: differences at or below this, documented, still qualify
MODELS = ("lgbm", "xgb", "rf")
TRIAL_KEYS = ("val_mse", "val_qlike", "rounds", "rounds_max", "patience", "complete")
PICKS = ("best_mse_k10", "best_mse_k25", "best_mse_k50", "best_qlike_k50")
CFG_ARRAYS = ("true_adj", "cfg_point", "cfg_trial", "cfg_rounds", "cfg_params")
AUX_ARRAYS = ("importance", "shap")


def rel_diff(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    both_nan = np.isnan(a) & np.isnan(b)
    den = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, np.abs(a - b) / den, 0.0)
    r = np.where(both_nan, 0.0, r)
    return np.where(np.isnan(a) ^ np.isnan(b), np.inf, r)


def bit_equal(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def load(p: Path) -> dict:
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def chunk_host(f: Path) -> str:
    """The H2_HOST file of the chunk a result file belongs to (Hoffman2 side), else ''."""
    for d in f.parents:
        h = d / "H2_HOST"
        if h.is_file():
            t = h.read_text()
            if t.startswith(
                "host="
            ):  # host=<h> class=<AVX-512|AVX2|AVX> cpu=... seconds=... pool=...
                return t.split(" ")[0][5:] + "/" + t.split(" ")[1][6:]
            return t.split(" ")[0]
        if d.name.startswith("c") and d.parent.name == "chunks":
            break
    return ""


# ----------------------------------------------------------------------------- stage 1
def stage1_records(side: Path) -> dict[str, dict[tuple[str, int], dict]]:
    out: dict[str, dict[tuple[str, int], dict]] = {m: {} for m in MODELS}
    for f in sorted(
        side.glob("stage1/*/*/*/tw*/chunks/c*/causal_tune_trees/*/*/trials_*.npz")
    ):
        z = load(f)
        model, bucket = f.parts[-3], f.parts[-2]
        for j, r in enumerate(z["row"]):
            rec = {
                k: z[k][j] for k in TRIAL_KEYS + PICKS + ("params", "seed") if k in z
            }
            if "n_kept" in z:
                rec["n_kept"] = z["n_kept"][j]
            rec["_file"] = f.relative_to(side).as_posix()
            rec["_host"] = chunk_host(f)
            rec["_bucket"] = bucket
            out[model][(bucket, int(r))] = rec
    return out


def compare_stage1(A: dict, B: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    trials, points = [], []
    for model in MODELS:
        for (bucket, row), a in sorted(A[model].items()):
            b = B[model].get((bucket, row))
            base = {"model": model, "bucket": bucket, "row": row}
            if b is None:
                points.append(base | {"matched": False})
                continue
            tt = []
            for t in range(len(a["val_mse"])):
                d = dict(base, trial=t)
                for k in ("val_mse", "val_qlike"):
                    d[f"a_{k}"], d[f"b_{k}"] = float(a[k][t]), float(b[k][t])
                    d[f"{k}_bit_identical"] = bit_equal(a[k][t], b[k][t])
                    d[f"{k}_rel_diff"] = float(rel_diff(a[k][t], b[k][t]))
                for k in ("rounds", "rounds_max", "patience", "complete"):
                    d[f"a_{k}"], d[f"b_{k}"] = a[k][t].item(), b[k][t].item()
                d["params_bit_identical"] = bit_equal(a["params"][t], b["params"][t])
                d["params_max_rel_diff"] = float(
                    np.max(rel_diff(a["params"][t], b["params"][t]))
                )
                d["trial_bit_identical"] = bool(
                    d["val_mse_bit_identical"]
                    and d["val_qlike_bit_identical"]
                    and d["params_bit_identical"]
                    and all(
                        d[f"a_{k}"] == d[f"b_{k}"]
                        for k in ("rounds", "rounds_max", "patience", "complete")
                    )
                )
                tt.append(d)
            trials += tt
            first_diff = next(
                (x["trial"] for x in tt if not x["trial_bit_identical"]), -1
            )
            p = base | {
                "matched": True,
                "trials": len(tt),
                "trials_bit_identical": sum(x["trial_bit_identical"] for x in tt),
                "first_differing_trial": first_diff,
                "max_rel_diff_val_mse": max(x["val_mse_rel_diff"] for x in tt),
                "max_rel_diff_val_qlike": max(x["val_qlike_rel_diff"] for x in tt),
                "max_rel_diff_params": max(x["params_max_rel_diff"] for x in tt),
                "a_host": a["_host"],
                "b_host": b["_host"],
                "b_file": b["_file"],
                "a_file": a["_file"],
            }
            for k in PICKS:
                p[f"a_{k}"], p[f"b_{k}"] = int(a[k]), int(b[k])
            p["picks_identical"] = all(p[f"a_{k}"] == p[f"b_{k}"] for k in PICKS)
            p["chosen_config_identical"] = bool(
                p["picks_identical"]
                and all(
                    bit_equal(a["params"][p[f"a_{k}"]], b["params"][p[f"b_{k}"]])
                    and a["rounds"][p[f"a_{k}"]] == b["rounds"][p[f"b_{k}"]]
                    for k in PICKS
                )
            )
            if "n_kept" in a or "n_kept" in b:
                p["a_n_kept"] = int(a.get("n_kept", -1))
                p["b_n_kept"] = int(b.get("n_kept", -1))
            points.append(p)
    return pd.DataFrame(trials), pd.DataFrame(points)


# ----------------------------------------------------------------------------- stage 2
def stage2_files(side: Path) -> dict[tuple[str, str, str], Path]:
    out = {}
    for f in sorted(side.glob("stage2/**/causal_tune_trees/*/*/trees_*.npz")):
        out[(f.parts[-2], f.parts[-3], f.parts[-5])] = f  # (bucket, model, path)
    return out


def compare_stage2(a_side: Path, b_side: Path) -> pd.DataFrame:
    """Per (bucket, model, path): the forecast rows both sides hold (matched by stamp: a 20-row check
    block against the first 20 rows of a fleet chunk), bit for bit."""
    fa, fb = stage2_files(a_side), stage2_files(b_side)
    rows = []
    for key in sorted(set(fa) & set(fb)):
        bucket, model, path = key
        a, b = load(fa[key]), load(fb[key])
        da, db = [str(x) for x in a["date"]], [str(x) for x in b["date"]]
        pos = {x: k for k, x in enumerate(db)}
        ia = np.array([k for k, x in enumerate(da) if x in pos], dtype=np.int64)
        ib = np.array([pos[da[k]] for k in ia], dtype=np.int64)
        d: dict = {
            "bucket": bucket,
            "model": model,
            "path": path,
            "rows": len(ia),
            "a_rows": len(da),
            "b_rows": len(db),
        }
        pa, pb = a["pred_adj"][ia], b["pred_adj"][ib]
        d["forecasts_bit_identical"] = int(sum(bit_equal(x, y) for x, y in zip(pa, pb)))
        d["forecasts_max_rel_diff"] = (
            float(np.max(rel_diff(pa, pb))) if len(ia) else np.nan
        )
        for k in CFG_ARRAYS:
            d[f"{k}_bit_identical"] = bit_equal(a[k][ia], b[k][ib])
        for k in AUX_ARRAYS:
            if a[k].size and b[k].size and a[k].shape[1:] == b[k].shape[1:]:
                xa, xb = a[k][ia], b[k][ib]
                d[f"{k}_bit_identical"] = bit_equal(xa, xb)
                d[f"{k}_max_rel_diff"] = (
                    float(np.max(rel_diff(xa, xb))) if xa.size else np.nan
                )
            else:
                d[f"{k}_bit_identical"] = bool(a[k].size == 0 and b[k].size == 0)
        if "n_kept" in a and "n_kept" in b:
            d["n_kept_equal"] = bool(np.array_equal(a["n_kept"][ia], b["n_kept"][ib]))
            d["n_kept_min"], d["n_kept_max"] = (
                int(a["n_kept"][ia].min()),
                int(a["n_kept"][ia].max()),
            )
        d["a_host"] = chunk_host(fa[key])
        d["b_file"] = fb[key].relative_to(b_side).as_posix()
        rows.append(d)
    return pd.DataFrame(rows)


def qualify(points: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        p = (
            points[(points["model"] == model) & points["matched"]]
            if len(points)
            else points
        )
        q = paths[paths["model"] == model] if len(paths) else paths
        d: dict = {"model": model, "points": len(p), "paths": len(q)}
        d["stage1_trials"] = int(p["trials"].sum()) if len(p) else 0
        d["stage1_trials_bit_identical"] = (
            int(p["trials_bit_identical"].sum()) if len(p) else 0
        )
        d["stage1_picks_identical"] = (
            bool(p["picks_identical"].all()) if len(p) else None
        )
        d["stage1_chosen_config_identical"] = (
            bool(p["chosen_config_identical"].all()) if len(p) else None
        )
        d["stage1_max_rel_diff"] = (
            float(
                p[
                    [
                        "max_rel_diff_val_mse",
                        "max_rel_diff_val_qlike",
                        "max_rel_diff_params",
                    ]
                ]
                .max()
                .max()
            )
            if len(p)
            else np.nan
        )
        d["stage2_forecasts"] = int(q["rows"].sum()) if len(q) else 0
        d["stage2_forecasts_bit_identical"] = (
            int(q["forecasts_bit_identical"].sum()) if len(q) else 0
        )
        d["stage2_cfg_identical"] = (
            bool(q[[f"{k}_bit_identical" for k in CFG_ARRAYS]].all().all())
            if len(q)
            else None
        )
        d["stage2_importance_shap_identical"] = (
            bool(q[[f"{k}_bit_identical" for k in AUX_ARRAYS]].all().all())
            if len(q)
            else None
        )
        d["stage2_max_rel_diff"] = (
            float(q["forecasts_max_rel_diff"].max()) if len(q) else np.nan
        )
        have = d["points"] > 0 and d["paths"] > 0
        bit = bool(
            have
            and d["stage1_trials_bit_identical"] == d["stage1_trials"]
            and d["stage1_picks_identical"]
            and d["stage2_forecasts_bit_identical"] == d["stage2_forecasts"]
            and d["stage2_cfg_identical"]
        )
        tol = bool(
            have
            and d["stage1_picks_identical"]
            and d["stage1_chosen_config_identical"]
            and d["stage2_cfg_identical"]
            and d["stage1_max_rel_diff"] <= REL_TOL
            and d["stage2_max_rel_diff"] <= REL_TOL
        )
        d["verdict"] = (
            "bit-identical"
            if bit
            else ("within 1e-12" if tol else ("differs" if have else "incomplete"))
        )
        d["qualifies"] = bit or tol
        rows.append(d)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- the design probe
def probe_hosts(base: Path, sides: list[str], cpu: dict[str, str]) -> pd.DataFrame:
    rows = []
    for side in sides:
        for f in sorted((base / side / "probe").glob("*.csv")):
            t = pd.read_csv(f)
            host = f.stem.split("_")[-1]
            kept = hashlib.sha256("".join(t["kept_sha"]).encode()).hexdigest()
            wins = hashlib.sha256("".join(t["window_sha"]).encode()).hexdigest()
            rows.append(
                {
                    "side": side,
                    "file": f.name,
                    "host": host,
                    "cpu": cpu.get(host, ""),
                    "bucket": t["bucket"].iloc[0],
                    "window_mask": int(t["window_mask"].iloc[0]),
                    "rows": len(t),
                    "p": int(t["p"].iloc[0]),
                    "n_kept_min": int(t["n_kept"].min()),
                    "n_kept_max": int(t["n_kept"].max()),
                    "X_sha": t["X_sha"].iloc[0][:16],
                    "y_sha": t["y_sha"].iloc[0][:16],
                    "windows_sha": wins[:16],
                    "kept_sets_sha": kept[:16],
                }
            )
    return pd.DataFrame(rows)


def design_diffs(base: Path, sides: list[str], cpu: dict[str, str]) -> pd.DataFrame:
    files = [f for s in sides for f in sorted((base / s / "probe_x").glob("*.npz"))]
    by_bucket: dict[str, list[Path]] = {}
    for f in files:
        by_bucket.setdefault(f.stem.split("_c")[0], []).append(f)
    rows = []
    for bucket, fs in sorted(by_bucket.items()):
        Z = {f: load(f) for f in fs}
        for i, fa in enumerate(fs):
            for fb in fs[i + 1 :]:
                A, B = Z[fa]["X"], Z[fb]["X"]
                ha, hb = fa.stem.split("_")[-1], fb.stem.split("_")[-1]
                d = {
                    "bucket": bucket,
                    "host_a": ha,
                    "cpu_a": cpu.get(ha, ""),
                    "host_b": hb,
                    "cpu_b": cpu.get(hb, ""),
                    "cells": A.size,
                }
                if A.shape != B.shape:
                    rows.append(d | {"identical": False, "note": "shape differs"})
                    continue
                diff = (A != B) & ~(np.isnan(A) & np.isnan(B))
                d["identical"] = not diff.any()
                d["y_identical"] = bit_equal(Z[fa]["y"], Z[fb]["y"])
                d["cells_differ"] = int(diff.sum())
                d["cols_differ"] = int(diff.any(0).sum())
                d["cols"] = A.shape[1]
                if diff.any():
                    ab = np.abs(A - B)[diff]
                    d["max_abs_diff"] = float(ab.max())
                    d["max_rel_diff"] = float(rel_diff(A[diff], B[diff]).max())
                    d["median_rel_diff"] = float(np.median(rel_diff(A[diff], B[diff])))
                    scale = np.nanmax(np.abs(A), axis=0)
                    d["max_abs_diff_over_col_scale"] = float(
                        np.nanmax(
                            np.where(
                                diff,
                                np.abs(A - B) / np.where(scale > 0, scale, 1.0),
                                0.0,
                            )
                        )
                    )
                rows.append(d)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- report
def fmt(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (float, np.floating)):
        return f"{x:.3g}"
    return str(x)


def table(df: pd.DataFrame, cols: list[str]) -> list[str]:
    cols = [c for c in cols if c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument(
        "--root", default="results/linear_subsection_trees_optuna/crosscluster"
    )
    ap.add_argument(
        "--pair", action="append", default=None, help="A:B side dirs to compare"
    )
    ap.add_argument(
        "--side-class",
        action="append",
        default=[],
        help="SIDE=CLASS (e.g. h2x512=AVX-512)",
    )
    ap.add_argument(
        "--fleet-side",
        default="carc",
        help="the side holding C's fleet chunks (class per chunk)",
    )
    ap.add_argument(
        "--fleet-min-job", type=int, default=0, help="C's run: Slurm job ids >= this"
    )
    a = ap.parse_args()
    base = Path(a.root) / a.tag
    pairs = [tuple(p.split(":")) for p in (a.pair or ["h2:carc"])]
    sides = sorted(
        {s for p in pairs for s in p}
        | {d.name for d in base.iterdir() if d.is_dir() and not d.name.startswith(".")}
    )
    hosts = (
        pd.read_csv(base / "hosts.csv")
        if (base / "hosts.csv").is_file()
        else pd.DataFrame(columns=["host", "cpu"])
    )
    cpu: dict[str, str] = {}
    for hf in sorted(
        Path(a.root).glob("*/hosts.csv")
    ):  # every tag's host map (one host, one CPU)
        t = read_csv_or_empty(hf)
        if len(t):
            cpu.update(
                {
                    str(h): str(c)
                    for h, c in zip(t["host"], t["cpu"])
                    if isinstance(c, str) and c
                }
            )
    cpu.update(
        {
            str(h): str(c)
            for h, c in zip(hosts["host"], hosts["cpu"])
            if isinstance(c, str) and c
        }
    )
    notes = (
        json.loads((base / "notes.json").read_text(encoding="utf-8"))
        if (base / "notes.json").is_file()
        else {}
    )
    L = [
        f"# Cross-cluster check `{a.tag}`: Hoffman2 (CARC's runtime) vs CARC",
        "",
        f"Generated by `experiments/optuna_h2_crosscheck.py --tag {a.tag}` from the CSVs in this folder.",
        "",
    ]
    L += [f"- {k}: {v}" for k, v in notes.items()]
    L += [
        "",
        f"Rule (campaign file, I5): a model qualifies for Hoffman2 only if its picks are identical and its "
        f"forecasts bit-identical, or every difference is at most {REL_TOL:g} relative and documented.",
        "",
    ]
    fc = fleet_classes(base, a.fleet_min_job) if a.fleet_min_job else pd.DataFrame()
    if len(fc):
        fc.to_csv(base / "fleet_chunk_classes.csv", index=False)
    chunk_class = (
        {r.chunk: r.cpu_class for r in fc.itertuples()} if len(fc) else {}
    )  # sorted by job / task: the last task to log a chunk wins
    fixed = dict(x.split("=", 1) for x in a.side_class)

    def side_class_of(side: str, rel_file: str) -> str:
        if side in fixed:
            return fixed[side]
        if side == a.fleet_side and isinstance(rel_file, str):
            key = "/".join(
                rel_file.split("/")[:7]
            )  # stage1/<b>/<seg>/<m>/tw<TW>/chunks/c<k>
            return chunk_class.get(key, "unknown")
        return "unknown"

    done_pairs: list[str] = []
    for A, B in pairs:
        if not ((base / A).is_dir() and (base / B).is_dir()):
            continue
        trials, points = compare_stage1(
            stage1_records(base / A), stage1_records(base / B)
        )
        paths = compare_stage2(base / A, base / B)
        if not len(points) and not len(paths):
            continue  # nothing of this pair on disk yet
        for col, side in (("a_class", A), ("b_class", B)):
            if len(points):
                points[col] = [
                    side_class_of(side, f)
                    for f in points.get(f"{col[0]}_file", [""] * len(points))
                ]
        qual = qualify(points, paths)
        sfx = f"{A}_vs_{B}"
        trials.to_csv(base / f"stage1_trials_{sfx}.csv", index=False)
        points.to_csv(base / f"stage1_points_{sfx}.csv", index=False)
        paths.to_csv(base / f"stage2_paths_{sfx}.csv", index=False)
        qual.to_csv(base / f"qualify_{sfx}.csv", index=False)
        done_pairs.append(sfx)
        L += [f"## {A} vs {B}", "", "### Verdict per model", ""]
        L += table(
            qual,
            [
                "model",
                "points",
                "stage1_trials",
                "stage1_trials_bit_identical",
                "stage1_picks_identical",
                "stage1_chosen_config_identical",
                "stage1_max_rel_diff",
                "paths",
                "stage2_forecasts",
                "stage2_forecasts_bit_identical",
                "stage2_cfg_identical",
                "stage2_importance_shap_identical",
                "stage2_max_rel_diff",
                "verdict",
                "qualifies",
            ],
        )
        if len(points):
            L += ["", "### Stage 1: tuning points (every trial)", ""]
            L += table(
                points,
                [
                    "model",
                    "bucket",
                    "row",
                    "matched",
                    "trials",
                    "trials_bit_identical",
                    "first_differing_trial",
                    "picks_identical",
                    "a_best_mse_k50",
                    "b_best_mse_k50",
                    "a_best_qlike_k50",
                    "b_best_qlike_k50",
                    "a_n_kept",
                    "b_n_kept",
                    "max_rel_diff_val_mse",
                    "a_class",
                    "b_class",
                    "a_host",
                ],
            )
        if len(paths):
            L += ["", "### Stage 2: refit block, per path", ""]
            L += table(
                paths,
                [
                    "model",
                    "bucket",
                    "path",
                    "rows",
                    "forecasts_bit_identical",
                    "forecasts_max_rel_diff",
                    "cfg_trial_bit_identical",
                    "cfg_rounds_bit_identical",
                    "importance_bit_identical",
                    "shap_bit_identical",
                    "n_kept_equal",
                    "a_host",
                ],
            )
        L.append("")
        print(f"{sfx}:\n{qual.to_string(index=False)}")
    (base / "pairs.txt").write_text(" ".join(done_pairs) + " ", encoding="utf-8")
    ph = probe_hosts(base, sides, cpu)
    dd = design_diffs(base, sides, cpu)
    ph.to_csv(base / "probe_hosts.csv", index=False)
    dd.to_csv(base / "design_diffs.csv", index=False)
    if len(ph):
        L += [
            "## The design, the target and the kept-column set on each machine (probe)",
            "",
        ]
        L += table(
            ph,
            [
                "side",
                "host",
                "cpu",
                "bucket",
                "window_mask",
                "rows",
                "p",
                "n_kept_min",
                "n_kept_max",
                "X_sha",
                "y_sha",
                "kept_sets_sha",
            ],
        )
    if len(dd):
        L += ["", "## Design differences between machines (element by element)", ""]
        L += table(
            dd,
            [
                "bucket",
                "host_a",
                "cpu_a",
                "host_b",
                "cpu_b",
                "identical",
                "y_identical",
                "cells_differ",
                "cols_differ",
                "cols",
                "max_abs_diff",
                "max_abs_diff_over_col_scale",
                "median_rel_diff",
            ],
        )
    if len(fc):
        last = fc.groupby("chunk").tail(1)
        g = (
            last.groupby(["stage", "bucket", "model", "cpu_class"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        L += [
            "",
            f"## C's run: the CPU class that computed each chunk (Slurm jobs >= {a.fleet_min_job})",
            "",
        ]
        L += table(g, list(g.columns))
        L += [
            "",
            "(one row per chunk in `fleet_chunk_classes.csv`: Slurm task, node, features, class)",
        ]
    (base / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {base}/SUMMARY.md")
    write_index(Path(a.root))


def cpu_class(features_or_cpu: str) -> str:
    """The vector level the design depends on: AVX-512 (CARC xeon-4116, Hoffman2 Xeon Gold / 6736P) or
    not (CARC epyc-7513 / epyc-7542: AVX2; Hoffman2 E5-2650 / E5-2670: AVX) -- probe-verified classes."""
    s = str(features_or_cpu).lower()
    if any(
        k in s
        for k in ("xeon-4116", "silver 4116", "gold 61", "gold 62", "gold 63", "6736p")
    ):
        return "AVX-512"
    if "epyc" in s:
        return "AVX2"
    if "xeon-2640v4" in s or "xeon-2640v3" in s:
        return "AVX2 (xeon-2640, design not probed)"
    if "e5-26" in s:
        return "AVX"
    return "unknown"


def fleet_classes(base: Path, min_job: int) -> pd.DataFrame:
    """Which CPU class computed each chunk of C's run (Slurm job ids >= min_job), from the pulled
    Slurm log lines (carc_chunk_tasks.csv); the last task to log a chunk is the one that wrote it."""
    t = read_csv_or_empty(base / "carc_chunk_tasks.csv")
    if not len(t):
        return t
    t["job"] = t["slurm_task"].str.split("_").str[0].astype(int)
    t["task"] = t["slurm_task"].str.split("_").str[1].astype(int)
    t = t[
        (t["job"] >= min_job) & t["job_name"].isin(["op_canary", "op_s1", "op_s2"])
    ].copy()
    t["cpu_class"] = t["carc_features"].map(cpu_class)
    parts = t["chunk"].str.split("/", expand=True)
    t["stage"], t["bucket"], t["model"], t["chunk_id"] = (
        parts[0],
        parts[1],
        parts[3],
        parts[6],
    )
    return t.sort_values(["stage", "bucket", "model", "chunk_id", "job", "task"])


def read_csv_or_empty(f: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(f)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def write_index(root: Path) -> None:
    """<root>/SUMMARY.md: every tag's verdicts, the design's CPU classes, and the offload records."""
    L = [
        "# Cross-cluster check of the Optuna campaign (checklist I5): Hoffman2 vs CARC",
        "",
        "Generated by `experiments/optuna_h2_crosscheck.py` from the CSVs under this folder; one "
        "sub-folder per check tag (its own SUMMARY.md has every table).",
        "",
    ]
    notes = root / "notes.json"
    if notes.is_file():
        L += [
            f"- {k}: {v}"
            for k, v in json.loads(notes.read_text(encoding="utf-8")).items()
        ] + [""]
    for tag in sorted(
        d for d in root.iterdir() if d.is_dir() and not d.name.startswith("offload_")
    ):
        listed = (
            tag / "pairs.txt"
        )  # the pairs of the tag's last run (older outputs are not indexed)
        quals = (
            [
                tag / f"qualify_{x.strip()}.csv"
                for x in listed.read_text().split()
                if x.strip()
            ]
            if listed.is_file()
            else []
        )
        quals = [q for q in quals if q.is_file()]
        ph = tag / "probe_hosts.csv"
        dd = tag / "design_diffs.csv"
        if not (quals or ph.is_file()):
            continue
        L += [f"## Tag `{tag.name}`", ""]
        for q in quals:
            t = read_csv_or_empty(q)
            L += [f"### {q.stem.removeprefix('qualify_').replace('_vs_', ' vs ')}", ""]
            L += table(
                t,
                [
                    "model",
                    "points",
                    "stage1_trials",
                    "stage1_trials_bit_identical",
                    "stage1_picks_identical",
                    "stage1_max_rel_diff",
                    "stage2_forecasts",
                    "stage2_forecasts_bit_identical",
                    "stage2_cfg_identical",
                    "stage2_max_rel_diff",
                    "verdict",
                    "qualifies",
                ],
            )
            L.append("")
        if ph.is_file():
            t = read_csv_or_empty(ph)
            if len(t):
                g = (
                    t.groupby(["bucket", "X_sha"])
                    .agg(
                        hosts=("host", lambda s: " ".join(sorted(set(map(str, s))))),
                        cpus=("cpu", lambda s: "; ".join(sorted(set(map(str, s))))),
                        kept_sets=("kept_sets_sha", "nunique"),
                        y=("y_sha", "nunique"),
                    )
                    .reset_index()
                )
                L += ["### The design X per machine (one row per distinct X)", ""]
                L += table(g, ["bucket", "X_sha", "hosts", "cpus", "kept_sets", "y"])
                L.append("")
        if dd.is_file():
            t = read_csv_or_empty(dd)
            t = t[~t["identical"].astype(bool)] if len(t) else t
            if len(t):
                L += ["### Where two machines' designs differ", ""]
                L += table(
                    t,
                    [
                        "bucket",
                        "host_a",
                        "host_b",
                        "cells_differ",
                        "cols_differ",
                        "cols",
                        "max_abs_diff",
                        "max_abs_diff_over_col_scale",
                        "median_rel_diff",
                    ],
                )
                L.append("")
    for off in sorted(root.glob("offload_*")):
        L += [f"## Offload `{off.name}`", ""]
        for name in (
            "carc_tasks.txt",
            "log.txt",
            "chunk_hosts.csv",
            "released_logs.txt",
        ):
            f = off / name
            if f.is_file():
                txt = f.read_text(encoding="utf-8").strip().splitlines()
                L += (
                    [f"`{name}` ({len(txt)} lines):", "", "```"]
                    + txt[:80]
                    + ["```", ""]
                )
    (root / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {root}/SUMMARY.md")


if __name__ == "__main__":
    main()
