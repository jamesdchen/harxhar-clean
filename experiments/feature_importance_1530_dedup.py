"""C1 feature importance of the 16:00-bar forecast re-run on the de-duplicated per-bar design.

Checklist I7 (plan writeup/CAMPAIGN_16H_2026-09-29.md, item F).  Commit 47f7f9c dropped the
twelve HAR x open / close session-edge interactions from the per-bar design (at 16:00 the six
``har_ma_k_x_open`` columns are all zero and the six ``har_ma_k_x_close`` columns are exact
copies of ``har_ma_1 .. har_ma_3125``): columns HAR + calendar 34 -> 22, live-feasible
244 -> 232, all features 640 -> 628.  The user's later decision (2026-09-29, commit f9a19b6)
gives every per-bar TREE fit the linear arms' identifiability mask: each refit uses only the
columns src/models/window_mask.window_keep keeps on its own 2000-session training window.

Same methodology as the first pass (experiments/feature_importance_1530.py + _trees.py,
results/feature_importance_1530/): the same 147 refits (every 10 sessions), the same causal
held-out tails, the same permutation draws, the same five models and three designs.  Before /
after differs only by the design (and, for the trees, the mask).  Two roots:

  results/feature_importance_1530_dedup/          de-dup + per-window mask (the trees masked)
  results/feature_importance_1530_dedup_nomask/   de-dup, no mask (the intermediate rung)

The linear models (per-bar ridge / lasso) are the same under both: their identifiability
mask already removed the session-edge columns, constant columns and copies at every tune.

This script sets the re-run's roots (env, before importing the first pass's modules) and adds
the re-run's own stages:

  ``python experiments/feature_importance_1530_dedup.py [--nomask] <stage> [args]``
  design / inputs / capture <bucket> <est> / linear [b] [e] / linear_part <b> <e> <k0> <k1> /
  merge_linear <b> <e> / aggregate      the first pass's stages, on this root
  gate_design     p = 22 / 232 / 628, no session-edge column, first pass's columns minus the
                  new ones = exactly the 12 session-edge columns, the same out-of-sample rows,
                  the same targets bit for bit, the other columns bit for bit
  share           (nomask) copy the design inputs and the linear walks from the mask root
                  (identical inputs; the linear models do not depend on the tree mask)
  tuned_extract   the causally tuned trees' own extracts (random search, refit every 10) from
                  the stored merged runs (experiments/a2b_extract_tree_importance.py)
  cadence         the campaign's trees now refit every session: at an importance refit the
                  model is identical under either cadence (same window, same config); checks
                  the every-session run's forecast and native importance on those days against
                  the every-10 run's and this study's refit
  before_after    before / after rankings (first pass vs this root; + the no-mask rung)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
NOMASK = "--nomask" in sys.argv
if NOMASK:
    sys.argv.remove("--nomask")
MASK_ROOT = "results/feature_importance_1530_dedup"
NOMASK_ROOT = "results/feature_importance_1530_dedup_nomask"
FIRST_ROOT = REPO / "results" / "feature_importance_1530"
STORED = (
    REPO / MASK_ROOT / "_work" / "stored"
)  # pulled merged runs of the cadence campaign
os.environ.setdefault("FEATIMP_OUT", NOMASK_ROOT if NOMASK else MASK_ROOT)
os.environ["FEATIMP_WINDOW_MASK"] = "0" if NOMASK else "1"
# the stored 16:00 trees the refits are checked against: the cadence campaign's T10 (the
# shipped config, refit every 10 sessions) on the de-dup design, unmasked (agent B)
os.environ.setdefault("FEATIMP_TREES", str(STORED / "t10"))
# the per-bar linear arms of the de-dup campaign (agent A; the canonical yhat_sub_* source)
os.environ.setdefault(
    "FEATIMP_STORED_LIN", "results/linear_subsection_dedup/arms_hoffman2"
)
# the coefficient captures: made by this re-run (capture stage), all buckets in one place
os.environ.setdefault("FEATIMP_CAPTURES_MD", f"{MASK_ROOT}/_work/linear_captures")
# A's stored lasso arms ran on another machine: the warm-homotopy float path moves by CPU
# architecture (A's samearch_attribution.csv), so the capture-vs-stored gap is recorded
os.environ.setdefault("FEATIMP_LIN_STORED_GATE", "report")
os.environ.setdefault(
    "FEATIMP_TUNED_IMP",
    f"{MASK_ROOT}/_work/tuned_importance_{'nomask' if NOMASK else 'mask'}",
)
sys.path.insert(0, str(REPO / "experiments"))
from feature_importance_1530_trees import (  # noqa: E402
    BUCKETS,
    OUT,
    SEG,
    TREE_MODELS,
    TW,
    WORK,
    load_input,
    n_refits,
)

# the session-edge interactions of the pre-dedup per-bar design (src/backtest/executor.py
# _build_har_and_calendar(session_edge=...)): HAR ladder x the open / close gate
HAR_LAGS = (1, 5, 25, 125, 625, 3125)
SESSION_EDGE = tuple(f"har_ma_{k}_x_{g}" for g in ("open", "close") for k in HAR_LAGS)
P_EXPECTED = {"baseline": 22, "live_feasible": 232, "all_features": 628}


def _first(bucket: str) -> dict:
    z = np.load(FIRST_ROOT / "_work" / f"design_{bucket}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


# ============================================================================ gates
def gate_design() -> pd.DataFrame:
    rows = []
    for b in BUCKETS:
        z = np.load(WORK / f"design_{b}.npz", allow_pickle=True)
        names = [str(v) for v in z["names"]]
        o = _first(b)
        onames = [str(v) for v in o["names"]]
        gone = [c for c in onames if c not in names]
        new_only = [c for c in names if c not in onames]
        X, Xo = z["X"], o["X"]
        idx = [onames.index(c) for c in names]
        same_cols = bool(np.array_equal(X, Xo[:, idx]))
        edge_old = {c: Xo[:, onames.index(c)] for c in SESSION_EDGE if c in onames}
        W = int(z["W"])
        # at 16:00: x_open all zero, x_close = har_ma_k (bit for bit) on every row
        xo_zero = (
            all(not np.any(edge_old[f"har_ma_{k}_x_open"]) for k in HAR_LAGS)
            if edge_old
            else None
        )
        xc_copy = (
            all(
                np.array_equal(
                    edge_old[f"har_ma_{k}_x_close"], Xo[:, onames.index(f"har_ma_{k}")]
                )
                for k in HAR_LAGS
            )
            if edge_old
            else None
        )
        r = dict(
            bucket=b,
            p_new=len(names),
            p_expected=P_EXPECTED[b],
            p_old=len(onames),
            session_edge_columns_new=sum("_x_" in c for c in names),
            old_minus_new=" ".join(gone),
            old_minus_new_is_session_edge=sorted(gone) == sorted(SESSION_EDGE),
            new_minus_old=" ".join(new_only),
            rows_new=X.shape[0],
            rows_old=Xo.shape[0],
            window=W,
            oos_stamps_equal=bool(
                np.array_equal(
                    np.asarray(z["date_oos"]).astype(str),
                    np.asarray(o["date_oos"]).astype(str),
                )
            ),
            target_bitwise_equal=bool(np.array_equal(z["y"], o["y"])),
            other_columns_bitwise_equal=same_cols,
            old_x_open_all_zero=xo_zero,
            old_x_close_equal_har_ma=xc_copy,
        )
        r["ok"] = bool(
            r["p_new"] == r["p_expected"]
            and r["session_edge_columns_new"] == 0
            and r["old_minus_new_is_session_edge"]
            and not new_only
            and r["oos_stamps_equal"]
            and r["target_bitwise_equal"]
            and same_cols
        )
        rows.append(r)
        print(
            f"GATE design {b}: p {r['p_new']} (want {r['p_expected']}; old {r['p_old']}), "
            f"'_x_' columns {r['session_edge_columns_new']}, old - new = session-edge 12: "
            f"{r['old_minus_new_is_session_edge']}, stamps / target / other columns equal: "
            f"{r['oos_stamps_equal']} / {r['target_bitwise_equal']} / {same_cols}; old x_open all 0 "
            f"{xo_zero}, old x_close == har_ma {xc_copy} -> {'OK' if r['ok'] else 'FAIL'}",
            flush=True,
        )
    G = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    G.to_csv(OUT / "gates_design.csv", index=False)
    assert G["ok"].all(), G
    return G


def share() -> None:
    """The no-mask root reads the same inputs and linear walks as the mask root (copied,
    md5-checked): the design does not depend on the tree mask, nor do the linear models."""
    import hashlib

    assert NOMASK
    src = REPO / MASK_ROOT / "_work"
    files = [src / f"input_{b}.npz" for b in BUCKETS] + [
        src / f"design_{b}.npz" for b in BUCKETS
    ]
    files += sorted((src / "linear").glob("linear_*_*.npz"))
    files = [f for f in files if "_part" not in f.name]
    for f in files:
        dst = WORK / f.relative_to(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        a = hashlib.md5(f.read_bytes()).hexdigest()
        b = hashlib.md5(dst.read_bytes()).hexdigest()
        assert a == b, f
    print(f"shared {len(files)} files from {src} into {WORK} (md5 equal)", flush=True)


def tuned_extract() -> None:
    """The tuned trees' (random search, refit every 10) own importance extracts from the stored
    merged runs: the unmasked de-dup runs (agent B) for the no-mask root."""
    root = STORED / ("rs10" if NOMASK else "rs10_mask")
    out = Path(os.environ["FEATIMP_TUNED_IMP"])
    out = out if out.is_absolute() else REPO / out
    if not root.is_dir():
        print(
            f"no stored tuned runs at {root}; tuned-tree table left pending", flush=True
        )
        return
    subprocess.run(
        [
            sys.executable,
            str(REPO / "experiments" / "a2b_extract_tree_importance.py"),
            "--root",
            str(root),
            "--out",
            str(out),
        ],
        check=True,
    )


def cadence() -> pd.DataFrame:
    """At each importance refit k (first tail row i = 10 k) the every-session run (T1) fits the
    same 2000-session window with the same configuration as the every-10 run (T10) and as this
    study's refit, so its forecast on day i and its native importance must be the same: the
    importance recorded at every 10th refit IS the daily-refit model's importance on those days.
    Checked on the stored unmasked de-dup runs (agent B) against the no-mask refits, and on the
    masked runs when their merged npz files are in _work/stored/{t1,t10}_mask."""
    rows = []
    for variant, root in (("nomask", REPO / NOMASK_ROOT), ("mask", REPO / MASK_ROOT)):
        sfx = "" if variant == "nomask" else "_mask"
        for b in BUCKETS:
            for m in TREE_MODELS:
                rel = Path(b) / SEG / m / f"tw{TW}" / "causal_tune_trees" / m / b
                f1 = STORED / f"t1{sfx}" / rel / f"trees_{SEG}.npz"
                f10 = STORED / f"t10{sfx}" / rel / f"trees_{SEG}.npz"
                r: dict = dict(variant=variant, bucket=b, model=m)
                T = _chunks_at(root, b, m)
                if not f1.is_file():
                    r["note"] = (
                        f"every-session run not present ({f1.parent.relative_to(REPO)})"
                    )
                    rows.append(r)
                    continue
                z1 = np.load(f1, allow_pickle=True)
                meta1 = json.loads(str(z1["meta"]))
                K = n_refits(len(z1["date"]))
                ks = np.arange(K) * 10  # the importance refits' first tail rows
                p1 = np.asarray(z1["pred_adj"], float)[ks]
                imp1 = np.asarray(z1["importance"], float)
                imp1 = imp1[ks] if imp1.shape[0] == len(z1["date"]) else imp1
                r.update(
                    t1_refit_every=meta1.get("refit_every"),
                    t1_importance_every=meta1.get("importance_every"),
                    refits=K,
                    t1_importance_rows_finite=int(np.isfinite(imp1).all(axis=1).sum()),
                )
                if f10.is_file():
                    z10 = np.load(f10, allow_pickle=True)
                    p10 = np.asarray(z10["pred_adj"], float)[ks]
                    imp10 = np.asarray(z10["importance"], float)
                    r["t1_vs_t10_forecast_max_abs"] = float(np.max(np.abs(p1 - p10)))
                    r["t1_vs_t10_importance_max_abs"] = float(
                        np.max(np.abs(imp1 - imp10))
                    )
                if T is not None:
                    pr = T["pred"][ks]
                    mdi32 = (
                        T["mdi"].astype(np.float32).astype(float)
                    )  # the runs store float32
                    r["t1_vs_refit_forecast_max_abs"] = float(np.max(np.abs(p1 - pr)))
                    r["t1_vs_refit_importance_max_abs"] = float(
                        np.max(np.abs(imp1 - mdi32))
                    )
                    r["t1_vs_refit_importance_rel"] = float(
                        np.max(np.abs(imp1 - mdi32)) / max(np.max(np.abs(imp1)), 1e-300)
                    )
                rows.append(r)
    C = pd.DataFrame(rows)
    (REPO / MASK_ROOT).mkdir(parents=True, exist_ok=True)
    C.to_csv(REPO / MASK_ROOT / "gates_cadence.csv", index=False, float_format="%.3g")
    print(C.to_string(), flush=True)
    return C


def _chunks_at(root: Path, b: str, m: str) -> dict | None:
    """The fleet's refit chunks of (b, m) under a root, in refit order (the canary's short files
    left out, as the first pass's _tree_chunks); None if the fleet has not run there."""
    fs = sorted((root / "_work" / "trees").glob(f"trees_{b}_{m}_*_*.npz"))
    fleet = []
    for f in fs:
        a, e = (int(x) for x in f.stem.split("_")[-2:])
        if (b == "baseline" and (a, e) == (0, 147)) or (
            b != "baseline" and e - a == 21
        ):
            fleet.append((a, f))
    if not fleet:
        return None
    zs = [np.load(f, allow_pickle=True) for _, f in sorted(fleet)]
    out = {
        k: np.concatenate([z[k] for z in zs])
        for k in ("k", "pred", "mdi", "split", "pred_gap")
    }
    assert (out["k"] == np.arange(len(out["k"]))).all(), (root, b, m)
    return out


def canonical() -> pd.DataFrame:
    """The masked refits vs the canonical 16:00 tree tables (results/spxw_pnl/
    yhat_subtree_<model>_<bucket>.parquet; the masked T10 once agent H has rebuilt them).  A
    table still equal to the unmasked T10 run (agent B) bit for bit is reported as such."""
    rows = []
    for b in BUCKETS:
        for m in TREE_MODELS:
            f = REPO / "results" / "spxw_pnl" / f"yhat_subtree_{m}_{b}.parquet"
            r: dict = dict(bucket=b, model=m, table=str(f.relative_to(REPO)))
            if not f.is_file():
                r["note"] = "table not present"
                rows.append(r)
                continue
            t = pd.read_parquet(f)
            ts = (
                pd.DatetimeIndex(t["t"])
                .tz_convert("America/New_York")
                .tz_localize(None)
            )
            y = pd.Series(
                t["yhat"].to_numpy(float), index=ts.strftime("%Y-%m-%d %H:%M:%S")
            )
            d = load_input(b)
            y = y.reindex(d["date_oos"]).to_numpy(float)
            z10 = np.load(
                STORED
                / "t10"
                / b
                / SEG
                / m
                / f"tw{TW}"
                / "causal_tune_trees"
                / m
                / b
                / f"trees_{SEG}.npz",
                allow_pickle=True,
            )
            g10 = float(np.nanmax(np.abs(y - np.asarray(z10["pred_adj"], float))))
            r["table_vs_unmasked_t10_max_abs"] = g10
            # the table is written from the run's pred_adj (a float round trip): <= 1e-12 = the run
            r["table_equals_unmasked_t10"] = bool(g10 <= 1e-12)
            r["table_mtime"] = pd.Timestamp(f.stat().st_mtime, unit="s").strftime(
                "%Y-%m-%d %H:%M"
            )
            for variant, root in (
                ("mask", REPO / MASK_ROOT),
                ("nomask", REPO / NOMASK_ROOT),
            ):
                T = _chunks_at(root, b, m)
                if T is not None:
                    g = np.abs(T["pred"] - y)
                    r[f"refit_{variant}_vs_table_max_abs"] = float(np.nanmax(g))
                    r[f"refit_{variant}_vs_table_mean_abs"] = float(np.nanmean(g))
                    r[f"refit_{variant}_vs_table_corr"] = float(
                        np.corrcoef(T["pred"], y)[0, 1]
                    )
            rows.append(r)
    C = pd.DataFrame(rows)
    C.to_csv(REPO / MASK_ROOT / "gates_canonical.csv", index=False, float_format="%.3g")
    print(C.to_string(), flush=True)
    return C


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "gate_design":
        gate_design()
    elif stage == "share":
        share()
    elif stage == "tuned_extract":
        tuned_extract()
    elif stage == "cadence":
        cadence()
    elif stage == "canonical":
        canonical()
    elif stage == "before_after":
        import feature_importance_1530_dedup_compare as cmp

        cmp.main()
    else:  # the first pass's stages on this root
        sys.argv = [
            str(REPO / "experiments" / "feature_importance_1530.py")
        ] + sys.argv[1:]
        import runpy

        runpy.run_path(sys.argv[0], run_name="__main__")
