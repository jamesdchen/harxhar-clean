"""Pooled-bank tree walks on new input sets / estimators (user request 2026-10-07): the bank's OWN walk, unchanged.

The pooled tree expert bank (harxhar-clean/results/unification_carc/tree_expert_*; src/unification.py, kind "tree")
fits one frozen configuration per arm on the pooled b2 panel (``_load_panel``: prep_cache_all_features_b2.npz,
300,317 rows x 1,144 columns), refitting at EVERY bar on the trailing 24,000 bars (``_walk_tree``; TREE_REFIT_EVERY
unset on the CARC bank), 100 chunks of [24000, 300317) (experiments/run_unification.py ``chunk_bounds``), seed 42,
threads = SLURM_CPUS_PER_TASK (8 on the bank: harxhar-clean/jobs/slurm/unification_tree.sbatch).  This runner calls
the bank's ``compute()`` (persistence, alignment gates, npz schema) and ``_walk_tree`` UNCHANGED, with three additions
made here:

  * MENU   the bank reads experiments/tree_menu.json (lost); ``_load_tree_menu`` is replaced by a one-entry menu built
           from --config (a JSON with "family" lgbm | xgb | rf, "params", "name", "sha"); the arm is registered as
           ``pooltree_<est>_<set>`` (kind "tree", grid "0").
  * FOREST ``_walk_tree`` knows lgbm / xgb only; family "rf" is walked by ``walk_rf`` below -- the same loop (same rows,
           same cadence env, one prediction per bar) with sklearn's RandomForestRegressor(**params, random_state=42,
           n_jobs=SLURM_CPUS_PER_TASK) (a forest's trees do not depend on n_jobs).  The pooled window is the one the
           forest was tuned on, so no leaf scaling.
  * SET    the panel's design columns: all_features = every column (the bank's design); baseline = the pooled backbone
           (src.unification._backbone_cols: HAR rungs + session-edge interactions + calendar, 52 columns);
           lean_vol_vix = backbone + the bucket's value + indicator columns (_bucket_cols, 144) in that order, as the
           pooled linear bucket arms (``_bucket_arm``) order them.  The in-process panel's X / names are replaced by the
           subset BEFORE compute() runs (compute's tree branch fits F = p.X).

Chunk geometry: --chunk-index k (0..99) gives the bank's [start, end); --rows N limits the chunk to its first N rows
(the gate).  --pieces k:a:b,... runs row pieces of bank chunks in one process (refit every bar only: each row is
fitted on its own trailing window, so pieces concatenate to the unsplit chunk; experiments/pooled_trees_merge.py).  Output: <out>/<arm>/chunk_<kkk>.npz + .meta.json, the bank's schema (row_id, t, y_fit, yhat, rv_raw,
baseline, valid_mask, ..., meta with tree_config), so experiments/dump_unif_yhat.py dumps the arm unchanged.

Env: UNIFY_CACHE_DIR = the folder holding prep_cache_all_features_b2.npz (the bank's cache, read only);
SLURM_CPUS_PER_TASK = threads (8 = the bank's); TREE_REFIT_EVERY must be unset (the bank's cadence: every bar).
--refit-every K (default 1 = the bank's walk) sets the bank's own TREE_REFIT_EVERY mechanism (the Hoffman2 half of the
bank ran K = 10): a refit at chunk-relative rows 0, K, 2K, ..., each model forecasting the next K bars; the arm is then
named pooltree_<est>_<set>_r<K> and its meta records refit_every.  For forests too expensive to refit every bar.

Run:  python experiments/pooled_trees_walk.py --est xgb --set all_features --config cluster/xgbtop_xgb_config.json
          --chunk-index 0 [--rows 20] --out results/pooled_trees_2026-10-07/chunks
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import src.unification as U  # noqa: E402

WARMUP = 24_000
N_CHUNKS = 100
SETS = ("all_features", "baseline", "lean_vol_vix")


def chunk_bounds(index: int, n_rows: int) -> tuple[int, int]:
    """experiments/run_unification.py chunk_bounds (same expression)."""
    if not 0 <= index < N_CHUNKS:
        raise SystemExit(f"chunk-index {index} outside [0, {N_CHUNKS})")
    span = n_rows - WARMUP
    return WARMUP + (span * index) // N_CHUNKS, WARMUP + (
        span * (index + 1)
    ) // N_CHUNKS


def set_columns(names: list[str], name: str) -> np.ndarray:
    if name == "all_features":
        return np.arange(len(names))
    bb = U._backbone_cols(names)
    if name == "baseline":
        return bb
    if name == "lean_vol_vix":
        return np.concatenate([bb, U._bucket_cols(names, "lean_vol_vix")])
    raise SystemExit(f"unknown set {name!r}")


def walk_rf(F, y, window, lo, hi, params) -> np.ndarray:
    """src.unification._walk_tree's loop for a random forest (same rows, cadence, one prediction per bar)."""
    from sklearn.ensemble import RandomForestRegressor

    n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    Xs = np.ascontiguousarray(F[lo - window : hi])
    ys = np.ascontiguousarray(y[lo - window : hi])
    out = np.empty(hi - lo, dtype=np.float64)
    cadence = max(1, int(os.environ.get("TREE_REFIT_EVERY", "1")))
    model = None
    for i in range(hi - lo):
        t = window + i
        if model is None or i % cadence == 0:
            model = RandomForestRegressor(**params, random_state=42, n_jobs=n_threads)
            model.fit(Xs[i:t], ys[i:t])
            # predict single-threaded: sklearn sums the trees' predictions under a lock in thread-arrival order, so a
            # threaded predict is not bit-reproducible run to run (the first canary's repeat differed in the last
            # bits); the fit is (per-tree seeds drawn up front) and a one-thread predict sums in tree order
            model.set_params(n_jobs=1)
        out[i] = float(model.predict(Xs[t : t + 1])[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", required=True, choices=("lgbm", "xgb", "rf"))
    ap.add_argument("--set", required=True, choices=SETS)
    ap.add_argument(
        "--config",
        required=True,
        help="JSON with family / params / name / sha (or rf_params)",
    )
    ap.add_argument("--chunk-index", type=int, default=-1)
    ap.add_argument(
        "--pieces",
        default="",
        help="k:a:b[,k:a:b...] = rows [a, b) of bank chunk k (chunk-relative), one process",
    )
    ap.add_argument(
        "--rows", type=int, default=0, help="first N rows of the chunk only (gate)"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--arm", default="", help="arm name (default pooltree_<est>_<set>[_r<K>])"
    )
    ap.add_argument(
        "--refit-every",
        type=int,
        default=1,
        help="refit cadence K (1 = the bank's walk)",
    )
    a = ap.parse_args()
    if os.environ.get("TREE_REFIT_EVERY", "1") != "1":
        raise SystemExit(
            "TREE_REFIT_EVERY must be unset: the cadence is --refit-every (default 1, the CARC bank's)"
        )
    if a.refit_every < 1:
        raise SystemExit("--refit-every must be >= 1")
    os.environ["TREE_REFIT_EVERY"] = str(a.refit_every)
    with open(a.config, encoding="utf-8") as f:
        cfg = json.load(f)
    params = dict(cfg.get("params") or cfg.get("rf_params") or {})
    family = cfg.get("family") or a.est
    if family != a.est:
        raise SystemExit(f"{a.config}: family {family!r} != --est {a.est!r}")
    entry = {
        "name": cfg.get("name", "?"),
        "family": family,
        "params": params,
        "sha": cfg.get("sha", "?"),
        "source": a.config,
        "set": a.set,
    }
    arm = a.arm or f"pooltree_{a.est}_{a.set}" + (
        f"_r{a.refit_every}" if a.refit_every > 1 else ""
    )
    U._load_tree_menu = lambda: [
        entry
    ]  # the frozen menu, one entry (the bank's menu file is lost)
    U.ARMS[arm] = U.ArmSpec(
        describe=f"pooled tree walk {a.est} on {a.set} ({entry['name']})",
        kind="tree",
        grid="0",
    )
    orig = U._walk_tree

    def walk(F, y, window, lo, hi, prm, fam):
        return (
            walk_rf(F, y, window, lo, hi, prm)
            if fam == "rf"
            else orig(F, y, window, lo, hi, prm, fam)
        )

    U._walk_tree = walk
    p = U._load_panel()
    cols = set_columns(p.names, a.set)
    if a.set != "all_features":
        p.X = np.ascontiguousarray(p.X[:, cols])
        p.names = [p.names[j] for j in cols]
    n = len(p.y)
    if a.pieces:
        return run_pieces(a, arm, entry, params, p, n)
    if a.chunk_index < 0:
        raise SystemExit("--chunk-index or --pieces is required")
    lo, hi = chunk_bounds(a.chunk_index, n)
    if a.rows:
        hi = min(hi, lo + a.rows)
    out = Path(a.out) / arm / f"chunk_{a.chunk_index:03d}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"pooled walk {arm}: {entry['name']} sha {entry['sha']} params {params}; design {p.X.shape[1]} columns; "
        f"chunk {a.chunk_index} rows [{lo}, {hi}); threads {os.environ.get('SLURM_CPUS_PER_TASK', '1')}",
        flush=True,
    )
    t0 = time.time()
    U.compute(
        argparse.Namespace(
            arm=arm,
            chunk_start=lo,
            chunk_end=hi,
            halo=WARMUP,
            window=WARMUP,
            output_file=str(out),
        )
    )
    meta = {
        "arm": arm,
        "chunk_index": a.chunk_index,
        "chunk_start": lo,
        "chunk_end": hi,
        "n_rows_panel": n,
        "rows_limited": bool(a.rows),
        "set": a.set,
        "n_design_cols": int(p.X.shape[1]),
        "threads": int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        "refit_every": a.refit_every,
        "wall_sec": round(time.time() - t0, 1),
    }
    with open(str(out).replace(".npz", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(
        f"DONE {arm} chunk {a.chunk_index} [{lo},{hi}) {meta['wall_sec']}s -> {out}",
        flush=True,
    )
    return 0


def piece_path(out: str, arm: str, k: int, a: int, b: int) -> Path:
    return Path(out) / arm / "pieces" / f"chunk_{k:03d}_{a:04d}-{b:04d}.npz"


def _gate_med(p, lo: int, hi: int) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        y_check = np.sqrt(np.asarray(p.rv_raw[lo:hi]) / np.asarray(p.baseline[lo:hi]))
    return float(np.nanmedian(np.abs(np.asarray(p.y[lo:hi]) - y_check)))


def gate_window(p, lo: int, hi: int, n: int) -> tuple[int, int]:
    """The rows compute() must walk so that the bank's persistence alignment gate (``_assert_fit_raw_alignment``: the
    MEDIAN over the rows of |y_fit - sqrt(rv_raw / baseline)| must be exactly 0; winsorized rows break the identity) can
    pass.  Over a whole chunk it always does; a few-row piece lying inside a run of winsorized (1/99-tail) rows does
    not (Hoffman2 5-row pieces 2026-10-09: rows [295629, 295634) etc.) -- even though its forecasts are right.  Such a
    piece is computed inside the smallest symmetric widening [lo - e, hi + e) (e = 1, 2, 4, ...; within [WARMUP, n))
    that passes the same test, and only its own rows are kept (refit at every bar: a row's forecast does not depend on
    the window it is walked in).  Checked here on the panel BEFORE the walk, so no compute is wasted."""
    if (
        getattr(p, "asinh_scale", None) is not None
        or getattr(p, "slope_q_lo", None) is not None
    ):
        return lo, hi  # the plain-y identity only (every tree arm here)
    if _gate_med(p, lo, hi) == 0.0:
        return lo, hi
    e = 1
    while True:
        lo2, hi2 = max(WARMUP, lo - e), min(n, hi + e)
        if _gate_med(p, lo2, hi2) == 0.0:
            return lo2, hi2
        if lo2 == WARMUP and hi2 == n:
            raise SystemExit(
                f"rows [{lo}, {hi}): no window passes the alignment identity"
            )
        e *= 2


def slice_piece(path: Path, lo: int, hi: int, lo2: int, hi2: int) -> None:
    """Keep rows [lo, hi) of a compute() npz written for [lo2, hi2): the row arrays sliced, the chunk sufficient
    statistics recomputed on the kept rows (as compute() computes them), meta chunk_start / chunk_end set to the piece
    and the walked window recorded."""
    z = np.load(path, allow_pickle=True)
    m = hi2 - lo2
    i0, i1 = lo - lo2, hi - lo2
    arr = {}
    for k in z.files:
        v = z[k]
        arr[k] = v[i0:i1] if (np.ndim(v) >= 1 and len(v) == m) else v
    assert np.array_equal(arr["row_id"], np.arange(lo, hi)), "row ids after slicing"
    resid = arr["y_fit"] - arr["yhat"]
    arr.update(
        sqrt_sse=np.float64(float(np.sum(resid**2))),
        sqrt_mse=np.float64(float(np.mean(resid**2))),
        sum_y=np.float64(float(np.sum(arr["y_fit"]))),
        sum_y2=np.float64(float(np.sum(arr["y_fit"] ** 2))),
        n=np.int64(hi - lo),
    )
    meta = json.loads(str(z["meta"]))
    meta.update(chunk_start=lo, chunk_end=hi, walked_window=[lo2, hi2])
    arr["meta"] = json.dumps(meta)
    z.close()
    tmp2 = path.with_name(path.stem + ".slice.npz")
    np.savez_compressed(tmp2, **arr)
    os.replace(tmp2, path)


def run_pieces(a, arm, entry, params, p, n) -> int:
    """Rows [a, b) of bank chunk k for every k:a:b of --pieces, one after another in this process (the panel loaded
    once).  At refit-every-bar (K = 1) every row's model is fitted on its own trailing 24,000 bars, so a piece's rows
    are exactly the unsplit chunk's (experiments/pooled_trees_merge.py concatenates a chunk's pieces into the bank's
    chunk npz).  Output <out>/<arm>/pieces/chunk_<kkk>_<aaaa>-<bbbb>.npz + .meta.json; an existing piece is skipped."""
    if a.refit_every != 1:
        raise SystemExit(
            "--pieces requires --refit-every 1 (a cadence > 1 anchors its refits at the chunk start)"
        )
    done = 0
    for spec in a.pieces.split(","):
        k, r0, r1 = (int(x) for x in spec.split(":"))
        c0, c1 = chunk_bounds(k, n)
        if not (0 <= r0 < r1 <= c1 - c0):
            raise SystemExit(
                f"piece {spec}: rows [{r0}, {r1}) outside chunk {k} ({c1 - c0} rows)"
            )
        out = piece_path(a.out, arm, k, r0, r1)
        if out.is_file():
            print(f"skip (exists) {out}", flush=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".part.npz")
        t0 = time.time()
        lo, hi = c0 + r0, c0 + r1
        lo2, hi2 = gate_window(p, lo, hi, n)
        U.compute(
            argparse.Namespace(
                arm=arm,
                chunk_start=lo2,
                chunk_end=hi2,
                halo=WARMUP,
                window=WARMUP,
                output_file=str(tmp),
            )
        )
        if (lo2, hi2) != (lo, hi):
            slice_piece(tmp, lo, hi, lo2, hi2)
            print(
                f"  rows [{lo},{hi}) computed inside [{lo2},{hi2}) for the bank's alignment gate",
                flush=True,
            )
        os.replace(tmp, out)  # a piece appears only when complete
        meta = {
            "arm": arm,
            "chunk_index": k,
            "chunk_start": c0,
            "chunk_end": c1,
            "piece_rows": [r0, r1],
            "rows": [c0 + r0, c0 + r1],
            "n_rows_panel": n,
            "set": a.set,
            "n_design_cols": int(p.X.shape[1]),
            "threads": int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
            "refit_every": 1,
            "wall_sec": round(time.time() - t0, 1),
            "config": entry["name"],
            "sha": entry["sha"],
        }
        with open(str(out).replace(".npz", ".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        done += 1
        print(
            f"DONE {arm} chunk {k} rows [{r0},{r1}) {meta['wall_sec']}s ({(meta['wall_sec'] / (r1 - r0)):.2f} s/row) -> {out}",
            flush=True,
        )
    print(f"pieces done: {done}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
