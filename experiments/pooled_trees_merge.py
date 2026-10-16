"""Merge the row pieces of the pooled tree walks (2026-10-07) into the bank's chunk files.

experiments/pooled_trees_walk.py --pieces writes rows [a, b) of bank chunk k as
<root>/<arm>/pieces/chunk_<kkk>_<aaaa>-<bbbb>.npz.  At refit-every-bar every row's model is fitted on its own trailing
24,000 bars, so the pieces of a chunk, concatenated in row order, ARE the unsplit chunk.  For every chunk whose pieces
tile [0, chunk length) exactly (no gap, no overlap; checked on row_id), this writes <root>/<arm>/chunk_<kkk>.npz in the
bank's schema (row_id, t, y_fit, yhat, rv_raw, baseline, valid_mask, the sufficient statistics recomputed over the
chunk, meta = the first piece's with chunk_start / chunk_end set to the chunk's and the piece list added) and its
.meta.json (wall_sec = the pieces' sum, the pieces' hosts).  A chunk whose npz exists is left alone (a chunk computed
whole, e.g. by a fleet task, is never overwritten).  experiments/dump_unif_yhat.py then reads the arm as a bank arm.

Run:  python experiments/pooled_trees_merge.py --root results/pooled_trees_2026-10-07/chunks [--arm pooltree_rf_baseline]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pooled_trees_walk import N_CHUNKS, WARMUP, chunk_bounds  # noqa: E402

N_ROWS_PANEL = 300317
PIECE = re.compile(r"^chunk_(\d{3})_(\d{4})-(\d{4})\.npz$")
ROW_KEYS = ("row_id", "t", "y_fit", "yhat", "rv_raw", "baseline", "valid_mask")


def merge_arm(arm_dir: Path) -> tuple[int, int, list[str]]:
    pieces: dict[int, list[tuple[int, int, Path]]] = {}
    for f in (arm_dir / "pieces").glob("chunk_*.npz"):
        m = PIECE.match(f.name)
        if m:
            pieces.setdefault(int(m.group(1)), []).append(
                (int(m.group(2)), int(m.group(3)), f)
            )
    written, waiting, problems = 0, 0, []
    for k in range(N_CHUNKS):
        out = arm_dir / f"chunk_{k:03d}.npz"
        if out.is_file() or k not in pieces:
            continue
        c0, c1 = chunk_bounds(k, N_ROWS_PANEL)
        ps = sorted(pieces[k])
        edges = [0]
        for a, b, _ in ps:
            if a != edges[-1]:
                break
            edges.append(b)
        if edges[-1] != c1 - c0 or len(edges) != len(ps) + 1:
            waiting += 1
            continue
        zs = [np.load(f, allow_pickle=True) for _, _, f in ps]
        arr = {
            key: np.concatenate([np.asarray(z[key]) for z in zs]) for key in ROW_KEYS
        }
        if not np.array_equal(arr["row_id"], np.arange(c0, c1, dtype=np.int64)):
            problems.append(
                f"{arm_dir.name} chunk {k}: row ids do not tile [{c0}, {c1})"
            )
            continue
        metas = [json.loads(str(z["meta"])) for z in zs]
        fixed = {
            key: metas[0].get(key)
            for key in (
                "arm",
                "kind",
                "window_bars",
                "grid",
                "n_design_cols",
                "tree_config",
            )
        }
        if any({key: mm.get(key) for key in fixed} != fixed for mm in metas):
            problems.append(f"{arm_dir.name} chunk {k}: the pieces' meta disagree")
            continue
        meta = dict(metas[0])
        meta.update(chunk_start=c0, chunk_end=c1, pieces=[[a, b] for a, b, _ in ps])
        resid = arr["y_fit"] - arr["yhat"]
        np.savez_compressed(
            out.with_name(out.stem + ".part.npz"),
            **arr,
            sqrt_sse=np.float64(np.sum(resid**2)),
            sqrt_mse=np.float64(np.mean(resid**2)),
            sum_y=np.float64(np.sum(arr["y_fit"])),
            sum_y2=np.float64(np.sum(arr["y_fit"] ** 2)),
            n=np.int64(c1 - c0),
            ols_dropped_cols=np.int64(0),
            ols_collinear_cols=np.int64(0),
            ols_masked_cols=np.int64(0),
            meta=json.dumps(meta),
        )
        out.with_name(out.stem + ".part.npz").replace(out)
        pm = []
        for _, _, f in ps:
            mj = f.with_name(f.stem + ".meta.json")
            nd = f.with_name(f.stem + ".node")
            d = json.loads(mj.read_text(encoding="utf-8")) if mj.is_file() else {}
            node = (
                nd.read_text(encoding="utf-8").replace("\n", " | ").strip()
                if nd.is_file()
                else ""
            )
            pm.append(
                {
                    "rows": d.get("piece_rows"),
                    "wall_sec": d.get("wall_sec"),
                    "threads": d.get("threads"),
                    "node": node,
                }
            )
        with open(out.with_name(out.stem + ".meta.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "arm": arm_dir.name,
                    "chunk_index": k,
                    "chunk_start": c0,
                    "chunk_end": c1,
                    "n_rows_panel": N_ROWS_PANEL,
                    "merged_from_pieces": len(ps),
                    "wall_sec": round(sum(x["wall_sec"] or 0 for x in pm), 1),
                    "pieces": pm,
                },
                fh,
                indent=1,
            )
        written += 1
    return written, waiting, problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--arm", default="")
    a = ap.parse_args()
    root = Path(a.root)
    arms = (
        [root / a.arm]
        if a.arm
        else sorted(d for d in root.glob("pooltree_*") if (d / "pieces").is_dir())
    )
    bad = []
    for d in arms:
        w, wait, probs = merge_arm(d)
        n_done = len(list(d.glob("chunk_*.npz")))
        print(
            f"{d.name}: merged {w} chunks now; {n_done} of {N_CHUNKS} chunks complete; {wait} chunks waiting for pieces"
            + (f"; PROBLEMS {probs}" if probs else "")
        )
        bad += probs
    assert WARMUP == 24000
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
