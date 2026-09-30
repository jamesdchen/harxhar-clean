"""CPU-class evidence for the masked 16:00 re-run (checklist I9).

Forecasts are bit-reproducible only within a CPU vector class: the executor's design moves at
~1e-11 between AVX-512 (xeon-4116) and AVX2 (epyc) nodes, and LightGBM's histogram bins can flip
on it (agents D / F / G, 2026-09-29).  Every masked fleet task is pinned to one class
(cluster/slurm/submit_trees_mask.sh, --constraint, default epyc-7513) and every finished chunk
records its node in NODE.  This script writes, under --out:

  class_census.csv  one row per (root, rung, class): chunks finished there (from NODE; a chunk
                    without NODE is "unrecorded") -- which class every rung ran on
  class_chunks.csv  the canaries that ran BEFORE the pinning, moved aside to c<k>_canary by the
                    fleet stage, against the same chunk re-run on the pinned class: rows, max
                    |diff| and max relative diff of pred_adj (and the QLIKE-rule forecasts where
                    written), bit-identical or not, both classes
  class_arms.csv    the class gate proper: the live_feasible LightGBM T10 and T1 arms run again on
                    the other class (results/linear_subsection_trees_mask_xeon, submit mode
                    classgate) against the pinned fleet's merged arms: rows, max |diff|, max
                    relative diff, bit-identical (the QLIKE / Sharpe difference on the deck days
                    is scored by experiments/trees_mask_ladder_1600.py from the stacked tables)

Usage: python experiments/trees_mask_class_gate.py [--out results/linear_subsection_trees_mask/class]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROOTS = {
    "untuned": ROOT / "results" / "linear_subsection_trees_mask",
    "tuned": ROOT / "results" / "linear_subsection_trees_tuned_mask",
    "lstm": ROOT / "results" / "linear_subsection_lstm_mask",
    "lstm10": ROOT / "results" / "linear_subsection_lstm_mask_re10",
    "xeon": ROOT / "results" / "linear_subsection_trees_mask_xeon",
}
SEG = "bar1600"


def node_class(d: Path) -> str:
    f = d / "NODE"
    if not f.is_file():
        return "unrecorded"
    text = f.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"AvailableFeatures=(\S+)", text)
    if not m:
        return "unknown"
    feats = m.group(1).split(",")
    hit = [c for c in feats if c.startswith(("epyc", "xeon"))]
    return hit[0] if hit else feats[0]


SIDE_DIRS = ("_canary", "_alone", "_repeat")  # moved-aside canaries and check re-runs


def fleet_chunks(base: Path) -> list[Path]:
    """The fleet's finished chunk dirs under base (not the side dirs, not the gates' scratch)."""
    return [
        d.parent
        for d in sorted(base.glob("**/chunks/c*/DONE"))
        if not d.parent.name.endswith(SIDE_DIRS)
        and "gates" not in d.relative_to(base).parts
    ]


def results_csv(chunk: Path, stem: str) -> Path | None:
    got = sorted(chunk.glob(f"causal_tune_*/*/*/{stem}_{SEG}.csv"))
    return got[0] if got else None


def diff(a: pd.Series, b: pd.Series) -> dict:
    a, b = a.to_numpy(float), b.to_numpy(float)
    rel = np.abs(a / b - 1.0)
    return {
        "rows": len(a),
        "bit_identical": bool(np.array_equal(a, b)),
        "max_abs": float(np.max(np.abs(a - b))) if len(a) else np.nan,
        "max_rel": float(np.max(rel)) if len(a) else np.nan,
        "median_rel": float(np.median(rel)) if len(a) else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/linear_subsection_trees_mask/class")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    census, chunks = [], []
    for kind, root in ROOTS.items():
        if not root.is_dir():
            continue
        for d in fleet_chunks(root):
            rel = d.relative_to(root).parts
            rung = rel[0] if kind in ("untuned", "tuned", "xeon") else kind
            census.append(
                {
                    "root": kind,
                    "rung": rung,
                    "chunk": d.name,
                    "class": node_class(d),
                    "dir": str(d.relative_to(ROOT)),
                }
            )
        for side in sorted(root.glob("**/chunks/c*_canary")):
            twin = side.with_name(side.name.removesuffix("_canary"))
            if not (twin / "DONE").is_file():
                continue
            row: dict = {
                "root": kind,
                "chunk_dir": str(twin.relative_to(ROOT)),
                "class_canary": node_class(side),
                "class_fleet": node_class(twin),
            }
            for stem in ("results", "results_qsel"):
                pa, pb = results_csv(side, stem), results_csv(twin, stem)
                if pa is None or pb is None:
                    continue
                ra = pd.read_csv(pa, float_precision="round_trip")
                rb = pd.read_csv(pb, float_precision="round_trip")
                same_dates = (
                    ra["date"].astype(str).tolist() == rb["date"].astype(str).tolist()
                )
                dd: dict = (
                    diff(ra["pred_adj"], rb["pred_adj"]) if same_dates else {"rows": 0}
                )
                row |= {f"{stem}_{k}": v for k, v in dd.items()} | {
                    f"{stem}_same_dates": same_dates
                }
            chunks.append(row)
    cz = pd.DataFrame(census)
    if len(cz):
        cz.groupby(["root", "rung", "class"]).size().rename(
            "chunks"
        ).reset_index().to_csv(out / "class_census.csv", index=False)
    pd.DataFrame(chunks).to_csv(out / "class_chunks.csv", index=False)

    arms = []
    for rung in ("t10", "t1"):
        for m, b in (("lgbm", "live_feasible"),):
            tail = (
                Path(rung)
                / b
                / SEG
                / m
                / "tw2000"
                / "causal_tune_trees"
                / m
                / b
                / f"results_{SEG}.csv"
            )
            px, pe = ROOTS["xeon"] / tail, ROOTS["untuned"] / tail
            row = {
                "rung": rung.upper(),
                "model": m,
                "bucket": b,
                "xeon_arm": px.is_file(),
                "fleet_arm": pe.is_file(),
            }
            if px.is_file() and pe.is_file():
                rx = pd.read_csv(px, float_precision="round_trip")
                re_ = pd.read_csv(pe, float_precision="round_trip")
                same = (
                    rx["date"].astype(str).tolist() == re_["date"].astype(str).tolist()
                )
                row["same_dates"] = same
                if same:
                    row |= diff(re_["pred_adj"], rx["pred_adj"])
                xc = sorted(
                    {
                        node_class(d)
                        for d in fleet_chunks(ROOTS["xeon"] / rung / b / SEG / m)
                    }
                )
                ec = sorted(
                    {
                        node_class(d)
                        for d in fleet_chunks(ROOTS["untuned"] / rung / b / SEG / m)
                    }
                )
                row |= {"class_alt_run": ";".join(xc), "class_fleet": ";".join(ec)}
            arms.append(row)
    pd.DataFrame(arms).to_csv(out / "class_arms.csv", index=False)
    print(
        pd.DataFrame(chunks).to_string(index=False)
        if chunks
        else "no moved-aside canary chunks"
    )
    print(pd.DataFrame(arms).to_string(index=False))
    if len(cz):
        print(cz.groupby(["root", "rung", "class"]).size().to_string())
    print(f"wrote {out}/class_census.csv, class_chunks.csv, class_arms.csv")


if __name__ == "__main__":
    main()
