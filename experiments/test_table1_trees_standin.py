"""Test of Table 1's --trees option without the real tree forecasts (2026-10-20).

Builds a stand-in for results/pooled_trees_rerun13_2026-10-20/w1000/forecasts.parquet from the linear forecasts of
Table 1: the ridge rows of each predictor set, copied as lgbm, xgb and rf, with the forecast scaled by a small
deterministic factor (exp of a sine of the row number, at most 5%, a different phase for each model). The numbers it
gives are NOT the trees' numbers. Then:

  1. runs writeup/make_rehaul_oos_rerun.py --trees <stand-in> into <OUT>/generated and <OUT>/res;
  2. checks that the linear rows of metrics.csv, dm.csv and subperiods.csv equal the committed CSVs of Table 1 exactly,
     that every line of the committed table and macro file (the notes line and the header comments aside) is in the new
     files unchanged, and that the tree rows are present;
  3. removes one bar from one stand-in series and checks that the generator stops (a tree bar drop is reported);
  4. with --compile: copies writeup/ to <OUT>/paper, puts the two new .tex files in its generated/, and compiles
     paper_rehaul.tex with latexmk.

Nothing is written inside the repository. Run:
  python experiments/test_table1_trees_standin.py --out <scratch directory> [--compile]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
LIN = REPO / "results" / "pooled_linear_rerun_2026-10-20_13_2012_w1000" / "forecasts.parquet"
RES = REPO / "results" / "oos_span_check_2026-10-20" / "table1_2012_w1000"
GEN = REPO / "writeup" / "generated"
GENERATOR = REPO / "writeup" / "make_rehaul_oos_rerun.py"
TREES = ["lgbm", "xgb", "rf"]


def standin(path: Path, drop_one: bool = False) -> None:
    g = pd.read_parquet(LIN)
    parts = []
    for j, m in enumerate(TREES):
        f = g[g["model"] == "ridge"].copy()
        k = np.arange(len(f), dtype=float)
        scale = np.exp(0.05 * np.sin(0.37 * k + 1.3 * (j + 1)))  # forecast scale, within exp(+-0.05)
        f["model"] = m
        f["plain"] = f["plain"] * scale
        f["pred_adj"] = f["pred_adj"] * np.sqrt(scale)  # plain = pred_adj^2 x B
        parts.append(f)
    out = pd.concat(parts, ignore_index=True)
    if drop_one:  # one bar of the span missing from one series
        i = out.index[(out["model"] == "xgb") & (out["predictor_set"] == "all_features")
                      & (pd.to_datetime(out["date"]) == pd.Timestamp("2016-06-01 11:00"))]
        assert len(i) == 1, len(i)
        out = out.drop(index=i)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def run(trees: Path, gen: Path, res: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GENERATOR), "--trees", str(trees), "--out-dir", str(gen), "--res-dir", str(res)],
        capture_output=True, text=True,
    )


def same_rows(new: pd.DataFrame, old: pd.DataFrame, name: str) -> None:
    lin = new[new["model"].isin(old["model"].unique())].reset_index(drop=True)
    pd.testing.assert_frame_equal(lin, old.reset_index(drop=True), check_exact=True, obj=name)
    print(f"ok  {name}: the {len(old)} linear rows are unchanged")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="scratch directory (outside the repository)")
    ap.add_argument("--compile", action="store_true", help="also compile paper_rehaul.tex in a copy of writeup/")
    a = ap.parse_args()
    out = a.out.resolve()
    assert REPO not in out.parents and out != REPO, "write the stand-in outside the repository"

    # 1. stand-in and run
    fc = out / "standin" / "forecasts.parquet"
    standin(fc)
    r = run(fc, out / "generated", out / "res")
    (out / "run.log").write_text(r.stdout + r.stderr)
    assert r.returncode == 0, r.stderr[-3000:]

    # 2. the linear rows are unchanged
    for name in ("metrics.csv", "dm.csv", "subperiods.csv"):
        new, old = pd.read_csv(out / "res" / name), pd.read_csv(RES / name)
        same_rows(new, old, name)
        if name != "subperiods.csv":
            assert set(new["model"]) - set(old["model"]) == set(TREES), (name, sorted(set(new["model"])))
    for name in ("rehaul_results_oos.tex", "rehaul_oos_rerun_numbers.tex"):
        new = (out / "generated" / name).read_text().splitlines()
        old = (GEN / name).read_text().splitlines()
        changed = [x for x in old if x not in new]
        # the notes line (the tree sentence, "and tree ensembles") is the only committed line that may change
        assert all(x.startswith("\\item \\textit{Notes:}") for x in changed), changed
        added = [x for x in new if x not in old]
        print(f"ok  {name}: {len(old) - len(changed)} of {len(old)} committed lines unchanged, {len(added)} added")
    tab = (out / "generated" / "rehaul_results_oos.tex").read_text()
    for row in ("LightGBM &", "XGBoost &", "Random forest &"):
        assert tab.count(row) == 1, row
    assert "Tree ensembles are left out" not in tab and "1998--2002 pre-sample" in tab

    # 3. a tree series missing a bar stops the generator
    standin(out / "standin_drop" / "forecasts.parquet", drop_one=True)
    r = run(out / "standin_drop" / "forecasts.parquet", out / "generated_drop", out / "res_drop")
    assert r.returncode != 0 and "drop 1 of the" in r.stderr, r.stderr[-2000:]
    print("ok  a tree series missing one bar stops the generator:", r.stderr.strip().splitlines()[-1])

    # 4. compile
    if a.compile:
        paper = out / "paper"
        if paper.exists():
            shutil.rmtree(paper)
        shutil.copytree(REPO / "writeup", paper, ignore=shutil.ignore_patterns("*.pdf", "*.zip", "archive", "overleaf*"))
        for name in ("rehaul_results_oos.tex", "rehaul_oos_rerun_numbers.tex"):
            shutil.copy(out / "generated" / name, paper / "generated" / name)
        r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "paper_rehaul.tex"],
                           cwd=paper, capture_output=True, text=True)
        (out / "compile.log").write_text(r.stdout + r.stderr)
        assert r.returncode == 0, f"compile failed, see {out / 'compile.log'}"
        print(f"ok  paper_rehaul.tex compiles: {paper / 'paper_rehaul.pdf'}")


if __name__ == "__main__":
    main()
