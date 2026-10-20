"""Checks of experiments/pooled_trees_rerun_13.py on 3 trading days (2020-01-03 .. 2020-01-07), 2026-10-20.

  1. walk every arm with --refit day and --refit bar (--part 1/1), and with --part 1/2 + 2/2;
  2. the --refit day rows on the first row of each day equal the --refit bar rows there;
  3. --part 1/2 + 2/2 equal --part 1/1, every column, bit for bit;
  4. resume: a bar piece cut back to its first rows (as after a time limit) and walked again equals the whole walk;
  5. merge (--n 1) of each cadence; the schema matches the linear forecasts.parquet that Table 1 reads;
  6. experiments/score_13bar_span.py scores the merged trees beside the linear run.

Run from the repo root: python results/pooled_trees_rerun13_2026-10-20/test_3day/run_checks.py [--skip-walks]
--bar-parts-skip est/set (repeatable) leaves an arm out of the bar --part 1/2 + 2/2 walks (cost).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = REPO / "experiments" / "pooled_trees_rerun_13.py"
DESIGN = REPO / "results" / "pooled_trees_rerun13_2026-10-20"
LINEAR = REPO / "results" / "pooled_linear_rerun_2026-10-19_13" / "forecasts.parquet"
SPAN = ["--first-day", "2020-01-03", "--last-day", "2020-01-07"]
ARMS = [(e, s) for s in ("baseline", "all_features") for e in ("lgbm", "xgb", "rf")]
TAG = "2020-01-03_2020-01-07"


def walk(est, s, refit, part, extra=()):
    cmd = [sys.executable, str(RUN), "walk", "--design", str(DESIGN), "--out", str(HERE), "--est", est, "--set", s,
           "--refit", refit, "--part", part, "--threads", "4", *SPAN, *extra]
    subprocess.run(cmd, check=True, cwd=REPO)


def piece(est, s, refit, i, n):
    return pd.read_parquet(HERE / "pieces" / f"{est}_{s}_{refit}_{TAG}_{i}of{n}.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-walks", action="store_true")
    ap.add_argument("--bar-parts-skip", action="append", default=[])
    a = ap.parse_args()
    if not a.skip_walks:
        for est, s in ARMS:
            for refit in ("day", "bar"):
                parts = ["1/1"] if refit == "bar" and f"{est}/{s}" in a.bar_parts_skip else ["1/1", "1/2", "2/2"]
                for p in parts:
                    walk(est, s, refit, p)
    out = {}
    for est, s in ARMS:
        day, bar = piece(est, s, "day", 1, 1), piece(est, s, "bar", 1, 1)
        assert len(day) == len(bar) == 39 and day["date"].equals(bar["date"])
        first = day.groupby(day["date"].dt.normalize())["date"].transform("min") == day["date"]
        assert first.sum() == 3
        same_first = day[first].reset_index(drop=True).equals(bar[first].reset_index(drop=True))
        # day: one fit a day (fit_row of the day's first row); bar: one fit a row
        assert day.groupby(day["date"].dt.normalize())["fit_row"].nunique().eq(1).all()
        assert bar["fit_row"].is_unique and (bar["fit_row"].diff().dropna() == 1).all()
        rest = ~first.to_numpy()
        diff_rest = float(np.max(np.abs(day["pred_adj"].to_numpy()[rest] - bar["pred_adj"].to_numpy()[rest])))
        parts = {}
        for refit in ("day", "bar"):
            if (HERE / "pieces" / f"{est}_{s}_{refit}_{TAG}_2of2.parquet").is_file():
                two = pd.concat([piece(est, s, refit, 1, 2), piece(est, s, refit, 2, 2)], ignore_index=True)
                one = piece(est, s, refit, 1, 1)
                parts[refit] = bool(two.equals(one))
        out[f"{est}/{s}"] = dict(day_first_rows_equal_bar=bool(same_first), parts_1of1_eq_1of2_2of2=parts,
                                 max_abs_day_minus_bar_other_rows=diff_rest)
        print(est, s, out[f"{est}/{s}"])
    # resume
    if not a.skip_walks:
        p = HERE / "pieces" / f"lgbm_baseline_bar_{TAG}_1of1.parquet"
        full = pd.read_parquet(p)
        (HERE / "resume").mkdir(exist_ok=True)
        q = HERE / "resume" / "pieces"
        q.mkdir(exist_ok=True)
        full.iloc[:10].to_parquet(q / p.name, index=False)
        subprocess.run([sys.executable, str(RUN), "walk", "--design", str(DESIGN), "--out", str(HERE / "resume"),
                        "--est", "lgbm", "--set", "baseline", "--refit", "bar", "--threads", "4", *SPAN],
                       check=True, cwd=REPO)
        again = pd.read_parquet(q / p.name)
        meta = json.loads((q / p.name).with_suffix(".json").read_text())
        out["resume_lgbm_baseline_bar"] = dict(equal=bool(again.equals(full)), refits_after_resume=meta["refits"])
        print("resume", out["resume_lgbm_baseline_bar"])
    # merge and schema
    lin = pq.read_schema(LINEAR)
    cols = ["model", "predictor_set", "date", "pred_adj", "true_adj", "B", "plain", "rv_actual"]
    for refit in ("day", "bar"):
        dest = HERE / f"forecasts_{refit}.parquet"
        subprocess.run([sys.executable, str(RUN), "merge", "--design", str(DESIGN), "--out", str(HERE), "--refit",
                        refit, "--n", "1", "--dest", str(dest), "--require-all", *SPAN], check=True, cwd=REPO)
        sch = pq.read_schema(dest)
        same = sch.names == cols and all(sch.field(c).type == lin.field(c).type for c in cols)
        f = pd.read_parquet(dest)
        out[f"merge_{refit}"] = dict(rows=len(f), arms=int(f.groupby(["model", "predictor_set"]).ngroups),
                                     schema_matches_linear=bool(same),
                                     types={c: str(sch.field(c).type) for c in cols})
        print("merge", refit, out[f"merge_{refit}"])
    # the linear run's rows on the same dates: same true_adj up to the CSV round trip, same B and rv_actual
    lf = pd.read_parquet(LINEAR, columns=cols)
    lf = lf[(lf["model"] == "ols")].set_index("date")
    f = pd.read_parquet(HERE / "forecasts_bar.parquet")
    for (m, s), g in f.groupby(["model", "predictor_set"]):
        g = g.set_index("date")
        L = lf.reindex(g.index)
        out.setdefault("against_linear_rows", {})[f"{m}/{s}"] = dict(
            B_equal=bool(np.array_equal(g["B"], L["B"])), rv_equal=bool(np.array_equal(g["rv_actual"], L["rv_actual"])),
            true_adj_max_rel_diff=float(np.max(np.abs(g["true_adj"] / L["true_adj"] - 1))))
    print(out["against_linear_rows"])
    (HERE / "checks.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
