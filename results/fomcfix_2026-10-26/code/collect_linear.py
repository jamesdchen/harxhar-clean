"""Compact linear forecasts of a cell of results/fomcfix_2026-10-26 (RULE.txt), on CARC, read-only on the walks.

Stitches each model's walk table (chunks in order: experiments/pooled_linear_rerun_2026_10_19_13.walk_table) and writes
one parquet: model, predictor_set, date, pred_adj, true_adj, B (the design's b_oos of the row), the rows from the
first 2013-12-27 16:00 row on (the final run's walk start; holiday cells start at their own boundary).

  main     <out dir> <dest parquet> [--compare <reference forecasts.parquet> --gate <json>]
           every set/model walked under <out dir>/_work/walk; --compare: gate L, pred_adj and true_adj bit for bit
           against the reference on every row of each model present in both (and the row sets equal)
  annual   <annual root> <dest parquet> --years 2014-2024
           the year walks of RULE.txt 7(b): <root>/y<Y>/_work/walk/<set>/<model>/y<Y>/, the rows of year Y kept, years
           stitched; the year's penalty from <root>/y<Y>/frozen_penalties.json in column alpha
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE / "experiments"))
import pooled_linear_rerun_2026_10_19_13 as O  # noqa: E402

MODELS = [(s, m) for s in O.SETS for m in O.EST] + [("baseline", "ols")]


def b_of(work: Path, s: str) -> pd.Series:
    d = pd.read_csv(work / f"design_{s}" / "dates_oos.csv", parse_dates=["date"])["date"]
    return pd.Series(np.load(work / f"design_{s}" / "b_oos.npy"), index=pd.DatetimeIndex(d))


def table(paths: list[Path]) -> pd.DataFrame:
    r = pd.concat([pd.read_csv(p, parse_dates=["date"]) for p in paths], ignore_index=True)
    assert r["date"].is_monotonic_increasing and r["date"].is_unique, paths[0]
    return r


def frame(r: pd.DataFrame, s: str, m: str, B: pd.Series) -> pd.DataFrame:
    b = B.reindex(pd.DatetimeIndex(r["date"])).to_numpy()
    assert np.isfinite(b).all(), (s, m)
    return pd.DataFrame(
        dict(model=m, predictor_set=s, date=r["date"], pred_adj=r["pred_adj"], true_adj=r["true_adj"], B=b)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("main", "annual"))
    ap.add_argument("src")
    ap.add_argument("dest")
    ap.add_argument("--compare", default="")
    ap.add_argument("--gate", default="")
    ap.add_argument("--years", default="2014-2024")
    a = ap.parse_args()
    src = Path(a.src).resolve()
    out = []
    if a.mode == "main":
        O.WORK = src / "_work"
        B = {s: b_of(O.WORK, s) for s in O.SETS}
        for s, m in MODELS:
            if not O.walk_dir(s, m).is_dir():
                continue
            r = O.walk_table(s, m)
            out.append(frame(r, s, m, B[s]))
            print(f"{s}/{m}: {len(r)} rows {r['date'].iloc[0]} .. {r['date'].iloc[-1]}", flush=True)
    else:
        y0, y1 = (int(v) for v in a.years.split("-"))
        for Y in range(y0, y1 + 1):
            root = src / f"y{Y}"
            fr = json.loads((root / "frozen_penalties.json").read_text())
            B = {s: b_of(root / "_work", s) for s in O.SETS}
            for s, m in MODELS:
                if m == "ols":
                    continue
                ps = sorted((root / "_work" / "walk" / s / m / f"y{Y}").glob("causal_tune_linear/*/*/results.csv"))
                assert len(ps) == 1, (Y, s, m, ps)
                r = table(ps)
                r = r[r["date"].dt.year == Y]
                f = frame(r, s, m, B[s])
                f["alpha"] = float(fr[m][s])
                f["year"] = Y
                out.append(f)
                print(f"y{Y} {s}/{m}: alpha {fr[m][s]!r}, {len(r)} rows", flush=True)
    fc = pd.concat(out, ignore_index=True)
    fc["date"] = pd.to_datetime(fc["date"]).astype("datetime64[ns]")
    Path(a.dest).parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(a.dest, index=False)
    print(f"wrote {a.dest}: {len(fc)} rows")
    if a.compare:
        ref = pd.read_parquet(a.compare, columns=["model", "predictor_set", "date", "pred_adj", "true_adj"])
        rec: dict = {"reference": a.compare, "models": {}, "fail": []}
        for (s, m), g in fc.groupby(["predictor_set", "model"]):
            q = ref[(ref["model"] == m) & (ref["predictor_set"] == s)].sort_values("date")
            g = g.sort_values("date")
            same_rows = pd.DatetimeIndex(g["date"]).equals(pd.DatetimeIndex(q["date"]))
            eq_p = same_rows and np.array_equal(g["pred_adj"].to_numpy(), q["pred_adj"].to_numpy())
            eq_t = same_rows and np.array_equal(g["true_adj"].to_numpy(), q["true_adj"].to_numpy())
            mx = float(np.max(np.abs(g["pred_adj"].to_numpy() - q["pred_adj"].to_numpy()))) if same_rows else None
            rec["models"][f"{s}/{m}"] = dict(
                rows=len(g), rows_ref=len(q), same_rows=same_rows, pred_adj_equal=eq_p, true_adj_equal=eq_t, max_abs=mx
            )
            if not (eq_p and eq_t):
                rec["fail"].append(f"{s}/{m}")
        rec["pass"] = not rec["fail"]
        Path(a.gate).write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec, indent=1))
        return 0 if rec["pass"] else 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
