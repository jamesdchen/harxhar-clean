"""Score 13-bar rerun forecasts as Table 1 does, on any span and across runs (2026-10-20).

Table 1 (writeup/make_rehaul_oos_rerun.py) scores every bar of the 13-bar sample on 2020-01-03 .. 2024-04-30:
QLIKE and MSE on the actual RV of the bar, R2_OOS against the expanding mean of the same clock slot, and
Diebold-Mariano (HLN) tests on daily average losses. This script applies the same scoring to the window check
(1,000-day window) and the 2012 reruns, on the full span and subperiods, and adds tests between runs.

Usage
  python experiments/score_13bar_span.py --run w2000=results/pooled_linear_rerun_2026-10-19_13 \
      --run w1000=results/pooled_linear_rerun_2026-10-20_13_w1000 \
      --span 2020-01-03:2024-04-30 --compare w1000:w2000 --out results/oos_span_check_2026-10-20/w1000_check

Every run of a call is scored on one common set of bars: on an evaluation day of the span (a positive 16:00
print, early closes dropped), a positive RV, and a finite positive forecast of every model of every run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "writeup"))
sys.path.insert(0, str(REPO))
import make_rehaul_oos_rerun as T1  # noqa: E402  (core, slot_benchmark, loss, daily, EARLY)
from src.evaluation.diebold_mariano import dm_test  # noqa: E402

MODELS = [m for m, _ in T1.MODELS]
SETS = [s for s, _ in T1.SETS]
KEYS = [(m, s) for m in MODELS for s in SETS if m != "ols" or s == "baseline"]


def load(path: Path) -> dict[tuple[str, str], pd.DataFrame]:
    g = pd.read_parquet(
        path / "forecasts.parquet",
        columns=["model", "predictor_set", "date", "plain", "rv_actual"],
    )
    out = {}
    for (m, s), f in g.groupby(["model", "predictor_set"]):
        f = f.set_index(pd.to_datetime(f["date"])).rename(columns={"rv_actual": "rv"})[
            ["plain", "rv"]
        ]
        assert not f.index.duplicated().any(), (path, m, s)
        out[(m, s)] = f
    assert set(out) == set(KEYS), (path, sorted(out))
    return out


def span_days(rv: pd.Series, first: str, last: str) -> pd.DatetimeIndex:
    s = rv[(rv.index.hour == 16) & (rv.index.minute == 0)]
    d = pd.DatetimeIndex(s.index.normalize())
    keep = (s.to_numpy() > 0) & np.isfinite(s.to_numpy()) & ~d.isin(T1.EARLY)
    d = d[keep]
    return d[(d >= pd.Timestamp(first)) & (d <= pd.Timestamp(last))]


def common_bars(runs: dict, rv: pd.Series, days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = None
    for fc in runs.values():
        for f in fc.values():
            p = f["plain"].to_numpy()
            ok = f.index[np.isfinite(p) & (p > 0)]
            idx = ok if idx is None else idx.intersection(ok)
    idx = idx[pd.DatetimeIndex(idx.normalize()).isin(days)]
    t = rv.reindex(idx).to_numpy()
    idx = idx[(t > 0) & np.isfinite(t)].sort_values()
    for name, fc in runs.items():  # the files' truth is the core_stats print
        for k, f in fc.items():
            assert np.array_equal(
                f["rv"].reindex(idx).to_numpy(), rv.reindex(idx).to_numpy()
            ), (name, k)
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        action="append",
        required=True,
        help="name=results directory with forecasts.parquet",
    )
    ap.add_argument(
        "--span",
        action="append",
        required=True,
        help="first:last, e.g. 2012-01-03:2019-12-31",
    )
    ap.add_argument(
        "--compare",
        action="append",
        default=[],
        help="A:B, tests of run A against run B",
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = {}
    for spec in a.run:
        name, p = spec.split("=", 1)
        runs[name] = load(REPO / p)
    rv = T1.core()
    met, dms = [], []
    for sp in a.span:
        first, last = sp.split(":")
        days = span_days(rv, first, last)
        idx = common_bars(runs, rv, days)
        bench = T1.slot_benchmark(rv, idx)
        y = rv.reindex(idx).to_numpy(float)
        den = float(((y - bench.to_numpy()) ** 2).sum())
        on16 = idx[(idx.hour == 16) & (idx.minute == 0)]
        L = {}
        for name, fc in runs.items():
            for k in KEYS:
                q, se = T1.loss(fc[k], idx, rv)
                q16, _ = T1.loss(fc[k], on16, rv)
                L[(name, k)] = (q, se)
                met.append(
                    dict(
                        span=sp,
                        run=name,
                        model=k[0],
                        set=k[1],
                        n_intervals=len(idx),
                        n_days=len(days),
                        qlike=float(q.mean()),
                        mse_e10=float(se.mean() * 1e10),
                        r2_oos_pct=float(100.0 * (1.0 - se.sum() / den)),
                        qlike_1600=float(q16.mean()),
                    )
                )
            pairs = [
                (
                    (name, (m, "all_features")),
                    (name, (m, "baseline")),
                    "all features vs HAR benchmark",
                )
                for m in MODELS
                if m != "ols"
            ]
            pairs += [
                ((name, k), (name, ("ols", "baseline")), "vs OLS")
                for k in KEYS
                if k[0] != "ols"
            ]
            for A, B, what in pairs:
                for li, ln in ((0, "qlike"), (1, "mse")):
                    r = dm_test(T1.daily(L[A][li], idx), T1.daily(L[B][li], idx))
                    dms.append(
                        dict(
                            span=sp,
                            run=A[0],
                            model=A[1][0],
                            set=A[1][1],
                            against=f"{B[0]}:{B[1][0]}_{B[1][1]}",
                            comparison=what,
                            loss=ln,
                            dm=float(r["dm"]),
                            p=float(r["p"]),
                            mean_diff=float(r["mean_diff"]),
                            T=int(r["T"]),
                            hac_lag=int(r["hac_lag"]),
                        )
                    )
        for cmp in a.compare:
            ra, rb = cmp.split(":")
            for k in KEYS:
                for li, ln in ((0, "qlike"), (1, "mse")):
                    r = dm_test(
                        T1.daily(L[(ra, k)][li], idx), T1.daily(L[(rb, k)][li], idx)
                    )
                    dms.append(
                        dict(
                            span=sp,
                            run=ra,
                            model=k[0],
                            set=k[1],
                            against=f"{rb}:{k[0]}_{k[1]}",
                            comparison=f"{ra} vs {rb}",
                            loss=ln,
                            dm=float(r["dm"]),
                            p=float(r["p"]),
                            mean_diff=float(r["mean_diff"]),
                            T=int(r["T"]),
                            hac_lag=int(r["hac_lag"]),
                        )
                    )
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    met, dms = pd.DataFrame(met), pd.DataFrame(dms)
    met.to_csv(out / "metrics.csv", index=False)
    dms.to_csv(out / "dm.csv", index=False)
    pd.set_option("display.width", 200)
    print(
        met.drop(columns=["n_days"]).to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )
    q = dms[(dms.loss == "qlike") & (dms.comparison != "vs OLS")]
    print(
        q[["span", "run", "model", "set", "comparison", "dm", "p"]].to_string(
            index=False, float_format=lambda v: f"{v:.3f}"
        )
    )


if __name__ == "__main__":
    main()
