"""POST HOC (not in RULE.txt; written after the scores were read): each model's monthly-refit rungs against Table 1's
every-bar walk of the full set and of the HAR benchmark (same 33,527 bars, QLIKE, DM HLN on daily averages), to separate
the cost of holding a fit for 21 sessions from the effect of k.  -> results/ladder_2026-10-26/posthoc_vs_walk.csv

usage: python results/ladder_2026-10-26/code/ladder_posthoc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent
ROOT = RES.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "notebooks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ladder_score as LS  # noqa: E402
from src.evaluation.diebold_mariano import dm_test  # noqa: E402

TEMPL = {
    "lin": ROOT / "results" / "window_sweep_2014start_2026-10-22" / "w3000" / "forecasts.parquet",
    "trees": ROOT / "results" / "pooled_trees_final_2014start_w3000" / "forecasts.parquet",
    "rf": ROOT / "results" / "pooled_rf_skx_2014start_w3000" / "forecasts.parquet",
}


def main() -> int:
    w = pd.read_parquet(RES / "forecasts_wide.parquet").set_index("date")
    lt = pd.read_parquet(TEMPL["lin"], columns=["model", "predictor_set", "date", "plain", "B"])
    tt = pd.concat([pd.read_parquet(TEMPL["trees"]), pd.read_parquet(TEMPL["rf"])], ignore_index=True)
    allt = pd.concat([lt[["model", "predictor_set", "date", "plain"]], tt[["model", "predictor_set", "date", "plain"]]])
    B = lt[(lt.model == "ridge") & (lt.predictor_set == "all_features")].set_index("date")["B"].reindex(w.index)
    rv = LS.core_rv()
    days = LS.evaluation_days(rv)
    idx = w.index[pd.DatetimeIndex(w.index.normalize()).isin(days)]
    y = rv.reindex(idx)
    idx = idx[(y.to_numpy() > 0) & np.isfinite(y.to_numpy())]
    S = pd.read_csv(RES / "scores.csv")
    rows = []
    for m in ("lgbm", "xgb", "rf_skx", "ridge", "lasso", "enet"):
        walk = {}
        for st in ("all_features", "baseline"):
            f = allt[(allt.model == m) & (allt.predictor_set == st)].set_index("date")["plain"]
            walk[st] = f.reindex(idx).to_numpy(float)
        yv = rv.reindex(idx).to_numpy(float)

        def ql(p: np.ndarray) -> np.ndarray:
            r = yv / p
            return r - np.log(r) - 1.0

        def daily(x: np.ndarray) -> np.ndarray:
            return pd.Series(x, index=pd.DatetimeIndex(idx.normalize())).groupby(level=0).mean().to_numpy()

        lw = {st: ql(p) for st, p in walk.items()}
        f = S[(S.model == m) & (S.span == "all") & (S.ranking.isin(["P", "both"]))].drop_duplicates("k")
        for r_ in f.itertuples():
            col = f"{m}|{r_.fitted_as}"
            p = w[col].reindex(idx).to_numpy(float) ** 2 * B.reindex(idx).to_numpy(float)
            q = ql(p)
            assert len(idx) == 33527 and np.isclose(q.mean(), r_.qlike, rtol=1e-12)
            for st, lab in (("all_features", "walk_full"), ("baseline", "walk_har")):
                d = dm_test(daily(q), daily(lw[st]))
                rows.append(
                    dict(
                        model=m,
                        k=int(r_.k),
                        rung=r_.rung,
                        qlike_monthly=float(q.mean()),
                        vs=lab,
                        qlike_walk=float(lw[st].mean()),
                        dm=float(d["dm"]),
                        p=float(d["p"]),
                        hac_lag=int(d["hac_lag"]),
                    )
                )
    out = pd.DataFrame(rows)
    out.to_csv(RES / "posthoc_vs_walk.csv", index=False)
    print(out[out.vs == "walk_full"].round(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
