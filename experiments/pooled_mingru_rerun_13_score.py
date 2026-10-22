"""Score the pooled MinGRU of the 13-bar rerun (experiments/pooled_mingru_rerun_13.py) as the paper scores the pooled
linear models and trees on the same design (2026-10-22).  Local: needs the repo's results/ and data/.

ACCURACY  (writeup/make_rehaul_results.py pair(); experiments/pooled_linear_rerun_2026_10_19_13.st_score)
  16:00 rows (the 15:30 forecast of the closing half hour) on the 1,082 days of the pooled linear rerun's scores.csv
  (common_span.csv first .. last, positive closing RV of rolling_eval.history(), early closes dropped, a forecast of
  every arm); daily QLIKE r - log r - 1, r = RV / plain; full set vs HAR benchmark: % change of the mean and
  src.evaluation.diebold_mariano.dm_test on the daily QLIKE difference (automatic lag, HLN); the same for the squared
  error.  Every bar of those days (rv_actual > 0; the linear rerun's scores_all_bars.csv rule): mean QLIKE and DM on
  the daily average.  Gates: the linear arms' 16:00 QLIKE equals scores.csv (1e-10) and their every-bar QLIKE equals
  scores_all_bars.csv; the MinGRU's rv_actual equals rolling_eval.history() on the 16:00 rows.
ECONOMICS (writeup/make_rehaul_economics.py: load, add_pooled_linear, add_pooled_trees, tree_16, pair_stats, hac_p)
  The sign strategy q = +1 if plain > iv_var else -1 on the 866 option days (the OLS row's iv_var, rv_unclipped), the
  quote book's straddle at mid and crossed at the quoted bid / ask (master_table_gui.rows._crossed_return); Sharpe
  ratio per-year annualized on the 614 post-warmup days; full vs HAR: the change at mid and at bid / ask, the
  Ledoit-Wolf p (hac_inference.lw_sharpe_test) and the fixed-b LLSW t / p (S = 33) of the mean return difference.
  The six pooled linear and six pooled tree arms are scored by the same code for context (gate: their Sharpe ratios
  equal the economics generator's console values).

    python experiments/pooled_mingru_rerun_13_score.py [--context-only] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "experiments", REPO / "notebooks", REPO / "writeup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import make_rehaul_economics as E  # noqa: E402
import pooled_linear_13 as pl13  # noqa: E402
import pooled_trees_13 as pt13  # noqa: E402
import rolling_eval_2026_10_15 as RE  # noqa: E402
from master_table_gui import rows as gui  # noqa: E402

from src.evaluation.diebold_mariano import dm_test  # noqa: E402

OUT = REPO / "results" / "pooled_mingru_rerun13_2026-10-22"
LIN = pl13.RUN
NAME = {
    "enet": "elastic net",
    "lasso": "lasso",
    "ridge": "ridge",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "random forest",
    "mingru": "MinGRU (pooled, ridge-init, QLIKE)",
}


def arms(
    context_only: bool,
) -> dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame | None]]:
    """(model, set) -> (16:00 frame [plain, rv] by day, every-bar frame [plain, rv] by bar end or None)."""
    out = {}
    fl = pd.read_parquet(
        LIN / "forecasts.parquet",
        columns=["model", "predictor_set", "date", "plain", "rv_actual"],
    )
    for e in pl13.EST:
        for s in pl13.SETS:
            g = fl[(fl["model"] == e) & (fl["predictor_set"] == s)]
            b = pd.DataFrame(
                {
                    "plain": g["plain"].to_numpy(float),
                    "rv": g["rv_actual"].to_numpy(float),
                },
                index=pd.DatetimeIndex(pd.to_datetime(g["date"])).as_unit("ns"),
            )
            out[(e, s)] = (pl13.at16(e, s), b)
    ft = pd.read_parquet(
        pt13.FC, columns=["model", "predictor_set", "date", "plain", "rv_actual"]
    )
    for e in pt13.EST:
        for s in pt13.SETS:
            g = ft[(ft["model"] == e) & (ft["predictor_set"] == s)]
            b = pd.DataFrame(
                {
                    "plain": g["plain"].to_numpy(float),
                    "rv": g["rv_actual"].to_numpy(float),
                },
                index=pd.DatetimeIndex(pd.to_datetime(g["date"])).as_unit("ns"),
            )
            out[(e, s)] = (pt13.at16(e, s), b)
    if not context_only:
        import pooled_mingru_13 as pm13

        for s in pm13.SETS:
            b = pm13.bars(s)
            lin_idx = out[("ridge", s)][1].index.sort_values()
            assert b.index.equals(lin_idx), (
                s,
                len(b),
                len(lin_idx),
            )  # the linear rerun's 40,497 walk bars
            assert np.array_equal(
                b["rv"].to_numpy(),
                out[("ridge", s)][1]["rv"].reindex(b.index).to_numpy(),
                equal_nan=True,
            ), s
            out[("mingru", s)] = (pm13.at16(s), b)
    return out


def accuracy(A: dict, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist = RE.history()
    cs = pd.read_csv(
        REPO / "results" / "rolling_eval_2026-10-15" / "common_span.csv"
    ).iloc[0]
    days = hist.index[(hist.index >= cs["first"]) & (hist.index <= cs["last"])]
    for (m, s), (a16, _) in A.items():
        p = a16["plain"]
        p.index = p.index.astype(days.dtype)
        days = days.intersection(
            p.index[np.isfinite(p.to_numpy()) & (p.to_numpy() > 0)]
        )
    sc = pd.read_csv(LIN / "scores.csv")
    assert len(days) == 1082 == int(sc["n_days"].iloc[0]), len(days)
    assert (
        str(days[0].date()) == sc["first"].iloc[0]
        and str(days[-1].date()) == sc["last"].iloc[0]
    )
    rv = hist.reindex(days)
    assert np.isfinite(rv.to_numpy()).all() and (rv > 0).all()
    q16, se16, qbar, recs = {}, {}, {}, []
    ab = pd.read_csv(LIN / "scores_all_bars.csv").set_index(["model", "predictor_set"])
    for (m, s), (a16, b) in A.items():
        a = a16.copy()
        a.index = a.index.astype(days.dtype)
        a = a.reindex(days)
        rel = float(np.max(np.abs(a["rv"].to_numpy() / rv.to_numpy() - 1.0)))
        assert rel <= 1e-12, (m, s, rel)
        r = (rv / a["plain"]).to_numpy(float)
        q16[(m, s)] = pd.Series(r - np.log(r) - 1.0, index=days)
        se16[(m, s)] = pd.Series(
            (rv.to_numpy() - a["plain"].to_numpy()) ** 2, index=days
        )
        bb = b[pd.DatetimeIndex(b.index.normalize()).isin(days) & (b["rv"] > 0)]
        rb = (bb["rv"] / bb["plain"]).to_numpy(float)
        qb = pd.Series(rb - np.log(rb) - 1.0, index=bb.index)
        qbar[(m, s)] = qb
        rec = dict(
            model=m,
            predictor_set=s,
            n_days=len(days),
            qlike_1600=float(q16[(m, s)].mean()),
            mse_1600_e10=float(se16[(m, s)].mean() * 1e10),
            n_bars=len(qb),
            qlike_all_bars=float(qb.mean()),
        )
        if m in pl13.EST:
            want = float(
                sc[(sc["model"] == m) & (sc["predictor_set"] == s)]["qlike"].iloc[0]
            )
            assert abs(rec["qlike_1600"] - want) < 1e-10, (
                m,
                s,
                rec["qlike_1600"],
                want,
            )
            assert (
                abs(rec["qlike_all_bars"] - float(ab.loc[(m, s), "qlike"])) < 1e-10
            ), (m, s)
            assert rec["n_bars"] == int(ab.loc[(m, s), "n_bars"]), (m, s)
        recs.append(rec)
    lev = pd.DataFrame(recs)
    pairs = []
    for m in dict.fromkeys(k[0] for k in A):
        a, b = (m, "all_features"), (m, "baseline")
        d = dm_test(q16[a].to_numpy(), q16[b].to_numpy())
        ds = dm_test(se16[a].to_numpy(), se16[b].to_numpy())
        da = qbar[a].groupby(qbar[a].index.normalize()).mean()
        db = qbar[b].groupby(qbar[b].index.normalize()).mean()
        assert da.index.equals(db.index)
        dd = dm_test(da.to_numpy(), db.to_numpy())
        pairs.append(
            dict(
                model=m,
                n_days=len(days),
                qlike_har=float(q16[b].mean()),
                qlike_full=float(q16[a].mean()),
                qlike_pct=float(100 * (q16[a].mean() / q16[b].mean() - 1)),
                dm=float(d["dm"]),
                p=float(d["p"]),
                hac_lag=int(d["hac_lag"]),
                mse_pct=float(100 * (se16[a].mean() / se16[b].mean() - 1)),
                dm_mse=float(ds["dm"]),
                p_mse=float(ds["p"]),
                qlike_all_bars_har=float(qbar[b].mean()),
                qlike_all_bars_full=float(qbar[a].mean()),
                qlike_all_bars_pct=float(100 * (qbar[a].mean() / qbar[b].mean() - 1)),
                dm_all_bars_daily=float(dd["dm"]),
                p_all_bars_daily=float(dd["p"]),
            )
        )
    pr = pd.DataFrame(pairs)
    lev.to_csv(out / "scores_accuracy.csv", index=False)
    pr.to_csv(out / "scores_accuracy_full_vs_har.csv", index=False)
    return lev, pr


def economics(A: dict, out: Path) -> pd.DataFrame:
    mt, model_set, series, fam = E.load()
    E.add_pooled_linear(series, fam)
    E.add_pooled_trees(series, fam)
    b = series[E.OLS]
    days = b.index
    daily = pd.read_parquet(
        E.RES / "master_table_daily.parquet",
        columns=["key", "day", "iv_var", "rv_unclipped"],
    )
    daily["day"] = pd.to_datetime(daily["day"]).dt.normalize()
    ref = daily[daily["key"] == E.OLS].set_index("day").sort_index()
    ref.index = ref.index.astype(days.dtype)
    ref = ref.reindex(days)
    iv, rv = ref["iv_var"].to_numpy(float), ref["rv_unclipped"].to_numpy(float)
    assert np.isfinite(iv).all() and np.isfinite(rv).all()

    def add(key: str, p: np.ndarray) -> None:
        """add_pooled_linear's row, line for line."""
        assert np.isfinite(p).all() and (p > 0).all(), key
        q = np.where(p > iv, 1.0, -1.0)
        ratio = rv / p
        f_s, y_s = p / iv, rv / iv
        assert key not in series, key
        series[key] = pd.DataFrame(
            {
                "q": q,
                "mid": q * b["R"].to_numpy(float),
                "x": gui._crossed_return(
                    q, b["ex"].to_numpy(), b["bid"].to_numpy(), b["ask"].to_numpy()
                ),
                "s": p - iv,
                "qlike": ratio - np.log(ratio) - 1.0,
                "es1": np.abs(y_s - 1.0)
                * ((np.minimum(f_s, y_s) <= 1.0) & (1.0 < np.maximum(f_s, y_s))),
                "R": b["R"].to_numpy(float),
                "ex": b["ex"].to_numpy(float),
                "bid": b["bid"].to_numpy(float),
                "ask": b["ask"].to_numpy(float),
            },
            index=days,
        )

    for e, _ in E.TREE_EST:
        add(f"pooltree_{e}_baseline", E.tree_16(e, "baseline", days, rv))
    if ("mingru", "all_features") in A:
        for s in ("baseline", "all_features"):
            a = A[("mingru", s)][0].copy()
            a.index = a.index.astype(days.dtype)
            a = a.reindex(days)
            assert np.allclose(a["rv"].to_numpy(float), rv, rtol=1e-12, atol=0.0), s
            add(f"mingru13_{s}", a["plain"].to_numpy(float))
    keys = {
        m: (f"poolpt13_{m}_baseline", f"poolpt13_{m}_all_features") for m in pl13.EST
    }
    keys.update(
        {
            m: (f"pooltree_{m}_baseline", f"pooltree_{m}_all_features")
            for m, _ in E.TREE_EST
        }
    )
    if "mingru13_all_features" in series:
        keys["mingru"] = ("mingru13_baseline", "mingru13_all_features")
    recs = []
    for m, (kb, ka) in keys.items():
        st = E.pair_stats(series, kb, ka)
        ga, gb = series[ka], series[kb]
        recs.append(
            dict(
                model=m,
                n_days=E.N_OPT,
                n_post_warmup=E.N_PW,
                sr_mid_har=st["sh_b"],
                sr_mid_full=st["sh_a"],
                d_sr_mid=st["d_mid"],
                lw_p_mid=st["lw_mid"],
                t_mean_diff_mid=st["t_mid"],
                fixedb_p_mid=E.hac_p(st["t_mid"]),
                sr_x_har=E.sharpe(gb["x"]),
                sr_x_full=E.sharpe(ga["x"]),
                d_sr_x=st["d_x"],
                lw_p_x=st["lw_x"],
                t_mean_diff_x=st["t_x"],
                fixedb_p_x=E.hac_p(st["t_x"]),
                mean_mid_har=float(gb["mid"].to_numpy()[E.WARMUP :].mean()),
                mean_mid_full=float(ga["mid"].to_numpy()[E.WARMUP :].mean()),
                long_days_har=int((gb["q"].to_numpy()[E.WARMUP :] > 0).sum()),
                long_days_full=int((ga["q"].to_numpy()[E.WARMUP :] > 0).sum()),
                same_position_days=int(
                    (
                        ga["q"].to_numpy()[E.WARMUP :] == gb["q"].to_numpy()[E.WARMUP :]
                    ).sum()
                ),
                qlike_866_har=float(gb["qlike"].mean()),
                qlike_866_full=float(ga["qlike"].mean()),
            )
        )
        for k in (kb, ka):
            yy = E.asl.sharpe_ann_by_year(
                pd.Series(
                    series[k]["mid"].to_numpy()[E.WARMUP :], index=days[E.WARMUP :]
                )
            )
            recs[-1][f"by_year_mid_{'har' if k == kb else 'full'}"] = json.dumps(
                {str(y): [n, round(v, 4)] for y, (n, v) in yy["by_year"].items()}
            )
    t = pd.DataFrame(recs)
    t.to_csv(out / "econ_full_vs_har.csv", index=False)
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-only", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    A = arms(a.context_only)
    lev, pr = accuracy(A, out)
    ec = economics(A, out)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(lev.round(5).to_string(index=False))
    print(pr.round(4).to_string(index=False))
    print(
        ec.drop(columns=[c for c in ec.columns if c.startswith("by_year")])
        .round(3)
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
