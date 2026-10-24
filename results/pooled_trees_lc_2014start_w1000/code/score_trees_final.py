"""Scores of the final pooled tree walks (first evaluation day 2014-01-02; windows of 3,000 and 3,500 trading days).

ACCURACY  as writeup/make_rehaul_oos_rerun.py scores Table 1 (its functions, imported): every bar of the evaluation days
          2014-01-02 .. 2024-04-30 (NYSE days with a positive closing-half-hour print, early closes dropped; asserted
          equal to rolling_eval_2026_10_15.history() on the span), the bars with a positive RV and a finite positive
          forecast of every model of the panel.  The panel is the run's trees and the linear rows of the same window's
          final linear run (results/window_sweep_2014start_2026-10-22/w<W>): asserted, the trees forecast every bar the
          linear rows score, so the scored set is the linear rows' own.  QLIKE, MSE (x 1e10), R2_OOS (the slot benchmark),
          and the 16:00 QLIKE (the 15:30 forecast of the closing half hour; the trees' 16:00 rows are asserted equal to
          pooled_trees_13.at16, the linear rows' reproduce their run's scores.csv on its 1,082 days to 1e-10).
TESTS     Diebold-Mariano (src.evaluation.diebold_mariano.dm_test: HLN, Newey-West lag max(h - 1, automatic)) on the
          daily average loss: all features vs the HAR benchmark of the same model (every bar and 16:00; QLIKE and MSE;
          negative = the full set has the lower loss), each arm vs the linear run's OLS, and with both windows the
          3,500-day arm vs the 3,000-day arm.
ECONOMICS the tree rows as writeup/make_rehaul_economics.add_pooled_trees builds them (tree_16 through pooled_trees_13
          pointed at the run: the sign of plain - iv_var on the OLS row's 866 option days, the quote book's straddle),
          HAR benchmark and full set, on the 614 post-warmup days: the D6 Sharpe ratio (make_rehaul_economics.sharpe)
          full vs HAR with hac_inference.d6_diff_test (block 10, 4,999 resamples, seed D6_SEED), mid and bid/ask; as
          robustness the D1 ratio (sharpe_days_d1) with the Ledoit-Wolf p, the pooled 614-day ratio (mean / sd x
          sqrt 252) and the LLSW t of the return difference -- as results/window_sweep_2014start_2026-10-22/code/
          score14.py does for the linear rows.

usage: python score_trees_final.py <shadow tree> <W>=<tree run dir>=<linear run dir> [...]
writes <tree run dir>/scores/{metrics,dm,econ}.csv and tables.txt; with two windows also cross_window_dm.csv in the
directory of each run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
MAIN = Path(sys.argv[1])
for p in (MAIN, MAIN / "notebooks", MAIN / "writeup", MAIN / "experiments"):
    sys.path.insert(0, str(p))

import hac_inference as hi
import make_rehaul_economics as E
import make_rehaul_oos_rerun as MR
import pooled_trees_13 as pt13
import rolling_eval_2026_10_15 as RE

from src.evaluation.diebold_mariano import dm_test

for _m in (hi, E, MR, pt13, RE):
    assert Path(_m.__file__).resolve().is_relative_to(MAIN.resolve()), _m.__file__
RUNS = {}
for arg in sys.argv[2:]:
    w, tr, lin = arg.split("=")
    RUNS[int(w)] = (Path(tr), Path(lin))
WS = sorted(RUNS)
FIRST, LAST = pd.Timestamp("2014-01-02"), pd.Timestamp("2024-04-30")
CK = (
    pd.Timestamp("2020-01-03"),
    pd.Timestamp("2024-04-30"),
    1082,
)  # the linear runs' scores.csv days
TREES = ("lgbm", "xgb", "rf")
LIN = ("ols", "ridge", "lasso", "enet")
SETS = ("baseline", "all_features")
NAME = {
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "rf": "Random forest",
    "ols": "OLS",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "enet": "Elastic net",
}
t0 = time.time()


def ann252(x) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252.0))


def dmc(D: pd.DataFrame, m: str, s: str, where: str) -> str:
    """The printed cell of the QLIKE test all features vs HAR benchmark of model m: DM (p)."""
    x = D[
        (D.model == m)
        & (D.set == s)
        & (D.comparison == "all features vs HAR benchmark")
        & (D.loss == "qlike")
        & (D.bars == where)
    ].iloc[0]
    return f"{x.dm:+.2f} ({x.p:.3f})"


def evaluation_days(rv: pd.Series) -> pd.DatetimeIndex:
    """make_rehaul_oos_rerun.evaluation_days on 2014-01-02 .. 2024-04-30 (its rule gives the count it asserts), checked
    against rolling_eval_2026_10_15.history() (the score stage's day rule)."""
    s = rv[(rv.index.hour == 16) & (rv.index.minute == 0)]
    d = pd.DatetimeIndex(s.index.normalize())
    keep = (s.to_numpy() > 0) & np.isfinite(s.to_numpy()) & ~d.isin(MR.EARLY)
    n = int(((d[keep] >= FIRST) & (d[keep] <= LAST)).sum())
    days = MR.evaluation_days(rv, FIRST, LAST, n)
    h = RE.history()
    hd = pd.DatetimeIndex(h.index[(h.index >= FIRST) & (h.index <= LAST)])
    assert np.array_equal(
        days.values.astype("datetime64[ns]"), hd.values.astype("datetime64[ns]")
    ), (len(days), len(hd))
    return days


def set_trees(W: int, run: Path) -> tuple[tuple[str, ...], dict]:
    """Point pooled_trees_13 at a final run (its arms, bar count and span replace the printed w1000 run's) and return
    the arms' frames (plain, rv) by bar end, as make_rehaul_oos_rerun.tree_forecasts."""
    f = pd.read_parquet(run / "forecasts.parquet", columns=["model", "predictor_set"])
    n = f.groupby(["model", "predictor_set"]).size()
    assert n.nunique() == 1, n.to_dict()
    ests = tuple(e for e in TREES if e in set(f["model"]))
    assert set(n.index) == {(e, s) for e in ests for s in SETS}, sorted(n.index)
    pt13.RUN, pt13.FC, pt13.EST, pt13.N_BARS_ARM = (
        run,
        run / "forecasts.parquet",
        ests,
        int(n.iloc[0]),
    )
    pt13._CACHE.clear()
    g = pt13._frame()  # the loader's gates: arms, bars an arm, plain = pred_adj^2 B, 13 bars a day, refit, span
    m = pt13.meta()
    assert (m["refit"], m["window_days"], m["span"]) == ("bar", W, (FIRST, LAST)), m
    out = {}
    for (e, s), x in g.groupby(["model", "predictor_set"]):
        out[(e, s)] = x.set_index(pd.to_datetime(x["date"])).rename(
            columns={"rv_actual": "rv"}
        )[["plain", "rv"]]
        assert not out[(e, s)].index.duplicated().any(), (e, s)
    return ests, out


rv = MR.core()
days = evaluation_days(rv)
print(
    f"{len(days)} evaluation days {days[0].date()} .. {days[-1].date()} ({time.time() - t0:.0f} s)",
    flush=True,
)
mt, model_set, series0, fam0 = E.load()
days866 = series0[E.OLS].index
pw = E.pw_days()
assert len(days866) == 866 and len(pw) == 614
dly = pd.read_parquet(
    E.RES / "master_table_daily.parquet",
    columns=["key", "day", "iv_var", "rv_unclipped"],
)
dly["day"] = pd.to_datetime(dly["day"]).dt.normalize()
ref = dly[dly["key"] == E.OLS].set_index("day").sort_index()
ref.index = ref.index.astype(days866.dtype)
ref = ref.reindex(days866)
iv, rv866 = ref["iv_var"].to_numpy(float), ref["rv_unclipped"].to_numpy(float)
bo = series0[E.OLS]

LOSS: dict = {}
IDX: dict = {}
for W in WS:
    trun, lrun = RUNS[W]
    ests, ft = set_trees(W, trun)
    fl = MR.forecasts("13", lrun)
    idx_l = MR.scored_set(fl, rv, days)
    idx = MR.scored_set(
        {**fl, **ft}, rv, days
    )  # also asserts every tree's rv_actual is the core_stats print
    miss = idx_l.difference(idx)
    assert idx.equals(idx_l), (
        f"w{W}: the trees miss {len(miss)} bars the linear rows score, e.g. {list(miss[:5])}"
    )
    IDX[W] = idx
    bench = MR.slot_benchmark(rv, idx)
    y = rv.reindex(idx).to_numpy(float)
    den = float(((y - bench.to_numpy()) ** 2).sum())
    on16 = idx[(idx.hour == 16) & (idx.minute == 0)]
    met, dms, econ = [], [], []
    for k, f in {**ft, **fl}.items():
        fam = "tree" if k in ft else "linear"
        q, se = MR.loss(f, idx, rv)
        q16, se16 = MR.loss(f, on16, rv)
        LOSS[(W, *k)] = (q, se, q16, se16)
        if fam == "tree":
            a16 = pt13.at16(*k).reindex(pd.DatetimeIndex(on16.normalize()))
            assert np.array_equal(
                a16[["plain", "rv"]].to_numpy(),
                f[["plain", "rv"]].reindex(on16).to_numpy(),
            ), k
        else:  # the linear run's own closing-interval QLIKE (scores.csv, 1,082 days 2020-01-03 .. 2024-04-30)
            ck = on16[(on16.normalize() >= CK[0]) & (on16.normalize() <= CK[1])]
            qc, _ = MR.loss(f, ck, rv)
            sc = pd.read_csv(lrun / "scores.csv")
            setcol = "set" if "set" in sc.columns else "predictor_set"
            want = sc[(sc["model"] == k[0]) & (sc[setcol] == k[1])]["qlike"].iloc[0]
            assert len(ck) == CK[2] and np.isclose(
                qc.mean(), want, rtol=1e-10, atol=0
            ), (W, k, qc.mean(), want)
        met.append(
            {
                "W": W,
                "family": fam,
                "model": k[0],
                "set": k[1],
                "n_bars": len(idx),
                "n_days": len(days),
                "n_days16": len(on16),
                "qlike": float(q.mean()),
                "mse_e10": float(se.mean() * 1e10),
                "r2_oos_pct": float(100.0 * (1.0 - se.sum() / den)),
                "qlike_1600": float(q16.mean()),
                "mse_1600_e10": float(se16.mean() * 1e10),
            }
        )
    L = {k[1:]: v for k, v in LOSS.items() if k[0] == W}
    for m in list(ests) + [x for x in LIN if x != "ols"]:
        fam = "tree" if m in ests else "linear"
        pairs = [
            ((m, "all_features"), (m, "baseline"), "all features vs HAR benchmark")
        ]
        pairs += [((m, s), ("ols", "baseline"), "vs OLS") for s in SETS]
        for a, b, what in pairs:
            for li, lname in ((0, "qlike"), (1, "mse")):
                for where, la, lb in (
                    ("every bar", MR.daily(L[a][li], idx), MR.daily(L[b][li], idx)),
                    ("16:00", L[a][li + 2], L[b][li + 2]),
                ):
                    r = dm_test(la, lb)
                    dms.append(
                        {
                            "W": W,
                            "family": fam,
                            "model": a[0],
                            "set": a[1],
                            "against": f"{b[0]}_{b[1]}",
                            "comparison": what,
                            "bars": where,
                            "loss": lname,
                            "mean_a": float(
                                np.mean(L[a][li if where == "every bar" else li + 2])
                            ),
                            "mean_b": float(
                                np.mean(L[b][li if where == "every bar" else li + 2])
                            ),
                            "dm": float(r["dm"]),
                            "p": float(r["p"]),
                            "mean_diff": float(r["mean_diff"]),
                            "T": int(r["T"]),
                            "hac_lag": int(r["hac_lag"]),
                        }
                    )
    # economics: the tree rows, traded like make_rehaul_economics.add_pooled_trees (both predictor sets)
    rows = {}
    for e in ests:
        for s in SETS:
            p = E.tree_16(e, s, days866, rv866)
            q = np.where(p > iv, 1.0, -1.0)
            rows[(e, s)] = {
                "p": p,
                "q": q,
                "mid": q * bo["R"].to_numpy(float),
                "x": E.gui._crossed_return(
                    q, bo["ex"].to_numpy(), bo["bid"].to_numpy(), bo["ask"].to_numpy()
                ),
            }
    for e in ests:
        h, fu = rows[(e, "baseline")], rows[(e, "all_features")]
        out: dict = {"W": W, "model": e}
        for col, lab in (("mid", "mid"), ("x", "ba")):
            hp, fp_ = E.post_warmup(h[col]), E.post_warmup(fu[col])
            sh, sf = E.sharpe(h[col]), E.sharpe(fu[col])
            bt = hi.d6_diff_test(fp_, hp, pw)
            assert abs(bt["delta"] - (sf - sh)) < 1e-10
            ll = hi.llsw_test(fp_ - hp)
            out |= {
                f"d6_{lab}_har": sh,
                f"d6_{lab}_full": sf,
                f"dd6_{lab}": sf - sh,
                f"d6_boot_p_{lab}": bt["p"],
                f"d1_{lab}_har": E.sharpe_days_d1(hp),
                f"d1_{lab}_full": E.sharpe_days_d1(fp_),
                f"lw_p_{lab}": hi.lw_sharpe_test(fp_, hp, block=hi.LW_BLOCK)["p"],
                f"llsw_t_{lab}": ll["t"],
                f"llsw_p_{lab}": ll["p"],
                f"sr614_{lab}_har": ann252(hp),
                f"sr614_{lab}_full": ann252(fp_),
            }
            out[f"dd1_{lab}"] = out[f"d1_{lab}_full"] - out[f"d1_{lab}_har"]
            out[f"dsr614_{lab}"] = out[f"sr614_{lab}_full"] - out[f"sr614_{lab}_har"]
        for lab, g in (("har", h), ("full", fu)):
            out[f"long_share_{lab}"] = float((E.post_warmup(g["q"]) > 0).mean())
            out[f"mean_log_plain_rv_{lab}"] = float(
                np.log(g["p"][E.WARMUP :] / rv866[E.WARMUP :]).mean()
            )
        econ.append(out)
    M, D, EC = pd.DataFrame(met), pd.DataFrame(dms), pd.DataFrame(econ)
    sd = trun / "scores"
    sd.mkdir(parents=True, exist_ok=True)
    M.to_csv(sd / "metrics.csv", index=False)
    D.to_csv(sd / "dm.csv", index=False)
    EC.to_csv(sd / "econ.csv", index=False)
    # tables
    T = [
        f"# Pooled trees, final design, W = {W:,} days: scores 2014-01-02 .. 2024-04-30",
        (
            f"{len(days):,} evaluation days; every bar: {len(idx):,} bars (the linear rows' scored set, asserted); "
            f"16:00: {len(on16):,} days.  QLIKE, MSE x 1e10, R2_OOS in %.  DM: all features vs HAR benchmark of the same "
            "model on the daily average loss (negative = full lower), p in parentheses."
        ),
        "",
        "| model | set | QLIKE every bar | MSE every bar | R2_OOS | QLIKE 16:00 | DM full vs HAR, every bar | DM 16:00 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for fam, ms in (("tree", ests), ("linear", LIN)):
        for m in ms:
            for s in SETS:
                r = M[(M.model == m) & (M.set == s)]
                if r.empty:
                    continue
                r = r.iloc[0]
                c1 = c2 = ""
                if s == "all_features":
                    c1, c2 = dmc(D, m, s, "every bar"), dmc(D, m, s, "16:00")
                T.append(
                    f"| {NAME[m]}{' (linear run)' if fam == 'linear' else ''} | {'HAR' if s == 'baseline' else 'full'} "
                    f"| {r.qlike:.5f} | {r.mse_e10:.3f} | {r.r2_oos_pct:.2f} | {r.qlike_1600:.5f} | {c1} | {c2} |"
                )
    T += [
        "",
        f"## Sign strategy, {len(pw)} post-warmup option days ({pw[0].date()} .. {pw[-1].date()}), trees",
        (
            "D6 Sharpe ratio (paper), ΔD6 = full − HAR with the D6 bootstrap p; D1 with the Ledoit–Wolf p; SR614 = the "
            "pooled 614-day ratio; LLSW t (p) of the daily return difference (mid)."
        ),
        "",
        (
            "| model | D6 HAR | D6 full | ΔD6 mid (p) | D6 b/a HAR | D6 b/a full | ΔD6 b/a (p) | D1 HAR / full | ΔD1 (LW p) | "
            "SR614 HAR / full | LLSW t (p) | long share HAR / full |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, x in EC.iterrows():
        T.append(
            f"| {NAME[x.model]} | {x.d6_mid_har:.2f} | {x.d6_mid_full:.2f} | {x.dd6_mid:+.2f} ({x.d6_boot_p_mid:.3f}) | "
            f"{x.d6_ba_har:.2f} | {x.d6_ba_full:.2f} | {x.dd6_ba:+.2f} ({x.d6_boot_p_ba:.3f}) | "
            f"{x.d1_mid_har:.2f} / {x.d1_mid_full:.2f} | {x.dd1_mid:+.2f} ({x.lw_p_mid:.3f}) | "
            f"{x.sr614_mid_har:.2f} / {x.sr614_mid_full:.2f} | {x.llsw_t_mid:.2f} ({x.llsw_p_mid:.3f}) | "
            f"{x.long_share_har:.3f} / {x.long_share_full:.3f} |"
        )
    (sd / "tables.txt").write_text("\n".join(T) + "\n", encoding="utf-8")
    print("\n".join(T), flush=True)
    print(f"W={W}: scored ({time.time() - t0:.0f} s)", flush=True)

if (
    len(WS) == 2
):  # the longer window against the shorter, arm by arm, on the bars both score
    W0, W1 = WS
    common = IDX[W0].intersection(IDX[W1])
    assert common.equals(IDX[W0]) and common.equals(IDX[W1]), (
        "the two windows score different bars"
    )
    rows_x = []
    for k in sorted(
        {k[1:] for k in LOSS if k[0] == W1} & {k[1:] for k in LOSS if k[0] == W0}
    ):
        a, b = LOSS[(W1, *k)], LOSS[(W0, *k)]
        for where, la, lb in (
            ("every bar", MR.daily(a[0], common), MR.daily(b[0], common)),
            ("16:00", a[2], b[2]),
        ):
            r = dm_test(la, lb)
            rows_x.append(
                {
                    "model": k[0],
                    "set": k[1],
                    "bars": where,
                    "loss": "qlike",
                    "W_a": W1,
                    "W_b": W0,
                    "qlike_a": float(np.mean(a[0] if where == "every bar" else a[2])),
                    "qlike_b": float(np.mean(b[0] if where == "every bar" else b[2])),
                    "dm": float(r["dm"]),
                    "p": float(r["p"]),
                    "T": int(r["T"]),
                    "hac_lag": int(r["hac_lag"]),
                }
            )
    X = pd.DataFrame(rows_x)
    for W in WS:
        X.to_csv(RUNS[W][0] / "scores" / "cross_window_dm.csv", index=False)
    print(X.to_string(index=False))
print(f"done in {time.time() - t0:.0f} s")
