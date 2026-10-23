"""Score the final pooled MinGRU (results/pooled_mingru_final_2014start_w3000, W = 3,000, first day 2014-01-02) as the
2014-start pooled linear rows are scored (results/window_sweep_2014start_2026-10-22/code/score14.py) and as the
previous MinGRU run (experiments/pooled_mingru_rerun_13_score.py), with the linear W = 3,000 run as context (2026-10-22).

(a) QLIKE 2014-01-02 .. 2024-04-30.  Days: rolling_eval_2026_10_15.history() days in the span (positive closing-half-
    hour print, early closes dropped) on which every arm forecasts.  16:00: the closing interval's QLIKE (the 15:30
    forecast against the bar's RV, r = RV / plain, r - log r - 1).  Every bar: every interval of those days with
    RV > 0, the day's average loss.  Full vs HAR: % change of the mean and src.evaluation.diebold_mariano.dm_test on
    the daily loss difference (automatic lag, HLN); the same for the 16:00 squared error (a check).
    Gate: the linear W = 3,000 arms reproduce part2/qlike.csv (16:00 and every-bar QLIKE, 1e-12).
(b) The sign strategy on the 866 option days exactly as make_rehaul_economics.add_pooled_linear (q = +1 if plain >
    iv_var else -1; mid and crossed at bid / ask), on the 614 post-warmup days: D6 (E.sharpe) with
    hac_inference.d6_diff_test (full minus HAR; block 10, LW_BOOT, D6_SEED), mid and bid / ask; D1 (sharpe_days_d1) with
    the Ledoit-Wolf p; the pooled 614-day ratio; the LLSW t of the mean return difference.
    Gate: the linear W = 3,000 arms reproduce part2/econ.csv (1e-10).

usage: python pooled_mingru_final_score.py <shadow tree> <MinGRU run dir> <linear W3000 run dir> <part2 dir> <out dir>
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
MAIN = Path(sys.argv[1])
for _p in (MAIN, MAIN / "notebooks", MAIN / "writeup", MAIN / "experiments"):
    sys.path.insert(0, str(_p))

import hac_inference as hi  # noqa: E402
import make_rehaul_economics as E  # noqa: E402
import pooled_linear_13 as pl13  # noqa: E402
import rolling_eval_2026_10_15 as RE  # noqa: E402
from master_table_gui import rows as gui  # noqa: E402

from src.evaluation.diebold_mariano import dm_test  # noqa: E402

for _m in (E, pl13, RE):
    assert Path(str(_m.__file__)).resolve().is_relative_to(MAIN.resolve()), _m.__file__
GRU, LIN, PART2, OUT = (Path(x) for x in sys.argv[2:6])
OUT.mkdir(parents=True, exist_ok=True)
SPAN = (pd.Timestamp("2014-01-02"), pd.Timestamp("2024-04-30"))
SETS = ("baseline", "all_features")
EST = ("enet", "lasso", "ridge")
W = 3000
t0 = time.time()


def ann252(x) -> float:
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252.0))


# ---------------------------------------------------------------------------------------------------- the MinGRU
g = pd.read_parquet(GRU / "forecasts.parquet")
n = g.groupby(["model", "predictor_set"]).size()
assert set(n.index) == {("mingru", s) for s in SETS} and n.nunique() == 1, n.to_dict()
v = g[["pred_adj", "B", "plain"]].to_numpy(float)
assert np.isfinite(v).all() and (g["plain"] > 0).all() and (g["B"] > 0).all()
assert float(np.max(np.abs(g["plain"] / (g["pred_adj"] ** 2 * g["B"]) - 1.0))) <= 1e-12
assert pd.to_datetime(g["date"]).min() == pd.Timestamp("2014-01-02 10:00:00")


def gru_bars(s: str) -> pd.DataFrame:
    x = g[g["predictor_set"] == s]
    out = pd.DataFrame(
        {
            "plain": x["plain"].to_numpy(float),
            "rv": x["rv_actual"].to_numpy(float),
            "true_adj": x["true_adj"].to_numpy(float),
            "B": x["B"].to_numpy(float),
        },
        index=pd.DatetimeIndex(pd.to_datetime(x["date"])).as_unit("ns"),
    ).sort_index()
    assert not out.index.duplicated().any()
    return out


def at16(b: pd.DataFrame) -> pd.DataFrame:
    b = b[(b.index.hour == 16) & (b.index.minute == 0)].copy()
    b.index = b.index.normalize()
    assert not b.index.duplicated().any()
    return b


# ---------------------------------------------------------------------------------------------------- the linear run
f = pd.read_parquet(
    LIN / "forecasts.parquet",
    columns=["model", "predictor_set", "date", "plain", "rv_actual"],
)
nl = f.groupby(["model", "predictor_set"]).size()
pl13.RUN, pl13.FC = LIN, LIN / "forecasts.parquet"
pl13.N_BARS_ARM, pl13.WALK_START, pl13.FIRST_DAY = (
    int(nl.iloc[0]),
    pd.to_datetime(f["date"]).min(),
    SPAN[0],
)
pl13._CACHE.clear()
pl13._frame()  # pooled_linear_13's checks on the W = 3,000 run

hist = RE.history()
hdays = hist.index[(hist.index >= SPAN[0]) & (hist.index <= SPAN[1])]
ARMS: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}
for s in SETS:
    b = gru_bars(s)
    ARMS[("mingru", s)] = (at16(b), b)
for e in EST:
    for s in SETS:
        x = f[(f["model"] == e) & (f["predictor_set"] == s)]
        b = pd.DataFrame(
            {"plain": x["plain"].to_numpy(float), "rv": x["rv_actual"].to_numpy(float)},
            index=pd.DatetimeIndex(pd.to_datetime(x["date"])).as_unit("ns"),
        ).sort_index()
        ARMS[(e, s)] = (pl13.at16(e, s), b)
# the MinGRU's bars and truth = the linear run's on the MinGRU's span
for s in SETS:
    bg, bl = ARMS[("mingru", s)][1], ARMS[("ridge", s)][1]
    assert bg.index.isin(bl.index).all(), s
    assert np.array_equal(
        bg["rv"].to_numpy(), bl["rv"].reindex(bg.index).to_numpy(), equal_nan=True
    ), s

L16, SE16, LBAR = {}, {}, {}
for k, (a, b) in ARMS.items():
    a = a.copy()
    a.index = a.index.astype(hdays.dtype)
    a = a[
        (a.index >= SPAN[0])
        & (a.index <= SPAN[1])
        & a.index.isin(hdays)
        & (a["rv"] > 0)
    ]
    r = (a["rv"] / a["plain"]).to_numpy(float)
    L16[k] = pd.Series(r - np.log(r) - 1.0, index=a.index)
    SE16[k] = pd.Series((a["rv"] - a["plain"]).to_numpy(float) ** 2, index=a.index)
    bb = b[
        pd.DatetimeIndex(b.index.normalize()).isin(hdays)
        & (b["rv"].to_numpy(float) > 0)
    ]
    rb = (bb["rv"] / bb["plain"]).to_numpy(float)
    LBAR[k] = pd.Series(rb - np.log(rb) - 1.0, index=bb.index)
d16 = None
for q in L16.values():
    d16 = q.index if d16 is None else d16.intersection(q.index)
assert d16 is not None
ib = None
for q in LBAR.values():
    ib = q.index if ib is None else ib.intersection(q.index)
assert ib is not None
for k in L16:
    L16[k], SE16[k], LBAR[k] = (
        L16[k].reindex(d16),
        SE16[k].reindex(d16),
        LBAR[k].reindex(ib),
    )
DBAR = {k: q.groupby(q.index.normalize()).mean() for k, q in LBAR.items()}
print(
    f"common: {len(d16)} days (16:00), {len(ib)} intervals ({time.time() - t0:.0f} s)",
    flush=True,
)

# gate (a): the linear arms reproduce part2/qlike.csv at W = 3,000
pq = pd.read_csv(PART2 / "qlike.csv")
gates: dict = {}
for e in EST:
    for s in SETS:
        row = pq[(pq.model == e) & (pq.set == s) & (pq.W == W)].iloc[0]
        d1 = abs(float(L16[(e, s)].mean()) - row.qlike16)
        d2 = abs(float(LBAR[(e, s)].mean()) - row.qlike_bar)
        assert (
            d1 < 1e-12
            and d2 < 1e-12
            and len(d16) == row.n_days16
            and len(ib) == row.n_bars
        ), (e, s, d1, d2)
        gates[f"qlike_{e}_{s}"] = max(d1, d2)

lev, pairs = [], []
for k in ARMS:
    lev.append(
        dict(
            model=k[0],
            predictor_set=k[1],
            n_days=len(d16),
            qlike_1600=float(L16[k].mean()),
            mse_1600_e10=float(SE16[k].mean() * 1e10),
            n_bars=len(ib),
            qlike_all_bars=float(LBAR[k].mean()),
        )
    )
for m in ("mingru",) + EST:
    a_, b_ = (m, "all_features"), (m, "baseline")
    d = dm_test(L16[a_].to_numpy(), L16[b_].to_numpy())
    ds = dm_test(SE16[a_].to_numpy(), SE16[b_].to_numpy())
    assert DBAR[a_].index.equals(DBAR[b_].index)
    dd = dm_test(DBAR[a_].to_numpy(), DBAR[b_].to_numpy())
    pairs.append(
        dict(
            model=m,
            n_days=len(d16),
            qlike_har=float(L16[b_].mean()),
            qlike_full=float(L16[a_].mean()),
            qlike_pct=float(100 * (L16[a_].mean() / L16[b_].mean() - 1)),
            dm=float(d["dm"]),
            p=float(d["p"]),
            hac_lag=int(d["hac_lag"]),
            mse_pct=float(100 * (SE16[a_].mean() / SE16[b_].mean() - 1)),
            dm_mse=float(ds["dm"]),
            p_mse=float(ds["p"]),
            n_bars=len(ib),
            qlike_all_bars_har=float(LBAR[b_].mean()),
            qlike_all_bars_full=float(LBAR[a_].mean()),
            qlike_all_bars_pct=float(100 * (LBAR[a_].mean() / LBAR[b_].mean() - 1)),
            dm_all_bars_daily=float(dd["dm"]),
            p_all_bars_daily=float(dd["p"]),
        )
    )
LEV, PAIRS = pd.DataFrame(lev), pd.DataFrame(pairs)
LEV.to_csv(OUT / "scores_accuracy.csv", index=False)
PAIRS.to_csv(OUT / "scores_accuracy_full_vs_har.csv", index=False)

# ---------------------------------------------------------------------------------------------------- (b) economics
mt, model_set, series, fam = E.load()
days866 = series[E.OLS].index
pw = E.pw_days()
assert len(days866) == 866 and len(pw) == 614
daily = pd.read_parquet(
    E.RES / "master_table_daily.parquet",
    columns=["key", "day", "iv_var", "rv_unclipped"],
)
daily["day"] = pd.to_datetime(daily["day"]).dt.normalize()
ref = daily[daily["key"] == E.OLS].set_index("day").sort_index()
ref.index = ref.index.astype(days866.dtype)
ref = ref.reindex(days866)
iv, rv866 = ref["iv_var"].to_numpy(float), ref["rv_unclipped"].to_numpy(float)
assert np.isfinite(iv).all() and np.isfinite(rv866).all()
E.LW_CACHE.clear()
if hasattr(E, "BOOT_CACHE"):
    E.BOOT_CACHE.clear()
E.add_pooled_linear(series, fam)  # the linear W = 3,000 arms (pl13 points at the run)
bo = series[E.OLS]


def add(key: str, p: np.ndarray) -> None:
    """add_pooled_linear's row, line for line."""
    assert np.isfinite(p).all() and (p > 0).all(), key
    q = np.where(p > iv, 1.0, -1.0)
    ratio = rv866 / p
    f_s, y_s = p / iv, rv866 / iv
    assert key not in series, key
    series[key] = pd.DataFrame(
        {
            "q": q,
            "mid": q * bo["R"].to_numpy(float),
            "x": gui._crossed_return(
                q, bo["ex"].to_numpy(), bo["bid"].to_numpy(), bo["ask"].to_numpy()
            ),
            "s": p - iv,
            "qlike": ratio - np.log(ratio) - 1.0,
            "es1": np.abs(y_s - 1.0)
            * ((np.minimum(f_s, y_s) <= 1.0) & (1.0 < np.maximum(f_s, y_s))),
            "R": bo["R"].to_numpy(float),
            "ex": bo["ex"].to_numpy(float),
            "bid": bo["bid"].to_numpy(float),
            "ask": bo["ask"].to_numpy(float),
        },
        index=days866,
    )


PLAIN866 = {}
for s in SETS:
    a = ARMS[("mingru", s)][0].copy()
    a.index = a.index.astype(days866.dtype)
    a = a.reindex(days866)
    assert np.allclose(a["rv"].to_numpy(float), rv866, rtol=1e-12, atol=0.0), s
    PLAIN866[("mingru", s)] = a["plain"].to_numpy(float)
    add(f"mingruF_{s}", PLAIN866[("mingru", s)])
for e in EST:
    for s in SETS:
        PLAIN866[(e, s)] = E.lin_16(e, s, days866, rv866)
keys = {m: (f"poolpt13_{m}_baseline", f"poolpt13_{m}_all_features") for m in EST}
keys["mingru"] = ("mingruF_baseline", "mingruF_all_features")
econ = []
for m, (kh, kf) in keys.items():
    h, fu = series[kh], series[kf]
    out: dict = dict(model=m)
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
    for lab, gg, kk in (("har", h, (m, "baseline")), ("full", fu, (m, "all_features"))):
        out[f"long_share_{lab}"] = float((E.post_warmup(gg["q"]) > 0).mean())
        out[f"long_days_{lab}"] = int((E.post_warmup(gg["q"]) > 0).sum())
        out[f"mean_log_plain_rv_{lab}"] = float(
            np.log(PLAIN866[kk][E.WARMUP :] / rv866[E.WARMUP :]).mean()
        )
        out[f"mean_mid_{lab}"] = float(E.post_warmup(gg["mid"]).mean())
        out[f"qlike_866_{lab}"] = float(gg["qlike"].mean())
        by = hi.d6_sharpe(E.post_warmup(gg["mid"]), dates=pw)["by_year"]
        out[f"d6_by_year_mid_{lab}"] = json.dumps(
            {str(y): [nn, round(sr, 4)] for y, (nn, _, sr) in by.items()}
        )
    out["same_position_days"] = int(
        (E.post_warmup(h["q"]) == E.post_warmup(fu["q"])).sum()
    )
    econ.append(out)
EC = pd.DataFrame(econ)
# gate (b): the linear arms reproduce part2/econ.csv at W = 3,000
pe = pd.read_csv(PART2 / "econ.csv")
for e in EST:
    want = pe[(pe.model == e) & (pe.W == W)].iloc[0]
    got = EC[EC.model == e].iloc[0]
    for c in [c for c in pe.columns if c not in ("W", "model")]:
        assert abs(float(got[c]) - float(want[c])) < 1e-10, (e, c, got[c], want[c])
    gates[f"econ_{e}"] = "equal (1e-10)"
EC.to_csv(OUT / "econ_full_vs_har.csv", index=False)
(OUT / "score_gates.json").write_text(json.dumps(gates, indent=1, default=str))

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
print(LEV.round(5).to_string(index=False))
print(PAIRS.round(4).to_string(index=False))
print(EC.drop(columns=[c for c in EC.columns if "by_year" in c]).round(3).T.to_string())
print(f"done in {time.time() - t0:.0f} s")
