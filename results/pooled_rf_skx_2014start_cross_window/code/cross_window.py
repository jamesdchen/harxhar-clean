"""Cross-window comparison of the pooled random forest walks, rf_skx (the scikit-learn forest on oneDAL; first evaluation
day 2014-01-02, refit at every bar): W = 1,000 / 2,000 / 3,000 / 3,500, all on CARC (this branch,
insurance-selling-2026-09-19-e34-trees-final, results/pooled_rf_skx_2014start_w<W>).  The logic is that of the
LightGBM / XGBoost comparison (insurance-selling-2026-09-19-e34-tree-lc, results/pooled_trees_lc_2014start_w1000/code/
cross_window.py) with the arms rf_skx / {HAR, full} in place of lgbm, xgb / {HAR, full}.

ACCURACY  every bar of the evaluation days 2014-01-02 .. 2024-04-30 and the 16:00 bar, scored as
          score_trees_final.py does (make_rehaul_oos_rerun's functions): QLIKE of plain against the core_stats RV on
          ONE common set of bars, the intersection of the windows' scored sets (each the linear rows' own set of that
          window, asserted equal to the trees' set there); the windows' sets are reported and asserted equal.
TESTS     Diebold-Mariano (src.evaluation.diebold_mariano.dm_test: HLN, Newey-West lag automatic) on the daily average
          QLIKE (every bar) and the 16:00 QLIKE, longer window against shorter (negative = the longer window has the
          lower loss), for the pairs 1k->2k, 2k->3k, 3k->3.5k and 1k->3k, per arm.
GAP       full - HAR by W (the daily average difference, with its NW standard error), and the change of the gap from
          window W1 to W2: extra gain = (full_W2 - full_W1) - (HAR_W2 - HAR_W1), the daily difference-in-differences;
          mean, Newey-West standard error (Bartlett, lag floor(4 (T/100)^(2/9)), divisor 1/T, as dm_test), t and the
          two-sided normal p.  Negative extra gain = the full set gains more from the longer window than the HAR
          benchmark does.
ECONOMICS D6 by W (and D1, the pooled 614-day ratio) from each run's scores/econ.csv (score_trees_final.py).

usage: python cross_window.py <shadow tree> <out dir> <W>=<tree run dir>=<linear run dir> [...]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
MAIN = Path(sys.argv[1])
OUT = Path(sys.argv[2])
for p in (MAIN, MAIN / "notebooks", MAIN / "writeup", MAIN / "experiments"):
    sys.path.insert(0, str(p))

import make_rehaul_oos_rerun as MR
import rolling_eval_2026_10_15 as RE

from src.evaluation.diebold_mariano import _nw_lag, dm_test

RUNS: dict[int, tuple[Path, Path]] = {}
for arg in sys.argv[3:]:
    w, tr, lin = arg.split("=")
    RUNS[int(w)] = (Path(tr), Path(lin))
WS = sorted(RUNS)
FIRST, LAST = pd.Timestamp("2014-01-02"), pd.Timestamp("2024-04-30")
ESTS, SETS = ("rf_skx",), ("baseline", "all_features")
NAME = {"rf_skx": "Random forest (oneDAL)", "baseline": "HAR", "all_features": "full"}
PAIRS = [
    (a, b)
    for a, b in ((1000, 2000), (2000, 3000), (3000, 3500), (1000, 3000))
    if a in RUNS and b in RUNS
]


def evaluation_days(rv: pd.Series) -> pd.DatetimeIndex:
    """score_trees_final.evaluation_days: the evaluation days of the span, checked against RE.history()."""
    s = rv[(rv.index.hour == 16) & (rv.index.minute == 0)]
    d = pd.DatetimeIndex(s.index.normalize())
    keep = (s.to_numpy() > 0) & np.isfinite(s.to_numpy()) & ~d.isin(MR.EARLY)
    n = int(((d[keep] >= FIRST) & (d[keep] <= LAST)).sum())
    days = MR.evaluation_days(rv, FIRST, LAST, n)
    h = RE.history()
    hd = pd.DatetimeIndex(h.index[(h.index >= FIRST) & (h.index <= LAST)])
    assert np.array_equal(
        days.values.astype("datetime64[ns]"), hd.values.astype("datetime64[ns]")
    )
    return days


def nw(d: np.ndarray) -> dict:
    """Mean of a daily series and its Newey-West standard error (dm_test's lag, Bartlett weights, 1/T)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    T = d.size
    m = float(d.mean())
    dc = d - m
    lag = _nw_lag(T)
    lrv = float(np.mean(dc * dc))
    for k in range(1, lag + 1):
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * float(np.dot(dc[k:], dc[:-k]) / T)
    se = math.sqrt(lrv / T)
    t = m / se
    return {
        "mean": m,
        "se": se,
        "t": t,
        "p": math.erfc(abs(t) / math.sqrt(2.0)),
        "T": T,
        "lag": lag,
    }


def trees(run: Path, W: int) -> dict:
    f = pd.read_parquet(
        run / "forecasts.parquet",
        columns=["model", "predictor_set", "date", "plain", "rv_actual"],
    )
    m = pd.read_json(run / "forecasts.json", typ="series")
    assert (
        m["refit"] == "bar"
        and int(m["window_days"]) == W
        and list(m["span"]) == ["2014-01-02", "2024-04-30"]
    ), m
    out = {}
    for e in ESTS:
        for s in SETS:
            g = f[(f["model"] == e) & (f["predictor_set"] == s)]
            assert len(g) == 33970, (W, e, s, len(g))
            out[(e, s)] = g.set_index(pd.to_datetime(g["date"])).rename(
                columns={"rv_actual": "rv"}
            )[["plain", "rv"]]
    return out


rv = MR.core()
days = evaluation_days(rv)
FT, IDXW = {}, {}
for W in WS:
    trun, lrun = RUNS[W]
    FT[W] = trees(trun, W)
    fl = MR.forecasts("13", lrun)
    idx_l = MR.scored_set(fl, rv, days)
    idx = MR.scored_set({**fl, **FT[W]}, rv, days)
    assert idx.equals(idx_l), f"w{W}: the trees miss bars the linear rows score"
    IDXW[W] = idx
idx = IDXW[WS[0]]
for W in WS[1:]:
    idx = idx.intersection(IDXW[W])
same = all(IDXW[W].equals(idx) for W in WS)
on16 = idx[(idx.hour == 16) & (idx.minute == 0)]
print(
    f"{len(days)} days; common scored set {len(idx):,} bars, 16:00 {len(on16):,} days; windows' own sets equal: {same}"
)
assert same, {W: len(IDXW[W]) for W in WS}
L: dict = {}  # (W, est, set) -> (daily every-bar QLIKE, 16:00 QLIKE)
rows_q = []
for W in WS:
    for k, f in FT[W].items():
        q, _ = MR.loss(f, idx, rv)
        q16, _ = MR.loss(f, on16, rv)
        L[(W, *k)] = (MR.daily(q, idx), q16)
        rows_q.append(
            {
                "W": W,
                "model": k[0],
                "set": k[1],
                "qlike": float(q.mean()),
                "qlike_1600": float(q16.mean()),
            }
        )
Q = pd.DataFrame(rows_q)
rows_dm = []
for a, b in PAIRS:
    for e in ESTS:
        for s in SETS:
            for i, where in ((0, "every bar"), (1, "16:00")):
                r = dm_test(L[(b, e, s)][i], L[(a, e, s)][i])
                rows_dm.append(
                    {
                        "pair": f"{a}->{b}",
                        "model": e,
                        "set": s,
                        "bars": where,
                        "qlike_W1": float(np.mean(L[(a, e, s)][i])),
                        "qlike_W2": float(np.mean(L[(b, e, s)][i])),
                        "diff": float(r["mean_diff"]),
                        "dm": float(r["dm"]),
                        "p": float(r["p"]),
                        "T": int(r["T"]),
                        "hac_lag": int(r["hac_lag"]),
                    }
                )
D = pd.DataFrame(rows_dm)
rows_g, rows_did = [], []
for e in ESTS:
    for i, where in ((0, "every bar"), (1, "16:00")):
        for W in WS:
            g = nw(L[(W, e, "all_features")][i] - L[(W, e, "baseline")][i])
            rows_g.append(
                {
                    "model": e,
                    "bars": where,
                    "W": W,
                    "gap_full_minus_har": g["mean"],
                    "se": g["se"],
                    "t": g["t"],
                    "p": g["p"],
                }
            )
        for a, b in PAIRS:
            dF = L[(b, e, "all_features")][i] - L[(a, e, "all_features")][i]
            dH = L[(b, e, "baseline")][i] - L[(a, e, "baseline")][i]
            g = nw(dF - dH)
            rows_did.append(
                {
                    "model": e,
                    "bars": where,
                    "pair": f"{a}->{b}",
                    "full_change": float(np.mean(dF)),
                    "har_change": float(np.mean(dH)),
                    "extra_gain": g["mean"],
                    "se": g["se"],
                    "t": g["t"],
                    "p": g["p"],
                    "T": g["T"],
                    "hac_lag": g["lag"],
                }
            )
G, DD = pd.DataFrame(rows_g), pd.DataFrame(rows_did)
ec = []
for W in WS:
    p = RUNS[W][0] / "scores" / "econ.csv"
    x = pd.read_csv(p)
    x = x[x["model"].isin(ESTS)]
    assert (x["W"] == W).all(), p
    ec.append(x)
EC = pd.concat(ec, ignore_index=True)
OUT.mkdir(parents=True, exist_ok=True)
Q.to_csv(OUT / "qlike_by_w.csv", index=False)
D.to_csv(OUT / "dm_pairs.csv", index=False)
G.to_csv(OUT / "gap_by_w.csv", index=False)
DD.to_csv(OUT / "gap_change_did.csv", index=False)
EC.to_csv(OUT / "econ_by_w.csv", index=False)
T = [
    "# Pooled random forest (oneDAL, rf_skx), refit at every bar, 2014-01-02 .. 2024-04-30: the window learning curve",
    "",
    (
        f"One common scored set: {len(idx):,} bars on {len(days):,} days (16:00: {len(on16):,} days); every window's own "
        "scored set (the linear rows' of that window) is this set.  W = 1,000 / 2,000 / 3,000 / 3,500: CARC "
        "(insurance-selling-2026-09-19-e34-trees-final, results/pooled_rf_skx_2014start_w<W>)."
    ),
    "",
    "## QLIKE by window",
    "",
    "| model | set | "
    + " | ".join(f"W {W:,} all bars" for W in WS)
    + " | "
    + " | ".join(f"W {W:,} 16:00" for W in WS)
    + " |",
    "|---|---|" + "---|" * (2 * len(WS)),
]
for e in ESTS:
    for s in SETS:
        r = Q[(Q.model == e) & (Q.set == s)].set_index("W")
        T.append(
            f"| {NAME[e]} | {NAME[s]} | "
            + " | ".join(f"{r.loc[W, 'qlike']:.5f}" for W in WS)
            + " | "
            + " | ".join(f"{r.loc[W, 'qlike_1600']:.5f}" for W in WS)
            + " |"
        )
T += [
    "",
    "## DM, longer window against shorter (daily average QLIKE; negative = the longer window lower), DM (p)",
    "",
    "| model | set | bars | " + " | ".join(f"{a:,}->{b:,}" for a, b in PAIRS) + " |",
    "|---|---|---|" + "---|" * len(PAIRS),
]
for e in ESTS:
    for s in SETS:
        for where in ("every bar", "16:00"):
            cells = []
            for a, b in PAIRS:
                x = D[
                    (D.pair == f"{a}->{b}")
                    & (D.model == e)
                    & (D.set == s)
                    & (D.bars == where)
                ].iloc[0]
                cells.append(f"{x['diff']:+.5f}: {x.dm:+.2f} ({x.p:.3f})")
            T.append(f"| {NAME[e]} | {NAME[s]} | {where} | " + " | ".join(cells) + " |")
T += [
    "",
    "## Gap full - HAR by window (daily average QLIKE difference; NW t in parentheses)",
    "",
    "| model | bars | " + " | ".join(f"W {W:,}" for W in WS) + " |",
    "|---|---|" + "---|" * len(WS),
]
for e in ESTS:
    for where in ("every bar", "16:00"):
        r = G[(G.model == e) & (G.bars == where)].set_index("W")
        T.append(
            f"| {NAME[e]} | {where} | "
            + " | ".join(
                f"{r.loc[W, 'gap_full_minus_har']:+.5f} ({r.loc[W, 't']:+.2f})"
                for W in WS
            )
            + " |"
        )
T += [
    "",
    "## Change of the gap between windows: extra gain = full's change - HAR's change (negative = full gains more)",
    "",
    "| model | bars | pair | full change | HAR change | extra gain | NW se | t | p |",
    "|---|---|---|---|---|---|---|---|---|",
]
for _, x in DD.iterrows():
    T.append(
        f"| {NAME[x.model]} | {x.bars} | {x.pair} | {x.full_change:+.5f} | {x.har_change:+.5f} | {x.extra_gain:+.5f} | "
        f"{x.se:.5f} | {x.t:+.2f} | {x.p:.3f} |"
    )
T += [
    "",
    "## Sign strategy by window, 614 post-warmup option days (score_trees_final.py econ.csv): D6 HAR / full, ΔD6 (bootstrap p)",
    "",
    "| model | W | D6 mid HAR / full | ΔD6 mid (p) | D6 b/a HAR / full | ΔD6 b/a (p) | D1 HAR / full (LW p) | SR614 HAR / full |",
    "|---|---|---|---|---|---|---|---|",
]
for _, x in EC.sort_values(["model", "W"]).iterrows():
    T.append(
        f"| {NAME[x.model]} | {int(x.W):,} | {x.d6_mid_har:.2f} / {x.d6_mid_full:.2f} | {x.dd6_mid:+.2f} ({x.d6_boot_p_mid:.3f}) | "
        f"{x.d6_ba_har:.2f} / {x.d6_ba_full:.2f} | {x.dd6_ba:+.2f} ({x.d6_boot_p_ba:.3f}) | {x.d1_mid_har:.2f} / "
        f"{x.d1_mid_full:.2f} ({x.lw_p_mid:.3f}) | {x.sr614_mid_har:.2f} / {x.sr614_mid_full:.2f} |"
    )
(OUT / "tables.txt").write_text("\n".join(T) + "\n", encoding="utf-8")
print("\n".join(T))
