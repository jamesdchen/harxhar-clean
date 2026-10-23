"""Cumulative W_a -> W_b changes of the pre-test validation loss (part1_learning.py's method: the paired per-day loss
difference scaled to the pooled % change on the rows both score, Newey-West, Bartlett, lag nw94_lag(T)), and the full
set's extra gain over HAR for the same pair (full's daily series minus HAR's), for the given (fold set, W_a, W_b).

usage: python part1_pairs.py <hac_inference dir> <panel csv> <out csv> <foldset>:<Wa>:<Wb> [...]
"""

import math
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, sys.argv[1])
import hac_inference as hi  # noqa: E402

PANEL = pd.read_csv(sys.argv[2])
OUT = sys.argv[3]
PAIRS = [(p.split(":")[0], int(p.split(":")[1]), int(p.split(":")[2])) for p in sys.argv[4:]]
FOLDSETS = {"fold3": [3], "folds1-3": [1, 2, 3], "folds0-3": [0, 1, 2, 3]}
_c: dict = {}


def frame(fam, est, s, W, folds) -> pd.DataFrame:
    k = (fam, est, s, W, tuple(folds))
    if k not in _c:
        g = PANEL[(PANEL.family == fam) & (PANEL.est == est) & (PANEL.set == s) & (PANEL.W == W) & PANEL.fold.isin(folds)]
        assert len(g) == len(folds), k
        parts = []
        for p in g.sort_values("fold")["path"]:
            f = pd.read_parquet(p)
            plain = f["pred_adj"].to_numpy(float) ** 2 * f["B"].to_numpy(float)
            rv = f["rv_actual"].to_numpy(float)
            ok = np.isfinite(rv) & (rv > 0) & np.isfinite(plain) & (plain > 0)
            r = np.where(ok, rv / np.where(ok, plain, 1.0), np.nan)
            parts.append(pd.DataFrame({"date": pd.to_datetime(f["date"]),
                                       "mse": (f["pred_adj"].to_numpy(float) - f["true_adj"].to_numpy(float)) ** 2,
                                       "qlike": r - np.log(r) - 1.0}))
        _c[k] = pd.concat(parts, ignore_index=True)
    return _c[k]


def rel(fa, fb, c) -> pd.Series:
    assert fa["date"].equals(fb["date"])
    xa, xb = fa[c].to_numpy(float), fb[c].to_numpy(float)
    k = np.isfinite(xa) & np.isfinite(xb)
    day = fa["date"].dt.normalize()[k].to_numpy()
    sa, sb = pd.Series(xa[k]).groupby(day).sum(), pd.Series(xb[k]).groupby(day).sum()
    return 100.0 * (sb - sa) / (int(k.sum()) / len(sa)) / xa[k].mean()


def nw(d: np.ndarray) -> tuple[float, float, float]:
    L = hi.nw94_lag(len(d))
    se = math.sqrt(hi.bartlett_lrv(d, L + 1) / len(d))
    return float(d.mean()), se, hi.normal_p(d.mean() / se)


rows = []
for fs, Wa, Wb in PAIRS:
    folds = FOLDSETS[fs]
    for fam, est in PANEL[["family", "est"]].drop_duplicates().itertuples(index=False):
        dd = {}
        for s in ("all_features", "baseline"):
            fa, fb = frame(fam, est, s, Wa, folds), frame(fam, est, s, Wb, folds)
            for c in ("mse", "qlike"):
                d = rel(fa, fb, c)
                dd[(s, c)] = d
                m, se, p = nw(d.to_numpy())
                rows.append(dict(foldset=fs, Wa=Wa, Wb=Wb, family=fam, est=est, set=s, crit=c, pct=m, se=se, p=p))
        for c in ("mse", "qlike"):
            x = (dd[("all_features", c)] - dd[("baseline", c)])
            assert x.notna().all()
            m, se, p = nw(x.to_numpy())
            rows.append(dict(foldset=fs, Wa=Wa, Wb=Wb, family=fam, est=est, set="full_minus_har", crit=c, pct=m, se=se, p=p))
R = pd.DataFrame(rows)
R.to_csv(OUT, index=False)
L = {"lgbm": "LightGBM", "xgb": "XGBoost", "rf": "RF", "ridge": "ridge", "lasso": "lasso", "enet": "enet"}
SL = {"all_features": "full", "baseline": "HAR", "full_minus_har": "extra gain (pp)"}
keys = [(fs, a, b) for fs, a, b in PAIRS]
print("| arm | crit | " + " | ".join(f"{fs} {a:,}→{b:,}" for fs, a, b in keys) + " |")
print("|---|---|" + "---|" * len(keys))
for (fam, est, s, c), g in R.groupby(["family", "est", "set", "crit"], sort=False):
    cells = []
    for fs, a, b in keys:
        x = g[(g.foldset == fs) & (g.Wa == a) & (g.Wb == b)].iloc[0]
        cells.append(f"{x.pct:+.2f}±{x.se:.2f} (p {x.p:.3f})")
    print(f"| {L[est]} {SL[s]} | {c.upper()} | " + " | ".join(cells) + " |")
