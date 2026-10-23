"""Part 1 of the 2014-start window sweep: the pre-test learning curve (read-only).

For every arm (model x predictor set) and window W, the validation predictions of the pre-test folds (fold 3, and the
larger common fold sets: folds 1-3 for W <= 3,000, folds 0-3 for W <= 2,500) give
  MSE   the validation MSE on the adjusted scale (the mean over the fold's rows; equal fold sizes);
  QLIKE r - ln r - 1, r = RV / plain, plain = pred_adj^2 x B, on the rows with RV > 0 and plain > 0 (pooled).
Changes are relative: 100 (L_b / L_a - 1) for W = a -> b (each step, and 1,000 -> W); the full - HAR gap
100 (L_full / L_HAR - 1); "full gains more" = the 1,000 -> W change of the full set minus that of the HAR set (in %
points).  Standard errors: from the per-day average validation loss (13 bars a day; the days are the same for every W),
the paired daily difference of the row losses on the rows valid in both, scaled so that its mean is the pooled
% change (d_t = 100 (S_b,t - S_a,t) / (N / T) / L_a, S the day's loss sum; for the gain difference full's d_t minus
HAR's), Newey-West (Bartlett, lag hac_inference.nw94_lag(T) = floor(4 (T/100)^(2/9)),
weights 1 - k / (lag + 1)); t = mean / SE, normal p.  T = 250 (fold 3), 750 (folds 1-3), 1,000 (folds 0-3) days.

usage: python part1_learning.py <hac_inference dir> <out prefix> <panel csv>
  panel csv: family,est,set,W,fold,path  (path: a parquet with date, pred_adj, true_adj, B, rv_actual)
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, sys.argv[1])
import hac_inference as hi  # noqa: E402

OUTP = sys.argv[2]
PANEL = pd.read_csv(sys.argv[3])
FOLDSETS = {"fold3": [3], "folds1-3": [1, 2, 3], "folds0-3": [0, 1, 2, 3]}
WS = sorted(PANEL["W"].unique())


def nw_se(d: np.ndarray) -> tuple[float, float, int]:
    T = len(d)
    L = hi.nw94_lag(T)
    lrv = hi.bartlett_lrv(d, L + 1)
    se = math.sqrt(lrv / T)
    return float(d.mean()), se, L


_cache: dict = {}


def losses(path: str) -> pd.DataFrame:
    if path not in _cache:
        f = pd.read_parquet(path)
        se = (f["pred_adj"].to_numpy(float) - f["true_adj"].to_numpy(float)) ** 2
        plain = f["pred_adj"].to_numpy(float) ** 2 * f["B"].to_numpy(float)
        rv = f["rv_actual"].to_numpy(float)
        ok = np.isfinite(rv) & (rv > 0) & np.isfinite(plain) & (plain > 0)
        r = np.where(ok, rv / np.where(ok, plain, 1.0), np.nan)
        _cache[path] = pd.DataFrame({"date": pd.to_datetime(f["date"]), "mse": se, "qlike": r - np.log(r) - 1.0})
    return _cache[path]


def arm_frame(family, est, s, W, folds) -> pd.DataFrame | None:
    g = PANEL[(PANEL.family == family) & (PANEL.est == est) & (PANEL.set == s) & (PANEL.W == W)
              & PANEL.fold.isin(folds)]
    if len(g) != len(folds):
        return None
    return pd.concat([losses(p) for p in g.sort_values("fold")["path"]], ignore_index=True)


def summary(fr: pd.DataFrame) -> dict:
    return dict(mse=float(fr["mse"].mean()), qlike=float(fr["qlike"].mean(skipna=True)), fr=fr)


def rel(sa: dict, sb: dict, c: str) -> pd.Series:
    """Daily series d_t whose mean is the pooled % change of loss c from a to b on the rows valid in both:
    d_t = 100 (S_b,t - S_a,t) / (N / T) / L_a, S = the day's sum of the row losses, N rows, T days, L_a = a's mean."""
    fa, fb = sa["fr"], sb["fr"]
    assert fa["date"].equals(fb["date"]), "validation rows differ"
    xa, xb = fa[c].to_numpy(float), fb[c].to_numpy(float)
    k = np.isfinite(xa) & np.isfinite(xb)
    day = fa["date"].dt.normalize()[k]
    s_a = pd.Series(xa[k]).groupby(day.to_numpy()).sum()
    s_b = pd.Series(xb[k]).groupby(day.to_numpy()).sum()
    N, T = int(k.sum()), len(s_a)
    return 100.0 * (s_b - s_a) / (N / T) / xa[k].mean()


rows, gaps = [], []
arms = PANEL[["family", "est", "set"]].drop_duplicates().itertuples(index=False)
arms = list(arms)
for fs, folds in FOLDSETS.items():
    summ = {}
    for a in arms:
        for W in WS:
            fr = arm_frame(a.family, a.est, a.set, W, folds)
            if fr is not None:
                summ[(a.family, a.est, a.set, W)] = summary(fr)
    for a in arms:
        Wk = [W for W in WS if (a.family, a.est, a.set, W) in summ]
        if len(Wk) < 2:
            continue
        base = summ[(a.family, a.est, a.set, Wk[0])]
        for i, W in enumerate(Wk):
            s = summ[(a.family, a.est, a.set, W)]
            row = dict(foldset=fs, family=a.family, est=a.est, set=a.set, W=W, T_days=int(s["fr"]["date"].dt.normalize().nunique()),
                       mse=s["mse"], qlike=s["qlike"])
            for c in ("mse", "qlike"):
                if i > 0:
                    p = summ[(a.family, a.est, a.set, Wk[i - 1])]
                    m, se, L = nw_se(rel(p, s, c).to_numpy())
                    row |= {f"{c}_step_pct": 100 * (s[c] / p[c] - 1), f"{c}_step_dpct_daily": m, f"{c}_step_se": se,
                            f"{c}_step_t": m / se, f"{c}_step_p": hi.normal_p(m / se), "nw_lag": L}
                    m, se, L = nw_se(rel(base, s, c).to_numpy())
                    row |= {f"{c}_vs{Wk[0]}_pct": 100 * (s[c] / base[c] - 1), f"{c}_vs{Wk[0]}_se": se,
                            f"{c}_vs{Wk[0]}_p": hi.normal_p(m / se)}
            rows.append(row)
    # the full - HAR gap by W, and the 1,000 -> W gain of the full set minus the HAR set's
    for fam, est in sorted({(a.family, a.est) for a in arms}):
        Wk = [W for W in WS if (fam, est, "baseline", W) in summ and (fam, est, "all_features", W) in summ]
        if not Wk:
            continue
        for W in Wk:
            h, f = summ[(fam, est, "baseline", W)], summ[(fam, est, "all_features", W)]
            g = dict(foldset=fs, family=fam, est=est, W=W)
            for c in ("mse", "qlike"):
                m, se, L = nw_se(rel(h, f, c).to_numpy())
                g |= {f"{c}_gap_pct": 100 * (f[c] / h[c] - 1), f"{c}_gap_se": se, f"{c}_gap_p": hi.normal_p(m / se)}
                if W != Wk[0]:
                    h0, f0 = summ[(fam, est, "baseline", Wk[0])], summ[(fam, est, "all_features", Wk[0])]
                    dd = rel(f0, f, c) - rel(h0, h, c)
                    assert dd.notna().all()
                    d = dd.to_numpy()
                    m, se, L = nw_se(d)
                    g |= {f"{c}_gain_full_minus_har_pp": (100 * (f[c] / f0[c] - 1)) - (100 * (h[c] / h0[c] - 1)),
                          f"{c}_gain_diff_se": se, f"{c}_gain_diff_p": hi.normal_p(m / se)}
            gaps.append(g)
R, G = pd.DataFrame(rows), pd.DataFrame(gaps)
R.to_csv(f"{OUTP}_curve.csv", index=False)
G.to_csv(f"{OUTP}_gap.csv", index=False)

LAB = {"ridge": "ridge", "lasso": "lasso", "enet": "elastic net", "lgbm": "LightGBM", "xgb": "XGBoost", "rf": "random forest"}
SL = {"baseline": "HAR", "all_features": "full"}


def f_(v, fmt):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


for fs in FOLDSETS:
    r = R[R.foldset == fs]
    if r.empty:
        continue
    print(f"\n### {fs}: validation loss by W (Δ% step = previous W -> W; Δ% vs {WS[0]}; (SE) NW, [p])")
    print("| arm | W | MSE | Δ% step (SE) [p] | Δ% vs base (SE) [p] | QLIKE | Δ% step (SE) [p] | Δ% vs base (SE) [p] |")
    print("|---|---|---|---|---|---|---|---|")
    for (fam, est, s), g in r.groupby(["family", "est", "set"], sort=False):
        base = g["W"].iloc[0]
        for _, x in g.iterrows():
            cells = [f"{LAB[est]} {SL[s]}", f"{int(x.W):,}", f"{x.mse:.6f}"]
            for c in ("mse", "qlike"):
                if c == "qlike":
                    cells.append(f"{x.qlike:.5f}")
                st = f"{f_(x.get(f'{c}_step_pct'), '{:+.2f}')} ({f_(x.get(f'{c}_step_se'), '{:.2f}')}) [{f_(x.get(f'{c}_step_p'), '{:.3f}')}]"
                vb = f"{f_(x.get(f'{c}_vs{base}_pct'), '{:+.2f}')} ({f_(x.get(f'{c}_vs{base}_se'), '{:.2f}')}) [{f_(x.get(f'{c}_vs{base}_p'), '{:.3f}')}]"
                cells += [st if x.W != base else "—", vb if x.W != base else "—"]
            print("| " + " | ".join(cells) + " |")
    gg = G[G.foldset == fs]
    print(f"\n### {fs}: full - HAR gap by W (%, SE, p) and the full set's extra gain over HAR from the first W (pp)")
    print("| model | W | MSE gap | MSE extra gain (pp) | QLIKE gap | QLIKE extra gain (pp) |")
    print("|---|---|---|---|---|---|")
    for _, x in gg.iterrows():
        def cell(c, k):
            if k == "gap":
                return f"{x[f'{c}_gap_pct']:+.2f} ({x[f'{c}_gap_se']:.2f}) [{x[f'{c}_gap_p']:.3f}]"
            v = x.get(f"{c}_gain_full_minus_har_pp")
            return "—" if v is None or np.isnan(v) else f"{v:+.2f} ({x[f'{c}_gain_diff_se']:.2f}) [{x[f'{c}_gain_diff_p']:.3f}]"
        print(f"| {LAB[x.est]} | {int(x.W):,} | {cell('mse', 'gap')} | {cell('mse', 'gain')} | {cell('qlike', 'gap')} | "
              f"{cell('qlike', 'gain')} |")
