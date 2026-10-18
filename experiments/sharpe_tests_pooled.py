"""Sharpe-ratio tests on the straddle returns of the pooled intraday models (2026-10-17).

Read-only on every input; writes only results/sharpe_tests_pooled_2026-10-17/.

DATA (the same daily series as Table E1, writeup/generated/rehaul_econ_e1_control.tex, built by
writeup/make_rehaul_economics.py: load() -> trade_stats())
  results/close_master_table/master_table_daily.parquet, column ret_mid (= q R, mid-quote return of the sign
  strategy q = sign(plain - iv_var)), keys:
    pooled intraday linear  poolpt_{ridge,lasso,enet}_{all_features,baseline}   (E1 Panel B = *_all_features)
    OLS HAR benchmark       sub_ols_baseline  (closing-interval OLS; no pooled OLS series exists)
    closing trees, fixed daily rung   subtree_daily_{all_features,baseline}_{lgbm,xgb,rf}
    closing trees, E1's ex-post picks subtree_optuna_tp25_all_features_lgbm, subtree_xgb_all_features,
                                      subtree_optuna_tp1_k10_all_features_rf  (re-derived by E1's rule:
                                      max mid Sharpe on days 253-866 among each estimator's all_features specs)
  Short straddle: -R with R = q * ret_mid of sub_ols_baseline (make_rehaul_economics.load: SHORT mid = -R).
  Pooled trees (experiment only, never in the paper's tables): results/pooled_trees_2026-10-07/yhat_pooltree_<est>_<set>
  and the bank arms results/spxw_pnl/yhat_tree00 (lgbm all_features) / yhat_tree10 (xgb all_features), traded with
  master_table_close.trade_days exactly as experiments/pooled_trees_score.py does; rf all_features is the INTERIM
  16:00 table.  These are Optuna-tuned bank arms refit at every bar, not a fixed-hyperparameter rung.

SAMPLES  oos614 = days t > 252 (253-866), the sample of every printed Sharpe ratio in E1 (E17);
         all866 = all option days, the sample of E1's HAC t-statistics.

TESTS (per-period daily Sharpe; SR_ann = SR_daily sqrt(252))
  Lo (2002): moments (r - mu, (r - mu)^2 - sigma^2), Newey-West Bartlett long-run covariance with lag
    floor(4 (T/100)^(2/9)), delta method on SR = mu / sigma; z = SR / SE, two-sided normal p.  se_iid = sqrt((1 + SR^2/2)/T).
  Ledoit-Wolf (2008): Delta = SR_i - SR_n on date-aligned pairs; v = (mu_i, mu_n, gamma_i, gamma_n), gamma = E r^2;
    HAC (same Bartlett kernel and lag; LW use a prewhitened QS kernel) delta-method SE -> p_hac.  Studentized circular
    block bootstrap (LW Algorithm 3.1 / Remark 3.2, as in R bootTimeInference::boot.time.inference): l = floor(T/b)
    blocks, Psi* = mean_j zeta_j zeta_j' with zeta_j = b^{-1/2} sum over block j of (y* - mean y*);
    d = |Delta| / s(Delta), d* = |Delta* - Delta| / s*(Delta*), PV = (#{d* >= d} + 1)/(M + 1), M = 4999.
    Block size: b = 5 primary (p_boot), b = 10 sensitivity, plus LW's calibration (bootTimeInference::block.size.calibrate:
    pseudo-sequences by circular block bootstrap with b_av = 5, b in {1,3,6,10}, K = 100 pseudo-sequences (R default 1000; cut for run time), M = 199,
    pick b whose 95% interval coverage of the true Delta is closest to 0.95) with its p-value as p_boot_bcal.
  Seed 20261017.

Run:  python experiments/sharpe_tests_pooled.py [--no-pooled-trees] [--no-calib]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "close_master_table"
OUT = ROOT / "results" / "sharpe_tests_pooled_2026-10-17"
ANN = math.sqrt(252.0)
WARMUP = 252
N_OPT = 866
M_BOOT = 4999
SEED = 20261017
B_PRIMARY, B_SENS = 5, 10
B_CAL = (1, 3, 6, 10)

SHORT, OLS = "always_short", "sub_ols_baseline"
EST_NAME = {"ridge": "Ridge", "lasso": "Lasso", "enet": "Elastic net", "lgbm": "LightGBM", "xgb": "XGBoost",
            "rf": "Random forest"}
# E1's printed SR (mid) at time of writing (writeup/generated/rehaul_econ_e1_control.tex, read 2026-10-17)
E1_SR = {
    SHORT: 0.34, OLS: 1.64,
    "poolpt_enet_all_features": 1.91, "poolpt_lasso_all_features": 1.56, "poolpt_ridge_all_features": 1.73,
    "TREE_lgbm": 1.86, "TREE_xgb": 1.59, "TREE_rf": 1.38,
}


# ============================================================================= statistics
def p2(z: float) -> float:
    """Two-sided standard-normal p-value (no scipy: the local miniconda scipy is ABI-broken against numpy 2)."""
    return math.erfc(abs(float(z)) / math.sqrt(2.0))


def nw_lag(T: int) -> int:
    return int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


def lrv(e: np.ndarray, L: int) -> np.ndarray:
    """Bartlett long-run covariance of the (T, k) centered moment matrix e (divides by T)."""
    T = e.shape[0]
    S = e.T @ e / T
    for k in range(1, L + 1):
        G = e[k:].T @ e[:-k] / T
        S += (1.0 - k / (L + 1.0)) * (G + G.T)
    return S


def lo_test(r: np.ndarray) -> dict:
    T = len(r)
    mu, s2 = r.mean(), r.var(ddof=0)
    sd = math.sqrt(s2)
    sr = mu / r.std(ddof=1)  # the paper's point estimate (ddof 1)
    L = nw_lag(T)
    e = np.column_stack([r - mu, (r - mu) ** 2 - s2])
    S = lrv(e, L)
    g = np.array([1.0 / sd, -mu / (2.0 * sd ** 3)])
    se = math.sqrt(g @ S @ g / T)
    z = sr / se
    return {"T": T, "SR_daily": sr, "SR_ann": sr * ANN, "SE": se, "SE_ann": se * ANN, "stat": z,
            "p_hac": p2(z), "se_iid": math.sqrt((1 + sr ** 2 / 2) / T), "lag": L}


def _delta_grad(m):
    mi, mn, gi, gn = m
    vi, vn = gi - mi ** 2, gn - mn ** 2
    d = mi / math.sqrt(vi) - mn / math.sqrt(vn)
    grad = np.array([gi / vi ** 1.5, -gn / vn ** 1.5, -0.5 * mi / vi ** 1.5, 0.5 * mn / vn ** 1.5])
    return d, grad


def lw_hac(ri: np.ndarray, rn: np.ndarray, L: int) -> tuple[float, float]:
    T = len(ri)
    Y = np.column_stack([ri, rn, ri ** 2, rn ** 2])
    m = Y.mean(0)
    d, grad = _delta_grad(m)
    Psi = lrv(Y - m, L)
    se = math.sqrt(grad @ Psi @ grad / T)
    return d, se


def _boot_delta_se(ri, rn, b, M, rng):
    """M circular-block bootstrap draws: Delta* (ddof 0 Sharpe) and the block-based SE (LW Remark 3.2)."""
    T = len(ri)
    l = T // b
    Tb = l * b
    Y = np.column_stack([ri, rn, ri ** 2, rn ** 2])
    Yc = np.concatenate([Y, Y[: b]], 0)  # circular
    out_d = np.empty(M)
    out_se = np.empty(M)
    chunk = max(1, 200000 // Tb)
    off = np.arange(b)
    for s in range(0, M, chunk):
        n = min(chunk, M - s)
        starts = rng.integers(0, T, size=(n, l))
        idx = (starts[:, :, None] + off[None, None, :])  # (n, l, b)
        Yb = Yc[idx]  # (n, l, b, 4); idx < T + b
        m = Yb.reshape(n, Tb, 4).mean(1)  # (n, 4)
        zeta = (Yb - m[:, None, None, :]).sum(2) / math.sqrt(b)  # (n, l, 4)
        Psi = np.einsum("nlj,nlk->njk", zeta, zeta) / l
        mi, mn, gi, gn = m.T
        vi, vn = gi - mi ** 2, gn - mn ** 2
        d = mi / np.sqrt(vi) - mn / np.sqrt(vn)
        grad = np.stack([gi / vi ** 1.5, -gn / vn ** 1.5, -0.5 * mi / vi ** 1.5, 0.5 * mn / vn ** 1.5], 1)
        se = np.sqrt(np.einsum("nj,njk,nk->n", grad, Psi, grad) / Tb)
        out_d[s: s + n], out_se[s: s + n] = d, se
    return out_d, out_se


def lw_boot_p(ri, rn, b, M, rng, d_hat, se_hat) -> float:
    db, seb = _boot_delta_se(ri, rn, b, M, rng)
    dstar = np.abs(db - d_hat) / seb
    dd = abs(d_hat) / se_hat
    return float((np.sum(dstar >= dd) + 1) / (M + 1))


def lw_calibrate(ri, rn, L, rng, b_vec=B_CAL, K=100, M=199, b_av=5, alpha=0.05) -> tuple[int, dict]:
    """bootTimeInference::block.size.calibrate (pseudo-data from a circular block bootstrap with block b_av)."""
    T = len(ri)
    Y = np.column_stack([ri, rn])
    d_true, _ = lw_hac(ri, rn, L)
    l = int(math.ceil(T / b_av))
    cover = {b: 0 for b in b_vec}
    for _ in range(K):
        starts = rng.integers(0, T, size=l)
        idx = ((starts[:, None] + np.arange(b_av)[None, :]) % T).ravel()[:T]
        P = Y[idx]
        pi, pn = P[:, 0], P[:, 1]
        d_k, se_k = lw_hac(pi, pn, L)
        for b in b_vec:
            db, seb = _boot_delta_se(pi, pn, b, M, rng)
            q = np.quantile(np.abs(db - d_k) / seb, 1 - alpha)
            if abs(d_k - d_true) <= q * se_k:
                cover[b] += 1
    cov = {b: c / K for b, c in cover.items()}
    b_opt = min(b_vec, key=lambda b: (abs(cov[b] - (1 - alpha)), b))
    return b_opt, cov


# ============================================================================= data
def load_master():
    d = pd.read_parquet(RES / "master_table_daily.parquet", columns=["key", "day", "q", "ret_mid"])
    d["day"] = pd.to_datetime(d["day"]).dt.normalize()
    w = d.pivot(index="day", columns="key", values="ret_mid").sort_index()
    q_ols = d[d.key == OLS].set_index("day")["q"].reindex(w.index)
    w[SHORT] = -(q_ols * w[OLS])  # -R
    assert len(w) == N_OPT, len(w)
    return w


def ex_post_trees(w: pd.DataFrame) -> dict:
    """The E1 tree row = highest mid Sharpe (days 253-866) among each estimator's all_features specs in the paper model set."""
    ps = set(pd.read_csv(RES / "paper_model_set.csv")["key"])
    out = {}
    for est in ("lgbm", "xgb", "rf"):
        cands = [k for k in w.columns if k in ps and k.startswith("subtree_") and "_all_features" in k
                 and (k.endswith(f"_{est}") or f"_{est}_" in k)]
        sr = {k: w[k].iloc[WARMUP:].mean() / w[k].iloc[WARMUP:].std(ddof=1) * ANN for k in cands}
        out[est] = max(sr, key=sr.get)
    return out


def load_pooled_trees(days: pd.DatetimeIndex) -> tuple[dict, list[str]]:
    """Pooled-tree mid returns via the master-table loader (experiments/pooled_trees_score.py's tables and scorer)."""
    for p in (ROOT, ROOT / "notebooks", ROOT / "experiments", ROOT / "writeup" / "master_table_gui"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import master_table_close as mtc  # noqa: E402
    mtc.set_arm_root("linear_subsection_dedup")
    deck = pd.read_parquet(mtc.base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    tgt = mtc.load_target()
    mtc._init(tgt, deck.index)
    tables = {
        ("lgbm", "baseline"): "results/pooled_trees_2026-10-07/yhat_pooltree_lgbm_baseline.parquet",
        ("lgbm", "all_features"): "results/spxw_pnl/yhat_tree00.parquet",
        ("xgb", "baseline"): "results/pooled_trees_2026-10-07/yhat_pooltree_xgb_baseline.parquet",
        ("xgb", "all_features"): "results/spxw_pnl/yhat_tree10.parquet",
        ("rf", "baseline"): "results/pooled_trees_2026-10-07/yhat_pooltree_rf_baseline.parquet",
        ("rf", "all_features"): "results/pooled_trees_2026-10-07/yhat_pooltree_rf_all_features_INTERIM1600.parquet",
    }
    out, notes = {}, []
    for (est, bset), rel in tables.items():
        path = ROOT / rel
        key = f"pooltree_{est}_{bset}"
        if not path.is_file():
            notes.append(f"{key}: missing {rel}")
            continue
        spec = mtc.Spec(path.stem.removeprefix("yhat_"), path.stem.removeprefix("yhat_"), "per-bar tree", "table", str(path))
        res = mtc.load_one(spec)
        if res["error"]:
            notes.append(f"{key}: load error {res['error']}")
            continue
        f = res["frame"]
        g = f.set_index(f.index.normalize())
        dk = deck.loc[days]
        t = mtc.trade_days(g.loc[days, "plain"], dk)
        if len(t) != len(days):
            notes.append(f"{key}: covers {len(t)} of {len(days)} days")
            continue
        r = t["mid"].reindex(days)
        # gate: the 866-day mid Sharpe must equal experiments/pooled_trees_score.py's levels.csv
        lv_dir = ROOT / "results/pooled_trees_2026-10-07" / (f"score_{est}" + ("_interim1600" if "INTERIM" in rel else ""))
        lv = pd.read_csv(lv_dir / "levels.csv")
        want = float(lv[(lv["bucket"] == bset) & (lv["rung"] == "pooled") & lv["forecast"].str.startswith("pooled")]["sharpe_mid"].iloc[0])
        got = float(r.mean() / r.std(ddof=1) * ANN)
        if abs(got - want) > 1e-8:
            notes.append(f"{key}: GATE FAIL Sharpe {got} vs levels.csv {want}")
        else:
            notes.append(f"{key}: gate ok (866-day Sharpe {got:.4f} = {lv_dir.name}/levels.csv)")
        out[key] = (r, rel)
    return out, notes


# ============================================================================= main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pooled-trees", action="store_true")
    ap.add_argument("--no-calib", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    w = load_master()
    days = w.index
    trees_ep = ex_post_trees(w)

    series = []  # (name, key, group, predictor set, source)
    src_m = "results/close_master_table/master_table_daily.parquet:ret_mid"
    series.append(("Short straddle", SHORT, "benchmark", "-", src_m + " (-R via sub_ols_baseline q)"))
    series.append(("OLS, HAR benchmark", OLS, "closing-interval (E1 Panel A)", "HAR benchmark", src_m))
    for bset in ("all_features", "baseline"):
        for e in ("ridge", "lasso", "enet"):
            k = f"poolpt_{e}_{bset}"
            grp = "pooled intraday linear" + (" (E1 Panel B)" if bset == "all_features" else " (not in E1)")
            series.append((EST_NAME[e], k, grp, "full" if bset == "all_features" else "HAR benchmark", src_m))
    for bset in ("all_features", "baseline"):
        for e in ("lgbm", "xgb", "rf"):
            series.append((EST_NAME[e], f"subtree_daily_{bset}_{e}", "closing-interval tree, fixed daily rung",
                           "full" if bset == "all_features" else "HAR benchmark", src_m))
    for e in ("lgbm", "xgb", "rf"):
        series.append((EST_NAME[e], trees_ep[e], "closing-interval tree, E1 ex-post pick", "full", src_m))

    notes: list[str] = []
    if not a.no_pooled_trees:
        try:
            pt, nts = load_pooled_trees(days)
            notes += nts
            for k, (r, rel) in pt.items():
                w[k] = r.to_numpy(float)
                est, bset = k.split("_", 2)[1], k.split("_", 2)[2]
                grp = "pooled intraday tree (experiment; Optuna bank arm, refit every bar)"
                if "INTERIM" in rel:
                    grp += " INTERIM"
                series.append((EST_NAME[est], k, grp, "full" if bset == "all_features" else "HAR benchmark",
                               rel + " via master_table_close.trade_days"))
        except Exception as ex:  # noqa: BLE001
            notes.append(f"pooled trees not loaded: {type(ex).__name__}: {ex}")

    rng = np.random.default_rng(SEED)
    rows = []
    calib = []
    for sample, sl in (("oos614", slice(WARMUP, None)), ("all866", slice(None))):
        R = {k: w[k].to_numpy(float)[sl] for _, k, *_ in series}
        for name, k, grp, pset, src in series:
            r = R[k]
            assert np.isfinite(r).all(), k
            lo = lo_test(r)
            e1 = E1_SR.get(k)
            if e1 is None and k in trees_ep.values():
                e1 = E1_SR[f"TREE_{[e for e, v in trees_ep.items() if v == k][0]}"]
            rows.append({"sample": sample, "test": "Lo2002 SR=0", "series": name, "key": k, "group": grp,
                         "predictor_set": pset, "comparator": "0", **{c: lo[c] for c in
                         ("T", "SR_daily", "SR_ann", "SE", "SE_ann", "stat", "p_hac", "se_iid", "lag")},
                         "e1_SR_ann": e1 if sample == "oos614" else np.nan, "source": src})
            if k == SHORT:
                continue
            for comp in (SHORT, OLS):
                if comp == k:
                    continue
                rn = R[comp]
                L = nw_lag(len(r))
                d, se = lw_hac(r, rn, L)
                p_b5 = lw_boot_p(r, rn, B_PRIMARY, M_BOOT, rng, d, se)
                p_b10 = lw_boot_p(r, rn, B_SENS, M_BOOT, rng, d, se)
                b_cal, p_bcal, cov = np.nan, np.nan, {}
                if sample == "oos614" and not a.no_calib:
                    b_cal, cov = lw_calibrate(r, rn, L, rng)
                    p_bcal = {5: p_b5, 10: p_b10}.get(b_cal) or lw_boot_p(r, rn, b_cal, M_BOOT, rng, d, se)
                    calib.append({"key": k, "comparator": comp, "b_cal": b_cal, **{f"cover_b{b}": v for b, v in cov.items()}})
                sr_i = r.mean() / r.std(ddof=1)
                sr_n = rn.mean() / rn.std(ddof=1)
                rows.append({"sample": sample, "test": "LW2008 dSR", "series": name, "key": k, "group": grp,
                             "predictor_set": pset, "comparator": "Short straddle" if comp == SHORT else "OLS, HAR benchmark",
                             "T": len(r), "SR_daily": sr_i, "SR_ann": sr_i * ANN, "SR_comp_daily": sr_n,
                             "dSR_daily": d, "dSR_ann": d * ANN, "SE": se, "SE_ann": se * ANN, "stat": d / se,
                             "p_hac": p2(d / se), "p_boot": p_b5, "block": B_PRIMARY,
                             "p_boot_b10": p_b10, "b_cal": b_cal, "p_boot_bcal": p_bcal, "lag": L, "M": M_BOOT,
                             "source": src})
            print(sample, k, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sharpe_tests_pooled.csv", index=False, float_format="%.6g")
    if calib:
        pd.DataFrame(calib).to_csv(OUT / "block_calibration.csv", index=False)
    (OUT / "notes.txt").write_text("\n".join([f"E1 ex-post tree picks: {trees_ep}", *notes]) + "\n", encoding="utf-8")
    print("\n".join([f"E1 ex-post tree picks: {trees_ep}", *notes]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
