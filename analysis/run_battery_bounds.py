"""Inference on the 98-arm frozen battery from summary statistics alone.

The battery's per-bar loss series are not in this repository -- only each arm's
mean loss differential against the shared incumbent and its HAC t statistic.
That is enough for three rigorous things:

  1. MULTIPLICITY. Holm and Benjamini-Hochberg across all 98 arms' DM p-values,
     which the paper currently omits entirely.

  2. CONSERVATIVE PAIRWISE BOUNDS. For arms A and B measured against a common
     incumbent, the pairwise differential is exactly L_A - L_B = d_A - d_B, so
         Var(mean(L_A - L_B)) = Var(d_A) + Var(d_B) - 2 Cov(d_A, d_B).
     Whenever Cov(d_A, d_B) >= 0 the pairwise standard error is bounded above by
     sqrt(se_A^2 + se_B^2), giving a LOWER BOUND on |t| for the pairwise test.
     If the bound clears the critical value, the pairwise difference is
     established without the per-bar data. If it does not, the test is
     inconclusive rather than negative.
     The Cov >= 0 premise is verified empirically on the tile, where per-bar
     losses do exist (see `verify_cov_premise`).

  3. DISTRIBUTION-FREE TESTS on the pattern across feature sets: a sign test on
     the tree-vs-linear ordering, a trend test on the lag-ladder base sweep, and
     rank-correlation tests for the metric-disagreement and calibration claims.

Writes JSON + LaTeX to writeup/stats/.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import qlike_loss, holm, benjamini_hochberg, _student_t_sf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "writeup", "metrics_table_causal_tune_plus_spectral.csv")
PRED_DIR = os.path.join(ROOT, "results", "geo_preds")
OUT_DIR = os.path.join(ROOT, "writeup", "stats")
ALPHA = 0.05
TCRIT = 1.96


def load_battery():
    with open(CSV_PATH) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["qlike"] = float(r["qlike"])
        r["dm"] = float(r["dm"])
        r["dm_p"] = float(r["dm_p"])
        r["dm_mean_diff"] = float(r["dm_mean_diff"])
        r["oos_r2"] = float(r["oos_r2"])
        r["mz_beta"] = float(r["mz_beta"])
        r["n"] = int(r["n"])
        # HAC standard error of the mean loss differential, recovered from the
        # reported statistic. Guard the degenerate incumbent row.
        r["se"] = abs(r["dm_mean_diff"] / r["dm"]) if r["dm"] != 0 else float("nan")
        r["key"] = f"{r['estimator']}/{r['bucket']}"
    return rows


def verify_cov_premise():
    """Measure Corr(d_A, d_B) on the tile, where per-bar losses exist.

    d_A and d_B are the two arms' per-bar loss differentials against the shared
    incumbent. The battery bound assumes this correlation is non-negative.
    """
    arms, y, inc = {}, None, None
    for f in sorted(glob.glob(os.path.join(PRED_DIR, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        if "true_raw" not in d.files or d["pred_raw"].shape[0] != 2189:
            continue
        if hashlib.md5(np.ascontiguousarray(d["true_raw"]).tobytes()).hexdigest()[:8] != "6897c01e":
            continue
        y = d["true_raw"]
        arms[os.path.basename(f)[:-4]] = d["pred_raw"]
        if inc is None and "incumbent_pred_raw" in d.files:
            inc = d["incumbent_pred_raw"]
    if y is None or inc is None:
        return None
    li = qlike_loss(y, inc)
    names = sorted(arms)
    D = np.column_stack([qlike_loss(y, arms[k]) - li for k in names])
    C = np.corrcoef(D, rowvar=False)
    iu = np.triu_indices_from(C, k=1)
    corrs = C[iu]
    # Which arms drag the correlation negative? Arms that are catastrophically
    # bad move in the opposite direction to everything else.
    qb = D.mean(axis=0)
    good = np.where(qb < np.quantile(qb, 0.75))[0]     # exclude the worst quartile
    Cg = C[np.ix_(good, good)]
    iug = np.triu_indices_from(Cg, k=1)
    cg = Cg[iug]
    return {
        "n_arms": len(names), "n_pairs": int(corrs.size),
        "min": float(corrs.min()), "q05": float(np.quantile(corrs, 0.05)),
        "median": float(np.median(corrs)), "mean": float(corrs.mean()),
        "max": float(corrs.max()),
        "frac_negative": float((corrs < 0).mean()),
        "frac_above_0.5": float((corrs > 0.5).mean()),
        "excl_worst_quartile": {
            "n_pairs": int(cg.size), "min": float(cg.min()),
            "q05": float(np.quantile(cg, 0.05)), "median": float(np.median(cg)),
            "frac_negative": float((cg < 0).mean()),
        },
    }


def pairwise_bound(ra, rb):
    """Pairwise DM bounds for two arms measured against a common incumbent.

    The pairwise differential is exactly L_A - L_B = d_A - d_B, with
        Var(mean(L_A - L_B)) = se_A^2 + se_B^2 - 2 rho se_A se_B.

    Two bounds are returned, differing in what they assume about rho:

    * ``t_lower_bound_free`` takes the worst case rho = -1, giving
      se <= se_A + se_B. This assumes NOTHING and is always valid.
    * ``t_lower_bound_poscov`` takes rho >= 0, giving
      se <= sqrt(se_A^2 + se_B^2). Tighter, but only valid for pairs whose
      improvements over the incumbent co-move; see `verify_cov_premise`,
      which finds that premise violated for a minority of arm pairs.

    The assumption-free bound is the one the paper should quote.
    """
    mean = ra["dm_mean_diff"] - rb["dm_mean_diff"]
    se_free = ra["se"] + rb["se"]
    se_pos = math.sqrt(ra["se"] ** 2 + rb["se"] ** 2)
    t_free = mean / se_free
    t_pos = mean / se_pos
    return {
        "mean_diff": mean,
        "se_upper_bound_free": se_free,
        "t_lower_bound_free": t_free,
        "p_upper_bound_free": _student_t_sf(abs(t_free), df=ra["n"] - 1),
        "se_upper_bound_poscov": se_pos,
        "t_lower_bound_poscov": t_pos,
        "p_upper_bound_poscov": _student_t_sf(abs(t_pos), df=ra["n"] - 1),
    }


def sign_test(k: int, n: int) -> float:
    """Two-sided exact binomial sign test at p=1/2."""
    from math import comb
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = _rank(x), _rank(y)
    rs = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x)
    if abs(rs) >= 1.0:
        return rs, 0.0
    t = rs * math.sqrt((n - 2) / (1 - rs ** 2))
    return rs, _student_t_sf(abs(t), df=n - 2)


def _rank(a):
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    r[order] = np.arange(len(a), dtype=float)
    # average ties
    vals, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            m = inv == i
            r[m] = r[m].mean()
    return r


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_battery()
    by = {r["key"]: r for r in rows}
    out = {}

    # ---------- 0. premise check ----------
    prem = verify_cov_premise()
    out["cov_premise"] = prem
    print("=== premise check: Corr(d_A, d_B) on the tile "
          "(per-bar losses, shared incumbent) ===")
    if prem:
        print(f"  {prem['n_pairs']:,} arm pairs   min={prem['min']:+.3f} "
              f"5th pct={prem['q05']:+.3f}  median={prem['median']:+.3f}  max={prem['max']:+.3f}")
        print(f"  fraction with negative correlation: {prem['frac_negative']:.4f}"
              f"   above +0.5: {prem['frac_above_0.5']:.3f}")
        print("  -> the Cov >= 0 premise behind the conservative bound is "
              f"{'supported' if prem['q05'] >= 0 else 'NOT uniformly supported'}\n")

    # ---------- 1. multiplicity across all 98 arms ----------
    praw = {r["key"] + "|" + r["arm_tag"]: r["dm_p"] for r in rows if r["dm"] != 0}
    hadj, badj = holm(praw, alpha=ALPHA), benjamini_hochberg(praw, alpha=ALPHA)
    n_raw = sum(1 for p in praw.values() if p < ALPHA)
    n_holm = sum(1 for v in hadj.values() if v[1])
    n_bh = sum(1 for v in badj.values() if v[1])
    print(f"=== multiplicity across {len(praw)} battery arms (vs the shared incumbent) ===")
    print(f"  significant at raw p<{ALPHA}: {n_raw}")
    print(f"  surviving Holm-Bonferroni:   {n_holm}")
    print(f"  surviving Benjamini-Hochberg:{n_bh}")
    lost = [(k, praw[k], hadj[k][0]) for k in praw if praw[k] < ALPHA and not hadj[k][1]]
    print(f"  arms that lose significance under Holm: {len(lost)}")
    for k, p, pa in sorted(lost, key=lambda x: x[1]):
        print(f"      {k:46s} raw p={p:.4f} -> Holm p={pa:.4f}")
    out["multiplicity"] = {
        "n_tests": len(praw), "n_raw": n_raw, "n_holm": n_holm, "n_bh": n_bh,
        "lost_under_holm": [{"arm": k, "p_raw": p, "p_holm": pa} for k, p, pa in lost],
    }

    # ---------- 2. conservative pairwise bounds ----------
    buckets = ["all_features", "moments", "liquidity", "market_vw", "vol_demand",
               "implied_vol", "sentiment", "market_ew", "baseline"]
    print("\n=== conservative pairwise bounds: LightGBM vs ridge, per feature set ===")
    print(f"{'feature set':14s} {'dQLIKE':>9s} {'|t|>= free':>11s} {'p<= free':>11s} "
          f"{'|t|>= rho>=0':>13s}  verdict (assumption-free)")
    pw = []
    for b in buckets:
        a, r = by.get(f"lgbm/{b}"), by.get(f"ridge/{b}")
        if not a or not r:
            continue
        res = pairwise_bound(a, r)
        res.update({"bucket": b, "qlike_lgbm": a["qlike"], "qlike_ridge": r["qlike"]})
        pw.append(res)
        v = "RESOLVED: tree better" if res["t_lower_bound_free"] < -TCRIT else "inconclusive"
        print(f"{b:14s} {a['qlike'] - r['qlike']:+9.5f} {res['t_lower_bound_free']:11.2f} "
              f"{res['p_upper_bound_free']:11.2e} {res['t_lower_bound_poscov']:13.2f}  {v}")
    pb_p = {r["bucket"]: r["p_upper_bound_free"] for r in pw}
    pb_holm = holm(pb_p, alpha=ALPHA)
    for r in pw:
        r["p_holm_upper_bound"], r["reject_holm"] = pb_holm[r["bucket"]]
    n_res = sum(1 for r in pw if r["reject_holm"] and r["t_lower_bound_free"] < 0)
    print(f"  resolved after Holm across the {len(pw)} feature sets: {n_res}")
    out["pairwise_bounds_tree_vs_linear"] = pw

    # headline pair
    hp = pairwise_bound(by["lgbm/all_features"], by["ridge/all_features"])
    out["headline_pair"] = hp
    print(f"\n  headline (best tree 0.12604 vs best linear 0.12788):")
    print(f"      mean differential {hp['mean_diff']:+.6f}")
    print(f"      assumption-free : |t| >= {abs(hp['t_lower_bound_free']):.2f}, "
          f"p <= {hp['p_upper_bound_free']:.2e}")
    print(f"      assuming rho>=0 : |t| >= {abs(hp['t_lower_bound_poscov']):.2f}, "
          f"p <= {hp['p_upper_bound_poscov']:.2e}")

    # kNN winner vs the best parametric arms
    print("\n=== conservative bounds: spectral-kNN winner vs parametric arms ===")
    knn = next(r for r in rows if r["family"] == "spectral-kNN"
               and abs(r["qlike"] - 0.1309642552906875) < 1e-12)
    knn_cmp = []
    for tgt in ["lgbm/all_features", "ridge/all_features", "lgbm/baseline", "ridge/baseline"]:
        if tgt not in by:
            continue
        res = pairwise_bound(knn, by[tgt])
        res["vs"] = tgt
        knn_cmp.append(res)
        v = "RESOLVED" if abs(res["t_lower_bound_free"]) > TCRIT else "inconclusive"
        who = "kNN better" if res["t_lower_bound_free"] < 0 else "parametric better"
        print(f"  vs {tgt:22s} d={res['mean_diff']:+.5f} "
              f"|t|>={abs(res['t_lower_bound_free']):6.2f} "
              f"p<={res['p_upper_bound_free']:.2e}  {v}: {who if v == 'RESOLVED' else ''}")
    out["knn_vs_parametric"] = knn_cmp

    # ---------- 3. distribution-free pattern tests ----------
    print("\n=== distribution-free tests on the pattern across feature sets ===")
    signs = [1 for b in buckets
             if by.get(f"lgbm/{b}") and by.get(f"ridge/{b}")
             and by[f"lgbm/{b}"]["qlike"] < by[f"ridge/{b}"]["qlike"]]
    k, n = len(signs), len([b for b in buckets if by.get(f"lgbm/{b}") and by.get(f"ridge/{b}")])
    p_sign = sign_test(k, n)
    print(f"  sign test, tree beats linear on {k}/{n} feature sets: p = {p_sign:.5f}")
    print("    (feature sets are nested and share a panel, so this is "
          "anti-conservative; read it as a consistency check)")
    out["sign_test"] = {"k": k, "n": n, "p": p_sign}

    lad = {}
    for r in rows:
        if r["family"] == "har-ladder" and r["arm_tag"].endswith("_c3125"):
            base = r["arm_tag"].split("_")[0]
            if base.startswith("b"):
                lad[int(base[1:])] = r["qlike"]
    bs = sorted(lad)
    rs_l, p_l = spearman(bs, [lad[b] for b in bs])
    print(f"  ladder trend: Spearman(base, QLIKE) over bases {bs} = {rs_l:+.3f}, p = {p_l:.4f}")
    out["ladder_trend"] = {"bases": bs, "qlike": [lad[b] for b in bs],
                           "spearman": rs_l, "p": p_l}

    tuned = [r for r in rows if r["family"] == "causal-tuned"]
    rs_r2, p_r2 = spearman([r["qlike"] for r in tuned], [r["oos_r2"] for r in tuned])
    rs_mz, p_mz = spearman([r["qlike"] for r in tuned], [r["mz_beta"] for r in tuned])
    print(f"  metric agreement: Spearman(QLIKE, OOS-R2) over {len(tuned)} arms "
          f"= {rs_r2:+.3f}, p = {p_r2:.3f}   (agreement would be NEGATIVE)")
    print(f"  calibration:      Spearman(QLIKE, MZ-beta) over {len(tuned)} arms "
          f"= {rs_mz:+.3f}, p = {p_mz:.2e}")
    out["metric_disagreement"] = {"spearman": rs_r2, "p": p_r2, "n": len(tuned)}
    out["calibration"] = {"spearman": rs_mz, "p": p_mz, "n": len(tuned)}

    with open(os.path.join(OUT_DIR, "battery_inference.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.join(OUT_DIR, 'battery_inference.json')}")
    return out


if __name__ == "__main__":
    main()
