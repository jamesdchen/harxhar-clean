"""Audit: every inference statistic quoted in the paper must trace to the JSONs.

Checks the specific claims made in prose against writeup/stats/*.json, so a
hand-typed number cannot silently drift from what the analysis produced.
Run: python3 analysis/audit_paper_numbers.py
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS = os.path.join(ROOT, "writeup", "stats")

bat = json.load(open(os.path.join(STATS, "battery_inference.json")))
tile = json.load(open(os.path.join(STATS, "tile_inference.json")))
hk = json.load(open(os.path.join(STATS, "hawkes_kernel.json")))
ke = json.load(open(os.path.join(STATS, "kernel_experiments.json")))
vb = json.load(open(os.path.join(STATS, "b2_kernel_verification.json")))
np_ = json.load(open(os.path.join(STATS, "nested_probes.json")))
C = {c["label"]: c for c in tile["contrasts"]}

fails = []


def check(claim, quoted, actual, tol=5e-3, rel=False):
    ok = (abs(quoted - actual) / max(abs(actual), 1e-12) < tol) if rel \
        else (abs(quoted - actual) < tol)
    print(f"  {'OK  ' if ok else 'FAIL'}  {claim:52s} paper={quoted:<12g} computed={actual:<12g}")
    if not ok:
        fails.append(claim)


print("=== multiplicity (Section: Multiplicity Across the Battery) ===")
m = bat["multiplicity"]
check("97 arms tested", 97, m["n_tests"], tol=0.5)
check("93 significant at raw p<0.05", 93, m["n_raw"], tol=0.5)
check("91 survive Holm", 91, m["n_holm"], tol=0.5)
check("93 survive BH", 93, m["n_bh"], tol=0.5)
check("2 arms lose significance", 2, len(m["lost_under_holm"]), tol=0.5)

print("\n=== pairwise bounds (Section: The Pairwise Model-Class Test) ===")
hp = bat["headline_pair"]
check("headline |t| >= 2.19", 2.19, abs(hp["t_lower_bound_free"]), tol=0.01)
check("headline p <= 0.029", 0.029, hp["p_upper_bound_free"], tol=1e-3)
check("rho>=0 variant |t| >= 3.00", 3.00, abs(hp["t_lower_bound_poscov"]), tol=0.01)
check("rho>=0 variant p <= 0.0027", 0.0027, hp["p_upper_bound_poscov"], tol=1e-4)
pw = bat["pairwise_bounds_tree_vs_linear"]
check("all 9 resolved after Holm", 9, sum(1 for r in pw if r["reject_holm"]), tol=0.5)
others = [abs(r["t_lower_bound_free"]) for r in pw if r["bucket"] != "all_features"]
check("other eight span from |t|>=3.73", 3.73, min(others), tol=0.01)
check("other eight span to |t|>=5.91", 5.91, max(others), tol=0.01)
check("sign test p = 0.0039", 0.0039, bat["sign_test"]["p"], tol=1e-4)

print("\n=== kNN bounds (Section: The headline cell is a selected minimum) ===")
k = {r["vs"]: r for r in bat["knn_vs_parametric"]}
check("vs lgbm/all_features |t| >= 6.96", 6.96,
      abs(k["lgbm/all_features"]["t_lower_bound_free"]), tol=0.01)
check("vs ridge/all_features |t| >= 6.22", 6.22,
      abs(k["ridge/all_features"]["t_lower_bound_free"]), tol=0.01)
check("vs ridge/baseline |t| >= 14.49", 14.49,
      abs(k["ridge/baseline"]["t_lower_bound_free"]), tol=0.01)
check("vs lgbm/baseline |t| >= 0.30 (inconclusive)", 0.30,
      abs(k["lgbm/baseline"]["t_lower_bound_free"]), tol=0.01)

print("\n=== auxiliary tests ===")
check("ladder Spearman = +1.000", 1.000, bat["ladder_trend"]["spearman"], tol=1e-6)
check("metric-disagreement rho = +0.152", 0.152, bat["metric_disagreement"]["spearman"], tol=1e-3)
check("metric-disagreement p = 0.32", 0.32, bat["metric_disagreement"]["p"], tol=5e-3)
check("calibration rho = +0.741", 0.741, bat["calibration"]["spearman"], tol=1e-3)
check("calibration p = 5.8e-9", 5.8e-9, bat["calibration"]["p"], tol=0.1, rel=True)

print("\n=== tile DM contrasts (Section: Per-Bar Inference on the Campaign Tile) ===")
check("tile n = 2189", 2189, tile["panel"]["n"], tol=0.5)
check("tile arms = 161", 161, tile["panel"]["n_arms"], tol=0.5)
check("xgb direct +0.102", 0.102, C["trees direct: xgb vs incumbent"]["delta"], tol=5e-4)
check("xgb direct t = 6.96", 6.96, C["trees direct: xgb vs incumbent"]["t"], tol=0.01)
check("lgbm direct +0.077", 0.077, C["trees direct: lgbm vs incumbent"]["delta"], tol=5e-4)
check("lgbm direct t = 6.46", 6.46, C["trees direct: lgbm vs incumbent"]["t"], tol=0.01)
check("k8000 vs k200 = -0.0204", -0.0204, C["local k=8000 vs k=200"]["delta"], tol=5e-5)
check("k8000 vs k200 t = -6.59", -6.59, C["local k=8000 vs k=200"]["t"], tol=0.01)
check("k16000 vs k8000 = +0.0029", 0.0029, C["local k=16000 vs k=8000"]["delta"], tol=5e-5)
check("k16000 vs k8000 t = 3.74", 3.74, C["local k=16000 vs k=8000"]["t"], tol=0.01)
check("cap256 vs cap32 = -0.0074", -0.0074, C["cap256 vs cap32"]["delta"], tol=5e-5)
check("cap256 vs cap32 t = -4.16", -4.16, C["cap256 vs cap32"]["t"], tol=0.01)
check("cap256 vs uncapped = -0.0077", -0.0077, C["cap256 vs uncapped"]["delta"], tol=5e-5)
check("cap256 vs uncapped t = -3.83", -3.83, C["cap256 vs uncapped"]["t"], tol=0.01)
check("cap256 vs cap128 = -0.0026", -0.0026, C["cap256 vs cap128"]["delta"], tol=5e-5)
check("cap256 vs cap128 t = -3.50", -3.50, C["cap256 vs cap128"]["t"], tol=0.01)
check("enet mid vs ridge = +0.0052", 0.0052, C["enet mid vs ridge"]["delta"], tol=5e-5)
check("enet mid vs ridge t = 3.15", 3.15, C["enet mid vs ridge"]["t"], tol=0.01)

print("\n=== claims that do NOT survive ===")
check("lgbm residual t = 1.06", 1.06, C["trees residual: lgbm vs ridge cadence"]["t"], tol=0.01)
check("lgbm residual p = 0.29", 0.29, C["trees residual: lgbm vs ridge cadence"]["p"], tol=5e-3)
check("xgb residual t = 1.11", 1.11, C["trees residual: xgb vs ridge cadence"]["t"], tol=0.01)
check("xgb residual p = 0.27", 0.27, C["trees residual: xgb vs ridge cadence"]["p"], tol=5e-3)
check("Fourier vs gates delta = -0.0037", -0.0037,
      C["Fourier clock vs session gates"]["delta"], tol=5e-5)
check("Fourier vs gates t = -1.64", -1.64, C["Fourier clock vs session gates"]["t"], tol=0.01)
check("Fourier vs gates p = 0.10", 0.101, C["Fourier clock vs session gates"]["p"], tol=5e-3)
check("Fourier vs gates p_Holm = 0.91", 0.908,
      C["Fourier clock vs session gates"]["p_holm"], tol=5e-3)
check("rank vs divide t = 2.73", 2.73, C["rank vs divide"]["t"], tol=0.01)
check("rank vs divide p_raw = 0.0064", 0.0064, C["rank vs divide"]["p"], tol=1e-4)
check("rank vs divide p_Holm = 0.071", 0.0705, C["rank vs divide"]["p_holm"], tol=1e-3)

print("\n=== the parsimony ties ===")
check("legchamp+clock tie p = 0.998", 0.998, C["tie: legchamp+clock vs champion"]["p"], tol=5e-3)
check("prune32+corr tie p = 0.951", 0.951, C["tie: prune32+corr vs champion"]["p"], tol=5e-3)
check("rawc stack tie p = 0.763", 0.763, C["tie: rawc stack vs champion"]["p"], tol=5e-3)

print("\n=== model confidence sets ===")
mcs = tile["mcs"]
check("top20 keeps 20", 20, len(mcs["top20"]["surviving"]), tol=0.5)
check("all-arms keeps 123", 123, len(mcs["all_arms"]["surviving"]), tol=0.5)
check("penalty keeps 2", 2, len(mcs["penalty"]["surviving"]), tol=0.5)
check("local_k keeps 3", 3, len(mcs["local_k"]["surviving"]), tol=0.5)
check("trees_vs_linear keeps 4", 4, len(mcs["trees_vs_linear"]["surviving"]), tol=0.5)
sens = tile["mcs_block_sensitivity"]
check("block sensitivity: all variants keep 20", 4,
      sum(1 for v in sens.values() if v["n_surviving"] == 20), tol=0.5)

print("\n=== Hawkes kernel statistics (Section: descriptive analysis) ===")
ce = hk["channel_exponents_clustered"]
check("beta_hat = 1.033", 1.033, ce["mean"], tol=5e-4)
check("clustered SE = 0.036", 0.036, ce["se"], tol=5e-4)
check("CI low = 0.963", 0.963, ce["ci_low"], tol=1e-3)
check("CI high = 1.104", 1.104, ce["ci_high"], tol=1e-3)
check("41 channel clusters", 41, ce["n_channels"], tol=0.5)
check("244 channel-session fits", 244, hk["channel_exponents"]["n_fits"], tol=0.5)
check("vs 1.36: t = -9.1", -9.1, ce["tests"]["paper's 1.36"]["t"], tol=0.05)
check("vs 1.36: p = 2.6e-11", 2.6e-11, ce["tests"]["paper's 1.36"]["p"], tol=0.05, rel=True)
check("vs 1.0: t = +0.9", 0.93, ce["tests"]["Hawkes literature 1.0"]["t"], tol=0.05)
check("vs 1.0: p = 0.36", 0.359, ce["tests"]["Hawkes literature 1.0"]["p"], tol=5e-3)
check("vs 1.6: t = -15.8", -15.81, ce["tests"]["rough-vol implied 1.6"]["t"], tol=0.05)
check("vs 1.6: p = 8.4e-19", 8.43e-19, ce["tests"]["rough-vol implied 1.6"]["p"], tol=0.05, rel=True)

p0 = hk["pooled"]
check("branching analogue = -0.038", -0.038, p0["branching_analogue"], tol=1e-6)
check("identity error < 1e-12", 0.0, abs(p0["identity_lhs"] - p0["identity_rhs"]), tol=1e-12)
check("pooled negative lags = 8", 8, p0["n_negative_lags"], tol=0.5)
check("pooled negative mass = 10.3%", 0.103, p0["negative_mass_share"], tol=5e-4)
check("self-kernel beta = 1.51", 1.505, p0["powerlaw"]["beta"], tol=5e-3)
check("self-kernel SE = 0.54", 0.537, p0["powerlaw"]["se"], tol=5e-3)
check("self-kernel fitted on 4 lags", 4, p0["powerlaw"]["n_points"], tol=0.5)
check("power-law R2 = 0.80", 0.797, p0["powerlaw"]["r2"], tol=5e-3)
check("exponential R2 = 0.44", 0.441, p0["exponential"]["r2"], tol=5e-3)

ss = hk["sessions"]
check("open negative mass = 87.8%", 0.878, ss["open"]["negative_mass_share"], tol=1e-3)
check("after negative mass = 86.3%", 0.863, ss["after"]["negative_mass_share"], tol=1e-3)
check("close negative mass = 56.9%", 0.569, ss["close"]["negative_mass_share"], tol=1e-3)
check("after n_hat = -0.260", -0.26, ss["after"]["branching_analogue"], tol=1e-6)
check("close n_hat = -0.130", -0.13, ss["close"]["branching_analogue"], tol=1e-6)
check("overnight n_hat = +0.020", 0.02, ss["overnight"]["branching_analogue"], tol=1e-6)
check("overnight cos = +0.83", 0.826, ss["overnight"]["cos_to_pooled"], tol=5e-3)
check("open cos = -0.55", -0.547, ss["open"]["cos_to_pooled"], tol=5e-3)
check("session rank-1 share = 43.8%", 0.438, hk["session_svd"]["variance_share"][0], tol=1e-3)
top3 = sum(hk["session_svd"]["variance_share"][:3])
check("session top-3 share = 84%", 0.844, top3, tol=5e-3)

exo = hk["exogenous_operator"]
r90 = [v["rank90"] for v in exo.values()]
check("operator rank90 min = 4", 4, min(r90), tol=0.5)
check("operator rank90 max = 6", 6, max(r90), tol=0.5)
nf = [v["negative_lag_fraction"] for v in exo.values()]
check("cross-kernel neg fraction min = 44.7%", 0.447, min(nf), tol=1e-3)
check("cross-kernel neg fraction max = 50.4%", 0.504, max(nf), tol=1e-3)
sv1 = [v["variance_share"][0] for v in exo.values()]
check("operator sv1 share min = 28%", 0.284, min(sv1), tol=5e-3)
check("operator sv1 share max = 58%", 0.576, max(sv1), tol=5e-3)

print("\n=== kernel experiments (Section: Reproducibility of the Kernel Estimate) ===")
val = ke["_validation"]
check("rebuild correlation = +0.65", 0.645, val["correlation"], tol=5e-3)
check("rebuilt n_hat = +1.000", 1.0003, val["rebuilt_n_hat"], tol=1e-3)
check("archived n_hat = -0.038", -0.038, val["archived_n_hat"], tol=1e-6)
check("validation NOT reproduced", 0, 1 if val["reproduced"] else 0, tol=0.5)
cells = {k: v for k, v in ke.items() if not k.startswith("_")}
check("7 experiment cells", 7, len(cells), tol=0.5)
check("all cells n = 242,934", 242934,
      min(v["n_rows"] for v in cells.values()), tol=0.5)
nh = [v["branching_analogue"] for v in cells.values()]
check("n_hat min = +0.906", 0.9055, min(nh), tol=1e-3)
check("n_hat max = +1.079", 1.0795, max(nh), tol=1e-3)
nm = [v["negative_mass_share"] for v in cells.values()]
check("negative mass max < 0.4%", 0.004, max(nm), tol=1e-3)
bt = [v["powerlaw"]["beta"] for v in cells.values() if v["powerlaw"]]
check("rebuilt beta min = 1.43", 1.427, min(bt), tol=5e-3)
check("rebuilt beta max = 1.58", 1.581, max(bt), tol=5e-3)
check("drop exog: n_hat = +0.927", 0.9265,
      cells["divide/marginal/5-95"]["branching_analogue"], tol=1e-3)
check("drop divisor: n_hat = +1.079", 1.0795,
      cells["dummies/partial/5-95"]["branching_analogue"], tol=1e-3)
check("winsor 5/95 QLIKE = 0.2648", 0.26478,
      cells["divide/marginal/5-95"]["oos_qlike"], tol=1e-4)
check("winsor 1/99 QLIKE = 0.2449", 0.24493,
      cells["divide/marginal/1-99"]["oos_qlike"], tol=1e-4)
check("winsor none QLIKE = 0.2411", 0.24113,
      cells["divide/marginal/none"]["oos_qlike"], tol=1e-4)
check("winsorization cost = 0.024", 0.024,
      cells["divide/marginal/5-95"]["oos_qlike"] - cells["divide/marginal/none"]["oos_qlike"],
      tol=5e-4)
dv = ke["_divisor_alone"]
check("divisor alone QLIKE = 0.558", 0.55793, dv["qlike"], tol=1e-4)
check("unconditional mean QLIKE = 1.320", 1.31959, dv["qlike_unconditional_mean"], tol=1e-3)
best = min(v["oos_qlike"] for v in cells.values())
frac = (dv["qlike_unconditional_mean"] - dv["qlike"]) / (dv["qlike_unconditional_mean"] - best)
check("divisor closes 71% of the gap", 0.71, frac, tol=5e-3)

print("\n=== b2_mmap rebuild + nested probes ===")
v = vb["validation"]
check("archived reproduces: corr = 1.0000", 1.0, v["correlation"], tol=1e-4)
check("max |diff| = 0.0005", 0.0005, v["max_abs_diff"], tol=1e-4)
check("VERDICT reproduced", 1, 1 if v["reproduced"] else 0, tol=0.5)
A = np_["A. har only"]; B = np_["B. har + exog ladder"]; E = np_["E. FULL (archived)"]
check("marginal n_hat = +0.913", 0.9133, A["n_hat_raw"], tol=1e-3)
check("marginal negative mass = 0.0%", 0.0, A["neg_mass"], tol=1e-3)
check("marginal beta = 1.506", 1.506, A["powerlaw"]["beta"], tol=5e-3)
check("marginal beta SE = 0.035", 0.035, A["powerlaw"]["se"], tol=5e-3)
check("marginal fitted on 10 lags", 10, A["powerlaw"]["n_points"], tol=0.5)
check("+exog n_hat = -0.139", -0.1394, B["n_hat_raw"], tol=1e-3)
check("+exog negative mass = 16.4%", 0.164, B["neg_mass"], tol=1e-3)
check("full n_hat = -0.363", -0.3626, E["n_hat_raw"], tol=1e-3)
check("full negative mass = 14.4%", 0.144, E["neg_mass"], tol=1e-3)
check("full negative lags = 9", 9, E["neg_lags"], tol=0.5)

print("\n" + "=" * 74)
if fails:
    print(f"AUDIT FAILED ({len(fails)}):")
    for f in fails:
        print("   ", f)
    sys.exit(1)
print(f"AUDIT PASSED - all {len(open(__file__).read().split('check('))-1} quoted statistics "
      f"trace to writeup/stats/*.json")
