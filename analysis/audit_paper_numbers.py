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
mem = json.load(open(os.path.join(STATS, "mem_branching.json")))
smk = json.load(open(os.path.join(STATS, "supervised_metric_knn.json")))
fg = json.load(open(os.path.join(STATS, "frequency_and_gram.json")))
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

print("\n=== MEM / branching ratio (Section: multiplicative error model) ===")
hm, m11, vr = mem["har_mem"], mem["mem11"], mem["variance_ratio"]
check("panel = 242,462 bars", 242462, mem["n_bars"], tol=0.5)
check("HAR-MEM n = 0.9917", 0.9917, hm["n"], tol=5e-4)
check("HAR-MEM SE = 0.0043", 0.0043, hm["n_se"], tol=5e-4)
check("HAR-MEM CI low = 0.9834", 0.9834, hm["ci_low"], tol=1e-3)
check("HAR-MEM CI high = 1.0001", 1.0001, hm["ci_high"], tol=1e-3)
check("HAR-MEM OOS QLIKE = 0.24184", 0.24184, hm["oos_qlike"], tol=1e-4)
check("HAR-MEM all alpha > 0", 1, 1 if min(hm["alpha"]) > 0 else 0, tol=0.5)
check("MEM(1,1) n = 0.9753", 0.9753, m11["n"], tol=5e-4)
check("MEM(1,1) SE = 0.0042", 0.0042, m11["n_se"], tol=5e-4)
check("MEM(1,1) alpha+beta = 0.9894", 0.98945, m11["alpha_plus_beta"], tol=1e-4)
check("MEM(1,1) OOS QLIKE = 0.25707", 0.25707, m11["oos_qlike"], tol=1e-4)
check("multiplier at point est = 121", 121.0, 1/(1-hm["n"]), tol=1.0)
check("multiplier at CI low = 60", 60.0, 1/(1-hm["ci_low"]), tol=1.0)
check("variance-ratio H = 0.897", 0.8974, vr["H"], tol=1e-3)
check("variance-ratio H SE = 0.007", 0.0073, vr["H_se"], tol=1e-3)

print("\n=== retrieval-metric race (Section: the cause is supervision) ===")
R = smk["results"]["W24"]
DM = R["dm_vs_ambient"]
check("pool = 48,579 views", 48579, R["n_pool"], tol=0.5)
check("evaluation rows = 24,288", 24288, R["n_test"], tol=0.5)
check("ambient QLIKE = 0.17323", 0.17323, R["ambient"]["qlike"]["100"], tol=1e-5)
check("path-only QLIKE = 0.16347", 0.16347, R["ambient_path"]["qlike"]["100"],
      tol=1e-5)
check("path-only delta = -0.00976", -0.00976, DM["ambient_path@100"]["mean_diff"],
      tol=1e-5)
check("path-only t = -11.3", -11.3, DM["ambient_path@100"]["t"], tol=0.05)
check("operator d=6 QLIKE = 0.17339", 0.17339, R["op_d6"]["qlike"]["100"],
      tol=1e-5)
check("operator d=6 delta = +0.00015", 0.00015, DM["op_d6@100"]["mean_diff"],
      tol=1e-5)
check("operator d=6 t = +0.3", 0.3, DM["op_d6@100"]["t"], tol=0.05)
check("operator d=6 Holm p = 1.0", 1.0, R["holm"]["op_d6@100"]["adjusted_p"],
      tol=1e-9)
check("operator scores d=5 QLIKE = 0.17554", 0.17554,
      R["op_scores_d5"]["qlike"]["100"], tol=1e-5)
check("operator scores delta = +0.00230", 0.00230,
      DM["op_scores_d5@100"]["mean_diff"], tol=1e-5)
check("operator scores t = +5.0", 5.0, DM["op_scores_d5@100"]["t"], tol=0.05)
check("PCA d=6 QLIKE = 0.18002", 0.18002, R["pca_d6"]["qlike"]["100"], tol=1e-5)
check("PCA d=6 delta = +0.00678", 0.00678, DM["pca_d6@100"]["mean_diff"],
      tol=1e-5)
check("PCA d=6 t = +17.5", 17.5, DM["pca_d6@100"]["t"], tol=0.05)
check("eigenmap d=6 QLIKE = 0.18154", 0.18154, R["lap_d6"]["qlike"]["100"],
      tol=1e-5)
check("eigenmap d=6 delta = +0.00831", 0.00831, DM["lap_d6@100"]["mean_diff"],
      tol=1e-5)
check("eigenmap d=6 t = +18.2", 18.2, DM["lap_d6@100"]["t"], tol=0.05)
check("anchor-only QLIKE = 0.18322", 0.18322, R["anchor_qlike"], tol=1e-5)
check("ambient view d = 516", 516, R["ambient"]["d"], tol=0.5)
check("path view d = 24", 24, R["ambient_path"]["d"], tol=0.5)
check("exogenous ladder coordinates = 492", 492,
      R["ambient"]["d"] - R["ambient_path"]["d"], tol=0.5)
G = {tuple(bk["block"]): bk for bk in fg["gram_vs_signal"]["blocks"]}
check("PC1 variance share = 96.2%", 0.962, fg["gram_vs_signal"]["pc1_var_share"],
      tol=5e-4)
check("PC1 explained-y share = 0.02%", 0.0002,
      fg["gram_vs_signal"]["pc1_r2_share"], tol=5e-5)
check("PC6-20 variance share = 0.18%", 0.0018, G[(6, 20)]["var_share"], tol=5e-5)
check("PC6-20 explained-y share = 79.9%", 0.799, G[(6, 20)]["r2_share"],
      tol=1e-3)

print("\n=== stripped-down model (Section: The Stripped-Down Model) ===")
dfp = json.load(open(os.path.join(STATS, "dumb_full_panel.json")))
edf = json.load(open(os.path.join(STATS, "effective_df.json")))
ew = json.load(open(os.path.join(STATS, "exog_when.json")))
eb = json.load(open(os.path.join(STATS, "exog_breakdown.json")))
oc = json.load(open(os.path.join(STATS, "october_2023.json")))
pa = json.load(open(os.path.join(STATS, "prep_audit.json")))
pd_ = json.load(open(os.path.join(STATS, "primal_dual.json")))
W = dfp["whole_panel"]
check("panel walked = 218,909 bars", 218909, dfp["n_scored"], tol=0.5)
check("stripped QLIKE = 0.13280", 0.13280, W["dumb"]["qlike"], tol=1e-5)
check("stripped columns = 53", 53, W["dumb"]["cols"], tol=0.5)
check("backbone QLIKE = 0.13571", 0.13571, W["backbone"]["qlike"], tol=1e-5)
check("shaped-penalty QLIKE = 0.13415", 0.13415, W["shaped492"]["qlike"],
      tol=1e-5)
check("raw-492 ridge QLIKE = 0.13617", 0.13617, W["ridge492"]["qlike"],
      tol=1e-5)
check("raw ridge worse than backbone", 1,
      1 if W["ridge492"]["qlike"] > W["backbone"]["qlike"] else 0, tol=0.5)
check("stripped vs shaped t = -5.63", -5.63, dfp["dm_vs_dumb"]["shaped492"]["t"],
      tol=0.01)
check("stripped vs backbone t = -9.59", -9.59,
      dfp["dm_vs_dumb"]["backbone"]["t"], tol=0.01)
check("stripped beats backbone in 7/8 eras", 7, dfp["dumb_wins_eras"], tol=0.5)

print("\n=== effective degrees of freedom ===")
B = {b["block"]: b for b in edf["by_block"]}
check("ridge df blocks 1-7 = 95.4", 95.4, edf["ridge_df_blocks_1_7"], tol=0.05)
check("ridge df block 8 = 104.7", 104.7, edf["ridge_df_block_8"], tol=0.05)
check("shape df block 1 = 47.2", 47.2, B[1]["shape"], tol=0.05)
check("shape df block 8 = 60.0", 60.0, B[8]["shape"], tol=0.05)
check("backbone df = 13.0", 13.0, B[4]["backbone"], tol=0.05)
check("ridge df in PC21+ = 72.08", 72.08, edf["spend"]["ridge"]["pc21plus"],
      tol=0.01)
check("ridge df in PC6-20 = 14.12", 14.12, edf["spend"]["ridge"]["pc6_20"],
      tol=0.01)
check("primal-dual identity < 1e-8", 1, 1 if pd_["identity_check"]["relative"]
      < 1e-8 else 0, tol=0.5)

print("\n=== era stability and the imputation defect ===")
E = {b["block"]: b for b in ew["blocks"]}
check("era 8 ridge delta = +0.03321", 0.03321, E[8]["ridge_minus_bb"], tol=1e-5)
check("era 8 shape delta = +0.00666", 0.00666, E[8]["shape_minus_bb"], tol=1e-5)
check("era 8 stripped delta = +0.00347", 0.00347,
      [b for b in dfp["eras"] if b["block"] == 8][0]["dumb"], tol=1e-5)
check("ridge beats backbone in 7/8 eras", 7, ew["ridge_wins_blocks"], tol=0.5)
check("whole-panel ridge-vs-backbone = +0.00046", 0.00046,
      ew["whole_panel"]["ridge + exog"] - ew["whole_panel"]["backbone only"],
      tol=1e-5)
check("failing era bars = 27,376", 27376, eb["concentration"]["n_era"], tol=0.5)
check("worst 0.1% carry 47.2%", 0.472, eb["concentration"]["levels"]["0.001"]["share"],
      tol=1e-3)
check("worst 1% carry 90.5%", 0.905, eb["concentration"]["levels"]["0.01"]["share"],
      tol=1e-3)
check("worst 0.1% = 27 bars", 27, eb["concentration"]["levels"]["0.001"]["k"],
      tol=0.5)
check("median per-bar diff = +0.00069", 0.00069, eb["concentration"]["median"],
      tol=1e-5)
check("voldemand carries 97.2%", 0.972, oc["attribution"][0]["share"], tol=1e-3)
check("voldemand failing contribution = -4.145", -4.145,
      oc["attribution"][0]["failing"], tol=1e-3)
check("backbone part on failing bars = +0.928", 0.928, oc["backbone_part"],
      tol=1e-3)
check("0 channels moved 3x", 0, oc["n_channels_moved"], tol=0.5)
check("29/41 channels exceed 20 IQR", 29, pa["n_over_20_iqr"], tol=0.5)
check("23/41 show the pathology", 23, pa["n_pathology"], tol=0.5)
check("worst channel = 2260 IQR", 2260.4, pa["channels"][0]["scaled_absmax"],
      tol=0.1)
check("voldemand coverage = 0.223", 0.223, pa["channels"][0]["raw_coverage"],
      tol=1e-3)

print("\n" + "=" * 74)
if fails:
    print(f"AUDIT FAILED ({len(fails)}):")
    for f in fails:
        print("   ", f)
    sys.exit(1)
print(f"AUDIT PASSED - all {len(open(__file__).read().split('check('))-1} quoted statistics "
      f"trace to writeup/stats/*.json")
