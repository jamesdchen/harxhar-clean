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

print("\n" + "=" * 74)
if fails:
    print(f"AUDIT FAILED ({len(fails)}):")
    for f in fails:
        print("   ", f)
    sys.exit(1)
print(f"AUDIT PASSED - all {len(open(__file__).read().split('check('))-1} quoted statistics "
      f"trace to writeup/stats/*.json")
