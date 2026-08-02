"""Per-bar forecast-comparison inference on the campaign tile.

Loads every arm in results/geo_preds that shares the main tile's target series,
builds the QLIKE loss matrix, and runs:
  1. pairwise Diebold-Mariano for the contrasts the paper's claims rest on,
     with Holm and Benjamini-Hochberg adjustment across the contrast family;
  2. Hansen-Lunde-Nason model confidence sets over the full arm set and over
     curated families;
  3. block-length sensitivity for the MCS.

Writes JSON + LaTeX to writeup/stats/.

NOTE ON PANEL. This tile is n = 2,189 bars. It is NOT the 218,934-bar frozen
battery panel, and no level here is comparable to a battery level. What the
tile can settle is whether the *claims* survive per-bar inference; the battery's
own pairwise question needs the battery's per-bar losses, which are not in this
repository (see run_battery_bounds.py for what can be said without them).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import (  # noqa: E402
    qlike_loss, dm_test, model_confidence_set, holm, benjamini_hochberg,
    politis_white_block_length,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR = os.path.join(ROOT, "results", "geo_preds")
OUT_DIR = os.path.join(ROOT, "writeup", "stats")
MAIN_TILE = "6897c01e"          # md5 prefix of the tile's true_raw
ALPHA_MCS = 0.10
ALPHA_DM = 0.05
B_BOOT = 5000
SEED = 20260802


def load_tile():
    arms, y = {}, None
    for f in sorted(glob.glob(os.path.join(PRED_DIR, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        if "true_raw" not in d.files or "pred_raw" not in d.files:
            continue
        if d["pred_raw"].shape[0] != 2189:
            continue
        if hashlib.md5(np.ascontiguousarray(d["true_raw"]).tobytes()).hexdigest()[:8] != MAIN_TILE:
            continue
        y = d["true_raw"]
        arms[os.path.basename(f)[:-4]] = d["pred_raw"]
        if "incumbent_pred_raw" in d.files and "incumbent" not in arms:
            arms["incumbent"] = d["incumbent_pred_raw"]
    if y is None:
        raise SystemExit("no tile arms found")
    return y, arms


# Each contrast is (label, arm_A, arm_B, claim). Negative t favours A.
CONTRASTS = [
    # --- the parsimony / tie claims -------------------------------------
    ("tie: legchamp+clock vs champion", "lcc_t1_final", "s_stack1000",
     "a nameable 5-shape state reproduces the wide champion"),
    ("tie: prune32+corr vs champion", "pcc_t1", "s_stack1000",
     "a 32-channel prune reproduces the wide champion"),
    ("tie: rawc stack vs champion", "hc_rawc", "s_stack1000",
     "the correction stack reproduces the wide champion"),

    # --- trees fail in both seats ---------------------------------------
    ("trees direct: lgbm vs incumbent", "pbt_lgbm_cad", "incumbent",
     "trees as direct forecaster on the kernel dictionary"),
    ("trees direct: xgb vs incumbent", "pbt_xgb_cad", "incumbent",
     "trees as direct forecaster on the kernel dictionary"),
    ("trees residual: lgbm vs ridge cadence", "pbt_lgbm_resid", "pbe_ridge_cad",
     "trees as residual reader vs the smooth linear reader"),
    ("trees residual: xgb vs ridge cadence", "pbt_xgb_resid", "pbe_ridge_cad",
     "trees as residual reader vs the smooth linear reader"),

    # --- selection loses to shrinkage -----------------------------------
    ("enet light vs ridge", "pbe_enet_3e-05", "pbe_ridge_cad",
     "light L1 that prunes little"),
    ("enet mid vs ridge", "pbe_enet_0.0003", "pbe_ridge_cad",
     "L1 that starts to bite"),
    ("enet heavy vs ridge", "pbe_enet_0.003", "pbe_ridge_cad",
     "L1 that bites hard"),

    # --- marginal adjustment: divide vs rank ----------------------------
    ("rank vs divide", "pb_rank_t1", "pb_panel_t1",
     "rank-transform the marginal instead of dividing"),

    # --- clock: continuous Fourier vs session gates ---------------------
    ("Fourier clock vs session gates", "pb_cfour_t1", "pb_cgate_t1",
     "continuous daily cycle vs discrete session gates"),

    # --- local averaging: neighbourhood size ----------------------------
    ("local k=8000 vs k=200", "i_loc_k8000", "i_loc_k200",
     "heavy local averaging vs sharp local structure"),
    ("local k=16000 vs k=8000", "i_loc_k16000", "i_loc_k8000",
     "does averaging keep paying"),

    # --- dimension reduction on a dense signal --------------------------
    ("PCovR-50 vs PCovR-25", "a_pcovr50", "a_pcovr25",
     "supervised DR retention"),
    ("PLS reg vs PCovR", "a_plsrg24000", "a_pcovr50",
     "which DR family"),

    # --- truncation / cap law -------------------------------------------
    ("cap256 vs cap32", "b2_cap256", "b2_cap32",
     "kernel truncation depth"),
    ("cap256 vs uncapped", "b2_cap256", "b2_uncap",
     "does removing truncation hurt"),
    ("cap256 vs cap128", "b2_cap256", "b2_cap128",
     "truncation fine structure"),
]

FAMILIES = {
    "top20": None,           # filled in at runtime: the 20 lowest-QLIKE arms
    "cap_ladder": ["b2_cap32", "b2_cap64", "b2_cap128", "b2_cap256", "b2_uncap", "incumbent"],
    "local_k": ["i_loc_k200", "i_loc_k1000", "i_loc_k4000", "i_loc_k8000",
                "i_loc_k16000", "incumbent"],
    "penalty": ["pbe_ridge_cad", "pbe_enet_3e-05", "pbe_enet_0.0003", "pbe_enet_0.003"],
    "trees_vs_linear": ["pbt_lgbm_resid", "pbt_xgb_resid", "pbe_ridge_cad",
                        "pbt_lgbm_cad", "pbt_xgb_cad", "incumbent"],
    "dim_reduction": ["a_pcovr25", "a_pcovr50", "a_pcovr75", "a_plsrg2400",
                      "a_plsrg24000", "a_plsrg240k", "a_pls12d8r", "a_omega0"],
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    y, arms = load_tile()
    names = sorted(arms)
    L = np.column_stack([qlike_loss(y, arms[k]) for k in names])
    qbar = L.mean(axis=0)
    order = np.argsort(qbar)
    rank = {names[i]: int(r + 1) for r, i in enumerate(order)}
    qmap = {names[i]: float(qbar[i]) for i in range(len(names))}
    col = {k: i for i, k in enumerate(names)}

    print(f"tile n={L.shape[0]}  arms={L.shape[1]}")
    print(f"best: {names[order[0]]} {qbar[order[0]]:.6f}   "
          f"incumbent: {qmap.get('incumbent', float('nan')):.6f} "
          f"(rank {rank.get('incumbent')})\n")

    # ---------------- pairwise DM over the contrast family ----------------
    results, skipped = [], []
    for label, a, b, claim in CONTRASTS:
        if a not in col or b not in col:
            skipped.append((label, [x for x in (a, b) if x not in col]))
            continue
        r = dm_test(L[:, col[a]], L[:, col[b]], h=1)
        results.append({
            "label": label, "claim": claim, "arm_a": a, "arm_b": b,
            "qlike_a": qmap[a], "qlike_b": qmap[b],
            "delta": qmap[a] - qmap[b],
            "t": r["t"], "p": r["p"], "se": r["se"], "lag": r["lag"],
        })

    praw = {r["label"]: r["p"] for r in results}
    hadj = holm(praw, alpha=ALPHA_DM)
    badj = benjamini_hochberg(praw, alpha=ALPHA_DM)
    for r in results:
        r["p_holm"], r["reject_holm"] = hadj[r["label"]]
        r["p_bh"], r["reject_bh"] = badj[r["label"]]

    print(f"=== pairwise DM, {len(results)} contrasts "
          f"(Holm/BH adjusted across the family) ===")
    print(f"{'contrast':40s} {'dQLIKE':>10s} {'t':>8s} {'p':>10s} {'p_Holm':>9s}  verdict")
    for r in sorted(results, key=lambda r: r["p"]):
        v = ("A better" if r["t"] < 0 else "B better") if r["reject_holm"] else "not resolved"
        print(f"{r['label']:40s} {r['delta']:+10.5f} {r['t']:8.2f} "
              f"{r['p']:10.2e} {r['p_holm']:9.2e}  {v}")
    for label, miss in skipped:
        print(f"  [skipped] {label}: missing {miss}")

    # ---------------- model confidence sets ----------------
    FAMILIES["top20"] = [names[i] for i in order[:20]]
    mcs_out = {}
    print(f"\n=== model confidence sets (alpha={ALPHA_MCS}, B={B_BOOT}) ===")
    for fam, members in FAMILIES.items():
        mem = [m for m in members if m in col]
        if len(mem) < 2:
            continue
        sub = L[:, [col[m] for m in mem]]
        res = model_confidence_set(sub, mem, alpha=ALPHA_MCS, B=B_BOOT, seed=SEED)
        mcs_out[fam] = res
        surv = sorted(res["surviving"], key=lambda k: qmap[k])
        print(f"\n  [{fam}] {len(mem)} arms -> MCS keeps {len(surv)}"
              f"  (block={res['block_len']:.1f})")
        for s in surv:
            print(f"      keep  {qmap[s]:.6f}  {s}")
        for nm, p, q in res["eliminated"]:
            print(f"      drop  {q:.6f}  {nm:24s} MCS p={p:.3f}")

    # full-set MCS
    res_all = model_confidence_set(L, names, alpha=ALPHA_MCS, B=B_BOOT, seed=SEED)
    mcs_out["all_arms"] = res_all
    print(f"\n  [all {len(names)} arms] MCS keeps {len(res_all['surviving'])}")
    print("      top of the surviving set: "
          + ", ".join(sorted(res_all["surviving"], key=lambda k: qmap[k])[:6]))

    # ---------------- block-length sensitivity ----------------
    print("\n=== MCS block-length sensitivity (top20) ===")
    base_b = mcs_out["top20"]["block_len"]
    sens = {}
    for mult in (0.5, 1.0, 2.0, 4.0):
        mem = FAMILIES["top20"]
        sub = L[:, [col[m] for m in mem]]
        r = model_confidence_set(sub, mem, alpha=ALPHA_MCS, B=B_BOOT,
                                 block_len=base_b * mult, seed=SEED)
        sens[f"x{mult}"] = {"block_len": r["block_len"], "n_surviving": len(r["surviving"])}
        print(f"    block={r['block_len']:6.1f}  survivors={len(r['surviving'])}")

    payload = {
        "panel": {"n": int(L.shape[0]), "n_arms": int(L.shape[1]),
                  "tile_hash": MAIN_TILE,
                  "note": "campaign tile, NOT the 218,934-bar frozen battery"},
        "qlike": qmap, "rank": rank,
        "contrasts": results, "skipped": skipped,
        "mcs": {k: {"surviving": v["surviving"],
                    "eliminated": v["eliminated"],
                    "pvalues": v["pvalues"],
                    "block_len": v["block_len"], "B": v["B"]}
                for k, v in mcs_out.items()},
        "mcs_block_sensitivity": sens,
        "settings": {"alpha_mcs": ALPHA_MCS, "alpha_dm": ALPHA_DM,
                     "B": B_BOOT, "seed": SEED},
    }
    with open(os.path.join(OUT_DIR, "tile_inference.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {os.path.join(OUT_DIR, 'tile_inference.json')}")
    return payload


if __name__ == "__main__":
    main()
