"""Check (d): the flag-on pretune / select and the three full-set walk chunks, summarized (no out-of-sample score).

usage: python summarize_d.py <d_on_run> <d_off_run> <a_off> <out json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ON, OFF, AOFF, OUTJ = (Path(v) for v in sys.argv[1:5])
KK = Path(r"C:\Users\james\AppData\Local\Temp\kk_wt\results\pooled_linear_rerun_2026-10-22_13_2012_w1000_latestart_nofomcrel_kkt")
rep: dict = {}
# ---- pretune / select
fo = json.loads((ON / "frozen_penalties.json").read_text())
ff = json.loads((OFF / "frozen_penalties.json").read_text())
mo = pd.read_csv(ON / "pretune_mse.csv")
mf = pd.read_csv(OFF / "pretune_mse.csv")
sel = {}
for m in ("ridge", "lasso", "enet"):
    for s in ("baseline", "all_features"):
        a_on, a_off = fo[m][s], ff[m][s]
        c_on = float(mo[(mo.model == m) & (mo.predictor_set == s) & (mo.alpha == a_on)]["mse_mean"].iloc[0])
        c_off = float(mf[(mf.model == m) & (mf.predictor_set == s) & (mf.alpha == a_off)]["mse_mean"].iloc[0])
        pt_on = json.loads((ON / "_work" / "pretune" / s / m / "run.json").read_text())
        pt_off = json.loads((OFF / "_work" / "pretune" / s / m / "run.json").read_text())
        ext_sec = sum(
            json.loads(p.read_text())["wall_sec"] for p in (ON / "_work" / "pretune" / s / m).glob("ext_*/run.json")
        )
        sel[f"{m}/{s}"] = dict(
            penalty_on=a_on,
            penalty_off=a_off,
            grid_rule=fo["info"]["grid_extensions"][m][s],
            pre2012_val_mse_on=c_on,
            pre2012_val_mse_off=c_off,
            rel_change_pct=100.0 * (c_on / c_off - 1.0),
            pretune_wall_sec_on=round(pt_on["wall_sec"], 1),
            pretune_ext_wall_sec_on=round(ext_sec, 1),
            pretune_wall_sec_off=round(pt_off["wall_sec"], 1),
            repair_on=pt_on["repair"],
            std_left_out_beyond_mask_per_fold=[f.get("std_left_out_beyond_mask") for f in pt_on["folds"]],
            std_left_out_columns_fold3=(pt_on["folds"][-1].get("std_left_out_columns") or [])[:60],
            std_binom_per_fold=[f.get("std_binom") for f in pt_on["folds"]],
            s_quantiles_5_50_95_per_fold=[f.get("std_s_quantiles_5_50_95") for f in pt_on["folds"]],
        )
rep["select"] = sel
# ---- walks
names = None
wk = {}
for m, c, est in (("lasso", 0, "reclasso"), ("enet", 2, "reclasticnet"), ("ridge", 0, "ridge")):
    d = ON / "_work" / "walk" / "all_features" / m / f"c{c}"
    if not (d / "run.json").exists():  # still running when summarized
        wk[m] = dict(chunk=f"c{c}", status="not finished when summarized")
        continue
    run = json.loads((d / "run.json").read_text())
    z = np.load(d / "trace.npz")
    names = list(z["names"]) + ["intercept"]
    off = json.loads((AOFF / "_work" / "walk" / "all_features" / m / f"c{c}" / "run.json").read_text())
    kk = json.loads((KK / "_work" / "walk" / "all_features" / m / f"c{c}" / "run.json").read_text())
    S = z["std_seed_s"]
    ss = z["std_seed_solve"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = S[1:] / S[:-1]
    ok = np.isfinite(ratio)
    rel = np.where(ok, np.abs(ratio - 1.0), 0.0)
    i, j = np.unravel_index(int(np.argmax(rel)), rel.shape)
    top = []
    colmax = rel.max(axis=0)
    for jj in np.argsort(-colmax)[:8]:
        ii = int(np.argmax(rel[:, jj]))
        top.append(dict(col=names[jj], max_rel=float(colmax[jj]), s_before=float(S[ii, jj]), s_after=float(S[ii + 1, jj]),
                        seed_solve_after=int(ss[ii + 1])))
    g0 = run["g0"]
    tune = int(((ss - g0) % 250 == 0).sum())
    w = dict(
        chunk=f"c{c}",
        rows=run["rows"],
        start=run["start"],
        end=run["end"],
        penalty=fo[m]["all_features"],
        weight_updates_cold_seeds=int(len(ss)),
        of_which_tune_boundary=tune,
        of_which_kkt_reseeds=int(run.get("kkt_reseeds", 0) or 0),
        of_which_mask_addition=int(len(z["reseed"])),
        columns_newly_masked_s_lt_1e12=int(run.get("std_seed_new_masked", 0)),
        max_rel_change_s_consecutive_seeds=float(rel.max()),
        max_rel_change_l2_weight=float(run["std_max_rel_change_l2_weight"]),
        median_per_seed_max_rel_change=float(run["std_median_of_per_seed_max_rel_change"]),
        share_column_seed_pairs_rel_change_gt_10pct=float((rel[ok] > 0.1).mean()),
        share_column_seed_pairs_rel_change_gt_1=float((rel[ok] > 1.0).mean()),
        top_columns=top,
        s_quantiles_5_50_95=run["std_s_quantiles_5_50_95"],
        evidence_out_per_seed_median=run.get("std_evid_out_per_seed_median"),
        evidence_out_per_seed_max=run.get("std_evid_out_per_seed_max"),
        evidence_out_per_seed_min=run.get("std_evid_out_per_seed_min"),
        evidence_out_columns_ever=run.get("std_evid_out_columns_ever"),
        evidence_out_top_columns_seeds=run.get("std_evid_out_top_columns_seeds"),
        evidence_reseeds=run.get("std_evid_reseeds"),
        evidence_reseed_columns=run.get("std_evid_reseed_columns"),
        binom_per_seed_median=run.get("std_binom_per_seed_median"),
        binom_per_seed_max=run.get("std_binom_per_seed_max"),
        binom_columns_seeds=run.get("std_binom_columns_seeds"),
        binom_reseeds=run.get("std_binom_reseeds"),
        binom_reseed_columns=run.get("std_binom_reseed_columns"),
        ridge_max_binom_per_bar=run.get("std_ridge_max_binom_per_bar"),
        wall_ratio_on_off_same_session=round(run["wall_sec"] / off["wall_sec"], 2),
        wall_sec_on=round(run["wall_sec"], 1),
        wall_sec_on_less_blockend_diagnostic=round(run["wall_sec"] - run.get("std_blockend_sec", 0.0), 1),
        wall_sec_off_same_session=round(off["wall_sec"], 1),
        wall_sec_off_kkt_run=round(kk["wall_sec"], 1),
        cpu_sec_on=round(run["cpu_sec"], 1),
        cpu_sec_off_same_session=round(off["cpu_sec"], 1),
    )
    if m != "ridge":
        w |= dict(
            kkt_checks=run["kkt_checks"],
            kkt_all_ok=bool(z["kkt_ok"].all()),
            kkt_max_rel_after_check=run["kkt_max_rel_after_check"],
            kkt_reseeds_off=off.get("kkt_reseeds"),
            blockend_checks=run.get("std_blockend_checks"),
            blockend_max_rel_dforecast=run.get("std_blockend_max_rel_dpred"),
            blockend_median_rel_dforecast=run.get("std_blockend_median_rel_dpred"),
            blockend_max_abs_dforecast=run.get("std_blockend_max_abs_dpred"),
            blockend_max_rel_s_drift=run.get("std_blockend_max_rel_s_drift"),
            blockend_sec=run.get("std_blockend_sec"),
            repair=run["repair"],
        )
        if "std_blockend" in z.files:
            B = z["std_blockend"]
            dd = np.abs(B[:, 2] - B[:, 1]) / np.abs(B[:, 1])
            w["blockend_rel_dforecast_quantiles_50_90_99"] = [float(v) for v in np.quantile(dd, [0.5, 0.9, 0.99])]
    else:
        R = z["std_ridge"]
        w |= dict(
            ridge_blocks=int(len(R)),
            ridge_max_within_block_rel_drift=float(R[:, 1].max()),
            ridge_median_within_block_rel_drift=float(np.median(R[:, 1])),
            ridge_max_cols_left_out_beyond_mask=int(R[:, 2].max()),
            ridge_blocks_with_left_out=int((R[:, 2] > 0).sum()),
        )
    wk[m] = w
rep["walks"] = wk
OUTJ.write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
