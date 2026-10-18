"""Sharpe tests for the POOLED INTRADAY OLS HAR benchmark (key a0), 2026-10-17 follow-up.

Series: results/close_master_table/master_table_daily.parquet, key "a0", column ret_mid (the master table's own
trade of results/spxw_pnl/yhat_a0.parquet's 16:00 forecast, plain = yhat^2 x baseline, q = sign(plain - iv_var)).
a0 is the pooled OLS row of tab:rh_oos (writeup/make_rehaul_oos_table.py: "a0 for the OLS"; make_rehaul_results.EARLIER).
Tests and helpers: experiments/sharpe_tests_pooled.py (unchanged).  Output: results/sharpe_tests_pooled_ols_2026-10-17/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import sharpe_tests_pooled as st  # noqa: E402

OUT = ROOT / "results" / "sharpe_tests_pooled_ols_2026-10-17"
A0 = "a0"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    w = st.load_master()
    days = w.index
    pt, notes = st.load_pooled_trees(days)
    for k, (r, _) in pt.items():
        w[k] = r.to_numpy(float)
    full = {"Pooled ridge": "poolpt_ridge_all_features", "Pooled lasso": "poolpt_lasso_all_features",
            "Pooled elastic net": "poolpt_enet_all_features", "Pooled LightGBM": "pooltree_lgbm_all_features",
            "Pooled XGBoost": "pooltree_xgb_all_features", "Pooled random forest (INTERIM)": "pooltree_rf_all_features"}
    rng = np.random.default_rng(st.SEED + 1)
    rows = []
    for sample, sl in (("oos614", slice(st.WARMUP, None)), ("all866", slice(None))):
        R = {k: w[k].to_numpy(float)[sl] for k in [A0, st.SHORT, st.OLS, *full.values()]}
        lo = st.lo_test(R[A0])
        rows.append({"sample": sample, "test": "Lo2002 SR=0", "series": "Pooled OLS, HAR benchmark", "key": A0,
                     "comparator": "0", **lo})
        pairs = [("Pooled OLS, HAR benchmark", A0, "Short straddle", st.SHORT),
                 ("Pooled OLS, HAR benchmark", A0, "Closing OLS, HAR benchmark", st.OLS)]
        pairs += [(n, k, "Pooled OLS, HAR benchmark", A0) for n, k in full.items()]
        for name, ki, cname, kn in pairs:
            ri, rn = R[ki], R[kn]
            L = st.nw_lag(len(ri))
            d, se = st.lw_hac(ri, rn, L)
            rows.append({"sample": sample, "test": "LW2008 dSR", "series": name, "key": ki, "comparator": cname,
                         "T": len(ri), "SR_daily": ri.mean() / ri.std(ddof=1), "SR_ann": ri.mean() / ri.std(ddof=1) * st.ANN,
                         "SR_comp_ann": rn.mean() / rn.std(ddof=1) * st.ANN, "dSR_daily": d, "dSR_ann": d * st.ANN,
                         "SE": se, "SE_ann": se * st.ANN, "stat": d / se, "p_hac": st.p2(d / se),
                         "p_boot": st.lw_boot_p(ri, rn, 5, st.M_BOOT, rng, d, se), "block": 5,
                         "p_boot_b10": st.lw_boot_p(ri, rn, 10, st.M_BOOT, rng, d, se), "lag": L, "M": st.M_BOOT})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sharpe_tests_pooled_ols.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    print(df[["sample", "series", "comparator", "T", "SR_ann", "SE_ann", "stat", "p_hac", "p_boot", "p_boot_b10", "dSR_ann", "lag"]]
          .round(4).to_string(index=False))
    print("\n".join(notes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
