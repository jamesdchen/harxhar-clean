"""Coefficient-stability diagnostic -- identify V, the features whose coefficient is REGIME-VARYING (and
hence benefit from LOCAL fitting), vs the globally-stable ones (fit globally; local fitting just adds
variance). For each candidate feature, regress the Hero-A close LEFTOVER residual r = y_true - pred_adj on
that feature WITHIN vol-regime terciles, and report the spread of the marginal coefficient across terciles
normalised by its estimation SE. ratio = std(beta_b)/mean(SE_b): HIGH => regime-varying => goes in V (the
local-only block); LOW => stable => stays in the global base. Operationalises 'only locally-fit the features
that benefit from local data'. Univariate per-feature (marginal effect on the leftover) for legibility.
"""

import glob
import json

import numpy as np
import pandas as pd

from resid_amortized import CACHE_ROOT, PERIODS_PER_DAY

CELL = "xgb_all_buckets_tw1000_enetreg2_rf480_slim"
LBL = "resid_subset_heroA_d8"
TRAIN_WIN = 1000 * PERIODS_PER_DAY
CAND = [
    "har_ma_1",
    "har_ma_5",
    "har_ma_25",
    "har_ma_125",
    "har_ma_625",
    "har_ma_3125",
    "cumrv_x_close",
    "har_ma_5_x_close",
    "har_ma_25_x_close",
    "har_ma_125_x_close",
    "adj_numobs_ma_25",
    "adj_sumbipow_ma_1",
    "adj_spread_ewstock_ma_25",
    "adj_vix_ma_25",
    "adj_voldemand_all_open_only_ma_125",
    "adj_stocktwits_attention_ma_1",
    "adj_sumautocov_ma_1",
]


def main() -> None:
    d = f"{CACHE_ROOT}/{CELL}"
    feats = json.load(open(f"{d}/feats.json"))
    Xs = np.load(f"{d}/Xs.npy", mmap_mode="r")
    A = pd.concat(
        [
            pd.read_csv(f)
            for f in glob.glob(f"results/resid_ab/{CELL}/{LBL}/chunk_*.csv")
        ]
    ).sort_values("k")
    k = A["k"].to_numpy().astype(np.int64)
    r = (A["y_true"] - A["pred_adj"]).to_numpy(np.float64)
    t = k + TRAIN_WIN
    hour = np.asarray(Xs[t, feats.index("hour")], dtype=np.float64)
    close = (hour >= 16) & (hour <= 19)
    tc, rc = t[close], r[close]
    regime = np.asarray(
        Xs[tc, feats.index("har_ma_25")], dtype=np.float64
    )  # vol-state regime axis
    terc = np.digitize(regime, np.quantile(regime, [1 / 3, 2 / 3]))
    print(
        "close bars=%d  regime=har_ma_25 terciles  (resid std=%.4f)"
        % (len(tc), rc.std()),
        flush=True,
    )

    rows = []
    for c in CAND:
        if c not in feats:
            continue
        x = np.asarray(Xs[tc, feats.index(c)], dtype=np.float64)
        blist, ses = [], []
        for b in (0, 1, 2):
            m = terc == b
            xb = np.column_stack([np.ones(int(m.sum())), x[m]])
            beta = np.linalg.lstsq(xb, rc[m], rcond=None)[0]
            res = rc[m] - xb @ beta
            s2 = float(np.sum(res**2)) / max(int(m.sum()) - 2, 1)
            se = float(np.sqrt(s2 * np.linalg.inv(xb.T @ xb)[1, 1]))
            blist.append(beta[1])
            ses.append(se)
        barr = np.array(blist)
        ratio = barr.std() / (np.mean(ses) + 1e-12)
        rows.append((c, ratio, barr[0], barr[1], barr[2]))
    rows.sort(key=lambda z: -z[1])
    print(
        "%-34s %6s   beta_loVol beta_mid beta_hiVol  (ratio HIGH=>regime-varying=>V/local; LOW=>global)"
        % ("feature", "ratio")
    )
    for c, ratio, b0, b1, b2 in rows:
        tag = "  <-- V (local)" if ratio > 2.0 else ""
        print(
            "%-34s %6.2f   %+.4f   %+.4f  %+.4f%s" % (c, ratio, b0, b1, b2, tag),
            flush=True,
        )
    print("COEF_STABILITY_DONE", flush=True)


if __name__ == "__main__":
    main()
