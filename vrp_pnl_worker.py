"""P(pass) rescoring inputs: rerun the 16 boundary-defining factorial cells and DUMP the daily payoff
components (per-day sums over SELECTED session bars of iv_bar and rv), so any strike k gives
pnl_day = k*sel_iv - sel_rv post-hoc -- the eval simulators then score P(pass) locally under any rule
set / sizing without another cluster pass. Config: the locked design (sqrt, alpha=1, tw=48000,
refit=1 every-bar, q=0.5). Writes results/vrp_pnl/cell_{tid:02d}.{json,npz}."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from src.models.ridge import fit_predict_ridge
from vrp_screen_worker import build

OUT = "results/vrp_pnl"
SURV = ["sumabsret", "sumvolume", "vd_voldemand_spx_open_only", "vd_voldemand_all_open_only",
        "vd_vvix", "sumbipow", "sn_stocktwits_sentcount", "sn_stocktwits_attention"]
CELLS: list[tuple[str, tuple[int, ...]]] = (
    [("core", ())]
    + [(f"solo_{c}", (i,)) for i, c in enumerate(SURV)]
    + [("activity", (0, 1)), ("activity_vvix", (0, 1, 4)),
       ("best6_novvix", (0, 1, 3, 6, 7)), ("best6_nosent", (0, 1, 3, 4)),
       ("best6", (0, 1, 3, 4, 6, 7)), ("all8", tuple(range(8)))]
)


def main() -> None:
    tid = int(os.environ.get("SGE_TASK_ID", os.environ.get("SLURM_ARRAY_TASK_ID", 1)) or 1)
    if "SGE_TASK_ID" in os.environ and "SLURM_ARRAY_TASK_ID" not in os.environ:
        tid -= 1
    name, bits = CELLS[tid]
    os.makedirs(OUT, exist_ok=True)

    df, raw, bundles, har_cols, HD, rv, iv_bar, hour, date = build()
    feat_cols = har_cols + [c for i in bits for c in bundles[SURV[i]]]
    X = np.column_stack([df[feat_cols].fillna(0).to_numpy(float), HD])
    y = df["sq"].to_numpy(float)
    p = fit_predict_ridge(X, y, train_win_periods=48_000, hyperparams=dict(alpha=1.0, _refit_frequency=1))
    pred = np.full(len(df), np.nan)
    pred[len(df) - len(p):] = p

    prem = iv_bar - np.where(np.isfinite(pred), pred, np.nan) ** 2
    vrp = iv_bar - rv
    sess = (hour >= 9) & (hour <= 15)
    mm = np.isfinite(prem) & np.isfinite(vrp) & sess
    q = pd.Series(prem[mm]).rolling(40 * 252).median().shift(1).to_numpy()
    sel = prem[mm] > np.nan_to_num(q, nan=np.inf)
    dsel = pd.DataFrame({"d": date[mm], "iv": np.where(sel, iv_bar[mm], 0.0), "rv": np.where(sel, rv[mm], 0.0)})
    g = dsel.groupby("d").sum()
    dd = (g["iv"] - g["rv"]).to_numpy()
    np.savez_compressed(f"{OUT}/cell_{tid:02d}.npz",
                        days=np.array([str(d) for d in g.index]),
                        sel_iv=g["iv"].to_numpy(), sel_rv=g["rv"].to_numpy())
    res = {"tid": tid, "name": name, "bits": list(bits),
           "sh_last1000": float(dd[-1000:].mean() / dd[-1000:].std()),
           "sh_full": float(dd.mean() / dd.std())}
    with open(f"{OUT}/cell_{tid:02d}.json", "w") as fh:
        json.dump(res, fh)
    print(f"tid={tid} {name} last1000={res['sh_last1000']:+.4f}")


if __name__ == "__main__":
    main()
