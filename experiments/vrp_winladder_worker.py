"""VRP window ladder (sec-7 ladder + 500d) x feature sets -- the P&L window-response curve the two-point
stage-2 comparison cannot see, incl. the window x feature interaction (volume/vvix decay at long tw).

Cells: windows {24,48,72,96,120,144,168}k bars x fsets {core, activity, best6} = 21 tasks.
Responses: sh_last1000(+hc), sh_eras, and sh_common(+hc) = Sharpe on the COMMON fixed eval period
(session bars at index >= 168000, the sec-7 convention) so all windows are scored on identical days.
Config locked by the design factorial: sqrt target, alpha=1, refit=1 every-bar, q=0.5.
Writes results/vrp_winladder/cell_{tid:02d}.json.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)

import itertools
import json
import os

import numpy as np
import pandas as pd

from src.models.ridge import fit_predict_ridge
from vrp_screen_worker import build

OUT = "results/vrp_winladder"
WINDOWS = [24_000, 48_000, 72_000, 96_000, 120_000, 144_000, 168_000]
FSETS = {
    "core": [],
    "activity": ["sumabsret", "sumvolume"],
    "best6": ["sumabsret", "sumvolume", "vd_voldemand_all_open_only", "vd_vvix",
              "sn_stocktwits_sentcount", "sn_stocktwits_attention"],
}
GRID = list(itertools.product(WINDOWS, FSETS))


def main() -> None:
    tid = int(os.environ.get("SGE_TASK_ID", os.environ.get("SLURM_ARRAY_TASK_ID", 1)) or 1)
    if "SGE_TASK_ID" in os.environ and "SLURM_ARRAY_TASK_ID" not in os.environ:
        tid -= 1
    tw, fname = GRID[tid]
    os.makedirs(OUT, exist_ok=True)

    df, raw, bundles, har_cols, HD, rv, iv_bar, hour, date = build()
    feat_cols = har_cols + [c for b in FSETS[fname] for c in bundles[b]]
    X = np.column_stack([df[feat_cols].fillna(0).to_numpy(float), HD])
    y = df["sq"].to_numpy(float)
    p = fit_predict_ridge(X, y, train_win_periods=tw, hyperparams=dict(alpha=1.0, _refit_frequency=1))
    pred = np.full(len(df), np.nan)
    pred[len(df) - len(p):] = p

    prem = iv_bar - np.where(np.isfinite(pred), pred, np.nan) ** 2
    vrp = iv_bar - rv
    sess = (hour >= 9) & (hour <= 15)
    common = np.arange(len(df)) >= 168_000  # sec-7 fixed-eval convention: identical days for all windows
    mm = np.isfinite(prem) & np.isfinite(vrp) & sess
    q = pd.Series(prem[mm]).rolling(40 * 252).median().shift(1).to_numpy()
    sel = prem[mm] > np.nan_to_num(q, nan=np.inf)
    res = {"tid": tid, "window": tw, "fset": fname}
    for tag, strike in (("", 1.0), ("_hc", 0.8)):
        B = np.where(sel, strike * iv_bar[mm] - rv[mm], 0.0)
        dd = pd.DataFrame({"d": date[mm], "p": B}).groupby("d")["p"].sum().to_numpy()
        n3 = len(dd) // 3
        res[f"sh_last1000{tag}"] = float(dd[-1000:].mean() / dd[-1000:].std())
        res[f"sh_eras{tag}"] = [float(dd[i*n3:(i+1)*n3].mean() / max(dd[i*n3:(i+1)*n3].std(), 1e-12)) for i in range(3)]
        Bc = np.where(sel & common[mm], strike * iv_bar[mm] - rv[mm], 0.0)
        dc = pd.DataFrame({"d": date[mm], "p": Bc}).groupby("d")["p"].sum().to_numpy()
        dc = dc[-(len(df) - 168_000) // 40:]  # trim to the common tail days
        res[f"sh_common{tag}"] = float(dc.mean() / dc.std()) if dc.std() > 0 else float("nan")
    with open(f"{OUT}/cell_{tid:02d}.json", "w") as fh:
        json.dump(res, fh)
    print(f"tid={tid} tw={tw} fset={fname} last1000={res['sh_last1000']:+.4f} common={res['sh_common']:+.4f}")


if __name__ == "__main__":
    main()
