"""Stage-2 VRP attribution: FULL 2^k factorial over the stage-1 surviving bundles (spec-driven).

Reads results/vrp_stage2_spec.json: {"survivors": [col, ...], "out": "results/vrp_stage2"}.
One array task = one bitmask cell: SLURM_ARRAY_TASK_ID (0-based) in [0, 2^k). Active bundles = set bits.
Reuses the stage-1 build (always-on RV-HAR ladder + hour dummies; bundle = MA ladder + availability
indicator; research walk-forward tw=48000/refit=480/alpha=1/sqrt target). Adds era-resolved responses so
interactions can be read on the recent regime. Exact main effects + ALL interactions, no aliasing.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (repo root + siblings on sys.path)

import json
import os

import numpy as np
import pandas as pd

from src.models.ridge import fit_predict_ridge
from vrp_screen_worker import build

SPEC = "results/vrp_stage2_spec.json"


def main() -> None:
    spec = json.load(open(SPEC))
    surv: list[str] = spec["survivors"]
    out = spec.get("out", "results/vrp_stage2")
    os.makedirs(out, exist_ok=True)
    tid = int(
        os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("SGE_TASK_ID", 1)) or 0
    )
    if "SGE_TASK_ID" in os.environ and "SLURM_ARRAY_TASK_ID" not in os.environ:
        tid -= 1  # SGE is 1-based
    k = len(surv)
    assert 0 <= tid < 2**k, f"tid {tid} out of range for k={k}"
    bits = [(tid >> i) & 1 for i in range(k)]

    df, raw, bundles, har_cols, HD, rv, iv_bar, hour, date = build()
    active = [surv[i] for i in range(k) if bits[i]]
    feat_cols = har_cols + [c for b in active for c in bundles[b]]
    X = np.column_stack([df[feat_cols].fillna(0).to_numpy(float), HD])
    y = df["sq"].to_numpy(float)
    p = fit_predict_ridge(
        X, y, train_win_periods=48_000, hyperparams=dict(alpha=1.0, _refit_frequency=1)
    )
    pred = np.full(len(df), np.nan)
    pred[len(df) - len(p) :] = p

    prem = iv_bar - np.where(np.isfinite(pred), pred, np.nan) ** 2
    vrp = iv_bar - rv
    sess = (hour >= 9) & (hour <= 15)
    mm = np.isfinite(prem) & np.isfinite(vrp) & sess
    q = pd.Series(prem[mm]).rolling(40 * 252).median().shift(1).to_numpy()
    sel = prem[mm] > np.nan_to_num(q, nan=np.inf)
    res = {"tid": tid, "bits": bits, "survivors": surv, "active": active}
    for tag, strike in (("", 1.0), ("_hc", 0.8)):
        B = np.where(sel, strike * iv_bar[mm] - rv[mm], 0.0)
        dd = pd.DataFrame({"d": date[mm], "p": B}).groupby("d")["p"].sum().to_numpy()
        n3 = len(dd) // 3
        res[f"sh_full{tag}"] = float(dd.mean() / dd.std())
        res[f"sh_last1000{tag}"] = float(dd[-1000:].mean() / dd[-1000:].std())
        res[f"sh_eras{tag}"] = [
            float(
                dd[i * n3 : (i + 1) * n3].mean()
                / max(dd[i * n3 : (i + 1) * n3].std(), 1e-12)
            )
            for i in range(3)
        ]
    with open(f"{out}/cell_{tid:04d}.json", "w") as fh:
        json.dump(res, fh)
    print(
        f"tid={tid} active={len(active)} sh_full={res['sh_full']:+.4f} last1000={res['sh_last1000']:+.4f}"
    )


if __name__ == "__main__":
    main()
