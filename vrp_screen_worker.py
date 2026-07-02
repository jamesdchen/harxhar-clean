"""Stage-1 VRP-attribution screen cell (SGE array task): foldover-Hadamard design over the 38 raw
all_buckets base columns, each toggled as a BUNDLE = {its powers-of-5 MA ladder + availability
indicator} -- the deployed feature construction (har.py), selected per-column. Always-on core:
RV's own HAR ladder + hour dummies. Walk-forward = fit_predict_ridge (RollingLeastSquares,
tw=48000, refit=1 (EVERY BAR), ridge a=1, sqrt target). Response: filtered-VRP daily Sharpe in RAW payoff
units (full / last-1000), plus the x0.8-strike basis-robust variants.

One task = one design row (SGE_TASK_ID 1..128). Writes results/vrp_screen/cell_{id:03d}.json.
Analysis: main effects = D^T r / n (orthogonal by construction; foldover de-aliases 2-ways).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from src.data.loading import load_raw_data
from src.features.extractors.har import generate_har_features
from src.models.ridge import fit_predict_ridge

OUT = "results/vrp_screen"
N_RUNS = 128


def design_row(idx: int, k: int) -> np.ndarray:
    """Row idx of the foldover Sylvester-Hadamard H_64 design (128 x k), +-1."""
    H = np.array([[1]])
    while H.shape[0] < 64:
        H = np.block([[H, H], [H, -H]])
    D = np.vstack([H[:, 1 : k + 1], -H[:, 1 : k + 1]])
    return D[idx]


def build():
    df = load_raw_data("data/core_stats.parquet", allow_missing=True)
    df["t"] = pd.to_datetime(df["t"])
    raw: list[str] = []
    for path, pre in (
        ("data/vix_and_voldemand.parquet", "vd_"),
        ("data/spy_and_sentiment.parquet", "sn_"),
        ("data/ewstock_stats.parquet", "ew_"),
        ("data/vwstock_stats.parquet", "vw_"),
    ):
        p = pd.read_parquet(path)
        p["t"] = pd.to_datetime(p["endbartime"])
        cols = [
            c
            for c in p.columns
            if c not in ("endbartime", "t") and p[c].dtype.kind in "fi"
        ]
        p = p[["t"] + cols].rename(columns={c: f"{pre}{c}" for c in cols})
        df = df.merge(p, on="t", how="left")
        raw += [f"{pre}{c}" for c in cols]
    raw += [
        c
        for c in (
            "sumret",
            "sumabsret",
            "sumret3",
            "sumret4",
            "sumpret2",
            "sumbipow",
            "sumautocov",
            "sumvolume",
        )
        if c in df.columns
    ]
    raw = [c for c in raw if "numobs" not in c and "sumret2" not in c.lower()]

    # availability indicators from PRE-fill missingness; then fill for the ladder build
    for c in raw:
        df[f"{c}_ind"] = df[c].isna().astype(float)
        df[c] = df[c].ffill().fillna(0.0)
    df = df.dropna(subset=["RV", "vd_vix"]).reset_index(drop=True)

    df["sq"] = np.sqrt(df["RV"])
    df, cols = generate_har_features(
        df, target_col="sq", exog_cols=raw
    )  # har_ma_* + {col}_ma_* ladders
    har_cols = [c for c in cols if c.startswith("har_ma")]
    bundles = {
        c: [f"{c}_ma_{lag}" for lag in (1, 5, 25, 125, 625, 3125)] + [f"{c}_ind"]
        for c in raw
    }

    rv = df["RV"].to_numpy(float)
    tt = df["t"]
    h2 = (tt.dt.hour * 2 + tt.dt.minute // 30).to_numpy()
    hour = tt.dt.hour.to_numpy()
    date = tt.dt.date.to_numpy()
    HD = pd.get_dummies(h2, drop_first=True).to_numpy(float)
    nbar = pd.Series(rv).groupby(date).transform("size").to_numpy()
    share = pd.Series(rv).groupby(h2).transform(lambda s: s.expanding().mean().shift(1))
    tot = pd.Series(rv).expanding().mean().shift(1)
    iv_bar = (
        (pd.Series(df["vd_vix"]).shift(1).to_numpy() / 100) ** 2
        / 252
        / nbar
        * np.nan_to_num((share / tot).to_numpy(), nan=1.0)
    )
    return df, raw, bundles, har_cols, HD, rv, iv_bar, hour, date


def main() -> None:
    tid = (
        int(os.environ.get("SGE_TASK_ID", os.environ.get("SLURM_ARRAY_TASK_ID", 1))) - 1
    )
    os.makedirs(OUT, exist_ok=True)
    df, raw, bundles, har_cols, HD, rv, iv_bar, hour, date = build()
    row = design_row(tid, len(raw))

    active = [raw[i] for i in range(len(raw)) if row[i] > 0]
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
    res = {"tid": tid, "row": row.tolist(), "cols": raw, "n_active": len(active)}
    for tag, strike in (("", 1.0), ("_hc", 0.8)):
        B = np.where(sel, strike * iv_bar[mm] - rv[mm], 0.0)
        dd = pd.DataFrame({"d": date[mm], "p": B}).groupby("d")["p"].sum().to_numpy()
        res[f"sh_full{tag}"] = float(dd.mean() / dd.std())
        res[f"sh_last1000{tag}"] = float(dd[-1000:].mean() / dd[-1000:].std())
    with open(f"{OUT}/cell_{tid:03d}.json", "w") as fh:
        json.dump(res, fh)
    print(
        f"tid={tid} n_active={len(active)} sh_full={res['sh_full']:+.4f} last1000={res['sh_last1000']:+.4f}"
    )


if __name__ == "__main__":
    main()
