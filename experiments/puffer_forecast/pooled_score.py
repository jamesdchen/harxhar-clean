"""Score the pooled weight-decay walks (pooled_walk.py) on two views.

  1. ALL test bars (panel rows [246064, 300317), 2020-01-02 .. 2024-04-30, round the clock): fit-space QLIKE
     (y^2 / f^2) and MSE (f - y)^2 -- y = sqrt(rv_raw / B), the ACTUAL RV on the fit scale since 2026-10-15 (data
     audit #10; was the panel's winsorized y) -- plus the plain QLIKE of the variance forecast f^2 B against rv_raw; rows with a
     zero target (RV printed 0, 33 bars) are left out of every QLIKE.  Contrasts on the DAILY means of the per-bar
     losses (ET calendar days): DM (HLN) and the HAC t of the daily differences.
  2. The 16:00 bar on the 866 trade days with the paper's scorer (close_linear_pretune.Scorer): the variance
     forecast f^2 B of the pooled 16:00 bar (its rv_raw is the per-bar target on 99.4 % of the days, its B the same
     baseline) through Scorer.daily_from_plain -- QLIKE, fit-space MSE, Sharpe mid / crossed; DM / HAC contrasts.

Comparators: wd = 0; the bank's pooled trees (refit at EVERY bar; all_features = the bank's 20 LightGBM tree
experts -- the mean of the 19 complete ones (00-18; expert 19 lacks chunks locally) and expert 00;
baseline = pooled LightGBM / XGBoost, results/pooled_trees_2026-10-07); at 16:00 also the per-bar enet and the
per-bar MinGRU softplus walk (walk_softplus_p100).

    python experiments/puffer_forecast/pooled_score.py RUN [RUN ...]       # results/.../pooled_wd_study/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (
    REPO,
    REPO / "experiments",
    REPO / "notebooks",
    REPO / "specs",
    REPO / "writeup",
    HERE,
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RES = REPO / "results" / "puffer_forecast_2026-10-12"
WORK = RES / "_work"
OUT = RES / "pooled_wd_study"
CLEAN = REPO.parent / "harxhar-clean" / "results"
SETS = ("baseline", "all_features")
T0, T1 = 246064, 300317  # test panel rows


def rows_info() -> pd.DataFrame:
    r = pd.read_csv(WORK / "pooled_walk" / "rows.csv")
    r["t"] = pd.to_datetime(r["t_et"])
    r["day"] = r["t"].dt.normalize()
    return r


def walk_forecasts(run: Path, r0: int) -> dict:
    """design -> seed-mean fit-space forecast over the test rows (panel order), from every block's refits."""
    out = {}
    pr = pd.concat(
        [pd.read_csv(b / "refit" / "preds.csv") for b in sorted(run.glob("block*"))]
    )
    pr = pr[pr["kind"] == "fc"]
    pr["design"] = pr["id"].str.extract(r"^prefit_(baseline|all_features)_r")[0]
    for d, g in pr.groupby("design"):
        m = g.groupby("row")["f"].agg(["mean", "count"])
        assert (m["count"] == m["count"].iloc[0]).all(), "seed counts differ"
        f = np.full(T1 - T0, np.nan)
        f[m.index.to_numpy() + r0 - T0] = m["mean"].to_numpy()
        assert np.isfinite(f).all(), (
            f"{run.name} {d}: {np.isnan(f).sum()} test rows without a forecast"
        )
        out[d] = f
    return out


def trees() -> dict:
    """(design, name) -> fit-space forecast over the test rows."""
    out = {}
    lo, hi = T0, T1
    # all_features: the bank's LightGBM tree experts.  Expert 19 (the selector's most frequent pick in the test
    # period) lacks chunks 91-99 in the local copy, so the selector's series cannot be rebuilt here: the mean of the
    # 19 complete experts 00-18, and expert 00's full dump (harxhar-clean/results/spxw_pnl/yhat_tree00.parquet)
    fs = []
    for k in range(19):
        parts = []
        for c in range(80, 100):
            z = np.load(
                CLEAN
                / "unification_carc"
                / f"tree_expert_{k:02d}"
                / f"chunk_{c:03d}.npz",
                allow_pickle=True,
            )
            parts.append(pd.Series(z["yhat"], index=z["row_id"]))
        fs.append(pd.concat(parts).reindex(np.arange(lo, hi)).to_numpy())
    fa = np.array(fs)
    assert np.isfinite(fa).all()
    out[("all_features", "bank LightGBM experts 00-18, mean")] = fa.mean(axis=0)
    t0 = pd.read_parquet(CLEAN / "spxw_pnl" / "yhat_tree00.parquet", columns=["yhat"])[
        "yhat"
    ].to_numpy()
    out[("all_features", "bank LightGBM expert 00")] = t0[lo - 24_000 : hi - 24_000]
    assert np.allclose(out[("all_features", "bank LightGBM expert 00")], fa[0])
    for est in ("lgbm", "xgb"):
        t = pd.read_parquet(
            RES.parent
            / "pooled_trees_2026-10-07"
            / f"yhat_pooltree_{est}_baseline.parquet",
            columns=["yhat"],
        )["yhat"].to_numpy()
        out[
            (
                "baseline",
                f"pooled {'LightGBM' if est == 'lgbm' else 'XGBoost'} (backbone)",
            )
        ] = t[lo - 24_000 : hi - 24_000]
    return out


def bar_losses(f: np.ndarray, y: np.ndarray, B: np.ndarray, rv: np.ndarray) -> dict:
    with np.errstate(divide="ignore", invalid="ignore"):
        q = y**2 / f**2
        ql = np.where(y > 0, q - np.log(q) - 1, np.nan)
        v = f**2 * B
        qp = rv / v
        qlp = np.where(rv > 0, qp - np.log(qp) - 1, np.nan)
    return dict(ql_fit=ql, mse_fit=(f - y) ** 2, ql_plain=qlp)


def main(runs: list[Path]) -> None:
    import atm_straddle_lib as asl
    import close_linear_pretune as CLP
    import feature_mcs_report as FMR
    import feature_mcs_runner as FR

    from src.evaluation.diebold_mariano import dm_test

    OUT.mkdir(parents=True, exist_ok=True)
    rows = rows_info()
    r0 = int(rows["panel_row"].iloc[0])
    test = rows.iloc[T0 - r0 : T1 - r0].reset_index(drop=True)
    ys = {
        d: np.fromfile(
            WORK / "pooled_walk" / f"pooled_walk_{d}" / "y.f32", "<f4"
        ).astype(np.float64)[T0 - r0 : T1 - r0]
        for d in SETS
    }
    B, rv = test["baseline"].to_numpy(), test["rv_raw"].to_numpy()
    # 2026-10-15 (data audit #10): the fit-space losses of view 1 are against the ACTUAL RV on the fit scale,
    # sqrt(rv_raw / B), not the panel's winsorized y (ys, the training target, kept for the share printed below)
    y_act = np.sqrt(rv / B)
    for d in SETS:
        print(
            f"view 1, {d}: bars whose winsorized y differs from sqrt(RV / B): "
            f"{int((np.abs(ys[d] - y_act) > 1e-6 * np.maximum(1.0, y_act)).sum())} of {len(y_act)}"
        )
    cands: dict = {}
    for run in runs:
        tag = run.name.split("_")[1]
        for d, f in walk_forecasts(run, r0).items():
            cands[(d, f"MinGRU pooled {tag}")] = f
    tr = trees()
    cands.update(tr)
    # ---- view 1: all test bars
    day = test["day"].to_numpy()
    v1, c1 = [], []
    daily = {}
    for (d, name), f in cands.items():
        # L = bar_losses(f, ys[d], B, rv)  # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10)
        L = bar_losses(f, y_act, B, rv)
        v1.append(
            dict(
                design=d,
                model=name,
                n_bars=len(f),
                qlike_fit=float(np.nanmean(L["ql_fit"])),
                mse_fit=float(np.mean(L["mse_fit"])),
                qlike_plain=float(np.nanmean(L["ql_plain"])),
                min_f=float(np.min(f)),
            )
        )
        daily[(d, name)] = pd.DataFrame(L).assign(day=day).groupby("day").mean()
    for d, name in cands:
        refs = [r for r in cands if r[0] == d and r != (d, name)]
        for ref in refs:
            if not name.startswith("MinGRU"):
                continue
            if not (ref[1].endswith("wd0") or not ref[1].startswith("MinGRU")):
                continue
            a, b = daily[(d, name)], daily[ref]
            row = dict(design=d, model=name, vs=ref[1])
            for k in ("ql_fit", "mse_fit", "ql_plain"):
                x, z = a[k].to_numpy(), b[k].to_numpy()
                ok = np.isfinite(x) & np.isfinite(z)
                dm = dm_test(x[ok], z[ok])
                t, _ = asl.newey_west_t(x[ok] - z[ok])
                row.update(
                    {
                        f"{k}_pct": float(100 * (x[ok].mean() / z[ok].mean() - 1)),
                        f"{k}_dm_p": float(dm["p"]),
                        f"{k}_hac_t": float(t),
                    }
                )
            c1.append(row)
    v1, c1 = pd.DataFrame(v1), pd.DataFrame(c1)
    v1.to_csv(OUT / "allbars_scores.csv", index=False)
    c1.to_csv(OUT / "allbars_contrasts.csv", index=False)
    # ---- view 2: the 16:00 bar on the 866 trade days (the paper's scorer)
    S = CLP.Scorer()
    days = pd.DatetimeIndex(S.days).normalize()
    at16 = test[(test["t"].dt.hour == 16) & (test["t"].dt.minute == 0)].reset_index()
    pos = at16.set_index("day")["index"].reindex(days).to_numpy()
    assert np.isfinite(pos).all()
    pos = pos.astype(int)
    Bday = S.B.reindex(S.days).to_numpy(float)
    inp = json.loads((FR.INP / "inputs.json").read_text())
    W = int(inp["W"])
    y_adj = np.load(FR.INP / "bar_y.npy")[W:]
    day_row = np.load(FR.INP / "ref_armf.npz")["day_row"]
    from close_murphy_splits_headline import elementary

    def from_plain(plain: np.ndarray) -> dict:
        dd = S.daily_from_plain(plain)
        # dd["mse_fit"] = (np.sqrt(plain / Bday) - y_adj[day_row]) ** 2  # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10)
        dd["mse_fit"] = (np.sqrt(plain / Bday) - S.y_adj) ** 2  # vs sqrt(actual RV / B)
        dd["es1"] = elementary(dd["plain"] / S.iv, S.target / S.iv, 1.0)
        return dd

    c16: dict = {}
    for (d, name), f in cands.items():
        c16[(d, name)] = from_plain(f[pos] ** 2 * B[pos])
    fcs = pd.read_parquet(
        FMR.OUT / "forecasts.parquet", columns=["est", "model", "row", "pred_adj"]
    )
    n = len(inp["dates"]) - W
    for d in SETS:
        c = fcs[(fcs["est"] == "enet") & (fcs["model"] == d)].sort_values("row")
        p = np.full(n, np.nan)
        p[c["row"].to_numpy()] = c["pred_adj"].to_numpy(float)
        c16[(d, "per-bar enet")] = FMR.cand_losses(S, p, y_adj, day_row)
        e = pd.read_csv(RES / "walk_softplus_p100" / "forecasts.csv")
        e = e[e["design"] == d]
        p = np.full(n, np.nan)
        p[e["row"].to_numpy() - W] = e["mean"].to_numpy()
        c16[(d, "per-bar MinGRU softplus walk")] = FMR.cand_losses(S, p, y_adj, day_row)
    v2, cn2 = [], []
    for (d, name), dd in c16.items():
        st = S.stats(dd)
        v2.append(
            dict(
                design=d,
                model=name,
                qlike=st["qlike"],
                mse_fit=float(np.mean(dd["mse_fit"])),
                es1=float(np.mean(dd["es1"])),
                sharpe_mid=st["sharpe_mid"],
                sharpe_crossed=st["sharpe_crossed"],
                pct_buy=st["pct_buy"],
                n_days=len(dd["ql"]),
            )
        )
    for (d, name), g in c16.items():
        if not name.startswith("MinGRU pooled"):
            continue
        for (d2, ref), h in c16.items():
            if (
                d2 != d
                or ref == name
                or (ref.startswith("MinGRU pooled") and not ref.endswith("wd0"))
            ):
                continue
            c = S.contrast(g, h)
            dm_mse = dm_test(g["mse_fit"], h["mse_fit"])
            t_x, _ = asl.newey_west_t(g["x"] - h["x"])
            cn2.append(
                dict(
                    design=d,
                    model=name,
                    vs=ref,
                    qlike_pct=c["dq_pct"],
                    p_qlike=c["dm_p"],
                    mse_pct=float(
                        100 * (np.mean(g["mse_fit"]) / np.mean(h["mse_fit"]) - 1)
                    ),
                    p_mse=float(dm_mse["p"]),
                    hac_t_mid=c["hac_t_mid"],
                    hac_t_crossed=float(t_x),
                )
            )
    v2, cn2 = pd.DataFrame(v2), pd.DataFrame(cn2)
    v2.to_csv(OUT / "at1600_scores.csv", index=False)
    cn2.to_csv(OUT / "at1600_contrasts.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    print(v1.round(4).to_string(index=False))
    print(c1.round(3).to_string(index=False))
    print(v2.round(4).to_string(index=False))
    print(cn2.round(3).to_string(index=False))


def extras(runs: list[Path]) -> None:
    """B's picks per block, best epochs against wd (every tune job of the A walks), weight norms at the best epoch."""
    import reference as RF

    sel, be, nm = [], [], []
    for run in runs:
        tag = run.name.split("_")[1]
        for b in sorted(run.glob("block*")):
            k = int(b.name[5:])
            sj = json.loads((b / "selection.json").read_text())
            for d, v in sj.items():
                sel.append(
                    dict(
                        walk=tag,
                        block=k,
                        design=d,
                        cfg=v["cfg"],
                        wd=v["wd"],
                        val_mse=v["val_mse"],
                        best_epochs=json.dumps(v["best_epoch"]),
                    )
                )
            jo = pd.read_csv(b / "tune" / "jobs_out.csv")
            jo["design"] = jo["id"].str.extract(r"^ptune_(baseline|all_features)_")[0]
            be.append(
                jo.assign(walk=tag, block=k)[
                    [
                        "walk",
                        "block",
                        "design",
                        "wd",
                        "best_epoch",
                        "epochs_run",
                        "best_val_mse",
                    ]
                ]
            )
            for f in sorted((b / "refit" / "params").glob("*.f32")):
                d = "all_features" if "_all_features_" in f.stem else "baseline"
                H = int(sj[d]["cfg"].split("_H")[1].split("_")[0])
                keep = np.fromfile(f.with_suffix(".keep.i32"), "<i4")
                w = RF.unpack(RF.read_f32(f), len(keep) + 1, H, 1)
                dec = float(
                    np.sqrt(
                        np.sum(w["We"][:, :-1] ** 2)
                        + np.sum(w["Wd"] ** 2)
                        + np.sum(w["Wl"][0] ** 2)
                    )
                )
                nm.append(
                    dict(
                        walk=tag,
                        block=k,
                        design=d,
                        cfg=sj[d]["cfg"],
                        wd=sj[d]["wd"],
                        fro_decayed=dec,
                    )
                )
    sel, be, nm = pd.DataFrame(sel), pd.concat(be), pd.DataFrame(nm)
    sel.to_csv(OUT / "selection.csv", index=False)
    be.to_csv(OUT / "tune_best_epochs.csv", index=False)
    nm.to_csv(OUT / "weight_norms_best_epoch.csv", index=False)
    pd.set_option("display.width", 250)
    print(sel[sel["walk"] == "wdgrid"].to_string(index=False))
    a = be[be["walk"] != "wdgrid"]
    print(
        a.groupby(["design", "wd"])[["best_epoch", "epochs_run"]]
        .agg(["median", "mean"])
        .round(2)
        .to_string()
    )
    print(nm.groupby(["design", "walk"])["fro_decayed"].median().round(3).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "extras":
        extras([Path(p) for p in sys.argv[2:]])
    else:
        main([Path(p) for p in sys.argv[1:]])
