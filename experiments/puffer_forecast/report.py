"""PufferLib-5.0 native MinGRU forecaster (2026-10-12, Phase 1): accuracy, training diagnostics, speed,
determinism and the walk-forward projection, from one run of cluster/puffer_forecast_phase1_carc.sbatch.

Accuracy rows (the window of record, rows w0 .. w1 - 1 = the 2,000 16:00 sessions ending 2019-12-31):
  val   the 125-session validation tail of the window (2019-07 .. 2019-12-31), forecast by the
        fit-block model at its early-stopped epoch (fit block = the window's first 1,850 sessions,
        25-session embargo): out of sample, but up to 150 sessions stale
  fc    the next 10 sessions (2020-01-02 .. 2020-01-15), forecast by the refit on the whole window
        (each seed for its own early-stopped epoch count)
minGRU forecast = the mean of the 5 seeds of the configuration with the lowest validation MSE of the
seed-averaged validation forecast (the LSTM spec's selection rule).  Comparators: the per-bar
pre-tuned elastic net / ridge / lasso (refit every session) and the tree arms (LightGBM, XGBoost,
random forest; trade days only) on the same rows, from results/feature_mcs_2026-10-08/forecasts.parquet.
Losses: fit-space MSE (f - y)^2 against the design's target y; QLIKE of the plain forecast f^2 B against
the 16:00 realized variance (targets.parquet true_raw / baseline, the scorer's convention, no floor).

    python experiments/puffer_forecast/report.py RUN_DIR
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "puffer_forecast_2026-10-12"
WORK = OUT / "_work"
FCS = REPO / "results" / "feature_mcs_2026-10-08" / "forecasts.parquet"
TARGETS = (
    REPO / "results" / "close_studies_2026-10-03" / "forecasts" / "targets.parquet"
)
DESIGNS = ("baseline", "all_features")
ID_RE = re.compile(
    r"^(tune|refit)_(baseline|all_features)_L(\d+)_H(\d+)_lr([0-9.]+)_s(\d+)(?:_(qlike|huber))?$"
)


def parse_ids(df: pd.DataFrame) -> pd.DataFrame:
    m = df["id"].str.extract(ID_RE)
    m.columns = ["kind", "design", "cL", "cH", "clr", "cseed", "lossv"]
    out = pd.concat([df, m], axis=1)
    out["cfg"] = (
        "L"
        + out["cL"].astype(str)
        + "_H"
        + out["cH"].astype(str)
        + "_lr"
        + out["clr"].astype(str)
    )
    return out


def qlike(rv: np.ndarray, plain: np.ndarray) -> np.ndarray:
    r = rv / plain
    return r - np.log(r) - 1.0


def main() -> None:  # noqa: C901 -- one report
    run = Path(sys.argv[1])
    ph = run / "phase1"
    prep = json.loads((WORK / "prep.json").read_text())
    dates = pd.read_csv(WORK / "dates.csv")
    date_of = pd.Series(pd.to_datetime(dates["date"]).to_numpy(), index=dates["row"])
    y_all = {
        m: np.fromfile(WORK / "data" / m / "y.f32", "<f4").astype(np.float64)
        for m in DESIGNS
    }
    jo = parse_ids(pd.read_csv(ph / "jobs_out.csv"))
    ep = pd.read_csv(ph / "epochs.csv")
    st = pd.read_csv(ph / "steps.csv")
    pr = pd.read_csv(ph / "preds.csv")
    tg = pd.read_parquet(TARGETS)
    tg["date"] = pd.to_datetime(tg["date"])
    tg = tg.set_index("date")
    env = (run / "env.txt").read_text()

    # ------------------------------------------------------------------ selection and ensembles
    mse_tune = jo[
        (jo["kind"] == "tune") & jo["lossv"].isna() & ~jo["id"].str.startswith("dup_")
    ]
    sel_rows, ens = [], {}
    for m in DESIGNS:
        yv = None
        best = None
        for cfg, g in mse_tune[mse_tune["design"] == m].groupby("cfg"):
            pv = pr[pr["id"].isin(g["id"]) & (pr["kind"] == "val")].pivot(
                index="row", columns="id", values="f"
            )
            rows = pv.index.to_numpy()
            yv = y_all[m][rows]
            f_avg = pv.mean(axis=1).to_numpy()
            mse_avg = float(np.mean((f_avg - yv) ** 2))
            mse_seeds = [
                float(np.mean((pv[c].to_numpy() - yv) ** 2)) for c in pv.columns
            ]
            r = dict(
                design=m,
                cfg=cfg,
                val_mse_seedavg=mse_avg,
                val_mse_seed_mean=float(np.mean(mse_seeds)),
                val_mse_seed_sd=float(np.std(mse_seeds)),
                best_epoch_mean=float(g["best_epoch"].mean()),
                best_epoch_min=int(g["best_epoch"].min()),
                best_epoch_max=int(g["best_epoch"].max()),
                epochs_run_mean=float(g["epochs_run"].mean()),
                t_train_mean=float(g["t_train"].mean()),
                n_params=int(g["n_params"].iloc[0]),
            )
            sel_rows.append(r)
            if best is None or mse_avg < best[1]:
                best = (cfg, mse_avg, pv)
        assert best is not None
        cfg, _, pv = best
        rid = [i.replace("tune_", "refit_", 1) for i in pv.columns]
        pf = pr[pr["id"].isin(rid) & (pr["kind"] == "fc")].pivot(
            index="row", columns="id", values="f"
        )
        ens[m] = dict(
            cfg=cfg, val=pv.mean(axis=1), fc=pf.mean(axis=1), val_seeds=pv, fc_seeds=pf
        )
    sel = pd.DataFrame(sel_rows).sort_values(["design", "val_mse_seedavg"])
    sel.to_csv(OUT / "configs.csv", index=False)

    # ------------------------------------------------------------------ comparators on the same rows
    fcs = pd.read_parquet(FCS, columns=["est", "model", "date", "pred_adj"])
    fcs = fcs[fcs["model"].isin(DESIGNS)]
    fcs["date"] = pd.to_datetime(fcs["date"])
    acc_rows, fc_table = [], []
    for m in DESIGNS:
        for span in ("val", "fc"):
            s = ens[m][span]
            rows = s.index.to_numpy()
            d = pd.DatetimeIndex(date_of.loc[rows].to_numpy())
            yv = y_all[m][rows]
            series = {f"minGRU ({ens[m]['cfg']})": s.to_numpy()}
            for est in ("enet", "ridge", "lasso", "lgbm", "xgb", "rf"):
                c = fcs[(fcs["est"] == est) & (fcs["model"] == m)].set_index("date")[
                    "pred_adj"
                ]
                series[est] = c.reindex(d).to_numpy(float)
            B = tg["baseline"].reindex(d).to_numpy(float)
            rv = tg["true_raw"].reindex(d).to_numpy(float)
            assert np.allclose(
                tg["true_adj"].reindex(d).to_numpy(float), yv, rtol=1e-6, atol=1e-6
            )
            for name, f in series.items():
                ok = np.isfinite(f)
                if ok.sum() == 0:
                    continue
                plain = f[ok] ** 2 * B[ok]
                acc_rows.append(
                    dict(
                        design=m,
                        span=span,
                        model=name,
                        n=int(ok.sum()),
                        mse_fit=float(np.mean((f[ok] - yv[ok]) ** 2)),
                        qlike_plain=float(np.mean(qlike(rv[ok], plain))),
                        mean_f=float(np.mean(f[ok])),
                        corr_y=float(np.corrcoef(f[ok], yv[ok])[0, 1])
                        if ok.sum() > 2
                        else np.nan,
                    )
                )
                if span == "fc":
                    for dd, ff in zip(d[ok], f[ok]):
                        fc_table.append(
                            dict(
                                design=m, model=name, date=dd.strftime("%Y-%m-%d"), f=ff
                            )
                        )
            if span == "fc":
                for dd, yy in zip(d, yv):
                    fc_table.append(
                        dict(
                            design=m,
                            model="target y",
                            date=dd.strftime("%Y-%m-%d"),
                            f=yy,
                        )
                    )
    acc = pd.DataFrame(acc_rows)
    acc.to_csv(OUT / "accuracy.csv", index=False)
    fct = (
        pd.DataFrame(fc_table)
        .pivot_table(index=["design", "date"], columns="model", values="f")
        .reset_index()
    )
    fct.to_csv(OUT / "forecasts_next10.csv", index=False)
    # matched-rows comparison: the trees exist only on trade days -> the 9 trade days of the 10
    # (all models on the same rows)

    # ------------------------------------------------------------------ training diagnostics
    # Muon clips when its clip coefficient max_grad_norm / (norm + 1e-6) < 1; max_grad_norm = 1.0 in every job
    st["clipped"] = st["grad_norm"] + 1e-6 > 1.0
    diag = (
        st.groupby("id")
        .agg(
            steps=("loss", "size"),
            gn_mean=("grad_norm", "mean"),
            gn_p99=("grad_norm", lambda x: x.quantile(0.99)),
            gn_max=("grad_norm", "max"),
            clip_frac=("clipped", "mean"),
            capped=("n_capped", "sum"),
            nonfinite=("n_nonfinite", "sum"),
        )
        .reset_index()
    )
    diag = parse_ids(diag)
    diag = diag.merge(jo[["id", "N", "epochs_run", "status"]], on="id")
    diag["cap_frac"] = diag["capped"] / (diag["N"] * diag["epochs_run"])
    diag.to_csv(OUT / "train_diagnostics.csv", index=False)
    per_epoch = (
        st.groupby(["id", "epoch"])
        .agg(
            gn=("grad_norm", "mean"),
            clip=("clipped", "mean"),
            cap=("n_capped", "sum"),
            nf=("n_nonfinite", "sum"),
            loss=("loss", "mean"),
        )
        .reset_index()
    )
    per_epoch = per_epoch.merge(ep, on=["id", "epoch"], how="left")
    per_epoch.to_csv(OUT / "train_epochs.csv", index=False)

    # ------------------------------------------------------------------ determinism
    det = []

    def same(a_dir: Path, b_dir: Path, ids: list[tuple[str, str]], tag: str) -> None:
        A = pd.read_csv(a_dir / "preds.csv")
        Bp = pd.read_csv(b_dir / "preds.csv")
        As = pd.read_csv(a_dir / "steps.csv")
        Bs = pd.read_csv(b_dir / "steps.csv")
        for i, j in ids:
            pa = A[A["id"] == i]["f"].to_numpy()
            pb = Bp[Bp["id"] == j]["f"].to_numpy()
            sa = As[As["id"] == i][["loss", "grad_norm"]].to_numpy()
            sb = Bs[Bs["id"] == j][["loss", "grad_norm"]].to_numpy()
            det.append(
                dict(
                    test=tag,
                    a=i,
                    b=j,
                    preds_identical=bool(len(pa) == len(pb) and np.array_equal(pa, pb)),
                    steps_identical=bool(
                        sa.shape == sb.shape and np.array_equal(sa, sb)
                    ),
                    n_steps=len(sa),
                )
            )

    for m in DESIGNS:
        i = f"tune_{m}_L20_H64_lr0.01_s{prep['seeds'][0]}"
        same(ph, ph, [(i, "dup_" + i)], "same wave, duplicate model")
    sc_ids = [
        ln.split()[0][3:]
        for ln in (WORK / "jobs_scaling.txt").read_text().splitlines()
        if ln.strip()
    ]
    for tag in ("scale_k1", "scale_k8", "scale_k24"):
        if (run / tag / "preds.csv").is_file():
            same(
                ph,
                run / tag,
                [(i, i) for i in sc_ids],
                f"phase1 (48 concurrent) vs {tag}",
            )
    detdf = pd.DataFrame(det)
    detdf.to_csv(OUT / "determinism.csv", index=False)

    # ------------------------------------------------------------------ speed
    sp_rows = []
    for tag in ("phase1", "scale_k1", "scale_k8", "scale_k24"):
        f = run / tag / "jobs_out.csv"
        if not f.is_file():
            continue
        j = pd.read_csv(f)
        waves = j.groupby(["wave_wall", "wave_size"]).size().reset_index()
        for mode in sorted(j["mode"].unique()):
            jj = j[j["mode"] == mode]
            sp_rows.append(
                dict(
                    run=tag,
                    mode=mode,
                    models=len(jj),
                    epochs=int(jj["epochs_run"].sum()),
                    steps=int((jj["epochs_run"] * jj["S"]).sum()),
                    samples=int((jj["epochs_run"] * jj["N"]).sum()),
                    t_train_mean=float(jj["t_train"].mean()),
                    t_setup_mean=float(jj["t_setup"].mean()),
                    t_capture_mean=float(jj["t_capture"].mean()),
                )
            )
        sp_rows.append(
            dict(
                run=tag,
                mode="ALL (wall)",
                models=len(j),
                epochs=int(j["epochs_run"].sum()),
                steps=int((j["epochs_run"] * j["S"]).sum()),
                samples=int((j["epochs_run"] * j["N"]).sum()),
                wall=float(waves["wave_wall"].sum()),
                n_waves=len(waves),
            )
        )
    speed = pd.DataFrame(sp_rows)
    for c in ("epochs", "steps", "samples"):
        speed[f"{c}_per_s"] = speed[c] / speed.get("wall", np.nan)
    speed.to_csv(OUT / "speed.csv", index=False)

    # ------------------------------------------------------------------ warm starts
    warm = []
    for m in DESIGNS:
        ids = [f"shift10_cold_{m}", f"shift10_warm10_{m}", f"shift10_warm25_{m}"]
        g = pr[pr["id"].isin(ids)].pivot(index="row", columns="id", values="f")
        if g.empty:
            continue
        rows = g.index.to_numpy()
        yv = y_all[m][rows]
        for i in ids:
            if i in g:
                jrow = jo[jo["id"] == i].iloc[0]
                warm.append(
                    dict(
                        design=m,
                        job=i,
                        epochs=int(jrow["epochs_run"]),
                        t_train=float(jrow["t_train"]),
                        mse_fit=float(np.mean((g[i].to_numpy() - yv) ** 2)),
                        max_abs_diff_vs_cold=float(
                            np.max(np.abs(g[i].to_numpy() - g[ids[0]].to_numpy()))
                        ),
                    )
                )
    warmdf = pd.DataFrame(warm)
    warmdf.to_csv(OUT / "warm_start.csv", index=False)

    # ------------------------------------------------------------------ loss variants (same configuration L20_H16_lr0.01)
    lv_rows = []
    for m in DESIGNS:
        for lossv in ("mse", "qlike", "huber"):
            sfx = "" if lossv == "mse" else f"_{lossv}"
            tids = [f"tune_{m}_L20_H16_lr0.01_s{s_}{sfx}" for s_ in prep["seeds"]]
            rids = [t.replace("tune_", "refit_", 1) for t in tids]
            for span, ids, kind in (("val", tids, "val"), ("fc", rids, "fc")):
                g = pr[pr["id"].isin(ids) & (pr["kind"] == kind)].pivot(
                    index="row", columns="id", values="f"
                )
                rows = g.index.to_numpy()
                d = pd.DatetimeIndex(date_of.loc[rows].to_numpy())
                f = g.mean(axis=1).to_numpy()
                yv = y_all[m][rows]
                plain = f**2 * tg["baseline"].reindex(d).to_numpy(float)
                lv_rows.append(
                    dict(
                        design=m,
                        loss=lossv,
                        span=span,
                        n=len(rows),
                        mse_fit=float(np.mean((f - yv) ** 2)),
                        qlike_plain=float(
                            np.mean(
                                qlike(tg["true_raw"].reindex(d).to_numpy(float), plain)
                            )
                        ),
                        best_epoch_mean=float(
                            jo[jo["id"].isin(tids)]["best_epoch"].mean()
                        ),
                    )
                )
    lvdf = pd.DataFrame(lv_rows)
    lvdf.to_csv(OUT / "loss_variants.csv", index=False)

    # ------------------------------------------------------------------ walk-forward projection (one GPU of this class)
    allw = speed[speed["mode"] == "ALL (wall)"].set_index("run")
    thr48 = float(allw.loc["phase1", "epochs_per_s"])
    thr1 = (
        float(allw.loc["scale_k1", "epochs_per_s"])
        if "scale_k1" in allw.index
        else np.nan
    )
    e_tune = float(jo[jo["mode"] == "tune"]["epochs_run"].mean())
    e_refit = float(
        jo[(jo["mode"] == "refit") & jo["id"].str.startswith("refit_")][
            "epochs_run"
        ].mean()
    )
    n_rows = 1088  # 16:00 sessions 2020-01-02 .. 2024-04-30 (covers the 866 trade days)
    n_cfg, n_seed = len(prep["grid"]), len(prep["seeds"])
    proj = []
    for k in (1, 10, 25):
        n_refit = int(np.ceil(n_rows / k))
        for scheme, n_tune_pts in (
            ("tune every 250 sessions (LSTM spec)", int(np.ceil(n_rows / 250))),
            ("full grid re-tuned at every refit", n_refit),
        ):
            ep_ = n_tune_pts * n_cfg * n_seed * e_tune + n_refit * n_seed * e_refit
            proj.append(
                dict(
                    refit_every=k,
                    tuning=scheme,
                    refits=n_refit * n_seed,
                    tune_fits=n_tune_pts * n_cfg * n_seed,
                    model_epochs=ep_,
                    minutes_per_design_concurrent=ep_ / thr48 / 60,
                    minutes_per_design_sequential=ep_ / thr1 / 60,
                    hours_96_designs_concurrent=96 * ep_ / thr48 / 3600,
                )
            )
    projdf = pd.DataFrame(proj)
    projdf.to_csv(OUT / "projection.csv", index=False)

    # ------------------------------------------------------------------ write
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(env)
    print(sel.to_string(index=False))
    print(acc.to_string(index=False))
    print(fct.to_string(index=False))
    print(
        diag.groupby(["design", "lossv"], dropna=False)[
            ["gn_mean", "gn_max", "clip_frac", "cap_frac", "nonfinite"]
        ]
        .agg(["mean", "max"])
        .to_string()
    )
    print(
        detdf.groupby("test")[["preds_identical", "steps_identical"]].mean().to_string()
    )
    print(speed.to_string(index=False))
    print(warmdf.to_string(index=False))
    print(lvdf.to_string(index=False))
    print(
        f"throughput: {thr48:.0f} model-epochs/s (48 concurrent), {thr1:.0f} (1 at a time); mean epochs tune {e_tune:.1f}, refit {e_refit:.1f}"
    )
    print(projdf.to_string(index=False))


if __name__ == "__main__":
    main()
