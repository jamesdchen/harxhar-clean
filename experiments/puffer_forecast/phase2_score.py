"""PufferLib-5.0 native MinGRU, Phase 2 (2026-10-13): score the walk-forward forecasts on the 866 trade days.

Phase 2 (cluster/puffer_forecast_phase2_h2.sh, experiments/puffer_forecast/phase2.py / phase2_select.py): for
each design (baseline, all_features), every 16:00 session 2020-01-02 .. 2024-04-30 (1,088 origins) is forecast
by a cold refit on the 2,000 sessions before it (5 seeds, each for its own early-stopped epoch count; the mean
of the 5 is the forecast); the configuration (12-cell grid) is re-tuned every 250 origins on the window ending
at the block's first origin (lowest validation MSE of the seed-averaged forecast).

Scored exactly like the other candidates (feature_mcs_report.cand_losses / close_linear_pretune.Scorer): plain
forecast f^2 x B, no recalibration, on the 866 trade days 2020-01-03 .. 2024-04-30: QLIKE against the 16:00
realized variance, fit-space MSE (f - y)^2, the sign(s) straddle's daily mid and crossed P&L (Sharpe
annualised), plus ES(1).  Comparators on the same design: the per-bar pre-tuned elastic net / ridge / lasso and
the trees (LightGBM, XGBoost, random forest) of results/feature_mcs_2026-10-08/forecasts.parquet, and the old
per-bar LSTM (master-table rows lstm_<design>, lstm_qsel_<design>).  Contrasts minGRU vs each: DM (HLN) on
the daily QLIKE and on the daily fit-space squared error, the HAC t of the daily mid and crossed P&L
differences (atm_straddle_lib.newey_west_t), positions that differ.

    python experiments/puffer_forecast/phase2_score.py RUN_DIR [OUT_NAME]
Writes results/puffer_forecast_2026-10-12/<OUT_NAME, default phase2>/{scores.csv, contrasts.csv, forecasts.csv,
selection.csv, compute.csv}; Phase 2b (phase2b.py: window mask, input clip, wider lr grid) uses OUT_NAME phase2b.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for _p in (
    REPO,
    REPO / "experiments",
    REPO / "notebooks",
    REPO / "specs",
    REPO / "writeup",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import close_linear_pretune as CLP  # noqa: E402
import feature_mcs_report as FMR  # noqa: E402
import feature_mcs_runner as FR  # noqa: E402

from src.evaluation.diebold_mariano import dm_test  # noqa: E402

OUT = REPO / "results" / "puffer_forecast_2026-10-12" / "phase2"
DESIGNS = ("baseline", "all_features")
COMPARATORS = ("enet", "ridge", "lasso", "lgbm", "xgb", "rf")


def run_gpu_models(run: Path) -> set[str]:
    """The GPU model names (nvidia-smi 'name' fields) in every env.txt / gpu_type.txt under a walk run."""
    names: set[str] = set()
    for f in list(run.rglob("env.txt")) + list(run.rglob("gpu_type.txt")):
        for line in f.read_text(errors="replace").splitlines():
            for tok in line.replace("|", ",").split(","):
                tok = tok.strip()
                if tok.startswith(("Tesla ", "NVIDIA ")):
                    names.add(tok)
    return names


def main() -> None:  # noqa: C901 -- one report
    import atm_straddle_lib as asl

    run = Path(sys.argv[1])
    gpus = run_gpu_models(run)
    if (
        len(gpus) != 1
    ):  # user, 2026-10-13: every NN number for the paper from ONE recorded GPU type
        raise SystemExit(
            f"{run}: GPU models across the run's env.txt files: {sorted(gpus) or 'none recorded'}; "
            "score only a walk run on one GPU type"
        )
    print(f"GPU model of the run: {next(iter(gpus))}")
    out = OUT.parent / sys.argv[2] if len(sys.argv) > 2 else OUT
    out.mkdir(parents=True, exist_ok=True)
    inp = json.loads((FR.INP / "inputs.json").read_text())
    W = int(inp["W"])
    fdates = pd.DatetimeIndex(pd.to_datetime(inp["dates"][W:]))
    y_adj = np.load(FR.INP / "bar_y.npy")[W:]
    S = CLP.Scorer()
    assert (pd.DatetimeIndex(S.dates) == fdates).all()
    day_row = np.load(FR.INP / "ref_armf.npz")["day_row"]
    fcs = pd.read_parquet(
        FMR.OUT / "forecasts.parquet", columns=["est", "model", "row", "pred_adj"]
    )
    mdq = pd.read_parquet(CLP.DAILY, columns=["key", "day", "plain"])
    mdq["day"] = CLP.as_day(mdq["day"])
    Bday = S.B.reindex(S.days).to_numpy(float)

    # ---------------------------------------------------------------- the minGRU walk-forward forecasts
    blocks = sorted(run.glob("block*"), key=lambda p: int(p.name[5:]))
    preds, sel_rows, fit_rows = [], [], []
    for b in blocks:
        k = int(b.name[5:])
        pr = pd.read_csv(b / "refit" / "preds.csv")
        pr = pr[pr["kind"] == "fc"]
        pr["design"] = pr["id"].str.extract(r"^refit_(baseline|all_features)_r")[0]
        preds.append(pr.assign(block=k))
        sel = json.loads((b / "selection.json").read_text())
        for m, v in sel.items():
            sel_rows.append(
                dict(
                    block=k,
                    design=m,
                    cfg=v["cfg"],
                    val_mse=v["val_mse"],
                    best_epochs=json.dumps(v["best_epoch"]),
                )
            )
        jo = pd.read_csv(b / "refit" / "jobs_out.csv")
        bad = jo[jo["status"] != "ok"]
        if len(bad):  # a non-ok refit's forecast must not enter the seed mean
            raise SystemExit(
                f"block {k}: {len(bad)} refits not ok: {bad[['id', 'status']].values.tolist()[:5]}"
            )
        if set(pr["id"]) != set(jo["id"]):
            raise SystemExit(
                f"block {k}: refits in jobs_out.csv and fc rows in preds.csv differ"
            )
        extra: dict[str, float] = {}
        if (
            "n_inputs" in jo
        ):  # Phase 2b: inputs kept by the window mask, standardised values clipped
            for m in DESIGNS:
                jm = jo[jo["id"].str.contains(f"_{m}_")]
                extra[f"n_inputs_{m}_min"] = int(jm["n_inputs"].min())
                extra[f"n_inputs_{m}_max"] = int(jm["n_inputs"].max())
                extra[f"clipped_{m}_per_refit"] = float(jm["n_clipped"].mean())

        def wall_of(
            dd: Path, j_: pd.DataFrame
        ) -> float:  # --pool runs: the run wall is in pool_summary.csv
            ps = dd / "pool_summary.csv"
            if ps.is_file():
                return float(pd.read_csv(ps)["wall"].sum())
            return float(j_.drop_duplicates("wave_wall")["wave_wall"].sum())

        fit_rows.append(
            dict(
                **extra,
                block=k,
                refits=len(jo),
                epochs=int(jo["epochs_run"].sum()),
                wave_wall=wall_of(b / "refit", jo),
                nonfinite=int(jo["nonfinite"].sum()),
                status_bad=int((jo["status"] != "ok").sum()),
            )
        )
        jt = pd.read_csv(b / "tune" / "jobs_out.csv")
        fit_rows[-1].update(
            tune_fits=len(jt),
            tune_epochs=int(jt["epochs_run"].sum()),
            tune_wall=wall_of(b / "tune", jt),
        )
    preds = pd.concat(preds)
    pd.DataFrame(sel_rows).to_csv(out / "selection.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(out / "compute.csv", index=False)
    ens = (
        preds.groupby(["design", "row"])["f"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    assert (ens["count"] == 5).all(), ens["count"].value_counts()
    ens.to_csv(out / "forecasts.csv", index=False)

    # ---------------------------------------------------------------- daily losses of every candidate
    def from_pred(pred: np.ndarray) -> dict:
        return FMR.cand_losses(S, pred, y_adj, day_row)

    def from_plain(plain: np.ndarray) -> dict:
        d = S.daily_from_plain(plain)
        f = np.sqrt(plain / Bday)
        d["mse_fit"] = (f - y_adj[day_row]) ** 2
        from close_murphy_splits_headline import elementary

        d["es1"] = elementary(d["plain"] / S.iv, S.target / S.iv, 1.0)
        return d

    cands: dict[tuple[str, str], dict] = {}
    for m in DESIGNS:
        e = ens[ens["design"] == m]
        pred = np.full(len(fdates), np.nan)
        pred[e["row"].to_numpy() - W] = e["mean"].to_numpy()
        assert np.isfinite(pred[day_row]).all(), (
            "minGRU forecast missing on a trade day"
        )
        cands[(m, "minGRU (PufferLib 5.0)")] = from_pred(pred)
        for est in COMPARATORS:
            c = fcs[(fcs["est"] == est) & (fcs["model"] == m)].sort_values("row")
            p = np.full(len(fdates), np.nan)
            p[c["row"].to_numpy()] = c["pred_adj"].to_numpy(float)
            cands[(m, est)] = from_pred(p)
        for key in (f"lstm_{m}", f"lstm_qsel_{m}"):
            mk = mdq[mdq["key"] == key].set_index("day").reindex(S.days)
            cands[(m, key)] = from_plain(mk["plain"].to_numpy(float))

    rows, con = [], []
    for (m, name), d in cands.items():
        st = S.stats(d)
        rows.append(
            dict(
                design=m,
                model=name,
                qlike=st["qlike"],
                mse_fit=float(np.mean(d["mse_fit"])),
                es1=float(np.mean(d["es1"])),
                sharpe_mid=st["sharpe_mid"],
                sharpe_crossed=st["sharpe_crossed"],
                pct_buy=st["pct_buy"],
                n_days=len(d["ql"]),
            )
        )
        if name == "minGRU (PufferLib 5.0)":
            continue
        g = cands[(m, "minGRU (PufferLib 5.0)")]
        c = S.contrast(g, d)
        dm_mse = dm_test(g["mse_fit"], d["mse_fit"])
        t_x, _ = asl.newey_west_t(g["x"] - d["x"])
        con.append(
            dict(
                design=m,
                vs=name,
                qlike_pct=c["dq_pct"],
                dm_qlike=c["dm"],
                p_qlike=c["dm_p"],
                mse_pct=float(
                    100 * (np.mean(g["mse_fit"]) / np.mean(d["mse_fit"]) - 1)
                ),
                dm_mse=float(dm_mse["dm"]),
                p_mse=float(dm_mse["p"]),
                hac_t_mid=c["hac_t_mid"],
                hac_t_crossed=float(t_x),
                hac_lag=c["hac_lag"],
                n_diff_position=c["n_diff_position"],
            )
        )
    scores = pd.DataFrame(rows)
    contrasts = pd.DataFrame(con)
    scores.to_csv(out / "scores.csv", index=False)
    contrasts.to_csv(out / "contrasts.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(pd.DataFrame(sel_rows).to_string(index=False))
    print(pd.DataFrame(fit_rows).to_string(index=False))
    print(scores.to_string(index=False))
    print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
