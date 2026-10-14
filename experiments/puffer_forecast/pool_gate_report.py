"""Trainer pool gate / timing and the packing experiment (2026-10-13), from one run of
cluster/puffer_forecast_pack_gate_carc.sbatch (or the first pool gate, cluster/puffer_forecast_pool_gate_carc.sbatch).

  gate      OLD (Phase 2b source, 48-model waves) vs NEW (--pool) on block 0 of Phase 2 and Phase 2b, tune and
            refit: every forecast (preds.csv, the trainer's %.9g text), every step's loss and gradient norm, the
            epochs run and best epochs must be identical
  timing    walls.txt (each run's wall, process start to exit) and the pool summaries
  packing   Phase 2b block-0 refits, NEW --pack K vs NEW --pool: forecast differences; fit-space QLIKE / MSE of
            the 5-seed mean over the block's 250 origins, with the seed-to-seed spread of the single-seed values
            as the noise yardstick
  exactness 16-refit groups (all_features, baseline; seed 20261012), separate vs packed K = 4 / 8 / 16, after 1
            epoch and at the end: per-model parameters (single-model layout, the decoder's unused value row
            excluded), max |diff| / max |param| and relative L2; forecasts

    python experiments/puffer_forecast/pool_gate_report.py RUN_DIR
Writes results/puffer_forecast_2026-10-12/pool/{gate.csv, walls.csv, packing.csv, pack_exactness.csv}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "puffer_forecast_2026-10-12" / "pool"
DATA = REPO / "results" / "puffer_forecast_2026-10-12" / "_work" / "data"


def read_sorted(d: Path, name: str, keys: list[str]) -> pd.DataFrame:
    f = d / name
    if not f.is_file():
        f = d / (name + ".gz")
    df = pd.read_csv(f, dtype=str)
    return df.sort_values(keys).reset_index(drop=True)


def same(a: Path, b: Path) -> dict:
    pa = read_sorted(a, "preds.csv", ["id", "kind", "row"])
    pb = read_sorted(b, "preds.csv", ["id", "kind", "row"])
    sa = read_sorted(a, "steps.csv", ["id", "epoch", "step"])
    sb = read_sorted(b, "steps.csv", ["id", "epoch", "step"])
    ja = pd.read_csv(a / "jobs_out.csv").set_index("id").sort_index()
    jb = pd.read_csv(b / "jobs_out.csv").set_index("id").sort_index()
    return dict(
        jobs=len(ja),
        preds=len(pa),
        preds_identical=bool(pa.equals(pb)),
        steps_identical=bool(
            sa[["id", "epoch", "step", "loss", "grad_norm"]].equals(
                sb[["id", "epoch", "step", "loss", "grad_norm"]]
            )
        ),
        epochs_identical=bool(
            ja["epochs_run"].equals(jb["epochs_run"])
            and ja["best_epoch"].equals(jb["best_epoch"])
        ),
    )


def fit_metrics(pr: pd.DataFrame, design: str) -> dict:
    y = np.fromfile(DATA / design / "y.f32", "<f4").astype(np.float64)
    pr = pr[pr["id"].str.contains(f"_{design}_r")].copy()
    pr["seed"] = pr["id"].str.extract(r"_s(\d+)$")[0]
    pr["y"] = y[pr["row"].to_numpy()]
    ens = pr.groupby("row").agg(f=("f", "mean"), y=("y", "first"))

    def ql(f, yy):
        r = yy**2 / f**2
        return float(np.mean(r - np.log(r) - 1))

    per_seed = [ql(g["f"].to_numpy(), g["y"].to_numpy()) for _, g in pr.groupby("seed")]
    per_seed_mse = [
        float(np.mean((g["f"] - g["y"]) ** 2)) for _, g in pr.groupby("seed")
    ]
    return dict(
        qlike=ql(ens["f"].to_numpy(), ens["y"].to_numpy()),
        mse=float(np.mean((ens["f"] - ens["y"]) ** 2)),
        qlike_seed_sd=float(np.std(per_seed)),
        mse_seed_sd=float(np.std(per_seed_mse)),
        n=len(ens),
    )


def single_layout(D: int, H: int, layers: int) -> np.ndarray:
    """mask of the single-model parameters compared (the decoder's value row excluded)."""
    m = np.ones(H * D + 2 * H + layers * 3 * H * H, bool)
    m[H * D + H : H * D + 2 * H] = False
    return m


def main() -> None:  # noqa: C901
    run = Path(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    gates = []
    for ph in ("phase2", "phase2b"):
        for part in ("tune", "refit"):
            a, b = run / f"{ph}_old_{part}", run / f"{ph}_new_{part}"
            if a.is_dir() and b.is_dir():
                gates.append(dict(phase=ph, part=part, **same(a, b)))
    gdf = pd.DataFrame(gates)
    gdf.to_csv(OUT / "gate.csv", index=False)
    walls = pd.read_csv(run / "walls.txt", sep=" ", names=["run", "wall_s"])
    for _, r in walls.iterrows():
        f = run / r["run"] / "pool_summary.csv"
        if f.is_file():
            ps = pd.read_csv(f)
            walls.loc[walls["run"] == r["run"], "setup_s"] = ps["sum_setup"].sum()
            walls.loc[walls["run"] == r["run"], "capture_s"] = ps["sum_capture"].sum()
            walls.loc[walls["run"] == r["run"], "n_capture"] = ps["n_capture"].sum()
        else:
            jo = run / r["run"] / "jobs_out.csv"
            if jo.is_file():
                j = pd.read_csv(jo)
                walls.loc[walls["run"] == r["run"], "setup_s"] = j["t_setup"].sum()
                walls.loc[walls["run"] == r["run"], "capture_s"] = j["t_capture"].sum()
    walls.to_csv(OUT / "walls.csv", index=False)
    # packing vs pool on the Phase 2b block-0 refits
    rows = []
    base = run / "phase2b_new_refit"
    if base.is_dir():
        pb = pd.read_csv(base / "preds.csv")
        for tag in ["phase2b_new_refit"] + [
            f"phase2b_pack{K}_refit" for K in (4, 8, 16)
        ]:
            d = run / tag
            if not (d / "preds.csv").is_file():
                continue
            pr = pd.read_csv(d / "preds.csv")
            mm = pr.merge(pb, on=["id", "kind", "row"], suffixes=("", "_pool"))
            for design in ("baseline", "all_features"):
                fm = fit_metrics(pr, design)
                sel = mm["id"].str.contains(f"_{design}_r")
                rows.append(
                    dict(
                        run=tag,
                        design=design,
                        **fm,
                        max_abs_df_vs_pool=float(
                            (mm.loc[sel, "f"] - mm.loc[sel, "f_pool"]).abs().max()
                        ),
                        wall_s=float(walls.loc[walls["run"] == tag, "wall_s"].iloc[0]),
                    )
                )
    pdf = pd.DataFrame(rows)
    pdf.to_csv(OUT / "packing.csv", index=False)
    # exactness
    ex = []
    for design in ("all_features", "baseline"):
        for v in ("", "_e1"):
            sep = run / f"sep_{design}{v}"
            if not sep.is_dir():
                continue
            js = pd.read_csv(sep / "jobs_out.csv").set_index("id")
            for K in (4, 8, 16):
                pk = run / f"pack{K}_{design}{v}"
                if not pk.is_dir():
                    continue
                ps_ = pd.read_csv(pk / "preds.csv").set_index("id")["f"]
                ss_ = pd.read_csv(sep / "preds.csv").set_index("id")["f"]
                worst_abs, worst_rel = 0.0, 0.0
                for jid, r in js.iterrows():
                    wa = np.fromfile(sep / "params" / f"{jid}.f32", "<f4").astype(
                        np.float64
                    )
                    wb = np.fromfile(pk / "params" / f"{jid}.f32", "<f4").astype(
                        np.float64
                    )
                    D = int(r["n_inputs"]) + 1
                    H = int(r["H"])
                    m = single_layout(D, H, int(r["layers"]))
                    assert len(wa) == len(wb) == len(m), (jid, len(wa), len(wb), len(m))
                    worst_abs = max(
                        worst_abs,
                        float(np.max(np.abs(wa[m] - wb[m])) / np.max(np.abs(wa[m]))),
                    )
                    worst_rel = max(
                        worst_rel,
                        float(np.linalg.norm(wa[m] - wb[m]) / np.linalg.norm(wa[m])),
                    )
                df = (ps_ - ss_.reindex(ps_.index)).abs()
                ex.append(
                    dict(
                        design=design,
                        epochs="1" if v else "full (seed's best epoch)",
                        K=K,
                        models=len(js),
                        epochs_run=int(js["epochs_run"].iloc[0]),
                        param_max_abs_over_max=worst_abs,
                        param_rel_l2_max=worst_rel,
                        forecast_max_abs_diff=float(df.max()),
                        forecast_mean_abs_diff=float(df.mean()),
                    )
                )
    edf = pd.DataFrame(ex)
    edf.to_csv(OUT / "pack_exactness.csv", index=False)
    pd.set_option("display.width", 220)
    print(gdf.to_string(index=False))
    print(walls.to_string(index=False))
    print(pdf.to_string(index=False))
    print(edf.to_string(index=False))
    env = (run / "env.txt").read_text() if (run / "env.txt").is_file() else ""
    (OUT / "env.txt").write_text(env)
    print(env)


if __name__ == "__main__":
    main()
