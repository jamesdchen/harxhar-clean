"""Decoupled weight decay (user, 2026-10-14): gate 6 inputs / checks, the wd calibration, and the walk inputs.

The trainer (native_g6) applies W <- W (1 - lr_t wd) to every weight matrix just before each Muon step (upstream
muon_weight_update's own formula W (1 - lr wd) - lr u; the encoder's last column -- its constant-input bias -- is
not decayed).  Job key wd, default 0 (nothing launched).

    python experiments/puffer_forecast/wd_study.py gate-inputs          # _work/gate6/
    python experiments/puffer_forecast/wd_study.py gate RUN             # gate-6 checks (needs gate4.py / gate5.py)
    python experiments/puffer_forecast/wd_study.py calib-inputs         # _work/wd_calib/calib.txt
    python experiments/puffer_forecast/wd_study.py calib RUN            # norms + analytic shrink factors
    python experiments/puffer_forecast/wd_study.py walk-inputs W1,W2,.. # _work/walk_softplus_wd<w>/, walk_softplus_wdgrid/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import overfit as O  # noqa: E402

WORK = O.WORK
RES = O.RES
TOY = "data=data/toy w0=0 w1=390 L=6 H=8 layers=2 fc_n=10 val_tail=60 embargo=25 patience=40 momentum=0.95 max_grad_norm=1.0 anneal=1 min_lr_ratio=0.0 loss=mse link=softplus mask=0 xstd=1"
BASE = (
    "data=data/baseline w0=381 w1=2381 L=5 H=16 layers=1 fc_n=1 val_tail=125 embargo=25 patience=20 momentum=0.95 "
    "max_grad_norm=1.0 anneal=1 min_lr_ratio=0.0 loss=mse link=softplus mask=1 xstd=0"
)


def write(d: Path, name: str, lines: list[str]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(("\n".join(lines) + "\n").encode())


def gate_inputs() -> None:
    g = WORK / "gate6"
    # one Muon step (batch >= N: one step per epoch, epochs=1), wd 0 and > 0, + gradcheck dumps of the same step
    step = [
        f"id=step_toy_wd0 mode=refit epochs=1 max_epochs=40 batch=512 lr=0.05 seed=5 wd=0 save_params=1 {TOY}",
        f"id=step_toy_wd0.5 mode=refit epochs=1 max_epochs=40 batch=512 lr=0.05 seed=5 wd=0.5 save_params=1 {TOY}",
        f"id=step_base_wd0 mode=refit epochs=1 max_epochs=200 batch=2048 lr=0.03 seed=20261012 wd=0 save_params=1 {BASE}",
        f"id=step_base_wd0.3 mode=refit epochs=1 max_epochs=200 batch=2048 lr=0.03 seed=20261012 wd=0.3 save_params=1 {BASE}",
    ]
    write(g, "step.txt", step)
    write(
        g,
        "step_grad.txt",
        [
            line.replace("save_params=1 ", "").replace("id=step_", "id=grad_")
            for line in step
        ],
    )
    # weight norms against wd on a toy job (40 epochs)
    write(
        g,
        "norm_toy.txt",
        [
            f"id=norm_toy_wd{w} mode=refit epochs=40 max_epochs=40 batch=32 lr=0.03 seed=5 wd={w} "
            f"save_params=1 {TOY}"
            for w in (0, 0.03, 0.1, 0.3, 1, 3, 10)
        ],
    )
    ok = f"mode=refit epochs=5 max_epochs=40 batch=32 lr=0.03 seed=5 {TOY}".replace(
        "link=softplus", "link=identity"
    )
    write(g, "bad_wd_neg.txt", [f"id=bad wd=-0.1 {ok}"])
    write(g, "bad_wd_nan.txt", [f"id=bad wd=nan {ok}"])
    write(g, "wd_pack.txt", [f"id=bad wd=0.1 {ok}"])
    print(sorted(p.name for p in g.iterdir()))


def decayed_mask(P: int, H: int, layers: int) -> dict:
    """True where the trainer decays: every entry but the encoder's last (constant-input) column."""
    we = np.ones((H, P), bool)
    we[:, P - 1] = False
    return dict(
        We=we,
        Wd=np.ones((2, H), bool),
        Wl=[np.ones((3 * H, H), bool) for _ in range(layers)],
    )


def gate(run: Path) -> dict:
    import gate5 as G5
    import reference as RF

    res: dict = {"env": (run / "env.txt").read_text(errors="replace").splitlines()[:3]}
    import gate4 as G4

    res["tune_g5b_vs_g6"] = G4.identical(run / "id_old_tune", run / "id_new_tune")
    for part in ("refit", "fct"):
        res[f"{part}_g5b_vs_g6"] = G4.identical(
            run / f"id_old_{part}", run / f"id_new_{part}"
        )
    res["tune_g6_pool_vs_wave"] = G4.identical(
        run / "id_new_wave_tune", run / "id_new_tune"
    )
    res["refit_g6_pool_vs_wave"] = G4.identical(
        run / "id_new_wave_refit", run / "id_new_refit"
    )
    g, ok = G5.guards(run)
    res["guards"], res["guards_ok"] = g.to_dict("records"), ok
    # one step vs float64: W_ref = Muon(decay(W_init)), and W_wd - W_0 = -lr wd W_init on decayed entries, 0 elsewhere
    steps = []
    for tag, wd, lr in (("toy", 0.5, 0.05), ("base", 0.3, 0.03)):
        meta = json.loads((run / "gradcheck" / f"grad_{tag}_wd{wd:g}.json").read_text())
        P, H, ly = meta["P"], meta["H"], meta["layers"]
        w0 = RF.read_f32(run / "gradcheck" / f"grad_{tag}_wd{wd:g}.params0.f32")
        gr = RF.read_f32(run / "gradcheck" / f"grad_{tag}_wd{wd:g}.grads.f32")
        n_wd = RF.read_f32(run / "step" / "params" / f"step_{tag}_wd{wd:g}.f32")
        n_0 = RF.read_f32(run / "step" / "params" / f"step_{tag}_wd0.f32")
        msk = decayed_mask(P, H, ly)
        mflat = np.concatenate(
            [msk["We"].ravel(), msk["Wd"].ravel()] + [x.ravel() for x in msk["Wl"]]
        )
        w = RF.unpack(w0.copy(), P, H, ly)
        w["We"] = np.where(msk["We"], w["We"] * (1 - lr * wd), w["We"])
        w["Wd"] = w["Wd"] * (1 - lr * wd)
        w["Wl"] = [x * (1 - lr * wd) for x in w["Wl"]]
        g64 = RF.unpack(gr, P, H, ly)
        mb = dict(
            We=np.zeros_like(w["We"]),
            Wd=np.zeros_like(w["Wd"]),
            Wl=[np.zeros_like(x) for x in w["Wl"]],
        )
        RF.muon_step(
            w, g64, mb, lr, float(meta["momentum"]), float(meta["max_grad_norm"])
        )
        ref = RF.pack(w)
        scale = float(np.max(np.abs(ref)))
        diff = n_wd - n_0
        expect = np.where(mflat, -lr * wd * w0, 0.0)
        steps.append(
            dict(
                job=f"{tag} wd={wd} lr={lr}",
                n_params=len(w0),
                decayed=int(mflat.sum()),
                not_decayed=int((~mflat).sum()),
                native_vs_f64_max_abs_over_max=float(
                    np.max(np.abs(n_wd - ref)) / scale
                ),
                decay_term_max_abs_err=float(np.max(np.abs(diff - expect))),
                decay_term_max_abs=float(np.max(np.abs(expect))),
                bias_column_unchanged_vs_wd0=bool(np.all(diff[~mflat] == 0.0)),
            )
        )
    res["one_step"] = steps
    # weight norms vs wd on the toy job (40 epochs)
    norms = []
    for wd in (0, 0.03, 0.1, 0.3, 1, 3, 10):
        jid = f"norm_toy_wd{wd:g}"
        fl = RF.read_f32(run / "norm" / "params" / f"{jid}.f32")
        meta = json.loads((run / "gradcheck" / "grad_toy_wd0.5.json").read_text())
        w = RF.unpack(fl, meta["P"], meta["H"], meta["layers"])
        norms.append(
            dict(
                wd=wd,
                fro_all=float(np.sqrt(np.sum(fl**2))),
                fro_encoder=float(np.linalg.norm(w["We"])),
                fro_decoder=float(np.linalg.norm(w["Wd"])),
                fro_layers=float(np.sqrt(sum(np.sum(x**2) for x in w["Wl"]))),
                # the decayed entries (all but the encoder's constant-input column) and that column
                fro_decayed=float(
                    np.sqrt(
                        np.sum(w["We"][:, :-1] ** 2)
                        + np.sum(w["Wd"] ** 2)
                        + sum(np.sum(x**2) for x in w["Wl"])
                    )
                ),
                norm_bias_column=float(np.linalg.norm(w["We"][:, -1])),
            )
        )
    nt = pd.DataFrame(norms)
    res["toy_norms"] = nt.to_dict("records")
    res["toy_decayed_norm_monotone"] = bool(
        np.all(np.diff(nt["fro_decayed"].to_numpy()) < 0)
    )
    res["toy_all_norm_monotone"] = bool(np.all(np.diff(nt["fro_all"].to_numpy()) < 0))
    out = RES / "review_build" / f"gate6_{run.name}.json"
    out.write_bytes(json.dumps(res, indent=1, default=str).encode())
    for k, v in res.items():
        if k not in ("guards", "env", "toy_norms", "one_step"):
            print(k, v)
    print(g.to_string(index=False))
    print(pd.DataFrame(steps).to_string(index=False))
    print(nt.to_string(index=False))
    return res


CALIB_CFG = ((20, 16, 0.03), (20, 64, 0.03), (5, 64, 0.01), (5, 16, 0.003))
CALIB_WD = (0, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3)


def calib_inputs() -> None:
    lines = []
    for d in ("baseline", "all_features"):
        for L, H, lr in CALIB_CFG:
            for wd in CALIB_WD:
                lines.append(
                    f"id=cal_{d}_L{L}_H{H}_lr{lr}_wd{wd:g} data=data/{d} mode=refit w0=381 w1=2381 L={L} H={H} "
                    f"lr={lr} wd={wd} epochs=20 seed=20261012 save_params=1 layers=1 batch=128 max_epochs=200 "
                    "patience=20 val_tail=125 embargo=25 momentum=0.95 max_grad_norm=1.0 anneal=1 "
                    "min_lr_ratio=0.0 fc_n=1 mask=1 xstd=0 loss=mse link=softplus"
                )
    write(WORK / "wd_calib", "calib.txt", lines)
    print(len(lines))


def shrink(
    lr: float, wd: float, epochs: int, S: int = 15, max_epochs: int = 200
) -> float:
    """prod over the run's steps of (1 - lr_t wd), lr_t the cosine schedule (one value per epoch, S steps)."""
    e = np.arange(epochs)
    lr_e = 0.5 * lr * (1 + np.cos(np.pi * e / max_epochs))
    return float(np.prod((1 - lr_e * wd) ** S))


def calib(run: Path) -> pd.DataFrame:
    import reference as RF

    jo = pd.read_csv(run / "jobs_out.csv").set_index("id")
    rows = []
    for jid, r in jo.iterrows():
        fl = RF.read_f32(run / "params" / f"{jid}.f32")
        keep = np.fromfile(run / "params" / f"{jid}.keep.i32", "<i4")
        P, H = len(keep) + 1, int(r["H"])
        w = RF.unpack(fl, P, H, 1)
        rows.append(
            dict(
                id=jid,
                design="all_features" if "_all_features_" in jid else "baseline",
                L=int(r["L"]),
                H=H,
                lr=float(r["lr"]),
                wd=float(r["wd"]),
                fro_all=float(np.sqrt(np.sum(fl**2))),
                fro_encoder=float(np.linalg.norm(w["We"])),
                fro_layer=float(np.linalg.norm(w["Wl"][0])),
                fro_decoder=float(np.linalg.norm(w["Wd"])),
                shrink_20ep=shrink(float(r["lr"]), float(r["wd"]), 20),
            )
        )
    t = pd.DataFrame(rows)
    base = t[t["wd"] == 0].set_index(["design", "L", "H", "lr"])["fro_all"]
    t["fro_rel_wd0"] = [
        r["fro_all"] / base[(r["design"], r["L"], r["H"], r["lr"])]
        for _, r in t.iterrows()
    ]
    out = RES / "wd_study"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "calibration_norms.csv", index=False)
    an = pd.DataFrame(
        [
            dict(lr=lr, epochs=E, wd=wd, shrink=shrink(lr, wd, E))
            for lr in (0.003, 0.01, 0.03)
            for E in (5, 20, 60)
            for wd in CALIB_WD
        ]
    )
    an.to_csv(out / "calibration_shrink.csv", index=False)
    pd.set_option("display.width", 220)
    print(
        t.pivot_table(
            index=["design", "L", "H", "lr"], columns="wd", values="fro_rel_wd0"
        )
        .round(3)
        .to_string()
    )
    print(
        an.pivot_table(index=["lr", "epochs"], columns="wd", values="shrink")
        .round(4)
        .to_string()
    )
    return t


def walk_inputs(grid: list[float]) -> None:
    src = WORK / "phase2c"
    m = json.loads((src / "blocks.json").read_text())
    # A: wd fixed in every tune and refit job
    for wd in grid:
        d = WORK / f"walk_softplus_wd{wd:g}"
        d.mkdir(exist_ok=True)
        for k in range(5):
            lines = (src / f"tune_block{k}.txt").read_text().splitlines()
            write(
                d,
                f"tune_block{k}.txt",
                [f"{x} link=softplus wd={wd:g}" for x in lines if x.strip()],
            )
        mm = json.loads(json.dumps(m))
        mm["common"]["link"] = "softplus"
        mm["common"]["wd"] = wd
        (d / "blocks.json").write_bytes(json.dumps(mm, indent=1).encode())
    # B: wd in each block's tune grid (12 configurations x the wd values); the selection carries it to the refits
    d = WORK / "walk_softplus_wdgrid"
    d.mkdir(exist_ok=True)
    for k in range(5):
        lines = [
            x
            for x in (src / f"tune_block{k}.txt").read_text().splitlines()
            if x.strip()
        ]
        out = []
        for x in lines:
            jid = x.split()[0]
            stem, seed = jid.rsplit("_s", 1)
            for wd in grid:
                out.append(
                    x.replace(jid, f"{stem}_wd{wd:g}_s{seed}", 1)
                    + f" link=softplus wd={wd:g}"
                )
        write(d, f"tune_block{k}.txt", out)
    mm = json.loads(json.dumps(m))
    mm["common"]["link"] = "softplus"
    (d / "blocks.json").write_bytes(json.dumps(mm, indent=1).encode())
    print([p.name for p in sorted(WORK.glob("walk_softplus_wd*"))])


STUDY = RES / "wd_study"
WALKS = {
    "0": "walk_wd0_13135336",
    "0.01": "walk_wd0.01_13135337",
    "0.03": "walk_wd0.03_13135338",
    "0.1": "walk_wd0.1_13135339",
    "0.3": "walk_wd0.3_13135340",
    "1": "walk_wd1_13135341",
    "grid": "walk_wdgrid_13135342",
}


def contrast(a: str, b: str) -> pd.DataFrame:
    """Walk score dir a vs b (seed-mean forecasts, the stack's scorer): DM (HLN) on daily QLIKE / fit-space squared
    error, HAC t of the daily mid / crossed P&L differences (as phase2_score's contrasts)."""
    repo = O.REPO
    for p in (
        repo,
        repo / "experiments",
        repo / "notebooks",
        repo / "specs",
        repo / "writeup",
    ):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import atm_straddle_lib as asl
    import close_linear_pretune as CLP
    import feature_mcs_report as FMR
    import feature_mcs_runner as FR
    from src.evaluation.diebold_mariano import dm_test

    inp = json.loads((FR.INP / "inputs.json").read_text())
    W = int(inp["W"])
    n = len(inp["dates"]) - W
    y_adj = np.load(FR.INP / "bar_y.npy")[W:]
    S = CLP.Scorer()
    day_row = np.load(FR.INP / "ref_armf.npz")["day_row"]
    rows = []
    for d in ("baseline", "all_features"):
        losses = []
        for name in (a, b):
            e = pd.read_csv(STUDY / name / "forecasts.csv")
            e = e[e["design"] == d]
            pred = np.full(n, np.nan)
            pred[e["row"].to_numpy() - W] = e["mean"].to_numpy()
            losses.append(FMR.cand_losses(S, pred, y_adj, day_row))
        g, h = losses
        c = S.contrast(g, h)
        dm_mse = dm_test(g["mse_fit"], h["mse_fit"])
        t_x, _ = asl.newey_west_t(g["x"] - h["x"])
        rows.append(
            dict(
                design=d,
                walk=a,
                vs=b,
                qlike_pct=c["dq_pct"],
                dm_qlike=c["dm"],
                p_qlike=c["dm_p"],
                mse_pct=float(
                    100 * (np.mean(g["mse_fit"]) / np.mean(h["mse_fit"]) - 1)
                ),
                dm_mse=float(dm_mse["dm"]),
                p_mse=float(dm_mse["p"]),
                hac_t_mid=c["hac_t_mid"],
                hac_t_crossed=float(t_x),
                n_diff_position=c["n_diff_position"],
            )
        )
    return pd.DataFrame(rows)


def study() -> None:
    import reference as RF

    runs = STUDY / "runs"
    # scores and the stack's contrasts (enet, trees) of every walk
    sc, cn = [], []
    for wd, run in WALKS.items():
        name = f"score_wd{wd}" if wd != "grid" else "score_wdgrid"
        s = pd.read_csv(STUDY / name / "scores.csv")
        sc.append(s[s["model"].str.startswith("minGRU")].assign(wd=wd))
        c = pd.read_csv(STUDY / name / "contrasts.csv")
        cn.append(c[c["vs"].isin(["enet", "lgbm", "xgb", "rf"])].assign(wd=wd))
    scores = pd.concat(sc)[
        [
            "wd",
            "design",
            "qlike",
            "mse_fit",
            "es1",
            "sharpe_mid",
            "sharpe_crossed",
            "pct_buy",
        ]
    ]
    stack = pd.concat(cn)[
        [
            "wd",
            "design",
            "vs",
            "qlike_pct",
            "p_qlike",
            "mse_pct",
            "p_mse",
            "hac_t_mid",
            "hac_t_crossed",
        ]
    ]
    vs0 = pd.concat(
        [
            contrast(
                f"score_wd{wd}" if wd != "grid" else "score_wdgrid", "score_wd0"
            ).assign(wd=wd)
            for wd in WALKS
            if wd != "0"
        ]
    )
    scores.to_csv(STUDY / "scores.csv", index=False)
    stack.to_csv(STUDY / "contrasts_vs_stack.csv", index=False)
    vs0.to_csv(STUDY / "contrasts_vs_wd0.csv", index=False)
    # B: the wd each block / design picked
    sel = pd.read_csv(STUDY / "score_wdgrid" / "selection.csv")
    sel["wd"] = sel["cfg"].str.extract(r"_wd([0-9.e+-]+)$")[0]
    sel.to_csv(STUDY / "grid_selection.csv", index=False)
    # best epochs against wd: every tune job of the fixed-wd walks (all 12 configurations, 5 seeds, 5 blocks)
    be = []
    for wd, run in WALKS.items():
        if wd == "grid":
            continue
        for k in range(5):
            jo = pd.read_csv(runs / run / f"block{k}" / "tune" / "jobs_out.csv")
            jo["design"] = jo["id"].str.extract(r"^tune_(baseline|all_features)_")[0]
            jo["cfg"] = jo["id"].str.extract(r"_(L\d+_H\d+_lr[0-9.]+)_s\d+$")[0]
            be.append(
                jo.assign(wd=float(wd), block=k)[
                    ["wd", "block", "design", "cfg", "best_epoch", "best_val_mse"]
                ]
            )
    be = pd.concat(be)
    best = (
        be.groupby(["design", "wd"])
        .agg(
            best_epoch_median=("best_epoch", "median"),
            best_epoch_mean=("best_epoch", "mean"),
            val_mse_median=("best_val_mse", "median"),
        )
        .reset_index()
    )
    best_cfg = be.groupby(["design", "cfg", "wd"])["best_epoch"].mean().unstack("wd")
    best.to_csv(STUDY / "best_epochs_vs_wd.csv", index=False)
    best_cfg.to_csv(STUDY / "best_epochs_by_cfg_vs_wd.csv")
    # weight norms at the best epoch: each block's first-origin refits (save_params), every seed
    nm = []
    for wd, run in WALKS.items():
        for k in range(5):
            selj = json.loads((runs / run / f"block{k}" / "selection.json").read_text())
            for f in sorted(
                (runs / run / f"block{k}" / "refit" / "params").glob("*.f32")
            ):
                jid = f.stem
                d = "all_features" if "_all_features_" in jid else "baseline"
                H = int(selj[d]["cfg"].split("_H")[1].split("_")[0])
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
                        wd=wd,
                        block=k,
                        design=d,
                        cfg=selj[d]["cfg"],
                        id=jid,
                        fro_decayed=dec,
                        fro_layer=float(np.linalg.norm(w["Wl"][0])),
                        fro_decoder=float(np.linalg.norm(w["Wd"])),
                    )
                )
    nm = pd.DataFrame(nm)
    nm.to_csv(STUDY / "weight_norms_best_epoch.csv", index=False)
    pd.set_option("display.width", 250)
    print(scores.round(4).to_string(index=False))
    print(vs0.round(3).to_string(index=False))
    print(stack[stack["vs"] == "enet"].round(3).to_string(index=False))
    print(
        stack[stack["vs"] != "enet"]
        .groupby(["wd", "design"])[["qlike_pct", "p_qlike", "mse_pct", "p_mse"]]
        .agg(["min", "max"])
        .round(3)
        .to_string()
    )
    print(sel.to_string(index=False))
    print(best.round(2).to_string(index=False))
    print(
        nm.groupby(["design", "wd"])[["fro_decayed", "fro_layer"]]
        .median()
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "study":
        study()
        sys.exit()
    if cmd == "gate-inputs":
        gate_inputs()
    elif cmd == "gate":
        gate(Path(sys.argv[2]))
    elif cmd == "calib-inputs":
        calib_inputs()
    elif cmd == "calib":
        calib(Path(sys.argv[2]))
    elif cmd == "walk-inputs":
        walk_inputs([float(x) for x in sys.argv[2].split(",")])
