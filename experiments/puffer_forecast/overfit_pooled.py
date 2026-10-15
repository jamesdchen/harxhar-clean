"""Pooled-panel memorisation run, round 1 (user, 2026-10-14): can the native MinGRU drive the in-sample loss to ~0
on the pooled tree bank's window (24,000 consecutive half-hour bars, one model for all 48 slots of the day)?

Inputs: experiments/puffer_forecast/prep_pooled.py (data/pooled_<set>, slice rows; window w0 = 47, w1 = 24047 =
panel rows [222053, 246053); the 480 bars after it out of sample).  Trainer native_g5b (gate 5b), refit mode,
loss mse in fit space, link=softplus, fc_train=1, fc_n=480, mask=1, xstd=0 (the panel is prescaled), batch 128,
1,000 epochs with the cosine schedule to 0, clip 1.0, one seed.  Grid: both sets x H {256, 1024} x layers {1, 2}
x L {5, 20, 48} x lr {0.003, 0.01}, plus one identity-link replica (all_features H1024 Ly2 L20 lr 0.01, chosen
before the run: the shape round 1 memorised fastest at lr 0.01).  N = 24,000 - (L - 1) training rows per job (the
trainer's sequences lie inside the window).

    python experiments/puffer_forecast/overfit_pooled.py make                 # _work/overfit_pooled_r1/task_XX_cC.txt
    python experiments/puffer_forecast/overfit_pooled.py report RUN [RUN...]  # results/.../overfit_pooled_r1/
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
POOL = WORK / "pooled"
TASKS = WORK / "overfit_pooled_r1"
W0, W1, OOS = 47, 24047, 480
BASE = dict(
    mode="refit",
    loss="mse",
    w0=W0,
    w1=W1,
    batch=128,
    max_epochs=1000,
    epochs=1000,
    momentum=0.95,
    max_grad_norm=1.0,
    anneal=1,
    min_lr_ratio=0.0,
    fc_n=OOS,
    mask=1,
    xstd=0,
    fc_train=1,
    seed=20261012,
    link="softplus",
)
SETS = ("all_features", "baseline")


def job(s: str, H: int, ly: int, L: int, lr: float, tag: str = "", **over) -> str:
    d = dict(BASE, H=H, layers=ly, L=L, lr=lr, **over)
    jid = f"pool_{s}_H{H}_Ly{ly}_L{L}_lr{lr}_{d['link']}{tag}"
    return O.kv(id=jid, data=f"data/pooled_{s}", **d)


def make() -> None:
    """Layout from a V100 calibration (13090280, 3 epochs each): s / epoch H1024 Ly2 L48 7.2, Ly2 L20 4.9, Ly1 L48 4.0,
    Ly1 L5 2.1; the 24 H256 models together 10 (P100 about 1.8x).  Tasks 0-18 (H1024: layers 2 one model per GPU,
    layers 1 two per GPU, the identity replica alone) heaviest first; tasks 19-20 the H256 models of one set each."""
    for f in TASKS.glob("task_*.txt"):
        f.unlink()
    TASKS.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[int, list[str]]] = []
    for L in (48, 20, 5):
        for s in SETS:
            for lr in (0.003, 0.01):
                tasks.append((1, [job(s, 1024, 2, L, lr)]))
    tasks.append((1, [job("all_features", 1024, 2, 20, 0.01, link="identity")]))
    for L in (48, 20, 5):
        for s in SETS:
            tasks.append((2, [job(s, 1024, 1, L, lr) for lr in (0.003, 0.01)]))
    for s in SETS:
        tasks.append(
            (
                12,
                [
                    job(s, 256, ly, L, lr)
                    for ly in (1, 2)
                    for L in (5, 20, 48)
                    for lr in (0.003, 0.01)
                ],
            )
        )
    rows: list[dict] = []
    for i, (c, lines) in enumerate(tasks):
        name = f"task_{i:02d}_c{c}.txt"
        (TASKS / name).write_bytes(("\n".join(lines) + "\n").encode())
        rows.append(
            dict(
                task=i,
                file=name,
                concurrent=c,
                jobs=len(lines),
                first=lines[0].split()[0],
            )
        )
    (TASKS / "tasks.json").write_bytes(json.dumps(rows, indent=1).encode())
    print(
        pd.DataFrame(rows).to_string(index=False), sum(r["jobs"] for r in rows), "jobs"
    )


def report(runs: list[Path]) -> None:
    out = RES / "overfit_pooled_r1"
    out.mkdir(parents=True, exist_ok=True)
    rows_t = pd.read_csv(POOL / "rows.csv")
    slot = rows_t["slot"].to_numpy()
    ys = {
        s: np.fromfile(POOL / f"pooled_{s}" / "y.f32", "<f4").astype(np.float64)
        for s in SETS
    }
    recs, curves = [], {}
    for run in runs:
        for tdir in sorted(
            p for p in run.iterdir() if p.is_dir() and (p / "jobs_out.csv").is_file()
        ):
            jo = pd.read_csv(tdir / "jobs_out.csv")
            # per-epoch curves reduced on the cluster from the step logs (epoch_curves_remote.py: row-weighted
            # epoch mean of the minibatch loss, mean / max grad norm), so the ~200 MB of step logs stay there
            ec = pd.read_csv(tdir / "epoch_curves.csv")
            pr = pd.read_csv(tdir / "preds.csv")
            env = (tdir / "env.txt").read_text(errors="replace").splitlines()
            gpu = env[2].split(",")[0] if len(env) > 2 else ""
            for _, r in jo.iterrows():
                jid = r["id"]
                s = "all_features" if "_all_features_" in jid else "baseline"
                y = ys[s]
                tr = pr[(pr["id"] == jid) & (pr["kind"] == "train")].sort_values("row")
                rr = (
                    tr["row"].to_numpy() - 0
                )  # preds rows are slice rows (w0 + ctx row)
                yy, f = y[rr], tr["f"].to_numpy().astype(np.float64)
                var = float(
                    np.var(y[W0:W1])
                )  # the window's target variance (all 24,000 bars)
                e2 = (f - yy) ** 2
                q = yy**2 / np.maximum(f, 1e-12) ** 2
                per_slot = pd.Series(e2).groupby(slot[rr]).mean() / pd.Series(
                    yy
                ).groupby(slot[rr]).var(ddof=0)
                fc = pr[(pr["id"] == jid) & (pr["kind"] == "fc")].sort_values("row")
                yo, fo = y[fc["row"].to_numpy()], fc["f"].to_numpy().astype(np.float64)
                qo = yo**2 / np.maximum(fo, 1e-12) ** 2
                e = (
                    ec[ec["id"] == jid]
                    .sort_values("epoch")
                    .rename(columns={"gn_mean": "gn"})
                )
                rel = e["loss_w"].to_numpy() / var
                ep = e["epoch"].to_numpy()
                curves[jid] = (ep, rel)

                def first(th, rel=rel, ep=ep):
                    k = np.flatnonzero(rel <= th)
                    return int(ep[k[0]]) if len(k) else np.nan

                rec = dict(
                    id=jid,
                    set=s,
                    H=int(r["H"]),
                    layers=int(r["layers"]),
                    L=int(r["L"]),
                    lr=float(r["lr"]),
                    link=r["link"],
                    epochs=int(r["epochs_run"]),
                    n_params=int(r["n_params"]),
                    n_train=len(tr),
                    one_minus_r2=float(e2.mean() / var),
                    in_sample_qlike=float(np.mean(q - np.log(q) - 1)),
                    one_minus_r2_slot1600=float(per_slot.get(32, np.nan)),
                    worst_slot_one_minus_r2=float(per_slot.max()),
                    worst_slot=int(per_slot.idxmax()),
                    epochs_to_1e2=first(1e-2),
                    epochs_to_1e3=first(1e-3),
                    epochs_to_1e6=first(1e-6),
                    final_epoch_train_loss_rel=float(rel[-1]) if len(rel) else np.nan,
                    gn_last_epoch_mean=float(e["gn"].iloc[-1]) if len(e) else np.nan,
                    min_f=float(f.min()),
                    oos_mse=float(np.mean((fo - yo) ** 2)),
                    oos_qlike=float(np.mean(qo - np.log(qo) - 1)),
                    oos_n=len(fo),
                    nonfinite=int(r["nonfinite"]),
                    status=r["status"],
                    wall_s=float(r["t_train"]),
                    gpu=gpu,
                    task=tdir.parent.name + "/" + tdir.name,
                )
                rec.update({f"slot_{k:02d}": float(v) for k, v in per_slot.items()})
                recs.append(rec)
    tab = pd.DataFrame(recs).sort_values(["set", "one_minus_r2"])
    tab.to_csv(out / "results.csv", index=False)
    pd.DataFrame(
        [
            dict(id=k, epoch=int(a), train_loss_rel=float(b))
            for k, (ep, rel) in curves.items()
            for a, b in zip(ep, rel)
        ]
    ).to_csv(out / "curves.csv.gz", index=False, float_format="%.6g")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for s in SETS:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
        for a, H in enumerate((256, 1024)):
            for b, ly in enumerate((1, 2)):
                ax = axes[a][b]
                sub = tab[(tab["set"] == s) & (tab["H"] == H) & (tab["layers"] == ly)]
                for _, r in sub.sort_values(["L", "lr"]).iterrows():
                    x, v = curves[r["id"]]
                    ax.semilogy(
                        x,
                        np.maximum(v, 1e-9),
                        lw=0.9,
                        ls="-" if r["link"] == "softplus" else ":",
                        label=f"L{r['L']} lr{r['lr']} {r['link']} ({r['one_minus_r2']:.1e})",
                    )
                for th in (1e-2, 1e-3, 1e-6):
                    ax.axhline(th, color="grey", ls=":", lw=0.7)
                ax.set_title(f"pooled {s}  H {H}  layers {ly}", fontsize=9)
                ax.legend(fontsize=6)
                ax.set_xlabel("epoch")
        axes[0][0].set_ylabel("train loss / window target var")
        fig.tight_layout()
        fig.savefig(out / f"curves_pooled_{s}.png", dpi=100)
        plt.close(fig)
    pd.set_option("display.width", 250)
    cols = [
        "id",
        "n_params",
        "one_minus_r2",
        "one_minus_r2_slot1600",
        "worst_slot_one_minus_r2",
        "in_sample_qlike",
        "epochs_to_1e2",
        "epochs_to_1e3",
        "epochs_to_1e6",
        "min_f",
        "oos_qlike",
        "oos_mse",
        "nonfinite",
        "wall_s",
        "gpu",
    ]
    print(tab[cols].to_string(index=False))
    trees_oos().to_csv(out / "oos_pooled_trees.csv", index=False)


def trees_oos() -> pd.DataFrame:
    """The pooled trees' forecasts of the same 480 out-of-sample bars (panel rows [246053, 246533)), in fit space:
    all_features = the bank's 20 LightGBM tree experts (harxhar-clean/results/unification_carc/tree_expert_XX,
    chunk 80) -- each, their mean and the expert the bank's selector held at that boundary (tree_selection.csv);
    baseline = the pooled tree walks on the backbone (results/pooled_trees_2026-10-07/yhat_pooltree_<est>_baseline,
    table row = panel row - 24,000)."""
    lo, hi = 246053, 246053 + OOS
    y = np.fromfile(POOL / "pooled_baseline" / "y.f32", "<f4").astype(np.float64)[
        W1 : W1 + OOS
    ]

    def score(f: np.ndarray) -> dict:
        q = y**2 / np.maximum(f, 1e-12) ** 2
        return dict(
            oos_qlike=float(np.mean(q - np.log(q) - 1)),
            oos_mse=float(np.mean((f - y) ** 2)),
            n=len(f),
        )

    rows = []
    clean = O.REPO.parent / "harxhar-clean" / "results"
    fs = []
    for k in range(20):
        z = np.load(
            clean / "unification_carc" / f"tree_expert_{k:02d}" / "chunk_080.npz",
            allow_pickle=True,
        )
        rid, yh = z["row_id"], z["yhat"]
        sel = (rid >= lo) & (rid < hi)
        assert sel.sum() == OOS and np.allclose(z["y_fit"][sel], y, atol=1e-12)
        fs.append(yh[sel])
        rows.append(
            dict(set="all_features", model=f"tree_expert_{k:02d}", **score(yh[sel]))
        )
    rows.append(
        dict(
            set="all_features",
            model="tree experts, mean of 20",
            **score(np.mean(fs, axis=0)),
        )
    )
    ts = pd.read_csv(clean / "tree_selection.csv")
    held = ts[ts["boundary_row"] <= lo].iloc[-1]
    rows.append(
        dict(
            set="all_features",
            model=f"selector's expert at the boundary ({held['selected_arm']})",
            **score(fs[int(held["selected_expert"])]),
        )
    )
    for est in ("lgbm", "xgb", "rf"):
        f = (
            O.REPO
            / "results"
            / "pooled_trees_2026-10-07"
            / f"yhat_pooltree_{est}_baseline.parquet"
        )
        if f.is_file():
            t = pd.read_parquet(f, columns=["yhat"])["yhat"].to_numpy()[
                lo - 24_000 : hi - 24_000
            ]
            rows.append(
                dict(set="baseline", model=f"pooltree_{est}_baseline", **score(t))
            )
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make()
    elif sys.argv[1] == "trees":
        trees_oos()
    else:
        report([Path(p) for p in sys.argv[2:]])
