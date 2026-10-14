"""PufferLib-5.0 native MinGRU: capacity / memorisation sweep (2026-10-13).

User: "what I need is for in sample loss to be trained to be essentially eliminated. we will go from there."
Can the native MinGRU drive the TRAINING loss to ~0 on one window?  Phase 2c inputs (prescaled design as is,
xstd=0; the regular stack's window mask, mask=1; no clip; sqrt-space target, loss mse; --pool), the 2,000
sessions ending 2019-12-31 (rows 381 .. 2380), both designs, mode refit (the whole window, no early stopping,
no validation split), cosine annealing to 0 over the run, one seed; fc_train=1 writes the final weights'
forecast of every training row (preds.csv kind "train"), so the in-sample loss is exact.

Round 1: H 64 / 256 / 1024, layers 1 / 2 / 4, L 5 / 20, lr 0.01 / 0.03 / 0.1, batch 128, 1,000 epochs,
max_grad_norm 1.0; plus, on the 6 largest models per design (H 1024, layers 2 / 4, L 20, the 3 lrs), a
variant with Muon's clip effectively off (max_grad_norm 1e30) and one with batch 32.

    python experiments/puffer_forecast/overfit.py make r1           # _work/overfit_r1/task_XX_cC.txt
    python experiments/puffer_forecast/overfit.py report r1 RUN...  # results/.../overfit_r1/ (table, PNGs)
"""

from __future__ import annotations

import gzip
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "results" / "puffer_forecast_2026-10-12" / "_work"
RES = REPO / "results" / "puffer_forecast_2026-10-12"
DESIGNS = ("baseline", "all_features")
W0, W1 = 381, 2381  # the 2,000 16:00 sessions ending 2019-12-31
SMALL_CONCURRENCY = (
    12  # task 0 (round 1: 13083705); a concurrency change is a gate-5 item
)
BASE = dict(
    mode="refit",
    loss="mse",
    w0=W0,
    w1=W1,
    layers=1,
    batch=128,
    max_epochs=1000,
    epochs=1000,
    patience=20,
    val_tail=125,
    embargo=25,
    momentum=0.95,
    max_grad_norm=1.0,
    anneal=1,
    min_lr_ratio=0.0,
    fc_n=1,
    mask=1,
    xstd=0,
    fc_train=1,
    seed=20261012,
)


def kv(**k) -> str:
    return " ".join(f"{a}={b}" for a, b in k.items())


def job(
    design: str, H: int, layers: int, L: int, lr: float, tag: str = "", **over
) -> str:
    jid = f"of_{design}_H{H}_Ly{layers}_L{L}_lr{lr}{tag}"
    d = dict(BASE, H=H, layers=layers, L=L, lr=lr, **over)
    d["epochs"] = d["max_epochs"]
    return kv(id=jid, data=f"data/{design}", **d)


def make_r1() -> None:
    out = WORK / "overfit_r1"
    out.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[int, list[str]]] = []
    small = [
        job(d, H, ly, L, lr)
        for d in DESIGNS
        for H in (64, 256)
        for ly in (1, 2, 4)
        for L in (5, 20)
        for lr in (0.01, 0.03, 0.1)
    ]
    tasks.append((SMALL_CONCURRENCY, small))
    for d in DESIGNS:
        for ly in (1, 2, 4):
            tasks.append(
                (
                    3,
                    [
                        job(d, 1024, ly, L, lr)
                        for L in (5, 20)
                        for lr in (0.01, 0.03, 0.1)
                    ],
                )
            )
    for d in DESIGNS:
        big = [(ly, lr) for ly in (2, 4) for lr in (0.01, 0.03, 0.1)]
        tasks.append(
            (
                3,
                [
                    job(d, 1024, ly, 20, lr, "_noclip", max_grad_norm=1e30)
                    for ly, lr in big
                ],
            )
        )
        tasks.append(
            (3, [job(d, 1024, ly, 20, lr, "_b32", batch=32) for ly, lr in big])
        )
    names: list[dict] = []
    for i, (c, lines) in enumerate(tasks):
        name = f"task_{i:02d}_c{c}.txt"
        (out / name).write_bytes(("\n".join(lines) + "\n").encode())
        names.append(dict(task=i, file=name, concurrent=c, jobs=len(lines)))
    (out / "tasks.json").write_bytes(json.dumps(names, indent=1).encode())
    print(json.dumps(names, indent=1), sum(n["jobs"] for n in names), "jobs")


def read_csv_any(p: Path) -> pd.DataFrame:
    if p.is_file():
        return pd.read_csv(p)
    gz = p.with_name(p.name + ".gz")
    with gzip.open(gz, "rt") as f:
        return pd.read_csv(io.StringIO(f.read()))


def variant_of(jid: str) -> str:
    for v in ("noclip", "b32", "b16", "b512", "b2000", "e3000", "e10000"):
        if jid.endswith("_" + v):
            return v
    return ""


def job_keys(rnd: str) -> dict:
    """id -> the job line's keys, from the round's task files (_work/overfit_<rnd>/task_*.txt)."""
    out = {}
    for f in sorted((WORK / f"overfit_{rnd}").glob("task_*.txt")):
        for line in f.read_text().splitlines():
            kv_ = dict(t.split("=", 1) for t in line.split())
            out[kv_["id"]] = kv_
    return out


def collect(runs: list[Path], mgn: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """One row per job (final-weights in-sample fit from preds kind "train"; curves from steps.csv)."""
    y = {
        d: np.fromfile(WORK / "data" / d / "y.f32", "<f4").astype(np.float64)
        for d in DESIGNS
    }
    rows: list = []
    for run in runs:
        for tdir in sorted(
            p for p in run.iterdir() if p.is_dir() and (p / "jobs_out.csv").is_file()
        ):
            jo = pd.read_csv(tdir / "jobs_out.csv")
            if not len(jo):
                continue
            st = read_csv_any(tdir / "steps.csv")
            pr = pd.read_csv(tdir / "preds.csv")
            env = (
                (tdir / "env.txt").read_text(errors="replace").splitlines()
                if (tdir / "env.txt").is_file()
                else []
            )
            gpu = env[2].split(",")[0] if len(env) > 2 else ""
            for _, r in jo.iterrows():
                jid = r["id"]
                design = "all_features" if "_all_features_" in jid else "baseline"
                tr = pr[(pr["id"] == jid) & (pr["kind"] == "train")].sort_values("row")
                yy = y[design][tr["row"].to_numpy()]
                var = float(
                    np.var(yy)
                )  # the window's training-target variance (population)
                f = tr["f"].to_numpy().astype(np.float64)
                mse = float(np.mean((f - yy) ** 2))
                pos = f > 0
                rr = yy[pos] ** 2 / f[pos] ** 2
                ql = float(np.mean(rr - np.log(rr) - 1)) if pos.any() else np.nan
                fc = (
                    pr[(pr["id"] == jid) & (pr["kind"] == "fc")]
                    .sort_values("row")["f"]
                    .to_numpy()
                )
                fcp = (
                    pr[(pr["id"] == jid) & (pr["kind"] == "fc_trainpath")]
                    .sort_values("row")["f"]
                    .to_numpy()
                )
                s1 = st[st["id"] == jid]
                N, B = int(r["N"]), int(r["batch"])
                g = (mgn or {}).get(jid, 1e30 if jid.endswith("_noclip") else 1.0)
                w = np.minimum(B, N - s1["step"].to_numpy() * B).astype(
                    np.float64
                )  # rows in each minibatch
                s1 = s1.assign(
                    wl=s1["loss"].to_numpy() * w,
                    w=w,
                    clip=(s1["grad_norm"].to_numpy() > g).astype(float),
                )
                e = (
                    s1.groupby("epoch")
                    .agg(
                        wl=("wl", "sum"),
                        w=("w", "sum"),
                        gn=("grad_norm", "mean"),
                        gn_max=("grad_norm", "max"),
                        clip=("clip", "mean"),
                    )
                    .reset_index()
                )
                rel = (e["wl"] / e["w"]).to_numpy() / var
                ep = e["epoch"].to_numpy()
                curve = (ep, rel, e["gn"].to_numpy())

                def first(th, rel=rel, ep=ep):
                    k = np.flatnonzero(rel <= th)
                    return int(ep[k[0]]) if len(k) else np.nan

                rows.append(
                    dict(
                        id=jid,
                        design=design,
                        H=int(r["H"]),
                        layers=int(r["layers"]),
                        L=int(r["L"]),
                        lr=float(r["lr"]),
                        batch=B,
                        max_grad_norm=g,
                        variant=variant_of(jid),
                        epochs=int(r["epochs_run"]),
                        n_params=int(r["n_params"]),
                        n_train=len(tr),
                        target_var=var,
                        in_sample_mse=mse,
                        one_minus_r2=mse / var,
                        in_sample_qlike=ql,
                        n_f_nonpositive=int((~pos).sum()),
                        final_epoch_train_loss_rel=float(rel[-1])
                        if len(rel)
                        else np.nan,
                        min_epoch_train_loss_rel=float(rel.min())
                        if len(rel)
                        else np.nan,
                        epochs_to_1e2=first(1e-2),
                        epochs_to_1e3=first(1e-3),
                        gn_first_epoch_mean=float(e["gn"].iloc[0])
                        if len(e)
                        else np.nan,
                        gn_last_epoch_mean=float(e["gn"].iloc[-1])
                        if len(e)
                        else np.nan,
                        gn_max_overall=float(e["gn_max"].max()) if len(e) else np.nan,
                        clipped_step_share=float(s1["clip"].mean())
                        if len(s1)
                        else np.nan,
                        nonfinite=int(r["nonfinite"]),
                        status=r["status"],
                        wall_s=float(r["t_train"]),
                        fc_equals_fc_trainpath=bool(
                            len(fc) > 0 and np.array_equal(fc, fcp)
                        ),
                        gpu=gpu,
                        task=tdir.parent.name + "/" + tdir.name,
                        _curve=curve,
                    )
                )
    # one row per job id: a later run (the runs list in order) replaces an earlier one unless the earlier row is ok
    # and the later is not (a resubmitted task's rerun replaces the failed or stale one)
    keep: dict = {}
    for r in rows:
        old = keep.get(r["id"])
        if old is None or r["status"] == "ok" or old["status"] != "ok":
            keep[r["id"]] = r
    if len(keep) < len(rows):
        print(
            f"collect: {len(rows) - len(keep)} duplicate job rows replaced by their later run"
        )
    curves = {k: r.pop("_curve") for k, r in keep.items()}
    out = pd.DataFrame(list(keep.values()))
    assert out["id"].is_unique
    return out, curves


def plot_curves(tab: pd.DataFrame, curves: dict, out: Path, tag: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = tab[tab["variant"] == ""]
    Hs = sorted(base["H"].unique())
    Ls = sorted(base["layers"].unique())
    for design in DESIGNS:
        fig, axes = plt.subplots(
            len(Ls),
            len(Hs),
            figsize=(5.2 * len(Hs), 3.6 * len(Ls)),
            sharey=True,
            squeeze=False,
        )
        for a, ly in enumerate(Ls):
            for b, H in enumerate(Hs):
                ax = axes[a][b]
                sub = base[
                    (base["design"] == design)
                    & (base["H"] == H)
                    & (base["layers"] == ly)
                ]
                for _, r in sub.sort_values(["L", "lr"]).iterrows():
                    x, v, _ = curves[r["id"]]
                    ax.semilogy(
                        x,
                        np.maximum(v, 1e-9),
                        lw=0.9,
                        ls="-" if r["L"] == sub["L"].max() else "--",
                        label=f"L{r['L']} lr{r['lr']} (final 1-R2 {r['one_minus_r2']:.1e})",
                    )
                ax.axhline(1e-2, color="grey", ls=":", lw=0.8)
                ax.axhline(1e-3, color="grey", ls="-.", lw=0.8)
                ax.set_title(f"{design}  H {H}  layers {ly}", fontsize=9)
                if len(sub):
                    ax.legend(fontsize=6)
                if a == len(Ls) - 1:
                    ax.set_xlabel("epoch")
                if b == 0:
                    ax.set_ylabel("train loss / target var")
        fig.suptitle(
            "epoch training loss (row-weighted mean of minibatch MSE, fit space) / window target variance; "
            "dotted 1e-2, dash-dot 1e-3",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(out / f"curves_{design}{tag}.png", dpi=100)
        plt.close(fig)
        var = tab[(tab["variant"] != "") & (tab["design"] == design)]
        if not len(var):
            continue
        keys = sorted({(r["H"], r["layers"], r["L"]) for _, r in var.iterrows()})
        fig, axes = plt.subplots(
            1, len(keys), figsize=(6 * len(keys), 4.2), sharey=True, squeeze=False
        )
        for ax, (H, ly, L) in zip(axes[0], keys):
            sub = tab[
                (tab["design"] == design)
                & (tab["H"] == H)
                & (tab["layers"] == ly)
                & (tab["L"] == L)
            ]
            for _, r in sub.sort_values(["variant", "lr"]).iterrows():
                x, v, _ = curves[r["id"]]
                vname = r["variant"] if r["variant"] else "base"
                ax.semilogy(
                    x,
                    np.maximum(v, 1e-9),
                    lw=0.9,
                    ls={"base": "-", "noclip": "--"}.get(vname, ":"),
                    label=f"{vname} lr{r['lr']} ({r['one_minus_r2']:.1e})",
                )
            ax.axhline(1e-2, color="grey", ls=":", lw=0.8)
            ax.axhline(1e-3, color="grey", ls="-.", lw=0.8)
            ax.set_title(
                f"{design}  H {H}  layers {ly}  L {L}: base / variants", fontsize=9
            )
            ax.set_xlabel("epoch")
            ax.legend(fontsize=6)
        axes[0][0].set_ylabel("train loss / target var")
        fig.tight_layout()
        fig.savefig(out / f"curves_{design}_variants{tag}.png", dpi=100)
        plt.close(fig)


def report(rnd: str, runs: list[Path]) -> None:
    out = RES / f"overfit_{rnd}"
    out.mkdir(parents=True, exist_ok=True)
    tab, curves = collect(
        runs, {k: float(v["max_grad_norm"]) for k, v in job_keys(rnd).items()}
    )
    tab = tab.sort_values(["design", "one_minus_r2"])
    tab.to_csv(out / "results.csv", index=False)
    cv = [
        dict(id=k, epoch=int(e), train_loss_rel=float(v), gn_mean=float(g))
        for k, (ep, rel, gn) in curves.items()
        for e, v, g in zip(ep, rel, gn)
    ]
    pd.DataFrame(cv).to_csv(out / "curves.csv.gz", index=False, float_format="%.6g")
    plot_curves(tab, curves, out)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    cols = [
        "id",
        "n_params",
        "one_minus_r2",
        "in_sample_qlike",
        "final_epoch_train_loss_rel",
        "epochs_to_1e2",
        "epochs_to_1e3",
        "gn_last_epoch_mean",
        "clipped_step_share",
        "nonfinite",
        "wall_s",
    ]
    print(tab[cols].to_string(index=False))


def diff_jobs(a: Path, b: Path) -> dict:
    """Job ids whose preds or steps differ between two runs, and whether each differs from its first step."""
    import pool_gate_report as G

    pa = G.read_sorted(a, "preds.csv", ["id", "kind", "row"])
    pb = G.read_sorted(b, "preds.csv", ["id", "kind", "row"])
    m = pa.merge(
        pb, on=["id", "kind", "row"], how="outer", suffixes=("_a", "_b"), indicator=True
    )
    bad_p = set(
        m[(m["_merge"] != "both") | (m["f_a"].astype(str) != m["f_b"].astype(str))][
            "id"
        ]
    )
    sa = G.read_sorted(a, "steps.csv", ["id", "epoch", "step"])
    sb = G.read_sorted(b, "steps.csv", ["id", "epoch", "step"])
    ms = sa.merge(
        sb,
        on=["id", "epoch", "step"],
        how="outer",
        suffixes=("_a", "_b"),
        indicator=True,
    )
    dd = ms[
        (ms["_merge"] != "both")
        | (ms["loss_a"].astype(str) != ms["loss_b"].astype(str))
        | (ms["grad_norm_a"].astype(str) != ms["grad_norm_b"].astype(str))
    ]
    bad_s = set(dd["id"])
    from_first = sorted(set(dd[(dd["epoch"] == "1") & (dd["step"] == "0")]["id"]))
    ja = pd.read_csv(a / "jobs_out.csv")
    return dict(
        jobs=len(ja),
        preds=len(pa),
        steps=len(sa),
        differing_jobs=sorted(bad_p | bad_s),
        differing_from_first_step=from_first,
    )


def trainrows_check(run: Path) -> pd.DataFrame:
    """fc_train jobs (save_params=1): kind "train" forecasts vs a float64 forward of the saved final weights."""
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import reference as RF
    from src.models.window_mask import window_keep

    jo = pd.read_csv(run / "jobs_out.csv")
    pr = pd.read_csv(run / "preds.csv")
    rows = []
    for _, r in jo.iterrows():
        jid, design = (
            r["id"],
            ("all_features" if "_all_features_" in r["id"] else "baseline"),
        )
        X32 = np.fromfile(WORK / "data" / design / "X.f32", "<f4")
        X64 = np.fromfile(WORK / "data" / design / "X.f64", "<f8")
        p = int((WORK / "data" / design / "meta.txt").read_text().split()[1])
        X32, X64 = X32.reshape(-1, p), X64.reshape(-1, p)
        w0, w1, L, H, ly = (
            int(r["w0"]),
            int(r["w1"]),
            int(r["L"]),
            int(r["H"]),
            int(r["layers"]),
        )
        keep = window_keep(X64[w0:w1])
        Z = np.concatenate(
            [X32[:, keep].astype(np.float64), np.ones((len(X32), 1))], axis=1
        )
        tr = pr[(pr["id"] == jid) & (pr["kind"] == "train")]
        out = dict(id=jid, n_train=len(tr), N=int(r["N"]))
        pf = run / "params" / f"{jid}.f32"
        if len(tr) and pf.is_file():
            w = RF.unpack(
                np.fromfile(pf, "<f4").astype(np.float64), len(keep) + 1, H, ly
            )
            rr = tr["row"].to_numpy()
            x = np.stack([Z[k - L + 1 : k + 1] for k in rr])
            dec, _ = RF.forward(w, x)
            f = float(r["mu_y"]) + float(r["sd_y"]) * dec[:, -1, 0]
            out.update(
                rows_ok=bool(
                    rr[0] == w0 + L - 1
                    and rr[-1] == w1 - 1
                    and len(rr) == w1 - w0 - L + 1
                ),
                max_abs_diff_vs_f64=float(np.max(np.abs(f - tr["f"].to_numpy()))),
                max_rel_diff_vs_f64=float(
                    np.max(np.abs(f - tr["f"].to_numpy())) / float(r["sd_y"])
                ),
            )
        fc = pr[(pr["id"] == jid) & (pr["kind"] == "fc")].sort_values("row")
        fcp = pr[(pr["id"] == jid) & (pr["kind"] == "fc_trainpath")].sort_values("row")
        if len(fcp):
            out.update(
                fc_rows=len(fc),
                fc_vs_trainpath_max_abs=float(
                    np.max(np.abs(fc["f"].to_numpy() - fcp["f"].to_numpy()))
                ),
                fc_vs_trainpath_identical=bool(
                    np.array_equal(fc["f"].to_numpy(), fcp["f"].to_numpy())
                ),
            )
        rows.append(out)
    return pd.DataFrame(rows)


def gate2(run: Path) -> None:
    out = {}
    for part in ("tune", "refit"):
        for a, b in (
            ("old_A", "old_B"),
            ("new_A", "new_B"),
            ("new_A", "new_wave"),
            ("old_A", "new_A"),
            ("old_B", "new_A"),
        ):
            out[f"{part}:{a}_vs_{b}"] = diff_jobs(
                run / f"{a}_{part}", run / f"{b}_{part}"
            )
    for v in ("pool", "wave"):
        t = trainrows_check(run / f"fct_{v}")
        out[f"fctrain_{v}"] = t.to_dict("records")
    out["fct_pool_vs_wave"] = diff_jobs(run / "fct_pool", run / "fct_wave")
    # fc_train=1 vs 0 on the same job: identical training and forecasts
    pr = pd.read_csv(run / "fct_pool" / "preds.csv")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pool_gate_report as G

    st = G.read_sorted(run / "fct_pool", "steps.csv", ["id", "epoch", "step"])
    sw = {}
    for d in DESIGNS:
        on, off = f"of_{d}_H64_Ly2_L5_lr0.03_gate", f"of_{d}_H64_Ly2_L5_lr0.03_gate_off"
        a = pr[(pr["id"] == on) & (pr["kind"] == "fc")]["f"].astype(str).tolist()
        b = pr[(pr["id"] == off) & (pr["kind"] == "fc")]["f"].astype(str).tolist()
        sa = (
            st[st["id"] == on][["epoch", "step", "loss", "grad_norm"]]
            .astype(str)
            .to_numpy()
            .tolist()
        )
        sb = (
            st[st["id"] == off][["epoch", "step", "loss", "grad_norm"]]
            .astype(str)
            .to_numpy()
            .tolist()
        )
        sw[d] = dict(
            fc_identical=a == b and len(a) == 10,
            steps_identical=sa == sb and len(sa) > 0,
            off_has_no_train_rows=int(((pr["id"] == off) & (pr["kind"] != "fc")).sum())
            == 0,
        )
    out["fc_train_on_vs_off"] = sw
    (RES / "overfit_r1" / "gate2.json").write_bytes(
        json.dumps(out, indent=1, default=str).encode()
    )
    print(json.dumps(out, indent=1, default=str)[:6000])


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if sys.argv[1] == "make" and sys.argv[2] == "r1":
        make_r1()
    elif sys.argv[1] == "report":
        report(sys.argv[2], [Path(p) for p in sys.argv[3:]])
    elif sys.argv[1] == "gate2":
        gate2(Path(sys.argv[2]))
