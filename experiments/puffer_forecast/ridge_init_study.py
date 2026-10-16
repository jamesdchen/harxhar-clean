"""Variant (b) (user, 2026-10-15): the linear shortcut initialised at the per-window closed-form ridge and fine-tuned
under its L2 penalty; the zero-initialised MinGRU branch and wd_enc on its encoder as in native_g8 (no trainer change:
the init enters through the job key lin_init).

The init, in the trainer's own parametrisation (lin_study.py): x = the job's kept columns (window_keep on the job's
window [w0, w1), as the trainer) and the constant column; z = (y - mu_y) / sd_y with the job's training-row mu_y /
sd_y; v_init = argmin mean(x.v - z)^2 + 1/2 lam sum pf_c v_c^2 (pf: backbone 0.01, exogenous 1, intercept 0) with
lam = wd_lin / sd_y^2 -- the stationary point of v's own objective while the branch is 0 (identity link; under
softplus link(c + s x.v) ~ mu + sd x.v near 0).  Tune jobs: the job's fit rows only (row0 .. W - val_tail - embargo:
never the embargo or validation rows); refits: the job's training rows (the 2,000-row window minus its first L - 1
rows, which carry no full sequence).  The stack-alpha inits (sanity walk): lam = 2 alpha pf_stack / (n pf).

    python experiments/puffer_forecast/ridge_init_study.py inits          # data/lin_init/<design>/*.f32, index.csv
    python experiments/puffer_forecast/ridge_init_study.py check          # init forecasts vs the stack's ridge
    python experiments/puffer_forecast/ridge_init_study.py calib-inputs   # _work/rinit_calib/
    python experiments/puffer_forecast/ridge_init_study.py calib RUN
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lin_study as LS  # noqa: E402
import wd_enc_study as E  # noqa: E402

WORK, RES, DATA = E.WORK, E.RES, E.DATA
SETS = E.SETS
INIT = DATA / "lin_init"
OUT = RES / "ridge_init_study"
WLS = (0.02, 0.2, 2.0)
LS_CFG = (5, 20)  # the grid's sequence lengths
VAL, EMB, W = 125, 25, 2000
# the stack's ridge alpha in each of its 250-row blocks (feature_mcs models_ridge alphas_blk)
ALPHA_BLK = {"all_features": [1000.0] * 6, "baseline": [0.01, 0.01, 0.1, 0.1, 0.1, 0.1]}


class Design:
    def __init__(self, d: str):
        n, p = map(int, (DATA / d / "meta.txt").read_text().split())
        self.d = d
        self.X = np.fromfile(DATA / d / "X.f32", "<f4").reshape(n, p)
        self.X64 = np.fromfile(DATA / d / "X.f64", "<f8").reshape(n, p)
        self.y = np.fromfile(DATA / d / "y.f32", "<f4").astype(np.float64)
        self.pf = E.design_pf(d)


def keep_of(D: Design, w0: int, w1: int) -> np.ndarray:
    from src.models.window_mask import window_keep

    return window_keep(D.X64[w0:w1]).astype(np.int64)


def gram(D: Design, keep: np.ndarray, r0: int, r1: int):
    """X'X / n, X'z / n and the trainer's mu_y / sd_y (float32) of the rows [r0, r1)."""
    rows = np.arange(r0, r1)
    Xk = np.hstack([D.X[rows][:, keep].astype(np.float64), np.ones((len(rows), 1))])
    yy = D.y[rows]
    ym = float(yy.mean())
    mu = float(np.float32(ym))
    sd = float(np.float32(np.sqrt(np.mean((yy - ym) ** 2))))
    z = (yy - mu) / sd
    n = len(rows)
    return Xk.T @ Xk / n, Xk.T @ z / n, mu, sd, n


def solve(A, b, pf, lam):
    return np.linalg.solve(A + 0.5 * lam * np.diag(pf), b)


def stack_lam(d: str, alpha: float, n: int) -> float:
    return 2.0 * alpha / n * (1.0 if d == "all_features" else 100.0)


def inits() -> None:
    """Every init the walks and checks read: tune windows (fit rows) and refit origins (training rows), L {5, 20} x
    wd_lin {0.02, 0.2, 2}; and the stack-alpha inits at L = 1 (the stack's own 2,000 rows) for the sanity walk."""
    import feature_mcs_runner as FR  # noqa: F401  (sys.path)

    bl = E.blocks()
    origins = sorted({o for b in bl for o in b["origins"]} | {2250})
    idx = []
    for d in SETS:
        D = Design(d)
        od = INIT / d
        od.mkdir(parents=True, exist_ok=True)

        def put(name: str, v: np.ndarray, **meta) -> None:
            v.astype("<f4").tofile(od / name)
            idx.append(dict(design=d, file=f"{d}/{name}", P=len(v), **meta))

        for b in bl:
            w0, w1 = b["tune_w0"], b["tune_w1"]
            keep = keep_of(D, w0, w1)
            pf = np.append(D.pf[keep], 0.0)
            for L in (1,) + LS_CFG:
                A, bb, mu, sd, n = gram(D, keep, w0 + L - 1, w1 - VAL - EMB)
                if L == 1:
                    lam = stack_lam(d, ALPHA_BLK[d][(w1 - W) // 250], n)
                    put(
                        f"tune_b{b['k']}_L1_stack.f32",
                        solve(A, bb, pf, lam),
                        kind="tune",
                        k=b["k"],
                        L=1,
                        wl="stack",
                        mu=mu,
                        sd=sd,
                        n=n,
                        lam=lam,
                    )
                    continue
                for wl in WLS:
                    lam = wl / sd**2
                    put(
                        f"tune_b{b['k']}_L{L}_wl{wl:g}.f32",
                        solve(A, bb, pf, lam),
                        kind="tune",
                        k=b["k"],
                        L=L,
                        wl=f"{wl:g}",
                        mu=mu,
                        sd=sd,
                        n=n,
                        lam=lam,
                    )
        for o in origins:
            w0, w1 = o - W, o
            keep = keep_of(D, w0, w1)
            pf = np.append(D.pf[keep], 0.0)
            xf = np.append(D.X[o][keep].astype(np.float64), 1.0)
            for L in (1,) + LS_CFG:
                A, bb, mu, sd, n = gram(D, keep, w0 + L - 1, w1)
                if L == 1:
                    alpha = ALPHA_BLK[d][(o - W) // 250]
                    lam = stack_lam(d, alpha, n)
                    v = solve(A, bb, pf, lam)
                    put(
                        f"r{o}_L1_stack.f32",
                        v,
                        kind="refit",
                        origin=o,
                        L=1,
                        wl="stack",
                        mu=mu,
                        sd=sd,
                        n=n,
                        lam=lam,
                        alpha=alpha,
                        fc_identity=mu
                        + sd * float(xf @ v.astype(np.float32).astype(np.float64)),
                    )
                    continue
                for wl in WLS:
                    lam = wl / sd**2
                    v = solve(A, bb, pf, lam)
                    put(
                        f"r{o}_L{L}_wl{wl:g}.f32",
                        v,
                        kind="refit",
                        origin=o,
                        L=L,
                        wl=f"{wl:g}",
                        mu=mu,
                        sd=sd,
                        n=n,
                        lam=lam,
                        fc_identity=mu
                        + sd * float(xf @ v.astype(np.float32).astype(np.float64)),
                    )
        print(d, "done", len(idx))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(idx).to_csv(OUT / "init_index.csv", index=False)


def check() -> pd.DataFrame:
    """The L = 1 stack-alpha inits' forecasts (identity: mu + sd x.v, v as stored in fp32) against the stack's ridge
    forecast at the stack's block starts (where the stack's mask is the window's own); keep vs the stack's mask and vs
    the trainer's saved keep (g8 walk, first origins)."""
    import feature_mcs_report as FMR

    ix = pd.read_csv(OUT / "init_index.csv")
    fcs = pd.read_parquet(
        FMR.OUT / "forecasts.parquet", columns=["est", "model", "row", "pred_adj"]
    )
    rows = []
    for d in SETS:
        D = Design(d)
        st = fcs[(fcs["est"] == "ridge") & (fcs["model"] == d)].set_index("row")[
            "pred_adj"
        ]
        for o in (2250, 2500, 2750, 3000, 3250):
            r = ix[
                (ix["design"] == d)
                & (ix["kind"] == "refit")
                & (ix["origin"] == o)
                & (ix["L"] == 1)
            ].iloc[0]
            keep = keep_of(D, o - W, o)
            _, mk, _ = E.arms_structure(d, o - W, o)
            rows.append(
                dict(
                    design=d,
                    origin=o,
                    alpha=r["alpha"],
                    P=int(r["P"]),
                    init_fc=r["fc_identity"],
                    stack_ridge_fc=float(st.loc[o - W]),
                    abs_diff=abs(r["fc_identity"] - float(st.loc[o - W])),
                    keep_eq_stack_mask=bool(
                        np.array_equal(keep, np.flatnonzero(~mk[:-1].astype(bool)))
                    ),
                )
            )
        # the trainer's own keep (g8 B walk, each block's first origin)
        run = RES / "lin_study" / "runs" / "walk_lingrid_13145622"
        for b in E.blocks():
            o = b["origins"][0]
            f = (
                run
                / f"block{b['k']}"
                / "refit"
                / "params"
                / f"refit_{d}_r{o}_s20261012.keep.i32"
            )
            tk = np.fromfile(f, "<i4")
            rows.append(
                dict(
                    design=d,
                    origin=o,
                    keep_eq_trainer=bool(np.array_equal(keep_of(D, o - W, o), tk)),
                    P=len(tk) + 1,
                )
            )
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "init_check.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.to_string(index=False))
    return t


def calib_inputs() -> None:
    d0 = WORK / "rinit_calib"
    bl = E.blocks()[0]
    t0, t1, o = bl["tune_w0"], bl["tune_w1"], bl["origins"][0]
    common = (
        "layers=1 val_tail=125 embargo=25 patience=20 momentum=0.95 max_grad_norm=1.0 anneal=1 min_lr_ratio=0.0 "
        "mask=1 xstd=0 loss=mse lin=1 dec_zero=1 max_epochs=200 batch=128 fc_n=1"
    )
    chk, val, ref, tune = [], [], [], []
    for d in SETS:
        # the init's forecast through the trainer (identity, nothing trained) at the stack's block starts
        for oo in (2250, 2500, 2750, 3000, 3250):
            chk.append(
                f"id=chk_{d}_r{oo} data=data/{d} mode=refit w0={oo - W} w1={oo} L=1 H=16 lr=0 seed=1 "
                f"epochs=1 link=identity lr_lin=0 lin_init=data/lin_init/{d}/r{oo}_L1_stack.f32 {common}"
            )
        # the init's validation loss (nothing trained), softplus and identity
        for wl in WLS:
            for lk in ("softplus", "identity"):
                val.append(
                    f"id=val_{d}_wl{wl:g}_{lk} data=data/{d} mode=tune w0={t0} w1={t1} L=5 H=16 lr=0 seed=1 "
                    f"link={lk} lr_lin=0 wd_lin={wl:g} lin_init=data/lin_init/{d}/tune_b0_L5_wl{wl:g}.f32 "
                    f"{common}"
                )
        for L, H, lr in ((20, 16, 0.03), (5, 64, 0.01)):
            for we in (0, 15):
                for wl in WLS:
                    for ll in (0.0003, 0.001, 0.01):
                        cfg = f"L{L}_H{H}_lr{lr:g}_we{we:g}_wl{wl:g}_ll{ll:g}"
                        x = (
                            f"data=data/{d} L={L} H={H} lr={lr:g} wd_enc={we:g} wd_lin={wl:g} lr_lin={ll:g} "
                            f"link=softplus {common}"
                        )
                        for s in range(5):
                            seed = 20261012 + s
                            tune.append(
                                f"id=dt_{d}_{cfg}_s{seed} mode=tune w0={t0} w1={t1} seed={seed} {x} "
                                f"lin_init=data/lin_init/{d}/tune_b0_L{L}_wl{wl:g}.f32"
                            )
                            for ep in (5, 10, 20, 40):
                                ref.append(
                                    f"id=dr_{d}_{cfg}_e{ep}_s{seed} mode=refit w0={o - W} w1={o} seed={seed} "
                                    f"epochs={ep} save_params=1 {x} "
                                    f"lin_init=data/lin_init/{d}/r{o}_L{L}_wl{wl:g}.f32"
                                )
    for name, lines in (
        ("check.txt", chk),
        ("val.txt", val),
        ("drift_refit.txt", ref),
        ("drift_tune.txt", tune),
    ):
        LS.write(d0, name, lines)
    print(len(chk), len(val), len(ref), len(tune))


def _cfg_parts(jid: str) -> dict:
    import re

    m = re.match(
        r"^d[rt]_(baseline|all_features)_L(\d+)_H(\d+)_lr([0-9.]+)_we([0-9.e+-]+)_wl([0-9.e+-]+)_ll([0-9.e+-]+)"
        r"(?:_e(\d+))?_s(\d+)$",
        jid,
    )
    assert m, jid
    d, L, H, lr, we, wl, ll, ep, seed = m.groups()
    return dict(
        design=d,
        L=int(L),
        H=int(H),
        lr=float(lr),
        we=float(we),
        wl=wl,
        ll=float(ll),
        ep=int(ep) if ep else None,
        seed=int(seed),
    )


def calib(run: Path) -> None:  # noqa: C901 -- one report
    import feature_mcs_report as FMR

    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    ix = pd.read_csv(OUT / "init_index.csv")
    fcs = pd.read_parquet(
        FMR.OUT / "forecasts.parquet", columns=["est", "model", "row", "pred_adj"]
    )
    b0 = E.blocks()[0]
    # ---- the init's forecast through the trainer
    if (run / "check").is_dir() and (run / "val").is_dir():
        pr = pd.read_csv(run / "check" / "preds.csv")
        pr = pr[pr["kind"] == "fc"]
        ck = []
        for _, r in pr.iterrows():
            d = "all_features" if "_all_features_" in r["id"] else "baseline"
            o = int(r["row"])
            st = fcs[
                (fcs["est"] == "ridge") & (fcs["model"] == d) & (fcs["row"] == o - W)
            ]["pred_adj"].iloc[0]
            loc = ix[
                (ix["design"] == d)
                & (ix["kind"] == "refit")
                & (ix["origin"] == o)
                & (ix["L"] == 1)
            ].iloc[0]
            ck.append(
                dict(
                    design=d,
                    origin=o,
                    trainer_fc=r["f"],
                    init_fc_local=loc["fc_identity"],
                    stack_ridge_fc=st,
                    trainer_vs_stack=abs(r["f"] - st),
                )
            )
        ck = pd.DataFrame(ck)
        ck.to_csv(OUT / "trainer_init_check.csv", index=False)
        print(ck.to_string(index=False))
        # ---- the init's validation loss, nothing trained (softplus vs identity) and the stack's rolling ridge
        ep = pd.read_csv(run / "val" / "epochs.csv")
        vr = []
        v0r, v1r = b0["tune_w1"] - VAL, b0["tune_w1"]
        for jid, g in ep.groupby("id"):
            d = "all_features" if "_all_features_" in jid else "baseline"
            y = Design(d).y[v0r:v1r]
            st = fcs[(fcs["est"] == "ridge") & (fcs["model"] == d)].set_index("row")[
                "pred_adj"
            ]
            sv = st.reindex(np.arange(v0r - W, v1r - W)).to_numpy()
            vr.append(
                dict(
                    job=jid,
                    val_mse_first_epoch=float(
                        g.sort_values("epoch")["val_mse"].iloc[0]
                    ),
                    val_mse_constant=bool(g["val_mse"].nunique() == 1),
                    stack_rolling_ridge_val_mse=float(np.mean((sv - y) ** 2)),
                )
            )
        vr = pd.DataFrame(vr)
        vr.to_csv(OUT / "init_val_loss.csv", index=False)
        print(vr.to_string(index=False))
    # ---- drift from the ridge init: block-0 refits after 5 .. 40 epochs
    jo = pd.read_csv(run / "drift_refit" / "jobs_out.csv")
    cl = E.day_classes()
    ext = cl[cl["day_class"] == "extreme_inputs"]["row"].to_numpy()
    oth = cl[cl["day_class"] == "other"]["row"].to_numpy()
    o = b0["origins"][0]
    rows = []
    cache: dict = {}
    for jid in jo["id"]:
        c = _cfg_parts(jid)
        d = c["design"]
        if d not in cache:
            D = Design(d)
            keep = keep_of(D, o - W, o)
            cache[d] = (D, keep)
        D, keep = cache[d]
        v = LS._f32(run / "drift_refit" / "params" / f"{jid}.lin.f32")
        v0 = LS._f32(INIT / d / f"r{o}_L{c['L']}_wl{c['wl']}.f32")
        key = (d, c["L"])
        if key not in cache:
            rr = np.arange(o - W + c["L"] - 1, o)
            cache[key] = np.hstack(
                [D.X[rr][:, keep].astype(np.float64), np.ones((len(rr), 1))]
            )
        X = cache[key]
        Xe = np.hstack([D.X[ext][:, keep].astype(np.float64), np.ones((len(ext), 1))])
        Xo = np.hstack([D.X[oth][:, keep].astype(np.float64), np.ones((len(oth), 1))])
        p0 = X @ v0
        rows.append(
            dict(
                **c,
                drift=float(
                    np.linalg.norm(X @ (v - v0)) / np.linalg.norm(p0 - p0.mean())
                ),
                **{f"ext_{i}": float(z) for i, z in enumerate(Xe @ v)},
                **{f"oth_{i}": float(z) for i, z in enumerate(Xo @ v)},
            )
        )
    t = pd.DataFrame(rows)
    gkey = ["design", "L", "H", "lr", "we", "wl", "ll", "ep"]
    ec = [c for c in t.columns if c.startswith("ext_")]
    oc = [c for c in t.columns if c.startswith("oth_")]
    agg = t.groupby(gkey).agg(drift=("drift", "mean")).reset_index()
    agg["seed_sd_vx_extreme"] = t.groupby(gkey)[ec].std().mean(axis=1).to_numpy()
    agg["seed_sd_vx_other"] = t.groupby(gkey)[oc].std().mean(axis=1).to_numpy()
    # ---- tunes: best epochs and the seed-mean validation MSE of each configuration
    tj = pd.read_csv(run / "drift_tune" / "jobs_out.csv")
    tp = pd.read_csv(run / "drift_tune" / "preds.csv")
    tp = tp[tp["kind"] == "val"]
    tj = pd.concat(
        [
            tj,
            pd.DataFrame([_cfg_parts(x) for x in tj["id"]])[
                ["design", "we", "wl", "ll"]
            ],
        ],
        axis=1,
    )
    tp = tp.merge(tj[["id", "design", "L", "H", "lr", "we", "wl", "ll"]], on="id")
    tv = []
    for k_, g in tp.groupby(["design", "L", "H", "lr", "we", "wl", "ll"]):
        d = k_[0]
        m = g.groupby("row")["f"].mean()
        y = Design(d).y[m.index.to_numpy()]
        be = tj[
            (tj["design"] == d)
            & (tj["L"] == k_[1])
            & (tj["we"] == k_[4])
            & (tj["wl"] == k_[5])
            & (tj["ll"] == k_[6])
        ]["best_epoch"]
        tv.append(
            dict(
                zip(["design", "L", "H", "lr", "we", "wl", "ll"], k_),
                val_mse_seedmean=float(np.mean((m.to_numpy() - y) ** 2)),
                best_epoch_median=float(be.median()),
            )
        )
    tv = pd.DataFrame(tv)
    # drift at the best epoch: log-linear interpolation of drift over 5 .. 40 epochs
    out = []
    for _, r in tv.iterrows():
        a = agg[
            (agg["design"] == r["design"])
            & (agg["L"] == r["L"])
            & (agg["we"] == r["we"])
            & (agg["wl"] == r["wl"])
            & (agg["ll"] == r["ll"])
        ].sort_values("ep")
        e = float(np.clip(r["best_epoch_median"], 5, 40))
        rec = dict(r)
        for col in ("drift", "seed_sd_vx_extreme", "seed_sd_vx_other"):
            rec[f"{col}_at_best"] = float(
                np.interp(np.log(e), np.log(a["ep"].to_numpy()), a[col].to_numpy())
            )
        out.append(rec)
    out = pd.DataFrame(out)
    agg.to_csv(OUT / "drift_by_epoch.csv", index=False)
    out.to_csv(OUT / "drift_at_best_epoch.csv", index=False)
    print(
        agg.pivot_table(
            index=["design", "ll", "we", "wl"], columns="ep", values="drift"
        )
        .round(3)
        .to_string()
    )
    print(out.round(4).to_string(index=False))
    print(
        out.groupby(["design", "ll"])[
            [
                "drift_at_best",
                "seed_sd_vx_extreme_at_best",
                "seed_sd_vx_other_at_best",
                "val_mse_seedmean",
                "best_epoch_median",
            ]
        ]
        .median()
        .round(4)
        .to_string()
    )


def walk_inputs(lls: list[float], ll_fixed: float) -> None:
    """B (rinit): the 12-configuration grid x wd_enc {0, 3, 15} x wd_lin {0.02, 0.2, 2} x lr_lin (lls) in each
    block's tune, both designs, the shortcut starting at its window's ridge (tune: fit rows; refits: the window).
    Sanity (rinit_sanity, softplus and identity): all_features, the stack-alpha ridge init, branch frozen (lr 0),
    lr_lin 0, L 1.  Fixed (rinit_we15): all_features, the 12 configurations at wd_enc 15, wd_lin 0.2, lr_lin ll_fixed."""
    src = WORK / "phase2c"
    m = json.loads((src / "blocks.json").read_text())
    lin = "lin=1 dec_zero=1"

    def man(d: Path, designs: list[str], common: dict, fmt: str) -> None:
        mm = json.loads(json.dumps(m))
        mm["designs"] = designs
        mm["common"].update(common)
        mm["lin_init_refit"] = fmt
        (d / "blocks.json").write_bytes(json.dumps(mm, indent=1).encode())

    def tune_lines(k: int, designs: tuple) -> list[str]:
        return [
            x
            for x in (src / f"tune_block{k}.txt").read_text().splitlines()
            if x.strip() and any(f"data=data/{d} " in x for d in designs)
        ]

    def L_of(x: str) -> int:
        return int(x.split(" L=")[1].split()[0])

    def d_of(x: str) -> str:
        return x.split(" data=data/")[1].split()[0]

    fmt = "data/lin_init/{design}/r{origin}_L{L}_wl{wl}.f32"
    d = WORK / "walk_softplus_rinit"
    for k in range(5):
        out = []
        for x in tune_lines(k, SETS):
            jid = x.split()[0]
            stem, seed = jid.rsplit("_s", 1)
            for we in (0.0, 3.0, 15.0):
                for wl in WLS:
                    for ll in lls:
                        sfx = f"_we{we:g}_wl{wl:g}" + (
                            f"_ll{ll:g}" if len(lls) > 1 else ""
                        )
                        out.append(
                            x.replace(jid, f"{stem}{sfx}_s{seed}", 1)
                            + f" link=softplus {lin} wd_enc={we:g} wd_lin={wl:g} lr_lin={ll:g} "
                            f"lin_init=data/lin_init/{d_of(x)}/tune_b{k}_L{L_of(x)}_wl{wl:g}.f32"
                        )
        LS.write(d, f"tune_block{k}.txt", out)
    man(
        d,
        list(SETS),
        dict(
            link="softplus",
            lin=1,
            dec_zero=1,
            **({"lr_lin": lls[0]} if len(lls) == 1 else {}),
        ),
        fmt,
    )
    for link in ("softplus", "identity"):
        d = WORK / f"walk_{link}_rinit_sanity"
        for k in range(5):
            out = []
            for x in tune_lines(k, ("all_features",)):
                jid = x.split()[0]
                if "_L5_H16_lr0.003_" not in jid:
                    continue
                seed = jid.rsplit("_s", 1)[1]
                y = (
                    x.replace(jid, f"id=tune_all_features_L1_H16_lr0_s{seed}", 1)
                    .replace(" L=5 ", " L=1 ")
                    .replace(" lr=0.003 ", " lr=0 ")
                )
                out.append(
                    f"{y} link={link} {lin} lr_lin=0 lin_init=data/lin_init/all_features/tune_b{k}_L1_stack.f32"
                )
            LS.write(d, f"tune_block{k}.txt", out)
        man(
            d,
            ["all_features"],
            dict(link=link, lin=1, dec_zero=1, lr_lin=0),
            "data/lin_init/{design}/r{origin}_L{L}_stack.f32",
        )
    d = WORK / "walk_softplus_rinit_we15"
    for k in range(5):
        out = [
            f"{x} link=softplus {lin} wd_enc=15 wd_lin=0.2 lr_lin={ll_fixed:g} "
            f"lin_init=data/lin_init/all_features/tune_b{k}_L{L_of(x)}_wl0.2.f32"
            for x in tune_lines(k, ("all_features",))
        ]
        LS.write(d, f"tune_block{k}.txt", out)
    man(
        d,
        ["all_features"],
        dict(
            link="softplus", lin=1, dec_zero=1, wd_enc=15, wd_lin=0.2, lr_lin=ll_fixed
        ),
        fmt,
    )
    print(
        {
            p.name: len((p / "tune_block0.txt").read_text().splitlines())
            for p in sorted(WORK.glob("walk_*rinit*"))
        }
    )


def drift(run: Path, tag: str) -> pd.DataFrame:
    """Each block's first-origin refits (save_params): v against its ridge init in prediction space on the window's
    training rows, and the seed spread of v.x on the extreme-input days' rows (and the other days')."""
    cl = E.day_classes()
    ext = cl[cl["day_class"] == "extreme_inputs"]["row"].to_numpy()
    oth = cl[cl["day_class"] == "other"]["row"].to_numpy()
    out = []
    for b in E.blocks():
        o = b["origins"][0]
        rd = run / f"block{b['k']}" / "refit"
        jo = pd.read_csv(rd / "jobs_out.csv").set_index("id")
        for d in SETS:
            ids = [i for i in jo.index if i.startswith(f"refit_{d}_r{o}_s")]
            if not ids:
                continue
            D = Design(d)
            keep = np.fromfile(rd / "params" / f"{ids[0]}.keep.i32", "<i4")
            L, wl = int(jo.loc[ids[0], "L"]), float(jo.loc[ids[0], "wd_lin"])
            f0 = INIT / d / f"r{o}_L{L}_wl{wl:g}.f32"
            if not f0.is_file():
                f0 = INIT / d / f"r{o}_L{L}_stack.f32"
            v0 = LS._f32(f0)
            rr = np.arange(o - W + L - 1, o)
            X = np.hstack([D.X[rr][:, keep].astype(np.float64), np.ones((len(rr), 1))])
            Xe = np.hstack(
                [D.X[ext][:, keep].astype(np.float64), np.ones((len(ext), 1))]
            )
            Xo = np.hstack(
                [D.X[oth][:, keep].astype(np.float64), np.ones((len(oth), 1))]
            )
            V = np.array([LS._f32(rd / "params" / f"{i}.lin.f32") for i in ids])
            p0 = X @ v0
            dr = [
                float(np.linalg.norm(X @ (v - v0)) / np.linalg.norm(p0 - p0.mean()))
                for v in V
            ]
            out.append(
                dict(
                    walk=tag,
                    block=b["k"],
                    design=d,
                    L=L,
                    wd_lin=wl,
                    lr_lin=float(jo.loc[ids[0], "lr_lin"]),
                    drift_mean=float(np.mean(dr)),
                    drift_max=float(np.max(dr)),
                    seed_sd_vx_extreme=float((Xe @ V.T).std(axis=1, ddof=1).mean()),
                    seed_sd_vx_other=float((Xo @ V.T).std(axis=1, ddof=1).mean()),
                    init_sd_vx_extreme=float((Xe @ v0).std()),
                )
            )
    return pd.DataFrame(out)


def study(walks: dict[str, Path]) -> None:  # noqa: C901 -- one report
    import feature_mcs_report as FMR

    from src.evaluation.diebold_mariano import dm_test

    cl = E.day_classes()["day_class"].to_numpy()
    sc = E._scoring()
    fcs = pd.read_parquet(
        FMR.OUT / "forecasts.parquet", columns=["est", "model", "row", "pred_adj"]
    )
    cands: dict = {}
    for d in SETS:
        cands[(d, "softplus walk of record")] = E.losses(
            E.walk_pred(E.REF_WALK / "forecasts.csv", d)
        )
        cands[(d, "fix #1 B")] = E.losses(
            E.walk_pred(RES / "wd_enc_study" / "score_wegrid" / "forecasts.csv", d)
        )
        cands[(d, "g8 B")] = E.losses(
            E.walk_pred(RES / "lin_study" / "score_lingrid" / "forecasts.csv", d)
        )
        for est in ("ridge", "enet"):
            c = fcs[(fcs["est"] == est) & (fcs["model"] == d)].sort_values("row")
            p = np.full(sc["n"], np.nan)
            p[c["row"].to_numpy()] = c["pred_adj"].to_numpy(float)
            cands[(d, est)] = E.losses(p)
    for tag in walks:
        fc = OUT / f"score_{tag}" / "forecasts.csv"
        for d in pd.read_csv(fc)["design"].unique():
            cands[(d, f"(b) {tag}")] = E.losses(E.walk_pred(fc, d))
    S = sc["S"]
    rows = []
    for (d, name), x in cands.items():
        st = S.stats(x)
        rec = dict(
            design=d,
            model=name,
            qlike=st["qlike"],
            mse_fit=float(np.mean(x["mse_fit"])),
            sharpe_mid=st["sharpe_mid"],
            sharpe_crossed=st["sharpe_crossed"],
        )
        for c in ("month_end", "extreme_inputs", "other"):
            rec[f"ql_{c}"] = float(x["ql"][cl == c].mean())
        rows.append(rec)
    scores = pd.DataFrame(rows)
    con = []

    def add(d, name, rd, rn):
        g, h = cands[(d, name)], cands[(rd, rn)]
        r = dict(design=d, model=name, vs=f"{rn} ({rd})", **E.contrast_row(g, h))
        for c in ("month_end", "extreme_inputs", "other"):
            r[f"dql_{c}"] = float((g["ql"] - h["ql"])[cl == c].mean())
            r[f"p_{c}"] = float(dm_test(g["ql"][cl == c], h["ql"][cl == c])["p"])
        con.append(r)

    for d, name in list(cands):
        if name.startswith("(b)") or name in ("fix #1 B", "g8 B"):
            for rn in ("enet", "ridge", "fix #1 B", "g8 B", "softplus walk of record"):
                if rn != name:
                    add(d, name, d, rn)
            if d == "all_features":
                add(d, name, "baseline", "softplus walk of record")
                if ("baseline", name) in cands:
                    add(d, name, "baseline", name)
    con = pd.DataFrame(con)
    sel = []
    for tag, run in walks.items():
        for b in sorted(run.glob("block*")):
            for d, v in json.loads((b / "selection.json").read_text()).items():
                sel.append(
                    dict(
                        walk=tag,
                        block=int(b.name[5:]),
                        design=d,
                        cfg=v["cfg"],
                        val_mse=v["val_mse"],
                        best_epochs=json.dumps(v["best_epoch"]),
                    )
                )
    sel = pd.DataFrame(sel)
    sens = pd.concat([LS.sensitivities(run, tag) for tag, run in walks.items()])
    spread = pd.DataFrame(
        [r for tag, run in walks.items() for r in E.seed_spread(run, tag)]
    )
    vs = pd.DataFrame(
        [r for tag, run in walks.items() for r in LS.variance_share(run, tag)]
    )
    dr = pd.concat([drift(run, tag) for tag, run in walks.items()])
    for name, t in (
        ("scores", scores),
        ("contrasts", con),
        ("selection", sel),
        ("sensitivities", sens),
        ("seed_spread_2020-03-18", spread),
        ("variance_share", vs),
        ("drift", dr),
    ):
        t.to_csv(OUT / f"{name}.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(scores.round(4).to_string(index=False))
    print(con.round(4).to_string(index=False))
    print(sel.to_string(index=False))
    cols = [c for c in sens.columns if c.startswith("d_")]
    print(sens.groupby(["walk", "design"])[cols].mean().round(3).to_string())
    print(spread.to_string(index=False))
    print(vs.round(3).to_string(index=False))
    print(dr.round(4).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "inits":
        inits()
    elif cmd == "check":
        check()
    elif cmd == "calib-inputs":
        calib_inputs()
    elif cmd == "calib":
        calib(Path(sys.argv[2]))
    elif cmd == "walk-inputs":  # LL1,LL2,.. LL_FIXED
        walk_inputs([float(x) for x in sys.argv[2].split(",")], float(sys.argv[3]))
    elif cmd == "study":
        study({a.split("=")[0]: Path(a.split("=")[1]) for a in sys.argv[2:]})
