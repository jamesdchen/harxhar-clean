"""Pooled ridge init (user, 2026-10-15): the per-bar ridge-init QLIKE recipe on the pooled panel.

native_g9, softplus, lin=1 with lin_init at each window's closed-form ridge, dec_zero=1, wd_enc on the branch's
encoder, loss=qlike, es=qlike, selection on the validation QLIKE (pooled_select.py --criterion qlike).  The pooled
walk's protocol (pooled_walk.py): trailing 24,000-bar windows, 5 blocks, fit / embargo 48 / validation 2,400 at
each block start, monthly refits (each on its trailing 24,000 bars, forecasting every bar to the next refit), 3
seeds, P100.

Penalty structure: the pooled stack's two-block ridge (src/unification.py ARMS["blk2_user"]: blocks backbone
(_backbone_cols, 52 columns) at alpha 1 and exog_all (1,092 columns) at alpha 100, USER_ALPHAS; per-block alphas by
column scaling under a global ridge, intercept unpenalised by centering, analysis/wf.walk_forward): backbone pf 0.01,
exogenous pf 1 (= 1 / 100 of the exogenous penalty), which enc_pf.f64 encodes for both pooled sets (the baseline set
is the 52 backbone columns: 0.01 throughout).  The ridge init solves, in the trainer's parametrisation,
argmin mean(x.v - z)^2 + 1/2 lam sum pf_c v_c^2 with lam = wd_lin / sd_y^2 (the exact gate-8 mapping:
alpha_exog = n wd_lin / (2 sd_y^2)) on the job's kept columns (window_keep on its window) and constant column;
tune jobs on their fit rows, refits on their training rows.

    python experiments/puffer_forecast/pooled_rinit.py pf | inits | check | calib-inputs | walk-inputs LL[,LL]
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

WORK, RES = E.WORK, E.RES
REPO = E.REPO
POOL = WORK / "pooled_walk"
INIT = WORK / "pooled_lin_init"  # shipped as data/pooled_lin_init/<design>/
OUT = RES / "pooled_rinit_study"
SETS = ("baseline", "all_features")
W, VAL, EMB = 24_000, 2_400, 48
LS_CFG = (20, 48)
WLS = (
    0.03,
    0.3,
    3.0,
)  # ~ alpha_exog 1e3 / 1e4 / 1e5 at sd^2 ~ 0.3, n ~ 22,000 (the tuned arm's pre-test mode 1e4)
CLEAN = REPO.parent / "harxhar-clean" / "results" / "unification_carc"


def info() -> dict:
    return json.loads((POOL / "plan.json").read_text())


class PDesign:
    def __init__(self, d: str):
        n, p = map(int, (POOL / f"pooled_walk_{d}" / "meta.txt").read_text().split())
        self.d, self.n, self.p = d, n, p
        self.X = np.memmap(
            POOL / f"pooled_walk_{d}" / "X.f32", "<f4", "r", shape=(n, p)
        )
        self.X64 = np.memmap(
            POOL / f"pooled_walk_{d}" / "X.f64", "<f8", "r", shape=(n, p)
        )
        self.y = np.fromfile(POOL / f"pooled_walk_{d}" / "y.f32", "<f4").astype(
            np.float64
        )
        f = POOL / f"pooled_walk_{d}" / "enc_pf.f64"  # (pf() writes it)
        self.pf = np.fromfile(f, "<f8") if f.is_file() else np.ones(p)


def pf() -> None:
    """enc_pf.f64 of both pooled sets from the stack's own block assignment (blk2_user: backbone 1 / exog 100)."""
    sys.path.insert(0, str(REPO))
    import prep_pooled as PP

    import src.unification as U

    z = np.load(PP.CACHE, allow_pickle=True)
    names = [str(s) for s in z["names"]]
    bb = U._backbone_cols(names)
    ex = U._exog_all_cols(names)
    assert len(set(bb) | set(ex)) == len(names) and not set(bb) & set(ex)
    assert U.USER_ALPHAS["backbone"] / U.USER_ALPHAS["exog"] == 0.01
    pf_all = np.ones(len(names))
    pf_all[bb] = U.USER_ALPHAS["backbone"] / U.USER_ALPHAS["exog"]
    for d, v in (("all_features", pf_all), ("baseline", pf_all[bb])):
        n, p = map(int, (POOL / f"pooled_walk_{d}" / "meta.txt").read_text().split())
        assert len(v) == p, (d, len(v), p)
        v.astype("<f8").tofile(POOL / f"pooled_walk_{d}" / "enc_pf.f64")
        print(d, p, "backbone", int((v < 1).sum()), "exogenous", int((v == 1).sum()))


def keep_of(D: PDesign, w0: int, w1: int) -> np.ndarray:
    from src.models.window_mask import window_keep

    return window_keep(np.asarray(D.X64[w0:w1])).astype(np.int64)


def gram(D: PDesign, keep: np.ndarray, r0: int, r1: int):
    Xk = np.empty((r1 - r0, len(keep) + 1))
    Xk[:, :-1] = np.asarray(D.X[r0:r1])[:, keep]
    Xk[:, -1] = 1.0
    yy = D.y[r0:r1]
    ym = float(yy.mean())
    mu = float(np.float32(ym))
    sd = float(np.float32(np.sqrt(np.mean((yy - ym) ** 2))))
    z = (yy - mu) / sd
    n = r1 - r0
    return Xk.T @ Xk / n, Xk.T @ z / n, mu, sd, n


def solve(A, b, pf, lam):
    return np.linalg.solve(A + 0.5 * lam * np.diag(pf), b)


def inits() -> None:
    """Tune windows (fit rows) and refit windows (training rows), L {20, 48} x wd_lin; plus L = 1 inits at the stack's
    blk2_user alphas (all_features: exog 100 / backbone 1; baseline: alpha_backbone 1e-8, ~ a0's min-norm OLS) at each
    block's first origin for the check."""
    pl = info()
    sys.path.insert(0, str(REPO))
    idx = []
    for d in SETS:
        D = PDesign(d)
        od = INIT / d
        od.mkdir(parents=True, exist_ok=True)

        def put(name, v, **m):
            np.asarray(v, "<f4").tofile(od / name)
            idx.append(dict(design=d, file=f"{d}/{name}", P=len(v), **m))

        for b in pl["blocks"]:
            o = b["refit_rows_slice"][0]
            keep = keep_of(D, o - W, o)
            pfk = np.append(D.pf[keep], 0.0)
            for L in LS_CFG:
                A, bb, mu, sd, n = gram(D, keep, o - W + L - 1, o - VAL - EMB)
                for wl in WLS:
                    lam = wl / sd**2
                    put(
                        f"tune_b{b['k']}_L{L}_wl{wl:g}.f32",
                        solve(A, bb, pfk, lam),
                        kind="tune",
                        k=b["k"],
                        L=L,
                        wl=f"{wl:g}",
                        mu=mu,
                        sd=sd,
                        n=n,
                        alpha_exog=n * wl / (2 * sd**2),
                    )
            for i, o in enumerate(b["refit_rows_slice"]):
                keep = keep_of(D, o - W, o)
                pfk = np.append(D.pf[keep], 0.0)
                xf = np.append(np.asarray(D.X[o], np.float64)[keep], 1.0)
                for L in (1,) + LS_CFG if i == 0 else LS_CFG:
                    A, bb, mu, sd, n = gram(D, keep, o - W + L - 1, o)
                    if L == 1:
                        alpha = (
                            100.0 if d == "all_features" else 1e-6
                        )  # (baseline: alpha_bb = 1e-8)
                        lam = 2.0 * alpha / n
                        v = solve(A, bb, pfk, lam)
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
                            alpha_exog=alpha,
                            fc_identity=mu
                            + sd * float(xf @ v.astype(np.float32).astype(float)),
                        )
                        continue
                    for wl in WLS:
                        lam = wl / sd**2
                        v = solve(A, bb, pfk, lam)
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
                            alpha_exog=n * wl / (2 * sd**2),
                            fc_identity=mu
                            + sd * float(xf @ v.astype(np.float32).astype(float)),
                        )
            print(d, "block", b["k"], "done", len(idx), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(idx).to_csv(OUT / "init_index.csv", index=False)


def stack_yhat(arm: str, panel_rows: np.ndarray) -> pd.Series:
    out = []
    for f in sorted((CLEAN / arm).glob("chunk_*.npz")):
        z = np.load(f, allow_pickle=True)
        rid = z["row_id"]
        if rid[-1] < panel_rows.min() or rid[0] > panel_rows.max():
            continue
        out.append(pd.Series(z["yhat"], index=rid))
    s = pd.concat(out)
    return s.reindex(panel_rows)


def multiplicity(D: PDesign, w0: int, w1: int, keep: np.ndarray) -> np.ndarray:
    """Copies of each kept column in the window (byte-identical non-constant columns): the stack's ridge keeps every
    copy (the penalty of a group of k copies acts as alpha / k on their sum), the trainer's window mask keeps one."""
    cols = np.ascontiguousarray(np.asarray(D.X64[w0:w1]).T)
    const = (cols == cols[:, :1]).all(axis=1)
    groups: dict[bytes, int] = {}
    first: dict[bytes, int] = {}
    for j in range(cols.shape[0]):
        if const[j]:
            continue
        k = cols[j].tobytes()
        groups[k] = groups.get(k, 0) + 1
        first.setdefault(k, j)
    mult = {j: groups[k] for k, j in first.items()}
    return np.array([mult[j] for j in keep])


def check() -> pd.DataFrame:
    """At each block's first origin, the L = 1 stack-alpha init's forecast against the stack: all_features vs
    blk2_user (backbone 1 / exog 100, 24,000-bar window, per-bar refit), baseline vs a0_ols_har (min-norm OLS); the
    init in the trainer's parametrisation (one column per byte-identical group, its pf) and, to isolate the
    mapping, the same solve with each kept column's pf divided by its number of copies (the stack's own penalty)."""
    ix = pd.read_csv(OUT / "init_index.csv")
    pl = info()
    r0 = pl["panel_slice"][0]
    rows = []
    for d, arm in (("all_features", "blk2_user"), ("baseline", "a0_ols_har")):
        D = PDesign(d)
        t = ix[(ix["design"] == d) & (ix["L"] == 1)]
        st = stack_yhat(arm, t["origin"].to_numpy() + r0)
        for _, r in t.iterrows():
            o = int(r["origin"])
            keep = keep_of(D, o - W, o)
            mult = multiplicity(D, o - W, o, keep)
            A, bb, mu, sd, n = gram(D, keep, o - W, o)
            xf = np.append(np.asarray(D.X[o], np.float64)[keep], 1.0)
            pfm = np.append(D.pf[keep] / mult, 0.0)
            vm = solve(A, bb, pfm, 2.0 * float(r["alpha_exog"]) / n)
            s = float(st.loc[o + r0])
            rows.append(
                dict(
                    design=d,
                    stack_arm=arm,
                    origin_slice=o,
                    panel_row=o + r0,
                    P=int(r["P"]),
                    dup_groups=int((mult > 1).sum()),
                    max_copies=int(mult.max()),
                    init_fc=float(r["fc_identity"]),
                    stack_fc=s,
                    abs_diff=abs(float(r["fc_identity"]) - s),
                    init_fc_pf_over_copies=mu + sd * float(xf @ vm),
                    abs_diff_pf_over_copies=abs(mu + sd * float(xf @ vm) - s),
                )
            )
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "init_check.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.to_string(index=False))
    return t


def _tune_src(k: int) -> list[str]:
    return [
        x
        for x in (WORK / "pwalk_wd0" / f"tune_block{k}.txt").read_text().splitlines()
        if x.strip()
    ]


LIN = "es=qlike lin=1 dec_zero=1"


def calib_inputs(
    lls: tuple[float, ...] = (0.0001, 0.0003, 0.001), sub: str = "pcal"
) -> None:
    """Block 0, the shortcut's drift under QLIKE: H64 L20 lr0.003, wd_enc 3, wd_lin x lr_lin {1e-4, 3e-4, 1e-3},
    3 seeds; tunes (es=qlike) and refits at the block's first origin after 3 / 10 epochs (save_params); per design."""
    pl = info()
    o = pl["blocks"][0]["refit_rows_slice"][0]
    for d in SETS:
        tune, ref = [], []
        for x in _tune_src(0):
            jid = x.split()[0]
            if f"ptune_{d}_L20_H64_lr0.003_s" not in jid:
                continue
            seed = jid.rsplit("_s", 1)[1]
            for wl in WLS:
                for ll in lls:
                    sfx = f"_we3_wl{wl:g}_ll{ll:g}"
                    lin = (
                        f"wd_enc=3 wd_lin={wl:g} lr_lin={ll:g} "
                        f"lin_init=data/pooled_lin_init/{d}/tune_b0_L20_wl{wl:g}.f32"
                    )
                    tune.append(
                        x.replace(
                            jid, f"id=pct_{d}_L20_H64_lr0.003{sfx}_s{seed}", 1
                        ).replace(" loss=mse", " loss=qlike")
                        + f" {LIN} {lin}"
                    )
                    for ep in (3, 10):
                        y = (
                            x.replace(
                                jid, f"id=pcr_{d}_L20_H64_lr0.003{sfx}_e{ep}_s{seed}", 1
                            )
                            .replace(" mode=tune ", " mode=refit ")
                            .replace(" loss=mse", " loss=qlike")
                            .replace(" fc_n=1 ", " fc_n=1 ")
                        )
                        y = y.replace(f" w0={o - W} w1={o} ", f" w0={o - W} w1={o} ")
                        lin_r = lin.replace(
                            f"tune_b0_L20_wl{wl:g}", f"r{o}_L20_wl{wl:g}"
                        )
                        ref.append(
                            f"{y} epochs={ep} save_params=1 lin=1 dec_zero=1 {lin_r}"
                        )
        LS.write(WORK / sub, f"tune_{d}.txt", tune)
        LS.write(WORK / sub, f"refit_{d}.txt", ref)
        print(d, len(tune), len(ref))
    # the shard-merge test: a small baseline tune (2 configurations x 3 seeds), whole and as two shards
    mt: dict[int, list[str]] = {0: [], 1: []}
    for x in _tune_src(0):
        jid = x.split()[0]
        for i, cfg in enumerate(("L20_H64_lr0.003", "L48_H64_lr0.001")):
            if f"ptune_baseline_{cfg}_s" in jid:
                seed = jid.rsplit("_s", 1)[1]
                L = cfg[1:3]
                mt[i].append(
                    x.replace(
                        jid, f"id=ptune_baseline_{cfg}_we3_wl0.3_ll0.0003_s{seed}", 1
                    ).replace(" loss=mse", " loss=qlike")
                    + f" {LIN} wd_enc=3 wd_lin=0.3 lr_lin=0.0003 "
                    f"lin_init=data/pooled_lin_init/baseline/tune_b0_L{L}_wl0.3.f32"
                )
    LS.write(WORK / "pcal", "mtest_all.txt", mt[0] + mt[1])
    LS.write(WORK / "pcal", "mtest_s0.txt", mt[0])
    LS.write(WORK / "pcal", "mtest_s1.txt", mt[1])
    m = json.loads((WORK / "pwalk_wd0" / "plan.json").read_text())
    m["designs"] = ["baseline"]
    m["wd_fixed"] = 0
    m["common"].update(loss="qlike", lin=1, dec_zero=1)
    m["lin_init_refit"] = "data/pooled_lin_init/{design}/r{origin}_L{L}_wl{wl}.f32"
    (WORK / "pcal" / "mtest_plan.json").write_bytes(json.dumps(m, indent=1).encode())
    print("merge test", len(mt[0]), len(mt[1]))


def walk_inputs(lls: list[float]) -> None:
    """Per design (separate inputs, one job per block): the 8 pooled configurations x wd_enc {0, 3, 15} x wd_lin x
    lr_lin in each block's tune (QLIKE loss, es=qlike), the shortcut at its window's ridge."""
    pl = json.loads((WORK / "pwalk_wd0" / "plan.json").read_text())
    for d in SETS:
        dd = WORK / f"pwalk_rinitq_{d}"
        for k in range(5):
            out = []
            for x in _tune_src(k):
                jid = x.split()[0]
                if not jid.startswith(f"id=ptune_{d}_"):
                    continue
                stem, seed = jid.rsplit("_s", 1)
                L = int(x.split(" L=")[1].split()[0])
                for we in (0, 3, 15):
                    for wl in WLS:
                        for ll in lls:
                            sfx = f"_we{we:g}_wl{wl:g}" + (
                                f"_ll{ll:g}" if len(lls) > 1 else ""
                            )
                            out.append(
                                x.replace(jid, f"{stem}{sfx}_s{seed}", 1).replace(
                                    " loss=mse", " loss=qlike"
                                )
                                + f" {LIN} wd_enc={we:g} wd_lin={wl:g} lr_lin={ll:g} "
                                f"lin_init=data/pooled_lin_init/{d}/tune_b{k}_L{L}_wl{wl:g}.f32"
                            )
            LS.write(dd, f"tune_block{k}.txt", out)
        m = json.loads(json.dumps(pl))
        m["designs"] = [d]
        m["wd_fixed"] = 0
        m["common"].update(
            loss="qlike",
            lin=1,
            dec_zero=1,
            **({"lr_lin": lls[0]} if len(lls) == 1 else {}),
        )
        m["lin_init_refit"] = "data/pooled_lin_init/{design}/r{origin}_L{L}_wl{wl}.f32"
        (dd / "plan.json").write_bytes(json.dumps(m, indent=1).encode())
        print(dd.name, len((dd / "tune_block0.txt").read_text().splitlines()))


def calib(runs: list[Path]) -> pd.DataFrame:
    """Block-0 drift of the shortcut under QLIKE: v after 3 / 10 epochs against its ridge init in prediction space on
    the window's training rows, the seed spread of v.x on the window's own most extreme rows (top 1 % by input norm:
    causal, pre-test), and each tune configuration's best epochs and seed-mean validation QLIKE."""
    import re

    pl = info()
    o = pl["blocks"][0]["refit_rows_slice"][0]
    rows, tv = [], []
    for run in runs:
        d = "all_features" if "all_features" in run.name else "baseline"
        D = PDesign(d)
        keep = keep_of(D, o - W, o)
        rr = np.arange(o - W + 19, o)  # L = 20
        X = np.empty((len(rr), len(keep) + 1))
        X[:, :-1] = np.asarray(D.X[rr])[:, keep]
        X[:, -1] = 1.0
        ext = np.argsort(-np.linalg.norm(X[:, :-1], axis=1))[: len(rr) // 100]
        jo = pd.read_csv(run / "refit" / "jobs_out.csv")
        V = {}
        for jid in jo["id"]:
            m = re.match(
                r"^pcr_\w+?_L20_H64_lr0.003_we3_wl([0-9.]+)_ll([0-9.e-]+)_e(\d+)_s(\d+)$",
                jid,
            )
            assert m, jid
            wl, ll, ep, seed = m.groups()
            V[(wl, float(ll), int(ep), int(seed))] = np.fromfile(
                run / "refit" / "params" / f"{jid}.lin.f32", "<f4"
            )
        for wl, ll, ep in sorted({k[:3] for k in V}):
            v0 = np.fromfile(INIT / d / f"r{o}_L20_wl{wl}.f32", "<f4").astype(
                np.float64
            )
            vs = np.array(
                [V[(wl, ll, ep, s)] for s in sorted({k[3] for k in V})], np.float64
            )
            p0 = X @ v0
            dr = [
                float(np.linalg.norm(X @ (v - v0)) / np.linalg.norm(p0 - p0.mean()))
                for v in vs
            ]
            rows.append(
                dict(
                    design=d,
                    wd_lin=wl,
                    lr_lin=ll,
                    epochs=ep,
                    drift_mean=float(np.mean(dr)),
                    drift_max=float(np.max(dr)),
                    seed_sd_vx_extreme_rows=float(
                        (X[ext] @ vs.T).std(axis=1, ddof=1).mean()
                    ),
                    seed_sd_vx_all_rows=float((X @ vs.T).std(axis=1, ddof=1).mean()),
                )
            )
        tj = pd.read_csv(run / "tune" / "jobs_out.csv")
        tp = pd.read_csv(run / "tune" / "preds.csv")
        tp = tp[tp["kind"] == "val"]
        tj["cfg"] = tj["id"].str.extract(r"_(we3_wl[0-9.]+_ll[0-9.e-]+)_s\d+$")[0]
        tp = tp.merge(tj[["id", "cfg"]], on="id")
        for cfg, g in tp.groupby("cfg"):
            f = g.groupby("row")["f"].mean()
            y = D.y[f.index.to_numpy()]
            ok = y > 0
            q = (y[ok] / f.to_numpy()[ok]) ** 2
            tv.append(
                dict(
                    design=d,
                    cfg=cfg,
                    val_qlike_seedmean=float(np.mean(q - np.log(q) - 1)),
                    best_epoch_median=float(
                        tj[tj["cfg"] == cfg]["best_epoch"].median()
                    ),
                )
            )
    t, tv = pd.DataFrame(rows), pd.DataFrame(tv)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "calibration_drift.csv", index=False)
    tv.to_csv(OUT / "calibration_tune.csv", index=False)
    pd.set_option("display.width", 250)
    print(
        t.pivot_table(
            index=["design", "lr_lin", "wd_lin"],
            columns="epochs",
            values=["drift_mean", "seed_sd_vx_extreme_rows"],
        )
        .round(4)
        .to_string()
    )
    print(tv.round(4).to_string(index=False))
    return t


def shard(designs_shards: dict[str, list[tuple[str, ...]]]) -> None:
    """Split each walk input's tune_block<k>.txt into shard files by configuration group (H, L) patterns."""
    for d, groups in designs_shards.items():
        dd = WORK / f"pwalk_rinitq_{d}"
        for k in range(5):
            lines = [
                x
                for x in (dd / f"tune_block{k}.txt").read_text().splitlines()
                if x.strip()
            ]
            used = 0
            for i, pats in enumerate(groups):
                sel = [x for x in lines if any(f"_{p}_" in x.split()[0] for p in pats)]
                used += len(sel)
                LS.write(dd, f"tune_block{k}.s{i}.txt", sel)
            assert used == len(lines), (d, k, used, len(lines))
        print(
            d,
            [
                len((dd / f"tune_block0.s{i}.txt").read_text().splitlines())
                for i in range(len(groups))
            ],
        )


def compose(select_runs: dict[tuple[str, int], Path], out: Path) -> Path:
    """One walk directory from the per-(design, block) select runs: block<k>/refit/{preds,jobs_out}.csv of both
    designs, block<k>/selection.json merged, block<k>/tune/jobs_out.csv, the env files."""
    import shutil

    for k in range(5):
        b = out / f"block{k}"
        (b / "refit").mkdir(parents=True, exist_ok=True)
        (b / "tune").mkdir(parents=True, exist_ok=True)
        pr, jo, tj, sel = [], [], [], {}
        for d in SETS:
            r = select_runs[(d, k)]
            pr.append(pd.read_csv(r / "refit" / "preds.csv"))
            jo.append(pd.read_csv(r / "refit" / "jobs_out.csv"))
            tj.append(pd.read_csv(r / "tune" / "jobs_out.csv"))
            sel.update(json.loads((r / "selection.json").read_text()))
            shutil.copyfile(r / "env.txt", b / f"env_{d}.txt")
        pd.concat(pr).to_csv(b / "refit" / "preds.csv", index=False)
        pd.concat(jo).to_csv(b / "refit" / "jobs_out.csv", index=False)
        pd.concat(tj).to_csv(b / "tune" / "jobs_out.csv", index=False)
        (b / "selection.json").write_text(json.dumps(sel, indent=1, sort_keys=True))
        shutil.copyfile(
            select_runs[("all_features", k)] / "gpu_type.txt", b / "gpu_type.txt"
        )
    return out


def pooled_stack(arm: str) -> np.ndarray:
    import pooled_score as PS

    s = stack_yhat(arm, np.arange(PS.T0, PS.T1))
    assert np.isfinite(s.to_numpy()).all(), arm
    return s.to_numpy()


def score(walk: Path) -> None:  # noqa: C901 -- one report
    """pooled_score's two views for the pooled ridge-init QLIKE walk, with the comparators of the request."""
    import atm_straddle_lib as asl
    import close_linear_pretune as CLP
    import pooled_score as PS

    from src.evaluation.diebold_mariano import dm_test

    rows = PS.rows_info()
    r0 = int(rows["panel_row"].iloc[0])
    test = rows.iloc[PS.T0 - r0 : PS.T1 - r0].reset_index(drop=True)
    ys = {d: PDesign(d).y[PS.T0 - r0 : PS.T1 - r0] for d in SETS}
    B, rv = test["baseline"].to_numpy(), test["rv_raw"].to_numpy()
    # 2026-10-15 (data audit #10): fit-space losses against the ACTUAL RV on the fit scale, sqrt(rv_raw / B)
    y_act = np.sqrt(rv / B)
    for (
        d
    ) in SETS:  # ys: the panel's winsorized training target, kept for this count only
        print(
            f"all bars, {d}: bars whose winsorized y differs from sqrt(RV / B): "
            f"{int((np.abs(ys[d] - y_act) > 1e-6 * np.maximum(1.0, y_act)).sum())} of {len(y_act)}"
        )
    cands: dict = {}
    for d, f in PS.walk_forecasts(walk, r0).items():
        cands[(d, "pooled ridge-init QLIKE (MinGRU)")] = f
    for d, f in PS.walk_forecasts(
        RES / "pooled_wd_study" / "runs" / "pwalk_wdgrid_13138604", r0
    ).items():
        cands[(d, "pooled wd study B (MinGRU)")] = f
    cands.update(PS.trees())
    cands[("all_features", "pooled stack ridge blk2_user")] = pooled_stack("blk2_user")
    cands[("all_features", "pooled stack ridge blk2_tuned")] = pooled_stack(
        "blk2_tuned"
    )
    cands[("baseline", "pooled stack OLS a0_ols_har")] = pooled_stack("a0_ols_har")
    me = "pooled ridge-init QLIKE (MinGRU)"
    day = test["day"].to_numpy()
    v1, c1, daily = [], [], {}
    for (d, name), f in cands.items():
        # L = PS.bar_losses(f, ys[d], B, rv)  # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10)
        L = PS.bar_losses(f, y_act, B, rv)
        v1.append(
            dict(
                design=d,
                model=name,
                qlike_fit=float(np.nanmean(L["ql_fit"])),
                mse_fit=float(np.mean(L["mse_fit"])),
                qlike_plain=float(np.nanmean(L["ql_plain"])),
            )
        )
        daily[(d, name)] = pd.DataFrame(L).assign(day=day).groupby("day").mean()

    def c_all(a, b):
        r = dict(model=f"{a[1]} ({a[0]})", vs=f"{b[1]} ({b[0]})")
        for k in ("ql_fit", "mse_fit", "ql_plain"):
            x, z = daily[a][k].to_numpy(), daily[b][k].to_numpy()
            ok = np.isfinite(x) & np.isfinite(z)
            t, _ = asl.newey_west_t(x[ok] - z[ok])
            r.update(
                {
                    f"{k}_pct": float(100 * (x[ok].mean() / z[ok].mean() - 1)),
                    f"{k}_dm_p": float(dm_test(x[ok], z[ok])["p"]),
                    f"{k}_hac_t": float(t),
                }
            )
        return r

    c1.append(c_all(("all_features", me), ("baseline", me)))
    for d in SETS:
        for d2, name in cands:
            if d2 == d and name != me:
                c1.append(c_all((d, me), (d, name)))
    # ---- the 16:00 bar on the 866 trade days
    S = CLP.Scorer()
    sc = E._scoring()
    cl = E.day_classes()["day_class"].to_numpy()
    days = pd.DatetimeIndex(S.days).normalize()
    at16 = test[(test["t"].dt.hour == 16) & (test["t"].dt.minute == 0)].reset_index()
    pos = at16.set_index("day")["index"].reindex(days).to_numpy().astype(int)
    Bday = S.B.reindex(S.days).to_numpy(float)

    def from_plain(plain):
        x = S.daily_from_plain(plain)
        # x["mse_fit"] = (np.sqrt(plain / Bday) - sc["y_adj"][sc["day_row"]]) ** 2  # 2026-10-15 (user): parked -- scored against the trimmed target; scores now use actual RV (data audit #10)
        x["mse_fit"] = (np.sqrt(plain / Bday) - S.y_adj) ** 2  # vs sqrt(actual RV / B)
        return x

    c16 = {k: from_plain(f[pos] ** 2 * B[pos]) for k, f in cands.items()}
    import feature_mcs_report as FMR

    fcs = pd.read_parquet(
        FMR.OUT / "forecasts.parquet", columns=["est", "model", "row", "pred_adj"]
    )
    for d in SETS:
        for est in ("enet", "ridge"):
            c = fcs[(fcs["est"] == est) & (fcs["model"] == d)].sort_values("row")
            p = np.full(sc["n"], np.nan)
            p[c["row"].to_numpy()] = c["pred_adj"].to_numpy(float)
            c16[(d, f"per-bar {est}")] = E.losses(p)
        c16[(d, "per-bar ridge-init QLIKE (MinGRU)")] = E.losses(
            E.walk_pred(RES / "qlike_study" / "score_rinitq" / "forecasts.csv", d)
        )
        c16[(d, "per-bar softplus walk of record (MinGRU)")] = E.losses(
            E.walk_pred(E.REF_WALK / "forecasts.csv", d)
        )
    v2, c2 = [], []
    for (d, name), x in c16.items():
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
        v2.append(rec)
    pairs = [
        (("all_features", me), ("baseline", me)),
        (
            ("all_features", "per-bar ridge-init QLIKE (MinGRU)"),
            ("baseline", "per-bar ridge-init QLIKE (MinGRU)"),
        ),
    ]
    for d in SETS:
        pairs += [((d, me), k) for k in c16 if k[0] == d and k[1] != me]
    for a, b in pairs:
        r = dict(
            model=f"{a[1]} ({a[0]})",
            vs=f"{b[1]} ({b[0]})",
            **E.contrast_row(c16[a], c16[b]),
        )
        c2.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, t in (
        ("allbars_scores", v1),
        ("allbars_contrasts", c1),
        ("at1600_scores", v2),
        ("at1600_contrasts", c2),
    ):
        pd.DataFrame(t).to_csv(OUT / f"{name}.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for t in (v1, c1, v2, c2):
        print(pd.DataFrame(t).round(4).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pf":
        pf()
    elif cmd == "inits":
        inits()
    elif cmd == "check":
        check()
    elif cmd == "calib-inputs":
        calib_inputs() if len(sys.argv) < 3 else calib_inputs(
            tuple(float(x) for x in sys.argv[2].split(",")), sys.argv[3]
        )
    elif cmd == "calib":
        calib([Path(x) for x in sys.argv[2:]])
    elif cmd == "shard":
        shard(
            {
                "all_features": [("L20_H64", "L48_H64"), ("L20_H256",), ("L48_H256",)],
                "baseline": [("L20_H64", "L48_H64", "L20_H256", "L48_H256")],
            }
        )
    elif cmd == "score":
        score(Path(sys.argv[2]))
    elif cmd == "walk-inputs":
        walk_inputs([float(x) for x in sys.argv[2].split(",")])
