"""Linear shortcut beside the MinGRU branch (user, 2026-10-15; native_g8, the ResNet-style variant).

f = link(c + s [o + v.x]): o the MinGRU branch's decoder output at the last step, x the last step's input row (the
window mask's kept columns and the constant column), v a learned vector (job key lin=1, v = 0 at the start) trained
by the same loss with its own Adam (lr_lin, bias-corrected, beta 0.9 / 0.999, eps 1e-8, the cosine shape of lr),
its own clip at max_grad_norm, and wd_lin an L2 penalty under Adam: g_c += wd_lin pf_c v_c (pf = enc_pf.f64
through keep: backbone 0.01, exogenous 1; the intercept none), added before the clip.  Under Adam a decoupled decay
has no ridge fixed point (it is a box |v_c| <= 1 / (wd pf_c)); the L2 term's fixed point is the ridge.  wd_enc on
the branch's encoder stays decoupled.  dec_zero=1 zeroes the decoder after every weight load (the branch starts at 0).

Mapping (identity link, MSE loss, full-batch fixed point): with z = (y - mu_y) / sd_y and o' = o + v.x the trainer
minimises  mean(o' - z)^2 + 1/2 (wd_lin / sd_y^2) sum_c pf_c v_c^2,  i.e. wd_lin_eff = wd_lin / sd_y^2; in the
stack's ridge units (sum of squares + alpha sum pf_stack theta^2, theta = sd_y v):
alpha pf_stack_c = n wd_lin pf_c / (2 sd_y^2).  Under softplus / qlike the objective is the same penalised one
(the same pf structure) but has no closed form; near o' = 0 softplus is f ~ mu + sd o' (s sigmoid(c) = sd_y).

    python experiments/puffer_forecast/lin_study.py gate-inputs      # _work/gate8/
    python experiments/puffer_forecast/lin_study.py gate RUN
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wd_enc_study as E  # noqa: E402

WORK, RES, DATA, REPO = E.WORK, E.RES, E.DATA, E.REPO
SETS = E.SETS
G8 = WORK / "gate8"
N_STACK = 2000  # the stack's ridge window (rows)
ALPHA = {
    "all_features": 1000.0,
    "baseline": 0.01,
}  # the stack's ridge alpha in block 1 (feature_mcs models_ridge)
R_STACK = {
    "all_features": 0.01,
    "baseline": 1.0,
}  # the stack's backbone factor (all_features R_BB, baseline R_ONE)
RIDGE_WIN = (
    250,
    2250,
)  # the stack's block-1 start window (its forecast row 2250 = pred index 250)
BASE = (
    "mode=refit w0=381 w1=2381 L=5 H=16 layers=1 fc_n=1 val_tail=125 embargo=25 patience=20 momentum=0.95 "
    "anneal=1 min_lr_ratio=0.0 mask=1 xstd=0 seed=20261012"
)


def write(d: Path, name: str, lines: list[str]) -> None:
    E.write(d, name, lines)


def n_kept(design: str, w0: int, w1: int) -> int:
    _, mk, _ = E.arms_structure(design, w0, w1)
    return int((~mk[:-1].astype(bool)).sum())


def wd_for_alpha(design: str, alpha: float, w0: int, w1: int, L: int) -> float:
    """wd_lin whose fixed point is the stack's ridge at alpha on [w0, w1) (identity link, MSE): alpha pf_stack =
    n wd pf / (2 sd^2), pf = 0.01 on the backbone (ours) vs R_STACK on it (the stack's) -- exogenous 1 in both."""
    y = np.fromfile(DATA / design / "y.f32", "<f4").astype(np.float64)[w0 + L - 1 : w1]
    var = float(np.mean((y - y.mean()) ** 2))
    n = len(y)
    scale = 1.0 if design == "all_features" else R_STACK[design] / 0.01
    return 2.0 * var * alpha * scale / n


def gate_inputs() -> None:
    G8.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261015)
    P = {d: n_kept(d, 381, 2381) + 1 for d in SETS}
    for d, sd in (("baseline", 0.2), ("all_features", 0.05)):
        rng.normal(0.0, sd, P[d]).astype("<f4").tofile(G8 / f"lin_v0_{d}.f32")
    np.zeros(3, "<f4").tofile(G8 / "lin_bad_len.f32")
    lines = []
    # 2. dL/dv: one minibatch (128) at v0, every loss x link (baseline), one all_features combination
    for d, combos in (
        (
            "baseline",
            [
                (lo, lk)
                for lo in ("mse", "huber", "qlike")
                for lk in ("identity", "softplus")
            ],
        ),
        ("all_features", [("mse", "softplus"), ("qlike", "identity")]),
    ):
        for lo, lk in combos:
            lines.append(
                f"id=grad_{d}_{lo}_{lk} data=data/{d} {BASE} epochs=1 max_epochs=200 batch=128 lr=0.03 "
                f"max_grad_norm=1.0 loss={lo} link={lk} qlike_cap_q=1.0 lin=1 lr_lin=0.01 wd_lin=0.05 "
                f"lin_init=lin_v0_{d}.f32"
            )
    # 3a / 4. one full-batch step (the gradient dump of the same step: the grad_ twins below)
    step = []
    for d in SETS:
        step.append(
            f"id=step_{d}_lin data=data/{d} {BASE} epochs=1 max_epochs=200 batch=2048 lr=0.03 max_grad_norm=1.0 "
            f"loss=mse link=softplus lin=1 lr_lin=0.01 wd_lin=0.5 lin_init=lin_v0_{d}.f32 save_params=1"
        )
        step.append(
            f"id=step_{d}_dz data=data/{d} {BASE} epochs=1 max_epochs=200 batch=2048 lr=0.03 max_grad_norm=1.0 "
            f"loss=mse link=softplus lin=1 lr_lin=0.01 wd_lin=0.5 lin_init=lin_v0_{d}.f32 dec_zero=1 save_params=1"
        )
    # 3b. K full-batch steps with the branch frozen (dec_zero, lr 0): v's whole trajectory vs float64; one with a
    # binding clip
    for d in SETS:
        for tag, mgn in (("", 1.0), ("_clip", 0.02)):
            step.append(
                f"id=kstep_{d}{tag} data=data/{d} {BASE} epochs=30 max_epochs=200 batch=2048 lr=0 "
                f"max_grad_norm={mgn} loss=mse link=identity lin=1 lr_lin=0.003 wd_lin=0.5 lin_init=lin_v0_{d}.f32 "
                "dec_zero=1 save_params=1"
            )
    write(G8, "step.txt", step)
    grad = list(lines)
    for x in step:
        jid = x.split()[0][3:]
        grad.append(
            x.replace(f"id={jid}", f"id=grad_{jid}").replace(" save_params=1", "")
        )
    write(G8, "grad.txt", grad)
    # 5. the shortcut alone (branch frozen) to convergence: full batch, 3000 epochs, identity / MSE, L = 1 (the
    # training rows are the stack's window rows), wd_lin mapped from the stack's alpha (and 1/10 of it)
    w0, w1 = RIDGE_WIN
    ridge = []
    for d in SETS:
        for k, a in ((1, ALPHA[d]), (10, ALPHA[d] / 10)):
            wd = wd_for_alpha(d, a, w0, w1, 1)
            for lrl in (0.01, 0.002):
                ridge.append(
                    f"id=ridge_{d}_a{a:g}_lr{lrl:g} data=data/{d} mode=refit w0={w0} w1={w1} L=1 H=16 layers=1 "
                    f"fc_n=1 epochs=3000 max_epochs=3000 batch=2048 lr=0 momentum=0.95 max_grad_norm=1.0 anneal=1 "
                    f"min_lr_ratio=0.0 mask=1 xstd=0 seed=20261012 loss=mse link=identity lin=1 lr_lin={lrl:g} "
                    f"wd_lin={wd:.9g} dec_zero=1 fc_train=1 save_params=1"
                )
    write(G8, "ridge.txt", ridge)
    # 6. pool reset: four same-shape jobs back to back in one slot vs each alone
    pr = []
    for i, (seed, init) in enumerate(
        ((1, ""), (2, "lin_init=lin_v0_all_features.f32"), (3, ""), (4, ""))
    ):
        pr.append(
            f"id=reset_{i} data=data/all_features mode=refit w0=381 w1=2381 L=5 H=16 layers=1 fc_n=1 "
            f"epochs=6 max_epochs=200 batch=128 lr=0.03 momentum=0.95 max_grad_norm=1.0 anneal=1 min_lr_ratio=0.0 "
            f"mask=1 xstd=0 seed={seed} loss=mse link=softplus lin=1 lr_lin=0.01 wd_lin=0.05 dec_zero=1 {init} "
            "save_params=1"
        )
    write(G8, "reset.txt", pr)
    for i, x in enumerate(pr):
        write(G8, f"reset_alone_{i}.txt", [x])
    # 7. guards (toy)
    ok = (
        "mode=refit epochs=5 max_epochs=40 batch=32 lr=0.03 seed=5 max_grad_norm=1.0 data=data/toy w0=0 w1=390 L=6 H=8 "
        "layers=2 fc_n=10 val_tail=60 embargo=25 patience=40 momentum=0.95 anneal=1 min_lr_ratio=0.0 loss=mse "
        "link=identity mask=0 xstd=1"
    )
    bad = {
        "bad_lin2": "lin=2 lr_lin=0.01",
        "bad_dz2": "dec_zero=2",
        "bad_lin_nolr": "lin=1",
        "bad_lrlin_neg": "lin=1 lr_lin=-0.01",
        "bad_lrlin_nan": "lin=1 lr_lin=nan",
        "bad_wdlin_neg": "lin=1 lr_lin=0.01 wd_lin=-1",
        "bad_wdlin_nan": "lin=1 lr_lin=0.01 wd_lin=nan",
        "bad_lrlin_nolin": "lr_lin=0.01",
        "bad_wdlin_nolin": "wd_lin=0.1",
        "bad_lininit_nolin": "lin_init=lin_bad_len.f32",
        "bad_dz_initfrom": "dec_zero=1 init_from=lin_bad_len.f32",
        "bad_lininit_len": "lin=1 lr_lin=0.01 lin_init=lin_bad_len.f32",
    }
    for name, keys in bad.items():
        write(G8, f"{name}.txt", [f"id=bad {keys} {ok}"])
    write(
        G8,
        "bad_wdlin_nopf.txt",
        [
            f"id=bad lin=1 lr_lin=0.01 wd_lin=0.1 {ok}".replace(
                "data=data/toy ", "data=data/toy_nopf "
            )
        ],
    )
    write(G8, "lin_pack.txt", [f"id=bad lin=1 lr_lin=0.01 {ok}"])
    write(G8, "dz_pack.txt", [f"id=bad dec_zero=1 {ok}"])
    write(G8, "ok_small.txt", [f"id=ok lin=1 lr_lin=0.01 wd_lin=0.1 dec_zero=1 {ok}"])
    print(
        sorted(p.name for p in G8.iterdir()),
        {d: P[d] for d in SETS},
        len(grad),
        len(ridge),
    )


CAL_LR = (0.001, 0.003, 0.01, 0.03)
CAL_EP = (5, 10, 20, 40, 80)
CAL_ALPHA = {"all_features": (10.0, 100.0, 1000.0), "baseline": (0.01, 0.1, 1.0)}


def calib_inputs() -> None:
    """The shortcut alone (branch frozen: dec_zero, lr 0) on the block-0 tune window: refits (identity, MSE,
    minibatch 128 as the walks) after 5 .. 80 epochs, lr_lin x the alpha-mapped wd_lin (distance to the closed-form
    ridge); tune jobs (softplus as the walks) for the early-stopped epochs."""
    d0 = WORK / "lin_calib"
    ref, tune = [], []
    for d in SETS:
        for a in CAL_ALPHA[d]:
            wd = wd_for_alpha(d, a, 381, 2381, 5)
            for lrl in CAL_LR:
                common = (
                    f"data=data/{d} {BASE} max_epochs=200 batch=128 lr=0 max_grad_norm=1.0 loss=mse lin=1 "
                    f"lr_lin={lrl:g} wd_lin={wd:.9g} dec_zero=1"
                )
                for ep in CAL_EP:
                    ref.append(
                        f"id=cal_{d}_a{a:g}_lr{lrl:g}_e{ep} {common} link=identity epochs={ep} save_params=1"
                    )
                tune.append(
                    f"id=calt_{d}_a{a:g}_lr{lrl:g} {common.replace('mode=refit', 'mode=tune')} link=softplus"
                )
    write(d0, "calib_refit.txt", ref)
    write(d0, "calib_tune.txt", tune)
    print(len(ref), len(tune))


def _lp(meta: dict) -> dict:
    return dict(
        mu_y=meta["mu_y"],
        sd_y=meta["sd_y"],
        link=meta["link"],
        link_c=meta["link_c"],
        link_s=meta["link_s"],
        loss_kind=meta["loss_kind"],
        huber_delta=meta["huber_delta"],
        qlike_eps=meta["qlike_eps"],
        qlike_cap=meta["qlike_cap"],
    )


def _f32(p: Path) -> np.ndarray:
    return np.fromfile(p, "<f4").astype(np.float64)


def _design_pf(design: str, keep: np.ndarray) -> np.ndarray:
    """wd_lin's factors through keep (the bias column's 0), from the design's enc_pf.f64."""
    return np.append(E.design_pf(design)[keep], 0.0)


def _batch_loss(
    lp: dict, o: np.ndarray, y: np.ndarray, mask: np.ndarray, n_real: int, cap: float
):
    import reference as RF

    f, dfdo = RF.link_ref(lp, o)
    lval, gf, capped = RF.loss_ref(
        lp["loss_kind"], f, y, lp["huber_delta"], lp["qlike_eps"], cap
    )
    return float(np.sum(mask * lval) / n_real), mask * gf * dfdo / n_real, capped


def adam_ref(v0, grad_fn, pf, wd, lrs, S, max_norm, epochs):
    """v's optimiser in float64: g += wd pf v (intercept pf 0), clip at max_norm, Adam (0.9, 0.999, 1e-8)."""
    v = v0.copy()
    m = np.zeros_like(v)
    s = np.zeros_like(v)
    t = 0
    for e in range(epochs):
        for _ in range(S):
            t += 1
            g = grad_fn(v) + wd * pf * v
            clip = min(max_norm / (np.sqrt(np.sum(g * g)) + 1e-6), 1.0)
            g = g * clip
            m = 0.9 * m + 0.1 * g
            s = 0.999 * s + 0.001 * g * g
            v = v - lrs[e] * (m / (1 - 0.9**t)) / (np.sqrt(s / (1 - 0.999**t)) + 1e-8)
    return v


def lin_sched(lr: float, max_epochs: int) -> np.ndarray:
    """The trainer's cosine schedule in float32 (anneal 1, min_lr_ratio 0)."""
    e = np.arange(max_epochs)
    c = np.cos(np.pi * e / max_epochs).astype(np.float32)
    return (np.float32(0.5) * np.float32(lr) * (np.float32(1.0) + c)).astype(np.float64)


def ridge_closed(
    X: np.ndarray, z: np.ndarray, pf: np.ndarray, lam: float
) -> np.ndarray:
    """argmin mean(Xv - z)^2 + 1/2 lam sum pf v^2 (float64)."""
    n = len(z)
    A = X.T @ X / n + 0.5 * lam * np.diag(pf)
    return np.linalg.solve(A, X.T @ z / n)


def window_xy(design: str, w0: int, w1: int, L: int, keep: np.ndarray):
    n, p = map(int, (DATA / design / "meta.txt").read_text().split())
    X = np.fromfile(DATA / design / "X.f32", "<f4").reshape(n, p)
    y = np.fromfile(DATA / design / "y.f32", "<f4").astype(np.float64)
    rows = np.arange(w0 + L - 1, w1)
    Xk = np.hstack([X[rows][:, keep].astype(np.float64), np.ones((len(rows), 1))])
    xf = np.append(X[w1][keep].astype(np.float64), 1.0)  # the forecast row
    return Xk, y[rows], xf


def gate(run: Path) -> dict:  # noqa: C901 -- one report
    import gate4 as G4
    import gate5 as G5

    res: dict = {"env": (run / "env.txt").read_text(errors="replace").splitlines()[:3]}
    res["tune_g7_vs_g8"] = G4.identical(run / "id_old_tune", run / "id_new_tune")
    res["refit_g7_vs_g8"] = G4.identical(run / "id_old_refit", run / "id_new_refit")
    res["fct_g7_vs_g8"] = G4.identical(run / "id_old_fct", run / "id_new_fct")
    res["tune_g8_pool_vs_wave"] = G4.identical(
        run / "id_new_wave_tune", run / "id_new_tune", tune=True
    )
    res["refit_g8_pool_vs_wave"] = G4.identical(
        run / "id_new_wave_refit", run / "id_new_refit"
    )
    res["tune_g8_off_written_vs_absent"] = G4.identical(
        run / "id_new_tune", run / "id_new_off_tune"
    )
    gd, ok = G5.guards(run)
    res["guards"], res["guards_ok"] = gd.to_dict("records"), ok
    gc = run / "gradcheck"
    # ---- 2. dL/dv vs float64 (analytic with the kernel's cap; central differences)
    grows = []
    for mf in sorted(gc.glob("grad_*.json")):
        jid = mf.stem
        meta = json.loads(mf.read_text())
        if not meta.get("lin"):
            continue
        B, T, P, n_real = meta["B"], meta["T"], meta["P"], meta["n_real"]
        lp = _lp(meta)
        obs = _f32(gc / f"{jid}.obs.f32").reshape(B, T, P)
        dec = _f32(gc / f"{jid}.dec.f32").reshape(B, T, 2)
        decp = _f32(gc / f"{jid}.decp.f32").reshape(B, T, 2)
        y = _f32(gc / f"{jid}.y.f32")
        mask = _f32(gc / f"{jid}.mask.f32")
        v0 = _f32(gc / f"{jid}.lin0.f32")
        lg = _f32(gc / f"{jid}.lingrad.f32")
        st = _f32(gc / f"{jid}.step.f32")
        x = obs[:, T - 1, :]
        o = dec[:, T - 1, 0]
        op = o + x @ v0
        L64, gl, capped = _batch_loss(lp, op, y, mask, n_real, lp["qlike_cap"])
        g64 = x.T @ gl
        # central differences of the uncapped loss in v (float64)
        h = 1e-6
        fd = np.empty(P)
        for c in range(P):
            dv = np.zeros(P)
            dv[c] = h
            lp_, _, _ = _batch_loss(lp, o + x @ (v0 + dv), y, mask, n_real, np.inf)
            lm_, _, _ = _batch_loss(lp, o + x @ (v0 - dv), y, mask, n_real, np.inf)
            fd[c] = (lp_ - lm_) / (2 * h)
        _, gl_unc, _ = _batch_loss(lp, op, y, mask, n_real, np.inf)
        g_unc = x.T @ gl_unc
        sc = np.max(np.abs(g64))
        grows.append(
            dict(
                job=jid,
                P=P,
                loss=["mse", "huber", "qlike"][meta["loss_kind"]],
                link=["identity", "softplus"][meta["link"]],
                n_capped=int((mask * capped).sum()),
                o_prime_max_abs_err=float(np.max(np.abs(decp[:, T - 1, 0] - op))),
                loss_rel_err=float(abs(st[0] / n_real - L64) / abs(L64)),
                native_vs_f64_max_abs_over_max=float(np.max(np.abs(lg - g64)) / sc),
                fd_vs_f64_uncapped_max_abs_over_max=float(
                    np.max(np.abs(fd - g_unc)) / np.max(np.abs(g_unc))
                ),
                max_abs_grad=float(sc),
            )
        )
    res["grad"] = grows
    # ---- 3a. one full-batch step: v1 vs float64 (pf through keep, the intercept with no L2)
    srows = []
    for d in SETS:
        jid = f"step_{d}_lin"
        meta = json.loads((gc / f"grad_{jid}.json").read_text())
        keep = np.fromfile(gc / f"grad_{jid}.keep.i32", "<i4")
        pf = _design_pf(d, keep)
        used = np.fromfile(gc / f"grad_{jid}.linpf.f32", "<f4")
        v0 = _f32(gc / f"grad_{jid}.lin0.f32")
        g = _f32(gc / f"grad_{jid}.lingrad.f32")
        v1 = _f32(run / "step" / "params" / f"{jid}.lin.f32")
        lrs = lin_sched(meta["lr_lin"], 200)
        ref = adam_ref(
            v0, lambda v, g=g: g, pf, meta["wd_lin"], lrs, 1, meta["max_grad_norm"], 1
        )
        ref_dec = adam_ref(
            v0,
            lambda v, g=g: g,
            np.append(pf[:-1], 1.0),
            meta["wd_lin"],
            lrs,
            1,
            meta["max_grad_norm"],
            1,
        )
        srows.append(
            dict(
                job=jid,
                P=len(v0),
                pf_used_eq_design_pf_through_keep=bool(
                    np.array_equal(used, pf.astype("<f4"))
                ),
                native_vs_f64_max_abs=float(np.max(np.abs(v1 - ref))),
                step_size=float(np.max(np.abs(ref - v0))),
                intercept_err_if_decayed=float(abs(v1[-1] - ref_dec[-1])),
                intercept_err=float(abs(v1[-1] - ref[-1])),
            )
        )
    # ---- 3b. 30 full-batch steps of v alone (branch frozen at 0): the trajectory in float64
    for d in SETS:
        for tag in ("", "_clip"):
            jid = f"kstep_{d}{tag}"
            meta = json.loads((gc / f"grad_{jid}.json").read_text())
            keep = np.fromfile(gc / f"grad_{jid}.keep.i32", "<i4")
            P, n_ctx, row0, N = meta["P"], meta["n_ctx"], meta["row0"], meta["N"]
            Z = _f32(gc / f"grad_{jid}.Z.f32").reshape(n_ctx, P)
            yr = _f32(gc / f"grad_{jid}.yrow.f32")
            X = Z[row0 : row0 + N]
            yy = yr[row0 : row0 + N]
            mu, sd = meta["mu_y"], meta["sd_y"]

            def grad_fn(v, X=X, yy=yy, mu=mu, sd=sd, N=N):
                return X.T @ (2.0 * (mu + sd * (X @ v) - yy) * sd / N)

            v0 = _f32(gc / f"grad_{jid}.lin0.f32")
            vk = _f32(run / "step" / "params" / f"{jid}.lin.f32")
            ref = adam_ref(
                v0,
                grad_fn,
                _design_pf(d, keep),
                meta["wd_lin"],
                lin_sched(meta["lr_lin"], 200),
                1,
                meta["max_grad_norm"],
                30,
            )
            srows.append(
                dict(
                    job=jid,
                    P=P,
                    native_vs_f64_max_abs=float(np.max(np.abs(vk - ref))),
                    step_size=float(np.max(np.abs(ref - v0))),
                    rel=float(np.max(np.abs(vk - ref)) / np.max(np.abs(ref - v0))),
                )
            )
    res["adam"] = srows
    # ---- 4. dec_zero: the branch is exactly 0 at the first step; one step leaves the encoder / MinGRU untouched
    import reference as RF

    zrows = []
    for d in SETS:
        jid = f"step_{d}_dz"
        meta = json.loads((gc / f"grad_{jid}.json").read_text())
        B, T, P, H, n_real = meta["B"], meta["T"], meta["P"], meta["H"], meta["n_real"]
        dec = _f32(gc / f"grad_{jid}.dec.f32")
        obs = _f32(gc / f"grad_{jid}.obs.f32").reshape(B, T, P)
        decp = _f32(gc / f"grad_{jid}.decp.f32").reshape(B, T, 2)
        v0 = _f32(gc / f"grad_{jid}.lin0.f32")
        y = _f32(gc / f"grad_{jid}.y.f32")
        mask = _f32(gc / f"grad_{jid}.mask.f32")
        st = _f32(gc / f"grad_{jid}.step.f32")
        lp = _lp(meta)
        op = obs[:, T - 1, :] @ v0
        L64, _, _ = _batch_loss(lp, op, y, mask, n_real, lp["qlike_cap"])
        p0 = RF.unpack(_f32(gc / f"grad_{jid}.params0.f32"), P, H, 1)
        p1 = RF.unpack(_f32(run / "step" / "params" / f"{jid}.f32"), P, H, 1)
        zrows.append(
            dict(
                job=jid,
                branch_output_max_abs=float(np.max(np.abs(dec))),
                o_prime_eq_v_dot_x_max_abs_err=float(
                    np.max(np.abs(decp[:, T - 1, 0] - op))
                ),
                first_step_loss_rel_err=float(abs(st[0] / n_real - L64) / L64),
                decoder_row0_norm_after_step=float(np.linalg.norm(p1["Wd"][0])),
                decoder_row1_norm_after_step=float(np.linalg.norm(p1["Wd"][1])),
                encoder_bit_identical=bool(np.array_equal(p0["We"], p1["We"])),
                mingru_bit_identical=bool(np.array_equal(p0["Wl"][0], p1["Wl"][0])),
            )
        )
    res["dec_zero"] = zrows
    # ---- 5. the shortcut alone to convergence = the ridge; and the stack's own ridge forecast
    import feature_mcs_report as FMR

    fcs = pd.read_parquet(
        FMR.OUT / "forecasts.parquet", columns=["est", "model", "row", "pred_adj"]
    )
    jo = pd.read_csv(run / "ridge" / "jobs_out.csv").set_index("id")
    pr = pd.read_csv(run / "ridge" / "preds.csv")
    rrows = []
    w0, w1 = RIDGE_WIN
    for jid, r in jo.iterrows():
        d = "all_features" if "_all_features_" in jid else "baseline"
        keep = np.fromfile(run / "ridge" / "params" / f"{jid}.keep.i32", "<i4")
        v = _f32(run / "ridge" / "params" / f"{jid}.lin.f32")
        X, y, xf = window_xy(d, w0, w1, 1, keep)
        mu, sd, wd, n = float(r["mu_y"]), float(r["sd_y"]), float(r["wd_lin"]), len(y)
        pf = _design_pf(d, keep)
        z = (y - mu) / sd
        lam = wd / sd**2
        vs = ridge_closed(X, z, pf, lam)

        def J(vv, X=X, z=z, pf=pf, lam=lam):
            return float(np.mean((X @ vv - z) ** 2) + 0.5 * lam * np.sum(pf * vv * vv))

        p = pr[pr["id"] == jid]
        f_train = p[p["kind"] == "train"].sort_values("row")["f"].to_numpy()
        f_fc = float(p[p["kind"] == "fc"]["f"].iloc[0])
        f_star = mu + sd * (X @ vs)
        fc_star = mu + sd * float(xf @ vs)
        # the stack's ridge in its own units: (X'X + alpha pf_stack) theta = X'y, alpha = n wd pf / (2 sd^2 pf_stack)
        pf_stack = pf.copy()
        if d == "baseline":
            pf_stack[:-1] = R_STACK[d]
        alpha_impl = (
            n * wd * pf[0] / (2 * sd**2 * pf_stack[0])
            if d == "baseline"
            else n * wd / (2 * sd**2)
        )
        th = np.linalg.solve(X.T @ X + alpha_impl * np.diag(pf_stack), X.T @ y)
        c = fcs[(fcs["est"] == "ridge") & (fcs["model"] == d) & (fcs["row"] == w0)]
        stack_fc = float(c["pred_adj"].iloc[0]) if len(c) else np.nan
        rrows.append(
            dict(
                job=jid,
                design=d,
                P=len(v),
                lr_lin=float(r["lr_lin"]),
                wd_lin=wd,
                wd_lin_eff=lam,
                alpha_implied=alpha_impl,
                objective_rel_gap=(J(v) - J(vs)) / J(vs),
                fit_max_abs_diff=float(np.max(np.abs(f_train - f_star))),
                fit_rmse_diff_over_sd=float(
                    np.sqrt(np.mean((f_train - f_star) ** 2)) / sd
                ),
                fc_trainer=f_fc,
                fc_closed_form=fc_star,
                fc_closed_form_stack_units=float(xf @ th),
                fc_stack_ridge=stack_fc,
                coef_pred_space_rel=float(
                    np.linalg.norm(X @ (v - vs)) / np.linalg.norm(X @ vs)
                ),
            )
        )
    res["ridge"] = rrows
    # ---- 6. pool reset: back to back in one slot == alone
    out = []
    pp = pd.read_csv(run / "reset_pool" / "preds.csv")
    for i in range(4):
        pa = pd.read_csv(run / f"reset_alone_{i}" / "preds.csv")
        jid = f"reset_{i}"
        a = pp[pp["id"] == jid].sort_values(["kind", "row"]).reset_index(drop=True)
        b = pa.sort_values(["kind", "row"]).reset_index(drop=True)
        par = (run / "reset_pool" / "params" / f"{jid}.f32").read_bytes()
        pal = (run / f"reset_alone_{i}" / "params" / f"{jid}.f32").read_bytes()
        la = (run / "reset_pool" / "params" / f"{jid}.lin.f32").read_bytes()
        lb = (run / f"reset_alone_{i}" / "params" / f"{jid}.lin.f32").read_bytes()
        out.append(
            dict(
                job=jid,
                preds_identical=bool(a.equals(b)),
                params_identical=par == pal,
                lin_identical=la == lb,
                kinds=sorted(set(a["kind"])),
            )
        )
    res["pool_reset"] = out
    outp = RES / "review_build" / f"gate8_{run.name}.json"
    outp.write_bytes(json.dumps(res, indent=1, default=str).encode())
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for k, v in res.items():
        if k.endswith(("_vs_g8", "_vs_wave", "_vs_absent")) or k == "guards_ok":
            print(k, v)
    print(gd.to_string(index=False))
    for k in ("grad", "adam", "dec_zero", "ridge", "pool_reset"):
        print(pd.DataFrame(res[k]).to_string(index=False))
    return res


def calib(run: Path) -> None:
    """Distance of the minibatch-trained shortcut (identity, MSE) to the closed-form ridge after 5 .. 80 epochs;
    the early-stopped epochs of the shortcut-alone tunes (softplus)."""
    jo = pd.read_csv(run / "calib_refit" / "jobs_out.csv").set_index("id")
    pr = pd.read_csv(run / "calib_refit" / "preds.csv")
    rows = []
    cache: dict = {}
    for jid, r in jo.iterrows():
        d = "all_features" if "_all_features_" in jid else "baseline"
        keep = np.fromfile(run / "calib_refit" / "params" / f"{jid}.keep.i32", "<i4")
        v = _f32(run / "calib_refit" / "params" / f"{jid}.lin.f32")
        mu, sd, wd = float(r["mu_y"]), float(r["sd_y"]), float(r["wd_lin"])
        key = (d, wd)
        if key not in cache:
            X, y, xf = window_xy(d, 381, 2381, 5, keep)
            z = (y - mu) / sd
            pf = _design_pf(d, keep)
            cache[key] = (X, z, xf, ridge_closed(X, z, pf, wd / sd**2))
        X, z, xf, vs = cache[key]
        a = jid.rsplit("_a", 1)[1].split("_")[0]
        rows.append(
            dict(
                design=d,
                alpha=float(a),
                lr_lin=float(r["lr_lin"]),
                epochs=int(r["epochs_run"]),
                pred_dist=float(
                    np.linalg.norm(X @ (v - vs))
                    / np.linalg.norm(X @ vs - np.mean(X @ vs))
                ),
                fc_diff_over_sd=float(
                    abs(
                        float(pr[(pr["id"] == jid) & (pr["kind"] == "fc")]["f"].iloc[0])
                        - (mu + sd * float(xf @ vs))
                    )
                    / sd
                ),
            )
        )
    t = pd.DataFrame(rows)
    tt = pd.read_csv(run / "calib_tune" / "jobs_out.csv")
    tt["design"] = tt["id"].str.extract(r"^calt_(baseline|all_features)_")[0]
    tt["alpha"] = tt["id"].str.extract(r"_a([0-9.e+-]+)_lr")[0].astype(float)
    out = RES / "lin_study"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "calibration_refit.csv", index=False)
    tt[["design", "alpha", "lr_lin", "wd_lin", "best_epoch", "best_val_mse"]].to_csv(
        out / "calibration_tune.csv", index=False
    )
    pd.set_option("display.width", 250)
    print(
        t.pivot_table(
            index=["design", "alpha", "lr_lin"], columns="epochs", values="pred_dist"
        )
        .round(3)
        .to_string()
    )
    print(
        tt[
            ["design", "alpha", "lr_lin", "wd_lin", "best_epoch", "best_val_mse"]
        ].to_string(index=False)
    )


def walk_inputs(
    lr_lin: float,
    wd_lins: list[float],
    wd_encs: list[float],
    wl_only: list[float],
    wl_mid: float,
) -> None:
    """B (lingrid): the 12-configuration grid x wd_enc x wd_lin in each block's tune, both designs.  A on
    all_features: shortcut only (branch frozen: lr 0, dec_zero; L5 H16; wd_lin in the tune grid), and the 12-config
    grid at (wd_enc 15, wd_lin mid) and (wd_enc 0, wd_lin mid).  softplus, lin=1, dec_zero=1, lr_lin fixed."""
    src = WORK / "phase2c"
    m = json.loads((src / "blocks.json").read_text())
    lin = f"link=softplus lin=1 lr_lin={lr_lin:g} dec_zero=1"

    def man(d: Path, designs: list[str], extra: dict) -> None:
        mm = json.loads(json.dumps(m))
        mm["designs"] = designs
        mm["common"].update(link="softplus", lin=1, lr_lin=lr_lin, dec_zero=1, **extra)
        (d / "blocks.json").write_bytes(json.dumps(mm, indent=1).encode())

    def tune_lines(k: int, designs: tuple) -> list[str]:
        return [
            x
            for x in (src / f"tune_block{k}.txt").read_text().splitlines()
            if x.strip() and any(f"data=data/{d} " in x for d in designs)
        ]

    # B
    d = WORK / "walk_softplus_lingrid"
    for k in range(5):
        out = []
        for x in tune_lines(k, SETS):
            jid = x.split()[0]
            stem, seed = jid.rsplit("_s", 1)
            for we in wd_encs:
                for wl in wd_lins:
                    out.append(
                        x.replace(jid, f"{stem}_we{we:g}_wl{wl:g}_s{seed}", 1)
                        + f" {lin} wd_enc={we:g} wd_lin={wl:g}"
                    )
        write(d, f"tune_block{k}.txt", out)
    man(d, list(SETS), {})
    # A1: the shortcut alone
    d = WORK / "walk_softplus_linonly"
    for k in range(5):
        out = []
        for x in tune_lines(k, ("all_features",)):
            jid = x.split()[0]
            if "_L5_H16_lr0.003_" not in jid:
                continue
            seed = jid.rsplit("_s", 1)[1]
            for wl in wl_only:
                y = x.replace(
                    jid, f"id=tune_all_features_L5_H16_lr0_wl{wl:g}_s{seed}", 1
                ).replace(" lr=0.003 ", " lr=0 ")
                out.append(f"{y} {lin} wd_lin={wl:g}")
        write(d, f"tune_block{k}.txt", out)
    man(d, ["all_features"], {})
    # A2 / A3: the grid at a fixed (wd_enc, wd_lin)
    for we in (15.0, 0.0):
        d = WORK / f"walk_softplus_lin_we{we:g}"
        for k in range(5):
            out = [
                f"{x} {lin} wd_enc={we:g} wd_lin={wl_mid:g}"
                for x in tune_lines(k, ("all_features",))
            ]
            write(d, f"tune_block{k}.txt", out)
        man(d, ["all_features"], {"wd_enc": we, "wd_lin": wl_mid})
    print(
        {
            p.name: len((p / "tune_block0.txt").read_text().splitlines())
            for p in sorted(WORK.glob("walk_softplus_lin*"))
        }
    )


STUDY = RES / "lin_study"
FIX1 = RES / "wd_enc_study" / "score_wegrid" / "forecasts.csv"


def lin_model(params: Path, jid: str, jo: pd.DataFrame):
    sys.path.insert(0, str(E.DIAG / "scripts"))
    import netlib as NL

    M = NL.load_model(params, jid, jo)
    M["v"] = _f32(params / f"{jid}.lin.f32")
    return M


def lin_forecast(
    M: dict, x: np.ndarray, x_lin: np.ndarray | None = None, o: np.ndarray | None = None
):
    """f = link(c + s [o + v.x_last]); o from the branch on x unless given, the shortcut on x_lin (default x)."""
    import reference as RF

    sys.path.insert(0, str(E.DIAG / "scripts"))
    import netlib as NL

    if o is None:
        _, o, _ = NL.forecast(M, x)
    xl = x if x_lin is None else x_lin
    f, _ = RF.link_ref(M["lp"], o + xl[:, -1, :] @ M["v"])
    return f, o


def sensitivities(run: Path, tag: str) -> pd.DataFrame:
    """af_diagnosis d06 with the shortcut: the forecast change from is_month_end switched on at the last step and
    from a +1 move of har_ma_1, in total, through the shortcut alone (branch input unchanged) and through the branch
    alone (shortcut input unchanged); each block's first-origin refits, every seed."""
    import feature_mcs_runner as FR

    sys.path.insert(0, str(E.DIAG / "scripts"))
    import netlib as NL

    models = FR.load_inputs()["models"]
    out = []
    for blk, b in enumerate(E.blocks()):
        o0 = b["origins"][0]
        rd = run / f"block{blk}" / "refit"
        jo = pd.read_csv(rd / "jobs_out.csv")
        for d in SETS:
            cols = list(models[d]["cols"])
            n, p = map(int, (DATA / d / "meta.txt").read_text().split())
            XX = np.fromfile(DATA / d / "X.f32", "<f4").reshape(n, p)
            for sd in range(20261012, 20261017):
                jid = f"refit_{d}_r{o0}_s{sd}"
                if not (rd / "params" / f"{jid}.lin.f32").is_file():
                    continue
                M = lin_model(rd / "params", jid, jo)
                keep = list(M["keep"])
                x = NL.seqs(XX, M["keep"], np.arange(o0 - 500, o0), M["L"])
                f0, o_0 = lin_forecast(M, x)
                rec = dict(walk=tag, block=blk, design=d, seed=sd, L=M["L"], H=M["H"])
                for name, col, how in (
                    ("month_end", "is_month_end", "set1"),
                    ("har_ma_1", "har_ma_1", "add1"),
                ):
                    c = keep.index(cols.index(col))
                    x1 = x.copy()
                    if how == "set1":
                        sel = x1[:, -1, c] == 0
                        x1[:, -1, c] = 1.0
                    else:
                        sel = np.ones(len(x1), bool)
                        x1[:, -1, c] += 1.0
                    f_tot, o_1 = lin_forecast(M, x1)
                    f_lin, _ = lin_forecast(M, x, x_lin=x1, o=o_0)
                    f_br, _ = lin_forecast(M, x1, x_lin=x, o=o_1)
                    rec[f"d_{name}"] = float(np.mean((f_tot - f0)[sel]))
                    rec[f"d_{name}_lin"] = float(np.mean((f_lin - f0)[sel]))
                    rec[f"d_{name}_branch"] = float(np.mean((f_br - f0)[sel]))
                out.append(rec)
    return pd.DataFrame(out)


def variance_share(run: Path, tag: str) -> list[dict]:
    """Pre-link a = c + s o + s v.x on every forecast row (seed means): the shares of var(a) from the shortcut
    s v.x, the branch s o and twice their covariance (f inverted through the softplus link)."""
    out = []
    for d in SETS:
        parts = []
        for b in sorted(run.glob("block*")):
            pr = pd.read_csv(b / "refit" / "preds.csv")
            pr = pr[pr["id"].str.startswith(f"refit_{d}_")]
            if pr.empty:
                continue
            jo = pd.read_csv(b / "refit" / "jobs_out.csv").set_index("id")
            f = pr[pr["kind"] == "fc"].set_index("id")["f"]
            lv = pr[pr["kind"] == "fc_lin"].set_index("id")["f"]
            row = pr[pr["kind"] == "fc"].set_index("id")["row"]
            mu, sd = (
                jo.loc[f.index, "mu_y"].to_numpy(),
                jo.loc[f.index, "sd_y"].to_numpy(),
            )
            c = np.log(np.expm1(mu))
            s = sd * (1 + np.exp(-c))
            a = np.log(np.expm1(f.to_numpy()))
            lin = s * lv.loc[f.index].to_numpy()
            parts.append(
                pd.DataFrame(
                    dict(row=row.to_numpy(), a=a - c, lin=lin, branch=a - c - lin)
                )
            )
        if not parts:
            continue
        t = pd.concat(parts).groupby("row")[["a", "lin", "branch"]].mean()
        va = float(t["a"].var())
        out.append(
            dict(
                walk=tag,
                design=d,
                share_lin=float(t["lin"].var() / va),
                share_branch=float(t["branch"].var() / va),
                share_2cov=float(2 * t[["lin", "branch"]].cov().iloc[0, 1] / va),
                corr=float(t[["lin", "branch"]].corr().iloc[0, 1]),
            )
        )
    return out


def study(walks: dict[str, Path]) -> None:  # noqa: C901 -- one report
    """walks: tag -> run dir; each scored by phase2_score into lin_study/score_<tag>/ first."""
    from src.evaluation.diebold_mariano import dm_test

    cl = E.day_classes()["day_class"].to_numpy()
    cands: dict = {}
    for d in SETS:
        cands[(d, "softplus walk of record")] = E.losses(
            E.walk_pred(E.REF_WALK / "forecasts.csv", d)
        )
        cands[(d, "enet")] = E.losses(E.enet_pred(d))
        cands[(d, "fix #1 B (wd_enc grid)")] = E.losses(E.walk_pred(FIX1, d))
    for tag in walks:
        fc = STUDY / f"score_{tag}" / "forecasts.csv"
        for d in pd.read_csv(fc)["design"].unique():
            cands[(d, f"g8 {tag}")] = E.losses(E.walk_pred(fc, d))
    S = E._scoring()["S"]
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
    for (d, name), g in cands.items():
        if not name.startswith("g8"):
            continue
        refs = [
            (d, "enet"),
            (d, "fix #1 B (wd_enc grid)"),
            ("baseline", "softplus walk of record"),
            (d, "softplus walk of record"),
        ]
        if d == "all_features" and ("baseline", name) in cands:
            refs.append(("baseline", name))
        for rd, rn in dict.fromkeys(refs):
            h = cands[(rd, rn)]
            r = dict(design=d, model=name, vs=f"{rn} ({rd})", **E.contrast_row(g, h))
            for c in ("month_end", "extreme_inputs", "other"):
                dq = (g["ql"] - h["ql"])[cl == c]
                r[f"dql_{c}"] = float(dq.mean())
                r[f"p_{c}"] = float(dm_test(g["ql"][cl == c], h["ql"][cl == c])["p"])
            con.append(r)
    # fix #1 B vs the enet, for the before / after of the gap
    for d in SETS:
        r = dict(
            design=d,
            model="fix #1 B (wd_enc grid)",
            vs=f"enet ({d})",
            **E.contrast_row(cands[(d, "fix #1 B (wd_enc grid)")], cands[(d, "enet")]),
        )
        con.append(r)
    r = dict(
        design="all_features",
        model="fix #1 B (wd_enc grid)",
        vs="fix #1 B (wd_enc grid) (baseline)",
        **E.contrast_row(
            cands[("all_features", "fix #1 B (wd_enc grid)")],
            cands[("baseline", "fix #1 B (wd_enc grid)")],
        ),
    )
    con.append(r)
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
    sens = pd.concat([sensitivities(run, tag) for tag, run in walks.items()])
    spread = pd.DataFrame(
        [r for tag, run in walks.items() for r in E.seed_spread(run, tag)]
    )
    vs = pd.DataFrame(
        [r for tag, run in walks.items() for r in variance_share(run, tag)]
    )
    STUDY.mkdir(parents=True, exist_ok=True)
    scores.to_csv(STUDY / "scores.csv", index=False)
    con.to_csv(STUDY / "contrasts.csv", index=False)
    sel.to_csv(STUDY / "selection.csv", index=False)
    sens.to_csv(STUDY / "sensitivities.csv", index=False)
    spread.to_csv(STUDY / "seed_spread_2020-03-18.csv", index=False)
    vs.to_csv(STUDY / "variance_share.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(scores.round(4).to_string(index=False))
    print(con.round(4).to_string(index=False))
    print(sel.to_string(index=False))
    cols = [c for c in sens.columns if c.startswith("d_")]
    print(sens.groupby(["walk", "design"])[cols].mean().round(3).to_string())
    print(spread.to_string(index=False))
    print(vs.round(3).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gate-inputs":
        gate_inputs()
    elif cmd == "calib-inputs":
        calib_inputs()
    elif cmd == "gate":
        gate(Path(sys.argv[2]))
    elif cmd == "calib":
        calib(Path(sys.argv[2]))
    elif cmd == "walk-inputs":  # lr_lin wd_lins wd_encs wl_only wl_mid
        a = sys.argv[2:]
        walk_inputs(
            float(a[0]),
            [float(x) for x in a[1].split(",")],
            [float(x) for x in a[2].split(",")],
            [float(x) for x in a[3].split(",")],
            float(a[4]),
        )
    elif cmd == "study":
        study({a.split("=")[0]: Path(a.split("=")[1]) for a in sys.argv[2:]})
