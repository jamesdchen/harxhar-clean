"""Encoder-input decay wd_enc (user, 2026-10-15; fix 1 of the all_features diagnosis, af_diagnosis/summary.md).

The trainer (native_g7) applies W_enc[:, c] <- W_enc[:, c] (1 - lr_t wd_enc pf[c]) to the encoder's input columns
just before each Muon step (after wd's decay when both are set), pf[c] the regular stack's penalty factor of the
c-th kept design column: close_linear_pretune.structure(Xa[window], bb_aug, R_BB) -- 1 for an exogenous column,
R_BB = 0.01 for a HAR + calendar backbone column (pf does not depend on the window for R_BB > 0; the window only
masks columns, and the trainer feeds its own window mask's kept columns).  The encoder's constant-input (bias)
column, the decoder and the MinGRU matrices are never touched.  pf comes from <data>/enc_pf.f64 (one float64 per
design column), mapped through the window's keep set by the trainer.

    python experiments/puffer_forecast/wd_enc_study.py pf                  # data/<design>/enc_pf.f64, toy dirs
    python experiments/puffer_forecast/wd_enc_study.py gate-inputs         # _work/gate7/
    python experiments/puffer_forecast/wd_enc_study.py gate RUN            # gate-7 checks
    python experiments/puffer_forecast/wd_enc_study.py calib RUN           # encoder column norms vs wd_enc
    python experiments/puffer_forecast/wd_enc_study.py walk-inputs W1,W2,..  # _work/walk_softplus_we<w>/, _wegrid/
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import overfit as O  # noqa: E402

REPO = O.REPO
for _p in (
    REPO,
    REPO / "experiments",
    REPO / "notebooks",
    REPO / "specs",
    REPO / "writeup",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
WORK = O.WORK
RES = O.RES
DATA = WORK / "data"
SETS = ("baseline", "all_features")
TOY_PF = np.array(
    [1.0, 0.3, 0.1, 0.01, 0.0, 2.0]
)  # the toy's 6 columns: proportionality, a zero, a factor > 1
TOY = (
    "data=data/toy w0=0 w1=390 L=6 H=8 layers=2 fc_n=10 val_tail=60 embargo=25 patience=40 momentum=0.95 "
    "anneal=1 min_lr_ratio=0.0 loss=mse link=softplus mask=0 xstd=1"
)
AF = (
    "data=data/all_features w0=381 w1=2381 L=5 H=16 layers=1 fc_n=1 val_tail=125 embargo=25 patience=20 "
    "momentum=0.95 anneal=1 min_lr_ratio=0.0 loss=mse link=softplus mask=1 xstd=0"
)


def write(d: Path, name: str, lines: list[str]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_bytes(("\n".join(lines) + "\n").encode())


def blocks() -> list[dict]:
    return json.loads((WORK / "phase2c" / "blocks.json").read_text())["blocks"]


def arms_structure(design: str, w0: int, w1: int):
    """The linear arms' (locked, maskout, pf) of the design's window [w0, w1) (R_BB), on the arms' own arrays."""
    import close_linear_pretune as CLP
    import feature_mcs_runner as FR

    if design not in _ARMS:
        _ARMS[design] = FR.model_arrays(FR.load_inputs()["models"][design]["cols"])
    A = _ARMS[design]
    return CLP.structure(A["bar_Xa"][w0:w1], A["bb_aug"], CLP.R_BB)


_ARMS: dict = {}


def pf() -> None:
    """data/<design>/enc_pf.f64 from the linear arms' structure (R_BB = 0.01) on every block's tune window (pf must
    be the same on all of them), the toy's factors, and the guard directories."""
    import close_linear_pretune as CLP
    import feature_mcs_runner as FR

    inp = FR.load_inputs()
    for d in SETS:
        cols = inp["models"][d]["cols"]
        A = FR.model_arrays(cols)
        X = np.fromfile(DATA / d / "X.f32", "<f4").reshape(-1, len(cols))
        assert np.array_equal(X, A["bar_Xa"][:, :-1].astype("<f4")), (
            d
        )  # the NN's design = the arms' design
        pfs = [
            CLP.structure(
                A["bar_Xa"][b["tune_w0"] : b["tune_w1"]], A["bb_aug"], CLP.R_BB
            )[2]
            for b in blocks()
        ]
        assert all(
            np.array_equal(pfs[0], x) for x in pfs
        )  # R_BB > 0: pf does not depend on the window
        p = pfs[0]
        assert p[-1] == 0.0  # the intercept: locked
        bb = A["bb_aug"][:-1]
        assert np.all(p[:-1][bb] == CLP.R_BB) and np.all(p[:-1][~bb] == 1.0)
        p[:-1].astype("<f8").tofile(DATA / d / "enc_pf.f64")
        print(d, len(cols), "backbone", int(bb.sum()), "exogenous", int((~bb).sum()))
    TOY_PF.astype("<f8").tofile(DATA / "toy" / "enc_pf.f64")
    # guard directories: the toy without / with a wrong-length / a negative / a NaN pf file
    for name, v in (
        ("toy_nopf", None),
        ("toy_badpf", TOY_PF[:5]),
        ("toy_negpf", np.where(TOY_PF == 0.1, -1.0, TOY_PF)),
        ("toy_nanpf", np.where(TOY_PF == 0.1, np.nan, TOY_PF)),
    ):
        g = DATA / name
        g.mkdir(exist_ok=True)
        for f in ("X.f32", "y.f32", "meta.txt"):
            shutil.copyfile(DATA / "toy" / f, g / f)
        if v is not None:
            np.asarray(v, "<f8").tofile(g / "enc_pf.f64")
        elif (g / "enc_pf.f64").exists():
            (g / "enc_pf.f64").unlink()


def gate_inputs() -> None:
    g = WORK / "gate7"
    # one Muon step (batch >= N: one step per epoch, epochs=1): wd_enc 0 and > 0, wd and wd_enc together; plus the
    # gradcheck dumps (initial weights, gradient) of the same step for the float64 reference
    step = [
        f"id=step_toy_we0 mode=refit epochs=1 max_epochs=40 batch=512 lr=0.05 seed=5 max_grad_norm=1.0 wd_enc=0 save_params=1 {TOY}",
        f"id=step_toy_we0.5 mode=refit epochs=1 max_epochs=40 batch=512 lr=0.05 seed=5 max_grad_norm=1.0 wd_enc=0.5 save_params=1 {TOY}",
        f"id=step_toy_wd0.5_we0.5 mode=refit epochs=1 max_epochs=40 batch=512 lr=0.05 seed=5 max_grad_norm=1.0 wd=0.5 wd_enc=0.5 save_params=1 {TOY}",
        f"id=step_af_we0 mode=refit epochs=1 max_epochs=200 batch=2048 lr=0.03 seed=20261012 max_grad_norm=1.0 wd_enc=0 save_params=1 {AF}",
        f"id=step_af_we0.3 mode=refit epochs=1 max_epochs=200 batch=2048 lr=0.03 seed=20261012 max_grad_norm=1.0 wd_enc=0.3 save_params=1 {AF}",
    ]
    write(g, "step.txt", step)
    write(
        g,
        "step_grad.txt",
        [x.replace("save_params=1 ", "").replace("id=step_", "id=grad_") for x in step],
    )
    # zero gradient (max_grad_norm=0: Muon's clip coefficient 0, update 0): 40 epochs of decay alone
    zg = [
        f"id=zg_toy_we{w:g}{'_wd0.3' if wd else ''} mode=refit epochs=40 max_epochs=40 batch=32 lr=0.05 seed=5 "
        f"max_grad_norm=0 wd={wd:g} wd_enc={w:g} save_params=1 {TOY}"
        for w, wd in ((0, 0), (0.5, 0), (2, 0), (0.5, 0.3))
    ]
    write(g, "zerograd.txt", zg)
    # the toy with its gradient: encoder column norms against wd_enc, 40 epochs
    write(
        g,
        "norm_toy.txt",
        [
            f"id=norm_toy_we{w:g} mode=refit epochs=40 max_epochs=40 batch=32 lr=0.03 seed=5 max_grad_norm=1.0 "
            f"wd_enc={w:g} save_params=1 {TOY}"
            for w in (0, 0.3, 1, 3, 10)
        ],
    )
    # pf through keep: refit windows of both designs (every block's tune window, three refit windows)
    bl = blocks()
    wins = [(b["tune_w0"], b["tune_w1"]) for b in bl] + [
        (o - 2000, o)
        for o in (bl[0]["origins"][0], bl[2]["origins"][125], bl[4]["origins"][-1])
    ]
    pm = []
    for d in SETS:
        for w0, w1 in wins:
            pm.append(
                f"id=pfmap_{d}_{w0}_{w1} mode=refit data=data/{d} w0={w0} w1={w1} L=5 H=16 layers=1 fc_n=1 "
                "epochs=1 max_epochs=200 batch=128 lr=0.03 seed=20261012 momentum=0.95 max_grad_norm=1.0 anneal=1 "
                "min_lr_ratio=0.0 loss=mse link=softplus mask=1 xstd=0 wd_enc=0.1 save_params=1"
            )
    write(g, "pfmap.txt", pm)
    ok = f"mode=refit epochs=5 max_epochs=40 batch=32 lr=0.03 seed=5 max_grad_norm=1.0 {TOY}".replace(
        "link=softplus", "link=identity"
    )
    write(g, "bad_we_neg.txt", [f"id=bad wd_enc=-0.1 {ok}"])
    write(g, "bad_we_nan.txt", [f"id=bad wd_enc=nan {ok}"])
    for name in ("nopf", "badpf", "negpf", "nanpf"):
        write(
            g,
            f"bad_we_{name}.txt",
            [
                f"id=bad wd_enc=0.1 {ok}".replace(
                    "data=data/toy ", f"data=data/toy_{name} "
                )
            ],
        )
    write(g, "we_pack.txt", [f"id=bad wd_enc=0.1 {ok}"])
    write(g, "ok_small.txt", [f"id=ok wd_enc=0.1 {ok}"])
    print(sorted(p.name for p in g.iterdir()), len(pm))


CALIB_CFG = (
    (20, 16),
    (20, 64),
    (5, 16),
    (5, 64),
)  # lr 0.03: the selected rate of every all_features block
CALIB_WE = (0, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)


def calib_inputs() -> None:
    lines = []
    for d in SETS:
        for L, H in CALIB_CFG:
            for w in CALIB_WE:
                for seed in (20261012, 20261013):
                    lines.append(
                        f"id=cal_{d}_L{L}_H{H}_lr0.03_we{w:g}_s{seed} data=data/{d} mode=refit w0=381 w1=2381 L={L} "
                        f"H={H} lr=0.03 wd_enc={w:g} epochs=10 seed={seed} save_params=1 layers=1 batch=128 "
                        "max_epochs=200 patience=20 val_tail=125 embargo=25 momentum=0.95 max_grad_norm=1.0 anneal=1 "
                        "min_lr_ratio=0.0 fc_n=1 mask=1 xstd=0 loss=mse link=softplus"
                    )
    write(WORK / "wd_enc_calib", "calib.txt", lines)
    print(len(lines))


def _f32(p: Path) -> np.ndarray:
    return np.fromfile(p, "<f4").astype(np.float64)


def design_pf(d: str) -> np.ndarray:
    return TOY_PF if d == "toy" else np.fromfile(DATA / d / "enc_pf.f64", "<f8")


def enc_mask(P: int, H: int, layers: int) -> np.ndarray:
    """Flat mask of the encoder's input columns (every encoder entry but its last, constant-input column)."""
    import reference as RF

    w = RF.unpack(np.zeros(H * P + 2 * H + layers * 3 * H * H), P, H, layers)
    w["We"][:, :-1] = 1.0
    return RF.pack(w) > 0


def decayed(
    w0: np.ndarray, P: int, H: int, layers: int, f_wd: float, f_enc: np.ndarray
) -> np.ndarray:
    """W0 times the decay factors (float64): f_wd on every matrix but the encoder's bias column, f_enc[c] on the
    encoder's input column c (on top of f_wd)."""
    import reference as RF

    w = RF.unpack(w0.copy(), P, H, layers)
    w["We"][:, :-1] *= f_wd * f_enc[None, :]
    w["Wd"] = w["Wd"] * f_wd
    w["Wl"] = [x * f_wd for x in w["Wl"]]
    return RF.pack(w)


def gate(run: Path) -> dict:  # noqa: C901 -- one report
    import gate4 as G4
    import gate5 as G5
    import reference as RF

    res: dict = {"env": (run / "env.txt").read_text(errors="replace").splitlines()[:3]}
    res["tune_g6_vs_g7"] = G4.identical(run / "id_old_tune", run / "id_new_tune")
    res["refit_g6_vs_g7"] = G4.identical(run / "id_old_refit", run / "id_new_refit")
    res["fct_g6_vs_g7"] = G4.identical(run / "id_old_fct", run / "id_new_fct")
    res["tune_g7_pool_vs_wave"] = G4.identical(
        run / "id_new_wave_tune", run / "id_new_tune"
    )
    res["refit_g7_pool_vs_wave"] = G4.identical(
        run / "id_new_wave_refit", run / "id_new_refit"
    )
    res["tune_g7_wd_enc0_vs_absent"] = G4.identical(
        run / "id_new_tune", run / "id_new_we0_tune"
    )
    g, ok = G5.guards(run)
    res["guards"], res["guards_ok"] = g.to_dict("records"), ok
    # ---- one Muon step vs float64: W_ref = Muon(decay(W0)); W_we - W_0 = the decay term exactly on the encoder's
    # input columns, 0 (bit for bit) on its bias column, the decoder and the MinGRU matrices
    gc = run / "gradcheck"
    steps = []
    for tag, d, lr, wd, we, ref0 in (
        ("toy_we0.5", "toy", 0.05, 0.0, 0.5, "toy_we0"),
        ("toy_wd0.5_we0.5", "toy", 0.05, 0.5, 0.5, "toy_we0"),
        ("af_we0.3", "all_features", 0.03, 0.0, 0.3, "af_we0"),
    ):
        meta = json.loads((gc / f"grad_{tag}.json").read_text())
        P, H, ly = meta["P"], meta["H"], meta["layers"]
        w0 = _f32(gc / f"grad_{tag}.params0.f32")
        gr = _f32(gc / f"grad_{tag}.grads.f32")
        keep = np.fromfile(gc / f"grad_{tag}.keep.i32", "<i4")
        used = np.fromfile(gc / f"grad_{tag}.encpf.f32", "<f4")
        pfv = design_pf(d)[keep]
        n_we = _f32(run / "step" / "params" / f"step_{tag}.f32")
        n_0 = _f32(run / "step" / "params" / f"step_{ref0}.f32")
        dec = decayed(w0, P, H, ly, 1 - lr * wd, 1 - lr * we * pfv)
        w = RF.unpack(dec.copy(), P, H, ly)
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
        em = enc_mask(P, H, ly)
        diff, expect = n_we - n_0, dec - w0
        rec = dict(
            job=f"{tag} lr={lr}",
            P=P,
            n_encoder_input=int(em.sum()),
            n_other=int((~em).sum()),
            pf_used_eq_design_pf_through_keep=bool(
                np.array_equal(used, np.append(pfv, 0.0).astype("<f4"))
            ),
            native_vs_f64_max_abs_over_max=float(
                np.max(np.abs(n_we - ref)) / np.max(np.abs(ref))
            ),
            decay_term_max_abs=float(np.max(np.abs(expect))),
            decay_term_max_abs_err=float(np.max(np.abs(diff - expect))),
        )
        if wd == 0:
            rec["bias_decoder_mingru_bit_identical_vs_wd_enc0"] = bool(
                np.all(diff[~em] == 0.0)
            )
        steps.append(rec)
    res["one_step"] = steps
    # ---- zero gradient (max_grad_norm = 0): 40 epochs of decay alone; every encoder input column = W0 times
    # prod_t (1 - lr_t wd)(1 - lr_t wd_enc pf_c), everything else W0 (wd = 0) or W0 prod_t (1 - lr_t wd)
    zg = run / "zerograd"
    jo = pd.read_csv(zg / "jobs_out.csv").set_index("id")
    ep = pd.read_csv(zg / "epochs.csv")
    meta = json.loads((gc / "grad_toy_we0.5.json").read_text())
    P, H, ly = meta["P"], meta["H"], meta["layers"]
    w_init = _f32(gc / "grad_toy_we0.5.params0.f32")
    base = _f32(zg / "params" / "zg_toy_we0.f32")
    em = enc_mask(P, H, ly)
    w0u = RF.unpack(w_init, P, H, ly)
    zrows = [
        dict(
            job="zg_toy_we0",
            final_bit_identical_to_initial=bool(np.array_equal(base, w_init)),
        )
    ]
    for jid, we, wd in (
        ("zg_toy_we0.5", 0.5, 0.0),
        ("zg_toy_we2", 2.0, 0.0),
        ("zg_toy_we0.5_wd0.3", 0.5, 0.3),
    ):
        fin = _f32(zg / "params" / f"{jid}.f32")
        S = int(jo.loc[jid, "S"])
        lrs = ep[ep["id"] == jid].sort_values("epoch")["lr"].to_numpy(float)
        f_wd = float(np.prod((1 - lrs * wd) ** S))
        f_enc = np.array([np.prod((1 - lrs * we * p) ** S) for p in TOY_PF])
        pred = decayed(w_init, P, H, ly, f_wd, f_enc)
        wf = RF.unpack(fin, P, H, ly)
        ratio = np.linalg.norm(wf["We"][:, :-1], axis=0) / np.linalg.norm(
            w0u["We"][:, :-1], axis=0
        )
        rec = dict(job=jid, wd_enc=we, wd=wd, S=S, epochs=len(lrs))
        big = np.abs(pred) > 1e-30
        rec["max_rel_err_vs_predicted"] = float(
            np.max(np.abs(fin[big] - pred[big]) / np.abs(pred[big]))
        )
        for c, p in enumerate(TOY_PF):
            rec[f"pf{p:g}_norm_ratio"] = float(ratio[c])
            rec[f"pf{p:g}_predicted"] = float(f_enc[c] * f_wd)
        rec["bias_column_bit_identical_to_initial"] = bool(
            np.array_equal(wf["We"][:, -1], w0u["We"][:, -1])
        )
        if wd == 0:
            rec["decoder_mingru_bias_bit_identical_vs_wd_enc0"] = bool(
                np.array_equal(fin[~em], base[~em])
            )
        zrows.append(rec)
    res["zero_grad"] = zrows
    # ---- the toy with its gradient: encoder column norms against wd_enc (40 epochs)
    nt = []
    for we in (0, 0.3, 1, 3, 10):
        w = RF.unpack(
            _f32(run / "norm" / "params" / f"norm_toy_we{we:g}.f32"), P, H, ly
        )
        cn = np.linalg.norm(w["We"][:, :-1], axis=0)
        rec = {"wd_enc": we}
        rec.update({f"col_pf{p:g}": float(cn[c]) for c, p in enumerate(TOY_PF)})
        rec.update(
            bias_col=float(np.linalg.norm(w["We"][:, -1])),
            decoder=float(np.linalg.norm(w["Wd"])),
            mingru=float(np.sqrt(sum(np.sum(x**2) for x in w["Wl"]))),
        )
        nt.append(rec)
    nt = pd.DataFrame(nt)
    res["toy_norms"] = nt.to_dict("records")
    # ---- pf through keep, column for column, against the linear arms' structure on the same window
    pm = []
    jo = pd.read_csv(run / "pfmap" / "jobs_out.csv")
    for jid in jo["id"]:
        d = "all_features" if "all_features" in jid else "baseline"
        w0_, w1_ = map(int, jid.rsplit("_", 2)[1:])
        keep = np.fromfile(run / "pfmap" / "params" / f"{jid}.keep.i32", "<i4")
        used = np.fromfile(run / "pfmap" / "params" / f"{jid}.encpf.f32", "<f4")
        _, mk, pfa = arms_structure(d, w0_, w1_)
        arms_keep = np.flatnonzero(~mk[:-1].astype(bool))
        pm.append(
            dict(
                design=d,
                w0=w0_,
                w1=w1_,
                n_kept=len(keep),
                kept_set_eq_arms=bool(np.array_equal(keep, arms_keep)),
                pf_eq_arms_column_for_column=bool(
                    np.array_equal(used[:-1], pfa[keep].astype("<f4"))
                    and used[-1] == 0.0
                    and pfa[-1] == 0.0
                ),
                n_pf_1=int((used[:-1] == 1.0).sum()),
                n_pf_0p01=int((used[:-1] == np.float32(0.01)).sum()),
            )
        )
    res["pf_map"] = pm
    # ---- trim: the pooled wd-grid block-0 tune
    rc = pd.read_csv(run / "rc.csv").set_index("name")
    mem = pd.read_csv(run / "gpu_mem.log", header=None, names=["ts", "mib"])
    ts = pd.to_datetime(mem["ts"].str.strip(), format="%Y/%m/%d %H:%M:%S.%f")
    mem["t"] = ts.dt.tz_localize("America/Los_Angeles").astype("int64") // 10**9
    tr = []
    for name in ("trim_new_baseline", "trim_new_all_features", "trim_old_baseline"):
        r = rc.loc[name]
        log = (run / f"{name}.log").read_text(errors="replace").splitlines()
        m = mem[(mem["t"] >= r["t0"]) & (mem["t"] <= r["t1"])]["mib"]
        tr.append(
            dict(
                run=name,
                rc=int(r["rc"]),
                wall_s=int(r["t1"] - r["t0"]),
                gpu_mem_peak_mib=int(m.max()) if len(m) else -1,
                trims=sum("did not fit; pool trimmed" in x for x in log),
                last_line=log[-1] if log else "",
            )
        )
    res["trim"] = tr
    out = RES / "review_build" / f"gate7_{run.name}.json"
    out.write_bytes(json.dumps(res, indent=1, default=str).encode())
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    for k, v in res.items():
        if k.endswith(("_vs_g7", "_vs_wave", "_vs_absent")) or k == "guards_ok":
            print(k, v)
    print(g.to_string(index=False))
    for k in ("one_step", "zero_grad", "pf_map", "trim"):
        print(pd.DataFrame(res[k]).to_string(index=False))
    print(nt.round(4).to_string(index=False))
    return res


def trim_vs_walk(run: Path, walk_tune: Path) -> dict:
    """The gate's native_g7 pooled wd-grid tune (one pool per design) against the grid walk's block-0 tune (native_g6,
    one pool per design x wd): every job's validation forecasts, epoch log and best epoch."""
    import pool_gate_report as G

    out = {}
    pw = G.read_sorted(walk_tune, "preds.csv", ["id", "kind", "row"])
    ew = G.read_sorted(walk_tune, "epochs.csv", ["id", "epoch"])
    jw = pd.read_csv(walk_tune / "jobs_out.csv", dtype=str).set_index("id").sort_index()
    for d in ("baseline", "all_features"):
        t = run / f"trim_new_{d}"
        pg = G.read_sorted(t, "preds.csv", ["id", "kind", "row"])
        eg = G.read_sorted(t, "epochs.csv", ["id", "epoch"])
        jg = pd.read_csv(t / "jobs_out.csv", dtype=str).set_index("id").sort_index()
        ids = set(jg.index)
        a = pw[pw["id"].isin(ids)].reset_index(drop=True)
        b = ew[ew["id"].isin(ids)].reset_index(drop=True)
        out[d] = dict(
            jobs=len(jg),
            preds_identical=bool(a.equals(pg.reset_index(drop=True))),
            epochs_identical=bool(b.equals(eg.reset_index(drop=True))),
            best_epoch_identical=bool(
                jw.loc[jg.index, "best_epoch"].equals(jg["best_epoch"])
            ),
            best_val_identical=bool(
                jw.loc[jg.index, "best_val_mse"].equals(jg["best_val_mse"])
            ),
        )
    print(out)
    return out


def shrink(lrs: np.ndarray, S: int, we: float, p: float) -> float:
    return float(np.prod((1 - lrs * we * p) ** S))


def calib(run: Path) -> pd.DataFrame:
    import reference as RF

    c = run / "calib"
    jo = pd.read_csv(c / "jobs_out.csv").set_index("id")
    ep = pd.read_csv(c / "epochs.csv")
    rows = []
    for jid, r in jo.iterrows():
        d = "all_features" if "_all_features_" in jid else "baseline"
        keep = np.fromfile(c / "params" / f"{jid}.keep.i32", "<i4")
        P, H = len(keep) + 1, int(r["H"])
        w = RF.unpack(_f32(c / "params" / f"{jid}.f32"), P, H, 1)
        pfv = design_pf(d)[keep]
        We = w["We"][:, :-1]
        lrs = ep[ep["id"] == jid].sort_values("epoch")["lr"].to_numpy(float)
        rows.append(
            dict(
                design=d,
                L=int(r["L"]),
                H=H,
                seed=int(r["seed"]),
                wd_enc=float(r["wd_enc"]),
                S=int(r["S"]),
                fro_exog=float(np.linalg.norm(We[:, pfv == 1.0]))
                if (pfv == 1.0).any()
                else np.nan,
                fro_backbone=float(np.linalg.norm(We[:, pfv < 1.0])),
                fro_bias_col=float(np.linalg.norm(w["We"][:, -1])),
                fro_layer=float(np.linalg.norm(w["Wl"][0])),
                fro_decoder=float(np.linalg.norm(w["Wd"])),
                shrink_pf1=shrink(lrs, int(r["S"]), float(r["wd_enc"]), 1.0),
                shrink_pf0p01=shrink(lrs, int(r["S"]), float(r["wd_enc"]), 0.01),
            )
        )
    t = pd.DataFrame(rows)
    key = ["design", "L", "H", "seed"]
    b = t[t["wd_enc"] == 0].set_index(key)
    for col in ("fro_exog", "fro_backbone", "fro_layer", "fro_decoder"):
        t[f"{col}_rel"] = [
            r[col] / b.loc[tuple(r[k] for k in key), col] for _, r in t.iterrows()
        ]
    out = RES / "wd_enc_study"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "calibration_norms.csv", index=False)
    pd.set_option("display.width", 250)
    for col in ("fro_exog_rel", "fro_backbone_rel", "fro_layer_rel"):
        print(col)
        print(
            t.pivot_table(index=["design", "L", "H"], columns="wd_enc", values=col)
            .round(3)
            .to_string()
        )
    print(
        t[t["design"] == "all_features"]
        .groupby("wd_enc")[["shrink_pf1", "shrink_pf0p01"]]
        .mean()
        .to_string()
    )
    return t


def walk_inputs(grid: list[float]) -> None:
    """A: one walk per fixed wd_enc on all_features; B: wd_enc in each block's tune grid ({0} + grid), both designs.
    The Phase 2c protocol and 12-configuration grid with link=softplus (the walk of record), wd_enc added."""
    src = WORK / "phase2c"
    m = json.loads((src / "blocks.json").read_text())
    for we in grid:
        d = WORK / f"walk_softplus_we{we:g}"
        for k in range(5):
            lines = [
                x
                for x in (src / f"tune_block{k}.txt").read_text().splitlines()
                if "data=data/all_features " in x
            ]
            write(
                d,
                f"tune_block{k}.txt",
                [f"{x} link=softplus wd_enc={we:g}" for x in lines],
            )
        mm = json.loads(json.dumps(m))
        mm["designs"] = ["all_features"]
        mm["common"]["link"] = "softplus"
        mm["common"]["wd_enc"] = we
        (d / "blocks.json").write_bytes(json.dumps(mm, indent=1).encode())
    d = WORK / "walk_softplus_wegrid"
    for k in range(5):
        out = []
        for x in (src / f"tune_block{k}.txt").read_text().splitlines():
            if not x.strip():
                continue
            jid = x.split()[0]
            stem, seed = jid.rsplit("_s", 1)
            for we in [0.0] + list(grid):
                out.append(
                    x.replace(jid, f"{stem}_we{we:g}_s{seed}", 1)
                    + f" link=softplus wd_enc={we:g}"
                )
        write(d, f"tune_block{k}.txt", out)
    mm = json.loads(json.dumps(m))
    mm["common"]["link"] = "softplus"
    (d / "blocks.json").write_bytes(json.dumps(mm, indent=1).encode())
    print(
        {
            p.name: len((p / "tune_block0.txt").read_text().splitlines())
            for p in sorted(WORK.glob("walk_softplus_we*"))
        }
    )


STUDY = RES / "wd_enc_study"
REF_WALK = (
    RES / "walk_softplus_p100"
)  # the softplus walk of record (P100): forecasts.csv design,row,mean,count,std
DIAG = RES / "af_diagnosis"
HARD_DAY = "2020-03-18 16:00:00"
FLAGS = ("is_month_end", "is_quarter_end", "is_opex")
HAR = ("har_ma_1", "har_ma_5", "har_ma_25")


def _scoring() -> dict:
    import close_linear_pretune as CLP
    import feature_mcs_runner as FR

    if "S" not in _SC:
        inp = json.loads((FR.INP / "inputs.json").read_text())
        W = int(inp["W"])
        _SC.update(
            S=CLP.Scorer(),
            W=W,
            n=len(inp["dates"]) - W,
            y_adj=np.load(FR.INP / "bar_y.npy")[W:],
            day_row=np.load(FR.INP / "ref_armf.npz")["day_row"],
        )
    return _SC


_SC: dict = {}


def day_classes() -> pd.DataFrame:
    """The diagnosis day classes on the 866 trade days (af_diagnosis d04 / d13): month_end (is_month_end on the
    16:00 row), else extreme_inputs (>= 10 of the window mask's kept columns with |x| beyond their max |x| over the
    2,000-session window), else other."""
    f = STUDY / "day_class.csv"
    if f.is_file():
        return pd.read_csv(f)
    import feature_mcs_runner as FR

    from src.models.window_mask import window_keep

    sc = _scoring()
    W = sc["W"]
    cols = list(FR.load_inputs()["models"]["all_features"]["cols"])
    n, p = map(int, (DATA / "all_features" / "meta.txt").read_text().split())
    X = np.fromfile(DATA / "all_features" / "X.f32", "<f4").reshape(n, p)
    X64 = np.fromfile(DATA / "all_features" / "X.f64", "<f8").reshape(n, p)
    me = cols.index("is_month_end")
    rows = []
    for day, o in zip(sc["S"].days, sc["day_row"] + W):
        k = window_keep(X64[o - W : o])
        mx = np.abs(X[o - W : o][:, k].astype(np.float64)).max(0)
        n_gt = int((np.abs(X[o, k].astype(np.float64)) > mx).sum())
        c = (
            "month_end"
            if X[o, me] > 0
            else ("extreme_inputs" if n_gt >= 10 else "other")
        )
        rows.append(dict(day=str(day)[:10], row=int(o), n_gt_max=n_gt, day_class=c))
    t = pd.DataFrame(rows)
    STUDY.mkdir(parents=True, exist_ok=True)
    t.to_csv(f, index=False)
    return t


def walk_pred(forecasts_csv: Path, design: str) -> np.ndarray:
    sc = _scoring()
    e = pd.read_csv(forecasts_csv)
    e = e[e["design"] == design]
    pred = np.full(sc["n"], np.nan)
    pred[e["row"].to_numpy() - sc["W"]] = e["mean"].to_numpy()
    return pred


def enet_pred(design: str) -> np.ndarray:
    import feature_mcs_report as FMR

    sc = _scoring()
    fcs = pd.read_parquet(
        FMR.OUT / "forecasts.parquet", columns=["est", "model", "row", "pred_adj"]
    )
    c = fcs[(fcs["est"] == "enet") & (fcs["model"] == design)].sort_values("row")
    p = np.full(sc["n"], np.nan)
    p[c["row"].to_numpy()] = c["pred_adj"].to_numpy(float)
    return p


def losses(pred: np.ndarray) -> dict:
    import feature_mcs_report as FMR

    sc = _scoring()
    return FMR.cand_losses(sc["S"], pred, sc["y_adj"], sc["day_row"])


def contrast_row(g: dict, h: dict) -> dict:
    import atm_straddle_lib as asl

    from src.evaluation.diebold_mariano import dm_test

    S = _scoring()["S"]
    c = S.contrast(g, h)
    dm_mse = dm_test(g["mse_fit"], h["mse_fit"])
    t_x, _ = asl.newey_west_t(g["x"] - h["x"])
    return dict(
        qlike_pct=c["dq_pct"],
        p_qlike=c["dm_p"],
        mse_pct=float(100 * (np.mean(g["mse_fit"]) / np.mean(h["mse_fit"]) - 1)),
        p_mse=float(dm_mse["p"]),
        hac_t_mid=c["hac_t_mid"],
        hac_t_crossed=float(t_x),
    )


def sensitivities(run: Path, tag: str) -> pd.DataFrame:
    """af_diagnosis d06 on a walk's saved first-origin models (each block, every seed): the forecast change from a
    calendar flag switched on at the last step (rows where it is 0) and from a +1 move of each HAR column, on the
    last 500 rows of the training window; encoder column norms (backbone / exogenous)."""
    sys.path.insert(0, str(DIAG / "scripts"))
    import feature_mcs_runner as FR
    import netlib as NL

    models = FR.load_inputs()["models"]
    out = []
    for blk, b in enumerate(blocks()):
        o0 = b["origins"][0]
        rd = run / f"block{blk}" / "refit"
        jo = pd.read_csv(rd / "jobs_out.csv")
        for m in SETS:
            cols = list(models[m]["cols"])
            n, p = map(int, (DATA / m / "meta.txt").read_text().split())
            XX = np.fromfile(DATA / m / "X.f32", "<f4").reshape(n, p)
            bbm = design_pf(m) < 1.0
            for sd in range(20261012, 20261017):
                jid = f"refit_{m}_r{o0}_s{sd}"
                if not (rd / "params" / f"{jid}.f32").is_file():
                    continue
                M = NL.load_model(rd / "params", jid, jo)
                keep = list(M["keep"])
                x = NL.seqs(XX, M["keep"], np.arange(o0 - 500, o0), M["L"])
                f0, _, _ = NL.forecast(M, x)
                rec = dict(
                    walk=tag,
                    block=blk,
                    design=m,
                    seed=sd,
                    ep=int(M["jo"]["epochs_run"]),
                    L=M["L"],
                    H=M["H"],
                )
                for fl in FLAGS:
                    c = keep.index(cols.index(fl))
                    x1 = x.copy()
                    sel = x1[:, -1, c] == 0
                    x1[:, -1, c] = 1.0
                    f1, _, _ = NL.forecast(M, x1)
                    rec[f"d_{fl}"] = float(np.mean((f1 - f0)[sel]))
                for h in HAR:
                    c = keep.index(cols.index(h))
                    x1 = x.copy()
                    x1[:, -1, c] += 1.0
                    f1, _, _ = NL.forecast(M, x1)
                    rec[f"d_{h}"] = float(np.mean(f1 - f0))
                cn = np.linalg.norm(M["net"]["We"][:, :-1], axis=0)
                kb = bbm[M["keep"]]
                rec["enc_colnorm_bb_med"] = float(np.median(cn[kb]))
                rec["enc_colnorm_exog_med"] = (
                    float(np.median(cn[~kb])) if (~kb).any() else np.nan
                )
                rec["enc_fro_exog"] = float(np.sqrt((cn[~kb] ** 2).sum()))
                rec["enc_fro_bb"] = float(np.sqrt((cn[kb] ** 2).sum()))
                out.append(rec)
    return pd.DataFrame(out)


def seed_spread(run: Path, tag: str, day: str = HARD_DAY) -> list[dict]:
    dates = pd.read_csv(WORK / "dates.csv")
    row = int(dates.loc[dates["date"] == day, "row"].iloc[0])
    y = np.fromfile(DATA / "all_features" / "y.f32", "<f4")[row]
    out = []
    for b in sorted(run.glob("block*")):
        pr = pd.read_csv(b / "refit" / "preds.csv")
        pr = pr[(pr["kind"] == "fc") & (pr["row"] == row)]
        for m in SETS:
            f = (
                pr[pr["id"].str.startswith(f"refit_{m}_")]
                .sort_values("id")["f"]
                .to_numpy()
            )
            if len(f):
                out.append(
                    dict(
                        walk=tag,
                        design=m,
                        day=day[:10],
                        y=float(y),
                        f_seeds=[round(float(v), 3) for v in f],
                        f_mean=float(f.mean()),
                        f_sd=float(f.std(ddof=1)),
                        f_min=float(f.min()),
                        f_max=float(f.max()),
                    )
                )
    return out


def study(walks: dict[str, Path]) -> None:  # noqa: C901 -- one report
    """walks: tag -> walk run dir (we<x> on all_features, wegrid on both designs); each scored by phase2_score into
    wd_enc_study/score_<tag>/ first."""
    cls = day_classes()
    cl = cls["day_class"].to_numpy()
    print(cls["day_class"].value_counts().to_dict())
    cands: dict = {}
    for m in SETS:
        cands[(m, "softplus walk of record")] = losses(
            walk_pred(REF_WALK / "forecasts.csv", m)
        )
        cands[(m, "enet")] = losses(enet_pred(m))
    for tag in walks:
        fc = STUDY / f"score_{tag}" / "forecasts.csv"
        for m in pd.read_csv(fc)["design"].unique():
            cands[(m, f"wd_enc {tag}")] = losses(walk_pred(fc, m))
    S = _scoring()["S"]
    rows = []
    for (m, name), d in cands.items():
        st = S.stats(d)
        rec = dict(
            design=m,
            model=name,
            qlike=st["qlike"],
            mse_fit=float(np.mean(d["mse_fit"])),
            sharpe_mid=st["sharpe_mid"],
            sharpe_crossed=st["sharpe_crossed"],
            pct_buy=st["pct_buy"],
        )
        for c in ("month_end", "extreme_inputs", "other"):
            rec[f"ql_{c}"] = float(d["ql"][cl == c].mean())
        rows.append(rec)
    scores = pd.DataFrame(rows)
    con = []
    for (m, name), g in cands.items():
        if not name.startswith("wd_enc"):
            continue
        if m == "all_features":
            refs = [
                ("all_features", "enet"),
                ("baseline", "softplus walk of record"),
                ("all_features", "softplus walk of record"),
            ]
        else:
            refs = [("baseline", "enet"), ("baseline", "softplus walk of record")]
        for rm, rn in refs:
            h = cands[(rm, rn)]
            r = dict(design=m, model=name, vs=f"{rn} ({rm})", **contrast_row(g, h))
            dq = g["ql"] - h["ql"]
            for c in ("month_end", "extreme_inputs", "other"):
                r[f"gap_share_{c}"] = float(dq[cl == c].sum() / dq.sum())
            con.append(r)
    con = pd.DataFrame(con)
    sel = []
    for tag, run in walks.items():
        for b in sorted(run.glob("block*")):
            for m, v in json.loads((b / "selection.json").read_text()).items():
                sel.append(
                    dict(
                        walk=tag,
                        block=int(b.name[5:]),
                        design=m,
                        cfg=v["cfg"],
                        val_mse=v["val_mse"],
                        best_epochs=json.dumps(v["best_epoch"]),
                    )
                )
    sel = pd.DataFrame(sel)
    sens = pd.concat([sensitivities(run, tag) for tag, run in walks.items()])
    spread = pd.DataFrame(
        [r for tag, run in walks.items() for r in seed_spread(run, tag)]
    )
    STUDY.mkdir(parents=True, exist_ok=True)
    scores.to_csv(STUDY / "scores.csv", index=False)
    con.to_csv(STUDY / "contrasts.csv", index=False)
    sel.to_csv(STUDY / "selection.csv", index=False)
    sens.to_csv(STUDY / "sensitivities.csv", index=False)
    spread.to_csv(STUDY / "seed_spread_2020-03-18.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(scores.round(4).to_string(index=False))
    print(con.round(3).to_string(index=False))
    print(sel.to_string(index=False))
    cols = [
        "d_is_month_end",
        "d_har_ma_1",
        "d_har_ma_5",
        "enc_colnorm_bb_med",
        "enc_colnorm_exog_med",
        "enc_fro_exog",
        "enc_fro_bb",
        "ep",
    ]
    print(sens.groupby(["walk", "design"])[cols].mean().round(3).to_string())
    print(spread.to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pf":
        pf()
    elif cmd == "gate-inputs":
        gate_inputs()
    elif cmd == "calib-inputs":
        calib_inputs()
    elif cmd == "gate":
        gate(Path(sys.argv[2]))
    elif cmd == "trim-vs-walk":
        trim_vs_walk(Path(sys.argv[2]), Path(sys.argv[3]))
    elif cmd == "calib":
        calib(Path(sys.argv[2]))
    elif cmd == "day-classes":
        print(day_classes()["day_class"].value_counts())
    elif cmd == "study":
        study({a.split("=")[0]: Path(a.split("=")[1]) for a in sys.argv[2:]})
    elif cmd == "walk-inputs":
        walk_inputs([float(x) for x in sys.argv[2].split(",")])
