"""The QLIKE round (user, 2026-10-15): the NN should conform to the story -- on QLIKE, all_features better than baseline,
as for the linear arms -- under a rule fixed in advance for both designs: QLIKE, the paper's evaluation loss.

  option 1  the configuration chosen by the lowest plain QLIKE of the seed-averaged validation forecast (phase2_select
            --criterion qlike: r = y^2 / f^2 on the 125 validation rows) on the existing tune outputs of the softplus
            walk of record, fix #1 B and (b) B; inside each tune job early stopping stays on the validation MSE (the
            saved validation forecasts are at the MSE-best epoch; each seed its own); only the changed refits re-run
  option 2  a (b)-style walk trained on QLIKE (loss=qlike, the trainer's kernel; qlike_eps / qlike_cap_q as defined)
            with early stopping on the validation QLIKE (native_g9 es=qlike) and the QLIKE selection

    python experiments/puffer_forecast/qlike_study.py gate-inputs        # _work/gate9/
    python experiments/puffer_forecast/qlike_study.py gate RUN
    python experiments/puffer_forecast/qlike_study.py calib-inputs       # _work/qcal/
    python experiments/puffer_forecast/qlike_study.py walk-inputs        # _work/walk_softplus_rinit_q/
    python experiments/puffer_forecast/qlike_study.py capped RUN [RUN..] # n_capped share of the QLIKE training steps
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
import ridge_init_study as RI  # noqa: E402
import wd_enc_study as E  # noqa: E402

WORK, RES, DATA = E.WORK, E.RES, E.DATA
SETS = E.SETS
OUT = RES / "qlike_study"


def gate_inputs() -> None:
    g = WORK / "gate9"
    src = [
        x
        for x in (WORK / "walk_softplus" / "tune_block0.txt").read_text().splitlines()
        if x.strip()
    ]
    LS.write(g, "tune_es.txt", [f"{x} es=qlike" for x in src])
    LS.write(
        g,
        "tune_qlike_es.txt",
        [f"{x.replace(' loss=mse ', ' loss=qlike ')} es=qlike" for x in src],
    )
    ok = (
        "mode=tune epochs=5 max_epochs=40 batch=32 lr=0.03 seed=5 max_grad_norm=1.0 data=data/toy w0=0 w1=390 L=6 H=8 "
        "layers=2 fc_n=10 val_tail=60 embargo=25 patience=10 momentum=0.95 anneal=1 min_lr_ratio=0.0 loss=mse "
        "link=softplus mask=0 xstd=1"
    )
    LS.write(g, "bad_es.txt", [f"id=bad es=foo {ok}"])
    LS.write(
        g,
        "bad_es_refit.txt",
        [f"id=bad es=qlike {ok}".replace("mode=tune", "mode=refit")],
    )
    LS.write(g, "es_pack.txt", [f"id=bad es=qlike {ok}"])
    LS.write(g, "ok_small.txt", [f"id=ok es=qlike {ok}"])
    print(sorted(p.name for p in g.iterdir()), len(src))


def es_ref(vals: np.ndarray, patience: int) -> tuple[int, float]:
    """The trainer's early-stopping rule on a per-epoch criterion: strictly lower, while e - best_epoch < patience."""
    best, be = np.inf, 0
    for e, v in enumerate(vals):
        if e - be < patience and np.isfinite(v) and v < best:
            best, be = float(v), e + 1
    return be, best


def gate(run: Path) -> dict:
    import gate4 as G4
    import gate5 as G5

    res: dict = {"env": (run / "env.txt").read_text(errors="replace").splitlines()[:3]}
    res["tune_g8_vs_g9"] = G4.identical(run / "id_old_tune", run / "id_new_tune")
    res["refit_g8_vs_g9"] = G4.identical(run / "id_old_refit", run / "id_new_refit")
    res["fct_g8_vs_g9"] = G4.identical(run / "id_old_fct", run / "id_new_fct")
    res["tune_g9_off_written_vs_absent"] = G4.identical(
        run / "id_new_tune", run / "id_new_off_tune"
    )
    res["es_tune_pool_vs_wave"] = G4.identical(
        run / "es_wave_tune", run / "es_tune", tune=True
    )
    gd, ok = G5.guards(run)
    res["guards"], res["guards_ok"] = gd.to_dict("records"), ok
    rows = []
    ys = {d: RI.Design(d).y for d in SETS}
    for part in ("es_tune", "es_qlike_tune"):
        jo = pd.read_csv(run / part / "jobs_out.csv").set_index("id")
        ep = pd.read_csv(run / part / "epochs.csv")
        pr = pd.read_csv(run / part / "preds.csv")
        pr = pr[pr["kind"] == "val"]
        n_be = n_bv = 0
        errs = []
        for jid, r in jo.iterrows():
            e = ep[ep["id"] == jid].sort_values("epoch")
            be, bv = es_ref(
                e["val_qlike"].to_numpy(np.float64), 20
            )  # the walk's patience
            n_be += int(be == int(r["best_epoch"]))
            n_bv += int(np.float32(bv) == np.float32(r["best_val_mse"]))
            d = "all_features" if "_all_features_" in jid else "baseline"
            p = pr[pr["id"] == jid].sort_values("row")
            y = ys[d][p["row"].to_numpy()]
            f = p["f"].to_numpy(np.float64)
            eps2 = 1e-6
            r_ = np.maximum(y * y, eps2) / (f * f + eps2)
            errs.append(abs(float(np.mean(r_ - np.log(r_) - 1)) - bv) / bv)
        rows.append(
            dict(
                part=part,
                jobs=len(jo),
                best_epoch_eq_val_qlike_argmin=n_be,
                best_val_eq_min_val_qlike=n_bv,
                val_preds_qlike_vs_logged_max_rel=float(max(errs)),
            )
        )
    res["es"] = rows
    out = RES / "review_build" / f"gate9_{run.name}.json"
    out.write_bytes(json.dumps(res, indent=1, default=str).encode())
    pd.set_option("display.width", 250)
    for k, v in res.items():
        if k.endswith(("_vs_g9", "_vs_absent", "_vs_wave")) or k == "guards_ok":
            print(k, v)
    print(gd.to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))
    return res


def calib_inputs() -> None:
    """Block-0 drift under QLIKE: (b)'s calibration jobs with loss=qlike (tunes es=qlike), lr_lin {0.0003, 0.001}."""
    d0 = WORK / "qcal"
    for name in ("drift_refit.txt", "drift_tune.txt"):
        src = (WORK / "rinit_calib" / name).read_text().splitlines()
        out = []
        for x in src:
            if not x.strip() or "_ll0.01_" in x:
                continue
            y = x.replace(" loss=mse ", " loss=qlike ")
            out.append(y + (" es=qlike" if " mode=tune " in y else ""))
        LS.write(d0, name, out)
        print(name, len(out))


def walk_inputs() -> None:
    """(b) B's tune lines trained on QLIKE: loss=qlike, early stopping es=qlike; the refits loss=qlike (common)."""
    src = WORK / "walk_softplus_rinit"
    d = WORK / "walk_softplus_rinit_q"
    for k in range(5):
        lines = [
            x
            for x in (src / f"tune_block{k}.txt").read_text().splitlines()
            if x.strip()
        ]
        assert all(" loss=mse " in x for x in lines)
        LS.write(
            d,
            f"tune_block{k}.txt",
            [f"{x.replace(' loss=mse ', ' loss=qlike ')} es=qlike" for x in lines],
        )
    m = json.loads((src / "blocks.json").read_text())
    m["common"]["loss"] = "qlike"
    (d / "blocks.json").write_bytes(json.dumps(m, indent=1).encode())
    print(d.name, len((d / "tune_block0.txt").read_text().splitlines()))


def capped(runs: list[Path]) -> pd.DataFrame:
    """Share of training samples whose QLIKE gradient hit the cap (steps.csv n_capped / the step's rows)."""
    rows = []
    for run in runs:
        for f in sorted(run.rglob("steps.csv.gz")):
            st = pd.read_csv(f)
            jo = pd.read_csv(f.parent / "jobs_out.csv").set_index("id")
            n = st.merge(jo[["N", "S", "batch"]], left_on="id", right_index=True)
            real = np.minimum(n["batch"], n["N"] - n["step"] * n["batch"])
            rows.append(
                dict(
                    run=run.name,
                    part=str(f.parent.relative_to(run)),
                    steps=len(st),
                    capped_share=float(st["n_capped"].sum() / real.sum()),
                    steps_with_any_capped=float((st["n_capped"] > 0).mean()),
                )
            )
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    return t


QSEL = WORK / "qsel"
WALK_RUNS = {
    "record": RES / "walks" / "runs" / "walk_softplus_13089746",
    "fix1": RES / "wd_enc_study" / "runs_final" / "walk_wegrid_13143151",
    "rinit": RES / "ridge_init_study" / "runs" / "walk_rinit_13148540",
}


def compose(tag: str, refit_root: Path) -> Path:
    """Option 1's walk: each block's QLIKE selection; the refits of the unchanged (block, design)s from the original
    walk, the changed ones from refit_root/refits_qsel_<tag>_b<k>_<job>/refit."""
    import shutil

    src = WALK_RUNS[tag]
    out = OUT / "runs" / f"qsel_{tag}"
    changed = {
        (c["walk"], c["block"]): c["changed"]
        for c in json.loads((QSEL / "changed.json").read_text())
    }
    out.mkdir(parents=True, exist_ok=True)
    for f in ("env.txt", "gpu_type.txt"):
        if (src / f).is_file():
            shutil.copyfile(src / f, out / f)
    for k in range(5):
        b = out / f"block{k}"
        (b / "tune").mkdir(parents=True, exist_ok=True)
        (b / "refit").mkdir(parents=True, exist_ok=True)
        for f in ("jobs_out.csv", "pool_summary.csv"):
            shutil.copyfile(src / f"block{k}" / "tune" / f, b / "tune" / f)
        shutil.copyfile(QSEL / f"{tag}_b{k}_sel.json", b / "selection.json")
        pr = pd.read_csv(src / f"block{k}" / "refit" / "preds.csv")
        jo = pd.read_csv(src / f"block{k}" / "refit" / "jobs_out.csv")
        ch = changed[(tag, k)]
        if ch:
            new = [
                p
                for p in refit_root.glob(f"refits_qsel_{tag}_b{k}_*")
                if (p / "refit" / "jobs_out.csv").is_file()
            ]
            assert len(new) == 1, (tag, k, new)
            nr = new[0]
            pat = "|".join(f"^refit_{d}_r" for d in ch)
            pr = pd.concat(
                [
                    pr[~pr["id"].str.contains(pat)],
                    pd.read_csv(nr / "refit" / "preds.csv"),
                ]
            )
            jo = pd.concat(
                [
                    jo[~jo["id"].str.contains(pat)],
                    pd.read_csv(nr / "refit" / "jobs_out.csv"),
                ]
            )
            for f in ("env.txt", "gpu_type.txt"):
                shutil.copyfile(nr / f, b / "refit" / f)
        pr.to_csv(b / "refit" / "preds.csv", index=False)
        jo.to_csv(b / "refit" / "jobs_out.csv", index=False)
        shutil.copyfile(
            src / f"block{k}" / "refit" / "pool_summary.csv",
            b / "refit" / "pool_summary.csv",
        )
    return out


def verdict(runs: dict[str, Path]) -> None:  # noqa: C901 -- one report
    """runs: label -> phase2_score forecasts.csv (both designs).  Per run: all_features vs baseline (QLIKE % and DM p),
    each design vs the enet and the stack's ridge, the day classes, Sharpe mid / crossed."""
    import feature_mcs_report as FMR

    sc = E._scoring()
    cl = E.day_classes()["day_class"].to_numpy()
    fcs = pd.read_parquet(
        FMR.OUT / "forecasts.parquet", columns=["est", "model", "row", "pred_adj"]
    )
    ref = {}
    for d in SETS:
        for est in ("enet", "ridge"):
            c = fcs[(fcs["est"] == est) & (fcs["model"] == d)].sort_values("row")
            p = np.full(sc["n"], np.nan)
            p[c["row"].to_numpy()] = c["pred_adj"].to_numpy(float)
            ref[(d, est)] = E.losses(p)
    S = sc["S"]
    rows, con = [], []
    for label, fc in runs.items():
        L = {d: E.losses(E.walk_pred(fc, d)) for d in SETS}
        for d in SETS:
            st = S.stats(L[d])
            rec = dict(
                run=label,
                design=d,
                qlike=st["qlike"],
                mse_fit=float(np.mean(L[d]["mse_fit"])),
                sharpe_mid=st["sharpe_mid"],
                sharpe_crossed=st["sharpe_crossed"],
            )
            for c in ("month_end", "extreme_inputs", "other"):
                rec[f"ql_{c}"] = float(L[d]["ql"][cl == c].mean())
            rows.append(rec)
            for est in ("enet", "ridge"):
                con.append(
                    dict(
                        run=label,
                        design=d,
                        vs=est,
                        **E.contrast_row(L[d], ref[(d, est)]),
                    )
                )
        con.append(
            dict(
                run=label,
                design="all_features",
                vs="baseline (same run)",
                **E.contrast_row(L["all_features"], L["baseline"]),
            )
        )
    for d in SETS:
        for est in ("enet", "ridge"):
            st = S.stats(ref[(d, est)])
            rec = dict(
                run=est,
                design=d,
                qlike=st["qlike"],
                mse_fit=float(np.mean(ref[(d, est)]["mse_fit"])),
                sharpe_mid=st["sharpe_mid"],
                sharpe_crossed=st["sharpe_crossed"],
            )
            for c in ("month_end", "extreme_inputs", "other"):
                rec[f"ql_{c}"] = float(ref[(d, est)]["ql"][cl == c].mean())
            rows.append(rec)
    for est in ("enet", "ridge"):
        con.append(
            dict(
                run=est,
                design="all_features",
                vs="baseline (same arm)",
                **E.contrast_row(ref[("all_features", est)], ref[("baseline", est)]),
            )
        )
    t, c = pd.DataFrame(rows), pd.DataFrame(con)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "scores.csv", index=False)
    c.to_csv(OUT / "contrasts.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(t.round(4).to_string(index=False))
    print(c.round(4).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gate-inputs":
        gate_inputs()
    elif cmd == "gate":
        gate(Path(sys.argv[2]))
    elif cmd == "calib-inputs":
        calib_inputs()
    elif cmd == "walk-inputs":
        walk_inputs()
    elif cmd == "compose":  # TAG REFIT_ROOT
        print(compose(sys.argv[2], Path(sys.argv[3])))
    elif cmd == "verdict":
        verdict({a.split("=")[0]: Path(a.split("=")[1]) for a in sys.argv[2:]})
    elif cmd == "capped":
        capped([Path(p) for p in sys.argv[2:]])
