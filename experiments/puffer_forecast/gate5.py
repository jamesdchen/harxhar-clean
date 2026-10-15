"""Gate 4 (re-scoped: the four speed items) and gate 5 (the rest + softplus) analyses, 2026-10-13.

python experiments/puffer_forecast/gate5.py gate4 RUN     # -> results/.../review_build/<run>.json, _timing.csv
python experiments/puffer_forecast/gate5.py gate5 RUN
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import gate4 as G4  # noqa: E402
import link_check as LC  # noqa: E402
import overfit as O  # noqa: E402
import reference as RF  # noqa: E402

OUT = G4.OUT


def guards(run: Path) -> tuple[pd.DataFrame, bool]:
    lines = (run / "guards.csv").read_text().splitlines()[1:]
    g = pd.DataFrame(
        [
            dict(
                case=x.split(",", 2)[0],
                rc=int(x.split(",", 2)[1]),
                last_line=x.split(",", 2)[2],
            )
            for x in lines
        ]
    )
    g["expect_rc"] = np.where(g["case"] == "ok_small", 0, 2)
    return g, bool((g["rc"] == g["expect_rc"]).all())


def nsys_after_fill(run: Path, v: str) -> dict:
    f = run / f"nsys_{v}_cuda_api_trace.csv.gz"
    if not f.is_file():
        return {}
    tr = pd.read_csv(f)
    gl = tr[tr["Name"].str.startswith("cudaGraphLaunch")]["Start (ns)"].to_numpy()
    w = tr[(tr["Start (ns)"] > gl[499]) & (tr["Start (ns)"] < gl[-1])]
    w = w[
        w["Name"].str.match(
            r"cuda(Free|Malloc|DeviceSynchronize|Memset|StreamSynchronize)"
        )
    ]
    agg = w.groupby("Name")["Duration (ns)"].agg(["count", "sum"])
    return dict(
        window_s=round(float((gl[-1] - gl[499]) / 1e9), 2),
        calls={
            k: [int(r["count"]), round(r["sum"] / 1e6, 1)] for k, r in agg.iterrows()
        },
    )


def walls(run: Path) -> pd.DataFrame:
    t = G4.timing(run)
    t["wall"] = (
        t["pool_wall_s"].fillna(t["wave_train_wall_s"])
        if "wave_train_wall_s" in t
        else t["pool_wall_s"]
    )
    return t


def gate4(run: Path) -> None:
    res: dict = {"env": (run / "env.txt").read_text(errors="replace").splitlines()[:3]}
    res["tune_old_vs_new"] = G4.identical(run / "id_old_tune", run / "id_new_tune")
    for part in ("refit", "fct", "warm"):
        res[f"{part}_old_vs_new"] = G4.identical(
            run / f"id_old_{part}", run / f"id_new_{part}"
        )
    res["tune_new_pool_vs_wave"] = G4.identical(
        run / "id_new_wave_tune", run / "id_new_tune"
    )
    res["refit_new_pool_vs_wave"] = G4.identical(
        run / "id_new_wave_refit", run / "id_new_refit"
    )
    res["fct_new_pool_vs_wave"] = G4.identical(
        run / "id_new_wave_fct", run / "id_new_fct"
    )
    res["refit_new_vs_gather_old"] = G4.identical(
        run / "id_new_refit", run / "id_new_gold_refit"
    )
    res["refit_new_vs_c32"] = G4.identical(
        run / "id_new_refit", run / "id_new_c32_refit"
    )
    res["fct_trainrows_f64"] = O.trainrows_check(run / "id_new_fct").to_dict("records")
    for v in ("old", "new"):
        res[f"nsys_{v}_after_fill"] = nsys_after_fill(run, v)
    tm = walls(run)
    tm.to_csv(OUT / f"{run.name}_timing.csv", index=False)
    b = tm[tm["name"].str.startswith("tB_")].copy()
    b[["v", "design", "K", "rep"]] = b["name"].str.extract(
        r"tB_(old|gold|new|oldc32|newc32)_(baseline|all_features)_K(\d+)_r(\d)"
    )
    sb = b.groupby(["design", "K", "v"])["wall"].median().unstack("v")
    rows = []
    for (d, K), r in sb.iterrows():
        rows.append(
            dict(
                suite=f"B {d} K{K}",
                old=r["old"],
                item1=r["gold"],
                item1_2=r["new"],
                item3=r["oldc32"],
                item1_2_3=r["newc32"],
            )
        )
    p = tm[tm["name"].str.match(r"t[RT]_")].copy()
    p[["kind", "v"]] = p["name"].str.extract(
        r"t([RT])_(old|gold|new|oldc32|newc32)_r\d"
    )
    for k, g in p.groupby("kind"):
        m = g.groupby("v")["wall"].median()
        rows.append(
            dict(
                suite={
                    "R": "Phase 2c block-0 refits (2,500)",
                    "T": "Phase 2c block-0 tune (120)",
                }[k],
                old=m.get("old"),
                item1=m.get("gold"),
                item1_2=m.get("new"),
                item3=m.get("oldc32"),
                item1_2_3=m.get("newc32"),
            )
        )
    sp = pd.DataFrame(rows)
    for c in ("item1", "item1_2", "item3", "item1_2_3"):
        sp[f"{c}_vs_old_%"] = (sp[c] / sp["old"] - 1) * 100
    sp["item2_alone_%"] = (sp["item1_2"] / sp["item1"] - 1) * 100
    sp.to_csv(OUT / f"{run.name}_speedups.csv", index=False)
    res["speedups"] = sp.round(4).to_dict("records")
    (OUT / f"{run.name}.json").write_bytes(
        json.dumps(res, indent=1, default=str).encode()
    )
    pd.set_option("display.width", 250)
    for k, v in res.items():
        if k not in ("speedups", "fct_trainrows_f64", "env"):
            print(k, v)
    print(
        pd.DataFrame(res["fct_trainrows_f64"])[
            ["id", "rows_ok", "max_rel_diff_vs_f64", "fc_vs_trainpath_identical"]
        ]
    )
    print(sp.round(3).to_string(index=False))


def gate5(run: Path) -> None:
    res: dict = {"env": (run / "env.txt").read_text(errors="replace").splitlines()[:3]}
    res["tune_old_vs_new"] = G4.identical(
        run / "id_old_tune", run / "id_new_tune", tune=True
    )
    for part in ("refit", "fct", "warm"):
        res[f"{part}_old_vs_new"] = G4.identical(
            run / f"id_old_{part}", run / f"id_new_{part}"
        )
    res["refit_link_identity_written_vs_absent"] = G4.identical(
        run / "id_new_refit", run / "id_new_linkid_refit"
    )
    res["tune_new_wave_vs_pool"] = G4.identical(
        run / "id_new_wave_tune", run / "id_new_tune", tune=True
    )
    res["refit_new_pool_vs_wave"] = G4.identical(
        run / "id_new_wave_refit", run / "id_new_refit"
    )
    res["fct_new_pool_vs_wave"] = G4.identical(
        run / "id_new_wave_fct", run / "id_new_fct"
    )
    for m in ("tf32ns", "tf32"):
        res[f"refit_new_vs_{m}"] = G4.identical(
            run / "id_new_refit", run / f"id_new_{m}_refit"
        )
    res["fct_trainrows_f64"] = O.trainrows_check(run / "id_new_fct").to_dict("records")
    g, ok = guards(run)
    res["guards"], res["guards_ok"] = g.to_dict("records"), ok
    res["stop_loss"] = G4.stop_check(run)
    # softplus: loss kernel, gradients, overfit job, Phase 2c block 0
    lc = LC.losscheck(run)
    res["softplus_losscheck"] = lc.to_dict("records")
    gc = RF.gradcheck(run)
    res["softplus_gradcheck"] = gc.to_dict("records")
    jo = pd.read_csv(run / "sp_overfit" / "jobs_out.csv")
    pr = pd.read_csv(run / "sp_overfit" / "preds.csv")
    y = np.fromfile(O.WORK / "data" / "baseline" / "y.f32", "<f4").astype(np.float64)
    ov = []
    for _, r in jo.iterrows():
        tr = pr[(pr["id"] == r["id"]) & (pr["kind"] == "train")]
        yy, f = y[tr["row"].to_numpy()], tr["f"].to_numpy()
        ov.append(
            dict(
                id=r["id"],
                link=r["link"],
                one_minus_r2=float(np.mean((f - yy) ** 2) / np.var(yy)),
                min_f=float(f.min()),
                status=r["status"],
                nonfinite=int(r["nonfinite"]),
                t_train=float(r["t_train"]),
            )
        )
    res["softplus_overfit"] = ov
    p2 = []
    for link in ("identity", "softplus"):
        sel = json.loads((run / f"p2c_{link}" / "selection.json").read_text())
        tj = pd.read_csv(run / f"p2c_{link}" / "tune" / "jobs_out.csv")
        rj = pd.read_csv(run / f"p2c_{link}" / "refit" / "jobs_out.csv")
        tp = pd.read_csv(run / f"p2c_{link}" / "tune" / "preds.csv")
        rp = pd.read_csv(run / f"p2c_{link}" / "refit" / "preds.csv")
        for d in O.DESIGNS:
            tpd = tp[tp["id"].str.contains(f"_{d}_")]
            rpd = rp[rp["id"].str.contains(f"_{d}_")]
            p2.append(
                dict(
                    link=link,
                    design=d,
                    cfg=sel[d]["cfg"],
                    val_mse=sel[d]["val_mse"],
                    tune_status_ok=bool(
                        (tj[tj["id"].str.contains(f"_{d}_")]["status"] == "ok").all()
                    ),
                    refit_status_ok=bool(
                        (rj[rj["id"].str.contains(f"_{d}_")]["status"] == "ok").all()
                    ),
                    refits=int(rj["id"].str.contains(f"_{d}_").sum()),
                    min_f_val=float(tpd["f"].min()),
                    min_f_fc=float(rpd["f"].min()),
                    n_f_nonpositive_fc=int((rpd["f"] <= 0).sum()),
                    links_recorded=sorted(
                        set(rj[rj["id"].str.contains(f"_{d}_")]["link"])
                    ),
                )
            )
    res["softplus_phase2c_block0"] = p2
    tm = walls(run)
    tm.to_csv(OUT / f"{run.name}_timing.csv", index=False)
    res["timing"] = tm[["name", "rc", "wall", "process_wall_s"]].to_dict("records")
    (OUT / f"{run.name}.json").write_bytes(
        json.dumps(res, indent=1, default=str).encode()
    )
    pd.set_option("display.width", 250)
    for k, v in res.items():
        if k not in (
            "guards",
            "fct_trainrows_f64",
            "env",
            "softplus_losscheck",
            "softplus_gradcheck",
            "timing",
            "softplus_phase2c_block0",
            "softplus_overfit",
        ):
            print(k, v)
    print(g.to_string(index=False))
    print(lc.T.to_string())
    print(gc.to_string(index=False))
    print(pd.DataFrame(ov).to_string(index=False))
    print(pd.DataFrame(p2).to_string(index=False))
    print(tm[["name", "rc", "wall", "process_wall_s"]].to_string(index=False))


def gate5b(run: Path) -> None:
    res: dict = {"env": (run / "env.txt").read_text(errors="replace").splitlines()[:3]}
    res["tune_g4_vs_g5b"] = G4.identical(
        run / "id_old_tune", run / "id_new_tune", tune=True
    )
    for part in ("refit", "fct"):
        res[f"{part}_g4_vs_g5b"] = G4.identical(
            run / f"id_old_{part}", run / f"id_new_{part}"
        )
    tm = walls(run)
    tm.to_csv(OUT / f"{run.name}_timing.csv", index=False)
    p = tm[tm["name"].str.match(r"t[RT]_")].copy()
    p[["kind", "v"]] = p["name"].str.extract(r"t([RT])_(old|g5|new)_r\d")
    caps = []
    for _, r in p.iterrows():
        ps = pd.read_csv(run / r["name"] / "pool_summary.csv")
        caps.append(int(ps["n_capture"].sum()))
    p["n_capture"] = caps
    s = (
        p.groupby(["kind", "v"])
        .agg(wall_median=("wall", "median"), n_capture=("n_capture", "median"))
        .reset_index()
    )
    s["kind"] = s["kind"].map({"R": "refits (2,500)", "T": "tune (120)"})
    s["v"] = s["v"].map({"old": "native_g4", "g5": "native_g5", "new": "native_g5b"})
    res["walls"] = s.to_dict("records")
    (OUT / f"{run.name}.json").write_bytes(
        json.dumps(res, indent=1, default=str).encode()
    )
    for k, v in res.items():
        if k != "walls":
            print(k, v)
    print(s.to_string(index=False))


def mps(run: Path) -> None:
    """N trainer processes on one GPU (with / without MPS): walls and byte-identity against one process."""
    import pool_gate_report as PG

    t = pd.read_csv(run / "timing.csv", dtype={"rcs": str})
    ref = run / "plain_N1_K48" / "p0"
    pref = PG.read_sorted(ref, "preds.csv", ["id", "kind", "row"])
    sref = PG.read_sorted(ref, "steps.csv", ["id", "epoch", "step"])
    rows = []
    for _, r in t.iterrows():
        d = run / r["name"]
        parts = sorted(d.glob("p*/"))
        if set(str(r["rcs"])) != {"0"} or not all(
            (x / "preds.csv").is_file() for x in parts
        ):
            rows.append(
                dict(
                    name=r["name"],
                    mps=int(r["mps"]),
                    N=int(r["N"]),
                    K_each=int(r["K"]),
                    wall_s=float(r["wall_s"]),
                    rcs=str(r["rcs"]),
                    jobs=0,
                    preds_identical=False,
                    steps_identical=False,
                )
            )
            continue
        pr = pd.concat(
            [PG.read_sorted(x, "preds.csv", ["id", "kind", "row"]) for x in parts]
        )
        st = pd.concat(
            [PG.read_sorted(x, "steps.csv", ["id", "epoch", "step"]) for x in parts]
        )
        pr = pr.sort_values(["id", "kind", "row"]).reset_index(drop=True)
        st = st.sort_values(["id", "epoch", "step"]).reset_index(drop=True)
        rows.append(
            dict(
                name=r["name"],
                mps=int(r["mps"]),
                N=int(r["N"]),
                K_each=int(r["K"]),
                wall_s=float(r["wall_s"]),
                rcs=str(r["rcs"]),
                jobs=len(pr),
                preds_identical=bool(pr.equals(pref)),
                steps_identical=bool(
                    st[["id", "epoch", "step", "loss", "grad_norm"]].equals(
                        sref[["id", "epoch", "step", "loss", "grad_norm"]]
                    )
                ),
            )
        )
    df = pd.DataFrame(rows)
    base = float(df.loc[df["name"] == "plain_N1_K48", "wall_s"].iloc[0])
    df["vs_single_process_%"] = (df["wall_s"] / base - 1) * 100
    df.to_csv(OUT / f"{run.name}.csv", index=False)
    print((run / "env.txt").read_text(errors="replace"))
    pd.set_option("display.width", 220)
    print(df.round(2).to_string(index=False))


def walkcmp(new: Path, old: Path = O.RES / "phase2c" / "runs" / "13082282") -> None:
    """A full identity walk (native_g5b) against Phase 2c 13082282, block by block, every difference listed."""
    import pool_gate_report as PG

    out: dict = {
        "new": str(new),
        "old": str(old),
        "gpu_new": (new / "gpu_type.txt").read_text().strip(),
        "gpu_old": (old / "env.txt").read_text().splitlines()[2],
    }
    for k in range(5):
        bn, bo = new / f"block{k}", old / f"block{k}"
        r: dict = {}
        r["tune"] = G4.identical(bo / "tune", bn / "tune", tune=True)
        dj = O.diff_jobs(bo / "tune", bn / "tune")
        # tune jobs differing in preds (the log rows past the stop are expected to differ)
        pa = PG.read_sorted(bo / "tune", "preds.csv", ["id", "kind", "row"])
        pb = PG.read_sorted(bn / "tune", "preds.csv", ["id", "kind", "row"])
        m = pa.merge(
            pb,
            on=["id", "kind", "row"],
            how="outer",
            suffixes=("_o", "_n"),
            indicator=True,
        )
        r["tune_jobs_preds_differ"] = sorted(
            set(m[(m["_merge"] != "both") | (m["f_o"] != m["f_n"])]["id"])
        )
        ja = pd.read_csv(bo / "tune" / "jobs_out.csv").set_index("id")
        jb = pd.read_csv(bn / "tune" / "jobs_out.csv").set_index("id")
        r["tune_jobs_best_epoch_differ"] = sorted(
            ja.index[(ja["best_epoch"] != jb["best_epoch"].reindex(ja.index))]
        )
        r["tune_jobs_differ_from_first_step"] = dj["differing_from_first_step"]
        so = json.loads((bo / "selection.json").read_text())
        sn = json.loads((bn / "selection.json").read_text())
        r["selection"] = {
            d: dict(
                cfg_old=so[d]["cfg"],
                cfg_new=sn[d]["cfg"],
                best_epoch_same=so[d]["best_epoch"] == sn[d]["best_epoch"],
                val_mse_old=so[d]["val_mse"],
                val_mse_new=sn[d]["val_mse"],
            )
            for d in so
        }
        r["jobs_refit_identical"] = (bo / "jobs_refit.txt").read_bytes() == (
            bn / "jobs_refit.txt"
        ).read_bytes()
        r["refit"] = G4.identical(bo / "refit", bn / "refit")
        ra = PG.read_sorted(bo / "refit", "preds.csv", ["id", "kind", "row"])
        rb = PG.read_sorted(bn / "refit", "preds.csv", ["id", "kind", "row"])
        mm = ra.merge(
            rb,
            on=["id", "kind", "row"],
            how="outer",
            suffixes=("_o", "_n"),
            indicator=True,
        )
        bad = mm[(mm["_merge"] != "both") | (mm["f_o"] != mm["f_n"])]
        r["refit_jobs_preds_differ"] = sorted(set(bad["id"]))[:50]
        r["n_refit_jobs_preds_differ"] = int(bad["id"].nunique())
        out[f"block{k}"] = r
    (O.RES / "review_build" / f"walkcmp_{new.name}.json").write_bytes(
        json.dumps(out, indent=1, default=str).encode()
    )
    print(json.dumps(out, indent=1, default=str)[:12000])


def scorecmp(new_name: str, old_name: str = "phase2c") -> None:
    """phase2_score outputs of two walks: every CSV of the same name compared cell for cell."""
    a, b = O.RES / old_name, O.RES / new_name
    for f in ("scores.csv", "contrasts.csv", "selection.csv", "forecasts.csv"):
        x, y = pd.read_csv(a / f), pd.read_csv(b / f)
        same = (
            x.shape == y.shape
            and x.columns.equals(y.columns)
            and bool((x.astype(str) == y.astype(str)).all().all())
        )
        print(f, "identical" if same else "DIFFERENT", x.shape, y.shape)
        if not same and x.shape == y.shape:
            d = x.astype(str) != y.astype(str)
            print(
                pd.concat(
                    [x[d.any(axis=1)], y[d.any(axis=1)]], keys=["old", "new"]
                ).to_string()[:4000]
            )


if __name__ == "__main__":
    if sys.argv[1] == "walkcmp":
        walkcmp(Path(sys.argv[2]))
    elif sys.argv[1] == "scorecmp":
        scorecmp(sys.argv[2])
    else:
        {"gate4": gate4, "gate5": gate5, "gate5b": gate5b, "mps": mps}[sys.argv[1]](
            Path(sys.argv[2])
        )
