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
    t["wall"] = t["pool_wall_s"].fillna(t["wave_train_wall_s"])
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


if __name__ == "__main__":
    {"gate4": gate4, "gate5": gate5}[sys.argv[1]](Path(sys.argv[2]))
