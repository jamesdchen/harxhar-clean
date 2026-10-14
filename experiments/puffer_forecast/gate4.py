"""Gate 4 (2026-10-13): the review build (native_of4) against the current binary (native_of3, 0e59b80c).

    python experiments/puffer_forecast/gate4.py RUN_DIR     # -> results/.../review_build/gate4.json, timing.csv

Identity (all new flags off): Phase 2c block-0 tune (preds and best_epoch identical; the new build never trains
past best_epoch + patience, so its log rows are the old rows minus the past-stop ones), 500 block-0 refits,
the fc_train jobs (+ train rows vs a float64 forward of the saved weights), a warm start, pool vs wave; the
guards; stop_loss (prefix identical, stop at the first chunk end under the threshold); wall-time A/Bs; nsys.
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
import pool_gate_report as G  # noqa: E402

OUT = O.RES / "review_build"


def frames(d: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pr = G.read_sorted(d, "preds.csv", ["id", "kind", "row"])
    st = (
        G.read_sorted(d, "steps.csv", ["id", "epoch", "step"])
        if (d / "steps.csv").exists() or (d / "steps.csv.gz").exists()
        else pd.DataFrame()
    )
    jo = pd.read_csv(d / "jobs_out.csv", dtype=str).set_index("id").sort_index()
    return pr, st, jo


def identical(a: Path, b: Path, tune: bool = False) -> dict:
    """b == a; for tune runs b's step rows may be a's minus the rows past the stop (b never trains past it)."""
    pa, sa, ja = frames(a)
    pb, sb, jb = frames(b)
    out = dict(
        jobs=len(ja),
        preds=len(pa),
        preds_identical=bool(pa.equals(pb)),
        best_epoch_identical=bool(ja["best_epoch"].equals(jb["best_epoch"])),
    )
    k = ["id", "epoch", "step"]
    v = ["loss", "grad_norm", "n_capped", "n_nonfinite"]
    m = sb.merge(sa, on=k, how="left", suffixes=("_b", "_a"), indicator=True)
    out["b_steps_all_in_a"] = bool((m["_merge"] == "both").all())
    out["common_steps_identical"] = bool(
        all((m[c + "_b"] == m[c + "_a"]).all() for c in v)
    )
    out["steps_a"], out["steps_b"] = len(sa), len(sb)
    if tune:
        # the rows a has and b lacks must all be past the stop (epoch > best_epoch + patience)
        mm = sa.merge(sb[k], on=k, how="left", indicator=True)
        extra = mm[mm["_merge"] == "left_only"]
        be = ja["best_epoch"].astype(int)
        past = (
            extra["epoch"].astype(int).to_numpy()
            > be.reindex(extra["id"]).to_numpy() + 20
        )
        out["dropped_rows"] = len(extra)
        out["dropped_rows_all_past_stop"] = bool(past.all())
        out["epochs_run_a"], out["epochs_run_b"] = (
            int(ja["epochs_run"].astype(int).sum()),
            int(jb["epochs_run"].astype(int).sum()),
        )
        out["b_stops_at_best_plus_patience"] = bool(
            (
                jb["epochs_run"].astype(int)
                == np.minimum(jb["best_epoch"].astype(int) + 20, 200)
            ).all()
        )
    else:
        out["steps_identical"] = bool(
            len(sa) == len(sb)
            and out["common_steps_identical"]
            and out["b_steps_all_in_a"]
        )
        out["epochs_run_identical"] = bool(ja["epochs_run"].equals(jb["epochs_run"]))
    return out


def stop_check(run: Path) -> dict:
    jo = pd.read_csv(run / "stop" / "jobs_out.csv").set_index("id")
    st = O.read_csv_any(run / "stop" / "steps.csv")
    a, b = "of_baseline_H64_Ly1_L5_lr0.01_stop", "of_baseline_H64_Ly1_L5_lr0.01_nostop"
    E = int(jo.loc[a, "epochs_run"])
    N, B, sd = int(jo.loc[a, "N"]), int(jo.loc[a, "batch"]), float(jo.loc[a, "sd_y"])
    sa = st[st["id"] == a].sort_values(["epoch", "step"])
    sb = st[(st["id"] == b) & (st["epoch"] <= E)].sort_values(["epoch", "step"])
    same = bool(
        np.array_equal(
            sa[["epoch", "step", "loss", "grad_norm"]].to_numpy(),
            sb[["epoch", "step", "loss", "grad_norm"]].to_numpy(),
        )
    )
    w = np.minimum(B, N - st["step"].to_numpy() * B)
    st = st.assign(wl=st["loss"] * w)
    rel = st[st["id"] == b].groupby("epoch")["wl"].sum() / N / sd**2
    ends = [e for e in range(10, 1001, 10)]
    first = next((e for e in ends if rel.get(e, np.inf) <= 1e-2), None)
    return dict(
        stop_epochs_run=E,
        nostop_epochs_run=int(jo.loc[b, "epochs_run"]),
        prefix_identical=same,
        first_chunk_end_under_threshold=first,
        stops_there=bool(first == E),
        rel_at_stop=float(rel.get(E, np.nan)),
        stop_loss=float(jo.loc[a, "stop_loss"]),
    )


def timing(run: Path) -> pd.DataFrame:
    t = pd.read_csv(run / "timing.csv")
    rows = []
    for _, r in t.iterrows():
        n = r["name"]
        d = run / n
        rec = dict(name=n, rc=int(r["rc"]), process_wall_s=float(r["wall_s"]))
        if (d / "pool_summary.csv").is_file():
            ps = pd.read_csv(d / "pool_summary.csv")
            rec.update(
                pool_wall_s=float(ps["wall"].sum()),
                sum_setup_s=float(ps["sum_setup"].sum()),
                n_capture=int(ps["n_capture"].sum()),
            )
        elif (d / "jobs_out.csv").is_file():
            jo = pd.read_csv(d / "jobs_out.csv")
            rec.update(
                wave_train_wall_s=float(
                    jo.drop_duplicates("wave_wall")["wave_train_wall"].sum()
                ),
                wave_wall_s=float(jo.drop_duplicates("wave_wall")["wave_wall"].sum()),
            )
        rows.append(rec)
    return pd.DataFrame(rows)


def main(run: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"env": (run / "env.txt").read_text(errors="replace").splitlines()[:3]}
    res["tune_old_vs_new"] = identical(
        run / "id_old_tune", run / "id_new_tune", tune=True
    )
    res["tune_new_wave_vs_new_pool"] = identical(
        run / "id_new_wave_tune", run / "id_new_tune", tune=True
    )
    for part in ("refit", "fct", "warm"):
        res[f"{part}_old_vs_new"] = identical(
            run / f"id_old_{part}", run / f"id_new_{part}"
        )
    res["refit_new_pool_vs_wave"] = identical(
        run / "id_new_refit", run / "id_new_wave_refit"
    )
    res["fct_new_pool_vs_wave"] = identical(run / "id_new_fct", run / "id_new_wave_fct")
    for m in ("tf32ns", "tf32"):
        res[f"refit_new_vs_{m}"] = identical(
            run / "id_new_refit", run / f"id_new_{m}_refit"
        )
    res["fct_trainrows_f64"] = O.trainrows_check(run / "id_new_fct").to_dict("records")
    keep = sorted((run / "id_new_fct" / "params").glob("*.keep.i32"))
    res["keep_files_written"] = len(keep)
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
    res["guards"] = g.to_dict("records")
    res["guards_ok"] = bool((g["rc"] == g["expect_rc"]).all())
    res["stop_loss"] = stop_check(run)
    tm = timing(run)
    tm.to_csv(OUT / f"{run.name}_timing.csv", index=False)
    for v in ("old", "new"):
        f = run / f"nsys_{v}_cuda_api_sum.csv"
        if f.is_file():
            api = pd.read_csv(f)
            name = [c for c in api.columns if c.lower() in ("name", "api name")][0]
            calls = [c for c in api.columns if "Calls" in c or "Num Calls" in c][0]
            tot = [c for c in api.columns if "Total Time" in c][0]
            sel = api[
                api[name].str.contains(
                    "cudaFree|cudaMalloc|cudaDeviceSynchronize|cudaStreamSynchronize|"
                    "cudaMemset|cudaGraphLaunch|cudaStreamQuery",
                    regex=True,
                )
            ]
            res[f"nsys_{v}"] = sel[[name, calls, tot]].to_dict("records")
    for v in (
        "old",
        "new",
    ):  # device-wide-sync-type calls after the pool's fill (from graph launch 500 on)
        f = run / f"nsys_{v}_cuda_api_trace.csv.gz"
        if f.is_file():
            tr = pd.read_csv(f)
            gl = tr[tr["Name"].str.startswith("cudaGraphLaunch")][
                "Start (ns)"
            ].to_numpy()
            if len(gl) > 500:
                w = tr[(tr["Start (ns)"] > gl[499]) & (tr["Start (ns)"] < gl[-1])]
                w = w[
                    w["Name"].str.match(
                        r"cuda(Free|Malloc|DeviceSynchronize|Memset|StreamSynchronize)"
                    )
                ]
                agg = w.groupby("Name")["Duration (ns)"].agg(["count", "sum"])
                res[f"nsys_{v}_after_fill"] = dict(
                    window_s=float((gl[-1] - gl[499]) / 1e9),
                    calls={
                        k: [int(r["count"]), round(r["sum"] / 1e6, 1)]
                        for k, r in agg.iterrows()
                    },
                )
    (OUT / f"{run.name}.json").write_bytes(
        json.dumps(res, indent=1, default=str).encode()
    )
    print(
        json.dumps(
            {k: v for k, v in res.items() if k not in ("guards", "fct_trainrows_f64")},
            indent=1,
            default=str,
        )[:9000]
    )
    print(g.to_string(index=False))
    pd.set_option("display.width", 220)
    print(tm.to_string(index=False))


def tf32ab(run: Path) -> None:
    """H1024 Ly2 L20 30 epochs: fp32 vs --tf32-ns vs --tf32 (in-sample 1-R2, forecast differences, walls)."""
    OUT.mkdir(parents=True, exist_ok=True)
    tm = pd.read_csv(run / "timing.csv").set_index("name")["wall_s"]
    tabs = {}
    for m in ("fp32", "tf32-ns", "tf32"):
        jo = pd.read_csv(run / f"ab_{m}" / "jobs_out.csv").set_index("id")
        pr = pd.read_csv(run / f"ab_{m}" / "preds.csv")
        tabs[m] = (jo, pr)
    rows = []
    ref_jo, ref_pr = tabs["fp32"]
    for m, (jo, pr) in tabs.items():
        for jid in jo.index:
            design = "all_features" if "_all_features_" in jid else "baseline"
            y = np.fromfile(O.WORK / "data" / design / "y.f32", "<f4").astype(
                np.float64
            )
            tr = pr[(pr["id"] == jid) & (pr["kind"] == "train")].sort_values("row")
            yy, f = y[tr["row"].to_numpy()], tr["f"].to_numpy()
            both = pr[(pr["id"] == jid) & pr["kind"].isin(["train", "fc"])].sort_values(
                ["kind", "row"]
            )
            rb = ref_pr[
                (ref_pr["id"] == jid) & ref_pr["kind"].isin(["train", "fc"])
            ].sort_values(["kind", "row"])
            fa, fr = both["f"].to_numpy(), rb["f"].to_numpy()
            rows.append(
                dict(
                    math=m,
                    id=jid,
                    math_col=jo.loc[jid, "math"],
                    one_minus_r2=float(np.mean((f - yy) ** 2) / np.var(yy)),
                    max_rel_diff_vs_fp32=float(np.max(np.abs(fa - fr) / np.abs(fr))),
                    max_abs_diff_over_sd=float(np.max(np.abs(fa - fr)) / np.std(yy)),
                    t_train_s=float(jo.loc[jid, "t_train"]),
                    process_wall_s=float(tm[m]),
                    nonfinite=int(jo.loc[jid, "nonfinite"]),
                )
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tf32ab.csv", index=False)
    env = (run / "env.txt").read_text(errors="replace").splitlines()[:3]
    print(env)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] == "tf32ab":
        tf32ab(Path(sys.argv[2]))
    else:
        main(Path(sys.argv[1]))
