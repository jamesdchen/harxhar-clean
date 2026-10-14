"""Speed benchmark table (2026-10-13) from one run of cluster/puffer_forecast_bench_h2.sh.

Steady-state wall per variant (setup, CUDA-graph capture, compilation and warm-up excluded; reported apart):
  native     the sum of `wave_train_wall` over the run's waves (forecast_train.cu: from the first epoch
             launch of the wave, after every model is set up and captured, to the last model's stop)
  PyTorch    the sum of the per-job (suite A) or per-K (suite B) synchronised `t_train`
model-epochs/s = model-epochs / steady-state wall; ratio = variant wall / native wall on the same jobs at the
same K (> 1: slower than native).

    python experiments/puffer_forecast/bench_report.py BENCH_RUN_DIR
Writes results/puffer_forecast_2026-10-12/bench/{bench.csv, env.txt, torch_runs.csv, native_runs.csv}.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "puffer_forecast_2026-10-12" / "bench"


def native(run: Path, tag: str) -> dict | None:
    f = run / tag / "jobs_out.csv"
    if not f.is_file():
        return None
    j = pd.read_csv(f)
    waves = j.drop_duplicates(["wave_wall", "wave_train_wall"])
    return dict(
        epochs=int(j["epochs_run"].sum()),
        wall=float(waves["wave_train_wall"].sum()),
        setup_capture=float((j["t_setup"] + j["t_capture"]).sum()),
        models=len(j),
        waves=len(waves),
    )


def main() -> None:
    run = Path(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    rows, nat_rows = [], []
    nat = {}
    for tag, suite, K in [
        ("nA_k1", "A", 1),
        ("nA_k8", "A", 8),
        ("nA_k24", "A", 24),
        ("nA_k48", "A", 48),
        ("nA_k1_again", "A", 1),
    ] + [
        (f"nB_{d}_K{K}", f"B {d}", K)
        for d in ("baseline", "all_features")
        for K in (8, 24, 48)
    ]:
        r = native(run, tag)
        if r is None:
            continue
        nat_rows.append(dict(tag=tag, suite=suite, K=K, **r))
        if tag != "nA_k1_again":
            nat[(suite, K)] = r
        rows.append(
            dict(
                suite=suite,
                variant="native PufferLib 5.0"
                + (" (repeat)" if "again" in tag else ""),
                K=K,
                models=r["models"],
                model_epochs=r["epochs"],
                wall_s=r["wall"],
                model_epochs_per_s=r["epochs"] / r["wall"],
                warmup_s=r["setup_capture"],
            )
        )
    t = pd.read_csv(run / "torch" / "torch_runs.csv")
    ok = t[t["variant"].str.startswith("torch") & t["t_train"].notna()]
    for v, g in ok[ok["suite"] == "A"].groupby("variant"):
        rows.append(
            dict(
                suite="A",
                variant=v,
                K=1,
                models=len(g),
                model_epochs=int(g["epochs"].sum()),
                wall_s=float(g["t_train"].sum()),
                model_epochs_per_s=g["epochs"].sum() / g["t_train"].sum(),
                warmup_s=float(g["t_warmup"].sum()),
                jobs_done=f"{len(g)}/24",
            )
        )
    for _, r in ok[ok["suite"] == "B"].iterrows():
        rows.append(
            dict(
                suite=f"B {r['design']}",
                variant=r["variant"],
                K=int(r["K"]),
                models=int(r["K"]),
                model_epochs=int(r["epochs"]),
                wall_s=float(r["t_train"]),
                model_epochs_per_s=r["epochs"] / r["t_train"],
                warmup_s=float(r["t_warmup"]),
            )
        )
    tab = pd.DataFrame(rows)

    def ratio(r):
        n = nat.get((r["suite"], r["K"]))
        if n is None:
            return float("nan")
        if r["suite"] == "A" and r["K"] == 1 and r["model_epochs"] != n["epochs"]:
            # partial max-autotune: compare per model-epoch
            return (r["wall_s"] / r["model_epochs"]) / (n["wall"] / n["epochs"])
        return r["wall_s"] / n["wall"]

    tab["wall_vs_native_same_K"] = tab.apply(ratio, axis=1)
    tab = tab.sort_values(["suite", "K", "variant"])
    tab.to_csv(OUT / "bench.csv", index=False)
    pd.DataFrame(nat_rows).to_csv(OUT / "native_runs.csv", index=False)
    shutil.copy(run / "torch" / "torch_runs.csv", OUT / "torch_runs.csv")
    shutil.copy(run / "env.txt", OUT / "env.txt")
    for extra in ("torch/torch_info.json", "code_md5.txt"):
        if (run / extra).is_file():
            shutil.copy(run / extra, OUT / Path(extra).name)
    pd.set_option("display.width", 250)
    print((run / "env.txt").read_text())
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    errs = (
        t[t.get("error", pd.Series(dtype=str)).notna()]
        if "error" in t
        else pd.DataFrame()
    )
    if len(errs):
        print(
            "errors:\n",
            errs[
                ["suite", "variant", "id" if "id" in errs else "K", "error"]
            ].to_string(index=False),
        )
    chk = t[t["variant"] == "check_stacked_vs_single"]
    if len(chk):
        print(
            chk[["design", "max_rel_param_diff_after_1_epoch"]].to_string(index=False)
        )


if __name__ == "__main__":
    main()
