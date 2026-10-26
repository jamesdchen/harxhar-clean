"""RULE.txt 7(c): the pre-sample tuning check of the trees, scored on the 2013 validation year only (laptop).

Reads the day-refit pieces of each variant (<runs>/<variant>/pieces/*.parquet + .json: date, pred_adj, true_adj, B,
plain, rv_actual) and, per class, ranks the transferred configuration T and its one-factor variants by the validation
MSE on the adjusted scale (pred_adj against true_adj, every 2013 bar) and reports QLIKE (plain against the bar's RV,
RV > 0); T's seed runs (43-46) give its seed range.  "Within noise": the Diebold-Mariano test (dm_test, HLN, daily
average loss over the 2013 days) of the best variant against T has p >= 0.05 (MSE; QLIKE too).  Asserted: every variant
of a class forecasts the same 2013 bars, all of them in 2013 (no score on 2014 or later).

usage: python tune2013_score.py <main worktree (src.evaluation)> <runs dir> <out dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MAIN, RUNS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(MAIN))
from src.evaluation.diebold_mariano import dm_test  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
rows, data = [], {}
for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
    ps = sorted((d / "pieces").glob("*.parquet"))
    js = [p.with_suffix(".json") for p in ps]
    if len(ps) != 1 or not js[0].is_file():
        print(f"{d.name}: incomplete ({len(ps)} pieces)")
        continue
    f = pd.read_parquet(ps[0]).sort_values("date")
    m = json.loads(js[0].read_text())
    t = pd.DatetimeIndex(f["date"])
    assert (t.year == 2013).all(), d.name
    est = m["est"]
    name = d.name
    variant = "T" if name == f"{est}_T" else name.split("__", 1)[1] if "__" in name else name.replace(f"{est}_", "")
    se = (f["pred_adj"].to_numpy() - f["true_adj"].to_numpy()) ** 2
    rv = f["rv_actual"].to_numpy(float)
    ok = np.isfinite(rv) & (rv > 0)
    r = rv[ok] / f["plain"].to_numpy(float)[ok]
    q = np.full(len(f), np.nan)
    q[ok] = r - np.log(r) - 1.0
    data[(est, variant)] = pd.DataFrame({"se": se, "q": q}, index=t)
    rows.append(
        dict(
            est=est,
            variant=variant,
            config=m["config"],
            sha=m["sha"],
            seed=m.get("seed", 42),
            params=json.dumps(m.get("params")),
            n_bars=len(f),
            n_days=int(t.normalize().nunique()),
            refits=m["refits"],
            mse_adj=float(se.mean()),
            qlike=float(np.nanmean(q)),
            n_qlike=int(ok.sum()),
            wall_sec=m["wall_sec"],
            threads=m["threads"],
            cpu_hours=m["wall_sec"] * m["threads"] / 3600.0,
            cpus=",".join(m.get("cpus", [])),
        )
    )
R = pd.DataFrame(rows)
out_rows, verdict = [], {}
for est, g in R.groupby("est"):
    T = (est, "T")
    idx = data[T].index
    for k in [k for k in data if k[0] == est]:
        assert data[k].index.equals(idx), k
    seeds = g[g["variant"].str.startswith("T_seed")]
    cand = g[~g["variant"].str.startswith("T_seed")].sort_values("mse_adj").reset_index(drop=True)
    best = cand.iloc[0]
    day = pd.DatetimeIndex(idx.normalize())
    res: dict = dict(
        best=best["variant"],
        best_mse=best["mse_adj"],
        T_mse=float(g[g.variant == "T"].mse_adj.iloc[0]),
        T_rank_mse=int(np.flatnonzero(cand["variant"] == "T")[0]) + 1,
        n_candidates=len(cand),
        seed_mse_min=float(pd.concat([seeds.mse_adj, g[g.variant == "T"].mse_adj]).min()),
        seed_mse_max=float(pd.concat([seeds.mse_adj, g[g.variant == "T"].mse_adj]).max()),
    )
    for loss in ("se", "q"):
        a = data[(est, best["variant"])][loss].groupby(day).mean().to_numpy()
        b = data[T][loss].groupby(day).mean().to_numpy()
        if best["variant"] == "T":
            res[f"dm_{loss}"], res[f"p_{loss}"] = 0.0, 1.0
        else:
            t_ = dm_test(a, b)
            res[f"dm_{loss}"], res[f"p_{loss}"] = float(t_["dm"]), float(t_["p"])
    res["T_within_noise_mse"] = bool(res["p_se"] >= 0.05)
    res["best_within_T_seed_range"] = bool(res["seed_mse_min"] <= res["best_mse"] <= res["seed_mse_max"])
    res["rank_qlike_T"] = int(np.flatnonzero(cand.sort_values("qlike")["variant"].to_numpy() == "T")[0]) + 1
    verdict[est] = res
    cand = cand.assign(rank_mse=np.arange(1, len(cand) + 1), rel_mse_vs_T_pct=100 * (cand.mse_adj / res["T_mse"] - 1))
    out_rows.append(cand)
    out_rows.append(seeds.assign(rank_mse=np.nan, rel_mse_vs_T_pct=100 * (seeds.mse_adj / res["T_mse"] - 1)))
A = pd.concat(out_rows, ignore_index=True)
A.to_csv(OUT / "tune2013_scores.csv", index=False)
(OUT / "tune2013_verdict.json").write_text(json.dumps(verdict, indent=1))
L = ["# Trees, pre-sample tuning check (2013 validation year, day refits, cell-B design, full set)", ""]
for est, g in A.groupby("est", sort=False):
    v = verdict[est]
    L += [
        f"## {est}: best {v['best']} (MSE {v['best_mse']:.6f}); T rank {v['T_rank_mse']} of {v['n_candidates']} "
        f"(MSE {v['T_mse']:.6f}); DM best vs T: MSE {v['dm_se']:+.2f} (p {v['p_se']:.3f}), QLIKE {v['dm_q']:+.2f} "
        f"(p {v['p_q']:.3f}); T within noise: {v['T_within_noise_mse']}; T's seed range {v['seed_mse_min']:.6f} .. "
        f"{v['seed_mse_max']:.6f}",
        "",
        "| variant | seed | MSE adj | % vs T | QLIKE | rank (MSE) | CPU-h |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, x in g.iterrows():
        rk = "" if pd.isna(x.rank_mse) else int(x.rank_mse)
        L.append(
            f"| {x.variant} | {x.seed} | {x.mse_adj:.6f} | {x.rel_mse_vs_T_pct:+.2f} | {x.qlike:.5f} | {rk} | "
            f"{x.cpu_hours:.1f} |"
        )
    L.append("")
L.append(f"CPU-hours (wall x threads), all variants: {R.cpu_hours.sum():.1f}")
(OUT / "tune2013_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
