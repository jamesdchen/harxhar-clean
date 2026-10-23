"""Read-only analysis of the W = 3,000 tuning outputs (2026-10-23, coordinator: why does the HAR pick drift to the
grid's edges?).  No GPU job; reads the completed tune runs and salvaged copies (collect_jobs) of one window.

Per set (HAR benchmark, full set), over every configuration with all 5 seeds complete:
  (1) the 5 seed QLIKEs (each seed's own validation forecast) and the SE of their mean; the QLIKE of the 5-seed-mean
      forecast (the selection criterion); a day-block bootstrap SE (the validation days of fold 3, moving blocks of
      10 days, 2,000 resamples, seed 20261023) of the QLIKE difference between every configuration and the pick
      (loss r - log r - 1, r = y^2 / f^2, bars with y > 0; the difference of the bar-weighted means)
  (2) the configurations within 1 and 2 SE of the pick, and the cheapest among them: cost = P100-seconds a refit
      ~ (P100-s an epoch of its (H, L) cell, measured on the tune runs: pool-phase seconds / epochs run) x the mean
      of its 5 seeds' refit epochs
  (3) the linear shortcut alone (epoch 0: the decoder starts at zero, so the forecast is softplus(c + s v.x) with the
      ridge init v of the configuration's (L, wd_lin), c = softplus^-1(mu_y), s = sd_y / sigmoid(c)): its validation
      QLIKE, and mean |f_net - f_shortcut| / f_shortcut of the configuration's seed-mean forecast
  (4) the refit epochs per seed of the top 20

    PMF_WINDOW_DAYS=3000 python pooled_mingru_final_tune_analysis.py <root> <out dir>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(ROOT / "code"))
import pooled_mingru_final_2014_w3000 as m  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
POOL_RE = re.compile(r"pool phase \d+: (\d+) jobs on \d+ slots in ([0-9.]+) s")
CFG_RE = re.compile(r"^L(\d+)_H(\d+)_")
B, BLOCK, SEED = 2000, 10, 20261023


def epoch_cost(s: str) -> dict[tuple[int, int], float]:
    """P100-s an epoch of a tune job, by (H, L): pool-phase seconds / epochs run, median over the set's runs."""
    sub = {}
    for ln in (ROOT / "inputs" / "submitted.txt").read_text().splitlines():
        p = ln.split()
        if len(p) >= 2:
            sub[p[1]] = p[0]
    cells: dict[tuple[int, int], list[float]] = {}
    for d in (ROOT / "runs").glob(f"tune_{s}_r*_*"):
        log, jo = d / "tune.log", d / "tune" / "jobs_out.csv"
        if not log.is_file() or not jo.is_file():
            continue
        pools = [
            p
            for p in (POOL_RE.search(x) for x in log.read_text().splitlines())
            if p is not None
        ]
        if not pools:
            continue
        j = pd.read_csv(jo, usecols=["id", "L", "H", "epochs_run"])
        sec = sum(float(p.group(2)) for p in pools)
        for (H, L), g in j.groupby(["H", "L"]):
            cells.setdefault((int(H), int(L)), []).append(
                sec * len(g) / len(j) / g["epochs_run"].sum()
            )
    return {k: float(np.median(v)) for k, v in cells.items()}


def main() -> int:
    data, runs = ROOT / "data", ROOT / "runs"
    rows = m.load_rows(data / "pmf_rows.csv")
    v0, v1 = m.FOLD["val"]
    day = rows["date"].iloc[v0:v1].dt.normalize().to_numpy()
    days, dcode = np.unique(day, return_inverse=True)
    nd = len(days)
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, nd - BLOCK + 1, size=(B, -(-nd // BLOCK)))
    boot_days = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(B, -1)[
        :, :nd
    ]
    summary: dict = {
        "window_days": m.WINDOW_DAYS,
        "val_days": int(nd),
        "bootstrap": dict(B=B, block=BLOCK, seed=SEED),
    }
    plan = json.loads((ROOT / "inputs" / "plan.json").read_text())
    for s in m.SETS:
        f_by_id, rec_by_id, y = m.collect_jobs(data, runs, s)
        t = m.tune_table(data, runs, s).set_index("cfg")
        t = t[np.isfinite(t["val_qlike"])].sort_values("val_qlike")
        yv = y[v0:v1]
        ok = yv > 0
        cfgs = list(t.index)
        pick = cfgs[0]
        # per configuration: seed QLIKEs, seed-mean forecast losses by day
        seedq, dsum = {}, np.zeros((nd, len(cfgs)))
        cnt = np.bincount(dcode[ok], minlength=nd).astype(float)
        fmean = {}
        for i, c in enumerate(cfgs):
            F = np.vstack([f_by_id[f"ftune_{s}_f3_{c}_s{sd}"] for sd in m.SEEDS])
            qs = (yv[ok][None, :] / F[:, ok]) ** 2
            seedq[c] = (qs - np.log(qs) - 1.0).mean(axis=1)
            fm = F.mean(axis=0)
            fmean[c] = fm
            q = (yv[ok] / fm[ok]) ** 2
            dsum[:, i] = np.bincount(
                dcode[ok], weights=q - np.log(q) - 1.0, minlength=nd
            )
        tot = cnt.sum()
        diff = dsum - dsum[:, [0]]  # configuration minus pick, by day
        se = np.empty(len(cfgs))
        for c0 in range(0, len(cfgs), 32):
            dd = diff[:, c0 : c0 + 32]
            bs = dd[boot_days].sum(axis=1) / cnt[boot_days].sum(axis=1)[:, None]
            se[c0 : c0 + 32] = bs.std(axis=0, ddof=1)
        se[0] = np.nan
        dq = diff.sum(axis=0) / tot
        ec = epoch_cost(s)
        tab = []
        for i, c in enumerate(cfgs):
            mt = CFG_RE.match(c)
            assert mt is not None
            L, H = int(mt.group(1)), int(mt.group(2))
            eps = [int(t.loc[c, f"best_epoch_s{sd}"]) for sd in m.SEEDS]
            sq = seedq[c]
            tab.append(
                dict(
                    rank=i + 1,
                    cfg=c,
                    H=H,
                    L=L,
                    val_qlike=float(t.loc[c, "val_qlike"]),
                    d_vs_pick=float(dq[i]),
                    boot_se_d=float(se[i]),
                    seed_qlike_mean=float(sq.mean()),
                    seed_qlike_se=float(sq.std(ddof=1) / np.sqrt(len(sq))),
                    seed_qlikes=json.dumps([round(float(x), 6) for x in sq]),
                    epochs=json.dumps(eps),
                    refit_cost_s=(ec.get((H, L), np.nan) * float(np.mean(eps))),
                )
            )
        T = pd.DataFrame(tab)
        T.to_csv(OUT / f"tune_noise_{s}.csv", index=False)
        within = {
            k: T[(T["rank"] > 1) & (T["d_vs_pick"] <= k * T["boot_se_d"])]
            for k in (1, 2)
        }
        res: dict = dict(
            pick=pick,
            pick_qlike=float(T.loc[0, "val_qlike"]),
            n_configs=len(T),
            pick_seed_qlikes=T.loc[0, "seed_qlikes"],
            pick_seed_se=float(T.loc[0, "seed_qlike_se"]),
            median_seed_se=float(T["seed_qlike_se"].median()),
            top20_boot_se_d=[round(float(x), 6) for x in T["boot_se_d"].iloc[1:21]],
            top20_d=[round(float(x), 6) for x in T["d_vs_pick"].iloc[1:21]],
            pick_refit_cost_s=float(T.loc[0, "refit_cost_s"]),
            epoch_cost=str(ec),
        )
        for k, w in within.items():
            cheap = w.sort_values("refit_cost_s").head(1)
            res[f"n_within_{k}se"] = int(len(w))
            res[f"cheapest_within_{k}se"] = (
                dict(
                    cheap.iloc[0][
                        [
                            "cfg",
                            "val_qlike",
                            "d_vs_pick",
                            "boot_se_d",
                            "refit_cost_s",
                            "epochs",
                        ]
                    ]
                )
                if len(cheap)
                else None
            )
        res["top20"] = T.head(21)[
            [
                "rank",
                "cfg",
                "val_qlike",
                "d_vs_pick",
                "boot_se_d",
                "seed_qlike_se",
                "epochs",
                "refit_cost_s",
            ]
        ].to_dict(orient="records")
        # (3) the shortcut alone, for every configuration's (L, wd_lin); f_net vs f_shortcut for the top 20
        D = m.TData(data, s)
        ii = pd.read_csv(ROOT / "inputs" / "init_index_tune.csv")
        ii = ii[ii["design"] == s]
        keep = m.window_keep64(
            D.X64, plan["tune"]["fold"]["w0"], plan["tune"]["fold"]["w1"]
        )
        Z = np.c_[np.asarray(D.X32[v0:v1])[:, keep], np.ones(v1 - v0)].astype(
            np.float64
        )
        short = {}
        for _, r in ii.iterrows():
            v = np.fromfile(
                data
                / "pmf_lin_init"
                / s
                / f"tune_f3_L{int(r['L'])}_wl{m.g(float(r['wl']))}.f32",
                "<f4",
            )
            mu, sd = float(np.float32(r["mu"])), float(np.float32(r["sd"]))
            c0 = np.log(np.expm1(mu))
            sc = sd * (1.0 + np.exp(-c0))
            a = c0 + sc * (Z @ v.astype(np.float64))
            f = np.where(a > 20, a, np.log1p(np.exp(np.minimum(a, 20))))
            q = (yv[ok] / f[ok]) ** 2
            short[(int(r["L"]), m.g(float(r["wl"])))] = (
                float(np.mean(q - np.log(q) - 1.0)),
                f,
            )
        sc_rows = []
        for i, c in enumerate(cfgs[:21]):
            mt = re.match(
                r"^L(\d+)_H(\d+)_lr[0-9.e+-]+_we[0-9.e+-]+_wl([0-9.e+-]+)_", c
            )
            assert mt is not None
            qs_, fs = short[(int(mt.group(1)), mt.group(3))]
            sc_rows.append(
                dict(
                    rank=i + 1,
                    cfg=c,
                    val_qlike=float(t.loc[c, "val_qlike"]),
                    shortcut_qlike=qs_,
                    mean_abs_rel_dev=float(np.mean(np.abs(fmean[c] - fs) / fs)),
                )
            )
        SC = pd.DataFrame(sc_rows)
        SC.to_csv(OUT / f"shortcut_{s}.csv", index=False)
        res["shortcut_top"] = sc_rows[:5]
        res["best_shortcut_alone"] = min((v[0], k) for k, v in short.items())
        summary[s] = res
        print(
            s,
            json.dumps(
                {k: v for k, v in res.items() if k not in ("top20", "shortcut_top")},
                default=str,
            ),
            flush=True,
        )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
