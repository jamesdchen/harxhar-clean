"""Reduce the FI-by-window replays (fiw_replay.py) to the P1 / P2 tables of RULE_fiw.txt, with P3 alongside.

Inputs: <in>/fiw_w<W>.npz, fiw_w<W>.json, origins_w<W>.csv, dates_oos_w<W>.csv (copied from CARC
/scratch1/jc_905/fi_window/out and w<W>/design_all_features), the sweep's committed files under
results/window_sweep_2014start_2026-10-22 (forecasts.parquet for the gate re-check, frozen_penalties.json for the column
names and penalties, part1/linear_gap.csv and part2/gap.csv for P3).

Conventions (RULE_fiw.txt): gamma = the coefficient on the window-standardized column (lasso / enet: the walk's
scaled-coordinate theta; ridge: theta_j s_j(t)); selected = gamma != 0; exogenous variable = one raw series (all its
value and availability averages), 46 of them, 552 columns; backbone (har_ma + calendar, 12 columns) excluded from every
measure.  Bootstrap: paired moving blocks of 12 refits, B = 2,000, seed 20261022, the same resampled origins for every W.

Usage: python fiw_reduce.py --in <dir with the CARC outputs> [--out <dir>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
OUT_DEF = HERE.parent
REPO = HERE.parents[2]
SWEEP = REPO / "results" / "window_sweep_2014start_2026-10-22"
WS = [1000, 1500, 2000, 2500, 3000, 3500]
MODELS = ("lasso", "enet", "ridge")
MNAME = {"lasso": "lasso", "enet": "elastic net", "ridge": "ridge"}
COMMON: dict[str, list[float | str]] = {
    "lasso": [1e-4, 1e-3],
    "enet": [1e-3, 1e-2],
    "ridge": [1e3, 1e4, "rel"],
}
BLOCK, NBOOT, SEED = 12, 2000, 20261022
CATS = {  # src/data/loading.py SUBGROUPS order of the FI run (results/rehaul_fi_pooled13_2026-10-22/run_info.json)
    "moments": [
        "sumret",
        "sumabsret",
        "sumret3",
        "sumret4",
        "sumpret2",
        "sumbipow",
        "sumautocov",
    ],
    "liquidity": [
        "sumvolume",
        "numobs",
        "turnover_ewstock",
        "buyturnover_ewstock",
        "sellturnover_ewstock",
        "effspread_ewstock",
        "spread_ewstock",
        "turnover_vwstock",
        "buyturnover_vwstock",
        "sellturnover_vwstock",
        "effspread_vwstock",
        "spread_vwstock",
        "ofi_ewstock",
        "ofi_vwstock",
    ],
    "market_ew": [
        "sumret2_ewstock",
        "sumret3_ewstock",
        "sumret4_ewstock",
        "sumabsret_ewstock",
        "sumbipow_ewstock",
        "sumpret2_ewstock",
    ],
    "market_vw": [
        "sumret2_vwstock",
        "sumret3_vwstock",
        "sumret4_vwstock",
        "sumabsret_vwstock",
        "sumbipow_vwstock",
        "sumpret2_vwstock",
    ],
    "sentiment": [
        "stocktwits_attention",
        "stocktwits_sentiment",
        "stocktwits_sentcount",
    ],
    "implied_vol": ["vix", "vvix", "vix3m"],
    "vol_demand": [
        "voldemand_spx_open_and_close",
        "voldemand_spx_open_only",
        "voldemand_all_open_and_close",
        "voldemand_all_open_only",
    ],
    "fomc": ["fomc_day", "fomc_until_inv", "fomc_since_inv"],
}
CAT_OF = {v: c for c, vs in CATS.items() for v in vs}


def series_of(c: str) -> str:
    if re.fullmatch(r"har_ma_\d+", c):
        return "har_ma"
    if c in {f"DOW_{d}" for d in range(5)} | {"hour"}:
        return "calendar"
    m = re.fullmatch(r"adj_(.+)_ma_\d+", c) or re.fullmatch(r"(.+)_avail_ma_\d+", c)
    assert m, c
    return m.group(1)


def common_alpha(a, wd: int) -> float:
    return 1e4 * wd / 1000.0 if a == "rel" else float(a)


def src_tag(m: str, a) -> str:
    return "walk" if a is None else ("S1_rel" if a == "rel" else f"S1_{float(a):g}")


def n_cover(w: np.ndarray, q: float) -> np.ndarray:
    """Per row: the fewest columns whose largest values sum to >= q (rows of w sum to 1)."""
    s = -np.sort(-w, axis=1)
    c = np.cumsum(s, axis=1)
    return (c < q - 1e-12).sum(axis=1) + 1


def per_refit(gam: np.ndarray, E: np.ndarray, Gx: np.ndarray) -> dict:
    """gam: (K, 564) gamma; E: exogenous columns (bool, 564); Gx: (552, 46) column -> variable."""
    ax = np.abs(gam[:, E])
    nz = ax > 0.0
    sel = (nz.astype(float) @ Gx) > 0
    a = ax @ Gx
    tot = a.sum(axis=1)
    w = a / np.where(tot > 0, tot, 1.0)[:, None]
    return {
        "V": sel.sum(axis=1),
        "C": nz.sum(axis=1),
        "sel": sel,
        "nzc": nz,
        "a": a,
        "w": w,
        "N_eff": np.where(tot > 0, 1.0 / np.maximum((w**2).sum(axis=1), 1e-300), 0.0),
        "n50": n_cover(w, 0.5),
        "n80": n_cover(w, 0.8),
        "exog_share": tot / np.abs(gam).sum(axis=1),
    }


def stats_of(R: dict, idx: np.ndarray | None = None) -> dict:
    """The per-W statistics of RULE_fiw.txt on the refits idx (None: all)."""
    if idx is None:
        idx = np.arange(len(R["V"]))
    V, C, sel, nzc, a = (
        R["V"][idx],
        R["C"][idx],
        R["sel"][idx],
        R["nzc"][idx],
        R["a"][idx],
    )
    A = a.sum(axis=0)
    wp = A / A.sum() if A.sum() > 0 else A
    out = {
        "V_median": float(np.median(V)),
        "V_mean": float(V.mean()),
        "V_half": int((sel.mean(axis=0) >= 0.5).sum()),
        "C_median": float(np.median(C)),
        "C_mean": float(C.mean()),
        "C_half": int((nzc.mean(axis=0) >= 0.5).sum()),
        "Neff_median": float(np.median(R["N_eff"][idx])),
        "Neff_mean": float(R["N_eff"][idx].mean()),
        "n50_median": float(np.median(R["n50"][idx])),
        "n50_mean": float(R["n50"][idx].mean()),
        "n80_median": float(np.median(R["n80"][idx])),
        "n80_mean": float(R["n80"][idx].mean()),
        "Neff_pooled": float(1.0 / (wp**2).sum()) if A.sum() > 0 else 0.0,
        "n50_pooled": int(n_cover(wp[None, :], 0.5)[0]),
        "n80_pooled": int(n_cover(wp[None, :], 0.8)[0]),
        "exog_share_median": float(np.median(R["exog_share"][idx])),
    }
    return out


P1 = ["V_median", "V_mean", "V_half", "C_median", "C_mean", "C_half"]
P2 = [
    "Neff_median",
    "Neff_mean",
    "n50_median",
    "n50_mean",
    "n80_median",
    "n80_mean",
    "Neff_pooled",
    "n50_pooled",
    "n80_pooled",
]


def block_idx(K: int, rng: np.random.Generator) -> np.ndarray:
    nb = -(-K // BLOCK)
    st = rng.integers(0, K - BLOCK + 1, size=nb)
    return np.concatenate([np.arange(s, s + BLOCK) for s in st])[:K]


def slope(vals: np.ndarray) -> float:
    x = np.array(WS, dtype=float) / 1000.0
    return float(np.polyfit(x, vals, 1)[0])


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=str(OUT_DEF))
    a = ap.parse_args()
    IN, OUT = Path(a.inp), Path(a.out)
    fr0 = json.loads((SWEEP / "w1000" / "frozen_penalties.json").read_text())
    names = list(fr0["designs"]["all_features"]["columns"])
    p = len(names)
    assert p == 564
    ser = [series_of(c) for c in names]
    bb = np.array([s in ("har_ma", "calendar") for s in ser])
    assert bb.sum() == 12
    E = ~bb
    xvars = []
    for s in ser:
        if s not in ("har_ma", "calendar") and s not in xvars:
            xvars.append(s)
    assert len(xvars) == 46 and set(xvars) == set(CAT_OF), set(xvars) ^ set(CAT_OF)
    Gx = np.zeros((int(E.sum()), len(xvars)))
    Gx[np.arange(int(E.sum())), [xvars.index(s) for s, e in zip(ser, E) if e]] = 1.0

    # ------------------------------------------------------------------ load, gate re-check, gamma
    Z, J, O, frozen, gates = {}, {}, {}, {}, {}
    for wd in WS:
        Z[wd] = np.load(IN / f"fiw_w{wd}.npz")
        J[wd] = json.loads((IN / f"fiw_w{wd}.json").read_text())
        O[wd] = pd.read_csv(IN / f"origins_w{wd}.csv")
        fr = json.loads((SWEEP / f"w{wd}" / "frozen_penalties.json").read_text())
        assert fr["designs"]["all_features"]["columns"] == names
        frozen[wd] = {m: float(fr[m]["all_features"]) for m in MODELS}
    for wd in WS[1:]:
        assert list(O[wd]["origin"]) == list(O[WS[0]]["origin"]), (
            wd
        )  # the same calendar origins for every W
    origins = list(O[WS[0]]["origin"])
    K = len(origins)
    for wd in WS:
        fc = pd.read_parquet(
            SWEEP / f"w{wd}" / "forecasts.parquet",
            columns=["model", "predictor_set", "date", "pred_adj"],
        )
        d = pd.read_csv(IN / f"dates_oos_w{wd}.csv", parse_dates=["date"])["date"]
        z = Z[wd]
        for m in MODELS:
            f = fc[(fc.model == m) & (fc.predictor_set == "all_features")].set_index(
                "date"
            )["pred_adj"]
            pr, ln, g, i = (
                z[f"replay_{m}_preds"],
                z[f"replay_{m}_preds_len"],
                z[f"replay_{m}_g"],
                z["oos_row"],
            )
            off = np.r_[0, np.cumsum(ln)]
            mx, mxt, nrow = 0.0, 0.0, 0
            for q in range(K):
                walk = f.reindex(d.iloc[g[q] : i[q] + 1]).to_numpy()
                assert np.isfinite(walk).all() and len(walk) == ln[q]
                dif = np.abs(pr[off[q] : off[q + 1]] - walk)
                mx, mxt, nrow = (
                    max(mx, float(dif.max())),
                    max(mxt, float(dif[-1])),
                    nrow + len(dif),
                )
            # the cold-seed batch fit at the frozen penalty against the walk at the origin (S1's link to the walk)
            bt = f"batch_{m}_{frozen[wd][m]:g}"
            pb, pw = z[f"{bt}_pred_t"], z[f"replay_{m}_pred_t"]
            rel = np.abs(pb - pw) / np.abs(pw)
            tb = i % 250 == 0
            gw, gb = gam_of(z, f"replay_{m}", m), gam_of(z, bt, m)
            gates[(wd, m)] = {
                "frozen_alpha": frozen[wd][m],
                "replay_rows": nrow,
                "replay_max_abs_all_rows": mx,
                "replay_max_abs_at_origins": mxt,
                "gate_pass_1e12": bool(mx <= 1e-12),
                "carc_gate": J[wd]["gates"][m],
                "batch_at_frozen_vs_walk_pred_max_rel": float(rel.max()),
                "batch_at_frozen_vs_walk_pred_n_rel_gt_1e10": int((rel > 1e-10).sum()),
                "batch_at_frozen_vs_walk_at_tune_boundary_max_abs": float(
                    np.abs(pb - pw)[tb].max()
                )
                if tb.any()
                else None,
                "n_origins_at_tune_boundary": int(tb.sum()),
                "batch_vs_walk_support_differs_origins": int(
                    ((gw != 0) != (gb != 0))[:, E].any(axis=1).sum()
                ),
                "batch_vs_walk_V_max_abs_diff": int(
                    np.abs(per_refit(gw, E, Gx)["V"] - per_refit(gb, E, Gx)["V"]).max()
                ),
            }
    gate_all = all(v["gate_pass_1e12"] for v in gates.values())

    # ------------------------------------------------------------------ per refit, per W statistics
    R: dict = {}
    for wd in WS:
        for m in MODELS:
            R[(wd, m, "walk")] = per_refit(gam_of(Z[wd], f"replay_{m}", m), E, Gx)
            for c in COMMON[m]:
                R[(wd, m, src_tag(m, c))] = per_refit(
                    gam_of(Z[wd], f"batch_{m}_{common_alpha(c, wd):g}", m), E, Gx
                )
    sources = {m: ["walk"] + [src_tag(m, c) for c in COMMON[m]] for m in MODELS}
    long_rows, summ_rows = [], []
    for (wd, m, s), r in R.items():
        for q in range(K):
            long_rows.append(
                {
                    "W": wd,
                    "model": m,
                    "source": s,
                    "k": q,
                    "origin": origins[q],
                    "V": int(r["V"][q]),
                    "C": int(r["C"][q]),
                    "N_eff": float(r["N_eff"][q]),
                    "n50": int(r["n50"][q]),
                    "n80": int(r["n80"][q]),
                    "exog_share": float(r["exog_share"][q]),
                }
            )
        alpha = (
            frozen[wd][m]
            if s == "walk"
            else common_alpha(s[3:] if s != "S1_rel" else "rel", wd)
        )
        summ_rows.append(dict(W=wd, model=m, source=s, alpha=alpha, K=K, **stats_of(r)))
    refits = pd.DataFrame(long_rows)
    summ = pd.DataFrame(summ_rows)

    # ------------------------------------------------------------------ trends with the paired block bootstrap
    rng = np.random.default_rng(SEED)
    boots = [block_idx(K, rng) for _ in range(NBOOT)]
    trend_rows = []
    for m in MODELS:
        for s in sources[m]:
            meas = P1 + P2
            pt = np.array(
                [[stats_of(R[(wd, m, s)])[k] for wd in WS] for k in meas]
            )  # (measures, W)
            bs = np.empty((NBOOT, len(meas), len(WS)))
            for b, ix in enumerate(boots):
                st = [stats_of(R[(wd, m, s)], ix) for wd in WS]
                bs[b] = np.array([[st[j][k] for j in range(len(WS))] for k in meas])
            for t, k in enumerate(meas):
                sl = np.array([slope(bs[b, t]) for b in range(NBOOT)])
                df = bs[:, t, -1] - bs[:, t, 0]
                lo, hi = np.percentile(sl, [2.5, 97.5])
                dlo, dhi = np.percentile(df, [2.5, 97.5])
                steps = int((np.diff(pt[t]) > 0).sum())
                verdict = "rises" if lo > 0 else ("falls" if hi < 0 else "flat")
                trend_rows.append(
                    dict(
                        model=m,
                        source=s,
                        prediction="P1" if k in P1 else "P2",
                        measure=k,
                        **{f"W{wd}": pt[t, j] for j, wd in enumerate(WS)},
                        slope_per_1000d=slope(pt[t]),
                        slope_lo=lo,
                        slope_hi=hi,
                        diff_3500_1000=pt[t, -1] - pt[t, 0],
                        diff_lo=dlo,
                        diff_hi=dhi,
                        steps_up_of_5=steps,
                        verdict=verdict,
                    )
                )
    trend = pd.DataFrame(trend_rows)
    verd = []
    for (m, s, pr), t in trend.groupby(["model", "source", "prediction"], sort=False):
        nr = int((t.verdict == "rises").sum())
        verd.append(
            {
                "model": m,
                "source": s,
                "prediction": pr,
                "measures": len(t),
                "rises": nr,
                "flat": int((t.verdict == "flat").sum()),
                "falls": int((t.verdict == "falls").sum()),
                "verdict": "SUPPORTED"
                if nr == len(t)
                else ("NOT SUPPORTED" if nr == 0 else "PARTLY"),
            }
        )
    verd = pd.DataFrame(verd)

    # ------------------------------------------------------------------ variables: frequencies, pooled shares, top 10
    var_rows = []
    for (wd, m, s), r in R.items():
        A = r["a"].sum(axis=0)
        wp = A / A.sum()
        fq = r["sel"].mean(axis=0)
        for j, v in enumerate(xvars):
            var_rows.append(
                {
                    "W": wd,
                    "model": m,
                    "source": s,
                    "variable": v,
                    "category": CAT_OF[v],
                    "sel_freq": fq[j],
                    "pooled_share": wp[j],
                    "per_refit_share_median": float(np.median(r["w"][:, j])),
                }
            )
    vtab = pd.DataFrame(var_rows)
    top_rows, top10 = [], {}
    for wd in WS:
        t = vtab[
            (vtab.W == wd) & (vtab.model == "ridge") & (vtab.source == "walk")
        ].sort_values("pooled_share", ascending=False)
        top10[wd] = list(t.variable.head(10))
        for rk, (_, row) in enumerate(t.head(10).iterrows(), 1):
            top_rows.append(
                {
                    "W": wd,
                    "rank": rk,
                    "variable": row.variable,
                    "category": row.category,
                    "pooled_share": row.pooled_share,
                    "in_W1000_top10": row.variable in top10.get(1000, []),
                }
            )
    tops = pd.DataFrame(top_rows)
    sp = pd.DataFrame(index=WS, columns=WS, dtype=float)
    for wa in WS:
        for wb in WS:
            xa = (
                vtab[(vtab.W == wa) & (vtab.model == "ridge") & (vtab.source == "walk")]
                .set_index("variable")
                .pooled_share
            )
            xb = (
                vtab[(vtab.W == wb) & (vtab.model == "ridge") & (vtab.source == "walk")]
                .set_index("variable")
                .pooled_share
            )
            sp.loc[wa, wb] = spearmanr(xa[xvars], xb[xvars])[0]
    overlap = {wd: len(set(top10[wd]) & set(top10[1000])) for wd in WS}
    # the same stability measures for every ridge source (S1: is the walk's reshuffle the penalty step 1e4 -> 1e3?)
    stab_rows = []
    for s in sources["ridge"]:
        r = vtab[(vtab.model == "ridge") & (vtab.source == s)]
        sh = {wd: r[r.W == wd].set_index("variable").pooled_share[xvars] for wd in WS}
        t10 = {wd: set(sh[wd].sort_values(ascending=False).index[:10]) for wd in WS}
        stab_rows.append(
            {
                "source": s,
                "spearman_1000_3500": spearmanr(sh[1000], sh[3500])[0],
                "spearman_1500_2000": spearmanr(sh[1500], sh[2000])[0],
                "top10_overlap_1000_3500": len(t10[1000] & t10[3500]),
                "top5_W1000": ", ".join(
                    sh[1000].sort_values(ascending=False).index[:5]
                ),
                "top5_W3500": ", ".join(
                    sh[3500].sort_values(ascending=False).index[:5]
                ),
            }
        )
    stab = pd.DataFrame(stab_rows)

    # ------------------------------------------------------------------ P3 (already measured; reformatted)
    g1 = pd.read_csv(SWEEP / "part1" / "linear_gap.csv")
    g2 = pd.read_csv(SWEEP / "part2" / "gap.csv")
    p3 = []
    for m in MODELS:
        for wd in WS:
            r1 = g1[(g1.est == m) & (g1.W == wd)].iloc[0]
            r2 = g2[(g2.model == m) & (g2.W == wd)].iloc[0]
            p3.append(
                {
                    "model": m,
                    "W": wd,
                    "fold3_mse_gap_pct": r1.mse_gap_pct,
                    "fold3_mse_gap_se": r1.mse_gap_se,
                    "fold3_extra_gain_pp": r1.mse_gain_full_minus_har_pp,
                    "fold3_extra_gain_se": r1.mse_gain_diff_se,
                    "fold3_extra_gain_p": r1.mse_gain_diff_p,
                    "fold3_qlike_gap_pct": r1.qlike_gap_pct,
                    "oos_bar_qlike_gap_pct": r2.gap_pct_bar,
                    "oos_bar_gap_p": r2.gap_p_bar,
                    "oos_1600_qlike_gap_pct": r2.gap_pct_16,
                    "oos_1600_gap_p": r2.gap_p_16,
                }
            )
    p3 = pd.DataFrame(p3)

    # ------------------------------------------------------------------ write
    OUT.mkdir(parents=True, exist_ok=True)
    refits.to_csv(OUT / "fiw_refits.csv", index=False)
    summ.to_csv(OUT / "fiw_summary.csv", index=False)
    trend.to_csv(OUT / "fiw_trend.csv", index=False)
    verd.to_csv(OUT / "fiw_verdicts.csv", index=False)
    vtab.to_csv(OUT / "fiw_variables.csv", index=False)
    tops.to_csv(OUT / "fiw_ridge_top10.csv", index=False)
    sp.to_csv(OUT / "fiw_ridge_share_spearman.csv")
    stab.to_csv(OUT / "fiw_ridge_share_stability.csv", index=False)
    p3.to_csv(OUT / "fiw_p3.csv", index=False)
    org = O[WS[0]][
        ["k", "origin", "session_index", "block_rows", "block_sessions"]
    ].copy()
    for wd in WS:
        org[f"oos_row_w{wd}"] = O[wd]["oos_row"].to_numpy()
        org[f"tune_boundary_w{wd}"] = O[wd]["tune_boundary"].to_numpy()
    org.to_csv(OUT / "fiw_origins.csv", index=False)
    # the walk's gamma (and the S1 gammas) per W, model, origin
    arr = {"columns": np.array(names), "origins": np.array(origins)}
    for wd in WS:
        for m in MODELS:
            arr[f"walk_w{wd}_{m}_gamma"] = gam_of(Z[wd], f"replay_{m}", m)
            arr[f"walk_w{wd}_{m}_mask"] = Z[wd][f"replay_{m}_mask"][:, :-1]
            for c in COMMON[m]:
                arr[f"{src_tag(m, c)}_w{wd}_{m}_gamma"] = gam_of(
                    Z[wd], f"batch_{m}_{common_alpha(c, wd):g}", m
                )
    np.savez_compressed(OUT / "fiw_gamma.npz", **arr)
    txt = tables_txt(
        summ, trend, verd, tops, overlap, sp, p3, gates, frozen, K, origins, stab
    )
    (OUT / "fiw_tables.txt").write_text(txt, encoding="utf-8")
    files = {
        "experiments/pool_linear_pretune.py": REPO
        / "experiments"
        / "pool_linear_pretune.py",
        "experiments/pooled_linear_rerun_2026_10_19_13.py": REPO
        / "experiments"
        / "pooled_linear_rerun_2026_10_19_13.py",
        "specs/causal_tune_linear.py": REPO / "specs" / "causal_tune_linear.py",
        "src/backtest/multi_stage.py": REPO / "src" / "backtest" / "multi_stage.py",
        "src/models/reclasso_har.py": REPO / "src" / "models" / "reclasso_har.py",
        "src/backtest/executor.py": REPO / "src" / "backtest" / "executor.py",
        "src/data/loading.py": REPO / "src" / "data" / "loading.py",
        "code/fiw_replay.py": HERE / "fiw_replay.py",
        "code/fiw_job.sh": HERE / "fiw_job.sh",
        "code/fiw_reduce.py": HERE / "fiw_reduce.py",
        **{
            f"sweep/w{wd}/forecasts.parquet": SWEEP / f"w{wd}" / "forecasts.parquet"
            for wd in WS
        },
    }
    info = {
        "created": "2026-10-22",
        "what": "FI by training window: the pooled linear models' full-set coefficients replayed from the 2014-start window "
        "sweep at monthly origins; P1 (lasso selection) and P2 (ridge weight spread) of RULE_fiw.txt",
        "rule": "RULE_fiw.txt (committed a5d373b before any computation)",
        "sweep": str(SWEEP.relative_to(REPO)),
        "windows_days": WS,
        "frozen_penalties": {str(wd): frozen[wd] for wd in WS},
        "common_penalties_S1": {m: [str(c) for c in COMMON[m]] for m in MODELS},
        "origins": {
            "K": K,
            "first": origins[0],
            "last": origins[-1],
            "rule": "first row of every 21st session from the first "
            "2014-01-02 row; the same calendar origins for every W",
        },
        "gate": {
            "all_pass_1e12": gate_all,
            "per_W_model": {f"w{wd}/{m}": gates[(wd, m)] for wd, m in gates},
        },
        "carc": dict(
            {
                f"w{wd}": {
                    k: J[wd][k]
                    for k in ("host", "numpy", "python", "env", "wall_sec", "workers")
                }
                for wd in WS
            },
            base="/scratch1/jc_905/fi_window",
            code="copy of /scratch1/jc_905/wsweep/code (experiments, specs, "
            "src), sha256 equal to this branch's files below",
        ),
        "bootstrap": {
            "kind": "paired moving blocks over refits",
            "block": BLOCK,
            "B": NBOOT,
            "seed": SEED,
        },
        "ridge_top10": top10,
        "ridge_top10_overlap_with_W1000": overlap,
        "code": {
            "head": subprocess.run(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"],
                capture_output=True,
                check=False,
                text=True,
            ).stdout.strip(),
            "sha256": {k: sha256(v) for k, v in files.items()},
            "python": platform.python_version(),
            "numpy": np.__version__,
            "interpreter": sys.executable,
        },
    }
    (OUT / "run_info.json").write_text(
        json.dumps(
            info, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)
        )
    )
    print(txt)


def gam_of(z, tag: str, m: str) -> np.ndarray:
    """gamma (K, 564): the coefficient on the window-standardized column (intercept dropped)."""
    th = z[f"{tag}_theta"][:, :-1]
    return th * z[f"{tag}_sc"][:, :-1] if m == "ridge" else th.copy()


def fmt(v, nd=1) -> str:
    if isinstance(v, (int, np.integer)) or (
        isinstance(v, float) and float(v).is_integer() and nd <= 1
    ):
        return f"{int(v):,}"
    return f"{v:,.{nd}f}"


def tables_txt(
    summ, trend, verd, tops, overlap, sp, p3, gates, frozen, K, origins, stab
) -> str:
    L = []
    L.append(
        f"FI by training window, 2014-start sweep (RULE_fiw.txt). K = {K} monthly origins {origins[0]} .. {origins[-1]}."
    )
    L.append(
        "gamma = coefficient on the window-standardized column; 46 exogenous variables (552 columns); backbone excluded."
    )
    L.append("")
    L.append(
        "GATE: replayed forecasts vs forecasts.parquet pred_adj (every replayed row g..t)"
    )
    L.append(
        "| W | model | alpha | rows | max abs | pass <=1e-12 | batch@frozen vs walk: max rel, n>1e-10 |"
    )
    L.append("|---|---|---|---|---|---|---|")
    for (wd, m), g in gates.items():
        L.append(
            f"| {wd:,} | {m} | {g['frozen_alpha']:g} | {g['replay_rows']:,} | {g['replay_max_abs_all_rows']:.1e} | "
            f"{g['gate_pass_1e12']} | {g['batch_at_frozen_vs_walk_pred_max_rel']:.1e}, "
            f"{g['batch_at_frozen_vs_walk_pred_n_rel_gt_1e10']} |"
        )
    L.append("")
    for pr, meas in (("P1", P1), ("P2", P2)):
        for m in MODELS:
            for s in trend[(trend.model == m)].source.unique():
                t = trend[
                    (trend.model == m) & (trend.source == s) & (trend.prediction == pr)
                ]
                al = [
                    summ[
                        (summ.W == wd) & (summ.model == m) & (summ.source == s)
                    ].alpha.iloc[0]
                    for wd in WS
                ]
                L.append(
                    f"### {pr} {MNAME[m]}, {s} (alpha by W: {' / '.join(f'{x:g}' for x in al)})"
                )
                L.append(
                    "| measure | "
                    + " | ".join(f"W {wd:,}" for wd in WS)
                    + " | slope /1000d [95%] | 3500-1000 [95%] | steps up | verdict |"
                )
                L.append("|---|" + "---|" * (len(WS) + 4))
                for _, r in t.iterrows():
                    nd = (
                        0
                        if r.measure.endswith(("_half", "_pooled"))
                        and not r.measure.startswith("Neff")
                        else 1
                    )
                    L.append(
                        f"| {r.measure} | "
                        + " | ".join(fmt(r[f"W{wd}"], nd) for wd in WS)
                        + f" | {r.slope_per_1000d:+.2f} [{r.slope_lo:+.2f}, {r.slope_hi:+.2f}] | "
                        f"{r.diff_3500_1000:+.1f} [{r.diff_lo:+.1f}, {r.diff_hi:+.1f}] | {r.steps_up_of_5}/5 | {r.verdict} |"
                    )
                v = verd[
                    (verd.model == m) & (verd.source == s) & (verd.prediction == pr)
                ].iloc[0]
                L.append(
                    f"-> {pr} {MNAME[m]} {s}: {v.verdict} ({v.rises} rise, {v.flat} flat, {v.falls} fall of {v.measures})"
                )
                L.append("")
    L.append(
        "### Ridge (walk): the 10 heaviest exogenous variables by pooled share of sum|gamma|, by W"
    )
    L.append("| rank | " + " | ".join(f"W {wd:,}" for wd in WS) + " |")
    L.append("|---|" + "---|" * len(WS))
    for rk in range(1, 11):
        cells = []
        for wd in WS:
            r = tops[(tops.W == wd) & (tops["rank"] == rk)].iloc[0]
            cells.append(f"{r.variable} ({r.pooled_share:.3f})")
        L.append(f"| {rk} | " + " | ".join(cells) + " |")
    L.append(
        "overlap with W 1,000's top 10: "
        + ", ".join(f"W {wd:,}: {overlap[wd]}" for wd in WS)
    )
    L.append("Spearman of the 46 pooled ridge shares between W's:")
    L.append("| W | " + " | ".join(f"{wd:,}" for wd in WS) + " |")
    L.append("|---|" + "---|" * len(WS))
    for wa in WS:
        L.append(
            f"| {wa:,} | " + " | ".join(f"{sp.loc[wa, wb]:.2f}" for wb in WS) + " |"
        )
    L.append(
        "Ridge share stability by source (walk = frozen penalties; S1 = a common penalty):"
    )
    L.append(
        "| source | Spearman 1000 vs 3500 | Spearman 1500 vs 2000 | top-10 overlap 1000 vs 3500 | top 5 at W 3,500 |"
    )
    L.append("|---|---|---|---|---|")
    for _, r in stab.iterrows():
        L.append(
            f"| {r.source} | {r.spearman_1000_3500:.2f} | {r.spearman_1500_2000:.2f} | "
            f"{r.top10_overlap_1000_3500} | {r.top5_W3500} |"
        )
    L.append("")
    L.append(
        "### P3 (already measured: part1/linear_gap.csv, part2/gap.csv; full - HAR, %; negative = full better)"
    )
    L.append(
        "| model | W | fold3 MSE gap (SE) | fold3 extra gain vs W1000 pp (SE) [p] | OOS every-bar QLIKE gap [p] | OOS 16:00 QLIKE gap [p] |"
    )
    L.append("|---|---|---|---|---|---|")
    for _, r in p3.iterrows():
        eg = (
            "-"
            if pd.isna(r.fold3_extra_gain_pp)
            else f"{r.fold3_extra_gain_pp:+.2f} ({r.fold3_extra_gain_se:.2f}) [{r.fold3_extra_gain_p:.3f}]"
        )
        L.append(
            f"| {MNAME[r.model]} | {r.W:,} | {r.fold3_mse_gap_pct:+.2f} ({r.fold3_mse_gap_se:.2f}) | {eg} | "
            f"{r.oos_bar_qlike_gap_pct:+.2f} [{r.oos_bar_gap_p:.3f}] | {r.oos_1600_qlike_gap_pct:+.2f} [{r.oos_1600_gap_p:.3f}] |"
        )
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
