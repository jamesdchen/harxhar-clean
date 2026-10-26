"""Old vs rebuilt CI-HAR (master key sub_ols_baseline): accuracy, the DM statistics against it, its sign strategy.

RULE   results/ci_har_rebuild_2026-10-26/RULE.txt (item 5, pre-registered).  The rebuilt file is
       results/ci_har_rebuild_2026-10-26/results_bar1600.csv (experiments/ci_har_rebuild_2026_10_26.py).
REUSE  nothing of the scoring or the strategy is re-implemented:
       experiments/rolling_eval_2026_10_15.py   read_key (plain = pred_adj^2 B, B = true_raw / true_adj^2), history (the
                                                actual 16:00 RV, early closes dropped), common_span (T_oos), metrics,
                                                ct_benchmark, dm_pair;
       writeup/pooled_linear_13.py, pooled_trees_13.py   the 12 pooled forecasts at the 16:00 bar (final rerun);
       src.evaluation.diebold_mariano.dm_test   the DM of Section 6 (make_rehaul_results.pair: QLIKE on T_oos);
       writeup/make_rehaul_econ_main.py (+ make_rehaul_economics.py)   load, the pooled rows, row_stats (D6 Sharpe at
                                                mid and bid/ask, Delta SR vs the short straddle with its bootstrap p,
                                                by-year terms) and pair (Delta SR with the bootstrap p).
       The new CI-HAR's trade is the master table's rule: q = +1 if plain > iv_var else -1 (ties short), mid q R,
       bid/ask master_table_gui.rows._crossed_return.
GATES  (stop on failure) the old file's numbers reproduce rolling_eval_2026-10-15/metrics.csv (QLIKE, MSE 1e-12)
       and the printed macros (rhrOlsQ, rhrVsBm<M><Scope><Set>{Pct,Dm}, the rhrVsBmPool... counts, rhrOlsSharpe,
       rhemShCiHar, rhemShXCiHar, rhemDShCiHarVsShort, rhemFullVsCiHar{Min,Med,Max}, rhemHarVsCiHar{Neg,Sig}); the old
       file's trade signs equal the master table's q on all 866 days.
WRITES results/ci_har_rebuild_2026-10-26/measure/{accuracy,dm_old_vs_new,vsbm,vsbm_counts,sharpe,sharpe_pairs,
       b_days,attribution,sharpe_attribution,flips,model_set_vs_ci,model_set_counts,vsbm_2014,gates,
       gates_model_set}.csv and summary.json.

Run:  python experiments/ci_har_rebuild_measure_2026_10_26.py     (repository root; sets REHAUL_FINAL_RUN=1)
      python experiments/ci_har_rebuild_measure_2026_10_26.py --attribution          (accuracy by flag only)
      python experiments/ci_har_rebuild_measure_2026_10_26.py --sharpe-attribution   (Sharpe ratio by flag only)
      python experiments/ci_har_rebuild_measure_2026_10_26.py --model-set            (Section 7 model-set counts only)
      python experiments/ci_har_rebuild_measure_2026_10_26.py --vsbm-2014            (pooled vs new CI-HAR, 2014-2024)
"""

from __future__ import annotations

import os

os.environ["REHAUL_FINAL_RUN"] = (
    "1"  # the paper's pooled forecasts are the final rerun (as every Section 6/7 generator)
)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "notebooks", REPO / "experiments", REPO / "writeup"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.chdir(REPO)

import make_rehaul_econ_main as em  # noqa: E402  (imports make_rehaul_economics as em.me)
import pooled_linear_13 as pl13  # noqa: E402
import pooled_trees_13 as pt13  # noqa: E402
import rolling_eval_2026_10_15 as rolling  # noqa: E402
from master_table_gui import rows as gui  # noqa: E402

from src.evaluation.diebold_mariano import dm_test  # noqa: E402

me = em.me
OUT = REPO / "results" / "ci_har_rebuild_2026-10-26"
MEAS = OUT / "measure"
OLD_PATH = rolling.OLS_ARM
NEW_PATH = OUT / "results_bar1600.csv"
CI = rolling.OLS  # sub_ols_baseline
GEN = REPO / "writeup" / "generated"
SPANS = {
    "2018-06-25..2024-04-30": (pd.Timestamp("2018-06-25"), pd.Timestamp("2024-04-30")),
    "T_oos 2020-01-03..2024-04-30": (pd.Timestamp("2020-01-03"), pd.Timestamp("2024-04-30")),
    "2014-01-02..2024-04-30": (pd.Timestamp("2014-01-02"), pd.Timestamp("2024-04-30")),
    "2014-01-02..2019-12-31": (pd.Timestamp("2014-01-02"), pd.Timestamp("2019-12-31")),
}
EST = ("ridge", "lasso", "enet", "lgbm", "xgb", "rf")
TOK = {"ridge": "Ridge", "lasso": "Lasso", "enet": "Enet", "lgbm": "Lgbm", "xgb": "Xgb", "rf": "Rf"}
LIN = ("ridge", "lasso", "enet")
SETS = ("baseline", "all_features")
XTOK = {"baseline": "Har", "all_features": "Full"}
GATES: list[dict] = []


def gate(name: str, ok: bool, detail: str) -> None:
    GATES.append({"gate": name, "pass": bool(ok), "detail": detail})
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        pd.DataFrame(GATES).to_csv(MEAS / "gates.csv", index=False)
        raise SystemExit(f"gate failed: {name}")


def printed(macros: dict, name: str) -> float:
    return float(em._norm(macros[name]).replace("p =", "").strip())


def near_printed(x: float, s: float, dp: int) -> bool:
    return abs(x - s) <= 0.5 * 10.0 ** (-dp) + 1e-9


# ============================================================================================ CI-HAR files
def ci_scored(path: Path) -> pd.DataFrame:
    """rolling_eval's scoring of the arm file at `path` (read_key + scored's day filter): plain, B, rv by day."""
    k = rolling.Key(CI, "OLS", "baseline", "close", path, "arm", "ols")
    f = rolling.read_key(k)
    hist = rolling.history()
    f = f[~f.index.isin(rolling.EARLY) & f.index.isin(hist.index)]
    return pd.DataFrame({"plain": f["plain"], "B": f["B"], "yhat": f["yhat"], "rv": hist.reindex(f.index)})


def qlike(rv: np.ndarray, p: np.ndarray) -> np.ndarray:
    return rolling.qlike_daily(np.asarray(rv, float), np.asarray(p, float))


# ============================================================================================ 1. accuracy
def accuracy(old: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist = rolling.history()
    rows, dms = [], []
    common = rolling.common_span()
    mt866 = master_days()
    span_days = {nm: (a, b) for nm, (a, b) in SPANS.items()}
    sets: list[tuple[str, pd.DatetimeIndex | None, tuple | None]] = [(nm, None, ab) for nm, ab in span_days.items()]
    sets += [("T_oos common span (1,082)", common, None), ("master 866 days", mt866, None)]
    for nm, idx, ab in sets:
        sel = {}
        for lab, f in (("old", old), ("new", new)):
            d = f.index if idx is None else f.index.intersection(idx)
            if ab is not None:
                d = d[(d >= ab[0]) & (d <= ab[1])]
            sel[lab] = d
        both = sel["old"].intersection(sel["new"])
        for lab, f in (("old", old), ("new", new)):
            for basis, d in (("own", sel[lab]), ("both", both)):
                if len(d) == 0:
                    rows.append({"span": nm, "file": lab, "days": basis, "n": 0})
                    continue
                bench = rolling.ct_benchmark(hist, d)
                m = rolling.metrics(f["rv"].reindex(d), f["plain"].reindex(d), bench)
                rows.append(
                    {
                        "span": nm,
                        "file": lab,
                        "days": basis,
                        "n": len(d),
                        "first": d[0].date(),
                        "last": d[-1].date(),
                        "qlike": m["qlike"],
                        "mse_e10": 1e10 * m["mse"],
                        "r2_oos_pct": m["r2_oos_pct"],
                    }
                )
        if len(both):
            r = rolling.dm_pair(old["rv"].reindex(both), new["plain"].reindex(both), old["plain"].reindex(both))
            dms.append({"span": nm, "n": len(both), **r})
    return pd.DataFrame(rows), pd.DataFrame(dms)


def master_days() -> pd.DatetimeIndex:
    d = pd.read_parquet(me.RES / "master_table_daily.parquet", columns=["key", "day"])
    d = d[d["key"] == CI]
    return pd.DatetimeIndex(pd.to_datetime(d["day"]).dt.normalize().sort_values()).as_unit("ns")


# ============================================================================================ 2. DM vs CI-HAR
def vsbm(old: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """make_rehaul_results.pair(by[k], by[OLS]) on T_oos (QLIKE % change, DM; plus squared error) for every key of
    FAMILY_ROWS, against the old and the new CI-HAR."""
    days = rolling.common_span()
    rv = rolling.history().reindex(days).to_numpy(float)
    ci = {"old": old["plain"].reindex(days).to_numpy(float), "new": new["plain"].reindex(days).to_numpy(float)}
    for lab, v in ci.items():
        assert np.isfinite(v).all() and (v > 0).all(), lab
    rr = em._macros(GEN / "rehaul_results_numbers.tex")
    rows = []
    for e in EST:
        for scope in ("Close", "Pool"):
            for s in SETS:
                if scope == "Pool":
                    g = (pl13 if e in LIN else pt13).at16(e, s)
                    key = f"poolpt13_{e}_{s}" if e in LIN else f"pooltree_{e}_{s}"
                    assert np.max(np.abs(g["rv"].reindex(days).to_numpy(float) / rv - 1.0)) <= 1e-12, key
                    p = g["plain"].reindex(days).to_numpy(float)
                else:
                    key = f"subpt_{e}_{s}" if e in LIN else f"subtree_daily_{s}_{e}"
                    p = rolling.scored(key)["plain"].reindex(days).to_numpy(float)
                assert np.isfinite(p).all() and (p > 0).all(), key
                qa = qlike(rv, p)
                r: dict = {"key": key, "est": e, "scope": scope, "set": s, "n": len(days)}
                for lab, c in ci.items():
                    qb = qlike(rv, c)
                    dm = dm_test(qa, qb)
                    assert dm["hac_lag"] == 6, dm
                    dms = dm_test((rv - p) ** 2, (rv - c) ** 2)
                    r[f"{lab}_pct"] = 100.0 * (qa.mean() / qb.mean() - 1.0)
                    r[f"{lab}_dm"] = float(dm["dm"])
                    r[f"{lab}_p"] = float(dm["p"])
                    r[f"{lab}_mse_pct"] = 100.0 * (((rv - p) ** 2).mean() / ((rv - c) ** 2).mean() - 1.0)
                    r[f"{lab}_dm_mse"] = float(dms["dm"])
                    r[f"{lab}_p_mse"] = float(dms["p"])
                m = f"rhrVsBm{TOK[e]}{scope}{XTOK[s]}"
                r["printed_pct"], r["printed_dm"] = printed(rr, m + "Pct"), printed(rr, m + "Dm")
                gate(
                    f"{m} old reproduces the printed Pct / Dm",
                    near_printed(r["old_pct"], r["printed_pct"], 1) and near_printed(r["old_dm"], r["printed_dm"], 2),
                    f"{r['old_pct']:.3f} / {r['old_dm']:.3f} vs printed {r['printed_pct']} / {r['printed_dm']}",
                )
                rows.append(r)
    v = pd.DataFrame(rows)
    # the Section 6 aggregates (make_rehaul_results macro section)
    cnt = []
    pool = v[v["scope"] == "Pool"]
    for lab in ("old", "new"):
        full = pool[pool["set"] == "all_features"]
        har = pool[pool["set"] == "baseline"]
        lin_f, tree_f = full[full["est"].isin(LIN)], full[~full["est"].isin(LIN)]
        narrow = sum(
            float(full.loc[full["est"] == e, f"{lab}_pct"].iloc[0])
            < float(har.loc[har["est"] == e, f"{lab}_pct"].iloc[0])
            for e in EST
        )
        cnt.append(
            {
                "which": lab,
                "rhrVsBmPoolLinFullPctMin": lin_f[f"{lab}_pct"].min(),
                "rhrVsBmPoolLinFullPctMax": lin_f[f"{lab}_pct"].max(),
                "rhrVsBmPoolTreeFullPctMin": tree_f[f"{lab}_pct"].min(),
                "rhrVsBmPoolTreeFullPctMax": tree_f[f"{lab}_pct"].max(),
                "rhrVsBmPoolHarPctMin": har[f"{lab}_pct"].min(),
                "rhrVsBmPoolHarPctMax": har[f"{lab}_pct"].max(),
                "rhrVsBmPoolLinFullSig": int((lin_f[f"{lab}_p"] < 0.05).sum()),
                "rhrVsBmPoolTreeFullSig": int((tree_f[f"{lab}_p"] < 0.05).sum()),
                "rhrVsBmPoolHarSig": int((har[f"{lab}_p"] < 0.05).sum()),
                "rhrVsBmPoolHarSigOne": int((har[f"{lab}_p"] < 0.01).sum()),
                "rhrVsBmPoolNarrow": int(narrow),
                "full_dm_min": full[f"{lab}_dm"].min(),
                "full_dm_max": full[f"{lab}_dm"].max(),
                "har_dm_min": har[f"{lab}_dm"].min(),
                "har_dm_max": har[f"{lab}_dm"].max(),
                "rhrVsBmRfPoolFullPval": float(full.loc[full["est"] == "rf", f"{lab}_p"].iloc[0]),
            }
        )
    c = pd.DataFrame(cnt).set_index("which")
    for nm in ("rhrVsBmPoolLinFullSig", "rhrVsBmPoolTreeFullSig", "rhrVsBmPoolHarSig", "rhrVsBmPoolHarSigOne",
               "rhrVsBmPoolNarrow"):  # fmt: skip
        gate(
            f"{nm} old reproduces the printed count",
            int(c.loc["old", nm]) == int(printed(rr, nm)),
            str(c.loc["old", nm]),
        )
    for nm in ("rhrVsBmPoolLinFullPctMin", "rhrVsBmPoolLinFullPctMax", "rhrVsBmPoolTreeFullPctMin",
               "rhrVsBmPoolTreeFullPctMax", "rhrVsBmPoolHarPctMin", "rhrVsBmPoolHarPctMax"):  # fmt: skip
        gate(
            f"{nm} old reproduces the printed range",
            near_printed(c.loc["old", nm], printed(rr, nm), 1),
            str(c.loc["old", nm]),
        )
    return v, c.reset_index()


# ============================================================================================ 3. sign strategy
def strategy(old: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    mt, model_set, series, fam = me.load()
    me.gate_against_master(mt, series, [em.SHORT, CI])
    me.add_pooled_linear(series, fam)
    me.add_pooled_trees(series, fam)
    em.add_har_trees(series)
    b = series[CI]
    days = b.index
    daily = pd.read_parquet(me.RES / "master_table_daily.parquet", columns=["key", "day", "plain", "iv_var", "q"])
    daily = daily[daily["key"] == CI].assign(day=lambda x: pd.to_datetime(x["day"]).dt.normalize())
    ref = daily.set_index("day").sort_index()
    ref.index = ref.index.astype(days.dtype)
    ref = ref.reindex(days)
    iv = ref["iv_var"].to_numpy(float)
    assert np.isfinite(iv).all()

    def frame(f: pd.DataFrame) -> pd.DataFrame:
        p = f["plain"].reindex(days).to_numpy(float)
        assert np.isfinite(p).all() and (p > 0).all()
        q = np.where(p > iv, 1.0, -1.0)
        g = b.copy()
        g["q"], g["s"] = q, p - iv
        g["mid"] = q * b["R"].to_numpy(float)
        g["x"] = gui._crossed_return(q, b["ex"].to_numpy(), b["bid"].to_numpy(), b["ask"].to_numpy())
        return g

    g_old, g_new = frame(old), frame(new)
    rel = float(np.max(np.abs(old["plain"].reindex(days).to_numpy(float) / ref["plain"].to_numpy(float) - 1.0)))
    gate("old file plain == master table plain on the 866 days", rel <= 1e-10, f"max rel {rel:.2e}")
    gate(
        "old file's trade == the master table's (q, mid, bid/ask) on the 866 days",
        np.array_equal(g_old["q"].to_numpy(float), b["q"].to_numpy(float))
        and np.allclose(g_old["mid"], b["mid"], rtol=0, atol=1e-10)
        and np.allclose(g_old["x"], b["x"], rtol=0, atol=1e-10),
        f"q equal on {int((g_old['q'] == b['q']).sum())} of {len(days)}",
    )
    full = [em.key(e, "all_features") for e in EST]
    har4 = [em.key("ridge", "baseline")] + [em.key(e, "baseline") for e in em.TREES]
    out_rows, pair_rows = [], []
    res: dict = {}
    for lab, g in (("old", g_old), ("new", g_new)):
        series[CI] = g
        # make_rehaul_economics caches its bootstrap / Ledoit-Wolf p by (key_a, key_b, col), not by the returns: clear
        # them, or the new CI-HAR would reuse the old one's p-values
        me.BOOT_CACHE.clear()
        me.LW_CACHE.clear()
        st = em.row_stats(series, CI)
        pairs = {k: em.pair(series, k, CI) for k in [em.key(e, s) for s in SETS for e in EST]}
        pw = slice(me.WARMUP, None)
        out_rows.append(
            {
                "which": lab,
                "sh_mid": st["sh_mid"],
                "sh_x": st["sh_x"],
                "mean_mid_pct": 100 * st["mean_mid"],
                "mean_x_pct": 100 * st["mean_x"],
                "dsh_short": st["dsh_short"],
                "p_short": st["p_short"],
                "dsh_short_x": st["dsh_short_x"],
                "p_short_x": st["p_short_x"],
                "t_short": st["t_short"],
                "sh_d1": st["sh_d1"],
                "sh_d2": st["sh_d2"],
                "be_mean": st["be_mean"],
                "be_sh": st["be_sh"],
                "long_days_614": int((g["q"].to_numpy()[pw] > 0).sum()),
                "long_days_866": int((g["q"].to_numpy() > 0).sum()),
                **{f"sr_{y}": v for y, v in st["by_year"].items()},
            }
        )
        for k, p in pairs.items():
            pair_rows.append({"which": lab, "key": k, **p})
        d_ci = [pairs[k]["dsh"] for k in full]
        res[lab] = {
            "FullVsCiHarMin": min(d_ci),
            "FullVsCiHarMed": float(np.median(d_ci)),
            "FullVsCiHarMax": max(d_ci),
            "FullVsCiHarPMin": min(pairs[k]["p"] for k in full),
            "FullVsCiHarPMax": max(pairs[k]["p"] for k in full),
            "FullVsCiHarPos": sum(pairs[k]["dsh"] > 0 for k in full),
            "FullVsCiHarSig": sum(pairs[k]["p"] < 0.05 for k in full),
            "FullVsCiHarPosX": sum(pairs[k]["dsh_x"] > 0 for k in full),
            "FullVsCiHarSigX": sum(pairs[k]["p_x"] < 0.05 for k in full),
            "HarVsCiHarNeg": sum(pairs[k]["dsh"] < 0 for k in har4),
            "HarVsCiHarSig": sum(pairs[k]["p"] < 0.05 for k in har4),
            "claim_every_full_above_ci": all(pairs[k]["dsh"] > 0 for k in full),
            "claim_no_full_minus_ci_significant": all(pairs[k]["p"] >= 0.05 and pairs[k]["p_x"] >= 0.05 for k in full),
            "claim_all_four_har_below_ci": all(pairs[k]["dsh"] < 0 for k in har4),
            "q_changed_days_866": int((g["q"].to_numpy() != g_old["q"].to_numpy()).sum()),
            "q_changed_days_614": int((g["q"].to_numpy()[pw] != g_old["q"].to_numpy()[pw]).sum()),
        }
        if lab == "old":
            ee = em._macros(GEN / "rehaul_econ_main_numbers.tex")
            rr = em._macros(GEN / "rehaul_results_numbers.tex")
            for nm, x, dp in (
                ("rhrOlsSharpe", st["sh_mid"], 2),
                ("rhemShCiHar", st["sh_mid"], 2),
                ("rhemShXCiHar", st["sh_x"], 2),
                ("rhemDShCiHarVsShort", st["dsh_short"], 2),
                ("rhemPDShCiHarVsShort", st["p_short"], 3),
                ("rhemFullVsCiHarMin", res[lab]["FullVsCiHarMin"], 2),
                ("rhemFullVsCiHarMed", res[lab]["FullVsCiHarMed"], 2),
                ("rhemFullVsCiHarMax", res[lab]["FullVsCiHarMax"], 2),
                ("rhemHarVsCiHarNeg", res[lab]["HarVsCiHarNeg"], 0),
                ("rhemHarVsCiHarSig", res[lab]["HarVsCiHarSig"], 0),
            ):
                src = rr if nm.startswith("rhr") else ee
                gate(f"{nm} old reproduces the printed value", near_printed(float(x), printed(src, nm), dp), f"{x}")
    return pd.DataFrame(out_rows), pd.DataFrame(pair_rows), res


# ============================================================================================ 4. the B days
def b_days(old: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """correctness-6: the days on which the old file's B differs from the pooled OLS's B at the 16:00 bar."""
    f = pl13._frame()
    g = f[(f["model"] == "ols") & (f["predictor_set"] == "baseline")]
    t = pd.to_datetime(g["date"])
    g = g[((t.dt.hour == 16) & (t.dt.minute == 0)).to_numpy()]
    bp = pd.Series(g["B"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(g["date"])).normalize().as_unit("ns"))
    bp = bp.sort_index()
    d = old.index.intersection(bp.index)
    ratio_old = old["B"].reindex(d) / bp.reindex(d)
    diff = d[np.abs(ratio_old.to_numpy() - 1.0) > 1e-9]
    gate(
        "correctness-6 reproduced: old B differs from the pooled B on 174 of 1,460 days, ratio min 0.773",
        len(d) == 1460 and len(diff) == 174 and round(float(ratio_old.min()), 3) == 0.773,
        f"{len(diff)} of {len(d)}, min ratio {float(ratio_old.min()):.4f}",
    )
    dn = new.index.intersection(bp.index)
    ratio_new = new["B"].reindex(dn) / bp.reindex(dn)
    diff_new = dn[np.abs(ratio_new.to_numpy() - 1.0) > 1e-9]
    rv = old["rv"].reindex(diff)
    qo = qlike(rv.to_numpy(), old["plain"].reindex(diff).to_numpy())
    qn = qlike(rv.to_numpy(), new["plain"].reindex(diff).to_numpy())
    tab = pd.DataFrame(
        {
            "B_old": old["B"].reindex(diff),
            "B_new": new["B"].reindex(diff),
            "B_pool": bp.reindex(diff),
            "ratio_old_pool": ratio_old.reindex(diff),
            "ratio_new_pool": (new["B"].reindex(diff) / bp.reindex(diff)),
            "plain_old": old["plain"].reindex(diff),
            "plain_new": new["plain"].reindex(diff),
            "rv": rv,
            "qlike_old": qo,
            "qlike_new": qn,
        }
    )
    tab.index.name = "day"
    toos = tab.index >= pd.Timestamp("2020-01-03")
    s = {
        "n_days_old_B_ne_pool": len(diff),
        "n_common_old_pool": len(d),
        "of_those_new_B_eq_pool": int((np.abs(tab["ratio_new_pool"] - 1.0) <= 1e-9).sum()),
        "max_abs_rel_new_vs_pool_on_those": float(np.max(np.abs(tab["ratio_new_pool"] - 1.0))),
        "n_days_new_B_ne_pool_all": len(diff_new),
        "n_common_new_pool": len(dn),
        "max_abs_rel_new_vs_pool_all": float(np.max(np.abs(ratio_new.to_numpy() - 1.0))),
        "qlike_old_on_174": float(qo.mean()),
        "qlike_new_on_174": float(qn.mean()),
        "n_174_in_T_oos": int(toos.sum()),
        "qlike_old_on_174_T_oos": float(qo[toos].mean()),
        "qlike_new_on_174_T_oos": float(qn[toos].mean()),
    }
    return tab, s


def attribution() -> pd.DataFrame:
    """Which of the three biting flags moves the CI-HAR: the build's variants (runs/<v>) scored like the files, on the
    days every variant has. old = the old-rules rebuild; b = + B_REGULAR_BARS_ONLY; b_ec = + DROP_EARLY_CLOSE_SESSIONS;
    new = + BURNIN_IN_BAR_SPACE (= runs/three, = the rebuilt file)."""
    fr_ = {v: ci_scored(OUT / "runs" / v / "results_bar1600.csv") for v in ("old", "b", "b_ec", "new")}
    hist = rolling.history()
    rows = []
    spans = {
        "2018-06-25..2024-04-30": (pd.Timestamp("2018-06-25"), pd.Timestamp("2024-04-30"), None),
        "T_oos common span (1,082)": (None, None, rolling.common_span()),
        "master 866 days": (None, None, master_days()),
    }
    for nm, (a, b, idx) in spans.items():
        d = fr_["old"].index
        for f in fr_.values():
            d = d.intersection(f.index)
        if idx is not None:
            d = d.intersection(idx)
        else:
            d = d[(d >= a) & (d <= b)]
        bench = rolling.ct_benchmark(hist, d)
        for v, f in fr_.items():
            m = rolling.metrics(f["rv"].reindex(d), f["plain"].reindex(d), bench)
            lr = np.abs(np.log(f["plain"].reindex(d).to_numpy() / fr_["old"]["plain"].reindex(d).to_numpy()))
            dm = rolling.dm_pair(f["rv"].reindex(d), f["plain"].reindex(d), fr_["old"]["plain"].reindex(d))
            rows.append(
                {
                    "span": nm,
                    "variant": v,
                    "n": len(d),
                    "qlike": m["qlike"],
                    "mse_e10": 1e10 * m["mse"],
                    "dm_qlike_vs_old": dm["dm_qlike"] if v != "old" else np.nan,
                    "median_abs_log_plain_vs_old": float(np.median(lr)),
                    "max_abs_log_plain_vs_old": float(np.max(lr)),
                    "days_B_differs_from_old": int(
                        (np.abs(f["B"].reindex(d) / fr_["old"]["B"].reindex(d) - 1) > 1e-12).sum()
                    ),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(MEAS / "attribution.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(out.to_string(index=False))
    return out


def sharpe_attribution() -> pd.DataFrame:
    """The CI-HAR sign strategy of each build variant (runs/<v>), traded as strategy() trades it: the D6 Sharpe ratio
    at mid and bid/ask (me.sharpe, 614 post-warmup days), positions changed against the old file, and the flipped
    post-warmup days with their mid returns (flips.csv)."""
    mt, model_set, series, fam = me.load()
    b = series[CI]
    days = b.index
    daily = pd.read_parquet(me.RES / "master_table_daily.parquet", columns=["key", "day", "iv_var"])
    daily = daily[daily["key"] == CI].assign(day=lambda x: pd.to_datetime(x["day"]).dt.normalize())
    ref = daily.set_index("day").sort_index()
    ref.index = ref.index.astype(days.dtype)
    iv = ref["iv_var"].reindex(days).to_numpy(float)
    R = b["R"].to_numpy(float)
    files = {"old file": OLD_PATH} | {v: OUT / "runs" / v / "results_bar1600.csv" for v in ("old", "b", "b_ec", "new")}
    q0 = None
    rows, flips = [], []
    pw = slice(me.WARMUP, None)
    for v, path in files.items():
        p = ci_scored(path)["plain"].reindex(days).to_numpy(float)
        q = np.where(p > iv, 1.0, -1.0)
        mid = q * R
        x = gui._crossed_return(q, b["ex"].to_numpy(), b["bid"].to_numpy(), b["ask"].to_numpy())
        if q0 is None:
            q0 = q
            assert np.array_equal(q, b["q"].to_numpy(float))
        ch = q[pw] != q0[pw]
        rows.append(
            {
                "variant": v,
                "sh_mid": me.sharpe(mid),
                "sh_x": me.sharpe(x),
                "mean_mid_pct": 100 * float(mid[pw].mean()),
                "q_changed_614": int(ch.sum()),
                "to_long": int((ch & (q[pw] > 0)).sum()),
                "to_short": int((ch & (q[pw] < 0)).sum()),
                "sum_mid_change_on_flips_pct": 100 * float((mid[pw] - q0[pw] * R[pw])[ch].sum()),
            }
        )
        if v == "new":
            d = days[pw][ch]
            flips = [
                {"day": dd.date(), "q_old": q0[pw][i], "q_new": q[pw][i], "R_pct": 100 * R[pw][i]}
                for dd, i in zip(d, np.flatnonzero(ch))
            ]
    out = pd.DataFrame(rows)
    out.to_csv(MEAS / "sharpe_attribution.csv", index=False)
    pd.DataFrame(flips).to_csv(MEAS / "flips.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(out.to_string(index=False))
        print(pd.DataFrame(flips).to_string(index=False))
    return out


def model_set() -> pd.DataFrame:
    """Section 7's closing-interval-scheme counts against the CI-HAR (make_rehaul_results: \\rhrNSet, \\rhrNLower,
    \\rhrNLowerSig, \\rhrNWorseSig over the model set's 93 other forecasts, pair() on T_oos; \\rhrSETopFiveOls, the
    share of the CI-HAR's squared error on its 5 largest days; the closing-interval lasso's MSE change against it
    without 24 Feb .. 30 Apr 2020), old and new CI-HAR; the old values gated against the printed macros."""
    days = rolling.common_span()
    rv = rolling.history().reindex(days).to_numpy(float)
    ms = [k for k in pd.read_csv(me.RES / "paper_model_set.csv")["key"] if k != CI]
    assert len(ms) == 93, len(ms)
    ci = {lab: ci_scored(p)["plain"].reindex(days).to_numpy(float) for lab, p in (("old", OLD_PATH), ("new", NEW_PATH))}
    crash = (days >= pd.Timestamp("2020-02-24")) & (days <= pd.Timestamp("2020-04-30"))  # make_rehaul_results.CRASH
    rows, per = [], []
    for k in ms:
        p = rolling.scored(k)["plain"].reindex(days).to_numpy(float)
        assert np.isfinite(p).all() and (p > 0).all(), k
        r: dict = {"key": k}
        for lab, c in ci.items():
            qa, qb = qlike(rv, p), qlike(rv, c)
            dm = dm_test(qa, qb)
            r[f"{lab}_pct"] = 100.0 * (qa.mean() / qb.mean() - 1.0)
            r[f"{lab}_dm"], r[f"{lab}_p"] = float(dm["dm"]), float(dm["p"])
        per.append(r)
    pf = pd.DataFrame(per)
    pf.to_csv(MEAS / "model_set_vs_ci.csv", index=False)
    rr = em._macros(GEN / "rehaul_results_numbers.tex")
    la = rolling.scored("subpt_lasso_all_features")["plain"].reindex(days).to_numpy(float)
    for lab, c in ci.items():
        e2 = (rv - c) ** 2
        top5 = 100.0 * np.sort(e2)[-5:].sum() / e2.sum()
        ex = ~crash
        lasso_ex = 100.0 * (((rv - la) ** 2)[ex].mean() / e2[ex].mean() - 1.0)
        rows.append(
            {
                "which": lab,
                "rhrNSet": len(pf),
                "rhrNLower": int((pf[f"{lab}_pct"] < 0).sum()),
                "rhrNLowerSig": int(((pf[f"{lab}_pct"] < 0) & (pf[f"{lab}_p"] < 0.05)).sum()),
                "rhrNWorseSig": int(((pf[f"{lab}_pct"] > 0) & (pf[f"{lab}_p"] < 0.05)).sum()),
                "rhrSETopFiveOls": top5,
                "lasso_full_mse_pct_vs_ci_ex_feb_apr_2020": lasso_ex,
            }
        )
    out = pd.DataFrame(rows).set_index("which")
    for nm in ("rhrNSet", "rhrNLower", "rhrNLowerSig", "rhrNWorseSig"):
        gate(
            f"{nm} old reproduces the printed count",
            int(out.loc["old", nm]) == int(printed(rr, nm)),
            str(out.loc["old", nm]),
        )
    gate(
        "rhrSETopFiveOls old reproduces the printed share",
        near_printed(float(out.loc["old", "rhrSETopFiveOls"]), printed(rr, "rhrSETopFiveOls"), 0),
        f"{out.loc['old', 'rhrSETopFiveOls']:.2f}",
    )
    out.reset_index().to_csv(MEAS / "model_set_counts.csv", index=False)
    lower_new = pf[(pf["new_pct"] < 0)]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(out.T.to_string())
        print("lower than the new CI-HAR:")
        print(lower_new.to_string(index=False))
        print("lower than the old CI-HAR:")
        print(pf[pf["old_pct"] < 0].to_string(index=False))
    return out


def vsbm_2014() -> pd.DataFrame:
    """The 12 pooled forecasts against the NEW CI-HAR on Table 1's span 2014-01-02 .. 2024-04-30 and on its Pre
    subperiod 2014-01-02 .. 2019-12-31 (the old CI-HAR starts 2018-06-25, so these comparisons exist only for the new
    one): QLIKE % change and DM, squared-error % change and DM (Section 6's pair() statistics on other days)."""
    new = ci_scored(NEW_PATH)
    hist = rolling.history()
    rows = []
    for nm, (a, b) in (
        ("2014-01-02..2024-04-30", (pd.Timestamp("2014-01-02"), pd.Timestamp("2024-04-30"))),
        ("2014-01-02..2019-12-31", (pd.Timestamp("2014-01-02"), pd.Timestamp("2019-12-31"))),
    ):
        days = new.index[(new.index >= a) & (new.index <= b)]
        rv = hist.reindex(days).to_numpy(float)
        c = new["plain"].reindex(days).to_numpy(float)
        for e in EST:
            for s in SETS:
                g = (pl13 if e in LIN else pt13).at16(e, s)
                miss = days[~days.isin(g.index)]
                assert len(miss) == 0, (e, s, len(miss))
                assert np.max(np.abs(g["rv"].reindex(days).to_numpy(float) / rv - 1.0)) <= 1e-12, (e, s)
                p = g["plain"].reindex(days).to_numpy(float)
                qa, qb = qlike(rv, p), qlike(rv, c)
                dm, dms = dm_test(qa, qb), dm_test((rv - p) ** 2, (rv - c) ** 2)
                rows.append(
                    {
                        "span": nm,
                        "n": len(days),
                        "est": e,
                        "set": s,
                        "pct": 100.0 * (qa.mean() / qb.mean() - 1.0),
                        "dm": float(dm["dm"]),
                        "p": float(dm["p"]),
                        "mse_pct": 100.0 * (((rv - p) ** 2).mean() / ((rv - c) ** 2).mean() - 1.0),
                        "dm_mse": float(dms["dm"]),
                        "p_mse": float(dms["p"]),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(MEAS / "vsbm_2014.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(out.round(4).to_string(index=False))
    return out


def main() -> None:
    MEAS.mkdir(parents=True, exist_ok=True)
    if "--vsbm-2014" in sys.argv:
        vsbm_2014()
        return
    if "--model-set" in sys.argv:
        model_set()
        pd.DataFrame(GATES).to_csv(MEAS / "gates_model_set.csv", index=False)
        return
    if "--attribution" in sys.argv:
        attribution()
        return
    if "--sharpe-attribution" in sys.argv:
        sharpe_attribution()
        return
    old, new = ci_scored(OLD_PATH), ci_scored(NEW_PATH)
    ref = rolling.scored(CI)
    gate(
        "ci_scored(old file) == rolling.scored(sub_ols_baseline)",
        ref.index.equals(old.index)
        and np.array_equal(ref["plain"].to_numpy(), old["plain"].to_numpy())
        and np.array_equal(ref["rv"].to_numpy(), old["rv"].to_numpy()),
        f"{len(old)} days",
    )
    early = pd.read_csv(OLD_PATH, parse_dates=["date"])
    n_ec_old = int(pd.DatetimeIndex(early["date"]).normalize().isin(rolling.EARLY).sum())
    newraw = pd.read_csv(NEW_PATH, parse_dates=["date"])
    n_ec_new = int(pd.DatetimeIndex(newraw["date"]).normalize().isin(rolling.EARLY).sum())
    print(f"rows on early-close days: old file {n_ec_old}, new file {n_ec_new}")
    print(f"new file: {len(newraw)} rows {newraw['date'].min()} .. {newraw['date'].max()}; scored days {len(new)}")

    acc, dmo = accuracy(old, new)
    mref = pd.read_csv(REPO / "results" / "rolling_eval_2026-10-15" / "metrics.csv")
    mref = mref[mref["key"] == CI].set_index("span")
    for sp, nm in (("common", "T_oos common span (1,082)"), ("mt866", "master 866 days")):
        r = acc[(acc["span"] == nm) & (acc["file"] == "old") & (acc["days"] == "own")].iloc[0]
        gate(
            f"old QLIKE / MSE on {sp} == rolling_eval metrics.csv",
            np.isclose(r["qlike"], mref.loc[sp, "qlike"], rtol=1e-12, atol=0)
            and np.isclose(r["mse_e10"] * 1e-10, mref.loc[sp, "mse"], rtol=1e-12, atol=0),
            f"{r['qlike']:.12f} / {r['mse_e10']:.6f}e-10",
        )
    rr = em._macros(GEN / "rehaul_results_numbers.tex")
    r = acc[(acc["span"] == "T_oos common span (1,082)") & (acc["file"] == "old") & (acc["days"] == "own")].iloc[0]
    gate("rhrOlsQ reproduced", f"{r['qlike']:.4f}" == em._norm(rr["rhrOlsQ"]), f"{r['qlike']:.4f}")
    acc.to_csv(MEAS / "accuracy.csv", index=False)
    dmo.to_csv(MEAS / "dm_old_vs_new.csv", index=False)
    attribution()

    v, c = vsbm(old, new)
    v.to_csv(MEAS / "vsbm.csv", index=False)
    c.to_csv(MEAS / "vsbm_counts.csv", index=False)

    tab, bs = b_days(old, new)
    tab.to_csv(MEAS / "b_days.csv")

    st, pairs, res = strategy(old, new)
    st.to_csv(MEAS / "sharpe.csv", index=False)
    pairs.to_csv(MEAS / "sharpe_pairs.csv", index=False)

    summary = {"early_close_rows": {"old": n_ec_old, "new": n_ec_new}, "b_days": bs, "strategy": res}
    (MEAS / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    pd.DataFrame(GATES).to_csv(MEAS / "gates.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 200):
        print(acc.to_string(index=False))
        print(dmo.to_string(index=False))
        print(v[["key", "old_pct", "old_dm", "old_p", "new_pct", "new_dm", "new_p"]].to_string(index=False))
        print(c.T.to_string())
        print(st.T.to_string())
        print(json.dumps(summary, indent=1, default=str))
    print(f"all {len(GATES)} gates pass")


if __name__ == "__main__":
    main()
