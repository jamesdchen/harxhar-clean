"""Stage 2 (reduce): the feature-importance tables of the pooled 13-bar linear models, monthly re-estimations.

Measures and conventions of experiments/rehaul_fi_compute.py (main worktree; its ranks_desc / kendall_w / rank_table
and BUCKET_ORDER are imported, not copied): coefficients read at the re-estimation's first forecast row, intercept
dropped; nonzero = |beta| > 0 (NZ_TOL = 0); predictor group = all columns of one raw series (value and availability
averages at every horizon), plus har_ma (the target's HAR components) and calendar (DOW_0..4, hour), both in category
"backbone" (the HAR benchmark block, penalty factor 0.01); window sd = np.std (ddof 0) of the prescaled design column
over the estimation window rows [t - 13,000, t); per-re-estimation rankings by ||beta_A||_1 (ranks 1 = largest, ties
averaged; for the ridge the same order as S_kA), Kendall's W with the tie correction over all groups.

Coefficients: the walk's own (fi13_replay.py, walk_theta.npz).  The batch re-estimations (fi13_fit.py, fit.npz) are
used only for the sensitivity check in run_info.json.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCR))
import fi13_fit as F  # noqa: E402

OUT = F.MAIN / "results" / "rehaul_fi_pooled13_2026-10-22"
MODELS = ("lasso", "ridge", "enet")

# ls_wt's src (the code that built the design) for the bucket map
sys.path.insert(0, str(F.LS))
from src.data import loading as LD  # noqa: E402

# experiments/rehaul_fi_compute.py's helpers (main worktree, read-only); its import chdirs to the main worktree
_cwd = Path.cwd()
_spec = importlib.util.spec_from_file_location("rehaul_fi_compute", F.MAIN / "experiments" / "rehaul_fi_compute.py")
RFC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RFC)
import os  # noqa: E402

os.chdir(_cwd)


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git_head(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def git_dirty(repo: Path, paths: list[str]) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--", *paths], capture_output=True, text=True
    ).stdout.strip()


def series_of(c: str) -> str:
    if re.fullmatch(r"har_ma_\d+", c):
        return "har_ma"
    if c in {f"DOW_{d}" for d in range(5)} | {"hour"}:
        return "calendar"
    m = re.fullmatch(r"adj_(.+)_ma_\d+", c) or re.fullmatch(r"(.+)_avail_ma_\d+", c)
    assert m, c
    return m.group(1)


def kind_of(c: str) -> str:
    if re.fullmatch(r"har_ma_\d+", c):
        return "har"
    if re.fullmatch(r"adj_.+_ma_\d+", c):
        return "value"
    if re.fullmatch(r".+_avail_ma_\d+", c):
        return "availability"
    return "calendar"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((F.DZ / "design.json").read_text())
    names = list(meta["names"])
    p = len(names)
    O = pd.read_csv(F.SCR / "origins.csv")
    K = len(O)
    Z = np.load(F.SCR / "walk_theta.npz")
    B = np.load(F.SCR / "fit.npz")
    assert np.array_equal(B["oos_row"], O["oos_row"].to_numpy())
    sd = B["sd"]  # (K, p)
    gates_fit = json.loads((F.SCR / "gates.json").read_text())
    gates_rep = json.loads((F.SCR / "gates_replay.json").read_text())

    # ---------------------------------------------------------------- groups (rehaul_fi_compute.data()'s mapping)
    series = [series_of(c) for c in names]
    order: list[str] = []
    for s in series:
        if s not in order:
            order.append(s)
    bmap = {"har_ma": "backbone", "calendar": "backbone"}
    for b in RFC.BUCKET_ORDER[1:]:
        for s in LD.SUBGROUPS[b]:
            assert s not in bmap, s
            bmap[s] = b
    assert set(order) == set(bmap), set(order) ^ set(bmap)
    units = sorted(order, key=lambda s: (RFC.BUCKET_ORDER.index(bmap[s]), order.index(s)))
    U = len(units)
    n_exog = sum(bmap[s] != "backbone" for s in units)
    sidx = np.array([units.index(s) for s in series])
    G = np.zeros((p, U))
    G[np.arange(p), sidx] = 1.0
    ncol = G.sum(axis=0).astype(int)
    bb_col = np.array([bmap[s] == "backbone" for s in series])
    har_col = np.array([s == "har_ma" for s in series])
    cal_col = np.array([s == "calendar" for s in series])
    exog_u = np.array([bmap[u] != "backbone" for u in units])
    assert bb_col.sum() == 12 and har_col.sum() == 6 and cal_col.sum() == 6
    kinds = np.array([kind_of(c) for c in names])

    grp = pd.DataFrame(dict(group=units, category=[bmap[u] for u in units], n_cols=ncol))
    prd = pd.DataFrame(
        dict(
            column=names,
            group=series,
            category=[bmap[s] for s in series],
            kind=kinds,
            horizon=[int(c.rsplit("_", 1)[1]) if re.search(r"_ma_\d+$", c) else np.nan for c in names],
        )
    )
    ref = O[["k", "origin", "oos_row", "design_row", "block_rows", "block_sessions", "block_last"]].copy()
    live_r = ~Z["ridge_mask"][:, :-1]  # the tune-boundary mask in force at each origin
    grp["K_live"] = ((live_r.astype(float) @ G) > 0).sum(axis=0)
    prd["K_live"] = live_r.sum(axis=0)
    prd["sd_mean"] = sd.mean(axis=0)
    prd["sd_median"] = np.median(sd, axis=0)
    ref["n_masked"] = (~live_r).sum(axis=1)
    ref["n_groups_live"] = ((live_r.astype(float) @ G) > 0).sum(axis=1)
    ref["n_exog_groups_live"] = ((live_r.astype(float) @ G) > 0)[:, exog_u].sum(axis=1)

    summ: dict = {}
    long: dict = {}
    for m in MODELS:
        th = Z[f"{m}_theta"][:, :-1]
        live = ~Z[f"{m}_mask"][:, :-1]
        absb = np.abs(th)
        nz = absb > RFC.NZ_TOL
        assert not (nz & ~live).any(), m  # masked columns carry 0
        ser_abs = absb @ G  # ||beta_kA||_1
        ser_sd = (absb * sd) @ G
        act = nz.astype(float) @ G  # active columns per group
        sel = act > 0
        glive = (live.astype(float) @ G) > 0
        df_s, R = RFC.rank_table(ser_abs, units)
        W_all = RFC.kendall_w(R)
        Rx = np.vstack([RFC.ranks_desc(r) for r in ser_abs[:, exog_u]])
        W_exog = RFC.kendall_w(Rx)
        mr = dict(zip(df_s["unit"], df_s["mean_rank"]))
        KA = sel.sum(axis=0)
        S_A = ser_abs.sum(axis=0) / ser_abs.sum()
        S_sd = ser_sd.sum(axis=0) / ser_sd.sum()
        pre = f"{m}_"
        if m in ("lasso", "enet"):
            grp[pre + "K_A"] = KA
            grp[pre + "K_A_over_K"] = KA / K
            grp[pre + "K_A_over_K_live"] = np.where(glive.sum(0) > 0, KA / np.maximum(glive.sum(0), 1), np.nan)
            grp[pre + "mean_active_cols"] = act.mean(axis=0)
            grp[pre + "S_A"] = S_A
            grp[pre + "S_sd_A"] = S_sd
            grp[pre + "mean_rank_l1"] = [mr[u] for u in units]
            # rank by K_A (ties: shared, method min), and a strict order breaking ties by S_A
            grp[pre + "rank_K_A"] = pd.Series(-KA).rank(method="min").astype(int).to_numpy()
            o = np.lexsort((-S_A, -KA))
            rk = np.empty(U, int)
            rk[o] = np.arange(1, U + 1)
            grp[pre + "rank_K_A_then_S_A"] = rk
            prd[pre + "K_j"] = nz.sum(axis=0)
            prd[pre + "sum_abs"] = absb.sum(axis=0)
            prd[pre + "sum_abs_sd"] = (absb * sd).sum(axis=0)
        else:
            grp[pre + "S_A"] = S_A
            grp[pre + "S_sd_A"] = S_sd
            grp[pre + "rank_S_A"] = pd.Series(-S_A).rank(method="min").astype(int).to_numpy()
            grp[pre + "rank_S_sd_A"] = pd.Series(-S_sd).rank(method="min").astype(int).to_numpy()
            grp[pre + "mean_rank_S_kA"] = [mr[u] for u in units]
            Rsd = np.vstack([RFC.ranks_desc(r) for r in ser_sd])
            df_sd, _ = RFC.rank_table(ser_sd, units)
            mr_sd = dict(zip(df_sd["unit"], df_sd["mean_rank"]))
            grp[pre + "mean_rank_S_sd_kA"] = [mr_sd[u] for u in units]
            prd[pre + "K_j"] = nz.sum(axis=0)
            prd[pre + "sum_abs"] = absb.sum(axis=0)
            prd[pre + "sum_abs_sd"] = (absb * sd).sum(axis=0)
        tot, tot_sd = absb.sum(axis=1), (absb * sd).sum(axis=1)
        ref[pre + "nz_total"] = nz.sum(axis=1)
        ref[pre + "nz_exog"] = nz[:, ~bb_col].sum(axis=1)
        ref[pre + "n_exog_groups_selected"] = sel[:, exog_u].sum(axis=1)
        ref[pre + "har_block_share"] = absb[:, bb_col].sum(axis=1) / tot
        ref[pre + "har_only_share"] = absb[:, har_col].sum(axis=1) / tot
        ref[pre + "har_block_share_sd"] = (absb * sd)[:, bb_col].sum(axis=1) / tot_sd
        Sk = ser_abs / tot[:, None]
        ref[pre + "top_group"] = [units[j] for j in Sk.argmax(axis=1)]
        xi = np.flatnonzero(exog_u)
        ref[pre + "top_exog_group"] = [units[xi[j]] for j in Sk[:, xi].argmax(axis=1)]
        ref[pre + "top_exog_group_share"] = Sk[:, xi].max(axis=1)
        ref[pre + "pred_at_origin"] = Z[f"{m}_pred"]
        # time series, long (origin x group)
        rows = dict(
            k=np.repeat(O["k"].to_numpy(), U),
            origin=np.repeat(O["origin"].to_numpy(), U),
            group=np.tile(units, K),
            category=np.tile([bmap[u] for u in units], K),
            live=glive.ravel().astype(int),
        )
        if m == "ridge":
            rows.update(
                S_kA=Sk.ravel(),
                S_sd_kA=(ser_sd / tot_sd[:, None]).ravel(),
                l1_norm=ser_abs.ravel(),
                rank_S_kA=R.ravel(),
            )
        else:
            rows.update(
                selected=sel.ravel().astype(int),
                l1_norm=ser_abs.ravel(),
                n_active_cols=act.ravel().astype(int),
                S_kA=Sk.ravel(),
                rank_l1=R.ravel(),
            )
        long[m] = pd.DataFrame(rows)
        exog_sel = sel[:, exog_u]
        summ[m] = dict(
            alpha=float(json.loads((F.MAIN_RUN / "frozen_penalties.json").read_text())[m]["all_features"]),
            K=K,
            W_all_groups=float(W_all),
            W_exog_groups=float(W_exog),
            nz_total_median=float(np.median(nz.sum(1))),
            nz_total_min=int(nz.sum(1).min()),
            nz_total_max=int(nz.sum(1).max()),
            nz_exog_median=float(np.median(nz[:, ~bb_col].sum(1))),
            nz_exog_min=int(nz[:, ~bb_col].sum(1).min()),
            nz_exog_max=int(nz[:, ~bb_col].sum(1).max()),
            live_cols_median=float(np.median(live.sum(1))),
            cols_ever=int(nz.any(0).sum()),
            cols_majority=int((nz.mean(0) >= 0.5).sum()),
            exog_groups_ever=int(exog_sel.any(0).sum()),
            exog_groups_majority=int((exog_sel.mean(0) >= 0.5).sum()),
            exog_groups_always=int(exog_sel.all(0).sum()),
            exog_groups_always_list=[u for u, a in zip(np.array(units)[exog_u], exog_sel.all(0)) if a],
            har_block_share_median=float(np.median(absb[:, bb_col].sum(1) / tot)),
            har_block_share_pooled=float(absb[:, bb_col].sum() / absb.sum()),
            har_only_share_median=float(np.median(absb[:, har_col].sum(1) / tot)),
            calendar_share_median=float(np.median(absb[:, cal_col].sum(1) / tot)),
            har_block_share_sd_median=float(np.median((absb * sd)[:, bb_col].sum(1) / tot_sd)),
            har_block_share_sd_pooled=float((absb * sd)[:, bb_col].sum() / (absb * sd).sum()),
            har_block_selected_every_refit=bool(sel[:, [units.index("har_ma"), units.index("calendar")]].all()),
            spearman_S_A_vs_S_sd_A_all=float(spearmanr(S_A, S_sd)[0]),
            spearman_S_A_vs_S_sd_A_exog=float(spearmanr(S_A[exog_u], S_sd[exog_u])[0]),
        )
        if m != "ridge":
            summ[m]["spearman_K_A_vs_S_A_exog"] = float(spearmanr(KA[exog_u], S_A[exog_u])[0])
        else:
            summ[m]["W_all_groups_sd"] = float(RFC.kendall_w(Rsd))
            summ[m]["W_exog_groups_sd"] = float(
                RFC.kendall_w(np.vstack([RFC.ranks_desc(r) for r in ser_sd[:, exog_u]]))
            )
        # sensitivity: the same group measures from the batch re-estimations (fit.npz)
        thb = np.abs(B[f"{m}_theta"][:, :-1])
        selb = ((thb > 0).astype(float) @ G) > 0
        Sb = (thb @ G).sum(0) / thb.sum()
        Ssdb = ((thb * sd) @ G).sum(0) / (thb * sd).sum()
        summ[m]["batch_vs_walk"] = dict(
            K_A_max_abs_diff=int(np.abs(selb.sum(0) - KA).max()),
            K_A_groups_differing=int((selb.sum(0) != KA).sum()),
            S_A_max_abs_diff=float(np.abs(Sb - S_A).max()),
            S_sd_A_max_abs_diff=float(np.abs(Ssdb - S_sd).max()),
            spearman_S_A=float(spearmanr(Sb, S_A)[0]),
            nz_total_median_batch=float(np.median((thb > 0).sum(1))),
        )
    # top-10 tables for the report
    tops = {}
    ex = grp[grp.category != "backbone"]
    for m in ("lasso", "enet"):
        t = grp.sort_values([f"{m}_K_A", f"{m}_S_A"], ascending=False)
        tops[f"{m}_by_K_A_then_S_A"] = t[["group", "category", f"{m}_K_A", f"{m}_K_A_over_K", f"{m}_K_A_over_K_live", f"{m}_S_A"]].head(12).to_dict("records")
    for key in ("ridge_S_A", "ridge_S_sd_A"):
        tops[key] = grp.sort_values(key, ascending=False)[["group", "category", "ridge_S_A", "ridge_S_sd_A", "K_live"]].head(12).to_dict("records")
        tops[key + "_exog"] = ex.sort_values(key, ascending=False)[["group", "ridge_S_A", "ridge_S_sd_A"]].head(10).to_dict("records")

    # ---------------------------------------------------------------- write
    grp.to_csv(OUT / "fi_groups.csv", index=False)
    prd.to_csv(OUT / "fi_predictors.csv", index=False)
    for m in MODELS:
        b = B[f"{m}_pred"]
        ref[f"{m}_batch_vs_walk_rel"] = np.abs(b - Z[f"{m}_pred"]) / np.abs(Z[f"{m}_pred"])
    ref.to_csv(OUT / "fi_refits.csv", index=False)
    long["ridge"].to_csv(OUT / "fi_time_ridge.csv", index=False)
    long["lasso"].to_csv(OUT / "fi_time_lasso.csv", index=False)
    long["enet"].to_csv(OUT / "fi_time_enet.csv", index=False)
    for m in MODELS:
        c = pd.DataFrame(Z[f"{m}_theta"], columns=names + ["intercept"])
        c.insert(0, "origin", O["origin"].to_numpy())
        c.insert(0, "k", O["k"].to_numpy())
        c.to_csv(OUT / f"fi_coefs_{m}.csv", index=False, float_format="%.17g")
    s = pd.DataFrame(sd, columns=names)
    s.insert(0, "origin", O["origin"].to_numpy())
    s.insert(0, "k", O["k"].to_numpy())
    s.to_csv(OUT / "fi_window_sd.csv", index=False, float_format="%.17g")

    fr = json.loads((F.MAIN_RUN / "frozen_penalties.json").read_text())
    files = {
        "ls_wt/experiments/pool_linear_pretune.py": F.LS / "experiments" / "pool_linear_pretune.py",
        "ls_wt/specs/causal_tune_linear.py": F.LS / "specs" / "causal_tune_linear.py",
        "ls_wt/experiments/pooled_linear_rerun_2026_10_19_13.py": F.LS / "experiments" / "pooled_linear_rerun_2026_10_19_13.py",
        "ls_wt/src/backtest/multi_stage.py": F.LS / "src" / "backtest" / "multi_stage.py",
        "ls_wt/src/models/reclasso_har.py": F.LS / "src" / "models" / "reclasso_har.py",
        "ls_wt/src/data/loading.py": F.LS / "src" / "data" / "loading.py",
        "main/experiments/rehaul_fi_compute.py": F.MAIN / "experiments" / "rehaul_fi_compute.py",
        "scratch/fi13_fit.py": F.SCR / "fi13_fit.py",
        "scratch/fi13_replay.py": F.SCR / "fi13_replay.py",
        "scratch/fi13_reduce.py": F.SCR / "fi13_reduce.py",
        "run/forecasts.parquet": F.MAIN_RUN / "forecasts.parquet",
        "run/frozen_penalties.json": F.MAIN_RUN / "frozen_penalties.json",
        "design/design.json": F.DZ / "design.json",
    }
    info = dict(
        created="2026-10-22",
        what="feature importance of the pooled 13-bar linear models (full predictor set), monthly re-estimations",
        design=dict(
            run=str(F.MAIN_RUN),
            design_cache=str(F.DZ),
            shape=meta["shape"],
            n_columns=p,
            n_har_benchmark_block=int(bb_col.sum()),
            har_benchmark_block=[c for c, b in zip(names, bb_col) if b],
            penalty_factor_har_block=F.R_BB,
            window_rows=F.W,
            window_days=1000,
            config=meta["config"],
            raw_series=len(meta["raw_cols"]),
            fomc_release_in_design=any("fomc_release" in c for c in names),
            n_groups=U,
            n_exog_groups=int(n_exog),
            categories={b: [u for u in units if bmap[u] == b] for b in RFC.BUCKET_ORDER},
        ),
        penalties=dict(
            ridge=fr["ridge"]["all_features"],
            lasso=fr["lasso"]["all_features"],
            enet=fr["enet"]["all_features"],
            enet_l1_ratio=0.5,
            units="scikit-learn (Ridge alpha; ElasticNet alpha, l1_ratio), penalty on column j = alpha * pf_j; intercept unpenalized",
        ),
        cadence=dict(
            rule="one re-estimation every 21 sessions (distinct design dates) from the first 2012-01-03 row; origin = the session's first row (10:00 bar); coefficients = the walk's at that row (window rows [t-13000, t))",
            note="sessions are distinct dates in the 13-bar design; 76 of the 3,151 sessions from 2012-01-03 have fewer than 13 rows (early closes / holiday sessions), so blocks are 264-273 rows, not a fixed 273",
            K=K,
            first_origin=str(O["origin"].iloc[0]),
            last_origin=str(O["origin"].iloc[-1]),
            last_block_sessions=int(O["block_sessions"].iloc[-1]),
            origins=O[["k", "origin", "oos_row", "design_row", "block_rows", "block_sessions"]].to_dict("records"),
        ),
        gates=dict(
            batch_refit_vs_published_pred_adj=gates_fit["gates"],
            walk_replay_vs_published_pred_adj=gates_rep["gates"],
            note="ridge: the batch closed form at t matches the published forecast to 1.8e-12 relative; lasso / elastic net: the batch solution differs from the walk's warm-homotopy solution at some origins, so every measure uses the walk's own coefficients (replayed from the last 250-solve tune boundary; every replayed row matches the published pred_adj)",
        ),
        summary=summ,
        top=tops,
        timings=dict(batch_fit_wall_sec=gates_fit["wall_sec"], replay_wall_sec=gates_rep["wall_sec"], workers=gates_rep["workers"]),
        code=dict(
            ls_wt_head=git_head(F.LS),
            ls_wt_branch="insurance-selling-2026-09-19-e34-latestart",
            ls_wt_dirty=git_dirty(F.LS, ["experiments/pool_linear_pretune.py", "specs/causal_tune_linear.py", "src"]),
            main_head=git_head(F.MAIN),
            sha256={k: sha256(v) for k, v in files.items()},
            python=platform.python_version(),
            numpy=np.__version__,
            interpreter=sys.executable,
        ),
        conventions="experiments/rehaul_fi_compute.py: NZ_TOL 0, groups = raw series + har_ma + calendar (category backbone), coefficients at the first forecast row of the re-estimation, intercept dropped, sd = window np.std ddof 0, per-re-estimation ranks by ||beta_A||_1 (rankdata average on -v), Kendall's W with tie correction over all groups; S_sd_A here is a share (normalized), the old reducer stored the unnormalized sum",
        files=dict(
            fi_groups="one row per group: K_live (re-estimations with >= 1 unmasked column); lasso/enet K_A, K_A/K, K_A/K_live, mean active columns, S_A, S_sd_A, mean per-re-estimation rank by ||beta_A||_1, rank by K_A (min) and strict rank K_A then S_A; ridge S_A, S_sd_A and their ranks, mean per-re-estimation ranks",
            fi_predictors="one row per column: group, category, kind, horizon, K_live, window sd (mean, median), per model K_j, sum |beta|, sum |beta| sd",
            fi_refits="one row per re-estimation: origin, block, masked columns, live groups, per model nonzero counts (total, exogenous), selected exogenous groups, HAR-block share (raw, sd), HAR-only share, top group and top exogenous group by S_kA, forecast at the origin, batch-vs-walk relative forecast gap",
            fi_time_ridge="long, origin x group: S_kA, S_sd_kA, ||beta_kA||_1, per-re-estimation rank, live",
            fi_time_lasso="long, origin x group: selected (0/1), ||beta_kA||_1, active columns, S_kA, rank, live",
            fi_time_enet="as fi_time_lasso, elastic net",
            fi_coefs_model="the walk's coefficients at each origin (564 columns + intercept)",
            fi_window_sd="window sd of every column at each origin",
        ),
    )
    (OUT / "run_info.json").write_text(json.dumps(info, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps(dict(summary=summ, top=tops), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
