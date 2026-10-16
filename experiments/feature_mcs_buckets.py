"""The bucket story check (2026-10-12): "no single bucket helps, all features does".

User (2026-10-12, the professor's story): "each individual bucket didn't see much improvement, but all features
saw a discernible difference" -- test it.  Designs (experiments/feature_mcs_runner.py, kinds addb / loob):
  addb_<B>   HAR + calendar + every series of bucket B
  loob_<B>   all features minus every series of bucket B
for the 8 buckets of src/data/loading.py SUBGROUPS that partition the 47 series of all_features (moments,
liquidity, market_ew, market_vw, sentiment, implied_vol, vol_demand, fomc), next to the anchors `baseline`
(HAR + calendar) and `all_features`, under each estimator: the pre-tuned ridge, lasso and elastic net (arm f,
CARC EPYC 7513, the feature-MCS runner) and, once results/feature_mcs_2026-10-08/trees/BUCKETS_DONE exists, the
tree agent's LightGBM / XGBoost / forest (trees/<est>/<model>.npz, the feature-MCS layout).

Per estimator and POOLED (each design's daily loss averaged over the estimators in the pool -- the linear three,
and with the trees all six; the pooled trade = the equal-weight average of the estimators' daily mid P&L):
  pairwise   addb_B vs HAR + calendar, all_features vs HAR + calendar, loob_B vs all_features: the mean losses,
             Diebold-Mariano (src.evaluation.diebold_mariano.dm_test) on QLIKE, winsorized QLIKE and the
             fit-space MSE; sign(s) Sharpe mid / crossed and the Newey-West t of the daily mid P&L difference
             (atm_straddle_lib.newey_west_t), the master table's conventions
  SPA/StepM  (experiments/feature_mcs_spa.py: Hansen's SPA, White's RC, Romano-Wolf StepM, the MCS module's
             circular block bootstrap as counts, B = 10,000, blocks 5 / 10 / 21) on QLIKE, winsorized QLIKE and
             the trade loss:
               A  benchmark HAR + calendar, alternatives {8 addb + all_features}: does any bucket alone, or all
                  features, beat HAR + calendar?
               B  benchmark all_features, alternatives the 8 loob: does removing a bucket HELP?
               C  B reversed (d = loss of loob_B - loss of all_features): does removing a bucket HURT?
Winsorized QLIKE: the MCS rule (daily differentials against the cross-candidate mean clipped at the 1 % / 99 %
points of the pooled differentials) over the estimator's 112 designs (the 96 MCS designs + the 16 bucket
designs); for a pool, over the pooled (averaged) 112 designs.

    python experiments/feature_mcs_buckets.py     (after experiments/feature_mcs_report.py)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _p in (
    REPO,
    REPO / "experiments",
    REPO / "notebooks",
    REPO / "specs",
    REPO / "writeup",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import close_linear_pretune as CLP  # noqa: E402
import feature_mcs_report as R  # noqa: E402
import feature_mcs_runner as FR  # noqa: E402
import feature_mcs_spa as SP  # noqa: E402
from close_kfull_testlib import spa_rc  # noqa: E402

from src.evaluation.diebold_mariano import dm_test  # noqa: E402

OUT = R.OUT
BUCKETS = FR.BUCKETS8
BUCKETS_DONE = R.TREE_DIR / "BUCKETS_DONE"
INTERIM: set[str] = (
    set()
)  # --interim <est,...> (2026-10-13): accept these trees' bucket files without a done flag
BUCKETS_XGB_RF_DONE = (
    R.TREE_DIR / "BUCKETS_XGB_RF_DONE"
)  # 2026-10-13: XGBoost + forest in, LightGBM pending
LOSS3 = ("qlike", "qlike_w", "mse_fit")  # pairwise DM losses
SPA_LOSSES = ("qlike", "qlike_w", "trade")
LIN = R.LIN
TREES = R.TREES


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bucket_preds(sh: str, fdates, day_row) -> dict | None:
    """The 16 bucket designs' forecasts of one estimator (None while any is missing)."""
    md = pd.read_csv(FR.INP / "models.csv")
    names = md[md["kind"].isin(("addb", "loob"))]["model"].tolist()
    if sh in LIN:
        D = FR.models_dir(R.EST_OF[sh])
        if not all((D / f"{m}.npz").is_file() for m in names):
            return None
        return {m: np.load(D / f"{m}.npz")["pred"] for m in names}
    D = R.TREE_DIR / sh
    flag = (
        BUCKETS_DONE.is_file()
        or (sh != "lgbm" and BUCKETS_XGB_RF_DONE.is_file())
        or sh in INTERIM
    )
    if not flag or not all((D / f"{m}.npz").is_file() for m in names):
        return None
    return {m: R.tree_pred(D / f"{m}.npz", fdates, day_row)[0] for m in names}


def main() -> None:  # noqa: C901
    t0 = time.time()
    import atm_straddle_lib as asl

    inp = json.loads((FR.INP / "inputs.json").read_text())
    W = int(inp["W"])
    fdates = pd.DatetimeIndex(pd.to_datetime(inp["dates"][W:]))
    y_adj = np.load(FR.INP / "bar_y.npy")[W:]
    day_row = np.load(FR.INP / "ref_armf.npz")["day_row"]
    S = CLP.Scorer()
    days = pd.DatetimeIndex(S.days)
    n = len(days)
    core = R.core_models()["model"].tolist()
    Ld = pd.read_parquet(OUT / "losses.parquet")
    gates: list[dict] = []

    def gate(g: str, item: str, value: str, ok: bool) -> None:
        gates.append(dict(gate=g, item=item, value=value, ok=bool(ok)))

    # ---- per estimator: daily losses of the 16 bucket designs + the 96 core designs (from losses.parquet)
    fc = pd.read_parquet(
        OUT / "forecasts.parquet", columns=["est", "model", "row", "pred_adj"]
    )
    per: dict[
        str, dict[str, dict[str, np.ndarray]]
    ] = {}  # est -> loss -> design -> daily
    blong = []
    for sh in R.SHORTS:
        bp = bucket_preds(sh, fdates, day_row)
        if bp is None:
            continue
        lo_d: dict[str, dict[str, np.ndarray]] = {
            lo: {} for lo in ("qlike", "mse_fit", "trade", "mid", "q", "xret")
        }
        for m, pred in bp.items():
            if sh in TREES:
                assert np.isfinite(pred[day_row]).all(), (sh, m)
            d = R.cand_losses(S, pred, y_adj, day_row)
            lo_d["qlike"][m], lo_d["mse_fit"][m], lo_d["trade"][m] = (
                d["ql"],
                d["mse_fit"],
                -d["mid"],
            )
            lo_d["mid"][m], lo_d["q"][m], lo_d["xret"][m] = d["mid"], d["q"], d["x"]
            blong.append(
                pd.DataFrame(
                    dict(
                        est=sh,
                        model=m,
                        day=days,
                        plain=d["plain"],
                        q=d["q"].astype(np.int8),
                        qlike=d["ql"],
                        mse_fit=d["mse_fit"],
                        trade_loss=-d["mid"],
                        mid_pnl=d["mid"],
                        crossed_ret=d["x"],
                    )
                )
            )
        sub = Ld[Ld["est"] == sh]
        for m in core:
            r_ = sub[sub["model"] == m].set_index("day").reindex(days)
            lo_d["qlike"][m] = r_["qlike"].to_numpy(float)
            lo_d["mse_fit"][m] = r_["mse_fit"].to_numpy(float)
            lo_d["trade"][m] = r_["trade_loss"].to_numpy(float)
            lo_d["mid"][m] = r_["mid_pnl"].to_numpy(float)
            lo_d["q"][m] = r_["q"].to_numpy(float)
        # the crossed returns of the anchors (not in losses.parquet): recompute from the forecasts
        for m in FR.ANCHORS:
            pa = (
                fc[(fc["est"] == sh) & (fc["model"] == m)]
                .sort_values("row")["pred_adj"]
                .to_numpy()
            )
            d = R.cand_losses(S, pa, y_adj, day_row)
            assert np.array_equal(d["ql"], lo_d["qlike"][m]), (sh, m)
            lo_d["xret"][m] = d["x"]
        per[sh] = lo_d
        gate(
            "bucket designs present and finite",
            sh,
            f"{len(bp)} designs",
            all(np.isfinite(lo_d["qlike"][m]).all() for m in bp),
        )
        # CPU classes of the bucket designs: linear = EPYC 7513 only (the MCS candidates' class); LightGBM one
        # class; XGBoost / forest: CARC EPYC 7513 / 7542 or Hoffman2 E5-2670 / E5-2650 v2, byte-identical for both
        # (results/trees_rfdev_preview163/xcpu_carc/SUMMARY.md, results/h2check_rfdev/compare_by_class.csv,
        # results/trees_xgbtop_1600/RUN_RECORD.md; coordinator 2026-10-12)
        D = FR.models_dir(R.EST_OF[sh]) if sh in LIN else R.TREE_DIR / sh
        cpus = set()
        for m in bp:
            jf = D / f"{m}.json"
            if jf.is_file():
                cpus |= {
                    c.strip()
                    for c in str(json.loads(jf.read_text()).get("cpu", "")).split("|")
                    if c.strip()
                }
        if sh in LIN:
            ok_c, rule = cpus == {"AMD EPYC 7513 32-Core Processor"}, "EPYC 7513 only"
        elif sh == "lgbm":
            ok_c, rule = cpus == {"AMD EPYC 7513 32-Core Processor"}, "EPYC 7513 only"
        else:
            okset = ("EPYC 7513", "EPYC 7542", "E5-2670", "E5-2650 v2")
            ok_c, rule = (
                len(cpus) >= 1 and all(any(k in c for k in okset) for c in cpus),
                "byte-identical classes",
            )
        gate(
            f"bucket designs' CPU class ({rule})",
            sh,
            "; ".join(sorted(cpus)) or "not recorded",
            ok_c,
        )
    # the linear job re-ran the anchors in its fresh root and gated them against the stored arm f
    for gf in sorted((FR.WORK / "buckets_run").glob("gate*.csv")):
        gd = pd.read_csv(gf)
        gate(
            "linear bucket job: the re-run anchors == the stored arm f (bit for bit)",
            gf.name,
            f"{int(gd['bitwise'].sum())} of {len(gd)} bitwise",
            bool(gd["ok"].all() and gd["bitwise"].all()),
        )
    if not per:
        say("no estimator has its 16 bucket designs yet")
        return
    if blong:
        pd.concat(blong, ignore_index=True).to_parquet(
            OUT / "bucket_losses.parquet", index=False, compression="zstd"
        )
    have = list(per)
    pools = {}
    lin_have = [s for s in LIN if s in have]
    if len(lin_have) == len(LIN):
        pools["pool_linear"] = list(LIN)
    if all(s in have for s in R.SHORTS):
        pools["pool_trees"] = list(TREES)
        pools["pool_all"] = list(R.SHORTS)
        pools["pool_5"] = [
            *LIN,
            "xgb",
            "rf",
        ]  # the coordinator's 5-estimator version, kept beside the six
    elif "lgbm" not in have and all(
        s in have for s in (*LIN, "xgb", "rf")
    ):  # LightGBM pending (2026-10-13)
        pools["pool_xgb_rf"] = ["xgb", "rf"]
        pools["pool_5"] = [*LIN, "xgb", "rf"]
    elif (
        "lgbm" in have
        and not any(s in have for s in ("xgb", "rf"))
        and all(s in have for s in LIN)
    ):
        pools["pool_4"] = [
            *LIN,
            "lgbm",
        ]  # INTERIM (2026-10-13): XGBoost / forest pending
    designs = core + [f"addb_{b}" for b in BUCKETS] + [f"loob_{b}" for b in BUCKETS]

    def fam_losses(members: list[str]) -> dict[str, dict[str, np.ndarray]]:
        """Each design's daily losses averaged over the estimators in `members` (one member = that estimator)."""
        out: dict[str, dict[str, np.ndarray]] = {}
        for lo in ("qlike", "mse_fit", "trade", "mid"):
            out[lo] = {
                m: np.mean([per[e][lo][m] for e in members], axis=0) for m in designs
            }
        # winsorized QLIKE over the family's 112 designs (the MCS rule)
        Lq = np.column_stack([out["qlike"][m] for m in designs])
        Lw = R.winsorize(Lq)[0]
        out["qlike_w"] = {m: Lw[:, j] for j, m in enumerate(designs)}
        out["xret"] = {
            m: np.mean([per[e]["xret"][m] for e in members], axis=0) for m in FR.ANCHORS
        } | {
            m: np.mean([per[e]["xret"][m] for e in members], axis=0)
            for m in designs
            if m.startswith(("addb_", "loob_"))
        }
        return out

    fams = {sh: [sh] for sh in have} | pools
    Wb = {blk: SP.cb_counts(n, R.B_BOOT, blk, R.SEED)[0] for blk in R.BLOCKS}
    ann = S.R._sharpe_ann
    pw_rows, spa_rows, stp_rows = [], [], []
    for fam, members in fams.items():
        F = fam_losses(members)
        base, allf = "baseline", "all_features"
        comps = [(f"addb_{b}", base, "add bucket vs HAR + calendar") for b in BUCKETS]
        comps += [(allf, base, "all features vs HAR + calendar")]
        comps += [
            (f"loob_{b}", allf, "leave bucket out vs all features") for b in BUCKETS
        ]
        for a, b_, kind in comps:
            row: dict[str, object] = dict(
                family=fam,
                members=",".join(members),
                design=a,
                reference=b_,
                comparison=kind,
            )
            for lo in LOSS3:
                va, vb = F[lo][a], F[lo][b_]
                row[f"{lo}"] = float(va.mean())
                row[f"{lo}_ref"] = float(vb.mean())
                row[f"dpct_{lo}"] = float(100 * (va.mean() / vb.mean() - 1))
                row[f"dm_{lo}"] = float(dm_test(va, vb)["dm"])
            row["sharpe_mid"], row["sharpe_mid_ref"] = (
                ann(F["mid"][a]),
                ann(F["mid"][b_]),
            )
            row["sharpe_crossed"], row["sharpe_crossed_ref"] = (
                ann(F["xret"][a]),
                ann(F["xret"][b_]),
            )
            row["hac_t_mid"] = float(asl.newey_west_t(F["mid"][a] - F["mid"][b_])[0])
            row["dm_trade"] = float(dm_test(F["trade"][a], F["trade"][b_])["dm"])
            pw_rows.append(row)
        # SPA / StepM
        tests = {
            "A: beat HAR + calendar": (
                base,
                [f"addb_{b}" for b in BUCKETS] + [allf],
                +1,
            ),
            "B: removing helps": (allf, [f"loob_{b}" for b in BUCKETS], +1),
            "C: removing hurts": (allf, [f"loob_{b}" for b in BUCKETS], -1),
        }
        for lo in SPA_LOSSES:
            for tname, (bench, alts, sgn) in tests.items():
                d = sgn * (
                    F[lo][bench][:, None] - np.column_stack([F[lo][a] for a in alts])
                )
                zero = np.all(d == 0, axis=0)
                d2, alts2 = d[:, ~zero], [a for a, z in zip(alts, zero) if not z]
                for blk in R.BLOCKS:
                    Ds = Wb[blk] @ d2 / n
                    r = spa_rc(d2, Ds)
                    spa_rows.append(
                        dict(
                            family=fam,
                            loss=lo,
                            test=tname,
                            block=blk,
                            benchmark=bench,
                            n_alt=len(alts2),
                            stat_spa=r["stat_spa"],
                            p_consistent=r["p_consistent"],
                            p_lower=r["p_lower"],
                            p_upper=r["p_upper"],
                            p_rc=r["p_rc"],
                            best=alts2[r["best"]],
                        )
                    )
                    for al in SP.FWER:
                        rej, crit = SP.stepm(d2, Ds, al)
                        stp_rows.append(
                            dict(
                                family=fam,
                                loss=lo,
                                test=tname,
                                block=blk,
                                fwer=al,
                                rejected=json.dumps([alts2[j] for j in rej]),
                                n_rejected=len(rej),
                            )
                        )
        say(f"{fam}: done")
    # gate: the bootstrap is a real resampling (the SPA script's gate, repeated here on these counts)
    for blk in R.BLOCKS:
        gate(
            "bootstrap counts are not the identity resample",
            f"block {blk}",
            f"{int((Wb[blk] == 1).all(axis=1).sum())} replicates with every day once",
            int((Wb[blk] == 1).all(axis=1).sum()) == 0,
        )
    pd.DataFrame(pw_rows).to_csv(OUT / "buckets_pairwise.csv", index=False)
    pd.DataFrame(spa_rows).to_csv(OUT / "buckets_spa.csv", index=False)
    pd.DataFrame(stp_rows).to_csv(OUT / "buckets_stepm.csv", index=False)
    G = pd.DataFrame(gates)
    G.to_csv(OUT / "buckets_gates.csv", index=False)
    say(
        f"bucket analysis done in {time.time() - t0:.0f}s for {list(fams)}; gates {int(G['ok'].sum())} / {len(G)} ok"
    )


if __name__ == "__main__":
    if "--interim" in sys.argv:
        INTERIM.update(sys.argv[sys.argv.index("--interim") + 1].split(","))
    main()
