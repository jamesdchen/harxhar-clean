"""Score the intraday-sequence LSTM (the 15:30 forecast of the 15:30-16:00 bar) against the per-bar LSTM,
the per-bar ridge and the best tree rung (checklist I10).

WHAT IS SCORED.  specs/causal_tune_lstm_intraday.py: one LSTM over the last N half-hour bars of the
near-24-hour panel ending with the 15:30 bar (bar-level inputs: the adjusted target, the adjusted
exogenous values of the input set, their availability / activity indicators, calendar and clock), N
tuned in {13, 48, 96}, refitted every session on a 2000-session window, per-window column mask, 5-seed
average; the per-bar arms' target, rows and tuning discipline.  Tables (results/spxw_pnl/, 16:00 only):

  intraday        yhat_lstm_intraday_<bucket>        MSE rule (the arm of record)
  intraday_qsel   yhat_lstm_intraday_qsel_<bucket>   QLIKE rule (recorded)
  perbar_lstm     the per-bar LSTM (a sequence of the last L sessions' 16:00 design rows): the first of
                  PERBAR_CANDIDATES on disk, labelled by provenance (md5 against the de-dup no-mask
                  snapshot results/spxw_pnl/dedup_nomask_2026-09-29/: identical = agent A's de-dup run,
                  no mask, refit every 10; otherwise the table rebuilt after it, agent H's masked run)
  ridge           yhat_sub_ridge_<bucket>, the per-bar ridge
  best_tree       of every yhat_subtree_* table of the bucket on disk (T10, T1, RS10, RS1, Optuna and
                  masked rungs when present), the one with the lowest 16:00 QLIKE on the scored days --
                  chosen EX POST on the same days it is compared on (said wherever it is shown)

ONE SCORER, the research scorer (16:00 bar recalibrated alone), imported: every table is loaded by
experiments/master_table_close.py's load_one (common target, score_linear_subsection_causal.causal_forecasts:
forecast = (f^2 + s) B, s = the forecast's own mean squared adjusted-scale error at 16:00 over the previous
250 sessions (>= 63), lagged one session); QLIKE = score_linear_subsection_causal.qlike; the last-30-min
trade = sign(s) on the straddle, mid and crossed (master_table_close.trade_days, gated against
score_linear_subsection.trade_1530); intervals: day-block bootstrap (score_linear_subsection.day_block_ci)
for QLIKE differences, master_table_close.paired_sharpe for Sharpe differences; DM =
src.evaluation.diebold_mariano.  SAME DAYS for every number of a bucket: the deck's trade days on which
every forecast of the bucket has a recalibrated 16:00 forecast.

ALSO: the configuration chosen at every tuning point (both rules: N, hidden, dropout, learning rate,
epochs) and the share of tuning points per value; the kept step columns per refit and per (tuning point,
N); the seed spread (each of the 5 networks scored alone beside their average); cluster use (sacct).

Outputs (--out, default results/linear_subsection_lstm_intraday/score/): lstmi_levels.csv, lstmi_pairs.csv,
lstmi_hyperparameter_path.csv, lstmi_hyperparameter_share.csv, lstmi_kept.csv, lstmi_seed_spread.csv,
lstmi_gates.csv, cluster_usage.csv, SUMMARY.md (written from those CSVs).  Exit 1 when a gate fails.

Run:  python experiments/score_lstm_intraday_1600.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import master_table_close as mtc  # noqa: E402
import score_linear_subsection as base  # noqa: E402
import score_linear_subsection_causal as slc  # noqa: E402
from src.evaluation.diebold_mariano import dm_test  # noqa: E402

SPXW = ROOT / "results" / "spxw_pnl"
SNAP_NOMASK = SPXW / "dedup_nomask_2026-09-29"
LROOT = ROOT / "results" / "linear_subsection_lstm_intraday"
OUT = LROOT / "score"
BUCKETS = ("live_feasible", "all_features", "baseline")
SEG, TW = "bar1600", 2000
PERBAR_CANDIDATES = ("yhat_lstm_mask_daily_{b}", "yhat_lstm_daily_{b}", "yhat_lstm_{b}")
PAIRS = (
    ("intraday", "perbar_lstm"),
    ("intraday", "ridge"),
    ("intraday", "best_tree"),
    ("intraday_qsel", "intraday"),
    ("perbar_lstm", "ridge"),
)
TRADE_GATE = 1e-12
RECAL_GATE = 1e-12


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def fmt_ci(v: float, lo: float, hi: float, nd: int) -> str:
    return f"{v:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def arm_dir(b: str) -> Path:
    return (
        LROOT
        / b
        / SEG
        / "lstm_intraday"
        / f"tw{TW}"
        / "causal_tune_lstm_intraday"
        / "lstm_intraday"
        / b
    )


def perbar_label(p: Path) -> str:
    snap = SNAP_NOMASK / p.name
    if snap.is_file() and md5(snap) == md5(p):
        return "per-bar LSTM, de-dup design, no mask, refit every 10 sessions (agent A, I1)"
    return f"per-bar LSTM, table rebuilt after the de-dup no-mask snapshot ({p.name}; agent H's masked run)"


def recal(pred_adj: pd.Series) -> pd.DataFrame:
    """load_one's research recalibration for an in-memory 16:00 forecast (index = naive-ET stamps)."""
    tgt = mtc._TGT
    assert tgt is not None
    j = pd.DataFrame({"pred_adj": pred_adj}).join(tgt, how="inner")
    r = pd.DataFrame(
        {
            "true_adj": j["true_adj"],
            "pred_adj": j["pred_adj"],
            "true_raw": j["true_raw"],
            "baseline": j["B"],
        },
        index=j.index,
    )
    r["e2"] = (r["true_adj"] - r["pred_adj"]) ** 2
    r["hhmm"] = r.index.strftime("%H:%M")
    r["day"] = r.index.normalize()
    return slc.causal_forecasts(r)


def usage_table(raw: Path) -> pd.DataFrame | None:
    """cluster_usage rows from the pull script's sacct dump (allocations, then '#STEPS', then batch steps)."""
    if not raw.is_file() or not raw.read_text(encoding="utf-8").strip():
        return None
    text = raw.read_text(encoding="utf-8")
    alloc, _, steps = text.partition("#STEPS")
    cols = [
        "JobID",
        "JobName",
        "AllocCPUS",
        "ElapsedRaw",
        "Start",
        "End",
        "State",
        "CPUTimeRAW",
        "TotalCPU",
        "Partition",
    ]
    rows = [
        ln.split("|")
        for ln in alloc.strip().splitlines()
        if ln.count("|") == len(cols) - 1
    ]
    d = pd.DataFrame(rows, columns=cols)
    d["AllocCPUS"] = d["AllocCPUS"].astype(int)
    d["CPUTimeRAW"] = d["CPUTimeRAW"].astype(float)
    d["Start"] = pd.to_datetime(d["Start"], errors="coerce")
    d["End"] = pd.to_datetime(d["End"], errors="coerce")
    d["stage"] = d["JobName"]
    st = [ln.split("|") for ln in steps.strip().splitlines() if ln.count("|") == 2]
    if st:
        s = pd.DataFrame(st, columns=["JobID", "TotalCPU", "MaxRSS"])
        s["JobID"] = s["JobID"].str.removesuffix(".batch")
        s = s.set_index("JobID")
        from trees_cadence_ladder_1600 import slurm_gib, slurm_seconds

        d["used_cpu_sec"] = d["JobID"].map(s["TotalCPU"].map(slurm_seconds))
        d["max_rss_gib"] = d["JobID"].map(s["MaxRSS"].map(slurm_gib))
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--tables", default=str(SPXW))
    a = ap.parse_args()
    out, tables = Path(a.out), Path(a.tables)
    out.mkdir(parents=True, exist_ok=True)
    deck = pd.read_parquet(base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    tgt = mtc.load_target()
    mtc._init(tgt, deck.index)
    gates: list[dict] = []
    lv_rows, pr_rows, seed_rows = [], [], []
    for b in BUCKETS:
        specs: dict[str, tuple[Path, str]] = {}
        p = tables / f"yhat_lstm_intraday_{b}.parquet"
        if not p.is_file():
            print(f"{p.name} missing: bucket {b} not scored")
            gates.append(
                {
                    "gate": "table on disk",
                    "forecast": p.stem,
                    "n": 0,
                    "value": np.nan,
                    "bound": np.nan,
                    "ok": False,
                }
            )
            continue
        specs["intraday"] = (p, "intraday-sequence LSTM, MSE rule (of record)")
        q = tables / f"yhat_lstm_intraday_qsel_{b}.parquet"
        if q.is_file():
            specs["intraday_qsel"] = (
                q,
                "intraday-sequence LSTM, QLIKE rule (recorded)",
            )
        for pat in PERBAR_CANDIDATES:
            pp = tables / f"{pat.format(b=b)}.parquet"
            if pp.is_file():
                specs["perbar_lstm"] = (pp, perbar_label(pp))
                break
        specs["ridge"] = (tables / f"yhat_sub_ridge_{b}.parquet", "per-bar ridge")
        trees = sorted(
            t
            for t in tables.glob("yhat_subtree_*.parquet")
            if f"_{b}_" in f"_{t.stem.removeprefix('yhat_')}_"
        )
        frames: dict[str, pd.DataFrame] = {}
        what: dict[str, str] = {}
        for key, (path, w) in list(specs.items()) + [
            (f"tree:{t.stem}", (t, "tree rung")) for t in trees
        ]:
            res = mtc.load_one(mtc.Spec(path.stem, path.stem, key, "table", str(path)))
            gates += res["gates"]
            if res["error"]:
                gates.append(
                    {
                        "gate": "load",
                        "forecast": path.stem,
                        "n": 0,
                        "value": np.nan,
                        "bound": np.nan,
                        "ok": key.startswith("tree:"),
                        "detail": res["error"],
                    }
                )
                continue
            frames[key] = res["frame"]
            what[key] = (
                w if not key.startswith("tree:") else path.stem.removeprefix("yhat_")
            )
        main_keys = [
            k
            for k in ("intraday", "intraday_qsel", "perbar_lstm", "ridge")
            if k in frames
        ]
        days = deck.index
        for k in main_keys:
            days = days.intersection(
                frames[k].index[frames[k]["pred_clock"].notna()].normalize()
            )
        days = days.sort_values()
        tree_keys = []
        for k in [k for k in frames if k.startswith("tree:")]:
            covers = days.isin(
                frames[k].index[frames[k]["pred_clock"].notna()].normalize()
            ).all()
            gates.append(
                {
                    "gate": "tree table covers the scored days (else not a best-tree candidate)",
                    "forecast": f"{b}:{k}",
                    "n": len(days),
                    "value": float(covers),
                    "bound": 1.0,
                    "ok": True,
                }
            )
            if covers:
                tree_keys.append(k)
        gates.append(
            {
                "gate": "scored days = the deck days",
                "forecast": b,
                "n": len(days),
                "value": len(deck.index) - len(days),
                "bound": 0,
                "ok": len(days) == len(deck.index),
            }
        )
        dk = deck.loc[days]
        per: dict[str, dict] = {}
        for k in main_keys + tree_keys:
            fd = frames[k].set_index(frames[k].index.normalize()).loc[days]
            qd = slc.qlike(fd["true_raw"], fd["pred_clock"])
            t = mtc.trade_days(fd["pred_clock"], dk)
            chk = base.trade_1530(
                fd["pred_clock"].set_axis(fd.index + pd.Timedelta(hours=16))
            )
            g = max(
                abs(mtc.sharpe(t["mid"].to_numpy()) - chk["Sharpe_mid"]),
                abs(mtc.sharpe(t["crossed"].to_numpy()) - chk["Sharpe_crossed"]),
            )
            gates.append(
                {
                    "gate": "trade_days Sharpe = trade_1530 Sharpe",
                    "forecast": f"{b}:{k}",
                    "n": len(t),
                    "value": g,
                    "bound": TRADE_GATE,
                    "ok": g <= TRADE_GATE,
                }
            )
            per[k] = {"q": pd.Series(qd.to_numpy(), index=fd.index), "t": t}
        best = (
            min(tree_keys, key=lambda k: float(per[k]["q"].mean()))
            if tree_keys
            else None
        )
        if best is not None:
            per["best_tree"] = per[best]
            what["best_tree"] = (
                f"{what[best]} (lowest QLIKE of {len(tree_keys)} tree tables on these days, chosen ex post)"
            )
        for k in main_keys + (["best_tree"] if best is not None else []):
            t = per[k]["t"]
            lv_rows.append(
                {
                    "bucket": b,
                    "forecast": k,
                    "what": what[k],
                    "n_days": len(days),
                    "qlike": float(per[k]["q"].mean()),
                    "sharpe_mid": mtc.sharpe(t["mid"].to_numpy()),
                    "sharpe_crossed": mtc.sharpe(t["crossed"].to_numpy()),
                    "pct_buy": float(100 * (t["q"] > 0).mean()),
                }
            )
        for k in tree_keys:
            lv_rows.append(
                {
                    "bucket": b,
                    "forecast": k,
                    "what": "tree rung (candidate for best_tree)",
                    "n_days": len(days),
                    "qlike": float(per[k]["q"].mean()),
                    "sharpe_mid": mtc.sharpe(per[k]["t"]["mid"].to_numpy()),
                    "sharpe_crossed": mtc.sharpe(per[k]["t"]["crossed"].to_numpy()),
                    "pct_buy": float(100 * (per[k]["t"]["q"] > 0).mean()),
                }
            )
        for ka, kb in PAIRS:
            if ka not in per or kb not in per:
                continue
            qa, qb = per[ka]["q"], per[kb]["q"]
            d = qa - qb
            lo, hi = base.day_block_ci(d)
            dm = dm_test(qa.to_numpy(), qb.to_numpy())
            ta, tb = per[ka]["t"], per[kb]["t"]
            row = {
                "bucket": b,
                "a": ka,
                "b": kb,
                "b_what": what[kb],
                "n_days": len(d),
                "qlike_a": float(qa.mean()),
                "qlike_b": float(qb.mean()),
                "qlike_diff": float(d.mean()),
                "qlike_pct": float(100 * d.mean() / qb.mean()),
                "qlike_ci_lo": lo,
                "qlike_ci_hi": hi,
                "dm": float(dm["dm"]),
                "dm_p": float(dm["p"]),
                "same_side_share": float(
                    (ta["q"].to_numpy() == tb["q"].to_numpy()).mean()
                ),
            }
            for fill in ("mid", "crossed"):
                v, l2, h2 = mtc.paired_sharpe(ta[fill].to_numpy(), tb[fill].to_numpy())
                row |= {
                    f"sharpe_{fill}_a": mtc.sharpe(ta[fill].to_numpy()),
                    f"sharpe_{fill}_b": mtc.sharpe(tb[fill].to_numpy()),
                    f"sharpe_{fill}_diff": v,
                    f"sharpe_{fill}_ci_lo": l2,
                    f"sharpe_{fill}_ci_hi": h2,
                }
            pr_rows.append(row)
        # ---- seed spread (the merged arm's per-seed forecasts, the same recalibration and days)
        npz = arm_dir(b) / f"lstm_intraday_{SEG}.npz"
        if npz.is_file():
            with np.load(npz, allow_pickle=False) as z:
                stamps = pd.DatetimeIndex(
                    pd.to_datetime(z["date"].astype(str))
                ).as_unit("ns")
                seeds, ens = z["pred_adj_seeds"], z["pred_adj"]
            gap = float(np.abs(seeds.mean(axis=1) - ens).max())
            gates.append(
                {
                    "gate": "seed average = pred_adj",
                    "forecast": b,
                    "n": len(ens),
                    "value": gap,
                    "bound": 1e-12,
                    "ok": gap <= 1e-12,
                }
            )
            cf = recal(pd.Series(ens, index=stamps))
            on = cf.index[cf.index.normalize().isin(days)]
            ref = frames["intraday"].loc[on, "pred_clock"]
            rel = float((cf.loc[on, "pred_clock"] / ref - 1.0).abs().max())
            gates.append(
                {
                    "gate": "in-memory recalibration = load_one (seed ensemble)",
                    "forecast": b,
                    "n": len(on),
                    "value": rel,
                    "bound": RECAL_GATE,
                    "ok": rel <= RECAL_GATE,
                }
            )
            for j in range(seeds.shape[1]):
                c = recal(pd.Series(seeds[:, j], index=stamps))
                fd = c.set_index(c.index.normalize()).loc[days]
                t = mtc.trade_days(fd["pred_clock"], dk)
                seed_rows.append(
                    {
                        "bucket": b,
                        "seed": j,
                        "qlike": float(
                            slc.qlike(fd["true_raw"], fd["pred_clock"]).mean()
                        ),
                        "sharpe_mid": mtc.sharpe(t["mid"].to_numpy()),
                        "sharpe_crossed": mtc.sharpe(t["crossed"].to_numpy()),
                        "pct_buy": float(100 * (t["q"] > 0).mean()),
                    }
                )
            seed_rows.append(
                {
                    "bucket": b,
                    "seed": "average",
                    "qlike": float(per["intraday"]["q"].mean()),
                    "sharpe_mid": mtc.sharpe(per["intraday"]["t"]["mid"].to_numpy()),
                    "sharpe_crossed": mtc.sharpe(
                        per["intraday"]["t"]["crossed"].to_numpy()
                    ),
                    "pct_buy": float(100 * (per["intraday"]["t"]["q"] > 0).mean()),
                    "mean_row_seed_sd_adj": float(seeds.std(axis=1).mean()),
                    "mean_abs_pred_adj": float(np.abs(ens).mean()),
                }
            )
    lv, pr, sd = pd.DataFrame(lv_rows), pd.DataFrame(pr_rows), pd.DataFrame(seed_rows)
    lv.to_csv(out / "lstmi_levels.csv", index=False)
    pr.to_csv(out / "lstmi_pairs.csv", index=False)
    sd.to_csv(out / "lstmi_seed_spread.csv", index=False)
    # ---- hyperparameters over time, kept columns
    paths, kept_rows = [], []
    for b in BUCKETS:
        d = arm_dir(b)
        if not (d / f"tune_trace_{SEG}.csv").is_file():
            continue
        tr = pd.read_csv(d / f"tune_trace_{SEG}.csv")
        paths.append(tr.assign(bucket=b))
        rf = pd.read_csv(d / f"refit_trace_{SEG}.csv")
        with np.load(d / f"lstm_intraday_{SEG}.npz", allow_pickle=False) as z:
            p_step = int(len(z["step_names"]))
        for rule, col in (("mse", "kept_n"), ("qlike", "kept_n_q")):
            kept_rows.append(
                {
                    "bucket": b,
                    "what": f"refit ({rule} rule)",
                    "p": p_step,
                    "n": len(rf),
                    "min": int(rf[col].min()),
                    "median": float(rf[col].median()),
                    "max": int(rf[col].max()),
                }
            )
        for c in [c for c in tr.columns if c.startswith("kept_n_N")]:
            v = tr.drop_duplicates("tune_row")[c]
            kept_rows.append(
                {
                    "bucket": b,
                    "what": f"tuning point, N = {c.removeprefix('kept_n_N')}",
                    "p": p_step,
                    "n": len(v),
                    "min": int(v.min()),
                    "median": float(v.median()),
                    "max": int(v.max()),
                }
            )
    hp = pd.concat(paths, ignore_index=True) if paths else pd.DataFrame()
    cols = [
        "bucket",
        "tune_row",
        "forecast_date",
        "rule",
        "seq_len",
        "hidden",
        "dropout",
        "lr",
        "epochs",
        "epochs_mean",
        "epochs_cap_hits",
        "val_mse",
        "val_qlike",
    ]
    if len(hp):
        hp[[c for c in cols if c in hp.columns]].to_csv(
            out / "lstmi_hyperparameter_path.csv", index=False
        )
        share = []
        for (b, rule), g in hp.groupby(["bucket", "rule"]):
            for ax in ("seq_len", "hidden", "dropout", "lr"):
                for v, n in g[ax].value_counts().sort_index().items():
                    share.append(
                        {
                            "bucket": b,
                            "rule": rule,
                            "axis": ax,
                            "value": v,
                            "tuning_points": int(n),
                            "share": float(n / len(g)),
                        }
                    )
        pd.DataFrame(share).to_csv(out / "lstmi_hyperparameter_share.csv", index=False)
    kp = pd.DataFrame(kept_rows)
    kp.to_csv(out / "lstmi_kept.csv", index=False)
    use = usage_table(LROOT / "logs" / "sacct_raw.txt")
    if use is not None:
        use.to_csv(out / "cluster_usage.csv", index=False)
    gt = pd.DataFrame(gates)
    gt.to_csv(out / "lstmi_gates.csv", index=False)
    write_summary(out, lv, pr, sd, hp, kp, gt, use)
    print(
        lv[~lv["forecast"].str.startswith("tree:")][
            ["bucket", "forecast", "qlike", "sharpe_mid", "sharpe_crossed", "pct_buy"]
        ]
        .round(4)
        .to_string(index=False)
    )
    n_bad = int((~gt["ok"].astype(bool)).sum())
    print(f"{len(gt)} gates, {n_bad} failed; outputs in {out}")
    return 1 if n_bad else 0


def write_summary(out, lv, pr, sd, hp, kp, gt, use) -> None:
    L = [
        "# The intraday-sequence LSTM at 16:00 (checklist I10)",
        "",
        "Written by `experiments/score_lstm_intraday_1600.py` from its own CSVs in this folder (`lstmi_levels.csv`, "
        "`lstmi_pairs.csv`, `lstmi_hyperparameter_path.csv`, `lstmi_hyperparameter_share.csv`, `lstmi_kept.csv`, "
        "`lstmi_seed_spread.csv`, `lstmi_gates.csv`"
        + (", `cluster_usage.csv`" if use is not None else "")
        + "). "
        "Every number below is read from them.",
        "",
        "## What is scored",
        "",
        "The forecast of the 15:30-16:00 bar's realized variance (row stamped 16:00, issued at 15:30) by an LSTM over the "
        "last N half-hour bars of the near-24-hour panel ending with the 15:30 bar (`specs/causal_tune_lstm_intraday.py`): "
        "one step per bar, bar-level inputs (the bar's adjusted target `adj_RV`, the adjusted exogenous values `adj_*` of "
        "the input set, their availability / activity indicators, the calendar columns of the per-bar designs and a "
        "half-hour clock) instead of the HAR ladders `har_ma_*`; N tuned in {13, 48, 96} bars with the per-bar LSTM's grid "
        "(hidden 16 / 64, dropout 0 / 0.2, learning rate 1e-3 / 1e-2), re-chosen every 250 sessions on a 125-session "
        "validation tail after a 25-session embargo (MSE rule of record, QLIKE rule recorded), refitted every session "
        "on a 2000-session window, per-window column mask, the average of 5 seeds. Input sets: `baseline` (target "
        "series + calendar / clock), `live_feasible`, `all_features`.",
        "",
        "Comparators: the per-bar LSTM (a sequence of the last L sessions' 16:00 rows of the per-bar design), the per-bar "
        "ridge, and the best tree rung on disk -- the tree table with the lowest 16:00 QLIKE on the same days, chosen "
        "ex post (so its comparison favours the tree).",
        "",
        "## Scorer and days",
        "",
        "The research scorer (the 16:00 bar recalibrated on its own; the master table's loader, target and trade "
        "imported): forecast = (f^2 + s) B with s = the forecast's own mean squared adjusted-scale error at 16:00 over "
        "the previous 250 sessions (at least 63), lagged one session; QLIKE against the per-bar spec's 16:00 target "
        "(lower is better). The last-30-min trade is the sign(s) rule on the straddle (nearest out-of-the-money call + "
        "nearest out-of-the-money put, same-day expiry, one position): buy when the recalibrated forecast exceeds the "
        "15:30 implied variance, sell otherwise; annualized Sharpe at mid and at crossed fills. Intervals: 95 % day-block "
        "bootstrap (21-day circular blocks, 2000 draws), paired. Differences are a - b.",
        "",
        "## Levels",
        "",
        "| input set | forecast | QLIKE | Sharpe mid | Sharpe crossed | % buy | days | what |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for x in lv[~lv["forecast"].str.startswith("tree:")].itertuples():
        L.append(
            f"| `{x.bucket}` | {x.forecast} | {x.qlike:.4f} | {x.sharpe_mid:.2f} | {x.sharpe_crossed:.2f} | {x.pct_buy:.1f} | {x.n_days} | {x.what} |"
        )
    L += [
        "",
        "## Paired differences (a - b) with 95 % intervals",
        "",
        "| input set | a - b | QLIKE diff (%) | QLIKE diff [interval] | DM t | Sharpe mid diff [interval] | Sharpe crossed diff [interval] | same side |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for x in pr.itertuples():
        L.append(
            f"| `{x.bucket}` | {x.a} - {x.b} | {x.qlike_pct:+.1f} % | {fmt_ci(x.qlike_diff, x.qlike_ci_lo, x.qlike_ci_hi, 4)} | {x.dm:+.2f} | "
            f"{fmt_ci(x.sharpe_mid_diff, x.sharpe_mid_ci_lo, x.sharpe_mid_ci_hi, 2)} | "
            f"{fmt_ci(x.sharpe_crossed_diff, x.sharpe_crossed_ci_lo, x.sharpe_crossed_ci_hi, 2)} | {100 * x.same_side_share:.0f} % |"
        )
    if len(hp):
        L += [
            "",
            "## Chosen configuration over time (MSE rule of record / QLIKE rule)",
            "",
            "| input set | tuning row | first forecast | rule | N | hidden | dropout | lr | epochs (seeds) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for x in hp.itertuples():
            L.append(
                f"| `{x.bucket}` | {x.tune_row} | {str(x.forecast_date)[:10]} | {x.rule} | {x.seq_len} | {x.hidden} | {x.dropout} | {x.lr:g} | {x.epochs} |"
            )
    if len(kp):
        L += [
            "",
            "## Kept step columns (per-window mask)",
            "",
            "| input set | where | step columns | fits | min kept | median | max |",
            "|---|---|---|---|---|---|---|",
        ]
        for x in kp.itertuples():
            L.append(
                f"| `{x.bucket}` | {x.what} | {x.p} | {x.n} | {x.min} | {x.median:g} | {x.max} |"
            )
    if len(sd):
        L += [
            "",
            "## Seed spread (each network alone vs the 5-seed average, same recalibration and days)",
            "",
            "| input set | seed | QLIKE | Sharpe mid | Sharpe crossed | % buy |",
            "|---|---|---|---|---|---|",
        ]
        for x in sd.itertuples():
            L.append(
                f"| `{x.bucket}` | {x.seed} | {x.qlike:.4f} | {x.sharpe_mid:.2f} | {x.sharpe_crossed:.2f} | {x.pct_buy:.1f} |"
            )
        avg = sd[sd["seed"].astype(str) == "average"]
        for x in avg.itertuples():
            L.append(
                f"\n`{x.bucket}`: mean over rows of the sd of the 5 seeds' adjusted-scale forecasts {x.mean_row_seed_sd_adj:.4f} (mean |forecast| {x.mean_abs_pred_adj:.4f})."
            )
    if use is not None and len(use):
        ev = (
            pd.concat(
                [
                    pd.DataFrame({"t": use["Start"], "d": use["AllocCPUS"]}),
                    pd.DataFrame({"t": use["End"], "d": -use["AllocCPUS"]}),
                ]
            )
            .dropna()
            .sort_values(["t", "d"])
        )
        L += [
            "",
            "## Cluster use (sacct, this campaign's jobs)",
            "",
            "| job | allocations | CPUs each | allocated CPU-h | used CPU-h | first start | last end |",
            "|---|---|---|---|---|---|---|",
        ]
        for st, g in use.groupby("stage", sort=False):
            used = (
                g["used_cpu_sec"].sum() / 3600 if "used_cpu_sec" in g else float("nan")
            )
            L.append(
                f"| {st} | {len(g)} | {int(g['AllocCPUS'].min())}-{int(g['AllocCPUS'].max())} | {g['CPUTimeRAW'].sum() / 3600:.1f} | {used:.1f} | {g['Start'].min()} | {g['End'].max()} |"
            )
        L += [
            "",
            f"All: {use['CPUTimeRAW'].sum() / 3600:.1f} allocated CPU-hours; peak concurrent allocated CPUs {int(ev['d'].cumsum().max())}; "
            f"wall-clock {use['Start'].min()} .. {use['End'].max()}.",
        ]
    bad = gt[~gt["ok"].astype(bool)]
    L += [
        "",
        "## Gates",
        "",
        f"{len(gt)} scorer gates, {len(bad)} failed (`lstmi_gates.csv`). The spec's own gates (target identity, "
        "causality x2, determinism, chunk = unchunked, pool = serial, mask) are in `../gates/`.",
        "",
        "Wording: sign(s) = the rule above; QLIKE differences are losses (negative = a lower loss); nothing here is a recommendation.",
    ]
    (out / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
