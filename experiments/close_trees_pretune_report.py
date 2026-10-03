"""Report stage of experiments/close_trees_pretune.py: tables, one figure and SUMMARY.md, all from
the run's own records (_work/<tag>/<model>/) and the CSVs this module writes.

Written under results/close_studies_2026-10-03/trees_pretune/ (prefix <tag>_<model>_):
  pretune_trials.csv       one row per (study, trial): configuration, fold validation MSE / QLIKE,
                           median rounds, fit seconds
  pretune_curve.csv        best-so-far validation MSE against the trial count (each study and the
                           merged studies), the chosen configuration's distance to the final one,
                           and the held-out-fold check: select on folds {0, 2} and score the
                           selected trial on folds {1, 3}, and the reverse (no extra fits)
  pretune_checkpoints.csv  the configuration chosen after k trials, k in a fixed ladder
  retune_log.csv           every light retune of both retune paths (copied from _work)
  retune_trials.csv        every retune trial (copied from _work)
  arms.csv                 the 866 trade days: QLIKE, sign(s) Sharpe mid / crossed, buy share,
                           and against the LOCAL shipped-config control: Diebold-Mariano on daily
                           QLIKE, HAC t on the paired daily P&L (mid) difference, the Sharpe
                           difference, max relative forecast difference; the stored master-table
                           rows scored the same way
  pretune_curve.png        best-so-far validation MSE and the held-out-fold score against trials
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import close_trees_pretune as C

LADDER = (1, 2, 5, 10, 25, 50, 100, 150, 200, 300, 500, 1000, 2000, 4000, 8000)
FOLD_SPLITS = (((0, 2), (1, 3)), ((1, 3), (0, 2)))  # (select on, score on); FOLDS = 4 only
AXIS_LABEL = {
    "num_leaves": "leaves",
    "min_child_samples": "min leaf rows",
    "feature_fraction": "col. share",
    "bagging_fraction": "row share",
    "lambda_l2": "L2",
    "learning_rate": "lr",
    "max_depth": "depth",
    "min_child_weight": "min leaf weight",
    "subsample": "row share",
    "colsample_bytree": "col. share",
    "reg_lambda": "L2",
}
COLOR = {"s0": "#2a78d6", "s1": "#4a3aa7", "merged": "#eb6834", "heldout": "#1baf7a"}


def prefix() -> str:
    return f"{C.TAG}_{C.MODEL}_"


def axis_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Largest axis difference between two configurations, as a share of the axis range (log
    units on log axes)."""
    out = 0.0
    for j, (_, s) in enumerate(C.spaces().items()):
        _, lo, hi, log = s
        if log:
            d = abs(math.log(a[j]) - math.log(b[j])) / (math.log(hi) - math.log(lo))
        else:
            d = abs(a[j] - b[j]) / (hi - lo)
        out = max(out, d)
    return out


def studies() -> list[dict]:
    seeds = sorted(int(p.stem.split("_s")[1]) for p in C.RUN.glob("pretune_s*.npz"))
    return C.load_pretune(tuple(seeds))["by_seed"]


def trials_table(recs: list[dict]) -> pd.DataFrame:
    rows = []
    axes = list(C.spaces())
    for r in recs:
        for i in range(len(r["val_mse"])):
            d = dict(study=f"s{r['seed']}", trial=i)
            d |= {a: r["params"][i, j] for j, a in enumerate(axes)}
            d["val_mse"] = r["val_mse"][i]
            for f in range(r["fold_val_mse"].shape[1]):
                d[f"fold{f}_mse"] = r["fold_val_mse"][i, f]
            d["val_qlike"] = float(np.mean(r["fold_val_qlike"][i]))
            d["rounds_median"] = C.median_rounds(r["fold_rounds"][i])
            d["cap_hits"] = int(np.sum(r["fold_rounds"][i] >= r["fold_rounds_max"][i]))
            d["fit_sec"] = float(np.sum(r["fold_sec"][i]))
            rows.append(d)
    return pd.DataFrame(rows)


def curve_for(name: str, P: np.ndarray, F: np.ndarray) -> pd.DataFrame:
    """Best-so-far along a trial sequence (params P, fold MSE F)."""
    v = F.mean(axis=1)
    n = len(v)
    final = int(np.argmin(v))
    rows = []
    for k in range(1, n + 1):
        b = int(np.argmin(v[:k]))
        d = dict(
            sequence=name,
            k=k,
            best_trial=b,
            best_val_mse=v[b],
            rel_gap_to_final=v[b] / v[final] - 1,
            gain_over_shipped=1 - v[b] / v[0],
            dist_to_final=axis_dist(P[b], P[final]),
        )
        if F.shape[1] == 4:
            ho = []
            for sel, sc in FOLD_SPLITS:
                bs = int(np.argmin(F[:k, list(sel)].mean(axis=1)))
                ho.append(F[bs, list(sc)].mean() / F[0, list(sc)].mean())
            d["heldout_rel_to_shipped"] = float(np.mean(ho))
            d["insample_rel_to_shipped"] = v[b] / v[0]
        rows.append(d)
    return pd.DataFrame(rows)


def interleave(recs: list[dict], key: str) -> np.ndarray:
    return C.merge_seeds(recs)[key]


def write_curves(recs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts = [curve_for(f"s{r['seed']}", r["params"], r["fold_val_mse"]) for r in recs]
    if len(recs) > 1:
        parts.append(curve_for("merged", interleave(recs, "params"), interleave(recs, "fold_val_mse")))
    cur = pd.concat(parts, ignore_index=True)
    cur.to_csv(C.OUT / f"{prefix()}pretune_curve.csv", index=False)
    axes = list(C.spaces())
    seqs = {f"s{r['seed']}": (r["params"], r["fold_val_mse"], r["fold_rounds"]) for r in recs}
    if len(recs) > 1:
        seqs["merged"] = tuple(interleave(recs, k) for k in ("params", "fold_val_mse", "fold_rounds"))  # type: ignore[assignment]
    rows = []
    for name, (P, F, R) in seqs.items():
        v = F.mean(axis=1)
        for k in [k for k in LADDER if k < len(v)] + [len(v)]:
            b = int(np.argmin(v[:k]))
            rows.append(
                dict(sequence=name, k=k, best_trial=b, best_val_mse=v[b], rounds=C.median_rounds(R[b]))
                | {a: P[b, j] for j, a in enumerate(axes)}
            )
    chk = pd.DataFrame(rows)
    chk.to_csv(C.OUT / f"{prefix()}pretune_checkpoints.csv", index=False)
    return cur, chk


def plot_curve(cur: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    for name, g in cur.groupby("sequence"):
        ax.plot(g["k"], g["insample_rel_to_shipped"] if "insample_rel_to_shipped" in g else g["best_val_mse"],
                color=COLOR.get(str(name), "#7a7a7a"), lw=2, label=f"{name}: best so far, all 4 folds")
        if name == ("merged" if "merged" in set(cur["sequence"]) else cur["sequence"].iloc[0]) and "heldout_rel_to_shipped" in g:
            ax.plot(g["k"], g["heldout_rel_to_shipped"], color=COLOR["heldout"], lw=2, ls=(0, (5, 2)),
                    label=f"{name}: selected on 2 folds, scored on the other 2")
    ax.axhline(1.0, color="#9a9a9a", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("trials")
    ax.set_ylabel("validation MSE / shipped configuration's")
    ax.grid(True, color="#e6e6e6", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Pre-2020 pre-tune: validation MSE of the selected trial", fontsize=10)
    fig.tight_layout()
    fig.savefig(C.OUT / f"{prefix()}pretune_curve.png")
    plt.close(fig)


def arms_table(info: dict) -> pd.DataFrame:
    from src.evaluation.diebold_mariano import dm_test

    preds: dict[str, np.ndarray] = {}
    sched: dict[str, str] = {}
    for f in sorted(C.RUN.glob("walk_*_r*.npz")):
        z = np.load(f, allow_pickle=False)
        name = f.stem[len("walk_"):]
        if len(z["pred"]) != len(info["y"]) - info["W"]:
            continue  # a restricted (smoke) walk
        preds[name] = z["pred"]
        sched[name] = str(z["schedule"])
    assert "control_r10" in preds, "no full-length control walk (refit every 10)"
    for k in C.STORED:
        preds[k] = C.load_stored(info, k)
    tab, P = C.score_table(info, preds)
    names = P["names"]
    rows = []
    ctrl_of = []
    for j, k in enumerate(names):
        # each local arm against the control of its own refit cadence; stored rows against r10
        cad = k.rsplit("_r", 1)[1] if k in sched else "10"
        ctrl_key = f"control_r{cad}" if f"control_r{cad}" in preds else "control_r10"
        jc = names.index(ctrl_key)
        ctrl_of.append(jc)
        q = dm_test(P["ql"][:, j], P["ql"][:, jc])
        h = dm_test(P["pnl"][:, j], P["pnl"][:, jc])
        rel = np.max(np.abs(preds[k] / preds[ctrl_key] - 1))
        rows.append(
            dict(
                dm_qlike_vs_control=q["dm"],
                dm_p=q["p"],
                hac_t_pnl_vs_control=h["dm"],
                hac_p=h["p"],
                same_position_share=float(np.mean(np.sign(P["pnl"][:, j]) * np.sign(P["pnl"][:, jc]) > 0)),
                max_rel_forecast_diff_vs_control=rel,
                schedule=sched.get(k, ""),
            )
        )
    out = pd.concat([tab, pd.DataFrame(rows)], axis=1)
    out["control"] = [names[j] for j in ctrl_of]
    out["dsharpe_mid_vs_control"] = out["sharpe_mid"].to_numpy() - out["sharpe_mid"].to_numpy()[ctrl_of]
    out["qlike_pct_vs_control"] = (out["qlike"].to_numpy() / out["qlike"].to_numpy()[ctrl_of] - 1) * 100
    out.to_csv(C.OUT / f"{prefix()}arms.csv", index=False)
    return out


def fmt_cfg(row: pd.Series | dict) -> str:
    parts = []
    for a in C.spaces():
        v = float(row[a])
        s = f"{v:.0f}" if C.spaces()[a][0] == "int" else (f"{v:.3g}")
        parts.append(f"{AXIS_LABEL.get(a, a)} {s}")
    return ", ".join(parts)


def write_summary(recs, cur, chk, arms, info) -> None:
    L = []
    w = L.append
    g = pd.read_csv(C.OUT / "gate.csv")
    meta = recs[0]["meta"]
    tt = trials_table(recs)
    tt.to_csv(C.OUT / f"{prefix()}pretune_trials.csv", index=False)
    rt = C.RUN / "retune_log_r10.csv"
    rlog = pd.read_csv(rt) if rt.is_file() else pd.DataFrame()
    rtr = C.RUN / "retune_trials_r10.csv"
    if rt.is_file():
        shutil.copy(rt, C.OUT / f"{prefix()}retune_log.csv")
    if rtr.is_file():
        shutil.copy(rtr, C.OUT / f"{prefix()}retune_trials.csv")
    rtt = pd.read_csv(rtr) if rtr.is_file() else pd.DataFrame()
    w("# Per-bar trees at 16:00: heavy pre-2020 pre-tune, light retune every 250 sessions")
    w("")
    w(
        "Generated by `experiments/close_trees_pretune.py report` (`experiments/close_trees_pretune_report.py`) from "
        "its own CSVs in this folder. Model: the tree spec's per-bar "
        f"{'LightGBM' if C.MODEL == 'lgbm' else 'XGBoost'} on the all_features design of the 16:00 bar "
        "(`specs/causal_tune_trees.py`), 2000-session window, per-window column mask on, every fit single-threaded; "
        "search space, round rules and model construction from `specs/causal_tune_trees_optuna_jobs.py`. "
        f"Run tag `{C.TAG}`; libraries {json.dumps(meta['versions'])}."
    )
    w("")
    w("## Gate")
    w("")
    w("| gate | item | value | reference | ok |")
    w("|---|---|---|---|---|")
    for _, r in g.iterrows():
        ref = "" if pd.isna(r["reference"]) else f"{r['reference']:.6g}"
        w(f"| {r['gate']} | {r['item']} | {r['value']:.6g} | {ref} | {r['ok']} |")
    w("")
    w("## Pre-tune (sessions before 2020-01-03 only)")
    w("")
    folds = meta["folds"]
    w(
        f"{len(recs)} independent Optuna TPE studies (sampler seeds {', '.join(str(C.TPE_SEED_BASE + r['seed']) for r in recs)}; "
        "trial 0 of each = the shipped configuration), merged afterwards by interleaving their trials (trial j of every "
        "study before trial j + 1 of any). Objective: mean validation MSE on the transformed target over "
        f"{len(folds)} blocked walk-forward folds of {meta['fold_len']} sessions, each fitted on the 2000 sessions before a "
        f"{meta['embargo']}-session embargo: "
        + "; ".join(f"fold {f['fold']} validates {f['val_first']} .. {f['val_last']} (fit from {f['fit_first']})" for f in folds)
        + f". Rows: the 16:00 sessions of the last-hour design cache ({meta['pre_sessions']} sessions {meta['pre_first'][:10]} .. "
        f"{meta['pre_last'][:10]}); on their {int(g.loc[g['item'] == 'common_sessions', 'value'].iloc[0])} sessions in common "
        "with the 16:00-bar cache the target is identical and "
        f"{int(g.loc[g['item'] == 'columns_identical', 'value'].iloc[0])} of 628 columns are identical (the others differ by "
        "the rolling scaling window). Early stopping on each fold's validation rows (the spec's patience and cap); a "
        "configuration's refit rounds = the median of its folds' early-stopped rounds."
    )
    w("")
    for r in recs:
        n = len(r["val_mse"])
        b = int(np.argmin(r["val_mse"]))
        w(
            f"- study s{r['seed']}: {n} trials, {r['fold_sec'].sum() / 3600:.2f} core-hours of fold fits "
            f"({r['fold_sec'].mean():.1f} s a fold fit); best trial {b}: validation MSE {r['val_mse'][b]:.5f} vs the shipped "
            f"configuration's {r['val_mse'][0]:.5f} ({(1 - r['val_mse'][b] / r['val_mse'][0]) * 100:.1f} % lower); "
            f"{fmt_cfg(dict(zip(C.spaces(), r['params'][b]))) }, rounds {C.median_rounds(r['fold_rounds'][b])}."
        )
    if len(recs) > 1:
        bs = [r["params"][int(np.argmin(r["val_mse"]))] for r in recs]
        w(
            f"- the studies' best configurations differ by up to {axis_dist(bs[0], bs[1]) * 100:.0f} % of an axis range "
            "(largest axis difference, log units on log axes)."
        )
    w("")
    w("Best-so-far against the trial count (`pretune_curve.csv`, `pretune_checkpoints.csv`, figure `pretune_curve.png`). "
      "'held-out' = select on folds {0, 2}, score the selected trial on folds {1, 3}, and the reverse, averaged; both as a "
      "ratio to the shipped configuration's MSE on the same folds:")
    w("")
    w("| sequence | trials | best trial | best val MSE | gap to final best | val MSE / shipped (4 folds) | held-out / shipped | distance to final config | rounds | configuration |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in chk.iterrows():
        c = cur[(cur["sequence"] == r["sequence"]) & (cur["k"] == r["k"])].iloc[0]
        ho = f"{c['heldout_rel_to_shipped']:.4f}" if "heldout_rel_to_shipped" in c else ""
        w(
            f"| {r['sequence']} | {int(r['k'])} | {int(r['best_trial'])} | {r['best_val_mse']:.5f} | "
            f"{c['rel_gap_to_final'] * 100:+.2f} % | {1 - c['gain_over_shipped']:.4f} | {ho} | "
            f"{c['dist_to_final'] * 100:.0f} % | {int(r['rounds'])} | {fmt_cfg(r)} |"
        )
    w("")
    if not rlog.empty:
        w("## Light retunes (every 250 sessions from 2020)")
        w("")
        w(
            f"At each retune row: {C.RETUNE_TRIALS} trials (trial 0 = the incumbent) of a TPE study (seed "
            f"{C.TPE_SEED_BASE} + study seed + row) confined to a box around the incumbent ({C.NEIGHBOURHOOD_FRACTION:g} of "
            "each axis's range, log units on log axes), validated on the 250 sessions before the row as 2 folds of 125 "
            "sessions (each its own 2000-session fit block, 25-session embargo), on the 16:00-bar design. retune_any "
            "switches to the best challenger on any lower validation MSE; retune_margin only when the HAC t of the paired "
            f"daily squared-error difference is below -z({C.MARGIN_ALPHA:g} / number of challengers)."
        )
        w("")
        w("| path | row | date | incumbent val MSE | best challenger | relative gain | HAC t | critical | switched | rounds in force |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in rlog.iterrows():
            w(
                f"| {r['path']} | {r['row']} | {r['date']} | {r['incumbent_val_mse']:.5f} | {r['best_challenger_val_mse']:.5f} "
                f"(trial {r['best_trial']}) | {r['rel_gain'] * 100:.2f} % | {r['hac_t']:+.2f} | -{r['z_crit']:.2f} | "
                f"{r['switched']} | {r['rounds_in_force']} |"
            )
        if not rtt.empty:
            w("")
            w(f"Retune fit time: {rtt['sec'].sum() / 3600:.2f} core-hours over {len(rtt)} trials.")
        w("")
    w("## Out of sample: the 866 trade days (2020-01-03 .. 2024-04-30)")
    w("")
    w(
        "Research scorer (16:00-bar recalibration, the deck's 15:30 sign(s) straddle: nearest out-of-the-money call + "
        "nearest out-of-the-money put, same-day expiry, one position); point estimates (the circular block bootstrap is "
        "off). Against this run's control of the same refit cadence (shipped configuration, W 2000, mask on, this run's "
        "libraries; stored rows against `control_r10`): "
        "Diebold-Mariano statistic on daily QLIKE (negative = lower QLIKE than the control) and HAC t on the paired daily "
        "P&L (mid) difference (positive = more P&L than the control), both `src/evaluation/diebold_mariano.dm_test` "
        "(Newey-West, Harvey-Leybourne-Newbold factor). Every arm of this run forecasts all 1469 forecast rows (suffix "
        "`_r<n>`: a refit every n sessions); the frozen and retune arms' 2018-06-25 .. 2019-12-31 forecasts lie inside the "
        "pre-tune's validation period and enter the trade days only through the scorer's trailing-250-session "
        "recalibration term."
    )
    w("")
    w("| arm | QLIKE | vs control | DM (p) | Sharpe mid | dSharpe | HAC t P&L (p) | Sharpe crossed | buy % | same position | max rel. forecast diff |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in arms.iterrows():
        w(
            f"| {r['arm']} | {r['qlike']:.4f} | {r['qlike_pct_vs_control']:+.2f} % | {r['dm_qlike_vs_control']:+.2f} "
            f"({r['dm_p']:.3f}) | {r['sharpe_mid']:.2f} | {r['dsharpe_mid_vs_control']:+.2f} | "
            f"{r['hac_t_pnl_vs_control']:+.2f} ({r['hac_p']:.3f}) | {r['sharpe_crossed']:.2f} | {r['pct_buy']:.1f} | "
            f"{r['same_position_share'] * 100:.1f} % | {r['max_rel_forecast_diff_vs_control']:.2e} |"
        )
    w("")
    ctrl = arms[arms["arm"] == "control_r10"].iloc[0]
    st = arms[arms["arm"] == "stored_T10"].iloc[0]
    w(
        f"This run's control (`control_r10`, LightGBM {meta['versions']['lightgbm']}) vs the stored shipped-config table "
        f"(`stored_T10`, the cluster's LightGBM 4.6.0, same configuration, cadence, window and mask): QLIKE "
        f"{ctrl['qlike']:.4f} vs {st['qlike']:.4f}, Sharpe mid {ctrl['sharpe_mid']:.2f} vs {st['sharpe_mid']:.2f}, "
        f"max relative forecast difference {st['max_rel_forecast_diff_vs_control']:.2e}, same position on "
        f"{st['same_position_share'] * 100:.1f} % of days."
    )
    w("")
    w("Arms: `control_r10` shipped configuration; `frozen_k<k>_r10` the pre-tune's best after its first k (merged) trials, "
      "frozen; `retune_any_r10` / `retune_margin_r10` the final pre-tuned configuration plus the light retunes above. "
      "Stored rows: the master table's forecast tables scored by this script (gate above).")
    est = cluster_estimate(recs, arms)
    if est:
        w("## CPU estimate for the full-size cluster run (from this run's fit times)")
        w("")
        for line in est:
            w(f"- {line}")
        w("")
    name = "SUMMARY.md" if (C.TAG, C.MODEL) == ("local", "lgbm") else f"SUMMARY_{C.TAG}_{C.MODEL}.md"
    (C.OUT / name).write_text("\n".join(L) + "\n", encoding="utf-8")


# the full-size run's defaults (cluster/submit_close_pretune_h2.sh, cluster/close_pretune_tasks_*.txt)
H2_STUDIES, H2_TRIALS, H2_FOLDS, H2_SLOTS, H2_RETUNE_TRIALS = 4, 2000, 7, 8, 15


def cluster_estimate(recs: list[dict], arms: pd.DataFrame) -> list[str]:
    """Fit CPU-hours of the full-size run at this run's measured fit times (this CPU, one thread)."""
    if C.TAG != "local":
        return []
    fs = np.concatenate([r["fold_sec"].ravel() for r in recs])
    slow = np.concatenate([r["fold_sec"].max(axis=1) for r in recs])  # the slowest fold of a trial
    pre = H2_STUDIES * H2_TRIALS * H2_FOLDS * fs.mean() / 3600
    wall = H2_TRIALS * slow.mean() / 3600
    out = [
        f"pre-tune, one model: {H2_STUDIES} studies x {H2_TRIALS} trials x {H2_FOLDS} folds x {fs.mean():.1f} s (this run's "
        f"mean fold fit, {len(fs)} fits) = {pre:.0f} fit CPU-hours; a study's wall time ~ {H2_TRIALS} x {slow.mean():.1f} s "
        f"(mean slowest fold of a trial here) = {wall:.1f} h, i.e. {H2_STUDIES * H2_SLOTS * wall:.0f} allocated slot-hours "
        f"with {H2_SLOTS} slots a study",
    ]
    walks = sorted(C.RUN.glob("walk_*_r10.npz"))
    if walks:
        secs = [np.load(f)["fit_sec"].mean() for f in walks]
        refit = float(np.mean(secs))
        n_arms = len(walks) + 2  # the full run adds two frozen checkpoints
        out.append(
            f"walk, one model: refit every 10 = {n_arms} arms x 147 refits x {refit:.1f} s = "
            f"{n_arms * 147 * refit / 3600:.1f} CPU-hours; refit every session = {n_arms} x 1469 x {refit:.1f} s = "
            f"{n_arms * 1469 * refit / 3600:.0f} CPU-hours (upper bounds: arms that share a configuration share fits); "
            f"retunes: 5 x {H2_RETUNE_TRIALS} trials x 2 folds x 2 paths x {fs.mean():.1f} s = "
            f"{5 * H2_RETUNE_TRIALS * 2 * 2 * fs.mean() / 3600:.1f} CPU-hours a walk task"
        )
    out.append(
        "XGBoost: the same counts at XGBoost's fit time, which this run did not measure at scale; Hoffman2 nodes "
        "differ in speed from this machine."
    )
    return out


def main() -> None:
    C.OUT.mkdir(parents=True, exist_ok=True)
    info = C.prepare_arrays()
    recs = studies()
    cur, chk = write_curves(recs)
    plot_curve(cur)
    arms = arms_table(info)
    write_summary(recs, cur, chk, arms, info)
    print(arms[["arm", "qlike", "sharpe_mid", "dm_qlike_vs_control", "hac_t_pnl_vs_control"]].to_string(index=False))
    C.say(f"report written to {C.OUT}")


if __name__ == "__main__":
    sys.exit(main())
