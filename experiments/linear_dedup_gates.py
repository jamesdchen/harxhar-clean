"""I1 gates: the per-bar 16:00 linear arms and the LSTM re-run on the de-duplicated per-bar design.

For every forecast of results/linear_subsection_dedup/arm_list.csv (experiments/linear_dedup_plan.py)
the new run (run_path) is set against the pre-dedup arm the master table scores (old_path):

  (a) design   the executor's feature-health table of the new run lists the design's columns:
               no har_ma_*_x_open / _x_close column; p = 22 / 232 / 628 for the HAR + calendar,
               live-feasible and all-features buckets (production ladder); where the old run's
               table is on disk, the old design minus the new one is exactly the twelve
               session-edge columns.  Recorded beside it: the spec's "masked cols per tune" (its
               identifiability mask: non-locked columns constant in the window or exact copies of
               an earlier column are zeroed at every tune) old vs new.
  (b) target   the same 16:00 stamps; true_adj and true_raw bitwise equal.
  (c) L1 / LS  recursive lasso and OLS: max |pred_adj new / old - 1| <= FLOAT_PATH_TOL.  Dropping a
               copy of a column cannot change an L1 or least-squares fit's predictions.
  (d) L2       ridge and elastic net: max and median |pred_adj new / old - 1| at 16:00, recorded
               (no bound: the question is how much the fit moved).
  plus, per forecast (change_by_arm.csv): the research scorer's 16:00 recalibration on each file
  (compare_mfiv_harlag._prep, the master table's arm reader) and the deck's 15:30 sign(s) trade
  (score_linear_subsection.trade_1530): positions that differ on the trade days, Sharpe mid /
  crossed old vs new; for the 13-bar table arms also the change over all 13 bars.
  (e) LSTM     the spec's own gates: experiments/gate_lstm.py identity (the LSTM's design, target
               and stamps = the linear spec's, sha256; p and the target hashes against the
               pre-dedup gate run) and determinism (local, pooled walk run twice), the cluster
               canary's repeat check (a chunk run twice, byte-identical results tables), the chunk
               reducer's gates; and the forecast change vs the pre-dedup LSTM (both selection
               rules), with the target identical.

FLOAT_PATH_TOL = 1e-8 relative: solver and cross-platform arithmetic (the canary reproduced the
pre-dedup lasso and OLS to 6e-15 / 8e-15; the same arm run on the local box and on Hoffman2 before
the change differed by 3e-11 for ridge); the failures this family has had were 1e6..1e10.

Writes results/linear_subsection_dedup/gates.csv (one row per gate and forecast) and
change_by_arm.csv (one row per forecast).
Run:  python experiments/linear_dedup_gates.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "experiments", ROOT / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import compare_mfiv_harlag as cmh  # noqa: E402
import score_linear_subsection as base  # noqa: E402

NEW = ROOT / "results" / "linear_subsection_dedup"
OLDLOGS = NEW / "old_runlogs"
LSTM_NEW = ROOT / "results" / "linear_subsection_lstm_dedup"
LSTM_OLD = ROOT / "results" / "linear_subsection_lstm"
FLOAT_PATH_TOL = 1e-8
EDGE = re.compile(r"^har_ma_\d+_x_(open|close)$")
HAR_RUNG = re.compile(r"^har_ma_\d+$")
MASK = re.compile(r"masked cols per tune min=(\d+) max=(\d+)")
BARS = (
    "bar1000",
    "bar1030",
    "bar1100",
    "bar1130",
    "bar1200",
    "bar1230",
    "bar1300",
    "bar1330",
    "bar1400",
    "bar1430",
    "bar1500",
    "bar1530",
    "bar1600",
)


def health_of(results_csv: Path) -> Path:
    return results_csv.with_name(results_csv.stem + "_feature_health.csv")


def lp(p: Path) -> Path:
    """Windows: the extended-length form for paths past MAX_PATH (the nested arm layout's
    feature-health tables reach ~265 characters)."""
    s = str(p.resolve())
    return Path("\\\\?\\" + s) if os.name == "nt" and len(s) >= 240 else p


def log_of(results_csv: Path) -> Path:
    """The arm's run.log: .../tw<tw>/run.log above causal_tune_linear/<est>/<bucket>/results_*.csv
    (or causal_tune_linear/incumbent_ols/results_*.csv)."""
    for p in results_csv.parents:
        if re.fullmatch(r"tw\d+", p.name):
            return p / "run.log"
    return results_csv.parent / "run.log"


def masked(log: Path, est: str) -> str:
    if not lp(log).is_file():
        return ""
    for line in lp(log).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(f"{est}/") and "masked cols" in line:
            m = MASK.search(line)
            if m:
                return f"{m.group(1)}-{m.group(2)}"
    return ""


def old_aux(r: pd.Series) -> tuple[Path, Path]:
    """Old feature-health table and run.log: beside a nested old arm, else the fetched copies."""
    old = ROOT / r["old_path"]
    if "arms_hoffman2" in old.parts:
        stem = f"{r['bucket']}_{r['estimator']}_bar1600"
        return OLDLOGS / f"{stem}_feature_health.csv", OLDLOGS / f"{stem}.log"
    return health_of(old), log_of(old)


def rel_change(new: np.ndarray, old: np.ndarray) -> np.ndarray:
    return np.abs(new / old - 1.0)


def trade(path: Path) -> tuple[pd.Series, dict]:
    r = cmh._prep(path)
    assert r is not None, path
    clk = r["pred_clock"].dropna()
    t = base.trade_1530(clk)
    f = clk[clk.index.strftime("%H:%M") == "16:00"]
    f.index = f.index.normalize()
    return f, t


def linear_gates(arms: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    deck = pd.read_parquet(base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index))
    gates: list[dict] = []
    change: list[dict] = []

    def g(gate: str, key: str, value, bound, ok: bool, note: str = "") -> None:
        gates.append(
            {
                "gate": gate,
                "forecast": key,
                "value": value,
                "bound": bound,
                "ok": bool(ok),
                "note": note,
            }
        )

    for _, r in arms.iterrows():
        key, est = r["key"], r["estimator"]
        newf, oldf = ROOT / r["run_path"], ROOT / r["old_path"]
        if not newf.is_file():
            g("run present", key, np.nan, "", False, f"missing {r['run_path']}")
            continue
        a, b = pd.read_csv(newf), pd.read_csv(oldf)
        c: dict = {
            "forecast": key,
            "family": r["family"],
            "master_table": r["master_table"],
            "bucket": r["bucket"],
            "estimator": est,
            "train_win": r["train_win"],
            "har_base": r["har_base"],
            "lag_scope": r["lag_scope"],
        }
        # (a) design
        hn = lp(health_of(newf))
        cols = list(pd.read_csv(hn)["feature"]) if hn.is_file() else []
        edge = [x for x in cols if EDGE.match(x)]
        c["p_new"] = len(cols)
        want = r["p_expected"]
        ok_p = bool(cols) and not edge and (pd.isna(want) or len(cols) == int(want))
        g(
            "(a) design: no session-edge column"
            + ("" if pd.isna(want) else f", p = {int(want)}"),
            key,
            len(cols),
            "" if pd.isna(want) else int(want),
            ok_p,
            f"session-edge columns present: {edge}" if edge else "",
        )
        ho, lo = old_aux(r)
        ho = lp(ho)
        if ho.is_file() and ho.stat().st_size:
            oc = list(pd.read_csv(ho)["feature"])
            dropped = sorted(set(oc) - set(cols))
            added = sorted(set(cols) - set(oc))
            c["p_old"] = len(oc)
            # two per target-HAR rung (x open, x close): 12 on the production ladder
            n_edge = 2 * sum(bool(HAR_RUNG.match(x)) for x in cols)
            ok_d = (
                len(dropped) == n_edge
                and all(EDGE.match(x) for x in dropped)
                and not added
            )
            g(
                "(a) old design minus new = the session-edge columns (2 per HAR rung; 12 production)",
                key,
                len(dropped),
                n_edge,
                ok_d,
                "" if ok_d else f"dropped {dropped[:6]}... added {added[:6]}",
            )
        if est != "ols":
            c["masked_per_tune_old"] = masked(lo, est)
            c["masked_per_tune_new"] = masked(log_of(newf), est)
        # (b) target
        same = len(a) == len(b) and bool((a["date"] == b["date"]).all())
        dt = float(np.max(np.abs(a["true_adj"] - b["true_adj"]))) if same else np.inf
        dr = float(np.max(np.abs(a["true_raw"] - b["true_raw"]))) if same else np.inf
        g(
            "(b) target: same stamps, true_adj and true_raw bitwise",
            key,
            max(dt, dr),
            0.0,
            same and dt == 0.0 and dr == 0.0,
            f"rows {len(a)} vs {len(b)}",
        )
        c["rows"] = len(a)
        if not same:
            change.append(c)
            continue
        rel = rel_change(a["pred_adj"].to_numpy(float), b["pred_adj"].to_numpy(float))
        c["pred_adj_1600_max_rel_change"] = float(rel.max())
        c["pred_adj_1600_median_rel_change"] = float(np.median(rel))
        c["pred_adj_1600_max_abs_change"] = float(
            np.max(np.abs(a["pred_adj"] - b["pred_adj"]))
        )
        if est in ("reclasso", "ols"):
            g(
                "(c) lasso / OLS: 16:00 forecast = pre-dedup (max rel)",
                key,
                float(rel.max()),
                FLOAT_PATH_TOL,
                float(rel.max()) <= FLOAT_PATH_TOL,
            )
        else:
            g(
                "(d) ridge / enet: 16:00 forecast change (max rel; median in note)",
                key,
                float(rel.max()),
                "",
                True,
                f"median {np.median(rel):.3e}",
            )
        # the trade on the research scorer's recalibration
        fn, tn = trade(newf)
        fo, to = trade(oldf)
        on = deck.index.intersection(fn.dropna().index).intersection(fo.dropna().index)
        iv = deck.loc[on, "iv_var"].to_numpy(float)
        qn = np.where(fn.reindex(on).to_numpy(float) > iv, 1, -1)
        qo = np.where(fo.reindex(on).to_numpy(float) > iv, 1, -1)
        c.update(
            trade_days=len(on),
            positions_changed=int((qn != qo).sum()),
            Sharpe_mid_old=to["Sharpe_mid"],
            Sharpe_mid_new=tn["Sharpe_mid"],
            Sharpe_crossed_old=to["Sharpe_crossed"],
            Sharpe_crossed_new=tn["Sharpe_crossed"],
        )
        # the 13-bar table arms: the change over every bar
        if int(r["segments_run"]) == len(BARS):
            mx, md, n = [], [], 0
            for bar in BARS:
                pn = Path(str(newf).replace("bar1600", bar))
                po = Path(str(oldf).replace("bar1600", bar))
                if not pn.is_file():
                    g("run present", key, np.nan, "", False, f"missing {bar}")
                    continue
                x, y = pd.read_csv(pn), pd.read_csv(po)
                ok = (
                    len(x) == len(y)
                    and bool((x["date"] == y["date"]).all())
                    and bool((x["true_adj"] == y["true_adj"]).all())
                )
                if not ok:
                    g(
                        "(b) target, every bar",
                        key,
                        np.nan,
                        0.0,
                        False,
                        f"{bar}: stamps or target differ",
                    )
                    continue
                rr = rel_change(
                    x["pred_adj"].to_numpy(float), y["pred_adj"].to_numpy(float)
                )
                mx.append(rr.max())
                md.append(np.median(rr))
                n += 1
            c["bars_compared"] = n
            c["pred_adj_allbars_max_rel_change"] = float(max(mx)) if mx else np.nan
            if est in ("reclasso", "ols"):
                g(
                    "(c) lasso: forecast = pre-dedup at all 13 bars (max rel)",
                    key,
                    c["pred_adj_allbars_max_rel_change"],
                    FLOAT_PATH_TOL,
                    n == len(BARS)
                    and c["pred_adj_allbars_max_rel_change"] <= FLOAT_PATH_TOL,
                )
            else:
                g(
                    "(d) ridge / enet: change at all 13 bars (max rel)",
                    key,
                    c["pred_adj_allbars_max_rel_change"],
                    "",
                    n == len(BARS),
                    f"{n} bars; max of per-bar medians {max(md):.3e}" if md else "",
                )
        change.append(c)
    return gates, change


def control_gates(arms: pd.DataFrame) -> list[dict]:
    """(c') attribution: every (forecast, bar) of control_arms.csv re-run with the PRE-dedup
    executor on the de-dup fleet's cluster and environment (experiments/linear_dedup_control_plan.py).
    control vs old = the cluster / environment / software drift alone; control vs new = the design
    change alone (same cluster, same environment)."""
    rows: list[dict] = []
    ca = NEW / "control_arms.csv"
    if not ca.is_file():
        return rows
    by_key = arms.set_index("key")
    for _, c in pd.read_csv(ca).iterrows():
        r = by_key.loc[c["forecast"]]
        bar = c["bar"]
        pn = ROOT / r["run_path"].replace("bar1600", bar)
        po = ROOT / r["old_path"].replace("bar1600", bar)
        pc = ROOT / r["run_path"].replace("bar1600", bar).replace(
            "results/linear_subsection_dedup",
            "results/linear_subsection_dedup_control",
            1,
        )
        base_row = {
            "forecast": c["forecast"],
            "bar": bar,
            "estimator": r["estimator"],
            "cluster_old": r["cluster_old"],
            "new_vs_old": c["max_rel_new_vs_old"],
        }
        if not pc.is_file():
            rows.append({**base_row, "ok": False, "note": "control run missing"})
            continue
        a, b, k = pd.read_csv(pn), pd.read_csv(po), pd.read_csv(pc)
        same = len(k) == len(a) == len(b) and bool((k["date"] == b["date"]).all())
        if not same:
            rows.append({**base_row, "ok": False, "note": "control stamps differ"})
            continue
        vs_old = rel_change(
            k["pred_adj"].to_numpy(float), b["pred_adj"].to_numpy(float)
        )
        vs_new = rel_change(
            k["pred_adj"].to_numpy(float), a["pred_adj"].to_numpy(float)
        )
        rows.append(
            {
                **base_row,
                "ok": True,
                "control_vs_old": float(vs_old.max()),
                "control_vs_new": float(vs_new.max()),
                "rows_control_ne_old": int((vs_old > FLOAT_PATH_TOL).sum()),
                "rows_control_ne_new": int((vs_new > FLOAT_PATH_TOL).sum()),
                "note": "",
            }
        )
    return rows


def samearch_gates(arms: pd.DataFrame) -> list[dict]:
    """(c'') the 16:00 pairs of the control, re-run pinned to the CPU architecture of the node their
    control ran on (experiments/linear_dedup_control_plan.py --samearch): the de-dup executor
    (results/linear_subsection_dedup_samearch) and the pre-dedup executor again
    (results/linear_subsection_dedup_control_rep).  control_rep vs control = same-architecture
    reproducibility; de-dup vs pre-dedup on the same architecture = the design change alone."""
    rows: list[dict] = []
    ca = NEW / "control_arms.csv"
    if (
        not ca.is_file()
        or not (ROOT / "results" / "linear_subsection_dedup_samearch").is_dir()
    ):
        return rows
    by_key = arms.set_index("key")
    ctl = pd.read_csv(ca)
    for _, c in ctl[ctl["bar"] == "bar1600"].iterrows():
        r = by_key.loc[c["forecast"]]
        run = r["run_path"]
        paths = {
            "old": ROOT / r["old_path"],
            "new": ROOT / run,
            "control": ROOT
            / run.replace(
                "results/linear_subsection_dedup",
                "results/linear_subsection_dedup_control",
                1,
            ),
            "control_rep": ROOT
            / run.replace(
                "results/linear_subsection_dedup",
                "results/linear_subsection_dedup_control_rep",
                1,
            ),
            "samearch": ROOT
            / run.replace(
                "results/linear_subsection_dedup",
                "results/linear_subsection_dedup_samearch",
                1,
            ),
        }
        row: dict = {
            "forecast": c["forecast"],
            "estimator": r["estimator"],
            "cluster_old": r["cluster_old"],
        }
        if not all(p.is_file() for p in paths.values()):
            row["note"] = "missing: " + ",".join(
                k for k, p in paths.items() if not p.is_file()
            )
            rows.append(row)
            continue
        f = {k: pd.read_csv(p)["pred_adj"].to_numpy(float) for k, p in paths.items()}
        for a, b in (
            ("samearch", "control_rep"),
            ("samearch", "control"),
            ("control_rep", "control"),
            ("samearch", "new"),
            ("samearch", "old"),
            ("control_rep", "old"),
        ):
            row[f"{a}_vs_{b}"] = float(rel_change(f[a], f[b]).max())
        rows.append(row)
    return rows


def lstm_gates() -> tuple[list[dict], list[dict]]:
    gates: list[dict] = []
    change: list[dict] = []

    def g(gate: str, key: str, value, bound, ok: bool, note: str = "") -> None:
        gates.append(
            {
                "gate": gate,
                "forecast": key,
                "value": value,
                "bound": bound,
                "ok": bool(ok),
                "note": note,
            }
        )

    p_want = {"baseline": 22, "live_feasible": 232, "all_features": 628}
    gs = LSTM_NEW / "gates" / "gate_lstm.csv"  # this run's gates (gate_lstm.py --out)
    gp = LSTM_OLD / "gates" / "gate_summary.csv"  # the pre-dedup run's gates
    if gs.is_file():
        new_s, old_s = pd.read_csv(gs), pd.read_csv(gp)
        for _, r in new_s.iterrows():
            key = f"lstm_{r['bucket']}"
            if r["gate"] == "identity":
                o = json.loads(
                    old_s[
                        (old_s["gate"] == "identity") & (old_s["bucket"] == r["bucket"])
                    ].iloc[0]["detail"]
                )
                shape = json.loads(r["X_shape"])
                p = shape[1]
                same_t = all(
                    r[k] == o[k] for k in ("sha_y", "sha_true_adj", "sha_dates")
                )
                kw_same = str(r["kwargs_differ"]) == "[]"
                ok = bool(r["ok"]) and p == p_want[r["bucket"]] and kw_same and same_t
                g(
                    "(e) LSTM identity: design = linear spec's (sha256), p, target and stamps = pre-dedup",
                    key,
                    p,
                    p_want[r["bucket"]],
                    ok,
                    f"X {shape} (pre-dedup {o['X_shape']}); sha_X {r['sha_X']} vs {o['sha_X']}; "
                    f"target/stamp hashes equal to pre-dedup {same_t}; "
                    f"stored linear true_adj max rel {r['stored_linear_true_adj_max_rel']}",
                )
            else:
                g(
                    f"(e) LSTM {r['gate']} (local, reduced budget {r['budget']})",
                    key,
                    r["rows"],
                    "",
                    bool(r["ok"]),
                    f"seconds {r['sec']}; {r['why'] if isinstance(r['why'], str) else ''}",
                )
    else:
        g(
            "(e) LSTM local gates",
            "lstm",
            np.nan,
            "",
            False,
            f"missing {gs.relative_to(ROOT)}",
        )
    can = sorted((LSTM_NEW / "logs").glob("lstm_dedup_canary.*.out"))
    if can:
        txt = can[-1].read_text(encoding="utf-8", errors="replace")
        n_ok = txt.count("byte-identical")
        g(
            "(e) LSTM cluster canary: chunk 0 run twice, results tables byte-identical",
            "lstm_live_feasible",
            n_ok,
            2,
            n_ok == 2 and "FAILED" not in txt,
            can[-1].name,
        )
    rg = LSTM_NEW / "reduce_gates.csv"
    if rg.is_file():
        t = pd.read_csv(rg)
        for _, r in t.iterrows():
            g(
                "(e) LSTM chunk reducer gates",
                f"lstm_{r.get('bucket', '')}",
                np.nan,
                "",
                bool(r["ok"]),
                str(r.get("why", "")),
            )
    for bucket in p_want:
        for infix, rule in (("", "mse"), ("qsel_", "qlike")):
            sub = (
                Path(bucket)
                / "bar1600"
                / "lstm"
                / "tw2000"
                / "causal_tune_lstm"
                / "lstm"
                / bucket
            )
            pn = LSTM_NEW / sub / f"results_{infix}bar1600.csv"
            po = LSTM_OLD / sub / f"results_{infix}bar1600.csv"
            key = f"lstm_{infix}{bucket}"
            if not pn.is_file():
                g(
                    "(e) LSTM run present",
                    key,
                    np.nan,
                    "",
                    False,
                    f"missing {pn.relative_to(ROOT)}",
                )
                continue
            a, b = pd.read_csv(pn), pd.read_csv(po)
            same = len(a) == len(b) and bool((a["date"] == b["date"]).all())
            dt = (
                float(np.max(np.abs(a["true_adj"] - b["true_adj"]))) if same else np.inf
            )
            g(
                "(b) LSTM target: same stamps, true_adj bitwise",
                key,
                dt,
                0.0,
                same and dt == 0.0,
            )
            if not same:
                continue
            rel = rel_change(
                a["pred_adj"].to_numpy(float), b["pred_adj"].to_numpy(float)
            )
            g(
                "(e) LSTM: 16:00 forecast change vs pre-dedup (max rel; median in note)",
                key,
                float(rel.max()),
                "",
                True,
                f"median {np.median(rel):.3e}",
            )
            fn, tn = trade(pn)
            fo, to = trade(po)
            change.append(
                {
                    "forecast": key,
                    "family": "LSTM",
                    "master_table": "A",
                    "bucket": bucket,
                    "estimator": f"lstm ({rule} rule)",
                    "rows": len(a),
                    "pred_adj_1600_max_rel_change": float(rel.max()),
                    "pred_adj_1600_median_rel_change": float(np.median(rel)),
                    "pred_adj_1600_max_abs_change": float(
                        np.max(np.abs(a["pred_adj"] - b["pred_adj"]))
                    ),
                    "Sharpe_mid_old": to["Sharpe_mid"],
                    "Sharpe_mid_new": tn["Sharpe_mid"],
                    "Sharpe_crossed_old": to["Sharpe_crossed"],
                    "Sharpe_crossed_new": tn["Sharpe_crossed"],
                    "trade_days": tn["deck_days"],
                }
            )
    return gates, change


def main() -> None:
    arms = pd.read_csv(NEW / "arm_list.csv")
    g1, c1 = linear_gates(arms)
    g2, c2 = lstm_gates()
    ctrl = pd.DataFrame(control_gates(arms))
    g3: list[dict] = []
    if len(ctrl):
        ctrl.to_csv(NEW / "control_attribution.csv", index=False)
        for f, d in ctrl.groupby("forecast", sort=True):
            ok = bool(d["ok"].all())
            cvo = float(d["control_vs_old"].max()) if "control_vs_old" in d else np.nan
            cvn = float(d["control_vs_new"].max()) if "control_vs_new" in d else np.nan
            g3.append(
                {
                    "gate": "(c') attribution: pre-dedup executor, same cluster as the re-run, "
                    "vs the stored pre-dedup arm (max rel over the bars that moved)",
                    "forecast": f,
                    "value": cvo,
                    "bound": "",
                    "ok": ok,
                    "note": f"bars {','.join(d['bar'])}; new vs old {d['new_vs_old'].max():.3e}; "
                    f"control vs new {cvn:.3e}; old run on {d['cluster_old'].iloc[0]}",
                }
            )
        cols = [
            x for x in ("new_vs_old", "control_vs_old", "control_vs_new") if x in ctrl
        ]
        print(ctrl.groupby("cluster_old")[cols].max().to_string())
    sa = pd.DataFrame(samearch_gates(arms))
    if len(sa) and "samearch_vs_control_rep" in sa:
        sa.to_csv(NEW / "samearch_attribution.csv", index=False)
        for _, s in sa.iterrows():
            ok = bool(pd.notna(s.get("samearch_vs_control_rep")))
            g3.append(
                {
                    "gate": "(c'') same CPU architecture: de-dup vs pre-dedup executor, 16:00 (max rel)",
                    "forecast": s["forecast"],
                    "value": s.get("samearch_vs_control_rep", np.nan),
                    "bound": FLOAT_PATH_TOL if s["estimator"] == "reclasso" else "",
                    "ok": ok
                    and (
                        s["estimator"] != "reclasso"
                        or s["samearch_vs_control_rep"] <= FLOAT_PATH_TOL
                    ),
                    "note": (
                        f"same-arch repeat of the pre-dedup run vs its first run "
                        f"{s.get('control_rep_vs_control', np.nan):.3e}; de-dup vs the stored arm "
                        f"{s.get('samearch_vs_old', np.nan):.3e}"
                    ),
                }
            )
        print(sa.drop(columns=["cluster_old"]).to_string())
    gates = pd.DataFrame(g1 + g2 + g3)
    gates.to_csv(NEW / "gates.csv", index=False)
    change = pd.DataFrame(c1 + c2)
    change.to_csv(NEW / "change_by_arm.csv", index=False)
    print(gates.groupby("gate")["ok"].agg(["size", "sum"]).to_string())
    bad = gates[~gates["ok"]]
    if len(bad):
        print(bad.to_string())
    by = change.groupby("estimator")
    print(
        by[["pred_adj_1600_max_rel_change", "pred_adj_1600_median_rel_change"]]
        .max()
        .to_string()
    )
    if "positions_changed" in change:
        print(
            f"positions changed (linear, all trade days): {int(change['positions_changed'].sum())}"
        )
    print(
        f"wrote {len(gates)} gate rows ({int(gates['ok'].sum())} ok), {len(change)} forecasts"
    )


if __name__ == "__main__":
    main()
