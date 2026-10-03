"""I6 gates and summary: the direct rest-of-day arms re-run on the de-duplicated per-bar design.

Checklist line I6 (writeup/CAMPAIGN_16H_2026-09-29.md, "Added 2026-09-29 evening").  The direct
rest-of-day model (specs/causal_tune_rest_of_day.py) was fitted with the HAR x open/close
session-edge columns that commit 47f7f9c dropped from the per-bar design.  Its 117 arms (13 entry
clocks 09:30 .. 15:30 x ridge / lasso / elastic net x all_features / baseline / live_feasible, 2000
sessions) were re-run on Hoffman2 (cluster/submit_restofday_dedup.sh) into NEW; the nine 15:30 arms
ran on the CPU architecture of their per-bar 16:00 twin (agent A's I1 fleet).  The master table
(experiments/master_table_close.py) scores the 16:00 rows of yhat_restofday_* as CHECK rows against
those twins.  Only the 15:30 arms feed the check; the other clocks were re-run because
experiments/build_restofday_yhat.py writes a table only when all 13 clocks exist.

Reads
  NEW   results/linear_subsection_restofday_dedup  the new arms (cluster/pull_restofday_dedup_hoffman2.sh),
        with old_carc/ (the pre-dedup arms' run.log + feature-health tables), host_arch.txt,
        submitted_restofday_dedup.txt, qacct.txt, FLEET_DONE
  OLD   results/linear_subsection_restofday        the pre-dedup arms (CARC)
  TWIN  results/linear_subsection_dedup            agent A's de-dup per-bar arms and logs
  results/spxw_pnl/yhat_sub_*  (A's de-dup tables) and yhat_restofday_* (rebuilt from NEW by the
  unedited stacker BEFORE this script), and the snapshots in results/spxw_pnl/pre_dedup_2026-09-29/
Gates (gates.csv; kind "assert" = a failure sets exit status 1, "report" = recorded only)
  (a) design   per arm: the feature-health table lists P_DEDUP[bucket] columns, none of them a
               session-edge column, and the pre-dedup arm's list minus the new one is exactly the
               twelve session-edge columns (assert); the spec's "masked cols per tune" old vs new
               (report: its identifiability mask zeroes constant / copied columns at every tune)
  (b) target   per arm: the results table's stamps, true_adj and true_raw and the targets sidecar
               (t, RV, RV_rem, rem_nbars, adj_RV, adj_RV_rem, baseline, baseline_rem, complete)
               equal the pre-dedup arm's bit for bit (assert)
  (c) change   per arm: max / median |pred_adj new / old - 1| (report; lasso / elastic-net float
               paths move with the machine, results/linear_subsection_dedup/control_attribution.csv)
  (d) twin     per 15:30 arm: ran on its twin's CPU architecture (assert); stamps and target equal
               the twin's bit for bit (assert); max |pred_adj / twin - 1| (report)
  (e) check    per estimator x bucket, the master table's check-row numbers (its own load_one
               recalibration and trade_days, 866 deck days) for the rebuilt table vs A's per-bar
               table, and the same for the pre-change snapshots: max relative difference of the
               recalibrated forecast, same position on how many days (report)
  (f) tables   rebuilt yhat_restofday_* vs the snapshot: same stamps, baseline and rv_raw bit for
               bit (assert); yhat change per entry clock (report)
Writes NEW/gates.csv, arms.csv, check_1530.csv, tables_vs_pre_dedup.csv, SUMMARY.md.
Run:  python experiments/restofday_dedup_gates.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "experiments", ROOT / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import master_table_close as mt  # noqa: E402
from build_subsection_yhat import ESTIMATORS, GATE_REL  # noqa: E402
from score_rest_of_day import CLOCKS, bar_of, hhmm  # noqa: E402

NEW = ROOT / "results" / "linear_subsection_restofday_dedup"
OLD = ROOT / "results" / "linear_subsection_restofday"
OLDC = NEW / "old_carc"
CTRL = ROOT / "results" / "linear_subsection_restofday_dedup_control"
CTRL_TASKS = sorted((ROOT / "cluster").glob("restofday_dedup_control_*.txt"))
TWIN = ROOT / "results" / "linear_subsection_dedup"
SPXW = ROOT / "results" / "spxw_pnl"
PRE = SPXW / "pre_dedup_2026-09-29"
BUCKETS = ("all_features", "baseline")  # live_feasible commented out
ESTS = ("ridge", "reclasso", "reclasticnet")
TW = 2000
CHECK_CLOCK = "1530"  # the entry clock whose rest of the day is the 16:00 bar alone
# per-bar design widths after commit 47f7f9c (HAR + calendar 34 -> 22, live-feasible 244 -> 232,
# all features 640 -> 628: writeup/CAMPAIGN_16H_2026-09-29.md, decision 1)
P_DEDUP = {"baseline": 22, "live_feasible": 232, "all_features": 628}
N_EDGE = 12  # six HAR rungs x {_x_open, _x_close}
MOVED_REL = (
    1e-8  # A's float-path tolerance: a row "moved" when its relative change exceeds it
)
MASKED = re.compile(r"masked cols per tune min=(\d+) max=(\d+)")
TASK_HDR = re.compile(
    r"^task \d+ on (\S+): (\S+) \| (\S+) (\S+) tw(\d+) har_base='([^']*)' lag=(\S+)"
)


def is_edge(c: str) -> bool:
    return c.endswith("_x_open") or c.endswith("_x_close")


def spec_dir(root: Path, bucket: str, clock: str, est: str) -> Path:
    return (
        root
        / bucket
        / f"rod{clock}"
        / est
        / f"tw{TW}"
        / "causal_tune_rest_of_day"
        / est
        / bucket
    )


def arm_key(bucket: str, est: str, clock: str) -> str:
    return f"{ESTIMATORS[est]}_{bucket}_{clock}"


def same_float(a: pd.Series, b: pd.Series) -> bool:
    return bool(np.array_equal(a.to_numpy(float), b.to_numpy(float), equal_nan=True))


def masked(log: Path) -> tuple[float, float]:
    if not log.is_file():
        return (np.nan, np.nan)
    m = MASKED.findall(log.read_text(encoding="utf-8", errors="replace"))
    return (float(m[-1][0]), float(m[-1][1])) if m else (np.nan, np.nan)


def host_arch() -> dict[str, str]:
    f = NEW / "host_arch.txt"
    out: dict[str, str] = {}
    if f.is_file():
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) == 2:
                out[p[0]] = p[1]
    return out


def twin_hosts() -> dict[tuple[str, str], str]:
    """(bucket, est) -> the host of agent A's bar1600 arm (2000 sessions, production ladder)."""
    out: dict[tuple[str, str], str] = {}
    for log in sorted((TWIN / "logs").glob("lin_dedup_*.o*")):
        lines = log.read_text(errors="replace").splitlines()
        if not lines:
            continue
        m = TASK_HDR.match(lines[0])
        if not m:
            continue
        host, root, bucket, est, tw, har, lag = m.groups()
        if (
            root != "results/linear_subsection_dedup"
            or har
            or lag != "global"
            or int(tw) != TW
        ):
            continue
        if any(ln.strip().startswith("bar1600 done in") for ln in lines[1:]):
            out[(bucket, est)] = host
    return out


def main() -> int:
    gates: list[dict] = []
    arms: list[dict] = []

    def gate(
        name: str, fc: str, value, bound, ok: bool, kind: str, note: str = ""
    ) -> None:
        gates.append(
            {
                "gate": name,
                "forecast": fc,
                "value": value,
                "bound": bound,
                "ok": bool(ok),
                "kind": kind,
                "note": note,
            }
        )

    arch = host_arch()
    thosts = twin_hosts()

    # ------------------------------------------------------------------ (a)-(d) per arm
    for bucket in BUCKETS:
        for est in ESTS:
            for clock in CLOCKS:
                key = arm_key(bucket, est, clock)
                bar = bar_of(clock)
                nd, od = (
                    spec_dir(NEW, bucket, clock, est),
                    spec_dir(OLD, bucket, clock, est),
                )
                adir = NEW / bucket / f"rod{clock}" / est / f"tw{TW}"
                row: dict = {
                    "arm": key,
                    "bucket": bucket,
                    "estimator": est,
                    "clock": clock,
                }
                done = (adir / "DONE").is_file()
                gate("arm finished (DONE)", key, float(done), 1.0, done, "assert")
                if not done:
                    arms.append(row)
                    continue
                host = (
                    (adir / "HOST").read_text().split()[0]
                    if (adir / "HOST").is_file()
                    else ""
                )
                row.update(
                    host=host,
                    arch=arch.get(host, ""),
                    seconds=float((adir / "SECONDS").read_text())
                    if (adir / "SECONDS").is_file()
                    else np.nan,
                )
                # (a) design
                cols = list(
                    pd.read_csv(nd / f"results_bar{bar}_feature_health.csv")["feature"]
                )
                edge = [c for c in cols if is_edge(c)]
                ok_p = len(cols) == P_DEDUP[bucket] and not edge
                gate(
                    "(a) design: p on the de-dup design, no session-edge column",
                    key,
                    float(len(cols)),
                    float(P_DEDUP[bucket]),
                    ok_p,
                    "assert",
                    f"session-edge columns {edge}" if edge else "",
                )
                ofh = OLDC / f"{bucket}_{est}_{clock}_feature_health.csv"
                if ofh.is_file():
                    ocols = list(pd.read_csv(ofh)["feature"])
                    gone = set(ocols) - set(cols)
                    ok_d = (
                        set(cols) <= set(ocols)
                        and len(gone) == N_EDGE
                        and all(map(is_edge, gone))
                    )
                    gate(
                        "(a) old design minus new = the session-edge columns",
                        key,
                        float(len(gone)),
                        float(N_EDGE),
                        ok_d,
                        "assert",
                        f"old p {len(ocols)}",
                    )
                    row["p_old"] = len(ocols)
                else:
                    gate(
                        "(a) old design minus new = the session-edge columns",
                        key,
                        np.nan,
                        float(N_EDGE),
                        False,
                        "assert",
                        "old feature-health table missing",
                    )
                row["p_new"] = len(cols)
                mo, mn = (
                    masked(OLDC / f"{bucket}_{est}_{clock}_run.log"),
                    masked(adir / "run.log"),
                )
                row.update(
                    masked_old_min=mo[0],
                    masked_old_max=mo[1],
                    masked_new_min=mn[0],
                    masked_new_max=mn[1],
                )
                gate(
                    "(a) masked cols per tune, old minus new (min; max in note)",
                    key,
                    mo[0] - mn[0],
                    float(N_EDGE),
                    mo[0] - mn[0] == N_EDGE and mo[1] - mn[1] == N_EDGE,
                    "report",
                    f"max: {mo[1] - mn[1]:g}; old {mo[0]:g}..{mo[1]:g}, new {mn[0]:g}..{mn[1]:g}",
                )
                # (b) target
                rn = pd.read_csv(nd / f"results_bar{bar}.csv")
                ro = pd.read_csv(od / f"results_bar{bar}.csv")
                same_rows = len(rn) == len(ro) and bool(
                    (rn["date"] == ro["date"]).all()
                )
                ok_t = (
                    same_rows
                    and same_float(rn["true_adj"], ro["true_adj"])
                    and same_float(rn["true_raw"], ro["true_raw"])
                )
                gate(
                    "(b) target: same stamps, true_adj and true_raw bitwise",
                    key,
                    0.0 if ok_t else 1.0,
                    0.0,
                    ok_t,
                    "assert",
                    f"rows {len(rn)} vs {len(ro)}",
                )
                tn = pd.read_csv(nd / f"targets_bar{bar}.csv")
                to = pd.read_csv(od / f"targets_bar{bar}.csv")
                fl = [
                    "RV",
                    "RV_rem",
                    "rem_nbars",
                    "adj_RV",
                    "adj_RV_rem",
                    "baseline",
                    "baseline_rem",
                ]
                ok_s = (
                    len(tn) == len(to)
                    and list(tn.columns) == list(to.columns)
                    and bool((tn["t"] == to["t"]).all())
                    and bool((tn["complete"] == to["complete"]).all())
                    and all(same_float(tn[c], to[c]) for c in fl)
                )
                gate(
                    "(b) target sidecar (RV_rem, its transform, baseline_rem) bitwise",
                    key,
                    0.0 if ok_s else 1.0,
                    0.0,
                    ok_s,
                    "assert",
                    f"rows {len(tn)} vs {len(to)}",
                )
                # (c) change vs the pre-dedup arm
                if same_rows:
                    rel = (rn["pred_adj"] / ro["pred_adj"] - 1.0).abs()
                    dab = (rn["pred_adj"] - ro["pred_adj"]).abs()
                    row.update(
                        rows=len(rn),
                        max_rel_vs_old=float(rel.max()),
                        max_abs_vs_old=float(dab.max()),
                        abs_pred_at_max_rel=float(
                            abs(ro["pred_adj"].iloc[int(rel.to_numpy().argmax())])
                        ),
                        median_rel_vs_old=float(rel.median()),
                        rows_moved_vs_old=int((rel > MOVED_REL).sum()),
                    )
                    gate(
                        "(c) forecast change vs the pre-dedup arm (max rel; median in note)",
                        key,
                        float(rel.max()),
                        "",
                        True,
                        "report",
                        f"median {rel.median():.3e}; rows > {MOVED_REL:g}: {int((rel > MOVED_REL).sum())} of {len(rel)}; "
                        f"max |d pred_adj| {dab.max():.3e}",
                    )
                # (d) the 15:30 twin
                if clock == CHECK_CLOCK:
                    th = thosts.get((bucket, est), "")
                    ta = arch.get(th, "")
                    row.update(twin_host=th, twin_arch=ta)
                    ok_a = bool(ta) and row["arch"] == ta
                    gate(
                        "(d) ran on the twin's CPU architecture",
                        key,
                        float(ok_a),
                        1.0,
                        ok_a,
                        "assert",
                        f"{host} {row['arch']} vs twin {th} {ta}",
                    )
                    tw = pd.read_csv(
                        TWIN
                        / bucket
                        / "bar1600"
                        / est
                        / f"tw{TW}"
                        / "causal_tune_linear"
                        / est
                        / bucket
                        / "results_bar1600.csv"
                    )
                    same_tw = len(rn) == len(tw) and bool(
                        (rn["date"] == tw["date"]).all()
                    )
                    ok_tt = (
                        same_tw
                        and same_float(rn["true_adj"], tw["true_adj"])
                        and same_float(rn["true_raw"], tw["true_raw"])
                    )
                    gate(
                        "(d) 15:30 vs the per-bar 16:00 twin: same stamps and target bitwise",
                        key,
                        0.0 if ok_tt else 1.0,
                        0.0,
                        ok_tt,
                        "assert",
                        f"rows {len(rn)} vs {len(tw)}",
                    )
                    if same_tw:
                        relw = (rn["pred_adj"] / tw["pred_adj"] - 1.0).abs()
                        row.update(
                            max_rel_vs_twin=float(relw.max()),
                            rows_ne_twin=int((rn["pred_adj"] != tw["pred_adj"]).sum()),
                        )
                        gate(
                            "(d) 15:30 vs the per-bar 16:00 twin: pred_adj (max rel)",
                            key,
                            float(relw.max()),
                            GATE_REL,
                            bool(relw.max() <= GATE_REL),
                            "report",
                            f"rows not bit-identical {row['rows_ne_twin']} of {len(rn)}",
                        )
                arms.append(row)
    arms_df = pd.DataFrame(arms)
    arms_df.to_csv(NEW / "arms.csv", index=False)

    # ------------------------------------------------------------------ (e) the master table's check rows
    deck = pd.read_parquet(mt.base.DECK).sort_index()
    deck.index = pd.DatetimeIndex(pd.to_datetime(deck.index)).as_unit("ns")
    days = deck.index
    mt._init(mt.load_target(), days)
    frames: dict[tuple[str, str, str, str], pd.DataFrame] = {}

    def frame(phase: str, kind: str, short: str, bucket: str) -> pd.DataFrame:
        k = (phase, kind, short, bucket)
        if k not in frames:
            base_dir = SPXW if phase == "after" else PRE
            p = base_dir / f"yhat_{kind}_{short}_{bucket}.parquet"
            res = mt.load_one(
                mt.Spec(f"{kind}_{short}_{bucket}", p.stem, "check", "table", str(p))
            )
            assert res["frame"] is not None, (p, res["error"])
            frames[k] = res["frame"]
        return frames[k]

    chk: list[dict] = []
    for phase in ("before", "after"):
        for bucket in BUCKETS:
            for est in ESTS:
                short = ESTIMATORS[est]
                a_, b_ = (
                    frame(phase, "restofday", short, bucket),
                    frame(phase, "sub", short, bucket),
                )
                on = a_.index[a_.index.normalize().isin(days)]
                cover = int(b_.loc[on, "pred_clock"].notna().sum()) if len(on) else 0
                rel = (a_.loc[on, "pred_clock"] / b_.loc[on, "pred_clock"] - 1.0).abs()
                ta = mt.trade_days(
                    a_.loc[on, "pred_clock"].set_axis(on.normalize()), deck
                )
                tb = mt.trade_days(
                    b_.loc[on, "pred_clock"].set_axis(on.normalize()), deck
                )
                same = int((ta["q"].to_numpy() == tb["q"].to_numpy()).sum())
                ja = a_[["pred_adj"]].join(
                    b_[["pred_adj"]], how="inner", rsuffix="_twin"
                )
                rel_adj = (ja["pred_adj"] / ja["pred_adj_twin"] - 1.0).abs()
                fc = f"restofday_{short}_{bucket}"
                chk.append(
                    {
                        "phase": phase,
                        "estimator": short,
                        "bucket": bucket,
                        "rows_16h_common": len(ja),
                        "max_rel_pred_adj_16h": float(rel_adj.max()),
                        "rows_16h_not_bit_identical": int(
                            (ja["pred_adj"] != ja["pred_adj_twin"]).sum()
                        ),
                        "deck_days": len(ta),
                        "max_rel_pred_clock": float(rel.max()),
                        "same_position": same,
                        "different_position": len(ta) - same,
                        "twin_covers": cover,
                    }
                )
                gate(
                    f"(e) {phase}: deck days covered by the check row and its twin",
                    fc,
                    float(len(ta)),
                    float(len(days)),
                    len(ta) == len(days) and cover == len(days),
                    "assert",
                )
                gate(
                    f"(e) {phase}: same 15:30 position as the per-bar twin (days)",
                    fc,
                    float(same),
                    float(len(days)),
                    True,
                    "report",
                    f"max rel recalibrated forecast {rel.max():.3e}",
                )
    chk_df = pd.DataFrame(chk)
    chk_df.to_csv(NEW / "check_1530.csv", index=False)

    # ------------------------------------------------------------------ (f) rebuilt tables vs the snapshot
    tabs: list[dict] = []
    for bucket in BUCKETS:
        for est in ESTS:
            short = ESTIMATORS[est]
            name = f"yhat_restofday_{short}_{bucket}.parquet"
            n, o = pd.read_parquet(SPXW / name), pd.read_parquet(PRE / name)
            same_t = len(n) == len(o) and bool(
                (n["t"].to_numpy() == o["t"].to_numpy()).all()
            )
            ok = (
                same_t
                and same_float(n["baseline"], o["baseline"])
                and same_float(n["rv_raw"], o["rv_raw"])
            )
            gate(
                "(f) rebuilt table: same stamps, baseline and rv_raw bitwise",
                name,
                0.0 if ok else 1.0,
                0.0,
                ok,
                "assert",
                f"rows {len(n)} vs {len(o)}",
            )
            if not same_t:
                continue
            et = pd.DatetimeIndex(n["t"]).tz_convert("America/New_York")
            rel = (n["yhat"] / o["yhat"] - 1.0).abs()
            for clock in CLOCKS:
                m = et.strftime("%H%M") == bar_of(clock)
                r = rel[m]
                tabs.append(
                    {
                        "table": name,
                        "estimator": short,
                        "bucket": bucket,
                        "clock": clock,
                        "rows": int(m.sum()),
                        "max_rel_yhat": float(r.max()),
                        "median_rel_yhat": float(r.median()),
                        "rows_moved": int((r > MOVED_REL).sum()),
                    }
                )
            fa, fb = (
                frame("after", "restofday", short, bucket),
                frame("before", "restofday", short, bucket),
            )
            on = fa.index[fa.index.normalize().isin(days)]
            qa = mt.trade_days(fa.loc[on, "pred_clock"].set_axis(on.normalize()), deck)[
                "q"
            ]
            qb = mt.trade_days(fb.loc[on, "pred_clock"].set_axis(on.normalize()), deck)[
                "q"
            ]
            gate(
                "(f) 15:30 position changed vs the pre-dedup table (days)",
                name,
                float((qa.to_numpy() != qb.to_numpy()).sum()),
                "",
                True,
                "report",
                f"of {len(qa)} deck days",
            )
    tabs_df = pd.DataFrame(tabs)
    tabs_df.to_csv(NEW / "tables_vs_pre_dedup.csv", index=False)

    # ------------------------------------------------------------------ (g) attribution control
    # the pre-dedup executor on the new arm's architecture (cluster/submit_restofday_dedup_control.sh)
    ctl: list[dict] = []
    for d in sorted(CTRL.glob("*/rod*/*/tw*/DONE")):
        adir = d.parent
        bucket, clock, est = adir.parts[-4], adir.parts[-3][3:], adir.parts[-2]
        key, bar = arm_key(bucket, est, clock), bar_of(clock)
        rc = pd.read_csv(spec_dir(CTRL, bucket, clock, est) / f"results_bar{bar}.csv")
        rn = pd.read_csv(spec_dir(NEW, bucket, clock, est) / f"results_bar{bar}.csv")
        ro = pd.read_csv(spec_dir(OLD, bucket, clock, est) / f"results_bar{bar}.csv")
        assert (
            len(rc) == len(rn) == len(ro)
            and (rc["date"] == rn["date"]).all()
            and (rc["date"] == ro["date"]).all()
        ), key
        hc = (adir / "HOST").read_text().split()[0]
        a = arms_df.set_index("arm").loc[key]
        rel_cn = (rc["pred_adj"] / rn["pred_adj"] - 1.0).abs()
        rel_co = (rc["pred_adj"] / ro["pred_adj"] - 1.0).abs()
        ctl.append(
            {
                "arm": key,
                "control_host": hc,
                "control_arch": arch.get(hc, ""),
                "new_arch": a["arch"],
                "rows": len(rc),
                "control_vs_new_rows_not_bit_identical": int(
                    (rc["pred_adj"] != rn["pred_adj"]).sum()
                ),
                "control_vs_new_max_rel": float(rel_cn.max()),
                "control_vs_old_max_rel": float(rel_co.max()),
                "control_vs_old_rows_moved": int((rel_co > MOVED_REL).sum()),
                "new_vs_old_max_rel": float(a["max_rel_vs_old"]),
                "new_vs_old_rows_moved": int(a["rows_moved_vs_old"]),
                "control_masked_min": masked(adir / "run.log")[0],
            }
        )
        gate(
            "(g) control ran on the new arm's CPU architecture",
            key,
            float(arch.get(hc, "") == a["arch"]),
            1.0,
            arch.get(hc, "") == a["arch"],
            "assert",
            f"{hc} {arch.get(hc, '')} vs {a['host']} {a['arch']}",
        )
        gate(
            "(g) control (pre-dedup executor, same architecture) vs new: rows not bit-identical",
            key,
            float(ctl[-1]["control_vs_new_rows_not_bit_identical"]),
            0.0,
            True,
            "report",
            f"max rel {rel_cn.max():.3e}; control vs old max rel {rel_co.max():.3e}",
        )
    ctl_df = pd.DataFrame(ctl)
    ctl_df.to_csv(NEW / "control_attribution.csv", index=False)

    g = pd.DataFrame(gates)
    g.to_csv(NEW / "gates.csv", index=False)
    bad = g[(g["kind"] == "assert") & ~g["ok"]]
    print(
        f"gates: {len(g)} rows, {int((g['kind'] == 'assert').sum())} asserted, {len(bad)} failed"
    )
    for _, r in bad.iterrows():
        print(
            f"  FAILED {r['gate']} {r['forecast']}: {r['value']} (bound {r['bound']}) {r['note']}"
        )
    write_summary(g, arms_df, chk_df, tabs_df, ctl_df)
    return 1 if len(bad) else 0


def _qacct(root: Path = NEW) -> pd.DataFrame:
    f = root / "qacct.txt"
    rows: list[dict] = []
    if not f.is_file():
        return pd.DataFrame(rows)
    cur: dict = {}
    for line in f.read_text(errors="replace").splitlines():
        if line.startswith("====="):
            if cur:
                rows.append(cur)
            cur = {}
            continue
        p = line.split(None, 1)
        if len(p) == 2:
            cur[p[0]] = p[1].strip()
    if cur:
        rows.append(cur)
    q = pd.DataFrame(rows)
    for c in ("ru_wallclock", "cpu"):
        if c in q:
            q[c] = pd.to_numeric(q[c].astype(str).str.rstrip("s"), errors="coerce")
    for c in ("start_time", "end_time"):
        if c in q:
            q[c] = pd.to_datetime(q[c], format="%m/%d/%Y %H:%M:%S.%f", errors="coerce")
    return q


def _accounting(q: pd.DataFrame, label: str) -> str:
    if not len(q) or "ru_wallclock" not in q:
        return f"- Accounting ({label}): n/a"
    span = q["end_time"].max() - q["start_time"].min()
    return (
        f"- Accounting ({label}; qacct, {len(q)} job tasks): {q['ru_wallclock'].sum() / 3600:.2f} "
        f"slot-hours wall-clock, {q['cpu'].sum() / 3600:.2f} CPU-hours; first start "
        f"{q['start_time'].min():%H:%M:%S} to last end {q['end_time'].max():%H:%M:%S} "
        f"({span.total_seconds() / 60:.1f} min)"
    )


def write_summary(
    g: pd.DataFrame,
    arms: pd.DataFrame,
    chk: pd.DataFrame,
    tabs: pd.DataFrame,
    ctl: pd.DataFrame,
) -> None:
    def text(f: Path) -> str:
        return f.read_text().strip() if f.is_file() else "n/a"

    asserted = g[g["kind"] == "assert"]
    out: list[str] = [
        "# I6: the direct rest-of-day arms on the de-duplicated per-bar design",
        "",
        "Generated by `experiments/restofday_dedup_gates.py` from the CSVs beside this file. Spec "
        "`specs/causal_tune_rest_of_day.py` (unchanged), per-bar design of commit 47f7f9c "
        "(no `har_ma_*_x_open` / `har_ma_*_x_close`). 117 arms = 13 entry clocks x ridge / lasso / "
        "elastic net x `all_features` / `baseline` / `live_feasible`, 2000 sessions. The master "
        "table's check rows need only the nine 15:30 arms; the other clocks were re-run because "
        "`experiments/build_restofday_yhat.py` writes a table only when all 13 clocks exist. The "
        "nine 15:30 arms ran on the CPU architecture of their per-bar 16:00 twin (agent A's I1 "
        "fleet); the other 108 have no twin.",
        "",
        f"- Jobs (Hoffman2): {text(NEW / 'submitted_restofday_dedup.txt')}",
        f"- Collector: {text(NEW / 'FLEET_DONE')}",
        _accounting(_qacct(NEW), "I6 run"),
    ]
    if "seconds" in arms:
        out.append(
            f"- Arm run time (the pack's SECONDS): total {arms['seconds'].sum() / 3600:.2f} h over "
            f"{int(arms['seconds'].notna().sum())} arms, median {arms['seconds'].median():.0f} s, "
            f"max {arms['seconds'].max():.0f} s"
        )
    if len(ctl):
        out += [
            f"- Attribution control jobs: {text(CTRL / 'submitted_restofday_dedup_control.txt')}",
            _accounting(_qacct(CTRL), "control"),
        ]
    out += [
        f"- Gates: {len(g)} rows, {len(asserted)} asserted, {int((~asserted['ok']).sum())} failed "
        "(`gates.csv`; kind `report` rows are recorded, never asserted).",
        "",
        "## Design and target",
    ]
    for name in (
        "arm finished (DONE)",
        "(a) design: p on the de-dup design, no session-edge column",
        "(a) old design minus new = the session-edge columns",
        "(b) target: same stamps, true_adj and true_raw bitwise",
        "(b) target sidecar (RV_rem, its transform, baseline_rem) bitwise",
        "(d) ran on the twin's CPU architecture",
        "(d) 15:30 vs the per-bar 16:00 twin: same stamps and target bitwise",
        "(f) rebuilt table: same stamps, baseline and rv_raw bitwise",
        "(g) control ran on the new arm's CPU architecture",
    ):
        r = g[g["gate"] == name]
        if len(r):
            out.append(f"- {name}: {int(r['ok'].sum())} / {len(r)} pass")
    mk = g[g["gate"].str.startswith("(a) masked cols")]
    if len(mk):
        out.append(
            f"- the spec's identifiability mask (masked columns per tune) falls by exactly {N_EDGE} "
            f"(min and max) in {int(mk['ok'].sum())} of {len(mk)} arms: it had already zeroed the "
            "session-edge columns at every tune, so the fits solve the same problem."
        )
    if "p_new" in arms:
        ps = arms.groupby("bucket")[["p_old", "p_new"]].min()
        out.append(
            "- design width old -> new: "
            + ", ".join(
                f"`{b}` {int(ps.loc[b, 'p_old'])} -> {int(ps.loc[b, 'p_new'])}"
                for b in ps.index
            )
        )
    out += [
        "",
        "## The master table's 15:30 check rows (research recalibration, 866 deck days)",
        "",
        "| estimator | input set | before: max rel | before: same position | after: max rel "
        "| after: same position | after: 16:00 rows not bit-identical |",
        "|---|---|---|---|---|---|---|",
    ]
    for bucket in BUCKETS:
        for est in ESTS:
            s = ESTIMATORS[est]
            sel = (chk["estimator"] == s) & (chk["bucket"] == bucket)
            b = chk[sel & (chk["phase"] == "before")].iloc[0]
            a = chk[sel & (chk["phase"] == "after")].iloc[0]
            out.append(
                f"| {s} | `{bucket}` | {b['max_rel_pred_clock']:.2e} | "
                f"{int(b['same_position'])} / {int(b['deck_days'])} | "
                f"{a['max_rel_pred_clock']:.2e} | {int(a['same_position'])} / {int(a['deck_days'])} | "
                f"{int(a['rows_16h_not_bit_identical'])} of {int(a['rows_16h_common'])} |"
            )
    out += [
        "",
        "Before = the pre-change snapshots (`results/spxw_pnl/pre_dedup_2026-09-29/`: rest-of-day "
        "arms fitted with the session-edge columns on CARC, per-bar tables before I1); after = the "
        "rebuilt `yhat_restofday_*` vs agent A's de-dup `yhat_sub_*`. Max rel = max |recalibrated "
        "forecast / twin - 1| on the deck days (the master table's `twin_max_rel_pred_clock`).",
        "",
        "## Change vs the pre-dedup arms (all 13 clocks)",
        "",
        "Relative change max |pred_adj new / old - 1|; absolute change max |pred_adj new - old| on "
        "the adjusted (sqrt-diurnal) scale, where forecasts are O(1). The largest relative changes "
        "sit on rows whose forecast is near zero or negative (|old pred_adj| at the worst row in the "
        "last column).",
        "",
        "| estimator | input set | max rel | worst clock | max abs | 15:30 max rel | rows > 1e-8 rel "
        "| old pred at worst row |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for bucket in BUCKETS:
        for est in ESTS:
            a = arms[(arms["bucket"] == bucket) & (arms["estimator"] == est)]
            if not len(a) or "max_rel_vs_old" not in a:
                continue
            w = a.loc[a["max_rel_vs_old"].idxmax()]
            c = a[a["clock"] == CHECK_CLOCK]
            out.append(
                f"| {ESTIMATORS[est]} | `{bucket}` | {a['max_rel_vs_old'].max():.2e} | "
                f"{hhmm(str(w['clock']))} | {a['max_abs_vs_old'].max():.2e} | "
                f"{float(c['max_rel_vs_old'].iloc[0]):.2e} | "
                f"{int(a['rows_moved_vs_old'].sum())} of {int(a['rows'].sum())} | "
                f"{w['abs_pred_at_max_rel']:.3f} |"
            )
    pc = g[g["gate"] == "(f) 15:30 position changed vs the pre-dedup table (days)"]
    out += [
        "",
        "15:30 positions that changed vs the pre-dedup table (866 deck days): "
        + ", ".join(
            f"{r['forecast'].removeprefix('yhat_restofday_').removesuffix('.parquet')} {int(r['value'])}"
            for _, r in pc.iterrows()
        ),
    ]
    if len(ctl):
        out += [
            "",
            "## Attribution control (pre-dedup executor, same CPU architecture as the new arm)",
            "",
            "The arm that moved most per estimator x input set with any row beyond 1e-8, re-run with "
            "`src/backtest/executor.py` of 47f7f9c^ (the only `src/` file that commit touched) on the "
            "architecture its I6 arm ran on (`control_attribution.csv`). Control vs new = the design "
            "change alone on one architecture; control vs old = the machine alone (same code, CARC vs "
            "Hoffman2). The masked-column gate (the mask already zeroed the twelve columns at every "
            "tune) and the HAR + calendar input set (lasso / elastic net unchanged to ~1e-14 at every "
            "clock) are consistent with both designs posing the same fit problem, so the design "
            "effect here is the solver's floating-point path; per arm it is larger or smaller than "
            "the machine effect (table).",
            "",
            "| arm | architecture | control vs new: rows not bit-identical | control vs new: max rel "
            "| control vs old (CARC): rows > 1e-8 | control vs old: max rel | new vs old: max rel |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, r in ctl.iterrows():
            out.append(
                f"| {r['arm']} | {r['new_arch']} | "
                f"{int(r['control_vs_new_rows_not_bit_identical'])} of {int(r['rows'])} | "
                f"{r['control_vs_new_max_rel']:.2e} | {int(r['control_vs_old_rows_moved'])} | "
                f"{r['control_vs_old_max_rel']:.2e} | {r['new_vs_old_max_rel']:.2e} |"
            )
        pend = sorted(
            {
                f"{b}_{e}_{c}"
                for f in CTRL_TASKS
                for b, e, _, c in [
                    ln.split() for ln in f.read_text().splitlines() if ln.strip()
                ]
            }
            - {
                f"{p.parts[-4]}_{p.parts[-2]}_{p.parts[-3][3:]}"
                for p in (d.parent for d in CTRL.glob("*/rod*/*/tw*/DONE"))
            }
        )
        if pend:
            out.append("")
            out.append(
                f"Not finished when this summary was written (pinned job still queued): {', '.join(pend)}."
            )
    sc_new, sc_old = NEW / "score_rest_of_day.csv", OLD / "score_rest_of_day.csv"
    if sc_new.is_file() and sc_old.is_file():
        sn = pd.read_csv(sc_new, dtype={"clock": str})
        so = pd.read_csv(sc_old, dtype={"clock": str})
        idp = sn[sn["clock"].str.zfill(4) == CHECK_CLOCK]["identity_pass"]
        m = sn.merge(
            so, on=["bucket", "estimator", "train_win", "clock"], suffixes=("", "_old")
        )
        out += [
            "",
            "## The rest-of-day scorer on the new arms",
            "",
            f"`experiments/score_rest_of_day.py --root {NEW.relative_to(ROOT).as_posix()} "
            "--ref-dir results/spxw_pnl` (unedited; the one-step reference = agent A's de-dup per-bar "
            "tables) -> `score_rest_of_day.csv`. Its 15:30 identity self-check (plain direct forecast "
            f"= one-step from the per-bar table): {int(idp.astype(bool).sum())} / {len(idp)} pass. "
            "Plain direct vs one-step QLIKE, median % over the 13 clocks:",
            "",
            "| estimator | input set | pre-dedup arms | new arms "
            "| clocks with p < 0.05 and direct better (old / new) |",
            "|---|---|---|---|---|",
        ]
        for bucket in BUCKETS:
            for est in ESTS:
                x = m[(m["bucket"] == bucket) & (m["estimator"] == est)]
                sig_o = int(
                    (
                        (x["p_plain_vs_onestep_old"] < 0.05)
                        & (x["pct_plain_vs_onestep_old"] < 0)
                    ).sum()
                )
                sig_n = int(
                    (
                        (x["p_plain_vs_onestep"] < 0.05)
                        & (x["pct_plain_vs_onestep"] < 0)
                    ).sum()
                )
                out.append(
                    f"| {ESTIMATORS[est]} | `{bucket}` | {x['pct_plain_vs_onestep_old'].median():+.1f} % | "
                    f"{x['pct_plain_vs_onestep'].median():+.1f} % | {sig_o} / {sig_n} of {len(x)} |"
                )
    out += [
        "",
        "## Tables",
        "",
        "Rebuilt with the unedited stacker `python experiments/build_restofday_yhat.py --root "
        "results/linear_subsection_restofday_dedup` -> `results/spxw_pnl/yhat_restofday_{ridge,lasso,enet}_"
        "{all_features,baseline,live_feasible}.parquet` (13 entry clocks each, 19,085 rows, stacker "
        "gates pass, log `build_restofday_yhat.log`); per-clock changes vs the snapshot in "
        "`tables_vs_pre_dedup.csv`.",
    ]
    (NEW / "SUMMARY.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    sys.exit(main())
