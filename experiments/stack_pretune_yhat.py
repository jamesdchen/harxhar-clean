"""Stack the pre-tuned per-bar linear forecasts (arm f) into 16:00 yhat tables for the master table.

User decision (2026-10-06): "use the pre-tuned version for all linear models".  In the paper,
every per-bar ridge / lasso / elastic net forecast is arm (f) of results/linear_pretune_2026-10-06/
(experiments/close_linear_pretune.py): the penalty pre-tuned on the sessions before 2020, the HAR +
calendar backbone penalized at 1/100 of the exogenous penalty (on baseline every column is
backbone: one penalty), and retuned at each 250-session block start in the trade period only when
the HAC t of the paired squared-error difference clears the margin.  Its export series:

  linear_pretune/<est>_<bucket>_f_retune_margin_bb    (all_features, lean_vol_vix)
  linear_pretune/<est>_baseline_f_retune_margin        (baseline)

for est in ridge / lasso / enet (results/linear_pretune_2026-10-06/forecasts.parquet: series_id,
row, date = the naive-ET 16:00 stamp, pred_adj = the adjusted-scale forecast).

This script writes, WITHOUT touching the stored one-penalty tables,

  results/spxw_pnl/yhat_subpt_<est>_<bucket>.parquet

with the schema of the other 16:00-only per-bar tables (the tree tables): t (UTC), yhat (the
f-arm pred_adj), baseline (B) and rv_raw, one row per 16:00 session of the stored
results/spxw_pnl/yhat_sub_<est>_<bucket>.parquet, whose t / baseline / rv_raw it carries (that
table's B and rv_raw come from the production join, experiments/build_subsection_yhat.py).

GATES (asserted; results/linear_pretune_2026-10-06/stack_gates.csv):
  * the stacked yhat equals the f-arm export bit for bit on every stamp;
  * the f-arm export's stamps are exactly the stored sub table's 16:00 stamps (so t, B and rv_raw
    are the stored table's on the same stamps);
  * arm (a) of the study (one penalty, causal re-choice: the spec's) reproduces the stored sub
    table's 16:00 yhat within the study's documented tolerance: ridge 1e-8 relative, lasso /
    elastic net 0.65 % (the warm-homotopy floating-point gap the 2026-10-03 study documented).

Run:  python experiments/stack_pretune_yhat.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results" / "linear_pretune_2026-10-06"
SPXW = ROOT / "results" / "spxw_pnl"
EXPORT = STUDY / "forecasts.parquet"
GATES_OUT = STUDY / "stack_gates.csv"
ESTS = ("ridge", "lasso", "enet")
BUCKETS = ("baseline", "all_features", "lean_vol_vix")  # the paper's three input sets
TOL_A = {
    "ridge": 1e-8,
    "lasso": 0.0065,
    "enet": 0.0065,
}  # the study's documented tolerances


def f_series(est: str, bucket: str) -> str:
    """The export series of arm (f): backbone 1/100 except on baseline (all backbone)."""
    # 2026-10-18 (user, new-notes 24): the frozen arms (one pre-sample pick per estimator and predictor set, no re-tune; per E23):
    # c_frozen on the HAR set (baseline), d_frozen_bb on the full / lean sets (standard grid, not the _w arms). Was:
    #     arm = "f_retune_margin" if bucket == "baseline" else "f_retune_margin_bb"
    arm = "c_frozen" if bucket == "baseline" else "d_frozen_bb"
    return f"linear_pretune/{est}_{bucket}_{arm}"


def table_1600(d: pd.DataFrame) -> pd.Series:
    """Boolean mask of a table's 16:00 ET rows (the master table's read_table_1600 rule)."""
    et = pd.to_datetime(d["t"]).dt.tz_convert("America/New_York").dt.tz_localize(None)
    return (et.dt.hour == 16) & (et.dt.minute == 0)


def naive_et(t: pd.Series) -> pd.DatetimeIndex:
    et = pd.to_datetime(t).dt.tz_convert("America/New_York").dt.tz_localize(None)
    return pd.DatetimeIndex(et.to_numpy()).as_unit("ns")


def main() -> None:
    ex = pd.read_parquet(EXPORT)
    gates: list[dict] = []
    for est in ESTS:
        for bucket in BUCKETS:
            sid = f_series(est, bucket)
            f = ex[ex["series_id"] == sid].sort_values("date")
            assert len(f) > 0, sid
            assert f["date"].is_unique and f["pred_adj"].notna().all(), sid
            src = SPXW / f"yhat_sub_{est}_{bucket}.parquet"
            st = pd.read_parquet(src)
            st16 = st[table_1600(st).to_numpy()].copy()
            stamps = naive_et(st16["t"])
            fx = pd.DatetimeIndex(pd.to_datetime(f["date"]).to_numpy()).as_unit("ns")
            same = len(stamps) == len(fx) and bool((stamps.sort_values() == fx).all())
            gates.append(
                {
                    "gate": "f-arm stamps = the stored sub table's 16:00 stamps",
                    "item": f"{est}_{bucket}",
                    "value": f"{len(fx)} / {len(stamps)}",
                    "ok": same,
                }
            )
            assert same, (sid, len(fx), len(stamps))
            st16.index = stamps
            st16 = st16.sort_index()
            out = pd.DataFrame(
                {
                    "t": st16["t"].to_numpy(),
                    "yhat": f.set_index(fx)["pred_adj"].reindex(st16.index).to_numpy(),
                    "baseline": st16["baseline"].to_numpy(),
                    "rv_raw": st16["rv_raw"].to_numpy(),
                }
            )
            assert out["yhat"].notna().all()
            dst = SPXW / f"yhat_subpt_{est}_{bucket}.parquet"
            assert dst.name != src.name
            out.to_parquet(dst, index=False)
            back = pd.read_parquet(dst)
            exact = bool((back["yhat"].to_numpy() == f["pred_adj"].to_numpy()).all())
            gates.append(
                {
                    "gate": "stacked yhat = the f-arm export, bit for bit",
                    "item": f"{est}_{bucket}",
                    "value": float(
                        np.max(
                            np.abs(back["yhat"].to_numpy() - f["pred_adj"].to_numpy())
                        )
                    ),
                    "ok": exact,
                }
            )
            assert exact, sid
            b_same = bool(
                (back["baseline"].to_numpy() == st16["baseline"].to_numpy()).all()
                and (back["t"] == st16["t"].reset_index(drop=True)).all()
            )
            gates.append(
                {
                    "gate": "t and B = the stored sub table's (16:00 rows)",
                    "item": f"{est}_{bucket}",
                    "value": "identical" if b_same else "differ",
                    "ok": b_same,
                }
            )
            assert b_same, sid
            # arm (a), the spec's one penalty, against the stored one-penalty table
            a = ex[ex["series_id"] == f"linear_pretune/{est}_{bucket}_a_causal"]
            a = a.set_index(
                pd.DatetimeIndex(pd.to_datetime(a["date"]).to_numpy()).as_unit("ns")
            )
            j = a["pred_adj"].reindex(st16.index)
            rel = float(np.nanmax(np.abs(j.to_numpy() / st16["yhat"].to_numpy() - 1.0)))
            ok_a = j.notna().all() and rel <= TOL_A[est]
            gates.append(
                {
                    "gate": "arm (a) = the stored one-penalty sub table (max rel; ridge 1e-8, lasso / enet 0.65 %)",
                    "item": f"{est}_{bucket}",
                    "value": rel,
                    "ok": bool(ok_a),
                }
            )
            assert ok_a, (est, bucket, rel)
            print(f"{dst.name}: {len(out)} rows; arm (a) vs stored max rel {rel:.2e}")
    pd.DataFrame(gates).to_csv(GATES_OUT, index=False, lineterminator="\n")
    print(f"{len(gates)} gates, all ok -> {GATES_OUT.relative_to(ROOT)}")
    facts()


def facts() -> None:
    """The facts the methods text quotes about arm (f), read from the study (paper_facts.csv)."""
    import sys

    sys.path.insert(0, str(ROOT / "experiments"))
    import close_linear_pretune as clp

    rows: list[dict] = []

    def add(item: str, value: object, source: str) -> None:
        rows.append({"item": item, "value": value, "source": source})

    import json

    # the pre-2020 16:00 rows of the last-hour design (the study's prep: _work/arrays/meta.json)
    meta = json.loads((clp.ARR / "meta.json").read_text())
    assert len({meta[b]["pre_n"] for b in BUCKETS}) == 1, (
        "the same pre-2020 sessions for every input set"
    )
    dates = pd.to_datetime(meta["all_features"]["pre_dates"])
    pre_n = int(meta["all_features"]["pre_n"])
    assert len(dates) == pre_n
    folds = clp.pre_folds(pre_n)
    v0, v1 = folds[0]["val"][0], folds[-1]["val"][1]
    src_d = "results/linear_pretune_2026-10-06/_work/arrays/meta.json (pre_dates) + close_linear_pretune.pre_folds"
    add("pre-2020 16:00 sessions in the design", pre_n, src_d)
    add("first pre-2020 16:00 session", str(dates[0].date()), src_d)
    add("pre-tune validation: first session", str(dates[v0].date()), src_d)
    add("pre-tune validation: last session", str(dates[v1 - 1].date()), src_d)
    add("pre-tune folds", clp.PRE_FOLDS, "close_linear_pretune.PRE_FOLDS")
    add(
        "pre-tune fold length (sessions)",
        clp.PRE_FOLD_LEN,
        "close_linear_pretune.PRE_FOLD_LEN",
    )
    add("fit window (sessions)", clp.TW, "close_linear_pretune.TW")
    add("embargo (sessions)", clp.EMBARGO, "close_linear_pretune.EMBARGO")
    add("pre-tune criterion", "mean validation MSE on the adjusted scale", "SUMMARY.md")
    add(
        "backbone penalty ratio (HAR + calendar / exogenous)",
        clp.R_BB,
        "close_linear_pretune.R_BB",
    )
    add(
        "retune: block length (sessions)", clp.TUNE_PER, "close_linear_pretune.TUNE_PER"
    )
    add(
        "retune: challengers",
        "the two grid neighbours of the incumbent penalty",
        "SUMMARY.md",
    )
    add(
        "retune: margin rule",
        f"switch only if the HAC t of the paired squared-error difference < -z({clp.MARGIN_ALPHA} / challengers)",
        "close_linear_pretune.MARGIN_ALPHA, SUMMARY.md",
    )
    add(
        "retune: validation",
        f"{clp.RETUNE_FOLDS} folds x {clp.VAL_TAIL} sessions, each with its own {clp.TW}-session fit",
        "close_linear_pretune",
    )
    rl = pd.read_csv(STUDY / "retune_log.csv")
    add(
        "retune dates (block starts)",
        "; ".join(sorted(rl["date"].astype(str).unique())),
        "retune_log.csv",
    )
    pk = pd.read_csv(STUDY / "penalty_picks.csv")
    est_long = {"ridge": "ridge", "lasso": "reclasso", "enet": "reclasticnet"}
    tot_f = tot_n = 0
    for est in ESTS:
        for bucket in BUCKETS:
            r = clp.R_ONE if bucket == "baseline" else clp.R_BB
            p = pk[
                (pk["bucket"] == bucket)
                & (pk["est"] == est_long[est])
                & (pk["r"] == r)
                & (pk["grid_kind"] == "spec")
            ]
            assert len(p) == 1, (est, bucket)
            g = rl[
                (rl["bucket"] == bucket)
                & (rl["est"] == est_long[est])
                & (rl["r"] == r)
                & (rl["grid_kind"] == "spec")
                & (rl["rule"] == "margin")
            ]
            fired = int(g["switched"].astype(bool).sum())
            tot_f += fired
            tot_n += len(g)
            add(
                f"{est} {bucket}: pre-tuned penalty",
                float(p["pick_mse"].iloc[0]),
                "penalty_picks.csv (spec grid)",
            )
            add(
                f"{est} {bucket}: retunes fired",
                f"{fired} of {len(g)}",
                "retune_log.csv (rule margin, spec grid)",
            )
            cat = pd.read_csv(STUDY / "catalogue.csv").set_index("series_id")
            add(
                f"{est} {bucket}: penalty by block",
                cat.loc[f_series(est, bucket), "alphas_by_block"],
                "catalogue.csv",
            )
    add("retunes fired, total", f"{tot_f} of {tot_n}", "retune_log.csv")
    out = STUDY / "paper_facts.csv"
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
    print(f"{len(rows)} facts -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
