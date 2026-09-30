"""I1: rebuild the canonical per-bar forecast tables from the de-duplicated re-runs (thin wrapper).

The stacking builders are NOT edited; they are imported and pointed at the new roots:

  linear   experiments/build_subsection_yhat.py   ARMS  -> results/linear_subsection_dedup/arms_hoffman2
           -> results/spxw_pnl/yhat_sub_{ridge,lasso,enet}_{baseline,live_feasible,all_features}.parquet
           (its pooled twins yhat_pool_* are NOT rewritten: the new root holds no pooled arm -- the
           pooled 48-bar path keeps the session-edge columns and was not re-run -- so the builder
           prints "skip yhat_pool_*" and leaves those tables as they are)
  VIX-only experiments/build_vixonly_yhat.py      LS -> results/linear_subsection_dedup, INCUMBENT ->
           its arms_carc/live_feasible (the builder's own gate then asserts that those arms equal the
           live_feasible tables just rebuilt, row for row)
           -> yhat_sub_{ridge,lasso,enet}_{vix_only,live_vix_only,free_vix_only}.parquet
  LSTM     experiments/build_subsection_lstm_yhat.py --root results/linear_subsection_lstm_dedup
           -> yhat_lstm_{bucket}.parquet and yhat_lstm_qsel_{bucket}.parquet (16:00 only)

FLATTEN (first, --linear): the cluster runs write the spec's nested layout
  results/linear_subsection_dedup/<bucket>/<bar>/<est>/tw2000/causal_tune_linear/<est>/<bucket>/results_<bar>.csv
and build_subsection_yhat reads the flat layout the older pull made,
  results/linear_subsection_dedup/arms_hoffman2/<bucket>/<est>/tw2000/results_<bar>.csv
(and build_vixonly_yhat / compare_mfiv_harlag the same files under arms_carc/live_feasible); the
nested tables are copied there byte for byte, 13 bars x 3 estimators x 3 buckets.

GATE (tables_vs_pre_dedup.csv): every rebuilt table against its pre-change copy in
results/spxw_pnl/pre_dedup_2026-09-29/: the same stamps, the same baseline B and rv_raw (bitwise),
and the change of the forecast yhat -- max and median |relative change| over all rows and at the
16:00 rows.  Recorded, not judged (the linear gates proper are experiments/linear_dedup_gates.py).

Run:  python experiments/rebuild_yhat_dedup.py [--linear] [--lstm]   (default: both)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "experiments", ROOT / "notebooks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_subsection_lstm_yhat as bly  # noqa: E402
import build_subsection_yhat as bsy  # noqa: E402
import build_vixonly_yhat as bvy  # noqa: E402

NEW = ROOT / "results" / "linear_subsection_dedup"
LSTM_NEW = ROOT / "results" / "linear_subsection_lstm_dedup"
SPXW = ROOT / "results" / "spxw_pnl"
PRE = SPXW / "pre_dedup_2026-09-29"
MAIN_BUCKETS = ("baseline", "live_feasible", "all_features")
ESTS = {"ridge": "ridge", "reclasso": "lasso", "reclasticnet": "enet"}
VIX_TABLE_BUCKETS = ("vix_only", "live_vix_only", "free_vix_only")


def flatten() -> int:
    n = 0
    for bucket in MAIN_BUCKETS:
        for est in ESTS:
            for bar in bsy.BARS:
                src = (
                    NEW
                    / bucket
                    / bar
                    / est
                    / f"tw{bsy.TW}"
                    / "causal_tune_linear"
                    / est
                    / bucket
                    / f"results_{bar}.csv"
                )
                if not src.is_file():
                    print(f"flatten: missing {src.relative_to(ROOT)}")
                    continue
                dsts = [NEW / "arms_hoffman2" / bucket / est / f"tw{bsy.TW}"]
                if bucket == "live_feasible":
                    dsts.append(NEW / "arms_carc" / bucket / est / f"tw{bsy.TW}")
                for d in dsts:
                    d.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, d / f"results_{bar}.csv")
                n += 1
    print(f"flatten: {n} one-bar arm tables copied into the flat layout")
    return n


def compare(names: list[str]) -> list[dict]:
    rows = []
    for name in names:
        new_p, old_p = SPXW / name, PRE / name
        if not new_p.is_file() or not old_p.is_file():
            rows.append(
                {"table": name, "ok": False, "why": "missing new or pre-dedup table"}
            )
            continue
        a, b = pd.read_parquet(new_p), pd.read_parquet(old_p)
        same_t = len(a) == len(b) and bool(
            (a["t"].to_numpy() == b["t"].to_numpy()).all()
        )
        r: dict = {
            "table": name,
            "rows": len(a),
            "rows_pre": len(b),
            "same_stamps": same_t,
        }
        if same_t:
            r["baseline_bitwise"] = bool(
                (a["baseline"].to_numpy() == b["baseline"].to_numpy()).all()
            )
            r["rv_raw_bitwise"] = bool(
                np.array_equal(
                    a["rv_raw"].to_numpy(), b["rv_raw"].to_numpy(), equal_nan=True
                )
            )
            rel = np.abs(a["yhat"].to_numpy() / b["yhat"].to_numpy() - 1.0)
            et = pd.to_datetime(a["t"]).dt.tz_convert("America/New_York")
            at16 = ((et.dt.hour == 16) & (et.dt.minute == 0)).to_numpy()
            r.update(
                yhat_max_rel_change=float(np.nanmax(rel)),
                yhat_median_rel_change=float(np.nanmedian(rel)),
                n_1600=int(at16.sum()),
                yhat_1600_max_rel_change=float(np.nanmax(rel[at16])),
                yhat_1600_median_rel_change=float(np.nanmedian(rel[at16])),
            )
            r["ok"] = r["baseline_bitwise"] and r["rv_raw_bitwise"]
        else:
            r["ok"] = False
        rows.append(r)
        print(
            f"{name}: stamps equal {same_t}"
            + (
                f", 16:00 yhat max |rel change| {r['yhat_1600_max_rel_change']:.2e} "
                f"(median {r['yhat_1600_median_rel_change']:.2e}), all bars max {r['yhat_max_rel_change']:.2e}"
                if same_t
                else ""
            )
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--lstm", action="store_true")
    a = ap.parse_args()
    do_lin = a.linear or not (a.linear or a.lstm)
    do_lstm = a.lstm or not (a.linear or a.lstm)
    names: list[str] = []
    if do_lin:
        flatten()
        bsy.ARMS = NEW / "arms_hoffman2"
        bsy.main()
        bvy.LS = NEW
        bvy.INCUMBENT = NEW / "arms_carc" / "live_feasible"
        bvy.main()
        names += [
            f"yhat_sub_{s}_{b}.parquet"
            for b in (*MAIN_BUCKETS, *VIX_TABLE_BUCKETS)
            for s in ESTS.values()
        ]
    if do_lstm:
        argv = sys.argv
        sys.argv = [
            "build_subsection_lstm_yhat.py",
            "--root",
            str(LSTM_NEW),
            "--out",
            str(SPXW),
        ]
        try:
            bly.main()
        finally:
            sys.argv = argv
        names += [
            f"yhat_lstm_{infix}{b}.parquet" for b in bly.BUCKETS for infix in bly.RULES
        ]
    out = NEW / "tables_vs_pre_dedup.csv"
    new = pd.DataFrame(compare(names))
    if out.is_file():  # keep the rows of the part not rebuilt this time
        old = pd.read_csv(out)
        new = pd.concat([old[~old["table"].isin(new["table"])], new], ignore_index=True)
    new.sort_values("table").to_csv(out, index=False)
    print(f"wrote {out.relative_to(ROOT)} ({len(new)} tables)")


if __name__ == "__main__":
    main()
