"""Stack the intraday-sequence LSTM forecasts into yhat tables (the per-bar LSTM tables' layout).

The twin of experiments/build_subsection_lstm_yhat.py for specs/causal_tune_lstm_intraday.py
(merged arms from experiments/reduce_lstm_intraday_chunks.py):

  <root>/<bucket>/bar1600/lstm_intraday/tw2000/causal_tune_lstm_intraday/lstm_intraday/<bucket>/
      results_bar1600.csv, results_qsel_bar1600.csv

Each is written in the notebooks' format -- t (UTC), yhat (the adjusted-scale forecast
pred_adj), baseline (the profile B), rv_raw -- with the linear stacker's own read_arm and
with_production_rv (same gates: B agrees with the production table to 1e-9, every stamp is in
it; the production rv_raw is carried).  Nothing of the spec's look-ahead pred_raw is carried.
THE TABLES HOLD THE 16:00 BAR ONLY (the forecast issued at 15:30).

Tables (--out, default results/spxw_pnl/):
  yhat_lstm_intraday_<bucket>.parquet        MSE-selected configuration (the arm of record)
  yhat_lstm_intraday_qsel_<bucket>.parquet   QLIKE-selected configuration

Run:  python experiments/build_subsection_lstm_intraday_yhat.py
          [--root results/linear_subsection_lstm_intraday] [--out results/spxw_pnl]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "notebooks", ROOT / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_subsection_yhat as bsy  # noqa: E402

LSTMI_ROOT = ROOT / "results" / "linear_subsection_lstm_intraday"
SPEC_DIR, MODEL, SEG, TW = "causal_tune_lstm_intraday", "lstm_intraday", "bar1600", 2000
BUCKETS = ("all_features", "baseline")  # live_feasible commented out
RULES = {"": "mse", "qsel_": "qlike"}  # results file infix -> selection rule


def arm_path(root: Path, bucket: str, infix: str = "") -> Path | None:
    p = (
        root
        / bucket
        / SEG
        / MODEL
        / f"tw{TW}"
        / SPEC_DIR
        / MODEL
        / bucket
        / f"results_{infix}{SEG}.csv"
    )
    return p if p.is_file() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(LSTMI_ROOT))
    ap.add_argument("--out", default=str(bsy.OUT))
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    root = root if root.is_absolute() else ROOT / root
    out = out if out.is_absolute() else ROOT / out
    written, failed = [], []
    for bucket in BUCKETS:
        for infix in RULES:
            name = f"yhat_lstm_intraday_{infix}{bucket}.parquet"
            path = arm_path(root, bucket, infix)
            if path is None:
                print(f"skip {name}: no merged arm under {root}")
                continue
            try:
                tab = bsy.read_arm(path).sort_values("t").reset_index(drop=True)
                assert not tab["t"].duplicated().any(), name
                tab = bsy.with_production_rv(tab, name)
            except AssertionError as e:
                print(f"GATE FAIL {name}: {e!r} -- not written")
                failed.append(name)
                continue
            tab.to_parquet(out / name, index=False)
            et = tab["t"].dt.tz_convert("America/New_York")
            print(
                f"wrote {name}: {len(tab):,} rows (bar ends {sorted(set(et.dt.strftime('%H:%M')))}), "
                f"{et.min().date()} .. {et.max().date()}"
            )
            written.append(name)
    print(f"{len(written)} tables written to {out}")
    if failed:
        sys.exit(f"{len(failed)} table(s) failed a gate: {', '.join(failed)}")


if __name__ == "__main__":
    main()
