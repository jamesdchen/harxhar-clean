"""Stack the per-bar LSTM forecasts into yhat tables the 0DTE notebooks and the master table read.

The LSTM twin of ``experiments/build_subsection_tree_yhat.py``.  The LSTM
campaign (``specs/causal_tune_lstm.py``: one LSTM on the per-bar design of the
linear and tree arms, SEGMENT = bar1600 only, TRAIN_WIN = 2000 sessions) writes
one merged arm per bucket (``experiments/reduce_lstm_chunks.py``):

  <root>/<bucket>/bar1600/lstm/tw<TW>/causal_tune_lstm/lstm/<bucket>/results_bar1600.csv
                                                                  results_qsel_bar1600.csv

Each is written in the notebooks' format -- ``t`` (UTC), ``yhat`` (the
adjusted-scale forecast pred_adj), ``baseline`` (the profile B) and ``rv_raw``
-- with the linear stacker's own ``read_arm`` and ``with_production_rv`` (same
gates: B agrees with the production table to 1e-9, every stamp is in it; the
production rv_raw is carried).  Nothing of the spec's look-ahead ``pred_raw`` is
carried.  THE TABLES HOLD THE 16:00 BAR ONLY (the forecast issued at 15:30): a
scorer that recalibrates on all 13 session bars cannot use them; the research
scorer's 16:00-bar recalibration can.

Tables (``--out``, default results/spxw_pnl/):
  yhat_lstm_<bucket>.parquet        MSE-selected configuration (the arm of record)
  yhat_lstm_qsel_<bucket>.parquet   QLIKE-selected configuration

Run:  python experiments/build_subsection_lstm_yhat.py
          [--root results/linear_subsection_lstm] [--out results/spxw_pnl]
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

LSTM_ROOT = ROOT / "results" / "linear_subsection_lstm"
SPEC_DIR = "causal_tune_lstm"
MODEL = "lstm"
SEG = "bar1600"
BUCKETS = bsy.BUCKETS
TW = bsy.TW
RULES = {"": "mse", "qsel_": "qlike"}  # results file infix -> selection rule


def resolve(p: str | Path) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def lstm_arm_path(
    root: Path, bucket: str, tw: int = TW, infix: str = "", seg: str = SEG
) -> Path | None:
    """The merged arm's results CSV (infix '' = MSE rule, 'qsel_' = QLIKE rule), else None."""
    p = (
        root
        / bucket
        / seg
        / MODEL
        / f"tw{tw}"
        / SPEC_DIR
        / MODEL
        / bucket
        / f"results_{infix}{seg}.csv"
    )
    return p if p.is_file() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(LSTM_ROOT))
    ap.add_argument("--out", default=str(bsy.OUT))
    ap.add_argument("--tw", type=int, default=TW)
    a = ap.parse_args()
    root, out = resolve(a.root), resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    written, failed = [], []
    for bucket in BUCKETS:
        for infix in RULES:
            name = f"yhat_lstm_{infix}{bucket}.parquet"
            path = lstm_arm_path(root, bucket, a.tw, infix)
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
