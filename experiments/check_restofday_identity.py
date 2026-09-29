"""IDENTITY gate of the rest-of-day campaign (specs/causal_tune_rest_of_day.py).

At the 15:30 entry the rest of the day is the 16:00 bar alone, so the direct
15:30 arm must BE the per-bar bar1600 arm:
  (a) target: RV_rem == RV and (adj_RV_rem, baseline_rem) == (adj_RV, baseline)
      on every 16:00 row, bit for bit (from the arm's targets sidecar);
  (b) forecast: the direct ridge / live_feasible arm's pred_adj reproduces the
      stored per-bar forecast (``yhat`` of the 16:00 rows of
      yhat_sub_ridge_live_feasible.parquet) on every common row, max relative
      difference <= IDENTITY_REL (the bound build_subsection_yhat.py gates the
      per-bar tables with; the per-bar arm reproduced CARC to 7e-12), and the
      stored table has no 16:00 row the direct arm lacks.
Exit status 0 iff both hold; the numbers are printed either way.

Run:  python experiments/check_restofday_identity.py --root <results root>
          [--ref results/spxw_pnl/yhat_sub_ridge_live_feasible.parquet]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IDENTITY_REL = 1e-9  # = build_subsection_yhat.GATE_REL
BUCKET, EST, CLOCK, SEG, TW = "live_feasible", "ridge", "1530", "bar1600", 2000


def arm_files(root: Path) -> tuple[Path, Path]:
    cands = [
        root / BUCKET / f"rod{CLOCK}" / EST / f"tw{TW}",  # campaign layout
        root,  # a bare HPC_RESULT_DIR
    ]
    for base in cands:
        d = base / "causal_tune_rest_of_day" / EST / BUCKET
        r, t = d / f"results_{SEG}.csv", d / f"targets_{SEG}.csv"
        if r.exists() and t.exists():
            return r, t
    raise FileNotFoundError(f"no {EST}/{BUCKET} {SEG} arm under {root}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ref", default="results/spxw_pnl/yhat_sub_ridge_live_feasible.parquet")
    a = ap.parse_args()
    res_f, tgt_f = arm_files(Path(a.root))
    ok = True

    tg = pd.read_csv(tgt_f, parse_dates=["t"])
    same_rv = bool((tg["RV_rem"] == tg["RV"]).all())
    same_adj = bool((tg["adj_RV_rem"] == tg["adj_RV"]).all())
    same_b = bool((tg["baseline_rem"] == tg["baseline"]).all())
    print(f"(a) target on {len(tg):,} 16:00 rows: RV_rem==RV {same_rv}, "
          f"adj_RV_rem==adj_RV {same_adj}, baseline_rem==baseline {same_b} (bit for bit)")
    ok &= same_rv and same_adj and same_b

    r = pd.read_csv(res_f, parse_dates=["date"]).set_index("date")["pred_adj"]
    ref = pd.read_parquet(a.ref)
    et = pd.DatetimeIndex(ref["t"]).tz_convert("America/New_York").tz_localize(None)
    ref = pd.Series(ref["yhat"].to_numpy(float), index=et)
    ref = ref[(ref.index.hour == 16) & (ref.index.minute == 0)]
    common = r.index.intersection(ref.index)
    rel = (r.loc[common] / ref.loc[common] - 1.0).abs()
    only_ref = ref.index.difference(r.index)
    only_dir = r.index.difference(ref.index)
    print(f"(b) forecast: direct arm {len(r):,} rows ({r.index.min().date()} .. {r.index.max().date()}), "
          f"stored per-bar 16:00 rows {len(ref):,} ({ref.index.min().date()} .. {ref.index.max().date()}); "
          f"common {len(common):,}; only stored {len(only_ref)}, only direct {len(only_dir)}")
    print(f"    max relative difference on the common rows {rel.max():.3e} "
          f"(median {rel.median():.3e}); bound {IDENTITY_REL:.0e}")
    if len(only_ref):
        print(f"    stored rows the direct arm lacks: {list(only_ref[:5].date)} ...")
    ok &= bool(len(common) > 0 and np.isfinite(rel.max()) and rel.max() <= IDENTITY_REL
               and len(only_ref) == 0)
    print("IDENTITY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
