"""Check the three pooled loaders under the switch in force (writeup/final_run.py), 2026-10-23.

Loads every arm through pooled_linear_13, pooled_trees_13 and pooled_mingru_13 (their gates run on load) and asserts
the cross-run invariants the generators rely on: the trees', the forest's and the MinGRU's bars are the linear walk's
bars from the first evaluation day, the 16:00 rows carry the same realized variance, and the metas agree on the window
and the first day.  Prints one line a check.  No output file.

Run:  python writeup/tests/check_final_loaders.py                      (the printed runs)
      REHAUL_FINAL_RUN=1 [REHAUL_ALLOW_STANDIN=1 REHAUL_TREES_RUN=...] python writeup/tests/check_final_loaders.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import final_run as fr  # noqa: E402
import pooled_linear_13 as pl13  # noqa: E402
import pooled_mingru_13 as pm13  # noqa: E402
import pooled_trees_13 as pt13  # noqa: E402


def main() -> None:
    print(fr.tag())
    lm, tm = pl13.meta(), pt13.meta()
    assert (lm["window_days"], lm["bars_day"], lm["span"][0]) == (
        tm["window_days"],
        tm["bars_day"],
        tm["span"][0],
    )
    print(
        f"ok  linear and trees: window {lm['window_days']:,} days ({lm['window_bars']:,} bars), first day "
        f"{lm['span'][0].date()}"
    )
    lin = pl13._frame()
    first = lm["span"][0]
    bars = {
        k: set(pd.to_datetime(g["date"]))
        for k, g in lin.groupby(["model", "predictor_set"])
    }
    b0 = bars[("ols", "baseline")]
    assert all(v == b0 for v in bars.values())
    b_eval = {t for t in b0 if t >= first}
    print(
        f"ok  linear: {len(bars)} arms x {len(b0):,} rows from {min(b0)}, {len(b_eval):,} from the first day"
    )
    tr = pt13._frame()
    for k, g in tr.groupby(["model", "predictor_set"]):
        assert set(pd.to_datetime(g["date"])) == b_eval, k
    print(
        f"ok  trees: {sorted(set(tr['model']))} x {sorted(set(tr['predictor_set']))}, {pt13.N_BARS_ARM:,} bars "
        "an arm = the linear bars from the first day"
    )
    mg = pm13._frame()
    mb = {s: set(pd.to_datetime(g["date"])) for s, g in mg.groupby("predictor_set")}
    if fr.FINAL:
        assert all(v == b_eval for v in mb.values())
        print(
            f"ok  MinGRU: {pm13.N_BARS_ARM:,} bars a set = the trees' bars (window {pm13.meta()['window_days']:,})"
        )
    else:
        assert all(v == b0 for v in mb.values())
        print(
            f"ok  MinGRU (printed): {pm13.N_BARS_ARM:,} bars a set = the linear walk's rows"
        )
    ref = pl13.at16("ols", "baseline")
    for e in pt13.EST:
        for s in pt13.SETS:
            a = pt13.at16(e, s)
            assert np.allclose(
                a["rv"].to_numpy(),
                ref["rv"].reindex(a.index).to_numpy(),
                rtol=1e-12,
                atol=0,
            )
    for s in pm13.SETS:
        a = pm13.at16(s)
        assert np.allclose(
            a["rv"].to_numpy(),
            ref["rv"].reindex(a.index).to_numpy(),
            rtol=1e-12,
            atol=0,
        )
    print(
        "ok  16:00 rows: every tree, forest and MinGRU arm carries the linear OLS row's realized variance"
    )


if __name__ == "__main__":
    main()
