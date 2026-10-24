"""Check the three pooled loaders under the switch in force (writeup/final_run.py), 2026-10-23.

Loads every arm through pooled_linear_13, pooled_trees_13 and pooled_mingru_13 (their gates run on load) and asserts
the cross-run invariants the generators rely on: the trees', the forest's and the MinGRU's bars are the linear walk's
bars from the first evaluation day, the 16:00 rows carry the same realized variance, and the metas agree on the window
and the first day.  Prints one line a check.  No output file.
2026-10-23 (final_run.py PENDING): a pending arm (the forest, the MinGRU) is reported "pending" and its checks are
skipped; the check asserts that its loader raises final_run.Pending (no printed run stands in) and that no finished
arm is pending.  REHAUL_REQUIRE_ALL=1 makes every arm required (a missing run fails as before).

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
    # 2026-10-23 (final_run.py PENDING): a pending arm is not read; its loader must refuse it (no printed stand-in)
    pend = _check_pending()
    if not pm13.PENDING:
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
    for e in pt13.EST_DONE:
        for s in pt13.SETS:
            a = pt13.at16(e, s)
            assert np.allclose(
                a["rv"].to_numpy(),
                ref["rv"].reindex(a.index).to_numpy(),
                rtol=1e-12,
                atol=0,
            )
    for s in pm13.SETS if not pm13.PENDING else ():
        a = pm13.at16(s)
        assert np.allclose(
            a["rv"].to_numpy(),
            ref["rv"].reindex(a.index).to_numpy(),
            rtol=1e-12,
            atol=0,
        )
    done = [
        n
        for n, p in (
            ("tree", False),
            ("forest", fr.RF_PENDING),
            ("MinGRU", pm13.PENDING),
        )
        if not p
    ]
    print(
        f"ok  16:00 rows: every finished arm ({', '.join(done)}) carries the linear OLS row's realized variance"
        + (f"; pending: {', '.join(pend)}" if pend else "")
    )


def _check_pending() -> list[str]:
    """The pending arms (final_run.py PENDING): only under the switch, only without forecasts, never with
    REHAUL_REQUIRE_ALL=1; each loader raises final_run.Pending on a read.  Returns their names."""
    out = []
    assert set(pt13.PENDING) <= {"rf"} and pt13.EST_DONE == tuple(
        e for e in pt13.EST if e not in pt13.PENDING
    ), (pt13.PENDING, pt13.EST_DONE)
    assert {"lgbm", "xgb"} <= set(pt13.EST_DONE), pt13.EST_DONE
    if not fr.FINAL or fr.REQUIRE_ALL:
        assert not pt13.PENDING and not pm13.PENDING, (
            "an arm is pending with the switch off or REHAUL_REQUIRE_ALL=1"
        )
    if pt13.PENDING:
        assert not (fr.RF_RUN / "forecasts.parquet").is_file(), (
            "the forest has forecasts but reads as pending"
        )
        assert "rf" not in set(pt13._frame()["model"]), (
            "the pending forest has rows in the trees' frame"
        )
        for s in pt13.SETS:
            try:
                pt13.at16("rf", s)
            except fr.Pending:
                continue
            raise AssertionError(
                f"pt13.at16('rf', {s!r}) returned rows for the pending forest"
            )
        print(
            f"ok  forest: PENDING (no {fr.rel(fr.RF_RUN / 'forecasts.parquet')}); pt13.at16 raises Pending"
        )
        out.append("forest")
    if pm13.PENDING:
        assert not (fr.MINGRU_RUN / "forecasts.parquet").is_file(), (
            "the MinGRU has forecasts but reads as pending"
        )
        for fn in (pm13.meta, pm13._frame, lambda: pm13.at16("baseline")):
            try:
                fn()
            except fr.Pending:
                continue
            raise AssertionError("the pending MinGRU loader returned rows")
        print(
            f"ok  MinGRU: PENDING (no {fr.rel(fr.MINGRU_RUN / 'forecasts.parquet')}); pm13 raises Pending"
        )
        out.append("MinGRU")
    return out


if __name__ == "__main__":
    main()
