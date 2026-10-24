"""The pooled linear runner (experiments/pool_linear_pretune.py) for the three-way training-sample comparison
(results/linear_train3way_2014start_w3000/RULE.txt), run unchanged with three additions made from outside the module:

  * --sample close: segment bar1600 of src/backtest/segmentation.py (the 15:30-16:00 ET half hour, one row a session),
    lag scope global as rth13 (the HAR components and trailing averages on the full 48-interval grid), 1 row a day;
  * --sample close_r: the 16:00 rows of a rth13 design cache (experiments/linear_train3way.py derive_close), for the
    design-cache modes only (pretune, walk with --design); 1 row a day;
  * mode design also writes dates_all.csv, the date label of every design row (the rows _backtest_and_save hands to
    the walk, rows 0 .. W - 1 included), captured from the output of src.backtest.executor.apply_horizon_shift; every
    array the design stores is unchanged.
  * --window-rows N (2026-10-23 addendum, equal training rows): the rolling window is N rows instead of
    --window-days x rows a day, for the design-cache modes only (pretune, walk with --design: a cache derived by
    experiments/linear_train3way.py derive_window, whose config records window = N).  The runner's W is the only
    window those modes use (cached_backtest, the walk's MultiStageBacktest, the pretune's fit block, config_dict);
    the spec's TRAIN_WIN (days) is read only by the design path.

usage: python experiments/linear_train3way_runner.py <the runner's arguments>
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pool_linear_pretune as P  # noqa: E402

P.SEGMENT_OF["close"] = "bar1600"
P.PPD_OF["close"] = 1
P.SEGMENT_OF["close_r"] = "bar1600"
P.PPD_OF["close_r"] = 1


def _arg(name: str, default: str = "") -> str:
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default


def main() -> None:
    mode, sample = sys.argv[1], _arg("--sample", "48")
    if sample == "close_r":
        assert mode in ("pretune", "walk") and _arg("--design"), "close_r: a derived design cache, pretune / walk only"
    rows = _arg("--window-rows")
    if rows:  # 2026-10-23 addendum: a row-count window on a derived design cache
        assert mode in ("pretune", "walk") and _arg("--design"), "--window-rows: a derived design cache only"
        i = sys.argv.index("--window-rows")
        del sys.argv[i : i + 2]
        cfg = P.configure

        def configure(a) -> None:
            cfg(a)
            P.W = int(rows)

        P.configure = configure
    if mode != "design":
        P.main()
        return
    import numpy as np
    import pandas as pd

    import src.backtest.executor as EX

    orig = EX.apply_horizon_shift
    cap: dict = {}

    def spy(X, y, dates, baselines, horizon):
        r = orig(X, y, dates, baselines, horizon)
        cap.setdefault("dates", []).append(pd.Series(r[2]).reset_index(drop=True))
        return r

    EX.apply_horizon_shift = spy
    try:
        P.main()
    finally:
        EX.apply_horizon_shift = orig
    assert len(cap["dates"]) == 1, len(cap["dates"])
    out = Path(_arg("--out")).resolve()
    d = pd.to_datetime(cap["dates"][0]).dt.strftime("%Y-%m-%d %H:%M:%S")
    n = int(np.load(out / "X.npy", mmap_mode="r").shape[0])
    oos = pd.read_csv(out / "dates_oos.csv", dtype=str)["date"]
    assert len(d) == n and d.iloc[n - len(oos) :].reset_index(drop=True).equals(oos), (len(d), n, len(oos))
    pd.DataFrame({"date": d}).to_csv(out / "dates_all.csv", index=False)
    print(f"dates_all.csv: {n} rows, {d.iloc[0]} .. {d.iloc[-1]}", flush=True)


if __name__ == "__main__":
    main()
