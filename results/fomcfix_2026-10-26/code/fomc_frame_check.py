"""Gate G1 of results/fomcfix_2026-10-26/RULE.txt (read-only, any machine): the two FOMC patches on the frame.

(a) load_raw_data with FOMC_SCHEDULE_2020_FIX on vs off: every column equal (NaN = NaN) except fomc_until_inv and
    fomc_since_inv; fomc_until_inv differs exactly on the bars 2020-01-29 15:00 .. 2020-03-13 20:00 (every bar after the
    2020-01-29 statement that ends before the 2020-03-15 17:00 announcement), fomc_since_inv exactly on 2020-03-15 18:30
    .. 2020-04-29 14:00; new values as the rule says (distance in bars of the frame to 2020-03-18 14:30, and from the
    2020-03-15 18:30 reopen).
(b) load_and_transform (the final design's arguments) on the FOMC series only, flags (patch 1, patch 2) in {0, 1}^2:
    the transformed columns adj_fomc_until_inv / adj_fomc_since_inv, every other column equal; patch 2 alone changes
    them on the share of regular-hours bars in 2014-01-02 .. 2024-04-30 reported (the clip's binding share); patch 1
    alone changes them only on rows from 2020-01-29 15:00 (winsor bounds: up to 240 bars after the last changed row).
usage: python fomc_frame_check.py <repo> <out json>
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
os.chdir(REPO)
sys.path.insert(0, str(REPO))
import src.backtest.executor as EX  # noqa: E402
import src.data.loading as LD  # noqa: E402
import src.features.transforms.target as TG  # noqa: E402

rec: dict = {}


def raw(fix: bool) -> pd.DataFrame:
    LD.FOMC_SCHEDULE_2020_FIX = fix
    return LD.load_raw_data("data", allow_missing=True, drop_dead_session=False)


def eq(a: pd.Series, b: pd.Series) -> np.ndarray:
    x, y = a.to_numpy(), b.to_numpy()
    try:
        return (x == y) | (pd.isna(x) & pd.isna(y))
    except TypeError:
        return np.asarray(a.astype(str) == b.astype(str))


# (a) the raw frame on the calendar-open grid (before the dead-session drop: the grid the distances are counted on)
off, on = raw(False), raw(True)
assert list(off.columns) == list(on.columns) and len(off) == len(on)
t = pd.to_datetime(off["t"])
assert t.equals(pd.to_datetime(on["t"]))
diff_cols = [c for c in off.columns if not eq(off[c], on[c]).all()]
assert sorted(diff_cols) == ["fomc_since_inv", "fomc_until_inv"], diff_cols
want = {
    "fomc_until_inv": ("2020-01-29 15:00", "2020-03-13 20:00"),
    "fomc_since_inv": ("2020-03-15 18:30", "2020-04-29 14:00"),
}
tv = t.to_numpy()
i_sched = int(np.flatnonzero(tv == np.datetime64("2020-03-18T14:30"))[0])
i_reopen = int(np.flatnonzero(tv == np.datetime64("2020-03-15T18:30"))[0])
assert tv[i_reopen - 1] < np.datetime64("2020-03-15T17:00"), tv[i_reopen - 1]
for c, (lo, hi) in want.items():
    d = ~eq(off[c], on[c])
    rows = np.flatnonzero(d)
    exp = np.flatnonzero((tv >= np.datetime64(pd.Timestamp(lo))) & (tv <= np.datetime64(pd.Timestamp(hi))))
    assert np.array_equal(rows, exp), (c, str(t.iloc[rows[0]]), str(t.iloc[rows[-1]]), len(rows), len(exp))
    ref = i_sched - rows if c == "fomc_until_inv" else rows - i_reopen
    assert np.array_equal(on[c].to_numpy()[rows], 1.0 / (1.0 + ref.astype(float))), c
    rec[f"raw_{c}"] = dict(
        first=str(t.iloc[rows[0]]),
        last=str(t.iloc[rows[-1]]),
        n_rows=len(rows),
        old_first=float(off[c].iloc[rows[0]]),
        new_first=float(on[c].iloc[rows[0]]),
        old_last=float(off[c].iloc[rows[-1]]),
        new_last=float(on[c].iloc[rows[-1]]),
    )
print("G1(a) raw frame:", json.dumps(rec, indent=1), flush=True)
del off, on

# (b) the transformed frame (the final design's load_and_transform arguments), FOMC series only
EXOG = [c for c in LD.FOMC_FEATURES if c in LD.ALL_FEATURES]
assert EXOG == ["fomc_day", "fomc_until_inv", "fomc_since_inv"], EXOG
fr = {}
for p1 in (0, 1):
    for p2 in (0, 1):
        LD.FOMC_SCHEDULE_2020_FIX, TG.FOMC_PROXIMITY_NO_WINSOR = bool(p1), bool(p2)
        df, cols = EX.load_and_transform(
            "data",
            EXOG,
            target_use_diurnal=True,
            target_winsor_window=240,
            dropna_with_exog=False,
            overnight_fill=True,
            impute_indicate=True,
            diurnal_mode="divide",
        )
        fr[(p1, p2)] = df
LD.FOMC_SCHEDULE_2020_FIX, TG.FOMC_PROXIMITY_NO_WINSOR = True, True
base = fr[(0, 0)]
tt = pd.to_datetime(base["t"])
mins = (tt.dt.hour * 60 + tt.dt.minute).to_numpy()
ev = (
    (mins >= 600)
    & (mins <= 960)
    & (tt >= "2014-01-02").to_numpy()
    & (tt < "2024-05-01").to_numpy()
    & ~tt.dt.normalize().isin(pd.to_datetime(list(LD.NYSE_EARLY_CLOSE_DATES))).to_numpy()
)
for k, df in fr.items():
    assert list(df.columns) == list(base.columns) and pd.to_datetime(df["t"]).equals(tt), k
    dc = [c for c in base.columns if not eq(base[c], df[c]).all()]
    allowed = {"fomc_until_inv", "fomc_since_inv", "adj_fomc_until_inv", "adj_fomc_since_inv"}
    assert set(dc) <= allowed, (k, dc)
    r: dict = {"differing_columns": dc}
    for c in ("adj_fomc_until_inv", "adj_fomc_since_inv"):
        d = ~eq(base[c], df[c])
        rows = np.flatnonzero(d)
        r[c] = dict(
            n_rows=int(d.sum()),
            share_rth_2014_2024=float(d[ev].mean()),
            first=str(tt.iloc[rows[0]]) if rows.size else None,
            last=str(tt.iloc[rows[-1]]) if rows.size else None,
        )
    rec[f"transformed_p1{k[0]}_p2{k[1]}_vs_p10_p20"] = r
    if k == (1, 0):
        assert all(
            r[c]["first"] is None or r[c]["first"] >= "2020-01-29 15:00"
            for c in ("adj_fomc_until_inv", "adj_fomc_since_inv")
        ), r
print("G1(b) transformed:", json.dumps({k: v for k, v in rec.items() if k.startswith("transformed")}, indent=1))
OUT.write_text(json.dumps(rec, indent=1))
