"""Re-derive src/data/loading.py NYSE_HOLIDAY_FUTURES_SESSIONS from the sample's own calendar and assert it (RULE.txt
addendum A).  Read-only.

Rule: a date is a holiday futures session when the 48-interval grid (load_raw_data, after the dead-session drop) has a
bar ending 10:00 .. 16:00 on it, no bar at 16:00, it is not an NYSE early close (NYSE_EARLY_CLOSE_DATES), and no
constituent series (the 24 *_ewstock / *_vwstock columns) printed on any bar of the date.  Checks: every such date is a
weekday that experiments/rolling_eval_2026_10_15.nyse_sessions (the NYSE rule calendar) does not list; no rule-calendar
holiday has a 16:00 bar; the derived dates equal the constant; counts in the evaluation span 2014-01-02 .. 2024-04-30.

usage: python holiday_sessions.py <repo> <out csv>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
os.chdir(REPO)
for p in (REPO, REPO / "experiments"):
    sys.path.insert(0, str(p))
import rolling_eval_2026_10_15 as RE  # noqa: E402

from src.data import loading as LD  # noqa: E402

df = LD.load_raw_data("data", allow_missing=True)
t = pd.to_datetime(df["t"])
mins = (t.dt.hour * 60 + t.dt.minute).to_numpy()
day = t.dt.normalize()
rth = (mins >= 600) & (mins <= 960)
early = day.isin(pd.to_datetime(list(LD.NYSE_EARLY_CLOSE_DATES))).to_numpy()
cons = [c for c in df.columns if c.endswith("_ewstock") or c.endswith("_vwstock")]
assert len(cons) == 24, cons
has16 = set(day[mins == 960])
cons_day = pd.Series(df[cons].notna().any(axis=1).to_numpy(), index=day.to_numpy()).groupby(level=0).any()
rows = []
for d, g in pd.Series(np.arange(len(df))[rth & ~early], index=day[rth & ~early].to_numpy()).groupby(level=0):
    if d in has16 or bool(cons_day.get(d, False)):
        continue
    rows.append(dict(date=d, dow=d.day_name(), n_rth_bars=len(g), last_rth_bar=str(t.iloc[g.to_numpy()].max().time())))
h = pd.DataFrame(rows)
nyse = RE.nyse_sessions("1998-01-01", "2024-12-31")
h["nyse_rule_session"] = h["date"].isin(nyse)
assert not h["nyse_rule_session"].any(), h[h["nyse_rule_session"]]
assert (h["date"].dt.dayofweek < 5).all()
wk = pd.bdate_range(t.min().normalize(), t.max().normalize())
rule_hol = wk[~wk.isin(nyse)]
assert not any(d in has16 for d in rule_hol), "a rule-calendar holiday with a 16:00 bar"
const = pd.DatetimeIndex(pd.to_datetime(list(LD.NYSE_HOLIDAY_FUTURES_SESSIONS)))
assert pd.DatetimeIndex(h["date"]).equals(const), (len(h), len(const))
ev = h[(h["date"] >= "2014-01-02") & (h["date"] <= "2024-04-30")]
print(
    f"{len(h)} holiday futures sessions {h['date'].min().date()} .. {h['date'].max().date()} = the constant; "
    f"rule-calendar holidays in the grid's span {len(rule_hol)}, of them with regular-hours bars {len(h)}; "
    f"in 2014-01-02 .. 2024-04-30: {len(ev)} dates, {int(ev['n_rth_bars'].sum())} regular-hours bars"
)
assert len(ev) == 64 and int(ev["n_rth_bars"].sum()) == 443, (len(ev), int(ev["n_rth_bars"].sum()))
h.to_csv(OUT, index=False)
