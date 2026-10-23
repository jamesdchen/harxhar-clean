"""LATE_START_WARMUP's zeroed values at any window, from the rule (src/backtest/executor.py late_start_warmup):
column j of a late-start series is 0 on design rows [0, hi_j), hi_j = min(n, L_j + W), L_j its first live row, which
does not depend on W.  L_j = the w1000 zero prefix - 13,000 (latestart_w1000.csv, measured on run B's design and
checked there against the design log's total).  Each window's total (sum of hi_j) is checked against its design log.

usage: python latestart_rule.py <latestart_w1000.csv> <run B design_all_features dir> <option_days.csv> <out_csv>
       [W_days=<design log> ...]
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

L0 = pd.read_csv(sys.argv[1])
D = Path(sys.argv[2])
OPT = pd.to_datetime(pd.read_csv(sys.argv[3])["day"])
OUT = Path(sys.argv[4])
LOGS = dict(a.split("=", 1) for a in sys.argv[5:])
N_ROWS, W0 = 84747, 13000
dates = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(D / "dates_oos.csv")["date"]))  # rows W0 .. n-1
assert len(dates) == N_ROWS - W0
k16 = (dates.hour == 16) & (dates.minute == 0) & dates.normalize().isin(OPT)
rows16 = W0 + np.flatnonzero(k16)
assert len(rows16) == 866
L = L0["zero_prefix_rows"].to_numpy() - W0
assert (L > 0).all()


def row_date(r: int) -> str:
    return str(dates[r - W0]) if W0 <= r < N_ROWS else ("never (past the data's end)" if r >= N_ROWS else "< row 13,000")


out = []
for wd in (1000, 1500, 2000, 2500, 3000, 3500):
    W = 13 * wd
    hi = np.minimum(N_ROWS, L + W)
    tot = int(hi.sum())
    chk = ""
    if str(wd) in LOGS:
        m = re.search(r"late-start warm-up: (\d+) columns, (\d+) values set to 0 \(window (\d+) rows\)",
                      Path(LOGS[str(wd)]).read_text())
        assert m and int(m.group(1)) == len(L) and int(m.group(3)) == W, (wd, m and m.groups())
        assert int(m.group(2)) == tot, (wd, int(m.group(2)), tot)
        chk = "log total equal"
    print(f"w{wd}: {len(L)} columns, {tot} values zeroed over the design ({chk or 'no log given'})")
    for (_, r), h in zip(L0.iterrows(), hi):
        z = rows16 < h
        out.append(dict(window=f"w{wd}", series=r["series"], k=int(r["k"]), column=r["column"], first_live_row=int(h - W),
                        zero_prefix_rows=int(h), live_from=row_date(int(h)), option_days_zeroed=int(z.sum()),
                        last_zeroed_option_day=str(dates[rows16[z][-1] - W0].date()) if z.any() else ""))
res = pd.DataFrame(out)
res.to_csv(OUT, index=False)
for w, g in res.groupby("window", sort=False):
    s = g.groupby("series", sort=False).agg(
        values=("option_days_zeroed", "sum"), all6=("option_days_zeroed", "min"), any=("option_days_zeroed", "max"),
        last=("last_zeroed_option_day", "max"), live_ma3125=("live_from", "last"))
    print(f"\n{w}: {int(g['option_days_zeroed'].sum())} values zeroed on the 866 option days' 16:00 rows")
    print(s.to_string())
