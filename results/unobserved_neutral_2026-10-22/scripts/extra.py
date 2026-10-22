"""Extra numbers for (b): option days affected at 16:00, the E28 example column, the 12 routine 16:00 changes."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ref, on = Path(sys.argv[1]), Path(sys.argv[2])
MAIN = Path(r"C:\Users\james\CC Allowed\harxhar-0dte-professor")
names = json.loads((on / "design.json").read_text())["names"]
Xo, Xn = np.load(ref / "X.npy", mmap_mode="r"), np.load(on / "X.npy", mmap_mode="r")
W = 13000
d = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(on / "dates_oos.csv")["date"]))
daily = pd.read_parquet(
    MAIN / "results" / "close_master_table" / "master_table_daily.parquet",
    columns=["key", "day"],
)
opt = pd.DatetimeIndex(
    sorted(
        pd.to_datetime(
            daily.loc[daily["key"] == "sub_ols_baseline", "day"]
        ).dt.normalize()
    )
)
opt = opt[(opt >= "2020-01-03") & (opt <= "2024-04-30")]
r16 = W + d.get_indexer(opt + pd.Timedelta("16h"))
assert len(r16) == 866 and (r16 >= W).all()
A, B = np.asarray(Xo[r16]), np.asarray(Xn[r16])
ch = A != B
days = opt[ch.any(axis=1)]
print(
    f"option days with a changed 16:00 value: {len(days)} of 866 ({days.min().date()} .. {days.max().date()}); "
    f"changed values {int(ch.sum())} in {int(ch.any(axis=0).sum())} columns"
)
pre = days[days < "2023-09-01"]
for dd in pre:
    i = list(opt).index(dd)
    print("  before 2023-09-01:", dd.date(), [names[j] for j in np.flatnonzero(ch[i])])
j = names.index("adj_voldemand_all_open_only_ma_1")
s = pd.Series(A[:, j], index=opt)
t = s[s.index >= "2023-09-01"]
print(
    f"adj_voldemand_all_open_only_ma_1 on the option days' 16:00 rows from 2023-09-01 (flag off): first "
    f"{t.iloc[0]:.4f}, last {t.iloc[-1]:.4f}, min {t.min():.4f}, max {t.max():.4f}, s.d. {t.std():.4f}, n {len(t)}; "
    f"flag on: all 0 = {bool((B[s.index >= '2023-09-01', j] == 0).all())}"
)
for nm in (
    "adj_voldemand_spx_open_and_close_ma_1",
    "adj_voldemand_spx_open_and_close_ma_3125",
):
    j = names.index(nm)
    m = ch[:, j]
    print(
        f"{nm}: {int(m.sum())} option-day 16:00 rows changed, max |d| {np.abs(A[m, j] - B[m, j]).max():.3f}, "
        f"old range {A[m, j].min():.3f} .. {A[m, j].max():.3f}"
    )
# all-features scaled values on the eval span: max |x| of the changed columns before/after
k = np.arange(W, Xo.shape[0])
