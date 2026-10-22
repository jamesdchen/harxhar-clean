"""(c) Compare the base and perturbed builds of perturb_build.py (UNOBSERVED_NEUTRAL on): does anything dated at or
before T* move when only data after T* change?  Also: the base build flag on vs flag off (pt_base_off.npz), which
columns the rule changes on the reduced bucket, and whether the rule fires after T* in 'stop'."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
T0 = pd.Timestamp("2016-06-15 15:30")
rows = []
a = np.load(HERE / "pt_base.npz", allow_pickle=True)
na = [str(x) for x in a["names"]]
da = pd.DatetimeIndex(a["dates"])
for scen in ("all", "avail", "stop"):
    pp = HERE / f"pt_{scen}.npz"
    if not pp.exists():
        print("missing", pp.name)
        continue
    b = np.load(pp, allow_pickle=True)
    nb = [str(x) for x in b["names"]]
    db = pd.DatetimeIndex(b["dates"])
    print(
        f"== {scen}: T* = {T0}; base X {a['X'].shape}, perturbed X {b['X'].shape}; same columns {na == nb}; "
        f"same rows {da.equals(db)}"
    )
    common = da.intersection(db)
    ia, ib = da.get_indexer(common), db.get_indexer(common)
    cn = [x for x in na if x in nb]
    ja, jb = [na.index(x) for x in cn], [nb.index(x) for x in cn]
    Xa, Xb = a["X"][np.ix_(ia, ja)], b["X"][np.ix_(ib, jb)]
    kx = common <= T0 + pd.Timedelta(
        "30min"
    )  # X rows dated <= T* + 30 min hold only bars <= T* (shift 1)
    ky = common <= T0
    eq = (Xa == Xb) | (np.isnan(Xa) & np.isnan(Xb))
    bad = [
        (cn[j], int((~eq[kx, j]).sum())) for j in range(len(cn)) if (~eq[kx, j]).any()
    ]
    ybad = int((a["y"][ia][ky] != b["y"][ib][ky]).sum())
    Bbad = int((a["baseline_all"][ia][ky] != b["baseline_all"][ib][ky]).sum())
    after_rows = int((~eq[~kx]).any(axis=1).sum())
    after_cols = int((~eq[~kx]).any(axis=0).sum())
    msg = ""
    if scen == "stop":
        stopped = [
            j
            for j, x in enumerate(cn)
            if re.match(
                r"adj_(voldemand_(spx|all)_open_only|vix|stocktwits_sentiment"
                r"|sumvolume)_ma_\d+$",
                x,
            )
        ]
        z = int((Xb[~kx][:, stopped] == 0).sum())
        zb = int((Xa[~kx][:, stopped] == 0).sum())
        msg = f"; stopped series' value averages after T*: {z} zeros perturbed vs {zb} base"
    print(
        f"   rows <= T*(+30m): {int(kx.sum())}; X columns that move on them: {len(bad)} of {len(cn)}; y moves on "
        f"{ybad} of {int(ky.sum())} rows <= T*; B moves on {Bbad}; (control) X rows after T* that move: {after_rows} "
        f"of {int((~kx).sum())} in {after_cols} columns{msg}"
    )
    for c in sorted(bad, key=lambda x: -x[1])[:25]:
        print(f"     {c[0]}: {c[1]} rows <= T*")
    rows.append(
        dict(
            scen=scen,
            cols_moved_before_T=len(bad),
            y_moved=ybad,
            B_moved=Bbad,
            rows_after_moved=after_rows,
            cols_after_moved=after_cols,
        )
    )
off = HERE / "pt_base_off.npz"
if off.exists():
    o = np.load(off, allow_pickle=True)
    assert [str(x) for x in o["names"]] == na and pd.DatetimeIndex(o["dates"]).equals(
        da
    )
    ch = a["X"] != o["X"]
    cols = [
        (na[j], int(ch[:, j].sum()), float(np.abs(a["X"][:, j] - o["X"][:, j]).max()))
        for j in range(len(na))
        if ch[:, j].any()
    ]
    print(
        f"== base flag on vs off: {len(cols)} of {len(na)} columns change, {int(ch.sum())} values; all changed values "
        f"are 0 under the flag: {bool((a['X'][ch] == 0).all())}; non-value columns changed: "
        f"{[c for c, _, _ in cols if not c.startswith('adj_')]}"
    )
pd.DataFrame(rows).to_csv(HERE / "perturb_results.csv", index=False)
