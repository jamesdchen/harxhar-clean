"""(b) UNOBSERVED_NEUTRAL on vs off on the 13-bar 2012 w1000 all-features design: which values change, by series, k
and class.  Read-only on both worktrees and on the main worktree's master_table_daily.parquet.

  python classify.py <design_off_dir> <design_on_dir> <out_prefix>

Full grid and availability rebuilt as load_and_transform does (load_raw_data(allow_missing), apply_overnight_fills,
obs = notna before the fill, rows with RV dropped); design rows = grid rows [3125:] (BURNIN_IN_BAR_SPACE), end stamps
10:00 .. 16:00 (rth13), NYSE early-close sessions dropped (DROP_EARLY_CLOSE_SESSIONS); checked against dates_oos.csv
and against the design's own <c>_avail_ma_<k> columns.

Classes of a rule-hit row (row t, series c, window = the k grid bars before t, all unobserved):
  iii  pre-start: t inside the LATE_START_WARMUP zero range of the column ([0, L + W), L = first live row)
  i    outage: the run of unobserved bars ending at t-1 covers at least one full regular session (all 13 bars
       10:00 .. 16:00 of a trading day); split terminal (no print of c after t) / temporary (c prints again later)
  ii   routine: neither (overnight / pre-open / intraday gaps); split by run length <= FFILL_LIMIT (26) / > 26
"""

import json
import sys
from pathlib import Path

WT = Path(r"C:\Users\james\AppData\Local\Temp\un_wt")
sys.path.insert(0, str(WT))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import src.backtest.executor as EX  # noqa: E402

MAIN = Path(r"C:\Users\james\CC Allowed\harxhar-0dte-professor")
d_off, d_on, outp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
m_off = json.loads((d_off / "design.json").read_text())
m_on = json.loads((d_on / "design.json").read_text())
names, raw = m_on["names"], m_on["raw_cols"]
assert names == m_off["names"]
Xo = np.load(d_off / "X.npy", mmap_mode="r")
Xn = np.load(d_on / "X.npy", mmap_mode="r")
n = Xo.shape[0]
W = 13000
FF = EX.FFILL_LIMIT

# ---- the grid and availability as load_and_transform builds them
g = EX.load_raw_data(str(WT / "data"), allow_missing=True)
EX.apply_overnight_fills(g, raw)
obs_full = {c: g[c].notna().to_numpy() for c in raw}
keep = g["RV"].notna().to_numpy()
t = pd.to_datetime(g["t"]).to_numpy()[keep]
obs = {c: v[keep] for c, v in obs_full.items()}
N = len(t)
tt = pd.DatetimeIndex(t)
mins = tt.hour * 60 + tt.minute
day = tt.normalize()
early = pd.to_datetime(list(EX.NYSE_EARLY_CLOSE_DATES))
pos = np.arange(N)
sel = (pos >= 3125) & (mins >= 600) & (mins <= 960) & ~day.isin(early)
G = pos[sel]  # grid row of each design row
assert len(G) == n, (len(G), n)
d_oos = pd.to_datetime(pd.read_csv(d_on / "dates_oos.csv")["date"])
assert (tt[G[W:]] == pd.DatetimeIndex(d_oos)).all()
dates = tt[G]
print(
    f"grid {N} rows, design {n} rows ({dates[0]} .. {dates[-1]}), W {W}, FFILL_LIMIT {FF}",
    flush=True,
)

# regular-session bounds on the grid: per trading day the first 10:00 bar and the 16:00 bar
rth = (mins >= 600) & (mins <= 960)
dcode, duniq = pd.factorize(day)
i10 = pd.Series(np.where(rth, pos, N + 1)).groupby(dcode).min().to_numpy()
i16 = pd.Series(np.where(rth, pos, -1)).groupby(dcode).max().to_numpy()
full_day = (i16 >= 0) & (i10 <= N)
i10f, i16f = i10[full_day], i16[full_day]
order = np.argsort(i16f)
i10f, i16f = i10f[order], i16f[order]
# latest complete regular session ending at or before bar t-1 of each design row
q = np.searchsorted(i16f, G - 1, side="right") - 1
s10 = np.where(q >= 0, i10f[np.maximum(q, 0)], N + 1)

# option days (866) 16:00 rows, and the walk span
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
r16 = dates.get_indexer(opt + pd.Timedelta("16h"))
assert len(opt) == 866 and (r16 >= W).all(), (len(opt), (r16 < 0).sum())
is16 = np.zeros(n, bool)
is16[r16] = True
first_eval = int(np.flatnonzero(dates >= "2012-01-03")[0])
lo_fit866 = int(r16.min() - W)
print(
    f"866 option-day 16:00 rows {r16.min()} .. {r16.max()}; first 2012 row {first_eval}; "
    f"first fit-window row of the option days {lo_fit866} ({dates[lo_fit866]})",
    flush=True,
)

rows, mism, tod_rows = [], [], []
per_series = {}
for c in raw:
    o = obs[c]
    last = np.maximum.accumulate(np.where(o, pos, -1))  # last observed bar <= j
    nxt = np.minimum.accumulate(np.where(o, pos, N + 1)[::-1])[
        ::-1
    ]  # next observed bar >= j
    L1 = last[G - 1]
    run = (G - 1) - L1  # unobserved bars ending at t-1 (all bars so far when L1 = -1)
    outage = (L1 < s10) & (run >= 13)
    terminal = nxt[G] > N  # no print of c at or after t
    # LATE_START_WARMUP age (executor._series_age_columns) on this grid
    if o.any():
        d0 = day[int(np.argmax(o))]
        age = None if d0 == day[0] else pos - int(np.argmax(day == d0))
    else:
        age = pos - (N + 1)
    for k in (1, 5, 25, 125, 625, 3125):
        vn = f"adj_{c}_ma_{k}"
        an = f"{c}_avail_ma_{k}"
        if vn not in names:
            continue
        jv = names.index(vn)
        hit = run >= k  # no observed bar among the k bars before t
        if an in names:  # the design's own availability average agrees
            av = np.asarray(Xn[:, names.index(an)])
            bad = int(((av < 0.5 / k) != hit).sum())
            exact = int(((av == 0.0) != hit).sum())
            if bad or exact:
                mism.append((an, bad, exact))
        if age is not None:
            live = age[G] >= k
            Lr = int(np.argmax(live)) if live.any() else n
            hi = min(n, Lr + W) if Lr > 0 else 0
        else:
            hi = 0
        pre = np.arange(n) < hi
        a, b = np.asarray(Xo[:, jv]), np.asarray(Xn[:, jv])
        ch = a != b
        dif = np.abs(a - b)
        assert (b[hit] == 0).all() and (a[~hit] == b[~hit]).all(), vn
        cls = {
            "iii": hit & pre,
            "i_term": hit & ~pre & outage & terminal,
            "i_temp": hit & ~pre & outage & ~terminal,
            "ii_le26": hit & ~pre & ~outage & (run <= FF),
            "ii_gt26": hit & ~pre & ~outage & (run > FF),
        }
        rec = dict(
            series=c,
            k=k,
            hit=int(hit.sum()),
            changed=int(ch.sum()),
            maxabs=float(dif.max()),
            pre_nonzero_off=int((a[pre] != 0).sum()),
        )
        for nm, m in cls.items():
            rec[f"hit_{nm}"] = int(m.sum())
            rec[f"ch_{nm}"] = int((m & ch).sum())
            rec[f"max_{nm}"] = float(dif[m].max()) if m.any() else 0.0
            rec[f"ch2012_{nm}"] = int((m & ch & (np.arange(n) >= first_eval)).sum())
            rec[f"ch16_{nm}"] = int((m & ch & is16).sum())
            rec[f"max16_{nm}"] = float(dif[m & is16].max()) if (m & is16).any() else 0.0
        rec["ch_fit866"] = int((ch & (np.arange(n) >= lo_fit866)).sum())
        rec["ch_pretune"] = int(
            (ch & (np.arange(n) >= 17956) & (np.arange(n) < 44281)).sum()
        )
        rec["ch_walkfit"] = int((ch & (np.arange(n) >= first_eval - W)).sum())
        rec["ch_varFF"] = int(
            (ch & (run > FF)).sum()
        )  # variant: unobserved > FFILL_LIMIT bars running
        rec["ch16_varFF"] = int((ch & (run > FF) & is16).sum())
        rec["ch2012_varFF"] = int(
            (ch & (run > FF) & (np.arange(n) >= first_eval)).sum()
        )
        rec["ch_varSess"] = int(
            (ch & outage & ~pre).sum()
        )  # alt: >= 1 full regular session unobserved
        for nm, m in cls.items():
            mm = m & ch
            if mm.any():
                dd = dates[mm]
                tod_rows.append(
                    pd.DataFrame(
                        dict(
                            series=c,
                            k=k,
                            cls=nm,
                            hhmm=dd.strftime("%H:%M"),
                            year=dd.year,
                            run=run[mm],
                            absd=dif[mm],
                        )
                    )
                )
        rec["ch16"] = int((ch & is16).sum())
        rec["max16"] = float(dif[is16].max())
        rec["days16"] = int(np.unique(dates[ch & is16].normalize()).size)
        rec["first_ch"] = str(dates[ch][0]) if ch.any() else ""
        rec["last_ch"] = str(dates[ch][-1]) if ch.any() else ""
        rows.append(rec)
    # outage episodes of c inside the design span: maximal unobserved runs covering >= 1 full regular session
    ep = []
    unob = ~o
    starts = np.flatnonzero(unob & np.r_[True, o[:-1]])
    ends = np.flatnonzero(unob & np.r_[o[1:], True])
    ns_all = np.maximum(
        0,
        np.searchsorted(i16f, ends, side="right")
        - np.searchsorted(i10f, np.where(starts > 0, starts, -1), side="right"),
    )
    for s, e, nsess in zip(starts, ends, ns_all):
        if nsess >= 1 and e >= G[0]:
            ep.append(
                (
                    str(tt[s]),
                    str(tt[e]),
                    nsess,
                    "pre-start" if s == 0 else ("terminal" if e == N - 1 else ""),
                )
            )
    per_series[c] = ep

r = pd.DataFrame(rows)
td = pd.concat(tod_rows, ignore_index=True)
td.groupby(["cls", "series", "k", "hhmm"]).agg(
    n=("absd", "size"),
    maxabs=("absd", "max"),
    run_min=("run", "min"),
    run_max=("run", "max"),
    y0=("year", "min"),
    y1=("year", "max"),
).reset_index().to_csv(f"{outp}_changed_by_tod.csv", index=False)
print("changed values by class and design-row time:")
print(td.groupby(["cls", "hhmm"]).size().unstack(0, fill_value=0).to_string())
print("changed values by class and year:")
print(td.groupby(["cls", "year"]).size().unstack(0, fill_value=0).to_string())
print("run length (bars) quantiles by class:")
print(td.groupby("cls")["run"].describe(percentiles=[0.05, 0.5, 0.95]).to_string())
r.to_csv(f"{outp}_by_series_k.csv", index=False)
with open(f"{outp}_outages.txt", "w") as fh:
    for c, ep in per_series.items():
        fh.write(f"{c}: {len(ep)} outage runs (>= 1 full session) in the design span\n")
        for e in ep[:40]:
            fh.write(f"    {e[0]} .. {e[1]}  sessions {e[2]} {e[3]}\n")
print(
    "availability-average mismatches (column, vs < 0.5/k, vs == 0):",
    mism if mism else "none",
    flush=True,
)
cl = ["iii", "i_term", "i_temp", "ii_le26", "ii_gt26"]
tot = {
    f"{p}_{x}": int(r[f"{p}_{x}"].sum())
    for p in ("hit", "ch", "ch2012", "ch16")
    for x in cl
}
mx = {x: float(r[f"max_{x}"].max()) for x in cl}
mx16 = {x: float(r[f"max16_{x}"].max()) for x in cl}
print(
    f"columns with a rule-hit row: {(r['hit'] > 0).sum()} of {len(r)} value columns; changed columns "
    f"{(r['changed'] > 0).sum()}; values changed {int(r['changed'].sum())} of {int(r['hit'].sum())} hit"
)
print(
    f"{'class':10s} {'hit':>10s} {'changed':>10s} {'ch>=2012':>10s} {'ch 16:00 opt':>12s} {'max|d|':>9s} {'max|d| 16':>9s}"
)
for x in cl:
    print(
        f"{x:10s} {tot['hit_' + x]:10d} {tot['ch_' + x]:10d} {tot['ch2012_' + x]:10d} {tot['ch16_' + x]:12d} "
        f"{mx[x]:9.3g} {mx16[x]:9.3g}"
    )
print(
    f"variant FFILL_LIMIT (unobserved > {FF} bars running): changes {int(r['ch_varFF'].sum())} (>= 2012 {int(r['ch2012_varFF'].sum())}, option 16:00 {int(r['ch16_varFF'].sum())}); variant full-session outage: {int(r['ch_varSess'].sum())}"
)
print(
    f"pre-start rows nonzero under flag off (should be 0): {int(r['pre_nonzero_off'].sum())}"
)
print(
    f"866 option-day 16:00 rows: values changed {int(r['ch16'].sum())} in {(r['ch16'] > 0).sum()} columns; "
    f"max |d| {r['max16'].max():.3g}; values changed in the option days' fit windows (rows >= {lo_fit866}): "
    f"{int(r['ch_fit866'].sum())} in {(r['ch_fit866'] > 0).sum()} columns"
)
print(
    f"values changed in the 2008-2011 pre-tune rows [17956, 44281): {int(r['ch_pretune'].sum())} in "
    f"{(r['ch_pretune'] > 0).sum()} columns; in the 2012 walk's fit windows and forecast rows (rows >= "
    f"{first_eval - W}): {int(r['ch_walkfit'].sum())} in {(r['ch_walkfit'] > 0).sum()} columns"
)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)
s = r.groupby("series")[[f"ch_{x}" for x in cl] + ["ch16", "max16"]].agg(
    {**{f"ch_{x}": "sum" for x in cl}, "ch16": "sum", "max16": "max"}
)
s = s[s.drop(columns="max16").sum(axis=1) > 0]
print(s.to_string())
k_tab = r.groupby("k")[[f"ch_{x}" for x in cl] + ["ch16"]].sum()
print(k_tab.to_string())
