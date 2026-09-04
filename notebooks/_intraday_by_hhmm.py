"""Re-score the intraday trade by 30-min stamp (not clock hour)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atm_straddle_lib as asl

REPO = asl.find_repo(Path(__file__).resolve().parent)
OUT = REPO / "results" / "atm_straddle_intraday"
CACHE = OUT / "cache"
path = REPO / "data" / "spxw_chain.parquet"
_st = os.stat(path)
ck = CACHE / f"chain_0dte_{_st.st_size}_{_st.st_mtime_ns}.parquet"
print("chain cache", ck.exists(), ck)
chain = pd.read_parquet(ck)
et = chain["et"]
mins = et.dt.hour * 60 + et.dt.minute
rth = (mins >= 9 * 60 + 30) & (mins <= 16 * 60)
chain = chain[rth].copy()
if "hours_to_expiration" in chain.columns:
    open_hte = (
        chain[(et.dt.hour == 9) & (et.dt.minute == 30)]
        .groupby("expiration")["hours_to_expiration"]
        .median()
    )
    half = open_hte[np.abs(open_hte.astype(float) - 6.5) > 0.2].index
    chain = chain[~chain["expiration"].isin(half)].copy()

live = chain[np.isfinite(chain["mid"]) & (chain["mid"] > 0)].copy()
spot = asl.stamp_spot(live, ["expiration", "timestamp"])
c = live[live["cp"] == "C"].copy()
p = live[live["cp"] == "P"].copy()
c["S"] = c.set_index(["expiration", "timestamp"]).index.map(spot)
p["S"] = p.set_index(["expiration", "timestamp"]).index.map(spot)
c = c[np.isfinite(c["S"])]
p = p[np.isfinite(p["S"])]
c_otm = c[c["strike"] >= c["S"]].copy()
c_otm["k_gap"] = c_otm["strike"].astype(float) - c_otm["S"]
c_pick = (
    c_otm.sort_values(["expiration", "timestamp", "k_gap", "strike"])
    .groupby(["expiration", "timestamp"], as_index=False)
    .first()
)
p_otm = p[p["strike"] <= p["S"]].copy()
p_otm["k_gap"] = p_otm["S"] - p_otm["strike"].astype(float)
p_pick = (
    p_otm.sort_values(["expiration", "timestamp", "k_gap", "strike"])
    .groupby(["expiration", "timestamp"], as_index=False)
    .first()
)
pkg = c_pick.merge(p_pick, on=["expiration", "timestamp"], suffixes=("_c", "_p"))
pkg["S"] = pkg["S_c"].astype(float)
pkg["K_c"] = pkg["strike_c"].astype(float)
pkg["K_p"] = pkg["strike_p"].astype(float)
pkg["entry"] = pkg["mid_c"].astype(float) + pkg["mid_p"].astype(float)
pkg["iv_hourly"] = (
    pkg[["impl_volatility_c", "impl_volatility_p"]]
    .apply(pd.to_numeric, errors="coerce")
    .mean(axis=1)
)
pkg = pkg.sort_values(["expiration", "timestamp"]).reset_index(drop=True)

pkg["nxt_ts"] = pkg.groupby("expiration")["timestamp"].shift(-1)
pkg["is_last"] = pkg["nxt_ts"].isna()
days = pd.to_datetime(pkg["expiration"])
if getattr(days.dt, "tz", None) is not None:
    days = days.dt.tz_convert("America/New_York").dt.tz_localize(None)
days = days.dt.normalize()
close = pd.read_parquet(CACHE / "gspc_close.parquet")["close"]
close.index = pd.to_datetime(close.index)
pkg["S_close"] = days.map(close.astype(float))
pkg["exit_settle"] = np.maximum(pkg["S_close"] - pkg["K_c"], 0.0) + np.maximum(
    pkg["K_p"] - pkg["S_close"], 0.0
)
nxt = pkg.loc[~pkg["is_last"], ["expiration", "nxt_ts", "K_c", "K_p"]].copy()
c_next = live[live["cp"] == "C"][["expiration", "timestamp", "strike", "mid"]].rename(
    columns={"timestamp": "nxt_ts", "strike": "K_c", "mid": "mid_c_nxt"}
)
p_next = live[live["cp"] == "P"][["expiration", "timestamp", "strike", "mid"]].rename(
    columns={"timestamp": "nxt_ts", "strike": "K_p", "mid": "mid_p_nxt"}
)
nxt = nxt.merge(c_next, on=["expiration", "nxt_ts", "K_c"], how="left")
nxt = nxt.merge(p_next, on=["expiration", "nxt_ts", "K_p"], how="left")
pkg = pkg.merge(
    nxt[["expiration", "nxt_ts", "K_c", "K_p", "mid_c_nxt", "mid_p_nxt"]],
    on=["expiration", "nxt_ts", "K_c", "K_p"],
    how="left",
)
pkg["exit_mark"] = pkg["mid_c_nxt"] + pkg["mid_p_nxt"]
pkg["exit"] = np.where(pkg["is_last"], pkg["exit_settle"], pkg["exit_mark"])
pkg = pkg[
    np.isfinite(pkg["entry"]) & np.isfinite(pkg["exit"]) & (pkg["entry"] > 0)
].copy()
pkg["R"] = pkg["exit"] / pkg["entry"] - 1.0
pkg["et"] = pd.to_datetime(pkg["timestamp"], utc=True).dt.tz_convert("America/New_York")
pkg["hhmm"] = pkg["et"].dt.strftime("%H:%M")
pkg["date"] = pkg["et"].dt.normalize().dt.tz_localize(None)

print("packages with R", len(pkg))
print("stamps", sorted(pkg["hhmm"].unique()))
print("raw long R by hhmm")
raw = pkg.groupby("hhmm")["R"].agg(["count", "mean", "std", "median"])
raw["t"] = raw["mean"] / raw["std"] * np.sqrt(raw["count"])
print(raw.to_string())

blk2 = asl.load_yhat_panel(asl.yhat_paths(REPO)["blk2"])
blk2 = blk2.set_index("t")[["rv_hat"]]
pkg["t"] = pd.to_datetime(pkg["timestamp"], utc=True)
work = pkg.merge(blk2.reset_index(), on="t", how="left")
work["iv_var_chris"] = work["iv_hourly"].astype(float) ** 2 * 0.5
work["signal"] = work["rv_hat"] - work["iv_var_chris"]
work = work.dropna(subset=["R", "signal"])
work["pos"] = np.where(work["signal"] > 0, 1.0, -1.0)
work = work.sort_values("t").reset_index(drop=True)
q = {
    "always short": pd.Series(-1.0, index=work.index),
    "sign(s)": work["pos"],
}
rows = []
for hhmm, g in work.groupby("hhmm", sort=True):
    for name, size in q.items():
        rp = (size * work["R"]).loc[g.index]
        st = asl.rule_row(rp, size.loc[g.index])
        rows.append({"hhmm": hhmm, "rule": name, **st.to_dict()})
htab = pd.DataFrame(rows)
outp = OUT / "rule_by_entry_hhmm.csv"
htab.to_csv(outp, index=False)
print("scored bars", len(work), "->", outp)
sub = htab[htab["rule"] == "always short"][
    ["hhmm", "n", "mean", "t_mean", "Sharpe_ann", "pct_buy"]
]
print("\nalways short by stamp")
print(
    htab[htab["rule"] == "always short"][
        ["hhmm", "n", "mean", "std", "t_mean", "Sharpe_ann"]
    ].to_string(index=False)
)
print("\nsign(s) by stamp")
print(
    htab[htab["rule"] == "sign(s)"][
        ["hhmm", "n", "mean", "t_mean", "Sharpe_ann", "pct_buy"]
    ].to_string(index=False)
)

rows = []
for name, size in q.items():
    rp = size * work["R"]
    st = asl.rule_row(rp, size)
    rows.append({"rule": name, **st.to_dict()})
tab = pd.DataFrame(rows).set_index("rule")
tab.to_csv(OUT / "rule_table_intraday_blk2.csv")
print("\npooled (still 16:00-contaminated)")
print(tab.to_string())

hour_rows = []
work["hour"] = work["et"].dt.hour
for hr, g in work.groupby("hour"):
    for name, size in q.items():
        rp = (size * work["R"]).loc[g.index]
        st = asl.rule_row(rp, size.loc[g.index])
        hour_rows.append({"hour": int(hr), "rule": name, **st.to_dict()})
htab_h = pd.DataFrame(hour_rows)
htab_h.to_csv(OUT / "rule_by_entry_hour.csv", index=False)
print("\nalways short / sign(s) by hour")
print(
    htab_h[htab_h["rule"].isin(["always short", "sign(s)"])][
        ["hour", "rule", "n", "mean", "t_mean", "Sharpe_ann"]
    ].to_string(index=False)
)
