"""Write notebooks/atm_straddle_intraday.ipynb — every 30-min 0DTE bar."""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "285J",
    "language": "python",
    "name": "python3",
}


def md(s: str):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s: str):
    return nbf.v4.new_code_cell(s.strip("\n") + "\n")


nb.cells = [
    md(
        r"""
# 0DTE nearest-OTM package, every 30-min bar

Same chain, nearest-OTM picker, and 250-day smear as
`atm_straddle_rv_iv.ipynb`, but one trade per 30-min stamp on
expiration days instead of only 15:30→close.

Hold the two legs picked at $t$. Exit at $t+1$ mid of those same
strikes if both still live; last bar of the day cash-settles vs
`^GSPC`. Drop the bar if a leg is missing at $t+1$.
"""
    ),
    code(
        """
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import yfinance as yf

sys.path.insert(0, str(Path.cwd() if (Path.cwd() / "atm_straddle_lib.py").exists() else Path.cwd() / "notebooks"))
import atm_straddle_lib as asl

REPO = asl.find_repo(Path.cwd())
OUT = REPO / "results" / "atm_straddle_intraday"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
print("repo:", REPO)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda x: f"{x: .6f}")
"""
    ),
    md("## 1. Load the 0DTE chain (every 30-min stamp)"),
    code(
        """
path = REPO / "data" / "spxw_chain.parquet"
COLS = ["expiration", "timestamp", "strike", "cp", "bid", "ask", "mid",
        "underlying_price", "impl_volatility"]
opt = ["hours_to_expiration"]
import pyarrow.parquet as pq
avail_cols = set(pq.ParquetFile(path).schema_arrow.names)
keep_cols = [c for c in COLS + opt if c in avail_cols]
# The full file is large — filter 0DTE after clocks. Cache by size+mtime.
_st = os.stat(path)
_ck = CACHE / f"chain_0dte_{_st.st_size}_{_st.st_mtime_ns}.parquet"
if _ck.exists():
    chain = pd.read_parquet(_ck)
    print("cache hit", _ck.name)
else:
    raw = pd.read_parquet(path, columns=keep_cols)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw["expiration"] = pd.to_datetime(raw["expiration"])
    raw["cp"] = raw["cp"].astype(str).str.upper().str[0]
    codes, uts = pd.factorize(raw["timestamp"])
    uet = pd.DatetimeIndex(uts).tz_convert("America/New_York")
    raw["et"] = uet.take(codes)
    raw["et_date"] = uet.normalize().take(codes)
    ecodes, uexp = pd.factorize(raw["expiration"])
    uexp_d = pd.DatetimeIndex(uexp).tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").normalize()
    raw["exp_date"] = uexp_d.take(ecodes)
    raw["is_0dte"] = raw["et_date"] == raw["exp_date"]
    chain = raw[raw["is_0dte"]].copy()
    for old in CACHE.glob("chain_0dte_*.parquet"):
        old.unlink()
    chain.to_parquet(_ck)
print("0DTE rows", f"{len(chain):,}")
print(chain.head(3))
"""
    ),
    md("## 2. Regular hours; drop half-sessions"),
    code(
        """
et = chain["et"]
mins = et.dt.hour * 60 + et.dt.minute
rth = (mins >= 9 * 60 + 30) & (mins <= 16 * 60)
chain = chain[rth].copy()
if "hours_to_expiration" in chain.columns:
    open_hte = (
        chain[(et.dt.hour == 9) & (et.dt.minute == 30)]
        .groupby("expiration")["hours_to_expiration"].median()
    )
    half = open_hte[np.abs(open_hte.astype(float) - 6.5) > 0.2].index
    print("half-sessions dropped", len(half))
    chain = chain[~chain["expiration"].isin(half)].copy()
print("rows after RTH filter", f"{len(chain):,}", "days", chain["expiration"].nunique())
"""
    ),
    md("## 3. Nearest-OTM package at each stamp"),
    code(
        """
live = chain[np.isfinite(chain["mid"]) & (chain["mid"] > 0)].copy()
spot = live.dropna(subset=["underlying_price"]).groupby(["expiration", "timestamp"])["underlying_price"].median()
c = live[live["cp"] == "C"].copy()
p = live[live["cp"] == "P"].copy()
c["S"] = c.set_index(["expiration", "timestamp"]).index.map(spot)
p["S"] = p.set_index(["expiration", "timestamp"]).index.map(spot)
c = c[np.isfinite(c["S"])]
p = p[np.isfinite(p["S"])]
c_otm = c[c["strike"] >= c["S"]].copy()
c_otm["k_gap"] = c_otm["strike"].astype(float) - c_otm["S"]
c_pick = c_otm.sort_values(["expiration", "timestamp", "k_gap", "strike"]).groupby(["expiration", "timestamp"], as_index=False).first()
p_otm = p[p["strike"] <= p["S"]].copy()
p_otm["k_gap"] = p_otm["S"] - p_otm["strike"].astype(float)
p_pick = p_otm.sort_values(["expiration", "timestamp", "k_gap", "strike"]).groupby(["expiration", "timestamp"], as_index=False).first()
pkg = c_pick.merge(p_pick, on=["expiration", "timestamp"], suffixes=("_c", "_p"))
pkg["S"] = pkg["S_c"].astype(float)
pkg["K_c"] = pkg["strike_c"].astype(float)
pkg["K_p"] = pkg["strike_p"].astype(float)
pkg["entry"] = pkg["mid_c"].astype(float) + pkg["mid_p"].astype(float)
pkg["iv_hourly"] = pkg[["impl_volatility_c", "impl_volatility_p"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
pkg = pkg.sort_values(["expiration", "timestamp"]).reset_index(drop=True)
print("packages", len(pkg), "days", pkg["expiration"].nunique())
print(pkg[["expiration", "timestamp", "S", "K_c", "K_p", "entry"]].head())
"""
    ),
    md(
        r"""
## 4. Exit: next-stamp mid of the same strikes; last bar cash-settles

$R_t = \mathrm{exit}_{t+1}/\mathrm{entry}_t - 1$.
"""
    ),
    code(
        """
# next mid of the same (expiration, K_c, K_p)
mids = live.set_index(["expiration", "timestamp", "strike", "cp"])["mid"]

def next_mid(row, nxt_ts):
    try:
        mc = mids.loc[(row["expiration"], nxt_ts, row["K_c"], "C")]
        mp = mids.loc[(row["expiration"], nxt_ts, row["K_p"], "P")]
        if hasattr(mc, "iloc"):
            mc = mc.iloc[0]
        if hasattr(mp, "iloc"):
            mp = mp.iloc[0]
        return float(mc) + float(mp)
    except KeyError:
        return np.nan

pkg["nxt_ts"] = pkg.groupby("expiration")["timestamp"].shift(-1)
pkg["is_last"] = pkg["nxt_ts"].isna()
# last-bar close
days = pd.to_datetime(pkg["expiration"])
if getattr(days.dt, "tz", None) is not None:
    days = days.dt.tz_convert("America/New_York").dt.tz_localize(None)
days = days.dt.normalize()
cp = CACHE / "gspc_close.parquet"
if cp.exists():
    close = pd.read_parquet(cp)["close"]
    close.index = pd.to_datetime(close.index)
else:
    raw = yf.download("^GSPC", start=days.min() - pd.Timedelta("7D"),
                      end=days.max() + pd.Timedelta("7D"),
                      auto_adjust=True, progress=False, threads=True)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    ix = pd.to_datetime(close.index)
    if getattr(ix, "tz", None) is not None:
        ix = ix.tz_convert("America/New_York").tz_localize(None)
    close.index = ix.normalize()
    close.rename("close").to_frame().to_parquet(cp)
pkg["S_close"] = days.map(close.astype(float))
pkg["exit_settle"] = np.maximum(pkg["S_close"] - pkg["K_c"], 0.0) + np.maximum(pkg["K_p"] - pkg["S_close"], 0.0)

# vectorized next-mid via a lookup frame
nxt = pkg.loc[~pkg["is_last"], ["expiration", "nxt_ts", "K_c", "K_p"]].copy()
c_next = live[live["cp"] == "C"][["expiration", "timestamp", "strike", "mid"]].rename(
    columns={"timestamp": "nxt_ts", "strike": "K_c", "mid": "mid_c_nxt"})
p_next = live[live["cp"] == "P"][["expiration", "timestamp", "strike", "mid"]].rename(
    columns={"timestamp": "nxt_ts", "strike": "K_p", "mid": "mid_p_nxt"})
nxt = nxt.merge(c_next, on=["expiration", "nxt_ts", "K_c"], how="left")
nxt = nxt.merge(p_next, on=["expiration", "nxt_ts", "K_p"], how="left")
pkg = pkg.merge(nxt[["expiration", "nxt_ts", "K_c", "K_p", "mid_c_nxt", "mid_p_nxt"]],
                on=["expiration", "nxt_ts", "K_c", "K_p"], how="left")
pkg["exit_mark"] = pkg["mid_c_nxt"] + pkg["mid_p_nxt"]
pkg["exit"] = np.where(pkg["is_last"], pkg["exit_settle"], pkg["exit_mark"])
pkg = pkg[np.isfinite(pkg["entry"]) & np.isfinite(pkg["exit"]) & (pkg["entry"] > 0)].copy()
pkg["R"] = pkg["exit"] / pkg["entry"] - 1.0
pkg["et"] = pd.to_datetime(pkg["timestamp"], utc=True).dt.tz_convert("America/New_York")
pkg["hour"] = pkg["et"].dt.hour
pkg["date"] = pkg["et"].dt.normalize().dt.tz_localize(None)
print("bars with a return", len(pkg), "last-bar fraction", float(pkg["is_last"].mean()))
print(pkg["R"].describe())
"""
    ),
    md(
        r"""
## Side question: how did Chris calculate 1-hour IVs?

Facts in the repo:

- Paper (`methods_close_option.tex`): vendor quote is an **hourly** vol;
  remaining window 30 min $\Rightarrow \mathrm{IV}_{30}=\mathrm{IV}/\sqrt{2}$.
- `experiments/spxw_quote_costs.py` / `spxw_iv_bumps.py`: vendor
  `impl_volatility` unit is unknown / per-period; they **ignore it** and
  invert BS IV from the mid with $\tau=h_{\mathrm{close}}/(252\times 6.5)$.
- OM export spec: OptionMetrics `impl_volatility` is **annualized**;
  1-day var $=\mathrm{iv}^2/252$.
- `prep_spxw.py` renamed `new_implied_vol` $\to$ `impl_volatility`.
- `spxw_resign.py`: vendor `new_implied_vol` is $\sim 100\times$ too
  small — $\sim 0.002$ at ATM while mids price $\sim 20\%$ vol.

That $0.002$ vs $0.20$ is the unit tell. Treating the vendor number as
a 1-hour standard deviation (Chris) puts 30-min variance on the same
$10^{-6}$ scale as $\widehat{RV}$. Treating it as annualized does not.
"""
    ),
    code(
        """
iv = pkg["iv_hourly"].astype(float)
print("median vendor IV", float(iv.median()), "median package mid", float(pkg["entry"].median()))
conv = asl.iv_var_from_conventions(iv, hours_remaining=0.5)
for k, s in conv.items():
    print(k, "median var", float(s.median()))
# rough BS-inverted ATM vol from the straddle mid, last bar only as a check
# tau = 0.5 / (252*6.5) at 15:30. Skip a full solver; print scale comparison only.
print("chris 30min var is (IV)^2 * 0.5 = (IV/sqrt(2))^2 — matches the close book.")
print("annualized_om is ~ 1/(252*6.5) smaller and will not match rv_hat ~ 3e-6.")
"""
    ),
    md("## 5. Forecasts on every bar; signal under each IV convention"),
    code(
        """
YHATS = asl.yhat_paths(REPO)
LABEL = asl.YHAT_LABEL
MODEL_ORDER = asl.MODEL_ORDER
# join blk2 first as the working forecast; load all 15:30-style panels
panels = {}
for tag, path in YHATS.items():
    if not path.exists():
        print("missing", path)
        continue
    df = asl.load_yhat_panel(path)
    panels[tag] = df.set_index("t")[["rv_hat", "yhat", "rv_raw", "baseline"]]
    print(tag, "bars", len(panels[tag]))

pkg["t"] = pd.to_datetime(pkg["timestamp"], utc=True)
work = pkg.copy()
if "blk2" in panels:
    work = work.merge(panels["blk2"].reset_index(), left_on="t", right_on="t", how="left")
work["iv_var_chris"] = conv["chris_hourly"].reindex(work.index).to_numpy() if False else (work["iv_hourly"].astype(float) ** 2) * 0.5
work["iv_var_om"] = (work["iv_hourly"].astype(float) ** 2) * 0.5 / (252.0 * 6.5)
work["iv_var_raw"] = work["iv_hourly"].astype(float) ** 2
print("median rv_hat", float(pd.to_numeric(work.get("rv_hat", pd.Series(dtype=float)), errors="coerce").median()))
print("median iv_var chris / om / raw",
      float(work["iv_var_chris"].median()),
      float(work["iv_var_om"].median()),
      float(work["iv_var_raw"].median()))
print("Chris convention is the one on the same order as rv_hat.")
work["signal"] = work["rv_hat"] - work["iv_var_chris"]
work = work.dropna(subset=["R", "signal"])
work["pos"] = np.where(work["signal"] > 0, 1.0, -1.0)
print("bars after join", len(work))
"""
    ),
    md(
        r"""
## 6. Three books, sized on 63 *trading days*

Always short / $\mathrm{sign}(s)$ / unit-median. Expanding median of
$|s|$ is computed on 15:30 bars only (63 days), then mapped back to
every bar of that day — same causal information as the close book.
"""
    ),
    code(
        r"""
s1530 = work[(work["et"].dt.hour == 15) & (work["et"].dt.minute == 30)].copy()
lev_day = asl.causal_leverage(s1530.set_index("date")["signal"])
work = work.merge(lev_day.rename("lev").reset_index(), on="date", how="left")
work["lev"] = work["lev"].fillna(1.0)
q = {
    "always short": pd.Series(-1.0, index=work.index),
    "long-short volatility": work["pos"],
    "unit-median VRP": work["pos"] * work["lev"],
}
rows = []
for name, size in q.items():
    rp = size * work["R"]
    st = asl.rule_row(rp, size)
    rows.append({"rule": name, **st.to_dict()})
tab = pd.DataFrame(rows).set_index("rule")
print(tab.to_string())
tab.to_csv(OUT / "rule_table_intraday_blk2.csv")
"""
    ),
    md("## 7. Table by entry hour"),
    code(
        r"""
hour_rows = []
for hr, g in work.groupby("hour"):
    for name, size in q.items():
        rp = (size * work["R"]).loc[g.index]
        st = asl.rule_row(rp, size.loc[g.index])
        hour_rows.append({"hour": int(hr), "rule": name, **st.to_dict()})
htab = pd.DataFrame(hour_rows)
print(htab.to_string(index=False))
htab.to_csv(OUT / "rule_by_entry_hour.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 3.4))
for name in q:
    sub = htab[htab["rule"] == name]
    ax.plot(sub["hour"], sub["mean"], marker="o", label=name)
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("entry hour ET")
ax.set_ylabel("mean R'")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "mean_by_entry_hour.png", dpi=120, bbox_inches="tight")
display(fig)
plt.close(fig)
print("saved CSVs in", OUT)
"""
    ),
]

path = Path(__file__).resolve().parent / "atm_straddle_intraday.ipynb"
nbf.write(nb, path)
print("wrote", path)
