"""Write notebooks/atm_straddle_rv_iv.ipynb as a 15:30-16:00 0DTE pipeline."""

from pathlib import Path

import nbformat as nbf

from _nb_io import carry_outputs

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
# Same-day-expiry nearest-out-of-the-money call and put, 15:30 → 16:00

One trade per expiration day. At 15:30 ET the "at-the-money" package is
defined as the **nearest out-of-the-money** legs:

- call: the smallest listed strike $K_c \ge S$ with a live midpoint quote
- put: the largest listed strike $K_p \le S$ with a live midpoint quote

When the index sits exactly on a strike the two legs share it and the
package is a straddle; otherwise it is a strangle one strike wide. The
package is held to the close and settled in cash against the official
S&P 500 close (yfinance `^GSPC`): the call pays
$\max(S_{\mathrm{close}}-K_c,0)$ and the put pays
$\max(K_p-S_{\mathrm{close}},0)$.

The signal compares the forecast of realized variance over the final
half hour with the variance the option market quotes for that same
half hour, both in variance units:
$s=\widehat{RV}-(\mathrm{IV}_{\mathrm{hourly}}/\sqrt{2})^2$, where
$\widehat{RV}=(m^2+\hat\sigma^2)B$ is the model's forecast mapped onto
the realized scale by a recalibration estimated from past data only
(§7). Two position rules are scored on the same 871 days. The headline
is the **long–short** book, which sells the package when the market's
quoted variance exceeds the forecast and buys it otherwise: on the
block-diagonal ridge forecast it earns an annualized Sharpe ratio of
1.63 ($t = 3.03$). The **always-short** book, which uses no forecast, is
the control, at a Sharpe ratio of 0.28.

The notebook runs as follows. Sections 1–6 build the instrument and its
return; §7 loads the variance forecasts and recalibrates them; §8 puts
the quoted implied volatility in the same units; §9 forms the signal and
the position; §10 tabulates the two rules across the seven forecasts;
§11 adds up the profit and loss; §12 reports information ratios against
the always-short book; §13 measures the 15:30 signal against the same afternoon's
settlement return — a sign effect, not a slope — and shows why
yesterday's signal carries nothing: the gap between forecast and
implied is renewed daily; §14 bets a fixed
fraction of wealth and sets out the two frames in which such a fraction
can be read; §15 diagnoses the buy days; §16 checks one row by hand. A
defined-risk variant with credit vertical spreads is under construction
and not shown in this deck.

Volatility-scale views of the forecasts ($\hat y\sqrt{B}$, $m\sqrt{B}$)
are in `atm_straddle_volmap.ipynb` where present and otherwise in
`atm_straddle_experimental.ipynb`; ensembles and alternative weightings
are in `atm_straddle_experimental.ipynb`; the every-half-hour version of
this book is in `atm_straddle_intraday.ipynb`.

Every cell reads from `data/` or from a table built by the cell before
it, and prints each table before using it.
"""
    ),
    code(
        """
import hashlib
import os
from pathlib import Path
import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import yfinance as yf
import statsmodels.api as sm

def find_repo(start: Path) -> Path:
    for q in [start.resolve(), *start.resolve().parents]:
        if (q / "data" / "spxw_chain.parquet").exists():
            return q
    raise FileNotFoundError("repo root not found")

REPO = find_repo(Path.cwd())
OUT = REPO / "results" / "atm_straddle_0dte_1530"
OUT.mkdir(parents=True, exist_ok=True)
# derived-input cache: every entry is keyed on the source file's
# size+mtime (and the request), so a changed input can never serve a
# stale hit. Delete the directory to force a cold rebuild.
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
print("repo:", REPO)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 24)
pd.set_option("display.float_format", lambda x: f"{x: .6f}")
"""
    ),
    md(
        r"""
## 1. Load the option chain (15:30 and 16:00 ET only)

The source file, `data/spxw_chain.parquet`, holds a quote for every
30-minute time stamp. This book needs only the 15:30 quotes, which are
the entry prices; the 16:00 rows are kept solely to compare the tape's
underlying price against the official close. The load therefore reads
the time stamps first, keeps those two Eastern times of day, and only
then loads the remaining columns for those rows.
"""
    ),
    code(
        """
path = REPO / "data" / "spxw_chain.parquet"
COLS = [
    "expiration",
    "timestamp",
    "strike",
    "cp",
    "bid",
    "ask",
    "mid",
    "underlying_price",
    "impl_volatility",
]
_st = os.stat(path)
_ck = CACHE / f"chain_15301600_{_st.st_size}_{_st.st_mtime_ns}.parquet"
if _ck.exists():
    chain = pd.read_parquet(_ck)
    print(f"cache hit {_ck.name} (source size+mtime matched; "
          "full-file stamp scan skipped)")
else:
    ts = pd.to_datetime(pd.read_parquet(path, columns=["timestamp"])["timestamp"], utc=True)
    et = ts.dt.tz_convert("America/New_York")
    keep = ts[
        ((et.dt.hour == 15) & (et.dt.minute == 30)) | ((et.dt.hour == 16) & (et.dt.minute == 0))
    ].unique()
    print("unique timestamps in file", ts.nunique())
    print("15:30/16:00 ET stamps kept", len(keep))
    del ts, et
    chain = pd.read_parquet(path, columns=COLS, filters=[("timestamp", "in", list(keep))])
    chain["timestamp"] = pd.to_datetime(chain["timestamp"], utc=True)
    chain["expiration"] = pd.to_datetime(chain["expiration"])
    chain["cp"] = chain["cp"].astype(str).str.upper().str[0]
    for _old in CACHE.glob("chain_15301600_*.parquet"):
        _old.unlink()
    chain.to_parquet(_ck)
print("rows", f"{len(chain):,}")
print("columns", list(chain.columns))
print("timestamp", chain["timestamp"].min(), "->", chain["timestamp"].max())
print("expirations", chain["expiration"].nunique())
print(chain.head(3))
"""
    ),
    md(
        r"""
## 2. Times of day and the same-day-expiry flag

The vendor's time stamps are in UTC; the trading session runs on New
York time. A quote row is same-day expiry (0DTE) when the Eastern
calendar date of its time stamp equals the option's expiration date.
"""
    ),
    code(
        """
# ~3M rows share ~4k unique stamps: convert clocks once per unique
# value and take() back, instead of per-row strftime/tz work.
codes, uts = pd.factorize(chain["timestamp"])
assert (codes >= 0).all(), "null timestamp in chain"
uet = pd.DatetimeIndex(uts).tz_convert("America/New_York")
chain["et"] = uet.take(codes)
hh = np.where(
    (uet.hour == 15) & (uet.minute == 30), "15:30",
    np.where((uet.hour == 16) & (uet.minute == 0), "16:00", "other"),
)
chain["hhmm"] = hh[codes]
chain["et_date"] = uet.normalize().take(codes)
ecodes, uexp = pd.factorize(chain["expiration"])
assert (ecodes >= 0).all(), "null expiration in chain"
uexp_d = (
    pd.DatetimeIndex(uexp)
    .tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    .normalize()
)
chain["exp_date"] = uexp_d.take(ecodes)
chain["is_0dte"] = chain["et_date"] == chain["exp_date"]
print("0DTE rows", f"{int(chain['is_0dte'].sum()):,}", "/", f"{len(chain):,}")
print("ET hour counts on 0DTE:")
print(chain.loc[chain["is_0dte"], "et"].dt.hour.value_counts().sort_index().to_string())
print("15:30 0DTE stamps", chain.loc[chain["is_0dte"] & (chain["hhmm"] == "15:30"), "timestamp"].nunique())
print("16:00 0DTE stamps", chain.loc[chain["is_0dte"] & (chain["hhmm"] == "16:00"), "timestamp"].nunique())
"""
    ),
    md(
        r"""
## 3. Keep only same-day-expiry rows at those two times

The 15:30 quote is the entry. The 16:00 rows serve only the
tape-versus-close check; settlement uses the official close from
yfinance (`^GSPC`), never a 16:00 option quote. The load already dropped
the other hours; this step drops whatever is not an expiration-day row.
"""
    ),
    code(
        """
book_chain = chain[chain["is_0dte"] & chain["hhmm"].isin(["15:30", "16:00"])].copy()
print("rows", f"{len(book_chain):,}")
print(book_chain.groupby("hhmm").agg(rows=("strike", "size"), days=("expiration", "nunique"), stamps=("timestamp", "nunique")))
print(book_chain.head(3))
del chain
"""
    ),
    md(
        r"""
## 4. Nearest out-of-the-money call and put at 15:30

The index level $S$ is the 15:30 print. Among quotes with a finite,
positive midpoint:

- call: the smallest strike $K_c \ge S$
- put: the largest strike $K_p \le S$

The number of days that lose a leg is printed. The entry price is
$\mathrm{mid}(K_c) + \mathrm{mid}(K_p)$.
"""
    ),
    code(
        """
e = book_chain[book_chain["hhmm"] == "15:30"].copy()
n_days = e["expiration"].nunique()
live = e[np.isfinite(e["mid"]) & (e["mid"] > 0)].copy()
live["S"] = live["underlying_price"].astype(float)
spot = live.dropna(subset=["S"]).groupby("expiration")["S"].first()
print("15:30 0DTE days", n_days, "with a spot", int(spot.notna().sum()))
print("15:30 0DTE rows", f"{len(e):,}", "with finite mid>0", f"{len(live):,}",
      f"({len(live) / max(len(e), 1):.1%} of rows)")

c = live[live["cp"] == "C"].copy()
p = live[live["cp"] == "P"].copy()
c["S"] = c["expiration"].map(spot)
p["S"] = p["expiration"].map(spot)
c = c[np.isfinite(c["S"])]
p = p[np.isfinite(p["S"])]

c_otm = c[c["strike"] >= c["S"]].copy()
c_otm["k_gap"] = c_otm["strike"].astype(float) - c_otm["S"]
c_pick = (
    c_otm.sort_values(["expiration", "k_gap", "strike"])
    .groupby("expiration", as_index=False)
    .first()
)
p_otm = p[p["strike"] <= p["S"]].copy()
p_otm["k_gap"] = p_otm["S"] - p_otm["strike"].astype(float)
p_pick = (
    p_otm.sort_values(["expiration", "k_gap", "strike"])
    .groupby("expiration", as_index=False)
    .first()
)
print("days with a nearest-OTM call (K>=S, live mid)", len(c_pick))
print("days with a nearest-OTM put  (K<=S, live mid)", len(p_pick))
print("days missing OTM call", n_days - len(c_pick), "missing OTM put", n_days - len(p_pick))

atm = c_pick.merge(p_pick, on="expiration", suffixes=("_c", "_p"))
atm["S"] = atm["S_c"].astype(float)
atm["K_c"] = atm["strike_c"].astype(float)
atm["K_p"] = atm["strike_p"].astype(float)
atm["entry"] = atm["mid_c"].astype(float) + atm["mid_p"].astype(float)
atm["day"] = atm["et_c"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
atm["same_strike"] = atm["K_c"] == atm["K_p"]
print("days with both legs", len(atm), "dropped", n_days - len(atm))
print("frac same strike (spot on a listed K)", float(atm["same_strike"].mean()))
print("median K_c - S", float((atm["K_c"] - atm["S"]).median()),
      "median S - K_p", float((atm["S"] - atm["K_p"]).median()))
print(atm[["expiration", "S", "K_c", "K_p", "mid_c", "mid_p", "entry", "same_strike"]].head(8))
print("---")
print(atm[["S", "K_c", "K_p", "entry"]].describe())
"""
    ),
    md(
        r"""
## 5. Exit: cash settlement at the official close

Each leg settles in cash at its intrinsic value against the official
S&P 500 close (`^GSPC` from yfinance), not against a 16:00 option
quote:

- call: $\max(S_{\mathrm{close}}-K_c, 0)$
- put: $\max(K_p-S_{\mathrm{close}}, 0)$

If the close lands between $K_p$ and $K_c$, both legs expire worthless
and the settlement is zero. As a check, the gap between this close and
the 16:00 underlying on the option tape is printed.
"""
    ),
    code(
        """
def load_gspc_close(days: pd.Series) -> pd.Series:
    days = pd.to_datetime(days)
    start = pd.Timestamp(days.min()) - pd.Timedelta("7D")
    end = pd.Timestamp(days.max()) + pd.Timedelta("7D")
    cp = CACHE / "gspc_close.parquet"
    if cp.exists():
        cached = pd.read_parquet(cp)["close"]
        cached.index = pd.to_datetime(cached.index)
        if cached.index.min() <= pd.Timestamp(days.min()) and cached.index.max() >= pd.Timestamp(days.max()):
            print(f"cache hit {cp.name} (covers {cached.index.min().date()} .. {cached.index.max().date()})")
            return cached.astype(float)
    raw = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False, threads=True)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    ix = pd.to_datetime(close.index)
    if getattr(ix, "tz", None) is not None:
        ix = ix.tz_convert("America/New_York").tz_localize(None)
    close.index = ix.normalize()
    close = close.astype(float)
    close.rename("close").to_frame().to_parquet(cp)
    return close


atm = atm.reset_index(drop=True)
exp_day = pd.to_datetime(atm["expiration"])
if getattr(exp_day.dt, "tz", None) is not None:
    exp_day = exp_day.dt.tz_convert("America/New_York").dt.tz_localize(None)
exp_day = exp_day.dt.normalize()
atm["S_close"] = exp_day.map(load_gspc_close(exp_day))

x = book_chain[book_chain["hhmm"] == "16:00"]
s_tape = (
    x.dropna(subset=["underlying_price"])
    .groupby("expiration")["underlying_price"]
    .first()
)
atm["S_1600_tape"] = atm["expiration"].map(s_tape).astype(float)
gap = (atm["S_close"] - atm["S_1600_tape"]).abs()
print("yfinance ^GSPC close vs 16:00 chain underlying")
print(pd.DataFrame({
    "S_close": atm["S_close"],
    "S_1600_tape": atm["S_1600_tape"],
    "abs_gap": gap,
}).describe())
print("days missing yfinance close", int(atm["S_close"].isna().sum()))
print("median |close - tape|", float(gap.median()) if gap.notna().any() else float("nan"))

atm["pay_c"] = np.maximum(atm["S_close"] - atm["K_c"], 0.0)
atm["pay_p"] = np.maximum(atm["K_p"] - atm["S_close"], 0.0)
atm["exit"] = atm["pay_c"] + atm["pay_p"]
print(atm[["expiration", "S", "K_c", "K_p", "S_close", "S_1600_tape", "pay_c", "pay_p", "entry", "exit"]].head(8))
print("frac settlement 0 (close between K_p and K_c)", float((atm["exit"] == 0).mean()))
atm = atm[np.isfinite(atm["entry"]) & np.isfinite(atm["exit"]) & (atm["entry"] > 0)].copy()
"""
    ),
    md(
        r"""
## 6. Return $R = \mathrm{exit}/P_{15:30} - 1$

The package costs its entry price at 15:30 and pays the two legs'
settlement values in cash at the close; $R$ is that payout divided by
the entry price, minus one.
"""
    ),
    code(
        """
atm["R"] = atm["exit"] / atm["entry"] - 1.0
atm = atm.set_index("day").sort_index()
print(atm[["expiration", "K_c", "K_p", "entry", "exit", "R"]].head(8))
print("---")
print(atm["R"].describe())
print("n", len(atm))
"""
    ),
    md(
        r"""
## 7. Variance forecasts

Seven forecasting models are read in, one stored table each, all produced by the paper's forecasting pipeline on the same panel:

- the HAR-plus-calendar least-squares forecast;
- the block-diagonal ridge forecast, the paper's headline model, with the FOMC calendar columns in its design;
- LightGBM and XGBoost, each on the wide all-features design with a frozen menu of settings;
- the lasso on the same all-features design, with its penalty chosen from past data only (the paper's protocol);
- the lasso at a fixed penalty of $10^{-4}$, set by hand — the centre of the tuning envelope rather than the paper's head-to-head entry;
- the elastic net, with its penalties chosen from past data only.

Each stored forecast is on the scale the models were fitted on, $y=\sqrt{RV/B}$ with $B$ the time-of-day profile, and winsorized. It is mapped back to a raw 30-minute variance by the Mincer–Zarnowitz recalibration: a regression of realized on forecast that maps the forecast onto the realized scale, including its variance term. Over the trailing 250 days $[t-250,t)$ the line $m=a+b\,\hat y$ and its residual variance $\hat\sigma^2$ are fitted against the unwinsorized realized value $y^{\mathrm{raw}}=\sqrt{RV^{\mathrm{raw}}/B}$, by weighted least squares with weights $1/\max(\hat y, q_{10})^2$, where $q_{10}$ is the tenth percentile of $\hat y$ within the window. The variance forecast is then $\widehat{RV}=(m^2+\hat\sigma^2)\,B$: the expected realized variance of the 15:30–16:00 bar. Nothing observed after the decision time enters the fit.

**Which forecast, and which bars it is fitted on.** The forecast table labels each half-hour bar by the time it ends: the row labelled $\tau$ carries the realized variance of the bar from $\tau-30$ to $\tau$ together with the forecast of that bar, which was issued at $\tau-30$. The 15:30 book therefore uses the row labelled 16:00 — the forecast issued at 15:30 for the very bar it trades. An earlier version of this notebook used the row labelled 15:30, which is the forecast of the 15:00–15:30 bar: it uses nothing observed after the decision time, but it is one bar stale. The recalibration is fitted only on the bars the paper scores, from 10:00 to 16:00; bars outside that window are mispredicted by a factor of roughly 50 to 100 and, when they were included, distorted the calibration (the mean ratio of forecast to realized variance fell from 1.14 to 1.08 once they were excluded). The fitted coefficients are then applied to every row.

Two practical notes. The two-parameter fit is solved in closed form, one day at a time, in a routine shared with the intraday notebook. Each model's 15:30 table is cached and keyed to its inputs — the source file, the fitting conventions, and the set of option days — so a re-run with unchanged inputs skips the computation, and any change to the inputs forces it to be redone. The seven tables are loaded in parallel.
"""
    ),
    code(
        """
from concurrent.futures import ThreadPoolExecutor

YHATS = {
    "a0": REPO / "results" / "spxw_pnl" / "yhat_a0.parquet",
    "blk2": REPO / "results" / "spxw_pnl" / "yhat_blk2_fomc1.parquet",
    "lgbm": REPO / "results" / "spxw_pnl" / "yhat_tree00.parquet",
    "xgb": REPO / "results" / "spxw_pnl" / "yhat_tree16.parquet",
    "lasso_t": REPO / "results" / "spxw_pnl" / "yhat_b2lasso_tuned.parquet",
    "lasso_f": REPO / "results" / "spxw_pnl" / "yhat_b2lasso.parquet",
    "enet": REPO / "results" / "spxw_pnl" / "yhat_b3enet_tuned.parquet",
}
# The second-order map (MZ smear) lives in atm_straddle_lib: flat
# 250-day window, fit restricted to the scored session bars
# (10:00-15:30 ET) so off-session dynamics cannot pollute the
# calibration. Delegate rather than duplicate.
import sys
sys.path.insert(0, str(REPO / "notebooks"))
import atm_straddle_lib as asl

second_order_raw = asl.second_order_raw
load_yhat_1530 = asl.load_yhat_1530


def load_yhat_1530_cached(tag: str, path: Path, need_dates) -> pd.DataFrame:
    h = hashlib.sha1()
    st = os.stat(path)
    h.update(f"v6-vec-gls:{st.st_size}:{st.st_mtime_ns}:flat{asl.WINDOW_DAYS}".encode())
    for d in sorted(need_dates):
        h.update(str(d).encode())
    cp = CACHE / f"yhat1530_{tag}_{h.hexdigest()[:16]}.parquet"
    if cp.exists():
        return pd.read_parquet(cp)
    out = load_yhat_1530(path, need_dates)
    for old in CACHE.glob(f"yhat1530_{tag}_*.parquet"):
        old.unlink()
    out.to_parquet(cp)
    return out


need_dates = set(atm.index)
with ThreadPoolExecutor(max_workers=len(YHATS)) as pool:
    futs = {
        tag: pool.submit(load_yhat_1530_cached, tag, path, need_dates)
        for tag, path in YHATS.items()
    }
    models = {tag: futs[tag].result() for tag in YHATS}

LABEL = {
    "a0": "HAR + calendar OLS",
    "blk2": "block-diag ridge",
    "lgbm": "LightGBM",
    "xgb": "XGBoost",
    "lasso_t": "lasso (causally tuned)",
    "lasso_f": "lasso (fixed 1e-4)",
    "enet": "elastic net (causally tuned)",
}
print("option days", len(atm), pd.Timestamp(atm.index.min()), "->", pd.Timestamp(atm.index.max()))
for tag, rv in models.items():
    n_ov = atm.index.intersection(rv.index).nunique()
    print(f"{LABEL[tag]}: 15:30 days {len(rv)}  {pd.Timestamp(rv.index.min())} -> {pd.Timestamp(rv.index.max())}  "
          f"days in both {n_ov}  option days with no forecast {len(atm) - n_ov}")
    print(rv.head(3)[["yhat", "baseline", "rv_hat", "rv_raw"]])
    print("median rv_hat", float(rv["rv_hat"].median()),
          "median rv_raw", float(rv["rv_raw"].median()))
    print("---")
"""
    ),
    md(
        r"""
## 8. Putting implied volatility on the same footing as realized variance

The option chain carries a vendor implied-volatility column. The vendor's documentation describes it as an **annualized** Black–Scholes volatility, which at the money would be about $0.20$. The values on the tape are about $0.002$ at the money — roughly a hundred times too small for that reading — yet the midpoint quotes price roughly 20% annualized volatility. The units resolve if the number is read as a **one-hour standard deviation** rather than an annualized volatility:

$$
\mathrm{Var}(1\mathrm{h})\approx 0.002^{2}=4\times 10^{-6},\qquad
\mathrm{IV}_{30}=\mathrm{IV}_{\mathrm{hourly}}/\sqrt{2},\qquad
\mathrm{iv\_var}=\mathrm{IV}_{30}^{2}=(\mathrm{IV}_{\mathrm{hourly}})^{2}/2.
$$

A variance of order $10^{-6}$ is the same scale as the realized-variance forecast $\widehat{RV}$. Reading $0.002$ as annualized instead and dividing by $252\times 6.5$ hours gives about $10^{-9}$, which cannot be compared with $\widehat{RV}$ at all. No Black–Scholes inversion of the midpoint quote is performed here. Because the window remaining at 15:30 is thirty minutes, this quantity is also the implied variance of the remainder of the session.
"""
    ),
    code(
        """
atm["iv_c"] = pd.to_numeric(atm["impl_volatility_c"], errors="coerce")
atm["iv_p"] = pd.to_numeric(atm["impl_volatility_p"], errors="coerce")
atm["iv_hourly"] = atm[["iv_c", "iv_p"]].mean(axis=1)
atm["iv_30"] = atm["iv_hourly"] / np.sqrt(2.0)
atm["iv_var"] = atm["iv_30"] ** 2
print(atm[["S", "K_c", "K_p", "entry", "iv_hourly", "iv_30", "iv_var"]].head(8))
print("---")
print(atm[["iv_hourly", "iv_30", "iv_var"]].describe())
print("days missing quoted IV", int(atm["iv_hourly"].isna().sum()))
"""
    ),
    md(
        r"""
## 9. The signal and the long–short volatility position

Both quantities are now in variance units. The signal is the gap between the forecast and the implied variance,
$s_t=\widehat{RV}_t-\bigl(\mathrm{IV}_{\mathrm{hourly},t}/\sqrt{2}\bigr)^{2}$.
The position is long the package when the signal is positive and short otherwise: $q_t=+1$ if $s_t>0$ and $q_t=-1$ if not. The position's return is $q_t R_t$.
"""
    ),
    code(
        """
def make_book(rv: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    px = atm.join(rv[["rv_hat"]], how="inner")
    px = px.dropna(subset=["R", "rv_hat", "iv_var", "entry"])
    px = px[(px["rv_hat"] > 0) & (px["iv_var"] > 0)]
    px["signal"] = px["rv_hat"] - px["iv_var"]
    px["pos"] = np.where(px["signal"] > 0, 1.0, -1.0)
    px["R_p"] = px["pos"] * px["R"]
    mu = float(px["R_p"].mean())
    sd = float(px["R_p"].std())
    r_short = -px["R"]
    stats = pd.Series({
        "n_days": len(px),
        "frac_long": float((px["pos"] > 0).mean()),
        "frac_short": float((px["pos"] < 0).mean()),
        "mean_Rp": mu,
        "Sharpe_ann": mu / sd * np.sqrt(252.0) if sd > 0 else float("nan"),
        "mean_Rp always-short": float(r_short.mean()),
        "Sharpe_ann always-short": float(r_short.mean() / r_short.std() * np.sqrt(252)),
        "start": pd.Timestamp(px.index.min()),
        "end": pd.Timestamp(px.index.max()),
    })
    return px, stats

books, stat_cols = {}, {}
for tag, rv in models.items():
    px, st = make_book(rv)
    books[tag] = px
    stat_cols[tag] = st
    px.to_parquet(OUT / f"daily_{tag}.parquet")
    print("wrote", OUT / f"daily_{tag}.parquet")

print(pd.DataFrame(stat_cols).rename(columns=LABEL).to_string())
print("---")
print("hand-check HAR+calendar (R_p = R if pos==1, else -R):")
print(books["a0"][["entry", "exit", "R", "rv_hat", "iv_hourly", "iv_30", "iv_var", "signal", "pos", "R_p"]].head(8))
"""
    ),
    # Unit-median VRP, long-books, book-variants, lesson, return-summary omitted.
    md(
        r"""
## 10. Rule table, grouped by strategy

Each rule is scored on the same days and on the same long-package return $R$ (midpoint quote at 15:30 to cash settlement); only the position $q_t$ differs. There is one block per rule, with **the seven forecasting models as rows**, scored on the days common to all seven books. Fills are at the midpoint quote. The always-short rule uses no forecast, so its seven rows are identical and it is shown as a single row.

**The rules** (each returns $R'_t = q_t R_t$):

- **always short:** $q_t=-1$ every day; no forecast is used.
- **long–short volatility:** $q_t=\mathrm{sign}(s_t)$ with
  $s_t=\widehat{RV}_t-\mathrm{IV}_{30,t}^{2}$ — long the package when the
  forecast exceeds implied variance, short otherwise.

The columns are the usual summary statistics (count, mean, standard deviation, minimum, quartiles, maximum), skewness, excess kurtosis, the $t$-statistic of the mean, $t=\sqrt{n}\cdot\mathrm{mean}/\mathrm{std}$, and the count and share of buy days (days with $q_t>0$). The $t$-statistic uses the raw daily mean and standard deviation, not the annualized Sharpe ratio.

Excess kurtosis (the `ex_kurt` column) follows Fisher's definition: the fourth standardized moment minus 3, so a normal distribution scores 0 rather than 3. Positive values mean fatter tails than a normal with the same variance; ordinary (Pearson) kurtosis is this value plus 3.

The annualized Sharpe ratio (the `Sharpe_ann` column) is $\mathrm{mean}/\mathrm{std}\times\sqrt{252}$ on the daily $R'$; every other moment column is daily and unannualized. At fixed $n$ the $t$-statistic and the Sharpe ratio carry the same information ($t=\mathrm{Sharpe}\times\sqrt{n/252}$); both are shown so the table can be read either way.

Only the scoring is restricted to the common days; each book itself is built on all of its own days.
"""
    ),
    code(
        r"""
def rule_sizes(px: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "always short": pd.Series(-1.0, index=px.index),
        "long-short volatility": px["pos"],
    }


def rule_row(r: pd.Series, size: pd.Series) -> pd.Series:
    r = pd.Series(r).astype(float)
    size = pd.Series(size, index=r.index).astype(float)
    x = r.dropna()
    n = int(len(x))
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    n_buy = int((size > 0).sum())
    n_sz = int(size.notna().sum())
    return pd.Series({
        "n": n,
        "mean": mu,
        "std": sd,
        "min": float(x.min()),
        "25%": float(x.quantile(0.25)),
        "50%": float(x.median()),
        "75%": float(x.quantile(0.75)),
        "max": float(x.max()),
        "skew": float(x.skew()),
        "ex_kurt": float(x.kurt()),
        "t_mean": mu / sd * np.sqrt(n) if sd > 0 else float("nan"),
        "Sharpe_ann": mu / sd * np.sqrt(252.0) if sd > 0 else float("nan"),
        "n_buy": n_buy,
        "pct_buy": 100.0 * n_buy / n_sz if n_sz else float("nan"),
    })


order = [
    "always short",
    "long-short volatility",
]
cols = ["n", "mean", "std", "min", "25%", "50%", "75%", "max",
        "skew", "ex_kurt", "t_mean", "Sharpe_ann", "n_buy", "pct_buy"]
MODEL_ORDER = ["a0", "blk2", "lgbm", "xgb", "lasso_t", "lasso_f", "enet"]

common = None
for tag in MODEL_ORDER:
    idx = books[tag].index
    common = idx if common is None else common.intersection(idx)
common = common.sort_values()
print("days per model:", {LABEL[t]: len(books[t]) for t in MODEL_ORDER})
print("common days:", len(common),
      pd.Timestamp(common.min()), "->", pd.Timestamp(common.max()))

rule_tabs = {}
for tag in MODEL_ORDER:
    px = books[tag]
    sizes = rule_sizes(px)
    rule_tabs[tag] = pd.DataFrame({
        name: rule_row((sizes[name] * px["R"]).loc[common], sizes[name].loc[common])
        for name in order
    }).T[cols]
    rule_tabs[tag].to_csv(OUT / f"rule_table_{tag}.csv")

for name in order:
    if name == "always short":
        # takes no forecast: every model row is identical, so print one
        # anonymous row (and assert the identity instead of trusting it)
        base = rule_tabs[MODEL_ORDER[0]].loc[name]
        for tag in MODEL_ORDER[1:]:
            assert np.allclose(rule_tabs[tag].loc[name], base), name
        tab = pd.DataFrame({"all models": base}).T[cols]
    else:
        tab = pd.DataFrame(
            {LABEL[tag]: rule_tabs[tag].loc[name] for tag in MODEL_ORDER}
        ).T[cols]
    print(name)
    print(tab.to_string())
    print("---")
    safe = "".join(ch if ch.isalnum() else "_" for ch in name)
    tab.to_csv(OUT / f"rule_by_strategy_{safe}.csv")
print("saved per-model rule_table_*.csv and per-rule rule_by_strategy_*.csv in", OUT)

fig, axes = plt.subplots(1, len(order), figsize=(14, 3.6), sharex=True, sharey=True)
px = books["blk2"]
sizes = rule_sizes(px)
pooled = pd.concat([sizes[name] * px["R"] for name in order], axis=0).dropna()
lo, hi = float(pooled.quantile(0.01)), float(pooled.quantile(0.99))
bins = np.linspace(lo, hi, 41)
for ax, name in zip(axes.ravel(), order):
    x = (sizes[name] * px["R"]).dropna()
    ax.hist(x.clip(lo, hi), bins=bins, color="C0", edgecolor="none")
    ax.axvline(0.0, color="k", lw=0.6)
    ax.set_title(name, fontsize=8)
    ax.set_xlabel(r"$R'$")
fig.suptitle("block-diag ridge, mid  $R'$  (1–99% window)")
fig.tight_layout()
fig.savefig(OUT / "rule_hists_blk2.png", dpi=120, bbox_inches="tight")
print("saved", OUT / "rule_hists_blk2.png")
display(fig)
plt.close(fig)
"""
    ),
    # Event-filter (FOMC+ME flat) cells omitted from generation.
    md(
        r"""
## 11. Profit and loss, without compounding

The contracts and positions are those of the rule table. Every series below is a **daily arithmetic** return or profit, summed rather than compounded. The published book is filled at the midpoint quote. Paying the full spread means a long pays the ask and a short receives the bid; a half-spread cost variant charges $\tfrac12(\mathrm{ask}-\mathrm{bid})$ against each trade. Profit in index points is $q(\mathrm{exit}-\mathrm{entry})$, and dollar profit applies the SPXW multiplier of $100$. The margin-scaled return divides dollar profit by an exchange-style short-straddle margin on short days and by the premium paid on long days. The plot is the cumulative sum of dollar profit.
"""
    ),
    code(
        r"""
import sys
sys.path.insert(0, str(REPO / "notebooks"))
import atm_straddle_lib as asl

px = books["blk2"].loc[common].copy()
if "bid_entry" not in px.columns:
    px["bid_entry"] = px["bid_c"].astype(float) + px["bid_p"].astype(float)
    px["ask_entry"] = px["ask_c"].astype(float) + px["ask_p"].astype(float)
hs = 0.5 * (px["ask_entry"] - px["bid_entry"])
sizes = rule_sizes(books["blk2"])
rows = []

def add_variant(series, q, rule, variant, unit):
    st = asl.rule_row(series, q)
    rows.append({"rule": rule, "variant": variant, "unit": unit, **st.to_dict()})

for name in ("always short", "long-short volatility"):
    q = sizes[name].loc[common]
    mid = (q * px["R"])
    add_variant(mid, q, name, "mid premium R", "return")
    signq = np.sign(q.replace(0, -1.0))
    crossed = asl.crossed_premium_return(signq, px["exit"], px["bid_entry"], px["ask_entry"]) * q.abs()
    add_variant(crossed, q, name, "crossed fill", "return")
    trade = px["entry"] + signq * hs
    tc = q * (px["exit"] - trade) / px["entry"]
    add_variant(tc, q, name, "half-spread TC", "return")
    pts = asl.points_pnl(q, px["exit"], px["entry"])
    add_variant(pts, q, name, "index-point P&L", "points")
    usd = pts * asl.SPX_MULTIPLIER
    add_variant(usd, q, name, "dollar P&L", "USD")
    margin_pts = [
        asl.cboe_short_straddle_margin_points(S, Kc, Kp, ent)
        for S, Kc, Kp, ent in zip(px["S"], px["K_c"], px["K_p"], px["entry"])
    ]
    margin_pts = pd.Series(margin_pts, index=px.index)
    capital = np.where(q < 0, margin_pts * asl.SPX_MULTIPLIER, px["entry"] * asl.SPX_MULTIPLIER)
    mret = usd / np.maximum(capital, 1e-8)
    add_variant(pd.Series(mret, index=px.index), q, name, "margin-scaled", "return")

var_tab = pd.DataFrame(rows)
print(var_tab.to_string(index=False))
var_tab.to_csv(OUT / "pnl_variants_blk2.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 3.4))
for name, ls in (("always short", "-"), ("long-short volatility", "--")):
    q = sizes[name].loc[common]
    ax.plot(
        px.index,
        asl.points_pnl(q, px["exit"], px["entry"]).cumsum() * asl.SPX_MULTIPLIER,
        ls, lw=1.2, label=f"{name} mid",
    )
    signq = np.sign(q.replace(0, -1.0))
    crossed_usd = (asl.crossed_premium_return(signq, px["exit"], px["bid_entry"], px["ask_entry"]) * q.abs() * px["entry"] * asl.SPX_MULTIPLIER)
    ax.plot(px.index, crossed_usd.cumsum(), ls, lw=1.0, alpha=0.7, label=f"{name} crossed")
ax.set_title("blk2 — cumulative $ P&L (non-compounded)")
ax.set_ylabel("USD")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "pnl_cum_usd_blk2.png", dpi=120, bbox_inches="tight")
print("saved", OUT / "pnl_cum_usd_blk2.png")
display(fig)
plt.close(fig)
"""
    ),
    md(
        r"""
## 12. Information ratio against always-short

The benchmark is the always-short book, $R^{\mathrm{AS}}_t=-R_t$: one short package every day. The portfolio is the long–short book, $R^p_t=q_t R_t$ with $q_t=\mathrm{sign}(s_t)$.

The **active return** is the daily difference $R^a_t=R^p_t-R^{\mathrm{AS}}_t$. On short days $q_t=-1$ and the two books coincide, so $R^a_t=0$. On buy days the position has flipped from short to long, so $R^a_t=q_tR_t-(-R_t)=(q_t+1)R_t$, which equals $2R_t$ for a $\pm1$ position. The series is those daily differences on the 871 common days.

The table's columns are:

- the **mean active return** (`mean_active`), the sample mean of that daily series, not annualized;
- the **tracking error**, the standard deviation of the active return with $n-1$ in the denominator, reported daily (`te_daily`) and annualized by $\sqrt{252}$ (`te_ann`);
- the **annualized information ratio** (`IR_ann`), $\overline{R^a}/\mathrm{std}(R^a)\times\sqrt{252}$, which equals the mean active return over the annualized tracking error — the Sharpe ratio of the *active* series, not of $R^p$;
- the **$t$-statistic of the active return** (`t_active`), $\overline{R^a}/\mathrm{std}(R^a)\times\sqrt{n}$, the usual test that the mean active return is zero — the same algebra as the $t$-statistic in the rule table, applied to $R^a$; the information ratio and this $t$ move together, but $t$ is not annualized;
- the **correlation to the benchmark** (`corr_to_bench`), $\mathrm{corr}(R^p,R^{\mathrm{AS}})$.
"""
    ),
    code(
        r"""
ir_rows = []
for tag in MODEL_ORDER:
    px = books[tag]
    sizes = rule_sizes(px)
    bench = (sizes["always short"] * px["R"]).loc[common]
    for name in ("long-short volatility",):
        port = (sizes[name] * px["R"]).loc[common]
        st = asl.information_ratio(port, bench)
        ir_rows.append({"model": LABEL[tag], "rule": name, **st.to_dict()})
ir_tab = (
    pd.DataFrame(ir_rows)
    .set_index(["rule", "model"])
    .sort_values(by="IR_ann", ascending=False)
    .sort_index(level="rule", sort_remaining=False)
)
print(ir_tab.to_string())
ir_tab.to_csv(OUT / "information_ratio_vs_always_short.csv")
print("IR = active return / tracking error; benchmark is always-short.")
"""
    ),
    md(
        r"""
## 13. Does the 15:30 signal predict the settlement return?

The signal is fixed at 15:30 and the package settles at 16:00 the same
day, so the natural test pairs $s_t$ with $R_t$: every input to the
signal is known before the position is taken, and nothing from the
settlement enters it. This is the relation the position rule of §9
trades, measured here as a regression object rather than as a book.
Six checks are printed, all with Newey–West standard errors (six lags)
and no tuning parameters; the percentile rank of $s_t$ is taken among
all days up to and including $t$ once 63 days of history exist, so it
compares today's signal only with the past.

- **Split by sign.** The mean of $R_t$ on days with $s_t>0$ against
  days with $s_t\le 0$, and the $t$-statistic of the difference. This
  is the long–short edge itself.
- **Top third against bottom third** of the rank of $s_t$: the same
  question restricted to the extremes.
- **Rank correlations.** Spearman's $\rho$ and Kendall's $\tau$
  between $s_t$ and $R_t$, with both sides ranked.
- **Regression on the percentile rank** of $s_t$: a monotone relation
  without leverage from extreme values of the signal.
- **Plain least squares** of $R_t$ on the raw signal, shown for
  completeness as the weak instrument it is.
- **Size of the move.** $|R_t|$ on the rank of $s_t$, alone and with
  today's log implied variance added.

The same-day content is a sign effect, not a slope. For the
block-diagonal ridge forecast the package returns about $+0.12$ on
days the forecast sits above implied variance and about $-0.11$ on the
others — a spread near $0.23$ with $t$ near $2.9$ — and the top third
of the ranked signal against the bottom third gives about the same
spread and $t$; every one of the seven forecasts shows the split, with
$t$ between about $1.9$ and $3.3$. The rank regression sees it too
($t$ around $2.6$ for the headline forecast, positive for all seven).
But the rank correlations are essentially zero ($\rho$ within
$\pm 0.02$, $p$ above $0.6$), and the raw least-squares slope is null
for the headline forecast ($t$ about $-0.5$; two of the tree forecasts
show a positive slope with $t$ above $3$, at an $R^2$ below $0.001$).
The reason is the shape of $R$: it has a point mass at exactly $-1$ on
the days the package expires worthless and a long right tail, so the
relation lives in the means of two groups — which side of the quoted
variance the forecast stands on — and not in a monotone ordering of
days. A slope, or a rank-against-rank correlation, is the wrong
instrument for that payoff; the sign split is the right one.

The size of the move is a separate matter. The rank of $s_t$ predicts
$|R_t|$ strongly ($t$ above $4$ for every forecast), but most of that
is the level of implied variance: the rank is high when options are
cheap relative to the forecast, and a cheap straddle moves more per
dollar of premium. With today's log implied variance in the
regression the rank's $t$ falls from about $4.3$ to about $1.7$ for
the headline forecast (to between about $1.3$ and $2.5$ across the
seven), while implied variance itself enters with $t$ near $-3.8$. It
is a price-level effect more than a variance forecast.

**Does yesterday's signal carry anything?** The second half of the
cell repeats the same checks with $s_{t-1}$ in place of $s_t$, as a
persistence check. It finds nothing: the sign split is flat (spread
within $\pm 0.03$, $|t|$ below $0.35$ for every forecast), the rank
regression and the tercile spread are insignificant ($t$ below about
$1.3$), and the rank correlations sit at zero. The reason is printed
at the end of the cell. The signal is the gap between two highly
persistent quantities — the variance forecast and the implied
variance — whose difference is largely renewed each day as the market
re-prices the afternoon's variance: yesterday's sign repeats today
about 65% of the time, against a 52% baseline if signs were
independent, and a book traded on yesterday's sign earns a Sharpe
ratio near $0.2$ against about $1.6$ on the same-day sign. The lagged
rank still predicts $|R_t|$, for the same price-level reason — with
yesterday's implied variance added, its $t$ falls from about $3.7$ to
about $0.6$ — and the large moves it flags fall mostly on buy days
(over half of the top-third days, against under 40% overall), where a
large settlement move is the profit. The edge is a same-afternoon
comparison; yesterday's is stale by the time the market re-prices.
"""
    ),
    code(
        r"""
from scipy import stats as sps


def hac_fit(y, x):
    # least squares with Newey-West (six-lag) standard errors
    return sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 6})


def battery(x_raw, x_rank, r, iv_log, label):
    # the same checks for whichever version of the signal is passed in:
    # x_raw = the signal, x_rank = its percentile rank, iv_log = log implied variance dated like x
    ok_raw = np.isfinite(x_raw) & np.isfinite(r)
    ok_rk = np.isfinite(x_rank) & np.isfinite(r)
    xr, yr = x_raw[ok_raw].to_numpy(), r[ok_raw].to_numpy()
    rkv, rv_ = x_rank[ok_rk].to_numpy(), r[ok_rk].to_numpy()
    out = {}
    # mean return split by the sign of the signal
    ind = (xr > 0).astype(float)
    fs = hac_fit(yr, ind)
    out["sign"] = {"model": label,
                   "mean_R|s>0": float(yr[ind == 1].mean()),
                   "mean_R|s<=0": float(yr[ind == 0].mean()),
                   "diff": float(fs.params[1]), "t_diff": float(fs.tvalues[1]),
                   "n_pos": int(ind.sum()), "n_nonpos": int((1 - ind).sum())}
    # top third against bottom third of the rank
    hi = rkv > 2.0 / 3.0
    lo = rkv < 1.0 / 3.0
    sel = hi | lo
    ft = hac_fit(rv_[sel], hi[sel].astype(float))
    out["tercile"] = {"model": label,
                      "mean_R_top": float(rv_[hi].mean()),
                      "mean_R_bottom": float(rv_[lo].mean()),
                      "diff": float(ft.params[1]), "t_diff": float(ft.tvalues[1]),
                      "n_top": int(hi.sum()), "n_bottom": int(lo.sum())}
    # rank association
    sp_rho, sp_p = sps.spearmanr(xr, yr)
    kt_tau, kt_p = sps.kendalltau(xr, yr)
    out["assoc"] = {"model": label,
                    "spearman_rho": float(sp_rho), "p_sp": float(sp_p),
                    "kendall_tau": float(kt_tau), "p_kt": float(kt_p),
                    "n": int(len(xr))}
    # regression on the percentile rank
    fit = hac_fit(rv_, rkv)
    out["rank"] = {"model": label, "b_rank": float(fit.params[1]),
                   "t": float(fit.tvalues[1]), "p": float(fit.pvalues[1]), "n": int(fit.nobs)}
    # plain least squares on the raw signal
    fit = hac_fit(yr, xr)
    out["raw"] = {"model": label,
                  "a": float(fit.params[0]), "b": float(fit.params[1]),
                  "t_b": float(fit.tvalues[1]), "p_b": float(fit.pvalues[1]),
                  "R2": float(fit.rsquared), "n": int(fit.nobs)}
    # size of the move on the rank, alone and with the level of implied variance
    fm = hac_fit(np.abs(rv_), rkv)
    out["mag"] = {"model": label, "b_absR": float(fm.params[1]),
                  "t": float(fm.tvalues[1]), "p": float(fm.pvalues[1]),
                  "n": int(fm.nobs)}
    ok3 = ok_rk & np.isfinite(iv_log)
    f3 = hac_fit(np.abs(r[ok3].to_numpy()),
                 np.column_stack([x_rank[ok3].to_numpy(), iv_log[ok3].to_numpy()]))
    out["mag_iv"] = {"model": label,
                     "t_rank_alone": float(fm.tvalues[1]),
                     "t_rank_with_iv": float(f3.tvalues[1]),
                     "t_iv": float(f3.tvalues[2]), "n": int(f3.nobs)}
    return out


same = {k: [] for k in ("sign", "tercile", "assoc", "rank", "raw", "mag", "mag_iv")}
lag = {k: [] for k in same}
fig_inputs = None
for tag in MODEL_ORDER:
    px = books[tag].loc[common]
    s = px["signal"].astype(float)
    r = px["R"].astype(float)
    iv_log = np.log(px["iv_var"].astype(float))
    rk_same = s.expanding(min_periods=63).rank(pct=True)   # today's signal ranked among all days up to today
    out_same = battery(s, rk_same, r, iv_log, LABEL[tag])
    out_lag = battery(s.shift(1), rk_same.shift(1), r, iv_log.shift(1), LABEL[tag])
    for k in same:
        same[k].append(out_same[k])
        lag[k].append(out_lag[k])
    if tag == "blk2":
        fig_inputs = (s, rk_same, r)

print("=== SAME DAY: the 15:30 signal against the 16:00 settlement return ===")
for key, title in (("sign", "(1) sign split: mean R_t by sign(s_t)"),
                   ("tercile", "(2) top third vs bottom third of the rank of s_t"),
                   ("assoc", "(3) rank association: Spearman / Kendall of (s_t, R_t)"),
                   ("rank", "(4) regression of R_t on the percentile rank of s_t"),
                   ("raw", "(5) plain least squares of R_t on raw s_t (a weak instrument for a sign effect)"),
                   ("mag", "(6) size of the move: |R_t| on the rank of s_t"),
                   ("mag_iv", "(6) ... with today's log implied variance added")):
    print(title)
    print(pd.DataFrame(same[key]).to_string(index=False))
    print("---")
same_tab = pd.concat({k: pd.DataFrame(v) for k, v in same.items()}, names=["statistic", None]).reset_index(level=0)
same_tab.to_csv(OUT / "sameday_battery.csv", index=False)

print("=== PERSISTENCE CHECK: yesterday's signal against today's return ===")
reg_tab = pd.DataFrame([dict(row, x="raw s (same day)") for row in same["raw"]]
                       + [dict(row, x="raw s (t-1)") for row in lag["raw"]])
reg_tab = reg_tab[["model", "x", "a", "b", "t_b", "p_b", "R2", "n"]]
print("plain least squares, both versions of the signal:")
print(reg_tab.to_string(index=False))
reg_tab.to_csv(OUT / "regression_R_on_signal.csv", index=False)
print("---")
for key, title, fname in (("rank", "regression of R_t on the rank of s_{t-1}", "lagged_robust_rank.csv"),
                          ("sign", "sign split: mean R_t by sign(s_{t-1})", "lagged_robust_sign.csv"),
                          ("assoc", "rank association of (s_{t-1}, R_t)", "lagged_robust_assoc.csv"),
                          ("tercile", "top third vs bottom third of the rank of s_{t-1}", "lagged_robust_tercile.csv"),
                          ("mag", "size of the move: |R_t| on the rank of s_{t-1}", "lagged_robust_magnitude.csv")):
    print(title)
    print(pd.DataFrame(lag[key]).to_string(index=False))
    print("---")
    pd.DataFrame(lag[key]).to_csv(OUT / fname, index=False)
print("size of the move on the lagged rank, with yesterday's log implied variance added:")
print(pd.DataFrame(lag["mag_iv"]).to_string(index=False))
print("---")

# why yesterday's signal carries nothing (block-diagonal ridge): the gap is renewed daily
px = books["blk2"].loc[common]
sg = np.sign(px["signal"].astype(float))
prev = sg.shift(1)
ok = np.isfinite(prev)
persist = float((sg[ok] == prev[ok]).mean())
p_pos = float((sg[ok] > 0).mean())
print(f"sign persistence: P[sign(s_t) = sign(s_t-1)] = {persist:.3f} "
      f"vs {p_pos**2 + (1 - p_pos)**2:.3f} if signs were independent")
r_lag = (prev * px["R"])[ok]
r_same = (sg * px["R"])[ok]
f_lag = sm.OLS(r_lag.to_numpy(), np.ones((len(r_lag), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
print(f"book traded on yesterday's sign: mean {float(r_lag.mean()):+.4f}, Newey-West t {float(f_lag.tvalues[0]):+.2f}, "
      f"Sharpe {float(r_lag.mean() / r_lag.std(ddof=1) * np.sqrt(252)):+.2f} | same-day sign: "
      f"Sharpe {float(r_same.mean() / r_same.std(ddof=1) * np.sqrt(252)):+.2f}")
rk_lag = px["signal"].astype(float).expanding(min_periods=63).rank(pct=True).shift(1)
absr = px["R"].abs()
buy = px["pos"] > 0
top = rk_lag > 2 / 3
print(f"mean |R| on buy days {float(absr[buy].mean()):.3f} vs sell days {float(absr[~buy].mean()):.3f}; "
      f"share of buy days among top-third lagged-rank days {float(buy[top].mean()):.2f} "
      f"vs {float(buy[np.isfinite(rk_lag)].mean()):.2f} overall")

s, rk_same, r = fig_inputs
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, x, xlab, ttl in (
    (axes[0], s, r"$s_t=\widehat{RV}_t-\mathrm{IV}_{30,t}^2$ at 15:30", "the 15:30 signal, raw"),
    (axes[1], rk_same, r"percentile rank of $s_t$ among days up to $t$", "the 15:30 signal, percentile rank"),
):
    ok = np.isfinite(x) & np.isfinite(r)
    ax.scatter(x[ok], r[ok], s=8, alpha=0.35)
    f = sm.OLS(r[ok].to_numpy(), sm.add_constant(x[ok].to_numpy())).fit()
    xx = np.linspace(float(x[ok].min()), float(x[ok].max()), 50)
    ax.plot(xx, f.params[0] + f.params[1] * xx, color="C3", lw=1.2)
    ax.set_xlabel(xlab)
    ax.set_ylabel(r"$R_t$ (settled at 16:00)")
    ax.set_title(f"block-diagonal ridge — $R_t$ against {ttl}")
fig.tight_layout()
fig.savefig(OUT / "sameday_scatter_blk2.png", dpi=120, bbox_inches="tight")
display(fig)
plt.close(fig)
"""
    ),
    # SECTION PARKED 2026-09-02 (user order): vol-target overlay held out of the deck.
    # md(
    # r"""
    # ## 15. Sizing by trailing book volatility
    #
    # The two rules above hold $|q_t|=1$ every day, so the book inherits the
    # market's volatility cycle: its own risk swings roughly $2.4\times$
    # between calm and stormy quarters. The lagged-signal slides show the
    # signal's information at day scale is its **sign** — so we do not try to
    # size by conviction. Instead we standardize **risk**: scale the whole
    # position by how volatile the book itself has recently been.
    #
    # Construction (causal throughout): let $\hat\sigma_t$ be the standard
    # deviation of the rule's own daily return $R'$ over the trailing 63
    # sessions, lagged one day. The scale factor is
    #
    # $$\ell_t=\min\!\left(\frac{\operatorname{median}_{u\le t-1}\hat\sigma_u}{\hat\sigma_t},\,3\right),
    # \qquad R''_t=\ell_t\,R'_t .$$
    #
    # The target in the numerator is the expanding median of $\hat\sigma$
    # itself, so average leverage is close to one by construction and there is
    # no free target parameter; 63 sessions is the standing quarter window and
    # 3 the standing leverage cap (it never binds — realized $\ell$ stays in
    # $[0.6,\,1.6]$). The first 63+1 sessions have no estimate and sit flat;
    # raw and scaled books are compared on the same remaining days.
    #
    # This overlay claims **no forecast information** — it reads only the
    # book's own past returns. The scoreboard is therefore risk stability
    # (the variability of the book's rolling volatility, its drawdown, its
    # per-year volatility), **not** Sharpe: a pure rescaling should leave
    # Sharpe roughly unchanged, and does. One honest limitation is structural:
    # a trailing estimator cannot see the first day of a regime change, so
    # single worst days keep their size; what shrinks is the quarter-to-year
    # wander of realized risk.
    # """
    # ),
    # code(
    # r"""
    # VT_WIN, VT_CAP = 63, 3.0  # standing quarter window, standing leverage cap
    #
    #
    # def vol_target(rp: pd.Series) -> pd.DataFrame:
    # rp = rp.astype(float)
    # sig = rp.rolling(VT_WIN, min_periods=VT_WIN).std(ddof=1).shift(1)
    # target = sig.expanding(min_periods=1).median().shift(1)
    # ell = (target / sig).clip(0.0, VT_CAP)
    # return pd.DataFrame({"raw": rp, "ell": ell, "scaled": ell * rp})
    #
    #
    # def risk_row(r: pd.Series) -> pd.Series:
    # rv = r.rolling(VT_WIN, min_periods=VT_WIN).std(ddof=1).dropna()
    # cum = r.cumsum()
    # mu, sd = float(r.mean()), float(r.std(ddof=1))
    # return pd.Series({
    # "n": len(r),
    # "vol_CV": float(rv.std(ddof=1) / rv.mean()),
    # "vol_max": float(rv.max()),
    # "vol_min": float(rv.min()),
    # "maxDD": float((cum - cum.cummax()).min()),
    # "worst_day": float(r.min()),
    # "skew": float(r.skew()),
    # "ex_kurt": float(r.kurt()),
    # "mean": mu,
    # "std": sd,
    # "Sharpe_ann": mu / sd * np.sqrt(252.0),
    # })
    #
    #
    # vt_rows, vt_lev, vt_year = [], [], []
    # for tag in MODEL_ORDER:
    # px = books[tag]
    # for name, q in rule_sizes(px).items():
    # f = vol_target((q * px["R"]).loc[common]).dropna(subset=["ell"])
    # for kind in ("raw", "scaled"):
    # vt_rows.append(risk_row(f[kind]).rename((name, tag, kind)))
    # vt_lev.append(pd.Series({
    # "mean_ell": float(f["ell"].mean()),
    # "median_ell": float(f["ell"].median()),
    # "min_ell": float(f["ell"].min()),
    # "max_ell": float(f["ell"].max()),
    # "pct_at_cap": 100.0 * float((f["ell"] >= VT_CAP - 1e-12).mean()),
    # }, name=(name, tag)))
    # for yr, g in f.groupby(f.index.year):
    # vt_year.append({"rule": name, "model": tag, "year": yr,
    # "vol_raw": float(g["raw"].std(ddof=1)),
    # "vol_scaled": float(g["scaled"].std(ddof=1))})
    #
    # vt_tab = pd.DataFrame(vt_rows)
    # vt_tab.index = pd.MultiIndex.from_tuples(vt_tab.index, names=["rule", "model", "book"])
    # vt_lev = pd.DataFrame(vt_lev)
    # vt_lev.index = pd.MultiIndex.from_tuples(vt_lev.index, names=["rule", "model"])
    # vt_year = pd.DataFrame(vt_year).set_index(["rule", "model", "year"]).sort_index()
    #
    # # always short takes no forecast: assert model-invariance, print one row pair
    # base = vt_tab.loc["always short"].loc["blk2"]
    # for tag in MODEL_ORDER:
    # assert np.allclose(vt_tab.loc["always short"].loc[tag], base)
    # print("always short (all models identical)")
    # print(pd.DataFrame({"raw": base.loc["raw"], "scaled": base.loc["scaled"]}).T.to_string())
    # print("---")
    # print("long-short volatility")
    # ls = vt_tab.loc["long-short volatility"]
    # print(pd.concat({LABEL[t]: ls.loc[t] for t in MODEL_ORDER}, axis=0).to_string())
    # print("---")
    # print("scale-factor diagnostics (raw |q| is 1 every day by construction)")
    # print(vt_lev.rename(index=LABEL, level="model").to_string())
    # print("---")
    # print("per-year book volatility, block-diag ridge long-short")
    # print(vt_year.loc[("long-short volatility", "blk2")].to_string())
    #
    # vt_tab.to_csv(OUT / "voltarget_scoreboard.csv")
    # vt_lev.to_csv(OUT / "voltarget_leverage.csv")
    # vt_year.to_csv(OUT / "voltarget_per_year.csv")
    # print("saved voltarget_{scoreboard,leverage,per_year}.csv in", OUT)
    #
    # f = vol_target(
    # (rule_sizes(books["blk2"])["long-short volatility"] * books["blk2"]["R"]).loc[common]
    # ).dropna(subset=["ell"])
    # fig, axes = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True)
    # axes[0].plot(f.index, f["ell"], lw=0.8, color="C0")
    # axes[0].axhline(1.0, color="k", lw=0.6)
    # axes[0].set_ylabel(r"scale $\ell_t$")
    # axes[0].set_title("block-diag ridge, long-short — trailing-volatility scale and its effect")
    # for kind, c, lab in (("raw", "C1", "raw ($|q|=1$)"), ("scaled", "C0", "vol-scaled")):
    # rv = f[kind].rolling(VT_WIN, min_periods=VT_WIN).std(ddof=1)
    # axes[1].plot(rv.index, rv, lw=0.9, color=c, label=lab)
    # axes[1].set_ylabel("rolling 63-day vol of $R'$")
    # axes[1].legend(fontsize=8)
    # fig.tight_layout()
    # fig.savefig(OUT / "voltarget_blk2.png", dpi=120, bbox_inches="tight")
    # print("saved", OUT / "voltarget_blk2.png")
    # display(fig)
    # plt.close(fig)
    # """
    # ),
    md(
        r"""
## 14. Betting a fixed fraction of wealth

Every table so far adds up profits and losses without compounding:
one unit of premium per day. A trader who reinvests instead bets a
fraction $f_t$ of current wealth each day, so wealth compounds,

$$W_T=\prod_{t}\bigl(1+f_t\,q_t R_t\bigr),$$

and the natural objective is the annualized growth rate of log
wealth, $g=252\cdot\overline{\log(1+f_t R'_t)}$, which is what
compounding actually maximizes. One structural fact frames the
exercise: **ruin is one bad day.** Any fraction
$f\ge 1/|\min_t R'_t|$ takes wealth to zero or below on the worst
day, so the short book without wings, whose worst day is near $-10$
premium units, can only ever bet a small fraction of wealth no matter
how good its average return is. (A defined-risk variant, under construction and not shown in this
deck, bounds the worst day by construction, which is exactly what
loosens this constraint.)

The fraction is estimated from the book's own past returns only:

$$\hat f_t=\min\!\Bigl(\tfrac{\hat\mu_{t-1}}{\widehat{E[R'^2]}_{t-1}},\,
\tfrac{1}{|\min_{u\le t-1}R'_u|}\Bigr)_{\!+}$$

— the growth-optimal (Kelly) fraction in its quadratic form, held
below the running ruin bound, computed on an expanding window once at
least 63 sessions of history exist and lagged one day. The half
fraction $\hat f_t/2$ is reported alongside as the standard
conservative choice, and for each path we print the growth rate, the
terminal wealth, the drawdown, and the worst single-day wealth factor.

**Two frames for "a fraction of wealth."** The phrase needs a unit
before any growth number can be read. In the **per-premium frame** —
the frame used throughout this section — $f_t$ is the share of wealth
deployed as body premium. For a bought straddle the premium is also
the capital at risk, so on long days the two coincide; for the short
book without wings the capital at risk is unbounded, which is exactly
why the ruin bound above, and not a collateral bound, governs the
admissible fraction. A true **capital-at-risk frame** — $f_t$ as the
share of wealth posted as collateral against the maximum loss, with
the day's return on that capital bounded below by $-1$ by
construction — exists only once the worst day is bounded; it applies to a
defined-risk variant that is under construction and not shown in
this deck. One unit of wealth
posted as collateral controls several times less premium exposure
than one unit deployed as premium, and the ratio varies from day to
day, so growth rates compare only within a frame, never across.
"""
    ),
    code(
        r"""
def causal_kelly(rs):
    mu = rs.expanding(min_periods=63).mean().shift(1)
    m2 = (rs**2).expanding(min_periods=63).mean().shift(1)
    cap = 1.0 / rs.expanding(min_periods=1).min().shift(1).abs()
    return np.minimum((mu / m2).clip(lower=0.0), cap).fillna(0.0)

def wealth_row(f, r):
    factors = 1.0 + np.asarray(f) * np.asarray(r)
    assert (factors > 0).all(), "a wealth factor hit zero — ruin"
    w = np.cumprod(factors)
    return pd.Series({
        "g_ann": 252.0 * float(np.mean(np.log(factors))),
        "terminal": float(w[-1]),
        "maxDD": float((w / np.maximum.accumulate(w) - 1.0).min()),
        "worst_day_factor": float(factors.min()),
    })

kelly_rows = {}
for tag in MODEL_ORDER:
    px = books[tag]
    for name, q in rule_sizes(px).items():
        rs = (q * px["R"]).loc[common].astype(float)
        r = rs.to_numpy()
        fk = causal_kelly(rs).to_numpy()
        row = pd.concat({"causal": wealth_row(fk, r), "half": wealth_row(fk / 2, r)})
        row.index = ["_".join(k) for k in row.index]
        row["mean_f"] = float(fk.mean())
        row["n"] = len(r)
        kelly_rows[(name, tag)] = row

kelly_tab = pd.DataFrame(kelly_rows).T
kelly_tab.index = pd.MultiIndex.from_tuples(kelly_tab.index, names=["rule", "model"])
for tag in MODEL_ORDER:
    safe = "".join(ch if ch.isalnum() else "_" for ch in tag)
    kelly_tab.xs(tag, level="model").to_csv(OUT / f"kelly_summary_{safe}.csv")
print("blk2 — causally estimated fraction of wealth, 871 common days")
print(kelly_tab.xs("blk2", level="model").T.to_string())
print("saved kelly_summary_<model>.csv in", OUT)

print("---")
print("causal growth across models, both rules")
spread = kelly_tab[["mean_f", "causal_g_ann", "causal_terminal"]].rename(index=LABEL, level="model")
print(spread.to_string(float_format=lambda x: f"{x:+.4f}"))

for name in ("always short", "long-short volatility"):
    rs = (rule_sizes(books["blk2"])[name] * books["blk2"]["R"]).loc[common].astype(float)
    fk = causal_kelly(rs)
    gy = pd.Series(np.log1p(fk.to_numpy() * rs.to_numpy()), index=rs.index)
    print(f"per-year annualized log-growth, blk2 {name}:")
    print((252.0 * gy.groupby(gy.index.year).mean()).to_string(float_format=lambda x: f"{x:+.3f}"))

px = books["blk2"]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for name, c in (("always short", "C1"), ("long-short volatility", "C0")):
    rs = (rule_sizes(px)[name] * px["R"]).loc[common].astype(float)
    fk = causal_kelly(rs)
    axes[0].plot(rs.index, fk, lw=0.9, color=c, label=name)
    w = np.cumprod(1.0 + fk.to_numpy() * rs.to_numpy())
    axes[1].plot(rs.index, np.log(w), lw=1.0, color=c, label=name)
rs = (rule_sizes(px)["long-short volatility"] * px["R"]).loc[common].astype(float)
fk = causal_kelly(rs)
w = np.cumprod(1.0 + (fk.to_numpy() / 2) * rs.to_numpy())
axes[1].plot(rs.index, np.log(w), lw=1.0, color="C2", label="long-short, half fraction")
axes[0].set_ylabel(r"fraction of wealth $\hat f_t$")
axes[0].set_title("blk2 — causally estimated fraction")
axes[0].legend(fontsize=8)
axes[1].set_ylabel("log wealth")
axes[1].set_title("blk2 — compounded wealth")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "kelly_causal_blk2.png", dpi=120, bbox_inches="tight")
print("saved", OUT / "kelly_causal_blk2.png")
display(fig)
plt.close(fig)
"""
    ),
    md(
        r"""
## 15. When does each forecast say buy?

A buy day is one with $q_t>0$. The always-short rule never buys. The
forecasts are compared on the days they share.
"""
    ),
    code(
        r"""
buy = {}
for tag in MODEL_ORDER:
    px = books[tag].loc[common]
    buy[tag] = (px["pos"] > 0)
diag_rows = []
for tag in MODEL_ORDER:
    b = buy[tag]
    px = books[tag].loc[common]
    diag_rows.append({
        "model": LABEL[tag],
        "n_buy": int(b.sum()),
        "pct_buy": 100.0 * float(b.mean()),
        "mean_R|buy": float(px.loc[b, "R"].mean()) if b.any() else float("nan"),
        "mean_R|sell": float(px.loc[~b, "R"].mean()),
        "median_|s|_buy": float(px.loc[b, "signal"].abs().median()) if b.any() else float("nan"),
    })
print(pd.DataFrame(diag_rows).to_string(index=False))
print("always-short n_buy = 0.")

jacc = pd.DataFrame(index=MODEL_ORDER, columns=MODEL_ORDER, dtype=float)
for a in MODEL_ORDER:
    for b in MODEL_ORDER:
        u = buy[a] | buy[b]
        jacc.loc[a, b] = float((buy[a] & buy[b]).sum() / u.sum()) if u.any() else 1.0
jacc.index = [LABEL[t] for t in MODEL_ORDER]
jacc.columns = jacc.index
print("Jaccard of buy-day sets")
print(jacc.round(3).to_string())
jacc.to_csv(OUT / "buy_signal_agreement.csv")
off = jacc.values[np.triu_indices(len(jacc), 1)]
print(f"mean pairwise Jaccard {float(off.mean()):.3f}")

idx = pd.DatetimeIndex(common)
fig, axes = plt.subplots(len(MODEL_ORDER), 1, figsize=(11, 7.2), sharex=True)
for ax, tag in zip(axes, MODEL_ORDER):
    days = idx[buy[tag].to_numpy()]
    ax.vlines(days, 0, 1, color="C0", lw=0.5)
    ax.set_yticks([])
    ax.set_ylabel(tag, rotation=0, ha="right", va="center", fontsize=8)
    ax.set_ylim(0, 1)
axes[-1].set_xlabel("date")
axes[0].set_title("buy days")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT / "buy_signal_diag_blk2.png", dpi=120, bbox_inches="tight")
print("saved", OUT / "buy_signal_diag_blk2.png")
display(fig)
plt.close(fig)
"""
    ),
    # SECTION PARKED 2026-09-02 (user order): credit verticals under construction — see the experimental notebook's lab.
    # md(
    # r"""
    # ## 17. Credit vertical spreads — capping the downside when selling
    #
    # Selling the straddle body leaves an unbounded downside. Buy a wing on
    # each side — the nearest strike with a live midpoint quote at least 25
    # (or 50) points further out of the money — and each short leg becomes
    # a **credit vertical spread**: a short call with a long call at a
    # higher strike, a short put with a long put at a lower strike. The
    # pair is exactly the short iron condor of the earlier version of this
    # section; the present framing makes the risk cap explicit. The net
    # credit $C$ is the body premium minus the wing premium. Days on which
    # $C \le 0$ (deep wing quotes missing, roughly one day in a thousand)
    # are dropped, and the count is printed.
    #
    # **Primary units: per body premium.** The day's return is
    #
    # $$R' = \frac{C - \text{settlement payout}}{P_{\mathrm{body}}},$$
    #
    # the package's profit or loss over the straddle's entry midpoint — the
    # same denominator as the straddle rows of §10, so the cost of the
    # wings reads directly against the plain book. A **capital-at-risk**
    # view is reported alongside: divide instead by the worst-case loss,
    # which is the larger of the two actual wing gaps minus the credit. The
    # nearest live wing can sit farther out than the nominal 25 or 50
    # points, so the actual gap, not the nominal width, is the honest worst
    # case; with it this view is bounded below by exactly $-1$, and the
    # cell checks that bound. Capital at risk is not the primary view
    # because it overweights the days with a rich credit — the
    # high-volatility days — since the smallest denominators carry most of
    # the variance of the series, and it therefore reads nearly free tail
    # insurance as a losing trade: an artifact of the unit, not a trading
    # result. Returns per unit of credit are worse still, because $C$ can
    # be tiny; they are not reported.
    #
    # The hedge is **asymmetric by design**: wings are bought only when the
    # book sells volatility. On the days when the rule buys volatility it
    # holds the plain straddle of §6 — a bought straddle already risks at
    # most its premium, and wings there would cap the very payoff tail the
    # long position exists to own; its return is already per body premium,
    # so the long-short composite is in one unit from day to day. The
    # dollar block at the end prices the insurance itself, in index points
    # per package: the cost of the wings per day with a paired Newey–West
    # $t$, the worst day and the drawdown with and without wings, and the
    # points the wings hand back on the days when the settlement lands
    # beyond a wing.
    #
    # A fractional-wealth reading closes the loop with §15, in the two
    # frames defined there and with the same estimator. The
    # **capital-at-risk frame** is well defined here because the worst day
    # is bounded at $-1$ by construction: no fraction below one can be
    # ruined. The bound itself excludes ruin, and the running ruin bound
    # inside the §15 estimator settles at one after the first day at $-1$
    # and never binds thereafter. This is the fully collateralized floor.
    # In the **per-premium frame** — the unit of §15 — the plain book over
    # the same days is printed beside the hedged one; that pair, and only
    # that pair, supports a comparison of growth with and without wings.
    # Where the trailing mean is negative throughout, the estimator
    # declines to bet and the row is zeros.
    # """
    # ),
    # code(
    # r"""
    # live1530 = book_chain[(book_chain["hhmm"] == "15:30") & np.isfinite(book_chain["mid"]) & (book_chain["mid"] > 0)].copy()
    # body = atm.reset_index()
    # close_map = pd.Series(atm["S_close"].to_numpy(), index=pd.to_datetime(atm["expiration"]).values)
    # close_map.index = pd.to_datetime(close_map.index).tz_localize(None).normalize()
    #
    # def maxdd(r):
    # cum = r.cumsum()
    # return float((cum - cum.cummax()).min())
    #
    # def score_verticals(width: float):
    # vs = asl.pick_wings(live1530, body, width=width)
    # print(f"width {width}: days with both wings {len(vs)} / body {len(body)} dropped {len(body)-len(vs)}")
    # vs = asl.settle_package(vs, close_map)
    # vs = vs[np.isfinite(vs["entry_ic"]) & np.isfinite(vs["exit_ic"]) & (vs["width"] > 0)].copy()
    # # short seller receives the body premium and pays for the wings
    # vs["credit"] = vs["entry_ic"]
    # # worst case loses the larger ACTUAL wing gap (the nearest live wing
    # # can sit farther out than the nominal width) minus the credit
    # vs["gap_max"] = np.maximum(vs["K_c_wing"] - vs["K_c"], vs["K_p"] - vs["K_p_wing"])
    # bad = ~((vs["credit"] > 0) & (vs["credit"] < vs["gap_max"]))
    # print(f"width {width}: dropped {int(bad.sum())} day(s) with credit <= 0 or credit >= max wing gap")
    # vs = vs[~bad].copy()
    # vs["max_loss"] = vs["gap_max"] - vs["credit"]
    # vs["pnl"] = vs["credit"] - vs["exit_ic"]          # index points per package
    # vs["R_prem"] = vs["pnl"] / vs["entry_body"]       # primary: per body premium
    # vs["R_risk"] = vs["pnl"] / vs["max_loss"]         # secondary: per capital at risk
    # n_floor = int((vs["R_risk"] < -1.0 - 1e-12).sum())
    # assert n_floor == 0, f"{n_floor} day(s) breach the -1 capital-at-risk floor"
    # print(f"width {width}: capital-at-risk floor check min R' {float(vs['R_risk'].min()):+.6f} (>= -1, 0 violations)")
    # vs["cap_binds"] = (vs["S_close"] >= vs["K_c_wing"]) | (vs["S_close"] <= vs["K_p_wing"])
    # if "day" in vs.columns:
    # vs = vs.set_index("day")
    # return vs.sort_index()
    #
    # vs_tabs = {}
    # for w in (25.0, 50.0):
    # vs = score_verticals(w)
    # vs_tabs[w] = vs
    # bind = vs["cap_binds"]
    # print(f"width {w}: cap binds (settle beyond a wing) on {int(bind.sum())} of {len(vs)} days "
    # f"({float(bind.mean()):.2%}); mean per-premium R' on those days {float(vs.loc[bind, 'R_prem'].mean()):+.4f}")
    # for tag in MODEL_ORDER:
    # px = books[tag]
    # joined = vs.join(px[["signal", "pos", "R"]], how="inner", rsuffix="_strad")
    # j = joined.loc[joined.index.intersection(common)]
    # series = {
    # ("always short", "per premium"): (j["R_prem"], pd.Series(-1.0, index=j.index)),
    # ("always short", "per max risk"): (j["R_risk"], pd.Series(-1.0, index=j.index)),
    # ("long-short volatility", "per premium"): (j["R_prem"].where(j["pos"] < 0, j["R"]), j["pos"]),
    # ("long-short volatility", "per max risk"): (j["R_risk"].where(j["pos"] < 0, j["R"]), j["pos"]),
    # }
    # tab = pd.DataFrame({key: asl.rule_row(r, sz) for key, (r, sz) in series.items()}).T
    # tab.index = pd.MultiIndex.from_tuples(tab.index, names=["rule", "units"])
    # safe = f"credit_spread_w{int(w)}_rule_by_strategy_" + "".join(ch if ch.isalnum() else "_" for ch in tag)
    # tab.to_csv(OUT / f"{safe}.csv")
    # if tag != "blk2":
    # continue
    # print(f"--- credit verticals width {w}, per body premium, blk2 ---")
    # print(f"days {len(j)} = wing days & common; "
    # f"{int((j['pos'] < 0).sum())} short-vertical, {int((j['pos'] > 0).sum())} long-straddle")
    # for name in ("always short", "long-short volatility"):
    # r, sz = series[(name, "per premium")]
    # print(name)
    # print(asl.rule_row(r, sz).to_string())
    # print("capital-at-risk view (bounded at -1), same days:")
    # for name in ("always short", "long-short volatility"):
    # r, _ = series[(name, "per max risk")]
    # print(f"  {name}: mean {float(r.mean()):+.5f} Sharpe {float(r.mean()/r.std(ddof=1)*np.sqrt(252)):+.3f} "
    # f"min {float(r.min()):+.3f} max {float(r.max()):+.3f}")
    # print("capped vs uncapped straddle, same days, per body premium:")
    # for name in ("always short", "long-short volatility"):
    # r, _ = series[(name, "per premium")]
    # bench = (-j["R"]) if name == "always short" else j["pos"] * j["R"]
    # print(f"  {name}: mean {float(r.mean()):+.5f} Sharpe {float(r.mean()/r.std(ddof=1)*np.sqrt(252)):+.3f} "
    # f"worst {float(r.min()):+.3f} maxDD {maxdd(r):+.2f} | uncapped: "
    # f"mean {float(bench.mean()):+.5f} Sharpe {float(bench.mean()/bench.std(ddof=1)*np.sqrt(252)):+.3f} "
    # f"worst {float(bench.min()):+.3f} maxDD {maxdd(bench):+.2f}")
    # # the insurance itself, in index points per package (always-short frame)
    # naked = (j["entry_body"] - j["exit"]).astype(float)
    # capped = j["pnl"].astype(float)
    # drag = naked - capped
    # t_drag = sm.OLS(drag.to_numpy(), np.ones((len(drag), 1))).fit(
    # cov_type="HAC", cov_kwds={"maxlags": 6}
    # )
    # jb = j["cap_binds"]
    # worst10 = naked.nsmallest(10).index
    # print(f"insurance in index points/package, always short, {len(j)} days:")
    # print(f"  wing drag {float(drag.mean()):+.3f}/day (paired HAC-6 t {float(t_drag.tvalues[0]):+.2f}); "
    # f"worst day naked {float(naked.min()):+.1f} vs capped {float(capped.min()):+.1f}; "
    # f"maxDD naked {maxdd(naked):+.1f} vs capped {maxdd(capped):+.1f}")
    # print(f"  cap binds on {int(jb.sum())} of these days: points saved there "
    # f"{float((capped - naked)[jb].sum()):+.1f}; over the 10 worst naked days "
    # f"{float((capped - naked)[worst10].sum()):+.1f}")
    #
    # print("=== fractional bets on the defined-risk book (estimator of section 15; frame named first) ===")
    # kv_rows = {}
    # for w, vs in vs_tabs.items():
    # for tag in MODEL_ORDER:
    # px = books[tag]
    # joined = vs.join(px[["pos", "R"]], how="inner", rsuffix="_strad")
    # j = joined.loc[joined.index.intersection(common)]
    # series = {
    # ("always short", "capital at risk"): j["R_risk"],
    # ("always short", "per premium"): j["R_prem"],
    # ("long-short volatility", "capital at risk"): j["R_risk"].where(j["pos"] < 0, j["R"]),
    # ("long-short volatility", "per premium"): j["R_prem"].where(j["pos"] < 0, j["R"]),
    # }
    # for (name, frame), rs in series.items():
    # rs = rs.astype(float)
    # r = rs.to_numpy()
    # fk = causal_kelly(rs).to_numpy()
    # row = pd.concat({"causal": wealth_row(fk, r), "half": wealth_row(fk / 2, r)})
    # row.index = ["_".join(k) for k in row.index]
    # row["mean_f"] = float(fk.mean())
    # if frame == "per premium":
    # # uncapped comparison lives ONLY in the shared premium unit
    # ru = (-px["R"] if name == "always short" else px["pos"] * px["R"]).loc[rs.index].astype(float)
    # fku = causal_kelly(ru).to_numpy()
    # unc = wealth_row(fku, ru.to_numpy())
    # row["uncapped_g_ann"] = unc["g_ann"]
    # row["uncapped_terminal"] = unc["terminal"]
    # row["uncapped_mean_f"] = float(fku.mean())
    # kv_rows[(name, frame, tag, int(w))] = row
    # kv = pd.DataFrame(kv_rows).T
    # kv.index = pd.MultiIndex.from_tuples(kv.index, names=["rule", "frame", "model", "w"])
    # for tag in MODEL_ORDER:
    # safe = "".join(ch if ch.isalnum() else "_" for ch in tag)
    # kv.xs(tag, level="model").to_csv(OUT / f"kelly_verticals_{safe}.csv")
    # print("blk2 (uncapped columns appear only in the per-premium frame — the shared unit):")
    # print(kv.xs("blk2", level="model").T.to_string(float_format=lambda x: f"{x:+.4f}", na_rep=""))
    # print("saved kelly_verticals_<model>.csv in", OUT)
    #
    # fig, ax = plt.subplots(figsize=(11, 3.4))
    # px = books["blk2"]
    # rp_s = (px["pos"] * px["R"]).loc[common].cumsum()
    # ax.plot(rp_s.index, rp_s.values, label="long-short straddle (uncapped)", lw=1.2)
    # for w, vs in vs_tabs.items():
    # j = vs.join(px[["pos", "R"]], how="inner", rsuffix="_strad")
    # j = j.loc[j.index.intersection(common)]
    # ls = j["R_prem"].where(j["pos"] < 0, j["R"])
    # ax.plot(ls.index, ls.cumsum().values,
    # label=f"long-short, verticals on sell days w={int(w)}", lw=1.1)
    # ax.set_title("cumulative $R'$ per body premium — long-short volatility, blk2 days")
    # ax.legend(fontsize=8)
    # fig.tight_layout()
    # fig.savefig(OUT / "credit_spread_vs_straddle_cum.png", dpi=120, bbox_inches="tight")
    # print("saved", OUT / "credit_spread_vs_straddle_cum.png")
    # display(fig)
    # plt.close(fig)
    # """
    # ),
    md(
        r"""
## 16. Checking one row by hand

The columns of a single row map onto the construction as follows.

- `K_c >= S` and `K_p <= S` at 15:30: the call strike sits at or
  above the spot and the put strike at or below it.
- `entry` is the 15:30 midpoint quote of the package, `mid_c + mid_p`.
- `exit` is the cash settlement,
  `max(S_close - K_c, 0) + max(K_p - S_close, 0)`, with `S_close`
  the official close from yfinance `^GSPC`.
- `R` is `exit/entry - 1`.
- `yhat` is the forecast on the fitted scale, $\widehat{\sqrt{RV/B}}$;
  `rv_hat = (m^2 + s2) * B` converts it to a 30-minute variance.
- The quoted implied volatility is hourly, so
  `iv_30 = iv_hourly / sqrt(2)` and `iv_var = iv_30**2`.
- `signal = rv_hat - iv_var`; the variance risk premium is
  $\mathrm{VRP}=-s$. The long-short volatility position `pos` is $+1$
  when the signal is positive and $-1$ otherwise.
"""
    ),
    md(
        r"""
---

*Housekeeping note (2026-09-02, not analysis).* Repo state to clean up
later. The two working directories are worktrees of one repo on sibling
branches that diverged at `5db8240`: this branch
(`grok/0dte-professor-notes`, +15) carries all the notebook work and
the close-option writeup prose; `paper2` (+1, `d3809a4`) carries the
F$_t$-measurability scorer fixes and the honest options macros in
`writeup/generated/`. The writeup therefore has two truths pending
consolidation (both options narratives are parked, so nothing renders
wrong yet). Plan sketched: (a) merge this branch into `paper2` (merge,
not rebase) and fast-forward this branch to the merge; (b) fast-forward
local `main` (behind `origin/main` by 6 campaign-sync commits that are
already ancestors of `paper2`); (c) fix `paper2`'s upstream (tracks
`origin/main`, should be `origin/paper2`) and delete-or-archive the two
dead local branches (`edge-features-legibility`,
`prop-exploration-2026-07-02`, upstreams gone); (d) untrack LaTeX aux
and volatile `.hpc` state via `.gitignore` to stop the chronic dirty
tree; (e) review the real uncommitted diffs in `harxhar-clean`
(`experiments/run_geometry_local.py`, spec/pack jsons) before touching
anything. Open decision: one unified line vs keeping this branch as a
deliberate fork; delete vs archive for the dead branches.
"""
    ),
]


path = Path(__file__).resolve().parent / "atm_straddle_rv_iv.ipynb"
n_kept = carry_outputs(nb, path)
nbf.write(nb, path)
print("wrote", path, "carried outputs for", n_kept, "code cells")
