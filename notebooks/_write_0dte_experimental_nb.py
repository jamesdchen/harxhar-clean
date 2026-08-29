"""Write notebooks/atm_straddle_experimental.ipynb — ensembles, R~s, extra weights."""

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
# 0DTE experimental lab (15:30 ATM package)

Same instrument, clocks, smear, and signal as
`atm_straddle_rv_iv.ipynb`. This notebook is the lab for:

1. ensembles of the seven models and of the three paper rules
2. $R \sim a + b\,s$ (and scaled $s$)
3. vol-space maps $\hat y\sqrt{B}$, $m\sqrt{B}$ (stand-in if
   `atm_straddle_volmap.ipynb` is not in the tree)
4. extra weighting rules (rank, inv-vol, dead-zone, vol-target, Kelly-ish)

Published mid-fill rule tables stay in the RV–IV notebook. Intraday
bars stay in `atm_straddle_intraday.ipynb`.
"""
    ),
    code(
        """
import hashlib
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
import statsmodels.api as sm

sys.path.insert(0, str(Path.cwd() if (Path.cwd() / "atm_straddle_lib.py").exists() else Path.cwd() / "notebooks"))
import atm_straddle_lib as asl

REPO = asl.find_repo(Path.cwd())
sys.path.insert(0, str(REPO / "notebooks"))
import atm_straddle_lib as asl
OUT = REPO / "results" / "atm_straddle_0dte_1530"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
print("repo:", REPO)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 24)
pd.set_option("display.float_format", lambda x: f"{x: .6f}")
"""
    ),
    md("## 1. Chain, nearest-OTM package, $R$, forecasts, signal"),
    code(
        """
path = REPO / "data" / "spxw_chain.parquet"
COLS = ["expiration", "timestamp", "strike", "cp", "bid", "ask", "mid", "underlying_price", "impl_volatility"]
_st = os.stat(path)
_ck = CACHE / f"chain_15301600_{_st.st_size}_{_st.st_mtime_ns}.parquet"
if _ck.exists():
    chain = pd.read_parquet(_ck)
else:
    ts = pd.to_datetime(pd.read_parquet(path, columns=["timestamp"])["timestamp"], utc=True)
    et = ts.dt.tz_convert("America/New_York")
    keep = ts[((et.dt.hour == 15) & (et.dt.minute == 30)) | ((et.dt.hour == 16) & (et.dt.minute == 0))].unique()
    chain = pd.read_parquet(path, columns=COLS, filters=[("timestamp", "in", list(keep))])
    chain["timestamp"] = pd.to_datetime(chain["timestamp"], utc=True)
    chain["expiration"] = pd.to_datetime(chain["expiration"])
    chain["cp"] = chain["cp"].astype(str).str.upper().str[0]
    chain.to_parquet(_ck)
codes, uts = pd.factorize(chain["timestamp"])
uet = pd.DatetimeIndex(uts).tz_convert("America/New_York")
chain["et"] = uet.take(codes)
hh = np.where((uet.hour == 15) & (uet.minute == 30), "15:30",
              np.where((uet.hour == 16) & (uet.minute == 0), "16:00", "other"))
chain["hhmm"] = hh[codes]
chain["et_date"] = uet.normalize().take(codes)
ecodes, uexp = pd.factorize(chain["expiration"])
uexp_d = pd.DatetimeIndex(uexp).tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").normalize()
chain["exp_date"] = uexp_d.take(ecodes)
chain["is_0dte"] = chain["et_date"] == chain["exp_date"]
book_chain = chain[chain["is_0dte"] & chain["hhmm"].isin(["15:30", "16:00"])].copy()
del chain
e = book_chain[book_chain["hhmm"] == "15:30"].copy()
live = e[np.isfinite(e["mid"]) & (e["mid"] > 0)].copy()
spot = live.dropna(subset=["underlying_price"]).groupby("expiration")["underlying_price"].median()
atm = asl.pick_nearest_otm(live, spot)
atm["day"] = atm["et_c"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)

def load_gspc_close(days):
    days = pd.to_datetime(days)
    cp = CACHE / "gspc_close.parquet"
    if cp.exists():
        cached = pd.read_parquet(cp)["close"]
        cached.index = pd.to_datetime(cached.index)
        if cached.index.min() <= pd.Timestamp(days.min()) and cached.index.max() >= pd.Timestamp(days.max()):
            return cached.astype(float)
    raw = yf.download("^GSPC", start=pd.Timestamp(days.min()) - pd.Timedelta("7D"),
                      end=pd.Timestamp(days.max()) + pd.Timedelta("7D"),
                      auto_adjust=True, progress=False, threads=True)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    ix = pd.to_datetime(close.index)
    if getattr(ix, "tz", None) is not None:
        ix = ix.tz_convert("America/New_York").tz_localize(None)
    close.index = ix.normalize()
    close.rename("close").to_frame().to_parquet(cp)
    return close.astype(float)

exp_day = pd.to_datetime(atm["expiration"])
if getattr(exp_day.dt, "tz", None) is not None:
    exp_day = exp_day.dt.tz_convert("America/New_York").dt.tz_localize(None)
exp_day = exp_day.dt.normalize()
atm["S_close"] = exp_day.map(load_gspc_close(exp_day))
atm["pay_c"] = np.maximum(atm["S_close"] - atm["K_c"], 0.0)
atm["pay_p"] = np.maximum(atm["K_p"] - atm["S_close"], 0.0)
atm["exit"] = atm["pay_c"] + atm["pay_p"]
atm = atm[np.isfinite(atm["entry"]) & np.isfinite(atm["exit"]) & (atm["entry"] > 0)].copy()
atm["R"] = atm["exit"] / atm["entry"] - 1.0
atm = atm.set_index("day").sort_index()
atm = asl.attach_iv_hourly_as_30min(atm)

from concurrent.futures import ThreadPoolExecutor
YHATS = asl.yhat_paths(REPO)
need_dates = set(atm.index)
with ThreadPoolExecutor(max_workers=len(YHATS)) as pool:
    futs = {tag: pool.submit(asl.load_yhat_1530_mz_cached, tag, path, need_dates, CACHE)
            for tag, path in YHATS.items()}
    models = {tag: futs[tag].result() for tag in YHATS}
LABEL = asl.YHAT_LABEL
MODEL_ORDER = asl.MODEL_ORDER

books = {}
for tag, rv in models.items():
    px = atm.join(rv[["rv_hat", "yhat", "m", "yhat_vol", "m_vol", "rv_raw", "baseline"]], how="inner")
    px = px.dropna(subset=["R", "rv_hat", "iv_var"])
    px = px[(px["rv_hat"] > 0) & (px["iv_var"] > 0)]
    px["signal"] = px["rv_hat"] - px["iv_var"]
    px["pos"] = np.where(px["signal"] > 0, 1.0, -1.0)
    books[tag] = px
common = None
for tag in MODEL_ORDER:
    common = books[tag].index if common is None else common.intersection(books[tag].index)
common = common.sort_values()
print("common days", len(common), pd.Timestamp(common.min()), "->", pd.Timestamp(common.max()))
"""
    ),
    md(
        r"""
## 5. Ensemble of the seven models

(a) mean $s$ then sign. (b) majority vote of sign. (c) equal-weight
average of unit-median $q$. (d) trailing-Sharpe and trailing-IR
weights on unit-median $q$ (expanding 63, lag 1). Also a 50/50 mix of
always-short and unit-median on blk2, and an equal-weight mix of the
three paper rules on blk2.
"""
    ),
    code(
        r"""
sig = pd.DataFrame({tag: books[tag]["signal"].loc[common] for tag in MODEL_ORDER})
pos = np.sign(sig).replace(0.0, -1.0)
R = books["blk2"]["R"].loc[common]
um = pd.DataFrame({tag: asl.rule_sizes(books[tag])["unit-median VRP"].loc[common] for tag in MODEL_ORDER})

ens = {}
ens["mean-s then sign"] = np.sign(sig.mean(axis=1)).replace(0.0, -1.0) * R
ens["majority vote"] = np.sign(pos.sum(axis=1)).replace(0.0, -1.0) * R
ens["EW unit-median q"] = um.mean(axis=1) * R

def trailing_w(q: pd.DataFrame, kind: str) -> pd.Series:
    w_hist = []
    idx = q.index
    out = pd.Series(0.0, index=idx)
    for i, t in enumerate(idx):
        if i < 63:
            w = np.ones(q.shape[1]) / q.shape[1]
        else:
            sl = slice(0, i)
            scores = []
            for col in q.columns:
                rp = (q[col] * R).iloc[sl]
                mu, sd = float(rp.mean()), float(rp.std(ddof=1))
                if kind == "sharpe":
                    scores.append(mu / sd if sd > 0 else 0.0)
                else:
                    active = rp - (-R.iloc[sl])
                    a_sd = float(active.std(ddof=1))
                    scores.append(float(active.mean()) / a_sd if a_sd > 0 else 0.0)
            w = np.maximum(np.array(scores, float), 0.0)
            w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
        out.iloc[i] = float((q.iloc[i] * w).sum())
    return out

ens["trail-Sharpe um q"] = trailing_w(um, "sharpe") * R
ens["trail-IR um q"] = trailing_w(um, "ir") * R
blk_sizes = asl.rule_sizes(books["blk2"])
ens["blk2 0.5 AS + 0.5 UM"] = (0.5 * blk_sizes["always short"] + 0.5 * blk_sizes["unit-median VRP"]).loc[common] * R
ens["blk2 EW 3 rules"] = (
    blk_sizes["always short"] + blk_sizes["long-short volatility"] + blk_sizes["unit-median VRP"]
).loc[common] / 3.0 * R
ens["blk2 unit-median (bench)"] = blk_sizes["unit-median VRP"].loc[common] * R
ens["blk2 long-short"] = blk_sizes["long-short volatility"].loc[common] * R
ens["always short"] = blk_sizes["always short"].loc[common] * R

ens_tab = pd.DataFrame({k: asl.rule_row(v, np.sign(v).replace(0, -1)) for k, v in ens.items()}).T
print(ens_tab.to_string())
ens_tab.to_csv(OUT / "ensemble_summary.csv")
pd.DataFrame(ens).to_csv(OUT / "ensemble_daily.csv")
print("saved ensemble_*.csv")
"""
    ),
    md(
        r"""
## 6. Regression of straddle returns on the signal

$R = a + b s$ and $R = a + b\,(s/\mathrm{med}_{u<t}|s_u|)$. OLS with
HAC lags $=6$. $b>0$ would mean the long package pays when the
forecast exceeds implied variance — the paper L/S book dying says
this is near zero or negative.
"""
    ),
    code(
        r"""
reg_rows = []
for tag in MODEL_ORDER:
    px = books[tag].loc[common]
    s = px["signal"].astype(float)
    r = px["R"].astype(float)
    med = s.abs().expanding(min_periods=63).median().shift(1).fillna(s.abs().median())
    for name, x in (("raw s", s), ("s / med|s|", s / med)):
        X = sm.add_constant(x.to_numpy())
        fit = sm.OLS(r.to_numpy(), X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        reg_rows.append({
            "model": LABEL[tag], "x": name,
            "a": float(fit.params[0]), "b": float(fit.params[1]),
            "t_b": float(fit.tvalues[1]), "p_b": float(fit.pvalues[1]),
            "R2": float(fit.rsquared), "n": int(fit.nobs),
        })
reg_tab = pd.DataFrame(reg_rows)
print(reg_tab.to_string(index=False))
reg_tab.to_csv(OUT / "regression_R_on_signal.csv", index=False)
print("reading: b>0 => long straddle when s>0 is the right side; L/S dying => b ~ 0 or <0.")

px = books["blk2"].loc[common]
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.scatter(px["signal"], px["R"], s=8, alpha=0.35)
X = sm.add_constant(px["signal"].to_numpy())
fit = sm.OLS(px["R"].to_numpy(), X).fit()
xx = np.linspace(px["signal"].min(), px["signal"].max(), 50)
ax.plot(xx, fit.params[0] + fit.params[1] * xx, color="C3", lw=1.2)
ax.set_xlabel(r"$s=\widehat{RV}-\mathrm{IV}_{30}^2$")
ax.set_ylabel(r"$R$")
ax.set_title("blk2  $R$ vs $s$")
fig.tight_layout()
fig.savefig(OUT / "regression_R_on_signal_blk2.png", dpi=120, bbox_inches="tight")
display(fig)
plt.close(fig)
"""
    ),
    md(
        r"""
## 7. Vol-space maps

Stand-in for `atm_straddle_volmap.ipynb`: $\hat y\sqrt{B}$ and
$m\sqrt{B}$ against $\mathrm{IV}_{30}$ and $\sqrt{\mathrm{rv\_raw}}$.
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2))
for col, tag in enumerate(("a0", "blk2")):
    px = books[tag].loc[common]
    axes[0, col].scatter(px["iv_30"], px["yhat_vol"], s=7, alpha=0.35, label=r"$\hat y\sqrt{B}$")
    axes[0, col].scatter(px["iv_30"], px["m_vol"], s=7, alpha=0.35, label=r"$m\sqrt{B}$")
    axes[0, col].set_title(LABEL[tag] + r" vs $\mathrm{IV}_{30}$")
    axes[0, col].legend(fontsize=7)
    axes[1, col].scatter(np.sqrt(px["rv_raw"].clip(lower=0)), px["yhat_vol"], s=7, alpha=0.35)
    axes[1, col].scatter(np.sqrt(px["rv_raw"].clip(lower=0)), px["m_vol"], s=7, alpha=0.35)
    axes[1, col].set_title(LABEL[tag] + r" vs $\sqrt{\mathrm{rv_{raw}}}$")
    lo, hi = 0, float(np.nanpercentile(px["iv_30"], 99))
    axes[0, col].plot([lo, hi], [lo, hi], color="k", lw=0.6)
fig.suptitle(r"vol-space maps, 15:30  $y=x$ in black")
fig.tight_layout()
fig.savefig(OUT / "volmap_a0_blk2.png", dpi=120, bbox_inches="tight")
print("saved", OUT / "volmap_a0_blk2.png")
display(fig)
plt.close(fig)

fig, axes = plt.subplots(1, 5, figsize=(14, 2.8), sharey=True)
for ax, tag in zip(axes, ("lgbm", "xgb", "lasso_t", "lasso_f", "enet")):
    px = books[tag].loc[common]
    ax.scatter(px["iv_30"], px["yhat_vol"], s=5, alpha=0.3)
    ax.set_title(tag, fontsize=8)
    ax.set_xlabel(r"$\mathrm{IV}_{30}$")
axes[0].set_ylabel(r"$\hat y\sqrt{B}$")
fig.tight_layout()
fig.savefig(OUT / "volmap_yhat_vs_iv_rest.png", dpi=120, bbox_inches="tight")
display(fig)
plt.close(fig)
"""
    ),
    md(
        r"""
## 8. Other well-motivated weighting strategies

Rank of $|s|$, inverse expanding vol of $R$, dead-zone at half the
expanding median, vol-target of the unit-median book, and a Kelly-ish
$q=\mathrm{clip}(s/\widehat{\mathrm{Var}}(R),-3,3)$. Scored on the
common days. blk2 shown; CSVs for every model.
"""
    ),
    code(
        r"""
extra_names = ["rank-|s|", "inv-vol of R", "dead-zone 0.5 med",
               "vol-target unit-median", "kelly-ish s/var(R)"]
for tag in MODEL_ORDER:
    px = books[tag]
    sizes = asl.extra_weight_sizes(px)
    tab = pd.DataFrame({
        name: asl.rule_row((sizes[name] * px["R"]).loc[common], sizes[name].loc[common])
        for name in extra_names + ["unit-median VRP", "always short"]
    }).T
    tab.to_csv(OUT / f"experimental_weights_{tag}.csv")
    if tag == "blk2":
        print("blk2 extra weights")
        print(tab.to_string())
print("saved experimental_weights_*.csv")
"""
    ),
]

path = Path(__file__).resolve().parent / "atm_straddle_experimental.ipynb"
nbf.write(nb, path)
print("wrote", path)
