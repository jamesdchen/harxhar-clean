"""Proposal 16 (pre-registered, one cell): a long strangle 25 points outside the deck's strikes on buy days,
the deck's short straddle on sell days, scaled to one straddle premium, against the deck's sign(s) rule.
Re-runnable; prints every number in 16_asymmetric_body.md. Run with the 285J python from notebooks/."""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Users/james/CC Allowed/harxhar-0dte-professor/notebooks")
import atm_straddle_lib as asl  # noqa: E402

REPO = "C:/Users/james/CC Allowed/harxhar-0dte-professor"
d = pd.read_parquet(REPO + "/results/atm_straddle_0dte_1530/daily_blk2.parquet")
ch = pd.read_parquet(
    REPO + "/data/spxw_chain.parquet",
    columns=["expiration", "timestamp", "strike", "cp", "bid", "ask"],
)
et = ch["timestamp"].dt.tz_convert("America/New_York")
ch = ch[(et.dt.hour == 15) & (et.dt.minute == 30)].copy()
ch["date"] = pd.to_datetime(ch["expiration"]).dt.normalize()
ch = ch[ch["date"].isin(d.index)]
ch["mid"] = asl.quote_mid(ch["bid"], ch["ask"]).to_numpy()
q = ch.set_index(["date", "cp", "strike"])[["bid", "ask", "mid"]]
W = 25.0
rows = []
for day, r in d.iterrows():
    kc, kp = r["K_c"] + W, r["K_p"] - W
    try:
        c = q.loc[(day, "C", kc)]
        p = q.loc[(day, "P", kp)]
    except KeyError:
        rows.append((day, np.nan, np.nan, np.nan, kc, kp))
        continue
    ok = (
        np.isfinite(c["mid"])
        and np.isfinite(p["mid"])
        and c["bid"] > 0
        and p["bid"] > 0
    )
    rows.append(
        (
            day,
            c["mid"] + p["mid"] if ok else np.nan,
            c["ask"] + p["ask"] if ok else np.nan,
            c["bid"] + p["bid"] if ok else np.nan,
            kc,
            kp,
        )
    )
st = pd.DataFrame(
    rows, columns=["date", "str_mid", "str_ask", "str_bid", "kc_w", "kp_w"]
).set_index("date")
d = d.join(st)
pay_w = np.maximum(d["S_close"] - d["kc_w"], 0) + np.maximum(
    d["kp_w"] - d["S_close"], 0
)
buy = d["signal"] > 0
quoted = buy & d["str_mid"].notna()
print(
    f"buy days {int(buy.sum())}; both 25-pt wings quoted with two-sided quotes on {int(quoted.sum())} ({100 * quoted.mean() / buy.mean():.1f}%)"
)
print(
    f"median strangle premium / straddle premium on those days: {(d.loc[quoted, 'str_mid'] / d.loc[quoted, 'entry']).median():.3f}; median straddle premium {d.loc[quoted, 'entry'].median():.2f} pts, strangle {d.loc[quoted, 'str_mid'].median():.2f} pts"
)
# frame (i): dollars at risk = the straddle premium; strangle units = entry / str_mid
units = (d["entry"] / d["str_mid"]).where(quoted, 1.0)
R_str_mid = (pay_w / d["str_mid"] - 1.0).where(
    quoted, d["R"]
)  # per strangle premium (falls back to the straddle when unquoted)
R_str_crossed = (pay_w / d["str_ask"] - 1.0).where(
    quoted, d["exit"] / (d["ask_c"] + d["ask_p"]) - 1.0
)
deck_mid = d["pos"] * d["R"]
deck_cr = np.where(
    d["pos"] > 0,
    d["exit"] / (d["ask_c"] + d["ask_p"]) - 1,
    1 - d["exit"] / (d["bid_c"] + d["bid_p"]),
)
cell_mid = np.where(
    buy, R_str_mid, -d["R"]
)  # frame (i): same dollars as one straddle premium
cell_cr = np.where(buy, R_str_crossed, 1 - d["exit"] / (d["bid_c"] + d["bid_p"]))
A = np.sqrt(asl.PERIODS_PER_YEAR)


def sh(x):
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * A)


def tt(x):
    x = np.asarray(x, float)
    return float(np.sqrt(len(x)) * x.mean() / x.std(ddof=1))


print(
    f"\nGATE deck: mid Sharpe {sh(deck_mid):.6f} (1.338322), crossed {sh(deck_cr):.3f}"
)
print(
    f"cell (strangle on buy days, per straddle-premium dollars): mid {sh(cell_mid):.3f} (t {tt(cell_mid):.2f}), crossed {sh(cell_cr):.3f} (t {tt(cell_cr):.2f})"
)
b = buy.to_numpy()
print(
    f"buy days only: deck long straddle mean {deck_mid[b].mean():+.3f}, hit {100 * (deck_mid[b] > 0).mean():.0f}%, win {deck_mid[b][deck_mid[b] > 0].mean():+.2f}, loss {deck_mid[b][deck_mid[b] <= 0].mean():+.2f}"
)
print(
    f"               cell long strangle mean {cell_mid[b].mean():+.3f}, hit {100 * (cell_mid[b] > 0).mean():.0f}%, win {cell_mid[b][cell_mid[b] > 0].mean():+.2f}, loss {cell_mid[b][cell_mid[b] <= 0].mean():+.2f}, worthless {100 * (pay_w[b] == 0).mean():.0f}% of buy days"
)
rng = np.random.default_rng(0)
idx = asl.circular_block_bootstrap_idx(rng, len(d), 21, 2000)
dm = np.array([sh(cell_mid[i]) - sh(deck_mid[i]) for i in idx])
dc = np.array([sh(cell_cr[i]) - sh(deck_cr[i]) for i in idx])
diff_c = cell_cr - deck_cr
print(
    f"\npaired cell - deck: mid dSharpe {sh(cell_mid) - sh(deck_mid):+.3f} [{np.percentile(dm, 2.5):+.2f},{np.percentile(dm, 97.5):+.2f}] | crossed dSharpe {sh(cell_cr) - sh(deck_cr):+.3f} [{np.percentile(dc, 2.5):+.2f},{np.percentile(dc, 97.5):+.2f}], daily mean diff {diff_c.mean():+.4f}, t {tt(diff_c):.2f}"
)
