"""0DTE-style variance to the 16:00 ET close, struck at remaining a0.

Uses dumped yhats only (no chain). Overlapping every-bar entries are reported
but the tradable books are non-overlapping: enter once at 10:00, or last bar only.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "spxw_pnl")
ANN_BAR = float(np.sqrt(252.0 * 48.0))
ANN_DAY = float(np.sqrt(252.0))
RTH = (10, 11, 12, 13, 14, 15)


def _sh(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3 or float(x.std()) == 0.0:
        return float("nan")
    return float(x.mean() / x.std())


def _ql(f: np.ndarray, y: np.ndarray) -> np.ndarray:
    f = np.maximum(f, 1e-18)
    y = np.maximum(y, 1e-18)
    return y / f - np.log(y / f) - 1.0


def _ann(x: np.ndarray, daily: bool) -> float:
    s = _sh(x)
    return s * (ANN_DAY if daily else ANN_BAR) if np.isfinite(s) else float("nan")


def _report(name: str, x: np.ndarray, daily: bool) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    sh = _sh(x)
    row = {
        "book": name,
        "n": int(x.size),
        "mean": float(x.mean()) if x.size else float("nan"),
        "sharpe": sh,
        "sharpe_ann": _ann(x, daily),
        "hit": float((x > 0).mean()) if x.size else float("nan"),
        "unit": "day" if daily else "bar",
    }
    print(
        f"{name:42s} n={row['n']:6d}  mean={row['mean']:+.4e}  "
        f"ann={row['sharpe_ann']:+.2f}  hit={row['hit']:.1%}  ({row['unit']})",
        flush=True,
    )
    return row


def main() -> None:
    a0 = pd.read_parquet(os.path.join(OUT, "yhat_a0.parquet"))
    b2 = pd.read_parquet(os.path.join(OUT, "yhat_blk2.parquet"))
    a0["t"] = pd.to_datetime(a0["t"], utc=True).astype("datetime64[ns, UTC]")
    b2["t"] = pd.to_datetime(b2["t"], utc=True).astype("datetime64[ns, UTC]")
    m = a0.merge(b2, on="t", suffixes=("_a", "_b"))
    m["pa"] = m["yhat_a"].to_numpy(float) ** 2 * m["baseline_a"].to_numpy(float)
    m["pb"] = m["yhat_b"].to_numpy(float) ** 2 * m["baseline_b"].to_numpy(float)
    m["rv"] = m["rv_raw_a"].to_numpy(float)
    ok = np.isfinite(m["rv"]) & np.isfinite(m["pa"]) & np.isfinite(m["pb"])
    ok &= (m["rv"] > 0) & (m["pa"] > 0) & (m["pb"] > 0)
    m = m.loc[ok].copy()
    m["et"] = m["t"].dt.tz_convert("America/New_York")
    m["day"] = m["et"].dt.normalize()
    m["hod"] = m["et"].dt.hour
    m["absrel"] = np.abs(np.log(m["pb"] / m["pa"]))
    d10 = float(m["absrel"].quantile(0.9))
    m = m.sort_values("t")

    # remaining-to-close on each ET calendar day (includes overnight bars that day)
    g = m.groupby("day", sort=False)
    m["rv_rem"] = g["rv"].transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1])
    m["pa_rem"] = g["pa"].transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1])
    m["pb_rem"] = g["pb"].transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1])
    m["n_rem"] = g.cumcount(ascending=False) + 1

    # RTH-only remaining (10:00-16:00 ET bars)
    rth = m[m["hod"].isin(RTH)].copy()
    gr = rth.groupby("day", sort=False)
    rth["rv_rem"] = gr["rv"].transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1])
    rth["pa_rem"] = gr["pa"].transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1])
    rth["pb_rem"] = gr["pb"].transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1])

    gap = rth["pb_rem"] - rth["pa_rem"]
    rth["pnl_unit"] = gap * (rth["rv_rem"] - rth["pa_rem"])
    pa2 = np.maximum(rth["pa_rem"].to_numpy(float), 1e-18) ** 2
    size = gap.to_numpy(float) / pa2
    lo, hi = np.nanquantile(size, [0.01, 0.99])
    rth["pnl_hess"] = np.clip(size, lo, hi) * (rth["rv_rem"] - rth["pa_rem"])
    rth["dq_rem"] = _ql(
        rth["pa_rem"].to_numpy(float), rth["rv_rem"].to_numpy(float)
    ) - _ql(rth["pb_rem"].to_numpy(float), rth["rv_rem"].to_numpy(float))
    rth["keep"] = rth["absrel"] <= d10

    # one entry per day: first RTH-post-open bar (10:00 hour)
    first = rth[rth["hod"] == 10].groupby("day", sort=False).head(1)
    last = rth.groupby("day", sort=False).tail(1)
    am = rth[rth["hod"].isin([10, 11])]
    filt = rth[rth["keep"]]
    first_f = first[first["absrel"] <= d10]
    last_f = last[last["absrel"] <= d10]
    am_f = am[am["absrel"] <= d10]

    print(
        f"D10 |log| = {d10:.4f}  RTH bars={len(rth):,}  days={rth['day'].nunique()}",
        flush=True,
    )
    rows = []
    one = _ql(filt["pa"].to_numpy(float), filt["rv"].to_numpy(float)) - _ql(
        filt["pb"].to_numpy(float), filt["rv"].to_numpy(float)
    )
    rows.append(_report("1bar QLIKE RTH drop D10", one, False))

    rows.append(_report("to-close QLIKE overlapping RTH", rth["dq_rem"], False))
    rows.append(_report("to-close QLIKE overlapping drop D10", filt["dq_rem"], False))
    rows.append(_report("to-close unit 10:00 entry", first["pnl_unit"], True))
    rows.append(_report("to-close unit 10:00 drop D10", first_f["pnl_unit"], True))
    rows.append(_report("to-close hess 10:00 drop D10", first_f["pnl_hess"], True))
    rows.append(_report("to-close QLIKE 10:00 drop D10", first_f["dq_rem"], True))
    rows.append(_report("last-bar unit drop D10", last_f["pnl_unit"], True))
    rows.append(_report("last-bar QLIKE drop D10", last_f["dq_rem"], True))
    rows.append(_report("last-bar hess drop D10", last_f["pnl_hess"], True))
    rows.append(_report("AM overlapping unit drop D10", am_f["pnl_unit"], False))
    rows.append(_report("AM overlapping QLIKE drop D10", am_f["dq_rem"], False))

    out = pd.DataFrame(rows)
    path = os.path.join(OUT, "toclose_varswap.csv")
    out.to_csv(path, index=False)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
