"""Is the exogenous revision an edge against the MARKET's strike?

The +3.2 Hessian book scores (RV - a0): a strike nobody offers. A real
counterparty strikes at MFIV, so the only market PnL is N*(RV - MFIV).
This scores the revision against the market forecast, fairly debiased:

  A. Forecast horse race on remaining 10:00->16:00 variance: a0, blk2,
     raw MFIV, causally debiased MFIV (expanding QLIKE-optimal scale),
     and debiased MFIV multiplied by the revision ratio (blk2/a0).
  B. Information: corr of log revision with log(RV / debiased MFIV),
     and sign agreement of the revision with (RV - debiased MFIV).
  C. Economic: Hessian book against raw and debiased market strikes.
  D. Time-of-day: every-bar paper vs listed-MTM Sharpe by bar of day
     (vega dies into the close; does listed converge to paper?), full
     sample and daily-0DTE era (expirations >= 2022-05-16).

Debias burn-in: expanding mean of RV/MFIV with min 63 obs (a quarter of
trading days); 126 reported as robustness. Scored rows start after it.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "spxw_pnl")
ANN_DAY = float(np.sqrt(252.0))
ANN_BAR = float(np.sqrt(252.0 * 48.0))  # matches spxw_mfiv_everybar.py
DAILY_0DTE = pd.Timestamp("2022-05-16")


def _ql(f: np.ndarray, y: np.ndarray) -> np.ndarray:
    f = np.maximum(f, 1e-18)
    y = np.maximum(y, 1e-18)
    return y / f - np.log(y / f) - 1.0


def _sh(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3 or float(x.std()) == 0.0:
        return float("nan")
    return float(x.mean() / x.std())


def _dm(d: np.ndarray) -> float:
    """DM t on a paired loss-difference series (iid s.e.; daily, h=0)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size < 3:
        return float("nan")
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))


def part_a_to_close() -> pd.DataFrame:
    tr = pd.read_parquet(os.path.join(OUT, "mfiv_toclose_trades.parquet"))
    tr = tr.sort_values("t0").reset_index(drop=True)
    rv = tr["rv_rem"].to_numpy(float)
    a0 = tr["pa_rem"].to_numpy(float)
    b2 = tr["pb_rem"].to_numpy(float)
    iv = tr["mfiv_int"].to_numpy(float)
    ok = np.isfinite(rv) & np.isfinite(a0) & np.isfinite(b2) & np.isfinite(iv)
    rv, a0, b2, iv = rv[ok], a0[ok], b2[ok], iv[ok]
    n = rv.size

    rows = []
    for burn in (63, 126):
        # QLIKE-optimal multiplicative debias: m* = E[y/f], causal (shifted).
        def deb(f: np.ndarray) -> np.ndarray:
            r = pd.Series(rv / np.maximum(f, 1e-18))
            m = r.expanding(min_periods=burn).mean().shift(1).to_numpy(float)
            return m * f

        cand = {
            "a0": a0,
            "blk2": b2,
            "mfiv_raw": iv,
            "mfiv_deb": deb(iv),
            "a0_deb": deb(a0),
            "blk2_deb": deb(b2),
            "mfiv_deb_x_rev": deb(iv) * (b2 / np.maximum(a0, 1e-18)),
        }
        scored = np.ones(n, bool)
        for f in cand.values():
            scored &= np.isfinite(f)
        base = _ql(cand["mfiv_deb"][scored], rv[scored])
        for name, f in cand.items():
            q = _ql(f[scored], rv[scored])
            rows.append(
                {
                    "burn": burn,
                    "forecast": name,
                    "n": int(scored.sum()),
                    "qlike": float(q.mean()),
                    "dm_vs_mfiv_deb": _dm(q - base),
                }
            )

        # Information content of the revision vs the debiased market strike.
        miv = cand["mfiv_deb"]
        s = scored
        lrev = np.log(np.maximum(b2[s], 1e-18) / np.maximum(a0[s], 1e-18))
        lsur = np.log(np.maximum(rv[s], 1e-18) / np.maximum(miv[s], 1e-18))
        gap = b2[s] - a0[s]
        hess = gap / np.maximum(a0[s], 1e-18) ** 2
        info = {
            "burn": burn,
            "n": int(s.sum()),
            "corr_lrev_lsurprise": float(np.corrcoef(lrev, lsur)[0, 1]),
            "sign_agree_vs_deb": float(
                (np.sign(gap) == np.sign(rv[s] - miv[s])).mean()
            ),
            "sh_hess_vs_raw_mfiv": _sh(hess * (rv[s] - iv[s])) * ANN_DAY,
            "sh_hess_vs_deb_mfiv": _sh(hess * (rv[s] - miv[s])) * ANN_DAY,
            "sh_hess_vs_a0": _sh(hess * (rv[s] - a0[s])) * ANN_DAY,
        }
        rows.append({"burn": burn, "forecast": "_info", "n": info["n"], **info})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "edge_vs_market_toclose.csv"), index=False)
    return out


def part_c_deployable_books() -> pd.DataFrame:
    """Expiry-settled books, 10:00 entry, per unit vega notional.

    Market payoff per unit notional is pay = (RV - MFIV)/MFIV: short the
    strip earns -pay. Positions are unitless functions of causal info
    (blk2, a0, debiased MFIV), so Sharpe is scale-free — no dollar-Hessian
    weighting to drown the log-scale information.
    """
    tr = pd.read_parquet(os.path.join(OUT, "mfiv_toclose_trades.parquet"))
    tr = tr.sort_values("t0").reset_index(drop=True)
    rv = tr["rv_rem"].to_numpy(float)
    a0 = tr["pa_rem"].to_numpy(float)
    b2 = tr["pb_rem"].to_numpy(float)
    iv = tr["mfiv_int"].to_numpy(float)
    t0 = pd.to_datetime(tr["expiration"]).to_numpy()
    ok = np.isfinite(rv) & np.isfinite(a0) & np.isfinite(b2) & np.isfinite(iv)
    rv, a0, b2, iv, t0 = rv[ok], a0[ok], b2[ok], iv[ok], t0[ok]

    burn = 63
    r = pd.Series(rv / np.maximum(iv, 1e-18))
    m = r.expanding(min_periods=burn).mean().shift(1).to_numpy(float)
    miv = m * iv
    s = np.isfinite(miv)
    rv, a0, b2, iv, miv, t0 = rv[s], a0[s], b2[s], iv[s], miv[s], t0[s]

    pay = (rv - iv) / np.maximum(iv, 1e-18)
    lvl = np.log(np.maximum(b2, 1e-18) / np.maximum(miv, 1e-18))
    lrev = np.log(np.maximum(b2, 1e-18) / np.maximum(a0, 1e-18))
    lsur = np.log(np.maximum(rv, 1e-18) / np.maximum(miv, 1e-18))

    books = {
        "always_short": (-np.ones_like(pay), pay),
        "timing_sign_blk2_vs_deb": (np.sign(lvl), pay),
        "sized_level_blk2_vs_deb": (lvl, pay),
        "sized_revision": (lrev, pay),
        "info_rev_x_lsurprise": (lrev, lsur),
        "info_lvl_x_lsurprise": (lvl, lsur),
    }
    eras = {
        "all": np.ones(rv.size, bool),
        "drop_2020": t0 >= np.datetime64("2021-01-01"),
        "daily_0dte": t0 >= np.datetime64(DAILY_0DTE.date()),
    }
    rows = []
    for era, msk in eras.items():
        for name, (pos, x) in books.items():
            pnl = (pos * x)[msk]
            rows.append(
                {
                    "era": era,
                    "book": name,
                    "n": int(msk.sum()),
                    "mean": float(np.nanmean(pnl)),
                    "sharpe_ann": _sh(pnl) * ANN_DAY,
                    "hit": float((pnl > 0).mean()),
                    "frac_long": float((pos[msk] > 0).mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "edge_vs_market_books.csv"), index=False)
    return out


def part_b_time_of_day() -> pd.DataFrame:
    tr = pd.read_parquet(os.path.join(OUT, "everybar_mtm_trades.parquet"))
    tr["expiration"] = pd.to_datetime(tr["expiration"])
    tr["bar"] = tr["et"].dt.hour * 2 + tr["et"].dt.minute // 30  # 20..31

    rows = []
    for era, sub in (
        ("all", tr),
        ("daily_0dte", tr[tr["expiration"] >= DAILY_0DTE]),
    ):
        for bar, g in sub.groupby("bar"):
            rows.append(
                {
                    "era": era,
                    "bar_et": f"{bar // 2:02d}:{30 * (bar % 2):02d}",
                    "n": len(g),
                    "sh_paper": _sh(g["paper"].to_numpy()) * ANN_BAR,
                    "sh_mtm": _sh(g["mtm"].to_numpy()) * ANN_BAR,
                    "hit_mtm": float((g["mtm"] > 0).mean()),
                }
            )
        # Last-k-bars books: the deployable "trade only when vega is dead".
        for k in (1, 2, 3, 4):
            last = sub[sub["bar"] >= sub["bar"].max() - (k - 1)]
            d = last.groupby("day")[["paper", "mtm"]].sum()
            rows.append(
                {
                    "era": era,
                    "bar_et": f"last{k}_daily",
                    "n": len(d),
                    "sh_paper": _sh(d["paper"].to_numpy()) * ANN_DAY,
                    "sh_mtm": _sh(d["mtm"].to_numpy()) * ANN_DAY,
                    "hit_mtm": float((d["mtm"] > 0).mean()),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "edge_vs_market_timeofday.csv"), index=False)
    return out


def main() -> None:
    a = part_a_to_close()
    print("=== A/B/C: to-close, market strike ===", flush=True)
    print(a.to_string(index=False), flush=True)
    c = part_c_deployable_books()
    print("\n=== C: expiry-settled books vs market strike ===", flush=True)
    print(c.to_string(index=False), flush=True)
    b = part_b_time_of_day()
    print("\n=== D: time-of-day, paper vs listed MTM ===", flush=True)
    print(b.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
