"""Smile relative value: express blk2's accuracy across STRIKES, not size.

Sizing the strip fails because its payoff is premium-dominated and
one-signed. A long/short portfolio across strikes of the SAME expiry,
premium-balanced (long leg spends $1, short leg receives $1), cancels
the variance risk premium to first order; what remains is relative
mispricing across the smile. blk2's role: it calibrates the physical
terminal density — each strike's model value is the mean payoff under
the empirical standardized 10:00->close return distribution scaled by
the (calibrated) remaining-variance forecast. Rank strikes by richness
= 1 - model/mid, short the richest quintile, buy the cheapest, settle
at intrinsic.

The identification is the SWAP TEST: identical machinery with the
variance input replaced by a0 (HAR incumbent) and by the unconditional
expanding mean. If blk2-ranking beats a0-ranking paired by day, the
exogenous increment is expressed in options; if all three tie, the
book trades static smile shape, not the forecast.

Causal everywhere: standardized-move pool, calibration multipliers and
the unconditional mean are expanding (min 63 days), shifted one day.
Books at mids, plus a crossed variant (buy at ask, short at bid) for
the friction floor. Days need >= 20 live OTM quotes.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "spxw_pnl")
ANN = float(np.sqrt(252.0))
BURN = 63
MIN_OTM = 20
Q = 0.2  # long/short quintile fraction of the OTM ladder
DAILY_0DTE = np.datetime64("2022-05-16")


def _sh(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3 or float(x.std()) == 0.0:
        return float("nan")
    return float(x.mean() / x.std())


def _tstat(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3 or float(x.std(ddof=1)) == 0.0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(x.size)))


def _leg_pnl(intr: np.ndarray, cost: np.ndarray) -> float:
    """Equal-dollar-per-strike leg: intrinsic returned per $1 of premium."""
    return float(np.mean(intr / np.maximum(cost, 1e-18)))


def main() -> None:
    tr = pd.read_parquet(os.path.join(OUT, "mfiv_toclose_trades.parquet"))
    tr = tr.sort_values("t0").reset_index(drop=True)
    tr["expiration"] = pd.to_datetime(tr["expiration"])

    ch = pd.read_parquet(
        os.path.join(ROOT, "data", "spxw_chain.parquet"),
        columns=["expiration", "strike", "cp", "timestamp", "bid", "ask", "mid"],
    )
    ch = ch[ch["timestamp"].isin(set(tr["t0"]))]

    # causal calibration multipliers and unconditional forecast
    rvr = tr["rv_rem"].to_numpy(float)
    c_b = (
        pd.Series(rvr / np.maximum(tr["pb_rem"].to_numpy(float), 1e-18))
        .expanding(min_periods=BURN)
        .mean()
        .shift(1)
        .to_numpy(float)
    )
    c_a = (
        pd.Series(rvr / np.maximum(tr["pa_rem"].to_numpy(float), 1e-18))
        .expanding(min_periods=BURN)
        .mean()
        .shift(1)
        .to_numpy(float)
    )
    unc = pd.Series(rvr).expanding(min_periods=BURN).mean().shift(1).to_numpy(float)
    v_in = {
        "blk2": c_b * tr["pb_rem"].to_numpy(float),
        "a0": c_a * tr["pa_rem"].to_numpy(float),
        "uncond": unc,
    }

    # causal pool of standardized 10:00->close moves
    umove = np.log(tr["S_T"].to_numpy(float) / tr["F"].to_numpy(float)) / np.sqrt(
        np.maximum(rvr, 1e-18)
    )

    day_rows = []
    grouped = dict(list(ch.groupby("expiration")))
    for i in range(len(tr)):
        if i < BURN or not all(np.isfinite(v[i]) for v in v_in.values()):
            continue
        exp_d = tr["expiration"].iloc[i]
        g = grouped.get(exp_d)
        if g is None:
            continue
        F = float(tr["F"].iloc[i])
        S_T = float(tr["S_T"].iloc[i])
        otm = g[
            (g["mid"] > 0)
            & (
                ((g["cp"] == "P") & (g["strike"] < F))
                | ((g["cp"] == "C") & (g["strike"] > F))
            )
        ]
        if len(otm) < MIN_OTM:
            continue
        K = otm["strike"].to_numpy(float)
        is_call = (otm["cp"] == "C").to_numpy()
        mid = otm["mid"].to_numpy(float)
        bid = otm["bid"].to_numpy(float)
        ask = otm["ask"].to_numpy(float)
        intr = np.where(is_call, np.maximum(S_T - K, 0.0), np.maximum(K - S_T, 0.0))

        U = umove[:i]
        U = U[np.isfinite(U)]
        if U.size < BURN:
            continue

        row: dict = {"expiration": exp_d, "n_otm": len(otm)}
        for name, v in v_in.items():
            sdev = np.sqrt(max(float(v[i]), 1e-18))
            S = F * np.exp(sdev * U)
            S *= F / S.mean()  # martingale normalization
            pay = np.where(
                is_call[:, None],
                np.maximum(S[None, :] - K[:, None], 0.0),
                np.maximum(K[:, None] - S[None, :], 0.0),
            )
            model = pay.mean(axis=1)
            rich = 1.0 - model / mid
            nq = max(int(len(otm) * Q), 1)
            order = np.argsort(rich)
            lo, hi = order[:nq], order[-nq:]
            row[f"pnl_{name}"] = _leg_pnl(intr[lo], mid[lo]) - _leg_pnl(
                intr[hi], mid[hi]
            )
            # friction floor: buy at ask, short at bid. A short needs a live
            # bid — restrict the rich leg to bid > 0 (tradability, not a
            # tuning constant); the long leg needs a live ask likewise.
            lox = lo[ask[lo] > 0]
            hix = order[::-1][(bid[order[::-1]] > 0)][:nq]
            if lox.size and hix.size:
                row[f"pnlx_{name}"] = _leg_pnl(intr[lox], ask[lox]) - _leg_pnl(
                    intr[hix], bid[hix]
                )
            else:
                row[f"pnlx_{name}"] = float("nan")
        day_rows.append(row)

    d = pd.DataFrame(day_rows)
    d.to_parquet(os.path.join(OUT, "smile_rv_daily.parquet"))

    rows = []
    eras = {
        "all": np.ones(len(d), bool),
        "daily_0dte": d["expiration"].to_numpy() >= DAILY_0DTE,
    }
    for era, msk in eras.items():
        for name in v_in:
            for pref, tag in (("pnl", "mid"), ("pnlx", "crossed")):
                p = d[f"{pref}_{name}"].to_numpy(float)[msk]
                rows.append(
                    {
                        "era": era,
                        "ranking": name,
                        "px": tag,
                        "n": int(msk.sum()),
                        "mean": float(np.nanmean(p)),
                        "sharpe_ann": _sh(p) * ANN,
                        "hit": float((p > 0).mean()),
                    }
                )
        diff = (d["pnl_blk2"] - d["pnl_a0"]).to_numpy(float)[msk]
        rows.append(
            {
                "era": era,
                "ranking": "blk2_minus_a0(paired)",
                "px": "mid",
                "n": int(msk.sum()),
                "mean": float(np.nanmean(diff)),
                "sharpe_ann": _tstat(diff),
                "hit": float((diff > 0).mean()),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "smile_rv_summary.csv"), index=False)
    print(f"days scored: {len(d)}", flush=True)
    print(out.to_string(index=False), flush=True)
    print(
        "\nnote: paired row's sharpe_ann column holds the t-stat of the "
        "daily blk2-a0 difference, not an annualized Sharpe.",
        flush=True,
    )


if __name__ == "__main__":
    main()
