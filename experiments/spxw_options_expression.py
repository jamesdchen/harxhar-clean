"""Can blk2's forecast accuracy be harnessed through 0DTE options?

Prior result (spxw_edge_vs_market.py): sign/size books vs the market
strike fail because sign(RV - MFIV) is essentially constant (VRP), and a
constant-multiplier debias left QLIKE 0.27 vs blk2's 0.064 — possibly a
debias artifact. Two questions decide whether an options expression
exists at all:

  A. ENCOMPASSING (causal): expanding log-log OLS of remaining RV on
     MFIV alone (the fair, state-dependent market benchmark) vs MFIV +
     blk2 jointly. If the joint fit beats MFIV-only OOS, the smile is
     CONDITIONALLY mispriced and blk2 has content beyond it; if not,
     the QLIKE gap was debias misspecification and no listed expression
     exists.
  B. EXPRESSION: the implementable trade is a variable-size short-vol
     program: base short of one vega-normalized strip per day (carries
     the VRP), overlay k*z_t where z_t is the causally standardized
     conditional-mispricing signal log f_joint - log f_market. The
     overlay is zero-mean over time so it cancels the premium wedge by
     construction; its value is cov(z, surprise). Also: abstain books
     (skip the top-z days) and the tail table P(RV > MFIV) by z
     quintile (can the model dodge the blow-through days?).

Causality: all regressions/moments expanding with min 63 obs (one
quarter), applied shifted one day; back-transform via QLIKE-optimal
multiplicative factor, also expanding+shifted. Overlay grid k in
{0.25, 0.5, 1} and abstain quantiles {0.8, 0.9} are documented grids,
not tuned constants. Mids only — no bid/ask on disk.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "spxw_pnl")
ANN = float(np.sqrt(252.0))
BURN = 63
DAILY_0DTE = np.datetime64("2022-05-16")


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
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size < 3 or float(d.std(ddof=1)) == 0.0:
        return float("nan")
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))


def _expanding_ols_forecast(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Row-t forecast from OLS fit on rows [0, t) (min BURN rows); NaN before."""
    n, p = X.shape
    Z = np.column_stack([np.ones(n), X])
    out = np.full(n, np.nan)
    for t in range(BURN, n):
        zt, yt = Z[:t], y[:t]
        beta, *_ = np.linalg.lstsq(zt, yt, rcond=None)
        out[t] = Z[t] @ beta
    return out


def _causal_scale(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """QLIKE-optimal multiplicative back-transform, expanding + shifted."""
    r = pd.Series(y / np.maximum(f, 1e-18))
    return r.expanding(min_periods=BURN).mean().shift(1).to_numpy(float) * f


def _causal_z(x: np.ndarray) -> np.ndarray:
    s = pd.Series(x)
    mu = s.expanding(min_periods=BURN).mean().shift(1)
    sd = s.expanding(min_periods=BURN).std().shift(1)
    return ((s - mu) / sd).to_numpy(float)


def main() -> None:
    tr = pd.read_parquet(os.path.join(OUT, "mfiv_toclose_trades.parquet"))
    tr = tr.sort_values("t0").reset_index(drop=True)
    rv = tr["rv_rem"].to_numpy(float)
    b2 = tr["pb_rem"].to_numpy(float)
    iv = tr["mfiv_int"].to_numpy(float)
    t0 = pd.to_datetime(tr["expiration"]).to_numpy()
    ok = np.isfinite(rv) & np.isfinite(b2) & np.isfinite(iv) & (rv > 0)
    rv, b2, iv, t0 = rv[ok], b2[ok], iv[ok], t0[ok]

    ly = np.log(rv)
    lm = np.log(np.maximum(iv, 1e-18)).reshape(-1, 1)
    lb = np.log(np.maximum(b2, 1e-18)).reshape(-1, 1)

    lf_m = _expanding_ols_forecast(ly, lm)  # market-only benchmark
    lf_b = _expanding_ols_forecast(ly, lb)  # model-only
    lf_j = _expanding_ols_forecast(ly, np.hstack([lm, lb]))  # joint

    cand = {
        "mfiv_loglog": np.exp(lf_m),
        "blk2_loglog": np.exp(lf_b),
        "joint_loglog": np.exp(lf_j),
        "blk2_raw": b2,
    }
    cand = {k: _causal_scale(rv, f) for k, f in cand.items()}
    s = np.ones(rv.size, bool)
    for f in cand.values():
        s &= np.isfinite(f)

    # A. encompassing QLIKE table
    rows = []
    base = _ql(cand["mfiv_loglog"][s], rv[s])
    for name, f in cand.items():
        q = _ql(f[s], rv[s])
        rows.append(
            {
                "forecast": name,
                "n": int(s.sum()),
                "qlike": float(q.mean()),
                "dm_vs_mfiv_loglog": _dm(q - base),
            }
        )
    enc = pd.DataFrame(rows)
    enc.to_csv(os.path.join(OUT, "options_expression_encompassing.csv"), index=False)

    # final full-sample joint coefficients, for the record
    Z = np.column_stack([np.ones(s.sum()), lm[s, 0], lb[s, 0]])
    beta, *_ = np.linalg.lstsq(Z, ly[s], rcond=None)
    print(
        f"joint log-log coefficients (full sample, descriptive): "
        f"const={beta[0]:+.3f}  log_mfiv={beta[1]:+.3f}  log_blk2={beta[2]:+.3f}",
        flush=True,
    )
    print(enc.to_string(index=False), flush=True)

    # B. books: base short + overlay / abstain, vs the RAW market strike
    pay = (rv - iv) / np.maximum(iv, 1e-18)  # per unit vega notional
    pay_dollar = rv - iv  # variance units
    z = _causal_z(lf_j - lf_m)  # conditional mispricing of the smile
    sb = s & np.isfinite(z) & np.isfinite(pay)

    books: dict[str, np.ndarray] = {
        "const_short_vega": (-pay)[sb],
        "const_short_dollar": (-pay_dollar)[sb],
        "overlay_pure_z": (z * pay)[sb],
    }
    for k in (0.25, 0.5, 1.0):
        books[f"short_var_k{k}"] = (-(1.0 - k * z) * pay)[sb]
    # premium-measurement books: blk2 makes the premium observable at
    # entry (prem = 1 - blk2/MFIV); size the short by it.
    prem = 1.0 - b2 / np.maximum(iv, 1e-18)
    zprem = _causal_z(np.log(np.maximum(iv, 1e-18) / np.maximum(b2, 1e-18)))
    books["short_prem_direct"] = (-prem * pay)[sb]
    for k in (0.25, 0.5):
        books[f"short_prem_z{k}"] = (-(1.0 + k * zprem) * pay)[sb]
    for aq in (0.8, 0.9):
        thr = (
            pd.Series(z)
            .expanding(min_periods=BURN)
            .quantile(aq)
            .shift(1)
            .to_numpy(float)
        )
        books[f"abstain_q{aq}"] = (np.where(z < thr, -pay, 0.0))[sb]

    rows = []
    eras = {"all": np.ones(int(sb.sum()), bool), "daily_0dte": t0[sb] >= DAILY_0DTE}
    for era, msk in eras.items():
        for name, pnl in books.items():
            p = pnl[msk]
            rows.append(
                {
                    "era": era,
                    "book": name,
                    "n": int(msk.sum()),
                    "mean": float(np.nanmean(p)),
                    "sharpe_ann": _sh(p) * ANN,
                    "hit": float((p > 0).mean()),
                }
            )
    bk = pd.DataFrame(rows)
    bk.to_csv(os.path.join(OUT, "options_expression_books.csv"), index=False)
    print("\n" + bk.to_string(index=False), flush=True)

    # C. tail table: can z dodge the blow-through (RV > MFIV) days?
    zq = pd.qcut(z[sb], 5, labels=False)
    tail = (
        pd.DataFrame({"zq": zq, "blow": (rv[sb] > iv[sb]).astype(int)})
        .groupby("zq")["blow"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )
    tail.to_csv(os.path.join(OUT, "options_expression_tail.csv"), index=False)
    print("\nblow-through days (RV > raw MFIV) by causal-z quintile:", flush=True)
    print(tail.to_string(index=False), flush=True)

    # D. risk-budget accounting: what does dodging the top-z days buy?
    thr80 = (
        pd.Series(z).expanding(min_periods=BURN).quantile(0.8).shift(1).to_numpy(float)
    )
    inb = sb & np.isfinite(thr80)
    zt = z[inb] < thr80[inb]
    p = pay[inb]
    short = -p
    rows = []
    for name, m in (
        ("all_days", np.ones(p.size, bool)),
        ("z_below_q80 (traded)", zt),
        ("z_above_q80 (dodged)", ~zt),
    ):
        x = short[m]
        rows.append(
            {
                "set": name,
                "n": int(m.sum()),
                "mean_shortpay": float(x.mean()),
                "worst_day": float(x.min()),
                "n_loss_days": int((x < 0).sum()),
                "sum_losses": float(x[x < 0].sum()) if (x < 0).any() else 0.0,
                "hit": float((x > 0).mean()),
            }
        )
    rb = pd.DataFrame(rows)
    rb.to_csv(os.path.join(OUT, "options_expression_riskbudget.csv"), index=False)
    print("\nrisk-budget accounting (short book, per unit vega):", flush=True)
    print(rb.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
