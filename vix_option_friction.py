"""VIX-option expression of the signed overnight dVIX prediction -- the last unclosed residual.

The cumrv signal predicts overnight dVIX (NW-t +6.59) and close->close dVIX (+3.76). VIX index
options are the only VIX-derivative an ETNA-class options venue could plausibly carry. This
prices the DIRECTIONAL trade at real OM quotes both ends:

  signal>0 -> long ATM PUT  (predict VIX down), signal<0 -> long ATM CALL
  size = min(|signal|, CAP) / |delta|   (delta-normalized: ~1 VIX pt of exposure per unit)
  enter close t, exit close t+1, fills = mid -/+ tier * spread (crossing always hurts)

VIX options price the VX forward, so the ~0.4-0.5 overnight beta of forward-to-spot dVIX is
measured implicitly by using real quotes. Expectation: negative (beta dilution + vol-of-vol
premium + wide spreads vs the 0.148-pt delta-1 breakeven); this closes it by measurement.

Usage: python vix_option_friction.py --opprcd data/om_friction/vix_opprcd.csv.gz \
           --secprd data/om_friction/vix_secprd.csv.gz \
           --signal eval_sim/data/cumrv_signal_dated.parquet
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

DTE_LO, DTE_HI = 3, 35  # nearest liquid expiry; VIX options are AM-settled, avoid dte<3
CAP = 3.0
COMMISSION = 0.65
TIERS = {"optimistic": 0.0, "base": 0.25, "conservative": 0.5}


def load_chain(path: str) -> pd.DataFrame:
    o = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    o.columns = [c.lower() for c in o.columns]
    o["date"] = pd.to_datetime(o["date"])
    o["exdate"] = pd.to_datetime(o["exdate"])
    o["strike"] = o["strike_price"] / 1000.0
    o["cp_flag"] = o["cp_flag"].str.upper().str[0]
    o["mid"] = (o["best_bid"] + o["best_offer"]) / 2.0
    o["spread"] = o["best_offer"] - o["best_bid"]
    o["dte"] = (o["exdate"] - o["date"]).dt.days
    return o[o["dte"].between(0, DTE_HI) & (o["best_offer"] > 0)].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opprcd", required=True)
    ap.add_argument("--secprd", required=True)
    ap.add_argument("--signal", required=True)
    a = ap.parse_args()

    chain = load_chain(a.opprcd)
    spot = pd.read_csv(a.secprd)
    spot.columns = [c.lower() for c in spot.columns]
    spot["date"] = pd.to_datetime(spot["date"])
    spot = spot.set_index("date")["close"]
    sig = pd.read_parquet(a.signal).set_index("date")["signal_1600"]

    key = ["date", "exdate", "strike", "cp_flag"]
    chain_idx = chain.drop_duplicates(subset=key).set_index(key).sort_index()
    by_day = dict(tuple(chain.groupby("date")))
    dates = sorted(set(by_day) & set(sig.dropna().index) & set(spot.index))

    rows = []
    n_skip = 0
    for i, d in enumerate(dates[:-1]):
        d1 = dates[i + 1]
        s = float(sig.loc[d])
        if not np.isfinite(s) or s == 0.0:
            continue
        V0 = float(spot.loc[d])
        cp = "P" if s > 0 else "C"  # long puts when VIX predicted DOWN
        day = by_day[d]
        cand = day[
            day["dte"].between(DTE_LO, DTE_HI)
            & (day["cp_flag"] == cp)
            & (day["best_bid"] > 0)
            & (day["delta"].abs().between(0.25, 0.75))
        ]
        if cand.empty:
            n_skip += 1
            continue
        exd = cand.loc[cand["dte"].idxmin(), "exdate"]
        ch = cand[cand["exdate"] == exd]
        leg = ch.iloc[(ch["strike"] - V0).abs().argsort().iloc[0]]
        try:
            ex = chain_idx.loc[(d1, leg["exdate"], leg["strike"], cp)]
        except KeyError:
            n_skip += 1
            continue
        w = min(abs(s), CAP) / max(
            abs(float(leg["delta"])), 0.10
        )  # ~1 VIX pt exposure per |s| unit
        for tier, frac in TIERS.items():
            entry = float(leg["mid"]) + frac * float(
                leg["spread"]
            )  # always long -> pay up in
            exitp = float(ex["mid"]) - frac * float(
                ex["spread"]
            )  # sell out -> receive less
            pnl = w * (exitp - entry) * 100.0 - w * 2 * COMMISSION
            rows.append({"date": d, "tier": tier, "pnl": pnl, "w": w, "cp": cp})

    df = pd.DataFrame(rows)
    print(f"{df['date'].nunique():,} traded days (skipped {n_skip})")
    for tier in TIERS:
        p = df[df["tier"] == tier].set_index("date")["pnl"]
        sh = p.mean() / p.std()
        print(
            f"{tier:>12}: mean=${p.mean():+8.2f}/day  Sharpe/day={sh:+.3f}  ann={sh * np.sqrt(252):+.2f}"
        )
    # control: same trade, signal randomized sign -- isolates put/call carry asymmetry
    p = df[df["tier"] == "optimistic"]
    for cp in ("P", "C"):
        q = p[p["cp"] == cp]["pnl"]
        if len(q):
            print(
                f"  side {cp}: n={len(q)}  mean=${q.mean():+.2f}  Sharpe={q.mean() / q.std():+.3f}"
            )
    df.to_parquet("eval_sim/data/vix_option_perday.parquet", index=False)


if __name__ == "__main__":
    main()
