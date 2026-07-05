"""Friction pricing of the cumrv x close edge in REAL option quotes (the datum-to-buy).

Takes the OptionMetrics export (writeup/om_friction_export_spec.md) + the dated causal signal and
prices the daily strategy -- sell (signal>0) / buy an ATM straddle at close t, exit close t+1 --
under three execution-cost tiers. Readout: gross option-space Sharpe (does RV-HAR transfer to
RV-implied?) and net Sharpe per tier vs the 0.1/day venue-viability threshold.

Usage:
  python price_option_friction.py --opprcd data/om_friction/spy_opprcd.csv.gz \
      --secprd data/om_friction/spy_secprd.csv.gz \
      --signal eval_sim/data/cumrv_signal_dated.parquet
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

DTE_LO, DTE_HI = 1, 5  # expiry window spanning the predicted close/AH vol horizon
COMMISSION = 0.65  # $/contract/side (confirm venue rate)
HEDGE_BP = 1.0  # underlying hedge cost, bp each way
TIERS = {
    "optimistic": 0.0,
    "base": 0.25,
    "conservative": 0.5,
}  # fraction of quoted spread crossed per side
TARGET_DAILY_STD = 400.0  # $ scale matching the eval_sim honest log


def load_chain(opprcd: str) -> pd.DataFrame:
    o = pd.read_parquet(opprcd) if opprcd.endswith(".parquet") else pd.read_csv(opprcd)
    o.columns = [c.lower() for c in o.columns]
    o["date"] = pd.to_datetime(o["date"])
    o["exdate"] = pd.to_datetime(o["exdate"])
    o["strike"] = o["strike_price"] / 1000.0
    o["cp_flag"] = o["cp_flag"].str.upper().str[0]
    o["mid"] = (o["best_bid"] + o["best_offer"]) / 2.0
    o["spread"] = o["best_offer"] - o["best_bid"]
    o["dte"] = (o["exdate"] - o["date"]).dt.days
    # permissive: zero-bid rows stay (they are the worthless-exit quotes for short winners);
    # entry legs get the strict bid>0 requirement in pick_straddle
    keep = o["dte"].between(0, DTE_HI + 5) & (o["best_offer"] > 0)
    return o.loc[keep].reset_index(drop=True)


def pick_straddle(day: pd.DataFrame, spot: float) -> pd.DataFrame | None:
    """ATM straddle in the nearest expiry with DTE in [DTE_LO, DTE_HI]: the C+P pair at the
    strike nearest spot where BOTH legs quote. Returns the 2-row frame or None."""
    cand = day[day["dte"].between(DTE_LO, DTE_HI) & (day["best_bid"] > 0)]
    if cand.empty:
        return None
    exd = cand.loc[cand["dte"].idxmin(), "exdate"]
    chain = cand[cand["exdate"] == exd]
    both = chain.groupby("strike")["cp_flag"].nunique()
    strikes = both[both == 2].index.to_numpy()
    if strikes.size == 0:
        return None
    k = strikes[np.argmin(np.abs(strikes - spot))]
    return chain[chain["strike"] == k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opprcd", required=True)
    ap.add_argument("--secprd", required=True)
    ap.add_argument(
        "--signal", required=True
    )  # parquet: date, signal (causal, known at close)
    ap.add_argument("--out", default="eval_sim/data/cumrv_option_net_pnl.npy")
    ap.add_argument("--secid", type=int, default=109820)  # 109820=SPY, 189691=XSP
    a = ap.parse_args()

    chain = load_chain(a.opprcd)
    spot = pd.read_csv(a.secprd)
    spot.columns = [c.lower() for c in spot.columns]
    spot = spot[spot["secid"] == a.secid]
    spot["date"] = pd.to_datetime(spot["date"])
    spot = spot.set_index("date")["close"]
    sig = pd.read_parquet(a.signal).set_index("date")["signal_1600"]

    # index the chain by (date, exdate, strike, cp) for the t+1 exit lookup
    key = ["date", "exdate", "strike", "cp_flag"]
    chain_idx = chain.drop_duplicates(subset=key).set_index(key).sort_index()
    by_day = dict(tuple(chain.groupby("date")))
    dates = sorted(set(by_day) & set(sig.dropna().index) & set(spot.index))

    rows = []
    n_nosig = n_nostrad = n_noexit = 0
    for i, d in enumerate(dates[:-1]):
        d1 = dates[i + 1]  # exit next trading day (holds over the close/AH window)
        s = float(sig.loc[d])
        if not np.isfinite(s) or s == 0.0:
            n_nosig += 1
            continue
        strad = pick_straddle(by_day[d], float(spot.loc[d]))
        if strad is None:
            n_nostrad += 1
            continue
        # exit quotes: same contracts at d1
        try:
            exit_legs = pd.concat(
                [
                    chain_idx.loc[(d1, r.exdate, r.strike, r.cp_flag)].to_frame().T
                    for r in strad.itertuples()
                ]
            )
        except KeyError:
            n_noexit += 1
            continue  # contract expired/not quoted at d1 (DTE=1 entries exercise-settle; TODO: settle at intrinsic)
        side = -np.sign(s)  # short straddle when cumrv above expectation
        w = min(
            abs(s), 3.0
        )  # |signal| sizing, capped (matches log construction spirit)
        entry_mid = float(strad["mid"].sum())
        exit_mid = float(exit_legs["mid"].sum())
        sp_in, sp_out = float(strad["spread"].sum()), float(exit_legs["spread"].sum())
        vega = float(strad["vega"].abs().sum()) if "vega" in strad else np.nan
        net_delta = float(
            (strad["delta"] * 1.0).sum()
        )  # residual entry delta, hedged in shares
        hedge_cost = abs(net_delta) * float(spot.loc[d]) * HEDGE_BP / 1e4 * 2  # in+out
        for tier, frac in TIERS.items():
            # crossing always hurts: enter long at mid+frac*sp / short at mid-frac*sp; reverse on exit
            entry = entry_mid + side * frac * sp_in
            exitp = exit_mid - side * frac * sp_out
            pnl = (
                side * w * (exitp - entry) * 100.0
                - w * (4 * COMMISSION)
                - w * hedge_cost * 100.0 / float(spot.loc[d])
            )
            rows.append(
                {
                    "date": d,
                    "tier": tier,
                    "pnl": pnl,
                    "w": w,
                    "vega": vega,
                    "side": side,
                }
            )

    df = pd.DataFrame(rows)
    df.to_parquet(a.out.replace(".npy", "_perday.parquet"), index=False)
    print(
        f"{df['date'].nunique():,} traded days  "
        f"(skipped: no-signal {n_nosig}, no-straddle {n_nostrad}, no-exit-quote {n_noexit})"
    )
    net = None
    for tier in TIERS:
        p = df[df["tier"] == tier].set_index("date")["pnl"]
        sh = p.mean() / p.std()
        print(
            f"{tier:>12}: mean=${p.mean():+.2f}/day  std=${p.std():.2f}  Sharpe/day={sh:+.3f}  ann={sh * np.sqrt(252):+.2f}"
        )
        if tier == "base":
            net = p
    if net is not None and net.mean() / net.std() >= 0.10:
        scaled = (net / net.std() * TARGET_DAILY_STD).to_numpy()
        np.save(a.out, scaled)
        print(
            f"VIABLE -> saved {a.out} ({scaled.size} days) — rerun imperial_pass_calc.py on it"
        )
    else:
        print(
            "BELOW 0.1/day at base costs — options eval-prop venue class is moot at this friction"
        )


if __name__ == "__main__":
    main()
