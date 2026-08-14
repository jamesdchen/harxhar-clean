"""Entry-hour sweep: is there a time of day where information beats premium?

At 10:00 the strip's premium dominates (mean pay -0.79 per unit, hit 99%),
so no forecast changes the optimal position. But the premium FRACTION may
thin as the session burns: from each 30-min entry bar t, entry cost is the
strip mark C_t, the model's fair value is blk2's remaining-RV forecast,
and the trade holds to expiry (terminal strip value = last C_next of the
day, i.e. settlement marks — plus an RV-settled paper variant). Wherever
1 - blk2_rem/C stops being large and one-signed, information starts
mattering: report per entry hour the premium fraction, the count of
model-cheap entries (blk2_rem > C: the model screams BUY), the constant
short book, the zero-mean overlay, and the cheap-long book.

Causality: overlay z is the causal z-score (per entry hour, expanding
min 63 days) of log(blk2_rem/C). Mids only.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "spxw_pnl")
ANN = float(np.sqrt(252.0))
BURN = 63
DAILY_0DTE = pd.Timestamp("2022-05-16")


def _sh(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3 or float(x.std()) == 0.0:
        return float("nan")
    return float(x.mean() / x.std())


def _causal_z(x: pd.Series) -> pd.Series:
    mu = x.expanding(min_periods=BURN).mean().shift(1)
    sd = x.expanding(min_periods=BURN).std().shift(1)
    return (x - mu) / sd


def main() -> None:
    tr = pd.read_parquet(os.path.join(OUT, "everybar_mtm_trades.parquet"))
    tr["expiration"] = pd.to_datetime(tr["expiration"])
    tr = tr.sort_values(["day", "et"]).reset_index(drop=True)

    g = tr.groupby("day", sort=False)
    tr["rv_rem"] = g["rv"].transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1])
    tr["b2_rem"] = g["pb"].transform(lambda s: s.iloc[::-1].cumsum().iloc[::-1])
    tr["C_term"] = g["C_next"].transform("last")

    c = tr["C"].to_numpy(float)
    tr["pay_listed"] = (tr["C_term"] - tr["C"]) / np.maximum(c, 1e-18)
    tr["pay_rv"] = (tr["rv_rem"] - tr["C"]) / np.maximum(c, 1e-18)
    tr["prem_frac"] = 1.0 - tr["b2_rem"] / np.maximum(c, 1e-18)
    tr["hhmm"] = tr["et"].dt.strftime("%H:%M")

    rows = []
    for era, sub in (
        ("all", tr),
        ("daily_0dte", tr[tr["expiration"] >= DAILY_0DTE]),
    ):
        for hh, gd in sub.groupby("hhmm"):
            gd = gd.sort_values("day")
            z = _causal_z(
                pd.Series(
                    np.log(
                        np.maximum(gd["b2_rem"].to_numpy(float), 1e-18)
                        / np.maximum(gd["C"].to_numpy(float), 1e-18)
                    ),
                    index=gd.index,
                )
            ).to_numpy(float)
            pl = gd["pay_listed"].to_numpy(float)
            pr = gd["pay_rv"].to_numpy(float)
            cheap = gd["prem_frac"].to_numpy(float) < 0.0
            zok = np.isfinite(z) & np.isfinite(pl)
            rows.append(
                {
                    "era": era,
                    "entry": hh,
                    "n": len(gd),
                    "prem_frac_med": float(gd["prem_frac"].median()),
                    "n_cheap": int(cheap.sum()),
                    "sh_const_short": _sh(-pl) * ANN,
                    "sh_const_short_rvsettle": _sh(-pr) * ANN,
                    "sh_overlay_z": _sh((z * pl)[zok]) * ANN,
                    "cheap_long_mean": float(pl[cheap].mean())
                    if cheap.any()
                    else float("nan"),
                    "cheap_long_hit": float((pl[cheap] > 0).mean())
                    if cheap.any()
                    else float("nan"),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "entry_hour_sweep.csv"), index=False)
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
