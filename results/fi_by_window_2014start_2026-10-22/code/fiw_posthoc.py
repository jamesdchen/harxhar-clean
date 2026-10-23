"""POST HOC (not in RULE_fiw.txt; written after seeing fiw_tables.txt): the P1 / P2 measures on the 42 exogenous variables
live at every W.

Why: LATE_START_WARMUP (src/backtest/executor.py) keeps a late-start series' value averages at the neutral 0 until the
scaler's window holds W live rows, so stocktwits_sentcount and the three voldemand_* series (starts 2009-10 / 2011) are
masked at more origins as W grows and at every origin at W = 3,500: the live exogenous universe shrinks from 46 to 42
variables with W, which by itself lowers V and can move the shares.  Here every measure is recomputed on the 42 variables
that are not among those four (their columns dropped from the numerator and the denominator), with the same paired block
bootstrap (fiw_reduce.py) for the slope on W.

Usage: python fiw_posthoc.py   (reads ../fiw_gamma.npz, writes ../fiw_posthoc_42.csv and prints a table)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import fiw_reduce as R

HERE = Path(__file__).resolve().parent
DROP = {
    "stocktwits_sentcount",
    "voldemand_spx_open_only",
    "voldemand_all_open_and_close",
    "voldemand_all_open_only",
}
MEAS = [
    "V_mean",
    "V_half",
    "C_mean",
    "C_half",
    "Neff_mean",
    "n50_mean",
    "n80_mean",
    "Neff_pooled",
    "n50_pooled",
    "n80_pooled",
]


def main() -> None:
    z = np.load(HERE.parent / "fiw_gamma.npz")
    names = list(z["columns"])
    ser = [R.series_of(c) for c in names]
    E = np.array([s not in ("har_ma", "calendar") and s not in DROP for s in ser])
    xv = [
        v
        for v in dict.fromkeys(ser)
        if v not in ("har_ma", "calendar") and v not in DROP
    ]
    assert len(xv) == 42
    Gx = np.zeros((int(E.sum()), len(xv)))
    Gx[
        np.arange(int(E.sum())), [xv.index(s) for s, e in zip(ser, E, strict=True) if e]
    ] = 1.0
    K = len(z["origins"])
    rng = np.random.default_rng(R.SEED)
    boots = [R.block_idx(K, rng) for _ in range(R.NBOOT)]
    rows = []
    for m in R.MODELS:
        for s in ["walk"] + [R.src_tag(m, c) for c in R.COMMON[m]]:
            RR = {wd: R.per_refit(z[f"{s}_w{wd}_{m}_gamma"], E, Gx) for wd in R.WS}
            pt = {wd: R.stats_of(RR[wd]) for wd in R.WS}
            bs = [[R.stats_of(RR[wd], ix) for wd in R.WS] for ix in boots]
            for k in MEAS:
                v = np.array([pt[wd][k] for wd in R.WS])
                sl = np.array(
                    [R.slope(np.array([b[j][k] for j in range(len(R.WS))])) for b in bs]
                )
                lo, hi = np.percentile(sl, [2.5, 97.5])
                rows.append(
                    {
                        "model": m,
                        "source": s,
                        "measure": k,
                        **{f"W{wd}": v[j] for j, wd in enumerate(R.WS)},
                        "slope_per_1000d": R.slope(v),
                        "slope_lo": lo,
                        "slope_hi": hi,
                        "verdict": "rises"
                        if lo > 0
                        else ("falls" if hi < 0 else "flat"),
                    }
                )
    t = pd.DataFrame(rows)
    t.to_csv(HERE.parent / "fiw_posthoc_42.csv", index=False)
    with pd.option_context(
        "display.width",
        250,
        "display.max_rows",
        500,
        "display.float_format",
        "{:.2f}".format,
    ):
        print(t)


if __name__ == "__main__":
    main()
