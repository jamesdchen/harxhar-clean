"""Walk-forward, end-to-end -- the way the rest of this repo evaluates everything. My earlier breakout
numbers (avgR, P(pass), the volume/trend filter wins) were IN-SAMPLE: I searched targets x filters on the
full 60 days and reported the winners. That is the exact overfit the target walk-forward exposed. So do the
WHOLE config selection walk-forward:

  config space = target {1,1.5,2,2.5,3,None} x volume{0,1} x trend{0,1}  (24 configs).
  Each day (after a warmup), pick the config with best trailing-Sharpe on PRIOR days only, apply it to that
  day, accumulate OOS. Report:
    - OOS adaptive (the honest, selection-included walk-forward)
    - in-sample-best config on the SAME OOS window (the overfit CEILING)
    - a PRE-COMMITTED config (volume filter + 2R, chosen a priori for mechanism, no search) on the OOS window
The gap adaptive<<ceiling = the selection overfit; the pre-committed number is the honest "does a sensible
fixed rule work OOS". 60d QQQ-5m = far too little to select 24 configs on -- the expected lesson is that it
overfits, which is the point. Usage: python breakout_wf.py --data <qqq_5m.parquet>
"""

from __future__ import annotations

import argparse

import pandas as pd

from breakout_dig import daily_R, detect, load_ohlcv
from breakout_maker import p_pass

TARGETS = [1.0, 1.5, 2.0, 2.5, 3.0, None]
WARM = 30


def sharpe(s: pd.Series) -> float:
    v = s.to_numpy()
    return float(v.mean() / v.std()) if v.std() > 0 else -9.0


def summarize(name, oR):
    r = oR.to_numpy()
    if len(r) < 5:
        print(f"[{name:30s}] n_days={len(r)} (too few)")
        return
    sh = float(r.mean() / r.std()) if r.std() > 0 else float("nan")
    pp = f"{max(p_pass(oR).values()):.2f}" if oR.mean() > 0 else "N/A(net-neg)"
    print(
        f"[{name:30s}] n_days={len(r):3d} avgDayR={r.mean():+.3f} Sharpe={sh:+.3f} P(pass)={pp}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    df = load_ohlcv(a.data)
    configs = [
        (t, v, tr) for t in TARGETS for v in (False, True) for tr in (False, True)
    ]
    per = {}
    for t, v, tr in configs:
        ents, ohlc = detect(df, vol_filt=v, trend_filt=tr)
        per[(t, v, tr)] = daily_R(ents, ohlc, t)[0]
    days = sorted(set().union(*[set(s.index) for s in per.values()]))

    # (1) honest walk-forward: select config on trailing days, apply OOS
    oos = {}
    for k in range(WARM, len(days)):
        tr = days[:k]
        best = max(configs, key=lambda cfg: sharpe(per[cfg].reindex(tr).fillna(0.0)))
        oos[days[k]] = per[best].get(days[k], 0.0)
    oR = pd.Series(oos)

    # (2) overfit ceiling: best single config chosen ON the OOS window
    oosdays = days[WARM:]
    ceil_cfg = max(
        configs, key=lambda cfg: sharpe(per[cfg].reindex(oosdays).fillna(0.0))
    )
    ceilR = per[ceil_cfg].reindex(oosdays).fillna(0.0)

    # (3) pre-committed a-priori config (volume + 2R), no search, on the OOS window
    preR = per[(2.0, True, False)].reindex(oosdays).fillna(0.0)

    print("--- WALK-FORWARD (OOS-only) ---")
    summarize("walk-fwd adaptive (honest)", oR)
    summarize(f"in-sample ceiling {ceil_cfg}", ceilR)
    summarize("pre-committed volume+2R", preR)


if __name__ == "__main__":
    main()
