"""Validate the availability indicators against each feed's publication calendar.

The composed prep fix records availability *before* the forward fill, which
moved voldemand's indicator from 0.6983 to 0.2224. That drop is only correct if
voldemand genuinely prints on roughly a fifth of the panel's bars. Nothing in
the pipeline checked it -- the number was taken on trust, which is the same
class of error as the rolling-IQR magnitude proxy that produced two
retractions. This script checks it against the raw feed.

The panel grid is 30-minute bars covering Sunday 18:30 through Friday 20:00 --
about 24 hours a day, five days a week. Most exogenous channels are US cash- or
extended-session equity quantities, so a channel that prints only inside its own
session is *correctly* unavailable for most of the grid. The question is whether
each channel's measured availability matches the session it actually publishes
in.

Reports per channel:

    avail       fraction of panel rows carrying a raw (pre-fill) value
    session     the time-of-day window holding the slots it usually prints in
    implied     that window's width as a fraction of the 48-slot day
    span        first and last print
    dead_bars   panel rows after the final print -- feed termination, where a
                forward fill has nothing left to bridge to

Written to writeup/stats/feed_calendars.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.loading import get_bucket, load_raw_data  # noqa: E402

# Bars per day on the 30-minute grid, and the reference sessions the equity
# feeds publish in. Kept explicit so a mismatch is legible rather than inferred.
SLOTS_PER_DAY = 48
REFERENCE_SESSIONS = {
    "US cash 09:30-16:00": 13 / SLOTS_PER_DAY,
    "US extended 04:30-20:00": 32 / SLOTS_PER_DAY,
    "24h": 1.0,
}
# A slot counts as part of a channel's session if it prints there on more than
# half the days that slot appears. Well separated in practice: session slots sit
# near 1.0 and non-session slots near 0.0.
SESSION_HIT_RATE = 0.5
# Feed termination is only interesting when it is long enough to span refits.
DEAD_BAR_FLOOR = 100


def slot_of(t: pd.Series) -> np.ndarray:
    return (t.dt.hour * 2 + (t.dt.minute >= 30)).to_numpy()


def main() -> None:
    bucket = os.environ.get("BUCKET", "all_features")
    cols = get_bucket(bucket)
    df = load_raw_data(os.path.join(ROOT, "data"), allow_missing=True)
    n = len(df)
    print(f"bucket {bucket}: {len(cols)} exogenous channels")
    print(f"raw panel {df.shape}, {df['t'].min()} .. {df['t'].max()}")

    slot = slot_of(df["t"])
    slot_rows = np.bincount(slot, minlength=SLOTS_PER_DAY).astype(float)
    t_all = df["t"].to_numpy()
    panel_end = df["t"].max()

    recs = []
    for c in cols:
        if c not in df.columns:
            recs.append({"channel": c, "present": False})
            continue
        obs = df[c].notna().to_numpy()
        hit = np.bincount(slot[obs], minlength=SLOTS_PER_DAY).astype(float)
        rate = np.where(slot_rows > 0, hit / np.maximum(slot_rows, 1.0), 0.0)
        active = np.flatnonzero(rate > SESSION_HIT_RATE)
        if active.size:
            lo, hi = int(active.min()), int(active.max())
            session = (f"{lo // 2:02d}:{30 * (lo % 2):02d}-"
                       f"{hi // 2:02d}:{30 * (hi % 2):02d}")
            implied = active.size / SLOTS_PER_DAY
        else:
            session, implied = "sparse", float("nan")
        idx = np.flatnonzero(obs)
        first, last = t_all[idx[0]], t_all[idx[-1]]
        dead = int((df["t"] > pd.Timestamp(last)).sum())
        gaps = np.diff(idx) if idx.size > 1 else np.array([np.nan])
        recs.append({
            "channel": c, "present": True,
            "avail": float(obs.mean()),
            "session": session, "implied": float(implied),
            "first": str(pd.Timestamp(first)), "last": str(pd.Timestamp(last)),
            "dead_bars": dead, "dead_frac": dead / n,
            "gap_p50": float(np.nanpercentile(gaps, 50)),
            "gap_p95": float(np.nanpercentile(gaps, 95)),
        })

    live = [r for r in recs if r.get("present")]
    tab = pd.DataFrame(live).sort_values("avail")
    pd.set_option("display.width", 200)
    print()
    print(tab[["channel", "avail", "session", "implied", "first", "last",
               "dead_bars", "gap_p50", "gap_p95"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nmean raw availability across {len(live)} channels: "
          f"{tab['avail'].mean():.4f}")
    for k, v in REFERENCE_SESSIONS.items():
        print(f"  reference {k:26s} = {v:.4f} of the grid")

    # ---- the claim under test: voldemand's indicator against its calendar
    vd = [r for r in live if "voldemand" in r["channel"]]
    print("\nthe 0.2224 availability indicator against the actual calendar:")
    for r in vd:
        print(f"  {r['channel']:32s} raw coverage {r['avail']:.4f}  "
              f"session {r['session']}  last print {r['last']}")
    vd_raw = float(np.mean([r["avail"] for r in vd])) if vd else float("nan")
    print(f"  indicator after the fix 0.2224 vs raw coverage {vd_raw:.4f}  "
          f"-> gap {0.2224 - vd_raw:+.4f}")
    print("  (the fix should sit slightly ABOVE raw coverage: the bounded "
          "26-bar fill legitimately bridges short gaps)")

    # ---- feed termination
    dead = sorted((r for r in live if r["dead_bars"] > DEAD_BAR_FLOOR),
                  key=lambda r: -r["dead_bars"])
    print(f"\npanel ends {panel_end}. Feeds that stop before it does:")
    for r in dead:
        print(f"  {r['channel']:32s} last print {r['last']}  "
              f"{r['dead_bars']:6d} dead bars ({r['dead_frac']:.2%})")

    # ---- October 2023, where the blow-up was attributed
    oct_mask = ((df["t"] >= "2023-10-01") & (df["t"] < "2023-11-01")).to_numpy()
    oct_rate = sorted(((c, float(df.loc[oct_mask, c].notna().mean()))
                       for c in cols if c in df.columns), key=lambda z: z[1])
    print(f"\nOctober 2023 ({int(oct_mask.sum())} bars), lowest print rates:")
    for c, v in oct_rate[:6]:
        print(f"  {c:32s} {v:.4f}")

    out = {
        "bucket": bucket,
        "panel_rows": n,
        "panel_start": str(df["t"].min()),
        "panel_end": str(panel_end),
        "channels": recs,
        "mean_avail": float(tab["avail"].mean()),
        "reference_sessions": REFERENCE_SESSIONS,
        "voldemand_raw_coverage": vd_raw,
        "voldemand_indicator_after_fix": 0.2224,
        "voldemand_indicator_before_fix": 0.6983,
        "dead_feeds": [{k: r[k] for k in
                        ("channel", "last", "dead_bars", "dead_frac")}
                       for r in dead],
        "october_2023_print_rate": dict(oct_rate),
        "verdict": {
            "indicator_matches_calendar": bool(
                vd and abs(0.2224 - vd_raw) < 0.01),
            "voldemand_feed_terminates_before_panel": bool(
                vd and any(r["dead_bars"] > DEAD_BAR_FLOOR for r in vd)),
        },
    }
    dst = os.path.join(ROOT, "writeup", "stats", "feed_calendars.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
