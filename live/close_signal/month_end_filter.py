"""The month-end filter: buy the month-end strangle only when it is cheap against the day.

    score_d = RV_d / IVask_d

RV_d is the day's realized variance so far -- the sum of the panel's 30-minute
``sumret2`` over the bars ending 10:00 .. 15:30 (09:30-15:30, the ES moments
the card already builds) -- and IVask_d the Black-76 package variance implied
by the 15:30 ASK of the nearest-OTM strangle (the price actually paid).  The
day's realized variance predicts the last half hour's (rank correlation 0.81
on 70 month-ends), so a high score means the option is priced for a calmer
close than the day has been.

Rule (set 2026-09-30 by the operator): buy the month-end strangle iff its score
is at or above the 80th percentile of the previous 250 ORDINARY (non-month-end)
sessions' scores.  Regime-relative on purpose: the fixed 2020-2025 cutoffs put
every 2026 month-end in the 'cheap' third and those lost; ranked against the
trailing year the order held (higher rank, smaller loss).  Evidence is thin
(62 in-sample month-ends, 8 in 2026) -- this is an operator's rule, not a
validated edge.

Live, the rule becomes a price: with today's RV and today's pair,

    ask_max = package_price(sqrt(RV / cutoff * minutes_left / 30), S, Kc, Kp)

The cutoff needs past 15:30 option asks, which the live job does not have, so
it is computed offline from the local Databento files and committed:

    python -m live.close_signal.month_end_filter --refresh     # after each SPXW pull

writes live/close_signal/state/month_end_cutoff.json.  A missing or stale
cutoff (older than STALE_DAYS) makes the month-end card say NO TRADE, with the
reason -- the rule is 'only if cheap', so without the filter there is no buy.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from live.ibkr.calendar_guard import is_last_session_of_month
from live.ibkr.pricing import invert_total_vol, package_price

REPO = Path(__file__).resolve().parents[2]
STATE_DIR = REPO / "live" / "close_signal" / "state"
CUTOFF_PATH = STATE_DIR / "month_end_cutoff.json"
OPRA_DIR = REPO / "data" / "archive" / "spxw_opra"
ET = "America/New_York"
#: the bars (end stamps) whose variance is the day so far: 09:30-15:30
RV_STAMPS = tuple(f"{h:02d}:{m:02d}" for h in range(10, 16) for m in (0, 30))
PERCENTILE = 0.80
WINDOW = 250
MIN_SESSIONS = 150
STALE_DAYS = 45


def day_rv(rows: pd.DataFrame, session: pd.Timestamp) -> float:
    """Sum of the session's sumret2 over the 12 bars ending 10:00 .. 15:30; nan if any is missing."""
    t = pd.to_datetime(rows["endbartime"])
    m = (t.dt.normalize() == pd.Timestamp(session).normalize()) & t.dt.strftime(
        "%H:%M"
    ).isin(RV_STAMPS)
    v = pd.to_numeric(rows.loc[m, "sumret2"], errors="coerce")
    if len(v) != len(RV_STAMPS) or not np.isfinite(v).all():
        return float("nan")
    return float(v.sum())


def ask_max(
    rv: float,
    cutoff: float,
    spot: float,
    kc: float,
    kp: float,
    minutes_left: float = 30.0,
) -> float:
    """The most the pair may cost for its score to reach the cutoff (nan when undefined)."""
    if not (np.isfinite(rv) and rv > 0 and np.isfinite(cutoff) and cutoff > 0):
        return float("nan")
    return float(
        package_price(math.sqrt(rv / cutoff * minutes_left / 30.0), spot, kc, kp)
    )


def load_cutoff(
    path: Path = CUTOFF_PATH, today: date | None = None
) -> tuple[float, str]:
    """(score cutoff, '') or (nan, reason)."""
    if not path.exists():
        return float("nan"), "no month-end cutoff file"
    try:
        c = json.loads(path.read_text(encoding="utf-8"))
        v = float(c["score_cutoff"])
        asof = pd.Timestamp(c["asof"]).date()
    except Exception as e:  # noqa: BLE001
        return float("nan"), f"unreadable month-end cutoff ({type(e).__name__})"
    if today is not None and (today - asof).days > STALE_DAYS:
        return float(
            "nan"
        ), f"month-end cutoff stale (as of {asof}; refresh after an SPXW pull)"
    return v, ""


# ---------------------------------------------------------------- offline --
def _pair_ask_1530(path: Path, day: pd.Timestamp) -> tuple[float, float, float, float]:
    """(spot, Kc, Kp, ask) of the nearest-OTM SPXW pair at 15:30 from a cbbo-1m file."""
    q = pd.read_parquet(path, columns=["ts_recv", "symbol", "bid_px_00", "ask_px_00"])
    t = pd.to_datetime(q["ts_recv"], utc=True).dt.tz_convert(ET).dt.tz_localize(None)
    stamp = pd.Timestamp(f"{day.date()} 15:30:00")
    b = q[t == stamp]
    if b.empty:
        b = q[t <= stamp]
        b = b[
            pd.to_datetime(b["ts_recv"], utc=True)
            == pd.to_datetime(b["ts_recv"], utc=True).max()
        ]
    sym = b["symbol"].astype(str)
    b = b.assign(k=sym.str.slice(13).astype(float) / 1000.0, cp=sym.str.slice(12, 13))
    b = b[(b["ask_px_00"] > 0) & (b["bid_px_00"] >= 0)]
    mid = ((b["bid_px_00"] + b["ask_px_00"]) / 2).to_numpy()
    book = pd.DataFrame(
        {
            "k": b["k"].to_numpy(),
            "cp": b["cp"].to_numpy(),
            "mid": mid,
            "ask": b["ask_px_00"].to_numpy(),
        }
    ).drop_duplicates(["k", "cp"], keep="last")
    piv = book.pivot(index="k", columns="cp", values="mid").dropna()
    if piv.empty or not {"C", "P"} <= set(piv.columns):
        return (float("nan"),) * 4
    diff = (piv["C"] - piv["P"]).abs().nsmallest(3).index
    spot = float(
        np.median(diff.to_numpy() + (piv["C"] - piv["P"]).loc[diff].to_numpy())
    )
    ks = np.sort(book["k"].unique())
    above, below = ks[ks >= spot], ks[ks <= spot]
    if not len(above) or not len(below):
        return (float("nan"),) * 4
    kc, kp = float(above.min()), float(below.max())
    a = book.set_index(["k", "cp"])["ask"]
    try:
        return spot, kc, kp, float(a.loc[(kc, "C")] + a.loc[(kp, "P")])
    except KeyError:
        return (float("nan"),) * 4


def daily_scores(panel: pd.DataFrame, opra_dir: Path = OPRA_DIR) -> pd.DataFrame:
    """score = day RV / 15:30 ask-implied variance for every session with an SPXW file."""
    rows = []
    for f in sorted(glob.glob(str(opra_dir / "cbbo1m_*.parquet"))):
        day = pd.Timestamp(Path(f).stem.split("_")[1])
        rv = day_rv(panel, day)
        if not np.isfinite(rv):
            continue
        spot, kc, kp, ask = _pair_ask_1530(Path(f), day)
        if not np.isfinite(ask) or ask <= 0:
            continue
        tv = invert_total_vol(spot, kc, kp, ask)
        if not np.isfinite(tv) or tv <= 0:
            continue
        rows.append(
            {
                "day": day,
                "rv_day": rv,
                "iv_ask": tv * tv,
                "score": rv / (tv * tv),
                "month_end": is_last_session_of_month(day.date()),
            }
        )
    return pd.DataFrame(rows).set_index("day").sort_index()


def cutoff_from(
    scores: pd.DataFrame, asof: pd.Timestamp | None = None
) -> dict[str, object]:
    """The 80th percentile of the last WINDOW ordinary sessions' scores (up to ``asof``)."""
    s = scores if asof is None else scores[scores.index <= asof]
    ordinary = s.loc[~s["month_end"].astype(bool), "score"].dropna().tail(WINDOW)
    if len(ordinary) < MIN_SESSIONS:
        raise ValueError(
            f"only {len(ordinary)} ordinary sessions with a score (< {MIN_SESSIONS})"
        )
    return {
        "score_cutoff": float(np.quantile(ordinary, PERCENTILE)),
        "percentile": PERCENTILE,
        "window": WINDOW,
        "n": int(len(ordinary)),
        "first": str(ordinary.index.min().date()),
        "asof": str(ordinary.index.max().date()),
        "definition": "score = RV(09:30-15:30, panel sumret2) / Black-76 variance of the 15:30 ask, "
        "nearest-OTM SPXW pair; buy the month-end iff score >= cutoff",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--refresh", action="store_true", help="recompute and write the cutoff"
    )
    ap.add_argument("--state-dir", default=str(STATE_DIR))
    a = ap.parse_args(argv)
    if not a.refresh:
        print(
            json.dumps(
                json.loads(CUTOFF_PATH.read_text()) if CUTOFF_PATH.exists() else {},
                indent=1,
            )
        )
        return 0
    panel = pd.read_parquet(
        Path(a.state_dir) / "panel_free.parquet", columns=["endbartime", "sumret2"]
    )
    scores = daily_scores(panel)
    out = cutoff_from(scores)
    (Path(a.state_dir) / "month_end_cutoff.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8"
    )
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
