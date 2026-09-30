"""Closing-auction imbalances on month-ends from Databento (streaming API), estimate first.

    python live/close_signal/pull_databento_imbalance.py --estimate     # availability + cost, nothing bought
    python live/close_signal/pull_databento_imbalance.py --pull --max-usd 40

Why: the month-end close is moved by known flows (index funds, rebalancing),
and the exchanges publish the closing-auction order imbalance per stock from
15:50 ET.  Summed over the large caps, the dollar imbalance at 15:50 is the
most direct read of which way and how hard those flows lean -- the one
predictor of the month-end close not yet tested (the model's forecast, the
day's realized move, the month-to-date move, momentum and quarter-end all
failed to predict it).

What it fetches, per month-end session D and dataset (NYSE-listed stocks on
XNYS.PILLAR, Nasdaq-listed on XNAS.ITCH): the `imbalance` schema for
ALL_SYMBOLS from 15:49 to 16:00 ET, closing-auction records only (the
opening ones are dropped before saving); saved as
data/archive/imbalance/<DATASET>_<D>.parquet.  Days already on disk are
skipped (resumable); the run stops at --max-usd.  The key is read from
DATABENTO_API_KEY or asked for with a hidden prompt: run this in your own
terminal.  --estimate prices a few sample days per dataset, reports each
dataset's available range, and buys nothing.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from live.ibkr.calendar_guard import is_last_session_of_month  # noqa: E402

OUT_DIR = REPO / "data" / "archive" / "imbalance"
DATASETS = ("XNYS.PILLAR", "XNAS.ITCH")
ET = "America/New_York"
WINDOW = ("15:49", "16:00")
FIRST = pd.Timestamp("2018-05-01")


def month_ends(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    days = pd.bdate_range(start, end)
    return [d for d in days if is_last_session_of_month(d.date())]


def window_utc(day: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    lo = pd.Timestamp(f"{day.date()} {WINDOW[0]}", tz=ET).tz_convert("UTC")
    hi = pd.Timestamp(f"{day.date()} {WINDOW[1]}", tz=ET).tz_convert("UTC")
    return lo, hi


def closing_only(df: pd.DataFrame) -> pd.DataFrame:
    """Closing-auction records ('C'); the field is text or its byte code depending on the client."""
    if "auction_type" not in df.columns:
        return df
    a = df["auction_type"]
    codes = a.astype(str).str.strip()
    return df[(codes == "C") | (codes == str(ord("C")))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--estimate", action="store_true")
    g.add_argument("--pull", action="store_true")
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--start", default=str(FIRST.date()))
    ap.add_argument(
        "--end", default=str((pd.Timestamp.today() - pd.Timedelta(days=1)).date())
    )
    ap.add_argument(
        "--sample", type=int, default=3, help="days priced per dataset in --estimate"
    )
    ap.add_argument("--max-usd", type=float, default=40.0)
    a = ap.parse_args(argv)
    try:
        import databento as db
    except ImportError:
        print("pip install databento first", file=sys.stderr)
        return 2
    days = month_ends(pd.Timestamp(a.start), pd.Timestamp(a.end))
    print(f"{len(days)} month-end sessions {days[0].date()} .. {days[-1].date()}")
    key = os.environ.get("DATABENTO_API_KEY") or getpass.getpass(
        "Databento API key (hidden): "
    )
    client = db.Historical(key)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    failed: list[str] = []
    for ds in a.datasets.split(","):
        try:
            rng = client.metadata.get_dataset_range(dataset=ds)
            ds_lo = (
                pd.Timestamp(rng["start"]).tz_convert("UTC")
                if pd.Timestamp(rng["start"]).tzinfo
                else pd.Timestamp(rng["start"], tz="UTC")
            )
            ds_hi = (
                pd.Timestamp(rng["end"]).tz_convert("UTC")
                if pd.Timestamp(rng["end"]).tzinfo
                else pd.Timestamp(rng["end"], tz="UTC")
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"{ds}: range unavailable ({type(e).__name__}: {str(e)[:120]}); skipped"
            )
            continue
        mine = [
            d for d in days if window_utc(d)[0] >= ds_lo and window_utc(d)[1] <= ds_hi
        ]
        print(
            f"\n{ds}: available {ds_lo.date()} .. {ds_hi}; month-ends in range {len(mine)}"
        )
        pick = (
            mine
            if a.pull
            else [
                mine[i]
                for i in np.linspace(0, len(mine) - 1, min(a.sample, len(mine)))
                .round()
                .astype(int)
            ]
        )
        costs = []
        for d in pick:
            f = OUT_DIR / f"{ds}_{d.date()}.parquet"
            if a.pull and f.exists():
                continue
            lo, hi = window_utc(d)
            kw = dict(
                dataset=ds, schema="imbalance", symbols="ALL_SYMBOLS", start=lo, end=hi
            )
            try:
                c = float(client.metadata.get_cost(**kw))
            except Exception as e:  # noqa: BLE001
                print(
                    f"  {d.date()}: cost refused ({type(e).__name__}: {str(e)[:120]}); skipped"
                )
                failed.append(f"{ds} {d.date()}")
                continue
            costs.append(c)
            if a.estimate:
                print(f"  {d.date()}: USD {c:.4f}")
                continue
            if spent + c > a.max_usd:
                print(
                    f"stopping: spent {spent:.2f} + next {c:.2f} would pass --max-usd {a.max_usd}"
                )
                break
            try:
                df = (
                    client.timeseries.get_range(**kw)
                    .to_df(pretty_ts=True, map_symbols=True)
                    .reset_index()
                )
            except Exception as e:  # noqa: BLE001
                print(
                    f"  {d.date()}: refused ({type(e).__name__}: {str(e)[:120]}); skipped"
                )
                failed.append(f"{ds} {d.date()}")
                continue
            df = closing_only(df)
            df.to_parquet(f, index=False)
            spent += c
            print(
                f"  {d.date()}: {len(df)} closing records, USD {c:.4f} (spent {spent:.2f})"
            )
        if a.estimate and costs:
            per = float(np.mean(costs))
            print(
                f"  ESTIMATE {ds}: mean USD {per:.4f} per month-end -> {len(mine)} month-ends ~ USD {per * len(mine):.2f}"
            )
    if a.pull:
        print(f"\nPULL: spent USD {spent:.2f}; files in {OUT_DIR}")
    if failed:
        print(f"skipped ({len(failed)}): {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
