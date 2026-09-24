"""Pull the ES history through Databento's STREAMING API (no batch queue) into one CSV.

    pip install databento                 # once, in the 285J env
    python live/close_signal/pull_databento_es.py [--start 2024-04-01] [--end 2026-09-25]

Reads the API key from DATABENTO_API_KEY, else asks for it with a hidden
prompt -- run this in your own terminal, never through a shared session.
Prints the cost estimate first (measured 2026-09-23: USD 3.21 for ES.v.0
ohlcv-1m over 2024-04-01 .. 2026-09-23 -- the continuous request is billed on
the underlying records, not far below the whole product's 5.15) and aborts
above --max-usd.  The raw DBN is written beside the CSV before any conversion.  The continuous volume-ranked front month ``ES.v.0``
(``stype_in="continuous"``) is what the ingest wants; ``end`` is exclusive and
is clipped to the dataset's available end.  Output columns are the portal's:
ts_event (ISO UTC), rtype, publisher_id, instrument_id, open .. volume, symbol.
Then:

    python -m live.close_signal.ingest es-databento data/archive/es_v0_ohlcv1m_databento.csv
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "data" / "archive" / "es_v0_ohlcv1m_databento.csv"
DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1m"
SYMBOL = "ES.v.0"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", default="2024-04-01")
    ap.add_argument(
        "--end", default="2026-09-25", help="exclusive; clipped to availability"
    )
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--max-usd", type=float, default=5.0)
    a = ap.parse_args(argv)
    try:
        import databento as db
    except ImportError:
        print("pip install databento first", file=sys.stderr)
        return 2
    key = os.environ.get("DATABENTO_API_KEY") or getpass.getpass(
        "Databento API key (hidden): "
    )
    client = db.Historical(key)
    avail = client.metadata.get_dataset_range(DATASET)
    avail_end = str(avail["end"])[:19]
    end = min(a.end, avail_end)
    kw = dict(
        dataset=DATASET,
        schema=SCHEMA,
        symbols=[SYMBOL],
        stype_in="continuous",
        start=a.start,
        end=end,
    )
    cost = float(client.metadata.get_cost(**kw))
    print(f"range {a.start} -> {end} (available end {avail_end}); cost USD {cost:.4f}")
    if cost > a.max_usd:
        print(f"above --max-usd {a.max_usd}; not pulled", file=sys.stderr)
        return 3
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    store = client.timeseries.get_range(**kw)
    # the paid bytes land on disk FIRST (the ingest reads .dbn.zst too), so a
    # conversion failure -- the 2026-09-23 pull died on a renamed kwarg with the
    # data only in memory -- never costs a second pull
    raw_path = out.with_name(out.stem + ".dbn.zst")
    store.to_file(raw_path)
    print(f"raw DBN saved to {raw_path}")
    try:
        df = store.to_df(price_type="float", pretty_ts=True, map_symbols=True)
    except TypeError:  # older databento clients spell it pretty_px
        df = store.to_df(pretty_px=True, pretty_ts=True, map_symbols=True)  # type: ignore[call-overload]
    df = df.reset_index()
    df.to_csv(out, index=False)
    print(
        f"{len(df)} rows, {df['ts_event'].min()} .. {df['ts_event'].max()}, "
        f"instruments {df['instrument_id'].nunique()}, wrote {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
